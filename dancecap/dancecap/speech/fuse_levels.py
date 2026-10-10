"""Per-phone speech level and noise floor on the master clock: the "quality at that moment" of the fusion weights (local, numpy, seconds).

Every phone hears the same room, but a phone near the speaker hears the words cleanly and a phone across the room hears them through noise
and reverberation. Two cheap, model-free proxies per phone and moment:

  snr  speech level of the phone at the moment minus the phone's own noise floor (10th percentile of the frame levels in a +-10 s window), dB
  rel  the phone's level minus the loudest phone's level at the same master moment, dB (<= 0; 0 = the phone nearest to the voice; the
       speaker-distance proxy)

Frame level = RMS of 25 ms frames every 10 ms of the 16 kHz wav (`speech/audio/NN-*.16k.wav`), moved to the master clock on a 50 ms grid
(max over the cell). Outside a phone's recording the value is NaN.
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from . import SOURCES, SPEECH

GRID = 0.05
FRAME = 400          # 25 ms at 16 kHz
HOP = 160            # 10 ms


def read_wav16(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        sr, n, ch, sw = w.getframerate(), w.getnframes(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(n)
    if sw != 2:
        raise ValueError(f"{path}: expected 16-bit PCM")
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, sr


def frame_db(x: np.ndarray, frame: int = FRAME, hop: int = HOP) -> np.ndarray:
    """dB (re full scale) of the RMS of overlapping frames: (n_frames,)."""
    if len(x) < frame:
        return np.full(1, -90.0)
    c = np.concatenate([[0.0], np.cumsum(x.astype(np.float64) ** 2)])
    starts = np.arange(0, len(x) - frame + 1, hop)
    ms = (c[starts + frame] - c[starts]) / frame
    return 10.0 * np.log10(ms + 1e-10)


def rolling_floor(db: np.ndarray, hop_s: float, half_s: float = 10.0, q: float = 10.0, step_s: float = 1.0) -> np.ndarray:
    """10th percentile of the frame levels in a +-half_s window, evaluated every step_s and interpolated back to every frame."""
    n = len(db)
    half = int(round(half_s / hop_s))
    step = max(1, int(round(step_s / hop_s)))
    centres = np.arange(0, n, step)
    vals = np.array([np.percentile(db[max(0, c - half): c + half + 1], q) for c in centres])
    return np.interp(np.arange(n), centres, vals)


class PhoneLevel:
    """Frame levels of one phone and its clock."""

    def __init__(self, cam: str, db: np.ndarray, clock, hop_s: float = HOP / 16000.0):
        self.cam, self.db, self.clock, self.hop_s = cam, db, clock, hop_s
        self.floor = rolling_floor(db, hop_s)

    def on_grid(self, m_lo: float, m_hi: float, grid: float = GRID) -> tuple[np.ndarray, np.ndarray]:
        """(level dB, floor dB) on the master grid cells [m_lo + k * grid, ...): NaN outside the recording. Cell value = max over the frames in it."""
        n = int(np.ceil((m_hi - m_lo) / grid))
        lvl = np.full(n, np.nan)
        flo = np.full(n, np.nan)
        # source time of every master cell edge (the clock is affine, to_source is monotone)
        for k in range(n):
            s0 = self.clock.to_source(m_lo + k * grid)
            s1 = self.clock.to_source(m_lo + (k + 1) * grid)
            i0 = int(np.floor(s0 / self.hop_s))
            i1 = int(np.ceil(s1 / self.hop_s))
            if i1 <= 0 or i0 >= len(self.db):
                continue
            i0, i1 = max(i0, 0), min(max(i1, i0 + 1), len(self.db))
            lvl[k] = self.db[i0:i1].max()
            flo[k] = self.floor[i0:i1].mean()
        return lvl, flo


class Levels:
    """The level tracks of all phones that have a clock, on one master grid."""

    def __init__(self, phones: dict[str, PhoneLevel], m_lo: float, m_hi: float, grid: float = GRID):
        self.m_lo, self.m_hi, self.grid = m_lo, m_hi, grid
        self.cams = sorted(phones)
        tracks = {c: phones[c].on_grid(m_lo, m_hi, grid) for c in self.cams}
        self.lvl = np.stack([tracks[c][0] for c in self.cams])        # (cams, cells)
        self.flo = np.stack([tracks[c][1] for c in self.cams])
        # smooth over 0.25 s (5 cells) with a max filter so a word's level is not the level of its quietest frame
        self.lvl_s = _maxfilt(self.lvl, 5)
        self.best = np.nanmax(np.where(np.isnan(self.lvl_s), -200.0, self.lvl_s), axis=0)

    def feats(self, cam: str, m0: float, m1: float) -> tuple[float, float]:
        """(snr dB, rel dB) of a phone over [m0, m1] (the 75th percentile of the cells, so a short word is not judged by its pauses)."""
        if cam not in self.cams:
            return 10.0, 0.0
        i = self.cams.index(cam)
        a = max(0, int((m0 - self.m_lo) / self.grid))
        b = min(self.lvl.shape[1], max(a + 1, int(np.ceil((m1 - self.m_lo) / self.grid))))
        seg = self.lvl_s[i, a:b]
        if seg.size == 0 or np.all(np.isnan(seg)):
            return 10.0, 0.0
        ok = ~np.isnan(seg)
        lv = float(np.percentile(seg[ok], 75))
        fl = float(np.nanmean(self.flo[i, a:b]))
        best = float(np.max(self.best[a:b])) if b > a else lv
        return lv - fl, min(0.0, lv - best)


def _maxfilt(a: np.ndarray, w: int) -> np.ndarray:
    out = a.copy()
    for s in range(1, w // 2 + 1):
        left = np.roll(a, s, axis=1)
        right = np.roll(a, -s, axis=1)
        left[:, :s] = np.nan
        right[:, -s:] = np.nan
        out = np.fmax(out, np.fmax(left, right))
    return out


def load_levels(clocks: dict, audio_dir: Path | None = None, cams: list[str] | None = None, m_lo: float = -10.0, m_hi: float = 362.0) -> Levels:
    audio_dir = Path(audio_dir) if audio_dir else SPEECH / "audio"
    phones: dict[str, PhoneLevel] = {}
    for cam, fname in SOURCES.items():
        clk = clocks.get(cam)
        if clk is None or (cams and cam not in cams):
            continue
        wav = audio_dir / (Path(fname).stem + ".16k.wav")
        if not wav.exists():
            continue
        x, sr = read_wav16(wav)
        phones[cam] = PhoneLevel(cam, frame_db(x), clk, hop_s=HOP / float(sr))
    return Levels(phones, m_lo, m_hi)
