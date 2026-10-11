"""The class audio of the recap film: one clean mix on the master clock, and the film-clock soundtrack derived from it.

Seven phones heard the same class. `class_mix` builds work/<take>/film/audio/class_mix.wav (48 kHz stereo, sample 0 = master time 0,
length = the master video = the capture's time base):

  1. every phone with a clock (work/<take>/sync/sync.json, `dancecap.speech.sources.load_clocks`: reference = slope * source + offset;
     00 and 03 have affine clocks) is put on the master clock with a sample-exact (fractional, drift-following) warp (`warp_to_master`);
  2. the BASE is the master phone (01, the picture's own sound). Another phone replaces it in a stretch only when it is CLEARLY better
     there (`plan_sources`): a speech-band SNR proxy (90th percentile of the 50 ms frame levels minus the 10th percentile over the
     wider neighbourhood) plus a bonus for resembling the room's consensus envelope must beat the base by `MARGIN_DB`, the stretch
     must be at least `MIN_STRETCH_S` long and every phone needs full coverage there;
  3. each replacement is spliced PHASE-ALIGNED (`splice_plan`): the cut sits at the quietest instant near the planned boundary, the
     incoming phone is shifted by the cross-correlation lag against the base in the stretch (a few ms of acoustic path) and gain-matched,
     then cross-faded over `XFADE_S` with a raised-cosine pair;
  4. gentle room-noise reduction (ffmpeg `afftdn`, tracking) at a modest strength and a 70 Hz high-pass for rumble; laughter and the
     teachers' dynamics are kept (no compressor);
  5. loudness -16 LUFS integrated, true peak <= -1 dBTP (static gain, a transparent look-ahead limiter only if a peak needs it).

Nothing leaves the machine: no third-party voice or speech service is involved (ffmpeg, numpy and scipy only).

`film_mix` derives the film soundtrack `<film dir>/audio/<direction.name>_mix.wav` from a direction: LOCKED segments copy the class mix at
their dance (= master) time at 1.0x; every slowed, held or replayed segment plays the room-tone bed instead (policy in docs/RECAP_FILM.md:
the teachers' speech is only ever heard at 1.0x locked to its picture). The bed is the speech-free `bed` stem of the chosen class mix
(`bed_for`: stems_regen/bed.wav for the regenerated mix, ...), else a room-tone loop of a stretch of the class mix that has NO word in it,
else silence: it is never a stretch that contains speech (S8: an earlier fallback looped a stretch with a teacher's words in it under the
replay). `slow_audio_report` / `python -m dancecap.film_recap_audio slow-check` measure what the film mix plays in every slowed, held or
replayed segment and fail when any of them is louder than `SLOW_MAX_DB` (room tone only).

`soundtrack_class_mix` picks class_mix_regen.wav (every teacher line regenerated, `dancecap.speech.v2v_regen`), class_mix_v2v_full.wav (the full cloned-voice
re-recording, `dancecap.speech.v2v_full`) or class_mix_v2v.wav (cleaned speech, `dancecap.speech.v2v_mix`) over class_mix.wav when they exist (switch:
film/audio/soundtrack.json {"class_mix": "auto" | "regen" | "v2v_full" | "v2v" | "original"});
class_mix.wav is never changed.

CLI (run from dancecap/):  python -m dancecap.film_recap_audio classmix | report | check | film-mix
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from .take import PACKAGE_ROOT

TAKE = "larissa_kadu_recap"
WORK = PACKAGE_ROOT / "work" / TAKE
FILM = WORK / "film"
AUDIO = FILM / "audio"
SOURCE_DIR = PACKAGE_ROOT.parent / "dances" / "Larissa-Kadu-recap"
SYNC_JSON = WORK / "sync" / "sync.json"
SPEECH = WORK / "speech"

SR = 48000          # output rate
AN_SR = 16000       # analysis rate (the speech package's own wavs)
MASTER = "01"
PHONES = ("00", "01", "02", "03", "05", "06")      # 04 has no clock
BAND = (250.0, 3500.0)
FRAME_S = 0.05
WIN_S = 8.0
HOP_S = 4.0
MARGIN_DB = 3.0          # a phone must beat the base by this much (quality points) to take over
MIN_NCC = 0.72           # ... and must resemble the room's consensus envelope at least this much
MIN_STRETCH_S = 10.0     # shortest replacement
XFADE_S = 0.12
SPLICE_SEARCH_S = 1.5    # look for the quietest instant this far around a boundary
MAX_LAG_S = 0.05
TARGET_LUFS = -16.0
TARGET_TP = -1.0

LOG = print


class AudioError(RuntimeError):
    """a problem the message explains (missing file, no clock, ffmpeg failure)"""


# ------------------------------------------------------------------------------------------------ files and ffmpeg


def ffmpeg(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", *args], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if check and r.returncode != 0:
        raise AudioError(f"ffmpeg {' '.join(args[:6])} ... failed: {r.stderr[-600:]}")
    return r


def extract_stereo(cam: str, *, source_dir: Path = SOURCE_DIR, out_dir: Path | None = None) -> Path:
    """the phone's stereo 48 kHz 16-bit wav in its OWN time (cached; the original mp4 is only read)"""
    out_dir = out_dir or (AUDIO / "src")
    out_dir.mkdir(parents=True, exist_ok=True)
    videos = sorted(source_dir.glob(f"{cam}-Recap-*.mp4"))
    if not videos:
        raise AudioError(f"no {cam}-Recap-*.mp4 in {source_dir}")
    out = out_dir / f"{cam}.48k.stereo.wav"
    if out.exists() and out.stat().st_mtime >= videos[0].stat().st_mtime:
        return out
    ffmpeg("-y", "-i", str(videos[0]), "-vn", "-map", "0:a:0", "-ac", "2", "-ar", str(SR), "-c:a", "pcm_s16le", str(out))
    return out


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """(float32 samples [n, channels], rate)"""
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return x, int(sr)


def write_wav(path: Path, x: np.ndarray, sr: int = SR, subtype: str = "PCM_16") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(x, -1.0, 1.0), sr, subtype=subtype)


# ------------------------------------------------------------------------------------------------ the master clock


def warp_to_master(x: np.ndarray, offset_s: float, slope: float, n_out: int, sr: int, *, win: int = 8192) -> tuple[np.ndarray, tuple[int, int]]:
    """Resample `x` (own time, [n] or [n, ch]) onto the master clock: master_t = slope * src_t + offset, so out[m] = x(src sample
    (m - offset*sr) / slope). Block-wise fractional delay (Hann windows, 50 % overlap-add, FFT phase ramp): the delay is constant in a
    block (it drifts by about 0.07 sample per 4096 samples at the worst clock here), exact at the block centres and smooth across.
    Returns (out [n_out, ch], (first, last+1) master samples that lie inside the source)."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]
    n_src, ch = x.shape
    hop = win // 2
    w = signal.windows.hann(win, sym=False).astype(np.float32)
    n_frames = (n_out + win) // hop + 1
    out = np.zeros((n_out + 2 * win, ch), dtype=np.float32)
    freqs = np.fft.rfftfreq(win)                                   # cycles per sample
    pad = 4 * win
    xp = np.zeros((n_src + 2 * pad, ch), dtype=np.float32)
    xp[pad:pad + n_src] = x
    for k in range(n_frames):
        m0 = k * hop - hop                                          # first master sample of the frame (frames start before 0)
        mc = m0 + win / 2.0
        d = (mc - offset_s * sr) / slope - mc                       # source sample = master sample + d
        n0 = int(math.floor(d))
        f = d - n0
        s0 = m0 + n0 + pad
        if s0 < 0 or s0 + win > xp.shape[0]:
            continue
        seg = xp[s0:s0 + win]
        if not seg.any():
            continue
        spec = np.fft.rfft(seg, axis=0)
        spec *= np.exp(2j * np.pi * freqs * f)[:, None].astype(np.complex64)   # y[m] = seg[m + f]
        y = np.fft.irfft(spec, n=win, axis=0).astype(np.float32) * w[:, None]
        a = m0 + win                                                # out has `win` samples of lead-in
        if a < 0 or a + win > out.shape[0]:
            continue
        out[a:a + win] += y
    res = out[win:win + n_out]
    first = max(0, int(math.ceil(offset_s * sr)))
    last = min(n_out, int(math.floor((slope * (n_src - 1) / sr + offset_s) * sr)) + 1)
    return res, (first, max(first, last))


# ------------------------------------------------------------------------------------------------ analysis


def band_db(x: np.ndarray, sr: int, band: tuple[float, float] = BAND, frame_s: float = FRAME_S) -> np.ndarray:
    """speech-band frame levels in dB (re full scale RMS), one value per `frame_s`"""
    if x.ndim > 1:
        x = x.mean(axis=1)
    sos = signal.butter(4, [band[0], min(band[1], sr / 2 - 100)], btype="band", fs=sr, output="sos")
    y = signal.sosfiltfilt(sos, x.astype(np.float64))
    n = int(round(frame_s * sr))
    m = len(y) // n
    e = (y[:m * n].reshape(m, n) ** 2).mean(axis=1)
    return 10.0 * np.log10(e + 1e-12)


def quality(e_db: np.ndarray, i0: int, i1: int, ctx0: int, ctx1: int) -> tuple[float, float, float]:
    """(speech level, noise floor, snr) in dB: 90th percentile of the window's frames, 10th percentile of the wider context"""
    w = e_db[i0:i1]
    c = e_db[ctx0:ctx1]
    s = float(np.percentile(w, 90))
    nz = float(np.percentile(c, 10))
    return s, nz, s - nz


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    d = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / d) if d > 1e-9 else 0.0


@dataclass
class Window:
    t0: float
    t1: float
    best: str
    base: str
    delta: float
    scores: dict[str, float] = field(default_factory=dict)
    ncc: dict[str, float] = field(default_factory=dict)
    snr: dict[str, float] = field(default_factory=dict)


def window_table(levels: dict[str, np.ndarray], cover: dict[str, tuple[float, float]], *, base: str = MASTER, duration: float,
                 win_s: float = WIN_S, hop_s: float = HOP_S, frame_s: float = FRAME_S, ncc_weight: float = 6.0) -> list[Window]:
    """Per window: quality = SNR proxy + ncc_weight * (resemblance of the phone's level contour to the median contour of the others).
    Only phones that cover the whole window (plus the context for the floor) take part."""
    out: list[Window] = []
    nf = lambda t: int(round(t / frame_s))
    t = 0.0
    while t + win_s <= duration + 1e-6:
        t0, t1 = t, t + win_s
        i0, i1 = nf(t0), nf(t1)
        c0, c1 = max(0, nf(t0 - 12.0)), min(min(len(v) for v in levels.values()), nf(t1 + 12.0))
        elig = [p for p in levels if cover[p][0] <= t0 + 0.05 and cover[p][1] >= t1 - 0.05]
        if base not in elig:
            t += hop_s
            continue
        scores: dict[str, float] = {}
        nccs: dict[str, float] = {}
        snrs: dict[str, float] = {}
        for p in elig:
            others = [levels[q][i0:i1] for q in elig if q != p]
            ref = np.median(np.stack(others), axis=0) if others else levels[p][i0:i1]
            ncc_p = ncc(levels[p][i0:i1], ref) if others else 1.0
            c_lo = max(c0, nf(cover[p][0]) + 1)
            _, _, snr = quality(levels[p], i0, i1, c_lo, c1)
            nccs[p], snrs[p] = ncc_p, snr
            scores[p] = snr + ncc_weight * ncc_p
        best = max(scores, key=scores.get)
        out.append(Window(t0, t1, best, base, scores[best] - scores[base], scores, nccs, snrs))
        t += hop_s
    return out


def plan_sources(windows: list[Window], *, base: str = MASTER, margin_db: float = MARGIN_DB, min_ncc: float = MIN_NCC,
                 min_stretch_s: float = MIN_STRETCH_S, hop_s: float = HOP_S, win_s: float = WIN_S) -> list[dict]:
    """Stretches [{t0, t1, cam, gain_db}] where another phone is clearly better than the base: the window's best phone beats the base by
    `margin_db` and resembles the room (ncc >= min_ncc); consecutive windows with the same winner merge; a stretch shorter than
    `min_stretch_s` is dropped. Hysteresis: a stretch keeps its phone while it is within half the margin of the base."""
    stretches: list[dict] = []
    cur: dict | None = None
    for w in windows:
        win_best = w.best if (w.best != base and w.delta >= margin_db and w.ncc.get(w.best, 0.0) >= min_ncc) else None
        if cur is not None and win_best is None and w.scores.get(cur["cam"], -1e9) - w.scores.get(base, 0.0) >= margin_db * 0.5 \
                and w.ncc.get(cur["cam"], 0.0) >= min_ncc:
            win_best = cur["cam"]
        if win_best is None:
            if cur:
                stretches.append(cur)
                cur = None
            continue
        if cur and cur["cam"] == win_best and w.t0 <= cur["t1"] + 1e-6:
            cur["t1"] = w.t1
            cur["windows"] += 1
            cur["delta_db"].append(round(w.scores[win_best] - w.scores[base], 2))
        else:
            if cur:
                stretches.append(cur)
            cur = {"t0": w.t0, "t1": w.t1, "cam": win_best, "windows": 1, "delta_db": [round(w.scores[win_best] - w.scores[base], 2)]}
    if cur:
        stretches.append(cur)
    keep = [s for s in stretches if s["t1"] - s["t0"] >= min_stretch_s]
    for s in keep:
        s["mean_delta_db"] = round(float(np.mean(s["delta_db"])), 2)
    return keep


def quietest_point(base_db: np.ndarray, t: float, search_s: float = SPLICE_SEARCH_S, frame_s: float = FRAME_S) -> float:
    """the time of the lowest base level within +-search_s of t (a splice goes where the room is quietest)"""
    i = int(round(t / frame_s))
    k = int(round(search_s / frame_s))
    lo, hi = max(0, i - k), min(len(base_db), i + k + 1)
    if hi <= lo:
        return t
    j = lo + int(np.argmin(base_db[lo:hi]))
    return (j + 0.5) * frame_s


def xcorr_lag(ref: np.ndarray, sig: np.ndarray, sr: int, max_lag_s: float = MAX_LAG_S, band: tuple[float, float] | None = None) -> tuple[float, float]:
    """lag (samples, fractional, positive = `sig` is LATER than `ref`) and the peak's z-score of their GCC-PHAT, with the estimator that
    placed the phones' clocks (`dancecap.recap_sync.estimate_delay`: PHAT weighting inside the speech band, Hann ramps on both window
    ends). Two things learned on these phones: a plain cross-correlation of voiced speech has equally high peaks at every multiple of the
    pitch period and picked lags of +-60 ms; and an UNTAPERED PHAT turns the window's hard edges into a correlation peak at lag 0 for any
    two signals (z 20 for independent noise), which would have 'confirmed' every clock. z >= 6 is the sync stage's acceptance (the noise
    maximum is z 3.9-4.5)."""
    from . import recap_sync as rs
    if ref.ndim > 1:
        ref = ref.mean(axis=1)
    if sig.ndim > 1:
        sig = sig.mean(axis=1)
    kw = {} if band is None else {"band": band}
    r = rs.estimate_delay(np.asarray(ref, np.float32), np.asarray(sig, np.float32), sr, lag_s=(-max_lag_s, max_lag_s), **kw)
    return float(r["delay_s"] * sr), float(r["z"])


def robust_lag(ref: np.ndarray, sig: np.ndarray, sr: int, *, win_s: float = 8.0, hop_s: float = 4.0, max_lag_s: float = MAX_LAG_S,
               min_peak: float = 6.0, min_windows: int = 2) -> dict:
    """The lag of `sig` against `ref` over a stretch, from the cross-correlation of overlapping sub-windows: the median of the windows
    whose PHAT sharpness (z) is >= min_peak (a stretch with two arrivals - the live voice and the loudspeaker - gives a broad peak over the
    whole stretch but a clear one in each short window). {lag_samples, lag_ms, mad_ms, n_windows, n_used, peak_median}; lag_samples is
    None when fewer than a third of the windows are usable."""
    n = min(len(ref), len(sig))
    w, h = int(win_s * sr), int(hop_s * sr)
    lags, peaks = [], []
    i = 0
    while i + w <= n:
        lag, pk = xcorr_lag(ref[i:i + w], sig[i:i + w], sr, max_lag_s)
        lags.append(lag)
        peaks.append(pk)
        i += h
    if not lags:
        lag, pk = xcorr_lag(ref[:n], sig[:n], sr, max_lag_s)
        lags, peaks = [lag], [pk]
    lags_a, peaks_a = np.array(lags), np.array(peaks)
    use = peaks_a >= min_peak
    if use.sum() < max(min_windows, len(lags_a) // 3):
        return {"lag_samples": None, "lag_ms": None, "mad_ms": None, "n_windows": len(lags_a), "n_used": int(use.sum()),
                "peak_median": float(np.median(peaks_a))}
    med = float(np.median(lags_a[use]))
    mad = float(np.median(np.abs(lags_a[use] - med)))
    return {"lag_samples": med, "lag_ms": med / sr * 1000.0, "mad_ms": mad / sr * 1000.0, "n_windows": len(lags_a), "n_used": int(use.sum()),
            "peak_median": float(np.median(peaks_a[use]))}


def lookahead_limit(x: np.ndarray, ceiling_db: float, sr: int = SR, look_s: float = 0.02) -> np.ndarray:
    """Transparent offline peak limiter: the gain needed to keep every sample (all channels) under the ceiling is spread over +-look_s
    (minimum filter, then a Hann moving average of the same half-width), so a peak is reached by a smooth dip that starts before it and
    never overshoots: the result stays under the ceiling by construction."""
    from scipy.ndimage import minimum_filter1d
    c = 10 ** (ceiling_db / 20.0)
    pk = np.abs(x).max(axis=1)
    if pk.max() <= c:
        return x
    r = np.minimum(1.0, c / np.maximum(pk, 1e-9))
    half = max(1, int(look_s * sr))
    g = minimum_filter1d(r, size=2 * half + 1, mode="nearest")
    k = signal.windows.hann(2 * half + 1, sym=True)
    k = (k / k.sum()).astype(np.float64)
    # only the neighbourhoods of reductions need the smoothing (the rest is exactly 1)
    idx = np.where(g < 1.0)[0]
    out_g = np.ones_like(g)
    if idx.size:
        lo, hi = max(0, idx[0] - half), min(len(g), idx[-1] + half + 1)
        out_g[lo:hi] = np.convolve(g[lo:hi], k, mode="same")
    return (x * np.minimum(out_g, 1.0)[:, None].astype(np.float32)).astype(np.float32)


def crossfade_curves(n: int) -> tuple[np.ndarray, np.ndarray]:
    """(fade out, fade in) raised-cosine pair, power complementary for uncorrelated material and amplitude complementary for aligned"""
    u = np.linspace(0.0, 1.0, n, dtype=np.float32)
    fin = 0.5 - 0.5 * np.cos(np.pi * u)
    return 1.0 - fin, fin


def splice(base: np.ndarray, repl: np.ndarray, i0: int, i1: int, xf: int) -> np.ndarray:
    """base with repl[i0:i1] swapped in (both [n, ch], on the same sample grid); the edges cross-fade over `xf` samples centred on i0 and i1"""
    out = base.copy()
    h = xf // 2
    a, b = max(0, i0 - h), min(len(base), i1 + h)
    fout, fin = crossfade_curves(xf)
    seg = repl[a:b].copy()
    old = base[a:b]
    gain = np.zeros(b - a, dtype=np.float32)          # weight of the replacement
    gain[:] = 1.0
    if i0 - h >= 0 and xf > 0:
        n = min(xf, b - a)
        gain[:n] = fin[:n]
    if i1 + h <= len(base) and xf > 0:
        n = min(xf, b - a)
        gain[-n:] = np.minimum(gain[-n:], fout[:n] if n == xf else fout[xf - n:])
    out[a:b] = old * (1.0 - gain[:, None]) + seg * gain[:, None]
    return out


# ------------------------------------------------------------------------------------------------ loudness


_LUFS = re.compile(r"I:\s+(-?[\d.]+|-inf)\s+LUFS")
_LRA = re.compile(r"LRA:\s+(-?[\d.]+)\s+LU")
_TP = re.compile(r"Peak:\s+(-?[\d.]+|-inf)\s+dBFS")


def measure_loudness(path: Path) -> dict:
    """integrated loudness (LUFS), loudness range (LU) and true peak (dBTP) with ffmpeg's ebur128"""
    r = ffmpeg("-i", str(path), "-af", "ebur128=peak=true:framelog=quiet", "-f", "null", "-", check=False)
    txt = r.stderr
    tail = txt[txt.rfind("Summary:"):] if "Summary:" in txt else txt
    i = _LUFS.search(tail)
    lra = _LRA.search(tail)
    tp = _TP.search(tail)
    if not i:
        raise AudioError(f"cannot read the loudness of {path.name}: {txt[-300:]}")
    return {"lufs": float(i.group(1)), "lra": float(lra.group(1)) if lra else None, "true_peak_dbtp": float(tp.group(1)) if tp else None}


def sample_peak_db(x: np.ndarray) -> float:
    p = float(np.max(np.abs(x))) if x.size else 0.0
    return 20.0 * math.log10(p) if p > 0 else -200.0


# ------------------------------------------------------------------------------------------------ the mix


def load_clocks() -> dict:
    from .speech.sources import load_clocks as lc
    return lc(SYNC_JSON, None)


def _master_duration() -> float:
    d = json.loads(SYNC_JSON.read_text(encoding="utf-8"))
    return float(d["master"]["duration_s"])


def _an_wav(cam: str) -> Path:
    return SPEECH / "audio" / next(p.name for p in sorted((SPEECH / "audio").glob(f"{cam}-Recap-*.16k.wav")))


def analysis_stack(clocks: dict, duration: float, cams=PHONES) -> tuple[dict[str, np.ndarray], dict[str, tuple[float, float]]]:
    """{cam: frame levels on the master clock}, {cam: (first, last) master seconds covered}"""
    n_out = int(round(duration * AN_SR))
    levels, cover = {}, {}
    for cam in cams:
        clk = clocks.get(cam)
        if clk is None:
            continue
        x, sr = read_wav(_an_wav(cam))
        assert sr == AN_SR, sr
        y, (a, b) = warp_to_master(x[:, 0], clk.offset, clk.slope, n_out, AN_SR)
        levels[cam] = band_db(y[:, 0], AN_SR)
        cover[cam] = (a / AN_SR, b / AN_SR)
        LOG(f"  analysed {cam}: covers {cover[cam][0]:.2f}-{cover[cam][1]:.2f} s")
    return levels, cover


def build_class_mix(out: Path | None = None, *, denoise: bool = True, margin_db: float = MARGIN_DB, only_base: bool = False,
                    cams=PHONES) -> dict:
    """the whole pipeline; returns the report dict (also written next to the wav as class_mix.json)"""
    out = out or (AUDIO / "class_mix.wav")
    AUDIO.mkdir(parents=True, exist_ok=True)
    clocks = load_clocks()
    if clocks.get(MASTER) is None:
        raise AudioError("no master clock")
    duration = _master_duration()
    n = int(round(duration * SR))
    LOG(f"class mix: master {duration:.3f} s, {n} samples at {SR} Hz")
    levels, cover = analysis_stack(clocks, duration, cams)
    wins = window_table(levels, cover, duration=duration)
    stretches = [] if only_base else plan_sources(wins, margin_db=margin_db)
    LOG(f"  {len(wins)} windows, {len(stretches)} replacement stretches: " + ", ".join(f"{s['cam']} {s['t0']:.0f}-{s['t1']:.0f}" for s in stretches))

    def stereo(cam: str) -> np.ndarray:
        x, sr = read_wav(extract_stereo(cam))
        clk = clocks[cam]
        y, _ = warp_to_master(x, clk.offset, clk.slope, n, SR)
        return y

    base = stereo(MASTER)
    base_db_16 = levels[MASTER]
    mix = base.copy()
    splices = []
    xf = int(XFADE_S * SR)
    cache: dict[str, np.ndarray] = {}
    for s in stretches:
        t0 = quietest_point(base_db_16, s["t0"])
        t1 = quietest_point(base_db_16, s["t1"])
        if t1 - t0 < MIN_STRETCH_S * 0.6:
            continue
        i0, i1 = int(t0 * SR), int(t1 * SR)
        cam = s["cam"]
        if cam not in cache:
            cache[cam] = stereo(cam)
        other = cache[cam]
        # alignment: the clocks are GCC-PHAT based and sub-millisecond (see docs/RECAP_FILM.md); the lag is measured again on the stretch
        # at 16 kHz, applied when the measurement is reliable and small, else the clock-exact alignment (0) stays
        rl = robust_lag(signal.resample_poly(base[i0:i1].mean(axis=1), 1, 3), signal.resample_poly(other[i0:i1].mean(axis=1), 1, 3), AN_SR)
        if rl["lag_samples"] is None or rl["mad_ms"] > 1.0 or abs(rl["lag_ms"]) > 2.0:
            # not confirmed: a phone whose alignment against the master cannot be measured (z < 6) would put the voice out of step with
            # the picture by up to its clock's uncertainty (05: +-16 ms, and one word measured 60 ms early): the master phone stays
            rec = {"cam": cam, "t0": round(t0, 3), "t1": round(t1, 3), "planned": [s["t0"], s["t1"]], "used": False,
                   "reason": "alignment against the master not confirmed (needs >= 2 windows with z >= 6 within 1 ms, lag <= 2 ms)",
                   "windows_used": f"{rl['n_used']}/{rl['n_windows']}", "phat_z_median": round(rl["peak_median"], 1), "mean_delta_db": s.get("mean_delta_db")}
            splices.append(rec)
            LOG(f"  not used: {rec}")
            continue
        applied = rl["lag_samples"] * (SR / AN_SR)
        aligned = shift_by(other, -applied)
        lag, peak = applied, rl["peak_median"]
        # level match over the stretch's speech frames (a constant gain per stretch, at most +-12 dB)
        gain_db = float(np.clip(_level_db(base[i0:i1]) - _level_db(aligned[i0:i1]), -12.0, 12.0))
        aligned = aligned * (10 ** (gain_db / 20.0))
        mix = splice(mix, aligned, i0, i1, xf)
        rec = {"cam": cam, "t0": round(t0, 3), "t1": round(t1, 3), "planned": [s["t0"], s["t1"]], "used": True, "lag_applied_ms": round(lag / SR * 1000, 4),
               "lag_measured_ms": None if rl["lag_ms"] is None else round(rl["lag_ms"], 4), "lag_mad_ms": None if rl["mad_ms"] is None else round(rl["mad_ms"], 4),
               "windows_used": f"{rl['n_used']}/{rl['n_windows']}", "phat_z_median": round(peak, 1), "gain_db": round(gain_db, 2),
               "mean_delta_db": s.get("mean_delta_db")}
        splices.append(rec)
        LOG(f"  splice {rec}")
    raw_path = out.with_name("class_mix_raw.wav")
    write_wav(raw_path, mix)
    final = _finish(mix, out, denoise=denoise)
    report = {"schema": "dancecap.film_recap/1/class_mix", "take": TAKE, "clock": "master = 01-Recap-Tina video time, seconds",
              "rate": SR, "channels": 2, "duration_s": round(n / SR, 3), "base": MASTER, "clocks": {c: (None if v is None else
              {"offset_s": round(v.offset, 6), "slope": v.slope}) for c, v in clocks.items()},
              "thresholds": {"margin_db": margin_db, "min_ncc": MIN_NCC, "min_stretch_s": MIN_STRETCH_S, "window_s": WIN_S, "hop_s": HOP_S,
                             "xfade_s": XFADE_S},
              "windows": len(wins), "stretches_planned": stretches, "splices": splices,
              "per_window_best": _window_summary(wins), "final": final}
    out.with_suffix(".json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


def shift_by(x: np.ndarray, lag_samples: float) -> np.ndarray:
    """x moved by `lag_samples` (positive = later) on the same grid, fractional"""
    if abs(lag_samples) < 1e-6:
        return x
    n = len(x)
    # out[m] = x[m - lag]: warp with offset = lag/SR
    y, _ = warp_to_master(x, lag_samples / SR, 1.0, n, SR)
    return y


def _resid_db(a: np.ndarray, b: np.ndarray) -> float:
    """energy of the difference of two mono-ised signals relative to the first, in dB (lower = more alike)"""
    am, bm = a.mean(axis=1), b.mean(axis=1)
    g = float(np.dot(am, bm) / max(np.dot(bm, bm), 1e-12))        # best scale of b
    d = am - g * bm
    return 10.0 * math.log10(float(np.dot(d, d)) / max(float(np.dot(am, am)), 1e-12) + 1e-12)


def _level_db(x: np.ndarray) -> float:
    """speech level of a stretch: 90th percentile of its 50 ms band levels"""
    e = band_db(x, SR)
    return float(np.percentile(e, 90))


def _window_summary(wins: list[Window]) -> dict:
    cnt: dict[str, int] = {}
    for w in wins:
        cnt[w.best] = cnt.get(w.best, 0) + 1
    gaps = [w.delta for w in wins if w.best != w.base]
    return {"best_counts": cnt, "max_delta_db": round(max((w.delta for w in wins), default=0.0), 2),
            "windows_best_beats_base_by_margin": sum(1 for w in wins if w.delta >= MARGIN_DB), "median_gap_db_when_other": round(float(np.median(gaps)), 2) if gaps else None}


def remove_latency(ref: np.ndarray, proc: np.ndarray, max_ms: float = 200.0) -> tuple[np.ndarray, int]:
    """ffmpeg's afftdn (FFT frames) delays the audio by about 25 ms: measured here by the cross-correlation of the processed signal against its input
    over three 20 s excerpts (must agree to a sample), then undone by dropping that many samples at the start (zeros at the end), so the processed mix
    stays sample-aligned with the master clock (a 25 ms lag was found by the word-onset check against the master phone's own audio)."""
    lags = []
    n = len(ref)
    for t0 in (0.15, 0.45, 0.75):
        i0 = int(n * t0)
        x = ref[i0:i0 + 20 * SR].mean(axis=1)
        y = proc[i0:i0 + 20 * SR].mean(axis=1)
        k = min(len(x), len(y))
        if k < 2 * SR:
            continue
        nfft = 1 << (2 * k - 1).bit_length()
        cc = np.fft.irfft(np.fft.rfft(y[:k], nfft) * np.conj(np.fft.rfft(x[:k], nfft)), nfft)
        m = int(max_ms / 1000 * SR)
        v = np.concatenate([cc[nfft - m:], cc[:m + 1]])
        lags.append(int(np.argmax(v)) - m)
    if not lags:
        return proc, 0
    d = int(np.median(lags))
    if d <= 0:
        return proc, 0
    out = np.zeros_like(proc)
    out[:len(proc) - d] = proc[d:]
    return out, d


def _finish(mix: np.ndarray, out: Path, *, denoise: bool) -> dict:
    """high-pass, gentle noise reduction, loudness; writes `out` and returns the measurements. The loudness is solved in float: a static
    gain G is raised until the mix, after the look-ahead peak limiter at the ceiling, measures TARGET_LUFS (3-4 passes); the true peak
    (ebur128, oversampled) must then be <= TARGET_TP, else the ceiling is lowered and the solve repeated."""
    tmp = out.with_name("class_mix_stage.wav")
    write_wav(tmp, mix, subtype="PCM_24")
    stages = ["highpass=f=70:poles=2"]
    if denoise:
        stages.append("afftdn=nr=10:nf=-50:tn=1:tr=1:om=o")
    den = out.with_name("class_mix_denoised.wav")
    ffmpeg("-y", "-i", str(tmp), "-af", ",".join(stages), "-c:a", "pcm_f32le", str(den))
    x0, _ = read_wav(den)
    x0, latency_samples = remove_latency(mix, x0)
    m0 = measure_loudness(den)
    probe = out.with_name("class_mix_probe.wav")
    ceiling = TARGET_TP - 0.4
    log = []
    final = None
    for outer in range(4):
        gain = TARGET_LUFS - m0["lufs"]
        y = x0
        m = m0
        for it in range(6):
            y = lookahead_limit(x0 * (10 ** (gain / 20.0)), ceiling)
            sf.write(str(probe), y, SR, subtype="FLOAT")
            m = measure_loudness(probe)
            err = TARGET_LUFS - m["lufs"]
            log.append({"ceiling_db": round(ceiling, 2), "gain_db": round(gain, 3), "lufs": m["lufs"], "true_peak_dbtp": m["true_peak_dbtp"]})
            if abs(err) <= 0.05:
                break
            gain += err
        final = (y, m, gain)
        if m["true_peak_dbtp"] is not None and m["true_peak_dbtp"] <= TARGET_TP - 0.02:
            break
        ceiling -= max(0.2, (m["true_peak_dbtp"] or 0.0) - TARGET_TP + 0.1)
    y, m, gain = final
    pre = np.abs(x0 * (10 ** (gain / 20.0))).max(axis=1)
    post = np.abs(y).max(axis=1)
    red = 20.0 * np.log10(np.maximum(post, 1e-9) / np.maximum(pre, 1e-9))
    limiter = {"samples_touched_pct": round(float((red < -0.05).mean() * 100), 3), "max_reduction_db": round(float(-red.min()), 2),
               "seconds_with_reduction_over_3_db": int(len({int(i // SR) for i in np.where(red < -3.0)[0]}))}
    write_wav(out, y, subtype="PCM_16")
    for p in (tmp, den, probe):
        p.unlink(missing_ok=True)
    mq = measure_loudness(out)         # the 16-bit file is what is delivered: verify it, not the float
    return {"filter_latency_ms": round(latency_samples / SR * 1000, 3), "before_norm": m0, "gain_db": round(gain, 2), "ceiling_dbfs": round(ceiling, 2), "solve": log, "after": mq,
            "sample_peak_db": round(sample_peak_db(y), 2), "limiter": limiter, "method": "static gain + look-ahead peak limiter (+-20 ms)",
            "denoise": "afftdn nr=10 dB nf=-50 dB tracking; high-pass 70 Hz" if denoise else "none (high-pass only)"}


# ------------------------------------------------------------------------------------------------ report and checks


def noise_floor_db(path: Path | np.ndarray, sr: int = SR) -> dict:
    """speech-band level statistics of a wav: 10th / 50th / 90th percentile of the 50 ms frames"""
    if isinstance(path, np.ndarray):
        x = path
    else:
        x, sr = read_wav(path)
    e = band_db(x, sr)
    return {"p10": round(float(np.percentile(e, 10)), 2), "p50": round(float(np.percentile(e, 50)), 2), "p90": round(float(np.percentile(e, 90)), 2),
            "snr_proxy": round(float(np.percentile(e, 90) - np.percentile(e, 10)), 2)}


def spectrogram_png(paths: dict[str, Path], out: Path, t0: float = 60.0, t1: float = 90.0) -> Path:
    """before / after spectrograms of the same master seconds (log frequency up to 8 kHz)"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(paths), 1, figsize=(11, 3.0 * len(paths)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (name, p) in zip(axes, paths.items()):
        x, sr = read_wav(p)
        seg = x[int(t0 * sr):int(t1 * sr)].mean(axis=1)
        f, t, S = signal.spectrogram(seg, sr, nperseg=2048, noverlap=1536)
        ax.pcolormesh(t + t0, f, 10 * np.log10(S + 1e-12), shading="auto", vmin=-110, vmax=-40)
        ax.set_ylim(50, 8000)
        ax.set_yscale("symlog", linthresh=500)
        ax.set_ylabel(name)
    axes[-1].set_xlabel("master seconds")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=90)
    plt.close(fig)
    return out


def _onset_ms(mono: np.ndarray, sr: int, centre_s: float, rise_db: float = 8.0, pre_s: float = 0.4, post_s: float = 0.5) -> float | None:
    """Time (ms, relative to `centre_s`) at which the speech-band level of `mono` rises `rise_db` above the pause before it and stays up for
    30 ms, searched in [centre - pre_s, centre + post_s]; None when there is no such rise (the pause was not quiet)."""
    fr = 0.01
    i0 = max(0, int((centre_s - pre_s) * sr))
    seg = mono[i0:int((centre_s + post_s) * sr)]
    if len(seg) < int(0.4 * sr):
        return None
    e = band_db(seg, sr, frame_s=fr)
    k = int(0.25 / fr)
    floor = float(np.percentile(e[:k], 50))
    for i in range(int(0.1 / fr), len(e) - 3):
        if (e[i:i + 3] >= floor + rise_db).all():
            return (i0 / sr + i * fr - centre_s) * 1000.0
    return None


def clean_onset_words(transcript: dict, n_words: int = 10) -> list[dict]:
    """words whose onset can be measured: confident (>= 0.95), 0.25-1.0 s long, after a pause of at least 0.4 s, spread over the class"""
    cands = []
    for u in transcript["utterances"]:
        ws = u["words"]
        for i, w in enumerate(ws):
            if w.get("kind") != "word" or w["start"] < 6 or w.get("conf", 0) < 0.95 or not (0.25 <= w["end"] - w["start"] <= 1.0):
                continue
            prev_end = ws[i - 1]["end"] if i > 0 else u["start"] - 1.0
            if w["start"] - prev_end < 0.4:
                continue
            cands.append(w)
    cands.sort(key=lambda w: w["start"])
    if len(cands) <= n_words:
        return cands
    return [cands[int(round(k * (len(cands) - 1) / max(1, n_words - 1)))] for k in range(n_words)]


def speech_activity_lag(mono: np.ndarray, sr: int, spans: list[tuple[float, float]], offset_fn=None, max_lag_s: float = 0.4) -> dict:
    """Global lag between the transcript's speech activity (1 inside a word, 0 outside) and the speech-band level of `mono`: the cross-correlation
    of the two on a 10 ms grid over the whole class. A well-synchronised soundtrack peaks at about 0 (the ASR times are good to a few tens of ms)."""
    fr = 0.01
    e = band_db(mono, sr, frame_s=fr)
    e = np.clip(e - np.percentile(e, 20), 0, None)
    act = np.zeros(len(e), dtype=np.float64)
    for a0, a1 in spans:
        if offset_fn is not None:
            a0, a1 = offset_fn(a0), offset_fn(a1)
        act[max(0, int(a0 / fr)):min(len(act), int(a1 / fr) + 1)] = 1.0
    e = e - e.mean()
    act = act - act.mean()
    n = len(e)
    nfft = 1 << (2 * n - 1).bit_length()
    cc = np.fft.irfft(np.fft.rfft(e, nfft) * np.conj(np.fft.rfft(act, nfft)), nfft)     # cc[k]: the level lags the activity by k frames
    m = int(max_lag_s / fr)
    vals = np.concatenate([cc[nfft - m:], cc[:m + 1]])
    lags = np.arange(-m, m + 1)
    j = int(np.argmax(vals))
    # parabolic refinement
    frac = 0.0
    if 0 < j < len(vals) - 1:
        den = vals[j - 1] - 2 * vals[j] + vals[j + 1]
        frac = 0.5 * (vals[j - 1] - vals[j + 1]) / den if abs(den) > 1e-12 else 0.0
    norm = math.sqrt(float(np.dot(e, e) * np.dot(act, act)))
    return {"lag_ms": round(float((lags[j] + frac) * fr * 1000), 1), "peak": round(float(vals[j] / norm), 3) if norm > 0 else 0.0}


def word_timing_check(mix_path: Path, transcript: Path | None = None, n_words: int = 10, offset_fn=None, reference: Path | None = None) -> dict:
    """Timing of the transcript's words against a soundtrack. `offset_fn(master s) -> soundtrack s` (identity for the class mix; the film mix has
    the inserts of the direction). For `n_words` words that start after a clear pause: the time the speech-band level rises above the pause
    (ms after the transcript's word start; the ASR times are good to a few tens of ms), the same measured in the master phone's own audio
    (`reference`: the mix must not move anything against it), and the global lag of the whole class's speech activity."""
    transcript = transcript or (SPEECH / "transcript_final.json")
    doc = json.loads(transcript.read_text(encoding="utf-8"))
    x, sr = read_wav(mix_path)
    mono = x.mean(axis=1)
    ref_mono = None
    ref_sr = sr
    if reference is not None and Path(reference).exists():
        r, ref_sr = read_wav(reference)
        ref_mono = r.mean(axis=1)
    rows = []
    for w in clean_onset_words(doc, n_words):
        t = offset_fn(w["start"]) if offset_fn else w["start"]
        on = _onset_ms(mono, sr, t)
        rec = {"word": w["text"], "start_transcript_s": round(w["start"], 3), "start_in_soundtrack_s": round(t, 3),
               "onset_after_start_ms": None if on is None else round(on, 1)}
        if ref_mono is not None:
            on_ref = _onset_ms(ref_mono, ref_sr, w["start"])
            rec["onset_in_master_phone_ms"] = None if on_ref is None else round(on_ref, 1)
            rec["mix_minus_master_phone_ms"] = None if on is None or on_ref is None else round(on - on_ref, 1)
        rows.append(rec)
    spans = [(w["start"], w["end"]) for u in doc["utterances"] for w in u["words"] if w["start"] >= 0 and w.get("kind") == "word"]
    out = {"words": rows, "activity_lag": speech_activity_lag(mono, sr, spans, offset_fn)}
    if ref_mono is not None:
        out["activity_lag_master_phone"] = speech_activity_lag(ref_mono, ref_sr, spans)
    deltas = [r["mix_minus_master_phone_ms"] for r in rows if r.get("mix_minus_master_phone_ms") is not None]
    ons = [r["onset_after_start_ms"] for r in rows if r["onset_after_start_ms"] is not None]
    out["summary"] = {"measured": len(ons), "median_onset_after_start_ms": round(float(np.median(ons)), 1) if ons else None,
                      "max_abs_mix_minus_master_phone_ms": round(float(np.max(np.abs(deltas))), 1) if deltas else None}
    return out


# ------------------------------------------------------------------------------------------------ the room tone bed and the film mix


def room_tone(mix: np.ndarray, words_spans: list[tuple[float, float]], sr: int = SR, length_s: float = 6.0, *, strict: bool = False) -> np.ndarray | None:
    """a loop of room tone: the quietest speech-free stretch of the class mix (no word within 0.5 s), `length_s` long, made loopable by a
    raised-cosine self cross-fade; stereo. If no speech-free stretch exists the quietest stretch overall is used (that stretch can hold
    speech: `strict=True` returns None instead, which is what the film mix asks for)."""
    mono = mix.mean(axis=1)
    e = band_db(mono, sr, band=(100.0, 6000.0), frame_s=0.1)
    n = int(length_s / 0.1)
    ok = np.ones(len(e), dtype=bool)
    for a, b in words_spans:
        ok[max(0, int((a - 0.5) / 0.1)):min(len(e), int((b + 0.5) / 0.1) + 1)] = False
    best, best_i = 1e9, 0
    for i in range(0, len(e) - n, 5):
        if not ok[i:i + n].all():
            continue
        v = float(np.mean(e[i:i + n]))
        if v < best:
            best, best_i = v, i
    if best >= 1e9:
        if strict:
            return None
        for i in range(0, len(e) - n, 5):
            v = float(np.mean(e[i:i + n]))
            if v < best:
                best, best_i = v, i
    seg = mix[int(best_i * 0.1 * sr):int((best_i * 0.1 + length_s) * sr)].copy()
    return loopable(seg, sr)


def loopable(seg: np.ndarray, sr: int = SR, xf_s: float = 1.0) -> np.ndarray:
    """a stretch made loopable: its tail is cross-faded into its head (raised cosine) and cut off; the loop is `xf_s` shorter"""
    xf = int(xf_s * sr)
    fout, fin = crossfade_curves(xf)
    loop = seg[:-xf].copy()
    loop[:xf] = seg[:xf] * fin[:, None] + seg[-xf:] * fout[:, None]
    return loop


STEM_DIRS = {"class_mix_regen.wav": "stems_regen", "class_mix_v2v_full.wav": "stems_v2v_full", "class_mix_v2v.wav": "stems"}


def bed_for(class_mix: Path) -> Path | None:
    """the room-tone `bed` stem of a class mix (stems_regen/bed.wav for class_mix_regen.wav, ...), when it exists. The stem holds the mix's
    room tone only (built from non-speech frames, ducked under the speech); the original class mix has none."""
    d = STEM_DIRS.get(class_mix.name)
    if d is None:
        return None
    p = class_mix.parent / d / "bed.wav"
    return p if p.exists() else None


def stem_bed_loop(stem: np.ndarray, sr: int = SR, length_s: float = 6.0, block_s: float = 0.5) -> np.ndarray | None:
    """a loop out of the bed stem: the stretch whose QUIETEST half second is loudest (a steady room tone, never a ducked-out hole), made loopable.
    None when the stem is shorter than a stretch or is silent throughout."""
    mono = stem.mean(axis=1)
    blk = int(block_s * sr)
    nb = len(mono) // blk
    k = int(round(length_s / block_s))
    if nb < k + 1:
        return None
    lv = np.array([10.0 * math.log10(float(np.mean(mono[i * blk:(i + 1) * blk] ** 2)) + 1e-18) for i in range(nb)])
    if float(lv.max()) < -110.0:
        return None
    best = max(range(nb - k), key=lambda i: float(lv[i:i + k].min()))
    return loopable(stem[best * blk:(best + k) * blk].copy(), sr)


def tile(loop: np.ndarray, n: int) -> np.ndarray:
    reps = int(math.ceil(n / len(loop)))
    return np.tile(loop, (reps, 1))[:n]


V2V_MIX = "class_mix_v2v.wav"
V2V_FULL_MIX = "class_mix_v2v_full.wav"      # V2: the full cloned-voice re-recording (dancecap.speech.v2v_full)
REGEN_MIX = "class_mix_regen.wav"            # V3: only regenerated ElevenLabs lines (dancecap.speech.v2v_regen)
SOUNDTRACK_CFG = "soundtrack.json"


def soundtrack_class_mix(audio_dir: Path | None = None) -> Path:
    """The class mix the film soundtrack is derived from (all 48 kHz stereo on the master clock, the same length). The switch is
    <audio dir>/soundtrack.json {"class_mix": "auto" | "regen" | "v2v_full" | "v2v" | "original"}:
    "regen" = `class_mix_regen.wav` (V3: every teacher line regenerated in the cloned voices), falling back to the v2v_full mix, the v2v mix and the original when a
    file is missing; "v2v_full" = `class_mix_v2v_full.wav` (the full cloned-voice re-recording, dancecap.speech.v2v_full), falling back to the v2v mix and then the original
    when a file is missing; "v2v" = `class_mix_v2v.wav` (V1: cleaned phones, 2 converted pieces), falling back to the original; "original" = the old
    `class_mix.wav`, which is never changed; "auto" (default) = the first of regen, v2v_full, v2v, original that exists."""
    d = audio_dir or AUDIO
    mode = "auto"
    cfg = d / SOUNDTRACK_CFG
    if cfg.exists():
        try:
            mode = str(json.loads(cfg.read_text(encoding="utf-8")).get("class_mix", "auto")).lower()
        except (OSError, ValueError):
            mode = "auto"
    full, v2v, regen = d / V2V_FULL_MIX, d / V2V_MIX, d / REGEN_MIX
    if mode in ("auto", "regen") and regen.exists():
        return regen
    if mode in ("auto", "regen", "v2v_full") and full.exists():
        return full
    if mode in ("auto", "regen", "v2v_full", "v2v") and v2v.exists():
        return v2v
    return d / "class_mix.wav"


def is_locked(seg: dict) -> bool:
    """a segment whose sound is the class mix at its dance time: speed 1, not a replay, not a hold, audio 'locked'"""
    return seg.get("audio", "locked") == "locked" and seg.get("speed", 1.0) == 1.0 and not seg.get("replay") and seg.get("hold_s") is None


def slow_segments(direction: dict) -> list[dict]:
    """the segments the film mix does NOT lock to the class sound (slow motion, replays, holds, audio 'bed'): [{id, film, speed, replay, hold_s, audio}]"""
    return [{"id": s.get("id"), "film": [float(s["film"][0]), float(s["film"][1])], "speed": s.get("speed", 1.0), "replay": bool(s.get("replay")),
             "hold_s": s.get("hold_s"), "audio": s.get("audio", "locked")} for s in direction.get("segments", []) if not is_locked(s)]


def film_bed(class_mix: Path, mix: np.ndarray, spans: list[tuple[float, float]], sr: int = SR) -> tuple[np.ndarray | None, float, str]:
    """(the bed loop, its gain, where it came from) for the segments that are not locked: the class mix's speech-free `bed` stem, else a loop of a
    stretch of the class mix with no word in it, else (None, 0, 'silence'). Never a stretch that has speech in it."""
    stem = bed_for(class_mix)
    if stem is not None:
        loop = stem_bed_loop(read_wav(stem)[0], sr)
        if loop is not None:
            return loop, 1.0, f"stem {stem.parent.name}/{stem.name}"
    loop = room_tone(mix, spans, sr, strict=True)
    if loop is not None:
        return loop, 10 ** (-3.0 / 20.0), "room tone of a speech-free stretch of the class mix"
    return None, 0.0, "silence"


def film_mix(direction_path: Path, class_mix: Path | None = None, out: Path | None = None, word_spans: list[tuple[float, float]] | None = None) -> dict:
    """Render `<film dir>/audio/<direction.name>_mix.wav` from a direction: sample 0 = film time 0, length = the film.
    LOCKED segments (speed 1, no replay) copy the class mix at their dance time; any other segment (slow motion, holds, replays, audio
    'bed') plays the room-tone bed (`film_bed`: the speech-free bed stem, else a speech-free loop, else silence), so the speech is never heard
    at another speed or out of sync with its picture. Junctions cross-fade over 0.15 s (silence->bed->speech, the speech never pre-empted:
    a locked segment starts on the speech's own time). The result lists what every such segment plays (level, peak)."""
    d = json.loads(direction_path.read_text(encoding="utf-8"))
    cm = class_mix or soundtrack_class_mix(direction_path.parent / "audio")
    out = out or (direction_path.parent / "audio" / f"{d['name']}_mix.wav")
    mix, sr = read_wav(cm)
    dur = float(d["timeline"]["film_duration_s"])
    n = int(round(dur * sr))
    spans = word_spans
    tr = SPEECH / "transcript_final.json"
    if spans is None:
        spans = []
        if tr.exists():
            t = json.loads(tr.read_text(encoding="utf-8"))
            spans = [(w["start"], w["end"]) for u in t["utterances"] for w in u["words"] if w["start"] >= 0]
    loop, bed_gain, bed_source = film_bed(cm, mix, spans, sr)
    bed = tile(loop, n + sr) if loop is not None else np.zeros((n + sr, mix.shape[1]), dtype=np.float32)
    film = np.zeros((n, mix.shape[1]), dtype=np.float32)
    bed_w = np.zeros(n, dtype=np.float32)        # weight of the bed
    xf = int(0.15 * sr)
    for s in d["segments"]:
        f0, f1 = int(round(s["film"][0] * sr)), int(round(s["film"][1] * sr))
        f1 = min(f1, n)
        if f1 <= f0:
            continue
        if is_locked(s):
            d0 = int(round(s["dance"][0] * sr))
            seg = mix[max(0, d0):d0 + (f1 - f0)]
            film[f0:f0 + len(seg)] = seg
            bed_w[f0:f1] = 0.0
        else:
            bed_w[f0:f1] = 1.0
    # smooth the bed weight at the junctions (a box of xf samples)
    if xf > 1:
        k = np.ones(xf, dtype=np.float32) / xf
        bed_w = np.convolve(bed_w, k, mode="same")
    film = film * (1.0 - bed_w[:, None]) + bed[:n] * bed_w[:, None] * bed_gain
    # the picture fades from black over its first 0.5 s and to black over its last 0.9 s (FilmOverlay): the sound follows
    fi, fo = min(n, int(0.12 * sr)), min(n, int(0.9 * sr))
    film[:fi] *= (0.5 - 0.5 * np.cos(np.pi * np.linspace(0, 1, fi, dtype=np.float32)))[:, None]
    film[n - fo:] *= (0.5 + 0.5 * np.cos(np.pi * np.linspace(0, 1, fo, dtype=np.float32)))[:, None]
    write_wav(out, film)
    slow = slow_segments(d)
    return {"out": out.name, "seconds": round(n / sr, 3), "bed_segments": len(slow), "bed_seconds": round(float(bed_w.sum() / sr), 2),
            "bed_source": bed_source, "slow": _slow_levels(slow, film, sr)}


SLOW_MAX_DB = -40.0     # the loudest a slowed / held / replayed stretch of the film mix may be (RMS dBFS): room tone, never speech (the speech sits near -20)
SLOW_EDGE_S = 0.2       # the junction cross-fades (0.15 s) are not judged


def _db(x: np.ndarray) -> float:
    return round(20.0 * math.log10(float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) + 1e-9), 1) if len(x) else -180.0


def _slow_levels(slow: list[dict], film: np.ndarray, sr: int) -> list[dict]:
    """per slowed segment: the RMS and peak (dBFS) of the film mix inside it (the 0.2 s next to a junction left out), and whether it is room tone only"""
    rows = []
    for sg in slow:
        a, b = sg["film"]
        i0, i1 = int(round((a + SLOW_EDGE_S) * sr)), int(round((b - SLOW_EDGE_S) * sr))
        x = film[max(0, i0):max(0, i1)]
        rows.append({**sg, "rms_db": _db(x), "peak_db": round(20.0 * math.log10(float(np.abs(x).max()) + 1e-9), 1) if len(x) else -180.0,
                     "room_tone_only": bool(len(x) == 0 or _db(x) <= SLOW_MAX_DB)})
    return rows


def slow_audio_report(direction_path: Path, mix_path: Path | None = None) -> list[dict]:
    """what the RENDERED film mix plays in each slowed, held or replayed segment of the direction (see `_slow_levels`)"""
    d = json.loads(direction_path.read_text(encoding="utf-8"))
    mix_path = mix_path or (direction_path.parent / "audio" / f"{d['name']}_mix.wav")
    film, sr = read_wav(mix_path)
    return _slow_levels(slow_segments(d), film, sr)


# ------------------------------------------------------------------------------------------------ CLI


def cmd_classmix(a: argparse.Namespace) -> int:
    rep = build_class_mix(denoise=not a.no_denoise, margin_db=a.margin_db, only_base=a.base_only)
    fin = rep["final"]["after"]
    print(f"class_mix.wav: {fin['lufs']} LUFS, true peak {fin['true_peak_dbtp']} dBTP, {len(rep['splices'])} splices")
    return 0


def cmd_report(a: argparse.Namespace) -> int:
    """before / after levels, spectrograms and the timing check"""
    out = AUDIO
    base = AUDIO / "src" / "01.48k.stereo.wav"
    raw, mix = AUDIO / "class_mix_raw.wav", AUDIO / "class_mix.wav"
    rep = {"levels": {}, "loudness": {}}
    for name, p in (("tina_original", base), ("assembled_unprocessed", raw), ("class_mix", mix)):
        if p.exists():
            rep["levels"][name] = noise_floor_db(p)
            rep["loudness"][name] = measure_loudness(p)
    png = spectrogram_png({k: v for k, v in (("Tina original", base), ("assembled", raw), ("class mix", mix)) if v.exists()}, out / "class_mix_spectrograms.png",
                          a.t0, a.t1)
    rep["spectrogram"] = png.name
    rep["word_timing"] = word_timing_check(mix, n_words=10, reference=base)
    fm = _film_mix_path()
    if fm is not None:
        rep["word_timing_film_mix"] = word_timing_check(fm[0], n_words=10, offset_fn=fm[1])
    (out / "class_mix_report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep, indent=1))
    return 0


def _film_mix_path():
    """(film mix wav, master -> film seconds function) of the written direction, or None"""
    dp = FILM / "direction.json"
    if not dp.exists():
        return None
    d = json.loads(dp.read_text(encoding="utf-8"))
    wav = AUDIO / f"{d['name']}_mix.wav"
    if not wav.exists():
        return None
    ins = d["timeline"].get("inserts", [])
    pts = sorted((i["park"] if i["kind"] == "replay" else i["core"][1], i["added_s"]) for i in ins)
    return wav, (lambda t: t + sum(a for p, a in pts if p <= t + 1e-9))


def cmd_film_mix(a: argparse.Namespace) -> int:
    r = film_mix(Path(a.direction), Path(a.class_mix) if a.class_mix else None)
    print(json.dumps(r))
    return 0


def cmd_slow_check(a: argparse.Namespace) -> int:
    dp = Path(a.direction) if a.direction else FILM / "direction.json"
    rows = slow_audio_report(dp, Path(a.mix) if a.mix else None)
    for r in rows:
        kind = "replay" if r["replay"] else f"hold {r['hold_s']} s" if r["hold_s"] is not None else f"speed {r['speed']}"
        print(f"{r['id']:28s} film {r['film'][0]:8.3f} .. {r['film'][1]:8.3f} s  {kind:10s} rms {r['rms_db']:7.1f} dB  peak {r['peak_db']:7.1f} dB  "
              f"{'room tone only' if r['room_tone_only'] else 'NOT room tone (speech?)'}")
    bad = [r for r in rows if not r["room_tone_only"]]
    print(f"{len(rows)} slowed / held / replayed segment(s), {len(bad)} above {SLOW_MAX_DB} dB")
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.film_recap_audio", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("classmix", help="build film/audio/class_mix.wav (master clock) from the phones' audio")
    c.add_argument("--no-denoise", action="store_true")
    c.add_argument("--margin-db", type=float, default=MARGIN_DB)
    c.add_argument("--base-only", action="store_true", help="the master phone only (no replacement stretches)")
    c.set_defaults(fn=cmd_classmix)
    r = sub.add_parser("report", help="before / after levels, spectrograms, word timing check")
    r.add_argument("--t0", type=float, default=60.0)
    r.add_argument("--t1", type=float, default=90.0)
    r.set_defaults(fn=cmd_report)
    f = sub.add_parser("film-mix", help="render <direction.name>_mix.wav (film clock) from a direction and the class mix")
    f.add_argument("direction")
    f.add_argument("--class-mix")
    f.set_defaults(fn=cmd_film_mix)
    k = sub.add_parser("slow-check", help="what the film mix plays in every slowed / held / replayed segment (must be room tone: exit 1 otherwise)")
    k.add_argument("direction", nargs="?")
    k.add_argument("--mix")
    k.set_defaults(fn=cmd_slow_check)
    a = ap.parse_args(argv)
    try:
        return a.fn(a)
    except AudioError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
