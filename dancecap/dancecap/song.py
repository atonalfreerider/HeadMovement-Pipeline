"""Song: the studio recording of a take's music, on the reference clock.

The phones heard the music through the room's speakers (reverb, crowd, AGC, phone codecs). With the studio track
(takes/<take>.toml [dance] song_file) the take gets a clean soundtrack and a beat grid tracked on noise-free audio.
COPYRIGHT: the studio track and every audio file derived from it stay under work/ (git-ignored) and never enter a
repo; write_private_wav() refuses any other destination.

    python -m dancecap.song align --take larissa_kadu        # song/song.json (+ song/align_detail.json)
    python -m dancecap.song soundtrack --take larissa_kadu   # song/studio_reference.wav + .json
    python -m dancecap.song beats --take larissa_kadu        # song/beats_studio.json + beats_studio_raw.json
    python -m dancecap.song compare --take larissa_kadu      # song/comparison.json + review/song_alignment.png
    python -m dancecap.song all --take larissa_kadu          # all of the above, in that order

Outputs (work/<take>/song/, git-ignored):
  song.json                 {version, take, song{title, artist, file (relative to the workspace), sha256_16, duration},
                            reference{cam, file, duration}, alignment{convention, slope, offset, anchors[{song_t, ref_t,
                            residual_ms, confidence, psr, inlier}], residual_ms_median, residual_ms_p95, cross_checks
                            {cam: residual_ms}, cross_check_detail, coarse, chroma_check, playback}, coverage{song,
                            reference, all_cameras}}
  align_detail.json         coarse slope-search curve, per-camera cross-check anchors (for the review plot)
  studio_reference.wav/.json  the studio track on the reference timeline (16-bit stereo; sample i at reference time
                            reference_start + i / sample_rate; time_to_audio = -reference_start, the capture.json
                            convention), method + timing verification
  studio_clicks_overlay.wav the soundtrack with clicks on the full-span studio grid (listening check)
  beats_studio.json         zouk grid [[t, type]] in REFERENCE seconds - the same bare format as beats.json, so it is
                            a drop-in --beats for timing/physics (their loaders expect a bare list). MEASURE-ALIGNED
                            with beats.json: its measure 0 is the studio measure whose downbeat matches beats.json's
                            first entry, so timing.load_beats' measure numbers (index // 8) are the same for both grids
                            (see measure_origin; without a phone grid: the first whole measure at reference >= 0)
  beats_studio_full.json    the same grid over the whole soundtrack span (may start before beats.json and at negative
                            reference times): measure k here = measure k - full_measure_offset in beats_studio.json
  beats_studio_raw.json     provenance (incl. measure_origin and beat_placement: where the beats sit relative to the
                            kick/bass energy and the high-frequency transients) + the grid with measure/slot in
                            reference and song time (cropped and full), beat_this beats/downbeats (song time), tempo,
                            phase checks
  comparison.json           studio grid vs the phone grid (work/<take>/beats.json): tempo, per-beat differences, drift,
                            bar phase / beat type / MEASURE NUMBER agreement, the analysis window, recommendation with
                            the adoption steps (timing --no-refine, physics, export, capture-time re-exports)
  review/song_alignment.png envelopes before/after alignment, anchor residuals, both grids over the analysis window

Accuracy: the song -> reference mapping is good to about +-1 ms absolute. That figure is set by the onset-feature
definition (independent features - another decode, STFT size and peak search - land 0.3-1 ms away), not by the anchor
scatter (residual_ms_*) or the fit's standard errors (*_se), which only measure precision and are much smaller.

Clock: reference_time = slope * song_time + offset (reference = the alignment.json reference camera's AUDIO clock,
cam06 for larissa_kadu). song_time = seconds into the gapless ffmpeg decode of the song file (an mp3's LAME encoder
delay is removed: sample 0 is the first sample of the original audio, not the container start_time).

Alignment (align_signals): onset features = half-wave rectified log-mel flux (40 mel bands 60 Hz - 7 kHz grouped
into 8 bands, 5 ms hop, ~10 ms FFT window: a 43 ms window biases noisy / reverberant phone onsets late by 3-7 ms,
see onset_features). (1) Coarse: the WHOLE studio track is cross-correlated with the whole
reference recording for every slope in [0.85, 1.15] (the studio features time-scaled by the slope; a DJ tempo change
smears a slope-1 correlation into noise), refined around the best slope; chroma (12 bins, 20 ms) is correlated at
that slope as a robustness check of the offset. (2) Local anchors: 8 s studio windows every 8 s over the overlap,
time-scaled by the current fit, are correlated with the reference features within +-0.25 s; the peak is refined to
sub-frame precision by re-warping the studio features (cubic) at fractional lags; r = mean per-band
Pearson correlation is the anchor confidence. (3) Robust affine fit:
Theil-Sen start, then least squares on anchors within max(4 robust sd, 3 ms) with r >= 0.08, iterated; anchors and fit
are recomputed once with the refined slope. Pitch check: the long-term spectra of the two recordings are compared
on a 0.01-semitone axis: a varispeed playback (resampled) shifts pitch by -12 log2(slope) semitones, a key-locked
DJ time-stretch keeps it.

Cross-checks (independent): the studio track is also aligned to other phones' own audio (their source clocks);
mapping those anchors through alignment.json's clocks must land on the direct song -> reference line. The
discrepancy (ms) is alignment.json's clock error for that camera (its anchors scatter by 10-60 ms); differences of
those errors predict the per-camera beat offsets beats.py measured (audio/quality.json), a second check.

Studio beats: beat_this with beats.py's helpers (BeatTracker, fix_beat_level instead of the analyzer's
split_measures doubling, repair_gaps, bar_phase from downbeats, kick/backbeat rotation check, zouk_grid with eighths
at the midpoints of quarter beats) on the studio track in song time, then mapped to reference time through the fit.

Soundtrack: if |slope - 1| > 1e-5 the track is time-stretched by 1/slope with pitch preserved (ffmpeg rubberband
filter - what a key-locked DJ tempo change does; the pitch check says which applies) else only shifted; sample k of
the stretched audio sits at reference offset + k / sr. The stretch is verified against the studio features (a
constant latency is measured and removed) and end-to-end against the reference recording.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import subprocess
import tomllib
import unicodedata
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from . import beats as B
from .take import PACKAGE_ROOT, WORKSPACE, Source, Take

VERSION = 1
ALIGN_SR = 24000
HOP_S = 0.005
WIN_S = 0.01  # FFT window of the alignment features (rounded up to a power of two: 256 at 24 kHz)
N_GROUPS = 8
CONVENTION = "reference_time = slope * song_time + offset"
WORK_ROOT = PACKAGE_ROOT / "work"
COPYRIGHT = ("Derived from the copyrighted studio recording supplied by the user: research use only; keep under work/ "
             "(git-ignored); never commit, upload or publish.")
DISTRIBUTION = ("Exporting this soundtrack to a capture copies it to head-movement/Assets/StreamingAssets/<capture>/"
                "audio.wav (git-ignored), and any Unity / Quest build then packs it into the player / APK: builds that "
                "contain the studio soundtrack are private research builds and must not be distributed or shared. "
                "Shareable builds use the phone audio (the export default, audio/audio_overlap.wav).")
ACCURACY_MS = 1.0
ACCURACY_NOTE = ("absolute accuracy about +-1 ms, set by the onset-feature definition (independent features - another "
                 "decode, STFT size and peak search - agree within 0.3-1.0 ms); residual_ms_* (anchor scatter) and "
                 "slope_se / offset_se (statistical precision) are much smaller and do not measure accuracy")
SONG_TIME = ("seconds into the gapless ffmpeg decode of the song file (mp3 LAME encoder delay removed; the container "
             "start_time is ignored)")


@dataclass
class SongConfig:
    slope_range: tuple[float, float] = (0.85, 1.15)
    slope_step: float = 0.0005
    anchor_window_s: float = 8.0
    anchor_step_s: float = 8.0
    anchor_search_s: float = 0.25
    min_r: float = 0.08  # short-window features correlate at r ~ 0.1-0.25 on phone audio
    inlier_k: float = 4.0
    inlier_floor_ms: float = 3.0
    cams: tuple[str, ...] | None = None  # cross-check cameras (None: every accepted camera but the reference)
    checkpoint: str = "final0"
    device: str | None = None  # beat_this: None = cuda when the GPU lock frees within gpu_wait_s, else cpu
    gpu_wait_s: float = 600.0
    min_bpm: float = 60.0
    rephase_margin: float = 1.0
    out_sr: int = 48000
    stretch_tol: float = 1e-5  # |slope - 1| below this (< 2 ms over 190 s): shift only, no stretch
    rubberband: str = "transients=crisp:detector=compound:phase=laminar:window=standard:channels=together"
    latency_tol_ms: float = 0.5  # stretch latency above this is removed
    window: tuple[float, float] | None = None  # analysis window (None: the take's)


# ------------------------------------------------------------------------------------------------ paths
def read_toml(path: str | Path) -> dict:
    """TOML is UTF-8 by spec. Path.read_text() without an encoding uses the locale (cp1252 on this Windows box), which
    turns 'Voce' with a circumflex into mojibake ('VocÃª'); take.Take.load reads its toml that way."""
    return tomllib.loads(Path(path).read_text(encoding="utf-8"))


def demojibake(s: str) -> str:
    """Undo UTF-8 text that was decoded as cp1252/latin-1 ('VocÃª' -> 'Você'); any other string is returned as is."""
    for enc in ("cp1252", "latin-1"):
        try:
            return s.encode(enc).decode("utf-8")
        except UnicodeError:
            continue
    return s


def name_key(name: str) -> str:
    """Comparison key for a file name: mojibake repaired, Unicode NFC (NTFS keeps NFC and NFD names distinct), casefold."""
    return unicodedata.normalize("NFC", demojibake(unicodedata.normalize("NFC", name))).casefold()


def resolve_path(p: str | Path) -> Path:
    """An existing path for p, tolerant (component by component) of NFC/NFD normalisation and of UTF-8 mojibake.
    Raises FileNotFoundError when there is no unique match."""
    p = Path(p)
    if p.exists():
        return p
    if p.parent == p:
        raise FileNotFoundError(p)
    parent = resolve_path(p.parent)
    key = name_key(p.name)
    hits = [c for c in parent.iterdir() if name_key(c.name) == key]
    if len(hits) != 1:
        raise FileNotFoundError(f"{p}: {'no' if not hits else 'ambiguous'} match in {parent} for {key!r}")
    return hits[0]


def rel_to_workspace(p: Path) -> str:
    try:
        return Path(os.path.relpath(p, WORKSPACE)).as_posix()
    except ValueError:  # another drive
        return Path(p).as_posix()


def sha256_16(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


@dataclass
class SongInfo:
    title: str
    artist: str
    path: Path

    @property
    def rel(self) -> str:
        return rel_to_workspace(self.path)


def take_toml(take: str | Path) -> Path:
    p = Path(take)
    return p if p.suffix == ".toml" else PACKAGE_ROOT / "takes" / f"{take}.toml"


def song_info(take: str | Path) -> SongInfo:
    """[dance] song / artist / song_file of a take (toml read as UTF-8; song_file relative to the toml)."""
    tp = take_toml(take)
    d = read_toml(tp).get("dance") or {}
    if not d.get("song_file"):
        raise SystemExit(f"{tp}: no [dance] song_file - the studio track is needed for dancecap.song")
    return SongInfo(d.get("song", ""), d.get("artist", ""), resolve_path((tp.parent / d["song_file"]).resolve()))


def assert_private(path: str | Path) -> Path:
    """Copyright guard: song-derived audio may only be written under work/ and only where git ignores it."""
    path = Path(path).resolve()
    root = WORK_ROOT.resolve()
    if root not in path.parents:
        raise PermissionError(f"refusing to write song-derived audio outside {root}: {path}")
    with contextlib.suppress(FileNotFoundError):
        r = subprocess.run(["git", "check-ignore", "-q", str(path)], cwd=PACKAGE_ROOT, capture_output=True)
        if r.returncode == 1:
            raise PermissionError(f"{path} is not git-ignored; refusing to write song-derived audio there")
    return path


def write_private_wav(path: Path, y: np.ndarray, sr: int) -> Path:
    path = assert_private(path)
    B.write_wav(path, y, sr)
    return path


def load_mono(path: str | Path, sr: int) -> np.ndarray:
    return B.decode_audio(resolve_path(path), sr)[0]


def _np(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return o.as_posix()
    raise TypeError(type(o))


def write_json(path: Path, obj, indent: int | None = 1) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=indent, default=_np), encoding="utf-8")
    return path


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------------------ features
def frame_dt(sr: int, hop_s: float = HOP_S) -> float:
    return round(hop_s * sr) / sr


def onset_features(x: np.ndarray, sr: int, hop_s: float = HOP_S, groups: int = N_GROUPS,
                   win_s: float = WIN_S) -> np.ndarray:
    """(groups, T) half-wave rectified log-mel flux: 40 mel bands 60 Hz - 7 kHz (beats.logmel_envelope's bands and
    60 dB floor) with a SHORT ~10 ms FFT window, adjacent bands averaged into `groups`.

    Why short: with log compression an onset registers when its energy in the window rises above the floor, so with a
    long window (43 ms, beats.logmel_envelope at 24 kHz) a noisy / reverberant phone registers onsets later than the
    clean studio master. Synthetic test: the 43 ms window biased the offset by +2.7 ms at 15 dB SNR and +6.9 ms at
    DRR -5 dB / SNR 10 dB; the 10.7 ms window stays within 0.7 ms. On larissa_kadu the switch moved the fit 6.0 ms
    earlier and cut the anchor residual p95 from 2.8 ms to 0.7 ms."""
    import librosa
    hop = int(round(hop_s * sr))
    n_fft = int(2 ** np.ceil(np.log2(win_s * sr)))
    m = librosa.feature.melspectrogram(y=np.asarray(x, np.float32), sr=sr, n_fft=n_fft, hop_length=hop, n_mels=40,
                                       fmin=60, fmax=min(7000, sr / 2 - 100))
    L = np.log10(np.maximum(m, 1e-6 * m.max() + 1e-20))
    F = np.maximum(0.0, np.diff(L, axis=1, prepend=L[:, :1]))
    nb = (L.shape[0] // groups) * groups
    return F[:nb].reshape(groups, nb // groups, -1).mean(1).astype(np.float32)


def _z(a: np.ndarray, axis: int = -1) -> np.ndarray:
    a = np.asarray(a, np.float64)
    return (a - a.mean(axis, keepdims=True)) / (a.std(axis, keepdims=True) + 1e-12)


def warp(F: np.ndarray, dt: float, slope: float, offset: float, t_ref: np.ndarray) -> np.ndarray:
    """Studio features sampled at reference times t_ref (song time (t - offset) / slope); 0 outside the song."""
    ts = np.arange(F.shape[1]) * dt
    u = (np.asarray(t_ref, float) - offset) / slope
    return np.stack([np.interp(u, ts, f, left=0.0, right=0.0) for f in F])


def _parabola(y: np.ndarray, k: int) -> float:
    """Sub-sample offset (-0.5..0.5) of the peak at k from a parabola through k-1, k, k+1."""
    if 0 < k < len(y) - 1:
        a, b, c = float(y[k - 1]), float(y[k]), float(y[k + 1])
        den = a - 2 * b + c
        if den < 0:
            return float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))
    return 0.0


def _decimate2(F: np.ndarray) -> np.ndarray:
    n = F.shape[1] // 2 * 2
    return F[:, :n].reshape(F.shape[0], -1, 2).mean(2)


# ------------------------------------------------------------------------------------------------ coarse
def slope_search(Fs: np.ndarray, Fr: np.ndarray, dt: float, cfg: SongConfig | None = None) -> dict:
    """Whole-track cross-correlation of the studio features, time-scaled by every slope on a grid, with the reference
    features (10 ms frames). Returns the best slope / offset, r (~ Pearson over the full overlap) and the curve."""
    from scipy import fft
    cfg = cfg or SongConfig()
    S, R = _decimate2(Fs), _decimate2(Fr)
    d2 = 2 * dt
    Ts = S.shape[1] * d2
    nb = S.shape[0]
    n = fft.next_fast_len(int(np.ceil(Ts * cfg.slope_range[1] / d2)) + 2 + R.shape[1])
    FR = fft.rfft(_z(R), n)

    def corr(s: float) -> tuple[np.ndarray, int]:
        m = int(np.floor(Ts * s / d2))
        G = _z(warp(S, d2, s, 0.0, np.arange(m) * d2))
        return fft.irfft((FR * np.conj(fft.rfft(G, n))).sum(0), n) / (nb * min(m, R.shape[1])), m

    def lag_of(i: int) -> int:
        return i if i <= n // 2 else i - n

    def evaluate(slopes):
        pk, off = [], []
        for s in slopes:
            c, _ = corr(s)
            i = int(np.argmax(c))
            pk.append(float(c[i]))
            off.append(lag_of(i) * d2)
        return np.asarray(pk), np.asarray(off)

    lo, hi = cfg.slope_range
    slopes = np.arange(lo, hi + cfg.slope_step / 2, cfg.slope_step)
    pk, off = evaluate(slopes)
    b = int(np.argmax(pk))
    fine = slopes[b] + np.arange(-10, 11) * cfg.slope_step / 10
    fpk, _ = evaluate(fine)
    k = int(np.argmax(fpk))
    s_best = float(fine[k] + _parabola(fpk, k) * cfg.slope_step / 10)
    c, _ = corr(s_best)
    i = int(np.argmax(c))
    off_best = (lag_of(i) + _parabola(c, i)) * d2
    lags = np.array([lag_of(j) for j in range(n)]) * d2
    side = np.abs(lags - lag_of(i) * d2) > 0.1
    j2 = int(np.argmax(np.where(side, c, -np.inf)))
    c1, _ = corr(1.0)
    i1 = int(np.argmax(c1))
    return {"slope": s_best, "offset": float(off_best), "r": float(c[i]),
            "second_peak_r": float(c[j2]), "second_peak_lag_s": round(float(lags[j2] - lag_of(i) * d2), 3),
            "r_at_slope_1": float(c1[i1]), "offset_at_slope_1": float(lag_of(i1) * d2),
            "curve": {"slopes": slopes, "r": pk, "offset": off}, "frame_s": d2,
            "method": f"whole-track onset-feature xcorr over slopes {lo}-{hi} step {cfg.slope_step} "
                      f"(10 ms frames), refined at step {cfg.slope_step / 10:g} + parabola"}


def chroma_check(xs: np.ndarray, xr: np.ndarray, sr: int, slope: float, offset: float, hop_s: float = 0.02) -> dict:
    """Offset from 12-bin chroma (harmony, insensitive to timbre / room) at the fitted slope, and at slope 1."""
    import librosa
    from scipy import fft
    hop = int(round(hop_s * sr))
    dtc = hop / sr
    Cs = librosa.feature.chroma_stft(y=xs.astype(np.float32), sr=sr, hop_length=hop)
    Cr = _z(librosa.feature.chroma_stft(y=xr.astype(np.float32), sr=sr, hop_length=hop))
    out = {}
    for name, s in (("fit_slope", slope), ("slope_1", 1.0)):
        m = int(np.floor(Cs.shape[1] * s))
        G = _z(warp(Cs, dtc, s, 0.0, np.arange(m) * dtc))
        n = fft.next_fast_len(m + Cr.shape[1])
        c = fft.irfft((fft.rfft(Cr, n) * np.conj(fft.rfft(G, n))).sum(0), n) / (12 * min(m, Cr.shape[1]))
        i = int(np.argmax(c))
        lag = (i if i <= n // 2 else i - n) + _parabola(c, i)
        out[name] = {"offset": round(lag * dtc, 4), "r": round(float(c[i]), 4)}
    d = out["fit_slope"]["offset"] - offset
    out.update(offset_diff_ms=round(1000 * d, 1), agrees=bool(abs(d) < 0.1), frame_s=dtc,
               method="12-bin chroma_stft (20 ms), whole-track xcorr at the fitted slope; agrees = within 100 ms")
    return out


def pitch_check(xs: np.ndarray, xr: np.ndarray, sr: int, slope: float, offset: float, song_span: tuple[float, float],
                max_st: float = 1.0) -> dict:
    """Pitch shift (semitones, reference above song) between the recordings over song_span: long-term spectra on a
    0.01-semitone axis 110-1760 Hz, spectral envelope removed, best shift. Varispeed playback would give
    -12 log2(slope); a key-locked time-stretch gives 0."""
    from scipy import signal
    a0, a1 = song_span
    seg_s = xs[int(a0 * sr):int(a1 * sr)]
    seg_r = xr[max(0, int((slope * a0 + offset) * sr)):int((slope * a1 + offset) * sr)]
    f, Ps = signal.welch(seg_s, sr, nperseg=32768, noverlap=16384)
    _, Pr = signal.welch(seg_r, sr, nperseg=32768, noverlap=16384)
    fq = 110 * 2 ** (np.arange(0, 48, 0.01) / 12)

    def prep(P):
        v = np.log(np.interp(fq, f, P) + 1e-20)
        return (v - np.convolve(v, np.ones(101) / 101, "same"))[200:-200]

    a, b = prep(Ps), prep(Pr)
    K = int(round(max_st / 0.01))
    sh = np.arange(-K, K + 1)
    c = np.array([np.corrcoef(a[max(0, -k):len(a) - max(0, k)], b[max(0, k):len(b) - max(0, -k)])[0, 1] for k in sh])
    i = int(np.argmax(c))
    best = (sh[i] + _parabola(c, i)) * 0.01
    vs = -12 * np.log2(slope)
    iv = int(np.clip(round(vs / 0.01) + K, 0, len(c) - 1))
    if abs(best) < 0.25 * abs(vs) and c[i] > c[iv] + 0.1:
        interp = "key-locked time-stretch: pitch preserved, tempo changed (DJ 'key lock' / 'master tempo')"
    elif abs(best - vs) < 0.25 * abs(vs):
        interp = "varispeed: pitch follows tempo (resampled playback)"
    else:
        interp = "unclear"
    return {"semitones": round(float(best), 3), "r": round(float(c[i]), 3), "r_at_0": round(float(c[K]), 3),
            "varispeed_semitones": round(float(vs), 3), "r_at_varispeed": round(float(c[iv]), 3),
            "song_span": [round(a0, 2), round(a1, 2)], "interpretation": interp}


# ------------------------------------------------------------------------------------------------ anchors + fit
def anchor_centers(slope: float, offset: float, dur_song: float, dur_ref: float, cfg: SongConfig) -> np.ndarray:
    """Song-time centres of anchor windows spread evenly over the song span the reference covers."""
    c0 = max(0.0, -offset / slope)
    c1 = min(dur_song, (dur_ref - offset) / slope)
    pad = cfg.anchor_window_s / 2 / slope + cfg.anchor_search_s / slope + 0.05
    a, b = c0 + pad, c1 - pad
    if b < a:
        return np.zeros(0)
    n = int(np.floor((b - a) / cfg.anchor_step_s)) + 1
    return a + ((b - a) - (n - 1) * cfg.anchor_step_s) / 2 + cfg.anchor_step_s * np.arange(n)


def warp_cubic(F: np.ndarray, dt: float, slope: float, offset: float, t_ref: np.ndarray) -> np.ndarray:
    """warp() with cubic-spline interpolation (less smoothing at fractional frame positions)."""
    from scipy.ndimage import map_coordinates
    u = (np.asarray(t_ref, float) - offset) / slope / dt
    return np.stack([map_coordinates(np.asarray(f, np.float64), [u], order=3, mode="constant", cval=0.0)
                     for f in F])


def fine_lag(Fs: np.ndarray, Fr: np.ndarray, dt: float, slope: float, offset: float, i0: int, nW: int, k: int,
             step: float = 0.05) -> float:
    """Sub-frame lag (frames) around integer lag k: the studio features are re-warped (cubic) at fractional lags
    k + e, e in [-1, 1] by `step`, against the fixed reference window at lag k; parabola on that fine curve.
    A parabola through the 3 integer-lag correlations is biased by up to ~1 ms at half-frame shifts (the 10 ms-window
    correlation peak is only ~2 frames wide); this is not."""
    R = _z(Fr[:, i0 + k:i0 + k + nW])
    t_ref = (i0 + k + np.arange(nW)) * dt
    eps = np.arange(-1.0, 1.0 + step / 2, step)
    r = np.array([np.mean(_z(warp_cubic(Fs, dt, slope, offset + (k + e) * dt, t_ref)) * R) for e in eps])
    j = int(np.argmax(r))
    return k + float(eps[j]) + _parabola(r, j) * step


def local_anchors(Fs: np.ndarray, Fr: np.ndarray, dt: float, slope: float, offset: float, centers,
                  window_s: float, search_s: float, refine: bool = True) -> list[dict]:
    """For each song-time centre: the studio window (time-scaled by slope) correlated with the reference features over
    +-search_s around the predicted position; r = mean per-band Pearson correlation; sub-frame peak by fine_lag
    (refine=True) or a 3-point parabola.
    psr = peak r / best r more than 50 ms away; edge = peak on the search boundary (unreliable)."""
    from numpy.lib.stride_tricks import sliding_window_view
    nW = int(round(window_s / dt))
    nS = int(round(search_s / dt))
    lags = np.arange(-nS, nS + 1)
    Ts = Fs.shape[1] * dt
    out = []
    for c in np.asarray(centers, float):
        i0 = int(round((slope * c + offset - window_s / 2) / dt))
        if i0 - nS < 0 or i0 + nW + nS > Fr.shape[1]:
            continue
        t_ref = (i0 + np.arange(nW)) * dt
        u = (t_ref - offset) / slope
        if u[0] < 0 or u[-1] > Ts:
            continue
        G = _z(warp(Fs, dt, slope, offset, t_ref))
        Wv = sliding_window_view(Fr[:, i0 - nS:i0 + nW + nS].astype(np.float64), nW, axis=1)  # (bands, lags, nW)
        Wz = (Wv - Wv.mean(-1, keepdims=True)) / (Wv.std(-1, keepdims=True) + 1e-12)
        r = (Wz * G[:, None, :]).mean(-1).mean(0)
        k = int(np.argmax(r))
        edge = bool(k in (0, len(r) - 1))
        lag = (fine_lag(Fs, Fr, dt, slope, offset, i0, nW, int(lags[k])) if refine and not edge
               else lags[k] + _parabola(r, k)) * dt
        side = np.abs(lags - lags[k]) * dt > 0.05
        psr = float(r[k] / max(float(r[side].max()), 1e-6)) if side.any() else float("inf")
        tc = (i0 + (nW - 1) / 2) * dt
        out.append({"song_t": (tc - offset) / slope, "ref_t": tc + lag, "lag_ms": 1000 * lag, "r": float(r[k]),
                    "psr": psr, "edge": edge})
    return out


def robust_affine(x, y, ok=None, k: float = 4.0, floor_s: float = 0.003, iters: int = 8) -> dict:
    """y = slope * x + offset: Theil-Sen start, then least squares on points within max(k robust sd, floor_s) of the
    line (and ok), iterated to a fixed point. Returns slope, offset, inlier mask, residuals, standard errors."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.ones(len(x), bool) if ok is None else np.asarray(ok, bool)
    idx = np.nonzero(ok)[0]
    if len(idx) < 2:
        raise ValueError("robust_affine: fewer than 2 usable points")
    i, j = np.triu_indices(len(idx), 1)
    dx = x[idx][j] - x[idx][i]
    good = np.abs(dx) > 1e-9
    a = float(np.median((y[idx][j] - y[idx][i])[good] / dx[good])) if good.any() else 1.0
    b = float(np.median(y[idx] - a * x[idx]))
    inl = ok.copy()
    thr = floor_s
    for _ in range(iters):
        r = y - (a * x + b)
        rr = r[inl]
        sd = 1.4826 * float(np.median(np.abs(rr - np.median(rr))))
        thr = max(k * sd, floor_s)
        new = ok & (np.abs(r) <= thr)
        if new.sum() < 2:
            break
        A = np.stack([x[new], np.ones(int(new.sum()))], 1)
        a, b = (float(v) for v in np.linalg.lstsq(A, y[new], rcond=None)[0])
        if np.array_equal(new, inl):
            break
        inl = new
    r = y - (a * x + b)
    n = int(inl.sum())
    xm = float(x[inl].mean())
    sxx = float(np.sum((x[inl] - xm) ** 2)) or 1e-12
    s2 = float(np.sum(r[inl] ** 2)) / max(1, n - 2)
    return {"slope": a, "offset": b, "inlier": inl, "residual": r, "threshold_s": thr, "n": n,
            "slope_se": float(np.sqrt(s2 / sxx)), "offset_se": float(np.sqrt(s2 * (1 / max(n, 1) + xm ** 2 / sxx))),
            "x_mean": xm, "sxx": sxx, "s2": s2}


def prediction_sd(fit: dict, x) -> np.ndarray:
    """1-sigma error of the fitted line at x (from the inlier scatter)."""
    x = np.asarray(x, float)
    return np.sqrt(fit["s2"] * (1 / max(fit["n"], 1) + (x - fit["x_mean"]) ** 2 / fit["sxx"]))


def _fit_anchors(anchors: list[dict], cfg: SongConfig) -> dict:
    x = np.array([a["song_t"] for a in anchors])
    y = np.array([a["ref_t"] for a in anchors])
    ok = np.array([a["r"] >= cfg.min_r and not a["edge"] for a in anchors])
    return robust_affine(x, y, ok, cfg.inlier_k, cfg.inlier_floor_ms / 1000)


def align_signals(xs: np.ndarray, xr: np.ndarray, sr: int, cfg: SongConfig | None = None,
                  Fs: np.ndarray | None = None, Fr: np.ndarray | None = None) -> dict:
    """Affine map song time -> reference time between two mono signals at the same rate (see the module doc)."""
    cfg = cfg or SongConfig()
    dt = frame_dt(sr)
    Fs = onset_features(xs, sr) if Fs is None else Fs
    Fr = onset_features(xr, sr) if Fr is None else Fr
    dur_s, dur_r = len(xs) / sr, len(xr) / sr
    coarse = slope_search(Fs, Fr, dt, cfg)
    s, o = coarse["slope"], coarse["offset"]
    for _ in range(2):  # anchors with the coarse fit, then again with the refined slope
        anchors = local_anchors(Fs, Fr, dt, s, o, anchor_centers(s, o, dur_s, dur_r, cfg), cfg.anchor_window_s,
                                cfg.anchor_search_s)
        if len(anchors) < 2:
            raise RuntimeError(f"only {len(anchors)} anchor windows fit in the overlap (slope {s:.5f}, offset {o:.3f})")
        fit = _fit_anchors(anchors, cfg)
        s, o = fit["slope"], fit["offset"]
    res_ms = 1000 * fit["residual"]
    inl = fit["inlier"]
    for a, r, f in zip(anchors, res_ms, inl):
        a.update(residual_ms=float(r), inlier=bool(f))
    song_cov = [max(0.0, -o / s), min(dur_s, (dur_r - o) / s)]
    span_sd = prediction_sd(fit, np.linspace(*song_cov, 50))
    ra = np.abs(res_ms[inl])
    return {"slope": s, "offset": o, "anchors": anchors, "fit": fit, "coarse": coarse, "dt": dt,
            "residual_ms_median": float(np.median(ra)), "residual_ms_p95": float(np.percentile(ra, 95)),
            "residual_ms_max": float(ra.max()), "n_anchors": len(anchors), "n_inliers": int(inl.sum()),
            "fit_sd_ms_max_over_coverage": float(1000 * span_sd.max()),
            "coverage": {"song": song_cov, "reference": [s * song_cov[0] + o, s * song_cov[1] + o]},
            "durations": {"song": dur_s, "reference": dur_r}}


def camera_check(Fs: np.ndarray, dt: float, dur_song: float, src: Source, Fc: np.ndarray, dur_cam: float,
                 slope: float, offset: float, cfg: SongConfig) -> dict:
    """Studio track vs one phone's OWN audio (source clock), mapped through that phone's alignment.json clock.
    residual_ms = median over inlier anchors of [alignment.json clock(cam anchor) - direct song->reference line]:
    the alignment.json clock error for this camera (independent of the direct fit)."""
    a0, b0 = slope / src.slope, (offset - src.offset) / src.slope  # predicted song -> camera source time
    anchors = local_anchors(Fs, Fc, dt, a0, b0, anchor_centers(a0, b0, dur_song, dur_cam, cfg), cfg.anchor_window_s,
                            cfg.anchor_search_s)
    if len(anchors) < 3:
        return {"n_anchors": len(anchors), "residual_ms": None, "note": "fewer than 3 anchors"}
    fit = _fit_anchors(anchors, cfg)
    x = np.array([a["song_t"] for a in anchors])
    y = np.array([a["ref_t"] for a in anchors])  # camera SOURCE seconds
    inl = fit["inlier"]
    d = 1000 * ((src.slope * y + src.offset) - (slope * x + offset))
    med = float(np.median(d[inl]))
    sc = slope / fit["slope"]
    oc = offset - slope * fit["offset"] / fit["slope"]
    fr = np.abs(1000 * fit["residual"][inl])
    line = np.polyfit(x[inl], d[inl], 1) if inl.sum() >= 2 else np.array([0.0, med])
    return {"residual_ms": round(med, 2), "residual_ms_p95": round(float(np.percentile(np.abs(d[inl]), 95)), 2),
            "discrepancy_line_ms": {"at_song_0": round(float(line[1]), 3), "per_song_s": round(float(line[0]), 5)},
            "residual_ms_spread": round(1.4826 * float(np.median(np.abs(d[inl] - med))), 2),
            "drift_ms_per_min": round(60e3 * (src.slope * fit["slope"] - slope), 3),
            "n_anchors": len(anchors), "n_inliers": int(inl.sum()),
            "fit_residual_ms_median": round(float(np.median(fr)), 2),
            "fit_residual_ms_p95": round(float(np.percentile(fr, 95)), 2),
            "song_span": [round(float(x.min()), 2), round(float(x.max()), 2)],
            "studio_derived_clock": {"slope": sc, "offset": oc},
            "alignment_json_clock": {"slope": src.slope, "offset": src.offset},
            "anchors": [[round(float(a), 4), round(float(b), 5), round(float(r["r"]), 3), round(float(dd), 2), bool(f)]
                        for a, b, r, dd, f in zip(x, y, anchors, d, inl)]}


# ------------------------------------------------------------------------------------------------ session
class Session:
    """A take + its song + decoded audio / features cached across stages."""

    def __init__(self, take: str, cfg: SongConfig | None = None):
        self.cfg = cfg or SongConfig()
        self.take = Take.load(take)
        self.info = song_info(take)
        self.out = self.take.work / "song"
        self.review = self.take.work / "review"
        self.alignment = read_json((PACKAGE_ROOT / "takes" / self.take.cfg["alignment"]).resolve())
        ref_name = self.alignment["reference"]
        self.ref = next(s for s in self.take.sources if s.path.name == ref_name)
        self.durations = {r["id"]: float(r["duration"]) for r in self.alignment["sources"]}
        self.window = tuple(self.cfg.window or self.take.window)
        self._cache: dict = {}

    def _get(self, key, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    def song_x(self, sr: int = ALIGN_SR) -> np.ndarray:
        return self._get(("song", sr), lambda: load_mono(self.info.path, sr))

    def cam_x(self, cam: str, sr: int = ALIGN_SR) -> np.ndarray:
        return self._get((cam, sr), lambda: load_mono(self.take.source(cam).path, sr))

    def song_F(self) -> np.ndarray:
        return self._get(("song", "F"), lambda: onset_features(self.song_x(), ALIGN_SR))

    def cam_F(self, cam: str) -> np.ndarray:
        return self._get((cam, "F"), lambda: onset_features(self.cam_x(cam), ALIGN_SR))

    def camera_span_union(self) -> tuple[float, float]:
        sp = [(s.to_reference(0.0), s.to_reference(self.durations[s.id])) for s in self.take.sources]
        return min(a for a, _ in sp), max(b for _, b in sp)

    def soundtrack_span(self, slope: float, offset: float) -> tuple[float, float]:
        """Reference interval where the song plays inside any camera's recording."""
        u0, u1 = self.camera_span_union()
        dur = len(self.song_x()) / ALIGN_SR
        return max(u0, offset), min(u1, slope * dur + offset)

    def song_json(self) -> dict:
        p = self.out / "song.json"
        if not p.exists():
            raise SystemExit(f"{p} missing: run 'python -m dancecap.song align --take {self.take.name}' first")
        return read_json(p)


# ------------------------------------------------------------------------------------------------ align stage
def run_align(sess: Session) -> dict:
    cfg = sess.cfg
    xs, xr = sess.song_x(), sess.cam_x(sess.ref.id)
    print(f"[song] {sess.info.title} - {sess.info.artist}: {len(xs) / ALIGN_SR:.2f} s studio vs cam{sess.ref.id} "
          f"{len(xr) / ALIGN_SR:.2f} s")
    res = align_signals(xs, xr, ALIGN_SR, cfg, sess.song_F(), sess.cam_F(sess.ref.id))
    s, o = res["slope"], res["offset"]
    co = res["coarse"]
    print(f"[song] coarse: slope {co['slope']:.6f} offset {co['offset']:.3f} s r {co['r']:.3f} "
          f"(slope 1: r {co['r_at_slope_1']:.3f}); fit: slope {s:.7f} offset {o:.4f} s, anchors {res['n_inliers']}/"
          f"{res['n_anchors']}, residual median {res['residual_ms_median']:.2f} ms p95 {res['residual_ms_p95']:.2f} ms")
    cov = res["coverage"]["song"]
    mid = (cov[0] + 0.2 * (cov[1] - cov[0]), cov[1] - 0.2 * (cov[1] - cov[0]))
    chroma = chroma_check(xs, xr, ALIGN_SR, s, o)
    pitch = pitch_check(xs, xr, ALIGN_SR, s, o, mid)
    print(f"[song] chroma offset {chroma['fit_slope']['offset']:.3f} s ({chroma['offset_diff_ms']:+.0f} ms); pitch "
          f"{pitch['semitones']:+.3f} st (varispeed would be {pitch['varispeed_semitones']:+.3f}): "
          f"{pitch['interpretation']}")

    # ---- cross-checks: the studio track against other phones' own audio
    cams = list(cfg.cams) if cfg.cams else [c.id for c in sess.take.sources if c.id != sess.ref.id]
    checks, dt = {}, res["dt"]
    for c in cams:
        src = sess.take.source(c)
        chk = camera_check(sess.song_F(), dt, len(xs) / ALIGN_SR, src, sess.cam_F(c), len(sess.cam_x(c)) / ALIGN_SR,
                           s, o, cfg)
        checks[c] = chk
        print(f"[song] cross-check cam{c}: alignment.json clock - studio line = {chk.get('residual_ms')} ms "
              f"(p95 {chk.get('residual_ms_p95')}, drift {chk.get('drift_ms_per_min')} ms/min, "
              f"{chk.get('n_inliers')}/{chk.get('n_anchors')} anchors, fit residual {chk.get('fit_residual_ms_median')} ms)")
    beats_check = beats_offset_check(sess.take.work, checks, sess.ref.id, s, o)
    if beats_check:
        print(f"[song] beats.py per-phone beat offsets vs the studio-predicted clock errors: median |diff| "
              f"{beats_check['median_abs_diff_ms']} ms " +
              ", ".join(f"cam{c} {r['measured_ms']:+.1f}/{r['predicted_ms']:+.1f}" for c, r in beats_check["rows"].items()))

    union = sess.camera_span_union()
    dur_s = len(xs) / ALIGN_SR
    all_song = [max(0.0, (union[0] - o) / s), min(dur_s, (union[1] - o) / s)]
    fit = res["fit"]
    song = {"title": sess.info.title, "artist": sess.info.artist, "file": sess.info.rel,
            "file_relative_to": "workspace root (HeadMovement/)", "sha256_16": sha256_16(sess.info.path),
            "duration": round(dur_s, 6), "song_time": SONG_TIME, "copyright": COPYRIGHT}
    out = {
        "version": VERSION, "take": sess.take.name, "song": song,
        "reference": {"cam": sess.ref.id, "file": sess.ref.path.name,
                      "duration_audio": round(len(xr) / ALIGN_SR, 6),
                      "clock": "reference camera AUDIO clock (alignment.json reference)"},
        "alignment": {
            "convention": CONVENTION, "slope": s, "offset": o,
            "accuracy_ms": ACCURACY_MS, "accuracy_note": ACCURACY_NOTE,
            "slope_se": fit["slope_se"], "offset_se": fit["offset_se"],
            "precision_note": "slope_se / offset_se / fit_sd_ms_* are statistical precision of the affine fit from "
                              "the anchor scatter, not absolute accuracy (see accuracy_ms)",
            "tempo_factor": 1.0 / s, "slope_minus_1_ppm": round(1e6 * (s - 1.0), 1),
            "fit_sd_ms_max_over_coverage": round(res["fit_sd_ms_max_over_coverage"], 3),
            "anchors": [{"song_t": round(a["song_t"], 6), "ref_t": round(a["ref_t"], 6),
                         "residual_ms": round(a["residual_ms"], 3), "confidence": round(a["r"], 4),
                         "psr": round(a["psr"], 3), "inlier": a["inlier"]} for a in res["anchors"]],
            "residual_ms_median": round(res["residual_ms_median"], 3),
            "residual_ms_p95": round(res["residual_ms_p95"], 3), "residual_ms_max": round(res["residual_ms_max"], 3),
            "n_anchors": res["n_anchors"], "n_inliers": res["n_inliers"],
            "cross_checks": {c: v.get("residual_ms") for c, v in checks.items()},
            "cross_check_detail": {c: {k: v for k, v in d.items() if k != "anchors"} for c, d in checks.items()},
            "cross_check_note": "residual_ms = median over anchors of (studio aligned to that phone's own audio, mapped "
                                "through its alignment.json clock) minus (direct song->reference line): the "
                                "alignment.json clock error of that phone. fit_residual_ms_* = precision of the "
                                "studio->phone fit itself. studio_derived_clock = the phone's clock re-derived through "
                                "the studio track (reference = slope * source + offset).",
            "beats_py_consistency": beats_check,
            "coarse": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in co.items() if k != "curve"},
            "chroma_check": chroma, "playback": {"tempo_factor": 1.0 / s,
                                                 "tempo_change_percent": round(100 * (1 / s - 1), 3), **pitch},
            "method": {"features": "half-wave rectified log-mel flux, 40 bands 60 Hz-7 kHz in 8 groups, "
                                   f"{1000 * res['dt']:.0f} ms hop, {int(2 ** np.ceil(np.log2(WIN_S * ALIGN_SR)))}-sample "
                                   f"FFT window, {ALIGN_SR} Hz mono (short window: unbiased by phone noise/reverb)",
                       "anchors": f"{cfg.anchor_window_s:g} s windows every {cfg.anchor_step_s:g} s, search "
                                  f"+-{cfg.anchor_search_s:g} s, parabolic sub-frame peak; confidence = mean per-band "
                                  "Pearson r",
                       "fit": f"Theil-Sen start, least squares on anchors within max({cfg.inlier_k:g} robust sd, "
                              f"{cfg.inlier_floor_ms:g} ms) with r >= {cfg.min_r:g}; two passes"},
        },
        "coverage": {"song": [round(v, 4) for v in res["coverage"]["song"]],
                     "reference": [round(v, 4) for v in res["coverage"]["reference"]],
                     "anchors_song": [round(min(a["song_t"] for a in res["anchors"]) - cfg.anchor_window_s / 2, 3),
                                      round(max(a["song_t"] for a in res["anchors"]) + cfg.anchor_window_s / 2, 3)],
                     "all_cameras": {"reference": [round(max(union[0], o), 4),
                                                   round(min(union[1], s * dur_s + o), 4)],
                                     "song": [round(v, 4) for v in all_song]},
                     "note": "song/reference = the span of the reference recording (cam audio); all_cameras = where "
                             "the song plays inside any camera's recording (the soundtrack span)"},
    }
    write_json(sess.out / "song.json", out)
    write_json(sess.out / "align_detail.json", {
        "coarse_curve": {k: np.round(v, 6) for k, v in co["curve"].items()},
        "cross_check_anchors_columns": ["song_t", "cam_source_t", "r", "discrepancy_ms", "inlier"],
        "cross_check_anchors": {c: d.get("anchors", []) for c, d in checks.items()}}, indent=None)
    print(f"[song] wrote {sess.out / 'song.json'}")
    return out


def beats_offset_check(work: Path, checks: dict, ref_cam: str, slope: float, offset: float) -> dict | None:
    """beats.py measured each phone's median beat offset vs its source phone on the reference clock over the
    all-camera overlap (audio/quality.json). Those offsets are (mostly) alignment.json clock errors, so they should
    equal the differences of the cross-check discrepancies evaluated at the overlap centre:
    predicted(c) = e_c - e_source with e_ref = 0."""
    q = work / "audio" / "quality.json"
    if not q.exists():
        return None
    qq = read_json(q)
    src, offs = qq.get("source_cam"), qq.get("beat_offsets_vs_source") or {}
    ov = qq.get("overlap")
    u = ((0.5 * (ov[0] + ov[1]) if ov else 0.0) - offset) / slope  # song time at the overlap centre
    e = {}
    for c, v in checks.items():
        ln = v.get("discrepancy_line_ms")
        if ln is not None:
            e[c] = ln["at_song_0"] + ln["per_song_s"] * u if ov else v["residual_ms"]
    e[ref_cam] = 0.0
    if src not in e:
        return None
    rows = {c: {"measured_ms": offs[c]["median_ms"], "predicted_ms": round(e[c] - e[src], 2),
                "diff_ms": round(offs[c]["median_ms"] - (e[c] - e[src]), 2)} for c in offs if c in e}
    diffs = [abs(r["diff_ms"]) for r in rows.values()]
    return {"source_cam": src, "overlap": ov, "rows": rows,
            "median_abs_diff_ms": round(float(np.median(diffs)), 2) if diffs else None,
            "note": "measured = beats.py median beat offset (cam - source cam) on the reference clock over the overlap; "
                    "predicted = cross-check discrepancy(cam) - discrepancy(source), both at the overlap centre "
                    "(linear in time: some alignment.json clocks drift). Agreement within a few ms means the studio "
                    "alignment and the phone-phone beat tracking tell the same story; the remainder is beat-tracking "
                    "noise, acoustic path differences and beat_this's own level-dependent onset timing on noisy "
                    "phones (its long log-mel window has the bias the short-window alignment features avoid; the "
                    "noisiest phone, cam06, deviates most)."}


# ------------------------------------------------------------------------------------------------ soundtrack
def stretch_audio(path: Path, slope: float, sr_out: int, cfg: SongConfig) -> tuple[np.ndarray, str]:
    """(n, 2) float32 at sr_out whose sample k holds song time k / (slope * sr_out), i.e. sample k sits at reference
    offset + k / sr_out: pitch-preserving time-stretch by tempo 1 / slope (ffmpeg rubberband; output duration =
    slope x song duration) when |slope - 1| > cfg.stretch_tol, else a plain decode (the slope is then ignored)."""
    import soxr
    path = resolve_path(path)
    sr_in = B.probe_sample_rate(path)
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-vn"]
    if abs(slope - 1.0) <= cfg.stretch_tol:
        method = f"shift only (|slope-1| <= {cfg.stretch_tol:g})"
    else:
        tempo = 1.0 / slope
        cmd += ["-af", f"rubberband=tempo={tempo:.12f}:pitch=1" + (f":{cfg.rubberband}" if cfg.rubberband else "")]
        method = f"pitch-preserving time-stretch by tempo {tempo:.8f} (ffmpeg rubberband, {cfg.rubberband})"
    raw = subprocess.check_output(cmd + ["-ac", "2", "-ar", str(sr_in), "-f", "f32le", "-"])
    y = np.frombuffer(raw, np.float32).reshape(-1, 2)
    if sr_in != sr_out:
        y = soxr.resample(y, sr_in, sr_out, quality="VHQ").astype(np.float32)
        method += f"; {sr_in} -> {sr_out} Hz by soxr VHQ"
    return np.ascontiguousarray(y), method


def place(y: np.ndarray, sr: int, offset: float, t0: float, t1: float) -> tuple[np.ndarray, float]:
    """Audio whose sample k sits at reference offset + k / sr, cut/zero-padded to [t0, t1). Returns (out, reference
    time of out[0]) - snapped to the sample grid of y (< 0.5 / sr from t0)."""
    m0 = int(round((t0 - offset) * sr))
    n = int(round((t1 - t0) * sr))
    out = np.zeros((n,) + y.shape[1:], np.float32)
    lo, hi = max(m0, 0), min(m0 + n, len(y))
    if hi > lo:
        out[lo - m0:hi - m0] = y[lo:hi]
    return out, offset + m0 / sr


def _fade_cut_edges(y: np.ndarray, sr: int, cut_start: bool, cut_end: bool, ms: float = 10.0) -> None:
    L = min(len(y) // 2, int(ms / 1000 * sr))
    ramp = np.linspace(0, 1, L, dtype=np.float32)[:, None]
    if cut_start:
        nz = np.nonzero(np.abs(y).max(1) > 0)[0]
        if len(nz):
            y[nz[0]:nz[0] + L] *= ramp[:len(y[nz[0]:nz[0] + L])]
    if cut_end:
        y[-L:] *= ramp[::-1]


def stretch_timing(Fs: np.ndarray, Fy: np.ndarray, dt: float, slope: float, offset_y: float, cfg: SongConfig) -> dict:
    """Measure where the studio content actually sits in a rendered track y (y time = slope * song + offset_y
    expected): constant latency (ms, positive = late) and rate error (ppm) from clean-vs-clean anchors."""
    c = cfg if cfg.anchor_window_s <= 8 else replace(cfg, anchor_window_s=8.0)
    centers = anchor_centers(slope, offset_y, Fs.shape[1] * dt, Fy.shape[1] * dt, c)
    an = local_anchors(Fs, Fy, dt, slope, offset_y, centers, c.anchor_window_s, 0.1)
    if len(an) < 2:
        return {"n_anchors": len(an), "latency_ms": 0.0, "rate_ppm": 0.0}
    x = np.array([a["song_t"] for a in an])
    d = np.array([a["ref_t"] for a in an]) - (slope * x + offset_y)
    ok = np.array([a["r"] >= 0.3 and not a["edge"] for a in an])
    if ok.sum() < 2:
        ok[:] = True
    f = robust_affine(x, d, ok, 4.0, 0.001)
    return {"n_anchors": len(an), "n_inliers": int(f["n"]), "latency_ms": round(1000 * float(np.median(d[f["inlier"]])), 3),
            "rate_ppm": round(1e6 * f["slope"], 2), "residual_ms_p95": round(1000 * float(np.percentile(
                np.abs(f["residual"][f["inlier"]]), 95)), 3), "r_median": round(float(np.median([a["r"] for a in an])), 3)}


def render_soundtrack(song_path: Path, slope: float, offset: float, t0: float, t1: float,
                      cfg: SongConfig | None = None, Fs: np.ndarray | None = None) -> tuple[np.ndarray, float, dict]:
    """The song on the reference timeline over [t0, t1): (y (n, 2) float32 at cfg.out_sr, reference time of y[0],
    meta). With Fs (studio onset features at ALIGN_SR) the stretch timing is measured and a constant latency removed."""
    import soxr
    cfg = cfg or SongConfig()
    sr = cfg.out_sr
    y_full, method = stretch_audio(song_path, slope, sr, cfg)
    meta = {"method": method, "tempo_factor": 1.0 / slope, "stretched": abs(slope - 1) > cfg.stretch_tol}
    if Fs is not None:
        dt = frame_dt(ALIGN_SR)
        mono = soxr.resample(y_full.mean(1), sr, ALIGN_SR, quality="HQ")
        chk = stretch_timing(Fs, onset_features(mono, ALIGN_SR), dt, slope, 0.0, cfg)
        meta["stretch_timing_before"] = chk
        lat = chk["latency_ms"] / 1000
        if abs(chk["latency_ms"]) > cfg.latency_tol_ms:
            k = int(round(lat * sr))  # content is k samples late: drop k samples (or pad when early)
            y_full = y_full[k:] if k > 0 else np.concatenate([np.zeros((-k, 2), np.float32), y_full])
            mono = soxr.resample(y_full.mean(1), sr, ALIGN_SR, quality="HQ")
            meta["latency_removed_samples"] = k
            meta["stretch_timing_after"] = stretch_timing(Fs, onset_features(mono, ALIGN_SR), dt, slope, 0.0, cfg)
        else:
            meta["latency_removed_samples"] = 0
    y, ref_start = place(y_full, sr, offset, t0, t1)
    song_dur_ref = len(y_full) / sr
    _fade_cut_edges(y, sr, cut_start=t0 > offset + 1e-3, cut_end=t1 < offset + song_dur_ref - 1e-3)
    return y, ref_start, meta


def run_soundtrack(sess: Session) -> dict:
    cfg = sess.cfg
    sj = sess.song_json()
    s, o = sj["alignment"]["slope"], sj["alignment"]["offset"]
    t0, t1 = sess.soundtrack_span(s, o)
    print(f"[song] soundtrack on the reference timeline {t0:.3f}..{t1:.3f} s (tempo factor {1 / s:.6f})")
    y, ref_start, meta = render_soundtrack(sess.info.path, s, o, t0, t1, cfg, sess.song_F())
    sr = cfg.out_sr
    # end-to-end check against the reference recording
    import soxr
    Fy = onset_features(soxr.resample(y.mean(1), sr, ALIGN_SR, quality="HQ"), ALIGN_SR)
    e2e = align_check(Fy, sess.cam_F(sess.ref.id), frame_dt(ALIGN_SR), ref_start, cfg)
    meta["end_to_end_vs_reference"] = e2e
    print(f"[song] stretch timing: {meta.get('stretch_timing_before')}; latency removed "
          f"{meta.get('latency_removed_samples')} samples; end-to-end vs cam{sess.ref.id}: {e2e}")
    peak = float(np.abs(y).max()) + 1e-9
    gain = min(1.0, 0.999 / peak)
    p = write_private_wav(sess.out / "studio_reference.wav", y * gain, sr)
    js = {"file": p.name, "sample_rate": sr, "channels": 2, "n_samples": int(len(y)),
          "reference_start": round(ref_start, 7), "reference_end": round(ref_start + len(y) / sr, 7),
          "time_to_audio": round(-ref_start, 7), "song_start": round((ref_start - o) / s, 7),
          "clock": "audio sample i is at reference time reference_start + i / sample_rate",
          "alignment": {"convention": CONVENTION, "slope": s, "offset": o},
          "span_note": "where the song plays inside any camera's recording (alignment.json clocks); outside the "
                       "song's own extent the samples are silent",
          "gain": round(gain, 6), "peak_before_gain": round(peak, 4), **meta,
          "timing_accuracy_ms": ACCURACY_MS,
          "timing_accuracy_note": "the soundtrack is placed on the reference clock to about +-1 ms absolute; "
                                  "end_to_end_vs_reference uses this module's onset features, and other onset "
                                  "definitions place it up to ~1.3 ms differently (feature-definition dependent)",
          "export_hint": f"PRIVATE BUILDS ONLY (see distribution): python -m dancecap export {sess.take.name} "
                         f"--capture <Name> --audio work/{sess.take.name}/song/studio_reference.wav --audio-t0 "
                         f"{ref_start:.7f}",
          "distribution": DISTRIBUTION,
          "render_hint": "ffmpeg ... -itsoffset <reference_start - video_reference_start> -i studio_reference.wav",
          "source_song_sha256_16": sj["song"]["sha256_16"], "copyright": f"{COPYRIGHT} {DISTRIBUTION}"}
    write_json(sess.out / "studio_reference.json", js)
    print(f"[song] wrote {p} ({len(y) / sr:.2f} s, gain {gain:.4f})")
    return js


def align_check(Fy: np.ndarray, Fr: np.ndarray, dt: float, ref_start: float, cfg: SongConfig) -> dict:
    """Anchors of a rendered reference-timeline track (y time u -> reference ref_start + u) against the reference."""
    centers = anchor_centers(1.0, ref_start, Fy.shape[1] * dt, Fr.shape[1] * dt, cfg)
    an = local_anchors(Fy, Fr, dt, 1.0, ref_start, centers, cfg.anchor_window_s, cfg.anchor_search_s)
    if len(an) < 2:
        return {"n_anchors": len(an)}
    f = _fit_anchors(an, cfg)
    x = np.array([a["song_t"] for a in an])
    d = 1000 * (np.array([a["ref_t"] for a in an]) - (x + ref_start))
    inl = f["inlier"]
    return {"n_anchors": len(an), "n_inliers": int(inl.sum()),
            "offset_error_ms_median": round(float(np.median(d[inl])), 3),
            "abs_error_ms_p95": round(float(np.percentile(np.abs(d[inl]), 95)), 3),
            "slope_minus_1_ppm": round(1e6 * (f["slope"] - 1), 2)}


# ------------------------------------------------------------------------------------------------ beats stage
def _device(cfg: SongConfig):
    import torch

    from . import gpu
    if cfg.device == "cpu" or not torch.cuda.is_available():
        return "cpu", contextlib.nullcontext()
    try:
        ctx = gpu.exclusive("song-beats", timeout_s=cfg.gpu_wait_s)
        ctx.__enter__()
    except TimeoutError:
        if cfg.device:
            raise
        print(f"[song] GPU busy ({gpu.LOCK.read_text() if gpu.LOCK.exists() else '?'}); beat_this on CPU")
        return "cpu", contextlib.nullcontext()
    return "cuda", B._Entered(ctx)


def studio_grid(x: np.ndarray, sr: int, cfg: SongConfig, tracker_out: dict | None = None) -> dict:
    """beats.py's grid construction on one clean signal (song time): beat level fixed explicitly (no split_measures
    doubling), gap repair, bar phase from beat_this downbeats, kick/backbeat rotation check (rephase only when the
    downbeats are ambiguous), zouk grid with eighths at quarter-beat midpoints."""
    if tracker_out is None:
        device, lock = _device(cfg)
        with lock:
            tracker = B.BeatTracker(cfg.checkpoint, device)
            tr = tracker(x, sr)
            del tracker
        tr["device"] = device
    else:
        tr = tracker_out
    b, d = tr["beats_refined"], tr["downbeats_refined"]
    q, level = B.fix_beat_level(b, d, cfg.min_bpm)
    q, gaps = B.repair_gaps(q)
    ph = B.bar_phase(q, d)
    grid = B.zouk_grid(q, ph["phase"])
    lo_o, lo_t = B.band_onset(x, sr)
    sn_o, sn_t = B.band_onset(x, sr, 1000.0, 4000.0)
    pv = level.get("parity_votes")
    ambiguous = ph["agreement"] < 0.8 or (pv is not None and min(pv) > 0.5 * max(pv))
    rot = B.rotation_scores(B.folded_profile(grid, lo_t, lo_o), B.folded_profile(grid, sn_t, sn_o))
    decision = {"by": "beat_this downbeats", "beat_this_ambiguous": bool(ambiguous), "eighth_rotation": rot}
    if rot["best"] != 0 and ambiguous and rot["margin"] > cfg.rephase_margin:
        q, phase = B.rephase(q, ph["phase"], rot["best"])
        grid = B.zouk_grid(q, phase)
        decision.update(by="kick on 1 + snare backbeat (beat_this downbeats ambiguous)", shifted_eighths=rot["best"])
    pcheck = B.phase_check(B.folded_profile(grid, lo_t, lo_o), B.folded_profile(grid, sn_t, sn_o))
    ibi = np.diff(q)
    i = np.arange(len(q), dtype=float)
    coef = np.polyfit(i, q, 1)
    resid = q - np.polyval(coef, i)
    lt, lb = B.local_bpm(q)
    tempo = {"bpm": round(60.0 / float(np.median(ibi)), 3), "bpm_linear_fit": round(60.0 / float(coef[0]), 3),
             "beat_period_s": round(float(coef[0]), 5), "measure_s": round(4 * float(coef[0]), 5),
             "linear_fit_resid_rms_ms": round(1000 * float(np.sqrt(np.mean(resid ** 2))), 2),
             "linear_fit_resid_max_ms": round(1000 * float(np.abs(resid).max()), 2),
             "local_bpm_range": [round(float(lb.min()), 3), round(float(lb.max()), 3)] if len(lb) else None,
             "beat_level": {**level, "gaps": gaps, "min_bpm": cfg.min_bpm}, "bar_phase": ph,
             "phase_decision": decision, "phase_check": pcheck, "analyzer_split_measures": B.analyzer_split_check(d),
             "beat_confidence": round(float(np.mean(tr["beat_peak_prob"])), 4) if len(tr["beat_peak_prob"]) else 0.0,
             "n_quarter_beats": int(len(q)), "n_measures": int(grid[:, 2].max() + 1) if len(grid) else 0}
    return {"grid": grid, "quarters": q, "tempo": tempo, "tracker": tr, "device": tr.get("device", "?")}


def to_reference_grid(grid_song: np.ndarray, slope: float, offset: float, t0: float, t1: float) -> tuple[np.ndarray, int]:
    """Song-time grid [t, type, measure, slot] -> reference time, whole measures inside [t0, t1] only, measures
    renumbered from 0. Returns (grid, song measure index of the first kept measure)."""
    g = np.asarray(grid_song, float).copy()
    g[:, 0] = slope * g[:, 0] + offset
    keep = []
    for m in np.unique(g[:, 2]).astype(int):
        rows = g[g[:, 2] == m]
        if len(rows) == 8 and rows[0, 0] >= t0 and rows[-1, 0] <= t1:
            keep.append(m)
    if not keep:
        return np.zeros((0, 4)), -1
    out = g[np.isin(g[:, 2].astype(int), keep)]
    first = int(keep[0])
    # renumber consecutively (whole measures kept are contiguous inside one interval)
    out[:, 2] = np.searchsorted(np.asarray(keep), out[:, 2].astype(int))
    return out, first


def measure_origin(grid_ref: np.ndarray, phone: np.ndarray | None, t_min: float = 0.0) -> dict:
    """Which measure of a reference grid [t, type, measure, slot] (measures numbered from 0, whole measures) becomes
    measure 0 of beats_studio.json, so that its measure numbers equal the phone grid's.

    timing.load_beats numbers measures as index // 8 of the bare list, and beats.json starts on a downbeat, so the
    phone grid's measure m starts at its entry 8 m. The first studio downbeat that has a phone entry within half an
    eighth fixes the correspondence: when that entry is a phone downbeat (entry 8 m), studio measure k is phone
    measure k - (k_s - m), and cropping the studio grid at measure k_s - m gives both grids the same numbers. If the
    phone grid starts after the studio grid that crop index is >= 0 (the usual case); if the phone grid starts
    earlier it is < 0: no crop can fix that in a bare list, so measure_index_offset reports how far the studio
    numbers lag (beats.json measure = beats_studio.json measure + offset). If the matched phone entry is not a
    downbeat the bar phases differ (compare reports it) and, as without a phone grid, the crop falls back to the first
    whole measure starting at reference >= t_min, with measure_index_offset None (numbering not matched).

    Returns {crop_measure, measure_index_offset, by, ...}."""
    G = np.asarray(grid_ref, float)
    if not len(G):
        return {"crop_measure": 0, "measure_index_offset": 0, "by": "empty grid"}
    m_of = G[:, 2].astype(int)

    def fallback(why: str, **extra) -> dict:
        k = next((int(m) for m in np.unique(m_of) if G[m_of == m, 0].min() >= t_min), int(m_of.max()) + 1)
        return {"crop_measure": k, "measure_index_offset": None,  # None: numbering not matched to a phone grid
                "by": f"first whole measure starting at reference >= {t_min:g} s ({why})", **extra}

    if phone is None or not len(phone):
        return fallback("no phone grid to number against")
    P = np.asarray(phone, float).reshape(-1, 2)
    eighth = float(np.median(np.diff(G[:, 0])))
    down = G[G[:, 3] == 0]
    for t, m in zip(down[:, 0], down[:, 2].astype(int)):
        j = int(np.argmin(np.abs(P[:, 0] - t)))
        d = float(P[j, 0] - t)
        if abs(d) >= 0.5 * eighth:
            continue
        info = {"studio_measure": int(m), "studio_t": round(float(t), 6), "phone_index": j,
                "phone_t": round(float(P[j, 0]), 6), "phone_minus_studio_ms": round(1000 * d, 2)}
        if j % 8:
            return fallback("the phone grid's bar phase differs: measure numbers cannot be matched", first_match=info,
                            phase_differs=True)
        k = int(m) - j // 8
        return {"crop_measure": max(k, 0), "measure_index_offset": max(-k, 0), "first_match": info,
                "by": "matched to beats.json's measure numbering: the first studio downbeat with a phone downbeat "
                      "within half an eighth"}
    return fallback("no studio downbeat matches a phone grid entry")


def crop_measures(grid_ref: np.ndarray, first: int) -> np.ndarray:
    """Measures >= first of a [t, type, measure, slot] grid, renumbered from 0."""
    G = np.asarray(grid_ref, float)
    out = G[G[:, 2] >= first].copy()
    out[:, 2] -= first
    return out


def _band_envelope(x: np.ndarray, sr: int, lo: float, hi: float, win_s: float) -> np.ndarray:
    """Zero-phase band-pass (4th-order Butterworth, filtfilt) + centred RMS over win_s: no filter / window delay."""
    from scipy import signal
    sos = signal.butter(4, [lo, min(hi, 0.45 * sr)], "bandpass", fs=sr, output="sos")
    y = signal.sosfiltfilt(sos, np.asarray(x, np.float64))
    n = max(1, int(round(win_s * sr)) | 1)
    return np.sqrt(np.convolve(y * y, np.ones(n) / n, "same"))


def _log_flux(e: np.ndarray, smooth_s: float, sr: int) -> np.ndarray:
    """Half-wave rectified central-difference log-envelope flux (no half-sample shift), Hann-smoothed (centred)."""
    f = np.maximum(0.0, np.gradient(np.log(e + 1e-4 * e.max() + 1e-12)))
    n = max(1, int(round(smooth_s * sr)) | 1)
    w = np.hanning(n + 2)[1:-1]
    return np.convolve(f, w / w.sum(), "same")


def ideal_onsets(sr: int, dur: float = 30.0, period: float = 0.8, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Calibration signal for beat_placement: a synthetic kick (50-120 Hz down-chirp, 70 ms decay) and a hi-hat
    (differentiated noise, 6 ms decay) starting exactly at known times. Returns (x, onset times)."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(0.2 * sr)) / sr
    kick = 0.9 * np.sin(2 * np.pi * (50 + 70 * np.exp(-t / 0.03)) * t) * np.exp(-t / 0.07)
    n = int(0.04 * sr)
    x = np.zeros(int(dur * sr))
    on = np.arange(0.5, dur - 0.5, period)
    for t0 in on:
        s = int(round(t0 * sr))
        hat = 0.15 * np.diff(rng.normal(0, 1, n) * np.exp(-np.arange(n) / (0.006 * sr)), prepend=0.0)
        x[s:s + len(kick)] += kick
        x[s:s + n] += hat
    return x + rng.normal(0, 1e-4, len(x)), on


def beat_placement(x: np.ndarray, sr: int, beats: np.ndarray, half_s: float = 0.1, step_s: float = 0.0005,
                   calibrate: bool = True) -> dict:
    """Where the beats sit relative to the music's onsets: zero-phase band features of x folded on the beat times
    (mean over beats of the feature at beat + lag, lag in +-half_s) and the lag of the folded peak. lag < 0 = that
    feature peaks BEFORE the beat. Features: kick/bass (30-150 Hz) RMS envelope over 20 ms (energy peak) and its log
    flux (attack), 2-8 kHz log flux over a 2 ms envelope (high-frequency transients). Lags are in x's time base.
    The features have their own lags (window and band-pass smearing, the kick's energy build-up): with calibrate,
    the same features on ideal_onsets (kick + hat starting exactly on the 'beat') give ideal_onset_lag_ms, and
    <feature>_vs_ideal_ms = lag - ideal lag = how far the beats are from where an instantaneous onset would put them
    (e.g. -16 = that feature's onsets come 16 ms before the beats, compared with an onset exactly on the beat)."""
    beats = np.asarray(beats, float)
    ts = np.arange(len(x)) / sr
    lags = np.arange(-half_s, half_s + step_s / 2, step_s)
    ok = (beats - half_s > ts[0]) & (beats + half_s < ts[-1])
    kick = _band_envelope(x, sr, 30.0, 150.0, 0.02)
    feats = {"kick_bass_envelope_peak": kick, "kick_bass_flux_peak": _log_flux(kick, 0.005, sr),
             "hf_2_8khz_flux_peak": _log_flux(_band_envelope(x, sr, 2000.0, 8000.0, 0.002), 0.003, sr)}
    out = {}
    for name, f in feats.items():
        p = np.mean([np.interp(t + lags, ts, f) for t in beats[ok]], 0)
        k = int(np.argmax(p))
        out[name + "_ms"] = round(1000 * float(lags[k] + _parabola(p, k) * step_s), 2)
    out["n_beats"] = int(ok.sum())
    if calibrate:
        xi, oi = ideal_onsets(sr)
        ideal = beat_placement(xi, sr, oi, half_s, step_s, calibrate=False)
        out["ideal_onset_lag_ms"] = {k: v for k, v in ideal.items() if k.endswith("_ms")}
        for k in feats:
            out[k + "_vs_ideal_ms"] = round(out[k + "_ms"] - ideal[k + "_ms"], 2)
    return out


def refine_effect(grid_ref: np.ndarray) -> dict | None:
    """How much timing.py's default refine_beats (local-linear tempo fit over +-1 measure) would move this grid."""
    try:
        from . import timing
        g = np.asarray(grid_ref, float)
        bg = timing.BeatGrid(g[:, 0], g[:, 1].astype(int), np.arange(len(g)) // 8, np.arange(len(g)) % 8)
        ref, rms = timing.refine_beats(bg)
        d = 1000 * (ref.t - g[:, 0])
        return {"rms_ms": round(1000 * rms, 2), "max_abs_ms": round(float(np.abs(d).max()), 2) if len(d) else 0.0}
    except Exception as e:  # timing.py is owned elsewhere; this is a diagnostic only
        return {"error": f"{type(e).__name__}: {e}"}


def run_beats(sess: Session) -> dict:
    cfg = sess.cfg
    sj = sess.song_json()
    s, o = sj["alignment"]["slope"], sj["alignment"]["offset"]
    x = sess.song_x(B.ANALYSIS_SR)
    res = studio_grid(x, B.ANALYSIS_SR, cfg)
    tr, tempo, grid_song = res["tracker"], res["tempo"], res["grid"]
    t0, t1 = sess.soundtrack_span(s, o)
    grid_full, first = to_reference_grid(grid_song, s, o, t0, t1)
    phone_path = sess.take.work / "beats.json"
    phone = np.asarray(read_json(phone_path), float).reshape(-1, 2) if phone_path.exists() else None
    origin = measure_origin(grid_full, phone)
    crop = int(origin["crop_measure"])
    grid_ref = crop_measures(grid_full, crop)
    origin.update(full_measure_offset=crop, phone_grid=f"work/{sess.take.name}/beats.json" if phone is not None else None,
                  note="beats_studio.json measure k = beats_studio_full.json measure k + full_measure_offset; "
                       "beats.json measure = beats_studio.json measure + measure_index_offset (0: same numbers; "
                       "None: not matched - no phone grid, bar phases differ or no matching downbeat)")
    tempo_ref = {"bpm": round(tempo["bpm"] / s, 3), "bpm_linear_fit": round(tempo["bpm_linear_fit"] / s, 3),
                 "measure_s": round(tempo["measure_s"] * s, 5), "note": "reference-time tempo = song tempo / slope"}
    n_meas = int(grid_ref[:, 2].max()) + 1 if len(grid_ref) else 0
    print(f"[song] beat_this {cfg.checkpoint} on {res['device']}: {len(tr['beats'])} beats, {len(tr['downbeats'])} "
          f"downbeats; song tempo {tempo['bpm']:.3f} BPM -> reference {tempo_ref['bpm']:.3f} BPM; bar phase by "
          f"{tempo['phase_decision']['by']} (agreement {tempo['bar_phase']['agreement']}), kick/backbeat "
          f"{'consistent' if tempo['phase_check']['consistent'] else 'CHECK'}; "
          f"{int(grid_full[:, 2].max()) + 1 if len(grid_full) else 0} measures in {t0:.2f}..{t1:.2f} s, "
          f"{n_meas} from measure {crop} on ({origin['by']}; measure_index_offset {origin['measure_index_offset']})")
    place_song = beat_placement(x, B.ANALYSIS_SR, res["quarters"])
    placement = {k: (round(v * s, 2) if k.endswith("_ms") and isinstance(v, float) else v)
                 for k, v in place_song.items()}  # song ms -> reference ms (the calibration lags are feature lags)
    hv, kv = placement["hf_2_8khz_flux_peak_vs_ideal_ms"], placement["kick_bass_envelope_peak_vs_ideal_ms"]
    kf = placement["kick_bass_flux_peak_vs_ideal_ms"]
    lead = -(hv + kf) / 2  # how far the attacks lead the beats
    marks_energy = abs(kv) <= 5.0 and lead > 5.0
    placement.update(
        beats_mark=("low-frequency energy (kick/bass envelope), attacks lead" if marks_energy else
                    "attacks (onsets)" if abs(hv) <= 5.0 and abs(kf) <= 5.0 else "unclear"),
        attacks_lead_ms=round(lead, 2),
        columns_note="*_ms: lag (reference ms) of the folded feature peak relative to the beats, < 0 = before the "
                     "beat; *_vs_ideal_ms: the same minus the lag that feature shows for a synthetic kick + hi-hat "
                     "starting exactly on the beat (ideal_onset_lag_ms) = how far that kind of onset is from the beat",
        method="zero-phase band-pass + centred envelopes of the studio track folded on the beat_this quarter beats "
               "(+-100 ms, 0.5 ms steps), in song time (independent of the alignment); kick/bass 30-150 Hz RMS over "
               "20 ms (peak) and its log flux (attack); 2-8 kHz log flux over a 2 ms envelope",
        meaning=f"compared with a kick + hi-hat starting exactly on the beat (< 0 = earlier than the beats): "
                f"high-frequency transients (2-8 kHz) {hv:+.1f} ms, kick attack {kf:+.1f} ms, kick/bass energy "
                f"(envelope peak) {kv:+.1f} ms." +
                (f" The beats therefore mark the low-frequency energy, not the attacks, which lead them by about "
                 f"{lead:.0f} ms." if marks_energy else "") +
                " This is how beat_this places beats on this mix, measured on the studio track itself (not an "
                "alignment error). Timing / asynchrony analyses that call a dancer early or late in absolute terms "
                "should state their reference: the grid as is (low-frequency energy), or the grid shifted by "
                "hf_2_8khz_flux_peak_vs_ideal_ms (attack / transient onsets). Values depend on the feature definition "
                "by a few ms.")
    out = [[round(float(t), 6), int(k)] for t, k in grid_ref[:, :2]]
    write_json(sess.out / "beats_studio.json", out, indent=None)
    write_json(sess.out / "beats_studio_full.json", [[round(float(t), 6), int(k)] for t, k in grid_full[:, :2]],
               indent=None)
    n_full = int(grid_full[:, 2].max()) + 1 if len(grid_full) else 0
    gs_full = grid_song[(grid_song[:, 2] >= first) & (grid_song[:, 2] < first + n_full)].copy()
    gs_full[:, 2] -= first
    gs = crop_measures(gs_full, crop)

    def rows(g):
        return [[round(float(a), 6), int(b), int(c), int(d)] for a, b, c, d in g]

    raw = {
        "version": VERSION, "take": sess.take.name,
        "provenance": {"song_file": sj["song"]["file"], "song_sha256_16": sj["song"]["sha256_16"],
                       "song_json": "work/" + sess.take.name + "/song/song.json",
                       "alignment": {"convention": CONVENTION, "slope": s, "offset": o,
                                     "accuracy_ms": ACCURACY_MS},
                       "beat_this": {"checkpoint": cfg.checkpoint, "device": res["device"], "fps": B.BEAT_THIS_FPS,
                                     "input": f"studio track, mono {B.ANALYSIS_SR} Hz, whole song",
                                     "postprocessor": "minimal + parabolic sub-frame refinement"},
                       "grid": "beats.py helpers: fix_beat_level (no split_measures doubling), repair_gaps, bar_phase, "
                               "rotation check, zouk_grid (eighths = quarter-beat midpoints); song time mapped through "
                               "the affine fit; whole measures inside the soundtrack span (beats_studio_full.json), "
                               "then cropped so measure 0 matches beats.json's measure 0 (beats_studio.json)",
                       "measure_origin": origin, "beat_placement": placement,
                       "span_reference": [t0, t1], "first_song_measure": first + crop,
                       "first_song_measure_full": first, "module": "dancecap.song"},
        "files": {"beats_studio.json": "reference_time (measure-aligned with beats.json)",
                  "beats_studio_full.json": "reference_time_full (whole soundtrack span)"},
        "grid_columns": ["t", "type", "measure", "slot"],
        "reference_time": rows(grid_ref),
        "song_time": rows(gs),
        "reference_time_full": rows(grid_full),
        "song_time_full": rows(gs_full),
        "song_time_all_measures": rows(grid_song),
        "beat_this_song_time": {"beats": B._r(tr["beats_refined"]), "downbeats": B._r(tr["downbeats_refined"]),
                                "beats_frame": B._r(tr["beats"]), "downbeats_frame": B._r(tr["downbeats"])},
        "quarters_song_time": B._r(res["quarters"]),
        "tempo_song": tempo, "tempo_reference": tempo_ref,
    }
    write_json(sess.out / "beats_studio_raw.json", raw)
    print(f"[song] beat placement vs onsets exactly on the beat: HF transients {hv:+.1f} ms, kick/bass energy peak "
          f"{kv:+.1f} ms, kick attack {kf:+.1f} ms (raw lags: HF {placement['hf_2_8khz_flux_peak_ms']:+.1f}, kick "
          f"envelope {placement['kick_bass_envelope_peak_ms']:+.1f} ms)")
    print(f"[song] wrote {sess.out / 'beats_studio.json'} ({len(out)} entries, measure-aligned), beats_studio_full.json "
          f"({len(grid_full)} entries) + beats_studio_raw.json")
    st = None
    if (sess.out / "studio_reference.wav").exists():  # always the written (16-bit) file: same overlay whichever way
        y, _sr = B.read_wav(sess.out / "studio_reference.wav")  # the stages were run (all, or beats alone)
        st = (y, read_json(sess.out / "studio_reference.json")["reference_start"])
    if st is not None:
        y, ref_start = st
        y = y if y.ndim == 2 else y[:, None]
        clicks = B.click_track(len(y), cfg.out_sr, grid_full, ref_start)
        mix = 0.6 * y + clicks[:, None]
        write_private_wav(sess.out / "studio_clicks_overlay.wav", mix / max(1.0, float(np.abs(mix).max()) / 0.98),
                          cfg.out_sr)
    return raw


# ------------------------------------------------------------------------------------------------ compare stage
def grid_tempo(t: np.ndarray, slot: np.ndarray) -> dict:
    """Quarter-note tempo of a zouk eighth grid (quarters = even slots)."""
    q = np.asarray(t, float)[np.asarray(slot) % 2 == 0]
    if len(q) < 3:
        return {"bpm": None}
    i = np.arange(len(q), dtype=float)
    coef = np.polyfit(i, q, 1)
    return {"bpm": round(60.0 / float(np.median(np.diff(q))), 3), "bpm_linear_fit": round(60.0 / float(coef[0]), 3),
            "measure_s": round(4 * float(coef[0]), 5), "n_quarters": int(len(q))}


def diff_stats(d_s: np.ndarray) -> dict:
    d = 1000 * np.asarray(d_s, float)
    if not len(d):
        return {"n": 0}
    med = float(np.median(d))
    return {"n": int(len(d)), "median_ms": round(med, 2), "mean_ms": round(float(d.mean()), 2),
            "sd_ms": round(float(d.std()), 2), "mad_sd_ms": round(1.4826 * float(np.median(np.abs(d - med))), 2),
            "abs_median_ms": round(float(np.median(np.abs(d))), 2),
            "abs_p95_ms": round(float(np.percentile(np.abs(d), 95)), 2), "abs_max_ms": round(float(np.abs(d).max()), 2),
            "centered_p95_ms": round(float(np.percentile(np.abs(d - med), 95)), 2),
            "centered_max_ms": round(float(np.abs(d - med).max()), 2)}


def compare_grids(studio: np.ndarray, phone: np.ndarray, window: tuple[float, float] | None = None,
                  bin_s: float = 20.0) -> dict:
    """Studio grid [t, type, (measure, slot)] vs phone grid [t, type] (slot = index % 8, measure = index // 8 as
    timing.load_beats numbers it; beats.json starts on a downbeat). Each studio entry inside the phone grid's span is
    matched to the nearest phone entry (within half an eighth); differences are phone - studio. Phase: slot agreement,
    the dominant slot shift, downbeat (slot 0) agreement, type agreement; numbering: measure agreement and the
    dominant measure shift (phone measure - studio measure) - a grid with the same phase can still number its measures
    differently when it starts at another measure."""
    S = np.asarray(studio, float)
    P = np.asarray(phone, float).reshape(-1, 2)
    s_t, s_type = S[:, 0], S[:, 1].astype(int)
    s_slot = S[:, 3].astype(int) if S.shape[1] >= 4 else np.arange(len(S)) % 8
    s_meas = S[:, 2].astype(int) if S.shape[1] >= 4 else np.arange(len(S)) // 8
    p_t, p_type, p_slot = P[:, 0], P[:, 1].astype(int), np.arange(len(P)) % 8
    p_meas = np.arange(len(P)) // 8
    eighth = float(np.median(np.diff(s_t)))
    ov = (max(s_t[0], p_t[0]) - 0.5 * eighth, min(s_t[-1], p_t[-1]) + 0.5 * eighth)

    def match(sel: np.ndarray) -> dict:
        idx = np.nonzero(sel)[0]
        if not len(idx):
            return {"n": 0}
        j = np.clip(np.searchsorted(p_t, s_t[idx]), 1, len(p_t) - 1)
        j = np.where(np.abs(p_t[j - 1] - s_t[idx]) < np.abs(p_t[j] - s_t[idx]), j - 1, j)
        d = p_t[j] - s_t[idx]
        ok = np.abs(d) < 0.5 * eighth
        idx, j, d = idx[ok], j[ok], d[ok]
        shift = (p_slot[j] - s_slot[idx]) % 8
        mshift = p_meas[j] - s_meas[idx]
        mv, mc = np.unique(mshift, return_counts=True)
        down = s_slot[idx] == 0
        even = s_slot[idx] % 2 == 0
        out = {"stats": diff_stats(d), "quarters_stats": diff_stats(d[even]),
               "slot_agreement": round(float(np.mean(shift == 0)), 4),
               "slot_shift_mode": int(np.bincount(shift, minlength=8).argmax()) if len(shift) else None,
               "slot_shift_counts": np.bincount(shift, minlength=8).tolist(),
               "downbeat_agreement": round(float(np.mean(p_slot[j][down] == 0)), 4) if down.any() else None,
               "type_agreement": round(float(np.mean(p_type[j] == s_type[idx])), 4),
               "measure_agreement": round(float(np.mean(mshift == 0)), 4) if len(mshift) else None,
               "measure_shift_mode": int(mv[np.argmax(mc)]) if len(mv) else None,
               "measure_shift_counts": {str(int(v)): int(c) for v, c in zip(mv, mc)},
               "n_matched": int(len(idx)), "n_studio": int(sel.sum())}
        return out | {"_idx": idx, "_j": j, "_d": d}

    sel = (s_t >= ov[0]) & (s_t <= ov[1])
    allm = match(sel)
    d, t = allm["_d"], s_t[allm["_idx"]]
    drift = {}
    if len(d) > 3:
        c = np.polyfit(t, 1000 * d, 1)
        edges = np.arange(t.min(), t.max() + bin_s, bin_s)
        bins = []
        for a, b in zip(edges[:-1], edges[1:]):
            m = (t >= a) & (t < b)
            if m.sum() >= 4:
                bins.append([round(float(0.5 * (a + b)), 2), round(1000 * float(np.median(d[m])), 2), int(m.sum())])
        drift = {"ms_per_min": round(60 * float(c[0]), 3), "intercept_ms_at_t0": round(float(np.polyval(c, t.min())), 2),
                 "bins": bins, "bins_columns": ["t_center", "median_ms", "n"],
                 "bin_median_range_ms": [min(b[1] for b in bins), max(b[1] for b in bins)] if bins else None}
    res = {"overlap": [round(ov[0], 4), round(ov[1], 4)], "eighth_s": round(eighth, 5),
           "all": {k: v for k, v in allm.items() if not k.startswith("_")}, "drift": drift,
           "tempo": {"studio": grid_tempo(s_t, s_slot), "phone": grid_tempo(p_t, p_slot),
                     "studio_in_overlap": grid_tempo(s_t[sel], s_slot[sel]),
                     "phone_in_overlap": grid_tempo(p_t[(p_t >= ov[0]) & (p_t <= ov[1])],
                                                    p_slot[(p_t >= ov[0]) & (p_t <= ov[1])])},
           "pairs": {"t_studio": s_t[allm["_idx"]], "d_ms": 1000 * d, "type": s_type[allm["_idx"]],
                     "slot_studio": s_slot[allm["_idx"]], "slot_phone": p_slot[allm["_j"]]}}
    if window is not None:
        w = match((s_t >= window[0]) & (s_t <= window[1]))
        rows = [[round(float(s_t[i]), 4), round(float(p_t[j]), 4), round(1000 * float(dd), 2), int(s_slot[i]),
                 int(p_slot[j]), int(s_type[i]), int(p_type[j]), int(s_meas[i]), int(p_meas[j])]
                for i, j, dd in zip(w.get("_idx", []), w.get("_j", []), w.get("_d", []))]
        res["window"] = {"range": list(window), **{k: v for k, v in w.items() if not k.startswith("_")},
                         "rows_columns": ["t_studio", "t_phone", "phone_minus_studio_ms", "slot_studio",
                                          "slot_phone", "type_studio", "type_phone", "measure_studio",
                                          "measure_phone"], "rows": rows}
    return res


def current_capture(work: Path) -> dict:
    """The take's existing capture export (work/<take>/export/<capture>/capture.json) - its name and time_to_audio -
    else the export default audio (audio/audio_overlap.json)."""
    caps = sorted((work / "export").glob("*/capture.json")) if (work / "export").exists() else []
    for c in caps:
        with contextlib.suppress(ValueError, KeyError, OSError):
            m = read_json(c)
            if m.get("time_to_audio") is not None:
                return {"capture": c.parent.name, "time_to_audio": float(m["time_to_audio"]),
                        "audio": (m.get("provenance") or {}).get("audio") or "?"}
    ao = work / "audio" / "audio_overlap.json"
    if ao.exists():
        return {"capture": "<Name>", "time_to_audio": float(read_json(ao)["time_to_audio"]),
                "audio": "audio/audio_overlap.wav"}
    return {"capture": "<Name>", "time_to_audio": None, "audio": None}


def adoption_steps(take: str, refine: dict | None, tempo_song: dict, capture: dict, soundtrack: dict | None,
                   origin: dict | None) -> list[str]:
    """How the studio grid (and optionally the clean soundtrack) reaches timing, physics and the viewer, given how
    those stages pick their beats: timing/physics take --beats, else timing/timing.json (physics) / timing/beats.json
    / beats.json; export_unity takes timing/timing.json's grid whenever that exists and ignores --beats then."""
    cap = capture.get("capture") or "<Name>"
    rf = (f"would move this grid by {refine['rms_ms']:.1f} ms rms (max {refine['max_abs_ms']:.1f} ms)"
          if refine and "rms_ms" in refine else "would smooth this grid")
    steps = [
        f"1. timing (in dancecap/): python -m dancecap timing {take} --beats work/{take}/song/beats_studio.json "
        f"--no-refine. Keep --no-refine for this clean grid: the default refine_beats (local-linear tempo fit over "
        f"+-1 measure) {rf} and so discards real musical timing - the studio quarter beats deviate "
        f"{tempo_song['linear_fit_resid_rms_ms']:.1f} ms rms from a constant tempo, with structure, and beat_this "
        f"tracks them on the noise-free master. timing/timing.json's beats.grid then holds the studio grid.",
        f"2. physics: python -m dancecap physics {take} - it reads the beats from timing/timing.json once step 1 has "
        f"run (passing the same --beats is equivalent).",
        f"3. export: python -m dancecap export {take} --capture {cap} - export takes the grid from "
        f"timing/timing.json whenever that file exists and then ignores --beats, so the viewer gets the studio grid "
        f"only through step 1.",
    ]
    if soundtrack is not None:
        rs = float(soundtrack["reference_start"])
        cur = capture.get("time_to_audio")
        chg = (f"capture.json time_to_audio changes from {cur:+.7f} ({capture.get('audio')}) to {-rs:+.7f}"
               if cur is not None else f"capture.json time_to_audio becomes {-rs:+.7f}")
        steps.append(f"4. clean audio (optional, PRIVATE builds only - see studio_reference.json distribution): add "
                     f"--audio work/{take}/song/studio_reference.wav --audio-t0 {rs:.7f} to step 3. Then {chg}, so "
                     f"every capture-time file (moves, counterbalance) must be re-exported (step 5).")
    steps.append(f"5. after every capture export (it rewrites the capture folder), and always when time_to_audio "
                 f"changed: python -m dancecap.moves export {take} --capture {cap}; python -m dancecap.counterbalance "
                 f"export --take {take} --capture {cap}.")
    off = (origin or {}).get("measure_index_offset")
    same = ("its measure numbers equal beats.json's" if off == 0 else
            f"its measure numbers are {off} behind beats.json's" if off else "its measure numbers could not be matched")
    steps.append(f"alternative to --beats: copy song/beats_studio.json over work/{take}/beats.json once accepted "
                 f"({same}); timing/beats.json, if present, is read first, re-running dancecap.beats rewrites "
                 f"beats.json with the phone grid, and timing still needs --no-refine.")
    return steps


def recommend(sj: dict, comp: dict, tempo_song: dict, ctx: dict | None = None) -> dict:
    """Adopt or not, with reasons, caveats and the adoption steps. ctx (optional): take, refine (refine_effect),
    capture (current_capture), soundtrack (studio_reference.json), measure_origin, beat_placement."""
    ctx = ctx or {}
    al = sj["alignment"]
    ph = comp["all"]
    st = ph["stats"]
    reasons, caveats = [], []
    align_ok = al["residual_ms_p95"] <= 5.0 and al["n_inliers"] >= 0.8 * al["n_anchors"]
    same_phase = (ph["slot_agreement"] or 0) >= 0.95
    m_shift = ph.get("measure_shift_mode")
    same_measures = (ph.get("measure_agreement") or 0) >= 0.95 and m_shift == 0
    studio_phase_ok = tempo_song["phase_check"]["consistent"] and tempo_song["bar_phase"]["agreement"] >= 0.9
    cc = {c: v for c, v in (al.get("cross_checks") or {}).items() if v is not None}
    reasons.append(f"alignment: affine fit to the reference audio clock with {al['n_inliers']}/{al['n_anchors']} "
                   f"anchors, residual median {al['residual_ms_median']:.2f} ms / p95 {al['residual_ms_p95']:.2f} ms "
                   f"(precision); absolute accuracy about +-{al.get('accuracy_ms', ACCURACY_MS):g} ms (feature-"
                   f"definition dependent); {len(cc)} other phones cross-checked through alignment.json (their clock "
                   f"errors {min(cc.values()):+.1f}..{max(cc.values()):+.1f} ms)" if cc else "")
    reasons.append(f"tracking: beat_this on the noise-free master (mean beat confidence "
                   f"{tempo_song['beat_confidence']:.2f}, downbeat bar-phase agreement "
                   f"{tempo_song['bar_phase']['agreement']:.0%}, kick/backbeat check "
                   f"{'consistent' if tempo_song['phase_check']['consistent'] else 'NOT consistent'}); constant "
                   f"tempo residual rms {tempo_song['linear_fit_resid_rms_ms']:.1f} ms")
    org = ctx.get("measure_origin") or {}
    if org.get("full_measure_offset"):
        reasons.append(f"measure numbering: beats_studio.json starts at beats.json's first downbeat (measure "
                       f"{org['full_measure_offset']} of the full-span grid; the {org['full_measure_offset']} leading "
                       f"measure(s) before it are only in beats_studio_full.json)")
    if same_phase and same_measures:
        reasons.append(f"same bar phase and measure numbers: eighth 0 coincides for {ph['downbeat_agreement']:.0%} of "
                       f"measures, slots agree {ph['slot_agreement']:.0%}, beat types {ph['type_agreement']:.0%}, "
                       f"measure numbers {ph['measure_agreement']:.0%} - adopting changes the timing only, not the "
                       f"counts, measure labels or per-measure results")
    elif same_phase:
        reasons.append(f"same bar phase (slots agree {ph['slot_agreement']:.0%}) but measure numbers DIFFER: beats.json "
                       f"measure = studio measure {m_shift:+d} (agreement {ph.get('measure_agreement') or 0:.0%}) - "
                       f"adopting would renumber timing.json per_measure / flagged_measures and any measure % 8 "
                       f"phrase logic")
    else:
        reasons.append(f"bar phase DIFFERS: the phone grid's slots are shifted by {ph['slot_shift_mode']} eighth(s) "
                       f"(slot agreement {ph['slot_agreement']:.0%}); the studio phase is "
                       f"{'supported' if studio_phase_ok else 'NOT clearly supported'} by its kick/backbeat check")
    reasons.append(f"timing: phone - studio median {st['median_ms']:+.1f} ms (systematic), scatter around it p95 "
                   f"{st['centered_p95_ms']:.1f} ms / max {st['centered_max_ms']:.1f} ms; drift "
                   f"{comp['drift'].get('ms_per_min', 0):+.2f} ms/min")
    sysc = comp.get("expected_systematic_ms")
    if sysc is not None:
        reasons.append(f"the systematic part is expected: the phone grid sits on cam{comp.get('phone_source_cam')} "
                       f"timing, whose alignment.json clock is {sysc:+.1f} ms off the reference audio clock; the studio "
                       "grid is on the reference camera's own audio clock")
    if align_ok and same_phase and same_measures:
        adopt, verdict = True, "adopt the studio grid (timing with --no-refine; see how)"
    elif align_ok and same_phase:
        adopt, verdict = "after renumbering", ("adopt the studio grid once its measure numbers match beats.json's "
                                               "(or downstream per-measure results are renumbered)")
    elif align_ok and studio_phase_ok:
        adopt, verdict = "after a listening check", "adopt after confirming the bar phase by ear (studio_clicks_overlay.wav)"
    else:
        adopt, verdict = False, "keep the phone grid until the alignment / phase issues above are resolved"
    caveats.append("both grids are on the reference camera's AUDIO clock: absolute 'on the beat' claims about the video "
                   "still need an audio-visual sync event (phone A/V offsets, ~2.9 ms per metre of sound travel)")
    bp = ctx.get("beat_placement") or {}
    if "hf_2_8khz_flux_peak_vs_ideal_ms" in bp:
        caveats.append(f"beat placement (studio track): compared with onsets exactly on the beat, the kick / bass "
                       f"energy sits {bp['kick_bass_envelope_peak_vs_ideal_ms']:+.1f} ms, the kick attack "
                       f"{bp['kick_bass_flux_peak_vs_ideal_ms']:+.1f} ms and the high-frequency transients "
                       f"{bp['hf_2_8khz_flux_peak_vs_ideal_ms']:+.1f} ms from the beats (beats mark: "
                       f"{bp.get('beats_mark', '?')}); absolute 'ahead of / behind the beat' timing claims should "
                       "state which reference they use (beats_studio_raw.json provenance.beat_placement)")
    pb = al.get("playback", {})
    if abs(pb.get("tempo_change_percent", 0)) > 0.05:
        caveats.append(f"the DJ played the track {pb['tempo_change_percent']:+.2f}% faster than the record "
                       f"({pb.get('interpretation', '')}); song-time BPM {tempo_song['bpm']:.2f} is the record's, "
                       f"reference-time BPM {tempo_song['bpm'] / al['slope']:.2f} is what was danced")
    take = ctx.get("take") or sj["take"]
    return {"adopt": adopt, "verdict": verdict, "reasons": [r for r in reasons if r], "caveats": caveats,
            "how": adoption_steps(take, ctx.get("refine"), tempo_song, ctx.get("capture") or {},
                                  ctx.get("soundtrack"), org),
            "note": "this stage never overwrites work/<take>/beats.json"}


def run_compare(sess: Session, plot: bool = True) -> dict:
    sj = sess.song_json()
    raw = read_json(sess.out / "beats_studio_raw.json")
    studio = np.asarray(raw["reference_time"], float)
    full = np.asarray(raw.get("reference_time_full") or raw["reference_time"], float)
    phone_path = sess.take.work / "beats.json"
    phone = np.asarray(read_json(phone_path), float).reshape(-1, 2)
    comp = compare_grids(studio, phone, sess.window)
    full_all = compare_grids(full, phone)["all"]
    tj = sess.take.work / "beats" / "tempo.json"
    q = sess.take.work / "audio" / "quality.json"
    src = read_json(q).get("source_cam") if q.exists() else None
    cc = sj["alignment"].get("cross_checks") or {}
    comp["phone_source_cam"] = src
    comp["expected_systematic_ms"] = cc.get(src) if src else None
    det = (sj["alignment"].get("cross_check_detail") or {}).get(src) or {}
    comp["expected_drift_ms_per_min"] = det.get("drift_ms_per_min")
    comp["expected_note"] = ("the phone grid is on the source camera's audio mapped through its clock in "
                             f"{Path(sess.take.cfg['alignment']).name}; "
                             "the cross-check measured that clock's error vs the reference audio clock (median and "
                             "drift), i.e. the difference the grids should show if both beat trackers agree")
    comp["tempo"]["studio_song_time"] = {k: raw["tempo_song"][k] for k in ("bpm", "bpm_linear_fit", "measure_s")}
    comp["tempo"]["tempo_factor"] = sj["alignment"]["tempo_factor"]
    if tj.exists():
        t = read_json(tj)
        comp["tempo"]["phone_tempo_json"] = {k: t.get(k) for k in ("bpm", "bpm_linear_fit", "bpm_overlap_linear_fit")}
    refine = refine_effect(studio)
    comp["refine_beats_effect"] = {**(refine or {}), "note": "how far timing.py's default refine_beats (local-linear "
                                   "tempo fit over +-1 measure) would move the studio grid; use timing --no-refine"}
    st_json = sess.out / "studio_reference.json"
    prov = raw.get("provenance") or {}
    org = prov.get("measure_origin") or {}
    ctx = {"take": sess.take.name, "refine": refine, "capture": current_capture(sess.take.work),
           "soundtrack": read_json(st_json) if st_json.exists() else None,
           "measure_origin": org, "beat_placement": prov.get("beat_placement")}
    rec = recommend(sj, comp, raw["tempo_song"], ctx)
    pairs = comp.pop("pairs")
    out = {"version": VERSION, "take": sess.take.name,
           "grids": {"phone": {"file": f"work/{sess.take.name}/beats.json", "n": int(len(phone)),
                               "span": [float(phone[0, 0]), float(phone[-1, 0])],
                               "origin": "dancecap.beats: all-phone beat_this vote on the phone recordings, on the "
                                         f"source camera's (cam{src}) timing"},
                     "studio": {"file": f"work/{sess.take.name}/song/beats_studio.json", "n": int(len(studio)),
                                "span": [float(studio[0, 0]), float(studio[-1, 0])],
                                "origin": "dancecap.song: beat_this on the studio track, affine-mapped; measure 0 = "
                                          "beats.json's measure 0", "measure_origin": org},
                     "studio_full": {"file": f"work/{sess.take.name}/song/beats_studio_full.json", "n": int(len(full)),
                                     "span": [float(full[0, 0]), float(full[-1, 0])],
                                     "full_measure_offset": org.get("full_measure_offset"),
                                     "measure_agreement_with_phone": full_all.get("measure_agreement"),
                                     "measure_shift_mode": full_all.get("measure_shift_mode"),
                                     "note": "the whole soundtrack span; NOT measure-aligned with beats.json (its "
                                             "measure k is beats.json's measure k - full_measure_offset)"}},
           "difference": "phone - studio (ms), each studio entry matched to the nearest phone entry within half an eighth",
           **comp, "recommendation": rec}
    write_json(sess.out / "comparison.json", out)
    a = comp["all"]["stats"]
    print(f"[song] grids: studio {comp['tempo']['studio_in_overlap']['bpm']} BPM vs phone "
          f"{comp['tempo']['phone_in_overlap']['bpm']} BPM; phone - studio median {a['median_ms']:+.1f} ms, |d| p95 "
          f"{a['abs_p95_ms']:.1f} max {a['abs_max_ms']:.1f} ms; slots agree {comp['all']['slot_agreement']:.0%}, "
          f"downbeats {comp['all']['downbeat_agreement']}, measure numbers {comp['all']['measure_agreement']} "
          f"(full-span grid: {full_all.get('measure_agreement')}, shift {full_all.get('measure_shift_mode')}); drift "
          f"{comp['drift'].get('ms_per_min')} ms/min; refine_beats would move the studio grid {refine}")
    if "window" in comp:
        w = comp["window"]["stats"]
        print(f"[song] window {sess.window}: median {w.get('median_ms')} ms, |d| max {w.get('abs_max_ms')} ms, "
              f"slots agree {comp['window'].get('slot_agreement')}, measures agree "
              f"{comp['window'].get('measure_agreement')}")
    print(f"[song] recommendation: {rec['verdict']}")
    for s in rec["how"]:
        print(f"[song]   {s}")
    if plot:
        out["review"] = str(plot_review(sess, sj, out, studio, phone, pairs))
        print(f"[song] wrote {out['review']}")
    return out


# ------------------------------------------------------------------------------------------------ review
def plot_review(sess: Session, sj: dict, comp: dict, studio: np.ndarray, phone: np.ndarray, pairs: dict) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    al = sj["alignment"]
    s, o = al["slope"], al["offset"]
    detail = read_json(sess.out / "align_detail.json")
    curve = detail["coarse_curve"]
    co = al["coarse"]
    Fs, Fr = sess.song_F(), sess.cam_F(sess.ref.id)
    dt = frame_dt(ALIGN_SR)
    xr = sess.cam_x(sess.ref.id)
    win = sess.window
    fig = plt.figure(figsize=(24, 30), dpi=90)
    gs = fig.add_gridspec(6, 2, height_ratios=[1.0, 0.8, 0.8, 1.0, 1.35, 1.0], hspace=0.5, wspace=0.14, top=0.968,
                          bottom=0.03)
    pb = al["playback"]
    fig.suptitle(f"{sess.take.name}: studio track '{sj['song']['title']}' ({sj['song']['artist']}) on the cam"
                 f"{sess.ref.id} audio clock - reference = {s:.7f} x song {o:+.4f} s (DJ tempo "
                 f"{pb['tempo_change_percent']:+.2f}%, pitch {pb['semitones']:+.2f} st)", fontsize=16, y=0.995)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(curve["slopes"], curve["r"], color="tab:blue", lw=1.2)
    ax.axvline(1.0, color="0.4", ls=":", lw=1.2, label=f"slope 1: r {co['r_at_slope_1']:.3f}")
    ax.axvline(co["slope"], color="tab:red", lw=1.2, label=f"best {co['slope']:.5f}: r {co['r']:.3f}")
    ax.set_xlabel("slope (reference s per song s)")
    ax.set_ylabel("whole-track correlation peak")
    ch = al["chroma_check"]
    ax.set_title(f"coarse search over slopes (onset features). chroma at that slope: offset "
                 f"{ch['fit_slope']['offset']:.3f} s ({ch['offset_diff_ms']:+.0f} ms vs fit)", loc="left", fontsize=11)
    ax.legend(fontsize=10)

    an = al["anchors"]
    xa = np.array([a["song_t"] for a in an])
    ya = np.array([a["ref_t"] for a in an])
    inl = np.array([a["inlier"] for a in an])
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(xa[inl], ya[inl] - xa[inl], "o", color="tab:blue", label="anchors (inliers)")
    if (~inl).any():
        ax.plot(xa[~inl], ya[~inl] - xa[~inl], "x", color="red", ms=9, label="anchors (rejected)")
    tt = np.linspace(0, sj["song"]["duration"], 50)
    ax.plot(tt, (s - 1) * tt + o, color="tab:red", lw=1, label=f"fit: slope {s:.6f}")
    ax.plot(tt, np.full_like(tt, co["offset_at_slope_1"]), color="0.4", ls=":",
            label=f"slope-1 best offset {co['offset_at_slope_1']:.3f} s")
    ax.set_xlabel("song time (s)")
    ax.set_ylabel("reference - song (s)")
    ax.set_title("anchor lags drift linearly: the recording runs faster than the record", loc="left", fontsize=11)
    ax.legend(fontsize=10)

    def env(F):
        e = F.sum(0).astype(float)
        return np.convolve(e, np.hanning(7) / np.hanning(7).sum(), "same")  # display only: ~25 ms smoothing

    es, er = env(Fs), env(Fr)
    ts = np.arange(len(es)) * dt
    tr = np.arange(len(er)) * dt
    cov = sj["coverage"]["reference"]
    ex = [(max(cov[0] + 3, 8.0), max(cov[0] + 3, 8.0) + 5), (cov[1] - 9, cov[1] - 4)]
    for row, (name, sl, of) in enumerate((("BEFORE (slope 1, best whole-track offset)", 1.0, co["offset_at_slope_1"]),
                                          ("AFTER (affine fit)", s, o))):
        for col, (a, b) in enumerate(ex):
            ax = fig.add_subplot(gs[1 + row, col])
            m = (tr >= a) & (tr <= b)
            sc = np.percentile(er[m], 99) + 1e-9
            ax.fill_between(tr[m], 0, er[m] / sc, color="0.55", lw=0, label=f"cam{sess.ref.id} onset envelope")
            u = (tr[m] - of) / sl
            sv = np.interp(u, ts, es, left=0, right=0)
            ax.fill_between(tr[m], 0, -sv / (np.percentile(sv, 99) + 1e-9), color="tab:blue" if row else "tab:red",
                            lw=0, alpha=0.8, label="studio onset envelope (mirrored)")
            for t, k in studio[(studio[:, 0] >= a) & (studio[:, 0] <= b)][:, :2]:
                if row and k == 1:
                    ax.axvline(t, color="k", lw=0.6, alpha=0.5)
            ax.set_xlim(a, b)
            ax.set_ylim(-1.3, 1.3)
            ax.set_yticks([])
            ax.set_title(f"{name}: reference {a:.1f}-{b:.1f} s" + (" (black = studio type-1 eighths)" if row else ""),
                         loc="left", fontsize=11)
            if col == 0:
                ax.legend(fontsize=9, loc="upper left")
    ax.set_xlabel("reference time (s)")

    ax = fig.add_subplot(gs[3, 0])
    res = np.array([a["residual_ms"] for a in an])
    ax.plot(xa[inl], res[inl], "o-", color="tab:blue", label="inlier anchors")
    if (~inl).any():
        ax.plot(xa[~inl], res[~inl], "x", color="red", ms=9, label="rejected")
    ax.axhspan(-al["residual_ms_p95"], al["residual_ms_p95"], color="tab:blue", alpha=0.08,
               label=f"+-p95 {al['residual_ms_p95']:.2f} ms")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("song time (s)")
    ax.set_ylabel("residual (ms)")
    ax.set_title(f"direct anchors (studio vs cam{sess.ref.id}): median |r| {al['residual_ms_median']:.2f} ms, p95 "
                 f"{al['residual_ms_p95']:.2f} ms ({al['n_inliers']}/{al['n_anchors']} inliers)", loc="left", fontsize=11)
    ax.legend(fontsize=9)
    ax = fig.add_subplot(gs[3, 1])
    cmap = plt.get_cmap("tab10")
    for k, (c, rows) in enumerate(sorted(detail["cross_check_anchors"].items())):
        if not rows:
            continue
        r = np.asarray([[a[0], a[3]] for a in rows if a[4]])
        if len(r):
            ax.plot(r[:, 0], r[:, 1], "o-", ms=3.5, lw=1, color=cmap(k % 10),
                    label=f"cam{c}: median {al['cross_checks'].get(c)} ms")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("song time (s)")
    al_name = Path(sess.take.cfg["alignment"]).name  # the take's clock file (alignment.json or alignment_studio.json)
    ax.set_ylabel(f"{al_name} clock - studio line (ms)")
    bc = al.get("beats_py_consistency") or {}
    ax.set_title(f"cross-checks: studio aligned to each phone's own audio, mapped through {al_name}"
                 + (f"\n(predicts beats.py's per-phone beat offsets within median {bc.get('median_abs_diff_ms')} ms)"
                    if bc else ""), loc="left", fontsize=11)
    ax.legend(fontsize=8.5, ncol=2)

    # ---- analysis window: both grids over the reference waveform
    ax = fig.add_subplot(gs[4, :])
    a, b = win[0] - 0.4, win[1] + 0.4
    t_ax = np.arange(len(xr)) / ALIGN_SR
    m = (t_ax >= a) & (t_ax <= b)
    xx, tt2 = xr[m], t_ax[m]
    step = max(1, len(xx) // 5000)
    n = len(xx) // step * step
    scl = 1.0 / (np.abs(xx).max() + 1e-9)
    ax.fill_between(tt2[:n:step], scl * xx[:n].reshape(-1, step).min(1), scl * xx[:n].reshape(-1, step).max(1),
                    color="0.75", lw=0, label=f"cam{sess.ref.id} audio (reference)")
    mr = (tr >= a) & (tr <= b)
    ax.plot(tr[mr], -1.0 + 0.8 * er[mr] / (er[mr].max() + 1e-9), color="tab:purple", lw=0.9,
            label=f"cam{sess.ref.id} onset envelope")
    sty = {1: ("tab:red", 2.2), 2: ("tab:green", 1.8), 3: ("0.35", 1.0)}
    for grid, y0, y1, lab in ((phone, 0.05, 1.0, "phone"), (studio[:, :2], -1.0, -0.05, "studio")):
        g = grid[(grid[:, 0] >= a) & (grid[:, 0] <= b)]
        for k in (1, 2, 3):
            gg = g[g[:, 1] == k]
            ax.vlines(gg[:, 0], y0, y1, color=sty[k][0], lw=sty[k][1], ls="-" if lab == "studio" else "--",
                      label=f"{lab} type {k}")
    w = comp.get("window", {})
    for row in w.get("rows", []):
        ax.text(row[0], -1.08, f"{row[3]}", ha="center", va="top", fontsize=9)
        ax.text(row[1], 1.04, f"{row[2]:+.0f}", ha="center", va="bottom", fontsize=8, color="tab:blue")
    ax.axvspan(*win, color="tab:orange", alpha=0.07)
    ax.set_xlim(a, b)
    ax.set_ylim(-1.25, 1.2)
    ax.set_yticks([])
    ws = w.get("stats", {})
    ax.set_title(f"analysis window {win[0]:g}-{win[1]:g} s: phone grid (top, dashed; numbers = phone - studio ms) vs "
                 f"studio grid (bottom, solid; numbers = eighth slot). median {ws.get('median_ms')} ms, |d| max "
                 f"{ws.get('abs_max_ms')} ms, slots agree {w.get('slot_agreement')}", loc="left", fontsize=11)
    ax.legend(fontsize=8.5, ncol=8, loc="upper left", bbox_to_anchor=(0, -0.09))
    ax.set_xlabel("reference time (s)", labelpad=1)

    ax = fig.add_subplot(gs[5, :])
    t, d = np.asarray(pairs["t_studio"]), np.asarray(pairs["d_ms"])
    ty = np.asarray(pairs["type"])
    for k in (3, 2, 1):
        mk = ty == k
        ax.plot(t[mk], d[mk], ".", color=sty[k][0], ms=4 if k == 3 else 6, label=f"type {k}")
    dr = comp["drift"]
    if dr.get("bins"):
        bb = np.asarray(dr["bins"])
        ax.plot(bb[:, 0], bb[:, 1], "-", color="k", lw=1.6, label="20 s medians")
        tt3 = np.array([t.min(), t.max()])
        ax.plot(tt3, dr["intercept_ms_at_t0"] + dr["ms_per_min"] * (tt3 - t.min()) / 60, color="tab:blue", lw=1.2,
                ls="--", label=f"linear drift {dr['ms_per_min']:+.2f} ms/min")
    ax.axvspan(*win, color="tab:orange", alpha=0.12, label="analysis window")
    ax.axhline(0, color="0.4", lw=0.6)
    st = comp["all"]["stats"]
    ax.set_title(f"phone - studio per grid entry over the overlap: median {st['median_ms']:+.1f} ms, |d| p95 "
                 f"{st['abs_p95_ms']:.1f} ms, max {st['abs_max_ms']:.1f} ms; slots agree "
                 f"{comp['all']['slot_agreement']:.0%}, types {comp['all']['type_agreement']:.0%}, measure numbers "
                 f"{comp['all'].get('measure_agreement') or 0:.0%}. "
                 f"Recommendation: {comp['recommendation']['verdict']}", loc="left", fontsize=11)
    ax.set_xlabel("reference time (s)")
    ax.set_ylabel("ms")
    ax.legend(fontsize=9, ncol=6)
    sess.review.mkdir(parents=True, exist_ok=True)
    p = sess.review / "song_alignment.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


# ------------------------------------------------------------------------------------------------ cli
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m dancecap.song",
                                 description="Studio track on the reference clock: alignment, soundtrack, studio beat "
                                             "grid, comparison with the phone grid.")
    ap.add_argument("stage", choices=["align", "soundtrack", "beats", "compare", "all"])
    ap.add_argument("--take", required=True)
    ap.add_argument("--cams", help="comma-separated cross-check cameras (default: every accepted camera but the "
                                   "reference)")
    ap.add_argument("--device", help="beat_this device: cuda or cpu (default cuda when the GPU lock frees up)")
    ap.add_argument("--gpu-wait", type=float, default=600.0, help="seconds to wait for the GPU lock")
    ap.add_argument("--sr-out", type=int, default=48000, help="soundtrack sample rate")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args(argv)
    cfg = SongConfig(cams=tuple(c.strip() for c in args.cams.split(",")) if args.cams else None, device=args.device,
                     gpu_wait_s=args.gpu_wait, out_sr=args.sr_out)
    sess = Session(args.take, cfg)
    stages = ["align", "soundtrack", "beats", "compare"] if args.stage == "all" else [args.stage]
    for st in stages:
        if st == "align":
            run_align(sess)
        elif st == "soundtrack":
            run_soundtrack(sess)
        elif st == "beats":
            run_beats(sess)
        elif st == "compare":
            run_compare(sess, plot=not args.no_plot)


if __name__ == "__main__":
    main()
