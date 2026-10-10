"""Audio clock alignment of the class-recap phones against the master camera (GCC-PHAT), for the take larissa_kadu_recap.

The recap is ONE six-minute lesson filmed by seven unsynchronised phones. The owner supplied camera-time-offsets.json
(master_t = video_t + offset, the master being 01-Recap-Tina.mp4, offset 0). This module VERIFIES those offsets from the
phones' own audio (the teachers talking), measures clock drift between the phones, looks for the one camera the JSON does
not list (04-Recap-Tony.mp4, a 77 s clip), ranks the audio quality of every phone and writes the numbers the take needs:

    python -m dancecap.recap_sync run [--src DIR] [--out DIR] [--alignment-out FILE] [--no-search] [--no-pose]
    python -m dancecap.recap_sync audio [--src DIR] [--out DIR]       only extract the 16 kHz mono wavs (ffmpeg)
    python -m dancecap.recap_sync plot [--out DIR]                    review/sync_overview.png from sync/sync.json

Convention (the same as Take / Source): reference_time = slope * source_time + offset, the reference clock being the
master camera's own video time (source_time = the video's PTS; every file here starts at PTS 0). With drift the fit is
affine (slope != 1) like prototype/larissa_kadu/alignment_studio.json.

METHOD. Sound travels from the speakers to every phone's microphone, so the same speech is in every file.
  1. WIDE pass: windows of the camera's audio are located in the WHOLE master by GCC-PHAT (phase transform, band
     limited to speech: delay d with master(t) = cam(t - d), offset = d - window start). It needs no prior; it is how
     the sign convention of the owner's numbers is checked and how an unlisted camera would be found.
  2. NARROW pass: 8 s windows every 4 s are searched +-0.3 s around the wide result. A window is accepted when its peak
     z-score (peak over the standard deviation of the correlation outside +-10 ms of it) is >= 6. Several windows give
     a different offset because the microphones are at different distances from the loudspeakers and the live voices
     (3 ms per metre): the offsets of the accepted windows are clustered (5 ms bins) and the densest cluster is the
     clock; the other clusters are reported (`clusters`) as the acoustic uncertainty of the offset.
  3. A robust straight line through the main cluster gives the affine clock; it is used only when the drift is
     significant (> 3 standard errors AND > 2 ms over the span), else a constant offset.
  4. A camera that the wide pass cannot place goes through `search_unlisted`: a speech-envelope correlation (10 s
     windows) against all the placed phones at once, plus a coarse time-scale scan, together with the same search on a
     known weak camera as a control (detection power). The verdict is written down either way.
  5. Audio quality: per 30 s window of the master clock the speech-band level, the noise floor, an SNR proxy and the
     agreement of the phone's speech envelope with the consensus of the other phones (env_ncc).
Limits: audio and video of one phone are assumed to be in step (their container start times and durations are
recorded here; the geometric check is dancecap.sync, later, and it will measure the per-phone VIDEO offsets of tens of ms
that A/V latency and sound travel leave).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

from .take import PACKAGE_ROOT, WORKSPACE

SR = 16000
BAND = (150.0, 4000.0)          # Hz, the speech band the correlation uses
QUALITY_BAND = (300.0, 3400.0)  # Hz, the band the audio-quality table uses
WIDE_WIN_S = 12.0               # wide-pass window
WIDE_N = 14                     # windows per camera in the wide pass
NARROW_WIN_S = 8.0              # narrow-pass window
NARROW_HOP_S = 4.0
NARROW_SEARCH_S = 0.3           # +- search around the wide result
MIN_Z_WIDE = 9.0                # a wide window counts when its peak z-score is at least this ...
MIN_PSR_WIDE = 1.3              # ... and its peak is at least this much above the next peak (>= 20 ms away)
MIN_Z_NARROW = 6.0              # narrow windows: the noise maximum over 9600 lags is z 3.9-4.5
CLUSTER_BIN_S = 0.005
CLUSTER_HALF_S = 0.012
TAPER_S = 0.05                  # Hann edges of every correlation window (no edge artefacts)
ENV_HOP_S = 0.01
QUALITY_WIN_S = 30.0
MASTER_ID = "01"
RECAP_DIR = WORKSPACE / "dances" / "Larissa-Kadu-recap"
WORK = PACKAGE_ROOT / "work" / "larissa_kadu_recap"
ALIGNMENT_OUT = WORKSPACE / "prototype" / "larissa_kadu" / "alignment_recap.json"
FILE_RE = re.compile(r"^(\d{2})-Recap-([A-Za-z]+)\.mp4$")


# ----------------------------------------------------------------------------------------------- audio io


def read_wav(path: Path) -> np.ndarray:
    """16-bit mono wav -> float32 in [-1, 1]."""
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2 or w.getnchannels() != 1:
            raise ValueError(f"{path}: expected 16-bit mono")
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0


def extract_audio(video: Path, wav: Path, sr: int = SR) -> Path:
    """First audio stream -> mono sr Hz 16-bit wav (ffmpeg, light)."""
    wav.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(video), "-map", "0:a:0", "-ac", "1", "-ar", str(sr),
                    "-c:a", "pcm_s16le", str(wav)], check=True)
    return wav


# ----------------------------------------------------------------------------------------------- GCC-PHAT


def taper(x: np.ndarray, seconds: float = TAPER_S, fs: int = SR) -> np.ndarray:
    """Hann ramps on both ends (a hard edge is broadband and PHAT whitening turns it into a correlation peak)."""
    n = min(int(seconds * fs), len(x) // 2)
    if n < 2:
        return x
    w = np.hanning(2 * n)
    y = np.asarray(x, np.float32).copy()
    y[:n] *= w[:n]
    y[-n:] *= w[n:]
    return y


def _nfft(n: int) -> int:
    return 1 << max(4, int(np.ceil(np.log2(max(n, 2)))))


def _phat(X: np.ndarray, Y: np.ndarray, n: int, fs: int, band: tuple[float, float]) -> np.ndarray:
    G = Y * np.conj(X)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    mag = np.abs(G)
    keep = (f >= band[0]) & (f <= band[1])
    floor = 1e-3 * (mag[keep].mean() if keep.any() else 1.0)
    G = np.where(keep, G / (mag + floor), 0.0)
    return np.fft.irfft(G, n)


def gcc_phat_spectrum(x: np.ndarray, y: np.ndarray, fs: int = SR, band: tuple[float, float] = BAND) -> tuple[np.ndarray, int]:
    """Cross-correlation of y against x with PHAT weighting inside `band`: returns (cc, nfft); cc[n] is the correlation
    at lag +n samples (cc[-n] the negative lags): a peak at +d means y(t) = x(t - d) (y is x delayed by d samples)."""
    n = _nfft(len(x) + len(y))
    return _phat(np.fft.rfft(taper(x, fs=fs), n), np.fft.rfft(taper(y, fs=fs), n), n, fs, band), n


def pick_delay(cc: np.ndarray, n: int, fs: int, lag_lo_s: float, lag_hi_s: float) -> dict:
    """Highest positive peak of cc inside [lag_lo_s, lag_hi_s] (sub-sample by a parabola) and its confidence."""
    lo, hi = int(np.floor(lag_lo_s * fs)), int(np.ceil(lag_hi_s * fs))
    idx = np.arange(lo, hi + 1)
    seg = cc[idx % n]
    if seg.size < 8 or not np.isfinite(seg).all():
        return {"delay_s": float("nan"), "z": 0.0, "psr": 0.0, "peak": 0.0}
    j = int(np.argmax(seg))
    peak = float(seg[j])
    if 0 < j < len(seg) - 1:
        a, b, c = seg[j - 1], seg[j], seg[j + 1]
        den = a - 2 * b + c
        frac = 0.5 * (a - c) / den if den < 0 else 0.0
    else:
        frac = 0.0
    guard = int(0.010 * fs)
    far = np.ones(len(seg), bool)
    far[max(0, j - guard):j + guard + 1] = False
    sd = float(seg[far].std()) if far.any() else 1.0
    far2 = np.ones(len(seg), bool)
    far2[max(0, j - 2 * guard):j + 2 * guard + 1] = False
    second = float(seg[far2].max()) if far2.any() else 0.0
    return {"delay_s": float((idx[j] + frac) / fs), "z": peak / sd if sd > 0 else 0.0,
            "psr": peak / second if second > 0 else 99.0, "peak": peak}


def estimate_delay(x: np.ndarray, y: np.ndarray, fs: int = SR, lag_s: tuple[float, float] | None = None,
                   band: tuple[float, float] = BAND) -> dict:
    """y(t) = x(t - d): d in seconds (positive: y later) with confidence; lag_s limits the search (default: any overlap)."""
    cc, n = gcc_phat_spectrum(x, y, fs, band)
    lo, hi = lag_s if lag_s is not None else (-len(x) / fs, len(y) / fs)
    return pick_delay(cc, n, fs, lo, hi)


class MasterIndex:
    """The master's spectrum computed once, so many windows can be located in it cheaply."""

    def __init__(self, y: np.ndarray, max_window_s: float, fs: int = SR, band: tuple[float, float] = BAND):
        self.y, self.fs, self.band = y, fs, band
        self.n = _nfft(len(y) + int(max_window_s * fs))
        self.Y = np.fft.rfft(taper(y, 0.5, fs), self.n)

    def locate(self, x: np.ndarray, lag_s: tuple[float, float] | None = None) -> dict:
        """Delay d (s) of the master relative to window x: master(t) = x(t - d), so master time = window time + d.
        The default search keeps the whole window inside the master (no partial overlaps, no edge artefacts)."""
        X = np.fft.rfft(taper(x, fs=self.fs), self.n)
        cc = _phat(X, self.Y, self.n, self.fs, self.band)
        lo, hi = lag_s if lag_s is not None else (0.0, len(self.y) / self.fs - len(x) / self.fs)
        return pick_delay(cc, self.n, self.fs, lo, hi)


# ----------------------------------------------------------------------------------------------- offsets of one camera


def good_wide(r: dict, min_z: float = MIN_Z_WIDE, min_psr: float = MIN_PSR_WIDE) -> bool:
    return bool(np.isfinite(r["delay_s"]) and r["z"] >= min_z and r["psr"] >= min_psr)


def window_starts(duration_s: float, win_s: float, n: int, lo: float = 0.0) -> np.ndarray:
    """n window start times spread over [lo, duration - win] (first at the start, last ending at the end)."""
    last = duration_s - win_s
    if last <= lo:
        return np.array([lo])
    return np.linspace(lo, last, n)


def wide_offsets(cam: np.ndarray, index: MasterIndex, n: int = WIDE_N, win_s: float = WIDE_WIN_S, fs: int = SR) -> list[dict]:
    """Offset of the camera (master_t = cam_t + offset) from n windows, each searched against the WHOLE master."""
    W = int(win_s * fs)
    rows = []
    for u0 in window_starts(len(cam) / fs, win_s, n):
        x = cam[int(u0 * fs):int(u0 * fs) + W]
        r = index.locate(x)
        rows.append({"u0": float(u0), "offset_s": r["delay_s"] - float(u0), "z": r["z"], "psr": r["psr"], "ok": good_wide(r)})
    return rows


def vote(rows: list[dict], bin_s: float = 0.05) -> dict:
    """Mode of the good windows' offsets (histogram bins of bin_s, then the median of the winning bin and its neighbours)."""
    ok = [r["offset_s"] for r in rows if r["ok"]]
    if not ok:
        return {"offset_s": None, "votes": 0, "windows": len(rows)}
    v = np.array(ok)
    bins = np.round(v / bin_s).astype(int)
    vals, counts = np.unique(bins, return_counts=True)
    best = vals[int(np.argmax(counts))]
    sel = v[np.abs(bins - best) <= 1]
    return {"offset_s": float(np.median(sel)), "votes": int(len(sel)), "windows": len(rows),
            "spread_ms": float(1000 * (sel.max() - sel.min()))}


def narrow_rows(cam: np.ndarray, master: np.ndarray, center_s: float, win_s: float = NARROW_WIN_S, hop_s: float = NARROW_HOP_S,
                search_s: float = NARROW_SEARCH_S, fs: int = SR, min_z: float = MIN_Z_NARROW, band: tuple[float, float] = BAND
                ) -> list[dict]:
    """Offset estimates for windows of the camera (hop_s apart), each searched only +-search_s around `center_s`.

    Row: u0 (window start in the camera), t_cam (its centre), t_master (centre on the master clock), offset_s (master_t
    - cam_t at the window), z, psr, ok (z >= min_z)."""
    W = int(win_s * fs)
    rows = []
    for u0 in np.arange(max(0.0, -center_s), len(cam) / fs - win_s, hop_s):
        x = cam[int(u0 * fs):int(u0 * fs) + W]
        m0 = int(round((u0 + center_s - search_s) * fs))      # first master sample of the slice
        m1 = m0 + W + int(2 * search_s * fs)
        if len(x) < W or m0 < 0 or m1 > len(master):
            continue
        r = estimate_delay(x, master[m0:m1], fs, lag_s=(0.0, 2 * search_s), band=band)
        off = m0 / fs + r["delay_s"] - float(u0)
        rows.append({"u0": float(u0), "t_cam": float(u0 + win_s / 2), "t_master": float(u0 + off + win_s / 2),
                     "offset_s": float(off), "z": float(r["z"]), "psr": float(r["psr"]), "ok": bool(r["z"] >= min_z)})
    return rows


def densest_cluster(values: np.ndarray, bin_s: float = CLUSTER_BIN_S, half_s: float = CLUSTER_HALF_S) -> np.ndarray:
    """Boolean mask of the values within +-half_s of the densest 3-bin neighbourhood (bins of bin_s)."""
    if len(values) == 0:
        return np.zeros(0, bool)
    bins = np.round(values / bin_s).astype(int)
    vals, counts = np.unique(bins, return_counts=True)
    score = np.array([counts[np.abs(vals - v) <= 1].sum() for v in vals])
    best = vals[int(np.argmax(score))]
    centre = float(np.median(values[np.abs(bins - best) <= 1]))
    return np.abs(values - centre) <= half_s


def secondary_clusters(values: np.ndarray, main: np.ndarray, bin_s: float = CLUSTER_BIN_S, min_n: int = 2) -> list[dict]:
    """Clusters (>= min_n windows) among the values the main cluster left out: [{offset_ms_from_main, n}]."""
    rest = values[~main]
    out = []
    ref = float(np.median(values[main])) if main.any() else 0.0
    while len(rest) >= min_n:
        m = densest_cluster(rest, bin_s)
        if m.sum() < min_n:
            break
        out.append({"offset_ms_from_main": round(1000 * (float(np.median(rest[m])) - ref), 1), "n": int(m.sum())})
        rest = rest[~m]
    return out


def fit_clock(t_cam: np.ndarray, t_master: np.ndarray, n_iter: int = 8, clip_s: float = 0.006) -> dict:
    """Robust straight line master_t = slope * cam_t + offset through (t_cam, t_master) points (window centres).

    Iteratively drops points further than max(clip_s, 3 x MAD) from the line. Returns the affine clock AND the constant
    offset (slope 1) with their residual scatter (rms and 95 % of |residual|, ms), the drift (ppm and total ms over the
    fitted span) and the standard error of the slope, so the caller can judge whether the slope is real."""
    n = len(t_cam)
    if n < 3:
        return {"ok": False, "n": n}
    v, m = np.asarray(t_cam, float), np.asarray(t_master, float)
    keep = np.ones(n, bool)
    slope, off = 1.0, float(np.median(m - v))
    for _ in range(n_iter):
        if keep.sum() >= 3 and np.ptp(v[keep]) > 20.0:
            A = np.stack([v[keep] - v[keep].mean(), np.ones(keep.sum())], 1)
            c = np.linalg.lstsq(A, m[keep], rcond=None)[0]
            slope, off = float(c[0]), float(c[1] - c[0] * v[keep].mean())
        else:
            slope, off = 1.0, float(np.median(m[keep] - v[keep]))
        res = m - (slope * v + off)
        mad = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep])))
        new = np.abs(res) <= max(clip_s, 3 * mad)
        if (new == keep).all():
            break
        keep = new if new.sum() >= 3 else keep
    res = m - (slope * v + off)
    const_off = float(np.median(m[keep] - v[keep]))
    res_c = m - (v + const_off)
    span = float(np.ptp(v[keep]))
    rms = float(np.sqrt(np.mean(res[keep] ** 2)))
    se = rms / (np.sqrt(keep.sum()) * max(float(np.std(v[keep])), 1e-9))
    return {"ok": True, "n": int(n), "n_inliers": int(keep.sum()), "slope": slope, "offset_s": off,
            "slope_se_ppm": float(1e6 * se), "drift_ppm": float((slope - 1.0) * 1e6),
            "drift_total_ms": float(1000 * (slope - 1.0) * span), "span_s": span,
            "residual_rms_ms": 1000 * rms, "residual_p95_ms": float(1000 * np.percentile(np.abs(res[keep]), 95)),
            "const_offset_s": const_off, "const_residual_rms_ms": float(1000 * np.sqrt(np.mean(res_c[keep] ** 2))),
            "const_residual_p95_ms": float(1000 * np.percentile(np.abs(res_c[keep]), 95)), "inlier_mask": keep.tolist()}


def decide_clock(fit: dict, min_total_ms: float = 2.0, min_sigmas: float = 3.0, min_span_s: float = 120.0) -> dict:
    """Affine only when the drift is real: > min_sigmas standard errors, > min_total_ms over the span, span >= min_span_s."""
    se = fit.get("slope_se_ppm") or 0.0
    real = (fit.get("ok") and fit["n_inliers"] >= 8 and fit["span_s"] >= min_span_s
            and abs(fit["drift_total_ms"]) > min_total_ms and abs(fit["drift_ppm"]) > min_sigmas * se)
    if real:
        return {"clock": "affine", "slope": fit["slope"], "offset_s": fit["offset_s"], "residual_ms": fit["residual_rms_ms"]}
    return {"clock": "constant", "slope": 1.0, "offset_s": fit["const_offset_s"], "residual_ms": fit["const_residual_rms_ms"]}


def measure_camera(cam: np.ndarray, master: np.ndarray, center_s: float, fs: int = SR) -> dict:
    """Narrow pass + cluster + clock decision for one camera whose offset is known to about +-0.1 s (the wide vote)."""
    rows = narrow_rows(cam, master, center_s, fs=fs)
    good = [r for r in rows if r["ok"]]
    out = {"windows": len(rows), "good_windows": len(good), "rows": rows}
    if len(good) < 3:
        out["ok"] = False
        return out
    off = np.array([r["offset_s"] for r in good])
    main = densest_cluster(off)
    t_cam = np.array([r["t_cam"] for r in good])[main]
    t_master = np.array([r["t_master"] for r in good])[main]
    fit = fit_clock(t_cam, t_master)
    if not fit.get("ok"):
        out["ok"] = False
        return out
    inl = np.array(fit["inlier_mask"], bool)
    # windows of the main cluster that the line rejected, and every other accepted window, are reported as such
    in_main = set(np.flatnonzero(main)[inl].tolist())
    for k, r in enumerate(good):
        r["cluster"] = "main" if k in in_main else "other"
    out.update({"ok": True, "fit": {k: v for k, v in fit.items() if k != "inlier_mask"}, "decision": decide_clock(fit),
                "n_main": int(inl.sum()), "clusters": secondary_clusters(off, main), "good_rows": good,
                "z_median": float(np.median([r["z"] for r in good]))})
    return out


# ----------------------------------------------------------------------------------------------- speech envelopes


def band_limited(x: np.ndarray, band: tuple[float, float] = QUALITY_BAND, fs: int = SR) -> np.ndarray:
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1.0 / fs)
    X[(f < band[0]) | (f > band[1])] = 0
    return np.fft.irfft(X, len(x))


def frame_db(x: np.ndarray, frame: int = 800) -> np.ndarray:
    """RMS level in dBFS of consecutive frames (800 samples = 50 ms)."""
    n = len(x) // frame
    if n == 0:
        return np.zeros(0)
    f = x[:n * frame].reshape(n, frame)
    return 20 * np.log10(np.sqrt((f ** 2).mean(1)) + 1e-9)


def log_envelope(x: np.ndarray, fs: int = SR, band: tuple[float, float] = QUALITY_BAND, hop_s: float = ENV_HOP_S) -> np.ndarray:
    """Speech-band log envelope at 100 Hz with the slow level changes (1 s moving average) removed: the syllable rhythm,
    which different microphones in one room share even when their waveforms (phases, reverberation) differ."""
    xb = band_limited(x, band, fs)
    n = int(hop_s * fs)
    m = len(xb) // n
    e = np.log(np.sqrt((xb[:m * n].reshape(m, n) ** 2).mean(1)) + 1e-6)
    k = max(1, int(1.0 / hop_s))
    return e - np.convolve(e, np.ones(k) / k, mode="same")


def xcorr_norm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Normalised cross-correlation of the short signal a at every shift inside the long signal b (valid shifts only)."""
    n = _nfft(len(a) + len(b))
    num = np.fft.irfft(np.fft.rfft(b, n) * np.conj(np.fft.rfft(a - a.mean(), n)), n)[:len(b) - len(a) + 1]
    c1 = np.concatenate([[0.0], np.cumsum(b)])
    c2 = np.concatenate([[0.0], np.cumsum(b ** 2)])
    L = len(a)
    s1, s2 = c1[L:] - c1[:-L], c2[L:] - c2[:-L]
    sd = np.sqrt(np.maximum(s2 / L - (s1 / L) ** 2, 1e-12))
    return num / (L * sd * max(float(a.std()), 1e-9))


def envelope_on_master(env: np.ndarray, slope: float, offset_s: float, t0: float, t1: float, hop_s: float = ENV_HOP_S) -> np.ndarray:
    """The camera's envelope sampled on the master grid t0 + j * hop_s (NaN outside the recording)."""
    t = t0 + np.arange(int(round((t1 - t0) / hop_s))) * hop_s
    tc = (t - offset_s) / slope
    k = tc / hop_s
    out = np.interp(k, np.arange(len(env)), env, left=np.nan, right=np.nan)
    return out


def consensus_scan(target_env: np.ndarray, refs: dict[str, tuple[np.ndarray, float, float]], seconds: tuple[float, float],
                   win_s: float = 10.0, hop_s: float = 5.0, scale: float = 1.0, t_range: tuple[float, float] = (-12.0, 365.0),
                   min_cams: int = 2, hop: float = ENV_HOP_S) -> list[dict]:
    """Locate windows of an unplaced camera's envelope in the placed cameras' envelopes, all at once.

    refs = {id: (envelope, slope, offset)}. For each window the normalised correlations against every ref are put on the
    master grid and averaged over the refs that cover the lag; the best lag is reported with its z-score (over the lags
    of that window). scale != 1 stretches the target's time axis by that factor (test of a time-scale error)."""
    T0, T1 = t_range
    G = int((T1 - T0) / hop)
    rows = []
    for u0 in np.arange(seconds[0], seconds[1] - win_s + 1e-6, hop_s):
        seg = target_env[int(u0 / hop):int((u0 + win_s) / hop)]
        if scale != 1.0:
            seg = np.interp(np.linspace(0, len(seg) - 1, int(len(seg) * scale)), np.arange(len(seg)), seg)
        acc, cnt = np.zeros(G), np.zeros(G)
        for _cid, (e, slope, off) in refs.items():
            if len(e) <= len(seg):
                continue
            ncc = xcorr_norm(seg, e)
            mt = (np.arange(len(ncc)) * hop) * slope + off          # master time of the window start
            gi = np.round((mt - T0) / hop).astype(int)
            m = (gi >= 0) & (gi < G)
            np.add.at(acc, gi[m], ncc[m])
            np.add.at(cnt, gi[m], 1)
        ok = cnt >= min_cams
        if not ok.any():
            continue
        avg = np.where(ok, acc / np.maximum(cnt, 1), np.nan)
        mu, sd = np.nanmean(avg), np.nanstd(avg)
        j = int(np.nanargmax(avg))
        rows.append({"u0": float(u0), "offset_s": float(T0 + j * hop - scale * u0), "z": float((avg[j] - mu) / sd),
                     "n_cams": int(cnt[j]), "scale": scale})
    return rows


def search_unlisted(target_env: np.ndarray, refs: dict[str, tuple[np.ndarray, float, float]], duration_s: float,
                    control_env: np.ndarray | None = None, control_refs: dict | None = None, control_expect_s: float | None = None,
                    scales: tuple[float, float, float] = (0.9, 1.2, 0.01)) -> dict:
    """Is there ANY offset at which the camera's speech envelope matches the placed phones? -> verdict + evidence.

    Windows of 10 s every 5 s are located against all placed phones (valid lags only); found = at least 3 windows with
    z >= 5.5 agreeing within 30 ms. The same scan on a known weak camera (`control_env`, offset known) tells what a real
    match looks like (z of its windows at the true offset); a coarse scan of the time scale (target stretched by 0.9-1.2)
    tells whether a sample-rate / speed error hides the match."""
    rows = consensus_scan(target_env, refs, (0.0, duration_s))
    zs = np.array([r["z"] for r in rows])
    offs = np.array([r["offset_s"] for r in rows])
    cand = None
    hit = zs >= 5.5
    if hit.sum() >= 3:
        m = densest_cluster(offs[hit], 0.01, 0.03)
        if m.sum() >= 3:
            cand = float(np.median(offs[hit][m]))
    out = {"found": cand is not None, "offset_s": cand, "windows": len(rows), "z_max": float(zs.max()) if len(zs) else 0.0,
           "z_median": float(np.median(zs)) if len(zs) else 0.0,
           "best_windows": [{"u0": r["u0"], "offset_s": round(r["offset_s"], 2), "z": round(r["z"], 2)} for r in
                            sorted(rows, key=lambda r: -r["z"])[:5]]}
    if control_env is not None and control_refs is not None and control_expect_s is not None:
        crow = consensus_scan(control_env, control_refs, (0.0, min(len(control_env) * ENV_HOP_S, 100.0)), hop_s=20.0)
        right = [r["z"] for r in crow if abs(r["offset_s"] - control_expect_s) < 0.05]
        wrong = [r["z"] for r in crow if abs(r["offset_s"] - control_expect_s) >= 0.05]
        out["control"] = {"windows": len(crow), "windows_at_true_offset": len(right),
                          "z_at_true_offset": [round(z, 1) for z in right], "z_elsewhere_max": round(max(wrong), 1) if wrong else None}
    lo, hi, st = scales
    seg_end = min(duration_s, 40.0)
    best = []
    for s in np.arange(lo, hi + 1e-9, st):
        r = consensus_scan(target_env, refs, (0.0, seg_end), win_s=seg_end, hop_s=seg_end, scale=float(s))
        if r:
            best.append((r[0]["z"], float(s), r[0]["offset_s"]))
    if best:
        best.sort(reverse=True)
        out["scale_scan"] = {"scales": [lo, hi, st], "segment_s": seg_end, "z_max": round(best[0][0], 2),
                             "z_median": round(float(np.median([b[0] for b in best])), 2),
                             "best": [{"scale": round(b[1], 3), "offset_s": round(b[2], 2), "z": round(b[0], 2)} for b in best[:3]]}
    return out


# ----------------------------------------------------------------------------------------------- audio quality


def quality_windows(x: np.ndarray, slope: float, offset_s: float, t_lo: float, t_hi: float, win_s: float = QUALITY_WIN_S,
                    fs: int = SR) -> list[dict]:
    """Speech-band level / noise floor / SNR proxy per `win_s` window of the MASTER clock (windows fully inside [t_lo, t_hi]).

    speech_db = 90th percentile of the 50 ms frame levels in the speech band, noise_db = 10th percentile, snr_db their
    difference, active_frac = share of frames more than 10 dB over the noise floor (speech or other loud sounds),
    clip_frac = share of samples above 0.98 of full scale."""
    xb = band_limited(x)
    rows = []
    t = np.ceil(max(t_lo, offset_s) / win_s) * win_s + 0.0
    while t + win_s <= t_hi + 1e-6:
        a = int(round((t - offset_s) / slope * fs))
        b = a + int(win_s * fs)
        if a >= 0 and b <= len(x):
            db = frame_db(xb[a:b])
            noise, speech = float(np.percentile(db, 10)), float(np.percentile(db, 90))
            raw = x[a:b]
            rows.append({"t0": float(t), "t1": float(t + win_s), "speech_db": round(speech, 1), "noise_db": round(noise, 1),
                         "snr_db": round(speech - noise, 1), "active_frac": round(float((db > noise + 10).mean()), 3),
                         "rms_db": round(float(20 * np.log10(np.sqrt((raw ** 2).mean()) + 1e-9)), 1),
                         "clip_frac": round(float((np.abs(raw) > 0.98).mean()), 5)})
        t += win_s
    return rows


def add_agreement(quality: dict[str, list[dict]], envs: dict[str, tuple[np.ndarray, float, float]], t0: float = -12.0,
                  t1: float = 365.0, min_others: int = 2) -> None:
    """env_ncc and rank per quality window: the correlation of the phone's speech envelope with the mean (z-scored)
    envelope of the other phones that cover the same window; rank 1 = highest env_ncc (ties and NaN last)."""
    grid = {cid: envelope_on_master(e, s, o, t0, t1) for cid, (e, s, o) in envs.items()}
    for cid, rows in quality.items():
        for r in rows:
            a, b = int(round((r["t0"] - t0) / ENV_HOP_S)), int(round((r["t1"] - t0) / ENV_HOP_S))
            mine = grid[cid][a:b]
            others = []
            for oid, g in grid.items():
                if oid == cid:
                    continue
                seg = g[a:b]
                if np.isfinite(seg).all() and seg.std() > 0:
                    others.append((seg - seg.mean()) / seg.std())
            if np.isfinite(mine).all() and mine.std() > 0 and len(others) >= min_others:
                ref = np.mean(others, axis=0)
                r["env_ncc"] = round(float(np.corrcoef(mine, ref)[0, 1]), 3)
                r["n_others"] = len(others)
            else:
                r["env_ncc"] = None
                r["n_others"] = len(others)
    slots: dict[float, list[tuple[str, float]]] = {}
    for cid, rows in quality.items():
        for r in rows:
            slots.setdefault(r["t0"], []).append((cid, r["env_ncc"] if r["env_ncc"] is not None else -9.0))
    for t, lst in slots.items():
        lst.sort(key=lambda p: -p[1])
        for rank, (cid, _v) in enumerate(lst, 1):
            next(r for r in quality[cid] if r["t0"] == t)["rank"] = rank


def quality_summary(quality: dict[str, list[dict]]) -> dict:
    out = {}
    for cid, rows in quality.items():
        if not rows:
            continue
        ncc = [r["env_ncc"] for r in rows if r.get("env_ncc") is not None]
        out[cid] = {"windows": len(rows), "snr_db_median": round(float(np.median([r["snr_db"] for r in rows])), 1),
                    "speech_db_median": round(float(np.median([r["speech_db"] for r in rows])), 1),
                    "noise_db_median": round(float(np.median([r["noise_db"] for r in rows])), 1),
                    "env_ncc_median": round(float(np.median(ncc)), 3) if ncc else None,
                    "rank1_windows": int(sum(1 for r in rows if r.get("rank") == 1)),
                    "clip_frac_max": max(r["clip_frac"] for r in rows)}
    order = sorted(out, key=lambda c: -(out[c]["env_ncc_median"] if out[c]["env_ncc_median"] is not None else -9))
    return {"per_camera": out, "order_by_env_ncc": order}


# ----------------------------------------------------------------------------------------------- probes and sources


def ffprobe_facts(path: Path) -> dict:
    """Stored/display size, rotation, fps, frames, durations and container start times of one video."""
    r = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
                       capture_output=True, text=True, check=True)
    d = json.loads(r.stdout)
    v = next(s for s in d["streams"] if s.get("codec_type") == "video" and s.get("width"))
    a = next((s for s in d["streams"] if s.get("codec_type") == "audio"), None)
    rot = 0
    for sd in v.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(round(float(sd["rotation"])))
    if not rot and (v.get("tags") or {}).get("rotate"):
        rot = -int(round(float(v["tags"]["rotate"])))
    w, h = int(v["width"]), int(v["height"])
    dw, dh = (h, w) if abs(rot) % 180 == 90 else (w, h)
    num, den = (int(x) for x in v["avg_frame_rate"].split("/"))
    return {"stored_size": [w, h], "display_size": [dw, dh], "rotation": rot, "codec": v.get("codec_name"),
            "fps": round(num / den, 3) if den else None, "frames": int(v.get("nb_frames") or 0),
            "video_duration_s": float(v["duration"]), "video_start_s": float(v.get("start_time", 0.0)),
            "audio": None if a is None else {"codec": a.get("codec_name"), "sample_rate": int(a["sample_rate"]),
                                             "channels": int(a.get("channels", 0)), "duration_s": float(a["duration"]),
                                             "start_s": float(a.get("start_time", 0.0))}}


def pose_json_facts(path: Path, stored: list[int], display: list[int]) -> dict:
    """Frame count, persons per frame and the coordinate range of a per-video 2D pose JSON (list over frames of persons of
    17 {X, Y, Z} keypoints), and whether the pixels are in the STORED or the DISPLAYED (rotation applied) orientation."""
    d = json.loads(path.read_text(encoding="utf-8"))
    cnt = np.array([len(f) for f in d])
    xs, ys = [], []
    for f in d[::5]:
        for p in f:
            for k in p:
                if k["Z"] > 0.3:
                    xs.append(k["X"])
                    ys.append(k["Y"])
    xs, ys = np.array(xs), np.array(ys)
    xmax, ymax = float(xs.max()), float(ys.max())

    def fits(size):
        return xmax <= size[0] * 1.03 + 8 and ymax <= size[1] * 1.03 + 8

    if stored == display:
        orient = "display (= stored: no rotation)"
    elif fits(display) and not fits(stored):
        orient = "display (rotation applied)"
    elif fits(stored) and not fits(display):
        orient = "stored"
    else:
        orient = "undecided"
    return {"frames": int(len(d)), "persons_per_frame": {"median": float(np.median(cnt)), "p10": float(np.percentile(cnt, 10)),
                                                         "p90": float(np.percentile(cnt, 90)), "max": int(cnt.max())},
            "x_range": [round(float(xs.min()), 1), round(xmax, 1)], "y_range": [round(float(ys.min()), 1), round(ymax, 1)],
            "orientation": orient}


def discover(src: Path) -> list[dict]:
    """The numbered videos of the recap folder: [{id, name, role, path}] sorted by id."""
    out = []
    for p in sorted(src.glob("*.mp4")):
        m = FILE_RE.match(p.name)
        if m:
            out.append({"id": m.group(1), "name": p.name, "role": m.group(2), "path": p})
    return out


def read_prior_offsets(src: Path) -> dict[str, float]:
    """camera-time-offsets.json by FILE NAME (its keys are Windows paths of another machine)."""
    p = src / "camera-time-offsets.json"
    raw = json.loads(p.read_text(encoding="utf-8"))
    return {k.replace("\\", "/").rsplit("/", 1)[-1]: float(v) for k, v in raw.items()}


# ----------------------------------------------------------------------------------------------- coverage and documents


def coverage_timeline(cams: list[dict], window: tuple[float, float]) -> dict:
    """Which cameras record at each master second inside the window: per-second counts and the runs of equal membership."""
    t0, t1 = window
    secs = int(np.ceil(t1 - t0))
    members = []
    for i in range(secs):
        a, b = t0 + i, min(t0 + i + 1, t1)
        mid = (a + b) / 2
        members.append([c["id"] for c in cams if c["coverage_master_s"][0] <= mid <= c["coverage_master_s"][1]])
    counts = [len(m) for m in members]
    runs = []
    for i, m in enumerate(members):
        if runs and runs[-1]["cams"] == m:
            runs[-1]["t1"] = t0 + i + 1
        else:
            runs.append({"t0": t0 + i, "t1": t0 + i + 1, "cams": m})
    for r in runs:
        r["t1"] = min(r["t1"], t1)
        r["n"] = len(r["cams"])
    hist = {str(k): int(sum(1 for c in counts if c == k)) for k in sorted(set(counts))}
    return {"window": [t0, t1], "cameras_per_second": counts, "seconds_with_n_cameras": hist, "runs": runs}


def alignment_doc(sync: dict) -> dict:
    """The alignment file Take.load reads: {"sources": [{id, name, slope, offset, accepted, ...}]} (the keys of
    prototype/larissa_kadu/alignment_studio.json). A camera that could not be placed keeps offset null and accepted false."""
    srcs = []
    for c in sync["cameras"]:
        placed = c.get("offset_s") is not None and c.get("placed", True)
        srcs.append({"id": c["id"], "name": c["file"], "duration": c["video_duration_s"], "numbered": True,
                     "slope": c["slope"] if placed else 1.0, "offset": c["offset_s"] if placed else None, "accepted": bool(placed),
                     "residual_ms": c.get("residual_ms"), "confidence": c.get("confidence"),
                     "offset_source": c.get("offset_source"), **({} if placed else {"note": c.get("note", "not placed")})})
    return {"reference": sync["master"]["file"], "convention": "reference_time = slope * source_time + offset",
            "note": "audio clocks of the class recap measured by dancecap.recap_sync (GCC-PHAT against the master, "
                    "speech of the teachers); the reference clock is the master camera's own video time. AUDIO clocks: "
                    "sound travel (3 ms/m) and phone A/V latency leave per-phone VIDEO offsets that dancecap.sync measures.",
            "sources": srcs}


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return o.name
    raise TypeError(f"not serialisable: {type(o)}")


# ----------------------------------------------------------------------------------------------- the whole job


def run(src: Path = RECAP_DIR, out: Path = WORK / "sync", alignment_out: Path | None = ALIGNMENT_OUT, search: bool = True,
        pose: bool = True, force_audio: bool = False, log=print) -> dict:
    cams = discover(src)
    prior = read_prior_offsets(src)
    audio_dir = out / "audio"
    for c in cams:
        wav = audio_dir / (c["path"].stem + ".wav")
        if force_audio or not wav.exists():
            log(f"[recap_sync] extracting audio of {c['name']}")
            extract_audio(c["path"], wav)
        c["wav"] = wav
        c["facts"] = ffprobe_facts(c["path"])
    master = next(c for c in cams if c["id"] == MASTER_ID)
    y = read_wav(master["wav"])
    master_dur = master["facts"]["video_duration_s"]
    index = MasterIndex(y, WIDE_WIN_S + 1.0)
    log(f"[recap_sync] master {master['name']}: {len(y) / SR:.2f} s of audio, video {master_dur:.3f} s")

    audio = {c["id"]: read_wav(c["wav"]) for c in cams}
    recs, wide_all, narrow_all = [], {}, {}
    for c in cams:
        f, x, p = c["facts"], audio[c["id"]], prior.get(c["name"])
        rec = {"id": c["id"], "file": c["name"], "role": c["role"], "is_master": c["id"] == MASTER_ID, "prior_offset_s": p,
               **{k: f[k] for k in ("stored_size", "display_size", "rotation", "codec", "fps", "frames", "video_duration_s",
                                    "video_start_s")},
               "audio": f["audio"], "audio_ok": bool(f["audio"]) and len(x) > 5 * SR,
               "av_start_diff_s": None if not f["audio"] else round(f["audio"]["start_s"] - f["video_start_s"], 4),
               "av_duration_diff_s": None if not f["audio"] else round(f["audio"]["duration_s"] - f["video_duration_s"], 3)}
        recs.append(rec)
        if c["id"] == MASTER_ID:
            rec.update({"offset_s": 0.0, "slope": 1.0, "residual_ms": 0.0, "confidence": 1.0, "offset_source": "master",
                        "drift_ppm": 0.0, "clock": "master", "placed": True})
            continue
        wide = wide_offsets(x, index)
        v = vote(wide)
        wide_all[c["id"]] = {"rows": wide, "vote": v}
        rec["wide_vote"] = v
        log(f"[recap_sync] cam {c['id']} wide: {v}")
        if v["offset_s"] is None or v["votes"] < 3:
            if p is not None:
                rec.update({"offset_s": p, "slope": 1.0, "residual_ms": None, "confidence": 0.0, "placed": True, "clock": "constant",
                            "offset_source": "owner's number, NOT verified (no wide window matched the master)"})
            else:
                rec.update({"offset_s": None, "slope": 1.0, "residual_ms": None, "confidence": 0.0, "placed": False,
                            "offset_source": "not found", "note": "no wide window matched the master"})
            continue
        if p is not None:
            rec["prior_check"] = {"wide_minus_prior_ms": round(1000 * (v["offset_s"] - p), 1),
                                  "wide_plus_prior_ms": round(1000 * (v["offset_s"] + p), 1),
                                  "convention": ("master_t = video_t + offset" if abs(v["offset_s"] - p) < 0.15 else
                                                 ("INVERTED sign" if abs(v["offset_s"] + p) < 0.15 else "prior does not match"))}
        m = measure_camera(x, y, v["offset_s"])
        narrow_all[c["id"]] = m
        if not m.get("ok"):
            rec.update({"offset_s": v["offset_s"], "slope": 1.0, "residual_ms": None, "confidence": 0.1, "placed": True,
                        "offset_source": "wide vote only", "clock": "constant"})
            continue
        d, fit = m["decision"], m["fit"]
        rec.update({"offset_s": d["offset_s"], "slope": d["slope"], "clock": d["clock"], "residual_ms": round(d["residual_ms"], 2),
                    "drift_ppm": round(fit["drift_ppm"], 2), "drift_total_ms": round(fit["drift_total_ms"], 2),
                    "slope_se_ppm": round(fit["slope_se_ppm"], 2), "placed": True,
                    "narrow": {"windows": m["windows"], "good_windows": m["good_windows"], "n_main": m["n_main"],
                               "z_median": round(m["z_median"], 1), "span_s": round(fit["span_s"], 1),
                               "const_offset_s": fit["const_offset_s"], "const_residual_rms_ms": round(fit["const_residual_rms_ms"], 2),
                               "affine_residual_rms_ms": round(fit["residual_rms_ms"], 2),
                               "residual_p95_ms": round(fit["residual_p95_ms"], 2)},
                    "clusters": m["clusters"],
                    "uncertainty_ms": round(max(2.0, max([abs(c["offset_ms_from_main"]) for c in m["clusters"] if c["n"] >= 3], default=0.0)), 1),
                    "confidence": round(float(min(1.0, m["n_main"] / 20.0) * m["n_main"] / max(1, m["good_windows"])), 3)})
        mid = 0.5 * rec["video_duration_s"]
        rec["offset_at_mid_s"] = round(rec["slope"] * mid + rec["offset_s"] - mid, 5)
        rec["offset_source"] = ("measured (owner's number confirmed within %.0f ms)" % abs(1000 * (rec["offset_at_mid_s"] - p))
                                if p is not None and abs(rec["offset_at_mid_s"] - p) < 0.05 else
                                ("measured (differs from the owner's JSON by %.0f ms)" % abs(1000 * (rec["offset_at_mid_s"] - p))
                                 if p is not None else "measured (not in the owner's JSON)"))
        if p is not None:
            rec["measured_minus_prior_ms"] = round(1000 * (rec["offset_at_mid_s"] - p), 1)
        log(f"[recap_sync] cam {c['id']}: {d['clock']} offset {rec['offset_s']:+.4f} s slope {rec['slope']:.7f} "
            f"rms {rec['residual_ms']:.2f} ms, {m['n_main']}/{m['good_windows']}/{m['windows']} windows, clusters {m['clusters']}")

    # envelopes of the placed cameras (needed by the search and by the quality table)
    placed = [r for r in recs if r.get("placed")]
    envs = {r["id"]: (log_envelope(audio[r["id"]]), r["slope"], r["offset_s"]) for r in placed}
    unlisted = {}
    if search:
        weakest = min((r for r in placed if not r["is_master"]), key=lambda r: r["confidence"], default=None)
        for r in recs:
            if r.get("placed") or r["prior_offset_s"] is not None:
                continue
            env = log_envelope(audio[r["id"]])
            refs = {cid: e for cid, e in envs.items() if cid != (weakest["id"] if weakest else "")}
            ctrl = None
            if weakest is not None:
                ctrl = {"control_env": envs[weakest["id"]][0], "control_refs": refs, "control_expect_s": weakest["offset_s"]}
            log(f"[recap_sync] cam {r['id']} not placed by the wide pass: envelope search against {len(refs)} phones "
                f"(control: cam {weakest['id'] if weakest else '-'})")
            res = search_unlisted(env, {cid: e for cid, e in envs.items()}, len(audio[r["id"]]) / SR, **(ctrl or {}))
            res["control_camera"] = weakest["id"] if weakest else None
            unlisted[r["id"]] = res
            log(f"[recap_sync] cam {r['id']} search: found={res['found']} z_max={res['z_max']:.1f} median {res['z_median']:.1f}"
                + (f" control {res['control']}" if "control" in res else ""))
            if res["found"]:
                rec_f = res["offset_s"]
                m = measure_camera(audio[r["id"]], y, rec_f)
                if m.get("ok"):
                    d = m["decision"]
                    r.update({"offset_s": d["offset_s"], "slope": d["slope"], "clock": d["clock"], "residual_ms": round(d["residual_ms"], 2),
                              "placed": True, "confidence": round(float(min(1.0, m["n_main"] / 20.0)), 3),
                              "offset_source": "measured (found by the envelope search)"})
                    r.pop("note", None)
    for r in recs:
        u = unlisted.get(r["id"])
        if not r.get("placed") and u:
            ctrl = u.get("control") or {}
            r["note"] = ("not placed: the wide pass found no matching window and the speech-envelope search against the placed phones "
                         f"has z max {u['z_max']:.1f} (median {u['z_median']:.1f}; noise only, a real match of the weakest placed phone "
                         f"gives z {ctrl.get('z_at_true_offset')}) at any offset and for time scales 0.9-1.2")
            r["offset_source"] = "not found"

    # coverage
    for r in recs:
        if r.get("placed"):
            a, b = r["offset_s"], r["slope"] * r["video_duration_s"] + r["offset_s"]
            r["coverage_master_s"] = [round(a, 3), round(b, 3)]
            r["coverage_in_window_s"] = [round(max(a, 0.0), 3), round(min(b, master_dur), 3)]
        else:
            r["coverage_master_s"] = None
            r["coverage_in_window_s"] = None
    window = (0.0, round(master_dur, 2))
    timeline = coverage_timeline([r for r in recs if r.get("coverage_master_s")], window)

    # audio quality + agreement with the other phones
    quality = {}
    for r in placed:
        quality[r["id"]] = quality_windows(audio[r["id"]], r["slope"], r["offset_s"], r["coverage_master_s"][0],
                                           r["coverage_master_s"][1])
    add_agreement(quality, envs)

    pose_facts = {}
    if pose:
        for c, r in zip(cams, recs):
            pj = src / (c["name"] + ".json")
            if pj.exists():
                pose_facts[c["id"]] = pose_json_facts(pj, r["stored_size"], r["display_size"])
                pose_facts[c["id"]]["frames_match_video"] = pose_facts[c["id"]]["frames"] == r["frames"]
                log(f"[recap_sync] cam {c['id']} pose json: {pose_facts[c['id']]}")

    doc = {"version": 2, "what": "audio clock alignment of the recap phones against the master (dancecap.recap_sync)",
           "convention": "reference_time = slope * source_time + offset (reference = the master camera's video time; source_time = PTS, "
                         "every file starts at PTS 0)",
           "master": {"id": MASTER_ID, "file": master["name"], "duration_s": master_dur},
           "method": {"wide": f"GCC-PHAT {BAND[0]:.0f}-{BAND[1]:.0f} Hz, {SR} Hz mono, {WIDE_N} windows of {WIDE_WIN_S:.0f} s searched in the "
                              f"whole master, accepted when z >= {MIN_Z_WIDE} and psr >= {MIN_PSR_WIDE}",
                      "narrow": f"windows of {NARROW_WIN_S:.0f} s every {NARROW_HOP_S:.0f} s, +-{NARROW_SEARCH_S} s around the wide vote, "
                                f"accepted when z >= {MIN_Z_NARROW}; densest {1000 * CLUSTER_BIN_S:.0f} ms cluster (+-{1000 * CLUSTER_HALF_S:.0f} ms) "
                                "= the clock; affine only when the drift is > 3 standard errors and > 2 ms"},
           "cameras": recs, "timeline": timeline, "audio_quality": quality, "audio_quality_summary": quality_summary(quality),
           "unlisted_search": unlisted, "pose_json": pose_facts,
           "wide": wide_all, "narrow": {k: {"windows": v["windows"], "good_windows": v["good_windows"],
                                            "rows": [{kk: (round(vv, 5) if isinstance(vv, float) else vv) for kk, vv in rr.items()}
                                                     for rr in v.get("good_rows", [])]} for k, v in narrow_all.items()}}
    out.mkdir(parents=True, exist_ok=True)
    (out / "sync.json").write_text(json.dumps(doc, indent=1, default=_jsonable), encoding="utf-8")
    adoc = alignment_doc(doc)
    (out / "alignment_recap.json").write_text(json.dumps(adoc, indent=1), encoding="utf-8")
    if alignment_out is not None:
        alignment_out.parent.mkdir(parents=True, exist_ok=True)
        alignment_out.write_text(json.dumps(adoc, indent=1) + "\n", encoding="utf-8")
    log(f"[recap_sync] -> {out / 'sync.json'}, {out / 'alignment_recap.json'}" + (f", {alignment_out.name}" if alignment_out else ""))
    return doc


def plot_overview(doc: dict, png: Path) -> Path:
    """review/sync_overview.png: per camera the accepted narrow windows (offset from the camera's clock, main cluster vs
    other), the coverage of the master timeline, the number of phones per second and the audio-quality heat map."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cams = [c for c in doc["cameras"] if not c["is_master"]]
    fig, ax = plt.subplots(4, 1, figsize=(13, 15), gridspec_kw={"height_ratios": [3, 2, 1, 2]})
    for k, c in enumerate(cams):
        rows = (doc["narrow"].get(c["id"]) or {}).get("rows") or []
        if not rows or not c.get("placed"):
            continue
        for r in rows:
            clk = c["slope"] * r["t_cam"] + c["offset_s"] - r["t_cam"]
            y = 1000 * (r["offset_s"] - clk)
            ax[0].scatter(r["t_master"], k * 80 + y, s=9, c="tab:blue" if r.get("cluster") == "main" else "tab:red", alpha=0.8)
        ax[0].axhline(k * 80, color="0.7", lw=0.5)
        ax[0].text(362, k * 80, f"{c['id']} {c['role']}  {c['offset_s']:+.4f} s  {c.get('clock', '')}", va="center", fontsize=8)
    ax[0].set_title("accepted narrow windows: offset minus the fitted clock (ms, +-40 per camera); blue = main cluster, red = other acoustic path")
    ax[0].set_xlim(-12, 440)
    ax[0].set_yticks([])
    for k, c in enumerate(doc["cameras"]):
        cv = c.get("coverage_master_s")
        if cv:
            ax[1].barh(k, cv[1] - cv[0], left=cv[0], color="tab:green" if c.get("placed") else "0.7")
        else:
            ax[1].text(5, k, f"{c['id']} {c['role']}: not placed", va="center", fontsize=8)
            continue
        ax[1].text(max(cv[1], 0) + 3, k, f"{c['id']} {c['role']}", va="center", fontsize=8)
    ax[1].set_yticks([])
    ax[1].invert_yaxis()
    ax[1].set_xlim(-12, 440)
    ax[1].set_title("coverage on the master clock (s)")
    n = doc["timeline"]["cameras_per_second"]
    ax[2].step(np.arange(len(n)), n, where="post")
    ax[2].set_xlim(-12, 440)
    ax[2].set_title("phones recording per master second")
    ids = [c for c in doc["audio_quality"] if doc["audio_quality"][c]]
    snr = np.full((len(ids), 12), np.nan)
    for i, cid in enumerate(ids):
        for r in doc["audio_quality"][cid]:
            snr[i, int(round(r["t0"] / QUALITY_WIN_S))] = r.get("env_ncc") if r.get("env_ncc") is not None else np.nan
    im = ax[3].imshow(snr, aspect="auto", cmap="viridis", vmin=0, vmax=1, extent=(0, 360, len(ids), 0))
    ax[3].set_yticks(np.arange(len(ids)) + 0.5)
    ax[3].set_yticklabels(ids)
    ax[3].set_title("env_ncc: agreement of each phone's speech envelope with the other phones, per 30 s (1 = same speech)")
    fig.colorbar(im, ax=ax[3], fraction=0.02)
    fig.tight_layout()
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=70)
    plt.close(fig)
    return png


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.recap_sync", description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["run", "audio", "plot"])
    ap.add_argument("--src", type=Path, default=RECAP_DIR)
    ap.add_argument("--out", type=Path, default=WORK / "sync")
    ap.add_argument("--alignment-out", type=Path, default=ALIGNMENT_OUT, help="the tracked alignment the take config names")
    ap.add_argument("--no-search", action="store_true", help="skip the envelope search for an unlisted camera")
    ap.add_argument("--no-pose", action="store_true", help="skip the 2D pose JSON facts")
    ap.add_argument("--force-audio", action="store_true", help="extract the wavs again even when they exist")
    a = ap.parse_args(argv)
    if a.cmd == "audio":
        for c in discover(a.src):
            extract_audio(c["path"], a.out / "audio" / (c["path"].stem + ".wav"))
        return 0
    if a.cmd == "plot":
        doc = json.loads((a.out / "sync.json").read_text(encoding="utf-8"))
        print(plot_overview(doc, a.out.parent / "review" / "sync_overview.png"))
        return 0
    doc = run(a.src, a.out, a.alignment_out, search=not a.no_search, pose=not a.no_pose, force_audio=a.force_audio)
    plot_overview(doc, a.out.parent / "review" / "sync_overview.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
