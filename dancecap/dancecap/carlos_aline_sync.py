"""Audio sync of the take carlos_aline: the two videos on one clock, and which song plays when.

The take is ONE Jack & Jill heat (Zouk SEA 2019, Carlos + Aline) filmed by two cameras, both horizontal, both on tripods
(the owner 2026-10-09; the heads pan to follow the couple):

    01  Carlos Aline Zouk Artist Jack Jill at Zouk SEA 2019.mp4   1920x1080 23.976 fps, 202.08 s, watermark, graded edit
    02  Zouk Sea 2019 Carlos and Aline <surname> Jack Jill.mp4    1920x906  30 fps,     186.27 s, wide, phone-like

plus the two songs the owner named (Kojo Funds - Check (With Raye) = track 1, David Guetta - Say My Name = track 22):

    python -m dancecap.carlos_aline_sync run [--src DIR] [--out DIR] [--alignment-out FILE] [--force-audio]
    python -m dancecap.carlos_aline_sync plot [--out DIR]        review/sync_overview.png from sync/sync.json
    python -m dancecap.carlos_aline_sync alignment [--out DIR] [--alignment-out FILE]   rewrite the alignment JSON from sync/sync.json

CLOCKS (the same convention as Take / Source: reference_time = slope * source_time + offset).
  * Reference clock = camera 01's own video time (it starts first and covers the whole heat; every reference time is >= 0).
  * Camera 02 on it: GCC-PHAT (dancecap.recap_sync machinery) of the two videos' audio, the same sound in both files,
    narrow windows along the whole overlap, a robust straight line (affine only when the drift is real).
  * Songs: the DJ plays each track with a TEMPO CHANGE AT CONSTANT PITCH (key lock; measured: the mean log-spectra of a video
    window and of the song stay within 1/96 octave while the tempo is 3.8-5.8 % off), so waveform correlation (GCC-PHAT) cannot
    lock; the rhythm can. Per 12 s window of the video audio the log-mel spectral flux (48 bands, 10 ms) is stretched by the
    tempo ratio a and cross-correlated against the song's flux (all bands summed in the frequency domain); the song position
    song_t = a * reference_t + b is the line through the windows (robust, dancecap.recap_sync.fit_clock), a re-measured at the
    fitted a. Song time is the time a player shows (the decoded mp3 from its first sample; the encoder delay is skipped).
  * The timeline says where each song is audible: the fine windows (6 s, 1.5 s apart) along each song's line; a song is
    audible where its windows reach z >= Z_FINE; the DJ mixes the second track in under the end of the first.

Limits: these are AUDIO clocks. The phone/camera A/V latency and the sound travel (3 ms per metre) are not in them; the
per-camera VIDEO offsets are measured later by dancecap.sync ([sync] required). The song clock precision is a few tens of ms
(spectral-flux frames blur onsets); it serves the beat grid, never the video sync.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

from . import recap_sync as rs
from .take import PACKAGE_ROOT, WORKSPACE

SR = rs.SR
HOP_S = 0.01                    # feature frame (160 samples at 16 kHz)
NFFT_FEAT = 1024
NMEL = 48
FMIN, FMAX = 60.0, 6000.0
WIN_S = 12.0                    # coarse / clock windows
WIN_HOP_S = 3.0
FINE_WIN_S = 6.0                # timeline windows
FINE_HOP_S = 1.5
Z_CLOCK = 8.0                   # a clock window counts at this z
Z_FINE = 6.0                    # a timeline window counts at this z (the line already fixes the lag: a +-0.25 s search)
FINE_SEARCH_S = 0.25
CLOCK_TOL_S = 0.06              # |window - line| inliers of the song clock
COARSE_SCALES = (0.90, 1.10, 0.002)
CAM_IDS = {"01": "Carlos Aline Zouk Artist Jack Jill at Zouk SEA 2019.mp4",
           "02": "Zouk Sea 2019 Carlos and Aline Borges Jack Jill.mp4"}
SONG_FILES = {"kojo": "Kojo Funds - Check (With Raye)_01.mp3",
              "guetta": "David Guetta - Say My Name (Lyrics) ft. Bebe Rexha, J Balvin.mp3"}
SONG_INFO = {"kojo": {"title": "Check (With Raye)", "artist": "Kojo Funds", "owner_track": 1},
             "guetta": {"title": "Say My Name (feat. Bebe Rexha, J Balvin)", "artist": "David Guetta", "owner_track": 22}}
MASTER_ID = "01"
SRC_DIR = WORKSPACE / "dances" / "Carlos-Aline"
WORK = PACKAGE_ROOT / "work" / "carlos_aline"
ALIGNMENT_OUT = WORKSPACE / "prototype" / "larissa_kadu" / "alignment_carlos_aline.json"
HZ = 30.0                       # the fused 3D grid
# Danced span on the reference clock, from frames of both videos (work/carlos_aline/NOTES_C1.md): see DANCE_SPAN.
DANCE_SPAN = {"start_s": 3.6, "end_s": 198.0,
              "start_why": ("camera 01 starts with the couple's walk-on: Carlos with the hostess (0-1.5 s), alone across the floor (1.5-3.5 s, "
                            "the first skips), Aline steps in from the right at 3.6 s and they are hand in hand at 4.6 s; only camera 01 "
                            "sees 3.6-14.8 s (camera 02 starts at 14.82 s, mid-dance)"),
              "end_why": ("the last figure and the bow: camera 02 shows the arms-out finish at 182.0 s (183.0 s the bow, they hug and "
                          "walk off from 184 s) = 196.8-198.0 s on the reference clock; camera 01 cuts to a zoom-out at 196.95 s")}


# ----------------------------------------------------------------------------------------------- affine clocks


def affine_invert(slope: float, offset: float) -> tuple[float, float]:
    """y = slope * x + offset  ->  x = slope' * y + offset'."""
    return 1.0 / slope, -offset / slope


def affine_compose(outer: tuple[float, float], inner: tuple[float, float]) -> tuple[float, float]:
    """outer(inner(x)) as (slope, offset); each is (slope, offset)."""
    return outer[0] * inner[0], outer[0] * inner[1] + outer[1]


def lattice(start_s: float, end_s: float, hz: float = HZ, anchor_s: float = 0.0) -> dict:
    """The fused 3D grid t_k = anchor + k / hz covering [start_s, end_s]: k0 = first lattice point >= start, k1 = last <= end."""
    eps = 1e-9
    k0 = int(math.ceil((start_s - anchor_s) * hz - eps))
    k1 = int(math.floor((end_s - anchor_s) * hz + eps))
    return {"anchor_s": anchor_s, "hz": hz, "k0": k0, "k1": k1, "frames": k1 - k0 + 1,
            "t_first_s": anchor_s + k0 / hz, "t_last_s": anchor_s + k1 / hz}


# ----------------------------------------------------------------------------------------------- features


def _mel_filters(nmel: int = NMEL, nfft: int = NFFT_FEAT, sr: int = SR, fmin: float = FMIN, fmax: float = FMAX) -> np.ndarray:
    mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)          # noqa: E731
    inv = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)         # noqa: E731
    pts = inv(np.linspace(mel(fmin), mel(fmax), nmel + 2))
    fr = np.fft.rfftfreq(nfft, 1.0 / sr)
    fb = np.zeros((nmel, len(fr)))
    for i in range(nmel):
        lo, c, hi = pts[i:i + 3]
        fb[i] = np.clip(np.minimum((fr - lo) / (c - lo), (hi - fr) / (hi - c)), 0.0, None)
    return fb


_FB: list[np.ndarray] = []


def log_mel(x: np.ndarray, hop: int = int(HOP_S * SR), nfft: int = NFFT_FEAT, block: int = 1500) -> np.ndarray:
    """Log-mel power (frames x NMEL) of a mono signal, frame i at i * HOP_S (Hann, nfft 1024 = 64 ms)."""
    if not _FB:
        _FB.append(_mel_filters())
    fb = _FB[0]
    n = (len(x) - nfft) // hop + 1
    if n <= 0:
        return np.zeros((0, fb.shape[0]), np.float32)
    win = np.hanning(nfft)
    out = np.empty((n, fb.shape[0]), np.float32)
    for a in range(0, n, block):
        b = min(n, a + block)
        idx = np.arange(nfft)[None, :] + hop * np.arange(a, b)[:, None]
        p = np.abs(np.fft.rfft(x[idx] * win, axis=1)) ** 2
        out[a:b] = np.log(p @ fb.T + 1e-8)
    return out


def flux_features(L: np.ndarray, local_s: float = 1.0) -> np.ndarray:
    """Half-wave spectral flux per band, its local (1 s) mean removed, each band scaled to unit variance: (frames x bands)."""
    d = np.vstack([np.zeros((1, L.shape[1])), np.clip(np.diff(L.astype(np.float64), axis=0), 0.0, None)])
    k = max(1, int(local_s / HOP_S))
    kern = np.ones(k) / k
    m = np.stack([np.convolve(d[:, j], kern, "same") for j in range(d.shape[1])], 1)
    d = d - m
    return (d / (d.std(0, keepdims=True) + 1e-6)).astype(np.float32)


def warp_features(F: np.ndarray, a: float) -> np.ndarray:
    """Time-stretch a feature matrix by the tempo ratio a (song_t = a * video_t): the output frame i sits at song time i * HOP_S
    and is the input at video time i * HOP_S / a (linear interpolation per band)."""
    n_out = max(2, int(len(F) * a))
    pos = np.arange(n_out) / a
    pos = np.clip(pos, 0, len(F) - 1)
    i0 = np.floor(pos).astype(int)
    i1 = np.minimum(i0 + 1, len(F) - 1)
    w = (pos - i0)[:, None].astype(np.float32)
    return F[i0] * (1.0 - w) + F[i1] * w


class FeatureIndex:
    """The song's features, Fourier transformed once, so many windows can be located in it."""

    def __init__(self, F: np.ndarray, max_win_frames: int):
        self.F = F
        self.n = rs._nfft(len(F) + max_win_frames)
        self.S = np.fft.rfft(F, self.n, axis=0)
        self.cum = np.concatenate([np.zeros((1, F.shape[1])), np.cumsum(F.astype(np.float64) ** 2, axis=0)])

    def ncc(self, win: np.ndarray) -> np.ndarray:
        """Normalised cross-correlation (all bands pooled) of `win` at every valid shift: ncc[j] = window at song frame j."""
        m, L = len(win), len(self.F)
        if m >= L:
            return np.zeros(0)
        w = win - win.mean(0, keepdims=True)
        W = np.fft.rfft(w, self.n, axis=0)
        cc = np.fft.irfft((self.S * np.conj(W)).sum(1), self.n)[:L - m + 1]
        e = (self.cum[m:] - self.cum[:-m]).sum(1)
        return cc / (np.sqrt(np.maximum(e, 1e-9) * float((w ** 2).sum())) + 1e-9)

    def locate(self, win: np.ndarray, lo_s: float | None = None, hi_s: float | None = None) -> dict:
        """Best position of the window (song time of its first frame, s) inside [lo_s, hi_s] with its z-score (peak over the
        spread of the correlation away from it) and the peak itself."""
        c = self.ncc(win)
        if len(c) < 20:
            return {"song_t0_s": float("nan"), "z": 0.0, "ncc": 0.0}
        lo = 0 if lo_s is None else max(0, int(round(lo_s / HOP_S)))
        hi = len(c) - 1 if hi_s is None else min(len(c) - 1, int(round(hi_s / HOP_S)))
        if hi <= lo + 2:
            return {"song_t0_s": float("nan"), "z": 0.0, "ncc": 0.0}
        j = lo + int(np.argmax(c[lo:hi + 1]))
        frac = 0.0
        if 0 < j < len(c) - 1:
            den = c[j - 1] - 2.0 * c[j] + c[j + 1]
            if den < 0:
                frac = float(np.clip(0.5 * (c[j - 1] - c[j + 1]) / den, -0.5, 0.5))
        far = np.ones(len(c), bool)
        far[max(0, j - 30):j + 31] = False
        sd = float(c[far].std()) if far.any() else 1.0
        z = float((c[j] - c[far].mean()) / sd) if sd > 0 else 0.0
        return {"song_t0_s": float((j + frac) * HOP_S), "z": z, "ncc": float(c[j])}


def window_rows(video_F: np.ndarray, idx: FeatureIndex, a: float, starts: np.ndarray, win_s: float, line: tuple[float, float] | None = None,
                search_s: float = FINE_SEARCH_S) -> list[dict]:
    """Locate windows of the video's flux (start times `starts`, length win_s) in the song at the tempo ratio a.

    Row: u0 (video time of the window start), t_cam (its centre), song_c (the song time at that centre), z, ncc. With `line`
    = (slope, offset) of song_t = slope * video_t + offset only +-search_s around the line's prediction are searched."""
    rows = []
    W = int(round(win_s / HOP_S))
    for u0 in starts:
        a0 = int(round(u0 / HOP_S))
        v = video_F[a0:a0 + W]
        if len(v) < W:
            continue
        w = warp_features(v, a)
        lo = hi = None
        if line is not None:
            pred = line[0] * u0 + line[1]
            lo, hi = max(0.0, pred - search_s), pred + search_s
            if hi <= 0:
                continue
        r = idx.locate(w, lo, hi)
        if not np.isfinite(r["song_t0_s"]):
            continue
        song_c = r["song_t0_s"] + 0.5 * len(w) * HOP_S
        rows.append({"u0": float(u0), "t_cam": float(u0 + 0.5 * win_s), "song_c": float(song_c), "z": r["z"], "ncc": r["ncc"]})
    return rows


def coarse_tempo(video_env: np.ndarray, song_env: np.ndarray, win_s: float = 20.0, hop_s: float = 10.0,
                 scales: tuple[float, float, float] = COARSE_SCALES, t_range: tuple[float, float] | None = None) -> list[dict]:
    """Per 20 s window of the video the tempo ratio a (song_t = a * video_t + b) and offset b with the best 1-D onset-envelope
    correlation: [{u0, a, b, z}]. The envelopes are (frames x 1) flux sums."""
    idx = FeatureIndex(song_env, int(win_s / HOP_S) + 10)
    out = []
    W = int(win_s / HOP_S)
    lo_t, hi_t = t_range or (0.0, len(video_env) * HOP_S)
    for u0 in np.arange(lo_t, hi_t - win_s + 1e-6, hop_s):
        v = video_env[int(u0 / HOP_S):int(u0 / HOP_S) + W]
        if len(v) < W:
            continue
        best = None
        zs = []
        for a in np.arange(scales[0], scales[1] + 1e-9, scales[2]):
            w = warp_features(v, float(a))
            c = idx.ncc(w)
            if len(c) < 20:
                continue
            j = int(np.argmax(c))
            zs.append(c[j])
            if best is None or c[j] > best[0]:
                best = (float(c[j]), float(a), j * HOP_S)
        if best is None:
            continue
        zz = np.array(zs)
        out.append({"u0": float(u0), "a": best[1], "b": best[2] - best[1] * float(u0), "ncc": best[0],
                    "z": float((best[0] - zz.mean()) / (zz.std() + 1e-9))})
    return out


# ----------------------------------------------------------------------------------------------- song clocks


def fit_song_clock(rows: list[dict], z_min: float = Z_CLOCK, tol_s: float = CLOCK_TOL_S, bin_s: float = 0.5) -> dict:
    """The straight line song_t = slope * video_t + offset through the window centres (t_cam, song_c) of the rows with z >= z_min:
    first the densest cluster of the per-window offsets (song_c - slope0 * t_cam, slope0 = the median tempo of the pairs),
    then the robust fit of dancecap.recap_sync.fit_clock. Returns {ok, slope, offset_s, residual_rms_ms, residual_p95_ms,
    n, n_inliers, span_s, slope_se_ppm, windows: [row + resid_ms + inlier]}."""
    good = [r for r in rows if r["z"] >= z_min]
    if len(good) < 4:
        return {"ok": False, "n": len(good)}
    t = np.array([r["t_cam"] for r in good])
    s = np.array([r["song_c"] for r in good])
    # per-window offset at slope 1 is useless (the tempo is 4-6 % off): use the pairwise slope of the densest consecutive pairs
    order = np.argsort(t)
    t, s = t[order], s[order]
    good = [good[i] for i in order]
    pair_slopes = np.diff(s) / np.maximum(np.diff(t), 1e-9)
    slope0 = float(np.median(pair_slopes[(pair_slopes > 0.5) & (pair_slopes < 2.0)])) if ((pair_slopes > 0.5) & (pair_slopes < 2.0)).any() else 1.0
    off = s - slope0 * t
    main = rs.densest_cluster(off, bin_s=bin_s, half_s=max(tol_s, 0.5))
    if main.sum() < 4:
        return {"ok": False, "n": len(good)}
    fit = rs.fit_clock(t[main], s[main], clip_s=tol_s)
    if not fit.get("ok"):
        return {"ok": False, "n": len(good)}
    res = s - (fit["slope"] * t + fit["offset_s"])
    inl = np.abs(res) <= max(tol_s, 3 * (fit["residual_rms_ms"] / 1000.0))
    out = {k: fit[k] for k in ("slope", "offset_s", "slope_se_ppm", "span_s", "residual_rms_ms", "residual_p95_ms") if k in fit}
    out.update({"ok": True, "n": len(good), "n_inliers": int(inl.sum()),
                "windows": [{**{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()},
                             "resid_ms": round(float(1000 * res[i]), 2), "inlier": bool(inl[i])} for i, r in enumerate(good)]})
    if inl.sum() >= 4:
        refit = rs.fit_clock(t[inl], s[inl], clip_s=tol_s)
        if refit.get("ok"):
            res2 = s - (refit["slope"] * t + refit["offset_s"])
            out.update({k: refit[k] for k in ("slope", "offset_s", "slope_se_ppm", "span_s") if k in refit})
            out["residual_rms_ms"] = float(1000 * np.sqrt(np.mean(res2[inl] ** 2)))
            out["residual_p95_ms"] = float(1000 * np.percentile(np.abs(res2[inl]), 95))
            for i, w in enumerate(out["windows"]):
                w["resid_ms"] = round(float(1000 * res2[i]), 2)
    return out


def audible_runs(rows: list[dict], z_min: float = Z_FINE, max_gap_s: float = 4.0, min_len_s: float = 4.0,
                 core_s: float = FINE_HOP_S) -> list[list[float]]:
    """Runs [t_start, t_end] (video seconds) where the song is present: each accepted window (z >= z_min) counts for the
    core_s around its CENTRE (a 6 s window straddling the end of a song would otherwise stretch the run by 3 s); runs closer than
    max_gap_s are joined, runs shorter than min_len_s dropped. Each row needs t_cam (the window centre) and z."""
    ok = sorted((r for r in rows if r["z"] >= z_min), key=lambda r: r["t_cam"])
    runs: list[list[float]] = []
    for r in ok:
        a, b = r["t_cam"] - 0.5 * core_s, r["t_cam"] + 0.5 * core_s
        if runs and a - runs[-1][1] <= max_gap_s:
            runs[-1][1] = max(runs[-1][1], b)
        else:
            runs.append([a, b])
    return [[float(a), float(b)] for a, b in runs if b - a >= min_len_s]


def snap_to_song_start(run_start_s: float, zero_s: float, max_gap_s: float = 8.0) -> float:
    """A song cannot be audible before its position 0: when the first detection comes within max_gap_s after the reference time at
    which the song's own start plays (zero_s; the intro is quiet or under the other song), the audible span starts there."""
    return zero_s if 0.0 <= run_start_s - zero_s <= max_gap_s else run_start_s


def timeline(songs: dict[str, tuple[float, float] | None], total_s: float) -> list[dict]:
    """Segments of the reference clock [0, total_s] by which songs are audible: songs = {name: (t0, t1) | None}.
    -> [{from_s, to_s, state, songs}] with state = 'none' | name | 'mix' (two songs at once); touching segments are merged."""
    cuts = {0.0, float(total_s)}
    for iv in songs.values():
        if iv:
            cuts.update(min(max(float(x), 0.0), total_s) for x in iv)
    pts = sorted(cuts)
    segs: list[dict] = []
    for a, b in zip(pts[:-1], pts[1:]):
        if b - a < 1e-9:
            continue
        mid = 0.5 * (a + b)
        on = [n for n, iv in songs.items() if iv and iv[0] <= mid <= iv[1]]
        state = "none" if not on else (on[0] if len(on) == 1 else "mix")
        if segs and segs[-1]["state"] == state and segs[-1]["songs"] == on:
            segs[-1]["to_s"] = b
        else:
            segs.append({"from_s": a, "to_s": b, "state": state, "songs": on})
    return segs


def song_start_in_reference(slope: float, offset_s: float, song_t: float = 0.0) -> float:
    """Reference time at which the song position song_t plays (song_t = slope * ref + offset)."""
    return (song_t - offset_s) / slope


# ----------------------------------------------------------------------------------------------- the run


def log(msg: str) -> None:
    print(msg, flush=True)


def _feats(wav: Path) -> tuple[np.ndarray, np.ndarray]:
    L = log_mel(rs.read_wav(wav))
    F = flux_features(L)
    return F, F.sum(1, keepdims=True).astype(np.float32)


def measure_song(name: str, vid_F: np.ndarray, vid_env: np.ndarray, song_F: np.ndarray, song_env: np.ndarray, dur_s: float) -> dict:
    """Tempo ratio, clock and audible extent of one song in the reference video's audio."""
    log(f"[carlos_aline_sync] {name}: coarse tempo scan")
    coarse = coarse_tempo(vid_env, song_env)
    zc = np.array([c["z"] for c in coarse])
    good = [c for c in coarse if c["z"] >= 4.0]
    if not good:
        return {"found": False, "coarse": coarse}
    a0 = float(np.median([c["a"] for c in good]))
    # the dominant tempo of the confident windows (the DJ may use one ratio per song)
    ar = np.array([c["a"] for c in good])
    near = ar[np.abs(ar - a0) < 0.012]
    a0 = float(np.median(near))
    idx = FeatureIndex(song_F, int((WIN_S + 2) * (1 / HOP_S) * 1.2))
    starts = np.arange(0.0, dur_s - WIN_S, WIN_HOP_S)
    rows = window_rows(vid_F, idx, a0, starts, WIN_S)
    fit = fit_song_clock(rows)
    if not fit["ok"]:
        return {"found": False, "coarse": coarse, "a0": a0}
    a = fit["slope"]
    for it in range(2):                       # re-measure at the fitted tempo, refit
        rows = window_rows(vid_F, idx, a, starts, WIN_S)
        fit2 = fit_song_clock(rows)
        if not fit2["ok"]:
            break
        fit, a = fit2, fit2["slope"]
        log(f"[carlos_aline_sync] {name}: iter {it}: a {fit['slope']:.5f} b {fit['offset_s']:+.3f} s rms {fit['residual_rms_ms']:.1f} ms "
            f"{fit['n_inliers']}/{fit['n']} windows")
    line = (fit["slope"], fit["offset_s"])
    fstarts = np.arange(0.0, dur_s - FINE_WIN_S, FINE_HOP_S)
    fine = window_rows(vid_F, idx, a, fstarts, FINE_WIN_S, line=line)
    for r in fine:
        r["win"] = FINE_WIN_S
        r["resid_ms"] = round(1000 * (r["song_c"] - (line[0] * r["t_cam"] + line[1])), 1)
    runs = audible_runs(fine)
    detected = [list(r) for r in runs]
    zero = song_start_in_reference(line[0], line[1], 0.0)
    if runs:
        runs[0][0] = snap_to_song_start(runs[0][0], zero)
    return {"found": True, "coarse": coarse, "tempo_ratio_song_per_video": a, "fit": fit, "fine": fine, "audible_runs_s": runs,
            "detected_runs_s": detected,
            "z_clock_median": float(np.median([w["z"] for w in fit["windows"]]))}


def run(src: Path = SRC_DIR, out: Path = WORK / "sync", alignment_out: Path | None = ALIGNMENT_OUT, force_audio: bool = False) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    adir = WORK / "audio"
    wavs: dict[str, Path] = {}
    for key, fname in {**{f"cam{c}": f for c, f in CAM_IDS.items()}, **SONG_FILES}.items():
        w = adir / f"{key}.wav"
        if force_audio or not w.exists():
            log(f"[carlos_aline_sync] extracting {key}")
            rs.extract_audio(src / fname, w)
        wavs[key] = w
    cam = {c: rs.read_wav(wavs[f"cam{c}"]) for c in CAM_IDS}
    dur = {c: len(cam[c]) / SR for c in cam}

    # 1. camera 02 on camera 01 (the same sound in both files): wide vote, narrow windows, line
    ca, cb = cam["01"], cam["02"]
    idxm = rs.MasterIndex(ca, rs.WIDE_WIN_S)
    wide = rs.wide_offsets(cb, idxm, n=16)
    v = rs.vote(wide)
    if v["offset_s"] is None:
        raise RuntimeError("camera 02's audio was not found in camera 01's (wide pass)")
    m = rs.measure_camera(cb, ca, v["offset_s"])
    if not m.get("ok"):
        raise RuntimeError("camera 02: narrow pass failed")
    d = m["decision"]
    log(f"[carlos_aline_sync] cam 02 on cam 01: {d['clock']} offset {d['offset_s']:+.4f} s slope {d['slope']:.7f} rms {d['residual_ms']:.2f} ms "
        f"({m['n_main']}/{m['good_windows']}/{m['windows']} windows), clusters {m['clusters']}")
    cam02 = {"offset_s": d["offset_s"], "slope": d["slope"], "clock": d["clock"], "residual_ms": d["residual_ms"]}

    # 2. songs in camera 01's audio (and, as a cross-check, in camera 02's)
    L1 = log_mel(ca)
    F1 = flux_features(L1)
    env1 = F1.sum(1, keepdims=True).astype(np.float32)
    songs: dict[str, dict] = {}
    for name in SONG_FILES:
        sF, senv = _feats(wavs[name])
        res = measure_song(name, F1, env1, sF, senv, dur["01"])
        res["song_duration_s"] = len(rs.read_wav(wavs[name])) / SR
        songs[name] = res
    # cross-check on camera 02 (its own audio, mapped through the 01 clock)
    F2 = flux_features(log_mel(cb))
    for name, res in songs.items():
        if not res.get("found"):
            continue
        sF, _ = _feats(wavs[name])
        idx = FeatureIndex(sF, int((WIN_S + 2) * (1 / HOP_S) * 1.2))
        a = res["fit"]["slope"]
        s1, o1 = res["fit"]["slope"], res["fit"]["offset_s"]
        # cam01 ref = cam02 slope*u + offset: song_t = s1 * (c2s * u + c2o) + o1
        pred = affine_compose((s1, o1), (cam02["slope"], cam02["offset_s"]))
        rows = window_rows(F2, idx, a * cam02["slope"], np.arange(0.0, dur["02"] - WIN_S, WIN_HOP_S), WIN_S, line=pred, search_s=0.6)
        zs = [r["z"] for r in rows]
        good = [r for r in rows if r["z"] >= Z_CLOCK]
        f2 = fit_song_clock(rows) if len(good) >= 4 else {"ok": False}
        res["cross_check_cam02"] = {"windows": len(rows), "good": len(good), "z_median_good": float(np.median([r["z"] for r in good])) if good else None,
                                    "fit_slope": f2.get("slope"), "fit_offset_s": f2.get("offset_s"),
                                    "predicted_slope": pred[0], "predicted_offset_s": pred[1],
                                    "offset_diff_ms_at_mid": (round(1000 * ((f2["slope"] * 100 + f2["offset_s"]) - (pred[0] * 100 + pred[1])), 1) if f2.get("ok") else None)}
        log(f"[carlos_aline_sync] {name} cross-check on cam 02: {res['cross_check_cam02']}")

    # 3. timeline, song starts, lattice
    ivs: dict[str, tuple[float, float] | None] = {}
    for name, res in songs.items():
        if res.get("found") and res["audible_runs_s"]:
            runs = res["audible_runs_s"]
            ivs[name] = (runs[0][0], runs[-1][1])
        else:
            ivs[name] = None
    tl = timeline(ivs, dur["01"])
    for name, res in songs.items():
        if not res.get("found"):
            continue
        s, o = res["fit"]["slope"], res["fit"]["offset_s"]
        iv = ivs[name]
        res["clock"] = {"convention": "song_time = slope * reference_time + offset", "slope": s, "offset_s": o,
                        "inverse": {"slope": 1 / s, "offset_s": -o / s},
                        "residual_rms_ms": res["fit"]["residual_rms_ms"], "residual_p95_ms": res["fit"]["residual_p95_ms"]}
        res["song_zero_at_reference_s"] = song_start_in_reference(s, o, 0.0)
        if iv:
            res["audible_reference_s"] = [round(iv[0], 2), round(iv[1], 2)]
            res["song_range_s"] = [round(s * iv[0] + o, 2), round(s * iv[1] + o, 2)]
    span = dict(DANCE_SPAN)
    lat = lattice(span["start_s"], span["end_s"])
    cams = []
    for cid, fname in CAM_IDS.items():
        rec = {"id": cid, "name": fname, "duration_s": round(dur[cid], 3), "numbered": True, "accepted": True}
        if cid == MASTER_ID:
            rec.update({"slope": 1.0, "offset": 0.0, "residual_ms": 0.0, "confidence": 1.0, "offset_source": "master"})
        else:
            rec.update({"slope": cam02["slope"], "offset": cam02["offset_s"], "residual_ms": round(cam02["residual_ms"], 2),
                        "confidence": round(min(1.0, m["n_main"] / 20.0), 3), "clock": cam02["clock"],
                        "offset_source": "measured by GCC-PHAT of the two videos' audio (same sound)"})
        rec["coverage_reference_s"] = [round(rec["slope"] * 0.0 + rec["offset"], 3), round(rec["slope"] * dur[cid] + rec["offset"], 3)]
        cams.append(rec)
    doc = {"version": 1, "what": "audio sync of the take carlos_aline (dancecap.carlos_aline_sync)",
           "convention": "reference_time = slope * source_time + offset; reference = camera 01's own video time",
           "reference": {"id": MASTER_ID, "name": CAM_IDS[MASTER_ID], "why": "starts first (the couple's walk-on), covers the whole heat, the best picture"},
           "cameras": cams,
           "video_pair": {"wide_vote": v, "n_main": m["n_main"], "good_windows": m["good_windows"], "windows": m["windows"],
                          "fit": m["fit"], "decision": d, "clusters": m["clusters"], "rows": m.get("good_rows", [])},
           "songs": {k: {kk: vv for kk, vv in s.items() if kk != "coarse"} for k, s in songs.items()},
           "songs_coarse": {k: s.get("coarse") for k, s in songs.items()},
           "song_info": SONG_INFO, "timeline": tl, "danced_span": span, "lattice": lat,
           "notes": {"pre_song": ("camera 01 0-13.3 s: no song (the sound is an announcement / entrance sound that fades out at 8.5-12.5 s, "
                                  "silence-like 12-13 s); Kojo Funds starts from its first sample at 13.31 s"),
                     "audio_clock_accuracy": ("song clocks: residual 2 ms rms to the line inside each song; camera 02's own audio gives the same "
                                              "line within 30 ms (spectral-flux frames blur onsets; phone vs clean mix)")},
           "heat": "one heat filmed twice: the two videos carry the same sound at one constant offset (cam 02 starts 14.8 s after cam 01)"}
    (out / "sync.json").write_text(json.dumps(doc, indent=1, default=_json), encoding="utf-8")
    adoc = alignment_doc(doc)
    (out / "alignment_carlos_aline.json").write_text(json.dumps(adoc, indent=1), encoding="utf-8")
    if alignment_out is not None:
        alignment_out.parent.mkdir(parents=True, exist_ok=True)
        alignment_out.write_text(json.dumps(adoc, indent=1) + "\n", encoding="utf-8")
    log(f"[carlos_aline_sync] -> {out / 'sync.json'}" + (f", {alignment_out.name}" if alignment_out else ""))
    return doc


def _json(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def alignment_doc(sync: dict) -> dict:
    """prototype/larissa_kadu/alignment_carlos_aline.json: Take.load reads `sources`; `songs` carries the song clocks."""
    srcs = [{"id": c["id"], "name": c["name"], "duration": c["duration_s"], "numbered": True, "slope": c["slope"], "offset": c["offset"],
             "accepted": True, "residual_ms": c["residual_ms"], "confidence": c["confidence"], "offset_source": c["offset_source"]}
            for c in sync["cameras"]]
    songs = {}
    for k, s in sync["songs"].items():
        if not s.get("found"):
            songs[k] = {"found": False}
            continue
        songs[k] = {"file": SONG_FILES[k], "title": SONG_INFO[k]["title"], "artist": SONG_INFO[k]["artist"],
                    "convention": "song_time = slope * reference_time + offset (song_time = the decoded mp3 from its first sample)",
                    "slope": s["clock"]["slope"], "offset": s["clock"]["offset_s"], "residual_ms": round(s["clock"]["residual_rms_ms"], 2),
                    "tempo_ratio_song_per_video": s["tempo_ratio_song_per_video"], "audible_reference_s": s.get("audible_reference_s"),
                    "song_zero_at_reference_s": round(s["song_zero_at_reference_s"], 3)}
    return {"reference": sync["reference"]["name"], "convention": sync["convention"],
            "note": ("audio clocks measured by dancecap.carlos_aline_sync; the reference clock is camera 01's own video time. AUDIO clocks: "
                     "dancecap.sync must still measure the per-camera VIDEO offsets before MAMMA ([sync] required)."),
            "sources": srcs, "songs": songs, "danced_span": sync["danced_span"], "lattice": sync["lattice"]}


def plot_overview(doc: dict, png: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(4, 1, figsize=(13, 13), gridspec_kw={"height_ratios": [2, 2, 2, 1.2]})
    pair = doc["video_pair"]["rows"]
    if pair:
        clk = doc["video_pair"]["decision"]
        ax[0].scatter([r["t_master"] for r in pair], [1000 * (r["offset_s"] - (clk["slope"] * r["t_cam"] + clk["offset_s"] - r["t_cam"])) for r in pair],
                      s=10, c=["tab:blue" if r.get("cluster") == "main" else "tab:red" for r in pair])
    ax[0].set_title("camera 02 on camera 01: window offset minus the fitted clock (ms)")
    col = {"kojo": "tab:orange", "guetta": "tab:green"}
    for name, s in doc["songs"].items():
        if not s.get("found"):
            continue
        w = s["fit"]["windows"]
        ax[1].scatter([r["t_cam"] for r in w], [r["resid_ms"] for r in w], s=10, c=col[name], label=f"{name} a={s['tempo_ratio_song_per_video']:.4f}")
        fine = s["fine"]
        ax[2].plot([r["t_cam"] for r in fine], [r["z"] for r in fine], c=col[name], label=name)
    ax[1].set_ylim(-80, 80)
    ax[1].set_title("song clock windows (12 s): residual to the song line (ms)")
    ax[1].legend()
    ax[2].axhline(Z_FINE, c="0.5", lw=0.6)
    ax[2].set_title("timeline windows (6 s): z along each song's line")
    ax[2].legend()
    for k, seg in enumerate(doc["timeline"]):
        ax[3].barh(0, seg["to_s"] - seg["from_s"], left=seg["from_s"], color={"none": "0.8", "mix": "tab:purple"}.get(seg["state"], col.get(seg["state"], "k")))
        ax[3].text(0.5 * (seg["from_s"] + seg["to_s"]), 0, seg["state"], ha="center", va="center", fontsize=8)
    sp = doc["danced_span"]
    ax[3].axvline(sp["start_s"], c="k", ls="--")
    ax[3].axvline(sp["end_s"], c="k", ls="--")
    ax[3].set_yticks([])
    ax[3].set_title("which song is audible (reference = camera 01 time, s); dashed = danced span")
    for a_ in ax:
        a_.set_xlim(0, 205)
    fig.tight_layout()
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=70)
    plt.close(fig)
    return png


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.carlos_aline_sync", description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["run", "plot", "alignment"])
    ap.add_argument("--src", type=Path, default=SRC_DIR)
    ap.add_argument("--out", type=Path, default=WORK / "sync")
    ap.add_argument("--alignment-out", type=Path, default=ALIGNMENT_OUT)
    ap.add_argument("--force-audio", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "plot":
        doc = json.loads((a.out / "sync.json").read_text(encoding="utf-8"))
        print(plot_overview(doc, a.out.parent / "review" / "sync_overview.png"))
        return 0
    if a.cmd == "alignment":
        doc = json.loads((a.out / "sync.json").read_text(encoding="utf-8"))
        adoc = alignment_doc(doc)
        (a.out / "alignment_carlos_aline.json").write_text(json.dumps(adoc, indent=1), encoding="utf-8")
        a.alignment_out.parent.mkdir(parents=True, exist_ok=True)
        a.alignment_out.write_text(json.dumps(adoc, indent=1) + "\n", encoding="utf-8")
        print(a.alignment_out.name)
        return 0
    doc = run(a.src, a.out, a.alignment_out, a.force_audio)
    plot_overview(doc, a.out.parent / "review" / "sync_overview.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
