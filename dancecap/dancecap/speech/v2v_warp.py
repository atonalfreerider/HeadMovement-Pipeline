"""Re-timing of a converted piece onto its source's word times (agent V2).

Speech-to-speech keeps the rhythm of the input only roughly; with a low "stability" it re-paces words inside a sentence (measured: 17 of 68 chosen pieces had
a local lag above 80 ms against the source), while the film's captions and 3D events follow the TRANSCRIPT clock. The global shift of `v2v.align_converted` cannot
fix a lag that changes inside a piece. This module follows the syllable rhythm:

    1. features: log energy of four speech bands (250-500, 500-1000, 1000-2000, 2000-4000 Hz) at a 10 ms hop for the source and the converted piece
       (floor-clipped and z-normalised, so the voice and the noise floor do not matter, only when and how much speech there is);
    2. DTW with a band (+-`BAND_S`) and a slope limit (the step pattern allows local rates 0.5-2), cost = distance of the feature vectors;
    3. the path becomes a monotone map  t_converted = f(t_source), smoothed (median + Gaussian) and rate-limited to [0.6, 1.6];
    4. WSOLA (waveform-similarity overlap-add) renders the converted audio along f without changing its pitch.

Only pieces whose local lag exceeds a threshold are re-timed (`needs_warp`), so most converted audio is left untouched.
"""
from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d, median_filter

BANDS = ((250, 500), (500, 1000), (1000, 2000), (2000, 4000))
FRAME_S = 0.01
BAND_S = 0.5
RATE_MIN, RATE_MAX = 0.6, 1.6


def band_features(x: np.ndarray, sr: int, frame_s: float = FRAME_S) -> np.ndarray:
    """[frames, 4] z-normalised log band energies (floor = 10th percentile of each band, speech clipped at +40 dB above it)"""
    n = int(round(frame_s * sr))
    cols = []
    for lo, hi in BANDS:
        sos = signal.butter(4, [lo, min(hi, sr / 2 - 100)], btype="band", fs=sr, output="sos")
        y = signal.sosfiltfilt(sos, x.astype(np.float64))
        m = len(y) // n
        e = 10 * np.log10((y[:m * n].reshape(m, n) ** 2).mean(axis=1) + 1e-12)
        e = np.clip(e - np.percentile(e, 10), 0, 40)
        cols.append((e - e.mean()) / (e.std() + 1e-9))
    return np.stack(cols, axis=1)


def dtw_path(a: np.ndarray, b: np.ndarray, band: int) -> np.ndarray:
    """DTW of feature sequences a [n, d] (source) and b [m, d] (converted) inside a band |i*m/n - j| <= band frames; steps (1,1), (1,2), (2,1) with a small
    penalty on the non-diagonal ones. Returns the path [(i, j), ...] from (0, 0) to (n-1, m-1)."""
    n, m = len(a), len(b)
    INF = 1e18
    D = np.full((n, m), INF)
    P = np.zeros((n, m), dtype=np.int8)
    cost = ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2)
    pen = 0.15
    for i in range(n):
        c = i * (m - 1) / max(n - 1, 1)
        j0, j1 = max(0, int(c - band)), min(m, int(c + band) + 1)
        for j in range(j0, j1):
            if i == 0 and j == 0:
                D[0, 0] = cost[0, 0]
                continue
            best, arg = INF, 0
            if i > 0 and j > 0 and D[i - 1, j - 1] < best:
                best, arg = D[i - 1, j - 1], 0
            if i > 0 and j > 1 and D[i - 1, j - 2] + pen < best:
                best, arg = D[i - 1, j - 2] + pen, 1
            if i > 1 and j > 0 and D[i - 2, j - 1] + pen < best:
                best, arg = D[i - 2, j - 1] + pen, 2
            if best < INF:
                D[i, j] = best + cost[i, j]
                P[i, j] = arg
    i, j = n - 1, m - 1
    if D[i, j] >= INF:                                    # the corner is outside the band (lengths differ a lot): end at the best reachable cell of the last row
        j = int(np.argmin(D[n - 1]))
    path = [(i, j)]
    while i > 0 or j > 0:
        s = P[i, j]
        if s == 0:
            i, j = i - 1, j - 1
        elif s == 1:
            i, j = i - 1, j - 2
        else:
            i, j = i - 2, j - 1
        if i < 0 or j < 0:
            break
        path.append((i, j))
    return np.array(path[::-1])


def path_to_map(path: np.ndarray, n: int, m: int, smooth_frames: float = 4.0) -> np.ndarray:
    """f[i] = converted frame (float) for every source frame i: the mean j of the path cells of row i, median + Gaussian smoothed, forced monotone and
    rate-limited to [RATE_MIN, RATE_MAX] frames per frame"""
    f = np.zeros(n)
    for i in range(n):
        js = path[path[:, 0] == i, 1]
        f[i] = js.mean() if len(js) else np.nan
    idx = np.arange(n)
    ok = ~np.isnan(f)
    f = np.interp(idx, idx[ok], f[ok])
    f = median_filter(f, size=7, mode="nearest")
    f = gaussian_filter1d(f, smooth_frames, mode="nearest")
    out = np.empty(n)
    out[0] = f[0]
    for i in range(1, n):
        step = f[i] - f[i - 1]
        out[i] = out[i - 1] + min(max(step, RATE_MIN), RATE_MAX)
    return out


def wsola(x: np.ndarray, sr: int, t_conv_of_out: np.ndarray, win_s: float = 0.032, hop_s: float = 0.008, search_s: float = 0.006) -> np.ndarray:
    """Render y such that y at output time t sounds like x at time t_conv_of_out(t) (seconds, one value per output sample index via interpolation), pitch
    preserved: WSOLA with a Hann window of win_s, output hop hop_s, and a waveform-similarity search of +-search_s around the nominal analysis position.
    `t_conv_of_out` has one entry per OUTPUT frame of hop_s (use `frame_map_to_hop`)."""
    W = int(win_s * sr)
    H = int(hop_s * sr)
    S = int(search_s * sr)
    win = np.hanning(W)
    n_out = (len(t_conv_of_out) - 1) * H + W
    y = np.zeros(n_out + W)
    norm = np.zeros(n_out + W)
    prev = None                                   # the analysis segment that follows the previous one naturally (for the similarity search)
    xpad = np.concatenate([np.zeros(W + S), x.astype(np.float64), np.zeros(2 * W + S)])
    off = W + S
    for k, tc in enumerate(t_conv_of_out):
        centre = int(round(tc * sr)) - W // 2
        start = centre
        if prev is not None:
            lo = max(-S, -(centre + off))
            best, bd = -1e18, 0
            ref = prev
            for d in range(lo, S + 1, 2):
                s0 = centre + d + off
                seg = xpad[s0:s0 + W]
                if len(seg) < W:
                    continue
                c = float(np.dot(seg, ref))
                if c > best:
                    best, bd = c, d
            start = centre + bd
        s0 = start + off
        seg = xpad[s0:s0 + W]
        if len(seg) < W:
            seg = np.pad(seg, (0, W - len(seg)))
        o = k * H
        y[o:o + W] += seg * win
        norm[o:o + W] += win
        nat = start + H                            # what would follow this segment naturally
        s1 = nat + off
        prev = xpad[s1:s1 + W]
        if len(prev) < W:
            prev = np.pad(prev, (0, W - len(prev)))
        prev = prev * win
    norm = np.maximum(norm, 1e-3)
    return (y / norm)[:n_out]


def frame_map_to_hop(f: np.ndarray, frame_s: float, hop_s: float, n_samples: int, sr: int) -> np.ndarray:
    """f[i] is the converted frame of source frame i; returns the converted TIME (s) for each output hop k (output time k*hop_s), clipped to the audio"""
    n_hops = int(np.ceil(n_samples / sr / hop_s))
    t_src = np.arange(n_hops) * hop_s / frame_s
    tc = np.interp(t_src, np.arange(len(f)), f) * frame_s
    return np.clip(tc, 0, None)


def retime(src: np.ndarray, conv: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
    """conv (already globally aligned, any length) re-timed to the source's rhythm; same length as src. Returns (audio, info{max_shift_ms, rate_min, rate_max})"""
    fs, fc = band_features(src, sr), band_features(conv, sr)
    path = dtw_path(fs, fc, int(BAND_S / FRAME_S))
    f = path_to_map(path, len(fs), len(fc))
    shift = (f - np.arange(len(f))) * FRAME_S
    tmap = frame_map_to_hop(f, FRAME_S, 0.008, len(src), sr)
    y = wsola(conv, sr, tmap)
    y = y[:len(src)] if len(y) >= len(src) else np.pad(y, (0, len(src) - len(y)))
    rates = np.diff(f)
    return y.astype(np.float32), {"max_shift_ms": round(float(np.abs(shift).max()) * 1000, 1), "rate_min": round(float(rates.min()), 2), "rate_max": round(float(rates.max()), 2)}


def needs_warp(local_lags_ms: list[float], threshold_ms: float = 50.0) -> bool:
    return bool(local_lags_ms) and max(abs(x) for x in local_lags_ms) > threshold_ms


def timing_shift_ms(src: np.ndarray, conv: np.ndarray, sr: int) -> dict:
    """how far the converted piece's rhythm is from the source's: DTW map of the speech-band features; returns {max_ms, p90_ms} of |f(i) - i| over the frames
    where the source has speech (energy of the lowest band group above its median)"""
    fs, fc = band_features(src, sr), band_features(conv, sr)
    path = dtw_path(fs, fc, int(BAND_S / FRAME_S))
    f = path_to_map(path, len(fs), len(fc))
    shift = np.abs(f - np.arange(len(f))) * FRAME_S * 1000
    act = fs.mean(axis=1) > np.percentile(fs.mean(axis=1), 40)
    s = shift[act] if act.any() else shift
    return {"max_ms": round(float(s.max()), 1), "p90_ms": round(float(np.percentile(s, 90)), 1)}
