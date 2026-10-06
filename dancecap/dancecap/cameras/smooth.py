"""Light, robust temporal smoothing of per-frame camera tracks (pose + log-focal).

Each signal (camera centre, unit quaternion, log focal) is fitted by a Gaussian-weighted local LINEAR
regression in reference time (sigma ~ 1-2 frames, so real handheld motion and fast zooms survive), with
Cauchy re-weighting of outliers (3 passes). Frames that failed localisation (weight 0) are filled by the
same regression when they have support within 3 sigma, else by linear / slerp interpolation between the
nearest good frames (held constant beyond the ends).
"""
from __future__ import annotations

import numpy as np


def _local_linear(t: np.ndarray, y: np.ndarray, w: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Weighted local linear fit at every t. Returns (yhat (F,D), support (F,) = sum of kernel weights)."""
    dt = t[None, :] - t[:, None]  # (i, j) = t_j - t_i
    K = np.exp(-0.5 * (dt / sigma) ** 2) * (np.abs(dt) <= 3 * sigma) * w[None, :]
    s0 = K.sum(1)
    s1 = (K * dt).sum(1)
    s2 = (K * dt * dt).sum(1)
    det = s0 * s2 - s1 ** 2
    # local linear where well conditioned, else local constant
    a = np.where(det > 1e-12 * np.maximum(s0, 1e-12) ** 2, s2 / np.where(det > 0, det, 1), 0)
    b = np.where(det > 1e-12 * np.maximum(s0, 1e-12) ** 2, -s1 / np.where(det > 0, det, 1), 0)
    lin = det > 1e-12 * np.maximum(s0, 1e-12) ** 2
    L = np.where(lin[:, None], K * (a[:, None] + b[:, None] * dt), K / np.maximum(s0, 1e-12)[:, None])
    return L @ y, s0


def robust_smooth(t: np.ndarray, y: np.ndarray, w: np.ndarray, sigma: float, iters: int = 3,
                  c: float = 3.0, min_scale: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Robust local-linear smoother. y (F,) or (F,D); w (F,) prior weights (0 = missing).

    Returns (smoothed y, final robust weights (F,), 0 for outliers/missing)."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    squeeze = y.ndim == 1
    Y = y.reshape(len(t), -1)
    Y = np.where(np.isfinite(Y), Y, 0.0)
    w0 = np.where(np.isfinite(y.reshape(len(t), -1)).all(1), np.asarray(w, float), 0.0)
    rw = w0.copy()
    yhat = Y.copy()
    for _ in range(iters):
        yhat, sup = _local_linear(t, Y, rw, sigma)
        r = np.linalg.norm(Y - yhat, axis=1)
        good = w0 > 0
        if good.sum() < 3:
            break
        scale = max(1.4826 * np.median(r[good]), min_scale, 1e-12)
        rw = w0 / (1 + (r / (c * scale)) ** 2)
    yhat, sup = _local_linear(t, Y, rw, sigma)
    # fill frames without support by interpolation between good frames
    # (support far below the typical full-kernel support = one-sided, extrapolating local fit)
    good = rw > 0.05 * max(rw.max(), 1e-12)
    weak = sup < 0.25 * (np.median(sup[good]) if good.any() else 1.0)
    good &= ~weak
    if weak.any() and good.sum() >= 1:
        for d in range(Y.shape[1]):
            yhat[weak, d] = np.interp(t[weak], t[good], yhat[good, d])
    out = yhat[:, 0] if squeeze else yhat
    return out, rw


def quat_from_R(R: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """Unit quaternions (x,y,z,w) with hemisphere continuity along the VALID frames (gaps are bridged)."""
    from scipy.spatial.transform import Rotation
    q = Rotation.from_matrix(R).as_quat()
    valid = np.ones(len(q), bool) if valid is None else np.asarray(valid, bool)
    last = None
    for i in range(len(q)):
        ref = last if last is not None else (q[i - 1] if i else None)
        if ref is not None and np.dot(q[i], ref) < 0:
            q[i] = -q[i]
        if valid[i]:
            last = q[i]
    return q


def R_from_quat(q: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    return Rotation.from_quat(q / np.linalg.norm(q, axis=1, keepdims=True)).as_matrix()


def smooth_track(t: np.ndarray, R: np.ndarray, tvec: np.ndarray, f: np.ndarray, ok: np.ndarray,
                 weight: np.ndarray | None = None, sigma_s: float = 0.04,
                 sigma_f_s: float = 0.04) -> dict[str, np.ndarray]:
    """Smooth one camera track. R (F,3,3) world->cam, tvec (F,3), f (F,) px, ok (F,) bool.

    Frames with ok=False must still carry a finite R (identity is fine); they get weight 0.
    Returns dict R, tvec, f, C (camera centres), outlier (bool: robust weight collapsed on an ok frame).
    """
    ok = np.asarray(ok, bool)
    w = ok.astype(float) if weight is None else np.where(ok, weight, 0.0)
    C = -np.einsum("fji,fj->fi", R, tvec)
    q = quat_from_R(np.where(ok[:, None, None], R, np.eye(3)), ok)
    # make missing quaternions continuous too (they get weight 0 anyway)
    Cs, rwc = robust_smooth(t, np.where(ok[:, None], C, np.nan), w, sigma_s)
    qs, rwq = robust_smooth(t, np.where(ok[:, None], q, np.nan), w, sigma_s)
    lf, rwf = robust_smooth(t, np.where(ok, np.log(np.where(f > 0, f, 1.0)), np.nan), w, sigma_f_s)
    Rs = R_from_quat(qs)
    ts = -np.einsum("fij,fj->fi", Rs, Cs)
    outlier = ok & ((rwc < 0.1 * w) | (rwq < 0.1 * w) | (rwf < 0.1 * w))
    return {"R": Rs, "tvec": ts, "f": np.exp(lf), "C": Cs, "outlier": outlier}
