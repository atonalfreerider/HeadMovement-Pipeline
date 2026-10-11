"""john_lula: the starting point of a ONE-camera body fit (agent J3b, 2026-10-11). Pure numpy.

MAMMA's multi-view start (triangulated points) does not exist with one camera. What one camera and a known floor give: the 2D keypoints of the
RTMW whole-body skeleton (people/teachers_<cam>.npz `kp_wb`, scores 0..1) and the per-frame camera (K, R, t) in the metric world frame (floor y = 0).

    pelvis_positions(...)  per grid frame and dancer the world position of the pelvis: the hip-mid ray at the depth the feet fix (ankle rays meet the
                           floor plane 0.09 m up, J2's convention) or, with no ankle, at the depth the torso length fixes; gaps are filled over time.
    stature_targets(cfg)   the T-pose crown-to-sole heights the shape pass is held to (stature + shoe sole, cameras.world_scale's convention).

Written to <take>/mamma/mono_init.npz {pos (T, 2, 3), conf (T, 2), depth (T, 2), source (T, 2)} by `dancecap.jl_front mono-init`; the ma_3d
optimiser (mamma-turing utils/mono.py) turns pos into the initial translation and searches the heading. Not a result: only a start.
"""
from __future__ import annotations

import numpy as np

# COCO-WholeBody body keypoints
L_SHO, R_SHO, L_HIP, R_HIP, L_ANK, R_ANK = 5, 6, 11, 12, 15, 16
ANKLE_ABOVE_FLOOR_M = 0.09          # the ankle joint's height above the floor (cameras.single_view / J2)
TORSO_OVER_STATURE = 0.288          # shoulder-mid to hip-mid distance / stature (SMPL-X neutral, upright)
SCORE_MIN = 0.35


def undistort_px(uv: np.ndarray, K: np.ndarray, k1: float) -> np.ndarray:
    """SIMPLE_RADIAL k1 (applied on normalised coordinates, as mamma_dyn.distort_simple_radial) removed: fixed-point iteration."""
    f, cx, cy = K[0, 0], K[0, 2], K[1, 2]
    xd = (uv - np.array([cx, cy])) / f
    x = xd.copy()
    for _ in range(8):
        r2 = (x ** 2).sum(-1, keepdims=True)
        x = xd / (1.0 + k1 * r2)
    return x * f + np.array([cx, cy])


def ray_world(uv: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray, k1: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Camera centre (3,) and unit ray direction (3,) in the world for pixel uv (R, t: world -> camera)."""
    if k1:
        uv = undistort_px(np.asarray(uv, float), K, k1)
    d_cam = np.linalg.solve(K, np.array([uv[0], uv[1], 1.0]))
    d = R.T @ d_cam
    return -R.T @ t, d / np.linalg.norm(d)


def floor_point(uv, K, R, t, k1: float = 0.0, height: float = 0.0):
    """Intersection of the pixel's ray with the plane y = height (world, Y up). Returns (point (3,), camera depth z) or None when the ray does not
    descend to it in front of the camera."""
    C, d = ray_world(uv, K, R, t, k1)
    if abs(d[1]) < 1e-6:
        return None
    s = (height - C[1]) / d[1]
    if s <= 0:
        return None
    X = C + s * d
    return X, float((R @ X + t)[2])


def point_at_depth(uv, K, R, t, depth: float, k1: float = 0.0) -> np.ndarray:
    """World point whose camera-frame z is `depth` on the pixel's ray."""
    if k1:
        uv = undistort_px(np.asarray(uv, float), K, k1)
    d_cam = np.linalg.solve(K, np.array([uv[0], uv[1], 1.0]))
    return R.T @ (d_cam * (depth / d_cam[2]) - t)


def point_at_range(uv, K, R, t, rng: float, k1: float = 0.0) -> np.ndarray:
    """World point on the pixel's ray whose HORIZONTAL distance (x, z) from the camera centre is `rng` (the camera is pitched down: equal camera depth is
    not equal range for points at different heights)."""
    C, d = ray_world(uv, K, R, t, k1)
    h = float(np.hypot(d[0], d[2]))
    return C + d * (rng / max(h, 1e-9))


def _range_of(P: np.ndarray, C: np.ndarray) -> float:
    return float(np.hypot(P[0] - C[0], P[2] - C[2]))


def _fill(x: np.ndarray, ok: np.ndarray, win: int = 5) -> np.ndarray:
    """x (T,) or (T, d): median-filter the valid samples over `win` frames, linear interpolation across the gaps, hold at both ends."""
    x = np.asarray(x, float)
    T = len(x)
    idx = np.flatnonzero(ok)
    if len(idx) == 0:
        return np.full_like(x, np.nan)
    sm = x.copy()
    h = win // 2
    for i in idx:
        lo, hi = max(0, i - h), min(T, i + h + 1)
        sel = ok[lo:hi]
        sm[i] = np.median(x[lo:hi][sel], axis=0)
    if x.ndim == 1:
        return np.interp(np.arange(T), idx, sm[idx])
    return np.stack([np.interp(np.arange(T), idx, sm[idx, j]) for j in range(x.shape[1])], 1)


def pelvis_positions(kp: np.ndarray, K: np.ndarray, R: np.ndarray, tvec: np.ndarray, k1: np.ndarray, stature: float, cam_ok: np.ndarray | None = None,
                     skel_in_partner: np.ndarray | None = None) -> dict:
    """One dancer over T frames. "depth" in the result is the HORIZONTAL range from the camera centre (m). kp (T, 133, 3) [x, y, score]; K (T, 3, 3), R (T, 3, 3), tvec (T, 3), k1 (T,): the camera of each frame (world: Y up, floor y = 0);
    stature (m). Returns {pos (T, 3), depth (T,) horizontal range m, conf (T,) 1 where the frame's own keypoints fixed it, source (T,) 0 gap / 1 feet / 2 torso}.
    Frames whose skeleton lies on the partner (skel_in_partner > 0.5) or whose camera is not ok carry no measurement."""
    T = len(kp)
    cam_ok = np.ones(T, bool) if cam_ok is None else np.asarray(cam_ok, bool)
    bad = np.zeros(T, bool) if skel_in_partner is None else np.asarray(skel_in_partner) > 0.5
    depth = np.full(T, np.nan)
    px = np.full((T, 2), np.nan)
    src = np.zeros(T, int)
    torso_m = TORSO_OVER_STATURE * stature
    for i in range(T):
        if not cam_ok[i] or bad[i]:
            continue
        s = kp[i, :, 2]
        hips = [j for j in (L_HIP, R_HIP) if s[j] >= SCORE_MIN]
        if not hips:
            continue
        hip_px = kp[i, hips, :2].mean(0)
        px[i] = hip_px
        ank = [j for j in (L_ANK, R_ANK) if s[j] >= SCORE_MIN]
        zs = []
        for j in ank:
            r = floor_point(kp[i, j, :2], K[i], R[i], tvec[i], float(k1[i]), ANKLE_ABOVE_FLOOR_M)
            if r is not None:
                zs.append(_range_of(r[0], -R[i].T @ tvec[i]))
        if zs:
            depth[i], src[i] = float(np.mean(zs)), 1
            continue
        sh = [j for j in (L_SHO, R_SHO) if s[j] >= SCORE_MIN]
        if len(sh) == 2 and len(hips) == 2:        # torso length in the picture (foreshortening makes the depth too large: left as the init's bias)
            px_len = np.linalg.norm(kp[i, sh, :2].mean(0) - hip_px)
            if px_len > 20:
                P = point_at_depth(hip_px, K[i], R[i], tvec[i], float(K[i, 0, 0] * torso_m / px_len), float(k1[i]))
                depth[i], src[i] = _range_of(P, -R[i].T @ tvec[i]), 2
    ok = np.isfinite(depth)
    out = {"pos": np.full((T, 3), np.nan), "depth": np.full(T, np.nan), "conf": ok.astype(float), "source": src}
    if not ok.any():
        return out
    dep = _fill(depth, ok)
    uv = _fill(px, ok)
    pos = np.zeros((T, 3))
    for i in range(T):
        pos[i] = point_at_range(uv[i], K[i], R[i], tvec[i], dep[i], float(k1[i]))
    out.update({"pos": pos, "depth": dep})
    return out


def stature_targets(cfg: dict, soles: dict) -> dict:
    """{"0": rest height of the lead (stature + sole), "1": follow} for MAMMA_STATURE (MAMMA body ids: 0 lead, 1 follow)."""
    d = cfg["dancers"]
    return {"0": float(d["lead"]["stature_m"]) + float(soles["lead"]), "1": float(d["follow"]["stature_m"]) + float(soles["follow"])}
