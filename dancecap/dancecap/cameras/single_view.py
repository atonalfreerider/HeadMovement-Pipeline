"""Metric geometry of ONE panning / zooming phone camera from the people in front of it (no baseline, no triangulation).

Given the per-frame rotation R_i and focal f_i of a camera about a fixed centre (dancecap.cameras.ptz, in a reference frame whose axes
mean nothing physical), the 2D body keypoints of the people in view (dancecap.cameras.people2d) and statures, this module recovers
  1. gravity (the vertical direction of the reference frame) from the long axes of standing people (and architectural verticals),
  2. the height of the camera above the floor from every upright person (head / feet rays + eye height), which does NOT depend on the
     focal (a person's horizon ratio), and
  3. their distances (ground-plane positions), which DO scale with the focal: the same picture is explained by a longer lens and a farther
     camera (the dolly-zoom ambiguity). The focal therefore has to come from elsewhere (architecture, monocular priors) and its uncertainty
     is carried through as a table of variants, the way agent C2 did for the two-camera take.

Conventions: reference frame = frame 0 of the track (R_0 = I), camera axes x right, y down, z forward; R_i maps reference -> camera i
(so a reference vector v is seen as R_i v). World frame (CONTRACTS.md): right-handed, Y up, floor y = 0, metres.
Pure numpy; no files.
"""
from __future__ import annotations

import numpy as np

from .ptz import Lens, pix_to_ray, rotations

# COCO-17 indices
NOSE, LEYE, REYE, LEAR, REAR, LSH, RSH, LELB, RELB, LWR, RWR, LHIP, RHIP, LKNEE, RKNEE, LANK, RANK = range(17)

EYE_OVER_STATURE = 0.936            # Drillis & Contini: eye height / stature (adults)
ANKLE_JOINT_M = 0.09                # height of the COCO ankle point above the floor for a shod foot (C2 measured 0.089 + sole)


def rays_in_reference(pix: np.ndarray, R: np.ndarray, f: np.ndarray | float, lens: Lens) -> np.ndarray:
    """(M,2) pixels of frames with rotations R (M,3,3) (reference -> camera) and focal f -> (M,3) unit rays in the REFERENCE frame."""
    f = np.broadcast_to(np.asarray(f, float), (len(pix),))
    d = pix_to_ray(pix, f, lens)
    return np.einsum("mba,mb->ma", R, d)            # R^T d


def person_axis_normals(foot_ref: np.ndarray, head_ref: np.ndarray) -> np.ndarray:
    """Unit normals of the vertical planes through the camera centre, a person's foot ray and head ray (all in the reference frame)."""
    n = np.cross(foot_ref, head_ref)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def horizon_basis(g_up: np.ndarray, azimuth_ref: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Right-handed world axes (X, Y=g_up, Z) in the reference frame, X as close as possible to `azimuth_ref` (default: the reference x axis)."""
    y = g_up / np.linalg.norm(g_up)
    a = np.array([1.0, 0, 0]) if azimuth_ref is None else azimuth_ref
    x = a - y * (a @ y)
    x /= np.linalg.norm(x)
    z = np.cross(x, y)
    return x, y, z


def camera_height(foot: np.ndarray, eye: np.ndarray, eye_height_m: np.ndarray | float, ankle_m: float = ANKLE_JOINT_M) -> np.ndarray:
    """Height of the camera centre above the floor from rays of a standing person's foot and eye, both in a Y-UP frame (M,3 unit rays).

    The foot (ankle point, `ankle_m` above the floor) is at the intersection of its ray with the plane y = ankle_m; the eye is straight
    above it at `eye_height_m`. With the camera at (0, h, 0): foot = lam * rf (lam = (ankle - h) / rf_y), eye horizontal position = foot's,
    eye = mu * re with mu chosen to match horizontally, eye_y = h + mu * re_y.  Linear in h:  h = (y_e - a q) / (1 - q),
    q = c * re_y / rf_y,  c = (rf_h . re_h) / |re_h|^2.  NaN where the geometry degenerates (foot ray not below the horizon, ...)."""
    rf_h = foot[:, [0, 2]]
    re_h = eye[:, [0, 2]]
    c = (rf_h * re_h).sum(1) / np.maximum((re_h ** 2).sum(1), 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = c * eye[:, 1] / foot[:, 1]
        h = (np.asarray(eye_height_m) - ankle_m * q) / (1.0 - q)
    bad = (foot[:, 1] > -1e-3) | (c <= 0) | ~np.isfinite(h)
    h = np.where(bad, np.nan, h)
    return h


def floor_points(foot: np.ndarray, h: float | np.ndarray, ankle_m: float = ANKLE_JOINT_M) -> np.ndarray:
    """Intersection of Y-up foot rays with the plane y = ankle_m for a camera at height h: (M,3) positions relative to the point below the camera."""
    lam = (ankle_m - np.asarray(h)) / foot[:, 1]
    p = lam[:, None] * foot
    p[:, 1] = ankle_m
    return p


def mean_pose_keypoints(kp: np.ndarray) -> dict[str, np.ndarray]:
    """Ankle / eye / shoulder / hip midpoints with the confidence of the weakest of the pair ((D,17,3) COCO keypoints -> dict of (D,3) x y conf)."""
    def mid(a, b):
        p = 0.5 * (kp[:, a, :2] + kp[:, b, :2])
        return np.concatenate([p, np.minimum(kp[:, a, 2], kp[:, b, 2])[:, None]], 1)

    def low(a, b):                 # the lower of two points of a pair (larger y in the image), with its own confidence
        pick = kp[:, a, 1] > kp[:, b, 1]
        return np.where(pick[:, None], kp[:, a, :], kp[:, b, :])
    return {"ank": mid(LANK, RANK), "ank_low": low(LANK, RANK), "eye": mid(LEYE, REYE), "sh": mid(LSH, RSH), "hip": mid(LHIP, RHIP),
            "knee": mid(LKNEE, RKNEE), "nose": kp[:, NOSE, :]}


def eye_height(stature_m: float | np.ndarray) -> np.ndarray:
    return EYE_OVER_STATURE * np.asarray(stature_m, float)


def world_from_gravity(rays_ref: np.ndarray, g_up: np.ndarray) -> np.ndarray:
    """Reference-frame rays (M,3) -> Y-up frame (M,3): columns are the world axes (X, Y = g_up, Z) in the reference frame."""
    x, y, z = horizon_basis(g_up)
    return rays_ref @ np.stack([x, y, z], 1)


def gravity_grid(g0: np.ndarray, half_deg: float, step_deg: float):
    """Unit vectors g near g0 (tangent-plane grid): yields (g, da_deg, db_deg)."""
    g0 = g0 / np.linalg.norm(g0)
    e1 = np.cross(g0, [1.0, 0, 0]) if abs(g0[0]) < 0.9 else np.cross(g0, [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(g0, e1)
    ang = np.arange(-half_deg, half_deg + 1e-9, step_deg)
    for a in ang:
        for b in ang:
            g = g0 + np.tan(np.radians(a)) * e1 + np.tan(np.radians(b)) * e2
            yield g / np.linalg.norm(g), a, b


def horizon_spread(foot_ref: np.ndarray, eye_ref: np.ndarray, eye_m: np.ndarray, g_up: np.ndarray, w: np.ndarray | None = None
                   ) -> tuple[float, float, int]:
    """For a candidate gravity: the camera height every person implies and how consistent they are (robust spread: weighted MAD, in metres).
    A tilted horizon makes near and far people disagree, so the spread is minimal for the true gravity. Returns (spread, median h, n)."""
    f = world_from_gravity(foot_ref, g_up)
    e = world_from_gravity(eye_ref, g_up)
    if np.median(f[:, 1]) > 0:          # the Y axis must point UP: feet are below the horizon, so flip when the median foot ray points up
        f, e = f * [1, -1, 1], e * [1, -1, 1]
    h = camera_height(f, e, eye_m)
    ok = np.isfinite(h) & (h > 0.3) & (h < 3.5)
    if ok.sum() < 20:
        return np.inf, np.nan, int(ok.sum())
    hv = h[ok]
    med = float(np.median(hv))
    return float(np.median(np.abs(hv - med))), med, int(ok.sum())


def rescale_up(up_cam: np.ndarray, f_from: float, f_to: float) -> np.ndarray:
    """Gravity (UP, camera frame) seen with focal f_from, re-expressed for focal f_to so that the HORIZON stays on the same pixel row.
    The horizon is the line  (u_x (x - cx) + u_y (y - cy)) / f + u_z = 0  in pixels; for the same line with another focal u_z scales by f_from / f_to.
    A monocular depth model estimated the floor normal with ITS focal; a different focal hypothesis keeps the picture, so the pitch changes."""
    u = np.array([up_cam[0], up_cam[1], up_cam[2] * f_from / f_to], float)
    return u / np.linalg.norm(u)


def plane_ransac(P: np.ndarray, rng: np.random.Generator, tol: float = 0.06, iters: int = 200, up_prior: np.ndarray | None = None
                 ) -> tuple[np.ndarray, float, np.ndarray]:
    """Plane n . x + d = 0 (unit n, camera frame) through the points P (M,3) by RANSAC + one SVD refit on the inliers. When `up_prior` is
    given the normal is flipped to point to the same side (n . up_prior > 0). Returns (n, d, inlier mask)."""
    best = None
    for _ in range(iters):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n = n / nn
        d = -(n @ a)
        inl = int((np.abs(P @ n + d) < tol).sum())
        if best is None or inl > best[0]:
            best = (inl, n, d)
    _, n, d = best
    m = np.abs(P @ n + d) < tol
    A = P[m]
    c = A.mean(0)
    n = np.linalg.svd(A - c, full_matrices=False)[2][2]
    if up_prior is not None and n @ up_prior < 0:
        n = -n
    d = -(n @ c)
    return n, float(d), np.abs(P @ n + d) < tol


def world_basis(up_ref: np.ndarray, x_ref: np.ndarray, forward_ref: np.ndarray) -> np.ndarray:
    """Columns (X, Y, Z) of the world frame in reference coordinates: Y = up, X = x_ref made horizontal (sign chosen so that the camera's
    forward direction has a NEGATIVE Z component: Z points towards the camera), Z = X x Y (right handed)."""
    y = up_ref / np.linalg.norm(up_ref)
    x = x_ref - y * (x_ref @ y)
    x /= np.linalg.norm(x)
    z = np.cross(x, y)
    if forward_ref @ z > 0:
        x = -x
        z = -z
    return np.stack([x, y, z], 1)


def camera_arrays(rv: np.ndarray, f: np.ndarray, W: np.ndarray, C: np.ndarray, size: tuple[int, int], k1: float):
    """Per-frame cameras of the world package: R (cam_from_world), tvec, K, dist (k1), and the world centre C (fixed) for every frame.
    rv / f: reference-frame rotation vectors and focals of the track; W: world axes in reference coordinates (columns)."""
    Rref = rotations(rv)
    R = np.einsum("nab,bc->nac", Rref, W)
    tvec = -np.einsum("nab,b->na", R, C)
    n = len(f)
    K = np.zeros((n, 3, 3))
    K[:, 0, 0] = K[:, 1, 1] = f
    K[:, 0, 2], K[:, 1, 2], K[:, 2, 2] = size[0] / 2.0, size[1] / 2.0, 1.0
    return R, tvec, K, np.full(n, float(k1)), np.tile(np.asarray(C, float), (n, 1))


def fill_bad_frames(rv: np.ndarray, f: np.ndarray, ok: np.ndarray, max_gap: int = 30) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Replace the rotation / focal of frames whose solve is not ok by a slerp / log-linear interpolation between the nearest ok frames, when both
    lie within `max_gap` frames of each other (a longer gap stays as it is and stays not ok). Returns (rv, f, ok, interpolated)."""
    from scipy.spatial.transform import Rotation as Rot, Slerp
    rv, f = rv.copy(), f.copy()
    good = np.where(ok)[0]
    interp = np.zeros(len(ok), bool)
    ok2 = ok.copy()
    if len(good) < 2:
        return rv, f, ok2, interp
    sl = Slerp(good, Rot.from_rotvec(rv[good]))
    for i in np.where(~ok)[0]:
        k = np.searchsorted(good, i)
        if k == 0 or k >= len(good) or good[k] - good[k - 1] > max_gap:
            continue
        rv[i] = sl([i]).as_rotvec()[0]
        a, b = good[k - 1], good[k]
        u = (i - a) / (b - a)
        f[i] = float(np.exp((1 - u) * np.log(f[a]) + u * np.log(f[b])))
        ok2[i], interp[i] = True, True
    return rv, f, ok2, interp
