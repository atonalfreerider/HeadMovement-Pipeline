r"""john_lula realism v2: FOOT SLIP measured against the video, and a stance FOOT LOCK post-process (2026-10-20).

Owner, 2026-10-20: "there is too much foot slip". This module measures the slip of a smplx-contract fit against the
PICTURE (not only against the fit itself) and writes a foot-locked copy of any fit folder (no pods, CPU, ~20 s a take).

MEASURE (`measure`)
  video feet   RTMW whole-body keypoints (people/teachers_01.npz kp_wb: ankle 15/16, big toe 17/20, small toe 18/21, heel
               19/22), kept only where they are the dancer's OWN (own_dancer: score >= KP_CONF, the teacher row confident and
               not on the partner, the point on the dancer's own SAM mask (+ MASK_TOL_PX) and not inside the partner's mask
               alone, or - SAM often stops at John's dark shoes on the light floor - within MASK_FAR_PX of the own mask and
               PARTNER_CLEAR_PX clear of the partner's; people/masks_01.npz) and the camera track ok (cameras/01.npz `ok`).
               L / R of the video feet are assigned to the fit's feet per frame by the smaller reprojection cost.
  floor rays   each keypoint's camera ray (per-frame K, R, centre; k1 undone) cut with a horizontal plane (y = 0 = the floor):
               at the fit point's OWN height (for "how far is the fit's foot from the camera ray") and at a FIXED height per
               keypoint (the dancer's median; for the fit-independent floor track: a per-frame fit height would turn a
               fitted heel raise into 16 cm along the ray). Along the ray 1 px = ~18 mm at 4.2 m (camera 1.43 m high).
  speeds       image speed after camera-motion compensation: the camera rotates / zooms about one centre (type_model
               fixed_centre), so a static point maps by the homography K1 R1 R0^T K0^-1 whatever its depth; |uv1 - H uv0|
               (px / frame at a 400 px tall person) is the foot's own motion; floor-position speed = ray-point motion.
  stance       per foot: compensated image speed < IMG_STILL_PX and floor speed < FLOOR_STILL_M (3-frame medians over the
               own keypoints), 1-frame gaps closed, the fit's lowest sole vertex < FIT_LOW_M, >= MIN_RUN frames; then
               SPLIT by the video's own floor track of the foot (the toes: they stay down through a heel raise; keypoint
               biases removed), split_still: a segment ends when the track leaves its anchor by LEAVE_LAT_M across /
               LEAVE_AL_M along the ray, and is kept only when its Theil-Sen drift is small (DRIFT_*): a slow glide - which
               the per-frame thresholds let through - is not pinned and no stick-slip staircase is made of it.
  slip         the fit's foot point (mean of the ankle and ball joints, horizontal) during each stance: mm / frame, per stance
               path (cm), spread (max from the stance median, cm), net; the fit's ankle joint / big-toe vertex vs the RTMW
               ankle / big-toe camera ray at the fit point's height (along / across the ray; the ankle carries a COCO-vs-
               SMPL-X definition offset of 5-9 cm along the ray, the big toe ~1 cm).

LOCK (`lock`; footlock_fit(src, dst): any fit folder -> a new folder; the source and the protected folders are never written)
  target   per stance run and foot ONE floor position of the BALL (SMPL-X toe joint 10 / 11): the fit's median, moved ACROSS the
           camera ray by the video (median of video_offset over the run: toe keypoints' fixed-height rays minus the fit,
           keypoint_bias removed; where reliable: RAY_MIN_SHARE observed, spread <= RAY_MAX_SPREAD_M, |shift| <=
           RAY_MAX_DIFF_M) and ALONG the ray x RAY_ALONG_W (the video's along-ray stance position is +-2.4-2.8 cm noise
           median, the fit's own depth is used); the heading pinned to the run's circular mean (YAW_MODE "auto"; a run in
           which the fit turns the foot > YAW_FREE_DEG keeps the fit's 2 Hz heading = a pivot on the ball; "fit" = always the
           fit's 2 Hz heading); the ankle behind the ball at the fit's own horizontal ankle-ball distance; heights (and so
           the foot's pitch / heel raises) stay the fit's own per frame.
  solve    (1) pelvis translation correction: horizontal, PELVIS_GAIN x the stance-weighted mean ankle correction, bridged
           over frames without stance, low-passed at PELVIS_HZ, bounded by PELVIS_MAX_M; (2) per frame analytic two-bone
           leg IK (hip + knee; the bend plane kept, so the knee never bends backwards) to the ankle target and an ankle
           rotation (<= ANKLE_MAX_DEG) turning the foot to the ball; a target out of reach (knee straighter than
           KNEE_MIN_DEG) is reached the way a planted foot does it - the heel rises about the ball (heel_raise_reach);
           corrections ramp in / out over RAMP frames (cosine; gaps <= 2 RAMP between two runs bridged); a foot on the
           floor outside a run keeps its world position against the pelvis shift; (3) no sole vertex below min(source, 0)
           - 1 mm (targets raised, re-solved). The other foot's own runs are solved the same way at the same time; the
           partner is not touched (each dancer alone).
  output   only transl and the hip / knee / ankle rotations change; joints = FK of the saved parameters (SMPL-X contract);
           every other key is the source's (reproj_px / quality / contact / stance are the SOURCE's: NOT recomputed), plus
           footlock_stance (T, 2), footlock_weight (T, 2), footlock_target (T, 2, 3 ball targets), footlock_pelvis_offset
           (T, 3) and the json report 'footlock'; dst/footlock.json.

    python -m dancecap.mono4d.jl_footlock measure [--fits best old] [--out ...]
    python -m dancecap.mono4d.jl_footlock lock   [--src DIR] [--dst DIR]      # default: best fit -> v2/footlock/smplx_footlock
    python -m dancecap.mono4d.jl_footlock check  [--before DIR] [--after DIR]   # slip, jitter, pelvis, knees, floor, 2D feet / body
    python -m dancecap.mono4d.jl_footlock sheet  [--before DIR] [--after DIR]   # before / after strip of 3 stance runs
Outputs: work/john_lula/realism/v2/footlock/ (video_feet.npz cache, measure.json, smplx_footlock/, check.json, sheets/).
Never touched: work/john_lula/{smplx, smplx_rf2_noslip, smplx_3db, avatar, export}/ and the source fit.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from ..take import PACKAGE_ROOT

TAKE, CAM = "john_lula", "01"
ROLES = ("lead", "follow")
NAME = {"lead": "John", "follow": "Lula"}
WHO = {"lead": 0, "follow": 1}
WORK = PACKAGE_ROOT / "work" / TAKE
V2 = WORK / "realism" / "v2"
OUT = V2 / "footlock"
BEST = WORK / "realism" / "refit" / "audit_patch" / "smplx_rf2_noslip_3db_patched"
OLD = WORK / "smplx_rf2_noslip"
FITS = {"best": BEST, "old": OLD}
DST = OUT / "smplx_footlock"
PROTECTED = [WORK / d for d in ("smplx", "smplx_rf2_noslip", "smplx_3db", "avatar", "export")] + [BEST]

FEET = ("left", "right")
LEG = {"left": (1, 4, 7, 10), "right": (2, 5, 8, 11)}               # hip, knee, ankle, toe (foot) joints of SMPL-X
KP = {"left": {"ankle": 15, "btoe": 17, "stoe": 18, "heel": 19}, "right": {"ankle": 16, "btoe": 20, "stoe": 21, "heel": 22}}
VTX = {"left": {"btoe": 5770, "stoe": 5780, "heel": 8846}, "right": {"btoe": 8463, "stoe": 8474, "heel": 8635}}
KP_NAMES = ("ankle", "btoe", "stoe", "heel")

KP_CONF = 0.5           # RTMW keypoint score
ROW_CONF = 0.5          # the teacher row's confidence (teachers_01.npz conf)
MASK_TOL_PX = 6         # own mask dilated by this many px (shoes at the mask's edge)
MASK_FAR_PX = 40        # ... or this far from the own mask when clear of the partner (SAM misses dark shoes)
PARTNER_CLEAR_PX = 12   # px from the partner's mask for the far test
REF_H_PX = 400.0        # image speeds are scaled to a person this tall (the take's median box: 422 / 401 px)
IMG_STILL_PX = 2.5      # compensated image speed of a still foot, px / frame at REF_H_PX (3-frame median)
FLOOR_STILL_M = 0.025   # floor-ray speed of a still foot, m / frame (3-frame median; along-ray noise ~ 18 mm / px)
FIT_LOW_M = 0.04        # the fit's lowest sole vertex below this = the fit's foot is near the floor
MIN_RUN = 4             # frames
LEAVE_LAT_M, LEAVE_AL_M = 0.025, 0.08   # a still segment ends when its video floor track leaves the anchor by this (across / along the ray)
DRIFT_LAT_M, DRIFT_AL_M = 0.015, 0.06   # ... and is kept only when its Theil-Sen drift is within this (+ 2 noise scales)
CAMC_DEFAULT = np.array([1.72844816, 1.43302452, 3.85261148])   # cameras/01.npz centre_world (fixed-centre track)
RAMP = 5                # frames of cosine blend at each run end
RAY_MIN_SHARE = 0.5     # share of a run's frames with an own-dancer ankle ray
RAY_MAX_SPREAD_M = 0.06  # median abs deviation of the run's ray points (m)
RAY_MAX_DIFF_M = 0.12   # the ray estimate may move the run's foot at most this far from the fit's median (m)
PELVIS_GAIN = 0.5
PELVIS_HZ = 1.0
PELVIS_MAX_M = 0.05
FLOOR_TOL_M = 0.001
CONTACT_LO, CONTACT_HI = 0.02, 0.06   # sole height (m): a foot below LO keeps its world position against the pelvis shift


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ================================================================================================ pure geometry
def camera_centre(R: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """(..., 3, 3), (..., 3) world->camera (Xc = R X + t) -> (..., 3) centre -R^T t."""
    return -np.einsum("...ji,...j->...i", R, tvec)


def undistort_k1(xd: np.ndarray, yd: np.ndarray, k1: float, iters: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of x_d = x (1 + k1 r^2) (normalised coordinates), fixed-point iteration."""
    x, y = xd.copy(), yd.copy()
    for _ in range(iters):
        d = 1.0 + k1 * (x * x + y * y)
        x, y = xd / d, yd / d
    return x, y


def pixel_rays(uv: np.ndarray, K: np.ndarray, R: np.ndarray, k1: float | np.ndarray = 0.0) -> np.ndarray:
    """(N, 2) pixels, (N, 3, 3) K, (N, 3, 3) world->camera R -> (N, 3) unit world ray directions."""
    xd = (uv[..., 0] - K[..., 0, 2]) / K[..., 0, 0]
    yd = (uv[..., 1] - K[..., 1, 2]) / K[..., 1, 1]
    x, y = undistort_k1(xd, yd, np.asarray(k1, float))
    dc = np.stack([x, y, np.ones_like(x)], -1)
    dw = np.einsum("...ji,...j->...i", R, dc)
    return dw / np.linalg.norm(dw, axis=-1, keepdims=True)


def ray_plane_y(C: np.ndarray, d: np.ndarray, h: np.ndarray | float) -> np.ndarray:
    """Points where rays C + s d (s > 0) meet the plane y = h; NaN where the ray does not reach it going forward."""
    h = np.asarray(h, float)
    dy = d[..., 1]
    s = (h - C[..., 1]) / np.where(np.abs(dy) > 1e-9, dy, np.nan)
    P = C + s[..., None] * d
    return np.where((s > 0)[..., None], P, np.nan)


def project(X: np.ndarray, R: np.ndarray, tvec: np.ndarray, K: np.ndarray, k1: float | np.ndarray = 0.0) -> np.ndarray:
    """(N, M, 3) world points with per-row camera (N, ...) -> (N, M, 2) pixels (NaN behind the camera)."""
    Xc = np.einsum("nab,nmb->nma", R, X) + tvec[:, None, :]
    z = Xc[..., 2]
    zs = np.where(z > 1e-6, z, np.nan)
    x, y = Xc[..., 0] / zs, Xc[..., 1] / zs
    k1 = np.asarray(k1, float).reshape(-1, 1) if np.ndim(k1) else float(k1)
    dd = 1 + k1 * (x * x + y * y)
    return np.stack([K[:, None, 0, 0] * x * dd + K[:, None, 0, 2], K[:, None, 1, 1] * y * dd + K[:, None, 1, 2]], -1)


def rotation_homography(K0: np.ndarray, R0: np.ndarray, K1: np.ndarray, R1: np.ndarray) -> np.ndarray:
    """Image map of static points between two views of a camera rotating / zooming about a fixed centre."""
    return K1 @ R1 @ np.swapaxes(R0, -1, -2) @ np.linalg.inv(K0)


def apply_h(H: np.ndarray, uv: np.ndarray) -> np.ndarray:
    p = np.einsum("...ab,...b->...a", H, np.concatenate([uv, np.ones(uv.shape[:-1] + (1,))], -1))
    return p[..., :2] / p[..., 2:3]


def compensated_speed(uv: np.ndarray, K: np.ndarray, R: np.ndarray) -> np.ndarray:
    """(T, 2) track (NaN = missing) -> (T,) px between frame i-1 and i after removing the camera's rotation / zoom
    (k1 ignored: |k1| ~ 0.01 changes a few-px step by < 0.1 px). Entry 0 is NaN."""
    out = np.full(len(uv), np.nan)
    H = rotation_homography(K[:-1], R[:-1], K[1:], R[1:])
    pred = apply_h(H, uv[:-1])
    out[1:] = np.linalg.norm(uv[1:] - pred, axis=-1)
    return out


def nanmedian_filter(x: np.ndarray, half: int = 1) -> np.ndarray:
    """Running median over +-half samples ignoring NaN (NaN where the window has no finite value)."""
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        w = x[max(0, i - half):i + half + 1]
        w = w[np.isfinite(w)]
        if len(w):
            out[i] = np.median(w)
    return out


def runs_of(m: np.ndarray, min_len: int = 1) -> list[tuple[int, int]]:
    """Inclusive (a, b) index ranges of the True runs of m that are at least min_len long."""
    m = np.asarray(m, bool)
    if not m.any():
        return []
    d = np.diff(np.r_[0, m.astype(np.int8), 0])
    a, b = np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1
    return [(int(x), int(y)) for x, y in zip(a, b) if y - x + 1 >= min_len]


def close_gaps(m: np.ndarray, gap: int = 1) -> np.ndarray:
    """Fill False gaps of at most `gap` samples that sit between True samples."""
    m = np.asarray(m, bool).copy()
    inv = runs_of(~m)
    for a, b in inv:
        if a > 0 and b < len(m) - 1 and b - a + 1 <= gap:
            m[a:b + 1] = True
    return m


def theil_sen_drift(y: np.ndarray) -> tuple[float, float]:
    """(n,) samples (NaN = missing) -> (drift over the span = median pairwise slope x (n - 1), its noise scale: 1.4826 x
    the MAD of the residuals about the line x sqrt(12 / m) with m finite samples; the drift of pure noise is ~ that)."""
    y = np.asarray(y, float)
    i = np.flatnonzero(np.isfinite(y))
    if len(i) < 3:
        return 0.0, np.inf
    di = i[None, :] - i[:, None]
    dy = y[i][None, :] - y[i][:, None]
    m = di > 0
    slope = float(np.median(dy[m] / di[m]))
    res = y[i] - slope * i
    res = res - np.median(res)
    sig = 1.4826 * float(np.median(np.abs(res)))
    return slope * (len(y) - 1), sig * np.sqrt(12.0 / len(i))


def split_still(lat: np.ndarray, al: np.ndarray, leave_lat: float, leave_al: float, drift_lat: float, drift_al: float,
                min_len: int) -> list[tuple[int, int]]:
    """A stance candidate's video floor track (lateral = across the camera ray, along the ray; NaN = no evidence) ->
    the sub-ranges (inclusive, local indices) in which the foot really stands still. Greedy: a segment ends when the
    5-sample median track leaves its anchor (median of the segment's first 3 finite samples) by more than leave_lat /
    leave_al; a segment is kept when it is at least min_len long and its Theil-Sen drift is within drift_* + 2 noise
    scales (a slow slide, cut into short segments by the leave test, fails the drift test: no stick-slip staircase)."""
    n = len(lat)
    sl, sa = nanmedian_filter(np.asarray(lat, float), 2), nanmedian_filter(np.asarray(al, float), 2)
    segs, i0, anc = [], 0, None
    seen: list[tuple[float, float]] = []
    for i in range(n):
        if np.isfinite(sl[i]) and np.isfinite(sa[i]):
            if anc is None:
                seen.append((sl[i], sa[i]))
                if len(seen) >= 3:
                    anc = np.median(np.array(seen), axis=0)
            elif abs(sl[i] - anc[0]) > leave_lat or abs(sa[i] - anc[1]) > leave_al:
                segs.append((i0, i - 1))
                i0, anc, seen = i, None, [(sl[i], sa[i])]
    segs.append((i0, n - 1))
    out = []
    for a, b in segs:
        if b - a + 1 < min_len:
            continue
        dl, el = theil_sen_drift(np.asarray(lat, float)[a:b + 1])
        da, ea = theil_sen_drift(np.asarray(al, float)[a:b + 1])
        if np.isfinite(el) and abs(dl) > drift_lat + 2 * el:
            continue
        if np.isfinite(ea) and abs(da) > drift_al + 2 * ea:
            continue
        out.append((a, b))
    return out


def separate_runs(runs: list[tuple[int, int]], min_gap: int, min_len: int) -> list[tuple[int, int]]:
    """Sorted inclusive runs -> the same with every gap between consecutive runs widened to >= min_gap frames by trimming
    both neighbours (half each), runs shorter than min_len after trimming dropped. Two stances of one foot with different
    floor targets must not touch: the foot then moves from one to the other over the gap (held_delta bridges it) instead
    of jumping between two frames."""
    out: list[tuple[int, int]] = []
    for a, b in sorted(runs):
        if out:
            pa, pb_ = out[-1]
            gap = a - pb_ - 1
            if gap < min_gap:
                need = min_gap - gap
                c1 = need // 2
                c2 = need - c1
                pb_, a = pb_ - c1, a + c2
                if pb_ - pa + 1 >= min_len:
                    out[-1] = (pa, pb_)
                else:
                    out.pop()
        if b - a + 1 >= min_len:
            out.append((a, b))
    # a dropped run can leave two survivors closer than min_gap: one more pass
    return out if all(y[0] - x[1] - 1 >= min_gap for x, y in zip(out[:-1], out[1:])) else separate_runs(out, min_gap, min_len)


def ramp_weights(T: int, runs: list[tuple[int, int]], n: int | None = None) -> np.ndarray:
    """1 inside the runs, a cosine fade over n (default RAMP) samples outside each run end, 0 elsewhere (max over
    overlapping runs)."""
    n = RAMP if n is None else n
    w = np.zeros(T)
    for a, b in runs:
        w[a:b + 1] = 1.0
        for k in range(1, n + 1):
            r = 0.5 * (1 + np.cos(np.pi * k / (n + 1)))
            if a - k >= 0:
                w[a - k] = max(w[a - k], r)
            if b + k < T:
                w[b + k] = max(w[b + k], r)
    return w


def lowpass(x: np.ndarray, fc: float, fs: float) -> np.ndarray:
    from scipy.signal import butter, filtfilt
    b, a = butter(2, fc / (fs / 2))
    return filtfilt(b, a, x, axis=0, padtype="odd", padlen=min(27, len(x) - 1))


def bound_norm(c: np.ndarray, cmax: float) -> np.ndarray:
    n = np.linalg.norm(c, axis=-1, keepdims=True)
    return c * np.minimum(1.0, cmax / np.maximum(n, 1e-12))


def min_rot(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """(N, 3) directions -> (N, 3, 3) minimal rotations taking u to v."""
    from scipy.spatial.transform import Rotation as Rot
    u = u / np.linalg.norm(u, axis=-1, keepdims=True)
    v = v / np.linalg.norm(v, axis=-1, keepdims=True)
    ax = np.cross(u, v)
    s = np.linalg.norm(ax, axis=-1, keepdims=True)
    ang = np.arctan2(s, np.sum(u * v, -1, keepdims=True))
    return Rot.from_rotvec(np.where(s > 1e-12, ax / np.maximum(s, 1e-12) * ang, 0.0)).as_matrix()


def two_bone(H: np.ndarray, K: np.ndarray, A: np.ndarray, At: np.ndarray, fwd: np.ndarray | None = None,
             min_bend_deg: float = 10.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Analytic two-bone IK. Hip H, knee K, ankle A (N, 3) and the ankle target At -> (new knee, reached ankle, clamped):
    bone lengths kept; the knee bends in the plane AND on the side of the CURRENT bend (the knee's offset from the
    ORIGINAL hip-ankle line, re-projected square to the new hip-target line), so the IK never turns a forward bend into a
    backward one; where the current bend is small (< min_bend_deg: its plane is ill-defined) fwd (N, 3: the thigh's
    forward = knee-cap direction, the hip frame's z) gives the plane: a near-straight knee bends forwards. A source knee
    bent BACKWARDS by more than min_bend_deg keeps its side (in this take those frames match the image's knee better as
    they are: a facing / twist error of the fit, not the lock's to fix). An unreachable target is clamped to the straight
    / folded limit along the hip-target line."""
    la, lb = np.linalg.norm(K - H, axis=1), np.linalg.norm(A - K, axis=1)
    v = At - H
    c = np.linalg.norm(v, axis=1)
    cmin, cmax = np.abs(la - lb) + 1e-4, la + lb - 1e-4
    clamped = (c > cmax) | (c < cmin)
    c = np.clip(c, cmin, cmax)
    d = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
    Ar = H + c[:, None] * d
    d0 = (A - H) / np.maximum(np.linalg.norm(A - H, axis=1, keepdims=True), 1e-12)
    p0 = (K - H) - np.sum((K - H) * d0, 1, keepdims=True) * d0
    if fwd is not None:
        u, w = K - H, A - K
        bend = np.degrees(np.arccos(np.clip(np.sum(u * w, 1) / (la * lb + 1e-12), -1, 1)))
        use = bend < min_bend_deg
        p0 = np.where(use[:, None], fwd, p0)
    pole = p0 - np.sum(p0 * d, 1, keepdims=True) * d
    pn = np.linalg.norm(pole, axis=1, keepdims=True)
    alt = np.cross(d, np.array([1.0, 0.0, 0.0]))                         # degenerate: any perpendicular
    pole = np.where(pn > 1e-9, pole / np.maximum(pn, 1e-12), alt / np.maximum(np.linalg.norm(alt, axis=1, keepdims=True), 1e-12))
    al = np.arccos(np.clip((la ** 2 + c ** 2 - lb ** 2) / (2 * la * c), -1, 1))
    Kn = H + la[:, None] * (np.cos(al)[:, None] * d + np.sin(al)[:, None] * pole)
    return Kn, Ar, clamped


def heel_raise_reach(H: np.ndarray, A_des: np.ndarray, B_des: np.ndarray, reach: np.ndarray, max_deg: float = 45.0,
                     step_deg: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """An ankle target the leg cannot reach (|A_des - H| > reach) is brought in the way a planted foot does it: the heel
    rises, the foot turning about the BALL B_des in the vertical plane of its heading (ankle-ball distance kept), by the
    smallest pitch (step_deg grid up to max_deg) that makes it reachable, else the pitch that comes closest.
    -> (new ankle targets (N, 3), pitch added (N,) deg; 0 where the target was reachable)."""
    A_des = np.asarray(A_des, float).copy()
    v = A_des - B_des
    hz = -v[:, [0, 2]]
    Lh = np.linalg.norm(hz, axis=1)
    hd = np.zeros_like(v)
    hd[:, [0, 2]] = hz / np.maximum(Lh, 1e-9)[:, None]                # heading: ankle -> ball, horizontal unit
    r = np.linalg.norm(v, axis=1)
    phi = np.arctan2(v[:, 1], Lh)
    need = np.linalg.norm(A_des - H, axis=1) > reach
    add = np.zeros(len(H))
    if need.any():
        grid = np.radians(np.arange(0.0, max_deg + 1e-9, step_deg))
        ph = phi[need, None] + grid[None]                                # (n, G)
        P = (B_des[need, None, :] + r[need, None, None] * (-np.cos(ph)[..., None] * hd[need, None, :]
                                                         + np.sin(ph)[..., None] * np.array([0.0, 1.0, 0.0])))
        d = np.linalg.norm(P - H[need, None, :], axis=-1)
        okg = d <= reach[need, None]
        gi = np.where(okg.any(1), okg.argmax(1), d.argmin(1))
        A_des[need] = P[np.arange(need.sum()), gi]
        add[need] = np.degrees(grid[gi])
    return A_des, add


def knee_flexion_deg(J: np.ndarray, foot: str) -> np.ndarray:
    """(T, >=12, 3) joints -> (T,) angle between the thigh and the shank (0 = straight)."""
    h, k, a, _ = LEG[foot]
    u, v = J[:, k] - J[:, h], J[:, a] - J[:, k]
    c = np.sum(u * v, -1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1) + 1e-12)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def knee_signed_flexion_deg(J: np.ndarray, Rw: np.ndarray, foot: str) -> np.ndarray:
    """(T, 55, 3) joints + (T, 55, 3, 3) world joint rotations -> (T,) knee flexion (deg) signed by the side of the
    hip-ankle line the knee lies on along the thigh's forward axis (the hip frame's z): + = a normal bend (knee in
    front), - = hyper-extension (knee behind the line)."""
    h, k, a, _ = LEG[foot]
    ang = knee_flexion_deg(J, foot)
    d = J[:, a] - J[:, h]
    d = d / np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-12)
    off = (J[:, k] - J[:, h]) - np.sum((J[:, k] - J[:, h]) * d, -1, keepdims=True) * d
    fwd = Rw[:, h][:, :, 2]
    return np.where(np.sum(off * fwd, -1) >= 0, ang, -ang)


def path_stats(P: np.ndarray) -> dict:
    """(n, 2) horizontal positions of one stance -> step mm / frame, path cm, spread cm (max from the median)."""
    st = 1e3 * np.linalg.norm(np.diff(P, axis=0), axis=1)
    med = np.median(P, axis=0)
    return {"mm_per_frame": float(np.mean(st)) if len(st) else 0.0, "path_cm": float(st.sum() / 10.0),
            "spread_cm": float(100 * np.linalg.norm(P - med, axis=1).max()), "net_cm": float(100 * np.linalg.norm(P[-1] - P[0]))}


# ================================================================================================ data
def load_fit(d: Path) -> dict:
    return {r: dict(np.load(Path(d) / f"{r}.npz", allow_pickle=True)) for r in ROLES}


def load_cams() -> dict:
    c = np.load(WORK / "cameras" / f"{CAM}.npz", allow_pickle=True)
    out = {k: np.asarray(c[k]) for k in ("t", "K", "R", "tvec", "dist", "ok")}
    out["C"] = camera_centre(out["R"].astype(float), out["tvec"].astype(float))
    return out


def frame_of(t_fit: np.ndarray, tc: np.ndarray) -> np.ndarray:
    j = np.clip(np.searchsorted(tc, t_fit), 1, len(tc) - 1)
    return np.where(np.abs(tc[j - 1] - t_fit) < np.abs(tc[j] - t_fit), j - 1, j)


def video_feet(force: bool = False) -> dict:
    """RTMW foot keypoints of both dancers with their distances to the two SAM masks (cached in v2/footlock/
    video_feet.npz): uv (F, 2 dancers, 2 feet, 4 kp, 2) px, score (F, 2, 2, 4), d_own / d_partner (F, 2, 2, 4) px from
    the point to the dancer's own / the partner's mask (0 = inside, capped at 255, 255 also = no mask), row_ok (F, 2),
    box_h (F, 2). The own-dancer test itself is own_dancer() (parameters, no re-decode)."""
    cache = OUT / "video_feet.npz"
    if cache.exists() and not force:
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    import cv2
    T_ = np.load(WORK / "people" / f"teachers_{CAM}.npz", allow_pickle=True)
    M = np.load(WORK / "people" / f"masks_{CAM}.npz")
    keys = set(M.files)
    kw = np.asarray(T_["kp_wb"], float)
    F = len(kw)
    ids = np.array([[KP[f][n] for n in KP_NAMES] for f in FEET])          # (2, 4)
    uv = kw[:, :, ids, :2]                                                 # (F, 2, 2, 4, 2)
    sc = kw[:, :, ids, 2]
    dist = np.full((F, 2, 2, 2, 4), 255.0, np.float32)                     # [frame, kp owner w, mask m, foot, kp]
    Hh, Ww = 960, 538
    t0 = time.time()
    for fr in range(F):
        for m in (0, 1):
            k = f"m_{fr}_{m}"
            mm = cv2.imdecode(np.frombuffer(M[k].tobytes(), np.uint8), cv2.IMREAD_UNCHANGED) if k in keys else None
            if mm is None or not (mm > 0).any():
                continue
            dt = cv2.distanceTransform(((mm > 0) == 0).astype(np.uint8), cv2.DIST_L2, 3)
            for w in (0, 1):
                xy = uv[fr, w].reshape(-1, 2)
                ok = np.isfinite(xy).all(1)
                xi = np.clip(np.round(np.nan_to_num(xy[:, 0])).astype(int), 0, Ww - 1)
                yi = np.clip(np.round(np.nan_to_num(xy[:, 1])).astype(int), 0, Hh - 1)
                inb = ok & (np.nan_to_num(xy[:, 0]) >= 0) & (np.nan_to_num(xy[:, 0]) < Ww) & (np.nan_to_num(xy[:, 1]) >= 0) & (np.nan_to_num(xy[:, 1]) < Hh)
                d = np.where(inb, np.minimum(dt[yi, xi], 255.0), 255.0)
                dist[fr, w, m] = d.reshape(2, 4)
        if fr % 500 == 0:
            log(f"video_feet masks {fr}/{F} ({time.time() - t0:.0f} s)")
    d_own = np.stack([dist[:, 0, 0], dist[:, 1, 1]], 1)
    d_par = np.stack([dist[:, 0, 1], dist[:, 1, 0]], 1)
    row_ok = (np.asarray(T_["conf"], float) >= ROW_CONF) & (np.asarray(T_["skel_in_partner"], float) <= 0.5)
    box = np.asarray(T_["box"], float)
    out = {"uv": uv, "score": sc, "d_own": d_own, "d_partner": d_par, "row_ok": row_ok,
           "box_h": box[..., 3] - box[..., 1], "t": np.asarray(T_["t"], float)}
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **out)
    return out


def own_dancer(d_own: np.ndarray, d_par: np.ndarray, tol: float = MASK_TOL_PX, far_tol: float = MASK_FAR_PX,
               clear: float = PARTNER_CLEAR_PX) -> np.ndarray:
    """A foot keypoint belongs to the dancer when it lies on (within tol px of) the own mask and not inside the
    partner's mask alone, OR - the SAM masks often stop at dark shoes on the light floor - within far_tol px of the own
    mask and at least clear px away from the partner's mask."""
    on_own = d_own <= tol
    in_par_only = (d_par <= 0) & (d_own > 0)
    near_free = (d_own <= far_tol) & (d_par >= clear)
    return (on_own & ~in_par_only) | near_free


class Fit:
    """FK of one dancer's fit: joints, the foot points (ankle, toe joint, big-toe / small-toe / heel vertices) and the
    lowest sole vertex per foot."""

    def __init__(self, z: dict):
        from .. import physics_body as pb
        self.z = z
        self.body = pb.load_body(z)
        self.idx = {f: self.body.sole_vertices(f, 0.02) for f in FEET}
        self.update()

    def update(self) -> None:
        from .. import physics_body as pb
        z = self.z
        self.R, self.p = pb.forward_kinematics(self.body, pb.full_pose(z), np.asarray(z["transl"], float))
        self.J = np.einsum("tjab,jb->tja", self.R, self.body.j_rest) + self.p
        self.sole_low, self.vtx = {}, {}
        for f in FEET:
            vid = np.array([VTX[f][n] for n in ("btoe", "stoe", "heel")])
            P = pb.lbs_points(self.body, self.R, self.p, np.r_[vid, self.idx[f]])
            self.vtx[f] = P[:, :3]
            self.sole_low[f] = P[:, 3:, 1].min(1)

    def points(self, f: str) -> np.ndarray:
        """(T, 4, 3): ankle joint, big toe, small toe, heel (the RTMW foot keypoints' counterparts)."""
        return np.concatenate([self.J[:, LEG[f][2]][:, None], self.vtx[f]], 1)

    def foot_point(self, f: str) -> np.ndarray:
        """(T, 3): mean of the ankle and toe (ball) joints."""
        return 0.5 * (self.J[:, LEG[f][2]] + self.J[:, LEG[f][3]])


# ================================================================================================ video evidence on the fit lattice
def evidence(fit: Fit, role: str, vf: dict, cams: dict) -> dict:
    """Per fit frame and fit foot: the video foot (L / R assigned to the fit's feet), its own-dancer validity, the
    compensated image speed, the floor-ray points of every keypoint at the fit's heights, the still mask."""
    t = np.asarray(fit.z["t"], float)
    T = len(t)
    tc = np.asarray(cams["t"], float)
    j = frame_of(t, tc)
    w = WHO[role]
    K, R, Cc, k1 = cams["K"][j].astype(float), cams["R"][j].astype(float), cams["C"][j].astype(float), cams["dist"][j].astype(float)
    cam_ok = cams["ok"][j].astype(bool) & (np.abs(tc[j] - t) <= 0.04)
    uv = vf["uv"][j, w]                     # (T, 2 video feet, 4, 2)
    valid = (vf["score"][j, w] >= KP_CONF) & own_dancer(vf["d_own"][j, w], vf["d_partner"][j, w]) & vf["row_ok"][j, w][:, None, None] & cam_ok[:, None, None]
    # L / R assignment: compare the video feet to the fit's projected feet (ankle + toes + heel)
    proj = {f: project(fit.points(f), R, cams["tvec"][j].astype(float), K, k1) for f in FEET}
    def cost(vfoot, f):
        e = np.linalg.norm(uv[:, vfoot] - proj[f], axis=-1)
        e = np.where(valid[:, vfoot], e, np.nan)
        n = np.isfinite(e).sum(1)
        return np.where(n > 0, np.nansum(e, 1) / np.maximum(n, 1), np.nan), n
    c_ll, n0 = cost(0, "left")
    c_rr, n1 = cost(1, "right")
    c_lr, _ = cost(1, "left")
    c_rl, _ = cost(0, "right")
    straight = np.nan_to_num(c_ll, nan=0) + np.nan_to_num(c_rr, nan=0)
    swapped = np.nan_to_num(c_lr, nan=0) + np.nan_to_num(c_rl, nan=0)
    have = (n0 > 0) | (n1 > 0)
    swap = have & (swapped < straight - 3.0)
    uvf = {"left": np.where(swap[:, None, None], uv[:, 1], uv[:, 0]), "right": np.where(swap[:, None, None], uv[:, 0], uv[:, 1])}
    vaf = {"left": np.where(swap[:, None], valid[:, 1], valid[:, 0]), "right": np.where(swap[:, None], valid[:, 0], valid[:, 1])}
    hpx = np.where(np.isfinite(vf["box_h"][j, w]) & (vf["box_h"][j, w] > 50), vf["box_h"][j, w], REF_H_PX)
    # fixed plane height per keypoint: the dancer's median height of that fit point while the sole is on the floor
    # (fit-independent floor tracks: a per-frame fit height would turn a fitted heel raise into 16 cm along the ray)
    hfix = np.zeros(4)
    for q in range(4):
        hs = np.concatenate([fit.points(f)[fit.sole_low[f] < 0.01, q, 1] for f in FEET])
        hfix[q] = float(np.median(hs)) if len(hs) else 0.0
    out = {"t": t, "frame": j, "cam_ok": cam_ok, "swap": swap, "feet": {}, "h_fixed": hfix}
    for f in FEET:
        P = fit.points(f)
        # image speed per keypoint, consecutive valid frames only, scaled to REF_H_PX
        sp = np.full((T, 4), np.nan)
        rays = np.full((T, 4, 3), np.nan)
        rays_fix = np.full((T, 4, 3), np.nan)
        for q in range(4):
            u = np.where(vaf[f][:, q, None], uvf[f][:, q], np.nan)
            s = compensated_speed(u, K, R)
            sp[:, q] = s * REF_H_PX / hpx
            d = pixel_rays(np.nan_to_num(u), K, R, k1)
            rp = ray_plane_y(Cc, d, P[:, q, 1])
            rays[:, q] = np.where(np.isfinite(u).all(1)[:, None], rp, np.nan)
            rf = ray_plane_y(Cc, d, np.full(T, hfix[q]))
            rays_fix[:, q] = np.where(np.isfinite(u).all(1)[:, None], rf, np.nan)
        img_sp = np.nanmedian(np.where(np.isfinite(sp), sp, np.nan), axis=1) if np.isfinite(sp).any() else np.full(T, np.nan)
        fl = np.linalg.norm(np.diff(rays[..., [0, 2]], axis=0), axis=-1)       # (T-1, 4)
        fl_sp = np.full(T, np.nan)
        with np.errstate(all="ignore"):
            fl_sp[1:] = np.nanmedian(fl, axis=1)
        img_m, fl_m = nanmedian_filter(img_sp, 1), nanmedian_filter(fl_sp, 1)
        n_valid = vaf[f].sum(1)
        still = (n_valid >= 1) & (img_m < IMG_STILL_PX) & (fl_m < FLOOR_STILL_M)
        out["feet"][f] = {"img_speed": img_sp, "floor_speed": fl_sp, "img_speed_m3": img_m, "floor_speed_m3": fl_m,
                          "rays": rays, "rays_fixed": rays_fix, "valid": vaf[f], "uv": uvf[f], "proj": proj[f], "still": still, "n_valid": n_valid}
    return out


def stance_runs(fit: Fit, ev: dict, centre: np.ndarray | None = None, report: dict | None = None) -> dict:
    """Video stance runs per foot on the fit lattice: still in the video (compensated image speed and floor-ray speed,
    1-frame gaps closed) AND the fit's sole low, for >= MIN_RUN frames; then each candidate is split by the video's own
    floor track of the foot (the fit's ball moved by video_offset, lateral / along the camera ray) so that a slow slide
    (zouk feet glide) is not pinned: split_still with LEAVE_* / DRIFT_*. report (optional) gets the counts."""
    if centre is None:
        centre = CAMC_DEFAULT
    cand = {}
    for f in FEET:
        m = ev["feet"][f]["still"] & (fit.sole_low[f] < FIT_LOW_M)
        m = close_gaps(m, 1) & (fit.sole_low[f] < FIT_LOW_M)
        cand[f] = runs_of(m, MIN_RUN)
    bias = keypoint_bias(fit, ev, cand)
    out, n_cand, n_split, n_drop = {}, 0, 0, 0
    for f in FEET:
        B = fit.J[:, LEG[f][3]][:, [0, 2]]
        V = B + video_offset(fit, ev, f, bias)
        lst = []
        for a, b in cand[f]:
            n_cand += 1
            v = V[a:b + 1]
            u = np.nanmean(v, axis=0) - np.asarray(centre, float)[[0, 2]] if np.isfinite(v).any() else B[a] - np.asarray(centre, float)[[0, 2]]
            u = u / max(float(np.linalg.norm(u)), 1e-9)
            nn = np.array([-u[1], u[0]])
            sub = split_still(v @ nn, v @ u, LEAVE_LAT_M, LEAVE_AL_M, DRIFT_LAT_M, DRIFT_AL_M, MIN_RUN)
            if len(sub) != 1 or sub[0] != (0, b - a):
                n_split += 1
            if not sub:
                n_drop += 1
            lst += [(a + x, a + y) for x, y in sub]
        out[f] = separate_runs(lst, RAMP, MIN_RUN)
    if report is not None:
        report.update({"candidates": n_cand, "split_or_trimmed": n_split, "dropped_as_slide": n_drop,
                       "runs": sum(len(v) for v in out.values())})
    return out


# ================================================================================================ slip metrics
def along_across(P: np.ndarray, Q: np.ndarray, centre: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Horizontal P - Q (n, 3) -> (distance, along, across) with along = component on the camera's horizontal viewing
    direction to Q (+ = P farther from the camera than Q), across = the perpendicular component."""
    dv = (P - Q)[:, [0, 2]]
    u = Q[:, [0, 2]] - np.asarray(centre, float)[[0, 2]]
    u = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-9)
    return np.linalg.norm(dv, axis=1), np.sum(dv * u, 1), u[:, 0] * dv[:, 1] - u[:, 1] * dv[:, 0]


def video_foot_point(fit: Fit, ev: dict, f: str) -> np.ndarray:
    """(T, 3) the foot point (ankle-toe joint mid) implied by the RTMW BIG TOE alone: its camera ray cut at the fit's
    big-toe vertex height plus the fit's own big-toe-to-foot-point offset (NaN without an own-dancer big toe)."""
    return ev["feet"][f]["rays"][:, 1] + (fit.foot_point(f) - fit.points(f)[:, 1])


TRACK_KP = (1, 2)       # big toe, small toe: they stay on the floor through a heel raise (ankle / heel do not)


def keypoint_bias(fit: Fit, ev: dict, runs: dict) -> np.ndarray:
    """(4, 2) per foot keypoint (ankle, big toe, small toe, heel) the median horizontal offset (fixed-height ray point -
    fit point) over the dancer's stance frames (both feet): the keypoint-definition offsets between RTMW and the
    SMPL-X points plus any constant bias of the fit. Zero where a keypoint has fewer than 20 frames."""
    acc = [[] for _ in KP_NAMES]
    for f in FEET:
        P = fit.points(f)
        rays = ev["feet"][f]["rays_fixed"]
        for a, b in runs[f]:
            for q in range(4):
                d = (rays[a:b + 1, q] - P[a:b + 1, q])[:, [0, 2]]
                acc[q].append(d[np.isfinite(d).all(1)])
    out = np.zeros((4, 2))
    for q in range(4):
        d = np.concatenate(acc[q]) if acc[q] else np.zeros((0, 2))
        if len(d) >= 20:
            out[q] = np.median(d, axis=0)
    return out


def video_offset(fit: Fit, ev: dict, f: str, bias: np.ndarray, kps=TRACK_KP) -> np.ndarray:
    """(T, 2) the horizontal shift of the fit's foot that the video asks for: per frame the mean over the own-dancer
    TOE keypoints of (fixed-height ray point - fit point - keypoint bias); NaN without a toe keypoint."""
    P = fit.points(f)
    d = (ev["feet"][f]["rays_fixed"] - P)[..., [0, 2]] - bias[None]
    d = d[:, list(kps)]
    with np.errstate(all="ignore"):
        return np.nanmean(np.where(np.isfinite(d).all(-1, keepdims=True), d, np.nan), axis=1)


def _q(x, p, nd=2):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return round(float(np.percentile(x, p)), nd) if len(x) else None


def slip_report(fit: Fit, ev: dict, runs: dict, centre: np.ndarray) -> dict:
    """The fit's foot slide during the video stance runs and its distance from the camera-ray floor points."""
    allstep, perrun = [], []
    cmp_ = {"ankle": ([], [], []), "btoe": ([], [], [])}
    for f in FEET:
        Pf = fit.foot_point(f)[:, [0, 2]]
        pts = fit.points(f)
        rays = ev["feet"][f]["rays"]
        for a, b in runs[f]:
            s = path_stats(Pf[a:b + 1])
            s.update({"foot": f, "a": a, "b": b, "n": b - a + 1, "t0": round(float(ev["t"][a]), 3)})
            allstep.append(1e3 * np.linalg.norm(np.diff(Pf[a:b + 1], axis=0), axis=1))
            for name, q in (("ankle", 0), ("btoe", 1)):
                r = rays[a:b + 1, q]
                ok = np.isfinite(r).all(1)
                if ok.any():
                    d, al, ac = along_across(pts[a:b + 1, q][ok], r[ok], centre)
                    for lst, v in zip(cmp_[name], (d, al, ac)):
                        lst.append(v)
                    s[f"fit_minus_ray_{name}_along_cm"] = round(float(100 * np.median(al)), 1)
                    s[f"fit_minus_ray_{name}_across_cm"] = round(float(100 * np.median(ac)), 1)
            perrun.append(s)
    st = np.concatenate(allstep) if allstep else np.zeros(0)
    rep = {"stance_runs": len(perrun), "stance_frames": int(sum(r["n"] for r in perrun))}
    rep["stance_share_of_foot_frames"] = round(rep["stance_frames"] / (2 * len(fit.J)), 3)
    rep["slide_mm_per_frame"] = {"mean": round(float(st.mean()), 2) if len(st) else None, "median": _q(st, 50), "p90": _q(st, 90), "p99": _q(st, 99)}
    for k in ("path_cm", "spread_cm", "net_cm"):
        v = np.array([r[k] for r in perrun])
        rep[f"per_stance_{k}"] = {"median": _q(v, 50), "p90": _q(v, 90), "max": round(float(v.max()), 1) if len(v) else None}
    v = np.array([r["spread_cm"] for r in perrun])
    rep["stances_spread_over_3cm_share"] = round(float((v > 3).mean()), 3) if len(v) else None
    rep["stances_spread_over_5cm_share"] = round(float((v > 5).mean()), 3) if len(v) else None
    for name, (D, AL, AC) in cmp_.items():
        if not D:
            continue
        d, al, ac = np.concatenate(D), np.concatenate(AL), np.concatenate(AC)
        rep[f"fit_vs_ray_{name}_cm"] = {"frames": int(len(d)), "dist_median": round(float(100 * np.median(d)), 1),
                                        "dist_p90": round(float(100 * np.percentile(d, 90)), 1),
                                        "along_median": round(float(100 * np.median(al)), 1),
                                        "along_mad": round(float(100 * np.median(np.abs(al - np.median(al)))), 1),
                                        "across_median": round(float(100 * np.median(ac)), 1),
                                        "across_mad": round(float(100 * np.median(np.abs(ac - np.median(ac)))), 1)}
    rep["fit_vs_ray_note"] = ("fit point (ankle joint / big-toe vertex) minus the RTMW keypoint's camera ray cut at that fit point's own "
                              "height, stance frames; +along = the fit's point is farther from the camera than the ray point; along-ray "
                              "sensitivity ~18 mm per px at 4.2 m (camera 1.43 m high), across ~5 mm per px")
    rep["runs"] = perrun
    return rep


def fit_floor_skate(fit: Fit, contact_m: float = 0.03) -> dict:
    """The fit-only measure (jl_refit.sole_metrics style): horizontal motion of the foot point between consecutive
    frames in which the foot's lowest sole vertex is below contact_m."""
    out = []
    for f in FEET:
        P = fit.foot_point(f)[:, [0, 2]]
        down = (fit.sole_low[f][:-1] < contact_m) & (fit.sole_low[f][1:] < contact_m)
        out.append(1e3 * np.linalg.norm(np.diff(P, axis=0), axis=1)[down])
    s = np.concatenate(out)
    return {"down_frames": int(len(s)), "median_mm": _q(s, 50), "p90_mm": _q(s, 90), "over_10mm_share": round(float((s > 10).mean()), 4)}


def _resolve(key) -> Path:
    return FITS[key] if key in FITS else Path(key)


def measure(fit_keys=("best", "old"), out: Path | None = None, runs_from: str | None = "best") -> dict:
    """measure.json: per fit and dancer the video stance runs, the fit's slide during them and the fit-vs-ray distances.
    The stance runs of `runs_from` (default: the best fit's) are used for every fit, so the numbers compare like for like
    (each fit's own run count is listed too)."""
    cams = load_cams()
    centre = np.median(cams["C"], axis=0)
    vf = video_feet()
    rep = {"take": TAKE, "params": _params(), "fits": {}}
    ref_runs: dict = {}
    order = list(fit_keys)
    if runs_from in order:
        order.remove(runs_from)
        order.insert(0, runs_from)
    for key in order:
        F = load_fit(_resolve(key))
        rep["fits"][key] = {"dir": str(_resolve(key)), "roles": {}}
        for role in ROLES:
            fit = Fit(F[role])
            ev = evidence(fit, role, vf, cams)
            own_runs = stance_runs(fit, ev, centre)
            if key == runs_from:
                ref_runs[role] = own_runs
            runs = ref_runs.get(role, own_runs)
            r = slip_report(fit, ev, runs, centre)
            r["own_stance_runs"] = sum(len(v) for v in own_runs.values())
            r["fit_only_skate_3cm"] = fit_floor_skate(fit)
            r["video_evidence"] = {f: {"frames_with_own_kp_share": round(float((ev["feet"][f]["n_valid"] > 0).mean()), 3),
                                       "still_share": round(float(ev["feet"][f]["still"].mean()), 3)} for f in FEET}
            r["lr_swapped_share"] = round(float(ev["swap"].mean()), 3)
            rep["fits"][key]["roles"][role] = r
            log(f"measure {key} {role}: {r['stance_runs']} runs, slide {r['slide_mm_per_frame']}, spread {r['per_stance_spread_cm']}")
    p = out or OUT / "measure.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rep, indent=1, default=float))
    return rep


def _params() -> dict:
    return {k: globals()[k] for k in ("KP_CONF", "ROW_CONF", "MASK_TOL_PX", "MASK_FAR_PX", "PARTNER_CLEAR_PX", "REF_H_PX", "IMG_STILL_PX",
                                     "FLOOR_STILL_M", "FIT_LOW_M", "MIN_RUN", "RAMP", "RAY_MIN_SHARE", "RAY_MAX_SPREAD_M", "RAY_MAX_DIFF_M",
                                     "PELVIS_GAIN", "PELVIS_HZ", "PELVIS_MAX_M", "FLOOR_TOL_M", "CONTACT_LO", "CONTACT_HI", "ANKLE_MAX_DEG",
                                     "RAY_ALONG_W", "YAW_FREE_DEG", "YAW_MODE", "KNEE_MIN_DEG", "LEAVE_LAT_M", "LEAVE_AL_M", "DRIFT_LAT_M", "DRIFT_AL_M")}


# ================================================================================================ lock
ANKLE_MAX_DEG = 45.0    # the foot-direction correction at the ankle is capped at this angle (heel raises included)
KNEE_MIN_DEG = 2.0      # the IK never straightens a knee beyond this flexion: the heel rises about the ball instead (heel_raise_reach)
YAW_MODE = "auto"       # foot heading in a run: "auto" = pinned unless the fit turns it > YAW_FREE_DEG, "fit" = the fit's (2 Hz low-pass)
RAY_ALONG_W = 0.0       # share of the video's ALONG-ray stance position used (across the ray: all of it)
YAW_FREE_DEG = 20.0     # a run in which the fit turns the foot more than this keeps the fit's (low-passed) heading


def held_delta(delta_in: np.ndarray, runs: list[tuple[int, int]], n: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(T, D) correction defined inside the runs -> (correction, weight) on all frames: inside a run the run's own
    values with weight 1; up to n frames outside a run end the end value held under a cosine fade (ramp_weights); a
    gap of at most 2 n frames between two runs is bridged by linear interpolation of the two end values at weight 1."""
    n = RAMP if n is None else n
    T = len(delta_in)
    out = np.zeros_like(delta_in, dtype=float)
    w = ramp_weights(T, runs, n)
    rs = sorted(runs)
    for a, b in rs:
        out[a:b + 1] = delta_in[a:b + 1]
    for i, (a, b) in enumerate(rs):
        pb_ = rs[i - 1][1] if i > 0 else None
        if pb_ is not None and a - pb_ - 1 <= 2 * n:
            g0, g1 = pb_ + 1, a - 1
            if g1 >= g0:
                x = np.linspace(0.0, 1.0, g1 - g0 + 3)[1:-1].reshape((-1,) + (1,) * (delta_in.ndim - 1))
                out[g0:g1 + 1] = (1 - x) * delta_in[pb_] + x * delta_in[a]
                w[g0:g1 + 1] = 1.0
        else:
            for k in range(1, n + 1):
                if a - k >= 0:
                    out[a - k] = delta_in[a]
        na = rs[i + 1][0] if i + 1 < len(rs) else None
        if na is None or na - b - 1 > 2 * n:
            for k in range(1, n + 1):
                if b + k < T:
                    out[b + k] = delta_in[b]
    return out, w


def bridge(m: np.ndarray, have: np.ndarray) -> np.ndarray:
    """(T, D) values valid where `have` -> linearly interpolated over the gaps, held at the ends (zeros if none)."""
    T = len(m)
    if not have.any():
        return np.zeros_like(m)
    x = np.arange(T)
    return np.stack([np.interp(x, x[have], m[have, d]) for d in range(m.shape[1])], 1)


def circ_mean(ang: np.ndarray) -> float:
    return float(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()))


def wrap_pi(x: np.ndarray) -> np.ndarray:
    return (np.asarray(x, float) + np.pi) % (2 * np.pi) - np.pi


def run_targets(fit: Fit, ev: dict, runs: dict, centre: np.ndarray, along_w: float | None = None) -> dict:
    """One floor target per stance run and foot for the BALL of the foot (the SMPL-X toe joint 10 / 11), horizontal:
    the fit's median ball position of the run, moved by the video's request (video_offset: all own-dancer foot
    keypoints' camera rays, keypoint_bias removed; median over the run) where that is reliable (RAY_MIN_SHARE observed,
    spread <= RAY_MAX_SPREAD_M, within RAY_MAX_DIFF_M of the fit): ACROSS the camera ray fully, ALONG the ray x along_w
    (default RAY_ALONG_W; the video's along-ray position of a stance is +-2.4-2.8 cm noise median at 4.2 m, 18 mm per px).
    The foot's heading is pinned to the run's circular-mean yaw unless the fit turns the foot more than YAW_FREE_DEG within
    the run (a pivot: then the fit's yaw, low-passed); heights and the foot's pitch stay the fit's own per frame."""
    along_w = RAY_ALONG_W if along_w is None else along_w
    out: dict = {}
    bias = keypoint_bias(fit, ev, runs)
    out["_bias_cm"] = {n: [round(float(100 * x), 1) for x in bias[q]] for q, n in enumerate(KP_NAMES)}
    fs = 1.0 / float(np.median(np.diff(np.asarray(fit.z["t"], float))))
    for f in FEET:
        h, k, a_, t_ = LEG[f]
        B = fit.J[:, t_]
        off = video_offset(fit, ev, f, bias)
        lst = []
        for a, b in runs[f]:
            seg = slice(a, b + 1)
            m_fit = np.median(B[seg][:, [0, 2]], axis=0)
            v = off[seg]
            ok = np.isfinite(v).all(1)
            share = float(ok.mean())
            rec = {"a": a, "b": b, "n": b - a + 1, "share": round(share, 3)}
            src, tgt = "fit", m_fit.copy()
            if ok.sum() >= 2:
                d = np.median(v[ok], axis=0)
                spread = float(np.median(np.linalg.norm(v[ok] - d, axis=1)))
                u = m_fit - np.asarray(centre, float)[[0, 2]]
                u = u / max(float(np.linalg.norm(u)), 1e-9)
                n_ = np.array([-u[1], u[0]])
                al, ac = float(d @ u), float(d @ n_)
                rec.update({"video_spread_cm": round(100 * spread, 1), "video_minus_fit_along_cm": round(100 * al, 1),
                            "video_minus_fit_across_cm": round(100 * ac, 1)})
                if share >= RAY_MIN_SHARE and spread <= RAY_MAX_SPREAD_M and float(np.linalg.norm(d)) <= RAY_MAX_DIFF_M:
                    src, tgt = "ray", m_fit + along_w * al * u + ac * n_
            vec = (fit.J[seg, t_] - fit.J[seg, a_])[:, [0, 2]]
            ang = np.arctan2(vec[:, 1], vec[:, 0])
            mang = circ_mean(ang)
            dev = np.degrees(np.abs(wrap_pi(ang - mang)))
            if len(ang) >= 8 and (YAW_MODE == "fit" or dev.max() > YAW_FREE_DEG):
                yaw = mang + lowpass(wrap_pi(np.unwrap(ang) - mang), 2.0, fs)
                ymode = "fit_lowpass"
            else:
                yaw = np.full(len(ang), mang)
                ymode = "pinned"
            rec.update({"src": src, "ball_xz": [round(float(x), 4) for x in tgt], "fit_median_ball_xz": [round(float(x), 4) for x in m_fit],
                        "yaw_mode": ymode, "yaw_dev_max_deg": round(float(dev.max()), 1), "_tgt": tgt, "_yaw": yaw})
            lst.append(rec)
        out[f] = lst
    return out


def run_frame_targets(fit: Fit, f: str, rec: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame ankle and ball targets (n, 3) of one run: the ball at the run's floor target and the fit's own ball
    height; the ankle behind it along the pinned heading at the fit's own horizontal ankle-ball distance and height."""
    _, _, a_, t_ = LEG[f]
    seg = slice(rec["a"], rec["b"] + 1)
    A, B = fit.J[seg, a_], fit.J[seg, t_]
    Lh = np.linalg.norm((B - A)[:, [0, 2]], axis=1)
    tgt, yaw = rec["_tgt"], rec["_yaw"]
    Bt = np.stack([np.full(len(B), tgt[0]), B[:, 1], np.full(len(B), tgt[1])], 1)
    At = np.stack([tgt[0] - Lh * np.cos(yaw), A[:, 1], tgt[1] - Lh * np.sin(yaw)], 1)
    return At, Bt


def _rotvec_cap(Q: np.ndarray, max_deg: float) -> np.ndarray:
    from scipy.spatial.transform import Rotation as Rot
    if not len(Q):
        return Q
    rv = Rot.from_matrix(Q).as_rotvec()
    n = np.linalg.norm(rv, axis=1, keepdims=True)
    cap = np.radians(max_deg)
    rv = rv * np.minimum(1.0, cap / np.maximum(n, 1e-12))
    return Rot.from_rotvec(rv).as_matrix()


def solve_leg(body, R: np.ndarray, J: np.ndarray, bp: np.ndarray, f: str, fr: np.ndarray, A_des: np.ndarray,
              T_des: np.ndarray) -> dict:
    """Hip + knee (two_bone) to put the ankle at A_des and the ankle rotation that turns the ankle->toe vector to
    T_des - ankle, on frames fr of the base state (R, J world rotations / joints). Writes bp (T, 21, 3) in place."""
    from scipy.spatial.transform import Rotation as Rot
    if not len(fr):
        return {"frames": 0, "clamped": 0}
    h, k, a, t_ = LEG[f]
    H, Kp, A, To = J[fr, h], J[fr, k], J[fr, a], J[fr, t_]
    la, lb = np.linalg.norm(Kp - H, axis=1), np.linalg.norm(A - Kp, axis=1)
    reach = np.sqrt(la ** 2 + lb ** 2 + 2 * la * lb * np.cos(np.radians(KNEE_MIN_DEG)))
    A_des, raised = heel_raise_reach(H, A_des, T_des, reach)
    Kn, Ar, clamped = two_bone(H, Kp, A, A_des, fwd=R[fr, h][:, :, 2])
    Q1 = min_rot(Kp - H, Kn - H)
    Q2 = min_rot(A - Kp, Ar - Kn)
    Rh2, Rk2 = Q1 @ R[fr, h], Q2 @ R[fr, k]
    Q3 = _rotvec_cap(min_rot(To - A, T_des - A_des), ANKLE_MAX_DEG)   # the foot direction asked for (an ankle miss shifts the toe with it)
    clamped = clamped | (np.linalg.norm(Ar - A_des, axis=1) > 1e-4)
    Ra2 = Q3 @ R[fr, a]
    tr = lambda M: np.swapaxes(M, -1, -2)   # noqa: E731
    bp[fr, h - 1] = Rot.from_matrix(tr(R[fr, body.parents[h]]) @ Rh2).as_rotvec()
    bp[fr, k - 1] = Rot.from_matrix(tr(Rh2) @ Rk2).as_rotvec()
    bp[fr, a - 1] = Rot.from_matrix(tr(Rk2) @ Ra2).as_rotvec()
    ang = lambda Q: np.degrees(np.linalg.norm(Rot.from_matrix(Q).as_rotvec(), axis=1))   # noqa: E731
    return {"frames": int(len(fr)), "clamped": int(clamped.sum()), "heel_raised": int((raised > 0).sum()),
            "heel_raise_deg_p95": _q(raised[raised > 0], 95, 1) if (raised > 0).any() else 0.0, "ankle_miss_mm_max": round(float(1e3 * np.linalg.norm(Ar - A_des, axis=1).max()), 1),
            "hip_deg_p95": _q(ang(Q1), 95, 1), "knee_deg_p95": _q(ang(Q2), 95, 1), "ankle_deg_p95": _q(ang(Q3), 95, 1)}


def lock_dancer(z: dict, ev: dict, runs: dict, centre: np.ndarray, gain: float = PELVIS_GAIN, pelvis_hz: float = PELVIS_HZ,
                pelvis_max: float = PELVIS_MAX_M) -> tuple[dict, dict, dict]:
    """Foot lock of one dancer (module doc, LOCK). -> (new npz dict, report, aux arrays)."""
    z = {k: (np.array(v) if isinstance(v, np.ndarray) else v) for k, v in z.items()}
    t = np.asarray(z["t"], float)
    T = len(t)
    fs = 1.0 / float(np.median(np.diff(t)))
    fit = Fit(z)
    fit0 = Fit(dict(z))
    low0 = {f: fit.sole_low[f].copy() for f in FEET}
    tg = run_targets(fit, ev, runs, centre)
    # per-foot ankle / toe corrections (relative to the source) and weights
    dA, dT, W = {}, {}, {}
    for f in FEET:
        _, _, a_, t_ = LEG[f]
        din_a, din_t = np.zeros((T, 3)), np.zeros((T, 3))
        rr = [(r["a"], r["b"]) for r in tg[f]]
        for r in tg[f]:
            seg = slice(r["a"], r["b"] + 1)
            At, Bt = run_frame_targets(fit, f, r)
            din_a[seg] = At - fit.J[seg, a_]
            din_t[seg] = Bt - fit.J[seg, t_]
        dA[f], W[f] = held_delta(din_a, rr)
        dT[f], _ = held_delta(din_t, rr)
    # (1) pelvis translation correction: horizontal, from the stance-weighted mean ankle correction
    den = W["left"] + W["right"]
    num = W["left"][:, None] * dA["left"] + W["right"][:, None] * dA["right"]
    have = den > 0.05
    m = np.zeros((T, 3))
    m[have] = num[have] / den[have, None]
    m[:, 1] = 0.0
    c = bridge(m, have)
    c = gain * lowpass(c, pelvis_hz, fs) if gain > 0 else np.zeros((T, 3))
    c[:, 1] = 0.0
    c = bound_norm(c, pelvis_max)
    z["transl"] = np.asarray(z["transl"], float) + c
    fit.update()
    R0, J0 = fit.R.copy(), fit.J.copy()
    bp0 = np.asarray(z["body_pose"], float).reshape(T, 21, 3).copy()
    # (2) desired ankle / toe after the pelvis shift
    des = {}
    for f in FEET:
        _, _, a_, t_ = LEG[f]
        g = np.clip((CONTACT_HI - low0[f]) / (CONTACT_HI - CONTACT_LO), 0.0, 1.0)
        w = W[f][:, None]
        # in / near a run: the correction left after the pelvis shift; elsewhere a foot on the floor keeps its world position
        des[f] = (J0[:, a_] + w * (dA[f] - c) + (1 - w) * g[:, None] * (-c),
                  J0[:, t_] + w * (dT[f] - c) + (1 - w) * g[:, None] * (-c))
    rep = {"pelvis_correction_mm": {"max": round(float(1e3 * np.linalg.norm(c, axis=1).max()), 1),
                                    "rms": round(float(1e3 * np.sqrt(np.mean(np.sum(c ** 2, 1)))), 1)},
           "targets": {f: {"runs": len(tg[f]), "ray": sum(r["src"] == "ray" for r in tg[f]), "fit_median": sum(r["src"] == "fit" for r in tg[f])} for f in FEET},
           "keypoint_bias_cm": tg["_bias_cm"], "ik": {}, "floor_raise_iterations": 0}
    lift = {f: np.zeros(T) for f in FEET}
    worst = 0.0
    for it in range(4):
        bp = bp0.copy()
        for f in FEET:
            _, _, a_, t_ = LEG[f]
            Ad, Td = des[f][0].copy(), des[f][1].copy()
            Ad[:, 1] += lift[f]
            Td[:, 1] += lift[f]
            fr = np.flatnonzero((np.linalg.norm(Ad - J0[:, a_], axis=1) > 1e-6) | (np.linalg.norm(Td - J0[:, t_], axis=1) > 1e-6))
            rep["ik"][f] = solve_leg(fit.body, R0, J0, bp, f, fr, Ad[fr], Td[fr])
        z["body_pose"] = bp.reshape(T, 63)
        fit.update()
        worst = 0.0
        for f in FEET:
            lim = np.minimum(low0[f], 0.0) - FLOOR_TOL_M
            deficit = lim - fit.sole_low[f]
            bad = deficit > 0
            worst = max(worst, float(deficit.max()))
            lift[f] = lift[f] + np.where(bad, deficit + 0.0005, 0.0)
        if worst <= 0:
            break
        rep["floor_raise_iterations"] = it + 1
    rep["floor_below_source_mm_max"] = round(1e3 * max(0.0, worst), 2)
    jo = np.asarray(z["joints"], float)
    z["joints"] = np.concatenate([fit.J, jo[:, 55:]], 1) if jo.shape[1] > 55 else fit.J
    st = np.zeros((T, 2), bool)
    tgt = np.full((T, 2, 3), np.nan)
    for q, f in enumerate(FEET):
        for r in tg[f]:
            st[r["a"]:r["b"] + 1, q] = True
            tgt[r["a"]:r["b"] + 1, q] = run_frame_targets(fit0, f, r)[1]
    aux = {"footlock_stance": st, "footlock_weight": np.stack([W["left"], W["right"]], 1).astype(np.float32),
           "footlock_target": tgt.astype(np.float32), "footlock_pelvis_offset": c.astype(np.float32)}
    rep["runs"] = {f: [{k: v for k, v in r.items() if not k.startswith("_")} for r in tg[f]] for f in FEET}
    rep["yaw_modes"] = {f: {m_: sum(r["yaw_mode"] == m_ for r in tg[f]) for m_ in ("pinned", "fit_lowpass")} for f in FEET}
    return z, rep, aux


def footlock_fit(src: Path, dst: Path, gain: float = PELVIS_GAIN, pelvis_hz: float = PELVIS_HZ,
                 pelvis_max: float = PELVIS_MAX_M, roles=ROLES) -> dict:
    """Any fit folder ({lead,follow}.npz, smplx contract) -> a new foot-locked folder (dst must differ from src and
    from the protected folders). Roles not in `roles` are copied unchanged."""
    src, dst = Path(src).resolve(), Path(dst).resolve()
    if dst == src or any(dst == p.resolve() for p in PROTECTED):
        raise SystemExit(f"refusing to write {dst}")
    cams = load_cams()
    vf = video_feet()
    F = load_fit(src)
    dst.mkdir(parents=True, exist_ok=True)
    report = {"src": str(src), "dst": str(dst), "params": _params() | {"PELVIS_GAIN": gain, "PELVIS_HZ": pelvis_hz, "PELVIS_MAX_M": pelvis_max},
              "made": time.strftime("%Y-%m-%d %H:%M:%S"), "roles": {}}
    for role in ROLES:
        z = F[role]
        if role in roles:
            fit = Fit(z)
            ev = evidence(fit, role, vf, cams)
            srep: dict = {}
            runs = stance_runs(fit, ev, np.median(cams["C"], axis=0), srep)
            zn, rep, aux = lock_dancer(z, ev, runs, np.median(cams["C"], axis=0), gain, pelvis_hz, pelvis_max)
            zn.update(aux)
            note = {"module": "dancecap.mono4d.jl_footlock", "src": str(src), "changed": ["transl", "body_pose (hips, knees, ankles)", "joints (FK)"],
                    "carried_from_source": "every other key incl. reproj_px / quality / contact / stance (NOT recomputed)"} | rep
            zn["footlock"] = np.array(json.dumps(note, default=float))
            src_note = str(z.get("source", ""))
            zn["source"] = np.array((src_note + " | " if src_note else "") + f"jl_footlock (video stance foot lock, {time.strftime('%Y-%m-%d')})")
            rep["stance_detection"] = srep
            report["roles"][role] = {k: v for k, v in rep.items() if k != "runs"}
            report["roles"][role]["runs"] = rep["runs"]
            log(f"footlock {role}: {json.dumps({k: v for k, v in rep.items() if k != 'runs'}, default=float)[:700]}")
        else:
            zn = z
        np.savez_compressed(dst / f"{role}.npz", **zn)
    (dst / "footlock.json").write_text(json.dumps(report, indent=1, default=float))
    return report


# ================================================================================================ check
def _accel_mm(X: np.ndarray) -> np.ndarray:
    """(T, 3) -> (T,) |second difference| mm / frame^2 (NaN at the ends)."""
    a = np.full(len(X), np.nan)
    a[1:-1] = 1e3 * np.linalg.norm(X[2:] - 2 * X[1:-1] + X[:-2], axis=-1)
    return a


def _stat(x, nd=2) -> dict:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0}
    return {"n": int(len(x)), "median": round(float(np.median(x)), nd), "p95": round(float(np.percentile(x, 95)), nd),
            "p99": round(float(np.percentile(x, 99)), nd), "max": round(float(x.max()), nd)}


def reproj_feet(fit: Fit, ev: dict, mask: np.ndarray | None = None) -> dict:
    """2D: the fit's ankle joint / big-toe / small-toe / heel vertices through the per-frame camera vs the own-dancer
    RTMW foot keypoints (L / R assigned as in evidence()), px at 538 x 960; mask (T,) bool restricts the frames."""
    cams = load_cams()
    j = ev["frame"]
    out = {}
    allv = []
    for f in FEET:
        e = ev["feet"][f]
        proj = project(fit.points(f), cams["R"][j].astype(float), cams["tvec"][j].astype(float), cams["K"][j].astype(float),
                       cams["dist"][j].astype(float))
        d = np.linalg.norm(e["uv"] - proj, axis=-1)
        v = e["valid"] & np.isfinite(d)
        if mask is not None:
            v &= mask[:, None]
        for q, n in enumerate(KP_NAMES):
            out.setdefault(n, []).append(d[v[:, q], q])
        allv.append(d[v])
    res = {n: _stat(np.concatenate(v), 2) for n, v in out.items()}
    res["all"] = _stat(np.concatenate(allv), 2)
    return res


def body_reproj(J: np.ndarray, role: str, t: np.ndarray) -> dict:
    """jl_fitcheck.reproj's measure (the 12 COCO body joints of the fit through the per-frame camera vs J3a's RTMW COCO-17,
    both confidences >= 0.5, skeleton not on the partner, per-frame median in % of the person's box height), on arrays
    loaded once (jl_fitcheck reads its NpzFile per frame: minutes per call)."""
    from .. import jl_fitcheck as FC
    cams = load_cams()
    T_ = np.load(WORK / "people" / f"teachers_{CAM}.npz", allow_pickle=True)
    kp_all, conf, sip, box = (np.asarray(T_[k]) for k in ("kp", "conf", "skel_in_partner", "box"))
    tc = np.asarray(cams["t"], float)
    j, near = FC.frame_map(t, tc)
    ok = cams["ok"][j].astype(bool) & near
    ri = WHO[role]
    coco = list(FC.COCO_TO_SMPLX)
    err = np.full(len(t), np.nan)
    for i in np.flatnonzero(ok):
        k = int(j[i])
        if conf[k, ri] < FC.CONF or sip[k, ri] > 0.5:
            continue
        kp = kp_all[k, ri]
        ids = [c for c in coco if kp[c, 2] >= FC.CONF]
        if len(ids) < 6:
            continue
        uv, z = FC.project(J[i, [FC.COCO_TO_SMPLX[c] for c in ids]], cams["R"][k].astype(float), cams["tvec"][k].astype(float),
                           cams["K"][k].astype(float), float(np.asarray(cams["dist"][k]).reshape(-1)[0]))
        hpx = max(float(box[k, ri, 3] - box[k, ri, 1]), 50.0)
        e = 100.0 * np.linalg.norm(uv - kp[ids, :2], axis=-1) / hpx
        if (z > 0).sum() >= 6:
            err[i] = float(np.median(e[z > 0]))
    fin = err[np.isfinite(err)]
    return {"frames": int(len(fin)), "median": round(float(np.median(fin)), 3), "p90": round(float(np.percentile(fin, 90)), 3), "_per_frame": err}


def check(before: Path = BEST, after: Path = DST, out: Path | None = None) -> dict:
    """Before / after: slip in the (before) video stance runs, jitter (accel_rr), pelvis acceleration, knees, floor,
    2D feet vs RTMW, parameter changes."""
    from scipy.spatial.transform import Rotation as Rot
    from .candidates import accel_rr_mm
    cams = load_cams()
    centre = np.median(cams["C"], axis=0)
    vf = video_feet()
    FB, FA = load_fit(before), load_fit(after)
    rep = {"before": str(before), "after": str(after), "roles": {}}

    for role in ROLES:
        fb, fa = Fit(FB[role]), Fit(FA[role])
        evb = evidence(fb, role, vf, cams)
        eva = evidence(fa, role, vf, cams)
        runs = stance_runs(fb, evb, centre)
        st = np.zeros(len(fb.J), bool)
        for f in FEET:
            for a, b in runs[f]:
                st[a:b + 1] = True
        t = np.asarray(fb.z["t"], float)
        r = {}
        sb, sa = slip_report(fb, evb, runs, centre), slip_report(fa, eva, runs, centre)
        keys = ("slide_mm_per_frame", "per_stance_path_cm", "per_stance_spread_cm", "per_stance_net_cm", "stances_spread_over_3cm_share",
                "stances_spread_over_5cm_share", "fit_vs_ray_ankle_cm", "fit_vs_ray_btoe_cm", "stance_runs", "stance_frames")
        r["slip"] = {k: {"before": sb.get(k), "after": sa.get(k)} for k in keys}
        r["fit_only_skate_3cm"] = {"before": fit_floor_skate(fb), "after": fit_floor_skate(fa)}
        ab, aa = accel_rr_mm(t, fb.J), accel_rr_mm(t, fa.J)
        inc = aa - ab
        r["accel_rr_mm"] = {"before": _stat(ab), "after": _stat(aa), "increase": _stat(inc),
                            "frames_up_over_5": int(np.nansum(inc > 5)), "frames_up_over_10": int(np.nansum(inc > 10))}
        pb_, pa_ = _accel_mm(fb.J[:, 0]), _accel_mm(fa.J[:, 0])
        r["pelvis_accel_mm"] = {"before": _stat(pb_), "after": _stat(pa_), "frames_up_over_5": int(np.nansum((pa_ - pb_) > 5))}
        leg = {}
        for f in FEET:
            a_ = LEG[f][2]
            xb, xa = _accel_mm(fb.J[:, a_]), _accel_mm(fa.J[:, a_])
            kb, ka = knee_flexion_deg(fb.J, f), knee_flexion_deg(fa.J, f)
            sgb = knee_signed_flexion_deg(fb.J, fb.R, f)
            sga = knee_signed_flexion_deg(fa.J, fa.R, f)
            leg[f] = {"ankle_accel_mm": {"before": _stat(xb), "after": _stat(xa), "frames_up_over_10": int(np.nansum((xa - xb) > 10))},
                      "knee_flexion_deg_min": {"before": round(float(kb.min()), 2), "after": round(float(ka.min()), 2)},
                      "knee_straight_lt_2deg_frames": {"before": int((kb < 2).sum()), "after": int((ka < 2).sum())},
                      "knee_hyperextended_lt_-1deg_frames": {"before": int((sgb < -1).sum()), "after": int((sga < -1).sum())},
                      "knee_signed_flexion_deg_min": {"before": round(float(sgb.min()), 2), "after": round(float(sga.min()), 2)},
                      "knee_bent_back_lt_-3deg_frames": {"before": int((sgb < -3).sum()), "after": int((sga < -3).sum())},
                      "knee_new_bent_back_frames": int(((sga < -1) & (sgb >= -1)).sum()),
                      "knee_unbent_by_ik_frames": int(((sgb < -1) & (sga >= -1)).sum()),
                      "sole_low_m": {"before": {"p1": _q(fb.sole_low[f], 1, 4), "min": round(float(fb.sole_low[f].min()), 4)},
                                     "after": {"p1": _q(fa.sole_low[f], 1, 4), "min": round(float(fa.sole_low[f].min()), 4)}},
                      "sole_below_source_minus_1mm_frames": int((fa.sole_low[f] < np.minimum(fb.sole_low[f], 0) - 0.001).sum())}
            ch = {}
            for jn, jj in zip(("hip", "knee", "ankle"), LEG[f][:3]):
                q0 = Rot.from_rotvec(np.asarray(fb.z["body_pose"], float).reshape(-1, 21, 3)[:, jj - 1])
                q1 = Rot.from_rotvec(np.asarray(fa.z["body_pose"], float).reshape(-1, 21, 3)[:, jj - 1])
                dd = np.degrees((q0.inv() * q1).magnitude())
                ch[jn] = {"p95_deg": _q(dd, 95, 2), "max_deg": round(float(dd.max()), 2)}
            leg[f]["local_rotation_change"] = ch
        r["legs"] = leg
        if "footlock_weight" in fa.z:                 # the feet NOT being locked: on the floor outside a run they should stay put
            Wl = np.asarray(fa.z["footlock_weight"], float)
            keep = {}
            for q, f in enumerate(FEET):
                m = (Wl[:, q] == 0) & (fb.sole_low[f] < CONTACT_LO)
                dd = 1e3 * np.linalg.norm((fa.J[:, LEG[f][3]] - fb.J[:, LEG[f][3]])[:, [0, 2]], axis=1)[m]
                keep[f] = _stat(dd, 2)
            r["unlocked_floor_foot_ball_shift_mm"] = keep
        dtr = 1e3 * np.linalg.norm(np.asarray(fa.z["transl"], float) - np.asarray(fb.z["transl"], float), axis=1)
        r["transl_change_mm"] = _stat(dtr, 1)
        r["joint_change_mm"] = {"pelvis": _stat(1e3 * np.linalg.norm(fa.J[:, 0] - fb.J[:, 0], axis=1), 1),
                                "wrists": _stat(1e3 * np.linalg.norm(fa.J[:, [20, 21]] - fb.J[:, [20, 21]], axis=-1).ravel(), 1)}
        r["reproj_feet_px_all_frames"] = {"before": reproj_feet(fb, evb), "after": reproj_feet(fa, eva)}
        r["reproj_feet_px_stance"] = {"before": reproj_feet(fb, evb, st), "after": reproj_feet(fa, eva, st)}
        bb, ba = body_reproj(fb.J, role, t), body_reproj(fa.J, role, t)
        dd = ba.pop("_per_frame") - bb.pop("_per_frame")
        r["body_reproj_pct_height"] = {"before": bb, "after": ba, "after_minus_before_frames_over_0.5pct": int(np.nansum(dd > 0.5)),
                                       "note": "jl_fitcheck reproj measure (COCO body joints, % of box height)"}
        jo = np.asarray(FA[role]["joints"], float)[:, :55]
        r["joints_vs_fk_max_mm"] = round(float(1e3 * np.abs(jo - fa.J).max()), 6)
        rep["roles"][role] = r
        log(f"check {role}: slide {r['slip']['slide_mm_per_frame']}, accel_rr {r['accel_rr_mm']['before'].get('median')}->{r['accel_rr_mm']['after'].get('median')}")
    p = out or OUT / "check.json"
    p.write_text(json.dumps(rep, indent=1, default=float))
    return rep


# ================================================================================================ sheet
def _pick_runs(rep_runs: dict, k: int = 3) -> list[tuple[str, str, dict]]:
    """The k stance runs to show: per dancer the largest before-spread run with a ray target, alternating dancers."""
    cands = []
    for role, rr in rep_runs.items():
        for f, lst in rr.items():
            for r in lst:
                if r.get("src") == "ray" and r["b"] - r["a"] + 1 >= 8:
                    cands.append((role, f, r))
    return cands


def sheet(before: Path = BEST, after: Path = DST, n_runs: int = 3, out: Path | None = None, picks: list | None = None) -> Path:
    """sheets/footlock_before_after.jpg: 3 stance runs x 5 frames, crops around the stance foot on the phone's frames:
    before legs RED, after legs GREEN, RTMW own-dancer foot keypoints YELLOW, the pinned target CYAN cross; a top view per
    run of the BALL of the foot (the pinned point): before path red, after green, the video's floor track of the ball (toe
    keypoints' rays, biases removed) yellow, target cyan."""
    import cv2
    cams = load_cams()
    vf = video_feet()
    centre = np.median(cams["C"], axis=0)
    FB, FA = load_fit(before), load_fit(after)
    fj = json.loads((WORK / "frames" / CAM / "frames.json").read_text())
    files = [WORK / "frames" / CAM / f["file"] for f in fj["frames"]]
    info = {}
    for role in ROLES:
        fb, fa = Fit(FB[role]), Fit(FA[role])
        evb = evidence(fb, role, vf, cams)
        runs = stance_runs(fb, evb, centre)
        sb = slip_report(fb, evb, runs, centre)
        info[role] = (fb, fa, evb, runs, sb, keypoint_bias(fb, evb, runs))
    if picks is None:
        # the worst run of each dancer (largest before-spread) and one TYPICAL run (spread nearest the median), all with
        # >= 10 frames and the toes seen in >= 60 % of them
        picks, typical = [], []
        for role in ROLES:
            fb, fa, evb, runs, sb, _ = info[role]
            ok = [r for r in sb["runs"] if r["n"] >= 10 and np.isfinite(evb["feet"][r["foot"]]["rays_fixed"][r["a"]:r["b"] + 1, 1]).all(1).mean() >= 0.6]
            if not ok:
                continue
            rs = sorted(ok, key=lambda r: -r["spread_cm"])
            picks.append((role, rs[0]["foot"], rs[0]["a"], rs[0]["b"]))
            med = float(np.median([r["spread_cm"] for r in sb["runs"]]))
            r_t = min(ok[1:] or ok, key=lambda r: abs(r["spread_cm"] - med))
            typical.append((abs(r_t["spread_cm"] - med), (role, r_t["foot"], r_t["a"], r_t["b"])))
        if typical:
            picks.append(min(typical)[1])
        picks = picks[:n_runs]
    TW, TH = 260, 260
    rows = []
    meta = []
    for role, f, a, b in picks:
        fb, fa, evb, runs, sb, bias = info[role]
        frames = np.unique(np.linspace(a, b, 5).round().astype(int))
        tiles = []
        h, k, an, to = LEG[f]
        tgt = np.asarray(fa.z["footlock_target"], float)[:, FEET.index(f)] if "footlock_target" in fa.z else None
        for i in frames:
            jf = int(evb["frame"][i])
            img = cv2.imread(str(files[jf]))
            Rm, tv, K, k1 = (cams[x][jf:jf + 1].astype(float) for x in ("R", "tvec", "K", "dist"))
            pts = {}
            for key, F_ in (("b", fb), ("a", fa)):
                X = np.concatenate([F_.J[i:i + 1, [h, k, an, to]], F_.vtx[f][i:i + 1]], 1)
                pts[key] = project(X, Rm, tv, K, k1)[0]
            cx, cy = pts["b"][2]
            S = 2.0
            x0, y0 = cx - TW / (2 * S), cy - TH / (2 * S) - 10
            M = np.array([[S, 0, -x0 * S], [0, S, -y0 * S]], float)
            tile = cv2.warpAffine(img, M, (TW, TH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(40, 40, 40))
            P = lambda u: (int(round((u[0] - x0) * S)), int(round((u[1] - y0) * S)))   # noqa: E731
            for key, col in (("b", (0, 0, 255)), ("a", (0, 230, 0))):
                u = pts[key]
                for p_, q_ in ((0, 1), (1, 2), (2, 3)):
                    cv2.line(tile, P(u[p_]), P(u[q_]), col, 2, cv2.LINE_AA)
                for q_ in (2, 3, 4, 6):
                    cv2.circle(tile, P(u[q_]), 3, col, -1, cv2.LINE_AA)
            e = evb["feet"][f]
            for q_ in range(4):
                if e["valid"][i, q_]:
                    cv2.circle(tile, P(e["uv"][i, q_]), 3, (0, 255, 255), -1, cv2.LINE_AA)
            if tgt is not None and np.isfinite(tgt[i]).all():
                u = project(tgt[i][None, None], Rm, tv, K, k1)[0, 0]
                c_ = P(u)
                cv2.drawMarker(tile, c_, (255, 255, 0), cv2.MARKER_CROSS, 14, 2)
            lab = f"{NAME[role]} {f} f{jf} t={evb['t'][i]:.2f}s"
            cv2.putText(tile, lab, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(tile, lab, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(tile)
        while len(tiles) < 5:
            tiles.append(np.zeros((TH, TW, 3), np.uint8))
        # top view (x right, z towards the camera = down)
        top = np.full((TH, TW, 3), 245, np.uint8)
        Pb, Pa = fb.J[a:b + 1, to][:, [0, 2]], fa.J[a:b + 1, to][:, [0, 2]]
        V = Pb + video_offset(fb, evb, f, bias)[a:b + 1]
        allp = np.concatenate([Pb, Pa, V[np.isfinite(V).all(1)]])
        mid = np.median(Pb, axis=0)
        span = max(0.12, float(np.abs(allp - mid).max()) * 2.3)
        sc = TW / span
        Q = lambda p: (int(round(TW / 2 + (p[0] - mid[0]) * sc)), int(round(TH / 2 + (p[1] - mid[1]) * sc)))   # noqa: E731
        cv2.line(top, (10, TH - 14), (10 + int(0.05 * sc), TH - 14), (0, 0, 0), 2)
        cv2.putText(top, "5 cm", (12, TH - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
        cam_dir = (centre[[0, 2]] - mid) / np.linalg.norm(centre[[0, 2]] - mid)
        c0 = (TW - 30, 30)
        cv2.arrowedLine(top, c0, (int(c0[0] + 22 * cam_dir[0]), int(c0[1] + 22 * cam_dir[1])), (120, 120, 120), 1, tipLength=0.3)
        cv2.putText(top, "cam", (TW - 45, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (90, 90, 90), 1, cv2.LINE_AA)
        for p in V:
            if np.isfinite(p).all():
                cv2.circle(top, Q(p), 3, (0, 200, 230), -1, cv2.LINE_AA)
        for Pp, col in ((Pb, (0, 0, 230)), (Pa, (0, 170, 0))):
            for u, v in zip(Pp[:-1], Pp[1:]):
                cv2.line(top, Q(u), Q(v), col, 2, cv2.LINE_AA)
            cv2.circle(top, Q(Pp[0]), 4, col, 1, cv2.LINE_AA)
        if tgt is not None and np.isfinite(tgt[a]).all():
            cv2.drawMarker(top, Q(tgt[a][[0, 2]]), (200, 160, 0), cv2.MARKER_CROSS, 14, 2)
        sp_b, sp_a = path_stats(Pb), path_stats(Pa)
        for kk, line in enumerate((f"top view, ball, {b - a + 1} fr", f"before: path {sp_b['path_cm']:.1f} cm, spread {sp_b['spread_cm']:.1f}",
                                   f"after:  path {sp_a['path_cm']:.1f} cm, spread {sp_a['spread_cm']:.1f}")):
            cv2.putText(top, line, (5, 16 + 15 * kk), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(top)
        rows.append(np.concatenate([np.concatenate([x, np.full((TH, 3, 3), 255, np.uint8)], 1) for x in tiles], 1))
        meta.append({"role": role, "foot": f, "a": int(a), "b": int(b), "t0": round(float(evb["t"][a]), 3), "t1": round(float(evb["t"][b]), 3),
                     "before": sp_b, "after": sp_a})
    W_ = rows[0].shape[1]
    head = np.full((34, W_, 3), 30, np.uint8)
    cv2.putText(head, "foot lock: before RED, after GREEN, RTMW own-dancer foot keypoints YELLOW, pinned ball target CYAN (top view: the video's ball track yellow)",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    img = np.concatenate([head] + [np.concatenate([r, np.full((4, W_, 3), 255, np.uint8)], 0) for r in rows], 0)
    p = out or OUT / "sheets" / "footlock_before_after.jpg"
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    (p.with_suffix(".json")).write_text(json.dumps(meta, indent=1, default=float))
    log(f"sheet -> {p}")
    return p


# ================================================================================================ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.mono4d.jl_footlock")
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--fits", nargs="*", default=["best", "old"], help="keys (best, old) or fit folders")
    m.add_argument("--out", type=Path)
    lk = sub.add_parser("lock")
    lk.add_argument("--src", type=Path, default=BEST)
    lk.add_argument("--dst", type=Path, default=DST)
    lk.add_argument("--gain", type=float, default=PELVIS_GAIN)
    lk.add_argument("--pelvis-hz", type=float, default=PELVIS_HZ)
    lk.add_argument("--pelvis-max", type=float, default=PELVIS_MAX_M)
    ck = sub.add_parser("check")
    ck.add_argument("--before", type=Path, default=BEST)
    ck.add_argument("--after", type=Path, default=DST)
    ck.add_argument("--out", type=Path)
    sh = sub.add_parser("sheet")
    sh.add_argument("--before", type=Path, default=BEST)
    sh.add_argument("--after", type=Path, default=DST)
    sh.add_argument("--out", type=Path)
    sub.add_parser("cache").add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "measure":
        rep = measure(a.fits, a.out)
        print(json.dumps({k: {r: {kk: vv for kk, vv in x.items() if kk != "runs"} for r, x in v["roles"].items()} for k, v in rep["fits"].items()}, indent=1)[:8000])
    elif a.cmd == "lock":
        rep = footlock_fit(a.src, a.dst, a.gain, a.pelvis_hz, a.pelvis_max)
        print(json.dumps({r: {k: v for k, v in x.items() if k != "runs"} for r, x in rep["roles"].items()}, indent=1, default=float))
    elif a.cmd == "check":
        rep = check(a.before, a.after, a.out)
        print(json.dumps(rep, indent=1, default=float)[:10000])
    elif a.cmd == "sheet":
        print(sheet(a.before, a.after, out=a.out))
    elif a.cmd == "cache":
        video_feet(a.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
