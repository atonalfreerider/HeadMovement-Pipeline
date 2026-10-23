"""Per-frame pose of a camera that WALKS (Steadicam, gimbal, handheld phone carried around the couple) from the dancers alone.

The static-room route (cameras.pipeline: SfM + localisation) needs a room model; cameras.mono_cam needs a fixed centre. A Steadicam that
circles the couple has neither: it is attached to the world of ANOTHER camera's body fit instead. That fit (smplx*/<role>.npz of
`fit_take`, both dancers, SMPL-X joints per 30 Hz instant in metres, Y up, floor y = 0) gives 3D points at every instant; the walking
camera's own 2D keypoints of the same two dancers (people/teachers_01.npz of `people_take`: RTMW COCO body, lead / follow rows) are
their images. Per frame that is a PnP problem with up to 24 correspondences (12 body joints x 2 dancers); over the whole video it is
one robust least-squares problem:

  focal      ONE focal for the video (the owner's camera type: no zoom), the principal point at the centre, no distortion: a grid
             search on ~240 frames spread in time (per-frame EPnP + LM at each focal, score = the median over frames of the median
             reprojection error), refined by a parabola through the best three;
  clock      the people take's video time + dt = the fit take's time; dt from the audio clocks (given) refined by the same score
             over a +-0.2 s grid (a 50 ms error is 10 cm at 2 m/s: the dancers fix it);
  per frame  RANSAC PnP (EPnP start, LM refine on the inliers) with the focal fixed: R, C, inliers, residual;
  joint      one sparse least-squares solve over EVERY frame: the reprojection of every correspondence (soft-L1 at SIGMA_PX) +
             the second difference of the camera centre (SIGMA_ACC: a walking operator's centre accelerates slowly) and of the
             rotation (the angular velocity changes slowly, SIGMA_ANG): frames with few keypoints (one dancer out of the
             vertical frame, an occlusion) are carried by their neighbours instead of jumping.

Body-joint definitions differ between COCO and SMPL-X (hips most), and a monocular fit's depth along ITS camera's axis is the
weakest coordinate: both show up as residuals here, in percent of the dancer's height in the picture (the jl_fitcheck measure).
This attach is therefore as good as the fit it hangs on; it is the start of a two-view refit, not a substitute for one.

    python -m dancecap.cameras.attach_moving solve --fit-take lucas_luiza_h --people-take lucas_luiza_v --out-take lucas_luiza --cam 01 \\
           [--src smplx_rf2_noslip] [--dt 0.008] [--focal-range 500 2600]
    python -m dancecap.cameras.attach_moving review --out-take lucas_luiza --cam 01 --fit-take lucas_luiza_h --people-take lucas_luiza_v \\
           --video <path> [--times 10 40 ...]

Outputs (work/<out_take>/cameras/, git-ignored): <cam>.npz in the cameras/<cam>.npz schema of CONTRACTS.md (t = the fit take's clock;
frame_index / pts = the people take's video frames; attached = True; dancer_px = the per-frame median dancer residual), <cam>_attach.json
(focal, dt, residuals per second, camera path facts), review/attach_<cam>.jpg (the fit's skeletons through the solved camera).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from ..take import PACKAGE_ROOT

ROLES = ("lead", "follow")
# COCO-17 body keypoint -> SMPL-X joint (dancecap.jl_fitcheck.COCO_TO_SMPLX: shoulders, elbows, wrists, hips, knees, ankles)
COCO_TO_SMPLX = {5: 16, 6: 17, 7: 18, 8: 19, 9: 20, 10: 21, 11: 1, 12: 2, 13: 4, 14: 5, 15: 7, 16: 8}
COCO_IDS = tuple(COCO_TO_SMPLX)
SMPLX_IDS = tuple(COCO_TO_SMPLX.values())
KP_CONF = 0.5          # keypoint confidence (RTMW, rescaled to 0..1 by people_mono)
ROLE_CONF = 0.4        # the row's identity / pose confidence
MIN_POINTS = 8         # a frame needs this many correspondences for its own PnP
SIGMA_PX = 6.0         # reprojection scale of the soft-L1 loss (px at the native size)
SIGMA_ACC = 1.5        # m/s^2: the camera centre's acceleration prior
SIGMA_ANG = 4.0        # rad/s^2: the camera's angular acceleration prior
FIT_MAX_DT = 0.05      # a video frame takes the fit at its time only when a fit instant lies within this (s)


# ------------------------------------------------------------------------------------------------ pure geometry


def project(X: np.ndarray, R: np.ndarray, tvec: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pinhole projection of world points X (..., 3) with x_cam = R X + tvec -> (pixels (..., 2), depth (...))."""
    Xc = X @ R.T + tvec
    z = Xc[..., 2]
    zz = np.where(z > 1e-6, z, 1e-6)
    return np.stack([K[0, 0] * Xc[..., 0] / zz + K[0, 2], K[1, 1] * Xc[..., 1] / zz + K[1, 2]], -1), z


def intrinsics(f: float, size: tuple[int, int]) -> np.ndarray:
    w, h = size
    return np.array([[f, 0.0, (w - 1) / 2.0], [0.0, f, (h - 1) / 2.0], [0.0, 0.0, 1.0]])


def interp_joints(t_fit: np.ndarray, J: np.ndarray, t: np.ndarray, max_dt: float = FIT_MAX_DT) -> tuple[np.ndarray, np.ndarray]:
    """Fit joints (T, n, 3) at times t by linear interpolation between the two neighbouring fit instants -> ((len(t), n, 3), valid).
    Invalid where no fit instant lies within max_dt, or a neighbour is not finite."""
    t_fit = np.asarray(t_fit, float)
    i = np.clip(np.searchsorted(t_fit, t), 1, len(t_fit) - 1)
    t0, t1 = t_fit[i - 1], t_fit[i]
    a = np.clip((t - t0) / np.maximum(t1 - t0, 1e-9), 0.0, 1.0)
    out = (1 - a)[:, None, None] * J[i - 1] + a[:, None, None] * J[i]
    near = np.minimum(np.abs(t - t0), np.abs(t - t1)) <= max_dt
    valid = near & (t >= t_fit[0] - max_dt) & (t <= t_fit[-1] + max_dt) & np.isfinite(out).all(axis=(1, 2))
    return out, valid


def correspondences(t_video: np.ndarray, kp: np.ndarray, conf: np.ndarray, skel_in_partner: np.ndarray | None,
                    t_fit: np.ndarray, joints: dict[str, np.ndarray], dt: float) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Per video frame (object points (n, 3), image points (n, 2), role of each point (n,)): both dancers' COCO body keypoints with
    their fit joints at the frame's time + dt. kp (F, 2, 17, 3) in pixels + confidence, conf (F, 2), joints[role] (T, 55+, 3)."""
    F = len(t_video)
    tt = np.asarray(t_video, float) + dt
    per_role = []
    for ri, role in enumerate(ROLES):
        Jr, ok = interp_joints(t_fit, np.asarray(joints[role], float)[:, list(SMPLX_IDS)], tt)
        per_role.append((Jr, ok))
    out = []
    for f in range(F):
        obj, img, rol = [], [], []
        for ri in range(2):
            Jr, ok = per_role[ri]
            if not ok[f] or not np.isfinite(conf[f, ri]) or conf[f, ri] < ROLE_CONF:
                continue
            if skel_in_partner is not None and np.isfinite(skel_in_partner[f, ri]) and skel_in_partner[f, ri] > 0.5:
                continue
            k = kp[f, ri][list(COCO_IDS)]
            m = np.isfinite(k).all(-1) & (k[:, 2] >= KP_CONF)
            if m.any():
                obj.append(Jr[f][m])
                img.append(k[m, :2])
                rol.append(np.full(int(m.sum()), ri))
        if obj:
            out.append((np.concatenate(obj), np.concatenate(img), np.concatenate(rol)))
        else:
            out.append((np.zeros((0, 3)), np.zeros((0, 2)), np.zeros(0, int)))
    return out


def pnp(obj: np.ndarray, img: np.ndarray, K: np.ndarray, thr_px: float, guess: tuple[np.ndarray, np.ndarray] | None = None
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | None:
    """RANSAC PnP (EPnP hypotheses) + LM refinement on the inliers -> (rvec, tvec, inlier mask, median inlier error px) or None."""
    import cv2
    if len(obj) < 6:
        return None
    o = np.ascontiguousarray(obj, np.float64).reshape(-1, 1, 3)
    p = np.ascontiguousarray(img, np.float64).reshape(-1, 1, 2)
    try:
        ok, rvec, tvec, inl = cv2.solvePnPRansac(o, p, K, None, iterationsCount=150, reprojectionError=float(thr_px), confidence=0.995,
                                                 flags=cv2.SOLVEPNP_EPNP)
    except cv2.error:
        return None
    if not ok or inl is None or len(inl) < 6:
        return None
    m = np.zeros(len(obj), bool)
    m[inl.ravel()] = True
    try:
        rvec, tvec = cv2.solvePnPRefineLM(o[m], p[m], K, None, rvec, tvec)
    except cv2.error:
        pass
    R = cv2.Rodrigues(rvec)[0]
    uv, z = project(obj, R, tvec.ravel(), K)
    if (z[m] <= 0).any():
        return None
    e = np.linalg.norm(uv - img, axis=-1)
    m = e <= thr_px
    if m.sum() < 6:
        return None
    return rvec.ravel(), tvec.ravel(), m, float(np.median(e[m]))


def frame_score(obs, K: np.ndarray, thr_px: float) -> np.ndarray:
    """Median reprojection error of each observed frame's own PnP (NaN = no solution)."""
    out = np.full(len(obs), np.nan)
    for i, (o, p, _) in enumerate(obs):
        r = pnp(o, p, K, thr_px)
        if r is not None:
            out[i] = r[3] if r[2].mean() >= 0.6 else np.nan
    return out


def sample_frames(obs, n: int, min_points: int = 14) -> np.ndarray:
    """Up to n frame indices with at least min_points correspondences, spread evenly in time."""
    good = np.array([i for i, (o, _, _) in enumerate(obs) if len(o) >= min_points])
    if len(good) <= n:
        return good
    return good[np.linspace(0, len(good) - 1, n).round().astype(int)]


def parabola_min(x: np.ndarray, y: np.ndarray) -> float:
    """The minimum of a parabola through the best grid point and its two neighbours (the grid point at an edge)."""
    i = int(np.nanargmin(y))
    if i == 0 or i == len(x) - 1 or not np.isfinite(y[i - 1:i + 2]).all():
        return float(x[i])
    a, b, _ = np.polyfit(x[i - 1:i + 2], y[i - 1:i + 2], 2)
    return float(x[i]) if a <= 0 else float(np.clip(-b / (2 * a), x[i - 1], x[i + 1]))


def search_focal(obs, size: tuple[int, int], f_range: tuple[float, float], thr_px: float, n_frames: int = 240) -> dict:
    idx = sample_frames(obs, n_frames)
    if len(idx) < 20:
        raise SystemExit(f"[attach_moving] only {len(idx)} frames with 14+ correspondences: cannot solve the focal")
    sub = [obs[i] for i in idx]
    grid = np.exp(np.linspace(math.log(f_range[0]), math.log(f_range[1]), 28))
    score = np.array([np.nanmedian(frame_score(sub, intrinsics(f, size), thr_px)) for f in grid])
    f0 = parabola_min(np.log(grid), score)
    fine = np.exp(np.linspace(f0 - 0.08, f0 + 0.08, 9))
    s2 = np.array([np.nanmedian(frame_score(sub, intrinsics(f, size), thr_px)) for f in fine])
    f_best = float(np.exp(parabola_min(np.log(fine), s2)))
    return {"focal_px": f_best, "frames": len(idx), "grid": [[round(float(g), 1), round(float(s), 2)] for g, s in zip(grid, score)],
            "fine": [[round(float(g), 1), round(float(s), 2)] for g, s in zip(fine, s2)]}


def search_dt(build_obs, K: np.ndarray, dt0: float, thr_px: float, half: float = 0.2, step: float = 0.02, n_frames: int = 200) -> dict:
    """The clock offset that minimises the median per-frame PnP error (build_obs(dt) -> correspondences)."""
    grid = np.round(np.arange(dt0 - half, dt0 + half + 1e-9, step), 4)
    idx = None
    score = []
    for dt in grid:
        obs = build_obs(float(dt))
        if idx is None:
            idx = sample_frames(obs, n_frames)
        score.append(float(np.nanmedian(frame_score([obs[i] for i in idx], K, thr_px))))
    score = np.array(score)
    return {"dt": parabola_min(grid, score), "grid": [[float(g), round(float(s), 3)] for g, s in zip(grid, score)]}


# ------------------------------------------------------------------------------------------------ the joint solve


def sparsity(fid: np.ndarray, F: int):
    """The Jacobian pattern of joint_solve: parameters [d (F, 3) rotation corrections, c (F, 3) centre corrections]; rows = 2 per
    correspondence (its frame's 6), then 3 per centre second difference (one axis of 3 frames), then 3 per rotation second
    difference (the 9 rotation numbers of 3 frames)."""
    from scipy.sparse import coo_matrix
    M = len(fid)
    n_c = 3 * max(0, F - 2)
    r_p = np.repeat(np.arange(2 * M), 6)
    c_p = np.concatenate([3 * fid[:, None] + np.arange(3), 3 * F + 3 * fid[:, None] + np.arange(3)], axis=1)
    c_p = np.repeat(c_p, 2, axis=0).ravel()
    i = np.arange(max(0, F - 2))
    k = np.arange(3)
    r_c = (2 * M + 3 * i[:, None, None] + k[None, :, None] + 0 * np.arange(3)[None, None, :]).ravel()
    c_c = (3 * F + 3 * (i[:, None, None] + np.arange(3)[None, None, :]) + k[None, :, None]).ravel()
    r_r = (2 * M + n_c + 3 * i[:, None, None] + k[None, :, None] + 0 * np.arange(9)[None, None, :]).ravel()
    c_r = (3 * i[:, None, None] + np.arange(9)[None, None, :] + 0 * k[None, :, None]).ravel()
    rows = np.concatenate([r_p, r_c, r_r])
    cols = np.concatenate([c_p, c_c, c_r])
    return coo_matrix((np.ones(len(rows), np.int8), (rows, cols)), shape=(2 * M + 2 * n_c, 6 * F)).tocsr()


def _rot(rvecs: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    return Rotation.from_rotvec(rvecs).as_matrix()


def joint_solve(obs, K: np.ndarray, R0: np.ndarray, C0: np.ndarray, fps: float, sigma_px: float = SIGMA_PX, sigma_acc: float = SIGMA_ACC,
                sigma_ang: float = SIGMA_ANG, max_nfev: int = 60) -> tuple[np.ndarray, np.ndarray, dict]:
    """Every frame's pose at once: R_i = exp(d_i) R0_i, C_i = C0_i + c_i (12 numbers -> 6 per frame). Residuals: reprojection of every
    correspondence (/sigma_px, soft-L1), centre second differences (/(sigma_acc dt^2)), relative-rotation second differences
    (/(sigma_ang dt^2)). Sparse Jacobian (each frame touches its own points and three smoothness rows). -> (R, C, info)."""
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    F = len(obs)
    dt = 1.0 / fps
    n_obs = np.array([len(o) for o, _, _ in obs])
    fid = np.repeat(np.arange(F), n_obs)
    O = np.concatenate([o for o, _, _ in obs]) if n_obs.sum() else np.zeros((0, 3))
    P = np.concatenate([p for _, p, _ in obs]) if n_obs.sum() else np.zeros((0, 2))
    M = len(O)
    s_c = sigma_acc * dt * dt
    s_r = sigma_ang * dt * dt

    def unpack(x):
        d = x[:3 * F].reshape(F, 3)
        c = x[3 * F:].reshape(F, 3)
        R = np.einsum("fij,fjk->fik", _rot(d), R0)
        return R, C0 + c

    def resid(x):
        R, C = unpack(x)
        Xc = np.einsum("mij,mj->mi", R[fid], O - C[fid])
        z = np.maximum(Xc[:, 2], 1e-3)
        uv = np.stack([K[0, 0] * Xc[:, 0] / z + K[0, 2], K[1, 1] * Xc[:, 1] / z + K[1, 2]], -1)
        r_proj = ((uv - P) / sigma_px).ravel()
        r_c = ((C[2:] - 2 * C[1:-1] + C[:-2]) / s_c).ravel()
        # relative rotations between neighbours (camera frame increments), then their change
        rel = Rotation.from_matrix(np.einsum("fij,fkj->fik", R[1:], R[:-1])).as_rotvec()
        r_r = ((rel[1:] - rel[:-1]) / s_r).ravel()
        return np.concatenate([r_proj, r_c, r_r])

    A = sparsity(fid, F)
    x0 = np.zeros(6 * F)
    t0 = time.time()
    res = least_squares(resid, x0, jac_sparsity=A, loss="soft_l1", f_scale=1.0, x_scale="jac", max_nfev=max_nfev,
                        method="trf", tr_solver="lsmr", verbose=0)
    R, C = unpack(res.x)
    return R, C, {"cost": float(res.cost), "nfev": int(res.nfev), "status": int(res.status), "seconds": round(time.time() - t0, 1),
                  "points": int(M), "frames": int(F)}


def robust_smooth(t: np.ndarray, R: np.ndarray, C: np.ndarray, w: np.ndarray, sigma_s: float = 0.25, rounds: int = 3,
                  k_mad: float = 4.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A walking camera's track from noisy per-frame poses: Gaussian-weighted (sigma_s) means of the centres and of the rotations
    (quaternions flipped into one hemisphere, mono4d.cand_fitb.smooth_rotvec), each frame weighted by w (its inliers); frames whose
    own pose is more than k_mad robust deviations from the smoothed track get weight 0, `rounds` times. -> (R, C, kept)."""
    from scipy.spatial.transform import Rotation
    from ..mono4d.cand_fitb import _gauss_weights, smooth_rotvec
    dt = float(np.median(np.diff(t)))
    sig = sigma_s / dt
    keep = np.asarray(w, float) > 0
    Cs, Rs = C, R
    for _ in range(rounds):
        W = _gauss_weights(len(t), sig, keep) * np.asarray(w, float)[None]
        W = W / np.maximum(W.sum(1, keepdims=True), 1e-12)
        Cs = W @ np.nan_to_num(C)
        rv = smooth_rotvec(Rotation.from_matrix(R).as_rotvec(), sig, keep)
        Rs = Rotation.from_rotvec(rv).as_matrix()
        dc = np.linalg.norm(C - Cs, axis=-1)
        da = np.degrees((Rotation.from_matrix(Rs).inv() * Rotation.from_matrix(R)).magnitude())
        new = keep.copy()
        for d in (dc, da):
            m = np.median(d[keep])
            mad = np.median(np.abs(d[keep] - m)) * 1.4826 + 1e-9
            new &= d <= m + k_mad * mad
        if new.sum() < 10 or (new == keep).all():
            keep = new if new.sum() >= 10 else keep
            break
        keep = new
    return Rs, Cs, keep


def fill_gaps(t: np.ndarray, R: np.ndarray, C: np.ndarray, have: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Frames without their own PnP: centre linearly, rotation by slerp between the nearest solved frames (held at the ends)."""
    from scipy.spatial.transform import Rotation, Slerp
    idx = np.flatnonzero(have)
    C2 = np.stack([np.interp(t, t[idx], C[idx, k]) for k in range(3)], -1)
    tq = np.clip(t, t[idx[0]], t[idx[-1]])
    R2 = Slerp(t[idx], Rotation.from_matrix(R[idx]))(tq).as_matrix()
    return R2, C2


def per_frame_residual(obs, R: np.ndarray, C: np.ndarray, K: np.ndarray, box_h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(median residual px, median residual in % of the dancer's box height) per frame (NaN = no correspondences)."""
    px = np.full(len(obs), np.nan)
    pct = np.full(len(obs), np.nan)
    for i, (o, p, rol) in enumerate(obs):
        if not len(o):
            continue
        uv, _ = project(o, R[i], -R[i] @ C[i], K)
        e = np.linalg.norm(uv - p, axis=-1)
        px[i] = float(np.median(e))
        h = np.maximum(box_h[i, rol], 50.0)
        pct[i] = float(np.median(100.0 * e / h))
    return px, pct


# ------------------------------------------------------------------------------------------------ I/O


def work(take: str) -> Path:
    return PACKAGE_ROOT / "work" / take


def load_fit(take: str, src: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    d = work(take) / src
    z = {r: np.load(d / f"{r}.npz", allow_pickle=True) for r in ROLES}
    t = np.asarray(z["lead"]["t"], float)
    if not np.allclose(t, np.asarray(z["follow"]["t"], float)):
        raise SystemExit(f"[attach_moving] {d}: lead and follow are on different time grids")
    return t, {r: np.asarray(z[r]["joints"], float) for r in ROLES}


def load_people(take: str) -> dict:
    z = np.load(work(take) / "people" / "teachers_01.npz", allow_pickle=True)
    out = {k: np.asarray(z[k]) for k in ("t", "frame_index", "kp", "conf", "box", "size")}
    out["skel_in_partner"] = np.asarray(z["skel_in_partner"]) if "skel_in_partner" in z.files else None
    return out


def solve(fit_take: str, people_take: str, out_take: str, cam: str, src: str = "smplx_rf2_noslip", dt0: float = 0.0,
          f_range: tuple[float, float] = (500.0, 2600.0), thr_frac: float = 0.03, search_clock: bool = True,
          focal: float | None = None, max_nfev: int = 0, sigma_s: float = 0.25) -> dict:
    t_fit, joints = load_fit(fit_take, src)
    pe = load_people(people_take)
    size = (int(pe["size"][0]), int(pe["size"][1]))
    t_v = np.asarray(pe["t"], float)
    fps = float(1.0 / np.median(np.diff(t_v)))
    thr = thr_frac * max(size)
    print(f"[attach_moving] {people_take}: {len(t_v)} frames {size[0]}x{size[1]} at {fps:.2f} fps; fit {fit_take}/{src}: {len(t_fit)} instants "
          f"{t_fit[0]:.2f}-{t_fit[-1]:.2f} s; RANSAC threshold {thr:.0f} px", flush=True)

    def build(dt):
        return correspondences(t_v, pe["kp"], pe["conf"], pe["skel_in_partner"], t_fit, joints, dt)

    obs = build(dt0)
    foc = search_focal(obs, size, f_range, thr)
    if focal:
        # the dancers alone often cannot fix the focal (lucas_luiza Steadicam: the per-frame score was flat, 19-22 px, from 500 to
        # 2600 px: a couple ~2 m away spans too little depth, focal trades against distance): a given focal (the lens) wins
        foc.update(searched_focal_px=foc["focal_px"], focal_px=float(focal), given=True)
    K = intrinsics(foc["focal_px"], size)
    print(f"[attach_moving] focal {foc['focal_px']:.1f} px ({foc['frames']} frames{', given' if focal else ''})", flush=True)
    clock = {"dt": dt0, "searched": False}
    if search_clock:
        clock = search_dt(build, K, dt0, thr)
        clock["searched"] = True
        if abs(clock["dt"] - dt0) > 1e-6:
            obs = build(clock["dt"])
            if not focal:
                foc2 = search_focal(obs, size, (foc["focal_px"] / 1.25, foc["focal_px"] * 1.25), thr)
                foc["focal_px_after_clock"] = foc2["focal_px"]
                K = intrinsics(foc2["focal_px"], size)
        print(f"[attach_moving] clock dt {clock['dt']:+.3f} s; focal {K[0, 0]:.1f} px", flush=True)
    dt = float(clock["dt"])
    # per-frame PnP (seeded by the previous frame), then gaps filled
    F = len(obs)
    R0 = np.full((F, 3, 3), np.nan)
    C0 = np.full((F, 3), np.nan)
    have = np.zeros(F, bool)
    n_inl = np.zeros(F, np.int32)
    import cv2
    for i, (o, p, _) in enumerate(obs):
        if len(o) < MIN_POINTS:
            continue
        r = pnp(o, p, K, thr)
        if r is None or r[2].sum() < MIN_POINTS:
            continue
        R = cv2.Rodrigues(r[0])[0]
        R0[i], C0[i], have[i], n_inl[i] = R, -R.T @ r[1], True, int(r[2].sum())
    if have.sum() < 10:
        raise SystemExit(f"[attach_moving] only {have.sum()} frames solved on their own")
    R0, C0 = fill_gaps(t_v, R0, C0, have)
    # the joint solve keeps only each frame's PnP inliers' kind of points: drop gross outliers against the initial poses
    obs_j = []
    for i, (o, p, rol) in enumerate(obs):
        if len(o):
            uv, z = project(o, R0[i], -R0[i] @ C0[i], K)
            m = (np.linalg.norm(uv - p, axis=-1) <= 2.5 * thr) & (z > 0)
            obs_j.append((o[m], p[m], rol[m]))
        else:
            obs_j.append((o, p, rol))
    w = np.where(have, n_inl, 0).astype(float)
    R, C, kept = robust_smooth(t_v, R0, C0, w, sigma_s)
    info = {"smoothing": f"robust Gaussian sigma {sigma_s} s (attach_moving.robust_smooth)", "frames_kept": int(kept.sum()),
            "frames_with_pnp": int(have.sum())}
    if max_nfev > 0:
        R, C, js = joint_solve(obs_j, K, R, C, fps, max_nfev=max_nfev)
        info["joint_solve"] = js
    box_h = np.asarray(pe["box"][..., 3] - pe["box"][..., 1], float)
    px, pct = per_frame_residual(obs_j, R, C, K, np.nan_to_num(box_h, nan=50.0))
    n_pts = np.array([len(o) for o, _, _ in obs_j])
    med = float(np.nanmedian(px))
    ok = (n_pts >= 4) & np.isfinite(px) & (px <= max(3.0 * med, 0.04 * max(size)))
    # a frame with no point at all is still a pose (carried by the smoothness) but not "ok" beyond 0.5 s from an ok frame
    t_ref = t_v + dt
    tvec = -np.einsum("fij,fj->fi", R, C)
    out_dir = work(out_take) / "cameras"
    out_dir.mkdir(parents=True, exist_ok=True)
    Kf = np.repeat(K[None], F, 0)
    np.savez_compressed(out_dir / f"{cam}.npz", t=t_ref, pts=t_v, frame_index=np.asarray(pe["frame_index"], np.int32), K=Kf, R=R, tvec=tvec,
                        dist=np.zeros(F), ok=ok, reproj_px=px.astype(np.float32), dancer_px=px.astype(np.float32), dancer_pct=pct.astype(np.float32),
                        n_inliers=n_pts.astype(np.int32), interpolated=~have, focal_px=np.full(F, K[0, 0], np.float32), centre_world=C,
                        pp_px=np.array([K[0, 2], K[1, 2]]), native_size=np.array(size, np.int32), work_size=np.array(size, np.int32),
                        camera_class=np.array("handheld"), type_model=np.array("walking: per-frame 6 dof from the dancers, one focal"),
                        attached=np.array(True), t_clock=np.array(f"{fit_take}'s reference clock = {people_take}'s video time + {dt:+.4f} s"))
    if not (out_dir / "world_from_sfm.json").exists():
        (out_dir / "world_from_sfm.json").write_text(json.dumps({
            "s": 1.0, "R": np.eye(3).tolist(), "t": [0.0, 0.0, 0.0],
            "note": f"no SfM model: the cameras are in the world frame of {fit_take} (dancecap.cameras.attach_moving / mono_cam)"}, indent=1))
    # report
    sec = np.floor(t_ref).astype(int)
    per_sec = []
    for s in range(int(np.nanmin(sec)), int(np.nanmax(sec)) + 1):
        m = (sec == s) & np.isfinite(pct)
        per_sec.append([s, round(float(np.median(pct[m])), 2) if m.sum() >= 5 else None, int(ok[sec == s].sum())])
    speed = np.linalg.norm(np.diff(C, axis=0), axis=-1) * fps
    couple = np.nanmean(np.stack([interp_joints(t_fit, joints[r][:, [0]], t_ref)[0][:, 0] for r in ROLES]), axis=0)
    dist = np.linalg.norm((C - couple)[:, [0, 2]], axis=-1)
    rep = {"fit_take": fit_take, "src": src, "people_take": people_take, "out_take": out_take, "cam": cam, "size": size, "fps": round(fps, 3),
           "focal": {k: v for k, v in foc.items() if k not in ("grid", "fine")}, "focal_grid": foc["grid"], "clock": clock,
           "frames": F, "own_pnp": int(have.sum()), "ok": int(ok.sum()), "solve": info,
           "residual_px": {"median": round(med, 2), "p90": round(float(np.nanpercentile(px, 90)), 2)},
           "residual_pct_height": {"median": round(float(np.nanmedian(pct)), 2), "p90": round(float(np.nanpercentile(pct, 90)), 2)},
           "camera": {"height_m": {"median": round(float(np.median(C[:, 1])), 3), "p5": round(float(np.percentile(C[:, 1], 5)), 3),
                                   "p95": round(float(np.percentile(C[:, 1], 95)), 3)},
                      "distance_to_couple_m": {"median": round(float(np.nanmedian(dist)), 2), "p5": round(float(np.nanpercentile(dist, 5)), 2),
                                               "p95": round(float(np.nanpercentile(dist, 95)), 2)},
                      "speed_m_s": {"median": round(float(np.median(speed)), 3), "p95": round(float(np.percentile(speed, 95)), 3)},
                      "path_length_m": round(float(np.sum(np.linalg.norm(np.diff(C, axis=0), axis=-1))), 1)},
           "per_second": {"columns": ["s", "median_pct_of_height", "ok_frames"], "rows": per_sec},
           "note": "residuals compare the fit's SMPL-X joints with RTMW COCO keypoints (definitions differ by 2-3 % of the height, hips most)"}
    (out_dir / f"{cam}_attach.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ("focal", "clock", "own_pnp", "ok", "residual_px", "residual_pct_height", "camera", "solve")},
                     indent=1, default=str)[:3000], flush=True)
    return rep


def review(out_take: str, cam: str, fit_take: str, people_take: str, video: Path, times: list[float] | None = None,
           src: str = "smplx_rf2_noslip", width: int = 360, cols: int = 4) -> Path:
    """Contact sheet: the fit's skeletons (lead green, follow red) projected through the solved camera on the walking camera's frames."""
    import cv2
    from ..jl_fitcheck import BONES
    z = np.load(work(out_take) / "cameras" / f"{cam}.npz", allow_pickle=True)
    t_fit, joints = load_fit(fit_take, src)
    t_ref, pts = np.asarray(z["t"], float), np.asarray(z["pts"], float)
    if not times:
        times = list(np.linspace(t_ref[0] + 2, t_ref[-1] - 2, 12))
    tiles = []
    cap = cv2.VideoCapture(str(video))
    for tr in times:
        i = int(np.argmin(np.abs(t_ref - tr)))
        cap.set(cv2.CAP_PROP_POS_MSEC, float(pts[i]) * 1000.0)
        ok, img = cap.read()
        if not ok:
            continue
        for role, col in zip(ROLES, ((60, 220, 60), (60, 60, 230))):
            J, v = interp_joints(t_fit, joints[role][:, :22], np.array([t_ref[i]]))
            if not v[0]:
                continue
            uv, d = project(J[0], z["R"][i], z["tvec"][i], z["K"][i])
            for a, b in BONES:
                if d[a] > 0 and d[b] > 0:
                    cv2.line(img, tuple(int(x) for x in uv[a]), tuple(int(x) for x in uv[b]), col, 3, cv2.LINE_AA)
        cv2.putText(img, f"{t_ref[i]:.1f}s {'ok' if z['ok'][i] else 'NOT ok'} {float(z['dancer_px'][i]):.0f}px", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        h = int(round(img.shape[0] * width / img.shape[1]))
        tiles.append(cv2.resize(img, (width, h)))
    cap.release()
    if not tiles:
        raise SystemExit("[attach_moving] no frame decoded")
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(len(tiles) // cols)])
    out = work(out_take) / "review" / f"attach_{cam}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.cameras.attach_moving", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("solve")
    s.add_argument("--fit-take", required=True)
    s.add_argument("--people-take", required=True)
    s.add_argument("--out-take", required=True)
    s.add_argument("--cam", required=True)
    s.add_argument("--src", default="smplx_rf2_noslip")
    s.add_argument("--dt", type=float, default=0.0, help="people take video time + dt = fit take time (s), the start of the search")
    s.add_argument("--no-clock-search", action="store_true")
    s.add_argument("--focal-range", type=float, nargs=2, default=(500.0, 2600.0))
    s.add_argument("--focal", type=float, help="the focal in px (the lens), when the dancers cannot fix it (a flat focal_grid)")
    s.add_argument("--max-nfev", type=int, default=0, help="> 0: the joint least-squares refinement after the smoothing (slow)")
    s.add_argument("--sigma", type=float, default=0.25, help="the track smoothing (s)")
    r = sub.add_parser("review")
    r.add_argument("--out-take", required=True)
    r.add_argument("--cam", required=True)
    r.add_argument("--fit-take", required=True)
    r.add_argument("--people-take", required=True)
    r.add_argument("--video", type=Path, required=True)
    r.add_argument("--src", default="smplx_rf2_noslip")
    r.add_argument("--times", type=float, nargs="*")
    a = ap.parse_args(argv)
    if a.cmd == "solve":
        solve(a.fit_take, a.people_take, a.out_take, a.cam, a.src, a.dt, tuple(a.focal_range), search_clock=not a.no_clock_search,
              focal=a.focal, max_nfev=a.max_nfev, sigma_s=a.sigma)
    else:
        print(review(a.out_take, a.cam, a.fit_take, a.people_take, a.video, a.times, a.src))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
