"""Two pan / tilt / zoom cameras and two dancers: relative pose, floor, up, metric scale (no room model needed).

Both cameras of the carlos_aline take stand on tripods (fixed centres C_A, C_B) and rotate. dancecap.cameras.ptz gives every frame
a rotation R(t) (reference frame of the camera -> camera) and a focal. A pixel is therefore a RAY in the camera's fixed REFERENCE
frame, d_ref = R(t)^T ray(pixel), whatever the pan, tilt and zoom: the two cameras are a plain stereo pair of two reference frames,
the dancers' joints seen at the same instant are the correspondences. This module solves

  1. relative orientation Q (reference B -> reference A) and the direction of the baseline from the joint correspondences
     (robust: essential matrix RANSAC for the start, then variable-projection bundle adjustment of the pixel reprojection of the
     triangulated joints; the scale of the baseline is free, = 1 here);
  2. the world frame from the dancers: UP and the floor from the ankles of the two dancers (plane fit + the verticals of their
     bodies), the METRIC SCALE from the known statures (anthropometric joint-height ratios of the standing dancers);
  3. uncertainty by resampling the time instants (halves, bootstrap).

Everything here is numpy / scipy on small arrays (no GPU, no files); the I/O glue is in dancecap.cameras.twocam_run.

Conventions: provisional frame = reference frame of camera A (x right, y down, z forward), C_A = 0, |C_B| = 1. The final
world frame is Y up, floor y = 0, metres (CONTRACTS.md); `world_from_prov()` returns (s, R, t) with X_world = s R X_prov + t.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

from . import ptz as P

# COCO-17 order of the YOLO pose model
NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN = range(17)
NAMES = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_sho", "r_sho", "l_elb", "r_elb", "l_wri", "r_wri", "l_hip", "r_hip",
         "l_kne", "r_kne", "l_ank", "r_ank"]
# joint weights in the geometric fit: the torso and legs move least between the instants of the two cameras
JOINT_W = np.array([0.8, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 0.5, 0.5, 0.3, 0.3, 1.0, 1.0, 0.9, 0.9, 1.0, 1.0])

# rigid segments of the body (COCO joint pairs): their 3D length is constant in time, whatever the pose; a wrong focal or a wrong
# baseline direction stretches depth and makes them breathe with the dancer's position, so they pin the geometry that the
# reprojection of a thin slab of people leaves loose (the depth / baseline / focal ambiguity of two near-parallel cameras)
BONES = [(L_SH, L_EL), (L_EL, L_WR), (R_SH, R_EL), (R_EL, R_WR), (L_HIP, L_KN), (L_KN, L_AN), (R_HIP, R_KN), (R_KN, R_AN),
         (L_SH, R_SH), (L_HIP, R_HIP), (L_SH, L_HIP), (R_SH, R_HIP)]
BONE_SIGMA_REL = 0.05       # one bone's length scatter that is NOT geometry (keypoint noise on a 25-45 cm segment), relative

# anthropometric heights above the floor of a standing person, as a fraction of the stature (Drillis and Contini 1966, the
# common table: eye 0.936, shoulder 0.818, hip joint 0.530, knee 0.285, ankle 0.039; COCO's "nose" is the nose tip, a little
# below the eyes). Spread of each ratio between people ~ +-2-3 %.
STAND_RATIO = {NOSE: 0.925, L_EYE: 0.936, R_EYE: 0.936, L_SH: 0.818, R_SH: 0.818, L_HIP: 0.530, R_HIP: 0.530,
               L_KN: 0.285, R_KN: 0.285, L_AN: 0.039, R_AN: 0.039}


# ------------------------------------------------------------------------------------------------ rays and triangulation


@dataclass
class View:
    """Everything needed to turn the pixels of one camera at a set of instants into reference-frame rays."""
    R: np.ndarray            # (T,3,3) reference -> camera, one per instant
    f: np.ndarray            # (T,) focal px
    lens: P.Lens

    def ray_ref(self, t_idx: np.ndarray, pix: np.ndarray) -> np.ndarray:
        """Unit rays in the reference frame of the camera for pixels (M,2) at instants t_idx (M,)."""
        d = P.pix_to_ray(pix, self.f[t_idx], self.lens)
        return np.einsum("mba,mb->ma", self.R[t_idx], d)

    def project_ref(self, t_idx: np.ndarray, d_ref: np.ndarray) -> np.ndarray:
        """Pixels of reference-frame directions (M,3) at instants t_idx; NaN behind the camera."""
        dc = np.einsum("mab,mb->ma", self.R[t_idx], d_ref)
        ok = dc[:, 2] > 1e-6
        out = np.full((len(d_ref), 2), np.nan)
        out[ok] = P.ray_to_pix(dc[ok], self.f[t_idx][ok], self.lens)
        return out


def triangulate(u1: np.ndarray, u2: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Midpoint triangulation of rays from C_A = 0 (direction u1, (M,3)) and C_B = b (direction u2). Returns the points
    (M,3), the closest-approach distance (M,) and a validity mask (both ranges positive)."""
    w = np.einsum("ma,ma->m", u1, u2)
    bu1 = u1 @ b
    bu2 = u2 @ b
    den = 1.0 - w * w
    den = np.where(np.abs(den) < 1e-9, 1e-9, den)
    l1 = (bu1 - w * bu2) / den           # range along u1 from C_A
    l2 = (w * bu1 - bu2) / den           # range along u2 from C_B (the point is b + l2 u2)
    p1 = l1[:, None] * u1
    p2 = b[None] + l2[:, None] * u2
    return 0.5 * (p1 + p2), np.linalg.norm(p1 - p2, axis=1), (l1 > 0) & (l2 > 0)


@dataclass
class Rig:
    """Relative pose of the two reference frames: u_prov = Q u_refB, C_A = 0, C_B = baseline (unit or metric)."""
    Q: np.ndarray            # (3,3) reference B -> reference A (provisional frame)
    b: np.ndarray            # (3,) camera B centre in the provisional frame
    scale: float = 1.0       # provisional units -> metres (set by the stature stage)


def baseline_from_angles(a: np.ndarray) -> np.ndarray:
    th, ph = a
    return np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])


def angles_from_baseline(b: np.ndarray) -> np.ndarray:
    b = b / np.linalg.norm(b)
    return np.array([np.arccos(np.clip(b[2], -1, 1)), np.arctan2(b[1], b[0])])


@dataclass
class Corr:
    """Correspondences: the same joint of the same person at the same instant in both cameras."""
    t: np.ndarray            # (M,) index of the instant (row of the two Views)
    person: np.ndarray       # (M,) 0 Carlos / 1 Aline
    joint: np.ndarray        # (M,) COCO joint
    pa: np.ndarray           # (M,2) pixels in camera A
    pb: np.ndarray           # (M,2) pixels in camera B
    w: np.ndarray            # (M,) weight (joint weight x confidence)

    def sel(self, m) -> "Corr":
        return Corr(self.t[m], self.person[m], self.joint[m], self.pa[m], self.pb[m], self.w[m])

    def __len__(self):
        return len(self.t)


def essential_start(ua: np.ndarray, ub: np.ndarray, thresh_rad: float = 2e-3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Q, baseline direction (unit) and the inlier mask from reference-frame rays by essential matrix RANSAC
    (normalised coordinates, cv2.recoverPose; the cheirality picks the sign of the baseline)."""
    xa = ua[:, :2] / ua[:, 2:3]
    xb = ub[:, :2] / ub[:, 2:3]
    E, m = cv2.findEssentialMat(xb, xa, np.eye(3), cv2.USAC_MAGSAC, 0.9999, thresh_rad, 20000)
    m = m.ravel().astype(bool)
    _, R, t, _ = cv2.recoverPose(E, xb[m], xa[m], np.eye(3))
    # cv2: X_a = R X_b + t  => Q = R maps B-reference rays to A-reference rays and C_B (X_b = 0) sits at t in the A frame
    return R, (t.ravel() / np.linalg.norm(t)), m


def bone_index(corr: "Corr") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rows (ia, ib) of the Corr arrays that hold the two ends of one bone of one person at one instant, and the bone id
    (index into BONES x person: bone * 2 + person)."""
    key = {}
    for i, (t, p, j) in enumerate(zip(corr.t, corr.person, corr.joint)):
        key[(int(t), int(p), int(j))] = i
    ia, ib, bid = [], [], []
    for (t, p, j), i in key.items():
        for k, (a, b) in enumerate(BONES):
            if j == a and (t, p, b) in key:
                ia.append(i); ib.append(key[(t, p, b)]); bid.append(2 * k + p)
    return np.array(ia, int), np.array(ib, int), np.array(bid, int)


def bone_residuals(X: np.ndarray, ok: np.ndarray, bi: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    """Relative deviation of every bone's 3D length from the median length of that bone (per person), divided by BONE_SIGMA_REL."""
    ia, ib, bid = bi
    if len(ia) == 0:
        return np.zeros(0)
    L = np.linalg.norm(X[ia] - X[ib], axis=1)
    good = ok[ia] & ok[ib]
    res = np.zeros(len(L))
    for b in np.unique(bid):
        m = (bid == b) & good
        if m.sum() < 8:
            continue
        med = np.median(L[m])
        res[m] = (L[m] - med) / (BONE_SIGMA_REL * med)
    return res


def rig_residuals(theta: np.ndarray, corr: Corr, va: View, vb: View, ua: np.ndarray, ub_ref: np.ndarray,
                  return_x: bool = False):
    """Pixel reprojection residuals (M,4) of the triangulated joints for rig parameters theta = rotvec(Q) + 2 baseline angles."""
    Q = Rot.from_rotvec(theta[:3]).as_matrix()
    b = baseline_from_angles(theta[3:5])
    ub = ub_ref @ Q.T
    X, gap, ok = triangulate(ua, ub, b)
    ra = va.project_ref(corr.t, X) - corr.pa
    # point seen from B: direction in the B reference frame = Q^T (X - b)
    rb = vb.project_ref(corr.t, (X - b[None]) @ Q) - corr.pb
    r = np.concatenate([ra, rb], 1)
    r = np.where(np.isfinite(r), r, 1e3)
    r[~ok] = 1e3
    if return_x:
        return r, X, ok
    return r


def unit_from_angles(a: np.ndarray) -> np.ndarray:
    return baseline_from_angles(a)


def solve_rig(corr: Corr, va: View, vb: View, theta0: np.ndarray | None = None, loss_px: float = 3.0,
              max_nfev: int = 80, bone_weight: float = 0.0, bone_px: float = 3.0, gravity: dict | None = None) -> tuple[Rig, dict]:
    """Relative orientation + baseline direction. bone_weight > 0 adds the rigid-bone terms (bone_residuals) in units of
    `bone_px` pixels per sigma.

    gravity = {"vA": up in reference frame A (unit), "vB": up in reference frame B (unit), "sigma_v_rad", "sigma_h" (prov
    units), "px_per_sigma"} adds the floor terms: an unknown UP direction u (2 angles) and floor offset d, the two architecture
    verticals as measurements (u ~ vA, u ~ Q vB), and the ankles' height u.X - d ~ 0 (robust: lifted feet are outliers)."""
    ua = va.ray_ref(corr.t, corr.pa)
    ub = vb.ray_ref(corr.t, corr.pb)
    info = {}
    bi = bone_index(corr) if bone_weight > 0 else None
    ank = np.isin(corr.joint, (L_AN, R_AN))
    if theta0 is None:
        Q, bdir, inl = essential_start(ua, ub)
        theta0 = np.concatenate([Rot.from_matrix(Q).as_rotvec(), angles_from_baseline(bdir)])
        info["essential_inliers"] = int(inl.sum())
    n0 = len(theta0)
    use_g = gravity is not None
    if use_g:
        vA = np.asarray(gravity["vA"], float) / np.linalg.norm(gravity["vA"])
        vB = np.asarray(gravity["vB"], float) / np.linalg.norm(gravity["vB"])
        sv, sh, pps = float(gravity["sigma_v_rad"]), float(gravity["sigma_h"]), float(gravity.get("px_per_sigma", 3.0))
        if len(theta0) == 5:
            r0, X0, ok0 = rig_residuals(theta0, corr, va, vb, ua, ub, return_x=True)
            u0 = vA
            d0 = float(np.median(X0[ank & ok0] @ u0))
            theta0 = np.concatenate([theta0, angles_from_baseline(u0), [d0]])

    def fun(th):
        if use_g:
            core, u, d = th[:5], unit_from_angles(th[5:7]), th[7]
        else:
            core = th
        r, X, ok = rig_residuals(core, corr, va, vb, ua, ub, return_x=True)
        parts = [(r * corr.w[:, None]).ravel()]
        if bi is not None:
            parts.append(bone_weight * bone_px * bone_residuals(X, ok, bi))
        if use_g:
            Q = Rot.from_rotvec(core[:3]).as_matrix()
            parts.append(pps * (u - vA) / sv)
            parts.append(pps * (u - Q @ vB) / sv)
            h = (X[ank] @ u - d) / sh
            h = np.where(ok[ank], h, 0.0)
            parts.append(pps * corr.w[ank] * h)
        return np.concatenate(parts)

    xs = np.array([0.01] * 3 + [0.01] * 2 + ([0.01, 0.01, 0.01] if use_g else []))
    sol = least_squares(fun, theta0, loss="soft_l1", f_scale=loss_px, x_scale=xs, max_nfev=max_nfev)
    core = sol.x[:5]
    r, X, okx = rig_residuals(core, corr, va, vb, ua, ub, return_x=True)
    e = np.linalg.norm(r.reshape(-1, 2, 2), axis=2)           # (M,2) per camera
    info.update({"theta": sol.x.tolist(), "median_px": float(np.median(e)), "p90_px": float(np.percentile(e, 90)),
                 "rms_px": float(np.sqrt((e ** 2).mean())), "n": int(len(corr)), "cost": float(sol.cost)})
    if use_g:
        u, d = unit_from_angles(sol.x[5:7]), float(sol.x[7])
        Q = Rot.from_rotvec(core[:3]).as_matrix()
        h = X[ank & okx] @ u - d
        info["gravity"] = {"u": u.tolist(), "d": d, "vA_vs_u_deg": float(np.degrees(np.arccos(np.clip(u @ vA, -1, 1)))),
                           "QvB_vs_u_deg": float(np.degrees(np.arccos(np.clip(u @ (Q @ vB), -1, 1)))),
                           "ankle_h_median": float(np.median(h)), "ankle_h_mad": float(1.4826 * np.median(np.abs(h - np.median(h))))}
    bi2 = bi if bi is not None else bone_index(corr)
    br = bone_residuals(X, okx, bi2)
    nzb = br[br != 0]
    info["bone_rel_sd"] = float(np.std(nzb) * BONE_SIGMA_REL) if len(nzb) else None
    info["bone_rel_mad"] = float(1.4826 * np.median(np.abs(nzb)) * BONE_SIGMA_REL) if len(nzb) else None
    return Rig(Rot.from_rotvec(core[:3]).as_matrix(), baseline_from_angles(core[3:5])), info


def triangulate_all(rig: Rig, corr: Corr, va: View, vb: View) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Joints in the provisional frame (M,3), reprojection error per correspondence (M,) (max of the two views) and validity."""
    ua = va.ray_ref(corr.t, corr.pa)
    ub = vb.ray_ref(corr.t, corr.pb)
    X, _, ok = triangulate(ua, ub @ rig.Q.T, rig.b)
    ra = va.project_ref(corr.t, X) - corr.pa
    rb = vb.project_ref(corr.t, (X - rig.b[None]) @ rig.Q) - corr.pb
    e = np.maximum(np.linalg.norm(ra, axis=1), np.linalg.norm(rb, axis=1))
    return X, np.where(np.isfinite(e), e, 1e6), ok


# ------------------------------------------------------------------------------------------------ world frame


def fit_plane(X: np.ndarray, w: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """Unit normal n and offset d of the weighted least-squares plane n.x = d."""
    w = np.ones(len(X)) if w is None else w
    c = (X * w[:, None]).sum(0) / w.sum()
    U, S, Vt = np.linalg.svd((X - c) * np.sqrt(w)[:, None], full_matrices=False)
    n = Vt[-1]
    return n, float(n @ c)


def ransac_plane(X: np.ndarray, thresh: float, iters: int = 2000, seed: int = 0, prior_n: np.ndarray | None = None,
                 prior_deg: float = 25.0) -> tuple[np.ndarray, float, np.ndarray]:
    """RANSAC plane through points X (M,3), optionally only normals within prior_deg of prior_n."""
    rng = np.random.default_rng(seed)
    best, bm = -1, None
    for _ in range(iters):
        i = rng.choice(len(X), 3, replace=False)
        n = np.cross(X[i[1]] - X[i[0]], X[i[2]] - X[i[0]])
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        if prior_n is not None and np.degrees(np.arccos(abs(n @ prior_n))) > prior_deg:
            continue
        d = n @ X[i[0]]
        m = np.abs(X @ n - d) < thresh
        if m.sum() > best:
            best, bm = int(m.sum()), m
    n, d = fit_plane(X[bm])
    for _ in range(3):
        m = np.abs(X @ n - d) < thresh
        n, d = fit_plane(X[m])
    return n, d, np.abs(X @ n - d) < thresh


def rotation_to_up(n: np.ndarray, forward_hint: np.ndarray) -> np.ndarray:
    """Rotation R (rows = world axes expressed in prov) such that world Y = n (up), world X horizontal = forward_hint x Y."""
    y = n / np.linalg.norm(n)
    x = np.cross(y, forward_hint)
    x /= np.linalg.norm(x)
    z = np.cross(x, y)
    return np.stack([x, y, z])      # world = R prov


def stand_scale(heights: np.ndarray, joints: np.ndarray, stature_m: float, up_ok: np.ndarray) -> dict:
    """Metric scale from standing frames: for each joint with an anthropometric ratio, expected height = ratio * stature;
    returns the median scale (metres per provisional unit) per joint and overall (median of joint medians)."""
    out = {}
    for j, r in STAND_RATIO.items():
        m = (joints == j) & up_ok & (heights > 0)
        if m.sum() < 20:
            continue
        out[j] = (float(np.median(r * stature_m / heights[m])), int(m.sum()))
    return out
