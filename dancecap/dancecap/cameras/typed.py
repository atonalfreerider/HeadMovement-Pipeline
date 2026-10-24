"""Camera-type constrained camera tracks (dancecap.camera_types -> the cameras stage).

The user's camera type per video (mount handheld | tripod, zoom yes | no; takes/<take>.toml [cameras."NN"]) removes
unknowns. stage_smooth applies it to the raw localisation (cameras/cache/loc_<cam>.npz: per-frame pose, focal, k1 and
each frame's inlier 2D-3D correspondences) BEFORE the light smoothing, in place of the zoom-depth guard (a fixed focal
or a fixed centre removes the focal / depth-along-the-axis ambiguity the guard exists for):

  class           model applied                                     when the residual demands it (fallback)
  handheld_zoom   per-frame pose + focal: TODAY, code path unchanged  -
  handheld        per-frame 6-dof pose, ONE focal + k1 per video     per-frame focal (today) + check failed
                  (joint robust fit over <= 40 frames spread in time)
  tripod          ONE pose (R, C) + ONE focal + k1 for the whole     shared centre + per-frame rotation (the head
                  video (joint robust fit), every frame held          pans), then the handheld model + check failed
  tripod_zoom     shared centre, per-frame rotation + focal           per-frame pose + focal (today) + check failed

Solver: batched robust Gauss-Newton / Levenberg-Marquardt (analytic Jacobians of the SIMPLE_RADIAL projection,
Huber IRLS at huber_px, Schur complement for the shared parameters). Each parameter block (rotation, centre, log focal,
k1) is per-frame, shared by the video, or held. The JOINT fits (one focal + k1, one constant pose, one shared centre)
use <= max_pts correspondences per frame over <= 40-80 frames spread in time; the FINAL per-frame solves (fixed focal:
per-frame pose; fixed centre: per-frame rotation [+ focal]) use EVERY correspondence of each frame, in chunks of frames
(a 300-point subset made real tracks 2-3 x shakier than today's all-point solution).

Residual test ("solve once robustly, keep per-frame only if the residual demands it"): a constrained model is accepted
when, over the frames' own static correspondences, the median over frames of the per-frame median reprojection error
is <= max(ratio x, + add_px over) today's per-frame solution's, and its p90 over frames <= max(max_px, ratio x
today's p90). Static points far away cannot see what matters for the closer dancers, so more checks guard the
constrained models: a fixed CENTRE needs today's per-frame centres not to drift (0.25 s low-pass vs its noise and the
point depth: every larissa_kadu phone declared 'tripod' passed the reprojection test with a fixed centre, 1.16-1.39 px,
and was stopped here), a fixed FOCAL needs today's per-frame focal to stay within the zoom bound (a zooming phone
declared 'no zoom' slides along its axis instead: dolly-zoom), and no per-frame model may make the track shakier than
today's (centre and rotation jitter <= jitter_ratio x today's + a small floor). A refused model is reported in
report.json cams.<cam>.camera_type (check.passed false + the reason, model = what was applied instead); the camera-type
view (dancecap.camera_types.describe: model_check) and the monitor show it next to the user's answer, and probe B
(written by the same qa step) opens the camera_types gate when its measurement contradicts the answer. strict=True
keeps the user's model regardless.

LOCKED tripods (tripod, no zoom, no pans) are also LOCALISED on keyframes only (localize_typed: one frame every
tripod_step_s and its next frame + the map's own frames, ~10 % of the per-frame matching); when the constant pose
fails the residual test there, every frame is localised. The keyframe drift check takes the per-frame noise (from the
pairs) out of the keyframes' spread, so it decides like the every-frame check. loc_<cam>.npz records how the camera
was localised (loc_mode keyframes | all, loc_class, loc_policy, loc_localised: loc_marker); a keyframes-only cache is
only valid for the constant pose: any other model (type or policy changed, or the constant pose refused) needs every
frame localised again, and stage_smooth refuses to export it (KeyframeCacheError).
Attached cameras (own sub-model + attach) keep the attach model, except tripods: one constant pose.

    python -m dancecap.cameras.typed <take> --cams 01,05 --as tripod,handheld   compare models, no files written
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass

import numpy as np

MODELS = ("per_frame", "fixed_focal", "constant", "fixed_centre")
BLOCKS = (("rot", 3), ("cen", 3), ("lf", 1), ("k1", 1))


@dataclass
class TypedSettings:
    enabled: bool = True
    strict: bool = False          # keep the user's model even when the residual test fails
    max_pts: int = 300            # correspondences per frame in the JOINT (shared-parameter) fits and the coarse pass
    eval_pts: int = 1500          # correspondences per frame when measuring a model
    # the FINAL per-frame solves (fixed focal: per-frame pose; fixed centre: per-frame rotation [+ focal]) use every
    # correspondence of each frame (final_pts 0 = all): a 300-point subset made larissa_kadu 07's centre track 2.2 x
    # shakier (0.96 -> 2.14 mm) than today's per-frame solution, which uses all of them. Frames are independent there,
    # so they are solved in chunks of <= chunk_pts correspondences (memory), coarse (max_pts) then fine (final_iters).
    final_pts: int = 0
    final_iters: int = 5  # from the coarse solution it converges in ~4 (larissa_kadu 03: |dC| 1e-6 vs 10)
    chunk_pts: int = 200_000
    intr_frames: int = 40         # frames in the joint focal + k1 fit
    const_frames: int = 80        # frames in the joint constant-pose / shared-centre fit
    ratio: float = 1.5            # residual test: median and p90 over frames ...
    add_px: float = 1.0           # ... within max(ratio x, + add_px) of the per-frame solution's
    max_px: float = 4.0           # p90 bound (stage_smooth's max_reproj_px)
    huber_px: float = 2.0
    iters: int = 25
    tripod_step_s: float = 2.0    # keyframe localisation of tripods
    min_frames: int = 5
    # translation guard for the fixed-centre models: far static points barely see a few cm of translation, the
    # dancers (closer) do. The per-frame centres' low-pass (0.25 s) may drift at most max(drift_k x its own noise,
    # drift_rel x the median static-point depth): locked (one constant pose) / pans (shared centre). Measured on
    # larissa_kadu's handheld phones (56-62 s): drift 0.0013-0.026 x depth (9-213 mm) at 16-355 x its noise (0.1-2.7
    # mm), so 0.0015 / 0.002 x depth (~12 / 16 mm at 8 m) separates still phones from tripods; a phone panned on a
    # tripod head off its optical centre also translates (a few cm) and is then (truthfully) kept per-frame + flagged.
    # Uncalibrated on real tripods: no tripod capture measured yet.
    drift_k: float = 4.0
    drift_rel_locked: float = 0.0015
    drift_rel_pans: float = 0.002
    # consequence checks of a FIXED focal (larissa_kadu cam09, a zooming phone declared 'no zoom': the static points
    # still fit at 1.85 px because the solver slides the camera along its axis to mimic the zoom - dolly-zoom - and
    # the centre jitter went 17 -> 412 mm): today's per-frame focal may deviate from the fixed one by at most
    # zoom_ratio (p90; probe B's zoom bound), and the high-frequency jitter (0.25 s) of the centre track may grow to
    # at most jitter_ratio x today's + jitter_add_rel x the point depth (~1 mm at 8 m), of the rotation track to at
    # most jitter_ratio x today's + rot_jitter_add_deg. Mostly RELATIVE: today's phones jitter ~1 mm, so the former
    # absolute floor (0.0005 x depth, ~4 mm) let a 2-3 x shakier track through.
    zoom_ratio: float = 1.05
    jitter_ratio: float = 1.25
    jitter_add_rel: float = 0.000125
    rot_jitter_add_deg: float = 0.003


# ------------------------------------------------------------------------------------------------ geometry

def so3_exp(w: np.ndarray) -> np.ndarray:
    """Rodrigues, vectorised: (n, 3) rotation vectors -> (n, 3, 3)."""
    w = np.atleast_2d(np.asarray(w, float))
    th = np.linalg.norm(w, axis=1)
    k = np.where(th[:, None] > 1e-12, w / np.maximum(th, 1e-12)[:, None], 0.0)
    K = np.zeros((len(w), 3, 3))
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 0] = -k[:, 2], k[:, 1], k[:, 2]
    K[:, 1, 2], K[:, 2, 0], K[:, 2, 1] = -k[:, 0], -k[:, 1], k[:, 0]
    s, c = np.sin(th)[:, None, None], (1 - np.cos(th))[:, None, None]
    return np.eye(3)[None] + s * K + c * (K @ K)


def project_batch(X, R, C, f, k1, cx, cy) -> tuple[np.ndarray, np.ndarray]:
    """Per-point projection (OpenCV pixel centres, one radial term). R (N,3,3), C (N,3), f, k1 (N,) -> (uv, z)."""
    Xc = np.einsum("nij,nj->ni", R, X - C)
    z = Xc[:, 2]
    zz = np.where(np.abs(z) > 1e-9, z, 1e-9)
    x, y = Xc[:, 0] / zz, Xc[:, 1] / zz
    d = 1 + k1 * (x * x + y * y)
    return np.stack([f * x * d + cx, f * y * d + cy], 1), z


def _residual_jac(X, xo, R, C, f, k1, cx, cy) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Residuals (N,2), Jacobian (N,2,8) w.r.t. [left rotation (3), centre (3), log focal, k1], valid (N,) (in front)."""
    D = X - C
    Xc = np.einsum("nij,nj->ni", R, D)
    z = Xc[:, 2]
    valid = z > 1e-6
    zz = np.where(valid, z, 1.0)
    x, y = Xc[:, 0] / zz, Xc[:, 1] / zz
    r2 = x * x + y * y
    dd = 1 + k1 * r2
    u, v = f * x * dd + cx, f * y * dd + cy
    res = np.stack([u, v], 1) - xo
    # d(u, v) / d(x, y)
    a11, a12 = f * (dd + 2 * k1 * x * x), f * 2 * k1 * x * y
    a21, a22 = a12, f * (dd + 2 * k1 * y * y)
    # d(x, y) / dXc
    inv = 1.0 / zz
    B = np.zeros((len(X), 2, 3))
    B[:, 0, 0], B[:, 0, 2] = inv, -x * inv
    B[:, 1, 1], B[:, 1, 2] = inv, -y * inv
    A = np.stack([np.stack([a11, a12], 1), np.stack([a21, a22], 1)], 1)  # (N,2,2)
    Jx = A @ B  # d(u,v)/dXc (N,2,3)
    S = np.zeros((len(X), 3, 3))  # -skew(Xc): Xc <- Exp(w) Xc  ~  Xc + w x Xc
    S[:, 0, 1], S[:, 0, 2], S[:, 1, 0] = Xc[:, 2], -Xc[:, 1], -Xc[:, 2]
    S[:, 1, 2], S[:, 2, 0], S[:, 2, 1] = Xc[:, 0], Xc[:, 1], -Xc[:, 0]
    J = np.empty((len(X), 2, 8))
    J[:, :, 0:3] = Jx @ S
    J[:, :, 3:6] = -(Jx @ R)
    J[:, 0, 6], J[:, 1, 6] = f * x * dd, f * y * dd
    J[:, 0, 7], J[:, 1, 7] = f * x * r2, f * y * r2
    return res, J, valid


@dataclass
class Batch:
    """Correspondences of several frames, sorted by frame: X (N,3), x (N,2), fid (N,) in 0..n-1, starts (n,)."""
    frames: np.ndarray
    X: np.ndarray
    x: np.ndarray
    fid: np.ndarray
    starts: np.ndarray

    @classmethod
    def of(cls, d: dict, frames, k: int, seed: int = 0) -> "Batch":
        XX, xx, ff = [], [], []
        for j, i in enumerate(frames):
            X, x = _corr(d, int(i), k, seed)
            XX.append(X), xx.append(x), ff.append(np.full(len(X), j))
        n = np.array([len(a) for a in XX])
        return cls(np.asarray(frames, int), np.concatenate(XX), np.concatenate(xx), np.concatenate(ff),
                   np.r_[0, np.cumsum(n)[:-1]].astype(int))


def _huber(e: np.ndarray, k: float) -> tuple[np.ndarray, np.ndarray]:
    rho = np.where(e <= k, 0.5 * e * e, k * (e - 0.5 * k))
    w = np.where(e <= k, 1.0, k / np.maximum(e, 1e-12))
    return rho, w


def gn_solve(b: Batch, R, C, lf, k1, cx, cy, roles: dict, huber: float = 2.0, iters: int = 25) -> dict:
    """Robust LM over a batch. R (n,3,3), C (n,3), lf, k1 (n,) per frame; roles {rot, cen, lf, k1: 'P' (per frame) |
    'S' (shared by all frames; must start equal) | 'F' (held)}. Returns the refined states + cost info."""
    R, C, lf, k1 = R.copy(), C.copy(), lf.astype(float).copy(), k1.astype(float).copy()
    n = len(b.frames)
    cols = {"P": [], "S": []}
    c0 = 0
    for name, k in BLOCKS:
        if roles.get(name, "F") in cols:
            cols[roles[name]] += list(range(c0, c0 + k))
        c0 += k
    P, S = np.array(cols["P"], int), np.array(cols["S"], int)
    npf, ns = len(P), len(S)
    fid = b.fid

    def cost_of(R_, C_, lf_, k1_):
        r, _ = project_batch(b.X, R_[fid], C_[fid], np.exp(lf_[fid]), k1_[fid], cx, cy)
        e = np.linalg.norm(r - b.x, axis=1)
        rho, _ = _huber(e, huber)
        return np.bincount(fid, rho, minlength=n)

    cost = cost_of(R, C, lf, k1)
    c_start = float(cost.sum())
    lam = np.full(n, 1e-3)
    lam_s = 1e-3
    for _ in range(iters):
        res, J, valid = _residual_jac(b.X, b.x, R[fid], C[fid], np.exp(lf[fid]), k1[fid], cx, cy)
        e = np.linalg.norm(res, axis=1)
        _, w = _huber(e, huber)
        w = np.where(valid, w, 0.0)
        Jw = J * w[:, None, None]
        if npf:
            Jp, Jpw = J[:, :, P], Jw[:, :, P]
            Hpp = np.add.reduceat(np.einsum("nri,nrj->nij", Jpw, Jp), b.starts, axis=0)
            gp = np.add.reduceat(np.einsum("nri,nr->ni", Jpw, res), b.starts, axis=0)
            dg = np.einsum("nii->ni", Hpp)
            Hd = Hpp + (lam[:, None] * np.maximum(dg, 1e-9))[:, :, None] * np.eye(npf)[None]
            Hinv = np.linalg.inv(Hd)
        if ns:
            Js, Jsw = J[:, :, S], Jw[:, :, S]
            Hss = np.einsum("nri,nrj->ij", Jsw, Js)
            gs = np.einsum("nri,nr->i", Jsw, res)
            Hssd = Hss + lam_s * np.diag(np.maximum(np.diag(Hss), 1e-9))
            if npf:
                Hsp = np.add.reduceat(np.einsum("nri,nrj->nij", Jsw, Jp), b.starts, axis=0)  # (n, ns, npf)
                M = np.einsum("nij,njk->nik", Hsp, Hinv)
                Sm = Hssd - np.einsum("nij,nkj->ik", M, Hsp)
                rhs = -gs + np.einsum("nij,nj->i", M, gp)
            else:
                Sm, rhs = Hssd, -gs
            ds = np.linalg.solve(Sm, rhs)
        else:
            ds = np.zeros(0)
        if npf:
            rhs_p = -gp - (np.einsum("nij,i->nj", Hsp, ds) if ns else 0.0)
            dp = np.einsum("nij,nj->ni", Hinv, rhs_p)
        else:
            dp = np.zeros((n, 0))
        full = np.zeros((n, 8))
        if npf:
            full[:, P] = dp
        if ns:
            full[:, S] = ds[None]
        R2 = so3_exp(full[:, 0:3]) @ R
        C2, lf2, k12 = C + full[:, 3:6], lf + full[:, 6], k1 + full[:, 7]
        new = cost_of(R2, C2, lf2, k12)
        if ns:  # shared parameters couple the frames: accept the whole step or none
            if new.sum() < cost.sum():
                R, C, lf, k1, rel = R2, C2, lf2, k12, (cost.sum() - new.sum()) / max(cost.sum(), 1e-12)
                cost, lam_s, lam = new, max(lam_s / 3, 1e-7), np.maximum(lam / 3, 1e-7)
                if rel < 1e-7:
                    break
            else:
                lam_s, lam = lam_s * 10, lam * 10
                if lam_s > 1e8:
                    break
        else:  # independent frames: accept per frame
            better = new < cost
            R[better], C[better], lf[better], k1[better] = R2[better], C2[better], lf2[better], k12[better]
            rel = np.where(better, (cost - new) / np.maximum(cost, 1e-12), 0.0)
            cost = np.where(better, new, cost)
            lam = np.where(better, np.maximum(lam / 3, 1e-7), lam * 10)
            if not better.any() or (rel.max() < 1e-7) or (lam.min() > 1e8):
                break
    return {"R": R, "C": C, "lf": lf, "k1": k1, "cost_before": round(c_start, 2), "cost_after": round(float(cost.sum()), 2)}


def gn_solve_frames(d: dict, frames, k_pts: int, R, C, lf, k1, cx, cy, roles: dict, huber: float = 2.0,
                    iters: int = 25, chunk_pts: int = 200_000) -> dict:
    """gn_solve for roles WITHOUT shared blocks (every frame is its own problem), over chunks of frames holding at
    most chunk_pts correspondences (k_pts per frame, 0 = every correspondence): the final per-frame solves use all
    of a frame's points (as today's localiser does) without building one batch of millions of points."""
    if "S" in roles.values():
        raise ValueError("gn_solve_frames: shared parameters couple the frames (use gn_solve)")
    frames = np.asarray(frames, int)
    k = int(k_pts) if k_pts and k_pts > 0 else 1 << 30
    npts = np.minimum(np.diff(np.asarray(d["corr_off"], np.int64))[frames], k)
    R, C = np.array(R, float, copy=True), np.array(C, float, copy=True)
    lf, k1 = np.array(lf, float, copy=True), np.array(k1, float, copy=True)
    cb = ca = 0.0
    i = 0
    while i < len(frames):
        j, tot = i + 1, int(npts[i])
        while j < len(frames) and tot + int(npts[j]) <= chunk_pts:
            tot += int(npts[j])
            j += 1
        b = Batch.of(d, frames[i:j], k)
        r = gn_solve(b, R[i:j], C[i:j], lf[i:j], k1[i:j], cx, cy, roles, huber, iters)
        R[i:j], C[i:j], lf[i:j], k1[i:j] = r["R"], r["C"], r["lf"], r["k1"]
        cb, ca = cb + r["cost_before"], ca + r["cost_after"]
        i = j
    return {"R": R, "C": C, "lf": lf, "k1": k1, "cost_before": round(cb, 2), "cost_after": round(ca, 2)}


# ------------------------------------------------------------------------------------------------ helpers

def _cc(w: int, h: int) -> tuple[float, float]:
    return w / 2 - 0.5, h / 2 - 0.5  # OpenCV pixel-centre principal point (cameras.report.project_cv)


def loc_mode(d) -> str:
    """'keyframes' when a raw localisation (loc_<cam>.npz dict / NpzFile) holds the locked-tripod keyframes only,
    'all' otherwise (caches written before the marker existed localised every frame)."""
    try:
        v = d["loc_mode"] if "loc_mode" in (d.files if hasattr(d, "files") else d) else None
    except (KeyError, TypeError):
        v = None
    return "keyframes" if v is not None and str(np.asarray(v)) == "keyframes" else "all"


def _corr_frames(d: dict, min_pts: int = 6) -> np.ndarray:
    """Localised frames with >= min_pts static correspondences and a usable focal: one frame of lucas_luiza's Steadicam (1 of
    5248) came back from P4Pf with f <= 0, and its log turned the joint model's start focal into NaN, so the one-focal model of
    the handheld camera was rejected (median 1e6 px) and the per-frame focal kept."""
    off = np.asarray(d["corr_off"])
    n = np.diff(off)
    f = np.asarray(d["f"], float)
    return np.flatnonzero(np.asarray(d["ok"], bool) & (n >= min_pts) & np.isfinite(f) & (f > 0))


def _corr(d: dict, i: int, k: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A frame's correspondences, at most k (evenly strided: deterministic and cheap)."""
    a, b = int(d["corr_off"][i]), int(d["corr_off"][i + 1])
    n = b - a
    sel = np.arange(n) if n <= k else np.round(np.linspace(0, n - 1, k)).astype(int)
    return d["corr_xyz"][a:b][sel].astype(float), d["corr_xy"][a:b][sel].astype(float)


def _centres(d: dict) -> np.ndarray:
    return -np.einsum("fji,fj->fi", np.asarray(d["R"], float), np.asarray(d["tvec"], float))


def _spread(idx: np.ndarray, k: int) -> np.ndarray:
    idx = np.asarray(idx, int)
    return idx if len(idx) <= k else idx[np.unique(np.round(np.linspace(0, len(idx) - 1, k)).astype(int))]


def _mean_rotation(Rs: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Weighted chordal L2 mean of rotation matrices (SVD projection of the weighted sum)."""
    w = np.ones(len(Rs)) if w is None else np.asarray(w, float)
    U, _, Vt = np.linalg.svd(np.einsum("n,nij->ij", w, Rs))
    return U @ np.diag([1.0, 1.0, float(np.sign(np.linalg.det(U @ Vt)) or 1.0)]) @ Vt


def _medians(b: Batch, R, C, f, k1, cx, cy) -> np.ndarray:
    """Per-frame median reprojection error (px) of a batch under per-frame states."""
    uv, z = project_batch(b.X, R[b.fid], C[b.fid], f[b.fid], k1[b.fid], cx, cy)
    e = np.where(z > 1e-6, np.linalg.norm(uv - b.x, axis=1), 1e6)
    ends = np.r_[b.starts[1:], len(e)]
    return np.array([np.median(e[a:z_]) for a, z_ in zip(b.starts, ends)])


def residual_test(e_new: np.ndarray, e_old: np.ndarray, s: TypedSettings) -> dict:
    e_new, e_old = np.asarray(e_new, float), np.asarray(e_old, float)
    ok = np.isfinite(e_new) & np.isfinite(e_old)
    if ok.sum() < 1:
        return {"passed": False, "frames": 0, "why": "no frames with correspondences"}
    mn, mo = float(np.median(e_new[ok])), float(np.median(e_old[ok]))
    pn, po = float(np.percentile(e_new[ok], 90)), float(np.percentile(e_old[ok], 90))
    lim_m, lim_p = max(s.ratio * mo, mo + s.add_px), max(s.max_px, s.ratio * po)
    passed = mn <= lim_m and pn <= lim_p
    out = {"passed": bool(passed), "frames": int(ok.sum()), "median_px": round(mn, 3), "p90_px": round(pn, 3),
           "per_frame_median_px": round(mo, 3), "per_frame_p90_px": round(po, 3),
           "limit_median_px": round(lim_m, 3), "limit_p90_px": round(lim_p, 3)}
    if not passed:
        out["why"] = (f"static reprojection {mn:.2f} / {pn:.2f} px (median / p90) vs today's {mo:.2f} / {po:.2f} px: "
                      f"the model does not explain the room points")
    return out


# ------------------------------------------------------------------------------------------------ models

class _Cam:
    """One camera's raw localisation, prepared for the models."""

    def __init__(self, d: dict, w: int, h: int, s: TypedSettings):
        self.d, self.s = d, s
        self.cx, self.cy = _cc(w, h)
        self.frames = _corr_frames(d)
        self.sparse = loc_mode(d) == "keyframes"  # localised on keyframes (+ their next frame) only
        self.R = np.asarray(d["R"], float)
        self.C = _centres(d)
        self.f = np.asarray(d["f"], float)
        self.k1 = np.asarray(d["k1"], float)
        self.ev = Batch.of(d, self.frames, s.eval_pts) if len(self.frames) else None
        fr = self.frames
        self.e_old = _medians(self.ev, self.R[fr], self.C[fr], self.f[fr], self.k1[fr], self.cx, self.cy) \
            if self.ev is not None else np.zeros(0)
        w_in = np.maximum(np.asarray(d["n_inliers"], float)[fr], 1.0) if len(fr) else np.ones(0)
        self.f0 = float(np.exp(np.average(np.log(self.f[fr]), weights=w_in))) if len(fr) else float("nan")
        self.k10 = float(np.median(self.k1[fr])) if len(fr) else 0.0
        self.D = 1.0
        if self.ev is not None:
            ends = np.r_[self.ev.starts[1:], len(self.ev.X)]
            dep = [float(np.median((self.ev.X[self.ev.starts[j]:ends[j]] - self.C[fr[j]]) @ self.R[fr[j]][2]))
                   for j in _spread(np.arange(len(fr)), 30)]
            D = float(np.median(dep)) if dep else 1.0
            self.D = D if np.isfinite(D) and D > 0 else 1.0

    def drift(self, lateral: bool = False) -> dict:
        """Low-pass (0.25 s) drift of today's per-frame centres vs its own noise, relative to the point depth.
        lateral=True: only perpendicular to the mean optical axis (a zooming camera's per-frame centres slide along
        the axis with the focal by construction)."""
        from .smooth import robust_smooth
        fr = self.frames
        t = np.asarray(self.d["t"], float)[fr]
        C = self.C[fr]
        if lateral:
            a = self.R[fr][:, 2, :].mean(0)
            a /= np.linalg.norm(a)
            C = C - np.outer(C @ a, a)
        if self.sparse:
            return self._drift_sparse(t, C, lateral)
        w = np.clip(np.asarray(self.d["n_inliers"], float)[fr] / 100.0, 0.2, 1.0)
        Cs, _ = robust_smooth(t, C, w, 0.25)
        sig = 1.4826 * float(np.median(np.linalg.norm(C - Cs, axis=1)))
        fps = 1.0 / max(float(np.median(np.diff(t))), 1e-3) if len(t) > 1 else 30.0
        noise = sig / np.sqrt(max(1.0, 2.5 * 0.25 * fps))
        drift = float(np.percentile(np.linalg.norm(Cs - np.median(Cs, 0), axis=1), 95))
        return {"drift_rel_depth": round(drift / self.D, 5), "noise_rel_depth": round(noise / self.D, 6),
                "drift_over_noise": round(drift / max(noise, 1e-12), 2), "median_point_depth": round(self.D, 3),
                "lateral_only": lateral}

    def _drift_sparse(self, t: np.ndarray, C: np.ndarray, lateral: bool) -> dict:
        """drift() on a keyframe localisation (samples ~2 s apart: no 0.25 s neighbourhood to low-pass over). Each
        keyframe is localised with its next frame (keyframe_subset(pairs=True)); the pair differences (33 ms: no real
        motion on a tripod) give the per-frame centre noise COVARIANCE (P4Pf noise lies mostly along the optical
        axis), whose p95 contribution is taken out of the keyframe means' spread in quadrature, and the noise is
        reported as the every-frame check's 0.25 s low-pass would see it - so both localisations decide alike.
        Without pairs (older caches) the noise is 0: stricter (falls back to every frame)."""
        dt = np.diff(t)
        pair = dt < 0.1
        grp = np.r_[0, np.cumsum(~pair)]  # runs of consecutive frames
        M = np.array([C[grp == g].mean(0) for g in np.unique(grp)])
        n_g = np.bincount(grp)
        raw = float(np.percentile(np.linalg.norm(M - np.median(M, 0), axis=1), 95)) if len(M) else 0.0
        noise = noise_in = 0.0
        if pair.sum() >= 3:
            dC = C[1:][pair] - C[:-1][pair]
            nrm = np.linalg.norm(dC, axis=1)
            dC = dC[nrm <= 5 * max(float(np.median(nrm)), 1e-12)]  # a failed frame is not noise
            cov = (dC.T @ dC) / max(1, len(dC)) / 2.0  # per-frame centre noise covariance
            w, U = np.linalg.eigh(cov)
            Lc = U * np.sqrt(np.clip(w, 0.0, None))
            Z = np.random.default_rng(0).standard_normal((4000, 3)) @ Lc.T  # per-frame noise samples (fixed seed)
            zn = np.linalg.norm(Z, axis=1)
            sig = 1.4826 * float(np.median(zn))  # what the every-frame check's 1.4826 x median |C - low-pass| reads
            fps = 1.0 / max(float(np.median(dt[pair])), 1e-3)
            noise = sig / np.sqrt(max(1.0, 2.5 * 0.25 * fps))
            noise_in = float(np.percentile(zn, 95)) / np.sqrt(max(1.0, float(np.median(n_g))))
        drift = float(np.sqrt(max(0.0, raw * raw - noise_in * noise_in)))
        return {"drift_rel_depth": round(drift / self.D, 5), "noise_rel_depth": round(noise / self.D, 6),
                "drift_over_noise": round(drift / max(noise, 1e-12), 2), "median_point_depth": round(self.D, 3),
                "lateral_only": lateral, "keyframes": int(len(M)), "raw_spread_rel_depth": round(raw / self.D, 5),
                "noise_spread_rel_depth": round(noise_in / self.D, 5), "pairs": int(pair.sum())}

    def drift_ok(self, dr: dict, rel: float) -> bool:
        return dr["drift_rel_depth"] <= max(self.s.drift_k * dr["noise_rel_depth"], rel)

    def hf_jitter(self, C: np.ndarray) -> float:
        """High-frequency centre jitter (RMS residual after a 0.25 s robust smooth) of per-frame centres over
        self.frames, relative to the median point depth."""
        from .smooth import robust_smooth
        t = np.asarray(self.d["t"], float)[self.frames]
        if len(t) < 5:
            return 0.0
        Cs, _ = robust_smooth(t, C, np.ones(len(t)), 0.25)
        return float(np.sqrt(np.mean(np.sum((C - Cs) ** 2, 1)))) / self.D

    def hf_rot(self, R: np.ndarray) -> float:
        """High-frequency rotation jitter (deg RMS after a 0.25 s robust smooth of the quaternions) over self.frames."""
        from .smooth import quat_from_R, robust_smooth
        t = np.asarray(self.d["t"], float)[self.frames]
        if len(t) < 5:
            return 0.0
        q = quat_from_R(np.asarray(R, float))
        qs, _ = robust_smooth(t, q, np.ones(len(t)), 0.25)
        qs /= np.linalg.norm(qs, axis=1, keepdims=True)
        ang = np.degrees(2 * np.arccos(np.clip(np.abs((q * qs).sum(1)), -1, 1)))
        return float(np.sqrt(np.mean(ang ** 2)))

    def focal_dev(self, f: float) -> float:
        """p90 over frames of |log(today's per-frame focal / f)|."""
        return float(np.percentile(np.abs(np.log(self.f[self.frames] / f)), 90))

    def states(self, frames, R=None, C=None, f=None, k1=None):
        n = len(frames)
        Rr = np.repeat(R[None], n, 0) if R is not None and R.ndim == 2 else (self.R[frames] if R is None else R)
        Cc = np.repeat(C[None], n, 0) if C is not None and C.ndim == 1 else (self.C[frames] if C is None else C)
        ff = np.full(n, f) if np.isscalar(f) else (self.f[frames] if f is None else f)
        kk = np.full(n, k1) if np.isscalar(k1) else (self.k1[frames] if k1 is None else k1)
        return Rr, Cc, np.log(ff), np.asarray(kk, float)

    def solve(self, frames, k_pts, roles, **init):
        b = Batch.of(self.d, frames, k_pts)
        R, C, lf, k1 = self.states(frames, **init)
        return gn_solve(b, R, C, lf, k1, self.cx, self.cy, roles, self.s.huber_px, self.s.iters)

    def solve_frames(self, frames, roles, **init):
        """FINAL per-frame solve (no shared blocks): a coarse pass on max_pts correspondences per frame, then a fine
        pass on every correspondence (final_pts 0) - the exported pose must not be noisier than today's."""
        s = self.s
        R, C, lf, k1 = self.states(frames, **init)
        r = gn_solve_frames(self.d, frames, s.max_pts, R, C, lf, k1, self.cx, self.cy, roles, s.huber_px, s.iters,
                            s.chunk_pts)
        if s.final_pts <= 0 or s.final_pts > s.max_pts:
            r = gn_solve_frames(self.d, frames, s.final_pts, r["R"], r["C"], r["lf"], r["k1"], self.cx, self.cy,
                                roles, s.huber_px, s.final_iters, s.chunk_pts)
        return r

    def evaluate(self, R, C, f, k1) -> np.ndarray:
        """Per-frame medians over self.frames (R (n,3,3) | (3,3), C (n,3) | (3,), f, k1 (n,) | scalar)."""
        n = len(self.frames)
        R = np.repeat(R[None], n, 0) if R.ndim == 2 else R
        C = np.repeat(C[None], n, 0) if C.ndim == 1 else C
        f = np.full(n, f) if np.isscalar(f) else f
        k1 = np.full(n, k1) if np.isscalar(k1) else k1
        return _medians(self.ev, R, C, f, k1, self.cx, self.cy)


def fit_intrinsics(cam: _Cam) -> tuple[float, float, dict]:
    """ONE focal + k1 for a video: joint robust fit (per-frame poses free) over <= intr_frames frames spread in time."""
    sel = _spread(cam.frames, cam.s.intr_frames)
    r = cam.solve(sel, max(40, cam.s.max_pts // 2), {"rot": "P", "cen": "P", "lf": "S", "k1": "S"},
                  f=cam.f0, k1=cam.k10)
    return float(np.exp(r["lf"][0])), float(r["k1"][0]), {"frames": int(len(sel)), "f0": round(cam.f0, 2),
                                                         "k1_0": round(cam.k10, 6), "cost_before": r["cost_before"],
                                                         "cost_after": r["cost_after"]}


def fit_constant(cam: _Cam, intr: tuple[float, float] | None) -> tuple[np.ndarray, np.ndarray, float, float]:
    """ONE pose (and one focal + k1 when intr is given as the start) explaining every frame's static points."""
    sel = _spread(cam.frames, cam.s.const_frames)
    w = np.maximum(np.asarray(cam.d["n_inliers"], float)[sel], 1.0)
    R0, C0 = _mean_rotation(cam.R[sel], w), np.median(cam.C[sel], 0)
    f0, k10 = intr if intr is not None else (cam.f0, cam.k10)
    r = cam.solve(sel, max(30, min(cam.s.max_pts, 6000 // max(1, len(sel)))),
                  {"rot": "S", "cen": "S", "lf": "S", "k1": "S"}, R=R0, C=C0, f=f0, k1=k10)
    return r["R"][0], r["C"][0], float(np.exp(r["lf"][0])), float(r["k1"][0])


def fit_shared_centre(cam: _Cam, intr: tuple[float, float] | None) -> tuple[np.ndarray, float | None, float | None]:
    """Fixed centre (tripod head): shared C, per-frame rotation, + per-frame focal (zoom: intr None) or the video's
    one focal + k1 (refined jointly)."""
    sel = _spread(cam.frames, cam.s.const_frames)
    C0 = np.median(cam.C[sel], 0)
    if intr is None:
        r = cam.solve(sel, max(30, cam.s.max_pts // 2), {"rot": "P", "cen": "S", "lf": "P", "k1": "F"}, C=C0)
        return r["C"][0], None, None
    r = cam.solve(sel, max(30, cam.s.max_pts // 2), {"rot": "P", "cen": "S", "lf": "S", "k1": "S"}, C=C0,
                  f=intr[0], k1=intr[1])
    return r["C"][0], float(np.exp(r["lf"][0])), float(r["k1"][0])


# ------------------------------------------------------------------------------------------------ refit

def _apply(d: dict, frames, R, C, f, k1, e_new: np.ndarray, model: str, cls: str, held: np.ndarray | None = None,
           frames_eval=None) -> dict:
    out = {k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v) for k, v in d.items()}
    frames = np.asarray(frames, int)
    out["R"][frames] = R
    out["tvec"][frames] = -np.einsum("nij,nj->ni", R, C)
    out["f"][frames] = f
    out["k1"][frames] = k1
    if "reproj_px" in out and frames_eval is not None:
        out["reproj_px"] = np.array(out["reproj_px"], float)
        out["reproj_px"][np.asarray(frames_eval, int)] = e_new
    F = len(d["t"])
    out["type_held"] = np.zeros(F, bool) if held is None else held
    if held is not None:
        if "source" in out:
            src = np.asarray(out["source"]).astype(object)
            src[held & ~np.asarray(d["ok"], bool)] = "type"
            out["source"] = src.astype(str)
        out["ok"] = np.asarray(out["ok"], bool) | held
    out["type_model"] = np.str_(model)
    out["camera_class"] = np.str_(cls)
    return out


def refit(d: dict, w: int, h: int, st, s: TypedSettings | None = None, log=None) -> tuple[dict, dict]:
    """Raw localisation dict -> (dict with the camera type's model applied, report). st: camera_types.StageType or
    None. The general class (handheld + zoom) and st=None return d itself: today's code path, unchanged."""
    s = s or TypedSettings()
    cls = getattr(st, "cls", "handheld_zoom")
    rep = {"class": cls, "source": getattr(st, "source", "default"), "model": "per_frame", "tried": []}
    if st is None or st.general or not s.enabled:
        rep["why"] = "general model (handheld + zoom): today's per-frame solver" if s.enabled else "disabled"
        return d, rep
    src = np.asarray(d.get("source", np.array([""] * len(d["t"]))))
    attached = bool((src == "attach").any())
    t0 = time.time()
    cam = _Cam(d, w, h, s)
    rep["frames_with_points"] = int(len(cam.frames))
    rep["localised"] = loc_mode(d)
    if len(cam.frames) < s.min_frames:
        rep.update(why=f"only {len(cam.frames)} frames with static points: model not applied",
                   check={"passed": False, "why": "too few frames"})
        return d, rep
    if attached and not st.tripod:
        rep["why"] = "attached camera: its own sub-model + attach model is kept (only a tripod gets one constant pose)"
        return d, rep
    fr = cam.frames
    lf = np.log(cam.f[fr])
    rep["per_frame"] = {"focal_median": round(float(np.exp(np.median(lf))), 2),
                        "focal_p5_p95_ratio": round(float(np.exp(np.percentile(lf, 95) - np.percentile(lf, 5))), 4),
                        "median_px": round(float(np.median(cam.e_old)), 3)}

    def tried(model, chk, extra=None):
        rep["tried"].append({"model": model, **chk, **(extra or {})})
        if log:
            log(f"[cameras] {cls}: {model} {'accepted' if chk['passed'] else 'REJECTED'} "
                f"(median {chk.get('median_px')} px vs per-frame {chk.get('per_frame_median_px')} px, "
                f"p90 {chk.get('p90_px')} vs {chk.get('per_frame_p90_px')})")

    def done(out, model, chk, **kw):
        rep.update(model=model, check=chk, seconds=round(time.time() - t0, 2),
                   at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **kw)
        return out, rep

    intr = None
    if st.fixed_focal and not attached:
        f_, k1_, info = fit_intrinsics(cam)
        intr = (f_, k1_)
        rep["intrinsics"] = {"f": round(f_, 3), "k1": round(k1_, 6),
                             "f_vs_per_frame_median_pct": round(100 * (f_ / rep["per_frame"]["focal_median"] - 1), 3),
                             **info}
    elif st.fixed_focal:  # attached: the sub-model's focal (its correspondences are bearings from the attach centre)
        intr = (cam.f0, cam.k10)
    F = len(d["t"])
    dr = cam.drift(lateral=bool(st.zoom)) if st.tripod else None
    if dr is not None:
        rep["centre_drift"] = dr

    def drift_check(chk, rel):
        if dr is None or cam.drift_ok(dr, rel):
            return dict(chk, drift_ok=True) if dr is not None else chk
        why = (f"the per-frame centres drift {dr['drift_rel_depth']:.4f} x the point depth "
               f"({dr['drift_over_noise']:.1f} x their noise): the camera translates")
        return dict(chk, passed=False, drift_ok=False, why=why + (f"; {chk['why']}" if chk.get("why") else ""))

    hf_old: dict = {}

    def focal_check(chk, f):
        """A fixed focal is contradicted when today's per-frame focal varies more than a zoom around it."""
        dev = cam.focal_dev(f)
        ok = bool(dev <= np.log(s.zoom_ratio))
        chk = dict(chk, focal_dev_p90_pct=round(100 * (np.exp(dev) - 1), 2), focal_ok=ok)
        if not ok:
            why = (f"today's per-frame focal deviates {100 * (np.exp(dev) - 1):.1f} % (p90) from one fixed focal: "
                   f"the camera zooms")
            chk.update(passed=False, why=why + (f"; {chk['why']}" if chk.get("why") else ""))
        return chk

    def jitter_check(chk, R_new, C_new=None):
        """The constrained track must not be shakier than today's: new <= jitter_ratio x today's + a small floor, for
        the centre (dolly-zoom: a wrong fixed focal turns a zoom into a slide along the axis that still fits the static
        points) and the rotation."""
        if not hf_old:
            hf_old.update(C=cam.hf_jitter(cam.C[fr]), R=cam.hf_rot(cam.R[fr]))
        whys = []
        if C_new is not None:
            hc = cam.hf_jitter(C_new)
            lim = s.jitter_ratio * hf_old["C"] + s.jitter_add_rel
            ok_c = bool(hc <= lim)
            chk = dict(chk, hf_centre_rel=round(hc, 6), hf_centre_rel_per_frame=round(hf_old["C"], 6),
                       hf_centre_rel_limit=round(lim, 6), jitter_ok=ok_c)
            if not ok_c:
                whys.append(f"the centre track gets {hc / max(hf_old['C'], 1e-12):.1f} x shakier than today's")
        hr = cam.hf_rot(R_new)
        lim_r = s.jitter_ratio * hf_old["R"] + s.rot_jitter_add_deg
        ok_r = bool(hr <= lim_r)
        chk = dict(chk, hf_rot_deg=round(hr, 5), hf_rot_deg_per_frame=round(hf_old["R"], 5),
                   hf_rot_deg_limit=round(lim_r, 5), rot_jitter_ok=ok_r)
        if not ok_r:
            whys.append(f"the rotation track gets {hr / max(hf_old['R'], 1e-12):.1f} x shakier than today's")
        if whys:
            chk.update(passed=False, why="; ".join(([chk["why"]] if chk.get("why") else []) + whys))
        return chk
    # 1. tripod (no zoom): one constant pose + one focal / k1
    if st.tripod and not st.zoom and not st.pans:
        Rk, Ck, fk, k1k = fit_constant(cam, intr if intr is not None else (cam.f0, cam.k10))
        e_new = cam.evaluate(Rk, Ck, fk, k1k)
        chk = focal_check(drift_check(residual_test(e_new, cam.e_old, s), s.drift_rel_locked), fk)
        tried("constant", chk, {"f": round(fk, 3), "k1": round(k1k, 6)})
        if chk["passed"] or s.strict:
            allf = np.arange(F)
            out = _apply(d, allf, np.repeat(Rk[None], F, 0), np.repeat(Ck[None], F, 0), np.full(F, fk),
                         np.full(F, k1k), e_new, "constant", cls, held=np.ones(F, bool), frames_eval=fr)
            return done(out, "constant", chk, held_frames=F,
                        pose={"C_sfm": Ck.round(5).tolist(), "f": round(fk, 3), "k1": round(k1k, 6)})
    if cam.sparse:  # keyframes only: every other model needs every frame localised (stage_loc / localize_typed)
        return done(d, "per_frame", {"passed": False, "keyframes_only": True,
                                     "why": "the constant pose does not fit the keyframes and this localisation holds "
                                            "keyframes only: every frame must be localised"})
    # 2. tripod (pans) / tripod + zoom: shared centre, per-frame rotation (+ focal with zoom), every correspondence
    if st.tripod:
        Cs, fs, k1s = fit_shared_centre(cam, intr)
        if fs is None:  # zoom: per-frame rotation + focal, k1 held per frame
            r = cam.solve_frames(fr, {"rot": "P", "lf": "P", "cen": "F", "k1": "F"}, C=Cs)
            Rf, ff, kf = r["R"], np.exp(r["lf"]), cam.k1[fr]
        else:
            r = cam.solve_frames(fr, {"rot": "P", "cen": "F", "lf": "F", "k1": "F"}, C=Cs, f=fs, k1=k1s)
            Rf, ff, kf = r["R"], np.full(len(fr), fs), np.full(len(fr), k1s)
        e_new = cam.evaluate(Rf, Cs, ff, kf)
        chk = drift_check(residual_test(e_new, cam.e_old, s), s.drift_rel_pans)
        if fs is not None:
            chk = focal_check(chk, fs)
        chk = jitter_check(chk, Rf)
        tried("fixed_centre", chk)
        if chk["passed"] or (s.strict and st.zoom):
            out = _apply(d, fr, Rf, np.repeat(Cs[None], len(fr), 0), ff, kf, e_new, "fixed_centre", cls,
                         frames_eval=fr)
            return done(out, "fixed_centre", chk, pose={"C_sfm": Cs.round(5).tolist()},
                        pans=bool(not st.zoom))
    # 3. no zoom: per-frame 6-dof pose with the video's one focal + k1 (not for attached cameras: see above), every
    # correspondence of each frame
    if intr is not None and not attached:
        r = cam.solve_frames(fr, {"rot": "P", "cen": "P", "lf": "F", "k1": "F"}, f=intr[0], k1=intr[1])
        e_new = cam.evaluate(r["R"], r["C"], intr[0], intr[1])
        chk = jitter_check(focal_check(residual_test(e_new, cam.e_old, s), intr[0]), r["R"], r["C"])
        tried("fixed_focal", chk)
        if chk["passed"] or s.strict:
            out = _apply(d, fr, r["R"], r["C"], np.full(len(fr), intr[0]), np.full(len(fr), intr[1]), e_new,
                         "fixed_focal", cls, frames_eval=fr)
            if st.tripod:  # a tripod that needed per-frame poses: the declared mount is contradicted
                chk = dict(chk, passed=False, why="per-frame poses needed: the camera moves (not a locked tripod)")
            return done(out, "fixed_focal", chk)
    return done(d, "per_frame", {"passed": False, "why": "no constrained model explains the static points as well "
                                                         "as today's per-frame solution: today's solver kept (type "
                                                         "conflict)"})


# ------------------------------------------------------------------------------------------------ localisation

LOC_MARKER_KEYS = ("loc_mode", "loc_class", "loc_policy", "loc_localised")  # additive keys of loc_<cam>.npz


def loc_marker(mode: str, cls: str = "handheld_zoom", policy: str = "user", localised: int | None = None) -> dict:
    """The marker stage_loc / front_pod write into cameras/cache/loc_<cam>.npz: how the camera was localised
    ('keyframes': a locked tripod's keyframes only, 'all': every window frame), for which class and policy."""
    return {"loc_mode": np.str_(mode), "loc_class": np.str_(cls), "loc_policy": np.str_(policy),
            "loc_localised": np.int64(-1 if localised is None else localised)}


def raw_dict(frames, poses, meta: dict | None = None) -> dict:
    """cameras/cache/loc_<cam>.npz layout (pipeline._save_raw) as a dict (+ the optional loc_marker keys)."""
    n = len(poses)
    corr_off = np.zeros(n + 1, np.int64)
    xy, xyz = [], []
    for i, fp in enumerate(poses):
        k = 0 if fp.corr is None or not fp.ok else len(fp.corr.xyz)
        if k:
            xy.append(fp.q_xy)
            xyz.append(fp.corr.xyz)
        corr_off[i + 1] = corr_off[i] + k
    return dict(t=np.array([r.t for r in frames]), ok=np.array([fp.ok for fp in poses]),
                R=np.array([fp.R for fp in poses]), tvec=np.array([fp.t for fp in poses]),
                f=np.array([fp.f for fp in poses]), k1=np.array([fp.k1 for fp in poses]),
                n_inliers=np.array([fp.n_inliers for fp in poses]), reproj_px=np.array([fp.reproj_px for fp in poses]),
                source=np.array([fp.source for fp in poses]), corr_off=corr_off,
                corr_xy=np.concatenate(xy) if xy else np.zeros((0, 2)),
                corr_xyz=np.concatenate(xyz) if xyz else np.zeros((0, 3)), **(meta or {}))


def keyframe_subset(t: np.ndarray, step_s: float, extra: list[int] | None = None, min_n: int = 5,
                    pairs: bool = False) -> list[int]:
    """Indices of the frames nearest to every step_s (first and last frame included) + extra (e.g. map frames).
    pairs=True adds each grid keyframe's next frame: the pair difference measures the per-frame noise (drift check)."""
    t = np.asarray(t, float)
    if len(t) == 0:
        return []
    grid = np.arange(t[0], t[-1] + 1e-9, step_s)
    base = {int(np.argmin(np.abs(t - g))) for g in grid} | {0, len(t) - 1}
    idx = base | {int(i) for i in (extra or [])}
    if len(idx) < min_n:
        idx |= {int(i) for i in np.round(np.linspace(0, len(t) - 1, min_n)).astype(int)}
    if pairs:
        idx |= {i + 1 if i + 1 < len(t) else i - 1 for i in base}
    return sorted(i for i in idx if 0 <= i < len(t))


def locked_tripod(st) -> bool:
    """The class localised on keyframes only: tripod, no zoom, not panning."""
    return bool(st is not None and getattr(st, "tripod", False) and not getattr(st, "zoom", True)
                and not getattr(st, "pans", None))


def localize_typed(L, frames, w: int, h: int, st, s: TypedSettings | None = None, log=print):
    """Locked tripods: localise keyframes (+ their next frame) only and fit one constant pose; every frame when the
    residual demands it. Other classes: localize.localize_camera unchanged. Returns (poses, info); info['mode']
    ('keyframes' | 'all') is what stage_loc records in loc_<cam>.npz (loc_marker)."""
    from .localize import FramePose, localize_camera
    s = s or TypedSettings()
    if not locked_tripod(st) or not s.enabled:
        return localize_camera(L, frames, w, h), {"mode": "all", "localised": len(frames)}
    t = np.array([r.t for r in frames])
    idx = keyframe_subset(t, s.tripod_step_s, [i for i, r in enumerate(frames) if r.key in L.refs], pairs=True)
    t0 = time.time()
    sub = localize_camera(L, [frames[i] for i in idx], w, h, log=False)
    poses = [FramePose() for _ in frames]
    for i, p in zip(idx, sub):
        poses[i] = p
    _, rep = refit(raw_dict(frames, poses, loc_marker("keyframes", st.cls, localised=len(idx))), w, h, st, s)
    info = {"mode": "keyframes", "localised": len(idx), "frames": len(frames), "seconds": round(time.time() - t0, 1),
            "check": rep.get("check"), "model": rep.get("model"), "centre_drift": rep.get("centre_drift")}
    if rep.get("model") == "constant":
        log(f"[cameras] {frames[0].cam}: tripod - {len(idx)}/{len(frames)} keyframes localised, one constant pose "
            f"(median {rep['check'].get('median_px')} px vs per-frame {rep['check'].get('per_frame_median_px')} px)")
        return poses, info
    log(f"[cameras] {frames[0].cam}: tripod keyframes do not fit one constant pose ({rep.get('model')}): localising "
        f"every frame")
    return localize_camera(L, frames, w, h), {**info, "mode": "all", "localised": len(frames),
                                               "why": "constant pose rejected on keyframes"}


# ------------------------------------------------------------------------------------------------ CLI (compare)

def parse_as(spec: str):
    """'tripod' | 'tripod_zoom' | 'handheld' | 'handheld_zoom' -> (mount, zoom)."""
    m, _, z = spec.partition("_")
    if m not in ("tripod", "handheld") or z not in ("", "zoom"):
        raise ValueError(f"bad camera type {spec!r}")
    return m, z == "zoom"


def compare(take: str, cams: list[str], as_types: list[str] | None = None, policy: str | None = None,
            s: TypedSettings | None = None) -> dict:
    """Today's per-frame tracks vs the camera-type models, from the cached raw localisation (no files written)."""
    from .. import camera_types as ct
    from ..take import Take
    from .pipeline import stage_smooth
    tk = Take.load(take)
    work = tk.work.resolve()
    sim3_p = work / "cameras" / "world_from_sfm.json"
    scale = json.loads(sim3_p.read_text())["s"] if sim3_p.exists() else None
    if as_types:
        types = {c: ct.StageType(c, *parse_as(a), None, "user") for c, a in zip(cams, as_types)}
    else:
        types = ct.stage_types(take, cams, policy)
    base = stage_smooth(work, cams, 0.04, 4.0, write=False, types={})
    new = stage_smooth(work, cams, 0.04, 4.0, write=False, types=types, typed_settings=s)
    out = {}
    for c in cams:
        if c not in base or c not in new:
            continue
        mb = ct.track_metrics(base[c]["t"], base[c]["R"], base[c]["C"], base[c]["f"], base[c]["ok"], scale)
        mn = ct.track_metrics(new[c]["t"], new[c]["R"], new[c]["C"], new[c]["f"], new[c]["ok"], scale)
        out[c] = {"type": new[c].get("camera_type"), "today": mb, "typed": mn,
                  "reproj_today": _rp(base[c]), "reproj_typed": _rp(new[c])}
    return out


def _rp(tr: dict) -> dict:
    r = tr["reproj_px"][tr["ok"]]
    r = r[np.isfinite(r)]
    return {"ok": int(tr["ok"].sum()), "median_px": round(float(np.median(r)), 3) if len(r) else None,
            "p90_px": round(float(np.percentile(r, 90)), 3) if len(r) else None}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m dancecap.cameras.typed", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("take")
    ap.add_argument("--cams", required=True)
    ap.add_argument("--as", dest="as_types", help="override per camera: tripod | tripod_zoom | handheld | "
                                                 "handheld_zoom (comma separated)")
    ap.add_argument("--policy", choices=["user", "auto", "off"])
    a = ap.parse_args(argv)
    cams = a.cams.split(",")
    res = compare(a.take, cams, a.as_types.split(",") if a.as_types else None, a.policy)
    print(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


if __name__ == "__main__":
    main()
