"""MAMMA with moving, zooming phone cameras: canonical frames -> ma_2d -> 30 Hz master grid -> ma_3d -> smplx/.

MAMMA assumes a static, synchronised rig. Our phones move, zoom and run at their own (variable) frame rates,
so this adapter keeps MAMMA's 2D network and optimiser and changes only what it is fed:

1. ma_2d runs per camera on the canonical frames (native rate) with OUR body masks
   (dancers/<cam>/body/mask_{i:04d}_{pid:02d}.png, ids 01 lead / 02 follow), never MAMMA's own segmentation.
2. Landmarks are undistorted (SIMPLE_RADIAL k1) and resampled onto the 30 Hz master grid: linear interpolation
   between consecutive valid frames with the uncertainty inflated by the interpolation distance; a landmark
   whose nearest valid frame is more than one frame interval away gets visibility 0 (and zero coordinates, so
   MAMMA masks it out entirely).
3. Cameras are resampled the same way (slerp R, lerp camera centre, log-focal, lerp principal point), mapped
   SfM -> world with cameras/world_from_sfm.json, and written per frame (cam_int (T,3,3), cam_ext (T,4,4) in
   METRES) into ma_cap-style npz files, which the turing-dynamic-cams branch of MAMMA accepts.
   Camera times include the per-camera residual video offsets of sync/offsets.json (dancecap.sync).
4. ma_3d runs from optimization/ with trusted ids (--trust-ids: re-ID only reports), dt = 1/30 and the up axis y,
   directly in the Y-up world frame (SMPL-X's canonical up is +Y, so MAMMA's upright init is right), with
   dancecap's config (MAMMA's contact config + a light temporal prior in the second stage, MA3D_OVERRIDES).
5. The result becomes smplx/{lead,follow}.npz (CONTRACTS.md) in the world frame. Two camera sets can be fitted
   (e.g. with and without the attached camera 05); "select" keeps the one that predicts the cameras it was not
   fitted on (or that shares) best.

MAMMA's uncertainty channel: ma_2d writes x, y (image px) and a log-variance in normalised crop units (the
crop of side box_px maps to [-1, 1]); ma_3d turns it into sigma_crop = sqrt(exp(lv)) / 2 * 512. We keep that
convention: sigma_img_px = sqrt(exp(lv)) / 2 * box_px, inflate in image pixels, and convert back.

Stages (the pod runs only "fit"; everything else is local):

    python -m dancecap mamma prepare larissa_kadu [--set c7=01,02,03,04,06,07,09 --set c8=all]  # ma_2d + inputs
    python -m dancecap mamma fit larissa_kadu --tag c8 [--fit-frames 0:3]   # ma_3d + export -> mamma/fits/<tag>/
    python -m dancecap mamma select larissa_kadu                        # pick a fit (held-out residuals) -> smplx/
    python -m dancecap mamma run larissa_kadu                           # all three locally
    python -m dancecap mamma --validate-synthetic [--noise 2]   # ground-truth check with moving/zooming cameras

Layout under work/<take>/mamma/: ma_2d_native/<cam>.npz (+ _boxes.npy, .manifest.json: recomputed when the body
masks, frames or ma_2d weights change), ma_2d/<take>_<tag>/cam<cam>.npz and ma_cap/<take>_<tag>/gt/ (grid inputs per
camera set), ma3d_config.yaml (MAMMA's contact config + dancecap overrides), prepare.json, pod_job.json,
ma_3d/<tag>/<seq>/ (raw ma_3d output), fits/<tag>/{smplx/,fit.json} (exported fit + per-camera residuals),
select.json. Paths come from dancecap.env so the same code runs on Windows and on a Linux Runpod pod.

MAMMA is non-commercial and must not be redistributed: its code stays in the local mamma-turing worktree and
its outputs (SMPL-X params, meshes) stay in work/ (git-ignored).
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import env
from .take import PACKAGE_ROOT, WORKSPACE

MAMMA_ROOT = env.mamma_root()  # local-only branch turing-dynamic-cams (never pushed)
MAMMA_DATA = env.mamma_data()  # weights + body models (git-ignored, live in the main checkout)
SMPLX_MODELS = MAMMA_DATA / "body_models" / "smplx_locked_head"
VERTS_512 = MAMMA_DATA / "body_models" / "downsampled_verts" / "verts_512.pkl"
PARTS_512 = MAMMA_DATA / "body_models" / "downsampled_verts" / "smplx_512_body_parts.npz"
MA2D_WEIGHTS = MAMMA_DATA / "weights" / "ma_2d" / "mamma_mask_full_cvpr.ckpt"
MA2D_CONFIG = "configs/train/models_2d/config_mammanet_mask_512.yaml"  # relative to landmarks/
MA3D_CONFIG = "config_files/contact_configs/config_real_gmf_small_vals_detr_exp_no_vtemplate.yaml"  # rel. optimization/
# dancecap overrides of MAMMA's contact config. Upstream second_run (500 Adam iterations, Geman-McClure reprojection)
# has no temporal term at all, so a frame whose reprojection gradient vanishes (the redescending loss gives no pull
# once a frame is far off) can stay metres away. A LIGHT acceleration prior (dt = 1/30 s; weights well below
# third_run's angular_acc 0.003 / pts3d_temp 0.03) pulls such frames back without smoothing real accents.
# third/fourth runs keep the upstream values.
MA3D_OVERRIDES = {"optim": {"second_run": {"losses": {"angular_acc_loss": {"weight": 0.0005},
                                                      "acceleration_loss": {"weight": 0.0002}}}}}
GRID_HZ = 30.0
N_LMK = 512
CROP_PX = 512  # MAMMA's sigma scale (losses.py: sigma / 2 * 512)
PIDS = (1, 2)  # 01 lead, 02 follow
ROLES = {1: "lead", 2: "follow"}
SIGMA_FLOOR_PX = 0.5
PARTS = ("body", "left_hand", "right_hand", "left_foot", "right_foot", "head")  # 512-point part split (quality)


def python() -> str:
    """Interpreter for the MAMMA scripts: the local venv on Windows, this interpreter on a pod (dancecap.env)."""
    return env.python_exe()


# ----------------------------------------------------------------------------------------------- geometry


def world_cameras(R_sfm: np.ndarray, t_sfm: np.ndarray, sim3: dict | None) -> tuple[np.ndarray, np.ndarray]:
    """SfM-frame cameras (x = R X_sfm + t) -> world-frame cameras in metres.

    world_from_sfm: X_w = s Rw X_sfm + tw. Then x_cam * s = R Rw^T X_w + (s t - R Rw^T tw), and projection is
    invariant to the overall scale, so R_w = R Rw^T, t_w = s t - R_w tw (camera coordinates now in metres).
    """
    if sim3 is None:
        return R_sfm.copy(), t_sfm.copy()
    s, Rw, tw = float(sim3["s"]), np.asarray(sim3["R"], float), np.asarray(sim3["t"], float)
    R_w = R_sfm @ Rw.T
    t_w = s * t_sfm - np.einsum("fij,j->fi", R_w, tw)
    return R_w, t_w


def sfm_cameras(R_w: np.ndarray, t_w: np.ndarray, sim3: dict) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of world_cameras (used to build synthetic SfM-frame inputs)."""
    s, Rw, tw = float(sim3["s"]), np.asarray(sim3["R"], float), np.asarray(sim3["t"], float)
    R_sfm = R_w @ Rw
    t_sfm = (t_w + np.einsum("fij,j->fi", R_w, tw)) / s
    return R_sfm, t_sfm


def extrinsics(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    E = np.zeros(R.shape[:-2] + (4, 4))
    E[..., :3, :3] = R
    E[..., :3, 3] = t
    E[..., 3, 3] = 1.0
    return E


def project(X: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pinhole projection of (..., N, 3) world points with per-frame (..., 3,3) K/R and (..., 3) t -> px, depth."""
    Xc = np.einsum("...ij,...nj->...ni", R, X) + t[..., None, :]
    z = Xc[..., 2]
    uv = np.einsum("...ij,...nj->...ni", K, Xc)
    with np.errstate(divide="ignore", invalid="ignore"):
        px = uv[..., :2] / uv[..., 2:3]
    return px, z


def distort_simple_radial(px: np.ndarray, K: np.ndarray, k1: float) -> np.ndarray:
    """COLMAP SIMPLE_RADIAL forward model (used to make synthetic distorted observations)."""
    f, cx, cy = K[0, 0], K[0, 2], K[1, 2]
    x, y = (px[..., 0] - cx) / f, (px[..., 1] - cy) / f
    r2 = x * x + y * y
    return np.stack([cx + f * x * (1 + k1 * r2), cy + f * y * (1 + k1 * r2)], -1)


def undistort_simple_radial(px: np.ndarray, K: np.ndarray, k1: float, iters: int = 10) -> np.ndarray:
    """Invert SIMPLE_RADIAL by fixed-point iteration (exact enough for |k1 r^2| << 1)."""
    if k1 == 0.0:
        return px.copy()
    f, cx, cy = K[0, 0], K[0, 2], K[1, 2]
    xd, yd = (px[..., 0] - cx) / f, (px[..., 1] - cy) / f
    x, y = xd.copy(), yd.copy()
    for _ in range(iters):
        r2 = x * x + y * y
        x, y = xd / (1 + k1 * r2), yd / (1 + k1 * r2)
    return np.stack([cx + f * x, cy + f * y], -1)


# ------------------------------------------------------------------------------------------- resampling


def logvar_to_sigma_px(logvar: np.ndarray, box_px: np.ndarray) -> np.ndarray:
    """ma_2d log-variance (normalised crop units) -> image-pixel sigma for a crop of side box_px."""
    return np.sqrt(np.exp(logvar)) / 2.0 * box_px


def sigma_px_to_logvar(sigma_px: np.ndarray, box_px: np.ndarray) -> np.ndarray:
    std = 2.0 * np.maximum(sigma_px, 1e-6) / np.maximum(box_px, 1.0)
    return 2.0 * np.log(std)


def master_grid(t_lo: float, t_hi: float, hz: float = GRID_HZ) -> np.ndarray:
    n = int(np.floor((t_hi - t_lo) * hz + 1e-9)) + 1
    return t_lo + np.arange(n) / hz


def _brackets(t_src: np.ndarray, grid: np.ndarray, frame_dt: float):
    """For each grid time: left/right source index, interpolation weight, and gaps in frame units."""
    j = np.clip(np.searchsorted(t_src, grid, side="right"), 1, len(t_src) - 1)
    i0, i1 = j - 1, j
    span = t_src[i1] - t_src[i0]
    w = np.clip((grid - t_src[i0]) / np.where(span > 0, span, 1.0), 0.0, 1.0)
    d0 = np.abs(grid - t_src[i0]) / frame_dt
    d1 = np.abs(t_src[i1] - grid) / frame_dt
    inside = (grid >= t_src[i0] - 1e-9) & (grid <= t_src[i1] + 1e-9)
    gap = np.where(inside, span / frame_dt, np.inf)  # outside the source range: never interpolate
    return i0, i1, w, d0, d1, gap


def resample_landmarks(t_src: np.ndarray, lm: np.ndarray, vis: np.ndarray, box_px: np.ndarray,
                       grid: np.ndarray, frame_dt: float | None = None,
                       extra: dict[str, np.ndarray] | None = None, extra_scale=1.0) -> dict[str, np.ndarray]:
    """Resample per-frame ma_2d output (F, P, N, 3) [x, y, logvar] onto grid times.

    A sample is valid when its coordinates are non-zero and vis > 0. Between two consecutive valid samples
    (gap <= 1.5 frame intervals) x, y and visibility are linearly interpolated and the image-pixel sigma is
    inflated by sigma_interp = |dx| * min(w, 1 - w) (zero on a frame, half the step at the midpoint). Otherwise
    the nearest valid sample is used if it lies within one frame interval, with sigma_interp = d * m (d =
    distance in frames, m = median per-frame landmark motion of that person in that camera); else the landmark
    is invisible (vis 0, coordinates 0). `extra` arrays (F, P, N), e.g. contacts, take the nearest sample.
    extra_scale multiplies sigma_interp before it is added in quadrature: ma_3d halves every sigma of a camera
    whose smallest crop sigma exceeds 15 px (losses.proj_pts_loss), which real ma_2d output always triggers, so
    build_inputs passes 2 there to keep the interpolation term at its intended pixel size. It is a scalar or one
    value per person (ma_3d decides the halving per camera AND person).
    """
    F, P, N, _ = lm.shape
    if frame_dt is None:
        frame_dt = float(np.median(np.diff(t_src))) if F > 1 else 1.0 / GRID_HZ
    T = len(grid)
    valid = (vis > 0) & (np.abs(lm[..., :2]).sum(-1) > 0)
    sig = logvar_to_sigma_px(lm[..., 2], box_px[..., None])
    i0, i1, w, d0, d1, gap = _brackets(t_src, grid, frame_dt)
    out_lm = np.zeros((T, P, N, 3), np.float32)
    out_vis = np.zeros((T, P, N), np.float32)
    out_extra = {k: np.zeros((T, P, N), np.float32) for k in (extra or {})}
    step = np.linalg.norm(np.diff(lm[..., :2], axis=0), axis=-1)  # (F-1, P, N)
    both_valid_step = valid[1:] & valid[:-1]
    scales = np.broadcast_to(np.asarray(extra_scale, float), (P,))
    for p in range(P):
        es = float(scales[p])
        m = np.median(step[:, p][both_valid_step[:, p]]) if both_valid_step[:, p].any() else 0.0
        v0, v1 = valid[i0, p], valid[i1, p]  # (T, N)
        ww = w[:, None]
        two = v0 & v1 & (gap[:, None] <= 1.5)
        x = (1 - ww)[..., None] * lm[i0, p, :, :2] + ww[..., None] * lm[i1, p, :, :2]
        s_lin = np.sqrt(((1 - ww) * sig[i0, p] + ww * sig[i1, p]) ** 2 +
                        (es * np.linalg.norm(lm[i1, p, :, :2] - lm[i0, p, :, :2], axis=-1)
                         * np.minimum(ww, 1 - ww)) ** 2)
        vis_lin = (1 - ww) * vis[i0, p] + ww * vis[i1, p]
        # nearest valid sample within one frame
        use0 = v0 & ((d0[:, None] <= d1[:, None]) | ~v1)
        use1 = v1 & ~use0
        dn = np.where(use0, d0[:, None], d1[:, None])
        near = (use0 | use1) & (dn <= 1.0 + 1e-6) & ~two
        src = np.where(use0, i0[:, None], i1[:, None])  # (T, N)
        n_idx = np.arange(N)[None, :]
        x_near = lm[src, p, n_idx, :2]
        s_near = np.sqrt(sig[src, p, n_idx] ** 2 + (es * dn * m) ** 2)
        vis_near = vis[src, p, n_idx]
        x_out = np.where(two[..., None], x, np.where(near[..., None], x_near, 0.0))
        s_out = np.where(two, s_lin, np.where(near, s_near, 1.0))
        v_out = np.where(two, vis_lin, np.where(near, vis_near, 0.0))
        box_g = (1 - w) * box_px[i0, p] + w * box_px[i1, p]
        out_lm[:, p, :, :2] = x_out
        out_lm[:, p, :, 2] = np.where(two | near, sigma_px_to_logvar(np.maximum(s_out, SIGMA_FLOOR_PX), box_g[:, None]), 0.0)
        out_vis[:, p] = v_out
        for k, arr in (extra or {}).items():
            nearest = np.where(np.abs(grid - t_src[i0]) <= np.abs(t_src[i1] - grid), i0, i1)
            out_extra[k][:, p] = np.where(two | near, arr[nearest, p], 0.0)
    return {"landmarks": out_lm, "visibilities": out_vis, **out_extra}


def _slerp_rotations(R: np.ndarray, i0: np.ndarray, i1: np.ndarray, w: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    q0 = Rotation.from_matrix(R[i0]).as_quat()
    q1 = Rotation.from_matrix(R[i1]).as_quat()
    q1 = np.where((q0 * q1).sum(-1, keepdims=True) < 0, -q1, q1)
    dot = np.clip((q0 * q1).sum(-1), -1.0, 1.0)
    th = np.arccos(dot)
    s = np.sin(th)
    small = s < 1e-8
    a = np.where(small, 1 - w, np.sin((1 - w) * th) / np.where(small, 1, s))
    b = np.where(small, w, np.sin(w * th) / np.where(small, 1, s))
    q = a[:, None] * q0 + b[:, None] * q1
    return Rotation.from_quat(q / np.linalg.norm(q, axis=-1, keepdims=True)).as_matrix()


def resample_cameras(t_src: np.ndarray, K: np.ndarray, R: np.ndarray, tvec: np.ndarray, ok: np.ndarray,
                     grid: np.ndarray, frame_dt: float | None = None) -> dict[str, np.ndarray]:
    """Per-frame cameras -> grid: slerp R, lerp camera centre, log-focal, lerp principal point.

    Only registered frames (ok) are used. A grid time is ok when it lies between two consecutive registered
    frames (gap <= 1.5 frame intervals) or within one frame interval of one; elsewhere the nearest registered
    camera is copied (so matrices stay finite) and ok is False (its landmarks must be dropped).
    """
    if frame_dt is None:
        frame_dt = float(np.median(np.diff(t_src))) if len(t_src) > 1 else 1.0 / GRID_HZ
    idx = np.flatnonzero(ok)
    if len(idx) == 0:
        raise ValueError("camera has no registered frames")
    ts = t_src[idx]
    if len(idx) == 1:
        T = len(grid)
        return {"K": np.repeat(K[idx], T, 0), "R": np.repeat(R[idx], T, 0), "t": np.repeat(tvec[idx], T, 0),
                "ok": np.abs(grid - ts[0]) / frame_dt <= 1.0}
    i0, i1, w, d0, d1, gap = _brackets(ts, grid, frame_dt)
    g0, g1 = idx[i0], idx[i1]
    two = gap <= 1.5
    near = np.minimum(d0, d1) <= 1.0 + 1e-6
    # outside a bracket: snap to the nearest registered frame
    w = np.where(two, w, np.where(d0 <= d1, 0.0, 1.0))
    Rg = _slerp_rotations(R, g0, g1, w)
    C = -np.einsum("fji,fj->fi", R, tvec)  # camera centres
    Cg = (1 - w)[:, None] * C[g0] + w[:, None] * C[g1]
    tg = -np.einsum("fij,fj->fi", Rg, Cg)
    Kg = np.zeros((len(grid), 3, 3))
    for a, b in ((0, 0), (1, 1)):
        Kg[:, a, b] = np.exp((1 - w) * np.log(K[g0, a, b]) + w * np.log(K[g1, a, b]))
    for a, b in ((0, 2), (1, 2)):
        Kg[:, a, b] = (1 - w) * K[g0, a, b] + w * K[g1, a, b]
    Kg[:, 2, 2] = 1.0
    return {"K": Kg, "R": Rg, "t": tg, "ok": two | near}


# ------------------------------------------------------------------------------------------- inputs


@dataclass
class CameraTrack:
    """One phone over the analysis window, world frame, metres; per canonical frame."""
    name: str
    t: np.ndarray        # (F,) reference seconds (sync-corrected)
    K: np.ndarray        # (F, 3, 3)
    R: np.ndarray        # (F, 3, 3)
    tvec: np.ndarray     # (F, 3) metres
    ok: np.ndarray       # (F,) bool
    k1: np.ndarray       # (F,) SIMPLE_RADIAL
    width: int
    height: int
    files: list[str]

    @property
    def frame_dt(self) -> float:
        return float(np.median(np.diff(self.t))) if len(self.t) > 1 else 1.0 / GRID_HZ


def read_sync_offsets(work: Path) -> dict[str, dict]:
    """sync/offsets.json -> {cam: {"delta": s, "rate": s/s, "t_ref": s}} (rate/t_ref optional, default 0 / 0).

    dancecap.sync always reports a drift rate but marks it "use_rate": true only when it should be applied.

    Accepts the dancecap.sync layout ({"offsets": {cam: {"delta": ..}}}) and plain {cam: seconds}. Cameras flagged
    "use": false in the file are returned with delta 0 (their estimate was not trusted).
    Refuses (SystemExit) a file measured on other audio clocks than the frames'/take's, and an unverifiable one for a
    take with [sync] required = true (dancecap.sync.offsets_check).
    """
    p = work / "sync" / "offsets.json"
    if not p.exists():
        return {}
    raw = json.loads(p.read_text())
    from . import sync as _sync  # lazy: sync imports this module
    chk = _sync.offsets_check(work, raw)
    if chk["status"] == "stale" or (chk["status"] == "unrecorded" and chk["required"]):
        raise SystemExit(f"[mamma] {p} is {chk['status']} ({'; '.join(chk['detail'])}): it was not measured on this "
                         f"take's audio clocks. Re-run 'python -m dancecap sync {work.name}' (never copy offsets.json "
                         "between takes)")
    raw = raw.get("offsets", raw)
    out = {}
    for cam, v in raw.items():
        if isinstance(v, dict):
            if v.get("use", True) is False:
                out[str(cam)] = {"delta": 0.0, "rate": 0.0, "t_ref": 0.0}
                continue
            d = v.get("delta", v.get("delta_c", v.get("offset", 0.0)))
            use_rate = v.get("use_rate", "rate" in v and "use_rate" not in v and "stable" not in v)
            rate = (v.get("rate") or 0.0) if use_rate else 0.0
            t_ref = v.get("t_ref", 0.0) or 0.0
        else:
            d, rate, t_ref = v, 0.0, 0.0
        try:
            out[str(cam)] = {"delta": float(d), "rate": float(rate), "t_ref": float(t_ref)}
        except (TypeError, ValueError):
            continue
    return out


def apply_offset(t: np.ndarray, off: dict | float | None) -> np.ndarray:
    """Sync-corrected reference time: t + delta (+ rate * (t - t_ref) when a drift was measured)."""
    if off is None:
        return np.asarray(t, float)
    if not isinstance(off, dict):
        return np.asarray(t, float) + float(off)
    t = np.asarray(t, float)
    return t + float(off.get("delta", 0.0)) + float(off.get("rate", 0.0)) * (t - float(off.get("t_ref", 0.0)))


def _sync_offsets(work: Path) -> dict[str, float]:
    """Constant part of the sync offsets (kept for callers that only need delta)."""
    return {c: v["delta"] for c, v in read_sync_offsets(work).items()}


def load_tracks(work: Path, cams: list[str] | None = None, allow_sfm_frame: bool = False,
                offsets: dict | None = None) -> dict[str, CameraTrack]:
    """frames/<cam>/frames.json + cameras/<cam>.npz (+ world_from_sfm.json, sync/offsets.json) -> tracks.

    Without cameras/world_from_sfm.json the cameras live in the arbitrary-scale SfM frame, which breaks the smplx/
    contract (Y-up metres, floor y=0) and MAMMA's metric priors: that is an error unless allow_sfm_frame is set
    (the caller must then keep the result out of smplx/). offsets=None reads sync/offsets.json, {} applies none.
    """
    sim3_p = work / "cameras" / "world_from_sfm.json"
    sim3 = json.loads(sim3_p.read_text()) if sim3_p.exists() else None
    if sim3 is None:
        if not allow_sfm_frame:
            raise SystemExit(f"[mamma] {sim3_p} is missing: cameras are still in the SfM frame (not metric, not Y-up). "
                             "Run the cameras stage's world step, or pass --allow-sfm-frame (result kept out of smplx/)")
        print("[mamma] WARNING: cameras/world_from_sfm.json missing: cameras used in the SfM frame "
              "(results will NOT be metric or Y-up and are never promoted to smplx/)")
    offsets = read_sync_offsets(work) if offsets is None else offsets
    tracks = {}
    for cam_npz in sorted((work / "cameras").glob("*.npz")):
        cam = cam_npz.stem
        if cams and cam not in cams:
            continue
        meta = json.loads((work / "frames" / cam / "frames.json").read_text())
        c = np.load(cam_npz)
        t = apply_offset(np.asarray(c["t"], float), offsets.get(cam))
        R, tv = world_cameras(np.asarray(c["R"], float), np.asarray(c["tvec"], float), sim3)
        F = len(t)
        tracks[cam] = CameraTrack(
            name=cam, t=t, K=np.asarray(c["K"], float), R=R, tvec=tv,
            ok=np.asarray(c["ok"], bool) if "ok" in c.files else np.ones(F, bool),
            k1=np.asarray(c["dist"], float).reshape(F, -1)[:, 0] if "dist" in c.files else np.zeros(F),
            width=int(meta["width"]), height=int(meta["height"]),
            files=[str(work / "frames" / cam / f["file"]) for f in meta["frames"]])
        if len(meta["frames"]) != F:
            raise ValueError(f"{cam}: cameras npz has {F} frames, frames.json {len(meta['frames'])}")
    if cams:
        missing = sorted(set(cams) - set(tracks))
        if missing:
            raise SystemExit(f"[mamma] no cameras/<cam>.npz for {missing}")
    return tracks


def mask_dir(work: Path, cam: str) -> Path:
    body = work / "dancers" / cam / "body"
    return body if body.exists() and any(body.glob("mask_*.png")) else work / "dancers" / cam / "masks"


def mask_boxes(work: Path, cam: str, n_frames: int) -> np.ndarray:
    """(F, 2) ma_2d crop side in px per frame and person: max(bbox w, h) * 1.2 (expand_to_aspect_ratio 1:1)."""
    import cv2
    boxes = np.zeros((n_frames, len(PIDS)), np.float32)
    d = mask_dir(work, cam)
    for i in range(n_frames):
        for p, pid in enumerate(PIDS):
            m = cv2.imread(str(d / f"mask_{i:04d}_{pid:02d}.png"), cv2.IMREAD_GRAYSCALE)
            if m is not None and m.any():
                _, _, bw, bh = cv2.boundingRect(m)
                boxes[i, p] = 1.2 * max(bw, bh)
    return boxes


_COPY_MARKER = ".dancecap_copy_of"


def _same_dir(a: Path, b: Path) -> bool:
    try:
        return os.path.normcase(os.path.realpath(a)) == os.path.normcase(os.path.realpath(b))
    except OSError:
        return False


def _is_junction(p: Path) -> bool:
    """Windows directory junction (os.readlink reads junctions as well as symlinks since Python 3.8)."""
    if p.is_symlink():
        return False
    try:
        os.readlink(p)
        return True
    except (OSError, ValueError):
        return False


def _remove_link(link: Path) -> None:
    """Remove a symlink, a Windows junction (the target is untouched) or one of our fallback copies."""
    if link.is_symlink():
        link.unlink()
    elif (link / _COPY_MARKER).exists():
        shutil.rmtree(link)
    elif _is_junction(link):
        os.rmdir(link)  # removes the junction only
    elif link.exists():
        raise RuntimeError(f"{link} is a real directory, refusing to replace it")


def link_dir(link: Path, target: Path) -> str:
    """Point `link` at directory `target`: junction on Windows (no admin rights), symlink elsewhere, copy as the last
    resort. An existing link/copy that points somewhere else (or a stale copy) is recreated. Returns the method."""
    target = target.resolve()
    if link.exists() or link.is_symlink():
        if (link / _COPY_MARKER).exists():
            if (link / _COPY_MARKER).read_text() == _dir_signature(target):
                return "copy (current)"
        elif _same_dir(link, target):
            return "link (current)"
        _remove_link(link)
    link.parent.mkdir(parents=True, exist_ok=True)
    if env.IS_WINDOWS:
        try:
            import _winapi
            _winapi.CreateJunction(str(target), str(link))
            return "junction"
        except (ImportError, OSError):
            pass
    else:
        try:
            os.symlink(target, link, target_is_directory=True)
            return "symlink"
        except OSError:
            pass
    shutil.copytree(target, link)
    (link / _COPY_MARKER).write_text(_dir_signature(target))
    return "copy"


def _dir_signature(d: Path, pattern: str = "*") -> str:
    """Cheap content signature of a directory: names, sizes and mtimes of its files."""
    h = hashlib.sha1(str(d.resolve()).encode())
    for f in sorted(d.glob(pattern)):
        if f.name == _COPY_MARKER:
            continue
        st = f.stat()
        h.update(f"{f.name}:{st.st_size}:{st.st_mtime_ns}\n".encode())
    return h.hexdigest()


def ma2d_manifest(work: Path, cam: str) -> dict:
    """What a camera's ma_2d output depends on: its mask dir (+ every mask file), frames.json, the weights."""
    d = mask_dir(work, cam)
    fj = work / "frames" / cam / "frames.json"
    w = MA2D_WEIGHTS.stat() if MA2D_WEIGHTS.exists() else None
    return {"mask_dir": d.relative_to(work).as_posix(), "n_masks": len(list(d.glob("mask_*.png"))),
            "masks_sig": _dir_signature(d, "mask_*.png"),
            "frames_json": [fj.stat().st_size, fj.stat().st_mtime_ns] if fj.exists() else None,
            "weights": [MA2D_WEIGHTS.name, w.st_size] if w else None, "config": MA2D_CONFIG}


def _manifest_matches(old: dict, new: dict) -> bool:
    keys = ("mask_dir", "n_masks", "masks_sig", "frames_json", "weights", "config")
    return all(old.get(k) == new.get(k) for k in keys)


# ------------------------------------------------------------------------------------------- MAMMA runs


def _env() -> dict:
    env_ = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    env_["PYTHONPATH"] = os.pathsep.join([str(MAMMA_ROOT), env_.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    if not env.IS_WINDOWS:
        env_.setdefault("PYOPENGL_PLATFORM", "egl")  # pyrender is imported by ma_3d even with --skip_scene_videos
    return env_


class _VramSampler:
    """Samples whole-GPU memory.used via nvidia-smi while a child runs (WDDM hides per-process numbers)."""

    def __init__(self, period_s: float = 1.0):
        self.period_s, self.peak_mib, self.base_mib = period_s, 0, None
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._run, daemon=True)

    def _read(self) -> int | None:
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=10).stdout
            return int(out.split()[0])
        except (OSError, ValueError, IndexError, subprocess.SubprocessError):
            return None

    def _run(self):
        while not self._stop.is_set():
            v = self._read()
            if v is not None:
                self.peak_mib = max(self.peak_mib, v)
            self._stop.wait(self.period_s)

    def __enter__(self):
        self.base_mib = self._read()
        self._th.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._th.join()


def _run_logged(cmd: list, cwd: Path, log: Path, label: str) -> dict:
    from . import gpu
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"[mamma] {label}: {' '.join(str(c) for c in cmd)}\n[mamma]   cwd={cwd} log={log}")
    with gpu.exclusive(label), _VramSampler() as vram, open(log, "w", encoding="utf-8") as fh:
        t0 = time.time()
        proc = subprocess.run([str(c) for c in cmd], cwd=cwd, env=_env(), stdout=fh, stderr=subprocess.STDOUT)
        elapsed = time.time() - t0
    stats = {"seconds": round(elapsed, 1), "gpu_used_peak_mib": vram.peak_mib, "gpu_used_before_mib": vram.base_mib,
             "returncode": proc.returncode, "host": "pod" if env.is_pod() else "local"}
    text = log.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        if line.startswith(("[mem]", "Re-ID:", "[sdf]", "[dt]", "[turing]", "Optimization took")):
            print(f"[mamma]   {line}")
    if proc.returncode != 0:
        tail = "\n".join(text.splitlines()[-30:])
        raise RuntimeError(f"{label} failed (exit {proc.returncode}); log {log}\n{tail}")
    return stats


def run_ma2d(work: Path, cams: list[str], out_dir: Path, force: bool = False) -> dict:
    """MAMMA ma_2d on the canonical frames with our body masks; writes out_dir/<cam>.npz (+ box sizes, manifest).

    A camera is recomputed when forced or when its manifest (mask dir, mask files' sizes/mtimes, frames.json,
    weights) differs from the one stored next to its output, so re-running the dancers stage never leaves stale
    landmarks or crop sizes behind. Returns {cam: "cached" | "computed"}.
    """
    stage = work / "mamma" / "ma2d_masks"
    out_dir.mkdir(parents=True, exist_ok=True)
    status, todo, manifests = {}, [], {}
    for cam in cams:
        man = ma2d_manifest(work, cam)
        manifests[cam] = man
        mp = out_dir / f"{cam}.manifest.json"
        old = json.loads(mp.read_text()) if mp.exists() else None
        if not force and old and _manifest_matches(old, man) and (out_dir / f"{cam}.npz").exists() \
                and (out_dir / f"{cam}_boxes.npy").exists():
            status[cam] = "cached"
            continue
        for f in (out_dir / f"{cam}.npz", out_dir / f"{cam}_boxes.npy", mp):
            f.unlink(missing_ok=True)  # ma_2d itself skips cameras whose npz exists
        how = link_dir(stage / cam / "masks", mask_dir(work, cam))
        print(f"[mamma] ma_2d {cam}: {'forced' if force else 'inputs changed' if old else 'no output yet'}; "
              f"masks {man['mask_dir']} ({man['n_masks']} files, {how})")
        todo.append(cam)
    if todo:
        cmd = [python(), "run_ma_2d.py", "--images_root_dir", work / "frames", "--cam_names", *todo,
               "--mask_path", stage, "--out_folder", out_dir, "--config_path", MA2D_CONFIG,
               "--weights", MA2D_WEIGHTS, "--downsampled-verts", VERTS_512, "--disable-visualizations"]
        stats = _run_logged(cmd, MAMMA_ROOT / "landmarks", work / "mamma" / "logs" / "ma_2d.log", "mamma ma_2d")
        print(f"[mamma] ma_2d {todo}: {stats['seconds']:.0f} s, GPU peak {stats['gpu_used_peak_mib']} MiB")
    for cam in todo:
        n = np.load(out_dir / f"{cam}.npz")["landmarks"].shape[0]
        np.save(out_dir / f"{cam}_boxes.npy", mask_boxes(work, cam, n))
        (out_dir / f"{cam}.manifest.json").write_text(json.dumps(
            dict(manifests[cam], created=time.strftime("%Y-%m-%dT%H:%M:%S")), indent=1))
        status[cam] = "computed"
    return status


def native_observations(native_dir: Path, tr: CameraTrack) -> dict[str, np.ndarray]:
    """One camera's native ma_2d output, ready for geometry: (F, P, N, 3) [x, y, logvar] with the SIMPLE_RADIAL
    distortion removed (MAMMA is pinhole) and zeroed where the camera is not registered; vis (F, P, N); box (F, P);
    extra {contacts, floor_contacts} when present."""
    cam = tr.name
    d2 = np.load(native_dir / f"{cam}.npz")
    lm = np.array(d2["landmarks"], np.float64)[:, :len(PIDS)]
    vis = np.array(d2["visibilities"], np.float64)[:, :len(PIDS)]
    if lm.shape[0] != len(tr.t):
        raise ValueError(f"{cam}: ma_2d has {lm.shape[0]} frames, camera track {len(tr.t)}")
    extra = {k: np.array(d2[k], np.float64)[:, :len(PIDS)] for k in ("contacts", "floor_contacts")
             if k in d2.files and d2[k].dtype != object and d2[k].shape[:2] == lm.shape[:2]}
    if lm.shape[1] < len(PIDS):  # ma_2d only exports up to the max observed id
        pad = len(PIDS) - lm.shape[1]
        lm = np.concatenate([lm, np.zeros((lm.shape[0], pad) + lm.shape[2:])], 1)
        vis = np.concatenate([vis, np.zeros((vis.shape[0], pad) + vis.shape[2:])], 1)
        extra = {k: np.concatenate([v, np.zeros((v.shape[0], pad) + v.shape[2:])], 1) for k, v in extra.items()}
    box_p = native_dir / f"{cam}_boxes.npy"
    box = np.load(box_p).astype(np.float64) if box_p.exists() else _boxes_from_landmarks(lm, vis)
    vis[~tr.ok] = 0.0  # camera not registered on a frame -> its landmarks cannot be used
    lm[~tr.ok] = 0.0
    observed = np.abs(lm[..., :2]).sum(-1) > 0  # zero coordinates = no detection; keep them zero
    for f in np.flatnonzero(tr.k1 != 0):  # pinhole model in MAMMA: undistort observations
        lm[f, ..., :2] = np.where(observed[f, ..., None],
                                  undistort_simple_radial(lm[f, ..., :2], tr.K[f], float(tr.k1[f])), 0.0)
    return {"lm": lm, "vis": vis, "box": box, "observed": observed, "extra": extra}


def _ma3d_halves(lm: np.ndarray, vis: np.ndarray, p: int) -> bool:
    """ma_3d (losses.proj_pts_loss) halves every sigma of a camera-person whose smallest loaded crop sigma > 15 px.
    Its min runs over all loaded points, where invisible ones carry logvar 0 (sigma 256), so only valid ones matter."""
    ok = (vis[:, p] > 0) & (np.abs(lm[:, p, :, :2]).sum(-1) > 0)
    if not ok.any():
        return False
    return float((np.sqrt(np.exp(lm[:, p, :, 2][ok])) / 2 * CROP_PX).min()) > 15


def nearest_index(t_src: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Index of the source sample nearest to each time (not the right-hand neighbour searchsorted returns)."""
    i = np.clip(np.searchsorted(t_src, t), 1, max(len(t_src) - 1, 1))
    if len(t_src) == 1:
        return np.zeros(len(t), int)
    left = np.abs(t - t_src[i - 1]) <= np.abs(t_src[i] - t)
    return np.where(left, i - 1, i)


def build_inputs(work: Path, tracks: dict[str, CameraTrack], native_dir: Path, grid: np.ndarray,
                 seq: str) -> dict:
    """Native ma_2d npz + camera tracks -> grid-resampled ma_2d + ma_cap npz under work/mamma/."""
    root = work / "mamma"
    gt = root / "ma_cap" / seq / "gt"
    lm_dir = root / "ma_2d" / seq
    for d in (gt, lm_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    report = {}
    for cam, tr in tracks.items():
        obs = native_observations(native_dir, tr)
        lm, vis, box, extra = obs["lm"], obs["vis"], obs["box"], obs["extra"]
        # per person, as ma_3d decides it; checked again on what ma_3d will actually load
        halved = np.array([_ma3d_halves(lm, vis, p) for p in range(len(PIDS))])
        rs = resample_landmarks(tr.t, lm, vis, box, grid, tr.frame_dt, extra, extra_scale=np.where(halved, 2.0, 1.0))
        cg = resample_cameras(tr.t, tr.K, tr.R, tr.tvec, tr.ok, grid, tr.frame_dt)
        rs["landmarks"][~cg["ok"]] = 0.0
        rs["visibilities"][~cg["ok"]] = 0.0
        for k in extra:
            rs[k][~cg["ok"]] = 0.0
        loaded = np.array([_ma3d_halves(rs["landmarks"], rs["visibilities"], p) for p in range(len(PIDS))])
        if (loaded != halved).any():  # resampling moved a camera-person across ma_3d's 15 px rule: redo with it
            halved = loaded
            rs = resample_landmarks(tr.t, lm, vis, box, grid, tr.frame_dt, extra,
                                    extra_scale=np.where(halved, 2.0, 1.0))
            rs["landmarks"][~cg["ok"]] = 0.0
            rs["visibilities"][~cg["ok"]] = 0.0
            for k in extra:
                rs[k][~cg["ok"]] = 0.0
        name = f"cam{cam}"
        np.savez(lm_dir / f"{name}.npz", landmarks=rs["landmarks"].astype(np.float32),
                 visibilities=rs["visibilities"].astype(np.float32),
                 contacts=rs.get("contacts", np.zeros_like(rs["visibilities"])).astype(np.float32),
                 floor_contacts=rs.get("floor_contacts", np.zeros_like(rs["visibilities"])).astype(np.float32))
        nearest = nearest_index(tr.t, grid)
        paths = [tr.files[i] for i in nearest] if tr.files else []
        rel = [Path(p).relative_to(work).as_posix() if Path(p).is_relative_to(work) else p for p in paths]
        np.savez(gt / f"{name}.npz", img_abs_path=np.array(paths), img_rel_path=np.array(rel),
                 cam_name=np.array(name), cam_int=cg["K"].astype(np.float32),
                 cam_ext=extrinsics(cg["R"], cg["t"]).astype(np.float32),
                 cam_img_w=int(tr.width), cam_img_h=int(tr.height), cam_portrait=bool(tr.height > tr.width),
                 frame_start=0, frame_end=len(grid), video_path=np.array(""),
                 distortion_model=np.array("radtan"), distortion_coeffs=np.zeros(4),
                 is_body_in_img=np.ones(len(grid), bool), t=grid, native_index=nearest)
        report[cam] = {"native_frames": int(len(tr.t)), "native_fps": round(1 / tr.frame_dt, 3),
                       "grid_cam_ok": float(cg["ok"].mean()),
                       "sigma_halved_by_ma3d": {ROLES[pid]: bool(halved[p]) for p, pid in enumerate(PIDS)},
                       "grid_visible_frac": {ROLES[pid]: float((rs["visibilities"][:, p] > 0).mean())
                                             for p, pid in enumerate(PIDS)}}
    np.savez(gt / "global.npz", params_paths=np.array([]), ioi_dir="", seq_name=seq,
             frames_len=int(len(grid)), fps=int(round(GRID_HZ)), frame_start=0, frame_end=int(len(grid)),
             people_len=len(PIDS), t=grid)
    return report


def _boxes_from_landmarks(lm: np.ndarray, vis: np.ndarray) -> np.ndarray:
    """Fallback crop size when no masks are around: 1.2 x the larger side of the visible landmark extent."""
    F, P = lm.shape[:2]
    out = np.full((F, P), 512.0)
    for f in range(F):
        for p in range(P):
            v = (vis[f, p] > 0) & (np.abs(lm[f, p, :, :2]).sum(-1) > 0)
            if v.sum() >= 3:
                ext = lm[f, p, v, :2].max(0) - lm[f, p, v, :2].min(0)
                out[f, p] = 1.2 * max(ext.max(), 1.0)
    return out


def write_ma3d_config(out: Path, base: str = MA3D_CONFIG, overrides: dict | None = None) -> Path:
    """MAMMA's contact config with dancecap's overrides merged in (deep merge) -> out (yaml)."""
    import yaml

    def merge(a: dict, b: dict) -> dict:
        for k, v in b.items():
            a[k] = merge(a.get(k) or {}, v) if isinstance(v, dict) else v
        return a

    cfg = yaml.safe_load((MAMMA_ROOT / "optimization" / base).read_text())
    cfg = merge(cfg, MA3D_OVERRIDES if overrides is None else overrides)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("# dancecap: " + base + " + mamma_dyn.MA3D_OVERRIDES (generated, do not edit)\n"
                   + yaml.safe_dump(cfg, sort_keys=False))
    return out


def run_ma3d(work: Path, seq: str, cam_names: list[str], config: str | Path = MA3D_CONFIG, tag: str = "run",
             frames: tuple[int, int] | None = None, extra_args: list[str] | None = None) -> tuple[Path, dict]:
    """ma_3d on work/mamma/{ma_2d,ma_cap}/<seq> -> work/mamma/ma_3d/<tag>/<seq>. config: path relative to
    optimization/ (MAMMA's own) or absolute (dancecap's ma3d_config.yaml). frames=(a, b): only grid frames a..b-1."""
    root = work / "mamma"
    out = root / "ma_3d" / tag
    if out.exists():
        shutil.rmtree(out)
    cmd = [python(), "run_ma_3d.py", "--seq_name", seq, "--ma_2d_dir", root / "ma_2d", "--ma_cap_dir", root / "ma_cap",
           "--config_file", config, "--cam_names", *cam_names, "--out_path", out,
           "--smplx-models", SMPLX_MODELS, "--downsampled-verts", VERTS_512,
           "--fps", str(GRID_HZ), "--trust-ids", "--up-axis=y", "--skip_detection_analysis", "--skip_scene_videos",
           *(["--start_frame", str(frames[0]), "--end_frame", str(frames[1])] if frames else []),
           *(extra_args or [])]
    log_p = root / "logs" / f"ma_3d_{tag}.log"
    stats = _run_logged(cmd, MAMMA_ROOT / "optimization", log_p, "mamma ma_3d")
    log = log_p.read_text(encoding="utf-8", errors="replace")
    stats["sdf_active"] = "[sdf] pytorch_sdf loaded" in log
    stats["reid_lines"] = [l for l in log.splitlines() if l.startswith("Re-ID:")]
    stats["dt_lines"] = [l for l in log.splitlines() if l.startswith(("[dt]", "[turing]"))][:4]
    for l in log.splitlines():
        if l.startswith("[mem]"):
            stats["torch_peak"] = l
    return out / seq, stats


# ------------------------------------------------------------------------------------------- export


def _load_verts512():
    import torch  # noqa: F401  (the pickle holds a torch tensor)
    with open(VERTS_512, "rb") as fh:
        m = pickle.load(fh)
    return np.asarray(m.cpu().numpy() if hasattr(m, "cpu") else m, np.float64)


def part_index() -> dict[str, np.ndarray]:
    """Indices of MAMMA's 512 points per body part (left/right head merged); 'body' = torso + arms + legs."""
    if not PARTS_512.exists():
        return {"body": np.arange(N_LMK)}
    d = np.load(PARTS_512)
    return {"body": d["body"], "left_hand": d["left_hand"], "right_hand": d["right_hand"],
            "left_foot": d["left_feet"], "right_foot": d["right_feet"],
            "head": np.concatenate([d["left_head"], d["right_head"]])}


def export_smplx(run_dir: Path, grid: np.ndarray, gt_dir: Path, lm_dir: Path, out_dir: Path,
                 frame0: int = 0) -> dict:
    """ma_3d smplx_params/verts_joints -> smplx/{lead,follow}.npz (contract), plus per-frame reprojection quality.

    frame0: first grid frame of a partial (--frames) run; the landmarks/cameras are sliced to match."""
    hands = np.load(SMPLX_MODELS / "smplx" / "SMPLX_NEUTRAL.npz", allow_pickle=True)
    hm_l, hm_r = np.asarray(hands["hands_meanl"], float), np.asarray(hands["hands_meanr"], float)
    D = _load_verts512()
    parts = part_index()
    cams = sorted(p.stem for p in gt_dir.glob("cam*.npz"))
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for p, pid in enumerate(PIDS):
        prm = np.load(run_dir / f"smplx_params_body_id-{p:02d}.npz", allow_pickle=True)
        vj = np.load(run_dir / f"verts_joints_body_id-{p:02d}.npz")
        pose = np.asarray(prm["smplx_pose"], float)  # (T,165): global, body, jaw, leye, reye, lhand, rhand
        T = len(pose)
        g = grid[frame0:frame0 + T]
        if len(g) != T:
            raise ValueError(f"ma_3d returned {T} frames, grid slice has {len(g)}")
        sl = slice(frame0, frame0 + T)
        verts = np.asarray(vj["pred_vertices"], float)
        joints = np.asarray(vj["pred_joints"], float)[:, :55]
        flat = bool(prm["smplx_export_flat_hand_mean"]) if "smplx_export_flat_hand_mean" in prm.files else False
        lh, rh = pose[:, 75:120], pose[:, 120:165]
        if not flat:  # contract hand poses are absolute (flat_hand_mean=True)
            lh, rh = lh + hm_l, rh + hm_r
        transl = np.asarray(prm["smplx_translation"], float)
        betas = np.asarray(prm["smplx_betas"], float).reshape(-1)
        rp = reprojection_parts(np.matmul(D, verts), cams, gt_dir, lm_dir, p, parts, sl)
        reproj = rp["all"]
        bad, suspect = outlier_frames(reproj, return_suspect=True)
        reproj_raw = reproj.copy()
        if bad.any() and (~bad).sum() >= 2:
            # ma_3d occasionally throws a single frame metres away (the redescending Geman-McClure loss gives no
            # pull back, upstream has no temporal term before the third stage). Re-fill those frames from their
            # good neighbours (slerp per joint, lerp translation), redo FK, and report them.
            pose, transl = _interpolate_frames(pose, transl, bad)
            verts, joints = _forward(pose, transl, betas, flat)
            lh, rh = pose[:, 75:120], pose[:, 120:165]
            if not flat:
                lh, rh = lh + hm_l, rh + hm_r
            rp = reprojection_parts(np.matmul(D, verts), cams, gt_dir, lm_dir, p, parts, sl)
            reproj = rp["all"]
            print(f"[mamma] {ROLES[pid]}: repaired {int(bad.sum())} outlier frame(s) {np.flatnonzero(bad).tolist()} "
                  f"(reprojection {np.round(reproj_raw[bad], 1).tolist()} px -> {np.round(reproj[bad], 1).tolist()} px)")
        part_px = np.stack([rp[k] for k in PARTS], 1)  # (T, n_parts), NaN = part not observed
        # quality = the worst of body (torso + limbs), feet and head, so a limb-level failure is visible (the median
        # over all 512 points hid a 16 cm wrist error). Hands are in quality_parts only: ma_2d's hand points are
        # the noisiest (~25-30 px) on every frame and would make quality uninformative.
        core = [k for k in PARTS if "hand" not in k]
        cp = np.stack([rp[k] for k in core], 1)
        worst = np.nanmax(np.where(np.isfinite(cp), cp, -np.inf), 1)
        worst = np.where(np.isfinite(worst), np.maximum(worst, np.nan_to_num(reproj, nan=0.0)), reproj)
        quality = np.exp(-np.nan_to_num(worst, nan=1e3) / 10.0)
        quality_parts = np.exp(-np.nan_to_num(part_px, nan=1e3) / 10.0)
        quality[bad] = np.minimum(quality[bad], 0.1)  # interpolated, not measured
        contact = prm["smplx_contact"] if prm["smplx_contact"].dtype != object else np.zeros((T, N_LMK))
        floor = prm["smplx_floor_contact"] if prm["smplx_floor_contact"].dtype != object else np.zeros((T, N_LMK))
        role = ROLES[pid]
        np.savez_compressed(
            out_dir / f"{role}.npz", t=g, global_orient=pose[:, :3], body_pose=pose[:, 3:66],
            jaw_pose=pose[:, 66:69], leye_pose=pose[:, 69:72], reye_pose=pose[:, 72:75],
            left_hand_pose=lh, right_hand_pose=rh, transl=transl, betas=betas, joints=joints,
            repaired=bad, suspect=suspect & ~bad, reproj_px_raw=reproj_raw,
            contact=np.asarray(contact, np.float32), floor_contact=np.asarray(floor, np.float32),
            quality=quality, quality_parts=quality_parts.astype(np.float32), reproj_parts_px=part_px.astype(np.float32),
            parts=np.array(PARTS), reproj_px=reproj, gender="neutral", model="smplx_locked_head (MAMMA)",
            flat_hand_mean=True, num_betas=int(np.asarray(prm["smplx_betas"]).shape[-1]), source="mamma ma_3d")
        summary[role] = {"frames": int(T), "median_reproj_px": float(np.nanmedian(reproj)),
                         "median_reproj_parts_px": {k: float(np.nanmedian(rp[k])) if np.isfinite(rp[k]).any() else None
                                                    for k in PARTS},
                         "quality_median": float(np.median(quality)),
                         "repaired_frames": np.flatnonzero(bad).tolist(),
                         "suspect_not_repaired": np.flatnonzero(suspect & ~bad).tolist(),
                         "betas0": float(np.asarray(prm["smplx_betas"]).reshape(-1)[0])}
    return summary


def outlier_frames(reproj: np.ndarray, min_px: float = 10.0, factor: float = 4.0, neighbour_factor: float = 3.0,
                   return_suspect: bool = False):
    """Frames to re-fill by interpolation: median reprojection above max(min_px, factor x sequence median) or
    undefined, plus the +-1 neighbours of such frames above neighbour_factor x median (partial blow-ups next to a bad
    frame must not anchor the interpolation). return_suspect also returns every frame above neighbour_factor x
    median (reported, not necessarily repaired)."""
    med = np.nanmedian(reproj) if np.isfinite(reproj).any() else 0.0
    bad = ~np.isfinite(reproj) | (reproj > max(min_px, factor * med))
    high = np.isfinite(reproj) & (reproj > neighbour_factor * med)
    near = np.zeros_like(bad)
    near[1:] |= bad[:-1]
    near[:-1] |= bad[1:]
    bad = bad | (near & high)
    return (bad, high) if return_suspect else bad


def _interpolate_frames(pose: np.ndarray, transl: np.ndarray, bad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Replace bad frames: per-joint slerp of axis-angle rotations, linear translation (ends: nearest good)."""
    from scipy.spatial.transform import Rotation, Slerp
    T = len(pose)
    good = np.flatnonzero(~bad)
    tb = np.flatnonzero(bad)
    tq = np.clip(tb, good[0], good[-1]).astype(float)
    pose, transl = pose.copy(), transl.copy()
    aa = pose.reshape(T, -1, 3)
    for j in range(aa.shape[1]):
        sl = Slerp(good.astype(float), Rotation.from_rotvec(aa[good, j]))
        aa[tb, j] = sl(tq).as_rotvec()
    pose = aa.reshape(T, -1)
    for k in range(3):
        transl[tb, k] = np.interp(tq, good, transl[good, k])
    return pose, transl


def _forward(pose: np.ndarray, transl: np.ndarray, betas: np.ndarray, flat_hand_mean: bool):
    """MAMMA's locked-head neutral SMPL-X FK (pose in ma_3d's 165-D layout) -> vertices, joints[:, :55]."""
    import smplx
    import torch
    T = len(pose)
    m = smplx.create(str(SMPLX_MODELS), model_type="smplx", gender="neutral", ext="npz", use_pca=False,
                     flat_hand_mean=flat_hand_mean, num_betas=len(betas), batch_size=T)
    f = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)
    with torch.no_grad():
        o = m(betas=f(betas)[None].expand(T, -1), global_orient=f(pose[:, :3]), body_pose=f(pose[:, 3:66]),
              jaw_pose=f(pose[:, 66:69]), leye_pose=f(pose[:, 69:72]), reye_pose=f(pose[:, 72:75]),
              left_hand_pose=f(pose[:, 75:120]), right_hand_pose=f(pose[:, 120:165]), transl=f(transl))
    return o.vertices.numpy().astype(np.float64), o.joints.numpy()[:, :55].astype(np.float64)


def _camera_errors(pts512: np.ndarray, cam: str, gt_dir: Path, lm_dir: Path, p: int,
                   sl: slice = slice(None)) -> tuple[np.ndarray, np.ndarray]:
    """(T, 512) pixel errors of the fitted points in one camera and the (T, 512) mask of visible observations."""
    g, l = np.load(gt_dir / f"{cam}.npz"), np.load(lm_dir / f"{cam}.npz")
    E, K = np.asarray(g["cam_ext"], float)[sl], np.asarray(g["cam_int"], float)[sl]
    px, z = project(pts512, K, E[:, :3, :3], E[:, :3, 3])
    obs, vis = l["landmarks"][sl, p, :, :2], l["visibilities"][sl, p]
    ok = (vis > 0.5) & (np.abs(obs).sum(-1) > 0) & (z > 0)
    return np.linalg.norm(px - obs, axis=-1), ok


def reprojection_parts(pts512: np.ndarray, cams: list[str], gt_dir: Path, lm_dir: Path, p: int,
                       parts: dict[str, np.ndarray] | None = None, sl: slice = slice(None)) -> dict[str, np.ndarray]:
    """Per-frame median pixel error over all cameras: 'all' 512 points and each body part (NaN = unobserved)."""
    parts = parts or {}
    T = pts512.shape[0]
    errs = {k: [[] for _ in range(T)] for k in ("all", *parts)}
    for cam in cams:
        e, ok = _camera_errors(pts512, cam, gt_dir, lm_dir, p, sl)
        for f in range(T):
            errs["all"][f].extend(e[f][ok[f]].tolist())
            for k, idx in parts.items():
                errs[k][f].extend(e[f, idx][ok[f, idx]].tolist())
    return {k: np.array([np.median(x) if len(x) >= 3 else np.nan for x in v]) for k, v in errs.items()}


def _reprojection(pts512: np.ndarray | None, cams: list[str], gt_dir: Path, lm_dir: Path, p: int) -> np.ndarray:
    """Per-frame median pixel error of the fitted 512 points against the visible grid landmarks (all cameras)."""
    if pts512 is None:
        return np.full(0, np.nan)
    return reprojection_parts(pts512, cams, gt_dir, lm_dir, p)["all"]


def camera_residuals(smplx_dir: Path, gt_dir: Path, lm_dir: Path, cams: list[str], frame0: int = 0) -> dict:
    """Median px error per camera and role of an exported fit (smplx/<role>.npz joints are not enough: re-run FK
    from the saved params) against grid landmarks in gt_dir/lm_dir, which may hold cameras the fit never saw."""
    D = _load_verts512()
    out = {}
    for p, pid in enumerate(PIDS):
        r = np.load(smplx_dir / f"{ROLES[pid]}.npz")
        pose = np.concatenate([r["global_orient"], r["body_pose"], r["jaw_pose"], r["leye_pose"], r["reye_pose"],
                               r["left_hand_pose"], r["right_hand_pose"]], 1)
        verts, _ = _forward(pose, r["transl"], r["betas"], True)
        pts = np.matmul(D, verts)
        sl = slice(frame0, frame0 + len(pose))
        for cam in cams:
            e, ok = _camera_errors(pts, f"cam{cam}", gt_dir, lm_dir, p, sl)
            per_frame = np.array([np.median(e[f][ok[f]]) if ok[f].sum() >= 3 else np.nan for f in range(len(e))])
            out.setdefault(cam, {})[ROLES[pid]] = {
                "median_px": float(np.nanmedian(per_frame)) if np.isfinite(per_frame).any() else None,
                "p90_px": float(np.nanpercentile(per_frame, 90)) if np.isfinite(per_frame).any() else None,
                "frames": int(np.isfinite(per_frame).sum())}
    return out


def _frame_image(work: Path | None, g, f: int) -> str | None:
    """Image of grid frame f for QA: img_rel_path under this machine's work dir first, then img_abs_path."""
    for key in ("img_rel_path", "img_abs_path"):
        if key in g.files and len(g[key]) > f:
            p = Path(str(g[key][f]))
            if not p.is_absolute() and work is not None:
                p = work / p
            if p.exists():
                return str(p)
    return None


def qa_sheet(work: Path, seq: str, out_png: Path, frames: tuple[int, ...] = (0, 60, 120, 179),
             cams: list[str] | None = None, tag: str = "run", width: int = 360, held_out: tuple[str, ...] = (),
             title: str | None = None, frame0: int = 0) -> Path:
    """Contact sheet: grid landmarks (green = lead obs, cyan = follow obs) vs the fitted 512 points (red /
    magenta) reprojected with the per-frame cameras, over the nearest canonical frame when it exists.
    held_out cameras (not used by this fit) get an orange HELD OUT label. frames index the fit; frame0 = the grid
    frame of the fit's first frame (partial --fit-frames runs)."""
    import cv2
    root = work / "mamma"
    gt_dir, lm_dir, run_dir = root / "ma_cap" / seq / "gt", root / "ma_2d" / seq, root / "ma_3d" / tag
    run_dir = next(run_dir.glob("*/verts_joints_body_id-00.npz")).parent if not (run_dir / "verts_joints_body_id-00.npz").exists() else run_dir
    D = _load_verts512()
    fitted = [np.matmul(D, np.load(run_dir / f"verts_joints_body_id-{p:02d}.npz")["pred_vertices"])
              for p in range(len(PIDS))]
    names = cams or sorted(p.stem for p in gt_dir.glob("cam*.npz"))
    rows = []
    for cam in names:
        g, l = np.load(gt_dir / f"{cam}.npz"), np.load(lm_dir / f"{cam}.npz")
        E, K = np.asarray(g["cam_ext"], float), np.asarray(g["cam_int"], float)
        w, h = int(g["cam_img_w"]), int(g["cam_img_h"])
        tiles = []
        for fi in frames:
            fi = min(fi, len(fitted[0]) - 1)
            f = min(frame0 + fi, len(E) - 1)  # grid frame
            path = _frame_image(work, g, f)
            img = cv2.imread(path) if path else None
            img = img if img is not None else np.full((h, w, 3), 40, np.uint8)
            r = max(2, w // 300)
            for p, (c_obs, c_fit) in enumerate((((0, 255, 0), (0, 0, 255)), ((255, 255, 0), (255, 0, 255)))):
                obs, vis = l["landmarks"][f, p, :, :2], l["visibilities"][f, p]
                for (x, y), v in zip(obs, vis):
                    if v > 0.5:
                        cv2.circle(img, (int(x), int(y)), r + 1, c_obs, -1)
                px, z = project(fitted[p][fi][None], K[f][None], E[f, :3, :3][None], E[f, :3, 3][None])
                for (x, y), zz in zip(px[0], z[0]):
                    if zz > 0 and 0 <= x < w and 0 <= y < h:
                        cv2.circle(img, (int(x), int(y)), r, c_fit, -1)
            s = width / w
            tile = cv2.resize(img, (width, int(h * s)))
            label = f"{cam} f{f}" + (" HELD OUT" if cam.removeprefix("cam") in held_out else "")
            cv2.putText(tile, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 140, 255) if "HELD" in label else (255, 255, 255), 1)
            tiles.append(tile)
        hmax = max(t.shape[0] for t in tiles)
        rows.append(np.concatenate([np.pad(t, ((0, hmax - t.shape[0]), (0, 0), (0, 0))) for t in tiles], 1))
    wmax = max(r.shape[1] for r in rows)
    sheet = np.concatenate([np.pad(r, ((0, 0), (0, wmax - r.shape[1]), (0, 0))) for r in rows], 0)
    if title:
        bar = np.zeros((28, sheet.shape[1], 3), np.uint8)
        cv2.putText(bar, title, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        sheet = np.concatenate([bar, sheet], 0)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), sheet)
    return out_png


# ------------------------------------------------------------------------------------------- orchestration


def take_work(take_name: str) -> Path:
    """work/<take> without reading takes/<take>.toml (the pod has no source videos or alignment files)."""
    return PACKAGE_ROOT / "work" / take_name


def parse_sets(specs: list[str] | None, all_cams: list[str]) -> dict[str, list[str]]:
    """["c7=01,02,..", "c8=all"] -> {tag: cams}; default: one set "all" with every camera."""
    if not specs:
        return {"all": list(all_cams)}
    sets = {}
    for s in specs:
        tag, _, cams = s.partition("=")
        if not tag or not cams:
            raise SystemExit(f"[mamma] bad --set {s!r} (want tag=01,02,.. or tag=all or tag=-05)")
        if cams == "all":
            sets[tag] = list(all_cams)
        elif cams.startswith("-"):
            drop = set(cams[1:].split(","))
            sets[tag] = [c for c in all_cams if c not in drop]
        else:
            sets[tag] = cams.split(",")
    return sets


def prepare(take_name: str, sets: dict[str, list[str]] | None = None, force_2d: bool = False,
            skip_2d: bool = False, allow_sfm_frame: bool = False, use_sync: bool = True,
            window: tuple[float, float] | None = None, coverage: str = "all", grid_anchor: float | None = None,
            grid_k: tuple[int, int] | None = None) -> dict:
    """Local part: ma_2d (cached per camera), sync offsets, grid resampling -> everything ma_3d needs, per camera set,
    under work/<take>/mamma/ (+ prepare.json and pod_job.json describing the fit)."""
    from .take import Take
    take = Take.load(take_name)
    work = take.work
    all_cams = sorted(p.stem for p in (work / "cameras").glob("*.npz"))
    sets = sets or {"all": all_cams}
    union = sorted({c for cams in sets.values() for c in cams})
    from .sync import sync_required
    sync_req = sync_required(take)  # takes/<take>.toml [sync] required = true (audio clocks only are not enough)
    if sync_req and not use_sync:
        raise SystemExit(f"[mamma] {take.name}: --no-sync refused, takes/{take.name}.toml sets [sync] required = true "
                         "(per-phone video offsets of tens of ms remain on its audio clocks)")
    offsets = read_sync_offsets(work) if use_sync else {}  # refuses stale / unverifiable files
    tracks = load_tracks(work, union, allow_sfm_frame=allow_sfm_frame, offsets=offsets)
    world = (work / "cameras" / "world_from_sfm.json").exists()
    missing = [c for c in tracks if not any(mask_dir(work, c).glob("mask_*_01.png"))]
    if missing and not skip_2d:
        raise SystemExit(f"[mamma] no dancer masks for {missing} under dancers/<cam>/body (run the dancers stage)")
    native = work / "mamma" / "ma_2d_native"
    ma2d_status = {} if skip_2d else run_ma2d(work, union, native, force=force_2d)
    if sync_req and not offsets:  # ma_2d_native is what sync needs: stop before any fit input / pod job is written
        raise SystemExit(f"[mamma] {take.name}: sync/offsets.json is missing and takes/{take.name}.toml requires it "
                         f"([sync] required = true). ma_2d is ready: run 'python -m dancecap sync {take.name}', then "
                         "'mamma prepare' again")
    lo0, hi0 = window or take.window
    # every grid frame inside EVERY camera's (sync-corrected) coverage: with offsets of up to ~70 ms the union
    # left the first/last frames to 4-6 cameras (seen in the 3-frame dry fit)
    firsts = {c: float(tr.t[tr.ok][0]) for c, tr in tracks.items()}
    lasts = {c: float(tr.t[tr.ok][-1]) for c, tr in tracks.items()}
    if coverage == "union":
        # larissa_kadu_full's segment takes (dancecap.front_full): cameras start / stop inside the window; every grid
        # frame that ANY camera covers is kept and each camera contributes only where it has frames (build_inputs
        # zeroes a camera's landmarks outside its registered frames: resample_cameras ok = False there)
        lo, hi = max(lo0, min(firsts.values())), min(hi0, max(lasts.values()))
    elif coverage == "all":
        lo, hi = max(lo0, max(firsts.values())), min(hi0, min(lasts.values()))
    else:
        raise ValueError(f"coverage must be 'all' or 'union', not {coverage!r}")
    if grid_anchor is not None:  # grid on a fixed lattice t_k = anchor + k / GRID_HZ (stitching segment fits)
        k_lo = int(np.ceil((lo - grid_anchor) * GRID_HZ - 1e-6))
        k_hi = int(np.floor((hi - grid_anchor) * GRID_HZ + 1e-6))
        if grid_k is not None:
            k_lo, k_hi = max(k_lo, int(grid_k[0])), min(k_hi, int(grid_k[1]))
        ks = np.arange(k_lo, k_hi + 1)
        grid = grid_anchor + ks / GRID_HZ
    else:
        ks = None
        grid = master_grid(lo, hi)
    trim = {"window": [lo0, hi0], "start_limited_by": [c for c, v in firsts.items() if v >= lo - 1e-9 and v > lo0],
            "end_limited_by": [c for c, v in lasts.items() if v <= hi + 1e-9 and v < hi0], "coverage": coverage}
    if coverage == "union":
        trim["start_limited_by"] = [c for c, v in firsts.items() if abs(v - lo) < 1e-9 and v > lo0]
        trim["end_limited_by"] = [c for c, v in lasts.items() if abs(v - hi) < 1e-9 and v < hi0]
        trim["cams_per_frame"] = {c: [float(firsts[c]), float(lasts[c])] for c in sorted(firsts)}
    if ks is not None:
        trim["grid_anchor"] = float(grid_anchor)
        trim["grid_k"] = [int(ks[0]), int(ks[-1])]
    if trim["start_limited_by"] or trim["end_limited_by"]:
        print(f"[mamma] grid {grid[0]:.4f}-{grid[-1]:.4f} s is inside the window {lo0:.4f}-{hi0:.4f} s: trimmed to "
              f"every camera's sync-corrected coverage (start: {trim['start_limited_by']}, end: "
              f"{trim['end_limited_by']})")
    cfg = write_ma3d_config(work / "mamma" / "ma3d_config.yaml")
    # camera-type model of each track (cameras/<cam>.npz, dancecap.camera_types): a 'constant' (tripod) track resamples
    # to one cam_ext for every grid frame (no resampling noise) and is valid on every frame of the window
    cam_models = {}
    for c in sorted(tracks):
        z = np.load(work / "cameras" / f"{c}.npz")
        cam_models[c] = {"class": str(z["camera_class"]) if "camera_class" in z.files else "handheld_zoom",
                         "model": str(z["type_model"]) if "type_model" in z.files else "per_frame"}
    prep = {"take": take.name, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "world_frame": world,
            "camera_types": cam_models,
            "grid": {"t0": float(grid[0]), "T": int(len(grid)), "hz": GRID_HZ, "t": grid.tolist(), "trim": trim},
            "sync": {"file": "sync/offsets.json" if offsets else None, "required": sync_req,
                     "offsets_ms": {c: round(1000 * offsets[c]["delta"], 2) for c in sorted(offsets) if c in tracks},
                     "rates": {c: offsets[c]["rate"] for c in sorted(offsets) if offsets[c].get("rate")}},
            "ma_2d": ma2d_status, "config": cfg.name, "config_overrides": MA3D_OVERRIDES,
            "base_config": MA3D_CONFIG, "sets": {}}
    for tag, cams in sets.items():
        seq = f"{take.name}_{tag}"
        rep = build_inputs(work, {c: tracks[c] for c in cams}, native, grid, seq)
        prep["sets"][tag] = {"seq": seq, "cams": list(cams), "inputs": rep}
    # residuals of every fit are measured on the set with the most cameras (held-out cameras included)
    eval_tag = max(sets, key=lambda k: len(sets[k]))
    if set(sets[eval_tag]) != set(union):
        seq = f"{take.name}_eval"
        build_inputs(work, tracks, native, grid, seq)
        prep["eval"] = {"seq": seq, "cams": union}
    else:
        prep["eval"] = {"seq": prep["sets"][eval_tag]["seq"], "cams": sets[eval_tag]}
    (work / "mamma" / "prepare.json").write_text(json.dumps(prep, indent=1))
    write_pod_job(work, prep)
    return prep


def fit(take_name: str, tag: str, frames: tuple[int, int] | None = None) -> dict:
    """Pod part: ma_3d on a prepared camera set + export -> work/<take>/mamma/fits/<tag>[_f<a>-<b>]/ (never smplx/).

    frames=(a, b) fits only grid frames a..b-1 (dry check); such fits are never selectable."""
    work = take_work(take_name)
    root = work / "mamma"
    prep = json.loads((root / "prepare.json").read_text())
    if tag not in prep["sets"]:
        raise SystemExit(f"[mamma] set {tag!r} not prepared (have {sorted(prep['sets'])})")
    s = prep["sets"][tag]
    grid = np.asarray(prep["grid"]["t"], float)
    run_tag = tag if frames is None else f"{tag}_f{frames[0]}-{frames[1]}"
    run_dir, stats = run_ma3d(work, s["seq"], [f"cam{c}" for c in s["cams"]], root / prep["config"], tag=run_tag,
                              frames=frames)
    out = root / "fits" / run_tag
    if out.exists():
        shutil.rmtree(out)
    gt_dir, lm_dir = root / "ma_cap" / s["seq"] / "gt", root / "ma_2d" / s["seq"]
    f0 = frames[0] if frames else 0
    summary = export_smplx(run_dir, grid, gt_dir, lm_dir, out / "smplx", frame0=f0)
    ev = prep["eval"]
    resid = camera_residuals(out / "smplx", root / "ma_cap" / ev["seq"] / "gt", root / "ma_2d" / ev["seq"],
                             ev["cams"], frame0=f0)
    for cam in resid:
        resid[cam]["in_fit"] = cam in s["cams"]
    rep = {"tag": tag, "run_tag": run_tag, "cams": s["cams"], "frames": list(frames) if frames else None,
           "partial": frames is not None, "world_frame": prep["world_frame"], "ma_3d": stats, "smplx": summary,
           "residuals": resid, "eval_seq": ev["seq"], "host": "pod" if env.is_pod() else "local",
           "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (out / "fit.json").write_text(json.dumps(rep, indent=1, default=str))
    print(f"[mamma] fit {run_tag}: " + ", ".join(f"{r} median {v['median_reproj_px']:.1f} px" for r, v in summary.items())
          + f" -> {out}")
    return rep


def _shared_median(fit_rep: dict, cams: list[str], role: str) -> float:
    v = [fit_rep["residuals"][c][role]["median_px"] for c in cams
         if c in fit_rep["residuals"] and fit_rep["residuals"][c][role]["median_px"] is not None]
    return float(np.median(v)) if v else float("nan")


def choose_fit(fits: dict[str, dict], worse_rel: float = 0.10, worse_px: float = 0.5,
               held_out_factor: float = 3.0) -> tuple[str, list[str]]:
    """Pick among fits of nested camera sets. Prefer more cameras, unless the extra cameras (a) make the residuals of
    the cameras both fits share worse by > worse_rel AND > worse_px for either dancer, or (b) are not predicted by the
    smaller fit (held-out median > held_out_factor x the smaller fit's own in-fit median): then they contradict the
    rest of the rig and the smaller set wins. Returns (tag, reasons)."""
    order = sorted(fits, key=lambda k: -len(fits[k]["cams"]))
    best, reasons = order[0], []
    for small in order[1:]:
        big = best
        extra = sorted(set(fits[big]["cams"]) - set(fits[small]["cams"]))
        if set(fits[small]["cams"]) - set(fits[big]["cams"]) or not extra:
            continue  # not nested: keep the bigger set
        shared = sorted(set(fits[big]["cams"]) & set(fits[small]["cams"]))
        for role in ("lead", "follow"):
            b, s = _shared_median(fits[big], shared, role), _shared_median(fits[small], shared, role)
            if np.isfinite(b) and np.isfinite(s) and b > s * (1 + worse_rel) and b - s > worse_px:
                reasons.append(f"{big}: adding {extra} worsens {role} residuals on shared cameras "
                               f"{s:.2f} -> {b:.2f} px")
                best = small
                break
            held = _shared_median(fits[small], extra, role)
            if np.isfinite(held) and np.isfinite(s) and held > held_out_factor * s:
                reasons.append(f"{small}: held-out {extra} {role} residual {held:.2f} px > {held_out_factor:g} x "
                               f"in-fit {s:.2f} px -> {extra} inconsistent with the rest of the rig")
                best = small
                break
        else:
            reasons.append(f"{big} kept over {small}: shared-camera residuals not worse and held-out {extra} "
                           f"predicted by {small} (lead {_shared_median(fits[small], extra, 'lead'):.2f} / follow "
                           f"{_shared_median(fits[small], extra, 'follow'):.2f} px vs in-fit "
                           f"{_shared_median(fits[small], shared, 'lead'):.2f} / "
                           f"{_shared_median(fits[small], shared, 'follow'):.2f})")
    return best, reasons


def select(take_name: str, tags: list[str] | None = None, force_tag: str | None = None, qa: bool = True) -> dict:
    """Choose among finished full fits (fits/<tag>/fit.json) and promote it to smplx/ (world frame only)."""
    work = take_work(take_name)
    root = work / "mamma"
    prep = json.loads((root / "prepare.json").read_text())
    fits = {}
    for p in sorted((root / "fits").glob("*/fit.json")):
        r = json.loads(p.read_text())
        if r.get("partial") or (tags and r["tag"] not in tags):
            continue
        fits[r["tag"]] = r
    if not fits:
        raise SystemExit(f"[mamma] no finished full fits under {root / 'fits'}")
    tag, reasons = (force_tag, [f"forced: {force_tag}"]) if force_tag else choose_fit(fits)
    chosen = fits[tag]
    sel = {"chosen": tag, "reasons": reasons, "cams": chosen["cams"], "candidates": {
        k: {"cams": v["cams"], "residuals": v["residuals"], "smplx": v["smplx"]} for k, v in fits.items()},
        "sync": prep.get("sync"), "config_overrides": prep.get("config_overrides")}
    if not chosen.get("world_frame", True) or not prep.get("world_frame", True):
        sel["promoted"] = False
        sel["why_not"] = "fit is in the SfM frame (no world_from_sfm.json): never written to smplx/"
    else:
        dst = work / "smplx"
        dst.mkdir(exist_ok=True)
        for f in (root / "fits" / tag / "smplx").glob("*.npz"):
            shutil.copy2(f, dst / f.name)
        sel["promoted"] = True
    if qa:
        try:
            g = np.asarray(prep["grid"]["t"])
            fr = tuple(int(x) for x in np.linspace(0, len(g) - 1, 4).round())
            ev = prep["eval"]
            for k, v in fits.items():
                held = tuple(c for c in ev["cams"] if c not in v["cams"])
                out = work / "review" / f"pose_mamma_{k}.jpg"
                qa_sheet(work, ev["seq"], out, frames=fr, tag=k, held_out=held,
                         title=f"MAMMA fit {k} ({len(v['cams'])} cams{', held out ' + ','.join(held) if held else ''})"
                               f"{'  <- CHOSEN' if k == tag else ''}")
                sel.setdefault("review", []).append(str(out.relative_to(work)))
        except Exception as e:  # QA needs the canonical frames (absent on a pod)
            sel["review_error"] = f"{type(e).__name__}: {e}"
    (root / "select.json").write_text(json.dumps(sel, indent=1, default=str))
    report = {"grid": {k: prep["grid"][k] for k in ("t0", "T", "hz")}, "selected": tag, "reasons": reasons,
              "ma_3d": chosen["ma_3d"], "smplx": chosen["smplx"], "residuals": chosen["residuals"],
              "promoted": sel["promoted"], "sync": prep.get("sync")}
    (root / "report.json").write_text(json.dumps(report, indent=1, default=str))
    print(f"[mamma] selected {tag}: " + "; ".join(reasons))
    return sel


def run(take_name: str, cams: list[str] | None = None, skip_2d: bool = False, sets: list[str] | None = None,
        allow_sfm_frame: bool = False) -> dict:
    """Everything locally: prepare -> fit every set -> select."""
    if cams and not sets:
        sets = [f"run={','.join(cams)}"]
    from .take import Take
    all_cams = sorted(p.stem for p in (Take.load(take_name).work / "cameras").glob("*.npz"))
    prep = prepare(take_name, parse_sets(sets, all_cams), skip_2d=skip_2d, allow_sfm_frame=allow_sfm_frame)
    for tag in prep["sets"]:
        fit(take_name, tag)
    return select(take_name)


# ------------------------------------------------------------------------------------------- pod job


def _tree_files(p: Path) -> list[Path]:
    return [p] if p.is_file() else [f for f in p.rglob("*") if f.is_file() and "__pycache__" not in f.parts]


def _size(paths: list[Path]) -> int:
    return sum(f.stat().st_size for p in paths for f in _tree_files(p))


def pod_upload_items(work: Path, prep: dict) -> dict[str, list[tuple[Path, str]]]:
    """(local path, archive name relative to the pod workspace) per upload group."""
    ws = WORKSPACE
    take = prep["take"]
    rel = lambda p: p.relative_to(ws).as_posix()  # noqa: E731
    root = work / "mamma"
    inputs = [root / "prepare.json", root / prep["config"]]
    for s in [*prep["sets"].values(), prep["eval"]]:
        inputs += [root / "ma_2d" / s["seq"], root / "ma_cap" / s["seq"]]
    inputs = list(dict.fromkeys(inputs))
    data = [SMPLX_MODELS / "smplx" / "SMPLX_NEUTRAL.npz", VERTS_512, PARTS_512]
    code = [PACKAGE_ROOT / "dancecap"]
    return {
        "code_dancecap": [(p, rel(p)) for p in code],
        # ma_3d needs optimization/ (+ capture/ for one lazy import); ma_2d (landmarks/) never runs on the pod
        "code_mamma": [(MAMMA_ROOT / d, f"mamma-turing/{d}") for d in ("optimization", "capture")
                       if (MAMMA_ROOT / d).exists()],
        "body_models": [(p, rel(p) if p.is_relative_to(ws) else f"mamma/data/{p.relative_to(MAMMA_DATA).as_posix()}")
                        for p in data if p.exists()],
        "inputs": [(p, f"dancecap/work/{take}/mamma/{p.relative_to(root).as_posix()}") for p in inputs],
    }


def write_pod_job(work: Path, prep: dict) -> Path:
    """work/<take>/mamma/pod_job.json: exactly what a Runpod session must upload, run and download for the fit."""
    take = prep["take"]
    items = pod_upload_items(work, prep)
    groups = {k: {"items": [{"local": str(p), "pod": a, "bytes": _size([p])} for p, a in v],
                  "bytes": _size([p for p, _ in v])} for k, v in items.items()}
    tags = list(prep["sets"])
    pod_ws = "/workspace/HeadMovement"
    fit_cmds = [f"cd {pod_ws}/dancecap && DANCECAP_POD=1 python -m dancecap mamma fit {take} --tag {t}" for t in tags]
    job = {
        "what": "MAMMA ma_3d fits of the prepared camera sets + selection (pose-prep stage output)",
        "take": take, "created": prep["created"], "sets": {t: prep["sets"][t]["cams"] for t in tags},
        "pod_workspace": pod_ws,
        "layout": "the pod workspace mirrors the local one: <ws>/dancecap, <ws>/mamma-turing, <ws>/mamma/data/...",
        "gpu": "one 80 GB A100/H100 (ma_3d used 3.2 GB peak on the 2080 Ti; 180 frames x 2 people took 1042 s there)",
        "estimate_minutes": {"setup": 12, "fit_each": 10, "total_max": 50},
        "upload": {"bundle": f"dancecap/work/{take}/mamma/pod_upload.tar.gz (python -m dancecap mamma bundle {take})",
                   "extract": f"mkdir -p {pod_ws} && tar -xzf pod_upload.tar.gz -C {pod_ws}",
                   "groups": groups, "bytes_raw": sum(g["bytes"] for g in groups.values())},
        "setup": [
            "apt-get update -qq && apt-get install -y -qq libegl1 libgl1 libgles2 > /dev/null",
            "pip install -q 'numpy<2' smplx trimesh pyrender scenepic pandas plotly matplotlib opencv-python-headless "
            "scipy tqdm pyyaml",
            "pip install -q --no-build-isolation git+https://github.com/cuevhv/pytorch_sdf@torch2.5-cu124 "
            "|| echo 'pytorch_sdf build FAILED: fourth_run intersection term inactive (ma_3d prints a loud banner)'",
            "python -c \"import torch, smplx, pyrender; print('ok', torch.__version__, torch.cuda.get_device_name(0))\"",
        ],
        "env": {"DANCECAP_POD": "1", "PYOPENGL_PLATFORM": "egl"},
        "dry_check": f"cd {pod_ws}/dancecap && DANCECAP_POD=1 python -m dancecap mamma fit {take} --tag {tags[0]} "
                     "--fit-frames 90:93",
        "run": fit_cmds + [f"cd {pod_ws}/dancecap && DANCECAP_POD=1 python -m dancecap mamma select {take} --no-qa"],
        "cwd": f"{pod_ws}/dancecap",
        "download": [f"dancecap/work/{take}/mamma/fits/", f"dancecap/work/{take}/mamma/ma_3d/",
                     f"dancecap/work/{take}/mamma/logs/", f"dancecap/work/{take}/mamma/select.json",
                     f"dancecap/work/{take}/mamma/report.json"],
        "download_estimate_mb": 65 * len(tags),
        "after_download_locally": f"python -m dancecap mamma select {take}   # re-selects, promotes to smplx/, "
                                  f"writes review/pose_mamma_<tag>.jpg (needs the local frames)",
        "notes": [
            "Do NOT upload frames/, cameras/ caches or dancers/: the fit only reads ma_2d/<seq>, ma_cap/<seq>, "
            "prepare.json and ma3d_config.yaml.",
            "img paths inside ma_cap npz are local; ma_3d runs with --skip_scene_videos so it never opens them.",
            "SMPLX_NEUTRAL.npz (MAMMA locked head) and the downsampled vertices are licensed non-commercial: "
            "the pod is deleted after the session.",
        ],
    }
    p = work / "mamma" / "pod_job.json"
    p.write_text(json.dumps(job, indent=1))
    return p


def bundle(take_name: str) -> Path:
    """Build the pod upload tar.gz described by pod_job.json (code + body models + prepared inputs)."""
    from . import remote
    work = take_work(take_name)
    prep = json.loads((work / "mamma" / "prepare.json").read_text())
    items = [x for v in pod_upload_items(work, prep).values() for x in v]
    out = work / "mamma" / "pod_upload.tar.gz"
    remote.make_bundle(out, items, exclude=("__pycache__", ".pyc", "/scene_debug/out", "/.git"))
    job_p = work / "mamma" / "pod_job.json"
    job = json.loads(job_p.read_text())
    job["upload"]["bundle_bytes"] = out.stat().st_size
    job_p.write_text(json.dumps(job, indent=1))
    print(f"[mamma] pod bundle {out} {out.stat().st_size / 1e6:.1f} MB")
    return out


# ------------------------------------------------------------------------------------------- synthetic validation


def _look_at(C: np.ndarray, target: np.ndarray, roll: np.ndarray) -> np.ndarray:
    """World->camera rotations (OpenCV: +Z forward, +Y down) looking from C (T,3) at target (T,3)."""
    z = target - C
    z /= np.linalg.norm(z, axis=-1, keepdims=True)
    up = np.array([0.0, 1.0, 0.0])
    x = np.cross(z, up)  # right = forward x up with Y-up world and +Y-down image
    x /= np.linalg.norm(x, axis=-1, keepdims=True)
    y = np.cross(z, x)
    R = np.stack([x, y, z], 1)  # rows = camera axes in world
    c, s = np.cos(roll), np.sin(roll)
    Rz = np.zeros((len(C), 3, 3))
    Rz[:, 0, 0], Rz[:, 0, 1], Rz[:, 1, 0], Rz[:, 1, 1], Rz[:, 2, 2] = c, -s, s, c, 1
    return Rz @ R


def synthetic_rig(n_cams: int, t: np.ndarray, seed: int = 0) -> list[dict]:
    """Handheld phones on a 3.5-5 m ring around the couple: walking drift, bob, aim jitter, roll and zoom."""
    rng = np.random.default_rng(seed)
    rig = []
    for c in range(n_cams):
        az = 2 * np.pi * c / n_cams + rng.uniform(-0.2, 0.2)
        r = rng.uniform(3.5, 5.0)
        h = rng.uniform(1.2, 1.8)
        ph = rng.uniform(0, 2 * np.pi, 6)
        drift = 0.4 * np.sin(2 * np.pi * t / rng.uniform(4, 8) + ph[0])  # sideways walk (m)
        az_t = az + drift / r
        C = np.stack([r * np.sin(az_t), h + 0.03 * np.sin(2 * np.pi * 1.8 * t + ph[1]), r * np.cos(az_t)], 1)
        C[:, [0, 2]] *= (1 + 0.05 * np.sin(2 * np.pi * t / 5 + ph[2]))[:, None]  # step in/out
        target = np.stack([0.15 * np.sin(2 * np.pi * t / 3 + ph[3]), 0.9 + 0.08 * np.sin(2 * np.pi * t / 2.3 + ph[4]),
                           0.15 * np.cos(2 * np.pi * t / 3.7 + ph[5])], 1)
        roll = np.deg2rad(4.0) * np.sin(2 * np.pi * t / 2.9 + ph[0])
        portrait = c % 2 == 0
        w, hh = (1080, 1920) if portrait else (1920, 1080)
        f0 = rng.uniform(1300, 1700) if portrait else rng.uniform(1500, 1900)
        f = f0 * (1 + 0.25 * np.sin(2 * np.pi * t / rng.uniform(3, 6) + ph[2]))  # zoom +-25 %
        K = np.zeros((len(t), 3, 3))
        K[:, 0, 0] = K[:, 1, 1] = f
        K[:, 0, 2], K[:, 1, 2], K[:, 2, 2] = w / 2, hh / 2, 1.0
        R = _look_at(C, target, roll)
        rig.append({"K": K, "R": R, "t": -np.einsum("fij,fj->fi", R, C), "w": w, "h": hh,
                    "k1": -0.04 if c == 1 else 0.0})
    return rig


def validate_synthetic(noise_px: float = 0.0, n_cams: int = 8, frames: int = 180, seed: int = 0,
                       config: str = MA3D_CONFIG, tag: str | None = None, anatomical: bool = False) -> dict:
    """Ground-truth check of the whole adapter with moving, zooming, unsynchronised cameras.

    GT: the synthetic couple (dancecap.synthetic) evaluated at 120 Hz. Each camera samples it at its own
    native rate and phase (30 fps at 0-3/120 s offsets, one 59.94-ish 60 fps camera), observes MAMMA's 512
    downsampled vertices (optionally with Gaussian pixel noise and SIMPLE_RADIAL distortion on one camera),
    and the cameras are handed over in a scaled/rotated SfM frame with world_from_sfm.json. The adapter
    resamples everything to the 30 Hz grid, runs ma_3d and exports smplx/; MPJPE is measured against the GT
    joints on the grid.

    anatomical=True flips the sign of the follow's left-elbow flexion in the synthetic body pose: dancecap.synthetic
    bends it +0.9 rad about y, a 52 degree hyperextension (a left elbow flexes about -y; MAMMA initialises it at
    -pi/1.7), which ma_3d cannot reach from its init and which no dancer does.
    """
    from . import synthetic
    tag = tag or f"noise{noise_px:g}" + ("_anat" if anatomical else "")
    work = PACKAGE_ROOT / "work" / "synthetic_mamma" / tag
    gt_dir = PACKAGE_ROOT / "work" / "synthetic_mamma" / ("gt120_anat" if anatomical else "gt120")
    seconds = frames / GRID_HZ + 1.0
    if not (gt_dir / "smplx" / "lead.npz").exists():
        sys.path.insert(0, str(WORKSPACE / "atlas_bridge"))
        original = synthetic.body_pose
        if anatomical:
            def body_pose(t, role):
                pose = original(t, role)
                if role == "follow":
                    pose[:, (18 - 1) * 3 + 1] *= -1.0  # left elbow: hyperextension -> flexion
                return pose
            synthetic.body_pose = body_pose
        try:
            synthetic.make(gt_dir, seconds=seconds, fps=120.0)
        finally:
            synthetic.body_pose = original
    gt = {r: np.load(gt_dir / "smplx" / f"{r}.npz") for r in ("lead", "follow")}
    t120 = gt["lead"]["t"]
    verts120 = {r: _synthetic_verts(gt[r]) for r in gt}
    D = _load_verts512()
    pts120 = np.stack([np.matmul(D, verts120[r]) for r in ("lead", "follow")], 1)  # (F120, 2, 512, 3)
    grid = master_grid(0.5, 0.5 + (frames - 1) / GRID_HZ)
    if work.exists():
        shutil.rmtree(work)
    rng = np.random.default_rng(seed + 1)
    sim3 = {"s": 0.37, "R": _rot(rng), "t": rng.normal(0, 2, 3).tolist()}
    (work / "cameras").mkdir(parents=True)
    (work / "cameras" / "world_from_sfm.json").write_text(json.dumps(sim3))
    native = work / "mamma" / "ma_2d_native"
    native.mkdir(parents=True)
    steps = [(4, k % 4) for k in range(n_cams - 1)] + [(2, 1)]  # (120 Hz step, phase): 30 fps x7, 60 fps x1
    for c, (step, phase) in enumerate(steps):
        cam = f"{c + 1:02d}"
        idx = np.arange(phase, len(t120), step)
        idx = idx[(t120[idx] >= grid[0] - 0.1) & (t120[idx] <= grid[-1] + 0.1)]
        t_c = t120[idx]
        rig = synthetic_rig(n_cams, t_c, seed)[c]
        X = pts120[idx]  # (F, 2, 512, 3)
        px, z = project(X, rig["K"][:, None], rig["R"][:, None], rig["t"][:, None])
        if rig["k1"]:
            px = np.stack([distort_simple_radial(px[f], rig["K"][f], rig["k1"]) for f in range(len(idx))])
        px = px + rng.normal(0, noise_px, px.shape) if noise_px > 0 else px
        inside = (z > 0.1) & (px[..., 0] >= 0) & (px[..., 0] < rig["w"]) & (px[..., 1] >= 0) & (px[..., 1] < rig["h"])
        box = np.zeros(px.shape[:2])
        for f in range(len(idx)):
            for p in range(2):
                q = px[f, p][inside[f, p]]
                box[f, p] = 1.2 * (q.max(0) - q.min(0)).max() if len(q) > 2 else 512.0
        sigma = max(noise_px, 1.0)
        lv = sigma_px_to_logvar(np.full(px.shape[:3], sigma), box[..., None])
        lm = np.concatenate([px, lv[..., None]], -1) * inside[..., None]
        np.savez(native / f"{cam}.npz", landmarks=lm.astype(np.float32), visibilities=inside.astype(np.float32),
                 contacts=np.zeros(inside.shape, np.float32), floor_contacts=np.zeros(inside.shape, np.float32))
        np.save(native / f"{cam}_boxes.npy", box.astype(np.float32))
        R_s, t_s = sfm_cameras(rig["R"], rig["t"], sim3)
        np.savez(work / "cameras" / f"{cam}.npz", t=t_c, K=rig["K"], R=R_s, tvec=t_s,
                 dist=np.full(len(t_c), rig["k1"]), ok=np.ones(len(t_c), bool),
                 reproj_px=np.zeros(len(t_c)), n_inliers=np.full(len(t_c), 1000))
        (work / "frames" / cam).mkdir(parents=True)
        (work / "frames" / cam / "frames.json").write_text(json.dumps(
            {"camera": cam, "width": rig["w"], "height": rig["h"], "frame_count": len(t_c),
             "frames": [{"file": f"{i:06d}.jpg", "t": float(tt)} for i, tt in enumerate(t_c)]}))
    tracks = load_tracks(work)
    for tr in tracks.values():
        tr.files = []
    seq = "synthetic"
    report = {"noise_px": noise_px, "n_cams": n_cams, "T": int(len(grid)), "sim3": sim3,
              "native": {c: f"{1 / tr.frame_dt:.2f} fps, t0={tr.t[0]:.4f}" for c, tr in tracks.items()}}
    report["cameras"] = build_inputs(work, tracks, native, grid, seq)
    run_dir, report["ma_3d"] = run_ma3d(work, seq, [f"cam{c}" for c in tracks], config, tag="run")
    report["smplx"] = export_smplx(run_dir, grid, work / "mamma" / "ma_cap" / seq / "gt",
                                   work / "mamma" / "ma_2d" / seq, work / "smplx")
    report["mpjpe"] = synthetic_mpjpe(work, gt_dir)
    (work / "report.json").write_text(json.dumps(report, indent=1, default=str))
    return report


def synthetic_mpjpe(work: Path, gt_dir: Path) -> dict:
    """Joint errors of work/smplx/{lead,follow}.npz against the 120 Hz synthetic ground truth on the grid."""
    out = {}
    for role in ("lead", "follow"):
        gt = np.load(gt_dir / "smplx" / f"{role}.npz")
        res = np.load(work / "smplx" / f"{role}.npz")
        gi = np.searchsorted(gt["t"], res["t"] - 1e-6)
        assert np.allclose(gt["t"][gi], res["t"], atol=1e-6), "grid must hit the 120 Hz GT samples"
        pred, true = res["joints"], np.asarray(gt["joints"])[gi]
        e = np.linalg.norm(pred - true, axis=-1)
        out[role] = {"body22_mm": float(e[:, :22].mean() * 1000), "all55_mm": float(e.mean() * 1000),
                     "body22_median_mm": float(np.median(e[:, :22]) * 1000),
                     "body22_p95_mm": float(np.percentile(e[:, :22], 95) * 1000),
                     "pelvis_mm": float(e[:, 0].mean() * 1000),
                     "pa_body22_mm": float(_pa_mpjpe(pred[:, :22], true[:, :22]) * 1000),
                     "worst_frame_mm": float(e[:, :22].mean(1).max() * 1000),
                     "repaired_frames": int(res["repaired"].sum()) if "repaired" in res.files else 0}
    return out


def _synthetic_verts(npz) -> np.ndarray:
    """Vertices of a dancecap smplx npz (standard SMPL-X neutral, flat hands), the GT surface."""
    import smplx
    import torch
    T = len(npz["t"])
    model = smplx.create(str(WORKSPACE / "smplx" / "SMPLX_NEUTRAL.npz"), model_type="smplx", gender="neutral",
                         use_pca=False, flat_hand_mean=True, num_betas=10, batch_size=T)
    f = lambda k: torch.tensor(np.asarray(npz[k]), dtype=torch.float32)
    with torch.no_grad():
        out = model(betas=f("betas")[None].expand(T, -1), global_orient=f("global_orient"), body_pose=f("body_pose"),
                    left_hand_pose=f("left_hand_pose"), right_hand_pose=f("right_hand_pose"),
                    jaw_pose=f("jaw_pose"), transl=f("transl"))
    return out.vertices.numpy().astype(np.float64)


def _rot(rng) -> list:
    from scipy.spatial.transform import Rotation
    return Rotation.from_rotvec(rng.normal(0, 1.0, 3)).as_matrix().tolist()


def _pa_mpjpe(pred: np.ndarray, true: np.ndarray) -> float:
    """Mean per-frame Procrustes-aligned (similarity) joint error."""
    errs = []
    for P, Q in zip(pred, true):
        mp, mq = P.mean(0), Q.mean(0)
        A, B = P - mp, Q - mq
        U, S, Vt = np.linalg.svd(A.T @ B)
        d = np.sign(np.linalg.det(U @ Vt))
        Dm = np.diag([1, 1, d])
        Rm = (U @ Dm @ Vt).T
        s = (S * np.diag(Dm)).sum() / (A ** 2).sum()
        errs.append(np.linalg.norm(s * A @ Rm.T - B, axis=-1).mean())
    return float(np.mean(errs))
