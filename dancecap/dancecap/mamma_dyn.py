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
4. ma_3d runs from optimization/ with trusted ids (--trust-ids: re-ID only reports), dt = 1/30 and the up axis y,
   directly in the Y-up world frame (SMPL-X's canonical up is +Y, so MAMMA's upright init is right).
5. The result becomes smplx/{lead,follow}.npz (CONTRACTS.md) in the world frame.

MAMMA's uncertainty channel: ma_2d writes x, y (image px) and a log-variance in normalised crop units (the
crop of side box_px maps to [-1, 1]); ma_3d turns it into sigma_crop = sqrt(exp(lv)) / 2 * 512. We keep that
convention: sigma_img_px = sqrt(exp(lv)) / 2 * box_px, inflate in image pixels, and convert back.

    python -m dancecap mamma larissa_kadu                 # full run (needs dancers/ + cameras/)
    python -m dancecap mamma --validate-synthetic [--noise 2]   # ground-truth check with moving/zooming cameras

MAMMA is non-commercial and must not be redistributed: its code stays in the local mamma-turing worktree and
its outputs (SMPL-X params, meshes) stay in work/ (git-ignored).
"""
from __future__ import annotations

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

from .take import PACKAGE_ROOT, WORKSPACE

MAMMA_ROOT = WORKSPACE / "mamma-turing"  # local-only branch turing-dynamic-cams (never pushed)
MAMMA_DATA = WORKSPACE / "mamma" / "data"  # weights + body models (git-ignored, live in the main checkout)
PYTHON = WORKSPACE / ".venv-nopo4d" / "Scripts" / "python.exe"
SMPLX_MODELS = MAMMA_DATA / "body_models" / "smplx_locked_head"
VERTS_512 = MAMMA_DATA / "body_models" / "downsampled_verts" / "verts_512.pkl"
MA2D_WEIGHTS = MAMMA_DATA / "weights" / "ma_2d" / "mamma_mask_full_cvpr.ckpt"
MA2D_CONFIG = "configs/train/models_2d/config_mammanet_mask_512.yaml"  # relative to landmarks/
MA3D_CONFIG = "config_files/contact_configs/config_real_gmf_small_vals_detr_exp_no_vtemplate.yaml"
GRID_HZ = 30.0
N_LMK = 512
CROP_PX = 512  # MAMMA's sigma scale (losses.py: sigma / 2 * 512)
PIDS = (1, 2)  # 01 lead, 02 follow
ROLES = {1: "lead", 2: "follow"}
SIGMA_FLOOR_PX = 0.5


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
                       extra: dict[str, np.ndarray] | None = None, extra_scale: float = 1.0) -> dict[str, np.ndarray]:
    """Resample per-frame ma_2d output (F, P, N, 3) [x, y, logvar] onto grid times.

    A sample is valid when its coordinates are non-zero and vis > 0. Between two consecutive valid samples
    (gap <= 1.5 frame intervals) x, y and visibility are linearly interpolated and the image-pixel sigma is
    inflated by sigma_interp = |dx| * min(w, 1 - w) (zero on a frame, half the step at the midpoint). Otherwise
    the nearest valid sample is used if it lies within one frame interval, with sigma_interp = d * m (d =
    distance in frames, m = median per-frame landmark motion of that person in that camera); else the landmark
    is invisible (vis 0, coordinates 0). `extra` arrays (F, P, N), e.g. contacts, take the nearest sample.
    extra_scale multiplies sigma_interp before it is added in quadrature: ma_3d halves every sigma of a camera
    whose smallest crop sigma exceeds 15 px (losses.proj_pts_loss), which real ma_2d output always triggers, so
    build_inputs passes 2 there to keep the interpolation term at its intended pixel size.
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
    for p in range(P):
        m = np.median(step[:, p][both_valid_step[:, p]]) if both_valid_step[:, p].any() else 0.0
        v0, v1 = valid[i0, p], valid[i1, p]  # (T, N)
        ww = w[:, None]
        two = v0 & v1 & (gap[:, None] <= 1.5)
        x = (1 - ww)[..., None] * lm[i0, p, :, :2] + ww[..., None] * lm[i1, p, :, :2]
        s_lin = np.sqrt(((1 - ww) * sig[i0, p] + ww * sig[i1, p]) ** 2 +
                        (extra_scale * np.linalg.norm(lm[i1, p, :, :2] - lm[i0, p, :, :2], axis=-1)
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
        s_near = np.sqrt(sig[src, p, n_idx] ** 2 + (extra_scale * dn * m) ** 2)
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


def _sync_offsets(work: Path) -> dict[str, float]:
    p = work / "sync" / "offsets.json"
    if not p.exists():
        return {}
    raw = json.loads(p.read_text())
    raw = raw.get("offsets", raw)
    out = {}
    for cam, v in raw.items():
        if isinstance(v, dict):
            v = v.get("delta", v.get("delta_c", v.get("offset", 0.0)))
        try:
            out[str(cam)] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def load_tracks(work: Path, cams: list[str] | None = None) -> dict[str, CameraTrack]:
    """frames/<cam>/frames.json + cameras/<cam>.npz (+ world_from_sfm.json, sync/offsets.json) -> tracks."""
    sim3_p = work / "cameras" / "world_from_sfm.json"
    sim3 = json.loads(sim3_p.read_text()) if sim3_p.exists() else None
    if sim3 is None:
        print("[mamma] WARNING: cameras/world_from_sfm.json missing: cameras used in the SfM frame "
              "(results will NOT be metric or Y-up)")
    offsets = _sync_offsets(work)
    tracks = {}
    for cam_npz in sorted((work / "cameras").glob("*.npz")):
        cam = cam_npz.stem
        if cams and cam not in cams:
            continue
        meta = json.loads((work / "frames" / cam / "frames.json").read_text())
        c = np.load(cam_npz)
        t = np.asarray(c["t"], float) + offsets.get(cam, 0.0)
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


def _junction(link: Path, target: Path) -> None:
    """Directory junction (no admin rights needed on Windows); copy as a fallback."""
    if link.exists():
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        import _winapi
        _winapi.CreateJunction(str(target.resolve()), str(link))
    except (ImportError, OSError):
        shutil.copytree(target, link)


# ------------------------------------------------------------------------------------------- MAMMA runs


def _env() -> dict:
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    env["PYTHONPATH"] = os.pathsep.join([str(MAMMA_ROOT), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    return env


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


def _run_logged(cmd: list[str], cwd: Path, log: Path, label: str) -> dict:
    from . import gpu
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"[mamma] {label}: {' '.join(str(c) for c in cmd)}\n[mamma]   cwd={cwd} log={log}")
    with gpu.exclusive(label), _VramSampler() as vram, open(log, "w", encoding="utf-8") as fh:
        t0 = time.time()
        proc = subprocess.run([str(c) for c in cmd], cwd=cwd, env=_env(), stdout=fh, stderr=subprocess.STDOUT)
        elapsed = time.time() - t0
    stats = {"seconds": round(elapsed, 1), "gpu_used_peak_mib": vram.peak_mib, "gpu_used_before_mib": vram.base_mib,
             "returncode": proc.returncode}
    text = log.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        if line.startswith(("[mem]", "Re-ID:", "[sdf]", "[dt]", "[turing]", "Optimization took")):
            print(f"[mamma]   {line}")
    if proc.returncode != 0:
        tail = "\n".join(text.splitlines()[-30:])
        raise RuntimeError(f"{label} failed (exit {proc.returncode}); log {log}\n{tail}")
    return stats


def run_ma2d(work: Path, cams: list[str], out_dir: Path) -> None:
    """MAMMA ma_2d on the canonical frames with our body masks; writes out_dir/<cam>.npz (+ box sizes)."""
    stage = work / "mamma" / "ma2d_masks"
    todo = [c for c in cams if not (out_dir / f"{c}.npz").exists()]
    for cam in todo:
        _junction(stage / cam / "masks", mask_dir(work, cam))
    if todo:
        cmd = [PYTHON, "run_ma_2d.py", "--images_root_dir", work / "frames", "--cam_names", *todo,
               "--mask_path", stage, "--out_folder", out_dir, "--config_path", MA2D_CONFIG,
               "--weights", MA2D_WEIGHTS, "--downsampled-verts", VERTS_512, "--disable-visualizations"]
        _run_logged(cmd, MAMMA_ROOT / "landmarks", work / "mamma" / "logs" / "ma_2d.log", "mamma ma_2d")
    for cam in cams:
        box_p = out_dir / f"{cam}_boxes.npy"
        if not box_p.exists():
            n = np.load(out_dir / f"{cam}.npz")["landmarks"].shape[0]
            np.save(box_p, mask_boxes(work, cam, n))


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
        d2 = np.load(native_dir / f"{cam}.npz")
        lm = np.array(d2["landmarks"], np.float64)[:, :len(PIDS)]
        vis = np.array(d2["visibilities"], np.float64)[:, :len(PIDS)]
        if lm.shape[0] != len(tr.t):
            raise ValueError(f"{cam}: ma_2d has {lm.shape[0]} frames, camera track {len(tr.t)}")
        if lm.shape[1] < len(PIDS):  # ma_2d only exports up to the max observed id
            pad = len(PIDS) - lm.shape[1]
            lm = np.concatenate([lm, np.zeros((lm.shape[0], pad) + lm.shape[2:])], 1)
            vis = np.concatenate([vis, np.zeros((vis.shape[0], pad) + vis.shape[2:])], 1)
        box_p = native_dir / f"{cam}_boxes.npy"
        box = np.load(box_p) if box_p.exists() else _boxes_from_landmarks(lm, vis)
        # camera not registered on a frame -> its landmarks cannot be used
        vis[~tr.ok] = 0.0
        lm[~tr.ok] = 0.0
        observed = np.abs(lm[..., :2]).sum(-1) > 0  # zero coordinates = no detection; keep them zero
        for f in np.flatnonzero(tr.k1 != 0):  # pinhole model in MAMMA: undistort observations
            lm[f, ..., :2] = np.where(observed[f, ..., None],
                                      undistort_simple_radial(lm[f, ..., :2], tr.K[f], float(tr.k1[f])), 0.0)
        extra = {k: np.array(d2[k], np.float64)[:, :len(PIDS)] for k in ("contacts", "floor_contacts")
                 if k in d2.files and d2[k].dtype != object and d2[k].shape[:2] == lm.shape[:2]}
        ok2d = (vis > 0) & observed
        crop_sigma = np.sqrt(np.exp(lm[..., 2][ok2d])) / 2 * CROP_PX
        halved = crop_sigma.size > 0 and crop_sigma.min() > 15  # ma_3d will halve this camera's sigmas
        rs = resample_landmarks(tr.t, lm, vis, box, grid, tr.frame_dt, extra, extra_scale=2.0 if halved else 1.0)
        cg = resample_cameras(tr.t, tr.K, tr.R, tr.tvec, tr.ok, grid, tr.frame_dt)
        rs["landmarks"][~cg["ok"]] = 0.0
        rs["visibilities"][~cg["ok"]] = 0.0
        for k in extra:
            rs[k][~cg["ok"]] = 0.0
        name = f"cam{cam}"
        np.savez(lm_dir / f"{name}.npz", landmarks=rs["landmarks"].astype(np.float32),
                 visibilities=rs["visibilities"].astype(np.float32),
                 contacts=rs.get("contacts", np.zeros_like(rs["visibilities"])).astype(np.float32),
                 floor_contacts=rs.get("floor_contacts", np.zeros_like(rs["visibilities"])).astype(np.float32))
        nearest = np.clip(np.searchsorted(tr.t, grid), 0, len(tr.t) - 1)
        paths = [tr.files[i] for i in nearest] if tr.files else []
        np.savez(gt / f"{name}.npz", img_abs_path=np.array(paths), img_rel_path=np.array(paths),
                 cam_name=np.array(name), cam_int=cg["K"].astype(np.float32),
                 cam_ext=extrinsics(cg["R"], cg["t"]).astype(np.float32),
                 cam_img_w=int(tr.width), cam_img_h=int(tr.height), cam_portrait=bool(tr.height > tr.width),
                 frame_start=0, frame_end=len(grid), video_path=np.array(""),
                 distortion_model=np.array("radtan"), distortion_coeffs=np.zeros(4),
                 is_body_in_img=np.ones(len(grid), bool), t=grid)
        report[cam] = {"native_frames": int(len(tr.t)), "native_fps": round(1 / tr.frame_dt, 3),
                       "grid_cam_ok": float(cg["ok"].mean()),
                       "grid_visible_frac": {ROLES[pid]: float((rs["visibilities"][:, p] > 0).mean())
                                             for p, pid in enumerate(PIDS)}}
    np.savez(gt / "global.npz", params_paths=np.array([]), ioi_dir=str(work), seq_name=seq,
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


def run_ma3d(work: Path, seq: str, cam_names: list[str], config: str = MA3D_CONFIG, tag: str = "run",
             extra_args: list[str] | None = None) -> tuple[Path, dict]:
    root = work / "mamma"
    out = root / "ma_3d" / tag
    if out.exists():
        shutil.rmtree(out)
    cmd = [PYTHON, "run_ma_3d.py", "--seq_name", seq, "--ma_2d_dir", root / "ma_2d", "--ma_cap_dir", root / "ma_cap",
           "--config_file", config, "--cam_names", *cam_names, "--out_path", out,
           "--smplx-models", SMPLX_MODELS, "--downsampled-verts", VERTS_512,
           "--fps", str(GRID_HZ), "--trust-ids", "--up-axis=y", "--skip_detection_analysis", "--skip_scene_videos",
           *(extra_args or [])]
    stats = _run_logged(cmd, MAMMA_ROOT / "optimization", root / "logs" / f"ma_3d_{tag}.log", "mamma ma_3d")
    log = (root / "logs" / f"ma_3d_{tag}.log").read_text(encoding="utf-8", errors="replace")
    stats["sdf_active"] = "[sdf] pytorch_sdf loaded" in log
    stats["reid_lines"] = [l for l in log.splitlines() if l.startswith("Re-ID:")]
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


def export_smplx(run_dir: Path, grid: np.ndarray, gt_dir: Path, lm_dir: Path, out_dir: Path) -> dict:
    """ma_3d smplx_params/verts_joints -> smplx/{lead,follow}.npz (contract), plus per-frame reprojection quality."""
    hands = np.load(SMPLX_MODELS / "smplx" / "SMPLX_NEUTRAL.npz", allow_pickle=True)
    hm_l, hm_r = np.asarray(hands["hands_meanl"], float), np.asarray(hands["hands_meanr"], float)
    D = _load_verts512()
    cams = sorted(p.stem for p in gt_dir.glob("cam*.npz"))
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for p, pid in enumerate(PIDS):
        prm = np.load(run_dir / f"smplx_params_body_id-{p:02d}.npz", allow_pickle=True)
        vj = np.load(run_dir / f"verts_joints_body_id-{p:02d}.npz")
        pose = np.asarray(prm["smplx_pose"], float)  # (T,165): global, body, jaw, leye, reye, lhand, rhand
        verts = np.asarray(vj["pred_vertices"], float)
        joints = np.asarray(vj["pred_joints"], float)[:, :55]
        flat = bool(prm["smplx_export_flat_hand_mean"]) if "smplx_export_flat_hand_mean" in prm.files else False
        lh, rh = pose[:, 75:120], pose[:, 120:165]
        if not flat:  # contract hand poses are absolute (flat_hand_mean=True)
            lh, rh = lh + hm_l, rh + hm_r
        transl = np.asarray(prm["smplx_translation"], float)
        betas = np.asarray(prm["smplx_betas"], float).reshape(-1)
        reproj = _reprojection(np.matmul(D, verts), cams, gt_dir, lm_dir, p)
        bad = outlier_frames(reproj)
        reproj_raw = reproj.copy()
        if bad.any() and (~bad).sum() >= 2:
            # ma_3d occasionally throws a single frame metres away (LBFGS step + redescending Geman-McClure
            # loss + no temporal term in the second stage) and cannot pull it back. Re-fill those frames from
            # their good neighbours (slerp per joint, lerp translation), redo FK, and report them.
            pose, transl = _interpolate_frames(pose, transl, bad)
            verts, joints = _forward(pose, transl, betas, flat)
            lh, rh = pose[:, 75:120], pose[:, 120:165]
            if not flat:
                lh, rh = lh + hm_l, rh + hm_r
            reproj = _reprojection(np.matmul(D, verts), cams, gt_dir, lm_dir, p)
            print(f"[mamma] {ROLES[pid]}: repaired {int(bad.sum())} outlier frame(s) {np.flatnonzero(bad).tolist()} "
                  f"(reprojection {np.round(reproj_raw[bad], 1).tolist()} px -> {np.round(reproj[bad], 1).tolist()} px)")
        quality = np.exp(-np.nan_to_num(reproj, nan=1e3) / 10.0)
        quality[bad] = np.minimum(quality[bad], 0.1)  # interpolated, not measured
        contact = prm["smplx_contact"] if prm["smplx_contact"].dtype != object else np.zeros((len(grid), N_LMK))
        floor = prm["smplx_floor_contact"] if prm["smplx_floor_contact"].dtype != object else np.zeros((len(grid), N_LMK))
        role = ROLES[pid]
        np.savez_compressed(
            out_dir / f"{role}.npz", t=grid, global_orient=pose[:, :3], body_pose=pose[:, 3:66],
            jaw_pose=pose[:, 66:69], leye_pose=pose[:, 69:72], reye_pose=pose[:, 72:75],
            left_hand_pose=lh, right_hand_pose=rh, transl=transl, betas=betas, joints=joints,
            repaired=bad, reproj_px_raw=reproj_raw,
            contact=np.asarray(contact, np.float32), floor_contact=np.asarray(floor, np.float32),
            quality=quality, reproj_px=reproj, gender="neutral", model="smplx_locked_head (MAMMA)",
            flat_hand_mean=True, num_betas=int(np.asarray(prm["smplx_betas"]).shape[-1]), source="mamma ma_3d")
        summary[role] = {"frames": int(len(grid)), "median_reproj_px": float(np.nanmedian(reproj)),
                         "repaired_frames": np.flatnonzero(bad).tolist(),
                         "betas0": float(np.asarray(prm["smplx_betas"]).reshape(-1)[0])}
    return summary


def outlier_frames(reproj: np.ndarray, min_px: float = 25.0, factor: float = 8.0) -> np.ndarray:
    """Frames whose median reprojection error is far above the sequence's (or undefined)."""
    med = np.nanmedian(reproj) if np.isfinite(reproj).any() else 0.0
    return ~np.isfinite(reproj) | (reproj > max(min_px, factor * med))


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


def _reprojection(pts512: np.ndarray | None, cams: list[str], gt_dir: Path, lm_dir: Path, p: int) -> np.ndarray:
    """Per-frame median pixel error of the fitted 512 points against the visible grid landmarks (all cameras)."""
    if pts512 is None:
        return np.full(0, np.nan)
    errs = [[] for _ in range(pts512.shape[0])]
    for cam in cams:
        g, l = np.load(gt_dir / f"{cam}.npz"), np.load(lm_dir / f"{cam}.npz")
        E, K = np.asarray(g["cam_ext"], float), np.asarray(g["cam_int"], float)
        px, z = project(pts512, K, E[:, :3, :3], E[:, :3, 3])
        obs, vis = l["landmarks"][:, p, :, :2], l["visibilities"][:, p]
        ok = (vis > 0.5) & (np.abs(obs).sum(-1) > 0) & (z > 0)
        e = np.linalg.norm(px - obs, axis=-1)
        for f in range(len(errs)):
            errs[f].extend(e[f][ok[f]].tolist())
    return np.array([np.median(e) if e else np.nan for e in errs])


def qa_sheet(work: Path, seq: str, out_png: Path, frames: tuple[int, ...] = (0, 60, 120, 179),
             cams: list[str] | None = None, tag: str = "run", width: int = 360) -> Path:
    """Contact sheet: grid landmarks (green = lead obs, cyan = follow obs) vs the fitted 512 points (red /
    magenta) reprojected with the per-frame cameras, over the nearest canonical frame when it exists."""
    import cv2
    root = work / "mamma"
    gt_dir, lm_dir, run_dir = root / "ma_cap" / seq / "gt", root / "ma_2d" / seq, root / "ma_3d" / tag / seq
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
        for f in frames:
            f = min(f, len(E) - 1)
            img = None
            if "img_abs_path" in g.files and len(g["img_abs_path"]) > f and Path(str(g["img_abs_path"][f])).exists():
                img = cv2.imread(str(g["img_abs_path"][f]))
            img = img if img is not None else np.full((h, w, 3), 40, np.uint8)
            r = max(2, w // 300)
            for p, (c_obs, c_fit) in enumerate((((0, 255, 0), (0, 0, 255)), ((255, 255, 0), (255, 0, 255)))):
                obs, vis = l["landmarks"][f, p, :, :2], l["visibilities"][f, p]
                for (x, y), v in zip(obs, vis):
                    if v > 0.5:
                        cv2.circle(img, (int(x), int(y)), r + 1, c_obs, -1)
                px, z = project(fitted[p][f][None], K[f][None], E[f, :3, :3][None], E[f, :3, 3][None])
                for (x, y), zz in zip(px[0], z[0]):
                    if zz > 0 and 0 <= x < w and 0 <= y < h:
                        cv2.circle(img, (int(x), int(y)), r, c_fit, -1)
            s = width / w
            tile = cv2.resize(img, (width, int(h * s)))
            cv2.putText(tile, f"{cam} f{f}", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            tiles.append(tile)
        hmax = max(t.shape[0] for t in tiles)
        rows.append(np.concatenate([np.pad(t, ((0, hmax - t.shape[0]), (0, 0), (0, 0))) for t in tiles], 1))
    wmax = max(r.shape[1] for r in rows)
    sheet = np.concatenate([np.pad(r, ((0, 0), (0, wmax - r.shape[1]), (0, 0))) for r in rows], 0)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), sheet)
    return out_png


# ------------------------------------------------------------------------------------------- orchestration


def run(take_name: str, cams: list[str] | None = None, skip_2d: bool = False, config: str = MA3D_CONFIG,
        window: tuple[float, float] | None = None) -> dict:
    from .take import Take
    take = Take.load(take_name)
    work = take.work
    tracks = load_tracks(work, cams)
    if not tracks:
        raise SystemExit(f"[mamma] no cameras/<cam>.npz under {work / 'cameras'} (run the cameras stage first)")
    missing = [c for c in tracks if not any(mask_dir(work, c).glob("mask_*_01.png"))]
    if missing and not skip_2d:
        raise SystemExit(f"[mamma] no dancer masks for {missing} under dancers/<cam>/body (run the dancers stage)")
    native = work / "mamma" / "ma_2d_native"
    if not skip_2d:
        run_ma2d(work, list(tracks), native)
    lo, hi = window or take.window
    lo = max(lo, min(tr.t[tr.ok][0] for tr in tracks.values()))
    hi = min(hi, max(tr.t[tr.ok][-1] for tr in tracks.values()))
    grid = master_grid(lo, hi)
    seq = take.name
    report = {"grid": {"t0": float(grid[0]), "T": int(len(grid)), "hz": GRID_HZ}}
    report["cameras"] = build_inputs(work, tracks, native, grid, seq)
    run_dir, report["ma_3d"] = run_ma3d(work, seq, [f"cam{c}" for c in tracks], config)
    report["smplx"] = export_smplx(run_dir, grid, work / "mamma" / "ma_cap" / seq / "gt",
                                   work / "mamma" / "ma_2d" / seq, work / "smplx")
    (work / "mamma" / "report.json").write_text(json.dumps(report, indent=1))
    return report


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
