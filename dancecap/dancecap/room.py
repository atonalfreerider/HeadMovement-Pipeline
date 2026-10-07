"""Static studio reconstruction: masked 3D Gaussian Splatting (gsplat 1.5.3) -> full splat, Quest splat, Quest mesh.

    python -m dancecap room larissa_kadu --stages prep             # local: write the compact training bundle
    python -m dancecap room larissa_kadu --pod                     # Runpod: upload bundle, train/quest/mesh/eval, fetch
    python -m dancecap room larissa_kadu --preset smoke --out work/larissa_kadu/room/smoke   # local code-path check
    python -m dancecap room larissa_kadu --stages mesh,eval        # reuse room/cache/splat_full.pt

Stages: prep (local only: needs pycolmap + the frame store) -> train (GPU heavy: the pod) -> quest -> mesh -> eval
(local or pod). prep writes room/bundle/ (see write_bundle): undistorted images at <= bundle_max_side, people masks,
cameras, SfM points and the room geometry (floor / mirror planes, room box) - everything later stages read, so they
run anywhere (Windows or a Linux pod) from the bundle alone. Presets (PRESETS): smoke (a few hundred steps, tiny
budgets: code path only), local (2080 Ti), pod (80 GB A100/H100, full quality: native resolution, 30k steps).

Inputs (SfM frame, see CONTRACTS.md): the main COLMAP model cameras/sfm (keyframe poses + sparse points for the
initialisation), cameras/<cam>.npz (per window-frame K, R, t), keyframes/ + frames/ images, people_masks(_kf)/
(255 = any person or phone screen: excluded from the loss). Only the static-model cameras are used by default
(01 02 03 04 07 09); 05/06 are attached rotation-only (~4-5 deg off) and would double the walls. Cameras stay fixed
in the SfM frame (no world-space normalisation / re-centring as in gsplat's simple_trainer): the PLYs are directly in
the frame of cameras/sfm and room/transform.json maps them to the world.

Training (under dancecap.gpu.exclusive): gsplat rasterization in 'antialiased' mode, MCMC densification with a
Gaussian cap, fixed cameras, masked L1 + D-SSIM, a per-image 3x3 + bias colour transform (phones differ in white
balance and exposure; the mean transform is held at identity so the splat colours are the "average phone"), a random
background per step (the room is closed: nothing may stay see-through), and two PLANAR REFLECTORS:
  - the glossy black floor (SfM floor plane, RANSAC on the sparse points), reflectance f0 + g (1 - cos)^5;
  - the mirror wall (detect_mirror: the dominant high wall whose lower part has sparse points reconstructed behind
    it along lines of sight from the cameras - only a mirror can show that), constant reflectance.
The photo is modelled as diffuse render + vis * f * (render from the camera mirrored about the plane), Gaussians
behind a reflector are killed (MCMC relocates them), and a clamped planar depth prior pulls the rendered depth onto a
reflector wherever it is the first surface. Without this 3DGS builds an upside-down room under the floor and a
mirrored studio behind the mirror. Limits: reflected content that no camera sees directly (ceiling lights, the wall
behind the main cameras) cannot be explained by the mirrored render, so the light streaks on the floor and the mirror's
picture of the far wall stay painted onto those planes (blurred, view-dependent SH).

Outputs in work/<take>/room/ (all in the SfM frame; room/transform.json carries world_from_sfm):
  splat_full.ply      standard 3DGS PLY (SH degree 3, logit opacity, log scales, wxyz quaternions): DIFFUSE room
  splat_quest.ply     <= quest_max Gaussians ranked by rendered contribution (LightGaussian global significance =
                      sum over training pixels of alpha * transmittance), cropped to the room box, fine-tuned at
                      SH degree 1
  room_mesh.ply/.obj  floor = the floor plane as a 25 cm quad grid with its own orthographic texture
                      (room_floor_tex.png); everything above a 12 cm floor band = TSDF fusion of the splat's rendered
                      depth, decimated so the total is <= mesh_max_tris, texture baked from the splat renders onto an
                      xatlas atlas (room_mesh_tex.png); vertex colours in the PLY
  eval.json           held-out masked PSNR/SSIM per camera; review/room_*.jpg review sheets
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import gpu

MAIN_CAMS = ("01", "02", "03", "04", "07", "09")
STAGES = ("prep", "train", "quest", "mesh", "eval")
BUNDLE_VERSION = 1
# fields fixed when the bundle is written (changing them needs a new prep); everything else is a training setting
PREP_FIELDS = ("cams", "bundle_max_side", "win_step_s", "holdout_win_s", "holdout_kf", "mirror")


@dataclass
class RoomConfig:
    cams: tuple[str, ...] = MAIN_CAMS
    bundle_max_side: int = 1920     # prep: bundle images downscaled by an integer factor until max(W, H) <= this
                                    # (larissa_kadu: 480x864 phones stay native, cam 02 1080x1920 stays native)
    max_side: int = 1000            # train/eval: bundle images resized (area) so that max(W, H) <= max_side
    win_step_s: float = 0.1         # window frames: one training frame every win_step_s per camera
    holdout_win_s: float = 1.0      # held-out window frames: one per second, half a training step off the grid
    holdout_kf: int = 1             # held-out keyframes per camera (middle of the take's keyframe list)
    kf_weight: float = 0.5          # fraction of training draws that are keyframes (they span 30-68 s: more poses)
    iters: int = 30000
    cap_max: int = 1_000_000
    sh_degree: int = 3
    sh_every: int = 1000
    init_opacity: float = 0.5
    init_scale: float = 1.0         # x mean distance to the 3 nearest SfM points
    lr_means: float = 1.6e-4        # x scene scale, exponential decay to 1 %
    lr_scales: float = 5e-3
    lr_quats: float = 1e-3
    lr_opacities: float = 5e-2
    lr_sh0: float = 2.5e-3
    lr_shN: float = 2.5e-3 / 20
    lr_app: float = 1e-3
    app_reg: float = 1e-2           # pulls the MEAN colour transform to identity (per-image deviations are free)
    opacity_reg: float = 0.01       # MCMC regularisers (gsplat simple_trainer defaults)
    scale_reg: float = 0.01
    ssim_lambda: float = 0.2
    refine_start: int = 500         # MCMC relocation/growth window and period (gsplat defaults 500 / 25000 / 100)
    refine_every: int = 100
    refine_stop: int = 25000
    floor_margin: float = 0.08      # m (SfM units ~ m): Gaussians below floor - margin are killed (reflections)
    reflections: bool = True        # model the glossy floor + mirror wall as planar reflectors (else clamp only)
    mirror: str = "auto"            # auto | on | off: treat the dominant high wall's lower part as a mirror
    floor_beta: float = 0.15        # initial floor reflectance f0 (learned)
    mirror_beta: float = 0.8        # initial mirror reflectance (learned)
    lr_refl: float = 2e-3
    refl_start: int = 500           # reflections + plane depth prior join after the first coarse geometry
    random_bg: bool = True          # random background per step: forces an opaque (closed) room
    plane_depth_w: float = 0.1      # planar depth prior weight (floor / mirror are the first surface -> depth on them)
    plane_depth_clamp: float = 0.25 # relative depth error beyond which the prior stops pulling
    reflect_frac: float = 0.4       # mask a mirrored strip of this fraction of the person's height under each person
    min_track: int = 3              # SfM points used for the initialisation
    max_point_err: float = 2.0
    quest_max: int = 150_000
    quest_sh: int = 1
    quest_iters: int = 3000
    room_pad: float = 1.0           # m around the observed room box (Quest crop)
    mesh_voxel: float = 0.03
    mesh_max_tris: int = 80_000
    mesh_tex: int = 2048            # walls/objects atlas
    floor_cell: float = 0.25        # m, floor quad grid
    floor_band: float = 0.12        # m, TSDF geometry below this height is replaced by the floor plane
    floor_texel: float = 0.005      # m, target floor texel (capped by floor_tex_max)
    floor_tex_max: int = 2048
    log_every: int = 1000
    seed: int = 0
    note: str = field(default="", repr=False)


# quality presets (overrides of RoomConfig); prep fields are not part of a preset - one bundle serves all three
PRESETS: dict[str, dict] = {
    # code path only: a few hundred steps at ~480 px, tiny budgets (minutes on the 2080 Ti)
    "smoke": dict(max_side=480, iters=300, cap_max=60_000, sh_degree=1, sh_every=100, refine_start=100,
                  refine_every=50, refine_stop=250, refl_start=100, log_every=100, quest_max=20_000, quest_iters=100,
                  mesh_voxel=0.06, mesh_max_tris=20_000, mesh_tex=1024, floor_tex_max=1024),
    # the local 2080 Ti (11 GB): cam 02 at half resolution, ~1 h for 30k steps at the 1M cap
    "local": dict(max_side=1000),
    # 80 GB A100/H100 pod: full quality - every camera at native resolution, 30k steps, 1M Gaussians, finer mesh
    "pod": dict(max_side=1920, iters=30000, cap_max=1_000_000, quest_iters=5000, mesh_voxel=0.025),
}


def preset(name: str, **over) -> "RoomConfig":
    if name not in PRESETS:
        raise ValueError(f"room: unknown preset {name!r}; choose from {sorted(PRESETS)}")
    cfg = RoomConfig(**PRESETS[name])
    for k, v in over.items():
        if v is not None:
            setattr(cfg, k, v)
    cfg.note = f"preset {name}"
    return cfg


# ----------------------------------------------------------------------------------------------------- views

@dataclass
class View:
    name: str          # keyframes/<cam>/<ms>.jpg or frames/<cam>/<i>.jpg (relative to work)
    cam: str
    kind: str          # "kf" | "win"
    t: float
    K: np.ndarray      # OpenCV intrinsics (pixel centre at integer coordinates) at the LOADED (scaled) resolution
    R: np.ndarray
    tvec: np.ndarray
    W: int
    H: int
    k1: float          # radial distortion at native resolution (images are undistorted on load)
    scale: float       # native / loaded
    mask: str          # people mask path (relative to work)
    split: str = "train"
    src: str = ""      # bundle views: the original frame (name is then images/<i>.jpg relative to the bundle)

    @property
    def label(self) -> str:
        return self.src or self.name

    def to_json(self) -> dict:
        d = asdict(self)
        for k in ("K", "R", "tvec"):
            d[k] = np.asarray(d[k], float).tolist()
        return d

    @classmethod
    def from_json(cls, d: dict) -> "View":
        d = dict(d)
        for k in ("K", "R", "tvec"):
            d[k] = np.asarray(d[k], float)
        return cls(**d)

    def resized(self, max_side: int) -> "View":
        """Same view loaded at max(W, H) <= max_side (area resize on load; K follows the pixel-centre convention)."""
        if max(self.W, self.H) <= max_side:
            return self
        f = max(self.W, self.H) / max_side
        W2, H2 = max(1, round(self.W / f)), max(1, round(self.H / f))
        sx, sy = self.W / W2, self.H / H2
        K = self.K.copy()
        K[0, 0] /= sx
        K[1, 1] /= sy
        K[0, 2] = (K[0, 2] + 0.5) / sx - 0.5
        K[1, 2] = (K[1, 2] + 0.5) / sy - 0.5
        return View(self.name, self.cam, self.kind, self.t, K, self.R, self.tvec, W2, H2, self.k1,
                    self.scale * max(sx, sy), self.mask, self.split, self.src)

    @property
    def center(self) -> np.ndarray:
        return -self.R.T @ self.tvec

    def viewmat(self) -> np.ndarray:
        M = np.eye(4)
        M[:3, :3], M[:3, 3] = self.R, self.tvec
        return M

    def K_gs(self) -> np.ndarray:
        """gsplat / COLMAP convention: pixel centres at +0.5."""
        K = self.K.copy()
        K[0, 2] += 0.5
        K[1, 2] += 0.5
        return K


def _scale_K(K: np.ndarray, s: int) -> np.ndarray:
    K = K.astype(np.float64).copy()
    K[0, 0] /= s
    K[1, 1] /= s
    K[0, 2] = (K[0, 2] + 0.5) / s - 0.5
    K[1, 2] = (K[1, 2] + 0.5) / s - 0.5
    return K


def _factor(w: int, h: int, max_side: int) -> int:
    return max(1, math.ceil(max(w, h) / max_side))


def collect_views(work: Path, cfg: RoomConfig) -> list[View]:
    """Keyframes from the main SfM model + time-subsampled window frames from cameras/<cam>.npz."""
    import pycolmap
    rec = pycolmap.Reconstruction(str(work / "cameras" / "sfm"))
    views: list[View] = []
    kf_by_cam: dict[str, list[View]] = {}
    for im in rec.images.values():
        parts = im.name.split("/")
        if parts[0] != "keyframes" or parts[1] not in cfg.cams:
            continue
        cam = rec.cameras[im.camera_id]
        p = np.asarray(cam.params, float)
        f, cx, cy = p[0], p[1] - 0.5, p[2] - 0.5
        k1 = float(p[3]) if cam.model.name == "SIMPLE_RADIAL" else 0.0
        if cam.model.name not in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
            raise ValueError(f"room: unsupported camera model {cam.model.name} for {im.name}")
        T = im.cam_from_world()
        T = T.matrix() if hasattr(T, "matrix") else np.asarray(T)
        s = _factor(cam.width, cam.height, cfg.bundle_max_side)
        K = _scale_K(np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]]), s)
        ms = parts[2].split(".")[0]
        v = View(im.name, parts[1], "kf", int(ms) / 1000, K, T[:, :3].copy(), T[:, 3].copy(), cam.width // s,
                 cam.height // s, k1, s, f"people_masks_kf/{parts[1]}/{ms}.png")
        kf_by_cam.setdefault(parts[1], []).append(v)
    for cam, vs in kf_by_cam.items():
        vs.sort(key=lambda v: v.t)
        for j in range(cfg.holdout_kf):  # spread held-out keyframes over the take, away from the ends
            vs[int(round((j + 1) * len(vs) / (cfg.holdout_kf + 1)))].split = "test"
        views += vs
    for cam in cfg.cams:
        d = np.load(work / "cameras" / f"{cam}.npz")
        meta = json.loads((work / "frames" / cam / "frames.json").read_text())
        w, h = meta["width"], meta["height"]
        s = _factor(w, h, cfg.bundle_max_side)
        t = d["t"]
        ok = d["ok"].astype(bool)
        t0 = t[0]
        picks: dict[int, str] = {}
        for g in np.arange(t0, t[-1] + 1e-6, cfg.win_step_s):
            i = int(np.argmin(np.abs(t - g)))
            if ok[i] and abs(t[i] - g) < 0.5 / 30:
                picks[i] = "train"
        for g in np.arange(t0 + cfg.win_step_s / 2 + 0.5, t[-1], cfg.holdout_win_s):
            i = int(np.argmin(np.abs(t - g)))
            if ok[i] and i not in picks:
                picks[i] = "test"
        for i, split in sorted(picks.items()):
            views.append(View(f"frames/{cam}/{i:06d}.jpg", cam, "win", float(t[i]), _scale_K(d["K"][i], s),
                              d["R"][i].copy(), d["tvec"][i].copy(), w // s, h // s, float(d["dist"][i]), s,
                              f"people_masks/{cam}/{i:06d}.png", split))
    return views


def reflect_mask(ignore: np.ndarray, frac: float, min_h_frac: float = 0.15) -> np.ndarray:
    """Add the glossy-floor reflection under every tall ignore component (standing people, mostly the dancers):
    per image column, a strip of `frac` x the component's height in that column below its lowest pixel. Seated
    spectators (components shorter than min_h_frac of the image) are left alone so the floor around them survives."""
    if frac <= 0:
        return ignore
    out = ignore.copy()
    H = ignore.shape[0]
    n, lab, st, _ = cv2.connectedComponentsWithStats(ignore.astype(np.uint8), connectivity=8)
    rows = np.arange(H)[:, None]
    for k in range(1, n):
        x, y, w, h, _a = st[k]
        if h < min_h_frac * H:
            continue
        comp = lab[y:y + h, x:x + w] == k
        has = comp.any(0)
        bottom = y + (h - 1 - np.argmax(comp[::-1], axis=0))     # lowest pixel per column
        top = y + np.argmax(comp, axis=0)
        ext = np.round(frac * (bottom - top + 1)).astype(int)
        strip = (rows > bottom[None]) & (rows <= (bottom + ext)[None]) & has[None]
        out[:, x:x + w] |= strip
    return out


def load_raw(root: Path, v: View) -> tuple[np.ndarray, np.ndarray]:
    """(H, W, 3) uint8 BGR undistorted + resized to (v.H, v.W), (H, W) uint8 people mask (255 = ignore).
    root = the take's work dir (frame-store views) or the bundle dir (bundle views: already undistorted)."""
    img = cv2.imread(str(root / v.name), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(root / v.name)
    mp = root / v.mask
    m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) if mp.exists() else np.zeros(img.shape[:2], np.uint8)
    if m.shape != img.shape[:2]:
        m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    if v.k1:
        Kn = _unscale_K(v.K, v.scale)
        mx, my = cv2.initUndistortRectifyMap(Kn, np.array([v.k1, 0, 0, 0, 0.0]), None, Kn,
                                             (img.shape[1], img.shape[0]), cv2.CV_32FC1)
        img = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        m = cv2.remap(m, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    if img.shape[:2] != (v.H, v.W):
        img = cv2.resize(img, (v.W, v.H), interpolation=cv2.INTER_AREA)
        m = cv2.resize(m, (v.W, v.H), interpolation=cv2.INTER_AREA)
    return img, m


def load_view(work: Path, v: View, cfg: RoomConfig) -> tuple[np.ndarray, np.ndarray]:
    """(H, W, 3) uint8 RGB undistorted + scaled, (H, W) bool valid (= not a person / reflection / screen)."""
    img, m = load_raw(work, v)
    ignore = reflect_mask(m > 64, cfg.reflect_frac)
    return img[:, :, ::-1].copy(), ~ignore


def _unscale_K(K: np.ndarray, s: int) -> np.ndarray:
    K = K.copy()
    K[0, 0] *= s
    K[1, 1] *= s
    K[0, 2] = (K[0, 2] + 0.5) * s - 0.5
    K[1, 2] = (K[1, 2] + 0.5) * s - 0.5
    return K


# ----------------------------------------------------------------------------------------------- geometry

def fit_floor(P: np.ndarray, up: np.ndarray, near: np.ndarray, radius: float = 10.0, thr: float = 0.02,
              iters: int = 3000, seed: int = 0) -> tuple[np.ndarray, float, int]:
    """RANSAC floor plane n.X + d = 0 (n along `up`) from points within `radius` of `near` and below their median
    height + 0.5 (excludes the ceiling, keeps a tilted floor). Returns (n, d, inliers)."""
    up = up / np.linalg.norm(up)
    h = P @ up
    sel = (np.linalg.norm(P - near, axis=1) < radius) & (h < np.median(h) + 0.5)
    Q = P[sel]
    if len(Q) < 30:
        raise ValueError("room: too few points near the dancers to fit the floor")
    rng = np.random.default_rng(seed)
    best = (0, up, -float(np.median(Q @ up)))
    for _ in range(iters):
        a, b, c = Q[rng.choice(len(Q), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-12:
            continue
        n /= np.linalg.norm(n)
        if n @ up < 0:
            n = -n
        if n @ up < math.cos(math.radians(15)):
            continue
        d = -n @ a
        k = int((np.abs(Q @ n + d) < thr).sum())
        if k > best[0]:
            best = (k, n, d)
    k, n, d = best
    inl = np.abs(Q @ n + d) < thr
    # least-squares refit on the inliers
    c = Q[inl].mean(0)
    _, _, vt = np.linalg.svd(Q[inl] - c)
    n = vt[2] * np.sign(vt[2] @ up)
    return n, float(-n @ c), int(inl.sum())


def load_world(work: Path) -> dict:
    w = json.loads((work / "cameras" / "world_from_sfm.json").read_text())
    return {"s": float(w["s"]), "R": np.array(w["R"], float), "t": np.array(w["t"], float)}


def sfm_points(work: Path):
    """(P, C, E, L) sparse points: from a bundle's points.npz, else the take's cameras/sfm (pycolmap)."""
    if (work / "points.npz").exists():
        d = np.load(work / "points.npz")
        return d["P"].astype(float), d["C"], d["E"].astype(float), d["L"]
    import pycolmap
    rec = pycolmap.Reconstruction(str(work / "cameras" / "sfm"))
    pts = list(rec.points3D.values())
    P = np.array([p.xyz for p in pts], float)
    C = np.array([p.color for p in pts], np.uint8)
    E = np.array([p.error for p in pts], float)
    L = np.array([p.track.length() for p in pts], int)
    return P, C, E, L


def detect_mirror(P: np.ndarray, w: dict, cams_w: np.ndarray, high_m: float = 2.8, thr: float = 0.05,
                  iters: int = 3000, seed: int = 0) -> dict | None:
    """The dominant vertical wall among the high sparse points (world height > high_m: banners, soffits) and the
    evidence that its lower part is a MIRROR: points reconstructed BEHIND the wall plane (seen from
    the cameras' side) can only be reflections: count the sparse points behind the wall whose line of sight from the
    cameras crosses the wall below its high (soffit) part. Returns the wall in world (n_w . X + d_w > 0 on the cameras' side),
    its extent and the evidence, or None when no wall is found."""
    Xw = (w["s"] * (w["R"] @ P.T)).T + w["t"]
    Q = Xw[Xw[:, 1] > high_m][:, [0, 2]]
    if len(Q) < 50:
        return None
    rng = np.random.default_rng(seed)
    best = (0, None, 0.0)
    for _ in range(iters):
        a, b = Q[rng.choice(len(Q), 2, replace=False)]
        dv = b - a
        if np.linalg.norm(dv) < 1.0:
            continue
        n = np.array([-dv[1], dv[0]]) / np.linalg.norm(dv)
        k = int((np.abs(Q @ n - n @ a) < thr).sum())
        if k > best[0]:
            best = (k, n, -float(n @ a))
    k, n2, d = best
    if n2 is None:
        return None
    if (cams_w[:, [0, 2]] @ n2 + d).mean() < 0:
        n2, d = -n2, -d
    sd = Xw[:, [0, 2]] @ n2 + d
    along = np.array([-n2[1], n2[0]])
    inl = (np.abs(sd) < thr) & (Xw[:, 1] > high_m)
    a_lo, a_hi = np.percentile(Xw[inl][:, [0, 2]] @ along, [1, 99])
    on_wall = (np.abs(sd) < thr) & (Xw[:, 1] > 1.5)
    top = float(np.clip(np.percentile(Xw[on_wall][:, 1], 1), 1.8, 3.0)) if on_wall.sum() > 20 else 2.6
    # points behind the wall whose line of sight from the cameras' centroid crosses the wall BELOW the soffit
    c = cams_w.mean(0)
    sc = c[[0, 2]] @ n2 + d
    lam = sc / np.where(np.abs(sc - sd) < 1e-9, 1e-9, sc - sd)
    Y = c + lam[:, None] * (Xw - c)
    aY = Y[:, [0, 2]] @ along
    through = (sd < -0.3) & (aY > a_lo) & (aY < a_hi) & (Y[:, 1] > 0) & (Y[:, 1] < top)
    behind = int(through.sum())
    return {"n_w": np.array([n2[0], 0.0, n2[1]]), "d_w": float(d), "inliers": k,
            "along_w": np.array([along[0], 0.0, along[1]]), "along_lo": float(a_lo), "along_hi": float(a_hi),
            "top_m": top, "seen_through_below_soffit": behind,
            "is_mirror": bool(behind > max(100, 0.03 * len(P))),
            "cams_dist_m": (cams_w[:, [0, 2]] @ n2 + d).round(2).tolist()}


def plane_to_sfm(n_w: np.ndarray, d_w: float, w: dict) -> tuple[np.ndarray, float]:
    """World plane n_w . X_w + d_w = 0 -> SfM plane n . X + d = 0 (unit n; distances in SfM units)."""
    n = w["R"].T @ n_w
    return n / np.linalg.norm(n), float((n_w @ w["t"] + d_w) / (w["s"] * np.linalg.norm(n)))


def room_geometry(work: Path, views: list[View], cfg: "RoomConfig | None" = None) -> dict:
    """Floor plane, mirror wall (planar reflectors) and room box in the SfM frame."""
    cfg = cfg or RoomConfig()
    P, C, E, L = sfm_points(work)
    w = load_world(work)
    up = w["R"][1]  # world Y in SfM coordinates (world = s R X + t)
    X0 = w["R"].T @ (np.zeros(3) - w["t"]) / w["s"]  # world origin (dancers' mean floor position) in SfM
    n, d, k = fit_floor(P, up, X0)
    centers = np.array([v.center for v in views])
    cams_w = (w["s"] * (w["R"] @ centers.T)).T + w["t"]
    good = (L >= 3) & (E < 2.0) & (P @ n + d > -0.1)
    reflectors = [{"name": "floor", "n": n, "d": float(d), "rect": None, "schlick": True,
                   "beta_init": cfg.floor_beta, "front_margin": 0.03}]
    mirror = detect_mirror(P, w, cams_w) if cfg.mirror != "off" else None
    if mirror is not None and (mirror["is_mirror"] or cfg.mirror == "on"):
        mn, md = plane_to_sfm(mirror["n_w"], mirror["d_w"], w)
        ax = w["R"].T @ mirror["along_w"]
        off = float(mirror["along_w"] @ w["t"])
        mirror.update({"n": mn, "d": md, "along": ax, "c": centers.mean(0),
                       "a_lo": (mirror["along_lo"] - off) / w["s"],
                       "a_hi": (mirror["along_hi"] - off) / w["s"]})
        reflectors.append({"name": "mirror", "n": mn, "d": md, "schlick": False, "beta_init": cfg.mirror_beta,
                           "front_margin": 0.05,
                           "rect": {"along": ax, "a_lo": mirror["a_lo"], "a_hi": mirror["a_hi"],
                                    "floor_n": n, "floor_d": float(d),
                                    "h_lo": -0.1, "h_hi": (mirror["top_m"] - 0.05) / w["s"]}})
    elif mirror is not None:
        mirror = {k2: v for k2, v in mirror.items()}  # reported only
    geo = {"floor_n": n, "floor_d": d, "mirror": mirror, "world": w}
    import torch
    good &= ~behind_reflectors(torch.as_tensor(P, dtype=torch.float32), geo, 0.05).numpy()
    lo, hi = np.percentile(P[good], 1, axis=0), np.percentile(P[good], 99, axis=0)
    lo, hi = np.minimum(lo, centers.min(0)), np.maximum(hi, centers.max(0))
    return {"floor_n": n, "floor_d": d, "floor_inliers": k, "up": up / np.linalg.norm(up),
            "tilt_vs_world_deg": float(np.degrees(np.arccos(np.clip(n @ up / np.linalg.norm(up), -1, 1)))),
            "box_lo": lo, "box_hi": hi, "origin": X0, "world": w, "mirror": mirror,
            "reflectors": reflectors,
            "scene_scale": float(1.1 * np.linalg.norm(centers - centers.mean(0), axis=1).max())}


def behind_mirror(means, geo: dict, margin: float):
    """Gaussians behind the mirror wall where the line of sight from the cameras' centroid crosses the mirror (below
    its top, within its extent): reflections, never real geometry. Space seen past the ends of the wall or above the
    soffit is kept."""
    import torch
    m = geo.get("mirror")
    if m is None or "n" not in m:
        return torch.zeros(len(means), dtype=torch.bool, device=means.device)
    t = lambda x: torch.as_tensor(np.asarray(x, np.float32), device=means.device)
    n, d, c = t(m["n"]), float(m["d"]), t(m["c"])
    sd = means @ n + d
    sc = float(c @ n + d)
    lam = sc / (sc - sd).clamp_min(1e-6)
    Y = c + lam[:, None] * (means - c)
    a = Y @ t(m["along"])
    h = Y @ t(geo["floor_n"]) + float(geo["floor_d"])
    return (sd < -2 * margin) & (a > m["a_lo"] - 0.3) & (a < m["a_hi"] + 0.3) & (h > -0.5) &         (h < m["top_m"] / geo["world"]["s"] + 0.3)


def behind_reflectors(means, geo: dict, margin: float):
    """Gaussians under the floor plane (by more than margin) or behind the mirror (see behind_mirror)."""
    import torch
    nf = torch.as_tensor(np.asarray(geo["floor_n"], np.float32), device=means.device)
    return (means @ nf + float(geo["floor_d"]) < -margin) | behind_mirror(means, geo, margin)


class Reflectors:
    """Planar reflectors (glossy floor, mirror wall): photo = direct render + vis * f * render of the scene mirrored
    about the plane (= render from the mirrored camera, a det -1 view matrix; Gaussians behind the plane excluded).
    vis = the plane is the first surface on the pixel ray (from the direct depth, no gradient); f = learned
    reflectance (floor: f0 + g (1 - cos)^5, Schlick-like Fresnel; mirror: constant f0)."""

    def __init__(self, planes: list[dict], device="cuda", enabled: bool = True):
        import torch
        init = [[math.log(p["beta_init"] / (1 - p["beta_init"])), -2.0] for p in planes]
        self.planes = planes
        self.dev = device
        self.enabled = enabled  # False: planes only feed the depth prior / visibility, no reflected render
        self.logit = torch.nn.Parameter(torch.tensor(init, dtype=torch.float32, device=device).reshape(-1, 2))

    def betas(self) -> dict:
        import torch
        b = torch.sigmoid(self.logit).detach().cpu().numpy()
        return {p["name"]: {"f0": float(b[i, 0]), "fresnel_gain": float(b[i, 1]) if p["schlick"] else 0.0}
                for i, p in enumerate(self.planes)}

    def render(self, splat, vm, K, W: int, H: int, sh_degree=None, reflect: bool = True, min_vis: float = 0.005,
               rasterize_mode: str = "antialiased", bg=None):
        """bg: (3,) background colour (training uses a random one so nothing can stay transparent: the room is
        closed, a see-through floor plus reflection would otherwise mimic a dark glossy floor)."""
        import torch
        bgs = None if bg is None else bg[None]
        out, alpha, info = splat.render(vm, K, W, H, sh_degree=sh_degree, mode="RGB+ED",
                                        rasterize_mode=rasterize_mode,
                                        backgrounds=bgs)
        rgb = out[0, ..., :3]
        depth = out[0, ..., 3]
        res = {"direct": rgb, "depth": depth, "alpha": alpha[0, ..., 0], "info": info, "vis": {}, "plane_z": {}}
        reflect = reflect and self.enabled
        res["rgb"] = rgb
        if not self.planes:
            return res
        f = lambda x: torch.as_tensor(np.asarray(x, np.float32), device=self.dev)
        R, t = vm[:3, :3], vm[:3, 3]
        ys, xs = torch.meshgrid(torch.arange(H, device=self.dev, dtype=torch.float32) + 0.5,
                                torch.arange(W, device=self.dev, dtype=torch.float32) + 0.5, indexing="ij")
        r = torch.stack([(xs - K[0, 2]) / K[0, 0], (ys - K[1, 2]) / K[1, 1], torch.ones_like(xs)], -1)
        ru = r / r.norm(dim=-1, keepdim=True)
        d_eff = torch.where(res["alpha"] > 0.5, depth, torch.full_like(depth, 1e6)).detach()
        total = rgb
        means = splat.params["means"].detach()
        for i, p in enumerate(self.planes):
            n = f(p["n"])
            nc = R @ n
            dc = float(p["d"]) - nc @ t
            den = r @ nc
            z = -dc / torch.where(den.abs() < 1e-6, torch.full_like(den, 1e-6), den)
            vis = (z > 0.05) & (z < d_eff * 1.15 + 0.1)
            rc = p["rect"]
            if rc is not None:
                X = (z[..., None] * r - t) @ R
                a = X @ f(rc["along"])
                h = X @ f(rc["floor_n"]) + float(rc["floor_d"])
                vis &= (a > rc["a_lo"]) & (a < rc["a_hi"]) & (h > rc["h_lo"]) & (h < rc["h_hi"])
            res["vis"][p["name"]] = vis
            res["plane_z"][p["name"]] = z
            if not reflect or vis.float().mean() < min_vis:
                continue
            Hm = torch.eye(3, device=self.dev) - 2 * torch.outer(n, n)
            vr = vm.clone()
            vr[:3, :3] = R @ Hm
            vr[:3, 3] = R @ (-2 * float(p["d"]) * n) + t
            front = means @ n + float(p["d"]) > p["front_margin"]
            o2, _, _ = splat.render(vr, K, W, H, sh_degree=sh_degree, mask=front, rasterize_mode=rasterize_mode,
                                    backgrounds=bgs)
            b = torch.sigmoid(self.logit[i])
            fr = b[0] + b[1] * (1 - (ru @ nc).abs()) ** 5 if p["schlick"] else b[0].expand(H, W)
            total = total + (vis.float() * fr)[..., None] * o2[0]
            res[f"refl_{p['name']}"] = o2[0]
        res["rgb"] = total
        return res


# ---------------------------------------------------------------------------------------------------- PLY io

def save_ply(path: Path, means, quats, log_scales, logit_opac, sh0, shN) -> None:
    """Standard 3DGS PLY (Kerbl et al.): f_rest channel-major, wxyz rotation, raw (logit / log) activations."""
    from plyfile import PlyData, PlyElement
    N = len(means)
    shN = np.asarray(shN, np.float32).reshape(N, -1, 3)
    rest = shN.transpose(0, 2, 1).reshape(N, -1)
    cols = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"] + [f"f_rest_{i}" for i in
                                                                               range(rest.shape[1])] + \
           ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    data = np.concatenate([means, np.zeros((N, 3)), np.asarray(sh0).reshape(N, 3), rest,
                           np.asarray(logit_opac).reshape(N, 1), log_scales, quats], axis=1).astype(np.float32)
    arr = np.empty(N, dtype=[(c, "f4") for c in cols])
    for i, c in enumerate(cols):
        arr[c] = data[:, i]
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(arr, "vertex")], byte_order="<").write(str(path))


def load_ply(path: Path) -> dict[str, np.ndarray]:
    from plyfile import PlyData
    v = PlyData.read(str(path))["vertex"]
    names = [p.name for p in v.properties]
    g = lambda *k: np.stack([np.asarray(v[x], np.float32) for x in k], 1)
    rest = sorted([n for n in names if n.startswith("f_rest_")], key=lambda s: int(s.split("_")[-1]))
    N = len(v["x"])
    shN = g(*rest).reshape(N, 3, -1).transpose(0, 2, 1) if rest else np.zeros((N, 0, 3), np.float32)
    return {"means": g("x", "y", "z"), "quats": g("rot_0", "rot_1", "rot_2", "rot_3"),
            "scales": g("scale_0", "scale_1", "scale_2"), "opacities": np.asarray(v["opacity"], np.float32),
            "sh0": g("f_dc_0", "f_dc_1", "f_dc_2")[:, None], "shN": shN}


# ------------------------------------------------------------------------------------------------ training

C0 = 0.28209479177387814


def ssim_map(a, b, win: int = 11, sigma: float = 1.5):
    """SSIM map of (1, 3, H, W) tensors in [0, 1] (Gaussian window, standard constants)."""
    import torch
    import torch.nn.functional as F
    x = torch.arange(win, device=a.device, dtype=a.dtype) - win // 2
    g = torch.exp(-x ** 2 / (2 * sigma ** 2))
    g = (g / g.sum())
    w = (g[:, None] @ g[None, :]).expand(a.shape[1], 1, win, win).contiguous()
    f = lambda z: F.conv2d(z, w, padding=win // 2, groups=a.shape[1])
    mu_a, mu_b = f(a), f(b)
    saa, sbb, sab = f(a * a) - mu_a ** 2, f(b * b) - mu_b ** 2, f(a * b) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return ((2 * mu_a * mu_b + c1) * (2 * sab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (saa + sbb + c2))


class Splat:
    """Gaussian parameters + render helper (SfM frame)."""

    def __init__(self, params: dict, sh_degree: int):
        self.params = params
        self.sh_degree = sh_degree

    @classmethod
    def from_points(cls, P: np.ndarray, C: np.ndarray, cfg: RoomConfig, device="cuda") -> "Splat":
        import torch
        from scipy.spatial import cKDTree
        dist, _ = cKDTree(P).query(P, k=4)
        sc = np.clip(dist[:, 1:].mean(1), 1e-3, 1.0) * cfg.init_scale
        N = len(P)
        K = (cfg.sh_degree + 1) ** 2
        rgb = C.astype(np.float32) / 255
        t = lambda x: torch.nn.Parameter(torch.as_tensor(np.asarray(x, np.float32), device=device))
        params = torch.nn.ParameterDict({
            "means": t(P), "scales": t(np.log(np.repeat(sc[:, None], 3, 1))),
            "quats": t(np.tile([1, 0, 0, 0], (N, 1))),
            "opacities": t(np.full(N, math.log(cfg.init_opacity / (1 - cfg.init_opacity)))),
            "sh0": t(((rgb - 0.5) / C0)[:, None]), "shN": t(np.zeros((N, K - 1, 3)))})
        return cls(params, cfg.sh_degree)

    @classmethod
    def from_ply(cls, path: Path, device="cuda") -> "Splat":
        import torch
        d = load_ply(path)
        params = torch.nn.ParameterDict({k: torch.nn.Parameter(torch.as_tensor(v, device=device))
                                         for k, v in d.items()})
        return cls(params, int(round(math.sqrt(d["shN"].shape[1] + 1))) - 1)

    def __len__(self):
        return len(self.params["means"])

    def render(self, viewmat, K, W, H, sh_degree=None, mode="RGB", rasterize_mode="antialiased", mask=None,
               colors=None, camera_model="pinhole", backgrounds=None):
        import torch
        from gsplat import rasterization
        p = self.params
        sel = (lambda x: x) if mask is None else (lambda x: x[mask])
        if colors is None:
            colors = torch.cat([p["sh0"], p["shN"]], 1)
            deg = self.sh_degree if sh_degree is None else sh_degree
        else:
            deg = None
        dev = p["means"].device
        vm = torch.as_tensor(np.asarray(viewmat, np.float32), device=dev)[None] if not torch.is_tensor(viewmat) \
            else viewmat[None]
        Ks = torch.as_tensor(np.asarray(K, np.float32), device=dev)[None] if not torch.is_tensor(K) else K[None]
        return rasterization(sel(p["means"]), sel(p["quats"]), torch.exp(sel(p["scales"])),
                             torch.sigmoid(sel(p["opacities"])), sel(colors), vm, Ks, int(W), int(H),
                             sh_degree=deg, render_mode=mode, rasterize_mode=rasterize_mode, packed=False,
                             near_plane=0.05, far_plane=200.0, camera_model=camera_model, backgrounds=backgrounds)

    def numpy(self) -> dict:
        return {k: v.detach().cpu().numpy() for k, v in self.params.items()}

    def save_ply(self, path: Path, sel=None, sh_degree: int | None = None) -> int:
        d = self.numpy()
        if sel is not None:
            d = {k: v[sel] for k, v in d.items()}
        deg = self.sh_degree if sh_degree is None else sh_degree
        shN = d["shN"][:, :(deg + 1) ** 2 - 1]
        save_ply(path, d["means"], d["quats"], d["scales"], d["opacities"], d["sh0"], shN)
        return len(d["means"])


class Appearance:
    """Per-image colour transform out = A @ rgb + b (identity init). Held-out images use their camera's mean."""

    def __init__(self, n: int, device="cuda"):
        import torch
        self.A = torch.nn.Parameter(torch.eye(3, device=device).repeat(n, 1, 1))
        self.b = torch.nn.Parameter(torch.zeros(n, 3, device=device))

    def apply(self, rgb, i: int):
        return rgb @ self.A[i].T + self.b[i]

    def reg(self):
        import torch
        I = torch.eye(3, device=self.A.device)
        return ((self.A.mean(0) - I) ** 2).sum() + (self.b.mean(0) ** 2).sum() + \
            1e-3 * (((self.A - I) ** 2).sum((1, 2)).mean() + (self.b ** 2).sum(1).mean())

    def mean_for(self, idx: list[int]):
        return self.A[idx].mean(0).detach(), self.b[idx].mean(0).detach()


def fit_affine(render: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Least-squares 3x4 colour transform render -> gt over the valid pixels (colour-corrected held-out metric)."""
    X = np.concatenate([render[valid], np.ones((int(valid.sum()), 1))], 1)
    M, *_ = np.linalg.lstsq(X, gt[valid], rcond=None)
    return np.clip(np.concatenate([render, np.ones(render.shape[:2] + (1,))], -1) @ M, 0, 1)


def masked_metrics(render: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> dict:
    """PSNR over valid pixels, SSIM averaged over valid pixels (float images in [0, 1])."""
    import torch
    mse = float(((render - gt) ** 2)[valid].mean())
    a = torch.as_tensor(render, dtype=torch.float32).permute(2, 0, 1)[None]
    b = torch.as_tensor(np.where(valid[..., None], gt, render), dtype=torch.float32).permute(2, 0, 1)[None]
    s = ssim_map(a, b)[0].mean(0).numpy()
    return {"psnr": float(-10 * math.log10(max(mse, 1e-10))), "ssim": float(s[valid].mean()),
            "valid_frac": float(valid.mean())}


def photo_loss(rgb, img, valid, cfg: RoomConfig):
    """Masked (1 - l) L1 + l D-SSIM; masked pixels take the render's own (detached) value inside the SSIM window."""
    import torch
    gt = img.float() / 255
    vf = valid.float()[..., None]
    nv = vf.sum().clamp_min(1)
    l1 = ((rgb - gt).abs() * vf).sum() / (nv * 3)
    gt_f = torch.where(valid[..., None], gt, rgb.detach())
    ss = ssim_map(rgb.permute(2, 0, 1)[None].clamp(0, 1), gt_f.permute(2, 0, 1)[None])[0].mean(0)
    loss = (1 - cfg.ssim_lambda) * l1 + cfg.ssim_lambda * (1 - (ss * vf[..., 0]).sum() / nv)
    with torch.no_grad():
        mse = ((((rgb - gt) ** 2) * vf).sum() / (nv * 3)).item()
    return loss, l1.item(), -10 * math.log10(max(mse, 1e-10))


def plane_depth_loss(res: dict, valid, clamp: float):
    """Planar depth prior: where a reflector plane (floor, mirror) is the first surface on the ray, the rendered
    expected depth should lie on it. Relative error, clamped (objects standing on the floor stop contributing once
    they are clearly off the plane), so it mainly removes the depth ambiguity that lets floaters fake the glossy
    reflections."""
    import torch
    tot = 0.0
    for name, vis in res["vis"].items():
        sel = vis & valid & (res["alpha"].detach() > 0.5)
        if sel.sum() < 100:
            continue
        z = res["plane_z"][name][sel]
        tot = tot + ((res["depth"][sel] - z).abs() / z).clamp(max=clamp).mean()
    return tot if torch.is_tensor(tot) else torch.zeros((), device=res["depth"].device)


def _kill(p, bad, value: float) -> None:
    import torch
    with torch.no_grad():
        p["opacities"][bad] = value


def train(work: Path, views: list[View], geo: dict, cfg: RoomConfig, log=print):
    import torch
    from gsplat.strategy import MCMCStrategy
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    dev = "cuda"
    tr = [v for v in views if v.split == "train"]
    t0 = time.time()
    data = []
    for v in tr:
        img, valid = load_view(work, v, cfg)
        data.append((torch.as_tensor(img, device=dev), torch.as_tensor(valid, device=dev),
                     torch.as_tensor(v.viewmat(), dtype=torch.float32, device=dev),
                     torch.as_tensor(v.K_gs(), dtype=torch.float32, device=dev)))
    log(f"[room] loaded {len(tr)} training views in {time.time() - t0:.0f}s "
        f"({sum(v.kind == 'kf' for v in tr)} keyframes, {sum(v.kind == 'win' for v in tr)} window frames), "
        f"valid pixels {np.mean([d[1].float().mean().item() for d in data]):.0%}")
    P, C, E, L = sfm_points(work)
    keep = (L >= cfg.min_track) & (E < cfg.max_point_err)
    with torch.no_grad():
        keep &= ~behind_reflectors(torch.as_tensor(P, dtype=torch.float32), geo, cfg.floor_margin).numpy()
    splat = Splat.from_points(P[keep], C[keep], cfg, dev)
    log(f"[room] init {len(splat)} Gaussians from {len(P)} SfM points (track >= {cfg.min_track}, err < "
        f"{cfg.max_point_err} px, not under the floor / behind the mirror); scene scale {geo['scene_scale']:.2f}; "
        f"reflectors {[r['name'] for r in geo['reflectors']]}")
    p = splat.params
    sc = geo["scene_scale"]
    lrs = {"means": cfg.lr_means * sc, "scales": cfg.lr_scales, "quats": cfg.lr_quats,
           "opacities": cfg.lr_opacities, "sh0": cfg.lr_sh0, "shN": cfg.lr_shN}
    opt = {k: torch.optim.Adam([{"params": p[k], "lr": lr, "name": k}], eps=1e-15) for k, lr in lrs.items()}
    sched = torch.optim.lr_scheduler.ExponentialLR(opt["means"], gamma=0.01 ** (1.0 / cfg.iters))
    app = Appearance(len(tr), dev)
    refl = Reflectors(geo["reflectors"], dev, enabled=cfg.reflections)
    app_opt = torch.optim.Adam([app.A, app.b], lr=cfg.lr_app)
    refl_opt = torch.optim.Adam([refl.logit], lr=cfg.lr_refl)
    strat = MCMCStrategy(cap_max=cfg.cap_max, refine_start_iter=cfg.refine_start, refine_stop_iter=cfg.refine_stop,
                         refine_every=cfg.refine_every)
    strat.check_sanity(p, opt)
    state = strat.initialize_state()
    kf_idx = [i for i, v in enumerate(tr) if v.kind == "kf"]
    win_idx = [i for i, v in enumerate(tr) if v.kind == "win"]
    dead_logit = math.log(strat.min_opacity / 2 / (1 - strat.min_opacity / 2))
    hist = []
    t0 = time.time()
    run_l, run_psnr = 0.0, 0.0
    for step in range(cfg.iters):
        pool = kf_idx if (kf_idx and (not win_idx or rng.random() < cfg.kf_weight)) else win_idx
        i = int(pool[rng.integers(len(pool))])
        img, valid, vm, K = data[i]
        H, W = img.shape[:2]
        deg = min(step // cfg.sh_every, cfg.sh_degree)
        bg = torch.rand(3, device=dev) if cfg.random_bg else None
        res = refl.render(splat, vm, K, W, H, sh_degree=deg, reflect=step >= cfg.refl_start, bg=bg)
        rgb = app.apply(res["rgb"], i)
        loss, l1, psnr = photo_loss(rgb, img, valid, cfg)
        if cfg.plane_depth_w > 0 and step >= cfg.refl_start:
            loss = loss + cfg.plane_depth_w * plane_depth_loss(res, valid, cfg.plane_depth_clamp)
        loss = loss + cfg.opacity_reg * torch.sigmoid(p["opacities"]).mean() + \
            cfg.scale_reg * torch.exp(p["scales"]).mean() + cfg.app_reg * app.reg()
        loss.backward()
        for o in opt.values():
            o.step()
            o.zero_grad(set_to_none=True)
        for o in (app_opt, refl_opt):
            o.step()
            o.zero_grad(set_to_none=True)
        if step % 100 == 0:  # clamp: kill Gaussians under the floor / behind the mirror (MCMC relocates dead ones)
            _kill(p, behind_reflectors(p["means"].detach(), geo, cfg.floor_margin), dead_logit)
        strat.step_post_backward(p, opt, state, step, res["info"], lr=sched.get_last_lr()[0])
        sched.step()
        run_l = 0.98 * run_l + 0.02 * l1 if step else l1
        run_psnr = 0.98 * run_psnr + 0.02 * psnr if step else psnr
        if step % cfg.log_every == 0 or step == cfg.iters - 1:
            hist.append({"step": step, "l1": run_l, "psnr": run_psnr, "n": len(splat),
                         "s": round(time.time() - t0, 1), "reflectance": refl.betas()})
            log(f"[room] step {step:5d}  L1 {run_l:.4f}  masked PSNR {run_psnr:5.2f}  N {len(splat):,}  "
                f"{time.time() - t0:5.0f}s  refl " + " ".join(f"{k} {v['f0']:.2f}/{v['fresnel_gain']:.2f}"
                                                            for k, v in refl.betas().items()), flush=True)
    with torch.no_grad():
        dead = behind_reflectors(p["means"], geo, cfg.floor_margin) | \
            (torch.sigmoid(p["opacities"]) < strat.min_opacity)
    _keep_only(splat, ~dead)
    return splat, app, refl, {"history": hist, "train_s": time.time() - t0, "n_final": len(splat),
                              "reflections": cfg.reflections,
                              "removed_behind_reflectors_or_dead": int(dead.sum()), "reflectance": refl.betas()}


def _keep_only(splat: Splat, keep) -> None:
    import torch
    for k in list(splat.params.keys()):
        splat.params[k] = torch.nn.Parameter(splat.params[k].detach()[keep])


def save_ckpt(path: Path, splat: Splat, app: Appearance, refl: Reflectors, views: list[View], extra: dict) -> None:
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    tr = [v.label for v in views if v.split == "train"]
    torch.save({"params": {k: v.detach().cpu() for k, v in splat.params.items()}, "sh_degree": splat.sh_degree,
                "A": app.A.detach().cpu(), "b": app.b.detach().cpu(), "refl_logit": refl.logit.detach().cpu(),
                "refl_names": [p["name"] for p in refl.planes], "train_names": tr, "extra": extra}, path)


def load_ckpt(path: Path, views: list[View], geo: dict, device="cuda"):
    import torch
    ck = torch.load(path, map_location=device, weights_only=False)
    splat = Splat(torch.nn.ParameterDict({k: torch.nn.Parameter(v.to(device)) for k, v in ck["params"].items()}),
                  ck["sh_degree"])
    tr = [v.label for v in views if v.split == "train"]
    if tr != ck["train_names"]:
        raise ValueError("room: checkpoint was trained on a different view set; rerun --stages train")
    app = Appearance(len(tr), device)
    app.A.data[:], app.b.data[:] = ck["A"].to(device), ck["b"].to(device)
    refl = Reflectors(geo["reflectors"], device, enabled=ck.get("extra", {}).get("reflections", True))
    if [p["name"] for p in refl.planes] != ck.get("refl_names", []):
        raise ValueError(f"room: checkpoint reflectors {ck.get('refl_names')} != {[p['name'] for p in refl.planes]}")
    if len(refl.planes):
        refl.logit.data[:] = ck["refl_logit"].to(device)
    return splat, app, refl, ck.get("extra", {})


# ------------------------------------------------------------------------------------------------- quest splat

def importance(splat: Splat, views: list[View], cfg: RoomConfig) -> np.ndarray:
    """LightGaussian global significance: sum over training pixels of each Gaussian's blending weight
    alpha_i * T_i, obtained as d(sum of a rendered all-ones channel)/d(per-Gaussian colour)."""
    import torch
    N = len(splat)
    imp = torch.zeros(N, device="cuda")
    for v in views:
        if v.split != "train":
            continue
        w = torch.ones(N, 1, device="cuda", requires_grad=True)
        out, _, _ = splat.render(v.viewmat(), v.K_gs(), v.W, v.H, colors=w)
        out.sum().backward()
        imp += w.grad[:, 0]
    return imp.cpu().numpy()


def in_box(means: np.ndarray, geo: dict, pad: float) -> np.ndarray:
    return np.all((means >= geo["box_lo"] - pad) & (means <= geo["box_hi"] + pad), axis=1)


def make_quest(work: Path, splat: Splat, app: Appearance, refl: Reflectors, views: list[View], geo: dict,
               cfg: RoomConfig, log=print) -> tuple[Splat, dict]:
    """Prune to quest_max by rendered significance (inside the room box), then fine-tune at SH degree quest_sh
    with the same reflection model and colour transforms (frozen)."""
    import torch
    imp = importance(splat, views, cfg)
    d = splat.numpy()
    box = in_box(d["means"], geo, cfg.room_pad)
    score = np.where(box, imp, -1.0)
    order = np.argsort(-score)
    keep = order[:min(cfg.quest_max, int(box.sum()))]
    kept_frac = float(imp[keep].sum() / max(imp.sum(), 1e-9))
    # opacity x volume ranking for comparison - how much rendered significance it would keep
    vol = np.exp(d["scales"]).prod(1) * (1 / (1 + np.exp(-d["opacities"])))
    ov = np.argsort(-np.where(box, vol, -1.0))[:len(keep)]
    log(f"[room] quest: {len(keep):,}/{len(splat):,} Gaussians keep {kept_frac:.1%} of the rendered significance "
        f"(opacity x volume ranking would keep {imp[ov].sum() / max(imp.sum(), 1e-9):.1%}); {int((~box).sum()):,} "
        f"outside the room box")
    K = (cfg.quest_sh + 1) ** 2
    q = Splat(torch.nn.ParameterDict({k: torch.nn.Parameter(torch.as_tensor(
        v[keep] if k != "shN" else v[keep][:, :K - 1], device="cuda").contiguous()) for k, v in d.items()}),
        cfg.quest_sh)
    tr = [v for v in views if v.split == "train"]
    rng = np.random.default_rng(cfg.seed + 1)
    p = q.params
    lrs = {"means": cfg.lr_means * geo["scene_scale"] * 0.1, "scales": cfg.lr_scales, "quats": cfg.lr_quats,
           "opacities": cfg.lr_opacities, "sh0": cfg.lr_sh0, "shN": cfg.lr_shN}
    opt = torch.optim.Adam([{"params": p[k], "lr": lr} for k, lr in lrs.items()], eps=1e-15)
    refl.logit.requires_grad_(False)
    cache: dict[int, tuple] = {}
    t0 = time.time()
    for step in range(cfg.quest_iters):
        i = int(rng.integers(len(tr)))
        v = tr[i]
        if i not in cache:
            img, valid = load_view(work, v, cfg)
            cache[i] = (torch.as_tensor(img, device="cuda"), torch.as_tensor(valid, device="cuda"),
                        torch.as_tensor(v.viewmat(), dtype=torch.float32, device="cuda"),
                        torch.as_tensor(v.K_gs(), dtype=torch.float32, device="cuda"))
        img, valid, vm, Kt = cache[i]
        res = refl.render(q, vm, Kt, v.W, v.H, bg=torch.rand(3, device="cuda") if cfg.random_bg else None)
        rgb = res["rgb"] @ app.A[i].detach().T + app.b[i].detach()
        loss, _, _ = photo_loss(rgb, img, valid, cfg)
        if cfg.plane_depth_w > 0:
            loss = loss + cfg.plane_depth_w * plane_depth_loss(res, valid, cfg.plane_depth_clamp)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        _kill(p, behind_reflectors(p["means"].detach(), geo, cfg.floor_margin), -10.0)
    refl.logit.requires_grad_(True)
    with torch.no_grad():
        dead = behind_reflectors(p["means"], geo, cfg.floor_margin) | (torch.sigmoid(p["opacities"]) < 0.005)
    _keep_only(q, ~dead)
    return q, {"n": len(q), "significance_kept": kept_frac, "finetune_s": time.time() - t0,
               "opacity_volume_significance_kept": float(imp[ov].sum() / max(imp.sum(), 1e-9)),
               "removed_after_finetune": int(dead.sum()), "sh_degree": cfg.quest_sh}


# ------------------------------------------------------------------------------------------------- evaluation

def render_view(splat: Splat, v: View, A=None, b=None, refl: Reflectors | None = None,
                rasterize_mode="antialiased") -> dict:
    """numpy dict: rgb (photo model: direct + reflections, colour transform A, b applied), direct (diffuse room,
    canonical colour), depth (expected z), alpha; refl_<name> when a reflector is visible."""
    import torch
    r = refl or Reflectors([], "cuda")
    with torch.no_grad():
        vm = torch.as_tensor(v.viewmat(), dtype=torch.float32, device="cuda")
        K = torch.as_tensor(v.K_gs(), dtype=torch.float32, device="cuda")
        res = r.render(splat, vm, K, v.W, v.H, reflect=refl is not None, rasterize_mode=rasterize_mode)
        rgb = res["rgb"]
        if A is not None:
            rgb = rgb @ A.T + b
        outd = {"rgb": rgb.clamp(0, 1).cpu().numpy(), "direct": res["direct"].clamp(0, 1).cpu().numpy(),
                "depth": res["depth"].cpu().numpy(), "alpha": res["alpha"].cpu().numpy()}
        for k, x in res.items():
            if k.startswith("refl_"):
                outd[k] = x.clamp(0, 1).cpu().numpy()
        outd["vis"] = {k: x.cpu().numpy() for k, x in res["vis"].items()}
    return outd


def evaluate(work: Path, splat: Splat, app: Appearance, refl: Reflectors | None, views: list[View],
             cfg: RoomConfig, rasterize_mode="antialiased") -> dict:
    tr = [v for v in views if v.split == "train"]
    by_cam: dict[str, list[int]] = {}
    for i, v in enumerate(tr):
        by_cam.setdefault(v.cam, []).append(i)
    rows = []
    for v in views:
        if v.split != "test":
            continue
        A, b = app.mean_for(by_cam.get(v.cam, list(range(len(tr)))))
        r = render_view(splat, v, A, b, refl, rasterize_mode=rasterize_mode)
        img, valid = load_view(work, v, cfg)
        gt = img.astype(np.float32) / 255
        m = masked_metrics(r["rgb"], gt, valid)
        mc = masked_metrics(fit_affine(r["rgb"], gt, valid), gt, valid)
        rows.append({"view": v.label, "cam": v.cam, "kind": v.kind, "t": round(v.t, 3), **m,
                     "psnr_cc": mc["psnr"], "ssim_cc": mc["ssim"], "alpha_mean": float(r["alpha"].mean())})
    summ = {}
    for key, sel in [("all", lambda r: True), ("kf", lambda r: r["kind"] == "kf"),
                     ("win", lambda r: r["kind"] == "win")] + \
            [(c, (lambda c: lambda r: r["cam"] == c)(c)) for c in sorted(by_cam)]:
        rs = [r for r in rows if sel(r)]
        if rs:
            summ[key] = {k: round(float(np.mean([r[k] for r in rs])), 3) for k in
                         ("psnr", "ssim", "psnr_cc", "ssim_cc", "valid_frac")} | {"n": len(rs)}
    return {"views": rows, "summary": summ}


def floor_depth_check(splat: Splat, views: list[View], geo: dict, max_views: int = 60) -> dict:
    """Does the rendered (direct) depth put the floor ON the floor plane? For pixels whose ray hits the floor plane
    before 12 m: the fraction whose rendered depth lies within 10 cm / 25 cm (height) of it. Floaters that fake the
    glossy reflections pull the depth off the plane."""
    tr = [v for v in views if v.split == "train"][::max(1, sum(v.split == "train" for v in views) // max_views)]
    n, d = geo["floor_n"], geo["floor_d"]
    within10, within25, tot = 0, 0, 0
    for v in tr:
        r = render_view(splat, v)
        yy, xx = np.mgrid[0:v.H, 0:v.W]
        ray = np.stack([(xx - v.K[0, 2]) / v.K[0, 0], (yy - v.K[1, 2]) / v.K[1, 1], np.ones(xx.shape)], -1)
        nc = v.R @ n
        dc = d - nc @ v.tvec
        zf = -dc / np.where(np.abs(ray @ nc) < 1e-9, 1e-9, ray @ nc)
        X = (ray * r["depth"][..., None] - v.tvec) @ v.R
        h = X @ n + d
        sel = (zf > 0.3) & (zf < 12) & (r["alpha"] > 0.5) & (r["depth"] > zf * 0.7)  # floor is (nearly) first hit
        within10 += int((sel & (np.abs(h) < 0.1)).sum())
        within25 += int((sel & (np.abs(h) < 0.25)).sum())
        tot += int(sel.sum())
    return {"floor_pixels": tot, "within_10cm": within10 / max(tot, 1), "within_25cm": within25 / max(tot, 1),
            "views": len(tr)}


# ------------------------------------------------------------------------------------------------- driver

def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def write_transform(out: Path, geo: dict, refl: dict | None = None) -> dict:
    w = geo["world"]
    tr = {"frame": "sfm",
          "note": "all room assets (splat_full.ply, splat_quest.ply, room_mesh.*) are in the SfM frame of "
                  "cameras/sfm; X_world = s * R @ X_sfm + t (world: right-handed, Y up, metres, floor y=0); "
                  "Unity = (x, y, -z) of world. Apply world_from_sfm at export.",
          "world_from_sfm": {"s": w["s"], "R": w["R"], "t": w["t"]},
          "floor_plane_sfm": {"n": geo["floor_n"], "d": geo["floor_d"], "inliers": geo["floor_inliers"],
                              "tilt_vs_world_up_deg": geo["tilt_vs_world_deg"],
                              "world_height_at_origin_m": None},
          "room_box_sfm": {"lo": geo["box_lo"], "hi": geo["box_hi"]}, "scene_scale": geo["scene_scale"]}
    n, d = geo["floor_n"], geo["floor_d"]
    X0 = geo["origin"]
    foot = X0 - (n @ X0 + d) * n
    tr["floor_plane_sfm"]["world_height_at_origin_m"] = float((w["s"] * w["R"] @ foot + w["t"])[1])
    m = geo.get("mirror")
    if m is not None:
        mm = {k: m[k] for k in ("is_mirror", "inliers", "seen_through_below_soffit", "top_m", "cams_dist_m")}
        mm["world"] = {"n": m["n_w"], "d": m["d_w"], "along": m["along_w"], "along_range": [m["along_lo"],
                                                                                             m["along_hi"]],
                       "height_range_m": [0.0, m["top_m"]],
                       "note": "vertical plane n.X + d = 0 (n towards the cameras); the mirror is the part with "
                               "along . X in along_range and world height in height_range_m"}
        if "n" in m:
            mm["sfm"] = {"n": m["n"], "d": m["d"], "along": m["along"], "along_range": [m["a_lo"], m["a_hi"]]}
        tr["mirror"] = mm
    if refl is not None:
        tr["reflectance"] = refl
        tr["reflectance_note"] = ("photo model = diffuse splat + f * mirrored splat; floor f = f0 + fresnel_gain * "
                                  "(1 - cos)^5. The exported splats/mesh are the diffuse room only: add planar "
                                  "reflections in Unity (floor plane, mirror rectangle) if wanted.")
    (out / "transform.json").write_text(json.dumps(_jsonable(tr), indent=1))
    return tr


# ------------------------------------------------------------------------------------------------- bundle

def _arrays(x):
    """Inverse of _jsonable for geometry: numeric (nested) lists -> float arrays, recursively."""
    if isinstance(x, dict):
        return {k: _arrays(v) for k, v in x.items()}
    if isinstance(x, list):
        if x and all(isinstance(e, (int, float)) and not isinstance(e, bool) for e in x):
            return np.asarray(x, float)
        if x and all(isinstance(e, list) for e in x):
            try:
                return np.asarray(x, float)
            except (ValueError, TypeError):
                pass
        return [_arrays(e) for e in x]
    return x


def _prep_cfg(cfg: RoomConfig) -> dict:
    return _jsonable({k: getattr(cfg, k) for k in PREP_FIELDS})


def _prep_inputs(work: Path, cfg: RoomConfig) -> dict:
    """Files a bundle is derived from (relative path -> mtime): a newer file makes the bundle stale."""
    files = [work / "cameras" / "world_from_sfm.json"] + \
        [work / "cameras" / "sfm" / f for f in ("cameras.bin", "images.bin", "points3D.bin")] + \
        [work / "cameras" / f"{c}.npz" for c in cfg.cams]
    return {f.relative_to(work).as_posix(): round(f.stat().st_mtime, 3) for f in files if f.exists()}


def bundle_fresh(work: Path, bundle: Path, cfg: RoomConfig) -> bool:
    mp = bundle / "manifest.json"
    if not mp.exists():
        return False
    m = json.loads(mp.read_text())
    return (m.get("version") == BUNDLE_VERSION and m.get("prep_config") == _prep_cfg(cfg)
            and m.get("inputs") == _prep_inputs(work, cfg))


def write_bundle(work: Path, bundle: Path, cfg: RoomConfig, log=print, force: bool = False) -> dict:
    """prep: everything train/quest/mesh/eval need, compact and platform-neutral (no pycolmap, no frame store):
        manifest.json  version, take, prep_config, inputs (mtimes), counts, bytes
        views.json     [View] with name images/<i>.jpg, mask masks/<i>.png (k1 = 0: undistorted), src = original frame
        geo.json       room_geometry (floor plane, mirror wall, reflectors, room box, world_from_sfm) in the SfM frame
        points.npz     SfM points P (N,3) f32, C (N,3) u8, E (N,) f32 reprojection error, L (N,) track length
        images/        undistorted frames at <= bundle_max_side, JPEG q95
        masks/         people masks (255 = ignore), same size, PNG
    Skipped when the manifest matches the current inputs and prep settings (force=True rewrites)."""
    if not force and bundle_fresh(work, bundle, cfg):
        m = json.loads((bundle / "manifest.json").read_text())
        log(f"[room] prep: bundle up to date ({m['n_views']} views, {m['bytes'] / 1e6:.0f} MB) -> {bundle}")
        return m
    import shutil
    t0 = time.time()
    if bundle.exists():
        shutil.rmtree(bundle)
    (bundle / "images").mkdir(parents=True)
    (bundle / "masks").mkdir(parents=True)
    views = collect_views(work, cfg)
    geo = room_geometry(work, [v for v in views if v.split == "train"], cfg)
    out_views = []
    for i, v in enumerate(views):
        img, m = load_raw(work, v)
        name, mask = f"images/{i:04d}.jpg", f"masks/{i:04d}.png"
        cv2.imwrite(str(bundle / name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(bundle / mask), np.where(m > 64, 255, 0).astype(np.uint8), [cv2.IMWRITE_PNG_COMPRESSION, 9])
        out_views.append(View(name, v.cam, v.kind, v.t, v.K, v.R, v.tvec, v.W, v.H, 0.0, v.scale, mask, v.split,
                              v.name))
    P, C, E, L = sfm_points(work)
    np.savez_compressed(bundle / "points.npz", P=P.astype(np.float32), C=C.astype(np.uint8),
                        E=E.astype(np.float32), L=L.astype(np.int32))
    (bundle / "views.json").write_text(json.dumps([v.to_json() for v in out_views]))
    (bundle / "geo.json").write_text(json.dumps(_jsonable(geo), indent=1))
    nbytes = sum(f.stat().st_size for f in bundle.rglob("*") if f.is_file())
    man = {"version": BUNDLE_VERSION, "take": work.name, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
           "prep_config": _prep_cfg(cfg), "inputs": _prep_inputs(work, cfg), "n_views": len(out_views),
           "n_train": sum(v.split == "train" for v in out_views), "n_points": int(len(P)),
           "resolutions": sorted({f"{v.cam}:{v.W}x{v.H}" for v in out_views}), "bytes": nbytes,
           "prep_s": round(time.time() - t0, 1)}
    (bundle / "manifest.json").write_text(json.dumps(man, indent=1))
    log(f"[room] prep: bundle {len(out_views)} views ({man['n_train']} train), {len(P):,} SfM points, "
        f"{nbytes / 1e6:.0f} MB in {man['prep_s']:.0f}s -> {bundle}")
    return man


def load_bundle(bundle: Path) -> tuple[list[View], dict, dict]:
    if not (bundle / "manifest.json").exists():
        raise FileNotFoundError(f"room: no training bundle at {bundle}; run --stages prep locally first")
    man = json.loads((bundle / "manifest.json").read_text())
    if man.get("version") != BUNDLE_VERSION:
        raise ValueError(f"room: bundle version {man.get('version')} != {BUNDLE_VERSION}; rerun --stages prep")
    views = [View.from_json(d) for d in json.loads((bundle / "views.json").read_text())]
    geo = _arrays(json.loads((bundle / "geo.json").read_text()))
    geo["floor_d"] = float(geo["floor_d"])
    return views, geo, man


# ------------------------------------------------------------------------------------------------- run

def run(work: Path, cfg: RoomConfig, stages=STAGES, out: Path | None = None, review: Path | None = None,
        log=print) -> dict:
    """prep (local) and/or the bundle stages. out defaults to work/room (review images -> work/review); with another
    out (e.g. a smoke run) the review sheets go to out/review so the real ones are not overwritten."""
    work = Path(work)
    bundle = work / "room" / "bundle"
    if "prep" in stages:
        write_bundle(work, bundle, cfg, log)
    elif (work / "cameras" / "sfm").exists() and (bundle / "manifest.json").exists() and \
            not bundle_fresh(work, bundle, cfg):
        log("[room] WARNING: the bundle is older than cameras/ (or was prepped with other settings); "
            "rerun --stages prep")
    rest = tuple(s for s in stages if s != "prep")
    if not rest:
        return {"bundle": json.loads((bundle / "manifest.json").read_text())}
    out = Path(out) if out is not None else work / "room"
    review = Path(review) if review is not None else (work / "review" if out == work / "room" else out / "review")
    return run_bundle(bundle, out, cfg, rest, review, log)


def run_bundle(bundle: Path, out: Path, cfg: RoomConfig, stages, review: Path, log=print) -> dict:
    import platform

    import torch

    from .env import is_pod
    views_b, geo, man = load_bundle(bundle)
    for k, v in man["prep_config"].items():  # the bundle decides the prep settings
        setattr(cfg, k, tuple(v) if isinstance(v, list) else v)
    views = [v.resized(cfg.max_side) for v in views_b]
    out.mkdir(parents=True, exist_ok=True)
    write_transform(out, geo)
    ntr = sum(v.split == "train" for v in views)
    m = geo.get("mirror") or {}
    log(f"[room] {len(views)} views ({ntr} train, {len(views) - ntr} held out) from cams {','.join(cfg.cams)} at "
        f"<= {cfg.max_side} px ({cfg.note or 'custom'}); floor plane {geo['floor_inliers']} inliers, tilt vs world up "
        f"{geo['tilt_vs_world_deg']:.2f} deg; wall/mirror: {'MIRROR' if m.get('is_mirror') else 'no mirror'} "
        f"({m.get('seen_through_below_soffit')} sparse points seen through the wall below {m.get('top_m', 0):.2f} m)")
    rep_path = out / "report.json"
    rep = json.loads(rep_path.read_text()) if rep_path.exists() else {}
    rep["config"] = _jsonable(asdict(cfg))
    rep["bundle"] = {k: man[k] for k in ("created", "n_views", "n_train", "bytes", "resolutions") if k in man}
    rep.setdefault("runs", []).append({"stages": list(stages), "preset": cfg.note, "pod": is_pod(),
                                       "host": platform.node(), "when": time.strftime("%Y-%m-%d %H:%M:%S"),
                                       "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
    rep["geometry"] = _jsonable({k: geo[k] for k in ("floor_inliers", "tilt_vs_world_deg", "scene_scale", "mirror")})
    rep["views"] = {"train": ntr, "test": len(views) - ntr,
                    "resolution": sorted({f"{v.cam}:{v.W}x{v.H}" for v in views}),
                    "per_cam": {c: {s: sum(v.cam == c and v.split == s for v in views) for s in ("train", "test")}
                                for c in cfg.cams}}
    ck = out / "cache" / "splat_full.pt"
    splat = app = refl = None
    with gpu.exclusive("room"):
        if "train" in stages:
            splat, app, refl, info = train(bundle, views, geo, cfg, log)
            save_ckpt(ck, splat, app, refl, views, info)
            n = splat.save_ply(out / "splat_full.ply")
            rep["train"] = info
            for k in ("quest", "mesh", "eval", "review"):  # derived from the previous splat: stale now
                rep.pop(k, None)
            log(f"[room] splat_full.ply: {n:,} Gaussians ({(out / 'splat_full.ply').stat().st_size / 1e6:.0f} MB)")
        if splat is None:
            if not ck.exists():
                raise FileNotFoundError(f"room: {ck} missing; run --stages train first")
            splat, app, refl, _ = load_ckpt(ck, views, geo)
        write_transform(out, geo, refl.betas())
        if "quest" in stages:
            q, qi = make_quest(bundle, splat, app, refl, views, geo, cfg, log)
            q.save_ply(out / "splat_quest.ply")
            qi["mb"] = round((out / "splat_quest.ply").stat().st_size / 1e6, 1)
            rep["quest"] = qi
            log(f"[room] splat_quest.ply: {qi['n']:,} Gaussians, SH {cfg.quest_sh}, {qi['mb']} MB")
        if "mesh" in stages:
            from . import room_mesh
            rep["mesh"] = room_mesh.build(bundle, splat, views, geo, cfg, log, out=out)
        if "eval" in stages:
            from . import room_review
            ev = evaluate(bundle, splat, app, refl, views, cfg)
            ev_direct = evaluate(bundle, splat, app, None, views, cfg)
            ev_classic = evaluate(bundle, splat, app, refl, views, cfg, rasterize_mode="classic")
            evq = None
            if (out / "splat_quest.ply").exists():
                evq = evaluate(bundle, Splat.from_ply(out / "splat_quest.ply"), app, refl, views, cfg)
            fd = floor_depth_check(splat, views, geo)
            rep["eval"] = {"full": ev["summary"], "full_without_reflections": ev_direct["summary"],
                           "full_classic_raster": ev_classic["summary"], "quest": evq and evq["summary"],
                           "floor_depth": fd}
            (out / "eval.json").write_text(json.dumps(_jsonable({**rep["eval"], "views": ev["views"]}), indent=1))
            s = ev["summary"]
            for k in s:
                log(f"[room] held-out {k:>4}: PSNR {s[k]['psnr']:.2f} dB (colour-corrected {s[k]['psnr_cc']:.2f}), "
                    f"SSIM {s[k]['ssim']:.3f} ({s[k]['ssim_cc']:.3f}), n={s[k]['n']}")
            for name, e in (("without reflections", ev_direct), ("classic raster", ev_classic), ("Quest splat", evq)):
                if e is not None:
                    log(f"[room] held-out all, {name}: PSNR {e['summary']['all']['psnr']:.2f} "
                        f"(cc {e['summary']['all']['psnr_cc']:.2f}), SSIM {e['summary']['all']['ssim']:.3f}")
            log(f"[room] floor depth: {fd['within_10cm']:.0%} of floor pixels within 10 cm of the floor plane "
                f"({fd['within_25cm']:.0%} within 25 cm)")
            rep["review"] = room_review.render_all(bundle, splat, app, refl, views, geo, cfg, ev, log, out=out,
                                                   review=review)
        torch.cuda.empty_cache()
    rep["gates"] = gates(rep, cfg)
    rep_path.write_text(json.dumps(_jsonable(rep), indent=1))
    return rep


# ------------------------------------------------------------------------------------------------- Runpod

POD_ROOT = "/workspace"
GSPLAT_WHEELS = "https://docs.gsplat.studio/whl/pt24cu124"  # prebuilt for the default image (torch 2.4, CUDA 12.4)
POD_PRICE_HINT = 1.39  # $/h A100 80 GB community (2026-10-06); remote.pod_session checks the live price + budget


def pod_plan(work: Path, preset_name: str = "pod", max_minutes: float = 110) -> dict:
    """The pod job as data: local inputs, the exact pod commands run_pod executes, expected outputs."""
    from .take import PACKAGE_ROOT
    take = work.name
    pod_dir = work / "room" / "pod"
    wd = f"{POD_ROOT}/dancecap"
    rel = f"work/{take}"
    run_ = (f"set -o pipefail; cd {wd} && DANCECAP_POD=1 PYTHONUNBUFFERED=1 python -m dancecap room {take} "
            f"--preset {preset_name}")
    arch = "TORCH_CUDA_ARCH_LIST='8.0;9.0'"
    return {
        "take": take, "preset": preset_name, "preset_overrides": PRESETS[preset_name], "max_minutes": max_minutes,
        "gpu": "80 GB A100/H100 via dancecap.remote.pod_session (community first, secure fallback); on-pod kill "
               "switch after max_minutes + local delete in finally",
        "local_prep": [f"python -m dancecap room {take} --stages prep"],
        "inputs": {"code": {"local": str(pod_dir / "code.tar.gz"), "pod": f"{POD_ROOT}/code.tar.gz",
                            "from": str(PACKAGE_ROOT / "dancecap")},
                   "bundle": {"local": str(pod_dir / "bundle.tar"), "pod": f"{POD_ROOT}/bundle.tar",
                              "from": str(work / "room" / "bundle")}},
        "pod_commands": {
            "setup": [
                f"set -e; mkdir -p {wd} && cd {POD_ROOT} && tar xzf code.tar.gz && tar xf bundle.tar && "
                f"rm -f bundle.tar code.tar.gz",
                "(apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libgl1 libgomp1 "
                ">/dev/null) || echo 'apt failed (open3d may lack libGL)'",
                "set -e; pip install -q 'numpy<2' 'opencv-python-headless==4.10.0.84' plyfile scipy matplotlib "
                "open3d xatlas ninja",
                f"pip install -q gsplat==1.5.3 --index-url {GSPLAT_WHEELS} || "
                f"{arch} pip install -q gsplat==1.5.3",
                f"set -e; {arch} python -c \"import torch, gsplat, open3d, xatlas; "
                "from gsplat.cuda._backend import _C; print('torch', torch.__version__, 'gsplat', gsplat.__version__, "
                "'open3d', open3d.__version__, torch.cuda.get_device_name(0))\"",
            ],
            "train": f"{run_} --stages train,quest 2>&1 | tee -a {wd}/{rel}/room/pod_train.log",
            "post": f"{run_} --stages mesh,eval 2>&1 | tee -a {wd}/{rel}/room/pod_post.log",
            "pack": f"cd {wd}/{rel} && tar czf {POD_ROOT}/room_out.tar.gz --exclude=room/bundle room "
                    f"$(ls review/room_*.jpg 2>/dev/null)",
        },
        "download": {"pod": f"{POD_ROOT}/room_out.tar.gz", "local": str(pod_dir / "room_out.tar.gz"),
                     "extract_to": str(work)},
        "outputs": ["room/splat_full.ply", "room/splat_quest.ply", "room/room_mesh.obj", "room/room_mesh.mtl",
                    "room/room_mesh.ply", "room/room_mesh_tex.png", "room/room_floor_tex.png", "room/transform.json",
                    "room/report.json", "room/eval.json", "room/cache/splat_full.pt", "room/cache/room_mesh_uv.npz",
                    "review/room_renders.jpg", "review/room_overview.jpg", "review/room_overview_quest.jpg",
                    "review/room_mesh.jpg", "room/pod_train.log", "room/pod_post.log"],
        "fallback": f"if 'post' fails on the pod (open3d / xatlas), run it locally after the download: "
                    f"python -m dancecap room {take} --preset local --stages mesh,eval (reads room/cache/splat_full.pt)",
    }


def pack_pod_inputs(work: Path, plan: dict, log=print) -> dict:
    """Write code.tar.gz + bundle.tar (JPEG/PNG: no recompression) and record their sizes in the plan."""
    import tarfile

    from .take import PACKAGE_ROOT
    (work / "room" / "pod").mkdir(parents=True, exist_ok=True)
    skip = lambda ti: None if "__pycache__" in ti.name or ti.name.endswith(".pyc") else ti
    code = Path(plan["inputs"]["code"]["local"])
    with tarfile.open(code, "w:gz") as tar:
        tar.add(str(PACKAGE_ROOT / "dancecap"), arcname="dancecap/dancecap", filter=skip)
    bun = Path(plan["inputs"]["bundle"]["local"])
    with tarfile.open(bun, "w") as tar:
        tar.add(str(work / "room" / "bundle"), arcname=f"dancecap/work/{work.name}/room/bundle")
    for k, p in (("code", code), ("bundle", bun)):
        plan["inputs"][k]["bytes"] = p.stat().st_size
        plan["inputs"][k]["mb"] = round(p.stat().st_size / 1e6, 1)
    up = sum(plan["inputs"][k]["bytes"] for k in ("code", "bundle")) / 1e6
    plan["upload_mb"] = round(up, 1)
    plan["upload_s_at_2_5_MBps"] = round(up / 2.5)
    log(f"[room] pod inputs: code {plan['inputs']['code']['mb']} MB, bundle {plan['inputs']['bundle']['mb']} MB "
        f"(~{plan['upload_s_at_2_5_MBps']} s upload at 2.5 MB/s)")
    return plan


def write_pod_job(work: Path, preset_name: str = "pod", max_minutes: float = 110, log=print) -> dict:
    plan = pack_pod_inputs(work, pod_plan(work, preset_name, max_minutes), log)
    man = json.loads((work / "room" / "bundle" / "manifest.json").read_text())
    plan["bundle_manifest"] = {k: man[k] for k in ("created", "n_views", "n_train", "n_points", "bytes", "resolutions")}
    plan["estimate"] = {"setup_min": "5-10 (pip; gsplat JIT build only if the prebuilt wheel is missing)",
                        "train_quest_min": "25-45 (30k steps, 1M Gaussians, native resolution)",
                        "mesh_eval_min": "~10", "download_mb": "~550 (splat_full.ply ~250 + checkpoint ~250 + rest)",
                        "cost_usd_max": round(POD_PRICE_HINT * (max_minutes + 6) / 60, 2),
                        "note": "pod_session refuses to start unless the remaining budget covers max_minutes + the "
                                "start timeout at the live price"}
    plan["run"] = f"python -m dancecap room {work.name} --pod"
    (work / "room" / "pod_job.json").write_text(json.dumps(plan, indent=1))
    log(f"[room] pod job -> {work / 'room' / 'pod_job.json'}")
    return plan


def run_pod(work: Path, preset_name: str = "pod", max_minutes: float = 110, log=print) -> dict:
    """prep (if stale) -> pack -> pod: setup, train+quest, mesh+eval (non-fatal), pack -> download -> extract."""
    import tarfile

    from . import remote
    write_bundle(work, work / "room" / "bundle", RoomConfig(), log)
    plan = write_pod_job(work, preset_name, max_minutes, log)
    pc = plan["pod_commands"]
    plog = work / "room" / "pod" / "pod.log"
    Path(plan["download"]["local"]).unlink(missing_ok=True)  # never extract a previous run's archive
    res: dict = {"plan": str(work / "room" / "pod_job.json")}
    with remote.pod_session("room", max_minutes=max_minutes) as pod:
        res["pod"] = {"id": pod.id, "gpu": pod.gpu, "cost_per_hr": pod.cost_per_hr}
        t0 = time.time()
        for k in ("code", "bundle"):
            pod.put(Path(plan["inputs"][k]["local"]), plan["inputs"][k]["pod"])
        res["upload_s"] = round(time.time() - t0)
        for c in pc["setup"]:
            pod.run(c, log=plog)
        t0 = time.time()
        res["train_rc"] = pod.run(pc["train"], check=False, log=plog)
        res["train_min"] = round((time.time() - t0) / 60, 1)
        if res["train_rc"] == 0:
            res["post_rc"] = pod.run(pc["post"], check=False, log=plog)
        pod.run(pc["pack"], check=False, log=plog)
        try:
            pod.get(plan["download"]["pod"], Path(plan["download"]["local"]))
        except Exception as e:  # noqa: BLE001 - the session deletes the pod either way
            res["download_error"] = str(e)
    got = Path(plan["download"]["local"])
    if got.exists() and "download_error" not in res:
        with tarfile.open(got) as tar:
            tar.extractall(work, filter="data")
        res["extracted"] = True
    (work / "room" / "pod" / "result.json").write_text(json.dumps(res, indent=1))
    if res.get("train_rc"):
        raise RuntimeError(f"room: pod training failed (rc {res['train_rc']}); see {plog}")
    return res


def gates(rep: dict, cfg: RoomConfig) -> dict:
    g = {}
    ev = rep.get("eval", {}).get("full", {})
    if "all" in ev:
        g["heldout_plausible"] = {"passed": ev["all"]["psnr_cc"] >= 23.0 and ev["all"]["ssim_cc"] >= 0.8 and
                                  ev.get("kf", ev["all"])["psnr_cc"] >= 20.0,
                                  "value": {"psnr_cc": ev["all"]["psnr_cc"], "ssim_cc": ev["all"]["ssim_cc"],
                                            "kf_psnr_cc": ev.get("kf", {}).get("psnr_cc")},
                                  "rule": "held-out masked colour-corrected PSNR >= 23 dB, SSIM >= 0.8, keyframes "
                                          "(other times/poses) >= 20 dB"}
    if "quest" in rep:
        g["quest_splat_budget"] = {"passed": rep["quest"]["n"] <= cfg.quest_max, "value": rep["quest"]["n"]}
    if "mesh" in rep:
        g["quest_mesh_budget"] = {"passed": rep["mesh"]["triangles"] <= cfg.mesh_max_tris,
                                  "value": rep["mesh"]["triangles"]}
        g["no_below_floor_geometry"] = {"passed": rep["mesh"]["below_floor_frac"] < 0.01,
                                        "value": rep["mesh"]["below_floor_frac"]}
        if "behind_mirror_frac" in rep["mesh"]:
            g["no_mirrored_room"] = {"passed": rep["mesh"]["behind_mirror_frac"] < 0.01,
                                     "value": rep["mesh"]["behind_mirror_frac"]}
    return g
