"""Photoreal per-dancer albedo textures for the SMPL-X avatars, baked from the multi-view footage.

    python -m dancecap texture larissa_kadu            -> work/<take>/avatar/{lead,follow}_albedo.png (+ coverage, uv mesh)
    python -m dancecap texture larissa_kadu --preset smoke      # code-path check -> avatar/smoke/, review/avatar_smoke_*
    python -m dancecap texture larissa_kadu --write-pod-job     # avatar/pod_job.json + upload tar (avatar_pod.py)
    python -m dancecap texture larissa_kadu --pod               # full bake (preset pod) on an 80 GB Runpod GPU

Presets (PRESETS): smoke | local (2080 Ti) | pod. The bake refuses smplx/ older than cameras/world_from_sfm.json (a fit
made in a previous world frame) unless allow_stale (the smoke preset allows it and says so).

Inputs: smplx/{lead,follow}.npz (world frame, MAMMA locked-head SMPL-X; FK is checked against the stored joints),
cameras/<cam>.npz + world_from_sfm.json, frames/<cam>, dancers/<cam>/{masks,body} (lead/follow), roles.json.

Method (texel-centric, one S x S atlas per dancer on the SMPL-X UV layout, see avatar_uv.py):
  0. Head fix (avatar_head.py): MAMMA's head orientation misses the COCO face keypoints by 1-3 eye distances; a
     smoothed neck rotation is fitted to them and adopted per dancer when it generalises to held-out cameras.
  1. Every texel is a surface point (face + barycentrics). For each sampled camera frame (every `step_s` of reference
     time per camera) both bodies are posed at the frame's own timestamp (linear between 30 Hz grid frames).
  2. Visibility: both bodies' meshes are rasterised into exact per-pixel z-buffers (avatar_raster.py: pure torch,
     runs on CPU, the local GPU and a Linux pod); a texel is visible if it lies within eps of the front surface.
     The same z-buffers give each dancer's model silhouette -> IoU with the SAM mask = per-view fit quality.
  3. Weight of a sample = visibility x facing (cos) x resolution (1 / texel footprint^2, footprint from depth/focal,
     foreshortening and motion blur of the texel during an assumed exposure) x distance inside the dancer's OWN full
     mask x not in the partner's mask x silhouette IoU x MAMMA fit quality x identity confidence (roles.json) x
     camera trust; head texels get a bonus in the HD cameras (02, 06). The follow's non-scalp texels (face included)
     reject hair-coloured pixels: per-camera Lab histogram model P(hair | colour) learned from dancers/<cam>/hair
     masks vs the rest of both dancers (the dancers stage's body masks miss the darker hair). hair_mode "exclude"
     (default, avatar_hair.py): her hair mask + strongly hair-coloured pixels next to it, dilated, get weight 0 on
     every body texel (hard, not relative); the scalp-under-hair region is learned from the samples' hair evidence;
     body and scalp holes are filled separately (class-aware body fill); blend "multiband" fuses per-camera layers in
     two frequency bands so view boundaries leave no seams. Also writes the hair colour sample for Unity.
  4. Each texel keeps its top-k samples PER CAMERA (so no single camera dominates the robust statistics).
  5. Fusion in linear RGB: per-camera then per-view exposure/white-balance gains against the consensus, pooled over
     both dancers, mid-tones only (two rounds); per-view sharpness (Laplacian variance in the mask vs the camera's
     median); weighted median of luminance + colour inlier test; weighted mean of the inliers with weights^1.5
     (face: ^4, so the best-registered high-resolution view dominates instead of a ghosted average).
  6. Registration pass(es): every view is re-sampled through a smooth optical flow (DIS, coverage-weighted Gaussian,
     clamped) from a render of the previous texture to the photo, which absorbs the fit's few-pixel misalignment.
     hair_mode "exclude": her (dilated) hair pixels are copied from the photo into the render and get zero weight,
     so the hair the texture no longer has cannot drag the flow (hair_flow_mask).
  7. Holes: left/right mirror texel, then diffusion over the 3D mesh (seam-aware: UV islands are joined through
     shared vertices), then a gutter dilation so mip-maps do not bleed the background into the islands.

Limits: SMPL-X is a naked body; baggy jeans / a loose top are projected onto the skin-tight surface, so cloth edges
are slightly smeared and the clothes' silhouette is lost. Hair is not part of the body texture (the follow's hair is
masked out except on the scalp, which takes the hair colour). Lighting is baked in (studio top light, no de-lighting).
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .avatar_raster import EMPTY_Z, pixel_bary, rasterize
from .avatar_uv import UVAtlas, build_atlas, write_uv_mesh, Body
from .mamma_dyn import SMPLX_MODELS, world_cameras
from .take import Take

ROLES = ("lead", "follow")
PIDS = {"lead": 1, "follow": 2}
LUMA = (0.2126, 0.7152, 0.0722)


@dataclass
class TextureConfig:
    size: int = 2048
    step_s: float = 1 / 15          # camera frames sampled every step_s of reference time (per camera)
    cams: tuple | None = None       # default: every cameras/<cam>.npz
    top_k: int = 4                  # samples kept per texel per camera
    depth_eps_m: float = 0.02       # visibility tolerance (raised to 3 texel footprints at grazing/low res)
    exposure_s: float = 1 / 120     # assumed shutter time for the motion-blur footprint
    min_cos: float = 0.2            # surfaces seen more obliquely than ~78 deg are ignored
    margin_frac: float = 0.0035     # mask-boundary ramp (fraction of image width, >= 2 px)
    hd_cams: tuple = ("02", "06")
    face_bonus: float = 3.0         # head texels in the HD cameras
    cam_weight: dict = field(default_factory=lambda: {"05": 0.25})  # 05: attached, measured 4-5 deg off
    iou_lo: float = 0.30            # silhouette IoU -> view quality ramp
    iou_hi: float = 0.70
    inlier_log_luma: float = 0.25   # |log Y - log Ymed| inlier threshold
    inlier_rgb: float = 0.15        # sRGB distance to the median sample
    sharpen: float = 1.5            # inlier weights ^ sharpen in the mean (favours the best views)
    face_sharpen: float = 4.0       # on the face: the best-registered, highest-resolution view dominates (no ghosting)
    gain_rounds: int = 2
    gutter_px: int = 16
    hair_roles: tuple = ("follow",)  # hair-coloured samples are rejected on these dancers' non-scalp texels
    hair_frame_step: int = 6        # frames sampled per camera for the hair colour model
    hair_mode: str = "exclude"      # hair roles: "exclude" = hair mask + colour model hard-excluded (dilated) from body
                                    # texels, scalp learned from the data, region-aware fill (avatar_hair.py);
                                    # "soft" = the old colour down-weighting only
    hair_dilate_frac: float = 0.006 # hair exclusion radius r0 (x her mask height, >= 2 px) ...
    hair_ramp_frac: float = 0.006   # ... then a ramp to full weight over this x her height (>= 2 px)
    hair_p_hard: float = 0.6        # P(hair | colour) above which a pixel near her hair mask counts as hair
    fill_classes: int = 4           # hair roles: colour classes of the class-aware body fill (<= 1: plain diffusion)
    fill_ev_bad: float = 0.5        # hair roles: a measured body texel whose sample weight was mostly on her hair
                                    # (evidence > this) never seeds a fill (avatar_hair.fill_guards)
    hairline_margin_m: float = 0.015  # learned scalp: only the hairline within this surface distance of the
                                    # anatomical scalp boundary follows the evidence (avatar_hair.hair_region_from_evidence)
    hair_mask_min_frac: float = 0.9 # hair_mode exclude refuses to bake when fewer of the follow's views have a real
                                    # dancers/<cam>/hair mask (the fallback full - body is weaker / absent);
    allow_missing_hair_masks: bool = False  # ... unless this is set (then it only warns)
    blend: str = "multiband"        # "multiband" = two-band per-camera fusion (no view seams, avatar_hair.multiband);
                                    # "mean" = one weighted mean of the samples (old)
    band_sigma_frac: float = 0.012  # multiband low-pass sigma (fraction of the atlas size; 24.6 texels at 2048)
    band_sigma_high_frac: float = 0.001  # multiband: the high band's camera weights smoothed over ~2 texels at 2048
    skin_ref_q: float = 0.75        # hair roles' face-core texels where the previous pass saw her hair over the face:
                                    # luminance reference = a weighted quantile ramping from 0.5 (no hair evidence) to
                                    # this (evidence >= 0.3), so lighter skin wins over darker hair there; 0.5 on clean
                                    # face texels (a light quantile everywhere picked specular / over-exposed samples)
    align_passes: int = 1           # extra passes that register each view to a render of the previous texture
    flow_sigma_frac: float = 0.02   # smoothing of the alignment flow (fraction of the dancers' crop height)
    flow_max_frac: float = 0.06     # alignment flow clamp (fraction of the crop height)
    head_fix: str = "auto"          # neck-rotation fix to the COCO face keypoints (avatar_head.py): auto|on|off;
                                    # auto adopts it per dancer when leave-one-camera-out says it generalises
    head_prior: float = 0.06
    head_smooth: float = 0.3
    device: str | None = None       # default cuda when available
    chunk: int = 400_000            # texels per fusion chunk
    t_range: tuple | None = None    # (a, b) s after the start of both dancers' smplx window; None = all of it
    allow_stale: bool = False       # bake even when smplx/ is older than cameras/world_from_sfm.json
    preset: str = "local"


# smoke: code-path check (few frames, 512^2, minutes on the 2080 Ti, stale smplx allowed, outputs in avatar/smoke/);
# local: the 2080 Ti default (every 1/15 s per camera, one registration pass); pod: 80 GB A100/H100 (every frame of
# the 30 fps cameras / every 2nd of cam 02, 6 samples per texel per camera, two registration passes)
PRESETS = {
    "smoke": dict(size=512, step_s=0.25, t_range=(0.0, 1.5), top_k=3, align_passes=1, allow_stale=True),
    "local": dict(),
    "pod": dict(step_s=1 / 30, top_k=6, align_passes=2),
}


def preset_config(name: str, **over) -> TextureConfig:
    if name not in PRESETS:
        raise ValueError(f"unknown texture preset {name!r} (choose from {', '.join(PRESETS)})")
    kw = {**PRESETS[name], **{k: v for k, v in over.items() if v is not None}}
    return TextureConfig(preset=name, **kw)


def smplx_freshness(work: Path) -> dict:
    """smplx/ must postdate cameras/world_from_sfm.json: a fit made before the world frame changed puts the bodies
    centimetres to decimetres away from where the (new-frame) cameras see them, which smears the whole texture."""
    w = work / "cameras" / "world_from_sfm.json"
    sm = [work / "smplx" / f"{r}.npz" for r in ROLES]
    if not all(p.exists() for p in sm):
        return {"status": "missing", "stale": True}
    ts = min(p.stat().st_mtime for p in sm)
    tw = w.stat().st_mtime if w.exists() else None
    stamp = lambda x: time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(x))
    out = {"smplx_mtime": stamp(ts), "world_from_sfm_mtime": stamp(tw) if tw else None,
           "stale": bool(tw is not None and ts < tw)}
    out["status"] = "STALE (smplx predates world_from_sfm.json: re-run the MAMMA fit)" if out["stale"] else "ok"
    return out


# ------------------------------------------------------------------------------------------------ views


def _view_list(work: Path, cams: list[str], t_lo: float, t_hi: float, step_s: float) -> list[tuple[str, int, float]]:
    """Per camera, the first frame of every step_s slot of ABSOLUTE reference time (over the camera's whole frame
    list), kept when it lies in [t_lo, t_hi]. The choice does not depend on the window, so the frames a pod bundle
    carries (avatar_pod.needed_frames over the whole window) are a superset of any later run's."""
    out = []
    for c in cams:
        meta = json.loads((work / "frames" / c / "frames.json").read_text())
        last = None
        for i, fr in enumerate(meta["frames"]):
            t = float(fr["t"])
            slot = int(np.floor(t / step_s + 1e-6))
            if slot != last:
                last = slot
                if t_lo <= t <= t_hi:
                    out.append((c, i, t))
    return out


def _read_mask(p: Path, shape) -> np.ndarray:
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    return np.zeros(shape, np.uint8) if m is None else (m > 127).astype(np.uint8)


def _load_view(work: Path, cam: str, i: int, margin_frac: float, hair_lut: np.ndarray | None = None,
               cfg: "TextureConfig | None" = None, face_fn=None) -> dict:
    meta_file = work / "frames" / cam / f"{i:06d}.jpg"
    img = cv2.imread(str(meta_file), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(meta_file)
    H, W = img.shape[:2]
    dd = work / "dancers" / cam
    full = {r: _read_mask(dd / "masks" / f"mask_{i:04d}_{PIDS[r]:02d}.png", (H, W)) for r in ROLES}
    body = {r: _read_mask(dd / "body" / f"mask_{i:04d}_{PIDS[r]:02d}.png", (H, W))
            if (dd / "body" / f"mask_{i:04d}_{PIDS[r]:02d}.png").exists() else full[r] for r in ROLES}
    hair, face, hair_src = None, None, {}
    if cfg is not None and cfg.hair_mode == "exclude":  # hair masks of the hair roles (dancers_hair.py split)
        hair, face = {}, {}
        for r in cfg.hair_roles:
            hp = dd / "hair" / f"mask_{i:04d}_{PIDS[r]:02d}.png"
            bp = dd / "body" / f"mask_{i:04d}_{PIDS[r]:02d}.png"
            hair[r] = _read_mask(hp, (H, W)) if hp.exists() else (full[r] & (1 - body[r]))
            hair_src[r] = "mask" if hp.exists() else ("derived" if bp.exists() else "none")  # report: counts
            face[r] = face_fn(r, H, W) if face_fn is not None else None  # projected face + throat (avatar_hair)
    out = view_data(img[:, :, ::-1].copy(), full, body, margin_frac, hair_lut, hair=hair, cfg=cfg, face=face)
    out["hair_src"] = hair_src
    return out


def hair_mask_coverage(work: Path, views: list[dict], roles=("follow",)) -> dict:
    """Per hair role, over the views where that dancer has a mask: how many have a real dancers/<cam>/hair mask,
    how many fall back to full - body ('derived') and how many have neither ('none': no hair exclusion at all)."""
    out = {}
    for r in roles:
        n = {"mask": 0, "derived": 0, "none": 0}
        for v in views:
            dd = work / "dancers" / v["cam"]
            if not (dd / "masks" / f"mask_{v['i']:04d}_{PIDS[r]:02d}.png").exists():
                continue
            if (dd / "hair" / f"mask_{v['i']:04d}_{PIDS[r]:02d}.png").exists():
                n["mask"] += 1
            elif (dd / "body" / f"mask_{v['i']:04d}_{PIDS[r]:02d}.png").exists():
                n["derived"] += 1
            else:
                n["none"] += 1
        tot = sum(n.values())
        out[r] = {**n, "views_with_dancer_mask": tot, "real_fraction": round(n["mask"] / tot, 4) if tot else 0.0}
    return out


HAIR_BIN = 8  # OpenCV 8-bit Lab quantised to 32^3 bins


def hair_probability(img_rgb: np.ndarray, lut: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB) // HAIR_BIN
    return lut[lab[..., 0], lab[..., 1], lab[..., 2]]


def hair_model(work: Path, cam: str, frame_step: int = 6, role_hair: str = "follow") -> tuple[np.ndarray | None, dict]:
    """Per-camera P(hair | Lab colour) from dancers/<cam>/hair masks (hair) vs the rest of both dancers (the follow
    below the top 45 % of her mask box - long hair rarely reaches lower - and the lead). Histogram ratio, 32^3 bins."""
    dd = work / "dancers" / cam
    pid = PIDS[role_hair]
    if not (dd / "hair").exists():
        return None, {"cam": cam, "status": "no hair masks"}
    meta = json.loads((work / "frames" / cam / "frames.json").read_text())
    nb = 256 // HAIR_BIN
    Hh, Hn = np.zeros((nb,) * 3), np.zeros((nb,) * 3)
    k15 = np.ones((15, 15), np.uint8)
    for i in range(0, len(meta["frames"]), frame_step):
        hp = dd / "hair" / f"mask_{i:04d}_{pid:02d}.png"
        img = cv2.imread(str(work / "frames" / cam / f"{i:06d}.jpg"))
        if img is None or not hp.exists():
            continue
        lab = (cv2.cvtColor(img, cv2.COLOR_BGR2LAB) // HAIR_BIN).reshape(-1, 3)
        hair = _read_mask(hp, img.shape[:2])
        fol = _read_mask(dd / "masks" / f"mask_{i:04d}_{pid:02d}.png", img.shape[:2])
        lead = _read_mask(dd / "masks" / f"mask_{i:04d}_{3 - pid:02d}.png", img.shape[:2])
        non = fol.astype(bool) & ~cv2.dilate(hair, k15).astype(bool)
        ys = np.nonzero(fol.any(1))[0]
        if len(ys):
            non[: int(ys[0] + 0.45 * (ys[-1] - ys[0]))] = False
        non = non | lead.astype(bool)
        for H, m in ((Hh, hair.astype(bool)), (Hn, non)):
            q = lab[m.ravel()]
            np.add.at(H, (q[:, 0], q[:, 1], q[:, 2]), 1)
    if Hh.sum() < 2000 or Hn.sum() < 2000:
        return None, {"cam": cam, "status": "too few pixels", "hair_px": int(Hh.sum()), "other_px": int(Hn.sum())}
    from scipy.ndimage import gaussian_filter
    dh = gaussian_filter(Hh / Hh.sum(), 1.0)
    dn = gaussian_filter(Hn / Hn.sum(), 1.0)
    lut = (dh / (dh + dn + 1e-7)).astype(np.float32)
    rep = {"cam": cam, "status": "ok", "hair_px": int(Hh.sum()), "other_px": int(Hn.sum()),
           "hair_recall": round(float((Hh * (lut > 0.5)).sum() / Hh.sum()), 3),
           "other_false_pos": round(float((Hn * (lut > 0.5)).sum() / Hn.sum()), 3)}
    return lut, rep


def view_data(img_rgb: np.ndarray, full: dict, body: dict, margin_frac: float,
              hair_lut: np.ndarray | None = None, hair: dict | None = None, cfg: "TextureConfig | None" = None,
              face: dict | None = None) -> dict:
    """Per-view inputs of bake_samples from an RGB image and {role: uint8 0/1} full/body masks. hair {role: 0/1 hair
    mask} + cfg.hair_mode "exclude" -> hair_keep / hair_px maps per hair role (avatar_hair.hair_maps)."""
    H, W = img_rgb.shape[:2]
    margin = max(2.0, margin_frac * W)
    dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(np.ceil(margin)) + 1,) * 2)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    face_lut = None
    if isinstance(hair_lut, tuple):  # (colour model, face-aware colour model for her face / throat pixels)
        hair_lut, face_lut = hair_lut
    out = {"img": img_rgb, "H": H, "W": W, "full": full, "dt": {}, "partner": {}, "sharp": {},
           "p_hair": hair_probability(img_rgb, hair_lut) if hair_lut is not None else None}
    for r in ROLES:
        out["dt"][r] = cv2.distanceTransform(full[r], cv2.DIST_L2, 3)
        out["dt"][r + "_body"] = out["dt"][r] if body[r] is full[r] or np.array_equal(body[r], full[r]) \
            else cv2.distanceTransform(body[r], cv2.DIST_L2, 3)
        out["partner"][r] = cv2.dilate(full["follow" if r == "lead" else "lead"], dil)
        core = out["dt"][r] > 3
        out["sharp"][r] = float(lap[core].var()) if core.sum() > 200 else float("nan")
    if hair is not None and cfg is not None:
        from .avatar_hair import hair_maps
        out["hair_keep"], out["hair_px"] = {}, {}
        for r, hm in hair.items():
            out["hair_keep"][r], out["hair_px"][r] = hair_maps(
                full[r], hm, out["p_hair"], cfg.hair_dilate_frac, cfg.hair_ramp_frac, cfg.hair_p_hard,
                face=(face or {}).get(r), p_face=hair_probability(img_rgb, face_lut) if face_lut is not None else None)
    return out


def _roles_swap(work: Path) -> dict:
    p = work / "dancers" / "roles.json"
    if not p.exists():
        return {}
    r = json.loads(p.read_text())
    return {c: {f["i"]: float(f.get("swap_prob", 0.0)) for f in v["frames"]} for c, v in r["cams"].items()}


# ------------------------------------------------------------------------------------------------ torch core


class _Geo:
    """Texel geometry on the device (shared by both dancers: same UV layout)."""

    def __init__(self, atlas: UVAtlas, device):
        import torch
        self.torch = torch
        self.dev = device
        self.tri = torch.as_tensor(atlas.faces[atlas.face], dtype=torch.long, device=device)  # (N,3)
        self.bary = torch.as_tensor(atlas.bary, dtype=torch.float32, device=device)
        self.faces = torch.as_tensor(atlas.faces, dtype=torch.long, device=device)
        self.region = torch.as_tensor(atlas.region, dtype=torch.uint8, device=device)
        self.size = int(atlas.size)
        self.vt = torch.as_tensor(atlas.vt, dtype=torch.float32, device=device)
        self.ft = torch.as_tensor(atlas.ft, dtype=torch.long, device=device)
        yx = torch.as_tensor(atlas.texel_yx, dtype=torch.long, device=device)
        self.texel_lin = yx[:, 0] * self.size + yx[:, 1]
        self._tex_cache: dict = {}

    def interp(self, vals):  # (V, C) -> (N, C)
        return (vals[self.tri] * self.bary[..., None]).sum(1)

    def normals(self, verts):
        torch = self.torch
        v = verts[self.faces]
        n = torch.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0], dim=1)
        out = torch.zeros_like(verts)
        for k in range(3):
            out.index_add_(0, self.faces[:, k], n)
        return torch.nn.functional.normalize(out, dim=1)

    def texture(self, colors):
        """Per-texel colours (N,3) -> (3,S,S) atlas image with the islands grown by 2 texels (bilinear-safe)."""
        torch = self.torch
        key = (colors.data_ptr(), tuple(colors.shape))
        hit = self._tex_cache.get(key)
        if hit is not None and hit[0] is colors:  # same tensor object (a freed temporary's address can be reused)
            return hit[1]
        S = self.size
        img = torch.zeros((S * S, 3), device=self.dev)
        have = torch.zeros(S * S, device=self.dev)
        img[self.texel_lin] = colors.float().to(self.dev)
        have[self.texel_lin] = 1.0
        img = img.view(S, S, 3).permute(2, 0, 1)[None]
        have = have.view(1, 1, S, S)
        for _ in range(2):
            num = torch.nn.functional.avg_pool2d(img * have, 3, 1, 1)
            den = torch.nn.functional.avg_pool2d(have, 3, 1, 1)
            grow = (have == 0) & (den > 0)
            img = torch.where(grow, num / den.clamp(min=1e-9), img)
            have = torch.where(grow, torch.ones_like(have), have)
        if len(self._tex_cache) > 4:
            self._tex_cache.clear()
        self._tex_cache[key] = (colors, img[0])  # the reference keeps the address from being reused while cached
        return img[0]


def project(torch, P, K, R, t, k1):
    """World points (N,3) -> pixel (N,2), depth (N,), camera-frame points; SIMPLE_RADIAL distortion."""
    Xc = P @ R.T + t
    z = Xc[:, 2]
    zc = z.clamp(min=1e-4)
    x, y = Xc[:, 0] / zc, Xc[:, 1] / zc
    d = 1 + k1 * (x * x + y * y)
    px = torch.stack([K[0, 0] * x * d + K[0, 2], K[1, 1] * y * d + K[1, 2]], 1)
    return px, z, Xc


def body_zbuffer(torch, geo, V, K, R, t, k1, H, W):
    """Exact triangle z-buffer of one posed body (avatar_raster): (H,W) depth (1e9 empty), face id, vertex px, z."""
    pxv, zv, _ = project(torch, V, K, R, t, float(k1))
    zb, fid = rasterize(torch, pxv, zv, geo.faces, H, W)
    return zb, fid, pxv, zv


def _sample_nearest(torch, img, px):
    """img (H, W) tensor, px (N,2) -> values at the rounded pixel (0 outside)."""
    H, W = img.shape
    u = px[:, 0].round().long()
    v = px[:, 1].round().long()
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    out = torch.zeros(len(px), device=img.device, dtype=img.dtype)
    out[ok] = img[v[ok], u[ok]]
    return out


def _sample_bilinear(torch, img, px):
    """img (3, H, W) float -> (N, 3)."""
    H, W = img.shape[1:]
    g = torch.stack([px[:, 0] / (W - 1) * 2 - 1, px[:, 1] / (H - 1) * 2 - 1], 1)[None, None]
    return torch.nn.functional.grid_sample(img[None], g, mode="bilinear", padding_mode="border",
                                           align_corners=True)[0, :, 0].T


def silhouette_iou(torch, zb_own, zb_other, mask):
    sil = ((zb_own < 1e8) & (zb_own <= zb_other)).float()[None, None]
    sil = torch.nn.functional.max_pool2d(sil, 5, 1, 2)
    sil = (-torch.nn.functional.max_pool2d(-sil, 5, 1, 2))[0, 0] > 0.5  # closing
    inter = (sil & mask).sum().item()
    union = (sil | mask).sum().item()
    return inter / union if union else 0.0, sil


def render(torch, geo, posed: dict, colors: dict, K, R, t, k1, H, W, device):
    """Rasterise the textured bodies -> (H,W,3) float sRGB, role index map (H,W) int (-1 background).
    colors {role: (N,3) per-texel sRGB}: looked up bilinearly in the atlas at each pixel's perspective-correct UV."""
    Kt = torch.as_tensor(np.asarray(K), dtype=torch.float32, device=device)
    Rt = torch.as_tensor(np.asarray(R), dtype=torch.float32, device=device)
    tt = torch.as_tensor(np.asarray(t), dtype=torch.float32, device=device)
    layers = []
    for r, v in posed.items():
        V = torch.as_tensor(np.asarray(v), dtype=torch.float32, device=device)
        layers.append((r,) + body_zbuffer(torch, geo, V, Kt, Rt, tt, k1, H, W))
    zs = torch.stack([lay[1] for lay in layers])
    zmin, win = zs.min(0)
    covered = zmin < EMPTY_Z
    img = torch.zeros((H, W, 3), device=device)
    idm = torch.full((H, W), -1, device=device, dtype=torch.int64)
    for n, (r, zb, fid, pxv, zv) in enumerate(layers):
        sel = covered & (win == n)
        if not sel.any():
            continue
        bary = pixel_bary(torch, pxv, zv, geo.faces, torch.where(sel, fid, torch.full_like(fid, -1)))
        ys, xs = torch.nonzero(sel, as_tuple=True)
        f = fid[ys, xs]
        uv = (geo.vt[geo.ft[f]] * bary[ys, xs][..., None]).sum(1)  # (P,2)
        g = torch.stack([2 * uv[:, 0] - 1, 1 - 2 * uv[:, 1]], 1)[None, None]
        tex = geo.texture(torch.as_tensor(colors[r], device=device))
        col = torch.nn.functional.grid_sample(tex[None], g, mode="bilinear", padding_mode="border",
                                              align_corners=False)[0, :, 0].T
        img[ys, xs] = col
        idm[ys, xs] = n
    return img.cpu().numpy(), idm.cpu().numpy()


def hair_flow_mask(data: dict, union: np.ndarray, dilate_frac: float = 0.01) -> np.ndarray | None:
    """Union of the hair roles' hair pixels (view_data hair_px, hair_mode exclude), dilated by dilate_frac x the
    dancers' height (>= 3 px): pixels the registration flow must ignore, because the hair-free texture render cannot
    match the photo's hair there (it pulled the flow - and the lead's face samples - towards her hair). None = none."""
    hair = None
    for hp in (data.get("hair_px") or {}).values():
        hair = hp > 0 if hair is None else hair | (hp > 0)
    if hair is None or not hair.any():
        return None
    ys = np.nonzero(union.any(1))[0]
    r = max(3, int(round(dilate_frac * (ys[-1] - ys[0])))) if len(ys) else 3
    return cv2.dilate(hair.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)) > 0


def align_flow(torch, geo, posed: dict, ref_colors: dict, view: dict, data: dict, cfg: TextureConfig, device):
    """Smooth 2D registration of the photo to a render of the current texture: (2,H,W) flow tensor such that the
    texel seen at x in the render is at x + flow(x) in the photo. DIS optical flow on the dancers' crop (render
    composited over the photo so the background matches), smoothed by a coverage-weighted Gaussian and clamped."""
    H, W = data["H"], data["W"]
    ren, rid = render(torch, geo, posed, ref_colors, view["K"], view["R"], view["t_w"], view["k1"], H, W, device)
    photo = data["img"]
    cov = rid >= 0
    comp = photo.copy()
    comp[cov] = (np.clip(ren[cov], 0, 1) * 255).astype(np.uint8)
    union = cov | (data["full"]["lead"] > 0) | (data["full"]["follow"] > 0)
    hair = hair_flow_mask(data, union)  # hair_mode exclude: the texture has no hair but the photo does
    if hair is not None:
        comp[hair] = photo[hair]  # identical in both images: the missing hair cannot drive the flow
        cov = cov & ~hair         # and zero weight in the smoothing / exposure match
    ys, xs = np.nonzero(union)
    zero = torch.zeros((2, H, W), device=device)
    if len(xs) < 100:
        return zero, {"status": "no dancers"}
    pad = int(0.08 * (ys.max() - ys.min()))
    x0, x1 = max(0, xs.min() - pad), min(W, xs.max() + pad + 1)
    y0, y1 = max(0, ys.min() - pad), min(H, ys.max() + pad + 1)
    s = min(1.0, 640.0 / (y1 - y0))
    a = cv2.cvtColor(comp[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
    b = cv2.cvtColor(photo[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
    if s < 1:
        a = cv2.resize(a, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        b = cv2.resize(b, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    # equalise the render's exposure to the photo inside the bodies (gains differ per camera)
    cm = cov[y0:y1, x0:x1]
    if s < 1:
        cm = cv2.resize(cm.astype(np.uint8), (a.shape[1], a.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    if cm.sum() > 100:
        ma, mb = a[cm].astype(float).mean(), b[cm].astype(float).mean()
        sa, sb = a[cm].astype(float).std() + 1e-3, b[cm].astype(float).std() + 1e-3
        a = np.clip((a.astype(float) - ma) * sb / sa + mb, 0, 255).astype(np.uint8)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    f = dis.calc(a, b, None)  # a(x) ~ b(x + f(x))
    wgt = cm.astype(np.float32)
    sig = max(1.0, cfg.flow_sigma_frac * a.shape[0])
    num = np.stack([cv2.GaussianBlur(f[..., k] * wgt, (0, 0), sig) for k in range(2)], -1)
    den = cv2.GaussianBlur(wgt, (0, 0), sig)[..., None]
    f = num / np.maximum(den, 1e-3)
    mag = np.linalg.norm(f, axis=-1, keepdims=True)
    lim = cfg.flow_max_frac * a.shape[0]
    f = f * np.minimum(1.0, lim / np.maximum(mag, 1e-6))
    f = f / s
    if s < 1:
        f = cv2.resize(f, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    full = np.zeros((H, W, 2), np.float32)
    full[y0:y1, x0:x1] = f
    on = cov[y0:y1, x0:x1]
    m = np.linalg.norm(f, axis=-1)[on] if on.any() else np.zeros(1)
    return torch.as_tensor(full, device=device).permute(2, 0, 1).contiguous(), \
        {"median_px": round(float(np.median(m)), 2), "p90_px": round(float(np.percentile(m, 90)), 2)}


# ------------------------------------------------------------------------------------------------ bake


def srgb_to_linear(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


def _lin_t(torch, x):
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _srgb_t(torch, x):
    x = x.clamp(0, 1)
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x.clamp(min=1e-8) ** (1 / 2.4) - 0.055)


class SampleStore:
    """Per texel, per camera: the top-k (weight, sRGB uint8, view id) samples."""

    def __init__(self, torch, n, n_cams, k, device, sharpen=None):
        self.sharpen = sharpen  # optional (n,) exponent of the inlier weights in the fused mean
        self.w = torch.zeros((n_cams, k, n), device=device)
        self.rgb = torch.zeros((n_cams, k, n, 3), device=device, dtype=torch.uint8)
        self.vid = torch.full((n_cams, k, n), -1, device=device, dtype=torch.int32)

    def add(self, torch, ci, w, rgb_u8, vid):
        mn, slot = self.w[ci].min(0)
        upd = torch.nonzero(w > mn).squeeze(1)
        if len(upd) == 0:
            return 0
        s = slot[upd]
        self.w[ci][s, upd] = w[upd]
        self.rgb[ci][s, upd] = rgb_u8[upd]
        self.vid[ci][s, upd] = vid
        return int(len(upd))


def bake_samples(views: list[dict], bodies: dict, atlas: UVAtlas, cfg: TextureConfig, cam_ids: list[str],
                 loader, device, ref_colors: dict | None = None, hair_region: dict | None = None) -> tuple[dict, list[dict]]:
    """Stream the views (dicts with cam, i, t, K, R, t_w, k1, ok, swap) into per-dancer SampleStores.
    loader(view) -> the _load_view dict. ref_colors {role: (N,3) sRGB tensor} = previous texture: each view is then
    registered to a render of it (smooth optical flow) before sampling. hair_region {role: (N,) bool scalp-under-hair
    texels} (hair_mode "exclude"; default the head minus the face). Returns stores and per-view stats; hair roles'
    stores also carry hair_ev (2, N) evidence and hair_pixels (avatar_hair.collect_hair_pixels)."""
    import torch
    geo = _Geo(atlas, device)
    face = (geo.region & 2) > 0
    expo = torch.where(face, torch.tensor(cfg.face_sharpen, device=device), torch.tensor(cfg.sharpen, device=device))
    stores = {r: SampleStore(torch, atlas.n, len(cam_ids), cfg.top_k, device, expo) for r in bodies}
    scalp = ((geo.region & 1) > 0) & ((geo.region & 2) == 0)
    head = (geo.region & 1) > 0
    hair_tex = {r: torch.as_tensor(np.asarray(hair_region[r]), device=device) if hair_region and r in hair_region
                else scalp for r in bodies}
    if cfg.hair_mode == "exclude":
        for r in cfg.hair_roles:
            if r in stores:
                stores[r].hair_ev = torch.zeros((2, atlas.n), device=device)
                stores[r].hair_pixels = []
    stats = []
    with ThreadPoolExecutor(4) as pool:
        futs = [pool.submit(loader, v) for v in views[:6]]
        for vi, view in enumerate(views):
            data = futs[vi].result()
            if vi + 6 < len(views):
                futs.append(pool.submit(loader, views[vi + 6]))
            st = {"cam": view["cam"], "i": view["i"], "t": view["t"], "roles": {}}
            if data.get("hair_src"):
                st["hair_src"] = data["hair_src"]
            stats.append(st)
            posed = {r: b.at(view["t"]) for r, b in bodies.items()}
            if any(p is None for p in posed.values()) or not view["ok"]:
                st["skipped"] = "outside smplx window or camera not ok"
                continue
            H, W = data["H"], data["W"]
            K = torch.as_tensor(view["K"], dtype=torch.float32, device=device)
            R = torch.as_tensor(view["R"], dtype=torch.float32, device=device)
            tv = torch.as_tensor(view["t_w"], dtype=torch.float32, device=device)
            C = -(R.T @ tv)
            geom, zb = {}, {}
            for r, (v, vel, q) in posed.items():
                V = torch.as_tensor(v, device=device)
                P = geo.interp(V)
                Nn = torch.nn.functional.normalize(geo.interp(geo.normals(V)), dim=1)
                Vel = geo.interp(torch.as_tensor(vel, device=device))
                px, z, _ = project(torch, P, K, R, tv, float(view["k1"]))
                geom[r] = (P, Nn, Vel, px, z, q)
                zb[r] = body_zbuffer(torch, geo, V, K, R, tv, view["k1"], H, W)[0]
            zall = torch.minimum(*zb.values()) if len(zb) > 1 else next(iter(zb.values()))
            img = torch.as_tensor(data["img"], device=device).permute(2, 0, 1).float() / 255.0
            flow = None
            if ref_colors is not None:
                flow, st["flow"] = align_flow(torch, geo, {r: p[0] for r, p in posed.items()}, ref_colors, view,
                                              data, cfg, device)
            margin = max(2.0, cfg.margin_frac * W)
            f = float(view["K"][0, 0])
            cw = float(cfg.cam_weight.get(view["cam"], 1.0))
            for r, (P, Nn, Vel, px, z, q) in geom.items():
                other = [o for o in geom if o != r]
                zo = zb[other[0]] if other else torch.full_like(zb[r], 1e9)
                mask_t = torch.as_tensor(data["full"][r] > 0, device=device)
                iou, _ = silhouette_iou(torch, zb[r], zo, mask_t)
                q_view = float(np.clip((iou - cfg.iou_lo) / (cfg.iou_hi - cfg.iou_lo), 0.05, 1.0))
                swap = float(view["swap"])
                q_all = q_view * np.sqrt(max(q, 0.0)) * cw * (1.0 - swap)
                st["roles"][r] = {"iou": round(iou, 3), "q_view": round(q_view, 3), "fit_q": round(q, 3),
                                  "sharp": data["sharp"][r]}
                if q_all <= 1e-3:
                    continue
                if flow is not None:  # sample the photo where the registration says this texel really is
                    px = px + _sample_bilinear(torch, flow, px)
                d = P - C
                dist = d.norm(dim=1).clamp(min=1e-3)
                dirn = d / dist[:, None]
                cos = -(Nn * dirn).sum(1)
                w_cos = ((cos - cfg.min_cos) / (1 - cfg.min_cos)).clamp(0, 1)
                pix_m = z.clamp(min=0.05) / f
                vperp = Vel - (Vel * dirn).sum(1, keepdim=True) * dirn
                blur_m = vperp.norm(dim=1) * cfg.exposure_s
                foot_mm = 1000 * torch.sqrt(pix_m ** 2 / cos.clamp(min=cfg.min_cos) + blur_m ** 2)
                eps = torch.clamp(3 * pix_m / cos.clamp(min=cfg.min_cos), min=cfg.depth_eps_m)
                zf = _sample_nearest(torch, zall, px)
                vis = (z <= zf + eps) & (z > 0.05)
                inside = (px[:, 0] >= 0) & (px[:, 0] <= W - 1) & (px[:, 1] >= 0) & (px[:, 1] <= H - 1)
                dt_own = _sample_nearest(torch, torch.as_tensor(data["dt"][r], device=device), px)
                w_mask = (dt_own / margin).clamp(0, 1)
                partner = _sample_nearest(torch, torch.as_tensor(data["partner"][r], device=device), px) > 0
                w = vis.float() * inside.float() * w_cos * w_mask * (~partner).float() / foot_mm ** 2 * q_all
                hk = (data.get("hair_keep") or {}).get(r) if cfg.hair_mode == "exclude" else None
                if r in cfg.hair_roles and hk is not None:  # hard exclusion around her hair (avatar_hair.py)
                    from .avatar_hair import collect_hair_pixels, exclude_hair
                    w = exclude_hair(torch, stores[r], w, px, hk, data["hair_px"][r], hair_tex[r], _sample_nearest)
                    hpx, hz, _ = project(torch, P[head].mean(0, keepdim=True), K, R, tv, float(view["k1"]))
                    collect_hair_pixels(stores[r].hair_pixels, data["img"], data["hair_px"][r],
                                        hpx[0].cpu().numpy(), float(hz[0]), f, vi)
                elif r in cfg.hair_roles:
                    if data.get("p_hair") is not None:  # hair colour only belongs on the scalp
                        ph = _sample_nearest(torch, torch.as_tensor(data["p_hair"], device=device), px)
                        w = torch.where(scalp, w, w * ((1 - ph) / 0.5).clamp(0, 1) ** 2)
                    else:  # fall back to the dancers stage's body mask (hair trimmed)
                        dt_body = _sample_nearest(torch, torch.as_tensor(data["dt"][r + "_body"], device=device), px)
                        w = torch.where(scalp, w, w * (dt_body / margin).clamp(0, 1))
                if view["cam"] in cfg.hd_cams:
                    w = torch.where(head, w * cfg.face_bonus, w)
                n_vis = int((w > 0).sum().item())
                st["roles"][r]["n_samples"] = n_vis
                if n_vis == 0:
                    continue
                rgb = (_sample_bilinear(torch, img, px) * 255).round().clamp(0, 255).to(torch.uint8)
                stores[r].add(torch, cam_ids.index(view["cam"]), w, rgb, vi)
            if (vi + 1) % 50 == 0:
                print(f"[texture] {vi + 1}/{len(views)} views", flush=True)
    return stores, stats


def _weighted_median(torch, x, w, q=None):
    """Along dim 0: x, w (S, N) -> (N,) weighted median and the index of the median sample. q (N,): per-column
    weighted quantile instead of 0.5 (avatar_hair: 0.75 on the follow's face / throat = the lighter, skin samples)."""
    xs, order = x.sort(0)
    ws = w.gather(0, order)
    cw = ws.cumsum(0)
    half = cw[-1:] * (0.5 if q is None else q[None])
    k = (cw < half).sum(0, keepdim=True).clamp(max=x.shape[0] - 1)
    return xs.gather(0, k)[0], order.gather(0, k)[0]


def fuse(torch, store: SampleStore, view_gain, view_sharp, cfg: TextureConfig, device, layers: bool = False):
    """Robust fusion of a SampleStore -> linear RGB (N,3), confidence (N,), inlier camera count (N,)
    [+ per-camera layers (T (C,N,3), sum of sharpened inlier weights (C,N), sum of inlier weights (C,N)) for
    avatar_hair.multiband when layers]."""
    Cn, k, N = store.w.shape
    S = Cn * k
    out_rgb = torch.zeros((N, 3), device=device)
    conf = torch.zeros(N, device=device)
    ncam = torch.zeros(N, device=device, dtype=torch.int32)
    luma = torch.tensor(LUMA, device=device)
    lay = (torch.zeros((Cn, N, 3), device=device), torch.zeros((Cn, N), device=device),
           torch.zeros((Cn, N), device=device)) if layers else None
    for a in range(0, N, cfg.chunk):
        b = min(N, a + cfg.chunk)
        w = store.w[:, :, a:b].reshape(S, -1)
        vid = store.vid[:, :, a:b].reshape(S, -1).long()
        rgb = _lin_t(torch, store.rgb[:, :, a:b].reshape(S, -1, 3).float() / 255.0)
        has = vid >= 0
        vg = view_gain[vid.clamp(min=0)]  # (S, n, 3)
        rgb = rgb * vg
        w = w * view_sharp[vid.clamp(min=0)] * has
        Y = (rgb * luma).sum(-1)
        lY = torch.log(Y + 1e-3)
        rq = getattr(store, "ref_q", None)
        med, imed = _weighted_median(torch, torch.where(w > 0, lY, torch.full_like(lY, 1e3)), w,
                                     rq[a:b] if rq is not None else None)
        ref = _srgb_t(torch, rgb.gather(0, imed[None, :, None].expand(1, -1, 3))[0])
        dcol = (_srgb_t(torch, rgb) - ref[None]).norm(dim=-1)
        inl = (w > 0) & ((lY - med[None]).abs() < cfg.inlier_log_luma) & (dcol < cfg.inlier_rgb)
        wi = torch.where(inl, w, torch.zeros_like(w))
        ws = wi ** (store.sharpen[a:b][None] if store.sharpen is not None else cfg.sharpen)
        tot = ws.sum(0)
        out_rgb[a:b] = (ws[..., None] * rgb).sum(0) / tot.clamp(min=1e-20)[:, None]
        conf[a:b] = wi.sum(0)
        camhit = inl.view(Cn, k, -1).any(1)
        ncam[a:b] = camhit.sum(0).int()
        if lay is not None:
            wsc = ws.view(Cn, k, -1)
            den = wsc.sum(1)
            lay[0][:, a:b] = (wsc[..., None] * rgb.view(Cn, k, -1, 3)).sum(1) / den.clamp(min=1e-30)[..., None]
            lay[1][:, a:b] = den
            lay[2][:, a:b] = wi.view(Cn, k, -1).sum(1)
    if lay is not None:
        return out_rgb, conf, ncam, lay
    return out_rgb, conf, ncam


def estimate_gains(torch, parts: list, view_cam, n_views, prev_gain, device, min_samples: int = 1000,
                   per_texel: int = 150_000, min_luma: float = 0.02):
    """Per-camera, then per-view linear RGB gains so each view's samples match the consensus. Exposure belongs to
    the camera, so both dancers' samples are pooled: parts = [(store, fused, conf, ncam)]. Only mid-tones count
    (linear luma > min_luma in both: black-cloth ratios are dominated by noise, black level and compression)."""
    luma = torch.tensor(LUMA, device=device)
    ratios, vids = [], []
    for store, fused, conf, ncam in parts:
        Cn, k, N = store.w.shape
        good = (ncam >= 2) & (conf > 0) & ((fused * luma).sum(1) > min_luma)
        idx = torch.nonzero(good).squeeze(1)
        if len(idx) > per_texel:
            idx = idx[torch.randperm(len(idx), device=device)[:per_texel]]
        vid = store.vid[:, :, idx].reshape(-1).long()
        w = store.w[:, :, idx].reshape(-1)
        rgb = _lin_t(torch, store.rgb[:, :, idx].reshape(-1, 3).float() / 255.0) * prev_gain[vid.clamp(min=0)]
        tgt = fused[idx][None].expand(Cn * k, -1, -1).reshape(-1, 3)
        ok = (vid >= 0) & (w > 0) & ((rgb * luma).sum(1) > min_luma) & (rgb.max(1).values < 0.95)
        ratios.append((tgt[ok] / rgb[ok].clamp(min=1e-4)).clamp(0.2, 5.0))
        vids.append(vid[ok])
    ratio, vv = torch.cat(ratios), torch.cat(vids)
    gain = prev_gain.clone()
    view_cam_t = torch.as_tensor(view_cam, device=device)
    cam_gain = {}
    for c in torch.unique(view_cam_t[vv]).tolist():
        sel = view_cam_t[vv] == c
        if sel.sum() >= min_samples:
            cam_gain[c] = ratio[sel].median(0).values
    order = torch.argsort(vv)
    vs, rs = vv[order], ratio[order]
    uniq, counts = torch.unique_consecutive(vs, return_counts=True)
    starts = torch.cumsum(counts, 0) - counts
    for v, a0, n in zip(uniq.tolist(), starts.tolist(), counts.tolist()):
        g = cam_gain.get(int(view_cam[v]), torch.ones(3, device=device))
        if n >= min_samples:
            g = g * (rs[a0:a0 + n].median(0).values / g).clamp(0.8, 1.25)
        gain[v] = gain[v] * g
    # keep the overall exposure: normalise by the sample-weighted geometric mean gain
    wv = torch.zeros(n_views, device=device).index_add_(0, vv, torch.ones_like(vv, dtype=torch.float32))
    lg = (torch.log(gain) * wv[:, None]).sum(0) / wv.sum().clamp(min=1)
    return gain / torch.exp(lg)


# ------------------------------------------------------------------------------------------------ hole filling


def _vertex_adjacency(faces: np.ndarray, n_verts: int):
    from scipy.sparse import coo_matrix
    i = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2], faces[:, 1], faces[:, 2], faces[:, 0]])
    j = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0], faces[:, 0], faces[:, 1], faces[:, 2]])
    A = coo_matrix((np.ones(len(i)), (i, j)), shape=(n_verts, n_verts)).tocsr()
    A.data[:] = 1.0
    return A


def fill_holes(atlas: UVAtlas, rgb_lin: np.ndarray, conf: np.ndarray, n_verts: int, conf_ref: float | None = None,
               iters: int = 400) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """-> filled linear RGB (N,3), alpha of measured data (N,), source (N,) uint8 0 measured, 1 mirror, 2 diffused."""
    ref = conf_ref if conf_ref is not None else (np.median(conf[conf > 0]) if (conf > 0).any() else 1.0)
    alpha = np.clip(conf / (0.05 * ref), 0, 1)
    alpha = alpha * alpha * (3 - 2 * alpha)  # smoothstep
    out = rgb_lin.copy()
    src = np.zeros(atlas.n, np.uint8)
    # 1) left/right mirror
    m = atlas.mirror
    has_m = m >= 0
    am = np.where(has_m, alpha[np.clip(m, 0, None)], 0)
    use = (alpha < 1) & (am > alpha)
    mc = rgb_lin[np.clip(m, 0, None)]
    blend = am / np.maximum(am + alpha * (1 - am), 1e-9)  # mirror fills the missing part of alpha
    out[use] = alpha[use, None] * rgb_lin[use] + (1 - alpha[use, None]) * mc[use]
    alpha2 = np.where(use, alpha + (1 - alpha) * am, alpha)
    src[use & (alpha < 0.5)] = 1
    del blend
    # 2) diffusion over the mesh vertices
    tri = atlas.faces[atlas.face]
    wv = np.zeros(n_verts)
    cv = np.zeros((n_verts, 3))
    wt = atlas.bary * alpha2[:, None] ** 2
    for kk in range(3):
        np.add.at(wv, tri[:, kk], wt[:, kk])
        for ch in range(3):
            np.add.at(cv[:, ch], tri[:, kk], wt[:, kk] * out[:, ch])
    known = wv > 1e-6
    col = np.where(known[:, None], cv / np.maximum(wv, 1e-12)[:, None], 0)
    if (~known).any() and known.any():
        A = _vertex_adjacency(atlas.faces, n_verts)
        deg = np.asarray(A.sum(1)).ravel()
        kn = known.astype(float)
        c = col.copy()
        have = kn.copy()
        for _ in range(iters):
            num = A @ (c * have[:, None])
            den = A @ have
            upd = (~known) & (den > 0)
            c[upd] = num[upd] / den[upd, None]
            have = np.where(upd, 1.0, have)
            if have.all():
                # a few smoothing passes on the unknown region only
                for _ in range(20):
                    c[~known] = (A @ c)[~known] / np.maximum(deg[~known, None], 1)
                break
        col = c
    vc = np.einsum("nkd,nk->nd", col[tri], atlas.bary)
    out = alpha2[:, None] * out + (1 - alpha2[:, None]) * vc
    src[(alpha2 < 0.5) & (src == 0)] = 2
    return out, alpha, src


def gutter(img: np.ndarray, covered: np.ndarray, px: int) -> np.ndarray:
    """Push island colours outwards by px texels (mip-map safe), fill the rest with the mean colour."""
    img = img.astype(np.float32).copy()
    have = covered.astype(np.float32)
    k = np.ones((3, 3), np.float32)
    for _ in range(px):
        num = cv2.filter2D(img * have[..., None], -1, k, borderType=cv2.BORDER_CONSTANT)
        den = cv2.filter2D(have, -1, k, borderType=cv2.BORDER_CONSTANT)
        grow = (have == 0) & (den > 0)
        img[grow] = num[grow] / den[grow, None]
        have[grow] = 1
    if (have == 0).any() and (have > 0).any():
        img[have == 0] = img[have > 0].mean(0)
    return img


# ------------------------------------------------------------------------------------------------ driver


def fuse_stores(stores: dict, stats: list[dict], view_cam: np.ndarray, cams: list[str], cfg: TextureConfig,
                device, gain=None, atlas: UVAtlas | None = None, groups: dict | None = None) -> tuple[dict, dict]:
    """Sharpness-normalised, exposure-compensated robust fusion of every role's store (gains shared across roles)
    -> ({role: (rgb_lin, conf, ncam, view_gain)}, report). gain: optional starting per-view gains (n_views, 3).
    cfg.blend "multiband" (needs atlas): the final colours are the two-band fusion of avatar_hair.multiband
    (groups {role: (N,) low-pass group per texel}; default the UV islands)."""
    import torch
    n_views = len(view_cam)
    sharp_w = {}
    for r in stores:
        sharp = np.array([s["roles"].get(r, {}).get("sharp", np.nan) for s in stats], float)
        rel = np.ones(n_views)
        for ci in range(len(cams)):
            sel = (view_cam == ci) & np.isfinite(sharp)
            if sel.sum():
                rel[sel] = np.clip(sharp[sel] / np.median(sharp[sel]), 0.5, 1.5)
        sharp_w[r] = torch.as_tensor(rel, dtype=torch.float32, device=device)
    gain = torch.ones((n_views, 3), device=device) if gain is None else torch.as_tensor(gain, device=device).float()
    fused = {r: fuse(torch, stores[r], gain, sharp_w[r], cfg, device) for r in stores}
    for _ in range(cfg.gain_rounds):
        gain = estimate_gains(torch, [(stores[r],) + tuple(fused[r]) for r in stores], view_cam, n_views, gain, device)
        fused = {r: fuse(torch, stores[r], gain, sharp_w[r], cfg, device) for r in stores}
    if cfg.blend == "multiband" and atlas is not None:
        from .avatar_hair import multiband
        for r in stores:
            rgb, conf, ncam, lay = fuse(torch, stores[r], gain, sharp_w[r], cfg, device, layers=True)
            sigma = max(1.0, cfg.band_sigma_frac * atlas.size)
            sig_h = cfg.band_sigma_high_frac * atlas.size
            fused[r] = (multiband(torch, lay, atlas, (groups or {}).get(r), stores[r].sharpen, sigma,
                                  sig_h if sig_h >= 0.5 else 0.0), conf, ncam)
            del lay
    g = gain.cpu().numpy()
    results, rep = {}, {}
    for r in stores:
        rgb, conf, ncam = fused[r]
        results[r] = (rgb.cpu().numpy(), conf.cpu().numpy(), ncam.cpu().numpy(), g)
        w = stores[r].w.sum(1).sum(1).cpu().numpy()
        rep[r] = {"camera_gain_rgb_median": {c: np.round(np.median(g[view_cam == i], 0), 3).tolist()
                                             for i, c in enumerate(cams) if (view_cam == i).any()},
                  "camera_weight_share": {c: round(float(w[i] / max(w.sum(), 1e-12)), 3) for i, c in enumerate(cams)}}
    return results, rep


def load_smplx(work: Path, role: str) -> dict:
    with np.load(work / "smplx" / f"{role}.npz", allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def model_dir_for(d: dict) -> Path:
    return SMPLX_MODELS if "locked_head" in str(d.get("model", "")) else _plain_model_dir()


def load_bodies(work: Path, device: str = "cpu", head_fix: bool = True, out: Path | None = None) -> tuple[dict, dict]:
    """Posed bodies of both dancers. FK is checked against the stored joints; when <out>/head_fix.npz (default
    avatar/) says a dancer's neck correction was adopted (head_fix=True), that dancer is posed with the corrected neck."""
    bodies, info = {}, {}
    hf = (out or work / "avatar") / "head_fix.npz"
    fix = None
    if head_fix and hf.exists():
        with np.load(hf) as z:
            fix = {k: z[k] for k in z.files}
    for r in ROLES:
        d = load_smplx(work, r)
        b = Body(d, model_dir_for(d), device)
        info[r] = {"model": str(d.get("model", "")), "fk_max_err_mm": round(b.fk_err_m * 1000, 4), "frames": len(b.t),
                   "head_fix": False}
        if b.fk_err_m > 1e-3:
            raise ValueError(f"{r}: FK misses the stored joints by {b.fk_err_m * 1000:.1f} mm (wrong model?)")
        if fix is not None and bool(fix[f"{r}_adopted"]) and len(fix[f"{r}_neck_pose"]) == len(b.t):
            d2 = dict(d)
            bp = np.asarray(d["body_pose"], float).copy()
            bp[:, 33:36] = fix[f"{r}_neck_pose"]
            d2["body_pose"] = bp
            d2.pop("joints", None)
            b2 = Body(d2, model_dir_for(d), device)
            b2.fk_err_m = b.fk_err_m
            b = b2
            info[r]["head_fix"] = True
        bodies[r] = b
    return bodies, info


def fit_head_fixes(work: Path, cams: list[str], cfg: TextureConfig, out_dir: Path | None = None) -> dict:
    """<out_dir>/head_fix.npz + report: per dancer the neck correction, its leave-one-camera-out check, adoption."""
    from .avatar_head import neck_pose_with, run_head_fix
    out, rep = {}, {}
    for r in ROLES:
        d = load_smplx(work, r)
        md = model_dir_for(d)
        kw = {"prior": cfg.head_prior, "smooth": cfg.head_smooth}
        delta, info = run_head_fix(work, d, r, md, cams, **kw)
        loo = []
        if info.get("status") == "ok" and cfg.head_fix == "auto":
            for c in cams:
                _, hi = run_head_fix(work, d, r, md, cams, holdout=c, **kw)
                if "holdout" in hi:
                    loo.append(hi["holdout"])
        ratios = np.array([h["after_cm"] / max(h["before_cm"], 1e-6) for h in loo])
        if cfg.head_fix == "on":
            adopt = info.get("status") == "ok"
        elif cfg.head_fix == "auto":
            adopt = bool(len(ratios) >= 3 and np.median(ratios) < 0.9 and (ratios < 1).mean() >= 0.75)
        else:
            adopt = False
        out[f"{r}_t"] = np.asarray(d["t"], float)
        out[f"{r}_delta_rotvec"] = delta
        out[f"{r}_neck_pose"] = neck_pose_with(d, delta)
        out[f"{r}_err_before_m"] = info.get("_before", np.full(len(delta), np.nan))
        out[f"{r}_err_after_m"] = info.get("_after", np.full(len(delta), np.nan))
        out[f"{r}_n_obs"] = info.get("_n_obs", np.zeros(len(delta), int))
        out[f"{r}_adopted"] = np.array(adopt)
        rep[r] = {**{k: v for k, v in info.items() if not k.startswith("_")}, "adopted": adopt,
                  "loo": loo, "loo_median_ratio": round(float(np.median(ratios)), 3) if len(ratios) else None,
                  "loo_improved_fraction": round(float((ratios < 1).mean()), 3) if len(ratios) else None}
        print(f"[texture] head fix {r}: face keypoints {info.get('median_err_before_cm')} -> "
              f"{info.get('median_err_after_cm')} cm, median correction {info.get('median_correction_deg')} deg, "
              f"held-out ratio {rep[r]['loo_median_ratio']} ({rep[r]['loo_improved_fraction']} of cams better) -> "
              f"{'ADOPTED' if adopt else 'not adopted'}", flush=True)
    od = out_dir or work / "avatar"
    od.mkdir(parents=True, exist_ok=True)
    np.savez(od / "head_fix.npz", **out)
    return rep


def _plain_model_dir() -> Path:
    """Directory layout smplx.create expects for the plain neutral model at WORKSPACE/smplx/SMPLX_NEUTRAL.npz."""
    from .take import WORKSPACE
    return WORKSPACE  # smplx.create(WORKSPACE, model_type='smplx') -> WORKSPACE/smplx/SMPLX_NEUTRAL.npz


def camera_views(work: Path, cams: list[str], t_lo: float, t_hi: float, step_s: float) -> tuple[list[dict], dict]:
    sim3_p = work / "cameras" / "world_from_sfm.json"
    sim3 = json.loads(sim3_p.read_text()) if sim3_p.exists() else None
    swaps = _roles_swap(work)
    views, cam_info = [], {}
    for c, i, t in _view_list(work, cams, t_lo, t_hi, step_s):
        if c not in cam_info:
            z = np.load(work / "cameras" / f"{c}.npz")
            Rw, tw = world_cameras(z["R"], z["tvec"], sim3)
            cam_info[c] = {"K": z["K"], "R": Rw, "t": tw, "k1": z["dist"] if "dist" in z.files else np.zeros(len(Rw)),
                           "ok": z["ok"] if "ok" in z.files else np.ones(len(Rw), bool),
                           "attached": bool(z["attached"]) if "attached" in z.files else False}
        ci = cam_info[c]
        views.append({"cam": c, "i": i, "t": t, "K": ci["K"][i], "R": ci["R"][i], "t_w": ci["t"][i],
                      "k1": float(ci["k1"][i]), "ok": bool(ci["ok"][i]), "swap": swaps.get(c, {}).get(i, 0.0)})
    return views, cam_info


def default_out(work: Path, cfg: TextureConfig) -> Path:
    return work / "avatar" / "smoke" if cfg.preset == "smoke" else work / "avatar"


def review_prefix(cfg: TextureConfig) -> str:
    return "avatar_smoke" if cfg.preset == "smoke" else "avatar"


def run(take: Take, cfg: TextureConfig | None = None, lock: bool = True, qa: bool = True,
        out: Path | None = None) -> dict:
    import contextlib
    import platform

    import torch

    from . import gpu
    from .env import is_pod
    cfg = cfg or TextureConfig()
    work = take.work
    out = out or default_out(work, cfg)
    out.mkdir(parents=True, exist_ok=True)
    fresh = smplx_freshness(work)
    if fresh["stale"]:
        msg = (f"[texture] smplx/: {fresh['status']} (smplx {fresh.get('smplx_mtime')}, world_from_sfm "
               f"{fresh.get('world_from_sfm_mtime')})")
        if not cfg.allow_stale:
            raise RuntimeError(msg + " - refusing to bake; --allow-stale (or --preset smoke) to run anyway")
        print(msg + " - baking anyway (allow_stale): bodies may be misaligned", flush=True)
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    t_start = time.time()
    cams = list(cfg.cams) if cfg.cams else sorted(p.stem for p in (work / "cameras").glob("[0-9][0-9].npz"))
    head_rep = fit_head_fixes(work, cams, cfg, out) if cfg.head_fix != "off" else {}
    if cfg.head_fix == "off":
        (out / "head_fix.npz").unlink(missing_ok=True)
    bodies, body_info = load_bodies(work, out=out)
    t_lo = max(b.t[0] for b in bodies.values())
    t_hi = min(b.t[-1] for b in bodies.values())
    if cfg.t_range:
        t_lo, t_hi = t_lo + cfg.t_range[0], min(t_hi, t_lo + cfg.t_range[1])
    views, cam_info = camera_views(work, cams, t_lo, t_hi, cfg.step_s)
    atlas = build_atlas(SMPLX_MODELS, cfg.size)
    print(f"[texture] preset {cfg.preset}: {len(views)} views from {len(cams)} cameras over t={t_lo:.2f}..{t_hi:.2f} s, "
          f"atlas {cfg.size}^2 with {atlas.n} texels, "
          "FK " + ", ".join(f"{r} {v['fk_max_err_mm']} mm" for r, v in body_info.items()), flush=True)
    luts, hair_rep = {}, {}
    from . import avatar_hair as ahair
    face_proj = {r: ahair.FaceProjector(bodies[r], model_dir_for(load_smplx(work, r))) for r in cfg.hair_roles}         if cfg.hair_mode == "exclude" else {}
    if face_proj:  # hair colour model that knows her face / throat is skin (the hair masks often cover her face)
        allv = {(v["cam"], v["i"]): v for v in camera_views(work, cams, t_lo - 1.0, t_hi + 1.0, 1e-4)[0]}
        r0 = cfg.hair_roles[0]
        fmask = lambda c: (lambda i, H, W: face_proj[r0].mask(allv[(c, i)], H, W) if (c, i) in allv else None)
        def model(c):
            lut, rep = hair_model(work, c, cfg.hair_frame_step)
            flut, frep = ahair.hair_lut_face_aware(work, c, cfg.hair_frame_step, fmask(c), r0)
            return ((lut, flut) if lut is not None else None), {**rep, "face_aware": frep}
    else:
        model = lambda c: hair_model(work, c, cfg.hair_frame_step)
    if cfg.hair_roles:
        with ThreadPoolExecutor(4) as pool:
            for c, (lut, hr) in zip(cams, pool.map(model, cams)):
                luts[c], hair_rep[c] = lut, hr
        print("[texture] hair colour model (recall / false positives on other dancer pixels"
              + (" / her face+throat pixels called hair" if face_proj else "") + "): " + ", ".join(
                  f"{c} {h.get('hair_recall', '-')}/{h.get('other_false_pos', '-')}"
                  + (f"/{h.get('face_aware', {}).get('face_marked_hair', '-')}" if face_proj else "")
                  for c, h in hair_rep.items()), flush=True)
    loader = lambda v: _load_view(work, v["cam"], v["i"], cfg.margin_frac, luts.get(v["cam"]), cfg,
                                  (lambda r, H, W, v=v: face_proj[r].mask(v, H, W)) if face_proj else None)
    regions = ahair.texel_regions(atlas, SMPLX_MODELS)
    islands = ahair.island_ids(atlas)
    hair_reg = {r: ahair.prior_region(regions) for r in cfg.hair_roles} if cfg.hair_mode == "exclude" else {}
    hair_pix, hair_rep_passes, prev_ev, last_ev, guard_rep = {}, [], {}, {}, {}
    hair_cov = hair_mask_coverage(work, views, cfg.hair_roles) if cfg.hair_mode == "exclude" else {}
    for r, hc in hair_cov.items():  # the fallback (full - body, or nothing) silently weakens the hair exclusion
        msg = (f"[texture] {r} hair masks: {hc['mask']} real / {hc['derived']} derived (full - body) / {hc['none']} "
               f"none of {hc['views_with_dancer_mask']} views with her mask ({hc['real_fraction']:.0%} real)")
        print(msg, flush=True)
        if hc["views_with_dancer_mask"] and hc["real_fraction"] < cfg.hair_mask_min_frac:
            if not cfg.allow_missing_hair_masks:
                raise RuntimeError(msg + f" < {cfg.hair_mask_min_frac:.0%}: run the dancers stage (hair split) for "
                                         "these frames first, or --allow-missing-hair-masks")
            print("[texture] WARNING: too few real hair masks; baking anyway (allow_missing_hair_masks)", flush=True)
    ctx = gpu.exclusive("texture") if (lock and device == "cuda") else contextlib.nullcontext()
    report: dict = {"config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(cfg).items()},
                    "host": {"pod": is_pod(), "platform": platform.platform(), "device": device,
                             "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None},
                    "smplx_freshness": fresh, "t_window": [float(t_lo), float(t_hi)],
                    "cams": cams, "n_views": len(views), "bodies": body_info, "atlas_texels": atlas.n,
                    "hair_model": hair_rep, "head_fix": head_rep, "passes": [], "roles": {r: {} for r in ROLES},
                    "hair_masks": hair_cov}
    view_cam = np.array([cams.index(v["cam"]) for v in views])
    n_verts = bodies["lead"].verts.shape[1]
    ref, final = None, None
    with ctx:
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for pas in range(1 + max(0, cfg.align_passes)):
            t0 = time.time()
            stores, stats = bake_samples(views, bodies, atlas, cfg, cams, loader, device, ref_colors=ref,
                                         hair_region={r: h[0] for r, h in hair_reg.items()} or None)
            for r in hair_reg:  # face core: lighter-quantile reference only where hair was seen over the face
                e0 = prev_ev.get(r)
                frac = np.where(e0[1] > 0, e0[0] / np.maximum(e0[1], 1e-20), 0.0) if e0 is not None                     else np.zeros(atlas.n)
                q = np.where(regions["skin_pref"], 0.5 + (cfg.skin_ref_q - 0.5) * np.clip(frac / 0.3, 0, 1), 0.5)
                stores[r].ref_q = torch.as_tensor(q, dtype=torch.float32, device=device)
            groups = {r: ahair.texel_groups(islands, hair_reg[r][0] if r in hair_reg else None) for r in ROLES}
            results, fusion_rep = fuse_stores(stores, stats, view_cam, cams, cfg, device, gain=None, atlas=atlas,
                                              groups=groups)
            ev = {r: stores[r].hair_ev.cpu().numpy() for r in hair_reg if getattr(stores[r], "hair_ev", None) is not None}
            hair_pix = {r: stores[r].hair_pixels for r in hair_reg if getattr(stores[r], "hair_pixels", None)}
            del stores
            if device == "cuda":
                torch.cuda.empty_cache()
            conf_ref = float(np.median(np.concatenate([results[r][1][results[r][1] > 0] for r in ROLES])))
            final = {}
            for r in ROLES:
                rgb, conf, ncam, g = results[r]
                if r in hair_reg:  # scalp and body filled separately, class-aware body fill (avatar_hair.py)
                    conf, bad, skin_only, gst = ahair.fill_guards(rgb, conf, regions, hair_reg[r][0], ev.get(r),
                                                                  cfg.fill_ev_bad)
                    filled, alpha, src = ahair.fill_region_aware(atlas, rgb, conf, n_verts, conf_ref, *hair_reg[r],
                                                                 classes=cfg.fill_classes, bad=bad, skin_only=skin_only)
                    gst["fill"] = dict(getattr(ahair.fill_classed, "last_stats", {}) or {})
                    guard_rep[r] = gst
                    hair_rep_passes.append({"pass": pas, "role": r, **{k: v for k, v in gst.items() if k != "fill"}})
                else:
                    filled, alpha, src = fill_holes(atlas, rgb, conf, n_verts, conf_ref)
                final[r] = (linear_to_srgb(filled).astype(np.float32), alpha, src, conf, ncam, g)
            used_reg = dict(hair_reg)  # the scalp region this pass sampled and filled with
            for r, e in ev.items():  # the next pass uses this pass's hair evidence
                hb, hs, hst = ahair.hair_region_from_evidence(atlas, e[0], e[1], regions,
                                                              margin_m=cfg.hairline_margin_m)
                hair_reg[r] = (hb, hs)
                hair_rep_passes.append({"pass": pas, "role": r, **hst})
            prev_ev, last_ev = dict(ev), dict(ev)
            fl = [s["flow"]["median_px"] for s in stats if "flow" in s and "median_px" in s["flow"]]
            report["passes"].append({"pass": pas, "aligned": ref is not None, "seconds": round(time.time() - t0, 1),
                                     "flow_median_px": round(float(np.median(fl)), 2) if fl else None,
                                     "flow_p90_of_view_medians_px": round(float(np.percentile(fl, 90)), 2) if fl else None})
            print(f"[texture] pass {pas}{' (aligned)' if ref is not None else ''}: {time.time() - t0:.0f} s"
                  + (f", flow median {np.median(fl):.1f} px" if fl else ""), flush=True)
            ref = {r: torch.as_tensor(final[r][0], device=device) for r in ROLES}
        for r in ROLES:
            report["roles"][r].update(fusion_rep[r])
        del ref
        if device == "cuda":
            report["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    for r in ROLES:
        srgb, alpha, src, conf, ncam, g = final[r]
        img = atlas.to_image(srgb)
        img = gutter(img, atlas.to_image(np.ones(atlas.n, np.uint8)) > 0, cfg.gutter_px)
        cv2.imwrite(str(out / f"{r}_albedo.png"), (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)[:, :, ::-1])
        cov = np.zeros((atlas.n, 3), np.uint8)
        cov[:, 0] = np.clip(255 * alpha, 0, 255).astype(np.uint8)        # R: measured-data alpha
        cov[:, 1] = np.clip(ncam * 32, 0, 255).astype(np.uint8)          # G: inlier cameras x 32
        cov[:, 2] = np.array([0, 128, 255], np.uint8)[src]               # B: 0 measured, 128 mirror, 255 diffused
        cv2.imwrite(str(out / f"{r}_coverage.png"), atlas.to_image(cov)[:, :, ::-1])
        extra = {}
        if r in used_reg:  # scalp-under-hair region (soft, 255 = hair) the bake used: roots the Unity hair
            hsoft = (np.clip(used_reg[r][1], 0, 1) * 255 + 0.5).astype(np.uint8)
            extra["hair_region"] = hsoft
            cv2.imwrite(str(out / f"{r}_hair_region.png"), atlas.to_image(hsoft))
            if r in last_ev:  # where her hanging hair covered the body in the photos (NOT where hair roots)
                hc8 = (np.clip(ahair.hair_cover(*last_ev[r]), 0, 1) * 255 + 0.5).astype(np.uint8)
                extra["hair_cover"] = hc8
                cv2.imwrite(str(out / f"{r}_hair_cover.png"), atlas.to_image(hc8))
            if r in guard_rep:
                report["roles"][r]["fill_guards"] = guard_rep[r]
            scalp = used_reg[r][0]
            report["roles"][r]["hair_sample"] = ahair.hair_sample(
                hair_pix.get(r, []), g, out, r, scalp_srgb=srgb[scalp].mean(0) if scalp.any() else None)
        np.savez_compressed(out / f"{r}_texels.npz", rgb_srgb=srgb.astype(np.float16), alpha=alpha.astype(np.float16),
                            conf=conf.astype(np.float32), ncam=ncam.astype(np.uint8), source=src,
                            view_gain=g.astype(np.float32), **extra)
        report["roles"][r]["texture_metrics"] = ahair.texture_metrics(srgb, regions, used_reg[r][0] if r in used_reg
                                                                      else None)
        meas = alpha > 0.5
        report["roles"][r].update({
            "measured_fraction": round(float(meas.mean()), 4),
            "mirror_fraction": round(float((src == 1).mean()), 4),
            "diffused_fraction": round(float((src == 2).mean()), 4),
            "median_inlier_cams": float(np.median(ncam[meas])) if meas.any() else 0.0,
            "face_measured_fraction": round(float(meas[(atlas.region & 2) > 0].mean()), 4),
            "median_view_iou": round(float(np.median([s["roles"][r]["iou"] for s in stats
                                                       if r in s.get("roles", {})] or [np.nan])), 3),
        })
    report["hair_region"] = hair_rep_passes
    report["uv_mesh"] = write_uv_mesh(out / "uv_mesh.npz", SMPLX_MODELS)
    report["views"] = stats
    report["runtime_s"] = {"bake": round(time.time() - t_start, 1)}
    (out / "report.json").write_text(json.dumps(report, indent=1, default=float))
    if qa:
        from .avatar_review import render_review
        paths = render_review(take, cfg, bodies=bodies, views=views, stats=stats, device=device, lock=lock, out=out,
                              prefix=review_prefix(cfg))
        report = json.loads((out / "report.json").read_text())  # render_review adds review_metrics
        report["review"] = [str(p) for p in paths]
    report["runtime_s"]["total"] = round(time.time() - t_start, 1)
    (out / "report.json").write_text(json.dumps(report, indent=1, default=float))
    return report
