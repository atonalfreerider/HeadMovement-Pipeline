"""Photoreal per-dancer albedo textures for the SMPL-X avatars, baked from the multi-view footage.

    python -m dancecap texture larissa_kadu            -> work/<take>/avatar/{lead,follow}_albedo.png (+ coverage, uv mesh)

Inputs: smplx/{lead,follow}.npz (world frame, MAMMA locked-head SMPL-X; FK is checked against the stored joints),
cameras/<cam>.npz + world_from_sfm.json, frames/<cam>, dancers/<cam>/{masks,body} (lead/follow), roles.json.

Method (texel-centric, one S x S atlas per dancer on the SMPL-X UV layout, see avatar_uv.py):
  0. Head fix (avatar_head.py): MAMMA's head orientation misses the COCO face keypoints by 1-3 eye distances; a
     smoothed neck rotation is fitted to them and adopted per dancer when it generalises to held-out cameras.
  1. Every texel is a surface point (face + barycentrics). For each sampled camera frame (every `step_s` of reference
     time per camera) both bodies are posed at the frame's own timestamp (linear between 30 Hz grid frames).
  2. Visibility: both bodies' texel points are splatted into a per-pixel z-buffer (densely: texels are ~1 mm,
     pixels >= 1.5 mm), closed with a 3x3 min filter; a texel is visible if it lies within eps of the front surface.
     The same z-buffers give each dancer's model silhouette -> IoU with the SAM mask = per-view fit quality.
  3. Weight of a sample = visibility x facing (cos) x resolution (1 / texel footprint^2, footprint from depth/focal,
     foreshortening and motion blur of the texel during an assumed exposure) x distance inside the dancer's OWN full
     mask x not in the partner's mask x silhouette IoU x MAMMA fit quality x identity confidence (roles.json) x
     camera trust; head texels get a bonus in the HD cameras (02, 06). The follow's non-scalp texels (face included)
     reject hair-coloured pixels: per-camera Lab histogram model P(hair | colour) learned from dancers/<cam>/hair
     masks vs the rest of both dancers (the dancers stage's body masks miss the darker hair).
  4. Each texel keeps its top-k samples PER CAMERA (so no single camera dominates the robust statistics).
  5. Fusion in linear RGB: per-camera then per-view exposure/white-balance gains against the consensus, pooled over
     both dancers, mid-tones only (two rounds); per-view sharpness (Laplacian variance in the mask vs the camera's
     median); weighted median of luminance + colour inlier test; weighted mean of the inliers with weights^1.5
     (face: ^4, so the best-registered high-resolution view dominates instead of a ghosted average).
  6. Registration pass(es): every view is re-sampled through a smooth optical flow (DIS, coverage-weighted Gaussian,
     clamped) from a render of the previous texture to the photo, which absorbs the fit's few-pixel misalignment.
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
    align_passes: int = 1           # extra passes that register each view to a render of the previous texture
    flow_sigma_frac: float = 0.02   # smoothing of the alignment flow (fraction of the dancers' crop height)
    flow_max_frac: float = 0.06     # alignment flow clamp (fraction of the crop height)
    head_fix: str = "auto"          # neck-rotation fix to the COCO face keypoints (avatar_head.py): auto|on|off;
                                    # auto adopts it per dancer when leave-one-camera-out says it generalises
    head_prior: float = 0.06
    head_smooth: float = 0.3
    device: str | None = None       # default cuda when available
    chunk: int = 400_000            # texels per fusion chunk


# ------------------------------------------------------------------------------------------------ views


def _view_list(work: Path, cams: list[str], t_lo: float, t_hi: float, step_s: float) -> list[tuple[str, int, float]]:
    out = []
    for c in cams:
        meta = json.loads((work / "frames" / c / "frames.json").read_text())
        last = None
        for i, fr in enumerate(meta["frames"]):
            t = float(fr["t"])
            if t < t_lo or t > t_hi:
                continue
            slot = int(np.floor((t - t_lo) / step_s))
            if slot != last:
                out.append((c, i, t))
                last = slot
    return out


def _read_mask(p: Path, shape) -> np.ndarray:
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    return np.zeros(shape, np.uint8) if m is None else (m > 127).astype(np.uint8)


def _load_view(work: Path, cam: str, i: int, margin_frac: float, hair_lut: np.ndarray | None = None) -> dict:
    meta_file = work / "frames" / cam / f"{i:06d}.jpg"
    img = cv2.imread(str(meta_file), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(meta_file)
    H, W = img.shape[:2]
    dd = work / "dancers" / cam
    full = {r: _read_mask(dd / "masks" / f"mask_{i:04d}_{PIDS[r]:02d}.png", (H, W)) for r in ROLES}
    body = {r: _read_mask(dd / "body" / f"mask_{i:04d}_{PIDS[r]:02d}.png", (H, W))
            if (dd / "body" / f"mask_{i:04d}_{PIDS[r]:02d}.png").exists() else full[r] for r in ROLES}
    return view_data(img[:, :, ::-1].copy(), full, body, margin_frac, hair_lut)


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
              hair_lut: np.ndarray | None = None) -> dict:
    """Per-view inputs of bake_samples from an RGB image and {role: uint8 0/1} full/body masks."""
    H, W = img_rgb.shape[:2]
    margin = max(2.0, margin_frac * W)
    dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(np.ceil(margin)) + 1,) * 2)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    out = {"img": img_rgb, "H": H, "W": W, "full": full, "dt": {}, "partner": {}, "sharp": {},
           "p_hair": hair_probability(img_rgb, hair_lut) if hair_lut is not None else None}
    for r in ROLES:
        out["dt"][r] = cv2.distanceTransform(full[r], cv2.DIST_L2, 3)
        out["dt"][r + "_body"] = out["dt"][r] if body[r] is full[r] or np.array_equal(body[r], full[r]) \
            else cv2.distanceTransform(body[r], cv2.DIST_L2, 3)
        out["partner"][r] = cv2.dilate(full["follow" if r == "lead" else "lead"], dil)
        core = out["dt"][r] > 3
        out["sharp"][r] = float(lap[core].var()) if core.sum() > 200 else float("nan")
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


def project(torch, P, K, R, t, k1):
    """World points (N,3) -> pixel (N,2), depth (N,), camera-frame points; SIMPLE_RADIAL distortion."""
    Xc = P @ R.T + t
    z = Xc[:, 2]
    zc = z.clamp(min=1e-4)
    x, y = Xc[:, 0] / zc, Xc[:, 1] / zc
    d = 1 + k1 * (x * x + y * y)
    px = torch.stack([K[0, 0] * x * d + K[0, 2], K[1, 1] * y * d + K[1, 2]], 1)
    return px, z, Xc


def zbuffer(torch, px, z, H, W, valid=None):
    """Splat points into an (H, W) min-depth buffer (1e9 = empty)."""
    u = px[:, 0].round().long()
    v = px[:, 1].round().long()
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z > 0.05)
    if valid is not None:
        ok &= valid
    zb = torch.full((H * W,), 1e9, device=px.device, dtype=torch.float32)
    zb.scatter_reduce_(0, (v * W + u)[ok], z[ok].float(), reduce="amin", include_self=True)
    return zb.view(H, W)


def minfilter(torch, zb, r=1):
    return -torch.nn.functional.max_pool2d(-zb[None, None], 2 * r + 1, 1, r)[0, 0]


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
    """Splat-render textured bodies -> (H,W,3) float sRGB, role index map (H,W) int (-1 background)."""
    cols, ids, pxs, zs = [], [], [], []
    Kt = torch.as_tensor(np.asarray(K), dtype=torch.float32, device=device)
    Rt = torch.as_tensor(np.asarray(R), dtype=torch.float32, device=device)
    tt = torch.as_tensor(np.asarray(t), dtype=torch.float32, device=device)
    for n, (r, v) in enumerate(posed.items()):
        P = geo.interp(torch.as_tensor(v, device=device))
        px, z, _ = project(torch, P, Kt, Rt, tt, float(k1))
        pxs.append(px)
        zs.append(z)
        cols.append(colors[r])
        ids.append(torch.full((len(z),), n, device=device, dtype=torch.int64))
    px, z, col, rid = torch.cat(pxs), torch.cat(zs), torch.cat(cols), torch.cat(ids)
    zb = zbuffer(torch, px, z, H, W)
    u, v = px[:, 0].round().long(), px[:, 1].round().long()
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z > 0.05)
    lin = (v * W + u).clamp(0, H * W - 1)
    win = ok & (z <= zb.view(-1)[lin] + 1e-5)
    img = torch.zeros((H * W, 3), device=device)
    idm = torch.full((H * W,), -1, device=device, dtype=torch.int64)
    img[lin[win]] = col[win].float()
    idm[lin[win]] = rid[win]
    img, idm = img.view(H, W, 3), idm.view(H, W)
    # close pinholes: fill empty pixels inside the 3x3 morphological closing of the covered pixels
    have = (idm >= 0).float()
    dil = torch.nn.functional.max_pool2d(have[None, None], 3, 1, 1)
    cover = (-torch.nn.functional.max_pool2d(-dil, 3, 1, 1))[0, 0] > 0.5
    num = torch.nn.functional.avg_pool2d((img * have[..., None]).permute(2, 0, 1)[None], 3, 1, 1)[0].permute(1, 2, 0)
    den = torch.nn.functional.avg_pool2d(have[None, None], 3, 1, 1)[0, 0]
    hole = cover & (have == 0) & (den > 0)
    img[hole] = num[hole] / den[hole][:, None]
    zid = idm.clone()
    zid[hole] = torch.nn.functional.max_pool2d(idm.float()[None, None], 3, 1, 1)[0, 0].long()[hole]
    return img.cpu().numpy(), zid.cpu().numpy()


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
                 loader, device, ref_colors: dict | None = None) -> tuple[dict, list[dict]]:
    """Stream the views (dicts with cam, i, t, K, R, t_w, k1, ok, swap) into per-dancer SampleStores.
    loader(view) -> the _load_view dict. ref_colors {role: (N,3) sRGB tensor} = previous texture: each view is then
    registered to a render of it (smooth optical flow) before sampling. Returns stores and per-view stats."""
    import torch
    geo = _Geo(atlas, device)
    face = (geo.region & 2) > 0
    expo = torch.where(face, torch.tensor(cfg.face_sharpen, device=device), torch.tensor(cfg.sharpen, device=device))
    stores = {r: SampleStore(torch, atlas.n, len(cam_ids), cfg.top_k, device, expo) for r in bodies}
    scalp = ((geo.region & 1) > 0) & ((geo.region & 2) == 0)
    head = (geo.region & 1) > 0
    stats = []
    with ThreadPoolExecutor(4) as pool:
        futs = [pool.submit(loader, v) for v in views[:6]]
        for vi, view in enumerate(views):
            data = futs[vi].result()
            if vi + 6 < len(views):
                futs.append(pool.submit(loader, views[vi + 6]))
            st = {"cam": view["cam"], "i": view["i"], "t": view["t"], "roles": {}}
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
            geom = {}
            for r, (v, vel, q) in posed.items():
                V = torch.as_tensor(v, device=device)
                P = geo.interp(V)
                Nn = torch.nn.functional.normalize(geo.interp(geo.normals(V)), dim=1)
                Vel = geo.interp(torch.as_tensor(vel, device=device))
                px, z, _ = project(torch, P, K, R, tv, float(view["k1"]))
                geom[r] = (P, Nn, Vel, px, z, q)
            zb = {r: zbuffer(torch, g[3], g[4], H, W) for r, g in geom.items()}
            zall = minfilter(torch, torch.minimum(*zb.values()) if len(zb) > 1 else next(iter(zb.values())))
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
                if r in cfg.hair_roles:
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


def _weighted_median(torch, x, w):
    """Along dim 0: x, w (S, N) -> (N,) weighted median and the index of the median sample."""
    xs, order = x.sort(0)
    ws = w.gather(0, order)
    cw = ws.cumsum(0)
    half = cw[-1:] * 0.5
    k = (cw < half).sum(0, keepdim=True).clamp(max=x.shape[0] - 1)
    return xs.gather(0, k)[0], order.gather(0, k)[0]


def fuse(torch, store: SampleStore, view_gain, view_sharp, cfg: TextureConfig, device):
    """Robust fusion of a SampleStore -> linear RGB (N,3), confidence (N,), inlier camera count (N,)."""
    Cn, k, N = store.w.shape
    S = Cn * k
    out_rgb = torch.zeros((N, 3), device=device)
    conf = torch.zeros(N, device=device)
    ncam = torch.zeros(N, device=device, dtype=torch.int32)
    luma = torch.tensor(LUMA, device=device)
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
        med, imed = _weighted_median(torch, torch.where(w > 0, lY, torch.full_like(lY, 1e3)), w)
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
                device, gain=None) -> tuple[dict, dict]:
    """Sharpness-normalised, exposure-compensated robust fusion of every role's store (gains shared across roles)
    -> ({role: (rgb_lin, conf, ncam, view_gain)}, report). gain: optional starting per-view gains (n_views, 3)."""
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


def load_bodies(work: Path, device: str = "cpu", head_fix: bool = True) -> tuple[dict, dict]:
    """Posed bodies of both dancers. FK is checked against the stored joints; when avatar/head_fix.npz says a
    dancer's neck correction was adopted (head_fix=True), that dancer is posed with the corrected neck."""
    bodies, info = {}, {}
    hf = work / "avatar" / "head_fix.npz"
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


def fit_head_fixes(work: Path, cams: list[str], cfg: TextureConfig) -> dict:
    """avatar/head_fix.npz + report: per dancer the neck correction, its leave-one-camera-out check, adoption."""
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
    (work / "avatar").mkdir(parents=True, exist_ok=True)
    np.savez(work / "avatar" / "head_fix.npz", **out)
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


def run(take: Take, cfg: TextureConfig | None = None, lock: bool = True, qa: bool = True) -> dict:
    import contextlib

    import torch

    from . import gpu
    cfg = cfg or TextureConfig()
    work = take.work
    out = work / "avatar"
    out.mkdir(parents=True, exist_ok=True)
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    t_start = time.time()
    cams = list(cfg.cams) if cfg.cams else sorted(p.stem for p in (work / "cameras").glob("[0-9][0-9].npz"))
    head_rep = fit_head_fixes(work, cams, cfg) if cfg.head_fix != "off" else {}
    if cfg.head_fix == "off":
        (work / "avatar" / "head_fix.npz").unlink(missing_ok=True)
    bodies, body_info = load_bodies(work)
    t_lo = max(b.t[0] for b in bodies.values())
    t_hi = min(b.t[-1] for b in bodies.values())
    views, cam_info = camera_views(work, cams, t_lo, t_hi, cfg.step_s)
    atlas = build_atlas(SMPLX_MODELS, cfg.size)
    print(f"[texture] {len(views)} views from {len(cams)} cameras, atlas {cfg.size}^2 with {atlas.n} texels, "
          "FK " + ", ".join(f"{r} {v['fk_max_err_mm']} mm" for r, v in body_info.items()), flush=True)
    luts, hair_rep = {}, {}
    if cfg.hair_roles:
        with ThreadPoolExecutor(4) as pool:
            for c, (lut, hr) in zip(cams, pool.map(lambda c: hair_model(work, c, cfg.hair_frame_step), cams)):
                luts[c], hair_rep[c] = lut, hr
        print("[texture] hair colour model (recall / false positives on other dancer pixels): " + ", ".join(
            f"{c} {h.get('hair_recall', '-')}/{h.get('other_false_pos', '-')}" for c, h in hair_rep.items()), flush=True)
    loader = lambda v: _load_view(work, v["cam"], v["i"], cfg.margin_frac, luts.get(v["cam"]))
    ctx = gpu.exclusive("texture") if (lock and device == "cuda") else contextlib.nullcontext()
    report: dict = {"config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(cfg).items()},
                    "cams": cams, "n_views": len(views), "bodies": body_info, "atlas_texels": atlas.n,
                    "hair_model": hair_rep, "head_fix": head_rep, "passes": [], "roles": {r: {} for r in ROLES}}
    view_cam = np.array([cams.index(v["cam"]) for v in views])
    n_verts = bodies["lead"].verts.shape[1]
    ref, gain0, final = None, None, None
    with ctx:
        for pas in range(1 + max(0, cfg.align_passes)):
            t0 = time.time()
            stores, stats = bake_samples(views, bodies, atlas, cfg, cams, loader, device, ref_colors=ref)
            results, fusion_rep = fuse_stores(stores, stats, view_cam, cams, cfg, device, gain=None)
            del stores
            if device == "cuda":
                torch.cuda.empty_cache()
            conf_ref = float(np.median(np.concatenate([results[r][1][results[r][1] > 0] for r in ROLES])))
            final = {}
            for r in ROLES:
                rgb, conf, ncam, g = results[r]
                filled, alpha, src = fill_holes(atlas, rgb, conf, n_verts, conf_ref)
                final[r] = (linear_to_srgb(filled).astype(np.float32), alpha, src, conf, ncam, g)
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
        np.savez_compressed(out / f"{r}_texels.npz", rgb_srgb=srgb.astype(np.float16), alpha=alpha.astype(np.float16),
                            conf=conf.astype(np.float32), ncam=ncam.astype(np.uint8), source=src,
                            view_gain=g.astype(np.float32))
        meas = alpha > 0.5
        report["roles"][r].update({
            "measured_fraction": round(float(meas.mean()), 4),
            "mirror_fraction": round(float((src == 1).mean()), 4),
            "diffused_fraction": round(float((src == 2).mean()), 4),
            "median_inlier_cams": float(np.median(ncam[meas])) if meas.any() else 0.0,
            "face_measured_fraction": round(float(meas[(atlas.region & 2) > 0].mean()), 4),
        })
    report["uv_mesh"] = write_uv_mesh(out / "uv_mesh.npz", SMPLX_MODELS)
    report["views"] = stats
    report["runtime_s"] = {"bake": round(time.time() - t_start, 1)}
    (out / "report.json").write_text(json.dumps(report, indent=1, default=float))
    if qa:
        from .avatar_review import render_review
        paths = render_review(take, cfg, bodies=bodies, views=views, stats=stats, device=device, lock=lock)
        report = json.loads((out / "report.json").read_text())  # render_review adds review_metrics
        report["review"] = [str(p) for p in paths]
    report["runtime_s"]["total"] = round(time.time() - t_start, 1)
    (out / "report.json").write_text(json.dumps(report, indent=1, default=float))
    return report
