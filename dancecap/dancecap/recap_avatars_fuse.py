"""Fuse several texture bakes of one dancer into one albedo, and paint out what the video cannot give (agent T2, class recap).

    python -m dancecap.recap_avatars texture-fuse --windows tx3,tx4,tx5 [--paint avatar/paint_out.json] [--dry-run]

WHY. `dancecap texture` bakes ONE window of the footage. Whatever happens in that window is in the albedo: her long hair swinging over her
white top, the partner's black sleeve in an embrace, a view that was misregistered for a second. Two or three windows of the same class
(inside chunk interiors, away from the seams) see those contaminations in DIFFERENT places, so a per-texel, confidence-weighted fusion with
an outlier test keeps what the windows agree on and drops what only one of them shows. The bake keeps, per dancer, `rgb_srgb` (the filled
albedo), `alpha` (measured-data weight 0..1), `conf` (sum of the inlier sample weights), `ncam` (inlier cameras) and, for her, `hair_cover`
(the share of the weight that saw her hair over the texel) in <role>_texels.npz; that is all the fusion needs (no photo is read again).

METHOD (per role; texels of the shared SMPL-X UV atlas, so every window's arrays line up index by index):
  1. exposure: each window has its own overall exposure (the bake normalises per window); per-channel gains from the texels that are
     measured in a window and in the others (median ratio of linear RGB, mid-tones only, clamped 0.8 .. 1.25, geometric mean 1);
  2. quality per window q = alpha x sqrt(conf / conf_ref, <= 4) x min(ncam, 3) / 3 x (1 - 0.7 hair_cover): a texel measured by many good
     samples in 3 cameras wins over a hole-filled one; texels that her hair covered in the photos count less;
  3. two bands: the LOW band (a Gaussian of sigma `low_sigma` texels, normalised by the covered texels) is fused with weights q^2 and a soft
     outlier factor exp(-(d / tau)^2), d = the distance of a window's low band to the first estimate (a streak that only one window shows
     is dropped); on `prefer_light` texels (her white top / neck: hair, a dark sleeve and shadows only darken it) a window that is much
     darker than the estimate gets a further factor. The HIGH band (detail) comes from the window with the largest smoothed q (soft argmax,
     power 4): no ghosting from two slightly different registrations;
  4. holes: texels no window measured keep the (alpha-weighted) filled colour; the gutter is rebuilt as the bake does.

PAINT-OUT. Things that are on the dancer in the video but are not part of the avatar (owner default: no headset, no belt pack, as in the demo)
bake into the albedo as dark patches. `paint_out` replaces a region by (1) a DONOR albedo of the same person on the same UV layout (the demo's
albedo: her own clean ears and jeans; `kind: donor`), colour-matched to the ring around the region (per-channel linear gain, smoothly
feathered), or (2) `kind: inpaint` (diffusion from the ring). A region is `centre` (a point of the REST body, metres) + `radius_m`, optionally
only the dark (`luma_max`) and / or grey (`sat_max`: (max - min) / max of the sRGB colour) texels of it, grown by `grow_texels`. The list lives in avatar/paint_out.json (likeness-adjacent, git-ignored).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)
Q_FLOOR = 2e-3
# window selection sharpness per role: her top carries a print and hair residue, so the cleanest window should win where it has good data (6th power with the
# window prior); his black tee / jeans / skin are smooth, a sharp winner makes patches of different exposure there (2nd power: a confidence-weighted mean)
POWERS = {"follow": {"low": 6.0, "high": 6.0}, "lead": {"low": 2.0, "high": 3.0}}


# ------------------------------------------------------------------------------------------------ colour helpers

def srgb_to_lin(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, np.float32), 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def lin_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, np.float32), 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.maximum(x, 1e-8) ** (1 / 2.4) - 0.055).astype(np.float32)


def luma(lin: np.ndarray) -> np.ndarray:
    return lin @ LUMA


# ------------------------------------------------------------------------------------------------ pure parts of the fusion

def exposure_gains(lin: list[np.ndarray], measured: list[np.ndarray], lo: float = 0.8, hi: float = 1.25, min_luma: float = 0.03,
                   min_texels: int = 2000) -> np.ndarray:
    """(M, 3) per-channel gains that bring every source to the common exposure. lin[i] (N, 3) linear RGB, measured[i] (N,) bool.
    Source i is compared with the CONSENSUS of the texels it shares with at least one other source (the per-texel geometric mean of every source
    that measured it, itself included), mid-tones only: the median log ratio per channel is its gain. Gains are clamped and normalised to a
    geometric mean of 1 (the overall exposure stays that of the bakes)."""
    M = len(lin)
    g = np.ones((M, 3), np.float32)
    if M < 2:
        return g
    stack = np.stack([np.log(np.maximum(x, 1e-4)) for x in lin])           # (M, N, 3)
    meas = np.stack(measured)                                              # (M, N)
    cnt = meas.sum(0)
    cons = (stack * meas[..., None]).sum(0) / np.maximum(cnt, 1)[:, None]   # (N, 3) log of the geometric mean over the measuring sources
    for i in range(M):
        use = meas[i] & (cnt >= 2)
        use &= (luma(lin[i]) > min_luma) & (lin[i].max(1) < 0.95)
        if int(use.sum()) >= min_texels:
            g[i] = np.clip(np.exp(np.median(cons[use] - stack[i][use], axis=0)), lo, hi)
    g /= np.exp(np.log(g).mean(0))[None]
    return g.astype(np.float32)


def source_quality(alpha: np.ndarray, conf: np.ndarray, ncam: np.ndarray, conf_ref: float, cover: np.ndarray | None = None,
                   cover_weight: float = 0.7) -> np.ndarray:
    """(N,) 0..~2 quality of one window's texel."""
    c = np.sqrt(np.clip(conf / max(conf_ref, 1e-12), 0.0, 4.0))
    q = np.clip(alpha, 0.0, 1.0) * c * (np.minimum(ncam, 3) / 3.0)
    if cover is not None:
        q = q * (1.0 - cover_weight * np.clip(cover, 0.0, 1.0))
    return q.astype(np.float32)


def blur_norm(img: np.ndarray, weight: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Normalised Gaussian smoothing of an image (H, W[, C]) by a weight image (H, W): (smoothed, smoothed weight). sigma <= 0: unchanged."""
    if sigma <= 0:
        return img, weight
    w = weight.astype(np.float32)
    ww = cv2.GaussianBlur(w, (0, 0), sigma)
    num = cv2.GaussianBlur(img * (w[..., None] if img.ndim == 3 else w), (0, 0), sigma)
    den = np.maximum(ww, 1e-6)
    return (num / (den[..., None] if img.ndim == 3 else den)).astype(np.float32), ww


def fuse_images(lin: list[np.ndarray], q: list[np.ndarray], covered: np.ndarray, low_sigma: float = 10.0, wsigma: float = 3.0, tau: float = 0.10,
                power_high: float = 6.0, light_texels: np.ndarray | None = None, dark_tau: float = 0.06, power_low: float = 6.0) -> tuple[np.ndarray, dict]:
    """Two-band, outlier-aware fusion of M linear-RGB images (H, W, 3) with quality images q (H, W) on the covered-texel mask (H, W).
    tau: the soft inlier distance in sRGB (a window whose low band lies this far from the estimate counts exp(-(d/tau)^2)).
    light_texels (H, W) bool: a window darker than the estimate by more than dark_tau (sRGB luma) is further down-weighted there.
    -> (fused linear RGB (H, W, 3), stats)."""
    M = len(lin)
    cov = covered.astype(np.float32)
    qs = [blur_norm(np.zeros_like(lin[0][..., 0]), q[i] * cov, wsigma)[1] for i in range(M)]       # smoothed quality
    low = [blur_norm(lin[i], np.maximum(q[i], 1e-4) * cov, low_sigma)[0] for i in range(M)]
    # weights relative to the best window of each texel (best = 1, so a high power never underflows); a floor keeps the filled colour of the windows
    # on texels nobody measured (never black)
    qf = [x + Q_FLOOR for x in qs]
    qmax = np.maximum.reduce(qf)
    w0 = [(x / qmax) ** power_low for x in qf]
    den = np.maximum(sum(w0), 1e-9)
    est = sum(w0[i][..., None] * low[i] for i in range(M)) / den[..., None]
    # soft outlier factor from the distance in sRGB (perceptual-ish), plus a darkness penalty on the texels that only get darker by contamination
    est_s = lin_to_srgb(est)
    fac = []
    for i in range(M):
        d = np.linalg.norm(lin_to_srgb(low[i]) - est_s, axis=-1)
        f = np.exp(-(d / tau) ** 2)
        if light_texels is not None:
            dl = luma(est_s) - luma(lin_to_srgb(low[i]))
            f = np.where(light_texels & (dl > dark_tau), f * np.exp(-((dl - dark_tau) / dark_tau) ** 2), f)
        fac.append(f.astype(np.float32))
    w1 = [w0[i] * fac[i] for i in range(M)]
    den1 = np.maximum(sum(w1), 1e-9)
    low_f = sum(w1[i][..., None] * low[i] for i in range(M)) / den1[..., None]
    # detail from the window that is best here (soft argmax)
    wq = [qf[i] / qmax * fac[i] for i in range(M)]
    wh = [x ** power_high + 1e-12 for x in wq]
    denh = sum(wh)
    high_f = sum(wh[i][..., None] * (lin[i] - low[i]) for i in range(M)) / denh[..., None]
    out = np.clip(low_f + high_f, 0.0, 1.0)
    wins = np.argmax(np.stack(wh), 0)
    stats = {"winner_share": [round(float(((wins == i) & covered).sum() / max(covered.sum(), 1)), 3) for i in range(M)],
             "outlier_mean_factor": [round(float(fac[i][covered].mean()), 3) for i in range(M)]}
    return out.astype(np.float32), stats


def agree_fraction(lin: list[np.ndarray], measured: list[np.ndarray], tol: float = 0.08) -> float:
    """Share of the texels measured by >= 2 windows whose sRGB colours agree within tol (a quick consistency number)."""
    idx = np.flatnonzero(np.sum(measured, axis=0) >= 2)
    if not len(idx) or len(lin) < 2:
        return float("nan")
    s = np.stack([lin_to_srgb(x[idx]) for x in lin])                 # (M, n, 3)
    m = np.stack([x[idx] for x in measured])                          # (M, n)
    med = np.median(np.where(m[..., None], s, np.nan), axis=0)
    d = np.nanmax(np.where(m, np.linalg.norm(s - med[None], axis=-1), 0.0), axis=0)
    return float((d < tol).mean())


# ------------------------------------------------------------------------------------------------ paint-out

def region_mask(p_rest: np.ndarray, centre, radius_m: float) -> np.ndarray:
    """(N,) bool: texels whose REST-body point lies within radius_m of `centre` (metres, the model's rest frame)."""
    return np.linalg.norm(np.asarray(p_rest, float) - np.asarray(centre, float)[None], axis=1) <= float(radius_m)


def grow_texels(mask_img: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask_img.astype(bool)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(mask_img.astype(np.uint8), k) > 0


def color_match_gain(target_lin: np.ndarray, donor_lin: np.ndarray, ring: np.ndarray, lo: float = 0.5, hi: float = 2.0) -> np.ndarray:
    """(3,) per-channel linear gain that makes the donor's mean on the ring (texels around the region, valid in both) equal the target's."""
    if int(ring.sum()) < 50:
        return np.ones(3, np.float32)
    t = target_lin[ring].mean(0)
    d = donor_lin[ring].mean(0)
    return np.clip(t / np.maximum(d, 1e-4), lo, hi).astype(np.float32)


def feather(mask: np.ndarray, px: float) -> np.ndarray:
    """(H, W) float 0..1: 1 inside the mask, a smooth ramp over ~px texels outside."""
    if px <= 0:
        return mask.astype(np.float32)
    d = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3)
    return np.clip(1.0 - d / float(px), 0.0, 1.0).astype(np.float32) ** 2 * (3 - 2 * np.clip(1.0 - d / float(px), 0.0, 1.0))


def window_prior(residue_pct: float, scale: float = 8.0, power: float = 2.0) -> float:
    """Trust of a window's bake on her white top / neck from its measured hair residue (avatar_hair.texture_metrics: the share of texels that are
    hair-coloured or hair-darkened): (1 / (1 + residue / scale)) ** power. With the sixth-power window selection of fuse_images a window with 6 %
    residue takes the texels from one with 18 % (the print on her top stays as sharp as the best window's) unless it has no good data there."""
    return float((1.0 / (1.0 + max(residue_pct, 0.0) / scale)) ** power)


def paint_one(img_srgb: np.ndarray, spec: dict, texel_mask: np.ndarray, atlas_index: np.ndarray,
              donor_srgb: np.ndarray | None) -> dict:
    """Replace the texels `texel_mask` (N,) of the albedo image (H, W, 3 sRGB, modified in place) according to `spec` (kind donor | inpaint).
    atlas_index (H, W) int: the texel index of each pixel (-1 uncovered). -> a report."""
    H, W = img_srgb.shape[:2]
    mimg = np.zeros((H, W), bool)
    # texel -> image pixel through atlas_index
    ys, xs = np.nonzero(atlas_index >= 0)
    sel = texel_mask[atlas_index[ys, xs]]
    mimg[ys[sel], xs[sel]] = True
    dark = spec.get("luma_max")
    if dark is not None:
        mimg &= (img_srgb @ LUMA) <= float(dark)       # only the dark texels of the region (a black pack on light jeans)
    lmin = spec.get("luma_min")
    if lmin is not None:                               # ... or only the LIGHT texels (pale patches on her dark red scalp)
        mimg &= (img_srgb @ LUMA) >= float(lmin)
    sat_max = spec.get("sat_max")
    if sat_max is not None:                            # ... and only the grey ones (a headset's black / grey plastic; hair and skin are chromatic)
        mx, mn = img_srgb.max(-1), img_srgb.min(-1)
        mimg &= (mx - mn) <= float(sat_max) * np.maximum(mx, 1e-3)
    sat_min = spec.get("sat_min")
    if sat_min is not None:                            # ... or only the CHROMATIC ones (a skin-coloured patch on grey cloth)
        mx, mn = img_srgb.max(-1), img_srgb.min(-1)
        mimg &= (mx - mn) >= float(sat_min) * np.maximum(mx, 1e-3)
    if int(mimg.sum()) == 0:
        return {"name": spec.get("name"), "texels": 0, "note": "nothing to paint"}
    grown = grow_texels(mimg, int(spec.get("grow_texels", 8))) & (atlas_index >= 0)
    ring = grow_texels(grown, int(spec.get("ring_texels", 14))) & ~grown & (atlas_index >= 0)
    lin = srgb_to_lin(img_srgb)
    rep = {"name": spec.get("name"), "kind": spec.get("kind", "donor"), "texels": int(grown.sum()), "ring": int(ring.sum())}
    if spec.get("kind") == "flat_median":              # the region becomes the median colour of its ring's non-chromatic texels (grey cloth: the bib's white paper -> the trousers' grey)
        mx, mn = img_srgb.max(-1), img_srgb.min(-1)
        calm = ring & ((mx - mn) <= float(spec.get("ring_sat_max", 0.14)) * np.maximum(mx, 1e-3))
        if spec.get("ring_luma_max") is not None:        # e.g. a black shirt: the ring's DARK texels only (the skin of an arm next to it is not the cloth)
            calm &= (img_srgb @ LUMA) <= float(spec["ring_luma_max"])
        if int(calm.sum()) < 50:
            return {"name": spec.get("name"), "texels": int(grown.sum()), "note": "no calm ring"}
        med = np.median(lin[calm], axis=0)
        fill = np.broadcast_to(med[None, None], lin.shape)
        rep["gain"] = None
        rep["median_srgb"] = [round(float(x), 3) for x in lin_to_srgb(med)]
        a = feather(grown, float(spec.get("feather_texels", 6)))
        a = np.where(atlas_index >= 0, a, 0.0)
        img_srgb[...] = lin_to_srgb(lin * (1 - a[..., None]) + fill * a[..., None])
        return rep
    if spec.get("kind", "donor") == "donor" and donor_srgb is not None:
        dl = srgb_to_lin(donor_srgb)
        g = color_match_gain(lin, dl, ring)
        rep["gain"] = [round(float(x), 3) for x in g]
        fill = dl * g[None, None]
    else:
        u8 = np.clip(img_srgb * 255 + 0.5, 0, 255).astype(np.uint8)
        hole = grown | (atlas_index < 0) if spec.get("unknown_outside") else grown      # unknown_outside: the empty atlas (black) next to an island is NOT a source colour (john_lula)
        if spec.get("ignore_sat_min") is not None:       # chromatic texels (skin, coral) are not a source colour either: grey cloth must not be filled from the skin next to it
            mx, mn = img_srgb.max(-1), img_srgb.min(-1)
            hole = hole | (((mx - mn) >= float(spec["ignore_sat_min"]) * np.maximum(mx, 1e-3)) & (atlas_index >= 0))
        fill = srgb_to_lin(cv2.inpaint(u8, hole.astype(np.uint8) * 255, 7, cv2.INPAINT_TELEA).astype(np.float32) / 255.0)
        rep["gain"] = None
    a = feather(grown, float(spec.get("feather_texels", 6)))
    a = np.where(atlas_index >= 0, a, 0.0)
    out = lin * (1 - a[..., None]) + fill * a[..., None]
    img_srgb[...] = lin_to_srgb(out)
    return rep


def scalp_weight(atlas, regions: dict, feather_m: float = 0.03) -> np.ndarray:
    """(N,) 0..1 per texel: the surface distance INSIDE the scalp boundary (the vertex field avatar_hair.texel_regions['scalp_dist_v'], interpolated over the
    triangle) over feather_m, so a correction fades out along the MESH (a UV-space feather breaks at the island seams) and is full in the middle of the scalp."""
    d = np.clip(np.asarray(regions["scalp_dist_v"], np.float32), -1.0, 1.0)[atlas.faces[atlas.face]]      # unreachable vertices are -inf: 0 x -inf would be NaN
    return np.clip((d * atlas.bary).sum(1) / float(feather_m), 0.0, 1.0).astype(np.float32)


def smooth_scalp(img_srgb: np.ndarray, atlas, regions: dict, strength: float = 0.7, sigma: float = 40.0, detail: float = 0.5, feather_m: float = 0.03,
                 warm: float = 0.0) -> dict:
    """A shaved scalp that the bake rendered in blotches (a fill in one place, a window's different exposure in another) -> its robust median colour with
    only (1 - strength) of the low-frequency variation left (sigma texels) and `detail` of the fine detail (stubble), faded along the mesh at the scalp's
    boundary (scalp_weight). In place on an albedo image (H, W, 3 sRGB). -> report."""
    w_t = scalp_weight(atlas, regions, feather_m)
    w_t = np.where(np.asarray(regions["prior"], bool), w_t, 0.0)
    w = atlas.to_image(w_t, 0.0).astype(np.float32)
    m = w > 0.5
    if int(m.sum()) < 500:
        return {"texels": int(m.sum()), "note": "scalp too small"}
    lin = srgb_to_lin(img_srgb)
    med = np.median(lin[m], axis=0)
    if warm > 0:                    # a shaved scalp reads warmer than the grey the bake leaves: `warm` of the median colour of his face skin (face_core)
        fc = atlas.to_image(np.asarray(regions["face_core"], bool), False)
        med = (1.0 - warm) * med + warm * np.median(lin[fc], axis=0)
    low, _ = blur_norm(lin, m.astype(np.float32), sigma)
    fine, _ = blur_norm(lin, m.astype(np.float32), 2.0)
    target = med[None, None] + (1.0 - strength) * (low - med[None, None]) + detail * (lin - fine)
    out = lin * (1.0 - w[..., None]) + np.clip(target, 0.0, 1.0) * w[..., None]
    img_srgb[...] = lin_to_srgb(out)
    return {"texels": int(m.sum()), "median_srgb": [round(float(x), 3) for x in lin_to_srgb(med)], "strength": strength, "sigma": sigma, "detail": detail, "warm": warm}


def smooth_region(img_srgb: np.ndarray, mask_img: np.ndarray, sigma: float = 5.0, strength: float = 0.6, feather: float = 6.0) -> dict:
    """The texels `mask_img` (H, W bool) replaced by (1 - strength) x themselves + strength x their normalised Gaussian blur (sigma texels, only the region's own texels
    contribute), feathered at the region's edge. Linear RGB, in place."""
    if int(mask_img.sum()) < 500:
        return {"texels": int(mask_img.sum()), "note": "region too small"}
    lin = srgb_to_lin(img_srgb)
    low, _ = blur_norm(lin, mask_img.astype(np.float32), sigma)
    a = (feather_in(mask_img, feather) * strength)[..., None]
    img_srgb[...] = lin_to_srgb(lin * (1 - a) + low * a)
    return {"texels": int(mask_img.sum()), "sigma": sigma, "strength": strength}


def feather_in(mask: np.ndarray, px: float) -> np.ndarray:
    """(H, W) float 0..1: 0 outside the mask, a ramp up over ~px texels INSIDE it."""
    if px <= 0:
        return mask.astype(np.float32)
    d = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    t = np.clip(d / float(px), 0.0, 1.0)
    return (t * t * (3 - 2 * t)).astype(np.float32)


def lift_dark(img_srgb: np.ndarray, mask_img: np.ndarray, strength: float = 0.5, sigma: float = 50.0, deadband: float = 0.03) -> dict:
    """Light cloth only gets darker by what does not belong to it: shading of the baked-in lighting, a partner's dark sleeve, her hair, a dirty fold. The
    luminance of the texels `mask_img` that lies below the local mean (a Gaussian of sigma texels over the region) by more than `deadband` is lifted by
    `strength` of the excess (colour kept: every channel scales with the luminance). Detail brighter than its surroundings is untouched. In place. -> report."""
    if int(mask_img.sum()) < 500:
        return {"texels": int(mask_img.sum()), "note": "region too small"}
    lin = srgb_to_lin(img_srgb)
    Y = luma(lin)
    w = mask_img.astype(np.float32)
    ref, _ = blur_norm(Y, w, sigma)
    excess = np.maximum(ref - Y - deadband, 0.0)
    gain = np.where(mask_img, (Y + strength * excess) / np.maximum(Y, 1e-4), 1.0)
    img_srgb[...] = lin_to_srgb(np.clip(lin * np.clip(gain, 1.0, 4.0)[..., None], 0.0, 1.0))
    return {"texels": int(mask_img.sum()), "strength": strength, "sigma": sigma, "deadband": deadband, "mean_lift_pct": round(float(100 * (gain[mask_img] - 1).mean()), 2)}


def region_texels(regions: dict, spec: dict, atlas=None, avatar_dir: Path | None = None) -> np.ndarray:
    """(N,) bool texels of a paint-out region: `centre` + `radius_m` on the rest body, or `region` = a named texel set of avatar_hair.texel_regions
    (scalp = the anatomical scalp minus the face core, ear, neck, upper, ...); `joints` restricts to the texels whose dominant joint is in the list."""
    if spec.get("region") == "joints":              # only `joints`: the texels whose dominant joint is in the list (cloth of the torso / legs)
        tm = np.isin(regions["dom"], spec["joints"]) & ~np.asarray(regions["head"], bool)
    elif spec.get("region"):
        tm = np.asarray(regions[spec["region"]], bool).copy()
        if spec["region"] == "prior":
            tm &= ~np.asarray(regions["face_core"], bool)
    else:
        tm = region_mask(regions["P0"], spec["centre"], float(spec["radius_m"]))
    if spec.get("joints") and spec.get("region") != "joints":
        tm &= np.isin(regions["dom"], spec["joints"])
    if spec.get("y_max") is not None:               # only below this REST-body height (m): the seat of the trousers, not the waist skin above it
        tm &= np.asarray(regions["P0"], float)[:, 1] <= float(spec["y_max"])
    return tm


def apply_paint(img: np.ndarray, role: str, paint_spec: dict, regions: dict, index: np.ndarray, donor: np.ndarray | None, stage: str | None = None,
                atlas=None, avatar_dir: Path | None = None) -> list[dict]:
    """Run the paint-out regions of `role` (and of `stage`: None = every region without a `stage`, "final" = the regions marked final) on img in place."""
    out = []
    for spec in paint_spec.get("regions", []):
        if spec.get("role") != role or spec.get("stage") != stage:
            continue
        if spec.get("kind") == "smooth_scalp":
            out.append({"name": spec.get("name"), "kind": "smooth_scalp", **smooth_scalp(img, atlas, regions, float(spec.get("strength", 0.7)), float(spec.get("sigma", 40.0)),
                                                                                    float(spec.get("detail", 0.5)), float(spec.get("feather_m", 0.03)), float(spec.get("warm", 0.0)))})
            continue
        tm = region_texels(regions, spec, atlas, avatar_dir)
        if spec.get("kind") == "smooth_region":     # a low-pass over the region (one low-resolution camera: the detail of a print / skin is registration noise, kept at 1 - strength)
            ys, xs = np.nonzero(index >= 0)
            m = np.zeros(index.shape, bool)
            sel = tm[index[ys, xs]]
            m[ys[sel], xs[sel]] = True
            out.append({"name": spec.get("name"), "kind": "smooth_region", **smooth_region(img, m, float(spec.get("sigma", 5.0)), float(spec.get("strength", 0.6)),
                                                                                         float(spec.get("feather_texels", 6)))})
            continue
        if spec.get("kind") == "lift_dark":
            ys, xs = np.nonzero(index >= 0)
            m = np.zeros(index.shape, bool)
            sel = tm[index[ys, xs]]
            m[ys[sel], xs[sel]] = True
            out.append({"name": spec.get("name"), "kind": "lift_dark", **lift_dark(img, m, float(spec.get("strength", 0.5)), float(spec.get("sigma", 50.0)),
                                                                                float(spec.get("deadband", 0.03)))})
            continue
        out.append(paint_one(img, spec, tm, index, donor))
    return out


def load_donor(work: Path, paint_spec: dict, role: str, donor_take: str | None = None) -> np.ndarray | None:
    name = donor_take or paint_spec.get("donor_take")
    if not name:
        return None
    dp = work.parent / name / "avatar" / f"{role}_albedo_face.png"
    return cv2.imread(str(dp))[:, :, ::-1].astype(np.float32) / 255.0 if dp.exists() else None


def paint_final(work: Path, paint: Path | None = None, roles=("follow",), target: str = "{role}_albedo_face.png", dry_run: bool = False) -> dict:
    """The regions marked "stage": "final" in paint_out.json, applied to the face-composited albedos (avatar/<role>_albedo_face.png, in place, the
    first state kept as <name>.prepaint.png): what the portrait composite leaves outside its mask (pale patches in her scalp, junk below the ears)."""
    import shutil

    from . import avatar_hair as ahair
    from .avatar_uv import build_atlas
    from .mamma_dyn import SMPLX_MODELS
    paint = paint or (work / "avatar" / "paint_out.json")
    spec = json.loads(paint.read_text(encoding="utf-8"))
    atlas = build_atlas(SMPLX_MODELS, 2048)
    regions = ahair.texel_regions(atlas, SMPLX_MODELS)
    index = atlas.index_image()
    rep: dict = {}
    for role in roles:
        f = work / "avatar" / target.format(role=role)
        img = cv2.imread(str(f))[:, :, ::-1].astype(np.float32) / 255.0
        donor = load_donor(work, spec, role)
        rep[role] = apply_paint(img, role, spec, regions, index, donor, stage="final", atlas=atlas, avatar_dir=work / "avatar")
        if rep[role] and not dry_run:
            bak = f.with_name(f.stem + ".prepaint.png")
            if not bak.exists():
                shutil.copy2(f, bak)
            cv2.imwrite(str(f), (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)[:, :, ::-1])
    return rep


# ------------------------------------------------------------------------------------------------ the stage

def load_bake(work_take: Path, role: str) -> dict:
    z = np.load(work_take / "avatar" / f"{role}_texels.npz", allow_pickle=True)
    d = {"rgb": z["rgb_srgb"].astype(np.float32), "alpha": z["alpha"].astype(np.float32), "conf": z["conf"].astype(np.float32),
         "ncam": z["ncam"].astype(np.int32), "source": z["source"]}
    d["cover"] = z["hair_cover"].astype(np.float32) / 255.0 if "hair_cover" in z.files else None
    d["hair_region"] = z["hair_region"] if "hair_region" in z.files else None
    return d


def run(work: Path, window_ids: list[str], paint: Path | None = None, out: Path | None = None, roles=("lead", "follow"), donor_take: str | None = None,
        dry_run: bool = False, low_sigma: float = 10.0, tau: float = 0.10, prefer_light: bool = True, use_prior: bool = True,
        light_roles: tuple[str, ...] = ("follow",), powers: dict | None = None, window_weights: dict | None = None,
        min_ncam: int = 2) -> dict:
    """work = work/<take>; the window bakes live in work/../<take>_<id>/avatar/. Writes <out or work/avatar>/<role>_albedo.png, _coverage.png,
    _texels.npz and fuse_report.json (the face composite and the groom read these)."""
    from . import avatar_hair as ahair
    from .avatar_texture import gutter
    from .avatar_uv import build_atlas
    from .mamma_dyn import SMPLX_MODELS
    out = out or (work / "avatar")
    out.mkdir(parents=True, exist_ok=True)
    atlas = build_atlas(SMPLX_MODELS, 2048)
    regions = ahair.texel_regions(atlas, SMPLX_MODELS)
    index = atlas.index_image()
    covered_img = index >= 0
    takes = [work.parent / f"{work.name}_{w}" for w in window_ids]
    rep: dict = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "windows": window_ids, "roles": {}}
    paint_spec = json.loads(paint.read_text(encoding="utf-8")) if paint and paint.exists() else {"regions": []}
    for role in roles:
        bakes = [load_bake(t, role) for t in takes]
        lin = [srgb_to_lin(b["rgb"]) for b in bakes]
        measured = [(b["alpha"] > 0.6) & (b["ncam"] >= min_ncam) for b in bakes]      # min_ncam 1: ONE camera (john_lula), no texel has two
        gains = exposure_gains(lin, measured)
        lin = [x * g[None] for x, g in zip(lin, gains)]
        conf_ref = float(np.median(np.concatenate([b["conf"][b["conf"] > 0] for b in bakes])))
        q = [source_quality(b["alpha"], b["conf"], b["ncam"], conf_ref, b["cover"] if role == "follow" else None) for b in bakes]
        priors = [1.0] * len(bakes)
        for i, wid in enumerate(window_ids):      # window_weights {window id: factor}: a window the owner / reviewer found dirtier counts less on every texel
            f = float((window_weights or {}).get(wid, 1.0))
            if f != 1.0:
                q[i] = q[i] * f
        if use_prior and role == "follow":      # her top / neck: the window whose bake shows the least hair residue counts more (on those texels only)
            upper = regions["upper"] | regions["neck"] | regions["shoulders"]
            priors = []
            for t, qi in zip(takes, q):
                tm = (json.loads((t / "avatar" / "report.json").read_text(encoding="utf-8")).get("roles", {}).get(role, {}).get("texture_metrics") or {})
                res = float(np.mean([tm[k]["hair_residue_pct"] for k in ("chest", "back", "shoulders") if k in tm] or [0.0]))
                pr = window_prior(res)
                qi[upper] *= pr
                priors.append(round(pr, 3))
        timg = [atlas.to_image(x, 0.0) for x in lin]
        qimg = [atlas.to_image(x, 0.0) for x in q]
        light = None
        if prefer_light and role in light_roles:      # light_roles: whose top is white (the recap: follow; carlos_aline: lead)
            light = atlas.to_image(regions["upper"] | regions["neck"] | regions["shoulders"], False) & ~atlas.to_image(regions["head"], False)
        pw = (powers or {}).get(role) or POWERS.get(role, {"low": 3.0, "high": 4.0})
        fused_img, st = fuse_images(timg, qimg, covered_img, low_sigma=low_sigma, tau=tau, light_texels=light, power_low=pw["low"], power_high=pw["high"])
        fused_lin = fused_img[atlas.texel_yx[:, 0], atlas.texel_yx[:, 1]]
        srgb = lin_to_srgb(fused_lin)
        rrep = {"exposure_gain": gains.round(3).tolist(), "window_prior": priors, "agree_fraction": round(agree_fraction(lin, measured), 3), **st,
                "measured_fraction": [round(float(m.mean()), 3) for m in measured], "conf_ref": round(conf_ref, 6)}
        img = atlas.to_image(srgb, 0.0)
        if paint_spec.get("regions"):
            rrep["paint"] = apply_paint(img, role, paint_spec, regions, index, load_donor(work, paint_spec, role, donor_take), stage=None,
                                        atlas=atlas, avatar_dir=out)
        # alpha / coverage of the fusion, then the gutter
        alpha = 1.0 - np.prod([1.0 - np.clip(b["alpha"], 0, 1) for b in bakes], axis=0)
        conf = np.sum([b["conf"] for b in bakes], axis=0)
        ncam = np.max([b["ncam"] for b in bakes], axis=0)
        if not dry_run:
            g = gutter(img, covered_img, 16)
            cv2.imwrite(str(out / f"{role}_albedo.png"), (np.clip(g, 0, 1) * 255 + 0.5).astype(np.uint8)[:, :, ::-1])
            cov = np.zeros((atlas.n, 3), np.uint8)
            cov[:, 0] = np.clip(255 * alpha, 0, 255).astype(np.uint8)
            cov[:, 1] = np.clip(ncam * 32, 0, 255).astype(np.uint8)
            cov[:, 2] = 0
            cv2.imwrite(str(out / f"{role}_coverage.png"), atlas.to_image(cov)[:, :, ::-1])
            extra = {}
            if bakes[0]["hair_region"] is not None:
                extra["hair_region"] = np.max([b["hair_region"] for b in bakes if b["hair_region"] is not None], axis=0)
            if bakes[0]["cover"] is not None:
                extra["hair_cover"] = (np.mean([b["cover"] for b in bakes], axis=0) * 255 + 0.5).astype(np.uint8)
            np.savez_compressed(out / f"{role}_texels.npz", rgb_srgb=lin_to_srgb(fused_lin).astype(np.float16), alpha=alpha.astype(np.float16),
                                conf=conf.astype(np.float32), ncam=ncam.astype(np.uint8), source=np.zeros(atlas.n, np.uint8), **extra)
        rep["roles"][role] = rrep
    if not dry_run:
        (out / "fuse_report.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
        rep["aux"] = install_aux(takes[0] / "avatar", out)
        (out / "TEXTURE_FROM.json").write_text(json.dumps(provenance(work, window_ids, rep, paint_spec), indent=1, default=str), encoding="utf-8")
    return rep


AUX_FILES = ("uv_mesh.npz", "head_fix.npz", "report.json", "follow_hair_region.png", "follow_hair.json", "follow_hair_strip.png")


def install_aux(src: Path, out: Path) -> dict:
    """The files of ONE window's bake that are not per-texel colour (the UV mesh, the head fix, the bake report, her hair region / colour sample) come from the
    first window of the list: the fused albedo needs none of them, the export and the groom read some. -> {file: installed | missing}."""
    import shutil
    res = {}
    for n in AUX_FILES:
        f = src / n
        if f.exists():
            shutil.copy2(f, out / n)
            res[n] = f"installed from {src.parent.name}"
        else:
            res[n] = "missing"
    return res


def provenance(work: Path, window_ids: list[str], rep: dict, paint_spec: dict) -> dict:
    """avatar/TEXTURE_FROM.json: which windows (their poses, cameras, masked camera stretches, view counts), how they were fused, what was painted out."""
    wins = []
    chosen = {}
    wf = work / "avatar" / "texture" / "windows_final.json"
    if wf.exists():
        chosen = {w["id"]: w for w in json.loads(wf.read_text(encoding="utf-8")).get("windows", [])}
    for w in window_ids:
        t = work.parent / f"{work.name}_{w}" / "avatar"
        r = json.loads((t / "report.json").read_text(encoding="utf-8")) if (t / "report.json").exists() else {}
        wins.append({"id": w, "t_window_s": r.get("t_window"), "cams": r.get("cams"), "n_views": r.get("n_views"),
                     "masked": (chosen.get(w) or {}).get("mask"), "chunk": (chosen.get(w) or {}).get("chunk"),
                     "settings": {k: (r.get("config") or {}).get(k) for k in ("preset", "step_s", "top_k", "body_ref_q", "align_passes", "hair_mode", "blend")},
                     "measured_fraction": {ro: (r.get("roles") or {}).get(ro, {}).get("measured_fraction") for ro in ("lead", "follow")},
                     "median_view_iou": {ro: (r.get("roles") or {}).get(ro, {}).get("median_view_iou") for ro in ("lead", "follow")}})
    return {"kind": "fusion of window bakes (dancecap.recap_avatars_fuse)", "created": rep.get("created"), "windows": wins,
            "poses": "the stitched smplx/ arrays at the time (snapshot avatar/texture/smplx_snapshot/); a window inside a chunk interior is the chunk fit bit for bit",
            "fusion": {k: v for k, v in rep["roles"].items()}, "paint_out": [r for r in paint_spec.get("regions", [])],
            "donor_take": paint_spec.get("donor_take")}
