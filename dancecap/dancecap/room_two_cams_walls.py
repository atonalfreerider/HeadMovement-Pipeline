"""Back wall, carpet edge and cornice of the new room from the two calibrated cameras (no room scan exists).

  wall_sweep()   plane sweep: for a hypothetical vertical wall plane (offset z_w, yaw phi: the wall's direction is (cos phi, 0, sin phi)) every pixel of camera 02 is lifted to the
                 plane and re-projected into camera 01; the zero-mean normalised correlation of the two edge-strength images
                 (people masked, the watermark corner of 01 masked) peaks at the plane both cameras see as the same surface.
  edge_scan()    a straight floor line (the carpet / cream-floor border, the wall base) is a polyline in the world: for each
                 hypothesis (z, yaw) the mean "light below - dark above" contrast along it, per camera, people masked. Two
                 cameras that find the same line at the same place agree on their geometry without sharing any pixel.
  cornice_scan() the wall's top edge (dark wood below, bright cove above) on the wall plane at height h.

All functions take per-frame world cameras (K, R = cam_from_world, tvec, k1) like cameras/<cam>.npz.
"""
from __future__ import annotations

import cv2
import numpy as np
import scipy.ndimage as ndi

from .cameras import room_draw as D
from .cameras import twocam_world as TW


def edge_map(gray: np.ndarray, sigma: float) -> np.ndarray:
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigma)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def zncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def cam_dict(arr: dict, i: int) -> dict:
    return dict(R=arr["R"][i], tvec=arr["tvec"][i], f=float(arr["K"][i, 0, 0]), c=(float(arr["K"][i, 0, 2]), float(arr["K"][i, 1, 2])),
                k1=float(arr["dist"][i]))


def plane_hits(arr: dict, i: int, size: tuple[int, int], zw: float, phi_deg: float, step: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """World points where the pixels of camera frame i (every `step`-th) meet the vertical plane through (0, y, zw) with normal
    (-sin phi, 0, cos phi), i.e. the line z = zw + x tan(phi). Returns (pixels (M,2), points (M,3), valid)."""
    w, h = size
    gx, gy = np.meshgrid(np.arange(0, w, step), np.arange(0, h, step))
    px = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float64)
    rays = TW.pixel_rays_world(px, arr["K"][i], arr["R"][i], float(arr["dist"][i]))
    C = arr["C"][i]
    n = np.array([-np.sin(np.radians(phi_deg)), 0.0, np.cos(np.radians(phi_deg))])
    den = rays @ n
    t = (zw * n[2] - C @ n) / np.where(np.abs(den) < 1e-9, 1e-9, den)
    X = C[None] + t[:, None] * rays
    return px, X, t > 0


def wall_sweep(arr_a: dict, arr_b: dict, frames: list[dict], zs: np.ndarray, phis: np.ndarray, y_range=(0.4, 3.0),
               stride: int = 3, pix_step: int = 3) -> dict:
    """frames: [{'ia': row in arr_a, 'ib': row in arr_b, 'eA': edge map of 01, 'eB': edge map of 02, 'mA': mask 01, 'mB': mask 02}].
    Returns {'zs', 'phis', 'score' (nz, nphi, nframes)}."""
    size_b = (frames[0]["eB"].shape[1], frames[0]["eB"].shape[0])
    score = np.full((len(zs), len(phis), len(frames)), np.nan)
    for k, fr in enumerate(frames):
        for b, phi in enumerate(phis):
            for a, zw in enumerate(zs):
                px, X, valid = plane_hits(arr_b, fr["ib"], size_b, zw, phi, step=pix_step)
                ok = valid & (X[:, 1] > y_range[0]) & (X[:, 1] < y_range[1]) & (np.abs(X[:, 0]) < 40)
                pa, za = D.project(X, **cam_dict(arr_a, fr["ia"]))
                eA = fr["eA"]
                ok &= (za > 1) & (pa[:, 0] > 5) & (pa[:, 0] < eA.shape[1] - 5) & (pa[:, 1] > 5) & (pa[:, 1] < eA.shape[0] - 5)
                idx = np.where(ok)[0][::stride]
                if len(idx) < 4000:
                    continue
                xa, ya = pa[idx, 0].astype(np.float32), pa[idx, 1].astype(np.float32)
                va = ndi.map_coordinates(eA, [ya, xa], order=1)
                ma = ndi.map_coordinates(fr["mA"], [ya, xa], order=0)
                pb = px[idx].astype(int)
                vb = fr["eB"][pb[:, 1], pb[:, 0]]
                mb = fr["mB"][pb[:, 1], pb[:, 0]]
                m = (ma == 0) & (mb == 0)
                if m.sum() >= 3000:
                    score[a, b, k] = zncc(va[m], vb[m])
    return {"zs": zs, "phis": phis, "score": score}


def peak(score: np.ndarray, zs: np.ndarray, phis: np.ndarray) -> dict:
    """Pooled maximum of the frame-mean score with a parabola in z (sub-grid) and the frame-to-frame spread of the per-frame optimum."""
    mean = np.nanmean(score, axis=2)
    a, b = np.unravel_index(np.nanargmax(mean), mean.shape)
    z_best = zs[a]
    if 0 < a < len(zs) - 1:
        y0, y1, y2 = mean[a - 1, b], mean[a, b], mean[a + 1, b]
        den = y0 - 2 * y1 + y2
        if den < 0:
            z_best = zs[a] + 0.5 * (y0 - y2) / den * (zs[1] - zs[0])
    per = []
    for k in range(score.shape[2]):
        s = score[:, :, k]
        if np.isfinite(s).any():
            i, j = np.unravel_index(np.nanargmax(s), s.shape)
            per.append((zs[i], phis[j], float(s[i, j])))
    per = np.array(per)
    good = per[:, 2] > 0.8 * np.median(per[:, 2]) if len(per) else np.zeros(0, bool)
    return {"z": float(z_best), "phi_deg": float(phis[b]), "score": float(mean[a, b]),
            "per_frame_z_median": float(np.median(per[good, 0])) if good.any() else None,
            "per_frame_z_mad": float(1.4826 * np.median(np.abs(per[good, 0] - np.median(per[good, 0])))) if good.any() else None,
            "per_frame_phi_median": float(np.median(per[good, 1])) if good.any() else None, "n_frames": int(len(per))}


def plateau_peak(score: np.ndarray, zs: np.ndarray, phis: np.ndarray, frac: float = 0.97) -> tuple[float, float]:
    """(z, phi) of the maximum of a contrast score whose top is a plateau (an edge seen through a +-band_px window): the centroid of
    the samples within `frac` of the maximum along the best-yaw column / the best-z row."""
    i, j = np.unravel_index(np.argmax(score), score.shape)
    col = score[:, j]
    m = col >= frac * col.max()
    row = score[i, :]
    mr = row >= frac * row.max()
    return float(np.mean(zs[m])), float(np.mean(phis[mr]))


def edge_scan(arr: dict, idx: list[int], images: list[np.ndarray], ignores: list[np.ndarray], zs: np.ndarray, phis: np.ndarray,
              x_half: float = 16.0, band_px: int = 7, rects=(), x_range: tuple[float, float] | None = None) -> dict:
    """Score of a floor line (y = 0, z = z0 + x tan(phi)) per hypothesis: mean contrast (below - above, clipped) over its visible
    unmasked samples, summed over the frames. Returns {'zs', 'phis', 'score' (nz, nphi), 'per_frame_best' (n,3)}."""
    xs = np.linspace(*(x_range or (-x_half, x_half)), 200)
    tot = np.zeros((len(zs), len(phis)))
    per = []
    for i, im, ign in zip(idx, images, ignores):
        g = cv2.GaussianBlur(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.5)
        h, w = g.shape
        cam = cam_dict(arr, i)
        S = np.zeros_like(tot)
        for a, z0 in enumerate(zs):
            for b, ph in enumerate(phis):
                X = np.stack([xs * np.cos(np.radians(ph)), np.zeros_like(xs), z0 + xs * np.sin(np.radians(ph))], 1)
                p, z = D.project(X, **cam)
                ok = (z > 1) & (p[:, 0] > 20) & (p[:, 0] < w - 20) & (p[:, 1] > 40) & (p[:, 1] < h - 40)
                xi, yi = p[ok, 0].astype(int), p[ok, 1].astype(int)
                for (x0, y0, x1, y1) in rects:
                    k2 = ~((xi > x0) & (yi > y0 - 40))
                    xi, yi = xi[k2], yi[k2]
                if len(xi) < 20:
                    continue
                free = ign[yi, xi] == 0
                if free.sum() < 20:
                    continue
                xi, yi = xi[free], yi[free]
                d = g[np.clip(yi + band_px, 0, h - 1), xi] - g[np.clip(yi - band_px, 0, h - 1), xi]
                S[a, b] = np.mean(np.clip(d, -30, 30))
        tot += S
        a, b = np.unravel_index(np.argmax(S), S.shape)
        per.append((zs[a], phis[b], S[a, b]))
    return {"zs": zs, "phis": phis, "score": tot / max(len(idx), 1), "per_frame_best": np.array(per)}


def cornice_scan(arr: dict, idx: list[int], images: list[np.ndarray], ignores: list[np.ndarray], zw: float, phi_deg: float,
                 heights: np.ndarray, x_half: float = 14.0, band_px: int = 8, rects=()) -> dict:
    """Score per height h of the horizontal line on the wall plane (z = zw + x tan(phi), y = h): mean (above - below) contrast, i.e.
    the bright cove above the dark wood. Returns {'heights', 'score' (nh,), 'per_frame_best'}."""
    xs = np.linspace(-x_half, x_half, 240)
    tot = np.zeros(len(heights))
    per = []
    for i, im, ign in zip(idx, images, ignores):
        g = cv2.GaussianBlur(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.5)
        h_, w_ = g.shape
        cam = cam_dict(arr, i)
        S = np.zeros(len(heights))
        for a, hh in enumerate(heights):
            X = np.stack([xs * np.cos(np.radians(phi_deg)), np.full_like(xs, hh), zw + xs * np.sin(np.radians(phi_deg))], 1)
            p, z = D.project(X, **cam)
            ok = (z > 1) & (p[:, 0] > 20) & (p[:, 0] < w_ - 20) & (p[:, 1] > 40) & (p[:, 1] < h_ - 40)
            xi, yi = p[ok, 0].astype(int), p[ok, 1].astype(int)
            for (x0, y0, x1, y1) in rects:
                k2 = ~((xi > x0) & (yi > y0 - 40))
                xi, yi = xi[k2], yi[k2]
            if len(xi) < 20:
                continue
            free = ign[yi, xi] == 0
            if free.sum() < 20:
                continue
            xi, yi = xi[free], yi[free]
            d = g[np.clip(yi - band_px, 0, h_ - 1), xi] - g[np.clip(yi + band_px, 0, h_ - 1), xi]
            S[a] = np.mean(np.clip(d, -40, 40))
        tot += S
        per.append((heights[int(np.argmax(S))], float(S.max())))
    return {"heights": heights, "score": tot / max(len(idx), 1), "per_frame_best": np.array(per)}


def ortho_texture(points: np.ndarray, shape: tuple[int, int], sources: list[dict], min_obs: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Median colour per texel of a planar patch seen by several frames of one camera.

    points: (H*W, 3) world position of every texel (row-major); sources: [{'arr': world camera arrays, 'i': row, 'img': bgr,
    'ignore': uint8 mask (255 = person / overlay)}]. Returns (image (H, W, 3) uint8 bgr, n_obs (H, W)); texels with fewer than
    `min_obs` observations are NaN-filled by the caller."""
    n = len(points)
    stack = np.full((len(sources), n, 3), np.nan, np.float32)
    for k, s in enumerate(sources):
        cam = cam_dict(s["arr"], s["i"])
        p, z = D.project(points, **cam)
        h, w = s["img"].shape[:2]
        ok = (z > 1) & (p[:, 0] > 2) & (p[:, 0] < w - 3) & (p[:, 1] > 2) & (p[:, 1] < h - 3)
        idx = np.where(ok)[0]
        xa, ya = p[idx, 0].astype(np.float32), p[idx, 1].astype(np.float32)
        m = ndi.map_coordinates(s["ignore"], [ya, xa], order=0)
        keep = m == 0
        idx, xa, ya = idx[keep], xa[keep], ya[keep]
        for c in range(3):
            stack[k, idx, c] = ndi.map_coordinates(s["img"][:, :, c].astype(np.float32), [ya, xa], order=1)
    nobs = np.isfinite(stack[:, :, 0]).sum(0)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(stack, axis=0)
    med[nobs < min_obs] = np.nan
    return med.reshape(shape + (3,)), nobs.reshape(shape)


def fill_holes(img: np.ndarray, default_bgr: tuple[float, float, float]) -> np.ndarray:
    """NaN texels -> a smooth fill (Telea inpaint at low resolution + the median colour where nothing is near)."""
    bad = ~np.isfinite(img[:, :, 0])
    out = np.where(np.isfinite(img), img, 0).astype(np.float32)
    if bad.all():
        out[:] = default_bgr
        return out.astype(np.uint8)
    h, w = bad.shape
    s = min(1.0, 400.0 / max(h, w))
    small = cv2.resize(out.astype(np.uint8), (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    msmall = cv2.resize(bad.astype(np.uint8) * 255, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
    fill = cv2.inpaint(small, msmall, 7, cv2.INPAINT_TELEA)
    fill = cv2.GaussianBlur(fill, (0, 0), 2.0)
    fill = cv2.resize(fill, (w, h), interpolation=cv2.INTER_LINEAR)
    out[bad] = fill[bad]
    return np.clip(out, 0, 255).astype(np.uint8)


def gain_match(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Per-channel gain that brings `src` to `ref` on the texels both have (median ratio); returns (3,)."""
    both = np.isfinite(src[:, :, 0]) & np.isfinite(ref[:, :, 0])
    if both.sum() < 200:
        return np.ones(3)
    return np.array([np.median(ref[both][:, c]) / max(np.median(src[both][:, c]), 1.0) for c in range(3)])
