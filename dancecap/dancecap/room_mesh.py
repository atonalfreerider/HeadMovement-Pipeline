"""Quest mesh of the static room from the trained splat: explicit floor plane + TSDF fusion of the rendered depth
above it, decimation, texture bake.

The floor is the SfM floor plane (thousands of sparse inliers within 2 cm), meshed as a 25 cm quad grid over the
region where the floor is the first visible surface (closed + hole-filled: the floor under seated spectators is never
seen) with its own orthographic texture (splat diffuse colour; unobserved texels inpainted). Its glossy reflections
are not in the texture beyond what the diffuse splat kept (ceiling-light streaks: the lights are never seen directly).

The splat (people-free by construction: people were masked out of its loss) is rendered from every training camera
(expected depth + canonical colour, i.e. without the per-image colour transform). Pixels that are not opaque
(alpha < min_alpha) or sit on a depth edge (blended depth) are dropped, the rest is fused into an Open3D scalable TSDF,
the surface is cropped to the room box / above the floor plane, small islands are removed and the result is quadric-
decimated to <= mesh_max_tris. Colour: vertex colours (room_mesh.ply) and a texture baked from the splat renders onto
an xatlas atlas (room_mesh.obj + .mtl + room_mesh_tex.png): every texel takes the colour of the views that see it
(depth test against the rendered depth), weighted towards the most frontal, closest view.

Everything stays in the SfM frame (room/transform.json has world_from_sfm). A small software renderer (painter's
order + triangle-id buffer) draws the mesh for the review sheets without an OpenGL context.
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np


def render_views(splat, views, max_views: int | None = None):
    """Canonical-colour RGB, expected depth and alpha of the splat from each view (numpy, loaded resolution)."""
    from .room import render_view
    out = []
    for v in views[:max_views]:
        r = render_view(splat, v)
        out.append({"rgb": r["direct"], "depth": r["depth"], "alpha": r["alpha"]})
    return out


def depth_ok(depth: np.ndarray, alpha: np.ndarray, min_alpha: float = 0.9, max_rel_jump: float = 0.03,
             max_depth: float = 20.0) -> np.ndarray:
    """Opaque pixels away from depth discontinuities (where expected depth blends two surfaces)."""
    d = depth
    jump = np.zeros_like(d)
    jump[:, 1:] = np.maximum(jump[:, 1:], np.abs(d[:, 1:] - d[:, :-1]))
    jump[:, :-1] = np.maximum(jump[:, :-1], np.abs(d[:, 1:] - d[:, :-1]))
    jump[1:] = np.maximum(jump[1:], np.abs(d[1:] - d[:-1]))
    jump[:-1] = np.maximum(jump[:-1], np.abs(d[1:] - d[:-1]))
    return (alpha > min_alpha) & (d > 0.1) & (d < max_depth) & (jump < max_rel_jump * np.maximum(d, 1e-3))


def tsdf_fuse(views, renders, voxel: float, trunc_mult: float = 4.0, max_depth: float = 20.0):
    import open3d as o3d
    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel, sdf_trunc=trunc_mult * voxel,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    used = 0
    for v, r in zip(views, renders):
        ok = depth_ok(r["depth"], r["alpha"], max_depth=max_depth)
        if ok.mean() < 0.05:
            continue
        depth = np.where(ok, r["depth"], 0).astype(np.float32)
        color = np.ascontiguousarray((r["rgb"] * 255).clip(0, 255).astype(np.uint8))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color), o3d.geometry.Image(depth), depth_scale=1.0, depth_trunc=max_depth,
            convert_rgb_to_intensity=False)
        intr = o3d.camera.PinholeCameraIntrinsic(v.W, v.H, v.K[0, 0], v.K[1, 1], v.K[0, 2], v.K[1, 2])
        vol.integrate(rgbd, intr, v.viewmat())
        used += 1
    return vol.extract_triangle_mesh(), used


def clean_mesh(mesh, geo: dict, pad: float, floor_margin: float, min_cluster_frac: float = 0.002):
    """Crop to the room box, drop geometry under the floor plane / behind the mirror, remove small islands."""
    import torch
    from .room import behind_reflectors
    V = np.asarray(mesh.vertices)
    keep = np.all((V >= geo["box_lo"] - pad) & (V <= geo["box_hi"] + pad), axis=1) &         ~behind_reflectors(torch.as_tensor(V, dtype=torch.float32), geo, floor_margin).numpy()
    mesh.remove_vertices_by_mask(~keep)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    if len(mesh.triangles):
        cl, n_tri, _ = mesh.cluster_connected_triangles()
        cl, n_tri = np.asarray(cl), np.asarray(n_tri)
        small = n_tri[cl] < min_cluster_frac * len(cl)
        mesh.remove_triangles_by_mask(small)
        mesh.remove_unreferenced_vertices()
    return mesh


# ------------------------------------------------------------------------------------------- texture baking

def raster_uv(uv: np.ndarray, faces: np.ndarray, size: int) -> np.ndarray:
    """Triangle id per texel (-1 = gutter). uv in [0, 1], v up (OBJ convention)."""
    ids = np.full((size, size), -1, np.int32)
    px = np.stack([uv[:, 0] * size - 0.5, (1 - uv[:, 1]) * size - 0.5], 1)
    pts = np.round(px[faces] * 16).astype(np.int32)  # 4 fractional bits
    for f in range(len(faces)):
        cv2.fillConvexPoly(ids, pts[f], int(f), lineType=cv2.LINE_8, shift=4)
    return ids


def barycentric(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Barycentric coordinates of 2D points p (N, 2) in triangles (a, b, c) (N, 2 each), clamped and renormalised."""
    v0, v1, v2 = b - a, c - a, p - a
    d00 = (v0 * v0).sum(1)
    d01 = (v0 * v1).sum(1)
    d11 = (v1 * v1).sum(1)
    d20 = (v2 * v0).sum(1)
    d21 = (v2 * v1).sum(1)
    den = d00 * d11 - d01 * d01
    den = np.where(np.abs(den) < 1e-12, 1e-12, den)
    w1 = (d11 * d20 - d01 * d21) / den
    w2 = (d00 * d21 - d01 * d20) / den
    w = np.stack([1 - w1 - w2, w1, w2], 1).clip(0, 1)
    return w / np.maximum(w.sum(1, keepdims=True), 1e-9)


def bake_texture(V, F, uv, vcol, views, renders, size: int, power: float = 4.0, log=print) -> tuple[np.ndarray, dict]:
    """Project the splat renders onto the atlas. Returns (size, size, 3) uint8 RGB and stats."""
    import torch
    t0 = time.time()
    ids = raster_uv(uv, F, size)
    tex_idx = np.nonzero(ids >= 0)
    fid = ids[tex_idx]
    px = np.stack([tex_idx[1] + 0.5, tex_idx[0] + 0.5], 1) / size
    px[:, 1] = 1 - px[:, 1]
    tri_uv = uv[F[fid]]
    w = barycentric(px, tri_uv[:, 0], tri_uv[:, 1], tri_uv[:, 2])
    P = (V[F[fid]] * w[..., None]).sum(1)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
    N = fn[fid]
    base = (vcol[F[fid]] * w[..., None]).sum(1)
    dev = "cuda"
    Pt = torch.as_tensor(P, dtype=torch.float32, device=dev)
    Nt = torch.as_tensor(N, dtype=torch.float32, device=dev)
    acc = torch.zeros(len(P), 3, device=dev)
    wsum = torch.zeros(len(P), device=dev)
    for v, r in zip(views, renders):
        R = torch.as_tensor(v.R, dtype=torch.float32, device=dev)
        t = torch.as_tensor(v.tvec, dtype=torch.float32, device=dev)
        Xc = Pt @ R.T + t
        z = Xc[:, 2]
        u = v.K[0, 0] * Xc[:, 0] / z.clamp_min(1e-3) + v.K[0, 2]
        vv = v.K[1, 1] * Xc[:, 1] / z.clamp_min(1e-3) + v.K[1, 2]
        inb = (z > 0.1) & (u > 0) & (u < v.W - 1) & (vv > 0) & (vv < v.H - 1)
        if not inb.any():
            continue
        D = torch.as_tensor(r["depth"], dtype=torch.float32, device=dev)
        A = torch.as_tensor(r["alpha"], dtype=torch.float32, device=dev)
        I = torch.as_tensor(r["rgb"], dtype=torch.float32, device=dev)
        grid = torch.stack([(u + 0.5) / v.W * 2 - 1, (vv + 0.5) / v.H * 2 - 1], -1)[None, None]
        samp = lambda img: torch.nn.functional.grid_sample(img.permute(2, 0, 1)[None] if img.dim() == 3 else
                                                           img[None, None], grid, align_corners=False)[0, :, 0].T
        d_s = samp(D)[:, 0]
        a_s = samp(A)[:, 0]
        c_s = samp(I)
        vis = inb & (a_s > 0.9) & ((d_s - z).abs() < 0.03 + 0.02 * z)
        cen = torch.as_tensor(v.center, dtype=torch.float32, device=dev)
        vd = cen - Pt
        dist = vd.norm(dim=1).clamp_min(1e-3)
        cos = ((vd / dist[:, None]) * Nt).sum(1).abs()
        wt = torch.where(vis, (cos * v.K[0, 0] / dist).clamp_min(0) ** power, torch.zeros_like(z))
        acc += wt[:, None] * c_s
        wsum += wt
    seen = (wsum > 0).cpu().numpy()
    col = np.where(seen[:, None], (acc / wsum.clamp_min(1e-12)[:, None]).cpu().numpy(), base)
    tex = np.zeros((size, size, 3), np.float32)
    tex[tex_idx] = col
    filled = ids >= 0
    # pad the gutters so bilinear / mip sampling does not bleed black into the charts
    tex8 = (tex.clip(0, 1) * 255).astype(np.uint8)
    hole = (~filled).astype(np.uint8)
    for _ in range(8):
        dil = cv2.dilate(tex8, np.ones((3, 3), np.uint8))
        grown = cv2.dilate(filled.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        new = grown & ~filled
        tex8[new] = dil[new]
        filled = grown
    stats = {"texels": int(len(P)), "texels_seen_frac": float(seen.mean()), "bake_s": round(time.time() - t0, 1),
             "atlas_fill": float((ids >= 0).mean()), "gutter_left": int(hole.sum())}
    return tex8, stats


def write_obj(path: Path, V, F, uv, tex_name: str) -> None:
    mtl = path.with_suffix(".mtl")
    mtl.write_text(f"newmtl room\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nillum 1\nmap_Kd {tex_name}\n")
    with open(path, "w") as f:
        f.write(f"# dancecap room mesh, SfM frame (see transform.json)\nmtllib {mtl.name}\nusemtl room\n")
        f.write("".join(f"v {a:.5f} {b:.5f} {c:.5f}\n" for a, b, c in V))
        f.write("".join(f"vt {a:.6f} {b:.6f}\n" for a, b in uv))
        f.write("".join(f"f {a}/{a} {b}/{b} {c}/{c}\n" for a, b, c in F + 1))


# --------------------------------------------------------------------------------------- software renderer

def bilinear(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear lookup of img (H, W, C) at float pixel coordinates (pixel centres at integers), edge-clamped."""
    H, W = img.shape[:2]
    x = np.clip(x, 0, W - 1)
    y = np.clip(y, 0, H - 1)
    x0 = np.minimum(np.floor(x).astype(np.int64), W - 2) if W > 1 else np.zeros(len(x), np.int64)
    y0 = np.minimum(np.floor(y).astype(np.int64), H - 2) if H > 1 else np.zeros(len(y), np.int64)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    x1, y1 = np.minimum(x0 + 1, W - 1), np.minimum(y0 + 1, H - 1)
    return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x1] * fx * (1 - fy) + img[y1, x0] * (1 - fx) * fy +
            img[y1, x1] * fx * fy)


def render_mesh(V, F, viewmat: np.ndarray, K: np.ndarray, W: int, H: int, vcol=None, uv=None, tex=None,
                shade: bool = False, bg=(30, 30, 30), near: float = 0.1, ortho: bool = False, ftex=None) -> np.ndarray:
    """Painter's-order triangle-id raster + per-pixel barycentric colour (vertex colours or texture(s): tex may be a
    list indexed by the per-face ftex). RGB uint8."""
    Xc = V @ viewmat[:3, :3].T + viewmat[:3, 3]
    z = Xc[:, 2]
    if ortho:
        p = Xc[:, :2] * np.array([K[0, 0], K[1, 1]]) + K[:2, 2]
    else:
        p = Xc[:, :2] / np.maximum(z, 1e-6)[:, None] * np.array([K[0, 0], K[1, 1]]) + K[:2, 2]
    zf = z[F]
    pf = p[F]
    ok = (zf > near).all(1) & (pf[..., 0].max(1) > 0) & (pf[..., 0].min(1) < W) & (pf[..., 1].max(1) > 0) & \
        (pf[..., 1].min(1) < H) & (np.abs(pf).max((1, 2)) < 1e5)
    order = np.nonzero(ok)[0]
    order = order[np.argsort(-zf[order].mean(1))]
    ids = np.full((H, W), -1, np.int32)
    pts = np.round(pf * 16).astype(np.int64)
    for f in order:
        cv2.fillConvexPoly(ids, pts[f].astype(np.int32), int(f), lineType=cv2.LINE_8, shift=4)
    yy, xx = np.nonzero(ids >= 0)
    out = np.empty((H, W, 3), np.float32)
    out[:] = np.array(bg, np.float32) / 255
    if not len(yy):
        return (out * 255).astype(np.uint8)
    fid = ids[yy, xx]
    q = np.stack([xx + 0.5, yy + 0.5], 1) - 0.5
    w = barycentric(q, pf[fid, 0], pf[fid, 1], pf[fid, 2])
    if not ortho:  # perspective-correct interpolation
        w = w / zf[fid]
        w /= w.sum(1, keepdims=True)
    if tex is not None and uv is not None:
        t_uv = (uv[F[fid]] * w[..., None]).sum(1)
        texs = tex if isinstance(tex, (list, tuple)) else [tex]
        tid = ftex[fid] if ftex is not None else np.zeros(len(fid), int)
        c = np.zeros((len(fid), 3), np.float32)
        for k, tx_img in enumerate(texs):
            sel = tid == k
            if sel.any():
                tx = t_uv[sel, 0] * tx_img.shape[1] - 0.5
                ty = (1 - t_uv[sel, 1]) * tx_img.shape[0] - 0.5
                c[sel] = bilinear(tx_img.astype(np.float32) / 255, tx, ty)
    elif vcol is not None:
        c = (vcol[F[fid]] * w[..., None]).sum(1)
    else:
        c = np.full((len(fid), 3), 0.8, np.float32)
    if shade:
        fn = np.cross(Xc[F[:, 1]] - Xc[F[:, 0]], Xc[F[:, 2]] - Xc[F[:, 0]])
        fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
        ray = Xc[F].mean(1)
        ray /= np.maximum(np.linalg.norm(ray, axis=1, keepdims=True), 1e-12)
        lam = np.abs((fn * ray).sum(1))[fid]
        c = c * (0.25 + 0.75 * lam[:, None])
    out[yy, xx] = c
    return (out.clip(0, 1) * 255).astype(np.uint8)


# ------------------------------------------------------------------------------------------------- floor

def floor_basis(geo: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Orthonormal (u, w) on the floor plane (u ~ world X, w ~ world Z) and the origin (dancers' floor point)."""
    n = np.asarray(geo["floor_n"], float)
    X = geo["world"]["R"][0]  # world X axis in SfM coordinates
    u = X - (X @ n) * n
    u /= np.linalg.norm(u)
    w = np.cross(u, n)  # right-handed (u, n, w) ~ (X, Y, Z)
    o = geo["origin"] - (geo["origin"] @ n + geo["floor_d"]) * n
    return n, u, w, o


def _floor_hits(v, r, geo):
    """Per pixel: floor-plane hit (u, w), its depth z, and whether the floor is the first surface there."""
    n, u, w, o = floor_basis(geo)
    yy, xx = np.mgrid[0:v.H, 0:v.W]
    ray = np.stack([(xx - v.K[0, 2]) / v.K[0, 0], (yy - v.K[1, 2]) / v.K[1, 1], np.ones(xx.shape)], -1)
    nc = v.R @ n
    dc = geo["floor_d"] - nc @ v.tvec
    den = ray @ nc
    z = -dc / np.where(np.abs(den) < 1e-9, 1e-9, den)
    X = (ray * z[..., None] - v.tvec) @ v.R
    first = (z > 0.3) & (z < 25) & (r["alpha"] > 0.5) & (r["depth"] > 0.85 * z)
    return (X - o) @ u, (X - o) @ w, z, first


def floor_region(views, renders, valids, geo, cell: float, min_obs: int = 3):
    """Floor cells (cell x cell, SfM units) seen as the first surface in >= min_obs unmasked photo pixels; closed and
    hole-filled (the floor under seated spectators is never seen but is certainly floor)."""
    from scipy import ndimage
    us, ws = [], []
    for v, r, ok in zip(views, renders, valids):
        a, b, z, first = _floor_hits(v, r, geo)
        sel = first & ok
        us.append(a[sel][::4])
        ws.append(b[sel][::4])
    us, ws = np.concatenate(us), np.concatenate(ws)
    lo = np.array([np.percentile(us, 0.2), np.percentile(ws, 0.2)]) - cell
    hi = np.array([np.percentile(us, 99.8), np.percentile(ws, 99.8)]) + cell
    nu, nw = np.ceil((hi - lo) / cell).astype(int)
    cnt = np.zeros((nw, nu), int)
    iu = np.clip(((us - lo[0]) / cell).astype(int), 0, nu - 1)
    iw = np.clip(((ws - lo[1]) / cell).astype(int), 0, nw - 1)
    np.add.at(cnt, (iw, iu), 1)
    reg = cnt >= min_obs
    reg = ndimage.binary_closing(reg, np.ones((3, 3)), iterations=2)
    reg = ndimage.binary_fill_holes(reg)
    lab, k = ndimage.label(reg)
    if k > 1:
        sizes = ndimage.sum(reg, lab, range(1, k + 1))
        reg = np.isin(lab, 1 + np.nonzero(sizes >= 0.02 * sizes.max())[0])
    return reg, lo, cell, cnt


def floor_mesh(reg: np.ndarray, lo: np.ndarray, cell: float, geo: dict):
    """Quad grid over the region (2 triangles per cell, shared vertices), UV = position in the region's box."""
    n, u, w, o = floor_basis(geo)
    nw, nu = reg.shape
    vid = -np.ones((nw + 1, nu + 1), int)
    jj, ii = np.nonzero(reg)
    for dj, di in ((0, 0), (0, 1), (1, 0), (1, 1)):
        vid[jj + dj, ii + di] = 0
    vj, vi = np.nonzero(vid == 0)
    vid[vj, vi] = np.arange(len(vj))
    P2 = np.stack([lo[0] + vi * cell, lo[1] + vj * cell], 1)
    V = o + P2[:, :1] * u + P2[:, 1:] * w
    a, b, c, d = vid[jj, ii], vid[jj, ii + 1], vid[jj + 1, ii + 1], vid[jj + 1, ii]
    # winding: normal along +n (u x w = -n, so go a -> d -> c)
    F = np.concatenate([np.stack([a, d, c], 1), np.stack([a, c, b], 1)])
    uv = np.stack([vi / nu, 1 - vj / nw], 1)
    return V, F, uv


def bake_floor(reg, lo, cell, geo, views, renders, valids, px_per_unit: float, max_side: int = 4096,
               power: float = 2.0):
    """Orthographic floor texture: each texel takes the splat's diffuse colour from the views where the floor is the
    first surface and the photo pixel is not a person (closest/most frontal views weighted up); unobserved texels
    inside the region are inpainted."""
    import torch
    n, u, w, o = floor_basis(geo)
    nw, nu = reg.shape
    ext = np.array([nu, nw]) * cell
    s = min(px_per_unit, max_side / ext.max())
    TW, TH = int(round(ext[0] * s)), int(round(ext[1] * s))
    ty, tx = np.mgrid[0:TH, 0:TW]
    pu = lo[0] + (tx + 0.5) / s
    pw = lo[1] + (ty + 0.5) / s
    dev = "cuda"
    P = torch.as_tensor(o + pu.reshape(-1, 1) * u + pw.reshape(-1, 1) * w, dtype=torch.float32, device=dev)
    nt = torch.as_tensor(n, dtype=torch.float32, device=dev)
    acc = torch.zeros(len(P), 3, device=dev)
    wsum = torch.zeros(len(P), device=dev)
    for v, r, ok in zip(views, renders, valids):
        R = torch.as_tensor(v.R, dtype=torch.float32, device=dev)
        t = torch.as_tensor(v.tvec, dtype=torch.float32, device=dev)
        Xc = P @ R.T + t
        z = Xc[:, 2]
        uu = v.K[0, 0] * Xc[:, 0] / z.clamp_min(1e-3) + v.K[0, 2]
        vv = v.K[1, 1] * Xc[:, 1] / z.clamp_min(1e-3) + v.K[1, 2]
        inb = (z > 0.3) & (uu > 0) & (uu < v.W - 1) & (vv > 0) & (vv < v.H - 1)
        if not inb.any():
            continue
        grid = torch.stack([(uu + 0.5) / v.W * 2 - 1, (vv + 0.5) / v.H * 2 - 1], -1)[None, None]
        smp = lambda img, mode="bilinear": torch.nn.functional.grid_sample(
            torch.as_tensor(img, dtype=torch.float32, device=dev).permute(2, 0, 1)[None] if img.ndim == 3 else
            torch.as_tensor(img, dtype=torch.float32, device=dev)[None, None], grid, mode=mode,
            align_corners=False)[0, :, 0].T
        d_s = smp(r["depth"])[:, 0]
        a_s = smp(r["alpha"])[:, 0]
        ok_s = smp(ok.astype(np.float32), "nearest")[:, 0]
        vis = inb & (a_s > 0.5) & (d_s > 0.85 * z) & (ok_s > 0.5)
        cen = torch.as_tensor(v.center, dtype=torch.float32, device=dev)
        vd = cen - P
        dist = vd.norm(dim=1).clamp_min(1e-3)
        cos = ((vd / dist[:, None]) @ nt).abs()
        wt = torch.where(vis, (cos.sqrt() * v.K[0, 0] / dist) ** power, torch.zeros_like(z))
        acc += wt[:, None] * smp(r["rgb"])
        wsum += wt
    seen = (wsum > 0).cpu().numpy().reshape(TH, TW)
    col = (acc / wsum.clamp_min(1e-12)[:, None]).cpu().numpy().reshape(TH, TW, 3)
    tex = (col.clip(0, 1) * 255).astype(np.uint8)
    inside = cv2.resize(reg.astype(np.uint8), (TW, TH), interpolation=cv2.INTER_NEAREST).astype(bool)
    hole = (inside & ~seen).astype(np.uint8)
    if hole.any():
        # inpaint at reduced resolution (fast), then upsample into the holes
        f = max(1, int(max(TW, TH) / 1024))
        small = cv2.resize(tex, (TW // f, TH // f), interpolation=cv2.INTER_AREA)
        hs = cv2.resize(hole, (TW // f, TH // f), interpolation=cv2.INTER_NEAREST)
        hs = cv2.dilate(hs, np.ones((3, 3), np.uint8))
        filled = cv2.inpaint(small, hs, 9, cv2.INPAINT_TELEA)
        up = cv2.resize(filled, (TW, TH), interpolation=cv2.INTER_LINEAR)
        tex[hole.astype(bool)] = up[hole.astype(bool)]
    stats = {"texture_px": [TW, TH], "texel_m": float(geo["world"]["s"] / s), "seen_frac": float(seen[inside].mean()),
             "area_m2": float(reg.sum() * (cell * geo["world"]["s"]) ** 2)}
    return tex, stats


# ------------------------------------------------------------------------------------------------- driver

def write_obj_parts(path: Path, parts: list[dict]) -> None:
    """One OBJ with a material per part: parts = [{name, V, F, uv, tex_file}]."""
    mtl = path.with_suffix(".mtl")
    mtl.write_text("".join(f"newmtl {p['name']}\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nillum 1\nmap_Kd {p['tex_file']}\n\n"
                           for p in parts))
    with open(path, "w") as f:
        f.write(f"# dancecap room mesh, SfM frame (see transform.json)\nmtllib {mtl.name}\n")
        off = 0
        for p in parts:
            f.write(f"o {p['name']}\n")
            f.write("".join(f"v {a:.5f} {b:.5f} {c:.5f}\n" for a, b, c in p["V"]))
            f.write("".join(f"vt {a:.6f} {b:.6f}\n" for a, b in p["uv"]))
            f.write(f"usemtl {p['name']}\n")
            f.write("".join(f"f {a}/{a} {b}/{b} {c}/{c}\n" for a, b, c in p["F"] + 1 + off))
            off += len(p["V"])


def build(work: Path, splat, views, geo: dict, cfg, log=print) -> dict:
    """Quest mesh = explicit floor (plane grid + orthographic floor texture) + TSDF of everything above the floor."""
    import open3d as o3d

    from .room import load_view
    out = work / "room"
    t0 = time.time()
    tr = [v for v in views if v.split == "train"]
    renders = render_views(splat, tr)
    valids = [load_view(work, v, cfg)[1] for v in tr]
    s = geo["world"]["s"]
    # floor
    cell = cfg.floor_cell / s
    reg, lo, cell, cnt = floor_region(tr, renders, valids, geo, cell)
    Vf, Ff, uvf = floor_mesh(reg, lo, cell, geo)
    ftex, fst = bake_floor(reg, lo, cell, geo, tr, renders, valids, px_per_unit=s / cfg.floor_texel,
                           max_side=cfg.floor_tex_max)
    cv2.imwrite(str(out / "room_floor_tex.png"), ftex[:, :, ::-1])
    log(f"[room] mesh: floor plane {fst['area_m2']:.0f} m^2, {len(Ff):,} triangles, texture {fst['texture_px']} "
        f"({fst['texel_m'] * 1000:.1f} mm/texel, {fst['seen_frac']:.0%} observed, rest inpainted)")
    # everything above the floor band: TSDF of the rendered depth
    mesh, used = tsdf_fuse(tr, renders, cfg.mesh_voxel)
    n_raw = len(mesh.triangles)
    mesh = clean_mesh(mesh, geo, cfg.room_pad, cfg.floor_margin)
    Vt = np.asarray(mesh.vertices)
    band = Vt @ geo["floor_n"] + geo["floor_d"] < cfg.floor_band / s
    mesh.remove_vertices_by_mask(band)
    mesh = clean_mesh(mesh, geo, cfg.room_pad, cfg.floor_margin)
    n_clean = len(mesh.triangles)
    log(f"[room] mesh: TSDF {cfg.mesh_voxel * 100:.0f} cm from {used} views -> {n_raw:,} triangles, "
        f"{n_clean:,} above the floor band after crop/islands ({time.time() - t0:.0f}s)")
    budget = cfg.mesh_max_tris - len(Ff)
    if n_clean > budget:
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=budget)
        mesh = clean_mesh(mesh, geo, cfg.room_pad, cfg.floor_margin, min_cluster_frac=0.0005)
    V = np.asarray(mesh.vertices).astype(np.float64)
    F = np.asarray(mesh.triangles).astype(np.int64)
    vcol = np.asarray(mesh.vertex_colors).astype(np.float32)
    # floor vertex colours from its texture (for the PLY)
    vcf = bilinear(ftex.astype(np.float32) / 255, uvf[:, 0] * ftex.shape[1] - 0.5,
                   (1 - uvf[:, 1]) * ftex.shape[0] - 0.5)
    allV = np.concatenate([V, Vf])
    allF = np.concatenate([F, Ff + len(V)])
    m_all = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(allV), o3d.utility.Vector3iVector(allF))
    m_all.vertex_colors = o3d.utility.Vector3dVector(np.concatenate([vcol, vcf]).astype(np.float64))
    m_all.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(out / "room_mesh.ply"), m_all, write_ascii=False)
    rep = {"tsdf_voxel_m": cfg.mesh_voxel, "views_fused": used, "triangles_raw": n_raw, "triangles_clean": n_clean,
           "triangles": int(len(allF)), "vertices": int(len(allV)), "floor": {"triangles": int(len(Ff)), **fst},
           "walls_triangles": int(len(F))}
    h = allV @ geo["floor_n"] + geo["floor_d"]
    rep["below_floor_frac"] = float((h < -0.03).mean())
    m = geo.get("mirror")
    if m is not None and "n" in m:
        import torch

        from .room import behind_mirror
        rep["behind_mirror_frac"] = float(behind_mirror(torch.as_tensor(allV, dtype=torch.float32), geo, 0.05)
                                          .float().mean())
    rep["height_pct_sfm"] = np.percentile(h, [0.5, 5, 50, 95, 99.5]).round(3).tolist()
    try:
        import xatlas
        t1 = time.time()
        vmap, idx, uv = xatlas.parametrize(V.astype(np.float32), F.astype(np.uint32))
        V2, F2, uv = V[vmap], idx.astype(np.int64), uv.astype(np.float64)
        vcol2 = vcol[vmap]
        log(f"[room] mesh: xatlas {len(V2):,} vertices, {time.time() - t1:.0f}s")
        tex, st = bake_texture(V2, F2, uv, vcol2, tr, renders, cfg.mesh_tex, log=log)
        cv2.imwrite(str(out / "room_mesh_tex.png"), tex[:, :, ::-1])
        write_obj_parts(out / "room_mesh.obj", [
            {"name": "room", "V": V2, "F": F2, "uv": uv, "tex_file": "room_mesh_tex.png"},
            {"name": "floor", "V": Vf, "F": Ff, "uv": uvf, "tex_file": "room_floor_tex.png"}])
        np.savez_compressed(out / "cache" / "room_mesh_uv.npz", V=np.concatenate([V2, Vf]),
                            F=np.concatenate([F2, Ff + len(V2)]), uv=np.concatenate([uv, uvf]),
                            vcol=np.concatenate([vcol2, vcf]).astype(np.float32),
                            ftex=np.concatenate([np.zeros(len(F2), int), np.ones(len(Ff), int)]))
        rep["texture"] = {"size": cfg.mesh_tex, **st, "obj_vertices": int(len(V2) + len(Vf))}
        log(f"[room] mesh: baked {cfg.mesh_tex}^2 atlas, {st['texels_seen_frac']:.0%} of texels seen by a view "
            f"({st['bake_s']}s)")
    except ImportError:
        log("[room] mesh: xatlas missing -> vertex colours only (pip install xatlas)")
    rep["mesh_s"] = round(time.time() - t0, 1)
    rep["files"] = sorted(p.name for p in out.glob("room_*") if p.suffix in (".ply", ".obj", ".mtl", ".png"))
    log(f"[room] room_mesh: {len(allF):,} triangles ({len(Ff):,} floor), {len(allV):,} vertices, below floor "
        f"{rep['below_floor_frac']:.2%}")
    return rep
