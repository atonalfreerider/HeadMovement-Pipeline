"""Pure-torch triangle rasteriser for the avatar texture stage (no nvdiffrast / CUDA extension build needed, so the
same code runs on the local 2080 Ti, on CPU in the tests and on a Linux Runpod pod).

Pixel convention = the rest of dancecap: pixel (u, v) has its centre at integer coordinates (u, v), i.e. projected
points are looked up with round(). Every triangle is expanded into the pixel centres of its bounding box (in face
chunks so memory stays bounded), tested with edge functions, and the nearest one wins through a single
scatter_reduce(amin) on a 64-bit key = (float32 depth bits << 32) | face id (positive float32 bit patterns order
like the floats). Depth is perspective-correct (1/z interpolated in screen space).

    zbuf, fid = rasterize(torch, px, z, faces, H, W)          # (H, W) depth (1e9 = empty), face id (-1 = empty)
    bary = pixel_bary(torch, px, z, faces, fid)                 # (H, W, 3) perspective-correct barycentrics
"""
from __future__ import annotations

EMPTY_Z = 1e9


def rasterize(torch, px, z, faces, H: int, W: int, near: float = 0.05, max_pairs: int = 6_000_000,
              max_tri_px: int = 400_000):
    """px (V,2) pixel coords, z (V,) camera depth, faces (F,3) long -> zbuf (H,W) float32, fid (H,W) int64.

    Triangles with a vertex closer than `near` (behind the camera) or covering more than max_tri_px pixel centres
    (degenerate, a few cm from the lens) are skipped."""
    dev = px.device
    faces = faces.long()
    tp = px[faces].double()  # (F,3,2): double for the edge functions (pixel coords up to ~4000)
    tz = z[faces].double()
    x0 = torch.ceil(tp[..., 0].min(1).values).clamp(0, W - 1)
    x1 = torch.floor(tp[..., 0].max(1).values).clamp(-1, W - 1)
    y0 = torch.ceil(tp[..., 1].min(1).values).clamp(0, H - 1)
    y1 = torch.floor(tp[..., 1].max(1).values).clamp(-1, H - 1)
    w = (x1 - x0 + 1).clamp(min=0).long()
    h = (y1 - y0 + 1).clamp(min=0).long()
    n = w * h
    A = tp[:, 0]
    area = (tp[:, 1, 0] - A[:, 0]) * (tp[:, 2, 1] - A[:, 1]) - (tp[:, 1, 1] - A[:, 1]) * (tp[:, 2, 0] - A[:, 0])
    ok = (tz > near).all(1) & (n > 0) & (n <= max_tri_px) & (area.abs() > 1e-9)
    idx = torch.nonzero(ok).squeeze(1)
    key = torch.full((H * W,), torch.iinfo(torch.int64).max, dtype=torch.int64, device=dev)
    if len(idx):
        cum = torch.cumsum(n[idx], 0)
        a = 0
        while a < len(idx):
            # largest b with sum(n[idx[a:b]]) <= max_pairs (at least one face)
            base = cum[a - 1] if a > 0 else torch.zeros((), dtype=cum.dtype, device=dev)
            b = int(torch.searchsorted(cum, base + max_pairs, right=True).item())
            b = max(b, a + 1)
            _raster_chunk(torch, idx[a:b], tp, tz, x0, y0, w, n, area, key, W)
            a = b
    empty = key == torch.iinfo(torch.int64).max
    fid = torch.where(empty, torch.full_like(key, -1), key & 0xFFFFFFFF)
    zbits = (key >> 32).to(torch.int32)
    zbuf = torch.where(empty, torch.full((H * W,), EMPTY_Z, device=dev), zbits.view(torch.float32))
    return zbuf.view(H, W), fid.view(H, W)


def _raster_chunk(torch, fi, tp, tz, x0, y0, w, n, area, key, W):
    cnt = n[fi]
    f = torch.repeat_interleave(fi, cnt)
    start = torch.cumsum(cnt, 0) - cnt
    local = torch.arange(len(f), device=fi.device) - torch.repeat_interleave(start, cnt)
    wf = w[f]
    X = x0[f] + (local % wf).double()
    Y = y0[f] + (local // wf).double()
    t = tp[f]  # (P,3,2)
    # edge functions: b0 opposite vertex 0 etc., normalised by the signed area
    def edge(i, j):
        return (t[:, j, 0] - t[:, i, 0]) * (Y - t[:, i, 1]) - (t[:, j, 1] - t[:, i, 1]) * (X - t[:, i, 0])
    ar = area[f]
    b0 = edge(1, 2) / ar
    b1 = edge(2, 0) / ar
    b2 = 1.0 - b0 - b1
    inside = (b0 >= -1e-9) & (b1 >= -1e-9) & (b2 >= -1e-9)
    zt = tz[f]
    inv = b0 / zt[:, 0] + b1 / zt[:, 1] + b2 / zt[:, 2]
    zz = (1.0 / inv.clamp(min=1e-12)).float()
    sel = inside & (zz > 0)
    if not sel.any():
        return
    pix = (Y[sel].long() * W + X[sel].long())
    k = (zz[sel].contiguous().view(torch.int32).long() << 32) | f[sel]
    key.scatter_reduce_(0, pix, k, reduce="amin", include_self=True)


def pixel_bary(torch, px, z, faces, fid):
    """Perspective-correct barycentrics (H,W,3) of each covered pixel centre in its face (zeros where fid < 0)."""
    H, W = fid.shape
    out = torch.zeros((H, W, 3), device=px.device)
    cov = fid >= 0
    if not cov.any():
        return out
    ys, xs = torch.nonzero(cov, as_tuple=True)
    f = faces.long()[fid[ys, xs]]
    t = px[f].double()
    zt = z[f].double()
    X, Y = xs.double(), ys.double()
    ar = (t[:, 1, 0] - t[:, 0, 0]) * (t[:, 2, 1] - t[:, 0, 1]) - (t[:, 1, 1] - t[:, 0, 1]) * (t[:, 2, 0] - t[:, 0, 0])

    def edge(i, j):
        return (t[:, j, 0] - t[:, i, 0]) * (Y - t[:, i, 1]) - (t[:, j, 1] - t[:, i, 1]) * (X - t[:, i, 0])
    b = torch.stack([edge(1, 2) / ar, edge(2, 0) / ar], 1)
    b = torch.cat([b, 1 - b.sum(1, keepdim=True)], 1).clamp(min=0)
    b = b / zt
    b = b / b.sum(1, keepdim=True).clamp(min=1e-12)
    out[ys, xs] = b.float()
    return out
