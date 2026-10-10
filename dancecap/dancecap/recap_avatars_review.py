"""Review renders of the recap avatars' heads (agent A1): the REST head of a teacher with the pinned betas, with and without the
face identity / petite taper, beside the owner's portrait. Pure torch + numpy (dancecap.avatar_raster), CPU, seconds.

    python -m dancecap.recap_avatars head-review          -> review/avatar/head_<role>.jpg (+ head_<role>_grey.jpg)

Rows: template (the SMPL-X head the fit keeps), identity (FLAME identity transplant), the take TOML's full face block; columns:
front, 3/4, profile and the portrait's own head yaw. Framing as the 2026-10-08 face review (15 deg lens at 1.4 m on the head
joint - 2 cm, soft key + fill + ambient). The textured rows use the demo's baked albedo of the same person (the recap's own
albedo does not exist before the stitch): the face texture is aligned to the demo's head, so it is a likeness check of the
SHAPE, not of the recap's texture.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

VIEW_AZ = (("front", 0.0), ("3/4", 35.0), ("profile", 90.0))
SIZE, FOV, RADIUS, DROP = 420, 15.0, 1.4, 0.02


def srgb2lin(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def lin2srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def vnormals(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    out = np.zeros_like(v)
    for k in range(3):
        np.add.at(out, f[:, k], n)
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


def camera(target: np.ndarray, az_deg: float, el_deg: float = 0.0, radius: float = RADIUS):
    """World (y up, the dancer faces +z, her right = -x) -> camera: az + turns the camera toward her right."""
    a, e = np.radians(az_deg), np.radians(el_deg)
    dirv = np.array([-np.sin(a) * np.cos(e), np.sin(e), np.cos(a) * np.cos(e)])
    C = target + dirv * radius
    zc = target - C
    zc /= np.linalg.norm(zc)
    xc = np.cross(zc, [0.0, 1.0, 0.0])
    xc /= np.linalg.norm(xc)
    yc = np.cross(zc, xc)
    R = np.stack([xc, yc, zc])
    return R, -R @ C, (xc, yc, zc)


def render(V, F, UV, N, tex_lin, target, az_deg, size=SIZE, el_deg=0.0, fov=FOV, radius=RADIUS):
    """V (n,3) split-skin vertices, F (m,3), UV (n,2) Unity UV (v up), N (n,3), tex_lin (H,W,3) linear RGB (None = grey clay)."""
    import torch

    from .avatar_raster import pixel_bary, rasterize
    R, t, (xc, yc, zc) = camera(target, az_deg, el_deg, radius)
    f = size / 2 / np.tan(np.radians(fov) / 2)
    Pc = V @ R.T + t
    px = np.c_[f * Pc[:, 0] / Pc[:, 2] + size / 2, f * Pc[:, 1] / Pc[:, 2] + size / 2]
    pxt = torch.as_tensor(px, dtype=torch.float32)
    zt = torch.as_tensor(Pc[:, 2], dtype=torch.float32)
    Ft = torch.as_tensor(F, dtype=torch.long)
    _, fid = rasterize(torch, pxt, zt, Ft, size, size)
    bary = pixel_bary(torch, pxt, zt, Ft, fid).numpy()
    fid = fid.numpy()
    ys, xs = np.nonzero(fid >= 0)
    tri = F[fid[ys, xs]]
    b = bary[ys, xs]
    n = (N[tri] * b[..., None]).sum(1)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    if tex_lin is None:
        alb = np.full((len(ys), 3), 0.55, np.float32)
    else:
        uv = (UV[tri] * b[..., None]).sum(1)
        H, W = tex_lin.shape[:2]
        u = np.clip(uv[:, 0] * (W - 1), 0, W - 1)
        v = np.clip((1 - uv[:, 1]) * (H - 1), 0, H - 1)
        u0, v0 = np.floor(u).astype(int), np.floor(v).astype(int)
        u1, v1 = np.minimum(u0 + 1, W - 1), np.minimum(v0 + 1, H - 1)
        fu, fv = (u - u0)[:, None], (v - v0)[:, None]
        alb = ((1 - fu) * (1 - fv) * tex_lin[v0, u0] + fu * (1 - fv) * tex_lin[v0, u1] + (1 - fu) * fv * tex_lin[v1, u0]
               + fu * fv * tex_lin[v1, u1])
    key = -0.35 * xc - 0.55 * yc - 0.75 * zc
    fill = 0.6 * xc - 0.1 * yc - 0.8 * zc
    key /= np.linalg.norm(key)
    fill /= np.linalg.norm(fill)
    sh = 0.30 + 0.75 * np.clip(n @ key, 0, None) + 0.25 * np.clip(n @ fill, 0, None)
    img = np.full((size, size, 3), 0.08, np.float32)
    img[ys, xs] = alb * sh[:, None]
    return (lin2srgb(img) * 255).round().astype(np.uint8)


def load_albedo(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path))
    if img is None:
        return None
    return srgb2lin(img.astype(np.float32) / 255)   # BGR order: the renders are written with cv2


def head_tiles(sk, verts: np.ndarray, model_path, tex_lin, yaw_portrait: float, size: int = SIZE) -> list[np.ndarray]:
    """front / 3/4 / profile / portrait-yaw renders of the head of the rest skin `sk` with vertices `verts`."""
    from dataclasses import replace

    from .export_unity import textured_skin
    tsk = textured_skin(replace(sk, vertices=verts), model_path)
    N = vnormals(verts, sk.faces)[tsk.vertex_ids]
    keep = np.linalg.norm(tsk.vertices - sk.joints[15], axis=1) < 0.40
    fk = keep[tsk.faces].all(1)
    target = sk.joints[15] - np.array([0.0, DROP, 0.0])
    views = list(VIEW_AZ) + [("portrait yaw", yaw_portrait)]
    return [render(tsk.vertices, tsk.faces[fk], tsk.uv, N, tex_lin, target, az, size) for _, az in views]


def put(img: np.ndarray, text: str, org=(6, 20), scale=0.55) -> np.ndarray:
    cv2.putText(img, text, org, 0, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, 0, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def portrait_tile(path: Path, rec: dict, size: int = SIZE) -> np.ndarray:
    """The portrait around the face (the SMIRK crop box, a little wider)."""
    img = cv2.imread(str(path))
    c, s = np.asarray(rec["smirk_center"]), float(rec["smirk_side"]) * 1.25
    A = np.array([[size / s, 0, size / 2 - size / s * c[0]], [0, size / s, size / 2 - size / s * c[1]]])
    return cv2.warpAffine(img, A, (size, size), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)


def sheet(rows: list[tuple[str, list[np.ndarray]]], portrait: np.ndarray | None, size: int = SIZE) -> np.ndarray:
    """One row per variant (label on the first tile); the portrait in the last column of the first row, repeated as a reference
    strip at the bottom of the sheet."""
    out = []
    for label, tiles in rows:
        t0 = tiles[0].copy()
        put(t0, label)
        out.append(np.concatenate([t0] + tiles[1:], 1))
    sheet_ = np.concatenate(out, 0)
    if portrait is not None:
        pad = np.zeros((size, sheet_.shape[1], 3), np.uint8)
        pad[:, :size] = put(portrait.copy(), "portrait")
        sheet_ = np.concatenate([sheet_, pad], 0)
    return sheet_


def write_report(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, default=float))


def albedo_compare(takes: list[str], work_root: Path, out: Path, size: int = 560) -> Path:
    """review/avatar/albedo_compare.jpg: the 2048^2 UV albedos of several bakes side by side (one row per teacher, one column per work dir),
    each in its portrait-face composite when it has one next to it."""
    rows = []
    for role in ("lead", "follow"):
        tiles = []
        for t in takes:
            d = work_root / t / "avatar"
            f = d / f"{role}_albedo_face.png"
            f = f if f.exists() else d / f"{role}_albedo.png"
            img = cv2.imread(str(f))
            tile = np.zeros((size, size, 3), np.uint8) if img is None else cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
            put(tile, f"{role} {t.replace('larissa_kadu_recap_', '')}{' +face' if f.name.endswith('_face.png') else ''}")
            tiles.append(tile)
        rows.append(np.concatenate(tiles, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.concatenate(rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return out
