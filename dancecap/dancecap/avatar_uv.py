"""SMPL-X UV atlas + posed-body geometry for the avatar texture stage (avatar_texture.py).

UV convention (also Unity's): texel (px, py) of an S x S image (row 0 = top) has its centre at
u = (px + 0.5) / S, v = 1 - (py + 0.5) / S. The SMPL-X npz carries vt (11313, 2) and ft (20908, 3); every vt index
belongs to exactly one mesh vertex, so the seam-split mesh is simply indexed by vt:
    uv_mesh.vertex_ids[vt] = SMPL-X vertex, faces = ft   (written by write_uv_mesh, see CONTRACTS.md).

Everything here is model-derived (SMPL-X, non-commercial): caches live in work/_avatar_cache (git-ignored).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .take import PACKAGE_ROOT

CACHE = PACKAGE_ROOT / "work" / "_avatar_cache"
HEAD_JOINTS = (15, 22, 23, 24)  # head, jaw, eyes


def model_npz(model_dir: Path) -> Path:
    return Path(model_dir) / "smplx" / "SMPLX_NEUTRAL.npz"


@dataclass
class UVAtlas:
    """Per-texel surface parametrisation of the S x S atlas (only covered texels are stored)."""
    size: int
    texel_yx: np.ndarray   # (N, 2) int32 row, col
    face: np.ndarray       # (N,) int32 SMPL-X face index
    bary: np.ndarray       # (N, 3) float32 barycentric weights over f[face] (== ft[face])
    mirror: np.ndarray     # (N,) int32 texel index of the left/right mirror point (-1 = none within 1 cm)
    region: np.ndarray     # (N,) uint8 bit 1 = head, bit 2 = face (front of the head), bit 4 = eyeball
    faces: np.ndarray      # (F, 3) SMPL-X vertex indices
    ft: np.ndarray         # (F, 3) vt indices
    vt: np.ndarray         # (VT, 2)

    @property
    def n(self) -> int:
        return len(self.face)

    def index_image(self) -> np.ndarray:
        """(S, S) int32 texel index, -1 where uncovered."""
        img = np.full((self.size, self.size), -1, np.int32)
        img[self.texel_yx[:, 0], self.texel_yx[:, 1]] = np.arange(self.n, dtype=np.int32)
        return img

    def to_image(self, values: np.ndarray, fill=0) -> np.ndarray:
        """Scatter per-texel values (N, ...) into an (S, S, ...) image."""
        values = np.asarray(values)
        img = np.full((self.size, self.size) + values.shape[1:], fill, values.dtype)
        img[self.texel_yx[:, 0], self.texel_yx[:, 1]] = values
        return img

    def surface(self, verts: np.ndarray) -> np.ndarray:
        """Texel points (..., N, 3) on a mesh with vertices (..., V, 3)."""
        tri = self.faces[self.face]  # (N, 3)
        return np.einsum("...nkd,nk->...nd", verts[..., tri, :], self.bary)


def uv_pixels(vt: np.ndarray, size: int) -> np.ndarray:
    """vt -> continuous texel coordinates (x, y) with texel centres on integers."""
    return np.stack([vt[:, 0] * size - 0.5, (1.0 - vt[:, 1]) * size - 0.5], 1)


def _bary(X: np.ndarray, ft: np.ndarray, face: np.ndarray, p: np.ndarray) -> np.ndarray:
    A, B, C = (X[ft[face, k]] for k in range(3))
    v0, v1, v2 = B - A, C - A, p - A
    d00, d01, d11 = (v0 * v0).sum(1), (v0 * v1).sum(1), (v1 * v1).sum(1)
    d20, d21 = (v2 * v0).sum(1), (v2 * v1).sum(1)
    den = d00 * d11 - d01 * d01
    den = np.where(np.abs(den) < 1e-12, 1e-12, den)
    b1 = (d11 * d20 - d01 * d21) / den
    b2 = (d00 * d21 - d01 * d20) / den
    return np.stack([1 - b1 - b2, b1, b2], 1)


def rasterize_uv(vt: np.ndarray, ft: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Face id + barycentrics of every texel centre covered by a UV triangle -> (yx (N,2), face (N,), bary (N,3))."""
    X = uv_pixels(vt, size)
    fid = np.zeros((size, size), np.int32)
    for i, tri in enumerate(ft):
        cv2.fillPoly(fid, [np.round(X[tri] * 16).astype(np.int32)], i + 1, lineType=cv2.LINE_8, shift=4)
    ys, xs = np.nonzero(fid)
    face = fid[ys, xs] - 1
    p = np.stack([xs, ys], 1).astype(np.float64)
    bary = _bary(X, ft, face, p)
    # fillPoly also marks boundary pixels outside thin triangles: give such texels the neighbouring triangle
    # (3x3 neighbourhood) that contains them best
    bad = np.flatnonzero(bary.min(1) < -1e-6)
    if len(bad):
        pad = np.pad(fid, 1)
        best_f, best_m = face[bad].copy(), bary[bad].min(1)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                cand = pad[ys[bad] + 1 + dy, xs[bad] + 1 + dx] - 1
                ok = cand >= 0
                b = _bary(X, ft, np.where(ok, cand, 0), p[bad])
                mval = np.where(ok, b.min(1), -np.inf)
                better = mval > best_m
                best_f[better], best_m[better] = cand[better], mval[better]
        face[bad] = best_f
        bary[bad] = _bary(X, ft, best_f, p[bad])
    # island-border texels outside every triangle: the closest point of their triangle (on an edge)
    out = np.flatnonzero(bary.min(1) < 0)
    if len(out):
        tri = X[ft[face[out]]]  # (n, 3, 2)
        q = p[out]
        best_d, best_b = np.full(len(out), np.inf), np.zeros((len(out), 3))
        for a, b in ((0, 1), (1, 2), (2, 0)):
            e = tri[:, b] - tri[:, a]
            s = np.clip(((q - tri[:, a]) * e).sum(1) / np.maximum((e * e).sum(1), 1e-12), 0, 1)
            d = np.linalg.norm(tri[:, a] + s[:, None] * e - q, axis=1)
            bb = np.zeros((len(out), 3))
            bb[:, a], bb[:, b] = 1 - s, s
            better = d < best_d
            best_d[better], best_b[better] = d[better], bb[better]
        bary[out] = best_b
    bary = np.clip(bary, 0, None)
    bary /= np.maximum(bary.sum(1, keepdims=True), 1e-9)
    return np.stack([ys, xs], 1).astype(np.int32), face.astype(np.int32), bary.astype(np.float32)


def vertex_regions(m) -> dict:
    """Per-vertex head / face / eyeball flags from the template mesh and LBS weights."""
    w = np.asarray(m["weights"], float)
    vtemp = np.asarray(m["v_template"], float)
    J = np.asarray(m["J_regressor"], float) @ vtemp
    head_v = w[:, list(HEAD_JOINTS)].sum(1) > 0.5
    # front of the head = the face: in rest pose SMPL-X looks along +z
    face_v = head_v & (vtemp[:, 2] > J[15, 2] + 0.02) & (vtemp[:, 1] < J[15, 1] + 0.10)
    return {"head": head_v, "face": face_v, "eye": w[:, [23, 24]].sum(1) > 0.5}


def _regions(m: dict, faces: np.ndarray, face: np.ndarray, bary: np.ndarray) -> np.ndarray:
    """Head / face / eyeball flags per texel (bits 1 / 2 / 4)."""
    reg = vertex_regions(m)
    tri = faces[face]
    agg = lambda flag: (flag[tri].astype(np.float32) * bary).sum(1) > 0.5
    return (agg(reg["head"]) * 1 + agg(reg["face"]) * 2 + agg(reg["eye"]) * 4).astype(np.uint8)


def build_atlas(model_dir: Path, size: int = 2048, cache: bool = True) -> UVAtlas:
    """UV atlas for the model's vt/ft at `size`, with a left/right mirror map (cached in work/_avatar_cache)."""
    path = model_npz(model_dir)
    m = np.load(path, allow_pickle=True)
    vt, ft, faces = np.asarray(m["vt"], float), np.asarray(m["ft"], np.int64), np.asarray(m["f"], np.int64)
    key = CACHE / f"uv_atlas_{size}_{path.parent.parent.name}.npz"
    if cache and key.exists() and key.stat().st_mtime > path.stat().st_mtime:
        z = np.load(key)
        return UVAtlas(size, z["texel_yx"], z["face"], z["bary"], z["mirror"], z["region"], faces, ft, vt)
    yx, face, bary = rasterize_uv(vt, ft, size)
    vtemp = np.asarray(m["v_template"], float)
    P0 = np.einsum("nkd,nk->nd", vtemp[faces[face]], bary)
    from scipy.spatial import cKDTree
    d, j = cKDTree(P0).query(P0 * np.array([-1.0, 1.0, 1.0]), workers=-1)
    mirror = np.where(d < 0.01, j, -1).astype(np.int32)
    region = _regions(m, faces, face, bary)
    if cache:
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez(key, texel_yx=yx, face=face, bary=bary, mirror=mirror, region=region)
    return UVAtlas(size, yx, face, bary, mirror, region, faces, ft, vt)


def write_uv_mesh(path: Path, model_dir: Path) -> dict:
    """Seam-split mesh description: Unity vertex k = SMPL-X vertex vertex_ids[k] with uv[k]; faces index k.
    Right-handed winding of the world frame (the Unity exporter reverses it, as for the skin)."""
    m = np.load(model_npz(model_dir), allow_pickle=True)
    f, ft, vt = np.asarray(m["f"], np.int64), np.asarray(m["ft"], np.int64), np.asarray(m["vt"], float)
    vid = np.full(len(vt), -1, np.int64)
    vid[ft.ravel()] = f.ravel()
    if (vid < 0).any():
        raise ValueError("vt entries without a face")
    clash = np.zeros(len(vt), bool)
    clash[ft.ravel()] = vid[ft.ravel()] != f.ravel()
    if clash.any():
        raise ValueError("a vt index maps to two different vertices; the split mesh needs (v, vt) pairs")
    np.savez(path, vertex_ids=vid.astype(np.int32), uv=vt.astype(np.float32), faces=ft.astype(np.int32),
             uv_convention=np.array("u right, v up; PNG row 0 = v 1 (Unity default import)"))
    return {"vertices": int(len(vt)), "faces": int(len(ft)), "smplx_vertices": int(len(m["v_template"]))}


class Body:
    """Posed SMPL-X surface of one dancer from smplx/<role>.npz (MAMMA locked-head model, contract flags)."""

    def __init__(self, d: dict, model_dir: Path, device: str = "cpu"):
        import smplx
        import torch
        self.t = np.asarray(d["t"], float)
        T = len(self.t)
        nb = int(d.get("num_betas", len(d["betas"])))
        flat = bool(d.get("flat_hand_mean", True))
        m = smplx.create(str(model_dir), model_type="smplx", gender="neutral", ext="npz", use_pca=False,
                         flat_hand_mean=flat, num_betas=nb, batch_size=T)
        f = lambda a: torch.tensor(np.asarray(a, np.float64), dtype=torch.float32)
        z3 = np.zeros((T, 3))
        with torch.no_grad():
            o = m(betas=f(np.asarray(d["betas"]).reshape(-1)[:nb])[None].expand(T, -1),
                  global_orient=f(d["global_orient"]), body_pose=f(d["body_pose"]),
                  jaw_pose=f(d.get("jaw_pose", z3)), leye_pose=f(d.get("leye_pose", z3)),
                  reye_pose=f(d.get("reye_pose", z3)), left_hand_pose=f(d["left_hand_pose"]),
                  right_hand_pose=f(d["right_hand_pose"]), transl=f(d["transl"]))
        self.verts = o.vertices.numpy().astype(np.float32)  # (T, V, 3) world (pose correctives included)
        self.joints = o.joints.numpy()[:, :55].astype(np.float64)
        self.fk_err_m = float(np.abs(self.joints - np.asarray(d["joints"], float)[:, :55]).max()) \
            if "joints" in d else float("nan")
        q = np.asarray(d.get("quality", np.ones(T)), float)
        rep = np.asarray(d.get("repaired", np.zeros(T, bool)), bool)
        self.quality = np.where(rep, 0.0, q)

    def at(self, t: float) -> tuple[np.ndarray, np.ndarray, float] | None:
        """Vertices and vertex velocity (m/s) at reference time t (linear between grid frames), + fit quality."""
        if t < self.t[0] - 1e-3 or t > self.t[-1] + 1e-3:
            return None
        k = int(np.clip(np.searchsorted(self.t, t) - 1, 0, len(self.t) - 2))
        a = float(np.clip((t - self.t[k]) / (self.t[k + 1] - self.t[k]), 0, 1))
        v = (1 - a) * self.verts[k] + a * self.verts[k + 1]
        vel = (self.verts[k + 1] - self.verts[k]) / (self.t[k + 1] - self.t[k])
        q = min(self.quality[k], self.quality[k + 1]) if 0 < a < 1 else self.quality[k if a < 0.5 else k + 1]
        return v, vel, float(q)


def vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v = verts[faces]
    n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    out = np.zeros_like(verts)
    for k in range(3):
        np.add.at(out, faces[:, k], n)
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
