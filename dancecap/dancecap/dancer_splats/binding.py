"""Binding math of the avatar-guided splats (pure torch: runs on the CPU in the tests, the 2080 Ti and a pod).

Body Gaussian i is bound to triangle f = face[i] of its dancer's posed SMPL-X mesh:
    frame   R_f(t) = [t1 | t2 | n] (t1 along edge v0->v1, n the unit normal, t2 = n x t1), k_f(t) = sqrt(|e1 x e2|)
    mean    mu_i(t) = sum_k b_ik v_{f,k}(t) + R_f(t) (k_f(t) / k_f(rest)) off_i + R_f(t) delta_i(t)
    rot     R_i(t)  = R_f(t) R(q_i)
    scale   s_i(t)  = exp(log_s_i) k_f(t) / k_f(rest)
(GaussianAvatars / SplattingAvatar binding; delta_i(t) = the per-Gaussian cosine residual, model.py).

Pose corrections without the SMPL-X model: the prep stores the fit's posed vertices, the GLOBAL joint rotations Rg_j(t)
and the posed joint positions c_j(t). A small local rotation r_j of body joint j (applied on the child side of the
joint) moves every point attached to joint j's frame by D_j = G'_j G_j^-1, which follows the chain
    D_j = D_parent(j) E_j,   E_j = rotation by Rg_j Rot(r_j) Rg_j^T about c_j,   D_root = T(dT) E_root
Joints without a correction (jaw, eyes, fingers) take their body ancestor's D. Vertices are re-skinned with the fit's
LBS weights: v' = sum_m w_m D_{j_m} v (exact up to second order in the small correction inside blend regions; the
pose correctives of the tiny change are ignored).
"""
from __future__ import annotations

import math

import numpy as np

SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396)


def _torch():
    import torch
    return torch


# ------------------------------------------------------------------------------------------------ rotations

def axis_angle_to_matrix(r):
    """(..., 3) axis-angle -> (..., 3, 3); smooth at 0 (Taylor terms)."""
    torch = _torch()
    th2 = (r * r).sum(-1, keepdim=True)
    th = torch.sqrt(th2 + 1e-12)
    small = th2 < 1e-8
    a = torch.where(small, 1 - th2 / 6, torch.sin(th) / th)
    b = torch.where(small, 0.5 - th2 / 24, (1 - torch.cos(th)) / (th2 + 1e-12))
    x, y, z = r.unbind(-1)
    O = torch.zeros_like(x)
    Kx = torch.stack([O, -z, y, z, O, -x, -y, x, O], -1).reshape(r.shape[:-1] + (3, 3))
    I = torch.eye(3, dtype=r.dtype, device=r.device).expand(Kx.shape)
    return I + a[..., None] * Kx + b[..., None] * (Kx @ Kx)


def quat_to_matrix(q):
    """(..., 4) wxyz (normalised here) -> (..., 3, 3)."""
    torch = _torch()
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(
        q.shape[:-1] + (3, 3))


def matrix_to_quat(R):
    """(..., 3, 3) rotation -> (..., 4) wxyz with w >= 0 (branch on the largest diagonal term; differentiable)."""
    torch = _torch()
    m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    t0 = 1 + m00 + m11 + m22
    t1 = 1 + m00 - m11 - m22
    t2 = 1 - m00 + m11 - m22
    t3 = 1 - m00 - m11 + m22
    s = lambda t: torch.sqrt(t.clamp_min(1e-10)) * 2  # noqa: E731
    q0 = torch.stack([s(t0) / 4, (m21 - m12) / s(t0), (m02 - m20) / s(t0), (m10 - m01) / s(t0)], -1)
    q1 = torch.stack([(m21 - m12) / s(t1), s(t1) / 4, (m01 + m10) / s(t1), (m02 + m20) / s(t1)], -1)
    q2 = torch.stack([(m02 - m20) / s(t2), (m01 + m10) / s(t2), s(t2) / 4, (m12 + m21) / s(t2)], -1)
    q3 = torch.stack([(m10 - m01) / s(t3), (m02 + m20) / s(t3), (m12 + m21) / s(t3), s(t3) / 4], -1)
    best = torch.stack([t0, t1, t2, t3], -1).argmax(-1)
    q = torch.stack([q0, q1, q2, q3], -2).gather(-2, best[..., None, None].expand(best.shape + (1, 4)))[..., 0, :]
    q = q * torch.where(q[..., :1] < 0, -1.0, 1.0)
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)


# ------------------------------------------------------------------------------------------------ triangles

def triangle_frames(V, faces):
    """V (..., Nv, 3), faces (F, 3) long -> R (..., F, 3, 3) with columns [t1, t2, n], k (..., F) = sqrt(|e1 x e2|)."""
    torch = _torch()
    v0, v1, v2 = V[..., faces[:, 0], :], V[..., faces[:, 1], :], V[..., faces[:, 2], :]
    e1, e2 = v1 - v0, v2 - v0
    c = torch.cross(e1, e2, dim=-1)
    area2 = c.norm(dim=-1)
    n = c / area2[..., None].clamp_min(1e-12)
    t1 = e1 / e1.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    t2 = torch.cross(n, t1, dim=-1)
    return torch.stack([t1, t2, n], -1), torch.sqrt(area2.clamp_min(1e-14))


def barycentric_points(V, faces, face_idx, bary):
    """Surface points sum_k b_k v_{f,k}: V (..., Nv, 3), face_idx (N,), bary (N, 3) -> (..., N, 3)."""
    tri = faces[face_idx]
    return (V[..., tri, :] * bary[..., None]).sum(-2)


def body_gaussians(V, faces, face_idx, bary, off, k_rest, log_scale, quat, resid=None, scale_cap: float | None = None,
                   frames=None, anchor=None):
    """World means / rotation matrices / scales of body Gaussians on the posed mesh V (..., Nv, 3).
    off (N, 3) local offset in triangle units (x k_f(t) / k_f(rest)), resid (..., N, 3) metres in the triangle frame."""
    torch = _torch()
    Rf, kf = frames if frames is not None else triangle_frames(V, faces)
    R = Rf[..., face_idx, :, :]
    ratio = (kf[..., face_idx] / k_rest[face_idx].clamp_min(1e-9))[..., None]
    loc = off * ratio
    if resid is not None:
        loc = loc + resid
    base = barycentric_points(V, faces, face_idx, bary) if anchor is None else anchor
    mu = base + (R @ loc[..., None])[..., 0]
    ls = log_scale if scale_cap is None else log_scale.clamp(max=math.log(scale_cap))
    scales = torch.exp(ls) * ratio
    Rw = R @ quat_to_matrix(quat)
    return mu, Rw, scales


def triangle_area_samples(V: np.ndarray, faces: np.ndarray, n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """n area-uniform surface samples (face index, barycentrics) of a mesh (numpy, the prep)."""
    rng = np.random.default_rng(seed)
    v = V[faces]
    area = 0.5 * np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1)
    f = rng.choice(len(faces), size=n, p=area / area.sum())
    u, w = rng.random(n), rng.random(n)
    flip = u + w > 1
    u[flip], w[flip] = 1 - u[flip], 1 - w[flip]
    return f.astype(np.int64), np.stack([1 - u - w, u, w], 1).astype(np.float32)


# ------------------------------------------------------------------------------------------------ pose deltas

def body_ancestor(parents: np.ndarray, n_body: int) -> np.ndarray:
    """(J,) index of each joint's nearest ancestor (or itself) among the first n_body joints."""
    out = np.zeros(len(parents), np.int64)
    for j in range(len(parents)):
        a = j
        while a >= n_body:
            a = int(parents[a])
        out[j] = a
    return out


def chain_levels(parents: np.ndarray, n_body: int) -> list[list[int]]:
    """Body joints grouped by depth in the kinematic tree (root first)."""
    depth = {0: 0}
    for j in range(1, n_body):
        p = int(parents[j])
        depth[j] = depth[p] + 1
    lv: dict[int, list[int]] = {}
    for j, d in depth.items():
        lv.setdefault(d, []).append(j)
    return [sorted(lv[d]) for d in sorted(lv)]


def pose_delta_transforms(Rg, Jp, r, dT, parents: np.ndarray, levels: list[list[int]]):
    """Rg (..., B, 3, 3) global rotations and Jp (..., B, 3) posed positions of the B body joints, r (..., B, 3) local
    corrections (axis-angle), dT (..., 3) root translation -> D (..., B, 4, 4) with D_j = G'_j G_j^-1."""
    torch = _torch()
    Q = Rg @ axis_angle_to_matrix(r) @ Rg.transpose(-1, -2)
    I = torch.eye(3, dtype=Rg.dtype, device=Rg.device)
    bottom = torch.zeros(Rg.shape[:-2] + (1, 4), dtype=Rg.dtype, device=Rg.device)
    bottom[..., 0, 3] = 1.0
    E = torch.cat([torch.cat([Q, ((I - Q) @ Jp[..., None])], -1), bottom], -2)  # (..., B, 4, 4)
    T = torch.cat([torch.cat([I.expand(dT.shape[:-1] + (3, 3)), dT[..., None]], -1), bottom[..., 0, :, :]], -2)
    # level by level (one batched matmul per depth of the tree, not one per joint); index tensors cached per device
    # (a list -> cuda tensor copy synchronises the stream)
    key = (str(Rg.device), Rg.shape[-3], tuple(int(p) for p in parents[:Rg.shape[-3]]))
    idx = _CHAIN_IDX.get(key)
    if idx is None:
        order = [j for lv in levels for j in lv]
        pos = {j: i for i, j in enumerate(order)}
        idx = _CHAIN_IDX[key] = (
            [(torch.as_tensor([pos[int(parents[j])] for j in lv], device=Rg.device),
              torch.as_tensor(lv, device=Rg.device)) for lv in levels[1:]],
            torch.as_tensor([pos[j] for j in range(Rg.shape[-3])], device=Rg.device))
    Dall = (T @ E[..., 0, :, :])[..., None, :, :]
    for pidx, jidx in idx[0]:
        Dall = torch.cat([Dall, Dall.index_select(-3, pidx) @ E.index_select(-3, jidx)], -3)
    return Dall.index_select(-3, idx[1])


_CHAIN_IDX: dict = {}


def dense_weights(w_idx, w_val, n_joints: int):
    """Top-k LBS weights (Nv, k) -> dense (Nv, J) (a matmul skins faster than a gather: its backward needs no atomics
    into the J transforms)."""
    torch = _torch()
    W = torch.zeros(w_idx.shape[0], n_joints, dtype=w_val.dtype, device=w_val.device)
    return W.scatter_(1, w_idx, w_val)


def skin_delta(V, D, W):
    """Re-skin posed vertices V (..., Nv, 3) by per-joint transforms D (..., J, 4, 4) with dense LBS weights W (Nv, J):
    v' = sum_j W_vj D_j v."""
    lead = D.shape[:-3]
    M = (W @ D[..., :3, :].reshape(lead + (D.shape[-3], 12))).reshape(lead + (W.shape[0], 3, 4))
    return (M[..., :3] @ V[..., None])[..., 0] + M[..., 3]


def apply_T(D, X):
    """4x4 transforms (..., 4, 4) applied to points (..., N, 3) (D broadcast over N)."""
    return (D[..., None, :3, :3] @ X[..., None])[..., 0] + D[..., None, :3, 3]


# ------------------------------------------------------------------------------------------------ residual / SH

def dct_basis(x, K: int):
    """Cosine basis phi_k(x) = cos(pi k x), k = 1..K, of the normalised window time x in [0, 1] -> (..., K)."""
    torch = _torch()
    k = torch.arange(1, K + 1, dtype=x.dtype, device=x.device)
    return torch.cos(math.pi * k * x[..., None])


def sh_rgb(dirs, coeffs, degree: int):
    """3DGS spherical harmonics (degree <= 2): dirs (N, 3) (any length), coeffs (N, K, 3) -> rgb (N, 3), +0.5, >= 0."""
    torch = _torch()
    degree = min(int(degree), int(round(math.sqrt(coeffs.shape[1]))) - 1)
    if dirs.is_cuda:  # gsplat's fused kernel (same convention; it normalises dirs itself)
        try:
            from gsplat.cuda._wrapper import spherical_harmonics
            c = coeffs.expand(dirs.shape[:-1] + coeffs.shape[-2:])
            return torch.clamp_min(spherical_harmonics(degree, dirs, c) + 0.5, 0.0)
        except ImportError:
            pass
    d = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    c = coeffs.expand(dirs.shape[:-1] + coeffs.shape[-2:])
    out = SH_C0 * c[..., 0, :]
    if degree >= 1:
        x, y, z = d[..., 0:1], d[..., 1:2], d[..., 2:3]
        out = out - SH_C1 * y * c[..., 1, :] + SH_C1 * z * c[..., 2, :] - SH_C1 * x * c[..., 3, :]
        if degree >= 2:
            xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
            out = (out + SH_C2[0] * xy * c[..., 4, :] + SH_C2[1] * yz * c[..., 5, :]
                   + SH_C2[2] * (2 * zz - xx - yy) * c[..., 6, :] + SH_C2[3] * xz * c[..., 7, :]
                   + SH_C2[4] * (xx - yy) * c[..., 8, :])
    return torch.clamp_min(out + 0.5, 0.0)


def rgb_to_sh0(rgb):
    return (rgb - 0.5) / SH_C0


# ------------------------------------------------------------------------------------------------ strands

def segment_frames(X, ref):
    """Strand nodes X (..., S+1, 3) -> segment frames (..., S, 3, 3) columns [d, b1, b2] and lengths (..., S).
    ref (..., 3): a direction not along the strand (the head's forward axis; x when nearly parallel), broadcast
    against X's leading dims (e.g. (B, 1, 3) for nodes (B, G, S+1, 3))."""
    torch = _torch()
    e = X[..., 1:, :] - X[..., :-1, :]
    L = e.norm(dim=-1)
    d = e / L[..., None].clamp_min(1e-9)
    r = ref[..., None, :].expand(d.shape)
    b1 = torch.cross(d, r, dim=-1)
    bad = b1.norm(dim=-1, keepdim=True) < 1e-3
    alt = torch.zeros_like(r)
    alt[..., 0] = 1.0
    b1 = torch.where(bad, torch.cross(d, alt, dim=-1), b1)
    b1 = b1 / b1.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    b2 = torch.cross(d, b1, dim=-1)
    return torch.stack([d, b1, b2], -1), L


def capsule_distance(P, a, b):
    """Distance of points P (..., N, 3) to the segments a-b (..., 3)."""
    ab = (b - a)[..., None, :]
    t = (((P - a[..., None, :]) * ab).sum(-1) / (ab * ab).sum(-1).clamp_min(1e-12)).clamp(0, 1)
    return (P - (a[..., None, :] + t[..., None] * ab)).norm(dim=-1)


def resample_polyline(P: np.ndarray, n: int) -> np.ndarray:
    """(M, 3) polyline -> n points equally spaced by arc length (numpy)."""
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    q = np.linspace(0, s[-1], n)
    return np.stack([np.interp(q, s, P[:, k]) for k in range(3)], 1)


def rest_strand(root: np.ndarray, normal: np.ndarray, centre: np.ndarray, radius: float, length: float,
                nodes: int, standoff: float = 0.01, side_az_deg: float = 100.0) -> np.ndarray:
    """Rest polyline of one guide in the head's rest frame (Y up, +Z facing, +X her left): from the root lifted by the
    stand-off, over the skull along a great circle to the equator (front strands are swept to the side, behind the ear,
    so no strand starts over the face), then straight down; resampled to `nodes` points of total `length`."""
    p0 = root + normal * standoff
    d0 = p0 - centre
    R = max(float(np.linalg.norm(d0)), radius + standoff)
    u0 = d0 / max(np.linalg.norm(d0), 1e-9)
    az = math.degrees(math.atan2(u0[0], u0[2]))
    if abs(az) < side_az_deg:
        az = math.copysign(side_az_deg, az if abs(az) > 1e-3 else 1.0)
    u1 = np.array([math.sin(math.radians(az)), 0.0, math.cos(math.radians(az))])
    if u0[1] < 0:  # root below the equator (nape): straight down from it
        pts = [p0]
    else:
        ang = math.acos(float(np.clip(u0 @ u1, -1, 1)))
        m = max(2, int(ang / 0.05))
        pts = []
        for s in np.linspace(0, 1, m):
            if ang < 1e-6:
                u = u0
            else:  # slerp
                u = (math.sin((1 - s) * ang) * u0 + math.sin(s * ang) * u1) / math.sin(ang)
            pts.append(centre + R * u)
    pts = np.array(pts)
    arc = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()) if len(pts) > 1 else 0.0
    if arc >= length:  # short strand: the arc only, cut at `length`
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        k = int(np.searchsorted(s, length))
        pts = np.vstack([pts[:k], pts[k - 1] + (pts[k] - pts[k - 1]) * (length - s[k - 1]) / max(seg[k - 1], 1e-9)])
    else:
        pts = np.vstack([pts, pts[-1] + np.array([0.0, -(length - arc), 0.0])])
    return resample_polyline(pts, nodes)
