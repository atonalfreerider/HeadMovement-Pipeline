"""Kinematics of a contract fit for the "sides" step (numpy and torch, the same formulas):

* FK of the 55 SMPL-X joints = physics_body.forward_kinematics (it reproduces a fit's own `joints` to 1e-15 m on the recap);
* MAMMA's 512 points by plain linear-blend skinning pre-blended per joint (pose_refine.Problem: D @ W and D @ (W x v_rest)), so a point is
  sum_j R_j A[k, j] + Bw[k, j] p_j - no pose correctives (as physics_body and the Unity skin).

Pose layout: (T, 55, 3) axis-angle (`pose55`), SMPL-X order: 0 global orient, 1..21 body, 22 jaw, 23 / 24 eyes, 25..39 left hand, 40..54
right hand. No take constants.
"""
from __future__ import annotations

import numpy as np

POSE_KEYS = ("global_orient", "body_pose", "jaw_pose", "leye_pose", "reye_pose", "left_hand_pose", "right_hand_pose")
POSE_SLICES = {"global_orient": (0, 1), "body_pose": (1, 22), "jaw_pose": (22, 23), "leye_pose": (23, 24), "reye_pose": (24, 25),
               "left_hand_pose": (25, 40), "right_hand_pose": (40, 55)}
N_J = 55

_MODEL: dict = {}


def betas_key(betas) -> tuple:
    return tuple(np.round(np.asarray(betas, float).ravel(), 8))


def model(fit: dict) -> dict:
    """{"parents" (55,), "j_rest" (55,3), "A" (512,55,3), "Bw" (512,55), "parts" {part: idx}} of a fit's shaped body (cached per betas)"""
    key = betas_key(fit["betas"])
    if key not in _MODEL:
        from .. import physics_body as pb
        from ..mamma_dyn import _load_verts512, part_index
        body = pb.load_body(fit)
        D = _load_verts512()
        V = D.shape[1]
        W = body.weights[:V]
        A = np.einsum("kv,vj,vd->kjd", D, W, body.v_rest[:V])
        _MODEL[key] = {"parents": np.asarray(body.parents, int), "j_rest": np.asarray(body.j_rest, float), "A": A, "Bw": D @ W,
                       "parts": part_index()}
    return _MODEL[key]


def pose55(fit: dict, ks=None) -> np.ndarray:
    """(n, 55, 3) axis-angle of frames ks (all frames when None)"""
    T = len(fit["t"])
    ks = np.arange(T) if ks is None else np.asarray(ks)
    parts = []
    for k in POSE_KEYS:
        a, b = POSE_SLICES[k]
        if k in fit:
            parts.append(np.asarray(fit[k], float)[ks].reshape(len(ks), b - a, 3))
        else:
            parts.append(np.zeros((len(ks), b - a, 3)))
    return np.concatenate(parts, 1)


def write_pose55(fit: dict, P: np.ndarray, ks) -> None:
    """write (n,55,3) axis-angle into a fit's pose keys at frames ks (in place; the arrays must be writable copies)"""
    ks = np.asarray(ks)
    for k in POSE_KEYS:
        if k not in fit:
            continue
        a, b = POSE_SLICES[k]
        arr = fit[k]
        arr[ks] = P[:, a:b].reshape((len(ks),) + arr.shape[1:]).astype(arr.dtype)


def rotmat(rv: np.ndarray) -> np.ndarray:
    """axis-angle (..., 3) -> (..., 3, 3)"""
    th = np.linalg.norm(rv, axis=-1, keepdims=True)
    k = rv / np.maximum(th, 1e-12)
    K = np.zeros(rv.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 0] = -k[..., 2], k[..., 1], k[..., 2]
    K[..., 1, 2], K[..., 2, 0], K[..., 2, 1] = -k[..., 0], -k[..., 1], k[..., 0]
    s, c = np.sin(th)[..., None], np.cos(th)[..., None]
    return np.eye(3) + s * K + (1 - c) * (K @ K)


def rotvec(R: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation as Rot
    R = np.asarray(R, float)
    return Rot.from_matrix(R.reshape(-1, 3, 3)).as_rotvec().reshape(R.shape[:-2] + (3,))


def fk(m: dict, Rl: np.ndarray, transl: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """local rotations (T,55,3,3), root translation (T,3) -> world rotations R (T,55,3,3), rigid offsets p (T,55,3), joints J (T,55,3)"""
    T = len(Rl)
    J0 = m["j_rest"]
    par = m["parents"]
    R = np.empty((T, N_J, 3, 3))
    J = np.empty((T, N_J, 3))
    for j in range(N_J):
        q = par[j]
        if q < 0:
            R[:, j] = Rl[:, j]
            J[:, j] = J0[j] + transl
        else:
            R[:, j] = R[:, q] @ Rl[:, j]
            J[:, j] = J[:, q] + np.einsum("tab,b->ta", R[:, q], J0[j] - J0[q])
    p = J - np.einsum("tjab,jb->tja", R, J0)
    return R, p, J


def points512(m: dict, R: np.ndarray, p: np.ndarray) -> np.ndarray:
    """MAMMA's 512 points (T,512,3) of a posed body (plain LBS, pre-blended)"""
    return np.einsum("tjab,kjb->tka", R, m["A"]) + np.einsum("kj,tjd->tkd", m["Bw"], p)


def forward(fit: dict, ks=None, pose: np.ndarray | None = None, transl: np.ndarray | None = None, points: bool = True) -> dict:
    """{"J" (n,55,3), "R" (n,55,3,3), "pts" (n,512,3)} of a fit at frames ks (optionally another pose / translation for those frames)"""
    m = model(fit)
    T = len(fit["t"])
    ks = np.arange(T) if ks is None else np.asarray(ks)
    P = pose55(fit, ks) if pose is None else np.asarray(pose, float)
    tr = np.asarray(fit["transl"], float)[ks] if transl is None else np.asarray(transl, float)
    R, p, J = fk(m, rotmat(P), tr)
    out = {"J": J, "R": R}
    if points:
        out["pts"] = points512(m, R, p)
    return out


# ------------------------------------------------------------------------------------------------ torch

class TorchBody:
    """the same kinematics in torch (one dancer, any device / dtype): fk(Rl (W,55,3,3), transl (W,3)) -> R, p, J; points(R, p)"""

    def __init__(self, fit: dict, device="cpu", dtype=None):
        import torch
        self.torch = torch
        dtype = dtype or torch.float32
        m = model(fit)
        f = dict(dtype=dtype, device=device)
        self.parents = [int(x) for x in m["parents"]]
        self.J0 = torch.tensor(m["j_rest"], **f)
        self.A = torch.tensor(m["A"], **f)
        self.Bw = torch.tensor(m["Bw"], **f)
        self.rel = torch.stack([self.J0[j] - (self.J0[q] if q >= 0 else 0 * self.J0[j]) for j, q in enumerate(self.parents)])
        # the kinematic tree by depth: each level's joints are posed in one batched product
        depth = [0] * N_J
        for j, q in enumerate(self.parents):
            depth[j] = 0 if q < 0 else depth[q] + 1
        self.levels = []
        for lv in range(1, max(depth) + 1):
            js = [j for j in range(N_J) if depth[j] == lv]
            self.levels.append((js, [self.parents[j] for j in js], torch.as_tensor(js, device=device)))

    def fk(self, Rl, transl):
        torch = self.torch
        Rs, Js = [None] * N_J, [None] * N_J
        Rs[0] = Rl[:, 0]
        Js[0] = self.J0[0] + transl
        for js, ps, jt in self.levels:
            Rp = torch.stack([Rs[q] for q in ps], 1)
            Jp = torch.stack([Js[q] for q in ps], 1)
            Rn = Rp @ Rl[:, jt]
            Jn = Jp + (Rp @ self.rel[jt][None, :, :, None])[..., 0]
            for i, j in enumerate(js):
                Rs[j] = Rn[:, i]
                Js[j] = Jn[:, i]
        R = torch.stack(Rs, 1)
        J = torch.stack(Js, 1)
        p = J - (R @ self.J0[None, :, :, None])[..., 0]
        return R, p, J

    def points(self, R, p, idx=None):
        """MAMMA's 512 points (or the subset idx) of a posed body"""
        torch = self.torch
        A, Bw = (self.A, self.Bw) if idx is None else (self.A[idx], self.Bw[idx])
        return torch.einsum("tjab,kjb->tka", R, A) + torch.einsum("kj,tjd->tkd", Bw, p)


def rodrigues_t(torch, rv):
    """axis-angle (..., 3) -> (..., 3, 3), smooth at 0: I + sinc(th) [rv]x + (1 - cos th) / th^2 [rv]x^2"""
    th = torch.sqrt((rv * rv).sum(-1) + 1e-12)[..., None, None]
    z = torch.zeros_like(rv[..., 0])
    K = torch.stack([torch.stack([z, -rv[..., 2], rv[..., 1]], -1), torch.stack([rv[..., 2], z, -rv[..., 0]], -1),
                     torch.stack([-rv[..., 1], rv[..., 0], z], -1)], -2)
    a = torch.sinc(th / np.pi)
    b = 0.5 * torch.sinc(th / (2 * np.pi)) ** 2
    eye = torch.eye(3, dtype=rv.dtype, device=rv.device)
    return eye + a * K + b * (K @ K)
