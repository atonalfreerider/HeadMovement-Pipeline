"""Light SMPL-X kinematics for the hand stages (refine_hands): joint FK, fingertips and LBS on vertex subsets.

smplx.create() poses all 10475 vertices with pose correctives on every call (~0.3 s per 176-frame batch on CPU, 10x
that with gradients), which makes per-camera leave-one-out IK of the fingers impractically slow. The hand stages only
need joint positions, five fingertips per hand and the surface of the hands/forearms, so this module re-implements
SMPL-X forward kinematics in torch from the same model file:

    J      = J_regressor @ (v_template + shapedirs . betas)          (shaped rest joints; exactly smplx's joints)
    G_j    = G_parent(j) [R(theta_j) | J_j - J_parent]                (global rigid transforms, 55 joints)
    v      = sum_j w_vj G_j (v_rest + posedirs . pose_feature - J_j)  (LBS incl. pose correctives, subset only)

Fingertips are smplx's extra joints (vertex_ids['smplx'] tips) so the 21-point hand order matches COCO-WholeBody /
OpenPose: wrist, thumb CMC MCP IP tip, index MCP PIP DIP tip, middle .., ring .., pinky .. (HAND21_L / HAND21_R).
Checked in tests against smplx (joints < 1e-5 m, tips < 1e-4 m).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# SMPL-X joints: 20/21 wrists; left fingers 25..39 = index1-3, middle1-3, pinky1-3, ring1-3, thumb1-3; right 40..54.
# Extra joints (smplx vertex_joint_selector): 66..70 left tips thumb,index,middle,ring,pinky; 71..75 right tips.
TIP_NAMES = ("thumb", "index", "middle", "ring", "pinky")
HAND21_L = (20, 37, 38, 39, 66, 25, 26, 27, 67, 28, 29, 30, 68, 34, 35, 36, 69, 31, 32, 33, 70)
HAND21_R = (21, 52, 53, 54, 71, 40, 41, 42, 72, 43, 44, 45, 73, 49, 50, 51, 74, 46, 47, 48, 75)
HAND21 = {"left": HAND21_L, "right": HAND21_R}
SIDES = ("left", "right")
WRIST = {"left": 20, "right": 21}
ELBOW = {"left": 18, "right": 19}
SHOULDER = {"left": 16, "right": 17}
COLLAR = {"left": 13, "right": 14}
FINGER_JOINTS = {"left": tuple(range(25, 40)), "right": tuple(range(40, 55))}
# 21-point skeleton edges (COCO-WholeBody hand order)
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
              (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]
FINGERS21 = {"thumb": (1, 2, 3, 4), "index": (5, 6, 7, 8), "middle": (9, 10, 11, 12), "ring": (13, 14, 15, 16),
             "pinky": (17, 18, 19, 20)}
TIPS21 = (4, 8, 12, 16, 20)
# body_pose (63) slot of joint j (1..21) is 3*(j-1)
def body_slot(j: int) -> slice:
    return slice(3 * (j - 1), 3 * j)


def tip_vertex_ids() -> dict:
    from smplx.vertex_ids import vertex_ids
    v = vertex_ids["smplx"]
    return {"left": [v["l" + n] for n in TIP_NAMES], "right": [v["r" + n] for n in TIP_NAMES]}


def model_npz(model_dir: Path) -> Path:
    p = Path(model_dir)
    for c in (p / "smplx" / "SMPLX_NEUTRAL.npz", p / "SMPLX_NEUTRAL.npz", p):
        if c.is_file():
            return c
    raise FileNotFoundError(f"no SMPLX_NEUTRAL.npz under {model_dir}")


def full_pose(d: dict) -> np.ndarray:
    """smplx npz (contract, hands absolute) -> (T, 55, 3) axis-angle in SMPL-X joint order."""
    T = len(d["t"])
    z3 = np.zeros((T, 3))
    return np.concatenate([d["global_orient"], d["body_pose"], d.get("jaw_pose", z3), d.get("leye_pose", z3),
                           d.get("reye_pose", z3), d["left_hand_pose"], d["right_hand_pose"]], 1).reshape(T, 55, 3)


def rodrigues(torch, rv):
    """(..., 3) axis-angle -> (..., 3, 3), differentiable at 0."""
    th2 = (rv * rv).sum(-1, keepdim=True)
    th = torch.sqrt(th2 + 1e-16)
    k = rv / th
    K = torch.zeros(rv.shape[:-1] + (3, 3), dtype=rv.dtype, device=rv.device)
    K[..., 0, 1], K[..., 0, 2] = -k[..., 2], k[..., 1]
    K[..., 1, 0], K[..., 1, 2] = k[..., 2], -k[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -k[..., 1], k[..., 0]
    s, c = torch.sin(th)[..., None], torch.cos(th)[..., None]
    eye = torch.eye(3, dtype=rv.dtype, device=rv.device).expand_as(K)
    small = (th2 < 1e-12)[..., None]
    Kr = torch.zeros_like(K)  # first-order form near zero keeps the gradient exact
    Kr[..., 0, 1], Kr[..., 0, 2] = -rv[..., 2], rv[..., 1]
    Kr[..., 1, 0], Kr[..., 1, 2] = rv[..., 2], -rv[..., 0]
    Kr[..., 2, 0], Kr[..., 2, 1] = -rv[..., 1], rv[..., 0]
    R = eye + s * K + (1 - c) * (K @ K)
    return torch.where(small, eye + Kr, R)


class SmplxKin:
    """Shaped SMPL-X kinematics for one dancer (fixed betas). All tensors float64 on `device`."""

    def __init__(self, model_dir: Path, betas: np.ndarray, device: str = "cpu", vert_subset: np.ndarray | None = None):
        import torch
        self.torch = torch
        self.device = device
        m = np.load(model_npz(model_dir), allow_pickle=True)
        nb = len(betas)
        sd = np.asarray(m["shapedirs"], float)[:, :, :nb]
        v_shaped = np.asarray(m["v_template"], float) + sd @ np.asarray(betas, float)
        Jr = np.asarray(m["J_regressor"], float)
        self.J_rest = Jr @ v_shaped                                  # (55, 3)
        self.parents = np.asarray(m["kintree_table"])[0].astype(np.int64)
        self.parents[0] = -1
        self.faces = np.asarray(m["f"], np.int64)
        self.weights_np = np.asarray(m["weights"], float)
        self.v_shaped_np = v_shaped
        self.posedirs_np = np.asarray(m["posedirs"], float)          # (V, 3, 486)
        self.tips = tip_vertex_ids()
        f64 = dict(dtype=torch.float64, device=device)
        self.J = torch.tensor(self.J_rest, **f64)
        self.rel = self.J.clone()
        self.rel[1:] = self.J[1:] - self.J[self.parents[1:]]
        tip_ids = self.tips["left"] + self.tips["right"]
        self._tip_ids = np.array(tip_ids)
        self.set_subset(vert_subset)
        self._tip = self._subset_tensors(self._tip_ids)

    # ---------------------------------------------------------------------------------------------- vertex subsets
    def _subset_tensors(self, ids: np.ndarray) -> dict:
        torch = self.torch
        f64 = dict(dtype=torch.float64, device=self.device)
        w = self.weights_np[ids]
        used = np.flatnonzero(w.max(0) > 0)
        return {"ids": ids, "v": torch.tensor(self.v_shaped_np[ids], **f64),
                "pd": torch.tensor(self.posedirs_np[ids], **f64),          # (n, 3, 486)
                "w": torch.tensor(w[:, used], **f64), "used": torch.tensor(used, device=self.device)}

    def set_subset(self, ids: np.ndarray | None):
        self.subset = None if ids is None else self._subset_tensors(np.asarray(ids, np.int64))

    # ---------------------------------------------------------------------------------------------- forward
    def forward(self, pose, transl, want_verts: bool = False):
        """pose (T, 55, 3) axis-angle, transl (T, 3) torch float64 -> dict(G (T,55,4,4) global transforms with the
        rest joint removed (LBS form), joints (T,55,3), tips (T,10,3) left thumb..pinky then right, verts (T,n,3)
        of the subset when want_verts)."""
        torch = self.torch
        T = pose.shape[0]
        R = rodrigues(torch, pose)                                     # (T,55,3,3)
        Gs = [None] * 55
        Gr = [None] * 55
        Gt = [None] * 55
        for j in range(55):
            p = int(self.parents[j])
            if p < 0:
                Gr[j], Gt[j] = R[:, 0], self.J[0].expand(T, 3)
            else:
                Gr[j] = Gr[p] @ R[:, j]
                Gt[j] = Gt[p] + (Gr[p] @ self.rel[j][:, None])[..., 0]
        Rg = torch.stack(Gr, 1)                                         # (T,55,3,3)
        Jp = torch.stack(Gt, 1)                                         # (T,55,3) posed joints (no transl)
        tl = Jp - (Rg @ self.J[None, :, :, None])[..., 0]               # LBS translation part
        out = {"R": Rg, "joints": Jp + transl[:, None], "tl": tl}
        # pose feature (rotation matrices of joints 1.. minus identity), as smplx
        eye = torch.eye(3, dtype=pose.dtype, device=pose.device)
        out["pose_feature"] = (R[:, 1:] - eye).reshape(T, -1)          # (T, 486)
        out["tips"] = self._lbs(out, self._tip, transl)
        if want_verts and self.subset is not None:
            out["verts"] = self._lbs(out, self.subset, transl)
        return out

    def _lbs(self, out, sub, transl):
        torch = self.torch
        vp = sub["v"][None] + torch.einsum("nkp,tp->tnk", sub["pd"], out["pose_feature"])   # (T,n,3)
        used = sub["used"]
        Rw = torch.einsum("nj,tjab->tnab", sub["w"], out["R"][:, used])
        tw = torch.einsum("nj,tja->tna", sub["w"], out["tl"][:, used])
        return (Rw @ vp[..., None])[..., 0] + tw + transl[:, None]

    # ---------------------------------------------------------------------------------------------- fast hand path
    def hand_base(self, out, side: str) -> dict:
        """Constants of `side`'s hand for hand_fk(), from a full forward() of the (fixed) body: the posed parent of
        the wrist (elbow) transform, the wrist rest offset, and the non-hand part of the tips' pose correctives."""
        torch = self.torch
        w = WRIST[side]
        p = int(self.parents[w])
        tips = slice(0, 5) if side == "left" else slice(5, 10)
        hj = [w, *FINGER_JOINTS[side]]
        pf = out["pose_feature"].detach().clone()                     # (T, 486) = joints 1..54
        for j in hj:
            pf[:, 9 * (j - 1):9 * j] = 0.0
        sub = {k: (v[tips] if k in ("v", "pd", "w") else v) for k, v in self._tip.items()}
        ids = self._tip_ids[tips]
        wt = torch.tensor(self.weights_np[ids][:, [p] + hj], dtype=torch.float64, device=self.device)  # + elbow
        transl = out["joints"][:, 0].detach() - out["R"][:, 0].detach() @ self.J[0] - out["tl"][:, 0].detach() \
            if False else None  # noqa (kept explicit below)
        return {"side": side, "Rp": out["R"][:, p].detach(), "pp": out["joints"][:, p].detach(),
                "pf_fixed": pf, "tip_v": sub["v"], "tip_pd": sub["pd"], "tip_w": wt, "hj": hj}

    def hand_fk(self, base: dict, wrist_rv, hand_pose):
        """Hand points for a new wrist rotation (T,3) and hand pose (T,15,3), body fixed -> (T, 21, 3) world.
        Equal to forward() + hand21() (tested)."""
        torch = self.torch
        side = base["side"]
        w = WRIST[side]
        T = hand_pose.shape[0]
        Rw_loc = rodrigues(torch, wrist_rv)                            # (T,3,3)
        Rw = base["Rp"] @ Rw_loc
        pw = base["pp"] + (base["Rp"] @ self.rel[w][:, None])[..., 0]
        Rl = rodrigues(torch, hand_pose)                               # (T,15,3,3) finger joints in kintree order
        f0 = FINGER_JOINTS[side][0]
        Rg = [None] * 15
        pg = [None] * 15
        for k in range(15):
            j = f0 + k
            par = int(self.parents[j])
            if par == w:
                Rpar, ppar = Rw, pw
            else:
                Rpar, ppar = Rg[par - f0], pg[par - f0]
            Rg[k] = Rpar @ Rl[:, k]
            pg[k] = ppar + (Rpar @ self.rel[j][:, None])[..., 0]
        Rall = torch.stack([base["Rp"], Rw] + Rg, 1)                  # (T,17,3,3) elbow, wrist, fingers
        pall = torch.stack([base["pp"], pw] + pg, 1)                  # (T,17,3)
        Jr = self.J[[int(self.parents[w])] + base["hj"]]               # (17,3)
        tl = pall - (Rall @ Jr[None, :, :, None])[..., 0]
        # pose correctives of the tips: fixed body part + this hand's joints
        eye = torch.eye(3, dtype=hand_pose.dtype, device=hand_pose.device)
        pf = base["pf_fixed"].clone()
        loc = torch.cat([Rw_loc[:, None], Rl], 1) - eye               # (T,16,3,3)
        for n, j in enumerate(base["hj"]):
            pf[:, 9 * (j - 1):9 * j] = loc[:, n].reshape(T, 9)
        vp = base["tip_v"][None] + torch.einsum("nkp,tp->tnk", base["tip_pd"], pf)
        Rt = torch.einsum("nj,tjab->tnab", base["tip_w"], Rall)
        tt = torch.einsum("nj,tja->tna", base["tip_w"], tl)
        tips = (Rt @ vp[..., None])[..., 0] + tt                        # (T,5,3) thumb..pinky
        # assemble the 21 points; joints are in the posed frame without transl -> base["pp"] already has transl
        pts = []
        for k in HAND21[side]:
            if k == w:
                pts.append(pw)
            elif k < 55:
                pts.append(pg[k - f0])
            else:
                pts.append(tips[:, (k - 66) % 5])
        return torch.stack(pts, 1)

    @staticmethod
    def hand21(out, side: str):
        """(T, 21, 3) hand points (COCO-WholeBody order) from forward() output."""
        import torch
        J = out["joints"]
        tips = out["tips"][:, :5] if side == "left" else out["tips"][:, 5:]
        return torch.stack([J[:, k] if k < 55 else tips[:, (k - 66) % 5] for k in HAND21[side]], 1)


def hand_vertex_ids(model_dir: Path, side: str, joints=None, thresh: float = 0.5) -> np.ndarray:
    """Vertices whose LBS weight on the wrist + fingers of `side` exceeds thresh (the hand surface, ~ MANO's 778)."""
    m = np.load(model_npz(model_dir), allow_pickle=True)
    w = np.asarray(m["weights"], float)
    js = list(joints) if joints is not None else [WRIST[side], *FINGER_JOINTS[side]]
    return np.flatnonzero(w[:, js].sum(1) > thresh)


def arm_vertex_ids(model_dir: Path, side: str, thresh: float = 0.5) -> np.ndarray:
    """Forearm + hand vertices (elbow, wrist, fingers)."""
    return hand_vertex_ids(model_dir, side, [ELBOW[side], WRIST[side], *FINGER_JOINTS[side]], thresh)
