r"""Swinging hair for LHM++ Gaussian avatars (2026-10-20; owner: "I need the larissa kadu recap to look LHM++ quality with
swinging hair"). LHM++ skins its hair Gaussians with the body (voxel-diffused LBS weights): a ponytail or long hair moves
rigidly with the head / upper back and never swings out in a spin. This module replaces the FREE part of the hair (the
ponytail tail, loose hair below the skull) by Gaussians riding on simulated guide strands; scalp-level hair keeps LHM++'s
own skinning.

    python -m dancecap.mono4d.splat_hair prepare --gaussians <role>_cano.ply|<role>.splatav --betas poses.npz|B.json
                                                 --role follow --style ponytail|loose --out HAIR [--device cuda]
    python -m dancecap.mono4d.splat_hair render-tracks --hair HAIR --input <render input dir> --role follow
                                                 --partner lead --groups g1,g2          # tracks + hair.json for a render
    python -m dancecap.mono4d.splat_hair simulate --hair HAIR --poses poses.npz --role follow --keys a:b [--partner lead]
    python -m dancecap.mono4d.splat_hair jl-proof {prep,job,import,video}      # the john_lula orbit proof (rigid vs swinging)

SWITCHING HAIR ON in an LHM++ render (remote/cand_lhmpp/lhmpp_render_pod.py): `prepare` once per avatar (from the canonical
PLY the render pod saves, <out>/gs/<role>_cano.ply, or a .splatav; LOOK at <role>_select.jpg / <role>_strands.jpg), then
`render-tracks` against the render's own input dir (poses.npz + views.npz) naming the groups that should swing, and pass
`--hair HAIR` to lhmpp_render_pod.py (jl_realism_render: `job --hair HAIR [--renders DIR]`). Without --hair the pod renders
exactly as before. The pod rebuilds the same avatar (same refs -> same canonical; checked: cano_vs_bundle_mm in its
report), so the bundle's Gaussian indices apply. hair.json {"uncover_fill": true} also recolours the body surface the hair
hid (experimental, off: it was not a clear win on john_lula).

Pipeline (one avatar, canonical = LHM++'s saved `*_cano.ply` = the render pod's `animate(...)[1]`: the shaped SMPL-X frame
of 10 betas, head / neck / spine NOT rotated (only hips and shoulders are: LHM++'s A-pose), so around the head the canonical
frame equals the SMPL-X zero-pose frame and a head Gaussian is posed by S_15(t) = G_15(t) T(-J_15) exactly (measured on
john_lula's scene_key3020.ply: median 0.0 mm)):

  (a) select_hair: hair colour (a colour model learnt from the scalp Gaussians: the ones sitting OUTSIDE the SMPL-X scalp
      above the hairline) AND position (near the head / neck / upper back, behind the face) AND connectivity (voxel
      clusters). Split: RIGID = scalp-level hair (within `scalp_shell_m` of the head capsule and above the hang level),
      FREE = the rest (the hanging tail / curtain). LOOK at the sheet (select_sheet).
  (b) build_strands: guide strands through the free hair, rest shape from the canonical Gaussians. Ponytail: a bundle of
      `n_strands` strands from the tie point (the top of the free hair mass at the back of the head) down the tail
      centreline, each offset in the tail cross-section. Loose: strands from the scalp boundary (the hang line), one per
      azimuth sector, each the opacity-weighted centreline of its sector's free Gaussians by height slab.
  (c) simulate: position-based dynamics (Verlet + PBD projections) per 30 Hz frame with `substeps` substeps: gravity,
      inextensible segments (follow-the-leader within max_stretch), bending (distance to the second neighbour, two-sided)
      + a shape-matching pull toward the head-carried rest shape that falls off along the strand (root held, tip free),
      bundle cohesion (linked strands never drift apart beyond link_slack: the tail stays one bundle), damping,
      collisions with the capsules of the body (head, neck, shoulder bar, chest, mid-back, upper arms, forearms:
      hair_groom.colliders' set; radii made rest-safe, LHM++'s hair rests ON the body) and of the PARTNER (head, neck,
      shoulders, chest, mid-back, arms), roots pinned to the head. Driven by the SMPL-X FK of the fit (10 betas, the same
      body LHM++ animates). Deterministic (no random numbers; float64 numpy).
  (d) bind: every free Gaussian rides on its nearest rest segment: segment frame per docs/SPLAT_AVATAR_FORMAT.md section 6
      (origin = point s, Y = unit tangent to point s+1, Z = the head's forward axis made orthogonal to Y, X = Y x Z; posed
      frame = rest frame carried by the head rotation, then the minimal rotation from the head-carried rest tangent to the
      simulated tangent). posed centre = origin + F * local_offset, posed rotation = F * local_rotation.
  (e) check: segment lengths (max stretch), points inside the head / body capsules, step-to-step speed (explosions), tip
      swing relative to the head-rigid tail in spins (check_sim; worst frames listed).

Strand TRACKS shipped to the render pod are stored HEAD-LOCAL in the canonical frame (X_world = S_15(t) X_stored): the pod
maps them with its own head transform of the frame (Kabsch of the rigid head Gaussians, canonical -> LHM++ posed), so the
strands sit exactly on LHM++'s head whatever tiny difference its FK has from ours. PodHair is the pod side (numpy only).

Units metres, world Y up right-handed (dancecap world), quaternions w x y z (gsplat / 3DGS / LHM++ convention) unless a
name says xyzw. Likeness data (avatars, renders) stays in work/.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

SH_C0 = 0.28209479177387814
J_PELVIS, J_SPINE1, J_SPINE2, J_SPINE3, J_NECK, J_HEAD = 0, 3, 6, 9, 12, 15
J_LCOLLAR, J_RCOLLAR, J_LSH, J_RSH, J_LELB, J_RELB = 13, 14, 16, 17, 18, 19
HEAD_JOINTS = (15, 22, 23, 24)

# default simulation parameters (per style). Lengths in m, times in s. Shape-matching stiffness k(s) = k_root *
# exp(-s / falloff_m) per substep (s = rest arc length from the root), clamped to [0, 1].
SIM = {
    "ponytail": {"substeps": 12, "iterations": 4, "gravity": 9.81, "damping_per_s": 1.6, "bend": 0.1,
                 "shape_root": 0.9, "shape_falloff_m": 0.05, "shape_floor": 0.0, "particle_radius_m": 0.008,
                 "friction": 0.3, "max_stretch": 1.01, "max_push_m": 0.01, "preroll_frames": 45, "link_slack": 0.25,
                 "link_stiffness": 0.5},
    "loose": {"substeps": 12, "iterations": 4, "gravity": 9.81, "damping_per_s": 2.2, "bend": 0.1,
              "shape_root": 0.95, "shape_falloff_m": 0.06, "shape_floor": 0.0, "particle_radius_m": 0.008,
              "friction": 0.3, "max_stretch": 1.01, "max_push_m": 0.01, "preroll_frames": 45, "link_slack": 0.4,
              "link_stiffness": 0.4},
}
SELECT = {
    "ponytail": {"n_strands": 9, "points": 12, "scalp_shell_m": 0.025, "hang_below_head_joint_m": 0.02,
                 "free_off_body_m": 0.008, "grow_m": 0.008, "under_m": 0.025, "behind_m": 0.05, "halfwidth_m": 0.10,
                 "behind_neck_m": 0.01,
                 "max_from_head_m": 0.60, "min_cluster_frac": 0.02},
    "loose": {"n_strands": 24, "points": 12, "scalp_shell_m": 0.030, "hang_below_head_joint_m": 0.00,
              "free_off_body_m": 0.008, "grow_m": 0.008, "under_m": 0.02, "behind_m": None, "halfwidth_m": None, "behind_neck_m": None,
              "max_from_head_m": 0.75, "min_cluster_frac": 0.02},
}


def log(msg: str) -> None:
    print(f"[splat_hair] {msg}", flush=True)


# ================================================================================================ small math (numpy)
def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """(..., 4) w x y z -> (..., 3, 3)."""
    q = np.asarray(q, float)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
                     np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
                     np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], -2)


def mat_to_quat(R: np.ndarray) -> np.ndarray:
    """(..., 3, 3) -> (..., 4) w x y z, w >= 0 (Shepperd, branch-free enough for numpy)."""
    R = np.asarray(R, float)
    shp = R.shape[:-2]
    R = R.reshape(-1, 3, 3)
    q = np.empty((len(R), 4))
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    cands = np.stack([tr, R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]], 1)
    k = cands.argmax(1)
    for i in range(4):
        m = k == i
        if not m.any():
            continue
        r = R[m]
        if i == 0:
            s = np.sqrt(np.maximum(1 + tr[m], 1e-12)) * 2
            q[m] = np.stack([0.25 * s, (r[:, 2, 1] - r[:, 1, 2]) / s, (r[:, 0, 2] - r[:, 2, 0]) / s,
                             (r[:, 1, 0] - r[:, 0, 1]) / s], 1)
        elif i == 1:
            s = np.sqrt(np.maximum(1 + r[:, 0, 0] - r[:, 1, 1] - r[:, 2, 2], 1e-12)) * 2
            q[m] = np.stack([(r[:, 2, 1] - r[:, 1, 2]) / s, 0.25 * s, (r[:, 0, 1] + r[:, 1, 0]) / s,
                             (r[:, 0, 2] + r[:, 2, 0]) / s], 1)
        elif i == 2:
            s = np.sqrt(np.maximum(1 + r[:, 1, 1] - r[:, 0, 0] - r[:, 2, 2], 1e-12)) * 2
            q[m] = np.stack([(r[:, 0, 2] - r[:, 2, 0]) / s, (r[:, 0, 1] + r[:, 1, 0]) / s, 0.25 * s,
                             (r[:, 1, 2] + r[:, 2, 1]) / s], 1)
        else:
            s = np.sqrt(np.maximum(1 + r[:, 2, 2] - r[:, 0, 0] - r[:, 1, 1], 1e-12)) * 2
            q[m] = np.stack([(r[:, 1, 0] - r[:, 0, 1]) / s, (r[:, 0, 2] + r[:, 2, 0]) / s,
                             (r[:, 1, 2] + r[:, 2, 1]) / s, 0.25 * s], 1)
    q *= np.where(q[:, :1] < 0, -1.0, 1.0)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return q.reshape(shp + (4,))


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of w x y z quaternions (broadcasting)."""
    aw, ax, ay, az = np.moveaxis(np.asarray(a, float), -1, 0)
    bw, bx, by, bz = np.moveaxis(np.asarray(b, float), -1, 0)
    return np.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], -1)


def rotvec_to_mat(v: np.ndarray) -> np.ndarray:
    """(..., 3) axis-angle -> (..., 3, 3) (Rodrigues)."""
    v = np.asarray(v, float)
    th = np.linalg.norm(v, axis=-1, keepdims=True)
    k = v / np.maximum(th, 1e-12)
    kx, ky, kz = np.moveaxis(k, -1, 0)
    z = np.zeros_like(kx)
    Kx = np.stack([np.stack([z, -kz, ky], -1), np.stack([kz, z, -kx], -1), np.stack([-ky, kx, z], -1)], -2)
    th = th[..., None]
    I = np.broadcast_to(np.eye(3), Kx.shape)
    return I + np.sin(th) * Kx + (1 - np.cos(th)) * (Kx @ Kx)


def min_rotation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(..., 3, 3) the minimal rotation taking unit vectors a to b (Rodrigues about a x b; a = -b: 180 deg about any
    perpendicular axis)."""
    a = np.asarray(a, float) / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)
    b = np.asarray(b, float) / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-12)
    v = np.cross(a, b)
    c = (a * b).sum(-1)
    s = np.linalg.norm(v, axis=-1)
    vx, vy, vz = np.moveaxis(v, -1, 0)
    z = np.zeros_like(vx)
    K = np.stack([np.stack([z, -vz, vy], -1), np.stack([vz, z, -vx], -1), np.stack([-vy, vx, z], -1)], -2)
    f = np.where(c > -1 + 1e-9, 1.0 / np.maximum(1 + c, 1e-9), 0.0)[..., None, None]
    R = np.broadcast_to(np.eye(3), K.shape) + K + (K @ K) * f
    bad = c <= -1 + 1e-9
    if np.any(bad):
        ab = a[bad]
        p = np.cross(ab, np.where(np.abs(ab[:, :1]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]]))
        p /= np.linalg.norm(p, axis=1, keepdims=True)
        R = R.copy()
        R[bad] = 2 * p[:, :, None] * p[:, None, :] - np.eye(3)
    del s
    return R


def segment_closest(P: np.ndarray, A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Closest points on segments A-B (broadcast) to P: (point, parameter t in [0,1])."""
    ab = B - A
    t = np.clip(((P - A) * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-12), 0.0, 1.0)
    return A + t[..., None] * ab, t


# ================================================================================================ LHM++ canonical PLY
def load_ply(path: Path) -> dict:
    """A 3DGS PLY (LHM++ GaussianModel.save_ply) -> xyz, rgb (0..1, colour = 0.5 + C0 * f_dc), opacity (0..1), scale
    (linear std, m), quat (w x y z, unit)."""
    from plyfile import PlyData
    v = PlyData.read(str(path))["vertex"].data
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1).astype(np.float64)
    q = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], 1).astype(np.float64)
    q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    return {"xyz": xyz, "rgb": np.clip(dc * SH_C0 + 0.5, 0, 1), "opacity": 1 / (1 + np.exp(-np.asarray(v["opacity"], float))),
            "scale": np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1).astype(np.float64)), "quat": q}


# ================================================================================================ SMPL-X rest body + FK
class RestBody:
    """Shaped SMPL-X zero pose of 10 betas in the SMPL-X native frame (no translation): the frame of LHM++'s canonical
    Gaussians around the head. Duck-types hair_groom.RestModel (v, J, f, w, dom, n) for hair_groom.scalp / colliders."""

    def __init__(self, betas, model_npz: Path | None = None, num_betas: int = 10):
        if model_npz is None:
            from ..take import WORKSPACE
            model_npz = WORKSPACE / "smplx" / "SMPLX_NEUTRAL.npz"
        m = np.load(str(model_npz), allow_pickle=True)
        b = np.asarray(betas, float).reshape(-1)[:num_betas]
        sd = np.asarray(m["shapedirs"], float)[:, :, :len(b)]
        self.v = np.asarray(m["v_template"], float) + sd @ b
        self.J = (np.asarray(m["J_regressor"], float) @ self.v)[:55]
        self.f = np.asarray(m["f"], np.int64)
        self.w = np.asarray(m["weights"], float)
        self.dom = self.w.argmax(1)
        self.parents = np.asarray(m["kintree_table"], np.int64)[0][:55].copy()
        self.parents[0] = -1
        vv = self.v[self.f]
        nrm = np.cross(vv[:, 1] - vv[:, 0], vv[:, 2] - vv[:, 0])
        n = np.zeros_like(self.v)
        for k in range(3):
            np.add.at(n, self.f[:, k], nrm)
        self.n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        self.betas = b
        self.height = float(self.v[:, 1].max() - self.v[:, 1].min())


def fk_world(rest: RestBody, root: np.ndarray, body: np.ndarray, pelvis: np.ndarray, extra: np.ndarray | None = None) -> tuple:
    """SMPL-X FK for F frames: root (F,3) / body (F,21,3 or 63) axis-angle [+ extra (F,33,3): jaw, leye, reye, 15 left
    hand, 15 right hand = joints 22..54], pelvis (F,3) = the posed pelvis joint (trans = pelvis - J_0).
    -> (Rg (F,J,3,3) world rotations, P (F,J,3) world joint positions), J = 22 or 55.
    Skin transform of joint j: X_posed = Rg_j (X_rest - J_j) + P_j."""
    F = len(root)
    loc = [np.asarray(root, float).reshape(F, 1, 3), np.asarray(body, float).reshape(F, 21, 3)]
    if extra is not None:
        loc.append(np.asarray(extra, float).reshape(F, 33, 3))
    loc = np.concatenate(loc, 1)
    nj = loc.shape[1]
    Rl = rotvec_to_mat(loc)
    Rg = np.zeros((F, nj, 3, 3))
    P = np.zeros((F, nj, 3))
    Rg[:, 0] = Rl[:, 0]
    P[:, 0] = np.asarray(pelvis, float)
    for j in range(1, nj):
        p = rest.parents[j]
        Rg[:, j] = Rg[:, p] @ Rl[:, j]
        P[:, j] = P[:, p] + np.einsum("fij,j->fi", Rg[:, p], rest.J[j] - rest.J[p])
    return Rg, P


def fk_keys(rest: RestBody, Z: dict, role: str, keys, full: bool = False) -> tuple:
    """fk_world of pose keys of an LHM++ render input poses.npz (per role root/body/jaw/leye/reye/lhand/rhand/pelvis)."""
    k = np.asarray(keys, int)
    extra = None
    if full:
        extra = np.concatenate([np.asarray(Z[f"{role}_{n}"][k], float).reshape(len(k), -1, 3)
                                for n in ("jaw", "leye", "reye", "lhand", "rhand")], 1)
    return fk_world(rest, Z[f"{role}_root"][k], Z[f"{role}_body"][k], Z[f"{role}_pelvis"][k], extra)


def lbs_pose(g: dict, rest: RestBody, Rg: np.ndarray, Pj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Plain 4-joint LBS of a zero-pose splat avatar (load_gaussians of a .splatav: lbs_joints / lbs_weights) for ONE
    frame (Rg (55,3,3), Pj (55,3)) -> posed centres, rotations (w x y z) (docs/SPLAT_AVATAR_FORMAT.md section 4)."""
    S = np.zeros((len(Rg), 3, 4))
    S[:, :, :3] = Rg
    S[:, :, 3] = Pj - np.einsum("jik,jk->ji", Rg, rest.J[:len(Rg)])
    M = np.einsum("nk,nkij->nij", g["lbs_weights"], S[g["lbs_joints"]])
    mean = np.einsum("nij,nj->ni", M[:, :, :3], g["xyz"]) + M[:, :, 3]
    U, _, Vt = np.linalg.svd(M[:, :, :3])
    Rn = U @ Vt
    Rn[np.linalg.det(Rn) < 0] *= -1
    return mean, quat_mul(mat_to_quat(Rn), g["quat"])


# ================================================================================================ check renderer (torch)
def look_at(C, P, up=(0.0, 1.0, 0.0)) -> tuple[np.ndarray, np.ndarray]:
    """World-to-camera R, t of a pinhole at C looking at P, image y down (OpenCV), world Y up."""
    z = np.asarray(P, float) - np.asarray(C, float)
    z /= np.linalg.norm(z)
    x = np.cross(-np.asarray(up, float), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return R, -R @ np.asarray(C, float)


def render_check(xyz, quat, scale, opacity, rgb, R, t, K, W: int, H: int, bg=(1.0, 1.0, 1.0), device: str = "cpu",
                 eps2d: float = 0.3, chunk: int = 40000, max_radius: int = 48, max_pairs: int = 4_000_000) -> np.ndarray:
    """Plain EWA splatting (3DGS forward model, eps2d dilation, front-to-back alpha compositing, alpha <= 0.99) in torch,
    for check sheets on machines without a gsplat build. Gaussians are depth sorted and composited in chunks (exact:
    chunk images are composited front to back with the running transmittance). -> uint8 RGB (H, W, 3)."""
    import torch
    f32 = lambda a: torch.as_tensor(np.ascontiguousarray(a, np.float32), device=device)  # noqa: E731
    X, q, s, o, c = f32(xyz), f32(quat), f32(scale), f32(opacity).reshape(-1), f32(rgb)
    Rm, tv, Km = f32(R), f32(t).reshape(3), f32(K)
    Xc = X @ Rm.T + tv
    z = Xc[:, 2]
    fx, fy, cx, cy = Km[0, 0], Km[1, 1], Km[0, 2], Km[1, 2]
    zs = z.clamp_min(1e-3)
    u = fx * Xc[:, 0] / zs + cx
    v = fy * Xc[:, 1] / zs + cy
    q = q / q.norm(dim=1, keepdim=True).clamp_min(1e-12)
    w_, x_, y_, z_ = q.unbind(1)
    Rq = torch.stack([torch.stack([1 - 2 * (y_ * y_ + z_ * z_), 2 * (x_ * y_ - w_ * z_), 2 * (x_ * z_ + w_ * y_)], -1),
                      torch.stack([2 * (x_ * y_ + w_ * z_), 1 - 2 * (x_ * x_ + z_ * z_), 2 * (y_ * z_ - w_ * x_)], -1),
                      torch.stack([2 * (x_ * z_ - w_ * y_), 2 * (y_ * z_ + w_ * x_), 1 - 2 * (x_ * x_ + y_ * y_)], -1)], -2)
    M = Rm[None] @ Rq * s[:, None, :]
    cov = M @ M.transpose(1, 2)
    Jm = torch.zeros(len(X), 2, 3, device=device)
    Jm[:, 0, 0] = fx / zs
    Jm[:, 0, 2] = -fx * Xc[:, 0] / zs ** 2
    Jm[:, 1, 1] = fy / zs
    Jm[:, 1, 2] = -fy * Xc[:, 1] / zs ** 2
    c2 = Jm @ cov @ Jm.transpose(1, 2)
    a, b, d = c2[:, 0, 0] + eps2d, c2[:, 0, 1], c2[:, 1, 1] + eps2d
    det = (a * d - b * b).clamp_min(1e-12)
    ia, ib, idd = d / det, -b / det, a / det
    mid = 0.5 * (a + d)
    lam = mid + torch.sqrt((mid * mid - det).clamp_min(0.1))
    rad = torch.ceil(3.0 * torch.sqrt(lam)).clamp(1, max_radius).long()
    ok = (z > 0.05) & (o >= 1 / 255) & (u + rad >= 0) & (u - rad < W) & (v + rad >= 0) & (v - rad < H)
    idx = torch.nonzero(ok).reshape(-1)
    idx = idx[torch.argsort(z[idx])]
    img = torch.zeros(H * W, 3, device=device)
    Tr = torch.ones(H * W, device=device)
    # chunks of consecutive (depth-ordered) Gaussians holding at most `max_pairs` (pixel, Gaussian) pairs each
    cum = torch.cumsum((2 * rad[idx] + 1) ** 2, 0).cpu().numpy()
    bounds, i0 = [], 0
    while i0 < len(idx):
        base = cum[i0 - 1] if i0 > 0 else 0
        i1 = int(np.searchsorted(cum, base + max_pairs, side="right"))
        i1 = min(max(i1, i0 + 1), i0 + chunk, len(idx))
        bounds.append((i0, i1))
        i0 = i1
    for i0, i1 in bounds:
        gi = idx[i0:i1]
        r = rad[gi]
        side = 2 * r + 1
        cnt = side * side
        off = torch.cumsum(cnt, 0) - cnt
        tot = int(cnt.sum())
        if tot == 0:
            continue
        g = torch.repeat_interleave(torch.arange(len(gi), device=device), cnt)
        loc = torch.arange(tot, device=device) - off[g]
        dx = (loc % side[g]) - r[g]
        dy = torch.div(loc, side[g], rounding_mode="floor") - r[g]
        G = gi[g]
        px = torch.round(u[G]).long() + dx
        py = torch.round(v[G]).long() + dy
        ins = (px >= 0) & (px < W) & (py >= 0) & (py < H)
        G, px, py, g = G[ins], px[ins], py[ins], g[ins]
        ddx = px.float() - u[G]
        ddy = py.float() - v[G]
        pw = -0.5 * (ia[G] * ddx * ddx + 2 * ib[G] * ddx * ddy + idd[G] * ddy * ddy)
        al = (o[G] * torch.exp(pw.clamp_max(0))).clamp_max(0.99)
        keep = al >= 1 / 255
        al, G, g = al[keep], G[keep], g[keep]
        pix = (py * W + px)[keep]
        order = torch.argsort(pix * (len(gi) + 1) + g)       # by pixel, then depth (g is depth-ordered in the chunk)
        pix, al, G = pix[order], al[order], G[order]
        l1 = torch.log1p(-al)
        cs = torch.cumsum(l1, 0)
        first = torch.ones_like(pix, dtype=torch.bool)
        first[1:] = pix[1:] != pix[:-1]
        start = torch.cummax(torch.where(first, torch.arange(len(pix), device=device), torch.zeros_like(pix)), 0).values
        excl = cs - l1 - (cs[start] - l1[start])
        wgt = al * torch.exp(excl) * Tr[pix]
        img.index_add_(0, pix, wgt[:, None] * c[G])
        chunkT = torch.zeros(H * W, device=device).index_add_(0, pix, l1)
        Tr = Tr * torch.exp(chunkT)
    out = img + Tr[:, None] * torch.tensor(bg, dtype=torch.float32, device=device)[None]
    return (out.clamp(0, 1).reshape(H, W, 3).cpu().numpy() * 255 + 0.5).astype(np.uint8)


# ================================================================================================ (a) hair selection
def load_gaussians(path: Path) -> dict:
    """Canonical Gaussians from an LHM++ PLY (`*_cano.ply`) or a splat avatar (`.splatav`, docs/SPLAT_AVATAR_FORMAT.md:
    Unity coordinates -> dancecap world: points (x, y, -z), quaternions xyzw (-qx, -qy, qz, qw) -> w x y z)."""
    path = Path(path)
    if path.suffix.lower() == ".ply":
        return load_ply(path)
    raw = path.read_bytes()
    assert raw[:4] == b"SPAV", raw[:4]
    hdr = np.frombuffer(raw[4:64], "<u4")
    n, rb, hb = int(hdr[1]), int(hdr[2]), int(hdr[5])
    dt = np.dtype([("mean", "<f4", 3), ("rotation", "<f4", 4), ("log_scale", "<f4", 3), ("opacity", "<f4"),
                   ("sh0", "<f4", 3), ("joints", "<u2", 4), ("weights", "<f4", 4), ("hair", "u1"), ("reserved", "u1", 3),
                   ("strand", "<i4"), ("segment", "<i4"), ("local_offset", "<f4", 3), ("local_rotation", "<f4", 4)])
    rec = np.frombuffer(raw[hb:hb + n * rb], dt)
    xyz = rec["mean"].astype(np.float64) * np.array([1.0, 1.0, -1.0])
    qu = rec["rotation"].astype(np.float64)
    q = np.stack([qu[:, 3], -qu[:, 0], -qu[:, 1], qu[:, 2]], 1)
    q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    return {"xyz": xyz, "rgb": np.clip(rec["sh0"].astype(np.float64) * SH_C0 + 0.5, 0, 1),
            "opacity": 1 / (1 + np.exp(-rec["opacity"].astype(np.float64))),
            "scale": np.exp(rec["log_scale"].astype(np.float64)), "quat": q,
            "lbs_joints": rec["joints"].astype(np.int64), "lbs_weights": rec["weights"].astype(np.float64)}


def head_capsule(rest: RestBody) -> dict:
    """hair_groom's head capsule (front-back capsule fitted to the scalp vertices) + head sphere, in rest coordinates."""
    from .. import hair_groom as HG
    sp = HG.scalp(rest)
    hc, hh, hr, res = HG.fit_head_capsule(rest.v[sp["scalp_v"]], sp["centre"], sp["radius"])
    return {"a": hc - np.array([0, 0, hh]), "b": hc + np.array([0, 0, hh]), "r": float(hr), "centre": sp["centre"],
            "sphere_r": float(sp["radius"]), "fit_p95_mm": round(1000 * res, 2), "crown_y": sp["crown_y"]}


def body_features(g: dict, rest: RestBody, hc: dict) -> dict:
    """Per Gaussian: signed distance to the SMPL-X rest surface (nearest vertex normal), its dominant joint, the distance
    outside the head capsule, azimuth / elevation about the head-sphere centre (deg; az 0 = facing +z, +90 = +x)."""
    from scipy.spatial import cKDTree
    from .. import hair_groom as HG
    x = g["xyz"]
    d, vi = cKDTree(rest.v).query(x)
    sd = ((x - rest.v[vi]) * rest.n[vi]).sum(1)
    cp, _ = segment_closest(x, hc["a"][None], hc["b"][None])
    dh = np.linalg.norm(x - cp, axis=1) - hc["r"]
    az, el = HG.az_el(x - hc["centre"])
    return {"sd": sd, "dist": d, "dom": rest.dom[vi], "dh": dh, "az": az, "el": el, "normal": rest.n[vi]}


def _lab(rgb: np.ndarray) -> np.ndarray:
    import cv2
    return cv2.cvtColor(np.clip(rgb, 0, 1).astype(np.float32)[None], cv2.COLOR_RGB2LAB)[0].astype(np.float64)


def _clusters(x: np.ndarray, cell: float) -> np.ndarray:
    """Voxel connected components (26-connected) of points x -> label per point (0..n-1)."""
    from scipy import ndimage
    gi = np.floor((x - x.min(0)) / cell).astype(int)
    occ = np.zeros(gi.max(0) + 1, bool)
    occ[tuple(gi.T)] = True
    lab, _ = ndimage.label(occ, structure=np.ones((3, 3, 3)))
    return lab[tuple(gi.T)] - 1


HAIR_REGION_JOINTS = (3, 6, 9, 12, 13, 14, 15, 16, 17, 22, 23, 24)


def select_hair(g: dict, rest: RestBody, style: str = "ponytail", params: dict | None = None) -> dict:
    """Hair Gaussians of a canonical avatar split into RIGID (scalp-level, keeps LHM++'s skinning) and FREE (simulated).

    1. region: Gaussians whose nearest SMPL-X vertex belongs to the head / neck / spine / collars / shoulders, within
       max_from_head_m of the head joint, outside the body (signed distance > -5 mm);
    2. colour model: HAIR reference = the crown (elevation > 45 deg, 3 mm .. 6 cm outside the scalp), SKIN / CLOTHES
       reference = the face (front of the head below the hairline) + the front of the torso; a Gaussian is hair-coloured
       when its Lab distance to the hair reference (Mahalanobis, opacity weighted) is below its distance to the other;
    3. scalp hair (RIGID): hair-coloured, above the hairline elevation (hair_groom.HAIRLINE_EL), at most scalp_shell_m
       outside the head capsule and above the hang level;
    4. FREE hair: off the body (body signed distance > free_off_body_m), outside the scalp shell or below the hang
       level (head joint y - hang_below_head_joint_m), behind the face (not in the face cone below the hairline),
       hair-coloured OR far off the body (> 2 x free_off_body_m: the hanging hair volume, whatever LHM++ painted there,
       so no ghost of the tail stays behind); then voxel clusters (1.5 cm) holding >= min_cluster_frac of the seeds and
       reaching the head (within 4 cm of the head capsule); then grown by grow_m into off-body Gaussians around them.
    -> {"hair", "free", "rigid": bool masks, "info": {...}, "features": body_features, "hair_colour": bool mask}"""
    from .. import hair_groom as HG
    P = {**SELECT[style], **(params or {})}
    hc = head_capsule(rest)
    f = body_features(g, rest, hc)
    x, op = g["xyz"], g["opacity"]
    J = rest.J
    region = (np.isin(f["dom"], HAIR_REGION_JOINTS) & (np.linalg.norm(x - J[J_HEAD], axis=1) < P["max_from_head_m"])
              & (f["sd"] > -0.005))
    lab = _lab(g["rgb"])
    crown = region & (f["el"] > 45) & (f["dh"] > 0.003) & (f["dh"] < 0.06) & np.isin(f["dom"], HEAD_JOINTS)
    face = (np.isin(f["dom"], HEAD_JOINTS) & (np.abs(f["az"]) < 45) & (f["el"] < HG.hairline_el(f["az"]) - 12)
            & (f["el"] > -40) & (f["dh"] < 0.02))
    torso_front = np.isin(f["dom"], (3, 6, 9)) & (x[:, 2] > J[J_SPINE3, 2] + 0.03) & (np.abs(f["sd"]) < 0.02)
    other = face | torso_front

    def wstats(m):
        w = op[m] + 1e-3
        mu = (lab[m] * w[:, None]).sum(0) / w.sum()
        C = np.cov((lab[m] - mu).T, aweights=w) + np.eye(3) * 4.0
        return mu, np.linalg.inv(C)
    if crown.sum() < 50:
        raise ValueError(f"select_hair: only {int(crown.sum())} crown Gaussians: no hair to learn from (bald?)")
    mu_h, Ih = wstats(crown)
    mu_o, Io = wstats(other)
    dhm = np.einsum("ni,ij,nj->n", lab - mu_h, Ih, lab - mu_h)
    dom_ = np.einsum("ni,ij,nj->n", lab - mu_o, Io, lab - mu_o)
    hairc = dhm < dom_
    hang_y = J[J_HEAD, 1] - P["hang_below_head_joint_m"]
    face_cone = (np.abs(f["az"]) < 70) & (f["el"] < HG.hairline_el(f["az"]))
    above_line = f["el"] >= HG.hairline_el(f["az"]) - 4
    shell = f["dh"] <= P["scalp_shell_m"]
    rigid = region & hairc & above_line & shell & (x[:, 1] >= hang_y - 0.02) & np.isin(f["dom"], HEAD_JOINTS)
    off = P["free_off_body_m"]
    cand = region & ~face_cone & (f["sd"] > off) & (~shell | (x[:, 1] < hang_y)) & (hairc | (f["sd"] > 2 * off))
    if P.get("behind_m") is not None:       # above the hang level only what is behind the head (the tail's root)
        cand &= (x[:, 1] < hang_y) | (x[:, 2] < hc["centre"][2] - P["behind_m"])
    if P.get("halfwidth_m") is not None:    # a ponytail is a bundle at the back: nothing out on the shoulders
        cand &= np.abs(x[:, 0] - J[J_HEAD, 0]) < P["halfwidth_m"]
    if P.get("behind_neck_m") is not None:  # ... and behind the neck (locks at the sides of the neck stay rigid)
        cand &= x[:, 2] < J[J_NECK, 2] - P["behind_neck_m"]
    seeds = np.nonzero(cand)[0]
    info = {"style": style, "params": P,
            "head_capsule": {k: (np.round(v, 4).tolist() if isinstance(v, np.ndarray) else v) for k, v in hc.items()},
            "region": int(region.sum()), "crown_ref": int(crown.sum()), "other_ref": int(other.sum()),
            "hair_coloured_in_region": int((region & hairc).sum()), "hang_y": round(float(hang_y), 4),
            "colour_hair_lab": np.round(mu_h, 2).tolist(), "colour_other_lab": np.round(mu_o, 2).tolist(),
            "free_candidates": int(len(seeds))}
    free = np.zeros(len(x), bool)
    if len(seeds):
        lb = _clusters(x[seeds], 0.015)
        sizes = np.bincount(lb)
        big = np.nonzero(sizes >= max(20, P["min_cluster_frac"] * len(seeds)))[0]
        keep = np.isin(lb, big)
        for c_ in big:          # a kept cluster must reach the head: no loose specks on the back
            m = lb == c_
            if f["dh"][seeds[m]].min() > 0.04:
                keep &= ~m
        free[seeds[keep]] = True
        info["clusters"] = int(len(sizes))
        info["clusters_kept"] = int(len(np.unique(lb[keep])))
        from scipy.spatial import cKDTree
        lat = np.abs(x[:, 0] - J[J_HEAD, 0]) < P["halfwidth_m"] if P.get("halfwidth_m") is not None else np.ones(len(x), bool)
        if P.get("behind_neck_m") is not None:
            lat &= x[:, 2] < J[J_NECK, 2] - P["behind_neck_m"]
        if P.get("grow_m", 0) > 0 and free.any():
            pool = np.nonzero(region & ~free & ~rigid & ~face_cone & lat & (f["sd"] > 0.5 * off))[0]
            if len(pool):
                d, _ = cKDTree(x[free]).query(x[pool], distance_upper_bound=P["grow_m"])
                free[pool[np.isfinite(d)]] = True
                info["grown"] = int(np.isfinite(d).sum())
        if P.get("under_m", 0) > 0 and free.any():
            # the hair LAYER under the hanging hair, on the body surface (LHM++ paints the hidden neck / back with hair
            # colour): it rides with the tail too, or a ghost of the tail stays painted on the neck when it swings out
            dark = lab[:, 0] < mu_o[0] - 8
            pool = np.nonzero(region & ~free & ~rigid & ~face_cone & lat & (f["sd"] > -0.006) & (x[:, 1] < hang_y)
                              & (f["normal"][:, 2] < -0.2) & (hairc | dark))[0]
            if len(pool):
                d, _ = cKDTree(x[free]).query(x[pool], distance_upper_bound=P["under_m"])
                free[pool[np.isfinite(d)]] = True
                info["under_layer"] = int(np.isfinite(d).sum())
    rigid &= ~free
    hair = rigid | free
    info.update({"rigid": int(rigid.sum()), "free": int(free.sum()), "hair": int(hair.sum()),
                 "free_opacity_sum": round(float(op[free].sum()), 1),
                 "free_y_range": [round(float(x[free, 1].min()), 3), round(float(x[free, 1].max()), 3)] if free.any() else None,
                 "head_joint": np.round(J[J_HEAD], 4).tolist()})
    return {"hair": hair, "free": free, "rigid": rigid, "info": info, "features": f, "hair_colour": hairc}


def select_sheet(g: dict, sel: dict, rest: RestBody, path: Path, device: str = "cpu") -> Path:
    """LOOK: rows = (1) the avatar's head / upper body, (2) FREE hair tinted green / RIGID hair tinted blue, (3) the free
    hair alone, (4) the avatar WITHOUT the free hair (what a swinging tail uncovers); columns = 6 azimuths."""
    import cv2
    J = rest.J
    ctr = np.array([J[J_HEAD, 0], J[J_HEAD, 1] - 0.12, J[J_HEAD, 2]])
    W, H = 420, 560
    K = np.array([[1500.0, 0, W / 2], [0, 1500.0, H / 2], [0, 0, 1]])
    rgb = g["rgb"]
    tint = rgb.copy()
    tint[sel["free"]] = 0.35 * rgb[sel["free"]] + 0.65 * np.array([0.1, 0.9, 0.1])
    tint[sel["rigid"]] = 0.35 * rgb[sel["rigid"]] + 0.65 * np.array([0.1, 0.3, 0.95])
    rows = []
    for mode in ("plain", "tint", "free", "rest"):
        m = np.ones(len(rgb), bool) if mode in ("plain", "tint") else (sel["free"] if mode == "free" else ~sel["free"])
        cols = []
        for az in (0, 60, 120, 180, 240, 300):
            C = ctr + 1.5 * np.array([math.sin(math.radians(az)), 0.15, math.cos(math.radians(az))])
            R, t = look_at(C, ctr)
            im = render_check(g["xyz"][m], g["quat"][m], g["scale"][m], g["opacity"][m],
                              (tint if mode == "tint" else rgb)[m], R, t, K, W, H, device=device)
            cv2.putText(im, f"{mode} az{az}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
            cols.append(im)
        rows.append(np.concatenate(cols, 1))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.concatenate(rows, 0)[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])
    return path


def uncover_fill(g: dict, sel: dict, rest: RestBody, style: str = "ponytail", params: dict | None = None) -> dict:
    """The body surface the free hair hid: LHM++ never saw it and paints it with the hair's colour (a dark "ghost" of the
    tail on the neck / back once the hair swings away). Zone = Gaussians that are not free, on or near the body surface
    (signed distance -2.5 .. +1.5 cm), facing back (normal z < -0.2), below the head joint, within zone_m of a free
    Gaussian (and inside the ponytail's half width). Their colour becomes the opacity-weighted median of the RING: body
    Gaussians at the same height (2 cm bins) just outside the zone (ring_m wider, same facing, not hair-coloured).
    Only the swinging render uses it (the rigid tail covers the zone). -> {"idx", "rgb", "info"}"""
    from scipy.spatial import cKDTree
    P = {**SELECT[style], **(params or {})}
    f = sel["features"]
    x = g["xyz"]
    J = rest.J
    zm, ring_m = float(P.get("fill_zone_m", 0.04)), float(P.get("fill_ring_m", 0.06))
    base = (~sel["free"]) & (f["sd"] > -0.025) & (f["sd"] < 0.015) & (f["normal"][:, 2] < -0.2) & (x[:, 1] < J[J_HEAD, 1])
    d, _ = cKDTree(x[sel["free"]]).query(x, distance_upper_bound=zm + ring_m)
    zone = base & (d < zm)
    if P.get("halfwidth_m") is not None:
        zone &= np.abs(x[:, 0] - J[J_HEAD, 0]) < P["halfwidth_m"]
    ring = base & ~zone & (d >= zm) & np.isfinite(d) & ~sel["hair_colour"] & ~sel["hair"]
    idx = np.nonzero(zone)[0]
    rgb = g["rgb"][idx].copy()
    yb = np.floor(x[:, 1] / 0.02).astype(int)
    filled = 0
    for b_ in np.unique(yb[idx]):
        rz = ring & (np.abs(yb - b_) <= 1)
        if rz.sum() < 8:
            continue
        w = g["opacity"][rz] + 1e-3
        med = np.array([_wquantile(g["rgb"][rz][:, c], w, 0.5)[0] for c in range(3)])
        m = yb[idx] == b_
        rgb[m] = med
        filled += int(m.sum())
    return {"idx": idx.astype(np.int32), "rgb": rgb, "info": {"zone": int(len(idx)), "filled": filled,
                                                              "ring": int(ring.sum())}}


# ================================================================================================ (b) guide strands
def _wmean(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    return (x * w[:, None]).sum(0) / max(float(w.sum()), 1e-12)


def _wquantile(v: np.ndarray, w: np.ndarray, q) -> np.ndarray:
    o = np.argsort(v)
    cw = np.cumsum(w[o])
    cw /= cw[-1]
    return np.interp(np.atleast_1d(q), cw, v[o])


def resample_polyline(P: np.ndarray, n: int) -> np.ndarray:
    """n points evenly spaced by arc length along the polyline P (M, 3) (M >= 2)."""
    d = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(d)]
    if s[-1] <= 1e-9:
        return np.repeat(P[:1], n, 0)
    u = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(u, s, P[:, k]) for k in range(3)], 1)


def _smooth_rows(C: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0 or len(C) < 3:
        return C
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(C, sigma, axis=0, mode="nearest")


def _onto_head(p: np.ndarray, hc: dict, lift: float = 0.004) -> np.ndarray:
    """The point of the head capsule surface (+ lift) closest to p."""
    cp, _ = segment_closest(p, hc["a"], hc["b"])
    d = p - cp
    n = d / max(float(np.linalg.norm(d)), 1e-9)
    return cp + n * (hc["r"] + lift)


def build_strands(g: dict, sel: dict, rest: RestBody, style: str = "ponytail", params: dict | None = None) -> dict:
    """Guide strands through the FREE hair (rest shape from the canonical Gaussians, opacity weighted).

    ponytail: the tail centreline = weighted centroids of horizontal slabs (slab_m) from the top of the free hair down
      to its end, smoothed; the ROOT (tie) = the top of the centreline pulled onto the head capsule surface; n_strands
      = 1 core + (n-1) around it: per slab the members are split by their angle about the centreline (a ring of n-1
      sectors) and radius (inner half-RMS = core); each strand = its members' centroid per slab (gaps filled with its
      mean offset), from the shared root to its last slab with hair.
    loose: n_strands sectors of equal hair weight by azimuth about the head-sphere centre (vertical axis); per sector
      the slab centroids from its top to its end; root = the top pulled onto the head capsule surface.
    All strands resampled to `points` points by arc length. -> {"rest": (S, N, 3), "length": (S,), "seg": (S,),
    "style", "info"}"""
    P = {**SELECT[style], **(params or {})}
    hc = head_capsule(rest)
    m = sel["free"]
    x = g["xyz"][m]
    w = g["opacity"][m] + 0.02
    slab = float(P.get("slab_m", 0.02))
    npts = int(P["points"])
    nS = int(P["n_strands"])
    wmin = float(P.get("slab_min_weight", 0.5))
    lines = []
    info = {"style": style, "free_gaussians": int(m.sum())}
    if style == "ponytail":
        y_top, y_bot = _wquantile(x[:, 1], w, [0.995, 0.003])
        edges = np.arange(y_top, y_bot, -slab)
        C, members = [], []
        for e in edges:
            mm = (x[:, 1] <= e) & (x[:, 1] > e - slab)
            if w[mm].sum() < wmin:
                continue
            C.append(_wmean(x[mm], w[mm]))
            members.append(np.nonzero(mm)[0])
        C = _smooth_rows(np.asarray(C), float(P.get("smooth_slabs", 1.0)))
        root = _onto_head(C[0], hc)
        T = np.gradient(C, axis=0)
        T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-9)
        nring = nS - 1
        pts = np.full((nS, len(C), 3), np.nan)
        rms_j = np.zeros(len(C))
        for j, mm in enumerate(members):
            e1 = np.array([1.0, 0.0, 0.0]) - T[j] * T[j, 0]
            e1 /= max(float(np.linalg.norm(e1)), 1e-9)
            e2 = np.cross(T[j], e1)
            o = x[mm] - C[j]
            a1, a2 = o @ e1, o @ e2
            r = np.hypot(a1, a2)
            rms = math.sqrt(float((w[mm] * r * r).sum() / w[mm].sum()))
            rms_j[j] = rms
            inner = r < 0.5 * rms
            sec = np.floor((np.arctan2(a2, a1) + math.pi) / (2 * math.pi / nring)).astype(int) % nring
            lab_ = np.where(inner, 0, 1 + sec)
            for k in range(nS):
                q = lab_ == k
                if w[mm][q].sum() >= 0.25 * wmin:
                    pts[k, j] = _wmean(x[mm][q], w[mm][q])
        pts[0] = np.where(np.isnan(pts[0]), C, pts[0])
        for k in range(1, nS):
            ok = ~np.isnan(pts[k, :, 0])
            if ok.sum() == 0:
                pts[k] = C
                continue
            last = int(np.nonzero(ok)[0].max())
            off = pts[k] - C
            mean_off = np.nanmean(off, axis=0)
            off = np.where(np.isnan(off), mean_off, off)
            n_ = np.linalg.norm(off, axis=1, keepdims=True)          # never farther out than 1.5 x the slab RMS radius
            off *= np.minimum(1.0, 1.5 * rms_j[:, None] / np.maximum(n_, 1e-9))
            off = _smooth_rows(off, 1.5)
            pts[k] = C + off
            pts[k, last + 1:] = np.nan
        for k in range(nS):
            pk = pts[k][~np.isnan(pts[k, :, 0])]
            lines.append(np.vstack([root, pk]))
        info.update({"tie": np.round(root, 4).tolist(), "slabs": int(len(C)), "y_top": round(float(y_top), 4),
                     "y_bottom": round(float(y_bot), 4)})
    else:
        c = hc["centre"]
        phi = np.arctan2(x[:, 0] - c[0], x[:, 2] - c[2])
        srt = np.sort(phi)
        gaps = np.diff(np.r_[srt, srt[0] + 2 * math.pi])
        cut = srt[int(np.argmax(gaps))] + 0.5 * float(gaps.max())         # split the circle at the largest gap
        u = np.mod(phi - cut, 2 * math.pi)
        qs = _wquantile(u, w, np.linspace(0, 1, nS + 1))
        for k in range(nS):
            mm = (u >= qs[k]) & (u <= qs[k + 1])
            if w[mm].sum() < 2 * wmin:
                continue
            xs, ws = x[mm], w[mm]
            y_top, y_bot = _wquantile(xs[:, 1], ws, [0.99, 0.005])
            Ck = []
            for e in np.arange(y_top, y_bot, -slab):
                q = (xs[:, 1] <= e) & (xs[:, 1] > e - slab)
                if ws[q].sum() >= 0.25 * wmin:
                    Ck.append(_wmean(xs[q], ws[q]))
            if len(Ck) < 2:
                continue
            Ck = _smooth_rows(np.asarray(Ck), float(P.get("smooth_slabs", 1.0)))
            lines.append(np.vstack([_onto_head(Ck[0], hc), Ck]))
        info["sectors_used"] = len(lines)
    rest_pts = np.stack([resample_polyline(L, npts) for L in lines])
    seg = np.linalg.norm(np.diff(rest_pts, axis=1), axis=2)
    nL = len(rest_pts)
    if style == "ponytail":                  # core <-> every ring strand, ring neighbours
        links = [(0, k) for k in range(1, nL)] + [(k, k % (nL - 1) + 1) for k in range(1, nL)] if nL > 2 else [(0, 1)]
    else:                                    # neighbouring sectors of the curtain
        links = [(k, k + 1) for k in range(nL - 1)]
    links = np.asarray(links, int).reshape(-1, 2)
    info.update({"strands": int(len(rest_pts)), "points": npts,
                 "length_m": [round(float(v), 4) for v in seg.sum(1)],
                 "segment_m": [round(float(v), 4) for v in seg.mean(1)]})
    info["links"] = int(len(links))
    return {"rest": rest_pts, "length": seg.sum(1), "seg": seg.mean(1), "style": style, "info": info, "links": links}


# ================================================================================================ (d) binding + posing
HEAD_FORWARD = np.array([0.0, 0.0, 1.0])


def segment_frames(pts: np.ndarray, forward=HEAD_FORWARD) -> np.ndarray:
    """(S, N, 3) strand points -> (S, N-1, 3, 3) segment frames, columns X, Y, Z: Y = unit tangent to the next point,
    Z = `forward` made orthogonal to Y (world up when they are parallel), X = Y x Z (docs/SPLAT_AVATAR_FORMAT.md 6)."""
    Y = np.diff(pts, axis=-2)
    Y /= np.maximum(np.linalg.norm(Y, axis=-1, keepdims=True), 1e-12)
    f = np.broadcast_to(np.asarray(forward, float), Y.shape)
    Z = f - (f * Y).sum(-1, keepdims=True) * Y
    bad = np.linalg.norm(Z, axis=-1) < 1e-6
    if np.any(bad):
        up = np.array([0.0, 1.0, 0.0])
        Zb = up - (Y[bad] @ up)[:, None] * Y[bad]
        Z = Z.copy()
        Z[bad] = Zb
    Z /= np.maximum(np.linalg.norm(Z, axis=-1, keepdims=True), 1e-12)
    X = np.cross(Y, Z)
    return np.stack([X, Y, Z], -1)


def bind(xyz: np.ndarray, quat: np.ndarray, strands_rest: np.ndarray, chunk: int = 4096) -> dict:
    """Every Gaussian (rest centre xyz, rotation quat w x y z) to its nearest rest strand segment: strand, segment,
    local_offset (segment frame), local_rotation (w x y z), distance. Ties go to the lower strand / segment index."""
    S, N, _ = strands_rest.shape
    A = strands_rest[:, :-1].reshape(-1, 3)
    B = strands_rest[:, 1:].reshape(-1, 3)
    best = np.empty(len(xyz), np.int64)
    dist = np.empty(len(xyz))
    for i0 in range(0, len(xyz), chunk):
        p = xyz[i0:i0 + chunk, None, :]
        cp, _ = segment_closest(p, A[None], B[None])
        d = np.linalg.norm(p - cp, axis=-1)
        best[i0:i0 + chunk] = d.argmin(1)
        dist[i0:i0 + chunk] = d.min(1)
    strand, seg = best // (N - 1), best % (N - 1)
    F = segment_frames(strands_rest)[strand, seg]
    off = np.einsum("nji,nj->ni", F, xyz - strands_rest[strand, seg])
    lrot = quat_mul(mat_to_quat(np.transpose(F, (0, 2, 1))), quat)
    return {"strand": strand.astype(np.int32), "segment": seg.astype(np.int32), "local_offset": off,
            "local_rotation": lrot, "distance": dist}


def pose_bound(track: np.ndarray, head_R: np.ndarray, strands_rest: np.ndarray, b: dict) -> tuple[np.ndarray, np.ndarray]:
    """Posed centres / rotations (w x y z) of bound Gaussians for ONE frame: track (S, N, 3) simulated world strand
    points, head_R (3, 3) the head's world rotation (canonical -> posed). Frame = rest frame carried by the head, then
    the minimal rotation from the carried rest tangent to the simulated tangent."""
    s, k = b["strand"], b["segment"]
    Fr = segment_frames(strands_rest)[s, k]
    p0 = track[s, k]
    t_sim = track[s, k + 1] - p0
    t_car = Fr[:, :, 1] @ head_R.T
    Rm = min_rotation(t_car, t_sim)
    Rf = Rm @ head_R
    mean = p0 + np.einsum("nij,nj->ni", Rf @ Fr, b["local_offset"])
    q = quat_mul(mat_to_quat(Rf @ Fr), b["local_rotation"])
    return mean, q


def draw_strands(img: np.ndarray, pts: np.ndarray, R, t, K, colour=(255, 40, 40), thick: int = 2) -> np.ndarray:
    """Strand polylines (S, N, 3) projected onto an RGB image (in place); root dot in yellow."""
    import cv2
    X = pts.reshape(-1, 3) @ np.asarray(R).T + np.asarray(t).reshape(3)
    uv = (X[:, :2] / np.maximum(X[:, 2:], 1e-6)) * np.array([K[0][0], K[1][1]]) + np.array([K[0][2], K[1][2]])
    uv = uv.reshape(pts.shape[0], pts.shape[1], 2)
    for s in range(len(uv)):
        cv2.polylines(img, [np.round(uv[s]).astype(np.int32)], False, colour, thick, cv2.LINE_AA)
        cv2.circle(img, tuple(int(v) for v in np.round(uv[s, 0])), 3, (255, 220, 0), -1)
    return img


# ================================================================================================ (c) simulation
def body_capsules(rest: RestBody, arms: bool = True) -> list[dict]:
    """hair_groom.colliders of this rest body (head capsule fitted to the scalp, neck, shoulder bar, chest, mid-back,
    upper arms / forearms): {name, a, b (joint ids), oa, ob (offsets from the joints, rest axes), r, first_point}."""
    from .. import hair_groom as HG
    sp = HG.scalp(rest)
    out = []
    for c in HG.colliders(rest, sp):
        if not arms and c["name"].startswith(("upperarm", "forearm")):
            continue
        out.append({"name": c["name"], "a": int(c["a"]), "b": int(c["b"]), "oa": HG.to_unity(c["oa_unity"]),
                    "ob": HG.to_unity(c["ob_unity"]), "r": float(c["r"]), "first_point": int(c.get("first_point", 2)),
                    "max_push_m": c.get("max_push_m")})
    return out


def capsule_track(caps: list[dict], Rg: np.ndarray, Pj: np.ndarray) -> np.ndarray:
    """(F, C, 2, 3) world end points of the capsules for FK frames (Rg (F,22,3,3), Pj (F,22,3)): end = Rg_j o + P_j."""
    E = np.zeros((len(Rg), len(caps), 2, 3))
    for i, c in enumerate(caps):
        E[:, i, 0] = np.einsum("fij,j->fi", Rg[:, c["a"]], c["oa"]) + Pj[:, c["a"]]
        E[:, i, 1] = np.einsum("fij,j->fi", Rg[:, c["b"]], c["ob"]) + Pj[:, c["b"]]
    return E


def rest_safe_radii(strands_rest: np.ndarray, caps: list[dict], rest: RestBody, pr: float, margin: float = 0.003,
                    floor_frac: float = 0.6) -> tuple[np.ndarray, list]:
    """Capsule radii the REST strands do not violate: r_eff = min(r, d_rest - pr - margin) for the points the capsule
    acts on, never below floor_frac x r (LHM++'s hair rests ON the body; a fat SMPL-X capsule would push it off)."""
    r_eff, notes = [], []
    for c in caps:
        a = rest.J[c["a"]] + c["oa"]
        b = rest.J[c["b"]] + c["ob"]
        p = strands_rest[:, c["first_point"]:].reshape(-1, 3)
        if c["name"] == "head":
            p = strands_rest[:, 2:].reshape(-1, 3)
        cp, _ = segment_closest(p, a[None], b[None])
        d = float(np.linalg.norm(p - cp, axis=1).min()) if len(p) else 1.0
        r = min(c["r"], max(floor_frac * c["r"], d - pr - margin))
        r_eff.append(r)
        if r < c["r"] - 1e-6:
            notes.append({"capsule": c["name"], "r": round(c["r"], 4), "r_eff": round(r, 4), "rest_min_dist": round(d, 4)})
    return np.asarray(r_eff), notes


def _slerp_mats(R0: np.ndarray, R1: np.ndarray, a: float) -> np.ndarray:
    """Rotation a of the way from R0 to R1 (3x3)."""
    D = R0.T @ R1
    c = np.clip((np.trace(D) - 1) / 2, -1.0, 1.0)
    th = math.acos(c)
    if th < 1e-9:
        return R0.copy()
    w = np.array([D[2, 1] - D[1, 2], D[0, 2] - D[2, 0], D[1, 0] - D[0, 1]]) / (2 * math.sin(th))
    return R0 @ rotvec_to_mat(w * th * a)


def simulate(strands_rest: np.ndarray, head_J: np.ndarray, head_R: np.ndarray, head_P: np.ndarray,
             caps_E: np.ndarray, caps_r: np.ndarray, caps_first: np.ndarray, style: str = "ponytail",
             params: dict | None = None, fps: float = 30.0, links: np.ndarray | None = None) -> dict:
    """Position-based dynamics of guide strands driven by the head, F frames at fps.

    strands_rest (S, N, 3) canonical rest points; the head maps canonical -> world as X = head_R[f] (X - head_J) +
    head_P[f]. caps_E (F, C, 2, 3) capsule end points per frame (self + partner), caps_r (C,), caps_first (C,) first
    point index each capsule acts on. Per frame `substeps` substeps (head / capsules interpolated: slerp / lerp):
      pin points 0, 1 to the head-carried rest; Verlet with damping (keep = 1 - damping dt) and gravity; `iterations` x
      [shape matching toward the carried rest k_i = 1 - (1 - k(s))^(1 / substeps x iterations) per pass, k(s) =
      shape_root exp(-s / shape_falloff_m) per substep; bending = distance i-2 .. i toward its rest (fraction `bend`);
      stretch i-1 .. i (Gauss-Seidel root to tip, the pinned points immovable)]; capsule push-out to r + particle radius
      with friction on the tangential motion; follow-the-leader length clamp (max_stretch).
    links (L, 2) strand pairs: bundle cohesion, per iteration the corresponding points (i >= 2) of linked strands are
      kept within (1 + link_slack) x their rest distance (one-sided: they may come together, not tear apart),
      stiffness link_stiffness, both points move.
    The first frame starts at the carried rest; preroll frames are the caller's (simulate from earlier, drop them).
    -> {"pos": (F, S, N, 3) world points at the end of each frame, "stats": {...}}"""
    Pm = {**SIM[style], **(params or {})}
    S, N, _ = strands_rest.shape
    F = len(head_R)
    ns, ni = int(Pm["substeps"]), int(Pm["iterations"])
    dt = 1.0 / fps / ns
    keep = max(0.0, 1.0 - Pm["damping_per_s"] * dt)
    g_dt2 = np.array([0.0, -Pm["gravity"], 0.0]) * dt * dt
    loc = strands_rest - head_J                       # head-relative rest
    seg = np.linalg.norm(np.diff(strands_rest, axis=1), axis=2)            # (S, N-1)
    bend_rest = np.linalg.norm(strands_rest[:, 2:] - strands_rest[:, :-2], axis=2)  # (S, N-2)
    s_arc = np.concatenate([np.zeros((S, 1)), np.cumsum(seg, 1)], 1)       # (S, N)
    k_sub = np.clip(Pm["shape_root"] * np.exp(-s_arc / Pm["shape_falloff_m"]) + Pm.get("shape_floor", 0.0), 0, 0.999)
    k_it = 1.0 - (1.0 - k_sub) ** (1.0 / ni)
    k_it[:, :2] = 0.0
    pr = Pm["particle_radius_m"]
    fric = Pm["friction"]
    bend = Pm["bend"]
    maxs = Pm["max_stretch"]
    cap_push = Pm.get("max_push_m")
    carried = lambda R, T: np.einsum("ij,snj->sni", R, loc) + T  # noqa: E731
    if links is not None and len(links):
        la, lb = np.asarray(links, int).T
        lrest = np.linalg.norm(strands_rest[la] - strands_rest[lb], axis=-1)[:, 2:]           # (L, N-2)
        lmax = lrest * (1.0 + Pm.get("link_slack", 0.25)) + 0.003
        lk = float(Pm.get("link_stiffness", 0.5))
    else:
        la = None
    pos = carried(head_R[0], head_P[0])
    prev = pos.copy()
    out = np.zeros((F, S, N, 3))
    out[0] = pos
    contacts = 0
    max_push = 0.0
    for f in range(1, F):
        for sub in range(ns):
            a = (sub + 1) / ns
            R = _slerp_mats(head_R[f - 1], head_R[f], a)
            T = (1 - a) * head_P[f - 1] + a * head_P[f]
            E = (1 - a) * caps_E[f - 1] + a * caps_E[f]
            tgt = carried(R, T)
            prev[:, :2] = pos[:, :2]
            pos[:, :2] = tgt[:, :2]
            v = (pos[:, 2:] - prev[:, 2:]) * keep
            prev[:, 2:] = pos[:, 2:]
            pos[:, 2:] = pos[:, 2:] + v + g_dt2
            for _ in range(ni):
                pos[:, 2:] += (tgt[:, 2:] - pos[:, 2:]) * k_it[:, 2:, None]
                for i in range(2, N):                    # bending: distance i-2 .. i, two-sided (pinned points fixed)
                    d = pos[:, i] - pos[:, i - 2]
                    ln = np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
                    c = bend * (ln - bend_rest[:, i - 2, None]) * d / ln
                    if i - 2 < 2:
                        pos[:, i] -= c
                    else:
                        pos[:, i - 2] += 0.5 * c
                        pos[:, i] -= 0.5 * c
                for i in range(2, N):
                    d = pos[:, i] - pos[:, i - 1]
                    ln = np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
                    c = (ln - seg[:, i - 1, None]) * d / ln
                    if i - 1 < 2:
                        pos[:, i] -= c
                    else:
                        pos[:, i - 1] += 0.5 * c
                        pos[:, i] -= 0.5 * c
                if la is not None:                      # bundle cohesion (one-sided max distance between strands)
                    d = pos[lb, 2:] - pos[la, 2:]
                    ln = np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-9)
                    ex = np.maximum(ln - lmax[..., None], 0.0)
                    c = 0.5 * lk * ex * d / ln
                    np.add.at(pos, (la, slice(2, None)), c)
                    np.add.at(pos, (lb, slice(2, None)), -c)
            # collisions (capsules), friction on the tangential motion
            for ci in range(len(caps_r)):
                i0 = int(max(2, caps_first[ci]))
                A, B = E[ci, 0], E[ci, 1]
                p = pos[:, i0:]
                cp, _ = segment_closest(p, A, B)
                d = p - cp
                ln = np.linalg.norm(d, axis=-1, keepdims=True)
                rr = caps_r[ci] + pr
                hit = (ln[..., 0] < rr)
                if not hit.any():
                    continue
                contacts += int(hit.sum())
                n = d / np.maximum(ln, 1e-9)
                depth_ = np.minimum(rr - ln, cap_push) if cap_push else rr - ln     # capped push-out per substep
                push = depth_ * n
                max_push = max(max_push, float(np.abs(rr - ln[hit]).max()))
                p2 = p + np.where(hit[..., None], push, 0.0)
                vel = p2 - prev[:, i0:]
                vn = (vel * n).sum(-1, keepdims=True) * n
                vt = vel - vn
                prev[:, i0:] = np.where(hit[..., None], p2 - (vn * 0.0 + vt * (1 - fric)), prev[:, i0:])
                pos[:, i0:] = p2
            for i in range(2, N):                        # follow-the-leader: every segment within [1/maxs, maxs] x rest
                d = pos[:, i] - pos[:, i - 1]
                ln = np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
                tgt_l = np.clip(ln, seg[:, i - 1, None] / maxs, seg[:, i - 1, None] * maxs)
                pos[:, i] = pos[:, i - 1] + d * tgt_l / ln
        out[f] = pos
    return {"pos": out, "stats": {"contacts": contacts, "max_push_m": round(max_push, 4), "frames": F,
                                  "substeps": ns, "iterations": ni, "params": Pm}}


def check_sim(pos: np.ndarray, strands_rest: np.ndarray, head_J, head_R, head_P, caps_E, caps_r, caps_first,
              names: list[str], fps: float = 30.0, pr: float = 0.0) -> dict:
    """(e) per-frame checks of a simulated track: segment stretch (max |l / l_rest - 1|), the deepest point inside a
    capsule (by name; head / body = self, partner_* = the partner), the largest point speed (m/s: explosions), and the
    swing = tip distance from the head-carried rest tip (m: the motion a rigid tail lacks)."""
    seg0 = np.linalg.norm(np.diff(strands_rest, axis=1), axis=2)
    seg = np.linalg.norm(np.diff(pos, axis=2), axis=3)
    stretch = np.abs(seg / seg0[None] - 1).max(axis=(1, 2))
    depth = {}
    for ci, nm in enumerate(names):
        i0 = int(max(2, caps_first[ci]))
        p = pos[:, :, i0:]
        cp, _ = segment_closest(p, caps_E[:, ci, 0][:, None, None], caps_E[:, ci, 1][:, None, None])
        dd = caps_r[ci] - np.linalg.norm(p - cp, axis=-1)
        depth[nm] = dd.max(axis=(1, 2))
    speed = np.r_[0.0, np.linalg.norm(np.diff(pos, axis=0), axis=-1).max(axis=(1, 2)) * fps]
    loc = strands_rest - head_J
    carried = np.einsum("fij,snj->fsni", head_R, loc) + head_P[:, None, None]
    tip = np.linalg.norm(pos[:, :, -1] - carried[:, :, -1], axis=-1)
    worst_in = max(depth, key=lambda k: float(depth[k].max())) if depth else None
    worst = sorted(((float(v[f]), int(f), k) for k, v in depth.items() for f in np.nonzero(v > 0.003)[0]), reverse=True)
    fast = np.argsort(-speed)[:8]
    return {"max_stretch": round(float(stretch.max()), 5), "p95_stretch": round(float(np.percentile(stretch, 95)), 5),
            "max_depth_in_capsule_m": {k: round(float(v.max()), 4) for k, v in depth.items()},
            "deepest_capsule": worst_in, "frames_with_point_inside_by_3mm": int(sum(
                (v > 0.003).sum() for v in depth.values())),
            "worst_inside_frames": [{"frame": f, "capsule": k, "depth_m": round(d, 4)} for d, f, k in worst[:12]],
            "fastest_frames": [{"frame": int(f), "speed_m_s": round(float(speed[f]), 2)} for f in fast],
            "max_speed_m_s": round(float(speed.max()), 3), "p99_speed_m_s": round(float(np.percentile(speed, 99)), 3),
            "swing_tip_m": {"mean": round(float(tip.mean()), 4), "p95": round(float(np.percentile(tip, 95)), 4),
                            "max": round(float(tip.max()), 4)},
            "per_frame": {"stretch": stretch, "speed": speed, "tip": tip.mean(1), "depth": depth}}


# ================================================================================================ pipeline (local)
BUNDLE_VERSION = 1


def prepare(gauss_path: Path, betas, style: str, out_dir: Path, role: str, sel_params: dict | None = None,
            strand_params: dict | None = None, sheet: bool = True, device: str = "cpu") -> dict:
    """(a) + (b) + (d) for one avatar: <out>/<role>_hair.npz (free / rigid indices, rest strands, binding, head-core
    indices for the pod's head transform, canonical free centres for the pod's cross-check) + <role>_hair.json
    + <role>_select.jpg / <role>_strands.jpg (LOOK)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    g = load_gaussians(gauss_path)
    rest = RestBody(betas)
    sel = select_hair(g, rest, style, sel_params)
    st = build_strands(g, sel, rest, style, {**(sel_params or {}), **(strand_params or {})})
    free = np.nonzero(sel["free"])[0]
    b = bind(g["xyz"][free], g["quat"][free], st["rest"])
    f = sel["features"]
    J = rest.J
    fill = uncover_fill(g, sel, rest, style, sel_params)
    core = np.nonzero((f["dom"] == J_HEAD) & (g["xyz"][:, 1] > J[J_HEAD, 1] + 0.02) & ~sel["free"]
                      & (np.linalg.norm(g["xyz"] - J[J_HEAD], axis=1) < 0.16) & (f["dh"] < 0.03))[0]
    np.savez_compressed(out_dir / f"{role}_hair.npz", version=BUNDLE_VERSION, free=free.astype(np.int32),
                        rigid=np.nonzero(sel["rigid"])[0].astype(np.int32), strands_rest=st["rest"],
                        strand=b["strand"], segment=b["segment"], local_offset=b["local_offset"],
                        local_rotation=b["local_rotation"], bind_distance=b["distance"], head_core=core.astype(np.int32),
                        links=st["links"].astype(np.int32), style=style, fill_idx=fill["idx"],
                        fill_rgb=fill["rgb"].astype(np.float32),
                        free_xyz=g["xyz"][free], free_quat=g["quat"][free], head_joint=J[J_HEAD], betas=rest.betas)
    info = {"role": role, "style": style, "source": str(gauss_path), "count": int(len(g["xyz"])),
            "select": {k: v for k, v in sel["info"].items()}, "strands": st["info"],
            "bind_distance_mm": {q: round(float(np.percentile(b["distance"], p_)) * 1e3, 2)
                                 for q, p_ in (("p50", 50), ("p95", 95), ("max", 100))},
            "per_strand_gaussians": np.bincount(b["strand"], minlength=len(st["rest"])).tolist(),
            "head_core": int(len(core)), "uncover_fill": fill["info"], "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (out_dir / f"{role}_hair.json").write_text(json.dumps(info, indent=1, default=float))
    if sheet:
        select_sheet(g, sel, rest, out_dir / f"{role}_select.jpg", device=device)
        strands_sheet(g, sel, st["rest"], rest, out_dir / f"{role}_strands.jpg", device=device)
    log(f"{role}: free {len(free)} rigid {int(sel['rigid'].sum())} strands {len(st['rest'])} lengths "
        f"{st['info']['length_m']} bind p95 {info['bind_distance_mm']['p95']} mm")
    return info


def strands_sheet(g: dict, sel: dict, strands: np.ndarray, rest: RestBody, path: Path, device: str = "cpu") -> Path:
    """LOOK: the rest strands drawn over the avatar (top) and over the free hair alone (bottom), 5 azimuths."""
    import cv2
    J = rest.J
    ctr = np.array([J[J_HEAD, 0], J[J_HEAD, 1] - 0.10, J[J_HEAD, 2]])
    W, H = 400, 520
    K = np.array([[1500.0, 0, W / 2], [0, 1500.0, H / 2], [0, 0, 1]])
    rows = []
    for m in (np.ones(len(g["xyz"]), bool), sel["free"]):
        cols = []
        for az in (90, 135, 180, 225, 270):
            C = ctr + 1.5 * np.array([math.sin(math.radians(az)), 0.1, math.cos(math.radians(az))])
            R, t = look_at(C, ctr)
            im = render_check(g["xyz"][m], g["quat"][m], g["scale"][m], g["opacity"][m], g["rgb"][m], R, t, K, W, H,
                              device=device)
            cols.append(draw_strands(np.ascontiguousarray(im), strands, R, t, K))
        rows.append(np.concatenate(cols, 1))
    cv2.imwrite(str(path), np.concatenate(rows, 0)[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])
    return Path(path)


def load_bundle(path: Path) -> dict:
    with np.load(str(path)) as z:
        return {k: z[k] for k in z.files}


def simulate_role(bundle: dict, Z: dict, role: str, keys, partner: str | None = None, style: str = "ponytail",
                  sim_params: dict | None = None, preroll: int | None = None, fps: float = 30.0) -> dict:
    """(c) + (e): simulate one avatar's strands over consecutive 30 Hz pose keys `keys` of an LHM++ render input
    (poses.npz dict Z), with `preroll` earlier keys simulated first and dropped. Self capsules = body_capsules of the
    avatar's rest body (radii made rest-safe), partner capsules (head, neck, shoulders, chest, mid-back, arms) from the
    partner's FK. -> {"keys", "tracks_local" (F, S, N, 3) head-local canonical, "tracks_world", "head_R", "head_P",
    "check", "stats", "capsules"}"""
    Pm = {**SIM[style], **(sim_params or {})}
    keys = np.asarray(keys, int)
    pre = int(Pm["preroll_frames"] if preroll is None else preroll)
    allk = np.r_[np.arange(keys[0] - pre, keys[0]), keys]
    assert np.all(np.diff(allk) == 1), "simulate_role: keys must be consecutive"
    rest = RestBody(bundle["betas"])
    Rg, Pj = fk_keys(rest, Z, role, allk)
    strands = bundle["strands_rest"]
    caps = body_capsules(rest)
    r_eff, notes = rest_safe_radii(strands, caps, rest, Pm["particle_radius_m"])
    first = [3 if c["name"].startswith(("upper", "fore")) else 2 for c in caps]
    names = [c["name"] for c in caps]
    E = capsule_track(caps, Rg, Pj)
    rr = list(r_eff)
    if partner:
        rp = RestBody(Z[f"{partner}_betas"])
        Rp, Pp = fk_keys(rp, Z, partner, allk)
        pc = [c for c in body_capsules(rp) if c["name"] in ("head", "neck", "shoulders", "chest", "midback", "upperarm_l",
                                                             "upperarm_r", "forearm_l", "forearm_r")]
        E = np.concatenate([E, capsule_track(pc, Rp, Pp)], 1)
        rr += [c["r"] for c in pc]
        first += [2] * len(pc)
        names += ["partner_" + c["name"] for c in pc]
    rr, first = np.asarray(rr), np.asarray(first)
    t0 = time.time()
    res = simulate(strands, rest.J[J_HEAD], Rg[:, J_HEAD], Pj[:, J_HEAD], E, rr, first, style, Pm, fps,
                   links=bundle.get("links"))
    pos = res["pos"][pre:]
    hR, hP = Rg[pre:, J_HEAD], Pj[pre:, J_HEAD]
    chk = check_sim(pos, strands, rest.J[J_HEAD], hR, hP, E[pre:], rr, first, names, fps)
    pf = chk.pop("per_frame")
    local = np.einsum("fji,fsnj->fsni", hR, pos - hP[:, None, None]) + rest.J[J_HEAD]
    fwd = Rg[pre:, 0] @ np.array([0.0, 0.0, 1.0])
    yaw = np.degrees(np.unwrap(np.arctan2(fwd[:, 0], fwd[:, 2])))
    rate = np.abs(np.gradient(yaw)) * fps
    spin = rate > 270
    chk["swing_tip_in_spins_m"] = round(float(pf["tip"][spin].mean()), 4) if spin.any() else None
    chk["swing_tip_calm_m"] = round(float(pf["tip"][rate < 60].mean()), 4) if (rate < 60).any() else None
    chk["spin_frames(yaw_rate>270deg/s)"] = int(spin.sum())
    chk["rest_safe_radii"] = notes
    chk["sim_seconds"] = round(time.time() - t0, 1)
    return {"keys": keys, "tracks_local": local, "tracks_world": pos, "head_R": hR, "head_P": hP, "check": chk,
            "stats": res["stats"], "capsule_names": names, "capsule_r": rr, "per_frame": pf, "yaw_rate": rate}


def save_tracks(path: Path, sim: dict, role: str) -> Path:
    np.savez_compressed(str(path), version=BUNDLE_VERSION, role=role, keys=sim["keys"].astype(np.int32),
                        tracks_local=sim["tracks_local"].astype(np.float32), head_R=sim["head_R"].astype(np.float32),
                        head_P=sim["head_P"].astype(np.float32))
    return Path(path)


# ================================================================================================ pod side
class PodHair:
    """Render-pod side (numpy only; loaded by file path from remote/cand_lhmpp/lhmpp_render_pod.py --hair DIR).

    DIR/hair.json: {"roles": {role: {"bundle": "<role>_hair.npz", "tracks": "<role>_tracks.npz"}}, "groups": [...]}.
    attach(role, cano_xyz, cano_quat): the pod's canonical Gaussians (its own LHM++ forward) -> cross-check against
    the bundle's canonical free centres, binding offsets / rotations recomputed from the pod's canonical (segment
    kept). pose(role, key, posed_xyz, ...) -> (free indices, means, quats w x y z) or None when the key has no track:
    the head transform of the frame = Kabsch of the head-core Gaussians (canonical -> LHM++ posed), the head-local
    track mapped with it, the free Gaussians posed on their segments (pose_bound)."""

    def __init__(self, hair_dir):
        self.dir = Path(hair_dir)
        self.cfg = json.loads((self.dir / "hair.json").read_text())
        self.groups = set(self.cfg.get("groups", []))
        self.R = {}
        self.report = {"roles": {}, "groups": sorted(self.groups)}

    def attach(self, role: str, cano_xyz: np.ndarray, cano_quat: np.ndarray) -> dict | None:
        rc = self.cfg.get("roles", {}).get(role)
        if not rc:
            return None
        b = load_bundle(self.dir / rc["bundle"])
        tr = load_bundle(self.dir / rc["tracks"])
        free = b["free"].astype(np.int64)
        x, q = np.asarray(cano_xyz, float), np.asarray(cano_quat, float)
        dx = np.linalg.norm(x[free] - b["free_xyz"], axis=1)
        st = b["strands_rest"]
        F = segment_frames(st)[b["strand"], b["segment"]]
        off = np.einsum("nji,nj->ni", F, x[free] - st[b["strand"], b["segment"]])
        lrot = quat_mul(mat_to_quat(np.transpose(F, (0, 2, 1))), q[free] / np.linalg.norm(q[free], axis=1, keepdims=True))
        core = b["head_core"].astype(np.int64)
        self.R[role] = {"free": free, "strands_rest": st, "frames_rest": F,
                        "bind": {"strand": b["strand"].astype(np.int64), "segment": b["segment"].astype(np.int64),
                                 "local_offset": off, "local_rotation": lrot},
                        "core": core, "core_cano": x[core], "keys": _track_key_map(tr),
                        "tracks": tr["tracks_local"].astype(np.float64), "fk_R": tr["head_R"], "fk_P": tr["head_P"],
                        "head_J": b["head_joint"], "dev": []}
        if "fill_idx" in b and len(b["fill_idx"]) and self.cfg.get("uncover_fill", False):   # experimental, off
            self.R[role]["fill"] = (b["fill_idx"].astype(np.int64), b["fill_rgb"].astype(np.float32))
        rep = {"free": int(len(free)), "strands": int(st.shape[0]), "points": int(st.shape[1]),
               "uncover_fill": int(len(self.R[role]["fill"][0])) if "fill" in self.R[role] else 0,
               "track_keys": int(len(tr["keys"])), "cano_vs_bundle_mm": {"median": round(float(np.median(dx)) * 1e3, 4),
                                                                          "max": round(float(dx.max()) * 1e3, 4)},
               "head_core": int(len(core))}
        self.report["roles"][role] = rep
        return rep

    def has(self, role: str, key: int) -> bool:
        return role in self.R and int(key) in self.R[role]["keys"]

    def core_indices(self, role: str) -> np.ndarray:
        return self.R[role]["core"]

    def fill(self, role: str):
        """(indices, rgb 0..1) recolour of the body surface the free hair hid (uncover_fill), or None."""
        return self.R.get(role, {}).get("fill")

    def pose(self, role: str, key: int, posed_core: np.ndarray):
        """posed_core: LHM++'s posed centres of core_indices(role) at this key."""
        r = self.R.get(role)
        if r is None or int(key) not in r["keys"]:
            return None
        i, w = r["keys"][int(key)]
        A = r["core_cano"]
        B = np.asarray(posed_core, float)
        ca, cb = A.mean(0), B.mean(0)
        U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
        D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
        Rk = Vt.T @ D @ U.T
        tk = cb - Rk @ ca
        res = np.linalg.norm((A - ca) @ Rk.T + cb - B, axis=1)
        # vs our FK head transform of that key: angle (deg), offset of the head joint (mm), core residual (mm)
        Rf, Pf = r["fk_R"][i].astype(float), r["fk_P"][i].astype(float)
        if w == 0.0:     # a track key (an alias between two track keys is not compared with the FK of either)
            ang = math.degrees(math.acos(float(np.clip((np.trace(Rk.T @ Rf) - 1) / 2, -1, 1))))
            r["dev"].append((ang, float(np.linalg.norm(Rk @ r["head_J"] + tk - Pf)) * 1e3, float(np.median(res)) * 1e3))
        tl = r["tracks"][i] if w == 0.0 else (1 - w) * r["tracks"][i] + w * r["tracks"][min(i + 1, len(r["tracks"]) - 1)]
        track = tl @ Rk.T + tk
        mean, q = pose_bound_cached(track, Rk, r["strands_rest"], r["frames_rest"], r["bind"])
        return r["free"], mean, q

    def summary(self) -> dict:
        for role, r in self.R.items():
            if r["dev"]:
                d = np.asarray(r["dev"])
                self.report["roles"][role]["kabsch_vs_fk"] = {
                    "frames": int(len(d)), "angle_deg_max": round(float(d[:, 0].max()), 4),
                    "head_joint_mm_max": round(float(d[:, 1].max()), 3), "core_resid_mm_median": round(float(np.median(d[:, 2])), 3)}
        return self.report


def _track_key_map(tr: dict) -> dict:
    """pose key -> (track row, weight of the next row): the track keys themselves (weight 0) + optional aliases
    (alias_keys / alias_row / alias_w: pose keys between two track keys, e.g. video frames at their pts)."""
    m = {int(k): (i, 0.0) for i, k in enumerate(tr["keys"])}
    if "alias_keys" in tr:
        for k, i, w in zip(tr["alias_keys"], tr["alias_row"], tr["alias_w"]):
            m.setdefault(int(k), (int(i), float(w)))
    return m


def render_tracks(hair_dir: Path, input_dir: Path, role: str, partner: str | None, groups: list,
                  style: str | None = None, sim_params: dict | None = None) -> dict:
    """Tracks for an LHM++ render input (poses.npz + views.npz of jl_realism_render / lhmpp_render_pod): the strands are
    simulated on the input's longest run of consecutive 30 Hz pose keys covering the times of every key the listed
    groups use (preroll before it when the run allows), and every other key of those groups (e.g. the video frames at
    their pts) becomes an ALIAS = linear blend of the two track rows around its time (head-local coordinates).
    Writes <hair_dir>/<role>_tracks.npz, <role>_tracks_check.json and hair.json (groups = the listed ones)."""
    hair_dir, input_dir = Path(hair_dir), Path(input_dir)
    with np.load(input_dir / "poses.npz") as z:
        Z = {k: np.asarray(z[k]) for k in z.files}
    with np.load(input_dir / "views.npz") as z:
        need = np.unique(np.concatenate([np.asarray(z[f"{g}_key"], int) for g in groups]))
    b = load_bundle(hair_dir / f"{role}_hair.npz")
    style = style or str(b.get("style", "ponytail"))
    T = np.asarray(Z["t"], float)
    d = np.diff(T)
    brk = np.nonzero(np.abs(d - 1 / 30.0) > 2e-3)[0]
    runs = np.split(np.arange(len(T)), brk + 1)
    t_lo, t_hi = T[need].min(), T[need].max()
    cover = [r_ for r_ in runs if len(r_) > 1 and T[r_[0]] <= t_lo + 1e-6 and T[r_[-1]] >= t_hi - 1e-6]
    if not cover:
        raise ValueError("render_tracks: no consecutive 30 Hz run of pose keys covers the groups' times")
    run = max(cover, key=len)
    pre = int({**SIM[style], **(sim_params or {})}["preroll_frames"])
    i0 = max(0, int(np.searchsorted(T[run], t_lo, side="right")) - 1)
    i1 = min(len(run) - 1, max(int(np.searchsorted(T[run], t_hi, side="left")), i0 + 1))
    use_pre = min(pre, i0)
    keys = run[i0:i1 + 1]
    sim = simulate_role(b, Z, role, keys, partner, style, sim_params, preroll=use_pre)
    tk = set(int(k) for k in keys)
    al = [int(k) for k in need if int(k) not in tk]
    rows, ws = [], []
    tt = T[keys]
    for k in al:
        j = int(np.clip(np.searchsorted(tt, T[k], side="right") - 1, 0, len(keys) - 1))
        jn = min(j + 1, len(keys) - 1)
        w = 0.0 if jn == j else float(np.clip((T[k] - tt[j]) / (tt[jn] - tt[j]), 0, 1))
        rows.append(j)
        ws.append(w)
    np.savez_compressed(hair_dir / f"{role}_tracks.npz", version=BUNDLE_VERSION, role=role, keys=keys.astype(np.int32),
                        tracks_local=sim["tracks_local"].astype(np.float32), head_R=sim["head_R"].astype(np.float32),
                        head_P=sim["head_P"].astype(np.float32), alias_keys=np.asarray(al, np.int32),
                        alias_row=np.asarray(rows, np.int32), alias_w=np.asarray(ws, np.float32))
    chk = dict(sim["check"])
    chk.update({"track_keys": [int(keys[0]), int(keys[-1])], "preroll": use_pre, "aliases": len(al), "groups": groups})
    (hair_dir / f"{role}_tracks_check.json").write_text(json.dumps(chk, indent=1, default=float))
    hj = hair_dir / "hair.json"
    cfg = json.loads(hj.read_text()) if hj.exists() else {"version": BUNDLE_VERSION, "roles": {}}
    cfg["roles"][role] = {"bundle": f"{role}_hair.npz", "tracks": f"{role}_tracks.npz"}
    cfg["groups"] = sorted(set(cfg.get("groups", [])) | set(groups))
    hj.write_text(json.dumps(cfg, indent=1))
    log(f"render tracks {role}: {len(keys)} track keys + {len(al)} aliases")
    return chk


def pose_bound_cached(track: np.ndarray, head_R: np.ndarray, strands_rest: np.ndarray, frames_rest: np.ndarray,
                      b: dict) -> tuple[np.ndarray, np.ndarray]:
    """pose_bound with the bound Gaussians' rest segment frames precomputed (frames_rest (n, 3, 3))."""
    s, k = b["strand"], b["segment"]
    p0 = track[s, k]
    t_sim = track[s, k + 1] - p0
    t_car = frames_rest[:, :, 1] @ head_R.T
    Rf = min_rotation(t_car, t_sim) @ head_R
    Fp = Rf @ frames_rest
    mean = p0 + np.einsum("nij,nj->ni", Fp, b["local_offset"])
    q = quat_mul(mat_to_quat(Fp), b["local_rotation"])
    return mean, q


# ================================================================================================ john_lula proof
def _jl_paths() -> dict:
    from ..take import PACKAGE_ROOT
    work = PACKAGE_ROOT / "work" / "john_lula"
    hair = work / "realism" / "v2" / "hair"
    return {"work": work, "src_input": work / "realism" / "lhmpp" / "input",
            "cano": work / "realism" / "lhmpp" / "pod_out" / "gs", "hair": hair, "bundle": hair / "bundle",
            "input": hair / "input", "pod_out": hair / "pod_out", "renders": hair / "renders", "videos": hair / "videos",
            "sheets": hair / "sheets", "job": PACKAGE_ROOT / "remote" / "cand_lhmpp" / "pod_job_jl_hair.json",
            "splatav": PACKAGE_ROOT.parent / "head-movement" / "Assets" / "StreamingAssets" / "08_JohnLula" / "splats"}


JL_WINDOW = (9.0, 20.0)
JL_CLOSE = {"W": 1080, "H": 1080, "dist_m": 1.5, "frame_h_m": 1.1, "head_sigma_s": 0.25, "dir_sigma_s": 2.0,
            "look_below_head_m": 0.14, "cam_below_head_m": 0.05}
JL_GROUPS = ("orbit_rigid", "orbit_swing", "close_rigid", "close_swing")


def _close_cams(Z: dict, keys: np.ndarray, rest_f: RestBody) -> dict:
    """Close-up of the follow: camera 1.5 m from her (smoothed) head on the side AWAY from the lead (the lead's pelvis
    -> hers, smoothed: he stays behind her), a little below head height, looking at her neck; one focal length."""
    from scipy.ndimage import gaussian_filter1d
    Rg, Pj = fk_keys(rest_f, Z, "follow", keys)
    head = gaussian_filter1d(Pj[:, J_HEAD], JL_CLOSE["head_sigma_s"] * 30, axis=0, mode="nearest")
    d = np.asarray(Z["follow_pelvis"][keys], float) - np.asarray(Z["lead_pelvis"][keys], float)
    d[:, 1] = 0
    d = gaussian_filter1d(d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9), JL_CLOSE["dir_sigma_s"] * 30,
                          axis=0, mode="nearest")
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    C = head + JL_CLOSE["dist_m"] * d - np.array([0.0, JL_CLOSE["cam_below_head_m"], 0.0])
    tgt = head - np.array([0.0, JL_CLOSE["look_below_head_m"], 0.0])
    R = np.stack([look_at(C[i], tgt[i])[0] for i in range(len(C))])
    tv = -np.einsum("nij,nj->ni", R, C)
    W, H = JL_CLOSE["W"], JL_CLOSE["H"]
    f = H * JL_CLOSE["dist_m"] / JL_CLOSE["frame_h_m"]
    K = np.repeat(np.array([[f, 0, (W - 1) / 2], [0, f, (H - 1) / 2], [0, 0, 1.0]])[None], len(C), 0)
    return {"K": K, "R": R, "tvec": tv, "C": C}


def jl_proof_prep(device: str = "cuda") -> dict:
    """Hair bundle + simulation + checks + local preview sheets + the pod input (refs / poses byte-identical to the
    realism render input, so the pod builds the SAME avatars; views = the existing orbit cameras over 9-20 s and a
    close-up of Lula, each twice: rigid / swinging)."""
    import shutil
    from ..gpu import exclusive
    p = _jl_paths()
    for k in ("bundle", "input", "sheets"):
        p[k].mkdir(parents=True, exist_ok=True)
    with np.load(p["src_input"] / "poses.npz") as z:
        Z = {k: np.asarray(z[k]) for k in z.files}
    with np.load(p["src_input"] / "views.npz") as z:
        V0 = {k: np.asarray(z[k]) for k in z.files}
    NV = 2643
    T = np.asarray(Z["t"], float)
    k0, k1 = int(round(JL_WINDOW[0] * 30)), int(round(JL_WINDOW[1] * 30))
    keys = NV + np.arange(k0, k1 + 1)
    assert np.allclose(T[keys], np.arange(k0, k1 + 1) / 30.0)
    with exclusive("splat_hair jl-proof prep"):
        info = prepare(p["cano"] / "follow_cano.ply", Z["follow_betas"], "ponytail", p["bundle"], "follow", device=device)
    b = load_bundle(p["bundle"] / "follow_hair.npz")
    sim = simulate_role(b, Z, "follow", keys, partner="lead", style="ponytail")
    save_tracks(p["bundle"] / "follow_tracks.npz", sim, "follow")
    (p["bundle"] / "hair.json").write_text(json.dumps({
        "version": BUNDLE_VERSION, "roles": {"follow": {"bundle": "follow_hair.npz", "tracks": "follow_tracks.npz"}},
        "groups": ["orbit_swing", "close_swing"]}, indent=1))
    chk = {k: v for k, v in sim["check"].items()}
    chk["stats"] = {k: v for k, v in sim["stats"].items() if k != "params"}
    chk["params"] = sim["stats"]["params"]
    log(f"check: {json.dumps(chk, default=float)}")
    # ---- views: orbit frames of the window (the existing orbit cameras) + the close-up, rigid and swinging
    ok = (V0["orbit_key"] >= keys[0]) & (V0["orbit_key"] <= keys[-1])
    oi = np.nonzero(ok)[0]
    rest_f = RestBody(Z["follow_betas"])
    cc = _close_cams(Z, keys, rest_f)
    groups = {}
    for g_ in ("orbit_rigid", "orbit_swing"):
        groups[g_] = {"W": 1080, "H": 1080, "K": V0["orbit_K"][oi], "R": V0["orbit_R"][oi], "tvec": V0["orbit_tvec"][oi],
                      "key": V0["orbit_key"][oi], "index": np.arange(len(oi)), "t": T[V0["orbit_key"][oi]]}
    for g_ in ("close_rigid", "close_swing"):
        groups[g_] = {"W": JL_CLOSE["W"], "H": JL_CLOSE["H"], "K": cc["K"], "R": cc["R"], "tvec": cc["tvec"], "key": keys,
                      "index": np.arange(len(keys)), "t": T[keys]}
    arrs = {}
    for g_, v in groups.items():
        for k in ("K", "R", "tvec"):
            arrs[f"{g_}_{k}"] = np.asarray(v[k], np.float32)
        arrs[f"{g_}_key"] = np.asarray(v["key"], np.int32)
        arrs[f"{g_}_index"] = np.asarray(v["index"], np.int32)
        arrs[f"{g_}_t"] = np.asarray(v["t"], np.float64)
    if p["input"].exists():
        shutil.rmtree(p["input"])
    shutil.copytree(p["src_input"] / "refs", p["input"] / "refs")
    shutil.copy2(p["src_input"] / "poses.npz", p["input"] / "poses.npz")
    np.savez_compressed(p["input"] / "views.npz", **arrs)
    vj = {"take": "john_lula", "passes": [list(JL_GROUPS)], "scene_key": -1,
          "groups": {g_: {"W": int(v["W"]), "H": int(v["H"]), "n": int(len(v["key"]))} for g_, v in groups.items()},
          "window_s": list(JL_WINDOW), "orbit_index_offset": int(oi[0]), "hair": {"groups": ["orbit_swing", "close_swing"]}}
    (p["input"] / "views.json").write_text(json.dumps(vj, indent=1))
    # ---- LOOK: local preview (approximate LBS avatar of jl_splatav for the body) rigid vs swinging at spin frames
    sheet = jl_preview_sheet(Z, b, sim, cc, keys, p, device)
    rep = {"hair": info, "check": chk, "views": vj, "preview_sheet": str(sheet), "keys": [int(keys[0]), int(keys[-1])]}
    (p["hair"] / "prep.json").write_text(json.dumps(rep, indent=1, default=float))
    return rep


def jl_preview_sheet(Z, b, sim, cc, keys, p, device="cuda", n=8) -> Path:
    """Local check render of Lula (approximate 4-joint LBS splat avatar from jl_splatav for the body; free hair on the
    simulated strands vs the same Gaussians skinned rigidly) at the fastest-turning frames, close-up camera."""
    import cv2
    from ..gpu import exclusive
    av = load_gaussians(p["splatav"] / "follow.splatav")
    rest = RestBody(Z["follow_betas"])
    free = b["free"].astype(np.int64)
    st = b["strands_rest"]
    bd = {"strand": b["strand"].astype(np.int64), "segment": b["segment"].astype(np.int64),
          "local_offset": b["local_offset"], "local_rotation": b["local_rotation"]}
    Fr = segment_frames(st)[bd["strand"], bd["segment"]]
    rate = sim["yaw_rate"]
    pick = []
    for i in np.argsort(-rate):
        if all(abs(i - j) > 12 for j in pick):
            pick.append(int(i))
        if len(pick) >= n - 2:
            break
    pick = sorted(pick + [int(np.argmin(rate[:60])), int(len(keys) // 2)])[:n]
    rows = {"rigid": [], "swing": []}
    W, H = 360, 360
    with exclusive("splat_hair preview"):
        for i in pick:
            Rg, Pj = fk_keys(rest, Z, "follow", keys[i:i + 1], full=True)
            mean, q = lbs_pose(av, rest, Rg[0], Pj[0])
            K = cc["K"][i].copy()
            K[:2] *= W / 1080.0
            for mode in ("rigid", "swing"):
                m, qq = mean.copy(), q.copy()
                if mode == "swing":
                    m[free], qq[free] = pose_bound_cached(sim["tracks_world"][i], sim["head_R"][i], st, Fr, bd)
                im = render_check(m, qq, av["scale"], av["opacity"], av["rgb"], cc["R"][i], cc["tvec"][i], K, W, H,
                                  device=device)
                im = np.ascontiguousarray(im)
                cv2.putText(im, f"{mode} t={keys[i] / 30 - 2643 / 30:.2f}s", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(im, f"yaw {rate[i]:.0f} deg/s", (6, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                            cv2.LINE_AA)
                rows[mode].append(im)
    out = p["sheets"] / "local_preview_rigid_vs_swing.jpg"
    cv2.imwrite(str(out), np.concatenate([np.concatenate(rows["rigid"], 1), np.concatenate(rows["swing"], 1)], 0)[:, :, ::-1],
                [cv2.IMWRITE_JPEG_QUALITY, 90])
    return out


def jl_proof_job(max_minutes: float = 25, only: tuple = (), tag: str = "") -> Path:
    """Pod job for the proof: install LHM++ (remote/cand_lhmpp/install.sh), render the groups (all four, or `only`) with
    --hair; tag != "": job pod_job_jl_hair_<tag>.json, pod report in pod_out_<tag>/."""
    from ..take import WORKSPACE
    p = _jl_paths()
    groups = tuple(only) or JL_GROUPS
    if tag:
        p["pod_out"] = p["hair"] / f"pod_out_{tag}"
        p["job"] = p["job"].with_name(f"pod_job_jl_hair_{tag}.json")
    ws = lambda q: Path(q).resolve().relative_to(WORKSPACE.resolve()).as_posix()  # noqa: E731
    rel = lambda q: ws(q)[len("dancecap/"):]  # noqa: E731
    vj = json.loads((p["input"] / "views.json").read_text())
    last = {g_: f"{vj['groups'][g_]['n'] - 1:05d}.png" for g_ in groups}
    only_arg = f" --only {','.join(groups)}" if only else ""
    d = {"schema": "dancecap.pod_job/1", "name": "jl_hair_proof" + (f"_{tag}" if tag else ""), "take": "john_lula",
         "max_minutes": max_minutes,
         "needs": [], "inputs": [ws(p["input"]) + "/", ws(p["bundle"]) + "/"],
         "exclude": ["dancecap/work/runpod/pywheels", "dancecap/work/runpod/wheelhouse", "mamma-turing"],
         "env": {"LHMPP_COMMIT": "906b5d9fb967ab42efb92f6fa55bf22cac86b653",
                 "LHMPP_ROOT": "/workspace/lhmpp_cand/LHM-plusplus", "LHMPP_VENV": "/workspace/lhmpp_cand/venv",
                 "LHMPP_MODEL": "LHMPP-700M-PixelShuffle", "TORCH_HOME": "/workspace/lhmpp_cand/torch_home",
                 "HF_HOME": "/workspace/lhmpp_cand/hf_home"},
         "commands": [
             {"name": "install", "timeout_min": 35, "cmd": "bash remote/cand_lhmpp/install.sh"},
             {"name": "render", "timeout_min": 20,
              "cmd": f"/workspace/lhmpp_cand/venv/bin/python remote/cand_lhmpp/lhmpp_render_pod.py --input "
                     f"{rel(p['input'])} --out {rel(p['pod_out'])} --renders {rel(p['renders'])} --hair {rel(p['bundle'])}"
                     f"{only_arg}"}],
         "outputs": [ws(p["pod_out"]) + "/"] + [ws(p["renders"] / g_) + "/" for g_ in groups],
         "verify": [ws(p["pod_out"] / "lhmpp_render_pod.json")] + [ws(p["renders"] / g_ / last[g_]) for g_ in groups],
         "after_download": ["python -m dancecap.mono4d.splat_hair jl-proof import --session <sid>"],
         "notes": ["john_lula swinging-hair proof (splat_hair): the realism run's LHM++ avatars (same refs, same poses = "
                   "the current best fit), orbit 9-20 s + a close-up of Lula, rigid vs swinging ponytail (free hair "
                   "Gaussians on simulated strands); crops go to our own pod only (owner consent 2026-10-19)"]}
    p["job"].write_text(json.dumps(d, indent=1), newline="\n")
    log(f"job -> {p['job']}")
    return p["job"]


def jl_proof_import(sid: str | None = None) -> dict:
    p = _jl_paths()
    vj = json.loads((p["input"] / "views.json").read_text())
    pods = {}
    for d_ in sorted(p["hair"].glob("pod_out*")):
        if (d_ / "lhmpp_render_pod.json").exists():
            pj = json.loads((d_ / "lhmpp_render_pod.json").read_text())
            pods[d_.name] = {k: pj.get(k) for k in ("hair", "errors", "timings", "summary", "versions")}
    groups = {g_: {"planned": gi["n"], "on_disk": len(list((p["renders"] / g_).glob("*.png")))}
              for g_, gi in vj["groups"].items()}
    root = p["work"].parent / "runpod" / "sessions"
    cands = [root / sid] if sid else sorted((q for q in root.glob("*") if (q / "result.json").exists()
                                             and '"jl_hair_proof' in (q / "result.json").read_text(errors="ignore")),
                                            key=lambda q: q.stat().st_mtime)
    sessions = []
    for c_ in cands:
        if (c_ / "result.json").exists():
            r = json.loads((c_ / "result.json").read_text())
            pd = r.get("pod") or {}
            sessions.append({"session": c_.name, "jobs": sorted((r.get("jobs") or {}).keys()), "pod_id": pd.get("id"),
                             "gpu": pd.get("gpu"), "cost_usd": pd.get("ledger_cost"),
                             "pod_minutes": round((pd.get("lifetime_s") or 0) / 60, 1)})
    rep = {"groups": groups, "sessions": sessions, "cost_usd_total": round(sum(s_["cost_usd"] or 0 for s_ in sessions), 4),
           "pods": pods}
    (p["hair"] / "import.json").write_text(json.dumps(rep, indent=1, default=str))
    log(f"import: {json.dumps(groups)} sessions {json.dumps(sessions)}")
    return rep


def jl_proof_video() -> dict:
    """Side-by-side MP4s (1920 x 1080, 30 fps, the heat's audio of 9-20 s): rigid | swinging, each panel 940 x 940 of
    the 1080 x 1080 renders over the studio backdrop of jl_realism_render; + contact sheets read back from the MP4s."""
    import cv2
    from . import jl_realism_render as JR
    p = _jl_paths()
    p["videos"].mkdir(parents=True, exist_ok=True)
    with np.load(p["input"] / "views.npz") as z:
        V = {k: z[k] for k in z.files}
    with np.load(p["input"] / "poses.npz") as z:
        Zj = {r: np.asarray(z[f"{r}_joints22"]) for r in ("lead", "follow")}
    NV = 2643
    out = {}
    Wc, Hc, pw = 1920, 1080, 940
    xl, xr, y0 = Wc // 2 - pw - 10, Wc // 2 + 10, 96
    for view in ("orbit", "close"):
        n = len(V[f"{view}_rigid_key"])
        keys = V[f"{view}_rigid_key"]
        a0 = float(keys[0] - NV) / 30.0

        def make(i, view=view, keys=keys):
            canvas = np.full((Hc, Wc, 3), 24, np.uint8)
            k = int(keys[i])
            J = {r: np.concatenate([Zj[r][k], np.zeros((33, 3))], 0) for r in ("lead", "follow")}
            for g_, x0 in ((f"{view}_rigid", xl), (f"{view}_swing", xr)):
                bg = JR.background(V[f"{g_}_K"][i], V[f"{g_}_R"][i], V[f"{g_}_tvec"][i], 1080, 1080,
                                   JR.shadow_points(J))
                fr = JR.composite(bg, JR._read_rgba(p["renders"] / g_ / f"{i:05d}.png"))
                canvas[y0:y0 + pw, x0:x0 + pw] = cv2.resize(fr, (pw, pw), interpolation=cv2.INTER_AREA)
            t = (k - NV) / 30.0
            return JR._text(canvas, [("LHM++ hair skinned to the body (rigid)", (xl + 8, 40), 30),
                                     ("swinging hair (simulated strands)", (xr + 8, 40), 30),
                                     (f"John & Lula  t = {t:5.2f} s", (xl + 8, Hc - 40), 24),
                                     ("same avatars, same pose, same camera", (xr + 8, Hc - 40), 24)])
        JR.VIDEOS = p["videos"]
        res = JR._encode(f"jl_hair_{view}_rigid_vs_swing.mp4", n, make, Wc, Hc, a0, False)
        out[view] = res
        cap = cv2.VideoCapture(str(p["videos"] / f"jl_hair_{view}_rigid_vs_swing.mp4"))
        tiles = []
        for q in np.linspace(10, n - 10, 9).round().astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(q))
            ok, fr = cap.read()
            if ok:
                tiles.append(cv2.resize(fr, (640, 360), interpolation=cv2.INTER_AREA))
        cap.release()
        if tiles:
            rows = [np.concatenate(tiles[i:i + 3], 1) for i in range(0, len(tiles) - len(tiles) % 3, 3)]
            cv2.imwrite(str(p["sheets"] / f"video_{view}_sheet.jpg"), np.concatenate(rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 88])
    (p["videos"] / "videos.json").write_text(json.dumps(out, indent=1, default=str))
    out["spin_sheet"] = str(jl_spin_sheet())
    return out


def jl_spin_sheet(n: int = 6) -> Path:
    """LOOK: the n frames with the largest simulated swing (tip offset from the head-carried rest tail, >= 0.5 s apart):
    close-up rigid | swinging (over the backdrop) and the orbit view cropped around Lula's head, rigid | swinging."""
    import cv2
    from . import jl_realism_render as JR
    p = _jl_paths()
    with np.load(p["input"] / "views.npz") as z:
        V = {k: z[k] for k in z.files}
    with np.load(p["input"] / "poses.npz") as z:
        Zj = {r: np.asarray(z[f"{r}_joints22"]) for r in ("lead", "follow")}
    tr = load_bundle(p["bundle"] / "follow_tracks.npz")
    b = load_bundle(p["bundle"] / "follow_hair.npz")
    hR, hP = tr["head_R"].astype(float), tr["head_P"].astype(float)
    hj = b["head_joint"]
    W_ = np.einsum("fij,fsnj->fsni", hR, tr["tracks_local"].astype(float) - hj) + hP[:, None, None]
    car = np.einsum("fij,snj->fsni", hR, b["strands_rest"] - hj) + hP[:, None, None]
    tip = np.linalg.norm(W_[:, :, -1] - car[:, :, -1], axis=-1).mean(1)
    pick = []
    for i in np.argsort(-tip):
        if all(abs(int(i) - j) >= 15 for j in pick):
            pick.append(int(i))
        if len(pick) >= n:
            break
    rows = []
    keys = V["close_rigid_key"]
    for i in sorted(pick):
        k = int(keys[i])
        J = {r: np.concatenate([Zj[r][k], np.zeros((33, 3))], 0) for r in ("lead", "follow")}
        tiles = []
        for view in ("close", "orbit"):
            for mode in ("rigid", "swing"):
                g_ = f"{view}_{mode}"
                bg = JR.background(V[f"{g_}_K"][i], V[f"{g_}_R"][i], V[f"{g_}_tvec"][i], 1080, 1080, JR.shadow_points(J))
                fr = JR.composite(bg, JR._read_rgba(p["renders"] / g_ / f"{i:05d}.png"))
                if view == "orbit":      # crop around her head
                    X = V[f"{g_}_R"][i] @ Zj["follow"][k][15] + V[f"{g_}_tvec"][i]
                    u = V[f"{g_}_K"][i] @ (X / X[2])
                    cx, cy = int(np.clip(u[0], 200, 880)), int(np.clip(u[1] + 60, 200, 880))
                    fr = cv2.resize(fr[cy - 200:cy + 200, cx - 200:cx + 200], (420, 420), interpolation=cv2.INTER_CUBIC)
                else:
                    fr = cv2.resize(fr, (420, 420), interpolation=cv2.INTER_AREA)
                fr = np.ascontiguousarray(fr)
                cv2.putText(fr, f"{view} {mode} t={(k - 2643) / 30:.2f}s", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 2, cv2.LINE_AA)
                tiles.append(fr)
        rows.append(np.concatenate(tiles, 1))
    path = p["sheets"] / "spins_rigid_vs_swing.jpg"
    cv2.imwrite(str(path), np.concatenate(rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return path


# ================================================================================================ CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.mono4d.splat_hair", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("prepare", help="select + strands + bind one avatar -> <out>/<role>_hair.npz/.json + sheets")
    s.add_argument("--gaussians", required=True, help="LHM++ *_cano.ply or a .splatav")
    s.add_argument("--betas", required=True, help="poses.npz (uses <role>_betas) or a .json list")
    s.add_argument("--role", required=True)
    s.add_argument("--style", default="ponytail", choices=sorted(SELECT))
    s.add_argument("--out", required=True)
    s.add_argument("--params", default="{}", help="JSON overrides of SELECT[style]")
    s.add_argument("--device", default="cpu")
    s = sub.add_parser("simulate", help="simulate a prepared avatar over consecutive pose keys -> <out>/<role>_tracks.npz")
    s.add_argument("--hair", required=True, help="dir with <role>_hair.npz")
    s.add_argument("--poses", required=True, help="LHM++ render input poses.npz")
    s.add_argument("--role", required=True)
    s.add_argument("--keys", required=True, help="a:b inclusive pose keys (consecutive 30 Hz)")
    s.add_argument("--partner", default=None)
    s.add_argument("--style", default="ponytail", choices=sorted(SIM))
    s.add_argument("--params", default="{}", help="JSON overrides of SIM[style]")
    s = sub.add_parser("render-tracks", help="tracks (+ aliases) for every key of the listed groups of an LHM++ render input")
    s.add_argument("--hair", required=True, help="dir with <role>_hair.npz (prepare); tracks + hair.json written here")
    s.add_argument("--input", required=True, help="render input dir (poses.npz, views.npz)")
    s.add_argument("--role", required=True)
    s.add_argument("--partner", default=None)
    s.add_argument("--groups", required=True, help="comma list of view groups that get swinging hair")
    s.add_argument("--params", default="{}", help="JSON overrides of SIM[style]")
    s = sub.add_parser("jl-proof", help="john_lula rigid vs swinging proof")
    s.add_argument("step", choices=["prep", "job", "import", "video"])
    s.add_argument("--only", default="", help="job: comma list of groups to render (default all four)")
    s.add_argument("--tag", default="", help="job: a tag for a second job (own job file + pod_out_<tag>/)")
    s.add_argument("--session", default=None)
    s.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    if a.cmd == "prepare":
        if a.betas.endswith(".npz"):
            with np.load(a.betas) as z:
                betas = np.asarray(z[f"{a.role}_betas"], float)
        else:
            betas = np.asarray(json.loads(Path(a.betas).read_text()), float)
        prepare(Path(a.gaussians), betas, a.style, Path(a.out), a.role, json.loads(a.params), device=a.device)
    elif a.cmd == "simulate":
        with np.load(a.poses) as z:
            Z = {k: np.asarray(z[k]) for k in z.files}
        k0, k1 = (int(v) for v in a.keys.split(":"))
        b = load_bundle(Path(a.hair) / f"{a.role}_hair.npz")
        sim = simulate_role(b, Z, a.role, np.arange(k0, k1 + 1), a.partner, a.style, json.loads(a.params))
        save_tracks(Path(a.hair) / f"{a.role}_tracks.npz", sim, a.role)
        (Path(a.hair) / f"{a.role}_tracks_check.json").write_text(json.dumps(sim["check"], indent=1, default=float))
        log(json.dumps(sim["check"], default=float))
    elif a.cmd == "render-tracks":
        render_tracks(Path(a.hair), Path(a.input), a.role, a.partner, [g for g in a.groups.split(",") if g],
                      sim_params=json.loads(a.params))
    elif a.step == "prep":
        jl_proof_prep(a.device)
    elif a.step == "job":
        jl_proof_job(only=tuple(g for g in a.only.split(",") if g), tag=a.tag)
    elif a.step == "import":
        jl_proof_import(a.session)
    else:
        jl_proof_video()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
