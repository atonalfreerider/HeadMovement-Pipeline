"""Body model for the physics stage: SMPL-X shape -> rigid segment masses, COM and inertia; per-frame FK.

The shaped rest mesh (T-pose, betas from the take) is closed (the eye sockets are capped, the separate eyeball
shells are dropped), voxelised, and every interior voxel is given to the SMPL-X joint whose LBS weight dominates
there (weights interpolated from the nearest surface vertices). With uniform density each of the 55 joints then
carries a rigid mass element (mass, rest COM, inertia about that COM) that moves with the joint's FK transform.
Whole-body COM / angular momentum are sums over those elements. Total volume comes from the divergence theorem
on the closed mesh (exact for the mesh); the voxels only split it into segments.

Segment groups are compared with de Leva (1996, J Biomech 29:1223) mass fractions. The boundaries differ by
construction (LBS skinning regions vs anatomical landmarks), so deviations of a few percent of body mass between
trunk / thigh / head are expected and reported, not corrected.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .take import WORKSPACE

DENSITY = 1010.0  # kg/m^3, whole-body average (lungs included); mass flagged +-10% when derived from it
G = 9.81
N_JOINTS = 55
MAMMA_LOCKED_HEAD = WORKSPACE / "mamma" / "data" / "body_models" / "smplx_locked_head" / "smplx"

# SMPL-X joint -> de Leva segment group. LBS weight of joint j = vertices moving with the bone that starts at j.
SEGMENT_GROUPS: dict[str, tuple[int, ...]] = {
    "head": (12, 15, 22, 23, 24),          # neck + head + jaw + eyes (de Leva head = vertex .. C7)
    "trunk": (0, 3, 6, 9, 13, 14),         # pelvis, spine1-3, collars
    "upper_arm_l": (16,), "upper_arm_r": (17,),
    "forearm_l": (18,), "forearm_r": (19,),
    "hand_l": (20,) + tuple(range(25, 40)), "hand_r": (21,) + tuple(range(40, 55)),
    "thigh_l": (1,), "thigh_r": (2,),
    "shank_l": (4,), "shank_r": (5,),
    "foot_l": (7, 10), "foot_r": (8, 11),
}
# de Leva (1996) Table 4, segment mass in % of body mass
DE_LEVA = {
    "male": {"head": 6.94, "trunk": 43.46, "upper_arm": 2.71, "forearm": 1.62, "hand": 0.61, "thigh": 14.16,
             "shank": 4.33, "foot": 1.37},
    "female": {"head": 6.68, "trunk": 42.57, "upper_arm": 2.55, "forearm": 1.38, "hand": 0.56, "thigh": 14.78,
               "shank": 4.81, "foot": 1.29},
}
DE_LEVA["neutral"] = {k: 0.5 * (DE_LEVA["male"][k] + DE_LEVA["female"][k]) for k in DE_LEVA["male"]}
FOOT_JOINTS = {"left": (7, 10), "right": (8, 11)}


def model_file(npz, gender: str | None = None) -> Path:
    """SMPL-X model npz for a contract smplx file: MAMMA's locked-head model when the file says so, else
    WORKSPACE/smplx/SMPLX_<GENDER>.npz."""
    g = (gender or (str(npz["gender"]) if "gender" in npz else "neutral")).upper()
    name = f"SMPLX_{g}.npz"
    if "model" in npz and "locked_head" in str(npz["model"]) and (MAMMA_LOCKED_HEAD / name).exists():
        return MAMMA_LOCKED_HEAD / name
    return WORKSPACE / "smplx" / name


@dataclass
class BodyModel:
    """Shaped SMPL-X rest body + rigid mass elements per joint."""
    gender: str
    model_path: Path
    betas: np.ndarray
    parents: np.ndarray            # (55,)
    v_rest: np.ndarray             # (V, 3) shaped rest vertices (T-pose)
    faces: np.ndarray              # (F, 3) body faces incl. eye caps (closed), eyeballs dropped
    j_rest: np.ndarray             # (55, 3) rest joints
    weights: np.ndarray            # (V, 55) LBS weights
    volume: float                  # m^3, closed mesh
    stature: float                 # m, rest mesh height
    elem_mass: np.ndarray = field(default=None)   # (55,) kg (after scale to total mass)
    elem_com: np.ndarray = field(default=None)    # (55, 3) rest-frame COM
    elem_inertia: np.ndarray = field(default=None)  # (55, 3, 3) about elem COM, rest axes, kg m^2
    mass: float = 0.0
    mass_source: str = ""
    mass_rel_sigma: float = 0.0
    voxel_m: float = 0.0
    voxel_volume: float = 0.0

    @property
    def rest_com(self) -> np.ndarray:
        return (self.elem_mass[:, None] * self.elem_com).sum(0) / self.mass

    def sole_vertices(self, foot: str, height: float = 0.02) -> np.ndarray:
        """Indices of the foot's sole: vertices of that foot's skinning region within `height` of its lowest point."""
        lab = self.weights.argmax(1)
        idx = np.flatnonzero(np.isin(lab, FOOT_JOINTS[foot]))
        y = self.v_rest[idx, 1]
        return idx[y < y.min() + height]

    def group_fractions(self) -> dict[str, float]:
        out = {}
        for name, joints in SEGMENT_GROUPS.items():
            out[name] = float(self.elem_mass[list(joints)].sum() / self.mass)
        return out

    def de_leva_check(self) -> dict:
        """Group mass fractions (left+right merged) vs de Leva (1996), in % of body mass."""
        ref = DE_LEVA.get(self.gender, DE_LEVA["neutral"])
        fr = self.group_fractions()
        rows = {}
        for k, r in ref.items():
            ours = 100 * sum(v for n, v in fr.items() if n == k or n.rsplit("_", 1)[0] == k)
            bilateral = k not in ("head", "trunk")
            ref_total = r * (2 if bilateral else 1)
            rows[k] = {"ours_pct": round(ours, 2), "de_leva_pct": round(ref_total, 2),
                       "diff_pct_points": round(ours - ref_total, 2), "ratio": round(ours / ref_total, 3)}
        lr = {k: round(100 * (fr[f"{k}_l"] - fr[f"{k}_r"]), 3) for k in ("upper_arm", "forearm", "hand", "thigh",
                                                                       "shank", "foot")}
        return {"segments": rows, "left_minus_right_pct": lr,
                "max_abs_diff_pct_points": round(max(abs(v["diff_pct_points"]) for v in rows.values()), 2),
                "note": "LBS skinning regions vs de Leva anatomical landmarks: boundaries differ by construction "
                        "(e.g. gluteal mass is skinned to the pelvis=trunk, de Leva puts it in the thigh)."}


# ----------------------------------------------------------------------------------------------- shape + mesh

def _closed_body_faces(faces: np.ndarray, n_verts: int) -> tuple[np.ndarray, np.ndarray]:
    """Drop the eyeball shells and cap the eye sockets with fans to a new centroid vertex each.

    Returns (faces, cap_loops) where cap_loops are the ordered boundary vertex loops (for the cap centroids)."""
    import trimesh
    m = trimesh.Trimesh(np.zeros((n_verts, 3)), faces, process=False)
    comp = trimesh.graph.connected_component_labels(m.face_adjacency, node_count=len(faces))
    big = np.bincount(comp).argmax()
    body = faces[comp == big]
    # boundary edges: directed edges whose reverse is absent
    e = np.concatenate([body[:, [0, 1]], body[:, [1, 2]], body[:, [2, 0]]])
    key = {tuple(x) for x in e.tolist()}
    bnd = [(a, b) for a, b in e.tolist() if (b, a) not in key]
    nxt = {}
    for a, b in bnd:
        nxt[b] = a  # walk the boundary opposite to the face winding -> cap faces get the right orientation
    loops, seen = [], set()
    for s in list(nxt):
        if s in seen:
            continue
        loop, v = [], s
        while v not in seen:
            seen.add(v)
            loop.append(v)
            v = nxt[v]
        loops.append(np.array(loop))
    return body, loops


def _cap(verts: np.ndarray, body: np.ndarray, loops: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Append one centroid per loop and fan faces (orientation consistent with the outward body faces)."""
    v = [verts]
    f = [body]
    n = len(verts)
    for loop in loops:
        c = n
        n += 1
        v.append(verts[loop].mean(0, keepdims=True))
        f.append(np.stack([loop, np.roll(loop, -1), np.full(len(loop), c)], 1))
    return np.concatenate(v), np.concatenate(f)


def mesh_volume_com(verts: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Divergence theorem on a closed mesh. verts (..., V, 3) -> volume (...,), COM (..., 3)."""
    a, b, c = verts[..., faces[:, 0], :], verts[..., faces[:, 1], :], verts[..., faces[:, 2], :]
    v6 = np.einsum("...i,...i->...", a, np.cross(b, c))  # 6 x signed tetra volume (origin apex)
    vol = v6.sum(-1) / 6.0
    com = (v6[..., None] * (a + b + c)).sum(-2) / (24.0 * vol[..., None])
    return vol, com


def load_body(npz, gender: str | None = None, model_path: Path | None = None) -> BodyModel:
    """Shaped rest body for a contract smplx npz (betas, gender, model field)."""
    path = Path(model_path) if model_path else model_file(npz, gender)
    m = np.load(path, allow_pickle=True)
    betas = np.asarray(npz["betas"], float).reshape(-1)
    sd = np.asarray(m["shapedirs"], float)[:, :, :len(betas)]
    v = np.asarray(m["v_template"], float) + sd @ betas
    W = np.asarray(m["weights"], float)
    J = np.asarray(m["J_regressor"], float) @ v
    parents = np.asarray(m["kintree_table"])[0].astype(np.int64)
    parents[0] = -1
    body, loops = _closed_body_faces(np.asarray(m["f"], np.int64), len(v))
    vc, fc = _cap(v, body, loops)
    vol, _ = mesh_volume_com(vc, fc)
    g = (gender or (str(npz["gender"]) if "gender" in npz else "neutral")).lower()
    # the caps' centroid vertices are only for volume; keep them in v_rest (weights of their loop's mean)
    Wc = np.concatenate([W] + [W[l].mean(0, keepdims=True) for l in loops])
    return BodyModel(gender=g, model_path=path, betas=betas, parents=parents, v_rest=vc, faces=fc, j_rest=J,
                     weights=Wc, volume=float(vol), stature=float(vc[:, 1].max() - vc[:, 1].min()))


def segment_masses(body: BodyModel, mass: float | None = None, voxel: float = 0.007, k_nn: int = 8) -> BodyModel:
    """Voxelise the closed rest mesh, label voxels by interpolated LBS weight argmax, uniform density.

    mass: total kg (user input) or None -> closed-mesh volume x DENSITY, flagged +-10%."""
    import open3d as o3d
    from scipy.spatial import cKDTree
    v, f = body.v_rest, body.faces
    lo, hi = v.min(0) - voxel, v.max(0) + voxel
    axes = [np.arange(lo[i] + voxel / 2, hi[i], voxel) for i in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(v.astype(np.float32)), o3d.core.Tensor(f.astype(np.uint32)))
    occ = scene.compute_occupancy(o3d.core.Tensor(grid.astype(np.float32)), nsamples=3).numpy() > 0.5
    pts = grid[occ]
    tree = cKDTree(v)
    lab = np.empty(len(pts), np.int64)
    for s in range(0, len(pts), 200_000):
        d, ii = tree.query(pts[s:s + 200_000], k=k_nn)
        w = 1.0 / np.maximum(d, 1e-4)
        acc = np.einsum("nk,nkj->nj", w, body.weights[ii])
        lab[s:s + 200_000] = acc.argmax(1)
    if mass is None:
        mass_total, source, rel = body.volume * DENSITY, f"closed-mesh volume x {DENSITY:.0f} kg/m^3", 0.10
    else:
        mass_total, source, rel = float(mass), "user input", 0.02
    vox_vol = len(pts) * voxel ** 3
    rho = mass_total / vox_vol  # density scaled so the voxel body carries exactly the total mass
    dm = rho * voxel ** 3
    em = np.zeros(N_JOINTS)
    ec = np.zeros((N_JOINTS, 3))
    ei = np.zeros((N_JOINTS, 3, 3))
    for j in range(N_JOINTS):
        p = pts[lab == j]
        if not len(p):
            ec[j] = body.j_rest[j]
            continue
        em[j] = dm * len(p)
        ec[j] = p.mean(0)
        r = p - ec[j]
        # point masses + each voxel's own cube inertia (dm a^2 / 6 on the diagonal)
        ei[j] = dm * ((r ** 2).sum() * np.eye(3) - r.T @ r) + np.eye(3) * em[j] * voxel ** 2 / 6
    body.elem_mass, body.elem_com, body.elem_inertia = em, ec, ei
    body.mass, body.mass_source, body.mass_rel_sigma = float(mass_total), source, rel
    body.voxel_m, body.voxel_volume = voxel, float(vox_vol)
    return body


# ----------------------------------------------------------------------------------------------- kinematics

def axis_angle_to_matrix(rv: np.ndarray) -> np.ndarray:
    """Rodrigues, (..., 3) -> (..., 3, 3)."""
    th = np.linalg.norm(rv, axis=-1, keepdims=True)
    k = rv / np.maximum(th, 1e-12)
    K = np.zeros(rv.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 0] = -k[..., 2], k[..., 1], k[..., 2]
    K[..., 1, 2], K[..., 2, 0], K[..., 2, 1] = -k[..., 0], -k[..., 1], k[..., 0]
    s, c = np.sin(th)[..., None], np.cos(th)[..., None]
    return np.eye(3) + s * K + (1 - c) * (K @ K)


def full_pose(npz) -> np.ndarray:
    """(T, 55, 3) axis-angle in SMPL-X joint order: global, body(21), jaw, leye, reye, lhand(15), rhand(15).
    Hand poses are absolute (contract: flat_hand_mean=True)."""
    T = len(npz["t"])
    get = lambda k, n: np.asarray(npz[k], float).reshape(T, n) if k in npz else np.zeros((T, n))
    parts = [get("global_orient", 3), get("body_pose", 63), get("jaw_pose", 3), get("leye_pose", 3),
             get("reye_pose", 3), get("left_hand_pose", 45), get("right_hand_pose", 45)]
    return np.concatenate(parts, 1).reshape(T, N_JOINTS, 3)


def forward_kinematics(body: BodyModel, pose: np.ndarray, transl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World rotation R (T,55,3,3) and translation p (T,55,3) of each joint's rigid frame such that a rest-space
    point x of that joint's segment moves to R @ x + p. Joint positions = R @ j_rest + p."""
    T = len(pose)
    Rl = axis_angle_to_matrix(pose)
    R = np.empty((T, N_JOINTS, 3, 3))
    pj = np.empty((T, N_JOINTS, 3))  # posed joint positions
    J = body.j_rest
    for j in range(N_JOINTS):
        par = body.parents[j]
        if par < 0:
            R[:, j] = Rl[:, j]
            pj[:, j] = J[j] + transl
        else:
            R[:, j] = R[:, par] @ Rl[:, j]
            pj[:, j] = pj[:, par] + np.einsum("tab,b->ta", R[:, par], J[j] - J[par])
    p = pj - np.einsum("tjab,jb->tja", R, J)
    return R, p


def lbs_points(body: BodyModel, R: np.ndarray, p: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Linear-blend-skinned positions (T, n, 3) of rest vertices idx (no pose blendshapes)."""
    W = body.weights[idx]                                          # (n, 55)
    x = body.v_rest[idx]
    Rb = np.matmul(W, R.reshape(len(R), N_JOINTS, 9)).reshape(len(R), len(idx), 3, 3)   # blended transforms
    pb_ = np.matmul(W, p)                                          # (T, n, 3)
    return np.einsum("tnab,nb->tna", Rb, x) + pb_


def element_world(body: BodyModel, R: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mass-element COMs (T,55,3) and world inertia about each element COM (T,55,3,3)."""
    c = np.einsum("tjab,jb->tja", R, body.elem_com) + p
    I = np.einsum("tjab,jbc,tjdc->tjad", R, body.elem_inertia, R)
    return c, I


def posed_mesh_com(body: BodyModel, R: np.ndarray, p: np.ndarray, chunk: int = 64) -> np.ndarray:
    """Cross-check: COM of the LBS-posed closed mesh (uniform density, divergence theorem), (T, 3)."""
    out = []
    V = np.arange(len(body.v_rest))
    for s in range(0, len(R), chunk):
        v = lbs_points(body, R[s:s + chunk], p[s:s + chunk], V)
        out.append(mesh_volume_com(v, body.faces)[1])
    return np.concatenate(out)


def rest_com_height(body: BodyModel) -> float:
    """Standing COM height above the soles in the rest pose (inverted-pendulum length for the XCoM)."""
    return float(body.rest_com[1] - body.v_rest[:, 1].min())


def summary(body: BodyModel) -> dict:
    return {"gender": body.gender, "model": str(body.model_path.relative_to(WORKSPACE)
                                                if body.model_path.is_relative_to(WORKSPACE) else body.model_path),
            "betas": [round(float(b), 4) for b in body.betas], "mass_kg": round(body.mass, 2),
            "mass_source": body.mass_source, "mass_rel_sigma": body.mass_rel_sigma,
            "closed_mesh_volume_l": round(body.volume * 1000, 2), "voxel_volume_l": round(body.voxel_volume * 1000, 2),
            "voxel_mm": body.voxel_m * 1000, "stature_m": round(body.stature, 3),
            "rest_com_height_m": round(rest_com_height(body), 3),
            "segment_fractions": {k: round(v, 4) for k, v in body.group_fractions().items()},
            "de_leva": body.de_leva_check()}

