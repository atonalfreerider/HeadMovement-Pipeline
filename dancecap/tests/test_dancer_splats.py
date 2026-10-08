"""dancecap.dancer_splats (avatar-guided dancer splats): binding math, skinning of the anchors with pose corrections,
hair strands, label / mask handling, the model's posing + MCMC on the CPU (no gsplat), the pod job spec, progress
parsing and the pipeline registry / plan inclusion.

    .venv-nopo4d\\Scripts\\python.exe -m pytest tests/test_dancer_splats.py -q
(The GPU path - gsplat rendering, training, eval - is exercised by the local smoke run:
 python -m dancecap.dancer_splats larissa_kadu prepare --preset smoke --out work/larissa_kadu/dancer_splats/smoke ...)
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch = pytest.importorskip("torch")

from dancecap.dancer_splats import (L_BG, L_BOTH, L_FOLLOW, L_HAIR, L_IGNORE, L_LEAD, R_FACE, R_SKIN,  # noqa: E402
                                    SplatConfig, preset)
from dancecap.dancer_splats import binding as bd  # noqa: E402
from dancecap.dancer_splats import data as dd  # noqa: E402

# SMPL-X kinematic tree (kintree_table[0] of SMPLX_NEUTRAL.npz, root = -1)
SMPLX_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 15, 15, 15, 20, 25, 26, 20,
                 28, 29, 20, 31, 32, 20, 34, 35, 20, 37, 38, 21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50, 21, 52, 53]


def _rot(rng):
    return bd.axis_angle_to_matrix(torch.as_tensor(rng.normal(size=3), dtype=torch.float64))


# ------------------------------------------------------------------------------------------------ binding

def test_rotation_conversions_roundtrip():
    rng = np.random.default_rng(0)
    q = torch.as_tensor(rng.normal(size=(200, 4)))
    q = q / q.norm(dim=-1, keepdim=True)
    q = q * torch.where(q[:, :1] < 0, -1.0, 1.0)
    R = bd.quat_to_matrix(q)
    assert torch.allclose(R @ R.transpose(-1, -2), torch.eye(3, dtype=R.dtype).expand_as(R), atol=1e-6)
    assert torch.allclose(bd.matrix_to_quat(R), q, atol=1e-5)
    from scipy.spatial.transform import Rotation
    rv = rng.normal(size=(50, 3)) * 1.3
    A = bd.axis_angle_to_matrix(torch.as_tensor(rv)).numpy()
    assert np.allclose(A, Rotation.from_rotvec(rv).as_matrix(), atol=1e-6)
    assert torch.allclose(bd.axis_angle_to_matrix(torch.zeros(3)), torch.eye(3))  # smooth at 0


def test_triangle_frames_and_body_binding():
    rng = np.random.default_rng(1)
    V = torch.as_tensor(rng.normal(size=(30, 3)))
    faces = torch.as_tensor(np.array([[i, (i + 1) % 30, (i + 7) % 30] for i in range(30)]), dtype=torch.long)
    R, k = bd.triangle_frames(V, faces)
    I = torch.eye(3, dtype=R.dtype)
    assert torch.allclose(R.transpose(-1, -2) @ R, I.expand_as(R), atol=1e-9)
    assert torch.allclose(torch.det(R), torch.ones(len(faces), dtype=R.dtype))
    v = V[faces]
    area2 = torch.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0], dim=-1).norm(dim=-1)
    assert torch.allclose(k ** 2, area2)
    assert torch.allclose((R[..., 2] * (v[:, 1] - v[:, 0])).sum(-1), torch.zeros(len(faces), dtype=R.dtype), atol=1e-9)
    # Gaussians at rest: zero offsets sit on the surface, a normal offset moves along the normal
    n = 64
    fi = torch.as_tensor(rng.integers(0, len(faces), n))
    b = torch.as_tensor(rng.dirichlet([1, 1, 1], n))
    off = torch.zeros(n, 3, dtype=V.dtype)
    off[:, 2] = 0.01
    ls = torch.full((n, 3), math.log(0.004), dtype=V.dtype)
    q = torch.tensor([[1.0, 0, 0, 0]], dtype=V.dtype).expand(n, 4)
    mu, Rw, sc = bd.body_gaussians(V, faces, fi, b, off, k, ls, q)
    anchor = bd.barycentric_points(V, faces, fi, b)
    assert torch.allclose(mu - anchor, 0.01 * R[fi][..., 2], atol=1e-9)
    assert torch.allclose(Rw, R[fi]) and torch.allclose(sc, torch.full_like(sc, 0.004))
    # a rigid motion of the mesh moves the Gaussians rigidly; a scaled mesh scales offsets and sizes
    Q = _rot(rng)
    T = torch.as_tensor(rng.normal(size=3))
    mu2, Rw2, sc2 = bd.body_gaussians(V @ Q.T + T, faces, fi, b, off, k, ls, q)
    assert torch.allclose(mu2, mu @ Q.T + T, atol=1e-9) and torch.allclose(Rw2, Q @ Rw, atol=1e-9)
    mu3, _, sc3 = bd.body_gaussians(2 * V, faces, fi, b, off, k, ls, q)
    assert torch.allclose(mu3, 2 * mu, atol=1e-9) and torch.allclose(sc3, 2 * sc)
    # the scale cap
    _, _, sc4 = bd.body_gaussians(V, faces, fi, b, off, k, torch.full_like(ls, math.log(0.5)), q, scale_cap=0.03)
    assert float(sc4.max()) <= 0.03 + 1e-9


def test_area_samples_cover_the_mesh_uniformly():
    V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [10, 0, 0], [12, 0, 0], [10, 2, 0.0]])
    F = np.array([[0, 1, 2], [3, 4, 5]])  # areas 0.5 and 2
    fi, b = bd.triangle_area_samples(V, F, 20000, seed=0)
    assert np.allclose(b.sum(1), 1) and (b >= 0).all()
    assert abs((fi == 1).mean() - 0.8) < 0.02


# ------------------------------------------------------------------------------------------------ skinning

def _chain(rng, T=3):
    """A random posed SMPL-X-shaped skeleton: global rotations Rg (T, 55, 3, 3), posed joints Jp, local rotations."""
    p = np.array(SMPLX_PARENTS)
    J0 = rng.normal(size=(55, 3)) * 0.3
    Rl = bd.axis_angle_to_matrix(torch.as_tensor(rng.normal(size=(T, 55, 3)) * 0.5))
    Rg = torch.zeros(T, 55, 3, 3, dtype=torch.float64)
    Jp = torch.zeros(T, 55, 3, dtype=torch.float64)
    tr = torch.as_tensor(rng.normal(size=(T, 3)))
    for j in range(55):
        if p[j] < 0:
            Rg[:, j], Jp[:, j] = Rl[:, j], torch.as_tensor(J0[j]) + tr
        else:
            Rg[:, j] = Rg[:, p[j]] @ Rl[:, j]
            Jp[:, j] = Jp[:, p[j]] + (Rg[:, p[j]] @ torch.as_tensor(J0[j] - J0[p[j]]))
    return p, J0, Rl, Rg, Jp, tr


def _fk(p, J0, Rl, tr, r, dT):
    """Reference forward kinematics with local corrections: G'_j = G'_p [Rl_j Rot(r_j), J0_j - J0_p]."""
    T = Rl.shape[0]
    G = torch.zeros(T, 55, 4, 4, dtype=torch.float64)
    for j in range(55):
        L = torch.eye(4, dtype=torch.float64).repeat(T, 1, 1)
        rot = Rl[:, j] @ bd.axis_angle_to_matrix(r[:, j]) if j < r.shape[1] else Rl[:, j]
        L[:, :3, :3] = rot
        if p[j] < 0:
            L[:, :3, 3] = torch.as_tensor(J0[j]) + tr + dT
            G[:, j] = L
        else:
            L[:, :3, 3] = torch.as_tensor(J0[j] - J0[p[j]])
            G[:, j] = G[:, p[j]] @ L
    return G


def test_pose_delta_transforms_match_forward_kinematics():
    rng = np.random.default_rng(2)
    p, J0, Rl, Rg, Jp, tr = _chain(rng)
    B = 22
    levels = bd.chain_levels(p, B)
    r = torch.as_tensor(rng.normal(size=(3, B, 3)) * 0.05)
    dT = torch.as_tensor(rng.normal(size=(3, 3)) * 0.02)
    D = bd.pose_delta_transforms(Rg[:, :B], Jp[:, :B], r, dT, p, levels)
    G0 = _fk(p, J0, Rl, tr, torch.zeros_like(r), torch.zeros_like(dT))
    G1 = _fk(p, J0, Rl, tr, r, dT)
    assert torch.allclose(G0[:, :, :3, :3], Rg, atol=1e-9) and torch.allclose(G0[:, :, :3, 3], Jp, atol=1e-9)
    ref = G1[:, :B] @ torch.linalg.inv(G0[:, :B])
    assert torch.allclose(D, ref, atol=1e-9)
    # joints without a correction (jaw, eyes, fingers) move exactly with their body ancestor
    anc = bd.body_ancestor(p, B)
    assert anc[22] == 15 and anc[25] == 20 and anc[54] == 21 and all(anc[j] == j for j in range(B))
    ref_all = G1 @ torch.linalg.inv(G0)
    assert torch.allclose(D[:, torch.as_tensor(anc)], ref_all, atol=1e-9)
    # zero correction = identity
    I = bd.pose_delta_transforms(Rg[:, :B], Jp[:, :B], torch.zeros_like(r), torch.zeros_like(dT), p, levels)
    assert torch.allclose(I, torch.eye(4, dtype=I.dtype).expand_as(I), atol=1e-10)


def test_skinning_of_anchors():
    """Vertices re-skinned by the corrections: one joint's vertices rotate about that joint; blends interpolate;
    surface anchors (barycentric points) follow their triangle."""
    rng = np.random.default_rng(3)
    p, J0, Rl, Rg, Jp, tr = _chain(rng, T=1)
    B = 22
    r = torch.zeros(1, B, 3, dtype=torch.float64)
    r[0, 18] = torch.tensor([0.0, 0.0, 0.05])  # left elbow, 2.9 deg
    dT = torch.zeros(1, 3, dtype=torch.float64)
    D = bd.pose_delta_transforms(Rg[:, :B], Jp[:, :B], r, dT, p, bd.chain_levels(p, B))[:, torch.as_tensor(
        bd.body_ancestor(p, B))]
    V = torch.as_tensor(rng.normal(size=(1, 6, 3)))
    W = torch.zeros(6, 55, dtype=torch.float64)
    W[0, 18] = 1          # forearm vertex
    W[1, 20] = 1          # wrist: child of the elbow -> same motion
    W[2, 0] = 1           # pelvis: unaffected
    W[3, 18], W[3, 16] = 0.5, 0.5   # blend elbow / shoulder
    W[4, 37] = 1          # a finger: follows the wrist (its body ancestor)
    W[5, 9] = 1
    Vn = bd.skin_delta(V, D, W)
    Q = Rg[0, 18] @ bd.axis_angle_to_matrix(r[0, 18]) @ Rg[0, 18].T
    c = Jp[0, 18]
    exp0 = (Q @ (V[0, 0] - c)) + c
    assert torch.allclose(Vn[0, 0], exp0, atol=1e-9) and torch.allclose(Vn[0, 1], (Q @ (V[0, 1] - c)) + c, atol=1e-9)
    assert torch.allclose(Vn[0, 2], V[0, 2]) and torch.allclose(Vn[0, 5], V[0, 5])
    assert torch.allclose(Vn[0, 3], 0.5 * ((Q @ (V[0, 3] - c)) + c) + 0.5 * V[0, 3], atol=1e-9)
    assert torch.allclose(Vn[0, 4], (Q @ (V[0, 4] - c)) + c, atol=1e-9)
    # dense weights from the bundle's top-k
    w_idx = torch.tensor([[18, 16, 0, 1], [20, 0, 1, 2]])
    w_val = torch.tensor([[0.5, 0.5, 0.0, 0.0], [1.0, 0, 0, 0]], dtype=torch.float64)
    Wd = bd.dense_weights(w_idx, w_val, 55)
    assert Wd.shape == (2, 55) and float(Wd[0, 18]) == 0.5 and float(Wd[1].sum()) == 1.0


def test_dct_basis_and_sh():
    x = torch.tensor([0.0, 0.5, 1.0])
    phi = bd.dct_basis(x, 8)
    assert phi.shape == (3, 8) and torch.allclose(phi[0], torch.ones(8))
    coeffs = torch.zeros(5, 9, 3)
    coeffs[:, 0] = bd.rgb_to_sh0(torch.full((5, 3), 0.3))
    rgb = bd.sh_rgb(torch.randn(5, 3), coeffs, 2)
    assert torch.allclose(rgb, torch.full((5, 3), 0.3), atol=1e-6)
    rgb_b = bd.sh_rgb(torch.randn(2, 5, 3), coeffs, 1)  # batch of views sharing the coefficients
    assert rgb_b.shape == (2, 5, 3)


# ------------------------------------------------------------------------------------------------ hair

def test_rest_strand_over_the_skull_then_down():
    c, r = np.array([0.0, 1.6, 0.0]), 0.09
    for az in (0.0, 30.0, 120.0, 180.0):  # front roots are swept to the side, never over the face
        el = math.radians(40)
        u = np.array([math.sin(math.radians(az)) * math.cos(el), math.sin(el), math.cos(math.radians(az)) * math.cos(el)])
        root = c + r * u
        P = bd.rest_strand(root, u, c, r, 0.5, 17, standoff=0.01)
        L = np.linalg.norm(np.diff(P, axis=0), axis=1)
        assert len(P) == 17 and abs(L.sum() - 0.5) < 0.01 and L.std() < 0.002
        assert np.allclose(P[0], root + 0.01 * u, atol=1e-6)
        assert (np.linalg.norm(P - c, axis=1) >= r - 1e-6).all()
        assert P[-1, 1] < P[0, 1] - 0.25  # hangs down
        if abs(az) < 90:
            assert abs(math.degrees(math.atan2(P[-1, 0] - c[0], P[-1, 2] - c[2]))) >= 99  # behind the ear
    X = torch.as_tensor(bd.rest_strand(c + [0, r, 0], np.array([0, 1.0, 0]), c, r, 0.4, 17))
    Rs, L = bd.segment_frames(X, torch.tensor([0.0, 0.0, 1.0], dtype=X.dtype))
    assert torch.allclose(Rs.transpose(-1, -2) @ Rs, torch.eye(3, dtype=Rs.dtype).expand_as(Rs), atol=1e-9)
    Rb, _ = bd.segment_frames(torch.stack([X, X]), torch.tensor([[0.0, 0.0, 1.0]] * 2, dtype=X.dtype))
    assert Rb.shape == (2, 16, 3, 3)
    d = bd.capsule_distance(torch.tensor([[0.0, 1.0, 0.0], [5.0, 0.0, 0.0]]), torch.zeros(3), torch.tensor([2.0, 0, 0]))
    assert torch.allclose(d, torch.tensor([1.0, 3.0]))


# ------------------------------------------------------------------------------------------------ labels

def test_label_encoding_and_decoding():
    H, W = 20, 30
    lead = np.zeros((H, W), bool)
    lead[2:15, 2:10] = True
    fol = np.zeros((H, W), bool)
    fol[5:18, 8:20] = True
    hair = np.zeros((H, W), bool)
    hair[5:9, 8:20] = True  # partly inside the lead's mask: stays the lead's
    people = np.zeros((H, W), bool)
    people[:, 24:] = True
    valid = np.ones((H, W), bool)
    valid[0, :] = False
    L = dd.encode_labels(lead, fol, hair, people, valid)
    assert L[3, 3] == L_LEAD and L[16, 15] == L_FOLLOW and L[6, 15] == L_HAIR and L[10, 9] == L_BOTH
    assert L[5, 25] == L_IGNORE and L[0, 3] == L_IGNORE and L[12, 22] == L_BG
    assert L[6, 9] == L_BOTH  # hair under the lead: owner unknown, not hair
    m = dd.decode_labels(L)
    assert m["fg"].sum() == (lead | fol)[1:].sum()
    assert not (m["valid"] & (L == L_IGNORE)).any()
    mt = dd.decode_labels(torch.as_tensor(L))  # same on tensors (the trainer)
    assert int(mt["fg"].sum()) == int(m["fg"].sum())


def test_others_ignore_drops_the_ring_around_the_dancers():
    import cv2
    H, W = 120, 160
    dancer = np.zeros((H, W), bool)
    dancer[20:100, 40:70] = True
    spectator = np.zeros((H, W), bool)
    spectator[30:90, 120:140] = True
    people = cv2.dilate((dancer | spectator).astype(np.uint8), np.ones((21, 21), np.uint8)) > 0  # detect.py: +10 px
    ign = dd.others_ignore(people, dancer, 10)
    ring = people & ~dancer & (cv2.dilate(dancer.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0)
    assert not (ign & ring).any()              # the dancer's outline stays visible to the silhouette loss
    assert ign[30:90, 120:140].all()            # the spectator ...
    assert ign[60, 112] and ign[60, 147]        # ... with its margin
    assert not (ign & dancer).any()
    assert dd.others_ignore(None, dancer) is None


def test_crop_and_scale_K_keep_pixel_centres():
    K = np.array([[500.0, 0, 240.0], [0, 500.0, 420.0], [0, 0, 1]])
    Kc = dd.crop_K(K, 100, 50)
    X = np.array([[0.1, -0.2, 3.0]])
    p0, _ = dd.project(X, K, np.eye(3), np.zeros(3))
    p1, _ = dd.project(X, Kc, np.eye(3), np.zeros(3))
    assert np.allclose(p1, p0 - [100, 50])
    Ks = dd.scale_K(Kc, 2.0, 2.0)
    p2, _ = dd.project(X, Ks, np.eye(3), np.zeros(3))
    assert np.allclose(p2, (p1 + 0.5) / 2 - 0.5)
    t = np.array([1.0, 1.1, 1.2])
    k, a = dd.lerp_index(t, 1.15)
    assert k == 1 and abs(a - 0.5) < 1e-9
    assert np.allclose(dd.at_time(np.array([0.0, 1.0, 3.0]), t, 1.15), 2.0)


# ------------------------------------------------------------------------------------------------ model (CPU)

def _synthetic_bundle(Tg: int = 6, hair: bool = True) -> dict:
    """A tiny bundle: a box mesh per dancer skinned to the pelvis / spine, a 55-joint SMPL-X tree, 3 grid frames."""
    rng = np.random.default_rng(5)
    g = np.linspace(-0.2, 0.2, 5)
    pts, faces = [], []
    for axis in range(3):
        for s in (-0.2, 0.2):
            base = len(pts)
            for u in g:
                for v in g:
                    p = [u, v]
                    p.insert(axis, s)
                    pts.append(p)
            for i in range(4):
                for j in range(4):
                    a, b, c, d = base + i * 5 + j, base + i * 5 + j + 1, base + (i + 1) * 5 + j, base + (i + 1) * 5 + j + 1
                    faces += [[a, b, d], [a, d, c]]
    V0 = np.array(pts, float) + [0, 1.0, 0]
    faces = np.array(faces)
    nv = len(V0)
    w_idx = np.zeros((nv, 4), np.int32)
    w_idx[:, 0] = np.where(V0[:, 1] > 1.0, 3, 0)
    w_idx[:, 1:] = [1, 2, 4]
    w_val = np.zeros((nv, 4), np.float32)
    w_val[:, 0] = 1
    t_grid = 10.0 + np.arange(Tg) / 30
    bod = {"t_grid": t_grid, "k_window": np.arange(1, Tg - 1), "faces": faces, "parents": np.array(SMPLX_PARENTS),
           "w_idx": w_idx, "w_val": w_val, "window": np.array([t_grid[1], t_grid[-2]])}
    init = {}
    for i, r in enumerate(("lead", "follow")):
        V = np.stack([V0 + [i * 0.6 + 0.01 * k, 0, 0] for k in range(Tg)]).astype(np.float32)
        Jp = np.zeros((Tg, 55, 3), np.float32)
        Jp[:] = V0.mean(0)
        Jp[:, 15] += [0, 0.5, 0]
        bod.update({f"{r}_verts": V, f"{r}_Rg": np.tile(np.eye(3, dtype=np.float32), (Tg, 55, 1, 1)),
                    f"{r}_Jp": Jp + np.array([i * 0.6, 0, 0], np.float32), f"{r}_v_rest": V0.astype(np.float32),
                    f"{r}_J_rest": Jp[0] * 1, f"{r}_quality": np.ones(Tg), f"{r}_reproj_px": np.zeros(Tg)})
        fi, b = bd.triangle_area_samples(V0, faces, 400, seed=i)
        reg = np.where(rng.random(400) < 0.3, R_SKIN | R_FACE, 0).astype(np.uint8)
        init[r] = {"face": fi.astype(np.int32), "bary": b, "rgb": rng.random((400, 3)).astype(np.float32), "region": reg}
    hr = None
    if hair:
        G, S = 4, 4
        rest = np.zeros((G, S + 1, 3), np.float32)
        for gi in range(G):
            rest[gi] = [[0.05 * gi, 0.1 - 0.05 * s, -0.1] for s in range(S + 1)]
        hr = {"rest_local": rest, "seg_len": np.full((G, S), 0.05, np.float32), "palette_srgb": np.full((8, 3), 0.4),
              "children": np.array(1), "child_spread": np.array(0.005), "width_root": np.array(0.02),
              "width_tip": np.array(0.01), "head_centre_local": np.zeros(3, np.float32), "head_radius": np.array(0.05),
              "colliders": np.array([[9, 12, 0.05, 2]], np.float32), "role": np.array("follow")}
    views = [{"i": j, "cam": f"0{1 + j % 2}", "split": "train", "t": float(t_grid[1 + j % (Tg - 2)]), "frame": j,
              "K": [[300, 0, 80], [0, 300, 60], [0, 0, 1]], "R": np.eye(3).tolist(), "t_w": [0, -1.0, 4.0],
              "W": 160, "H": 120} for j in range(4)]
    return {"bodies": bod, "init": init, "hair": hr, "views": views, "manifest": {"created": "x", "take": "synth"},
            "novel": None}


def _cfg():
    c = preset("smoke")
    c.n_lead = c.n_follow = 300
    c.device = "cpu"
    return c


def test_model_posing_on_cpu():
    from dancecap.dancer_splats.model import AGSModel
    B = _synthetic_bundle()
    m = AGSModel(B, _cfg(), "cpu")
    assert set(m.groups) == {"lead", "follow", "hair"} and len(m.groups["hair"]) == 4 * 4 * 1 * 2
    ts = [10.0 + 1.5 / 30, 10.0 + 3.25 / 30]
    with torch.no_grad():
        Gs = m.gaussians_b(ts)
    N = m.n_gaussians()
    assert Gs["means"].shape == (2, N, 3) and Gs["R"].shape == (2, N, 3, 3) and Gs["opac"].shape == (N,)
    # at init every body Gaussian sits on its anchor (zero tangent / normal offset), residual off
    nb = len(m.groups["lead"]) + len(m.groups["follow"])
    assert torch.allclose(Gs["means"][:, :nb], Gs["anchor"][:, :nb], atol=1e-6)
    # the lerp in time: the synthetic lead drifts +1 cm per grid frame in x
    lead = slice(0, len(m.groups["lead"]))
    dx = (Gs["means"][1, lead, 0] - Gs["means"][0, lead, 0]).mean()
    assert abs(float(dx) - 0.0175) < 1e-4
    # a root translation correction moves the whole dancer by dT (capped)
    with torch.no_grad():
        m.pose_t.data[0, :, 1] = 0.01
        G2 = m.gaussians_b(ts)
    dy = G2["means"][:, lead, 1] - Gs["means"][:, lead, 1]
    exp = m.cfg.pose_cap_m * math.tanh(0.01 / m.cfg.pose_cap_m)
    assert torch.allclose(dy, torch.full_like(dy, exp), atol=1e-6)
    fol = slice(len(m.groups["lead"]), nb)
    assert torch.allclose(G2["means"][:, fol], Gs["means"][:, fol], atol=1e-7)
    # hair roots never move, frozen or not
    with torch.no_grad():
        m.nodes.data += 0.05
        X, _, _ = m.hair_nodes([1, 2])
    assert torch.allclose(X[:, :, 0], (m.h_rest[None, :, 0] + m.Jp["follow"][1:3, 15][:, None]), atol=1e-6)
    single = m.gaussians(ts[0])
    assert single["means"].shape == (N, 3)


def test_model_mcmc_relocation_stays_on_anchors_and_state_roundtrip(tmp_path):
    from dancecap.dancer_splats.model import AGSModel
    B = _synthetic_bundle()
    m = AGSModel(B, _cfg(), "cpu")
    g = m.groups["lead"]
    with torch.no_grad():
        g.p["opac"].data[:50] = -20.0  # dead
    n0 = len(g)
    rep = m.relocate_and_grow(1000)
    assert rep["lead"]["relocated"] == 50 and rep["lead"]["grown"] == int(0.05 * n0)
    assert len(g) == n0 + rep["lead"]["grown"]
    assert all(len(v) == len(g) for v in g.p.values()) and all(len(v) == len(g) for v in g.buf.values())
    assert float(torch.sigmoid(g.p["opac"]).min()) > m.cfg.min_opacity
    b = g.buf["bary"]
    assert torch.allclose(b.sum(1), torch.ones(len(g)), atol=1e-5) and float(b.min()) >= 0
    assert int(g.buf["face"].max()) < len(m.faces)
    # optimisers follow the grown parameters
    loss = sum(p.sum() for p in g.p.values())
    loss.backward()
    g.step()
    p = tmp_path / "model.pt"
    m.save(p, {"train_s": 1.0})
    m2 = AGSModel.load(B, p, "cpu")
    assert {k: len(v) for k, v in m2.groups.items()} == {k: len(v) for k, v in m.groups.items()}
    assert torch.allclose(m2.groups["lead"].p["off"], g.p["off"]) and m2.extra["train_s"] == 1.0


def test_train_regularisers_and_schedule_on_cpu():
    from dancecap.dancer_splats.model import AGSModel
    from dancecap.dancer_splats.train import Schedule, regularisers
    B = _synthetic_bundle()
    m = AGSModel(B, _cfg(), "cpu")
    P = m.posed([1, 2])
    reg = regularisers(m, m.cfg, True, False, P, [1, 2])
    assert {"opac", "scale", "prior", "app", "pose", "resid", "hair_len", "hair_acc", "hair_l2", "hair_col"} <= set(reg)
    assert all(torch.isfinite(v) for v in reg.values())
    sum(reg.values()).backward()
    s = Schedule((100, 200, 50))
    assert s.stage(0) == (0, 0) and s.stage(100) == (1, 0) and s.stage(349) == (2, 49)
    s.shrink(150, 0.5)
    assert s.len[0] == 100 and s.len[1] == 50 + 75 and s.len[2] == 25 and s.stage(149)[0] == 1
    # the hard stop: the schedule ends exactly at the steps done (later stages dropped)
    t = Schedule((100, 200, 50))
    t.truncate(130)
    assert t.len == [100, 30, 0] and t.total == 130
    t = Schedule((100, 200, 50))
    t.truncate(100)
    assert t.len == [100, 0, 0] and t.total == 100


def test_capsule_ends_follow_the_bone_with_their_offsets():
    from dancecap.dancer_splats.model import AGSModel
    B = _synthetic_bundle()
    B["hair"]["colliders"] = np.array([[9, 9, 0.12, 0], [15, 15, 0.07, 0]], np.float32)
    B["hair"]["collider_oa"] = np.array([[0.1, 0.14, 0.0], [0.0, 0.07, -0.03]], np.float32)
    B["hair"]["collider_ob"] = np.array([[-0.1, 0.14, 0.0], [0.0, 0.07, 0.01]], np.float32)
    m = AGSModel(B, _cfg(), "cpu")
    P = m.posed([1, 2], use_delta=False)
    a, b = m.collider_ends([1, 2], P["follow"])
    J = P["follow"]["J"]
    assert torch.allclose(a[:, 0], J[:, 9] + torch.tensor([0.1, 0.14, 0.0]), atol=1e-6)
    assert torch.allclose(b[:, 1], J[:, 15] + torch.tensor([0.0, 0.07, 0.01]), atol=1e-6)
    # a rotated bone carries its offset: 90 deg about y maps +x to -z
    Ry = torch.tensor([[0.0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    m.Rg["follow"][:, 9] = Ry
    a2, _ = m.collider_ends([1, 2], P["follow"])
    assert torch.allclose(a2[:, 0], J[:, 9] + Ry @ torch.tensor([0.1, 0.14, 0.0]), atol=1e-6)
    # with pose corrections the capsule turns with the corrected bone
    with torch.no_grad():
        m.pose_r.data[1, :, 9] = torch.tensor([0.0, 0.03, 0.0])
    Pd = m.posed([1, 2])
    a3, _ = m.collider_ends([1, 2], Pd["follow"])
    Rc = Pd["follow"]["D"][:, 9, :3, :3] @ Ry
    assert torch.allclose(a3[:, 0], Pd["follow"]["J"][:, 9] + (Rc @ torch.tensor([0.1, 0.14, 0.0])), atol=1e-6)
    # an old bundle without offsets: capsules between the joints
    del B["hair"]["collider_oa"], B["hair"]["collider_ob"]
    m0 = AGSModel(B, _cfg(), "cpu")
    a0, b0 = m0.collider_ends([1], m0.posed([1], use_delta=False)["follow"])
    assert torch.allclose(a0[0, 0], m0.Jp["follow"][1, 9]) and torch.allclose(b0[0, 1], m0.Jp["follow"][1, 15])


def _groom_and_rest(y0=-1.3, hj_off=(0.0, 0.0, 0.0)):
    """A sphere 'head' (radius 0.085) on a rest skeleton whose soles are at y0; the groom written like hair_groom.py:
    offsets from the head joint, Unity z flipped, head_joint_rest in the soles-at-0 frame."""
    import scipy.spatial
    rng = np.random.default_rng(0)
    J = np.zeros((55, 3))
    J[15] = [0.01, y0 + 1.6, -0.01]                  # head joint
    J[12] = J[15] + [0, -0.16, -0.01]                 # neck
    J[9] = J[15] + [0, -0.33, 0.02]                   # spine3
    J[6] = J[9] + [0, -0.05, 0]
    c = J[15] + [-0.01, 0.063, -0.011]                # skull centre, 6 cm above the head joint
    u = rng.normal(size=(800, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    head = c + 0.085 * u
    feet = np.array([[0, y0, 0], [0.1, y0, 0], [0, y0, 0.1]])
    v = np.vstack([head, feet])
    faces = scipy.spatial.ConvexHull(head).simplices
    nrm = np.cross(head[faces[:, 1]] - head[faces[:, 0]], head[faces[:, 2]] - head[faces[:, 0]])
    flip = (nrm * (head[faces].mean(1) - c)).sum(1) < 0
    faces[flip] = faces[flip][:, ::-1]
    top = np.nonzero((head[:, 1] - c[1]) > 0.03)[0][:40]
    fz = lambda x: [float(x[0]), float(x[1]), float(-x[2])]  # noqa: E731
    guides = []
    for i, vid in enumerate(top):
        f = np.nonzero((faces == vid).any(1))[0][0]
        tri = faces[f]
        bary = np.array([0.6, 0.2, 0.2])
        pos = (v[tri] * bary[:, None]).sum(0)
        guides.append({"vid": tri.tolist(), "bary": bary.tolist(), "len": 0.45 + 0.002 * i,
                       "pos_unity": fz(pos - J[15])})
    cols = [{"name": "head", "a": 15, "b": 15, "r": 0.07, "oa_unity": fz([0, 0.07, -0.03]),
             "ob_unity": fz([0, 0.07, 0.01])},
            {"name": "neck", "a": 12, "b": 15, "r": 0.05, "oa_unity": [0, 0, 0], "ob_unity": [0, 0, 0]},
            {"name": "shoulders", "a": 9, "b": 9, "r": 0.12, "oa_unity": fz([0.11, 0.14, 0]),
             "ob_unity": fz([-0.11, 0.14, 0])},
            {"name": "midback", "a": 6, "b": 6, "r": 0.09, "oa_unity": fz([0.07, 0, 0]), "ob_unity": fz([-0.07, 0, 0])}]
    groom = {"model": {"head_joint_rest": (J[15] - [0, y0, 0] + np.array(hj_off)).tolist(),
                       "head_sphere_centre_unity": fz(c - J[15]), "head_radius_m": 0.085},
             "strands": {"segments": 16, "children": 3}, "shape": {"standoff_m": 0.01}, "guides": guides,
             "colour": {"strand_palette_linear": [[0.1, 0.05, 0.04]] * 8}, "colliders": cols, "role": "follow"}
    pf = {"v_rest": v.astype(np.float32), "faces": faces.astype(np.int64), "J_rest": J.astype(np.float32)}
    return groom, pf, c, J


def test_hair_rest_uses_the_head_joint_frame_and_hangs_down():
    groom, pf, c, J = _groom_and_rest()
    h = dd.hair_rest(groom, pf)
    # the skull centre: head joint + flip_z(head_sphere_centre_unity), ~6 cm above the head joint (not at the feet)
    assert np.allclose(h["head_centre_local"], c - J[15], atol=1e-5)
    assert 0.04 < h["head_centre_local"][1] < 0.08
    assert float(h["root_check_mm"]) < 1.0
    loc = h["rest_local"]
    # the strands hang: tips well below their roots, inextensible
    drop = loc[:, 0, 1] - loc[:, -1, 1]
    assert (drop > 0.25).all() and np.median(drop) > 0.3
    assert np.allclose(np.linalg.norm(np.diff(loc, axis=1), axis=2), h["seg_len"], atol=1e-5)
    # nothing inside the skull or the capsules (offsets applied, z flipped back) after the rest drape
    X = loc + J[15]
    assert (np.linalg.norm(X[:, 1:] - c, axis=-1) >= 0.085).all()
    cols = dd.groom_colliders(groom, J)
    assert np.allclose(h["collider_oa"][2], [0.11, 0.14, 0]) and np.allclose(h["collider_ob"][0], [0, 0.07, 0.01])
    assert np.allclose(cols["A_rest"][2], J[9] + [0.11, 0.14, 0])
    for i in range(len(cols["r"])):
        P = torch.as_tensor(X[:, 1:].reshape(-1, 3))
        d = bd.capsule_distance(P, torch.as_tensor(cols["A_rest"][i]), torch.as_tensor(cols["B_rest"][i]))
        assert float(d.min()) >= cols["r"][i] - 1e-6, cols["names"][i]


def test_hair_rest_refuses_a_groom_of_another_fit():
    groom, pf, _, _ = _groom_and_rest(hj_off=(0.0, 0.03, 0.0))
    with pytest.raises(ValueError, match="head joint"):
        dd.hair_rest(groom, pf)


# ------------------------------------------------------------------------------------------------ pod job

def test_pod_job_spec(tmp_path, monkeypatch):
    from dancecap import remote_jobs as rj
    from dancecap.dancer_splats import pod
    b = tmp_path / "work" / "synth" / "dancer_splats" / "bundle"
    b.mkdir(parents=True)
    (b / "manifest.json").write_text("{}")
    monkeypatch.setattr(pod, "PACKAGE_ROOT", tmp_path)
    d = pod.spec("synth")
    job = rj.parse_job(d, "test")
    assert job.name == "dancer_splats" and job.needs == ["gsplat"]
    assert rj.needed_features([job]) == ["base", "gsplat", "check"]
    assert "gsplat" in rj.FEATURES and rj.FEATURES[-1] == "check"
    assert not (set(job.needs) & rj.MODEL_FEATURES)  # no SMPL-X / MAMMA models in the bundle
    assert [c.name for c in job.commands] == ["train", "eval"]
    assert "--time-budget-min" in job.commands[0].cmd and job.commands[0].timeout_min <= 30
    assert job.inputs == [("dancecap/work/synth/dancer_splats/bundle/", "dancecap/work/synth/dancer_splats/bundle/")]
    assert all(o.startswith("dancecap/work/synth/") for o in job.outputs)
    assert "dancecap/work/synth/dancer_splats/model.pt" in job.verify and "smplx" not in json.dumps(job.outputs)
    # the trainer's checkpoint comes back even after a failure (a retry resumes from it); no retry inputs by default
    assert "dancecap/work/synth/dancer_splats/cache/" in job.outputs
    assert "cache/ckpt.pt" not in json.dumps(job.verify)
    loc = tmp_path / "work" / "synth" / "dancer_splats"
    (loc / "cache").mkdir()
    (loc / "cache" / "ckpt.pt").write_bytes(b"x")
    assert pod.retry_inputs("synth") == ["dancecap/work/synth/dancer_splats/cache/ckpt.pt"]
    rj_retry = rj.parse_job(pod.spec("synth", retry=True), "test")
    assert ("dancecap/work/synth/dancer_splats/cache/ckpt.pt",) * 2 in [tuple(x) for x in rj_retry.inputs]
    assert pod.spec("synth")["inputs"] == ["dancecap/work/synth/dancer_splats/bundle/"]
    for f in ("model.pt", "pose_delta.npz"):  # train finished, eval failed: the retry uploads the model, not the ckpt
        (loc / f).write_bytes(b"x")
    assert pod.retry_inputs("synth") == ["dancecap/work/synth/dancer_splats/model.pt",
                                         "dancecap/work/synth/dancer_splats/pose_delta.npz"]
    # the budget: the worst case at the A100 SXM community price fits the approved $1.50 per session
    assert 1.39 * (pod.SESSION_MAX_MINUTES + 6) / 60 <= pod.MAX_COST <= 1.5
    assert job.max_minutes + 10 <= pod.SESSION_MAX_MINUTES
    assert d["max_cost_usd"] == 1.5 and "--max-cost 1.50" in d["run"] and "--jobs dancer_splats" in d["run"]
    # the bundle stays small: MAMMA code and the mamma / dancers wheels are excluded
    ex = tuple(job.exclude)
    assert rj._excluded("mamma-turing/ma_3d/run.py", ex)
    assert rj._excluded("dancecap/work/runpod/wheelhouse/x/pytorch_sdf-0.1.0-cp311-cp311-linux_x86_64.whl", ex)
    assert rj._excluded("dancecap/work/runpod/pywheels/cp311-linux/opencv_python-4.11.0.86-cp37-abi3-x.whl", ex)
    assert not rj._excluded("dancecap/work/runpod/pywheels/cp311-linux/opencv_python_headless-4.11.0.86-x.whl", ex)
    assert not rj._excluded("dancecap/work/runpod/wheelhouse/x/gsplat-1.5.3-cp311-cp311-linux_x86_64.whl", ex)
    assert not rj._excluded("dancecap/work/synth/dancer_splats/bundle/images/0000.jpg", ex)


def test_bootstrap_has_the_gsplat_feature():
    sh = (Path(__file__).resolve().parents[1] / "remote" / "bootstrap.sh").read_text(encoding="utf-8")
    assert "feature_gsplat()" in sh and 'install_wheel_or_build gsplat' in sh
    from dancecap import remote_jobs as rj
    assert rj.BOOTSTRAP_MIN["gsplat"] > rj.BOOTSTRAP_MIN_CACHED["gsplat"]


def test_uploads_in_flight(tmp_path, monkeypatch):
    from dancecap import remote_jobs as rj
    from dancecap.dancer_splats import pod
    monkeypatch.setattr(rj, "SESSIONS", tmp_path)
    s1 = tmp_path / "20261007-175818-front-full"
    s1.mkdir()
    (s1 / "runner.json").write_text(json.dumps({"pid": os.getpid()}))
    (s1 / "runner.log").write_text("[pod-runner 18:01:48] upload ...\n")
    assert pod.uploads_in_flight() == [s1.name]
    (s1 / "runner.log").write_text("[pod-runner 18:01:48] upload ...\n[pod-runner 18:20:00] upload done in 1092.0s\n")
    assert pod.uploads_in_flight() == []
    (s1 / "runner.log").write_text("[pod-runner 18:01:48] upload ...\n")
    (s1 / "runner.json").write_text(json.dumps({"pid": 0}))  # runner gone: not uploading
    assert pod.uploads_in_flight() == []


# ------------------------------------------------------------------------------------------------ progress

def test_progress_lines_are_parsed():
    from dancecap import progress as P
    from dancecap.pipeline import runpod_view
    lp = P.LineParser()
    ev = lp.feed("[dancer_splats] step 250/25000 stage S1 psnr 18.20 iou 0.810 N_gauss 170000 14s")
    tick = [e for e in ev if e["kind"] == "tick"][0]
    assert tick["step"] == "splats/train" and tick["done"] == 250 and tick["total"] == 25000
    assert tick["extra"]["stage"] == "S1" and tick["extra"]["psnr"] == 18.2 and tick["extra"]["elapsed_s"] == 14
    end = lp.feed("[dancer_splats] step 25000/25000 stage S3 psnr 28.00 iou 0.950 N_gauss 170500 1200s")
    assert end[-1]["kind"] == "end"
    ev = lp.feed("[dancer_splats] eval 25/175 views")
    assert ev[0]["step"] == "splats/eval" and ev[0]["done"] == 25
    pr = runpod_view.parse_runner_log("[dancer_splats] step 250/25000 stage S1 psnr 18.20 iou 0.810 N_gauss 170000 14s")
    assert pr["progress"]["done"] == 250 and pr["progress"]["total"] == 25000 and "S1" in pr["progress"]["step"]


def test_eta_of_a_planned_session():
    from dancecap import progress_eta as pe
    from dancecap.dancer_splats import pod
    job = {"name": "dancer_splats", "commands": [
        {"name": "train", "cmd": "python -m dancecap.dancer_splats synth train --preset pod", "timeout_min": 30},
        {"name": "eval", "cmd": "python -m dancecap.dancer_splats synth eval --preset pod", "timeout_min": 10}],
        "setup": ["pip install -q --no-deps lpips==0.1.4 || true"]}
    est = pe.estimate_session([job], 255e6, ["base", "gsplat", "check"], 1.39, gpu="NVIDIA A100-SXM4-80GB",
                              rates={"keys": {}})
    ph = {r["phase"]: r for r in est["phases"]}
    assert 15 < ph["dancer_splats/train"]["p50_min"] < 30  # 25k steps x the seed rate
    assert 0 < ph["dancer_splats/eval"]["p50_min"] < 10
    assert est["p50_min"] < pod.SESSION_MAX_MINUTES and est["p50_cost"] < pod.MAX_COST
    assert pe.gpu_command("dancer_splats")


# ------------------------------------------------------------------------------------------------ pipeline

def test_registry_stage_and_session():
    from dancecap.pipeline import registry
    s = registry.get("dancer_splats")
    assert s["implemented"] and s["runnable"] == "session" and s["runs_on"] == "pod" and s["session"] == "S_splats"
    assert s["deps"] == ["fit_qa", "texture", "hair_groom"] and s["soft_deps"] == ["face_pod"]
    assert not {"smplx_detail", "hands_build", "hands_pod", "hands_survey"} & set(s["deps"] + s["soft_deps"])
    assert "smplx_detail" not in json.dumps(s)
    assert set(s["done_when"]) == {"dancer_splats/model.pt", "dancer_splats/metrics.json", "dancer_splats/pose_delta.npz"}
    assert s["gate_human"] and s["progress"]["pattern"].startswith("^\\[dancer_splats\\] step")
    sp = registry.POD_SESSIONS["S_splats"]
    assert sp["max_cost"] == 1.5 and sp["max_minutes"] == 55 and sp["fixed_budget"] and sp["optional"]
    (argv,) = registry.session_command("S_splats", ["dancer_splats"], {"take": "abc", "max_cost": 1.5,
                                                                      "max_minutes": 55})
    assert argv[argv.index("--jobs") + 1] == "dancer_splats" and argv[argv.index("--max-cost") + 1] == "1.50"
    assert registry.session_prep("S_splats", ["dancer_splats"], {"take": "abc"}) == [
        ["-m", "dancecap.dancer_splats", "abc", "prepare"], ["-m", "dancecap.dancer_splats", "abc", "write-pod-job"]]
    assert registry.stages_for_remote_job("dancer_splats") == ["dancer_splats"]
    assert "dancer_splats" in {x["id"] for x in registry.variant_stages("multi")}
    assert any(x["id"] == "S_splats" for x in registry.registry()["sessions"])


def test_plan_prices_the_splat_session_at_the_approved_ceiling(tmp_path):
    from _pipeline_fixtures import make_root, make_take

    from dancecap.pipeline import paths, plan
    root = make_root(tmp_path)
    try:
        make_take(root)
        p = plan.make_plan("synth", {"target": "dancer_splats", "scope": "only", "include_optional": ["dancer_splats"]})
        step = next(st for st in p["steps"] if st["stage"] == "dancer_splats")
        assert step["action"] == "run" and step["session"] == "S_splats"
        (sess,) = p["sessions"]
        assert sess["id"] == "S_splats" and sess["jobs"] == ["dancer_splats"]
        assert sess["max_minutes"] == 55 and sess["max_cost_usd"] == 1.5
        assert sess["est_min"]["p90"] <= 55 and sess["cost_usd"]["p90"] <= 1.5
        assert sess["upload_bytes"] == 260_000_000  # fixed bundle, not per camera-minute
        assert "dancer_splats" in p["review_gates"]
        argv = step["argv_display"][0]
        assert "--jobs" in argv and "dancer_splats" in argv
    finally:
        paths.set_root(None)


def test_config_presets():
    c = preset("pod")
    assert c.window == (58.0, 61.5) and c.holdout_cams == ("03",) and sum(c.iters) == 25000 and c.time_budget_min == 24
    s = preset("smoke")
    assert sum(s.iters) <= 400 and s.time_budget_min <= 2
    assert isinstance(SplatConfig().init_samples, int)


def test_eval_crops_and_tables():
    from dancecap.dancer_splats import evaluate as ev
    v = {"K": [[500, 0, 100], [0, 500, 100], [0, 0, 1]], "R": np.eye(3).tolist(), "t_w": [0, 0, 0], "W": 200, "H": 200}
    box = ev.crop_box(v, np.array([0.0, 0.0, 2.0]), 0.16)  # 32 cm at 2 m, f 500 -> 80 px square around the centre
    assert box == (60, 60, 140, 140)
    assert ev.crop_box(v, np.array([0.0, 0.0, -2.0]), 0.16) is None  # behind the camera
    gt = np.random.default_rng(0).random((200, 200, 3)).astype(np.float32)
    sel = np.zeros((200, 200), bool)
    sel[60:140, 60:140] = True
    s = ev.image_metrics(gt, gt, sel)
    assert s["psnr"] > 90 and abs(s["ssim"] - 1) < 1e-4
    rows = [{"split": "test_cam", "cam": "03", "splat_psnr_cc": 25.0, "avatar_psnr_cc": 20.0, "splat_lead_psnr_cc": 26.0,
             "ghost_frac": 0.0},
            {"split": "test_time", "cam": "02", "crop_face_follow_splat_psnr_cc": 22.0,
             "crop_face_follow_avatar_psnr_cc": 18.0, "crop_face_follow_px_side": 40, "ghost_frac": 0.001}]
    t = ev.compact_tables(ev.summarise(rows))
    assert t["splat_vs_avatar"]["heldout_cam"]["splat"]["psnr_cc"] == 25.0
    assert t["splat_vs_avatar"]["heldout_cam"]["splat_lead"]["psnr_cc"] == 26.0
    assert t["crops"]["cam_02"]["face_follow"]["avatar"]["psnr_cc"] == 18.0
