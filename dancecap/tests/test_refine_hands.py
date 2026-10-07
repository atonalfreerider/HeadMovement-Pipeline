"""Hand stages: fast SMPL-X kinematics vs smplx, robust hand triangulation, grid resampling of detections, contact
episode classification, refinement helpers (model-dependent tests skip when the SMPL-X model is not installed).

    .venv-nopo4d\\Scripts\\python.exe -m pytest dancecap/tests/test_refine_hands.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dancecap import face_common as fc  # noqa: E402
from dancecap import hand_kin as hk  # noqa: E402
from dancecap import refine_hands as rh  # noqa: E402

needs_model = pytest.mark.skipif(not fc.smplx_model_npz().exists(), reason="SMPL-X model not installed")


def _model_dir():
    return fc.smplx_model_npz().parent.parent


def _random_pose(T, rng, scale=0.3):
    P = rng.normal(scale=scale, size=(T, 55, 3))
    P[:, 0] = rng.normal(scale=1.0, size=(T, 3))
    return P


def test_hand21_layout():
    for s in hk.SIDES:
        idx = hk.HAND21[s]
        assert len(idx) == 21 and len(set(idx)) == 21
        assert idx[0] == hk.WRIST[s]
        assert all(i >= 66 for i in (idx[4], idx[8], idx[12], idx[16], idx[20]))   # fingertips are extra joints
    assert len(hk.HAND_EDGES) == 20


@needs_model
def test_fk_matches_smplx():
    import smplx
    import torch
    rng = np.random.default_rng(0)
    T = 4
    betas = rng.normal(scale=0.5, size=16)
    P = _random_pose(T, rng)
    tr = rng.normal(size=(T, 3))
    kin = hk.SmplxKin(_model_dir(), betas)
    out = kin.forward(torch.tensor(P), torch.tensor(tr))
    m = smplx.create(str(_model_dir()), model_type="smplx", gender="neutral", ext="npz", use_pca=False,
                     flat_hand_mean=True, num_betas=16, batch_size=T)
    f = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)
    o = m(betas=f(betas)[None].expand(T, -1), global_orient=f(P[:, 0]), body_pose=f(P[:, 1:22].reshape(T, 63)),
          jaw_pose=f(P[:, 22]), leye_pose=f(P[:, 23]), reye_pose=f(P[:, 24]),
          left_hand_pose=f(P[:, 25:40].reshape(T, 45)), right_hand_pose=f(P[:, 40:55].reshape(T, 45)), transl=f(tr))
    J = o.joints.detach().numpy().astype(float)
    assert np.abs(out["joints"].numpy() - J[:, :55]).max() < 2e-4
    for s in hk.SIDES:
        assert np.abs(kin.hand21(out, s).numpy() - J[:, list(hk.HAND21[s])]).max() < 3e-4
    # LBS of a vertex subset equals the smplx surface (pose correctives included)
    ids = np.array([0, 100, 4933, 7669, 8079])
    kin.set_subset(ids)
    out = kin.forward(torch.tensor(P), torch.tensor(tr), want_verts=True)
    assert np.abs(out["verts"].numpy() - o.vertices.detach().numpy()[:, ids]).max() < 3e-4


@needs_model
def test_fast_hand_fk_equals_full():
    import torch
    rng = np.random.default_rng(1)
    T = 3
    kin = hk.SmplxKin(_model_dir(), rng.normal(scale=0.5, size=10))
    P = torch.tensor(_random_pose(T, rng))
    tr = torch.tensor(rng.normal(size=(T, 3)))
    out = kin.forward(P, tr)
    for s, f0 in (("left", 25), ("right", 40)):
        base = kin.hand_base(out, s)
        P2 = P.clone()
        P2[:, f0:f0 + 15] += 0.2
        P2[:, hk.WRIST[s]] -= 0.15
        full = kin.hand21(kin.forward(P2, tr), s)
        fast = kin.hand_fk(base, P2[:, hk.WRIST[s]], P2[:, f0:f0 + 15])
        assert float((full - fast).abs().max()) < 1e-6


def _cams(n=5, seed=0):
    rng = np.random.default_rng(seed)
    Ps = []
    for k in range(n):
        a = 2 * np.pi * k / n
        C = np.array([3 * np.cos(a), 1.5 + rng.normal(scale=0.1), 3 * np.sin(a)])
        z = -C / np.linalg.norm(C)
        x = np.cross(z, [0, 1, 0])
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        R = np.stack([x, y, z])
        K = np.array([[900.0, 0, 480], [0, 900, 640], [0, 0, 1]])
        Ps.append(K @ np.concatenate([R, (-R @ C)[:, None]], 1))
    return np.array(Ps)


def test_triangulate_hand_rejects_outlier_view():
    rng = np.random.default_rng(2)
    X = rng.normal(scale=0.05, size=(21, 3)) + [0, 1.2, 0]
    Ps = _cams(5)
    x = np.stack([rh._proj(P, X) for P in Ps]) + rng.normal(scale=0.5, size=(5, 21, 2))
    x[3] += 40.0                                    # one view detected the partner's hand
    w = np.ones((5, 21))
    L = np.array([np.linalg.norm(rh._proj(P, X[[0]]) - rh._proj(P, X[[12]])) for P in Ps])
    Xt, inl = rh.triangulate_hand(Ps, x, w, np.maximum(L, 30.0), rh.DetConfig())
    assert Xt is not None and not inl[3] and inl.sum() == 4
    assert np.abs(Xt - X).max() < 0.01


def test_to_grid_interpolates_and_drops_far_frames():
    from types import SimpleNamespace
    t = np.array([0.0, 1 / 30, 2 / 30, 3 / 30])
    tr = SimpleNamespace(t=t, frame_dt=1 / 30, k1=np.zeros(4), K=np.tile(np.eye(3), (4, 1, 1)))
    kp = np.zeros((2, 21, 2))
    kp[0] += 10.0
    kp[1] += 20.0
    nat = {("rtmh", "lead", "left", "01"): {"i": np.array([0, 1]), "kp": kp, "w": np.ones((2, 21))}}
    grid = np.array([0.5 / 30, 2.95 / 30])
    G = rh.to_grid(nat, {"01": tr}, grid)[("rtmh", "lead", "left", "01")]
    assert np.allclose(G[0][0], 15.0) and np.allclose(G[1][0], 1.0)
    assert np.allclose(G[1][1], 0.0)                 # nearest detection is > 1 frame away


def test_contact_episodes_hold_and_near_miss():
    T = 120
    t = 56 + np.arange(T) / 30
    gap = np.full(T, 0.12)
    gap[10:60] = -0.005                             # a 1.7 s hold
    gap[80:90] = 0.04                               # a close pass, never touching
    geo = {("lead", "left"): {"gap": gap, "pen": np.maximum(0, -gap), "region": np.array(["right hand"] * T)}}
    verdict = np.zeros((T, 4), np.int8)
    verdict[10:60] = 1
    verdict[80:90] = [1, -1, -1, 1]
    ev = {"verdict": {("lead", "left"): verdict}}
    eps, series = rh.contact_episodes(geo, ev, t, rh.ContactConfig())
    holds = [e for e in eps if e["kind"] == "hold"]
    near = [e for e in eps if e["kind"] == "near-miss"]
    assert len(holds) == 1 and holds[0]["i0"] <= 11 and holds[0]["i1"] >= 58 and holds[0]["hand_hand"]
    assert len(near) == 1 and 78 <= near[0]["i0"] <= 82


def test_ramp_window():
    k, w = rh._ramp(100, 20, 30, 6)
    assert k[0] == 14 and k[-1] == 36
    assert np.allclose(w[(k >= 20) & (k <= 30)], 1.0)
    assert 0 < w[0] < 0.2 and 0 < w[-1] < 0.2 and np.all(np.diff(w[:7]) > 0)


def test_torch_normals_cube_outward():
    import torch
    V = torch.tensor([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
                     dtype=torch.float64)[None] - 0.5
    F = torch.tensor([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
                      [1, 2, 6], [1, 6, 5], [0, 4, 7], [0, 7, 3]])
    N = rh._torch_normals(V, F)[0].numpy()
    assert np.all((N * V[0].numpy()).sum(1) > 0.5)
