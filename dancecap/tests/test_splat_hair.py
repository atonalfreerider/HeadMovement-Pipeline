r"""Unit tests for the pure helpers of dancecap.mono4d.splat_hair (synthetic strands and Gaussians; no avatar, no footage,
no SMPL-X model file).

    .venv-nopo4d\Scripts\python.exe -m pytest dancecap/tests/test_splat_hair.py -q      (from the workspace root)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dancecap.mono4d import splat_hair as SH  # noqa: E402


def rand_rot(rng, n=None):
    v = rng.normal(size=(3,) if n is None else (n, 3))
    return SH.rotvec_to_mat(v)


def hanging_strands(S=3, N=8, L=0.35, root=(0.0, 0.40, -0.10)):
    """S straight strands hanging down from points near `root`, N points each, length L."""
    out = []
    for s in range(S):
        r0 = np.asarray(root, float) + np.array([0.01 * (s - (S - 1) / 2), 0.0, 0.0])
        out.append(r0 + np.linspace(0, L, N)[:, None] * np.array([0.0, -1.0, 0.0]))
    return np.stack(out)


def test_quat_mat_roundtrip_and_product():
    rng = np.random.default_rng(0)
    R = rand_rot(rng, 50)
    q = SH.mat_to_quat(R)
    assert np.allclose(np.linalg.norm(q, axis=1), 1.0)
    assert np.allclose(SH.quat_to_mat(q), R, atol=1e-10)
    R2 = rand_rot(rng, 50)
    q12 = SH.quat_mul(SH.mat_to_quat(R), SH.mat_to_quat(R2))
    assert np.allclose(SH.quat_to_mat(q12), R @ R2, atol=1e-10)


def test_min_rotation_maps_a_to_b_including_opposite():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(40, 3))
    b = rng.normal(size=(40, 3))
    b[0] = -a[0]
    b[1] = a[1]
    R = SH.min_rotation(a, b)
    an = a / np.linalg.norm(a, axis=1, keepdims=True)
    bn = b / np.linalg.norm(b, axis=1, keepdims=True)
    assert np.allclose(np.einsum("nij,nj->ni", R, an), bn, atol=1e-9)
    assert np.allclose(np.linalg.det(R), 1.0)
    assert np.allclose(R[1], np.eye(3), atol=1e-9)


def test_resample_polyline_even_spacing_and_endpoints():
    P = np.array([[0, 0, 0], [1, 0, 0], [1, 2, 0.0]])
    Q = SH.resample_polyline(P, 7)
    d = np.linalg.norm(np.diff(Q, axis=0), axis=1)
    assert np.allclose(Q[0], P[0]) and np.allclose(Q[-1], P[-1])
    assert np.allclose(d, 0.5)


def test_bind_then_pose_at_rest_is_identity():
    rng = np.random.default_rng(2)
    st = hanging_strands()
    x = st[rng.integers(0, 3, 200), rng.integers(1, 7, 200)] + rng.normal(scale=0.01, size=(200, 3))
    q = SH.mat_to_quat(rand_rot(rng, 200))
    b = SH.bind(x, q, st)
    m, qq = SH.pose_bound(st, np.eye(3), st, b)
    assert np.allclose(m, x, atol=1e-12)
    assert np.allclose(np.abs((qq * q).sum(1)), 1.0, atol=1e-10)


def test_pose_rigid_motion_moves_gaussians_rigidly():
    rng = np.random.default_rng(3)
    st = hanging_strands()
    x = st[:, 3] + rng.normal(scale=0.01, size=(3, 3))
    q = SH.mat_to_quat(rand_rot(rng, 3))
    b = SH.bind(x, q, st)
    R = rand_rot(rng)
    t = np.array([0.3, -0.2, 1.0])
    track = st @ R.T + t
    m, qq = SH.pose_bound(track, R, st, b)
    assert np.allclose(m, x @ R.T + t, atol=1e-10)
    want = SH.quat_mul(SH.mat_to_quat(R), q)
    assert np.allclose(np.abs((qq * want).sum(1)), 1.0, atol=1e-10)
    # cached variant (pod side) agrees
    Fr = SH.segment_frames(st)[b["strand"], b["segment"]]
    m2, q2 = SH.pose_bound_cached(track, R, st, Fr, b)
    assert np.allclose(m2, m) and np.allclose(q2, qq)


def _sim(st, F, headR, headP, params=None, caps=None, links=None):
    E = np.zeros((F, 0, 2, 3)) if caps is None else caps[0]
    r = np.zeros(0) if caps is None else caps[1]
    first = np.zeros(0, int) if caps is None else caps[2]
    return SH.simulate(st, np.zeros(3), headR, headP, E, r, first, "ponytail", params, links=links)


def test_static_hanging_strand_stays_put_and_is_deterministic():
    st = hanging_strands()
    F = 40
    R = np.repeat(np.eye(3)[None], F, 0)
    P = np.zeros((F, 3))
    a = _sim(st, F, R, P)["pos"]
    b = _sim(st, F, R, P)["pos"]
    assert np.array_equal(a, b)                                   # deterministic
    assert np.abs(a[-1] - st).max() < 2e-3                         # hangs where it rests (straight down)


def test_spinning_head_swings_tip_out_and_keeps_lengths():
    st = hanging_strands(S=3, N=10, L=0.4)
    F = 60
    ang = np.radians(400.0) * np.arange(F) / 30.0                  # a 400 deg/s spin about the vertical axis
    R = np.stack([SH.rotvec_to_mat(np.array([0.0, a_, 0.0])) for a_ in ang])
    P = np.zeros((F, 3))
    links = np.array([[0, 1], [0, 2], [1, 2]])
    res = _sim(st, F, R, P, links=links)
    pos = res["pos"]
    seg0 = np.linalg.norm(np.diff(st, axis=1), axis=2)
    seg = np.linalg.norm(np.diff(pos, axis=2), axis=3)
    assert np.abs(seg / seg0[None] - 1).max() <= 0.0101           # inextensible (max_stretch 1.01)
    car = np.einsum("fij,snj->fsni", R, st)
    tip = np.linalg.norm(pos[:, :, -1] - car[:, :, -1], axis=-1)
    assert tip[-20:].mean() > 0.05                                 # the tail lags / flies out in the spin
    r_tip = np.linalg.norm(pos[-1, :, -1, [0, 2]].T - 0.0, axis=-1)
    assert r_tip.mean() > np.linalg.norm(st[:, -1, [0, 2]], axis=-1).mean()   # centrifugal: tip moves outward
    assert np.isfinite(pos).all() and np.abs(np.diff(pos, axis=0)).max() * 30 < 20.0   # no explosion


def test_capsule_collision_keeps_points_outside():
    st = hanging_strands(S=1, N=8, L=0.35, root=(0.0, 0.40, 0.0))
    F = 30
    R = np.repeat(np.eye(3)[None], F, 0)
    P = np.zeros((F, 3))
    E = np.zeros((F, 1, 2, 3))
    E[:, 0, 0] = [-0.2, 0.2, -0.06]                                 # a bar just behind the strand, moving through it
    E[:, 0, 1] = [0.2, 0.2, -0.06]
    E[:, 0, :, 2] += np.linspace(0, 0.08, F)[:, None]
    res = _sim(st, F, R, P, caps=(E, np.array([0.04]), np.array([2])))
    pos = res["pos"]
    cp, _ = SH.segment_closest(pos[:, 0, 2:], E[:, 0, 0][:, None], E[:, 0, 1][:, None])
    d = np.linalg.norm(pos[:, 0, 2:] - cp, axis=-1)
    assert d.min() > 0.04 - 0.005


def test_pod_hair_roundtrip(tmp_path):
    rng = np.random.default_rng(5)
    st = hanging_strands()
    n_all = 300
    cano = rng.normal(scale=0.05, size=(n_all, 3)) + np.array([0, 0.45, 0.0])
    free = np.arange(200, 260)
    cano[free] = st[rng.integers(0, 3, len(free)), rng.integers(1, 7, len(free))] + rng.normal(scale=0.008, size=(len(free), 3))
    quat = SH.mat_to_quat(rand_rot(rng, n_all))
    core = np.arange(0, 120)
    b = SH.bind(cano[free], quat[free], st)
    np.savez(tmp_path / "follow_hair.npz", version=1, free=free, strands_rest=st, strand=b["strand"], segment=b["segment"],
             local_offset=b["local_offset"], local_rotation=b["local_rotation"], head_core=core, free_xyz=cano[free],
             free_quat=quat[free], head_joint=np.zeros(3), betas=np.zeros(10))
    R = rand_rot(rng)
    t = np.array([0.1, 0.9, -0.3])
    swing = st.copy()
    swing[:, 4:] += np.array([0.03, 0.0, 0.02])                    # some swing in head-local coordinates
    np.savez(tmp_path / "follow_tracks.npz", keys=np.array([7]), tracks_local=swing[None], head_R=R[None], head_P=t[None])
    (tmp_path / "hair.json").write_text(json.dumps({"roles": {"follow": {"bundle": "follow_hair.npz",
                                                                         "tracks": "follow_tracks.npz"}},
                                                    "groups": ["g_swing"]}))
    ph = SH.PodHair(tmp_path)
    rep = ph.attach("follow", cano, quat)
    assert rep["cano_vs_bundle_mm"]["max"] < 1e-6
    assert ph.has("follow", 7) and not ph.has("follow", 8) and not ph.has("lead", 7)
    posed = cano @ R.T + t
    fidx, m, q = ph.pose("follow", 7, posed[ph.core_indices("follow")])
    m_ref, q_ref = SH.pose_bound(swing @ R.T + t, R, st, b)
    assert np.array_equal(fidx, free)
    assert np.allclose(m, m_ref, atol=1e-9) and np.allclose(np.abs((q * q_ref).sum(1)), 1, atol=1e-9)
    s = ph.summary()["roles"]["follow"]["kabsch_vs_fk"]
    assert s["angle_deg_max"] < 1e-4 and s["head_joint_mm_max"] < 1e-6


def test_render_check_single_gaussian_lands_on_its_pixel():
    pytest.importorskip("torch")
    K = np.array([[100.0, 0, 32], [0, 100.0, 24], [0, 0, 1]])
    img = SH.render_check(np.array([[0.1, 0.0, 2.0]]), np.array([[1.0, 0, 0, 0]]), np.full((1, 3), 0.02), np.array([0.95]),
                          np.array([[1.0, 0.0, 0.0]]), np.eye(3), np.zeros(3), K, 64, 48, bg=(0, 0, 0))
    yx = np.unravel_index(img[:, :, 0].argmax(), img.shape[:2])
    assert yx == (24, 37) and img[24, 37, 0] > 200 and img[24, 37, 1] < 10


def test_track_key_map_aliases_blend_rows():
    tr = {"keys": np.array([10, 11, 12]), "alias_keys": np.array([3, 4]), "alias_row": np.array([0, 1]),
          "alias_w": np.array([0.25, 0.5])}
    m = SH._track_key_map(tr)
    assert m[10] == (0, 0.0) and m[12] == (2, 0.0)
    assert m[3] == (0, 0.25) and m[4] == (1, 0.5)
