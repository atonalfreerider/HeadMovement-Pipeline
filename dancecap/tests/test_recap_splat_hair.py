r"""Unit tests for the pure helpers of dancecap.mono4d.recap_splat_hair (the recap follow's hair Gaussians bound to the
Unity hair guides): segment frames, binding round trip, the posing reference (rest = head-rigid, re-anchoring, rigid
swing, lever damping, attach blend). Synthetic guides and Gaussians only (no avatar, no Unity, no SMPL-X model file).

    .venv-nopo4d\Scripts\python.exe -m pytest dancecap/tests/test_recap_splat_hair.py -q      (from the workspace root)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dancecap.mono4d import recap_splat_hair as RH  # noqa: E402
from dancecap.mono4d import splat_hair as SH  # noqa: E402


def guides(G=3, P=6):
    """head-bone rest guides (Unity coords): roots over the head, hanging down and back with a bend"""
    r = np.zeros((G, P, 3))
    for g in range(G):
        for i in range(P):
            r[g, i] = [0.03 * (g - 1), 0.08 - 0.04 * i, 0.04 + 0.012 * i + (0.02 * (i - 2) if i > 2 else 0.0)]
    return r


def rand_q(rng, n):
    q = rng.normal(size=(n, 4))
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def hair(rest, rng, n=300, spread=0.01):
    G, P, _ = rest.shape
    g = rng.integers(0, G, n)
    s = rng.integers(0, P - 1, n)
    t = rng.uniform(0, 1, n)
    c = rest[g, s] + t[:, None] * (rest[g, s + 1] - rest[g, s]) + rng.uniform(-spread, spread, (n, 3))
    return c, rand_q(rng, n)


def axis_angle_q(axis, deg):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    h = np.radians(deg) / 2
    return np.concatenate([axis * np.sin(h), [np.cos(h)]])


def test_segment_frames_are_rotations_with_tangent_y():
    r = guides()
    F = RH.segment_frames(r)
    assert F.shape == (3, 5, 3, 3)
    assert np.allclose(np.einsum("...ji,...jk->...ik", F, F), np.eye(3), atol=1e-12)
    assert np.allclose(np.linalg.det(F), 1.0)
    t = np.diff(r, axis=1)
    t /= np.linalg.norm(t, axis=-1, keepdims=True)
    assert np.allclose(F[..., :, 1], t)
    # a tangent along her forward axis (Unity -Z) falls back to Unity +Y for Z
    F2 = RH.segment_frames(np.array([[[0, 0, 0], [0, 0, -0.03]]], float))
    assert np.allclose(F2[0, 0][:, 2], [0, 1, 0])


def test_quaternion_helpers():
    rng = np.random.default_rng(1)
    q = rand_q(rng, 40)
    R = RH.q_to_mat(q)
    assert np.allclose(np.abs((RH.q_from_mat(R) * q).sum(1)), 1.0)
    v = rng.normal(size=(40, 3))
    assert np.allclose(RH.q_rotate(q, v), np.einsum("nij,nj->ni", R, v))
    q2 = rand_q(rng, 40)
    assert np.allclose(RH.q_to_mat(RH.q_mul(q, q2)), R @ RH.q_to_mat(q2))


def test_bind_round_trip_and_nearest_segment():
    rng = np.random.default_rng(2)
    r = guides()
    c, q = hair(r, rng)
    b = RH.bind(c, q, r)
    F = RH.segment_frames(r)[b["strand"], b["segment"]]
    back = r[b["strand"], b["segment"]] + np.einsum("nij,nj->ni", F, b["local_offset"])
    assert np.allclose(back, c, atol=1e-12)
    qb = RH.q_mul(RH.q_from_mat(F), b["local_rotation"])
    assert np.allclose(np.abs((qb * q).sum(1)), 1.0)
    # the chosen segment is the nearest one
    A, B = r[:, :-1].reshape(-1, 3), r[:, 1:].reshape(-1, 3)
    cp, _ = SH.segment_closest(c[:, None], A[None], B[None])
    d = np.linalg.norm(c[:, None] - cp, axis=-1).min(1)
    assert np.allclose(b["distance"], d)


def _pose(c, q, b, r, sim, hs_p, hs_q, hn_p, hn_q):
    return RH.pose_reference(c, q, b, r, r, sim, hs_p, hs_q, hn_p, hn_q, attach={"y_top_m": 0.5, "width_m": 0.05})


def test_pose_at_rest_is_head_rigid_and_reanchored():
    rng = np.random.default_rng(3)
    r = guides()
    c, q = hair(r, rng)
    b = RH.bind(c, q, r)
    hs_q, hn_q = axis_angle_q([0.3, 1, 0.2], 40), axis_angle_q([-0.2, 1, 0.5], -70)
    hs_p, hn_p = np.array([1.0, 1.5, 2.0]), np.array([-0.5, 1.4, 0.3])
    sim = hs_p + RH.q_rotate(np.broadcast_to(hs_q, r.shape[:-1] + (4,)), r)
    pos, rot = _pose(c, q, b, r, sim, hs_p, hs_q, hn_p, hn_q)
    expect = hn_p + RH.q_rotate(np.broadcast_to(hn_q, c.shape[:-1] + (4,)), c)
    assert np.abs(pos - expect).max() < 1e-9
    qe = RH.q_mul(np.broadcast_to(hn_q, q.shape), q)
    assert np.allclose(np.abs((rot * qe).sum(1)), 1.0)


def test_rigid_swing_about_x_carries_near_hair_and_lever_damps_far_hair():
    rng = np.random.default_rng(4)
    r = guides()
    c, q = hair(r, rng, n=400, spread=0.008)                # |across| <= 1.4 cm < lever near: turns fully
    far = r[1, 3] + np.array([0.0, 0.0, -0.10])             # 10 cm in front of guide 1 (beyond lever far)
    c = np.vstack([c, far[None]])
    q = np.vstack([q, [[0, 0, 0, 1.0]]])
    b = RH.bind(c, q, r)
    R = axis_angle_q([1, 0, 0], 70)                         # perpendicular to every tangent (x = const per guide)
    root = r[1, 0]
    sim = r.copy()
    sim[1] = root + RH.q_rotate(np.broadcast_to(R, (r.shape[1], 4)), r[1] - root)
    I = np.array([0, 0, 0, 1.0])
    pos, _ = _pose(c, q, b, r, sim, np.zeros(3), I, np.zeros(3), I)
    on1 = (b["strand"] == 1)[:-1]
    expect = root + RH.q_rotate(np.broadcast_to(R, (on1.sum(), 4)), c[:-1][on1] - root)
    assert np.abs(pos[:-1][on1] - expect).max() < 1e-9
    # the far Gaussian follows its guide point's displacement (head rotation = identity), not the 70 deg lever arm
    k = b["segment"][-1]
    s = b["strand"][-1]
    t = (r[s, k + 1] - r[s, k]) / np.linalg.norm(r[s, k + 1] - r[s, k])
    along = np.dot(c[-1] - r[s, k], t) * t
    perp = c[-1] - r[s, k] - along
    assert np.linalg.norm(perp) > RH.LEVER["far_m"]
    t_sim = (sim[s, k + 1] - sim[s, k]) / np.linalg.norm(sim[s, k + 1] - sim[s, k])
    assert np.linalg.norm(pos[-1] - (sim[s, k] + np.linalg.norm(along) * np.sign(np.dot(along, t)) * t_sim + perp)) < 0.02


def test_attach_and_neighbour_weights():
    a = RH.attach_weight(np.array([0.05, 0.0, -0.025, -0.05, -0.2]))
    assert np.allclose(a, [0, 0, 0.5, 1, 1])
    nb, w = RH.neighbour(np.array([0, 0, 3, 4]), np.array([0.2, 0.9, 1.0, 0.7]), 5)
    assert nb.tolist() == [0, 1, 4, 4] and np.allclose(w, [0, 0.4, 0.5, 0])
