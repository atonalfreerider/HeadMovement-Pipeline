"""mesh_contact: exact distances, winding-number inside test, finger self-intersections, capsule spacing. Tiny
synthetic meshes only (no take data, CPU).

    .venv-nopo4d\\Scripts\\python.exe -m pytest dancecap/tests/test_mesh_contact.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dancecap import mesh_contact as mc  # noqa: E402


def box(lo=(0, 0, 0), hi=(1, 1, 1)):
    """Axis-aligned box: 8 vertices, 12 outward (CCW) triangles."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    V = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    # vertex index = 4*ix + 2*iy + iz
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    F = []
    for a, b, c, d in quads:
        F += [(a, b, c), (a, c, d)]
    F = np.array(F)
    # orient outward
    ctr = V.mean(0)
    for i, (a, b, c) in enumerate(F):
        n = np.cross(V[b] - V[a], V[c] - V[a])
        if n @ (V[[a, b, c]].mean(0) - ctr) < 0:
            F[i] = (a, c, b)
    return V, F


def test_point_triangle_regions():
    A, B, C = np.array([0.0, 0, 0]), np.array([1.0, 0, 0]), np.array([0.0, 1, 0])
    P = np.array([[0.2, 0.2, 0.5], [-1, -1, 0.0], [2, -0.5, 0.0], [0.5, -1, 0.3], [1, 1, 0.0]])
    q, bary = mc.point_triangle(P, *(np.broadcast_to(x, P.shape) for x in (A, B, C)))
    assert np.allclose(q[0], [0.2, 0.2, 0]) and np.allclose(q[1], A) and np.allclose(q[2], B)
    assert np.allclose(q[3], [0.5, 0, 0]) and np.allclose(q[4], [0.5, 0.5, 0])
    assert np.allclose(bary.sum(1), 1) and np.allclose(bary[:, :, None].__mul__(np.stack([A, B, C])).sum(1), q)
    # brute force: no point of a dense sample of the triangle is closer
    rng = np.random.default_rng(0)
    u, v = rng.random((2, 20000))
    m = u + v <= 1
    S = A + u[m, None] * (B - A) + v[m, None] * (C - A)
    for p, qq in zip(P, q):
        assert np.linalg.norm(p - qq) <= np.linalg.norm(S - p, axis=1).min() + 1e-9


def test_winding_numbers_inside_outside_and_with_a_hole():
    V, F = box()
    P = np.array([[0.5, 0.5, 0.5], [0.9, 0.1, 0.2], [1.5, 0.5, 0.5], [0.5, 0.5, -0.01]])
    w = mc.winding_numbers(P, V, F, "cpu")
    assert np.allclose(w, [1, 1, 0, 0], atol=1e-6)
    # a missing face (like the SMPL-X mouth hole): still inside by w > 0.5, where ray parity can flip
    w2 = mc.winding_numbers(P[:1], V, F[2:], "cpu")
    assert 0.5 < w2[0] < 1.0


def test_signed_distance_fixes_the_nearest_vertex_normal_sign():
    """The old gap: nearest VERTEX, signed by that vertex's normal. Just outside the middle of a big face the nearest
    vertex is a corner whose (averaged) normal points diagonally: the old sign says 'inside' and the distance is the
    corner's (0.7 m); the exact gap is +5 cm."""
    V, F = box()
    p = np.array([[0.5, 0.5, 1.05]])
    from dancecap.avatar_uv import vertex_normals
    N = vertex_normals(V, F)
    j = int(np.argmin(np.linalg.norm(V - p, axis=1)))
    old = float((p[0] - V[j]) @ N[j])
    assert old < 0                                           # the old method: sunk
    r = mc.signed_distance(p, V, mc.Topology(F, len(V)), device="cpu")
    assert abs(r["sd"][0] - 0.05) < 1e-9 and not r["inside"][0]
    r = mc.signed_distance(np.array([[0.5, 0.5, 0.97]]), V, mc.Topology(F, len(V)), device="cpu")
    assert abs(r["sd"][0] + 0.03) < 1e-9 and r["inside"][0]
    # far points skip the winding number (outside)
    r = mc.signed_distance(np.array([[5.0, 5, 5]]), V, mc.Topology(F, len(V)), device="cpu", max_depth=0.16)
    assert r["sd"][0] > 0 and np.isnan(r["w"][0])


def test_closest_points_on_a_face_subset():
    V, F = box()
    topo = mc.Topology(F[:2], len(V))                      # only the x = 0 face
    r = mc.closest_points(np.array([[0.5, 0.5, 0.5]]), V, topo)
    assert abs(r["dist"][0] - 0.5) < 1e-9 and np.allclose(r["q"][0], [0, 0.5, 0.5])


def _two_finger_hand(offset_x: float, joints=(25, 28)):
    """Two unit-ish boxes as two finger segments of a left hand (SMPL-X joints 25 index1, 28 middle1)."""
    Va, Fa = box((0, 0, 0), (0.02, 0.01, 0.01))
    Vb, Fb = box((offset_x, 0.002, 0.002), (offset_x + 0.02, 0.008, 0.008))
    V = np.concatenate([Va, Vb])
    F = np.concatenate([Fa, Fb + len(Va)])
    W = np.zeros((len(V), 55))
    W[:8, joints[0]] = 1
    W[8:, joints[1]] = 1
    return V, F, W


def test_self_intersections_between_fingers_only():
    V, F, W = _two_finger_hand(0.01)                       # middle1 box pokes through index1 box
    hf = mc.HandFaces(W, F, "left")
    r = mc.self_intersections(V, hf, return_faces=True)
    assert r["n"] > 0 and list(r["pairs"]) == ["index1-middle1"] and len(r["faces"]) > 0
    V2, F2, W2 = _two_finger_hand(0.05)                    # apart
    assert mc.self_intersections(V2, mc.HandFaces(W2, F2, "left"))["n"] == 0
    V3, F3, W3 = _two_finger_hand(0.01, joints=(25, 26))   # same finger (index1 / index2): a knuckle crease
    assert mc.self_intersections(V3, mc.HandFaces(W3, F3, "left"))["n"] == 0
    V4, F4, W4 = _two_finger_hand(0.01, joints=(20, 25))   # palm vs proximal phalanx: skin folding, ignored
    assert mc.self_intersections(V4, mc.HandFaces(W4, F4, "left"))["n"] == 0
    V5, F5, W5 = _two_finger_hand(0.01, joints=(20, 27))   # palm vs distal phalanx: counted
    assert mc.self_intersections(V5, mc.HandFaces(W5, F5, "left"))["n"] > 0
    n, tot = mc.crossing_series(np.stack([V, V2]), hf)
    assert n[0] > 0 and n[1] == 0


def test_capsule_spacing_ratio():
    radii = {f: np.full(3 if f != "thumb" else 2, 0.008) for f in mc.CHAINS21}
    lay = mc.capsule_layout(radii, samples=3)
    J = np.zeros((1, 21, 3))
    for i, (f, ch) in enumerate(mc.CHAINS21.items()):
        for n, k in enumerate(ch):
            J[0, k] = [0.1 * i, 0.0, 0.03 * n]               # fingers 10 cm apart, along z
    assert mc.spacing_ratio(J, lay)[0] > 5
    J[0, list(mc.CHAINS21["middle"])] = J[0, list(mc.CHAINS21["index"])] + [0.012, 0, 0]   # 12 mm from index
    assert abs(mc.spacing_ratio(J, lay)[0] - 0.012 / 0.016) < 1e-9
