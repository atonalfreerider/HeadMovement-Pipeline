"""sides.geom (DESIGN 8): the GLI against a numerical double integral, its symmetries, through vs around a bone end, the signed closest-point
separation (F2), margins on hand-built poses, swing-twist round trips, a 360-degree chain wrap, heading unwrapping at 1000 deg/s; numpy
and torch give the same numbers.

    .venv-nopo4d\\Scripts\\python.exe -m pytest dancecap/tests/test_sides_geom.py -q -p no:cacheprovider     (from the workspace root)
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from dancecap.sides import geom

torch = pytest.importorskip("torch")


def gli_numeric(p1, p2, p3, p4, n=400):
    """Gauss's double integral (1/4pi) int int (r1 - r2).(dr1 x dr2) / |r1 - r2|^3, Gauss-Legendre"""
    x, w = np.polynomial.legendre.leggauss(n)
    s, ws = 0.5 * (x + 1), 0.5 * w
    u, v = p2 - p1, p4 - p3
    r1 = p1[None] + s[:, None] * u[None]
    r2 = p3[None] + s[:, None] * v[None]
    d = r1[:, None] - r2[None]
    num = (d * np.cross(u, v)[None, None]).sum(-1)
    den = np.linalg.norm(d, axis=-1) ** 3
    return float((ws[:, None] * ws[None] * num / den).sum() / (4 * math.pi))


@pytest.mark.parametrize("h", [0.5, 0.1, -0.2])
def test_gli_matches_double_integral(h):
    p1, p2 = np.array([-0.3, 0.0, 0.0]), np.array([0.25, 0.05, 0.0])
    p3, p4 = np.array([0.02, -0.2, h]), np.array([-0.05, 0.3, h + 0.1])
    g = float(geom.gli_seg(p1, p2, p3, p4))
    assert abs(g - gli_numeric(p1, p2, p3, p4)) < 1e-6


def test_gli_parallel_zero_and_symmetries():
    p1, p2 = np.array([0, 0, 0.0]), np.array([1, 0, 0.0])
    q1, q2 = np.array([0, 0.3, 0.1]), np.array([1, 0.3, 0.1])
    assert abs(float(geom.gli_seg(p1, p2, q1, q2))) < 1e-9
    a1, a2 = np.array([-0.3, 0, 0.0]), np.array([0.3, 0.02, 0.0])
    b1, b2 = np.array([0.0, -0.3, 0.05]), np.array([0.01, 0.3, 0.06])
    g = float(geom.gli_seg(a1, a2, b1, b2))
    assert abs(g) > 0.05
    assert abs(float(geom.gli_seg(b1, b2, a1, a2)) - g) < 1e-12          # the linking integral is symmetric in the pair
    assert abs(float(geom.gli_seg(a2, a1, b1, b2)) + g) < 1e-12          # reversing one segment flips the sign
    # chain GLI = the sum over its segment pairs; numpy = torch
    A = np.stack([a1, 0.5 * (a1 + a2), a2])[None]
    B = np.stack([b1, 0.5 * (b1 + b2), b2])[None]
    G, M = geom.gli_chain(A, B, pairs=True)
    assert abs(float(G[0]) - g) < 1e-9 and M.shape == (1, 2, 2)
    Gt = geom.gli_chain(torch.tensor(A), torch.tensor(B), xp=torch)
    assert abs(float(Gt[0]) - float(G[0])) < 1e-9


def _sweep(path):
    """segment B (vertical, 40 cm) moved along `path` (n,3) of its midpoint, against A (horizontal, x in -0.15..0.15)"""
    A0, A1 = np.array([-0.15, 0, 0.0]), np.array([0.15, 0, 0.0])
    off = np.array([0, 0, 0.2])
    B0 = path - off
    B1 = path + off
    G = geom.gli_seg(A0[None], A1[None], B0, B1)
    return G, (A0, A1, B0, B1)


def test_through_jumps_around_does_not():
    n = 40
    # through: B crosses A's interior (y from +5 cm to -5 cm at x = 0)
    thr = np.stack([np.zeros(n), np.linspace(0.05, -0.05, n), np.zeros(n)], 1)
    G, _ = _sweep(thr)
    ev = geom.passthrough_events(G)
    assert len(ev) == 1 and abs(G[ev[0]] - G[ev[0] - 1]) > 0.5
    # around the bone end: B passes beyond A's end (x = 0.25), from y +5 cm to -5 cm, G changes sign through ~0
    aro = np.stack([np.full(n, 0.25), np.linspace(0.05, -0.05, n), np.zeros(n)], 1)
    G2, _ = _sweep(aro)
    assert np.sign(G2[0]) != np.sign(G2[-1])
    assert len(geom.passthrough_events(G2)) == 0


def test_signed_sep_flips_at_a_through_crossing_and_is_inactive_at_bone_ends():
    n = 40
    thr = np.stack([np.zeros(n), np.linspace(0.05, -0.05, n), np.zeros(n)], 1)
    G, (A0, A1, B0, B1) = _sweep(thr)
    ds, s, t, dist, inter = geom.signed_sep(A0[None], A1[None], B0, B1)
    k = geom.passthrough_events(G)[0]
    assert np.sign(ds[k - 1]) != np.sign(ds[k])                  # the F2 property: d_s flips exactly where G jumps
    assert inter[k - 1] and inter[k]
    assert (np.sign(ds[inter]) == np.sign(G[inter])).all()       # interior closest points: sign(d_s) = sign(G)
    # around the end: the closest point on A sits at its end -> inactive
    aro = np.stack([np.full(n, 0.25), np.linspace(0.05, -0.05, n), np.zeros(n)], 1)
    _, (A0, A1, B0, B1) = _sweep(aro)
    _, s2, _, _, inter2 = geom.signed_sep(A0[None], A1[None], B0, B1)
    assert not inter2.any() and np.allclose(s2, 1.0)
    # torch = numpy
    dst, *_ = geom.signed_sep(torch.tensor(A0[None]), torch.tensor(A1[None]), torch.tensor(B0), torch.tensor(B1), xp=torch)
    assert np.allclose(dst.numpy(), ds, atol=1e-12)


def _dancer(cx: float, facing: float, arm_up: float = 0.0) -> dict:
    """a hand-built point set: pelvis at (cx, 1.0, 0), facing +z (facing = +1) or -z (-1)"""
    f = facing
    o = np.array([cx, 0.0, 0.0])
    left = np.array([f, 0, 0.0])            # +z facing: the body's left is +x
    P = {"hipL": o + [0, 1.0, 0] + 0.1 * left, "hipR": o + [0, 1.0, 0] - 0.1 * left,
         "shL": o + [0, 1.5, 0] + 0.2 * left, "shR": o + [0, 1.5, 0] - 0.2 * left,
         "knL": o + [0, 0.55, 0.02 * f] + 0.1 * left, "knR": o + [0, 0.55, 0.02 * f] - 0.1 * left,
         "anL": o + [0, 0.1, 0] + 0.1 * left, "anR": o + [0, 0.1, 0] - 0.1 * left, "head": o + [0, 1.7, 0]}
    P["elL"], P["elR"] = P["shL"] + [0, -0.28 + arm_up, 0] + 0.02 * left, P["shR"] + [0, -0.28 + arm_up, 0] - 0.02 * left
    P["wrL"], P["wrR"] = P["elL"] + [0, -0.25, 0.05 * f], P["elR"] + [0, -0.25, 0.05 * f]
    P["palmL"], P["palmR"] = P["wrL"] + [0, -0.08, 0], P["wrR"] + [0, -0.08, 0]
    return {k: np.asarray(v, float)[None] for k, v in P.items()}


def test_margins_on_hand_built_poses():
    # face to face, 40 cm apart along z: him at z = -0.2 facing +z, her at z = +0.2 facing -z
    L = _dancer(0.0, 1.0)
    F = _dancer(0.0, -1.0)
    for k in L:
        L[k] = L[k] + [0, 0, -0.2]
        F[k] = F[k] + [0, 0, 0.2]
    m = geom.margins(L, F)
    assert m["R1_lead"]["m"][0] > 0.9 and m["R1_follow"]["m"][0] > 0.9 and m["R2"]["m"][0] > 0.9
    # his right palm on her back (behind her: z > her centre) -> R4 front/back < 0; at her side -> flank > 0
    L2 = dict(L)
    L2["palmR"] = np.array([[0.0, 1.25, 0.32]])
    m2 = geom.margins(L2, F)
    assert m2["R4_fb"]["m"][0] < -0.05
    L2["palmR"] = np.array([[0.2, 1.25, 0.2]])
    assert geom.margins(L2, F)["R4_fl"]["m"][0] > 0.0
    # his right elbow above her left elbow (both near): R6 > 0, gated
    L3 = dict(L)
    F3 = dict(F)
    F3["elL"] = np.array([[-0.2, 1.2, 0.15]])
    L3["elR"] = np.array([[-0.2, 1.28, 0.10]])
    m3 = geom.margins(L3, F3)
    assert m3["R6"]["gate"][0] and m3["R6"]["m"][0] > 0.07
    # his right knee between hers (R11 > 0)
    L4 = dict(L)
    L4["knR"] = np.array([[0.0, 0.55, 0.15]])
    assert geom.margins(L4, F)["R11_lead"]["m"][0] > 0.05
    # numpy = torch on the whole set
    Lt = {k: torch.tensor(v) for k, v in L3.items()}
    Ft = {k: torch.tensor(v) for k, v in F3.items()}
    mt = geom.margins(Lt, Ft, xp=torch)
    for k in m3:
        assert np.allclose(np.asarray(m3[k]["m"], float), mt[k]["m"].numpy(), atol=1e-9), k


def test_swivel_and_numpy_torch():
    S = np.array([[0.0, 1.5, 0.0]])
    W = np.array([[0.5, 1.5, 0.0]])
    E_down = np.array([[0.25, 1.35, 0.0]])
    E_up = np.array([[0.25, 1.65, 0.0]])
    phi_d, w_d = geom.swivel(S, E_down, W)
    phi_u, _ = geom.swivel(S, E_up, W)
    assert abs(phi_d[0]) < 1e-6 and abs(abs(phi_u[0]) - 180) < 1e-6 and w_d[0] > 0.5
    pt, _ = geom.swivel(torch.tensor(S), torch.tensor(E_down), torch.tensor(W), xp=torch)
    assert abs(float(pt[0]) - phi_d[0]) < 1e-9


def test_swing_twist_round_trip():
    rng = np.random.default_rng(0)
    axis = np.array([0.0, -1.0, 0.0])
    for tw in (-170.0, -40.0, 0.0, 25.0, 179.0):
        sw_axis = np.cross(axis, rng.normal(size=3))
        sw_axis /= np.linalg.norm(sw_axis)
        Rs = geom.rotvec_to_mat(sw_axis * np.radians(35.0))
        Rt = geom.rotvec_to_mat(axis * np.radians(tw))
        R = Rs @ Rt
        got, Rs2 = geom.swing_twist(R, axis)
        assert abs(geom.wrap_deg(got - tw)) < 1e-6
        assert np.allclose(Rs2, Rs, atol=1e-9)
        gt, _ = geom.swing_twist(torch.tensor(R), axis, xp=torch)
        assert abs(float(gt) - float(got)) < 1e-6
    # quaternions of 180-degree rotations keep their relative signs
    R180 = geom.rotvec_to_mat(np.array([1.0, 1.0, 0]) / math.sqrt(2) * math.pi)
    q = geom.mat_to_quat(R180)
    assert np.allclose(np.abs(q[1:3]), [math.sqrt(0.5)] * 2, atol=1e-9) and q[1] * q[2] > 0


def test_chain_wrap_found_and_unwound_with_joints_within_1mm():
    """a spine wound 360 deg about its own (vertical) bone axis: R16 finds the wrap; taking 360 deg out of the twists moves no joint"""
    up = np.array([0.0, 1.0, 0.0])
    tw = np.array([175.0, 170.0, 15.0])                 # sums to 360 = wound one full turn
    Rl = np.stack([geom.rotvec_to_mat(up * np.radians(a)) for a in tw])[None]
    ct = geom.chain_twist(Rl, np.stack([up] * 3))
    assert abs(abs(float(ct["wrap"][0])) - 360.0) < 1e-6 and abs(float(ct["wrapped"][0])) < 1e-6

    def fk(R):
        J = [np.zeros(3)]
        G = np.eye(3)
        bones = [np.array([0.0, 0.1, 0.0]), np.array([0.0, 0.12, 0.0]), np.array([0.0, 0.15, 0.0])]   # along the twist axis
        for i in range(3):
            G = G @ R[i]
            J.append(J[-1] + G @ bones[i])
        return np.array(J), G
    J0, G0 = fk(Rl[0])
    un = tw - 360.0 * np.array([0.3, 0.35, 0.35])       # the fixed shares of DESIGN 2.4 / pipeline-native 2.5
    Ru = np.stack([geom.rotvec_to_mat(up * np.radians(a)) for a in un])
    J1, G1 = fk(Ru)
    assert np.allclose(G0, G1, atol=1e-9)               # the chain's end orientation is the same
    assert np.abs(J1 - J0).max() < 1e-3                # joints within 1 mm (bones along the twist axis: a pure wrap)
    ct2 = geom.chain_twist(Ru[None], np.stack([up] * 3))
    assert abs(float(ct2["wrap"][0])) < 1e-6


def test_heading_unwrap_at_1000_deg_per_s():
    t = np.arange(0, 3.0, 1 / 30)
    psi = 1000.0 * t
    wr = geom.wrap_deg(psi)
    un = geom.unwrap_deg(wr)
    assert np.allclose(un - un[0], psi - psi[0], atol=1e-6)
    ut = geom.unwrap_deg(torch.tensor(wr), xp=torch)
    assert np.allclose(ut.numpy() - float(ut[0]), psi - psi[0], atol=1e-6)
    # NaN gaps are bridged
    wr2 = wr.copy()
    wr2[10:12] = np.nan
    un2 = geom.unwrap_deg(wr2)
    assert np.isnan(un2[10]) and abs(un2[20] - un2[0] - (psi[20] - psi[0])) < 1e-6
    # heading of a vector: +z = 0, +x = 90 deg
    assert abs(float(geom.heading(np.array([0, 0, 1.0])))) < 1e-9 and abs(float(geom.heading(np.array([1.0, 0, 0]))) - 90) < 1e-9
