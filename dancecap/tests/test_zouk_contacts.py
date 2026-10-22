"""Unit tests for dancecap.zouk_contacts' pure parts (capsule geometry, wrist limits, holds, Viterbi / hysteresis, occlusion, window splits;
no footage, no GPU, no body model).

    .venv-nopo4d\\Scripts\\python.exe -m pytest dancecap/tests/test_zouk_contacts.py -q -p no:cacheprovider      (from the workspace root)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dancecap import zouk_contacts as zc  # noqa: E402


def _brute_seg_seg(p1, q1, p2, q2, n=401):
    s = np.linspace(0, 1, n)
    a = p1 + s[:, None] * (q1 - p1)
    b = p2 + s[:, None] * (q2 - p2)
    return np.linalg.norm(a[:, None] - b[None], axis=-1).min()


def test_seg_seg_matches_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(60):
        p1, q1, p2, q2 = rng.normal(size=(4, 3))
        d, s, t = zc.seg_seg(p1, q1, p2, q2)
        assert d == pytest.approx(_brute_seg_seg(p1, q1, p2, q2), abs=3e-3)
        assert 0 <= s <= 1 and 0 <= t <= 1
    # degenerate: a point (sphere) against a segment, parallel segments
    d, _, _ = zc.seg_seg(np.zeros(3), np.zeros(3), np.array([1.0, -1, 0]), np.array([1.0, 1, 0]))
    assert d == pytest.approx(1.0, abs=1e-6)
    d, _, _ = zc.seg_seg(np.zeros(3), np.array([1.0, 0, 0]), np.array([0.0, 0.5, 0]), np.array([1.0, 0.5, 0]))
    assert d == pytest.approx(0.5, abs=1e-6)


def test_seg_seg_torch_same_and_differentiable():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(1)
    P = rng.normal(size=(8, 4, 3))
    dn, _, _ = zc.seg_seg(P[:, 0], P[:, 1], P[:, 2], P[:, 3])
    Pt = torch.tensor(P, requires_grad=True)
    dt, _, _ = zc.seg_seg(Pt[:, 0], Pt[:, 1], Pt[:, 2], Pt[:, 3], torch)
    assert np.allclose(dt.detach().numpy(), dn, atol=1e-6)
    dt.sum().backward()
    assert torch.isfinite(Pt.grad).all()


def test_clearance_sign_is_overlap_depth():
    caps = {"r": np.array([0.03, 0.02])}
    A = np.array([[[0, 0, 0], [0, 0.04, 0]]], float)          # (1 frame, 2 capsules, 3)
    B = np.array([[[0.1, 0, 0], [0.1, 0.04, 0]]], float)
    c = zc.clearance(caps, A, B, caps, A, B, np.array([0]), np.array([1]))
    # axes 0.04 apart, radii 0.03 + 0.02 -> overlap 1 cm
    assert c[0, 0, 0] == pytest.approx(-0.01, abs=1e-9)


def test_capsule_points_kinds():
    J = np.zeros((1, 55, 3))
    J[0, 3] = [1, 0, 0]
    J[0, 6] = [1, 1, 0]
    R = np.tile(np.eye(3), (1, 55, 1, 1))
    R[0, 6] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]                # 90 deg about z
    tips = np.zeros((1, 10, 3))
    tips[0, 7] = [5, 5, 5]
    caps = {"A": [("j", 3), ("lerp", 3, 6, 0.25)], "B": [("tip", 7), ("loc", 6, (0.1, 0.0, 0.0))]}
    A, B = zc.capsule_points(caps, J, R, tips)
    assert np.allclose(A[0, 0], [1, 0, 0]) and np.allclose(B[0, 0], [5, 5, 5])
    assert np.allclose(A[0, 1], [1, 0.25, 0])
    assert np.allclose(B[0, 1], [1, 1.1, 0])                     # the local x of joint 6 is world y


def _wrist_pose(side, R):
    from scipy.spatial.transform import Rotation as Ro
    P = np.zeros((1, 55, 3))
    P[0, zc.WRIST[side]] = Ro.from_matrix(R).as_rotvec()
    return P


@pytest.mark.parametrize("side", ["left", "right"])
def test_wrist_split_twist_then_bend(side):
    from scipy.spatial.transform import Rotation as Ro
    a, n, t = zc._rest_axes(side)
    flex_axis, dev_axis = np.cross(a, n), np.cross(a, t)
    R = Ro.from_rotvec(np.radians(70) * a).as_matrix() @ Ro.from_rotvec(np.radians(50) * flex_axis).as_matrix()
    w = zc.wrist_angles(_wrist_pose(side, R), side)
    assert w["flexion"][0] == pytest.approx(50, abs=0.1) and w["deviation"][0] == pytest.approx(0, abs=0.1)
    assert w["twist"][0] == pytest.approx(70, abs=0.1) and w["bend"][0] == pytest.approx(50, abs=0.1)
    R = Ro.from_rotvec(np.radians(-40) * a).as_matrix() @ Ro.from_rotvec(np.radians(-30) * dev_axis).as_matrix()
    w = zc.wrist_angles(_wrist_pose(side, R), side)
    assert w["deviation"][0] == pytest.approx(-30, abs=0.1) and w["twist"][0] == pytest.approx(-40, abs=0.1)
    # past 90 deg the swing vector stays continuous: a 110-deg flexion is flexion, not 170 deg of deviation
    R = Ro.from_rotvec(np.radians(110) * flex_axis).as_matrix()
    w = zc.wrist_angles(_wrist_pose(side, R), side)
    assert w["bend"][0] == pytest.approx(110, abs=0.1) and w["flexion"][0] == pytest.approx(110, abs=0.1)
    assert abs(w["deviation"][0]) < 0.1
    # a combined swing: its parts are bend x direction
    R = Ro.from_rotvec(np.radians(60) * (flex_axis + dev_axis) / np.sqrt(2)).as_matrix()
    w = zc.wrist_angles(_wrist_pose(side, R), side)
    assert w["bend"][0] == pytest.approx(60, abs=0.1)
    assert w["flexion"][0] == pytest.approx(60 / np.sqrt(2), abs=0.2) and w["deviation"][0] == pytest.approx(60 / np.sqrt(2), abs=0.2)


def test_limit_excess():
    ang = {"flexion": np.array([0.0, 100.0, -90.0, 0.0, 0.0]), "deviation": np.array([0.0, 0.0, 0.0, 35.0, -50.0]),
           "twist": np.array([0.0, 0.0, 0.0, 0.0, 120.0]), "bend": np.array([0.0, 100.0, 90.0, 35.0, 50.0])}
    ex = zc.limit_excess(ang)
    assert all(ex[k][0] == 0 for k in ex)
    assert ex["flexion"][1] == pytest.approx(20) and ex["bend"][1] == pytest.approx(15)
    assert ex["extension"][2] == pytest.approx(20) and ex["bend"][2] == pytest.approx(5)
    assert ex["radial"][3] == pytest.approx(10)
    assert ex["ulnar"][4] == pytest.approx(10) and ex["twist"][4] == pytest.approx(30)


def _toy_arm_caps(fold_deg):
    """hand capsules (palm + one finger) and the forearm along -x from the wrist at the origin, the hand bent by fold_deg toward -y"""
    th = np.radians(fold_deg)
    d = np.array([np.cos(th), -np.sin(th), 0.0])                    # hand direction (0 deg: straight on, +x)
    names = ["palm_middle_R", "middle1_R", "forearm_prox_R", "forearm_dist_R"]
    parts = ["hand_R", "hand_R", "forearm_R", "forearm_R"]
    A = np.array([[0.015 * d, 0.09 * d, [-0.25, 0, 0], [-0.125, 0, 0]]])
    B = np.array([[0.09 * d, 0.17 * d, [-0.125, 0, 0], [-0.005, 0, 0]]])
    return {"names": names, "parts": parts, "r": np.array([0.016, 0.009, 0.045, 0.031])}, A, B


def test_self_forearm_clearance_folding():
    c_straight = zc.self_forearm_clearance(*_toy_arm_caps(0), "right")[0]
    c_bent = zc.self_forearm_clearance(*_toy_arm_caps(90), "right")[0]
    c_folded = zc.self_forearm_clearance(*_toy_arm_caps(170), "right")[0]
    assert c_straight > 0.05 and c_bent > 0 and c_folded < -0.02


def test_viterbi_switching_cost_is_hysteresis():
    sc = np.zeros((30, 2))
    sc[:, 0] = 1.0
    sc[10:13, 1] = 2.0                 # a 3-frame blip of state 1 (+1 per frame)
    tr = np.array([[0, 5.0], [5.0, 0]])
    assert (zc.viterbi(sc, tr) == 0).all()
    sc[10:25, 1] = 2.0                 # a long run: worth switching
    p = zc.viterbi(sc, tr)
    assert (p[11:24] == 1).all() and p[0] == 0 and p[-1] == 0


def test_runs_and_min_runs():
    m = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1], bool)
    assert zc._runs(m, 0) == [(1, 2), (5, 5), (7, 9)]
    assert zc._runs(m, 1) == [(1, 2), (5, 9)]
    lab = np.array([0] * 10 + [1] * 2 + [0] * 10 + [2] * 3 + [3] * 12)
    out = zc._min_runs(lab, 5, keep=(2,))
    assert (out[10:12] == 0).all()              # the short run of 1 joins its neighbours
    assert (out[22:25] == 2).all()              # a kept label stays


def test_site_hold_mapping():
    assert zc.site_hold("back_L1_hi", "face") == "upper_back"
    assert zc.site_hold("side_L", "face") == "upper_back"
    assert zc.site_hold("back_C_lo", "face") == "wrap_back"
    assert zc.site_hold("back_R2_mid", "face") == "wrap_back"
    assert zc.site_hold("back_L1_hi", "away") == "behind_back"
    assert zc.site_hold("shoulder_front_L", "away") == "shoulder_top"
    assert zc.site_hold("chest_L", "face") == "chest"
    assert zc.site_hold("forearm_L", "face") == "arm"
    assert zc.site_hold("hand_R", "side") == "hand_hold"
    # every candidate site has a hold in every facing class, and every hold region names existing candidate sites
    for s in zc.CAND_SITES:
        for fc in ("face", "away", "side"):
            assert zc.site_hold(s, fc) in zc.HOLDS
    for h, sl in zc.TORSO_HOLD_SITES.items():
        assert h in zc.HOLDS and all(s in zc.CAND_SITES for s in sl)


def test_ray_occluded():
    C = np.zeros((1, 3))
    X = np.array([[[0, 0, 2.0], [0.5, 0, 2.0]]])                  # two points 2 m in front of the camera
    A = np.array([[[-0.1, 0, 1.0]]])
    B = np.array([[[0.1, 0, 1.0]]])                               # a capsule across the first ray at 1 m
    occ = zc.ray_occluded(C, X, A, B, np.array([0.05]), 0.03)
    assert occ[0, 0] and not occ[0, 1]
    # a point ON the occluder's surface is not hidden by it (the ray stops margin short of the point)
    X2 = np.array([[[0, 0, 0.95]]])
    assert not zc.ray_occluded(C, X2, A, B, np.array([0.05]), 0.03)[0, 0]


def test_torso_frame_and_in_frame():
    J = np.zeros((1, 55, 3))
    J[0, zc.SHOULDER["left"]] = [0.2, 1.4, 0]
    J[0, zc.SHOULDER["right"]] = [-0.2, 1.4, 0]
    J[0, zc.SPINE3] = [0, 1.3, 0]
    F = zc.torso_frame(J)[0]
    assert np.allclose(F[:, 0], [1, 0, 0]) and np.allclose(F[:, 2], [0, 0, 1])
    p = zc.in_frame(np.array([[0.1, 1.3, -0.12]]), J)[0]
    assert np.allclose(p, [0.1, 0.0, -0.12])                       # her left, at the spine-3 height, behind her (on her back)


def test_split_runs_keeps_chunks_together():
    wins = [{"run": [0, 100], "core": [0, 50]}, {"run": [0, 100], "core": [40, 100]}, {"run": [200, 260], "core": [200, 260]},
            {"run": [300, 900], "core": [300, 600]}, {"run": [300, 900], "core": [590, 900]}]
    parts = zc.split_runs(wins, 2)
    assert sorted(i for p in parts for i in p) == [0, 1, 2, 3, 4]
    for p in parts:
        if 0 in p or 1 in p:
            assert 0 in p and 1 in p and p.index(0) < p.index(1)
        if 3 in p or 4 in p:
            assert 3 in p and 4 in p and p.index(3) < p.index(4)


def test_hold_library_consistent():
    assert set(zc.HOLD_NAMES) >= {"upper_back", "wrap_back", "behind_back", "shoulder_top", "chest", "hand_hold", "arm", "free"}
    for fc, pri in zc.HOLD_PRIOR.items():
        assert set(pri) <= set(zc.HOLDS) and 0.9 < sum(pri.values()) < 1.1
    # the zouk rule the owner gave: face to face, his right palm on her upper LEFT back is the most likely hold
    assert max(zc.HOLD_PRIOR["face"], key=zc.HOLD_PRIOR["face"].get) == "upper_back"
    assert set(zc.UPPER_LEFT_BACK) <= set(zc.TORSO_HOLD_SITES["upper_back"])


def test_slerp_rotvec_ends_and_middle():
    from scipy.spatial.transform import Rotation as Ro
    a = np.array([[0.0, 0, 0], [0.3, 0.1, -0.2]])
    b = np.array([[0.0, 0, np.pi / 2], [0.3, 0.1, -0.2]])
    assert np.allclose(zc._slerp_rotvec(a, b, np.array([0.0, 0.5])), a, atol=1e-9)
    assert np.allclose(zc._slerp_rotvec(a, b, np.array([1.0, 1.0])), b, atol=1e-9)
    mid = zc._slerp_rotvec(a[:1], b[:1], np.array([0.5]))
    assert np.allclose(mid, [[0, 0, np.pi / 4]], atol=1e-9)


# ---------------------------------------------------------------------------------------------- the post-hoc gate's terms (2026-10-22 review)

def test_torch_twist_and_elbow_terms_match_the_numpy_angles():
    """wrist_terms_t's twist and elbow_terms_t (the re-solve's loss terms) equal wrist_angles / elbow_angles (the gate's measures)"""
    import torch
    from scipy.spatial.transform import Rotation as Ro
    rng = np.random.default_rng(0)
    P = np.zeros((300, 55, 3))
    P[:, zc.WRIST["right"]] = rng.normal(size=(300, 3)) * 0.9
    P[:, zc.ELBOW["right"]] = rng.normal(size=(300, 3)) * 0.8
    wa, ea = zc.wrist_angles(P, "right"), zc.elbow_angles(P, "right")
    b, f, d, tw = zc.wrist_terms_t(torch, torch.tensor(Ro.from_rotvec(P[:, zc.WRIST["right"]]).as_matrix()), "right", with_twist=True)
    ok = wa["bend"] < 170
    assert np.abs(np.degrees(tw.numpy())[ok] - wa["twist"][ok]).max() < 1e-6
    assert np.abs(np.degrees(b.numpy()) - wa["bend"]).max() < 1e-6
    ef, eo = zc.elbow_terms_t(torch, torch.tensor(Ro.from_rotvec(P[:, zc.ELBOW["right"]]).as_matrix()), "right")
    assert np.abs(np.degrees(ef.numpy()) - ea["flexion"]).max() < 1e-6
    assert np.abs(np.degrees(eo.numpy()) - ea["off_hinge"]).max() < 1e-6


def test_local_rotation_speed_and_gate_constants():
    """_local_rot_speed: degrees per frame of a joint's local rotation; the gate's thresholds are the review's numbers"""
    from scipy.spatial.transform import Rotation as Ro
    P = np.zeros((4, 55, 3))
    for k in range(4):
        P[k, zc.WRIST["right"]] = Ro.from_euler("z", 5.0 * k, degrees=True).as_rotvec()
    sp = zc._local_rot_speed(P, zc.WRIST["right"])
    assert np.allclose(sp, 5.0, atol=1e-6)
    g = zc.GATE
    assert g["twist_deg"] == 90.0 and g["hyper_deg"] == -5.0 and g["add_mps"] == 0.55 and g["coco_worse_px"] == 3.0
