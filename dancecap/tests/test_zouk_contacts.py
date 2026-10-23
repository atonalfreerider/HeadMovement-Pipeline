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


# ---------------------------------------------------------------------------------------------- the final pass (2026-10-22): new guards + targets

def test_whip_frames_new_whip_fast_and_calmed_flick():
    """a whip = new added speed AND faster than the input / reference there, or faster than max(p99, input, reference) + margin; a fix that
    only calms a flick of the input (slower than the input, though 'added' against it) is not a whip"""
    z = np.zeros(4)
    v_in = np.array([2.5, 0.3, 0.3, 0.3])
    v0 = v_in.copy()
    a0 = z.copy()
    v1 = np.array([1.2, 0.95, 0.3, 2.2])                # 0: calmed flick; 1: a new whip; 2: nothing; 3: faster than p99
    a1 = np.array([1.5, 0.7, 0.1, 0.2])
    w = zc.whip_frames(a0, a1, v_in, v0, v1, p99=1.8)
    assert w.tolist() == [False, True, False, True]
    # added speed the reference already had is not new
    assert not zc.whip_frames(np.array([0.7]), np.array([0.75]), np.array([0.3]), np.array([0.9]), np.array([0.95]), 2.0)[0]


def test_new_fold_flag():
    """a wrist folded past 85 deg that neither the reference nor the input folds as far (+10 deg) is new"""
    bc = np.array([90.0, 120.0, 120.0, 80.0])
    br = np.array([40.0, 115.0, 40.0, 10.0])
    bi = np.array([40.0, 40.0, 40.0, 10.0])
    assert zc.new_fold(bc, br, bi).tolist() == [True, False, True, False]


def test_window_check_vetoes_coco_elbow_and_mamma():
    """per-window 2D checks: the COCO wrist (> 3 px worse), the COCO elbow (> 4 px), MAMMA's hand points (> 6 px), each with enough data"""
    wins = [{"id": f"r4:{i}", "t_span": [i, i + 1.0]} for i in range(5)]
    chk = {"r4:0": {"n": 50, "v3_px": 10.0, "v4_px": 13.5},
           "r4:1": {"n": 50, "v3_px": 10.0, "v4_px": 10.0, "elbow_n": 40, "elbow_v3_px": 9.0, "elbow_v4_px": 13.5},
           "r4:2": {"n": 50, "v3_px": 10.0, "v4_px": 10.0, "mamma_n": 30, "mamma_v3_px": 12.0, "mamma_v4_px": 18.5},
           "r4:3": {"n": 5, "v3_px": 10.0, "v4_px": 30.0, "elbow_n": 3, "elbow_v3_px": 9.0, "elbow_v4_px": 30.0},
           "r4:4": {"n": 50, "v3_px": 10.0, "v4_px": 12.0, "elbow_n": 40, "elbow_v3_px": 9.0, "elbow_v4_px": 12.0,
                    "mamma_n": 30, "mamma_v3_px": 12.0, "mamma_v4_px": 17.0}}
    v = zc.window_check_vetoes(chk, wins)
    assert sorted(v) == ["r4:0", "r4:1", "r4:2"]
    assert "COCO wrist" in v["r4:0"]["why"] and "COCO elbow" in v["r4:1"]["why"] and "MAMMA" in v["r4:2"]["why"]
    assert all(x["gate"] for x in v.values())


def test_exact_window_checks_new_penetration_and_floating():
    """exact (her mesh): a window whose hand goes > 1 cm deeper on >= 3 frames, or floats > 2.5 cm off her back where the reference touched
    her on >= 10 frames, is vetoed; coming out of her onto her surface is not"""
    T = 100
    names = np.array(["upper_back"] * T)
    p0 = np.zeros(T)
    g0 = np.zeros(T)
    p0[10:20] = 0.08                                     # the reference 8 cm inside her ...
    g0[10:20] = -0.08
    p1, g1 = np.zeros(T), np.full(T, 0.002)             # ... the candidate on her surface: fine
    p1[40:44] = 0.03                                     # 4 frames 3 cm deeper: new penetration
    g1[60:75] = 0.04                                     # 15 frames 4 cm off her where the reference touched her: floating
    wins = [{"id": "a", "span": [5, 25], "t_span": [0, 1]}, {"id": "b", "span": [35, 50], "t_span": [1, 2]},
            {"id": "c", "span": [55, 80], "t_span": [2, 3]}]
    v, nums = zc.exact_window_checks(p0, p1, g0, g1, names, wins)
    assert sorted(v) == ["b", "c"]
    assert nums["a"]["pen_frames_gt2cm"] == [10, 0] and nums["b"]["deeper_frames"] == 4 and nums["c"]["float_frames"] == 15


def test_despike_poses_slerps_the_glitch_frames_only():
    """the right arm + finger rotations on [a, b] become the slerp between the frames just outside; other joints / frames untouched"""
    from scipy.spatial.transform import Rotation as Ro
    T = 40
    P = np.zeros((T, 55, 3))
    for k in range(T):
        P[k, zc.ELBOW["right"]] = Ro.from_euler("x", 2.0 * k, degrees=True).as_rotvec()
        P[k, zc.ELBOW["left"]] = Ro.from_euler("x", 1.0 * k, degrees=True).as_rotvec()
    P[20:23, zc.ELBOW["right"]] = Ro.from_euler("y", 80.0, degrees=True).as_rotvec()       # 3-frame glitch at 20..22 (0.667-0.733 s)
    Q, done = zc.despike_poses(P, [[20 / 30.0, 22 / 30.0]])
    assert done == [[round(20 / 30, 3), round(22 / 30, 3)]]
    ang = Ro.from_rotvec(Q[18:25, zc.ELBOW["right"]]).as_euler("xyz", degrees=True)[:, 0]
    assert np.allclose(ang, 2.0 * np.arange(18, 25), atol=1e-6)
    assert np.array_equal(Q[:, zc.ELBOW["left"]], P[:, zc.ELBOW["left"]]) and np.array_equal(Q[:20], P[:20])


def test_visible_side_targets_bring_an_inside_palm_out_toward_the_phone():
    """a box 'torso'; the phone in front of its +z face: a palm inside comes out onto the +z side near the phone's ray; a palm outside stays"""
    import torch
    from dancecap import hand_contact as hc
    import trimesh
    m = trimesh.creation.box(extents=(0.4, 0.5, 0.3))
    m = m.subdivide().subdivide().subdivide()
    V = np.asarray(m.vertices, np.float32)
    F = np.asarray(m.faces, np.int64)
    surf = hc.GpuSurface(F, len(V), "cpu")
    pc0 = np.array([[0.02, 0.05, 0.08], [0.02, 0.05, 0.30]])          # inside (7 cm under the +z face), outside (15 cm in front of it)
    C = np.array([[0.0, 0.0, 3.0], [0.0, 0.0, 3.0]])
    Vt = torch.as_tensor(np.repeat(V[None], 2, 0))
    r = zc.visible_side_targets(surf, Vt, pc0, C, np.ones(len(V), bool), lift_m=0.015, alpha=2.0)
    assert r["pushed"].tolist() == [True, False]
    assert abs(r["target"][0, 2] - (0.15 + 0.015)) < 0.01                        # on the +z face, lifted
    assert np.linalg.norm(r["target"][0, :2] - pc0[0, :2]) < 0.03                 # close to the phone's ray
    assert np.allclose(r["target"][1], pc0[1])


def test_window_check_vetoes_curation_keep_exempts_only_the_named_check():
    """a curation "keep" (decided by looking) waives only the checks it names for that window"""
    wins = [{"id": "r4:5", "t_span": [83.3, 88.3]}]
    chk = {"r4:5": {"n": 50, "v3_px": 10.0, "v4_px": 10.0, "mamma_n": 30, "mamma_v3_px": 14.7, "mamma_v4_px": 22.1,
                    "elbow_n": 40, "elbow_v3_px": 9.0, "elbow_v4_px": 9.0}}
    assert "r4:5" in zc.window_check_vetoes(chk, wins)
    assert zc.window_check_vetoes(chk, wins, exempt={"r4:5": ["mamma"]}) == {}
    chk["r4:5"]["v4_px"] = 14.0                                      # the COCO wrist 4 px worse: still vetoed
    v = zc.window_check_vetoes(chk, wins, exempt={"r4:5": ["mamma"]})
    assert "COCO wrist" in v["r4:5"]["why"] and "MAMMA" not in v["r4:5"]["why"]


# ------------------------------------------------------------------------------------------------ the closing pass

def test_despike_hermite_keeps_the_motion_s_velocity_and_only_the_named_joints():
    """a constant-speed rotation with a 3-frame glitch: the Hermite despike restores the straight motion exactly (no dead stop at the
    ends); "arm" leaves his other joints and the root alone, "body" also interpolates the root translation"""
    from scipy.spatial.transform import Rotation as Ro
    T = 40
    P = np.zeros((T, 55, 3))
    tr = np.zeros((T, 3))
    for k in range(T):
        P[k, zc.ELBOW["right"]] = Ro.from_euler("x", 3.0 * k, degrees=True).as_rotvec()
        P[k, zc.SPINE3] = Ro.from_euler("y", 1.0 * k, degrees=True).as_rotvec()
        tr[k] = [0.02 * k, 0, 0]
    clean_P, clean_tr = P.copy(), tr.copy()
    P[20:23, zc.ELBOW["right"]] = Ro.from_euler("y", 80.0, degrees=True).as_rotvec()
    P[20:23, zc.SPINE3] = Ro.from_euler("x", 30.0, degrees=True).as_rotvec()
    tr[20:23] += [0, 0.1, 0]
    Q, trq, done = zc.despike_hermite(P, tr, [[20 / 30.0, 22 / 30.0, "arm"]])
    assert done == [[round(20 / 30, 3), round(22 / 30, 3), "arm"]]
    assert np.allclose(Q[:, zc.ELBOW["right"]], clean_P[:, zc.ELBOW["right"]], atol=1e-6)
    assert np.array_equal(Q[:, zc.SPINE3], P[:, zc.SPINE3]) and np.array_equal(trq, tr)
    Q2, trq2, _ = zc.despike_hermite(P, tr, [[20 / 30.0, 22 / 30.0, "body"]])
    assert np.allclose(Q2[:, zc.SPINE3], clean_P[:, zc.SPINE3], atol=1e-6) and np.allclose(trq2, clean_tr, atol=1e-9)
    # too close to the film's ends: not applied
    assert zc.despike_hermite(P, tr, [[1 / 30.0, 2 / 30.0, "arm"]])[2] == []


def _box_torso(n_sub: int = 3):
    import trimesh
    m = trimesh.creation.box(extents=(0.4, 0.5, 0.3))
    for _ in range(n_sub):
        m = m.subdivide()
    return np.asarray(m.vertices, np.float32), np.asarray(m.faces, np.int64)


def test_ray_hit_targets_slide_along_the_phone_s_ray():
    """a box 'torso' (front face z = +0.15), the phone at z = 3 m: an inside palm slides along the phone's ray onto the face (its pixels in
    that phone kept), a palm 3 cm in front is laid onto it, one 20 cm in front is kept, the observed ray moves it on the surface"""
    V, F = _box_torso()
    C = np.array([[0.1, 0.0, 3.0]] * 4)
    pc0 = np.array([[0.02, 0.05, 0.05], [0.0, 0.0, 0.18], [0.0, 0.0, 0.35], [0.0, 0.0, 0.10]])
    thru = pc0.copy()
    thru[3] = [0.10, 0.08, 0.10]                                          # the phone shows the hand 10 cm to the side
    mode = np.array([zc.RAY_INSIDE, zc.RAY_FLOAT, zc.RAY_FLOAT, zc.RAY_OBS])
    r = zc.ray_hit_targets(np.repeat(V[None], 4, 0), F, C, thru, pc0, mode, lift_m=0.015)
    assert r["pushed"].tolist() == [True, True, False, True] and r["on_surface"].tolist() == [True, True, False, True]
    for i in (0, 1, 3):
        assert r["target"][i, 2] == pytest.approx(0.15 + 0.015, abs=2e-3)
    d = (pc0[0] - C[0]) / np.linalg.norm(pc0[0] - C[0])
    off = (r["target"][0] - C[0]) - ((r["target"][0] - C[0]) @ d) * d
    assert np.linalg.norm(off) < 0.016                                   # on the phone's ray (up to the lift along the face normal)
    assert np.allclose(r["target"][2], pc0[2])
    assert r["target"][3, 0] > 0.08
    # a slide longer than max_slide_m is not taken
    r2 = zc.ray_hit_targets(V[None], F, C[:1], pc0[:1], pc0[:1], mode[:1], lift_m=0.015, max_slide_m=0.05)
    assert not r2["pushed"][0]


def test_unfold_wrist_compresses_only_the_fold():
    from scipy.spatial.transform import Rotation as Ro
    a, n, _ = zc._rest_axes("right")
    flex_axis = np.cross(a, n)
    P = np.zeros((3, 55, 3))
    P[0, zc.WRIST["right"]] = np.radians(170) * flex_axis
    P[1, zc.WRIST["right"]] = np.radians(40) * flex_axis
    P[2, zc.WRIST["right"]] = np.radians(170) * flex_axis
    Q, b = zc.unfold_wrist(P, weight=np.array([1.0, 1.0, 0.0]))
    b1 = zc.wrist_angles(Q, "right")["bend"]
    assert b[0] == pytest.approx(170, abs=0.1)
    assert b1[0] == pytest.approx(75 + 0.1 * 95, abs=0.5)
    assert np.allclose(Q[1], P[1]) and np.allclose(Q[2], P[2])


class _FakeObs:
    """one or two phones at fixed places (pinhole, f = 1000 px, 1920 x 1080), every frame ok"""

    def __init__(self, centres, T=5):
        self.cams = [str(i) for i in range(len(centres))]
        self.z = {}
        for c, C in zip(self.cams, centres):
            C = np.asarray(C, float)
            fwd = -C / np.linalg.norm(C)
            right = np.cross(fwd, [0, 1.0, 0])
            right /= np.linalg.norm(right)
            down = np.cross(fwd, right)
            R = np.stack([right, down, fwd])
            K = np.array([[1000.0, 0, 960], [0, 1000.0, 540], [0, 0, 1]])
            self.z[c] = {"R": np.repeat(R[None], T, 0), "t": np.repeat((-R @ C)[None], T, 0), "K": np.repeat(K[None], T, 0),
                         "ok": np.ones(T, bool)}

    def project(self, cam, X, ks):
        z = self.z[cam]
        Xc = np.einsum("kij,kpj->kpi", z["R"][ks], X) + z["t"][ks][:, None]
        uv = np.einsum("kij,kpj->kpi", z["K"][ks], Xc)
        return uv[..., :2] / uv[..., 2:], Xc[..., 2]


def test_joint_visible_and_coco_errors_vis(monkeypatch):
    """a capsule between phone 1 and the wrist hides it from phone 1 only; the COCO errors then count only phone 0's detections"""
    obs = _FakeObs([[0, 0, 3.0], [3.0, 0, 0]])
    ks = np.arange(2)
    X = np.zeros((2, 3))
    A = np.array([[[1.5, -0.2, 0]]] * 2)
    B = np.array([[[1.5, 0.2, 0]]] * 2)
    monkeypatch.setattr(zc, "occluders", lambda bodies, kl, kf: (A, B, np.array([0.1])))
    vis = zc.joint_visible(obs, ks, X, None, None, None)
    assert vis.tolist() == [[True, False], [True, False]]

    class Judge:
        def joint(self, c, role, side, k, which):
            uv, _ = obs.project(c, X[:1, None], np.array([k]))
            return uv[0, 0] + (np.array([5.0, 0]) if c == "0" else np.array([50.0, 0])), True
    e = zc.coco_errors_vis(Judge(), obs, X, ks, "wrist", vis)
    assert np.allclose(e, [5.0, 5.0])


def test_observed_thru_moves_the_palm_to_the_observed_hand_at_its_depth(monkeypatch):
    """MAMMA shows the hand 30 px right of the motion's: the palm moves so that it projects 30 px right, at the same depth"""
    from dancecap import hand_contact as hc
    T = 6
    obs = _FakeObs([[0, 0, 3.0]], T)
    ks = np.arange(T)
    nlm = 12
    lm_model = np.random.default_rng(0).normal(0, 0.03, (T, nlm, 3))
    uv, _ = obs.project("0", lm_model, ks)
    monkeypatch.setattr(hc, "landmark_parts", lambda: {"right_hand": np.arange(nlm)})
    monkeypatch.setattr(hc, "PIDX", {"lead": 0, "follow": 1})
    obs.z["0"]["lm"] = np.zeros((T, 2, nlm, 2))
    obs.z["0"]["lm"][:, 0] = uv + [30.0, 0]
    obs.z["0"]["vis"] = np.ones((T, 2, nlm))
    pc = np.zeros((T, 3))
    X, used = zc.observed_thru(obs, "0", ks, lm_model, pc, np.ones(T, bool), zc.HoldCfg(), min_lm=6)
    assert used.all()
    u1, d1 = obs.project("0", X[:, None], ks)
    u0, d0 = obs.project("0", pc[:, None], ks)
    assert np.allclose(u1[:, 0] - u0[:, 0], [30.0, 0], atol=1e-6) and np.allclose(d1, d0)


def test_rate_limit_spreads_a_push_and_bounds_its_change():
    d = np.zeros((20, 3))
    d[10] = [0.10, 0, 0]                                                  # a 10 cm push on one frame
    r = zc.rate_limit(d, 0.015)
    assert np.linalg.norm(np.diff(r, axis=0), axis=1).max() <= 0.015 + 1e-9
    assert r[10, 0] > 0.05 and r[7, 0] > 0.0 and r[13, 0] > 0.0          # kept large where needed, spread to the neighbours


def test_unfold_and_smooth_wrist_stay_continuous_through_180_degrees():
    """a fold passing 180 deg flips the shortest arc between frames; the unfolded / smoothed wrist must not flick (2026-10-22, 348.1 s)"""
    from scipy.spatial.transform import Rotation as Ro
    a, n, _ = zc._rest_axes("right")
    flex_axis = np.cross(a, n)
    T = 12
    P = np.zeros((T, 55, 3))
    for k in range(T):
        ang = np.radians(170 + 2.0 * k)                                   # 170 -> 192 deg about one axis
        P[k, zc.WRIST["right"]] = Ro.from_rotvec(ang * flex_axis).as_rotvec()   # scipy wraps it: the axis flips past 180
    Q, _ = zc.unfold_wrist(P)
    assert zc._local_rot_speed(Q, zc.WRIST["right"]).max() < 5.0
    S = zc.smooth_wrist(P, 2.0)
    assert zc._local_rot_speed(S, zc.WRIST["right"]).max() < 5.0


def test_whip_frames_added_needs_the_window_s_own_motion_and_speed():
    """the closing pass's whip: added against the input AND against the start AND moving at least whip_min_mps; a fast frame stays a whip"""
    n = 5
    a0 = np.zeros(n)
    a1 = np.full(n, 0.7)
    v_in = np.full(n, 0.3)
    v0 = np.full(n, 0.3)
    v1 = np.array([0.5, 0.9, 0.9, 0.9, 3.0])
    a_start = np.array([0.7, 0.7, 0.2, 0.7, 0.7])
    b = zc.whip_frames_added(a0, a1, a_start, v_in, v0, v1, p99=2.0)
    assert b.tolist() == [False, True, False, True, True]


def test_mamma_swapped_when_the_right_hand_is_nearer_his_left():
    from dancecap import hand_contact as hc

    class O:
        cams = ["0"]
        z = {"0": {"lm": None, "vis": None}}

        def project(self, c, X, ks):
            return X[..., :2] * 100.0, X[..., 2]
    o = O()
    nlm = 8
    old = (hc.landmark_parts, hc.PIDX)
    try:
        hc.landmark_parts = lambda: {"right_hand": np.arange(nlm)}
        hc.PIDX = {"lead": 0, "follow": 1}
        lm_r = np.zeros((2, nlm, 3))
        lm_r[..., 2] = 3
        lm_l = lm_r.copy()
        lm_l[..., 0] = 1.0                                                # his left hand 100 px to the right
        obs_lm = np.zeros((2, 2, nlm, 2))
        obs_lm[0, 0] = [5.0, 0]                                           # frame 0: MAMMA's right hand at the right hand
        obs_lm[1, 0] = [95.0, 0]                                          # frame 1: at his left hand (swapped)
        o.z["0"]["lm"] = obs_lm + 1e-3
        o.z["0"]["vis"] = np.ones((2, 2, nlm))
        sw = zc.mamma_swapped(o, "0", np.arange(2), [lm_r], lm_l, zc.HoldCfg())
        assert sw.tolist() == [False, True]
    finally:
        hc.landmark_parts, hc.PIDX = old


def test_window_check_vetoes_relative_coco_threshold():
    """the closing pass: where the motion already misses the COCO wrist by 30 px, 6 px worse is within 0.3 x 30; 12 px worse is not"""
    wins = [{"id": "a", "t_span": [0, 1]}]
    chk = {"a": {"n": 50, "v3_px": 30.0, "v4_px": 36.0}}
    assert zc.window_check_vetoes(chk, wins) != {}
    assert zc.window_check_vetoes(chk, wins, coco_rel=0.3) == {}
    chk["a"]["v4_px"] = 42.0
    assert zc.window_check_vetoes(chk, wins, coco_rel=0.3) != {}


# ------------------------------------------------------------------------------------------------ the last pass


def _speeds(T, **over):
    s = {"pelvis": np.ones(T), "spine3": np.ones(T), "root": np.full(T, 0.1)}
    s.update(over)
    return s


def test_spike_ranges_short_deep_and_fold_runs_only():
    """runs of <= 5 frames deeper than 5 cm or bent past 120 deg are spikes (runs 1 frame apart joined); a 6-frame run is not (steps 2 / 3);
    the kind is "arm" unless the pelvis / spine / root move much faster inside the run than around it"""
    T = 200
    pen = np.zeros(T)
    bend = np.full(T, 30.0)
    pen[20:22] = 0.09                                                   # 2 frames, 9 cm
    pen[40:46] = 0.07                                                   # 6 frames: not a spike
    pen[60:62] = 0.06
    pen[63:65] = 0.06                                                   # 1 frame apart: one run of 5 frames
    bend[100:103] = 150.0                                               # a short fold
    pen[150] = 0.04                                                     # under 5 cm: nothing
    out = zc.spike_ranges(pen, bend, _speeds(T))
    assert [(s["a"], s["b"]) for s in out] == [(20, 21), (60, 64), (100, 102)]
    assert [s["why"] for s in out] == [["deep"], ["deep"], ["fold"]]
    assert all(s["kind"] == "arm" for s in out)
    assert out[0]["pen_cm"] == 9.0 and out[2]["bend_deg"] == 150.0
    # the pelvis jumps 12 deg / frame inside the first run (2 deg around it): the whole body
    pel = np.full(T, 2.0)
    pel[21] = 12.0
    out = zc.spike_ranges(pen, bend, _speeds(T, pelvis=pel))
    assert out[0]["kind"] == "body" and out[0]["body_why"] == ["pelvis"] and out[1]["kind"] == "arm"
    # ... but not when it moves that fast around the run as well (a fast turn, not a glitch)
    pel[8:34] = 12.0
    assert zc.spike_ranges(pen, bend, _speeds(T, pelvis=pel))[0]["kind"] == "arm"


def test_spike_knots_skip_other_spikes_and_need_free_neighbours():
    T = 50
    blocked = np.zeros(T, bool)
    blocked[10:12] = True
    blocked[15:17] = True
    K = zc.spike_knots(10, 11, T, 4, blocked)
    assert K.tolist() == [6, 7, 8, 9, 12, 13, 14, 17]                   # frames 15-16 (the next spike) skipped
    assert zc.spike_knots(0, 2, T, 4) is None                           # no frame before it
    assert zc.spike_knots(2, 3, T, 4).tolist() == [0, 1, 4, 5, 6, 7]    # fewer knots where the film starts
    b2 = np.zeros(T, bool)
    b2[19:23] = True
    assert zc.spike_knots(20, 22, T, 4, b2) is None                     # the frame just before the run is itself blocked


def test_despike_c2_reproduces_a_smooth_motion_and_keeps_the_knots():
    """a rotation turning along a cubic in time with a 3-frame glitch: the C2 spline through 4 frames on each side gives the cubic back
    exactly; the knots and every other joint are untouched for "arm"; "body" also splines the root and keeps the legs' global rotations"""
    from scipy.spatial.transform import Rotation as Ro
    T = 40
    P = np.zeros((T, 55, 3))
    tr = np.zeros((T, 3))
    ang = lambda k: 0.5 * k + 0.02 * k ** 2 - 0.0004 * k ** 3                      # noqa: E731  (deg)
    for k in range(T):
        P[k, zc.WRIST["right"]] = Ro.from_euler("z", ang(k), degrees=True).as_rotvec()
        P[k, zc.SPINE3] = Ro.from_euler("y", 1.0 * k, degrees=True).as_rotvec()
        P[k, zc.PELVIS] = Ro.from_euler("y", 0.5 * k, degrees=True).as_rotvec()
        P[k, zc.HIP["left"]] = Ro.from_euler("x", 0.3 * k, degrees=True).as_rotvec()
        tr[k] = [0.01 * k + 0.0005 * k ** 2, 0, 0]
    clean = P.copy()
    clean_tr = tr.copy()
    P[20:23, zc.WRIST["right"]] = Ro.from_euler("x", 70.0, degrees=True).as_rotvec()
    P[20:23, zc.SPINE3] = Ro.from_euler("x", 20.0, degrees=True).as_rotvec()
    P[20:23, zc.PELVIS] = Ro.from_euler("x", 15.0, degrees=True).as_rotvec()
    tr[20:23] += [0, 0.1, 0]
    K = zc.spike_knots(20, 22, T, 4)
    Q, trq = zc.despike_c2(P, tr, 20, 22, K, "arm")
    assert np.allclose(Q[:, zc.WRIST["right"]], clean[:, zc.WRIST["right"]], atol=1e-9)
    assert np.array_equal(Q[:, zc.SPINE3], P[:, zc.SPINE3]) and np.array_equal(trq, tr)
    assert np.array_equal(Q[K], P[K])
    Q2, trq2 = zc.despike_c2(P, tr, 20, 22, K, "body")
    assert np.allclose(Q2[:, zc.SPINE3], clean[:, zc.SPINE3], atol=1e-9) and np.allclose(trq2, clean_tr, atol=1e-12)
    # the legs keep their global rotations under the despiked pelvis (knees etc. unchanged, the hip re-expressed)
    for k in range(20, 23):
        g_old = Ro.from_rotvec(P[k, zc.PELVIS]) * Ro.from_rotvec(P[k, zc.HIP["left"]])
        g_new = Ro.from_rotvec(Q2[k, zc.PELVIS]) * Ro.from_rotvec(Q2[k, zc.HIP["left"]])
        assert (g_old.inv() * g_new).magnitude() < 1e-9
    assert np.array_equal(Q2[:, zc.KNEE["left"]], P[:, zc.KNEE["left"]])


def test_envelope_is_a_raised_cosine_blend():
    e = zc.envelope(15, 4, 15)
    assert len(e) == 34 and np.all(e[15:19] == 1.0)
    assert np.all((e[:15] > 0) & (e[:15] < 1)) and np.all(np.diff(e[:16]) > 0)
    assert np.allclose(e[:15], e[19:][::-1])
    assert e[0] < 0.02                                                  # it starts from (nearly) nothing


def test_last_windows_cores_merge_and_flags():
    T = 400
    pen = np.zeros(T)
    gap = np.full(T, 0.08)
    bend = np.full(T, 40.0)
    pen[100:110] = 0.06                                                 # push frames
    bend[120:140] = 105.0
    bend[128:132] = 130.0                                               # a fold: its whole run past 100 deg is unfolded
    bend[300:310] = 110.0                                               # past 100 deg but never past 120: left alone
    gap[90:150] = 0.0
    cfg = zc.LastCfg()
    w = zc.last_windows(pen, gap, bend, cfg)
    assert len(w) == 1                                                  # 10 frames apart: one window
    w = w[0]
    assert w["core"] == [100 - cfg.pad, 139 + cfg.pad]
    assert w["span"] == [100 - cfg.pad - cfg.ramp, 139 + cfg.pad + cfg.ramp]
    assert w["push"] == list(range(100, 110)) and w["unfold"] == list(range(120, 140))
    assert w["touch"][0] == 90 and w["touch"][-1] == 149 and w["pull"] == []
    # the back holds' floating frames (2-5 cm off her) inside a window are pulled onto her; farther ones are left alone
    gap[60:90] = 0.03
    gap[70:75] = 0.09
    back = np.zeros(T, bool)
    back[50:200] = True
    w = zc.last_windows(pen, gap, bend, cfg, back=back)[0]
    span = range(w["span"][0], w["span"][1] + 1)
    assert w["pull"] == [k for k in span if 60 <= k < 90 and not 70 <= k < 75]


def test_sphere_dirs_and_pick_exit():
    D = zc.sphere_dirs(48)
    assert np.allclose(np.linalg.norm(D, axis=1), 1.0) and abs(D.mean(0)).max() < 0.05
    dirs = np.eye(3)
    # frame 1's shortest exit is along y (5.7 cm) but its x exit (6 cm) is far nearer frame 0's and frame 2's: the path stays on x
    t = np.array([[0.05, 0.09, np.inf],
                  [0.06, 0.057, np.inf],
                  [0.05, 0.09, np.inf],
                  [0.04, 0.02, np.inf]])                                # frame 3: y is shorter by 2 cm, worth one jump
    d, ok = zc.pick_exit(t, dirs, np.zeros((4, 3)), 0.095, w_jump=0.3)
    assert ok.all()
    assert np.allclose(d[:3], [[0.05, 0, 0], [0.06, 0, 0], [0.05, 0, 0]]) and np.allclose(d[3], [0, 0.02, 0])
    # a frame without an exit takes the cap along the direction with the least depth left
    t2 = np.array([[0.05, np.inf, np.inf], [np.inf, np.inf, np.inf]])
    left = np.array([[0.0, 0.0, 0.0], [0.03, 0.02, 0.05]])
    d, ok = zc.pick_exit(t2, dirs, left, 0.095, w_jump=0.0)
    assert ok.tolist() == [True, False] and np.allclose(d[1], [0, 0.095, 0])


def _synthetic_arm_window(W=6, nl=2, nr=2):
    """an ArmWindow built by hand: a straight right arm along -x (shoulder at 0, elbow at -0.3, wrist at -0.55), identity rotations, two
    hand points rigid with the wrist, one forearm point half on the elbow"""
    torch = pytest.importorskip("torch")
    w = object.__new__(zc.ArmWindow)
    w.torch = torch
    w.ks = np.arange(W)
    w.nl, w.nr, w.nc = nl, nr, W - nl - nr
    t = lambda a: torch.tensor(np.asarray(a, float), dtype=torch.float64)  # noqa: E731
    eye = np.tile(np.eye(3), (W, 1, 1))
    w.G14 = w.G17 = w.G19 = w.G21 = t(eye)
    w.R19 = t(eye)
    w.S, w.E, w.Wr = t(np.zeros((W, 3))), t(np.tile([-0.3, 0, 0], (W, 1))), t(np.tile([-0.55, 0, 0], (W, 1)))
    w.b1, w.b2 = t([-0.3, 0, 0]), t([-0.25, 0, 0])
    w.ea = w.b2 / w.b2.norm()
    V = np.tile(np.array([[-0.62, 0.0, 0.02], [-0.65, 0.01, 0.0], [-0.45, 0.0, 0.03]]), (W, 1, 1))
    w.Vt = t(V)
    w.V0 = V
    w.A17 = w.Vt - w.S[:, None]
    w.A19 = w.Vt - w.E[:, None]
    w.A21 = w.Vt - w.Wr[:, None]
    w.w17, w.w19, w.wh = (t(x)[None, :, None] for x in ([0, 0, 0], [0, 0, 0.5], [1, 1, 0.5]))
    w.C0 = t(np.tile([-0.6, 0, 0], (W, 1)))
    w.tips0 = t(np.tile([[-0.75, 0, 0]] * 5, (W, 1, 1)))
    w.Cl = w.C0 - w.Wr
    w.Tl = w.tips0 - w.Wr[:, None]
    w.env = t(zc.envelope(nl, w.nc, nr))
    w.ey = t([0.0, 1.0, 0.0])
    w.Pw = np.zeros((W, 55, 3))
    return w, torch


def test_arm_window_zero_is_the_base_and_the_numbers_do_what_they_say():
    from scipy.spatial.transform import Rotation as Ro
    w, torch = _synthetic_arm_window()
    W = len(w.ks)
    f = w.forward(torch.zeros((W, w.NV), dtype=torch.float64))
    assert np.allclose(f["X"].numpy(), w.V0) and np.allclose(f["W"].numpy(), w.Wr.numpy()) and np.allclose(f["C"].numpy(), w.C0.numpy())
    # the forearm's own twist moves no joint and keeps the hand's points (the hand's world rotation is a separate number)
    v = torch.zeros((W, w.NV), dtype=torch.float64)
    v[:, 4] = 0.7
    f = w.forward(v)
    assert np.allclose(f["E"].numpy(), w.E.numpy()) and np.allclose(f["W"].numpy(), w.Wr.numpy())
    assert np.allclose(f["X"].numpy()[:, :2], w.V0[:, :2])
    # the elbow's hinge: the forearm swings in the elbow's flexion plane (no off-hinge part), the elbow stays
    v = torch.zeros((W, w.NV), dtype=torch.float64)
    v[:, 3] = 0.5
    f = w.forward(v)
    P1 = np.zeros((1, 55, 3))
    P1[0, zc.ELBOW["right"]] = Ro.from_matrix(f["R19"].numpy()[w.nl]).as_rotvec()
    ang = zc.elbow_angles(P1, "right")
    assert abs(ang["off_hinge"][0]) < 1e-6 and abs(abs(ang["flexion"][0]) - np.degrees(0.5)) < 1e-6
    assert np.allclose(f["E"].numpy(), w.E.numpy())
    assert np.linalg.norm(f["W"].numpy()[w.nl] - w.Wr.numpy()[w.nl]) == pytest.approx(2 * 0.25 * np.sin(0.25), abs=1e-9)
    # a world rotation of the hand turns its points about the wrist, the wrist stays
    v = torch.zeros((W, w.NV), dtype=torch.float64)
    v[:, 5:8] = torch.tensor([0.0, 0.0, 0.4])
    f = w.forward(v)
    assert np.allclose(f["W"].numpy(), w.Wr.numpy())
    r = Ro.from_rotvec([0, 0, 0.4]).as_matrix()
    assert np.allclose(f["C"].numpy()[w.nl], r @ (w.C0.numpy()[w.nl] - w.Wr.numpy()[w.nl]) + w.Wr.numpy()[w.nl])
    # every frame's numbers times the envelope: the change fades to (nearly) nothing at the window's ends
    v = torch.zeros((W, w.NV), dtype=torch.float64)
    v[:, 0:3] = torch.tensor([0.0, 0.0, 0.3])
    p = w.params(v).numpy()
    assert np.allclose(p[w.nl:w.nl + w.nc, 2], 0.3) and np.allclose(p[:, 2], 0.3 * w.env.numpy())
