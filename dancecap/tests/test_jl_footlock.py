r"""Unit tests for the pure helpers of dancecap.mono4d.jl_footlock (synthetic numbers; no footage, no SMPL-X model file).

    .venv-nopo4d\Scripts\python.exe -m pytest dancecap/tests/test_jl_footlock.py -q      (from the workspace root)
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dancecap import physics_body as pb  # noqa: E402
from dancecap.mono4d import jl_footlock as FL  # noqa: E402

SMPLX_PARENTS = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 15, 15, 15, 20, 25, 26, 20, 28, 29,
                          20, 31, 32, 20, 34, 35, 20, 37, 38, 21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50, 21, 52, 53])


def camera(f=808.0, height=1.43, pitch_deg=12.0, yaw_deg=0.0, pos=(0.0, 0.0)):
    """World -> camera (OpenCV: +Z forward, +Y down) for a camera at (pos[0], height, pos[1]) looking along -Z (yawed,
    pitched down)."""
    p, y = np.deg2rad(pitch_deg), np.deg2rad(yaw_deg)
    fwd = np.array([-np.sin(y) * np.cos(p), -np.sin(p), -np.cos(y) * np.cos(p)])
    right = np.array([np.cos(y), 0.0, -np.sin(y)])
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd])
    C = np.array([pos[0], height, pos[1]])
    K = np.array([[f, 0, 269.0], [0, f, 480.0], [0, 0, 1.0]])
    return K, R, -R @ C, C


def test_ray_floor_roundtrip_with_k1():
    K, R, t, C = camera()
    X = np.array([[0.3, 0.02, -4.2], [-0.5, 0.09, -3.6], [0.1, 0.0, -5.0]])
    k1 = -0.0123
    uv = FL.project(X[None], R[None], t[None], K[None], k1)[0]
    d = FL.pixel_rays(uv, np.repeat(K[None], 3, 0), np.repeat(R[None], 3, 0), k1)
    P = FL.ray_plane_y(np.repeat(C[None], 3, 0), d, X[:, 1])
    assert np.allclose(P, X, atol=1e-6)
    assert np.allclose(FL.camera_centre(R, t), C)


def test_ray_above_horizon_is_nan():
    K, R, t, C = camera(pitch_deg=0.0)
    d = FL.pixel_rays(np.array([[269.0, 100.0]]), K[None], R[None])      # above the principal point = above the horizon
    assert np.isnan(FL.ray_plane_y(C[None], d, 0.0)).all()


def test_rotation_homography_and_compensated_speed():
    K0, R0, t0, C = camera(yaw_deg=0.0)
    K1, R1, _, _ = camera(f=900.0, yaw_deg=4.0, pitch_deg=13.0)
    t1 = -R1 @ C                                                           # same centre: pure rotation + zoom
    X = np.array([[0.2, 0.05, -4.0]])
    u0 = FL.project(X[None], R0[None], t0[None], K0[None])[0]
    u1 = FL.project(X[None], R1[None], t1[None], K1[None])[0]
    H = FL.rotation_homography(K0, R0, K1, R1)
    assert np.allclose(FL.apply_h(H, u0), u1, atol=1e-6)
    Ks, Rs = np.stack([K0, K1]), np.stack([R0, R1])
    s = FL.compensated_speed(np.concatenate([u0, u1]), Ks, Rs)
    assert np.isnan(s[0]) and s[1] < 1e-6                                 # a static point does not move once the camera is removed
    X2 = X + [0.05, 0.0, 0.0]                                              # the foot moved 5 cm sideways
    u1b = FL.project(X2[None], R1[None], t1[None], K1[None])[0]
    s2 = FL.compensated_speed(np.concatenate([u0, u1b]), Ks, Rs)
    assert 5.0 < s2[1] < 20.0                                              # ~ f * 0.05 / 4.2 = 10.7 px at f 900


def test_runs_gaps_and_ramps():
    m = np.array([0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1], bool)
    assert FL.runs_of(m) == [(1, 2), (4, 7), (10, 10)]
    assert FL.runs_of(m, 3) == [(4, 7)]
    assert FL.close_gaps(m, 1).tolist() == [0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1]
    w = FL.ramp_weights(12, [(5, 6)], 2)
    assert w[5] == w[6] == 1.0 and 0 < w[4] < 1 and w[3] < w[4] and w[2] == 0 and w[8] > 0 and w[9] == 0


def test_held_delta_holds_and_bridges():
    T = 30
    d = np.zeros((T, 1))
    d[5:10] = np.arange(5, 10)[:, None]                                   # run 1: values 5..9
    d[13:16] = -1.0                                                       # run 2, gap of 3 frames (<= 2 n)
    d[25:27] = 2.0                                                        # run 3, far
    out, w = FL.held_delta(d, [(5, 9), (13, 15), (25, 26)], n=2)
    assert out[3, 0] == out[4, 0] == 5.0                                  # head of run 1 holds its first value
    assert np.allclose(out[10:13, 0], [9 + (-10) * x for x in (0.25, 0.5, 0.75)])   # gap bridged linearly
    assert (w[10:13] == 1).all()
    assert out[16, 0] == out[17, 0] == -1.0 and 0 < w[17] < w[16] < 1     # tail of run 2 held under the fade
    assert out[27, 0] == out[28, 0] == 2.0 and w[29] == 0


def test_bridge_and_bound():
    m = np.array([[0.0, 0, 0], [0, 0, 0], [2, 0, 2], [0, 0, 0], [4, 0, 0]])
    have = np.array([False, False, True, False, True])
    b = FL.bridge(m, have)
    assert np.allclose(b[:, 0], [2, 2, 2, 3, 4]) and np.allclose(b[:, 2], [2, 2, 2, 1, 0])
    assert np.allclose(FL.bridge(m, np.zeros(5, bool)), 0)
    c = FL.bound_norm(np.array([[0.03, 0, 0.04], [0.001, 0, 0]]), 0.02)
    assert np.isclose(np.linalg.norm(c[0]), 0.02) and np.allclose(c[1], [0.001, 0, 0])


def test_two_bone_keeps_lengths_bend_side_and_clamps():
    rng = np.random.default_rng(0)
    H = np.zeros((50, 3))
    K = np.array([0.0, -0.42, 0.06]) + 0.01 * rng.standard_normal((50, 3))   # knee in front (+z) of the hip-ankle line
    A = np.array([0.0, -0.84, -0.02]) + 0.01 * rng.standard_normal((50, 3))
    At = A + 0.03 * rng.standard_normal((50, 3))
    Kn, Ar, cl = FL.two_bone(H, K, A, At)
    la, lb = np.linalg.norm(K - H, axis=1), np.linalg.norm(A - K, axis=1)
    assert np.allclose(np.linalg.norm(Kn - H, axis=1), la, atol=1e-9)
    assert np.allclose(np.linalg.norm(Ar - Kn, axis=1), lb, atol=1e-9)
    reach = np.linalg.norm(At - H, axis=1) < la + lb - 1e-4
    assert np.allclose(Ar[reach], At[reach], atol=1e-9) and not cl[reach].any()
    assert (Kn[:, 2] > Ar[:, 2] - 1e-9).all() or (np.sign(Kn[:, 2] - 0.5 * Ar[:, 2]) > 0).all()   # bend stays forward
    Kf, Af, clf = FL.two_bone(H[:1], K[:1], A[:1], np.array([[0.0, -2.0, 0.0]]))   # out of reach: straight, clamped
    assert clf.all() and np.isclose(np.linalg.norm(Af - H[:1]), la[0] + lb[0] - 1e-4)


def test_knee_flexion_and_path_stats():
    J = np.zeros((2, 12, 3))
    J[:, 1] = [0, 0.9, 0]
    J[0, 4], J[0, 7] = [0, 0.5, 0], [0, 0.1, 0]                            # straight
    J[1, 4], J[1, 7] = [0, 0.5, 0], [0, 0.1 + 0.4 * (1 - np.cos(np.pi / 6)), -0.4 * np.sin(np.pi / 6)]   # shank bent 30 deg back
    k = FL.knee_flexion_deg(J, "left")
    assert np.isclose(k[0], 0, atol=1e-2) and np.isclose(k[1], 30, atol=1e-4)
    P = np.array([[0.0, 0.0], [0.01, 0.0], [0.01, 0.01], [0.0, 0.01]])
    s = FL.path_stats(P)
    assert np.isclose(s["path_cm"], 3.0) and np.isclose(s["mm_per_frame"], 10.0) and np.isclose(s["net_cm"], 1.0)


def test_own_dancer_rules():
    d_own = np.array([0.0, 4.0, 20.0, 20.0, 0.0, 60.0])
    d_par = np.array([30.0, 0.0, 30.0, 5.0, 0.0, 100.0])
    ok = FL.own_dancer(d_own, d_par, tol=6, far_tol=40, clear=12)
    # inside own; on own edge but inside the partner only -> no; near & clear of partner; near but touching partner; inside both; far
    assert ok.tolist() == [True, False, True, False, True, False]


def test_along_across_signs():
    C = np.array([0.0, 1.4, 0.0])
    Q = np.array([[0.0, 0.0, -4.0]])
    d, al, ac = FL.along_across(Q + [0.0, 0.0, -0.05], Q, C)              # 5 cm farther from the camera
    assert np.isclose(al[0], 0.05) and np.isclose(ac[0], 0.0) and np.isclose(d[0], 0.05)
    d, al, ac = FL.along_across(Q + [0.03, 0.0, 0.0], Q, C)
    assert np.isclose(al[0], 0.0) and np.isclose(abs(ac[0]), 0.03)


def test_angles():
    assert np.isclose(FL.circ_mean(np.array([np.pi - 0.1, -np.pi + 0.1])), np.pi) or np.isclose(FL.circ_mean(np.array([np.pi - 0.1, -np.pi + 0.1])), -np.pi)
    assert np.allclose(FL.wrap_pi(np.array([3 * np.pi / 2, -3 * np.pi / 2, 0.2])), [-np.pi / 2, np.pi / 2, 0.2])


def _toy_body():
    rng = np.random.default_rng(3)
    j = np.zeros((55, 3))
    j[0] = [0, 0.95, 0]
    j[1], j[2] = [0.08, 0.86, 0], [-0.08, 0.86, 0]
    j[4], j[5] = [0.1, 0.48, 0.01], [-0.1, 0.48, 0.01]
    j[7], j[8] = [0.1, 0.09, -0.03], [-0.1, 0.09, -0.03]
    j[10], j[11] = [0.12, 0.02, 0.1], [-0.12, 0.02, 0.1]
    for i in range(55):
        if not j[i].any():
            j[i] = j[max(SMPLX_PARENTS[i], 0)] + 0.05 * rng.standard_normal(3)
    return SimpleNamespace(parents=SMPLX_PARENTS, j_rest=j)


def test_solve_leg_reaches_targets_through_fk():
    body = _toy_body()
    rng = np.random.default_rng(1)
    T = 6
    pose = 0.15 * rng.standard_normal((T, 55, 3))
    pose[:, 4, 0] = 0.4                                                    # bent left knee
    transl = 0.01 * rng.standard_normal((T, 3))
    R, p = pb.forward_kinematics(body, pose, transl)
    J = np.einsum("tjab,jb->tja", R, body.j_rest) + p
    shift = np.array([0.02, 0.0, -0.015])
    A_des, T_des = J[:, 7] + shift, J[:, 10] + shift + [0.0, 0.0, 0.01]   # move the foot and turn it a little
    bp = pose[:, 1:22].copy()
    fr = np.arange(T)
    info = FL.solve_leg(body, R, J, bp, "left", fr, A_des, T_des)
    pose2 = pose.copy()
    pose2[:, 1:22] = bp
    R2, p2 = pb.forward_kinematics(body, pose2, transl)
    J2 = np.einsum("tjab,jb->tja", R2, body.j_rest) + p2
    assert info["clamped"] == 0
    assert np.allclose(J2[:, 7], A_des, atol=1e-6)                        # the ankle is where asked
    v, vd = J2[:, 10] - J2[:, 7], T_des - A_des
    cosang = np.sum(v * vd, 1) / (np.linalg.norm(v, axis=1) * np.linalg.norm(vd, axis=1))
    assert (cosang > 1 - 1e-9).all()                                       # the foot points at the toe target
    assert np.allclose(J2[:, :4], J[:, :4], atol=1e-12) and np.allclose(J2[:, 8], J[:, 8], atol=1e-12)   # pelvis, other leg untouched


def test_theil_sen_drift():
    rng = np.random.default_rng(5)
    y = 0.004 * np.arange(40) + 0.003 * rng.standard_normal(40)
    d, e = FL.theil_sen_drift(y)
    assert abs(d - 0.004 * 39) < 0.01 and e < 0.005
    d0, _ = FL.theil_sen_drift(np.r_[0.01 * rng.standard_normal(20), np.nan, 0.01 * rng.standard_normal(9)])
    assert abs(d0) < 0.03
    assert FL.theil_sen_drift(np.array([np.nan, 1.0]))[1] == np.inf


def test_split_still_drops_a_slow_slide():
    rng = np.random.default_rng(2)
    still1 = 0.006 * rng.standard_normal(30)
    slide = 0.004 * np.arange(1, 41)                                         # 4 mm / frame sideways for 40 frames (16 cm)
    still2 = slide[-1] + 0.006 * rng.standard_normal(25)
    lat = np.r_[still1, slide, still2]
    al = 0.02 * rng.standard_normal(len(lat))
    segs = FL.split_still(lat, al, 0.025, 0.08, 0.015, 0.06, 4)
    covered = np.zeros(len(lat), bool)
    for a, b in segs:
        covered[a:b + 1] = True
    assert covered[:25].all() and covered[-20:].all()                       # both still parts kept
    assert covered[38:62].sum() <= 6                                         # (almost) nothing of the slide is pinned
    one = FL.split_still(0.006 * rng.standard_normal(50), 0.02 * rng.standard_normal(50), 0.025, 0.08, 0.015, 0.06, 4)
    assert one == [(0, 49)]


def test_heel_raise_reach():
    H = np.array([[0.0, 0.9, 0.0]])
    B = np.array([[0.0, 0.02, 0.25]])                                        # ball far forward of the hip
    A = np.array([[0.0, 0.08, 0.12]])                                        # ankle behind the ball
    reach = np.array([np.linalg.norm(A[0] - H[0]) - 0.02])                   # 2 cm short
    A2, add = FL.heel_raise_reach(H, A, B, reach)
    assert add[0] > 0 and A2[0, 1] > A[0, 1]                                 # the heel went up
    assert np.isclose(np.linalg.norm(A2[0] - B[0]), np.linalg.norm(A[0] - B[0]))   # about the ball
    assert np.linalg.norm(A2[0] - H[0]) <= reach[0] + 1e-9
    A3, add3 = FL.heel_raise_reach(H, A, B, np.array([2.0]))                 # reachable: unchanged
    assert add3[0] == 0 and np.allclose(A3, A)


def test_two_bone_forward_pole_never_bends_backwards():
    H = np.zeros((2, 3))
    K = np.array([[0.0, -0.42, -0.0005], [0.0, -0.42, 0.001]])              # 0: almost straight, knee a hair behind; 1: almost straight
    A = np.array([[0.0, -0.84, 0.0], [0.0, -0.84, 0.0]])
    At = A + [0.0, 0.05, 0.0]                                                # bring the ankle 5 cm up: the knee must bend
    fwd = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    Kn, Ar, _ = FL.two_bone(H, K, A, At, fwd=fwd)
    assert (Kn[:, 2] > 0.01).all()                                           # both knees now bend forward (+z)
    Kb, _, _ = FL.two_bone(H, K, A, At)                                      # without the hint the near-straight knee follows its noise
    assert Kb[0, 2] < 0


def test_two_bone_keeps_a_clearly_bent_back_knee_side():
    H = np.zeros((1, 3))
    K = np.array([[0.0, -0.40, -0.12]])                                      # source knee bent backwards ~33 deg (a fit error)
    A = np.array([[0.0, -0.80, 0.0]])
    Kn, _, _ = FL.two_bone(H, K, A, A + [0.0, 0.02, 0.0], fwd=np.array([[0.0, 0.0, 1.0]]))
    assert Kn[0, 2] < 0                                                      # the lock does not flip it (the facing fix will)


def test_separate_runs():
    assert FL.separate_runs([(0, 9), (10, 19)], 4, 3) == [(0, 7), (12, 19)]          # touching -> 4-frame gap
    assert FL.separate_runs([(0, 9), (12, 19)], 4, 3) == [(0, 8), (13, 19)]
    assert FL.separate_runs([(0, 9), (20, 29)], 4, 3) == [(0, 9), (20, 29)]          # far apart: unchanged
    out = FL.separate_runs([(0, 9), (10, 12), (13, 25)], 4, 3)
    assert all(y[0] - x[1] - 1 >= 4 for x, y in zip(out[:-1], out[1:])) and all(b - a + 1 >= 3 for a, b in out)
