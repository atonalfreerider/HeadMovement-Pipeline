"""dancecap.pose_refine on tiny synthetic inputs (no body models, no landmarks)."""
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from dancecap import physics_body as pb
from dancecap import pose_refine as pr

# SMPL-X body parents (joints 0-21); the rest of the 55 hang off the head / wrists and do not matter here
PARENTS = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19] + [15] * 3 + [20] * 15
                   + [21] * 15)


def _leg_body():
    J = np.zeros((55, 3))
    J[1], J[2] = [0.09, -0.08, 0.0], [-0.09, -0.08, 0.0]       # hips
    J[4], J[5] = [0.1, -0.48, 0.0], [-0.1, -0.48, 0.0]         # knees
    J[7], J[8] = [0.1, -0.88, -0.02], [-0.1, -0.88, -0.02]     # ankles
    J[10], J[11] = [0.11, -0.94, 0.11], [-0.11, -0.94, 0.11]   # feet
    J[3], J[6], J[9], J[12], J[15] = [0, 0.1, 0], [0, 0.25, 0], [0, 0.4, 0], [0, 0.55, 0], [0, 0.65, 0]
    return SimpleNamespace(j_rest=J, parents=PARENTS)


def _npz(T=5, knee=0.6):
    bp = np.zeros((T, 21, 3))
    bp[:, 3] = [knee, 0, 0]    # left knee (joint 4) flexed about x
    bp[:, 0] = [-0.3, 0, 0.1]  # left hip (joint 1)
    return {"t": np.arange(T) / 30.0, "global_orient": np.tile([0.0, 0.4, 0.0], (T, 1)), "body_pose": bp.reshape(T, 63),
            "transl": np.tile([0.2, 0.95, -0.1], (T, 1))}


def _fk(body, z):
    R, p = pb.forward_kinematics(body, pb.full_pose(z), np.asarray(z["transl"], float))
    return R, p, np.einsum("tjab,jb->tja", R, body.j_rest) + p


def test_leg_ik_moves_ankle_rigidly():
    body, z = _leg_body(), _npz()
    R, p, J = _fk(body, z)
    frames = np.array([1, 2, 3])
    delta = np.array([[0.02, -0.015, 0.0], [0.0, 0.01, -0.02], [-0.01, -0.02, 0.01]])
    rep = pr.leg_ik(body, z, R, J, "left", delta, frames)
    R2, _, J2 = _fk(body, z)
    assert rep["clamped"] == 0
    np.testing.assert_allclose(J2[frames, 7], J[frames, 7] + delta, atol=1e-6)       # ankle reached the target
    np.testing.assert_allclose(R2[frames, 7], R[frames, 7], atol=1e-6)               # ankle world orientation kept
    np.testing.assert_allclose(J2[frames, 10] - J2[frames, 7], J[frames, 10] - J[frames, 7], atol=1e-6)  # foot rigid
    np.testing.assert_allclose(np.linalg.norm(J2[:, 4] - J2[:, 1], axis=1), np.linalg.norm(J[:, 4] - J[:, 1], axis=1),
                               atol=1e-9)                                               # bones keep their lengths
    np.testing.assert_allclose(J2[[0, 4]], J[[0, 4]], atol=1e-12)                     # untouched frames
    np.testing.assert_allclose(J2[:, 8], J[:, 8], atol=1e-12)                          # other leg untouched


def test_leg_ik_clamps_unreachable_target():
    body, z = _leg_body(), _npz(T=2, knee=0.05)
    R, p, J = _fk(body, z)
    rep = pr.leg_ik(body, z, R, J, "left", np.array([[0.0, -0.3, 0.0]]), np.array([0]))
    assert rep["clamped"] == 1
    _, _, J2 = _fk(body, z)
    la = np.linalg.norm(J[0, 4] - J[0, 1]) + np.linalg.norm(J[0, 7] - J[0, 4])
    assert np.linalg.norm(J2[0, 7] - J2[0, 1]) <= la + 1e-6


def test_filter_rotations_sign_flip_and_lowpass():
    T = 120
    ang = np.linspace(0, 6 * np.pi, T)               # three full turns about Y: axis-angle wraps, quaternions flip
    aa = Rot.from_euler("y", ang[:, None]).as_rotvec()
    out = pr.filter_rotations(aa, 5.0, 30.0)
    err = np.degrees((Rot.from_rotvec(out) * Rot.from_rotvec(aa).inv()).magnitude())
    assert err[10:-10].max() < 0.5                    # a smooth turn passes unchanged (no wrap artefacts)
    rng = np.random.default_rng(0)
    noisy = (Rot.from_rotvec(rng.normal(0, 0.02, (T, 3))) * Rot.from_rotvec(aa)).as_rotvec()
    e_in = np.degrees((Rot.from_rotvec(noisy) * Rot.from_rotvec(aa).inv()).magnitude())
    e_out = np.degrees((Rot.from_rotvec(pr.filter_rotations(noisy, 5.0, 30.0)) * Rot.from_rotvec(aa).inv()).magnitude())
    assert e_out[10:-10].mean() < 0.6 * e_in[10:-10].mean()


def test_runs_and_ramps():
    m = np.array([0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1], bool)
    assert pr.runs_of(m, 3) == [(1, 3), (8, 11)]
    w = pr.ramp_weights(len(m), pr.runs_of(m, 3), 2)
    assert w[1] == w[3] == w[8] == 1.0 and 0 < w[4] < 1 and w[4] > w[5] > 0 and w[0] > 0


def test_min_rot():
    rng = np.random.default_rng(1)
    u, v = rng.normal(size=(6, 3)), rng.normal(size=(6, 3))
    Q = pr.min_rot(u, v)
    un, vn = u / np.linalg.norm(u, axis=1, keepdims=True), v / np.linalg.norm(v, axis=1, keepdims=True)
    np.testing.assert_allclose(np.einsum("nab,nb->na", Q, un), vn, atol=1e-9)
    np.testing.assert_allclose(pr.min_rot(u, u), np.tile(np.eye(3), (6, 1, 1)), atol=1e-9)


def test_contact_weights_hold_planted_feet_and_free_slides():
    stance = np.array([1, 1, 1, 1, 0, 1], bool)
    speed = np.array([0.0, 0.04, 0.15, 0.30, 0.0, 0.0])  # m/s: planted, planted, gliding, sliding, swing, planted
    w = pr.contact_weights(stance, speed, (0.05, 0.25))
    assert np.allclose(w, [1.0, 0.5, 0.0, 0.0, 0.0])  # each pair takes the lower of its two frames


def test_contact_slip_holds_the_patch_but_frees_pivots_and_rolls():
    import torch
    rng = np.random.default_rng(0)
    sole = np.c_[rng.uniform(-0.05, 0.05, 60), np.full(60, 0.03), rng.uniform(-0.12, 0.12, 60)]
    sole[:6] = np.c_[rng.uniform(-0.004, 0.004, 6), np.zeros(6), rng.uniform(-0.004, 0.004, 6)]  # the planted patch
    c = sole[:6].mean(0)

    def slip(next_frame):
        s = torch.tensor(np.stack([sole, next_frame]), dtype=torch.float64)
        return float(pr.contact_slip(s, torch.ones(1, dtype=torch.float64)))

    assert abs(slip(sole + [0.01, 0, 0]) - (10 / 3) ** 2) < 1e-6  # the patch slid 10 mm: (10 mm / 3 mm)^2
    pivot = (Rot.from_euler("y", 25, degrees=True).apply(sole - c) + c)  # turning about the patch
    assert slip(pivot) < 1e-9
    roll = sole.copy()
    roll[:6, 1] += 0.02  # the old patch lifts, the foot rolls onto other points: no sideways motion
    roll[30:36, 1] = -0.001
    assert slip(roll) < 1e-9
