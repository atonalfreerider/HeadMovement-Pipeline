"""Unit tests for dancecap.recap_attach (synthetic phone and teachers; no footage, no pod).

    .venv-nopo4d\\Scripts\\python.exe -m pytest dancecap/tests/test_recap_attach.py -q      (from the workspace root)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Ro

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dancecap import recap_attach as ra  # noqa: E402
from dancecap import recap_people as rp  # noqa: E402
from test_recap_people import look_at, skeleton  # noqa: E402


def synthetic(q_deg=(1.2, -0.8, 0.5), shift=(0.15, -0.05, 0.1), noise_px=1.0, F=60, seed=0):
    """A phone whose stored pose (R2's attach) is off by the constant rotation Q and centre shift, watching two teachers that walk about.
    Returns (cam with the WRONG stored pose, observations built from the TRUE pose, true params)."""
    rng = np.random.default_rng(seed)
    C_true = np.array([1.0, 1.2, -5.0])
    R0, _ = look_at(C_true, (0.3, 1.0, 0.0))
    Q = Ro.from_rotvec(np.radians(q_deg)).as_matrix()
    # the stored (wrong) pose: R_stored(t) such that R_stored Q = R_true(t)  ->  R_true = R_stored Q  (the model of ra.reproject)
    sway = [Ro.from_euler("xyz", [0.5 * np.sin(i / 7.0), 0.8 * np.cos(i / 9.0), 0.0], degrees=True).as_matrix() for i in range(F)]
    R_true = np.stack([R0 @ s for s in sway])
    R_stored = R_true @ Q.T
    C_stored = C_true + np.array(shift)
    K = np.array([[1100.0, 0, 539.5], [0, 1100.0, 959.5], [0, 0, 1.0]])
    t = np.arange(F) / 30.0
    cam = rp.Cam("06", t, np.ones(F, bool), np.arange(F), np.repeat(K[None], F, 0), R_stored, -np.einsum("fij,j->fi", R_stored, C_stored),
                 np.repeat(C_stored[None], F, 0), np.zeros(F), (1080, 1920), True, t, None, None)
    X = np.zeros((F, 2, 17, 3))
    uv = np.zeros((F, 2, 17, 2))
    for i in range(F):
        for s in range(2):
            X[i, s] = skeleton(-0.4 + 0.9 * s + 0.01 * i, 0.1 * np.sin(i / 10.0))
            Xc = (X[i, s] - C_true) @ R_true[i].T
            uv[i, s] = np.c_[K[0, 0] * Xc[:, 0] / Xc[:, 2] + K[0, 2], K[1, 1] * Xc[:, 1] / Xc[:, 2] + K[1, 2]] + rng.normal(0, noise_px, (17, 2))
    ob = {"i": np.repeat(np.arange(F), 2), "slot": np.tile([0, 1], F), "X": X.reshape(-1, 17, 3), "uv": uv.reshape(-1, 17, 2), "valid": np.ones((2 * F, 17), bool),
          "size": np.full(2 * F, 330.0), "t": np.repeat(t, 2)}
    return cam, ob, np.r_[np.radians(q_deg), C_true]


def test_reprojection_at_the_true_parameters_is_the_noise_level_and_a_wrong_pose_is_worse():
    cam, ob, truth = synthetic()
    x0 = np.r_[np.zeros(3), cam.C[0]]
    sel = np.ones(len(ob["i"]), bool)
    good = ra.stats(truth, cam, ob, sel)
    bad = ra.stats(x0, cam, ob, sel)
    assert good["median_joint_px"] < 1.8 and bad["median_joint_px"] > 3 * good["median_joint_px"]


def test_the_fit_recovers_the_rotation_and_the_centre_of_the_attach():
    cam, ob, truth = synthetic(noise_px=1.5)
    x0 = np.r_[np.zeros(3), cam.C[0]]
    sel = np.ones(len(ob["i"]), bool)
    x = ra.fit(cam, ob, sel, x0)
    assert np.degrees(np.linalg.norm(x[:3] - truth[:3])) < 0.25
    assert np.linalg.norm(x[3:] - truth[3:]) < 0.06
    assert ra.stats(x, cam, ob, sel)["median_joint_px"] < 2.2


def test_wrong_associations_and_joints_behind_the_camera_do_not_ruin_the_fit():
    cam, ob, truth = synthetic(noise_px=1.0)
    rng = np.random.default_rng(3)
    bad = rng.choice(len(ob["i"]), 18, replace=False)
    ob["uv"][bad] += rng.uniform(150, 400, (18, 1, 2))                  # 15 % of the observations are the wrong person
    ob["X"][5, 3] = np.array([0.0, 1.0, -30.0])                         # a triangulation outlier: a joint far behind the phone
    x0 = np.r_[np.zeros(3), cam.C[0]]
    sel = np.ones(len(ob["i"]), bool)
    x = ra.fit(cam, ob, sel, x0)
    assert np.degrees(np.linalg.norm(x[:3] - truth[:3])) < 0.4 and np.linalg.norm(x[3:] - truth[3:]) < 0.1
    r = ra.residuals(x0, cam, ob, sel)
    assert np.isfinite(r).all() and np.abs(r).max() <= 40.0


def test_observations_use_only_projected_associations_and_enough_joints():
    cam, ob, _ = synthetic(F=12)
    P = np.full((12, 2, 17, 3), np.nan, np.float32)
    P[..., :2] = ob["uv"].reshape(12, 2, 17, 2)
    P[..., 2] = 0.9
    cam.kp, cam.n = P, np.full(12, 2, np.int16)
    X = np.full((12, 2, 17, 3), np.nan, np.float32)
    X[:] = ob["X"].reshape(12, 2, 17, 3)
    tr = rp.Track3D(np.arange(12), np.zeros((12, 2), np.int32), X, np.zeros((12, 2), np.int8), np.zeros((12, 2), np.float32), (0, 0))
    A = rp.Assoc("06", np.tile([0, 1], (12, 1)).astype(np.int16), np.full((12, 2), 0.02, np.float32), np.full((12, 2), 2, np.int8), np.full((12, 2), 17, np.int8))
    o = ra.observations(cam, A, tr, step=3)
    assert len(o["i"]) == 8 and set(o["slot"]) == {0, 1}
    A.kind[:] = 1                                                        # members of the 3D candidate (a phone that took part in the triangulation): not independent
    try:
        ra.observations(cam, A, tr, step=3)
        raise AssertionError("must refuse")
    except RuntimeError:
        pass
