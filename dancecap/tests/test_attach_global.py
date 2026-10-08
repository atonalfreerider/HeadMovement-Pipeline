"""Unit tests: one attach for the whole dance of an attached camera (cameras/attach_global.py). Synthetic geometry.

    .venv-nopo4d\\Scripts\\python.exe -m pytest tests/test_attach_global.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dancecap.cameras import attach_global as ag  # noqa: E402
from dancecap.cameras.attach import align_vectors  # noqa: E402

W, H, F = 480, 640, 600.0
U_MAIN = np.array([0.0, -1.0, 0.0])
U_SUB = np.array([0.05, -0.99, 0.04])
U_SUB /= np.linalg.norm(U_SUB)


def _look(z):
    z = z / np.linalg.norm(z)
    x = np.cross(z, [0, 1.0, 0])
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z])


def _take(seg, t0, n, yaw_true, centre_fn, seed, noise=2.0, step=1):
    """A take of an attached camera: sub-model rotations R_sub = R_main Rs_true, dancers 4-6 m away that wander."""
    rng = np.random.default_rng(seed)
    Rs = ag.rs_of(yaw_true, U_SUB, U_MAIN)
    t = t0 + np.arange(n) / 30.0
    R_sub, fid, X, uv = [], [], [], []
    K = np.zeros((n, 3, 3))
    K[:, 0, 0] = K[:, 1, 1] = F
    K[:, 0, 2], K[:, 1, 2], K[:, 2, 2] = W / 2 - 0.5, H / 2 - 0.5, 1.0
    for i in range(n):
        C = centre_fn(t[i])
        tgt = np.array([2.0 * np.sin(t[i] / 7.0), 1.0, 5.0 + 1.5 * np.cos(t[i] / 11.0)])
        R_m = _look(tgt - C + 0.4 * np.array([np.sin(t[i] / 3.0), 0, 0]))     # the operator pans
        R_sub.append(R_m @ Rs)
        if i % step == 0:
            P = tgt + rng.normal(0, [0.35, 0.5, 0.35], (14, 3))
            Xc = (P - C) @ R_m.T
            q = Xc[:, :2] / Xc[:, 2:] * F + [W / 2 - 0.5, H / 2 - 0.5] + rng.normal(0, noise, (14, 2))
            fid += [i] * 14
            X.append(P), uv.append(q)
    return {"seg": seg, "t": t, "R": np.array(R_sub), "K": K, "k1": np.zeros(n), "fidx": np.array(fid, int),
            "X": np.concatenate(X), "uv": np.concatenate(uv), "u_sub": U_SUB, "u_sub_n": 500, "u_main": U_MAIN}


def _yaw_err_deg(sol, yaw_true):
    d = ag.yaw_of(np.asarray(sol["Rs"]), U_SUB, U_MAIN) - yaw_true
    return abs(np.degrees((d + np.pi) % (2 * np.pi) - np.pi))


def test_rs_of_yaw_of_roundtrip_and_vertical_constraint():
    for yaw in (-2.5, -0.3, 0.0, 1.1, 3.0):
        Rs = ag.rs_of(yaw, U_SUB, U_MAIN)
        assert np.allclose(Rs @ U_SUB, U_MAIN, atol=1e-9)
        assert abs(((ag.yaw_of(Rs, U_SUB, U_MAIN) - yaw + np.pi) % (2 * np.pi)) - np.pi) < 1e-9
    assert np.allclose(align_vectors(U_SUB, U_MAIN) @ U_SUB, U_MAIN, atol=1e-9)


def test_path_interpolation():
    knots = np.array([0.0, 10.0, 20.0])
    C = np.array([[0.0, 0, 0], [1.0, 0, 0], [1.0, 2.0, 0]])
    p = ag.path_at(np.array([-5.0, 0.0, 5.0, 10.0, 15.0, 25.0]), knots, C)
    assert np.allclose(p[:, 0], [0, 0, 0.5, 1, 1, 1]) and np.allclose(p[:, 1], [0, 0, 0, 0, 1, 2])


def test_solve_recovers_one_yaw_and_a_moving_centre_over_two_takes():
    yaw_true = np.radians(-70.0)
    path = lambda t: np.array([0.0 + 0.6 * np.sin(t / 25.0), 1.4, -1.0 + 0.012 * t])     # the operator walks ~0.7 m
    takes = [_take("a", 0.0, 900, yaw_true, path, 1), _take("b", 30.0, 900, yaw_true, path, 2)]
    sol = ag.solve(takes, knot_s=8.0, lams=(0.0, 30.0), log=lambda *a, **k: None)
    assert sol["kind"] == "piecewise", sol["constant"]
    assert sol["chosen"]["holdout_median_px"] < 0.7 * sol["constant"]["holdout_median_px"]
    assert _yaw_err_deg(sol, yaw_true) < 1.5
    assert sol["per_segment"]["a"]["fit_px_median"] < 4.0 and sol["per_segment"]["b"]["fit_px_median"] < 4.0
    # the recovered path follows the truth to a few cm
    t = np.linspace(2, 55, 20)
    err = np.linalg.norm(ag.path_at(t, np.array(sol["knots"]), np.array(sol["C"])) - np.array([path(x) for x in t]), axis=1)
    assert np.median(err) < 0.15, np.median(err)


def test_solve_keeps_a_constant_centre_for_a_still_operator():
    yaw_true = np.radians(40.0)
    still = lambda t: np.array([0.3, 1.5, -0.5])
    takes = [_take("a", 0.0, 900, yaw_true, still, 3), _take("b", 30.0, 900, yaw_true, still, 4)]
    sol = ag.solve(takes, knot_s=8.0, lams=(0.0, 30.0), log=lambda *a, **k: None)
    assert sol["kind"] == "constant"
    assert _yaw_err_deg(sol, yaw_true) < 1.5
    assert sol["per_segment"]["a"]["fit_px_median"] < 3.5


def test_dancer_observations_use_sync_corrected_times(tmp_path, monkeypatch):
    """A 200 ms offset between the cameras moves the instants the dancers are compared at (the unsynced check reads it as
    a camera error): with the delta the triangulated points come from the matching frames."""
    import json
    work = tmp_path / "w"
    (work / "sync").mkdir(parents=True)
    (work / "sync" / "offsets.json").write_text(json.dumps({"offsets": {"01": {"delta": 0.25, "use": True},
                                                                         "02": {"delta": 0.0, "use": False}, "03": {"delta": 0.1}}}))
    assert ag.sync_deltas(work) == {"01": 0.25, "03": 0.1}
    assert ag.sync_deltas(tmp_path / "none") == {}


def test_rotate_poses_keeps_the_images_points_and_moves_the_centre():
    from types import SimpleNamespace

    from dancecap.cameras.localize import FramePose
    rng = np.random.default_rng(5)
    Rs = ag.rs_of(np.radians(33.0), U_SUB, U_MAIN)
    X = 345.0 * rng.normal([0, 0, 1], [0.2, 0.25, 0.0], (300, 3)) / 1.0
    poses, C_sub, uv0 = [], [], []
    for i in range(4):
        R = Rotation.from_rotvec(rng.normal(0, 0.1, 3)).as_matrix()
        c = rng.normal(0, 8.0, 3)                                        # the sub-model's per-frame centre (slid)
        Xc = (X - c) @ R.T
        uv = Xc[:, :2] / Xc[:, 2:] * F + [W / 2 - 0.5, H / 2 - 0.5]
        poses.append(FramePose(ok=True, R=R, t=-R @ c, f=F, n_inliers=300, corr=SimpleNamespace(xyz=X.copy()), q_xy=uv))
        uv0.append(uv)
    C = np.array([[0.1 * i, 1.4, 0.3 * i] for i in range(4)])            # the centre path in the main frame
    ag.rotate_poses(poses, Rs, C)
    for i, p in enumerate(poses):
        assert np.allclose(-p.R.T @ p.t, C[i], atol=1e-9) and p.source == "attach"
        Xc = p.corr.xyz @ p.R.T + p.t
        uv = Xc[:, :2] / Xc[:, 2:] * F + [W / 2 - 0.5, H / 2 - 0.5]
        assert np.abs(uv - uv0[i]).max() < 1e-6                           # the picture's points are unchanged
        assert np.allclose(np.linalg.norm(p.corr.xyz - C[i], axis=1), 10.0)


def test_pick_setting_prefers_the_smoothest_within_the_tolerance():
    paths = [{"lam": 0.0, "knot_s": 4.0, "holdout_median_px": 11.60},
             {"lam": 0.0, "knot_s": 8.0, "holdout_median_px": 11.80},
             {"lam": 3000.0, "knot_s": 4.0, "holdout_median_px": 11.85},
             {"lam": 3000.0, "knot_s": 8.0, "holdout_median_px": 11.90},
             {"lam": 30000.0, "knot_s": 8.0, "holdout_median_px": 13.0}]
    best = ag.pick_setting(paths, tie=0.03)
    assert (best["lam"], best["knot_s"]) == (3000.0, 8.0)                 # 11.90 <= 1.03 x 11.60; 13.0 is not
    assert ag.pick_setting(paths, tie=0.0)["holdout_median_px"] == 11.60
