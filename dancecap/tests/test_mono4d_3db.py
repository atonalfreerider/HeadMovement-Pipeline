r"""Unit tests for the SAM 3D Body prior inside MAMMA_MONO (2026-10-19): mamma-turing utils/mono.py pose_prior_loss /
similarity_to / heading_from_aa, dancecap.mono4d.cand_fitb's prior helpers and candidates.accel_rr_mm (the jitter
metric). Synthetic numbers only; no footage, no pod.

    .venv-nopo4d\Scripts\python.exe -m pytest dancecap/tests/test_mono4d_3db.py -q      (from the workspace root)
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dancecap.mono4d import candidates as C  # noqa: E402
from dancecap.mono4d import cand_fitb as F  # noqa: E402


def _mono_module():
    from dancecap import env
    p = Path(env.mamma_root()) / "optimization" / "utils" / "mono.py"
    if not p.exists():
        pytest.skip("mamma-turing checkout not present")
    pytest.importorskip("torch")
    spec = importlib.util.spec_from_file_location("mamma_mono_3db", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    if not hasattr(m, "pose_prior_loss"):
        pytest.skip("mamma-turing utils/mono.py predates pose_prior_loss")
    return m


def _skeleton(T=6, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(scale=0.3, size=(22, 3))
    base[0] = 0.0
    J = np.repeat(base[None], T, 0) + rng.normal(scale=0.02, size=(T, 22, 3))
    J += np.array([0.0, 0.9, 0.0]) + np.linspace(0, 0.3, T)[:, None, None] * np.array([1.0, 0.0, 0.0])
    return J


def _rot_y(th):
    return np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])


# ------------------------------------------------------------------------------------------------ mamma-turing mono.py


def test_pose_prior_loss_ignores_position_and_size_but_not_facing():
    import torch
    m = _mono_module()
    J = _skeleton()
    X = torch.tensor(J, dtype=torch.float64)
    conf = torch.ones(len(J), dtype=torch.float64)
    root = J[:, :1]
    # the prior placed elsewhere and 10 % larger: no loss (root-relative, per-frame scale)
    P = torch.tensor(1.1 * (J - root) + root + [2.0, 0.0, -1.0], dtype=torch.float64)
    assert float(m.pose_prior_loss(X, P, conf, 1.0)) < 1e-10
    # turned by 40 degrees about the vertical: the facing IS what the prior says (rotate=False) ...
    Pr = torch.tensor(np.einsum("ij,tnj->tni", _rot_y(np.radians(40)), J - root) + root, dtype=torch.float64)
    assert float(m.pose_prior_loss(X, Pr, conf, 1.0)) > 0.05
    # ... unless the rotation is aligned away (rotate=True: articulation only)
    assert float(m.pose_prior_loss(X, Pr, conf, 1.0, rotate=True)) < 1e-10


def test_pose_prior_loss_confidence_nan_and_weight():
    import torch
    m = _mono_module()
    J = _skeleton(T=4)
    X = torch.tensor(J, dtype=torch.float64, requires_grad=True)
    P = J.copy()
    P[:, 5] += [0.0, 0.0, 0.08]                          # one joint 8 cm off on every frame
    rho = 0.08 ** 2 / (0.08 ** 2 + 0.1 ** 2)
    l1 = m.pose_prior_loss(X, torch.tensor(P), torch.ones(4, dtype=torch.float64), 1.0)
    # mean over the 21 non-pelvis joints: one joint carries rho (the per-frame scale moves the others a hair)
    assert float(l1) == pytest.approx(rho / 21, rel=0.05)
    assert float(m.pose_prior_loss(X, torch.tensor(P), torch.ones(4, dtype=torch.float64), 3.0)) == pytest.approx(3 * float(l1))
    # no confidence: no loss and no gradient
    l0 = m.pose_prior_loss(X, torch.tensor(P), torch.zeros(4, dtype=torch.float64), 1.0)
    l0.backward()
    assert float(l0) == 0.0 and float(X.grad.abs().sum()) == 0.0
    # a frame without a prior (NaN) is skipped, the others still count
    Pn = P.copy()
    Pn[1] = np.nan
    ln = m.pose_prior_loss(torch.tensor(J), torch.tensor(Pn), torch.ones(4, dtype=torch.float64), 1.0)
    assert float(ln) == pytest.approx(float(l1), rel=0.05)


def test_pose_prior_gradient_is_robust_to_a_gross_outlier():
    import torch
    m = _mono_module()
    J = _skeleton(T=2)
    P = J.copy()
    P[:, 7] += [0.0, 0.0, 0.05]                          # 5 cm: a real disagreement, pulled
    P[:, 11] += [0.0, 0.0, 1.0]                          # 1 m: a broken prior frame, hardly pulled (Geman-McClure)
    X = torch.tensor(J, dtype=torch.float64, requires_grad=True)
    m.pose_prior_loss(X, torch.tensor(P), torch.ones(2, dtype=torch.float64), 1.0, scale=False).backward()
    g = X.grad.numpy()
    assert g[:, 7, 2].max() < 0                          # descent moves joint 7 towards the prior (+z)
    assert np.abs(g[:, 11]).max() < 0.05 * np.abs(g[:, 7]).max()


def test_similarity_to_recovers_scale_and_rotation():
    import torch
    m = _mono_module()
    J = _skeleton(T=3)
    P = J - J[:, :1]
    R = _rot_y(0.7)
    X = 0.9 * np.einsum("ij,tnj->tni", R, P)
    s, Rh = m.similarity_to(torch.tensor(X), torch.tensor(P), torch.ones(3, 22, dtype=torch.float64), rotate=True)
    assert np.allclose(s.numpy(), 0.9, atol=1e-6) and np.allclose(Rh.numpy(), R[None], atol=1e-6)
    s2, _ = m.similarity_to(torch.tensor(5 * P), torch.tensor(P), torch.ones(3, 22, dtype=torch.float64))
    assert np.allclose(s2.numpy(), 1.25)                 # clamped to s_range


def test_heading_from_axis_angle_matches_the_yaw_search_convention():
    import torch
    m = _mono_module()
    from scipy.spatial.transform import Rotation
    th = torch.tensor([0.0, 0.5, -2.0, 3.0], dtype=torch.float64)
    assert np.allclose(m.heading_from_aa(m._axis_angle_y(th)).numpy(), th.numpy())
    aa = np.random.default_rng(2).normal(size=(5, 3))
    assert np.allclose(m.aa_to_mat(torch.tensor(aa)).numpy(), Rotation.from_rotvec(aa).as_matrix(), atol=1e-10)


# ------------------------------------------------------------------------------------------------ cand_fitb prior helpers


def test_mask_conf_and_grid_index():
    c = F.mask_conf(np.array([100, 100, 100, 95, 70, 40, 10.0]))
    assert c[0] == 1.0 and c[3] == 1.0 and 0.25 < c[4] < 1.0 and c[5] == 0.25 and c[6] == 0.25
    k, ok = F.grid_index(np.array([0.0, 1 / 30, 2 / 30]), np.array([0.0005, 1 / 30 + 0.001, 0.1]))
    assert k.tolist() == [0, 1, 2] and ok.tolist() == [True, True, False]


def test_smooth_series_keeps_a_constant_and_skips_invalid_frames():
    X = np.ones((10, 2, 3))
    X[4] = 50.0                                           # a broken sample, marked invalid
    valid = np.ones(10, bool)
    valid[4] = False
    assert np.allclose(F.smooth_series(X, 1.5, valid), 1.0)
    noisy = np.sin(np.arange(40) / 5.0)[:, None] + 0.1 * (-1) ** np.arange(40)[:, None]
    sm = F.smooth_series(noisy, 1.5, np.ones(40, bool))
    acc = lambda y: np.abs(np.diff(y, 2, axis=0)).mean()  # noqa: E731
    assert acc(sm) < 0.2 * acc(noisy)
    assert np.allclose(F.smooth_series(X, 0, valid), X)


def test_smooth_rotvec_survives_the_quaternion_sign_and_the_wrap():
    from scipy.spatial.transform import Rotation
    T = 9
    aa = np.tile([[0.0, np.pi - 0.02, 0.0]], (T, 1))
    aa[::2] = [0.0, -(np.pi - 0.02), 0.0]                # the same heading +-178 deg: a 4-degree wobble about the wrap
    out = F.smooth_rotvec(aa[:, None], 1.5, np.ones(T, bool))[:, 0]
    ang = np.degrees(np.linalg.norm((Rotation.from_rotvec(out) * Rotation.from_rotvec([0, np.pi, 0]).inv()).as_rotvec(), axis=-1))
    assert ang.max() < 2.1                                # stays at the half turn (an average of rotvecs would give ~0)
    base = Rotation.from_rotvec(np.tile([[0.3, 0.2, -0.1]], (T, 1)))
    assert np.allclose(F.smooth_rotvec(base.as_rotvec()[:, None], 2.0, np.ones(T, bool))[:, 0], base.as_rotvec(), atol=1e-9)


def test_orient_to_world_agrees_with_moving_points_to_the_world():
    from scipy.spatial.transform import Rotation
    rng = np.random.default_rng(4)
    F_ = 4
    R = Rotation.from_rotvec(rng.normal(size=(F_, 3))).as_matrix()
    t = rng.normal(size=(F_, 3))
    go_cam = rng.normal(size=(F_, 3))
    v = rng.normal(size=(7, 3))                           # body-frame offsets
    cam_pts = np.einsum("fij,nj->fni", Rotation.from_rotvec(go_cam).as_matrix(), v) + [0, 0, 3]
    world_pts = C.to_world(cam_pts, R, t)
    gw = F.orient_to_world(go_cam, R)
    rel = np.einsum("fij,nj->fni", Rotation.from_rotvec(gw).as_matrix(), v)
    assert np.allclose(world_pts - world_pts[:, :1], rel - rel[:, :1], atol=1e-9)


# ------------------------------------------------------------------------------------------------ jitter metric


def test_accel_rr_mm():
    T = 10
    t = np.arange(T) / 30
    J = np.zeros((T, 22, 3))
    J[:, :, 0] = np.linspace(0, 1, T)[:, None]            # the whole body glides: root-relative still, no jitter
    J[:, 5, 1] = 0.01 * np.arange(T)                      # one joint at constant velocity relative to the pelvis: none
    assert np.nanmax(C.accel_rr_mm(t, J)) < 1e-9
    J[4, 5, 1] += 0.021                                   # a 21 mm spike on one joint: |a| = 21, 42, 21 mm/f^2 / 21 joints
    a = C.accel_rr_mm(t, J)
    assert np.isnan(a[0]) and np.isnan(a[-1])
    assert a[4] == pytest.approx(2.0) and a[3] == pytest.approx(1.0) and a[5] == pytest.approx(1.0)
    t2 = t.copy()
    t2[6:] += 0.2                                         # a gap: no acceleration across it
    assert np.isnan(C.accel_rr_mm(t2, J)[5:7]).all()
    s = C.score_arrays(10 + t, J, "world", 10 + t, J, (10, 11))
    assert s["accel_err"]["median"] == pytest.approx(0.0) and s["accel_rr"]["n"] == T - 2
    assert s["accel_rr"]["median"] == pytest.approx(s["accel_rr_ref"]["median"])


# ------------------------------------------------------------------------------------------------ prior_interp (john_lula)


def test_prior_interp_resamples_29_994_fps_frames_onto_the_30_hz_lattice():
    from scipy.spatial.transform import Rotation
    rng = np.random.default_rng(5)
    Fs = 90
    ts = np.arange(Fs) * 500 / 14997                       # john_lula's frame clock (29.994 fps)
    tg = np.arange(88) / 30.0                              # the fit lattice k / 30 (drifts up to 16.7 ms off the frames)
    base = _skeleton(T=1, seed=6)[0] - [0.0, 0.9, 0.0]
    Jw = base[None] + 0.05 * np.sin(ts)[:, None, None] * rng.normal(size=(1, 22, 3)) + [0.0, 0.9, 3.0]
    Jw[:, :, 0] += 0.5 * ts[:, None]                       # the body moves (pelvis kept, not smoothed)
    R = np.repeat(Rotation.from_rotvec([0.0, 0.3, 0.0]).as_matrix()[None], Fs, 0)
    tv = np.repeat(np.array([[0.1, -0.2, 0.5]]), Fs, 0)
    Jc = np.einsum("fij,fnj->fni", R, Jw) + tv[:, None]    # x_cam = R X + t
    gcam = np.tile([0.1, 0.2, 0.3], (Fs, 1))
    body = np.tile(np.linspace(-0.2, 0.2, 63), (Fs, 1))
    valid = np.ones(Fs, bool)
    valid[40] = False                                      # one frame without a prior
    g = F.prior_interp(Jc, gcam, body, R, tv, ts, valid, np.full(Fs, 0.8), tg, smooth=0.0)
    lin = np.stack([np.stack([np.interp(tg, ts, Jw[:, j, c]) for c in range(3)], -1) for j in range(22)], 1)
    ok = g["ok"]
    near40 = np.abs(tg - ts[40]) < 1 / 29.994             # grid instants bracketed by frame 40
    assert not ok[near40 & (np.minimum(np.abs(tg - ts[39]), np.abs(tg - ts[41])) > F.PRIOR_DT_MAX)].any()
    assert ok[~near40].all() and ok.sum() >= len(tg) - 2
    assert np.allclose(g["joints"][ok & ~near40], lin[ok & ~near40], atol=1e-9)
    assert np.allclose(g["conf"][ok], 0.8) and (g["conf"][~ok] == 0).all() and np.isnan(g["joints"][~ok]).all()
    assert g["dt_nearest_ms"].max() <= 1e3 / 29.994 / 2 + 1e-6
    gw = Rotation.from_matrix(R[0].T @ Rotation.from_rotvec(gcam[0]).as_matrix()).as_rotvec()
    assert np.allclose(g["global_orient"][ok], gw, atol=1e-9)        # constant orientation survives nlerp
    assert np.allclose(g["body_pose"][ok], body[0], atol=1e-9)
    # every grid frame carries a starting pose (mono.py prior_init copies them all): the gaps are filled
    assert (~ok).any() and g["init_filled"].sum() == (~ok).sum()
    assert np.allclose(g["global_orient"], gw, atol=1e-9) and np.allclose(g["body_pose"], body[0], atol=1e-9)
    # smoothing on the source frames leaves a constant pose alone and keeps NaN-free output where ok
    g2 = F.prior_interp(Jc, gcam, body, R, tv, ts, valid, np.full(Fs, 0.8), tg, smooth=1.5)
    assert np.isfinite(g2["joints"][g2["ok"]]).all() and np.allclose(g2["body_pose"][g2["ok"]], body[0], atol=1e-9)
