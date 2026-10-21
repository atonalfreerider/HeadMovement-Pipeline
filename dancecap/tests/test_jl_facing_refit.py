r"""Unit tests for the facing correction of john_lula (2026-10-20): mamma-turing utils/mono.py facing_hint_loss / torso_forward
(MAMMA_FACING_HINT) and the pure helpers of dancecap.mono4d.jl_facing_refit (prior turns, runs, chunk choice). Synthetic numbers
only; no footage, no pod.

    .venv-nopo4d\Scripts\python.exe -m pytest dancecap/tests/test_jl_facing_refit.py -q      (from the workspace root)
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dancecap.mono4d import jl_facing as JF  # noqa: E402
from dancecap.mono4d import jl_facing_refit as R  # noqa: E402


def _mono_module():
    from dancecap import env
    p = Path(env.mamma_root()) / "optimization" / "utils" / "mono.py"
    if not p.exists():
        pytest.skip("mamma-turing checkout not present")
    pytest.importorskip("torch")
    spec = importlib.util.spec_from_file_location("mamma_mono_face", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    if not hasattr(m, "facing_hint_loss"):
        pytest.skip("mamma-turing utils/mono.py predates facing_hint_loss")
    return m


def _body(yaw_deg: float, n: int = 1) -> np.ndarray:
    """A minimal upright SMPL-X-ordered skeleton (22 joints; hips 1/2, shoulders 16/17) facing heading yaw_deg (atan2(x, z))."""
    J = np.zeros((22, 3))
    J[:, 1] = 1.0
    J[1] = [0.1, 0.9, 0.0]     # left hip (+X = the body's left when facing +Z)
    J[2] = [-0.1, 0.9, 0.0]
    J[16] = [0.18, 1.4, 0.0]   # left shoulder
    J[17] = [-0.18, 1.4, 0.0]
    J[0] = [0.0, 0.95, 0.0]
    J[15] = [0.0, 1.6, 0.05]
    Ry = R.rot_y(np.radians([yaw_deg]))[0]
    P = J[0].copy()
    J = (J - P) @ Ry.T + P + np.array([0.5, 0.0, -0.3])
    return np.repeat(J[None], n, 0)


# ------------------------------------------------------------------------------------------------ mamma-turing mono.py
def test_torso_forward_matches_jl_facing():
    import torch
    m = _mono_module()
    J = np.concatenate([_body(a) for a in (0.0, 37.0, -120.0, 175.0)])
    f_t = m.torso_forward(torch.tensor(J)).numpy()
    f_n = JF.torso_axes(J)[0]
    assert np.allclose(f_t, f_n, atol=1e-9)
    assert np.allclose(R.yaw_deg(f_n), [0.0, 37.0, -120.0, 175.0], atol=1e-6)


def test_facing_hint_zero_inside_band_and_across_the_wrap():
    import torch
    m = _mono_module()
    J = torch.tensor(np.concatenate([_body(-170.0), _body(10.0)]))
    tgt = torch.tensor(np.radians([170.0, -40.0]))
    tol = torch.tensor(np.radians([30.0, 60.0]))
    conf = torch.ones(2, dtype=torch.float64)
    assert float(m.facing_hint_loss(J, tgt, tol, conf, 1.0)) == 0.0          # 20 deg across +-180 inside 30; 50 inside 60


def test_facing_hint_penalises_excess_only_and_respects_conf_and_nan():
    import torch
    m = _mono_module()
    J = torch.tensor(np.concatenate([_body(100.0), _body(100.0), _body(100.0)]))
    tgt = torch.tensor([0.0, float("nan"), 0.0], dtype=torch.float64)
    tol = torch.tensor(np.radians([60.0, 60.0, 60.0]))
    conf = torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64)
    e = np.radians(40.0)
    want = np.log1p((e / m.FACING_SIGMA_RAD) ** 2) / 3.0                       # only frame 0 counts; mean over T = 3
    assert float(m.facing_hint_loss(J, tgt, tol, conf, 1.0)) == pytest.approx(want, rel=1e-6)
    assert float(m.facing_hint_loss(J, tgt, tol, torch.zeros(3, dtype=torch.float64), 1.0)) == 0.0


def test_facing_hint_ignores_a_lying_torso():
    import torch
    m = _mono_module()
    J = _body(0.0)
    J[0, [16, 17], 1] = 0.9                       # shoulders level with the hips, torso lying along +Z: up = +Z, forward vertical
    J[0, 16, 2] = J[0, 17, 2] = 0.2
    loss = m.facing_hint_loss(torch.tensor(J), torch.tensor([np.pi]), torch.tensor([0.1]), torch.ones(1, dtype=torch.float64), 1.0)
    assert float(loss) == 0.0


def test_facing_hint_gradient_turns_the_body_into_the_band():
    import torch
    m = _mono_module()
    base = torch.tensor(_body(0.0)[0])
    yaw = torch.tensor([np.radians(-150.0)], requires_grad=True)
    opt = torch.optim.Adam([yaw], lr=0.05)
    tgt, tol, conf = torch.tensor([0.0], dtype=torch.float64), torch.tensor([np.radians(60.0)]), torch.ones(1, dtype=torch.float64)
    for _ in range(400):
        c, s = torch.cos(yaw[0]), torch.sin(yaw[0])
        Ry = torch.stack([torch.stack([c, torch.zeros_like(c), s]), torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64),
                          torch.stack([-s, torch.zeros_like(c), c])])
        P = base[0]
        J = ((base - P) @ Ry.T + P)[None]
        loss = m.facing_hint_loss(J, tgt, tol, conf, 1.0)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert abs(np.degrees(float(yaw))) <= 61.0                                # pulled across 90 deg of excess into the band


# ------------------------------------------------------------------------------------------------ jl_facing_refit helpers
def test_rotate_about_vertical_keeps_the_root_relative_pose():
    from scipy.spatial.transform import Rotation as Rot
    J = np.concatenate([_body(20.0), _body(-75.0)])
    go = np.array([[0.0, np.radians(20.0), 0.0], [0.1, np.radians(-75.0), 0.05]])
    d = np.radians([30.0, -180.0])
    Jn, gon = R.rotate_about_vertical(J, go, d)
    assert np.allclose(Jn[:, 0], J[:, 0])                                      # pelvis stays
    dist = lambda X: np.linalg.norm(X[:, :, None] - X[:, None], axis=-1)        # noqa: E731
    assert np.allclose(dist(Jn), dist(J))                                      # rigid: the root-relative pose is unchanged
    assert np.allclose(R.wrap_deg(R.yaw_deg(R.torso_forward_np(Jn)) - R.yaw_deg(R.torso_forward_np(J))), [30.0, -180.0], atol=1e-6)
    # global_orient composed with the same turn: R(go') = R_y(d) R(go)
    assert np.allclose(Rot.from_rotvec(gon).as_matrix(), R.rot_y(d) @ Rot.from_rotvec(go).as_matrix(), atol=1e-9)


def test_prior_action_cases():
    cam = 10.0
    assert R.prior_action(0, cam, 30.0) == ("keep", 0.0)                       # front, inside the band
    a, d = R.prior_action(0, cam, cam + 170.0)                                  # back where the picture says front
    assert a == "flip180" and d == 180.0
    a, d = R.prior_action(0, cam, cam - 100.0)                                  # side, 40 deg outside the front band
    assert a == "rotate" and d == pytest.approx(40.0 + R.PRIOR_MARGIN_DEG)
    assert R.prior_action(0, cam, cam - 75.0)[0] == "keep"                      # 15 deg outside: a boundary case
    a, d = R.prior_action(0, cam, cam - 75.0, from_owner=True)                  # ... unless the owner says so
    assert a == "rotate" and d == pytest.approx(15.0 + R.PRIOR_MARGIN_DEG)
    a, d = R.prior_action(2, cam, cam + 80.0)                                   # back wanted, side seen: turn away from the phone
    assert a == "rotate" and d == pytest.approx(40.0 + R.PRIOR_MARGIN_DEG)     # the smaller turn: rel -80 -> -125
    y = cam + 80.0 + d
    assert abs(R.wrap_deg(cam - y)) >= R.BACK_MIN + R.PRIOR_MARGIN_DEG - 1e-6


def test_smooth_rotation_runs_ramps_and_flips():
    T = 40
    delta = np.zeros(T)
    kind = np.array(["keep"] * T, dtype=object)
    delta[10:16] = [30, 31, 29, 30, 32, 30]
    kind[10:16] = "rotate"
    kind[30] = "flip180"
    delta[30] = 180.0
    out = R.smooth_rotation_runs(delta, kind, ramp=5)
    assert np.all(np.abs(out[10:16] - 30.3) < 2.0)
    assert out[30] == 180.0 and out[29] == 0.0 and out[31] == 0.0
    assert 0 < out[9] < out[10] and 0 < out[16] < out[15] and out[4] == 0.0 and out[22] == 0.0
    assert np.all(np.diff(out[4:11]) >= 0)                                      # monotone ramp in
    assert np.count_nonzero(out) == 6 + 10 + 1


def test_windows_to_refit_skips_false_alarms_and_uses_the_owner():
    owner = np.array(["a"] * 50 + ["b"] * 50)
    windows = [{"id": "w0", "role": "lead", "start": 10, "end": 12}, {"id": "w1", "role": "lead", "start": 48, "end": 52},
               {"id": "w2", "role": "follow", "start": 80, "end": 85}]
    verdicts = [{"role": "follow", "start": 80, "end": 85, "verdict": "false_alarm"}, {"role": "lead", "start": 10, "end": 12,
                                                                                         "verdict": "confirmed"}]
    out = R.windows_to_refit(windows, verdicts, owner, lambda f: f)
    assert out == {"a": ["w0", "w1"], "b": ["w1"]}


def test_heading_path_meets_the_fit_at_both_ends_and_follows_the_prior():
    T = 20
    fit = np.linspace(40.0, 140.0, T)                      # the export's slerp: a slow, even turn
    prior = np.concatenate([np.linspace(30.0, -50.0, 10), np.linspace(-50.0, 170.0, 10)])
    prior = R.wrap_deg(prior + 360.0 * 3)                   # wrapped like any yaw
    a, b = 3, 16
    new = R.heading_path(fit, prior, a, b)
    assert new.shape == (b - a + 1,)
    full = np.concatenate([[fit[a - 1]], new, [fit[b + 1]]])
    pu = np.degrees(np.unwrap(np.radians(prior[a - 1:b + 2])))
    off = full - pu
    assert abs(R.wrap_deg(off[0] - (fit[a - 1] - pu[0]))) < 1e-9 and abs(R.wrap_deg(full[-1] - fit[b + 1])) < 1e-9
    assert np.allclose(np.diff(off), np.diff(off)[0])      # the offset is linear: the prior's shape is kept
    assert np.abs(np.diff(full)).max() < 40.0               # no jump


def test_runs_of():
    assert R.runs_of(np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 1], bool)) == [(1, 2), (4, 4), (7, 9)]
