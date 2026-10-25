"""dancecap.sides.solve, the terms on tiny torch problems (CPU, 60 frames; DESIGN 8): no push-out of a correct-side overlap, E_gli /
E_sep exactly 0 when correct or unlinked, a planted wrong-side stretch goes back through the short path with no pass-through event, a
planted 15-cm elbow swing is removed, a decided R9 sign survives a 2D pull through, a 120-deg knee twist goes under 45 deg without moving
the joints, the trust region holds, the ramps are C1; and the torch FK reproduces a fit's joints."""
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dancecap.sides import geom  # noqa: E402
from dancecap.sides import solve as sv  # noqa: E402

T = 60


def chains(T=T, gap=0.02, sign=1.0):
    """his arm A (shoulder, elbow, wrist) crossing her arm B: A runs along x at height y = sign * gap / 2, B along z at -sign * gap / 2,
    the forearms crossing near the origin (the bone axes pass `gap` apart, closest points interior) -> (A, B) (T, 3, 3) float64"""
    A = np.zeros((T, 3, 3))
    B = np.zeros((T, 3, 3))
    for k in range(T):          # the forearms cross in their interiors (not at the elbows: F2's bone-end case)
        A[k] = [[-0.4, sign * gap / 2, 0.05], [-0.12, sign * gap / 2, 0.02], [0.2, sign * gap / 2, -0.03]]
        B[k] = [[0.05, -sign * gap / 2, -0.4], [0.02, -sign * gap / 2, -0.12], [-0.03, -sign * gap / 2, 0.2]]
    return torch.tensor(A, dtype=torch.float64), torch.tensor(B, dtype=torch.float64)


def sign_of(A, B):
    return float(np.sign(geom.gli_chain(A.numpy(), B.numpy()).mean()))


def test_hold_terms_exactly_zero_when_correct_or_unlinked():
    A, B = chains(gap=0.02)
    s = torch.full((T,), sign_of(A, B), dtype=torch.float64)
    eg, es = sv.hold_terms(torch, A, B, s, d0=0.01)
    assert float(eg) == 0.0
    # correct side but the axes only 2 cm apart: E_sep's hinge is max(0, d0 - s d_s) with d_s = 2 cm > d0 -> 0
    assert float(es) == 0.0
    # unlinked (far apart): G ~ 0, the pairs farther than dist_max
    A2 = A.clone()
    A2[..., 1] += 0.5
    eg2, es2 = sv.hold_terms(torch, A2, B, s, d0=0.01)
    assert float(eg2) == 0.0 and float(es2) == 0.0
    # not held (s = 0): zero whatever the geometry
    eg3, es3 = sv.hold_terms(torch, A, B, torch.zeros(T, dtype=torch.float64), d0=0.01)
    assert float(eg3) == 0.0 and float(es3) == 0.0


def _opt(params, loss_fn, iters=300, lr=0.01):
    opt = torch.optim.Adam(params, lr=lr)
    for _ in range(iters):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()


def test_correct_side_overlap_is_not_pushed_out():
    """a correct-side pair whose capsules overlap (axes 2 cm apart, radii ~4 cm each): no term pushes them apart"""
    A0, B = chains(gap=0.02)
    s = torch.full((T,), sign_of(A0, B), dtype=torch.float64)
    x = torch.zeros_like(A0, requires_grad=True)

    def loss():
        A = A0 + x
        eg, es = sv.hold_terms(torch, A, B, s)
        return 8 * es + 2 * eg + ((x / 0.01) ** 2).sum()
    _opt([x], loss, 100)
    assert float(x.detach().abs().max()) < 1e-3


def test_wrong_side_goes_back_through_the_short_path_without_events():
    """frames 20-39 planted on the wrong side (his bone 1 cm through hers); E_sep moves them back through the short path: every frame
    ends on the decided side, the move is a few cm, and the sequence has no GLI pass-through event"""
    A0, B = chains(gap=0.02, sign=1.0)
    s_dec = sign_of(A0, B)
    A0 = A0.clone()
    A0[20:40, :, 1] -= 0.03                          # through her bone: now 1 cm on the other side
    assert sign_of(A0[20:40], B[20:40]) == -s_dec
    s = torch.full((T,), s_dec, dtype=torch.float64)
    x = torch.zeros_like(A0, requires_grad=True)

    def loss():
        A = A0 + x
        eg, es = sv.hold_terms(torch, A, B, s)
        return 8 * es + 2 * eg + ((x / 0.05) ** 2).sum()
    _opt([x], loss, 600, lr=0.003)
    A = (A0 + x).detach().numpy()
    G = geom.gli_chain(A, B.numpy())
    assert (np.sign(G) == s_dec).all()
    assert len(geom.passthrough_events(G)) == 0
    assert float(np.abs(x.detach().numpy()).max()) < 0.06


def test_r9_sign_survives_a_2d_pull_through():
    """a 2D-like pull drags his bone 3 cm through hers; the decided sign holds (E_sep + E_gli), the pull alone would cross"""
    A0, B = chains(gap=0.02)
    s_dec = sign_of(A0, B)
    s = torch.full((T,), s_dec, dtype=torch.float64)
    target = A0.clone()
    target[..., 1] -= 0.04                           # where the "2D" wants it: 2 cm through her bone
    for with_hold in (False, True):
        x = torch.zeros_like(A0, requires_grad=True)

        def loss():
            A = A0 + x
            e2d = ((A - target) ** 2).sum() / 0.01 ** 2 * 0.05
            if not with_hold:
                return e2d
            eg, es = sv.hold_terms(torch, A, B, s)
            return e2d + 8 * es + 2 * eg
        _opt([x], loss, 500, lr=0.003)
        sg = sign_of((A0 + x).detach(), B)
        if with_hold:
            assert sg == s_dec
        else:
            assert sg == -s_dec


def test_band_term_removes_a_planted_swing():
    """an elbow height with a 15-cm 2-Hz swing and a calm 'triangulated' elbow: E_band (+ a weak prior) removes most of the swing"""
    t = np.arange(T) / 30.0
    y0 = 0.05 * t + 0.15 * np.sin(2 * np.pi * 2.0 * t)
    fir = torch.tensor(sv.fir_band(30.0, 1.0, 5.0), dtype=torch.float32).flip(0)[None, None]
    ref = 0.0
    yt = torch.tensor(y0, dtype=torch.float32)
    rms0 = float(sv.band_rms_t(torch, yt, fir))
    assert rms0 > 0.05
    B = torch.tensor(__import__("dancecap.hand_contact", fromlist=["temporal_basis"]).temporal_basis(T, 5), dtype=torch.float32)
    x = torch.zeros(B.shape[1], requires_grad=True)

    def loss():
        y = yt + B @ x
        rms = sv.band_rms_t(torch, y, fir)
        return 10 * (torch.relu(rms - ref - 0.003) * 100) ** 2 + ((B @ x) ** 2).sum() * 0.1
    _opt([x], loss, 800, lr=0.01)
    rms1 = float(sv.band_rms_t(torch, yt + B @ x.detach(), fir))
    assert rms1 < 0.2 * rms0


def test_knee_twist_goes_under_45_without_moving_the_joints():
    """a knee rotated 120 deg about its own bone (pure twist) with 30 deg of flexion: the hinge (5 + 30 sin flex) brings the twist under
    45 deg; a pure twist about the bone does not move the child joint, so the 2D of the joints is unchanged"""
    ax = np.array([0.0, -1.0, 0.0])
    flex = sv.geom.rotvec_to_mat(np.array([math.radians(30.0), 0.0, 0.0]))
    tw = sv.geom.rotvec_to_mat(ax * math.radians(120.0))
    R0 = torch.tensor(np.repeat((flex @ tw)[None], T, 0), dtype=torch.float32)
    axt = torch.tensor(ax, dtype=torch.float32)
    child = torch.tensor([0.0, -0.4, 0.0])
    v = torch.zeros((T, 1, 3), requires_grad=True)
    rad = torch.tensor([math.radians(120.0)])

    def pose():
        d, _ = sv.trust_delta(torch, v, rad)
        return sv.__dict__["bd"].rodrigues_t(torch, d[:, 0]) if False else None

    from dancecap.sides import body as bd

    def Rl():
        d, _ = sv.trust_delta(torch, v, rad)
        # the twist is corrected in the joint's own frame: R0 exp(d), which keeps the bone (child) direction when d is along ax
        return R0 @ bd.rodrigues_t(torch, d[:, 0])

    def loss():
        R = Rl()
        lim = sv.knee_twist_limit(torch, R)
        e = (sv.twist_excess(torch, R, axt, lim) ** 2).sum() * 0.05
        pos = (R @ child)                                         # the child joint (2D proxy: its position)
        return e + ((pos - (R0 @ child)) ** 2).sum() / 0.005 ** 2
    _opt([v], loss, 600, lr=0.02)
    R = Rl().detach()
    twd, _ = geom.swing_twist(R.numpy().astype(float), ax)
    assert np.abs(twd).max() < 45.0
    moved = (R @ child - R0 @ child).norm(dim=-1).max()
    assert float(moved) < 0.005


def test_trust_region_holds():
    rad = torch.tensor([math.radians(25.0), math.radians(10.0)])
    v = torch.randn(T, 2, 3) * 50.0
    d, th = sv.trust_delta(torch, v, rad)
    ang = d.norm(dim=-1)
    assert (ang <= rad[None] + 1e-6).all()
    assert (th <= 1.0).all() and float(th.max()) > 0.99


def test_ramps_are_c1():
    w = sv.ramp_weights(60, 20, 20)
    assert w[20:40].min() == 1.0
    assert 0 < w[0] < 0.02 and 0 < w[-1] < 0.02
    d = np.diff(np.concatenate([[0.0], w, [0.0]]))
    assert np.abs(d).max() < 0.1                                 # no jump into or out of the window
    assert abs(d[1]) < 0.02 and abs(d[20]) < 0.02                 # zero slope at the ramp ends


def test_torch_fk_equals_the_numpy_fk():
    from dancecap.sides import body as bd
    from dancecap.take import WORKSPACE
    model = WORKSPACE / "mamma" / "data" / "body_models" / "smplx_locked_head" / "smplx" / "SMPLX_NEUTRAL.npz"
    if not model.exists():
        pytest.skip("no SMPL-X model")
    rng = np.random.default_rng(0)
    n = 5
    fit = {"betas": np.zeros(16), "t": np.arange(n) / 30.0, "transl": rng.normal(size=(n, 3)) * 0.1,
           "model": np.array("smplx_locked_head"), "gender": np.array("neutral")}
    for k, (a, b) in bd.POSE_SLICES.items():
        fit[k] = rng.normal(size=(n, (b - a) * 3)) * 0.3
    o = bd.forward(fit, points=True)
    tb = bd.TorchBody(fit, "cpu", torch.float64)
    R, p, J = tb.fk(torch.tensor(bd.rotmat(bd.pose55(fit))), torch.tensor(fit["transl"]))
    assert np.abs(J.numpy() - o["J"]).max() < 1e-9
    assert np.abs(tb.points(R, p).numpy() - o["pts"]).max() < 1e-9
    # rodrigues_t = rotmat, smooth at 0
    rv = torch.tensor(rng.normal(size=(7, 3)) * 0.7)
    assert np.abs(bd.rodrigues_t(torch, rv).numpy() - bd.rotmat(rv.numpy())).max() < 1e-12
    z = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    bd.rodrigues_t(torch, z).sum().backward()
    assert torch.isfinite(z.grad).all()


def test_g1_hinge_matches_the_gate_statistic():
    """E_g1 is zero for the base itself and for a change inside the gate's tolerance, and grows once one camera's per-frame median
    error (median / p90 over frames) passes the base's + tolerance - margin"""
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(3)
    eb = torch.tensor(rng.uniform(3.0, 8.0, (40, 12)), dtype=torch.float32)
    m = torch.ones_like(eb, dtype=torch.bool)
    args = (5, 0.2, 1.0, 3.0, 0.1, 0.5)
    assert float(sv.g1_hinge(torch, eb.clone(), eb, m, *args)) == 0.0
    assert float(sv.g1_hinge(torch, eb + 0.3, eb, m, *args)) == 0.0
    worse = eb.clone().requires_grad_(True)
    h = sv.g1_hinge(torch, worse + 4.0, eb, m, *args)
    assert float(h) > 1.0
    h.backward()
    assert float(worse.grad.abs().sum()) > 0
    # too few visible points in too many frames: the camera is not measured (as in the gate)
    m2 = torch.zeros_like(m)
    m2[:4] = True
    assert sv.g1_hinge(torch, eb + 4.0, eb, m2, *args) is None
