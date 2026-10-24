"""cameras.anchored: the dancer judge of a moving camera's poses, the motion gate, growth and the prior gate."""
import numpy as np

from dancecap.cameras import anchored as A


def _look_at(C, target):
    """OpenCV camera at C looking at target (y down)."""
    z = np.asarray(target, float) - C
    z /= np.linalg.norm(z)
    x = np.cross([0.0, 1.0, 0.0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return R, -R @ C


def _project(R, t, f, pp, X):
    Xc = X @ R.T + t
    return f * Xc[:, :2] / Xc[:, 2:3] + pp


class _Judge(A.Judge):
    """A Judge on synthetic data: a fixed camera and the dancers' joints at the same instants (no files)."""

    def __init__(self, fixed, kp_f, t, pp, world=True):
        self.offset_s, self.w, self.h, self.pp = 0.0, 720, 1280, pp
        self.fx, self.kt_f, self.kp_f = fixed, t, kp_f
        self.kt_m = t
        # the SfM frame is y-down with the floor at y = 0: world y = -y
        self.world = {"s": 1.0, "R": np.diag([1.0, -1.0, -1.0]), "t": np.zeros(3), "ankle_h": 0.084} if world else None


def _scene(seed=0, n=6):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 30.0
    X = rng.uniform([-0.5, -1.7, -0.5], [0.5, -0.2, 0.5], (n, 2, 17, 3))  # two dancers' joints near the origin, y up = -y
    X[:, :, 15:17, 1] = -0.084  # both ankles on the floor (y = 0) at the ankle joint's height
    Rf, tf = _look_at(np.array([0.0, -1.5, -6.0]), [0, -1, 0])
    ff, ppf = 1500.0, np.array([639.5, 359.5])
    kf = np.concatenate([_project(Rf, tf, ff, ppf, X.reshape(-1, 3)).reshape(n, 2, 17, 2), np.ones((n, 2, 17, 1))], -1)
    fixed = {"t": t, "R": np.repeat(Rf[None], n, 0), "tvec": np.repeat(tf[None], n, 0), "C": np.repeat(np.array([[0.0, -1.5, -6.0]]), n, 0),
             "f": np.full(n, ff), "pp": np.repeat(ppf[None], n, 0), "k1": np.zeros(n), "ok": np.ones(n, bool), "dt": 1 / 30}
    Cm = np.array([4.0, -1.5, 2.0])
    Rm, tm = _look_at(Cm, [0, -1, 0])
    fm, ppm = 1700.0, np.array([359.5, 639.5])
    km = np.concatenate([_project(Rm, tm, fm, ppm, X.reshape(-1, 3)).reshape(n, 2, 17, 2), np.ones((n, 2, 17, 1))], -1)
    return t, fixed, kf, km, (Rm, tm, fm, ppm, Cm)


def test_judge_accepts_the_true_pose_and_refuses_the_other_wall():
    t, fixed, kf, km, (Rm, tm, fm, ppm, Cm) = _scene()
    J = _Judge(fixed, kf, t, ppm)
    n = len(t)
    res, nj = J.residual(t, km, np.repeat(Rm[None], n, 0), np.repeat(tm[None], n, 0), np.full(n, fm), np.zeros(n))
    assert (nj == 34).all() and np.nanmax(res) < 0.5  # exact data: ~0 px
    epi, xf, _ = J.judge(t, km, np.repeat(Rm[None], n, 0), np.repeat(tm[None], n, 0), np.full(n, fm), np.zeros(n))
    assert np.nanmax(xf) < 1.0 and J.anchor(epi, xf).all()
    # the other wall: the true camera reflected across a plane between two identical walls (x = -2), as a registration
    # on the repeated structure puts it - its view direction comes from that wall, not from the dancers
    Cw = Cm * [-1, 1, 1] + [-4.0, 0, 0]
    R2, t2 = _look_at(Cw, np.array([0.0, -1.0, 0.0]) * [-1, 1, 1] + [-4.0, 0, 0])
    epi2, xf2, _ = J.judge(t, km, np.repeat(R2[None], n, 0), np.repeat(t2[None], n, 0), np.full(n, fm), np.zeros(n))
    assert J.refused(epi2, xf2).all() and not J.anchor(epi2, xf2).any()
    # a pose on the symmetric side that still faces the dancers can pass the epipolar test (one-dimensional: both
    # cameras at one height) and even the transfer: the motion gate is the defence there (test_motion_gate_...)
    R3, t3 = _look_at(np.array([-4.0, -1.5, 2.5]), [0, -1, 0])
    epi3, _, _ = J.judge(t, km, np.repeat(R3[None], n, 0), np.repeat(t3[None], n, 0), np.full(n, fm), np.zeros(n))
    assert (epi3 < A.LOOSE * A.TAU_PX).any()
    J.world = None  # no world file: the epipolar test alone, the transfer unknown
    assert np.isnan(J.judge(t, km, np.repeat(R2[None], n, 0), np.repeat(t2[None], n, 0), np.full(n, fm), np.zeros(n))[1]).all()
    few = km.copy()
    few[:, :, 4:, 2] = 0.1  # 4 joints per dancer measurable: 8 in all - still judged; 3 each: not
    assert np.isfinite(J.residual(t, few, np.repeat(Rm[None], n, 0), np.repeat(tm[None], n, 0), np.full(n, fm), np.zeros(n))[0]).all()
    few[:, :, 3, 2] = 0.1
    assert np.isnan(J.residual(t, few, np.repeat(Rm[None], n, 0), np.repeat(tm[None], n, 0), np.full(n, fm), np.zeros(n))[0]).all()


def test_motion_gate_and_grow_stop_at_a_jump_to_another_wall():
    room = 12.0
    n = 12
    t = np.arange(n) / 29.0
    C = np.c_[np.linspace(0, 0.3, n), np.zeros(n), np.zeros(n)]
    R = np.repeat(np.eye(3)[None], n, 0)
    C[6:9] += [8.0, 0.0, 0.0]  # frames 6-8 on another wall: a 8-unit jump in one frame
    ok = np.ones(n, bool)
    refused = np.zeros(n, bool)  # no dancers measurable: continuity alone
    acc0 = np.zeros(n, bool)
    acc0[0] = True
    acc = A.grow(t, ok, C, R, refused, acc0, room)
    assert acc[:6].all() and not acc[6:9].any()
    assert not acc[9:].any()  # 3 frames past the last accepted one: beyond NEAR, nothing seeds them
    acc0[11] = True  # an anchor on the far side grows backward
    acc = A.grow(t, ok, C, R, refused, acc0, room)
    assert acc[9:].all() and not acc[6:9].any()
    refused[3] = True  # the judge refuses frame 3: it stays out
    acc = A.grow(t, ok, C, R, refused, acc0, room)
    assert not acc[3]


def test_prior_gate_keeps_the_matches_of_the_prior_s_wall():
    R, t = _look_at(np.array([0.0, -1.5, -6.0]), [0, -1, 0])
    rng = np.random.default_rng(1)
    X = rng.uniform([-4, -3, 3], [4, 0, 6], (50, 3))
    w, h, f = 1280, 720, 1500.0
    kp = _project(R, t, f, np.array([w / 2.0, h / 2.0]), X)  # COLMAP px
    wrong = X.copy()
    wrong[:, 0] *= -1.0  # the same-looking structure mirrored to the other wall
    g = A.prior_gate(np.r_[X, wrong], np.r_[kp, kp], R, t, f, 0.0, w, h)
    assert g[:50].all() and g[50:].mean() < 0.2
    behind = X.copy()
    behind[:, 2] = -behind[:, 2] - 12.0
    assert not A.prior_gate(behind, kp, R, t, f, 0.0, w, h).any()
