"""dancecap.cameras.twocam: relative pose of two panning cameras from synchronised joints (synthetic) + floor / scale."""
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from dancecap.cameras import ptz as P
from dancecap.cameras import twocam as T


def _world(seed=0, n_t=60, noise=0.5, fa=4500.0, fb=3100.0):
    rng = np.random.default_rng(seed)
    lens = P.Lens(1920, 1080)
    Qtrue = Rot.from_euler("YXZ", [-9, 2, 1], degrees=True).as_matrix()      # B reference -> A reference
    b_true = np.array([0.8, -0.05, 0.6]); b_true /= np.linalg.norm(b_true)
    scale = 3.5                                                               # baseline length, "metres"
    # a dancer: 17 joints around a standing figure, walking on the floor plane y = 1.4 (y down) at 9 m from camera A
    rows = []
    for t in range(n_t):
        base = np.array([rng.uniform(-2.5, 2.5), 1.4, rng.uniform(7, 11)])
        for p in range(2):
            c = base + np.array([0.5 * p, 0, 0.3 * p])
            for j in range(17):
                h = {0: 1.68, 5: 1.45, 6: 1.45, 11: 0.95, 12: 0.95, 13: 0.5, 14: 0.5, 15: 0.07, 16: 0.07}.get(j, 1.2)
                rows.append((t, p, j, c + np.array([rng.normal(0, 0.15), -h, rng.normal(0, 0.1)])))
    t_idx = np.array([r[0] for r in rows]); person = np.array([r[1] for r in rows]); joint = np.array([r[2] for r in rows])
    X = np.array([r[3] for r in rows])
    ra = [Rot.from_euler("YXZ", [3 * np.sin(k / 9), 0.4 * np.sin(k / 7), 0], degrees=True).as_matrix() for k in range(n_t)]
    rb = [Rot.from_euler("YXZ", [4 + 3 * np.sin(k / 8), 0.3, 0], degrees=True).as_matrix() for k in range(n_t)]
    va = T.View(np.array(ra), np.full(n_t, fa), lens)
    vb = T.View(np.array(rb), np.full(n_t, fb), lens)
    C_B = scale * b_true
    pa = va.project_ref(t_idx, X)
    d_b_prov = X - C_B
    pb = vb.project_ref(t_idx, d_b_prov @ Qtrue)            # Q^T d
    pa = pa + rng.normal(0, noise, pa.shape); pb = pb + rng.normal(0, noise, pb.shape)
    ok = np.isfinite(pa).all(1) & np.isfinite(pb).all(1) & (pa[:, 0] > 0) & (pa[:, 0] < 1920) & (pb[:, 0] > 0) & (pb[:, 0] < 1920) \
        & (pa[:, 1] > 0) & (pa[:, 1] < 1080) & (pb[:, 1] > 0) & (pb[:, 1] < 1080)
    corr = T.Corr(t_idx[ok], person[ok], joint[ok], pa[ok], pb[ok], np.ones(ok.sum()))
    return corr, va, vb, Qtrue, b_true, X[ok]


def test_rig_recovered_from_joints():
    corr, va, vb, Qtrue, b_true, X = _world()
    rig, info = T.solve_rig(corr, va, vb)
    ang = np.degrees(np.arccos(np.clip((np.trace(rig.Q @ Qtrue.T) - 1) / 2, -1, 1)))
    assert ang < 0.1
    assert np.degrees(np.arccos(np.clip(rig.b @ b_true, -1, 1))) < 0.5
    assert info["median_px"] < 1.0


def test_triangulation_is_metric_up_to_baseline_scale():
    corr, va, vb, Qtrue, b_true, X = _world(seed=2)
    rig, _ = T.solve_rig(corr, va, vb)
    Xp, err, ok = T.triangulate_all(rig, corr, va, vb)
    assert ok.mean() > 0.99
    # points are X scaled by 1/|C_B| (here 3.5): pairwise distances scale uniformly
    i = np.arange(0, len(X), 7)[:80]
    d_true = np.linalg.norm(X[i][:, None] - X[i][None], axis=2)
    d_est = np.linalg.norm(Xp[i][:, None] - Xp[i][None], axis=2)
    m = d_true > 1
    ratio = d_est[m] / d_true[m]
    assert abs(np.median(ratio) - 1 / 3.5) < 0.01
    assert np.percentile(np.abs(ratio / np.median(ratio) - 1), 90) < 0.05


def test_plane_and_up_frame():
    rng = np.random.default_rng(1)
    n_true = np.array([0.1, -0.98, 0.15]); n_true /= np.linalg.norm(n_true)
    a = np.cross(n_true, [1, 0, 0]); a /= np.linalg.norm(a); b = np.cross(n_true, a)
    pts = 5 * rng.uniform(-1, 1, (300, 1)) * a + 5 * rng.uniform(-1, 1, (300, 1)) * b + 2.0 * n_true + rng.normal(0, 0.01, (300, 3))
    out = np.vstack([pts, rng.uniform(-5, 5, (60, 3))])
    n, d, m = T.ransac_plane(out, 0.05)
    assert abs(abs(n @ n_true) - 1) < 1e-3
    assert m.sum() >= 290
    R = T.rotation_to_up(n * np.sign(-n[1]), np.array([0, 0, 1.0]))
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9) and abs(np.linalg.det(R) - 1) < 1e-9


def _rigid_world(seed=4, n_t=120, noise=1.0, fa=2600.0, fb=1900.0):
    """Two dancers as RIGID skeletons (fixed bone lengths) walking and turning on the floor; y is DOWN (camera-like frame)."""
    rng = np.random.default_rng(seed)
    lens = P.Lens(1920, 1080)
    # standing template (x right, y down, z forward), metres: joint -> (x, y_below_head_top)
    H = 1.70
    tmpl = np.zeros((17, 3))
    ys = {0: 0.10, 1: 0.12, 2: 0.12, 3: 0.12, 4: 0.12, 5: 0.30, 6: 0.30, 7: 0.55, 8: 0.55, 9: 0.80, 10: 0.80, 11: 0.88, 12: 0.88,
          13: 1.28, 14: 1.28, 15: 1.62, 16: 1.62}
    xs = {0: 0, 1: -0.03, 2: 0.03, 3: -0.07, 4: 0.07, 5: -0.19, 6: 0.19, 7: -0.22, 8: 0.22, 9: -0.24, 10: 0.24, 11: -0.09, 12: 0.09,
          13: -0.10, 14: 0.10, 15: -0.10, 16: 0.10}
    for j in range(17):
        tmpl[j] = [xs[j], ys[j] - H, 0.0]                 # head top at y = -H above the sole (y down: up is negative)
    Qtrue = Rot.from_euler("YXZ", [-9, 2, 1], degrees=True).as_matrix()
    b_true = np.array([0.8, -0.05, 0.6]); b_true /= np.linalg.norm(b_true)
    C_B = 3.5 * b_true
    rows = []
    for t in range(n_t):
        base = np.array([rng.uniform(-2.5, 2.5), 1.4, rng.uniform(7, 11)])
        for p in range(2):
            yaw = rng.uniform(0, 2 * np.pi)
            Ry = Rot.from_euler("y", yaw).as_matrix()
            pose = tmpl * np.array([1, 1 - 0.0, 1])
            lean = rng.uniform(-0.25, 0.25)
            Rl = Rot.from_euler("x", lean).as_matrix()
            Xj = (pose @ Rl.T) @ Ry.T + base + np.array([0.6 * p, 0, 0.3 * p])
            for j in range(17):
                rows.append((t, p, j, Xj[j]))
    t_idx = np.array([r[0] for r in rows]); person = np.array([r[1] for r in rows]); joint = np.array([r[2] for r in rows])
    X = np.array([r[3] for r in rows])
    ra = [Rot.from_euler("YXZ", [3 * np.sin(k / 9), 0.4 * np.sin(k / 7), 0], degrees=True).as_matrix() for k in range(n_t)]
    rb = [Rot.from_euler("YXZ", [4 + 3 * np.sin(k / 8), 0.3, 0], degrees=True).as_matrix() for k in range(n_t)]
    va = T.View(np.array(ra), np.full(n_t, fa), lens)
    vb = T.View(np.array(rb), np.full(n_t, fb), lens)
    pa = va.project_ref(t_idx, X) + rng.normal(0, noise, (len(X), 2))
    pb = vb.project_ref(t_idx, (X - C_B) @ Qtrue) + rng.normal(0, noise, (len(X), 2))
    ok = np.isfinite(pa).all(1) & np.isfinite(pb).all(1) & (pa[:, 0] > 0) & (pa[:, 0] < 1920) & (pb[:, 0] > 0) & (pb[:, 0] < 1920) \
        & (pa[:, 1] > 0) & (pa[:, 1] < 1080) & (pb[:, 1] > 0) & (pb[:, 1] < 1080)
    corr = T.Corr(t_idx[ok], person[ok], joint[ok], pa[ok], pb[ok], np.ones(ok.sum()))
    return corr, va, vb, Qtrue, b_true


def test_rigid_bones_penalise_a_wrong_focal():
    corr, va, vb, Qtrue, b_true = _rigid_world()
    rig, info = T.solve_rig(corr, va, vb, bone_weight=1.0)
    assert info["bone_rel_mad"] < 0.03
    costs = {}
    for fb in (1500.0, 1900.0, 2400.0):
        vb2 = T.View(vb.R, np.full(len(vb.f), fb), vb.lens)
        _, inf = T.solve_rig(corr, va, vb2, bone_weight=1.0)
        costs[fb] = inf["cost"]
    assert costs[1900.0] < costs[1500.0] and costs[1900.0] < costs[2400.0]


def test_gravity_terms_recover_floor_and_keep_the_rig():
    corr, va, vb, Qtrue, b_true = _rigid_world(seed=7)
    # in this synthetic frame y is DOWN, the floor is y = 1.4: the 'up' direction is -y
    up = np.array([0.0, -1.0, 0.0])
    g = {"vA": up, "vB": Qtrue.T @ up, "sigma_v_rad": np.radians(0.5), "sigma_h": 0.1, "px_per_sigma": 3.0}
    rig, info = T.solve_rig(corr, va, vb, gravity=g)
    assert info["median_px"] < 1.5
    gi = info["gravity"]
    assert gi["vA_vs_u_deg"] < 0.5 and gi["QvB_vs_u_deg"] < 0.5
    # lower ankles sit on the floor: height scatter small relative to the 1.4 m camera height scale
    assert gi["ankle_h_mad"] < 0.2 * 3.5
