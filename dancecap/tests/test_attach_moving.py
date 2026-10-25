"""dancecap.cameras.attach_moving on a synthetic walking camera: two moving skeletons, a camera circling them at a known focal."""
import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("scipy")

from dancecap.cameras import attach_moving as am  # noqa: E402


def _look_at(C, target):
    z = target - C
    z /= np.linalg.norm(z)
    x = np.cross(z, [0.0, 1.0, 0.0])          # camera x right (world up = +Y, image y down)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z])                 # rows: camera axes in world -> cam_from_world


def _scene(F=150, fps=29.0, f=1100.0, size=(720, 1280), noise=1.5, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(F) / fps
    # 22 SMPL-X-like joints per dancer: a rigid random body (0..1.6 m tall) that walks and turns
    base = rng.uniform([-0.25, 0.0, -0.15], [0.25, 1.6, 0.15], size=(22, 3))
    joints = {}
    for r, off in zip(am.ROLES, (-0.4, 0.4)):
        J = np.empty((F, 22, 3))
        for i, ti in enumerate(t):
            a = 0.8 * ti + (0 if r == "lead" else 1.0)
            Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
            J[i] = base @ Ry.T + [off + 0.3 * np.sin(ti), 0.0, 0.2 * np.cos(0.7 * ti)]
        joints[r] = J
    K = am.intrinsics(f, size)
    ang = 0.6 * t
    C = np.stack([3.0 * np.sin(ang), 1.4 + 0.05 * np.sin(2 * t), 3.0 * np.cos(ang)], -1)
    R = np.stack([_look_at(c, np.array([0.0, 0.9, 0.0])) for c in C])
    kp = np.full((F, 2, 17, 3), np.nan)
    for i in range(F):
        for ri, r in enumerate(am.ROLES):
            uv, z = am.project(joints[r][i][list(am.SMPLX_IDS)], R[i], -R[i] @ C[i], K)
            kp[i, ri, list(am.COCO_IDS), :2] = uv + rng.normal(0, noise, uv.shape)
            kp[i, ri, list(am.COCO_IDS), 2] = 0.9
    conf = np.full((F, 2), 0.9)
    return t, joints, kp, conf, K, R, C, size, fps


def test_sparsity_shape_matches_residuals():
    fid = np.array([0, 0, 1, 2, 2, 2])
    A = am.sparsity(fid, 4)
    assert A.shape == (2 * 6 + 2 * 3 * 2, 24)
    # every correspondence row touches exactly its frame's 6 numbers
    assert A[0].nnz == 6 and set(A[0].indices) == {0, 1, 2, 12, 13, 14}


def test_interp_joints_invalid_far_from_fit():
    tf = np.array([0.0, 1 / 30, 2 / 30])
    J = np.zeros((3, 1, 3))
    J[:, 0, 0] = [0.0, 1.0, 2.0]
    out, ok = am.interp_joints(tf, J, np.array([1 / 60, 0.5]))
    assert ok.tolist() == [True, False]
    assert out[0, 0, 0] == pytest.approx(0.5)


def test_recovers_focal_and_walking_path():
    t, joints, kp, conf, K, R_true, C_true, size, fps = _scene()
    obs = am.correspondences(t, kp, conf, None, t, joints, 0.0)
    foc = am.search_focal(obs, size, (600.0, 2400.0), thr_px=30.0, n_frames=60)
    assert foc["focal_px"] == pytest.approx(1100.0, rel=0.03)
    Kh = am.intrinsics(foc["focal_px"], size)
    import cv2
    R0 = np.zeros_like(R_true)
    C0 = np.zeros_like(C_true)
    for i, (o, p, _) in enumerate(obs):
        r = am.pnp(o, p, Kh, 30.0)
        R0[i] = cv2.Rodrigues(r[0])[0]
        C0[i] = -R0[i].T @ r[1]
    R, C, info = am.joint_solve(obs, Kh, R0, C0, fps)
    err = np.linalg.norm(C - C_true, axis=-1)
    assert np.median(err) < 0.05
    px, pct = am.per_frame_residual(obs, R, C, Kh, np.full((len(t), 2), 400.0))
    assert np.nanmedian(px) < 4.0


def test_robust_smooth_removes_jitter_and_outlier_frames():
    rng = np.random.default_rng(1)
    t = np.arange(300) / 29.0
    C_true = np.stack([np.sin(0.5 * t), 1.2 + 0 * t, np.cos(0.5 * t)], -1) * [2.0, 1.0, 2.0]
    R_true = np.stack([_look_at(c, np.array([0.0, 0.9, 0.0])) for c in C_true])
    C = C_true + rng.normal(0, 0.08, C_true.shape)
    C[::37] += 1.5                                    # gross PnP failures
    w = np.full(len(t), 20.0)
    R, Cs, kept = am.robust_smooth(t, R_true, C, w, 0.25)
    raw = np.median(np.linalg.norm(C - C_true, axis=-1))
    assert np.median(np.linalg.norm(Cs - C_true, axis=-1)) < 0.5 * raw
    assert not kept[::37].any()
