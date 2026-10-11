"""Pure-geometry tests of dancecap.cameras.single_view / manhattan / couple (synthetic, no media)."""
import numpy as np

from dancecap.cameras import couple, manhattan, single_view as SV


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def test_camera_height_recovers_height_of_a_standing_person():
    h_cam, stature = 1.45, 1.75
    eye = SV.eye_height(stature)
    rng = np.random.default_rng(0)
    pos = np.c_[rng.uniform(-3, 3, 50), np.zeros(50), rng.uniform(4, 12, 50)]       # floor positions in front of the camera (z forward)
    cam = np.array([0, h_cam, 0.0])
    foot = _unit(pos + [0, SV.ANKLE_JOINT_M, 0] - cam)
    head = _unit(pos + [0, eye, 0] - cam)
    h = SV.camera_height(foot, head, eye)
    assert np.allclose(h, h_cam, atol=1e-9)


def test_floor_points_roundtrip_and_focal_scales_distance():
    h_cam = 1.5
    pos = np.array([[1.0, 0, 7.0], [-2.0, 0, 9.0]])
    foot = _unit(pos + [0, SV.ANKLE_JOINT_M, 0] - [0, h_cam, 0])
    p = SV.floor_points(foot, h_cam)
    assert np.allclose(p[:, [0, 2]], pos[:, [0, 2]])
    # the same pixels with a focal 10 % longer: the rays move towards the axis, the person is 10 % farther, the height does not change
    f1, f2 = 1000.0, 1100.0
    px = lambda r, f: np.c_[r[:, 0] / r[:, 2] * f, -r[:, 1] / r[:, 2] * f]
    back = lambda p_, f: _unit(np.c_[p_[:, 0] / f, -p_[:, 1] / f, np.ones(len(p_))])      # y of the image grows downwards: world y up = -image y
    head = _unit(pos + [0, SV.eye_height(1.7), 0] - [0, h_cam, 0])
    pf, ph = px(foot, f1), px(head, f1)
    h1 = SV.camera_height(back(pf, f1), back(ph, f1), SV.eye_height(1.7))
    h2 = SV.camera_height(back(pf, f2), back(ph, f2), SV.eye_height(1.7))
    assert np.allclose(h1, h_cam, atol=1e-6)
    assert np.all(np.abs(h2 - h_cam) < 0.05)                    # nearly independent of the focal
    d1 = np.linalg.norm(SV.floor_points(back(pf, f1), h_cam)[:, [0, 2]], axis=1)
    d2 = np.linalg.norm(SV.floor_points(back(pf, f2), h_cam)[:, [0, 2]], axis=1)
    assert np.allclose(d2 / d1, 1.1, atol=0.02)


def test_horizon_basis_is_right_handed_and_vertical():
    x, y, z = SV.horizon_basis(np.array([0.1, 1.0, 0.05]))
    assert np.allclose(np.cross(x, y), -z) or np.allclose(np.cross(x, y), z)
    assert abs(x @ y) < 1e-12 and abs(z @ y) < 1e-12 and abs(np.linalg.det(np.stack([x, y, z]))) > 0.99


def test_manhattan_fit_finds_vertical_and_a_horizontal_family():
    rng = np.random.default_rng(1)
    up = _unit([0.06, 1.0, -0.04])
    a = _unit(np.cross(up, [0, 0, 1.0]))
    b = np.cross(up, a)
    n = []
    for d, k in ((up, 150), (a, 300), (b, 100)):                # lines of direction d have normals perpendicular to d
        u = _unit(np.cross(d, rng.normal(size=(k, 3))))
        n.append(u + rng.normal(scale=0.004, size=u.shape))
    n.append(_unit(rng.normal(size=(150, 3))))                  # clutter
    n = _unit(np.concatenate(n))
    fr = manhattan.fit_frame(n, np.ones(len(n)), np.array([0, 1.0, 0]))
    assert abs(fr.M[:, 0] @ up) > np.cos(np.radians(0.5))
    assert max(abs(fr.M[:, 1] @ a), abs(fr.M[:, 2] @ a)) > np.cos(np.radians(1.0))
    assert fr.share_v > 0.15 and fr.share_a + fr.share_b > 0.3


def test_couple_pick_takes_the_dark_torso_and_his_partner():
    # frame 0: dark man + overlapping lighter woman + a far person; frame 1: nobody dark
    box = np.array([[100, 200, 200, 500], [150, 200, 260, 520], [400, 300, 430, 420], [100, 200, 200, 500]], float)
    rgb = np.array([[20, 20, 20], [150, 90, 90], [30, 30, 30], [200, 200, 200]], float)
    frame = np.array([0, 0, 0, 1])
    lead, foll, score = couple.pick(frame, box, rgb, 2)
    assert (lead[0], foll[0]) == (0, 1)
    assert lead[1] == -1 and foll[1] == -1


def test_rescale_up_keeps_the_horizon_row():
    f0, f1 = 800.0, 1000.0
    pitch = np.radians(6.0)                                     # camera looks down by 6 deg: up (camera frame, y down, z forward) = (0, -cos, -sin)
    u0 = np.array([0.0, -np.cos(pitch), -np.sin(pitch)])
    y_hor0 = 480 - f0 * np.tan(pitch)                           # horizon row at the picture centre column
    u1 = SV.rescale_up(u0, f0, f1)
    assert abs(np.linalg.norm(u1) - 1) < 1e-12
    pitch1 = np.degrees(np.arctan2(-u1[2], -u1[1]))
    assert abs((480 - f1 * np.tan(np.radians(pitch1))) - y_hor0) < 1e-6


def test_plane_ransac_and_world_basis_and_camera_arrays():
    rng = np.random.default_rng(3)
    n_true = _unit([0.05, -1.0, -0.1])                          # up, camera frame
    d_true = 1.4
    a = _unit(np.cross(n_true, [1.0, 0, 0]))
    b = np.cross(n_true, a)
    P = -d_true * n_true + rng.uniform(-5, 5, (400, 1)) * a + rng.uniform(1, 12, (400, 1)) * b + rng.normal(scale=0.01, size=(400, 3))
    P[:80] += rng.normal(scale=1.0, size=(80, 3))               # outliers
    n, d, inl = SV.plane_ransac(P, rng, tol=0.05, up_prior=n_true)
    assert abs(n @ n_true) > 0.999 and abs(abs(d) - d_true) < 0.03
    W = SV.world_basis(n_true, np.array([1.0, 0, 0]), np.array([0, 0, 1.0]))
    assert abs(np.linalg.det(W) - 1) < 1e-9 and np.allclose(W.T @ W, np.eye(3), atol=1e-9)
    assert np.array([0, 0, 1.0]) @ W[:, 2] <= 0                 # the camera looks towards -Z
    R, tvec, K, dist, C = SV.camera_arrays(np.zeros((2, 3)), np.array([800.0, 1600.0]), W, np.array([1.0, 1.4, 5.0]), (538, 960), -0.01)
    # the camera centre in the world is C: R C + tvec = 0, and a point 5 m in front of the camera projects at the principal point
    assert np.allclose(np.einsum("nab,nb->na", R, C) + tvec, 0, atol=1e-12)
    fwd_world = W.T @ np.array([0, 0, 1.0])
    X = C[0] + 5 * fwd_world
    xc = R[0] @ X + tvec[0]
    assert np.allclose(xc[:2], 0, atol=1e-9) and K[1, 0, 0] == 1600.0 and K[0, 0, 2] == 269.0


def test_track_pairs_follows_the_continuous_couple():
    # two candidate pairs per frame: a near (big) couple whose floor position moves smoothly and a decoy that jumps around with a higher score in some frames
    n = 6
    box = np.zeros((4 * n, 4))
    rgb = np.zeros((4 * n, 3))
    centre = np.zeros((4 * n, 2))
    cands = []
    for t in range(n):
        d, l, d2, l2 = 4 * t, 4 * t + 1, 4 * t + 2, 4 * t + 3
        centre[[d, l]] = [0.1 * t, 4.0]
        centre[[d2, l2]] = [(-1) ** t * 3.0, 7.0 + t]
        cands.append([(2.9e4, d2, l2), (3e4, d, l)] if t != 2 else [(5e4, d2, l2), (2.5e4, d, l)])      # the decoy scores higher in frame 2
    lead, foll = couple.track_pairs(cands, centre)
    assert [int(v) % 4 for v in lead] == [0] * n and [int(v) % 4 for v in foll] == [1] * n


def test_fill_bad_frames_interpolates_short_gaps_only():
    from scipy.spatial.transform import Rotation as Rot
    n = 60
    ang = np.linspace(0, 0.3, n)
    rv = np.stack([ang, 0 * ang, 0 * ang], 1)
    f = np.full(n, 800.0)
    ok = np.ones(n, bool)
    ok[10:13] = False                     # short gap -> interpolated
    ok[30:50] = False                     # 20 frames <= max_gap 30 -> interpolated as well
    rv_bad = rv.copy()
    rv_bad[~ok] = [1.0, 1.0, 1.0]         # garbage in the bad frames
    rv2, f2, ok2, interp = SV.fill_bad_frames(rv_bad, f, ok, max_gap=25)
    assert ok2.all() and interp.sum() == 23
    assert np.abs(Rot.from_rotvec(rv2).as_rotvec() - rv).max() < 1e-6
    ok[30:50] = False
    ok[3] = False
    rv3, f3, ok3, interp3 = SV.fill_bad_frames(rv_bad, f, ok, max_gap=5)
    assert ok3[10] and not ok3[35] and interp3[3]


def test_lines_detect_and_plane_normal_are_consistent():
    import cv2
    from dancecap.cameras import lines
    img = np.zeros((300, 400), np.uint8)
    cv2.line(img, (20, 60), (380, 90), 255, 2)               # a long bright segment
    cv2.line(img, (200, 10), (205, 290), 255, 2)
    seg = lines.detect(img, None, min_len=60)
    assert len(seg) >= 2
    n, mid = lines.plane_normals(seg, 600.0, np.array([200.0, 150.0]))
    # the plane of a segment contains its end rays and its middle ray
    from dancecap.cameras.ptz import Lens, pix_to_ray
    r1 = pix_to_ray(seg[:, 0:2].astype(float), np.full(len(seg), 600.0), Lens(400, 300))
    assert np.abs((n * r1).sum(1)).max() < 1e-9 and np.abs((n * mid).sum(1)).max() < 1e-9
    # an ignore mask over a segment removes it
    mask = np.zeros_like(img)
    cv2.rectangle(mask, (0, 40), (399, 110), 255, -1)
    seg2 = lines.detect(img, mask, min_len=60)
    assert len(seg2) < len(seg)


def test_geom_priors_small_handles_bool_and_float_maps():
    from dancecap.cameras import geom_priors
    m = np.zeros((960, 538), bool)
    m[100:500, 100:300] = True
    s = geom_priors.small(m, 384)
    assert s.dtype == bool and max(s.shape) == 384 and s.any()
    x = np.random.default_rng(0).normal(size=(960, 538, 3)).astype(np.float32)
    assert geom_priors.small(x, 384).shape == (384, 215, 3)
    assert geom_priors.small(x[:200, :100], 384).shape == (200, 100, 3)           # smaller than the target: unchanged
