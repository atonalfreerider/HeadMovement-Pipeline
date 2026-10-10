"""dancecap.cameras.twocam_world: camera arrays in the world frame, yaw re-orientation, floor rays."""
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from dancecap.cameras import twocam_world as W
from dancecap.cameras import room_draw as D


def test_camera_arrays_project_a_known_world_point():
    rng = np.random.default_rng(0)
    Rcp = Rot.from_euler("YXZ", [5, -2, 0.5], degrees=True).as_matrix()[None]
    Rw = Rot.from_euler("y", 33, degrees=True).as_matrix() @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0.0]])   # some prov -> world rotation (det +1)
    assert abs(np.linalg.det(Rw) - 1) < 1e-9
    s, tw = 4.0, np.array([1.0, 0.5, -2.0])
    C_prov = np.array([0.9, -0.1, 0.4])
    cam = W.camera_arrays(Rcp, np.array([3000.0]), (960.0, 540.0), 0.0, C_prov, s, Rw, tw)
    Xp = np.array([[0.2, 0.1, 2.0]])                       # a point in the provisional frame (units of the baseline)
    Xw = W.to_world(Xp, s, Rw, tw)
    # camera frame: metres, X_c = s R_cp (X_prov - C_prov)
    Xc_expected = s * (Rcp[0] @ (Xp[0] - C_prov))
    Xc = cam["R"][0] @ (Xw[0] - cam["C"][0])
    assert np.allclose(Xc, Xc_expected, atol=1e-9)
    assert np.allclose(cam["tvec"][0], -cam["R"][0] @ cam["C"][0])
    p, z = D.project(Xw, cam["R"][0], cam["tvec"][0], 3000.0, (960.0, 540.0))
    assert np.allclose(p[0], np.array([960.0, 540.0]) + 3000.0 * Xc_expected[:2] / Xc_expected[2])
    assert np.allclose(cam["R"][0] @ cam["R"][0].T, np.eye(3), atol=1e-9)


def test_re_yaw_keeps_up_and_moves_axes():
    Rw = np.eye(3)
    tw = np.array([1.0, 2.0, 3.0])
    R2, t2 = W.re_yaw(Rw, tw, 30.0)
    X = np.array([[0.5, 0.2, -0.4]])
    a = W.to_world(X, 1.0, Rw, tw)
    b = W.to_world(X, 1.0, R2, t2)
    assert np.allclose(a[:, 1], b[:, 1])                    # heights unchanged
    assert np.allclose(np.linalg.norm(a[:, [0, 2]], axis=1), np.linalg.norm(b[:, [0, 2]], axis=1))
    assert np.allclose(R2 @ R2.T, np.eye(3))


def test_pixel_rays_hit_the_floor_where_projected():
    R = Rot.from_euler("x", 3, degrees=True).as_matrix() @ np.diag([1.0, -1.0, -1.0])   # looks along -z, y down in the image (world->camera)
    C = np.array([2.0, 1.6, 8.0])
    tvec = -R @ C
    K = np.array([[3000.0, 0, 960], [0, 3000.0, 540], [0, 0, 1]])
    Xw = np.array([[0.5, 0.0, -1.0], [-2.0, 0.0, 0.5]])
    p, z = D.project(Xw, R, tvec, 3000.0, (960.0, 540.0))
    d = W.pixel_rays_world(p, K, R)
    hit = W.ray_floor_hit(C, d)
    assert np.allclose(hit, Xw, atol=1e-6)
