"""dancecap.room_two_cams / room_two_cams_walls: the pure parts (ray-plane hits, edge scan, orthographic texture, OBJ writer)."""
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from dancecap import room_two_cams as RT
from dancecap import room_two_cams_walls as WL
from dancecap.cameras import room_draw as D
from dancecap.cameras import twocam_world as TW


def _cam(f=3000.0, h=1.4, z=9.0, pitch_deg=2.0, size=(1920, 1080)):
    """A world camera 'looking along -z': arrays in the cameras/<cam>.npz layout, one frame."""
    R = Rot.from_euler("x", pitch_deg, degrees=True).as_matrix() @ np.diag([1.0, -1.0, -1.0])
    C = np.array([1.0, h, z])
    return {"K": np.array([[[f, 0, size[0] / 2], [0, f, size[1] / 2], [0, 0, 1.0]]]), "R": R[None], "tvec": (-R @ C)[None], "C": C[None],
            "dist": np.zeros(1)}


def test_plane_hits_lie_on_the_plane():
    arr = _cam()
    px, X, ok = WL.plane_hits(arr, 0, (1920, 1080), -9.0, 3.0, step=40)
    n = np.array([-np.sin(np.radians(3.0)), 0, np.cos(np.radians(3.0))])
    d = X[ok] @ n - (-9.0 * n[2])
    assert np.abs(d).max() < 1e-6
    assert ok.mean() > 0.5


def test_edge_scan_finds_a_floor_edge():
    arr = _cam()
    cd = WL.cam_dict(arr, 0)
    h, w = 1080, 1920
    img = np.full((h, w, 3), 60, np.uint8)
    z_true = -6.5
    # bright pale floor in front of (below) the line z = z_true, dark carpet behind it
    gx, gy = np.meshgrid(np.arange(w), np.arange(h))
    px = np.stack([gx.ravel(), gy.ravel()], 1).astype(float)
    rays = TW.pixel_rays_world(px, arr["K"][0], arr["R"][0], 0.0)
    hit = TW.ray_floor_hit(arr["C"][0], rays)
    bright = (hit[:, 2] > z_true) & np.isfinite(hit[:, 2])
    img.reshape(-1, 3)[bright] = 200
    sc = WL.edge_scan(arr, [0], [img], [np.zeros((h, w), np.uint8)], np.arange(-8.0, -5.0, 0.1), np.arange(-2, 2.1, 1.0), band_px=3)
    z, phi = WL.plateau_peak(sc["score"], sc["zs"], sc["phis"])
    assert abs(z - z_true) < 0.15
    assert abs(phi) <= 1.0


def test_ortho_texture_recovers_a_floor_pattern():
    arr = _cam(pitch_deg=4.0)
    h, w = 1080, 1920
    gx, gy = np.meshgrid(np.arange(w), np.arange(h))
    px = np.stack([gx.ravel(), gy.ravel()], 1).astype(float)
    hit = TW.ray_floor_hit(arr["C"][0], TW.pixel_rays_world(px, arr["K"][0], arr["R"][0], 0.0))
    colour = lambda X, Z: np.stack([100 + 10 * np.floor(X), 50 + 8 * np.floor(Z + 20), np.full_like(X, 120)], -1)    # noqa: E731
    img = np.zeros((h * w, 3), np.float32)
    ok = np.isfinite(hit[:, 0]) & (np.abs(hit[:, 0]) < 40) & (hit[:, 2] > -40)
    img[ok] = colour(hit[ok, 0], hit[ok, 2])
    img = img.reshape(h, w, 3).clip(0, 255).astype(np.uint8)
    X, Z = np.meshgrid(np.linspace(-2.3, 2.3, 24) + 1.0, np.linspace(-4.5, -3.0, 12))
    pts = np.stack([X.ravel(), np.zeros(X.size), Z.ravel()], 1)
    tex, nobs = WL.ortho_texture(pts, X.shape, [{"arr": arr, "i": 0, "img": img, "ignore": np.zeros((h, w), np.uint8)}], min_obs=1)
    assert np.isfinite(tex[:, :, 0]).mean() > 0.9
    # the pattern is constant inside 1 m cells: texels deep inside a cell must match; compare the cell index along X and Z
    bgr = colour(X, Z)
    good = np.isfinite(tex[:, :, 0]) & (np.abs(X - np.round(X)) > 0.2) & (np.abs(Z - np.round(Z)) > 0.2)
    assert good.sum() > 40
    assert np.abs(tex[good] - bgr[good]).max() < 12


def test_grid_mesh_and_obj_roundtrip(tmp_path):
    from dancecap.export_unity import read_obj, read_mtl_textures
    V, UV, Fc = RT.grid_mesh(np.array([0.0, 0.0, 0.0]), np.array([1.0, 0, 0]), np.array([0, 0, 2.0]), 3, 2)
    assert len(V) == 12 and len(Fc) == 12 and UV.min() >= 0 and UV.max() <= 1
    Vw, UVw, Fw = RT.grid_mesh(np.array([0.0, 2.7, -9.0]), np.array([1.0, 0, 0]), np.array([0, -0.9, 0]), 4, 3)
    RT.write_obj(tmp_path / "room_mesh.obj", {"floor": (V, UV, Fc), "room": (Vw, UVw, Fw)}, {"floor": "f.png", "room": "r.png"})
    o = read_obj(tmp_path / "room_mesh.obj")
    assert set(o["groups"]) == {"floor", "room"}
    assert len(o["v"]) == len(V) + len(Vw)
    assert o["groups"]["floor"][0].shape == (12, 3) and o["groups"]["room"][0].shape == (24, 3)
    assert read_mtl_textures(tmp_path / "room_mesh.mtl") == {"floor": "f.png", "room": "r.png"}
    # all floor normals point the same way (a consistent winding)
    P = o["v"]
    f = o["groups"]["floor"][0]
    n = np.cross(P[f[:, 1]] - P[f[:, 0]], P[f[:, 2]] - P[f[:, 0]])
    assert (np.sign(n[:, 1]) == np.sign(n[0, 1])).all()


def test_review_drawing_projects_the_floor_grid():
    arr = _cam()
    cd = WL.cam_dict(arr, 0)
    img = np.zeros((1080, 1920, 3), np.uint8)
    D.draw_floor_grid(img, cd, (-6, 6), (-8, 4), 1.0)
    assert img.any()
    p, z = D.project(np.array([[1.0, 0.0, -9.0]]), **cd)
    assert z[0] > 0 and 0 < p[0, 1] < 1080
