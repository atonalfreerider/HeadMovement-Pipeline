"""cameras.plausible: the doppelganger bands of a static-room map and the localiser's gate (lucas_luiza 2026-10-24)."""
import numpy as np

from dancecap.cameras import plausible as PL
from dancecap.cameras.localize import FramePose, Localizer, LocSettings


def _room(seed=0):
    """A Steadicam '01' orbiting at height ~1.5 (y down: C_y = -1.5) with 5 copies 9.5 below the floor and 2 far outside
    the room, a tripod '02' at one spot, and a room of points (radius ~6, floor and walls)."""
    rng = np.random.default_rng(seed)
    a = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    c01 = np.c_[3 * np.cos(a), -1.5 + 0.1 * rng.standard_normal(40), 3 * np.sin(a)]
    ghosts = np.c_[rng.uniform(-1, 1, 5), np.full(5, 8.0), rng.uniform(-1, 1, 5)]
    far = np.array([[30.0, -1.6, 25.0], [31.0, -1.4, 26.0]])
    c02 = np.tile([[-5.0, -1.2, 0.0]], (30, 1)) + 0.001 * rng.standard_normal((30, 3))
    C = np.r_[c01, ghosts, far, c02]
    cams = np.array(["01"] * 47 + ["02"] * 30)
    U = np.tile([[0.0, -1.0, 0.0]], (len(C), 1)) + 0.05 * rng.standard_normal((len(C), 3))
    th = rng.uniform(0, 2 * np.pi, 3000)
    P = np.c_[6 * np.cos(th) * rng.uniform(0.2, 1.0, 3000), rng.uniform(-4, 0, 3000), 6 * np.sin(th) * rng.uniform(0.2, 1.0, 3000)]
    names = [f"frames/{c}/{i:06d}.jpg" for i, c in enumerate(cams)]
    return names, cams, C, U, P


def test_bands_flag_the_copies_below_the_floor_and_outside_the_room_only():
    names, cams, C, U, P = _room()
    B = PL.bands(cams, C, U, P)
    assert np.allclose(B.up, [0, -1, 0], atol=0.05)
    fl, kept = PL.flag(names, cams, C, B)
    assert kept == {}
    assert sorted(fl) == sorted(names[40:47])  # the 5 below the floor + the 2 far ones; no orbit image, no tripod image
    assert all("height" in fl[n] for n in names[40:45])
    assert all("horizontal" in fl[n] for n in names[45:47])


def test_a_majority_outside_the_band_is_kept_and_reported():
    names, cams, C, U, P = _room()
    keep = np.r_[np.arange(40), np.arange(47, 77)]  # the orbit and the tripod, no copies
    names, cams, C, U = [names[i] for i in keep], cams[keep], C[keep].copy(), U[keep]
    C[:18, 1] += 9.5  # 18 of camera 01's 40 images (45 % > MAX_SHARE) under the floor
    B = PL.bands(cams, C, U, P)
    fl, kept = PL.flag(names, cams, C, B)
    assert not any(n.startswith("frames/01/") for n in fl)
    assert "01" in kept and "01" in B.exempt


def test_a_camera_that_moves_inside_the_room_is_not_a_doppelganger():
    """A tripod carried to another spot for the last 30 % (review of feb7bfe): outside its own band, inside the room."""
    names, cams, C, U, P = _room()
    C = C.copy()
    b = np.flatnonzero(cams == "02")[21:]  # 9 of 30
    C[b] = [-2.0, -1.2, 3.0]  # 4+ units across the floor
    B = PL.bands(cams, C, U, P)
    assert B.cams["02"].xz_tol < 3.0  # outside the band ...
    fl, _ = PL.flag(names, cams, C, B)
    assert not any(names[i] in fl for i in b)  # ... but inside the room: kept
    assert B.why("02", C[b[0]]) is None


def test_a_pose_at_ceiling_height_inside_the_room_is_implausible():
    """Review of 096e020: band AND room on the height axis let poses 3.6-5.2 above the Steadicam's band through as
    'inside the room' (the room's height span reaches the ceiling); the height axis is the band's (with the room's
    floor), not the room's."""
    names, cams, C, U, P = _room()
    B = PL.bands(cams, C, U, P)
    b = B.cams["01"]
    tol = max(b.h_tol, B.box.height_floor())
    assert tol < 3.0
    ceiling = np.array([3.0, -1.5 - 3.0, 0.0])  # 3 units above the orbit, under the room's top (p99 of the points)
    assert B.box.h_lo - B.box.height_floor() < float(ceiling @ B.up) < B.box.h_hi + B.box.height_floor()
    assert "height" in B.why("01", ceiling)
    assert B.why("01", np.array([3.0, -1.5 - 0.5 * tol, 0.0])) is None


def test_model_bands_exempt_the_majority_camera_from_the_localiser(monkeypatch):
    names, cams, C, U, P = _room()
    keep = np.r_[np.arange(40), np.arange(47, 77)]  # no copies: 18 of camera 01's 40 images (45 %) under the floor
    names, cams, C, U = [names[i] for i in keep], cams[keep], C[keep].copy(), U[keep]
    C[:18, 1] += 9.5
    multi = np.ones(len(P), bool)
    monkeypatch.setattr(PL, "model_arrays", lambda rec: (names, cams, C, U, P, multi))
    B = PL.model_bands(object())
    assert "01" not in B.cams and "01" in B.exempt and "02" in B.cams
    assert B.why("01", np.array([0.0, 8.0, 0.0])) is None  # no band: never gated


def test_sub_model_settings_never_gate_and_keep_the_old_cache_key():
    from dancecap.cameras.localize import settings_key, sub_settings
    ls = LocSettings(kf_by_time=True)
    sub = sub_settings(ls, True)
    assert sub.fix_focal and not sub.plausible and sub.kf_by_time
    old = {k: v for k, v in sub.__dict__.items() if k != "plausible"}  # a locsub cache key made before the gate
    assert settings_key(sub) == dict(sorted(old.items()))
    assert settings_key(ls)["plausible"] is True


def test_a_camera_with_few_images_gets_no_band():
    names, cams, C, U, P = _room()
    m = (cams == "02")
    m[np.flatnonzero(m)[PL.MIN_IMAGES - 1:]] = False  # keep MIN_IMAGES - 1 tripod images
    keep = (cams == "01") | m
    B = PL.bands(cams[keep], C[keep], U[keep], P)
    assert "02" not in B.cams and B.why("02", np.array([100.0, 100.0, 100.0])) is None


def test_the_tripod_band_has_the_room_floor():
    names, cams, C, U, P = _room()
    B = PL.bands(cams, C, U, P)
    b = B.cams["02"]
    assert b.h_tol == B.floor and b.xz_tol == B.floor and B.floor > 0.3  # a still camera: MAD ~0 -> the floor decides
    assert B.why("02", np.array([-5.0, -1.2, 0.2])) is None
    assert B.why("02", np.array([-5.0, -1.2, 0.0]) + np.array([0, 0, 3 * B.floor])) is None  # outside its band, in the room
    assert "height" in B.why("02", np.array([-5.0, 9.0, 0.0]))
    assert "room's centre" in B.why("02", np.array([40.0, -1.2, 0.0]))


def test_localizer_gate_rejects_an_implausible_pose_and_keeps_a_plausible_one():
    names, cams, C, U, P = _room()
    L = Localizer.__new__(Localizer)
    L.s, L.bands = LocSettings(), PL.bands(cams, C, U, P)

    class Q:
        cam = "01"

    def pose(c):
        R = np.eye(3)
        return FramePose(ok=True, R=R, t=-R @ np.asarray(c, float), f=1000.0, n_inliers=200, reproj_px=1.2, source="loc")

    good = L.gate(Q, pose([3.0, -1.5, 0.0]))
    assert good.ok and good.source == "loc"
    bad = L.gate(Q, pose([0.0, 8.0, 0.0]))
    assert not bad.ok and bad.source == "implausible" and bad.n_inliers == 200
    L.bands = None  # LocSettings(plausible=False)
    assert L.gate(Q, pose([0.0, 8.0, 0.0])).ok


def test_check_and_clean_a_pycolmap_model_with_a_copy():
    import pytest
    pycolmap = pytest.importorskip("pycolmap")
    names, cams, C, U, P = _room()
    rec = pycolmap.Reconstruction()
    cam = pycolmap.Camera(model="SIMPLE_PINHOLE", width=640, height=480, params=[500.0, 320.0, 240.0], camera_id=1)
    rec.add_camera_with_trivial_rig(cam)
    pids = [rec.add_point3D(p, pycolmap.Track(), np.zeros(3, np.uint8)) for p in P[:200]]
    for i, (n, c) in enumerate(zip(names, C)):
        R = np.eye(3)
        im = pycolmap.Image(name=n, camera_id=1, image_id=i + 1)
        im.points2D = [pycolmap.Point2D(np.array([10.0 + k, 10.0])) for k in range(2)]
        rec.add_image_with_trivial_frame(im, pycolmap.Rigid3d(pycolmap.Rotation3d(R), -R @ c))
        for k in range(2):  # every point seen by two images: the copies' own points (i 40..46) die with them
            rec.add_observation(pids[(2 * i + k) % len(pids)], pycolmap.TrackElement(i + 1, k))
    doc = PL.check_model(rec)
    assert sorted(doc["flagged"]) == sorted(names[40:47]) and doc["per_cam"] == {"01": 7}
    out = PL.clean_model(rec)
    assert out["removed"]["images"] == 7 and out["passes"] == 2  # pass 2 finds nothing more
    assert sum(im.has_pose for im in rec.images.values()) == len(names) - 7
    assert PL.check_model(rec)["flagged"] == {}
