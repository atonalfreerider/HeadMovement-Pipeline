"""Camera-stage settings for the two-view take (lucas_luiza runbook step 5, C1-C5): [cameras] map overrides through ONE
helper used by every builder of a take's map-image list (and map.json's own list over later edits), loc-part's
references of every camera, one COLMAP camera per no-zoom video, per-camera-pair link counts in the map stats, the
world X camera, the world's dancers on a two-view sync, the two-camera G2a gates, no crash with a main model of < 2
cameras (and no skip of a larger model's subset run), adopted tracks, readers of the [cameras] table that skip x_cam /
map, the SAM host-RAM estimate of landscape QA frames. Synthetic data only.

    ../.venv/bin/python -m pytest tests/test_cameras_twoview_settings.py -q
"""
from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dancecap import camera_types as ct  # noqa: E402
from dancecap import front_pod as fp  # noqa: E402
from dancecap import jl_lowconf as lc  # noqa: E402
from dancecap.cameras import pipeline as P  # noqa: E402
from dancecap.cameras.features import ImageRec, list_images  # noqa: E402
from dancecap.cameras.mapping import (MapSettings, build_database, intrinsics_modes, map_settings,  # noqa: E402
                                      prior_focal_px, select_map_images)
from dancecap.cameras.report import gates  # noqa: E402
from dancecap.take import PACKAGE_ROOT, Take  # noqa: E402

# the [cameras] table of the runbook's step 6 (before [cameras."01"])
TWO_VIEW = """
[cameras]
x_cam = "02"
map = { win_step_s = 2.0, dense_step_s = 6.0, shared_intrinsics = ["01"] }
loc = { n_cross = 4 }

[cameras."01"]
file = "Video-44260.mp4"
mount = "handheld"
zoom = false
source = "user"
set_at = "2026-10-23T10:10:00Z"

[cameras."02"]
file = "tripod.mp4"
mount = "tripod"
zoom = true
pans = true
source = "user"
set_at = "2026-10-23T16:00:00Z"
zoom_events = [[125.5, 126.5, 1.16]]
"""

_T0 = SimpleNamespace(rotation=SimpleNamespace(matrix=lambda: np.eye(3)), translation=np.zeros(3))  # an identity cam_from_world


class _Stop(Exception):
    pass


def _recs(cams=("01", "02"), t0=16.8, t1=213.5):
    """Window frames at 29 fps (01, the Steadicam) and 29.97 fps (02, every 2nd frame of 59.94) + keyframes every 2 s."""
    out = {}
    for cam, fps in zip(cams, (29.0, 59.94 / 2)):
        n = int((t1 - t0) * fps)
        win = [ImageRec(cam, "win", i, t0 + i / fps, f"frames/{cam}/{i:06d}.jpg", "") for i in range(n)]
        kf = [ImageRec(cam, "kf", j, t0 + 2.0 * j, f"keyframes/{cam}/{j:07d}.jpg", "") for j in range(int((t1 - t0) / 2) + 1)]
        out[cam] = sorted(win + kf, key=lambda r: (r.t, r.kind))
    return out


# ------------------------------------------------------------------------------------------------ C1 map settings


def test_map_settings_merges_the_toml_map_table_over_the_defaults():
    s = map_settings(tomllib.loads(TWO_VIEW))
    d = MapSettings()
    assert (s.win_step_s, s.dense_step_s, s.shared_intrinsics) == (2.0, 6.0, ("01",))
    assert isinstance(s.shared_intrinsics, tuple)  # TOML arrays are lists
    assert s.focal_prior_px == {} and s.focal_prior == d.focal_prior and s.link_min_inliers == d.link_min_inliers
    s2 = map_settings({"cameras": {"map": {"focal_prior_px": {"01": 1100}, "link_min_pairs": 2}}})
    assert s2.focal_prior_px == {"01": 1100.0} and isinstance(s2.focal_prior_px["01"], float) and s2.link_min_pairs == 2
    assert map_settings({"cameras": {"map": {"shared_intrinsics": "01"}}}).shared_intrinsics == ("01",)
    with pytest.raises(ValueError, match="win_stp_s"):
        map_settings({"cameras": {"map": {"win_stp_s": 2.0}}})


def test_map_settings_refuses_a_zooming_shared_camera():
    cfg = tomllib.loads(TWO_VIEW)
    cfg["cameras"]["map"]["shared_intrinsics"] = ["01", "02"]  # 02 is declared zoom = true
    with pytest.raises(ValueError, match=r"\['02'\] declared zoom = true"):
        map_settings(cfg)
    cfg["cameras"]["map"]["shared_intrinsics"] = ["01", "03"]  # no block: nothing declared
    assert map_settings(cfg).shared_intrinsics == ("01", "03")


def test_map_settings_defaults_without_a_map_table():
    assert map_settings(None) == map_settings({}) == map_settings({"cameras": {}}) == MapSettings()
    assert map_settings({"cameras": {"loc": {"kf_by_time": True}, "01": {"mount": "handheld"}}}) == MapSettings()
    seen = 0
    for p in sorted((PACKAGE_ROOT / "takes").glob("*.toml")):  # every take without [cameras] map keeps today's map
        cfg = tomllib.loads(p.read_text(encoding="utf-8"))
        if "map" not in (cfg.get("cameras") or {}):
            assert map_settings(cfg) == MapSettings(), p.name
            seen += 1
    assert seen > 0


def test_map_settings_new_fields_are_per_instance():
    a, b = MapSettings(), MapSettings()
    a.focal_prior_px["01"] = 900.0
    assert b.focal_prior_px == {} and MapSettings().shared_intrinsics == ()
    assert map_settings({}).focal_prior_px is not MapSettings().focal_prior_px


def test_two_view_map_table_cuts_the_map_images():
    recs = _recs()
    default, _ = select_map_images(recs, MapSettings())
    imgs, dense = select_map_images(recs, map_settings(tomllib.loads(TWO_VIEW)))
    for cam in ("01", "02"):
        w = [r for r in imgs if r.cam == cam and r.kind == "win"]
        assert 97 <= len(w) <= 100  # one window image per 2 s
        assert len([r for r in default if r.cam == cam and r.kind == "win"]) > 2.5 * len(w)
    assert {r.key for r in imgs if r.kind == "kf"} <= dense  # keyframes always dense
    assert 30 <= len([k for k in dense if "/win_" in k and k.startswith("01/")]) <= 34  # every 6 s


def _fake_take(tmp_path, cfg, cams=("01",)):
    work = tmp_path / "tt"
    for cam in cams:
        (work / "frames" / cam).mkdir(parents=True, exist_ok=True)
        (work / "frames" / cam / "frames.json").write_text(json.dumps({
            "width": 720, "height": 1280, "frames": [{"file": f"{i:06d}.jpg", "t": 16.8 + i / 29} for i in range(120)]}))
    return SimpleNamespace(name="tt", work=work, cfg=cfg, sources=[SimpleNamespace(id=c) for c in cams])


def _capture_map_images(monkeypatch, got):
    def fake(work, recs, ms):
        got.append(ms)
        raise _Stop
    monkeypatch.setattr(P, "map_images", fake)
    monkeypatch.setattr(P, "load_model", lambda work: SimpleNamespace(images={}, points3D={}))


def test_loc_part_and_map_extend_use_the_takes_map_settings(tmp_path, monkeypatch):
    """front_pod loc-part and map_extend list the map images with the map's own settings: the Localizer keeps only
    listed images as references, so the default 2/3 s list would drop ~90 % of a 29 fps camera's 2 s map images."""
    from dancecap import map_extend
    cfg = tomllib.loads(TWO_VIEW)
    take = _fake_take(tmp_path, cfg)
    monkeypatch.setattr(Take, "load", classmethod(lambda cls, name: take))
    got: list = []
    _capture_map_images(monkeypatch, got)
    with pytest.raises(_Stop):
        fp.loc_part("tt", "01", 0, 10)
    with pytest.raises(_Stop):
        map_extend.extend("tt", "01")
    assert got == [map_settings(cfg)] * 2 and got[0].win_step_s == 2.0


def test_loc_part_references_every_camera_of_a_take_with_a_map_table(tmp_path, monkeypatch):
    """loc-part localises against the map images of every camera (stage_loc's references: Localizer.candidates falls
    back to the other cameras' nearest images) for a take with a [cameras] map table; other takes keep their own
    camera's only (as they were localised)."""
    cfg = tomllib.loads(TWO_VIEW)
    take = _fake_take(tmp_path, cfg, cams=("01", "02"))
    monkeypatch.setattr(Take, "load", classmethod(lambda cls, name: take))
    got: list = []

    def fake(work, recs, ms):
        got.append(sorted(recs))
        raise _Stop
    monkeypatch.setattr(P, "map_images", fake)
    monkeypatch.setattr(P, "load_model", lambda work: SimpleNamespace(images={}, points3D={}))
    for c in ("01", "02"):
        with pytest.raises(_Stop):
            fp.loc_part("tt", c, 0, 10)
    assert got == [["01", "02"], ["01", "02"]]
    assert fp.loc_ref_cams(take, "02") == ["01", "02"]
    take.cfg = {k: v for k, v in cfg.items()}
    take.cfg["cameras"] = {k: v for k, v in cfg["cameras"].items() if k != "map"}  # an earlier take: unchanged
    with pytest.raises(_Stop):
        fp.loc_part("tt", "02", 0, 10)
    assert got[-1] == ["02"] and fp.loc_ref_cams(take, "01") == ["01"]


def test_loc_part_localiser_gets_the_other_cameras_references(tmp_path, monkeypatch):
    """The real map_images + Localizer of loc-part on a synthetic two-camera model: 02's map images are references
    (cams_reg) of 01's localisation, and a frame's cross-camera candidates are 02's nearest image."""
    from dancecap.cameras import localize
    cfg = tomllib.loads(TWO_VIEW)
    take = _fake_take(tmp_path, cfg, cams=("01", "02"))
    monkeypatch.setattr(Take, "load", classmethod(lambda cls, name: take))
    take.camera_types = lambda cams: {}
    recs = list_images(take.work, ["01", "02"])
    imgs, _ = select_map_images(recs, map_settings(cfg))
    model = SimpleNamespace(images={k: SimpleNamespace(has_pose=True, name=r.name, points2D=[], camera=SimpleNamespace(params=[1000.0, 0, 0, 0.0]),
                                                       cam_from_world=_T0) for k, r in enumerate(imgs)}, points3D={})
    monkeypatch.setattr(P, "load_model", lambda work: model)
    got = {}

    def fake_localize(L, frames, w, h):
        got["L"], got["frames"] = L, frames
        raise _Stop
    monkeypatch.setattr(localize, "localize_camera", fake_localize)
    with pytest.raises(_Stop):
        fp.loc_part("tt", "01", 0, 10)
    L = got["L"]
    assert sorted(L.cams_reg) == ["01", "02"] and {r.cam for r in got["frames"]} == {"01"} and len(got["frames"]) == 10
    q = got["frames"][5]
    cand = L.candidates(q, cross=True)
    assert {r.rec.cam for r in cand} == {"01", "02"}


def test_map_images_follow_the_map_json_list(tmp_path, capsys):
    """[cameras] map edited after the map was built (runbook step 10's win_step_s 3.0 before a map re-run that did
    not finish): loc / attach / loc-part keep the map's own images (sfm_work/map.json) with a WARNING; unchanged
    settings give exactly select_map_images."""
    work = tmp_path / "w"
    (work / "cameras" / "sfm_work").mkdir(parents=True)
    recs = _recs()
    built = map_settings(tomllib.loads(TWO_VIEW))
    sel, dense = select_map_images(recs, built)
    with pytest.raises(RuntimeError, match="records no map_images"):
        P.map_images(work, recs, None)
    assert P.map_images(work, recs, MapSettings()) == select_map_images(recs, MapSettings())  # no map yet
    (work / "cameras" / "sfm_work" / "map.json").write_text(json.dumps({"images": len(sel), "map_images": [r.key for r in sel]}))
    assert P.map_images(work, recs, built) == (sel, dense) and "WARNING" not in capsys.readouterr().out
    imgs, d2 = P.map_images(work, recs, MapSettings())  # the defaults would select ~3x the window images
    assert imgs == sel and d2 <= {r.key for r in sel} and "WARNING map images" in capsys.readouterr().out
    one = {"01": recs["01"]}
    assert P.map_images(work, one, MapSettings())[0] == [r for r in sel if r.cam == "01"]
    assert P.map_images(work, recs, None)[0] == sel  # no settings: the recorded list (cameras/reuse.py presolve_sub)
    (work / "cameras" / "sfm_work" / "map.json").write_text(json.dumps({"images": 1}))  # a map.json without the list
    assert P.map_images(work, recs, MapSettings()) == select_map_images(recs, MapSettings())


def test_attach_zoom_reattach_uses_the_takes_map_settings(tmp_path, monkeypatch):
    from dancecap.cameras import attach_zoom
    cfg = tomllib.loads(TWO_VIEW)
    take = _fake_take(tmp_path, cfg, cams=("01", "02"))
    for cam in ("01", "02"):
        (take.work / "cameras").mkdir(parents=True, exist_ok=True)
        np.savez(take.work / "cameras" / f"{cam}.npz", t=np.zeros(1))
    monkeypatch.setattr(Take, "load", classmethod(lambda cls, name: take))
    monkeypatch.setattr(ct, "types_for_work", lambda work, policy=None: {})
    model = SimpleNamespace(images={1: SimpleNamespace(has_pose=True, name="keyframes/01/0.jpg")}, points3D={})
    monkeypatch.setattr(P, "load_model", lambda work: model)
    monkeypatch.setattr(P, "LOOSE_FILE_KEYS", P.LOOSE_FILE_KEYS)  # reattach sets it for the process: restored after
    got = []

    def fake_attach(work, recs, F, ls, ms, *a, **k):
        got.append(ms)
        raise _Stop
    monkeypatch.setattr(P, "stage_attach", fake_attach)
    with pytest.raises(_Stop):
        attach_zoom.reattach(take.work, ["02"], log=lambda *a: None)
    assert got == [map_settings(cfg)]


# ------------------------------------------------------------------------------------------------ C2 intrinsics


def test_intrinsics_modes_per_camera():
    s = map_settings(tomllib.loads(TWO_VIEW))
    assert intrinsics_modes({"02", "01"}, s) == {"01": "shared", "02": "per_image"}
    assert intrinsics_modes({"01", "02"}, MapSettings()) == {"01": "per_image", "02": "per_image"}
    assert intrinsics_modes({"01"}, s, single_camera=True) == {"01": "single"}
    assert prior_focal_px("01", 720, 1280, MapSettings()) == pytest.approx(1.2 * 1280)
    assert prior_focal_px("01", 720, 1280, MapSettings(focal_prior_px={"01": 1000.0})) == 1000.0
    assert prior_focal_px("02", 1280, 720, MapSettings(focal_prior_px={"01": 1000.0})) == pytest.approx(1536.0)


def _look_at(C, target, up=np.array([0, 1.0, 0])):
    z = target - C
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return R, -R @ C


def _two_view_db(tmp_path, s, sizes=None):
    """Tiny images of 2 cameras (01: 2 frames + 1 keyframe, 02: 2 frames) with synthetic correspondences of one rigid
    scene: 200 points seen by every image -> build_database stats."""
    work = tmp_path / "w"
    sizes = sizes or {}
    names = [("01", "frames", 0), ("01", "frames", 1), ("01", "keyframes", 0), ("02", "frames", 0), ("02", "frames", 1)]
    images = []
    for k, (cam, kind, i) in enumerate(names):
        w, h = sizes.get(k, (240, 320) if cam == "01" else (320, 240))
        d = work / kind / cam
        d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / f"{i:06d}.jpg"), np.full((h, w, 3), 90 + 20 * k, np.uint8))
        images.append(ImageRec(cam, "win" if kind == "frames" else "kf", i, float(k), f"{kind}/{cam}/{i:06d}.jpg", ""))
    rng = np.random.default_rng(3)
    X = rng.uniform([-1.5, -1.0, 5.0], [1.5, 1.0, 8.0], (200, 3))
    kp = []
    for k, r in enumerate(images):
        w, h = sizes.get(k, (240, 320) if r.cam == "01" else (320, 240))
        R, t = _look_at(np.array([-0.6 + 0.3 * k, 0.1 * k, 0.0]), np.array([0.0, 0.0, 6.5]))
        Xc = X @ R.T + t
        f = 1.2 * max(w, h)
        kp.append(np.stack([f * Xc[:, 0] / Xc[:, 2] + w / 2 - 0.5, f * Xc[:, 1] / Xc[:, 2] + h / 2 - 0.5], 1).astype(np.float32))
    F = SimpleNamespace(get=lambda r: {"kpts": kp[images.index(r)]})
    m = np.stack([np.arange(200), np.arange(200)], 1)
    matches = {(0, 3): m, (1, 4): m[:100], (0, 1): m, (3, 4): m}
    return build_database(work, F, images, matches, work / "db" / "t.db", s), work / "db" / "t.db", images


def test_build_database_shares_one_camera_per_listed_video(tmp_path):
    import pycolmap
    s = MapSettings(shared_intrinsics=("01",), focal_prior_px={"01": 500.0})
    st, db_path, images = _two_view_db(tmp_path, s)
    assert st["intrinsics"]["01"] == {"mode": "shared", "cameras": 1, "focal_prior_px": 500.0}
    assert st["intrinsics"]["02"] == {"mode": "per_image", "cameras": 2, "focal_prior_px": 384.0}
    db = pycolmap.Database.open(db_path)
    by = {im.name: im for im in db.read_all_images()}
    c01 = {by[r.name].camera_id for r in images if r.cam == "01"}
    c02 = [by[r.name].camera_id for r in images if r.cam == "02"]
    assert len(c01) == 1 and len(set(c02)) == 2 and not c01 & set(c02)  # frames AND keyframes of 01 on one camera
    cam = db.read_camera(next(iter(c01)))
    assert cam.model.name == "SIMPLE_RADIAL" and cam.has_prior_focal_length and cam.params[0] == pytest.approx(500.0)
    assert db.read_camera(c02[0]).params[0] == pytest.approx(1.2 * 320)
    db.close()
    # link counts per camera pair: (0,3) 200 and (1,4) 100 correspondences between 01 and 02
    e = st["per_campair"]["01-02"]
    assert e["pairs"] == 2 and e["link_pairs"] == 1 and e["max_inliers"] >= 150 and st["link_min_inliers"] == 150
    assert st["per_campair"]["01-01"]["link_pairs"] == 1


def test_build_database_default_is_one_camera_per_image(tmp_path):
    st, db_path, images = _two_view_db(tmp_path, MapSettings())
    assert {c: e["mode"] for c, e in st["intrinsics"].items()} == {"01": "per_image", "02": "per_image"}
    assert st["intrinsics"]["01"]["cameras"] == 3 and st["intrinsics"]["01"]["focal_prior_px"] == 384.0
    import pycolmap
    db = pycolmap.Database.open(db_path)
    assert db.num_cameras() == len(images) == db.num_images()
    db.close()


def test_build_database_shared_video_with_two_image_sizes(tmp_path):
    """COLMAP's single-camera import skips an image of another size: it gets the video's second camera."""
    st, _, _ = _two_view_db(tmp_path, MapSettings(shared_intrinsics=("01",)), sizes={2: (360, 480)})
    assert st["intrinsics"]["01"]["mode"] == "shared" and st["intrinsics"]["01"]["cameras"] == 2


def test_focal_prior_px_is_scaled_to_a_second_image_size(tmp_path):
    """focal_prior_px is the focal at the video's largest image size: a smaller image's camera gets it scaled."""
    import pycolmap
    s = MapSettings(shared_intrinsics=("01",), focal_prior_px={"01": 600.0})
    st, db_path, images = _two_view_db(tmp_path, s, sizes={2: (360, 480)})  # 01's keyframe is 360x480, frames 240x320
    db = pycolmap.Database.open(db_path)
    by = {im.name: im for im in db.read_all_images()}
    f = {r.kind: db.read_camera(by[r.name].camera_id).params[0] for r in images if r.cam == "01"}
    db.close()
    assert f["kf"] == pytest.approx(600.0) and f["win"] == pytest.approx(600.0 * 320 / 480)
    assert st["intrinsics"]["01"]["focal_prior_px"] == pytest.approx(400.0)  # its first image's camera
    assert st["link_min_pairs"] == 3 and st["link_min_inliers"] == 150
    assert prior_focal_px("01", 240, 320, s, 480) == pytest.approx(400.0) and prior_focal_px("01", 240, 320, s) == 600.0


class _Rec:
    """A pycolmap.Reconstruction stand-in for stage_map: registered images of the given cameras (at the origin: the
    doppelganger check of cameras.plausible finds nothing to drop)."""

    def __init__(self, cams):
        self.images = {k: SimpleNamespace(has_pose=True, name=f"keyframes/{c}/{k:07d}.jpg", cam_from_world=_T0) for k, c in enumerate(cams)}
        self.points3D = {}

    def num_reg_images(self):
        return len(self.images)

    def num_points3D(self):
        return 0

    def compute_mean_reprojection_error(self):
        return 0.8

    def write_binary(self, p):
        Path(p, "images.bin").write_bytes(b"")


def _stage_map(tmp_path, monkeypatch, groups, cfg=TWO_VIEW):
    import contextlib

    from dancecap.cameras.mapping import intrinsics_modes as modes_of
    work = tmp_path / "w"
    (work / "cameras" / "sfm_work").mkdir(parents=True)
    calls = []

    def fake_db(work, F, images, matches, db_path, s, cams=None, single_camera=False):
        db_path.write_bytes(b"")
        calls.append((sorted(cams) if cams else None, single_camera))
        md = modes_of({r.cam for r in images if cams is None or r.cam in cams}, s, single_camera)
        return {"pairs": 9, "raw_ok": 9, "verified": 9, "pair_inliers": {(0, 1): 200},
                "per_campair": {"01-02": {"pairs": 4, "max_inliers": 400, "sum_inliers": 900, "link_pairs": 3}},
                "intrinsics": {c: {"mode": m, "cameras": 1, "focal_prior_px": 1536.0} for c, m in md.items()},
                "link_min_inliers": s.link_min_inliers, "link_min_pairs": s.link_min_pairs}
    monkeypatch.setattr(P.gpu, "exclusive", lambda label: contextlib.nullcontext())
    monkeypatch.setattr(P, "match_pairs", lambda F, images, pairs, cache: {})
    monkeypatch.setattr(P, "targeted_pairs", lambda images, matches, ms: ([], []))
    monkeypatch.setattr(P, "build_database", fake_db)
    monkeypatch.setattr(P, "link_components", lambda images, pair_inliers, ms: groups)
    monkeypatch.setattr(P, "run_mapper", lambda work, db, out, ms: (_Rec(sorted(groups[0]) * 2), {0: _Rec(sorted(groups[0]) * 2)}))
    monkeypatch.setattr(P, "run_sub_mapper", lambda work, db, out, ms: _Rec([n for g in groups[1:] for n in sorted(g)
                                                                             if out.name == "sub_" + "_".join(sorted(g))]))
    monkeypatch.setattr(P, "refine_model", lambda rec, px: {"mean_reproj_after": 0.8})
    recs = {c: [r for r in rs if r.t < 40] for c, rs in _recs().items()}
    info = P.stage_map(work, recs, SimpleNamespace(get=lambda r: None), map_settings(tomllib.loads(cfg)))
    return info, json.loads((work / "cameras" / "sfm_work" / "map.json").read_text()), calls


def test_stage_map_writes_the_intrinsics_and_link_counts(tmp_path, monkeypatch):
    """The map.json keys cameras.twoview map-gate reads: intrinsics_mode per camera, verification.per_campair link_pairs,
    link_min_inliers, link_min_pairs (and no pair_inliers)."""
    info, mj, calls = _stage_map(tmp_path, monkeypatch, [{"01", "02"}])
    assert mj["intrinsics_mode"] == {"01": "shared", "02": "per_image"}
    v = mj["verification"]
    assert v["per_campair"]["01-02"]["link_pairs"] == 3 and (v["link_min_inliers"], v["link_min_pairs"]) == (150, 3)
    assert "pair_inliers" not in v and mj["main_cams"] == ["01", "02"] and calls == [(None, False)]
    assert mj["map_images"] == [r.key for r in P.map_images(tmp_path / "w", {c: [r for r in rs if r.t < 40]
                                                                             for c, rs in _recs().items()}, None)[0]]


def test_stage_map_intrinsics_mode_of_a_sub_model_camera(tmp_path, monkeypatch):
    """A camera mapped in a one-camera sub-model (single_camera) is recorded as 'single', not database_all's mode."""
    info, mj, calls = _stage_map(tmp_path, monkeypatch, [{"02"}, {"01"}])
    assert calls == [(None, False), (["02"], False), (["01"], True)]
    assert mj["intrinsics_mode"] == {"01": "single", "02": "per_image"}
    assert mj["sub_models"]["01"]["intrinsics_mode"] == {"01": "single"}


# ------------------------------------------------------------------------------------------------ C3 world X axis


def test_world_x_cam():
    cfg = tomllib.loads(TWO_VIEW)
    types = {"01": ct.StageType("01", "handheld", False, source="user"), "02": ct.StageType("02", "tripod", True, True, "user")}
    assert P.world_x_cam(cfg, ["01", "02"], types)[0] == "02"  # the TOML key
    assert P.world_x_cam({}, ["01", "02"], types) == ("02", "first main-model tripod (tripod_zoom)")
    assert P.world_x_cam({}, ["01", "02"], {}) == ("01", "first main camera")  # today's rule
    assert P.world_x_cam({}, ["01", "06"], None)[0] == "06"
    assert P.world_x_cam({}, ["01", "03", "06"], {"03": ct.StageType("03", "tripod", False)})[0] == "03"
    cam, why = P.world_x_cam({"cameras": {"x_cam": "05"}}, ["01", "02"], types)  # not a main-model camera
    assert cam == "02" and "x_cam '05' is not a main-model camera" in why


# ------------------------------------------------------------------------------------------------ C4 gates


def _cam(main=True):
    return {"in_main_model": main, "attached": False, "hd": True, "map_reproj_px": 0.9, "frames": 100, "frames_ok": 99,
            "focal": {"plausible": True}, "maxdim": 1280}


def test_two_camera_gates():
    rep = {"cams": {"01": _cam(), "02": _cam()}, "dancer_loo_check": {}}
    g = gates(rep, 2)
    assert g["cameras_registered"]["passed"] and g["cameras_registered"]["accept"] == 2
    dc = g["dancer_check"]
    assert dc["passed"] and dc["not_measured"] == ["01", "02"] and dc["value"] == {}
    rep["cams"]["02"] = _cam(main=False)
    g = gates(rep, 2)
    assert not g["cameras_registered"]["passed"] and g["dancer_check"]["not_measured"] == ["01"]


def test_two_camera_gates_still_need_every_other_camera_checked():
    """No LOO at all (2 main cameras): an unregistered third camera fails dancer_check, as it does when another
    camera produced an entry."""
    cams = {"01": _cam(), "02": _cam(), "03": {**_cam(main=False), "frames_ok": 0}}
    g = gates({"cams": cams, "dancer_loo_check": {}}, 3)
    assert g["dancer_check"] == {**g["dancer_check"], "value": {}, "passed": False, "not_measured": ["01", "02"]}
    cams["04"] = {**_cam(main=False), "attached": True}
    g = gates({"cams": cams, "dancer_loo_check": {"04": {"circular": True, "median_px": 5.0, "holdout": {"median_px": 7.0}}}}, 4)
    assert not g["dancer_check"]["passed"]


def test_cameras_registered_accept_counts_the_take_not_the_run():
    """A --cams subset qa of a 6-camera take still needs 5 main cameras; a two-camera take needs both."""
    cams = {"01": _cam(), "02": _cam()}
    assert not gates({"cams": cams}, 2, n_take=6)["cameras_registered"]["passed"]
    assert gates({"cams": cams}, 2, n_take=6)["cameras_registered"]["accept"] == 5
    assert gates({"cams": cams}, 2, n_take=2)["cameras_registered"]["passed"]
    assert gates({"cams": {"01": _cam()}}, 1, n_take=2)["cameras_registered"]["accept"] == 2


def test_build_report_flags_an_adopted_track(tmp_path, monkeypatch):
    """report.json cams.<cam>.adopted names the localisation an adopted cameras/<cam>.npz replaced (its statistics
    describe that localisation), and dancer_px is never written into the adopted file."""
    work = _work_with_frames(tmp_path)
    np.savez(work / "cameras" / "02.npz", t=np.zeros(3))
    np.savez(work / "cameras" / "01.npz", t=np.zeros(3))
    (work / "cameras" / "orig").mkdir()
    np.savez(work / "cameras" / "orig" / "02_loc.npz", t=np.zeros(3))
    monkeypatch.setattr(P, "_model_errors", lambda model: {"01": [0.5], "02": [0.6]})
    recs = {"01": [], "02": []}
    rep = P.build_report(work, ["01", "02"], recs, _model("01", "02"), {}, np.zeros((0, 3)), np.zeros((0, 3)), None, {},
                         0.04, 4.0, n_take=2)
    assert rep["cams"]["02"]["adopted"] == "cameras/orig/02_loc.npz" and "adopted" not in rep["cams"]["01"]
    assert rep["gates_G2a"]["cameras_registered"]["accept"] == 2
    pf = np.arange(3, dtype=float)
    assert not P.store_dancer_px(work, "02", pf) and P.store_dancer_px(work, "01", pf)
    with np.load(work / "cameras" / "02.npz") as z:
        assert "dancer_px" not in z.files
    with np.load(work / "cameras" / "01.npz") as z:
        np.testing.assert_array_equal(z["dancer_px"], pf.astype(np.float32))
    assert not P.store_dancer_px(work, "01", None) and not P.store_dancer_px(work, "03", pf)


def test_five_or_more_camera_gates_unchanged():
    cams = {f"0{i}": _cam() for i in range(1, 7)}
    loo = {c: {"median_px": 6.0, "median_rel_height": 0.005, "circular": False} for c in cams}
    g = gates({"cams": cams, "dancer_loo_check": loo}, 6)
    r = g["cameras_registered"]
    assert (r["accept"], r["passed"], r["note"]) == (5, True, "passes on the main static model alone (>= 5 cameras)")
    assert g["dancer_check"]["passed"] and "not_measured" not in g["dancer_check"]
    cams["06"] = _cam(main=False)
    cams["05"] = _cam(main=False)
    g = gates({"cams": cams, "dancer_loo_check": {}}, 6)
    assert not g["cameras_registered"]["passed"] and "dancer_check" not in g  # 4 main: as before
    del loo["06"]
    g = gates({"cams": {f"0{i}": _cam() for i in range(1, 7)}, "dancer_loo_check": loo}, 6)
    assert not g["dancer_check"]["passed"]  # a main camera without its LOO still fails with >= 3 main cameras


# ------------------------------------------------------------------------------------------------ C5 < 2 main cameras


def _work_with_frames(tmp_path, cams=("01", "02")):
    work = tmp_path / "tt"
    for cam in cams:
        (work / "frames" / cam).mkdir(parents=True)
        (work / "frames" / cam / "frames.json").write_text(json.dumps({"width": 720, "height": 1280, "frames": []}))
    (work / "cameras" / "cache").mkdir(parents=True)
    return work


def _model(*cams):
    """A main model whose registered images are one keyframe per camera (model_cams reads the image names)."""
    return SimpleNamespace(images={k: SimpleNamespace(has_pose=True, name=f"keyframes/{c}/0.jpg") for k, c in enumerate(cams)},
                           points3D={}, compute_mean_reprojection_error=lambda: 0.5)


def _loc(work, cam, n=12):
    from dancecap.cameras.localize import FramePose
    frames = [ImageRec(cam, "win", i, 16.8 + i / 30, f"frames/{cam}/{i:06d}.jpg", "") for i in range(n)]
    poses = [FramePose(True, np.eye(3), np.array([0.01 * i, 0, 0]), 1100.0, 0.0, 200, 0.6, "loc") for i in range(n)]
    P._save_raw(work / "cameras" / "cache" / f"loc_{cam}.npz", frames, poses)


STATURES = ({"lead": 1.75, "follow": 1.63}, "user")


def test_attach_with_one_main_camera_reports_instead_of_crashing(tmp_path, monkeypatch):
    work = _work_with_frames(tmp_path)
    _loc(work, "01")  # the one main-model camera is localised
    (work / "cameras" / "cache" / "loc_02.npz").write_bytes(b"stale")
    monkeypatch.setattr(P, "load_model", lambda w: _model("01"))
    info = P.stage_attach(work, {"01": [], "02": []}, None, None, MapSettings(), ["01", "02"], 0.04, 4.0, STATURES, types={})
    assert info["02"]["attached"] is False and "the main model holds only 1 camera(s) ['01']" in info["02"]["reason"]
    assert not (work / "cameras" / "cache" / "loc_02.npz").exists()  # no placement in this model, as 'no sub-model'
    assert (work / "cameras" / "cache" / "loc_01.npz").exists()


def test_attach_subset_of_a_larger_model_raises_and_keeps_every_file(tmp_path, monkeypatch):
    """`cameras <take> --stages attach --cams 05` of a 3-camera model: no main camera's localisation in the run is an
    error (as the crash in main_vertical was), never a skip that deletes 05's attached tracks."""
    work = _work_with_frames(tmp_path, cams=("01", "02", "05"))
    for c in ("01", "02", "05"):
        _loc(work, c)
    np.savez(work / "cameras" / "05.npz", t=np.zeros(12))
    monkeypatch.setattr(P, "load_model", lambda w: _model("01", "02"))
    with pytest.raises(RuntimeError, match=r"main model holds \['01', '02'\], this run has \[\]"):
        P.stage_attach(work, {"05": []}, None, None, MapSettings(), ["05"], 0.04, 4.0, STATURES, types={})
    (work / "cameras" / "cache" / "loc_01.npz").unlink()  # a failed loc-join upstream of `--stages attach,qa`
    with pytest.raises(RuntimeError, match=r"this run has \['02'\]"):
        P.stage_attach(work, {c: [] for c in ("01", "02", "05")}, None, None, MapSettings(), ["01", "02", "05"], 0.04, 4.0,
                       STATURES, types={})
    assert (work / "cameras" / "cache" / "loc_05.npz").exists() and (work / "cameras" / "05.npz").exists()


def _run_take(tmp_path, monkeypatch, cfg=None, model=("01",), tracks=("01",), cams=("01", "02")):
    work = tmp_path / "tt" if (tmp_path / "tt").exists() else _work_with_frames(tmp_path, cams)
    take = SimpleNamespace(name="tt", work=work, cfg=cfg or {}, sources=[SimpleNamespace(id=c) for c in cams])
    monkeypatch.setattr(P, "load_model", lambda w: _model(*model))
    monkeypatch.setattr(P, "camera_types_for", lambda work, types=None, policy=None: {})
    trs = {c: {"t": 16.8 + np.arange(30) / 30, "R": np.tile(np.eye(3), (30, 1, 1)), "ok": np.ones(30, bool)} for c in tracks}
    monkeypatch.setattr(P, "stage_smooth", lambda *a, **k: trs)
    return take, work


def test_world_with_one_main_camera_writes_the_reason(tmp_path, monkeypatch):
    take, work = _run_take(tmp_path, monkeypatch)
    (work / "cameras" / "world_from_sfm.json").write_text('{"s": 1.0}')
    rep = P.run(take, ["01", "02"], ("world",))
    assert rep["world"]["solved"] is False and "the main model holds only 1 camera(s)" in rep["world"]["reason"]
    assert json.loads((work / "cameras" / "report.json").read_text())["world"] == rep["world"]
    # an earlier run's world frame (maybe of another map) is moved out of every reader's way (twoview, mamma, qa)
    assert not (work / "cameras" / "world_from_sfm.json").exists()
    assert (work / "cameras" / "world_from_sfm.prev.json").read_text() == '{"s": 1.0}'
    assert "moved to world_from_sfm.prev.json" in rep["world"]["world_from_sfm"]
    assert P.run(take, ["01", "02"], ("world",))["world"]["world_from_sfm"] is None  # nothing left to move
    monkeypatch.setattr(P, "stage_smooth", lambda *a, **k: {})  # no localisation at all: the same record
    assert P.run(take, ["01", "02"], ("world",))["world"]["solved"] is False


def test_world_qa_with_one_main_camera(tmp_path, monkeypatch):
    """Job A's `--stages world,qa` on a model of one camera: report.json with the skip, qa without a world frame;
    a later qa run that finds a world file again drops the stale skip record."""
    take, work = _run_take(tmp_path, monkeypatch)
    (work / "cameras" / "world_from_sfm.json").write_text('{"s": 1.0}')
    got = []

    def fake_report(work, cams, recs, model, tracks, pts, rgb, sim3, map_info, sigma_s, max_px, attach=None, n_take=None):
        got.append((sim3, n_take))
        return {"cams": {}}
    monkeypatch.setattr(P, "build_report", fake_report)
    monkeypatch.setattr(P, "camera_type_probe", lambda *a, **k: None)
    monkeypatch.setattr(P, "localisation_record", lambda work, cams: {})
    P.run(take, ["01", "02"], ("world", "qa"))
    disk = json.loads((work / "cameras" / "report.json").read_text())
    assert disk["world"]["solved"] is False and got == [(None, 2)]
    assert P.run(take, ["01", "02"], ("qa",))["world"]["solved"] is False and got[-1] == (None, 2)  # still no world
    (work / "cameras" / "world_from_sfm.prev.json").replace(work / "cameras" / "world_from_sfm.json")  # restored by hand
    rep = P.run(take, ["01"], ("qa",))
    assert "world" not in rep and got[-1] == ({"s": 1.0}, 2)  # n_take: the take's cameras, not the --cams subset
    # a record that is not a skip (front_full's {"copied_from"}) is never dropped
    rep_p = work / "cameras" / "report.json"
    rep_p.write_text(json.dumps({"world": {"copied_from": "larissa_kadu_38"}}))
    assert P.run(take, ["01"], ("qa",))["world"] == {"copied_from": "larissa_kadu_38"}


def test_world_of_a_larger_model_without_its_tracks_raises(tmp_path, monkeypatch):
    take, work = _run_take(tmp_path, monkeypatch, model=("01", "02"), tracks=("01",))
    (work / "cameras" / "world_from_sfm.json").write_text('{"s": 1.0}')
    with pytest.raises(RuntimeError, match=r"world needs the main-model cameras' localisations"):
        P.run(take, ["01"], ("world",))
    assert (work / "cameras" / "world_from_sfm.json").read_text() == '{"s": 1.0}'
    assert not (work / "cameras" / "report.json").exists()


def _capture_world(monkeypatch):
    got = {}
    monkeypatch.setattr(P, "triangulate_dancers", lambda work, tracks: got.setdefault("t", {c: tr["t"] for c, tr in tracks.items()}))
    monkeypatch.setattr(P, "statures_from_take", lambda cfg: STATURES)
    monkeypatch.setattr(P, "sole_from_take", lambda cfg: ({}, "default"))
    monkeypatch.setattr(P, "main_vertical", lambda work, tracks, cams: {"fused": {}})

    def fake_bootstrap(tracks, dancers, pts, st, src, x_cam="06", **k):
        got["x_cam"], got["bootstrap_t"] = x_cam, {c: tr["t"] for c, tr in tracks.items()}
        raise _Stop
    monkeypatch.setattr(P, "bootstrap", fake_bootstrap)
    return got


def test_run_passes_the_world_x_cam_to_bootstrap(tmp_path, monkeypatch):
    take, work = _run_take(tmp_path, monkeypatch, cfg=tomllib.loads(TWO_VIEW), model=("01", "02"), tracks=("01", "02"))
    got = _capture_world(monkeypatch)
    with pytest.raises(_Stop):
        P.run(take, ["01", "02"], ("world",))
    assert got["x_cam"] == "02"
    np.testing.assert_array_equal(got["t"]["01"], got["bootstrap_t"]["01"])  # no sync/offsets.json: audio clocks


def _offsets(work, method, d01=0.025, **o):
    (work / "sync").mkdir(exist_ok=True)
    doc = {"version": 1, "gauge": "camera 02 = 0", "method": method,
           "offsets": {"01": {"delta": d01, "stable": True, "use": True, "rate": 2e-5, "use_rate": False, "t_ref": 100.0, **o},
                       "02": {"delta": 0.0}}}
    (work / "sync" / "offsets.json").write_text(json.dumps(doc))


def test_world_triangulates_on_a_two_view_sync(tmp_path, monkeypatch):
    """World before sync (verifier): with cameras.twoview's sync/offsets.json the world's dancers are triangulated on
    the synced clocks (t + delta, as world_scale chain); dancecap.sync's offsets leave the world on the audio clocks."""
    take, work = _run_take(tmp_path, monkeypatch, model=("01", "02"), tracks=("01", "02"))
    t = 16.8 + np.arange(30) / 30
    tr = {c: {"t": t, "ok": np.ones(30, bool)} for c in ("01", "02")}
    out, rec = P.world_sync_tracks(work, tr)
    assert out is tr and rec is None  # no file
    _offsets(work, "landmarks")
    assert P.world_sync_tracks(work, tr)[1] is None  # dancecap.sync: not applied (larissa_kadu, carlos_aline)
    _offsets(work, P.TWO_VIEW_SYNC)
    out, rec = P.world_sync_tracks(work, tr)
    np.testing.assert_allclose(out["01"]["t"], t + 0.025)  # use_rate false: the constant delta only
    np.testing.assert_allclose(out["02"]["t"], t)  # the gauge camera
    assert tr["01"]["t"] is t and out["01"] is not tr["01"]  # the input tracks are not modified
    assert rec == {"applied": True, "file": "sync/offsets.json", "method": "two_view_epipolar", "gauge": "camera 02 = 0",
                   "offsets_ms": {"01": 25.0, "02": 0.0}, "rate_ms_per_100s": {"01": 0.0, "02": 0.0}, "not_in_file": []}
    _offsets(work, P.TWO_VIEW_SYNC, use_rate=True)
    np.testing.assert_allclose(P.world_sync_tracks(work, tr)[0]["01"]["t"], t + 0.025 + 2e-5 * (t - 100.0))
    _offsets(work, P.TWO_VIEW_SYNC, use=False)  # an untrusted estimate: delta 0
    np.testing.assert_allclose(P.world_sync_tracks(work, tr)[0]["01"]["t"], t)
    _offsets(work, P.TWO_VIEW_SYNC)
    got = _capture_world(monkeypatch)
    with pytest.raises(_Stop):
        P.run(take, ["01", "02"], ("world",))
    np.testing.assert_allclose(got["t"]["01"], t + 0.025)
    np.testing.assert_allclose(got["bootstrap_t"]["01"], t)  # the cameras themselves stay on their frame times


def test_stage_smooth_keeps_an_adopted_track(tmp_path):
    work = _work_with_frames(tmp_path)
    np.savez(work / "cameras" / "02.npz", t=np.zeros(2), type_model=np.str_("fixed_centre"))
    np.savez(work / "cameras" / "01.npz", t=np.zeros(2))
    (work / "cameras" / "orig").mkdir()
    np.savez(work / "cameras" / "orig" / "02_loc.npz", t=np.zeros(2))
    assert P.adopted_track(work, "02") == work / "cameras" / "orig" / "02_loc.npz" and P.adopted_track(work, "01") is None
    P.stage_smooth(work, ["01", "02"], 0.04, 4.0, types={})  # no localisation caches
    assert (work / "cameras" / "02.npz").exists() and not (work / "cameras" / "01.npz").exists()
    # with a localisation cache: the track is returned (world / qa), the adopted file is not rewritten
    from dancecap.cameras.localize import FramePose
    frames = [ImageRec(c, "win", i, 16.8 + i / 30, f"frames/{c}/{i:06d}.jpg", "") for c in ("01", "02") for i in range(12)]
    for c in ("01", "02"):
        poses = [FramePose(True, np.eye(3), np.array([0.01 * i, 0, 0]), 1100.0, 0.0, 200, 0.6, "loc") for i in range(12)]
        P._save_raw(work / "cameras" / "cache" / f"loc_{c}.npz", [r for r in frames if r.cam == c], poses)
    tr = P.stage_smooth(work, ["01", "02"], 0.04, 4.0, types={})
    assert sorted(tr) == ["01", "02"] and len(tr["02"]["t"]) == 12
    with np.load(work / "cameras" / "02.npz") as z:
        assert str(z["type_model"]) == "fixed_centre" and len(z["t"]) == 2
    with np.load(work / "cameras" / "01.npz") as z:
        assert len(z["t"]) == 12


# ------------------------------------------------------------------------------------------------ [cameras] readers


def test_jl_lowconf_with_x_cam_and_map_in_the_cameras_table(tmp_path, monkeypatch):
    (tmp_path / "takes").mkdir()
    (tmp_path / "takes" / "tt.toml").write_text(TWO_VIEW, encoding="utf-8")
    monkeypatch.setattr(lc, "PACKAGE_ROOT", tmp_path)
    iv = lc.intervals(tmp_path / "work" / "tt")
    assert [(x["level"], x["from"], x["to"]) for x in iv] == [("zoom_step", 125.0, 127.0)]


def _old_cams(cfg):  # room_world_*.member_list's comprehension before x_cam / map
    return {k: {x: y for x, y in v.items() if x not in ("source", "set_at")} for k, v in (cfg.get("cameras") or {}).items()}


@pytest.mark.parametrize("mod", ["room_world_two_cams", "room_world_single_cam"])
def test_room_world_camera_types_skip_x_cam_and_map(tmp_path, monkeypatch, mod):
    import importlib
    m = importlib.import_module(f"dancecap.{mod}")
    real = tomllib.loads((PACKAGE_ROOT / "takes" / f"{m.DANCE}.toml").read_text(encoding="utf-8"))
    (tmp_path / "takes").mkdir()
    (tmp_path / "takes" / f"{m.DANCE}.toml").write_text(TWO_VIEW, encoding="utf-8")
    (tmp_path / "cameras").mkdir()
    (tmp_path / "cameras" / "rig.json").write_text(json.dumps({k: 1 for k in ("scale", "world_from_prov", "camera_height_m",
                                                                                "focal_1x_px", "dancers_mean_distance_m", "height")}))
    monkeypatch.setattr(m, "PACKAGE_ROOT", tmp_path)
    monkeypatch.setattr(m, "WORK", tmp_path)
    got = json.loads(next(e.data for x in m.member_list() if x.name == "camera_types" for e in x.entries))
    assert sorted(got) == ["01", "02"] and "mount" in got["01"] and "source" not in got["02"]
    two = tomllib.loads(TWO_VIEW)
    with pytest.raises(AttributeError):  # what x_cam did to the old comprehension
        _old_cams(two)
    assert got == json.loads(json.dumps(_old_cams({"cameras": {k: two["cameras"][k] for k in ("01", "02")}})))
    # the real take: the same camera_types.json as before while its [cameras] holds camera tables only (today)
    new = {k: {x: y for x, y in v.items() if x not in ("source", "set_at")} for k, v in ct.cam_tables(real).items()}
    if all(ct.CAM_RE.match(k) and isinstance(v, dict) for k, v in (real.get("cameras") or {}).items()):
        assert _old_cams(real) == new and new


# ------------------------------------------------------------------------------------------------ SAM host RAM


def _old_sam_gb(n, w, h):  # the formula before the landscape QA frames
    px = w * h
    work_px = min(1.0, 1280 / max(w, h)) ** 2 * px
    return round((n * (3 * px + 2 * px + 2 * work_px + 960 * 540 * 3)) / 2**30 * 1.1 + 3.0, 1)


def test_sam_memory_sizes_landscape_qa_frames(tmp_path, monkeypatch):
    assert fp._qa_frame_px(1280, 720) == 1707 * 960 and fp._qa_frame_px(1920, 1080) == 1707 * 960
    assert fp._qa_frame_px(720, 1280) == fp._qa_frame_px(1080, 1920) == 540 * 960
    for cam, (w, h, n) in {"01": (720, 1280, 5705), "02": (1280, 720, 5896), "03": (1080, 1920, 2302)}.items():
        (tmp_path / "frames" / cam).mkdir(parents=True)
        (tmp_path / "frames" / cam / "frames.json").write_text(json.dumps({"width": w, "height": h, "frames": [{}] * n}))
    assert fp._sam_mem_gb(tmp_path, "01") == _old_sam_gb(5705, 720, 1280)  # portrait: unchanged
    assert fp._sam_mem_gb(tmp_path, "03") == _old_sam_gb(2302, 1080, 1920)
    extra = 5896 * 3 * (1707 * 960 - 540 * 960) * 1.1 / 2**30
    assert fp._sam_mem_gb(tmp_path, "02") == pytest.approx(_old_sam_gb(5896, 1280, 720) + extra, abs=0.11)
    assert extra > 10  # a whole tripod camera: ~11 GB more than the 960x540 allowance
    from dancecap import recap_front as rf
    (tmp_path / "cameras").mkdir()
    for cam, (w, h) in {"01": (720, 1280), "02": (1280, 720)}.items():
        np.savez(tmp_path / "cameras" / f"{cam}.npz", t=np.arange(1800) / 30.0, native_size=np.array([w, h]))
    monkeypatch.setattr(rf, "dance_work", lambda: tmp_path)
    chunk = {"t0": 0.0, "t1": 56.0}
    n = int(56.0 * 30) + 1
    assert rf.sam_mem_gb(chunk, "01") == _old_sam_gb(n, 720, 1280)
    assert rf.sam_mem_gb(chunk, "02") == pytest.approx(_old_sam_gb(n, 1280, 720) + n * 3 * (1707 - 540) * 960 * 1.1 / 2**30,
                                                       abs=0.11)


def test_a_moving_camera_pairs_with_its_own_images_only_within_seq_only_s():
    """lucas_luiza: the Steadicam's images minutes apart, facing look-alike walls, formed a self-consistent copy of the
    room; its own pairs are now limited to seq_only_s (the dense exhaustive set too), cross-camera pairs are unchanged."""
    from dancecap.cameras.features import ImageRec
    from dancecap.cameras.mapping import MapSettings, map_pairs
    imgs = [ImageRec(c, "win", i, 10.0 + 2.0 * i, f"frames/{c}/{i:06d}.jpg", "") for c in ("01", "02") for i in range(30)]
    dense = {r.key for r in imgs if r.index % 3 == 0}
    s = MapSettings(seq_only_cams=("01",), seq_only_s=8.0)
    pairs = map_pairs(imgs, dense, s)
    same01 = [(a, b) for a, b in pairs if imgs[a].cam == imgs[b].cam == "01"]
    assert same01 and all(abs(imgs[a].t - imgs[b].t) <= 8.0 for a, b in same01)
    assert any(abs(imgs[a].t - imgs[b].t) > 8.0 for a, b in pairs if imgs[a].cam == imgs[b].cam == "02")
    old = map_pairs(imgs, dense, MapSettings())
    cross = lambda P: {(a, b) for a, b in P if imgs[a].cam != imgs[b].cam}  # noqa: E731
    assert cross(pairs) == cross(old)
    assert map_settings({"cameras": {"map": {"seq_only_cams": "01"}}}).seq_only_cams == ("01",)
