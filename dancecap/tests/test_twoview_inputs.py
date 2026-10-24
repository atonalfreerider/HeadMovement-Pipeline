"""Unit tests for the two-view chunk inputs (dancecap.carlos_aline_inputs --take / --pts-from frames / --pose-from people / --teachers-floor) and
jl_export.reliability_meta for one and for two views (synthetic arrays and cameras; no footage, no ffprobe, no pod).

    ../.venv/bin/python -m pytest tests/test_twoview_inputs.py -q        (from dancecap/)
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dancecap import carlos_aline_inputs as ci  # noqa: E402
from dancecap import jl_export as JE  # noqa: E402
from dancecap import jl_lowconf as lc  # noqa: E402
from dancecap import take as take_mod  # noqa: E402
from dancecap.recap_front import people_arrays  # noqa: E402
from dancecap.take import Source, Take  # noqa: E402


# ------------------------------------------------------------------------------------------------ pts -> frame_index


def _pts_29(n=6000):
    return np.round(4.689655 + np.arange(n) / 29.0, 6)             # the Steadicam: 29 fps CFR


def _pts_5994(n=14000):
    return np.round(np.arange(n) * 1001 / 60000, 6)                 # the tripod original: 59.94 fps


def test_frame_index_steps_of_one_frame():
    pts_all = _pts_29()
    rng = np.random.default_rng(0)
    pts = pts_all[100:401] + rng.uniform(-0.0004, 0.0004, 301)     # showinfo vs ffprobe rounding
    fi, d = ci.frame_index_from_pts(pts_all, pts)
    assert fi.dtype == np.int32 and np.array_equal(fi, np.arange(100, 401)) and d <= 0.0004
    assert ci.step_gate(fi, 1)["pass"] and not ci.step_gate(fi, 2)["pass"]


def test_frame_index_steps_of_two_frames():
    pts_all = _pts_5994()
    pts = pts_all[1007:7007:2] + 0.0003
    fi, _ = ci.frame_index_from_pts(pts_all, pts)
    assert np.array_equal(fi, np.arange(1007, 7007, 2))
    g = ci.step_gate(fi, 2)
    assert g["pass"] and g["value"] == [2] and g["threshold"] == 2
    gap = np.delete(fi, 50)                                        # a dropped frame: one step of 4
    assert not ci.step_gate(gap, 2)["pass"] and ci.step_gate(gap, 2)["value"] == [2, 4]
    assert not ci.step_gate(fi[::-1], 2)["pass"]                   # not increasing


def test_frame_index_enforces_one_ms():
    pts_all = _pts_5994()
    pts = pts_all[10:20].copy()
    pts[3] += 0.0009
    assert ci.frame_index_from_pts(pts_all, pts)[0][3] == 13
    pts[3] += 0.0006                                               # 1.5 ms: no video frame within 1 ms
    with pytest.raises(ValueError, match="1 of 10 frames"):
        ci.frame_index_from_pts(pts_all, pts)
    with pytest.raises(ValueError):                                # half way between two frames (8.3 ms)
        ci.frame_index_from_pts(pts_all, pts_all[10:12] + 1001 / 120000)
    assert ci.frame_index_from_pts(pts_all, pts_all[[0, -1]])[0].tolist() == [0, len(pts_all) - 1]   # both ends


# ------------------------------------------------------------------------------------------------ pose tables from people/<cam>.npz


def _people(valid: np.ndarray, seed: int = 1) -> dict:
    F, N = valid.shape
    rng = np.random.default_rng(seed)
    kpts = np.zeros((F, N, 17, 3), np.float32)
    kpts[..., :2] = rng.uniform(100, 600, (F, N, 17, 2))
    kpts[..., 2] = 0.9
    boxes = np.zeros((F, N, 4), np.float32)
    boxes[..., :2], boxes[..., 2:] = kpts[..., :2].min(2), kpts[..., :2].max(2)
    return {"kpts": kpts, "valid": valid, "boxes": boxes, "scores": rng.uniform(0.3, 1, (F, N)).astype(np.float32)}


def test_pose_table_rows_are_the_video_frames_and_slots_the_detections():
    valid = np.array([[1, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 1], [1, 1, 0, 0, 0]], bool)
    ppl = _people(valid)
    fi = np.array([10, 12, 14, 16])
    pt = ci.people_pose_table(ppl, fi, 20)
    assert pt["kp"].shape == (20, 5, 17, 3) and pt["kp"].dtype == np.float32
    assert pt["n"].tolist() == [0] * 10 + [3, 0, 0, 0, 5, 0, 2, 0, 0, 0]       # last valid slot + 1 at row frame_index[i]
    assert np.array_equal(pt["kp"][10, 0], ppl["kpts"][0, 0]) and np.array_equal(pt["kp"][10, 2], ppl["kpts"][0, 2])
    assert np.isnan(pt["kp"][10, 1]).all() and np.isnan(pt["kp"][11]).all() and np.isnan(pt["kp"][14, :4]).all()
    assert np.array_equal(pt["box"][16, 1], ppl["boxes"][3, 1]) and pt["det_conf"][16, 1] == ppl["scores"][3, 1] and pt["det_conf"][10, 1] == 0
    assert np.isnan(pt["rgb"]).all()
    assert ci.pose_rows_match(pt, ppl, fi)
    # the gate fails on each mismatch alone: a joint, the frame numbering, n
    bad = {k: v.copy() for k, v in pt.items()}
    bad["kp"][16, 0, 3, 0] += 1
    assert not ci.pose_rows_match(bad, ppl, fi)
    assert not ci.pose_rows_match(pt, ppl, np.array([10, 12, 14, 17]))
    bad = {k: v.copy() for k, v in pt.items()}
    bad["n"][14] = 4
    assert not ci.pose_rows_match(bad, ppl, fi)
    # recap_front.people_arrays rebuilds people/<cam>.npz of a chunk from the table: the slots (= teachers.row) are the detector's
    pa = people_arrays(pt["kp"], pt["n"], fi)
    assert np.array_equal(pa["valid"], valid) and np.array_equal(pa["kpts"][valid], ppl["kpts"][valid])
    # ... except a detection with fewer than 3 joints of conf >= 0.3: valid for the detector, a miss in the chunk (recap_people.person_box)
    ppl["kpts"][3, 1, 2:, 2] = 0.1
    pt = ci.people_pose_table(ppl, fi, 20)
    pa = people_arrays(pt["kp"], pt["n"], fi)
    assert ci.pose_rows_match(pt, ppl, fi) and not pa["valid"][3, 1] and pa["valid"][3, 0]


def test_pose_table_refuses_rows_that_are_not_distinct_video_frames():
    ppl = _people(np.ones((3, 2), bool))
    for fi in ([0, 0, 1], [0, 1, 9], [-1, 0, 1], [0, 1]):
        with pytest.raises(ValueError):
            ci.people_pose_table(ppl, np.array(fi), 9)


def test_teacher_row_gates_count_what_the_chunk_keeps():
    F = 10
    valid = np.zeros((F, 4), bool)
    valid[:, [0, 2]] = True
    ppl = _people(valid)
    fi = np.arange(F) * 2 + 5
    row = np.tile(np.array([0, 2], np.int16), (F, 1))
    row[4, 1] = -1                                                 # a miss is not counted
    pt = ci.people_pose_table(ppl, fi, 30)
    g = ci.teacher_row_gates(row, ppl, pt, fi)
    assert g == {"teacher_rows_valid_in_pose_table": {"value": 1.0, "threshold": 1.0, "pass": True},
                 "teacher_rows_valid_in_chunk_people": {"value": 1.0, "threshold": ci.CHUNK_ROW_MIN_SHARE, "pass": True}}
    ppl["kpts"][[1, 7], 2, 2:, 2] = 0.1                            # the follow on two faint detections (2 joints >= 0.3)
    pt = ci.people_pose_table(ppl, fi, 30)
    g = ci.teacher_row_gates(row, ppl, pt, fi)
    assert g["teacher_rows_valid_in_pose_table"]["pass"]
    assert g["teacher_rows_valid_in_chunk_people"]["value"] == round(17 / 19, 4) and not g["teacher_rows_valid_in_chunk_people"]["pass"]
    row[2, 0] = 1                                                  # a slot the detector does not hold
    assert ci.teacher_row_gates(row, ppl, pt, fi)["teacher_rows_valid_in_pose_table"]["value"] == round(18 / 19, 4)


# ------------------------------------------------------------------------------------------------ floor: ankle-midpoint ray at the ankle height


def _rot(axis, deg):
    a = np.radians(deg)
    k = np.asarray(axis, float) / np.linalg.norm(axis)
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * Kx + (1 - np.cos(a)) * Kx @ Kx


def _look_at(C, target):
    """world -> camera rotation (OpenCV: x right, y down, z forward) of a camera at C looking at target, Y up."""
    z = target - C
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 1, 0])
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z])


SIM3 = {"s": 2.5, "R": _rot([0.3, 1, -0.2], 37).tolist(), "t": [0.3, -0.2, 0.1]}       # X_world = s R X_sfm + t
ANKLE_Y = (0.093, 0.124)                                                                 # lucas_luiza: 0.039 x 1.75 + 0.025, 0.039 x 1.63 + 0.06


def _camera(F, C_w=(1.0, 1.6, 5.0), k1=-0.08):
    """cameras/<cam>.npz arrays in the SfM frame of SIM3, plus a projector of world points."""
    Rw, s, tw = np.asarray(SIM3["R"]), SIM3["s"], np.asarray(SIM3["t"])
    C_w = np.asarray(C_w, float)
    Rcw = _look_at(C_w, np.array([0.0, 0.5, 0.0]))
    R = Rcw @ Rw                                                   # x_cam ~ Rcw (X_w - C_w) = s Rcw Rw (X_sfm - C_sfm)
    C_sfm = Rw.T @ (C_w - tw) / s
    K = np.array([[1100.0, 0, 640], [0, 1100.0, 360], [0, 0, 1]])
    cam = {"K": np.repeat(K[None], F, 0), "R": np.repeat(R[None], F, 0), "tvec": np.repeat((-R @ C_sfm)[None], F, 0),
           "dist": np.full(F, k1), "ok": np.ones(F, bool)}

    def project(Xw):
        Xc = (np.asarray(Xw, float) - C_w) @ Rcw.T
        n = Xc[..., :2] / Xc[..., 2:]
        n = n * (1 + k1 * (n ** 2).sum(-1, keepdims=True))
        return n * 1100.0 + [640, 360]
    return cam, project


def test_floor_is_where_the_ankle_ray_meets_the_floor():
    F = 5
    cam, project = _camera(F)
    kp = np.full((F, 2, 17, 3), np.nan, np.float32)
    lead, follow = np.array([0.4, 0.0, -0.3]), np.array([-0.5, 0.0, 0.6])
    for f in range(F):
        kp[f, 0, 15, :2], kp[f, 0, 15, 2] = project(lead), 0.9                  # one ankle seen
        kp[f, 0, 16, :2], kp[f, 0, 16, 2] = project(lead + [0.3, 0, 0]), 0.1  # below FLOOR_CONF: ignored
        kp[f, 1, 15:17, :2], kp[f, 1, 15:17, 2] = project(follow), 0.8         # both ankles on one point
    fl = ci.floor_xz(kp, cam, SIM3)
    assert fl.shape == (F, 2, 2) and fl.dtype == np.float32
    assert np.allclose(fl[:, 0], lead[[0, 2]], atol=1e-3) and np.allclose(fl[:, 1], follow[[0, 2]], atol=1e-3)
    # two ankles 0.25 m apart: the midpoint ray lands within a few mm of the feet's midpoint
    kp[0, 1, 15, :2], kp[0, 1, 16, :2] = project(follow + [0.125, 0, 0]), project(follow - [0.125, 0, 0])
    assert np.allclose(ci.floor_xz(kp, cam, SIM3)[0, 1], follow[[0, 2]], atol=0.01)
    # NaN: camera not ok, no ankle, a ray above the horizon (an 'ankle' 3 m up), a grazing ray beyond FLOOR_MAX_M
    cam["ok"][1] = False
    kp[2, 0, 15:17, 2] = 0.0
    kp[3, 0, 15, :2] = project([0.4, 3.0, -0.3])
    kp[4, 1, 15:17, :2] = project([-30.0, 0.0, -40.0])
    fl = ci.floor_xz(kp, cam, SIM3)
    assert np.isnan(fl[1]).all() and np.isnan(fl[2, 0]).all() and np.isfinite(fl[2, 1]).all()
    assert np.isnan(fl[3, 0]).all() and np.isnan(fl[4, 1]).all() and np.isfinite(fl[0]).all()


def test_floor_at_each_roles_ankle_height():
    """The ankles sit ANKLE_Y above the floor y = 0: cast to their own height the rays give the feet's (x, z); cast to y = 0 they land
    h_cam / (h_cam - h_ankle) farther from the camera, a different amount per role."""
    C = np.array([1.0, 1.6, 5.0])
    cam, project = _camera(1, C_w=C)
    lead, follow = np.array([0.4, ANKLE_Y[0], -0.3]), np.array([-0.5, ANKLE_Y[1], 0.6])
    kp = np.full((1, 2, 17, 3), np.nan, np.float32)
    kp[0, 0, 15:17, :2], kp[0, 1, 15:17, :2], kp[0, :, 15:17, 2] = project(lead), project(follow), 0.9
    fl = ci.floor_xz(kp, cam, SIM3, np.array(ANKLE_Y))
    assert np.allclose(fl[0, 0], lead[[0, 2]], atol=1e-3) and np.allclose(fl[0, 1], follow[[0, 2]], atol=1e-3)
    fl0 = ci.floor_xz(kp, cam, SIM3)
    for k, X in enumerate((lead, follow)):
        far = C[[0, 2]] + C[1] / (C[1] - X[1]) * (X[[0, 2]] - C[[0, 2]])
        assert np.allclose(fl0[0, k], far, atol=1e-3)
    assert ci.ankle_heights({"lead": {"stature_m": 1.75, "shoe_sole_m": 0.025}, "follow": {"stature_m": 1.63, "shoe_sole_m": 0.06}}) == (0.0932, 0.1236)
    assert ci.ankle_heights({}) == (round(0.039 * 1.75 + 0.025, 4), round(0.039 * 1.65 + 0.025, 4))     # cameras.world's defaults


def test_floor_gates():
    F = 100
    row = np.zeros((F, 2), np.int16)
    row[:10, 1] = -1                                               # the follow unmatched: not a both-role frame
    fl = np.zeros((F, 2, 2), np.float32)
    fl[:, 1, 0] = 0.8
    g = ci.floor_gates(fl, row)
    assert g["floor_finite_share"]["pass"] and g["floor_finite_share"]["value"] == 1.0
    assert g["floor_pair_distance_median_m"]["pass"] and abs(g["floor_pair_distance_median_m"]["value"] - 0.8) < 1e-6
    fl[10:30, 0] = np.nan                                          # 20 of the 90 both-role frames: 77.8 %
    g = ci.floor_gates(fl, row)
    assert not g["floor_finite_share"]["pass"] and abs(g["floor_finite_share"]["value"] - 70 / 90) < 1e-3
    fl[:, 1, 0] = 2.5
    assert not ci.floor_gates(fl, row)["floor_pair_distance_median_m"]["pass"]
    g = ci.floor_gates(np.full((F, 2, 2), np.nan, np.float32), row)
    assert not g["floor_finite_share"]["pass"] and g["floor_pair_distance_median_m"]["value"] is None


# ------------------------------------------------------------------------------------------------ prepare on a synthetic two-view take


CLOCKS = {"01": {"slope": 0.9999163457150201, "offset": 12.12290391571289, "offset_source": "audio clock (test)"},
          "02": {"slope": 1.0, "offset": 0.0, "offset_source": "audio clock (test)"}}


def _two_view_take(root: Path, n01=40, n02=40, start02=1007) -> tuple[ci.Spec, dict]:
    """work/<take>/ with the ingest's frames.json (01: 29 fps, every frame; 02: 59.94 fps, every 2nd), the camera stage's tracks (no pts), the world,
    people/<cam>.npz and the seeded teachers (frame_index -1, floor NaN); the feet stand at the roles' ankle height ANKLE_Y."""
    work = root / "work" / "tv_take"
    (work / "cameras").mkdir(parents=True)
    (work / "people").mkdir()
    pts_all = {"01": _pts_29(), "02": _pts_5994()}
    sel = {"01": np.arange(150, 150 + n01), "02": np.arange(start02, start02 + 2 * n02, 2)}
    size = {"01": (720, 1280), "02": (1280, 720)}
    (work / "cameras" / "world_from_sfm.json").write_text(json.dumps(SIM3))
    lead, follow = np.array([0.4, ANKLE_Y[0], -0.3]), np.array([-0.5, ANKLE_Y[1], 0.6])
    for cam in ("01", "02"):
        fdir = work / "frames" / cam
        fdir.mkdir(parents=True)
        pts = pts_all[cam][sel[cam]] + 0.0002
        t = CLOCKS[cam]["slope"] * pts + CLOCKS[cam]["offset"]
        meta = {"camera": cam, "source": f"{cam}.mp4", "width": size[cam][0], "height": size[cam][1], "rotation": 0,
                **({"decimate_every": 2} if cam == "02" else {}),
                "frames": [{"file": f"{i:06d}.jpg", "pts": round(float(p), 6), "t": float(tt)} for i, (p, tt) in enumerate(zip(pts, t))]}
        (fdir / "frames.json").write_text(json.dumps(meta))
        F = len(pts)
        c, project = _camera(F, C_w=(1.0, 1.6, 5.0) if cam == "01" else (-4.0, 2.0, 1.0))
        np.savez(work / "cameras" / f"{cam}.npz", t=t, **c, reproj_px=np.ones(F, np.float32))
        valid = np.zeros((F, 4), bool)
        valid[:, [0, 2]] = True
        valid[::7, 3] = True
        ppl = _people(valid, seed=int(cam))
        for k, X in ((0, lead), (2, follow)):
            ppl["kpts"][:, k, 15:17, :2] = project(X)
        np.savez(work / "people" / f"{cam}.npz", t=t, **ppl)
        row = np.tile(np.array([0, 2], np.int16), (F, 1))
        row[5, 1] = -1
        tk = np.full((F, 2, 17, 3), np.nan, np.float32)
        tk[:, 0], tk[:, 1] = ppl["kpts"][:, 0], ppl["kpts"][:, 2]
        tk[5, 1] = np.nan
        np.savez(work / "people" / f"teachers_{cam}.npz", t=t, frame_index=np.full(F, -1, np.int32), ok=np.ones(F, bool), row=row, kp=tk,
                 box=np.zeros((F, 2, 4), np.float32), conf=np.where(row >= 0, 0.9, 0).astype(np.float32),
                 floor=np.full((F, 2, 2), np.nan, np.float32))
    sp = ci.Spec("tv_take", "tv-capture", work, {"01": root / "src" / "Video-01.mp4", "02": root / "src" / "Video 02.mp4"}, CLOCKS, ankle_y=ANKLE_Y)
    return sp, pts_all


def _fns(pts_all):
    def pts_fn(path, cache):
        return pts_all["01" if "01" in path.name else "02"]

    def probe_fn(path):
        return {"width": 1, "height": 1, "avg_fps": 29.0 if "01" in path.name else 59.94, "duration": 216.48}
    return pts_fn, probe_fn


TWO_VIEW = {"pts_from": "frames", "pose_from": "people", "teachers_floor": True}


def _snapshot(work: Path) -> dict:
    return {str(p.relative_to(work)): p.read_bytes() for p in sorted(work.rglob("*")) if p.is_file()}


def test_prepare_two_view_take(tmp_path):
    sp, pts_all = _two_view_take(tmp_path)
    pts_fn, probe_fn = _fns(pts_all)
    res = ci.prepare(sp=sp, **TWO_VIEW, pts_fn=pts_fn, probe_fn=probe_fn)
    work = sp.work
    rc = json.loads((work / "cameras" / "recap_cams.json").read_text())
    assert rc["take"] == "tv_take" and rc["capture"] == "tv-capture"
    assert rc["cams"]["01"]["file"] == "Video-01.mp4"
    assert (rc["cams"]["01"]["width"], rc["cams"]["01"]["height"]) == (720, 1280)                # frames.json, not the probe
    assert rc["cams"]["01"]["offset"] == sp.clocks["01"]["offset"] and rc["cams"]["02"]["avg_fps"] == 59.94
    written = json.loads((work / "cameras" / "orig" / ci.WRITTEN).read_text())
    for cam, step, first in (("01", 1, 150), ("02", 2, 1007)):
        z = np.load(work / "cameras" / f"{cam}.npz")
        fm = json.loads((work / "frames" / cam / "frames.json").read_text())
        assert np.array_equal(z["pts"], [f["pts"] for f in fm["frames"]])                      # pts from frames.json
        assert z["frame_index"][0] == first and set(np.diff(z["frame_index"])) == {step}
        assert z["native_size"].tolist() == [rc["cams"][cam]["width"], rc["cams"][cam]["height"]] and "K" in z.files
        assert "pts" not in np.load(work / "cameras" / "orig" / f"{cam}.npz").files              # the camera stage's own track is kept
        assert written[cam] == ci._sha1(work / "cameras" / f"{cam}.npz")
        pt = np.load(work / "people" / "pose_tables" / f"{cam}.npz")
        ppl = np.load(work / "people" / f"{cam}.npz")
        assert len(pt["n"]) == len(pts_all[cam]) and np.array_equal(pt["kp"][z["frame_index"][3], 2], ppl["kpts"][3, 2])
        tt = np.load(work / "people" / f"teachers_{cam}.npz")
        assert np.array_equal(tt["frame_index"], z["frame_index"]) and tt["frame_index"].dtype == np.int32
        assert np.allclose(tt["floor"][:, 0], [0.4, -0.3], atol=0.01) and np.isnan(tt["floor"][5, 1]).all()
        assert np.allclose(tt["floor"][np.arange(len(z["t"])) != 5, 1], [-0.5, 0.6], atol=0.01)   # at its own ankle height: no outward push
        assert res[cam]["floor_plane_y_m"] == list(ANKLE_Y)
        g = res[cam]["gates"]
        assert all(v["pass"] for v in g.values()), g
        assert list(g) == ["clock_max_dev_ms", "frame_index_steps", "pose_rows_match_people", "floor_finite_share", "floor_pair_distance_median_m",
                           "teacher_rows_valid_in_pose_table", "teacher_rows_valid_in_chunk_people"]
        assert g["frame_index_steps"]["value"] == [step]
        assert g["floor_pair_distance_median_m"]["value"] == pytest.approx(np.hypot(0.9, 0.9), abs=0.01)
    assert json.loads((work / "people" / "inputs.json").read_text()) == res
    # a second run starts from cameras/orig/ again: the same outputs
    before = _snapshot(work)
    res2 = ci.prepare(sp=sp, **TWO_VIEW, pts_fn=pts_fn, probe_fn=probe_fn)
    assert res2 == res and _snapshot(work) == before
    # the camera stage rewrote 02.npz (no pts): the kept original is refreshed
    raw = {k: v for k, v in np.load(work / "cameras" / "02.npz").items() if k not in ("pts", "frame_index", "native_size", "work_size")}
    raw["reproj_px"] = np.full(len(raw["t"]), 7.0, np.float32)
    np.savez(work / "cameras" / "02.npz", **raw)
    ci.prepare(sp=sp, **TWO_VIEW, pts_fn=pts_fn, probe_fn=probe_fn)
    assert np.all(np.load(work / "cameras" / "orig" / "02.npz")["reproj_px"] == 7.0)
    assert np.all(np.load(work / "cameras" / "02.npz")["reproj_px"] == 7.0)


def test_prepare_backup_follows_the_tools_that_replace_the_track(tmp_path):
    """cameras/orig/<cam>.npz is refreshed whenever cameras/<cam>.npz is not the file prepare wrote last: the qa stage's in-place rewrite (keeps pts,
    adds dancer_px), a PTZ adoption (twoview.adopt_track moves the track to cameras/orig/<cam>_loc.npz and installs the PTZ one) and its undo."""
    sp, pts_all = _two_view_take(tmp_path)
    pts_fn, probe_fn = _fns(pts_all)
    cdir = sp.work / "cameras"

    def run():
        ci.prepare(sp=sp, **TWO_VIEW, pts_fn=pts_fn, probe_fn=probe_fn)
        return dict(np.load(cdir / "02.npz")), dict(np.load(cdir / "orig" / "02.npz"))
    run()
    z = dict(np.load(cdir / "02.npz"))
    z["dancer_px"] = np.full(len(z["t"]), 3.0, np.float32)                                     # qa: in place, every key kept
    np.savez(cdir / "02.npz", **z)
    cur, kept = run()
    assert "dancer_px" in cur and "dancer_px" in kept and "pts" in cur
    ptz = {k: v for k, v in cur.items() if k not in ("pts", "frame_index", "native_size", "work_size", "dancer_px")}
    ptz["reproj_px"] = np.full(len(ptz["t"]), 5.0, np.float32)
    shutil.move(str(cdir / "02.npz"), str(cdir / "orig" / "02_loc.npz"))                       # adopt
    np.savez(cdir / "02.npz", **ptz)
    cur, kept = run()
    assert np.all(cur["reproj_px"] == 5.0) and np.all(kept["reproj_px"] == 5.0) and "dancer_px" not in cur
    assert (cdir / "orig" / "02_loc.npz").exists()                                             # the adoption marker is left alone
    shutil.move(str(cdir / "orig" / "02_loc.npz"), str(cdir / "02.npz"))                       # undo
    cur, kept = run()
    assert np.all(cur["reproj_px"] == 1.0) and "dancer_px" in cur and np.all(kept["reproj_px"] == 1.0)
    assert set(np.diff(cur["frame_index"])) == {2}


def test_prepare_checks_every_camera_before_writing(tmp_path):
    """A refusal on camera 02 leaves camera 01's files, recap_cams.json and inputs.json as they were."""
    sp, pts_all = _two_view_take(tmp_path)
    pts_fn, probe_fn = _fns(pts_all)
    work = sp.work
    fm = json.loads((work / "frames" / "02" / "frames.json").read_text())
    fm["frames"][20]["pts"] += 0.0020                                                          # 2 ms off its video frame
    (work / "frames" / "02" / "frames.json").write_text(json.dumps(fm))
    before = _snapshot(work)
    with pytest.raises(ValueError, match="within 1 ms"):
        ci.prepare(sp=sp, **TWO_VIEW, pts_fn=pts_fn, probe_fn=probe_fn)
    assert _snapshot(work) == before and not (work / "cameras" / "recap_cams.json").exists()
    fm["frames"][20]["pts"] -= 0.0020
    (work / "frames" / "02" / "frames.json").write_text(json.dumps(fm))
    ppl = dict(np.load(work / "people" / "02.npz"))
    np.savez(work / "people" / "02.npz", **{**ppl, "t": ppl["t"] + 0.002})                     # people not on the track's frames
    before = _snapshot(work)
    with pytest.raises(ValueError, match="people/02.npz is not on the frames"):
        ci.prepare(sp=sp, **TWO_VIEW, pts_fn=pts_fn, probe_fn=probe_fn)
    assert _snapshot(work) == before
    np.savez(work / "people" / "02.npz", **ppl)
    tt = dict(np.load(work / "people" / "teachers_02.npz"))
    np.savez(work / "people" / "teachers_02.npz", **{k: v[:-1] if k != "size" else v for k, v in tt.items()})
    before = _snapshot(work)
    with pytest.raises(ValueError, match="teachers_02.npz is not on the frames"):
        ci.prepare(sp=sp, **TWO_VIEW, pts_fn=pts_fn, probe_fn=probe_fn)
    assert _snapshot(work) == before
    (work / "people" / "teachers_02.npz").unlink()
    with pytest.raises(FileNotFoundError, match="teachers_02.npz missing"):
        ci.prepare(sp=sp, **TWO_VIEW, pts_fn=pts_fn, probe_fn=probe_fn)
    np.savez(work / "people" / "teachers_02.npz", **tt)
    (work / "people" / "02.npz").unlink()
    with pytest.raises(FileNotFoundError, match="02.npz missing"):
        ci.prepare(sp=sp, **TWO_VIEW, pts_fn=pts_fn, probe_fn=probe_fn)
    np.savez(work / "people" / "02.npz", **ppl)
    # --pts-from index on the camera stage's tracks (no frame_index): refused, naming the step-11 command
    before = _snapshot(work)
    with pytest.raises(SystemExit, match="--pts-from frames --pose-from people --teachers-floor"):
        ci.prepare(sp=sp, pose_from="people", pts_fn=pts_fn, probe_fn=probe_fn)
    assert _snapshot(work) == before
    shutil.move(str(work / "cameras" / "02.npz"), str(tmp_path / "02.npz"))
    before = _snapshot(work)
    with pytest.raises(FileNotFoundError, match="no camera track cameras/02.npz"):
        ci.prepare(sp=sp, **TWO_VIEW, pts_fn=pts_fn, probe_fn=probe_fn)
    assert _snapshot(work) == before
    shutil.move(str(tmp_path / "02.npz"), str(work / "cameras" / "02.npz"))
    assert all(v["pass"] for r in ci.prepare(sp=sp, **TWO_VIEW, pts_fn=pts_fn, probe_fn=probe_fn).values() for v in r["gates"].values())


def test_prepare_two_view_gates_fail_and_refusals(tmp_path):
    sp, pts_all = _two_view_take(tmp_path)
    pts_fn, probe_fn = _fns(pts_all)
    fm = json.loads((sp.work / "frames" / "02" / "frames.json").read_text())
    del fm["frames"][20]                                                           # a frame missing: a step of 4, and the track has one row more
    (sp.work / "frames" / "02" / "frames.json").write_text(json.dumps(fm))
    with pytest.raises(ValueError, match="not the canonical frames"):
        ci.prepare(sp=sp, pts_from="frames", pose_from="people", pts_fn=pts_fn, probe_fn=probe_fn)
    sp2 = ci.Spec(sp.take, sp.capture, sp.work, sp.videos, {**sp.clocks, "01": {**sp.clocks["01"], "offset": sp.clocks["01"]["offset"] + 0.004}})
    fm["frames"].insert(20, {"file": "x.jpg", "pts": round(float(pts_all["02"][1007 + 39]), 6), "t": 0.0})                   # an odd video frame
    (sp.work / "frames" / "02" / "frames.json").write_text(json.dumps(fm))
    res = ci.prepare(sp=sp2, pts_from="frames", pose_from="people", pts_fn=pts_fn, probe_fn=probe_fn)
    assert not res["01"]["gates"]["clock_max_dev_ms"]["pass"] and res["01"]["gates"]["clock_max_dev_ms"]["value"] == pytest.approx(4.0, abs=0.01)
    assert res["02"]["gates"]["frame_index_steps"] == {"value": [1, 2, 3], "threshold": 2, "pass": False}
    assert res["02"]["gates"]["clock_max_dev_ms"]["value"] == pytest.approx(1001 / 60, abs=0.3)   # that frame's pts is 16.7 ms from the track's t
    assert "floor_finite_share" not in res["02"]["gates"]
    with pytest.raises(SystemExit):
        ci.prepare(sp=sp, pts_from="frames", pose_from="podout", pts_fn=pts_fn, probe_fn=probe_fn)


def test_main_exit_code_follows_the_gates(tmp_path, monkeypatch):
    sp, pts_all = _two_view_take(tmp_path)
    pts_fn, probe_fn = _fns(pts_all)
    real, seen = ci.prepare, []

    def prepare(**kw):
        seen.append(kw)
        return real(**kw, sp=state["sp"], pts_fn=pts_fn, probe_fn=probe_fn)
    state = {"sp": sp}
    monkeypatch.setattr(ci, "prepare", prepare)
    argv = ["prepare", "--take", "tv_take", "--pts-from", "frames", "--pose-from", "people", "--teachers-floor"]
    assert ci.main(argv) == 0
    assert seen[0] == {"take": "tv_take", "pts_from": "frames", "pose_from": "people", "teachers_floor": True, "clocks": None}
    late = {**sp.clocks, "01": {**sp.clocks["01"], "offset": sp.clocks["01"]["offset"] + 0.004}}
    state["sp"] = ci.Spec(sp.take, sp.capture, sp.work, sp.videos, late, ankle_y=ANKLE_Y)
    assert ci.main(argv) == 1                                                                  # the 4 ms clock gate fails: exit 1, files written
    assert not json.loads((sp.work / "people" / "inputs.json").read_text())["01"]["gates"]["clock_max_dev_ms"]["pass"]
    monkeypatch.setattr(ci, "prepare", lambda **kw: {"01": {"frames": 3}, "02": {"frames": 3}})
    assert ci.main(["prepare"]) == 0                                                           # C3's run: no gates


def test_spec_of_a_toml_take(tmp_path, monkeypatch):
    """Another take: names and videos from its TOML, CAPTURE = the source folder's name, the clocks of its alignment, cameras/orig/, the ankle
    heights of [dancers]."""
    (tmp_path / "takes").mkdir()
    (tmp_path / "clock").mkdir()
    (tmp_path / "clock" / "alignment.json").write_text(json.dumps({"sources": [
        {"id": "01", "name": "Steadicam.mp4", "slope": 0.9999163457150201, "offset": 12.12290391571289},
        {"id": "02", "name": "Tripod 02.mp4", "slope": 1.0, "offset": 0.0},
        {"id": "03", "name": "Rejected.mp4", "slope": 1.0, "offset": 3.0, "accepted": False}]}))
    (tmp_path / "takes" / "tv_spec.toml").write_text('name = "tv_spec"\nsource_dir = "../src/lucas-luisa-jnj"\nalignment = "../clock/alignment.json"\n'
                                                     'window = [16.8, 213.5]\n\n[dancers.lead]\nstature_m = 1.75\nshoe_sole_m = 0.025\n\n'
                                                     '[dancers.follow]\nstature_m = 1.63\nshoe_sole_m = 0.06\n')
    monkeypatch.setattr(take_mod, "PACKAGE_ROOT", tmp_path)
    sp = ci.spec("tv_spec")
    assert sp.take == "tv_spec" and sp.capture == "lucas-luisa-jnj" and sp.work == tmp_path / "work" / "tv_spec" and sp.orig == "orig"
    assert sp.cams == ("01", "02") and sp.videos["02"] == (tmp_path / "src" / "lucas-luisa-jnj" / "Tripod 02.mp4").resolve()
    assert sp.clocks["01"] == {"slope": 0.9999163457150201, "offset": 12.12290391571289, "offset_source": "audio clock (../clock/alignment.json)"}
    assert sp.ankle_y == (0.0932, 0.1236)
    with pytest.raises(FileNotFoundError):                                                     # --clocks sync: the take's sync/sync.json
        ci.spec("tv_spec", "sync")


# ------------------------------------------------------------------------------------------------ carlos_aline's default run is unchanged


def test_prepare_carlos_default_route(tmp_path, monkeypatch):
    """C3's run: clocks from sync/sync.json, C2's tracks carry frame_index (pts = the video's pts there), pose tables from the podout files,
    cameras/c2_orig/ (no written.json), inputs.json without gates."""
    work = tmp_path / "carlos_aline"
    (work / "cameras").mkdir(parents=True)
    (work / "sync").mkdir()
    pod = work / "room_work" / "podout"
    pod.mkdir(parents=True)
    monkeypatch.setattr(ci, "WORK", work)
    monkeypatch.setattr(ci, "PEOPLE", work / "people")
    monkeypatch.setattr(ci, "SOURCE", tmp_path / "dances" / "Carlos-Aline")
    monkeypatch.setattr(ci, "POSE_SRC", {"01": pod / "pose_A.npz", "02": pod / "pose_B.npz"})
    (work / "sync" / "sync.json").write_text(json.dumps({"cameras": [
        {"id": "01", "slope": 1.0, "offset": 0.0, "offset_source": "master"},
        {"id": "02", "slope": 1.0, "offset": 14.815753692354448, "offset_source": "GCC-PHAT"}]}))
    sp = ci.spec()
    assert (sp.take, sp.capture, sp.work, sp.orig) == ("carlos_aline", "Carlos-Aline", work, "c2_orig")
    assert sp.clocks["02"]["offset_source"] == "GCC-PHAT" and sp.ankle_y == ci.ankle_heights({})
    pts_all = {"01": np.arange(300) / 23.976, "02": np.arange(200) / 30.0}
    for cam, sel, off, src in (("01", np.arange(10, 60), 0.0, "pose_A"), ("02", np.arange(5, 40), 14.815753692354448, "pose_B")):
        np.savez(work / "cameras" / f"{cam}.npz", t=pts_all[cam][sel] + off, frame_index=sel, K=np.repeat(np.eye(3)[None], len(sel), 0))
        np.savez(pod / f"{src}.npz", frame=np.array([3, 3, 7]), conf=np.array([0.5, 0.9, 0.7]), kp=np.ones((3, 17, 3)), box=np.ones((3, 4)),
                 rgb=np.ones((3, 3)), n_frames=np.array(len(pts_all[cam])))
    by_name = {ci.VIDEO["01"]: "01", ci.VIDEO["02"]: "02"}
    seen = []

    def pts_fn(path, cache):
        seen.append((path, cache))
        return pts_all[by_name[path.name]]

    def probe_fn(path):
        return {"width": 1920, "height": 1080 if by_name[path.name] == "01" else 906, "avg_fps": 23.976, "duration": 202.08}
    res = ci.prepare(pts_fn=pts_fn, probe_fn=probe_fn)
    assert seen[0] == (tmp_path / "dances" / "Carlos-Aline" / ci.VIDEO["01"], work / "frames" / "01_pts.json")
    rc = json.loads((work / "cameras" / "recap_cams.json").read_text())
    common = {"rotation": 0, "avg_fps": 23.976, "duration": 202.08, "slope": 1.0, "accepted": True}
    assert rc == {"take": "carlos_aline", "capture": "Carlos-Aline", "cams": {
        "01": {"id": "01", "file": ci.VIDEO["01"], "width": 1920, "height": 1080, **common, "offset": 0.0, "offset_source": "master"},
        "02": {"id": "02", "file": ci.VIDEO["02"], "width": 1920, "height": 906, **common, "offset": 14.815753692354448, "offset_source": "GCC-PHAT"}}}
    assert list(rc["cams"]["01"]) == ["id", "file", "width", "height", "rotation", "avg_fps", "duration", "slope", "offset", "accepted",
                                      "offset_source"]
    z = np.load(work / "cameras" / "02.npz")
    assert list(z.files) == ["t", "frame_index", "K", "pts", "native_size", "work_size"] and np.array_equal(z["pts"], pts_all["02"][5:40])
    assert z["native_size"].tolist() == [1920, 906] and (work / "cameras" / "c2_orig" / "02.npz").exists() and not (work / "cameras" / "orig").exists()
    assert sorted(p.name for p in (work / "cameras" / "c2_orig").iterdir()) == ["01.npz", "02.npz"]
    pt = np.load(work / "people" / "pose_tables" / "01.npz")
    assert pt["n"][3] == 2 and pt["n"].shape == (300,) and set(pt.files) == {"kp", "n", "box", "det_conf", "rgb"}
    assert res == json.loads((work / "people" / "inputs.json").read_text())
    assert res["01"] == {"frames": 50, "t": [round(10 / 23.976, 3), round(59 / 23.976, 3)], "clock_vs_pts_max_dev_s": 0.0, "pose_frames": 300,
                         "max_persons": 2}
    assert all("gates" not in r for r in res.values())


# ------------------------------------------------------------------------------------------------ jl_export.reliability_meta


MONO_TEXT = ("one handheld phone (1280x720, no zoom step): the distance of the couple from the camera is +-10 percent (focal length), the pose "
             "relative to the pelvis, the left-right position and the height above the floor are good; hands are the raw fit's")


def _take(name, n_src, cfg=None, window=(10.0, 100.0)):
    srcs = [Source(f"{i + 1:02d}", Path(f"/v/{i + 1}.mp4"), 1.0, 0.0) for i in range(n_src)]
    return Take(name, srcs, window, window, 2.0, {"name": name, **(cfg or {})})


def _fit(work: Path, folder: str, absent: slice):
    (work / folder).mkdir(parents=True, exist_ok=True)
    t = 10.0 + np.arange(100) / 30.0
    a = np.zeros(100, bool)
    a[absent] = True
    np.savez(work / folder / "lead.npz", t=t, absent=a)


def _work(tmp_path, name="tv_reliab_take"):
    work = tmp_path / name
    (work / "cameras").mkdir(parents=True)
    np.savez(work / "cameras" / "01.npz", native_size=np.array([1280, 720]), zoom_2x=np.zeros(10, bool), K=np.repeat(np.eye(3)[None], 10, 0))
    _fit(work, "smplx_rf2_noslip", slice(10, 20))
    _fit(work, "smplx", slice(40, 45))
    return work


def test_reliability_meta_one_view_unchanged(tmp_path):
    work = _work(tmp_path)
    old = {"camera": MONO_TEXT, "provisional": True, "low_confidence": lc.intervals(work), "monocular": True}
    assert old["low_confidence"][0]["from"] == pytest.approx(10.333, abs=1e-3)
    assert JE.reliability_meta(work, _take(work.name, 1)) == old
    # two sources but [stitch] mono over the whole window: the single-video route (lucas_luiza_h) keeps the monocular text
    assert JE.reliability_meta(work, _take(work.name, 2, {"stitch": {"mono": [[10.0, 100.0]]}})) == old
    # two (or eight) sources fitted from one camera: a chunk take's [segment] cams (lucas_luiza_h_gNNNN), [derived] cams (larissa_kadu_38_mono06)
    assert JE.reliability_meta(work, _take(work.name, 2, {"segment": {"parent": "x", "cams": ["01"]}})) == old
    assert JE.reliability_meta(work, _take(work.name, 8, {"derived": {"cams": ["06"]}})) == old


def test_reliability_meta_two_views(tmp_path):
    work = _work(tmp_path)
    cfg = {"cameras": {"01": {"mount": "handheld"}, "02": {"mount": "tripod"}}}
    m = JE.reliability_meta(work, _take(work.name, 2, cfg))
    assert m["monocular"] is False and m["provisional"] is True and m["views"] == ["01", "02"]
    assert m["camera"] == "two views triangulated (handheld 01 + tripod 02)" and "single_view_s" not in m
    assert [iv["from"] for iv in m["low_confidence"]] == [round(10 + 40 / 30, 3)]              # the exported fit (smplx/ here), not smplx_rf2_noslip
    cfg["export"] = {"camera_names": {"01": "Steadicam"}}
    cfg["stitch"] = {"mono": [[10.0, 14.8]]}
    m = JE.reliability_meta(work, _take(work.name, 2, cfg))
    assert m["camera"] == "two views triangulated (Steadicam 01 + tripod 02)" and m["single_view_s"] == [[10.0, 14.8]] and m["monocular"] is False
    assert JE.reliability_meta(work, _take(work.name, 3))["camera"] == "three views triangulated (camera 01 + camera 02 + camera 03)"
    # the fitted camera set, when the TOML names one, gives the views (in its order)
    m = JE.reliability_meta(work, _take(work.name, 3, {**cfg, "segment": {"cams": ["02", "01"]}}))
    assert m["views"] == ["02", "01"] and m["camera"] == "two views triangulated (tripod 02 + Steadicam 01)"


def test_reliability_meta_two_views_with_plain_cameras_key(tmp_path, monkeypatch):
    """[cameras] x_cam = "02" and map = {...} (cameras.pipeline's X axis and map settings) beside the camera tables: the low-confidence intervals still
    read the zoom steps of the camera tables (jl_lowconf's zoom-step loop raised AttributeError on the str before it used camera_types.cam_tables)."""
    work = _work(tmp_path)
    (tmp_path / "takes").mkdir()
    (tmp_path / "takes" / f"{work.name}.toml").write_text('[cameras]\nx_cam = "02"\nmap = { win_step_s = 2.0 }\n\n[cameras."02"]\nmount = "tripod"\n'
                                                          'zoom_events = [[50.0, 50.4, 1.16]]\n')
    monkeypatch.setattr(lc, "PACKAGE_ROOT", tmp_path)
    m = JE.reliability_meta(work, _take(work.name, 2, {"cameras": {"x_cam": "02", "map": {"win_step_s": 2.0}, "02": {"mount": "tripod"}}}))
    assert m["monocular"] is False and m["camera"] == "two views triangulated (camera 01 + tripod 02)"
    assert [(iv["from"], iv["to"], iv["level"]) for iv in m["low_confidence"] if iv["role"] == "both"] == [(49.5, 50.9, "zoom_step")]
