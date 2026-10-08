"""camera_video.py: the Unity camera conversion, the .hmj frame container, undistortion, track merging.

    ..\\.venv-nopo4d\\Scripts\\python.exe -m pytest tests/test_camera_video.py -q      (from dancecap/)
"""
from __future__ import annotations

import cv2
import numpy as np

from dancecap import camera_video as cv
from dancecap import camera_video_check as chk


def _rot(rng):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


def test_unity_camera_reproduces_the_opencv_projection():
    """A world point projected with K, R, t (OpenCV) lands on the same pixel through the Unity-coordinate track
    (x, y, -z flip, optical axis + image-up, left-handed right = up x forward)."""
    rng = np.random.default_rng(1)
    W, H, f = 1080, 1920, 1147.0
    for _ in range(20):
        R = _rot(rng)
        # keep the camera looking roughly +z so the points are in front
        R = R @ np.diag([1, 1, 1.0])
        t = rng.normal(size=3)
        X = rng.normal(size=(8, 3)) * 0.5 + R.T @ (np.array([0, 0, 4.0]) - t)
        xc = X @ R.T + t
        cam = (xc[:, :2] / xc[:, 2:3]) * f + np.array([(W - 1) / 2, (H - 1) / 2])
        pos, fwd, up = cv.cam_pose_unity(R[None], t[None])
        vfov = float(np.degrees(2 * np.arctan(H / (2 * f))))
        flip = np.array([1.0, 1.0, -1.0])
        px = chk.project(X * flip, pos[0], fwd[0], up[0], vfov, (W, H))
        ok = xc[:, 2] > 0.1
        assert ok.any()
        assert np.abs(px[ok] - cam[ok]).max() < 1e-6


def test_hmj_roundtrip(tmp_path):
    p = tmp_path / "x.hmj"
    wr = cv.HmjWriter(p, 64, 48)
    frames = []
    for i in range(5):
        img = np.full((48, 64, 3), 20 * i, np.uint8)
        frames.append(img)
        wr.add(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes(), 10.0 + i / 30.0)
    wr.close()
    recs, w, h = cv.read_hmj(p)
    assert (w, h, len(recs)) == (64, 48, 5)
    assert abs(recs[3][2] - (10.0 + 3 / 30.0)) < 1e-12
    for i, r in enumerate(recs):
        assert abs(float(cv.hmj_frame(p, r).mean()) - 20 * i) < 3
    assert p.stat().st_size == recs[-1][0] + recs[-1][1] + len(recs) * cv.RECORD.size + cv.TRAILER.size


def test_undistort_moves_a_distorted_point_back_to_its_pinhole_pixel():
    w, h, f, k = 400, 600, 500.0, 0.05
    cx, cy = (w - 1) / 2, (h - 1) / 2
    xu, yu = 120.0, -200.0                                   # pinhole pixel offsets from the centre
    r2 = (xu / f) ** 2 + (yu / f) ** 2
    src = (cx + xu * (1 + k * r2), cy + yu * (1 + k * r2))   # where the distorted image holds that scene point
    img = np.zeros((h, w, 3), np.uint8)
    cv2.circle(img, (int(round(src[0])), int(round(src[1]))), 3, (255, 255, 255), -1)
    out = cv.Undistorter(w, h)(img, f, k)
    ys, xs = np.nonzero(out[..., 0] > 128)
    assert abs(xs.mean() - (cx + xu)) < 1.0 and abs(ys.mean() - (cy + yu)) < 1.0


def test_fit_size_is_even_and_never_upscales():
    assert cv.fit_size(1080, 1920, 1280) == (720, 1280)
    assert cv.fit_size(480, 848, 1280) == (480, 848)
    w, h = cv.fit_size(480, 639, 256)
    assert w % 2 == 0 and h % 2 == 0


def _track(raw0, n, delta, take):
    raw = raw0 + np.arange(n) / 30.0
    z = np.zeros((n, 3))
    return cv.CamTrack("01", raw, raw + delta, np.full(n, delta), z, z + [0, 0, 1], z + [0, 1, 0], np.full(n, 50.0),
                       np.full(n, 1000.0), np.zeros(n), np.ones(n, bool), (480, 864), [take])


def test_merge_tracks_later_wins_in_overlap():
    a, b = _track(0.0, 300, 0.01, "a"), _track(5.0, 300, 0.05, "b")
    m = cv.merge_tracks([a, b])
    assert np.all(np.diff(m.raw_t) > 0)
    assert np.allclose(m.delta[m.raw_t < 4.99], 0.01) and np.allclose(m.delta[m.raw_t >= 5.0 - 1e-9], 0.05)
    assert m.takes == ["a", "b"]
    assert m.raw_t[0] == 0.0 and m.raw_t[-1] == b.raw_t[-1]


# ---------------------------------------------------------------------------------------------------- pose policy (gaps)

def _cov(t, ok, fixed=False):
    return cv.PoseCoverage(t, ok, cv.MAX_UNLOCALISED_FIXED_S if fixed else cv.MAX_UNLOCALISED_S)


def test_unlocalised_run_is_bridged_not_dropped():
    """Phone 06 in the first version: 17 frames (0.5 s) not localised in the middle of the POV shot left a 0.6000023 s hole
    between its localised neighbours, greater than the 0.6f limit in float32, and the video blinked out. The run is bridged."""
    t = 54.0 + np.arange(0, 3.0, 1 / 30)
    ok = np.ones(len(t), bool)
    ok[(t > 54.74) & (t < 55.29)] = False
    runs = cv.unlocalised_runs(t, ok)
    assert len(runs) == 1 and 0.55 < runs[0][2] < 0.65
    q = np.arange(t[0], t[-1], 1 / 30)
    assert _cov(t, ok).available(q).all()
    # the first version dropped the unlocalised rows: its localised neighbours were this far apart, at the edge of the 0.6 s
    # limit it applied (in float32 0.6000023 > 0.6f), which is why the picture vanished mid-shot
    assert np.diff(t[ok]).max() > 0.55


def test_long_unlocalised_runs_are_trusted_only_for_fixed_phones():
    t = np.arange(0, 20, 1 / 30)
    ok = np.ones(len(t), bool)
    ok[(t > 5) & (t < 9)] = False                       # a 4 s run: past the 3 s handheld limit, inside the 8 s fixed limit
    q = np.array([4.0, 7.0, 10.0])
    assert _cov(t, ok).available(q).tolist() == [True, False, True]
    assert _cov(t, ok, fixed=True).available(q).all()
    ok2 = np.ones(len(t), bool)
    ok2[(t > 5) & (t < 15)] = False                     # 10 s: no phone is trusted that long
    assert _cov(t, ok2, fixed=True).available(np.array([10.0])).tolist() == [False]


def test_rows_further_apart_than_a_second_mean_the_phone_was_not_recording():
    t = np.concatenate([np.arange(0, 5, 1 / 30), np.arange(7, 12, 1 / 30)])
    c = _cov(t, np.ones(len(t), bool))
    assert c.available(np.array([4.0, 6.0, 8.0])).tolist() == [True, False, True]
    # a dropped-frame hole of 0.6 s (rows missing, not unlocalised) is bridged, float32 rounding or not
    t2 = np.concatenate([np.arange(0, 5, 1 / 30), np.arange(5.6000023, 10, 1 / 30)])
    assert _cov(t2, np.ones(len(t2), bool)).available(np.arange(0.1, 9.9, 1 / 30)).all()


def test_end_poses_are_held_within_the_tolerance_only():
    t = np.arange(10, 20, 1 / 30)
    c = _cov(t, np.ones(len(t), bool))
    assert c.available(np.array([9.6, 20.3])).all()
    assert not c.available(np.array([9.4, 20.6])).any()


def test_all_unlocalised_keeps_the_phone():
    t = np.arange(0, 5, 1 / 30)
    assert _cov(t, np.zeros(len(t), bool)).available(np.array([2.0])).tolist() == [True]


def test_despike_removes_a_spike_and_keeps_a_zoom():
    t = np.arange(0, 20, 1 / 30)
    v = np.full(len(t), 54.0)
    v[(t > 5) & (t < 5.25)] = 70.0                                    # 7 samples, 16 deg: an estimation spike
    out, n = cv.despike_vfov(t, v)
    assert n >= 7 and np.allclose(out, 54.0)
    ramp = 40 + 20 * np.clip((t - 8) / 4, 0, 1)                      # a real zoom: a ramp 40 -> 60 over 4 s ...
    out2, n2 = cv.despike_vfov(t, ramp)
    assert n2 == 0 and np.allclose(out2, ramp)
    step = np.where(t < 10, 40.0, 60.0)                              # ... and a step that stays
    out3, n3 = cv.despike_vfov(t, step)
    assert n3 == 0 and np.allclose(out3, step)


def test_track_policy_marks_a_phone_that_never_moves():
    pos = np.tile([1.0, 2.0, 3.0], (50, 1))
    ok = np.ones(50, bool)
    p = cv.track_policy(np.arange(50) / 30, pos, ok)
    assert p["fixed_position"] and p["max_unlocalised_s"] == cv.MAX_UNLOCALISED_FIXED_S
    pos2 = pos + np.linspace(0, 0.5, 50)[:, None]
    p2 = cv.track_policy(np.arange(50) / 30, pos2, ok)
    assert not p2["fixed_position"] and p2["max_unlocalised_s"] == cv.MAX_UNLOCALISED_S


def test_direction_pov_windows_for_a_fly_shot_and_a_tour(tmp_path):
    d = {"capture": "x", "segments": [
        {"id": "fly", "film": [10.0, 23.0], "dance": [50.0, 63.0], "speed": 1.0,
         "camera": {"mode": "fly_to_camera", "camera": "06", "fly_s": 1.5, "fade_in_s": 0.8, "hold_s": 6.5, "fade_out_s": 0.8}},
        {"id": "tour", "film": [30.0, 40.0], "dance": [70.0, 80.0],
         "camera": {"mode": "camera_tour", "cameras": [{"camera": "04", "dance_t": [71.0, 74.0]}]}},
        {"id": "orbit", "film": [40.0, 45.0], "dance": [80.0, 85.0], "camera": {"mode": "orbit"}}]}
    p = tmp_path / "direction_x.json"
    p.write_text(__import__("json").dumps(d), encoding="utf-8")
    w = cv.direction_pov_windows([p])
    assert [x["camera"] for x in w] == ["06", "04"]
    fly = w[0]
    assert abs(fly["pose"][0] - (50.0 - 0.3)) < 1e-6 and abs(fly["pose"][1] - (63.0 + 0.3)) < 1e-6
    assert abs(fly["video"][0] - (51.5 - 0.3)) < 1e-6 and abs(fly["video"][1] - (51.5 + 8.1 + 0.3)) < 1e-6
    assert w[1]["pose"] == w[1]["video"] == [71.0, 74.0]


def test_coverage_report_flags_missing_pose_and_video(tmp_path):
    """The regression sweep: a pose at every 1/30 s of the shot and a video frame at every step of the video window."""
    import json
    t = np.arange(40, 70, 1 / 30)
    ok = np.ones(len(t), bool)
    ok[(t > 54.74) & (t < 55.29)] = False                            # the phone-06 case: must pass
    tj = {"id": "06", "t": t.tolist(), "ok": ok.astype(int).tolist(), "max_unlocalised_s": 3.0, "max_sample_gap_s": 1.0}
    (tmp_path / "06.track.json").write_text(json.dumps(tj), encoding="utf-8")
    wr = cv.HmjWriter(tmp_path / "06.full.hmj", 8, 8)
    jpg = cv2.imencode(".jpg", np.zeros((8, 8, 3), np.uint8))[1].tobytes()
    for x in t:
        wr.add(jpg, float(x))
    wr.close()
    win = {"camera": "06", "segment": "fly", "file": "d.json", "pose": [48.7, 62.3], "video": [50.2, 58.9]}
    rep = cv.coverage_report(tmp_path, [win])[0]
    assert rep["ok"] and not rep["pose_missing"] and not rep["video_missing"]
    assert rep["unlocalised_runs_in_window"]
    # a phone that stopped recording for 2 s inside the video window: both are reported
    t2 = np.concatenate([t[t < 54], t[t > 56]])
    tj2 = dict(tj, t=t2.tolist(), ok=[1] * len(t2))
    (tmp_path / "06.track.json").write_text(json.dumps(tj2), encoding="utf-8")
    wr = cv.HmjWriter(tmp_path / "06.full.hmj", 8, 8)
    for x in t2:
        wr.add(jpg, float(x))
    wr.close()
    bad = cv.coverage_report(tmp_path, [win])[0]
    assert not bad["ok"] and bad["pose_missing"] and bad["video_missing"]
    assert cv.coverage_report(tmp_path, [dict(win, camera="09")])[0]["ok"] is False


def test_export_refuses_a_take_without_sync_offsets(monkeypatch):
    monkeypatch.setattr(cv.md, "read_sync_offsets", lambda work: {})
    monkeypatch.setitem(cv.OPTIONS, "allow_unsynced", False)
    try:
        cv.require_sync("no_such_take")
    except SystemExit as e:
        assert "--allow-unsynced" in str(e) and "offsets.json" in str(e)
    else:
        raise AssertionError("an unsynced export must be an error")
    monkeypatch.setitem(cv.OPTIONS, "allow_unsynced", True)
    assert cv.require_sync("no_such_take") == {}
