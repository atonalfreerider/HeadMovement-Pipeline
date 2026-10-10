"""dancecap.recap_pov: bad stretches, per-chunk video offsets, teacher boxes, which phone when, the crop (synthetic data only).

    .venv-nopo4d/Scripts/python.exe -m pytest dancecap/tests/test_recap_pov.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dancecap import camera_video as cv  # noqa: E402
from dancecap import recap_pov as rp  # noqa: E402


def phone(cam, t_end=60.0, cover=1.0, h=0.4, size=(1080, 1920), pose=True, cx=0.5, cy=0.5):
    t = np.arange(0, t_end, 1 / 30)
    ok = np.random.default_rng(1).random(len(t)) < cover
    w = 0.3
    box = np.stack([np.full(len(t), cx - w / 2), np.full(len(t), cy - h / 2), np.full(len(t), cx + w / 2), np.full(len(t), cy + h / 2)], axis=1)
    box = np.where(ok[:, None], box, np.nan)
    grid = np.arange(0, t_end, 0.1)
    return rp.PhoneData(cam, t, ok, box, size, grid, np.full(len(grid), pose))


def test_bad_stretches_are_padded_and_merge_the_qa_file(tmp_path):
    f = tmp_path / "bad.json"
    f.write_text(json.dumps({"g7699": {"02": [[256.0, 261.0]], "05": [[296.0, 300.0]]}}))
    bad = rp.bad_stretches(bad_file=f)
    assert rp.overlaps_bad(bad["01"], 20.0, 22.0) and rp.overlaps_bad(bad["01"], 331.0, 332.0)
    assert not rp.overlaps_bad(bad["01"], 40.0, 50.0)
    assert rp.overlaps_bad(bad["02"], 260.5, 262.0) and rp.overlaps_bad(bad["02"], 250.0, 243.0 + 1.0)
    assert rp.overlaps_bad(bad["05"], 298.0, 299.0) and rp.overlaps_bad(bad["05"], 205.0, 206.0)
    assert rp.overlaps_bad(bad["06"], 160.0, 170.0) and not rp.overlaps_bad(bad["06"], 100.0, 110.0)
    # the disputed "02 wrong in 0-35 s" is NOT in the table
    assert not rp.overlaps_bad(bad["02"], 5.0, 30.0)
    # padding: one second each side
    assert rp.overlaps_bad(bad["01"], 8.2, 8.5) and not rp.overlaps_bad(bad["01"], 7.0, 7.9)


def test_chunk_offsets_sit_at_the_chunk_centres_and_interpolate():
    pw = rp.piecewise_offsets_from([
        {"centre": 30.0, "offsets": {"00": {"delta": 0.113, "use": True}, "01": {"delta": 0.0}}},
        {"centre": 90.0, "offsets": {"00": {"delta": 0.125}, "01": {"delta": 0.0}, "03": {"delta": 0.002}}},
        {"centre": 150.0, "offsets": {"00": {"delta": 0.105, "use": False}, "01": {"delta": 0.0}}}])
    assert pw["00"] == [[30.0, 0.113], [90.0, 0.125]]          # use: false is skipped
    assert abs(rp.offset_at(pw["00"], 60.0) - 0.119) < 1e-9
    assert rp.offset_at(pw["00"], 0.0) == 0.113 and rp.offset_at(pw["00"], 500.0) == 0.125     # flat outside
    assert list(pw["03"]) == [[90.0, 0.002]]


def test_camera_video_applies_piecewise_offsets_and_several_clips_per_phone():
    t = np.array([0.0, 30.0, 60.0, 90.0])
    out = cv.apply_offset(t, {"piecewise": [[0.0, 0.1], [60.0, 0.2]]})
    assert np.allclose(out, [0.1, 30.15, 60.2, 90.2])
    assert np.allclose(cv.apply_offset(t, {"delta": 0.05}), t + 0.05)
    assert cv.merge_ranges([(10, 14), (15, 20), (30, 33), (12, 13)], gap=2.0) == [(10, 20), (30, 33)]
    clips = cv._clip_args([["06", "10", "14"], ["06", "40", "45"], ["01", "3", "8"]])
    assert clips == {"06": [(10.0, 14.0), (40.0, 45.0)], "01": [(3.0, 8.0)]}


def test_union_boxes_take_both_teachers_or_the_merged_embrace():
    box = np.array([[[100, 100, 200, 400], [180, 120, 300, 420]],        # both
                    [[100, 100, 300, 420], [np.nan] * 4],                 # merged: one box holds both
                    [[100, 100, 200, 400], [np.nan] * 4],                 # one teacher missing, not merged
                    [[100, 100, 200, 400], [900, 120, 1100, 420]]], float)  # one outside the frame
    conf = np.array([[0.9, 0.9], [0.45, 0.0], [0.9, 0.0], [0.9, 0.9]])
    merged = np.array([[False, False], [True, False], [False, False], [False, False]])
    ok, u = rp.union_boxes(box, conf, (1000, 1000), merged=merged)
    assert ok.tolist() == [True, True, False, False]
    assert np.allclose(u[0], [0.1, 0.1, 0.3, 0.42]) and np.allclose(u[1], [0.1, 0.1, 0.3, 0.42]) and np.isnan(u[2]).all()


def test_a_phone_is_usable_when_it_sees_both_teachers_has_a_pose_and_is_not_in_a_bad_stretch():
    cfg = rp.PovCfg()
    p = phone("01")
    assert rp.window_quality(p, 10, 14, [], cfg)["strong"]
    assert rp.window_quality(p, 10, 14, [(12, 20, "bad")], cfg) is None
    assert rp.window_quality(phone("01", pose=False), 10, 14, [], cfg) is None
    assert rp.window_quality(phone("01", cover=0.3), 10, 14, [], cfg) is None
    q = rp.window_quality(phone("06", cover=0.7), 10, 14, [], cfg)
    assert q is not None and not q["strong"]                    # 06 sees both in about half its frames: accepted only when nothing better exists


def test_the_phones_circle_never_the_same_twice_in_a_row_and_each_appears():
    phones = {c: phone(c) for c in ("00", "01", "02", "03")}
    wins = [(i * 5.0, i * 5.0 + 4.0) for i in range(8)]
    got = rp.choose_phones(wins, phones, {})
    cams = [g["camera"] for g in got]
    assert all(a != b for a, b in zip(cams, cams[1:]))
    assert set(cams) == {"00", "01", "02", "03"} and all(cams.count(c) == 2 for c in set(cams))


def test_a_phone_in_its_bad_stretch_is_skipped_and_a_window_nobody_sees_is_none():
    phones = {"00": phone("00"), "01": phone("01")}
    bad = {"01": [(0.0, 100.0, "x")]}
    assert all(g["camera"] == "00" for g in rp.choose_phones([(5, 9), (20, 24)], phones, bad))
    assert rp.choose_phones([(5, 9)], phones, {"00": [(0, 100, "x")], "01": [(0, 100, "x")]}) == [None]
    # a window that carries a 3D arrow takes a directly localised phone when one qualifies
    p5 = {"05": phone("05", h=0.6), "02": phone("02", h=0.3)}
    assert rp.choose_phones([(5, 9, rp.MAIN_PHONES)], p5, {})[0]["camera"] == "02"


def test_the_crop_frames_the_teachers_high_in_the_view_and_stays_inside_the_frame():
    ph = phone("00", h=0.3, size=(480, 864), cx=0.5, cy=0.55)
    c = rp.crop_track(ph, 10.0, 14.0)
    assert len(c["cx"]) == len(c["cy"]) == len(c["zoom"]) == 41 and c["t0"] == 10.0 and c["dt"] == 0.1
    z = np.array(c["zoom"])
    # 0.5 / 0.3 = 1.67, capped by the source resolution (864 px high into 1920: 3.4 * 864 / 1920 = 1.53)
    assert 1.4 < z.min() <= z.max() < 1.6
    vh = 1 / z
    # the view stays inside the frame: its centre is at least half its height from the edges
    cy_frame = 0.5 - np.array(c["cy"]) / 2
    assert (cy_frame - vh / 2 >= -1e-6).all() and (cy_frame + vh / 2 <= 1 + 1e-6).all()
    assert c["inside"] >= 0.9
    # the teachers' box centre (0.55) sits 40 % from the top of the view
    top = cy_frame - vh / 2
    assert np.allclose((0.55 - top) / vh, 0.4, atol=0.03) or c["inside"] < 1.0


def test_a_phone_that_already_fills_with_the_teachers_zooms_only_to_make_room_and_never_past_the_fill_limit():
    ph = phone("01", h=0.52, cy=0.5)
    c = rp.crop_track(ph, 5.0, 9.0)
    z = np.array(c["zoom"])
    assert 1.0 <= z.min() and z.max() <= 1.3                # centred teachers: a little zoom makes room to lift them to 40 % from the top
    assert (0.52 * z <= rp.PovCfg().max_fill + 0.01).all()
    low = rp.crop_track(phone("01", h=0.4, cy=0.62), 5.0, 9.0)       # the teachers stand low in the frame: the view rises with a zoom
    assert min(low["zoom"]) > 1.15 and low["inside"] >= 0.9


def test_a_landscape_phones_9_16_strip_must_hold_both_teachers():
    cfg = rp.PovCfg()
    wide = phone("02", size=(848, 480))                    # box width 0.3 of a landscape frame: the 9:16 strip of it is 0.32 wide
    assert rp.window_quality(wide, 5, 9, [], cfg) is not None
    t = np.arange(0, 60, 1 / 30)
    box = np.tile(np.array([0.25, 0.1, 0.75, 0.9]), (len(t), 1))        # the teachers stand far apart: half the landscape frame
    far = rp.PhoneData("02", t, np.ones(len(t), bool), box, (848, 480), np.arange(0, 60, 0.1), np.ones(600, bool))
    assert rp.window_quality(far, 5, 9, [], cfg) is None


def test_clips_come_from_the_directions_pov_shots():
    d = {"segments": [
        {"id": "a", "dance": [10.0, 14.0], "camera": {"mode": "camera_tour", "cameras": [{"camera": "05", "dance_t": [10.0, 14.0]}]}},
        {"id": "b", "dance": [20.0, 25.0], "camera": {"mode": "fly_to_camera", "camera": "05"}},
        {"id": "c", "dance": [30.0, 33.0], "camera": {"mode": "orbit"}}]}
    assert rp.clips_from_direction(d, pad=0.5) == {"05": [(9.5, 14.5), (19.5, 25.5)]}
