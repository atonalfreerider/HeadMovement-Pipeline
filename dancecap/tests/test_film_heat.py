"""dancecap.film_heat: the auto-directed film of a heat (chapters from the song structure, a camera that stays in front of the wall, an overhead
now and then, the capture's own soundtrack as the mix) and its library entry.

    .venv-nopo4d/Scripts/python.exe -m pytest dancecap/tests/test_film_heat.py -q
"""
from __future__ import annotations

import json
import struct
import sys
import wave
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dancecap import film_heat as fh  # noqa: E402
from dancecap import film_library as fl  # noqa: E402

SONGS = {"timeline": [{"from_s": 0.0, "to_s": 13.31, "state": "none", "songs": []},
                      {"from_s": 13.31, "to_s": 113.47, "state": "kojo", "songs": ["kojo"]},
                      {"from_s": 113.47, "to_s": 122.25, "state": "mix", "songs": ["kojo", "guetta"]},
                      {"from_s": 122.25, "to_s": 194.25, "state": "guetta", "songs": ["guetta"]},
                      {"from_s": 194.25, "to_s": 202.083, "state": "none", "songs": []}],
         "no_beat": [[0.0, 13.31], [194.25, 202.083]]}
TITLES = {"entrance": "Entrance", "kojo": "First song", "mix": "DJ mix", "guetta": "Second song", "end": "Bow"}


def _direction(**kw):
    args = dict(name="heat", title="Heat", capture="99_Heat", take="heat", songs=SONGS, dance_range=[3.6, 197.9], film_end=198.0, titles=TITLES)
    args.update(kw)
    return fh.build_direction(**args)


def test_chapters_follow_the_song_structure_and_the_last_runs_to_the_film_end():
    ch = fh.chapters_from_songs(SONGS, TITLES, 198.0)
    assert [c["title"] for c in ch] == ["Entrance", "First song", "DJ mix", "Second song", "Bow"]
    assert [c["start"] for c in ch] == [0.0, 13.31, 113.47, 122.25, 194.25]
    assert ch[-1]["end"] == 198.0 and all(a["end"] == b["start"] for a, b in zip(ch, ch[1:]))


def test_a_song_without_a_title_is_still_a_chapter():
    ch = fh.chapters_from_songs(SONGS, {}, 198.0)
    assert [c["title"] for c in ch][:2] == ["Entrance and announcement", "Kojo"]


def test_the_timeline_is_required():
    with pytest.raises(fh.HeatError, match="timeline"):
        fh.chapters_from_songs({}, TITLES, 100.0)


def test_shots_cover_the_film_and_go_overhead_now_and_then_but_not_at_the_ends():
    shots = fh.shot_plan(198.0, fh.DEFAULTS)
    assert shots[0]["start"] == 0.0 and shots[-1]["end"] == 198.0
    assert all(a["end"] == b["start"] for a, b in zip(shots, shots[1:]))
    overheads = [s for s in shots if s["kind"] == "overhead"]
    assert 4 <= len(overheads) <= 8 and all(s["end"] - s["start"] == fh.DEFAULTS["overhead_s"] for s in overheads)
    assert shots[0]["kind"] == "orbit" and shots[-1]["kind"] == "orbit"
    assert all(s["start"] >= fh.DEFAULTS["no_overhead_before_s"] for s in overheads)


def test_orbit_arcs_stay_in_front_of_the_wall_and_alternate_direction():
    cfg = fh.DEFAULTS
    front = cfg["front_azimuth_deg"]
    for i in range(8):
        cam = fh.orbit_camera(i, 22.0, cfg)
        end = cam["azimuth_deg"] + cam["drift_deg_s"] * 22.0
        for az in (cam["azimuth_deg"], end):
            assert abs(az - front) <= 70.0, (i, az)       # never behind the dancers' side of the room (the wall is at the other side)
        assert (cam["drift_deg_s"] > 0) == (i % 2 == 0)
        assert cam["fit_span"] and cam["radius_m"] <= 3.0   # both dancers framed, close


def test_direction_has_no_cuts_no_captions_and_the_beat_counter_only_while_a_song_plays():
    d = _direction(beat_start_s=13.31, beat_end_s=194.25)
    segs = d["segments"]
    assert segs[0]["film"][0] == 0.0 and segs[-1]["film"][1] == 198.0
    for a, b in zip(segs, segs[1:]):
        assert a["film"][1] == b["film"][0] and a["dance"][1] == b["dance"][0]      # no cuts: film time = dance time
        assert a["film"] == a["dance"] and a["speed"] == 1.0
    for s in segs:
        mid = sum(s["film"]) / 2
        assert s["layers"]["beat_counter"] == (13.31 <= mid <= 194.25), s["id"]
        assert s["layers"]["room"] is True and s["layers"]["skeleton_lines"] == 0.0 and s["annotations"] == []
    assert d["timeline"]["film_duration_s"] == 198.0 and "captions" not in d


def test_chapter_anchors_are_segments_that_start_at_the_chapter():
    d = _direction()
    by_id = {s["id"]: s for s in d["segments"]}
    assert [c["title"] for c in d["chapters"]] == list(TITLES.values())
    for c in d["chapters"]:
        assert by_id[c["segment"]]["film"][0] == pytest.approx(c["film_s"])
        assert by_id[c["segment"]]["chapter"] == c["title"]


def test_the_library_reads_the_chapters_of_the_generated_direction(tmp_path):
    p = tmp_path / "work" / "film" / "direction.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps(_direction()), encoding="utf-8")
    ch = fl.derive_chapters(fl.read_direction(p))
    assert [c["title"] for c in ch] == list(TITLES.values()) and ch[-1]["endS"] == 198.0


def _wav(path: Path, seconds: float):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(1000)
        w.writeframes(struct.pack("<h", 0) * int(seconds * 1000))


def test_a_preset_writes_the_direction_and_the_captures_soundtrack_as_the_mix(tmp_path, monkeypatch):
    streaming = tmp_path / "StreamingAssets"
    cap = streaming / "99_Heat"
    cap.mkdir(parents=True)
    (cap / "capture.json").write_text(json.dumps({"times": "times.json", "audio": "audio.wav"}), encoding="utf-8")
    (cap / "times.json").write_text(json.dumps([3.6 + k / 30 for k in range(5000)]), encoding="utf-8")
    _wav(cap / "audio.wav", 202.0)
    songs = tmp_path / "segments.json"
    songs.write_text(json.dumps(SONGS), encoding="utf-8")
    monkeypatch.setattr(fh, "STREAMING", streaming)
    preset = {"id": "heat", "title": "Heat", "capture": "99_Heat", "direction": str(tmp_path / "film" / "direction.json"),
              "heat": {"name": "heat", "take": "heat", "songs": str(songs), "titles": TITLES}}
    path, d, _notes = fh.build_from_preset(preset)
    assert path.exists() and (path.parent / "audio" / "heat_mix.wav").exists()
    assert d["timeline"]["film_duration_s"] == pytest.approx(round(3.6 + 4999 / 30 + 0.1, 3))
    entry, _ = fl.build_entry({"id": "heat", "title": "Heat", "capture": "99_Heat", "direction": str(path)}, library_dir=tmp_path / "lib")
    # the bow starts after this short film: no chapter for it, and the last chapter runs to the film end
    assert entry["available"] and len(entry["chapters"]) == 4 and entry["durationS"] == d["timeline"]["film_duration_s"]
    assert entry["chapters"][-1]["endS"] == d["timeline"]["film_duration_s"]


def test_the_looks_are_data_driven_and_the_default_film_is_unchanged():
    base = _direction()
    assert base["avatar_opacity"] == 0.95
    for s in base["segments"]:
        assert "avatars" not in s["layers"] and "rhythm_score" not in s["layers"] and "rhythm_rings" not in s["layers"]
        assert s["layers"]["room"] is True and s["layers"]["skeleton_lines"] == 0.0
    look = _direction(cfg={"avatar_opacity": 0.25, "room": False, "skeleton_lines": 1.0, "rhythm_score": True})
    assert look["avatar_opacity"] == 0.25
    for s in look["segments"]:
        lay = s["layers"]
        assert lay["room"] is False and lay["skeleton_lines"] == 1.0 and lay["rhythm_score"] is True and lay["rhythm_rings"] is True
        assert "avatars" not in lay                                   # the one number at the top of the direction rules every segment
        if s["view_state"] == "overhead":
            assert lay["grid"] is True                                # no room: the floor grid stays under the dancers from above too


def _pov_cfg(**kw):
    from dancecap import film_heat as FH
    c = dict(FH.DEFAULTS)
    c.update({"pov_cameras": ["01", "02"], **kw})
    return c


def _table(ok01, ok02, n=800, dt=0.25):
    return {"cams": {"01": {"t0": 0.0, "dt": dt, "ok": [ok01(i * dt) for i in range(n)]},
                     "02": {"t0": 0.0, "dt": dt, "ok": [ok02(i * dt) for i in range(n)]}}}


def test_pov_shots_alternate_cameras_avoid_overheads_and_keep_the_orbits():
    from dancecap import film_heat as FH
    cfg = _pov_cfg()
    end = 199.0
    shots = FH.place_povs(FH.shot_plan(end, cfg), end, cfg, _table(lambda t: 1, lambda t: 1))
    povs = [s for s in shots if s["kind"] == "pov"]
    assert len(povs) >= 5
    assert [p["camera"] for p in povs][:4] == ["01", "02", "01", "02"]
    assert shots[0]["start"] == 0.0 and abs(shots[-1]["end"] - end) < 1e-6
    for a, b in zip(shots, shots[1:]):
        assert abs(a["end"] - b["start"]) < 1e-6
    for s in shots:
        if s["kind"] == "orbit":
            assert s["end"] - s["start"] >= cfg["min_shot_s"] - 1e-6
    ov = [s for s in shots if s["kind"] == "overhead"]
    for p in povs:
        assert all(p["start"] >= o["end"] + cfg["pov_gap_s"] - 1e-6 or p["end"] <= o["start"] - cfg["pov_gap_s"] + 1e-6 for o in ov)


def test_a_camera_that_cannot_show_the_dance_is_not_used_and_no_table_means_no_pov():
    from dancecap import film_heat as FH
    cfg = _pov_cfg()
    shots = FH.place_povs(FH.shot_plan(199.0, cfg), 199.0, cfg, _table(lambda t: 0, lambda t: 1))
    assert {s["camera"] for s in shots if s["kind"] == "pov"} == {"02"}
    assert FH.place_povs(FH.shot_plan(199.0, cfg), 199.0, cfg, None) == FH.shot_plan(199.0, cfg)


def test_a_pov_camera_block_fills_its_shot():
    from dancecap import film_heat as FH
    cfg = _pov_cfg()
    c = FH.pov_camera_block("01", 9.0, cfg)
    assert c["mode"] == "fly_to_camera" and c["camera"] == "01" and c["composite"] == "3d_over_video"
    assert abs(c["fly_s"] + c["fade_in_s"] + c["hold_s"] + c["fade_out_s"] + c["pull_back_s"] - 9.0) < 1e-6


def test_a_rarely_showable_camera_still_gets_its_best_window_and_an_overhead_gives_way():
    from dancecap import film_heat as FH
    cfg = _pov_cfg(pov_min_s=5.0)
    # camera 02 can be shown only 22-31 s, under the first overhead (22-28 s)
    shots = FH.place_povs(FH.shot_plan(199.0, cfg), 199.0, cfg, _table(lambda t: 1, lambda t: 1 if 22.0 <= t < 31.0 else 0))
    p02 = [s for s in shots if s["kind"] == "pov" and s["camera"] == "02"]
    assert len(p02) == 1 and 22.0 <= p02[0]["start"] and p02[0]["end"] <= 31.0
    assert not any(s["kind"] == "overhead" and s["start"] < p02[0]["end"] and s["end"] > p02[0]["start"] for s in shots)
    for a, b in zip(shots, shots[1:]):
        assert abs(a["end"] - b["start"]) < 1e-6


def test_teacher_rows_follow_the_teachers_own_frame_index():
    """lucas_luiza (two-view): the tripod's camera track holds native frame indices 1007, 1009, ... while the teacher track has one row per take
    frame - indexing the teachers by frame_index read the wrong rows (the tripod looked showable 24 % of the time instead of 66 %)."""
    import numpy as np
    from dancecap.film_heat import teacher_rows

    class Npz(dict):
        @property
        def files(self):
            return list(self)
    tp = Npz(frame_index=np.array([1007, 1009, 1011, 1013]), t=np.array([16.8, 16.833, 16.867, 16.9]), box=np.zeros((4, 2, 4)))
    assert teacher_rows(tp, [1009, 1013, 1500], np.array([16.833, 16.9, 30.0])).tolist() == [1, 3, -1]
    tp2 = Npz(t=np.array([0.0, 0.0333, 0.0667]), box=np.zeros((3, 2, 4)))
    assert teacher_rows(tp2, [5, 6, 7], np.array([0.0, 0.07, 1.0])).tolist() == [0, 2, -1]
    tp3 = Npz(box=np.zeros((3, 2, 4)))
    assert teacher_rows(tp3, [0, 2, 3], np.zeros(3)).tolist() == [0, 2, -1]


def test_a_pov_shot_needs_the_camera_s_pose_over_all_of_it():
    """lucas_luiza: with pov_min_ok 0.75 two Steadicam shots ran into pose gaps at their ends (camera_video's POV check failed)."""
    from dancecap.film_heat import pov_pose_full
    tab = {"cams": {"01": {"t0": 0.0, "dt": 0.25, "ok": [1] * 40, "pose": [1] * 30 + [0] * 10}, "02": {"t0": 0.0, "dt": 0.25, "ok": [1] * 40}}}
    assert pov_pose_full(tab, "01", 1.0, 6.0) and not pov_pose_full(tab, "01", 1.0, 7.3) and not pov_pose_full(tab, "01", 0.2, 3.0)
    assert pov_pose_full(tab, "02", 1.0, 7.0) and not pov_pose_full(tab, "03", 1.0, 2.0)    # no pose rows: older tables pass; no camera: no
