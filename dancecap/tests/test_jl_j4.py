"""john_lula analyses / capture / library pieces (agent J4): the pod-job builder, the low-confidence flags, the one-song heat film (timeline split) and the exporter's flag pass-through.

    .venv-nopo4d/Scripts/python.exe -m pytest dancecap/tests/test_jl_j4.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dancecap import export_unity as eu  # noqa: E402
from dancecap import film_heat as fh  # noqa: E402
from dancecap import jl_lowconf as lc  # noqa: E402
from dancecap.farm import jl_analysis as ja  # noqa: E402

SONGS = {"timeline": [{"from_s": 0.0, "to_s": 3.4, "state": "intro"}, {"from_s": 3.4, "to_s": 80.2, "state": "music"}, {"from_s": 80.2, "to_s": 88.12, "state": "outro"}],
         "no_beat": [[0.0, 3.4], [80.2, 88.12]]}
TITLES = {"intro": "Intro", "music_1": "First", "music_2": "Second", "music_3": "Third", "outro": "Applause"}


def test_split_timeline_cuts_one_state_into_equal_runs_and_leaves_the_rest():
    s = fh.split_timeline(SONGS, {"music": {"every_s": 25.6}})
    names = [t["state"] for t in s["timeline"]]
    assert names == ["intro", "music_1", "music_2", "music_3", "outro"]
    m = [t for t in s["timeline"] if t["state"].startswith("music")]
    assert abs(m[0]["from_s"] - 3.4) < 1e-6 and abs(m[-1]["to_s"] - 80.2) < 1e-6
    assert max(abs((t["to_s"] - t["from_s"]) - 25.6) for t in m) < 1e-3
    assert all(a["to_s"] == b["from_s"] for a, b in zip(s["timeline"], s["timeline"][1:]))
    assert fh.split_timeline(SONGS, None) is SONGS and fh.split_timeline(SONGS, {"nothing": {"every_s": 5}})["timeline"] == SONGS["timeline"]


def test_the_split_heat_has_five_chapters_no_beat_counter_in_the_no_beat_stretches_and_one_opacity_value():
    songs = fh.split_timeline(SONGS, {"music": {"every_s": 25.6}})
    d = fh.build_direction(name="jl", title="T", capture="08_X", take="x", songs=songs, dance_range=[0.0, 88.07], film_end=88.12, titles=TITLES,
                           cfg={"avatar_opacity": 0.25, "room": False, "skeleton_lines": 1.0, "rhythm_score": True, "no_overhead_after_s": 80.0})
    assert [c["title"] for c in d["chapters"]] == ["Intro", "First", "Second", "Third", "Applause"]
    assert d["avatar_opacity"] == 0.25
    for s in d["segments"]:
        mid = 0.5 * (s["film"][0] + s["film"][1])
        assert s["layers"]["beat_counter"] == (3.4 < mid < 80.2), s["id"]
        assert s["layers"]["room"] is False and s["layers"]["rhythm_score"] is True
    assert not any(s["camera"]["mode"] == "overhead_floorcraft" and s["film"][0] >= 80.0 for s in d["segments"])
    assert d["segments"][-1]["film"][1] == 88.12


def test_the_analysis_jobs_carry_the_owner_the_one_camera_and_the_no_beat_intervals():
    seg = {"no_beat": [[80.19, 88.12], [0.0, 3.397]]}
    assert ja.no_beat_arg(seg) == "0:3.397,80.19:88.12"
    jobs = {j["id"]: j for j in ja.job_defs("0:3.4")}
    t = jobs["jlan-timing"]
    assert t["kind"] == ja.MEM_KIND and t["steps"][0][0] == "timing"
    args = t["steps"][0][2]
    assert "--smplx-dir smplx_rf2_noslip" in args and "--n-cams 1" in args and "--no-beat 0:3.4" in args and "beats_heat.json" in args
    assert [s[0] for s in t["steps"]] == ["timing", "counterbalance", "floorcraft"]
    assert jobs["jlan-physics"]["steps"][0][1] == "physics" and ja.OWNER == "J4"


def _npz(path: Path, absent: list[tuple[int, int]], spike: tuple[int, int] | None = None):
    t = np.arange(300) / 30.0
    a = np.zeros(300, bool)
    for i, j in absent:
        a[i:j + 1] = True
    kw = {"t": t, "absent": a}
    if spike:
        d = np.zeros(300, bool)
        d[spike[0]:spike[1] + 1] = True
        kw["depth_spike_repaired"] = d
    np.savez(path, **kw)


def test_low_confidence_intervals_come_from_the_fits_absent_runs_the_weak_list_and_the_zoom_steps(tmp_path):
    w = tmp_path / "john_lula"
    (w / "smplx_rf2_noslip").mkdir(parents=True)
    (w / "review").mkdir()
    _npz(w / "smplx_rf2_noslip" / "lead.npz", [(30, 60)], spike=(90, 92))
    _npz(w / "smplx_rf2_noslip" / "follow.npz", [])
    (w / "review" / "weak.json").write_text(json.dumps([
        {"chunk": "all", "t_s": [0, 10], "what": "MONOCULAR: everything"},
        {"chunk": "a", "t_s": [5.0, 6.0], "what": "follow: hidden behind John (skeleton on the partner), landmarks zeroed: interpolated"},
        {"chunk": "a", "t_s": [7.0, 8.0], "what": "lead: reprojection 5 % or more"},
        {"chunk": "a", "t_s": [1.0, 2.0], "what": "lead: no 2D evidence fed the fit: the temporal prior's interpolation"}]), encoding="utf-8")
    iv = lc.intervals(w)
    kinds = {(x["role"], x["level"]) for x in iv}
    assert ("lead", "interpolated") in kinds and ("lead", "repaired") in kinds and ("follow", "hidden") in kinds and ("lead", "poor_2d") in kinds
    assert sum(1 for x in iv if x["from"] == 1.0) == 1            # the absent run itself; a weak entry that only restates an absent run is not repeated
    assert all(x["level"] != "zoom_step" or x["role"] == "both" for x in iv)
    ab = next(x for x in iv if x["level"] == "interpolated")
    assert abs(ab["from"] - 1.0) < 1e-6 and abs(ab["to"] - 2.0) < 1e-6


def test_reasons_are_per_role_and_overlap_based():
    ivs = [{"from": 5.0, "to": 6.0, "role": "follow", "level": "hidden", "reason": "r"}, {"from": 20.0, "to": 21.0, "role": "both", "level": "zoom_step", "reason": "z"}]
    assert lc.reasons_at(ivs, 5.5, None, "follow") and not lc.reasons_at(ivs, 5.5, None, "lead")
    assert lc.reasons_at(ivs, 19.0, 20.2, "lead")
    assert not lc.reasons_at(ivs, 7.0, 8.0, "follow")
    items = [{"t": 5.5}, {"t": 7.0}]
    assert lc._flag_list(items, ivs, None, "t", role="follow") == 1 and "low_confidence" in items[0] and "low_confidence" not in items[1]


def test_the_unity_timing_copy_keeps_the_low_confidence_flag_only_where_it_is_set():
    keep = ("t", "foot")
    assert eu._step_for_unity({"t": 1.0, "foot": "left", "x": 9}, keep) == {"t": 1.0, "foot": "left"}
    assert eu._step_for_unity({"t": 1.0, "foot": "left", "low_confidence": ["a"]}, keep)["low_confidence"] == ["a"]
