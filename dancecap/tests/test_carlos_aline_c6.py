"""C6 (carlos_aline): the pure parts of the soundtrack, the no-beat step filter, the refit job builder and the capture patching."""
import json

import numpy as np
import pytest

from dancecap import carlos_aline_audio as au
from dancecap import carlos_aline_export as ex
from dancecap import timing as tm
from dancecap.farm import ca_analysis as can
from dancecap.farm import ca_fix


def test_equal_power_is_power_complementary():
    t = np.linspace(0, 10, 1001)
    o, i = au.equal_power(t, 2.0, 5.0)
    assert np.allclose(o ** 2 + i ** 2, 1.0)
    assert o[0] == 1.0 and i[0] == 0.0 and o[-1] < 1e-12 and i[-1] == 1.0


PLAN = {"song_start": {"kojo": 13.3, "guetta": 113.4}, "handover": [120.1, 122.2], "end_cut": 194.25, "heat_into_kojo": [13.15, 13.45], "guetta_into_heat": [193.95, 194.45]}


def test_gain_curves_cover_the_timeline_with_unit_power():
    t = np.arange(0, 200, 0.01)
    g = au.gain_curves(t, PLAN)
    p = g["heat"] ** 2 + g["kojo"] ** 2 + g["guetta"] ** 2
    assert np.allclose(p, 1.0, atol=1e-9)
    at = lambda s: int(round(s / 0.01))
    assert g["heat"][at(5)] == 1.0 and g["kojo"][at(60)] == 1.0 and g["guetta"][at(150)] == 1.0
    assert g["kojo"][at(121)] > 0 and g["guetta"][at(121)] > 0          # inside the hand-over both play
    assert g["heat"][at(199)] == 1.0 and g["guetta"][at(199)] == pytest.approx(0.0, abs=1e-9)


def test_plan_from_segments_uses_the_governing_seam():
    seg = {"governing": {"switch_s": 122.25, "kojo_until_s": 120.1, "guetta_from_s": 122.17}}
    songs = {"kojo": {"song_zero_at_reference_s": 13.313, "audible_reference_s": [13.31, 122.25]},
             "guetta": {"song_zero_at_reference_s": 113.466, "audible_reference_s": [113.47, 194.25]}}
    p = au.plan_from_segments(seg, songs)
    assert p["handover"] == [120.1, 122.17] and p["end_cut"] == 194.25 and p["song_start"]["kojo"] == 13.313
    seg["governing"] = {"switch_s": 122.0}          # a re-cut without a seam: one second before the switch
    assert au.plan_from_segments(seg, songs)["handover"] == [121.0, 122.0]


def test_lufs_gain_is_clamped():
    assert au.lufs_gain_db(-12.0, -10.0) == pytest.approx(-2.0)
    assert au.lufs_gain_db(-12.0, -90.0) == 18.0


def test_drop_no_beat_keeps_events_outside():
    ev = [{"t": 2.0}, {"t": 13.31}, {"t": 50.0}, {"t": 121.0}, {"t": 195.0}]
    kept, dropped = tm.drop_no_beat(ev, ((0.0, 13.31), (120.1, 122.17), (194.25, 203.0)))
    assert [e["t"] for e in kept] == [13.31, 50.0]          # the interval is half open [a, b)
    assert [e["t"] for e in dropped] == [2.0, 121.0, 195.0]
    assert tm.drop_no_beat(ev, ())[1] == []


def test_no_beat_arg_adds_the_governing_seam():
    seg = {"no_beat": [[0.0, 13.31], [194.25, 202.083]], "governing": {"kojo_until_s": 120.1009, "guetta_from_s": 122.1712}}
    assert can.no_beat_arg(seg) == "0:13.31,120.101:122.171,194.25:202.083"


def _tpl():
    steps = [{"name": n, "cmd": f"x {n} cag1488", "timeout_min": 5} for n in ("chunk_take", "front", "qa_mask", "await_pin", "fit", "refine", "quality", "evidence")]
    steps[2]["save"] = ["a"]
    return {"id": "cag1488", "take": "carlos_aline_cag1488", "dance": "carlos_aline", "cams": ["01", "02"], "steps": steps, "outputs": [{"path": "dancecap/work/carlos_aline_cag1488/smplx/"}],
            "chunk": {"id": "cag1488", "take": "carlos_aline_cag1488", "sync_fixed": {"offsets_ms": {"01": 0.0, "02": -62.0}}}, "env": {"A": "1"}, "profile": "carlos_aline"}


def test_fix_job_has_one_qa_step_and_the_teacher_evidence_env():
    est = {"front_min": 10.0, "fit_min": 6.0, "refine_min": 1.0, "quality_min": 3.0, "total_min_4090": 24.0}
    j = ca_fix.build_fix_job(_tpl(), "cax1408", 1408, 1768, est, "why")
    names = [s["name"] for s in j["steps"]]
    assert names.count("qa_mask") == 1 and names == ["chunk_take", "front", "qa_mask", "await_pin", "fit", "refine", "quality", "evidence"]
    assert j["steps"][2].get("save") == ["a"]
    assert j["id"] == "cax1408" and j["take"] == "carlos_aline_cax1408" and j["chunk"]["T"] == 361
    assert j["env"]["HM_TEACHER_EVIDENCE"] == "1" and "cag1488" not in json.dumps(j["steps"])


def test_patch_capture_json_merges_provenance(tmp_path):
    (tmp_path / "capture.json").write_text(json.dumps({"provenance": {"take": "x"}}))
    done = ex.patch_capture_json([tmp_path, tmp_path / "missing"], {"song": {"n": 2}})
    assert len(done) == 1
    assert json.loads((tmp_path / "capture.json").read_text())["provenance"] == {"take": "x", "song": {"n": 2}}
