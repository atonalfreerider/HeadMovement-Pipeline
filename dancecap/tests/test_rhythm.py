"""dancecap.rhythm / rhythm_entrain: the rhythm entrainment analysis and the causal live score (synthetic takes; no data of any real take).

    .venv-nopo4d/Scripts/python.exe -m pytest dancecap/tests/test_rhythm.py -q
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dancecap import rhythm as rh  # noqa: E402
from dancecap import rhythm_entrain as ent  # noqa: E402

CFG = rh.Config()


def _final(kind, **kw):
    inp = rh.synthetic_inputs(kind, **kw)
    return rh.run_analysis(inp, CFG, with_sensitivity=False)


def _score(kind, **kw):
    inp = rh.synthetic_inputs(kind, **kw)
    return rh.aggregate(rh.run_series(inp, CFG), CFG)


def test_a_sway_and_steps_locked_to_the_grid_score_at_the_top():
    d = _final("locked")
    assert d["final"]["score"] >= 90
    assert all(d["final"]["components"][k] > 0.8 for k in rh.COMPONENTS)
    assert d["final"]["levels"]["lead"] == "halfbar"              # the synthetic sway is one cycle per two pulses
    assert d["final"]["confidence"]["label"] == "high"


def test_unrelated_motion_scores_near_the_bottom():
    s = [_score("random", seed=k)["score"] for k in range(4)]
    assert np.mean(s) <= 12 and max(s) <= 22


def test_a_sway_at_the_wrong_tempo_gets_no_entrainment_credit():
    f = _score("wrong_tempo", body_tempo=1.12)
    for k in ("body", "feet", "recent"):
        assert f[k] < 0.1, k
    assert f["score"] < 15


def test_a_constant_phase_offset_is_not_penalised_by_design():
    # the audio-video offset is unpinned (+-30 ms) and where in the beat a weight transfer lands is not established: only the tempo and a steady phase count
    # (the accents component IS absolute: it asks whether the stops, hits and steps sit on the music's strong onsets)
    a, b = _score("locked", seed=3), _score("offset", seed=3, offset_ms=140.0)
    for k in ("body", "feet", "recent", "couple"):
        assert abs(a[k] - b[k]) < 0.05, k
    assert b["accents"] < a["accents"]


def test_the_score_does_not_depend_on_where_the_grid_starts():
    inp = rh.synthetic_inputs("locked", seed=2)
    a = rh.aggregate(rh.run_series(inp, CFG), CFG)["score"]
    g = inp.grid.shifted(0.211)
    inp2 = rh.Inputs(g, inp.steps, inp.music, inp.motion, inp.t0, inp.t1, dict(inp.meta), {}, inp.body)
    assert abs(a - rh.aggregate(rh.run_series(inp2, CFG), CFG)["score"]) < 4


def test_jitter_in_the_footwork_lowers_the_footwork_component_monotonically():
    s = [_score("locked", jitter_ms=j, seed=3)["feet"] for j in (10, 150, 260)]       # a pulse is 714 ms: 260 ms of jitter is nearly uniform
    assert s[0] > s[1] > s[2] and s[0] > 0.9 and s[2] < 0.3


def test_step_classes_follow_the_dancers_own_locked_step_rhythm():
    inp = rh.synthetic_inputs("locked", jitter_ms=5.0)
    d = rh.run_analysis(inp, CFG, with_sensitivity=False)
    cols = d["steps"]["columns"]
    rows = [dict(zip(cols, r)) for r in d["steps"]["rows"] if r[cols.index("valid")]]
    assert len(rows) > 100
    late = [r for r in rows if r["t"] > inp.grid.t[0] + 60]
    assert np.mean([r["class"] == "on" for r in late]) > 0.9
    assert {r["kind"] for r in rows} == {"beat"}                      # steps on the pulses only
    assert rows[0]["class"] == "neutral" and rows[1]["class"] == "neutral"       # no lock evidence yet: no colour


def test_a_step_on_the_and_is_labelled_and():
    inp = rh.synthetic_inputs("locked", jitter_ms=5.0, rate="eighth")
    d = rh.run_analysis(inp, CFG, with_sensitivity=False)
    cols = d["steps"]["columns"]
    kinds = {r[cols.index("kind")] for r in d["steps"]["rows"] if r[cols.index("valid")]}
    assert kinds == {"beat", "and"}


def test_the_entrainment_levels_are_found_without_being_told():
    # a sway at 1.5 cycles per pulse = three in two... the level set contains it; a sway at the half-bar level is chosen as such
    inp = rh.synthetic_inputs("locked")
    A = rh.run_series(inp, CFG)
    m = A.valid & (A.elapsed > 30)
    assert np.mean(A.S.best_body["lead"][m] == ent.LEVEL_NAMES.index("halfbar")) > 0.95


def test_the_live_score_is_causal():
    inp = rh.synthetic_inputs("locked", seed=1)
    full = rh.run_series(inp, CFG)
    cut = 90.0
    b = inp.body
    m = b.t <= cut
    body2 = ent.Body(b.t[m], {r: {k: v[m] for k, v in b.sig[r].items()} for r in rh.ROLES}, b.source)
    steps2 = {r: [s for s in inp.steps[r] if s["t"] <= cut] for r in rh.ROLES}
    inp2 = rh.Inputs(inp.grid, steps2, inp.music, inp.motion, inp.t0, cut, dict(inp.meta), {}, body2)
    part = rh.run_series(inp2, CFG)
    n = len(part.t) - 3
    for k in rh.COMPONENTS:
        assert np.nanmax(np.abs(full.comps[k][:n] - part.comps[k][:n])) < 1e-9, k


def test_the_live_score_starts_neutral_and_settles_up_for_a_good_dance_and_down_for_a_bad_one():
    good = rh.analyse_one(rh.synthetic_inputs("locked"), CFG)["live"]
    bad = rh.analyse_one(rh.synthetic_inputs("random"), CFG)["live"]
    assert abs(good["score"][0] - 50.0) < 1e-6 and abs(bad["score"][0] - 50.0) < 1e-6
    assert good["score"][-1] > 85
    assert bad["score"][-1] < 25
    assert good["evidence"][-1] > 0.9 and bad["evidence"][-1] < 0.4


def test_the_score_is_frozen_in_the_no_beat_stretches():
    inp = rh.synthetic_inputs("locked", seed=4)
    live = rh.analyse_one(inp, CFG)["live"]
    st = live["state"]
    assert st[0] == 1                                                  # before the first slot: no song
    pre = np.where(st == 1)[0]
    assert np.allclose(live["score"][pre[pre < 40]], 50.0)            # neutral and frozen until the beat starts
    assert live["confidence"][0] == 0.0
    assert live["state"][-1] == 1 and abs(live["score"][-1] - live["score"][np.where(st == 0)[0][-1]]) < 1e-9


def test_a_seam_gap_in_the_grid_is_found_frozen_and_the_entrainment_restarts():
    inp = rh.synthetic_inputs("locked", seed=5)
    g = inp.grid
    keep = ~((g.t > 40.0) & (g.t < 46.0))
    rows = [[t, ty, m, s] for t, ty, m, s, k in zip(g.t, g.type, g.measure, g.slot, keep) if k]
    inp.grid = rh.build_grid(rows, no_beat_none=[(0.0, 10.0)], conf=np.ones(len(rows)), default_conf=1.0)
    assert any(k == "seam" for _, _, k in inp.grid.no_beat)
    d = rh.run_analysis(inp, CFG, with_sensitivity=False)
    cols = d["steps"]["columns"]
    inside = [r for r in d["steps"]["rows"] if 40.4 < r[0] < 45.6]
    assert inside and all(not r[cols.index("valid")] for r in inside)
    live = d["live"]
    i0, i1 = int((41.0 - live["t0"]) * live["hz"]), int((45.0 - live["t0"]) * live["hz"])
    assert set(live["state"][i0:i1]) == {2}
    assert max(live["score"][i0:i1]) - min(live["score"][i0:i1]) < 1e-9
    A = rh.run_series(inp, CFG)
    j = int(np.searchsorted(A.t, 46.5))
    assert A.elapsed[j] < 2.0                                          # a new grid segment: the averages restarted


def test_the_audio_video_shift_barely_moves_the_score_of_a_body_lock():
    inp = rh.synthetic_inputs("locked", jitter_ms=15.0)
    s = rh.sensitivity(inp, CFG)
    assert len(s["rows"]) == 3 and s["range"][0] <= s["range"][1]
    assert s["range"][1] - s["range"][0] < 8                           # +-30 ms is a few degrees of a 1.6 s cycle


def test_output_is_json_and_complete():
    d = _final("locked")
    back = json.loads(json.dumps(d, allow_nan=False))
    for k in ("final", "final_card", "describe", "steps", "measures", "windows", "live", "no_beat", "claims", "title", "formula", "config"):
        assert k in back
    live = back["live"]
    n = live["n"]
    assert all(len(live[k]) == n for k in (*rh.COMPONENTS, "score", "confidence", "evidence", "state"))
    assert "not an official score" in back["title"] and "entrainment" in back["title"]
    assert back["final_card"]["to_t"] > back["final_card"]["from_t"]
    assert abs(sum(back["formula"]["weights"].values()) - 1.0) < 1e-9
    assert back["measures"] and all("score" in m for m in back["measures"])
    assert back["final"]["score_interval_90"][0] <= back["final"]["score"] <= back["final"]["score_interval_90"][1]
    assert any("validates a mapping" in c for c in back["claims"])


def test_accents_from_a_click_track_and_a_break(tmp_path):
    from scipy.io import wavfile
    sr = 22050
    t = np.arange(int(sr * 30.0)) / sr
    x = np.zeros_like(t)
    for k in range(60):                                                # a click every 0.5 s ... with a gap 12-13 s
        s0 = int(k * 0.5 * sr)
        if 12.0 <= k * 0.5 < 13.0:
            continue
        n = np.arange(int(0.02 * sr))
        x[s0:s0 + len(n)] += np.sin(2 * np.pi * 1800 * n / sr) * np.exp(-n / (0.004 * sr)) * (1.0 if k % 2 == 0 else 0.5)
    x += 0.02 * np.sin(2 * np.pi * 110 * t) * ((t < 12.0) | (t > 13.0))
    p = tmp_path / "clicks.wav"
    wavfile.write(str(p), sr, (x * 20000).astype(np.int16))
    m = rh.music_from_audio(p, 100.0, CFG)
    assert m.source == "audio" and len(m.acc_t) > 30
    assert m.acc_t.min() > 100.0                                       # reference time = t0 + sample time
    assert any(abs(a - 112.1) < 0.7 and b - a >= 0.8 for a, b in m.breaks), m.breaks
    strong = m.acc_t[m.acc_s >= 0.8]
    on_even = [abs(((x - 100.0) + 0.25) % 1.0 - 0.25) < 0.04 for x in strong]
    assert np.mean(on_even) > 0.8                                      # the strong accents are the full-level clicks


def test_without_a_soundtrack_the_grid_downbeats_stand_in():
    inp = rh.synthetic_inputs("locked")
    m = rh.music_from_grid(inp.grid)
    assert m.source == "grid" and m.acc_s.max() == 1.0 and len(m.acc_t) == len(inp.grid)


def test_load_timing_and_a_fake_take(tmp_path):
    P = 60.0 / 80.0
    rows, t = [], 12.0
    for m in range(20):
        for s in range(8):
            rows.append([round(t + (m * 8 + s) * P / 2, 6), (1, 3, 3, 2, 1, 3, 2, 3)[s], m, s])
    steps = {r: [{"t": round(rows[i][0] + 0.01, 4), "foot": "left", "sigma": 0.02} for i in range(0, len(rows) - 2, 2)] for r in rh.ROLES}
    d = {"beats": {"grid": rows, "tempo_bpm": 80.0}, "footsteps": steps, "dropped_no_beat": {"lead": {"intervals": [[0.0, 12.0]]}, "follow": {"intervals": [[0.0, 12.0]]}},
         "absolute_timing": {"valid": False}, "inputs": {"beats": "fake"}}
    work = tmp_path / "work"
    (work / "timing").mkdir(parents=True)
    (work / "timing" / "timing.json").write_text(json.dumps(d))
    inp = rh.load_inputs("fake", None, None, None, CFG, work)
    assert len(inp.grid) == 160 and inp.music.source == "grid" and not inp.meta["av_pinned"] and inp.body is None
    assert any(k == "none" for _, _, k in inp.grid.no_beat)
    doc = rh.run_analysis(inp, CFG, with_sensitivity=False)
    assert doc["final"]["components"]["feet"] is not None
    assert any("foot touchdowns only" in c for c in doc["claims"])      # no body signals: said so


def test_a_take_without_a_beat_grid_has_no_score(tmp_path):
    d = {"beats": {"grid": None, "skipped": True}, "footsteps": {"lead": [], "follow": []}}
    work = tmp_path / "work"
    (work / "timing").mkdir(parents=True)
    (work / "timing" / "timing.json").write_text(json.dumps(d))
    with pytest.raises(ValueError, match="no beat grid"):
        rh.load_inputs("nodance", None, None, None, CFG, work)


def test_nothing_crashes_with_no_steps_and_no_body():
    inp = rh.synthetic_inputs("locked")
    inp.steps = {"lead": [], "follow": []}
    inp.body = None
    d = rh.run_analysis(inp, CFG, with_sensitivity=True)
    json.dumps(d, allow_nan=False)                                     # unknown values are null, never NaN
    assert d["final"]["score"] is None or math.isfinite(d["final"]["score"])
    assert d["final"]["insufficient_evidence"] is True


def test_the_wrong_tempo_baseline_leaves_unrelated_noise_near_zero():
    inp = rh.synthetic_inputs("random", seed=11)
    A = rh.run_series(inp, CFG)
    m = A.valid & (A.elapsed > 30)
    assert np.nanmean(A.comps["body"][m]) < 0.15


def test_install_writes_the_viewers_copy_and_names_it_in_capture_json(tmp_path):
    d = _final("locked")
    cap = tmp_path / "99_Cap"
    cap.mkdir()
    (cap / "capture.json").write_text(json.dumps({"version": 3, "fps": 30}))
    other = tmp_path / "nothing_here"
    done = rh.install_into(d, [cap, other])
    assert done == [str(cap)]
    m = json.loads((cap / "capture.json").read_text())
    assert m["rhythm"] == "rhythm.json" and m["fps"] == 30
    v = json.loads((cap / "rhythm.json").read_text())
    assert v["kind"] == "rhythm_view" and v["live"]["n"] == d["live"]["n"] and v["steps"]["rows"]
    assert [c["label"] for c in v["components"]] == ["Body sway", "Footwork", "Recent form", "Couple sync", "Accents"]
    assert [c["key"] for c in v["components"]] == list(rh.COMPONENTS)
    assert all(r[2] in (0, 1, 2, 3) and r[1] in (0, 1) for r in v["steps"]["rows"])
    assert v["final"]["insufficient_evidence"] is False and "lock_evidence_z" in v["final"]
    assert (cap / "rhythm.json").stat().st_size < 600_000


def test_the_controls_helpers_build_the_wrong_grids():
    from dancecap import rhythm_controls as rc
    inp = rh.synthetic_inputs("locked")
    g = rc.stretch_grid(inp.grid, 1.2)
    assert abs((g.t[-1] - g.t[0]) - (inp.grid.t[-1] - inp.grid.t[0]) / 1.2) < 1e-6
    w = rc.wandering_grid(inp.grid, 0)
    assert len(w.t) == len(inp.grid.t) and np.all(np.diff(w.t) > 0)
    c = rc.crop_grid(inp.grid, 40.0, 80.0)
    assert c.t[0] >= 40.0 and c.t[-1] <= 80.0 and any(k == "none" for _, _, k in c.no_beat)
