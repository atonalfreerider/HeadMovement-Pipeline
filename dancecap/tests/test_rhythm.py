"""dancecap.rhythm: the rhythm analysis and the causal live score (synthetic takes; no data of any real take).

    .venv-nopo4d/Scripts/python.exe -m pytest dancecap/tests/test_rhythm.py -q
"""
from __future__ import annotations

import json
import math
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dancecap import rhythm as rh  # noqa: E402

CFG = rh.Config()


def _final(kind, **kw):
    inp = rh.synthetic_inputs(kind, **kw)
    return rh.run_analysis(inp, CFG, with_sensitivity=False)


def test_perfectly_on_beat_steps_score_near_the_top():
    d = _final("on_beat")
    assert d["final"]["score"] >= 90
    assert all(d["final"]["components"][k] > 0.8 for k in ("lock", "tempo", "flow", "sync"))


def test_random_steps_score_near_the_bottom():
    scores = [_final("random", seed=s)["final"]["score"] for s in range(4)]
    assert max(scores) <= 10 and np.mean(scores) <= 6


def test_steady_off_beat_steps_do_not_get_the_flow_credit():
    d = _final("off_beat")
    assert d["final"]["components"]["lock"] < 0.05
    assert d["final"]["components"]["flow"] < 0.3
    assert d["final"]["score"] < 35


def test_a_small_constant_offset_is_inside_the_audio_video_uncertainty():
    d = _final("offset_consistent", offset_ms=30.0, jitter_ms=15.0)
    assert d["final"]["score"] >= 80
    assert abs(d["describe"]["both"]["binary_grid"]["mean_offset_ms"] - 30.0) < 8


def test_jitter_lowers_the_score_monotonically():
    s = [_final("on_beat", jitter_ms=j, seed=3)["final"]["score"] for j in (10, 50, 90)]
    assert s[0] > s[1] > s[2]


def test_step_classes_and_offsets():
    inp = rh.synthetic_inputs("on_beat", jitter_ms=5.0)
    d = rh.run_analysis(inp, CFG, with_sensitivity=False)
    cols = d["steps"]["columns"]
    rows = [dict(zip(cols, r)) for r in d["steps"]["rows"] if r[cols.index("valid")]]
    assert len(rows) > 100
    assert np.mean([r["class"] == "on" for r in rows]) > 0.95
    assert all(abs(r["offset_ms"]) < 60 for r in rows)
    assert {r["kind"] for r in rows} == {"beat"}                      # steps on the pulses only


def test_a_step_on_the_and_is_labelled_and():
    inp = rh.synthetic_inputs("on_beat", jitter_ms=5.0, rate="eighth")
    d = rh.run_analysis(inp, CFG, with_sensitivity=False)
    cols = d["steps"]["columns"]
    kinds = {r[cols.index("kind")] for r in d["steps"]["rows"] if r[cols.index("valid")]}
    assert kinds == {"beat", "and"}


def test_three_in_two_steps_are_recognised_as_a_lock_not_as_random():
    # steps every 2/3 of a pulse starting on a pulse: a 3-against-2 cadence that the plain beat / 'and' grid cannot see
    inp = rh.synthetic_inputs("on_beat", jitter_ms=10.0)
    P = 60.0 / 84.0
    t0 = inp.grid.t[0]
    tr = [t0 + k * (2.0 / 3.0) * P for k in range(int((inp.grid.t[-1] - t0) / ((2.0 / 3.0) * P)) - 1)]
    for r in rh.ROLES:
        inp.steps[r] = [{"t": x + float(np.random.default_rng(i).normal(0, 0.01)), "foot": "left", "sigma": 0.015} for i, x in enumerate(tr)]
    d = rh.run_analysis(inp, CFG, with_sensitivity=False)
    assert d["final"]["levels"]["lead"].startswith("triplet")
    assert d["final"]["components"]["lock"] > 0.6


def test_the_live_score_is_causal():
    inp = rh.synthetic_inputs("on_beat", seed=1)
    ev = rh.build_events(inp, CFG)
    full = rh.live_timeline(inp, ev, CFG)
    cut = 40.0
    inp2 = rh.synthetic_inputs("on_beat", seed=1)
    for r in rh.ROLES:
        inp2.steps[r] = [s for s in inp2.steps[r] if s["t"] <= cut]
    ev2 = rh.build_events(inp2, CFG)
    part = rh.live_timeline(inp2, ev2, CFG)
    i = int((cut - 0.3 - inp.t0) * CFG.live_hz)
    for k in ("score", "lock", "accents", "tempo", "flow", "sync"):
        assert abs(full[k][i] - part[k][i]) < 1e-9, k


def test_the_live_score_starts_neutral_and_rises_for_a_good_dance_and_falls_for_a_bad_one():
    good = rh.analyse_one(rh.synthetic_inputs("on_beat"), CFG)["live"]
    bad = rh.analyse_one(rh.synthetic_inputs("random"), CFG)["live"]
    assert abs(good["score"][0] - 50.0) < 1e-6 and abs(bad["score"][0] - 50.0) < 1e-6
    assert good["score"][-1] > 85
    assert bad["score"][-1] < 15
    # it falls where a dancer drifts off the beat
    inp = rh.synthetic_inputs("on_beat", seed=2)
    mid = 0.5 * (inp.grid.t[0] + inp.grid.t[-1])
    for r in rh.ROLES:
        for s in inp.steps[r]:
            if s["t"] > mid:
                s["t"] += 0.185
    live = rh.analyse_one(inp, CFG)["live"]
    i_mid = int((mid - inp.t0) * CFG.live_hz)
    assert live["score"][i_mid] > live["score"][-1] + 15


def test_the_score_is_frozen_in_the_no_beat_stretches():
    inp = rh.synthetic_inputs("on_beat", seed=4)
    live = rh.analyse_one(inp, CFG)["live"]
    st = live["state"]
    assert st[0] == 1                                                  # before the first slot: no song
    pre = np.where(st == 1)[0]
    assert np.allclose(live["score"][pre[pre < 100]], 50.0)           # neutral and frozen until the beat starts
    assert live["confidence"][0] == 0.0
    assert live["state"][-1] == 1 and abs(live["score"][-1] - live["score"][np.where(st == 0)[0][-1]]) < 1e-9


def test_a_seam_gap_in_the_grid_is_found_and_frozen():
    inp = rh.synthetic_inputs("on_beat", seed=5)
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


def test_the_audio_video_shift_moves_the_score_a_little_for_a_tight_dancer():
    inp = rh.synthetic_inputs("on_beat", jitter_ms=15.0)
    s = rh.sensitivity(inp, CFG)
    assert len(s["rows"]) == 3 and s["range"][0] <= s["range"][1]
    assert s["range"][1] - s["range"][0] < 12                          # +-30 ms is inside the kernel


def test_the_final_equals_the_whole_dance_and_the_live_ends_near_it():
    d = _final("on_beat")
    assert abs(d["live"]["score"][-1] - d["final"]["score"]) < 6


def test_output_is_json_and_complete():
    d = _final("on_beat")
    txt = json.dumps(d)
    back = json.loads(txt)
    for k in ("final", "final_card", "describe", "steps", "measures", "windows", "live", "no_beat", "claims", "title", "formula", "config"):
        assert k in back
    live = back["live"]
    n = live["n"]
    assert all(len(live[k]) == n for k in ("score", "lock", "accents", "tempo", "flow", "sync", "confidence", "state"))
    assert "not an official score" in back["title"]
    assert back["final_card"]["to_t"] > back["final_card"]["from_t"]
    assert abs(sum(back["formula"]["weights"].values()) - 1.0) < 1e-9
    assert back["measures"] and all("score" in m for m in back["measures"])


def test_lock_chance_is_what_uniform_phases_score():
    rng = np.random.default_rng(0)
    d = rng.uniform(-185.5, 185.5, 200000)
    mc = float(np.exp(-0.5 * (d / 60.0) ** 2).mean())
    assert abs(mc - rh.lock_chance(60.0, 371.0)) < 0.003


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
    inp = rh.synthetic_inputs("on_beat")
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
    tp = tmp_path / "timing.json"
    tp.write_text(json.dumps(d))
    work = tmp_path / "work"
    (work / "timing").mkdir(parents=True)
    (work / "timing" / "timing.json").write_text(json.dumps(d))
    inp = rh.load_inputs("fake", None, None, None, CFG, work)
    assert len(inp.grid) == 160 and inp.music.source == "grid" and not inp.meta["av_pinned"]
    assert any(k == "none" for _, _, k in inp.grid.no_beat)
    doc = rh.run_analysis(inp, CFG, with_sensitivity=False)
    assert doc["final"]["score"] > 80


def test_nothing_crashes_with_no_steps():
    inp = rh.synthetic_inputs("on_beat")
    inp.steps = {"lead": [], "follow": []}
    d = rh.run_analysis(inp, CFG, with_sensitivity=True)
    assert d["final"]["score"] is None or (math.isfinite(d["final"]["score"]) and d["final"]["score"] <= 55)
    json.dumps(d, allow_nan=False)                                     # unknown values are null, never NaN


def test_install_writes_the_viewers_copy_and_names_it_in_capture_json(tmp_path):
    d = _final("on_beat")
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
    assert [c["label"] for c in v["components"]] == ["Beat lock", "Accents", "Tempo", "Flow", "Couple sync"]
    assert all(r[2] in (0, 1, 2) and r[1] in (0, 1) for r in v["steps"]["rows"])
    assert (cap / "rhythm.json").stat().st_size < 600_000
