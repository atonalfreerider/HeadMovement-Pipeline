"""dancecap.timing_free: the beat-free step statistics (pure numpy, no fits)."""
import json

import numpy as np

from dancecap import timing_free as tf


def test_intervals_sorted_and_empty():
    assert np.allclose(tf.intervals([3.0, 1.0, 2.5]), [1.5, 0.5])
    assert len(tf.intervals([1.0])) == 0
    assert len(tf.intervals([])) == 0


def test_bouts_split_on_gap_and_drop_single_steps():
    times = [0.0, 0.5, 1.0, 1.5, 10.0, 20.0, 20.4, 20.8]
    b = tf.bouts(times, gap_s=1.5)
    assert [(x["t0"], x["t1"], x["n"]) for x in b] == [(0.0, 1.5, 4), (20.0, 20.8, 3)]
    assert b[0]["cadence_hz"] == 2.0           # 3 intervals in 1.5 s
    assert tf.bouts([], 1.5) == []
    assert tf.bouts([5.0], 1.5) == []          # one step is not a pattern


def test_steps_per_window_counts():
    w = tf.steps_per_window([0.5, 1.0, 9.9, 10.0, 25.0], 0.0, 30.0, win_s=10.0)
    assert [x["n"] for x in w] == [3, 1, 1]
    assert w[0]["t0"] == 0.0 and w[-1]["t1"] == 30.0


def test_couple_offsets_signed_nearest_within_limit():
    lead = [1.0, 2.0, 3.0, 10.0]
    follow = [1.05, 2.1, 2.98, 20.0]
    o = tf.couple_offsets(lead, follow, max_s=0.5)
    assert o["n_pairs"] == 3 and abs(o["share_of_lead_steps_paired"] - 0.75) < 1e-9
    assert o["median_ms"] == 50.0                 # offsets +50, +100, -20 ms
    assert tf.couple_offsets([], follow)["n_pairs"] == 0
    assert tf.couple_offsets(lead, [])["median_ms"] is None


def test_step_summary_per_foot_and_bouts_json_safe():
    ev = [{"t": t, "foot": f} for t, f in [(1.0, "left"), (1.5, "right"), (2.0, "left"), (2.5, "right"), (30.0, "left")]]
    s = tf.step_summary(ev, 0.0, 40.0)
    assert s["n_touchdowns"] == 5 and s["n_bouts"] == 1 and s["per_foot"]["left"]["n"] == 3
    assert s["per_foot"]["right"]["interval"]["median_s"] == 1.0
    assert abs(s["share_in_bouts"] - 1.5 / 40.0) < 1e-3
    assert s["longest_still_s"] == 27.5
    json.dumps(s)                                  # plain python types only


def test_document_declares_what_is_skipped():
    assert any("beat grid" in x for x in tf.SKIPPED)
    assert any("asynchrony" in x for x in tf.SKIPPED)


def test_split_windows_tile_the_take_with_margins():
    t = np.arange(0, 100.0, 1 / 30)             # 3000 samples
    w = tf.split_windows(t, core_s=30.0, margin_s=3.0)
    cores = [(c0, c1) for _, _, c0, c1 in w]
    assert cores[0][0] == 0 and cores[-1][1] == len(t)
    assert all(a[1] == b[0] for a, b in zip(cores[:-1], cores[1:]))        # the cores tile the take
    assert all(i0 <= c0 and c1 <= i1 for i0, i1, c0, c1 in w)
    assert w[0][0] == 0 and w[-1][1] == len(t)                              # no context outside the take
    assert w[1][2] - w[1][0] == 90 and w[1][1] - w[1][3] == 90              # 3 s of context on both sides
    short = tf.split_windows(np.arange(0, 70.0, 1 / 30), core_s=30.0, margin_s=3.0)   # a 10 s tail joins the last core
    assert len(short) == 2 and short[-1][3] == 2100


def test_in_core_is_half_open():
    ev = [{"t": 1.0}, {"t": 2.0}, {"t": 3.0}]
    assert [e["t"] for e in tf.in_core(ev, 1.0, 3.0)] == [1.0, 2.0]
    assert [e["t"] for e in tf.in_core(ev, 2.0, None)] == [2.0, 3.0]
    assert [e["t"] for e in tf.in_core(ev, -float("inf"), 2.0)] == [1.0]
