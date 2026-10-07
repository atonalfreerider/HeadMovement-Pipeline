"""Counterbalance detector tests: helpers on analytic inputs, the synthetic counterbalance take (work/synthetic_cb,
ground truth in truth.json) incl. VIEWER_SPEC 14's +-1 frame edges, joint / parameter noise, in-memory variants
(clockwise, 720 deg, static lean-away, no lean, two-hand hold), no false positive on the ordinary synthetic take
(work/synthetic), the strict AND of the criteria, analysed coverage with a gap, and the Unity copy (capture clock
time_to_audio != 0, motion check).

    ..\\.venv-nopo4d\\Scripts\\python.exe -m pytest tests/test_counterbalance.py -q      (from dancecap/)
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from dancecap import counterbalance as cb
from dancecap import synthetic as syn

ROOT = Path(__file__).resolve().parents[1]
SYN = ROOT / "work" / "synthetic"
SYN_CB = ROOT / "work" / "synthetic_cb"
MODEL = ROOT.parent / "smplx" / "SMPLX_NEUTRAL.npz"
FPS = 30.0
needs_model = pytest.mark.skipif(not MODEL.exists(), reason="SMPL-X model missing")
needs_syn = pytest.mark.skipif(not (SYN / "smplx" / "lead.npz").exists(), reason="run: python -m dancecap synthetic")


# ----------------------------------------------------------------------------------------------- helpers

def test_ramps():
    assert cb._up(0.5, 0, 1) == pytest.approx(0.5)
    assert cb._up(-1, 0, 1) == 0 and cb._up(2, 0, 1) == 1
    assert cb._down(0.03, 0.03, 0.08) == 1 and cb._down(0.08, 0.03, 0.08) == 0
    assert cb._down(0.055, 0.03, 0.08) == pytest.approx(0.5)


def test_hysteresis_merges_and_drops():
    s = np.zeros(100)
    s[10:30] = 0.8
    s[31:33] = 0.4          # gap of one frame below off -> merged (merge_gap 2)
    s[33:40] = 0.6
    s[60:62] = 0.9          # too short
    s[80:90] = 0.4          # never reaches on
    runs = cb.hysteresis(s, 0.5, 0.35, min_len=5, merge_gap=2)
    assert runs == [(10, 39)]


def test_level_edges_interpolate():
    t = np.arange(10) / 30
    y = np.array([0, 0.2, 0.4, 0.6, 1, 1, 0.8, 0.4, 0.1, 0])
    t0, t1 = cb.level_edges(t, y, 0.5, 2, 7)
    assert t0 == pytest.approx((2 + 0.5) / 30)          # 0.4 -> 0.6 crosses 0.5 half way
    assert t1 == pytest.approx((6 + 0.75) / 30)         # 0.8 -> 0.4 crosses 0.5 at 3/4


def test_best_pair_ties_go_to_the_nearest_pair():
    """With bent arms every pair scores 0: the reported pair must be the nearest one, not pair 0 by default."""
    ps = np.zeros((2, 5))
    pd = np.ones((2, 5))
    pd[0, 3] = 0.08
    ps[1, 2], pd[1, 2], pd[1, 4] = 0.4, 0.9, 0.05        # a real score still wins over a nearer zero-score pair
    assert cb.best_pairs(ps, pd).tolist() == [3, 2]


def test_strict_and_of_criteria():
    """A weak criterion cannot be outvoted: capping any one at 0.07 caps both tiers (geometric means did not)."""
    full = {c: np.ones(50) for c in cb.CRITERIA}
    for c in cb.CRITERIA:
        s = dict(full)
        s[c] = np.full(50, 0.07)
        cbs, pvs = cb.tier_scores(s)
        assert pvs.max() <= 0.07
        assert cbs.max() <= (0.07 if c in ("lean", "connection") else 1.0)
        assert cb.hysteresis(pvs, 0.5, 0.35, 5, 2) == []


def test_point_drift_ignores_jitter_but_not_a_slide():
    rng = np.random.default_rng(0)
    t = np.arange(25) / 30
    lin, p90 = [], []
    for _ in range(50):                                        # raw (unsmoothed) 2 cm jitter of a planted point
        still = rng.normal(0, 0.02, (25, 2))
        lin.append(cb.point_drift(t, still))
        p90.append(cb.point_drift(t, still, "p90"))
    assert np.mean(lin) < 0.6 * np.mean(p90) and np.mean(lin) < 0.03
    slide = np.stack([np.linspace(0, 0.10, 25), np.zeros(25)], 1)
    assert cb.point_drift(t, slide) == pytest.approx(0.10, abs=1e-9)
    step = np.zeros((25, 2))
    step[12:, 0] = 0.3                                         # a step of 30 cm
    assert cb.point_drift(t, step) > 0.2


def test_pivot_point_of_rotating_foot():
    """A foot turning about its ball (s = 1) and about a point 30 % along it: the least-moving point is found."""
    th = np.linspace(0, math.pi, 40)
    for s_true in (1.0, 0.3):
        piv = np.array([0.5, -0.2])
        d = 0.15 * np.stack([np.cos(th), np.sin(th)], 1)      # ankle -> ball, rotating
        a = piv - s_true * d
        s = cb.pivot_s(a, d, -0.2, 1.2)
        assert s == pytest.approx(s_true, abs=1e-6)
        assert np.abs(a + s * d - piv).max() < 1e-9
    a = np.zeros((10, 2))
    d = np.tile([0.15, 0.0], (10, 1))
    assert cb.pivot_s(a, d, -0.2, 1.2) == 1.0


def test_floor_angle_is_ccw_from_above():
    # +x -> -z is counter-clockwise seen from above in the right-handed Y-up world (positive about +Y)
    a = cb.floor_angle(np.array([[1.0, 0.0], [0.0, -1.0]]))
    assert a[1] - a[0] == pytest.approx(math.pi / 2)


def test_contiguous_segments():
    t = np.concatenate([np.arange(10), np.arange(20, 30)]) / 30
    assert cb.contiguous_segments(t) == [(0, 9), (10, 19)]
    assert cb.contiguous_segments(np.arange(10) / 30) == [(0, 9)]


def test_beats_from_zouk_grid():
    sec = 60 / 76.4
    grid = [[round(4 * sec * m + sec * j / 2, 6), 1 if j in (0, 4) else (2 if j in (3, 6) else 3)]
            for m in range(4) for j in range(8)][3:]          # starts mid-measure
    t = np.array([r[0] for r in grid])
    ty = np.array([r[1] for r in grid])
    beat, bt, meas = cb.beats_from_grid((t, ty))
    assert beat == pytest.approx(sec, rel=1e-4)
    assert np.allclose(np.diff(bt), sec, atol=1e-5)
    assert np.allclose(np.diff(meas), 4 * sec, atol=1e-5)
    assert meas[0] == pytest.approx(4 * sec, abs=1e-5)


def _pivot_doc():
    piv = {"t0": 10.2, "t1": 12.0, "coarse": [10.3, 11.9], "pivot": [1.0, 0.0, 2.0], "swept_signed_deg": 360.0,
           "arc_start_deg": 90.0, "leader_path": {"t": [10.2], "pos": [[0.0, 0.0, 3.0]]},
           "beat_ticks": [{"t": 10.5, "pos": [0.5, 0.0, -1.0]}]}
    return {"version": 1, "take": "x", "frame": "world", "time_base": "reference", "notes": [],
            "coverage": [[9.0, 13.0]], "track": {"t": [10.0, 10.1]},
            "intervals": [{"kind": "pivot", "t0": 10.0, "t1": 12.0, "coarse": [10.1, 11.9], "pivot_t0": 10.2,
                           "pivot_t1": 12.0, "pivot": [1.0, 0.0, 2.0], "swept_signed_deg": 360.0,
                           "arc_start_deg": 90.0, "com_track": {"t": [10.0], "com": [[1.0, 0.9, 2.0]]},
                           "leader_path": {"t": [10.2], "pos": [[0.0, 0.0, 3.0]]},
                           "beat_ticks": [{"t": 10.5, "pos": [0.5, 0.0, -1.0]}], "pivots": [piv]},
                          {"kind": "counterbalance", "t0": 12.5, "t1": 12.9, "coarse": [12.5, 12.9],
                           "pivot_t0": None, "pivot_t1": None, "pivot": None, "swept_signed_deg": None,
                           "arc_start_deg": None, "com_track": {"t": [12.5], "com": [[0.0, 0.9, 1.0]]},
                           "leader_path": None, "beat_ticks": [], "pivots": []}]}


def test_unity_doc_conversion():
    doc = _pivot_doc()
    u = cb.unity_doc(doc, -29.97, "cap", "ok")
    iv = u["intervals"][0]
    assert u["frame"] == "unity" and u["time_base"] == "capture" and u["time_to_audio"] == -29.97
    assert iv["t0"] == pytest.approx(10.0 - 29.97) and iv["t0_reference"] == 10.0
    assert iv["pivot_t0"] == pytest.approx(10.2 - 29.97) and iv["pivot_t1_reference"] == 12.0
    assert iv["pivot"] == [1.0, 0.0, -2.0]
    assert iv["com_track"]["com"] == [[1.0, 0.9, -2.0]] and iv["com_track"]["t"] == [pytest.approx(10.0 - 29.97)]
    assert iv["beat_ticks"][0]["pos"] == [0.5, 0.0, 1.0] and iv["beat_ticks"][0]["t"] == pytest.approx(10.5 - 29.97)
    assert iv["leader_path"]["pos"] == [[0.0, 0.0, -3.0]]
    assert iv["swept_signed_deg_unity_up"] == -360.0
    pv = iv["pivots"][0]
    assert pv["t0"] == pytest.approx(10.2 - 29.97) and pv["pivot"] == [1.0, 0.0, -2.0]
    assert pv["swept_signed_deg_unity_up"] == -360.0
    cbi = u["intervals"][1]                                   # a counterbalance without a pivot: nulls survive
    assert cbi["pivot"] is None and cbi["leader_path"] is None and cbi["swept_signed_deg_unity_up"] is None
    assert u["track"]["t"] == [pytest.approx(10.0 - 29.97), pytest.approx(10.1 - 29.97)]
    assert u["coverage"] == [[pytest.approx(9.0 - 29.97), pytest.approx(13.0 - 29.97)]]
    assert u["coverage_reference"] == [[9.0, 13.0]]
    assert doc["intervals"][0]["pivot"] == [1.0, 0.0, 2.0]       # source untouched


# ----------------------------------------------------------------------------------------------- synthetic takes

@pytest.fixture(scope="module")
def cb_truth():
    if not (SYN_CB / "smplx" / "lead.npz").exists() or not (SYN_CB / "truth.json").exists():
        if not MODEL.exists():
            pytest.skip("SMPL-X model missing")
        from dancecap import synthetic_counterbalance as sc
        sc.make()
    tr = json.loads((SYN_CB / "truth.json").read_text())
    if tr["counterbalance"] and "pivot_t0" not in tr["counterbalance"][0]:
        pytest.skip("old truth.json: run python -m dancecap.counterbalance synthetic")
    return tr


def _beats(rows) -> tuple:
    g = (np.array([r[0] for r in rows]), np.array([r[1] for r in rows]))
    beat, bt, _ = cb.beats_from_grid(g)
    return beat, bt


def _detect(work: Path, t0: float = 0.0, t1: float = 1e9, **noise) -> tuple[dict, dict]:
    w = syn.window(work, t0, t1, **noise)
    beat, bt = _beats(w["beats"])
    res = cb.detect(w["npz"], beat_s=beat, beat_times=bt)
    return res, cb.to_json(res, work.name, {})


def _assert_matches_truth(doc: dict, truth: dict, iou: float = 0.9, edge_frames: float = 5.0,
                          pivot_edge_frames: float = 3.0, com_m: float = 0.02) -> dict:
    ev = cb.evaluate(doc, truth)
    assert ev["n_detected"] == 1, ev
    assert ev["iou"] >= iou and ev["pivot_iou"] >= iou, ev
    assert ev["n_pivots"] == 1, ev                           # the pivot does not blink off mid-way
    assert max(abs(v) for v in ev["edge_err_frames"].values()) <= edge_frames, ev
    assert max(abs(v) for v in ev["pivot_edge_err_frames"].values()) <= pivot_edge_frames, ev
    assert ev["pivot_err_m"] <= 0.03, ev
    assert ev["swept_err_deg"] <= 15.0, ev
    assert ev["direction_ok"] and ev["pivot_foot_ok"], ev
    assert ev["com_floor_err_m"]["max"] <= com_m, ev
    return ev


@needs_model
def test_truth_is_self_consistent(cb_truth):
    g = cb_truth["counterbalance"][0]
    assert g["kind"] == "pivot"
    assert g["pivot_drift_max_m"] < 1e-6                    # the anchored ball really is fixed
    assert g["swept_deg"] == pytest.approx(360.0, abs=1e-3)
    assert 0.8 <= g["leader_radius_m"] <= 1.0
    assert g["free_foot_clearance_min_m"] > 0.03           # the free foot is off the floor
    assert g["pivot_t1"] - g["pivot_t0"] == pytest.approx(2 * 4 * 60 / 76.4, abs=1e-4)   # two measures at 76.4 BPM
    # the counterbalance (yellow axis) contains the pivot and holds throughout it
    assert g["t0"] <= g["pivot_t0"] and g["pivot_t1"] <= g["t1"]
    assert g["counterbalance_check"]["min_score_in_pivot"] >= 0.5
    assert g["setup"][0] <= g["t0"] and g["t1"] <= g["release"][1]   # lean-in / -out inside set-up and release


@needs_model
def test_synthetic_edges_within_one_frame(cb_truth):
    """VIEWER_SPEC 14: the yellow axis within +-1 frame of the truth counterbalance interval, the pivot ring / arc
    within +-1 frame of the pivot; dot under the combined COM <= 2 cm; pivot ring at the anchored foot <= 3 cm."""
    res, doc = _detect(SYN_CB)
    ev = cb.evaluate(doc, cb_truth)
    assert ev["n_detected"] == 1 and ev["n_pivots"] == 1, ev
    for k in ("t0", "t1"):
        assert abs(ev["edge_err_frames"][k]) <= 1.0, ev
        assert abs(ev["pivot_edge_err_frames"][k]) <= 1.0, ev
    assert ev["com_floor_err_m"]["max"] <= 0.02, ev
    assert ev["pivot_err_m"] <= 0.03, ev
    assert ev["swept_err_deg"] <= 15.0 and ev["direction_ok"] and ev["pivot_foot_ok"], ev
    iv = doc["intervals"][0]
    assert iv["kind"] == "pivot"
    assert iv["connection"]["kind"] == "hand-hand"
    assert iv["connection"]["pair"] == "lead_r_hand-follow_l_hand"
    assert iv["confidence"] > 0.8 and iv["pivot_confidence"] > 0.8
    assert abs(iv["leader_radius_m"] - cb_truth["counterbalance"][0]["leader_radius_m"]) < 0.03
    assert len(iv["beat_ticks"]) >= 7                       # ~8 beats in the pivot
    lp = iv["leader_path"]
    assert lp["t"][0] == pytest.approx(iv["pivot_t0"], abs=1e-3) and lp["t"][-1] == pytest.approx(iv["pivot_t1"], abs=1e-3)


@needs_model
@pytest.mark.parametrize("noise,seed", [(0.01, 0), (0.01, 1), (0.02, 2), (0.03, 3)])
def test_synthetic_counterbalance_joint_noise(cb_truth, noise, seed):
    _, doc = _detect(SYN_CB, joint_noise_m=noise, seed=seed)
    _assert_matches_truth(doc, cb_truth)


@needs_model
@pytest.mark.parametrize("noise,seed", [(0.01, 0), (0.02, 0), (0.02, 1), (0.02, 5)])
def test_synthetic_counterbalance_parameter_noise(cb_truth, noise, seed):
    """Noisy SMPL-X parameters (joints dropped -> FK joints): transl noise and the same in degrees on every joint
    rotation. 2 cm / 2 deg used to split the pivot in two (the axis blinking off mid-pivot)."""
    _, doc = _detect(SYN_CB, transl_noise_m=noise, rot_noise_deg=100 * noise, seed=seed)
    _assert_matches_truth(doc, cb_truth, com_m=0.02 + noise)     # the COM itself carries the transl noise


@needs_model
def test_ordinary_part_of_cb_take_has_none(cb_truth):
    g = cb_truth["counterbalance"][0]
    _, doc = _detect(SYN_CB, 0.0, g["setup"][0])
    assert doc["intervals"] == []
    _, doc = _detect(SYN_CB, g["release"][1], 1e9)
    assert doc["intervals"] == []


@needs_model
@needs_syn
@pytest.mark.parametrize("noise", [0.0, 0.01, 0.02])
def test_ordinary_synthetic_take_has_none(noise):
    res, doc = _detect(SYN, joint_noise_m=noise, seed=3)
    assert doc["intervals"] == []
    assert float(np.max(res["counterbalance"])) < 0.35 and float(np.max(res["pivot_score"])) < 0.35


@needs_model
@needs_syn
def test_ordinary_synthetic_six_second_windows():
    """The real take is analysed in 6 s windows: none of the ordinary synthetic windows may fire."""
    for t0 in (0.0, 6.0, 12.0, 18.0):
        _, doc = _detect(SYN, t0, t0 + 6.0, joint_noise_m=0.01, seed=int(t0))
        assert doc["intervals"] == [], t0


@needs_model
def test_json_contract(cb_truth, tmp_path):
    res = cb.run("synthetic_cb", review=False, out_path=tmp_path / "counterbalance.json")
    doc = json.loads((tmp_path / "counterbalance.json").read_text())
    assert doc["version"] == 1 and doc["frame"] == "world" and doc["time_base"] == "reference"
    assert doc["take"] == "synthetic_cb" and isinstance(doc["params"], dict)
    assert doc["coverage"] == [[pytest.approx(res["t"][0], abs=1e-3), pytest.approx(res["t"][-1], abs=1e-3)]]
    iv = doc["intervals"][0]
    for k in ("t0", "t1", "confidence", "pivot", "pivot_foot", "leader_radius_m", "swept_deg", "direction",
              "connection", "scores", "com_track", "kind", "pivot_t0", "pivot_t1", "pivots"):
        assert k in iv, k
    assert iv["kind"] in cb.KINDS
    assert set(iv["scores"]) == {"anchor", "circling", "lean", "connection"}
    assert all(0 <= v <= 1 for v in iv["scores"].values()) and 0 <= iv["confidence"] <= 1
    assert iv["pivot_foot"] in ("left", "right") and iv["direction"] in ("cw", "ccw")
    assert len(iv["pivot"]) == 3 and iv["pivot"][1] == 0.0
    assert iv["t0"] <= iv["pivot_t0"] < iv["pivot_t1"] <= iv["t1"]
    ct = iv["com_track"]
    assert len(ct["t"]) == len(ct["com"]) and all(len(c) == 3 for c in ct["com"])
    assert ct["t"][0] < iv["t0"] and ct["t"][-1] > iv["t1"]    # margin for the viewer's fades
    assert doc["source"]["smplx_dir"].replace("\\", "/").endswith("synthetic_cb/smplx")
    assert doc["source"]["t_range"] == doc["coverage"][0]
    assert len(doc["source"]["lead_probe"]["t"]) == len(doc["source"]["lead_probe"]["transl"]) >= 2
    tr = doc["track"]
    assert len(tr["t"]) == len(tr["counterbalance"]) == len(tr["pivot"]) == len(tr["combined"]) == len(res["t"])


@needs_model
def test_gap_in_time_grid_is_two_segments(cb_truth):
    """Frames missing before the pivot: analysed as two segments; coverage says so; the pivot is still found."""
    w = syn.window(SYN_CB, 0.0, 1e9)
    t = np.asarray(w["npz"]["lead"]["t"])
    keep = ~((t > 3.0) & (t < 4.0))
    npzs = {r: {k: (np.asarray(v)[keep] if k != "betas" and np.ndim(v) >= 1 and len(v) == len(t) else v)
                for k, v in w["npz"][r].items()} for r in cb.ROLES}
    beat, bt = _beats(w["beats"])
    res = cb.detect(npzs, beat_s=beat, beat_times=bt)
    doc = cb.to_json(res, "gap", {})
    assert len(doc["coverage"]) == 2 and doc["coverage"][0][1] <= 3.0 and doc["coverage"][1][0] >= 4.0
    assert len(doc["track"]["t"]) == int(keep.sum())
    ev = cb.evaluate(doc, cb_truth)
    assert ev["n_detected"] == 1 and ev["pivot_iou"] > 0.95, ev


@needs_model
def test_joint_com_fallback_warns(cb_truth):
    """Without SMPL-X parameters the joint-segment COM is used - loudly (RuntimeWarning + params.warnings)."""
    w = syn.window(SYN_CB, 8.0, 18.0)
    npzs = {r: {"t": w["npz"][r]["t"], "joints": w["npz"][r]["joints"]} for r in cb.ROLES}
    beat, bt = _beats(w["beats"])
    with pytest.warns(RuntimeWarning, match="joint-segment COM"):
        res = cb.detect(npzs, beat_s=beat, beat_times=bt)
    assert len(res["info"]["warnings"]) == 2
    assert all("fallback" in v for v in res["info"]["com_source"].values())


# ----------------------------------------------------------------------------------------------- in-memory variants

def _variant(**kw) -> tuple[dict, dict, object]:
    from dancecap import synthetic_counterbalance as sc
    sys.path.insert(0, str(ROOT.parent / "atlas_bridge"))
    from atlas_bridge.beats import zouk_grid
    spec = replace(sc.Spec(), **kw)
    g = sc.generate(spec)
    T = len(g["t"])
    npzs = {r: dict(t=g["t"], global_orient=g[r]["global_orient"], body_pose=g[r]["body_pose"],
                    left_hand_pose=np.zeros((T, 45)), right_hand_pose=np.zeros((T, 45)), jaw_pose=np.zeros((T, 3)),
                    transl=g[r]["transl"], betas=sc.BETAS[r], joints=g[r]["joints"], gender=np.array("neutral"))
            for r in sc.ROLES}
    beat, bt = _beats(zouk_grid(sc.BPM, spec.duration + 1.0))
    res = cb.detect(npzs, beat_s=beat, beat_times=bt)
    return res, cb.to_json(res, "variant", {}), g["spec"]


@needs_model
@pytest.mark.parametrize("turns", [-1.0, 2.0])
def test_variant_direction_and_turns(turns):
    res, doc, spec = _variant(turns=turns)
    assert len(doc["intervals"]) == 1
    iv = doc["intervals"][0]
    assert iv["kind"] == "pivot" and len(iv["pivots"]) == 1
    assert iv["direction"] == ("ccw" if turns > 0 else "cw")
    assert iv["swept_deg"] == pytest.approx(abs(turns) * 360.0, abs=15.0)
    assert abs(iv["pivot_t0"] - spec.t1) * FPS <= 1.0 and abs(iv["pivot_t1"] - spec.t2) * FPS <= 1.0


@needs_model
def test_variant_static_lean_away_is_counterbalance_without_pivot(cb_truth):
    """turns = 0: leaning away, hands joined, no circling -> the yellow axis (kind 'counterbalance'), no ring / arc.
    The lean schedule equals the default take's, so its counterbalance truth applies."""
    res, doc, spec = _variant(turns=0.0)
    assert len(doc["intervals"]) == 1
    iv = doc["intervals"][0]
    assert iv["kind"] == "counterbalance" and iv["pivots"] == [] and iv["pivot"] is None
    g = cb_truth["counterbalance"][0]
    assert abs(iv["t0"] - g["t0"]) * FPS <= 1.0 and abs(iv["t1"] - g["t1"]) * FPS <= 1.0


@needs_model
def test_variant_no_lean_is_not_counterbalance():
    """The leader circles the anchored foot with a straight-arm hold but nobody leans away: nothing (the old
    geometric mean reached 0.67 here and only the minimum duration dropped it)."""
    res, doc, _ = _variant(lean_follow_deg=0.0, lean_lead_deg=0.0)
    assert doc["intervals"] == []
    assert float(res["counterbalance"].max()) < 0.35


@needs_model
def test_variant_two_hand_hold():
    """Spec(hold='two'): both pairs really joined (generate() refuses otherwise) and detected as one pivot."""
    res, doc, spec = _variant(hold="two")
    assert spec.turn_deg == 0.0
    assert len(doc["intervals"]) == 1 and doc["intervals"][0]["kind"] == "pivot"
    assert doc["intervals"][0]["connection"]["kind"] == "hand-hand"


# ----------------------------------------------------------------------------------------------- Unity export

def _fake_capture(root: Path, name: str, t_ref: np.ndarray, lead_transl: np.ndarray, time_to_audio: float) -> Path:
    from dancecap.export_unity import N_JOINTS, to_unity_points, write_motion
    d = root / name
    d.mkdir(parents=True)
    T = len(t_ref)
    (d / "times.json").write_text(json.dumps([round(float(x), 6) for x in t_ref]))
    (d / "capture.json").write_text(json.dumps({"version": 3, "fps": 30.0, "frame_count": T, "times": "times.json",
                                                "time_to_audio": time_to_audio,
                                                "audio_offset": float(t_ref[0] + time_to_audio),
                                                "provenance": {"take": "synthetic_cb"}}))
    write_motion(d / "smplx_lead.bin", "lead", t_ref, to_unity_points(lead_transl), np.zeros((T, N_JOINTS, 4)),
                 np.zeros((T, N_JOINTS, 3)))
    return d


@needs_model
def test_export_uses_capture_clock_and_checks_motion(cb_truth, tmp_path):
    """capture time = reference + capture.json time_to_audio (the viewer's audio clock, as the moves copies);
    motion mismatch and non-overlapping captures are refused."""
    from dancecap.moves.labels import CaptureClock
    src = tmp_path / "counterbalance.json"
    cb.run("synthetic_cb", review=False, out_path=src)
    doc = json.loads(src.read_text())
    npz = cb.load_npzs(SYN_CB / "smplx")["lead"]
    t, tr = np.asarray(npz["t"], float), np.asarray(npz["transl"], float)
    streaming = tmp_path / "StreamingAssets"
    cap = _fake_capture(streaming, "CAP", t, tr, -29.97)
    rep = cb.export("synthetic_cb", "CAP", streaming=streaming, src=src, mirror=False)
    assert rep["time_to_audio"] == pytest.approx(-29.97) and rep["motion_check"].startswith("ok")
    u = json.loads((cap / "counterbalance.json").read_text())
    clock = CaptureClock.from_capture(cap)
    iv, iv_ref = u["intervals"][0], doc["intervals"][0]
    assert u["time_base"] == "capture" and u["time_to_audio"] == pytest.approx(-29.97)
    assert iv["t0"] == pytest.approx(clock.to_capture(iv_ref["t0"]), abs=1e-4)
    assert iv["pivot_t1"] == pytest.approx(clock.to_capture(iv_ref["pivot_t1"]), abs=1e-4)
    assert iv["t0_reference"] == iv_ref["t0"]
    assert u["track"]["t"][0] == pytest.approx(doc["track"]["t"][0] - 29.97, abs=1e-4)
    assert iv["pivot"][2] == pytest.approx(-iv_ref["pivot"][2])
    assert json.loads((cap / "capture.json").read_text())["counterbalance"] == "counterbalance.json"
    assert cb.capture_matches(cap, "synthetic_cb")            # `synthetic` would not wipe this capture
    # a capture holding different motion is refused (unless forced)
    bad = _fake_capture(streaming, "BAD", t, tr + np.array([0.5, 0.0, 0.0]), 0.0)
    assert not cb.capture_matches(bad, "synthetic_cb")
    with pytest.raises(ValueError, match="differs"):
        cb.export("synthetic_cb", "BAD", streaming=streaming, src=src, mirror=False)
    rep = cb.export("synthetic_cb", "BAD", streaming=streaming, src=src, mirror=False, force=True)
    assert rep["motion_check"] == "MISMATCH (forced)"
    # a capture of another window is refused
    _fake_capture(streaming, "OTHER", t + 100.0, tr, 0.0)
    with pytest.raises(ValueError, match="overlap"):
        cb.export("synthetic_cb", "OTHER", streaming=streaming, src=src, mirror=False)
