"""Floor craft tests: the leader axis from the pose parameters (and the shoulder fallback), plants / moves / rotations on
synthetic motion (a plant, a move, a rotation inside a plant, a rotation across a move), the held plant axis, one stand
cut in two by the greedy search (merged), two stands without travel between them (a shift; a turn across it stays one
rotation), a counterbalance pivot cutting the moves / rotations (the pivot owns the leader's circling), sway in place vs
slow travel, the zig-zag filter, the JSON contract, the Unity copy (capture clock, z mirror, Euler Y) and the real take
when its fits exist.

    ..\\.venv-nopo4d\\Scripts\\python.exe -m pytest tests/test_floorcraft.py -q      (from dancecap/)
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from dancecap import counterbalance as cb
from dancecap import floorcraft as fc

ROOT = Path(__file__).resolve().parents[1]
FPS = 30.0
BEAT = 60.0 / 76.4
V3_TYPES = {"t_axis", "transition", "axis_pivot", "dial"}     # version-3 event types


def ease(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def make_npz(t: np.ndarray, xz: np.ndarray, yaw_deg: np.ndarray, pose: bool = True) -> dict:
    """Lead npz with the chest (spine3) at xz (T, 2) [x, z], 1.3 m high, facing the floor angle yaw_deg (from +x
    towards -z): global_orient = rotation about +Y by 90 + yaw (SMPL-X faces +z at rest), spine chain at rest;
    shoulders 0.18 m to either side along the chest's left."""
    T = len(t)
    th = np.radians(90.0 + np.asarray(yaw_deg, float))
    left = np.stack([np.cos(th), np.zeros(T), -np.sin(th)], 1)        # R_y(th) @ (1, 0, 0)
    chest = np.stack([xz[:, 0], np.full(T, 1.3), xz[:, 1]], 1)
    J = np.repeat(chest[:, None, :], 55, 1).copy()
    J[:, fc.L_SHOULDER] = chest + 0.18 * left + [0, 0.1, 0]
    J[:, fc.R_SHOULDER] = chest - 0.18 * left + [0, 0.1, 0]
    npz = {"t": t, "joints": J, "transl": chest - [0, 1.0, 0], "betas": np.zeros(10)}
    if pose:
        npz["global_orient"] = np.stack([np.zeros(T), th, np.zeros(T)], 1)
        npz["body_pose"] = np.zeros((T, 63))
    return npz


def scenario(sway_m: float = 0.04, seed: int = 0) -> tuple[dict, dict]:
    """Plant A (0, 0) 0-3 s facing 30 deg; move to B (1.5, 0) 3-4.5 s; plant B 4.5-9 s, turning +90 deg (ccw)
    6-7 s; move to C (1.5, 1.2) 9-10.5 s turning -60 deg on the way (9.3-10.2 s); plant C 10.5-13 s. In-place sway
    +-sway_m."""
    t = np.arange(0, 13.0, 1 / FPS)
    x = 1.5 * ease((t - 3.0) / 1.5)
    z = 1.2 * ease((t - 9.0) / 1.5)
    yaw = 30.0 + 90.0 * ease(t - 6.0) - 60.0 * ease((t - 9.3) / 0.9)
    sway = sway_m * np.sin(2 * math.pi * t / (2 * BEAT))               # a weight change every beat
    rng = np.random.default_rng(seed)
    xz = np.stack([x + sway, z], 1) + rng.normal(0, 0.005, (len(t), 2))
    truth = {"plants": [((0.0, 0.0), (0.0, 3.0), 30.0), ((1.5, 0.0), (4.5, 9.0), 75.0), ((1.5, 1.2), (10.5, 13.0), 60.0)],
             "turn_in_B": 90.0, "move_BC_turn": -60.0}
    return make_npz(t, xz, yaw + rng.normal(0, 0.5, len(t))), truth


def fake_cbdoc(t0: float, t1: float, point=(1.7, 0.0, 0.6)) -> dict:
    """counterbalance.json with one pivot interval (world, reference s)."""
    tt = np.arange(t0, t1, 0.1)
    com = np.stack([np.full(len(tt), 1.6), np.full(len(tt), 0.95), np.full(len(tt), 0.6)], 1)
    pv = {"t0": t0, "t1": t1, "pivot": list(point), "pivot_foot": "left", "leader_radius_m": 0.5,
          "swept_deg": 360.0, "swept_signed_deg": -360.0, "direction": "cw", "arc_start_deg": 10.0, "confidence": 0.8,
          "leader_path": {"t": [t0, t1], "pos": [[1.5, 0, 0.2], [1.5, 0, 1.0]]},
          "beat_ticks": [{"t": t0 + 0.4, "pos": [1.5, 0.0, 0.3]}]}
    return {"intervals": [{"kind": "pivot", "t0": t0 - 0.1, "t1": t1 + 0.1, "confidence": 0.8,
                           "connection": {"pair": "lead_r_hand-follow_l_shoulder"},
                           "com_track": {"t": tt.tolist(), "com": com.tolist()}, "pivots": [pv]}]}


# ----------------------------------------------------------------------------------------------- axis

@pytest.mark.parametrize("yaw", [-120.0, 0.0, 45.0, 170.0])
def test_axis_yaw_from_global_orient(yaw):
    t = np.arange(0, 2, 1 / FPS)
    npz = make_npz(t, np.zeros((len(t), 2)), np.full(len(t), yaw))
    o, left, src = fc.raw_axis(npz)
    assert src.startswith("spine3")
    ax = fc.smooth_axis(t, o, left, 2.0)
    assert float(fc.wrap180(ax["yaw"][len(t) // 2] - yaw)) == pytest.approx(0.0, abs=0.5)
    f = fc.fwd_of(yaw)
    assert np.allclose(ax["forward"][len(t) // 2], f, atol=1e-3)
    # the long axis is the dancer's LEFT: forward = +x -> left = -z (right-handed, Y up)
    if yaw == 0.0:
        assert np.allclose(fc.long_of_fwd(f), [0.0, -1.0], atol=1e-9)


def test_shoulder_fallback_matches_pose():
    npz, _ = scenario()
    a = fc.analyse(npz, BEAT)
    del npz["global_orient"], npz["body_pose"]
    b = fc.analyse(npz, BEAT)
    assert b["info"]["axis_source"].startswith("shoulder")
    assert np.abs(fc.wrap180(np.asarray(a["track"]["yaw_deg"]) - np.asarray(b["track"]["yaw_deg"]))).max() < 2.0


def test_unwrapped_yaw_keeps_full_turns():
    t = np.arange(0, 4, 1 / FPS)
    npz = make_npz(t, np.zeros((len(t), 2)), -540.0 * ease(t / 4))
    ax = fc.smooth_axis(t, *fc.raw_axis(npz)[:2], 2.0)
    assert ax["yaw"][-1] - ax["yaw"][0] == pytest.approx(-540.0, abs=3.0)


# ----------------------------------------------------------------------------------------------- segmentation

def test_zigzag_swings():
    y = np.concatenate([np.linspace(0, 10, 10), np.linspace(10, 60, 20), np.linspace(60, 50, 10),
                        np.linspace(50, 100, 20), np.linspace(100, 20, 30)])
    piv = fc.zigzag(y, 25.0)
    sw = [y[b] - y[a] for a, b in zip(piv[:-1], piv[1:])]
    assert sw == pytest.approx([100.0, -80.0])          # the 10 deg dip at 60 is sway, not two rotations


def path_ends_at_plants(res: dict) -> None:
    """Every move's path runs from its `from` point to its `to` point (plant crosses joined by the path alone)."""
    for m in res["moves"]:
        assert m["path"]["pos"][0] == m["from"] and m["path"]["pos"][-1] == m["to"], m["id"]
        assert len(m["path"]["t"]) == len(m["path"]["pos"]) >= 2
        P = np.asarray(m["path"]["pos"], float)
        assert m["length_m"] == pytest.approx(float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()), abs=2e-3)


@pytest.mark.parametrize("seed", [0, 1])
def test_plants_moves_rotations_on_synthetic_motion(seed):
    npz, truth = scenario(seed=seed)
    res = fc.analyse(npz, BEAT)
    P = res["plants"]
    assert len(P) == 3, [(p["t0"], p["t1"]) for p in P]
    for p, (xz, (t0, t1), yaw) in zip(P, truth["plants"]):
        assert np.hypot(p["origin"][0] - xz[0], p["origin"][2] - xz[1]) < 0.05, p
        assert p["t0"] <= t0 + 0.6 and p["t1"] >= t1 - 0.6, p                  # most of the held span
        assert p["radius_m"] <= fc.FCConfig().plant_radius_m + 1e-9
        assert p["origin"][1] == 0.0
    assert abs(fc.wrap180(P[0]["yaw_deg"] - 30.0)) < 3 and abs(fc.wrap180(P[2]["yaw_deg"] - 60.0)) < 3
    assert P[1]["turn_deg"] == pytest.approx(90.0, abs=5.0)
    # plant B turns +90 at 6-7 s: its established axis is the one held longest (120 deg, 7-9 s), not the mean (~75)
    assert P[1]["yaw_source"] == "held" and abs(fc.wrap180(P[1]["yaw_deg"] - 120.0)) < 3
    assert P[1]["yaw_mean_deg"] == pytest.approx(75.0, abs=12.0)
    assert P[1]["held"][0] >= 6.9 and P[1]["held"][1] >= 8.5
    M = {(m["from_plant"], m["to_plant"]): m for m in res["moves"]}
    ab, bc = M[(0, 1)], M[(1, 2)]
    assert ab["kind"] == bc["kind"] == "travel"
    assert ab["displacement_m"] == pytest.approx(1.5, abs=0.05) and abs(ab["heading_deg"]) < 3       # along +x
    assert bc["displacement_m"] == pytest.approx(1.2, abs=0.05) and bc["heading_deg"] == pytest.approx(-90, abs=3)
    assert bc["yaw_change_deg"] == pytest.approx(-60.0, abs=5.0) and abs(ab["yaw_change_deg"]) < 5
    # the path runs plant cross -> axis origin -> plant cross (the plants absorb the slow ends of the move)
    path_ends_at_plants(res)
    assert ab["length_m"] >= ab["displacement_m"] - 1e-3 and ab["origin_path_m"] < ab["length_m"]
    assert ab["path"]["pos"][0] == P[0]["origin"] and ab["path"]["pos"][-1] == P[1]["origin"]
    R = res["rotations"]
    in_b = [r for r in R if r["context"] == "plant" and r["plant"] == 1]
    assert len(in_b) == 1 and in_b[0]["signed_deg"] == pytest.approx(90.0, abs=6.0) and in_b[0]["direction"] == "ccw"
    assert 5.8 <= in_b[0]["t0"] <= 6.3 and 6.7 <= in_b[0]["t1"] <= 7.2               # trimmed to the turn itself
    assert in_b[0]["centre"] == P[1]["origin"] and in_b[0]["plants"] == [1]
    in_bc = [r for r in R if r["context"] == "move" and r["move"] == bc["id"]]
    assert len(in_bc) == 1 and in_bc[0]["signed_deg"] == pytest.approx(-60.0, abs=6.0) and in_bc[0]["direction"] == "cw"
    assert len(R) == 2                                                                # no rotation from sway / noise
    ev = res["events"]
    assert [e["t0"] for e in ev] == sorted(e["t0"] for e in ev)
    assert {e["type"] for e in ev} - V3_TYPES == {"plant", "move", "rotation"}


def test_one_stand_cut_by_the_greedy_search_is_merged(monkeypatch):
    """A stand that steps 0.20 m sideways (sway +-4 cm): the greedy search's running radius passes 0.20 m right after
    the step and cuts it in two, but the whole span is within 0.15 m of its centroid with no drift - one plant."""
    t = np.arange(0, 6, 1 / FPS)
    x = 0.20 * ease((t - 3.0) / 0.3) + 0.04 * np.sin(2 * math.pi * t / (2 * BEAT))
    npz = make_npz(t, np.stack([x, np.zeros(len(t))], 1), np.zeros(len(t)))
    res = fc.analyse(npz, BEAT)
    assert len(res["plants"]) == 1 and res["plants"][0]["duration_s"] >= 5.9 and res["moves"] == []
    assert res["plants"][0]["radius_m"] <= 0.2
    monkeypatch.setattr(fc, "merge_plants", lambda t_, o_, spans, cfg, *a: list(spans))
    assert len(fc.analyse(npz, BEAT)["plants"]) == 2                     # what the greedy search alone gives


def test_step_between_touching_stands_is_cut_out(monkeypatch):
    """Two stands 0.35 m apart, the leader stepping from one to the other (3.0-4.0 s, fastest at 3.5 s) while turning
    -90 deg. The greedy search splits the step between the two plants (a 1-frame gap); the step is cut back out of
    them as a 'step' move over its fast part, drawn cross to cross, and the turn across it stays ONE rotation."""
    t = np.arange(0, 8, 1 / FPS)
    x = 0.35 * ease((t - 3.0) / 1.0) + 0.06 * np.sin(2 * math.pi * t / (2 * BEAT))
    yaw = -90.0 * ease((t - 2.9) / 1.2)
    npz = make_npz(t, np.stack([x, np.zeros(len(t))], 1), yaw)
    res = fc.analyse(npz, BEAT)
    P, M, R = res["plants"], res["moves"], res["rotations"]
    assert len(P) == 2 and len(M) == 1, (P, M)
    m = M[0]
    assert m["kind"] == "step" and m["from_plant"] == 0 and m["to_plant"] == 1
    assert m["t0"] < 3.5 < m["t1"] and m["duration_s"] >= 0.25 and m["origin_path_m"] >= 0.15, m
    assert m["displacement_m"] == pytest.approx(0.35, abs=0.06)
    ax = fc.smooth_axis(t, *fc.raw_axis(npz)[:2], 2.0)
    for p in P:                                     # both plants still pass the full plant test without the step
        assert fc.plant_ok(ax["t"], ax["origin"], int(round(p["t0"] * FPS)), int(round(p["t1"] * FPS)), fc.FCConfig())
    path_ends_at_plants(res)
    assert res["info"]["counts"]["steps"] == 1 and res["info"]["counts"]["travel_moves"] == 0
    assert len(R) == 1 and R[0]["signed_deg"] == pytest.approx(-90.0, abs=4.0), R
    assert R[0]["plants"] == [0, 1] and R[0]["swing"]["parts"] == 1
    assert R[0]["centre"] == (P[R[0]["plant"]]["origin"] if R[0]["context"] == "plant" else R[0]["centre"])
    assert [e.get("kind") for e in res["events"] if e["type"] == "move"] == ["step"]
    # without expose_steps the step is a 1-frame gap between the two plants
    monkeypatch.setattr(fc, "expose_steps", lambda t_, o_, spans, *a: spans)
    gap = fc.analyse(npz, BEAT)["moves"][0]
    assert gap["duration_s"] < 0.1 and gap["origin_path_m"] < 0.05 and gap["kind"] == "step"


def pivot_scenario() -> tuple[dict, dict]:
    """Plant A at (0, 0) facing 60 deg; 2.8-3.5 s he turns in place to 0 deg (facing the pivot C = (0.4, 0)); 3.5-7.5 s
    he circles C clockwise (radius 0.4 m, -450 deg, facing C, so his axis winds -450 deg with him); 7.6-8.8 s he walks
    1.2 m along +x; plant B 8.8-11 s."""
    t = np.arange(0, 11, 1 / FPS)
    phi = 180.0 - 450.0 * ease((t - 3.5) / 4.0)                          # his angle about C (from +x towards -z)
    xz = np.stack([0.4 + 0.4 * np.cos(np.radians(phi)), -0.4 * np.sin(np.radians(phi))], 1)
    end = xz[int(np.searchsorted(t, 7.5))].copy()
    walk = ease((t - 7.6) / 1.2)[:, None]
    xz = np.where((t > 7.5)[:, None], end + walk * np.array([1.2, 0.0]), xz)
    yaw = np.where(t < 2.8, 60.0, np.where(t < 3.5, 60.0 - 60.0 * (t - 2.8) / 0.7, phi + 180.0 - 360.0))
    yaw = np.where(t > 7.5, -450.0, yaw)
    return make_npz(t, xz, yaw), {"A": (0.0, 0.0), "B": (end[0] + 1.2, end[1])}


def test_pivot_cuts_moves_and_rotations():
    """The pivot owns the leader's circling: no plant, move or rotation overlaps it; the turn before it stays a
    rotation; his axis winding round inside it is on the pivot (leader_axis); the move after it starts at the pivot."""
    npz, truth = pivot_scenario()
    res = fc.analyse(npz, BEAT, None, fake_cbdoc(3.5, 7.5, (0.4, 0.0, 0.0)))
    pv = res["pivots"][0]
    la = pv["leader_axis"]
    assert la["t0"] <= 3.5 and la["t1"] >= 7.5 and la["t1"] - la["t0"] < 4.0 + 2.5 / FPS
    assert la["turn_deg"] == pytest.approx(-450.0, abs=8.0) and la["direction"] == "cw"
    assert la["length_m"] == pytest.approx(0.4 * math.radians(450.0), rel=0.05)
    for rec in res["plants"] + res["moves"] + res["rotations"]:
        assert rec["t1"] <= 3.5 + 1e-6 or rec["t0"] >= 7.5 - 1e-6, rec               # nothing inside the pivot
    assert [p["id"] for p in res["plants"]] == [0, 1]
    assert np.hypot(res["plants"][0]["origin"][0] - truth["A"][0], res["plants"][0]["origin"][2] - truth["A"][1]) < 0.05
    assert np.hypot(res["plants"][1]["origin"][0] - truth["B"][0], res["plants"][1]["origin"][2] - truth["B"][1]) < 0.05
    pre = [r for r in res["rotations"] if r["t1"] <= 3.5 + 1e-6]
    assert len(pre) == 1 and pre[0]["signed_deg"] == pytest.approx(-60.0, abs=6.0) and pre[0]["plant"] == 0
    assert not any(r["t0"] >= 7.5 - 1e-6 for r in res["rotations"])               # he walks out without turning
    after = [m for m in res["moves"] if m["from_pivot"] == 0]
    assert len(after) == 1 and after[0]["to_plant"] == 1 and after[0]["kind"] == "travel"
    assert after[0]["t0"] >= 7.5 - 1e-6 and after[0]["path"]["pos"][0] == la["path"]["pos"][-1]
    assert all(m["from_pivot"] in (None, 0) and m["to_pivot"] in (None, 0) for m in res["moves"])
    path_ends_at_plants(res)
    # plants + moves + the pivot tile the take
    tiles = sorted([(r["t0"], r["t1"]) for r in res["plants"] + res["moves"]] + [(la["t0"], la["t1"])])
    assert tiles[0][0] == pytest.approx(0.0) and tiles[-1][1] == pytest.approx(float(npz["t"][-1]), abs=1e-3)
    assert all(b[0] - a[1] < 1.01 / FPS and b[0] >= a[1] - 1.01 / FPS for a, b in zip(tiles[:-1], tiles[1:]))


def test_turn_straddling_a_plant_edge_stays_whole():
    """A -60 deg turn that starts in the plant's slow tail and ends in the next plant's slow head is ONE rotation (the
    pieces inside the plants are below 25 deg) and the pieces add up to the swing."""
    t = np.arange(0, 9, 1 / FPS)
    x = 1.5 * ease((t - 3.0) / 3.0)                                      # slow eased move: long plant tails
    yaw = -60.0 * ease((t - 2.6) / 3.8)
    res = fc.analyse(make_npz(t, np.stack([x, np.zeros(len(t))], 1), yaw), BEAT)
    R = res["rotations"]
    assert len(R) == 1 and R[0]["signed_deg"] == pytest.approx(-60.0, abs=4.0) and R[0]["context"] == "move", R
    assert R[0]["swing"]["deg"] == pytest.approx(R[0]["signed_deg"], abs=0.5)


def test_sway_in_place_is_one_plant():
    """A basic in place (+-12 cm weight changes, no net drift) for 6 s: one plant over (nearly) all of it."""
    t = np.arange(0, 6, 1 / FPS)
    xz = np.stack([0.12 * np.sin(2 * math.pi * t / (2 * BEAT)), 0.03 * np.sin(2 * math.pi * t / BEAT)], 1)
    res = fc.analyse(make_npz(t, xz, np.full(len(t), 10.0)), BEAT)
    assert len(res["plants"]) == 1 and res["plants"][0]["duration_s"] >= 5.5
    assert res["rotations"] == []


@pytest.mark.parametrize("speed", [0.3, 0.5])
def test_slow_travel_is_not_a_plant(speed):
    """Travelling at 0.3-0.5 m/s stays within 0.2 m of its centroid for a beat but drifts: no plant."""
    t = np.arange(0, 6, 1 / FPS)
    xz = np.stack([speed * t, np.zeros(len(t))], 1)
    res = fc.analyse(make_npz(t, xz, np.zeros(len(t))), BEAT)
    assert res["plants"] == []
    assert len(res["moves"]) == 1 and res["moves"][0]["from_plant"] is None and res["moves"][0]["to_plant"] is None
    assert res["moves"][0]["length_m"] == pytest.approx(speed * t[-1], rel=0.03)


def test_short_stop_is_not_a_plant():
    """Travelling at 0.6 m/s (a travelling step per beat) with a stop of 0.4 beat: no plant (>= 1 beat held)."""
    t = np.arange(0, 5, 1 / FPS)
    stop = 0.4 * BEAT
    x = 0.6 * (np.minimum(t, 1.6) + np.maximum(t - 1.6 - stop, 0.0))
    res = fc.analyse(make_npz(t, np.stack([x, np.zeros(len(t))], 1), np.zeros(len(t))), BEAT)
    assert res["plants"] == [], res["plants"]


def test_gap_in_time_grid_splits_segments():
    npz, _ = scenario()
    keep = (npz["t"] < 5.0) | (npz["t"] > 6.0)
    cut = {k: (v[keep] if isinstance(v, np.ndarray) and v.ndim and len(v) == len(keep) else v) for k, v in npz.items()}
    res = fc.analyse(cut, BEAT)
    assert len(res["info"]["coverage"]) == 2
    for m in res["moves"]:
        assert not (m["t0"] < 5.0 < m["t1"]) and not (m["t0"] < 6.0 < m["t1"])     # nothing crosses the gap
    ids = [p["id"] for p in res["plants"]]
    assert ids == list(range(len(ids)))


# ----------------------------------------------------------------------------------------------- contract + Unity

def test_json_contract():
    npz, _ = scenario()
    res = fc.analyse(npz, BEAT, None, fake_cbdoc(9.2, 10.3))
    doc = fc.to_json(res, "synthetic_fc", fc.FCConfig(), {"lead_probe": cb.lead_probe(npz),
                                                          "t_range": [0.0, float(npz["t"][-1])]})
    doc = json.loads(json.dumps(doc))                    # strict JSON (no NaN / numpy types)
    assert doc["version"] == fc.VERSION == 3 and doc["frame"] == "world" and doc["time_base"] == "reference"
    assert set(doc) >= {"t_axes", "transitions", "axis_pivots", "dials", "balance"}
    assert set(doc["params"]) >= {"t_axis", "dial", "balance"} and doc["params"]["t_axis"]["pivot_deg"] == 30.0
    for ta in doc["t_axes"]:
        assert set(ta) >= {"id", "t0", "t1", "t_formed", "origin", "yaw_deg", "yaw_unwrapped_deg", "forward",
                           "parallel", "start_reason", "end_reason", "same_place_as", "prev", "next", "pivot_in",
                           "pivot_out", "dial", "occupancy"} and not any(k.startswith("_") for k in ta)
        assert ta["t0"] <= ta["t_formed"] <= ta["t1"]
        assert abs(np.dot(ta["forward"], ta["parallel"])) < 1e-3 and ta["origin"][1] == 0.0
    for x in doc["transitions"]:
        assert set(x) >= {"from_t", "to_t", "t0", "t1", "from", "to", "distance_m", "same_place", "reason",
                          "axis_pivot", "dial", "counterbalance_pivot"}
    b = doc["balance"]
    assert len(b["t"]) == len(npz["t"]) and set(b["lead"]) >= {"stance", "double", "support", "chest", "com",
                                                                "com_offset_m", "tilt_deg", "score"}
    assert set(doc) >= {"plants", "moves", "rotations", "pivots", "counterbalances", "events", "track", "axis",
                        "params", "notes", "coverage"}
    for p in doc["plants"]:
        assert set(p) >= {"id", "t0", "t1", "origin", "yaw_deg", "yaw_mean_deg", "yaw_source", "held", "forward",
                          "long_axis", "radius_m", "turn_deg"}
        assert abs(np.hypot(p["forward"][0], p["forward"][2]) - 1) < 1e-3
        assert abs(np.dot(p["forward"], p["long_axis"])) < 1e-3
        assert abs(fc.wrap180(math.degrees(math.atan2(-p["forward"][2], p["forward"][0])) - p["yaw_deg"])) < 0.1
    for m in doc["moves"]:
        assert set(m) >= {"id", "kind", "from_plant", "to_plant", "from_pivot", "to_pivot", "from", "to", "path",
                          "length_m", "origin_path_m", "displacement_m", "yaw_change_deg"}
        assert m["kind"] in ("travel", "shift") and "pivot" not in m
    for r in doc["rotations"]:
        assert set(r) >= {"id", "signed_deg", "context", "plant", "plants", "move", "centre", "swing"}
        assert "pivot" not in r
    assert set(doc["pivots"][0]["leader_axis"]) >= {"t0", "t1", "path", "length_m", "turn_deg", "from_yaw_deg"}
    tr = doc["track"]
    assert len(tr["t"]) == len(tr["origin"]) == len(tr["yaw_deg"]) == len(tr["speed_m_s"])
    assert doc["axis"]["long_m"] == 2.0 and doc["axis"]["short_m"] == 0.8


def test_unity_doc_conversion():
    npz, _ = scenario()
    res = fc.analyse(npz, BEAT, None, fake_cbdoc(9.2, 10.3))
    doc = json.loads(json.dumps(fc.to_json(res, "synthetic_fc", fc.FCConfig(), {})))
    u = fc.unity_doc(doc, 3.25, "99_Test", "not checked")
    assert u["time_base"] == "capture" and u["frame"] == "unity" and u["time_to_audio"] == 3.25
    p0, q0 = doc["plants"][1], u["plants"][1]
    assert q0["t0"] == pytest.approx(p0["t0"] + 3.25) and q0["t0_reference"] == p0["t0"]
    assert q0["origin"] == [p0["origin"][0], 0.0, -p0["origin"][2]]
    assert q0["forward"] == [p0["forward"][0], 0.0, -p0["forward"][2]]
    # Euler Y: Quaternion.Euler(0, y, 0) * Vector3.forward = (sin y, 0, cos y) = the Unity forward vector
    y = math.radians(q0["euler_y_deg"])
    assert [math.sin(y), math.cos(y)] == pytest.approx([q0["forward"][0], q0["forward"][2]], abs=2e-3)
    r, ru = doc["rotations"][0], u["rotations"][0]
    assert ru["signed_deg_unity_euler_y"] == -r["signed_deg"] and ru["direction"] == r["direction"]
    assert ru["centre"][2] == -r["centre"][2]
    m, mu = doc["moves"][0], u["moves"][0]
    assert mu["path"]["t"][0] == pytest.approx(m["path"]["t"][0] + 3.25)
    assert mu["path"]["pos"][-1][2] == -m["path"]["pos"][-1][2]
    pv, pu = doc["pivots"][0], u["pivots"][0]
    assert pu["pivot"] == [pv["pivot"][0], 0.0, -pv["pivot"][2]] and pu["swept_signed_deg_unity_up"] == 360.0
    assert pu["beat_ticks"][0]["t"] == pytest.approx(pv["beat_ticks"][0]["t"] + 3.25)
    la, lu = pv["leader_axis"], pu["leader_axis"]
    assert lu["t0"] == pytest.approx(la["t0"] + 3.25) and lu["t0_reference"] == la["t0"]
    assert lu["path"]["pos"][0] == [la["path"]["pos"][0][0], 0.0, -la["path"]["pos"][0][2]]
    assert lu["path"]["t"][-1] == pytest.approx(la["path"]["t"][-1] + 3.25)
    assert lu["turn_deg"] == la["turn_deg"] and lu["turn_deg_unity_euler_y"] == -la["turn_deg"]
    assert q0["euler_y_mean_deg"] == fc.unity_euler_y(p0["yaw_mean_deg"])
    assert q0["held"][0] == pytest.approx(p0["held"][0] + 3.25) and q0["held_reference"] == p0["held"]
    assert u["events"][0]["t0"] == pytest.approx(doc["events"][0]["t0"] + 3.25)
    assert u["track"]["euler_y_deg"][0] == pytest.approx(90.0 - doc["track"]["yaw_deg"][0])


def test_export_uses_capture_clock(tmp_path):
    npz, _ = scenario()
    res = fc.analyse(npz, BEAT)
    doc = fc.to_json(res, "synthetic_fc", fc.FCConfig(), {"t_range": [0.0, float(npz["t"][-1])]})
    src = tmp_path / "floorcraft.json"
    src.write_text(json.dumps(doc))
    cap = tmp_path / "stream" / "99_Test"
    cap.mkdir(parents=True)
    times = npz["t"][::3].tolist()
    (cap / "times.json").write_text(json.dumps(times))
    (cap / "capture.json").write_text(json.dumps({"version": 3, "times": "times.json", "time_to_audio": 2.5}))
    rep = fc.export("synthetic_fc", "99_Test", streaming=tmp_path / "stream", src=src, mirror=False)
    out = json.loads((cap / "floorcraft.json").read_text())
    assert rep["time_to_audio"] == 2.5 and out["time_to_audio"] == 2.5
    assert out["plants"][0]["t0"] == pytest.approx(doc["plants"][0]["t0"] + 2.5)
    assert json.loads((cap / "capture.json").read_text())["floorcraft"] == "floorcraft.json"
    assert out["capture_range_reference"] == [round(times[0], 4), round(times[-1], 4)]
    # a capture of another time window is refused
    (cap / "times.json").write_text(json.dumps([100.0, 101.0]))
    with pytest.raises(ValueError):
        fc.export("synthetic_fc", "99_Test", streaming=tmp_path / "stream", src=src, mirror=False)


# ----------------------------------------------------------------------------------------------- T axis (version 3)

def walk(dur: float, xz_of_t, yaw_of_t, seed: int = 0) -> dict:
    """Lead npz over 0..dur s with the chest at xz_of_t(t) (T, 2) facing yaw_of_t(t) (deg), + fit jitter."""
    t = np.arange(0, dur, 1 / FPS)
    rng = np.random.default_rng(seed)
    xz = np.asarray(xz_of_t(t), float) + rng.normal(0, 0.005, (len(t), 2))
    return make_npz(t, xz, np.asarray(yaw_of_t(t), float) + rng.normal(0, 0.5, len(t)))


def taxes(npz: dict, cbdoc: dict | None = None, **kw) -> dict:
    return fc.analyse(npz, BEAT, fc.FCConfig(), cbdoc, **kw)


F30, P30 = fc.t_frame(30.0)                    # forward / parallel (his left) of a T facing 30 deg


def test_t_region_is_a_plus_of_two_corridors():
    cfg = fc.FCConfig()
    O = np.zeros(2)
    inside = lambda a, b: bool(fc.t_inside(a * F30 + b * P30, O, 30.0, cfg))   # noqa: E731
    assert inside(1.1, 0.0) and inside(-1.1, 0.2) and inside(0.0, 1.1) and inside(0.25, -1.1)    # along the axes
    assert not inside(0.5, 0.5) and not inside(1.3, 0.0) and not inside(0.0, -1.3) and not inside(0.4, 0.35)


@pytest.mark.parametrize("axis", ["forward", "side"])
def test_walking_back_and_forth_or_side_to_side_keeps_the_t(axis):
    d = F30 if axis == "forward" else P30
    amp = 0.5 if axis == "forward" else 0.7
    res = taxes(walk(12.0, lambda t: amp * np.sin(2 * math.pi * t / (4 * BEAT))[:, None] * d, lambda t: 30 + 0 * t))
    T = res["t_axes"]
    assert len(T) == 1 and not res["axis_pivots"] and not res["transitions"]
    assert T[0]["t0"] <= BEAT and T[0]["t1"] == pytest.approx(12.0 - 1 / FPS, abs=0.05)
    assert T[0]["yaw_deg"] == pytest.approx(30.0, abs=2.0) and T[0]["end_reason"] == "end"
    occ = T[0]["occupancy"]["forward_m" if axis == "forward" else "side_m"]
    assert occ[1] - occ[0] > 0.9 * 2 * amp - 0.25                                  # he used the corridor


def test_a_single_step_out_and_back_keeps_the_t():
    # hold at the origin, one diagonal step out (0.45 m, outside both corridors) for half a beat and back
    out = (F30 + P30) / math.sqrt(2) * 0.45
    bump = lambda t: np.clip(1 - np.abs(t - 5.0) / (0.5 * BEAT), 0, 1)                # noqa: E731
    res = taxes(walk(10.0, lambda t: bump(t)[:, None] * out, lambda t: 30 + 0 * t))
    assert len(res["t_axes"]) == 1


def test_leaving_the_corridor_makes_a_new_t():
    B = (F30 + P30) / math.sqrt(2) * 1.0                                               # 1 m diagonally: outside
    res = taxes(walk(10.0, lambda t: ease((t - 4.0) / 1.5)[:, None] * B, lambda t: 30 + 0 * t))
    T = res["t_axes"]
    assert len(T) == 2 and not res["axis_pivots"]                                     # no reorientation
    assert T[0]["end_reason"] == "exit" and T[1]["start_reason"] == "exit" and T[1]["same_place_as"] is None
    assert np.allclose(np.array(T[1]["origin"])[[0, 2]], B, atol=0.12)
    assert np.allclose(np.array(T[0]["origin"])[[0, 2]], 0, atol=0.05)
    assert 4.0 <= T[0]["t1"] <= 5.0 and T[1]["t_formed"] <= 6.5
    x = res["transitions"]
    assert len(x) == 1 and x[0]["distance_m"] == pytest.approx(1.0, abs=0.12) and not x[0]["same_place"]
    assert x[0]["from"] == T[0]["origin"] and x[0]["to"] == T[1]["origin"] and x[0]["axis_pivot"] is None


@pytest.mark.parametrize("turn,pivot", [(45.0, True), (-45.0, True), (20.0, False)])
def test_turn_beyond_30_deg_is_a_pivot_and_rotates_the_t_in_place(turn, pivot):
    res = taxes(walk(10.0, lambda t: np.zeros((len(t), 2)), lambda t: 30 + turn * ease((t - 5.0) / 0.5)))
    T, AP = res["t_axes"], res["axis_pivots"]
    if not pivot:
        assert len(T) == 1 and not AP                                               # sway-size turns keep the T
        return
    assert len(T) == 2 and len(AP) == 1
    ap = AP[0]
    assert ap["signed_deg"] == pytest.approx(turn, abs=5.0) and ap["direction"] == ("ccw" if turn > 0 else "cw")
    assert ap["stayed"] and T[1]["same_place_as"] == 0 and T[1]["origin"] == T[0]["origin"]
    assert ap["at"] == T[0]["origin"] and ap["from_t"] == 0 and ap["to_t"] == 1
    assert T[0]["end_reason"] == "rotation" and T[0]["pivot_out"] == 0 and T[1]["pivot_in"] == 0
    assert 4.7 <= ap["t0"] <= 5.3 and 5.0 <= ap["t1"] <= 5.8
    assert res["transitions"][0]["same_place"] and res["transitions"][0]["distance_m"] == 0.0
    assert T[1]["yaw_deg"] == pytest.approx(fc.wrap180(30 + turn), abs=3.0)
    assert {"lead", "follow"} >= set(ap["evaluation"]["per_dancer"]) and ap["evaluation"]["subject"] == "lead"


def test_turn_and_leave_is_a_pivot_and_a_new_place():
    B = F30 * 0.0 + P30 * 0.0 + np.array([1.0, -0.6])
    res = taxes(walk(10.0, lambda t: ease((t - 4.0) / 1.2)[:, None] * B, lambda t: 30 + 90 * ease((t - 4.0) / 1.2)))
    T, AP = res["t_axes"], res["axis_pivots"]
    assert len(T) == 2 and len(AP) == 1 and not AP[0]["stayed"] and T[1]["same_place_as"] is None
    assert AP[0]["signed_deg"] == pytest.approx(90.0, abs=6.0)
    assert np.linalg.norm(np.array(AP[0]["at"])[[0, 2]] - 0.5 * B) < 0.35          # mid-turn, on his path


def test_spin_in_place_of_720_deg_is_one_dial():
    res = taxes(walk(10.0, lambda t: np.zeros((len(t), 2)), lambda t: 30 - 720 * ease((t - 3.5) / 3.0)))
    D = res["dials"]
    assert len(D) == 1 and D[0]["kind"] == "leader_spin" and D[0]["subject"] == "lead"
    assert D[0]["degrees"] == pytest.approx(-720.0, abs=15.0) and D[0]["direction"] == "cw"
    assert D[0]["turns"] == pytest.approx(2.0, abs=0.05) and np.allclose(D[0]["centre"], [0, 0, 0], atol=0.05)
    assert all(ap["dial"] == 0 for ap in res["axis_pivots"]) and D[0]["axis_pivots"]
    assert D[0]["t_axis"] == 0 and D[0]["evaluation"]["subject"] == "lead"


def follow_npz(t: np.ndarray, P: np.ndarray) -> dict:
    """A follower standing on her LEFT foot at the floor point P [x, z] (ankle behind, ball 0.13 m ahead along +x;
    right foot lifted 0.3 m), chest 1.25 m above P."""
    T = len(t)
    J = np.zeros((T, 55, 3))
    J[:] = [P[0], 1.25, P[1]]
    J[:, 7] = [P[0] - 0.06, 0.08, P[1]]
    J[:, 10] = [P[0] + 0.07, 0.02, P[1]]
    J[:, 8] = [P[0] - 0.06, 0.38, P[1] + 0.2]
    J[:, 11] = [P[0] + 0.07, 0.32, P[1] + 0.2]
    return {"t": t, "joints": J}


@pytest.mark.parametrize("lean_m,verdict", [(0.0, "good"), (0.2, "needs work")])
def test_counterbalance_of_720_deg_is_a_dial_with_its_evaluation(lean_m, verdict):
    P = np.array([1.0, 0.0])
    circ = lambda t: P + 0.5 * np.stack([np.cos(np.radians(90 - 720 * ease((t - 3.0) / 4.0))),    # noqa: E731
                                         -np.sin(np.radians(90 - 720 * ease((t - 3.0) / 4.0)))], -1)
    npz = walk(10.0, circ, lambda t: -720 * ease((t - 3.0) / 4.0))
    t = npz["t"]
    lead_com = np.stack([npz["joints"][:, 9, 0], np.full(len(t), 1.0), npz["joints"][:, 9, 2]], 1)
    m = {"lead": 80.0, "follow": 55.0}
    target = np.array([P[0], 0.95, P[1] + lean_m])                       # the combined COM (lean_m beside the foot)
    fol_com = ((m["lead"] + m["follow"]) * target - m["lead"] * lead_com) / m["follow"]
    cbdoc = fake_cbdoc(3.0, 7.0, point=(P[0], 0.0, P[1]))
    cbdoc["intervals"][0]["pivots"][0].update({"swept_deg": 720.0, "swept_signed_deg": -720.0, "leader_radius_m": 0.5})
    res = taxes(npz, cbdoc, follow=follow_npz(t, P), coms={"lead": lead_com, "follow": fol_com}, masses=m)
    D = [d for d in res["dials"] if d["kind"] == "counterbalance"]
    assert len(D) == 1 and D[0]["degrees"] == -720.0 and D[0]["abs_deg"] == 720.0 and D[0]["turns"] == 2.0
    assert D[0]["centre"] == [1.0, 0.0, 0.0] and D[0]["counterbalance_pivot"] == 0 and D[0]["subject"] == "couple"
    ev = D[0]["evaluation"]
    assert ev["verdict"] == verdict and ev["subject"] == "couple" and ev["frames"] > 100
    assert ev["mean_offset_m"] == pytest.approx(lean_m, abs=0.03)
    assert res["pivots"][0]["evaluation"] == ev and res["pivots"][0]["dial"] == D[0]["id"]
    assert not res["axis_pivots"] or all(ap["t1"] <= 3.0 + 1e-6 or ap["t0"] >= 7.0 - 1e-6
                                         for ap in res["axis_pivots"])   # the dial owns the circling
    assert all(ta["t1"] <= 3.05 or ta["t0"] >= 6.95 for ta in res["t_axes"])      # no T inside the pivot
    cpl = res["balance"]["couple"]
    k = [i for i, pv in enumerate(cpl["pivot"]) if pv == 0]
    assert k and cpl["pivot"][0] is None and all(cpl["com"][i] is not None for i in k)
    assert np.nanmedian([cpl["com_offset_m"][i] for i in k]) == pytest.approx(lean_m, abs=0.03)


def stance_J(T: int, chest_x: float = 0.0) -> np.ndarray:
    """Joints with the LEFT foot planted (ankle (0, 0.08, 0), ball (0.13, 0.02, 0)), the right foot lifted 0.3 m
    beside it, the chest (spine3) at (chest_x, 1.3, 0)."""
    J = np.zeros((T, 55, 3))
    J[:, 7], J[:, 10] = [0.0, 0.08, 0.0], [0.13, 0.02, 0.0]
    J[:, 8], J[:, 11] = [0.0, 0.38, 0.25], [0.13, 0.32, 0.25]
    J[:, fc.CHEST] = [chest_x, 1.3, 0.0]
    return J


@pytest.mark.parametrize("com_z,chest_z,score,verdict", [(0.0, 0.0, 1.0, "good"), (0.15, 0.25, 0.0, "needs work"),
                                                           (0.08, 0.08, 0.5, None)])
def test_balance_axis_perfect_vs_leaning(com_z, chest_z, score, verdict):
    """COM over the planted foot (anywhere along its heel-toe line) = 1; leaning sideways 15 cm off it with the chest
    25 cm off = 0; 8 cm off (the middle of the 4..12 cm ramp; tilt 3.5 deg) = 0.5."""
    cfg = fc.FCConfig()
    t = np.arange(0, 2.0, 1 / FPS)
    J = stance_J(len(t))
    J[:, fc.CHEST] = [0.1, 1.3, chest_z]
    com = np.tile([0.1, 1.0, com_z], (len(t), 1))
    b = fc.balance_axis(t, J, com, cfg)
    assert (b["stance"] == 1).all() and not b["double"].any()                     # the planted left foot
    assert np.allclose(b["support"], [0.1, 0.0], atol=1e-6)                        # nearest point of the sole line
    assert np.allclose(b["com_offset_m"], com_z, atol=1e-6)
    assert np.allclose(b["score"], score, atol=0.02)
    assert np.allclose(b["tilt_deg"], math.degrees(math.atan2(chest_z, 1.3)), atol=0.05)
    com[:, 0] = 0.15                                                                # over the toes: still on the foot
    assert np.allclose(fc.balance_axis(t, J, com, cfg)["com_offset_m"], com_z, atol=1e-6)
    if verdict:
        ev = fc.evaluate_turn({"lead": b}, "lead", 0, len(t) - 1, cfg)
        assert ev["verdict"] == verdict and ev["aligned_share"] == (1.0 if verdict == "good" else 0.0)


def test_balance_axis_double_support_and_airborne():
    cfg = fc.FCConfig()
    t = np.arange(0, 2.0, 1 / FPS)
    J = stance_J(len(t))
    J[:, 8, 1], J[:, 11, 1] = 0.08, 0.02                       # both feet down, 0.25 m apart (z)
    com = np.tile([0.06, 1.0, 0.18], (len(t), 1))            # nearer the right foot
    b = fc.balance_axis(t, J, com, cfg)
    assert b["double"].all() and (b["stance"] == 2).all() and np.allclose(b["com_offset_m"], 0.07, atol=1e-6)
    J[30:40, 7:12, 1] += 0.4                                 # a jump: both feet 0.4 m up for 10 frames
    b = fc.balance_axis(t, J, com, cfg)
    k = slice(32, 38)
    assert (b["stance"][k] == 0).all() and (b["score"][k] == 0).all() and np.isnan(b["support"][k]).all()
    assert (b["stance"][:28] == 2).all() and (b["stance"][42:] == 2).all()


def test_unity_copy_of_the_version_3_records():
    res = taxes(walk(10.0, lambda t: np.zeros((len(t), 2)), lambda t: 30 + 45 * ease((t - 5.0) / 0.5)))
    doc = json.loads(fc.dumps(fc.to_json(res, "synthetic_fc", fc.FCConfig(), {})))
    u = fc.unity_doc(doc, 3.25, "99_Test", "not checked")
    ta, tu = doc["t_axes"][1], u["t_axes"][1]
    assert tu["t_formed"] == pytest.approx(ta["t_formed"] + 3.25) and tu["t_formed_reference"] == ta["t_formed"]
    assert tu["origin"] == [ta["origin"][0], 0.0, -ta["origin"][2]]
    y = math.radians(tu["euler_y_deg"])
    assert [math.sin(y), math.cos(y)] == pytest.approx([tu["forward"][0], tu["forward"][2]], abs=2e-3)
    ap, au = doc["axis_pivots"][0], u["axis_pivots"][0]
    assert au["signed_deg_unity_euler_y"] == -ap["signed_deg"] and au["at"][2] == -ap["at"][2]
    assert au["evaluation"]["window"][0] == pytest.approx(ap["evaluation"]["window"][0] + 3.25)
    x, xu = doc["transitions"][0], u["transitions"][0]
    assert xu["t0"] == pytest.approx(x["t0"] + 3.25) and xu["to"] == [x["to"][0], 0.0, -x["to"][2]]
    b, bu = doc["balance"], u["balance"]
    assert bu["t"][0] == pytest.approx(b["t"][0] + 3.25) and bu["t_reference"] == b["t"]
    assert bu["lead"]["chest"][0] == [b["lead"]["chest"][0][0], b["lead"]["chest"][0][1], -b["lead"]["chest"][0][2]]
    assert all((s is None and q is None) or q == [s[0], 0.0, -s[2]]
               for s, q in zip(b["lead"]["support"], bu["lead"]["support"]))


def test_dumps_is_json_with_flat_arrays():
    doc = {"a": [1, 2, 3], "b": [[0.0, 1.0], [2.0, None]], "c": [{"x": 1}], "d": {"e": "f"}}
    txt = fc.dumps(doc)
    assert json.loads(txt) == doc and '[1,2,3]' in txt and '[[0.0,1.0],[2.0,null]]' in txt


# ----------------------------------------------------------------------------------------------- review fixes (v3.1)

def moving_follow_npz(t: np.ndarray, P: np.ndarray) -> dict:
    """A follower whose chest moves along P (T, 2) [x, z], 1.25 m up, feet lifted (no stance needed here)."""
    J = np.zeros((len(t), 55, 3))
    J[:, :, 0], J[:, :, 1], J[:, :, 2] = P[:, None, 0], 1.25, P[:, None, 1]
    return {"t": t, "joints": J}


@pytest.mark.parametrize("orbit_sign,dial", [(+1.0, False), (-1.0, True)])
def test_couple_turn_dial_only_when_he_goes_round_the_centre(orbit_sign, dial):
    """The couple swaps places about their combined COM (it stays put) while the lead spins -300 deg (cw). When the
    swap goes the OTHER way (ccw, +150 deg: the review's HUD 16.1-17.3 s case) his turn is his own spin under the
    arms, not a couple turn about that point: no dial (his chest moves too much for a spin dial) - the turn stays an
    axis pivot. When both go round it the same way (cw -300 deg) it is a couple-turn dial."""
    C = np.array([0.5, 0.3])
    t = np.arange(0, 10.0, 1 / FPS)
    k = ease((t - 4.0) / 1.5)
    phase = 180.0 + orbit_sign * (150.0 if orbit_sign > 0 else 300.0) * k
    lead = C + 0.45 * fc.fwd_of(phase)
    fol = C + 0.45 * fc.fwd_of(phase + 180.0)
    npz = make_npz(t, lead, 30.0 - 300.0 * k)
    com_l = np.stack([lead[:, 0], np.full(len(t), 1.0), lead[:, 1]], 1)
    com_f = np.stack([fol[:, 0], np.full(len(t), 0.95), fol[:, 1]], 1)
    res = taxes(npz, follow=moving_follow_npz(t, fol), coms={"lead": com_l, "follow": com_f},
                masses={"lead": 70.0, "follow": 70.0})
    AP = res["axis_pivots"]
    assert sum(abs(ap["signed_deg"]) for ap in AP) == pytest.approx(300.0, abs=20.0)
    turn_dials = [d for d in res["dials"] if d["kind"] != "counterbalance"]
    if not dial:
        assert not turn_dials and all(ap["dial"] is None for ap in AP)
        log = res["info"]["turn_groups"]
        assert log and "does not go round it" in log[0]["rejected"]["couple_turn"]
    else:
        assert len(turn_dials) == 1 and turn_dials[0]["kind"] == "couple_turn"
        assert -320.0 < turn_dials[0]["orbit_deg"] < -240.0         # over the pivot's extent (edges trimmed)


def test_t_origin_is_averaged_over_where_he_dances_on_it():
    """He forms the T at the origin (1 beat still), then dances 0.5 m up the stem for 6 s: the T sits at the median
    of where he danced on it (not at the forming window), its heading the mean heading."""
    def xz(t):
        fwd = 0.5 * ease((t - 1.5) / 0.6) + 0.1 * np.sin(2 * math.pi * t / (2 * BEAT)) * (t > 2.2)
        return fwd[:, None] * F30
    res = taxes(walk(9.0, xz, lambda t: 30 + 0 * t))
    T = res["t_axes"]
    assert len(T) == 1
    o, fo = np.array(T[0]["origin"])[[0, 2]], np.array(T[0]["formed_origin"])[[0, 2]]
    assert float(o @ F30) == pytest.approx(0.5, abs=0.1) and float(fo @ F30) < 0.2
    assert T[0]["occupancy"]["centre_offset_m"] < 0.08 and T[0]["yaw_deg"] == pytest.approx(30.0, abs=2.0)


def test_a_t_that_would_only_flash_is_dropped_and_its_heading_is_a_waypoint():
    """30 deg for 3 s, then +60 deg held only ~1.2 beats (a T would form and hand over at once), then +90 deg more,
    held: two Ts (the brief one dropped), and both reorientations are still axis pivots (via the brief heading)."""
    b1 = 3.0 + 0.3 + 1.2 * BEAT
    res = taxes(walk(10.0, lambda t: np.zeros((len(t), 2)),
                     lambda t: 30 + 60 * ease((t - 3.0) / 0.3) + 90 * ease((t - b1) / 0.3)))
    T, AP = res["t_axes"], res["axis_pivots"]
    assert len(T) == 2 and T[1]["dropped_before"] == 1 and res["info"]["counts"]["dropped_short_ts"] == 1
    assert T[0]["yaw_deg"] == pytest.approx(30, abs=3) and T[1]["yaw_deg"] == pytest.approx(180 - 360, abs=3) or \
        T[1]["yaw_deg"] == pytest.approx(180, abs=3)
    assert [round(ap["signed_deg"] / 10) * 10 for ap in AP] == [60, 90]
    assert AP[0]["from_t"] == 0 and AP[0]["to_kind"] == "brief_t" and AP[1]["to_t"] == 1 and AP[1]["via_brief"]
    assert T[0]["pivot_out"] == AP[0]["id"] and T[1]["pivot_in"] == AP[1]["id"]
    for ta in T[:-1]:
        assert ta["t1"] - ta["t_formed"] >= BEAT - 1e-3


def test_turning_into_and_out_of_a_counterbalance_and_before_the_first_t_is_recorded():
    """-120 deg before the first T (lead-in), -90 deg on the way into a counterbalance pivot, the pivot's -720 deg
    (its dial), +60 deg on the way out to the next T: each is an axis pivot (contexts lead_in / into_counterbalance
    / out_of_counterbalance), none across the pivot itself, none folded into a dial."""
    P = np.array([1.0, 0.0])
    ang = lambda t: 90 - 720 * ease((t - 3.0) / 4.0)                                       # noqa: E731
    circ = lambda t: P + 0.5 * np.stack([np.cos(np.radians(ang(t))), -np.sin(np.radians(ang(t)))], -1)  # noqa: E731
    yaw = lambda t: (-120 * ease((t - 0.1) / 0.5) - 90 * ease((t - 1.9) / 0.6) - 720 * ease((t - 3.0) / 4.0)  # noqa
                     + 60 * ease((t - 7.4) / 0.4))
    npz = walk(11.0, circ, yaw)
    cbdoc = fake_cbdoc(3.0, 7.0, point=(P[0], 0.0, P[1]))
    cbdoc["intervals"][0]["pivots"][0].update({"swept_deg": 720.0, "swept_signed_deg": -720.0})
    res = taxes(npz, cbdoc)
    AP = res["axis_pivots"]
    ctx = {ap["context"]: ap for ap in AP}
    assert set(ctx) == {"lead_in", "into_counterbalance", "out_of_counterbalance"}, [a["context"] for a in AP]
    assert ctx["lead_in"]["signed_deg"] == pytest.approx(-120, abs=10) and ctx["lead_in"]["to_t"] == 0
    assert ctx["into_counterbalance"]["signed_deg"] == pytest.approx(-90, abs=10)
    assert ctx["into_counterbalance"]["to_t"] is None and ctx["into_counterbalance"]["counterbalance_pivot"] == 0
    assert ctx["out_of_counterbalance"]["signed_deg"] == pytest.approx(60, abs=10)
    assert all(ap["t1"] <= 3.0 + 1e-6 or ap["t0"] >= 7.0 - 1e-6 for ap in AP) and all(ap["dial"] is None for ap in AP)
    assert [d["kind"] for d in res["dials"]] == ["counterbalance"]


def feet_J(T: int, left: tuple | None, right: tuple | None) -> np.ndarray:
    """Joints with the chest at (0.06, 1.3, 0); each foot planted at its (x, z) heel point (ankle 8 cm up, ball 13 cm
    ahead along +x) or lifted 0.3 m when None."""
    J = np.zeros((T, 55, 3))
    J[:, fc.CHEST] = [0.06, 1.3, 0.0]
    for (ia, ib), pos in (((7, 10), left), ((8, 11), right)):
        x, z, up = (pos[0], pos[1], 0.0) if pos is not None else (0.0, 0.3, 0.3)
        J[:, ia], J[:, ib] = [x, 0.08 + up, z], [x + 0.13, 0.02 + up, z]
    return J


def test_only_a_pivot_on_one_foot_is_graded():
    cfg = fc.FCConfig()
    t = np.arange(0, 1.0, 1 / FPS)
    n = len(t)
    com = np.tile([0.06, 1.0, 0.0], (n, 1))                               # over the left foot's sole line
    one = fc.balance_axis(t, feet_J(n, (0.0, 0.0), None), com, cfg)
    ev = fc.graded_turn({"lead": one}, "lead", 0, n - 1, cfg)
    assert ev["turn_kind"] == "one_foot" and ev["feet"]["pivot_foot"] == "left" and ev["graded"]
    assert ev["verdict"] == "good" and "pivot on the left foot" in ev["text"]
    both = fc.balance_axis(t, feet_J(n, (0.0, 0.0), (0.0, 0.3)), np.tile([0.06, 1.0, 0.15], (n, 1)), cfg)
    ev = fc.graded_turn({"lead": both}, "lead", 0, n - 1, cfg)
    assert ev["turn_kind"] == "both_feet" and not ev["graded"] and ev["verdict"] == "not graded"
    assert ev["balance_verdict"] == "needs work" and ev["text"].startswith("not graded")
    J = feet_J(n, (0.0, 0.0), None)                                       # left planted, then the right 0.5 m away
    J[n // 2:] = feet_J(n - n // 2, None, (0.5, 0.0))
    step = fc.balance_axis(t, J, com, cfg)
    ev = fc.graded_turn({"lead": step}, "lead", 0, n - 1, cfg)
    assert ev["turn_kind"] == "stepped" and not ev["graded"] and ev["feet"]["stance_changes"] >= 1


# ----------------------------------------------------------------------------------------------- real take

REAL = ROOT / "work" / "larissa_kadu_38" / "smplx" / "lead.npz"


@pytest.mark.skipif(not REAL.exists(), reason="larissa_kadu_38 fits not present (local data)")
def test_real_take_record_is_consistent():
    npzs = cb.load_npzs(REAL.parent)
    npz = npzs["lead"]
    beat, _, _ = cb.beats_from_grid(cb.load_grid(ROOT / "work" / "larissa_kadu_38"))
    cbp = ROOT / "work" / "larissa_kadu_38" / "counterbalance" / "counterbalance.json"
    res = fc.analyse(npz, beat, None, json.loads(cbp.read_text()) if cbp.exists() else None, follow=npzs["follow"])
    cfg = fc.FCConfig()
    P = res["plants"]
    assert 5 <= len(P) <= 40                                   # a record to read, not one cross per step
    for a, b in zip(P[:-1], P[1:]):
        assert a["t1"] < b["t0"]                               # plants never overlap
    assert all(p["radius_m"] <= 0.2 + 1e-9 and p["drift_m_s"] <= 0.15 + 1e-9 for p in P)
    # no stand recorded twice: two neighbouring plants whose union passes the plant test are one plant
    ax = fc.smooth_axis(np.asarray(npz["t"], float), *fc.raw_axis(npz)[:2], cfg.cutoff_hz)
    for a, b in zip(P[:-1], P[1:]):
        i0, i1 = np.searchsorted(ax["t"], a["t0"] - 1e-4), np.searchsorted(ax["t"], b["t1"] + 1e-4) - 1
        assert not fc.plant_ok(ax["t"], ax["origin"], int(i0), int(i1), cfg), (a["id"], b["id"])
    # every move is one step or a travel; every path joins its plant crosses; steps between touching plants are cut
    # out of them (the user's take: every junction resolves to a real step window)
    for m in res["moves"]:
        one_step = m["from_plant"] is not None and m["to_plant"] is not None and \
            m["origin_path_m"] < cfg.step_max_m and m["duration_s"] <= cfg.step_max_beats * beat + 1e-6
        assert m["kind"] == ("step" if one_step else "travel"), m["id"]
        assert m["duration_s"] >= cfg.touch_min_s and m["origin_path_m"] >= cfg.touch_path_m, m["id"]
    path_ends_at_plants(res)
    # nothing overlaps a pivot; plants + moves + pivots tile the take
    spans = [(p["leader_axis"]["t0"], p["leader_axis"]["t1"]) for p in res["pivots"] if p["leader_axis"]]
    for rec in P + res["moves"] + res["rotations"]:
        assert all(rec["t1"] <= s0 + 1e-6 or rec["t0"] >= s1 - 1e-6 for s0, s1 in spans), rec["id"]
    tiles = sorted([(p["t0"], p["t1"]) for p in P] + [(m["t0"], m["t1"]) for m in res["moves"]] + spans)
    assert tiles[0][0] == pytest.approx(float(npz["t"][0]), abs=0.12)
    assert all(abs(b[0] - a[1]) < 1e-6 for a, b in zip(tiles[:-1], tiles[1:]))
    assert all(abs(r["signed_deg"]) >= 25.0 for r in res["rotations"])
    # the user's counterbalance pivot (HUD 32.0-36.15): the leader's circling is on the pivot, not a 1000 deg rotation
    if res["pivots"]:
        assert all(abs(r["signed_deg"]) < 500 for r in res["rotations"])
        assert res["pivots"][0]["leader_axis"]["turn_deg"] < -500
    # version 3: Ts never overlap, each T reorientation > 30 deg is an axis pivot, the user's counterbalance (HUD
    # 32.0-36.15) is a ~700 deg dial over her anchored foot with an evaluation, and no T forms inside it
    T = res["t_axes"]
    assert 5 <= len(T) <= 40 and all(a["t1"] <= b["t0"] + 1e-6 for a, b in zip(T[:-1], T[1:]))
    for ap in res["axis_pivots"]:
        assert abs(ap["signed_deg"]) > 30.0
        if ap["from_t"] is not None:
            assert T[ap["from_t"]]["pivot_out"] <= ap["id"]               # the first pivot leaving that T
        if ap["to_t"] is not None:
            assert T[ap["to_t"]]["pivot_in"] >= ap["id"]                  # the last pivot reaching it
        # only a pivot on one foot is graded; turns on both feet / stepped turns are recorded, not graded
        ev = ap["evaluation"]
        assert ev["graded"] == (ev["turn_kind"] == "one_foot" and ev["verdict"] in ("good", "ok", "needs work"))
        assert ev["graded"] or ev["verdict"] in ("not graded", "unknown")
    # no T only flashes: each holds >= t_min_hold_beats after forming (the last one may be cut by the take end)
    for ta in T[:-1]:
        assert ta["t1"] - ta["t_formed"] >= cfg.t_min_hold_beats * beat - 1e-3, ta["id"]
    # the T sits where he danced on it: the median of his chest over a long T is near its origin
    for ta in T:
        if ta["t1"] - ta["t_formed"] > 3 * beat and ta["same_place_as"] is None:
            assert ta["occupancy"]["centre_offset_m"] <= 0.12, ta["id"]
    # a dial about a shared point is only made when he really goes round it (orbit agrees with the turn)
    for d in res["dials"]:
        if d["kind"] in ("couple_turn", "leader_around_follower"):
            assert np.sign(d["orbit_deg"]) == np.sign(d["degrees"])
            assert abs(d["orbit_deg"]) >= cfg.dial_orbit_ratio * abs(d["degrees"])
    if res["pivots"]:
        d = [x for x in res["dials"] if x["kind"] == "counterbalance"]
        assert len(d) == len(res["pivots"]) and abs(d[0]["degrees"]) > 600 and d[0]["evaluation"]["frames"] > 60
        assert d[0]["centre"] == res["pivots"][0]["pivot"] and d[0]["evaluation"]["verdict"] != "unknown"
        assert all(ta["t1"] <= d[0]["t0"] + 0.05 or ta["t0"] >= d[0]["t1"] - 0.05 for ta in T)
