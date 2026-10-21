"""Unit tests for dancecap.hand_contact's pure parts (regions, palm frame, hold sub-type, ramps, hysteresis input; no footage, no GPU).

    .venv-nopo4d\\Scripts\\python.exe -m pytest dancecap/tests/test_hand_contact.py -q      (from the workspace root)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dancecap import hand_contact as hc  # noqa: E402


def test_region_split_front_back():
    j = np.array([13, 13, 9, 9, 9, 6, 6, 3, 18, 21, 30, 15])
    d = np.array([-0.8, 0.5, -0.9, 0.9, 0.0, -0.6, 0.6, -0.7, 0.0, 0.0, 0.0, 0.0])
    names = [hc.REGIONS[i] for i in hc.region_of(j, d)]
    assert names == ["shoulder_blade", "shoulder", "shoulder_blade", "chest", "waist", "back", "chest", "back", "forearm", "hand", "hand",
                     "neck_head"]


def test_ramp_weights_shape_and_ends():
    w = hc.ramp_weights(10, 4)
    assert len(w) == 18
    assert np.all(w[4:14] == 1.0)
    assert 0 < w[0] < w[1] < w[3] < 1 and np.allclose(w[:4], w[-4:][::-1])


def _hand(wrist, index, pinky, middle, side):
    J = np.zeros((1, 55, 3))
    i, m, r, p = hc.mcp_joints(side)
    J[0, 20 if side == "left" else 21] = wrist
    J[0, i], J[0, p], J[0, m], J[0, r] = index, pinky, middle, (np.asarray(middle) + np.asarray(pinky)) / 2
    return J


def test_palm_normal_rest_pose_points_down():
    # SMPL-X rest pose: left hand along +x, index in front (+z), palm down
    Jl = _hand([0.7, 1.4, 0], [0.8, 1.4, 0.03], [0.8, 1.4, -0.03], [0.8, 1.4, 0.0], "left")
    Jr = _hand([-0.7, 1.4, 0], [-0.8, 1.4, 0.03], [-0.8, 1.4, -0.03], [-0.8, 1.4, 0.0], "right")
    assert np.allclose(hc.palm_normal(Jl, "left")[0], [0, -1, 0], atol=1e-6)
    assert np.allclose(hc.palm_normal(Jr, "right")[0], [0, -1, 0], atol=1e-6)


def test_hold_subtype():
    # A's left palm faces down (-y), fingers +x; B's right palm faces up: palms facing
    Ja = _hand([0, 1, 0], [0.1, 1, 0.03], [0.1, 1, -0.03], [0.1, 1, 0], "left")
    # B's right hand: palm up = mirror the rest-pose right hand about y; fingers +x (same way) -> palm to palm
    Jb_same = _hand([0, 0.98, 0], [0.1, 0.98, 0.03], [0.1, 0.98, -0.03], [0.1, 0.98, 0], "right")
    assert (hc.palm_normal(Jb_same, "right")[0] @ hc.palm_normal(Ja, "left")[0]) < -0.9
    assert hc.hold_subtype(Ja, "left", Jb_same, "right") == "palm_to_palm"
    # fingers pointing the other way (-x), palm still up -> handshake / grip
    Jb_opp = _hand([0.1, 0.98, 0], [0.0, 0.98, -0.03], [0.0, 0.98, 0.03], [0.0, 0.98, 0], "right")
    assert hc.palm_normal(Jb_opp, "right")[0][1] > 0.9
    assert hc.hold_subtype(Ja, "left", Jb_opp, "right") == "handshake"
    # B's palm also down: one hand over the back of the other
    Jb_down = _hand([0, 0.98, 0], [0.1, 0.98, 0.03], [0.1, 0.98, -0.03], [0.1, 0.98, 0], "left")
    assert hc.hold_subtype(Ja, "left", Jb_down, "left") == "hand_on_hand"


def test_templates_cover_every_hold():
    for h in list(hc.HOLD_OF_REGION.values()) + ["handshake", "palm_to_palm", "hand_on_hand"]:
        assert h in hc.TEMPLATE_OF_HOLD
        t = hc.TEMPLATES[hc.TEMPLATE_OF_HOLD[h]]
        assert t["lo"] < t["c0"] < t["hi"]


def test_score_series_contradiction_kills_score():
    T = 20
    geo = {("lead", "left"): {"gap": np.full(T, 0.005)}}
    v_for = np.zeros((T, 6), np.int8)
    v_for[:, :3] = 1
    v_against = np.zeros((T, 6), np.int8)
    v_against[:, :4] = -1
    mc = np.full((T, 6), 0.2)
    s1 = hc.score_series(geo, {("lead", "left"): {"verdict": v_for, "mamma_contact": mc}})[("lead", "left")]["score"]
    s2 = hc.score_series(geo, {("lead", "left"): {"verdict": v_against, "mamma_contact": mc}})[("lead", "left")]["score"]
    assert s1.min() > 0.9 and s2.max() == 0.0
    # a 7 cm gap counts only with >= 2 supporting views
    geo_far = {("lead", "left"): {"gap": np.full(T, 0.07)}}
    s3 = hc.score_series(geo_far, {("lead", "left"): {"verdict": v_for, "mamma_contact": mc}})[("lead", "left")]["score"]
    assert 0.5 < s3.min() <= 1.0


def test_keep_runs():
    m = np.array([1, 1, 0, 1, 1, 1, 1, 0, 1], bool)
    assert hc._keep_runs(m, 3).tolist() == [0, 0, 0, 1, 1, 1, 1, 0, 0]


def _ev(i, role, side, i0, i1):
    return {"id": i, "role": role, "side": side, "i0": i0, "i1": i1}


def test_split_parts_keeps_overlapping_same_hand_together():
    todo = [(_ev(0, "lead", "left", 0, 100), None), (_ev(1, "lead", "left", 90, 200), None),      # same hand, overlapping -> together
            (_ev(2, "lead", "right", 0, 100), None), (_ev(3, "follow", "left", 500, 600), None),
            (_ev(4, "lead", "left", 1000, 1100), _ev(5, "follow", "right", 1000, 1100))]
    parts = hc.split_parts(todo, 3, 10, 5000)
    where = {x[0]["id"]: i for i, p in enumerate(parts) for x in p}
    assert where[0] == where[1]
    assert sorted(len(p) for p in parts) == [1, 2, 2] or sum(len(p) for p in parts) == 5
    # no two parts move the same hand in overlapping windows
    for i, p in enumerate(parts):
        for j, q in enumerate(parts):
            if i >= j:
                continue
            for x in p:
                for y in q:
                    hx = {(x[0]["role"], x[0]["side"])} | ({(x[1]["role"], x[1]["side"])} if x[1] else set())
                    hy = {(y[0]["role"], y[0]["side"])} | ({(y[1]["role"], y[1]["side"])} if y[1] else set())
                    if hx & hy:
                        assert x[0]["i1"] + 12 < y[0]["i0"] - 12 or y[0]["i1"] + 12 < x[0]["i0"] - 12


def test_speed_violations_flags_added_jump_not_input_speed():
    t = np.arange(60) / 30.0
    X_in = np.stack([np.sin(2 * np.pi * t) * 0.3, np.zeros(60), np.zeros(60)], 1)      # the input moves fast (up to ~1.9 m/s)
    same = hc.speed_violations(X_in, X_in.copy(), p99=2.0, add_max=0.55, margin=0.25)
    assert same["max_add"] == 0 and len(same["jump"]) == 0 and len(same["fast"]) == 0
    X_new = X_in.copy()
    X_new[30:, 1] += 0.03                                                                # a 3 cm step in one frame = 0.9 m/s added
    sv = hc.speed_violations(X_in, X_new, p99=2.0, add_max=0.55, margin=0.25)
    assert sv["jump"].tolist() == [29] and abs(sv["max_add"] - 0.9) < 1e-6
    # a slow 3 cm drift over a second is no jump
    X_slow = X_in.copy()
    X_slow[:, 1] += np.linspace(0, 0.03, 60)
    assert len(hc.speed_violations(X_in, X_slow, 2.0, 0.55, 0.25)["jump"]) == 0


def test_ramp_for_long_enough_for_the_move():
    assert hc.ramp_for(0.0, 0.35, 12, 45) == 12
    r = hc.ramp_for(0.20, 0.35, 12, 45)                   # 20 cm at <= 0.35 m/s peak: pi / 2 x 0.2 / (r / 30) <= 0.35
    assert np.pi / 2 * 0.20 / (r / 30.0) <= 0.35 + 1e-9 and r <= 45
    assert hc.ramp_for(2.0, 0.35, 12, 45) == 45


def test_curated_events_merge_and_veto():
    evs = [{"id": 0, "i0": 10, "i1": 40, "t0": 0.33, "t1": 1.33, "hold": "hand_on_chest", "region": "chest", "duration_s": 1.0},
           {"id": 1, "i0": 46, "i1": 90, "t0": 1.53, "t1": 3.0, "hold": "hand_on_chest", "region": "chest", "duration_s": 1.5},
           {"id": 2, "i0": 200, "i1": 260, "t0": 6.67, "t1": 8.67, "hold": "hand_on_hand", "region": "hand", "duration_s": 2.0}]
    cur = {"veto": {"2": "worse in the video"}, "keep": {},
           "overrides": {"0": {"merge": [1], "hold": "hand_on_shoulder_blade", "region": "shoulder_blade",
                               "anchor": {"joint": 9, "local_xy": [0.07, 0.05], "side": "back"}, "why": "on her back"}}}
    out = hc.curated_events(evs, cur)
    assert [e["id"] for e in out] == [0, 2]
    e0 = out[0]
    assert (e0["i0"], e0["i1"], e0["merged"], e0["hold"]) == (10, 90, [1], "hand_on_shoulder_blade") and e0["anchor"]["side"] == "back"
    assert evs[0]["i1"] == 40                                                            # the input list is not modified
    ok, why = hc.refinable(dict(evs[2], evidence_median=1.0), {}, cur["veto"])
    assert not ok and why.startswith("curated veto")
