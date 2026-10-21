"""v3 of the class-recap direction (dancecap.film_recap_flow): the back-to-the-axis finder, the pivot target, the duration rule and its easing,
the side-on azimuth, where a transition sits, carving a pivot window into the segments, the schedule's holds. Pure functions, no capture data."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from dancecap import film_recap_flow as fl  # noqa: E402


def utt(uid, text, t0=10.0, dt=0.3, speaker="Kadu"):
    ws = []
    t = t0
    for w in text.split():
        ws.append({"text": w, "start": round(t, 3), "end": round(t + dt * 0.8, 3)})
        t += dt
    return {"id": uid, "start": t0, "end": round(t, 3), "speaker": speaker, "text": text, "words": ws}


def test_back_to_axis_finds_the_phrase_and_its_close_variants():
    us = [utt("u1", "bring her back to the axis, rotate lateral."), utt("u2", "go back to axis now", 30.0), utt("u3", "bring her to the axis, rotation,", 50.0),
          utt("u4", "the axis of his chest turns", 70.0), utt("u5", "you return to the axis", 90.0), utt("u6", "bring the follower back to the axis", 110.0)]
    found = fl.back_to_axis(us)
    assert [f["utt"] for f in found] == ["u1", "u2", "u3", "u5", "u6"]
    by = {f["utt"]: f for f in found}
    assert by["u1"]["variant"] == "back to the axis" and by["u1"]["words"].startswith("bring")
    assert by["u3"]["variant"] == "to the axis"                      # no 'back': a close variant, flagged as such
    assert by["u6"]["words"] == "bring the follower back to the axis"
    assert by["u1"]["t0"] == pytest.approx(10.0) and by["u1"]["axis_t"] == pytest.approx(11.5)


def test_drop_axis_events_only_takes_the_phrase_s_axis_call():
    ph = [{"axis_t": 11.5}]
    ev = [{"id": "a", "start": 11.52, "triggers": ["axis"], "tags": ["overlay_T_axis"]},
          {"id": "b", "start": 12.4, "triggers": ["rotate"], "tags": ["overlay_T_axis"]},
          {"id": "c", "start": 30.0, "triggers": ["axis"], "tags": ["overlay_T_axis"]}]
    keep, dropped = fl.drop_axis_events(ev, ph)
    assert [e["id"] for e in dropped] == ["a"] and [e["id"] for e in keep] == ["b", "c"]


def _cb(*ivs):
    out = []
    for kind, t0, t1, pivot in ivs:
        out.append({"kind": kind, "t0": t0, "t1": t1, "pivot": pivot, "pivot_foot": "left" if pivot else None, "pivot_t0": t0 if pivot else None,
                    "pivot_t1": t1 if pivot else None, "swept_deg": 200.0 if pivot else None})
    return {"time_to_audio": 0.0, "intervals": out}


def test_pivot_target_prefers_the_counterbalance_pivot_then_the_couple_com():
    cb = _cb(("counterbalance", 20.0, 21.0, None), ("pivot", 24.0, 26.0, [0.1, 0.0, 0.2]))
    t = fl.pivot_target(cb, 22.0)
    assert t["target"]["type"] == "pivot_foot" and 24.0 <= t["target"]["interval_t"] <= 26.0   # inside the interval: the viewer resolves it there
    t = fl.pivot_target(_cb(("counterbalance", 20.0, 21.0, None)), 21.5)
    assert t["target"] == {"type": "couple_com"} and t["interval"]["kind"] == "counterbalance"
    assert fl.pivot_target(_cb(), 5.0)["interval"] is None


def test_the_duration_rule_and_its_easing():
    cfg = fl.FlowCfg()
    assert fl.smootherstep(0) == 0 and fl.smootherstep(1) == 1 and fl.smootherstep(0.5) == pytest.approx(0.5)
    assert fl.smootherstep(0.25) + fl.smootherstep(0.75) == pytest.approx(1.0)            # symmetric ease in / out
    a = fl.rule_duration("flight", 0, 0, 0, cfg)
    b = fl.rule_duration("flight", 90, 0, 0, cfg)
    c = fl.rule_duration("flight", 180, 3, 3, cfg)
    assert a == cfg.rule["flight"]["lo"] and a < b < c <= cfg.rule["flight"]["hi"]
    assert fl.rule_duration("phone_in", 30, 0, 0, cfg) == fl.rule_duration("phone_out", 30, 0, 0, cfg)   # in and out are timed alike
    # a phone way past an overhead: the crane part on top, and never quicker than the plain crane (owner: "Keep overheads, crane up")
    assert fl.rule_duration("phone_in", 0, 0, 0, cfg, crane=True) == pytest.approx(max(cfg.rule["phone_in"]["lo"] + cfg.crane_extra_s, cfg.rule["crane_up"]["lo"]))
    # the arcs never whip: the slowest rule for the largest turn keeps the peak (1.875 x mean) under the limit
    T = fl.rule_duration("phone_to_phone", 170, 0, 0, cfg)
    assert 1.875 * 170 / (T * (1 - cfg.axis_share)) <= fl.MAX_ARC_DEG_S


def test_side_start_continues_from_the_previous_view_when_it_is_side_on():
    assert fl.side_start(0.0, 80.0, 45.0) == pytest.approx(80.0)        # 10 deg off side-on (+90): no travel
    a = fl.side_start(0.0, 10.0, 45.0)                                 # along the dancers' line: to the nearest cone edge
    assert a == pytest.approx(50.0)
    assert fl.side_start(0.0, -170.0, 40.0) == pytest.approx(-125.0)            # 5 deg inside the cone round -90


def test_place_keeps_the_phone_video_and_lands_before_the_word():
    cfg = fl.FlowCfg()
    st, _ = fl.place("phone_out", 100.0, 2.0, cfg)
    assert st == pytest.approx(100.0 - cfg.depart_before_s)            # the phone keeps the screen until its segment ends
    st, _ = fl.place("phone_in", 100.0, 2.0, cfg)
    assert st + 2.0 == pytest.approx(100.0 + cfg.phone_arrive_after_s)  # the phone's picture fills the frame at its segment start (0.3 s before the words)
    st, _ = fl.place("flight", 100.0, 2.0, cfg)
    assert st + 2.0 == pytest.approx(100.0 + cfg.arrive_after_s)       # a 3D view has arrived as the words begin
    st, _ = fl.place("phone_to_phone", 100.0, 3.0, cfg)
    assert st == pytest.approx(98.5)
    st, _ = fl.place("phone_out", 100.0, 2.0, cfg, word=100.8)         # a call 0.8 s in: the view has arrived before it
    assert st + 2.0 <= 100.8 - 0.1 + 1e-9
    assert fl.place("replay_in", 100.0, 1.2, cfg) == (100.0, 100.0)


def seg(sid, d0, d1, mode="orbit", off=0.0, annos=()):
    return {"id": sid, "dance": [d0, d1], "film": [d0 + off, d1 + off], "speed": 1.0, "camera": {"mode": mode}, "layers": {}, "annotations": list(annos),
            "audio": "locked", "view_state": "orbit"}


def test_carve_replaces_a_window_and_keeps_the_timeline_contiguous():
    a = {"id": "x_ar", "kind": "arrow3d", "dance_t": [3.0, 5.5], "text": "t"}
    segs = [seg("s1", 0.0, 4.0, annos=[a]), seg("s2", 4.0, 9.0, "camera_tour"), seg("s3", 9.0, 12.0)]
    new = {"id": "piv", "speed": 1.0, "camera": {"mode": "physics_3q"}, "layers": {}, "annotations": [], "audio": "locked", "view_state": "physics"}
    out = fl.carve(segs, 3.2, 6.0, new)
    assert [s["id"] for s in out] == ["s1", "piv", "s2_r", "s3"]
    for p, q in zip(out, out[1:]):
        assert p["dance"][1] == pytest.approx(q["dance"][0]) and p["film"][1] == pytest.approx(q["film"][0])
    assert out[0]["dance"] == [0.0, 3.2] and out[1]["dance"] == [3.2, 6.0]
    assert out[0]["annotations"] == []                                  # 0.2 s of the arrow left before the window: dropped (< 0.3 s)


def shot(kind, f0, f1, mode=None, phone=None, imp=1, replay=False):
    return fl.Shot([0], kind, mode or {"3d": "orbit", "phone": "camera_tour", "overhead": "overhead_floorcraft"}[kind], f0, f1, f0, f1, replay, phone, imp, "orbit")


def test_transition_types():
    t = fl.transition_type
    assert t(shot("3d", 0, 5), shot("3d", 5, 9)) == "flight"
    assert t(shot("3d", 0, 5), shot("phone", 5, 9, phone="00")) == "phone_in"
    assert t(shot("phone", 0, 5, phone="00"), shot("3d", 5, 9)) == "phone_out"
    assert t(shot("phone", 0, 5, phone="00"), shot("phone", 5, 9, phone="02")) == "phone_to_phone"
    assert t(shot("3d", 0, 5), shot("overhead", 5, 9)) == "crane_up"
    assert t(shot("overhead", 0, 5), shot("3d", 5, 9)) == "crane_down"
    assert t(shot("3d", 0, 5), shot("3d", 5, 9, replay=True)) == "replay_in"


def test_schedule_flags_a_view_that_cannot_hold_between_its_way_in_and_out():
    cfg = fl.FlowCfg()
    ok = [shot("3d", 0, 6), shot("phone", 6, 12, phone="00"), shot("3d", 12, 18)]
    for s in ok:
        s.a0 = s.a1 = 0.0
        s.r0 = s.r1 = 3.5
        s.h0 = s.h1 = 1.6
    assert fl.schedule(ok, [], None, cfg) == []
    hold = ok[2].trans["start_film"] if ok[2].trans else None
    assert hold is not None and hold - ok[1].arrive >= cfg.hold_s["phone"] - 1e-9
    short = [shot("3d", 0, 6), shot("phone", 6, 6.5, phone="00"), shot("3d", 6.5, 14)]
    for s in short:
        s.a0 = s.a1 = 0.0
        s.r0 = s.r1 = 3.5
        s.h0 = s.h1 = 1.6
    probs = fl.schedule(short, [], None, cfg)
    assert probs and probs[0]["k"] == 1                                  # the 0.5 s phone shot cannot hold 1.4 s of its video (late start + squeeze)


def test_shots_join_equal_cameras_and_split_at_a_dance_jump():
    a = seg("a", 0, 3)
    b = seg("b", 3, 5)
    c = seg("c", 1, 2)                                                    # the dance jumps back: a replay is always a new shot
    c["replay"] = True
    sh = fl.shots_of([a, b, c])
    assert [x.segs for x in sh] == [[0, 1], [2]]


def test_no_arc_whips_past_the_limit_however_short_the_rule():
    """v3.1: the quicker phone ways never turn faster than MAX_ARC_DEG_S at the smootherstep peak, also after the plan's 0.1-deg rounding and
    after a squeeze (the arc floor is part of the rule and of the squeeze floor)"""
    cfg = fl.FlowCfg()
    for typ in ("flight", "phone_in", "phone_out", "phone_to_phone"):
        for daz in range(0, 181, 5):
            for d in (daz, daz + 0.049):
                T = fl.rule_duration(typ, d, 0, 0, cfg)
                assert 1.875 * round(d, 1) / (round(T, 3) * fl.arc_share(typ, cfg)) <= fl.MAX_ARC_DEG_S + 1e-9
                assert fl.squeeze_floor(typ, d, cfg) >= fl.arc_floor(typ, d, cfg) - 1e-12
    assert fl.MAX_ARC_DEG_S <= 200.0


def test_the_frustum_surface_reads_on_the_quickest_phone_way():
    """every phone entry / exit shows the surface standing in the world for at least frustum_readable_s, down to the squeezed minimum"""
    cfg = fl.FlowCfg()
    for typ, roles in (("phone_in", ("enter",)), ("phone_out", ("leave",)), ("phone_to_phone", ("enter", "leave"))):
        T = fl.squeeze_floor(typ, 0.0, cfg)
        for role in roles:
            env = fl.frustum_envelope(role, typ, 50.0, T, cfg)
            assert env == sorted(env)                                         # a, b, c, d in order: fade in, opaque, fade out
            assert fl.frustum_readable(role, typ, env, 50.0, T, cfg) >= cfg.frustum_readable_s - 1e-9, (typ, role, T)
    # the axis dolly is not counted: a way spent entirely on the optical axis would read 0 s
    cfg2 = fl.FlowCfg(axis_share=1.0)
    env = fl.frustum_envelope("enter", "phone_in", 50.0, 1.0, cfg2)
    assert fl.frustum_readable("enter", "phone_in", env, 50.0, 1.0, cfg2) == pytest.approx(max(0.0, 50.0 - 0.5 * (env[0] + env[1])))


def test_the_phone_ways_are_quicker_than_v3_0_and_the_crane_is_not():
    """owner 2026-10-21: shorter flights into / out of the phones and a shorter phone hold; the overheads keep their crane as it was"""
    cfg = fl.FlowCfg()
    v30 = {"phone_in": 1.3, "phone_out": 1.3, "phone_to_phone": 1.9}
    for typ, lo in v30.items():
        assert fl.rule_duration(typ, 0, 0, 0, cfg) < lo
        assert fl.rule_duration(typ, 90, 1, 0, cfg) < {"phone_in": 2.4, "phone_out": 2.4, "phone_to_phone": 3.0}[typ]
    assert cfg.hold_s["phone"] < 2.0
    assert fl.rule_duration("crane_up", 0, 0, 0, cfg) == pytest.approx(1.6) and fl.rule_duration("crane_down", 40, 0, 0, cfg) >= 1.6
    assert fl.rule_duration("phone_out", 0, 0, 0, cfg, crane=True) >= fl.rule_duration("crane_up", 0, 0, 0, cfg)


def test_cfg_from_doc_reads_the_rules_a_direction_was_built_with():
    cfg = fl.FlowCfg()
    cfg.rule["phone_in"]["lo"] = 1.3
    doc = {"flow": {"defaults": {"rule": cfg.rule, "axis_share": 0.4, "ground_m": [0.5, 2.4], "hold_s": {"3d": 1.0, "phone": 2.0, "overhead": 2.2},
                                 "max_arc_deg_s": 160.0}}}
    c = fl.cfg_from_doc(doc)
    assert c.axis_share == 0.4 and c.rule["phone_in"]["lo"] == 1.3 and c.hold_s["phone"] == 2.0 and c.ground_m == (0.5, 2.4)
    assert fl.cfg_from_doc({}).axis_share == fl.FlowCfg().axis_share
