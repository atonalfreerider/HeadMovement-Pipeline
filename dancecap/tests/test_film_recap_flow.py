"""v3 of the class-recap direction (dancecap.film_recap_flow): the back-to-the-axis finder, the pivot target, the duration rule and its easing,
the side-on azimuth, where a transition sits, carving a pivot window into the segments, the schedule's holds; v4 (owner 2026-10-22): the C2 ramps
and way profiles, the live blend, the way durations and caps, the targets, the orbit plan, the long-hold planner, the cranes' holds; v4.1 (owner
2026-10-22 after the v3 films): the cruise crane, the targets, the no-reversal way link, the 20-s-hold planner, Jev's need series, the legend's
place and the crop that clears it, the overhead window; v4.2: the meter windows, the closer orbits and their eases; v4.3 (owner 2026-10-23):
the profile track's windows, solver, blend, the inline-lift twin, PoseModel playing the table, the revision gate. Pure functions, no capture data."""
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np

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


# ---------------------------------------------------------------- v3.2: the 3D views take only the time they need (owner "~50% phone")


def test_v3_2_phone_ways_rest_on_their_floors():
    """v3.2: the phone ways are as short as the frustum's readable floor and the 200 deg/s arc floor allow; the way out of a phone starts at the
    next segment (the phone keeps its full-frame video to its end)"""
    cfg = fl.FlowCfg()
    assert cfg.depart_before_s == 0.0
    for typ in ("phone_in", "phone_out", "phone_to_phone"):
        T = fl.rule_duration(typ, 0, 0, 0, cfg)
        assert T >= fl.readable_floor(typ, cfg) and T - fl.readable_floor(typ, cfg) < 0.1     # (the floor, not an old base, sets the shortest way)
        roles = {"phone_in": ("enter",), "phone_out": ("leave",), "phone_to_phone": ("enter", "leave")}[typ]
        for role in roles:
            env = fl.frustum_envelope(role, typ, 10.0, T, cfg)
            assert fl.frustum_readable(role, typ, env, 10.0, T, cfg) >= cfg.frustum_readable_s - 1e-9
    assert fl.rule_duration("phone_in", 30, 0, 0, cfg) == fl.rule_duration("phone_out", 30, 0, 0, cfg)


def test_full_frame_counts_from_the_arrival_to_the_way_out():
    sh = [shot("3d", 0, 6), shot("phone", 6, 12, phone="00"), shot("3d", 12, 18)]
    sh[1].trans = {"type": "phone_in", "start_film": 5.0, "arrive_film": 6.0}
    sh[2].trans = {"type": "phone_out", "start_film": 11.6, "arrive_film": 12.5}
    assert fl.full_frame_s(sh) == pytest.approx(5.6)


def test_split_at_film_keeps_the_time_map_and_refuses_a_replay():
    segs = [seg("a", 0.0, 4.0, off=2.0), seg("b", 4.0, 9.0, off=2.0)]
    assert fl.split_at_film(segs, 8.0)                                     # film 8.0 = master 6.0, inside b
    assert [s["id"] for s in segs] == ["a", "b", "b_r"]
    assert segs[1]["dance"] == [4.0, 6.0] and segs[2]["film"] == [8.0, 11.0]
    assert fl.split_at_film(segs, 6.0) and len(segs) == 3                  # an existing boundary: nothing to do
    r = seg("r", 1.0, 3.0, off=10.0)
    r["replay"] = True
    assert not fl.split_at_film([r], 12.0)


def test_chapters_must_open_their_shots():
    a, b, c = seg("a", 0, 3), seg("b", 3, 5, "camera_tour"), seg("c", 5, 8, "camera_tour")
    assert fl.chapters_start_shots([a, b, c], {"a", "b"})                # b opens the phone shot b + c
    assert not fl.chapters_start_shots([a, b, c], {"c"})                 # c is in the middle of it


def test_join_idx_gives_the_survivor_s_view_to_the_gone_segments():
    a = seg("a", 0, 3, "physics_3q")
    a["layers"] = {"skeleton_mode": "physics"}
    b = seg("b", 3, 5, "orbit")
    b["layers"] = {"source_video": "00"}
    segs = [a, b]
    assert fl.join_idx(segs, [0], [1], "3d", None, None)
    assert segs[1]["camera"] == {"mode": "physics_3q"} and segs[1]["v3_merged_from"] == "orbit" and "source_video" not in segs[1]["layers"]
    assert not fl.join_idx([seg("x", 0, 2, "camera_tour"), seg("y", 2, 4)], [0], [1], "phone", "00", None)   # no phone data: not possible


def test_locked_dance_maps_film_to_master_only_on_locked_segments():
    a, b = seg("a", 10.0, 14.0, off=5.0), seg("b", 14.0, 20.0, off=5.0)
    r = seg("r", 12.0, 13.0, off=10.0)
    r["replay"] = True
    assert fl.locked_dance([a, b], 21.0, -1.0) == pytest.approx(16.0)
    assert fl.locked_dance([r], 22.5, -1.0) == -1.0                      # inside a replay: the fallback


# ---------------------------------------------------------------- v4 (owner 2026-10-22): moves / 3, phone holds x 3, slow orbits, C2 glides, faint frustums


def test_v4_ramps_are_c2_at_their_window_edges():
    """every v4 weight is a smootherstep ramp: value 0 / 1 and zero 1st and 2nd derivative at both ends of its window, so a blend of LIVE end
    states leaves the departing view with its own velocity and acceleration and lands on the arriving view's"""
    h = 1e-4
    for a, b in ((0.0, 0.75), (0.25, 1.0), (0.35, 0.65)):
        for u in (a, b):
            d1 = (fl.ramp(u + h, a, b) - fl.ramp(u - h, a, b)) / (2 * h)
            d2 = (fl.ramp(u + h, a, b) - 2 * fl.ramp(u, a, b) + fl.ramp(u - h, a, b)) / (h * h)
            assert abs(d1) < 1e-3 and abs(d2) < 1e-1
        assert fl.ramp(a, a, b) == 0.0 and fl.ramp(b, a, b) == 1.0


def test_v4_profiles_end_on_the_views_and_phone_out_is_phone_in_reversed():
    g = 0.15
    w0, w1 = fl.profile_weights("phone_in", 0.0, g), fl.profile_weights("phone_in", 1.0, g)
    assert w0["glide"] == 0 and w1["glide"] == 1 and w0["d_in"] == 1 and w1["d_in"] == 0      # from the orbit to the portal, then on the phone
    assert w0["picture_in"] == 0 and w1["picture_in"] == 1 and w0["outline_in"] == pytest.approx(g) and w1["outline_in"] == pytest.approx(1.0)
    for u in (0.0, 0.1, 0.3, 0.5, 0.62, 0.8, 1.0):
        i, o = fl.profile_weights("phone_in", 1.0 - u, g), fl.profile_weights("phone_out", u, g)
        assert o["glide"] == pytest.approx(1.0 - i["glide"]) and o["d_out"] == pytest.approx(i["d_in"])
        assert o["picture_out"] == pytest.approx(i["picture_in"]) and o["outline_out"] == pytest.approx(i["outline_in"])
    p0, p1 = fl.profile_weights("phone_to_phone", 0.0, g), fl.profile_weights("phone_to_phone", 1.0, g)
    assert p0["d_out"] == 0 and p1["d_out"] == 1 and p0["d_in"] == 1 and p1["d_in"] == 0
    assert p0["picture_out"] == 1 and p1["picture_out"] == 0 and p0["picture_in"] == 0 and p1["picture_in"] == 1
    # the eye is on the phone's optical axis (the glide done) before the dolly ends: the picture is always in front of the camera
    assert fl.V4_PROFILE["phone_in"]["glide"][1] < fl.V4_PROFILE["phone_in"]["dolly"][1]


def test_v4_live_blend_keeps_the_orbits_velocity():
    """a way from a turning orbit (A: azimuth w t) into a still phone (B): at the way's start the camera turns exactly at the orbit's rate (no
    stop), with no acceleration jump, and at its end at the phone's (0) - the C2 promise of UNITY_CONTRACT_v4.md section 2"""
    w, T, t0 = 6.0, 3.5, 10.0

    def az(t):
        u = (t - t0) / T
        gl = fl.profile_weights("phone_in", min(1.0, max(0.0, u)))["glide"]
        return (1 - gl) * (w * t) + gl * 30.0
    h = 1e-4
    assert (az(t0 + h) - az(t0 - h)) / (2 * h) == pytest.approx(w, abs=1e-3)
    assert abs((az(t0 + h) - 2 * az(t0) + az(t0 - h)) / (h * h)) < 1e-1
    assert abs((az(t0 + T + h) - az(t0 + T - h)) / (2 * h)) < 1e-3


def test_v4_way_durations_keep_the_picture_readable_and_the_crane_under_its_cap():
    cfg4 = fl.V4Cfg()
    for typ in ("phone_in", "phone_out", "phone_to_phone"):
        T = fl.v4_way_T(typ, cfg4)
        lo, hi, _ = cfg4.dur[typ]
        assert lo <= T <= hi and 2.5 <= T <= 5.0
        for role, u in fl.picture_readable_u(typ).items():
            assert u * T >= cfg4.readable_s - 1e-9, (typ, role)
    for typ in ("crane_up", "crane_down"):
        T = fl.v4_way_T(typ, cfg4)
        assert 1.875 * 74.0 / T <= cfg4.cap_crane_deg_s + 1e-9          # the pitch from the orbit (about 16 deg) to 89.9 deg within the cap
    assert cfg4.cap_az_deg_s <= 168.0 / 5 and cfg4.ghost_opacity == 0.15  # well below v3.2's measured 168 deg/s; the owner's 15 %


def test_v4_targets_are_the_owners_factors_of_the_baseline():
    base = {"moves": {"total": 75}, "phone_holds": {"mean_s": 4.36, "median_s": 3.87, "min_s": 1.4}}
    tg = fl.v4_hold_targets(base, fl.V4Cfg())
    assert tg["moves_max"] == 25 and tg["hold_mean_min_s"] == pytest.approx(13.08) and tg["hold_median_min_s"] == pytest.approx(11.61)
    assert tg["hold_min_s"] == pytest.approx(4.2) and tg["orbit_rate_deg_s"] == [4.0, 10.0] and tg["way_s"] == [2.5, 5.0]


def test_v4_orbit_plan_links_two_phones_without_reversing():
    cfg4 = fl.V4Cfg()
    pl = fl.orbit_rate_plan(0.0, 100.0, 30.0, 3.5, 3.5, cfg4)
    lo, hi = cfg4.orbit_rate_deg_s
    assert lo <= abs(pl["w"]) <= hi
    for r in (pl["r_out"], pl["r_in"]):
        assert abs(r) < 2.0 or math.copysign(1.0, r) == math.copysign(1.0, pl["w"])   # the ways turn the orbit's way: no reversal
    # the stretch arrives where the phone is: a_out + w x (L - T_out - T_in) + r_in = az_to (mod 360)
    a_in = pl["a_out"] + pl["w"] * (30.0 - 7.0)
    assert fl.wrap180(a_in + pl["r_in"] - 100.0) == pytest.approx(0.0, abs=0.05)
    assert abs(pl["excess_deg"]) < 5.0
    # free ends (the opening, the film's end): the way eases into / out of the orbit at its natural turn, half the rate x the way
    op = fl.orbit_rate_plan(None, 40.0, 9.7, 0.0, 3.5, cfg4, 1.0)
    assert op["r_in"] == pytest.approx(0.5 * op["w"] * 3.5) and op["excess_deg"] == 0.0
    assert fl.orbit_excess(0.0, 170.0, 9.0, 4.5, cfg4) > fl.orbit_excess(0.0, 170.0, 60.0, 4.5, cfg4)   # a short stretch cannot carry a big turn
    assert fl.orbit_excess(0.0, 120.0, 30.0, 4.5, cfg4) < 1e-6


def test_v4_dp_plans_long_holds_round_a_protected_window_within_the_move_budget():
    cfg4 = fl.V4Cfg()
    B = [float(x) for x in range(0, 121)]
    prot = [(0.0, 4.3), (50.0, 62.0)]

    def free(a, b):
        return not any(p0 < b - 1e-3 and p1 > a + 1e-3 for p0, p1 in prot)

    def hold_q(cam, a, b):
        return {"score": 0.9 if cam == "00" else 0.8, "cover": 1.0, "height": 0.4, "strong": True}
    runs, info = fl.v4_dp(B, ["00", "01"], hold_q, lambda a, b, f: None, free, lambda a, b: 0.6 * (b - a),
                          lambda cam, a, b, q: 1.0 * (b - a), cfg4, max_moves=4, hold_min=12.0, open_end=4.3, az_of=lambda cam, f: 0.0)
    assert info["ok"] and runs[0]["kind"] == "orbit" and runs[0]["f0"] == 0.0 and runs[-1]["f1"] == 120.0
    for a, b in zip(runs, runs[1:]):
        assert a["f1"] == b["f0"] and a["kind"] != b["kind"]
    phones = [r for r in runs if r["kind"] == "phone"]
    assert phones and all(12.0 - 1e-6 <= r["f1"] - r["f0"] <= cfg4.hold_max_s for r in phones)
    assert all(free(r["f0"], r["f1"]) for r in phones)                                         # never over the protected window
    assert len(runs) - 1 <= 4                                                                   # the move budget
    for k, r in enumerate(runs):
        if r["kind"] == "orbit" and 0 < k < len(runs) - 1:
            assert r["f1"] - r["f0"] >= 2 * cfg4.way_margin_s + cfg4.pure_orbit_min_s - 1e-6
            assert free(r["f0"], r["f0"] + cfg4.way_margin_s) and free(r["f1"] - cfg4.way_margin_s, r["f1"])


def test_v4_crane_down_keeps_the_overheads_hold_and_the_orbit_after_it():
    cfg4 = fl.V4Cfg()
    sh = [shot("3d", 0, 10), shot("overhead", 10, 16), shot("3d", 16, 40), shot("3d", 40, 50, replay=True)]
    ways = [{"k": 1, "type": "crane_up", "start": 5.5, "T": 4.8, "notes": []}, {"k": 2, "type": "crane_down", "start": 10.6, "T": 4.8, "notes": []},
            {"k": 3, "type": "replay_in", "start": 40.0, "T": 2.5, "notes": []}]
    fl.v4_fit_cranes(ways, sh, cfg4)
    assert ways[1]["start"] >= 5.5 + 4.8 + cfg4.hold_s["overhead"] - 1e-9 and ways[1]["notes"]
    ways[1]["start"], ways[2]["start"] = 16.0, 21.0
    fl.v4_fit_cranes(ways, sh, cfg4)
    assert ways[1]["start"] + 4.8 + cfg4.hold_s["3d"] <= 21.0 + 1e-9                     # the orbit holds before the replay's flash


# ---------------------------------------------------------------- v4.1 (owner 2026-10-22 after the v3 films): longer holds placed by Jev's need_3d,
# slow orbits that never turn back, ways >= 4 s under 12 deg/s, cruise cranes under 20 deg/s, the physics legend in the 9:16 top band


def test_v41_cruise_crane_is_c2_and_peaks_at_one_over_one_minus_ease():
    e, h = 0.25, 1e-4
    assert fl.cruise(0.0, e) == 0.0 and fl.cruise(1.0, e) == pytest.approx(1.0)
    for x in (0.0, 1.0):
        d1 = (fl.cruise(min(1.0, x + h), e) - fl.cruise(max(0.0, x - h), e)) / (min(1.0, x + h) - max(0.0, x - h))
        assert abs(d1) < 1e-3
    for x in (h, 1 - h):                                                     # 2nd derivative ~0 just inside both ends
        d2 = (fl.cruise(x + h, e) - 2 * fl.cruise(x, e) + fl.cruise(x - h, e)) / (h * h)
        assert abs(d2) < 0.5
    rates = [(fl.cruise(x + h, e) - fl.cruise(x - h, e)) / (2 * h) for x in [i / 200 for i in range(1, 200)]]
    assert max(rates) == pytest.approx(1.0 / (1.0 - e), rel=1e-3) and min(rates) >= -1e-9
    # the v4.1 cranes: 5.1 s keeps a 74.4-deg elevation change under the 20 deg/s pitch cap; a smootherstep crane would need 7 s
    c41 = fl.v41_cfg()
    T = fl.v4_way_T("crane_up", c41)
    assert T == pytest.approx(5.1) and c41.crane_span_deg / (1 - c41.crane_ease) / T <= c41.cap_crane_deg_s
    assert 1.875 * c41.crane_span_deg / T > c41.cap_crane_deg_s
    w = fl.profile_weights("crane_up", 0.5, 0.15, fl.V41_PROFILE["crane_up"])
    assert w["elevation"] == pytest.approx(0.5) and fl.V41_PROFILE["phone_in"] == fl.V4_PROFILE["phone_in"]
    assert fl.V4Cfg().prof() is fl.V4_PROFILE and c41.prof() is fl.V41_PROFILE        # v4.0 keeps its profiles (reproducible)


def test_v41_targets_are_the_owners_numbers():
    tg = fl.v41_targets(fl.v41_cfg())
    # the owner's decision of 2026-10-22: the CHECK holds the delivered meaning-first plan to >= 25 %; the PLANNER keeps its 45 % (so a
    # rebuild still delivers the same plan)
    assert tg["moves_max"] == 22 and tg["share_min"] == 0.25 and tg["share_plan_min"] == 0.45 and tg["share_decision"]
    assert fl.v41_cfg().share_min == 0.45 and tg["hold_min_s"] == 20.0 and tg["hold_mean_min_s"] == 25.0
    assert tg["way_min_s"] == 4.0 and tg["orbit_rate_deg_s"][1] <= 6.0 and tg["orbit_rate_deg_s"][0] > 0
    assert (tg["cap_az_deg_s"], tg["cap_rot_deg_s"], tg["cap_speed_m_s"], tg["cap_crane_deg_s"]) == (12.0, 12.0, 1.2, 20.0)
    c = fl.v41_cfg()
    for typ in ("phone_in", "phone_out", "replay_in", "replay_out", "crane_up", "crane_down"):
        assert fl.v4_way_T(typ, c) >= 4.0
    assert c.ghost_opacity == 0.15


def test_v41_way_peak_and_no_reversal_need_the_phone_ahead_of_the_orbit():
    # a way that only eases the orbit into a still phone half its natural turn ahead turns back for a moment; 0.75 of it never does
    assert fl._WAY_MIN[int(round(0.5 / 0.01))] < 0 and fl._WAY_MIN[int(round(0.75 / 0.01))] >= -1e-6
    assert fl.way_peak_factor(0.5) == pytest.approx(1.337, abs=0.01) and fl.way_peak_factor(0.75) == pytest.approx(1.778, abs=0.01)
    c = fl.v41_cfg()
    lk = fl.orbit_link(0.0, 90.0, 30.0, 4.5, 4.5, c)
    assert lk is not None and lk["rho"] >= 0.5 + c.orbit_lead - 1e-9 and lk["peak_az_est"] <= c.link_frac * c.cap_az_deg_s
    lo, hi = c.orbit_rate_deg_s
    assert lo <= abs(lk["w"]) <= hi
    # the stretch arrives on the phone: a_out + w (L - T_out - T_in) + r_in = az_to (mod 360), and both ways turn the orbit's way
    a_in = lk["a_out"] + lk["w"] * (30.0 - 9.0) + lk["r_in"]
    assert fl.wrap180(a_in - 90.0) == pytest.approx(0.0, abs=0.05)
    assert lk["r_out"] * lk["w"] > 0 and lk["r_in"] * lk["w"] > 0
    assert fl.orbit_link(0.0, 175.0, 14.0, 4.5, 4.5, c) is None            # a short stretch cannot carry a half turn slowly
    pl = fl.orbit_rate_plan(0.0, 90.0, 30.0, 4.5, 4.5, c)
    assert pl["w"] == lk["w"] and pl["r_in"] == lk["r_in"]


def test_v41_dp_plans_holds_of_20_s_ending_on_a_phone_without_phone_to_phone():
    c = fl.v41_cfg()
    B = [float(x) for x in range(0, 121)]
    prot = [(0.0, 4.3), (50.0, 62.0)]

    def free(a, b):
        return not any(p0 < b - 1e-3 and p1 > a + 1e-3 for p0, p1 in prot)

    def hold_q(cam, a, b, final):
        return {"score": 0.9, "cover": 1.0, "height": 0.4, "strong": True, "level": 0}
    runs, info = fl.v41_dp(B, ["00", "01"], hold_q, free, lambda a, b: 0.3 * (b - a), lambda cam, a, b, q: 1.0 * (b - a), c, max_moves=5, hold_min=20.0,
                           open_end=4.3, m_in=lambda cam, f: 4.5, m_out=lambda cam, f: 4.5, az_of=lambda cam, f: {"00": 0.0, "01": 100.0}[cam])
    assert info["ok"] and runs[0]["kind"] == "orbit" and runs[0]["f0"] == 0.0 and runs[-1]["f1"] == 120.0
    phones = [r for r in runs if r["kind"] == "phone"]
    assert phones and all(20.0 - 1e-6 <= r["f1"] - r["f0"] <= c.hold_max_s for r in phones) and all(free(r["f0"], r["f1"]) for r in phones)
    for a, b in zip(runs, runs[1:]):
        assert a["f1"] == b["f0"] and a["kind"] != b["kind"]                 # never phone to phone
    assert len(runs) - 1 <= 5 and runs[-1]["kind"] == "phone"               # the film may end on a phone (no way out)
    for k, r in enumerate(runs):
        if r["kind"] == "orbit" and 0 < k < len(runs) - 1:
            assert r["f1"] - r["f0"] >= 4.5 + 4.5 + c.pure_orbit_min_s - 1e-6
            assert free(r["f0"], r["f0"] + 4.5) and free(r["f1"] - 4.5, r["f1"])


def test_v41_need_series_maps_utterances_to_film_and_weights_by_speech(tmp_path):
    import json
    p = tmp_path / "need3d.json"
    p.write_text(json.dumps({"utterances": [{"id": "u000", "start": 1.0, "end": 3.0, "need_3d": 0.0, "overhead": 0.1, "text": "hi"},
                                            {"id": "u001", "start": 5.0, "end": 9.0, "need_3d": 3.0, "overhead": 0.9, "text": "the axis"}]}), encoding="utf-8")
    doc = {"segments": [seg("a", 0.0, 6.0), seg("b", 6.0, 12.0)], "timeline": {"film_duration_s": 12.0}}
    n = fl.NeedSeries(p, doc)
    assert n.mean(n.mask([(0.0, 12.0)]))[0] == pytest.approx(2.0, abs=0.05)          # 2 s at 0, 4 s at 3: speech-weighted
    assert n.mean(n.mask([(0.0, 12.0)]))[1] == pytest.approx(6.0, abs=0.15)
    i = int(4.0 / n.dt)
    assert not n.speech[i] and n.filled[i] == 0.0                                 # a silence takes the lower neighbour
    assert n.mean(n.mask([(3.5, 4.5)]))[0] is None


def test_v41_legend_moves_to_the_top_band_and_the_crop_clears_it():
    import numpy as np
    from dancecap import recap_pov as rp
    x0, x1, y0, y1 = fl.legend_rect(fl.V41_LEGEND)
    assert (round(x0, 3), round(x1, 3), round(y0, 4), round(y1, 4)) == (0.473, 0.987, 0.1442, 0.2155)
    assert fl.legend_rect(fl.LEGEND_UNITY_V3)[2] == pytest.approx(0.2025, abs=1e-4)        # the present Unity place: head height
    assert y0 >= 0.14                                                                      # inside the 9:16 safe area (top 14 %)
    t = np.arange(0.0, 10.0, 0.1)
    low = rp.PhoneData("x", t, np.ones(len(t), bool), np.tile([0.3, 0.3, 0.7, 0.7], (len(t), 1)), (1080, 1920))
    high = rp.PhoneData("x", t, np.ones(len(t), bool), np.tile([0.3, 0.06, 0.7, 0.55], (len(t), 1)), (1080, 1920))
    cfg41 = fl.dataclasses_replace(rp.PovCfg(), **fl.V41_CROP)
    rect = fl.legend_rect(fl.V41_LEGEND)
    assert fl.legend_clear_frames(low, cfg41, rect).all()                    # the v4.1 framing puts the heads under the legend
    assert not fl.legend_clear_frames(low, rp.PovCfg(), rect).all()          # the v4.0 framing (box from 15 % down) does not
    assert not fl.legend_clear_frames(high, cfg41, rect).any()               # a phone framed tight at the top cannot


def test_v41_overhead_window_lets_a_crane_down_leave_early_but_keeps_its_hold():
    c = fl.v41_cfg()
    segs = [seg("a", 0.0, 100.0), seg("o", 100.0, 106.0, mode="overhead_floorcraft"), seg("b", 106.0, 200.0)]
    segs[0]["chapter_card"] = {"film_out": 3.0}
    prot = fl.v4_protected(segs, c)
    ov = next(p for p in prot if p["kind"] == "overhead")
    up = 100.0 + c.arrive_after_s
    assert ov["f"][0] == pytest.approx(up - c.dur["crane_up"][1] - c.overhead_gap_s)
    down = max(up + c.hold_s["overhead"], 106.0 - c.overhead_leave_early_s)
    assert ov["f"][1] == pytest.approx(down + c.dur["crane_down"][1] + c.overhead_gap_s)
    assert ov["f"][1] < 106.0 + c.dur["crane_down"][1]                      # earlier than the v4.0 window's end
    p40 = next(p for p in fl.v4_protected([dict(s) for s in segs], fl.V4Cfg()) if p["kind"] == "overhead")
    assert p40["f"] == [100.0 - 5.0, 106.0 + 5.0]                            # v4.0 unchanged


# ---------------------------------------------------------------------------------------------- v4.2 (owner 2026-10-22 after v4.1)
def _seg(sid, f0, f1, physics, cam=None, card=None, tr=None, view="orbit"):
    c = dict(cam or {"mode": "orbit", "radius_m": 3.6, "radius_vertical_m": 4.3, "height_m": 2.15, "look_height_m": 0.95, "inline_lift_max_m": 0.25,
                     "v4": {"role": "slow_orbit", "run": 0}})
    if tr:
        c["transition_in"] = tr
    s = {"id": sid, "film": [f0, f1], "dance": [f0, f1], "camera": c, "view_state": view, "annotations": [],
         "layers": {"skeleton_mode": "physics", "physics": True, "legend": "x"} if physics else {"skeleton_mode": "rhythm"}}
    if card:
        s["chapter_card"] = {"kind": "chapter", "title": sid, "film_in": card[0], "film_out": card[1]}
    return s


def _doc(segs):
    return {"segments": segs, "timeline": {"film_duration_s": segs[-1]["film"][1]}, "flow": {"version": 2, "defaults": {"clear_radius_m": 1.4, "ground_m": [0.5, 2.4]}}}


def test_v42_meter_shows_a_few_seconds_at_every_switch_on_of_the_physics_view():
    phone = {"mode": "camera_tour", "cameras": [{"camera": "00"}], "v4": {"role": "phone_hold"}}
    segs = [_seg("a", 0.0, 10.0, False), _seg("b", 10.0, 30.0, True), _seg("c", 30.0, 40.0, True, cam=phone,
            tr={"type": "phone_in", "start_s": -4.0, "duration_s": 4.0, "layers_switch_s": 0.0}),
            _seg("d", 40.0, 50.0, False), _seg("e", 50.0, 70.0, True, card=(0.35, 2.95)), _seg("f", 70.0, 72.0, False), _seg("g", 72.0, 73.0, True),
            _seg("h", 73.0, 80.0, False)]
    doc = _doc(segs)
    pieces = fl.physics_timeline(doc)
    assert [(p["a"], p["b"], p["on"]) for p in pieces] == [(0.0, 10.0, False), (10.0, 40.0, True), (40.0, 50.0, False), (50.0, 70.0, True),
                                                           (70.0, 72.0, False), (72.0, 73.0, True), (73.0, 80.0, False)]
    m = fl.meter_windows(doc)
    assert m["switch_ons"] == 3 and m["dropped"] == 1 and m["schema"] == "hm film meter v1" and m["revision"] == "4.2"
    w = m["windows"]
    # the first: at the switch-on (10 s), 0.3-s fade in, shown to 14 s, 1-s fade out to 15 s; the physics view that stays on (b -> c) is no new switch-on
    assert (w[0]["start_s"], w[0]["full_s"], w[0]["fade_s"], w[0]["end_s"], w[0]["delay_s"]) == (10.0, 10.3, 14.0, 15.0, 0.0)
    # the second meets e's chapter card (50.35-52.95 s): it starts 0.2 s after the card
    assert w[1]["start_s"] == pytest.approx(53.15) and w[1]["end_s"] == pytest.approx(58.15) and w[1]["after"] and w[1]["delay_s"] == pytest.approx(3.15)
    # the 1-s physics view at 72 s: too short for a window (dropped, with a note); a meter never outlasts the physics view
    assert len(w) == 2 and any("72.00" in n for n in m["notes"])
    al = [fl.meter_alpha(m, t) for t in (9.99, 10.0, 10.15, 10.3, 12.0, 14.0, 14.5, 15.0, 15.01)]
    assert al[0] == 0.0 and al[1] == 0.0 and al[2] == pytest.approx(0.5) and al[3] == 1.0 and al[4] == 1.0 and al[5] == 1.0
    assert al[6] == pytest.approx(0.5) and al[7] == 0.0 and al[8] == 0.0
    # smooth: no step anywhere, never over the card, never without the physics view
    ts = [i / 300.0 for i in range(0, 80 * 300)]
    a = [fl.meter_alpha(m, t) for t in ts]
    assert max(abs(x - y) for x, y in zip(a[1:], a[:-1])) <= 1.875 / (300 * 0.3) + 1e-6      # the smootherstep's steepest step over the 0.3-s fade in
    for t, x in zip(ts, a):
        if x > 0:
            assert any(p["on"] and p["a"] <= t <= p["b"] for p in pieces)
            assert not (50.35 <= t <= 52.95)


def test_v42_meter_pulls_its_fade_in_when_the_physics_view_ends_early():
    doc = _doc([_seg("a", 0.0, 5.0, False), _seg("b", 5.0, 7.5, True), _seg("c", 7.5, 12.0, False)])
    w = fl.meter_windows(doc)["windows"]
    assert len(w) == 1 and w[0]["start_s"] == 5.0 and w[0]["end_s"] == 7.5 and w[0]["fade_s"] == pytest.approx(6.5)
    assert fl.meter_alpha({"windows": w}, 7.5) == 0.0 and fl.meter_alpha({"windows": w}, 6.0) == 1.0


def test_v42_layer_start_follows_the_way_and_transit_layers_count():
    # out of a phone the 3D layers apply from the way's start; a phone-to-phone way's transit layers switch the physics view on over their window
    phone = {"mode": "camera_tour", "cameras": [{"camera": "00"}], "v4": {"role": "phone_hold"}}
    phone2 = {"mode": "camera_tour", "cameras": [{"camera": "05"}], "v4": {"role": "phone_hold"}}
    segs = [_seg("a", 0.0, 10.0, False, cam=phone), _seg("b", 10.0, 20.0, False, cam=phone2,
            tr={"type": "phone_to_phone", "start_s": -2.0, "duration_s": 4.0, "layers_switch_s": 2.0,
                "transit_layers": {"film": [8.0, 12.0], "layers": {"skeleton_mode": "physics"}}}), _seg("c", 20.0, 30.0, False)]
    doc = _doc(segs)
    assert fl.layer_starts(doc) == [0.0, 12.0, 20.0]
    p = fl.physics_timeline(doc)
    assert [(x["a"], x["b"], x["on"]) for x in p] == [(0.0, 8.0, False), (8.0, 12.0, True), (12.0, 30.0, False)]
    w = fl.meter_windows(doc)["windows"]
    assert len(w) == 1 and w[0]["start_s"] == 8.0 and w[0]["end_s"] == 12.0


def test_v42_closer_orbit_is_0_8_x_about_the_couple_and_keeps_the_look_down_angle():
    cam = {"mode": "orbit", "radius_m": 3.6, "radius_vertical_m": 4.3, "height_m": 2.15, "look_height_m": 0.95, "inline_lift_max_m": 0.25,
           "v4": {"role": "slow_orbit", "run": 2}}
    c, info = fl.closer_block(cam)
    assert c["radius_m"] == pytest.approx(2.88) and c["radius_vertical_m"] == pytest.approx(3.44) and c["height_m"] == pytest.approx(1.91)
    assert c["inline_lift_max_m"] == pytest.approx(0.2) and c["lens_distance_scale"] == pytest.approx(1.25) and not info["clamped"]
    # the eye moves in along its line of sight: the same elevation, 0.8 x the distance to the look point
    for rk in ("radius_m", "radius_vertical_m"):
        assert math.atan2(c["height_m"] - 0.95, c[rk]) == pytest.approx(math.atan2(2.15 - 0.95, cam[rk]), abs=1e-3)
        assert math.hypot(c[rk], c["height_m"] - 0.95) == pytest.approx(0.8 * math.hypot(cam[rk], 1.2), abs=2e-3)
    assert c["v4"]["closer"]["v41"]["radius_vertical_m"] == 4.3 and cam["radius_m"] == 3.6          # the v4.1 record; the input untouched
    # a radius that 0.8 x would put inside the couple's clear radius stops at it (reported)
    c2, info2 = fl.closer_block(dict(cam, radius_m=1.6))
    assert c2["radius_m"] == pytest.approx(1.4) and info2["clamped"] and "clear radius" in info2["clamped"][0]
    # apply_closer: every slow-orbit block and the ways' 3D-end estimates; a phone shot is untouched
    phone = {"mode": "camera_tour", "cameras": [{"camera": "00"}], "v4": {"role": "phone_hold"}}
    tr = {"type": "phone_out", "start_s": 0.0, "duration_s": 4.5, "from": {"kind": "phone", "radius_m": 4.6, "height_m": 1.6},
          "to": {"kind": "3d", "radius_m": 4.3, "height_m": 2.15}}
    doc = _doc([_seg("a", 0.0, 10.0, False, cam=phone), _seg("b", 10.0, 20.0, True, cam=dict(cam, transition_in=tr))])
    doc["flow"]["defaults"]["v4"] = {"orbit": {"block": {k: v for k, v in cam.items() if k != "v4"}}}
    rep = fl.apply_closer(doc)
    b = doc["segments"][1]["camera"]
    # a phone way's 3D end stays v4.1's (the orbit eases back to v4.1's distance at it: the way keeps its peaks)
    assert b["radius_vertical_m"] == pytest.approx(3.44) and b["transition_in"]["to"]["radius_m"] == pytest.approx(4.3)
    assert b["closer_in_s"] == [14.5, 20.0]                         # the way out ends at 14.5 s; the part ends at 20 s
    assert b["transition_in"]["from"]["radius_m"] == 4.6 and doc["segments"][0]["camera"] == phone
    assert doc["flow"]["defaults"]["v4"]["orbit"]["block"]["radius_m"] == pytest.approx(2.88) and doc["flow"]["defaults"]["inline_lift_max_m"] == pytest.approx(0.2)
    assert rep["runs"][0]["v42"]["radius_m"] == pytest.approx(2.88) and rep["runs"][0]["eye_distance_m"]["vertical"]["v42"] == pytest.approx(3.571, abs=1e-3)


def test_v42_closer_orbit_eases_back_to_v41_distance_at_every_phone_way():
    """the phone ways keep v4.1's paths (their peaks under the caps): the orbit is at v4.1's distance at each phone way and 0.8 x inside"""
    orbit = {"mode": "orbit", "radius_m": 3.6, "radius_vertical_m": 4.3, "height_m": 2.15, "look_height_m": 0.95, "inline_lift_max_m": 0.25,
             "v4": {"role": "slow_orbit", "run": 1}}
    p0 = {"mode": "camera_tour", "cameras": [{"camera": "00"}], "v4": {"role": "phone_hold"}}
    p1 = {"mode": "camera_tour", "cameras": [{"camera": "06"}], "v4": {"role": "phone_hold"},
          "transition_in": {"type": "phone_in", "start_s": -4.0, "duration_s": 4.0}}
    ob = dict(orbit, transition_in={"type": "phone_out", "start_s": 0.0, "duration_s": 4.0})
    over = {"mode": "overhead_floorcraft", "v4": {"role": "overhead"}, "transition_in": {"type": "crane_up", "start_s": -4.75, "duration_s": 5.1}}
    ob2 = dict(orbit, transition_in={"type": "crane_down", "start_s": -2.0, "duration_s": 5.1})
    p2 = dict(p0, transition_in={"type": "phone_in", "start_s": -4.0, "duration_s": 4.0})
    doc = _doc([_seg("a", 0.0, 10.0, False, cam=p0), _seg("b", 10.0, 25.0, True, cam=ob), _seg("b2", 25.0, 40.0, True, cam=copy.deepcopy(ob)),
                _seg("c", 40.0, 60.0, True, cam=p1), _seg("d", 60.0, 80.0, True, cam=dict(ob, run=2)), _seg("o", 80.0, 86.0, True, cam=over),
                _seg("e", 86.0, 91.0, True, cam=ob2), _seg("f", 91.0, 100.0, True, cam=p2)])
    rep = fl.apply_closer(doc)
    segs = doc["segments"]
    b, b2, d, e = (segs[i]["camera"] for i in (1, 2, 4, 6))
    assert b == b2                                                         # one orbit part: the same block (one shot)
    assert b["closer_in_s"] == [14.0, 20.0] and b["closer_out_s"] == [30.0, 36.0]
    assert d["closer_in_s"] == [64.0, 70.0] and "closer_out_s" not in d       # the way after it is a crane: no ease needed there
    # after a crane down the ease may reach back into the crane (it reads the live orbit): 87 -> 84 s (the crane starts at 84 s)
    assert e["closer_out_s"] == [84.0, 87.0] and "closer_in_s" not in e
    assert len(rep["eases"]) == 4
    for cam, t, want in ((b, 14.0, 1.25), (b, 17.0, 1.125), (b, 25.0, 1.0), (b, 36.0, 1.25), (b, 12.0, 1.25), (d, 75.0, 1.0), (e, 87.0, 1.25), (e, 84.0, 1.0)):
        assert fl.closer_factor(cam, t) == pytest.approx(want, abs=1e-6), (t, want)
    # the dolly is slow: 0.89 m of eye distance (9:16 4.46 -> 3.57) over a 6-s smootherstep peaks at 1.875 x 0.89 / 6 = 0.28 m/s
    ts = [14.0 + i * 0.01 for i in range(601)]
    dist = [fl.closer_factor(b, t) * math.hypot(3.44, 1.91 - 0.95) for t in ts]
    assert max(abs(y - x) / 0.01 for x, y in zip(dist, dist[1:])) == pytest.approx(1.875 * (4.464 - 3.571) / 6.0, abs=0.01)
    assert fl.closer_factor(orbit, 50.0) == 1.0                              # a v4.1 block


# ---------------------------------------------------------------------------------------------- v4.3 (owner 2026-10-23: the initial counterbalances in profile)
def test_v43_versions_out_dirs_and_the_revision_gate():
    assert "v4.3" in fl.VERSIONS and "4.3" in fl.V41_STYLE
    assert fl._v4_out("v4.2") == fl.OUT_V4 / "v4.2" and fl._v4_out("v4") == fl.OUT_V4 and fl._v4_out("v4.3") == fl.OUT_V4
    assert fl._v4_out("v4.1") == fl.OUT_V41 and fl._v4_out("v4.0") == fl.OUT_V40
    c = fl.v43_cfg()
    assert c.version == "4.3" and c.prof() is fl.V41_PROFILE and c.orbit_r_h == fl.v42_cfg().orbit_r_h
    # a revision this code has no targets for is an error (never v4.0's targets by accident)
    assert fl.cmd_check_v4({"flow": {"version": 2, "revision": "4.9"}}, {}, {}, None) == 1


def test_v43_track_blend_is_the_closer_weight_and_joins_the_law_c2():
    cam = {"closer_in_s": [34.746, 40.746], "closer_out_s": [93.499, 99.499]}
    t = np.arange(1030, 2995) / 30.0
    w, w1, w2 = fl.track_blend(t, tuple(cam["closer_in_s"]), tuple(cam["closer_out_s"]))
    assert np.allclose(w, [fl.closer_weight(cam, float(x)) for x in t], atol=1e-12)
    # the derivatives are the weight's own (central differences), and all three vanish at and outside both ends: a C2 join onto the law
    h = 1e-4
    for x in (36.0, 38.5, 95.0, 97.9):
        wp, _, _ = fl.track_blend(np.array([x + h]), (34.746, 40.746), (93.499, 99.499))
        wm, _, _ = fl.track_blend(np.array([x - h]), (34.746, 40.746), (93.499, 99.499))
        w0, d1, d2 = fl.track_blend(np.array([x]), (34.746, 40.746), (93.499, 99.499))
        assert (wp - wm)[0] / (2 * h) == pytest.approx(d1[0], abs=1e-6) and (wp - 2 * w0 + wm)[0] / h ** 2 == pytest.approx(d2[0], abs=1e-3)
    for x in (34.746, 99.499, 30.0, 105.0):
        assert [float(v[0]) for v in fl.track_blend(np.array([x]), (34.746, 40.746), (93.499, 99.499))] == [0.0, 0.0, 0.0]


def test_v43_view_angle_and_the_inline_lift_twin():
    # the view looks at the couple from the eye's azimuth: an eye 90 deg off the line sees a profile two-shot, on the line one dancer hides the other
    assert fl.view_angle_to_line([90.0, -90.0, 0.0, 180.0, 450.0, 45.0], [0.0] * 6).tolist() == [90.0, 90.0, 180.0, 0.0, 90.0, 135.0]
    pl = np.zeros((5, 3))
    pf = np.array([[1.0, 0, 0], [1.0, 0, 0], [1.0, 0, 0], [0.1, 0, 0], [0.45, 0, 0]])
    lift = fl.inline_lift_twin(np.array([0.0, 180.0, 50.0, 0.0, 0.0]), pl, pf)
    # along the line (either end): 1.9 m when the dancers stand >= 0.7 m apart; 45 deg or more off it: 0; within 0.2 m: no line, 0; half way
    # between 0.2 and 0.7 m apart: half (the smoothstep of the distance)
    assert lift[:4].tolist() == pytest.approx([1.9, 1.9, 0.0, 0.0]) and lift[4] == pytest.approx(1.9 * 0.5)
    # the Hann window of FilmCamera.Orbit: a constant stays, a step becomes a smooth, monotone ramp over +-1.2 s
    y = fl.hann_mean(np.r_[np.zeros(100), np.ones(100)])
    assert len(y) == 200 - 72 and fl.hann_mean(np.ones(80)).tolist() == pytest.approx([1.0] * 8) and np.all(np.diff(y) >= -1e-12)


def _v43_doc(cb_ivs, layers_cb=()):
    """a synthetic direction: phone 0-10 s, the orbit (run 2) 10-110 s (way out 10-14 s; the way in 104-110 s belongs to the next phone shot),
    chapters 1-3 over it"""
    orbit = {"mode": "orbit", "world_azimuth_deg": 0.0, "drift_deg_s": 3.0, "radius_m": 2.88, "radius_vertical_m": 3.44, "height_m": 1.91,
             "look_height_m": 0.95, "inline_lift_max_m": 0.2, "ground_level": True, "closer_in_s": [14.0, 20.0], "closer_out_s": [98.0, 104.0],
             "v4": {"role": "slow_orbit", "run": 2}, "transition_in": {"type": "phone_out", "start_s": 0.0, "duration_s": 4.0}}
    p0 = {"mode": "camera_tour", "cameras": [{"camera": "00"}], "v4": {"role": "phone_hold", "run": 1}}
    p1 = {"mode": "camera_tour", "cameras": [{"camera": "06"}], "v4": {"role": "phone_hold", "run": 3},
          "transition_in": {"type": "phone_in", "start_s": -6.0, "duration_s": 6.0}}
    segs = [_seg("p0", 0.0, 10.0, False, cam=p0)]
    edges = [10.0, 30.0, 50.0, 70.0, 90.0, 110.0]
    for i, (a, b) in enumerate(zip(edges, edges[1:])):
        s = _seg(f"o{i}", a, b, True, cam=copy.deepcopy(orbit))
        if i in layers_cb:
            s["layers"]["counterbalance"] = True
        segs.append(s)
    segs.append(_seg("p1", 110.0, 130.0, False, cam=p1))
    for s in segs:
        s["speed"] = 1.0
    doc = _doc(segs)
    doc["chapters"] = [{"title": "0", "film_s": 0.0, "segment": "p0"}, {"title": "1", "film_s": 5.0, "segment": "p0"}, {"title": "2", "film_s": 50.0, "segment": "o2"},
                       {"title": "3", "film_s": 90.0, "segment": "o4"}, {"title": "4", "film_s": 112.0, "segment": "p1"}]
    cb = {"intervals": [{"kind": "counterbalance", "t0": a, "t1": b, "confidence": 0.7} for a, b in cb_ivs] + [{"kind": "pivot", "t0": 60.0, "t1": 62.0}]}
    return doc, cb


def test_v43_windows_score_each_side_outside_the_crossing_and_the_ways():
    doc, cb = _v43_doc([(20.0, 40.0), (75.0, 85.0), (101.0, 108.0)], layers_cb=(1, 2, 4))
    sh = fl.shots_of(doc["segments"])[1]
    w = fl.v43_windows(doc, sh, 14.0, 104.0, cb)
    # the crossing is the longest stretch of the track between two counterbalance intervals (the pivot does not count)
    assert w["crossing"] == [40.0, 75.0] and w["chapters_film"] == [5.0, 112.0]
    by = {r["name"]: r for r in w["windows"]}
    assert by["counterbalance 20.00-40.00"]["scored"] == [[20.0, 40.0, "first"]]
    assert by["counterbalance 75.00-85.00"]["scored"] == [[75.0, 85.0, "second"]]
    # the last interval reaches into the way in (v4.2's, unchanged): only its part before 104 s is scored
    assert by["counterbalance 101.00-108.00"]["scored"] == [[101.0, 104.0, "second"]] and "phone way" in by["counterbalance 101.00-108.00"]["not_scored"]
    # counterbalance-layer segments: 30-50 s -> 30-40 on the first side (40-50 is the crossing); 50-70 is all crossing; 90-110: 90-104 on the second
    assert by["o1"]["scored"] == [[30.0, 40.0, "first"]] and "crossing" in by["o1"]["not_scored"]
    assert by["o2"]["scored"] == [] and "20.00 s inside the crossing" in by["o2"]["not_scored"]
    assert by["o4"]["scored"] == [[90.0, 104.0, "second"]]


def test_v43_solver_holds_both_profiles_crosses_once_and_keeps_the_rules():
    """the quadratic programme on an almost still couple (the line along +x): from the law at 3 deg/s it slows into the first profile, crosses
    once in the stretch without an interval, holds the second profile and rejoins the law with the same rate and acceleration at both ends"""
    cfg = dict(fl.V43_TRACK, knot_s=2.0, hz=10.0)      # (10 Hz keeps the test quick; the build samples at 30)
    a, b = 14.0, 104.0
    t = np.arange(int(a * 10), int(b * 10) + 1) / 10.0
    W = 2.4
    law = 80.0 + W * (t - a)                          # the eye 80 deg off the line where the way out leaves it, 296 where the way in starts (as run 2)
    theta = 2.0 * np.sin(t)                           # a couple that sways a little
    items = [{"name": f"w{p0}", "kind": "interval", "part": [p0, p1], "side": side, "m": (t >= p0) & (t <= p1), "w0": 2.0}
             for p0, p1, side in ((22.0, 40.0, 90.0), (75.0, 98.0, 270.0))]
    cross = (t >= 40.0) & (t <= 75.0)
    sol = fl.solve_profile_track(t, law, W, theta, (14.0, 20.0), (98.0, 104.0), items, cfg, rate_max=np.where(cross, 7.0, 6.0))
    phi, rate, acc, inside = sol["phi"], sol["rate"], sol["accel"], sol["inside"]
    assert rate[inside].min() >= 1.0 and rate[inside].max() <= 7.0 and np.abs(acc[inside]).max() <= 2.0
    assert rate[inside & ~cross].max() <= 6.0                         # only the crossing may pass v4.1's 6 deg/s
    num = np.gradient(phi, 1 / 10.0)                  # the sampled curve agrees (what PoseModel and Unity see)
    assert num[inside].min() >= 1.0 - 1e-3 and num[inside].max() <= 7.0 + 1e-3
    for it in items:
        assert abs(it["median"] - 90.0) <= 15.0 and it["share"] >= 0.9, it
    ang = fl.view_angle_to_line(phi, theta)
    assert len(np.nonzero(np.diff(np.floor((phi - theta) / 180.0)))[0]) == 1 and ang[(t > 40) & (t < 75)].min() < 5.0
    # the law at and outside both ends, exactly; next to them the rate has had at most one sample of the accel limit to move (C2: w, w', w'' = 0 there)
    assert phi[0] == law[0] and phi[-1] == law[-1] and abs(rate[1] - W) <= 2.0 * 0.1 and abs(rate[-2] - W) <= 2.0 * 0.1


def test_v43_posemodel_plays_the_table_inside_its_window_only():
    doc, _ = _v43_doc([(20.0, 40.0)])
    i0, i1 = 14 * 30, 104 * 30
    tt = np.arange(i0, i1 + 1) / 30.0
    deg = (3.0 * (tt - 10.0) + 20.0 * np.sin(np.pi * (tt - 14.0) / 90.0) ** 3).tolist()
    lift = (0.3 * np.sin(np.pi * (tt - 14.0) / 90.0) ** 2).tolist()
    doc["flow"]["azimuth_tracks"] = {"run2": {"film": [14.0, 104.0], "t0": i0 / 30.0, "dt": 1 / 30.0, "n": len(tt), "deg": deg, "lift_m": lift,
                                              "law": {"world_azimuth_deg": 0.0, "drift_deg_s": 3.0, "F0": 10.0}}}
    for s in doc["segments"][1:-1]:
        s["camera"]["azimuth_track"] = "run2"

    class G:
        def centre(self, d):
            return np.zeros(3)

        def phone(self, cam, d):
            return {"pos": np.array([4.0, 1.5, 0.0]), "fwd": np.array([-1.0, 0.0, 0.0]), "vfov": 50.0, "aspect": 0.5625}
    pm = fl.PoseModel(doc, G(), cfg4=fl.v43_cfg())
    assert len(pm.shots) == 3                                        # the tracked blocks are still one shot
    for f in (12.0, 14.0, 104.0, 106.0):                             # the ways and the window's own ends: the law, no lift
        st = pm.state(1, f)
        assert st["az"] == pytest.approx(3.0 * (f - 10.0), abs=1e-12) and st["eye"][1] == pytest.approx(1.91)
    for f in (30.0, 59.0 + 1 / 60.0, 80.0):                          # inside: the table, linear between its samples, and its lift
        st = pm.state(1, f)
        assert st["az"] == pytest.approx(np.interp(f, tt, deg)) and st["eye"][1] == pytest.approx(1.91 + np.interp(f, tt, lift))
    # an untracked twin of the same direction: identical (bit for bit) outside the window, different inside
    doc2 = copy.deepcopy(doc)
    for s in doc2["segments"]:
        s["camera"].pop("azimuth_track", None)
    pm2 = fl.PoseModel(doc2, G(), cfg4=fl.v43_cfg())
    for f in (11.0, 13.9, 104.0, 105.0, 108.0):
        assert np.array_equal(pm.pose(f)["eye"], pm2.pose(f)["eye"]) and np.array_equal(pm.pose(f)["dir"], pm2.pose(f)["dir"])
    assert not np.array_equal(pm.pose(50.0)["eye"], pm2.pose(50.0)["eye"])
    # the direction is the untracked one + the track keys
    doc2["flow"].pop("azimuth_tracks")
    assert json.dumps(fl._strip_tracks(doc), sort_keys=True) == json.dumps(fl._strip_tracks(doc2), sort_keys=True)


class _StillCouple:
    """a PelvisLine stand-in: the couple stands still along +x (lead at the origin, the follow 0.6 m away), heads 1.6 / 1.5 m up"""
    PELVIS, SPINE3, NECK, HEAD, L_SH, R_SH = 0, 9, 12, 15, 16, 17

    def __init__(self):
        self.t = np.arange(0, 131 * 30) / 30.0
        self.theta = np.zeros(len(self.t))
        self.sha = {"lead": "lead-sha", "follow": "follow-sha"}

    def theta_at(self, d):
        return np.zeros(len(np.atleast_1d(d)))

    def joint_at(self, role, j, d):
        d = np.atleast_1d(np.asarray(d, float))
        x = 0.0 if role == "lead" else 0.6
        y = {0: 0.95, 9: 1.3, 12: 1.45, 15: 1.6 if role == "lead" else 1.5}.get(j, 1.0)
        return np.tile([x, y, 0.0], (len(d), 1))

    def damped(self, role, doc, film, tau=0.6):
        p = self.joint_at(role, 0, film)
        p[:, 1] = 0.0
        return p


def test_v43_fix1_the_crossing_lift_choice_and_the_fast_pass_option():
    """fix round 1: the decided track raises the eye in its crossing up to the band's top (crossing_lift "raised", the brief's; measured in the
    editor: it shows the follow's face a little longer), "none" is the other choice (no lift there; its numbers are reported next to it); the lift
    never moves the azimuth; the optional fast pass runs through the line faster only inside its own window, inline for less time"""
    doc, cb = _v43_doc([(22.0, 40.0), (75.0, 98.0)])
    for s in doc["segments"][1:-1]:
        s["camera"].update({"world_azimuth_deg": 70.4, "drift_deg_s": 2.4})      # psi 80 where the way out leaves the eye, 296 where the way in starts
    line = _StillCouple()
    cfg = dict(fl.V43_TRACK, knot_s=2.0, hz=10.0)
    quiet = lambda *a, **k: None                                                  # noqa: E731
    assert cfg["crossing_lift"] == "raised" and cfg["fast_pass"] is None           # the decided v4.3
    blk, rp = fl.profile_track(doc, 1, line, cb, cfg, log=quiet)
    assert blk["crossing"]["lift"] == "raised" and max(blk["lift_m"]) == pytest.approx(2.4 - 1.91, abs=0.01) and blk["rules"]["crossing_lift"] == "raised"
    assert rp["lift_alt_mode"] == "none" and float(np.max(rp["lift_alt"])) < 0.01 and blk["lift_m"][0] == 0.0 and blk["lift_m"][-1] == 0.0
    assert len(blk["crossing"]["line_at_s"]) == 1 and 40.0 < blk["crossing"]["line_at_s"][0] < 75.0 and "fast_pass" not in blk["rules"]
    blk_n, rp_n = fl.profile_track(doc, 1, line, cb, dict(cfg, crossing_lift="none"), log=quiet)
    assert max(blk_n["lift_m"]) < 0.01 and blk_n["deg"] == blk["deg"] and rp_n["lift_alt_mode"] == "raised"      # the lift never moves the azimuth
    assert float(np.max(rp_n["lift_alt"])) == pytest.approx(max(blk["lift_m"]), abs=1e-4)
    with pytest.raises(fl.fr.RecapError):
        fl.profile_track(doc, 1, line, cb, dict(cfg, crossing_lift="up"), log=quiet)
    # the fast pass: faster than 7 only inside its window round the decided crossing, <= 7 in the rest of the gap, <= 6 outside the gap
    blk_f, rp_f = fl.profile_track(doc, 1, line, cb, dict(cfg, fast_pass=dict(fl.V43_FAST_PASS)), log=quiet)
    fp = blk_f["rules"]["fast_pass"]
    t, rate, ins = rp_f["t"], rp_f["rate_s"], (rp_f["t"] > 14.0) & (rp_f["t"] < 104.0)
    fw = (t >= fp["film"][0]) & (t <= fp["film"][1])
    gap = (t >= 40.0) & (t <= 75.0)
    assert blk_f["rules"]["rate_max_deg_s"] == 12.0 and blk_f["rules"]["rate_max_deg_s_gap"] == 7.0
    assert fp["film"] == [pytest.approx(blk["crossing"]["line_at_s"][0] - 4.0, abs=0.11), pytest.approx(blk["crossing"]["line_at_s"][0] + 4.0, abs=0.11)]
    assert 8.0 < rate[fw].max() <= 12.0 and rate[ins & gap & ~fw].max() <= 7.0 and rate[ins & ~gap].max() <= 6.0 and rate[ins].min() >= 1.0
    assert np.abs(rp_f["accel_s"][ins]).max() <= 2.0
    cr = blk_f["crossing"]["line_at_s"]
    assert len(cr) == 1 and fp["film"][0] < cr[0] < fp["film"][1]
    # it spends less time near the line than the decided track (the angle to the line under 15 deg), the profiles' windows still met
    assert np.sum(rp_f["angle"] < 15.0) < np.sum(rp["angle"] < 15.0)
    for w_ in blk_f["windows"]:
        for p_ in w_.get("planned", []):
            assert abs(p_["median_deg"] - 90.0) <= 15.0, w_
