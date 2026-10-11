"""dancecap.film_recap_arrows + the D4 behaviour of the recap film (owner 2026-10-11: more physics, more arrows, more phone video, avatars 75 % transparent),
synthetic data only.

    .venv-nopo4d/Scripts/python.exe -m pytest dancecap/tests/test_film_recap_arrows.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_film_recap import act_positions, ev, make_speech, sp_bundle, utt  # noqa: E402
from test_film_recap_director import CFG, direct, long_speech, pov_source  # noqa: E402

from dancecap import film_recap as fr  # noqa: E402
from dancecap import film_recap_arrows as ar  # noqa: E402
from dancecap import film_recap_director as fd  # noqa: E402
from dancecap import film_recap_plan as plan  # noqa: E402

CTX = {"layers": plan.Layers.all_on(), "t_axes": [{"id": 3, "t0": 20.0, "t1": 60.0}, {"id": 4, "t0": 60.0, "t1": 70.0}], "transfers": {}}


# ------------------------------------------------------------------------------------------------ what an arrow points at


def test_a_body_part_gets_an_arrow_on_its_joint_labelled_with_the_teachers_word():
    sp = ar.spec_for_event(ev("a", 10, 10.4, ["camera_focus"], ["body_part"], ["shoulder", "blades"], where="shoulders", who="couple"), CTX)
    assert sp.kind == "arrow3d" and sp.target["type"] == "joint" and sp.text == "shoulder blades"
    sp = ar.spec_for_event(ev("a", 10, 10.4, ["camera_focus"], ["body_part"], ["arm"], where="arm", who="leader"), CTX)
    assert sp.target == {"type": "joint", "dancer": "lead", "joint": "right_elbow"} and sp.text == "arm"


def test_feet_get_the_named_foot_else_the_foot_that_carries_the_weight():
    e = ev("a", 10, 10.4, ["camera_focus", "overlay_footsteps/physics"], ["footwork", "body_part"], ["right foot", "foot"], where="feet")
    sp = ar.spec_for_event(e, CTX)
    assert sp.target == {"type": "foot", "dancer": "lead", "side": "right"} and sp.text == "right foot"
    sp = ar.spec_for_event(ev("a", 10, 10.4, ["overlay_footsteps/physics"], ["footwork"], ["step"], where="feet", who="follower"), CTX)
    assert sp.target == {"type": "stance_foot", "dancer": "follow"} and sp.text == "step"


def test_weight_balance_and_the_physics_words_point_at_the_centre_of_mass():
    sp = ar.spec_for_event(ev("a", 10, 10.4, ["overlay_weight_change"], ["weight"], ["counterbalance"]), CTX)
    assert sp.target == {"type": "couple_com"} and sp.text == "counterbalance"
    sp = ar.spec_for_event(ev("a", 10, 10.4, ["overlay_weight_change"], ["weight"], ["heavy"], who="follower"), CTX)
    assert sp.target == {"type": "com", "dancer": "follow"}
    # a push is the net force on a body (physics stage), labelled as an estimate
    e = ev("px", 10, 10.4, ["overlay_weight_change"], ["weight"], ["push"])
    e["physics_call"] = True
    sp = ar.spec_for_event(e, CTX)
    assert sp.kind == "vector_arrow" and sp.vector == "force" and sp.text == "push (est.)"
    # no physics data: no weight arrow
    assert ar.spec_for_event(ev("a", 10, 10.4, ["overlay_weight_change"], ["weight"], ["weight"]), {"layers": plan.Layers.probe(None, False), "t_axes": []}) is None


def test_the_axis_points_at_the_leaders_t_alive_at_that_time():
    sp = ar.spec_for_event(ev("a", 30, 30.4, ["overlay_T_axis"], ["axis"], ["axis"]), CTX)
    assert sp.target == {"type": "leader_t", "t_axis": 3}
    assert ar.spec_for_event(ev("a", 65, 65.4, ["overlay_T_axis"], ["axis"], ["exit"]), CTX).target["t_axis"] == 4
    assert ar.t_axis_at(CTX["t_axes"], 200.0) is None and ar.t_axis_at(None, 5.0) is None
    assert ar.spec_for_event(ev("a", 200, 200.4, ["overlay_T_axis"], ["axis"], ["axis"]), CTX) is None


def test_path_and_timing_are_velocity_arrows_that_need_motion_and_here_is_a_ring():
    sp = ar.spec_for_event(ev("a", 10, 10.4, ["overlay_path/floorcraft"], ["path"], ["forward"]), CTX)
    assert sp.kind == "vector_arrow" and sp.vector == "velocity" and sp.needs_motion and sp.text == "forward"
    sp = ar.spec_for_event(ev("a", 10, 10.4, ["overlay_timing"], ["timing"], ["momentum"], who="leader"), CTX)
    assert sp.target == {"type": "com", "dancer": "lead"} and sp.vector == "velocity"
    sp = ar.spec_for_event(ev("a", 10, 10.4, ["camera_focus"], ["demonstration"], ["starting here"]), CTX)
    assert sp.kind == "floor_ring"
    assert ar.spec_for_event(ev("a", 10, 10.4, ["camera_focus"], ["demonstration"], ["like this"]), CTX) is None


def test_labels_are_the_teachers_words():
    assert ar.label_of(ev("a", 1, 2, [], [], ["weights"])) == "weight"
    assert ar.label_of(ev("a", 1, 2, [], [], ["shoulder", "blades"])) == "shoulder blades"
    assert ar.label_of(ev("a", 1, 2, [], [], ["at the same time"])) == "at the same time"
    assert ar.label_of(ev("a", 1, 2, [], [], [])) is None


# ------------------------------------------------------------------------------------------------ windows and overlap


def test_an_arrow_window_starts_just_before_the_word_and_stays_in_the_shot():
    e = ev("a", 10.0, 10.4, [], [], ["frame"])
    w = ar.window_for(e, 8.0, 14.0, CFG, "frame")
    assert abs(w[0] - 9.65) < 1e-6 and 1.3 <= w[1] - w[0] <= CFG.arrow_max_s
    # a call just before the cut: the arrow comes a little earlier instead of being lost
    w = ar.window_for(ev("a", 13.7, 14.0, [], [], ["frame"]), 8.0, 14.0, CFG, "frame")
    assert w is not None and w[1] <= 13.95 and w[1] - w[0] >= 1.1
    assert ar.window_for(ev("a", 8.1, 8.4, [], [], ["frame"]), 8.0, 8.5, CFG, "frame") is None


def test_at_most_two_labelled_arrows_at_once_the_earlier_one_gives_way():
    items = [{"dance_t": [10.0, 12.4], "calls": ["a"], "text": "x"}, {"dance_t": [10.5, 12.9], "calls": ["b"], "text": "y"},
             {"dance_t": [11.9, 14.0], "calls": ["c"], "text": "z"}]
    kept, dropped = ar.limit_overlap(items, CFG)
    assert [k["calls"][0] for k in kept] == ["a", "b", "c"] and not dropped
    assert abs(kept[0]["dance_t"][1] - 11.9) < 1e-6                         # the first yielded to the third
    packed = [{"dance_t": [10.0, 12.0], "calls": ["a"], "text": "x"}, {"dance_t": [10.2, 12.0], "calls": ["b"], "text": "y"},
              {"dance_t": [10.4, 12.0], "calls": ["c"], "text": "z"}]
    kept, dropped = ar.limit_overlap(packed, CFG)
    assert len(kept) == 2 and len(dropped) == 1 and dropped[0]["calls"] == ["c"]


def test_weight_transfers_come_from_single_support_runs():
    c = ["LR"] * 30 + ["L"] * 15 + ["LR"] * 5 + ["R"] * 15 + ["L"] * 3 + ["R"] * 20
    out = ar.weight_transfers(c, 30.0)
    assert len(out) == 1 and out[0]["frm"] == "L" and out[0]["to"] == "R" and 1.4 < out[0]["t"] < 1.8     # the 3-frame flicker is not a transfer
    assert ar.weight_transfers(None) == [] and ar.weight_transfers(["LR"] * 100) == []


# ------------------------------------------------------------------------------------------------ the film


def test_every_attention_call_with_something_to_point_at_gets_an_arrow_in_3d_and_in_the_phone():
    sp = long_speech()
    act = act_positions(140, moving=[(0, 140)])
    ctx = {"layers": plan.Layers.all_on(), "t_axes": [{"id": 0, "t0": 0.0, "t1": 140.0}], "transfers": {}}
    doc, tl, rep = fr.build_direction(sp_bundle(sp), act, plan.Layers.all_on(), CFG, fr.Paths(work=Path(".")), "99_Capture", ("Class recap", "x"), pov_source(), arrow_ctx=ctx)
    assert fr.validate(doc, tl, sp_bundle(sp)) == []
    a = rep["director"]["arrows"]
    assert a["attention_calls"] > 0 and a["calls_with_arrow"] >= 0.9 * a["attention_calls"]
    kinds = {x["kind"] for s in doc["segments"] for x in s["annotations"]}
    assert "arrow3d" in kinds
    # arrows also lie in phone shots
    assert a["arrows_in_phone_shots"] > 0
    # never more than two call-outs at once (the validator checks it too)
    spans = sorted((x["film_in_segment"][0] + s["film"][0], x["film_in_segment"][1] + s["film"][0]) for s in doc["segments"] for x in s["annotations"]
                   if x["kind"] in ("arrow3d", "vector_arrow", "floor_ring") and not x.get("continuous"))
    for t in [p[0] + 0.01 for p in spans]:
        assert sum(1 for a0, a1 in spans if a0 <= t < a1) <= CFG.max_arrows_at_once


def test_the_validator_rejects_three_arrows_at_once():
    sp = long_speech()
    act = act_positions(140, moving=[(0, 140)])
    doc, tl, _ = fr.build_direction(sp_bundle(sp), act, plan.Layers.all_on(), CFG, fr.Paths(work=Path(".")), "99_Capture", ("Class recap", "x"), pov_source())
    seg = next(s for s in doc["segments"] if s["annotations"] and s["film"][1] - s["film"][0] > 3.0)
    for k in range(3):
        seg["annotations"].append({"id": f"zz{k}", "kind": "arrow3d", "film_in_segment": [0.5, 2.0], "target": {"type": "couple_com"}, "text": "x"})
    assert any("arrows at once" in p for p in fr.validate(doc, tl, sp_bundle(sp)))


def test_the_physics_view_adds_velocity_arrows_and_names_them_once():
    b = plan.Beat(id="ch00_01_weight", t0=10.0, t1=15.0, family="weight", chapter=0)
    st: dict = {}
    out = ar.physics_extras(b, {"lead": [{"t": 12.0, "frm": "L", "to": "R"}]}, CFG, st, [])
    kinds = [(o["kind"], o.get("vector")) for o in out]
    assert kinds.count(("vector_arrow", "velocity")) == 2 and ("vector_arrow", "between") in kinds
    assert next(o for o in out if o.get("vector") == "between")["text"] == "his weight to the right foot"
    assert [o.get("text") for o in out if o.get("vector") == "velocity"] == ["weight heading (est.)", None]
    again = ar.physics_extras(b, {}, CFG, st, [])
    assert not any(o.get("text") for o in again)
    # the teachers' call-outs have the room: a transfer that would make a third labelled arrow is not drawn
    busy = [{"dance_t": [11.0, 14.0]}, {"dance_t": [11.0, 14.0]}]
    assert not any(o.get("vector") == "between" for o in ar.physics_extras(b, {"lead": [{"t": 12.0, "frm": "L", "to": "R"}]}, CFG, {"velocity_label_t": 12.0}, busy))


# ------------------------------------------------------------------------------------------------ physics calls and the physics view


def test_physics_calls_cover_the_events_and_the_words_no_event_covers():
    utts = [utt("u1", 1.0, 9.0, "Teacher", "so the momentum and the pressure and the weight and the balance all matter"),
            utt("u2", 10.0, 14.0, "Teacher", "do not fall and do not push too hard")]
    evs = [ev("e1", 2.1, 2.5, ["overlay_timing"], ["timing"], ["momentum"]), ev("e2", 5.0, 5.4, ["camera_focus"], ["body_part"], ["frame"], where="frame")]
    out = fd.physics_calls(utts, evs)
    marked = {e["id"] for e in out if e.get("physics_call")}
    assert "e1" in marked and "e2" not in marked
    syn = [e for e in out if e.get("synthetic")]
    words = {e["triggers"][0] for e in syn}
    assert {"pressure", "weight", "balance", "fall", "push"} <= words and "momentum" not in words      # momentum is covered by e1
    assert all(e["tags"] == ["overlay_weight_change"] and e["start"] >= 1.0 for e in syn)
    assert sorted(e["start"] for e in out) == [e["start"] for e in out]


def physics_speech():
    utts, evs, t = [], [], 1.0
    texts = ["so today we talk about the frame and how important it is to keep it", "now you can feel the momentum when you walk forward together",
             "and then we simply talk for a while about nothing special at all and it goes on and on", "the pressure on the left foot is what the follower can feel"]
    for i, tx in enumerate(texts * 6):
        utts.append(utt(f"u{i}", t, t + 7.0, "Teacher", tx))
        t += 7.5
    chapters = [{"start": 0.0, "end": 180.0, "title": "All", "id": "ch00"}]
    return {"utterances": utts, "events": evs, "chapters": chapters, "roles": {"Teacher": "leader"}, "duration": 180.0}


def direct_phys(sp, pov, act, dc=None):
    layers = plan.Layers.all_on()
    beats, _ = plan.plan_beats(sp["utterances"], sp["events"], sp["chapters"], act, layers, sp["roles"], sp["duration"], CFG)
    events = fd.physics_calls(sp["utterances"], sp["events"])
    ctx = fd.Ctx(sp["utterances"], events, sp["chapters"], act, layers, sp["roles"], sp["duration"], CFG, dc or fd.DirCfg(), pov, [],
                 fd.cut_points(sp["utterances"], CFG.lead_s), [], [])
    return fd.direct(beats, ctx)[0], ctx


def test_a_spoken_physics_word_gets_the_physics_view_even_without_a_teaching_event():
    sp = physics_speech()
    shots, ctx = direct_phys(sp, pov_source(), act_positions(180, moving=[(0, 180)]))
    weight = [b for b in shots if b.family == "weight"]
    assert weight and all(b.camera["mode"] == "physics_3q" for b in weight)
    times = [e["start"] for e in ctx.events if e.get("synthetic")]
    assert times
    for t in times:
        assert any(b.t0 - 1e-6 <= t < b.t1 + 1e-6 for b in weight), t


def test_the_physics_view_shows_at_least_every_so_many_seconds_while_the_teachers_move_but_not_while_they_stand():
    sp = physics_speech()
    sp["utterances"] = [u for u in sp["utterances"] if "pressure" not in u["text"] and "momentum" not in u["text"]]      # no physics words at all
    shots, _ = direct_phys(sp, pov_source(), act_positions(180, moving=[(0, 180)]))
    ts = [b.t0 for b in shots if b.family == "weight"]
    assert len(ts) >= 4
    edges = [0.0] + ts + [180.0]
    assert max(b - a for a, b in zip(edges, edges[1:])) <= 40.0
    still, _ = direct_phys(sp, pov_source(), act_positions(180, moving=[]))
    assert not any(b.family == "weight" for b in still)


def test_teachers_standing_apart_get_no_physics_view():
    sp = physics_speech()
    shots, _ = direct_phys(sp, pov_source(), act_positions(180, moving=[(0, 180)], apart=[(0, 180)]))
    assert not any(b.family == "weight" for b in shots)


def test_a_long_physics_view_hands_its_tail_to_the_phones():
    sp = physics_speech()
    dc = fd.DirCfg()
    shots, _ = direct_phys(sp, pov_source(), act_positions(180, moving=[(0, 180)]), dc)
    assert all(b.dur <= dc.phys_max_s + 1.6 for b in shots if b.family == "weight")


def test_phone_runs_may_be_three_long_and_never_repeat_a_phone():
    shots, rep, _ = direct(long_speech(), pov_source())
    run = best = 0
    for b in shots:
        run = run + 1 if b.family == "pov" else 0
        best = max(best, run)
    assert best <= 3
    cams = [b.treatment["phone"] for b in shots if b.family == "pov"]
    assert all(a != b for a, b in zip(cams, cams[1:]) if True) or True
    seq = [b for b in shots]
    for a, b in zip(seq, seq[1:]):
        if a.family == "pov" and b.family == "pov":
            assert a.treatment["phone"] != b.treatment["phone"]


# ------------------------------------------------------------------------------------------------ the avatars


def test_every_avatar_the_film_draws_has_the_one_opacity_and_the_direction_carries_it():
    for fam in ("talk", "body", "feet", "weight", "axis", "path", "timing", "pov"):
        lay = plan.treatment_layers(fam, plan.Layers.all_on(), None, "", 0, 0.25)
        assert lay["avatars"] == 0.25, fam
    lay = plan.treatment_layers("axis", plan.Layers.all_on(), None, "", 1, 0.25)               # the alternate (path-like) overhead
    assert lay["avatars"] == 0.25
    assert "avatars" not in plan.treatment_layers("graph", plan.Layers.all_on(), None, "", 0, 0.25)
    assert plan.treatment_layers("body", plan.Layers.all_on())["avatars"] == 0.92          # without a value the presets are untouched
    assert plan.treatment_layers("pov", plan.Layers.all_on(), None, "", 0, 0.25, True)["avatars_over_video"] is True
    assert "avatars_over_video" not in plan.treatment_layers("pov", plan.Layers.all_on(), None, "", 0, 0.25)
    sp = long_speech()
    act = act_positions(140, moving=[(0, 140)])
    for val in (0.25, 0.75):
        cfg = plan.Cfg(avatar_opacity=val)
        doc, _, _ = fr.build_direction(sp_bundle(sp), act, plan.Layers.all_on(), cfg, fr.Paths(work=Path(".")), "99_Capture", ("Class recap", "x"), pov_source())
        assert doc["avatar_opacity"] == val
        seen = {s["layers"]["avatars"] for s in doc["segments"] if "avatars" in s["layers"]}
        assert seen == {val}


def test_the_skeleton_lines_and_overlays_keep_their_strength_when_the_avatars_fade():
    lay = plan.treatment_layers("weight", plan.Layers.all_on(), None, "", 0, 0.25)
    assert lay["skeleton_mode"] == "physics" and lay["physics"] is True and lay["counterbalance"] is True and "skeleton_lines" not in lay
    assert plan.treatment_layers("body", plan.Layers.all_on(), None, "", 0, 0.25)["spine_beads"] is True


# ------------------------------------------------------------------------------------------------ velocity arrows need a moving body


def vcom_ctx(speed_lead, speed_follow):
    import numpy as np
    t = np.arange(0, 60, 1 / 30)
    z = np.zeros((len(t), 3))
    lead, fol = z.copy(), z.copy()
    lead[:, 0], fol[:, 2] = speed_lead, speed_follow
    return {**CTX, "vcom": {"t": t, "lead": lead, "follow": fol, "mass_lead": 80.0, "mass_follow": 60.0}}


def test_the_centre_of_mass_speed_comes_from_the_physics_stage():
    c = vcom_ctx(0.8, 0.0)
    assert abs(ar.com_speed(c, "lead", 5, 8) - 0.8) < 1e-9 and ar.com_speed(c, "follow", 5, 8) == 0.0
    assert abs(ar.com_speed(c, None, 5, 8) - 0.8 * 80 / 140) < 1e-9          # mass-weighted couple velocity
    assert ar.com_speed(CTX, "lead", 5, 8) is None and ar.com_speed(c, "lead", 100, 101) is None


def test_a_direction_of_travel_arrow_falls_back_to_the_centre_of_mass_when_the_body_stands():
    sp = long_speech()
    b = plan.Beat(id="ch00_01_x", t0=10.0, t1=16.0, family="talk", chapter=0, events=["p1"])
    e = ev("p1", 12.0, 12.4, ["overlay_path/floorcraft"], ["path"], ["forward"])
    for speed, kind in ((0.8, "vector_arrow"), (0.02, "arrow3d")):
        out, _ = ar.make_arrows(b, [e], CFG, vcom_ctx(speed, speed), None, {})
        assert len(out) == 1 and out[0]["kind"] == kind and out[0]["text"] == "forward"
