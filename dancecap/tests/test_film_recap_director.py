"""dancecap.film_recap_director: director mode of the recap film (phone POV shots circling, feature zooms, closer overhead), synthetic data only.

    .venv-nopo4d/Scripts/python.exe -m pytest dancecap/tests/test_film_recap_director.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_film_recap import act_positions, ev, make_speech, sp_bundle, utt  # noqa: E402

from dancecap import film_recap as fr  # noqa: E402
from dancecap import film_recap_director as fd  # noqa: E402
from dancecap import film_recap_plan as plan  # noqa: E402
from dancecap import recap_pov as rp  # noqa: E402

CFG = plan.Cfg()


def phone(cam, t_end=120.0, h=0.4, pose=True):
    t = np.arange(0, t_end, 1 / 30)
    ok = np.ones(len(t), bool)
    box = np.tile(np.array([0.35, 0.5 - h / 2, 0.65, 0.5 + h / 2]), (len(t), 1))
    grid = np.arange(0, t_end, 0.1)
    return rp.PhoneData(cam, t, ok, box, (1080, 1920), grid, np.full(len(grid), pose))


def pov_source(cams=("00", "01", "02", "03"), bad=None):
    return fd.PovSource({c: phone(c) for c in cams}, bad or {})


def long_speech():
    """100 s of class: talk, a body part, weight, talk, the axis ..."""
    utts, evs, t = [], [], 1.0
    texts = ["so today we talk about the frame and how important it is to keep it", "look at the shoulders and the chest and keep the frame open",
             "now the weight goes back to the left foot and we control the balance", "and then we simply talk for a while about nothing special at all",
             "bring her back to the axis and rotate to the exit", "that is the end of the first part of the lesson thank you everybody"]
    for i, tx in enumerate(texts * 3):
        d = 7.5
        utts.append(utt(f"u{i}", t, t + d, "Kadu" if i % 2 else "Larissa", tx))
        if "shoulders" in tx:
            evs.append(ev(f"b{i}", t + 1.5, t + 1.9, ["camera_focus"], ["body_part"], ["shoulders"], who="follower", where="shoulders"))
            evs.append(ev(f"c{i}", t + 3.5, t + 3.9, ["camera_focus"], ["body_part"], ["chest"], who="follower", where="chest"))
        if "weight" in tx:
            evs.append(ev(f"w{i}", t + 1.0, t + 1.4, ["overlay_weight_change"], ["weight"], ["weight"]))
        if "axis" in tx:
            evs.append(ev(f"a{i}", t + 1.2, t + 1.6, ["overlay_T_axis"], ["axis"], ["axis"]))
        t += d + 0.5
    chapters = [{"start": 0.0, "end": 60.0, "title": "First half", "id": "ch00"}, {"start": 60.0, "end": 140.0, "title": "Second half", "id": "ch01"}]
    return {"utterances": utts, "events": evs, "chapters": chapters, "roles": {"Kadu": "leader", "Larissa": "follower"}, "duration": 140.0}


def direct(sp, pov, act=None):
    act = act or act_positions(140, moving=[(0, 140)])
    layers = plan.Layers.all_on()
    beats, rep = plan.plan_beats(sp["utterances"], sp["events"], sp["chapters"], act, layers, sp["roles"], sp["duration"], CFG)
    inserts, _ = plan.plan_inserts(sp["utterances"], act, sp["duration"], CFG)
    ctx = fd.Ctx(sp["utterances"], sp["events"], sp["chapters"], act, layers, sp["roles"], sp["duration"], CFG, fd.DirCfg(), pov, inserts,
                 fd.cut_points(sp["utterances"], CFG.lead_s), [], [])
    shots, drep = fd.direct(beats, ctx)
    return shots, drep, ctx


def test_cut_points_are_pauses_and_sentence_edges():
    sp = make_speech()
    cuts = fd.cut_points(sp["utterances"], 0.3)
    assert 0.7 in cuts and any(abs(c - 5.05) < 1e-6 for c in cuts)            # sentence start less the lead, sentence end plus 0.05
    assert fd.snap(5.2, cuts, 0.8) == 5.05 and fd.snap(5.4, cuts, 0.1) == 5.4


def test_an_attention_call_is_a_body_part_weight_axis_path_steps_or_timing_event_not_a_demonstration_alone():
    dc = fd.DirCfg()
    assert fd.is_attention(ev("a", 1, 2, ["camera_focus"], ["body_part"], ["hip"], where="hip"), dc)
    assert fd.is_attention(ev("a", 1, 2, ["overlay_T_axis"], ["axis"], ["axis"]), dc)
    assert not fd.is_attention(ev("a", 1, 2, ["graph_moves"], ["move_name"], ["lunge"]), dc)
    assert not fd.is_attention(ev("a", 1, 2, ["camera_focus"], ["body_part"], ["hip"], where="hip", conf=0.4), dc)
    assert fd.is_watch(ev("a", 1, 2, ["camera_focus"], ["demonstration"], ["do this"]))
    assert not fd.is_watch(ev("a", 1, 2, ["camera_focus"], ["body_part"], ["frame"], where="frame"))


def test_features_name_joints_never_hands_and_keep_the_partner_in_the_frame():
    for name, spec in fd.FEATURES.items():
        assert not any("wrist" in j or "hand" in j for j in spec["joints"] if name != "arm")
        assert "hand" not in " ".join(spec["joints"])
    assert fd.feature_of([ev("a", 1, 2, ["camera_focus"], ["body_part"], ["hands"], where="hands")]) == ("frame", "both")
    assert fd.feature_of([ev("a", 1, 2, ["camera_focus"], ["body_part"], ["hip"], who="follower", where="hip")]) == ("hip", "follow")
    cam = fd.feature_camera("chest", "follow", 0, 0.7)
    assert cam["mode"] == "feature_zoom" and cam["partner_joints"] and cam["min_distance_m"] >= 1.5
    assert fd.feature_camera("chest", "follow", 1, 0.7)["azimuth_offset_deg"] == -fd.feature_camera("chest", "follow", 0, 0.7)["azimuth_offset_deg"]


def test_the_phones_carry_a_large_share_and_circle_with_3d_between():
    sp = long_speech()
    shots, rep, _ = direct(sp, pov_source())
    assert 0.34 <= rep["pov_share"] <= 0.52
    fam = [b.family for b in shots]
    # never more than max_pov_run phone shots in a row
    run = best = 0
    for f in fam:
        run = run + 1 if f == "pov" else 0
        best = max(best, run)
    assert best <= fd.DirCfg().max_pov_run
    cams = [b.treatment["phone"] for b in shots if b.family == "pov"]
    assert len(set(cams)) >= 3 and all(cams[i] != cams[i + 1] for i in range(len(cams) - 1))
    # shots tile the class
    assert shots[0].t0 == 0.0 and abs(shots[-1].t1 - sp["duration"]) < 1e-6
    for a, b in zip(shots, shots[1:]):
        assert abs(a.t1 - b.t0) < 1e-6


def test_phone_shots_last_three_to_a_little_over_six_seconds_and_the_film_opens_in_3d():
    shots, _, _ = direct(long_speech(), pov_source())
    pov = [b for b in shots if b.family == "pov"]
    assert pov and all(2.6 <= b.dur <= 7.6 for b in pov)
    assert shots[0].family != "pov"


def test_a_phone_in_its_bad_stretch_is_never_shown_there():
    sp = long_speech()
    bad = {c: [(0.0, 140.0, "x")] for c in ("00", "01", "02")}
    shots, _, _ = direct(sp, pov_source(bad=bad))
    assert {b.treatment["phone"] for b in shots if b.family == "pov"} <= {"03"}
    shots2, rep2, _ = direct(sp, pov_source(cams=("00",), bad={"00": [(0.0, 140.0, "x")]}))
    assert rep2["pov_share"] == 0.0 and all(b.family != "pov" for b in shots2)


def test_every_attention_call_has_a_row_and_a_body_call_gets_a_feature_zoom_or_the_phone():
    sp = long_speech()
    shots, rep, ctx = direct(sp, pov_source())
    rows = fd.attention_table(shots, sp["events"], ctx.dc)
    assert {r["id"] for r in rows} == {e["id"] for e in sp["events"]}
    body = [r for r in rows if r["id"][0] in "bc"]
    assert body and all(("3D zoom on" in r["shown"]) or ("phone" in r["shown"]) for r in body)
    zooms = [b for b in shots if b.camera and b.camera["mode"] == "feature_zoom"]
    assert zooms and all(b.camera["fov_deg"] <= 40 and b.camera["pad_m"] > 0 for b in zooms)
    ov = [b for b in shots if b.family == "axis"]
    assert ov and all(b.camera["mode"] == "overhead_floorcraft" and b.camera["fill"] >= 0.55 and b.camera["min_fill"] >= 0.45 for b in ov)


def test_a_shot_after_a_phone_cuts_in_without_a_blend_and_phone_shots_carry_their_crop():
    shots, _, _ = direct(long_speech(), pov_source())
    for i, b in enumerate(shots):
        if b.family == "pov":
            cam = b.camera
            assert cam["mode"] in ("camera_tour", "fly_to_camera") and cam["crop"]["dt"] == 0.1 and len(cam["crop"]["zoom"]) >= 30
            if cam["mode"] == "camera_tour":
                assert cam["blend_s"] == 0.0 and cam["cameras"][0]["dance_t"] == [round(b.t0, 3), round(b.t1, 3)]
        elif i > 0 and shots[i - 1].family == "pov":
            assert b.camera["blend_s"] == 0.0


def test_build_direction_in_director_mode_is_valid_and_reports_the_phones():
    sp = long_speech()
    act = act_positions(140, moving=[(0, 140)])
    doc, tl, rep = fr.build_direction(sp_bundle(sp), act, plan.Layers.all_on(), CFG, fr.Paths(work=Path(".")), "99_Capture", ("Class recap", "x"), pov_source())
    assert fr.validate(doc, tl, sp_bundle(sp)) == []
    assert doc["recap_film"]["director_mode"] and 0.34 <= doc["recap_film"]["pov_share"] <= 0.52
    assert rep["director"]["attention"] and rep["director"]["pov_seconds_per_phone"]
    povs = [s for s in doc["segments"] if s["camera"]["mode"] in ("camera_tour", "fly_to_camera")]
    assert povs and all(s["layers"]["source_video"] == s["treatment"]["phone"] and s["layers"]["avatars_fade_with_video"] is True for s in povs)


def test_the_old_direction_is_unchanged_without_a_pov_source():
    sp = make_speech()
    act = act_positions(60, moving=[(5, 45)])
    doc, tl, rep = fr.build_direction(sp_bundle(sp), act, plan.Layers.all_on(), CFG, fr.Paths(work=Path(".")), "99_Capture", ("Class recap", "x"))
    assert not doc["recap_film"]["director_mode"] and all("treatment" not in s for s in doc["segments"])


def test_overhead_and_avatar_looks_of_the_recap_film():
    """D2 owner asks: overhead views closer (the couple fills 55-70 % of the width at the tightest, never < 45 %); the dancers are not ghosts"""
    for v in (0, 1):
        o = plan.camera_for("axis", variant=v)
        assert 0.55 <= o["fill"] <= 0.70 and o["min_fill"] >= 0.45
    assert 0.55 <= plan.camera_for("path")["fill"] <= 0.70
    assert plan.L_TALK["avatars"] >= 0.9 and plan.L_BODY["avatars"] >= 0.85 and plan.L_POV["avatars_fade_with_video"] is True
    assert plan.camera_for("talk")["avoid_inline"] is True


def test_a_call_without_a_motion_family_or_a_phone_still_gets_a_deliberate_zoom():
    """no phones at all and standing teachers: the weight call is the physics view, the body part a feature zoom, never a plain orbit"""
    sp = long_speech()
    shots, rep, ctx = direct(sp, None, act=act_positions(140, still=True))
    rows = fd.attention_table(shots, sp["events"], ctx.dc)
    assert rows and all(r["film_family"] != "talk" for r in rows if r["id"][0] in "bcw")
    assert any("3D zoom on" in r["shown"] for r in rows if r["id"][0] in "bc") and any("physics" in r["shown"] for r in rows if r["id"][0] == "w")
    assert rep["pov_share"] == 0.0
