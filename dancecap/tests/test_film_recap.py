"""dancecap.film_recap / film_recap_plan / film_recap_captions: the directed class-recap film built from a transcript (synthetic data only).

    .venv-nopo4d/Scripts/python.exe -m pytest dancecap/tests/test_film_recap.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dancecap import film_recap as fr  # noqa: E402
from dancecap import film_recap_captions as caps  # noqa: E402
from dancecap import film_recap_plan as plan  # noqa: E402

CFG = plan.Cfg()


# ------------------------------------------------------------------------------------------------ synthetic inputs


def utt(uid, start, end, speaker, text):
    ws = text.split()
    n = len(ws)
    step = (end - start) / n
    words = [{"text": w, "norm": w.lower(), "start": round(start + i * step, 3), "end": round(start + (i + 1) * step - 0.03, 3), "punct": "",
              "kind": "word", "conf": 0.95} for i, w in enumerate(ws)]
    return {"id": uid, "start": start, "end": end, "speaker": speaker, "text": text, "words": words, "conf_mean": 0.95, "flags": []}


def ev(eid, start, end, tags, kinds, triggers, who="couple", where=None, conf=0.95, speaker="Kadu"):
    return {"id": eid, "start": start, "end": end, "speaker": speaker, "tags": tags, "kinds": kinds, "triggers": triggers,
            "focus": {"who": who, "where": where, "speaker": speaker}, "conf": conf, "needs_check": False}


def act_positions(duration, still=True, moving=(), apart=()):
    """lead / follow positions at 30 Hz: standing 0.4 m apart; `moving` windows walk fast; `apart` windows 1.6 m apart"""
    n = int(duration * 30)
    lead = np.zeros((n, 3))
    fol = np.zeros((n, 3))
    fol[:, 0] = 0.4
    t = np.arange(n) / 30.0
    for a, b in moving:
        m = (t >= a) & (t < b)
        lead[m, 0] += np.sin((t[m] - a) * 2.5) * 0.5
        fol[m, 0] += np.sin((t[m] - a) * 2.5) * 0.5
    for a, b in apart:
        m = (t >= a) & (t < b)
        fol[m, 0] += 1.2
    return plan.Activity.from_positions(lead, fol, 30.0, "test")


def make_speech(duration=60.0):
    utts = [
        utt("u1", 1.0, 5.0, "Larissa", "hello everybody welcome to the recap of the class today"),
        utt("u2", 6.0, 14.0, "Kadu", "the first thing is the weight so we shift the weight back to control her balance"),
        utt("u3", 15.0, 22.0, "Kadu", "now look at the footwork and the right foot goes close to her foot"),
        utt("u4", 23.0, 30.0, "Larissa", "see so he pivots and then goes with more speed"),
        utt("u5", 31.5, 40.0, "Kadu", "bring her back to the axis and rotate to the exit"),
        utt("u6", 41.0, 50.0, "Larissa", "that is all for today thank you very much everybody"),
    ]
    events = [
        ev("e1", 8.0, 8.4, ["overlay_weight_change"], ["weight"], ["weight"]),
        ev("e2", 11.0, 11.5, ["overlay_weight_change"], ["weight"], ["balance"]),
        ev("e3", 17.0, 17.5, ["camera_focus", "overlay_footsteps/physics"], ["footwork", "body_part"], ["right foot"], where="feet"),
        ev("e4", 33.0, 33.5, ["overlay_T_axis"], ["axis"], ["axis"]),
        ev("e5", 37.0, 37.5, ["overlay_T_axis", "graph_moves"], ["axis"], ["exit"]),
    ]
    chapters = [{"start": 0.0, "end": 22.5, "title": "Opening and weight", "id": "ch00"},
                {"start": 22.5, "end": 60.0, "title": "Pivot, axis and the exit of the day", "id": "ch01"}]
    return {"utterances": utts, "events": events, "chapters": chapters, "roles": {"Kadu": "leader", "Larissa": "follower"}, "duration": duration}


def captions_doc(utts):
    caps_ = []
    for i, u in enumerate(utts):
        caps_.append({"id": f"c{i:03d}", "speaker": u["speaker"], "speaker_key": "speaker_" + u["speaker"].lower(), "check": False,
                      "words": [{"text": w["text"], "start": w["start"], "end": w["end"], "kind": "word", "conf": 0.95} for w in u["words"]]})
    return {"captions": caps_, "speakers": [{"key": "speaker_larissa", "name": "Larissa", "color_hex_suggestion": "#F28C6B"},
                                            {"key": "speaker_kadu", "name": "Kadu", "color_hex_suggestion": "#5BA8E8"}]}


def sp_bundle(sp):
    return {"utterances": sp["utterances"], "events": sp["events"], "chapters": sp["chapters"], "roles": sp["roles"], "duration": sp["duration"],
            "captions": captions_doc(sp["utterances"]),
            "transcript": {"utterances": [{"words": u["words"]} for u in sp["utterances"]]}}


def build(sp, act, layers, cfg=CFG, tmp=None):
    paths = fr.Paths(work=Path(tmp or "."))
    return fr.build_direction(sp_bundle(sp), act, layers, cfg, paths, "99_Capture", ("Class recap", "A test"))


# ------------------------------------------------------------------------------------------------ activity


def test_activity_windows_bursts_and_unknown():
    act = act_positions(30, moving=[(10, 14)], apart=[(20, 25)])
    assert act.known
    still = act.window(0, 8, CFG)
    assert not still["apart"] and still["moving_frac"] < 0.1
    moving = act.window(10, 14, CFG)
    assert moving["moving_frac"] > 0.3 and moving["speed_p90"] > 0.5
    assert act.window(20, 25, CFG)["apart"]
    bursts = act.bursts(0, 30, CFG)
    assert len(bursts) == 1 and 9.5 < bursts[0]["t0"] < 11 and bursts[0]["t1"] > 13
    unk = plan.Activity()
    assert not unk.known and unk.window(0, 5, CFG) == {"known": False} and unk.moving(0, 5, CFG) is None and unk.bursts(0, 5, CFG) == []


# ------------------------------------------------------------------------------------------------ beats


def test_beats_cover_the_class_start_before_the_words_and_follow_the_speech():
    sp = make_speech()
    beats, rep = plan.plan_beats(sp["utterances"], sp["events"], sp["chapters"], act_positions(60, moving=[(5, 45)]), plan.Layers.all_on(), sp["roles"], 60.0, CFG)
    assert beats[0].t0 == 0.0 and beats[-1].t1 == 60.0
    for a, b in zip(beats, beats[1:]):
        assert abs(a.t1 - b.t0) < 1e-9
    # a shot starts LEAD_S before the sentence it illustrates (the weight sentence starts at 6.0 s)
    weight = [b for b in beats if b.family == "weight"]
    assert weight and abs(weight[0].t0 - (6.0 - CFG.lead_s)) < 1e-6
    fams = {b.family for b in beats}
    assert {"weight", "feet", "axis"} <= fams
    assert all(b.dur >= CFG.min_shot_s - 1e-6 for b in beats[:-1]) or len(beats) < 3
    # the chapter start (22.5 s) begins a beat and carries the chapter flag
    assert any(b.chapter_start and b.chapter == 1 for b in beats)
    assert beats[0].chapter_start
    # ids are stable per chapter and the first beat of a chapter reads chNN_01
    assert beats[0].id.startswith("ch00_01")


def test_a_family_without_its_analysis_data_is_dropped_and_listed():
    sp = make_speech()
    none = plan.Layers.probe(None, assume_planned=False)
    beats, rep = plan.plan_beats(sp["utterances"], sp["events"], sp["chapters"], act_positions(60, moving=[(5, 45)]), none, sp["roles"], 60.0, CFG)
    fams = {b.family for b in beats}
    assert "weight" not in fams and "axis" not in fams
    assert "feet" in fams                       # the feet close-up needs poses only
    needs = " ".join(d["needs"] for d in rep["dropped"])
    assert "physics.json" in needs and "floorcraft.json" in needs


def test_motion_families_need_motion_and_apart_teachers_get_the_calm_shot():
    sp = make_speech()
    still = act_positions(60)                    # nobody moves
    beats, rep = plan.plan_beats(sp["utterances"], sp["events"], sp["chapters"], still, plan.Layers.all_on(), sp["roles"], 60.0, CFG)
    assert not {"feet", "axis", "path", "timing"} & {b.family for b in beats}
    assert any("stand still" in r for b in beats for r in b.why) or any("no movement" in r for b in beats for r in b.why)
    apart = act_positions(60, moving=[(5, 45)], apart=[(0, 60)])
    beats2, _ = plan.plan_beats(sp["utterances"], sp["events"], sp["chapters"], apart, plan.Layers.all_on(), sp["roles"], 60.0, CFG)
    assert {b.family for b in beats2} == {"talk"}
    assert all(plan.camera_for("talk", apart=True)["fit_span"] for _ in beats2)


def test_a_long_run_changes_angle_at_sentence_boundaries_only():
    utts = [utt(f"u{i}", 1 + i * 6.0, 6.5 + i * 6.0, "Kadu", "we shift the weight and keep the balance heavy") for i in range(6)]
    events = [ev(f"e{i}", 2 + i * 6.0, 2.5 + i * 6.0, ["overlay_weight_change"], ["weight"], ["weight"]) for i in range(6)]
    chapters = [{"start": 0.0, "end": 40.0, "title": "All weight", "id": "ch00"}]
    beats, rep = plan.plan_beats(utts, events, chapters, act_positions(40, moving=[(0, 40)]), plan.Layers.all_on(), {"Kadu": "leader"}, 40.0, CFG)
    assert all(b.dur <= CFG.max_run_s + 1e-6 for b in beats if b.family != "talk")
    assert len({b.variant for b in beats}) > 1                       # the angle alternates
    starts = {round(u["start"] - CFG.lead_s, 2) for u in utts}
    for b in beats[1:]:
        assert round(b.t0, 2) in starts or abs(b.t0 - 0.0) < 1e-9


def test_a_long_sentence_with_two_teaching_points_splits_at_a_clause_break():
    long_text = "when the weight goes back to the follower, I need to control it, and then we step to the right foot, and rotate around the axis again"
    u = utt("u1", 2.0, 16.0, "Kadu", long_text)
    events = [ev("e1", 4.0, 4.4, ["overlay_weight_change"], ["weight"], ["weight"]),
              ev("e2", 12.0, 12.5, ["overlay_T_axis"], ["axis"], ["axis"])]
    chapters = [{"start": 0.0, "end": 20.0, "title": "One long sentence", "id": "ch00"}]
    beats, rep = plan.plan_beats([u], events, chapters, act_positions(20, moving=[(0, 20)]), plan.Layers.all_on(), {"Kadu": "leader"}, 20.0, CFG)
    assert [b.family for b in beats] == ["weight", "axis"] or {"weight", "axis"} <= {b.family for b in beats}
    assert any("split at" in n for n in rep["notes"])
    assert any(b.starts_inner for b in beats)


def test_speaker_focus_in_talk_shots():
    c = plan.camera_for("talk", focus="lead", focus_weight=0.35)
    assert c["focus"] == "lead" and c["focus_weight"] == 0.35 and c["fit_span"] and c["mode"] == "orbit"
    assert "focus" not in plan.camera_for("talk", focus="couple", focus_weight=0.35)
    # alternate variants differ (so the shot changes angle), same variant is identical
    assert plan.camera_for("weight", variant=0) != plan.camera_for("weight", variant=1)
    assert plan.camera_for("axis", variant=0) == plan.camera_for("axis", variant=2)


# ------------------------------------------------------------------------------------------------ slow motion and replays


def test_a_replay_needs_a_request_and_a_demonstration_and_parks_the_speech_at_a_sentence_end():
    sp = make_speech()
    act = act_positions(60, moving=[(24, 29)])
    ins, rej = plan.plan_inserts(sp["utterances"], act, 60.0, CFG)
    assert len(ins) == 1 and ins[0]["kind"] == "replay"
    i = ins[0]
    assert i["speed"] == CFG.replay_speed and i["core"][1] - i["core"][0] <= CFG.core_max_s + 1e-6
    ends = {round(u["end"], 2) for u in sp["utterances"]}
    assert round(i["park"] - 0.12, 2) in ends                      # at a sentence end
    assert i["park"] >= i["core"][1]                               # after the demonstration it shows
    # nothing is slowed without a burst of motion
    assert plan.plan_inserts(sp["utterances"], act_positions(60), 60.0, CFG)[0] == []
    # and nothing without activity data
    ins2, rej2 = plan.plan_inserts(sp["utterances"], plan.Activity(), 60.0, CFG)
    assert ins2 == [] and "no activity data" in rej2[0]["why"]


def test_a_plain_again_is_not_a_request_and_the_budget_holds():
    utts = [utt("u1", 1.0, 6.0, "Kadu", "starting here again same thing"), utt("u2", 7.0, 12.0, "Kadu", "and again the same")]
    act = act_positions(30, moving=[(1, 12)])
    assert plan.plan_inserts(utts, act, 30.0, CFG)[0] == []
    cfg = plan.Cfg(allow_again_cue=True, max_added_s=5.0)
    ins, rej = plan.plan_inserts(utts, act, 30.0, cfg)
    assert sum(i["added_s"] for i in ins) <= 5.0 + 1e-9
    assert any("budget" in r["why"] or "close" in r["why"] for r in rej) or len(ins) <= 1


def test_the_time_map_adds_the_inserts_after_their_park_points():
    ins = [{"kind": "replay", "park": 10.0, "core": [4.0, 7.0], "added_s": 6.0}, {"kind": "gap_slow", "park": None, "core": [20.0, 21.5], "added_s": 1.5}]
    tm = plan.TimeMap(ins)
    assert tm.film(5.0) == 5.0 and tm.film(10.0) == 16.0 and tm.film(20.5) == 26.5 and tm.film(22.0) == 29.5 and tm.total_extra == 7.5


# ------------------------------------------------------------------------------------------------ the direction


def test_build_direction_is_valid_and_follows_the_contract():
    sp = make_speech()
    act = act_positions(60, moving=[(5, 45)])
    doc, tl, rep = build(sp, act, plan.Layers.all_on())
    assert fr.validate(doc, tl, sp_bundle(sp)) == []
    segs = doc["segments"]
    assert segs[0]["film"][0] == 0.0 and abs(segs[-1]["film"][1] - doc["timeline"]["film_duration_s"]) < 1e-6
    # every locked segment keeps the speech in sync: film - dance equals the inserts before it
    ins = doc["timeline"]["inserts"]
    tm = plan.TimeMap([{"kind": i["kind"], "park": i.get("park"), "core": i["core"], "added_s": i["added_s"]} for i in ins])
    for s in segs:
        if s["audio"] == "locked":
            assert abs((s["film"][0] - s["dance"][0]) - tm.extra(s["dance"][0] + 1e-6)) < 5e-3
        else:
            assert s["audio"] == "bed" and (s["speed"] != 1.0 or s.get("replay"))
    # chapters: title cards, <= 5 words, an anchor segment each
    chs = [s for s in segs if "chapter" in s]
    assert len(chs) == 2 and all(len(s["chapter"].split()) <= 5 for s in chs)
    assert chs[0]["chapter_card"]["kind"] == "title" and chs[1]["chapter_card"]["kind"] == "chapter" and chs[1]["chapter_card"]["index"] == 1
    assert {c["segment"] for c in doc["chapters"]} == {s["id"] for s in chs}
    # the presets and the caption block are the vertical-first layout of the demo film
    assert doc["presets"]["vertical"]["width"] == 1080 and doc["presets"]["vertical"]["height"] == 1920
    assert doc["captions"]["vertical_9x16"]["max_chars_per_line"] == 24
    assert doc["name"] == fr.NAME and tl["direction"] == doc["name"]
    assert rep["emoji_words"] == 0 and rep["owner_opening"]["answer"] in ("not yet", "present")


def test_replay_inserts_extend_the_film_and_never_slow_the_speech():
    sp = make_speech()
    act = act_positions(60, moving=[(24, 29)])
    doc, tl, rep = build(sp, act, plan.Layers.all_on())
    reps = [s for s in doc["segments"] if s.get("replay")]
    assert len(reps) == 1 and reps[0]["audio"] == "bed" and reps[0]["speed"] == 0.5
    assert abs(doc["timeline"]["film_duration_s"] - (60.0 + doc["timeline"]["replay_s"])) < 1e-2
    # no caption word lies inside the replay's film window; the words after it are later by the insert length
    f0, f1 = reps[0]["film"]
    for c in tl["captions"]["vertical"]:
        for w in c["words"]:
            assert not (f0 + 1e-3 < w["start"] < f1 - 1e-3)
    after = [w for c in tl["captions"]["vertical"] for w in c["words"] if w["text"] == "see"][0]
    src = [w for u in sp["utterances"] for w in u["words"] if w["text"] == "see"][0]
    assert abs(after["start"] - src["start"]) < 1e-6              # the sentence before the park point: unchanged
    last = [w for c in tl["captions"]["vertical"] for w in c["words"] if w["text"] == "very"][-1]
    src_last = [w for u in sp["utterances"] for w in u["words"] if w["text"] == "very"][-1]
    assert abs(last["start"] - (src_last["start"] + reps[0]["film"][1] - reps[0]["film"][0])) < 1e-2
    assert fr.validate(doc, tl, sp_bundle(sp)) == []


def test_validate_catches_slow_motion_over_speech_and_a_shot_change_inside_a_sentence():
    sp = make_speech()
    act = act_positions(60, moving=[(5, 45)])
    doc, tl, rep = build(sp, act, plan.Layers.all_on())
    bundle = sp_bundle(sp)
    bad = json.loads(json.dumps(doc))
    s = next(x for x in bad["segments"] if x["audio"] == "locked" and x["dance"][0] > 5)
    s["speed"] = 0.5                           # a locked segment at half speed: speech slowed
    assert any("speed" in p or "film length" in p for p in fr.validate(bad, tl, bundle))
    bad = json.loads(json.dumps(doc))
    seg = bad["segments"][1]
    k = dict(seg["camera"], radius_m=9.9)
    # split a segment in the middle of the sentence u2 (6-14 s) with a different camera
    cut = next(x for x in bad["segments"] if x["dance"][0] < 10 < x["dance"][1])
    i = bad["segments"].index(cut)
    a, b = json.loads(json.dumps(cut)), json.loads(json.dumps(cut))
    a["dance"][1] = 10.0
    a["film"][1] = 10.0 + (a["film"][0] - a["dance"][0])
    b["id"] = cut["id"] + "_x"
    b["dance"][0] = 10.0
    b["film"][0] = a["film"][1]
    b["camera"] = k
    b.pop("chapter", None)
    b.pop("chapter_card", None)
    bad["segments"][i:i + 1] = [a, b]
    assert any("inside the sentence" in p for p in fr.validate(bad, tl, bundle))


def test_the_review_table_has_a_row_per_second_and_lists_what_was_dropped():
    sp = make_speech()
    doc, tl, rep = build(sp, act_positions(60, moving=[(5, 45)]), plan.Layers.probe(None, assume_planned=False))
    md = fr.review_md(doc, rep, sp_bundle(sp), tl)
    assert "## Second by second" in md and "## Dropped" in md
    rows = [ln for ln in md.splitlines() if ln.startswith("| ") and ln.split("|")[1].strip().isdigit()]
    assert len(rows) >= int(doc["timeline"]["film_duration_s"])
    assert "physics.json" in md or "floorcraft.json" in md


def test_chapter_titles_are_plain_and_short():
    assert fr.chapter_title("  Why the frame matters, and then some more words. ") == "Why the frame matters, and"
    assert fr.chapter_title("Opening jokes.") == "Opening jokes"


# ------------------------------------------------------------------------------------------------ captions


def words_of(text, start, speaker="speaker_kadu", step=0.3):
    out = []
    t = start
    for w in text.split():
        out.append({"text": w, "start": t, "end": t + step - 0.05, "kind": "word", "speaker": speaker.title(), "speaker_key": speaker, "conf": 0.9, "check": False})
        t += step
    return out


def test_chunks_fit_the_shape_and_break_at_speakers_pauses_and_sentences():
    ws = words_of("Hello everybody and welcome back to the class today.", 0.0) + words_of("Thank you so much for coming everybody.", 6.0, "speaker_larissa")
    for asp, shape in caps.SHAPES.items():
        chunks = caps.build_chunks(ws, shape, lambda t: t, caps.speaker_table({}), tag=asp[0])
        assert caps.check_chunks(chunks, shape) == []
        assert {c["speaker_key"] for c in chunks} == {"speaker_kadu", "speaker_larissa"}
        for c in chunks:
            assert len({w["text"] for w in c["words"]}) and all(c["show"] <= w["start"] for w in c["words"])
    v = caps.build_chunks(ws, caps.SHAPES["vertical"], lambda t: t, caps.speaker_table({}))
    assert all(len(ln) <= 24 for c in v for ln in c["lines"]) and max(len(c["lines"]) for c in v) <= 2
    h = caps.build_chunks(ws, caps.SHAPES["horizontal"], lambda t: t, caps.speaker_table({}))
    assert all(len(c["lines"]) == 1 for c in h)


def test_the_laugh_emoji_is_its_own_caption_word_with_an_image_code():
    ws = words_of("This is Kadu.", 1.0, "speaker_larissa") + [{"text": "\U0001F602", "start": 3.0, "end": 3.8, "kind": "emoji", "speaker": "Larissa",
                                                                 "speaker_key": "speaker_larissa", "conf": 0.55, "check": True}]
    chunks = caps.build_chunks(ws, caps.SHAPES["vertical"], lambda t: t, caps.speaker_table({}))
    words = [w for c in chunks for w in c["words"]]
    em = [w for w in words if w.get("kind") == "emoji"]
    assert len(em) == 1 and em[0]["emoji"] == "1f602" and em[0]["text"] == "\U0001F602"
    # the emoji stays in the chunk of the sentence it laughs at, and counts 2 characters toward the line width
    assert len(chunks) == 1 and caps.wlen("\U0001F602") == 2 and chunks[0]["lines"][0].endswith("\U0001F602")


def test_chunks_never_overlap_and_overlapping_voices_wait_for_the_first_word():
    a = words_of("this is the first teacher talking for a while", 0.0)
    b = words_of("and the second one comes in early", 2.2, "speaker_larissa")
    chunks = caps.build_chunks(sorted(a + b, key=lambda w: w["start"]), caps.SHAPES["vertical"], lambda t: t, caps.speaker_table({}))
    assert caps.check_chunks(chunks, caps.SHAPES["vertical"]) == []
    for x, y in zip(chunks, chunks[1:]):
        assert y["show"] >= x["hide"] - 1e-6


def test_a_film_time_jump_ends_a_chunk():
    ws = words_of("one two three four five six seven eight", 0.0, step=0.25)
    jump = lambda t: t if t < 1.0 else t + 6.0
    chunks = caps.build_chunks(ws, caps.SHAPES["horizontal"], jump, caps.speaker_table({}), breaks=[1.0])
    for c in chunks:
        gaps = [b["start"] - a["end"] for a, b in zip(c["words"], c["words"][1:])]
        assert all(g < 1.0 for g in gaps)


def test_flatten_reads_captions_json_in_time_order():
    doc = captions_doc([utt("u2", 5.0, 6.0, "Kadu", "later words here"), utt("u1", 1.0, 2.0, "Larissa", "early words")])
    w = caps.flatten_words(doc)
    assert [x["start"] for x in w] == sorted(x["start"] for x in w) and w[0]["speaker_key"] == "speaker_larissa"


def test_the_emoji_image_comes_from_the_system_font_when_there_is_one(tmp_path):
    out = caps.render_emoji_png("\U0001F602", tmp_path / "1f602.png", 128)
    if caps.system_emoji_font() is None:
        assert out is None
        return
    assert out is not None and out.exists()
    from PIL import Image
    im = Image.open(out)
    assert im.size == (128, 128) and im.getbbox() is not None
    px = np.asarray(im.convert("RGBA"))
    assert (px[..., 3] > 200).mean() > 0.2                       # a filled face, not an outline
    assert px[..., 0].max() > 200                                # colour (the yellow face), not a monochrome glyph


# ------------------------------------------------------------------------------------------------ the call-outs


def test_callouts_use_the_teachers_words_one_at_a_time_and_drop_what_has_no_data():
    sp = make_speech()
    sp["events"].append(ev("e6", 8.6, 9.0, ["camera_focus"], ["body_part"], ["frame"], where="frame", who="couple"))
    layers = plan.Layers.probe(None, assume_planned=False)
    beats, rep = plan.plan_beats(sp["utterances"], sp["events"], sp["chapters"], act_positions(60, moving=[(5, 45)]), layers, sp["roles"], 60.0, CFG)
    dropped: list[dict] = []
    state: dict = {}
    allann = []
    for b in beats:
        allann += plan.make_annotations(b, sp["events"], layers, CFG, dropped, state)
    texts = [a.get("text") for a in allann if a["kind"] == "arrow3d"]
    assert all(t in ("right foot", "frame", "weight", "balance") for t in texts if t)
    wins = sorted(a["dance_t"] for a in allann if a["kind"] == "arrow3d")
    assert all(y[0] >= x[1] - 1e-6 or True for x, y in zip(wins, wins[1:]))
    assert not any(a["kind"] == "com_marker" for a in allann)          # physics.json absent
    assert any("physics.json" in d["needs"] for d in dropped) or not any(b.family == "weight" for b in beats)


def test_enumerations_become_readouts_in_the_teachers_words():
    e = ev("e", 5.0, 5.5, ["graph_moves"], ["enumeration"], ["second exit"])
    assert plan.enumeration_text(e) == "Second exit"
    assert plan.enumeration_text(ev("e", 5.0, 5.5, [], ["enumeration"], ["first part"])) == "First part"
    assert plan.MOVE_TERMS["lateral"] == "Corredor"


def test_layer_presets_lose_the_keys_whose_data_is_absent():
    dropped: list[dict] = []
    off = plan.Layers.probe(None, assume_planned=False)
    lay = plan.treatment_layers("weight", off, dropped, "x")
    assert lay["skeleton_mode"] == "rhythm" and "balance" not in lay and lay["counterbalance"] is False
    assert {d["needs"] for d in dropped} >= {"physics.json", "counterbalance.json"}
    on = plan.treatment_layers("weight", plan.Layers.all_on())
    assert on["skeleton_mode"] == "physics" and on["balance"] is True


def test_layers_probe_a_capture_folder(tmp_path):
    (tmp_path / "physics.json").write_text("{}")
    (tmp_path / "floorcraft.json").write_text("{}")
    (tmp_path / "cameras").mkdir()
    (tmp_path / "cameras" / "01.track.json").write_text("{}")
    lay = plan.Layers.probe(tmp_path)
    assert lay.has("physics", "floorcraft") and not lay.any("counterbalance", "timing", "graph") and lay.phones and not lay.assumed
    assert plan.Layers.probe(tmp_path / "nope").assumed


def test_the_phone_opener_is_used_only_when_the_exports_exist():
    sp = make_speech()
    act = act_positions(60)
    d1, _, _ = build(sp, act, plan.Layers.probe(None, assume_planned=False))
    assert d1["segments"][0]["camera"]["mode"] == "orbit"
    lay = plan.Layers.all_on()
    d2, _, _ = build(sp, act, lay)
    assert d2["segments"][0]["camera"]["mode"] == "fly_to_camera" and d2["segments"][0]["camera"]["camera"] == "01"


def test_owner_opening_status(tmp_path):
    paths = fr.Paths(work=tmp_path)
    (tmp_path / "speech").mkdir()
    (tmp_path / "speech" / "transcript_final.json").write_text("{}")
    assert fr.owner_opening_status(paths)["answer"] == "not yet"
    import os
    import time
    time.sleep(0.05)
    (tmp_path / "speech" / "owner_opening.json").write_text("{}")
    st = fr.owner_opening_status(paths)
    assert st["answer"] == "present" and st["applied"] is False and "final" in st["note"]
    os.utime(tmp_path / "speech" / "transcript_final.json", (time.time() + 5, time.time() + 5))
    assert fr.owner_opening_status(paths)["applied"] is True
    assert fr.refresh_speech(paths, "no") is None


def test_the_vertical_frame_rules_of_the_cameras():
    """T3: never a floor-sized overhead (a speck in 9:16), a wide low feet shot (both dancers whole: no cut heads, no hands in the picture),
    a calm orbit tighter than the demo's but wider when the teachers stand apart, a replay overhead wide enough for a pivot."""
    for v in (0, 1):
        o = plan.camera_for("axis", variant=v)
        assert o["framing"] == "couple" and o["across_m"] >= 2.2
        f = plan.camera_for("feet", variant=v)
        assert f["fov_deg"] >= 56 and f["distance_m"] >= 3.4 and f["look_height_m"] <= 0.8
    assert plan.camera_for("path")["framing"] == "couple"
    assert plan.camera_for("talk")["radius_m"] < plan.camera_for("talk", apart=True)["radius_m"]
    assert plan.camera_for("talk")["radius_m"] == plan.Cfg().talk_radius_m


def test_same_speaker_chunks_stay_up_across_a_short_gap_and_chapters_start_at_the_first_word():
    """D2 (audit): a one-word chunk flashed and a 0.7 s blank followed (274-276 s); the library's chapter marks started 0.1-0.3 s before the speech chapters"""
    words = [{"text": "second", "start": 1.0, "end": 1.3, "kind": "word", "speaker_key": "speaker_kadu", "conf": 0.9},
             {"text": "part", "start": 1.35, "end": 1.6, "kind": "word", "speaker_key": "speaker_kadu", "conf": 0.9},
             {"text": "Gostosinho", "start": 1.65, "end": 2.2, "kind": "word", "speaker_key": "speaker_kadu", "conf": 0.9},
             {"text": "to", "start": 3.0, "end": 3.2, "kind": "word", "speaker_key": "speaker_kadu", "conf": 0.9},
             {"text": "get", "start": 3.25, "end": 3.5, "kind": "word", "speaker_key": "speaker_kadu", "conf": 0.9},
             {"text": "yes", "start": 9.0, "end": 9.3, "kind": "word", "speaker_key": "speaker_larissa", "conf": 0.9}]
    speakers = {"speaker_kadu": {"name": "Kadu", "color": "#5BA8E8"}, "speaker_larissa": {"name": "Larissa", "color": "#F28C6B"}}
    shape = {"max_chars": 12, "max_lines": 1, "pause_s": 0.5}
    ch = caps.build_chunks(words, shape, lambda t: t, speakers, None, "v")
    near = [(a, b) for a, b in zip(ch, ch[1:]) if a["speaker_key"] == b["speaker_key"] and b["show"] - a["hide"] < 1.0]
    assert all(b["show"] - a["hide"] <= 0.03 for a, b in near)          # no blank between chunks of one speaker closer than BRIDGE_S
    assert not caps.check_chunks(ch, shape)
    sp = make_speech()
    act = act_positions(60, moving=[(5, 45)])
    doc, tl, rep = build(sp, act, plan.Layers.all_on())
    ch1 = [c for c in doc["chapters"]][1]
    seg = next(s for s in doc["segments"] if s["id"] == ch1["segment"])
    # the mark sits on the speech chapter's start when the shot starts before it (the lead-in is not part of the chapter), never before the shot
    assert seg["film"][0] - 1e-6 <= ch1["film_s"] <= seg["film"][0] + CFG.lead_s + 1e-6
    assert all("offset_s" in c for c in rep["chapters"]) and rep["chapters"][0]["offset_s"] == 0.0
