"""dancecap.speech final transcript (final_merge.py, final_exports.py, final_direct.py, final_eval.py, final_review.py) with SYNTHETIC data only.
Offline, no GPU, no audio, no real transcripts.

    ..\\.venv-nopo4d\\Scripts\\python -m pytest tests/test_speech_final.py -q
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dancecap.speech import final_direct as FD, final_eval as FEV, final_exports as FX, final_merge as FM, final_review as FR  # noqa: E402

LAUGH = "\U0001F602"
TEXT = "so we start with the counterbalance and then we turn to the left".split()


def cw(i, text, start, punct="", share=0.99, slot=None, norm=None):
    return {"text": text, "norm": norm or text.lower(), "start": start, "end": start + 0.3, "punct": punct, "share": share, "margin": 0.5, "null_share": 0.0, "tally": [], "hyps": 10,
            "phones": 5, "engines": 4, "eligible_phones": 6, "t_spread": 0.0, "filler": text in ("um", "uh"), "flags": [], "slot": i + 100 if slot is None else slot, "i": i,
            "speaker_edge": False, "speaker_conf": 0.85, "voice_conf": 1.0, "interjection": False}


def mk_consensus(words=TEXT, split=6, t0=10.0, dt=0.4, puncts=None):
    puncts = puncts or {}
    ws = [cw(i, w, t0 + i * dt, puncts.get(i, "")) for i, w in enumerate(words)]
    turns = [
        {"id": "t000", "voice": 1, "speaker": "Larissa", "words": ws[:split]},
        {"id": "t001", "voice": 0, "speaker": "Kadu", "words": ws[split:]},
    ]
    return {"turns": turns, "naming": {"naming_confidence": 0.85}, "unplaced_sources": {}}


def span(sid, i0, i1, text_words, t0=10.0, dt=0.4, extra=None):
    words = [(i, w) for i, w in enumerate(text_words)]
    s = {"id": sid, "slots": [100 + i0, 100 + i1], "start": t0 + i0 * dt, "end": t0 + i1 * dt + 0.3, "consensus_text": " ".join(text_words[i0: i1 + 1]), "listen_to": None}
    s.update(extra or {})
    return s


def dec(sid, text, confidence=0.8, decision="choose_candidate", flags=(), needs_owner=False):
    return {"id": sid, "text": text, "confidence": confidence, "decision": decision, "flags": list(flags), "needs_owner": needs_owner, "reason": "test", "candidate": None,
            "alternatives": []}


NAMES = {"larissa", "kadu"}


def build(consensus, spans, decisions, overrides=None, raw=None):
    return FM.build_final(consensus, {s["id"]: s for s in spans}, {d["id"]: d for d in decisions}, raw, NAMES, overrides or {})


def spoken(final):
    return [w for u in final["utterances"] for w in u["words"] if w["kind"] != "event"]


# ------------------------------------------------------------------------------------------------ decision text


def test_parse_text_tokens_names_and_events():
    toks = FM.parse_text("Hello! What's up? I don't know who this is Kadu \U0001F602", NAMES)
    assert [t["norm"] for t in toks if t["kind"] != "event"] == ["hello", "whats", "up", "i", "dont", "know", "who", "this", "is", "kadu"]
    assert toks[0]["punct"] == "!" and toks[2]["punct"] == "?" and toks[0]["text"] == "hello"        # lower case, punctuation split off
    assert [t["text"] for t in toks if t["norm"] in ("i", "kadu")] == ["I", "Kadu"]                    # names and I keep their capital
    assert toks[-1]["kind"] == "event" and toks[-1]["text"] == LAUGH


def test_parse_text_expansions_fillers_and_proper_nouns():
    toks = FM.parse_text("Um, we're gonna go to Gostosinho", NAMES)
    assert toks[0]["kind"] == "filler" and toks[0]["norm"] == "um" and toks[0]["punct"] == ","
    assert [t["norm"] for t in toks][2:5] == ["going", "to", "go"]                                      # gonna = going to, as in the consensus
    assert toks[-1]["text"] == "Gostosinho"                                                              # a capital inside a sentence is a proper noun


# ------------------------------------------------------------------------------------------------ merging a span


def test_merge_span_keeps_equal_words_replaces_and_inserts():
    c = mk_consensus()
    words = FM.flatten_consensus(c)
    # span of 'counterbalance and then' -> 'counter balance exit and then now'
    sp = span("d0001", 5, 7, TEXT)
    idx = [5, 6, 7]
    seq, rec = FM.merge_span(words, idx, sp, dec("d0001", "exit and then now", 0.7, flags=["term", "unclear"]), NAMES)
    assert [w["text"] for w in seq] == ["exit", "and", "then", "now"]
    kept = [w for w in seq if w["edit"] == "kept"]
    assert [w["text"] for w in kept] == ["and", "then"] and kept[0]["start"] == words[6]["start"]      # kept words keep their consensus time
    new = [w for w in seq if w["edit"] in ("changed", "inserted")]
    assert [w["text"] for w in new] == ["exit", "now"] and all(w["start"] is None for w in new)
    assert rec["removed"] == ["counterbalance"] and rec["added"] == ["exit", "now"]
    assert all("unclear" in w["flags"] for w in seq) and not any("term" in w["flags"] for w in kept)    # only unclear / needs_owner / joke ... mark every word


def test_merge_span_confidence_rules():
    c = mk_consensus()
    c["turns"][0]["words"][1]["share"] = 0.7
    words = FM.flatten_consensus(c)
    seq, _ = FM.merge_span(words, [1], span("d0002", 1, 1, TEXT), dec("d0002", "we", 0.9, "keep_consensus"), NAMES)
    assert seq[0]["conf"] == 0.9                                                                          # a verified word is at least the editor's confidence
    seq, _ = FM.merge_span(words, [1], span("d0002", 1, 1, TEXT), dec("d0002", "we", 0.4, "uncertain", flags=["unclear"]), NAMES)
    assert seq[0]["conf"] == 0.4                                                                          # a doubtful span caps its words


def test_merge_span_punctuation_inherited_only_when_the_last_old_word_goes():
    c = mk_consensus(puncts={5: "."})
    words = FM.flatten_consensus(c)
    seq, _ = FM.merge_span(words, [5], span("d0003", 5, 5, TEXT), dec("d0003", "balance"), NAMES)
    assert seq[0]["punct"] == "."                                                                        # the vote's full stop survives a replaced word
    seq, _ = FM.merge_span(words, [5], span("d0003", 5, 5, TEXT), dec("d0003", "counterbalance now"), NAMES)
    assert seq[0]["punct"] == "" and seq[1]["punct"] == "."                                               # the kept word does not inherit it, the new last one does


def test_apply_decisions_end_to_end_no_word_lost_or_duplicated():
    c = mk_consensus()
    spans = [span("d0001", 1, 1, TEXT), span("d0002", 5, 5, TEXT), span("d0003", 8, 8, TEXT)]
    decisions = [dec("d0001", "we", 0.9, "keep_consensus"), dec("d0002", "balance of", 0.7), dec("d0003", "", 0.65, "keep_consensus")]
    final = build(c, spans, decisions)
    ws = [w["text"] for w in spoken(final)]
    assert ws == ["so", "we", "start", "with", "the", "balance", "of", "and", "then", "turn", "to", "the", "left"]    # 'we' (idx 8) removed by the empty decision
    assert final["checks"]["lost_words"] == 0 and final["checks"]["duplicated_words"] == 0 and final["checks"]["non_monotone"] == 0
    starts = [w["start"] for w in spoken(final)]
    assert starts == sorted(starts)
    assert final["stats"]["words_changed_by_adjudication"] == 2 and final["stats"]["words_v0"] == 13


def test_spans_with_overlapping_slots_are_separated_by_time():
    c = mk_consensus()
    a = span("d0001", 2, 4, TEXT)
    b = span("d0002", 4, 6, TEXT)
    b["slots"] = [100 + 3, 100 + 6]            # overlaps a's slots (102-104) but not its time window (the opening-joke situation)
    b["start"], b["end"] = 10.0 + 5 * 0.4, 10.0 + 6 * 0.4 + 0.3
    final = build(c, [a, b], [dec("d0001", "start with the", 0.9, "keep_consensus"), dec("d0002", "balance and", 0.7)])
    assert final["adjudication"]["clashes"] == []
    assert "balance and" in " ".join(w["text"] for w in spoken(final))


def test_insert_only_span_gets_a_time_between_its_neighbours():
    c = mk_consensus()
    gap = span("d0001", 3, 3, TEXT)
    c["turns"][0]["words"][4]["start"] += 0.5                  # leave a hole after 'with'
    c["turns"][0]["words"][4]["end"] += 0.5
    # a dropped word: an empty slot between 'with' and 'the' -> the span has no consensus word
    ins = {"id": "d0009", "slots": [1000, 1000], "start": 11.5, "end": 11.95, "consensus_text": "", "listen_to": None}
    final = build(c, [ins], [dec("d0009", "really", 0.8)])
    ws = spoken(final)
    k = [w["text"] for w in ws].index("really")
    assert ws[k - 1]["end"] <= ws[k]["start"] + 0.05 and ws[k]["start"] <= ws[k + 1]["start"] and ws[k]["edit"] == "inserted"
    assert any(f.startswith("time_") for f in ws[k]["flags"])


def test_raw_times_place_a_new_word_and_repeats_do_not_steal_the_neighbour():
    hyp = SimpleNamespace(engine="parakeet-tdt-0.6b-v3", toks=[SimpleNamespace(norm="really", t0=11.62, t1=11.9), SimpleNamespace(norm="we", t0=10.4, t1=10.7)])
    raw = FM.RawTimes([hyp])
    assert raw.find("really", 11.0, 12.0) == (11.62, 11.9)
    assert raw.find("nothing", 0, 100) is None
    toks = [{"norm": "we", "start": None, "end": None}]
    left = {"norm": "we", "start": 10.4, "end": 10.7}
    FM.place_run(toks, 10.0, 10.4, raw, left, None)                   # the only 'we' in the evidence is the neighbour's own word
    assert toks[0]["time_src"] == "window"
    toks = [{"norm": "really", "start": None, "end": None}]
    FM.place_run(toks, 11.4, 12.0, raw, None, None)
    assert toks[0]["time_src"] == "engines" and toks[0]["start"] == 11.62


def test_fix_times_makes_starts_monotone_and_trims_overlaps():
    ws = [{"kind": "word", "start": 1.0, "end": 1.5, "flags": []}, {"kind": "word", "start": 0.98, "end": 1.0, "flags": []}, {"kind": "event", "start": 0.5, "end": 0.9, "flags": []},
          {"kind": "word", "start": 1.2, "end": 1.4, "flags": []}]
    moved = FM.fix_times(ws)
    assert moved >= 1
    sp = [w for w in ws if w["kind"] == "word"]
    assert all(b["start"] >= a["start"] for a, b in zip(sp, sp[1:])) and all(a["end"] <= b["start"] + 1e-9 or a["end"] - b["start"] < 0.02 for a, b in zip(sp, sp[1:]))
    assert ws[2]["start"] == 0.5                                       # events are never moved


# ------------------------------------------------------------------------------------------------ events, speakers


def test_laugh_token_is_its_own_word_after_the_joke_with_the_override_time():
    c = mk_consensus()
    spans = [span("d0001", 2, 4, TEXT)]
    final = build(c, spans, [dec("d0001", f"start with {LAUGH} the", 0.5, "rewrite", flags=["joke", "emoji", "owner_fact"], needs_owner=True)],
                  overrides={"event_times": {"d0001": [{"t0": 11.6, "t1": 11.9, "joke": "j", "evidence": "owner fact"}]}})
    ev = [w for u in final["utterances"] for w in u["words"] if w["kind"] == "event"]
    assert len(ev) == 1 and ev[0]["text"] == LAUGH and (ev[0]["start"], ev[0]["end"]) == (11.6, 11.9) and "emoji" in ev[0]["flags"] and ev[0]["joke"] == "j"
    order = [w["text"] for u in final["utterances"] for w in u["words"]]
    assert order.index(LAUGH) == order.index("with") + 1 and final["events"][0]["text"] == LAUGH
    assert "joke" in ev[0]["flags"] and final["stats"]["events"] == 1


def test_speaker_window_by_time_and_by_span_and_interjection_merge():
    c = mk_consensus()
    c["turns"][0]["words"] += [dict(w) for w in c["turns"][1]["words"][:0]]
    final = build(c, [span("d0001", 8, 8, TEXT)], [dec("d0001", "we", 0.9, "keep_consensus")],
                  overrides={"speaker_windows": [{"span": "d0001", "voice": 1, "name": "Larissa", "interjection": True, "conf": 0.6}]})
    w = next(x for x in spoken(final) if x.get("span") == "d0001") if False else None
    turns = final["turns"]
    inter = [t for t in turns if t["interjection"]]
    assert len(inter) == 1 and inter[0]["speaker"] == "Larissa" and inter[0]["n_words"] == 1
    # a mini-turn between two stretches of the SAME voice is not an interjection
    c2 = mk_consensus(split=13)
    c2["turns"][0]["words"][5]["interjection"] = True
    final2 = build(c2, [], [])
    assert not any(t["interjection"] for t in final2["turns"]) and final2["stats"]["interjections_merged"] == 1 and len(final2["turns"]) == 1


def test_turns_and_utterances_split_at_speaker_change_punctuation_and_pause():
    c = mk_consensus(puncts={3: "."})
    c["turns"][1]["words"][3]["start"] += 2.0                 # a 2 s pause inside Kadu's turn
    c["turns"][1]["words"][3]["end"] += 2.0
    for w in c["turns"][1]["words"][4:]:
        w["start"] += 2.0
        w["end"] += 2.0
    final = build(c, [], [])
    assert [t["speaker"] for t in final["turns"]] == ["Larissa", "Kadu"]
    texts = [u["text"] for u in final["utterances"]]
    assert texts[0] == "So we start with." and len(final["utterances"]) >= 4 and texts[1].startswith("The counterbalance")
    assert final["turns"][0]["utterances"] and all(u["turn"] in {t["id"] for t in final["turns"]} for u in final["utterances"])


def test_verify_final_detects_a_duplicate_and_a_lost_word():
    c = mk_consensus()
    final = build(c, [], [])
    assert final["checks"]["duplicated_words"] == 0 and final["checks"]["lost_words"] == 0
    u = final["utterances"][0]
    u["words"].append(dict(u["words"][0], start=u["words"][-1]["start"] + 1.0, end=u["words"][-1]["end"] + 1.0))      # the same v0 word twice
    final["utterances"][1]["words"].pop(0)                                                                        # a word lost without a record
    chk = FM.verify_final(final, c)
    assert chk["duplicated_words"] == 1 and chk["lost_words"] == 1


def test_load_batches_requires_a_decision_for_every_span(tmp_path):
    (tmp_path / "batch_01.json").write_text(json.dumps({"batch": 1, "of": 1, "time_range": [0, 1], "n_spans": 1, "spans": [{"id": "d0001", "slots": [1, 1], "start": 0, "end": 1}]}), encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        FM.load_batches(tmp_path)
    (tmp_path / "batch_01.decisions.json").write_text(json.dumps([]), encoding="utf-8")
    with pytest.raises(ValueError):
        FM.load_batches(tmp_path)
    (tmp_path / "batch_01.decisions.json").write_text(json.dumps([dec("d0001", "x")]), encoding="utf-8")
    spans, decisions, heads = FM.load_batches(tmp_path)
    assert list(spans) == ["d0001"] and decisions["d0001"]["text"] == "x" and heads[0]["batch"] == 1


def test_systematic_report_lists_leftover_mishearings():
    c = mk_consensus(words="we do the exhibit then the counterbalance and the legs engaged".split())
    final = build(c, [], [])
    rep = FM.systematic_report(final, [{"term": "exit", "mishearings": ["exhibit", "exhale"]}, {"term": "legs engaged", "mishearings": ["lights engage"]}])
    assert rep[0]["occurrences"] == 0 and "exhibit" in rep[0]["mishearings_left"] and not rep[0]["ok"]
    assert rep[1]["occurrences"] == 1 and rep[1]["ok"]


# ------------------------------------------------------------------------------------------------ exports


def long_final():
    words = ("so this is a rather long sentence that has to be split into several captions because it does not fit in two lines of forty two characters "
             "and we keep talking for a while longer so that the splitting rules are exercised properly").split()
    c = mk_consensus(words=words, split=len(words), dt=0.35)
    return build(c, [], [])


def test_captions_respect_the_line_limits_and_keep_word_times():
    final = long_final()
    cap = FX.build_captions(final)
    assert cap["captions"] and cap["stats"]["max_line_chars"] <= FX.MAX_CHARS and cap["stats"]["max_lines_used"] <= 2
    for c in cap["captions"]:
        assert len(c["lines"]) <= 2 and all(FX.clen(l) <= 42 for l in c["lines"])
        assert c["end"] - c["start"] <= FX.MAX_CUE_S + 0.5
        assert all(c["start"] - 1e-6 <= w["start"] <= w["end"] <= c["end"] + 1e-6 for w in c["words"])
        assert c["speaker_key"] == "speaker_larissa"
    assert sum(1 for c in cap["captions"] for w in c["words"]) == len(spoken(final))
    starts = [c["start"] for c in cap["captions"]]
    assert starts == sorted(starts)


def test_captions_carry_the_laugh_as_its_own_word_and_skip_the_pre_roll_and_fillers():
    words = ["um", "before", "the", "start", "hello", "everyone", "this", "is", "fun"]
    c = mk_consensus(words=words, split=len(words), t0=-3.0, dt=0.6)
    c["turns"][0]["words"][4]["punct"] = "!"
    final = build(c, [span("d0001", 4, 4, words, t0=-3.0, dt=0.6)], [dec("d0001", f"hello {LAUGH}", 0.6, "rewrite", flags=["joke"])],
                  overrides={"event_times": {"d0001": [{"t0": 0.0, "t1": 0.5}]}})
    cap = FX.build_captions(final)
    all_words = [w for k in cap["captions"] for w in k["words"]]
    assert all(w["start"] >= 0 for w in all_words)                                       # nothing before the film starts
    assert not any(w["text"].lower().startswith("um") for w in all_words)                # fillers are not captions
    emo = [w for w in all_words if w["kind"] == "emoji"]
    assert len(emo) == 1 and emo[0]["text"] == LAUGH
    assert any(LAUGH in k["text"] for k in cap["captions"])


def test_srt_and_vtt_files_have_valid_timestamps_and_speaker_prefixes(tmp_path):
    final = long_final()
    cap = FX.build_captions(final)
    srt = FX.write_srt(cap["captions"], tmp_path / "a.srt").read_text(encoding="utf-8")
    vtt = FX.write_vtt(cap["captions"], tmp_path / "a.vtt").read_text(encoding="utf-8")
    blocks = [b for b in srt.strip().split("\n\n") if b]
    assert len(blocks) == len(cap["captions"])
    for b in blocks:
        n, t, *lines = b.split("\n")
        assert re.fullmatch(r"\d\d:\d\d:\d\d,\d\d\d --> \d\d:\d\d:\d\d,\d\d\d", t) and lines[0].startswith("Larissa: ")
    assert vtt.startswith("WEBVTT") and re.search(r"\d\d:\d\d:\d\d\.\d\d\d --> ", vtt)
    assert FX._ts(3661.5, ",") == "01:01:01,500" and FX._ts(-2.0, ",") == "00:00:00,000"


def test_readable_text_marks_low_confidence_words(tmp_path):
    c = mk_consensus()
    final = build(c, [span("d0001", 5, 5, TEXT)], [dec("d0001", "balance", 0.4, "uncertain", flags=["unclear"], needs_owner=True)])
    p = FX.write_txt(final, tmp_path / "t.txt")
    t = p.read_text(encoding="utf-8")
    assert "*balance*" in t and "Proofread first" in t and "d0001" in t and "[0:10.0] Larissa:" in t


# ------------------------------------------------------------------------------------------------ chapters, teaching events, jokes


def class_final():
    words = ("bring her back to the axis and shift the weight into the lunge then slowing down again the second exit we did with the right foot close to her foot "
             "and my arm is short so give her space").split()
    c = mk_consensus(words=words, split=len(words), dt=0.4)
    return build(c, [], [])


def test_chapters_are_snapped_contiguous_and_checked():
    final = class_final()
    spec = [{"start": 0.0, "title": "Opening part", "summary": "s", "terms": ["axis"]}, {"start": 14.1, "title": "A very long title with too many words", "summary": "s", "terms": ["unicorn"]}]
    doc = FD.build_chapters(final, {"chapters": spec}, 40.0)
    ch = doc["chapters"]
    assert ch[0]["start"] == 0.0 and ch[0]["end"] == ch[1]["start"] and ch[1]["end"] == 40.0
    assert ch[0]["id"] == "ch00" and ch[1]["id"] == "ch01"
    assert any("title has 8 words" in x for x in doc["checks"]) and any("unicorn" in x for x in doc["checks"]) and any("expected 8-15" in x for x in doc["checks"])
    assert ch[0]["terms_not_in_text"] == [] and ch[0]["speakers"]


def test_teaching_events_follow_the_rules_and_the_curation():
    final = class_final()
    doc = FD.build_teaching_events(final, {}, [{"id": "ch00", "start": 0.0, "end": 100.0}])
    by_trig = {t: e for e in doc["events"] for t in e["triggers"]}
    assert "overlay_T_axis" in by_trig["axis"]["tags"]
    assert "overlay_weight_change" in by_trig["weight"]["tags"]
    assert by_trig["slowing"]["tags"][0] == "slow_motion"
    assert "replay" in by_trig["again"]["tags"]
    assert "overlay_T_axis" in by_trig["exit"]["tags"] and "graph_moves" in by_trig["exit"]["tags"]           # "second exit" + "exit" = one moment
    assert any("camera_focus" in e["tags"] and e["focus"]["where"] == "arm" for e in doc["events"])
    assert all(t in FD.TAGS for e in doc["events"] for t in e["tags"]) and all(e["chapter"] == "ch00" for e in doc["events"])
    assert not any("caption_only" in e["tags"] and len(e["tags"]) > 1 for e in doc["events"])
    # curation: drop, retag and add
    cur = {"teaching_events": {"drop": [{"near": by_trig["again"]["start"], "trigger": "again"}], "retag": [{"near": by_trig["axis"]["start"], "trigger": "axis", "tags": ["camera_focus"]}],
                               "add": [{"phrase": "right foot", "tags": ["overlay_footsteps/physics"], "kind": "demonstration", "note": "n"}]}}
    doc2 = FD.build_teaching_events(final, cur, [{"id": "ch00", "start": 0.0, "end": 100.0}])
    trig2 = {t: e for e in doc2["events"] for t in e["triggers"]}
    assert "again" not in trig2 and trig2["axis"]["tags"] == ["camera_focus"] and doc2["dropped_by_curation"] == 1 and doc2["added_by_curation"] == 1


def test_teaching_event_subject_uses_roles_and_the_speakers_own_role():
    assert FD._subject("bring the followers back".split(), "Kadu") == "follower"
    assert FD._subject("my arm is short".split(), "Kadu") == "leader"
    assert FD._subject("my arm is short".split(), "Larissa") == "follower"
    assert FD._subject("i bring her back".split(), "Kadu") == "couple"
    assert FD._subject("the frame is closed".split(), "Larissa") == "couple"


def test_jokes_json_lists_tokens_candidates_and_explained_stretches():
    c = mk_consensus()
    c["wordless_sounds"] = [{"t0": 10.5, "t1": 11.0, "peak_excess_db": 15.0, "phones": 4}, {"t0": 30.0, "t1": 30.8, "peak_excess_db": 20.0, "phones": 6}]
    final = build(c, [span("d0001", 2, 2, TEXT)], [dec("d0001", f"start {LAUGH}", 0.5, "rewrite", flags=["joke"], needs_owner=True)],
                  overrides={"event_times": {"d0001": [{"t0": 11.2, "t1": 11.5}]}})
    over = {"joke_candidates": [{"id": "jc_x", "t0": 20.0, "t1": 21.0, "speaker": "Kadu", "text": "t", "why": "w"}]}
    doc = FD.build_jokes(final, c, over, [{"t0": 40.0, "t1": 42.0, "peak": 0.2, "phones": 3, "family": "laughter"}])
    assert doc["tokens_in_transcript"][0]["token"] == LAUGH and "joke" in doc["tokens_in_transcript"][0]["flags"] and doc["tokens_in_transcript"][0]["after_words"].endswith("start")
    ids = [x["id"] for x in doc["laugh_candidates"]]
    assert "jc_x" in ids and all("laugh_candidate" in x["flags"] for x in doc["laugh_candidates"])
    assert any(x["t0"] == 30.0 for x in doc["laugh_candidates"]) and any(x["t0"] == 40.0 for x in doc["laugh_candidates"])
    assert [x["t0"] for x in doc["explained_wordless_sounds"]] == [10.5]                      # speech is there: not a laugh candidate
    assert all(x["approve_emoji"] is None for x in doc["laugh_candidates"])


# ------------------------------------------------------------------------------------------------ evaluation, review page


def test_evaluation_sequences_bands_and_expected_accuracy():
    final = class_final()
    cf = FEV.seq_final(final, (-1e9, 1e9))
    assert cf[0][:3] == ["bring", "her", "back"] and len(cf[0]) == len(cf[1])
    assert FEV.seq_final(final, (0.0, 1.0))[0] == []                                           # times are master seconds: the window selects words
    ea = FEV.expected_accuracy(final, {"turns": []}, None)
    assert ea["n_words"] == len(spoken(final)) and 0 < ea["error_central"] < 0.1 and ea["error_low"] <= ea["error_central"] <= ea["error_high"]
    assert [b["band"] for b in ea["bands"]][0] == "conf >= 0.9" and "range" in ea["statement"]
    assert FEV._lev(["a", "b", "c"], ["a", "x", "c"]) == 1 and FEV._lev([], ["a"]) == 1


def test_span_effects_sign_convention():
    c = mk_consensus()
    final = build(c, [span("d0001", 5, 5, TEXT)], [dec("d0001", "balance")])
    toks = [SimpleNamespace(norm=w.lower(), tk=10.0 + i * 0.4 + 0.15) for i, w in enumerate(TEXT)]
    toks[5] = SimpleNamespace(norm="balance", tk=10.0 + 5 * 0.4 + 0.15)
    hyp = SimpleNamespace(cover=(0.0, 100.0), toks=toks)
    eff = FEV.span_effects(final, c, [hyp])
    assert eff["spans"][0]["delta_edits_per_source"] > 0 and eff["toward"] == 1 and eff["away"] == 0     # the source says 'balance': the edit moved toward it


def test_review_page_is_self_contained_escaped_and_lists_the_owner_spans():
    c = mk_consensus()
    final = build(c, [span("d0001", 5, 5, TEXT), span("d0002", 8, 8, TEXT)],
                  [dec("d0001", "<b>balance</b>", 0.4, "uncertain", flags=["unclear"], needs_owner=True), dec("d0002", f"we {LAUGH}", 0.7, "rewrite", flags=["joke"])],
                  overrides={"event_times": {"d0002": [{"t0": 14.0, "t1": 14.4}]}})
    chapters = FD.build_chapters(final, {"chapters": [{"start": 0.0, "title": "One", "summary": "s&s", "terms": []}]}, 30.0)
    jokes = {"found_by_pipeline": {"summary": "x"}, "tokens_in_transcript": [], "laugh_candidates": []}
    who = [{"id": "c01", "master_t0": 1, "master_t1": 2, "cam": "00", "category": "a", "reason": "r<", "text": "t", "file": "c01.wav", "name": None, "confidence": None}]
    ev = {"rows": [{"cam": "00", "engine": "e", "n_ref": 10, "pwer_v0": 0.1, "pwer_final": 0.2}], "expected_accuracy": {"statement": "st"}}
    html_ = FR.build_review(final, chapters, jokes, who, ev, {"d0001": "review/final_clips/d0001.wav"})
    assert html_.startswith("<!doctype html>") and "<b>balance</b>" not in html_ and "&lt;b&gt;balance&lt;/b&gt;" in html_       # decision text is escaped
    assert "http://" not in html_ and "https://" not in html_ and "src=\"review/final_clips/d0001.wav\"" in html_ and "review/clips/c01.wav" in html_
    assert LAUGH in html_ and "Proofread first" in html_ and "ch00" in html_ and "Who is who" in html_
    assert html_.count("class=\"row") == len(final["utterances"])


def test_choose_phone_skips_a_phone_that_had_not_started(monkeypatch):
    class Clk:
        def __init__(self, offset):
            self.offset, self.slope = offset, 1.0

        def to_master(self, t):
            return t + self.offset

    clocks = {"00": Clk(-8.9), "01": Clk(0.0), "03": Clk(60.9), "05": Clk(7.4)}
    monkeypatch.setattr(FR, "wav_seconds", lambda p: 300.0)
    sp = {"id": "d1", "start": 5.0, "end": 6.0, "listen_to": {"phone": "03"}}
    assert FR.choose_phone(sp, clocks) == "05" or FR.choose_phone(sp, clocks) == "01"           # phone 03 starts at 60.9 s: not recording at 5 s
    assert FR.choose_phone(sp, clocks) != "03"
    sp2 = {"id": "d2", "start": 100.0, "end": 101.0, "listen_to": {"phone": "03"}}
    assert FR.choose_phone(sp2, clocks) == "03"                                                  # recording: the span's own choice stands
    sp3 = {"id": "d3", "start": -8.0, "end": -6.0, "listen_to": {"phone": "01"}}
    assert FR.choose_phone(sp3, clocks) == "00"                                                  # before the master: only phone 00
    sp4 = {"id": "d4", "start": 500.0, "end": 501.0, "listen_to": None}
    assert FR.choose_phone(sp4, clocks) is None


def test_cut_clip_without_ffmpeg_reuses_existing_files_only(tmp_path, monkeypatch):
    class Clk:
        offset, slope = 0.0, 1.0

        def to_master(self, t):
            return t

    monkeypatch.setattr(FR, "wav_seconds", lambda p: 300.0)
    sp = {"id": "d9", "start": 10.0, "end": 12.0, "listen_to": {"phone": "01"}}
    assert FR.cut_clip(sp, {"01": Clk()}, out_dir=tmp_path, make=False) is None
    (tmp_path / "d9_01_9.0s.wav").write_bytes(b"x")
    assert FR.cut_clip(sp, {"01": Clk()}, out_dir=tmp_path, make=False).name == "d9_01_9.0s.wav"


# ------------------------------------------------------------------------------------------------ corrections after the audit: span overrides, speaker doubts, breaks


def test_span_overrides_replace_the_decision_with_a_recorded_reason():
    c = mk_consensus()
    spans = [span("d0001", 5, 5, TEXT)]
    decisions = [dec("d0001", "balance of", 0.7, flags=["unclear", "term"], needs_owner=True)]
    ov = {"span_overrides": {"d0001": {"text": "counterbalance", "confidence": 0.5, "needs_owner": False, "drop_flags": ["unclear"], "why": "audit: nothing supports the 'of'"}}}
    final = build(c, spans, decisions, overrides=ov)
    row = final["adjudication"]["spans"][0]
    assert row["text"] == "counterbalance" and row["decision"] == "rewrite" and row["confidence"] == 0.5 and not row["needs_owner"] and "unclear" not in row["flags"] and "term" in row["flags"]
    assert row["reason"] == "audit: nothing supports the 'of'" and row["editor_reason"] == "test"
    assert row["override"]["was"]["text"] == "balance of" and row["override"]["was"]["needs_owner"] is True
    assert final["adjudication"]["span_overrides"][0]["span"] == "d0001" and final["stats"]["spans_overridden"] == 1 and final["stats"]["spans_needs_owner"] == 0
    assert [w["text"] for w in spoken(final)] == TEXT                                   # the override undoes the editor's 'of'
    assert final["checks"]["lost_words"] == 0 and final["proofread_first"] == []


def test_span_overrides_are_strict_about_ids_fields_and_reasons():
    c = mk_consensus()
    spans = [span("d0001", 5, 5, TEXT)]
    decisions = [dec("d0001", "balance")]
    with pytest.raises(ValueError, match="unknown span"):
        build(c, spans, decisions, overrides={"span_overrides": {"d0999": {"text": "x", "why": "w"}}})
    with pytest.raises(ValueError, match="why"):
        build(c, spans, decisions, overrides={"span_overrides": {"d0001": {"text": "x"}}})
    with pytest.raises(ValueError, match="unknown fields"):
        build(c, spans, decisions, overrides={"span_overrides": {"d0001": {"txt": "x", "why": "w"}}})
    new, log = FM.apply_span_overrides({}, {"d0001": dec("d0001", "a", 0.9)}, {"d0001": {"confidence": 0.3, "add_flags": ["unclear"], "why": "w"}})
    assert new["d0001"]["decision"] == "choose_candidate" and new["d0001"]["flags"] == ["unclear"] and log[0]["now"]["confidence"] == 0.3     # no text change: the decision type stays


def test_speaker_window_without_a_voice_only_lowers_the_confidence_and_flags(tmp_path):
    c = mk_consensus()
    ov = {"speaker_windows": [{"t0": 10.3, "t1": 10.9, "conf": 0.4, "flags": ["speaker_unresolved"], "why": "x"},
                              {"t0": 12.1, "t1": 12.5, "voice": 0, "name": "Kadu", "interjection": True, "conf": 0.7, "why": "y"}]}
    final = build(c, [], [], overrides=ov)
    ws = {}
    for u in final["utterances"]:
        for w in u["words"]:
            ws.setdefault(w["text"], w)                                                                                        # the first 'we' (10.4 s), not the one at 13.2 s
    assert ws["we"]["flags"] == ["speaker_unresolved"] and "speaker_unresolved" not in ws["start"].get("flags", [])         # only the words whose centre is in the window
    u = next(u for u in final["utterances"] if any(w["text"] == "we" for w in u["words"]))
    assert u["speaker"] == "Larissa" and "speaker_unresolved" in u["flags"] and final["stats"]["speaker_unresolved_words"] == 1
    turn = final["turns"][0]
    assert turn["speaker"] == "Larissa" and turn["voice_conf"] < 1.0                                                          # the turn's mean voice confidence dropped
    relabelled = [t for t in final["turns"] if t["interjection"]]
    assert relabelled and relabelled[0]["speaker"] == "Kadu"
    assert any("speaker_override" in w.get("flags", []) for u2 in final["utterances"] for w in u2["words"])
    cap = FX.build_captions(final)
    assert any("speaker_unresolved" in k["flags"] for k in cap["captions"])
    t = FX.write_txt(final, tmp_path / "t.txt").read_text(encoding="utf-8")
    assert "(speaker not settled)" in t


def test_utterance_break_splits_a_sentence_without_changing_a_word_or_an_id():
    c = mk_consensus()
    plain = build(c, [], [])
    cut = build(c, [], [], overrides={"utterance_breaks": [{"t": 11.0, "why": "structure only"}]})
    assert [w["text"] for w in spoken(plain)] == [w["text"] for w in spoken(cut)]
    assert [w["start"] for w in spoken(plain)] == [w["start"] for w in spoken(cut)]
    assert [u["id"] for u in plain["utterances"]] == ["u000", "u001"]
    assert [u["id"] for u in cut["utterances"]] == ["u000", "u000b", "u001"]                 # the other ids stay what they were
    first, second = cut["utterances"][0], cut["utterances"][1]
    assert first["text"] == "So we start" and second["text"] == "with the counterbalance" and second["cap_first"] is False and second["continues"] == "u000" and "continues" not in first
    assert cut["turns"][0]["utterances"] == ["u000", "u000b"] and cut["turns"][0]["n_words"] == plain["turns"][0]["n_words"]
    assert FX.build_captions(cut)["captions"][0]["utterance"] == "u000"


# ------------------------------------------------------------------------------------------------ the owner's opening


def opening_overrides(**extra):
    op = {"segments": [{"start": 10.0, "end": 11.0, "speaker": "Kadu", "text": "Hello there!"}, {"start": 11.0, "end": 12.0, "speaker": "Larissa", "text": "Good morning, Kadu"}],
          "emoji": [{"t": 11.5, "speaker": "Larissa", "joke": "greeting"}], "chapter_summary": "The owner's summary."}
    op.update(extra)
    return {"owner_opening": op}


def test_owner_opening_replaces_the_stretch_with_the_owners_words_speakers_and_emoji():
    c = mk_consensus()
    spans = [span("d0001", 1, 3, TEXT)]
    decisions = [dec("d0001", "we start with", 0.4, "uncertain", flags=["unclear"], needs_owner=True)]
    final = build(c, spans, decisions, overrides=opening_overrides())
    ws = spoken(final)
    assert [w["text"] for w in ws] == ["hello", "there", "good", "morning", "Kadu", "counterbalance", "and", "then", "we", "turn", "to", "the", "left"]
    owner_words = ws[:5]
    assert all(w["conf"] == 0.95 and w["conf_src"] == "owner" and "owner_opening" in w["flags"] and "needs_owner" not in w["flags"] and "unclear" not in w["flags"] for w in owner_words)
    assert [u["speaker"] for u in final["utterances"][:2]] == ["Kadu", "Larissa"] and final["utterances"][0]["text"] == "Hello there!"
    assert all(10.0 <= w["start"] < 12.0 for w in owner_words) and [w["start"] for w in ws] == sorted(w["start"] for w in ws)
    ev = final["events"]
    assert len(ev) == 1 and ev[0]["start"] == 11.5 and ev[0]["joke"] == "greeting" and ev[0]["laugher"] == "Larissa" and "owner_opening" in ev[0]["flags"]
    order = [w["text"] for u in final["utterances"] for w in u["words"]]
    assert order.index(LAUGH) == order.index("morning") + 1                                      # right after the last word spoken before the laugh
    rec = final["owner_opening"]
    assert rec["from"] == 10.0 and rec["until"] == 12.0 and rec["removed_v0"] == 5 and rec["added"][0] == "hello"
    assert final["checks"]["lost_words"] == 0 and final["checks"]["non_monotone"] == 0 and final["checks"]["overlap_gt_20ms"] == 0
    row = final["adjudication"]["spans"][0]
    assert row["owner_resolved"] and row["decision"] == "owner" and not row["needs_owner"] and row["editor_text"] == "we start with" and "unclear" not in row["flags"]
    assert final["stats"]["spans_needs_owner"] == 0 and final["stats"]["spans_unclear"] == 0 and final["stats"]["spans_owner_resolved"] == 1 and final["proofread_first"] == []


def test_owner_opening_word_times_replace_the_estimate_and_are_validated():
    c = mk_consensus()
    seg = {"start": 10.0, "end": 12.0, "speaker": "Larissa", "text": "Hi, I'm Kadu and this is Larissa!", "word_times": [[10.0, 10.2], [10.3, 10.5], [10.5, 10.9], [11.2, 11.3], [11.3, 11.5], [11.5, 11.6], [11.6, 12.0]]}
    final = build(c, [], [], overrides={"owner_opening": {"segments": [seg], "emoji": [{"t": 10.95, "t1": 11.2, "speaker": "Larissa"}]}})
    ws = [w for w in spoken(final) if "owner_opening" in w.get("flags", [])]
    assert [w["text"] for w in ws] == ["hi", "I'm", "Kadu", "and", "this", "is", "Larissa"]
    assert [(w["start"], w["end"]) for w in ws] == [tuple(x) for x in seg["word_times"]] and all("time_owner" in w.get("flags", []) for w in ws)
    assert final["utterances"][0]["text"].startswith("Hi, I'm Kadu 😂 and this is Larissa!") and final["events"][0]["start"] == 10.95
    for bad in ([[10.0, 10.2]], [[10.0, 10.2], [10.3, 10.2]] + seg["word_times"][2:], [[10.4, 10.5], [10.0, 10.2]] + seg["word_times"][2:]):
        with pytest.raises(ValueError, match="word_times"):
            build(c, [], [], overrides={"owner_opening": {"segments": [dict(seg, word_times=bad)]}})


def test_a_pause_split_utterance_does_not_start_with_a_capital_but_a_new_sentence_does():
    ws = [cw(0, "we", 10.0), cw(1, "go", 10.3), cw(2, "left", 10.6, "."), cw(3, "then", 11.0), cw(4, "we", 11.4), cw(5, "stop", 13.5), cw(6, "now", 14.0, "!")]
    c = {"turns": [{"id": "t000", "voice": 1, "speaker": "Larissa", "words": ws}], "naming": {"naming_confidence": 0.85}, "unplaced_sources": {}}
    final = build(c, [], [])
    u = final["utterances"]
    assert len(u) == 3
    assert u[0]["text"] == "We go left." and "cap_first" not in u[0]                              # a turn starts with a capital
    assert u[1]["text"] == "Then we" and "cap_first" not in u[1]                                  # after a full stop: a new sentence
    assert u[2]["text"] == "stop now!" and u[2]["cap_first"] is False                              # the pause split a sentence that "we" left open: no capital on "stop"
    cap = FX.build_captions(final)["captions"]
    assert [k["text"] for k in cap if k["utterance"] == u[2]["id"]][0].startswith("stop now!")
    assert "] Larissa: stop now!" in FX.write_txt(final, Path(__file__).parent / "_cap_first.txt").read_text(encoding="utf-8")
    (Path(__file__).parent / "_cap_first.txt").unlink()


def test_owner_opening_marks_a_second_voice_between_two_stretches_of_the_other_voice():
    c = mk_consensus(split=13)                                                 # Larissa says everything
    ov = {"owner_opening": {"segments": [{"start": 10.0, "end": 11.0, "speaker": "Larissa", "text": "Hello"}, {"start": 11.0, "end": 11.6, "speaker": "Kadu", "text": "Hi"},
                                         {"start": 11.6, "end": 12.4, "speaker": "Larissa", "text": "And so"}]}}
    final = build(c, [], [], overrides=ov)
    assert [(t["speaker"], t["interjection"]) for t in final["turns"][:3]] == [("Larissa", False), ("Kadu", True), ("Larissa", False)]
    # an utterance break inside the stretch the owner replaced is dropped (the owner's segments are the structure there)
    ov2 = dict(opening_overrides(), utterance_breaks=[{"t": 11.5, "why": "old structure"}])
    assert not any(u["id"].endswith("b") for u in build(mk_consensus(), [], [], overrides=ov2)["utterances"])
    assert any(u["id"].endswith("b") for u in build(mk_consensus(), [], [], overrides={"utterance_breaks": [{"t": 11.5, "why": "no owner opening"}]})["utterances"])
    final2 = build(c, [], [], overrides={"owner_opening": {"segments": [], "emoji": []}})
    assert final2["owner_opening"] is None and [w["text"] for w in spoken(final2)] == TEXT       # an empty answer changes nothing


def test_owner_opening_validates_segments_and_loads_from_its_own_file(tmp_path):
    c = mk_consensus()
    with pytest.raises(ValueError, match="needs start < end"):
        build(c, [], [], overrides={"owner_opening": {"segments": [{"start": 11.0, "end": 10.0, "speaker": "Kadu", "text": "x"}]}})
    ov_p, op_p = tmp_path / "final_overrides.json", tmp_path / "owner_opening.json"
    ov_p.write_text(json.dumps({"event_times": {}}), encoding="utf-8")
    base = FM.load_overrides(ov_p, op_p)
    assert "owner_opening" not in base and base["span_overrides"] == {} and base["utterance_breaks"] == [] and base["owner_clips"] == []
    op_p.write_text(json.dumps(opening_overrides()["owner_opening"]), encoding="utf-8")
    assert FM.load_overrides(ov_p, op_p)["owner_opening"]["segments"][0]["text"] == "Hello there!"
    assert "owner_opening" not in FM.load_overrides(ov_p, None)


def test_owner_opening_feeds_the_chapter_summary_and_approves_the_laugh_candidate():
    c = mk_consensus()
    final = build(c, [], [], overrides=opening_overrides())
    ov = opening_overrides()
    ov["chapters"] = [{"start": 0.0, "title": "Opening", "summary": "curated", "terms": []}, {"start": 12.0, "title": "Next", "summary": "s", "terms": []}]
    ch = FD.build_chapters(final, ov, 30.0)["chapters"]
    assert ch[0]["summary"] == "The owner's summary." and ch[0]["summary_source"] == "owner_opening" and ch[1]["summary"] == "s"
    cands = {"joke_candidates": [{"id": "jc_a", "t0": 11.2, "t1": 11.8, "speaker": None, "text": None, "why": "w"}, {"id": "jc_b", "t0": 20.0, "t1": 21.0, "speaker": None, "text": None, "why": "w"}]}
    jk = FD.build_jokes(final, c, cands, [])
    got = {x["id"]: x["approve_emoji"] for x in jk["laugh_candidates"]}
    assert got == {"jc_a": True, "jc_b": None}
    assert jk["tokens_in_transcript"][0]["owner_given"] is True and jk["tokens_in_transcript"][0]["needs_owner"] is False


def test_jokes_do_not_repeat_a_loud_stretch_that_a_candidate_already_covers():
    c = mk_consensus()
    c["wordless_sounds"] = [{"t0": 30.2, "t1": 30.8, "peak_excess_db": 20.0, "phones": 6}]
    final = build(c, [], [])
    jk = FD.build_jokes(final, c, {"joke_candidates": [{"id": "jc_x", "t0": 30.0, "t1": 31.0, "speaker": None, "text": None, "why": "w"}]}, [])
    assert [x["id"] for x in jk["laugh_candidates"]] == ["jc_x"]


def test_review_page_shows_the_owner_questions_with_a_player_per_clip():
    c = mk_consensus()
    final = build(c, [], [])
    page = FR.build_review(final, None, None, [], None, {}, owner_questions="1. What is said? Listen to review/final_clips/q1_00_3.0s.wav <b>now</b>\n2. Again review/final_clips/q1_00_3.0s.wav")
    assert 'id="questions"' in page and page.count('src="review/final_clips/q1_00_3.0s.wav"') == 1            # one player per distinct clip
    assert "&lt;b&gt;now&lt;/b&gt;" in page and "<b>now</b>" not in page
    assert 'id="questions"' not in FR.build_review(final, None, None, [], None, {})
