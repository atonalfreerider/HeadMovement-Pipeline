"""dancecap.speech fusion (fuse.py, fuse_post.py, fuse_batches.py, fuse_eval.py, fuse_levels.py): the alignment / voting core and its post-processing with
SYNTHETIC hypotheses. Offline, no GPU, no audio, no real transcripts.

    ..\\.venv-nopo4d\\Scripts\\python -m pytest tests/test_speech_fuse.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dancecap.speech import fuse, fuse_batches as FB, fuse_eval as FE, fuse_levels as FL, fuse_post as FP  # noqa: E402

ENGINES = ["whisper-large-v3", "whisper-large-v3-turbo", "parakeet-tdt-0.6b-v3", "canary-1b-v2"]


def mk(hid, cam, engine, words, t0=70.0, dt=0.4, cover=(0.0, 400.0), jitter=None, conf=None, punct=None):
    """A hypothesis from a list of words (None = a word the engine did not write, to keep the timeline of the others)."""
    raw = []
    for i, w in enumerate(words):
        if w is None:
            continue
        j = 0.0 if jitter is None else jitter[i % len(jitter)]
        s = t0 + i * dt + j
        raw.append({"word": w, "start": s, "end": s + 0.3, "prob": conf})
    h = fuse.Hyp(hid, cam, engine, fuse.make_toks(raw, hid), cover, f"{engine}@{cam}", fuse.ENGINE_PRIOR[engine])
    return h


TRUTH = "we start with the counterbalance and then we turn to the left".split()


def corrupt(words, subs=None, drops=()):
    out = list(words)
    for i, w in (subs or {}).items():
        out[i] = w
    for i in drops:
        out[i] = None
    return out


# ------------------------------------------------------------------------------------------------ tokens


def test_pieces_canonical_forms():
    assert fuse.pieces("Gonna") == [("going", "going"), ("to", "to")]
    assert fuse.pieces("OK,") == [("okay", "OK")]
    assert fuse.pieces("Don't") == [("dont", "Don't")]
    assert fuse.pieces("...") == []
    assert fuse.pieces("Um,")[0][0] == "um" and "um" in fuse.FILLERS
    assert fuse.pieces("uhm")[0][0] == "um" and fuse.pieces("er")[0][0] == "uh"
    assert fuse.pieces("2")[0][0] == "two"
    assert fuse.pieces("Bônus")[0][0] == "bonus"                      # accents are dropped for comparison, the display keeps them
    assert fuse.pieces("no 'cause") == [("no", "no"), ("because", "cause")]


def test_trailing_punct_classes():
    assert fuse.trailing_punct("Hello!") == "!"
    assert fuse.trailing_punct("Really?") == "?"
    assert fuse.trailing_punct("to,") == ","
    assert fuse.trailing_punct("done.") == "."
    assert fuse.trailing_punct("so...") == "..."
    assert fuse.trailing_punct("plain") == ""


def test_make_toks_splits_time_of_expansions_by_letters():
    toks = fuse.make_toks([{"word": "gonna", "start": 1.0, "end": 2.0, "prob": 0.9}])
    assert [t.norm for t in toks] == ["going", "to"]
    assert toks[0].t0 == 1.0 and toks[-1].t1 == 2.0 and toks[0].t1 == pytest.approx(toks[1].t0)
    assert toks[0].t1 - toks[0].t0 > toks[1].t1 - toks[1].t0          # "going" is longer than "to"


def test_sim_orders_sounds_and_spellings():
    assert fuse.sim("same", "same") == 1.0
    assert fuse.sim("to", "too") >= 0.85 and fuse.sim("to", "two") >= 0.85
    assert fuse.sim("followers", "follower") >= 0.7
    assert fuse.sim("counterbalance", "counterbounds") > 0.5
    assert fuse.sim("the", "weight") < 0.4 and fuse.sim("dance", "forward") == 0.0
    assert fuse.sim("a", "b") == fuse.sim("b", "a")


def test_merge_compounds_needs_two_other_hypotheses():
    a = mk(0, "01", ENGINES[0], ["the", "counter", "balance", "works"])
    b = mk(1, "01", ENGINES[2], ["the", "counterbalance", "works"])
    c = mk(2, "02", ENGINES[2], ["the", "counterbalance", "works"])
    assert fuse.merge_compounds([a, b]) == 0                          # only one other hypothesis writes the compound
    assert [t.norm for t in a.toks] == ["the", "counter", "balance", "works"]
    assert fuse.merge_compounds([a, b, c]) == 1
    assert [t.norm for t in a.toks] == ["the", "counterbalance", "works"]
    assert [t.idx for t in a.toks] == [0, 1, 2]


# ------------------------------------------------------------------------------------------------ alignment and voting


def six_hyps(P):
    """Six hypotheses of TRUTH, each with different mistakes (no two the same); phones 01..03 x two engines."""
    hs = [
        mk(0, "01", ENGINES[0], corrupt(TRUTH, {4: "counter bounds"}), conf=0.8),
        mk(1, "01", ENGINES[2], corrupt(TRUTH, drops=(1,))),
        mk(2, "02", ENGINES[0], corrupt(TRUTH, {9: "learn"}), conf=0.7),
        mk(3, "02", ENGINES[3], corrupt(TRUTH, {7: "we"}, drops=(10,))),
        mk(4, "03", ENGINES[2], corrupt(TRUTH, {2: "which"})),
        mk(5, "03", ENGINES[1], corrupt(TRUTH, {5: "an"}), conf=0.9),
    ]
    return hs


def test_consensus_recovers_the_truth_from_hypotheses_with_independent_errors():
    P = fuse.Params(weight_mode="uniform", refine_rounds=1)
    C = fuse.consensus(six_hyps(P), P)
    words = [r.winner for r in C.results if r.winner]
    assert words == [fuse.pieces(w)[0][0] for w in TRUTH] or words == [w for w in TRUTH]
    # most words are unanimous or near: the share of each winner is the weight that wrote it
    shares = {r.winner: r.share for r in C.results if r.winner}
    assert shares["we"] >= 0.8 and shares["the"] >= 0.8
    # the contested word keeps its alternatives with who wrote them
    slot = next(r for r in C.results if r.winner == "counterbalance")
    alts = {n for n, _, _ in slot.tally if n}
    assert "counterbalance" in alts and slot.share >= 0.7


def test_alignment_survives_timestamp_smear_and_jitter():
    P = fuse.Params(weight_mode="uniform", refine_rounds=1)
    hs = six_hyps(P)
    # one engine with timestamps off by up to 0.9 s (whisper after a pause) must still align word for word
    hs[0] = mk(0, "01", ENGINES[0], corrupt(TRUTH, {4: "counter bounds"}), jitter=[0.9, -0.8, 0.7, -0.6, 0.5, 0.0], conf=0.8)
    C = fuse.consensus(hs, P)
    assert [r.winner for r in C.results if r.winner] == TRUTH


def test_time_penalty_keeps_a_repeated_word_in_its_place():
    # "the" occurs twice; hypothesis B hears only the second sentence and must align to the SECOND "the"
    a = mk(0, "01", ENGINES[2], ["the", "cat", "sat", "the", "dog", "ran"], dt=1.0)
    b = mk(1, "02", ENGINES[2], ["the", "dog", "ran"], t0=73.0, dt=1.0)
    c = mk(2, "03", ENGINES[3], ["the", "cat", "sat", "the", "dog", "ran"], dt=1.0)
    P = fuse.Params(weight_mode="uniform", refine_rounds=1, compounds=False)
    C = fuse.consensus([a, b, c], P)
    slots = C.slots
    second_the = [s for s in slots if any(t.norm == "the" and t.t0 > 72 for t in s.mem.values())]
    assert len(second_the) == 1 and 1 in second_the[0].mem            # b's "the" is in the slot of the second "the"
    assert sum(1 for s in slots if 1 in s.mem) == 3


def test_a_word_written_by_one_hypothesis_of_six_is_voted_out_and_a_missing_word_is_voted_in():
    P = fuse.Params(weight_mode="uniform", refine_rounds=1, compounds=False)
    base = "alpha beta gamma delta epsilon zeta".split()
    hs = [mk(i, f"0{i % 3 + 1}", ENGINES[i % 4], list(base)) for i in range(6)]
    hs[0] = mk(0, "01", ENGINES[0], ["alpha", "beta", "gamma", "extra", "delta", "epsilon", "zeta"])        # inserts a word nobody else writes
    hs[1] = mk(1, "02", ENGINES[1], ["alpha", "beta", "gamma", None, "epsilon", "zeta"])                    # misses "delta"
    C = fuse.consensus(hs, P)
    got = [r.winner for r in C.results if r.winner]
    assert got == base
    extra = next(r for r in C.results if "extra" in [n for n, _, _ in r.tally])
    assert extra.winner is None and extra.null_share > 0.8


def test_hypotheses_that_did_not_record_do_not_vote_nothing():
    P = fuse.Params(weight_mode="uniform", refine_rounds=1, compounds=False)
    first = "one two three four".split()
    second = "five six seven eight".split()
    # phone 03 only recorded the second half (cover starts at 71.7); its absence in the first half must not count as a "nothing" vote
    hs = [mk(0, "01", ENGINES[2], first + second), mk(1, "02", ENGINES[2], first + second), mk(2, "03", ENGINES[3], [None] * 4 + second, cover=(71.7, 400.0))]
    C = fuse.consensus(hs, P)
    r1 = next(r for r in C.results if r.winner == "one")
    assert r1.share == pytest.approx(1.0) and r1.voters == 2
    r5 = next(r for r in C.results if r.winner == "six")
    assert r5.voters == 3


def test_fillers_are_judged_among_the_engines_that_write_fillers():
    P = fuse.Params(weight_mode="uniform", refine_rounds=1, compounds=False)
    words = ["so", "um", "we", "start"]
    hs = [mk(0, "01", ENGINES[0], ["so", "we", "start"]), mk(1, "02", ENGINES[0], ["so", "we", "start"]),
          mk(2, "01", ENGINES[2], words), mk(3, "02", ENGINES[2], words), mk(4, "01", ENGINES[1], words)]
    C = fuse.consensus(hs, P)
    got = [r.winner for r in C.results if r.winner]
    assert got == ["so", "um", "we", "start"]                         # whisper-large-v3 never writes fillers and so does not vote them out
    assert next(r for r in C.results if r.winner == "um").filler


def test_overlap_keep_is_off_by_default_and_flags_the_slot():
    P = fuse.Params(weight_mode="uniform", refine_rounds=1, compounds=False)
    base = "a1 b2 c3 d4".split()
    hs = [mk(i, f"0{i % 3 + 1}", ENGINES[i % 4], base) for i in range(5)]
    hs[0] = mk(0, "01", ENGINES[0], ["a1", "b2", "xx9", "c3", "d4"])
    C = fuse.consensus(hs, P, overlaps=[(70.7, 71.1)])
    odd = next(r for r in C.results if "xx9" in [n for n, _, _ in r.tally])
    assert odd.winner is None and odd.overlap and not odd.kept_by_overlap


# ------------------------------------------------------------------------------------------------ weights


def test_logistic_fit_recovers_a_signal():
    rng = np.random.default_rng(1)
    X = np.c_[np.ones(500), rng.normal(size=500)]
    y = (rng.random(500) < 1 / (1 + np.exp(-(0.3 + 2.0 * X[:, 1])))).astype(float)
    b = fuse.logistic_fit(X, y, l2=0.5)
    assert b[1] > 1.2 and abs(b[0] - 0.3) < 0.5


def test_calibrated_weights_are_probabilities_and_high_confidence_counts_more():
    P = fuse.Params(weight_mode="cal", refine_rounds=1)
    rng = np.random.default_rng(3)
    words = [f"w{i}" for i in range(80)]
    hs = []
    for k in range(6):
        ws = list(words)
        for i in rng.choice(80, 14, replace=False):
            ws[i] = f"bad{k}_{i}"
        h = mk(k, f"0{k % 3 + 1}", ENGINES[k % 4], ws, dt=0.3, conf=None)
        for t in h.toks:       # the engine's confidence is informative: wrong words get a low probability
            t.conf = 0.3 if t.norm.startswith("bad") else 0.95
        hs.append(h)
    C = fuse.consensus(hs, P)
    ws_ = [t.w for h in C.hyps for t in h.toks]
    assert 0.05 <= min(ws_) and max(ws_) <= 0.98
    bad = np.mean([t.w for h in C.hyps for t in h.toks if t.norm.startswith("bad")])
    good = np.mean([t.w for h in C.hyps for t in h.toks if not t.norm.startswith("bad")])
    assert good > bad + 0.2
    assert [r.winner for r in C.results if r.winner] == words


# ------------------------------------------------------------------------------------------------ evaluation


def exact_edit(a, b):
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        nd = [i]
        for j, y in enumerate(b, 1):
            nd.append(min(d[j] + 1, nd[j - 1] + 1, d[j - 1] + (x != y)))
        d = nd
    return d[-1]


def test_banded_edit_distance_matches_the_exact_one():
    rng = np.random.default_rng(5)
    vocab = list("abcdefgh")
    for _ in range(20):
        a = list(rng.choice(vocab, 40))
        b = list(a)
        for _ in range(6):
            i = int(rng.integers(0, len(b)))
            op = rng.integers(0, 3)
            if op == 0:
                b[i] = str(rng.choice(vocab))
            elif op == 1:
                b.insert(i, str(rng.choice(vocab)))
            elif len(b) > 5:
                del b[i]
        ta = [i * 0.4 for i in range(len(a))]
        tb = [i * 0.4 for i in range(len(b))]
        assert FE.edit_distance(a, ta, b, tb, band_s=8.0) == exact_edit(a, b)


def test_loso_fused_is_closer_to_the_held_out_transcript_than_a_single_engine():
    rng = np.random.default_rng(11)
    words = [f"w{i}" for i in range(120)]
    hs, hid = [], 0
    for cam in ["01", "02", "03", "05"]:
        for e in ENGINES:
            ws = list(words)
            for i in rng.choice(120, 22, replace=False):
                ws[i] = f"x{hid}_{i}"            # each (phone, engine) makes its own errors
            hs.append(mk(hid, cam, e, ws, t0=70.0, dt=0.3))
            hid += 1
    P = fuse.Params(weight_mode="uniform", refine_rounds=1, compounds=False)
    res = FE.loso(hs, hs, P, log=lambda *a: None)
    best_single = min(res["single_engine_pwer_mean"].values())
    assert res["fused_pwer_mean"] < best_single * 0.75
    assert res["fused_beats_best_engine_mean"].split("/")[0] == res["fused_beats_best_engine_mean"].split("/")[1]
    res2 = FE.loso(hs, hs, P, cross_engine=True, log=lambda *a: None)
    assert res2["variant"] == "leave source and engine out" and res2["fused_pwer_mean"] < min(res2["single_engine_pwer_mean"].values())


def test_reliability_curve_has_a_monotone_trend():
    rng = np.random.default_rng(2)
    words = [f"w{i}" for i in range(150)]
    hs, hid = [], 0
    for cam in ["01", "02", "03", "05"]:
        for e in ENGINES:
            ws = list(words)
            for i in rng.choice(150, 30, replace=False):
                ws[i] = f"x{hid}_{i}"
            hs.append(mk(hid, cam, e, ws, t0=70.0, dt=0.3))
            hid += 1
    P = fuse.Params(weight_mode="uniform", refine_rounds=1, compounds=False)
    rc = FE.reliability_curve(hs, hs, P, log=lambda *a: None)
    top = rc["bins"][-1]
    assert top["slots_x_heldout"] > 100 and top["p_agree"] > 0.7


def test_clock_residuals_find_a_shifted_phone():
    P = fuse.Params(weight_mode="uniform", refine_rounds=1, compounds=False)
    words = [f"w{i}" for i in range(60)]
    hs, hid = [], 0
    for cam, shift in [("01", 0.0), ("02", 0.0), ("03", 0.0), ("05", 0.25)]:       # phone 05's clock is 250 ms late
        for e in ENGINES:
            hs.append(mk(hid, cam, e, words, t0=70.0 + shift, dt=0.4))
            hid += 1
    C = fuse.consensus(hs, P)
    res = FE.clock_residuals(C, min_voters=8)
    off = {c: v["offset_vs_other_phones_ms"] for c, v in res["per_phone"].items()}
    assert off["05"] > 150 and abs(off["01"]) < 120


# ------------------------------------------------------------------------------------------------ post-processing


def word(text, t0, t1, punct="", share=1.0, filler=False, slot=0):
    return {"text": text, "norm": text.lower(), "start": t0, "end": t1, "punct": punct, "share": share, "margin": 1.0, "null_share": 0.0, "tally": [], "hyps": 5, "phones": 3,
            "engines": 3, "eligible_phones": 3, "t_spread": 0.0, "filler": filler, "flags": [], "slot": slot}


def test_render_capitalises_after_sentence_ends_and_places_events():
    ws = [word("hello", 1.0, 1.4, "!"), word("this", 3.6, 3.8), word("is", 3.8, 3.9), word("um", 4.0, 4.2, filler=True), word("kadu", 4.3, 4.6, ".")]
    ev = [{"t": 2.0, "token": FP.LAUGH}]
    txt = FP.render(ws, ev)
    assert txt == f"Hello! {FP.LAUGH} This is um kadu."
    assert FP.render(ws, ev, clean=True) == "Hello! This is kadu."
    assert FP.sentence_breaks(ws) == [0, 4]


def test_assign_speakers_snaps_the_boundary_to_the_silence_between_words():
    ws = [word("one", 10.0, 10.3), word("two", 10.4, 10.7), word("three", 10.8, 11.1), word("four", 12.0, 12.3), word("five", 12.4, 12.7)]
    diar = {"turns": [{"start": 0.0, "end": 10.6, "voice": 1, "name": "Larissa", "confidence": 1.0}, {"start": 11.4, "end": 20.0, "voice": 0, "name": "Kadu", "confidence": 0.9}]}
    spk = FP.assign_speakers(ws, diar, 0.85)
    assert [s["voice"] for s in spk] == [1, 1, 1, 0, 0]               # the cut is the 0.9 s pause, not the nominal boundary at 11.0 s
    assert spk[4]["speaker_conf"] == pytest.approx(0.9 * 0.85, abs=1e-3)
    assert not any(s["speaker_edge"] for s in spk)


def test_interjection_moves_only_short_or_backchannel_words():
    ws = [word("careful", 5.0, 5.4), word("with", 5.5, 5.7), word("this", 5.8, 6.0), word("yes", 7.0, 7.3), word("go", 7.4, 7.6)]
    diar = {"turns": [{"start": 0.0, "end": 20.0, "voice": 0, "name": "Kadu", "confidence": 1.0}]}
    spk = FP.assign_speakers(ws, diar, 0.85)
    FP.apply_interjections(ws, spk, [{"start": 4.9, "end": 6.1, "voice": 1, "margin": 0.4}, {"start": 6.9, "end": 7.7, "voice": 1, "margin": 0.4}], 0.85, {0: "Kadu", 1: "Larissa"})
    assert [s["voice"] for s in spk[:3]] == [0, 0, 0]                 # three content words overlapping the window stay with the base voice ...
    assert "overlaps_other_voice" in ws[0]["flags"] and ws[0]["maybe_voice"]["name"] == "Larissa"
    assert spk[3]["voice"] == 1 and spk[3]["interjection"]           # ... a window with at most two words moves them
    assert spk[4]["voice"] == 1


def test_glossary_hits_and_known_confusions():
    gl = {("counterbalance",): {"canonical": "counterbalance", "kind": "technique", "weight": 95, "variant": False, "form": "counterbalance", "key": "counterbalance"},
          ("weight", "change"): {"canonical": "weight change", "kind": "technique", "weight": 95, "variant": False, "form": "weight change", "key": "weight change"}}
    ws = [word(x, i, i + 0.3) for i, x in enumerate("we do the lights engage and a weight change with counterbalance".split())]
    hits = FP.glossary_hits(ws, gl)
    assert {h["term"] for h in hits} == {"counterbalance", "weight change"}
    sus = FP.suspects(ws)
    assert any(s["probably"] == "legs engaged" for s in sus)


def test_events_join_the_laugh_of_a_joke_candidate_and_keep_a_lone_joke_token():
    diar = {"emoji_tokens": [{"t": 2.0, "joke": "j1", "alt_t": 5.08, "evidence": "e1"}, {"t": 6.8, "joke": "j2", "alt_t": 6.8, "evidence": "e2"}]}
    em = {"events": [{"id": "ev000", "kind": "laughter", "t0": 2.0, "t1": 4.0, "peak": 0.13, "cams": {"01": 0.13, "02": 0.09}}]}
    evs = FP.gather_events(em, diar)
    assert [e["id"] for e in evs] == ["ev000", "jk01"]
    assert evs[0]["needs_owner_confirmation"] and evs[0]["alt_t"] == 5.08 and evs[0]["token"] == "\U0001F602"
    assert evs[1]["type"] == "joke_reaction_candidate" and evs[1]["t"] == 6.8


def test_wordless_events_find_loud_stretches_without_words():
    cells = 800
    lvl = np.full((3, cells), -50.0)
    flo = np.full((3, cells), -50.0)
    lvl[:, 400:440] = -30.0                                           # 2 s of sound on all three phones, cells of 50 ms -> m = 20..22
    lv = SimpleNamespace(m_lo=0.0, m_hi=40.0, grid=0.05, lvl_s=lvl, flo=flo, cams=["a", "b", "c"])
    quiet_word = [word("x", 5.0, 5.3)]
    out = FP.wordless_events(lv, quiet_word)
    assert len(out) == 1 and 19.9 <= out[0]["t0"] <= 20.2 and 21.8 <= out[0]["t1"] <= 22.2 and out[0]["phones"] == 3
    assert FP.wordless_events(lv, [word("y", 20.0, 22.0)]) == []       # a word explains the sound


def test_joke_scan_flags_introductions_and_hair_remarks():
    sents = [{"start": 3.0, "end": 5.0, "text": "This is Larissa.", "text_clean": "This is Larissa."}, {"start": 6.0, "end": 7.0, "text": "I can see my hair.", "text_clean": "I can see my hair."},
             {"start": 8.0, "end": 9.0, "text": "Then we turn.", "text_clean": "Then we turn."}]
    turns = [{"id": "t0", "speaker": "Larissa", "voice": 1, "sentences": sents[:1]}, {"id": "t1", "speaker": "Kadu", "voice": 0, "sentences": sents[1:]}]
    out = FP.joke_scan(turns, {}, {}, [])
    kinds = {(o["turn"], k) for o in out for k in o["kinds"]}
    assert ("t0", "self_introduction") in kinds and ("t1", "self_appearance") in kinds
    assert all(o["start"] != 8.0 for o in out)


# ------------------------------------------------------------------------------------------------ batches


def test_split_batches_sizes_and_coverage():
    P = FB.BatchParams()
    for n in (1, 10, 15, 41, 100, 178, 300, 480):
        spans = [{"start": float(i), "end": i + 0.5} for i in range(n)]
        bs = FB.split_batches(spans, P)
        assert sum(len(b) for b in bs) == n
        if n >= P.per_batch[0]:
            assert all(P.per_batch[0] <= len(b) <= P.per_batch[1] for b in bs)
        flat = [s["start"] for b in bs for s in b]
        assert flat == sorted(flat)
    assert len(FB.split_batches([{"start": 0, "end": 1}] * 178, P)) == 9


def test_flag_slots_reasons_and_span_grouping():
    P = fuse.Params(weight_mode="uniform", refine_rounds=1, compounds=False)
    base = "alpha beta gamma delta epsilon zeta eta theta".split()
    hs = [mk(i, f"0{i % 3 + 1}", ENGINES[i % 4], list(base)) for i in range(8)]
    for i in (0, 1, 2):                                                # three hypotheses disagree on "gamma" and "delta"
        hs[i] = mk(i, f"0{i % 3 + 1}", ENGINES[i % 4], ["alpha", "beta", "gammas", "deltas", "epsilon", "zeta", "eta", "theta"])
    C = fuse.consensus(hs, P)
    words = [{"slot": r.idx, "norm": r.winner, "i": k} for k, r in enumerate(x for x in C.results if x.winner)]
    BP = FB.BatchParams()
    flags = FB.flag_slots(C, words, [], {}, BP, [])
    flagged_words = {C.results[k].winner for k in flags}
    assert flagged_words == {"gamma", "delta"}
    runs = FB.group_spans(flags, C, 0, 14)
    assert len(runs) == 1 and len(runs[0]) == 2                        # adjacent flagged slots are one span
    assert all(r.startswith(("low_share", "low_margin", "null_contest")) for k in flags for r in flags[k])


def test_readings_group_hypotheses_by_text_with_sources():
    P = fuse.Params(weight_mode="uniform", refine_rounds=1, compounds=False)
    base = "alpha beta gamma delta".split()
    hs = [mk(i, f"0{i % 3 + 1}", ENGINES[i % 4], list(base)) for i in range(6)]
    hs[0] = mk(0, "01", ENGINES[0], ["alpha", "beta", "gammas", "delta"], conf=0.6)
    hs[1] = mk(1, "02", ENGINES[1], ["alpha", "beta", "gammas", "delta"], conf=0.7)
    C = fuse.consensus(hs, P)
    k = next(r.idx for r in C.results if r.winner == "gamma")
    rd = FB.readings_for(C, [k])
    assert rd[0]["norm"] == "gamma" and rd[0]["n_sources"] == 4 and rd[1]["norm"] == "gammas" and rd[1]["n_sources"] == 2
    assert rd[1]["engines"] == ["wl3", "wl3t"] and len(rd[1]["sources"]) == 2
    assert any(src.startswith("01/wl3 0.6") for src in rd[1]["sources"]) and any(src.startswith("02/wl3t 0.7") for src in rd[1]["sources"])
    assert abs(sum(r["weight_share"] for r in rd) - 1.0) < 0.02


def test_owner_facts_file_is_optional(tmp_path):
    assert FB.load_owner_facts(tmp_path / "missing.json") == ([], [])
    p = tmp_path / "f.json"
    p.write_text('{"facts": [{"id": "a", "text": "t", "windows": [{"name": "w1", "start": 1.0, "end": 2.0}]}]}', encoding="utf-8")
    facts, wins = FB.load_owner_facts(p)
    assert facts[0]["id"] == "a" and wins == [(1.0, 2.0, "w1")]
    assert FB.owner_key(1.5, wins) == "w1" and FB.owner_key(2.0, wins) is None


# ------------------------------------------------------------------------------------------------ levels


def test_frame_db_and_levels_features():
    sr = 16000
    t = np.arange(sr * 4) / sr
    x = np.where((t > 1.0) & (t < 2.0), 0.1 * np.sin(2 * np.pi * 300 * t), 0.001 * np.sin(2 * np.pi * 50 * t)).astype(np.float32)
    db = FL.frame_db(x)
    assert db[150] > -25 and db[10] < -55                              # 10 ms frames: loud at 1.5 s, quiet at 0.1 s
    floor = FL.rolling_floor(db, 0.01, half_s=2.0)
    assert floor[150] < db[150] - 20

    class Clk:
        def to_source(self, m):
            return m

    ph = FL.PhoneLevel("a", db, Clk())
    ph2 = FL.PhoneLevel("b", db - 6.0, Clk())
    lv = FL.Levels({"a": ph, "b": ph2}, 0.0, 4.0)
    snr, rel = lv.feats("a", 1.2, 1.8)
    snr_b, rel_b = lv.feats("b", 1.2, 1.8)
    assert snr > 20 and rel == pytest.approx(0.0, abs=0.5) and rel_b == pytest.approx(-6.0, abs=0.8)
    assert lv.feats("zz", 1.0, 2.0) == (10.0, 0.0)                     # unknown phone: neutral
