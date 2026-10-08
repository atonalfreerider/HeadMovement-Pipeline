"""Narration speech recognition: Whisper engine selection (never SAPI by itself), the bilingual (en + pt) transcript
format and fusion, the worker's region remapping, and phrase -> move mapping for English speech with Portuguese and
English move names.

    ..\\.venv-nopo4d\\Scripts\\python -m pytest tests/test_narration_asr.py -q

Offline and GPU-free, except the two smoke tests (DANCECAP_ASR_SMOKE=1, ~2 min): test_whisper_end_to_end_smoke
synthesises "lateral / corredor / vira jeenya / boomerang" with Windows SAPI and transcribes it with the real Whisper
engine; test_whisper_portuguese_names_smoke speaks 24 Portuguese move names with IPA pronunciations (sapi_synth.ps1
'ipa' lines) and checks large-v3 and turbo against the real graph.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dancecap.moves.annotate import asr  # noqa: E402
from dancecap.moves.annotate import asr_worker as W  # noqa: E402
from dancecap.moves.annotate.pipeline import NarrationSettings, TimelineContext, keep_previous, map_and_build, \
    process, transcript_tag  # noqa: E402
from dancecap.moves.narration import Lexicon, map_utterances, phon, pt_respell  # noqa: E402

HERE = Path(__file__).resolve().parent
SAPI_SYNTH = HERE.parent / "dancecap" / "moves" / "annotate" / "sapi_synth.ps1"

# a small vocabulary shaped like taxonomy/zouk_graph.json: Brazilian Portuguese display names + English names/aliases
MOVES = [
    {"slug": "lateral", "name": "Corredor", "name_en": "Lateral", "aliases": ["Lateral", "corridor", "Passo Lateral"]},
    {"slug": "viradinha", "name": "Viradinha", "name_en": "Viradinha", "aliases": ["veraginha", "Little Turn"]},
    {"slug": "boomerang", "name": "Bumerangue", "name_en": "Boomerang", "aliases": ["Boomerang", "Bumerang"]},
    {"slug": "dip-back", "name": "Cambré", "name_en": "Dip Back", "aliases": ["Dip Back", "backbend"]},
    {"slug": "yoyo", "name": "Ioiô", "name_en": "Yoyo", "aliases": ["Yoyo", "yo-yo"]},
    {"slug": "simple-turn", "name": "Giro Simples", "name_en": "Simple Turn", "aliases": ["Simple Turn", "giro"]},
    {"slug": "counterbalance-pivot", "name": "Contrapeso", "name_en": "Counterbalance Pivot",
     "aliases": ["counterbalance", "Contrapeso"]},
    {"slug": "hold-behind-crossed-arms", "name": "Cestinha", "name_en": "Hold Behind Crossed Arms", "aliases": []},
    {"slug": "switch-hands", "name": "Troca de Mãos", "name_en": "Switch Hands", "aliases": ["Switch Hands"]},
    {"slug": "toalha", "name": "Toalha", "aliases": ["towel"]},
    {"slug": "chicote", "name": "Chicote", "aliases": ["whip"]},
    {"slug": "sarrada", "name": "Sarrada", "name_en": "Sarrada", "aliases": ["cicada"]},
    {"slug": "bonus", "name": "Bônus", "name_en": "Bonus", "aliases": ["Bonus"]},
    {"slug": "open-break", "name": "Open Break", "name_en": "Open Break"},
    {"slug": "forward-basic", "name": "Forward Basic", "name_en": "Forward Basic"},
    {"slug": "body-roll", "name": "Ondulação", "name_en": "Body Roll", "aliases": ["Body Roll"]},
    {"slug": "turn", "name": "Turn", "name_en": "Turn"},
    {"slug": "pirouette", "name": "Giro no Eixo", "name_en": "Pirouette", "aliases": ["Pirouette", "pirueta"]},
    {"slug": "tilt-left", "name": "Inclinação para a Esquerda", "name_en": "Tilt Left", "aliases": ["Tilt Left"]},
    {"slug": "piao", "name": "Pião", "aliases": ["spinning top"]},
]


@pytest.fixture(scope="module")
def lex():
    return Lexicon(MOVES)


def words_of(text: str, t0: float = 0.0, prob: float = 0.95, step: float = 0.4) -> list[dict]:
    return [{"word": w, "start": round(t0 + i * step, 3), "end": round(t0 + i * step + 0.3, 3), "prob": prob}
            for i, w in enumerate(text.split())]


def utt(text: str, t0: float = 0.0, prob: float = 0.95) -> dict:
    ws = words_of(text, t0, prob)
    return {"start": ws[0]["start"], "end": ws[-1]["end"], "text": text, "words": ws}


def best(lex, phrase):
    c = lex.candidates(phrase)
    return (c[0].move, c[0].score, c[0].method) if c else (None, 0.0, None)


# ---------------------------------------------------------------------------------------------------------------
# phrase -> move
# ---------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("phrase,move", [
    ("Corredor", "lateral"), ("corridor", "lateral"), ("lateral", "lateral"), ("passo lateral", "lateral"),
    ("Viradinha", "viradinha"), ("little turn", "viradinha"), ("vira dinha", "viradinha"),
    ("Bumerangue", "boomerang"), ("boomerang", "boomerang"), ("bumerangi", "boomerang"), ("boomerangy", "boomerang"),
    ("Cambré", "dip-back"), ("cambre", "dip-back"), ("dip back", "dip-back"),
    ("Ioiô", "yoyo"), ("ioio", "yoyo"), ("yoyo", "yoyo"), ("yo yo", "yoyo"),
    ("giro simples", "simple-turn"), ("simple turn", "simple-turn"),
    ("contrapeso", "counterbalance-pivot"), ("contra peso", "counterbalance-pivot"),
    ("kontrapeso", "counterbalance-pivot"), ("counterbalance", "counterbalance-pivot"),
    ("cestinha", "hold-behind-crossed-arms"), ("cestinya", "hold-behind-crossed-arms"),
    ("troca de mãos", "switch-hands"), ("troca de maos", "switch-hands"), ("switch hands", "switch-hands"),
    ("toalha", "toalha"), ("toalia", "toalha"), ("towel", "toalha"),
    ("chicote", "chicote"), ("shikoti", "chicote"),
    ("sarrada", "sarrada"), ("cicada", "sarrada"),
    ("ondulação", "body-roll"), ("ondulacao", "body-roll"), ("body roll", "body-roll"), ("bodyroll", "body-roll"),
    ("pirueta", "pirouette"), ("giro no eixo", "pirouette"),
])
def test_phrase_names_move(lex, phrase, move):
    ev = map_utterances([utt(phrase)], lex)
    moves = [e for e in ev if e["kind"] == "move"]
    assert moves and moves[0]["move"] == move, (phrase, ev)
    assert moves[0]["status"] in ("matched", "close") and moves[0]["confidence"] >= 0.7, moves[0]
    assert moves[0]["candidates"] and moves[0]["candidates"][0]["move"] == move


@pytest.mark.parametrize("phrase", ["nice", "beautiful", "watch this", "look at her", "the music", "camera",
                                    "okay now", "spinning", "basic", "walia", "vigilante", "hero simples",
                                    "come bray", "shadow position", "leg lift"])
def test_unclear_or_unrelated_phrases_are_not_accepted(lex, phrase):
    for e in map_utterances([utt(phrase)], lex):
        assert e["kind"] != "move" or e["status"] not in ("matched", "close"), (phrase, e)


def test_unclear_phrase_keeps_candidates(lex):
    ev = map_utterances([utt("walia")], lex)[0]  # Whisper's rendering of "toalha" on the real narration
    assert ev["kind"] == "unknown" and ev["candidates"][0]["move"] == "toalha"
    ev = map_utterances([utt("vira genia")], lex)[0]
    assert ev["kind"] == "unknown" and ev["candidates"][0]["move"] == "viradinha"


def test_short_portuguese_name_heard_as_english_spelling(lex):
    """large-v3 wrote a Brazilian "Pião" as "Piau": a 4-letter Portuguese name still matches phonetically."""
    ev = map_utterances([utt("Piau,")], lex)[0]
    assert ev["kind"] == "move" and ev["move"] == "piao", ev
    ev = map_utterances([utt("piano")], lex)[0]          # a look-alike English word: a candidate, never accepted
    assert ev["kind"] == "unknown" and ev["candidates"][0]["move"] == "piao"


def test_portuguese_respelling_and_phonetic_form():
    assert "virajinia" in pt_respell("Viradinha") and "toalia" in pt_respell("Toalha")
    assert "bumerangi" in pt_respell("Bumerangue") and "piaun" in pt_respell("Pião")
    assert phon("Walia") == "ualia" and phon("shikochi") == "xikoxi" and phon("how") == "ou"


def test_whisper_punctuated_list(lex):
    """Whisper writes a narrated list with commas: phrases never span them; items are whole phrases."""
    ws = [{"word": w, "start": 1.0 + i, "end": 1.4 + i, "prob": 0.9} for i, w in
          enumerate(["Open", "Break,", "Viradinha,", "Corridor,", "Tilted", "Turn,", "Shadow", "Position,"])]
    ev = map_utterances([{"start": 1.0, "end": 9.0, "words": ws}], lex)
    got = [(e["kind"], e.get("move"), e.get("status"), e["phrase"]) for e in ev]
    assert got[0] == ("move", "open-break", "matched", "open break")
    assert got[1] == ("move", "viradinha", "matched", "viradinha")
    assert got[2] == ("move", "lateral", "matched", "corridor")
    assert got[3][:4] == ("move", "turn", "partial", "tilted turn")       # names more than the graph's move
    assert ev[3]["confidence"] <= 0.5 and ev[3]["leftover"] == ["tilted"] and ev[3]["a0"] == pytest.approx(5.0)
    assert got[4] == ("unknown", None, "unknown", "shadow position")      # one unknown item, not "position"
    assert len(ev) == 5


def test_long_punctuated_list_with_unknown_items(lex):
    ws = [{"word": w, "start": float(i), "end": i + 0.5, "prob": 0.9} for i, w in
          enumerate(["Corredor,", "xyzzy,", "Viradinha,", "qwerty", "plugh,", "Open", "Break,", "Bumerangue."])]
    ev = map_utterances([{"start": 0.0, "end": 8.0, "words": ws}], lex)
    assert [(e["kind"], e.get("move")) for e in ev] == [("move", "lateral"), ("unknown", None), ("move", "viradinha"),
                                                        ("unknown", None), ("move", "open-break"),
                                                        ("move", "boomerang")]
    assert ev[3]["phrase"] == "qwerty plugh" and ev[3]["a0"] == pytest.approx(3.0)


def test_portuguese_and_counting_fillers(lex):
    ev = map_utterances([utt("agora o corredor"), utt("e depois a viradinha", 3.0), utt("1, 2, 3, 4", 6.0)], lex)
    assert [(e["kind"], e.get("move")) for e in ev] == [("move", "lateral"), ("move", "viradinha")]


def test_whisper_alternative_from_other_language(lex):
    """The fused utterance names no move, the other language pass does: used, confidence <= its conf_cap."""
    u = utt("very genia", 0.0, 0.6)
    u["alternatives"] = [{"text": "Viradinha", "prob": 0.9, "words": words_of("Viradinha", 0.0, 0.9),
                          "kind": "whisper-pt", "conf_cap": 0.8}]
    ev = map_utterances([u], lex)[0]
    assert ev["kind"] == "move" and ev["move"] == "viradinha" and ev["method"] == "alternative"
    assert 0.35 < ev["confidence"] <= 0.8


# ---------------------------------------------------------------------------------------------------------------
# engine selection: Whisper, never SAPI by itself
# ---------------------------------------------------------------------------------------------------------------

def _av(whisper: bool, cached=("large-v3", "large-v3-turbo"), sapi: bool = True) -> dict:
    fw = {"available": whisper, "python": "py-asr" if whisper or cached == () else None, "model": "large-v3",
          "models_cached": list(cached), "model_cached": "large-v3" in cached, "cuda_devices": 1}
    if not whisper:
        fw.update(reason="faster-whisper is not installed", fix=asr.setup_help())
    av = {"faster-whisper": fw, "sapi": {"available": sapi, "english_only": True}}
    av["auto"] = "faster-whisper" if whisper else None
    av["summary"] = asr.engine_summary(av)
    return av


def test_auto_picks_whisper_never_sapi():
    av = _av(True)
    assert asr.resolve("auto", av) == ("faster-whisper", "large-v3")
    assert asr.resolve(None, av) == ("faster-whisper", "large-v3")
    assert asr.resolve("faster-whisper", av) == ("faster-whisper", "large-v3")
    assert asr.resolve("faster-whisper:large-v3-turbo", av) == ("faster-whisper", "large-v3-turbo")
    assert asr.resolve("sapi", av) == ("sapi", None)                    # only when asked for by name
    assert "Whisper large-v3" in av["summary"] and "GPU" in av["summary"]


def test_whisper_unavailable_is_an_error_with_the_fix_not_sapi():
    av = _av(False, cached=())
    av["faster-whisper"]["python"] = None
    with pytest.raises(asr.ASRError) as e:
        asr.resolve("auto", av)
    msg = str(e.value)
    assert "Whisper is unavailable" in msg and "pip install" in msg and ".venv-asr" in msg
    assert "UNAVAILABLE" in av["summary"] and "never used automatically" in av["summary"]
    # installed, but the model was not downloaded
    av2 = _av(True, cached=("large-v3-turbo",))
    with pytest.raises(asr.ASRError, match="large-v3 is not downloaded"):
        asr.resolve("auto", av2)
    assert asr.resolve("faster-whisper:large-v3-turbo", av2) == ("faster-whisper", "large-v3-turbo")
    with pytest.raises(asr.ASRError, match="unknown speech engine"):
        asr.resolve("vosk", av)


def test_transcribe_dispatch(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(asr, "transcribe_whisper", lambda wav, lex, info, model, gpu_wait, log=print:
                        calls.append(("whisper", model)) or {"engine": "faster-whisper"})
    monkeypatch.setattr(asr, "transcribe_sapi", lambda wav, lex: calls.append(("sapi", None)) or {"engine": "sapi"})
    monkeypatch.setattr(asr, "engines", lambda refresh=False, model=None: _av(True))
    assert asr.transcribe(tmp_path / "a.wav", "auto")["engine"] == "faster-whisper"
    assert asr.transcribe(tmp_path / "a.wav", "faster-whisper:large-v3-turbo")["engine"] == "faster-whisper"
    assert calls == [("whisper", "large-v3"), ("whisper", "large-v3-turbo")]
    monkeypatch.setattr(asr, "engines", lambda refresh=False, model=None: {**_av(False, cached=()),
                                                                            "sapi": {"available": True}})
    calls.clear()
    with pytest.raises(asr.ASRError):
        asr.transcribe(tmp_path / "a.wav", "auto")
    assert calls == []                                                   # no silent SAPI fallback
    assert asr.transcribe(tmp_path / "a.wav", "sapi")["engine"] == "sapi" and calls == [("sapi", None)]


def test_hotwords_prioritise_portuguese_names_and_skip_misspellings(lex):
    hw = asr.hotwords(lex)
    terms = [t.strip() for t in hw.split(",")]
    assert terms[:3] == ["Corredor", "Viradinha", "Bumerangue"]
    assert {"Cambré", "Ioiô", "Contrapeso", "Toalha", "Giro", "Simples", "Pirouette"} <= set(terms)
    assert not {"veraginha", "cicada", "Veraginha", "Cicada"} & set(terms)  # learned misspellings never prompted


# ---------------------------------------------------------------------------------------------------------------
# worker: VAD regions joined with gaps and mapped back
# ---------------------------------------------------------------------------------------------------------------

def test_worker_gap_concatenation_and_remap():
    sr = W.SR
    audio = np.zeros(20 * sr, np.float32)
    regions = [(2 * sr, 3 * sr), (10 * sr, 11 * sr), (15 * sr, int(16.5 * sr))]
    cat, table = W.concat_with_gaps(audio, regions, 1.0)
    assert cat.size == (1 + 1 + 1.5 + 2 * 1.0) * sr and [t[2] for t in table] == [2 * sr, 10 * sr, 15 * sr]
    # Whisper's word times in the joined audio; the 2nd word starts in the gap before region 2 (as Whisper does)
    words = [{"word": "Corredor,", "start": 0.1, "end": 0.8}, {"word": "Viradinha,", "start": 1.5, "end": 2.9},
             {"word": "Toalha,", "start": 4.1, "end": 4.6}]
    segs = [{"start": 0.0, "end": 4.6}]
    W.remap_pass(words, segs, table)
    assert (words[0]["start"], words[0]["end"]) == pytest.approx((2.1, 2.8))
    assert (words[1]["start"], words[1]["end"]) == pytest.approx((10.0, 10.9))   # start clipped to its region
    assert (words[2]["start"], words[2]["end"]) == pytest.approx((15.1, 15.6))
    assert segs[0]["start"] == pytest.approx(2.0) and segs[0]["end"] == pytest.approx(15.6)
    quiet, g = W.normalise_gain(np.full(100, 0.01, np.float32))
    assert g == pytest.approx(30.0) and np.abs(quiet).max() < 0.4               # +30 dB at most, never clipping
    loud, g = W.normalise_gain(np.full(100, 0.5, np.float32))
    assert g == 0.0 and loud.max() == pytest.approx(0.5)


# ---------------------------------------------------------------------------------------------------------------
# transcript format + fusion of the en / pt passes
# ---------------------------------------------------------------------------------------------------------------

def _w(word, s, e, p, seg=0):
    return {"word": word, "start": s, "end": e, "prob": p, "seg": seg}


def fake_worker_output() -> dict:
    regions = [{"start": 1.0, "end": 2.0}, {"start": 5.0, "end": 6.0}, {"start": 9.0, "end": 10.0},
               {"start": 13.0, "end": 14.2}, {"start": 30.0, "end": 31.0}]
    en = [_w("Open", 1.0, 1.4, 0.99), _w("Break,", 1.4, 1.8, 0.98),
          _w("very", 5.0, 5.3, 0.4), _w("genia,", 5.3, 5.8, 0.3),
          _w("Bônus,", 9.1, 9.6, 0.33),
          _w("Simple", 13.0, 13.4, 0.3), _w("Turn,", 13.4, 13.8, 0.7),
          _w("Thank", 30.0, 30.3, 0.2, seg=1), _w("you.", 30.3, 30.6, 0.2, seg=1)]
    pt = [_w("Open", 1.0, 1.4, 0.97), _w("Break,", 1.4, 1.8, 0.96),
          _w("Viradinha,", 5.0, 5.8, 0.9),
          _w("Bônus,", 13.0, 13.2, 0.86), _w("Bônus,", 13.25, 13.4, 0.5), _w("Simple", 13.4, 13.6, 0.9),
          _w("Turn,", 13.6, 14.0, 0.9)]
    seg_ok = {"start": 0, "end": 20, "text": "", "avg_logprob": -0.2, "no_speech_prob": 0.05}
    seg_bad = {"start": 30, "end": 31, "text": "Thank you.", "avg_logprob": -1.4, "no_speech_prob": 0.8}
    return {"engine": "faster-whisper", "model": "large-v3", "device": "cuda", "compute_type": "float16",
            "duration": 40.0, "gain_db": 0.0, "load_s": 10.0, "seconds": 25.0, "segmentation": "gaps",
            "vram_used_mib": {"before_load": 4500, "after_load": 8400, "peak_sampled": 8600},
            "hotwords": "Corredor, Viradinha", "regions": regions,
            "passes": [{"language": "en", "requested": "en", "seconds": 7.0, "words": en,
                        "segments": [seg_ok, seg_bad]},
                       {"language": "pt", "requested": "pt", "seconds": 7.0, "words": pt,
                        "segments": [seg_ok]}]}


def test_transcript_format_and_fusion(lex):
    t = asr.build_transcript(fake_worker_output(), lex, "guest", ["w"], 31.5, 4510)
    assert t["version"] == 2 and t["engine"] == "faster-whisper" and t["model"] == "large-v3"
    assert t["language"] == "en+pt" and t["languages"] == ["en", "pt"] and t["gpu"] == "guest"
    for k in ("words", "utterances", "raw", "fusion", "timing", "vram_mib", "warnings", "created", "duration"):
        assert k in t
    assert [p["language"] for p in t["raw"]["passes"]] == ["en", "pt"]          # the raw recogniser output is kept
    assert "Thank you." in t["raw"]["passes"][0]["text"] and t["vram_mib"]["after_exit"] == 4510
    for w in t["words"]:
        assert set(w) >= {"word", "start", "end", "prob", "lang"}
    texts = [(u["text"], u["lang"]) for u in t["utterances"]]
    assert texts[0] == ("Open Break,", "en")                 # same words: English prior
    assert texts[1] == ("Viradinha,", "pt")                  # the Portuguese pass names the move
    assert all("Thank" not in x for x, _ in texts)           # Whisper's no-speech rule drops the hallucination
    alt = t["utterances"][1]["alternatives"][0]
    assert alt["text"] == "very genia," and alt["kind"] == "whisper-en" and alt["conf_cap"] == 0.8
    # "Bônus" heard once by each pass but put in neighbouring utterances, plus a "Bônus, Bônus" decoder repetition:
    # one bonus in the fused transcript
    assert sum(normal == "bonus" for normal in (w["word"].strip(",").lower().replace("ô", "o")
                                                for w in t["words"])) == 1
    assert t["fusion"]["chosen"] == {"en": 1, "pt": 2}
    # through the mapping: a result with the recogniser info for the UI
    res = map_and_build(t, {"entries": [{"a": 0.0, "v": 100.0, "cam": "c", "rate": 1.0, "paused": False,
                                         "ev": "start"}]}, TimelineContext(), lex, NarrationSettings(snap="none"))
    assert res["asr"] == {"engine": "faster-whisper", "model": "large-v3", "device": "cuda", "gpu": "guest",
                          "language": "en+pt", "seconds": 31.5}
    assert [e["move"] for e in res["events"] if e["kind"] == "move"] == ["open-break", "viradinha", "bonus",
                                                                         "simple-turn"]


def test_hyphen_split_name_is_not_a_decoder_repetition(lex):
    """turbo wrote a synthesised "Ioiô" as `Yo` + `-Yo,` 0.2 s apart in both passes: one yoyo, not "yo"; a real
    "Bônus, Bônus" decoder repetition is still one bonus."""
    regions = [{"start": 1.0, "end": 2.0}, {"start": 5.0, "end": 6.0}]
    seg = {"start": 0, "end": 10, "text": "", "avg_logprob": -0.2, "no_speech_prob": 0.05}
    en = [_w("Yo", 1.0, 1.2, 0.6), _w("-Yo,", 1.4, 1.7, 0.7), _w("Bônus,", 5.0, 5.3, 0.8),
          _w("Bônus,", 5.4, 5.6, 0.5)]
    pt = [_w("Yo", 1.0, 1.2, 0.5), _w("-Yo,", 1.4, 1.7, 0.6)]
    out = {"engine": "faster-whisper", "model": "large-v3-turbo", "device": "cuda", "duration": 8.0,
           "regions": regions, "passes": [{"language": "en", "requested": "en", "words": en, "segments": [seg]},
                                          {"language": "pt", "requested": "pt", "words": pt, "segments": [seg]}]}
    t = asr.build_transcript(out, lex, "lock", [], 5.0, None)
    assert [u["text"] for u in t["utterances"]] == ["Yo -Yo,", "Bônus,"]
    ev = map_utterances(t["utterances"], lex)
    assert [(e["kind"], e.get("move")) for e in ev] == [("move", "yoyo"), ("move", "bonus")]
    # the rule itself
    a, b = {"word": "Yo", "start": 1.0, "end": 1.2}, {"word": "Yo", "start": 1.4, "end": 1.7}
    assert asr._decoder_repeat(a, b, None) and not asr._decoder_repeat(a, b, lex)       # "yo yo" is a term
    assert not asr._decoder_repeat(a, {**b, "word": "-Yo,"}, None)                       # hyphen-split word
    bo = {"word": "Bônus,", "start": 1.0, "end": 1.2}
    assert asr._decoder_repeat(bo, {**bo, "start": 1.3, "end": 1.5}, lex)
    assert not asr._decoder_repeat(bo, {**bo, "start": 1.8, "end": 2.0}, lex)            # >= 0.5 s apart
    vv = {"word": "very", "start": 1.0, "end": 1.2}
    assert not asr._decoder_repeat(vv, {**vv, "start": 1.3, "end": 1.5}, lex)            # not a move name


def test_no_speech_transcript(lex):
    out = {"engine": "faster-whisper", "model": "large-v3", "device": "cuda", "duration": 19.2, "regions": [],
           "passes": [{"language": "en", "requested": "en", "words": [], "segments": []},
                      {"language": "pt", "requested": "pt", "words": [], "segments": []}]}
    t = asr.build_transcript(out, lex, "lock", [], 12.0, None)
    assert t["words"] == [] and t["utterances"] == [] and any("no speech" in w for w in t["warnings"])


def test_retranscribe_keeps_previous_engine_files(tmp_path, lex, monkeypatch):
    nd = tmp_path / "n1"
    nd.mkdir()
    import wave
    with wave.open(str(nd / "audio.wav"), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(16000)
        w.writeframes(b"\0\0" * 16000)
    (nd / "log.json").write_text(json.dumps({"entries": [{"a": 0.0, "v": 0.0, "cam": "c", "rate": 1.0,
                                                          "paused": False, "ev": "start"}]}))
    old_t = json.dumps({"version": 1, "engine": "sapi", "words": [], "utterances": []})
    (nd / "transcript.json").write_text(old_t)
    (nd / "result.json").write_text('{"engine": "sapi"}')
    monkeypatch.setattr(asr, "transcribe", lambda wav, engine, lex, log=print:
                        asr.build_transcript(fake_worker_output(), lex, "lock", [], 1.0, None))
    res = process(nd, TimelineContext(), lex, NarrationSettings(), retranscribe=True, log=lambda *a: None)
    assert res["engine"] == "faster-whisper"
    assert (nd / "transcript.sapi.json").read_text() == old_t
    assert json.loads((nd / "result.sapi.json").read_text())["engine"] == "sapi"
    assert json.loads((nd / "transcript.json").read_text(encoding="utf-8"))["engine"] == "faster-whisper"
    # the same engine again: still kept, under its engine + model name; an identical copy is not made twice
    assert [p.name for p in keep_previous(nd, "faster-whisper")] == ["transcript.faster-whisper-large-v3.json",
                                                                     "result.faster-whisper-large-v3.json"]
    assert keep_previous(nd) == []


def _tr(created="2026-10-07T16:26:00+00:00", **kw) -> dict:
    return {"version": 2, "engine": "faster-whisper", "model": "large-v3", "gpu": "lock", "language": "en+pt",
            "words": [], "utterances": [], "created": created, **kw}


def _narration_dir(nd: Path) -> Path:
    import wave
    nd.mkdir()
    with wave.open(str(nd / "audio.wav"), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(16000)
        w.writeframes(b"\0\0" * 16000)
    (nd / "log.json").write_text(json.dumps({"entries": [{"a": 0.0, "v": 0.0, "cam": "c", "rate": 1.0,
                                                          "paused": False, "ev": "start"}]}))
    return nd


def test_transcript_tag():
    assert transcript_tag(_tr()) == "faster-whisper-large-v3"
    assert transcript_tag(_tr(model="large-v3-turbo")) == "faster-whisper-large-v3-turbo"
    assert transcript_tag(_tr(model="large-v3-turbo", gpu="cpu", language="en")) == \
        "faster-whisper-large-v3-turbo-cpu-en"
    assert transcript_tag({"engine": "sapi", "model": "MS-1033-80-DESK"}) == "sapi"
    assert transcript_tag({"engine": "import"}) == "import" and transcript_tag(None) == "previous"


def test_retranscribe_never_overwrites_an_earlier_whisper_transcript(tmp_path, lex, monkeypatch):
    """Re-transcribing with turbo / on the CPU (engine 'faster-whisper' too) keeps the large-v3 en+pt transcript."""
    nd = _narration_dir(tmp_path / "n2")
    best = json.dumps(_tr())
    (nd / "transcript.json").write_text(best)
    (nd / "result.json").write_text('{"r": "large-v3 result"}')
    cpu_out = fake_worker_output()
    cpu_out.update(model="large-v3-turbo", device="cpu", compute_type="int8")
    cpu_out["passes"] = cpu_out["passes"][:1]                     # the CPU fallback: English pass only
    monkeypatch.setattr(asr, "transcribe", lambda wav, engine, lex, log=print:
                        asr.build_transcript(cpu_out, lex, "cpu", [], 1.0, None))
    res = process(nd, TimelineContext(), lex, NarrationSettings(), retranscribe=True, log=lambda *a: None)
    assert (nd / "transcript.faster-whisper-large-v3.json").read_text() == best
    assert json.loads((nd / "result.faster-whisper-large-v3.json").read_text()) == {"r": "large-v3 result"}
    why = res["asr"]["retranscribe"]                              # the UI shows 're-transcribe recommended'
    assert "large-v3-turbo" in why and "CPU" in why and "English pass only" in why
    t_cpu = (nd / "transcript.json").read_text(encoding="utf-8")
    process(nd, TimelineContext(), lex, NarrationSettings(), retranscribe=True, log=lambda *a: None)
    assert (nd / "transcript.faster-whisper-large-v3-turbo-cpu-en.json").read_text(encoding="utf-8") == t_cpu
    # a second large-v3 transcript whose backup name is taken by different content: a timestamped pair
    (nd / "transcript.json").write_text(json.dumps(_tr(created="2026-10-08T09:10:11+00:00", note="second")))
    assert [p.name for p in keep_previous(nd)] == ["transcript.faster-whisper-large-v3.20261008-091011.json",
                                                   "result.faster-whisper-large-v3.20261008-091011.json"]
    assert (nd / "transcript.faster-whisper-large-v3.json").read_text() == best          # never overwritten
    assert keep_previous(nd) == []


def test_retranscribe_reason():
    assert asr.retranscribe_reason(_tr()) is None                                         # the best setup
    assert "SAPI" in asr.retranscribe_reason({"engine": "sapi", "language": "en-US"})
    assert "weaker on Portuguese" in asr.retranscribe_reason(_tr(model="large-v3-turbo"))
    assert asr.retranscribe_reason({"engine": "import"}) is None and asr.retranscribe_reason(None) is None


# ---------------------------------------------------------------------------------------------------------------
# a busy GPU: queue (not a silent CPU turbo run), 'Transcribe on CPU now', guest, policy
# ---------------------------------------------------------------------------------------------------------------

class _FakeLock:
    """gpu.exclusive stand-in: TimeoutError for the first `busy` calls (a pipeline stage holds the lock)."""

    def __init__(self, busy: int, on_wait=None):
        self.busy, self.on_wait, self.calls = busy, on_wait, []

    def __call__(self, label, timeout_s=0.0, poll_s=1.0):
        import contextlib
        self.calls.append(timeout_s)

        @contextlib.contextmanager
        def cm():
            if self.busy > 0:
                self.busy -= 1
                if self.on_wait:
                    self.on_wait()
                raise TimeoutError("GPU busy")
            yield
        return cm()


INFO = {"python": "py-asr", "model": "large-v3", "models_cached": ["large-v3", "large-v3-turbo"], "cuda_devices": 1}


@pytest.fixture
def fake_gpu(monkeypatch):
    from dancecap import gpu
    runs = []

    def worker(py, wav, model, device, compute, passes, hw, log=print, timeout=3600):
        runs.append((model, device, tuple(passes)))
        out = fake_worker_output()
        out.update(model=model, device=device)
        out["passes"] = [p for p in out["passes"] if p["requested"] in passes]
        return out
    monkeypatch.setattr(asr, "run_worker", worker)
    monkeypatch.setattr(asr, "gpu_free_min_mib", lambda *a, **k: 1000)   # too little VRAM to run beside the holder
    monkeypatch.setattr(asr, "gpu_used_mib", lambda: 3000)
    monkeypatch.setattr(asr, "_lock_holder", lambda: "4242 mamma-fit")
    monkeypatch.setattr(asr, "QUEUE_POLL_S", 0.01)
    for k in ("DANCECAP_ASR_GPU_BUSY", "DANCECAP_ASR_GPU_GUEST", "DANCECAP_ASR_GPU_QUEUE_H"):
        monkeypatch.delenv(k, raising=False)
    return gpu, runs


def test_busy_gpu_queues_the_narration(fake_gpu, monkeypatch, lex, tmp_path):
    gpu, runs = fake_gpu
    monkeypatch.setattr(gpu, "exclusive", _FakeLock(3))
    logs: list[str] = []
    t = asr.transcribe_whisper(tmp_path / "audio16k.wav", lex, INFO, None, 0.01, log=logs.append)
    assert runs == [("large-v3", "cuda", ("en", "pt"))] and t["gpu"] == "lock"     # waited, then large-v3 en+pt
    assert any("GPU queue:" in m and "mamma-fit" in m and "CPU now" in m for m in logs)
    assert any(w.startswith("queued") for w in t["warnings"]) and asr.retranscribe_reason(t) is None


def test_cpu_now_skips_the_gpu_queue(fake_gpu, monkeypatch, lex, tmp_path):
    gpu, runs = fake_gpu
    wav = tmp_path / "audio16k.wav"
    asr.skip_gpu_queue(wav)                     # a stale click from an earlier run of this recording: ignored
    lock = _FakeLock(10 ** 6, on_wait=lambda: len(lock.calls) == 3 and asr.skip_gpu_queue(wav))
    monkeypatch.setattr(gpu, "exclusive", lock)
    t = asr.transcribe_whisper(wav, lex, INFO, None, 0.01, log=lambda *a: None)
    assert len(lock.calls) == 3 and runs == [("large-v3-turbo", "cpu", asr.CPU_PASSES)] and t["gpu"] == "cpu"
    assert any("CPU now" in w for w in t["warnings"])
    why = asr.retranscribe_reason(t)
    assert "large-v3-turbo" in why and "CPU fallback" in why


def test_busy_gpu_policy_guest_and_queue_limit(fake_gpu, monkeypatch, lex, tmp_path):
    gpu, runs = fake_gpu
    wav = tmp_path / "audio16k.wav"
    monkeypatch.setattr(gpu, "exclusive", _FakeLock(10 ** 6))
    monkeypatch.setenv("DANCECAP_ASR_GPU_BUSY", "cpu")              # the old behaviour, by choice
    t = asr.transcribe_whisper(wav, lex, INFO, None, 0.01, log=lambda *a: None)
    assert t["gpu"] == "cpu" and runs[-1][1] == "cpu" and any("DANCECAP_ASR_GPU_BUSY" in w for w in t["warnings"])
    monkeypatch.delenv("DANCECAP_ASR_GPU_BUSY")
    monkeypatch.setenv("DANCECAP_ASR_GPU_QUEUE_H", "0.0000001")      # queue limit reached: CPU, flagged
    t = asr.transcribe_whisper(wav, lex, INFO, None, 0.01, log=lambda *a: None)
    assert t["gpu"] == "cpu" and any("gave up waiting" in w for w in t["warnings"]) and asr.retranscribe_reason(t)
    monkeypatch.setattr(asr, "gpu_free_min_mib", lambda *a, **k: 8000)  # enough VRAM beside the lock holder
    t = asr.transcribe_whisper(wav, lex, INFO, None, 0.01, log=lambda *a: None)
    assert t["gpu"] == "guest" and runs[-1] == ("large-v3", "cuda", ("en", "pt"))


# ---------------------------------------------------------------------------------------------------------------
# real engine (opt-in): SAPI speaks, Whisper transcribes
# ---------------------------------------------------------------------------------------------------------------

@pytest.mark.skipif(os.environ.get("DANCECAP_ASR_SMOKE") != "1", reason="set DANCECAP_ASR_SMOKE=1 (GPU, ~1 min)")
def test_whisper_end_to_end_smoke(tmp_path, lex):
    if not (os.name == "nt" and shutil.which("powershell") and SAPI_SYNTH.exists()):
        pytest.skip("needs Windows SAPI to synthesise the narration")
    av = asr.engines(refresh=True)
    if not av["faster-whisper"].get("available"):
        pytest.skip(av["faster-whisper"].get("reason"))
    script = tmp_path / "script.txt"
    script.write_text("pause 1.0\nlateral\npause 2\ncorredor\npause 2\nvira jeenya\npause 2\nboomerang\npause 1.5\n",
                      encoding="utf-8")
    wav = tmp_path / "audio.wav"
    r = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(SAPI_SYNTH),
                        "-Out", str(wav), "-Script", str(script)], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and wav.exists(), r.stderr
    w16 = asr.to_wav16k(wav, tmp_path / "audio16k.wav")
    t = asr.transcribe(w16, "auto", lex, log=lambda *a: None)
    assert t["engine"] == "faster-whisper" and t["version"] == 2 and t["raw"]["passes"]
    ev = map_utterances(t["utterances"], lex)
    moves = [e["move"] for e in ev if e["kind"] == "move"]
    # SAPI's English voice says "corredor" so badly that Whisper hears "Quarter" (measured): not asserted
    assert moves[0] == "lateral" and "viradinha" in moves and "boomerang" in moves, (t["utterances"], ev)
    assert moves.index("viradinha") < moves.index("boomerang")
    assert t["timing"]["wall_s"] > 0 and t["gpu"] in ("lock", "guest", "cpu")
    if t["gpu"] != "cpu":
        assert t["vram_mib"]["after_exit"] is not None


# 24 Brazilian Portuguese move names as an English speaker says them (IPA for the en-US SAPI voice; the Portuguese
# tap r and nh / lh came out as "t" / "n" with this voice, so English r, nj, lj): (name, slug of the real graph, ipa)
PT_NAMES = [
    ("Corredor", "lateral", "Corredor/koʊhɛˈdɔɹ"), ("Viradinha", "viradinha", "Viradinha/viɹəˈdʒinjə"),
    ("Bumerangue", "boomerang", "Bumerangue/bumɛˈɹæŋgi"), ("Cambré", "dip-back", "Cambré/kæmˈbɹeɪ"),
    ("Ioiô", "yoyo", "Ioiô/joʊˈjoʊ"), ("Sarrada", "sarrada", "Sarrada/sɑˈhɑdə"),
    ("Contrapeso", "counterbalance-pivot", "Contrapeso/koʊntɹəˈpeɪzʊ"),
    ("Giro Simples", "simple-turn", "Giro/ˈʒiɹʊ Simples/ˈsimplis"),
    ("Cestinha", "hold-behind-crossed-arms", "Cestinha/sɛsˈtʃinjə"), ("Chicote", "chicote", "Chicote/ʃiˈkɔtʃi"),
    ("Toalha", "toalha", "Toalha/toʊˈɑljə"), ("Mergulho", "dive", "Mergulho/mɛɹˈguljʊ"),
    ("Cobrinha", "snake", "Cobrinha/koʊˈbɹinjə"), ("Balão", "balao", "Balão/bəˈlaʊ"),
    ("Abraço", "hug", "Abraço/ɑˈbɹɑsʊ"), ("Boneca", "boneca", "Boneca/boʊˈnɛkə"),
    ("Giro Duplo", "double-turn", "Giro/ˈʒiɹʊ Duplo/ˈduplʊ"),
    ("Giro no Eixo", "pirouette", "Giro/ˈʒiɹʊ no/nʊ Eixo/ˈeɪʃʊ"), ("Trava", "stop", "Trava/ˈtɹɑvə"),
    ("Gostosinho", "gostosinho", "Gostosinho/gɔstoʊˈzinjʊ"), ("Soltinho", "soltinho", "Soltinho/soʊlˈtʃinjʊ"),
    ("Pião", "piao", "Pião/piˈaʊ"), ("Torção", "twist", "Torção/tɔɹˈsaʊ"), ("Avião", "airplane", "Avião/ɑviˈaʊ")]


@pytest.mark.skipif(os.environ.get("DANCECAP_ASR_SMOKE") != "1", reason="set DANCECAP_ASR_SMOKE=1 (GPU, ~1 min)")
def test_whisper_portuguese_names_smoke(tmp_path):
    """Portuguese pronunciation, the real graph vocabulary: auto (large-v3 en+pt) must get nearly all 24 names
    (measured 24/24; turbo 17/24) and turbo's "Yo -Yo" must still be Ioiô (it was lost to the repetition filter)."""
    import difflib
    if not (os.name == "nt" and shutil.which("powershell") and SAPI_SYNTH.exists()):
        pytest.skip("needs Windows SAPI to synthesise the narration")
    av = asr.engines(refresh=True)
    if not av["faster-whisper"].get("available"):
        pytest.skip(av["faster-whisper"].get("reason"))
    from dancecap.moves.graph import load_graph
    lex = Lexicon.from_graph(load_graph())
    script = tmp_path / "script.txt"
    script.write_text("\n".join(["pause 1.0"] + [x for _, _, ipa in PT_NAMES for x in (f"ipa {ipa}", "pause 1.6")])
                      + "\n", encoding="utf-8")
    wav = tmp_path / "audio.wav"
    r = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(SAPI_SYNTH),
                        "-Out", str(wav), "-Script", str(script)], capture_output=True, text=True, timeout=180)
    assert r.returncode == 0 and wav.exists(), r.stderr
    w16 = asr.to_wav16k(wav, tmp_path / "audio16k.wav")
    want = [slug for _, slug, _ in PT_NAMES]
    for engine, need in (("auto", 21), ("faster-whisper:large-v3-turbo", 12)):
        t = asr.transcribe(w16, engine, lex, log=lambda *a: None)
        got = [e["move"] for e in map_utterances(t["utterances"], lex) if e["kind"] == "move"]
        hit = sum(b.size for b in difflib.SequenceMatcher(None, want, got, autojunk=False).get_matching_blocks())
        assert hit >= need, (engine, hit, [u["text"] for u in t["utterances"]])
        assert "yoyo" in got, (engine, [u["text"] for u in t["utterances"]])
