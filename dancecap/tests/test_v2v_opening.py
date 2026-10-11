"""dancecap.speech.v2v_opening (agent O1): the opening's pieces from the owner's words, the choice, and the laugh events of the assembly. Synthetic data only.

    .venv-nopo4d/Scripts/python.exe -m pytest tests/test_v2v_opening.py -q
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dancecap.speech import v2v as V  # noqa: E402
from dancecap.speech import v2v_mix as M  # noqa: E402
from dancecap.speech import v2v_opening as O  # noqa: E402

SR = V.SR


def ow(text, t0, t1, speaker="Larissa", punct=""):
    return {"text": text, "punct": punct, "t0": t0, "t1": t1, "speaker": speaker}


WORDS = [ow("hello", 1.0, 1.5, punct="!"), ow("I'm", 1.6, 1.8), ow("Kadu", 1.8, 2.2), ow("and", 3.3, 3.4), ow("this", 3.4, 3.7), ow("is", 3.7, 3.9), ow("Larissa", 3.9, 5.0, punct="!"),
         ow("yeah", 5.3, 5.8, "Kadu", ","), ow("you", 5.8, 6.0, "Kadu"), ow("see", 6.0, 6.1, "Kadu"), ow("my", 6.1, 6.2, "Kadu"), ow("hair", 6.2, 6.5, "Kadu", "?"),
         ow("yeah", 6.9, 7.3, punct=","), ow("it's", 7.4, 7.6), ow("very", 7.6, 7.8), ow("red", 7.8, 8.1, punct=".")]


def test_display_case_and_grouping_split_at_the_laugh_and_at_a_change_of_speaker(monkeypatch):
    d = O.display(WORDS)
    assert [w["text"] for w in d][:4] == ["Hello", "I'm", "Kadu", "and"] and d[7]["text"] == "Yeah" and d[12]["text"] == "Yeah" and d[13]["text"] == "it's"
    gs = O.group(d)
    assert [[w["text"] for w in g] for g in gs] == [["Hello", "I'm", "Kadu"], ["and", "this", "is", "Larissa"], ["Yeah", "you", "see", "my", "hair"], ["Yeah", "it's", "very", "red"]]


def test_pieces_cover_the_owner_words_without_overlapping_words(monkeypatch):
    monkeypatch.setattr(O, "owner_words", lambda: WORDS)
    ps, wd = O.pieces()
    assert [p.id for p in ps] == ["n0", "n1", "n2", "n3"] and [p.speaker for p in ps] == ["Larissa", "Larissa", "Kadu", "Larissa"]
    assert ps[0].text == "Hello! I'm Kadu" and ps[2].text == "Yeah, you see my hair?"
    for p, q in zip(ps, ps[1:]):
        assert p.t1 <= q.w0 and p.t0 < p.w0 and p.w1 < p.t1 + 1e-9
    assert ps[-1].t1 <= 8.43 - 0.04 + 1e-9                       # never into "we talked to" (the next piece, 8.43 s)


def test_choose_prefers_clear_words_then_small_drift_then_lively_pitch_and_honours_prefer():
    ps = [V.Piece("n0", "Larissa", 0.6, 2.6, 1.0, 2.2, 3, "Hello! I'm Kadu")]
    cands = [{"tag": "nt1", "kind": "tts", "wer": 0.33, "onset_pred_max_ms": 5, "f0": {"spread_st": 9}},
             {"tag": "nt2", "kind": "tts", "wer": 0.0, "onset_pred_max_ms": 80, "f0": {"spread_st": 2}},
             {"tag": "nt3", "kind": "tts", "wer": 0.0, "onset_pred_max_ms": 10, "f0": {"spread_st": 2}},
             {"tag": "nt4", "kind": "tts", "wer": 0.0, "onset_pred_max_ms": 12, "f0": {"spread_st": 7}},
             {"tag": "n1", "kind": "sts", "wer": 1.0, "lag_ms": 0.0, "f0": {"spread_st": 12}}]
    assert O.choose({"n0": cands}, ps)["n0"]["tag"] == "nt4"
    assert O.choose({"n0": cands}, ps, prefer={"n0": "nt2"})["n0"]["tag"] == "nt2"


def test_assemble_cuts_laughter_events_from_the_original_into_the_original_stem(tmp_path, monkeypatch):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg needed for ebur128")
    monkeypatch.setattr(V, "V2V", tmp_path / "v2v")
    n = int(12 * SR)
    rng = np.random.default_rng(2)
    orig = rng.normal(0, 0.01, (n, 2)).astype(np.float32)
    sf.write(str(tmp_path / "class_mix.wav"), orig, SR, subtype="PCM_16")
    p = V.Piece("p001", "Larissa", 6.0, 10.0, 6.3, 9.7, 3, "a b c")
    (tmp_path / "v2v" / "aligned" / "A").mkdir(parents=True)
    t = np.arange(int(4 * SR)) / SR
    sf.write(str(tmp_path / "v2v" / "aligned" / "A" / "p001.wav"), (0.2 * np.sin(2 * np.pi * 200 * t) * (t > 0.3)).astype(np.float32), SR, subtype="PCM_16")
    rep = M.assemble([p], [], {"p001": {"variant": "A", "cam": "01"}}, orig_path=tmp_path / "class_mix.wav", out_wav=tmp_path / "o" / "m.wav", stems_dir=tmp_path / "o" / "s",
                     report_path=tmp_path / "o" / "m.json", log=lambda *_: None, events=[{"id": "laugh1", "t0": 2.0, "t1": 3.0}], event_gain_db=-6.0)
    st = {k: sf.read(str(tmp_path / "o" / "s" / f"{k}.wav"), dtype="float32")[0] for k in ("larissa", "original", "bed")}
    rms = lambda a, t0, t1: float(np.sqrt((a[int(t0 * SR):int(t1 * SR)].astype(np.float64) ** 2).mean()))     # noqa: E731
    assert rms(st["original"], 2.2, 2.8) > 20 * rms(st["original"], 4.0, 5.5) and rms(st["original"], 6.5, 9.5) == 0.0     # only the event is the original
    assert rms(st["bed"], 2.2, 2.8) < 0.05 * rms(st["bed"], 4.0, 5.5)                                                        # no bed under it
    assert rep["events"][0]["id"] == "laugh1" and rep["event_gain_db"] == -6.0 and rep["original_pieces"] == 0
