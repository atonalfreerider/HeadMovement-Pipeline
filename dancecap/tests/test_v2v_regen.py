"""dancecap.speech.v2v_regen: the regenerated-lines soundtrack (pure parts; the HTTP layer is mocked, synthetic audio only).

    .venv-nopo4d/Scripts/python.exe -m pytest dancecap/tests/test_v2v_regen.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dancecap.speech import v2v as V  # noqa: E402
from dancecap.speech import v2v_regen as R  # noqa: E402
from dancecap.speech.v2v_check import onset_near, speech_env_db  # noqa: E402

SR = V.SR


def test_sts_recipes_are_new_and_varied():
    rs = R.sts_recipes(32)
    assert len(rs) == 32
    assert len({r[3] for r in rs}) == 32 and min(r[3] for r in rs) >= 101      # no V2 seed (1..16) is repeated
    assert len({r[0] for r in rs}) >= 6 and len({r[1] for r in rs}) >= 6
    assert any(r[2] == R.MODEL_EN for r in rs) and any(r[2] == V.MODEL_DEFAULT for r in rs)
    assert all(r[0] in R.INPUTS and r[1] in R.SET_KEYS for r in rs)


def test_accept_and_rank():
    assert R.accept(0.15, 80.0) and R.accept(0.0, None)
    assert not R.accept(0.16, 10.0) and not R.accept(0.05, 81.0) and not R.accept(None, 0.0)
    cs = [{"tag": "a", "wer": 0.05, "lag_ms": 200.0}, {"tag": "b", "wer": 0.12, "lag_ms": 30.0}, {"tag": "c", "wer": 0.3, "lag_ms": 5.0}]
    assert min(cs, key=R.rank_key)["tag"] == "b"


def test_choose_final_prefers_accepted_sts_then_tts_then_flagged():
    table = {
        "s1": [{"tag": "t1", "kind": "sts", "wer": 0.1, "lag_ms": 20.0}, {"tag": "tts1", "kind": "tts", "wer": 0.0, "lag_ms": 10.0}],
        "s2": [{"tag": "t1", "kind": "sts", "wer": 0.4, "lag_ms": 20.0}, {"tag": "tts1", "kind": "tts", "wer": 0.2, "lag_ms": 10.0}, {"tag": "tts2", "kind": "tts", "wer": 0.1, "lag_ms": 10.0}],
        "s3": [{"tag": "t1", "kind": "sts", "wer": 0.4, "lag_ms": 20.0}],
        "s4": [],
    }
    f = R.choose_final(table, ["s1", "s2", "s3", "s4"])
    assert f["s1"]["kind"] == "sts" and f["s1"]["flag"] is None
    assert f["s2"] == {"kind": "tts", "tag": "tts2", "wer": 0.1, "lag_ms": 10.0, "flag": "tts"}
    assert f["s3"]["kind"] == "sts" and "no better take" in f["s3"]["flag"]
    assert f["s4"]["kind"] == "original"


def test_tts_text_spans_and_alignment_words():
    ws = R.piece_tts_words([
        {"text": "so", "punct": ",", "start": 1.0, "end": 1.2, "kind": "word"},
        {"text": "uh", "punct": "", "start": 1.3, "end": 1.4, "kind": "filler"},
        {"text": "we", "punct": "", "start": 1.5, "end": 1.6, "kind": "word"},
        {"text": "sacada", "punct": ".", "start": 1.7, "end": 2.3, "kind": "word"},
    ])
    assert [w["text"] for w in ws] == ["so", "we", "sacada"]
    text, spans = R.tts_text_spans(ws)
    assert text == "so, we sacada."
    assert [text[a:b] for a, b in spans] == ["so", "we", "sacada"]
    n = len(text)
    al = {"characters": list(text), "starts": [i * 0.1 for i in range(n)], "ends": [i * 0.1 + 0.1 for i in range(n)]}
    wt = R.word_times_from_alignment(al, spans)
    assert wt[0] == (0.0, 0.2) and abs(wt[1][0] - 0.4) < 1e-9 and abs(wt[2][1] - 1.3) < 1e-9


def test_plan_runs_splits_at_pauses_of_either_side():
    tw = [(0.0, 0.3), (0.35, 0.6), (1.2, 1.5), (1.55, 1.8)]
    aw = [(0.0, 0.3), (0.32, 0.6), (0.62, 0.9), (0.95, 1.2)]
    assert R.plan_runs(tw, aw) == [(0, 2), (2, 4)]                  # the transcript pause at 0.6-1.2
    aw2 = [(0.0, 0.3), (0.32, 0.6), (1.0, 1.3), (1.32, 1.6)]
    assert R.plan_runs([(0, .3), (.35, .6), (.65, .9), (.95, 1.2)], aw2) == [(0, 2), (2, 4)]     # the take's own pause
    assert R.plan_runs(tw[:1], aw[:1]) == [(0, 1)]


def test_speed_for_dead_band_and_limits():
    assert R.speed_for(3.0, 3.1) == 1.0
    assert R.speed_for(4.0, 3.0) == 1.2                              # clipped to the API limit
    assert abs(R.speed_for(3.3, 3.0) - 1.1) < 1e-9
    assert R.speed_for(2.0, 4.0) == 0.85


def burst(sr, t0, d, f0=200.0):
    n = int(d * sr)
    t = np.arange(n) / sr
    return t0, (np.sin(2 * np.pi * f0 * t) * np.hanning(n) * 0.5 + np.sin(2 * np.pi * 3 * f0 * t) * np.hanning(n) * 0.3).astype(np.float32)


def test_fit_take_puts_word_onsets_on_the_transcript_times():
    a_on = [0.10, 0.70, 1.25]                                        # the take's words
    x = np.zeros(int(2.0 * SR), np.float32)
    for t0 in a_on:
        s, b = burst(SR, t0, 0.3)
        x[int(s * SR):int(s * SR) + len(b)] += b
    aw = [(t, t + 0.3) for t in a_on]
    tw = [(0.30, 0.55), (0.90, 1.2), (1.60, 1.9)]                      # transcript: later, with a different rhythm
    y, rep = R.fit_take(x, SR, tw, aw, n_out=int(2.5 * SR), t_origin=0.0)
    assert len(y) == int(2.5 * SR) and np.isfinite(y).all()
    env = speech_env_db(y, SR)
    for t0, _ in tw:
        o = onset_near(env, t0, before=0.25, after=0.25)
        assert o is not None and abs(o - t0) < 0.05, (t0, o)
    assert rep["onset_pred_max_ms"] < 60 and 0.6 <= rep["rate_min"] and rep["rate_max"] <= 1.7


def test_fit_take_pause_becomes_silence_not_a_stretch():
    # the take speaks two words back to back; the transcript has a 1 s pause between them: the second run is placed later, the gap is silent
    a_on = [0.1, 0.45]
    x = np.zeros(int(1.2 * SR), np.float32)
    for t0 in a_on:
        s, b = burst(SR, t0, 0.3)
        x[int(s * SR):int(s * SR) + len(b)] += b
    y, rep = R.fit_take(x, SR, [(0.2, 0.5), (1.6, 1.9)], [(0.1, 0.4), (0.45, 0.75)], n_out=int(2.5 * SR))
    assert len(rep["runs"]) == 2
    gap = y[int(0.75 * SR):int(1.5 * SR)]
    assert np.abs(gap).max() < 1e-3
    assert np.abs(y[int(1.6 * SR):int(1.8 * SR)]).max() > 0.05


def test_room_tone_from_gaps_only_leaves_out_speech():
    from dancecap.speech import v2v_mix as M
    rng = np.random.default_rng(1)
    n = 10 * SR
    t = np.arange(n) / SR
    x = rng.normal(0, 0.01, n)
    x[:8 * SR] += 0.5 * np.sin(2 * np.pi * 1000 * t[:8 * SR])               # a "voice" that fills 80 % of the block
    orig = np.stack([x, x], axis=1).astype(np.float32)

    def tone_ratio(bed):
        sp = np.abs(np.fft.rfft(bed[:, 0].astype(np.float64)))
        f = np.fft.rfftfreq(len(bed), 1 / SR)
        return sp[(f > 980) & (f < 1020)].mean() / sp[(f > 1500) & (f < 1900)].mean()

    plain = M.room_tone(orig, n)
    gaps = M.room_tone(orig, n, speech_spans=[(-1.0, 8.2)], min_gap_frames=24)
    assert tone_ratio(plain) > 2.0                                           # the 15th percentile still holds the voice when it fills the block
    assert tone_ratio(gaps) < 1.6                                            # built from the pauses only: a flat room


def test_realign_runs_moves_the_late_stretch_onto_the_transcript():
    def sig(onsets, n=int(3.0 * SR)):
        x = np.zeros(n, np.float32)
        for t0 in onsets:
            s, b = burst(SR, t0, 0.3)
            x[int(s * SR):int(s * SR) + len(b)] += b
        return x
    y = sig([0.2, 1.7])                                   # the converted piece: the second word 0.25 s late
    ref = sig([0.2, 1.45])                                # the phone: on the transcript
    z, rows = R.realign_runs(y, ref, [0.2, 1.45], [0.5, 1.75], max_shift_s=0.35)
    assert len(z) == len(y) and rows and abs(rows[0]["shift_ms"] + 250) < 40
    env = speech_env_db(z, SR)
    assert abs(onset_near(env, 1.45, before=0.3, after=0.3) - 1.45) < 0.04
    assert abs(onset_near(env, 0.2, before=0.3, after=0.3) - 0.2) < 0.04               # the first word stays
    # a piece that is already on the transcript, or whose phone disagrees with the transcript, is left alone
    z2, rows2 = R.realign_runs(ref, ref, [0.2, 1.45], [0.5, 1.75])
    assert rows2 == [] and z2 is ref
    z3, rows3 = R.realign_runs(y, sig([0.2, 1.1]), [0.2, 1.45], [0.5, 1.75])           # phone 0.35 s from the transcript: not trustworthy
    assert rows3 == []


def test_fit_take_uses_the_audible_onset_not_the_alignment_start():
    # the alignment claims the word starts 80 ms before the sound does
    a_true = 0.50
    x = np.zeros(int(1.5 * SR), np.float32)
    s, b = burst(SR, a_true, 0.3)
    x[int(s * SR):int(s * SR) + len(b)] += b
    y, rep = R.fit_take(x, SR, [(0.40, 0.70)], [(a_true - 0.08, a_true + 0.3)], n_out=int(1.5 * SR))
    o = onset_near(speech_env_db(y, SR), 0.40, before=0.25, after=0.25)
    assert o is not None and abs(o - 0.40) < 0.04, o
