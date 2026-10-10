"""dancecap.speech.v2v_full: the full cloned-voice re-recording (pure parts; the HTTP layer is mocked, synthetic audio only).

    .venv-nopo4d/Scripts/python.exe -m pytest dancecap/tests/test_v2v_full.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dancecap.speech import v2v as V  # noqa: E402
from dancecap.speech import v2v_full as F  # noqa: E402

SR = V.SR


def words(spec, speaker="Larissa", utt=None):
    """[(t0, t1)] -> word dicts"""
    return [{"t0": a, "t1": b, "text": f"w{i}", "speaker": speaker, "kind": "word", "utt": (utt[i] if utt else "u0")} for i, (a, b) in enumerate(spec)]


def test_sentence_split_cuts_at_pauses_and_respects_bounds():
    # 40 words of 0.3 s, a 0.4 s pause after every 8th word
    t, spec = 0.0, []
    for i in range(40):
        spec.append((t, t + 0.3))
        t += 0.3 + (0.4 if i % 8 == 7 else 0.05)
    parts = F.sentence_split(words(spec))
    assert sum(len(p) for p in parts) == 40                       # nothing lost, nothing duplicated
    assert all(F.MIN_TAIL_S <= p[-1]["t1"] - p[0]["t0"] <= F.MAX_S + 3.0 for p in parts)
    assert parts[0][-1]["t0"] < parts[1][0]["t0"]
    for p in parts[:-1]:                                          # soft cuts fall in the 0.4 s pauses
        nxt = parts[parts.index(p) + 1][0]
        assert nxt["t0"] - p[-1]["t1"] >= 0.3


def test_sentence_split_hard_gap_and_long_run_cut_at_longest_gap():
    ws = words([(0.0, 0.3), (0.35, 0.6)] + [(5.0, 5.3), (5.35, 5.6)])
    parts = F.sentence_split(ws)
    assert [len(p) for p in parts] == [2, 2]                      # a 4.4 s silence always ends a piece
    # a run with no pauses at all, 20 s: cut into pieces <= max_s + 1
    ws = words([(i * 0.35, i * 0.35 + 0.3) for i in range(60)])
    parts = F.sentence_split(ws)
    assert len(parts) >= 3 and all(p[-1]["t1"] - p[0]["t0"] <= F.MAX_S + 3.0 + 0.01 for p in parts)


def test_sentence_split_prefers_utterance_change_and_merges_short_tail():
    spec = [(i * 0.5, i * 0.5 + 0.4) for i in range(12)]          # 6 s, gaps 0.1
    utt = ["a"] * 6 + ["b"] * 6
    parts = F.sentence_split(words(spec, utt=utt), max_s=4.0)
    assert any(p[-1]["utt"] == "a" and p[-1]["text"] == "w5" for p in parts)    # cut where the utterance changes
    ws = words([(0, 0.4), (0.5, 3.0), (3.2, 3.6)])                # tail of 0.4 s after a 3 s piece
    assert len(F.sentence_split(ws)) == 1


def test_decide_policy_owner_rule():
    assert F.decide_policy(0.30, 0.10)["use"] == "original"       # worse by 0.20 and above 0.25: garbled
    assert F.decide_policy(0.24, 0.05)["use"] == "converted"      # worse by 0.19 but below 0.25 absolute: kept (cloned voice wanted)
    assert F.decide_policy(0.50, 0.45)["use"] == "converted"      # worse by only 0.05: kept even though high
    assert F.decide_policy(0.60, 0.50)["use"] == "converted"      # exactly at the margin (0.10): not "more than"
    assert F.decide_policy(None, 0.1)["use"] == "original" and F.decide_policy(None, 0.1)["flag"]
    assert F.decide_policy(0.8, None)["use"] == "converted"
    d = F.decide_policy(0.9, 0.1)
    assert d["use"] == "original" and "garbled" in d["flag"]


def test_pick_best_try():
    t = [{"name": "a", "wer": 0.4}, {"name": "b", "wer": 0.2}, {"name": "c", "wer": 0.2}, {"name": "d", "wer": None}]
    assert F.pick_best_try(t)["name"] == "b"
    assert F.pick_best_try([{"name": "x", "wer": None}]) is None and F.pick_best_try([]) is None


def noise_burst(n, seed):
    return np.random.default_rng(seed).normal(0, 0.1, n).astype(np.float32)


def test_gcc_phat_delay_integer_and_fractional():
    x = noise_burst(SR, 1)
    y = np.roll(x, 48)                                            # 1 ms later
    y[:48] = 0
    d, conf = F.gcc_phat_delay(x, y)
    assert abs(d - 48) < 0.2 and conf > 10
    z = F.fractional_shift(x, 12.4)
    d2, _ = F.gcc_phat_delay(x, z)
    assert abs(d2 - 12.4) < 0.3


def test_fractional_shift_roundtrip_keeps_length_and_signal():
    x = noise_burst(4000, 2)
    y = F.fractional_shift(F.fractional_shift(x, 7.3), -7.3)
    assert len(y) == len(x)
    assert np.corrcoef(x[100:-100], y[100:-100])[0, 1] > 0.99


def test_beamform_aligns_and_reduces_noise_and_skips_uncorrelated():
    rng = np.random.default_rng(3)
    from scipy.signal import butter, sosfiltfilt
    sig = sosfiltfilt(butter(2, [200, 3500], btype="band", fs=SR, output="sos"), rng.normal(0, 1, SR * 2)).astype(np.float32)
    sig /= sig.std()
    a = sig + 0.8 * rng.normal(0, 1, len(sig)).astype(np.float32)
    b = F.fractional_shift(sig, 20.0) * 0.5 + 0.4 * rng.normal(0, 1, len(sig)).astype(np.float32)
    c = rng.normal(0, 1, len(sig)).astype(np.float32)             # nothing to do with it
    out, info = F.beamform([a, b, c], [10.0, 9.0, 8.0], max_shift_s=0.003, min_conf=6.0)
    assert "delay_ms" in info["phones"][1] and abs(info["phones"][1]["delay_ms"] - 20 / 48) < 0.1
    assert "skipped" in info["phones"][2]
    assert np.corrcoef(out, sig)[0, 1] > np.corrcoef(a, sig)[0, 1]          # better than the best phone alone
    single, info1 = F.beamform([a], [10.0])
    assert np.allclose(single, a) and len(info1["phones"]) == 1


def test_convert_file_cache_budget_and_fake_request(tmp_path, monkeypatch):
    monkeypatch.setattr(F, "LEDGER", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(V, "LEDGER", tmp_path / "ledger.jsonl")
    src = tmp_path / "s.wav"
    sf.write(str(src), noise_burst(SR * 2, 5), SR, subtype="PCM_16")
    calls = []

    def fake(session, vid, audio, **kw):
        calls.append(kw)
        V.ledger_add({"credits": 21, "purpose": kw.get("purpose")})
        return (np.zeros(24000 * 2, dtype="<i2")).tobytes(), {"credits": 21, "request_id": "r1", "seconds": 0.1, "bytes": 96000}

    m = F.convert_file(None, "vid", src, tmp_path / "conv", "x", request=fake)
    assert not m["cached"] and calls[0]["remove_background_noise"] is True and calls[0]["settings"]["stability"] == 0.8
    assert (tmp_path / "conv" / "x.wav").exists()
    m2 = F.convert_file(None, "vid", src, tmp_path / "conv", "x", request=fake)
    assert m2["cached"] and len(calls) == 1
    F.convert_file(None, "vid", src, tmp_path / "conv", "x", settings_key="S", request=fake)       # other settings: a new request
    assert len(calls) == 2 and F.session_credits() == 42
    with pytest.raises(V.V2VError):
        F.convert_file(None, "vid", src, tmp_path / "conv", "y", seed=3, request=fake, budget=50)  # 42 + 21 > 50
    assert len(calls) == 2


def test_align_to_source_restores_length(tmp_path):
    rng = np.random.default_rng(4)
    from scipy.signal import butter, sosfiltfilt
    n = SR * 4
    x = sosfiltfilt(butter(2, 3.0, fs=SR, output="sos"), rng.normal(0, 1, n))
    x = (np.abs(x) / np.abs(x).max() * rng.normal(0, 0.3, n)).astype(np.float32)
    sf.write(str(tmp_path / "src.wav"), x, SR, subtype="PCM_16")
    shifted = np.roll(x, int(0.03 * SR))
    from scipy import signal as sg
    sf.write(str(tmp_path / "conv.wav"), sg.resample_poly(shifted, 1, 2).astype(np.float32), 24000, subtype="PCM_16")
    rep = F.align_to_source(tmp_path / "conv.wav", tmp_path / "src.wav", tmp_path / "al.wav")
    y, sr = sf.read(str(tmp_path / "al.wav"))
    assert sr == SR and len(y) == n and rep["clipped_samples"] == 0
    assert abs(rep["lag_global_ms"] - 30) < 10


def piece(pid, speaker="Larissa", n=10):
    return V.Piece(id=pid, speaker=speaker, t0=0.0, t1=3.0, w0=0.2, w1=2.8, n_words=n, text="a b c")


def test_build_decisions_policy_and_bad_tries():
    ps = [piece("s001"), piece("s002"), piece("s003"), piece("s004")]
    tries = {"s001": [{"tag": "t1", "wer": 0.05}, {"tag": "t2", "wer": 0.3}],
             "s002": [{"tag": "t1", "wer": 0.6}, {"tag": "t2", "wer": 0.45}],      # best 0.45 vs original 0.10: garbled -> original
             "s003": [{"tag": "t1", "wer": 0.2}],                                   # worse by 0.15 but below 0.25 absolute: kept
             "s004": [{"tag": "t1", "wer": 0.0}, {"tag": "t2", "wer": 0.4}]}
    orig = {"s001": 0.2, "s002": 0.10, "s003": 0.05, "s004": 0.1}
    d = F.build_decisions(ps, tries, orig, bad_tries={"s004": {"t1"}})
    assert d["s001"]["variant"] == "t1" and d["s001"]["flag"] is None
    assert d["s002"]["variant"] is None and "garbled" in d["s002"]["flag"]
    assert d["s003"]["variant"] == "t1"
    assert d["s004"]["variant"] is None            # the t1 try failed the voice check; the remaining t2 (0.4 vs 0.1) is garbled
    assert d["s004"]["n_tries"] == 1


def test_need_try_stops_at_target_or_max():
    assert F.need_try([]) and F.need_try([{"wer": 0.5}])
    assert not F.need_try([{"wer": 0.5}, {"wer": 0.08}])
    assert not F.need_try([{"wer": 0.9}] * F.MAX_TRIES * 3, max_tries=8)


# ------------------------------------------------------------------------------------------------ v2v_splice
from dancecap.speech import v2v_splice as SP  # noqa: E402


def test_align_pairs_and_word_status():
    pairs = SP.align_pairs(["a", "b", "c", "d"], ["a", "x", "c", "d", "e"])
    assert (0, 0) in pairs and (1, 1) in pairs and (2, 2) in pairs and (3, 3) in pairs and (None, 4) in pairs
    ref, ok, spans = SP.word_status("it is the Entire dance", [["it", 0.0, 0.2], ["is", 0.25, 0.4], ["a", 0.45, 0.5], ["entire", 0.6, 1.0], ["dance,", 1.1, 1.5]])
    assert ref == ["it", "is", "the", "entire", "dance"]
    assert ok == [True, True, False, True, True] and spans[2] is None and spans[4] == (1.1, 1.5)


def test_choose_path_prefers_switching_only_when_it_pays():
    ok = [[True, True, False, False], [False, False, True, True]]     # try 0 right in words 0-1, try 1 right in words 2-3
    cuts = {(0, 1, 1): 1.0, (1, 0, 1): 1.0}
    path, cost = SP.choose_path(ok, cuts, 4)
    assert path == [0, 0, 1, 1] and abs(cost - SP.SWITCH_COST) < 1e-9
    path2, cost2 = SP.choose_path(ok, {}, 4)                           # no valley anywhere: no switching, the better try wins
    assert len(set(path2)) == 1 and cost2 == 2.0
    ok1 = [[True, False, True, True], [True, True, True, True]]        # one wrong word in try 0 but try 1 is right everywhere: stay in try 1
    assert SP.choose_path(ok1, {(0, 1, 0): 0.5, (1, 0, 0): 0.5}, 4)[0] == [1, 1, 1, 1]


def test_cut_between_needs_a_valley_in_both():
    env = np.full(300, -60.0)
    env[:100] = -20.0
    env[200:] = -20.0                                                  # speech 0-1 s, pause 1-2 s, speech 2-3 s
    a_sp = [(0.1, 0.9), (2.1, 2.9)]
    b_sp = [(0.1, 0.9), (2.05, 2.9)]
    c = SP.cut_between(a_sp, b_sp, 0, env, env, -20.0, -20.0)
    assert c is not None and 0.9 <= c <= 2.1
    assert SP.cut_between([None, (2.1, 2.9)], b_sp, 0, env, env, -20.0, -20.0) is None
    assert SP.cut_between(a_sp, b_sp, 0, np.full(300, -20.0), np.full(300, -20.0), -20.0, -20.0) is None      # no valley: continuous speech


def test_render_crossfade_is_continuous_and_picks_segments():
    sr = 8000
    a = np.ones(sr * 2, dtype=np.float32)
    b = 3 * np.ones(sr * 2, dtype=np.float32)
    y, ch = SP.render([a, b], [0, 0, 1, 1], {(0, 1, 1): 1.0}, sr, fade_s=0.02)
    assert y[int(0.5 * sr)] == pytest.approx(1.0) and y[int(1.5 * sr)] == pytest.approx(3.0)
    assert ch == [{"after_word": 1, "from": 0, "to": 1, "t": 1.0}]
    assert np.abs(np.diff(y)).max() < 0.5                              # no click at the cut
    y2, ch2 = SP.render([a, b], [1, 1, 1], {}, sr)
    assert np.allclose(y2, b) and ch2 == []


# ------------------------------------------------------------------------------------------------ v2v_warp
from dancecap.speech import v2v_warp as W  # noqa: E402


def syllables(n, sr=SR, seed=0):
    """bursts of a voiced-like tone at irregular times (a stand-in for syllables); returns the signal and the burst start times"""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / sr
    x = np.zeros(n)
    starts = []
    pos = 0.15
    while pos < n / sr - 0.4:
        d = rng.uniform(0.12, 0.3)
        a, b = int(pos * sr), int((pos + d) * sr)
        f0 = rng.uniform(120, 220)
        seg = sum(np.sin(2 * np.pi * f0 * h * t[a:b]) / h for h in range(1, 12)) * np.hanning(b - a)
        x[a:b] += seg
        starts.append(pos)
        pos += d + rng.uniform(0.08, 0.25)
    return (x / np.abs(x).max() * 0.3).astype(np.float32), starts


def test_retime_follows_a_local_stretch():
    x, starts = syllables(SR * 4)
    # the "converted" piece says the same thing but slower in the middle: source time t maps to converted time g(t) = t + 0.12 * sin(pi t / 4)
    t_out = np.arange(len(x)) / SR
    g = t_out + 0.12 * np.sin(np.pi * t_out / 4.0)
    conv = np.interp(g * SR, np.arange(len(x)), x).astype(np.float32)
    y, info = W.retime(x, conv, SR)
    assert len(y) == len(x)
    ref = V.window_lags(x, conv, SR, 0.0, win_s=1.2, hop_s=0.4, search_s=0.35, min_peak=0.3)
    after = V.window_lags(x, y, SR, 0.0, win_s=1.2, hop_s=0.4, search_s=0.35, min_peak=0.3)
    assert max(abs(w["lag_s"]) for w in ref) > 0.06                 # the stretch is really there before
    assert max(abs(w["lag_s"]) for w in after) < 0.04               # and gone after the re-timing
    assert info["rate_min"] >= W.RATE_MIN - 1e-6 and info["rate_max"] <= W.RATE_MAX + 1e-6


def test_retime_of_an_aligned_piece_changes_little_and_keeps_pitch():
    x, _ = syllables(SR * 3, seed=1)
    y, info = W.retime(x, x.copy(), SR)
    assert info["max_shift_ms"] < 20
    ex, ey = V.speech_envelope(x, SR), V.speech_envelope(y, SR)
    assert np.corrcoef(ex, ey)[0, 1] > 0.9                           # same rhythm
    fx = np.abs(np.fft.rfft(x))
    fy = np.abs(np.fft.rfft(y))
    assert abs(np.argmax(fx) - np.argmax(fy)) / np.argmax(fx) < 0.05     # same pitch (dominant frequency)


def test_needs_warp():
    assert W.needs_warp([10, -70]) and not W.needs_warp([10, -30]) and not W.needs_warp([])


# ------------------------------------------------------------------------------------------------ v2v_check
from dancecap.speech import v2v_check as K  # noqa: E402


def test_onset_near_and_local_lag_follow_a_known_shift():
    x = np.zeros(SR * 4, dtype=np.float32)
    tt = np.arange(SR * 4) / SR
    for t0 in (1.0, 2.5):                                             # two isolated bursts of a voiced-like tone
        a, b = int(t0 * SR), int((t0 + 0.4) * SR)
        x[a:b] = (0.3 * np.sin(2 * np.pi * 160 * tt[a:b]) * np.hanning(b - a)).astype(np.float32)
    env = K.speech_env_db(x, SR)
    t = 2.5
    o = K.onset_near(env, t)
    assert o is not None and abs(o - t) < 0.12
    y = np.roll(x, int(0.12 * SR))                                   # the same speech 120 ms later
    ey = K.speech_env_db(y, SR)
    lag = K.local_lag_ms(ey, env, 0.5, 3.0)
    assert lag is not None and abs(lag - 120) < 15
    assert K.local_lag_ms(env, np.full_like(env, -60.0), 0.5, 3.0) is None      # a flat reference has no speech structure


def test_join_steps_reports_level_steps_of_the_words():
    sr = SR
    sig = np.zeros(sr * 6, dtype=np.float32)
    rng = np.random.default_rng(0)
    sig[:sr * 3] = rng.normal(0, 0.1, sr * 3)                          # piece a: 0-3 s
    sig[sr * 3:] = rng.normal(0, 0.1 * 10 ** (-5 / 20), sr * 3)        # piece b: 3-6 s, 5 dB quieter
    pieces = [{"id": "a", "t0": 0.0, "t1": 3.0}, {"id": "b", "t0": 3.0, "t1": 6.0}]
    js = K.join_steps(pieces, sig)
    assert js["joins"] == 1 and abs(js["steps"][0]["speech_step_db"] + 5) < 0.6 and len(js["over_3db"]) == 1
    assert K.clipping(np.array([0.5, -1.0]))["samples_ge_0.999"] == 1


def test_pick_clean_words_uses_the_trusted_reference_only():
    x, starts = syllables(SR * 6, seed=4)
    env = K.speech_env_db(x, SR)
    ws = [{"t0": s, "t1": s + 0.25, "text": f"w{i}", "kind": "word"} for i, s in enumerate(starts)]
    ws += [{"t0": 99.0, "t1": 99.3, "text": "far", "kind": "word"}]
    got = K.pick_clean_words(ws, 5, min_gap=0.05, t_min=0.0, ref_env=env)
    assert got and all(w["text"] != "far" for w in got)
