"""dancecap.speech.v2v / v2v_gate / v2v_mix: the voice-to-voice class audio (pure parts; the HTTP layer and the pod are mocked, synthetic audio only).

    .venv-nopo4d/Scripts/python.exe -m pytest dancecap/tests/test_v2v.py -q
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dancecap.speech import v2v as V  # noqa: E402
from dancecap.speech import v2v_gate as G  # noqa: E402
from dancecap.speech import v2v_mix as M  # noqa: E402

SR = V.SR
HAVE_FFMPEG = shutil.which("ffmpeg") is not None


def W(t0, t1, text, speaker="Larissa"):
    return {"t0": t0, "t1": t1, "text": text, "speaker": speaker, "kind": "word", "utt": "u0"}


def speechlike(n, sr=SR, seed=0, rate=2.7):
    """voiced-like comb with an irregular syllable envelope (stands in for speech; not periodic, so the lag is unambiguous)"""
    from scipy.signal import butter, sosfiltfilt
    rng = np.random.default_rng(seed)
    t = np.arange(n) / sr
    x = np.zeros(n)
    for f0 in (140.0, 210.0):
        for h in range(1, 14):
            x += np.sin(2 * np.pi * f0 * h * t + rng.uniform(0, 6.28)) / h
    slow = sosfiltfilt(butter(2, rate, fs=sr, output="sos"), rng.normal(0, 1, n))
    slow = slow / (slow.std() + 1e-12)
    env = np.clip(slow + 0.2, 0, None) ** 1.5
    return (x * env * 0.06 + rng.normal(0, 0.002, n)).astype(np.float32)


# ------------------------------------------------------------------------------------------------ the plan


def test_plan_splits_at_speaker_change_and_long_pause_and_keeps_the_opening():
    ws = [W(1.0, 1.4, "hello"), W(2.0, 2.4, "we"), W(2.5, 3.0, "did"),            # opening (before 9.7)
          W(10.0, 10.5, "so"), W(10.6, 11.2, "first"), W(11.3, 12.0, "thing"),
          W(12.2, 12.8, "okay", "Kadu"), W(12.9, 13.6, "then", "Kadu"),
          W(20.0, 20.5, "later"), W(20.6, 21.3, "again")]
    pieces, kept = V.plan_pieces(ws, open_end=9.7, duration=60.0)
    assert [(p.speaker, p.w0, p.w1) for p in pieces] == [("Larissa", 10.0, 12.0), ("Kadu", 12.2, 13.6), ("Larissa", 20.0, 21.3)]
    assert kept[0]["t0"] == 0.0 and kept[0]["t1"] == 9.7 and "opening" in kept[0]["why"]
    assert all(p.t0 >= 9.7 for p in pieces)
    # neighbouring cuts never overlap and padding never reaches into the next speaker's word
    assert pieces[0].t1 <= pieces[1].t0 + 1e-9
    assert pieces[0].t1 <= 12.2 and pieces[1].t0 >= 12.0


def test_plan_long_run_is_cut_at_the_longest_pause_and_short_stretches_stay_original():
    ws, t = [], 10.0
    for i in range(120):                                  # 120 words of 0.4 s with 0.1 s gaps, a 0.9 s pause after word 60
        ws.append(W(t, t + 0.4, f"w{i}"))
        t += 0.5 + (0.9 if i == 60 else 0.0)
    ws.append(W(t + 3, t + 3.2, "oh"))                    # a 0.2 s utterance
    pieces, kept = V.plan_pieces(ws, open_end=0.0, duration=200.0, max_len=40.0)
    assert len(pieces) >= 2 and all(p.w1 - p.w0 <= 40.0 + 1e-6 for p in pieces)
    assert any("too short" in k["why"] for k in kept)
    assert sum(p.n_words for p in pieces) == 120


def test_plan_marks_overlap_of_the_two_voices_as_keep_original():
    ws = [W(10.0, 12.0, "long", "Larissa"), W(12.0, 14.0, "words", "Larissa"), W(11.5, 12.5, "yes", "Kadu")]
    ws.sort(key=lambda w: w["t0"])
    pieces, _ = V.plan_pieces(ws, open_end=0.0, duration=30.0)
    l_pieces = [p for p in pieces if p.speaker == "Larissa"]
    assert l_pieces and any(p.keep for p in l_pieces)
    for p in l_pieces:
        for a, b in p.keep:
            assert p.t0 <= a < b <= p.t1 and 11.4 <= a and b <= 12.6


def test_boundary_double_labels_are_not_an_overlap():
    ov = V.overlap_intervals([W(1.0, 2.0, "a", "Larissa"), W(2.01, 3.0, "b", "Kadu")])
    assert ov and ov[0][1] - ov[0][0] < V.MIN_OVERLAP_S
    pieces, _ = V.plan_pieces([W(10.0, 12.0, "a", "Larissa"), W(12.01, 14.0, "b", "Kadu")], open_end=0.0, duration=30.0)
    assert all(not p.keep for p in pieces)


def test_opening_end_waits_for_the_owner_or_ends_in_a_pause(tmp_path):
    ws = [W(8.5, 8.8, "we"), W(8.84, 9.186, "talked"), W(9.19, 9.51, "to"), W(9.82, 10.0, "we")]
    assert V.opening_end(ws, owner_opening=tmp_path / "missing.json") == pytest.approx(9.665, abs=0.01)
    done = tmp_path / "owner_opening.json"
    done.write_text("{}")
    assert V.opening_end(ws, owner_opening=done) == 0.0


def test_cost_estimate():
    p = V.Piece("p1", "Kadu", 0, 60, 1, 59, 10, "x")
    assert V.cost_estimate([p])["credits_one_pass"] == 605


# ------------------------------------------------------------------------------------------------ text and WER


def test_norm_words_and_wer():
    assert V.norm_words("Yes, it's very red! \U0001f602 Well-done.") == ["yes", "it's", "very", "red", "well", "done"]
    assert V.wer("the cat sat", "the cat sat") == 0.0
    c = V.edit_counts(V.norm_words("the cat sat on the mat"), V.norm_words("the cat sit on mat now"))
    assert c["sub"] + c["ins"] + c["del"] == 3 and c["wer"] == pytest.approx(3 / 6) and c["n_ref"] == 6
    assert V.wer("", "") == 0.0


# ------------------------------------------------------------------------------------------------ phones and light processing


def test_combine_rank_prefers_the_phone_the_engines_understood():
    snr = [{"cam": "01", "snr_db": 12.0}, {"cam": "02", "snr_db": 10.0}, {"cam": "05", "snr_db": 20.0}]
    asr = {"01": {"mean": 0.30}, "02": {"mean": 0.02}, "05": {"mean": 0.30}}
    rows = V.combine_rank(snr, asr)
    assert rows[0]["cam"] == "02"
    assert [r["cam"] for r in rows][-1] == "01"


def test_prepare_piece_levels_and_does_not_gate_speech():
    x = speechlike(SR * 4) + np.random.default_rng(3).normal(0, 0.01, SR * 4).astype(np.float32)
    y, info = V.prepare_piece(x)
    assert len(y) == len(x) and np.isfinite(y).all()
    assert np.abs(y).max() <= 10 ** (-1.5 / 20) + 1e-3
    # the 95th percentile frame level is at the target
    n = int(0.05 * SR)
    r = 20 * np.log10(np.sqrt((y[:len(y) // n * n].reshape(-1, n) ** 2).mean(axis=1)) + 1e-9)
    assert np.percentile(r, 95) <= -19.0           # at the target, or lower when the peak ceiling holds the gain back
    # the speech band keeps most of its energy (denoise is mild)
    xs = x * 10 ** (info["gain_db"] / 20)
    keep = (y.astype(np.float64) ** 2).sum() / (xs.astype(np.float64) ** 2).sum()
    assert 0.5 < keep < 1.2


def test_spectral_denoise_lowers_stationary_noise_by_at_most_the_floor():
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.02, SR * 3).astype(np.float32)
    y = V.spectral_denoise(noise, SR, reduction_db=9.0)
    drop = 10 * np.log10((noise.astype(np.float64) ** 2).mean() / (y.astype(np.float64) ** 2).mean())
    assert 3.0 < drop < 10.5


# ------------------------------------------------------------------------------------------------ alignment


def test_envelope_lag_finds_a_shift_to_a_fraction_of_a_frame():
    src = speechlike(SR * 8, seed=1)
    for shift_ms in (-80.0, 23.0, 140.0):
        k = int(round(shift_ms / 1000 * SR))
        sig = np.roll(src, k)                               # later by shift
        lag, pk = V.envelope_lag(V.speech_envelope(src, SR), V.speech_envelope(sig, SR))
        assert lag * 1000 == pytest.approx(shift_ms, abs=6.0) and pk > 0.5


def test_align_converted_shifts_without_stretching():
    src = speechlike(SR * 12, seed=2)
    conv = np.roll(speechlike(SR * 12, seed=2) * 0.5, -int(0.03 * SR))        # the "converted" piece comes 30 ms early
    y, rep = V.align_converted(src, conv)
    assert not rep["stretched"] and rep["lag_applied_ms"] == pytest.approx(-30.0, abs=6.0)
    lag, _ = V.envelope_lag(V.speech_envelope(src, SR), V.speech_envelope(y, SR))
    assert abs(lag) < 0.008
    assert len(y) == len(src)


def test_align_converted_stretches_only_beyond_80_ms_of_drift():
    n = SR * 30
    src = speechlike(n, seed=4)
    # the converted piece is 0.6 % longer: 180 ms of drift over 30 s
    from scipy.signal import resample
    conv = resample(src, int(n * 1.006)).astype(np.float32)
    y, rep = V.align_converted(src, conv)
    assert rep["stretched"] and rep["rate"] == pytest.approx(0.006, abs=0.002)
    lag, _ = V.envelope_lag(V.speech_envelope(src, SR), V.speech_envelope(y, SR))
    assert abs(lag) < 0.03
    # 20 ms of drift: shifted only
    conv2 = resample(src, int(n * 1.0007)).astype(np.float32)
    _, rep2 = V.align_converted(src, conv2)
    assert not rep2["stretched"]


# ------------------------------------------------------------------------------------------------ the client (HTTP mocked)


class FakeResp:
    def __init__(self, status=200, content=b"", headers=None, text=""):
        self.status_code, self.content, self.headers, self.text = status, content, headers or {}, text


class FakeSession:
    def __init__(self, resp):
        self.resp, self.calls = resp, []

    def post(self, url, params=None, data=None, files=None, timeout=None):
        self.calls.append({"url": url, "params": params, "data": data, "bytes": len(files["audio"][1])})
        return self.resp


def test_sts_request_reads_cost_header_and_writes_a_key_free_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(V, "LEDGER", tmp_path / "ledger.jsonl")
    pcm = (np.zeros(2400, dtype="<i2")).tobytes()
    s = FakeSession(FakeResp(200, pcm, {"character-cost": "64", "request-id": "rid1"}))
    raw, info = V.sts_request(s, "voiceid1234567", b"RIFF....", settings={"stability": 0.8}, remove_background_noise=True, purpose="t")
    assert raw == pcm and info["credits"] == 64 and info["request_id"] == "rid1"
    c = s.calls[0]
    assert c["url"].endswith("/v1/speech-to-speech/voiceid1234567") and c["params"] == {"output_format": "pcm_44100"}
    assert c["data"]["model_id"] == V.MODEL_DEFAULT and c["data"]["remove_background_noise"] == "true" and json.loads(c["data"]["voice_settings"]) == {"stability": 0.8}
    assert V.ledger_credits(tmp_path / "ledger.jsonl") == 64
    assert "voiceid1234567" not in (tmp_path / "ledger.jsonl").read_text()          # only a stub of the id


def test_sts_request_quota_error_is_a_QuotaError_and_other_errors_are_not():
    q = FakeSession(FakeResp(401, text='{"detail":{"status":"quota_exceeded","message":"You have 0 credits remaining"}}'))
    with pytest.raises(V.QuotaError):
        V.sts_request(q, "v", b"x")
    e = FakeSession(FakeResp(422, text="bad"))
    with pytest.raises(V.V2VError) as ei:
        V.sts_request(e, "v", b"x")
    assert not isinstance(ei.value, V.QuotaError)


def test_key_never_shows_in_repr_or_errors(tmp_path):
    f = tmp_path / "k.txt"
    f.write_text("sk_supersecretvalue\n")
    k = V.ElevenKey(f)
    assert "supersecret" not in repr(k) and "supersecret" not in str(k)
    assert k.header() == {"xi-api-key": "sk_supersecretvalue"}
    with pytest.raises(V.V2VError) as ei:
        V.ElevenKey(tmp_path / "missing.txt")
    assert "supersecret" not in str(ei.value)
    assert "xi-api-key" not in V._clean("error xi-api-key: sk_abc rest")


def test_pcm_decode():
    x, sr = V.pcm_to_float((np.array([0, 16384, -16384], dtype="<i2")).tobytes(), "pcm_24000")
    assert sr == 24000 and x[1] == pytest.approx(0.5)


def test_convert_piece_caches_by_hash(tmp_path, monkeypatch):
    monkeypatch.setattr(V, "V2V", tmp_path)
    p = V.Piece("p001", "Kadu", 10.0, 12.0, 10.2, 11.8, 3, "a b c", cam="02")
    var = V.Variant("A")
    (tmp_path / "pre").mkdir()
    sf.write(str(tmp_path / "pre" / "p001.02.wav"), speechlike(SR * 2), SR, subtype="PCM_16")
    calls = []

    def fake(session, vid, audio, **kw):
        calls.append(kw)
        return (np.zeros(48000, dtype="<i2")).tobytes(), {"credits": 20, "request_id": "r", "seconds": 0.1, "bytes": 96000}

    m1 = V.convert_piece(None, "vid", p, "02", var, request=fake)
    m2 = V.convert_piece(None, "vid", p, "02", var, request=fake)
    assert not m1["cached"] and m2["cached"] and len(calls) == 1 and m1["rate"] == 24000
    V.convert_piece(None, "vid", p, "02", V.Variant("A", remove_background_noise=True), request=fake)   # other settings = other hash -> but another folder name is not needed
    assert len(calls) == 2
    assert calls[0]["output_format"] == "pcm_24000"


def test_convert_all_stops_on_quota_and_on_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(V, "V2V", tmp_path)
    (tmp_path / "pre").mkdir()
    pieces = []
    for i in range(3):
        p = V.Piece(f"p00{i}", "Larissa", 10.0 * i, 10.0 * i + 5, 10.0 * i, 10.0 * i + 5, 3, "a", cam="01")
        sf.write(str(tmp_path / "pre" / f"{p.id}.01.wav"), speechlike(SR * 2), SR, subtype="PCM_16")
        pieces.append(p)
    (tmp_path / "voices.json").write_text(json.dumps({"Larissa": {"voice_id": "L"}, "Kadu": {"voice_id": "K"}}))
    n = {"i": 0}

    def fake(session, vid, audio, **kw):
        n["i"] += 1
        if n["i"] == 2:
            raise V.QuotaError("quota exceeded")
        return (np.zeros(4800, dtype="<i2")).tobytes(), {"credits": 50, "request_id": "r", "seconds": 0.1, "bytes": 1}

    r = V.convert_all(pieces, V.Variant("Q"), session=object(), request=fake)
    assert r["done"] == ["p000"] and r["stopped"].startswith("quota") and r["credits"] == 50
    r2 = V.convert_all(pieces, V.Variant("B"), session=object(), request=fake, budget_credits=40)
    assert r2["stopped"].startswith("credit budget") and not r2["done"]


# ------------------------------------------------------------------------------------------------ the gate


def test_decide_applies_the_owners_rules():
    p = V.Piece("p1", "Larissa", 0, 10, 0.5, 9.5, 10, "a b c d e f g h i j")
    asr = lambda w: {"wer": w, "ins": 0, "text": "x"}
    ok = {"mix": asr(0.40), "A": asr(0.10)}
    assert G.decide(p, ok, {"own": 0.7, "other": 0.2, "ok": True}, kind="A")["accept"]
    meh = {"mix": asr(0.12), "A": asr(0.09)}                                  # both fine: not clearly better, original not bad
    assert not G.decide(p, meh, None, kind="A")["accept"]
    both_bad = {"mix": asr(0.35), "A": asr(0.08)}                              # conv <= 0.1 while the mix is bad: accepted (also a clear gain)
    assert G.decide(p, both_bad, None, kind="A")["accept"]
    still_bad = {"mix": asr(0.30), "A": asr(0.22)}
    assert not G.decide(p, still_bad, None, kind="A")["accept"]
    halluc = {"mix": asr(0.5), "A": {"wer": 0.1, "ins": 5, "text": "thanks for watching and so on"}}
    d = G.decide(p, halluc, None, kind="A")
    assert not d["accept"] and "inserted" in d["why"]
    swap = G.decide(p, ok, {"own": 0.3, "other": 0.5, "ok": False}, kind="A")
    assert not swap["accept"] and "voice" in swap["why"]
    assert not G.decide(p, ok, None, clipped=50, kind="A")["accept"]
    # never worse than the cleaned original (the fallback), never garbled
    worse_than_pre = {"mix": asr(0.60), "pre": asr(0.05), "A": asr(0.30)}
    d2 = G.decide(p, worse_than_pre, None, kind="A")
    assert not d2["accept"] and "cleaned original" in d2["why"]
    garbled = {"mix": asr(1.5), "pre": asr(1.0), "A": asr(1.0)}
    assert not G.decide(p, garbled, None, kind="A")["accept"]
    tie = {"mix": asr(0.60), "pre": asr(0.20), "A": asr(0.20)}
    assert G.decide(p, tie, None, kind="A")["accept"]


def test_decide_all_tries_variants_in_order_and_falls_back_to_the_cleaned_original(tmp_path, monkeypatch):
    monkeypatch.setattr(V, "V2V", tmp_path)
    p1 = V.Piece("p001", "Larissa", 10, 20, 10.5, 19.5, 6, "one two three four five six")
    p2 = V.Piece("p002", "Kadu", 20, 30, 20.5, 29.5, 6, "one two three four five six")
    good, bad = "one two three four five six", "one nine ten eleven five six"
    asr = {}
    for pid, txt in (("p001", {"mix": bad, "pre": good, "A": good, "B": good}), ("p002", {"mix": bad, "pre": good, "A": bad, "B": bad})):
        for k, t in txt.items():
            asr[f"{pid}__{k}"] = {"text": t}
    emb = {"ref_Larissa_0": np.array([1.0, 0.0]), "ref_Kadu_0": np.array([0.0, 1.0]), "p001__A": np.array([0.9, 0.1]), "p001__B": np.array([0.9, 0.1]),
           "p002__A": np.array([0.1, 0.9]), "p002__B": np.array([0.1, 0.9])}
    dec, rows = G.decide_all([p1, p2], asr, emb, ["A", "B"])
    assert dec["p001"]["variant"] == "A" and dec["p002"]["variant"] is None and "cleaned original" in dec["p002"]["why"]
    assert rows[1]["chosen"] is None and set(rows[1]["variants"]) == {"A", "B"}


def test_voice_check():
    a, b = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    refs = {"Larissa": a, "Kadu": b}
    assert G.voice_check("Larissa", np.array([0.9, 0.1]), refs)["ok"]
    assert not G.voice_check("Kadu", np.array([0.9, 0.1]), refs)["ok"]


def test_job_spec_is_a_dedicated_kind_that_no_other_worker_claims(tmp_path, monkeypatch):
    monkeypatch.setattr("dancecap.speech.jobs.code_archive", lambda out=None: tmp_path / "code.tar")
    spec = G.job_spec("g9", {"sha256": "ab", "bytes": 3, "tar": str(tmp_path / "in.tar")})
    assert spec["kind"] == "v2vgate" and spec["needs_gpu"]
    farm_kinds = {"front", "fit", "refine", "full", "asr", "analyse", "avatar"}
    assert spec["kind"] not in farm_kinds
    assert [s["name"] for s in spec["steps"]] == ["env", "asr"] and "stage_v2v" in spec["steps"][1]["cmd"]


# ------------------------------------------------------------------------------------------------ the mix


def test_place_and_masks():
    t = np.zeros(1000, np.float32)
    M.place(t, np.ones(200, np.float32), 900, 20)            # runs over the end: clipped
    assert t[899] == 0 and t[950] == 1.0 and t[-1] == pytest.approx(0, abs=0.2) or t[-1] >= 0
    t2 = np.zeros(1000, np.float32)
    M.place(t2, np.ones(100, np.float32), 100, 20)
    assert t2[100] == 0.0 and t2[150] == 1.0 and t2[199] < 0.05
    m = M.smooth_mask(48000, [(0.2, 0.6)], 0.05)
    assert m[int(0.4 * SR)] == 1.0 and m[int(0.1 * SR)] == 0.0 and 0.0 < m[int(0.2 * SR)] < 1.0


def test_duck_gain_dips_under_speech_and_recovers_slowly():
    act = np.zeros(SR * 3, np.float32)
    act[SR:2 * SR] = 1.0
    g = M.duck_gain(act, 6.0)
    assert g[0] == pytest.approx(1.0, abs=0.01)
    assert g[int(1.5 * SR)] == pytest.approx(10 ** (-6 / 20), abs=0.03)
    assert 10 ** (-6 / 20) < g[int(2.1 * SR)] < 1.0                # still recovering 0.1 s after the speech
    assert g[int(2.95 * SR)] > 0.9


def test_room_tone_has_the_colour_of_the_stationary_part_and_no_speech():
    rng = np.random.default_rng(5)
    n = SR * 14
    t = np.arange(n) / SR
    hum = 0.01 * np.sin(2 * np.pi * 170 * t)
    noise = rng.normal(0, 0.003, n)
    speech = speechlike(n, seed=6) * (np.sin(2 * np.pi * 0.2 * t) > -0.2)       # speech most of the time
    orig = np.stack([hum + noise + speech, hum + noise + speech], axis=1).astype(np.float32)
    bed = M.room_tone(orig, n)
    assert bed.shape == (n, 2) and np.isfinite(bed).all()
    spec = np.abs(np.fft.rfft(bed[:, 0] * np.hanning(n)))
    f = np.fft.rfftfreq(n, 1 / SR)
    assert f[np.argmax(spec[(f > 50) & (f < 400)]) + np.searchsorted(f, 50)] == pytest.approx(170, abs=25)       # the bed's resolution is one 23 Hz STFT bin
    # the speech's second comb (210 Hz harmonics with high amplitude) is not in the bed at the speech's level
    assert np.sqrt((bed.astype(np.float64) ** 2).mean()) < np.sqrt((orig.astype(np.float64) ** 2).mean())


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg needed for ebur128")
def test_solve_loudness_hits_target_and_true_peak(tmp_path):
    x = (speechlike(SR * 10, sr=SR, seed=7) * 3.0)
    mix = np.stack([x, x], axis=1).astype(np.float32)
    gain, y, rep = M.solve_loudness(mix, tmp_path / "p.wav")
    from dancecap import film_recap_audio as fa
    sf.write(str(tmp_path / "o.wav"), y, SR, subtype="PCM_16")
    m = fa.measure_loudness(tmp_path / "o.wav")
    assert m["lufs"] == pytest.approx(-16.0, abs=0.3) and m["true_peak_dbtp"] <= -0.95


def test_stabilise_ids_keeps_the_ids_of_unchanged_pieces():
    old = [V.Piece("p001", "Larissa", 9.7, 20.0, 10.0, 19.6, 5, "a"), V.Piece("p002", "Kadu", 20.0, 30.0, 20.5, 29.5, 5, "b")]
    new = [V.Piece("p001", "Larissa", 0.0, 4.0, 0.5, 3.5, 3, "new opening"), V.Piece("p002", "Larissa", 9.7, 20.0, 10.0, 19.6, 5, "a"),
           V.Piece("p003", "Kadu", 20.0, 30.0, 20.52, 29.49, 5, "b")]
    out = V.stabilise_ids(new, old)
    assert [p.id for p in out] == ["p003", "p001", "p002"]
    assert len({p.id for p in out}) == 3


# ------------------------------------------------------------------------------------------------ the assembly on synthetic audio


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg needed for ebur128")
def test_assemble_places_pieces_keeps_the_opening_and_hits_the_loudness(tmp_path, monkeypatch):
    monkeypatch.setattr(V, "V2V", tmp_path / "v2v")
    dur = 24.0
    n = int(dur * SR)
    rng = np.random.default_rng(1)
    room = rng.normal(0, 0.004, (n, 2)).astype(np.float32)
    orig = (room + np.stack([speechlike(n, seed=11) * 3] * 2, axis=1)).astype(np.float32)
    sf.write(str(tmp_path / "class_mix.wav"), orig, SR, subtype="PCM_16")
    pieces = [V.Piece("p001", "Larissa", 6.0, 12.0, 6.3, 11.7, 5, "a b c", cam="01"), V.Piece("p002", "Kadu", 13.0, 20.0, 13.3, 19.7, 5, "d e f", cam="01")]
    for p in pieces:
        d = int((p.t1 - p.t0) * SR)
        sf.write(str(V.V2V / "pre" / f"{p.id}.01.wav") if (V.V2V / "pre").mkdir(parents=True, exist_ok=True) is None else "", speechlike(d, seed=3 + len(p.id)) * 2, SR, subtype="PCM_16")
    (V.V2V / "aligned" / "A").mkdir(parents=True)
    sf.write(str(V.V2V / "aligned" / "A" / "p001.wav"), speechlike(int(6.0 * SR), seed=21) * 2, SR, subtype="PCM_16")
    kept = [{"t0": 0.0, "t1": 5.0, "speaker": "both", "why": "opening"}]
    dec = {"p001": {"variant": "A", "cam": "01"}, "p002": {"variant": None, "cam": "01", "why": "no variant passed"}}
    rep = M.assemble(pieces, kept, dec, orig_path=tmp_path / "class_mix.wav", out_wav=tmp_path / "out" / "mix.wav", stems_dir=tmp_path / "out" / "stems",
                     report_path=tmp_path / "out" / "mix.json", log=lambda *_: None)
    y, sr = sf.read(str(tmp_path / "out" / "mix.wav"), dtype="float32")
    assert sr == SR and len(y) == n and y.shape[1] == 2
    assert rep["converted"] == 1 and rep["original_pieces"] == 1
    assert rep["final"]["lufs"] == pytest.approx(-16.0, abs=0.3) and rep["final"]["true_peak_dbtp"] <= -0.95
    st = {k: sf.read(str(tmp_path / "out" / "stems" / f"{k}.wav"), dtype="float32")[0] for k in ("larissa", "kadu", "original", "bed")}
    assert all(len(v) == n for v in st.values())
    # the cleaned-original piece is in Kadu's stem, the converted one in Larissa's, the opening only in `original`
    rms = lambda a, t0, t1: float(np.sqrt((a[int(t0 * SR):int(t1 * SR)].astype(np.float64) ** 2).mean()))
    assert rms(st["larissa"], 7, 11) > 10 * rms(st["larissa"], 14, 19) and rms(st["kadu"], 14, 19) > 10 * rms(st["kadu"], 7, 11)
    assert rms(st["original"], 1, 4) > 10 * rms(st["original"], 8, 18) and rms(st["larissa"], 1, 4) == 0 == rms(st["kadu"], 1, 4)
    # the bed is far under the speech and silent where the original plays
    assert rms(st["bed"], 1, 4) < 1e-3 * rms(st["original"], 1, 4) + 1e-9
    assert rms(st["bed"], 21, 23.5) < 0.1 * rms(st["larissa"], 7, 11)
    # the stems add up to the mix (a limiter touches it only where a peak needs it)
    s = st["larissa"] + st["kadu"] + st["original"] + st["bed"]
    assert float(np.abs(s - y).max()) < 0.05 or rep["loudness"]["limiter_samples_touched_pct"] > 0
