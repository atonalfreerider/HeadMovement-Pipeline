"""dancecap.film_recap_audio: the class audio mix (alignment, source choice, splices, loudness) and the film-clock soundtrack (synthetic audio only).

    .venv-nopo4d/Scripts/python.exe -m pytest dancecap/tests/test_film_recap_audio.py -q
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
from dancecap import film_recap_audio as fa  # noqa: E402

SR = 16000
HAVE_FFMPEG = shutil.which("ffmpeg") is not None


def speechlike(n, sr=SR, seed=0):
    """band-limited noise bursts with a pitch-like comb: stands in for voiced speech"""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / sr
    x = np.zeros(n)
    for f0 in (140.0, 200.0):
        for h in range(1, 18):
            x += np.sin(2 * np.pi * f0 * h * t + rng.uniform(0, 6.28)) / h
    env = (np.sin(2 * np.pi * 0.7 * t) > -0.2).astype(float) * (0.5 + 0.5 * np.sin(2 * np.pi * 3.1 * t) ** 2)
    return (x * env * 0.08 + rng.normal(0, 0.003, n)).astype(np.float32)


# ------------------------------------------------------------------------------------------------ the master clock


@pytest.mark.parametrize("off,slope", [(0.0, 1.0), (1.3440033, 1.0), (-8.9023, 1.0000145503), (7.123456, 0.9999873421)])
def test_warp_follows_a_clock_to_a_fraction_of_a_sample(off, slope):
    rng = np.random.default_rng(1)
    freqs, ph, amp = rng.uniform(100, 3000, 20), rng.uniform(0, 6.28, 20), rng.uniform(0.2, 1, 20) / 20
    f = lambda t: (amp[None, :] * np.sin(2 * np.pi * freqs[None, :] * t[:, None] + ph[None, :])).sum(1)
    n_src, n_out = SR * 12, SR * 11
    x = f(np.arange(n_src) / SR).astype(np.float32)
    y, (a, b) = fa.warp_to_master(x, off, slope, n_out, SR)
    t = (np.arange(n_out) / SR - off) / slope
    ok = (t > 0.3) & (t < 11.7)
    err = np.abs(y[:, 0] - f(np.clip(t, 0, None)).astype(np.float32))[ok]
    assert err.max() < 2e-3
    assert a >= 0 and b > a


def test_warp_reports_where_the_source_lies_on_the_master():
    x = np.ones(SR * 4, dtype=np.float32)
    y, (a, b) = fa.warp_to_master(x, 2.0, 1.0, SR * 10, SR)
    assert abs(a - 2 * SR) <= 1 and abs(b - 6 * SR) <= 2
    assert np.abs(y[: SR, 0]).max() < 1e-6 and y[3 * SR, 0] > 0.99


def test_phat_finds_the_lag_that_a_plain_correlation_of_voiced_speech_misses():
    x = speechlike(SR * 6, seed=2)
    n = len(x)
    for lag in (0, 7, -23, 61):
        y = np.roll(x, lag)
        y = y + np.random.default_rng(5).normal(0, 0.004, n).astype(np.float32)
        est, z = fa.xcorr_lag(x, y, SR, 0.02)
        assert abs(est - lag) < 0.3 and z > 6


def test_the_lag_of_a_stretch_is_the_median_of_its_windows_and_none_when_unreliable():
    x = speechlike(SR * 20, seed=3)
    y = np.roll(x, 9) + np.random.default_rng(6).normal(0, 0.003, len(x)).astype(np.float32)
    r = fa.robust_lag(x, y, SR)
    assert r["lag_samples"] is not None and abs(r["lag_samples"] - 9) < 0.3 and r["mad_ms"] < 0.2
    noise = np.random.default_rng(7).normal(0, 0.05, len(x)).astype(np.float32)
    r2 = fa.robust_lag(x, noise, SR)
    assert r2["lag_samples"] is None


# ------------------------------------------------------------------------------------------------ which phone, where


def levels_for(duration, quality):
    """per-phone band levels (dB, 50 ms frames) of a talk with pauses; `quality` = {cam: (speech dB, floor dB)}"""
    n = int(duration / fa.FRAME_S)
    t = np.arange(n) * fa.FRAME_S
    talk = (np.sin(2 * np.pi * t / 5.0) > -0.3)
    out = {}
    for cam, (s, nz) in quality.items():
        rng = np.random.default_rng(int(cam))
        env = np.where(talk, s, nz) + rng.normal(0, 0.5, n)
        out[cam] = env
    return out


def test_another_phone_replaces_the_master_only_when_clearly_better_and_long_enough():
    dur = 80.0
    lv = levels_for(dur, {"01": (-30, -48), "02": (-26, -52), "05": (-30.5, -47)})
    cover = {c: (0.0, dur) for c in lv}
    wins = fa.window_table(lv, cover, duration=dur)
    st = fa.plan_sources(wins)
    assert st and all(s["cam"] == "02" for s in st)                 # 02: +4 dB speech and -4 dB floor, i.e. 8 dB more SNR
    assert all(s["t1"] - s["t0"] >= fa.MIN_STRETCH_S for s in st)
    assert all(s["cam"] != "05" for s in st)                       # 05 is within the margin of the master
    # nobody is clearly better: the master stays
    lv2 = levels_for(dur, {"01": (-30, -48), "02": (-29.5, -48.5), "05": (-30.5, -47.5)})
    assert fa.plan_sources(fa.window_table(lv2, {c: (0.0, dur) for c in lv2}, duration=dur)) == []
    # a phone that does not cover the whole window takes no part
    cover3 = {"01": (0.0, dur), "02": (30.0, dur), "05": (0.0, dur)}
    st3 = fa.plan_sources(fa.window_table(lv, cover3, duration=dur))
    assert all(s["t0"] >= 29.9 for s in st3 if s["cam"] == "02")


def test_a_splice_goes_where_the_room_is_quietest_and_leaves_no_click():
    sr = fa.SR
    base_db = np.full(int(30 / fa.FRAME_S), -30.0)
    base_db[int(10.2 / fa.FRAME_S)] = -60.0                           # the quietest instant near 10 s
    assert abs(fa.quietest_point(base_db, 9.5) - 10.2) < 0.06
    n = sr * 3
    t = np.arange(n) / sr
    a = np.stack([0.2 * np.sin(2 * np.pi * 220 * t)] * 2, axis=1).astype(np.float32)
    b = np.stack([0.2 * np.sin(2 * np.pi * 220 * t + 1.1)] * 2, axis=1).astype(np.float32)
    xf = int(fa.XFADE_S * sr)
    out = fa.splice(a, b, sr, 2 * sr, xf)
    assert np.allclose(out[: sr - xf], a[: sr - xf]) and np.allclose(out[sr + xf: 2 * sr - xf], b[sr + xf: 2 * sr - xf])
    assert np.allclose(out[2 * sr + xf:], a[2 * sr + xf:])
    # no click: the largest sample step in the crossfade is that of a plain sine (+ the small step of the phase difference)
    step = np.abs(np.diff(out[:, 0])).max()
    assert step < 0.2 * 2 * np.pi * 220 / sr * 1.5
    fo, fi = fa.crossfade_curves(100)
    assert abs(fo[0] - 1) < 1e-6 and abs(fi[-1] - 1) < 1e-6 and np.allclose(fo + fi, 1.0)


def test_the_limiter_never_passes_the_ceiling_and_leaves_quiet_audio_alone():
    sr = fa.SR
    t = np.arange(sr * 4) / sr
    x = (0.05 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    x[sr * 2: sr * 2 + 50] = 1.6                                       # a 1 ms burst far above the ceiling
    x = np.stack([x, x], axis=1)
    y = fa.lookahead_limit(x, -1.5, sr)
    assert np.abs(y).max() <= 10 ** (-1.5 / 20) + 1e-6
    quiet = x[: sr]
    assert np.array_equal(fa.lookahead_limit(quiet, -1.5, sr), quiet)
    # away from the burst nothing changes (the dip is +-20 ms wide)
    assert np.allclose(y[: sr], x[: sr]) and np.allclose(y[3 * sr:], x[3 * sr:])


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg is not installed")
def test_loudness_measurement_and_the_final_stage_hit_the_targets(tmp_path):
    sr = fa.SR
    rng = np.random.default_rng(3)
    t = np.arange(sr * 20) / sr
    speech = speechlike(len(t), sr)
    speech = speech * 4.0
    speech[sr * 5: sr * 5 + 20] = 0.95                                  # a loud transient: the limiter has work to do
    x = np.stack([speech, speech], axis=1).astype(np.float32)
    x += rng.normal(0, 0.004, x.shape).astype(np.float32)
    out = tmp_path / "mix.wav"
    res = fa._finish(x, out, denoise=False)
    m = res["after"]
    assert abs(m["lufs"] - fa.TARGET_LUFS) <= 0.25
    assert m["true_peak_dbtp"] <= fa.TARGET_TP + 1e-6
    y, sr_out = sf.read(str(out))
    assert sr_out == fa.SR and y.ndim == 2 and y.shape[1] == 2 and np.abs(y).max() < 0.95
    again = fa.measure_loudness(out)
    assert abs(again["lufs"] - m["lufs"]) < 0.05


def test_the_loudness_text_is_parsed():
    class R:
        stderr = "...\n  Summary:\n\n  Integrated loudness:\n    I:         -16.0 LUFS\n    Threshold: -26.0 LUFS\n\n  Loudness range:\n    LRA:         9.1 LU\n\n  True peak:\n    Peak:       -1.4 dBFS\n"
        returncode = 0

    orig = fa.ffmpeg
    fa.ffmpeg = lambda *a, **k: R()
    try:
        m = fa.measure_loudness(Path("x.wav"))
    finally:
        fa.ffmpeg = orig
    assert m == {"lufs": -16.0, "lra": 9.1, "true_peak_dbtp": -1.4}


# ------------------------------------------------------------------------------------------------ the film soundtrack


def test_the_film_mix_copies_locked_segments_and_plays_the_room_tone_in_inserts(tmp_path):
    sr = fa.SR
    n = sr * 20
    t = np.arange(n) / sr
    tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)          # "speech" everywhere ...
    quiet = (np.random.default_rng(0).normal(0, 0.002, n)).astype(np.float32)
    mix = np.stack([tone, tone], axis=1)
    mix[sr * 9: sr * 17] = quiet[sr * 9: sr * 17, None]                    # ... except a quiet room-tone stretch
    cm = tmp_path / "audio" / "class_mix.wav"
    cm.parent.mkdir(parents=True)
    sf.write(str(cm), mix, sr, subtype="PCM_16")
    direction = {"name": "film", "timeline": {"film_duration_s": 24.0},
                 "segments": [{"id": "a", "film": [0, 10], "dance": [0, 10], "speed": 1.0, "audio": "locked"},
                              {"id": "r", "film": [10, 14], "dance": [3, 5], "speed": 0.5, "audio": "bed", "replay": True},
                              {"id": "b", "film": [14, 24], "dance": [10, 20], "speed": 1.0, "audio": "locked"}]}
    dp = tmp_path / "direction.json"
    dp.write_text(json.dumps(direction), encoding="utf-8")
    res = fa.film_mix(dp, cm, tmp_path / "audio" / "film_mix.wav", word_spans=[])
    assert res["bed_segments"] == 1
    y, rate = sf.read(str(tmp_path / "audio" / "film_mix.wav"))
    assert rate == sr and len(y) == 24 * sr
    # locked: sample-exact copies at film = dance (+ the inserted 4 s after the replay)
    assert np.allclose(y[sr: 9 * sr, 0], mix[sr: 9 * sr, 0], atol=2e-4)
    assert np.allclose(y[15 * sr: 23 * sr, 0], mix[11 * sr: 19 * sr, 0], atol=2e-4)
    # the insert plays the room tone, not the 440 Hz "speech" of the replayed seconds
    mid = y[11 * sr: 13 * sr, 0]
    assert np.abs(mid).max() < 0.02


def test_the_room_tone_loop_avoids_speech_and_loops_seamlessly():
    sr = fa.SR
    n = sr * 30
    t = np.arange(n) / sr
    x = np.stack([(0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)] * 2, axis=1)
    x[sr * 18: sr * 26] = np.random.default_rng(1).normal(0, 0.003, (sr * 8, 2)).astype(np.float32)
    loop = fa.room_tone(x, [(0.0, 17.5), (26.5, 30.0)], sr, length_s=4.0)
    assert np.abs(loop).max() < 0.05
    tiled = fa.tile(loop, len(loop) * 3)
    seam = np.abs(np.diff(tiled[:, 0]))[len(loop) - 2: len(loop) + 2].max()
    assert seam < 0.02


def test_the_filter_latency_is_measured_and_removed():
    sr = fa.SR
    x = speechlike(sr * 60, sr, seed=11)
    ref = np.stack([x, x], axis=1)
    d = 1198                                                              # afftdn's delay on the real class: 24.96 ms
    proc = np.zeros_like(ref)
    proc[d:] = ref[:-d] * 0.7
    out, got = fa.remove_latency(ref, proc)
    assert got == d
    k = slice(sr, 50 * sr)
    assert np.allclose(out[k, 0] / 0.7, ref[k, 0], atol=1e-4)
    # nothing to remove: untouched
    same, zero = fa.remove_latency(ref, ref * 0.5)
    assert zero == 0 and np.array_equal(same, ref * 0.5)


# ------------------------------------------------------------------------------------------------ the soundtrack's class mix (V1: voice-to-voice)


def test_soundtrack_class_mix_prefers_v2v_with_a_switch_and_an_original_fallback(tmp_path):
    (tmp_path / "class_mix.wav").write_bytes(b"x")
    assert fa.soundtrack_class_mix(tmp_path) == tmp_path / "class_mix.wav"                  # no v2v mix yet: the original
    (tmp_path / "class_mix_v2v.wav").write_bytes(b"x")
    assert fa.soundtrack_class_mix(tmp_path) == tmp_path / "class_mix_v2v.wav"              # auto: v2v when present
    (tmp_path / "soundtrack.json").write_text(json.dumps({"class_mix": "original"}))
    assert fa.soundtrack_class_mix(tmp_path) == tmp_path / "class_mix.wav"                  # the switch forces the old mix
    (tmp_path / "soundtrack.json").write_text(json.dumps({"class_mix": "v2v"}))
    assert fa.soundtrack_class_mix(tmp_path) == tmp_path / "class_mix_v2v.wav"
    (tmp_path / "class_mix_v2v.wav").unlink()
    assert fa.soundtrack_class_mix(tmp_path) == tmp_path / "class_mix.wav"                  # "v2v" without the file: fallback
    (tmp_path / "soundtrack.json").write_text("{not json")
    assert fa.soundtrack_class_mix(tmp_path) == tmp_path / "class_mix.wav"


def test_soundtrack_class_mix_v2v_full_switch_and_fallbacks(tmp_path):
    cfg = tmp_path / "soundtrack.json"
    assert fa.soundtrack_class_mix(tmp_path) == tmp_path / "class_mix.wav"
    (tmp_path / "class_mix_v2v.wav").write_bytes(b"x")
    assert fa.soundtrack_class_mix(tmp_path) == tmp_path / "class_mix_v2v.wav"
    (tmp_path / "class_mix_v2v_full.wav").write_bytes(b"x")
    assert fa.soundtrack_class_mix(tmp_path) == tmp_path / "class_mix_v2v_full.wav"          # auto: the newest
    cfg.write_text(json.dumps({"class_mix": "v2v"}))
    assert fa.soundtrack_class_mix(tmp_path) == tmp_path / "class_mix_v2v.wav"               # V1's mix stays selectable
    cfg.write_text(json.dumps({"class_mix": "v2v_full"}))
    assert fa.soundtrack_class_mix(tmp_path) == tmp_path / "class_mix_v2v_full.wav"
    cfg.write_text(json.dumps({"class_mix": "original"}))
    assert fa.soundtrack_class_mix(tmp_path) == tmp_path / "class_mix.wav"
    (tmp_path / "class_mix_v2v_full.wav").unlink()
    cfg.write_text(json.dumps({"class_mix": "v2v_full"}))
    assert fa.soundtrack_class_mix(tmp_path) == tmp_path / "class_mix_v2v.wav"               # v2v_full missing: falls back to V1's mix


def test_soundtrack_class_mix_regen_switch_and_fallbacks(tmp_path):
    cfg = tmp_path / "soundtrack.json"
    for name in ("class_mix.wav", "class_mix_v2v.wav", "class_mix_v2v_full.wav"):
        (tmp_path / name).write_bytes(b"x")
    assert fa.soundtrack_class_mix(tmp_path) == tmp_path / "class_mix_v2v_full.wav"          # no regenerated mix yet
    (tmp_path / "class_mix_regen.wav").write_bytes(b"x")
    assert fa.soundtrack_class_mix(tmp_path) == tmp_path / "class_mix_regen.wav"             # auto: the newest, the regenerated lines
    cfg.write_text(json.dumps({"class_mix": "regen"}))
    assert fa.soundtrack_class_mix(tmp_path) == tmp_path / "class_mix_regen.wav"
    cfg.write_text(json.dumps({"class_mix": "v2v_full"}))
    assert fa.soundtrack_class_mix(tmp_path) == tmp_path / "class_mix_v2v_full.wav"          # the earlier mixes stay selectable
    cfg.write_text(json.dumps({"class_mix": "v2v"}))
    assert fa.soundtrack_class_mix(tmp_path) == tmp_path / "class_mix_v2v.wav"
    cfg.write_text(json.dumps({"class_mix": "original"}))
    assert fa.soundtrack_class_mix(tmp_path) == tmp_path / "class_mix.wav"
    (tmp_path / "class_mix_regen.wav").unlink()
    cfg.write_text(json.dumps({"class_mix": "regen"}))
    assert fa.soundtrack_class_mix(tmp_path) == tmp_path / "class_mix_v2v_full.wav"          # regen missing: falls back down the list
    (tmp_path / "class_mix_v2v_full.wav").unlink()
    assert fa.soundtrack_class_mix(tmp_path) == tmp_path / "class_mix_v2v.wav"
