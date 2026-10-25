"""Unit tests for dancecap.heat_beats (synthetic audio / beats: no model, no GPU, no pod, no footage).

    .venv-nopo4d\\Scripts\\python.exe -m pytest dancecap/tests/test_heat_beats.py -q      (from the workspace root)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dancecap import heat_beats as hb  # noqa: E402

SR = 22050
P, T0 = 0.8, 0.2  # 75 BPM


def synth(duration: float = 30.0, k_first: int = 4, k_last: int = 24, seed: int = 0) -> np.ndarray:
    """Noise everywhere, a 60 Hz kick on every pulse k_first..k_last (the 'music')."""
    rng = np.random.default_rng(seed)
    x = 0.01 * rng.standard_normal(int(duration * SR))
    t = np.arange(int(0.25 * SR)) / SR
    kick = 0.8 * np.sin(2 * np.pi * 60 * t) * np.exp(-t / 0.06)
    for k in range(k_first, k_last + 1):
        i = int((T0 + P * k) * SR)
        x[i:i + len(kick)] += kick[: len(x) - i]
    return x


def fake_tracker(duration: float = 30.0, k_first: int = 0, jitter_ms: float = 6.0, seed: int = 1) -> dict:
    """beat_this-like output: a beat on every pulse of the whole clip (it tracks noise too), 2-pulse downbeats."""
    rng = np.random.default_rng(seed)
    k = np.arange(k_first, int((duration - T0) / P))
    b = T0 + P * k + rng.normal(0, jitter_ms / 1000, len(k))
    return {"beats_refined": b, "downbeats_refined": b[::2], "beat_prob": np.ones(int(duration * 50))}


# ------------------------------------------------------------------------------------------------ pure helpers
def test_flux_at_limits():
    F = np.arange(100, dtype=float)
    assert hb.flux_at(F, 0.1, 5.0, 0.15) == 51.0  # frames 48..51
    assert hb.flux_at(F, 0.1, 0.0, 0.05) is None  # starts before the signal
    assert hb.flux_at(F, 0.1, 9.99, 0.05) is None  # ends after


def test_rolling_max_and_level_runs():
    a = np.array([0, 0, 5, 0, 0, 0, 0, 0, 5, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0], float)
    r = hb.rolling_max(a, 3)
    assert r[1] == 5 and r[3] == 5 and r[5] == 0
    t = np.arange(20) * 0.1
    lv = np.where((t > 0.5) & (t < 1.5), 10.0, 0.0)
    runs = hb.level_runs(t, lv, 5.0, min_s=0.5, gap_s=0.0)
    assert len(runs) == 1 and abs(runs[0][0] - 0.6) < 1e-9 and abs(runs[0][1] - 1.5) < 1e-9
    # a gap shorter than gap_s merges two runs; a run shorter than min_s is dropped
    lv2 = np.zeros(40)
    lv2[2:10] = 9
    lv2[11:20] = 9
    lv2[30:32] = 9
    t2 = np.arange(40) * 0.1
    runs2 = hb.level_runs(t2, lv2, 5.0, min_s=0.5, gap_s=0.25)
    assert len(runs2) == 1 and runs2[0][0] == 0.2 and abs(runs2[0][1] - 2.0) < 1e-9


def test_snap_to_pulses_respects_the_phase():
    t, k = hb.snap_to_pulses(3.5, T0, P, bar=4, phase=0)
    assert k == 4 and abs(t - (T0 + 4 * P)) < 1e-12
    t, k = hb.snap_to_pulses(3.5, T0, P, bar=4, phase=1)
    assert k == 5
    t, k = hb.snap_to_pulses(3.5, T0, P, bar=1, phase=0)  # any pulse
    assert k == 4


def test_music_segments():
    s = hb.music_segments((3.4, 80.2), 88.1)
    assert [x["state"] for x in s] == ["intro", "music", "outro"]
    assert s[0]["from_s"] == 0.0 and s[-1]["to_s"] == 88.1 and s[1]["from_s"] == 3.4
    assert [x["state"] for x in hb.music_segments((0.0, 88.1), 88.1)] == ["music"]
    assert hb.music_segments(None, 10.0) == [{"from_s": 0.0, "to_s": 10.0, "state": "none"}]


def test_presence_ramp():
    r = hb.presence_ramp(np.array([0.0, hb.Z_LO, 0.5 * (hb.Z_LO + hb.Z_HI), hb.Z_HI, 10.0, np.nan]))
    assert np.allclose(r, [0, 0, 0.5, 1, 1, 0])


def test_bar_structure_phase():
    assert hb.bar_structure_phase([4, 100]) == {"phase": 0, "agreement": 1.0, "n": 2}
    assert hb.bar_structure_phase([5, 101, 4])["phase"] == 1
    assert hb.bar_structure_phase([]) == {"phase": None, "agreement": 0.0, "n": 0}


def test_fold_lag_finds_a_known_shift():
    dt = 0.005
    pulses = T0 + P * np.arange(2, 30)
    F = np.zeros(int(25 / dt))
    for p in pulses:
        i = int(round((p + 0.010) / dt))  # onsets 10 ms (two frames) after the lattice
        if i < len(F):
            F[i] = 1.0
    lag = hb.fold_lag(F, dt, pulses[pulses < 24.0])
    assert lag is not None and abs(lag - 0.010) < 0.0015


# ------------------------------------------------------------------------------------------------ contrast
def test_lattice_contrast_high_on_the_kick_low_on_noise():
    x = synth()
    from dancecap import beats as B
    from dancecap import song as S
    hop = int(round(S.HOP_S * SR))
    F = B.band_onset(x, SR, 30.0, 150.0, hop=hop)[0]
    dt = hop / SR
    z_music = hb.lattice_contrast(F, dt, T0, P, [12.0])[0]
    z_noise = hb.lattice_contrast(F, dt, T0, P, [27.0])[0]
    assert z_music > 6.0
    assert abs(z_noise) < 3.0
    # a window with fewer than 3 pulses is nan
    assert np.isnan(hb.lattice_contrast(F, dt, T0, P, [0.5], win_s=1.0)[0])


# ------------------------------------------------------------------------------------------------ the whole analysis
def test_analyse_finds_the_music_span_the_tempo_and_the_bar_phase():
    x = synth(30.0, 4, 24)
    a = hb.analyse(x, SR, fake_tracker(30.0), 30.0)
    assert abs(60.0 / a["model"]["period"] - 75.0) < 0.05
    assert a["span"] is not None
    # the music starts on pulse 4 (3.4 s) and its last pulse is 24: the span ends at the end of pulse 24 (pulse 25, 20.2 s)
    assert abs(a["span"][0] - (T0 + 4 * P)) < 1e-3 and abs(a["span"][1] - (T0 + 25 * P)) < 0.05
    assert a["edges_k"] == [4, 25]
    # whole measures only, inside the span: pulses 4..24 = 21 pulses = 5 whole bars (the 6th is partial) from the chosen phase
    g = a["gov"]
    assert len(g) % 8 == 0 and len(g) >= 8 * 4
    assert g[0, 0] >= a["span"][0] - 1e-6 and g[-1, 0] <= a["span"][1] + 1e-6
    assert list(g[:8, 1]) == [1, 3, 3, 2, 1, 3, 2, 3]  # the zouk slot types
    assert np.all(np.diff(g[:, 0]) > 0)
    # the pulses outside the music keep a low confidence, inside a high one
    t = a["t_pulse"]
    c = a["conf"]
    assert c[(t > 6) & (t < 18)].min() > 0.9
    assert c[(t > 24) & (t < 29)].max() < 0.2


def test_analyse_phase_override_and_noise_only_clip():
    x = synth(30.0, 4, 24)
    a = hb.analyse(x, SR, fake_tracker(30.0), 30.0, phase=2)
    assert a["phase"] == 2 and a["decision"]["confidence"] == "given"
    # a clip with no music at all has no span and no governing beats
    rng = np.random.default_rng(3)
    noise = 0.01 * rng.standard_normal(30 * SR)
    b = hb.analyse(noise, SR, fake_tracker(30.0), 30.0)
    assert b["span"] is None and len(b["gov"]) == 0


def test_validate_reports_the_kick_lag_and_a_clean_tracker():
    x = synth(30.0, 4, 24)
    tr = fake_tracker(30.0)
    a = hb.analyse(x, SR, tr, 30.0)
    v = hb.validate(x, SR, a, tr)
    assert v["beat_this_vs_lattice"]["n_within_third_period"] == len(tr["beats_refined"])
    assert abs(v["beat_this_vs_lattice"]["median_ms"]) < 3.0
    assert v["per_beat_onset_support"]["bass_in_music"] is not None


def test_merged_rows_flag_governing_and_weak_beats():
    x = synth(30.0, 4, 24)
    a = hb.analyse(x, SR, fake_tracker(30.0), 30.0)
    m = hb.merged_rows(a)
    col = {c: i for i, c in enumerate(m["columns"])}
    rows = m["rows"]
    gov_t = {round(float(t), 6) for t in a["gov"][:, 0]}
    assert {r[col["t"]] for r in rows if r[col["governing"]]} == gov_t
    assert all(r[col["in_music"]] for r in rows if r[col["governing"]])
    assert max(r[col["confidence"]] for r in rows if r[col["t"]] > 26) < 0.3


# ------------------------------------------------------------------------------------------------ the take config
def test_john_lula_toml_is_a_single_view_take_with_the_chunk_arithmetic():
    import tomllib

    from dancecap.farm.chunks import equal_windows
    from dancecap.take import PACKAGE_ROOT
    cfg = tomllib.loads((PACKAGE_ROOT / "takes" / "john_lula.toml").read_text(encoding="utf-8"))
    assert cfg["name"] == "john_lula" and cfg["alignment"].endswith("work/john_lula/clock/alignment.json")
    assert "sync" not in cfg  # one camera: nothing to sync against
    assert set(k for k in cfg["cameras"]) == {"01"}
    assert cfg["cameras"]["01"]["mount"] == "handheld" and cfg["cameras"]["01"]["zoom"] is True
    assert abs(cfg["dancers"]["lead"]["stature_m"] - 5.9167 * 0.3048) < 1e-3  # 5'11"
    assert abs(cfg["dancers"]["follow"]["stature_m"] - 5.5833 * 0.3048) < 1e-3  # 5'7"
    assert cfg["dancers"]["follow"]["hair_style"] == "ponytail"
    assert "song_file" not in cfg["dance"] and cfg["dance"]["song"] == "unknown"
    # the grid: 30 Hz anchored at 0, frames k = 0..2642 (the last frame's pts is 88.084 s)
    t0, t1 = cfg["window"]
    n = int(np.floor(t1 * 30.0 + 1e-9)) + 1
    assert t0 == 0.0 and n == 2643
    c = cfg["chunks"]
    r = equal_windows(n, 30.0, c["chunk_s"], c["overlap_s"])
    assert r["n"] == 2 and r["windows"] == [[0, 1381], [1262, 2642]] and r["overlap_k"] == 120


def synth_mid(duration: float = 20.0, seed: int = 0) -> np.ndarray:
    """A phone in a studio: the beat is audible all through, but only in the 1-4 kHz band (a 2 kHz click on every pulse), no bass."""
    rng = np.random.default_rng(seed)
    n = int(duration * SR)
    x = 0.01 * rng.standard_normal(n)
    # the room's steady low rumble (60-120 Hz tones with a slow random level): the bass band carries no lattice
    tt = np.arange(n) / SR
    lvl = 0.3 * (1.0 + 0.3 * np.interp(tt, np.arange(0, duration + 1, 0.37), rng.standard_normal(len(np.arange(0, duration + 1, 0.37)))))
    x += lvl * (np.sin(2 * np.pi * 61 * tt) + 0.7 * np.sin(2 * np.pi * 97 * tt + 1.0) + 0.5 * np.sin(2 * np.pi * 131 * tt + 2.0))
    t = np.arange(int(0.05 * SR)) / SR
    click = 0.5 * np.sin(2 * np.pi * 2000 * t) * np.exp(-t / 0.01)
    for k in range(int((duration - T0) / P)):
        i = int((T0 + P * k) * SR)
        x[i:i + len(click)] += click[: len(x) - i]
    return x


def test_a_bass_less_clip_that_is_music_all_through_still_gets_a_grid():
    """2026-10-22 (lk03_mono20, a 20-s phone cut from the middle of a song): the bass band never reached MUSIC_Z, so the whole
    clip was 'no beat' and beats_heat.json was empty; the clearest band now decides at FALLBACK_Z and says so."""
    a = hb.analyse(synth_mid(), SR, fake_tracker(20.0), 20.0)
    assert a["span"] is not None and a["music_band"] == "mid_1_4k" and len(a["gov"]) >= 8
    b = hb.analyse(0.01 * np.random.default_rng(3).standard_normal(20 * SR), SR, fake_tracker(20.0), 20.0)
    assert b["span"] is None and b["music_band"] is None              # noise only: still no music
    c = hb.analyse(synth(30.0), SR, fake_tracker(30.0), 30.0)
    assert c["music_band"] == "bass_30_150"                           # a bass kick keeps the original rule


def test_span_input_silences_outside_and_keeps_inside_beats():
    import numpy as np
    from dancecap import heat_beats as H
    sr = 100
    x = np.ones(1000)
    tr = {"beats_refined": np.array([1.0, 3.0, 5.0, 8.0]), "downbeats_refined": np.array([1.0, 8.0]), "beat_prob": np.ones(500)}
    y, out = H.span_input(x, tr, (2.0, 6.0), sr)
    assert y[:200].sum() == 0 and y[600:].sum() == 0 and y[200:600].sum() == 400
    assert out["beats_refined"].tolist() == [3.0, 5.0]
    assert out["downbeats_refined"].tolist() == []
    assert len(out["beat_prob"]) == 500


def test_shift_moves_every_time_field_and_span():
    from dancecap import heat_beats as H
    seg = {"timeline": [{"from_s": 0.0, "to_s": 2.0, "state": "intro"}, {"from_s": 2.0, "to_s": 9.5, "state": "s1", "songs": ["s1"]}],
           "no_beat": [[0.0, 2.0]], "governing": {"s1": [2.5, 9.0]}, "music_span": [2.0, 9.5], "danced_span": [1.0, 9.0], "version": 1}
    out = H._shift_t(seg, 14.0)
    assert out["timeline"][1] == {"from_s": 16.0, "to_s": 23.5, "state": "s1", "songs": ["s1"]}
    assert out["no_beat"] == [[14.0, 16.0]] and out["governing"] == {"s1": [16.5, 23.0]} and out["music_span"] == [16.0, 23.5]
    assert out["version"] == 1
    assert H._shift_t([{"t": 1.5, "measure_in_take": 3}], 14.0) == [{"t": 15.5, "measure_in_take": 3}]
