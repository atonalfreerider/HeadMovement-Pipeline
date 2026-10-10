"""Unit tests for dancecap.carlos_aline_beats (synthetic beats and steps: no audio, no GPU, no pod, no footage).

    .venv-nopo4d\\Scripts\\python.exe -m pytest dancecap/tests/test_carlos_aline_beats.py -q      (from the workspace root)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dancecap import carlos_aline_beats as cb  # noqa: E402
from dancecap.take import PACKAGE_ROOT  # noqa: E402

KOJO = cb.Clock(0.9615305360969022, -12.800733843537316)
GUETTA = cb.Clock(0.9425378343464413, -106.94604487997859)


def mixed_level_beats(period: float = 0.714, t0: float = 0.28, n: int = 160, jitter_ms: float = 8.0, seed: int = 2) -> np.ndarray:
    """What beat_this does on Kojo Funds: runs at the pulse level alternate with runs at twice that rate (an extra beat
    half-way between pulses), frame-quantised-like jitter on every beat."""
    rng = np.random.default_rng(seed)
    out = []
    k = 0
    while k < n:
        run = int(rng.integers(3, 14))
        dense = (len(out) // 7) % 2 == 1
        for j in range(run):
            out.append(t0 + (k + j) * period)
            if dense:
                out.append(t0 + (k + j + 0.5) * period)
        k += run
    b = np.sort(np.asarray(out)) + rng.normal(0, jitter_ms / 1000.0, len(out))
    return np.sort(b)


# ------------------------------------------------------------------------------------------------ clocks
def test_clock_direction_and_round_trip():
    assert KOJO.to_ref(0.0) == pytest.approx(13.3129, abs=1e-3)  # song position 0 plays at reference 13.31 s
    assert GUETTA.to_ref(0.0) == pytest.approx(113.466, abs=1e-3)
    t = np.array([0.0, 10.0, 104.8])
    assert np.allclose(KOJO.to_song(KOJO.to_ref(t)), t)
    # slope < 1: a reference second holds less than a song second -> the heat plays SLOWER than the studio file
    assert KOJO.tempo_ratio < 1 < KOJO.duration_stretch
    assert KOJO.duration_stretch == pytest.approx(1.040, abs=1e-3) and GUETTA.duration_stretch == pytest.approx(1.061, abs=1e-3)
    assert 60.0 / 0.7137 * KOJO.tempo_ratio == pytest.approx(80.83, abs=0.05)


# ------------------------------------------------------------------------------------------------ pulse model
def test_pulse_period_folds_to_the_dance_level():
    P, info = cb.pulse_period(mixed_level_beats())
    assert P == pytest.approx(0.714, abs=0.01)
    assert 60 <= 60.0 / P < 120 and info["factor"] in (1.0, 2.0)
    P2, _ = cb.pulse_period(np.arange(0, 60, 0.357))  # all beats at twice the pulse rate (168 BPM)
    assert P2 == pytest.approx(0.714, abs=0.01)
    P3, _ = cb.pulse_period(np.arange(0, 60, 1.43))  # half time (42 BPM): doubled up into [60, 120)
    assert 60 <= 60.0 / P3 < 120
    with pytest.raises(ValueError):
        cb.pulse_period(np.array([0.0, 0.7]))


def test_fit_pulses_recovers_the_line_from_mixed_level_beats():
    m = cb.fit_pulses(mixed_level_beats())
    assert m["period"] == pytest.approx(0.714, abs=0.0015)
    frac = ((m["t0"] - 0.28) / 0.714) % 1.0  # the lattice origin is a whole number of pulses from the true one
    assert min(frac, 1.0 - frac) < 0.01
    assert m["constant"] and m["rms_ms"] < 12
    # the half-integer (eighth) beats are not on the lattice
    assert m["n_support"] < m["n_fit"]
    k = np.arange(0, 100)
    assert np.allclose(np.diff(cb.pulse_times(m, k)), m["period"])


def test_fit_pulses_flags_a_tempo_change_and_follows_it():
    k = np.arange(0, 150)
    t_true = 0.3 + 0.7 * k + 0.0008 * k ** 2  # accelerating period: 0.7 -> 0.94 s
    m = cb.fit_pulses(t_true + np.random.default_rng(0).normal(0, 0.004, len(k)))
    assert not m["constant"]
    truth = t_true
    off = int(np.argmin(np.abs(truth - m["t"][0]))) - int(m["k"][0])  # the lattice index starts at the seed run, not at beat 0
    pt = cb.pulse_times(m, np.arange(20, 120) - off)
    assert np.abs(pt - truth[20:120]).max() < 0.03


def test_fit_pulses_span_ignores_a_tempo_drift_outside():
    t = np.concatenate([np.arange(0, 80, 0.7), 80 + np.cumsum(np.linspace(0.7, 0.5, 40))])
    m = cb.fit_pulses(t, span=(-1, 78))
    assert m["period"] == pytest.approx(0.7, abs=0.002) and m["constant"]


def test_downbeat_votes_and_ambiguity():
    model = {"period": 0.7, "t0": 0.0}
    d4 = 0.7 * np.arange(1, 200, 4)  # bar start on k = 1 mod 4
    v = cb.downbeat_votes(d4, model)
    assert v["phase"] == 1 and not v["ambiguous"] and v["agreement"] == 1.0
    # two-pulse bars (beat_this on Kojo Funds): k = 1, 3 mod 4 both vote -> ambiguous
    v2 = cb.downbeat_votes(0.7 * np.arange(1, 200, 2), model)
    assert v2["ambiguous"] and v2["half_bar_ratio"] > 0.5
    # a downbeat far from any pulse is not a vote
    assert cb.downbeat_votes(np.array([0.35]), model)["n"] == 0


def test_zouk_rows_whole_measures_and_types():
    model = {"period": 0.5, "t0": 10.0, "constant": True}
    rows = cb.zouk_rows(model, 3, 22, phase=2)  # first bar start: k = 6; whole measures need k0 + 4 <= 22
    assert rows.shape[1] == 5 and len(rows) % 8 == 0
    assert [int(x) for x in rows[:8, 1]] == [1, 3, 3, 2, 1, 3, 2, 3] and [int(x) for x in rows[:8, 3]] == list(range(8))
    assert rows[0, 0] == pytest.approx(10.0 + 6 * 0.5) and rows[0, 4] == 6.0
    assert np.allclose(np.diff(rows[:8, 0]), 0.25)  # eighths are pulse midpoints
    assert rows[:, 2].max() == len(rows) // 8 - 1 and rows[-1, 4] <= 21.5
    assert len(cb.zouk_rows(model, 0, 3, phase=0)) == 0  # not even one whole measure


def test_whole_measures_filter_and_renumber():
    model = {"period": 1.0, "t0": 0.0, "constant": True}
    rows = cb.zouk_rows(model, 0, 40, 0)
    out = cb.whole_measures(rows, 10.0, 30.0)
    assert out[0, 0] >= 10.0 and out[-1, 0] <= 30.0 and out[0, 2] == 0 and len(out) % 8 == 0
    assert (np.diff(out[:, 2]) >= 0).all() and cb.whole_measures(rows[:0], 0, 1).shape[0] == 0


# ------------------------------------------------------------------------------------------------ the mix
def test_timeline_two_songs_with_a_mix():
    seg = cb.timeline((13.31, 122.25), (113.47, 194.25), 202.08)
    assert [s["state"] for s in seg] == ["none", "kojo", "mix", "guetta", "none"]
    assert seg[0]["to_s"] == 13.31 and seg[2]["from_s"] == 113.47 and seg[2]["to_s"] == 122.25 and seg[-1]["to_s"] == 202.08
    assert cb.state_at(seg, 5.0) == "none" and cb.state_at(seg, 118.0) == "mix" and cb.state_at(seg, 150.0) == "guetta"
    assert cb.state_at(seg, 200.0) == "none" and cb.state_at(seg, 1e9) == "none"
    assert [s["state"] for s in cb.timeline((10, 50), (60, 90), 100)] == ["none", "kojo", "none", "guetta", "none"]


def test_audio_switch():
    t = np.arange(100.0, 130.0, 1.0)
    z_k = np.where(t < 118, 7.0, 1.0)
    z_g = np.where(t < 116, 0.5, np.where(t < 118, 3.0, 8.0))
    sw = cb.audio_switch(t, z_k, z_g, (113.47, 122.25))
    assert 116.0 <= sw["switch_s"] <= 118.5  # Guetta first leads from 118
    assert cb.audio_switch(t, np.full(30, 7.0), np.full(30, 1.0), (113.47, 122.25))["switch_s"] == 122.25  # never leads
    assert cb.audio_switch(t, np.full(30, 1.0), np.full(30, 7.0), (113.47, 122.25))["switch_s"] == pytest.approx(114.0)
    assert cb.audio_switch(np.array([]), np.array([]), np.array([]), (113.0, 122.0))["switch_s"] == pytest.approx(117.5)


def _grid(t0: float, period: float, k0: int, k1: int, phase: int = 0) -> np.ndarray:
    g = cb.zouk_rows({"period": period, "t0": t0, "constant": True}, k0, k1, phase)
    return g


def test_govern_is_time_ordered_whole_measures():
    a = _grid(0.0, 0.74, 0, 160)  # song A: bars of 2.96 s from 0
    b = _grid(100.3, 0.67, 0, 100)  # song B starts at 100.3
    ga, gb = cb.govern(a, b, switch=110.0)
    both = np.vstack([ga, gb])
    assert (np.diff(both[:, 0]) > 0).all()  # strictly time ordered: usable as one bare beat list
    assert len(ga) % 8 == 0 and len(gb) % 8 == 0
    assert ga[:, 0].max() < gb[:, 0].min()
    centres = [ga[ga[:, 2] == m][:, 0].mean() for m in np.unique(ga[:, 2])]
    assert max(centres) < 110.0
    assert gb[0, 3] == 0 and gb[0, 0] >= 110.0 - 1.5  # B starts on a downbeat, near the switch
    # a switch before B exists / after A ends keeps both lists sane
    ga2, gb2 = cb.govern(a, b, switch=0.0)
    assert len(ga2) == 0 and len(gb2) > 0
    ga3, gb3 = cb.govern(a, b, switch=1e6)
    assert len(ga3) == len(a) and len(gb3) == 0


# ------------------------------------------------------------------------------------------------ the dancers' steps (hook)
def _steps_on(pulses: np.ndarray, lo: float, hi: float, jitter: float = 0.02, keep: float = 0.8, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = pulses[(pulses >= lo) & (pulses <= hi)]
    p = p[rng.random(len(p)) < keep]
    return p + rng.normal(0, jitter, len(p))


def test_phase_lock_and_follow_choice():
    pk = 0.742 * np.arange(0, 250)  # Kojo pulses
    pg = 5.0 + 0.670 * np.arange(0, 250)  # Guetta pulses (another tempo)
    ov = (113.47, 122.25)
    on_k = _steps_on(pk, 90, 140)
    on_g = _steps_on(pg, 90, 140)
    lk, lg = cb.phase_lock(on_k, pk, *ov), cb.phase_lock(on_k, pg, *ov)
    assert lk["R"] > 0.8 > lg["R"] + 0.2 and lk["n"] >= 6
    assert cb.follow_choice(on_k, {"kojo": pk, "guetta": pg}, ov)["follows"] == "kojo"
    assert cb.follow_choice(on_g, {"kojo": pk, "guetta": pg}, ov)["follows"] == "guetta"
    # dancers on neither grid (random times): undecided
    rnd = np.sort(np.random.default_rng(3).uniform(113.5, 122.2, 14))
    assert cb.follow_choice(rnd, {"kojo": pk, "guetta": pg}, ov)["follows"] == "undecided"
    # too few steps in the mix
    assert cb.follow_choice(on_k[:2], {"kojo": pk, "guetta": pg}, ov)["follows"] == "undecided"
    assert cb.phase_lock(np.array([1.0]), pk, 0, 5)["R"] is None


def test_follow_curve_switches_with_the_dancers():
    pk = 0.742 * np.arange(0, 250)
    pg = 5.0 + 0.670 * np.arange(0, 250)
    steps = np.sort(np.concatenate([_steps_on(pk, 20, 118), _steps_on(pg, 118, 190, seed=1)]))
    cur = cb.follow_curve(steps, {"kojo": pk, "guetta": pg}, 20.0, 190.0)
    early = [r for r in cur if r["t1"] < 110]
    late = [r for r in cur if r["t0"] > 125]

    def mean_r(rows, s):
        return float(np.mean([r[s]["R"] for r in rows if r[s] and r[s]["R"] is not None]))
    assert mean_r(early, "kojo") > mean_r(early, "guetta") + 0.3
    assert mean_r(late, "guetta") > mean_r(late, "kojo") + 0.3


def test_load_steps_formats(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps([3.0, 1.0, 2.0]))
    (tmp_path / "b.json").write_text(json.dumps({"steps": [2.0, 1.0]}))
    (tmp_path / "c.json").write_text(json.dumps({"footsteps": {"lead": [{"t": 5.0, "foot": "L"}, {"t": 1.0}],
                                                              "follow": [{"t": 3.0}], "note": "x"}}))
    (tmp_path / "d.json").write_text(json.dumps([{"t": 2.0}, {"t": 1.0}]))
    assert cb.load_steps(tmp_path / "a.json").tolist() == [1.0, 2.0, 3.0]
    assert cb.load_steps(tmp_path / "b.json").tolist() == [1.0, 2.0]
    assert cb.load_steps(tmp_path / "c.json").tolist() == [1.0, 3.0, 5.0]
    assert cb.load_steps(tmp_path / "d.json").tolist() == [1.0, 2.0]


# ------------------------------------------------------------------------------------------------ onset evidence
def _flux(period: float, n: int, dt: float = 0.005, lag: float = 0.0, seed: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Onset flux with an asymmetric pattern per pulse (a strong peak at +lag and a weaker one 0.3 P earlier)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) * dt
    x = rng.random(n) * 0.2
    pulses = np.arange(1.0, t[-1] - 1.0, period)
    for p in pulses:
        for off, a in ((lag, 1.0), (lag - 0.3 * period, 0.45)):
            i = int(round((p + off) / dt))
            if 0 <= i < n - 2:
                x[i:i + 2] += a
    return x, pulses


def test_profile_shift_recovers_a_known_delay():
    dt, P = 0.005, 0.7
    studio, pulses = _flux(P, 8000)
    for delay in (-0.031, 0.0, 0.012, 0.045):
        heat, _ = _flux(P, 8000, lag=delay)  # the heat's onsets come `delay` later
        lags = np.arange(-0.3, 0.3001, 0.0025)
        sp = cb.fold_profile(studio, dt, 0.0, pulses, lags)
        hp = cb.fold_profile(heat, dt, 0.0, pulses, lags)
        sh, ncc = cb.profile_shift(hp, sp, 0.0025)
        assert sh == pytest.approx(delay, abs=0.004) and ncc > 0.6
    # fewer than 6 pulses inside the signal: no profile
    assert cb.fold_profile(studio, dt, 0.0, pulses[:3] + 1000.0, lags) is None


def test_window_flux_shift():
    rng = np.random.default_rng(0)
    a = rng.random(3000) ** 3
    b = np.roll(a, 7)  # b(l) = a(l - 7 frames): a ~ b(l + 7) -> shift = -7 frames
    sh, r = cb.window_flux_shift(a, b, 0.005)
    assert sh == pytest.approx(-0.035, abs=0.0021) and r > 0.9


def test_profile_peak_and_drift_fit_and_nearest():
    lags = np.arange(-0.3, 0.3001, 0.0025)
    prof = np.exp(-0.5 * ((lags + 0.0153) / 0.01) ** 2)
    assert cb.profile_peak(prof, lags, 0.05) == pytest.approx(-0.0153, abs=0.002)
    t = np.arange(10, 100, 8.0)
    f = cb.drift_fit(t, -4.0 + 0.05 * (t - 10) + np.random.default_rng(1).normal(0, 0.3, len(t)))
    assert f["drift_ms_per_min"] == pytest.approx(3.0, abs=0.7) and f["n"] == len(t)
    # an outlier window does not move the median / drift much
    y = np.full(len(t), -5.0)
    y[3] = 40.0
    f2 = cb.drift_fit(t, y)
    assert f2["median_ms"] == pytest.approx(-5.0, abs=0.1) and abs(f2["drift_ms_per_min"]) < 1.0
    d = cb.nearest_offsets_signed(np.array([1.02, 1.97]), np.array([1.0, 2.0, 3.0]))
    assert d == pytest.approx([-0.02, 0.03])
    assert cb.nearest_offsets(np.array([1.02, 1.6]), np.array([1.0, 2.0]), 0.1).tolist() == pytest.approx([-0.02])


def test_confidence_and_audibility():
    z = np.array([0.0, 1.5, 3.75, 6.0, 9.0])
    assert cb.audibility(z).tolist() == pytest.approx([0.0, 0.0, 0.5, 1.0, 1.0])
    c = cb.beat_confidence(np.array([1.0] * 5), z, np.zeros(5, bool))
    assert c.tolist() == pytest.approx([0.0, 0.0, 0.5, 1.0, 1.0])
    ci = cb.beat_confidence(np.array([1.0, 0.5]), np.array([9.0, 9.0]), np.array([True, False]))
    assert ci.tolist() == pytest.approx([0.7, 0.5])


def test_tracker_support_and_heat_support():
    prob = np.zeros(200)
    prob[50] = 0.9
    assert cb.tracker_support(prob, np.array([1.0, 1.04, 1.2, -1.0, 99.0])).tolist() == pytest.approx([0.9, 0.9, 0.0, 0.0, 0.0])
    dt = 0.005
    G = np.random.default_rng(0).random(2000) * 0.1
    G[1000:1004] += 1.0  # an onset at t = 5.0 s
    s = cb.heat_support(G, dt, 0.0, np.array([5.01, 2.5]), -0.0, half_s=0.015)
    assert s[0] > 0.9 and s[1] < 0.7
    s2 = cb.heat_support(G, dt, 0.0, np.array([5.03]), -0.02, half_s=0.015)  # a bias correction puts the beat on the onset
    assert s2[0] > 0.9


# ------------------------------------------------------------------------------------------------ take config and written outputs
def test_take_toml_has_both_songs_and_the_clocks_are_slower_than_the_studio():
    p = PACKAGE_ROOT / "takes" / "carlos_aline.toml"
    if not p.exists():
        pytest.skip("takes/carlos_aline.toml is absent")
    import tomllib
    d = tomllib.loads(p.read_text(encoding="utf-8"))["dance"]
    assert d["song_file"].endswith(".mp3") and d["song_file_2"].endswith(".mp3") and d["artist_2"] and d["song_2"]
    al = PACKAGE_ROOT.parent / "prototype" / "larissa_kadu" / "alignment_carlos_aline.json"
    if not al.exists():
        pytest.skip("alignment_carlos_aline.json is absent")
    songs = json.loads(al.read_text(encoding="utf-8"))["songs"]
    for s in cb.SONGS:
        assert songs[s]["slope"] < 1.0 and songs[s]["audible_reference_s"][0] < songs[s]["audible_reference_s"][1]


def test_written_governing_grid_is_a_drop_in_beat_list():
    p = PACKAGE_ROOT / "work" / "carlos_aline" / "song" / "beats_studio.json"
    if not p.exists():
        pytest.skip("work/carlos_aline/song/beats_studio.json has not been written (python -m dancecap.carlos_aline_beats run)")
    arr = np.asarray(json.loads(p.read_text()), float)
    assert arr.shape[1] == 2 and len(arr) % 8 == 0
    assert (np.diff(arr[:, 0]) > 0).all()  # time ordered
    typ = arr[:, 1].astype(int).reshape(-1, 8)
    assert (typ == np.array([1, 3, 3, 2, 1, 3, 2, 3])).all()
    seg = json.loads((p.parent / "segments.json").read_text())
    none = [(a, b) for a, b in seg["no_beat"]]
    assert not any(a <= t < b for t in arr[:, 0] for a, b in none)  # nothing in the announcement or the applause
    assert arr[0, 0] >= seg["timeline"][1]["from_s"]
