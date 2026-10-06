"""Unit tests for dancecap.timing (pure numpy/scipy, no GPU) + a small end-to-end check on work/synthetic.

    .venv-nopo4d\\Scripts\\python.exe -m pytest dancecap/tests/test_timing.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dancecap import timing as T  # noqa: E402
from dancecap.take import PACKAGE_ROOT  # noqa: E402

SYN = PACKAGE_ROOT / "work" / "synthetic"


def _zouk_grid(bpm: float, duration: float, t0: float = 0.0) -> list[list[float]]:
    """Same pattern as atlas_bridge.beats.zouk_grid (8 entries per 4-beat measure)."""
    measure, out, t = 4 * 60.0 / bpm, [], t0
    while t < duration:
        for j in range(8):
            out.append([t + measure * j / 8, 1 if j in (0, 4) else (2 if j in (3, 6) else 3)])
        t += measure
    return [b for b in out if b[0] < duration]


def _grid(bpm=88.0, duration=30.0) -> T.BeatGrid:
    arr = np.array(_zouk_grid(bpm, duration))
    i = np.arange(len(arr))
    return T.BeatGrid(arr[:, 0], arr[:, 1].astype(int), i // 8, i % 8)


def _jittered_times(n=600, fps=30.0, seed=0):
    rng = np.random.default_rng(seed)
    return np.cumsum(np.full(n, 1 / fps) + rng.uniform(-0.004, 0.004, n))


def test_estimate_noise():
    t = np.arange(900) / 30.0
    x = np.stack([0.3 * np.sin(2 * np.pi * 0.8 * t + k) for k in range(6)], 1)
    rng = np.random.default_rng(1)
    assert abs(T.estimate_noise(x + rng.normal(0, 0.01, x.shape)) - 0.01) < 0.0015
    assert T.estimate_noise(x) < 0.0005


def test_spline_derivatives_on_real_timestamps():
    """Quintic spline on jittered timestamps: derivatives match analytic ones; jerk peak is not shifted."""
    t = _jittered_times()
    w = 2 * np.pi * 1.3
    J = np.zeros((len(t), 1, 3))
    J[:, 0, 0] = 0.2 * np.sin(w * t)
    k = T.spline_kinematics(t, J, 1e-5, k=5)  # clean signal: nearly interpolating
    inner = slice(30, -30)
    assert np.allclose(k.vel[inner, 0, 0], 0.2 * w * np.cos(w * t[inner]), atol=0.005)
    assert np.allclose(k.acc[inner, 0, 0], -0.2 * w ** 2 * np.sin(w * t[inner]), atol=0.08)
    jerk_true = -0.2 * w ** 3 * np.cos(w * t[inner])
    assert np.sqrt(np.mean((k.jerk[inner, 0, 0] - jerk_true) ** 2)) < 0.02 * np.abs(jerk_true).max()
    # with 2 mm noise and the matching smoothing the jerk is attenuated but NOT shifted in time
    noisy = J + np.random.default_rng(2).normal(0, 0.002, J.shape)
    k = T.spline_kinematics(t, noisy, 0.002, k=5)
    # phase of the jerk via least squares on cos/sin: spline ~0, forward differences lag ~1.5 frames
    def phase(sig, tt):
        A = np.stack([np.cos(w * tt), np.sin(w * tt)], 1)
        c = np.linalg.lstsq(A, sig, rcond=None)[0]
        return np.arctan2(c[1], -c[0]) / w  # seconds of shift relative to -cos
    assert abs(phase(k.jerk[inner, 0, 0], t[inner])) < 0.004
    fd = J[:, 0, 0]
    tt = t
    for _ in range(3):
        fd = np.diff(fd) / np.diff(tt)
        tt = tt[:-1]
    assert abs(phase(fd[30:-30], tt[30:-30])) > 0.03  # forward differences are early by ~1.5 frames (~50 ms)
    # multi-evaluation from one fit
    te = np.linspace(t[0], t[-1], 2000)
    a, b = T.spline_kinematics(t, J, 0.0005, te=(t, te))
    assert a.pos.shape == (len(t), 1, 3) and b.pos.shape == (2000, 1, 3)


def test_butterworth_cross_check_agrees():
    t = _jittered_times()
    J = np.zeros((len(t), 1, 3))
    J[:, 0, 1] = 0.1 * np.sin(2 * np.pi * 1.0 * t)
    s = T.spline_kinematics(t, J, 0.0005)
    b = T.butter_kinematics(t, J, cutoff_hz=6.0)
    inner = slice(40, -40)
    assert np.sqrt(np.mean((s.vel[inner, 0, 1] - b.vel[inner, 0, 1]) ** 2)) < 0.03  # 5% of the 0.63 m/s peak


def test_hinge_touchdown_impact_and_soft():
    rng = np.random.default_rng(3)
    for tau, shape in ((1.2345, "impact"), (2.0161, "soft")):
        t = np.arange(0.9, 2.4, 1 / 30) + 0.007
        u = np.maximum(0, tau - t)
        y = 0.05 + (0.6 * u + 2.0 * u * u if shape == "impact" else 1.7 * u * u)
        sel = (t > tau - 0.25) & (t < tau + 0.2)
        est, rms = T.hinge_touchdown(t[sel], y[sel], tau - 0.2, tau + 0.15)
        assert abs(est - tau) < 0.002, (shape, est - tau)
        est, _ = T.hinge_touchdown(t[sel], y[sel] + rng.normal(0, 0.001, sel.sum()), tau - 0.2, tau + 0.15)
        assert abs(est - tau) < 0.03


def _walker(t, td_left, td_right, swing=0.5, clear=0.08):
    """Minimal 55-joint body: feet lift in a sin^2 bump ending exactly at each touchdown time."""
    J = np.zeros((len(t), 55, 3))
    J[:, 0] = [0, 0.95, 0]
    J[:, 1] = [0.1, 0.9, 0]
    J[:, 2] = [-0.1, 0.9, 0]
    for side, tds, x in (("left", td_left, 0.1), ("right", td_right, -0.1)):
        h = np.zeros(len(t))
        for td in tds:
            ph = (t - (td - swing)) / swing
            m = (ph > 0) & (ph < 1)
            h[m] = np.maximum(h[m], clear * np.sin(np.pi * ph[m]) ** 2)
        a, toe = T.FOOT_JOINTS[side]["ankle"], T.FOOT_JOINTS[side]["toe"]
        knee = 4 if side == "left" else 5
        J[:, a] = np.stack([np.full(len(t), x), 0.07 + h, np.zeros(len(t))], 1)
        J[:, toe] = np.stack([np.full(len(t), x), 0.01 + 0.8 * h, np.full(len(t), 0.12)], 1)
        J[:, knee] = np.stack([np.full(len(t), x), 0.5 + 0.5 * h, 0.3 * h], 1)
    return J


def test_detect_touchdowns_contact_model():
    t = _jittered_times(n=540)
    beat = 0.6
    td_l = np.arange(1.0, t[-1] - 0.3, 2 * beat) + 0.0123
    td_r = td_l + beat
    td_r = td_r[td_r < t[-1] - 0.3]
    J = _walker(t, td_l, td_r)
    cfg = T.TimingConfig()
    kin, fine = T.spline_kinematics(t, J, 0.0005, T._needed(cfg), te=(t, np.linspace(t[0], t[-1], 4 * len(t))))
    ev = T.detect_touchdowns(t, J, kin, 0.0005, cfg, fine)
    for foot, truth in (("left", td_l), ("right", td_r)):
        det = np.array([e["t"] for e in ev if e["foot"] == foot])
        pairs = T.match(det, truth, 0.1)
        assert len(pairs) == len(truth) == len(det), (foot, det, truth)
        err = np.array([det[i] - truth[j] for i, j in pairs])
        assert np.median(np.abs(err)) < 0.01, err
    assert all(e["method"] == "template" for e in ev)


def test_beat_grid_density_warning_and_coarsen():
    """A 77 BPM track split by beat_this (min_bpm=120) looks like a 154 BPM 8-count grid: warn, coarsen=2 fixes."""
    cfg = T.TimingConfig()
    dense = _grid(154.0, 40.0)
    d = T.grid_diagnostics(dense, cfg)
    assert d["grid_measure_bpm"] > 120 and len(d["warnings"]) == 2
    assert d["chance_hit_rate_analytic"] > 0.5
    fixed = T.coarsen(dense, 2)
    d2 = T.grid_diagnostics(fixed, cfg)
    assert abs(d2["grid_measure_bpm"] - 77.0) < 0.5 and not any("halved" in w for w in d2["warnings"])
    assert list(fixed.type[:8]) == [1, 3, 3, 2, 1, 3, 2, 3]


def test_refine_beats_removes_jitter_keeps_linear():
    g = _grid(88.0, 30.0)
    r, rms = T.refine_beats(g)
    assert rms < 1e-9 and np.allclose(r.t, g.t)
    rng = np.random.default_rng(0)
    noisy = T.BeatGrid(g.t + rng.normal(0, 0.015, len(g.t)), g.type, g.measure, g.pos)
    r, rms = T.refine_beats(noisy)
    assert np.sqrt(np.mean((r.t - g.t) ** 2)) < 0.012 and rms > 0.005


def test_assign_and_circular_stats():
    cfg = T.TimingConfig(allowed_types=(1,))
    g = _grid(88.0, 30.0)
    beats = g.t[g.type == 1]
    ev = beats[2:20] + 0.030
    bi, per = T.assign_beats(ev, g, cfg)
    assert np.allclose(g.t[bi], beats[2:20])
    asyn = ev - g.t[bi]
    c = T.circular(asyn, per)
    assert abs(c["circ_mean_ms"] - 30) < 0.5 and c["R"] > 0.99 and c["rayleigh_p"] < 1e-6
    # positions restrict further: only count 0 of each measure
    cfg2 = T.TimingConfig(allowed_positions=(0,))
    bi2, per2 = T.assign_beats(ev, g, cfg2)
    assert set(g.pos[bi2]) == {0} and np.allclose(per2, 4 * 60 / 88.0)


def test_surrogate_chance_and_floor():
    cfg = T.TimingConfig(n_surrogates=500)
    g = _grid(88.0, 60.0)
    rng = np.random.default_rng(0)
    rand = np.sort(rng.uniform(2, 55, 80))
    bi, per = T.assign_beats(rand, g, cfg)
    obs = {"hit_rate": float(np.mean(np.abs(rand - g.t[bi]) <= cfg.hit_tol_s)),
           "V": float(np.mean(np.cos(2 * np.pi * (rand - g.t[bi]) / per)))}
    s = T.surrogate(rand, g, cfg, rng, obs)
    chance = T.grid_diagnostics(g, cfg)["chance_hit_rate_analytic"]
    assert abs(s["hit_rate_mean"] - chance) < 0.05 and s["hit_rate_p"] > 0.05
    on = g.t[(g.t > 2) & (g.t < 55)][::2]
    bi, per = T.assign_beats(on, g, cfg)
    s2 = T.surrogate(on, g, cfg, rng, {"hit_rate": 1.0, "V": 1.0})
    assert s2["shift_hit_rate_p"] >= s2["p_floor_isochronous"] * 0.8  # isochronous: the shift test cannot beat the floor
    assert abs(s2["p_floor_isochronous"] - 0.1 / (60 / 88 / 2)) < 1e-6
    assert s2["hit_rate_p"] < 0.01 and s2["V_p"] < 0.01  # ... the jitter null rejects chance for a precise dancer


def test_jitter_null_short_window_power_and_size():
    """6 s at 88 BPM, every eighth allowed: 8 perfect on-beat steps reject chance (jitter null, V-test, Rayleigh);
    random event trains are rejected at about the nominal 5 % rate (the test is not anti-conservative)."""
    cfg = T.TimingConfig(n_surrogates=1000)
    g = _grid(88.0, 30.0)
    rng = np.random.default_rng(3)
    on = g.t[(g.t > 10) & (g.t < 16)][::2][:8] + rng.normal(0, 0.012, 8)
    bi, per = T.assign_beats(on, g, cfg)
    a = on - g.t[bi]
    st = T.circular(a, per)
    obs = {"hit_rate": float(np.mean(np.abs(a) <= cfg.hit_tol_s)), "V": st["V"]}
    s = T.surrogate(on, g, cfg, rng, obs)
    assert s["hit_rate_p"] < 0.01 and s["V_p"] < 0.01 and st["v_test_p"] < 0.01 and st["rayleigh_p"] < 0.01
    assert s["shift_hit_rate_p"] > 0.05  # the circular shift has no power here (floor 0.29)
    rej = []
    for k in range(60):
        ev = np.sort(rng.uniform(10, 16, 8))
        bi, per = T.assign_beats(ev, g, cfg)
        a = ev - g.t[bi]
        o = {"hit_rate": float(np.mean(np.abs(a) <= cfg.hit_tol_s)), "V": T.circular(a, per)["V"]}
        rej.append(T.surrogate(ev, g, T.TimingConfig(n_surrogates=400), rng, o)["V_p"] < 0.05)
    assert np.mean(rej) < 0.15


def test_group_contrast_two_sided_control():
    """An EARLY control group is not 'no effect': the two-sided p catches it, and late_ok fails."""
    rng = np.random.default_rng(0)
    cfg = T.TimingConfig(n_perm=3000)

    def evs(shift_in_group):
        out = []
        for m in range(6):
            for k in range(4):
                a = rng.normal(0, 0.01) + (shift_in_group if m in (2, 3) else 0.0)
                out.append({"async_ms": 1000 * a, "measure": m, "_async_mc": a + rng.normal(0, 0.005, 50)})
        return out

    late, early = evs(0.08), evs(-0.08)
    gl = T.group_contrast(late, lambda m: m in (2, 3), cfg, rng)
    ge = T.group_contrast(early, lambda m: m in (2, 3), cfg, rng)
    assert gl["perm_p_greater"] < 0.01 and gl["perm_p_two_sided"] < 0.01
    assert ge["perm_p_greater"] > 0.9 and ge["perm_p_two_sided"] < 0.01
    assert T.late_ok({"follow": {"late_measures_vs_rest": gl}, "lead": {"late_measures_vs_rest": ge}}) is False
    null = T.group_contrast(evs(0.0), lambda m: m in (2, 3), cfg, rng)
    expect = null["perm_p_two_sided"] >= 0.05 and null["diff_ci95_ms"][0] <= 0 <= null["diff_ci95_ms"][1]
    assert T.late_ok({"follow": {"late_measures_vs_rest": gl}, "lead": {"late_measures_vs_rest": null}}) is expect


def test_measure_effects_flags_late_measure():
    cfg = T.TimingConfig(n_perm=4000)
    rng = np.random.default_rng(5)
    measure = np.repeat(np.arange(9), 4)
    asyn = rng.normal(0, 0.012, len(measure))
    asyn[measure == 3] += 0.09
    me = T.measure_effects(asyn, measure, cfg, rng)
    flagged = [m["measure"] for m in me["measures"] if m["q_bh"] < 0.05]
    assert flagged == [3] and me["global_p"] < 0.01


def test_match_one_to_one():
    a = np.array([1.0, 1.05, 3.0])
    b = np.array([1.02, 2.9, 5.0])
    assert sorted(T.match(a, b, 0.11)) == [(0, 0), (2, 1)]
    assert T.match(np.zeros(0), b, 0.1) == []


def test_analyze_end_to_end_synthetic_walker():
    """analyze() on two walkers: lead on the beat, follow 80 ms late in measure 2 -> stats + late flag."""
    bpm = 88.0
    q = 60 / bpm
    t = np.arange(int(20 * 30)) / 30.0
    tds = np.arange(1, 28) * q
    tds = tds[tds < t[-1] - 0.3]
    meas = np.floor(tds / (4 * q) + 1e-9).astype(int)
    late = np.where(meas == 2, 0.08, 0.0)
    joints, ts = {}, {}
    for role, shift in (("lead", 0 * late), ("follow", late)):
        td = tds + shift
        joints[role] = _walker(t, td[0::2], td[1::2], swing=0.45)
        ts[role] = t
    beats = np.array(_zouk_grid(bpm, 21.0))
    i = np.arange(len(beats))
    grid = T.BeatGrid(beats[:, 0], beats[:, 1].astype(int), i // 8, i % 8)
    cfg = T.TimingConfig(mc_draws=8, workers=1, n_surrogates=300, n_perm=2000)
    res = T.analyze(ts, joints, grid, cfg)
    lead, follow = res["asynchrony"]["lead"]["footsteps"], res["asynchrony"]["follow"]["footsteps"]
    assert lead["n"] >= len(tds) - 2 and abs(lead["mean_ms"]) < 10
    pm = {m["measure"]: m for m in follow["per_measure"]}
    assert abs(pm[2]["mean_ms"] - 80) < 15 and pm[2]["ci95_ms"][0] > 30
    assert 2 in [m["measure"] for m in follow["measure_effects"]["measures"] if m["q_bh"] < 0.05]
    rel = {m["measure"]: m for m in res["asynchrony"]["follow_minus_lead"]["per_measure"]}
    assert rel[2]["ci95_ms"][0] > 30
    for e in res["footsteps"]["follow"]:
        assert {"t", "foot", "sigma", "beat_t", "async_ms", "measure", "count"} <= set(e)
    assert res["absolute_timing"]["valid"] is False
    json.dumps(T._clean(res))  # json-able


def test_validation_report_contract():
    """When the synthetic validation has been run, check timing.json / validation.json against the contract."""
    p = SYN / "timing" / "timing.json"
    if not p.exists():
        pytest.skip("run: python -m dancecap timing synthetic --validate")
    d = json.loads(p.read_text())
    assert {"beats", "footsteps", "accents", "asynchrony", "absolute_timing", "notes", "kinematics"} <= set(d)
    for role in T.ROLES:
        for e in d["footsteps"][role]:
            assert e["sigma"] is None or e["sigma"] > 0
        assert "surrogate" in d["asynchrony"][role]["footsteps"]
    k = np.load(SYN / "timing" / "kinematics.npz")
    for role in T.ROLES:
        assert k[f"{role}_jerk"].shape == k[f"{role}_pos"].shape and k[f"{role}_jerk_reliable"].shape == (55,)
        assert float(k[f"{role}_jerk_cutoff_hz"]) > 0 and k[f"{role}_jerk_noise_sd"].shape == k[f"{role}_t"].shape
    v = SYN / "timing" / "validation.json"
    if v.exists():
        g = json.loads(v.read_text())["gate"]
        for role in T.ROLES:
            assert g["clean"]["per_dancer"][role]["median_abs_err_ms"] <= 20


def _noisy_sine_joints(noise, seed=0, n=180):
    t = np.arange(n) / 30.0
    J = np.zeros((n, 55, 3))
    for j in range(55):
        J[:, j, 0] = 0.15 * np.sin(2 * np.pi * 1.2 * t + j)
        J[:, j, 1] = 0.5 + 0.05 * np.sin(2 * np.pi * 2.3 * t + 2 * j)
    return t, J + np.random.default_rng(seed).normal(0, noise, J.shape)


def test_derivative_cutoff_follows_noise():
    """Per joint: clean -> reliable acc / jerk; 1 cm noise -> lower cutoffs; a near-still joint gets the lowest
    cutoff, is flagged unreliable and has no spurious jerk peaks; a moving joint's jerk error stays bounded."""
    cfg = T.TimingConfig()
    idx = list(cfg.accent_joints.values())
    t, J = _noisy_sine_joints(0.0)
    J[:, idx[1]] = J[0, idx[1]]  # a still joint
    c0 = T.choose_cutoffs(t, J, idx, 0.0005, cfg)
    assert c0["acc"]["reliable"][0] and c0["jerk"]["reliable"][0] and not c0["jerk"]["reliable"][1]
    t, Jn = _noisy_sine_joints(0.01)
    Jn[:, idx[1]] = J[:, idx[1]] + np.random.default_rng(5).normal(0, 0.01, (len(t), 3))
    c1 = T.choose_cutoffs(t, Jn, idx, 0.01, cfg)
    assert c1["jerk"]["cutoff_hz"][0] < c0["jerk"]["cutoff_hz"][0] and c1["acc"]["cutoff_hz"][0] < c0["acc"]["cutoff_hz"][0]
    assert c1["jerk"]["cutoff_hz"][1] == min(c1["jerk"]["cutoff_hz"]) and not c1["jerk"]["reliable"][1]
    k_new = T.spline_kinematics(t, Jn, 0.01, idx)
    T.apply_cutoffs(t, Jn, c1, (k_new,))
    sl = slice(20, -20)
    still_new = np.linalg.norm(k_new.jerk[sl, idx[1]], axis=-1).max()
    assert still_new < 10.0, still_new  # true jerk 0; noise SD at the lowest cutoff is a few m/s^3
    true_jerk = -0.15 * (2 * np.pi * 1.2) ** 3 * np.cos(2 * np.pi * 1.2 * t[sl] + idx[0])
    e_new = np.sqrt(np.mean((k_new.jerk[sl, idx[0], 0] - true_jerk) ** 2))
    assert e_new < 0.5 * np.abs(true_jerk).max(), e_new


def test_resolve_sync_keeps_explicit_and_never_mutates(tmp_path):
    (tmp_path / "sync").mkdir()
    (tmp_path / "sync" / "offsets.json").write_text(json.dumps({"offsets": {"01": {"delta_s": 0.0, "sd_s": 0.003},
                                                                            "02": {"delta_s": 0.01, "sd_s": 0.004}}}))
    cfg = T.TimingConfig()
    r = T.resolve_sync(T.TimingConfig(**dict(cfg.__dict__)), tmp_path)
    assert r.n_cams == 2 and abs(r.sync_sigma_s - np.sqrt((0.003 ** 2 + 0.004 ** 2) / 2)) < 1e-9
    assert cfg.n_cams is None  # the original is untouched
    r2 = T.resolve_sync(T.TimingConfig(n_cams=8, sync_sigma_s=0.005), tmp_path)
    assert r2.n_cams == 8 and r2.sync_sigma_s == 0.005 and r2.sync_source == "config"


def test_find_beats_never_uses_audio_clock_file(tmp_path):
    (tmp_path / "zouk-time-analysis.json").write_text("[[0.5, 1]]")
    with pytest.raises(FileNotFoundError):
        T.find_beats(tmp_path)


@pytest.mark.skipif(not (SYN / "smplx" / "lead.npz").exists(), reason="run: python -m dancecap synthetic")
def test_short_noisy_windows_per_dancer():
    """6 s windows (like the real take) with 1 cm joint noise, several seeds: per dancer median |error| and recall,
    and a small MC whose detect_rate stays near 1 (draws centred on the raw data, shift-tolerant matching)."""
    from dancecap import synthetic as S
    meds, recs = {r: [] for r in T.ROLES}, {r: [] for r in T.ROLES}
    for s0, seed in ((3.0, 0), (9.0, 1), (15.0, 2)):
        w = S.window(SYN, s0, s0 + 6.0, joint_noise_m=0.01, seed=seed)
        cfg = T.TimingConfig(mc_draws=10 if seed == 0 else 0, workers=1, n_surrogates=200, n_perm=500)
        res = T.analyze({r: w["npz"][r]["t"] for r in T.ROLES}, {r: w["npz"][r]["joints"] for r in T.ROLES},
                        T._grid_from(w["beats"]), cfg)
        sc = T.score_touchdowns(res, w["truth"], span=(s0, s0 + 6.0))
        for r in T.ROLES:
            meds[r].append(sc[r]["median_abs_err_ms"])
            recs[r].append(sc[r]["recall"])
            if seed == 0:
                rate = np.median([e["detect_rate"] for e in res["footsteps"][r]])
                assert rate >= 0.8, (r, rate)
                assert all(np.isfinite(e["sigma"]) and 0 < e["sigma"] < 0.08 for e in res["footsteps"][r])
    for r in T.ROLES:
        assert np.median(meds[r]) <= 25 and max(meds[r]) <= 40, (r, meds[r])
        assert min(recs[r]) >= 0.8, (r, recs[r])
