"""Unit tests for dancecap.song: synthetic alignment (known slope / offset + noise + room reverb), robust fit, pitch
check, grid comparison math, non-ASCII path handling, soundtrack placement / stretch timing, the copyright guard, and
a contract check on work/larissa_kadu/song (skipped when that take has not been processed).

    .venv-nopo4d\\Scripts\\python.exe -m pytest dancecap/tests/test_song.py -q
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dancecap import beats as B  # noqa: E402
from dancecap import song as S  # noqa: E402
from dancecap.take import PACKAGE_ROOT, WORKSPACE  # noqa: E402

REAL = PACKAGE_ROOT / "work" / "larissa_kadu" / "song"
SR = S.ALIGN_SR
HAVE_FFMPEG = shutil.which("ffmpeg") is not None
NAME = "Ficar Sem Você"  # NFC 'ê'


# ------------------------------------------------------------------------------------------------ synthetic music
def bank(sr: int) -> dict:
    rng = np.random.default_rng(7)
    t = np.arange(int(0.2 * sr)) / sr
    kick = np.sin(2 * np.pi * (50 + 70 * np.exp(-t / 0.03)) * t) * np.exp(-t / 0.07)
    n = int(0.12 * sr)
    snare = rng.normal(0, 1, n) * np.exp(-np.arange(n) / (0.025 * sr)) + 0.5 * np.sin(2 * np.pi * 190 * np.arange(n) / sr)
    n = int(0.04 * sr)
    hat = np.diff(rng.normal(0, 1, n) * np.exp(-np.arange(n) / (0.006 * sr)), prepend=0.0)
    return {"kick": 0.9 * kick, "snare": 0.35 * snare, "hat": 0.15 * hat}


def pluck(f0: float, sr: int, dur: float = 0.35) -> np.ndarray:
    t = np.arange(int(dur * sr)) / sr
    return sum(np.sin(2 * np.pi * f0 * h * t) / h for h in (1, 2, 3, 4)) * np.exp(-t / 0.12) * 0.25


def song_events(dur: float, bpm: float = 75.0, seed: int = 0) -> list[tuple[float, str, float]]:
    """A non-periodic 'song': a drum pattern that changes per measure, fills at random times and plucked notes
    (C major scale) at random eighths, so a long cross-correlation has one clear peak."""
    rng = np.random.default_rng(seed)
    e8 = 30.0 / bpm
    ev = []
    m = 0.3
    scale = 261.63 * 2 ** (np.array([0, 2, 4, 5, 7, 9, 11, 12]) / 12)
    while m < dur:
        kicks = [(0, 3, 5), (0, 3, 6), (0, 4), (0, 2, 3, 6)][rng.integers(4)]
        for j in range(8):
            t = m + j * e8
            ev.append((t, "hat", 0.6 + 0.4 * rng.random()))
            if j in kicks:
                ev.append((t, "kick", 1.0))
            if j in (2, 6):
                ev.append((t, "snare", 0.8 + 0.2 * rng.random()))
            if rng.random() < 0.35:
                ev.append((t, f"note{rng.integers(len(scale))}", 0.7 + 0.3 * rng.random()))
        m += 8 * e8
    for t in rng.uniform(0, dur, int(dur * 0.8)):  # off-grid fills
        ev.append((float(t), "snare" if rng.random() < 0.5 else "hat", 0.5))
    return ev


def room_ir(sr: int, drr_db: float = 0.0, rt60: float = 0.4, seed: int = 0) -> np.ndarray:
    """Direct sound + exponentially decaying diffuse tail from 4 ms, tail energy = direct energy / 10^(drr_db/10)."""
    rng = np.random.default_rng(seed)
    n, k = int(rt60 * sr), int(0.004 * sr)
    tail = rng.normal(0, 1, n - k) * np.exp(-np.arange(n - k) / (rt60 / 6.9 * sr))
    tail *= np.sqrt(10 ** (-drr_db / 10) / np.sum(tail ** 2))
    ir = np.zeros(n)
    ir[0] = 1.0
    ir[k:] += tail
    return ir


def render(events, dur: float, sr: int, slope: float = 1.0, offset: float = 0.0, snr_db: float | None = None,
           drr_db: float | None = None, varispeed: bool = False, seed: int = 1) -> np.ndarray:
    """Render song-time events at reference time slope * t + offset (key-locked: sounds keep their pitch and length);
    varispeed=True scales pitch by 1 / slope as a resampled playback would. Optional room reverb (direct-to-reverberant
    ratio drr_db, RT60 0.4 s) and white noise at snr_db."""
    b = bank(sr)
    scale = 261.63 * 2 ** (np.array([0, 2, 4, 5, 7, 9, 11, 12]) / 12)
    x = np.zeros(int(dur * sr))
    for t, kind, g in events:
        s = int(round((slope * t + offset) * sr))
        if kind.startswith("note"):
            w = pluck(scale[int(kind[4:])] / (slope if varispeed else 1.0), sr)
        else:
            w = b[kind]
        if s < 0 or s >= len(x):
            continue
        e = min(len(x), s + len(w))
        x[s:e] += g * w[:e - s]
    if drr_db is not None:
        from scipy.signal import fftconvolve
        x = fftconvolve(x, room_ir(sr, drr_db, seed=seed))[:len(x)]
    if snr_db is not None:
        x = x + np.random.default_rng(seed).normal(0, np.sqrt(np.mean(x ** 2) / 10 ** (snr_db / 10)), len(x))
    return (x / (np.abs(x).max() + 1e-9) * 0.8).astype(np.float32)


# ------------------------------------------------------------------------------------------------ alignment
@pytest.mark.parametrize("slope,offset,drr_db,snr_db", [(0.98164, -6.9613, 0.0, 15.0), (1.00015, 2.3170, -5.0, 10.0),
                                                     (1.0, -0.5432, None, 20.0)])
def test_align_recovers_slope_and_offset(slope, offset, drr_db, snr_db):
    """Known slope/offset (a DJ tempo change, a phone-clock-sized slope, none) with room reverb and noise on the
    'phone' side: recovered within 1e-4 and 2 ms (the short-window features are what keeps the offset unbiased)."""
    ev = song_events(75.0, seed=3)
    xs = render(ev, 75.0, SR)
    xr = render(ev, 62.0, SR, slope, offset, snr_db=snr_db, drr_db=drr_db, seed=11)
    res = S.align_signals(xs, xr, SR)
    assert abs(res["slope"] - slope) < 1e-4
    assert abs(res["offset"] - offset) < 0.002
    u = np.linspace(*res["coverage"]["song"], 100)
    err = (res["slope"] * u + res["offset"]) - (slope * u + offset)
    assert np.abs(err).max() < 0.002, f"max mapping error {1000 * np.abs(err).max():.2f} ms"
    assert res["n_inliers"] >= 5 and res["residual_ms_p95"] < 3.0
    assert abs(res["coarse"]["slope"] - slope) < 2e-3  # the coarse whole-track search alone is close
    if abs(slope - 1) > 0.01:  # a DJ tempo change: a slope-1 correlation is much weaker than the right slope
        assert res["coarse"]["r"] > 3 * res["coarse"]["r_at_slope_1"]


def test_robust_affine_rejects_outliers():
    rng = np.random.default_rng(0)
    x = np.linspace(10, 180, 22)
    y = 0.98 * x - 7.0 + rng.normal(0, 0.0008, len(x))
    y[[3, 11, 17]] += [0.4, -0.15, 1.2]  # repeated-chorus style mismatches
    f = S.robust_affine(x, y)
    assert abs(f["slope"] - 0.98) < 3e-5 and abs(f["offset"] + 7.0) < 0.002
    assert not f["inlier"][[3, 11, 17]].any() and f["inlier"].sum() == len(x) - 3
    sd = S.prediction_sd(f, [x.min(), x.mean(), x.max()])
    assert sd[1] < sd[0] and sd[1] < 0.001


def test_local_anchor_subframe_precision():
    """A pure shift by a non-integer number of 5 ms frames is recovered to well under a frame."""
    ev = song_events(40.0, seed=5)
    shift = 0.0123  # 2.46 frames
    xs = render(ev, 40.0, SR)
    xr = render(ev, 40.0, SR, 1.0, shift)
    dt = S.frame_dt(SR)
    an = S.local_anchors(S.onset_features(xs, SR), S.onset_features(xr, SR), dt, 1.0, 0.0, [10.0, 20.0, 30.0], 8.0, 0.1)
    lags = np.array([a["lag_ms"] for a in an])
    assert len(an) == 3 and np.abs(lags - 1000 * shift).max() < 1.0


def test_pitch_check_tells_key_lock_from_varispeed():
    ev = song_events(70.0, seed=2)
    s, o = 0.97, -1.0
    xs = render(ev, 70.0, SR)
    key = S.pitch_check(xs, render(ev, 66.0, SR, s, o, snr_db=30.0), SR, s, o, (10.0, 60.0))
    var = S.pitch_check(xs, render(ev, 66.0, SR, s, o, snr_db=30.0, varispeed=True), SR, s, o, (10.0, 60.0))
    vs = -12 * np.log2(s)
    assert abs(key["semitones"]) < 0.1 and key["interpretation"].startswith("key-locked")
    assert abs(var["semitones"] - vs) < 0.1 and var["interpretation"].startswith("varispeed")


# ------------------------------------------------------------------------------------------------ grids
def grid(t0: float, bpm: float, measures: int) -> np.ndarray:
    return B.zouk_grid(t0 + 60.0 / bpm * np.arange(4 * measures + 1), 0)


def test_compare_grids_constant_offset():
    st = grid(10.0, 76.4, 20)
    ph = st[:, :2].copy()
    ph[:, 0] += 0.007
    r = S.compare_grids(st, ph, window=(20.0, 26.0))
    a = r["all"]
    assert a["n_matched"] == len(st) and a["stats"]["median_ms"] == pytest.approx(7.0, abs=1e-6)
    assert a["stats"]["abs_max_ms"] == pytest.approx(7.0, abs=1e-6) and a["stats"]["centered_max_ms"] < 1e-6
    assert a["slot_agreement"] == 1.0 and a["type_agreement"] == 1.0 and a["downbeat_agreement"] == 1.0
    assert a["slot_shift_mode"] == 0 and abs(r["drift"]["ms_per_min"]) < 1e-6
    assert r["tempo"]["studio"]["bpm"] == pytest.approx(76.4, abs=1e-3)
    w = r["window"]
    assert w["n_studio"] == int(np.sum((st[:, 0] >= 20) & (st[:, 0] <= 26))) == len(w["rows"])
    assert all(row[2] == pytest.approx(7.0, abs=1e-3) for row in w["rows"])


def test_compare_grids_drift_and_quarters():
    st = grid(10.0, 76.4, 30)
    ph = st[:, :2].copy()
    ph[:, 0] += 0.002 * (ph[:, 0] - 10.0) / 60.0  # +2 ms per minute
    r = S.compare_grids(st, ph)
    assert r["drift"]["ms_per_min"] == pytest.approx(2.0, abs=1e-6)
    assert r["drift"]["intercept_ms_at_t0"] == pytest.approx(0.0, abs=1e-6)
    assert r["all"]["quarters_stats"]["n"] == len(st) // 2


def test_compare_grids_phase_shift():
    """The phone grid believes the measure starts 2 eighths later: slots/downbeats disagree, times still match."""
    st = grid(10.0, 76.4, 12)
    t = st[2:, 0]
    ph = np.stack([t, [B.ZOUK_TYPES[i % 8] for i in range(len(t))]], 1)
    r = S.compare_grids(st, ph)["all"]
    assert r["stats"]["abs_max_ms"] < 1e-6
    assert r["slot_agreement"] == 0.0 and r["downbeat_agreement"] == 0.0 and r["slot_shift_mode"] == 6
    assert r["type_agreement"] < 0.5


def test_compare_grids_measure_numbering():
    """Same phase and times, but the studio grid starts 2 measures before the phone grid: slots and types agree while
    every measure number is off by 2 (timing.load_beats numbers measures as index // 8) - and cropping at the
    measure_origin fixes it."""
    full = grid(-0.6, 76.4, 20)  # measures of ~3.14 s; the phone grid starts at its measure 2
    ph = full[16:, :2].copy()
    r = S.compare_grids(full, ph)["all"]
    assert r["slot_agreement"] == 1.0 and r["type_agreement"] == 1.0 and r["stats"]["abs_max_ms"] < 1e-6
    assert r["measure_agreement"] == 0.0 and r["measure_shift_mode"] == -2
    org = S.measure_origin(full, ph)
    assert org["crop_measure"] == 2 and org["measure_index_offset"] == 0 and org["first_match"]["phone_index"] == 0
    st = S.crop_measures(full, org["crop_measure"])
    assert st[0, 0] == pytest.approx(ph[0, 0]) and st[0, 2] == 0 and len(st) == len(ph)
    r2 = S.compare_grids(st, ph, window=(20.0, 26.0))
    assert r2["all"]["measure_agreement"] == 1.0 and r2["all"]["measure_shift_mode"] == 0
    assert all(row[7] == row[8] for row in r2["window"]["rows"])  # measure_studio == measure_phone


def test_measure_origin_cases():
    full = grid(-0.6, 76.4, 12)
    # phone grid starts BEFORE the studio grid: no crop can fix a bare list -> reported offset
    ph_early = np.concatenate([np.stack([full[0, 0] - 0.3926 * np.arange(16, 0, -1),
                                         np.tile(B.ZOUK_TYPES, 2)], 1), full[:, :2]])
    o = S.measure_origin(full, ph_early)
    assert o["crop_measure"] == 0 and o["measure_index_offset"] == 2
    # no phone grid: first whole measure at reference >= 0 (drops the negative-time measure)
    o = S.measure_origin(full, None)
    assert o["crop_measure"] == 1 and S.crop_measures(full, 1)[0, 0] >= 0.0
    # phone bar phase differs by 2 eighths: numbers cannot be matched -> fallback, offset None
    ph_shift = np.stack([full[10:, 0], [B.ZOUK_TYPES[i % 8] for i in range(len(full) - 10)]], 1)
    o = S.measure_origin(full, ph_shift)
    assert o.get("phase_differs") and o["measure_index_offset"] is None and o["crop_measure"] == 1


def _rec_inputs(comp_studio, phone):
    sj = {"take": "t", "alignment": {"residual_ms_p95": 1.0, "residual_ms_median": 0.3, "n_inliers": 22,
                                     "n_anchors": 23, "cross_checks": {"02": 9.9}, "slope": 0.98, "playback": {}}}
    tempo = {"phase_check": {"consistent": True}, "bar_phase": {"agreement": 1.0}, "beat_confidence": 0.96,
             "linear_fit_resid_rms_ms": 15.9, "bpm": 75.0}
    return sj, S.compare_grids(comp_studio, phone), tempo


def test_recommend_measure_claim_and_steps():
    full = grid(-0.6, 76.4, 20)
    ph = full[16:, :2].copy()
    # un-cropped: the old false claim must not appear; the shift is spelled out
    sj, comp, tempo = _rec_inputs(full, ph)
    rec = S.recommend(sj, comp, tempo)
    txt = " ".join(rec["reasons"])
    assert "measure numbers DIFFER" in txt and "-2" in txt and "not the counts" not in txt
    assert rec["adopt"] == "after renumbering"
    # cropped: same numbers -> adopt, and the steps carry --no-refine, the timing.json route and the re-exports
    st = S.crop_measures(full, 2)
    sj, comp, tempo = _rec_inputs(st, ph)
    ctx = {"take": "t", "refine": {"rms_ms": 6.2, "max_abs_ms": 20.0},
           "capture": {"capture": "02_X", "time_to_audio": -29.9713246, "audio": "audio/audio_overlap.wav"},
           "soundtrack": {"reference_start": -3.1194311}, "measure_origin": {"full_measure_offset": 2,
                                                                             "measure_index_offset": 0}}
    rec = S.recommend(sj, comp, tempo, ctx)
    assert rec["adopt"] is True and "not the counts, measure labels" in " ".join(rec["reasons"])
    how = " ".join(rec["how"])
    assert "--beats work/t/song/beats_studio.json --no-refine" in how and "6.2 ms rms" in how
    assert "ignores --beats" in how and "timing/timing.json" in how
    assert "-29.9713246" in how and "+3.1194311" in how and "--audio-t0 -3.1194311" in how
    assert "dancecap.moves export t --capture 02_X" in how and "dancecap.counterbalance export --take t" in how


def test_beat_placement_sign():
    """Hats on every eighth, kicks on some: with the true beat times the 2-8 kHz flux peaks at ~0 ms; beats placed
    10 ms late see it at ~-10 ms; the kick energy (envelope) peaks after its attack; calibrated against ideal onsets
    every feature reads ~0 for true beats and ~-10 ms for beats 10 ms late."""
    sr = B.ANALYSIS_SR
    bpm = 75.0
    ev = song_events(40.0, bpm=bpm, seed=6)
    x = render(ev, 40.0, sr)
    e8 = 30.0 / bpm
    q = 0.3 + 2 * e8 * np.arange(int((40.0 - 0.3) / (2 * e8)))
    p0 = S.beat_placement(x, sr, q)
    p1 = S.beat_placement(x, sr, q + 0.010)
    assert abs(p0["hf_2_8khz_flux_peak_ms"]) < 2.0, p0
    assert p1["hf_2_8khz_flux_peak_ms"] == pytest.approx(p0["hf_2_8khz_flux_peak_ms"] - 10.0, abs=1.0)
    assert p0["kick_bass_envelope_peak_ms"] > p0["kick_bass_flux_peak_ms"]
    assert p0["n_beats"] > 40
    for k in ("kick_bass_envelope_peak", "kick_bass_flux_peak", "hf_2_8khz_flux_peak"):
        assert abs(p0[k + "_vs_ideal_ms"]) < 1.5, (k, p0)
        assert p1[k + "_vs_ideal_ms"] == pytest.approx(-10.0, abs=1.5), (k, p1)


def test_to_reference_grid_keeps_whole_measures():
    g = grid(0.2, 75.0, 10)  # song time, measures of 3.2 s
    s, o = 0.98, -6.0
    out, first = S.to_reference_grid(g, s, o, 0.0, 25.0)
    assert len(out) % 8 == 0 and len(out)
    assert out[0, 0] >= 0.0 and out[-1, 0] <= 25.0
    assert np.array_equal(out[:, 2], np.repeat(np.arange(len(out) // 8), 8))
    assert np.array_equal(out[:, 1].astype(int), np.tile(B.ZOUK_TYPES, len(out) // 8))
    src = g[g[:, 2] == first]
    assert np.allclose(out[:8, 0], s * src[:, 0] + o)


# ------------------------------------------------------------------------------------------------ paths
def test_demojibake_and_name_key():
    assert S.demojibake("VocÃª") == "Você"
    assert S.demojibake("Você") == "Você"
    assert S.demojibake("plain.mp3") == "plain.mp3"
    nfd = unicodedata.normalize("NFD", NAME)
    assert nfd != NAME and S.name_key(nfd) == S.name_key(NAME) == S.name_key("Ficar Sem VocÃª")


def test_resolve_path_unicode_variants(tmp_path):
    d = tmp_path / "danças"
    d.mkdir()
    f = d / f"{NAME}.mp3"
    f.write_bytes(b"x")
    nfd = tmp_path / unicodedata.normalize("NFD", "danças") / unicodedata.normalize("NFD", f"{NAME}.mp3")
    moj = d / "Ficar Sem VocÃª.mp3"
    for p in (f, nfd, moj):
        assert S.resolve_path(p).samefile(f)
    with pytest.raises(FileNotFoundError):
        S.resolve_path(d / "Ficar Sem Voce.mp3")


def test_song_info_reads_toml_as_utf8(tmp_path):
    media = tmp_path / "media"
    media.mkdir()
    (media / f"{NAME}.mp3").write_bytes(b"id3")
    takes = tmp_path / "takes"
    takes.mkdir()
    rel = "../media/" + unicodedata.normalize("NFD", f"{NAME}.mp3")  # NFD in the toml, NFC on disk
    (takes / "t.toml").write_text(f'name = "t"\n[dance]\nsong = "{NAME} (feat. Urias)"\nartist = "Davi Sabbag"\n'
                                  f'song_file = "{rel}"\n', encoding="utf-8")
    info = S.song_info(takes / "t.toml")
    assert info.title == f"{NAME} (feat. Urias)" and info.artist == "Davi Sabbag"
    assert info.path.samefile(media / f"{NAME}.mp3")
    assert S.sha256_16(info.path) == __import__("hashlib").sha256(b"id3").hexdigest()[:16]
    assert "\\" not in info.rel


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not on PATH")
def test_decode_non_ascii_path(tmp_path):
    p = tmp_path / f"{NAME} áção.wav"
    x = 0.3 * np.sin(2 * np.pi * 440 * np.arange(SR) / SR)
    B.write_wav(p, x, SR)
    y = S.load_mono(p, SR)
    assert abs(len(y) - SR) <= 1 and np.corrcoef(y[:SR - 1], x[:SR - 1])[0, 1] > 0.99
    y2 = S.load_mono(p.parent / unicodedata.normalize("NFD", p.name), SR)  # NFD spelling resolves too
    assert len(y2) == len(y)


def test_copyright_guard(tmp_path):
    with pytest.raises(PermissionError):
        S.assert_private(tmp_path / "studio.wav")
    with pytest.raises(PermissionError):
        S.assert_private(PACKAGE_ROOT / "takes" / "studio.wav")
    assert S.assert_private(S.WORK_ROOT / "some_take" / "song" / "studio.wav").name == "studio.wav"


# ------------------------------------------------------------------------------------------------ soundtrack
def test_place_on_reference_timeline():
    sr = 1000
    y = np.zeros((5000, 2), np.float32)
    y[2500] = 1.0  # at reference offset + 2.5 s
    out, t0 = S.place(y, sr, offset=-3.0, t0=-1.0, t1=3.0)
    assert t0 == pytest.approx(-1.0) and len(out) == 4000
    assert int(np.argmax(out[:, 0])) == 500  # -0.5 s reference -> 0.5 s after out[0]
    out2, t02 = S.place(y, sr, offset=-3.0, t0=-4.0, t1=0.0)  # starts before the song: zero padded
    assert t02 == pytest.approx(-4.0) and np.all(out2[:1000] == 0) and int(np.argmax(out2[:, 0])) == 3500


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not on PATH")
def test_stretch_audio_timing(tmp_path):
    """Rubber Band stretch of a synthetic track (non-ASCII file name): content lands at slope * song time within
    ~1 ms and the duration scales by slope."""
    try:
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo", "-t", "0.1", "-af",
                        "rubberband=tempo=1.01", "-f", "null", "-"], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        pytest.skip("ffmpeg without librubberband")
    ev = song_events(36.0, seed=4)
    x = render(ev, 36.0, 44100)
    p = tmp_path / f"{NAME}.wav"
    B.write_wav(p, np.stack([x, x], 1), 44100)
    slope = 0.98
    cfg = S.SongConfig(anchor_window_s=4.0, anchor_step_s=4.0)
    y, method = S.stretch_audio(p, slope, 48000, cfg)
    assert "rubberband" in method and y.shape[1] == 2
    assert abs(len(y) / 48000 - slope * 36.0) < 0.05
    import soxr
    Fs = S.onset_features(render(ev, 36.0, SR), SR)
    Fy = S.onset_features(soxr.resample(y.mean(1), 48000, SR), SR)
    chk = S.stretch_timing(Fs, Fy, S.frame_dt(SR), slope, 0.0, cfg)
    assert chk["n_anchors"] >= 5 and abs(chk["latency_ms"]) < 1.5 and abs(chk["rate_ppm"]) < 100
    y1, m1 = S.stretch_audio(p, 1.0 + 1e-7, 48000, cfg)  # within stretch_tol: decode only
    assert "shift only" in m1 and abs(len(y1) / 48000 - 36.0) < 0.01


# ------------------------------------------------------------------------------------------------ real take
@pytest.mark.skipif(not (REAL / "song.json").exists(), reason="larissa_kadu song stage not run")
def test_real_song_contract():
    sj = json.loads((REAL / "song.json").read_text(encoding="utf-8"))
    assert sj["version"] == 1 and sj["take"] == "larissa_kadu"
    assert set(sj["song"]) >= {"title", "artist", "file", "sha256_16", "duration"} and len(sj["song"]["sha256_16"]) == 16
    al = sj["alignment"]
    assert al["convention"] == "reference_time = slope * song_time + offset"
    assert set(al) >= {"slope", "offset", "anchors", "residual_ms_median", "residual_ms_p95", "cross_checks"}
    assert all(set(a) >= {"song_t", "ref_t", "residual_ms", "confidence", "inlier"} for a in al["anchors"])
    assert al["residual_ms_p95"] < 5.0 and al["n_inliers"] >= 0.8 * al["n_anchors"]
    for a in al["anchors"]:
        assert abs(a["ref_t"] - (al["slope"] * a["song_t"] + al["offset"]) - a["residual_ms"] / 1000) < 1e-5
    assert set(sj["coverage"]) >= {"song", "reference"}
    assert (WORKSPACE / sj["song"]["file"]).exists() or S.resolve_path(WORKSPACE / sj["song"]["file"]).exists()


@pytest.mark.skipif(not (REAL / "beats_studio.json").exists(), reason="larissa_kadu studio beats not run")
def test_real_beats_studio_contract():
    from dancecap import timing
    g = np.asarray(json.loads((REAL / "beats_studio.json").read_text()), float)
    assert g.ndim == 2 and g.shape[1] == 2 and len(g) % 8 == 0 and np.all(np.diff(g[:, 0]) > 0)
    assert np.array_equal(g[:, 1].astype(int), np.tile(B.ZOUK_TYPES, len(g) // 8))
    bg = timing.load_beats(REAL / "beats_studio.json")  # drop-in for the timing loader
    assert len(bg.t) == len(g)
    raw = json.loads((REAL / "beats_studio_raw.json").read_text(encoding="utf-8"))
    sj = json.loads((REAL / "song.json").read_text(encoding="utf-8"))
    st = np.asarray(raw["song_time"], float)
    assert len(st) == len(g) and "provenance" in raw
    assert np.allclose(sj["alignment"]["slope"] * st[:, 0] + sj["alignment"]["offset"], g[:, 0], atol=2e-6)
    assert 55 < 60 / np.median(np.diff(g[::2, 0])) < 120  # quarter-note BPM, not the doubled analyzer grid
    # measure-aligned with beats.json: same measure numbers under timing.load_beats (index // 8), no negative times
    org = raw["provenance"]["measure_origin"]
    phone_p = PACKAGE_ROOT / "work" / "larissa_kadu" / "beats.json"
    if phone_p.exists():
        ph = np.asarray(json.loads(phone_p.read_text()), float)
        assert org["measure_index_offset"] == 0 and abs(g[0, 0] - ph[0, 0]) < 0.05 and g[0, 1] == 1
        n = min(len(g), len(ph))  # entry i of one grid is entry i of the other: same index // 8 measure numbers
        assert np.abs(g[:n, 0] - ph[:n, 0]).max() < 0.1
    assert g[0, 0] >= 0.0
    full = np.asarray(json.loads((REAL / "beats_studio_full.json").read_text()), float)
    k = 8 * org["full_measure_offset"]
    assert np.array_equal(full[k:k + len(g)], g) and len(full) >= len(g)
    assert np.allclose(np.asarray(raw["reference_time_full"], float)[:, :2], full, atol=1e-6)
    bp = raw["provenance"]["beat_placement"]
    assert {"hf_2_8khz_flux_peak_ms", "kick_bass_envelope_peak_ms", "meaning"} <= set(bp)


@pytest.mark.skipif(not (REAL / "comparison.json").exists(), reason="larissa_kadu comparison not run")
def test_real_comparison_and_privacy():
    c = json.loads((REAL / "comparison.json").read_text(encoding="utf-8"))
    assert {"tempo", "all", "drift", "window", "recommendation"} <= set(c)
    assert c["window"]["range"] == [56.0, 62.0]
    assert c["all"]["measure_agreement"] == 1.0 and c["all"]["measure_shift_mode"] == 0
    assert c["grids"]["studio_full"]["measure_shift_mode"] == -c["grids"]["studio_full"]["full_measure_offset"] or \
        c["grids"]["studio_full"]["full_measure_offset"] == 0
    how = " ".join(c["recommendation"]["how"])
    assert "--no-refine" in how and "ignores --beats" in how and "time_to_audio" in how
    js = json.loads((REAL / "studio_reference.json").read_text(encoding="utf-8"))
    assert js["time_to_audio"] == pytest.approx(-js["reference_start"])
    assert abs(js["end_to_end_vs_reference"]["offset_error_ms_median"]) < 2.0
    assert "must not be distributed" in js["distribution"] and js["export_hint"].startswith("PRIVATE")
    assert js["timing_accuracy_ms"] == 1.0
    assert json.loads((REAL / "song.json").read_text(encoding="utf-8"))["alignment"]["accuracy_ms"] == 1.0
    # copyright: the soundtrack is git-ignored and no song audio is tracked anywhere in the workspace repo
    r = subprocess.run(["git", "check-ignore", "-q", str(REAL / "studio_reference.wav")], cwd=PACKAGE_ROOT)
    assert r.returncode == 0
    tracked = subprocess.run(["git", "ls-files"], cwd=WORKSPACE, capture_output=True, text=True,
                             encoding="utf-8").stdout.splitlines()
    assert not [t for t in tracked if t.lower().endswith((".mp3", ".wav")) and ("ficar" in t.lower() or "/song/" in t)]
