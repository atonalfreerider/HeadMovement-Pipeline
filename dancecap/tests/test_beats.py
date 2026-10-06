"""Unit tests for dancecap.beats + an end-to-end run on a synthetic 3-phone zouk take + a contract check on
work/larissa_kadu (skipped when that take has not been processed).

    .venv-nopo4d\\Scripts\\python.exe -m pytest dancecap/tests/test_beats.py -q
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dancecap import beats as B  # noqa: E402
from dancecap.take import PACKAGE_ROOT, Source, Take  # noqa: E402

REAL = PACKAGE_ROOT / "work" / "larissa_kadu"


# ------------------------------------------------------------------------------------------------ synthetic music
def drum_track(times_ref: np.ndarray, kinds: list[str], dur_s: float, sr: int, slope=1.0, offset=0.0,
               noise=0.0, seed=0) -> np.ndarray:
    """Render kick/snare/hat events given in REFERENCE seconds onto a phone whose clock is
    reference = slope * source + offset (so each event lands at source time (t - offset) / slope)."""
    rng = np.random.default_rng(seed)
    x = np.zeros(int(dur_s * sr), np.float32)
    k_t = np.arange(int(0.18 * sr)) / sr
    kick = (np.sin(2 * np.pi * (55 + 60 * np.exp(-k_t / 0.03)) * k_t) * np.exp(-k_t / 0.06)).astype(np.float32)
    sn = (np.random.default_rng(1).normal(0, 1, int(0.08 * sr)) * np.exp(-np.arange(int(0.08 * sr)) / (0.02 * sr)))
    hat = (np.random.default_rng(2).normal(0, 1, int(0.03 * sr)) * np.exp(-np.arange(int(0.03 * sr)) / (0.005 * sr)))
    hat = np.diff(hat, prepend=0)
    bank = {"kick": 0.9 * kick, "snare": 0.35 * sn.astype(np.float32), "hat": 0.12 * hat.astype(np.float32)}
    for t, k in zip(times_ref, kinds):
        s = int(round((t - offset) / slope * sr))
        if 0 <= s < len(x):
            e = min(len(x), s + len(bank[k]))
            x[s:e] += bank[k][:e - s]
    if noise:
        x += rng.normal(0, noise, len(x)).astype(np.float32)
    return x


def zouk_events(bpm: float, t0: float, t1: float):
    """Zouk pattern per measure: kick on eighths 0, 3, 5; snare on 2, 6; hats on every eighth."""
    e8 = 30.0 / bpm
    times, kinds, downs = [], [], []
    m = t0
    while m < t1:
        downs.append(m)
        for j in range(8):
            t = m + j * e8
            times.append(t)
            kinds.append("hat")
            if j in (0, 3, 5):
                times.append(t)
                kinds.append("kick")
            if j in (2, 6):
                times.append(t)
                kinds.append("snare")
        m += 8 * e8
    return np.array(times), kinds, np.array(downs)


# ------------------------------------------------------------------------------------------------ grid logic
def test_zouk_grid_slots_types_and_midpoints():
    q = 1.0 + np.arange(13) * 60 / 77.0
    g = B.zouk_grid(q, phase=1)
    assert len(g) == 16  # quarters 1..12 hold two whole measures (closing eighth needs quarter 9 / 13)
    assert g[0, 0] == pytest.approx(q[1]) and g[8, 0] == pytest.approx(q[5])
    assert g[:8, 1].tolist() == [1, 3, 3, 2, 1, 3, 2, 3]
    assert g[1, 0] == pytest.approx(0.5 * (q[1] + q[2]))  # eighth = midpoint of quarter beats
    assert g[:, 3].tolist() == list(range(8)) * 2 and g[:, 2].tolist() == [0] * 8 + [1] * 8


def test_fix_beat_level_eighths_and_half_notes():
    q = np.arange(80) * 60 / 77.0
    eighths = np.sort(np.concatenate([q, q[:-1] + 30 / 77.0]))  # tracker locked onto eighths (154 BPM)
    downs = q[1::4]  # downbeats on the odd... quarter index 1 -> eighth index 2
    out, info = B.fix_beat_level(eighths, downs)
    assert info["factor"] == 0.5 and 60 / np.median(np.diff(out)) == pytest.approx(77, abs=0.01)
    assert np.isin(np.round(downs, 6), np.round(out, 6)).all()  # parity kept the downbeats
    half = q[::2]  # tracker on half notes (38.5 BPM)
    out, info = B.fix_beat_level(half, half[::2])
    assert info["factor"] == 2 and 60 / np.median(np.diff(out)) == pytest.approx(77, abs=0.01)


def test_repair_gaps_inserts_missing_and_drops_extra():
    q = np.arange(40) * 0.78
    b = np.delete(q, 10)
    out, info = B.repair_gaps(b)
    assert info["inserted"] == 1 and np.allclose(out, q)
    b = np.sort(np.append(q, q[20] + 0.3))
    out, info = B.repair_gaps(b)
    assert info["dropped"] == 1 and np.allclose(out, q)


def test_bar_phase_majority_vote():
    q = np.arange(40) * 0.78
    downs = q[2::4] + 0.01
    ph = B.bar_phase(q, np.append(downs, q[5]))  # one wrong downbeat
    assert ph["phase"] == 2 and ph["agreement"] == pytest.approx(10 / 11, abs=1e-3)


def test_analyzer_split_check_flags_slow_track():
    d = np.arange(20) * 4 * 60 / 77.0
    s = B.analyzer_split_check(d)
    assert s["split"] == 19 and s["resulting_grid_bpm_median"] == pytest.approx(154, abs=0.1)
    d = np.arange(20) * 4 * 60 / 130.0
    assert B.analyzer_split_check(d)["split"] == 0


def test_phase_check_kick_and_backbeat():
    bpm = 77.0
    times, kinds, downs = zouk_events(bpm, 0.5, 40)
    q = np.arange(downs[0], downs[-1] + 1e-6, 60 / bpm)
    ot = np.arange(0, 41, 0.005)

    def onsets(kind):
        o = np.zeros_like(ot)
        o[np.searchsorted(ot, times[np.array(kinds) == kind])] = 1.0
        return o

    ok, os_ = onsets("kick"), onsets("snare")
    for phase, shift in ((0, 0), (2, 4), (1, 2), (3, 6)):  # grid bar starts `shift` eighths after the true downbeat
        g = B.zouk_grid(q, phase)
        r = B.phase_check(B.folded_profile(g, ot, ok), B.folded_profile(g, ot, os_))
        assert r["consistent"] == (shift == 0) and r["best"] == (8 - shift) % 8, r
    # off-eighth quarters (tracker locked onto the 'and's): rephase by the best odd shift restores the truth
    qe = q[:-1] + 30 / bpm
    g = B.zouk_grid(qe, 0)
    r = B.rotation_scores(B.folded_profile(g, ot, ok), B.folded_profile(g, ot, os_))
    assert r["best"] == 7
    q2, ph2 = B.rephase(qe, 0, r["best"])
    g2 = B.zouk_grid(q2, ph2)
    assert np.abs(g2[g2[:, 3] == 0, 0][:, None] - downs[None, :]).min(1).max() < 1e-9


def test_pick_peaks_subframe():
    t = np.arange(0, 10, 0.01)
    true = np.array([1.2345, 2.0123, 3.3337, 7.777])
    p = sum(np.exp(-0.5 * ((t - c) / 0.02) ** 2) for c in true)
    out = B.pick_peaks(t, p)
    assert len(out) == 4 and np.abs(out - true).max() < 1.5e-3
    assert B.snap(np.array([2.03, 5.0]), out).tolist() == [pytest.approx(2.0123, abs=2e-3)]
    assert B.disagreement(np.array([1.24, 4.0]), out).tolist() == [4.0]


# ------------------------------------------------------------------------------------------------ audio
def test_to_reference_clock_and_rate():
    sr, slope, offset = 8000, 1.0002, 3.3
    x = np.zeros(20 * sr, np.float32)
    x[10 * sr] = 1.0  # impulse at source time 10 s -> reference 3.3 + 10.002
    y, t0 = B.to_reference(x, sr, slope, offset, 5.0, 20.0, 16000)
    assert abs(t0 - 5.0) < 0.5 / 16000 and len(y) == 15 * 16000
    t_peak = t0 + np.argmax(np.abs(y)) / 16000
    assert t_peak == pytest.approx(offset + slope * 10.0, abs=1.5e-4)
    y, t0 = B.to_reference(x, sr, 1.0, offset, 0.0, 5.0, sr)  # before the recording starts: zeros
    assert np.all(y[: int(3.3 * sr) - 1] == 0)


def test_envelope_snr_ranks_noisy_phone_last():
    sr = 16000
    times, kinds, _ = zouk_events(77, 0.2, 20)
    envs = {}
    for c, n in (("a", 0.003), ("b", 0.006), ("c", 0.01), ("d", 0.2)):
        x = drum_track(times, kinds, 20, sr, noise=n, seed=hash(c) % 100)
        envs[c] = B.logmel_envelope(x, sr)
    snr = B.envelope_snr(envs)
    assert min(snr, key=lambda c: snr[c]["snr_db"]) == "d"
    assert snr["d"]["snr_db"] < snr["a"]["snr_db"] - 3


def test_click_track_and_wav_roundtrip(tmp_path):
    sr = 8000
    grid = np.array([[1.0, 1], [1.5, 3], [2.0, 2]])
    c = B.click_track(3 * sr, sr, grid, t_audio0=0.5)
    on = np.nonzero(np.abs(c) > 1e-4)[0]
    starts = on[np.insert(np.diff(on) > 10, 0, True)]
    assert np.allclose(starts / sr + 0.5, grid[:, 0], atol=2 / sr)
    assert np.abs(c[int(0.5 * sr):int(0.53 * sr)]).max() > np.abs(c[int(1.0 * sr):int(1.03 * sr)]).max()  # type 1 loud
    B.write_wav(tmp_path / "x.wav", c, sr)
    y, sr2 = B.read_wav(tmp_path / "x.wav")
    assert sr2 == sr and np.abs(y - c).max() < 1e-4


# ------------------------------------------------------------------------------------------------ end to end
def _beat_this_ok() -> bool:
    try:
        import beat_this.inference  # noqa: F401
        import torchaudio  # noqa: F401
    except Exception:
        return False
    return (Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "beat_this-final0.ckpt").exists()


@pytest.mark.skipif(not _beat_this_ok(), reason="beat_this / final0 checkpoint not installed")
def test_end_to_end_synthetic_three_phones():
    """3 phones with different clocks (slope, offset) and noise; truth 77 BPM zouk with known downbeats."""
    name = "_test_beats"
    work = PACKAGE_ROOT / "work" / name
    shutil.rmtree(work, ignore_errors=True)
    src_dir = work / "src"
    src_dir.mkdir(parents=True)
    bpm, sr = 77.0, 22050
    times, kinds, downs = zouk_events(bpm, 1.0, 62.0)
    phones = {"01": (1.0, 0.0, 0.03), "02": (1.00015, 2.5, 0.002), "03": (0.9998, -1.2, 0.08)}
    sources, rows = [], []
    for c, (slope, off, noise) in phones.items():
        x = drum_track(times, kinds, 58.0, sr, slope, off, noise, seed=int(c))
        B.write_wav(src_dir / f"{c}.wav", 0.8 * x / np.abs(x).max(), sr)
        sources.append(Source(c, src_dir / f"{c}.wav", slope, off))
        rows.append({"id": c, "duration": 58.0, "slope": slope, "offset": off})
    ov = (max(s.to_reference(0) for s in sources), min(s.to_reference(58.0) for s in sources))
    (work / "alignment.json").write_text(json.dumps({"sources": rows, "maximum_overlap": [
        {"start": ov[0], "end": ov[1]}]}))
    take = Take(name, sources, (20.0, 26.0), (20.0, 26.0), 2.0, {"alignment": str(work / "alignment.json")})
    try:
        res = B.run(take, B.BeatsConfig(device="cpu"), plot=True)
        t = res["tempo"]
        assert res["source"] == "02"  # least noise
        assert t["bpm"] == pytest.approx(bpm, abs=0.5)  # phones 01/03 lock onto eighths: level fix must undo it
        assert t["phase_check"]["consistent"]
        assert t["analyzer_split_measures"]["resulting_grid_bpm_median"] == pytest.approx(2 * bpm, abs=1)
        g = np.array(json.loads((work / "beats" / "beats.json").read_text()))
        assert g[0, 1] == 1 and len(g) % 8 == 0
        assert g[:16, 1].tolist() == [1, 3, 3, 2, 1, 3, 2, 3] * 2
        d = g[::8, 0]  # grid downbeats must be true downbeats (reference clock) within 15 ms
        err = np.abs(d[:, None] - downs[None, :]).min(1)
        assert np.median(err) < 0.015 and err.max() < 0.03
        e8 = np.abs(g[:, 0][:, None] - (downs[:, None] + np.arange(8) * 30 / bpm).ravel()[None, :]).min(1)
        assert np.median(e8) < 0.015
        meta = json.loads((work / "audio" / "audio_overlap.json").read_text())
        assert meta["source_cam"] == "02" and meta["reference_start"] == pytest.approx(ov[0], abs=1e-4)
        y, ysr = B.read_wav(work / "audio" / "audio_overlap.wav")
        assert ysr == sr and len(y) / ysr == pytest.approx(ov[1] - ov[0], abs=1e-3)
        # first kick inside the wav sits at its reference time
        k = times[(np.array(kinds) == "kick") & (times > ov[0] + 0.5)][0]
        i = int((k - meta["reference_start"]) * ysr)
        seg = np.abs(y[i - int(0.05 * ysr): i + int(0.05 * ysr)])
        onset = i - int(0.05 * ysr) + np.argmax(seg > 0.3 * seg.max())
        assert abs(onset / ysr + meta["reference_start"] - k) < 0.003
        assert (work / "review" / "beats_waveform.png").exists() and (work / "beats.json").exists()
        assert (work / "audio" / "clicks_overlay.wav").exists()
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ------------------------------------------------------------------------------------------------ real take contract
@pytest.mark.skipif(not (REAL / "beats" / "tempo.json").exists(), reason="run `python -m dancecap beats larissa_kadu`")
def test_real_take_contract():
    g = np.array(json.loads((REAL / "beats" / "beats.json").read_text()))
    tempo = json.loads((REAL / "beats" / "tempo.json").read_text())
    meta = json.loads((REAL / "audio" / "audio_overlap.json").read_text())
    assert g.shape[1] == 2 and len(g) % 8 == 0 and np.all(np.diff(g[:, 0]) > 0)
    assert g[:, 1].astype(int).tolist() == [1, 3, 3, 2, 1, 3, 2, 3] * (len(g) // 8)
    assert 60 <= tempo["bpm"] < 120 and tempo["phase_check"]["consistent"]
    assert g[0, 0] <= meta["reference_start"] and g[-1, 0] >= meta["reference_end"] - tempo["measure_s"]
    y, sr = B.read_wav(REAL / "audio" / "audio_overlap.wav")
    assert len(y) == meta["n_samples"] and sr == meta["sample_rate"]
    assert meta["time_to_audio"] == pytest.approx(-meta["reference_start"])
    # downstream loaders accept the file (timing: BeatGrid measures, physics: beat times)
    from dancecap import physics, timing
    bg = timing.load_beats(REAL / "beats" / "beats.json")
    assert bg.pos[0] == 0 and np.all(bg.type[bg.pos == 0] == 1)
    assert len(physics.load_beat_times(REAL / "beats.json")) == len(g)
    assert 240 / np.median(np.diff(g[::8, 0])) < 120  # not the analyzer's dense grid
