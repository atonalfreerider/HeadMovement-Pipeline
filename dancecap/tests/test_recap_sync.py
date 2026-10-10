"""Unit tests for dancecap.recap_sync (synthetic speech-like audio; no footage, no GPU).

    .venv-nopo4d\\Scripts\\python.exe -m pytest dancecap/tests/test_recap_sync.py -q      (from the workspace root)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dancecap import recap_sync as rs  # noqa: E402
from _needs import WORKSPACE, needs_paths, needs_take  # noqa: E402

FS = rs.SR


def speech_like(seconds: float, seed: int) -> np.ndarray:
    """Noise in the speech band with a syllable-rate (3-6 Hz) on/off envelope and a few pauses."""
    rng = np.random.default_rng(seed)
    n = int(seconds * FS)
    x = rng.normal(0, 1, n)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1 / FS)
    X[(f < 200) | (f > 3500)] = 0
    x = np.fft.irfft(X, n)
    t = np.arange(n) / FS
    env = np.zeros(n)
    pos = 0.0
    while pos < seconds:
        d = rng.uniform(0.08, 0.25)
        gap = rng.uniform(0.03, 0.12) if rng.random() > 0.08 else rng.uniform(0.5, 1.5)
        env[(t >= pos) & (t < pos + d)] = rng.uniform(0.3, 1.0)
        pos += d + gap
    y = x * np.convolve(env, np.hanning(int(0.02 * FS)) / np.hanning(int(0.02 * FS)).sum(), mode="same")
    return (0.3 * y / np.abs(y).max()).astype(np.float32)


def delayed_copy(master: np.ndarray, offset_s: float, slope: float, seconds: float, noise: float, seed: int) -> np.ndarray:
    """What a phone with the clock master_t = slope * cam_t + offset hears of the master, plus its own noise."""
    n = int(seconds * FS)
    cam_t = np.arange(n) / FS
    mt = slope * cam_t + offset_s
    out = np.interp(mt * FS, np.arange(len(master)), master, left=0.0, right=0.0)
    rng = np.random.default_rng(seed)
    return (out + noise * rng.normal(0, 1, n)).astype(np.float32)


# --------------------------------------------------------------------------------------------- correlation


def test_estimate_delay_recovers_a_known_delay_to_a_fraction_of_a_millisecond():
    base = speech_like(10, 1)
    d = int(0.2437 * FS)
    y = np.concatenate([np.zeros(d, np.float32), base])[:len(base)]
    r = rs.estimate_delay(base, y, FS, lag_s=(0.0, 1.0))
    assert abs(r["delay_s"] - 0.2437) < 0.0004
    assert r["z"] > 15


def test_wide_locate_finds_a_window_in_the_whole_master_and_rejects_an_unrelated_one():
    master = speech_like(90, 2)
    index = rs.MasterIndex(master, 13.0)
    win = master[int(31.37 * FS):int((31.37 + 12) * FS)]
    r = index.locate(win)
    assert abs(r["delay_s"] - 31.37) < 0.001 and rs.good_wide(r)
    other = speech_like(12, 99)
    assert not rs.good_wide(index.locate(other))


def test_no_edge_artefact_near_the_master_start():
    """A window of speech unrelated to the master must not produce a confident peak at the master's very start."""
    master = speech_like(60, 3)
    index = rs.MasterIndex(master, 13.0)
    z = [index.locate(speech_like(12, 200 + k), lag_s=(-6.0, 1.0))["z"] for k in range(3)]
    assert max(z) < rs.MIN_Z_WIDE


# --------------------------------------------------------------------------------------------- clocks and clusters


def test_measure_camera_finds_offset_and_real_drift_and_ignores_a_second_acoustic_path():
    master = speech_like(240, 4)
    slope, offset = 1.00006, 7.4
    cam = delayed_copy(master, offset, slope, 220, noise=0.03, seed=5)
    m = rs.measure_camera(cam, master, 7.43)          # the wide vote is only good to ~0.1 s
    assert m["ok"] and m["n_main"] >= 40
    d = m["decision"]
    assert d["clock"] == "affine"
    assert abs(d["slope"] - slope) < 8e-6
    assert abs(d["offset_s"] - offset) < 0.003
    assert d["residual_ms"] < 3


def test_measure_camera_constant_clock_when_there_is_no_drift():
    master = speech_like(150, 6)
    cam = delayed_copy(master, 3.21, 1.0, 140, noise=0.03, seed=7)
    m = rs.measure_camera(cam, master, 3.2)
    assert m["ok"] and m["decision"]["clock"] == "constant" and m["decision"]["slope"] == 1.0
    assert abs(m["decision"]["offset_s"] - 3.21) < 0.002


def test_densest_cluster_and_secondary_clusters():
    v = np.concatenate([np.full(10, 1.000) + np.linspace(-0.003, 0.003, 10), np.full(4, 1.045), [1.5]])
    main = rs.densest_cluster(v)
    assert main[:10].all() and not main[10:].any()
    sec = rs.secondary_clusters(v, main)
    assert len(sec) == 1 and sec[0]["n"] == 4 and abs(sec[0]["offset_ms_from_main"] - 45) < 3


def test_fit_clock_affine_vs_constant_decision():
    rng = np.random.default_rng(8)
    t = np.linspace(10, 330, 70)
    drifting = 1.00002 * t + 5.0 + rng.normal(0, 0.0015, t.size)
    flat = t + 5.0 + rng.normal(0, 0.0015, t.size)
    fa, ff = rs.fit_clock(t, drifting), rs.fit_clock(t, flat)
    assert rs.decide_clock(fa)["clock"] == "affine" and abs(fa["drift_ppm"] - 20) < 3
    assert rs.decide_clock(ff)["clock"] == "constant"
    assert not rs.fit_clock(t[:2], flat[:2])["ok"]


# --------------------------------------------------------------------------------------------- envelope search


def _refs(master, seeds):
    envs = {"01": (rs.log_envelope(master), 1.0, 0.0)}
    for k, (off, sd) in enumerate(seeds, start=2):
        envs[f"{k:02d}"] = (rs.log_envelope(delayed_copy(master, off, 1.0, 100, 0.05, sd)), 1.0, off)
    return envs


def test_envelope_search_finds_a_hidden_camera_and_reports_nothing_for_unrelated_audio():
    master = speech_like(120, 9)
    refs = _refs(master, [(2.0, 10), (-3.0, 11)])
    hidden = delayed_copy(master, 40.0, 1.0, 40, 0.05, 12)
    res = rs.search_unlisted(rs.log_envelope(hidden), refs, 40.0, scales=(1.0, 1.0, 0.01))
    assert res["found"] and abs(res["offset_s"] - 40.0) < 0.05
    unrelated = rs.search_unlisted(rs.log_envelope(speech_like(40, 77)), refs, 40.0, scales=(1.0, 1.0, 0.01))
    assert not unrelated["found"]


# --------------------------------------------------------------------------------------------- timeline, alignment, quality


def test_coverage_timeline_counts_cameras_per_second():
    cams = [{"id": "01", "coverage_master_s": [0.0, 10.0]}, {"id": "02", "coverage_master_s": [3.5, 8.0]},
            {"id": "03", "coverage_master_s": [-2.0, 4.0]}]
    tl = rs.coverage_timeline(cams, (0.0, 10.0))
    assert tl["cameras_per_second"] == [2, 2, 2, 3, 2, 2, 2, 2, 1, 1]
    assert tl["seconds_with_n_cameras"] == {"1": 2, "2": 7, "3": 1}
    assert tl["runs"][0] == {"t0": 0.0, "t1": 3.0, "cams": ["01", "03"], "n": 2}


def test_alignment_doc_is_read_by_take_and_leaves_an_unplaced_camera_out(tmp_path):
    from dancecap.take import Take
    sync = {"master": {"file": "01-Recap-Tina.mp4"},
            "cameras": [{"id": "01", "file": "01-Recap-Tina.mp4", "video_duration_s": 360.4, "slope": 1.0, "offset_s": 0.0,
                         "residual_ms": 0.0, "confidence": 1.0, "offset_source": "master", "placed": True},
                        {"id": "02", "file": "02-Recap-Alyssa.mp4", "video_duration_s": 358.4, "slope": 1.00001, "offset_s": 1.3437,
                         "residual_ms": 2.0, "confidence": 0.9, "offset_source": "measured", "placed": True},
                        {"id": "04", "file": "04-Recap-Tony.mp4", "video_duration_s": 77.3, "slope": 1.0, "offset_s": None,
                         "residual_ms": None, "confidence": 0.0, "offset_source": "not found", "placed": False, "note": "no match"}]}
    doc = rs.alignment_doc(sync)
    assert [s["accepted"] for s in doc["sources"]] == [True, True, False]
    assert doc["sources"][2]["offset"] is None
    takes = tmp_path / "takes"
    takes.mkdir()
    (tmp_path / "align.json").write_text(json.dumps(doc), encoding="utf-8")
    (takes / "t.toml").write_text('name = "t"\nsource_dir = "../src"\nalignment = "../align.json"\nwindow = [0.0, 100.0]\n',
                                  encoding="utf-8")
    take = Take.load(takes / "t.toml")
    assert [s.id for s in take.sources] == ["01", "02"]
    assert abs(take.source("02").to_reference(100.0) - (1.0000100 * 100.0 + 1.3437)) < 1e-9


def test_quality_windows_and_agreement_rank_the_cleaner_phone_first():
    master = speech_like(100, 13)
    clean = delayed_copy(master, 0.0, 1.0, 100, 0.01, 14)
    noisy = delayed_copy(master, 0.0, 1.0, 100, 0.5, 15)
    other = delayed_copy(master, 0.0, 1.0, 100, 0.05, 16)
    audio = {"01": clean, "02": noisy, "03": other}
    quality = {k: rs.quality_windows(v, 1.0, 0.0, 0.0, 100.0) for k, v in audio.items()}
    assert [r["t0"] for r in quality["01"]] == [0.0, 30.0, 60.0]
    assert quality["01"][0]["snr_db"] > quality["02"][0]["snr_db"]
    envs = {k: (rs.log_envelope(v), 1.0, 0.0) for k, v in audio.items()}
    rs.add_agreement(quality, envs, t0=-2.0, t1=102.0)
    assert quality["01"][0]["env_ncc"] > quality["02"][0]["env_ncc"]
    assert quality["02"][0]["rank"] == 3
    summ = rs.quality_summary(quality)
    assert summ["order_by_env_ncc"][-1] == "02"


def test_pose_json_orientation_is_display_for_a_rotated_video(tmp_path):
    frames = [[[{"X": 20.0 + 5 * i, "Y": 100.0 + 30 * i, "Z": 0.9} for i in range(17)]] for _ in range(10)]
    p = tmp_path / "x.json"
    p.write_text(json.dumps(frames), encoding="utf-8")
    f = rs.pose_json_facts(p, stored=[640, 480], display=[480, 640])
    assert f["orientation"].startswith("display") and f["frames"] == 10
    f2 = rs.pose_json_facts(p, stored=[480, 640], display=[640, 480])      # coordinates only fit the 480 x 640 frame
    assert f2["orientation"] == "stored"


# --------------------------------------------------------------------------------------------- the real take (skipped without it)


@needs_take("larissa_kadu_recap")
@needs_paths(WORKSPACE / "prototype" / "larissa_kadu" / "alignment_recap.json", why="recap alignment")
def test_real_recap_take_loads_with_the_master_clock_and_without_tony():
    from dancecap.take import Take
    take = Take.load("larissa_kadu_recap")
    ids = [s.id for s in take.sources]
    assert ids == ["00", "01", "02", "03", "05", "06"]
    assert take.source("01").slope == 1.0 and take.source("01").offset == 0.0
    assert take.window == (0.0, 360.48)
    assert take.cfg["dance"]["category"] == "lesson" and "song" not in take.cfg["dance"]
    assert take.cfg["chunks"]["chunk_s"] == 60.0 and take.cfg["chunks"]["overlap_s"] == 4.0   # [chunks] also carries mode / profile now
    types = take.camera_types()
    assert set(types) == set(ids) and all(t.mount == "handheld" and t.zoom is False and t.source == "user" for t in types.values())
    names = {s.path.name for s in take.sources}
    assert "02-Recap-Alyssa.mp4" in names and not any("Tony" in n for n in names)


@pytest.mark.parametrize("cam,expect", [("02", 1.3469), ("03", 60.9115), ("05", 7.3962), ("06", 29.3179), ("00", -8.9173)])
@needs_paths(WORKSPACE / "prototype" / "larissa_kadu" / "alignment_recap.json", why="recap alignment")
def test_real_alignment_agrees_with_the_owners_offsets_within_a_frame(cam, expect):
    doc = json.loads((WORKSPACE / "prototype" / "larissa_kadu" / "alignment_recap.json").read_text(encoding="utf-8"))
    src = next(s for s in doc["sources"] if s["id"] == cam)
    mid = 180.0
    assert abs(src["slope"] * mid + src["offset"] - mid - expect) < 0.045     # one 30 fps frame = 33 ms; sign convention confirmed
