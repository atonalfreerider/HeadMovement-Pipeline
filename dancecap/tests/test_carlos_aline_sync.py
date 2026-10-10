"""Unit tests for dancecap.carlos_aline_sync (synthetic audio-like features; no footage, no GPU, no pod) and the take config.

    .venv-nopo4d\\Scripts\\python.exe -m pytest dancecap/tests/test_carlos_aline_sync.py -q      (from the workspace root)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dancecap import carlos_aline_sync as cs  # noqa: E402
from dancecap.take import PACKAGE_ROOT, WORKSPACE, Take  # noqa: E402

HOP = cs.HOP_S


def synthetic_song(seconds: float = 120.0, bands: int = 6, seed: int = 3) -> np.ndarray:
    """Sparse random onset features at 10 ms (a rhythm that is not periodic, so the lag is unique)."""
    rng = np.random.default_rng(seed)
    n = int(seconds / HOP)
    F = np.zeros((n, bands), np.float32)
    for j in range(bands):
        k = rng.choice(n, size=int(n * 0.03), replace=False)
        F[k, j] = rng.uniform(1.0, 4.0, len(k))
    k = np.ones(5) / 5
    F = np.stack([np.convolve(F[:, j], k, "same") for j in range(bands)], 1)
    return (F / F.std(0, keepdims=True)).astype(np.float32)


def stretched(song: np.ndarray, a: float, b_s: float, seconds: float, noise: float = 0.3, seed: int = 4) -> np.ndarray:
    """What the video hears: song_t = a * video_t + b  ->  video frame at time u is the song at a * u + b (plus noise)."""
    rng = np.random.default_rng(seed)
    u = np.arange(int(seconds / HOP)) * HOP
    pos = (a * u + b_s) / HOP
    ok = (pos >= 0) & (pos < len(song) - 1)
    out = np.zeros((len(u), song.shape[1]), np.float32)
    i0 = np.floor(pos[ok]).astype(int)
    w = (pos[ok] - i0)[:, None]
    out[ok] = song[i0] * (1 - w) + song[i0 + 1] * w
    return out + noise * rng.standard_normal(out.shape).astype(np.float32)


# ------------------------------------------------------------------------------------------------ clocks


def test_affine_invert_and_compose_roundtrip():
    s, o = 0.9615, -12.8
    si, oi = cs.affine_invert(s, o)
    assert si * (s * 50.0 + o) + oi == pytest.approx(50.0)
    c = cs.affine_compose((s, o), (1.0, 14.8158))        # song_t of a cam-02 time
    assert c[0] * 10.0 + c[1] == pytest.approx(s * (10.0 + 14.8158) + o)


def test_lattice_covers_the_span_on_the_30_hz_grid():
    lat = cs.lattice(3.6, 198.0)
    assert lat["hz"] == 30.0 and lat["anchor_s"] == 0.0
    assert lat["t_first_s"] >= 3.6 - 1e-9 and lat["t_first_s"] - 3.6 < 1 / 30 + 1e-9
    assert lat["t_last_s"] <= 198.0 + 1e-9 and 198.0 - lat["t_last_s"] < 1 / 30 + 1e-9
    assert lat["frames"] == lat["k1"] - lat["k0"] + 1
    assert lat["k0"] == 108 and lat["k1"] == 5940          # 3.6 s and 198.0 s are on the lattice
    shifted = cs.lattice(1.01, 2.0, hz=30.0, anchor_s=0.5)
    assert (shifted["t_first_s"] - 0.5) * 30 == pytest.approx(round((shifted["t_first_s"] - 0.5) * 30))


def test_song_start_in_reference():
    assert cs.song_start_in_reference(0.9615, -12.8) == pytest.approx(12.8 / 0.9615)
    assert cs.song_start_in_reference(0.942, -106.85, 10.0) == pytest.approx((10.0 + 106.85) / 0.942)


# ------------------------------------------------------------------------------------------------ features


def test_warp_features_stretches_time():
    F = np.zeros((1000, 2), np.float32)
    F[500, 0] = 1.0
    w = cs.warp_features(F, 0.5)                        # song slower by 2: the onset at 5 s of the video sits at song 2.5 s
    assert len(w) == 500 and int(np.argmax(w[:, 0])) == 250
    w2 = cs.warp_features(F, 2.0)
    assert len(w2) == 2000 and int(np.argmax(w2[:, 0])) == 1000


def test_log_mel_and_flux_shapes_and_onset():
    sr = cs.SR
    x = np.zeros(sr * 3, np.float32)
    x[sr + 800:sr + 800 + 1600] = np.sin(2 * np.pi * 1000 * np.arange(1600) / sr).astype(np.float32)   # a 0.1 s tone at 1.05 s
    L = cs.log_mel(x)
    assert L.shape[1] == cs.NMEL and abs(len(L) - ((len(x) - cs.NFFT_FEAT) // 160 + 1)) == 0
    F = cs.flux_features(L)
    assert F.shape == L.shape
    t_on = int(np.argmax(F.sum(1))) * HOP
    assert 0.95 < t_on < 1.15


def test_feature_index_locates_a_window():
    song = synthetic_song(60.0)
    idx = cs.FeatureIndex(song, 1500)
    win = song[int(20.0 / HOP):int(30.0 / HOP)] + 0.2 * np.random.default_rng(1).standard_normal((1000, song.shape[1])).astype(np.float32)
    r = idx.locate(win)
    assert r["song_t0_s"] == pytest.approx(20.0, abs=0.02)
    assert r["z"] > 8
    r2 = idx.locate(win, 40.0, 50.0)                    # the true place is outside the allowed range: low score at a wrong place
    assert abs(r2["song_t0_s"] - 20.0) > 5 and r2["z"] < r["z"]


def test_window_rows_and_song_clock_recover_tempo_and_offset():
    a, b = 0.9615, -12.8
    song = synthetic_song(120.0)
    video = stretched(song, a, b, seconds=100.0)
    idx = cs.FeatureIndex(song, 1500)
    starts = np.arange(14.0, 100.0 - cs.WIN_S, cs.WIN_HOP_S)       # the song starts at video time 13.3 s
    rows = cs.window_rows(video, idx, a, starts, cs.WIN_S)
    fit = cs.fit_song_clock(rows, z_min=6.0)
    assert fit["ok"]
    assert fit["slope"] == pytest.approx(a, abs=3e-4)
    assert fit["offset_s"] == pytest.approx(b, abs=0.06)
    assert fit["residual_rms_ms"] < 30
    # a wrong tempo guess (0.2 % off) still locks and the fit corrects it
    rows2 = cs.window_rows(video, idx, a * 1.002, starts, cs.WIN_S)
    fit2 = cs.fit_song_clock(rows2, z_min=6.0)
    assert fit2["ok"] and fit2["slope"] == pytest.approx(a, abs=1e-3)
    # restricted to the line, the rows keep their place
    line = (fit["slope"], fit["offset_s"])
    fine = cs.window_rows(video, idx, fit["slope"], np.arange(20.0, 60.0, 1.5), cs.FINE_WIN_S, line=line)
    assert fine and all(abs(r["song_c"] - (line[0] * r["t_cam"] + line[1])) <= cs.FINE_SEARCH_S + 1e-6 for r in fine)


def test_fit_song_clock_needs_enough_confident_windows():
    rows = [{"u0": float(i), "t_cam": i + 6.0, "song_c": 0.96 * (i + 6.0), "z": 2.0, "ncc": 0.1} for i in range(10)]
    assert not cs.fit_song_clock(rows)["ok"]


def test_coarse_tempo_finds_the_ratio():
    song = synthetic_song(120.0, bands=1)
    a = 0.942
    video = stretched(song, a, -30.0, seconds=100.0, noise=0.2)
    res = cs.coarse_tempo(video, song, win_s=20.0, hop_s=20.0, t_range=(40.0, 100.0))
    assert res
    best = max(res, key=lambda r: r["z"])
    assert best["a"] == pytest.approx(a, abs=0.003)


# ------------------------------------------------------------------------------------------------ timeline


def test_audible_runs_use_window_centres_join_close_runs_and_drop_short_ones():
    rows = [{"u0": u, "win": 6.0, "t_cam": u + 3.0, "z": z} for u, z in
            [(0.0, 3.0), (1.5, 3.0), (3.0, 9.0), (4.5, 9.0), (6.0, 9.0), (7.5, 2.0), (9.0, 9.0), (10.5, 9.0), (30.0, 9.0), (60.0, 9.0), (61.5, 9.0), (63.0, 9.0), (64.5, 9.0)]]
    runs = cs.audible_runs(rows)
    assert len(runs) == 2                                  # the lone window at 30 s (1.5 s of core) is dropped, the hole at 7.5 s joined
    assert runs[0] == pytest.approx([5.25, 14.25]) and runs[1] == pytest.approx([62.25, 68.25])
    assert len(cs.audible_runs(rows, min_len_s=0.0)) == 3
    assert cs.audible_runs([{"u0": 0.0, "t_cam": 3.0, "z": 1.0}]) == []


def test_snap_to_song_start():
    assert cs.snap_to_song_start(17.0, 13.31) == 13.31      # first detection 3.7 s after the song's own start
    assert cs.snap_to_song_start(30.0, 13.31) == 30.0       # too late: the song was not audible from its start
    assert cs.snap_to_song_start(10.0, 13.31) == 10.0       # earlier than the song can start: keep the detection


def test_timeline_marks_the_mix():
    tl = cs.timeline({"kojo": (13.0, 125.0), "guetta": (113.0, 197.0)}, 202.0)
    assert [s["state"] for s in tl] == ["none", "kojo", "mix", "guetta", "none"]
    assert tl[2]["from_s"] == 113.0 and tl[2]["to_s"] == 125.0 and tl[2]["songs"] == ["kojo", "guetta"]
    assert tl[0]["from_s"] == 0.0 and tl[-1]["to_s"] == 202.0
    assert all(abs(a["to_s"] - b["from_s"]) < 1e-9 for a, b in zip(tl[:-1], tl[1:]))
    assert [s["state"] for s in cs.timeline({"kojo": None, "guetta": None}, 10.0)] == ["none"]


# ------------------------------------------------------------------------------------------------ alignment / take config


def _sync_stub() -> dict:
    return {"reference": {"name": cs.CAM_IDS["01"]}, "convention": "reference_time = slope * source_time + offset",
            "cameras": [{"id": "01", "name": cs.CAM_IDS["01"], "duration_s": 202.08, "slope": 1.0, "offset": 0.0, "residual_ms": 0.0,
                         "confidence": 1.0, "offset_source": "master"},
                        {"id": "02", "name": cs.CAM_IDS["02"], "duration_s": 186.27, "slope": 1.0, "offset": 14.8158, "residual_ms": 0.45,
                         "confidence": 1.0, "offset_source": "measured"}],
            "songs": {"kojo": {"found": True, "tempo_ratio_song_per_video": 0.96, "audible_reference_s": [13.3, 125.0], "song_zero_at_reference_s": 13.3,
                               "clock": {"slope": 0.96, "offset_s": -12.8, "residual_rms_ms": 2.0}},
                      "guetta": {"found": False}},
            "danced_span": dict(cs.DANCE_SPAN), "lattice": cs.lattice(3.6, 198.0)}


def test_alignment_doc_is_readable_by_take_load(tmp_path):
    doc = cs.alignment_doc(_sync_stub())
    assert [s["id"] for s in doc["sources"]] == ["01", "02"]
    assert doc["sources"][1]["offset"] == pytest.approx(14.8158)
    assert doc["songs"]["kojo"]["slope"] == 0.96 and doc["songs"]["guetta"] == {"found": False}
    assert "C:/" not in json.dumps(doc) and "\\Users" not in json.dumps(doc)
    toml = tmp_path / "t.toml"
    (tmp_path / "al.json").write_text(json.dumps(doc), encoding="utf-8")
    toml.write_text('name = "t"\nsource_dir = "src"\nalignment = "al.json"\nwindow = [3.6, 198.0]\n', encoding="utf-8")
    t = Take.load(toml)
    assert [s.id for s in t.sources] == ["01", "02"]
    assert t.source("02").to_reference(0.0) == pytest.approx(14.8158)


def test_take_config_and_alignment_agree_with_the_module():
    cfg = PACKAGE_ROOT / "takes" / "carlos_aline.toml"
    al = WORKSPACE / "prototype" / "larissa_kadu" / "alignment_carlos_aline.json"
    if not (cfg.exists() and al.exists()):
        pytest.skip("take config / alignment not present")
    t = Take.load(cfg)
    assert t.name == "carlos_aline"
    assert t.window == (cs.DANCE_SPAN["start_s"], cs.DANCE_SPAN["end_s"])
    assert sorted(s.id for s in t.sources) == ["01", "02"]
    assert t.source("01").slope == 1.0 and t.source("01").offset == 0.0
    assert t.source("02").offset == pytest.approx(14.8158, abs=0.01)
    assert t.cfg["sync"]["required"] is True
    assert t.cfg["dance"]["level"] == "professional" and t.cfg["dance"]["category"] == "jack_and_jill"
    assert t.cfg["chunks"]["chunk_s"] == 60.0 and t.cfg["chunks"]["overlap_s"] == 4.0
    d = t.cfg["dancers"]
    assert d["lead"]["name"] == "Carlos" and d["follow"]["name"] == "Aline"
    assert d["lead"]["stature_m"] == pytest.approx(1.727) and d["follow"]["stature_m"] == pytest.approx(1.676)
    text = cfg.read_text(encoding="utf-8")
    assert "C:/Users" not in text and "C:\\Users" not in text
    ct = t.camera_types()
    assert ct["01"].mount == "tripod" and ct["02"].mount == "tripod"
