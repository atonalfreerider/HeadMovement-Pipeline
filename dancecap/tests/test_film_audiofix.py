"""dancecap.film_audiofix: the recorder's AAC burst is replaced from the film mix, the video is untouched, the raw file is kept."""
from __future__ import annotations

import json
import shutil
import subprocess

import numpy as np
import pytest
import soundfile as sf

import sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # run from the workspace root too
from dancecap import film_audiofix as fa  # noqa: E402

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="ffmpeg / ffprobe not on PATH")


def _mix(n: int, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fa.SR
    # speech-like: noise bursts under a slow envelope plus a tone (correlates well, never silent for long)
    env = 0.25 + 0.2 * np.sin(2 * np.pi * 1.3 * t)
    # band-limited noise (a 16-tap moving average: below ~3 kHz, like speech; AAC keeps it)
    noise = np.convolve(rng.standard_normal(n), np.ones(16) / 4.0, mode="same")
    x = env * (0.6 * np.sin(2 * np.pi * 220 * t) + 0.12 * noise)
    return np.stack([x, 0.9 * x], axis=1).astype(np.float32)


def test_bursts_finds_a_spike_the_mix_does_not_have():
    ref = _mix(fa.SR)
    rec = ref.copy()
    i = int(0.5 * fa.SR)
    rec[i:i + 300] += 50.0
    peak, found = fa.bursts(rec, ref)
    assert peak == pytest.approx(50.0, rel=1e-3)
    assert len(found) == 1 and found[0]["t0"] <= 0.5 < found[0]["t1"] + 1e-9
    peak, found = fa.bursts(ref + 0.01, ref)
    assert found == [] and peak == pytest.approx(0.01, abs=1e-6)


def test_sync_windows_measure_the_lag():
    ref = _mix(4 * fa.SR, seed=3)
    late = np.concatenate([np.zeros((96, 2), np.float32), ref[:-96]])   # 2 ms late
    w = fa.sync_windows(late, ref, n_windows=5)
    assert len(w) == 3   # 4 s: three 1.3 s windows
    assert all(x["lag_ms"] == pytest.approx(2.0, abs=0.03) for x in w)
    assert min(x["r"] for x in w) > 0.99
    w0 = fa.sync_windows(ref, ref, n_windows=5)
    assert all(x["lag_ms"] == 0.0 and x["r"] > 0.9999 for x in w0)


@needs_ffmpeg
def test_fix_replaces_the_audio_and_keeps_the_raw(tmp_path):
    dur = 3.0
    mix = _mix(int(10 * fa.SR))
    sf.write(tmp_path / "mix.wav", mix, fa.SR, subtype="PCM_16")
    start = 4.0
    # the "recorder's" track: the mix from film 4 s with a burst at film 5.5 s
    bad = mix[int(start * fa.SR):int((start + dur) * fa.SR)].copy()
    j = int(1.5 * fa.SR)
    bad[j:j + 600] = 0.999
    bad[j + 600:j + 700] = -0.999
    sf.write(tmp_path / "bad.wav", bad, fa.SR, subtype="FLOAT")
    video = tmp_path / "film_vertical.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc=size=64x112:rate=30:duration={dur}", "-i",
                    str(tmp_path / "bad.wav"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", str(video)],
                   check=True)
    report = tmp_path / "film_vertical.report.json"
    report.write_text(json.dumps({"kind": "film_render_report", "pass": True}), encoding="utf-8")
    (tmp_path / "film_vertical.report.md").write_text("# Film render report\n", encoding="utf-8")

    res = fa.fix(video, tmp_path / "mix.wav", start=start, report=report, log=lambda *_: None)

    assert res["pass"], [c for c in res["checks"] if not c["pass"]]
    raw = tmp_path / "film_vertical_rawaudio.mp4"
    assert raw.exists() and video.exists()
    assert res["video_md5"]["raw"] == res["video_md5"]["new"]
    assert res["residual"]["new_bursts"] == []
    rb = res["residual"]["raw_bursts"]
    assert len(rb) == 1 and rb[0]["film_t0"] == pytest.approx(5.5, abs=0.02)
    assert all(abs(w["lag_ms"]) <= fa.SYNC_TOL_MS for w in res["sync"]["windows"])
    assert json.loads(report.read_text(encoding="utf-8"))["audio_fix"]["pass"] is True
    assert "## Audio track replaced" in (tmp_path / "film_vertical.report.md").read_text(encoding="utf-8")
    assert json.loads((tmp_path / "film_vertical.audiofix.json").read_text(encoding="utf-8"))["raw"] == str(raw)

    # a rerun reads the kept raw file and writes the same result (the md section is replaced, not repeated)
    res2 = fa.fix(video, tmp_path / "mix.wav", start=start, report=report, log=lambda *_: None)
    assert res2["pass"] and raw.exists()
    assert (tmp_path / "film_vertical.report.md").read_text(encoding="utf-8").count("## Audio track replaced") == 1
