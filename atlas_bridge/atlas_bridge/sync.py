"""Audio-based synchronisation of handheld phone videos (and an optional clean music track).

Every source is placed on the *reference timeline* (the music track if given,
otherwise the reference video). `offset` = reference-timeline time at which the
source's t=0 occurs. Then all videos are cut to the common overlap window and
re-encoded at a constant frame rate - phones record variable frame rate, and
both Atlas and MAMMA index frames, not timestamps.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

SR = 16000


@dataclass
class SourceSync:
    name: str
    path: str
    offset: float  # seconds on the reference timeline where this source starts
    duration: float
    confidence: float  # peak-to-sidelobe ratio of the GCC-PHAT correlation; > ~5 is solid


def probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type,width,height,r_frame_rate,avg_frame_rate",
         "-of", "json", str(path)], capture_output=True, text=True, check=True).stdout
    info = json.loads(out)
    v = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    has_audio = any(s["codec_type"] == "audio" for s in info["streams"])

    def rate(r: str) -> float:
        n, d = r.split("/")
        return float(n) / float(d) if float(d) else 0.0

    return {
        "duration": float(info["format"]["duration"]),
        "has_audio": has_audio,
        "width": int(v["width"]) if v else None,
        "height": int(v["height"]) if v else None,
        "fps": rate(v["avg_frame_rate"]) if v else None,
        "vfr": bool(v) and v["r_frame_rate"] != v["avg_frame_rate"],
    }


def load_audio(path: Path, max_seconds: float | None = None) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error", "-i", str(path)]
    if max_seconds:
        cmd += ["-t", str(max_seconds)]
    cmd += ["-vn", "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def gcc_phat(sig: np.ndarray, ref: np.ndarray, max_lag_s: float | None = None) -> tuple[float, float]:
    """Delay (seconds) of `sig` relative to `ref`, and a confidence score.

    Returns tau such that sig(t) ~= ref(t + tau), i.e. sig starts `tau` seconds
    into ref's timeline.
    """
    n = len(sig) + len(ref)
    nfft = 1 << (n - 1).bit_length()
    S = np.fft.rfft(sig, nfft)
    R = np.fft.rfft(ref, nfft)
    X = R * np.conj(S)
    X /= np.abs(X) + 1e-12
    cc = np.fft.irfft(X, nfft)
    max_shift = nfft // 2
    if max_lag_s is not None:
        max_shift = min(max_shift, int(max_lag_s * SR))
    cc = np.concatenate((cc[-max_shift:], cc[: max_shift + 1]))
    peak = int(np.argmax(np.abs(cc)))
    lag = peak - max_shift
    mag = np.abs(cc)
    guard = int(0.05 * SR)
    side = np.concatenate((mag[: max(0, peak - guard)], mag[peak + guard:]))
    confidence = float(mag[peak] / (np.percentile(side, 99.9) + 1e-12)) if side.size else 0.0
    return lag / SR, confidence


def sync_sources(videos: dict[str, Path], music: Path | None, reference: str | None,
                 analysis_seconds: float = 90.0) -> tuple[str, list[SourceSync]]:
    """Place every video on a shared timeline. Returns (reference_name, syncs)."""
    if music is not None:
        ref_name, ref_path = "music", music
    else:
        ref_name = reference or next(iter(videos))
        ref_path = videos[ref_name]
    ref_audio = load_audio(ref_path)  # full reference: videos may start anywhere inside it

    syncs = []
    if music is None:
        syncs.append(SourceSync(ref_name, str(ref_path), 0.0, probe(ref_path)["duration"], float("inf")))
    for name, path in videos.items():
        if music is None and name == ref_name:
            continue
        info = probe(path)
        if not info["has_audio"]:
            raise RuntimeError(f"{path} has no audio track - set its offset manually in session.toml [sync.offsets]")
        audio = load_audio(path, analysis_seconds)
        offset, conf = gcc_phat(audio, ref_audio)
        syncs.append(SourceSync(name, str(path), offset, info["duration"], conf))
    return ref_name, syncs


def common_window(syncs: list[SourceSync], videos_only: set[str]) -> tuple[float, float]:
    vids = [s for s in syncs if s.name in videos_only]
    start = max(s.offset for s in vids)
    end = min(s.offset + s.duration for s in vids)
    if end <= start:
        raise RuntimeError("videos do not overlap in time")
    return start, end


def cut_constant_rate(src: Path, dst: Path, start_in_src: float, duration: float, fps: float,
                      keep_audio: bool = False) -> None:
    """Cut [start, start+duration) of src and re-encode at a constant frame rate."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-v", "error", "-ss", f"{start_in_src:.6f}", "-i", str(src),
           "-t", f"{duration:.6f}", "-vf", f"fps={fps}", "-fps_mode", "cfr",
           "-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p"]
    cmd += ["-c:a", "aac"] if keep_audio else ["-an"]
    cmd.append(str(dst))
    subprocess.run(cmd, check=True)


def extract_wav(src: Path, dst: Path, start: float = 0.0, duration: float | None = None) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.6f}", "-i", str(src)]
    if duration is not None:
        cmd += ["-t", f"{duration:.6f}"]
    cmd += ["-vn", "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le", str(dst)]
    subprocess.run(cmd, check=True)


def write_sync(path: Path, reference: str, syncs: list[SourceSync], window: tuple[float, float], fps: float,
               frame_count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "reference": reference,
        "fps": fps,
        "window": list(window),
        "frame_count": frame_count,
        "sources": [asdict(s) for s in syncs],
    }, indent=2))
