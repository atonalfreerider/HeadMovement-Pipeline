"""Audio of the recap sources: light ffmpeg extraction on the PC (the only heavy-looking thing done locally; it is a decode).

For every source `NN-Recap-X.mp4`, into work/<take>/speech/audio/:
  NN-Recap-X.16k.wav   mono 16 kHz 16-bit  (what every ASR / VAD / embedding model here takes; uploaded to the farm volume)
  NN-Recap-X.48k.wav   mono 48 kHz 16-bit  (kept for a model that prefers it; not uploaded)
  NN-Recap-X.m4a       the source's own AAC stream, copied unchanged (stereo where the phone recorded stereo)
  audio_manifest.json  sha256, bytes, duration, sample rate / channels of the original, and a stereo-vs-dual-mono verdict

The mono downmix is the plain average of the channels (ffmpeg `-ac 1`); the phones' channels are within 1.5 dB of each other
(measured with volumedetect on the first 120 s), so nothing cancels.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import wave
from pathlib import Path

from . import SPEECH, SR, SOURCES

AUDIO = SPEECH / "audio"
NO_WINDOW = 0x08000000 if hasattr(subprocess, "STARTUPINFO") else 0


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def probe_audio(video: Path) -> dict:
    """{codec, sample_rate, channels, bit_rate, duration} of the first audio stream (ffprobe)."""
    p = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                        "stream=codec_name,sample_rate,channels,bit_rate,duration", "-of", "json", str(video)],
                       capture_output=True, text=True, creationflags=NO_WINDOW)
    if p.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {video.name}: {p.stderr.strip()[:300]}")
    s = (json.loads(p.stdout).get("streams") or [{}])[0]
    return {"codec": s.get("codec_name"), "sample_rate": int(s.get("sample_rate") or 0), "channels": int(s.get("channels") or 0),
            "bit_rate": int(s.get("bit_rate") or 0), "duration": float(s.get("duration") or 0.0)}


def wav_info(path: Path) -> dict:
    with wave.open(str(path), "rb") as w:
        return {"sample_rate": w.getframerate(), "channels": w.getnchannels(), "frames": w.getnframes(),
                "duration": round(w.getnframes() / w.getframerate(), 4)}


def _run(cmd: list[str]) -> None:
    p = subprocess.run(cmd, capture_output=True, text=True, creationflags=NO_WINDOW)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {' '.join(cmd[:6])}...: {p.stderr.strip()[-400:]}")


def channel_levels(video: Path, seconds: float = 120.0) -> dict:
    """mean / max volume (dB) of the left and right channel over the first `seconds` (ffmpeg volumedetect). Tells a stereo mic from
    dual mono and shows a dead channel."""
    out = {}
    for ch in (0, 1):
        p = subprocess.run(["ffmpeg", "-nostats", "-i", str(video), "-t", str(seconds), "-vn", "-af",
                            f"pan=mono|c0=c{ch},volumedetect", "-f", "null", "-"], capture_output=True, text=True,
                           creationflags=NO_WINDOW)
        row = {}
        for line in p.stderr.splitlines():
            for key in ("mean_volume", "max_volume"):
                if f"{key}:" in line:
                    try:
                        row[key] = float(line.split(f"{key}:")[1].split("dB")[0])
                    except ValueError:
                        pass
        out[f"ch{ch}"] = row
    try:
        out["lr_mean_diff_db"] = round(out["ch0"]["mean_volume"] - out["ch1"]["mean_volume"], 2)
        out["verdict"] = "dual mono" if abs(out["ch0"]["max_volume"] - out["ch1"]["max_volume"]) < 0.05 else "stereo"
    except KeyError:
        out["verdict"] = "unknown"
    return out


def extract_one(video: Path, out_dir: Path, *, force: bool = False, want_48k: bool = True) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = video.stem
    w16, w48, m4a = out_dir / f"{stem}.16k.wav", out_dir / f"{stem}.48k.wav", out_dir / f"{stem}.m4a"
    if force or not w16.exists():
        tmp = w16.with_name(w16.name + ".part.wav")
        _run(["ffmpeg", "-y", "-v", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le", str(tmp)])
        tmp.replace(w16)
    if want_48k and (force or not w48.exists()):
        tmp = w48.with_name(w48.name + ".part.wav")
        _run(["ffmpeg", "-y", "-v", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", "48000", "-c:a", "pcm_s16le", str(tmp)])
        tmp.replace(w48)
    if force or not m4a.exists():
        tmp = m4a.with_name(m4a.name + ".part.m4a")
        _run(["ffmpeg", "-y", "-v", "error", "-i", str(video), "-vn", "-c:a", "copy", str(tmp)])
        tmp.replace(m4a)
    src = probe_audio(video)
    row = {"file": video.name, "original": src, "channels": channel_levels(video),
           "wav16k": {"name": w16.name, "bytes": w16.stat().st_size, "sha256": sha256_file(w16), **wav_info(w16)},
           "m4a": {"name": m4a.name, "bytes": m4a.stat().st_size, "sha256": sha256_file(m4a)}}
    if w48.exists():
        row["wav48k"] = {"name": w48.name, "bytes": w48.stat().st_size, **wav_info(w48)}
    row["duration_check_s"] = round(row["wav16k"]["duration"] - src["duration"], 3)
    return row


def extract_all(source_dir: Path | str, ids: list[str] | None = None, *, force: bool = False, want_48k: bool = True,
                out_dir: Path = AUDIO) -> dict:
    """Extract every source (or `ids`); writes audio_manifest.json and returns it."""
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("ffmpeg / ffprobe are not on PATH")
    from .sources import discover
    vids = discover(source_dir)
    rows = {}
    for sid, fname in SOURCES.items():
        if ids and sid not in ids:
            continue
        v = vids.get(sid)
        if v is None:
            raise FileNotFoundError(f"{fname} not found in {source_dir}")
        rows[sid] = extract_one(v, out_dir, force=force, want_48k=want_48k)
        print(f"[audio] {fname}: {rows[sid]['wav16k']['duration']:.2f} s, {rows[sid]['original']['sample_rate']} Hz "
              f"{rows[sid]['original']['channels']} ch -> 16 kHz mono, {rows[sid]['channels']['verdict']}", flush=True)
    man_path = out_dir / "audio_manifest.json"
    man = json.loads(man_path.read_text(encoding="utf-8")) if man_path.exists() else {"sources": {}}
    man["sources"].update(rows)
    man["note"] = ("16 kHz mono wavs are the ASR inputs; the m4a files are the sources' own AAC streams (copied, not re-encoded). "
                   "No file here is ever committed or published (voices of real people).")
    man_path.write_text(json.dumps(man, indent=1), encoding="utf-8")
    return man
