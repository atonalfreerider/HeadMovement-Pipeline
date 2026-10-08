"""Media helpers of the monitor: HTTP Range responses, thumbnails, contact strips and H.264 proxies (ffmpeg).

    python -m dancecap.pipeline.monitor.media proxy <take> <cam>     work/<take>/moves/proxy/<cam>.mp4

The proxy is the narration tool's (same path, same timestamp-preserving command), so either tool can make it.
Thumbnails and strips are cached under work/monitor/thumbs/ (git-ignored). ffmpeg runs without a console window.
"""
from __future__ import annotations

import hashlib
import mimetypes
import os
import re
import shutil
import subprocess
import sys
from email.utils import formatdate
from pathlib import Path

from .. import paths

NO_WINDOW = 0x08000000 if os.name == "nt" else 0
CHUNK = 1 << 20
mimetypes.add_type("video/mp4", ".mp4")
mimetypes.add_type("audio/mp4", ".m4a")


def parse_range(header: str | None, size: int):
    """None (send everything), 'unsatisfiable', or (start, end) inclusive (RFC 9110; multi-range ignored)."""
    if not header:
        return None
    m = re.fullmatch(r"\s*bytes\s*=\s*(\d*)\s*-\s*(\d*)\s*", header)
    if not m:
        return None
    a, b = m.groups()
    if a == "" and b == "":
        return None
    if a == "":
        n = int(b)
        if n == 0 or size == 0:
            return "unsatisfiable"
        return max(0, size - n), size - 1
    start = int(a)
    if start >= size:
        return "unsatisfiable"
    end = int(b) if b else size - 1
    if end < start:
        return None
    return start, min(end, size - 1)


def file_response(path: Path, range_header: str | None, head: bool = False, mimetype: str | None = None):
    from flask import Response
    st = path.stat()
    size = st.st_size
    mime = mimetype or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    headers = {"Accept-Ranges": "bytes", "Content-Type": mime, "Last-Modified": formatdate(st.st_mtime, usegmt=True),
               "Cache-Control": "no-cache", "ETag": f'"{st.st_mtime_ns:x}-{size:x}"'}
    r = parse_range(range_header, size)
    if r == "unsatisfiable":
        headers.update({"Content-Range": f"bytes */{size}", "Content-Length": "0"})
        return Response(b"", status=416, headers=headers)
    start, end = r if r else (0, size - 1)
    length = max(0, end - start + 1)
    if r:
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    headers["Content-Length"] = str(length)
    status = 206 if r else 200
    if head:
        return Response(b"", status=status, headers=headers)

    def gen():
        with open(path, "rb") as f:
            f.seek(start)
            left = length
            while left > 0:
                chunk = f.read(min(CHUNK, left))
                if not chunk:
                    break
                left -= len(chunk)
                yield chunk

    return Response(gen(), status=status, headers=headers, direct_passthrough=True)


def _ff() -> str:
    ff = shutil.which("ffmpeg")
    if not ff:
        raise RuntimeError("ffmpeg not found on PATH")
    return ff


def _cache(key: str, ext: str = ".jpg") -> Path:
    d = paths.monitor_dir() / "thumbs"
    d.mkdir(parents=True, exist_ok=True)
    return d / (hashlib.sha1(key.encode()).hexdigest()[:20] + ext)


def thumbnail(src: Path, t: float = 1.0, width: int = 320) -> Path:
    width = max(64, min(int(width), 1280))
    t = max(0.0, float(t))
    st = src.stat()
    out = _cache(f"thumb|{src}|{st.st_size}|{st.st_mtime_ns}|{t:.2f}|{width}")
    if out.exists():
        return out
    tmp = out.with_suffix(".part.jpg")
    r = subprocess.run([_ff(), "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", str(src), "-frames:v", "1",
                        "-vf", f"scale={width}:-2", "-q:v", "4", str(tmp)], capture_output=True, text=True,
                       creationflags=NO_WINDOW, stdin=subprocess.DEVNULL, timeout=60)
    if r.returncode != 0 or not tmp.exists():
        tmp.unlink(missing_ok=True)
        raise RuntimeError("thumbnail failed")
    os.replace(tmp, out)
    return out


def strip(src: Path, duration_s: float | None, n: int = 12, height: int = 90) -> Path:
    """n frames side by side (a contact strip), sampled evenly over the video."""
    n = max(2, min(int(n), 24))
    height = max(40, min(int(height), 240))
    st = src.stat()
    out = _cache(f"strip|{src}|{st.st_size}|{st.st_mtime_ns}|{n}|{height}")
    if out.exists():
        return out
    dur = float(duration_s or 0) or 60.0
    fps = n / max(dur, 1.0)
    tmp = out.with_suffix(".part.jpg")
    r = subprocess.run([_ff(), "-v", "error", "-y", "-i", str(src), "-vf",
                        f"fps={fps:.6f},scale=-2:{height},tile={n}x1", "-frames:v", "1", "-q:v", "5", str(tmp)],
                       capture_output=True, text=True, creationflags=NO_WINDOW, stdin=subprocess.DEVNULL, timeout=300)
    if r.returncode != 0 or not tmp.exists():
        tmp.unlink(missing_ok=True)
        raise RuntimeError("strip failed")
    os.replace(tmp, out)
    return out


def proxy_path(take: str, cam: str) -> Path:
    return paths.take_work(take) / "moves" / "proxy" / f"{cam}.mp4"


def make_proxy(src: Path, out: Path, timescale: str = "90000") -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".part.mp4")
    cmd = [_ff(), "-v", "error", "-y", "-i", str(src), "-map", "0:v:0", "-map", "0:a:0?", "-c:v", "libx264",
           "-preset", "veryfast", "-crf", "24", "-vf", "scale=-2:'min(960,ih)'", "-pix_fmt", "yuv420p",
           "-fps_mode", "passthrough", "-enc_time_base", "demux", "-video_track_timescale", timescale,
           "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(tmp)]
    print(f"[media] proxy {src.name} -> {out}", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, creationflags=NO_WINDOW, stdin=subprocess.DEVNULL)
    if r.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg proxy failed: {r.stderr.strip()[-300:]}")
    os.replace(tmp, out)
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m dancecap.pipeline.monitor.media")
    ap.add_argument("action", choices=["proxy"])
    ap.add_argument("take")
    ap.add_argument("cam")
    a = ap.parse_args(argv)
    from .. import takes
    from ..util import CAM_RE
    if not CAM_RE.match(a.cam):
        print("cam must be two digits", file=sys.stderr)
        return 2
    cfg, _ = takes.load_cfg(a.take)
    v = next((x for x in takes.videos(a.take, cfg) if x["cam"] == a.cam), None)
    if not v or not v.get("path") or not Path(v["path"]).exists():
        print(f"no video for camera {a.cam}", file=sys.stderr)
        return 1
    make_proxy(Path(v["path"]), proxy_path(a.take, a.cam))
    return 0


if __name__ == "__main__":
    sys.exit(main())
