"""Decode native-resolution frames with exact timestamps on the shared reference clock.

Phones record variable frame rate, so frames are never resampled here: every decoded frame keeps its
presentation timestamp (PTS), which is mapped through the camera's audio clock fit onto the reference
timeline. Downstream stages interpolate per-camera measurements to whatever grid they need.

work/<take>/frames/<cam>/000000.jpg + frames.json  (window, every frame)
work/<take>/keyframes/<cam>/<ref_ms>.jpg + keyframes.json  (sparse snapshots for the static room)
"""
from __future__ import annotations

import json
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from .take import Source, Take

SHOWINFO = re.compile(r"\bn:\s*(\d+)\s+pts:\s*(-?\d+)\s+pts_time:\s*(-?[\d.]+)")


_PASSTHROUGH: list[str] = []


def passthrough_args() -> list[str]:
    """ffmpeg's 'keep every frame, never duplicate/drop' flag: -fps_mode (ffmpeg >= 5.1) or -vsync (older, e.g. the
    apt ffmpeg 4.4 of the Ubuntu 22.04 Runpod image used by dancecap.front_full). Same frames, same PTS."""
    if not _PASSTHROUGH:
        try:
            v = subprocess.run(["ffmpeg", "-hide_banner", "-version"], capture_output=True, text=True).stdout
            m = re.search(r"ffmpeg version n?(\d+)\.(\d+)", v)
            new = m is None or (int(m.group(1)), int(m.group(2))) >= (5, 1)
        except OSError:
            new = True
        _PASSTHROUGH.extend(["-fps_mode", "passthrough"] if new else ["-vsync", "passthrough"])
    return list(_PASSTHROUGH)


def probe(path: Path) -> dict:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,time_base,avg_frame_rate,r_frame_rate:stream_side_data=rotation",
         "-of", "json", str(path)], capture_output=True, text=True)
    if r.returncode != 0:
        # older ffprobe (the apt 4.4 of the Ubuntu 22.04 Runpod image, dancecap.front_full) rejects the
        # stream_side_data section: -show_streams carries the same side_data_list (+ the legacy rotate tag)
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams", "-of", "json",
                            str(path)], capture_output=True, text=True, check=True)
    s = json.loads(r.stdout)["streams"][0]
    rotation = 0
    for sd in s.get("side_data_list", []) or []:
        if "rotation" in sd:
            rotation = int(round(float(sd["rotation"])))
    if not rotation and (s.get("tags") or {}).get("rotate"):
        rotation = -int(round(float(s["tags"]["rotate"])))  # tag 90 == display-matrix rotation -90
    w, h = int(s["width"]), int(s["height"])
    if abs(rotation) % 180 == 90:
        w, h = h, w
    num, den = (int(x) for x in s["avg_frame_rate"].split("/"))
    tb = s.get("time_base") or ""
    return {"width": w, "height": h, "rotation": rotation, "avg_fps": num / den if den else 0.0,
            **({"time_base": tb} if "/" in tb else {})}


def _old_ffmpeg() -> bool:
    """ffmpeg < 5.1 (passthrough_args decides it): its showinfo prints pts_time to 6 SIGNIFICANT digits only."""
    return "-vsync" in passthrough_args()


def showinfo_pts(m: re.Match, time_base: str | None) -> float:
    """Source PTS (s) of one showinfo line. ffmpeg >= 5.1 prints pts_time to 6 decimals and that value is kept as
    it always was; an older ffmpeg (the apt 4.4 of the Ubuntu 22.04 Runpod image) prints 6 significant digits
    (123.457 for 123.456522: up to 0.5 ms off), so there the exact integer pts x stream time base is used, rounded to
    6 decimals like the newer printout (checked against pts_time: within 1 ms, else pts_time)."""
    t = float(m.group(3))
    if time_base and _old_ffmpeg():
        num, den = (int(x) for x in time_base.split("/"))
        if den:
            exact = round(int(m.group(2)) * num / den, 6)
            if abs(exact - t) < 1e-3:
                return exact
    return t


def decode_range(src: Path, start: float, end: float, out_dir: Path, quality: int = 95,
                 every: int = 1) -> list[dict]:
    """Decode frames of `src` whose PTS lies in [start, end] (source seconds) to JPEGs.

    Returns [{"file", "pts"}] in presentation order. Uses ffmpeg autorotation; PTS come from the
    showinfo filter with -copyts so they are the source's own timestamps.
    """
    info = probe(src)
    w, h = info["width"], info["height"]
    frame_bytes = w * h * 3
    out_dir.mkdir(parents=True, exist_ok=True)
    seek = max(0.0, start - 1.0)
    decoded = 0
    with tempfile.TemporaryFile() as log:
        # -t bounds the read; frames stream to temporary JPEGs so memory stays flat at 1080p/60
        proc = subprocess.Popen(
            ["ffmpeg", "-v", "info", "-nostats", "-ss", f"{seek:.6f}", "-t", f"{end - seek + 0.5:.6f}",
             "-copyts", "-i", str(src), "-map", "0:v:0", "-vf", "showinfo", *passthrough_args(),
             "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
            stdout=subprocess.PIPE, stderr=log)
        try:
            while True:
                buf = proc.stdout.read(frame_bytes)
                if len(buf) < frame_bytes:
                    break
                img = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
                cv2.imwrite(str(out_dir / f"_tmp{decoded:06d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, quality])
                decoded += 1
        finally:
            proc.stdout.close()
            proc.wait()
        log.seek(0)
        pts = [showinfo_pts(m, info.get("time_base"))
               for m in SHOWINFO.finditer(log.read().decode("utf-8", "replace"))]

    if len(pts) < decoded:
        raise RuntimeError(f"{src.name}: {decoded} frames but only {len(pts)} showinfo timestamps")
    records = []
    kept = 0
    for i in range(decoded):
        tmp = out_dir / f"_tmp{i:06d}.jpg"
        t = pts[i]
        if start - 1e-6 <= t <= end + 1e-6 and kept % every == 0:
            name = f"{len(records):06d}.jpg"
            tmp.replace(out_dir / name)
            records.append({"file": name, "pts": t})
        else:
            tmp.unlink()
        if start - 1e-6 <= t <= end + 1e-6:
            kept += 1
    return records


def ingest_window(take: Take, cams: list[str] | None = None, workers: int = 3) -> dict:
    t0, t1 = take.window
    out_root = take.work / "frames"

    max_fps = float((take.cfg.get("ingest") or {}).get("max_fps") or 0)

    def one(s: Source) -> tuple[str, dict]:
        info = probe(s.path)
        # [ingest] max_fps (optional, e.g. 30 for larissa_kadu_full's segment takes): a 60 fps phone keeps every
        # 2nd frame (by decode order inside the window; PTS unchanged) - the fused grid is 30 Hz
        every = max(1, int(round(info["avg_fps"] / max_fps))) if max_fps and info["avg_fps"] > 1.5 * max_fps else 1
        recs = decode_range(s.path, s.to_source(t0), s.to_source(t1), out_root / s.id, every=every)
        for r in recs:
            r["t"] = s.to_reference(r["pts"])
        dts = np.diff([r["pts"] for r in recs])
        meta = {
            "camera": s.id, "source": s.path.name, "width": info["width"], "height": info["height"],
            "rotation": info["rotation"], "slope": s.slope, "offset": s.offset,
            "frame_count": len(recs), **({"decimate_every": every} if every > 1 else {}),
            "fps_median": float(1 / np.median(dts)) if len(dts) else 0.0,
            "frame_interval_ms": {"min": float(dts.min() * 1000), "max": float(dts.max() * 1000)} if len(dts) else {},
            "frames": recs,
        }
        (out_root / s.id / "frames.json").write_text(json.dumps(meta, indent=1))
        return s.id, meta

    chosen = [s for s in take.sources if cams is None or s.id in cams]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        metas = dict(pool.map(one, chosen))
    return metas


def ingest_keyframes(take: Take, cams: list[str] | None = None, workers: int = 3) -> dict:
    """One sharp-ish frame per camera every keyframe_step seconds across keyframe_span (reference time)."""
    k0, k1 = take.keyframe_span
    times = np.arange(k0, k1 + 1e-9, take.keyframe_step)
    out_root = take.work / "keyframes"

    def one(s: Source) -> tuple[str, list[dict]]:
        recs = []
        tmp = out_root / s.id / "_tmp"
        for t in times:
            st = s.to_source(float(t))
            if st < 0:
                continue
            # decode a short burst and keep the sharpest frame (least motion blur)
            burst = decode_range(s.path, st - 0.15, st + 0.15, tmp)
            if not burst:
                continue
            best = max(burst, key=lambda r: _sharpness(tmp / r["file"]))
            name = f"{int(round(s.to_reference(best['pts']) * 1000)):07d}.jpg"
            (tmp / best["file"]).replace(out_root / s.id / name)
            recs.append({"file": name, "pts": best["pts"], "t": s.to_reference(best["pts"])})
            for r in burst:
                p = tmp / r["file"]
                if p.exists():
                    p.unlink()
        if tmp.exists():
            tmp.rmdir()
        (out_root / s.id / "keyframes.json").write_text(json.dumps(recs, indent=1))
        return s.id, recs

    chosen = [s for s in take.sources if cams is None or s.id in cams]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return dict(pool.map(one, chosen))


def _sharpness(path: Path) -> float:
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return float(cv2.Laplacian(g, cv2.CV_64F).var()) if g is not None else 0.0
