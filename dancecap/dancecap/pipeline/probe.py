"""Stage `probe` (local CPU, seconds per video): ffprobe facts per source video + the camera-type suggestion.

    python -m dancecap.pipeline probe <take> [--cams 01,02] [--no-suggest]

Writes work/<take>/probe/probe.json ({cameras: {cam: Probe}}) and, when the shared camera-types module provides a
suggestion function (probe A), probe/camera_types_suggested.json plus the `suggested` keys in the take TOML (the user
always wins; a confirmed type is never changed).

Probe: duration_s, fps (median of the packet intervals over the first 10 s; never assumed constant), fps_nominal,
vfr, width, height, rotation (display matrix; applied at ingest), display_w/h, codec, audio (+ codec, rate, channels),
size_bytes, nb_frames, playable (Edge decodes H.264/VP9/AV1, not HEVC), problems.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

from . import paths
from .util import write_json_atomic

PLAYABLE = {"h264", "vp8", "vp9", "av1"}
NO_WINDOW = 0x08000000 if os.name == "nt" else 0


def _run(argv: list[str], timeout: float = 60) -> str:
    p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, creationflags=NO_WINDOW,
                       stdin=subprocess.DEVNULL, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        raise RuntimeError((p.stderr or p.stdout or "ffprobe failed").strip()[:300])
    return p.stdout


def _ratio(s: str | None) -> float | None:
    if not s or s in ("0/0", "N/A"):
        return None
    try:
        if "/" in s:
            a, b = s.split("/")
            return float(a) / float(b) if float(b) else None
        return float(s)
    except ValueError:
        return None


def probe_file(path: str | Path, packets_s: float = 10.0) -> dict:
    """ffprobe one media file. Raises RuntimeError when ffprobe cannot read it."""
    path = Path(path)
    out = json.loads(_run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)]))
    streams = out.get("streams", []) or []
    fmt = out.get("format", {}) or {}
    v = next((s for s in streams if s.get("codec_type") == "video" and not (s.get("disposition") or {}).get("attached_pic")), None)
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    res: dict = {"size_bytes": int(fmt.get("size") or path.stat().st_size), "duration_s": None, "problems": [],
                 "audio": a is not None}
    dur = _ratio(fmt.get("duration"))
    if a is not None:
        res.update({"audio_codec": a.get("codec_name"), "sample_rate": int(a.get("sample_rate") or 0) or None,
                    "channels": a.get("channels")})
    if v is None:
        res["duration_s"] = dur
        res["kind"] = "audio" if a is not None else "unknown"
        if a is None:
            res["problems"].append("no audio or video stream")
        return res
    res["kind"] = "video"
    rot = 0
    for sd in v.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(round(float(sd["rotation"])))
    if not rot and (v.get("tags") or {}).get("rotate"):
        rot = int((v.get("tags") or {}).get("rotate") or 0)
    rot = rot % 360
    w, h = int(v.get("width") or 0), int(v.get("height") or 0)
    dw, dh = (h, w) if rot in (90, 270) else (w, h)
    r_fps, avg_fps = _ratio(v.get("r_frame_rate")), _ratio(v.get("avg_frame_rate"))
    vdur = _ratio(v.get("duration")) or dur
    res.update({"codec": v.get("codec_name"), "width": w, "height": h, "rotation": rot, "display_w": dw,
                "display_h": dh, "duration_s": round(vdur, 4) if vdur else None, "fps_nominal": r_fps,
                "fps_avg": avg_fps, "nb_frames": int(v["nb_frames"]) if str(v.get("nb_frames", "")).isdigit() else None,
                "pix_fmt": v.get("pix_fmt"), "playable": (v.get("codec_name") or "").lower() in PLAYABLE})
    # median frame interval from the packet timestamps of the first seconds (VFR phones: never assume constant fps)
    fps_med, vfr = avg_fps or r_fps, False
    try:
        txt = _run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-read_intervals", f"%+{packets_s:g}",
                    "-show_entries", "packet=pts_time", "-of", "csv=p=0", str(path)], timeout=60)
        ts = sorted(float(x) for x in txt.split() if x.strip() and x.strip() != "N/A")
        dts = [b - a for a, b in zip(ts, ts[1:]) if b > a]
        if len(dts) >= 10:
            med = statistics.median(dts)
            fps_med = 1.0 / med
            q = sorted(dts)
            p5, p95 = q[int(0.05 * (len(q) - 1))], q[int(0.95 * (len(q) - 1))]
            vfr = p95 > 1.2 * p5
    except (RuntimeError, ValueError, subprocess.TimeoutExpired):
        pass
    if r_fps and avg_fps and abs(r_fps - avg_fps) / r_fps > 0.005:
        vfr = True
    res["fps"] = round(fps_med, 3) if fps_med else None
    res["vfr"] = bool(vfr)
    if not res["duration_s"]:
        res["problems"].append("unknown duration")
    if not a:
        res["problems"].append("no audio: this camera cannot join the audio clocks (it will be excluded)")
    if not res["playable"]:
        res["problems"].append(f"{res['codec']}: not playable in the monitor window (make a proxy)")
    return res


def check_video(pr: dict, min_s: float = 5.0) -> list[str]:
    """Hard problems (block take creation) vs the soft ones already listed in pr['problems']."""
    hard = []
    if pr.get("kind") != "video":
        hard.append("not a video")
    elif not pr.get("duration_s") or pr["duration_s"] < min_s:
        hard.append(f"too short ({pr.get('duration_s') or 0:.1f} s < {min_s:g} s)")
    if pr.get("kind") == "video" and pr.get("fps") and not (10 <= pr["fps"] <= 240):
        hard.append(f"implausible frame rate {pr['fps']}")
    return hard


def run(take: str, cams: list[str] | None = None, suggest: bool = True) -> dict:
    from . import camtypes, takes
    cfg, _ = takes.load_cfg(take)
    vids = [v for v in takes.videos(take, cfg) if not cams or v["cam"] in cams]
    out_dir = paths.take_work(take) / "probe"
    prev = (json.loads((out_dir / "probe.json").read_text(encoding="utf-8")) if (out_dir / "probe.json").exists()
            else {}).get("cameras", {})
    res, sugg = dict(prev), {}
    for v in vids:
        p = v.get("path")
        t0 = time.time()
        if not p or not Path(p).exists():
            res[v["cam"]] = {"error": "file not found", "file": v.get("file")}
            print(f"[probe] {v['cam']}: file not found ({v.get('file')})", flush=True)
            continue
        try:
            pr = probe_file(p)
        except Exception as e:  # noqa: BLE001 - one bad file never stops the others
            res[v["cam"]] = {"error": str(e)[:300], "file": v.get("file")}
            print(f"[probe] {v['cam']}: {e}", flush=True)
            continue
        pr["file"] = v.get("file")
        res[v["cam"]] = pr
        print(f"[probe] {v['cam']}: {pr.get('display_w')}x{pr.get('display_h')} {pr.get('fps')} fps"
              f"{' VFR' if pr.get('vfr') else ''} {pr.get('duration_s')} s audio={pr.get('audio')} "
              f"{pr.get('codec')} ({time.time() - t0:.1f}s)", flush=True)
        if suggest and not v["excluded"]:
            s = camtypes.suggest_for_video(Path(p), pr)
            if s:
                sugg[v["cam"]] = s
    write_json_atomic(out_dir / "probe.json", {"version": 1, "take": take, "at": time.time(), "cameras": res})
    if sugg:
        allp = out_dir / "camera_types_suggested.json"
        old = (json.loads(allp.read_text(encoding="utf-8")) if allp.exists() else {}).get("cameras", {})
        old.update(sugg)
        write_json_atomic(allp, {"version": 1, "take": take, "by": "probe_a", "cameras": old})
        for cam, s in sugg.items():
            try:
                camtypes.write_suggestion(take, cam, s)
            except Exception as e:  # noqa: BLE001
                print(f"[probe] {cam}: suggestion not written to the TOML ({e})", flush=True)
    else:
        print("[probe] camera-type suggestion: not available (dancecap.camera_types provides it)", flush=True)
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.pipeline probe")
    ap.add_argument("take")
    ap.add_argument("--cams")
    ap.add_argument("--no-suggest", action="store_true")
    a = ap.parse_args(argv)
    res = run(a.take, a.cams.split(",") if a.cams else None, suggest=not a.no_suggest)
    bad = [c for c, r in res.items() if r.get("error")]
    return 1 if bad and len(bad) == len(res) else 0


if __name__ == "__main__":
    sys.exit(main())
