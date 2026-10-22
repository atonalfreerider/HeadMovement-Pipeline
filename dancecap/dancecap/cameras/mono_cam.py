"""Camera solve of ONE handheld phone video of a couple dancing (unknown phone, unknown room): per-frame rotation + zoom about a fixed
centre, gravity, metric scale and the take's world frame -> work/<take>/cameras/01.npz (the generic successor of dancecap.room_single_cam).

    python -m dancecap.cameras.mono_cam probe    --take T --video V [--work-root DIR]       ffprobe + OpenCV decode check, every frame's pts
    python -m dancecap.cameras.mono_cam skeleton --take T --video V [--yolo PATH]           (GPU) every round(fps/2)-th frame, free focal
    python -m dancecap.cameras.mono_cam track    --take T --video V [--shards 2]            (GPU) every frame against the skeleton, matches kept
    python -m dancecap.cameras.mono_cam moge     --take T --video V [--install-moge]        (GPU) MoGe-2 on ~14 wide skeleton frames
    python -m dancecap.cameras.mono_cam pod      --take T --video V ...                     = probe, skeleton, track, moge (one call on the pod)
    python -m dancecap.cameras.mono_cam world    --take T --video V [--f1x PX] [--from-j2 DIR]   (CPU, after the people stage) -> cameras/

The pod part writes work/<take>/camera_work/ (probe.json, video_pts.json, skel/, track_s<k>.npz + _matches.npz, geom/moge.npz, mono_cam.json);
`world` reads it with the take TOML's dancers and work/<take>/people/teachers_01.npz and writes cameras/ (01.npz, 01_world.npz, rig.json,
world_from_sfm.json, recap_cams.json) and review/camera/contact.jpg. The video path is explicit (on a pod it sits at a mapped path).

Model (measured on john_lula, work/john_lula/NOTES_J2.md): the phone is held at one place and rotated (pan / tilt / roll + zoom about a fixed
centre; a few mm of parallax over 88 s). Hence
  rotation + zoom  the static background (people masked by YOLO11-seg) gives every frame's R_i, f_i (dancecap.cameras.ptz: a skeleton of every
                   ~0.5 s frame, solved jointly, then every frame against it, ~1 px). No depth enters: the rotation-only cost is FLAT in the
                   focal, so the focal is a hypothesis;
  focal            MoGe-2's focal on the WIDE frames of the skeleton (its 2x estimates are biased low: john_lula 0.54-0.64 x the track's): the
                   median per-frame ratio s = f_moge / f_skeleton re-solves the skeleton with every focal fixed at s x f_skeleton (zoom ratios
                   kept, lens k1 free), then every frame is re-solved from the stored matches on the CPU. Distances scale with the focal; the
                   camera height does not;
  gravity          MoGe-2's floor plane (RANSAC, lower image half) of each wide frame, re-expressed for the track's focal at the same horizon row
                   and carried to the reference frame (frame 0) by the track's rotation: one up vector, with its spread;
  metric scale     the two dancers' statures: the camera height from the head / feet rays of every upright dancer (eye 0.936 x stature + sole,
                   COCO ankle 0.039 x stature + sole; dancecap.cameras.single_view), median of the two dancers' medians (wide state preferred);
  world            right-handed, Y up, metres, floor y = 0, origin = the trimmed mean of both dancers' lower-ankle floor points (wide state,
                   4-MAD trim), Z = the horizontal direction from that origin towards the camera (the camera looks along about -Z), X = Y x Z.
                   The camera centre is therefore (0, height, distance). No room lines are used (optional later).
Bad frames (residual >= 4.5 px at a 960 px long side or < 60 matches) are replaced by the slerp of their ok neighbours within 1 s.
The pure math (world_solve, camera height, focal ratio, frame checks) takes arrays and is unit-tested (tests/test_mono_cam.py, synthetic).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import ptz as P
from . import ptz_run as PR
from . import single_view as SV
from ..take import PACKAGE_ROOT

CAM = "01"
ZOOM_RATIO_SPLIT = 1.3                 # a frame is in the tight state when its focal is above 1.3 x the 1x focal
SKEL_F0_FRACTIONS = (0.45, 0.65, 0.95, 1.35, 1.9)   # initial focals of the skeleton solve, x the long side of the frame
MOGE_N_FRAMES = 14
MOGE_WIDE_RATIO = 1.15                 # MoGe frames: skeleton focal within 15 % of the minimum (the wide state)
OK_RMS_PX_960 = 4.5                    # frame residual limit at a 960 px long side (scaled with the frame)
OK_MIN_MATCHES = 60
MIN_UPRIGHT = 30                       # fewer upright dancer samples than this: every dancer sample is used (rig.json says so)
MIN_ROLE = 10
UPRIGHT_DEG = 7.0
MIN_DET_CONF = 0.5
ANKLE_OVER_STATURE = 0.039             # COCO ankle point above the sole (dancecap.cameras.world)
SOLE_ASSUMED_M = 0.025
H_RANGE_M = (0.4, 3.5)                 # plausible camera heights of one sample
MOGE_COMMIT = "07444410f1e33f402353b99d6ccd26bd31e469e8"       # the pins of dancecap.cameras.single_cam_pod (MoGe-2 before the V3 rewrite)
UTILS3D_COMMIT = "3fab839f0be9931dac7c8488eb0e1600c236e183"
SHARD_RE = re.compile(r"track_s(\d+)\.npz$")


def log(msg: str) -> None:
    print(f"[mono_cam {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def take_dir(take: str, work_root: str | Path | None = None) -> Path:
    return Path(work_root) / take if work_root else PACKAGE_ROOT / "work" / take


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, Path):
        return str(o)
    return o


def record(cw: Path, name: str, seconds: float, /, **numbers) -> None:
    """camera_work/mono_cam.json: seconds and key numbers of every subcommand (merged, latest run wins). Positional-only
    name: the skeleton's numbers carry a "step" (the first pod run failed on `record() got multiple values for 'step'`)."""
    p = cw / "mono_cam.json"
    d = json.loads(p.read_text()) if p.exists() else {"schema": "dancecap.mono_cam/1", "steps": {}}
    d["steps"][name] = {"seconds": round(float(seconds), 1), "at": time.strftime("%Y-%m-%dT%H:%M:%S"), **_jsonable(numbers)}
    cw.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, indent=1))


# ------------------------------------------------------------------------------------------------ frame checks (pure)


def size_problems(stream: dict, decoded: dict) -> list[str]:
    """Why the frame size the solve would use differs from what OpenCV decodes (ptz_feats.read_video_frames). ptz_run.skeleton / track
    size the lens (principal point) from ffprobe's CODED size; the people stage works in DISPLAY pixels (rotation applied)."""
    out = []
    dec = (int(decoded["width"]), int(decoded["height"]))
    coded = (int(stream["coded_width"]), int(stream["coded_height"]))
    disp = (int(stream["width_display"]), int(stream["height_display"]))
    rot = int(stream.get("rotation", 0))
    if dec != coded:
        out.append(f"OpenCV decodes {dec[0]}x{dec[1]} but the stream is coded {coded[0]}x{coded[1]} (rotation {rot}): ptz_run sizes the lens "
                   "from the coded size, the principal point would be wrong")
    if dec != disp:
        out.append(f"OpenCV decodes {dec[0]}x{dec[1]} but the display size is {disp[0]}x{disp[1]} (rotation {rot}): the people stage's DISPLAY "
                   "pixels would not match the frames")
    if rot % 360 and decoded.get("orientation_auto") is not None and not decoded["orientation_auto"]:
        out.append(f"the stream carries rotation {rot} and OpenCV does not apply it: the frames are not upright")
    return out


def check_frames(n_pts: int, n_decoded: int | None = None, track_frames: np.ndarray | None = None) -> None:
    """The camera file has one row per decoded frame = per presentation time: raise ValueError on any mismatch."""
    if n_pts <= 0:
        raise ValueError("the video has no frames (empty pts table)")
    if n_decoded is not None and int(n_decoded) != int(n_pts):
        raise ValueError(f"OpenCV decodes {n_decoded} frames but the stream has {n_pts} presentation times: frame k of the track would not be "
                         "pts[k]; re-mux / re-encode the video (constant frame count) before the camera solve")
    if track_frames is not None:
        u = np.unique(np.asarray(track_frames, int))
        missing = np.setdiff1d(np.arange(n_pts), u)
        extra = u[(u < 0) | (u >= n_pts)]
        if len(missing) or len(extra):
            raise ValueError(f"the track covers {len(u)} frames, the video has {n_pts}: missing {missing[:8].tolist()}"
                             f"{' ...' if len(missing) > 8 else ''}, outside {extra[:8].tolist()}")


# ------------------------------------------------------------------------------------------------ pod: probe


def _ffprobe_stream(video: Path) -> dict:
    """Coded size, rates, packet count (ptz_run.probe) + display size and rotation (ingest.probe) + duration."""
    from ..ingest import probe as display_probe
    c = PR.probe(str(video))
    d = display_probe(Path(video))
    o = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=duration:format=duration", "-of", "json",
                        str(video)], capture_output=True, text=True, check=True).stdout
    j = json.loads(o)
    dur = (j.get("streams") or [{}])[0].get("duration") or (j.get("format") or {}).get("duration")
    return {"coded_width": c["width"], "coded_height": c["height"], "fps": c["fps"], "frames_packets": c["frames"],
            "width_display": d["width"], "height_display": d["height"], "rotation": int(d["rotation"]), "avg_fps": d["avg_fps"],
            "duration_s": float(dur) if dur not in (None, "N/A") else None}


def _decode_count(video: Path) -> dict:
    """What OpenCV (ptz_feats.read_video_frames) decodes: the first frame's size and the number of frames (grab loop, as read_video_frames)."""
    import cv2
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(video)
    try:
        auto = cap.get(cv2.CAP_PROP_ORIENTATION_AUTO) if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO") else -1
        meta = cap.get(cv2.CAP_PROP_ORIENTATION_META) if hasattr(cv2, "CAP_PROP_ORIENTATION_META") else 0
        ok, im = cap.read()
        if not ok:
            raise RuntimeError(f"OpenCV cannot decode the first frame of {video}")
        n = 1
        while cap.grab():
            n += 1
    finally:
        cap.release()
    return {"width": int(im.shape[1]), "height": int(im.shape[0]), "frames": n,
            "orientation_auto": None if auto < 0 else bool(auto), "orientation_meta": float(meta)}


def stage_probe(video: Path, cw: Path) -> dict:
    from .newcam import video_pts
    t0 = time.time()
    video = Path(video)
    if not video.exists():
        raise FileNotFoundError(video)
    cw.mkdir(parents=True, exist_ok=True)
    st = _ffprobe_stream(video)
    dec = _decode_count(video)
    probs = size_problems(st, dec)
    if probs:
        raise SystemExit("[mono_cam] frame size problem: " + "; ".join(probs) + ". Make an upright copy first (ffmpeg applies the rotation and "
                         "drops the tag): ffmpeg -i IN.mp4 -map 0:v:0 -map 0:a? -c:v libx264 -crf 12 -preset medium -c:a copy -fps_mode "
                         "passthrough UPRIGHT.mp4, and run every stage (people too) on it")
    pts = video_pts(video, cw / "video_pts.json")
    check_frames(len(pts), dec["frames"])
    pr = {"video": video.name, "size_bytes": video.stat().st_size, "width": dec["width"], "height": dec["height"], **st,
          "frames": int(len(pts)), "frames_decoded": dec["frames"], "pts0": float(pts[0]), "pts_last": float(pts[-1]),
          "orientation_auto": dec["orientation_auto"], "decoded_by": "OpenCV (ptz_feats.read_video_frames)"}
    (cw / "probe.json").write_text(json.dumps(_jsonable(pr), indent=1))
    dt = np.diff(pts)
    nums = {"width": pr["width"], "height": pr["height"], "rotation": pr["rotation"], "fps": pr["fps"], "frames": pr["frames"],
            "frame_interval_ms": [float(dt.min() * 1e3), float(np.median(dt) * 1e3), float(dt.max() * 1e3)] if len(dt) else []}
    record(cw, "probe", time.time() - t0, **nums)
    log(f"probe: {json.dumps(_jsonable(nums))}")
    return pr


def load_probe(cw: Path) -> dict:
    p = cw / "probe.json"
    if not p.exists():
        raise SystemExit(f"[mono_cam] {p} is missing: run `probe` first")
    return json.loads(p.read_text())


def load_pts(path: Path) -> np.ndarray:
    return np.array(json.loads(Path(path).read_text())["pts"], float)


# ------------------------------------------------------------------------------------------------ pod: skeleton, track


def resolve_yolo(name: str) -> str:
    """An existing path, the package's weights/<name>, or the official ultralytics asset downloaded ONCE here (the track shards run in
    parallel and would race on the download)."""
    p = Path(name)
    if p.exists():
        return str(p.resolve())
    for c in (PACKAGE_ROOT / "weights" / p.name,):
        if c.exists():
            return str(c)
    from ultralytics.utils.downloads import attempt_download_asset
    return str(Path(attempt_download_asset(name)).resolve())


def skeleton_f0s(width: int, height: int) -> tuple[float, ...]:
    return tuple(float(round(max(width, height) * k)) for k in SKEL_F0_FRACTIONS)


def stage_skeleton(video: Path, cw: Path, yolo: str = "yolo11x-seg.pt", step: int | None = None) -> dict:
    t0 = time.time()
    pr = load_probe(cw)
    step = step or max(1, int(round(pr["fps"] / 2)))
    info = PR.skeleton(str(video), cw / "skel", step=step, yolo=resolve_yolo(yolo), log=log, f0s=skeleton_f0s(pr["width"], pr["height"]))
    size = np.load(cw / "skel" / "skeleton.npz")["size"]
    if (int(size[0]), int(size[1])) != (pr["width"], pr["height"]):
        raise RuntimeError(f"skeleton size {size.tolist()} != decoded {pr['width']}x{pr['height']}")
    if info["n_reached"] < info["n_frames"]:
        log(f"WARNING: only {info['n_reached']}/{info['n_frames']} skeleton frames reached by the init chain")
    nums = {k: info[k] for k in ("step", "n_frames", "n_reached", "n_corr", "median_px", "p90_px", "f_min", "f_max", "k1", "k2", "f_start")}
    record(cw, "skeleton", time.time() - t0, **nums)
    log(f"skeleton: {json.dumps(_jsonable(nums))}")
    return info


def shard_files(d: Path) -> list[Path]:
    """track_s<k>.npz of a folder in shard order (not the _matches files)."""
    got = [(int(m.group(1)), p) for p in d.glob("track_s*.npz") if (m := SHARD_RE.search(p.name))]
    return [p for _, p in sorted(got)]


def stage_track(video: Path, cw: Path, shards: int = 2, yolo: str = "yolo11x-seg.pt") -> dict:
    """Every frame against the skeleton: `shards` contiguous blocks as parallel `ptz_run track` processes on the one GPU."""
    t0 = time.time()
    pr = load_probe(cw)
    n = int(pr["frames"])
    for p in [*cw.glob("track_s*.npz"), *cw.glob("track_s*.log"), *cw.glob("frames_*.json")]:
        p.unlink()                                               # a previous run with another shard count must not leak into the union
    yolo = resolve_yolo(yolo)
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONPATH=os.pathsep.join([str(PACKAGE_ROOT), os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep))
    procs = []
    for k, block in enumerate(np.array_split(np.arange(n), max(1, shards))):
        fj = cw / f"frames_{k}.json"
        fj.write_text(json.dumps(block.tolist()))
        cmd = [sys.executable, "-u", "-m", "dancecap.cameras.ptz_run", "track", str(video), "--skeleton", str(cw / "skel"),
               "--out", str(cw / f"track_s{k}.npz"), "--frames", str(fj), "--yolo", yolo]
        fh = open(cw / f"track_s{k}.log", "w")
        procs.append((k, subprocess.Popen(cmd, cwd=str(PACKAGE_ROOT), env=env, stdout=fh, stderr=subprocess.STDOUT), fh, len(block)))
        log(f"track shard {k}: frames {block[0]}..{block[-1]}")
    bad = []
    for k, proc, fh, _ in procs:
        rc = proc.wait()
        fh.close()
        if rc:
            tail = (cw / f"track_s{k}.log").read_text(errors="replace").strip().splitlines()[-3:]
            bad.append(f"shard {k} rc {rc}: {' | '.join(tail)}")
    if bad:
        raise RuntimeError("track failed: " + "; ".join(bad))
    tr = [dict(np.load(p)) for p in shard_files(cw)]
    frames = np.concatenate([t["frame"] for t in tr])
    check_frames(n, track_frames=frames)
    rms = np.concatenate([t["rms_px"] for t in tr])
    nums = {"shards": len(procs), "frames": int(len(frames)), "rms_median_px": float(np.nanmedian(rms)),
            "rms_p90_px": float(np.nanpercentile(rms, 90)),
            "frames_without_matches": int(np.isnan(rms).sum())}
    record(cw, "track", time.time() - t0, **nums)
    log(f"track: {json.dumps(_jsonable(nums))}")
    return nums


# ------------------------------------------------------------------------------------------------ pod: MoGe-2


def pick_moge_frames(sk_frame: np.ndarray, sk_f: np.ndarray, reached: np.ndarray | None = None, n: int = MOGE_N_FRAMES,
                     wide_ratio: float = MOGE_WIDE_RATIO) -> list[int]:
    """Skeleton frames for MoGe-2: the wide state (focal within `wide_ratio` of the 2nd percentile) spread evenly over time; topped up with
    the next-widest frames when the wide state has fewer than n."""
    ok = np.ones(len(sk_frame), bool) if reached is None else np.asarray(reached, bool)
    fr, f = np.asarray(sk_frame)[ok], np.asarray(sk_f, float)[ok]
    wide = np.where(f <= wide_ratio * np.percentile(f, 2))[0]
    if len(wide) >= n:
        pick = wide[np.unique(np.round(np.linspace(0, len(wide) - 1, n)).astype(int))]
    else:
        rest = np.setdiff1d(np.arange(len(fr)), wide)
        pick = np.concatenate([wide, rest[np.argsort(f[rest])][:n - len(wide)]])
    return sorted(int(v) for v in fr[pick])


def _pip(args: list[str]) -> None:
    """pip install with the environment's numpy / torch pinned (a dependency must never swap them)."""
    pins = [f"numpy=={np.__version__}"]
    try:
        import torch
        pins.append(f"torch=={torch.__version__.split('+')[0]}")
    except ImportError:
        pass
    con = Path(os.environ.get("TMPDIR", "/tmp") if os.name != "nt" else os.environ.get("TEMP", ".")) / "mono_cam_pins.txt"
    con.write_text("\n".join(pins) + "\n")
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-c", str(con), *args], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"pip install {' '.join(args)} failed: {r.stderr[-600:]}")


def install_moge() -> None:
    """MoGe-2 (the pre-V3 commit) and its pinned utils3d WITHOUT dependencies, then the few small packages they import (single_cam_pod's recipe)."""
    _pip(["--no-deps", f"git+https://github.com/microsoft/MoGe.git@{MOGE_COMMIT}",
          f"git+https://github.com/EasternJournalist/utils3d.git@{UTILS3D_COMMIT}"])
    _pip(["click", "trimesh", "scipy", "matplotlib"])


def stage_moge(video: Path, cw: Path, n: int = MOGE_N_FRAMES, install: bool = False) -> dict:
    t0 = time.time()
    if install:
        install_moge()
        log(f"MoGe-2 installed ({time.time() - t0:.0f} s)")
    sk = np.load(cw / "skel" / "skeleton.npz")
    frames = pick_moge_frames(sk["frame"], sk["f"], sk["reached"] if "reached" in sk.files else None, n)
    out = cw / "geom"
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONPATH=os.pathsep.join([str(PACKAGE_ROOT), os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep))
    cmd = [sys.executable, "-u", "-m", "dancecap.cameras.geom_priors", str(video), "--frames", ",".join(map(str, frames)), "--out", str(out),
           "--models", "moge"]
    with open(cw / "moge.log", "w") as fh:
        rc = subprocess.run(cmd, cwd=str(PACKAGE_ROOT), env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
    rep = json.loads((out / "geom_report.json").read_text()) if (out / "geom_report.json").exists() else {}
    if rc or not (out / "moge.npz").exists():
        raise RuntimeError(f"MoGe-2 failed (rc {rc}): {rep.get('moge', {}).get('error', 'see camera_work/moge.log')}; install it with --install-moge "
                           f"(MoGe {MOGE_COMMIT[:8]} + utils3d {UTILS3D_COMMIT[:8]} --no-deps, then click trimesh scipy matplotlib)")
    m = np.load(out / "moge.npz")
    fx = m["intrinsics"][:, 0, 0] * float(m["size"][0])
    nums = {"frames": [int(v) for v in m["frame"]], "focal_px": np.round(fx, 1).tolist(), "focal_median_px": float(np.median(fx))}
    record(cw, "moge", time.time() - t0, **nums)
    log(f"moge: {json.dumps(_jsonable(nums))}")
    return nums


def stage_pod(video: Path, cw: Path, shards: int = 2, yolo: str = "yolo11x-seg.pt", moge_frames: int = MOGE_N_FRAMES,
              install: bool = False, step: int | None = None) -> dict:
    t0 = time.time()
    stage_probe(video, cw)
    stage_skeleton(video, cw, yolo, step)
    stage_track(video, cw, shards, yolo)
    stage_moge(video, cw, moge_frames, install)
    record(cw, "pod", time.time() - t0)
    return json.loads((cw / "mono_cam.json").read_text())


# ------------------------------------------------------------------------------------------------ world: inputs


@dataclass
class Inputs:
    """Where `world` reads from (camera_work/ of the pod job, or J2's john_lula folders for the validation) and caches its CPU solves."""
    probe: Path
    pts: Path
    skeleton: Path
    pairs: Path
    tracks: list[Path]
    matches: list[Path]
    moge: Path
    people: Path
    work: Path
    source: str = "camera_work"
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_camera_work(cls, cw: Path, people: Path) -> "Inputs":
        tr = shard_files(cw)
        return cls(cw / "probe.json", cw / "video_pts.json", cw / "skel" / "skeleton.npz", cw / "skel" / "pairs.npz", tr,
                   [p.with_name(p.stem + "_matches.npz") for p in tr], cw / "geom" / "moge.npz", people, cw)

    @classmethod
    def from_j2(cls, src: Path, cw: Path) -> "Inputs":
        """john_lula's J2 layout: room_work/ptz/{skeleton,pairs}.npz, room_work/pod/{track_s?,track_s?_matches,moge}.npz, people/teachers_01.npz;
        probe.json / video_pts.json from `probe` into the (scratch) camera_work `cw`, which also takes the caches."""
        rw = src / "room_work"
        tr = shard_files(rw / "pod")
        return cls(cw / "probe.json", cw / "video_pts.json", rw / "ptz" / "skeleton.npz", rw / "ptz" / "pairs.npz", tr,
                   [p.with_name(p.stem + "_matches.npz") for p in tr], rw / "pod" / "moge.npz", src / "people" / f"teachers_{CAM}.npz", cw,
                   source=f"J2 layout of {src}")


def dancers_from_toml(path: Path) -> dict:
    """{lead|follow: {stature_m, shoe_sole_m, sole_source}} of takes/<take>.toml; the statures are required (they are the metre)."""
    cfg = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    d = cfg.get("dancers") or {}
    out = {}
    for r in ("lead", "follow"):
        row = d.get(r) or {}
        if "stature_m" not in row:
            raise SystemExit(f"[mono_cam] {path} [dancers.{r}] stature_m is missing: the metre of a single camera comes from the statures")
        sole = row.get("shoe_sole_m", d.get("shoe_sole_m"))
        out[r] = {"stature_m": float(row["stature_m"]), "shoe_sole_m": float(SOLE_ASSUMED_M if sole is None else sole),
                  "sole_source": "take TOML" if sole is not None else f"assumed {SOLE_ASSUMED_M} m"}
    return out


def body_heights(dancers: dict) -> tuple[dict, dict]:
    """Eye and COCO-ankle heights above the floor (m) of the shod dancers, keyed by role id (1 lead, 2 follow)."""
    eye, ank = {}, {}
    for rid, r in ((1, "lead"), (2, "follow")):
        s, sole = dancers[r]["stature_m"], dancers[r]["shoe_sole_m"]
        eye[rid] = float(SV.eye_height(s)) + sole
        ank[rid] = ANKLE_OVER_STATURE * s + sole
    return eye, ank


# ------------------------------------------------------------------------------------------------ world: focal


def skeleton_wide_focal(f: np.ndarray) -> float:
    """The 1x focal of a solution: median of the focals below 1.3 x the 2nd percentile."""
    f = np.asarray(f, float)
    f = f[np.isfinite(f)]
    lo = np.percentile(f, 2)
    return float(np.median(f[f < ZOOM_RATIO_SPLIT * lo]))


def focal_from_moge(moge_frames: np.ndarray, moge_f: np.ndarray, sk_f: np.ndarray, step: int) -> dict:
    """Scale s of the skeleton's focals that matches MoGe-2 on the wide frames: median of the per-frame ratios f_moge / f_skeleton (the same
    frames, so a slow zoom drift of the wide state does not bias it). MoGe's tight-state focals are biased low and are not used."""
    k = np.asarray(moge_frames, float) / step
    f_at = np.exp(np.interp(k, np.arange(len(sk_f)), np.log(np.asarray(sk_f, float))))
    f1 = skeleton_wide_focal(sk_f)
    wide = f_at < ZOOM_RATIO_SPLIT * f1
    if wide.sum() < 3:
        raise ValueError(f"only {int(wide.sum())} MoGe-2 frames are in the wide state")
    r = np.asarray(moge_f, float)[wide] / f_at[wide]
    s = float(np.median(r))
    return {"s": s, "f1x_px": s * f1, "f_skeleton_1x_px": f1, "n_wide": int(wide.sum()), "n_frames": int(len(k)),
            "ratio_p10_p90": np.percentile(r, [10, 90]).round(4).tolist(), "f_moge_wide_median_px": float(np.median(np.asarray(moge_f)[wide])),
            "f_moge_tight_over_track": np.round(np.asarray(moge_f)[~wide] / (f_at[~wide] * s), 3).tolist()}


def resolve_skeleton(sk: dict, pairs: P.Pairs, s: float, size: tuple[int, int]) -> P.PtzSolution:
    """The skeleton with every focal FIXED at s x its own (zoom ratios kept), from its rotations / s: first no lens, then k1, then k1 on the
    inlier pairs twice (room_single_cam.stage_hypotheses for one s)."""
    n = len(sk["frame"])
    lens0 = P.Lens(*size, float(sk["lens"][0]), float(sk["lens"][1]))
    keep = P.reject_outliers(P.PtzSolution(sk["rv"], sk["f"], lens0), pairs, 2.5)
    sel = np.where(keep)[0][::3]
    pk = P.Pairs(pairs.i[sel], pairs.j[sel], pairs.p[sel], pairs.q[sel])
    sol = P.solve_ptz(n, pk, sk["rv"] / s, sk["f"] * s, P.Lens(*size, 0.0, 0.0), free=(), max_nfev=30, fix_f=True)
    sol = P.solve_ptz(n, pk, sol.rv, sol.f, sol.lens, free=("k1",), max_nfev=30, fix_f=True)
    lens = P.Lens(*size, sol.lens.k1, 0.0)
    keep = P.reject_outliers(P.PtzSolution(sol.rv, sol.f, lens), pairs, 3.0)
    pa = P.Pairs(pairs.i[keep], pairs.j[keep], pairs.p[keep], pairs.q[keep])
    sol = P.solve_ptz(n, pa, sol.rv, sol.f, lens, free=("k1",), max_nfev=40, fix_f=True)
    k2 = P.reject_outliers(sol, pa, 2.0)
    pb = P.Pairs(pa.i[k2], pa.j[k2], pa.p[k2], pa.q[k2])
    sol = P.solve_ptz(n, pb, sol.rv, sol.f, sol.lens, free=("k1",), max_nfev=40, fix_f=True)
    sol.info["pairs"] = len(pb)
    return sol


def refine_all_frames(sol_rv: np.ndarray, sol_f: np.ndarray, lens: P.Lens, matches: dict, step: int, n_frames: int) -> dict:
    """Every frame's (rotation, focal) from the stored matches against the re-solved skeleton (CPU); a frame without matches gets the
    skeleton's interpolation with rms NaN and 0 matches (it is not ok)."""
    res = PR.refine_tracks(matches, sol_rv, sol_f, lens, step)
    out = {"frame": np.arange(n_frames), "rv": np.zeros((n_frames, 3)), "f": np.zeros(n_frames), "rms_px": np.full(n_frames, np.nan),
           "n_inl": np.zeros(n_frames, int)}
    sk = {"rv": sol_rv, "f": sol_f}
    for i in range(n_frames):
        out["rv"][i], out["f"][i] = PR._interp_skeleton(sk, i / step)
    fr = res["frame"].astype(int)
    inside = (fr >= 0) & (fr < n_frames)
    for k in ("rv", "f", "rms_px", "n_inl"):
        out[k][fr[inside]] = res[k][inside]
    return out


def smooth_focal(f: np.ndarray, k: int = 5) -> np.ndarray:
    """Median filter of log f over k frames: removes the per-frame focal jitter of the track, keeps the ~0.7 s zoom steps."""
    lf = np.log(f)
    x = np.pad(lf, k // 2, mode="edge")
    return np.exp(np.median(np.lib.stride_tricks.sliding_window_view(x, k), axis=1))


def frame_ok(rms: np.ndarray, n_inl: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    thr = OK_RMS_PX_960 * max(1.0, max(size) / 960.0)
    with np.errstate(invalid="ignore"):
        return np.isfinite(rms) & (rms < thr) & (np.asarray(n_inl) >= OK_MIN_MATCHES)


def zoom_state(f_s: np.ndarray, ok: np.ndarray) -> tuple[float, np.ndarray]:
    """(1x focal, tight-state flags): the 1x focal = median of the ok focals below 1.3 x their 2nd percentile."""
    base = f_s[ok] if ok.sum() > 10 else f_s
    lo = np.percentile(base, 2)
    f1 = float(np.median(base[base < ZOOM_RATIO_SPLIT * lo]))
    return f1, f_s > ZOOM_RATIO_SPLIT * f1


# ------------------------------------------------------------------------------------------------ world: gravity (MoGe-2 floor)


def moge_floor(mg: dict, R_ref: np.ndarray, f: np.ndarray, f1x: float, width: int, seed: int = 0, min_pts: int = 500) -> dict:
    """Floor plane of every MoGe-2 frame (the model's camera frame and metric): the unit UP normal, the model's camera height and inlier share,
    its focal (intrinsics are normalised by the image width: x `width` px of this frame), the normal re-expressed for the track's focal
    (same horizon row) and carried to the reference frame with the frame's rotation. Track rows = video frames."""
    pts, mask, fr = mg["points"].astype(np.float32), mg["mask"], np.asarray(mg["frame"], int)
    H = pts.shape[1]
    rng = np.random.default_rng(seed)
    out = {"frame": [], "up_cam": [], "h_model": [], "inl": [], "f_model": [], "up_ref": [], "f_track": [], "state_2x": []}
    for i, f0 in enumerate(fr):
        if not 0 <= f0 < len(f):
            continue
        X = pts[i]
        sel = mask[i] & np.isfinite(X).all(-1) & (np.arange(H)[:, None] > 0.45 * H)
        Pf = X[sel]
        if len(Pf) < min_pts:
            continue
        n, d, inl = SV.plane_ransac(Pf, rng, tol=0.06, iters=200, up_prior=np.array([0, -1.0, 0]))
        f_mod = float(mg["intrinsics"][i][0, 0]) * width
        up_t = SV.rescale_up(n, f_mod, float(f[f0]))
        out["frame"].append(int(f0)); out["up_cam"].append(n); out["h_model"].append(abs(d)); out["inl"].append(float(inl.mean()))
        out["f_model"].append(f_mod); out["up_ref"].append(R_ref[f0].T @ up_t); out["f_track"].append(float(f[f0]))
        out["state_2x"].append(float(f[f0]) > ZOOM_RATIO_SPLIT * f1x)
    res = {k: np.array(v) for k, v in out.items()}
    if not len(res["frame"]):
        raise ValueError("no MoGe-2 frame has a usable floor")
    return res


def gravity_from_moge(mg: dict, min_inl: float = 0.3) -> tuple[np.ndarray, dict]:
    """UP in the reference frame: trimmed mean of the per-frame floor normals of the wide frames whose floor plane holds >= min_inl of the
    lower-half points (all wide frames when fewer than 3 do; all frames when no frame is wide), with the spread in degrees."""
    wide = ~mg["state_2x"]
    if not wide.any():
        wide = np.ones(len(wide), bool)
    sel = wide & (mg["inl"] >= min_inl)
    if sel.sum() < 3:
        sel = wide
    U = mg["up_ref"][sel]
    u = U.mean(0)
    u /= np.linalg.norm(u)
    for _ in range(3):
        ang = np.degrees(np.arccos(np.clip(U @ u, -1, 1)))
        keep = ang < max(2.0, 2.5 * np.median(ang))
        u = U[keep].mean(0)
        u /= np.linalg.norm(u)
    ang = np.degrees(np.arccos(np.clip(mg["up_ref"] @ u, -1, 1)))
    return u, {"n_frames_used": int(sel.sum()), "spread_deg_median": float(np.median(ang[sel])), "spread_deg_max": float(ang[sel].max()),
               "spread_deg_tight": [round(float(a), 2) for a in ang[~wide]], "frames": mg["frame"].tolist(),
               "angle_to_mean_deg": ang.round(2).tolist(), "floor_inlier_share": np.round(mg["inl"], 2).tolist()}


# ------------------------------------------------------------------------------------------------ world: dancers and the world frame (pure)


def dancer_detections(kp: np.ndarray, conf: np.ndarray, frame_index: np.ndarray, size: tuple[int, int], box: np.ndarray | None = None,
                      skip: np.ndarray | None = None, edge: np.ndarray | None = None, min_conf: float = MIN_DET_CONF) -> dict:
    """teachers_<cam>.npz arrays -> one row per (frame, role) with what the metric analysis needs.

    kp (F, 2, 17, 3) [lead, follow] COCO-17 (x, y, conf) in display px, conf (F, 2), frame_index (F,), box (F, 2, 4) xyxy, skip (F, 2) rows
    whose identity / skeleton is a guess (merged boxes, hidden behind the partner), edge (F, 2) at the image border (no height sample).
    foot_ok = the lower ankle is seen inside the frame (floor positions); height_ok = also eyes, shoulders, hips and a person > 1/16 of the frame
    (camera height samples). Same tests as room_single_cam.people()."""
    n = kp.shape[0]
    k = np.asarray(kp, float).reshape(-1, 17, 3).copy()
    bad = ~np.isfinite(k).all(-1)
    k[bad] = 0.0
    c = np.nan_to_num(np.asarray(conf, float).reshape(-1))
    W, H = size
    mk = SV.mean_pose_keypoints(k)
    a = mk["ank_low"]
    sk = np.zeros(len(c), bool) if skip is None else np.asarray(skip, bool).reshape(-1)
    ok = (c >= min_conf) & ~sk
    inside = (a[:, 0] > 3) & (a[:, 0] < W - 3) & (a[:, 1] > 0) & (a[:, 1] < H - 10)
    foot_ok = ok & (a[:, 2] > 0.3) & inside
    if box is not None:
        bh = np.nan_to_num(np.asarray(box, float).reshape(-1, 4)[:, 3] - np.asarray(box, float).reshape(-1, 4)[:, 1])
    else:
        bh = k[:, :, 1].max(1) - k[:, :, 1].min(1)
    height_ok = foot_ok & (a[:, 2] > 0.5) & (mk["eye"][:, 2] > 0.5) & (mk["sh"][:, 2] > 0.5) & (mk["hip"][:, 2] > 0.4) & (bh > H / 16.0)
    if edge is not None:
        height_ok &= ~np.asarray(edge, bool).reshape(-1)
    return {"frame": np.repeat(np.asarray(frame_index, int), 2), "role": np.tile(np.array([1, 2], np.int8), n), "foot": a[:, :2].copy(),
            "eye": mk["eye"][:, :2].copy(), "sh": mk["sh"][:, :2].copy(), "foot_ok": foot_ok, "height_ok": height_ok}


def camera_height_from_samples(h: np.ndarray, valid: np.ndarray, upright: np.ndarray, role: np.ndarray, two_x: np.ndarray,
                               min_upright: int = MIN_UPRIGHT, min_role: int = MIN_ROLE) -> dict:
    """Camera height = median of the two dancers' medians (each over its wide-state samples when it has >= min_role, else over all states).
    Fewer than `min_upright` upright samples in all: every valid dancer sample is used and `fallback_all_samples` says so."""
    up_ok = valid & upright
    fallback = int(up_ok.sum()) < min_upright
    base = valid if fallback else up_ok
    out = {"n_upright": int(up_ok.sum()), "n_valid": int(valid.sum()), "fallback_all_samples": bool(fallback), "roles": {}}
    meds = []
    for rid, name in ((1, "lead"), (2, "follow")):
        st = {}
        for state, m in (("wide", ~two_x), ("tight", two_x), ("all", np.ones(len(two_x), bool))):
            v = h[base & (role == rid) & m]
            st[state] = {"n": int(len(v)), "median": float(np.median(v)) if len(v) else None,
                         "p25": float(np.percentile(v, 25)) if len(v) else None, "p75": float(np.percentile(v, 75)) if len(v) else None}
        use = "wide" if st["wide"]["n"] >= min_role else ("all" if st["all"]["n"] >= 3 else None)
        st["used"] = use
        if use:
            meds.append(st[use]["median"])
        out["roles"][name] = st
    out["role_medians"] = meds
    out["camera_height_m"] = float(np.median(meds)) if meds else None
    return out


def trimmed_mean_2d(xz: np.ndarray, k: float = 4.0, floor_mad: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    """Mean of the points within k robust sigmas (1.4826 MAD, at least floor_mad) of the median on both axes; returns (mean, kept)."""
    med = np.median(xz, 0)
    mad = 1.4826 * np.median(np.abs(xz - med), 0)
    keep = (np.abs(xz - med) < k * np.maximum(mad, floor_mad)).all(1)
    return xz[keep].mean(0), keep


def world_solve(R_ref: np.ndarray, f: np.ndarray, lens: P.Lens, up_ref: np.ndarray, dets: dict, eye_m: dict, ankle_m: dict,
                two_x: np.ndarray, use_frame: np.ndarray | None = None, upright_deg: float = UPRIGHT_DEG, min_upright: int = MIN_UPRIGHT) -> dict:
    """The metric world of a fixed-centre camera from the dancers.

    R_ref (F,3,3) reference -> camera i, f (F,) px, lens (k1 / centre), up_ref: gravity (UP, either sign) in the reference frame; dets: frame,
    role (1 lead, 2 follow), foot / eye / sh pixels, foot_ok, height_ok (dancer_detections); eye_m / ankle_m {role: m above the floor};
    two_x (F,) tight-state flags; use_frame (F,) frames whose rotation is trusted.
    Returns Wm (3x3, columns = world X, Y, Z in reference coordinates), C (camera centre, world) = (0, h, d), the camera height summary, the
    origin samples, and every detection's world floor point (NaN where the foot is not usable)."""
    fr = np.asarray(dets["frame"], int)
    role = np.asarray(dets["role"])
    keep_fr = (fr >= 0) & (fr < len(f))
    fr_c = np.clip(fr, 0, len(f) - 1)
    use = keep_fr & (np.ones(len(fr), bool) if use_frame is None else np.asarray(use_frame, bool)[fr_c])
    Rd, fd = R_ref[fr_c], f[fr_c]
    rays = lambda pix: SV.rays_in_reference(np.asarray(pix, float), Rd, fd, lens)          # noqa: E731
    foot, eye, sh = rays(dets["foot"]), rays(dets["eye"]), rays(dets["sh"])
    up = np.asarray(up_ref, float) / np.linalg.norm(up_ref)
    chk = use & dets["foot_ok"]
    if np.median((foot[chk] if chk.any() else foot) @ up) > 0:          # feet are below the horizon: UP must point away from them
        up = -up
    Wh = np.stack(SV.horizon_basis(up), 1)
    fw, ew = foot @ Wh, eye @ Wh
    normal = SV.person_axis_normals(foot, sh)
    upright = np.abs(normal @ up) < np.sin(np.radians(upright_deg))
    He = np.where(role == 1, eye_m[1], eye_m[2])
    Ha = np.where(role == 1, ankle_m[1], ankle_m[2])
    h = SV.camera_height(fw, ew, He, Ha)
    two_d = np.asarray(two_x, bool)[fr_c]
    valid = use & dets["height_ok"] & np.isfinite(h) & (h > H_RANGE_M[0]) & (h < H_RANGE_M[1])
    hs = camera_height_from_samples(h, valid, upright, role, two_d, min_upright)
    h_cam = hs["camera_height_m"]
    if h_cam is None:
        raise ValueError(f"no camera height: {hs['n_valid']} valid dancer samples (eyes, shoulders, hips and the lower ankle seen)")
    # floor points of the dancers' lower ankles (camera-centred horizon axes), the origin = their trimmed mean in the wide state
    below = fw[:, 1] < -1e-3
    pos = np.full((len(fr), 3), np.nan)
    fo = use & dets["foot_ok"] & below
    pos[fo] = SV.floor_points(fw[fo], h_cam, Ha[fo])
    sel = fo & ~two_d
    origin_state = "wide"
    if sel.sum() < 20:
        sel, origin_state = fo, "all (fewer than 20 wide-state feet)"
    if sel.sum() < 3:
        raise ValueError(f"only {int(sel.sum())} dancer floor points: no origin")
    o_xz, kept = trimmed_mean_2d(pos[sel][:, [0, 2]])
    o_h = np.array([o_xz[0], 0.0, o_xz[1]])
    d = float(np.hypot(*o_xz))
    if d < 0.05:
        raise ValueError("the dancers' mean floor position is below the camera: the Z axis (towards the camera) is undefined")
    z_h = -o_h / d                                               # horizontal, from the dancers' origin towards the camera (camera at (0, h, 0))
    Z = Wh @ z_h
    Y = up
    X = np.cross(Y, Z)
    Wm = np.stack([X, Y, Z], 1)
    rot = Wm.T @ Wh                                              # horizon-basis coordinates -> world coordinates
    C = rot @ (np.array([0.0, h_cam, 0.0]) - o_h)
    Xw = (rot @ (pos - o_h).T).T
    Xw[:, 1] = 0.0
    idx = np.where(sel)[0]
    n_feet = {"lead": int((role[idx[kept]] == 1).sum()), "follow": int((role[idx[kept]] == 2).sum())}
    dist_samples = np.hypot(Xw[idx[kept], 0] - C[0], Xw[idx[kept], 2] - C[2])
    return {"Wm": Wm, "up_ref": up, "C": C, "camera_height_m": h_cam, "distance_m": d, "height": hs, "h_sample": h, "upright": upright,
            "valid": valid, "floor_world": Xw, "origin_state": origin_state, "origin_feet_samples": n_feet,
            "origin_kept": np.isin(np.arange(len(fr)), idx[kept]), "dancer_distance_percentiles_m": np.percentile(dist_samples, [5, 25, 50, 75, 95]),
            "dancer_extent_x_m": np.percentile(Xw[idx[kept], 0], [5, 95]), "dancer_extent_z_m": np.percentile(Xw[idx[kept], 2], [5, 95])}


# ------------------------------------------------------------------------------------------------ world: outputs


def camera_npz(R: np.ndarray, tvec: np.ndarray, K: np.ndarray, k1: float, C: np.ndarray, pts: np.ndarray, ok: np.ndarray, rms: np.ndarray,
               n_inl: np.ndarray, interp: np.ndarray, f: np.ndarray, two_x: np.ndarray, size: tuple[int, int], h_cam: float) -> dict:
    """cameras/01.npz: the demo's cameras/<cam>.npz schema (CONTRACTS.md) + the single-camera keys of room_single_cam.write_cameras;
    t = pts = the video's own presentation times (the identity clock of a one-camera take), frame_index = video frame."""
    n = len(pts)
    if not (len(R) == len(K) == len(ok) == len(f) == n):
        raise ValueError(f"camera arrays of {len(R)} frames for {n} presentation times")
    pts = np.asarray(pts, float)
    return dict(t=pts, pts=pts, frame_index=np.arange(n, dtype=np.int32), K=np.asarray(K, float), R=np.asarray(R, float),
                tvec=np.asarray(tvec, float), dist=np.full(n, float(k1)), ok=np.asarray(ok, bool), reproj_px=np.asarray(rms, np.float32),
                n_inliers=np.asarray(n_inl, np.int32), interpolated=np.asarray(interp, bool), focal_px=np.asarray(f, np.float32),
                zoom_2x=np.asarray(two_x, bool), centre_world=np.asarray(C, float), pp_px=np.array([size[0] / 2.0, size[1] / 2.0]),
                camera_height_m=np.array(float(h_cam)), native_size=np.array(size, np.int32), work_size=np.array(size, np.int32),
                camera_class=np.array("handheld_zoom"), type_model=np.array("fixed_centre"), attached=np.array(False),
                t_clock=np.array("video time of the single camera = the take's reference clock (identity clock); "
                                 "frame k at its presentation time pts[k] s"))


def recap_cams(take: str, video_name: str, pr: dict) -> dict:
    """cameras/recap_cams.json (jl_inputs' shape): the video's facts (display size, rotation, rate) and its identity clock."""
    return {"take": take, "cams": {CAM: {"id": CAM, "file": video_name, "width": int(pr["width"]), "height": int(pr["height"]),
                                         "rotation": int(pr.get("rotation", 0)), "avg_fps": float(pr.get("avg_fps") or pr["fps"]),
                                         "duration": pr.get("duration_s"), "slope": 1.0, "offset": 0.0, "accepted": True,
                                         "offset_source": "identity clock (one camera)"}}}


def write_outputs(cam_dir: Path, take: str, video_name: str, pr: dict, cams: dict, rig: dict) -> None:
    cam_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cam_dir / f"{CAM}.npz", **cams)
    n = len(cams["t"])
    np.savez_compressed(cam_dir / f"{CAM}_world.npz", t=cams["t"], R=cams["R"], tvec=cams["tvec"], C=np.tile(cams["centre_world"], (n, 1)),
                        K=cams["K"], dist=cams["dist"], ok=cams["ok"])
    (cam_dir / "world_from_sfm.json").write_text(json.dumps({
        "s": 1.0, "R": np.eye(3).tolist(), "t": [0.0, 0.0, 0.0],
        "gravity_source": "MoGe-2 floor normals of the wide frames carried through the track",
        "scale_source": "the dancers' statures (camera height from head / feet rays of upright dancers)",
        "floor_source": "plane y = 0 (the dancers' lower ankles at 0.039 x stature + sole)",
        "x_axis": "Z = horizontal direction from the dancers' mean floor position to the camera, X = Y x Z",
        "note": "no SfM model exists: the cameras are stored in the world frame, this file is the identity (dancecap.cameras.mono_cam)"}, indent=1))
    (cam_dir / "rig.json").write_text(json.dumps(_jsonable(rig), indent=1))
    (cam_dir / "recap_cams.json").write_text(json.dumps(_jsonable(recap_cams(take, video_name, pr)), indent=1))


def review_sheet(video: Path, out: Path, cams: dict, dets: dict, ws: dict, dancers: dict, n: int = 8) -> str:
    """A few frames with the 1 m floor grid, the world axes and a stature pole on each dancer's floor point (lead blue, follow red)."""
    import cv2
    from . import ptz_feats as PF
    from . import room_draw as D
    F_ = len(cams["ok"])
    okf = np.where(cams["ok"])[0]
    if not len(okf):
        return "no ok frame"
    want = sorted({int(okf[np.argmin(np.abs(okf - t))]) for t in np.linspace(0, F_ - 1, n)})
    W, H = (int(v) for v in cams["work_size"])
    got = dict(PF.read_video_frames(str(video), W, H, frames=want))
    d = float(np.hypot(cams["centre_world"][0], cams["centre_world"][2]))
    tiles, labels = [], []
    for fi in want:
        if fi not in got:
            continue
        im = got[fi].copy()
        K = cams["K"][fi]
        cd = dict(R=cams["R"][fi], tvec=cams["tvec"][fi], f=float(K[0, 0]), c=(float(K[0, 2]), float(K[1, 2])), k1=float(cams["dist"][fi]))
        D.draw_floor_grid(im, cd, (-5, 5), (-6, max(-4.0, d - 1.0)), 1.0, color=(0, 220, 255), thickness=1, major=5)
        D.draw_axes(im, cd, (0, 0, 0), 1.0)
        sel = np.where((dets["frame"] == fi) & np.isfinite(ws["floor_world"][:, 0]))[0]
        for j in sel:
            r = "lead" if dets["role"][j] == 1 else "follow"
            col = (255, 80, 0) if r == "lead" else (80, 80, 255)
            X = ws["floor_world"][j]
            top = dancers[r]["stature_m"] + dancers[r]["shoe_sole_m"]
            D.draw_points(im, X[None], cd, col, 5)
            D.draw_segment3d(im, np.array([X[0], 0.0, X[2]]), np.array([X[0], top, X[2]]), cd, col, 2, 20)
        tiles.append(im)
        labels.append(f"t={cams['t'][fi]:.1f}s f={K[0, 0]:.0f}px{' 2x' if cams['zoom_2x'][fi] else ''}{'' if cams['ok'][fi] else ' NOT OK'}")
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "contact.jpg"), D.contact_sheet(tiles, 4, 540 if W <= H else 720, labels), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return str(out / "contact.jpg")


# ------------------------------------------------------------------------------------------------ world: the stage


def _cached(path: Path, force: bool, make):
    if path.exists() and not force:
        return dict(np.load(path))
    d = make()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **d)
    return d


def stage_world(take: str, video: Path, work_root: str | Path | None = None, inputs: Inputs | None = None, f1x: float | None = None,
                toml: Path | None = None, people: Path | None = None, force: bool = False, review: bool = True, out_take: Path | None = None) -> dict:
    """The local CPU part: focal, every frame's rotation, gravity, height, origin, axes -> cameras/ and review/camera/ of the take."""
    t0, c0 = time.time(), time.process_time()
    td = out_take or take_dir(take, work_root)
    cw = td / "camera_work"
    cw.mkdir(parents=True, exist_ok=True)
    inp = inputs or Inputs.from_camera_work(cw, Path(people) if people else td / "people" / f"teachers_{CAM}.npz")
    if people:
        inp.people = Path(people)
    pr = json.loads(inp.probe.read_text())
    pts = load_pts(inp.pts)
    n = len(pts)
    check_frames(n, pr.get("frames_decoded"))
    size = (int(pr["width"]), int(pr["height"]))
    if not inp.tracks:
        raise SystemExit(f"[mono_cam] no track_s<k>.npz in {inp.work}: run the pod part first")
    check_frames(n, track_frames=np.concatenate([np.load(p)["frame"] for p in inp.tracks]))
    sk = dict(np.load(inp.skeleton))
    if (int(sk["size"][0]), int(sk["size"][1])) != size:
        raise ValueError(f"skeleton size {sk['size'].tolist()} != video {size}")
    step = int(sk["frame"][1] - sk["frame"][0]) if len(sk["frame"]) > 1 else int(round(pr["fps"]))
    timing = {}

    # focal hypothesis: MoGe-2 on the wide frames (or --f1x, or the phone-class prior)
    mg = dict(np.load(inp.moge)) if inp.moge.exists() else None
    f_sk1 = skeleton_wide_focal(sk["f"])
    if f1x:
        foc = {"s": f1x / f_sk1, "f1x_px": f1x, "f_skeleton_1x_px": f_sk1, "source": "given (--f1x)"}
    elif mg is not None:
        fm = mg["intrinsics"][:, 0, 0] * size[0]
        foc = {**focal_from_moge(mg["frame"], fm, sk["f"], step),
               "source": "MoGe-2 focal on the wide frames (median per-frame ratio to the skeleton)"}
    else:
        foc = {"s": 0.80 * max(size) / f_sk1, "f1x_px": 0.80 * max(size), "f_skeleton_1x_px": f_sk1,
               "source": "NO MoGe-2: phone-class prior 0.80 x the long side (+-10 %)"}
        log("WARNING: no MoGe-2 result, focal = 0.80 x the long side")
    s = float(foc["s"])
    log(f"focal: s = {s:.4f}, 1x {foc['f1x_px']:.1f} px ({foc['source']})")

    # the skeleton re-solved at that focal, then every frame from the stored matches
    ta = time.time()
    pr_ = np.load(inp.pairs)
    pairs = P.Pairs(pr_["i"], pr_["j"], pr_["p"], pr_["q"])

    def make_skel():
        sol = resolve_skeleton(sk, pairs, s, size)
        log(f"skeleton re-solved at s {s:.4f}: median {sol.median_px:.3f} px, p90 {sol.p90_px:.3f}, k1 {sol.lens.k1:+.4f} ({time.time() - ta:.0f} s)")
        return {"frame": sk["frame"], "rv": sol.rv, "f": sol.f, "lens": np.array([sol.lens.k1, 0.0, 0.0, 0.0]), "size": np.array(size),
                "median": sol.median_px, "p90": sol.p90_px, "rms": sol.rms_px, "pairs": sol.info["pairs"], "s": s}
    skel_path, track_path = inp.work / "final" / f"skel_s{s:.4f}.npz", inp.work / "final" / f"track_s{s:.4f}.npz"
    fresh = force or not skel_path.exists()                      # a new skeleton solve invalidates the refined track
    sol = _cached(skel_path, fresh, make_skel)
    timing["skeleton_resolve_s"] = time.time() - ta
    lens = P.Lens(*size, float(sol["lens"][0]), 0.0)
    tb = time.time()

    def make_track():
        m = PR.load_matches(inp.matches, len(sol["rv"]))
        r = refine_all_frames(sol["rv"], sol["f"], lens, m, step, n)
        log(f"track refined: {n} frames, rms median {np.nanmedian(r['rms_px']):.2f} px ({time.time() - tb:.0f} s)")
        return r
    trk = _cached(track_path, fresh, make_track)
    timing["track_refine_s"] = time.time() - tb

    # frames: smooth focal, ok flags, repaired gaps, zoom state
    f_s = smooth_focal(trk["f"])
    ok0 = frame_ok(trk["rms_px"], trk["n_inl"], size)
    rv_f, f_f, ok, interp = SV.fill_bad_frames(trk["rv"], f_s, ok0, max_gap=int(round(pr["fps"])))
    fmain, two_x = zoom_state(f_f, ok)
    R_ref = P.rotations(rv_f)

    # gravity
    if mg is None:
        raise SystemExit("[mono_cam] gravity needs MoGe-2 (camera_work/geom/moge.npz): run `moge` on the pod")
    mfl = moge_floor(mg, R_ref, f_f, fmain, size[0])
    up_ref, grav = gravity_from_moge(mfl)

    # dancers -> height, origin, axes
    dancers = dancers_from_toml(Path(toml) if toml else PACKAGE_ROOT / "takes" / f"{take}.toml")
    eye_m, ankle_m = body_heights(dancers)
    tt = dict(np.load(inp.people))
    skip = np.zeros(tt["conf"].shape, bool)
    if "merged" in tt:
        skip |= tt["merged"].astype(bool)
    if "skel_in_partner" in tt:
        skip |= np.nan_to_num(tt["skel_in_partner"]) > 0.5
    dets = dancer_detections(tt["kp"], tt["conf"], tt["frame_index"], size, tt.get("box"), skip, tt.get("edge"))
    ws = world_solve(R_ref, f_f, lens, up_ref, dets, eye_m, ankle_m, two_x, use_frame=ok)
    Wm, C, h_cam = ws["Wm"], ws["C"], ws["camera_height_m"]
    R, tvec, K, dist, _ = SV.camera_arrays(rv_f, f_f, Wm, C, size, float(sol["lens"][0]))
    cams = camera_npz(R, tvec, K, float(sol["lens"][0]), C, pts, ok, trk["rms_px"], trk["n_inl"], interp, f_f, two_x, size, h_cam)
    secs_wall, secs_cpu = time.time() - t0, time.process_time() - c0
    rms = trk["rms_px"]
    rig = {"schema": "dancecap.mono_cam.rig/1", "take": take, "video": Path(video).name, "inputs": inp.source,
           "model": "handheld phone: fixed centre, per-frame rotation and focal (zoom), one k1 (dancecap.cameras.mono_cam)",
           "image_px": list(size), "principal_point_px": [size[0] / 2, size[1] / 2], "k1": float(sol["lens"][0]),
           "focal": {**foc, "focal_1x_px": fmain, "focal_2x_px": float(np.median(f_f[two_x & ok])) if (two_x & ok).any() else None,
                     "zoom_ratio": float(np.median(f_f[two_x & ok]) / fmain) if (two_x & ok).any() else None,
                     "tight_frames": int(two_x.sum()),
                     "note": "the rotation-only cost is flat in the focal: a hypothesis; distances scale with it, the camera height does not"},
           "focal_1x_px": fmain, "camera_height_m": h_cam, "centre_world_m": C, "dancers_mean_distance_m": ws["distance_m"],
           "dancer_distance_percentiles_m": ws["dancer_distance_percentiles_m"].round(2), "dancer_extent_x_m": ws["dancer_extent_x_m"].round(2),
           "dancer_extent_z_m": ws["dancer_extent_z_m"].round(2),
           "world_axes_in_reference_frame": {"X": Wm[:, 0], "Y": Wm[:, 1], "Z": Wm[:, 2]},
           "gravity": {"source": "MoGe-2 floor normals of the wide frames carried through the track", "up_ref": ws["up_ref"],
                       **{k: v for k, v in grav.items() if k != "angle_to_mean_deg"}},
           "height": {"definition": f"head / feet rays of every upright dancer (axis within {UPRIGHT_DEG} deg of gravity), "
                                    "eye = 0.936 x stature + sole, "
                                    f"COCO ankle = {ANKLE_OVER_STATURE} x stature + sole; median of the role medians (wide state preferred)",
                      "eye_m": eye_m, "ankle_m": ankle_m, **ws["height"]},
           "dancers": dancers, "origin_rule": f"trimmed mean of both dancers' lower-ankle floor points ({ws['origin_state']} state, 4-MAD trim)",
           "origin_feet_samples": ws["origin_feet_samples"],
           "axes_rule": "Y = up (MoGe-2 floor normal), Z = horizontal direction from the dancers' origin to the camera, X = Y x Z",
           "moge": {"camera_height_model_m_median": float(np.median(mfl["h_model"])), "focal_model_px_wide_median":
                    float(np.median(mfl["f_model"][~mfl["state_2x"]])) if (~mfl["state_2x"]).any() else None,
                    "note": "the model's own metric scale is not used (people fix the metre)"},
           "track": {"frames": n, "ok_frames": int(ok.sum()), "interpolated_frames": int(interp.sum()), "not_ok_frames": int((~ok).sum()),
                     "reproj_median_px": float(np.nanmedian(rms)), "reproj_p90_px": float(np.nanpercentile(rms, 90)),
                     "ok_rule": f"rms < {OK_RMS_PX_960 * max(1.0, max(size) / 960.0):.2f} px and >= {OK_MIN_MATCHES} matches, gaps <= 1 s slerped",
                     "skeleton_step": step, "skeleton_frames": int(len(sk["frame"])), "skeleton_resolved_median_px": float(sol["median"]),
                     "skeleton_resolved_p90_px": float(sol["p90"]), "skeleton_pairs": int(sol["pairs"])},
           "seconds": {"world_wall": round(secs_wall, 1), "world_cpu": round(secs_cpu, 1), **{k: round(v, 1) for k, v in timing.items()}}}
    write_outputs(td / "cameras", take, Path(video).name, pr, cams, rig)
    np.savez_compressed(cw / "dancers_world.npz", frame=dets["frame"].astype(np.int32), role=dets["role"], X=ws["floor_world"].astype(np.float32),
                        foot_ok=dets["foot_ok"], height_ok=dets["height_ok"], upright=ws["upright"], h_sample=ws["h_sample"].astype(np.float32),
                        valid=ws["valid"], origin_kept=ws["origin_kept"])
    if review:
        try:
            got = review_sheet(Path(video), td / "review" / "camera", cams, dets, ws, dancers)
            log(f"review: {got}")
        except Exception as e:          # never fatal
            log(f"review failed: {type(e).__name__}: {e}")
    record(cw, "world", secs_wall, cpu_s=secs_cpu, focal_1x_px=fmain, camera_height_m=h_cam, distance_m=ws["distance_m"],
           ok_frames=int(ok.sum()), frames=n, gravity_spread_deg=grav["spread_deg_median"])
    log(f"world: 1x {fmain:.1f} px, height {h_cam:.3f} m, distance {ws['distance_m']:.2f} m, ok {int(ok.sum())}/{n} "
        f"({secs_wall:.0f} s wall, {secs_cpu:.0f} s CPU)")
    return {"rig": rig, "cams": cams, "ws": ws, "dets": dets, "R_ref": R_ref, "f": f_f, "lens": lens, "up_ref": up_ref, "two_x": two_x,
            "ok": ok, "trk": trk, "moge_floor": mfl, "focal": foc, "eye_m": eye_m, "ankle_m": ankle_m}


# ------------------------------------------------------------------------------------------------ CLI


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.cameras.mono_cam", description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("probe", "skeleton", "track", "moge", "pod", "world"))
    ap.add_argument("--take", required=True)
    ap.add_argument("--video", required=True, help="the phone video (on a pod: its mapped path)")
    ap.add_argument("--work-root", help="folder holding <take>/ (default PACKAGE_ROOT/work)")
    ap.add_argument("--yolo", default="yolo11x-seg.pt", help="YOLO11-seg weights (ultralytics downloads the official asset)")
    ap.add_argument("--shards", type=int, default=2, help="parallel track processes on the one GPU")
    ap.add_argument("--step", type=int, help="skeleton step (default round(fps / 2))")
    ap.add_argument("--moge-frames", type=int, default=MOGE_N_FRAMES)
    ap.add_argument("--install-moge", action="store_true", help="pip-install MoGe-2 (pinned, --no-deps) before the moge step")
    ap.add_argument("--f1x", type=float, help="world: the 1x focal (px) instead of MoGe-2's")
    ap.add_argument("--people", help="world: teachers npz (default <take>/people/teachers_01.npz)")
    ap.add_argument("--toml", help="world: take TOML (default takes/<take>.toml)")
    ap.add_argument("--from-j2", help="world: read J2's john_lula layout from this take folder (validation); "
                                      "probe.json from `probe` into --work-root")
    ap.add_argument("--force", action="store_true", help="world: redo the cached skeleton re-solve / track refinement")
    ap.add_argument("--no-review", action="store_true")
    a = ap.parse_args(argv)
    video = Path(a.video)
    td = take_dir(a.take, a.work_root)
    cw = td / "camera_work"
    if a.cmd == "probe":
        stage_probe(video, cw)
    elif a.cmd == "skeleton":
        stage_skeleton(video, cw, a.yolo, a.step)
    elif a.cmd == "track":
        stage_track(video, cw, a.shards, a.yolo)
    elif a.cmd == "moge":
        stage_moge(video, cw, a.moge_frames, a.install_moge)
    elif a.cmd == "pod":
        print(json.dumps(stage_pod(video, cw, a.shards, a.yolo, a.moge_frames, a.install_moge, a.step), indent=1))
    elif a.cmd == "world":
        inp = Inputs.from_j2(Path(a.from_j2), cw) if a.from_j2 else None
        r = stage_world(a.take, video, a.work_root, inp, a.f1x, Path(a.toml) if a.toml else None, Path(a.people) if a.people else None,
                        a.force, not a.no_review)
        keys = ("focal_1x_px", "camera_height_m", "dancers_mean_distance_m", "centre_world_m", "track")
        print(json.dumps(_jsonable({k: r["rig"][k] for k in keys}), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
