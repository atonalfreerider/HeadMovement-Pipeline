"""carlos_aline: the inputs the class-recap front end (dancecap.recap_front, run with HM_FRONT_DANCE=carlos_aline) expects (agent C3).

C2 left two camera tracks (cameras/01.npz, 02.npz: per-frame rotation + focal of a tripod head that pans / tilts / zooms) and the YOLO11x-pose
detections of every person of both videos (room_work/podout/pose_A.npz, pose_B.npz). The recap front wants, for a take work dir:

    cameras/<cam>.npz       the demo's schema + `pts` (the video's own presentation time of each canonical frame) + `native_size`
    cameras/recap_cams.json {cam: {file, width, height, rotation, avg_fps, slope, offset}} (reference time = slope * source + offset)
    people/pose_tables/<cam>.npz   kp (F, P, 17, 3) [x, y, conf] NaN padded, n (F,), rgb (F, P, 3) torso colour, box (F, P, 4), det_conf (F, P)

    python -m dancecap.carlos_aline_inputs prepare        (local, light: numpy only)

C2's original track files are kept as cameras/c2_orig/<cam>.npz before the two additive keys are written.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

from .take import PACKAGE_ROOT, WORKSPACE

DANCE = "carlos_aline"
CAPTURE = "Carlos-Aline"
WORK = PACKAGE_ROOT / "work" / DANCE
PEOPLE = WORK / "people"
SOURCE = WORKSPACE / "dances" / CAPTURE                       # git-ignored junction to the owner's folder
CAMS = ("01", "02")
VIDEO = {"01": "Carlos Aline Zouk Artist Jack Jill at Zouk SEA 2019.mp4", "02": "Zouk Sea 2019 Carlos and Aline Borges Jack Jill.mp4"}
POSE_SRC = {"01": WORK / "room_work" / "podout" / "pose_A.npz", "02": WORK / "room_work" / "podout" / "pose_B.npz"}


def log(msg: str) -> None:
    print(f"[carlos_aline_inputs {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def probe(path: Path) -> dict:
    import subprocess
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,avg_frame_rate,duration",
                          "-of", "json", str(path)], capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    a, b = s["avg_frame_rate"].split("/")
    return {"width": int(s["width"]), "height": int(s["height"]), "avg_fps": float(a) / float(b), "duration": float(s["duration"])}


def cams_table() -> dict:
    sync = json.loads((WORK / "sync" / "sync.json").read_text())
    rows = {c["id"]: c for c in sync["cameras"]}
    out = {}
    for cam in CAMS:
        p = probe(SOURCE / VIDEO[cam])
        out[cam] = {"id": cam, "file": VIDEO[cam], "width": p["width"], "height": p["height"], "rotation": 0, "avg_fps": p["avg_fps"], "duration": p["duration"],
                    "slope": float(rows[cam]["slope"]), "offset": float(rows[cam]["offset"]), "accepted": True, "offset_source": rows[cam].get("offset_source", "")}
    return out


def pose_table(src: Path, n_frames: int | None = None) -> dict:
    """The flat detection list of people2d (frame, box, conf, kp, rgb) as per-frame tables, persons ordered by confidence within a frame."""
    z = dict(np.load(src))
    fr = z["frame"].astype(int)
    F = int(z["n_frames"]) if n_frames is None else n_frames
    n = np.bincount(fr, minlength=F).astype(np.int16)
    P = int(max(1, n.max()))
    kp = np.full((F, P, 17, 3), np.nan, np.float32)
    box = np.full((F, P, 4), np.nan, np.float32)
    dconf = np.zeros((F, P), np.float32)
    rgb = np.full((F, P, 3), np.nan, np.float32)
    order = np.lexsort((-z["conf"], fr))
    slot = np.zeros(F, int)
    for i in order:
        f = fr[i]
        j = slot[f]
        slot[f] += 1
        kp[f, j], box[f, j], dconf[f, j], rgb[f, j] = z["kp"][i], z["box"][i], z["conf"][i], z["rgb"][i]
    return {"kp": kp, "n": n, "box": box, "det_conf": dconf, "rgb": rgb}


def prepare(force: bool = False) -> dict:
    from .cameras.newcam import video_pts
    cdir = WORK / "cameras"
    bak = cdir / "c2_orig"
    bak.mkdir(exist_ok=True)
    table = cams_table()
    (cdir / "recap_cams.json").write_text(json.dumps({"take": DANCE, "capture": CAPTURE, "cams": table}, indent=1))
    res = {}
    for cam in CAMS:
        src = cdir / f"{cam}.npz"
        if not (bak / f"{cam}.npz").exists():
            shutil.copyfile(src, bak / f"{cam}.npz")
        z = dict(np.load(bak / f"{cam}.npz", allow_pickle=True))
        pts_all = video_pts(SOURCE / VIDEO[cam], WORK / "frames" / f"{cam}_pts.json")
        fi = np.asarray(z["frame_index"], int)
        z["pts"] = pts_all[fi].astype(np.float64)
        z["native_size"] = np.array([table[cam]["width"], table[cam]["height"]], np.int32)
        z["work_size"] = z["native_size"]
        np.savez_compressed(src, **z)
        dev = float(np.max(np.abs((table[cam]["slope"] * z["pts"] + table[cam]["offset"]) - z["t"])))
        pt = pose_table(POSE_SRC[cam], len(pts_all))
        (PEOPLE / "pose_tables").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(PEOPLE / "pose_tables" / f"{cam}.npz", **pt)
        res[cam] = {"frames": int(len(fi)), "t": [round(float(z["t"][0]), 3), round(float(z["t"][-1]), 3)], "clock_vs_pts_max_dev_s": round(dev, 5),
                    "pose_frames": int(len(pt["n"])), "max_persons": int(pt["kp"].shape[1])}
        log(f"{cam}: {res[cam]}")
    (PEOPLE / "inputs.json").write_text(json.dumps(res, indent=1))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.carlos_aline_inputs", description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("prepare",))
    a = ap.parse_args(argv)
    prepare()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
