"""john_lula: the small input files the single-camera front (dancecap.jl_front, run with HM_FRONT_DANCE=john_lula) expects (agent J3b).

    cameras/recap_cams.json   {cam: {file, width, height, rotation, avg_fps, slope, offset, ...}}: the video's facts and its clock (identity: reference time = source time)

    python -m dancecap.jl_inputs prepare        (local, light: ffprobe only)
"""
from __future__ import annotations

import argparse
import json

from .take import PACKAGE_ROOT, WORKSPACE

DANCE = "john_lula"
CAPTURE = "John-Lula"
WORK = PACKAGE_ROOT / "work" / DANCE
SOURCE = WORKSPACE / "dances" / CAPTURE                      # git-ignored junction to the owner's folder
VIDEO = "john_lula_jnj.mp4"


def probe(path) -> dict:
    import subprocess
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,avg_frame_rate,duration", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    a, b = s["avg_frame_rate"].split("/")
    return {"width": int(s["width"]), "height": int(s["height"]), "avg_fps": float(a) / float(b), "duration": float(s["duration"])}


def cams_table(probe_fn=probe) -> dict:
    p = probe_fn(SOURCE / VIDEO)
    return {"01": {"id": "01", "file": VIDEO, "width": p["width"], "height": p["height"], "rotation": 0, "avg_fps": p["avg_fps"], "duration": p["duration"],
                   "slope": 1.0, "offset": 0.0, "accepted": True, "offset_source": "identity clock (one camera)"}}


def prepare() -> dict:
    table = cams_table()
    (WORK / "cameras").mkdir(parents=True, exist_ok=True)
    (WORK / "cameras" / "recap_cams.json").write_text(json.dumps({"take": DANCE, "capture": CAPTURE, "cams": table}, indent=1))
    return table


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.jl_inputs", description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("prepare",))
    ap.parse_args(argv)
    print(json.dumps(prepare(), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
