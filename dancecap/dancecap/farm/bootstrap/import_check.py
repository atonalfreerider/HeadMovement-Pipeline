"""Headless import check of every module the farm runs on a pod (no GPU needed, no display).

    python import_check.py [--out /workspace/hm/env/import_check.json] [--cpu]

Imports each dancecap module in a fresh interpreter (so one failure cannot hide another), with the pod's environment
(PYOPENGL_PLATFORM=egl, MPLBACKEND=Agg) and records ok / error per module plus the third-party packages and versions that
the analysis stage needs. Exit code 1 when a REQUIRED module fails (modules that do not exist yet in this code version, e.g.
dancecap.stitch before it is committed, are reported as 'absent', not as failures).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time

REQUIRED = [
    # front / fit / refine
    "dancecap.mamma_dyn", "dancecap.mamma_chunks", "dancecap.mamma_seams", "dancecap.pose_refine", "dancecap.pose_quality",
    "dancecap.front_pod", "dancecap.front_full", "dancecap.cameras.pipeline", "dancecap.sync", "dancecap.dancers",
    # analysis
    "dancecap.timing", "dancecap.physics", "dancecap.physics_body", "dancecap.physics_spline", "dancecap.counterbalance",
    "dancecap.floorcraft", "dancecap.analysis_review", "dancecap.judgments", "dancecap.shoes", "dancecap.review",
    "dancecap.camera_types", "dancecap.camera_video_check", "dancecap.clock_studio",
    # camera-gate evidence
    "dancecap.cameras.video_zoom", "dancecap.cameras.video_motion", "dancecap.cameras.gate_cameras",
    "dancecap.cameras.gate_final", "dancecap.cameras.gate_images", "dancecap.cameras.attach_global",
    # moves / narration
    "dancecap.moves.narration", "dancecap.moves.labels", "dancecap.moves.path", "dancecap.moves.review",
]
OPTIONAL = ["dancecap.stitch", "dancecap.farm.worker", "dancecap.farm.chunks", "dancecap.farm.world", "dancecap.farm.timeline",
            "dancecap.beats", "dancecap.song"]
PACKAGES = ["numpy", "scipy", "matplotlib", "trimesh", "cv2", "torch", "smplx", "yaml", "pandas", "rapidfuzz", "librosa", "soxr",
            "faster_whisper", "ctranslate2", "typesafe_sdk", "open3d", "pycolmap", "lightglue", "sam2", "ultralytics",
            "pytorch_sdf", "beat_this", "gsplat"]


def try_import(mod: str) -> dict:
    if importlib.util.find_spec(mod.split(".")[0]) is None:
        return {"ok": False, "status": "absent"}
    try:
        spec = importlib.util.find_spec(mod)
    except (ImportError, ValueError):
        spec = None
    if spec is None:
        return {"ok": False, "status": "absent"}
    t0 = time.time()
    p = subprocess.run([sys.executable, "-c", f"import {mod}"], capture_output=True, text=True, timeout=300,
                       env=dict(os.environ, MPLBACKEND="Agg", PYOPENGL_PLATFORM="egl", DANCECAP_POD="1"))
    if p.returncode == 0:
        return {"ok": True, "status": "ok", "seconds": round(time.time() - t0, 1)}
    return {"ok": False, "status": "error", "error": (p.stderr.strip().splitlines() or ["?"])[-1][:300]}


def package_versions() -> dict:
    out = {}
    for name in PACKAGES:
        if importlib.util.find_spec(name) is None:
            out[name] = None
            continue
        p = subprocess.run([sys.executable, "-c", f"import {name} as m; print(getattr(m, '__version__', 'ok'))"],
                           capture_output=True, text=True, timeout=120)
        out[name] = p.stdout.strip().splitlines()[-1] if p.returncode == 0 and p.stdout.strip() else f"IMPORT ERROR {p.stderr.strip()[-120:]}"
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    mods = {m: try_import(m) for m in REQUIRED + OPTIONAL}
    pk = package_versions()
    bad = [m for m in REQUIRED if mods[m]["status"] != "ok"]
    doc = {"created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "python": sys.version.split()[0],
           "modules": mods, "packages": pk, "failed": bad}
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=1)
    print(json.dumps({"failed": bad, "absent": [m for m, v in mods.items() if v["status"] == "absent"],
                      "packages_missing": [k for k, v in pk.items() if v is None]}, indent=1))
    for m in bad:
        print(f"  {m}: {mods[m].get('error') or mods[m]['status']}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
