"""Uthana Video-to-Motion 3.0 on lucas_luiza: can ONE hosted tool give body pose, planted feet and hand / arm placement
from one camera, in place of our set of tools? (owner 2026-11-12: "test this technology, especially version 3.0. we will need
to use the same technique of dancer isolation before processing. the end goal should evaluate unification of tools to solve
foot slip, hand and arm placement, and body pose").

Uthana (uthana.com, hosted only, no weights): Video-to-Motion 3.0 (2026-11-08) = one person per video, "adds foot IK" and "a
dedicated hand refinement step"; output FBX / GLB / BVH on a retargeted rig. Run through the web app in the owner's Chrome
(their account, $5 of credit = ~83 output seconds at $0.06/s), so the test is the tripod alone (the one-phone case, and the
"stable camera" Uthana asks for) on three hmr_test clips that each stress one target:
    E1 closed hold (hands in hold), H1 embrace (hands / arms under occlusion), H3 fast spins (feet), both dancers.

    python -m dancecap.uthana_test job       # local: the pod job that builds the isolated clips
    python -m dancecap.uthana_test inputs    # pod: work/lucas_luiza/uthana/inputs/<clip>_<role>.mp4 (+ .json)
    (upload in the web app, VTM 3.0 only; download BVH -> work/lucas_luiza/uthana/out/<clip>_<role>.bvh)
    python -m dancecap.uthana_test score     # pod: pose / feet / hands against the two-view fit and our other tools

Isolation = round 2 of the 4DAnyone test (dancecap.fda_masked): the dancer's own SAM 2.1 mask from lucas_luiza_h, slot per
frame by box IoU against the two-view teacher box, dilated 5 px, everything else grey 128, at the tripod's NATIVE 1280 x 720
(our camera geometry stays exact for the score). Likeness-derived files stay in git-ignored work/.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from .fda_masked import fill_masks, role_masks, soft
from .fda_test import TAKE, WORK, log
from .hmr_test import CLIPS, ROLES

UCLIPS = ("E1", "H1", "H3")
CAM = "02"
OD = WORK / TAKE / "uthana"
CRF = 20


def clip_rows() -> dict:
    """{clip: camera-02 rows} covering each hmr_test clip's core (ll grid k0..k1, both ends), consecutive."""
    from . import mamma_dyn as md
    from .jl_fitcheck import frame_map
    ll = WORK / TAKE
    offs = md.read_sync_offsets(ll)
    t = np.load(ll / "cameras" / f"{CAM}.npz")["t"]
    out = {}
    for c in UCLIPS:
        k0, k1 = CLIPS[c]
        r, ok = frame_map(md.offset_label(np.array([k0 / 30.0, k1 / 30.0]), offs.get(CAM)), t, max_dt=0.04)
        if not ok.all():
            raise SystemExit(f"clip {c}: no camera-02 row at an end")
        out[c] = list(range(int(r[0]), int(r[1]) + 1))
    return out


def cmd_inputs(args) -> None:
    import cv2
    from . import mamma_dyn as md
    ll = WORK / TAKE
    offs = md.read_sync_offsets(ll)
    c2 = dict(np.load(ll / "cameras" / f"{CAM}.npz"))
    T = dict(np.load(ll / "people" / f"teachers_{CAM}.npz"))
    M = np.load(WORK / "lucas_luiza_h" / "people" / "masks_01.npz")
    fdir = OD / "frames"
    out = OD / "inputs"
    out.mkdir(parents=True, exist_ok=True)
    rep = {}
    for clip, rows in clip_rows().items():
        rows = np.array(rows)
        frames = [cv2.imread(str(fdir / f"{r:06d}.jpg")) for r in rows]
        masks, info = role_masks(CAM, rows, M, T, c2)
        for role in ROLES:
            name = f"{clip}_{role}"
            tmp = out / f"_{name}"
            tmp.mkdir(exist_ok=True)
            ms = fill_masks(masks[role])
            for i, (fr, m) in enumerate(zip(frames, ms)):
                a = soft(m)[..., None]
                cv2.imwrite(str(tmp / f"{i:05d}.jpg"), (fr.astype(np.float32) * a + 128.0 * (1 - a)).astype(np.uint8),
                            [cv2.IMWRITE_JPEG_QUALITY, 97])
            mp4 = out / f"lucas_luiza_{name}.mp4"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "30000/1001", "-i", str(tmp / "%05d.jpg"),
                            "-c:v", "libx264", "-crf", str(CRF), "-preset", "slow", "-pix_fmt", "yuv420p",
                            "-movflags", "+faststart", str(mp4)], check=True)
            subprocess.run(["rm", "-rf", str(tmp)], check=True)
            tau = md.apply_offset(c2["t"][rows], offs.get(CAM))
            meta = {"clip": clip, "role": role, "cam": CAM, "rows": rows.tolist(), "tau": tau.tolist(),
                    "fps": "30000/1001", "frames": len(rows), "seconds": len(rows) * 1001 / 30000,
                    "mask": {"offset": info["offset"], "iou_mean": info["iou_mean"], **info[role]},
                    "bytes": mp4.stat().st_size}
            (out / f"lucas_luiza_{name}.json").write_text(json.dumps(meta, indent=1, default=float))
            rep[name] = {k: meta[k] for k in ("frames", "seconds", "bytes")} | {"mask_frames": info[role]["frames_with_mask"]}
            log(f"{name}: {len(rows)} frames ({meta['seconds']:.1f} s), masks {info[role]['frames_with_mask']}, "
                f"{meta['bytes'] / 1e6:.1f} MB")
    tot = sum(v["seconds"] for v in rep.values())
    (out / "inputs.json").write_text(json.dumps({"clips": rep, "seconds_total": tot, "cost_usd_at_0.06": tot * 0.06}, indent=1))
    log(f"total {tot:.1f} s of video -> ${tot * 0.06:.2f} at VTM 3.0's $0.06/s")


def cmd_job(args) -> None:
    rows = sorted({r for v in clip_rows().values() for r in v})
    frames = [{"local": str(WORK / TAKE / "frames" / CAM / f"{r:06d}.jpg"),
               "pod": f"dancecap/work/{TAKE}/uthana/frames/{r:06d}.jpg"} for r in rows]
    inputs = [f"dancecap/work/{TAKE}/cameras/{CAM}.npz", f"dancecap/work/{TAKE}/cameras/world_from_sfm.json",
              f"dancecap/work/{TAKE}/sync/offsets.json", f"dancecap/work/{TAKE}/clock/alignment.json",
              f"dancecap/work/{TAKE}/people/teachers_{CAM}.npz", "dancecap/work/lucas_luiza_h/people/masks_01.npz"]
    job = {"schema": "dancecap.pod_job/1", "name": "lucas_luiza_uthana_inputs", "take": TAKE, "cwd": "dancecap",
           "max_minutes": 25, "needs": ["ffmpeg"], "base": True, "cpu": True, "inputs": inputs + frames,
           "exclude": ["dancecap/work/runpod/pywheels", "dancecap/work/runpod/wheelhouse", "mamma-turing"],
           "commands": [{"name": "inputs", "timeout_min": 15, "cmd": "python -m dancecap.uthana_test inputs"}],
           "outputs": [f"dancecap/work/{TAKE}/uthana/inputs/"],
           "verify": [f"dancecap/work/{TAKE}/uthana/inputs/inputs.json"],
           "notes": ["Uthana test (dancecap.uthana_test): SAM-isolated one-dancer tripod clips of E1 / H1 / H3 for upload "
                     "to Uthana's web app (owner's account, owner's go 2026-11-12)."]}
    p = WORK / TAKE / "jobs" / "pod" / "lucas_luiza_uthana_inputs.json"
    p.write_text(json.dumps(job, indent=1))
    log(f"job {p}: {len(frames)} frames")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("job")
    sub.add_parser("inputs")
    a = ap.parse_args(argv)
    {"job": cmd_job, "inputs": cmd_inputs}[a.cmd](a)


if __name__ == "__main__":
    main()
