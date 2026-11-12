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


ANCHOR_STEP = 15          # frames between SAM anchors (0.5 s)
KP_TORSO = (5, 6, 11, 12)  # COCO shoulders, hips


def cmd_sam_prep(args) -> None:
    """Per clip: the raw tripod clip (sam_<clip>.mp4) and SAM 2.1 anchors (anchors_<clip>.json, jl_people_sam's format with
    john = lead, lula = follow): frames where both dancers have a clean SAM mask from lucas_luiza_h (joint slot, sanity
    filter), their teacher boxes and the masks barely overlap, at least ANCHOR_STEP frames apart. Prompt = the dancer's own
    mask box + positive points on its own torso keypoints (inside its mask) + negative points on the partner's torso."""
    import cv2
    ll = WORK / TAKE
    c2 = dict(np.load(ll / "cameras" / f"{CAM}.npz"))
    T = dict(np.load(ll / "people" / f"teachers_{CAM}.npz"))
    M = np.load(WORK / "lucas_luiza_h" / "people" / "masks_01.npz")
    from .fda_masked import mask_box
    out = OD / "sam"
    out.mkdir(parents=True, exist_ok=True)
    rep = {}
    for clip, rows in clip_rows().items():
        rows = np.array(rows)
        masks, info = role_masks(CAM, rows, M, T, c2)
        ls = T["lead_slot"][rows].astype(int)
        slot = {"lead": ls, "follow": 1 - ls}
        tmp = out / f"_raw_{clip}"
        tmp.mkdir(exist_ok=True)
        for i, r in enumerate(rows):
            (tmp / f"{i:05d}.jpg").write_bytes((OD / "frames" / f"{r:06d}.jpg").read_bytes())
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "30000/1001", "-i", str(tmp / "%05d.jpg"),
                        "-c:v", "libx264", "-crf", "10", "-pix_fmt", "yuv420p", str(out / f"sam_{clip}.mp4")], check=True)
        subprocess.run(["rm", "-rf", str(tmp)], check=True)
        anchors, last = [], -10 ** 9
        for i in range(len(rows)):
            ml, mf = masks["lead"][i], masks["follow"][i]
            if ml is None or mf is None or i - last < ANCHOR_STEP:
                continue
            if (ml & mf).sum() > 0.02 * min(ml.sum(), mf.sum()):
                continue
            a = {"frame": i}
            ok = True
            for role, key, other in (("lead", "john", "follow"), ("follow", "lula", "lead")):
                m = masks[role][i]
                kp = T["kp"][rows[i], slot[role][i]]
                kq = T["kp"][rows[i], slot[other][i]]
                pos = [[float(x), float(y)] for x, y, cf in kp[list(KP_TORSO)]
                       if cf >= 0.5 and 0 <= int(y) < m.shape[0] and 0 <= int(x) < m.shape[1] and m[int(y), int(x)]]
                neg = [[float(x), float(y)] for x, y, cf in kq[list(KP_TORSO)]
                       if cf >= 0.5 and 0 <= int(y) < m.shape[0] and 0 <= int(x) < m.shape[1] and not m[int(y), int(x)]]
                if len(pos) < 2:
                    ok = False
                    break
                a[key] = {"box": mask_box(m).tolist(), "pos": pos, "neg": neg}
            if ok:
                anchors.append(a)
                last = i
        gaps = np.diff([0] + [a["frame"] for a in anchors] + [len(rows) - 1])
        (out / f"anchors_{clip}.json").write_text(json.dumps(anchors))
        rep[clip] = {"frames": len(rows), "anchors": len(anchors), "max_gap_frames": int(gaps.max())}
        log(f"SAM prep {clip}: {len(anchors)} anchors, longest gap {gaps.max()} frames")
    (out / "sam_prep.json").write_text(json.dumps(rep, indent=1))


def sam_masks(clip: str, n: int) -> dict:
    """jl_people_sam output -> {role: [bool mask per frame]}: each frame takes the run from its NEAREST anchor (forward run of
    the anchor before, backward run of the anchor after); the partner's mask (dilated 7 px) is removed from the dancer's."""
    import cv2
    z = np.load(OD / "sam" / f"sam_{clip}.npz")
    runs = json.loads(str(z["runs"]))
    anchors = sorted({r["anchor"] for r in runs})
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    out = {r: [] for r in ROLES}
    for i in range(n):
        a = min(anchors, key=lambda x: abs(x - i))
        d = "f" if i >= a else "b"
        if i == a:
            d = "f" if f"m_{a}_f_{i}_0" in z.files else "b"
        m = {o: cv2.imdecode(z[f"m_{a}_{d}_{i}_{o}"], cv2.IMREAD_GRAYSCALE) > 127 for o in (0, 1)}
        out["lead"].append(m[0] & ~cv2.dilate(m[1].astype(np.uint8), k).astype(bool))
        out["follow"].append(m[1] & ~cv2.dilate(m[0].astype(np.uint8), k).astype(bool))
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
        if args.sam:   # the SAM 2.1 re-segmentation (anchored propagation) replaces the people stage's masks
            sm = sam_masks(clip, len(rows))
            for r in ROLES:
                info[r] = {**info[r], "source": "sam2.1 anchored propagation (uthana_test sam-prep + jl_people_sam)",
                           "frames_with_mask": int(sum(m.sum() > 500 for m in sm[r]))}
            masks = {r: [m if m.sum() > 500 else None for m in sm[r]] for r in ROLES}
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
           "max_minutes": 45, "needs": ["ffmpeg", "dancers"], "base": True, "min_vram_gb": 24, "inputs": inputs + frames,
           "exclude": ["dancecap/work/runpod/pywheels", "dancecap/work/runpod/wheelhouse", "mamma-turing"],
           "commands": [
               {"name": "sam_prep", "timeout_min": 8, "cmd": "python -m dancecap.uthana_test sam-prep"},
               {"name": "sam_ckpt", "timeout_min": 6,   # the people stage's fetcher (sha256-checked; curl hit an SSL error)
                "cmd": "python -c 'from dancecap.front_pod import fetch_sam2; fetch_sam2()'"},
               *[{"name": f"sam_{c}", "timeout_min": 12,
                  "cmd": f"python dancecap/jl_people_sam.py work/{TAKE}/uthana/sam/sam_{c}.mp4 work/{TAKE}/uthana/sam/anchors_{c}.json "
                         f"work/{TAKE}/uthana/sam/sam_{c}.npz --ckpt /workspace/HeadMovement/mamma/data/weights/sam2/sam2.1_hiera_large.pt "
                         f"--frames /workspace/sam_frames_{c}"} for c in UCLIPS],
               {"name": "inputs", "timeout_min": 15, "cmd": "python -m dancecap.uthana_test inputs --sam"}],
           "outputs": [f"dancecap/work/{TAKE}/uthana/inputs/", f"dancecap/work/{TAKE}/uthana/sam/anchors_E1.json",
                       f"dancecap/work/{TAKE}/uthana/sam/anchors_H1.json", f"dancecap/work/{TAKE}/uthana/sam/anchors_H3.json",
                       f"dancecap/work/{TAKE}/uthana/sam/sam_prep.json"],
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
    p = sub.add_parser("inputs")
    p.add_argument("--sam", action="store_true", help="use sam/sam_<clip>.npz (the SAM 2.1 re-segmentation)")
    sub.add_parser("sam-prep")
    a = ap.parse_args(argv)
    {"job": cmd_job, "inputs": cmd_inputs, "sam-prep": cmd_sam_prep}[a.cmd](a)


if __name__ == "__main__":
    main()
