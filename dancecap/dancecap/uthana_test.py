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
from .hmr_test import CLIP_NOTE, CLIPS, ROLES, interp_grid

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


# ------------------------------------------------------------------------------------------------ BVH + score (pod)

# SMPL body joints scored for every method (the 15 that every rig here defines): pelvis, hips, knees, ankles, neck, head,
# shoulders, elbows, wrists
J15 = [0, 1, 2, 4, 5, 7, 8, 12, 15, 16, 17, 18, 19, 20, 21]
TORSO = [0, 1, 2, 12, 16, 17]          # for the arm check: Procrustes on the torso only
ARM = [18, 19, 20, 21]                 # elbows, wrists
PARTNER_TORSO = [0, 12, 16, 17]   # pelvis, neck, shoulders: joints every method here has (Uthana's UE5 rig has no SMPL spine joints)
CONTACT_M = 0.15
# BVH joint name candidates -> SMPL index (Mixamo / UE / generic names; the first one found wins)
BVH_NAMES = {0: ["Hips", "hips", "pelvis", "Pelvis", "root"], 1: ["LeftUpLeg", "LeftThigh", "thigh_l", "l_hip", "LeftHip"],
             2: ["RightUpLeg", "RightThigh", "thigh_r", "r_hip", "RightHip"], 4: ["LeftLeg", "LeftShin", "calf_l", "l_knee", "LeftKnee"],
             5: ["RightLeg", "RightShin", "calf_r", "r_knee", "RightKnee"], 7: ["LeftFoot", "foot_l", "l_ankle", "LeftAnkle"],
             8: ["RightFoot", "foot_r", "r_ankle", "RightAnkle"], 12: ["Neck", "neck", "neck_01", "Neck1"],
             15: ["Head", "head"], 16: ["LeftArm", "upperarm_l", "l_shoulder", "LeftUpperArm"],
             17: ["RightArm", "upperarm_r", "r_shoulder", "RightUpperArm"], 18: ["LeftForeArm", "lowerarm_l", "l_elbow", "LeftLowerArm"],
             19: ["RightForeArm", "lowerarm_r", "r_elbow", "RightLowerArm"], 20: ["LeftHand", "hand_l", "l_wrist", "LeftWrist"],
             21: ["RightHand", "hand_r", "r_wrist", "RightWrist"]}


def read_bvh(path: Path) -> tuple[list[str], np.ndarray, float]:
    """BVH -> (joint names, world positions (F, J, 3) in the file's units, frame time). Euler channels in their listed
    order (intrinsic: R = R_c1 R_c2 R_c3), position channels where present (root, sometimes all)."""
    from scipy.spatial.transform import Rotation as Rot
    tok = path.read_text().split()
    i, names, parents, offsets, chans, stack = 0, [], [], [], [], []
    last = -1
    while tok[i] != "MOTION":
        t = tok[i]
        if t in ("ROOT", "JOINT"):
            names.append(tok[i + 1]); parents.append(stack[-1] if stack else -1); last = len(names) - 1; i += 2
        elif t == "End":
            names.append(names[last] + "_end"); parents.append(last); last = len(names) - 1; i += 2
        elif t == "{":
            stack.append(last); i += 1
        elif t == "}":
            stack.pop(); last = stack[-1] if stack else -1; i += 1
        elif t == "OFFSET":
            offsets.append([float(x) for x in tok[i + 1:i + 4]]); i += 4
            if len(chans) < len(names):
                chans.append([])
        elif t == "CHANNELS":
            n = int(tok[i + 1]); chans[-1] = tok[i + 2:i + 2 + n]; i += 2 + n
        else:
            i += 1
    nf = int(tok[i + 2]); ft = float(tok[i + 5]); vals = np.array(tok[i + 6:], float).reshape(nf, -1)
    J = len(names)
    pos = np.zeros((nf, J, 3)); Rg = np.zeros((nf, J, 3, 3)); col = 0
    for j in range(J):
        ch = chans[j] if j < len(chans) else []
        p = np.tile(np.asarray(offsets[j], float), (nf, 1)); R = np.tile(np.eye(3), (nf, 1, 1))
        rot_axes, rot_cols = "", []
        for c in ch:
            if c.endswith("position"):
                p[:, "XYZ".index(c[0])] = vals[:, col] + (0 if parents[j] >= 0 else 0)
            else:
                rot_axes += c[0]; rot_cols.append(col)
            col += 1
        if rot_axes:
            R = Rot.from_euler(rot_axes, vals[:, rot_cols], degrees=True).as_matrix()   # uppercase = intrinsic
        if parents[j] < 0:
            Rg[:, j] = R; pos[:, j] = p
        else:
            P = parents[j]
            pos[:, j] = pos[:, P] + np.einsum("fij,fj->fi", Rg[:, P], p); Rg[:, j] = Rg[:, P] @ R
    return names, pos, ft


def bvh_to_smpl(names: list[str], pos: np.ndarray) -> tuple[np.ndarray, dict]:
    """(F, 22, 3) with the BVH joints mapped onto SMPL indices (NaN where no name matched) + the mapping used."""
    out = np.full((pos.shape[0], 22, 3), np.nan)
    used = {}
    low = {n.lower().split(":")[-1]: k for k, n in enumerate(names)}
    for j, cands in BVH_NAMES.items():
        for c in cands:
            k = low.get(c.lower())
            if k is not None:
                out[:, j] = pos[:, k]; used[j] = names[k]; break
    return out, used


def sim3(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    from .mono4d.rig import umeyama
    ok = np.isfinite(src).all(-1) & np.isfinite(dst).all(-1)
    u = umeyama(src[ok], dst[ok], with_scale=True)
    return float(u["s"]), np.array(u["R"]), np.array(u["t"])


def procrustes_err(X: np.ndarray, Y: np.ndarray, fit: list, eval_: list) -> float:
    """Similarity fitted on the `fit` joints only, error (mm) on the `eval_` joints."""
    from .mono4d.rig import umeyama
    u = umeyama(X[fit], Y[fit], with_scale=True)
    Z = float(u["s"]) * X @ np.array(u["R"]).T + np.array(u["t"])
    return float(np.linalg.norm(Z[eval_] - Y[eval_], axis=1).mean() * 1000)


def clip_metrics(J: np.ndarray, B: dict, role: str, stance: np.ndarray) -> dict:
    """One method, one dancer, one clip, on the grid frames where J is finite. J (F, 22, 3) in the method's own frame;
    B = {role: (F, 22, 3) two-view world joints}. Placed by ONE Sim3 per clip onto B0's dancer (every method alike)."""
    from .hmr_test import pa_mpjpe
    Y = B[role]
    ok = np.isfinite(J[:, J15]).all((1, 2)) & np.isfinite(Y[:, J15]).all((1, 2))
    if ok.sum() < 10:
        return {"frames": int(ok.sum())}
    s, R, t = sim3(J[ok][:, J15].reshape(-1, 3), Y[ok][:, J15].reshape(-1, 3))
    W = s * J @ R.T + t
    out = {"frames": int(ok.sum()), "scale": s}
    out["pa_mm"] = float(np.median([pa_mpjpe(J[i, J15], Y[i, J15]) for i in np.flatnonzero(ok)]))
    out["root_traj_cm"] = float(np.median(np.linalg.norm(W[ok, 0] - Y[ok, 0], axis=1)) * 100)
    out["arm_mm"] = float(np.median([procrustes_err(J[i], Y[i], TORSO, ARM) for i in np.flatnonzero(ok)]))
    # feet: ankle skate on the reference's stance frames (cm/s, horizontal), and how high the stance ankle floats above
    # this method's own lowest ankle level in the clip (cm, median)
    sk, fl = [], []
    floor = np.nanpercentile(W[ok][:, [7, 8], 1], 5)
    for side, j in ((0, 7), (1, 8)):
        both = stance[:-1, side] & stance[1:, side] & ok[:-1] & ok[1:]
        v = np.linalg.norm(W[1:, j][:, [0, 2]] - W[:-1, j][:, [0, 2]], axis=1) * 30 * 100
        sk.append(v[both])
        st = stance[:, side] & ok
        fl.append((W[st, j, 1] - floor) * 100)
    sk, fl = np.concatenate(sk), np.concatenate(fl)
    out["skate_cm_s"] = float(np.median(sk)) if len(sk) else None
    out["skate_p90_cm_s"] = float(np.percentile(sk, 90)) if len(sk) else None
    out["stance_float_cm"] = float(np.median(fl)) if len(fl) else None
    out["_W"] = W
    return out


def contact_metrics(Wm: dict, B: dict) -> dict:
    """Hands on the partner: per dancer and hand, frames where the two-view fit has the wrist within CONTACT_M of the
    partner's torso joints; the method (each dancer placed by its own clip Sim3) -> recall (also within CONTACT_M) and the
    median |d_method - d_ref| (cm) over those frames."""
    out = {}
    for role, other in (("lead", "follow"), ("follow", "lead")):
        if role not in Wm or other not in Wm:
            continue
        for hand, j in (("left", 20), ("right", 21)):
            d0 = np.linalg.norm(B[role][:, j][:, None] - B[other][:, PARTNER_TORSO], axis=-1).min(1)
            dm = np.linalg.norm(Wm[role][:, j][:, None] - Wm[other][:, PARTNER_TORSO], axis=-1).min(1)
            c = (d0 < CONTACT_M) & np.isfinite(dm)
            if c.sum() < 5:
                continue
            out[f"{role}_{hand}"] = {"contact_frames": int(c.sum()), "recall": float((dm[c] < CONTACT_M).mean()),
                                     "dist_err_cm": float(np.median(np.abs(dm[c] - d0[c])) * 100)}
    return out


def cmd_score(args) -> None:
    from .hmr_test import (Ref, load_b1, load_b2, load_candidate, out_dir, placement_b, to_world)
    ref = Ref(TAKE)
    od = out_dir(TAKE)
    res = {"schema": "dancecap.uthana_test/1", "clips": {}, "bvh": {}}
    meths = {}
    for r in ROLES:   # every baseline as world joints on the grid (placement B, the hmr_test convention)
        for m, loader in (("B1 mono fit", lambda: load_b1(ref, r)), ("B2 SAM 3D Body", lambda: load_b2(ref, r)),
                          ("ExMR", lambda: load_candidate(ref, od, "exmr", r))):
            Jc, K = loader()
            fin = np.isfinite(Jc).all((1, 2)) & np.isfinite(K).all((1, 2))
            Jc[~fin], K[~fin] = np.nan, np.nan
            Jw = np.full((ref.NG, 22, 3), np.nan)
            Jw[:, :22] = to_world(placement_b(Jc, K, ref.K02, ref.zB0[r]), ref.R02, ref.t02)[:, :22]
            meths.setdefault(m, {})[r] = Jw
    # Uthana: BVH per clip and dancer, frame i at clip start + i * frame_time, onto the grid (+ a +-6 frame lag search)
    U = {}
    for clip in UCLIPS:
        for r in ROLES:
            p = OD / "out" / f"{clip}_{r}.bvh"
            if not p.exists():
                continue
            meta = json.loads((OD / "inputs" / f"lucas_luiza_{clip}_{r}.json").read_text())
            names, pos, ft = read_bvh(p)
            J, used = bvh_to_smpl(names, pos)
            tq = meta["tau"][0] + np.arange(len(J)) * ft
            fr = ref.frames[clip]
            best = None
            for lag in range(-6, 7):
                x = (ref.tg[fr] - (tq[0] + lag / 30.0)) / ft
                i0 = np.clip(np.floor(x).astype(int), 0, len(J) - 2)
                w = np.clip(x - i0, 0, 1)[:, None, None]
                Jg = J[i0] * (1 - w) + J[i0 + 1] * w
                Jg[(x < 0) | (x > len(J) - 1)] = np.nan
                Y = ref.B0[r]["joints"][fr, :22]
                from .hmr_test import pa_mpjpe
                ok = np.isfinite(Jg[:, J15]).all((1, 2))
                if ok.sum() < 10:
                    continue
                e = float(np.median([pa_mpjpe(Jg[i, J15], Y[i, J15]) for i in np.flatnonzero(ok)[::3]]))
                if best is None or e < best[0]:
                    best = (e, lag, Jg)
            U.setdefault(r, np.full((ref.NG, 22, 3), np.nan))[fr] = best[2]
            res["bvh"][f"{clip}_{r}"] = {"frames": len(J), "frame_time": ft, "joints_mapped": {str(k): v for k, v in used.items()},
                                        "n_bvh_joints": len(names), "lag_frames": best[1]}
    meths["Uthana VTM 3.0"] = U
    for clip in UCLIPS:
        fr = ref.frames[clip]
        B = {r: ref.B0[r]["joints"][fr, :22].astype(float) for r in ROLES}
        res["clips"][clip] = {"note": CLIP_NOTE[clip], "frames": int(len(fr)), "methods": {}}
        ref_row = {}
        for r in ROLES:
            st = np.asarray(ref.B0[r]["stance"], bool)[fr]
            ref_row[r] = clip_metrics(B[r].copy(), B, r, st)
        res["clips"][clip]["methods"]["B0 two-view (reference)"] = {r: {k: v for k, v in ref_row[r].items() if k != "_W"} for r in ROLES}
        for m, D in meths.items():
            row, Wm = {}, {}
            for r in ROLES:
                if r not in D:
                    continue
                st = np.asarray(ref.B0[r]["stance"], bool)[fr]
                cm = clip_metrics(D[r][fr], B, r, st)
                if "_W" in cm:
                    Wm[r] = cm.pop("_W")
                row[r] = cm
            row["contact"] = contact_metrics(Wm, B)
            res["clips"][clip]["methods"][m] = row
        log(f"{clip}: " + "; ".join(f"{m}: " + ", ".join(f"{r} PA {v[r].get('pa_mm', float('nan')):.0f} skate {v[r].get('skate_cm_s') or float('nan'):.1f}"
                                               for r in ROLES if r in v) for m, v in res["clips"][clip]["methods"].items()))
    sd = OD / "score"
    sd.mkdir(parents=True, exist_ok=True)
    (sd / "score.json").write_text(json.dumps(res, indent=1, default=float))
    log(f"score: {sd / 'score.json'}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("job")
    p = sub.add_parser("inputs")
    p.add_argument("--sam", action="store_true", help="use sam/sam_<clip>.npz (the SAM 2.1 re-segmentation)")
    sub.add_parser("sam-prep")
    sub.add_parser("score")
    a = ap.parse_args(argv)
    {"job": cmd_job, "inputs": cmd_inputs, "sam-prep": cmd_sam_prep, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
