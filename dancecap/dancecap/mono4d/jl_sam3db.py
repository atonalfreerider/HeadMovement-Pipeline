r"""john_lula realism, stage 1 (2026-10-19): SAM 3D Body on ALL 2643 frames of the ONE handheld phone, both dancers,
as the per-frame 3D pose prior of the MAMMA_MONO refit (fit B + SAM 3D Body prior, docs/MONO4D_NOVEL_VIEWS.md 10.6).

Owner, 2026-10-19: "Yes: use skinned LHM++ splats posed by the MAMMA + SAM 3D Body fit for the john_lula tutorial video.
Run the full john_lula process now (SAM 3D Body on all frames, the prior refit, LHM++ with my whole photo mask, renders)".

The larissa_kadu_38 recipe (cand_sam3d_body: sam-3d-body b5c765a, checkpoint facebook/sam-3d-body-dinov3 @ 11aaa34,
variant mask255 = the pre-registered one) on the whole take, unchanged where it can be:
  - frames: the take's own frames/01/*.jpg (538 x 960, re-created by jl_lhmpp_preview; canonical index = video frame
    index = cameras/01.npz row), uploaded as they are (no copy in the input dir)
  - prompts: each dancer SEPARATELY with OUR SAM 2.1 mask (people/masks_01.npz, who 0 = John / lead, 1 = Lula /
    follow, the bib identity of J3a) as 0/255 + the tight box of that mask; no detector, no SAM 3, no FOV estimator
  - intrinsics: the PER-FRAME K of cameras/01.npz as cam_int (the phone zooms: f 806-1636 px)
  - a dancer is skipped on a frame whose mask is not usable (prep.json lists every one with its reason): mask missing
    or not decodable, area < MIN_MASK_PX, tight box shorter than MIN_BOX_H, or teachers_01.npz `visible` (own mask area
    / its local 90th percentile over +-2 s, zoom-compensated) < MIN_VISIBLE (the dancer is all but hidden)
  - MHR -> SMPL-X: Meta's official converter (facebookresearch/MHR @ 877cba5, pytorch path, per-frame identity) ON THE
    POD with --force-stub (pymomentum segfaults there); SMPL-X joints = forward kinematics of the fitted parameters
  - pod side: remote/cand_sam3d_body/sam3db_pod.py infer --chunk (resumable chunks of 200 frames, stops before the
    kill switch) / merge / convert; the vertices (~1.2 GB) stay on the pod unless the conversion fails
Clock: the identity clock (one camera: frames.json t = video pts = cameras/01.npz t to 5e-7 s; no sync delta).
World: x_cam = R X + t with cameras/01.npz's track (mamma_dyn.load_tracks; world_from_sfm.json = identity).

    python -m dancecap.mono4d.jl_sam3db check-access   # HEAD on the gated model.ckpt with the owner's token
    python -m dancecap.mono4d.jl_sam3db prep           # realism/sam3db/input/{meta.json, prep.json, smplx/}
    python -m dancecap.mono4d.jl_sam3db job            # remote/cand_sam3d_body/pod_job_jl.json
    (DANCECAP_RUNPOD_CAP=... python -m dancecap remote run --jobs <that json> --max-cost 3 --max-minutes 95 --gpus ...)
    python -m dancecap.mono4d.jl_sam3db import [--session SID]   # -> realism/sam3db/smplx_3db.npz + {lead,follow}.npz
    python -m dancecap.mono4d.jl_sam3db qa             # coverage, jitter, joints vs masks, identity, contact sheets
    python -m dancecap.mono4d.jl_sam3db prior          # realism/sam3db/prior_3db_full.npz on the fit lattice k / 30
    python -m dancecap.mono4d.jl_sam3db prior-slice --k0 0 --k1 1381 --out <chunk work>/mamma/prior_3db.npz

OUTPUTS (realism/sam3db/, likeness-derived, never committed):
    smplx_3db.npz   the form cand_fitb's prior writer reads (= pod_out/smplx_mask255.npz: frame_i, t, <role>_joints (F,
                    55, 3) camera frame, <role>_smplx_{betas, global_orient, body_pose, transl, left/right_hand_pose,
                    expression}, <role>_fit_err_cm) + <role>_valid, <role>_prompted, <role>_joints_world,
                    <role>_smplx_global_orient_world, <role>_{mask_px, visible, role_conf, skel_in_partner}, cam_ok,
                    and after `qa` <role>_qa_ok (valid and no identity flag)
    lead.npz, follow.npz   per dancer, plain names (joints_cam / joints_world, ...)
    qa.json, qa_frames.npz, sheets/contact_12.jpg, sheets/flagged.jpg
    prior_3db_full.npz      MAMMA_POSE_PRIOR form (cand_fitb.prior_interp) on the 2643 grid instants of smplx/
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

from ..take import PACKAGE_ROOT, WORKSPACE

TAKE, CAM = "john_lula", "01"
ROLES = ("lead", "follow")
WHO = {"lead": 0, "follow": 1}             # people/masks_01.npz / teachers_01.npz index (jl_avatars.WHO)
NAME = {"lead": "John", "follow": "Lula"}
ROLE_PID = {"lead": "01", "follow": "02"}
WORK = PACKAGE_ROOT / "work" / TAKE
REAL = WORK / "realism"
OUT = REAL / "sam3db"
INPUT = OUT / "input"
POD_OUT = OUT / "pod_out"
SHEETS = OUT / "sheets"
JOB_DIR = PACKAGE_ROOT / "remote" / "cand_sam3d_body"
JOB = JOB_DIR / "pod_job_jl.json"
VAR = "mask255"                            # the pre-registered recipe of cand_sam3d_body (candidate sam3d_body)
CHUNK = 200
MIN_MASK_PX = 1500                         # 538 x 960: a dancer's median mask is 23-29 k px (zoomed in up to 170 k)
MIN_BOX_H = 80
MIN_VISIBLE = 0.15
TRIMMED_SMPLX = PACKAGE_ROOT / "work" / "larissa_kadu_38" / "mono4d" / "sam3d_body" / "input" / "smplx" / "SMPLX_NEUTRAL.npz"
POD_ROOT = "/workspace/s3db_cand"
LICENCE = ("SAM 3D Body code + checkpoint: SAM License (Meta; gated on Hugging Face); DINOv3 code: DINOv3 licence; MHR "
           "+ MHR->SMPL-X converter: Apache-2.0; SMPL-X model: SMPL-X licence, non-commercial. Research use only; "
           "outputs and likeness data stay in work/ and our own pods.")
# QA thresholds (identity / failure flags)
DILATE_PX = 7
ON_PARTNER_SHARE, ON_PARTNER_OWN = 0.5, 0.3
DUP_2D, DUP_PELVIS_M = 0.06, 0.15
SWAP_RATIO, SWAP_MIN = 0.5, 0.12
FLIP_HALF, FLIP_DEG = 5, 60.0         # facing self-consistency: +-5 frames, > 60 deg off their mean = a flip
# MHR -> SMPL-X conversion failure: the converter's per-frame fit error (take median 1.06 cm, p99 1.26 / 1.38 cm). Audit
# 2026-10-19: lead frame 502 had 10.2 cm (a crumpled body, 0.10 body heights off SAM 3D Body's own 2D keypoints) and was
# qa_ok with prior conf 1.0; every other frame of both dancers is <= 2.34 cm
MAX_FIT_ERR_CM = 3.0
COCO_SMPLX = {5: 16, 6: 17, 7: 18, 8: 19, 9: 20, 10: 21, 13: 4, 14: 5, 15: 7, 16: 8}   # no hips (different definitions)
BONES = [(0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8), (6, 9), (7, 10), (8, 11), (9, 12), (12, 15),
         (9, 13), (9, 14), (13, 16), (14, 17), (16, 18), (17, 19), (18, 20), (19, 21)]


JOB_NAME = "jl_sam3d_body"
REF_SIZE = (538, 960)                      # the frame size MIN_MASK_PX / MIN_BOX_H were set for (john_lula)


def configure(take: str, cam: str = "01") -> None:
    """Point the stage at another one-camera take (pipeline/mono_route; the 2026-10-21 trial overrode these constants from a
    glue script): its work folder, outputs, pod job (work/<take>/jobs/pod_job_sam3db.json, job name <take>_sam3d_body) and
    the dancers' names from its TOML. Everything else (the recipe, thresholds) is the same."""
    import tomllib
    global TAKE, CAM, WORK, REAL, OUT, INPUT, POD_OUT, SHEETS, JOB_DIR, JOB, JOB_NAME, NAME
    TAKE, CAM = take, cam
    WORK = PACKAGE_ROOT / "work" / take
    REAL = WORK / "realism"
    OUT = REAL / "sam3db"
    INPUT = OUT / "input"
    POD_OUT = OUT / "pod_out"
    SHEETS = OUT / "sheets"
    JOB_DIR = WORK / "jobs"
    JOB = JOB_DIR / "pod_job_sam3db.json"
    JOB_NAME = f"{take}_sam3d_body"[:60]
    try:
        cfg = tomllib.loads((PACKAGE_ROOT / "takes" / f"{take}.toml").read_text(encoding="utf-8"))
        NAME = {r: str((cfg.get("dancers") or {}).get(r, {}).get("name") or r) for r in ROLES}
    except (OSError, ValueError):
        NAME = {r: r for r in ROLES}


def has_fit() -> bool:
    """The take has a first-pass MAMMA fit (smplx/ on the 30 Hz lattice): john_lula's realism route was a REFIT. A new take
    (pipeline analyze) has none yet, so qa() and prior() use their no-fit forms (qa_nofit, prior_nofit)."""
    return (WORK / "smplx" / "lead.npz").exists() and (WORK / "smplx" / "follow.npz").exists()


def size_rules(W: int, H: int) -> tuple[int, float]:
    """(min mask px, min box height px) for a W x H frame: john_lula's 1500 px / 80 px scaled by area / height."""
    return int(round(MIN_MASK_PX * (W * H) / (REF_SIZE[0] * REF_SIZE[1]))), float(MIN_BOX_H * H / REF_SIZE[1])


def log(msg: str) -> None:
    print(f"[jl_sam3db {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _ws(p: Path) -> str:
    import os   # no resolve(): a worktree reaches work/ through a junction, which resolve() would follow out of the workspace
    r = Path(os.path.relpath(os.path.abspath(p), os.path.abspath(WORKSPACE))).as_posix()
    if r.startswith(".."):
        raise ValueError(f"{p} is outside the workspace {WORKSPACE}")
    return r


def _rel(p: Path) -> str:                  # relative to <ws>/dancecap, the pod commands' cwd
    return _ws(p)[len("dancecap/"):]


def _decode(z, key: str):
    import cv2
    if key not in z.files:
        return None
    m = cv2.imdecode(np.frombuffer(z[key].tobytes(), np.uint8), cv2.IMREAD_GRAYSCALE)
    return None if m is None else m > 127


# ================================================================================================ prep + job
def prep(decode_on_pod: bool = False) -> dict:
    """realism/sam3db/input/{meta.json, prep.json, smplx/}. decode_on_pod: the frames are decoded on the pod from the video
    (job(video=...)), so the JPEGs need not exist here (frames.json does: its file names are what the pod writes)."""
    from .cand_sam3d_body import _mask_box
    fj = json.loads((WORK / "frames" / CAM / "frames.json").read_text())
    frames = fj["frames"]
    W, H = int(fj["width"]), int(fj["height"])
    cam = np.load(WORK / "cameras" / f"{CAM}.npz", allow_pickle=True)
    F = len(frames)
    t = np.array([f["t"] for f in frames], float)
    if len(cam["t"]) != F or np.abs(np.asarray(cam["t"], float) - t).max() > 1e-3:
        raise SystemExit("cameras/01.npz and frames.json are not index-aligned")
    if np.asarray(cam["frame_index"]).tolist() != list(range(F)):
        raise SystemExit("cameras/01.npz frame_index is not the video frame index")
    miss = [] if decode_on_pod else [f["file"] for f in frames if not (WORK / "frames" / CAM / f["file"]).exists()]
    if miss:
        raise SystemExit(f"{len(miss)} frames missing on disk, e.g. {miss[:3]} (jl_lhmpp_preview frames)")
    mz = np.load(WORK / "people" / f"masks_{CAM}.npz")
    with np.load(WORK / "people" / f"teachers_{CAM}.npz", allow_pickle=True) as tz:
        tp = {k: np.asarray(tz[k]) for k in ("visible", "role_conf", "skel_in_partner", "edge")}
    K = np.asarray(cam["K"], float)
    ok = np.asarray(cam["ok"], bool)
    zoom = np.asarray(cam["zoom_2x"], bool) if "zoom_2x" in cam.files else np.zeros(F, bool)
    min_px, min_box = size_rules(W, H)
    rows, skipped = [], {r: [] for r in ROLES}
    t0 = time.time()
    for i, fr in enumerate(frames):
        row = {"file": fr["file"], "i": i, "t": float(t[i]), "K": K[i].tolist(), "cam_ok": bool(ok[i]),
               "zoom_2x": bool(zoom[i]), "mask_boxes": {}, "mask_box_all": {}, "mask_px": {}, "skip": {}, "visible": {},
               "role_conf": {}, "skel_in_partner": {}}
        for role in ROLES:
            o = WHO[role]
            m = _decode(mz, f"m_{i}_{o}")
            box, px, reason = None, 0, None
            if m is None:
                reason = "no mask"
            elif m.shape != (H, W):
                reason = f"mask shape {m.shape} != {(H, W)}"
            else:
                px = int(m.sum())
                box = _mask_box(m)
            vis = float(tp["visible"][i, o])
            if reason is None:
                if box is None or px < min_px:
                    reason = f"area {px} px < {min_px}"
                elif box[3] - box[1] < min_box:
                    reason = f"box height {box[3] - box[1]:.0f} px < {min_box:.0f}"
                elif vis < MIN_VISIBLE:
                    reason = f"visible {vis:.3f} < {MIN_VISIBLE}"
            row["mask_boxes"][role] = None if reason else box
            row["mask_box_all"][role] = box
            row["mask_px"][role] = px
            row["skip"][role] = reason
            row["visible"][role] = round(vis, 4)
            row["role_conf"][role] = round(float(tp["role_conf"][i, o]), 4)
            row["skel_in_partner"][role] = round(float(tp["skel_in_partner"][i, o]), 4)
            if reason:
                skipped[role].append([i, round(float(t[i]), 3), reason])
        rows.append(row)
    meta = {"take": TAKE, "cam": CAM, "width": W, "height": H, "role_pid": ROLE_PID, "role_who": WHO,
            "mask_npz_key": "m_{i}_{who}", "clock": "identity (frames.json t = video pts = cameras/01.npz t)",
            "rules": {"min_mask_px": min_px, "min_box_h_px": round(min_box, 1), "min_visible": MIN_VISIBLE},
            "decode_on_pod": bool(decode_on_pod), "frames": rows}
    INPUT.mkdir(parents=True, exist_ok=True)
    (INPUT / "meta.json").write_text(json.dumps(meta), newline="\n")
    dst = INPUT / "smplx" / "SMPLX_NEUTRAL.npz"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(TRIMMED_SMPLX, dst)   # cand_sam3d_body._trim_smplx's file (checked against the full model there)
    f = np.array([K[i, 0, 0] for i in range(F)])
    info = {"frames": F, "size": [W, H], "t": [float(t[0]), float(t[-1])], "decode_s": round(time.time() - t0, 1),
            "focal_px": {"min": float(f.min()), "median": float(np.median(f)), "max": float(f.max())},
            "cam_not_ok_frames": int((~ok).sum()), "zoom_2x_frames": int(zoom.sum()),
            "prompted": {r: int(sum(1 for x in rows if x["mask_boxes"][r])) for r in ROLES},
            "skipped": {r: len(v) for r, v in skipped.items()},
            "skip_reasons": {r: {k: sum(1 for x in v if x[2].split(" ")[0] == k) for k in ("no", "mask", "area", "box",
                                                                                         "visible")} for r, v in skipped.items()},
            "skipped_frames": skipped, "rules": meta["rules"], "smplx_trimmed": {"from": str(TRIMMED_SMPLX),
                                                                               "bytes": dst.stat().st_size},
            "bytes_input_dir": int(sum(p.stat().st_size for p in INPUT.rglob("*") if p.is_file()))}
    (INPUT / "prep.json").write_text(json.dumps(info, indent=1), newline="\n")
    log(f"prep: {F} frames, prompted {info['prompted']}, skipped {info['skipped']}, focal {info['focal_px']}")
    return info


def job(max_minutes: float = 75, chunk: int = CHUNK, reserve_min: float = 30.0, video: Path | None = None,
        video_pod: str | None = None) -> Path:
    """The pod job. video: decode the frames ON THE POD from the source video (13 MB instead of 349 MB of JPEGs, audit rank 4;
    the 2026-10-21 trial's recipe), mapped to `video_pod` (workspace-relative); without it the take's frames/01 JPEGs are
    uploaded (john_lula's run). 4090 first (min_vram_gb 24): the default 80 GB list put the trial on an A100 at 2.0 s per
    person-frame against 0.355 on a 4090."""
    py = f"{POD_ROOT}/venv/bin/python"
    pod = "remote/cand_sam3d_body/sam3db_pod.py"
    io_ = f"--input {_rel(INPUT)} --out {_rel(POD_OUT)} --variants {VAR}"
    frames_dir = _rel(WORK / "frames" / CAM)
    masks = _rel(WORK / "people" / f"masks_{CAM}.npz")
    if video is not None:
        video_pod = video_pod or f"dances/{TAKE}/{Path(video).name}"
        frames_in = [{"local": str(Path(video).resolve()), "pod": video_pod}]
    else:
        frames_in = [_ws(WORK / "frames" / CAM) + "/"]
    d = {"schema": "dancecap.pod_job/1", "name": JOB_NAME, "take": TAKE, "max_minutes": max_minutes, "needs": [],
         "min_vram_gb": 24,
         "inputs": [_ws(INPUT) + "/", *frames_in, _ws(WORK / "people" / f"masks_{CAM}.npz")],
         "exclude": ["dancecap/work/runpod/pywheels", "dancecap/work/runpod/wheelhouse", "mamma-turing"],
         "secrets": {"HF_TOKEN": "C:/Users/johnb/Desktop/sam3d-body-key-hf.txt"},
         "env": {"S3DB_ROOT": POD_ROOT, "S3DB_COMMIT": "b5c765a0d89d789985e186d396315e7590887b94",
                 "S3DB_HF_REPO": "facebook/sam-3d-body-dinov3", "S3DB_HF_REV": "11aaa346c7204874a1cbafe3d39a979080b2c55a",
                 "DINOV3_COMMIT": "5aa7e93ab42ae3526291f6b4e3fcd22637e03326",
                 "MHR_COMMIT": "877cba5ee61a5f77d1457bf4de93a66695132bd9", "MHR_ASSETS_RELEASE": "v1.0.1",
                 "TORCH_HOME": f"{POD_ROOT}/torch_home", "HF_HOME": f"{POD_ROOT}/hf_home", "UV_LINK_MODE": "copy",
                 "TQDM_DISABLE": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1"},
         "commands": [
             {"name": "install", "timeout_min": 10, "cmd": "bash remote/cand_sam3d_body/install.sh"},
             # optional: a timeout / the kill-switch reserve leaves the finished chunks, which merge + convert still use
             {"name": "infer", "timeout_min": 50, "optional": True,
              "cmd": f"{py} {pod} infer {io_} --chunk {chunk} --frames-dir {frames_dir} --masks-npz {masks} "
                     f"--reserve-min {reserve_min:g}"},
             {"name": "merge", "timeout_min": 5, "cmd": f"{py} {pod} merge {io_} --chunk {chunk}"},
             # the vertices leave pod_out only when the conversion succeeded (else they come home for a local convert)
             {"name": "convert", "timeout_min": 35,
              "cmd": f"{py} {pod} convert {io_} --force-stub && mkdir -p {POD_ROOT}/chunks_done && "
                     f"mv {_rel(POD_OUT)}/chunks {POD_ROOT}/chunks_done/{TAKE}_$(date +%s)"}],
         "outputs": [_ws(POD_OUT) + "/"],
         "verify": [_ws(POD_OUT / "sam3db_pod.json"), _ws(POD_OUT / f"mhr_{VAR}.npz"), _ws(POD_OUT / f"smplx_{VAR}.npz")],
         "after_download": ["python -m dancecap.mono4d.jl_sam3db import --session <sid>",
                            "python -m dancecap.mono4d.jl_sam3db qa", "python -m dancecap.mono4d.jl_sam3db prior"],
         "notes": ["SAM 3D Body (SAM License, gated HF checkpoint downloaded on the pod with the owner's HF token) on ALL "
                   "2643 frames of john_lula's one phone (538x960), each dancer prompted with OUR SAM 2.1 mask (0/255) + "
                   "its tight box and the per-frame K of cameras/01.npz; official MHR->SMPL-X converter (pytorch, "
                   "--force-stub) on the pod. Checkpoints stay on the pod; the dancers' frames go to our own pod only "
                   "(owner consent 2026-10-19); the bundle holds a trimmed SMPL-X neutral model (owner's licence)."]}
    if video is not None:   # decode the frames the input lists (meta.json file names, in decode order) from the video
        d["commands"].insert(1, {"name": "decode", "timeout_min": 10,
                                 "cmd": f"{py} remote/cand_sam3d_body/decode_frames.py ../{video_pod} {_rel(INPUT)}/meta.json "
                                        f"{frames_dir}"})
        d["after_download"] = [f"python -m dancecap.mono4d.jl_sam3db --take {TAKE} {c}" for c in ("import", "qa", "prior")]
        d["notes"] = [f"SAM 3D Body on the {TAKE} frames (decoded on the pod from the source video), each dancer prompted with "
                      "the people stage's SAM 2.1 mask (0/255) + its tight box and the per-frame K of cameras/01.npz; as "
                      "jl_sam3d_body otherwise (pipeline/mono_route)"]
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    JOB.write_text(json.dumps(d, indent=1), newline="\n")
    log(f"pod job -> {JOB}")
    return JOB


# ================================================================================================ import
def _session_meta(sid: str | None) -> dict:
    root = PACKAGE_ROOT / "work" / "runpod" / "sessions"
    if sid is None:
        cands = sorted((p for p in root.glob("*") if (p / "result.json").exists()
                        and f'"{JOB_NAME}"' in (p / "result.json").read_text(errors="ignore")),
                       key=lambda p: p.stat().st_mtime)
        if not cands:
            return {}
        sid = cands[-1].name
    p = root / sid / "result.json"
    if not p.exists():
        return {"session": sid}
    r = json.loads(p.read_text())
    pod = r.get("pod") or {}
    jb = (r.get("jobs") or {}).get(JOB_NAME) or {}
    return {"session": sid, "gpu": pod.get("gpu"), "cost_per_hr": pod.get("cost_per_hr"),
            "cost_usd": pod.get("ledger_cost"), "pod_minutes": round((pod.get("lifetime_s") or 0) / 60, 1),
            "job_status": jb.get("status"), "commands_rc": jb.get("commands"), "job_seconds": jb.get("seconds"),
            "download_mb": jb.get("download_mb")}


def do_import(sid: str | None = None, fk_sample: int = 300) -> dict:
    from .. import mamma_dyn as md
    from . import candidates as C
    from .cand_fitb import _fk_joints, orient_to_world
    meta = json.loads((INPUT / "meta.json").read_text())
    rows = meta["frames"]
    F = len(rows)
    pod = json.loads((POD_OUT / "sam3db_pod.json").read_text())
    z = np.load(POD_OUT / f"smplx_{VAR}.npz", allow_pickle=True)
    mh = np.load(POD_OUT / f"mhr_{VAR}.npz", allow_pickle=True)
    fi = [r["i"] for r in rows]
    for name, a in (("smplx", z), ("mhr", mh)):
        if list(a["frame_i"]) != fi:
            raise SystemExit(f"{name}_{VAR}.npz: frame indices differ from input/meta.json")
    tr = md.load_tracks(WORK, [CAM])[CAM]
    t = np.array([r["t"] for r in rows], float)
    clock_err = float(np.abs(tr.t - t).max())
    if clock_err > 1e-3:
        raise SystemExit(f"the track's t differs from the frames' by {clock_err:.2e} s")
    K = np.array([r["K"] for r in rows], float)
    k_err = float(np.abs(K - tr.K).max())
    f_err = float(np.nanmax(np.abs(np.asarray(mh["lead_focal_length"], float) - K[:, 0, 0]))) \
        if "lead_focal_length" in mh.files else None
    ses = _session_meta(sid)
    arrays = {k: np.asarray(z[k]) for k in z.files}
    arrays["t"] = np.asarray(tr.t, float)
    arrays["cam_ok"] = np.asarray(tr.ok, bool)
    rep = {"session": ses, "clock_max_err_s": clock_err, "K_vs_track_max_abs": k_err,
           "focal_used_vs_K_max_abs_px": f_err, "pod": {k: pod.get(k) for k in ("model", "variants", "merge")},
           "convert": {k: v for k, v in (pod.get("convert") or {}).items() if k != VAR},
           "convert_failed": pod.get("convert_failed"), "roles": {}}
    rng = np.random.default_rng(0)
    for role in ROLES:
        J = np.asarray(z[f"{role}_joints"], float)
        valid = np.asarray(mh[f"{role}_valid"], bool) & np.isfinite(J[:, :22]).all((1, 2))
        prompted = np.array([bool(r["mask_boxes"][role]) for r in rows])
        Jw = C.to_world(J, tr.R, tr.tvec)
        gcam = np.asarray(z[f"{role}_smplx_global_orient"], float)
        gw = np.full_like(gcam, np.nan)
        gw[valid] = orient_to_world(gcam[valid], tr.R[valid])
        arrays[f"{role}_valid"] = valid
        arrays[f"{role}_prompted"] = prompted
        arrays[f"{role}_joints_world"] = np.where(valid[:, None, None], Jw, np.nan).astype(np.float32)
        arrays[f"{role}_smplx_global_orient_world"] = gw.astype(np.float32)
        for k in ("mask_px", "visible", "role_conf", "skel_in_partner"):
            arrays[f"{role}_{k}"] = np.array([r[k][role] for r in rows], float)
        # the converter's parameters must reproduce its joints (camera frame); the world orientation the world joints
        ks = np.flatnonzero(valid)
        ks = np.sort(rng.choice(ks, min(fk_sample, len(ks)), replace=False)) if len(ks) else ks
        g = lambda n: np.asarray(z[f"{role}_smplx_{n}"], float)[ks]  # noqa: E731
        chk = {}
        if len(ks):
            fk_c = _fk_joints(g("betas"), gcam[ks], g("body_pose"), g("transl"))
            fk_w = _fk_joints(g("betas"), gw[ks], g("body_pose"), np.zeros((len(ks), 3)))
            chk = {"frames_checked": int(len(ks)),
                   "fk_vs_joints_camera_mm_max": round(1e3 * float(np.abs(fk_c - J[ks, :22]).max()), 3),
                   "fk_world_orient_vs_world_joints_rr_mm_max": round(1e3 * float(np.abs(
                       (fk_w - fk_w[:, :1]) - (Jw[ks, :22] - Jw[ks, :1])).max()), 3)}
        err = np.asarray(z[f"{role}_fit_err_cm"], float)
        rep["roles"][role] = {"frames": F, "prompted": int(prompted.sum()), "valid": int(valid.sum()),
                              "prompted_but_no_output": int((prompted & ~valid).sum()),
                              "fit_err_cm": {"median": float(np.nanmedian(err[valid])) if valid.any() else None,
                                             "p90": float(np.nanpercentile(err[valid], 90)) if valid.any() else None},
                              "checks": chk}
        # per dancer, plain names
        dz = {"t": arrays["t"], "frame_i": np.asarray(fi), "valid": valid, "prompted": prompted, "cam_ok": arrays["cam_ok"],
              "joints_cam": J.astype(np.float32), "joints_world": arrays[f"{role}_joints_world"],
              "global_orient_cam": gcam.astype(np.float32), "global_orient_world": arrays[f"{role}_smplx_global_orient_world"],
              "fit_err_cm": err.astype(np.float32), "frame": np.array("camera (OpenCV, phone 01) + world (CONTRACTS)"),
              "clock": np.array("identity clock: t = video pts (frames.json) = cameras/01.npz t"),
              "model": np.array("SMPL-X 2020 neutral, 10 betas, 10 expr, flat_hand_mean, use_pca False (the converter's)")}
        for n in ("betas", "body_pose", "transl", "left_hand_pose", "right_hand_pose", "expression"):
            if f"{role}_smplx_{n}" in z.files:
                dz["transl_cam" if n == "transl" else n] = np.asarray(z[f"{role}_smplx_{n}"], np.float32)
        for n in ("pred_cam_t", "focal_length", "pred_keypoints_2d", "bbox"):
            if f"{role}_{n}" in mh.files:
                dz[f"mhr_{n}"] = np.asarray(mh[f"{role}_{n}"], np.float32)
        for k in ("mask_px", "visible", "role_conf", "skel_in_partner"):
            dz[k] = arrays[f"{role}_{k}"]
        np.savez_compressed(OUT / f"{role}.npz", **dz)
        log(f"{role}: prompted {int(prompted.sum())}, valid {int(valid.sum())}, fit err {rep['roles'][role]['fit_err_cm']}, "
            f"checks {chk}")
    np.savez_compressed(OUT / "smplx_3db.npz", **arrays)
    meta_out = {"method": "SAM 3D Body (facebook/sam-3d-body-dinov3 @ 11aaa34, sam-3d-body b5c765a) per frame and dancer, "
                          "prompted with our SAM 2.1 mask (0/255) + its tight box, per-frame K of cameras/01.npz; MHR -> "
                          "SMPL-X by the official converter (MHR 877cba5, pytorch, per-frame identity, stub faces)",
                "licence": LICENCE, "variant": VAR, "take": TAKE, "cam": CAM, "rules": meta["rules"],
                "converter": (pod.get("convert") or {}).get("mhr_model"), **ses}
    (OUT / "meta.json").write_text(json.dumps(meta_out, indent=1, default=str))
    (OUT / "import.json").write_text(json.dumps(rep, indent=1, default=str))
    return rep


# ================================================================================================ QA
def _project(P: np.ndarray, K: np.ndarray) -> np.ndarray:
    """(J, 3) camera-frame points -> (J, 2) pixels; NaN behind the camera."""
    P = np.asarray(P, float)
    zz = np.where(P[:, 2:3] > 1e-3, P[:, 2:3], np.nan)
    return (K @ (P / zz).T).T[:, :2]


def _mamma_on_frames(t_frames: np.ndarray, folder: str = "smplx") -> dict:
    """The take's MAMMA fit (30 Hz lattice) linearly interpolated to the frame times: {role: (F, 22, 3) world, absent}."""
    out = {}
    for role in ROLES:
        z = np.load(WORK / folder / f"{role}.npz", allow_pickle=True)
        tg, J = np.asarray(z["t"], float), np.asarray(z["joints"], float)[:, :22]
        a = np.clip(np.searchsorted(tg, t_frames, side="right") - 1, 0, len(tg) - 2)
        w = np.clip((t_frames - tg[a]) / (tg[a + 1] - tg[a]), 0, 1)[:, None, None]
        absent = np.asarray(z["absent"], bool) if "absent" in z.files else np.zeros(len(tg), bool)
        out[role] = {"joints": (1 - w) * J[a] + w * J[a + 1], "absent": absent[a] | absent[a + 1], "t": tg, "J": J}
    return out


def _pa(A: np.ndarray, B: np.ndarray) -> tuple[float, float]:
    """PA-MPJPE (mm) of A aligned to B by a similarity, and the scale applied to A."""
    A, B = A - A.mean(0), B - B.mean(0)
    U, S, Vt = np.linalg.svd(A.T @ B)
    d = np.sign(np.linalg.det(U @ Vt))
    Rm = U @ np.diag([1, 1, d]) @ Vt
    s = (S * [1, 1, d]).sum() / max((A ** 2).sum(), 1e-12)
    return 1e3 * float(np.linalg.norm(s * A @ Rm - B, axis=-1).mean()), float(s)


def facing(X: np.ndarray) -> np.ndarray:
    """(F, >=22, 3) world joints (Y up) -> (F, 3) horizontal unit forward of the torso (hips + shoulders x spine)."""
    lr = (X[:, 1] - X[:, 2]) + (X[:, 16] - X[:, 17])
    f = np.cross(lr, X[:, 12] - X[:, 0])
    f[:, 1] = 0.0
    n = np.linalg.norm(f, axis=1, keepdims=True)
    return np.where(n > 1e-9, f / np.where(n > 1e-9, n, 1.0), np.nan)


def facing_flips(fw: np.ndarray, valid: np.ndarray, half: int = FLIP_HALF) -> np.ndarray:
    """(F,) degrees between each valid frame's facing and the circular mean of its valid neighbours within +-half frames
    (itself excluded; NaN with < 4 neighbours): a smooth spin stays small, an isolated front/back flip is ~180."""
    F = len(fw)
    dev = np.full(F, np.nan)
    for i in np.flatnonzero(valid):
        nb = [j for j in range(max(0, i - half), min(F, i + half + 1)) if j != i and valid[j] and np.isfinite(fw[j]).all()]
        if len(nb) < 4 or not np.isfinite(fw[i]).all():
            continue
        m = fw[nb].sum(0)
        m /= np.linalg.norm(m) + 1e-12
        dev[i] = np.degrees(np.arccos(np.clip(float(fw[i] @ m), -1.0, 1.0)))
    return dev


def _stats(x) -> dict:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0}
    return {"n": int(len(x)), "median": round(float(np.median(x)), 3), "p10": round(float(np.percentile(x, 10)), 3),
            "p90": round(float(np.percentile(x, 90)), 3), "max": round(float(x.max()), 3)}


def tpose_heights(betas: np.ndarray, batch: int = 256) -> np.ndarray:
    """(F, 10) SMPL-X neutral betas -> (F,) rest height in metres (crown to sole, zero pose; NaN rows stay NaN)."""
    import torch
    import smplx
    b = np.asarray(betas, np.float32)
    ok = np.isfinite(b).all(1)
    out = np.full(len(b), np.nan)
    m = smplx.SMPLX(model_path=str(WORKSPACE / "smplx" / "SMPLX_NEUTRAL.npz"), gender="neutral", num_betas=10,
                    use_pca=False, flat_hand_mean=True)
    idx = np.flatnonzero(ok)
    for s0 in range(0, len(idx), batch):
        k = idx[s0:s0 + batch]
        z = lambda n: torch.zeros(len(k), n)  # noqa: E731
        with torch.no_grad():
            v = m(betas=torch.as_tensor(b[k]), global_orient=z(3), body_pose=z(63), left_hand_pose=z(45),
                  right_hand_pose=z(45), jaw_pose=z(3), leye_pose=z(3), reye_pose=z(3), expression=z(10),
                  transl=z(3)).vertices.numpy()
        out[k] = v[:, :, 1].max(1) - v[:, :, 1].min(1)
    return out


def qa(sheets: bool = True) -> dict:
    if not has_fit():
        return qa_nofit()
    import cv2
    from . import candidates as C
    from .. import mamma_dyn as md
    s = np.load(OUT / "smplx_3db.npz", allow_pickle=True)
    meta = json.loads((INPUT / "meta.json").read_text())
    rows = meta["frames"]
    F = len(rows)
    t = np.asarray(s["t"], float)
    K = np.array([r["K"] for r in rows], float)
    tr = md.load_tracks(WORK, [CAM])[CAM]
    mz = np.load(WORK / "people" / f"masks_{CAM}.npz")
    with np.load(WORK / "people" / f"teachers_{CAM}.npz", allow_pickle=True) as tz:
        kp = np.asarray(tz["kp"], float)                                     # (F, 2, 17, 3) COCO-17, per role
    mam = _mamma_on_frames(t, "smplx")
    Jc = {r: np.asarray(s[f"{r}_joints"], float)[:, :22] for r in ROLES}
    Jw = {r: np.asarray(s[f"{r}_joints_world"], float)[:, :22] for r in ROLES}
    valid = {r: np.asarray(s[f"{r}_valid"], bool) for r in ROLES}
    prompted = {r: np.asarray(s[f"{r}_prompted"], bool) for r in ROLES}
    keys = ("own", "partner", "outside", "teach_own", "teach_other", "mam_same", "mam_other", "rr_mam_mm", "pa_mam_mm",
            "scale_vs_mam", "depth_ratio", "pelvis_xz_mm", "box_h", "mam_own")
    Q = {r: {k: np.full(F, np.nan) for k in keys} for r in ROLES}
    dup2d, dup_pel = np.full(F, np.nan), np.full(F, np.nan)
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * DILATE_PX + 1, 2 * DILATE_PX + 1))
    W, H = meta["width"], meta["height"]
    t0 = time.time()
    uv_all = {r: np.full((F, 22, 2), np.nan) for r in ROLES}
    for i in range(F):
        ms = {r: _decode(mz, f"m_{i}_{WHO[r]}") for r in ROLES}
        dil = {r: (cv2.dilate(ms[r].astype(np.uint8), kern) > 0) if ms[r] is not None else None for r in ROLES}
        # MAMMA's joints in this frame's camera
        mcam = {r: (tr.R[i] @ mam[r]["joints"][i].T).T + tr.tvec[i] for r in ROLES}
        muv = {r: _project(mcam[r], K[i]) for r in ROLES}
        for r in ROLES:
            m2 = muv[r]
            insm = np.isfinite(m2).all(1) & (m2[:, 0] >= 0) & (m2[:, 0] < W) & (m2[:, 1] >= 0) & (m2[:, 1] < H)
            if not mam[r]["absent"][i] and insm.any() and dil[r] is not None:   # the MAMMA fit's own share: a baseline
                Q[r]["mam_own"][i] = dil[r][m2[insm, 1].astype(int), m2[insm, 0].astype(int)].mean()
            if not valid[r][i]:
                continue
            o = "follow" if r == "lead" else "lead"
            uv = _project(Jc[r][i], K[i])
            uv_all[r][i] = uv
            inside = np.isfinite(uv).all(1) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
            b = rows[i]["mask_box_all"][r]
            h = float(b[3] - b[1]) if b else np.nan
            Q[r]["box_h"][i] = h
            if inside.any() and dil[r] is not None:
                u, v = uv[inside, 0].astype(int), uv[inside, 1].astype(int)
                own = dil[r][v, u]
                par = (ms[o][v, u] & ~own) if ms[o] is not None else np.zeros(len(u), bool)
                Q[r]["own"][i] = own.mean()
                Q[r]["partner"][i] = par.mean()
                Q[r]["outside"][i] = (~own & ~par).mean()
            # teachers (RTMW COCO-17, matched to the role by the mask): own role vs the other role
            d = {}
            for rr_, key in ((r, "teach_own"), (o, "teach_other")):
                kk = kp[i, WHO[rr_]]
                js = [(c, sj) for c, sj in COCO_SMPLX.items() if kk[c, 2] >= 0.5 and np.isfinite(uv[sj]).all()]
                if len(js) >= 4 and np.isfinite(h):
                    Q[r][key][i] = np.mean([np.linalg.norm(uv[sj] - kk[c, :2]) for c, sj in js]) / h
            for rr_, key in ((r, "mam_same"), (o, "mam_other")):
                m2 = muv[rr_]
                okj = np.isfinite(m2).all(1) & np.isfinite(uv).all(1)
                if okj.sum() >= 10 and np.isfinite(h):
                    Q[r][key][i] = np.linalg.norm(uv[okj] - m2[okj], axis=1).mean() / h
            if not mam[r]["absent"][i]:
                A, B = Jw[r][i], mam[r]["joints"][i]
                Q[r]["rr_mam_mm"][i] = 1e3 * np.linalg.norm((A - A[:1]) - (B - B[:1]), axis=-1)[1:].mean()
                Q[r]["pa_mam_mm"][i], Q[r]["scale_vs_mam"][i] = _pa(A, B)
                Q[r]["depth_ratio"][i] = Jc[r][i, 0, 2] / max(mcam[r][0, 2], 1e-6)
                Q[r]["pelvis_xz_mm"][i] = 1e3 * np.linalg.norm((A[0] - B[0])[[0, 2]])
        if valid["lead"][i] and valid["follow"][i]:
            a, b2 = uv_all["lead"][i], uv_all["follow"][i]
            hh = np.nanmean([Q["lead"]["box_h"][i], Q["follow"]["box_h"][i]])
            okj = np.isfinite(a).all(1) & np.isfinite(b2).all(1)
            if okj.any():
                dup2d[i] = np.linalg.norm(a[okj] - b2[okj], axis=1).mean() / hh
            dup_pel[i] = np.linalg.norm(Jc["lead"][i, 0] - Jc["follow"][i, 0])
        if i % 500 == 0:
            log(f"qa: frame {i}/{F} ({time.time() - t0:.0f} s)")
    dup = np.nan_to_num(dup2d, nan=9) < DUP_2D
    dup &= np.nan_to_num(dup_pel, nan=9) < DUP_PELVIS_M
    flags, rep = {}, {"thresholds": {"dilate_px": DILATE_PX, "on_partner": [ON_PARTNER_SHARE, ON_PARTNER_OWN],
                                     "duplicate": [DUP_2D, DUP_PELVIS_M], "swap": [SWAP_RATIO, SWAP_MIN]},
                      "roles": {}}
    # identity: a SWAP is a skeleton that matches the OTHER dancer's 2D evidence much better than its own, by BOTH
    # references (RTMW teacher keypoints of the role and the take's MAMMA fit; either alone can itself lie on the
    # partner when the dancer is hidden). on_partner (joints over the partner's mask) is reported only: a dancer
    # behind the partner projects there legitimately (the MAMMA fit shows the same, `mamma_..._own_mask_share`).
    # A DUPLICATE frame (both prompts gave the same body) is charged to the dancer whose skeleton sits on the other's
    # evidence, else to both.
    on_other = {}
    for r in ROLES:
        q = Q[r]
        on_other[r] = (q["teach_other"] < q["teach_own"]) | (q["mam_other"] < q["mam_same"])
    for r in ROLES:
        q = Q[r]
        o = "follow" if r == "lead" else "lead"
        on_partner = (np.nan_to_num(q["partner"]) >= ON_PARTNER_SHARE) & (np.nan_to_num(q["own"], nan=1) < ON_PARTNER_OWN)
        t_swap = (q["teach_other"] < SWAP_RATIO * q["teach_own"]) & (q["teach_own"] > SWAP_MIN)
        m_swap = (q["mam_other"] < SWAP_RATIO * q["mam_same"]) & (q["mam_same"] > SWAP_MIN)
        swap = t_swap & m_swap
        dup_r = dup & (on_other[r] | ~on_other[o])
        fw = facing(np.nan_to_num(Jw[r]))
        dev = facing_flips(fw, valid[r])
        flip = np.nan_to_num(dev) > FLIP_DEG
        fm = facing(mam[r]["joints"])
        dfm = np.degrees(np.arccos(np.clip((fw * fm).sum(1), -1, 1)))
        Q[r]["facing_dev_deg"], Q[r]["facing_vs_mamma_deg"] = dev, np.where(valid[r] & ~mam[r]["absent"], dfm, np.nan)
        conv = valid[r] & (np.nan_to_num(np.asarray(s[f"{r}_fit_err_cm"], float), nan=99.0) > MAX_FIT_ERR_CM)
        bad = valid[r] & (swap | dup_r | flip | conv)
        flags[r] = {"on_partner": on_partner & valid[r], "teacher_swap": t_swap & valid[r], "mamma_swap": m_swap & valid[r],
                    "swap": swap & valid[r], "duplicate": dup_r & valid[r], "facing_flip": flip & valid[r],
                    "convert": conv, "bad": bad}
        acc = C.accel_rr_mm(t, np.where(valid[r][:, None, None], Jw[r], np.nan))
        accm = C.accel_rr_mm(mam[r]["t"], mam[r]["J"])
        rf = np.load(WORK / "smplx_rf2_noslip" / f"{r}.npz")
        accr = C.accel_rr_mm(np.asarray(rf["t"], float), np.asarray(rf["joints"], float)[:, :22])
        gaps = np.diff(np.r_[0, (~valid[r]).astype(int), 0])
        runs = np.flatnonzero(gaps == -1) - np.flatnonzero(gaps == 1)
        sec = np.floor(t).astype(int)
        per_s = [round(float(valid[r][sec == k].mean()), 2) for k in range(int(sec.max()) + 1)]
        hgt = tpose_heights(np.where(valid[r][:, None], np.asarray(s[f"{r}_smplx_betas"], float), np.nan))
        rep["roles"][r] = {
            "name": NAME[r], "frames": F, "prompted": int(prompted[r].sum()), "valid": int(valid[r].sum()),
            "tpose_height_m (SAM 3D Body betas; owner's stature lead 1.803 / follow 1.702 barefoot)": _stats(hgt),
            "coverage": round(float(valid[r].mean()), 4), "longest_gap_frames": int(runs.max()) if len(runs) else 0,
            "seconds_below_80pct": [k for k, v in enumerate(per_s) if v < 0.8],
            "jitter_accel_rr_mm_per_frame2": {"sam3db_raw_median": round(float(np.nanmedian(acc)), 2),
                                             "mamma_smplx_median": round(float(np.nanmedian(accm)), 2),
                                             "mamma_smplx_rf2_noslip_median": round(float(np.nanmedian(accr)), 2)},
            "joints_in_own_mask_share": _stats(q["own"][valid[r]]),
            "mamma_smplx_joints_in_own_mask_share (baseline)": _stats(q["mam_own"]),
            "joints_on_partner_only_share": _stats(q["partner"][valid[r]]),
            "joints_outside_both_share": _stats(q["outside"][valid[r]]),
            "frames_own_share_below_0.5": int((q["own"][valid[r]] < 0.5).sum()),
            "teacher_2d_err_over_height": {"own": _stats(q["teach_own"]), "other": _stats(q["teach_other"])},
            "mamma_2d_dist_over_height": {"same_role": _stats(q["mam_same"]), "other_role": _stats(q["mam_other"])},
            "vs_mamma_3d (agreement, not ground truth)": {"rr_mm": _stats(q["rr_mam_mm"]), "pa22_mm": _stats(q["pa_mam_mm"]),
                                                         "procrustes_scale": _stats(q["scale_vs_mam"]),
                                                         "pelvis_depth_ratio_3db_over_mamma": _stats(q["depth_ratio"]),
                                                         "pelvis_floor_dist_mm": _stats(q["pelvis_xz_mm"])},
            "facing": {"self_consistency_dev_deg": _stats(q["facing_dev_deg"]),
                       "vs_mamma_deg (agreement, not ground truth)": _stats(q["facing_vs_mamma_deg"]),
                       "vs_mamma_over_90deg_frames": int((q["facing_vs_mamma_deg"] > 90).sum()),
                       "vs_mamma_over_90deg_by_visible": {f"{lo}-{hi}": [int(((q["facing_vs_mamma_deg"] > 90) & m).sum()),
                                                                          int((np.isfinite(q["facing_vs_mamma_deg"]) & m).sum())]
                                                          for lo, hi in ((0, .3), (.3, .5), (.5, .8), (.8, 9))
                                                          for m in [(np.asarray(s[f"{r}_visible"]) >= lo) & (np.asarray(s[f"{r}_visible"]) < hi)]},
                       "vs_mamma_over_90deg_frame_list": np.flatnonzero(q["facing_vs_mamma_deg"] > 90).tolist()},
            "flags": {k: int(v.sum()) for k, v in flags[r].items()},
            "flag_frames": {k: np.flatnonzero(v).tolist()[:200] for k, v in flags[r].items() if k != "bad"},
            "flag_share_of_valid": round(float(flags[r]["bad"].sum() / max(valid[r].sum(), 1)), 4)}
        log(f"{r}: {json.dumps({k: rep['roles'][r][k] for k in ('valid', 'coverage', 'flags', 'jitter_accel_rr_mm_per_frame2')})}")
    rep["duplicate_frames"] = np.flatnonzero(dup).tolist()
    np.savez_compressed(OUT / "qa_frames.npz", **{f"{r}_{k}": v for r in ROLES for k, v in Q[r].items()},
                        **{f"{r}_flag_{k}": v for r in ROLES for k, v in flags[r].items()}, dup_2d=dup2d, dup_pelvis_m=dup_pel)
    # the identity flag joins the prior-writer file (prior_3db_full reads <role>_qa_ok)
    arrays = {k: np.asarray(s[k]) for k in s.files}
    for r in ROLES:
        arrays[f"{r}_qa_ok"] = valid[r] & ~flags[r]["bad"]
    np.savez_compressed(OUT / "smplx_3db.npz", **arrays)
    worst = max(rep["roles"][r]["flag_share_of_valid"] for r in ROLES)
    rep["verdict"] = {"max_flag_share": worst,
                      "poisons_prior": bool(worst > 0.05 or min(rep["roles"][r]["coverage"] for r in ROLES) < 0.8),
                      "rule": "go_on false when > 5 % of a dancer's valid frames carry an identity / duplicate / "
                              "facing-flip / conversion (fit err > MAX_FIT_ERR_CM) flag or coverage < 80 %; flagged frames "
                              "get no prior (qa_ok)", "max_fit_err_cm": MAX_FIT_ERR_CM}
    (OUT / "qa.json").write_text(json.dumps(rep, indent=1, default=str))
    if sheets:
        contact_sheet(s, rows, uv_all, valid, Q, flags)
    return rep


def _tile(i: int, rows: list, uv_all: dict, valid: dict, label: str, mz, size=(380, 560)) -> np.ndarray:
    import cv2
    img = cv2.imread(str(WORK / "frames" / CAM / rows[i]["file"]))
    col_mask = {"lead": (60, 220, 60), "follow": (230, 60, 230)}
    col_sk = {"lead": (0, 140, 255), "follow": (255, 220, 0)}
    S = 3                                                         # draw at 3x for legible lines on a 538 x 960 frame
    big = cv2.resize(img, None, fx=S, fy=S, interpolation=cv2.INTER_CUBIC)
    for r in ROLES:
        m = _decode(mz, f"m_{i}_{WHO[r]}")
        if m is not None:
            cs, _ = cv2.findContours(cv2.resize(m.astype(np.uint8), None, fx=S, fy=S, interpolation=cv2.INTER_NEAREST),
                                     cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(big, cs, -1, col_mask[r], 3, cv2.LINE_AA)
    for r in ROLES:
        if not valid[r][i]:
            continue
        uv = uv_all[r][i] * S
        for a, b in BONES:
            if np.isfinite(uv[[a, b]]).all():
                cv2.line(big, tuple(int(x) for x in uv[a]), tuple(int(x) for x in uv[b]), col_sk[r], 5, cv2.LINE_AA)
        for p in uv[np.isfinite(uv).all(1)]:
            cv2.circle(big, tuple(int(x) for x in p), 6, (255, 255, 255), -1, cv2.LINE_AA)
    bx = np.array([b for b in rows[i]["mask_box_all"].values() if b], float) * S
    cx, cy = (bx[:, 0].min() + bx[:, 2].max()) / 2, (bx[:, 1].min() + bx[:, 3].max()) / 2
    hh = max(bx[:, 3].max() - bx[:, 1].min(), (bx[:, 2].max() - bx[:, 0].min()) * size[1] / size[0]) * 1.15
    ww = hh * size[0] / size[1]
    x0, y0 = int(np.clip(cx - ww / 2, 0, max(big.shape[1] - ww, 0))), int(np.clip(cy - hh / 2, 0, max(big.shape[0] - hh, 0)))
    tile = big[y0:int(y0 + hh), x0:int(x0 + ww)]
    tile = cv2.resize(tile, size, interpolation=cv2.INTER_AREA)
    for k, line in enumerate(label.split("\n")):
        cv2.putText(tile, line, (8, 24 + 22 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(tile, line, (8, 24 + 22 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return tile


def _grid(tiles: list, cols: int = 6) -> np.ndarray:
    rows = []
    for k in range(0, len(tiles), cols):
        r = tiles[k:k + cols]
        r += [np.zeros_like(tiles[0])] * (cols - len(r))
        rows.append(np.concatenate(r, 1))
    return np.concatenate(rows, 0)


def contact_sheet(s, rows, uv_all, valid, Q, flags) -> list[Path]:
    """12 frames spread over the dance (both dancers valid), and up to 12 flagged / worst frames. Masks: John green,
    Lula magenta (outlines); SAM 3D Body skeletons: John orange, Lula cyan."""
    import cv2
    mz = np.load(WORK / "people" / f"masks_{CAM}.npz")
    t = np.asarray(s["t"], float)
    both = valid["lead"] & valid["follow"]
    picks = []
    for tt in np.linspace(2.0, 86.0, 12):
        cand = np.flatnonzero(both)
        picks.append(int(cand[np.argmin(np.abs(t[cand] - tt))]))

    def lab(i, extra=""):
        own = "/".join(f"{Q[r]['own'][i]:.2f}" if np.isfinite(Q[r]['own'][i]) else "-" for r in ROLES)
        return f"f{i} t={t[i]:.2f}s{extra}\nin own mask J/L {own}"

    SHEETS.mkdir(parents=True, exist_ok=True)
    out = []
    p = SHEETS / "contact_12.jpg"
    cv2.imwrite(str(p), _grid([_tile(i, rows, uv_all, valid, lab(i), mz) for i in picks]), [cv2.IMWRITE_JPEG_QUALITY, 88])
    out.append(p)
    # flagged (identity / duplicate) first, then the lowest own-mask shares
    bad = []
    for r in ROLES:
        bad += [(0, int(i), r) for i in np.flatnonzero(flags[r]["bad"])]
        own = np.where(valid[r], Q[r]["own"], np.nan)
        order = np.argsort(np.nan_to_num(own, nan=9))
        bad += [(1, int(i), r) for i in order[:12] if np.isfinite(own[i])]
    seen, sel = set(), []
    for pri, i, r in sorted(bad):
        if i in seen or any(abs(i - j) < 8 for j in seen):
            continue
        seen.add(i)
        sel.append((i, r, pri))
        if len(sel) == 12:
            break
    if sel:
        tiles = []
        for i, r, pri in sel:
            fl = [k for k in ("on_partner", "teacher_swap", "mamma_swap", "duplicate", "facing_flip", "convert") if flags[r][k][i]]
            tiles.append(_tile(i, rows, uv_all, valid, lab(i, f" {NAME[r]}") + ("\n" + ",".join(fl) if fl else
                                                                                 "\nlowest own share"), mz))
        p = SHEETS / "flagged.jpg"
        cv2.imwrite(str(p), _grid(tiles), [cv2.IMWRITE_JPEG_QUALITY, 88])
        out.append(p)
    log(f"sheets: {[str(x) for x in out]}")
    return out


def qa_nofit() -> dict:
    """QA without a first-pass fit (a new take: pipeline analyze runs SAM 3D Body BEFORE the fit; the 2026-10-21 trial's lite
    QA): the checks that need no MAMMA fit - valid output, the converter's fit error, joints on the partner's mask instead of
    the dancer's own, a 2D duplicate of the partner, facing flips. Writes <role>_qa_ok into smplx_3db.npz and qa.json."""
    import cv2
    s = np.load(OUT / "smplx_3db.npz", allow_pickle=True)
    meta = json.loads((INPUT / "meta.json").read_text())
    rows = meta["frames"]
    F = len(rows)
    K = np.array([r["K"] for r in rows], float)
    mz = np.load(WORK / "people" / f"masks_{CAM}.npz")
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * DILATE_PX + 1, 2 * DILATE_PX + 1))
    valid = {r: np.asarray(s[f"{r}_valid"], bool) for r in ROLES}
    Jc = {r: np.asarray(s[f"{r}_joints"], float)[:, :22] for r in ROLES}
    own, part = {r: np.full(F, np.nan) for r in ROLES}, {r: np.full(F, np.nan) for r in ROLES}
    uv = {r: np.full((F, 22, 2), np.nan) for r in ROLES}
    W, H = int(meta["width"]), int(meta["height"])
    for i in range(F):
        if not any(valid[r][i] for r in ROLES):
            continue
        ms = {r: _decode(mz, f"m_{i}_{WHO[r]}") for r in ROLES}
        dil = {r: (cv2.dilate(ms[r].astype(np.uint8), kern) > 0) if ms[r] is not None else None for r in ROLES}
        for r in ROLES:
            if not valid[r][i]:
                continue
            q = _project(Jc[r][i], K[i])
            uv[r][i] = q
            o = [x for x in ROLES if x != r][0]
            ok = np.isfinite(q).all(1)
            xi = np.clip(np.round(np.nan_to_num(q[:, 0])).astype(int), 0, W - 1)
            yi = np.clip(np.round(np.nan_to_num(q[:, 1])).astype(int), 0, H - 1)
            a = (dil[r][yi, xi] & ok) if dil[r] is not None else np.zeros(22, bool)
            b = (dil[o][yi, xi] & ok) if dil[o] is not None else np.zeros(22, bool)
            own[r][i], part[r][i] = a.mean(), (b & ~a).mean()
    box_h = {r: np.array([((rw["mask_box_all"][r] or [0, 0, 0, 1])[3] - (rw["mask_box_all"][r] or [0, 0, 0, 0])[1])
                          for rw in rows], float) for r in ROLES}
    both = valid["lead"] & valid["follow"]
    d2 = np.full(F, np.nan)
    d2[both] = np.nanmean(np.linalg.norm(uv["lead"][both] - uv["follow"][both], axis=-1), 1) / np.maximum(
        np.maximum(box_h["lead"][both], box_h["follow"][both]), 1)
    dup = both & (np.nan_to_num(d2, nan=1.0) < DUP_2D)
    rep = {"roles": {}, "duplicate_frames": np.flatnonzero(dup).tolist(), "mode": "no first-pass fit (qa_nofit)",
           "note": "no MAMMA comparison: the take has no smplx/ yet"}
    arrays = {k: np.asarray(s[k]) for k in s.files}
    for r in ROLES:
        err = np.asarray(s[f"{r}_fit_err_cm"], float)
        fw = facing(np.asarray(s[f"{r}_joints_world"], float))
        flips = np.nan_to_num(facing_flips(fw, valid[r]), nan=0.0) > FLIP_DEG
        flags = {"fit_err": valid[r] & (np.nan_to_num(err, nan=99.0) > MAX_FIT_ERR_CM),
                 "on_partner": valid[r] & (np.nan_to_num(part[r]) > ON_PARTNER_SHARE) & (np.nan_to_num(own[r], nan=1.0) < ON_PARTNER_OWN),
                 "duplicate": valid[r] & dup, "facing_flip": valid[r] & flips}
        bad = np.zeros(F, bool)
        for v in flags.values():
            bad |= v
        arrays[f"{r}_qa_ok"] = valid[r] & ~bad
        rep["roles"][r] = {"frames": F, "valid": int(valid[r].sum()), "coverage": round(float(valid[r].mean()), 4),
                           "flags": {k: int(v.sum()) for k, v in flags.items()}, "qa_ok": int(arrays[f"{r}_qa_ok"].sum()),
                           "own_mask_share_median": round(float(np.nanmedian(own[r])), 3) if np.isfinite(own[r]).any() else None,
                           "fit_err_cm_median": round(float(np.nanmedian(err[valid[r]])), 3) if valid[r].any() else None,
                           "jitter_accel_rr_mm_per_frame2": None}
    ok_share = min(rep["roles"][r]["qa_ok"] / max(1, F) for r in ROLES)
    rep["verdict"] = {"usable_as_prior": bool(ok_share > 0.3), "qa_ok_share_min": round(ok_share, 3)}
    np.savez_compressed(OUT / "smplx_3db.npz", **arrays)
    (OUT / "qa.json").write_text(json.dumps(rep, indent=1))
    log(f"qa (no fit): {json.dumps({r: rep['roles'][r]['flags'] for r in ROLES})}, qa_ok "
        f"{[rep['roles'][r]['qa_ok'] for r in ROLES]} of {F}")
    return rep


def fit_lattice() -> np.ndarray:
    """The take's 30 Hz fit lattice t_k = k / 30 over the camera track's frames (k = 0 .. floor(30 t_last)): what mamma
    prepare --grid-anchor 0 grids a one-camera take on (prior_nofit; the trial hard-coded T = 599)."""
    from .. import mamma_dyn as md
    tr = md.load_tracks(WORK, [CAM])[CAM]
    T = int(np.floor(float(np.max(tr.t)) * 30.0 + 1e-6)) + 1
    return np.arange(T) / 30.0


# ================================================================================================ prior (MAMMA_POSE_PRIOR)
def prior(smooth: float | None = None) -> dict:
    """realism/sam3db/prior_3db_full.npz: the SAM 3D Body prior (cand_fitb.prior_interp) on the take's fit lattice (the
    2643 instants of smplx/: t_k = k / 30), both dancers; conf = cand_fitb.mask_conf of the zoom-compensated visible
    mask share relative to its median, 0 where a frame carries an identity / duplicate flag (qa_ok)."""
    from .. import mamma_dyn as md
    from . import candidates as C
    from .cand_fitb import PRIOR_SIGMA_M, PRIOR_SMOOTH, mask_conf, prior_interp
    smooth = PRIOR_SMOOTH if smooth is None else smooth
    if not has_fit():
        return prior_nofit(smooth)
    s = np.load(OUT / "smplx_3db.npz", allow_pickle=True)
    if "lead_qa_ok" not in s.files:
        raise SystemExit("run qa first (it writes <role>_qa_ok)")
    tr = md.load_tracks(WORK, [CAM])[CAM]
    tg = np.asarray(np.load(WORK / "smplx" / "lead.npz")["t"], float)
    T = len(tg)
    if np.abs(tg - np.arange(T) / 30.0).max() > 1e-9:
        raise SystemExit("smplx/ t is not the 30 Hz lattice k / 30")
    res = {r: None for r in ROLES}
    rep = {"T": T, "smooth_frames": smooth, "sigma_m": PRIOR_SIGMA_M, "roles": {}}
    mam = {r: np.load(WORK / "smplx" / f"{r}.npz", allow_pickle=True) for r in ROLES}
    for r in ROLES:
        ok = np.asarray(s[f"{r}_qa_ok"], bool)
        vis = np.asarray(s[f"{r}_visible"], float)
        conf = np.zeros(len(vis))
        conf[ok] = mask_conf(vis[ok])
        res[r] = prior_interp(np.asarray(s[f"{r}_joints"], float)[:, :22], np.asarray(s[f"{r}_smplx_global_orient"], float),
                              np.asarray(s[f"{r}_smplx_body_pose"], float), tr.R, tr.tvec, tr.t, ok, conf, tg, smooth)
        g = res[r]
        acc = C.accel_rr_mm(tg, g["joints"])
        Jm = np.asarray(mam[r]["joints"], float)[:, :22]
        absent = np.asarray(mam[r]["absent"], bool)
        use = g["ok"] & ~absent
        rr = 1e3 * np.linalg.norm((g["joints"][use] - g["joints"][use, :1]) - (Jm[use] - Jm[use, :1]), axis=-1)[:, 1:].mean(1)
        rep["roles"][r] = {"grid_frames_with_prior": int(g["ok"].sum()), "conf_mean": round(float(g["conf"][g["ok"]].mean()), 3),
                           "conf_below_0.5": int((g["conf"][g["ok"]] < 0.5).sum()),
                           "dt_nearest_ms_max": round(float(g["dt_nearest_ms"].max()), 2),
                           "jitter_accel_rr_median": round(float(np.nanmedian(acc)), 2),
                           "vs_mamma_smplx_rr_mm (agreement, not ground truth)": _stats(rr)}
        log(f"prior {r}: {rep['roles'][r]}")
    P = OUT / "prior_3db_full.npz"
    np.savez_compressed(P, joints=np.stack([res[r]["joints"] for r in ROLES], 1).astype(np.float32),
                        conf=np.stack([res[r]["conf"] for r in ROLES], 1).astype(np.float32),
                        global_orient=np.stack([res[r]["global_orient"] for r in ROLES], 1).astype(np.float32),
                        body_pose=np.stack([res[r]["body_pose"] for r in ROLES], 1).astype(np.float32),
                        sigma_m=PRIOR_SIGMA_M, rotate=False, scale=True, t=tg, k=np.arange(T),
                        frame_a=np.stack([res[r]["frame_a"] for r in ROLES], 1), frame_w=np.stack([res[r]["frame_w"] for r in ROLES], 1),
                        smooth_frames=smooth, source=str(OUT / "smplx_3db.npz"))
    rep["path"] = str(P)
    (OUT / "prior_3db_full.json").write_text(json.dumps(rep, indent=1))
    return rep


def prior_nofit(smooth: float | None = None) -> dict:
    """prior() for a take without a first-pass fit: the same prior (cand_fitb.prior_interp) on the take's own 30 Hz lattice
    (fit_lattice), with no comparison against a fit. The 2026-10-21 trial's prior_nofit, the lattice from the track."""
    from .. import mamma_dyn as md
    from . import candidates as C
    from .cand_fitb import PRIOR_SIGMA_M, PRIOR_SMOOTH, mask_conf, prior_interp
    smooth = PRIOR_SMOOTH if smooth is None else smooth
    s = np.load(OUT / "smplx_3db.npz", allow_pickle=True)
    if "lead_qa_ok" not in s.files:
        raise SystemExit("run qa first (it writes <role>_qa_ok)")
    tr = md.load_tracks(WORK, [CAM])[CAM]
    tg = fit_lattice()
    T = len(tg)
    res, rep = {}, {"T": T, "smooth_frames": smooth, "sigma_m": PRIOR_SIGMA_M, "roles": {}, "mode": "no first-pass fit"}
    for r in ROLES:
        ok = np.asarray(s[f"{r}_qa_ok"], bool)
        vis = np.asarray(s[f"{r}_visible"], float)
        conf = np.zeros(len(vis))
        conf[ok] = mask_conf(vis[ok])
        res[r] = g = prior_interp(np.asarray(s[f"{r}_joints"], float)[:, :22], np.asarray(s[f"{r}_smplx_global_orient"], float),
                                  np.asarray(s[f"{r}_smplx_body_pose"], float), tr.R, tr.tvec, tr.t, ok, conf, tg, smooth)
        acc = C.accel_rr_mm(tg, g["joints"])
        rep["roles"][r] = {"grid_frames_with_prior": int(g["ok"].sum()),
                           "conf_mean": round(float(g["conf"][g["ok"]].mean()), 3) if g["ok"].any() else None,
                           "dt_nearest_ms_max": round(float(g["dt_nearest_ms"].max()), 2),
                           "jitter_accel_rr_median": round(float(np.nanmedian(acc)), 2) if np.isfinite(acc).any() else None}
        log(f"prior (no fit) {r}: {rep['roles'][r]}")
    P = OUT / "prior_3db_full.npz"
    np.savez_compressed(P, joints=np.stack([res[r]["joints"] for r in ROLES], 1).astype(np.float32),
                        conf=np.stack([res[r]["conf"] for r in ROLES], 1).astype(np.float32),
                        global_orient=np.stack([res[r]["global_orient"] for r in ROLES], 1).astype(np.float32),
                        body_pose=np.stack([res[r]["body_pose"] for r in ROLES], 1).astype(np.float32),
                        sigma_m=PRIOR_SIGMA_M, rotate=False, scale=True, t=tg, k=np.arange(T),
                        frame_a=np.stack([res[r]["frame_a"] for r in ROLES], 1), frame_w=np.stack([res[r]["frame_w"] for r in ROLES], 1),
                        smooth_frames=smooth, source=str(OUT / "smplx_3db.npz"))
    rep["path"] = str(P)
    (OUT / "prior_3db_full.json").write_text(json.dumps(rep, indent=1))
    return rep


def prior_slice(k0: int, k1: int, out: Path) -> Path:
    """The full-lattice prior cut to a chunk's fit grid k0..k1 (inclusive; mamma prepare --grid-anchor 0 --grid-k=k0:k1)."""
    z = np.load(OUT / "prior_3db_full.npz", allow_pickle=True)
    k = np.asarray(z["k"], int)
    sel = (k >= k0) & (k <= k1)
    if sel.sum() != k1 - k0 + 1:
        raise SystemExit(f"k {k0}..{k1} not inside the prior's lattice 0..{k.max()}")
    d = {n: (np.asarray(z[n])[sel] if np.ndim(z[n]) and len(z[n]) == len(k) else z[n]) for n in z.files}
    d["source"] = np.array(f"{OUT / 'prior_3db_full.npz'} k {k0}..{k1}")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **d)
    log(f"prior slice k {k0}..{k1} -> {out}")
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m dancecap.mono4d.jl_sam3db")
    ap.add_argument("--take", default="john_lula", help="the one-camera take (default john_lula: this stage's first take)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check-access")
    pp = sub.add_parser("prep")
    pp.add_argument("--decode-on-pod", action="store_true", help="the frames are decoded on the pod (job --video)")
    j = sub.add_parser("job")
    j.add_argument("--max-minutes", type=float, default=75)
    j.add_argument("--chunk", type=int, default=CHUNK)
    j.add_argument("--reserve-min", type=float, default=30.0)
    j.add_argument("--video", help="decode the frames on the pod from this video (not uploaded as JPEGs)")
    im = sub.add_parser("import")
    im.add_argument("--session")
    q = sub.add_parser("qa")
    q.add_argument("--no-sheets", action="store_true")
    pr = sub.add_parser("prior")
    pr.add_argument("--smooth", type=float)
    ps = sub.add_parser("prior-slice")
    ps.add_argument("--k0", type=int, required=True)
    ps.add_argument("--k1", type=int, required=True)
    ps.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.take != TAKE:
        configure(a.take)
    if a.cmd == "check-access":
        from .cand_sam3d_body import check_access
        print(json.dumps(check_access(), indent=1))
    elif a.cmd == "prep":
        info = prep(a.decode_on_pod)
        print(json.dumps({k: v for k, v in info.items() if k != "skipped_frames"}, indent=1))
    elif a.cmd == "job":
        print(job(a.max_minutes, a.chunk, a.reserve_min, video=Path(a.video) if a.video else None))
    elif a.cmd == "import":
        print(json.dumps(do_import(a.session), indent=1, default=str))
    elif a.cmd == "qa":
        rep = qa(not a.no_sheets)
        print(json.dumps({r: {k: rep["roles"][r][k] for k in ("valid", "coverage", "flags", "jitter_accel_rr_mm_per_frame2")}
                          for r in ROLES} | {"verdict": rep["verdict"]}, indent=1))
    elif a.cmd == "prior":
        print(json.dumps(prior(a.smooth), indent=1))
    else:
        print(prior_slice(a.k0, a.k1, Path(a.out)))


if __name__ == "__main__":
    main()
