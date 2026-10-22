r"""Generic LHM++ splat avatars of a ONE-phone take (pipeline/mono_route step "avatars", 2026-10-22).

Per dancer one canonical 3D Gaussian avatar (LHM++ LHMPP-700M-PixelShuffle @ 906b5d9, 160k Gaussians; code Apache-2.0 +
NVIDIA EG3D parts, WEIGHTS CC BY-NC 4.0) built on a Runpod pod from reference crops of that dancer chosen automatically from
the take's own outputs, plus a turntable PNG per dancer for a quick look. The john_lula version (jl_lhmpp_preview +
jl_realism_render) is take-specific (zoom steps, held-out windows, a photo crop, renders into phone frames); this one needs
only what `python -m dancecap.pipeline analyze <video>` writes:

    video          Take.load(T).sources[0], decoded here with cv2 (frame index = the people tables' row)
    people/        masks_01.npz (m_<frame>_<slot> PNG 0/255, slot 0 lead / 1 follow, box = tight mask bbox, area),
                   teachers_01.npz (COCO-17 kp, conf, role_conf, visible, skel_in_partner, edge)
    cameras/01.npz per-frame R, tvec, t (the fit's facing seen from the phone)
    smplx_rf2_noslip/ (else smplx/)   the stitched fit: betas (LHM++ shape mode ref: betas[:10]) and global_orient (facing).
                   Optional: without it the facing comes from the 2D shoulders and the betas from shape/mamma_pin_betas.json
                   (the fits step's stature-only shape) or zeros.

FRAMES (choose_frames, pure). Per dancer and frame: the HARD gate - own mask present (>= AREA_MIN_PX), not cut by the frame
edge, teachers ok - then the CLEAN gate: a large own mask (>= 0.6 x the dancer's 90th-percentile area), detector conf, role_conf
(identity), visible (people_mono: own area / local p90), no confident torso joint inside the partner's mask only
(skel_in_partner), small box IoU of the two dancers' masks, every body keypoint (shoulders, hips, knees, ankles) seen, a slow
pose (pose_speed: joints relative to the hips, torso lengths / s - fast frames are blurred and caught mid-move), and from the
mask survey in prep: partner_occ (partner pixels inside the closed own silhouette = the partner in FRONT of the dancer, a
white patch in the crop), main_frac (largest component's share of the mask: a leak, or the dancer cut in two) and head_cov
(the expected head disk above the shoulders covered by the own mask: SAM loses the head in fast turns / hair whips; the
first lk03_mono20 sheet had 3 headless follow crops). Quality q = min(1, area / p90) x conf x role_conf x min(1, sharpness /
median)^0.5 (Laplacian inside the eroded mask) / (1 + (speed / SPEED_SOFT)^2). Greedy: 00 = the most frontal frame
(frontal_pick: |yaw| small, the face seen, arms below the head), then max q x (yaw distance to the picks / 180 + 0.15) x a
time-spread factor, >= min_gap_s apart; when the clean frames cannot fill n the gap halves twice, then RELAXED frames are
admitted (flagged "relaxed"). Facing yaw = the fit's where it exists (cand_lhmpp.body_yaw_deg convention: 0 = the dancer faces
the phone, +-180 = back to it), else yaw_2d (shoulder width for |cos|, front / back from the 68 face landmarks' scores; RTMW
also scores faces on some back views - trial_mono_jl20's lead - so the fit's facing always wins where it exists).

CROPS (prep). LHM++'s input format exactly as cand_lhmpp / jl_lhmpp_preview write it: the (cleaned) mask's tight bbox, white
outside the mask, PadRatioWithScale(5/3, [840]) centred on a white 504 x 840 canvas (cand_lhmpp.lhm_crop); 00 = most frontal.
An optional cut-out photo per dancer (--photo lead=cutout.png: RGBA, alpha = the dancer) is appended after the video crops.

POD (job). remote/cand_lhmpp/install.sh (LHM++'s own py3.10 venv, measured 120-440 s) + remote/cand_lhmpp/lhmpp_render_pod.py
UNCHANGED (john_lula's final-render script: refs/<role>/*.png + poses.npz + views.npz/json; with no render passes it builds the
two avatars - one infer_single_view each, shape mode ref - writes gs/<role>_cano.ply and turntable/<role>.png and stops; ~2.3
min with the 40 s model load) + an OPTIONAL export command, remote/cand_lhmpp/recap_lhmpp_pod.py UNCHANGED on the same input
(no A2 views, no preview groups): LHM++'s own LBS weights, export/<role>_export.npz (docs/SPLAT_AVATAR_FORMAT.md section 7),
what a splat player needs to pose the avatar (~2 min). poses.npz holds ONE pose key per dancer (the fit at the dancer's
frontal reference frame, else the zero pose): it only feeds the pods' transfer check (the canonical avatar does not depend on
the pose).

    python -m dancecap.mono4d.lhmpp_generic prep --take T [--out DIR] [--n 8] [--photo lead=cutout.png]
    python -m dancecap.mono4d.lhmpp_generic job --take T [--out DIR] [--no-export]       -> <out>/<T>_lhmpp_avatars.json
    python -m dancecap remote run --jobs <that json> --max-cost 1 ...                     (mono_route: ctx.remote([job]))
    python -m dancecap.mono4d.lhmpp_generic import --take T [--session SID]

Layout: work/<take>/realism/lhmpp_generic/{input/ (the pod's only input: refs/<role>/NN.png, poses.npz, views.npz, views.json,
refs.json, prep.json), pod_out/ (gs/, turntable/, export/, lhmpp_render_pod.json, recap_lhmpp_pod.json), refs_sheet.jpg (LOOK
at it: every crop + its frame with the mask outline), <take>_lhmpp_avatars.json}; import -> work/<take>/avatar_lhmpp/
{gs/<role>_cano.ply, turntable/<role>.png, export/<role>_export.npz (when built), refs_sheet.jpg, avatar_lhmpp.json}.
Likeness data (frames, crops, splats) stays in work/ (git-ignored) and our own pod.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from ..take import PACKAGE_ROOT, WORKSPACE
from . import cand_lhmpp as CL

ROLES = ("lead", "follow")
SLOT = {"lead": 0, "follow": 1}           # people/ tables: slot 0 = lead, 1 = follow
N_REFS = 8
MIN_GAP_S = 1.0
AREA_MIN_PX = 1500
EDGE_PX = 3                               # mask bbox closer than this to a frame edge = cut
BODY_KP = (5, 6, 11, 12, 13, 14, 15, 16)  # COCO-17 shoulders, hips, knees, ankles: the whole body is seen
FACE_KP = (0, 1, 2)                       # nose, eyes (the face score when the whole-body table is missing)
FACE68 = slice(23, 91)                    # COCO-WholeBody / RTMW kp_wb: the 68 face landmarks
FACE_FRONT, FACE_BACK = 0.8, 0.65         # face68 mean score: >= front = the face is seen, < back = seen from behind
FRONTAL_DEG = 35.0
SHOULDER_TORSO = 0.7                      # frontal shoulder width / shoulder-mid-to-hip-mid length (yaw_2d)
HEAD_UP, HEAD_R = 0.4, 0.15               # expected head centre: shoulder middle + 0.4 x (shoulders - hips); radius 0.15 x torso
SPEED_KP = (0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)   # nose + body joints for the pose speed
SPEED_SOFT = 3.0                          # torso lengths / s: quality x 1 / (1 + (speed / SPEED_SOFT)^2)
CLEAN = {"area_rel": 0.6, "conf": 0.5, "role_conf": 0.85, "visible": 0.6, "skel_in_partner": 0.0, "box_iou": 0.2,
         "kp": 0.3, "speed": 3.0, "partner_occ": 0.02, "main_frac": 0.9, "head_cov": 0.6}
RELAXED = {"area_rel": 0.4, "conf": 0.35, "role_conf": 0.7, "visible": 0.4, "skel_in_partner": 0.25, "box_iou": 0.35,
           "kp": 0.15, "speed": 5.0, "partner_occ": 0.06, "main_frac": 0.75, "head_cov": 0.35}
HARD = ("mask", "cut", "not ok")
MEASURED = ("partner_occ", "main_frac", "head_cov", "sharp")   # masks_info keys from the mask survey (NaN = not surveyed)
SURVEY_MAX = 300                          # mask-surveyed candidate frames per dancer (evenly over the take)
SHARP_H = 256                             # sharpness on the crop scaled down to at most this height
FIT_DIRS = ("smplx_rf2_noslip", "smplx")
SMPLX_MODEL = WORKSPACE / "mamma" / "data" / "body_models" / "smplx_locked_head" / "smplx" / "SMPLX_NEUTRAL.npz"
POD_PY = "/workspace/lhmpp_cand/venv/bin/python"
POD_ENV = {"LHMPP_COMMIT": CL.LHMPP_COMMIT, "LHMPP_ROOT": "/workspace/lhmpp_cand/LHM-plusplus",
           "LHMPP_VENV": "/workspace/lhmpp_cand/venv", "LHMPP_MODEL": CL.MODEL_NAME,
           "TORCH_HOME": "/workspace/lhmpp_cand/torch_home", "HF_HOME": "/workspace/lhmpp_cand/hf_home"}
EXCLUDE = ["dancecap/work/runpod/pywheels", "dancecap/work/runpod/wheelhouse", "mamma-turing"]
JOB_MINUTES = 40.0                        # >= install (20) + build (10) timeouts; expected install 2-7.5, build ~2.5, export ~2
TILE_H = 280
ROLE_BGR = {"lead": (0, 190, 0), "follow": (0, 0, 230)}


def log(msg: str) -> None:
    print(f"[lhmpp_generic {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ws_rel(p: Path) -> str:
    """Workspace-relative posix path WITHOUT resolving links (a worktree reaches work/ through a junction)."""
    r = Path(os.path.relpath(os.path.abspath(p), os.path.abspath(WORKSPACE))).as_posix()
    if r.startswith(".."):
        raise ValueError(f"{p} is outside the workspace {WORKSPACE}")
    return r


def work_dir(take: str) -> Path:
    return PACKAGE_ROOT / "work" / take


def default_dir(take: str) -> Path:
    return work_dir(take) / "realism" / "lhmpp_generic"


def job_name(take: str) -> str:
    return f"{take}_lhmpp_avatars"[:60]


# ------------------------------------------------------------------------------------------------ pure helpers (unit-tested)


def yaw_2d(kp: np.ndarray, face: np.ndarray | None = None) -> np.ndarray:
    """Signed facing yaw (deg) from COCO-17 keypoints (..., 17, 3): |cos| = |x_Lsh - x_Rsh| / (SHOULDER_TORSO x torso length);
    front or back from the face score when given (the 68 face landmarks' mean score: >= FACE_FRONT front, < FACE_BACK back;
    the detectors' left / right shoulders flip on back views, lk03_mono20 check 2026-10-22), else from the shoulder order (the
    person's left shoulder on the image RIGHT = facing the camera); the side from the nose against the shoulder middle.
    NaN where the shoulders or hips are not seen at all. 0 = frontal, +-180 = back (sign conventions differ from the fit's)."""
    kp = np.asarray(kp, float)
    ls, rs, lh, rh, nose = kp[..., 5, :], kp[..., 6, :], kp[..., 11, :], kp[..., 12, :], kp[..., 0, :]
    seen = np.minimum.reduce([ls[..., 2], rs[..., 2], lh[..., 2], rh[..., 2]]) > 0.05
    smid, hmid = (ls[..., :2] + rs[..., :2]) / 2, (lh[..., :2] + rh[..., :2]) / 2
    torso = np.linalg.norm(smid - hmid, axis=-1)
    c = np.clip((ls[..., 0] - rs[..., 0]) / np.maximum(SHOULDER_TORSO * torso, 1e-6), -1.0, 1.0)
    if face is not None:
        fz = np.asarray(face, float)
        c = np.where(fz >= FACE_FRONT, np.abs(c), np.where(fz < FACE_BACK, -np.abs(c), c))
    side = np.where(nose[..., 2] > 0.3, np.sign(nose[..., 0] - smid[..., 0]), 1.0)
    side = np.where(side == 0, 1.0, side)
    yaw = np.degrees(np.arctan2(side * np.sqrt(np.maximum(0.0, 1 - c * c)), c))
    return np.where(seen & (torso > 1.0), yaw, np.nan)


def body_yaw_deg(global_orient: np.ndarray, pelvis: np.ndarray, R: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """cand_lhmpp.body_yaw_deg for N frames: signed yaw of the camera seen from the body (0 = the dancer faces the phone)."""
    from scipy.spatial.transform import Rotation as Rot
    go = np.asarray(global_orient, float).reshape(-1, 3)
    fwd = Rot.from_rotvec(go).apply([0.0, 0.0, 1.0])
    R, tvec = np.asarray(R, float).reshape(-1, 3, 3), np.asarray(tvec, float).reshape(-1, 3)
    C = -np.einsum("nji,nj->ni", R, tvec)
    d = C - np.asarray(pelvis, float).reshape(-1, 3)
    f2, d2 = fwd[:, [0, 2]], d[:, [0, 2]]
    return np.degrees(np.arctan2(f2[:, 0] * d2[:, 1] - f2[:, 1] * d2[:, 0], np.sum(f2 * d2, 1)))


def box_iou_pairs(box: np.ndarray) -> np.ndarray:
    """(F,) IoU of the two dancers' boxes (F, 2, 4) [x0, y0, x1, y1]; 0 where a box is missing."""
    b = np.asarray(box, float)
    a, c = b[:, 0], b[:, 1]
    iw = np.clip(np.minimum(a[:, 2], c[:, 2]) - np.maximum(a[:, 0], c[:, 0]), 0, None)
    ih = np.clip(np.minimum(a[:, 3], c[:, 3]) - np.maximum(a[:, 1], c[:, 1]), 0, None)
    inter = iw * ih
    ua = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]) + (c[:, 2] - c[:, 0]) * (c[:, 3] - c[:, 1]) - inter
    out = np.where(ua > 0, inter / np.where(ua > 0, ua, 1.0), 0.0)
    return np.nan_to_num(out, nan=0.0)


def face_score(teachers: dict) -> np.ndarray:
    """(F, 2) how well the face is seen: the 68 face landmarks' mean score (kp_wb), else the nose + eyes of the COCO-17 kp."""
    if "kp_wb" in teachers and np.asarray(teachers["kp_wb"]).shape[-2] >= FACE68.stop:
        return np.asarray(teachers["kp_wb"], float)[..., FACE68, 2].mean(-1)
    return np.asarray(teachers["kp"], float)[..., FACE_KP, 2].mean(-1)


def head_cov(own: np.ndarray, kp17: np.ndarray) -> float:
    """Share of the expected head disk (centre = shoulder middle + HEAD_UP x (shoulder middle - hip middle), radius HEAD_R x
    that torso length; it follows a leaning torso) that the dancer's own mask covers: ~0 when the mask lost the head (a fast
    turn, hair whip, the partner's arm across it; lk03_mono20 follow f31 / f69: 0.0, good frames >= 0.76). NaN without
    shoulders / hips. The part of the disk outside the image counts as not covered."""
    k = np.asarray(kp17, float)
    if min(k[5, 2], k[6, 2], k[11, 2], k[12, 2]) <= 0.05:
        return float("nan")
    smid, hmid = (k[5, :2] + k[6, :2]) / 2, (k[11, :2] + k[12, :2]) / 2
    L = float(np.linalg.norm(smid - hmid))
    if L < 2:
        return float("nan")
    c = smid + HEAD_UP * (smid - hmid)
    r = max(2.0, HEAD_R * L)
    H, W = own.shape
    n_all = max(1, int(round(math.pi * r * r)))
    x0, x1 = int(max(0, np.floor(c[0] - r))), int(min(W, np.ceil(c[0] + r) + 1))
    y0, y1 = int(max(0, np.floor(c[1] - r))), int(min(H, np.ceil(c[1] + r) + 1))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    yy, xx = np.mgrid[y0:y1, x0:x1]
    disk = (xx - c[0]) ** 2 + (yy - c[1]) ** 2 <= r * r
    return float(np.asarray(own, bool)[y0:y1, x0:x1][disk].sum() / max(int(disk.sum()), n_all))


def pose_speed(kp: np.ndarray, t: np.ndarray) -> np.ndarray:
    """(F, 2) how fast the pose changes: median over the nose + body joints of their speed RELATIVE to the hip middle (camera
    pan and travel removed), in torso lengths per second (central differences on t; the first / last frame take their
    neighbour's). Fast frames are motion-blurred and caught mid-move (lk03_mono20 follow: a back-bend 6.7 /s and a spin
    6.1 /s against a median 0.8 /s). NaN where the hips are not seen."""
    kp = np.asarray(kp, float)
    t = np.asarray(t, float)
    F = kp.shape[0]
    out = np.full(kp.shape[:2], np.nan)
    if F < 3:
        return out
    hmid = (kp[..., 11, :2] + kp[..., 12, :2]) / 2
    L = np.linalg.norm((kp[..., 5, :2] + kp[..., 6, :2]) / 2 - hmid, axis=-1)
    rel = kp[..., :2] - hmid[..., None, :]
    dt = np.maximum(t[2:] - t[:-2], 1e-6)
    d = np.linalg.norm(rel[2:] - rel[:-2], axis=-1) / dt[:, None, None]
    sp = np.median(d[:, :, list(SPEED_KP)], -1) / np.maximum(L[1:-1], 1.0)
    seen = np.minimum(kp[2:, :, 11:13, 2].min(-1), kp[:-2, :, 11:13, 2].min(-1)) > 0.05
    out[1:-1] = np.where(seen, sp, np.nan)
    out[0], out[-1] = out[1], out[-2]
    return out


def frame_stats(teachers: dict, masks_info: dict, facing: np.ndarray | None = None) -> dict:
    """Per frame (F) and slot (2) the numbers the gates and the quality read."""
    kp = np.asarray(teachers["kp"], float)
    F = kp.shape[0]
    W, H = (int(v) for v in (masks_info.get("size") if masks_info.get("size") is not None else teachers["size"]))

    def arr(src: dict, k: str, default: float) -> np.ndarray:
        return np.asarray(src[k], float).reshape(F, 2) if k in src else np.full((F, 2), default)

    area = np.nan_to_num(np.asarray(masks_info["area"], float).reshape(F, 2))
    box = np.asarray(masks_info["box"], float).reshape(F, 2, 4)
    p90 = np.array([np.percentile(area[:, s][area[:, s] >= AREA_MIN_PX], 90) if (area[:, s] >= AREA_MIN_PX).any() else 1.0
                    for s in range(2)])
    with np.errstate(invalid="ignore"):
        cut = ~np.isfinite(box).all(-1) | (box[..., 0] < EDGE_PX) | (box[..., 1] < EDGE_PX) | \
            (box[..., 2] > W - EDGE_PX) | (box[..., 3] > H - EDGE_PX)
    if "edge" in teachers:
        cut |= np.asarray(teachers["edge"], bool).reshape(F, 2)
    face = face_score(teachers)
    y2 = yaw_2d(kp, face)
    if facing is not None:
        fz = np.asarray(facing, float).reshape(F, 2)
        yaw = np.where(np.isfinite(fz), fz, y2)
        src = np.where(np.isfinite(fz), "fit", np.where(np.isfinite(y2), "2d", "none"))
    else:
        yaw, src = y2, np.where(np.isfinite(y2), "2d", "none")
    st = {"F": F, "W": W, "H": H, "t": np.asarray(teachers["t"], float).reshape(F),
          "frame": np.asarray(teachers.get("frame_index", np.arange(F)), int).reshape(F),
          "ok": np.asarray(teachers.get("ok", np.ones(F, bool)), bool).reshape(F),
          "area": area, "area_rel": area / p90[None], "box": box, "cut": cut, "box_iou": box_iou_pairs(box),
          "conf": arr(teachers, "conf", 1.0), "role_conf": arr(teachers, "role_conf", 1.0),
          "visible": arr(teachers, "visible", 1.0), "skel_in_partner": arr(teachers, "skel_in_partner", 0.0),
          "kp_min": kp[:, :, BODY_KP, 2].min(-1), "face": face,
          "speed": pose_speed(kp, np.asarray(teachers["t"], float).reshape(F)),
          # share of the wrists below the nose (an unseen wrist counts as down): arms over the head hide the face
          "arms_down": np.mean([(kp[:, :, w, 1] > kp[:, :, 0, 1]) | (kp[:, :, w, 2] < 0.3) for w in (9, 10)], 0),
          "yaw": np.where(np.isfinite(yaw), yaw, 90.0), "yaw_src": src}   # unknown facing: a profile, never "frontal"
    for k in MEASURED:
        st[k] = arr(masks_info, k, np.nan)
        st[k + "_measured"] = k in masks_info
    return st


def gate_fails(st: dict, th: dict) -> dict:
    """{reason: (F, 2) bool, True = the frame fails it}: the HARD reasons + the thresholds th (CLEAN or RELAXED)."""
    F = st["F"]
    with np.errstate(invalid="ignore"):
        out = {"mask": st["area"] < AREA_MIN_PX, "cut": st["cut"], "not ok": np.repeat(~st["ok"][:, None], 2, 1),
               "area": st["area_rel"] < th["area_rel"], "conf": st["conf"] < th["conf"],
               "role_conf": st["role_conf"] < th["role_conf"], "visible": st["visible"] < th["visible"],
               "skel_in_partner": st["skel_in_partner"] > th["skel_in_partner"] + 1e-9,
               "box_iou": np.repeat((st["box_iou"] > th["box_iou"])[:, None], 2, 1), "keypoints": st["kp_min"] < th["kp"],
               "speed": ~(st["speed"] <= th["speed"])}
        measured = st["main_frac_measured"] or st["partner_occ_measured"]
        sv = np.isfinite(st["main_frac"]) | np.isfinite(st["partner_occ"])     # the survey reached the frame
        out["not surveyed"] = ~sv if measured else np.zeros((F, 2), bool)
        for k in ("partner_occ", "main_frac", "head_cov"):
            if st[k + "_measured"]:
                out[k] = sv & (~(st[k] <= th[k]) if k == "partner_occ" else ~(st[k] >= th[k]))
            else:
                out[k] = np.zeros((F, 2), bool)
    return out


def quality(st: dict, s: int) -> np.ndarray:
    """(F,) q of slot s: min(1, area / p90) x conf x role_conf x min(1, sharp / median)^0.5 (1 where not measured)
    x 1 / (1 + (pose speed / SPEED_SOFT)^2)."""
    sh = st["sharp"][:, s]
    fin = np.isfinite(sh) & (sh > 0)
    med = float(np.median(sh[fin])) if fin.any() else 1.0
    sf = np.where(fin, np.clip(np.where(fin, sh, 0.0) / max(med, 1e-9), 0, 1) ** 0.5, 1.0)
    sp = np.nan_to_num(st["speed"][:, s], nan=SPEED_SOFT)
    return (np.clip(st["area_rel"][:, s], 0, 1) * np.clip(st["conf"][:, s], 0, 1) * np.clip(st["role_conf"][:, s], 0, 1) * sf
            / (1.0 + (sp / SPEED_SOFT) ** 2))


def frontal_pick(st: dict, s: int, q: np.ndarray, rows: np.ndarray) -> int:
    """Image 00, the frontal reference LHM++ leans on for the face: the first non-empty pool of |yaw| <= FRONTAL_DEG with the
    face seen (>= FACE_FRONT); the face seen and |yaw| <= 2 x FRONTAL_DEG (the 2D yaw underestimates narrow shoulders);
    |yaw| <= FRONTAL_DEG; all rows. Best q x face^2 x (0.5 + 0.5 cos yaw) x (0.5 + 0.5 arms_down)."""
    rows = np.asarray(rows, int)
    yaw, face = np.abs(st["yaw"][rows, s]), st["face"][rows, s]
    for pool in ((yaw <= FRONTAL_DEG) & (face >= FACE_FRONT), (yaw <= 2 * FRONTAL_DEG) & (face >= FACE_FRONT),
                 yaw <= FRONTAL_DEG, np.ones(len(rows), bool)):
        if pool.any():
            break
    fr = rows[pool]
    sc = q[fr] * st["face"][fr, s] ** 2 * (0.5 + 0.5 * np.cos(np.radians(st["yaw"][fr, s]))) * \
        (0.5 + 0.5 * st["arms_down"][fr, s])
    return int(fr[np.argmax(sc)])


def _greedy(st: dict, s: int, q: np.ndarray, rows: np.ndarray, n: int, gap: float, seed: list[int]) -> list[int]:
    """Seed (or the most frontal row, frontal_pick), then max q x (yaw distance to the picks / 180 + 0.15) x (0.6 + 0.4 min(1,
    time distance / (span / n))) among rows >= gap from every pick."""
    picked = [int(r) for r in seed]
    rows = np.asarray(rows, int)
    if not len(rows):
        return picked
    t, yaw = st["t"], st["yaw"][:, s]
    span = max(float(t[rows].max() - t[rows].min()), 1e-6)
    if not picked:
        picked.append(frontal_pick(st, s, q, rows))
    while len(picked) < n:
        P = np.asarray(picked, int)
        dt = np.abs(t[rows][:, None] - t[P][None]).min(1)
        okm = (dt >= gap - 1e-9) & ~np.isin(rows, P)
        if not okm.any():
            break
        dy = np.abs((yaw[rows][:, None] - yaw[P][None] + 180.0) % 360.0 - 180.0).min(1)
        sc = np.where(okm, q[rows] * (dy / 180.0 + 0.15) * (0.6 + 0.4 * np.minimum(1.0, dt / (span / n))), -1.0)
        k = int(np.argmax(sc))
        if sc[k] < 0:
            break
        picked.append(int(rows[k]))
    return picked


def choose_frames(teachers: dict, masks_info: dict, n: int = N_REFS, min_gap_s: float = MIN_GAP_S,
                  facing: np.ndarray | None = None) -> dict:
    """Per dancer the n reference frames (module docstring, FRAMES). teachers: people/teachers_01.npz arrays (t, kp, conf,
    role_conf, visible, skel_in_partner, edge, frame_index, ok, size); masks_info: area (F, 2), box (F, 2, 4), size (W, H) and
    optionally the survey's partner_occ / main_frac / sharp (F, 2; NaN = not surveyed); facing: (F, 2) yaw deg (NaN unknown).
    Returns {role: {picks: [record], ...counts}}; picks[0] is the most frontal frame."""
    st = frame_stats(teachers, masks_info, facing)
    fc, fr = gate_fails(st, CLEAN), gate_fails(st, RELAXED)
    clean = ~np.any(np.stack(list(fc.values())), 0)
    relaxed = ~np.any(np.stack(list(fr.values())), 0)
    out = {}
    for role in ROLES:
        s = SLOT[role]
        q = quality(st, s)
        rc = np.nonzero(clean[:, s])[0]
        rr = np.nonzero(relaxed[:, s] & ~clean[:, s])[0]
        picks, gap = _greedy(st, s, q, rc, n, min_gap_s, []), min_gap_s
        for g in (min_gap_s / 2, min_gap_s / 4):
            if len(picks) >= n:
                break
            picks, gap = _greedy(st, s, q, rc, n, g, picks), g
        if len(picks) < n and len(rr):
            picks = _greedy(st, s, q, np.r_[rc, rr], n, gap, picks)
        recs = []
        for f in picks:
            recs.append({"row": int(f), "frame": int(st["frame"][f]), "t": round(float(st["t"][f]), 4),
                         "yaw_deg": round(float(st["yaw"][f, s]), 1), "yaw_src": str(st["yaw_src"][f, s]),
                         "q": round(float(q[f]), 4), "relaxed": bool(not clean[f, s]),
                         "area": int(st["area"][f, s]), "box": [int(v) for v in st["box"][f, s]],
                         **{k: (None if not np.isfinite(st[k][f, s]) else round(float(st[k][f, s]), 4))
                            for k in ("conf", "role_conf", "visible", "skel_in_partner", "face", "speed",
                                      "partner_occ", "main_frac", "head_cov", "sharp")},
                         "box_iou": round(float(st["box_iou"][f]), 4)})
        out[role] = {"picks": recs, "frames": int(st["F"]), "hard_ok": int((~np.any(np.stack([fc[k][:, s] for k in HARD]),
                                                                                      0)).sum()),
                     "clean": int(len(rc)), "relaxed_extra": int(len(rr)), "min_gap_used_s": gap,
                     "n_relaxed_picks": int(sum(r["relaxed"] for r in recs)),
                     "drop_counts": {k: int(v[:, s].sum()) for k, v in fc.items() if v[:, s].any()},
                     "yaw_src": {k: int(sum(r["yaw_src"] == k for r in recs)) for k in ("fit", "2d", "none")}}
    return out


def fill_holes(m: np.ndarray) -> np.ndarray:
    """Boolean mask with every enclosed background region filled (flood fill from the border)."""
    m = np.asarray(m, bool)
    pad = np.pad(m, 1).astype(np.uint8)
    ff = pad.copy()
    cv2.floodFill(ff, np.zeros((ff.shape[0] + 2, ff.shape[1] + 2), np.uint8), (0, 0), 1)
    return m | (ff[1:-1, 1:-1] == 0)


def clean_mask(m: np.ndarray, keep_frac: float = 0.05) -> np.ndarray:
    """The mask without stray components smaller than keep_frac x the largest one (they would inflate the crop's bbox)."""
    m = np.asarray(m, bool)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    if n <= 2:
        return m
    a = stats[1:, cv2.CC_STAT_AREA]
    keep = np.r_[False, a >= keep_frac * a.max()]
    return keep[lab]


def mask_stats(own: np.ndarray, partner: np.ndarray | None, kp17: np.ndarray | None = None) -> dict:
    """main_frac (largest component / mask), partner_occ (partner pixels inside the closed, hole-filled own silhouette that
    the own mask does not cover, / own area: the partner in front of the dancer = a white patch in the crop) and head_cov
    (head_cov with the dancer's keypoints in mask pixels; NaN without them)."""
    own = np.asarray(own, bool)
    a = int(own.sum())
    if a == 0:
        return {"main_frac": float("nan"), "partner_occ": float("nan"), "head_cov": float("nan")}
    n, _lab, stats, _ = cv2.connectedComponentsWithStats(own.astype(np.uint8), 8)
    comp = stats[1:, cv2.CC_STAT_AREA]
    m = clean_mask(own)
    ys = np.nonzero(m.any(1))[0]
    k = max(5, int(round(0.1 * (ys[-1] - ys[0] + 1)))) | 1
    closed = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
    closed = fill_holes(closed)
    occ = 0.0 if partner is None else float((np.asarray(partner, bool) & closed & ~own).sum() / a)
    return {"main_frac": float(comp.max() / comp.sum()), "partner_occ": occ,
            "head_cov": float("nan") if kp17 is None else head_cov(own, kp17)}


def crop_ref(img_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """One reference crop in LHM++'s input format (cand_lhmpp.lhm_crop: tight bbox of the cleaned mask, white outside it,
    PadRatioWithScale onto a white 504 x 840 canvas): uint8 RGB (840, 504, 3)."""
    c = CL.lhm_crop(img_rgb, clean_mask(mask))
    return (np.clip(c, 0, 1) * 255 + 0.5).astype(np.uint8)


def photo_ref(path: Path) -> np.ndarray:
    """A cut-out photo of a dancer: RGBA (alpha = the dancer) -> LHM++ crop; any other image is taken as already cut out on
    white and only padded to the 504 x 840 canvas."""
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise SystemExit(f"cannot read the photo {path}")
    if im.ndim == 3 and im.shape[2] == 4:
        return crop_ref(np.ascontiguousarray(im[:, :, 2::-1]), im[:, :, 3] > 127)
    rgb = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB) if im.ndim == 2 else im[:, :, ::-1]
    return (np.clip(CL.pad_ratio_with_scale(rgb.astype(np.float32) / 255.0), 0, 1) * 255 + 0.5).astype(np.uint8)


def rest_joints(betas: np.ndarray, model: Path = SMPLX_MODEL) -> np.ndarray | None:
    """(55, 3) zero-pose SMPL-X joints of the betas (J_regressor of the shaped template); None without the model file."""
    if not Path(model).exists():
        return None
    with np.load(model, allow_pickle=True) as m:
        b = np.asarray(betas, float).reshape(-1)
        v = np.asarray(m["v_template"], float) + np.einsum("vcb,b->vc", np.asarray(m["shapedirs"], float)[:, :, :len(b)], b)
        return np.asarray(m["J_regressor"], float) @ v


def pose_key(fit: dict | None, t: float, betas: np.ndarray) -> dict:
    """One pose key of one dancer for the pod (lhmpp_render_pod / recap_lhmpp_pod poses.npz names): the fit at the lattice
    instant nearest t (not absent), else the zero pose with the rest joints of the betas."""
    if fit is not None:
        T = np.asarray(fit["t"], float)
        bad = np.asarray(fit.get("absent", np.zeros(len(T), bool)), bool)
        d = np.where(bad, np.inf, np.abs(T - t))
        k = int(np.argmin(d))
        if np.isfinite(d[k]):
            g = lambda n, w: np.asarray(fit[n], np.float32)[k].reshape(1, w)  # noqa: E731
            J = np.asarray(fit["joints"], float)[k]
            return {"root": g("global_orient", 3), "body": g("body_pose", 63), "jaw": g("jaw_pose", 3),
                    "leye": g("leye_pose", 3), "reye": g("reye_pose", 3), "lhand": g("left_hand_pose", 45),
                    "rhand": g("right_hand_pose", 45), "pelvis": J[None, 0].astype(np.float64),
                    "joints22": J[None, :22].astype(np.float32), "t": float(T[k]), "src": "fit"}
    J = rest_joints(betas)
    src = "zero pose (rest joints of the betas)" if J is not None else "zero pose, no SMPL-X model here: joints 0"
    J = np.zeros((55, 3)) if J is None else J
    z = lambda w: np.zeros((1, w), np.float32)  # noqa: E731
    return {"root": z(3), "body": z(63), "jaw": z(3), "leye": z(3), "reye": z(3), "lhand": z(45), "rhand": z(45),
            "pelvis": J[None, 0].astype(np.float64), "joints22": J[None, :22].astype(np.float32), "t": float(t), "src": src}


def write_pod_inputs(inp: Path, keys: dict, betas: dict) -> None:
    """poses.npz (one key; per role root/body/jaw/leye/reye/lhand/rhand, pelvis, joints22, betas), an empty views.npz and
    views.json {"groups": {}, "passes": []}: what lhmpp_render_pod.py and recap_lhmpp_pod.py read besides refs/."""
    inp.mkdir(parents=True, exist_ok=True)
    Z = {"t": np.array([keys[ROLES[0]]["t"]], float)}
    for role in ROLES:
        for k in ("root", "body", "jaw", "leye", "reye", "lhand", "rhand", "pelvis", "joints22"):
            Z[f"{role}_{k}"] = keys[role][k]
        Z[f"{role}_betas"] = np.asarray(betas[role], np.float64).reshape(-1)
    np.savez(inp / "poses.npz", **Z)
    np.savez(inp / "views.npz")
    (inp / "views.json").write_text(json.dumps({"groups": {}, "passes": [],
                                                "note": "lhmpp_generic: avatars only (no render passes, no A2 views)"},
                                               indent=1), encoding="utf-8", newline="\n")


def job_doc(take: str, inp_ws: str, out_ws: str, export: bool = True, max_minutes: float = JOB_MINUTES) -> dict:
    """The dancecap.pod_job/1 document (inp_ws / out_ws workspace-relative, no trailing '/'; the commands run in dancecap/)."""
    cin, cout = inp_ws.removeprefix("dancecap/"), out_ws.removeprefix("dancecap/")
    args = f"--input {cin} --out {cout} --renders {cout}/renders"
    cmds = [{"name": "install", "timeout_min": 20, "cmd": "bash remote/cand_lhmpp/install.sh"},
            {"name": "build", "timeout_min": 10, "cmd": f"{POD_PY} remote/cand_lhmpp/lhmpp_render_pod.py {args}"}]
    if export:
        cmds.append({"name": "export", "timeout_min": 8, "optional": True,
                     "cmd": f"{POD_PY} remote/cand_lhmpp/recap_lhmpp_pod.py {args}"})
    return {"schema": "dancecap.pod_job/1", "name": job_name(take), "take": take, "cwd": "dancecap",
            "max_minutes": float(max_minutes), "needs": [], "min_vram_gb": 24,
            # its own venv (install.sh): no bootstrap base / check, and the install starts at pod-ready (early_commands)
            "base": False, "inputs": [inp_ws + "/"],
            "exclude": list(EXCLUDE), "env": dict(POD_ENV), "commands": cmds, "outputs": [out_ws + "/"],
            "verify": [f"{out_ws}/lhmpp_render_pod.json"] + [f"{out_ws}/gs/{r}_cano.ply" for r in ROLES] +
                      [f"{out_ws}/turntable/{r}.png" for r in ROLES],
            "after_download": [f"python -m dancecap.mono4d.lhmpp_generic import --take {take} --session <sid>"],
            "notes": [f"LHM++ ({CL.MODEL_NAME} @ {CL.LHMPP_COMMIT[:7]}, CC BY-NC 4.0 weights) canonical splat avatars of the "
                      f"two dancers of {take} from reference crops of its one phone video chosen by dancecap.mono4d."
                      "lhmpp_generic (masks + teachers + fit facing); shape ref (fit betas[:10]); lhmpp_render_pod.py with "
                      "no render passes -> gs/<role>_cano.ply + turntable/<role>.png" +
                      ("; optional export = recap_lhmpp_pod.py on the same input (LHM++'s LBS weights, "
                       "export/<role>_export.npz)" if export else "") +
                      "; weights stay on the pod; crops go to our own pod only"]}


def ply_info(path: Path) -> dict:
    """Vertex count and float properties of a binary PLY header, and whether the file is long enough for them."""
    with open(path, "rb") as f:
        head = f.read(8192)
    if b"end_header" not in head:
        return {"ok": False, "why": "no PLY header"}
    h = head.split(b"end_header")[0].decode("ascii", "replace")
    n = [int(x.split()[-1]) for x in h.splitlines() if x.startswith("element vertex")]
    props = [x.split()[-1] for x in h.splitlines() if x.startswith("property")]
    size = Path(path).stat().st_size
    need = len(h) + len("end_header\n") + (n[0] if n else 0) * 4 * len(props)
    return {"ok": bool(n and n[0] > 0 and size >= need), "gaussians": n[0] if n else 0, "properties": len(props),
            "bytes": size}


# ------------------------------------------------------------------------------------------------ take data (I/O)


def load_teachers(work: Path) -> dict:
    with np.load(work / "people" / "teachers_01.npz", allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def load_fits(work: Path) -> tuple[str | None, dict]:
    for d in FIT_DIRS:
        if all((work / d / f"{r}.npz").exists() for r in ROLES):
            fits = {}
            for r in ROLES:
                with np.load(work / d / f"{r}.npz", allow_pickle=True) as z:
                    fits[r] = {k: z[k] for k in z.files}
            return d, fits
    return None, {}


def fit_facing(fits: dict, work: Path, T: dict) -> tuple[np.ndarray | None, np.ndarray]:
    """(F, 2) facing yaw of every teachers row from the fit (NaN where no fit instant within 0.6 lattice steps / absent) and
    the (F,) reference times of the rows (cameras/01.npz t by frame index, else the teachers' t)."""
    F = len(T["t"])
    t_ref = np.asarray(T["t"], float).copy()
    cam = work / "cameras" / "01.npz"
    if not cam.exists():
        return None, t_ref
    with np.load(cam, allow_pickle=True) as z:
        fi_c = np.asarray(z["frame_index"] if "frame_index" in z.files else np.arange(len(z["t"])), int)
        Rc, tc, tt = np.asarray(z["R"], float), np.asarray(z["tvec"], float), np.asarray(z["t"], float)
    pos = {int(f): i for i, f in enumerate(fi_c)}
    rows = np.array([pos.get(int(f), -1) for f in np.asarray(T.get("frame_index", np.arange(F)), int)])
    has = rows >= 0
    t_ref[has] = tt[rows[has]]
    if not fits:
        return None, t_ref
    yaw = np.full((F, 2), np.nan)
    for role in ROLES:
        d = fits[role]
        G = np.asarray(d["t"], float)
        hz = 1.0 / float(np.median(np.diff(G))) if len(G) > 1 else 30.0
        k1 = np.clip(np.searchsorted(G, t_ref), 0, len(G) - 1)
        k0 = np.clip(k1 - 1, 0, len(G) - 1)
        k = np.where(np.abs(G[k0] - t_ref) <= np.abs(G[k1] - t_ref), k0, k1)
        absent = np.asarray(d.get("absent", np.zeros(len(G), bool)), bool)
        ok = has & (np.abs(G[k] - t_ref) <= 0.6 / hz) & ~absent[k]
        if ok.any():
            go = np.asarray(d["global_orient"], float)[k[ok]]
            pel = np.asarray(d["joints"], float)[k[ok], 0]
            yaw[ok, SLOT[role]] = body_yaw_deg(go, pel, Rc[rows[ok]], tc[rows[ok]])
    return yaw, t_ref


def betas_for(work: Path, fits: dict, fit_dir: str | None) -> tuple[dict, str]:
    if fits:
        return {r: np.asarray(fits[r]["betas"], float).reshape(-1) for r in ROLES}, f"{fit_dir}/<role>.npz betas"
    pin = work / "shape" / "mamma_pin_betas.json"
    if pin.exists():
        d = json.loads(pin.read_text(encoding="utf-8"))
        return {r: np.asarray(d[str(SLOT[r])], float) for r in ROLES}, "shape/mamma_pin_betas.json (statures, no fit)"
    return {r: np.zeros(16) for r in ROLES}, "zeros (no fit, no pinned shape)"


class _Masks:
    """people/masks_01.npz: m_<row>_<slot> PNG bytes; decoded masks resized to the video frame when it differs."""

    def __init__(self, path: Path):
        self.z = np.load(path)
        self.keys = set(self.z.files)

    def get(self, row: int, slot: int, shape: tuple[int, int] | None = None) -> np.ndarray | None:
        k = f"m_{int(row)}_{int(slot)}"
        if k not in self.keys:
            return None
        m = cv2.imdecode(np.frombuffer(self.z[k].tobytes(), np.uint8), cv2.IMREAD_GRAYSCALE)
        if m is None:
            return None
        if shape is not None and m.shape != tuple(shape):
            if m.shape == tuple(shape)[::-1] and m.shape[0] != m.shape[1]:
                raise SystemExit(f"the video decodes {shape[::-1]} (w x h) but the masks are {m.shape[::-1]}: rotated")
            m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
        return m > 127

    def close(self):
        self.z.close()


def for_frames(video: Path, wanted: dict, fn) -> set:
    """Decode the video sequentially (cv2, the people tables' frame order) and call fn(frame_index, bgr) for the wanted ones."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    seen, last, fi = set(), max(wanted) if wanted else -1, 0
    while fi <= last:
        if not cap.grab():
            break
        if fi in wanted:
            ok, fr = cap.retrieve()
            if ok:
                fn(fi, fr)
                seen.add(fi)
        fi += 1
    cap.release()
    return seen


def sharpness(gray: np.ndarray, m: np.ndarray) -> float:
    """Variance of the Laplacian over the dancer's eroded mask (the silhouette edge against the background left out), on the
    bbox scaled down to at most SHARP_H rows."""
    ys, xs = np.nonzero(m)
    c = gray[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    mm = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.uint8)
    if c.shape[0] > SHARP_H:
        sz = (max(1, int(round(c.shape[1] * SHARP_H / c.shape[0]))), SHARP_H)
        c, mm = cv2.resize(c, sz, interpolation=cv2.INTER_AREA), cv2.resize(mm, sz, interpolation=cv2.INTER_NEAREST)
    inner = cv2.erode(mm, np.ones((5, 5), np.uint8)) > 0
    lap = cv2.Laplacian(c, cv2.CV_32F)
    return float(lap[inner].var()) if inner.sum() > 50 else float(lap.var())


def survey(video: Path, M: _Masks, rows: dict, frame_of_row: np.ndarray, kp: np.ndarray, size: tuple[int, int]) -> dict:
    """partner_occ, main_frac, head_cov, sharp (F, 2) of the candidate rows {slot: rows} (one sequential decode; kp (F, 2, 17,
    3) in the people tables' pixels, size = their (W, H)); NaN elsewhere."""
    F = len(kp)
    out = {k: np.full((F, 2), np.nan) for k in MEASURED}
    want: dict = {}
    for s, rr in rows.items():
        for r in rr:
            want.setdefault(int(frame_of_row[r]), []).append((int(r), int(s)))

    def fn(fi, bgr):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        for r, s in want[fi]:
            own, par = M.get(r, s, gray.shape), M.get(r, 1 - s, gray.shape)
            if own is None or not own.any():
                continue
            k = np.array(kp[r, s], float)
            k[:, 0] *= gray.shape[1] / float(size[0])
            k[:, 1] *= gray.shape[0] / float(size[1])
            ms = mask_stats(own, par, k)
            for name in ("partner_occ", "main_frac", "head_cov"):
                out[name][r, s] = ms[name]
            out["sharp"][r, s] = sharpness(gray, clean_mask(own))

    for_frames(video, want, fn)
    return out


def _thin(rows: np.ndarray, n: int) -> np.ndarray:
    return rows if len(rows) <= n else rows[np.unique(np.linspace(0, len(rows) - 1, n).round().astype(int))]


def _label(t: np.ndarray, lines: list[tuple[str, int]]) -> np.ndarray:
    for txt, y in lines:
        cv2.putText(t, txt, (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(t, txt, (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return t


def _tile_h(img_bgr: np.ndarray, h: int, w: int) -> np.ndarray:
    s = h / img_bgr.shape[0]
    r = cv2.resize(img_bgr, (max(1, int(round(img_bgr.shape[1] * s))), h), interpolation=cv2.INTER_AREA)
    if r.shape[1] >= w:
        return np.ascontiguousarray(r[:, :w])
    return np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0)), constant_values=255)


def contact_sheet(path: Path, rows: list[dict]) -> Path:
    """Per dancer a title, the crops (as the pod gets them) and, below each, its frame with the dancer's mask outline."""
    w = int(round(TILE_H * 504 / 840))
    blocks = []
    for r in rows:
        top = [_label(_tile_h(c, TILE_H, w), [(a, 16), (b, TILE_H - 8)]) for c, a, b in r["crops"]]
        bot = [_tile_h(c, TILE_H, w) for c in r["frames"]]
        bot += [np.full((TILE_H, w, 3), 255, np.uint8)] * (len(top) - len(bot))
        line = lambda ts: np.concatenate([np.pad(t, ((0, 0), (0, 4), (0, 0)), constant_values=255) for t in ts], 1)  # noqa: E731
        body = np.concatenate([line(top), np.full((4, line(top).shape[1], 3), 255, np.uint8), line(bot)], 0)
        bar = np.full((28, body.shape[1], 3), 255, np.uint8)
        cv2.putText(bar, r["title"], (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        blocks.append(np.concatenate([bar, body], 0))
    Wm = max(b.shape[1] for b in blocks)
    sheet = np.concatenate([np.pad(b, ((0, 8), (0, Wm - b.shape[1]), (0, 0)), constant_values=255) for b in blocks], 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return path


# ------------------------------------------------------------------------------------------------ prep / job / import


def prep(take: str, out_dir: Path | None = None, n: int = N_REFS, min_gap_s: float = MIN_GAP_S,
         photos: dict | None = None) -> dict:
    """Choose the reference frames, write the pod's input folder (<out>/input/) and the contact sheet (<out>/refs_sheet.jpg).
    Local CPU, seconds for a 20 s take (one decode for the mask survey, one for the picks)."""
    from ..take import Take
    t0 = time.time()
    tk = Take.load(take)
    if len(tk.sources) != 1:
        raise SystemExit(f"{take}: lhmpp_generic needs exactly one video ({len(tk.sources)} sources)")
    video = Path(tk.sources[0].path)
    if not video.exists():
        raise SystemExit(f"{take}: the video {video} is missing")
    work = work_dir(take)
    out = Path(out_dir) if out_dir else default_dir(take)
    inp = out / "input"
    T = load_teachers(work)
    F = len(T["t"])
    M = _Masks(work / "people" / "masks_01.npz")
    size = tuple(int(v) for v in T["size"]) if "size" in T else None
    minfo = {"area": np.asarray(M.z["area"], float), "box": np.asarray(M.z["box"], float), "size": size}
    if size is None:
        cap = cv2.VideoCapture(str(video))
        minfo["size"] = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        cap.release()
    fit_dir, fits = load_fits(work)
    facing, t_ref = fit_facing(fits, work, T)
    # the mask survey on the candidates of the RELAXED gate (teachers + boxes only), at most SURVEY_MAX per dancer
    st0 = frame_stats(T, minfo, facing)
    f0 = gate_fails(st0, RELAXED)
    cand = ~np.any(np.stack(list(f0.values())), 0)
    rows = {SLOT[r]: _thin(np.nonzero(cand[:, SLOT[r]])[0], SURVEY_MAX) for r in ROLES}
    log(f"{take}: {F} frames, video {video.name} {minfo['size']}, fit {fit_dir or 'none'}; surveying "
        f"{ {r: len(rows[SLOT[r]]) for r in ROLES} } candidate dancer-frames")
    minfo.update(survey(video, M, rows, st0["frame"], np.asarray(T["kp"], float), minfo["size"]))
    sel = choose_frames(T, minfo, n, min_gap_s, facing)
    for role in ROLES:
        c = sel[role]
        log(f"{role}: {len(c['picks'])} picks (relaxed {c['n_relaxed_picks']}, gap {c['min_gap_used_s']} s) from "
            f"{c['clean']} clean of {c['hard_ok']} hard-ok / {c['frames']}; "
            f"{[(p['frame'], p['t'], p['yaw_deg']) for p in c['picks']]}")
        if not c["picks"]:
            raise SystemExit(f"{take} {role}: no usable reference frame (drops {c['drop_counts']})")
    # crops
    if (inp / "refs").exists():
        shutil.rmtree(inp / "refs")
    if (out / "pod_out").exists():      # outputs of earlier inputs: never imported by mistake
        shutil.rmtree(out / "pod_out")
    want: dict = {}
    for role in ROLES:
        for k, p in enumerate(sel[role]["picks"]):
            want.setdefault(p["frame"], []).append((role, k, p))
    crops: dict = {r: {} for r in ROLES}
    ctx: dict = {r: {} for r in ROLES}

    def fn(fi, bgr):
        for role, k, p in want[fi]:
            m = M.get(p["row"], SLOT[role], bgr.shape[:2])
            crops[role][k] = crop_ref(np.ascontiguousarray(bgr[:, :, ::-1]), m)
            th = bgr.copy()
            cs, _ = cv2.findContours(clean_mask(m).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(th, cs, -1, ROLE_BGR[role], max(2, bgr.shape[0] // 300))
            ctx[role][k] = th

    got = for_frames(video, want, fn)
    if set(want) - got:
        raise SystemExit(f"{take}: frames {sorted(set(want) - got)[:8]} could not be decoded from {video.name}")
    M.close()
    names = {r: ((tk.cfg.get("dancers") or {}).get(r) or {}).get("name") or r for r in ROLES}
    refs_files: dict = {}
    sheet_rows = []
    for role in ROLES:
        d = inp / "refs" / role
        d.mkdir(parents=True, exist_ok=True)
        files, tiles = [], []
        for k, p in enumerate(sel[role]["picks"]):
            cv2.imwrite(str(d / f"{k:02d}.png"), crops[role][k][:, :, ::-1])
            files.append({"file": f"{k:02d}.png", "kind": "video", "frame": p["frame"], "t": p["t"]})
            tiles.append((crops[role][k][:, :, ::-1], f"{k:02d} f{p['frame']} {p['t']:.1f}s",
                          f"yaw {p['yaw_deg']:.0f} {p['yaw_src']} f{p['face'] or 0:.2f} v{p['speed'] or 0:.1f} q{p['q']:.2f}"
                          + (" R" if p["relaxed"] else "")))
        ph = (photos or {}).get(role)
        if ph:
            k = len(files)
            im = photo_ref(Path(ph))
            cv2.imwrite(str(d / f"{k:02d}.png"), im[:, :, ::-1])
            files.append({"file": f"{k:02d}.png", "kind": "photo", "source": Path(ph).name})
            tiles.append((im[:, :, ::-1], f"{k:02d} photo", Path(ph).name[:24]))
        refs_files[role] = files
        sheet_rows.append({"title": f"{names[role]} ({role}): {len(files)} LHM++ reference crops (504 x 840, white outside the "
                                    f"mask; 00 = most frontal); below: the frame, mask outline",
                           "crops": tiles, "frames": [ctx[role][k] for k in range(len(sel[role]["picks"]))]})
    betas, betas_src = betas_for(work, fits, fit_dir)
    keys = {}
    for role in ROLES:
        p0 = sel[role]["picks"][0]
        keys[role] = pose_key(fits.get(role), float(t_ref[p0["row"]]), betas[role])
    write_pod_inputs(inp, keys, betas)
    rules = {"n": n, "min_gap_s": min_gap_s, "clean": CLEAN, "relaxed": RELAXED, "area_min_px": AREA_MIN_PX,
             "edge_px": EDGE_PX, "frontal_deg": FRONTAL_DEG, "survey_max": SURVEY_MAX,
             "facing": f"fit {fit_dir} (cand_lhmpp.body_yaw_deg)" if facing is not None else "2D shoulders (yaw_2d)"}
    refs_doc = {"take": take, "video": video.name, "rules": rules,
                **{r: {**{k: v for k, v in sel[r].items()}, "files": refs_files[r]} for r in ROLES}}
    (inp / "refs.json").write_text(json.dumps(refs_doc, indent=1, default=float), encoding="utf-8", newline="\n")
    sheet = contact_sheet(out / "refs_sheet.jpg", sheet_rows)
    info = {"take": take, "video": video.name, "size": list(minfo["size"]), "frames": F, "fit_dir": fit_dir,
            "betas_source": betas_src, "betas10": {r: [round(float(x), 4) for x in betas[r][:10]] for r in ROLES},
            "pose_key": {r: {"t": keys[r]["t"], "src": keys[r]["src"]} for r in ROLES},
            "refs": {r: len(refs_files[r]) for r in ROLES},
            "relaxed_picks": {r: sel[r]["n_relaxed_picks"] for r in ROLES},
            "surveyed": {r: int(len(rows[SLOT[r]])) for r in ROLES}, "input": ws_rel(inp), "sheet": ws_rel(sheet),
            "bytes": int(sum(p.stat().st_size for p in inp.rglob("*") if p.is_file())),
            "seconds": round(time.time() - t0, 1)}
    (inp / "prep.json").write_text(json.dumps(info, indent=1), encoding="utf-8", newline="\n")
    log(f"prep: {json.dumps({k: info[k] for k in ('refs', 'relaxed_picks', 'fit_dir', 'betas_source', 'bytes', 'seconds')})}")
    log(f"contact sheet -> {sheet}")
    return info


def job(take: str, out_dir: Path | None = None, path: Path | None = None, export: bool = True,
        max_minutes: float = JOB_MINUTES) -> Path:
    """Write the pod job json (default <out>/<take>_lhmpp_avatars.json) for the prepared input folder; returns its path."""
    out = Path(out_dir) if out_dir else default_dir(take)
    inp = out / "input"
    for role in ROLES:
        if not any((inp / "refs" / role).glob("*.png")):
            raise SystemExit(f"{take}: no reference crops in {inp / 'refs' / role}: run prep first")
    for f in ("poses.npz", "views.npz", "views.json"):
        if not (inp / f).exists():
            raise SystemExit(f"{take}: {inp / f} is missing: run prep first")
    d = job_doc(take, ws_rel(inp), ws_rel(out / "pod_out"), export=export, max_minutes=max_minutes)
    p = Path(path) if path else out / f"{job_name(take)}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, indent=1), encoding="utf-8", newline="\n")
    log(f"job {d['name']} -> {p}")
    return p


def _json(p: Path) -> dict:
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _session_meta(take: str, sid: str | None) -> dict:
    from ..pipeline import paths
    root = paths.sessions_dir()
    name = job_name(take)
    if sid is None:
        cands = sorted((p for p in root.glob("*") if (p / "result.json").exists()
                        and f'"{name}"' in (p / "result.json").read_text(encoding="utf-8", errors="ignore")),
                       key=lambda p: p.stat().st_mtime) if root.is_dir() else []
        if not cands:
            return {}
        sid = cands[-1].name
    p = root / sid / "result.json"
    if not p.exists():
        return {"session": sid}
    r = json.loads(p.read_text(encoding="utf-8"))
    pod = r.get("pod") or {}
    j = (r.get("jobs") or {}).get(name) or {}
    return {"session": sid, "gpu": pod.get("gpu"), "cost_per_hr": pod.get("cost_per_hr"), "cost_usd": pod.get("ledger_cost"),
            "pod_minutes": round((pod.get("lifetime_s") or 0) / 60, 1), "job_status": j.get("status"),
            "commands": j.get("commands"), "job_seconds": j.get("seconds")}


def import_(take: str, session: str | None = None, out_dir: Path | None = None, dest: Path | None = None) -> str:
    """Verify the pod outputs and copy them into work/<take>/avatar_lhmpp/ (gs/<role>_cano.ply, turntable/<role>.png,
    export/<role>_export.npz when built, refs_sheet.jpg, avatar_lhmpp.json). Returns a one-line summary; raises when an
    avatar is missing."""
    out = Path(out_dir) if out_dir else default_dir(take)
    po = out / "pod_out"
    dst = Path(dest) if dest else work_dir(take) / "avatar_lhmpp"
    pj = po / "lhmpp_render_pod.json"
    if not pj.exists():
        raise SystemExit(f"{take}: {pj} is missing (the pod job did not come home)")
    pod = json.loads(pj.read_text(encoding="utf-8"))
    rec, prep_, refs = (_json(f) for f in (po / "recap_lhmpp_pod.json", out / "input" / "prep.json", out / "input" / "refs.json"))
    dancers, problems = {}, []
    for role in ROLES:
        ply, tt = po / "gs" / f"{role}_cano.ply", po / "turntable" / f"{role}.png"
        if not ply.exists():
            raise SystemExit(f"{take}: {ply} is missing (pod errors: {pod.get('errors')})")
        pi = ply_info(ply)
        if not pi["ok"]:
            problems.append(f"{role} PLY {pi}")
        (dst / "gs").mkdir(parents=True, exist_ok=True)
        shutil.copy2(ply, dst / "gs" / ply.name)
        im = cv2.imread(str(tt)) if tt.exists() else None
        if im is None:
            problems.append(f"{role} turntable missing")
        else:
            (dst / "turntable").mkdir(parents=True, exist_ok=True)
            shutil.copy2(tt, dst / "turntable" / tt.name)
        ex = po / "export" / f"{role}_export.npz"
        if ex.exists():
            (dst / "export").mkdir(parents=True, exist_ok=True)
            shutil.copy2(ex, dst / "export" / ex.name)
        pd = (pod.get("dancers") or {}).get(role) or {}
        dancers[role] = {"cano_ply": ws_rel(dst / "gs" / ply.name), "ply": pi,
                         "turntable": ws_rel(dst / "turntable" / tt.name) if im is not None else None,
                         "turntable_size": None if im is None else [int(im.shape[1]), int(im.shape[0])],
                         "export": ws_rel(dst / "export" / ex.name) if ex.exists() else None,
                         **{k: pd.get(k) for k in ("n_refs", "refs", "betas_used", "rest_height_m", "infer_s",
                                                   "fk10_vs_ref_joints_mm", "opacity_median", "scale_median_mm", "error")},
                         "picks": [{k: p.get(k) for k in ("frame", "t", "yaw_deg", "yaw_src", "q", "relaxed")}
                                   for p in (refs.get(role) or {}).get("picks", [])]}
    if (out / "refs_sheet.jpg").exists():
        shutil.copy2(out / "refs_sheet.jpg", dst / "refs_sheet.jpg")
    ses = _session_meta(take, session)
    summ = {"schema": "dancecap.avatar_lhmpp/1", "take": take, "model": CL.MODEL_NAME, "commit": CL.LHMPP_COMMIT,
            "licence": CL.LICENCE, "how": "dancecap.mono4d.lhmpp_generic: crops chosen from masks + teachers + fit facing; "
            "lhmpp_render_pod.py (shape mode ref = betas[:10]) -> canonical A-pose 3DGS PLY; optional recap_lhmpp_pod.py "
            "export (LHM++ LBS weights, SPLAT_AVATAR_FORMAT.md section 7)",
            "session": ses, "prep": {k: prep_.get(k) for k in ("fit_dir", "betas_source", "refs", "relaxed_picks", "pose_key")},
            "dancers": dancers, "pod": {"versions": pod.get("versions"), "timings": pod.get("timings"),
                                        "errors": pod.get("errors"), "commit": pod.get("commit")},
            "export_pod": {k: rec.get(k) for k in ("timings", "errors")} if rec else None,
            "problems": problems, "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "avatar_lhmpp.json").write_text(json.dumps(summ, indent=1, default=str), encoding="utf-8", newline="\n")
    g = {r: dancers[r]["ply"].get("gaussians") for r in ROLES}
    nref = {r: dancers[r].get("n_refs") or len(dancers[r]["picks"]) for r in ROLES}
    line = (f"LHM++ avatars: lead {g['lead']} / follow {g['follow']} Gaussians from {nref['lead']} / {nref['follow']} crops "
            f"(betas: {prep_.get('betas_source', '?')}), turntables "
            f"{'ok' if all(dancers[r]['turntable'] for r in ROLES) else 'MISSING'}"
            + ("; LBS export" if all(dancers[r]["export"] for r in ROLES) else "; no LBS export")
            + f" -> {ws_rel(dst).removeprefix('dancecap/')}/" + (f" (session {ses['session']})" if ses.get("session") else "")
            + (f"; pod errors: {'; '.join(map(str, pod['errors']))[:200]}" if pod.get("errors") else "")
            + (f"; PROBLEMS: {'; '.join(problems)[:200]}" if problems else ""))
    log(line)
    return line


# ------------------------------------------------------------------------------------------------ CLI


def _photos(items: list[str] | None) -> dict:
    out = {}
    for it in items or []:
        role, _, p = it.partition("=")
        if role not in ROLES or not p:
            raise SystemExit(f"--photo wants lead=PATH or follow=PATH, got {it!r}")
        out[role] = p
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.mono4d.lhmpp_generic",
                                 description="LHM++ splat avatars of a one-phone take (prep -> job -> remote run -> import)")
    ap.add_argument("cmd", choices=["prep", "job", "import"])
    ap.add_argument("--take", required=True)
    ap.add_argument("--out", default=None, help="working folder (default work/<take>/realism/lhmpp_generic)")
    ap.add_argument("--n", type=int, default=N_REFS, help="reference crops per dancer (prep)")
    ap.add_argument("--min-gap", type=float, default=MIN_GAP_S, help="seconds between two picks of a dancer (prep)")
    ap.add_argument("--photo", action="append", help="role=PATH: a cut-out photo (RGBA) appended to that dancer's crops (prep)")
    ap.add_argument("--no-export", action="store_true", help="job: no LBS-weights export command")
    ap.add_argument("--max-minutes", type=float, default=JOB_MINUTES)
    ap.add_argument("--job-path", default=None, help="job: where to write the json (default <out>/<take>_lhmpp_avatars.json)")
    ap.add_argument("--session", default=None, help="import: the remote session id (default: the newest that ran the job)")
    a = ap.parse_args(argv)
    out = Path(a.out) if a.out else None
    if a.cmd == "prep":
        prep(a.take, out, a.n, a.min_gap, _photos(a.photo))
    elif a.cmd == "job":
        print(job(a.take, out, Path(a.job_path) if a.job_path else None, export=not a.no_export, max_minutes=a.max_minutes))
    else:
        print(import_(a.take, a.session, out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
