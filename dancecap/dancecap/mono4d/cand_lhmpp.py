r"""mono4d track "lhmpp" (2026-10-19): LHM++ appearance from ONE handheld phone (06) of larissa_kadu_38.

LHM++ (github.com/aigc3d/LHM-plusplus @ 906b5d9, Tongyi Lab / Alibaba; code Apache-2.0 except the EG3D parts under the
NVIDIA source-code licence - LICENSE_NVIDIA - and the WEIGHTS under CC BY-NC 4.0 - LICENSE_WEIGHT; the ModelScope model
card says "Apache License 2.0", the repo's LICENSE_WEIGHT is the stricter text, so treat the weights as non-commercial)
builds an animatable 3D Gaussian human (160k Gaussians, RGB, rigged to SMPL-X through a diffused voxel skinning field)
from a few pose-free images in about a second.

What is tested here is APPEARANCE ONLY: the avatar is animated with the REFERENCE poses (the 8-phone MAMMA SMPL-X fit,
smplx/<role>.npz) and rendered into three phones it never saw (01, 03, 05) at their native resolution, then compared
with the real frames inside the dancers' SAM 2.1 masks.

Inputs (prep, local CPU) -> work/<take>/mono4d/lhmpp/input/:
    refs/<role>/<set>/NN.png   reference crops of phone 06 in LHM++'s own input format: the dancer's mask bbox, white
                               outside the mask, scaled + centred on a white 504x840 (w x h) canvas exactly like its
                               PadRatioWithScale(5/3, [840]) pipeline; NN = 00 is the most frontal view
                    sets:  w8  = 8 frames inside 58-62 s;  t16 = 16 frames from the whole take (30-68 s) of phone 06,
                           chosen for visibility (projected reference joints inside the dancer's own mask), size,
                           sharpness and spread of body yaw relative to the phone
    refs.json                  the chosen frames (i, t, yaw_deg, vis, area, sharpness) per role and set
    views.json                 the 30 held-out views: phones 01/03/05 every 0.4 s of 58-62 s (avatar_texture.camera_views:
                               frame i, t = sync-corrected pose time, K, R, t_w world->camera OpenCV, width, height, k1)
    poses.json                 per role: betas (16, MAMMA's SMPL-X neutral = the standard SMPLX_NEUTRAL.npz, identical
                               file), and per view the reference SMPL-X parameters at the view's time (per-joint slerp,
                               linear translation between grid frames), the reference pelvis and 22 body joints
    real/<cam>_<i>.jpg, masks/<cam>_<i>_<01|02>.png   the held-out frames and dancer masks
    baseline/<cam>_<i>.png, baseline/<cam>_<i>_id.png the current footage-baked texture avatar (avatar/<role>_texels:
                               naked SMPL-X + baked albedo, BAKED FROM ALL 8 PHONES INCLUDING 01/03/05) rendered locally
                               with avatar_texture.render on the same posed bodies: an in-sample reference, not a
                               one-phone competitor
Pod (remote/cand_lhmpp/: install.sh + lhmpp_pod.py, python 3.10 venv = the upstream-tested torch 2.3.0 / cu121 stack):
    per role x set x shape mode one LHM++ forward (LHMPP-700M-PixelShuffle, the default GS-output model, from
    ModelScope Damo_XR_Lab), then animate_gs_model with the reference poses and gsplat rasterisation (the renderer's own
    settings) of BOTH dancers together into each held-out view (occlusion between them handled), plus a role channel.
    shape modes:  ref   betas = the reference betas[:10] fed in, use_pred_shape_for_render False (shape and pose from
                        the 8-phone fit: appearance isolated)
                  pred  LHM++'s own pipeline (SMPLX-FREE: betas 0 in, its predicted shape merged for rendering)
    Pose transfer (SMPL-X -> LHM++'s SMPL-X): no conversion of the joint rotations is needed - LHM++ uses the
    standard SMPL-X kinematic tree (smplx.create(..., use_pca=False, flat_hand_mean=True), the same 55 joints and hand
    convention as the reference); only the root translation changes because LHM++ has 10 betas (we 16) or its own
    shape: trans = pelvis_ref - J0_rest(betas_used). The pod checks it: standard SMPL-X FK of the transferred
    parameters vs the reference joints, and the posed Gaussians' distance to that posed mesh.
Score (pod, system python + dancecap): per view and dancer, a crop around the dancer's real mask; inside the mask
    PSNR / SSIM / LPIPS (+ after a least-squares colour fit: *_cc) with dancer_splats.evaluate._crop_scores (the same
    metric code as the multi-view dancer splats); silhouette IoU of the rendered role vs the real mask.
Import (local): verification, a PSNR/SSIM cross-check, contact sheets real | LHM++ | baked avatar, report.json.

2026-10-19 run (RTX 4090 secure, 6.6 pod-min, $0.10; three earlier attempts $0.22: a pod that never booted, chumpy
missing, CUDA OOM without flash-attn): all 8 avatars built (peak 8.8 GB with flash-attn), 30 held-out views x 2 dancers.
Medians over 60 dancer-views, inside the real masks: w8_ref PSNR 12.6 dB / SSIM 0.23 / LPIPS 0.203, IoU 0.75,
coverage 0.86; the 8-phone baked avatar 11.2 / 0.20 / 0.216, IoU 0.69, coverage 0.75 (it has no clothes / hair
geometry) but better colour where both cover the dancer (PSNR 14.5 vs 13.5 dB, LPIPS 0.126 vs 0.149).

    python -m dancecap.mono4d.cand_lhmpp prep [--no-baseline]
    python -m dancecap.mono4d.cand_lhmpp job [--max-minutes 60]          # remote/cand_lhmpp/pod_job.json
    (DANCECAP_RUNPOD_CAP=... python -m dancecap remote run --jobs <that json> --max-cost 3 --gpus ...)
    python -m dancecap.mono4d.cand_lhmpp score --dir <lhmpp dir>          # pod (or local without LPIPS)
    python -m dancecap.mono4d.cand_lhmpp import [--session SID]

COMPARE MODE (2026-10-19, the owner's avatar-vs-moving-splats decision): every view of the appearance-comparison spec
(work/<take>/mono4d/compare/views.json, dancecap.mono4d.appearance_compare spec) rendered by two LHM++ variants, both
dancers posed by the reference SMPL-X with the SAME transfer as above (shape mode ref: betas[:10], trans = pelvis_ref -
J0_rest) and rendered together (mutual occlusion: one gsplat rasterisation of both dancers' Gaussians):
    lhmpp_1phone   the earlier best w8_ref (8 crops of phone 06, 58-62 s) with every crop DROPPED whose dancer mask has
                   a white-patch risk (below), replaced by the next-best phone-06 frames under the same greedy rule
    lhmpp_8phone   16 crops from all 8 phones, 2 per phone, same rules (global greedy: best frontal first, then quality x
                   yaw spread, cap 2 per phone, 0.2 s apart per phone; area / sharpness normalised per phone); frames
                   within 0.05 s of a spec real view of the same phone are not used (its real-view panels stay
                   out-of-frame)
    selection      _pick's base gate (vis >= 0.9, inside the image, not cut by the border) + no OCC_LIMITS exceeded;
                   when the clean frames cannot fill the set at 0.2 s spacing the greedy continues at 0.1 s, then 2
                   frames, and only then admits the least-bad flagged crops (marked relaxed; none were needed)
    white-patch risk, from both reference bodies rasterised into the frame (avatar_raster, ~1 cm cells) and the SAM 2.1
    masks:
        white_frac     part of the dancer's projected body WITHOUT the hands (eroded 2 cells for fit error) that the
                       dancer's own mask does not cover: it turns white in the crop (the partner's arm / hand / hair in
                       front, a mask hole or miss); limit 1.5 % (06/959: 2 % already gave the visible patch on the
                       lead's back; the shoulder bands of 06/896, 887, 905 were 3.6-9.2 %)
        bad_hole_frac  enclosed holes of the dancer's mask on the visible body or under the partner's mask, relative
                       to the mask area (legit gaps such as an arm on the hip are kept); limit 0.2 %
        occ_model      part of the projected body behind the partner's body (> 3 cm nearer): reported only (hand
                       holding alone exceeds 3 %); white_partner_frac, white_vis_frac: reported only
    compare-prep   (local CPU, ~3 min) -> compare/lhmpp/input/ (the pod's only input, 19 MB): refs/<role>/<variant>/
                   NN.png, refs.json (picks with their stats, dropped crops and why), cameras.json (every render: id,
                   group, out path, pose key, K, R, tvec, W, H), poses.json (per pose key the reference SMPL-X
                   parameters), real/ + masks/ (the 18 real views, for the score), prep.json; and compare/lhmpp/
                   candidates.json (every dancer-frame of 58-62 s on all phones; --reuse-candidates skips the survey),
                   refs_<role>_<variant>.jpg, dropped_<role>.jpg (the earlier w8 crops with the verdict)
    compare-job    -> remote/cand_lhmpp/pod_job_compare.json (install.sh + lhmpp_compare_pod.py + compare-score)
    compare-score  (pod, system python, LPIPS) masked metrics of the 18 real views -> compare/lhmpp/pod_out/metrics.json
    compare-import (local) native close-up crops, renders/<variant>/manifest.json, review sheets, compare/lhmpp/report.json
Renders: compare/renders/<variant>/<view_id>.png (RGBA, straight alpha, transparent background; real views at the
phone's native size, turntable 1080 x 1080), role/<view_id>.png (R lead, G follow, B alpha; real views),
clip_vr_seat/<j:04d>.png + clip_cam03/<j:04d>.png (j = clip index 0..104, grid k = clip.k[0] + j),
closeup/<view_id>_<box>.png (native-pixel crop of the real-view render), closeup_zoom/<view_id>_<box>.png (the same
box re-rendered at 512 px on the long side: more detail than the photo has).
2026-10-19 compare run (session 20261019-182851-batch, RTX 4090 secure $0.89/h, 8.5 pod-min, $0.126; a first attempt
found no stock and created no pod): 4 avatars x 160k Gaussians, 2 x 276 renders, 0 failures. Clean crops: phone 06 31 /
28 of 86 / 109 base-gate dancer-frames (lead / follow); 6 of the 8 earlier lead crops and 7 of the 8 follow crops were
dropped (min spacing fell to 0.1 s); the white patch on the lead's back is gone. Medians over 36 dancer-views of the
18 real views (inside the real masks): 1phone PSNR 12.3 dB / SSIM 0.30 / LPIPS 0.188, 8phone 12.8 / 0.35 / 0.180.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from ..take import PACKAGE_ROOT, WORKSPACE

TAKE, CAM = "larissa_kadu_38", "06"
WINDOW = (58.0, 62.0)
HOLDOUT = ("01", "03", "05")
STEP_S = 0.4
ROLES = ("lead", "follow")
PID = {"lead": "01", "follow": "02"}
WORK = PACKAGE_ROOT / "work" / TAKE
OUT = WORK / "mono4d" / "lhmpp"
INPUT = OUT / "input"
POD_OUT = OUT / "pod_out"
JOB_DIR = PACKAGE_ROOT / "remote" / "cand_lhmpp"
LHMPP_COMMIT = "906b5d9fb967ab42efb92f6fa55bf22cac86b653"
MODEL_NAME = "LHMPP-700M-PixelShuffle"
SETS = {"w8": 8, "t16": 16}
SHAPES = ("ref", "pred")
CANVAS_HW = (840, 504)  # LHM++ PadRatioWithScale(target_ratio=5/3, tgt_max_size_list=[840]): h 840, w (840/(5/3))//14*14
LICENCE = ("LHM++ code Apache-2.0 (EG3D-derived parts: NVIDIA source code licence, non-commercial); LHM++ weights "
           "CC BY-NC 4.0 per the repo's LICENSE_WEIGHT (ModelScope card: Apache-2.0); SMPL-X / FLAME / MANO model files "
           "of LHMPP-Prior: their own non-commercial licences. Research use only; splats of the dancers stay local "
           "(never committed, CONTRACTS licence row).")


def log(msg: str) -> None:
    print(f"[cand_lhmpp] {msg}", flush=True)


def _ws(p: Path) -> str:
    return str(Path(p).resolve().relative_to(WORKSPACE.resolve())).replace("\\", "/")


def _key(cam: str, i: int) -> str:
    return f"{cam}_{int(i):06d}"


# ------------------------------------------------------------------------------------------------ geometry helpers


ROT_KEYS = (("global_orient", 1), ("body_pose", 21), ("jaw_pose", 1), ("leye_pose", 1), ("reye_pose", 1),
            ("left_hand_pose", 15), ("right_hand_pose", 15))


def _load_fit(role: str) -> dict:
    with np.load(WORK / "smplx" / f"{role}.npz", allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def interp_params(d: dict, t: float) -> dict:
    """Reference SMPL-X parameters at time t: per-joint slerp of the 55 rotations, linear translation and joints."""
    from scipy.spatial.transform import Rotation as Rot
    T = np.asarray(d["t"], float)
    k = int(np.clip(np.searchsorted(T, t) - 1, 0, len(T) - 2))
    a = float(np.clip((t - T[k]) / (T[k + 1] - T[k]), 0.0, 1.0))
    out = {"t": float(t), "k": k, "a": a}
    for name, n in ROT_KEYS:
        r0 = Rot.from_rotvec(np.asarray(d[name][k], float).reshape(n, 3))
        r1 = Rot.from_rotvec(np.asarray(d[name][k + 1], float).reshape(n, 3))
        rel = (r0.inv() * r1).as_rotvec()
        out[name] = (r0 * Rot.from_rotvec(rel * a)).as_rotvec().reshape(-1)
    out["transl"] = (1 - a) * np.asarray(d["transl"][k], float) + a * np.asarray(d["transl"][k + 1], float)
    J = (1 - a) * np.asarray(d["joints"][k], float) + a * np.asarray(d["joints"][k + 1], float)
    out["joints"] = J
    return out


def project(P: np.ndarray, K, R, t) -> tuple[np.ndarray, np.ndarray]:
    Xc = P @ np.asarray(R, float).T + np.asarray(t, float)
    z = Xc[:, 2]
    uv = (Xc[:, :2] / np.maximum(z, 1e-6)[:, None]) * np.diag(K)[:2] + np.asarray(K)[:2, 2]
    return uv, z


def body_yaw_deg(global_orient: np.ndarray, pelvis: np.ndarray, R, t) -> float:
    """Signed yaw of the camera seen from the body: 0 = the dancer faces the phone, +-180 = back to it."""
    from scipy.spatial.transform import Rotation as Rot
    fwd = Rot.from_rotvec(global_orient).apply([0.0, 0.0, 1.0])  # SMPL-X canonical body faces +Z
    C = -np.asarray(R, float).T @ np.asarray(t, float)
    d = C - pelvis
    f2, d2 = np.array([fwd[0], fwd[2]]), np.array([d[0], d[2]])
    ang = math.atan2(f2[0] * d2[1] - f2[1] * d2[0], f2 @ d2)
    return float(np.degrees(ang))


def _mask(cam: str, i: int, role: str) -> np.ndarray | None:
    m = cv2.imread(str(WORK / "dancers" / cam / "masks" / f"mask_{int(i):04d}_{PID[role]}.png"), cv2.IMREAD_GRAYSCALE)
    return None if m is None else m > 127


# ------------------------------------------------------------------------------------------------ reference crops


def pad_ratio_with_scale(img: np.ndarray) -> np.ndarray:
    """LHM++ core.datasets.data_utils.PadRatioWithScale(target_ratio=5/3, tgt_max_size_list=[840], val=True) for one
    float image in [0, 1]: scale to fit 840 x 504 keeping the aspect (INTER_AREA both ways), centre on white."""
    H, W = CANVAS_HW
    h, w = img.shape[:2]
    s = min(H / h, W / w)
    nw, nh = int(w * s), int(h * s)
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.ones((H, W, 3), img.dtype)
    y0, x0 = (H - nh) // 2, (W - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = r
    return out


def lhm_crop(img_rgb: np.ndarray, m: np.ndarray) -> np.ndarray:
    """src_center_crop_according_to_mask (tight mask bbox, enlarge 1.0) with the background set to white."""
    ys, xs = np.nonzero(m)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    f = img_rgb[y0:y1, x0:x1].astype(np.float32) / 255.0
    mm = m[y0:y1, x0:x1, None].astype(np.float32)
    return pad_ratio_with_scale(f * mm + (1 - mm))


def _candidates(fits: dict, t_lo: float, t_hi: float, stride: int) -> list[dict]:
    """Per frame of phone 06 and dancer: visibility of the reference body in the dancer's own mask, size, sharpness,
    yaw. vis = fraction of the 22 projected body joints (+ head) inside the dancer's mask (dilated 1.5 % of the image
    height) and inside the image: low when the partner hides part of the body or the body leaves the frame."""
    from .. import mamma_dyn as md
    tr = md.load_tracks(WORK, [CAM])[CAM]
    fj = json.loads((WORK / "frames" / CAM / "frames.json").read_text())
    frames = fj["frames"]
    Wimg, Himg = int(fj["width"]), int(fj["height"])
    rad = max(3, int(round(0.015 * Himg)))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rad + 1, 2 * rad + 1))
    rows = []
    for i in range(0, len(frames), stride):
        t = float(tr.t[i])
        if not (t_lo <= t <= t_hi) or not bool(tr.ok[i]):
            continue
        img = None
        for role in ROLES:
            d = fits[role]
            if t < d["t"][0] or t > d["t"][-1]:
                continue
            m = _mask(CAM, i, role)
            if m is None or m.sum() < 2000:
                continue
            p = interp_params(d, t)
            J = p["joints"][list(range(22)) + [15]]
            uv, z = project(J, tr.K[i], tr.R[i], tr.tvec[i])
            md_ = cv2.dilate(m.astype(np.uint8), ker) > 0
            u, v = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
            inside = (u >= 0) & (u < Wimg) & (v >= 0) & (v < Himg) & (z > 0)
            hit = np.zeros(len(u), bool)
            hit[inside] = md_[v[inside], u[inside]]
            ys, xs = np.nonzero(m)
            border = xs.min() <= 3 or ys.min() <= 3 or xs.max() >= Wimg - 4 or ys.max() >= Himg - 4
            if img is None:
                img = cv2.imread(str(WORK / "frames" / CAM / frames[i]["file"]), cv2.IMREAD_GRAYSCALE)
            crop = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            sharp = float(cv2.Laplacian(cv2.resize(crop, None, fx=0.5, fy=0.5), cv2.CV_32F).var())
            rows.append({"role": role, "i": i, "t": round(t, 4), "vis": float(hit.mean()), "inimg": float(inside.mean()),
                         "area": int(m.sum()), "box": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                         "border": bool(border), "sharp": sharp,
                         "yaw_deg": round(body_yaw_deg(np.asarray(p["global_orient"]), p["joints"][0], tr.R[i],
                                                       tr.tvec[i]), 1)})
    return rows


def _pick(rows: list[dict], n: int, min_dt: float) -> list[dict]:
    """Greedy: best near-frontal first, then maximise quality x (yaw distance to the picked set + 0.15)."""
    ok = [r for r in rows if r["vis"] >= 0.9 and r["inimg"] == 1.0 and not r["border"]]
    if len(ok) < n:
        ok = sorted(rows, key=lambda r: -(r["vis"] - 0.5 * r["border"]))[:max(n * 3, len(ok))]
    if not ok:
        return []
    a_med = np.median([r["area"] for r in ok])
    s_med = np.median([r["sharp"] for r in ok])
    for r in ok:
        r["q"] = r["vis"] ** 2 * min(1.0, r["area"] / a_med) * min(1.0, r["sharp"] / s_med) ** 0.5
    front = [r for r in ok if abs(r["yaw_deg"]) <= 40] or ok
    picked = [max(front, key=lambda r: r["q"])]
    while len(picked) < n:
        best, best_s = None, -1.0
        for r in ok:
            if any(abs(r["t"] - p["t"]) < min_dt for p in picked):
                continue
            dy = min(abs((r["yaw_deg"] - p["yaw_deg"] + 180) % 360 - 180) for p in picked)
            s = r["q"] * (dy / 180.0 + 0.15)
            if s > best_s:
                best, best_s = r, s
        if best is None:
            break
        picked.append(best)
    return picked


def prep_refs(fits: dict) -> dict:
    fj = json.loads((WORK / "frames" / CAM / "frames.json").read_text())
    frames = fj["frames"]
    t_take = (max(float(f["t"][0]) for f in fits.values()), min(float(f["t"][-1]) for f in fits.values()))
    rows_w = _candidates(fits, *WINDOW, stride=1)
    rows_t = _candidates(fits, *t_take, stride=2)
    out = {}
    d = INPUT / "refs"
    if d.exists():
        shutil.rmtree(d)
    for role in ROLES:
        out[role] = {}
        for name, n in SETS.items():
            rows = [r for r in (rows_w if name == "w8" else rows_t) if r["role"] == role]
            picks = _pick(rows, n, 0.2 if name == "w8" else 0.6)
            sd = d / role / name
            sd.mkdir(parents=True, exist_ok=True)
            tiles = []
            for k, r in enumerate(picks):
                img = cv2.imread(str(WORK / "frames" / CAM / frames[r["i"]]["file"]))[:, :, ::-1]
                c = lhm_crop(img, _mask(CAM, r["i"], role))
                u8 = (np.clip(c, 0, 1) * 255 + 0.5).astype(np.uint8)
                cv2.imwrite(str(sd / f"{k:02d}.png"), u8[:, :, ::-1])
                tiles.append(cv2.resize(u8, (126, 210), interpolation=cv2.INTER_AREA))
            cv2.imwrite(str(OUT / f"refs_{role}_{name}.jpg"), np.concatenate(tiles, 1)[:, :, ::-1],
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            out[role][name] = {"n": len(picks), "candidates": len(rows),
                               "frames": [{k2: r[k2] for k2 in ("i", "t", "yaw_deg", "vis", "area", "sharp", "box")}
                                          for r in picks]}
            log(f"{role} {name}: {len(picks)} refs from {len(rows)} candidates, yaw "
                f"{[int(r['yaw_deg']) for r in picks]}")
    return out


# ------------------------------------------------------------------------------------------------ held-out views, poses


def prep_views(fits: dict) -> tuple[list[dict], dict]:
    from ..avatar_texture import camera_views
    views, _ = camera_views(WORK, list(HOLDOUT), WINDOW[0], WINDOW[1], STEP_S)
    rows = []
    for d in ("real", "masks"):
        if (INPUT / d).exists():
            shutil.rmtree(INPUT / d)
        (INPUT / d).mkdir(parents=True)
    for v in views:
        if not v["ok"]:
            continue
        fj = json.loads((WORK / "frames" / v["cam"] / "frames.json").read_text())
        src = WORK / "frames" / v["cam"] / fj["frames"][v["i"]]["file"]
        img = cv2.imread(str(src))
        H, W = img.shape[:2]
        key = _key(v["cam"], v["i"])
        shutil.copy2(src, INPUT / "real" / f"{key}.jpg")
        have = {}
        for role in ROLES:
            p = WORK / "dancers" / v["cam"] / "masks" / f"mask_{v['i']:04d}_{PID[role]}.png"
            have[role] = p.exists()
            if p.exists():
                shutil.copy2(p, INPUT / "masks" / f"{key}_{PID[role]}.png")
        rows.append({"key": key, "cam": v["cam"], "i": int(v["i"]), "t": float(v["t"]), "t_frame": float(v["t_frame"]),
                     "K": np.asarray(v["K"], float).tolist(), "R": np.asarray(v["R"], float).tolist(),
                     "t_w": np.asarray(v["t_w"], float).tolist(), "k1": float(v["k1"]), "width": W, "height": H,
                     "masks": have})
    poses = {}
    for role in ROLES:
        d = fits[role]
        fr = []
        for v in rows:
            p = interp_params(d, v["t"])
            fr.append({"key": v["key"], "t": v["t"], "root_pose": p["global_orient"].tolist(),
                       "body_pose": p["body_pose"].tolist(), "jaw_pose": p["jaw_pose"].tolist(),
                       "leye_pose": p["leye_pose"].tolist(), "reye_pose": p["reye_pose"].tolist(),
                       "lhand_pose": p["left_hand_pose"].tolist(), "rhand_pose": p["right_hand_pose"].tolist(),
                       "transl": p["transl"].tolist(), "pelvis": p["joints"][0].tolist(),
                       "joints22": p["joints"][:22].tolist()})
        poses[role] = {"betas": np.asarray(d["betas"], float).reshape(-1).tolist(),
                       "num_betas": int(d.get("num_betas", 16)), "flat_hand_mean": bool(d.get("flat_hand_mean", True)),
                       "model": str(d.get("model", "")), "frames": fr}
    return rows, poses


def check_transfer(poses: dict) -> dict:
    """Local check of the parameter interpolation and of the 16 -> 10 betas truncation LHM++ needs (standard
    SMPL-X, smplx package, the workspace's SMPLX_NEUTRAL.npz = MAMMA's file)."""
    import smplx
    import torch
    out = {}
    for role in ROLES:
        P = poses[role]
        F = len(P["frames"])
        f = lambda k: torch.tensor(np.array([fr[k] for fr in P["frames"]], np.float32))  # noqa: E731
        res = {}
        for nb in (16, 10):
            m = smplx.create(str(WORKSPACE), model_type="smplx", gender="neutral", ext="npz", use_pca=False,
                             flat_hand_mean=True, num_betas=nb, batch_size=F)
            b = torch.tensor(np.asarray(P["betas"][:nb], np.float32))[None].expand(F, -1)
            with torch.no_grad():
                o = m(betas=b, global_orient=f("root_pose"), body_pose=f("body_pose"), jaw_pose=f("jaw_pose"),
                      leye_pose=f("leye_pose"), reye_pose=f("reye_pose"), left_hand_pose=f("lhand_pose"),
                      right_hand_pose=f("rhand_pose"), transl=f("transl"))
            J = o.joints.numpy()[:, :22]
            ref = np.array([fr["joints22"] for fr in P["frames"]])
            e = np.linalg.norm(J - ref, axis=-1)
            pel = J[:, 0] - ref[:, 0]
            el = np.linalg.norm((J - J[:, :1]) - (ref - ref[:, :1]), axis=-1)
            res[f"b{nb}"] = {"mpjpe_mm_median": round(float(np.median(e) * 1e3), 2),
                             "mpjpe_mm_max": round(float(e.max() * 1e3), 2),
                             "pelvis_offset_mm_median": round(float(np.median(np.linalg.norm(pel, axis=1)) * 1e3), 2),
                             "root_rel_mm_median": round(float(np.median(el) * 1e3), 2)}
        out[role] = res
    return out


# ------------------------------------------------------------------------------------------------ baseline (local)


def prep_baseline(views: list[dict], device: str = "cpu") -> dict:
    """The current footage-baked texture avatar (8 phones, includes 01/03/05) rendered into the held-out views."""
    import torch

    from ..avatar_review import _texel_colors
    from ..avatar_texture import _Geo, load_bodies, render
    from ..avatar_uv import build_atlas
    from ..mamma_dyn import SMPLX_MODELS
    rep = json.loads((WORK / "avatar" / "report.json").read_text())
    size = int(rep["config"]["size"])
    bodies, info = load_bodies(WORK)
    atlas = build_atlas(SMPLX_MODELS, size)
    geo = _Geo(atlas, device)
    colors = _texel_colors(torch, WORK / "avatar", device)
    bd = INPUT / "baseline"
    if bd.exists():
        shutil.rmtree(bd)
    bd.mkdir(parents=True)
    t0 = time.time()
    for v in views:
        posed = {r: b.at(v["t"])[0] for r, b in bodies.items()}
        ren, rid = render(torch, geo, posed, colors, np.array(v["K"]), np.array(v["R"]), np.array(v["t_w"]), v["k1"],
                          v["height"], v["width"], device)
        u8 = (np.clip(ren, 0, 1) * 255 + 0.5).astype(np.uint8)
        cv2.imwrite(str(bd / f"{v['key']}.png"), u8[:, :, ::-1])
        idm = np.zeros(rid.shape, np.uint8)
        for n, r in enumerate(bodies):
            idm[rid == n] = ROLES.index(r) + 1
        cv2.imwrite(str(bd / f"{v['key']}_id.png"), idm)
    return {"views": len(views), "seconds": round(time.time() - t0, 1), "atlas": size, "bodies": info,
            "note": "avatar/<role>_texels.npz baked from cams " + ",".join(rep.get("cams", []))}


def prep(baseline: bool = True, reuse_refs: bool = False) -> dict:
    INPUT.mkdir(parents=True, exist_ok=True)
    fits = {r: _load_fit(r) for r in ROLES}
    if reuse_refs and (INPUT / "refs.json").exists():
        refs = json.loads((INPUT / "refs.json").read_text())
    else:
        refs = prep_refs(fits)
        (INPUT / "refs.json").write_text(json.dumps(refs, indent=1))
    views, poses = prep_views(fits)
    (INPUT / "views.json").write_text(json.dumps({"take": TAKE, "input_cam": CAM, "window": list(WINDOW),
                                                  "holdout": list(HOLDOUT), "step_s": STEP_S, "views": views}, indent=1))
    (INPUT / "poses.json").write_text(json.dumps(poses))
    chk = check_transfer(poses)
    log(f"transfer check (standard SMPL-X FK of the interpolated parameters vs the interpolated reference joints): "
        f"{json.dumps(chk)}")
    info = {"refs": {r: {s: refs[r][s]["n"] for s in SETS} for r in ROLES}, "views": len(views),
            "transfer_check": chk}
    if baseline:
        info["baseline"] = prep_baseline(views)
        log(f"baseline renders: {info['baseline']['seconds']} s")
    info["bytes"] = int(sum(p.stat().st_size for p in INPUT.rglob("*") if p.is_file()))
    (INPUT / "prep.json").write_text(json.dumps(info, indent=1, default=str))
    return info


# ------------------------------------------------------------------------------------------------ pod job


def job(max_minutes: float = 60) -> Path:
    rel = lambda p: _ws(p)[len("dancecap/"):]  # noqa: E731 - cwd of the commands is dancecap/
    d = {"schema": "dancecap.pod_job/1", "name": "cand_lhmpp", "take": TAKE, "max_minutes": max_minutes, "needs": [],
         "inputs": [_ws(INPUT) + "/"],
         "exclude": ["dancecap/work/runpod/pywheels", "dancecap/work/runpod/wheelhouse", "mamma-turing"],
         "env": {"LHMPP_COMMIT": LHMPP_COMMIT, "LHMPP_ROOT": "/workspace/lhmpp_cand/LHM-plusplus",
                 "LHMPP_VENV": "/workspace/lhmpp_cand/venv", "LHMPP_MODEL": MODEL_NAME,
                 "TORCH_HOME": "/workspace/lhmpp_cand/torch_home", "HF_HOME": "/workspace/lhmpp_cand/hf_home"},
         "commands": [
             {"name": "install", "timeout_min": 35, "cmd": "bash remote/cand_lhmpp/install.sh"},
             {"name": "infer", "timeout_min": 15,
              "cmd": f"/workspace/lhmpp_cand/venv/bin/python remote/cand_lhmpp/lhmpp_pod.py --input {rel(INPUT)} "
                     f"--out {rel(POD_OUT)}"},
             {"name": "score", "timeout_min": 8,
              "cmd": f"(python -m pip install -q --no-deps lpips==0.1.4 || echo 'lpips install failed: no LPIPS'); "
                     f"python -m dancecap.mono4d.cand_lhmpp score --dir {rel(OUT)}"}],
         "outputs": [_ws(POD_OUT) + "/"],
         "verify": [_ws(POD_OUT / "lhmpp_pod.json"), _ws(POD_OUT / "metrics.json")],
         "after_download": ["python -m dancecap.mono4d.cand_lhmpp import --session <sid>"],
         "notes": [f"LHM++ ({MODEL_NAME}, CC BY-NC 4.0 weights from ModelScope Damo_XR_Lab; prior files from HF "
                   "3DAIGC/LHMPP-Prior) builds one Gaussian avatar per dancer from phone-06 crops, animated with the "
                   "8-phone reference SMPL-X poses and rendered into held-out phones 01/03/05; weights stay on the pod; "
                   "the dancers' crops and the held-out frames go to our own pod only (owner consent 2026-10-19)"]}
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    p = JOB_DIR / "pod_job.json"
    p.write_text(json.dumps(d, indent=1), newline="\n")
    return p


# ------------------------------------------------------------------------------------------------ score (pod)


def _crop_box(m: np.ndarray, pad: float = 0.1) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(m)
    H, W = m.shape
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    px, py = int(round(pad * (x1 - x0))) + 2, int(round(pad * (y1 - y0))) + 2
    return max(0, x0 - px), max(0, y0 - py), min(W, x1 + px), min(H, y1 + py)


def _read_rgb(p: Path) -> np.ndarray | None:
    im = cv2.imread(str(p), cv2.IMREAD_COLOR)
    return None if im is None else im[:, :, ::-1].astype(np.float32) / 255.0


def score_dir(root: Path, lpips_net: bool = True) -> dict:
    """Masked metrics of every rendered variant and of the baseline, per view and dancer, on a crop around the dancer's
    real mask (the same _crop_scores as the dancer splats). Silhouette IoU of the rendered role vs the real mask."""
    from ..dancer_splats.evaluate import _crop_scores
    inp, po = root / "input", root / "pod_out"
    views = json.loads((inp / "views.json").read_text())["views"]
    pod = json.loads((po / "lhmpp_pod.json").read_text()) if (po / "lhmpp_pod.json").exists() else {}
    variants = sorted(p.name for p in (po / "renders").iterdir() if p.is_dir()) if (po / "renders").exists() else []
    device = "cpu"
    net = None
    if lpips_net:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            from ..dancer_splats.evaluate import _lpips
            net = _lpips(device)
        except Exception as e:  # noqa: BLE001
            log(f"LPIPS unavailable: {e}")
    rows = []
    for v in views:
        gt = _read_rgb(inp / "real" / f"{v['key']}.jpg")
        masks = {r: cv2.imread(str(inp / "masks" / f"{v['key']}_{PID[r]}.png"), cv2.IMREAD_GRAYSCALE) for r in ROLES}
        preds = {}
        for var in variants:
            rgb = _read_rgb(po / "renders" / var / f"{v['key']}.png")
            rid = cv2.imread(str(po / "renders" / var / f"{v['key']}_role.png"), cv2.IMREAD_COLOR)
            if rgb is not None:
                preds[var] = (rgb, None if rid is None else rid[:, :, ::-1].astype(np.float32) / 255.0)
        b = _read_rgb(inp / "baseline" / f"{v['key']}.png")
        if b is not None:
            bid = cv2.imread(str(inp / "baseline" / f"{v['key']}_id.png"), cv2.IMREAD_GRAYSCALE)
            w = np.stack([(bid == 1), (bid == 2), (bid > 0)], -1).astype(np.float32)
            preds["baseline_avatar"] = (b, w)
        for role in ROLES:
            m = masks[role]
            if m is None or (m > 127).sum() < 50:
                continue
            m = m > 127
            x0, y0, x1, y1 = _crop_box(m)
            sel = m[y0:y1, x0:x1]
            for var, (rgb, rw) in preds.items():
                s = _crop_scores(rgb[y0:y1, x0:x1], gt[y0:y1, x0:x1], sel, net, device)
                row = {"key": v["key"], "cam": v["cam"], "t": round(v["t"], 3), "role": role, "variant": var,
                       "crop": [int(x0), int(y0), int(x1), int(y1)], **{k: (None if s.get(k) is None else float(s[k]))
                                                                          for k in s}}
                if rw is not None:
                    own = rw[:, :, ROLES.index(role)] > 0.5
                    u = (own | m).sum()
                    row["iou"] = float((own & m).sum() / u) if u else None
                    row["coverage"] = float((own & m).sum() / m.sum())  # real-mask pixels the render covers
                    # appearance where the geometry agrees: the real mask AND the rendered dancer
                    sb = (own & m)[y0:y1, x0:x1]
                    s2 = _crop_scores(rgb[y0:y1, x0:x1], gt[y0:y1, x0:x1], sb, net, device)
                    row.update({f"{k}_both": float(s2[k]) for k in ("psnr", "ssim", "lpips", "psnr_cc")
                                if s2.get(k) is not None})
                rows.append(row)
    summ = summarise(rows)
    out = {"metrics": "dancecap.dancer_splats.evaluate._crop_scores on a crop = the dancer's real-mask bbox + 10 %; "
                      "psnr/ssim/lpips inside the mask (pixels the render misses count as black), *_cc after a "
                      "least-squares colour fit; *_both = inside the real mask AND the rendered dancer (appearance where "
                      "the geometry agrees); iou = rendered role weight > 0.5 vs the real mask; coverage = fraction of the "
                      "real mask the rendered dancer covers", "lpips": net is not None, "device": device,
           "variants": variants + ["baseline_avatar"], "rows": rows, "summary": summ, "pod": pod.get("summary")}
    (po / "metrics.json").write_text(json.dumps(out, indent=1, default=float))
    log(json.dumps(summ, indent=1, default=float)[:4000])
    return out


def summarise(rows: list[dict]) -> dict:
    keys = ("psnr", "ssim", "lpips", "psnr_cc", "ssim_cc", "lpips_cc", "iou", "coverage", "psnr_both", "ssim_both",
            "lpips_both", "psnr_cc_both")
    out: dict = {}
    for var in sorted({r["variant"] for r in rows}):
        out[var] = {}
        for grp, sel in [("all", lambda r: True)] + [(role, lambda r, ro=role: r["role"] == ro) for role in ROLES] + \
                [(f"cam{c}", lambda r, c=c: r["cam"] == c) for c in HOLDOUT]:
            rr = [r for r in rows if r["variant"] == var and sel(r)]
            out[var][grp] = {"n": len(rr), **{k: (round(float(np.median([r[k] for r in rr if r.get(k) is not None])), 4)
                                                  if any(r.get(k) is not None for r in rr) else None) for k in keys}}
    return out


# ------------------------------------------------------------------------------------------------ import (local)


def _session_meta(sid: str | None) -> dict:
    root = PACKAGE_ROOT / "work" / "runpod" / "sessions"
    if sid is None:
        cands = sorted((p for p in root.glob("*") if (p / "result.json").exists()
                        and '"cand_lhmpp"' in (p / "result.json").read_text(errors="ignore")),
                       key=lambda p: p.stat().st_mtime)
        if not cands:
            return {}
        sid = cands[-1].name
    p = root / sid / "result.json"
    if not p.exists():
        return {"session": sid}
    r = json.loads(p.read_text())
    pod = r.get("pod") or {}
    job_ = (r.get("jobs") or {}).get("cand_lhmpp") or {}
    led = _ledger_rows([pod.get("id")]) if pod.get("id") else []
    return {"session": sid, "pod_id": pod.get("id"), "gpu": pod.get("gpu"),
            "cloud": (led[0].get("cloud") if led else None) or pod.get("cloud"),
            "cost_per_hr": pod.get("cost_per_hr"), "cost_usd": pod.get("ledger_cost"),
            "pod_minutes": round((pod.get("lifetime_s") or 0) / 60, 1), "job_status": job_.get("status"),
            "job_seconds": job_.get("seconds")}


def _ledger_rows(pod_ids: list[str]) -> list[dict]:
    try:
        led = json.loads((PACKAGE_ROOT / "work" / "runpod" / "ledger.json").read_text())
    except Exception:  # noqa: BLE001
        return []
    rows = led if isinstance(led, list) else led.get("rows", led.get("pods", []))
    keep = ("pod_id", "gpu", "cloud", "cost_per_hr", "created", "deleted_reason", "cost")
    return [{k: x.get(k) for k in keep} for x in rows if x.get("pod_id") in set(pod_ids)]


def _tile(img: np.ndarray, box, h: int, label: str | None = None, sub: str | None = None) -> np.ndarray:
    x0, y0, x1, y1 = box
    c = img[y0:y1, x0:x1]
    s = h / max(1, c.shape[0])
    t = cv2.resize(c, (max(1, int(round(c.shape[1] * s))), h), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    t = np.ascontiguousarray(t)
    for k, (txt, y) in enumerate(((label, 18), (sub, h - 8))):
        if txt:
            cv2.putText(t, txt, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(t, txt, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return t


def contact_sheet(root: Path, metrics: dict, variants: list[str], per_cam: int = 3, h: int = 300) -> Path:
    inp, po = root / "input", root / "pod_out"
    views = json.loads((inp / "views.json").read_text())["views"]
    by = {}
    for r in metrics["rows"]:
        by[(r["key"], r["role"], r["variant"])] = r
    rows_img = []
    for cam in HOLDOUT:
        vs = [v for v in views if v["cam"] == cam]
        for v in [vs[int(round(k))] for k in np.linspace(0, len(vs) - 1, per_cam)]:
            gt = cv2.imread(str(inp / "real" / f"{v['key']}.jpg"))
            mm = [cv2.imread(str(inp / "masks" / f"{v['key']}_{PID[r]}.png"), cv2.IMREAD_GRAYSCALE) for r in ROLES]
            u = np.zeros(gt.shape[:2], bool)
            for m in mm:
                if m is not None:
                    u |= m > 127
            if not u.any():
                continue
            box = _crop_box(u, 0.12)
            tiles = [_tile(gt, box, h, f"real cam{cam} t={v['t']:.2f}")]
            for var in variants + ["baseline_avatar"]:
                p = (inp / "baseline" / f"{v['key']}.png") if var == "baseline_avatar" else \
                    (po / "renders" / var / f"{v['key']}.png")
                im = cv2.imread(str(p))
                if im is None:
                    continue
                sc = [by.get((v["key"], r, var)) for r in ROLES]
                sub = " ".join(f"{r[0].upper()}:{s['psnr']:.1f}dB" for r, s in zip(ROLES, sc) if s and s.get("psnr"))
                tiles.append(_tile(im, box, h, var if var != "baseline_avatar" else "baked avatar (8 phones)", sub))
            rows_img.append(np.concatenate([np.concatenate([t, np.full((h, 4, 3), 255, np.uint8)], 1) for t in tiles], 1))
    W = max(r.shape[1] for r in rows_img)
    rows_img = [np.pad(r, ((0, 4), (0, W - r.shape[1]), (0, 0)), constant_values=255) for r in rows_img]
    sheet = np.concatenate(rows_img, 0)
    p = root / "contact_sheet.jpg"
    cv2.imwrite(str(p), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return p


def avatar_sheet(root: Path, variants: list[str], h: int = 360) -> Path | None:
    """Inputs and canonical avatars: per dancer the reference crops of each set, then the A-pose turntables (front,
    left, back, right) of each variant."""
    po = root / "pod_out"
    rows = []
    for role in ROLES:
        for name in SETS:
            p = root / f"refs_{role}_{name}.jpg"
            if p.exists():
                im = cv2.imread(str(p))
                rows.append(_tile(im, (0, 0, im.shape[1], im.shape[0]), h // 2, f"{role} input crops {name} (phone 06)"))
        for var in variants:
            p = po / "turntable" / f"{role}_{var}.png"
            if p.exists():
                im = cv2.imread(str(p))
                rows.append(_tile(im, (0, 0, im.shape[1], im.shape[0]), h, f"{role} LHM++ {var}: A-pose front/left/back/right"))
    if not rows:
        return None
    W = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 4), (0, W - r.shape[1]), (0, 0)), constant_values=255) for r in rows]
    out = root / "avatars.jpg"
    cv2.imwrite(str(out), np.concatenate(rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return out


def import_run(sid: str | None = None, pods: list[str] | None = None) -> dict:
    po = POD_OUT
    if not (po / "metrics.json").exists():
        raise SystemExit(f"{po}/metrics.json missing: run the pod job (or score) first")
    metrics = json.loads((po / "metrics.json").read_text())
    pod = json.loads((po / "lhmpp_pod.json").read_text())
    # local cross-check of PSNR / SSIM on 6 random rows (same code, no LPIPS)
    from ..dancer_splats.evaluate import _crop_scores
    rng = np.random.default_rng(0)
    rows = [r for r in metrics["rows"] if r["variant"] != "baseline_avatar"]
    chk = []
    for r in [rows[k] for k in rng.choice(len(rows), size=min(6, len(rows)), replace=False)]:
        gt = _read_rgb(INPUT / "real" / f"{r['key']}.jpg")
        rgb = _read_rgb(po / "renders" / r["variant"] / f"{r['key']}.png")
        m = cv2.imread(str(INPUT / "masks" / f"{r['key']}_{PID[r['role']]}.png"), cv2.IMREAD_GRAYSCALE) > 127
        x0, y0, x1, y1 = r["crop"]
        s = _crop_scores(rgb[y0:y1, x0:x1], gt[y0:y1, x0:x1], m[y0:y1, x0:x1], None, "cpu")
        chk.append({"key": r["key"], "role": r["role"], "variant": r["variant"], "psnr_pod": r["psnr"],
                    "psnr_local": s["psnr"], "ssim_pod": r["ssim"], "ssim_local": s["ssim"]})
    variants = [v for v in metrics["variants"] if v != "baseline_avatar"]
    main = [v for v in ("w8_ref", "t16_ref", "t16_pred") if v in variants]
    sheet = contact_sheet(OUT, metrics, main)
    av = avatar_sheet(OUT, [v for v in ("w8_ref", "t16_ref", "t16_pred") if v in variants])
    report = {"take": TAKE, "input_cam": CAM, "window": list(WINDOW), "holdout": list(HOLDOUT),
              "model": MODEL_NAME, "commit": LHMPP_COMMIT, "licence": LICENCE, "session": _session_meta(sid),
              "pod": pod.get("summary"), "transfer": pod.get("transfer"), "summary": metrics["summary"],
              "lpips": metrics.get("lpips"), "crosscheck": chk, "contact_sheet": str(sheet), "avatar_sheet": str(av),
              "prep": json.loads((INPUT / "prep.json").read_text()) if (INPUT / "prep.json").exists() else None}
    if pods:  # every pod this track created (failed attempts included), from the ledger
        rows = _ledger_rows(pods)
        report["pods"] = rows
        report["cost_usd_all_attempts"] = round(sum(float(x.get("cost") or 0) for x in rows), 4)
    (OUT / "report.json").write_text(json.dumps(report, indent=1, default=float))
    log(f"report -> {OUT / 'report.json'}; sheet -> {sheet}")
    return report


# ================================================================================================ compare mode


CMP = WORK / "mono4d" / "compare"
CMP_OUT = CMP / "lhmpp"
CMP_INPUT = CMP_OUT / "input"
CMP_POD_OUT = CMP_OUT / "pod_out"
CMP_RENDERS = CMP / "renders"
ALL_CAMS = ("01", "02", "03", "04", "05", "06", "07", "09")
CMP_WINDOW = WINDOW                       # 58-62 s: the earlier w8 window (the spec's clip is 58.0-61.5 s inside it)
CMP_VARIANTS = {
    "lhmpp_1phone": {"cams": ("06",), "n": 8, "min_dt": 0.2, "per_cam": None, "seed": "w8", "avoid_view_s": None,
                     "what": "LHM++ from 8 crops of ONE phone (06), 58-62 s: the earlier best w8_ref with the crops that "
                             "risk white patches (partner occlusion / mask holes) replaced by the next-best phone-06 "
                             "frames"},
    "lhmpp_8phone": {"cams": ALL_CAMS, "n": 16, "min_dt": 0.2, "per_cam": 2, "seed": None, "avoid_view_s": 0.05,
                     "what": "LHM++ from 16 crops of ALL 8 phones (2 per phone), 58-62 s, same selection rules: the "
                             "upper bound of this method with the full rig"},
}
OCC_LIMITS = {"white_frac": 0.015, "bad_hole_frac": 0.002}  # occ_model (hand holding counts) is reported only
OCC_CELL_M = 0.01                         # z-buffer cell ~ 1 cm at the nearer dancer
OCC_ERODE_CELLS = 2                       # visible-body erosion before the mask test (fit error ~ 1-2 cm)
ZOOM_PX = 512


def _cmp_spec() -> dict:
    p = CMP / "views.json"
    if not p.exists():
        raise SystemExit(f"{p} missing: run python -m dancecap.mono4d.appearance_compare spec first")
    return json.loads(p.read_text())


def _holes(m: np.ndarray, min_px: int) -> np.ndarray:
    """Enclosed background components of a mask (they do not touch the image border), each >= min_px."""
    n, lab, st, _ = cv2.connectedComponentsWithStats((~m).astype(np.uint8), connectivity=4)
    H, W = m.shape
    keep = np.zeros(n, bool)
    for k in range(1, n):
        x, y, w, h, a = st[k]
        keep[k] = a >= min_px and x > 0 and y > 0 and x + w < W and y + h < H
    return keep[lab]


def _body_zbuffer(torch, V: np.ndarray, faces, K, R, t, s: float, Ws: int, Hs: int, hand_faces=None):
    """Depth of one posed body (avatar_raster.rasterize, exact triangles) at scale s of the frame; inf = empty.
    With hand_faces (bool per face) also the cells whose front face belongs to a hand."""
    from ..avatar_raster import EMPTY_Z, rasterize
    Xc = V @ np.asarray(R, float).T + np.asarray(t, float)
    z = Xc[:, 2]
    K = np.asarray(K, float)
    zz = np.maximum(z, 1e-6)
    u = ((K[0, 0] * Xc[:, 0] / zz + K[0, 2]) + 0.5) * s - 0.5
    v = ((K[1, 1] * Xc[:, 1] / zz + K[1, 2]) + 0.5) * s - 0.5
    px = torch.tensor(np.stack([u, v], 1), dtype=torch.float32)
    zb, fid = rasterize(torch, px, torch.tensor(z, dtype=torch.float32), faces, Hs, Ws)
    zb = zb.numpy().astype(np.float32)
    zb[zb >= EMPTY_Z * 0.5] = np.inf
    if hand_faces is None:
        return zb
    fid = fid.numpy()
    return zb, (fid >= 0) & np.asarray(hand_faces)[np.maximum(fid, 0)]


def hand_face_mask(faces: np.ndarray, lbs_weights: np.ndarray) -> np.ndarray:
    """Faces whose dominant skinning joint is a wrist (20, 21) or a finger (25-54) of SMPL-X."""
    lab = np.asarray(lbs_weights)[np.asarray(faces)].sum(1).argmax(1)
    return (lab == 20) | (lab == 21) | (lab >= 25)


def occlusion_stats(torch, cam: str, i: int, t: float, tr, bodies: dict, faces, masks: dict, Wimg: int,
                    Himg: int, hand_faces=None) -> dict:
    """White-patch risk of each dancer's crop in frame i of a phone (see the module doc, OCC_LIMITS)."""
    posed = {r: bodies[r].at(t) for r in ROLES}
    if any(p is None for p in posed.values()):
        return {}
    K, R, tv = tr.K[i], tr.R[i], tr.tvec[i]
    zc = min(float(np.median(posed[r][0] @ np.asarray(R).T[:, 2] + tv[2])) for r in ROLES)
    s = float(np.clip(zc / (float(K[0, 0]) * OCC_CELL_M), 0.1, 1.0))
    Ws, Hs = int(math.ceil(Wimg * s)), int(math.ceil(Himg * s))
    zh = {r: _body_zbuffer(torch, posed[r][0], faces, K, R, tv, s, Ws, Hs, hand_faces) for r in ROLES}
    zb = {r: zh[r][0] for r in ROLES}
    hand = {r: zh[r][1] for r in ROLES}
    ms = {r: (cv2.resize(masks[r].astype(np.float32), (Ws, Hs), interpolation=cv2.INTER_AREA) > 0.5)
          if masks.get(r) is not None else np.zeros((Hs, Ws), bool) for r in ROLES}
    ker = np.ones((3, 3), np.uint8)
    out = {}
    for r in ROLES:
        o = ROLES[1 - ROLES.index(r)]
        zs, zo = zb[r], zb[o]
        ps, po = np.isfinite(zs), np.isfinite(zo)
        n_body = max(1, int(ps.sum()))
        occl = ps & po & (zo < zs - 0.03)
        vis = ps & ~(po & (zo < zs))
        vis_er = cv2.erode(vis.astype(np.uint8), ker, iterations=OCC_ERODE_CELLS).astype(bool)
        nv = max(1, int(vis_er.sum()))
        # the dancer's whole projected body without the hands (hand holding is not a white patch that matters),
        # eroded for fit error: where the own mask does not cover it the crop turns white - partner in front
        # (arm / hand / hair across a shoulder), mask hole or mask miss
        body = cv2.erode((ps & ~hand[r]).astype(np.uint8), ker, iterations=OCC_ERODE_CELLS).astype(bool)
        nb_ = max(1, int(body.sum()))
        white = body & ~ms[r]
        m = masks.get(r)
        rec = {"scale": round(s, 3), "occ_model": round(float(occl.sum() / n_body), 4),
               "white_frac": round(float(white.sum() / nb_), 4),
               "white_partner_frac": round(float((white & ms[o]).sum() / nb_), 4),
               "white_vis_frac": round(float((vis_er & ~ms[r]).sum() / nv), 4), "body_cells": nb_}
        if m is not None and m.any():
            h = _holes(m, max(20, int(0.0005 * m.sum())))
            hs = cv2.resize(h.astype(np.float32), (Ws, Hs), interpolation=cv2.INTER_AREA) > 0.25
            bad = hs & (vis | ms[o])
            rec["hole_frac"] = round(float(h.sum() / m.sum()), 5)
            rec["bad_hole_frac"] = round(float(bad.sum() / max(1, ms[r].sum())), 5)
        else:
            rec["hole_frac"] = rec["bad_hole_frac"] = None
        rec["drop"] = [k for k, lim in OCC_LIMITS.items() if rec.get(k) is not None and rec[k] > lim]
        out[r] = rec
    return out


def _cmp_candidates(cam: str, fits: dict, bodies: dict, faces, tracks: dict, t_lo: float, t_hi: float,
                    hand_faces=None) -> list[dict]:
    """_candidates (same visibility / size / sharpness / yaw definitions) for any phone + occlusion_stats on every
    frame that passes the base gate for at least one dancer. Phone 02 (59.94 fps) uses every 2nd frame."""
    import torch
    tr = tracks[cam]
    fj = json.loads((WORK / "frames" / cam / "frames.json").read_text())
    frames = fj["frames"]
    Wimg, Himg = int(fj["width"]), int(fj["height"])
    stride = 2 if float(fj.get("fps_median") or 30.0) > 45 else 1
    rad = max(3, int(round(0.015 * Himg)))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rad + 1, 2 * rad + 1))
    rows = []
    for i in range(0, len(frames), stride):
        t = float(tr.t[i])
        if not (t_lo <= t <= t_hi) or not bool(tr.ok[i]):
            continue
        masks = {r: _mask(cam, i, r) for r in ROLES}
        img, occ, frame_rows = None, None, []
        for role in ROLES:
            d = fits[role]
            m = masks[role]
            if t < d["t"][0] or t > d["t"][-1] or m is None or m.sum() < 2000:
                continue
            p = interp_params(d, t)
            J = p["joints"][list(range(22)) + [15]]
            uv, z = project(J, tr.K[i], tr.R[i], tr.tvec[i])
            md_ = cv2.dilate(m.astype(np.uint8), ker) > 0
            u, v = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
            inside = (u >= 0) & (u < Wimg) & (v >= 0) & (v < Himg) & (z > 0)
            hit = np.zeros(len(u), bool)
            hit[inside] = md_[v[inside], u[inside]]
            ys, xs = np.nonzero(m)
            border = xs.min() <= 3 or ys.min() <= 3 or xs.max() >= Wimg - 4 or ys.max() >= Himg - 4
            if img is None:
                img = cv2.imread(str(WORK / "frames" / cam / frames[i]["file"]), cv2.IMREAD_GRAYSCALE)
            crop = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            sharp = float(cv2.Laplacian(cv2.resize(crop, None, fx=0.5, fy=0.5), cv2.CV_32F).var())
            frame_rows.append({"cam": cam, "role": role, "i": i, "t": round(t, 4), "vis": float(hit.mean()),
                               "inimg": float(inside.mean()), "area": int(m.sum()),
                               "box": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                               "border": bool(border), "sharp": sharp,
                               "yaw_deg": round(body_yaw_deg(np.asarray(p["global_orient"]), p["joints"][0], tr.R[i],
                                                             tr.tvec[i]), 1), "W": Wimg, "H": Himg})
        if any(r["vis"] >= 0.9 and r["inimg"] == 1.0 and not r["border"] for r in frame_rows):
            occ = occlusion_stats(torch, cam, i, t, tr, bodies, faces, masks, Wimg, Himg, hand_faces)
        for r in frame_rows:
            r["occ"] = (occ or {}).get(r["role"])
            rows.append(r)
    return rows


def _base_ok(r: dict) -> bool:
    return r["vis"] >= 0.9 and r["inimg"] == 1.0 and not r["border"]


def _occ_drop(o: dict | None) -> list[str]:
    """The OCC_LIMITS a crop exceeds (re-evaluated from the stored values, so a limit change needs no new survey)."""
    return [k for k, lim in OCC_LIMITS.items() if o and o.get(k) is not None and o[k] > lim]


def _occ_ok(r: dict) -> bool:
    return r.get("occ") is not None and not _occ_drop(r["occ"])


def _occ_badness(r: dict) -> float:
    o = r.get("occ") or {}
    return max([float(o.get(k) or 0) / lim for k, lim in OCC_LIMITS.items()] + [0.0]) if o else 99.0


def _set_q(rows: list[dict]) -> None:
    """_pick's quality, with area and sharpness normalised PER PHONE (phones differ 5x in pixel count)."""
    for cam in {r["cam"] for r in rows}:
        rr = [r for r in rows if r["cam"] == cam]
        a_med = float(np.median([r["area"] for r in rr]))
        s_med = float(np.median([r["sharp"] for r in rr]))
        for r in rr:
            r["q"] = r["vis"] ** 2 * min(1.0, r["area"] / a_med) * min(1.0, r["sharp"] / s_med) ** 0.5


def _greedy(ok: list[dict], n: int, min_dt: float, seed: list[dict], per_cam: int | None) -> list[dict]:
    """_pick's rule generalised: best near-frontal first (unless seeded), then maximise q x (yaw distance to the picked
    set / 180 + 0.15); per phone at least min_dt apart and at most per_cam picks."""
    picked = list(seed)

    def allowed(r):
        if any(r["cam"] == p["cam"] and abs(r["t"] - p["t"]) < min_dt for p in picked):
            return False
        return not (per_cam and sum(p["cam"] == r["cam"] for p in picked) >= per_cam)

    if not picked:
        front = [r for r in ok if abs(r["yaw_deg"]) <= 40] or ok
        if not front:
            return []
        picked.append(max(front, key=lambda r: r["q"]))
    while len(picked) < n:
        best, best_s = None, -1.0
        for r in ok:
            if not allowed(r):
                continue
            dy = min(abs((r["yaw_deg"] - p["yaw_deg"] + 180) % 360 - 180) for p in picked)
            s = r["q"] * (dy / 180.0 + 0.15)
            if s > best_s:
                best, best_s = r, s
        if best is None:
            break
        picked.append(best)
    return picked


def _select_variant(name: str, rows_all: list[dict], role: str, spec: dict) -> dict:
    """One variant's crops of one dancer: base gate (_pick's: vis >= 0.9, inside the image, not cut by the border) +
    no OCC_LIMITS exceeded, then _greedy. When the clean candidates cannot fill the set at min_dt, the greedy continues
    with min_dt / 2 and then 2 frames; only then the least-bad base candidates are admitted, 4 at a time (flagged
    relaxed)."""
    cfg = CMP_VARIANTS[name]
    rows = [r for r in rows_all if r["role"] == role and r["cam"] in cfg["cams"]]
    avoid = []
    if cfg["avoid_view_s"]:
        views = [(v["cam"], float(v["t"])) for v in spec["real_views"]]
        avoid = [r for r in rows if any(r["cam"] == c and abs(r["t"] - tv) < cfg["avoid_view_s"] for c, tv in views)]
        rows = [r for r in rows if r not in avoid]
    base = [r for r in rows if _base_ok(r)]
    ok = [r for r in base if _occ_ok(r)]
    _set_q(base)
    seed, kept, dropped = [], [], []
    if cfg["seed"] == "w8":
        prev = json.loads((INPUT / "refs.json").read_text())[role]["w8"]["frames"]
        by_i = {r["i"]: r for r in rows if r["cam"] == "06"}
        for k, f in enumerate(prev):
            r = by_i.get(int(f["i"]))
            if r is not None and r in ok:
                seed.append(r)
                kept.append(r["i"])
            else:
                why = ["not a candidate"] if r is None else (["base gate"] if not _base_ok(r) else
                                                             _occ_drop(r.get("occ")) or ["no occlusion stats"])
                dropped.append({"slot": k, "i": int(f["i"]), "t": f["t"], "yaw_deg": f["yaw_deg"], "why": why,
                                "occ": None if r is None else r.get("occ")})
        if seed and dropped and dropped[0]["slot"] == 0:  # the frontal 00 went: choose a new frontal start first
            cand = [r for r in ok if abs(r["yaw_deg"]) <= 40 and r not in seed and
                    all(abs(r["t"] - p["t"]) >= cfg["min_dt"] for p in seed)]
            if cand:
                seed.insert(0, max(cand, key=lambda r: r["q"]))
    picks = _greedy(ok, cfg["n"], cfg["min_dt"], seed, cfg["per_cam"])
    min_dt_used = cfg["min_dt"]
    for dt in (cfg["min_dt"] / 2, 2.0 / 30):  # short: clean frames closer in time before any flagged crop
        if len(picks) >= cfg["n"]:
            break
        picks = _greedy(ok, cfg["n"], dt, picks, cfg["per_cam"])
        min_dt_used = dt
    pool = sorted([r for r in base if r not in ok], key=_occ_badness)
    relaxed = []
    while len(picks) < cfg["n"] and pool:
        short = None if not cfg["per_cam"] else             {c for c in cfg["cams"] if sum(p["cam"] == c for p in picks) < cfg["per_cam"]}
        cand = [r for r in pool if short is None or r["cam"] in short]
        if not cand:
            break
        for r in cand[:4]:
            pool.remove(r)
            relaxed.append(r)
        picks = _greedy(ok + relaxed, cfg["n"], min_dt_used, picks, cfg["per_cam"])
    return {"picks": picks, "dropped_seed": dropped, "seed_kept": kept, "min_dt_used": round(min_dt_used, 4),
            "relaxed": [(r["cam"], r["i"]) for r in relaxed if r in picks],
            "avoided_frames": len(avoid), "candidates": len(rows), "base_ok": len(base), "clean": len(ok)}


def _pick_record(r: dict, slot: int, sel: dict) -> dict:
    keep = ("cam", "i", "t", "yaw_deg", "vis", "inimg", "area", "sharp", "box", "W", "H", "occ")
    rec = {"slot": slot, **{k: r.get(k) for k in keep}}
    rec["relaxed"] = (r["cam"], r["i"]) in {tuple(x) for x in sel["relaxed"]}
    rec["replacement"] = bool(sel["dropped_seed"]) and r["cam"] == "06" and r["i"] not in sel["seed_kept"]
    rec["file"] = f"work/{TAKE}/frames/{r['cam']}/{int(r['i']):06d}.jpg"
    return rec


def compare_prep_refs(spec: dict, fits: dict, reuse: bool = False) -> dict:
    from .. import mamma_dyn as md
    from ..avatar_texture import load_bodies
    cpath = CMP_OUT / "candidates.json"
    if reuse and cpath.exists():
        rows = json.loads(cpath.read_text())["rows"]
    else:
        import smplx
        from ..avatar_texture import model_dir_for
        import torch
        t0 = time.time()
        bodies, _ = load_bodies(WORK, head_fix=False)
        d = fits["lead"]
        sm = smplx.create(str(model_dir_for(d)), model_type="smplx", gender="neutral", ext="npz", use_pca=False,
                          flat_hand_mean=True, num_betas=int(d.get("num_betas", 16)))
        faces_np = np.asarray(sm.faces, np.int64)
        faces = torch.tensor(faces_np)
        hand_faces = hand_face_mask(faces_np, sm.lbs_weights.detach().numpy())
        tracks = md.load_tracks(WORK, list(ALL_CAMS))
        rows = []
        for cam in ALL_CAMS:
            t1 = time.time()
            rr = _cmp_candidates(cam, fits, bodies, faces, tracks, *CMP_WINDOW, hand_faces=hand_faces)
            rows += rr
            nb = sum(_base_ok(r) for r in rr)
            nc = sum(_base_ok(r) and _occ_ok(r) for r in rr)
            log(f"phone {cam}: {len(rr)} dancer-frames, base gate {nb}, clean {nc} ({time.time() - t1:.0f} s)")
        CMP_OUT.mkdir(parents=True, exist_ok=True)
        cpath.write_text(json.dumps({"window": list(CMP_WINDOW), "limits": OCC_LIMITS, "cell_m": OCC_CELL_M,
                                     "erode_cells": OCC_ERODE_CELLS, "seconds": round(time.time() - t0, 1),
                                     "rows": rows}, default=float))
    d = CMP_INPUT / "refs"
    if d.exists():
        shutil.rmtree(d)
    out = {}
    for name, cfg in CMP_VARIANTS.items():
        out[name] = {"what": cfg["what"], "rules": {k: v for k, v in cfg.items() if k != "what"}}
        for role in ROLES:
            sel = _select_variant(name, rows, role, spec)
            if sel["picks"]:  # NN = 00 is the most frontal crop (LHM++'s input convention of the first run)
                f0 = min(range(len(sel["picks"])), key=lambda k: abs(sel["picks"][k]["yaw_deg"]))
                sel["picks"].insert(0, sel["picks"].pop(f0))
            sd = d / role / name
            sd.mkdir(parents=True, exist_ok=True)
            tiles, recs = [], []
            for k, r in enumerate(sel["picks"]):
                fjs = json.loads((WORK / "frames" / r["cam"] / "frames.json").read_text())["frames"]
                img = cv2.imread(str(WORK / "frames" / r["cam"] / fjs[r["i"]]["file"]))[:, :, ::-1]
                c = lhm_crop(img, _mask(r["cam"], r["i"], role))
                u8 = (np.clip(c, 0, 1) * 255 + 0.5).astype(np.uint8)
                cv2.imwrite(str(sd / f"{k:02d}.png"), u8[:, :, ::-1])
                rec = _pick_record(r, k, sel)
                recs.append(rec)
                lab = f"{r['cam']}/{r['i']} {r['yaw_deg']:.0f}d" + (" R" if rec["relaxed"] else "") + \
                      (" new" if rec["replacement"] else "")
                tiles.append(_tile(np.ascontiguousarray(u8[:, :, ::-1]), (0, 0, u8.shape[1], u8.shape[0]), 280, lab))
            sheet = np.concatenate(tiles, 1)
            cv2.imwrite(str(CMP_OUT / f"refs_{role}_{name}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
            out[name][role] = {"n": len(recs), "frames": recs, **{k: sel[k] for k in
                                                                  ("dropped_seed", "seed_kept", "relaxed", "min_dt_used",
                                                                   "avoided_frames", "candidates", "base_ok", "clean")}}
            used = ", ".join(f"{x['cam']}/{x['i']}" for x in recs)
            log(f"{name} {role}: {len(recs)} crops ({used}); clean candidates {sel['clean']} of {sel['base_ok']}; "
                f"dropped seeds {[x['i'] for x in sel['dropped_seed']]}; relaxed {sel['relaxed']}")
    _dropped_sheet(out)
    return out


def _dropped_sheet(refs: dict) -> None:
    """The earlier w8 crops of each dancer with the white-patch verdict (kept / dropped + why)."""
    prev = json.loads((INPUT / "refs.json").read_text())
    for role in ROLES:
        dr = {x["slot"]: x for x in refs["lhmpp_1phone"][role]["dropped_seed"]}
        tiles = []
        for k, f in enumerate(prev[role]["w8"]["frames"]):
            im = cv2.imread(str(INPUT / "refs" / role / "w8" / f"{k:02d}.png"))
            if im is None:
                continue
            if k in dr:
                o = dr[k].get("occ") or {}
                lab = "DROP " + ",".join(dr[k]["why"] or [])
                sub = f"white {o.get('white_frac')} occ {o.get('occ_model')} hole {o.get('bad_hole_frac')}"
                im = cv2.copyMakeBorder(im[6:-6, 6:-6], 6, 6, 6, 6, cv2.BORDER_CONSTANT, value=(0, 0, 255))
            else:
                lab, sub = "kept", None
            tiles.append(_tile(im, (0, 0, im.shape[1], im.shape[0]), 280, f"06/{f['i']} {lab}", sub))
        if tiles:
            cv2.imwrite(str(CMP_OUT / f"dropped_{role}.jpg"), np.concatenate(tiles, 1), [cv2.IMWRITE_JPEG_QUALITY, 90])


def _zoom_K(K, box, px: int) -> tuple[list, int, int]:
    """Camera of a box of a real view rendered at px on its long side (OpenCV pixel centres at integers)."""
    K = np.asarray(K, float)
    x0, y0, x1, y1 = [int(v) for v in box]
    w, h = x1 - x0, y1 - y0
    sc = px / max(w, h)
    K2 = np.array([[K[0, 0] * sc, 0, (K[0, 2] - x0 + 0.5) * sc - 0.5],
                   [0, K[1, 1] * sc, (K[1, 2] - y0 + 0.5) * sc - 0.5], [0, 0, 1.0]])
    return K2.tolist(), int(round(w * sc)), int(round(h * sc))


def compare_prep_views(spec: dict, fits: dict) -> tuple[list[dict], dict, dict]:
    renders, keys = [], {}

    def pose(key: str, t: float) -> str:
        keys.setdefault(key, float(t))
        return key

    def cam(id_, group, out, key, K, R, tvec, W, H, **kw):
        renders.append({"id": id_, "group": group, "out": out, "pose": key, "K": np.asarray(K, float).tolist(),
                        "R": np.asarray(R, float).tolist(), "tvec": np.asarray(tvec, float).reshape(3).tolist(),
                        "W": int(W), "H": int(H), **kw})

    for d in ("real", "masks"):
        if (CMP_INPUT / d).exists():
            shutil.rmtree(CMP_INPUT / d)
        (CMP_INPUT / d).mkdir(parents=True)
    rv = {v["id"]: v for v in spec["real_views"]}
    for v in spec["real_views"]:
        k = pose(f"real_{v['id']}", v["t"])
        cam(v["id"], "real", f"{v['id']}.png", k, v["K"], v["R"], v["tvec"], v["W"], v["H"], role_map=True,
            cam=v["cam"], frame=int(v["frame"]), t=float(v["t"]), k1=float(v.get("k1", 0.0)))
        shutil.copy2(PACKAGE_ROOT / v["file"], CMP_INPUT / "real" / f"{v['id']}.jpg")
        for role in ROLES:
            p = PACKAGE_ROOT / v["masks"][role]
            if p.exists():
                shutil.copy2(p, CMP_INPUT / "masks" / f"{v['id']}_{PID[role]}.png")
    tt = spec["novel"]["turntable"]
    k = pose(f"tt_{tt['instant']}", tt["t"])
    for v in tt["views"]:
        cam(v["id"], "turntable", f"{v['id']}.png", k, v["K"], v["R"], v["tvec"], v["W"], v["H"], t=float(tt["t"]),
            azimuth_deg=v.get("azimuth_deg"))
    vr, clip = spec["novel"]["vr_seat"], spec["clip"]
    k0 = int(clip["k"][0])
    for j, t in enumerate(clip["t"]):
        k = pose(f"grid_{k0 + j}", t)
        cam(f"clip_vr_seat/{j:04d}", "clip_vr_seat", f"clip_vr_seat/{j:04d}.png", k, vr["K"], vr["R"], vr["tvec"],
            vr["W"], vr["H"], clip_index=j, grid_k=k0 + j, t=float(t))
    c3 = clip["cam03"]
    for j, fr in enumerate(c3["frames"]):
        k = pose(f"c03_{int(fr['frame'])}", fr["t"])
        cam(f"clip_cam03/{j:04d}", "clip_cam03", f"clip_cam03/{j:04d}.png", k, fr["K"], fr["R"], fr["tvec"], c3["W"],
            c3["H"], clip_index=j, frame=int(fr["frame"]), t=float(fr["t"]))
    for cu in spec["closeups"]:
        for cc in cu["cams"].values():
            v = rv[cc["view_id"]]
            for bname, b in cc["boxes"].items():
                K2, W2, H2 = _zoom_K(v["K"], b["box"], ZOOM_PX)
                cam(f"{v['id']}_{bname}", "closeup_zoom", f"closeup_zoom/{v['id']}_{bname}.png", f"real_{v['id']}", K2,
                    v["R"], v["tvec"], W2, H2, view_id=v["id"], box=[int(x) for x in b["box"]], box_name=bname,
                    instant=cu["instant"], t=float(v["t"]))
    poses = {}
    for role in ROLES:
        d = fits[role]
        fr = []
        for key, t in keys.items():
            p = interp_params(d, t)
            fr.append({"key": key, "t": t, "root_pose": p["global_orient"].tolist(), "body_pose": p["body_pose"].tolist(),
                       "jaw_pose": p["jaw_pose"].tolist(), "leye_pose": p["leye_pose"].tolist(),
                       "reye_pose": p["reye_pose"].tolist(), "lhand_pose": p["left_hand_pose"].tolist(),
                       "rhand_pose": p["right_hand_pose"].tolist(), "transl": p["transl"].tolist(),
                       "pelvis": p["joints"][0].tolist(), "joints22": p["joints"][:22].tolist()})
        poses[role] = {"betas": np.asarray(d["betas"], float).reshape(-1).tolist(),
                       "num_betas": int(d.get("num_betas", 16)), "flat_hand_mean": bool(d.get("flat_hand_mean", True)),
                       "model": str(d.get("model", "")), "frames": fr}
    return renders, poses, keys


def compare_prep(reuse_candidates: bool = False) -> dict:
    spec = _cmp_spec()
    CMP_INPUT.mkdir(parents=True, exist_ok=True)
    fits = {r: _load_fit(r) for r in ROLES}
    refs = compare_prep_refs(spec, fits, reuse=reuse_candidates)
    (CMP_INPUT / "refs.json").write_text(json.dumps(refs, indent=1, default=float))
    renders, poses, keys = compare_prep_views(spec, fits)
    (CMP_INPUT / "cameras.json").write_text(json.dumps({"take": TAKE, "spec": _ws(CMP / "views.json"),
                                                        "spec_created": spec.get("created"),
                                                        "variants": list(CMP_VARIANTS), "renders": renders}))
    (CMP_INPUT / "poses.json").write_text(json.dumps(poses))
    chk = check_transfer(poses)
    groups = {}
    for r in renders:
        groups[r["group"]] = groups.get(r["group"], 0) + 1
    info = {"renders": groups, "pose_keys": len(keys), "transfer_check": chk,
            "refs": {v: {r: refs[v][r]["n"] for r in ROLES} for v in CMP_VARIANTS},
            "bytes": int(sum(p.stat().st_size for p in CMP_INPUT.rglob("*") if p.is_file()))}
    (CMP_INPUT / "prep.json").write_text(json.dumps(info, indent=1, default=str))
    log(f"compare prep: {json.dumps(info, default=str)}")
    return info


def compare_job(max_minutes: float = 40) -> Path:
    rel = lambda p: _ws(p)[len("dancecap/"):]  # noqa: E731 - cwd of the commands is dancecap/
    outs = [_ws(CMP_POD_OUT) + "/"] + [_ws(CMP_RENDERS / v) + "/" for v in CMP_VARIANTS]
    d = {"schema": "dancecap.pod_job/1", "name": "cmp_lhmpp", "take": TAKE, "max_minutes": max_minutes, "needs": [],
         "inputs": [_ws(CMP_INPUT) + "/"],
         "exclude": ["dancecap/work/runpod/pywheels", "dancecap/work/runpod/wheelhouse", "mamma-turing"],
         "env": {"LHMPP_COMMIT": LHMPP_COMMIT, "LHMPP_ROOT": "/workspace/lhmpp_cand/LHM-plusplus",
                 "LHMPP_VENV": "/workspace/lhmpp_cand/venv", "LHMPP_MODEL": MODEL_NAME,
                 "TORCH_HOME": "/workspace/lhmpp_cand/torch_home", "HF_HOME": "/workspace/lhmpp_cand/hf_home"},
         "commands": [
             {"name": "install", "timeout_min": 35, "cmd": "bash remote/cand_lhmpp/install.sh"},
             {"name": "render", "timeout_min": 20,
              "cmd": f"/workspace/lhmpp_cand/venv/bin/python remote/cand_lhmpp/lhmpp_compare_pod.py --input "
                     f"{rel(CMP_INPUT)} --out {rel(CMP_POD_OUT)} --renders {rel(CMP_RENDERS)} "
                     f"--variants {','.join(CMP_VARIANTS)}"},
             {"name": "score", "timeout_min": 8, "optional": True,
              "cmd": "(python -m pip install -q --no-deps lpips==0.1.4 || echo 'lpips install failed: no LPIPS'); "
                     "python -m dancecap.mono4d.cand_lhmpp compare-score"}],
         "outputs": outs,
         "verify": [_ws(CMP_POD_OUT / "lhmpp_compare_pod.json")] +
                   [_ws(CMP_RENDERS / v / "i0_c03.png") for v in CMP_VARIANTS] +
                   [_ws(CMP_RENDERS / v / "clip_vr_seat" / "0104.png") for v in CMP_VARIANTS],
         "after_download": ["python -m dancecap.mono4d.cand_lhmpp compare-import --session <sid>"],
         "notes": [f"LHM++ ({MODEL_NAME}, CC BY-NC 4.0 weights) avatars of both dancers from phone-06 crops "
                   "(lhmpp_1phone) and from 2 crops of each of the 8 phones (lhmpp_8phone), posed with the reference "
                   "SMPL-X and rendered into every view of mono4d/compare/views.json (both dancers together); weights "
                   "stay on the pod; crops and frames go to our own pod only (owner consent 2026-10-19)"]}
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    p = JOB_DIR / "pod_job_compare.json"
    p.write_text(json.dumps(d, indent=1), newline="\n")
    return p


def _read_rgba(p: Path) -> tuple[np.ndarray, np.ndarray] | None:
    im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if im is None or im.ndim != 3 or im.shape[2] != 4:
        return None
    a = im[:, :, 3].astype(np.float32) / 255.0
    return im[:, :, 2::-1].astype(np.float32) / 255.0, a


def compare_score(lpips_net: bool = True) -> dict:
    """Masked metrics of each variant on the 18 real views (score_dir's definitions: crop = the dancer's real-mask
    bbox + 10 %, inside the mask, the render composited over black; *_both = real mask AND rendered dancer)."""
    from ..dancer_splats.evaluate import _crop_scores
    cams = json.loads((CMP_INPUT / "cameras.json").read_text())["renders"]
    real = [c for c in cams if c["group"] == "real"]
    variants = [v for v in CMP_VARIANTS if (CMP_RENDERS / v).exists()]
    device, net = "cpu", None
    if lpips_net:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            from ..dancer_splats.evaluate import _lpips
            net = _lpips(device)
        except Exception as e:  # noqa: BLE001
            log(f"LPIPS unavailable: {e}")
    rows = []
    for c in real:
        gt = _read_rgb(CMP_INPUT / "real" / f"{c['id']}.jpg")
        for var in variants:
            ra = _read_rgba(CMP_RENDERS / var / f"{c['id']}.png")
            rid = cv2.imread(str(CMP_RENDERS / var / "role" / f"{c['id']}.png"), cv2.IMREAD_COLOR)
            if ra is None or gt is None:
                continue
            rgb = ra[0] * ra[1][..., None]
            rw = None if rid is None else rid[:, :, ::-1].astype(np.float32) / 255.0
            for role in ROLES:
                m = cv2.imread(str(CMP_INPUT / "masks" / f"{c['id']}_{PID[role]}.png"), cv2.IMREAD_GRAYSCALE)
                if m is None or (m > 127).sum() < 50:
                    continue
                m = m > 127
                x0, y0, x1, y1 = _crop_box(m)
                s = _crop_scores(rgb[y0:y1, x0:x1], gt[y0:y1, x0:x1], m[y0:y1, x0:x1], net, device)
                row = {"view": c["id"], "cam": c["cam"], "t": round(c["t"], 4), "role": role, "variant": var,
                       "crop": [int(x0), int(y0), int(x1), int(y1)],
                       **{k: (None if s.get(k) is None else float(s[k])) for k in s}}
                if rw is not None:
                    own = rw[:, :, ROLES.index(role)] > 0.5
                    u = (own | m).sum()
                    row["iou"] = float((own & m).sum() / u) if u else None
                    row["coverage"] = float((own & m).sum() / m.sum())
                    sb = (own & m)[y0:y1, x0:x1]
                    s2 = _crop_scores(rgb[y0:y1, x0:x1], gt[y0:y1, x0:x1], sb, net, device)
                    row.update({f"{k}_both": float(s2[k]) for k in ("psnr", "ssim", "lpips", "psnr_cc")
                                if s2.get(k) is not None})
                rows.append(row)
    keys = ("psnr", "ssim", "lpips", "psnr_cc", "ssim_cc", "lpips_cc", "iou", "coverage", "psnr_both", "ssim_both",
            "lpips_both", "psnr_cc_both")
    summ = {}
    for var in variants:
        summ[var] = {}
        groups = [("all", lambda r: True)] + [(ro, lambda r, ro=ro: r["role"] == ro) for ro in ROLES] + \
            [(f"cam{c}", lambda r, c=c: r["cam"] == c) for c in ("03", "02", "06")]
        for g, sel in groups:
            rr = [r for r in rows if r["variant"] == var and sel(r)]
            summ[var][g] = {"n": len(rr), **{k: (round(float(np.median([r[k] for r in rr if r.get(k) is not None])), 4)
                                                 if any(r.get(k) is not None for r in rr) else None) for k in keys}}
    out = {"metrics": "score_dir's: dancecap.dancer_splats.evaluate._crop_scores on the dancer's real-mask bbox + 10 %, "
                      "inside the real mask (render composited over black: pixels it misses count as black); *_cc after "
                      "a least-squares colour fit; *_both = real mask AND rendered dancer; iou / coverage of the "
                      "rendered role (> 0.5) vs the real mask. Real frames are not undistorted (k1 of phone 02 = "
                      "0.0139: <= ~7 px at the corners)", "lpips": net is not None, "device": device,
           "variants": variants, "rows": rows, "summary": summ}
    CMP_POD_OUT.mkdir(parents=True, exist_ok=True)
    (CMP_POD_OUT / "metrics.json").write_text(json.dumps(out, indent=1, default=float))
    log(json.dumps(summ, default=float)[:3000])
    return out


def _session_meta_cmp(sid: str | None) -> dict:
    root = PACKAGE_ROOT / "work" / "runpod" / "sessions"
    if sid is None:
        cands = sorted((p for p in root.glob("*") if (p / "result.json").exists()
                        and '"cmp_lhmpp"' in (p / "result.json").read_text(errors="ignore")),
                       key=lambda p: p.stat().st_mtime)
        if not cands:
            return {}
        sid = cands[-1].name
    p = root / sid / "result.json"
    if not p.exists():
        return {"session": sid}
    r = json.loads(p.read_text())
    pod = r.get("pod") or {}
    job_ = (r.get("jobs") or {}).get("cmp_lhmpp") or {}
    led = _ledger_rows([pod.get("id")]) if pod.get("id") else []
    steps = {x["step"]: x["seconds"] for x in r.get("timings", []) if "step" in x}
    return {"session": sid, "pod_id": pod.get("id"), "gpu": pod.get("gpu"),
            "cloud": (led[0].get("cloud") if led else None) or pod.get("cloud"),
            "cost_per_hr": pod.get("cost_per_hr"), "cost_usd": pod.get("ledger_cost"),
            "pod_minutes": round((pod.get("lifetime_s") or 0) / 60, 1), "job_status": job_.get("status"),
            "commands": job_.get("commands"), "job_seconds": job_.get("seconds"), "download_mb": job_.get("download_mb"),
            "steps_s": steps}


def compare_import(sid: str | None = None, pods: list[str] | None = None) -> dict:
    """Verify the pod's renders against cameras.json, cut the native close-up crops, write each variant's
    manifest.json, review sheets and compare/lhmpp/report.json."""
    spec = _cmp_spec()
    cams = json.loads((CMP_INPUT / "cameras.json").read_text())["renders"]
    refs = json.loads((CMP_INPUT / "refs.json").read_text())
    pod = json.loads((CMP_POD_OUT / "lhmpp_compare_pod.json").read_text())
    metrics = json.loads((CMP_POD_OUT / "metrics.json").read_text()) if (CMP_POD_OUT / "metrics.json").exists() else None
    ses = _session_meta_cmp(sid)
    poses = json.loads((CMP_INPUT / "poses.json").read_text())
    pose_t = {fr["key"]: fr["t"] for fr in poses["lead"]["frames"]}
    rv = {v["id"]: v for v in spec["real_views"]}
    report = {"take": TAKE, "spec": _ws(CMP / "views.json"), "spec_created": spec.get("created"), "model": MODEL_NAME,
              "commit": LHMPP_COMMIT, "licence": LICENCE, "session": ses, "pod": pod.get("summary"),
              "transfer": pod.get("transfer"), "variants": {}}
    for var, cfg in CMP_VARIANTS.items():
        vdir = CMP_RENDERS / var
        pv = pod.get("variants", {}).get(var, {})
        files, missing, bad = {}, [], []
        for c in cams:
            p = vdir / c["out"]
            if not p.exists():
                missing.append(c["out"])
                continue
            im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if im is None or im.shape[:2] != (c["H"], c["W"]) or im.ndim != 3 or im.shape[2] != 4:
                bad.append(c["out"])
            files.setdefault(c["group"], []).append(c)
        # native-pixel close-up crops of the real-view renders (the photo's own pixels for the side-by-side)
        cdir = vdir / "closeup"
        cdir.mkdir(parents=True, exist_ok=True)
        closeups = []
        for cu in spec["closeups"]:
            for cc in cu["cams"].values():
                src = vdir / f"{cc['view_id']}.png"
                im = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
                for bname, b in cc["boxes"].items():
                    x0, y0, x1, y1 = [int(x) for x in b["box"]]
                    name = f"{cc['view_id']}_{bname}"
                    if im is not None:
                        cv2.imwrite(str(cdir / f"{name}.png"), im[y0:y1, x0:x1])
                    closeups.append({"id": name, "instant": cu["instant"], "view_id": cc["view_id"], "box": bname,
                                     "box_px": [x0, y0, x1, y1], "native": f"closeup/{name}.png",
                                     "zoom": f"closeup_zoom/{name}.png",
                                     "best_face_cam": cu["best_cam"], "best_hair_cam": cu["best_cam_hair"]})
        # what this variant saw, per real view
        inputs = {r: refs[var][r]["frames"] for r in ROLES}
        seen = {}
        for vid, v in rv.items():
            same = [(r, f) for r in ROLES for f in inputs[r] if f["cam"] == v["cam"]]
            exact = sorted({r for r, f in same if int(f["i"]) == int(v["frame"])})
            near = min((abs(float(f["t"]) - float(v["t"])) for _, f in same), default=None)
            seen[vid] = {"phone_used": bool(same), "exact_frame_input_for": exact,
                         "nearest_input_frame_same_phone_s": None if near is None else round(near, 3)}
        groups = {g: len(x) for g, x in files.items()}
        man = {"schema": "dancecap.mono4d.appearance_compare.renders/1", "method": var, "renderer": "lhmpp",
               "what": cfg["what"], "take": TAKE, "spec": _ws(CMP / "views.json"), "spec_created": spec.get("created"),
               "model": f"LHM++ {MODEL_NAME} @ {LHMPP_COMMIT[:7]} (weights CC BY-NC 4.0)",
               "pose": "reference SMPL-X work/larissa_kadu_38/smplx/{lead,follow}.npz, per-joint slerp at each view's "
                       "time; LHM++ shape mode ref (betas[:10]), trans = pelvis_ref - J0_rest(betas[:10]); both dancers "
                       "in one rasterisation (mutual occlusion)",
               "render": {**pod.get("render_settings", {}), "format": "RGBA PNG, straight alpha, transparent background",
                          "sizes": "real views at the phone's native size (02/06 1080 x 1920, 03 480 x 848); turntable "
                                   "and VR seat 1080 x 1080; closeup_zoom 512 px on the long side", "distortion":
                          "pinhole (the real phone's k1 ignored: <= ~7 px at the corners of phone 02)"},
               "inputs": {"rules": refs[var]["rules"], "limits": OCC_LIMITS,
                          **{r: {"n": refs[var][r]["n"], "frames": [{k: f.get(k) for k in
                                                                     ("slot", "cam", "i", "t", "yaw_deg", "vis",
                                                                      "occ", "relaxed", "replacement", "file")}
                                                                    for f in refs[var][r]["frames"]],
                                 "dropped_seed": refs[var][r].get("dropped_seed")} for r in ROLES}},
               "avatar": {r: {k: pv.get(r, {}).get(k) for k in ("gaussians", "n_refs", "infer_s", "rest_height_m",
                                                                 "fk_vs_ref_joints_mm", "gauss_to_own_mesh_mm",
                                                                 "gauss_to_ref_mesh_mm", "opacity_median",
                                                                 "scale_median_mm", "peak_gpu_gb", "error")}
                          | {"cano_ply": _ws(CMP_POD_OUT / "gs" / f"{r}_{var}_cano.ply")} for r in ROLES},
               "scene_ply": _ws(CMP_POD_OUT / "gs" / pv["scene_ply"]) if pv.get("scene_ply") else None,
               "files": {
                   "real": [{"id": c["id"], "file": c["out"], "role": f"role/{c['id']}.png", "cam": c["cam"],
                             "frame": c["frame"], "t": pose_t[c["pose"]], "seen_by_this": seen[c["id"]]}
                            for c in files.get("real", [])],
                   "turntable": [{"id": c["id"], "file": c["out"], "t": pose_t[c["pose"]],
                                  "azimuth_deg": c.get("azimuth_deg")} for c in files.get("turntable", [])],
                   "clip_vr_seat": {"dir": "clip_vr_seat", "n": groups.get("clip_vr_seat", 0),
                                    "frames": [{"j": c["clip_index"], "grid_k": c["grid_k"], "t": c["t"]}
                                               for c in files.get("clip_vr_seat", [])]},
                   "clip_cam03": {"dir": "clip_cam03", "n": groups.get("clip_cam03", 0),
                                  "frames": [{"j": c["clip_index"], "frame": c["frame"], "t": c["t"]}
                                             for c in files.get("clip_cam03", [])]},
                   "closeups": closeups,
                   "instant_clip_index": {x["id"]: int(x["k"]) - int(spec["clip"]["k"][0]) for x in spec["instants"]}},
               "counts": groups, "missing": missing, "bad_size_or_not_rgba": bad,
               "pod": {k: ses.get(k) for k in ("session", "gpu", "pod_minutes", "cost_usd", "cost_per_hr")},
               "metrics_summary": None if metrics is None else metrics["summary"].get(var),
               "caveats": ["phone 06 is lhmpp_1phone's input phone; 02 / 06 / 03 views of lhmpp_8phone are other frames "
                           "of input phones (no input within 0.05 s of a real view)" if var == "lhmpp_8phone" else
                           "phone 06 is the input phone: real views on 06 can be exact input frames (seen_by_this)",
                           "naked-body pose: hair and loose clothes follow the SMPL-X skinning field of LHM++ (no "
                           "dynamics); the fit's hands are rigid (MAMMA)"]}
        (vdir / "manifest.json").write_text(json.dumps(man, indent=1, default=float))
        report["variants"][var] = {"counts": groups, "missing": len(missing), "bad": len(bad),
                                   "gaussians": {r: pv.get(r, {}).get("gaussians") for r in ROLES},
                                   "manifest": _ws(vdir / "manifest.json"),
                                   "metrics": None if metrics is None else metrics["summary"].get(var)}
        log(f"{var}: {groups}; missing {len(missing)}, bad {len(bad)}")
    report["sheets"] = [str(p) for p in _cmp_sheets(spec, cams)]
    if pods:
        rows = _ledger_rows(pods)
        report["pods"] = rows
        report["cost_usd_all_attempts"] = round(sum(float(x.get("cost") or 0) for x in rows), 4)
    report["summary"] = {v: report["variants"][v] for v in CMP_VARIANTS} | {"session": ses}
    (CMP_OUT / "report.json").write_text(json.dumps(report, indent=1, default=float))
    log(f"report -> {CMP_OUT / 'report.json'}")
    return report


def _on_grey(rgba_path: Path, box=None, grey: int = 128) -> np.ndarray | None:
    im = cv2.imread(str(rgba_path), cv2.IMREAD_UNCHANGED)
    if im is None:
        return None
    if box is not None:
        x0, y0, x1, y1 = box
        im = im[y0:y1, x0:x1]
    a = im[:, :, 3:4].astype(np.float32) / 255.0
    return (im[:, :, :3] * a + grey * (1 - a) + 0.5).astype(np.uint8)


def _cmp_sheets(spec: dict, cams: list[dict], h: int = 360) -> list[Path]:
    """Quick-look sheets (the compose step builds the real side-by-sides): real | 1phone | 8phone per real view; the
    turntable; the VR seat at the six instants; the close-ups (real native | 1phone zoom | 8phone zoom)."""
    out = []
    variants = [v for v in CMP_VARIANTS if (CMP_RENDERS / v).exists()]
    rows = []
    for v in spec["real_views"]:
        gt = cv2.imread(str(CMP_INPUT / "real" / f"{v['id']}.jpg"))
        tiles = [_tile(gt, v["crop"], h, f"real {v['id']} t={v['t']:.2f}")]
        for var in variants:
            im = _on_grey(CMP_RENDERS / var / f"{v['id']}.png")
            if im is not None:
                tiles.append(_tile(im, v["crop"], h, var))
        rows.append(np.concatenate([np.pad(t, ((0, 0), (0, 4), (0, 0)), constant_values=255) for t in tiles], 1))
    for k in range(0, len(rows), 6):
        part = rows[k:k + 6]
        W = max(r.shape[1] for r in part)
        part = [np.pad(r, ((0, 4), (0, W - r.shape[1]), (0, 0)), constant_values=255) for r in part]
        p = CMP_OUT / f"sheet_real_{k // 6}.jpg"
        cv2.imwrite(str(p), np.concatenate(part, 0), [cv2.IMWRITE_JPEG_QUALITY, 88])
        out.append(p)
    for var in variants:
        tt = [_on_grey(CMP_RENDERS / var / f"{v['id']}.png") for v in spec["novel"]["turntable"]["views"]]
        tt = [cv2.resize(t, (300, 300), interpolation=cv2.INTER_AREA) for t in tt if t is not None]
        if tt:
            grid = np.concatenate([np.concatenate(tt[i:i + 6], 1) for i in range(0, len(tt) - len(tt) % 6, 6)], 0)
            p = CMP_OUT / f"sheet_turntable_{var}.jpg"
            cv2.imwrite(str(p), grid, [cv2.IMWRITE_JPEG_QUALITY, 88])
            out.append(p)
        k0 = int(spec["clip"]["k"][0])
        vr = [_on_grey(CMP_RENDERS / var / "clip_vr_seat" / f"{int(x['k']) - k0:04d}.png") for x in spec["instants"]]
        vr = [cv2.resize(t, (360, 360), interpolation=cv2.INTER_AREA) for t in vr if t is not None]
        if vr:
            p = CMP_OUT / f"sheet_vr_seat_{var}.jpg"
            cv2.imwrite(str(p), np.concatenate(vr, 1), [cv2.IMWRITE_JPEG_QUALITY, 88])
            out.append(p)
    rows = []
    for cu in spec["closeups"]:
        for cc in cu["cams"].values():
            gt = cv2.imread(str(CMP_INPUT / "real" / f"{cc['view_id']}.jpg"))
            tiles = []
            for bname, b in cc["boxes"].items():
                tiles.append(_tile(gt, b["box"], 200, f"{cc['view_id']} {bname}"))
                for var in variants:
                    im = _on_grey(CMP_RENDERS / var / "closeup_zoom" / f"{cc['view_id']}_{bname}.png")
                    if im is not None:
                        tiles.append(_tile(im, (0, 0, im.shape[1], im.shape[0]), 200, var.replace("lhmpp_", "")))
            rows.append(np.concatenate([np.pad(t, ((0, 0), (0, 3), (0, 0)), constant_values=255) for t in tiles], 1))
    if rows:
        W = max(r.shape[1] for r in rows)
        rows = [np.pad(r, ((0, 3), (0, W - r.shape[1]), (0, 0)), constant_values=255) for r in rows]
        p = CMP_OUT / "sheet_closeups.jpg"
        cv2.imwrite(str(p), np.concatenate(rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 88])
        out.append(p)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.mono4d.cand_lhmpp")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--no-baseline", action="store_true")
    p.add_argument("--reuse-refs", action="store_true")
    p = sub.add_parser("job")
    p.add_argument("--max-minutes", type=float, default=60)
    p = sub.add_parser("score")
    p.add_argument("--dir", default=str(OUT))
    p.add_argument("--no-lpips", action="store_true")
    p = sub.add_parser("import")
    p.add_argument("--session", default=None)
    p.add_argument("--pods", default="", help="comma-separated pod ids of every attempt (costs from the ledger)")
    p = sub.add_parser("compare-prep")
    p.add_argument("--reuse-candidates", action="store_true")
    p = sub.add_parser("compare-job")
    p.add_argument("--max-minutes", type=float, default=40)
    p = sub.add_parser("compare-score")
    p.add_argument("--no-lpips", action="store_true")
    p = sub.add_parser("compare-import")
    p.add_argument("--session", default=None)
    p.add_argument("--pods", default="", help="comma-separated pod ids of every attempt (costs from the ledger)")
    a = ap.parse_args(argv)
    if a.cmd == "prep":
        print(json.dumps(prep(baseline=not a.no_baseline, reuse_refs=a.reuse_refs), indent=1, default=str))
    elif a.cmd == "job":
        print(job(a.max_minutes))
    elif a.cmd == "score":
        d = Path(a.dir)
        if not d.is_absolute():
            d = PACKAGE_ROOT / d
        score_dir(d, lpips_net=not a.no_lpips)
    elif a.cmd == "import":
        print(json.dumps(import_run(a.session, [x for x in a.pods.split(",") if x])["summary"], indent=1, default=float))
    elif a.cmd == "compare-prep":
        print(json.dumps(compare_prep(reuse_candidates=a.reuse_candidates), indent=1, default=str))
    elif a.cmd == "compare-job":
        print(compare_job(a.max_minutes))
    elif a.cmd == "compare-score":
        compare_score(lpips_net=not a.no_lpips)
    elif a.cmd == "compare-import":
        print(json.dumps(compare_import(a.session, [x for x in a.pods.split(",") if x])["summary"], indent=1,
                         default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
