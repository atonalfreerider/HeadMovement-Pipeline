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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
