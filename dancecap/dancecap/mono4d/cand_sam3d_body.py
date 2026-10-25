r"""mono4d track "sam3d_body" (2026-10-19): Meta SAM 3D Body on ONE handheld phone (06) of larissa_kadu_38, 58-62 s.

SAM 3D Body ("3DB", github.com/facebookresearch/sam-3d-body @ b5c765a, SAM License; checkpoint facebook/sam-3d-body-dinov3
@ 11aaa34 on Hugging Face, gated, DINOv3 ViT-H+ backbone code from facebookresearch/dinov3 @ 5aa7e93, the last commit
before the 3DB release) is a promptable single-person mesh recoverer: one crop per person, optional mask prompt, output
= MHR body (127-joint Momentum rig, 18439-vertex LOD1 mesh) in the camera frame (OpenCV axes, metres, translation from
the predicted (s, tx, ty) and the given focal length).

What runs (pod, dancecap/remote/cand_sam3d_body/): the 120 canonical frames of the window (1080x1920, unchanged), each
dancer prompted SEPARATELY and DIRECTLY with OUR SAM 2.1 masks (dancers/06/masks/mask_{i:04d}_{01|02}.png) and the
tight box of that mask, our intrinsics (cameras/06.npz K: f 1147.09 px, c = (539.5, 959.5)) passed as cam_int; no
detector, no SAM 3, no FOV estimator, inference_type "full" (body decoder + both hand decoders). The paper's Table 8
reports Hi4D MPJPE 76.4 -> 47.0 with mask prompts. The official repo uses two mask scales (notebook: 0/255 for an
external mask; demo SAM 2 path: 0/1), so both run:
    mask255  box + mask 0/255 (the notebook's external-mask recipe)     -> candidate "sam3d_body"  (pre-registered)
    mask01   box + mask 0/1   (the demo's SAM 2 path)                   -> candidate "sam3d_body_mask01"
    nomask   box only (same box)                                        -> candidate "sam3d_body_nomask"
MHR -> SMPL-X: the OFFICIAL converter (facebookresearch/MHR @ 877cba5, tools/mhr_smpl_conversion, Apache-2.0):
Conversion(method="pytorch").convert_sam3d_output_to_smpl (barycentric MHR->SMPL-X surface map mhr2smplx_mapping.npz,
then an Adam fit of SMPL-X neutral, 10 betas, per frame, own identity per frame) on pred_vertices + pred_cam_t; SMPL-X
joints = smplx forward of the fitted parameters (joints[:, :55], FK joints of the SMPL-X 2020 neutral model). Variant
"sam3d_body_1id" = mask255 through convert_mhr2smpl(single_identity=True) per dancer (one shape over the 120 frames).
The converter needs the MHR model only for its LOD1 faces. MHR.from_files (pymomentum-cpu 0.1.114, torch 2.8) SEGFAULTED
on the pod (rc 139 before any output, session 20261019-175225-batch), so the jobs now pass --force-stub: the same faces
from the tool's own assets/mhr_face_mask.ply (the full LOD1 mesh, 36874 faces), checked equal to the 3DB head's faces.
In that session the conversion was run LOCALLY on the downloaded MHR outputs (same script, same stub path, RTX 2080 Ti,
~70 s per variant); sam3db_pod.json convert.where and each candidate's meta (converter_where) say so.
Roles: the prompts are per role (01 = lead, 02 = follow), checked with candidates.assign_roles.
SAM 3D Body is per frame and per person: no temporal model, no relative placement of the two dancers (paper App. C).

    python -m dancecap.mono4d.cand_sam3d_body check-access   # HF account of the key file + gated access (no download)
    python -m dancecap.mono4d.cand_sam3d_body prep        # frames + masks + K + trimmed SMPL-X -> mono4d/sam3d_body/input/
    python -m dancecap.mono4d.cand_sam3d_body job         # dancecap/remote/cand_sam3d_body/pod_job.json
    python -m dancecap.mono4d.cand_sam3d_body job-body4d  # dancecap/remote/cand_sam3d_body/pod_job_body4d.json
    (DANCECAP_RUNPOD_CAP=... python -m dancecap remote run --jobs <abs paths of both, comma-separated> --max-cost ...)

SAM-Body4D variants (job cand_sam3d_body4d, remote/cand_sam3d_body/body4d_pod.py; gaomingqi/sam-body4d @ 21af102, MIT):
the app's own OfflineApp.on_4d_generation with SAM 3 BYPASSED (our SAM 2.1 masks are its masklets; facebook/sam3 is
gated and not requested), our K as cam_int (instead of MoGe-2), its fork of 3DB (Kalman-smoothed body / hand pose,
EMA-smoothed global rotation, MHR shape + scale frozen at each identity's first frame - per identity: upstream shares
the first identity's with every other one), Diffusion-VAS amodal mask + RGB completion of occluded frames (resolutions
transposed for the portrait video):
    body4d         completion ON                                       -> candidate "sam3d_body4d"
    body4d_nocomp  completion OFF (temporal smoothing + frozen shape)    -> candidate "sam3d_body4d_nocomp"
    python -m dancecap.mono4d.cand_sam3d_body import [--session SID]   # roles, save candidates, QA sheet
The whole-take run of the same recipe (john_lula, all 2643 frames, chunked + resumable pod side): mono4d/jl_sam3db.py.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np

from ..take import PACKAGE_ROOT, WORKSPACE
from . import candidates as C

# Hugging Face token of an account with access to the gated facebook/sam-3d-body-dinov3 (a key FILE, never its value; sent to the pod as
# a runner secret): DANCECAP_HF_KEY_FILE, else ~/Desktop/sam3d-body-key-hf.txt (the Windows PC kept it on its Desktop too)
HF_KEY_FILE = os.environ.get("DANCECAP_HF_KEY_FILE", str(Path.home() / "Desktop" / "sam3d-body-key-hf.txt")).replace("\\", "/")
TAKE, CAM = "larissa_kadu_38", "06"
WINDOW = (58.0, 62.0)
WORK = PACKAGE_ROOT / "work" / TAKE
OUT = WORK / "mono4d" / "sam3d_body"
INPUT = OUT / "input"
POD_OUT = OUT / "pod_out"
JOB_DIR = PACKAGE_ROOT / "remote" / "cand_sam3d_body"
S3DB_COMMIT = "b5c765a0d89d789985e186d396315e7590887b94"
HF_REPO, HF_REV = "facebook/sam-3d-body-dinov3", "11aaa346c7204874a1cbafe3d39a979080b2c55a"
DINOV3_COMMIT = "5aa7e93ab42ae3526291f6b4e3fcd22637e03326"
MHR_COMMIT = "877cba5ee61a5f77d1457bf4de93a66695132bd9"
MHR_ASSETS_RELEASE = "v1.0.1"
VARIANTS = {"mask255": "sam3d_body", "mask01": "sam3d_body_mask01", "nomask": "sam3d_body_nomask"}
# SAM-Body4D (job cand_sam3d_body4d, body4d_pod.py): SAM 3 bypassed with our masks; with / without its occlusion completion
B4D_COMMIT = "21af1020979ef32ddf6be3597ef59a68bad2f1bf"
B4D_VARIANTS = {"body4d": "sam3d_body4d", "body4d_nocomp": "sam3d_body4d_nocomp"}
B4D_REVS = {"VAS_MASK_REV": "7a79fa0", "VAS_RGB_REV": "428c072", "DA2_REV": "cbbb86a"}
LICENCE = ("SAM 3D Body code + checkpoint: SAM License (Meta; gated on Hugging Face); DINOv3 code: DINOv3 licence; MHR "
           "+ MHR->SMPL-X converter: Apache-2.0; SMPL-X model (fit + joints): SMPL-X licence, non-commercial. "
           "Research use only; outputs stay local.")
ROLE_PID = {"lead": "01", "follow": "02"}


def log(msg: str) -> None:
    print(f"[cand_sam3d_body] {msg}", flush=True)


def _ws(p: Path) -> str:
    return str(Path(p).resolve().relative_to(WORKSPACE.resolve())).replace("\\", "/")


# ------------------------------------------------------------------------------------------------ prep + job


def _mask_box(m: np.ndarray) -> list[float] | None:
    ys, xs = np.nonzero(m)
    if not len(xs):
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def _trim_smplx(src: Path, dst: Path) -> dict:
    """SMPL-X 2020 neutral with 10 shape + 10 expression directions, float32 (what smplx.SMPLX(num_betas=10,
    num_expression_coeffs=10) reads anyway): ~110 MB -> ~45 MB over the shared uplink. Checked against the full file."""
    import torch
    import smplx
    z = dict(np.load(src, allow_pickle=True))
    sd = z["shapedirs"]
    z["shapedirs"] = np.concatenate([sd[:, :, :10], sd[:, :, 300:310]], axis=2).astype(np.float32)
    for k in ("posedirs", "J_regressor", "weights", "v_template"):
        z[k] = np.asarray(z[k], np.float32)
    dst.parent.mkdir(parents=True, exist_ok=True)
    z = {k: v for k, v in z.items() if v.dtype != object}   # part2num / joint2num (dicts): smplx never reads them, NumPy 2.5 will not save them
    np.savez_compressed(dst, **z)
    torch.manual_seed(0)
    kw = dict(gender="neutral", use_pca=False, flat_hand_mean=True)
    a, b = smplx.SMPLX(model_path=str(src), **kw), smplx.SMPLX(model_path=str(dst), **kw)
    P = dict(betas=torch.randn(4, 10), expression=0.5 * torch.randn(4, 10), body_pose=0.3 * torch.randn(4, 63),
             global_orient=0.3 * torch.randn(4, 3), left_hand_pose=0.3 * torch.randn(4, 45),
             right_hand_pose=0.3 * torch.randn(4, 45), transl=torch.randn(4, 3), jaw_pose=0.1 * torch.randn(4, 3),
             leye_pose=torch.zeros(4, 3), reye_pose=torch.zeros(4, 3))
    with torch.no_grad():
        ja, jb = a(**P), b(**P)
    dj = float((ja.joints - jb.joints).abs().max())
    dv = float((ja.vertices - jb.vertices).abs().max())
    if dj > 1e-5 or dv > 1e-5:
        raise SystemExit(f"trimmed SMPL-X differs from the full model: joints {dj}, vertices {dv}")
    return {"bytes": dst.stat().st_size, "max_joint_diff_m": dj, "max_vertex_diff_m": dv}


def prep(t0: float = WINDOW[0], t1: float = WINDOW[1]) -> dict:
    import cv2
    fj = json.loads((WORK / "frames" / CAM / "frames.json").read_text())
    frames = fj["frames"]
    cam = np.load(WORK / "cameras" / f"{CAM}.npz")
    if len(cam["t"]) != len(frames) or np.abs(cam["t"] - np.array([f["t"] for f in frames])).max() > 1e-3:
        raise SystemExit("cameras/06.npz and frames.json are not index-aligned")
    roles = json.loads((WORK / "dancers" / "roles.json").read_text())["cams"][CAM]["frames"]
    sel = [i for i, f in enumerate(frames) if t0 <= f["t"] <= t1]
    for sub in ("frames", "masks"):
        d = INPUT / sub
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    rows, ious = [], {r: [] for r in C.ROLES}
    from .holdout import box_iou
    for i in sel:
        src = WORK / "frames" / CAM / frames[i]["file"]
        shutil.copy2(src, INPUT / "frames" / src.name)
        r = roles[i]
        if abs(r["t"] - frames[i]["t"]) > 1e-3:
            raise SystemExit(f"roles.json frame {i} t {r['t']} != {frames[i]['t']}")
        row = {"file": src.name, "i": i, "t": frames[i]["t"], "K": cam["K"][i].tolist(), "boxes": {}, "mask_boxes": {},
               "mask_px": {}}
        for role in C.ROLES:
            mp = WORK / "dancers" / CAM / "masks" / f"mask_{i:04d}_{ROLE_PID[role]}.png"
            m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if m is None:
                raise SystemExit(f"missing mask {mp}")
            shutil.copy2(mp, INPUT / "masks" / mp.name)
            mb = _mask_box(m > 127)
            rb = (r.get(role) or {}).get("box")
            row["boxes"][role], row["mask_boxes"][role], row["mask_px"][role] = rb, mb, int((m > 127).sum())
            if rb and mb:
                ious[role].append(box_iou(np.array(rb, float), np.array(mb, float)))
        rows.append(row)
    meta = {"take": TAKE, "cam": CAM, "window": [t0, t1], "width": fj["width"], "height": fj["height"],
            "role_pid": ROLE_PID, "frames": rows}
    (INPUT / "meta.json").write_text(json.dumps(meta, indent=1))
    sm = _trim_smplx(WORKSPACE / "smplx" / "SMPLX_NEUTRAL.npz", INPUT / "smplx" / "SMPLX_NEUTRAL.npz")
    info = {"frames": len(rows), "first": rows[0]["file"], "last": rows[-1]["file"], "t": [rows[0]["t"], rows[-1]["t"]],
            "f_px": float(np.median(cam["K"][sel][:, 0, 0])), "c_px": cam["K"][sel[0]][:2, 2].tolist(),
            "size": [fj["width"], fj["height"]], "mask_box_vs_roles_box_iou_median":
                {r: float(np.median(v)) for r, v in ious.items() if v}, "smplx_trimmed": sm,
            "bytes": int(sum(p.stat().st_size for p in INPUT.rglob("*") if p.is_file()))}
    (INPUT / "prep.json").write_text(json.dumps(info, indent=1))
    return info


def job(max_minutes: float = 25, limit: int = 0) -> Path:
    rel = lambda p: _ws(p)[len("dancecap/"):]  # noqa: E731  (cwd of the commands is <ws>/dancecap)
    lim = f" --limit {limit}" if limit else ""
    root = "/workspace/s3db_cand"
    d = {"schema": "dancecap.pod_job/1", "name": "cand_sam3d_body", "take": TAKE, "max_minutes": max_minutes, "needs": [],
         # its own uv venv (install.sh): no bootstrap base / check for the image python
         "base": False,
         "inputs": [_ws(INPUT) + "/"],
         # no MAMMA code, body models or cached wheels: frames + masks + code only over the shared 2.5 MB/s uplink
         "exclude": ["dancecap/work/runpod/pywheels", "dancecap/work/runpod/wheelhouse", "mamma-turing"],
         "secrets": {"HF_TOKEN": HF_KEY_FILE},
         "env": {"S3DB_ROOT": root, "S3DB_COMMIT": S3DB_COMMIT, "S3DB_HF_REPO": HF_REPO, "S3DB_HF_REV": HF_REV,
                 "DINOV3_COMMIT": DINOV3_COMMIT, "MHR_COMMIT": MHR_COMMIT, "MHR_ASSETS_RELEASE": MHR_ASSETS_RELEASE,
                 "TORCH_HOME": f"{root}/torch_home", "HF_HOME": f"{root}/hf_home", "UV_LINK_MODE": "hardlink",
                 "TQDM_DISABLE": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1"},
         "commands": [
             {"name": "install", "timeout_min": 10, "cmd": "bash remote/cand_sam3d_body/install.sh"},
             {"name": "infer", "timeout_min": 8,
              "cmd": f"{root}/venv/bin/python remote/cand_sam3d_body/sam3db_pod.py infer --input {rel(INPUT)} "
                     f"--out {rel(POD_OUT)} --variants {','.join(VARIANTS)}{lim}"},
             {"name": "convert", "timeout_min": 7,
              "cmd": f"{root}/venv/bin/python remote/cand_sam3d_body/sam3db_pod.py convert --input {rel(INPUT)} "
                     f"--out {rel(POD_OUT)} --variants {','.join(VARIANTS)} --single-identity mask255 --force-stub"}],
         # --force-stub: MHR.from_files (pymomentum-cpu 0.1.114 + torch 2.8) segfaulted on the pod (rc 139, session
         # 20261019-175225-batch); the converter's pytorch path only needs the LOD1 faces, which the stub takes from the
         # tool's own assets/mhr_face_mask.ply and checks against the 3DB head's faces
         "outputs": [_ws(POD_OUT) + "/"],
         "verify": [_ws(POD_OUT / "sam3db_pod.json")] + [_ws(POD_OUT / f"mhr_{v}.npz") for v in VARIANTS]
                   + [_ws(POD_OUT / f"smplx_{v}.npz") for v in VARIANTS],
         "after_download": ["python -m dancecap.mono4d.cand_sam3d_body import --session <sid>"],
         "notes": ["SAM 3D Body (SAM License, gated HF checkpoint downloaded on the pod with the owner's HF token) on the "
                   "120 canonical frames of phone 06, 58-62 s, each dancer prompted with OUR SAM 2.1 mask + its box and "
                   "our K; variants mask255 / mask01 / nomask; official MHR->SMPL-X converter (pytorch) on the pod. "
                   "Checkpoints stay on the pod; the dancers' frames go to our own pod only (owner consent 2026-10-19); "
                   "the bundle holds a trimmed SMPL-X neutral model (owner's licence) for the fit."]}
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    p = JOB_DIR / "pod_job.json"
    p.write_text(json.dumps(d, indent=1), newline="\n")
    return p


def job_body4d(max_minutes: float = 42, limit: int = 0) -> Path:
    """SAM-Body4D variants as a second job of the same batch (no `after`: install_body4d.sh runs install.sh itself, so
    it also works alone): body4d_nocomp (temporal smoothing + frozen per-identity shape, completion off) first, then
    body4d (Diffusion-VAS occlusion completion on), both optional so the conversion always runs."""
    rel = lambda p: _ws(p)[len("dancecap/"):]  # noqa: E731
    lim = f" --limit {limit}" if limit else ""
    root = "/workspace/s3db_cand"
    py = f"{root}/venv/bin/python"
    common = (f"{py} remote/cand_sam3d_body/body4d_pod.py run --input {rel(INPUT)} --out {rel(POD_OUT)} "
              f"--body4d-root {root}/sam-body4d --ckpt-root {root}/ckpt{lim}")
    base = json.loads(((JOB_DIR / "pod_job.json") if (JOB_DIR / "pod_job.json").exists() else job()).read_text())
    d = {"schema": "dancecap.pod_job/1", "name": "cand_sam3d_body4d", "take": TAKE, "max_minutes": max_minutes,
         "needs": [], "inputs": base["inputs"], "exclude": base["exclude"], "secrets": base["secrets"],
         "env": {**base["env"], "B4D_COMMIT": B4D_COMMIT, **B4D_REVS},
         "commands": [
             {"name": "install_b4d", "timeout_min": 12, "cmd": "bash remote/cand_sam3d_body/install_body4d.sh"},
             {"name": "b4d_nocomp", "timeout_min": 6, "optional": True,
              "cmd": f"{common} --work {root}/b4d_work/nocomp --tag body4d_nocomp --no-completion"},
             {"name": "b4d_comp", "timeout_min": 18, "optional": True,
              "cmd": f"{common} --work {root}/b4d_work/comp --tag body4d"},
             {"name": "convert_b4d", "timeout_min": 6,
              "cmd": f"{py} remote/cand_sam3d_body/sam3db_pod.py convert --input {rel(INPUT)} --out {rel(POD_OUT)} "
                     f"--variants {','.join(B4D_VARIANTS)} --force-stub"}],
         "outputs": [_ws(POD_OUT) + "/"],
         "verify": [_ws(POD_OUT / "body4d_pod.json")] + [_ws(POD_OUT / f"{k}_{v}.npz") for v in B4D_VARIANTS
                                                         for k in ("mhr", "smplx")],
         "after_download": ["python -m dancecap.mono4d.cand_sam3d_body import --session <sid>"],
         "notes": ["SAM-Body4D (gaomingqi/sam-body4d, MIT) with SAM 3 BYPASSED: our SAM 2.1 lead/follow masks as its "
                   "masklets, our K as cam_int, its own OfflineApp.on_4d_generation (fork of 3DB with Kalman/EMA "
                   "temporal smoothing and frozen per-identity shape; Diffusion-VAS amodal + RGB completion of occluded "
                   "frames in variant body4d). Public checkpoints (Diffusion-VAS, Depth-Anything-V2-L) and the gated "
                   "3DB checkpoint are downloaded on the pod and never leave it."]}
    p = JOB_DIR / "pod_job_body4d.json"
    p.write_text(json.dumps(d, indent=1), newline="\n")
    return p


# ------------------------------------------------------------------------------------------------ import


def _session_meta(sid: str | None) -> dict:
    root = PACKAGE_ROOT / "work" / "runpod" / "sessions"
    if sid is None:
        cands = sorted((p for p in root.glob("*") if (p / "result.json").exists()
                        and '"cand_sam3d_body"' in (p / "result.json").read_text(errors="ignore")),
                       key=lambda p: p.stat().st_mtime)
        if not cands:
            return {}
        sid = cands[-1].name
    p = root / sid / "result.json"
    if not p.exists():
        return {"session": sid}
    r = json.loads(p.read_text())
    pod = r.get("pod") or {}
    cloud = None
    try:
        led = json.loads((PACKAGE_ROOT / "work" / "runpod" / "ledger.json").read_text())
        rows = led if isinstance(led, list) else led.get("rows", led.get("pods", []))
        cloud = next((x.get("cloud") for x in rows if x.get("pod_id") == pod.get("id")), None)
    except Exception:  # noqa: BLE001
        pass
    jb = (r.get("jobs") or {}).get("cand_sam3d_body") or {}
    return {"session": sid, "gpu": pod.get("gpu"), "cloud": cloud, "cost_per_hr": pod.get("cost_per_hr"),
            "cost_usd": pod.get("ledger_cost"), "pod_minutes": round((pod.get("lifetime_s") or 0) / 60, 1),
            "job_status": jb.get("status"), "job_seconds": jb.get("seconds")}


def _qa_sheet(meta: dict, J: dict, K: np.ndarray, path: Path, picks: int = 6) -> None:
    """Projected SMPL-X body joints (lead orange, follow cyan) on a few frames of phone 06."""
    import cv2
    rows = meta["frames"]
    idx = np.linspace(0, len(rows) - 1, picks).round().astype(int)
    bones = [(0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8), (6, 9), (7, 10), (8, 11), (9, 12), (12, 15),
             (9, 13), (9, 14), (13, 16), (14, 17), (16, 18), (17, 19), (18, 20), (19, 21)]
    tiles = []
    for k in idx:
        img = cv2.imread(str(INPUT / "frames" / rows[k]["file"]))
        for role, col in (("lead", (0, 140, 255)), ("follow", (255, 220, 0))):
            P = J[role][k, :22]
            if not np.isfinite(P).all() or (P[:, 2] <= 0).any():
                continue
            uv = (K @ (P / P[:, 2:3]).T).T[:, :2]
            for a, b in bones:
                cv2.line(img, tuple(int(x) for x in uv[a]), tuple(int(x) for x in uv[b]), col, 3, cv2.LINE_AA)
        bx = np.array([b for b in rows[k]["mask_boxes"].values() if b], float)
        cx, cy = (bx[:, 0].min() + bx[:, 2].max()) / 2, (bx[:, 1].min() + bx[:, 3].max()) / 2
        h = max(bx[:, 3].max() - bx[:, 1].min(), 1.2 * (bx[:, 2].max() - bx[:, 0].min())) * 1.15
        w = h * 350 / 550
        x0, y0 = int(max(0, cx - w / 2)), int(max(0, cy - h / 2))
        x1, y1 = int(min(img.shape[1], x0 + w)), int(min(img.shape[0], y0 + h))
        tile = img[y0:y1, x0:x1].copy()
        cv2.putText(tile, f"t {rows[k]['t']:.2f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        tiles.append(cv2.resize(tile, (350, 550)))
    cv2.imwrite(str(path), np.concatenate(tiles, axis=1), [cv2.IMWRITE_JPEG_QUALITY, 88])


def _score_raw_time(t_raw: np.ndarray, J: dict) -> dict:
    """Diagnostic: the same joints scored on phone 06's RAW frame times (no sync delta), as the harness would see them
    with the camera pose looked up at the raw time; nothing is saved."""
    from .. import mamma_dyn as md
    from .holdout import load_fit
    tr = md.load_tracks(WORK, [CAM])[CAM]
    k = np.array([int(np.argmin(np.abs(tr.t - x))) for x in t_raw])
    out = {}
    for role in C.ROLES:
        r = load_fit(WORK / "smplx", role)
        Jw = C.to_world(J[role], tr.R[k], tr.tvec[k])
        s = C.score_arrays(t_raw, Jw, "world", r["t"], r["joints"], WINDOW)
        out[role] = {m: (s[m] or {}).get("median") for m in ("pa22_mm", "rr_mm", "pelvis_mm")}
    return out


def do_import(sid: str | None = None, score: bool = True) -> dict:
    from .. import mamma_dyn as md
    meta = json.loads((INPUT / "meta.json").read_text())
    pod = json.loads((POD_OUT / "sam3db_pod.json").read_text())
    ses = _session_meta(sid)
    rows = meta["frames"]
    t_raw = np.array([r["t"] for r in rows])
    off = md.read_sync_offsets(WORK).get(CAM)
    t = C.frame_to_ref(TAKE, CAM, t_raw)  # reference seconds (t + sync delta): the reference fit's and load_tracks' clock
    K = np.array(rows[0]["K"], float)
    out = {"session": ses, "sync_offset_06": off, "candidates": {}}
    todo = [(v, VARIANTS[v], f"smplx_{v}.npz", "per-frame identity") for v in VARIANTS]
    if (POD_OUT / "smplx_mask255_1id.npz").exists():
        todo.append(("mask255", "sam3d_body_1id", "smplx_mask255_1id.npz", "single identity per dancer"))
    todo += [(v, B4D_VARIANTS[v], f"smplx_{v}.npz", "per-frame identity") for v in B4D_VARIANTS
             if (POD_OUT / f"smplx_{v}.npz").exists()]
    b4d = json.loads((POD_OUT / "body4d_pod.json").read_text()) if (POD_OUT / "body4d_pod.json").exists() else {}
    for var, name, fn, conv_mode in todo:
        p = POD_OUT / fn
        if not p.exists():
            log(f"{fn} missing: {name} not imported")
            out["candidates"][name] = {"missing": fn}
            continue
        z = np.load(p, allow_pickle=True)
        if list(z["frame_i"]) != [r["i"] for r in rows]:
            raise SystemExit(f"{fn}: frame indices differ from input/meta.json")
        tracks = {role: np.asarray(z[f"{role}_joints"], float) for role in C.ROLES}   # (F, 55, 3) camera frame
        ass = C.assign_roles(tracks, TAKE, CAM, K, t_raw)  # roles.json boxes are on the raw frame times
        agree = ass.get("lead") == "lead" and ass.get("follow") == "follow"
        J = {r: tracks[ass[r]] for r in C.ROLES if r in ass} if ass.get("lead") and ass.get("follow") else tracks
        err = {r: float(np.nanmedian(z[f"{r}_fit_err_cm"])) for r in C.ROLES if f"{r}_fit_err_cm" in z.files}
        vinfo = pod.get("variants", {}).get(var, {}) or b4d.get(var, {})
        cinfo = pod.get("convert", {}).get(fn.replace("smplx_", "").replace(".npz", ""), {})
        if var in B4D_VARIANTS:
            meth = (f"SAM-Body4D (gaomingqi/sam-body4d @ {B4D_COMMIT[:7]}, its fork of SAM 3D Body, {HF_REPO}) with "
                    f"SAM 3 bypassed: our SAM 2.1 lead/follow masks as masklets (0/255, the app's recipe), our K as "
                    f"cam_int; its Kalman / EMA temporal smoothing; MHR shape + scale frozen at each identity's first "
                    f"frame (per identity - upstream shares the first identity's with the others); Diffusion-VAS "
                    f"occlusion completion {'ON' if var == 'body4d' else 'OFF'}")
        else:
            meth = f"SAM 3D Body ({HF_REPO}) prompted per dancer with our SAM 2.1 mask + mask box, our K; variant {var}"
        m = {"method": f"{meth}; MHR -> SMPL-X by the official converter (pytorch, {conv_mode})",
             "version": {"sam_3d_body": S3DB_COMMIT, "hf_revision": HF_REV, "dinov3": DINOV3_COMMIT, "mhr": MHR_COMMIT,
                         "mhr_assets": MHR_ASSETS_RELEASE,
                         **({"sam_body4d": B4D_COMMIT, **B4D_REVS} if var in B4D_VARIANTS else {})},
             "licence": LICENCE, "input_cam": CAM, "variant": var, "conversion": conv_mode,
             "converter_mode": cinfo.get("mhr_model") or pod.get("convert", {}).get("mhr_model"),
             "converter_where": pod.get("convert", {}).get("where", "pod"),
             "converter_checks": pod.get("convert", {}).get("checks"),
             "conversion_fit_err_cm_median": err, "roles_from": "prompt identity (01 lead, 02 follow), checked by "
                                                                 "candidates.assign_roles",
             "assign_roles": ass, "assign_roles_agrees_with_prompts": agree,
             "gpu": ses.get("gpu") or (pod.get("model") or {}).get("gpu"), "cloud": ses.get("cloud"),
             "cost_usd": ses.get("cost_usd"), "wall_min": ses.get("pod_minutes"), "session": ses.get("session"),
             "pod_infer_s": vinfo.get("seconds") or vinfo.get("seconds_4d"), "time": f"t = phone 06 frame time + its "
             f"sync delta ({(off or {}).get('delta', 0) * 1e3:.1f} ms, sync/offsets.json, candidates.frame_to_ref)",
             "source": str(p), "notes": ("per person; temporal smoothing within the app's frame batches; no relative "
                                         "placement of the two dancers" if var in B4D_VARIANTS else
                                         "per frame and per person (no temporal model, no relative placement)")
                                        + "; cost/wall are the whole pod session shared by all sam3d_body variants"}
        if var in B4D_VARIANTS:
            m["body4d"] = {k: vinfo.get(k) for k in ("completion", "batch_size", "occluded_frames", "completed_segments",
                                                      "peak_gb", "app_inputs")}
        raw = _score_raw_time(t_raw, J)
        m["score_on_raw_frame_time"] = raw
        for role in C.ROLES:
            C.save(TAKE, name, role, t, J[role], "camera", m)
        _qa_sheet(meta, J, K, C.cand_dir(TAKE, name) / "qa_joints.jpg")
        res = C.score(TAKE, name, WINDOW) if score else None
        out["candidates"][name] = {"assign_roles": ass, "agrees": agree, "fit_err_cm": err,
                                   "score": res["roles"] if res else None, "score_on_raw_frame_time": raw}
        log(f"{name}: roles {ass.get('lead')}/{ass.get('follow')} (agree {agree}); fit err {err}")
    (OUT / "import.json").write_text(json.dumps(out, indent=1, default=str))
    return out


KEY_FILE = HF_KEY_FILE


def check_access(key_file: str = KEY_FILE) -> dict:
    """Offline-cheap gate before a pod: which HF account the key file's token belongs to and whether it can read the
    gated repos (HEAD request on one small file each; nothing is downloaded, the token is never printed)."""
    from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url
    tok = Path(key_file).read_text().strip()
    out = {}
    try:
        w = HfApi().whoami(token=tok)
        at = (w.get("auth") or {}).get("accessToken") or {}
        out["account"], out["token_role"] = w.get("name"), at.get("role")
        out["can_read_gated_repos"] = (at.get("fineGrained") or {}).get("canReadGatedRepos")
    except Exception as e:  # noqa: BLE001
        out["whoami_error"] = f"{type(e).__name__}: {str(e).replace(tok, '<token>')[:200]}"
    # list_repo_files / model cards succeed WITHOUT access: only a HEAD on a gated file proves it (the checkpoint itself)
    for repo, f, rev in ((HF_REPO, "model.ckpt", HF_REV), (HF_REPO, "assets/mhr_model.pt", HF_REV),
                         ("facebook/sam3", "config.json", None)):
        key = f"{repo}/{f}"
        try:
            m = get_hf_file_metadata(hf_hub_url(repo, f, revision=rev), token=tok)
            out[key] = f"access ok ({m.size} bytes, revision {str(m.commit_hash)[:7]})"
        except Exception as e:  # noqa: BLE001
            msg = str(e).replace(tok, "<token>")
            out[key] = ("NOT AUTHORIZED (access request not approved for this account)"
                        if ("not in the authorized list" in msg or "403" in msg) else f"{type(e).__name__}: {msg[:200]}")
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m dancecap.mono4d.cand_sam3d_body")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prep")
    ca = sub.add_parser("check-access")
    ca.add_argument("--key-file", default=KEY_FILE)
    j = sub.add_parser("job")
    j.add_argument("--max-minutes", type=float, default=25)
    j.add_argument("--limit", type=int, default=0)
    jb = sub.add_parser("job-body4d")
    jb.add_argument("--max-minutes", type=float, default=42)
    jb.add_argument("--limit", type=int, default=0)
    im = sub.add_parser("import")
    im.add_argument("--session")
    im.add_argument("--no-score", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "prep":
        print(json.dumps(prep(), indent=1))
    elif a.cmd == "check-access":
        print(json.dumps(check_access(a.key_file), indent=1))
    elif a.cmd == "job":
        print(job(a.max_minutes, a.limit))
    elif a.cmd == "job-body4d":
        print(job_body4d(a.max_minutes, a.limit))
    else:
        print(json.dumps(do_import(a.session, not a.no_score), indent=1, default=str))


if __name__ == "__main__":
    main()
