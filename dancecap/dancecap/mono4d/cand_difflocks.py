r"""mono4d track "difflocks" (2026-10-19): DiffLocks strand hair for the follower from ONE frontal phone frame.

DiffLocks (github.com/Meshcapade/difflocks @ fcc7374, Rosu et al., CVPR 2025; code and weights under the MPI
non-commercial research licence of difflocks.is.tue.mpg.de, downloaded by the owner to HeadMovement/models/difflocks)
turns one frontal RGB image into ~100K hair strands (256 points each) on the SMPL-X template head: MediaPipe face crop
(2.8 x the face, 770 px) -> DINOv2-L/14-reg features -> a conditional hourglass diffusion transformer (k-diffusion,
NATTEN neighbourhood + flash-attention global/cross attention, 100 DPM++ 2M SDE steps) that samples a 256 x 256 x 64
scalp texture of strand latents + a density map -> strand VAE decoder; rgb2material predicts melanin / redness.

Licence: the checkpoints may go to OUR pod for this run and to work/, never into a repo or a shared bundle. Strands,
renders and crops of the dancer stay in work/ (git-ignored).

Frames (prep, local, light): every phone frame where the 8-phone reference fit (smplx/follow.npz) has her FACE towards
the phone (head forward = global orientation chain 0-3-6-9-12-15 applied to SMPL-X +Z), the head large in frame, the
head and torso calm for the 0.7 s before the frame (hair at rest), the face inside her own SAM mask and not behind the
lead, sharp. Phone 06 (the one-phone scenario) first. Each chosen moment gives a square head + shoulders crop of the
canonical frame (0.8 m at the head's depth, 1024 px) in three variants: raw; fg (pixels outside her dilated body +
hair masks black, like DiffLocks' own padding); gray (the same with a mid-gray background: black backgrounds pushed
rgb2material to blonde / white in run 1). DiffLocks re-crops it itself with MediaPipe (2.8 x the face); its face
choice (the LARGEST detection at confidence >= 0.1) is replaced on the pod by the detection nearest to her fitted face
centre (run 1, session 20261019-165713: on phone 02 DiffLocks grew hair for the 'Dream' print on her T-shirt).

Pod (remote/cand_difflocks/: install.sh, difflocks_pod.py, blender_export.sh; pod_job.json written by `job`):
    DiffLocks at the pinned commit in a venv on the pod's torch 2.5.1/cu124 (bootstrap 'base'); flash-attn 2.7.4.post1
    prebuilt wheel (cu12, torch 2.5, cxx11abi FALSE); NATTEN has no binary wheels (PyPI: sdists only; the shi-labs
    wheel server's TLS certificate had expired on 2026-10-19), so neighbourhood attention runs through an exact pure
    PyTorch implementation of NATTEN's na2d_qk / na2d_av (same clamped 7 x 7 window rule; checked against a brute-force
    loop on the pod before inference). MediaPipe runs on the CPU delegate (one image). Only 'model_ema' of the 9.7 GB
    training checkpoint is uploaded (`strip-ckpt`: 2.43 GB; inference_difflocks.py reads only ckpt['model_ema']).
    Then `render` (system python + this module): orthographic front / sides / back / top renders of the strands on the
    template body and overlays of the strands - rigidly attached to her fitted head at that moment - into 2-3 other
    phones, with silhouette IoU against the phones' hair masks; optional Blender 4.1.1 .blend/.abc export.
Import (local): verification, stats, contact sheet input crop | DiffLocks | current groom (Unity) | real phones.

Frames: DiffLocks' output frame is the SMPL-X neutral template (zero betas, zero pose; its smplx_base.ply head equals
v_template to 0 mm), metres, x = her left, y up, z forward. Alignment to the world at a moment: similarity (Umeyama)
from the template head vertices (FLAME region above the mouth) to her posed SMPL-X head (locked-head model, her 16
betas, reference pose at the moment).

    python -m dancecap.mono4d.cand_difflocks strip-ckpt   # model_ema only -> work/difflocks_ckpt/ (licensed, work/)
    python -m dancecap.mono4d.cand_difflocks prep         # scan, picks, crops, alignment -> mono4d/difflocks/input/
    python -m dancecap.mono4d.cand_difflocks job          # remote/cand_difflocks/pod_job.json
    (DANCECAP_RUNPOD_CAP=... python -m dancecap remote run --jobs ../dancecap/remote/cand_difflocks/pod_job.json
     --max-cost 3 --gpus ...)
    python -m dancecap.mono4d.cand_difflocks render --dir <pod_out> --mesh <smplx_base.ply>   # pod (or local)
    python -m dancecap.mono4d.cand_difflocks import [--session SID]
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

NAME = "cand_difflocks"
TAKE = "larissa_kadu_38"
ROLE, PID, LEAD_PID = "follow", "02", "01"
PREF_CAM = "06"
CAMS = ("01", "02", "03", "04", "05", "06", "07", "09")
WORK = PACKAGE_ROOT / "work" / TAKE
OUT = WORK / "mono4d" / "difflocks"
INPUT = OUT / "input"
POD_OUT = OUT / "pod_out"
JOB_DIR = PACKAGE_ROOT / "remote" / "cand_difflocks"
CKPT_SRC = WORKSPACE / "models" / "difflocks" / "checkpoints"
CKPT_EMA = PACKAGE_ROOT / "work" / "difflocks_ckpt" / "scalp_v9_40k_06730000_ema.pth"
POD_CKPT = "models/difflocks/checkpoints"  # workspace-relative on the pod (deleted by the job's cleanup command)
DIFFLOCKS_COMMIT = "fcc73747dc60320c30228b6711000a53fc0c9d84"
DINOV2_COMMIT = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
FLASH_ATTN_WHEEL = ("https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/"
                    "flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl")
PORTRAIT = WORKSPACE / "dances" / "Larissa-Kadu-demo" / "larissa.jpg"
GROOM_SHOTS = WORK / "review" / "face_shape_v4" / "ship" / "unity_after_hair" / "shots"  # 04 demo, v2 groom, 2026-10-08
HEAD_J, NECK_J = 15, 12
HEAD_CHAIN = (0, 3, 6, 9, 12, 15)
TORSO_CHAIN = (0, 3, 6, 9)
CROP_M = 0.80          # crop side at the head's depth
CROP_PX = 1024
CALM_WIN = (-0.7, 0.1)  # s around the frame: hair at rest needs a calm head before the frame
TEMPLATE_CROWN_Y = 0.4193  # SMPL-X neutral template, top of the head (m)
# HAIR_REFERENCE.md (larissa_kadu, 'world' = that take's world, true = world x 0.959; larissa_kadu_38's world is metric)
REF = {"tip_drop_true_m": 0.525, "tip_drop_longest_q_true_m": 0.554, "back_strand_true_m": 0.597,
       "tip_drop_frac_height": 0.313, "back_strand_frac_height": 0.356,
       "width_ear_true_m": 0.239 * 0.959, "width_shoulder_true_m": 0.280 * 0.959,
       "width_profile_world_m": [0.18, 0.22, 0.24, 0.26, 0.26, 0.28, 0.28, 0.27, 0.25, 0.22, 0.15, 0.11],
       "true_per_world": 0.959, "stature_true_m": 1.651, "colour_video_p50": "#732C24", "colour_portrait_mid": "#63131F",
       "part": "deep side part on her LEFT, swept to her right and back (portrait); not resolvable in the video",
       "texture": "portrait sleek straight; video soft loose waves, fuller"}
LICENCE = ("DiffLocks code + checkpoints: MPI non-commercial research licence (difflocks.is.tue.mpg.de); checkpoints "
           "only on our pod for the run and in work/, never in a repo or a shared bundle; DINOv2 weights Apache-2.0; "
           "outputs of the dancer stay in work/ (research use, consented demo)")


def log(msg: str) -> None:
    print(f"[{NAME}] {msg}", flush=True)


def _ws(p: Path) -> str:
    return str(Path(p).resolve().relative_to(WORKSPACE.resolve())).replace("\\", "/")


def _dc(p: Path) -> str:
    """Path relative to dancecap/ (the pod commands' cwd)."""
    return str(Path(p).resolve().relative_to(PACKAGE_ROOT.resolve())).replace("\\", "/")


# ================================================================================================ checkpoint strip


def strip_ckpt(src: Path = CKPT_SRC / "difflocks_diffusion" / "scalp_v9_40k_06730000.pth", dst: Path = CKPT_EMA) -> dict:
    """Keep only what DiffLocksInference reads (ckpt['model_ema']; inference/img2hair.py) + step/config for the record.
    Restricted unpickler (weights_only) + mmap: the 9.7 GB file is never fully read into memory."""
    import torch
    t0 = time.time()
    ck = torch.load(src, map_location="cpu", mmap=True, weights_only=True)
    keep = {"model_ema": ck["model_ema"], "step": ck.get("step"), "epoch": ck.get("epoch"), "config": ck.get("config")}
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".part")
    torch.save(keep, tmp)
    tmp.replace(dst)
    chk = torch.load(dst, map_location="cpu", mmap=True, weights_only=True)
    same = all(torch.equal(chk["model_ema"][k], v) for k, v in ck["model_ema"].items())
    info = {"src": str(src), "src_bytes": src.stat().st_size, "dst": str(dst), "dst_bytes": dst.stat().st_size,
            "tensors": len(ck["model_ema"]), "params": int(sum(v.numel() for v in ck["model_ema"].values())),
            "identical": bool(same), "step": ck.get("step"), "seconds": round(time.time() - t0, 1)}
    (dst.parent / "strip.json").write_text(json.dumps(info, indent=1))
    log(f"model_ema {info['params'] / 1e6:.0f} M params, {info['dst_bytes'] / 1e9:.2f} GB (of {info['src_bytes'] / 1e9:.2f}), "
        f"identical {same}")
    return info


# ================================================================================================ geometry


ROT_KEYS = (("global_orient", 1), ("body_pose", 21), ("jaw_pose", 1), ("leye_pose", 1), ("reye_pose", 1),
            ("left_hand_pose", 15), ("right_hand_pose", 15))


def _load_fit(role: str = ROLE) -> dict:
    with np.load(WORK / "smplx" / f"{role}.npz", allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def interp_params(d: dict, t: float) -> dict:
    """Reference SMPL-X parameters at time t: per-joint slerp of the rotations, linear translation and joints."""
    from scipy.spatial.transform import Rotation as Rot
    T = np.asarray(d["t"], float)
    k = int(np.clip(np.searchsorted(T, t) - 1, 0, len(T) - 2))
    a = float(np.clip((t - T[k]) / (T[k + 1] - T[k]), 0.0, 1.0))
    out = {"t": float(t), "k": k, "a": a}
    for name, n in ROT_KEYS:
        r0 = Rot.from_rotvec(np.asarray(d[name][k], float).reshape(n, 3))
        r1 = Rot.from_rotvec(np.asarray(d[name][k + 1], float).reshape(n, 3))
        out[name] = (r0 * Rot.from_rotvec((r0.inv() * r1).as_rotvec() * a)).as_rotvec().reshape(-1)
    out["transl"] = (1 - a) * np.asarray(d["transl"][k], float) + a * np.asarray(d["transl"][k + 1], float)
    out["joints"] = (1 - a) * np.asarray(d["joints"][k], float) + a * np.asarray(d["joints"][k + 1], float)
    return out


def chain_rotation(global_orient: np.ndarray, body_pose: np.ndarray, chain=HEAD_CHAIN):
    """Global rotation of the last joint of a pelvis-rooted chain (SMPL-X body_pose index = joint - 1). Vectorised."""
    from scipy.spatial.transform import Rotation as Rot
    go = np.asarray(global_orient, float).reshape(-1, 3)
    bp = np.asarray(body_pose, float).reshape(len(go), 21, 3)
    R = Rot.from_rotvec(go)
    for j in chain[1:]:
        R = R * Rot.from_rotvec(bp[:, j - 1])
    return R


def project(P: np.ndarray, K, R, t, k1: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Pinhole + COLMAP SIMPLE_RADIAL (mamma_dyn.project + distort_simple_radial)."""
    Xc = np.asarray(P, float) @ np.asarray(R, float).T + np.asarray(t, float)
    z = Xc[:, 2]
    K = np.asarray(K, float)
    x, y = Xc[:, 0] / np.maximum(z, 1e-6), Xc[:, 1] / np.maximum(z, 1e-6)
    if k1:
        r2 = x * x + y * y
        x, y = x * (1 + k1 * r2), y * (1 + k1 * r2)
    return np.stack([K[0, 0] * x + K[0, 2], K[1, 1] * y + K[1, 2]], -1), z


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, float]:
    """Similarity dst ~ s R src + T; returns (s, R, T, rms)."""
    ms, md_ = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md_
    U, S, Vt = np.linalg.svd(b.T @ a / len(src))
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = float(np.trace(np.diag(S) @ D) / (a ** 2).sum(1).mean())
    T = md_ - s * R @ ms
    rms = float(np.sqrt((((s * src @ R.T + T) - dst) ** 2).sum(1).mean()))
    return s, R, T, rms


def _mask(cam: str, i: int, pid: str, kind: str = "masks") -> np.ndarray | None:
    m = cv2.imread(str(WORK / "dancers" / cam / kind / f"mask_{int(i):04d}_{pid}.png"), cv2.IMREAD_GRAYSCALE)
    return None if m is None else m > 127


def _disc_frac(m: np.ndarray | None, c: np.ndarray, r: float) -> float:
    if m is None:
        return 0.0
    H, W = m.shape
    x0, x1 = int(max(0, c[0] - r)), int(min(W, c[0] + r + 1))
    y0, y1 = int(max(0, c[1] - r)), int(min(H, c[1] + r + 1))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = (xx - c[0]) ** 2 + (yy - c[1]) ** 2 <= r * r
    return float(m[y0:y1, x0:x1][d].mean()) if d.any() else 0.0


# ================================================================================================ prep: scan + picks


class _Motion:
    """Per fit-grid frame: head pose and the calm measure (max head / torso angular speed and head speed in the window
    CALM_WIN around the frame)."""

    def __init__(self, fit: dict):
        self.T = np.asarray(fit["t"], float)
        self.J = np.asarray(fit["joints"], float)
        Rh = chain_rotation(fit["global_orient"], fit["body_pose"], HEAD_CHAIN)
        Rt = chain_rotation(fit["global_orient"], fit["body_pose"], TORSO_CHAIN)
        self.fwd, self.up, self.left = Rh.apply([0, 0, 1.0]), Rh.apply([0, 1.0, 0]), Rh.apply([1.0, 0, 0])
        dt = np.diff(self.T)
        self.w_head = np.degrees((Rh[:-1].inv() * Rh[1:]).magnitude()) / dt
        self.w_torso = np.degrees((Rt[:-1].inv() * Rt[1:]).magnitude()) / dt
        self.v_head = np.linalg.norm(np.diff(self.J[:, HEAD_J], axis=0), axis=1) / dt

    def k(self, t: float) -> int:
        return int(np.argmin(np.abs(self.T - t)))

    def calm(self, t: float) -> dict:
        lo, hi = np.searchsorted(self.T, t + CALM_WIN[0]), np.searchsorted(self.T, t + CALM_WIN[1])
        lo, hi = int(max(0, lo)), int(min(len(self.w_head), max(hi, lo + 1)))
        return {"w_head": float(self.w_head[lo:hi].max()), "w_torso": float(self.w_torso[lo:hi].max()),
                "v_head": float(self.v_head[lo:hi].max())}


def _view(mo: _Motion, k: int, C: np.ndarray) -> dict:
    head = mo.J[k, HEAD_J]
    d = C - head
    dist = float(np.linalg.norm(d))
    d = d / dist
    facing = float(np.degrees(np.arccos(np.clip(d @ mo.fwd[k], -1, 1))))
    az = float(np.degrees(math.atan2(d @ mo.left[k], d @ mo.fwd[k])))  # 0 front, +90 her left, +-180 behind
    el = float(np.degrees(math.asin(np.clip(d @ mo.up[k], -1, 1))))
    return {"facing_deg": round(facing, 1), "az_deg": round(az, 1), "el_deg": round(el, 1), "dist_m": round(dist, 3)}


def _face_center(mo: _Motion, k: int) -> np.ndarray:
    return mo.J[k, HEAD_J] + 0.03 * mo.up[k] + 0.08 * mo.fwd[k]


def scan(cams=CAMS) -> list[dict]:
    from .. import mamma_dyn as md
    fit = _load_fit()
    mo = _Motion(fit)
    tracks = md.load_tracks(WORK, list(cams))
    rows = []
    for cam in cams:
        tr = tracks[cam]
        stride = 1 if cam == PREF_CAM else (4 if len(tr.t) > 2000 else 2)
        cam_rows = []
        for i in range(0, len(tr.t), stride):
            t = float(tr.t[i])
            if not tr.ok[i] or t < mo.T[0] or t > mo.T[-1]:
                continue
            k = mo.k(t)
            if abs(mo.T[k] - t) > 0.025:
                continue
            C = -tr.R[i].T @ tr.tvec[i]
            v = _view(mo, k, C)
            if v["facing_deg"] > 40:
                continue
            fc = _face_center(mo, k)
            uv, z = project(fc[None], tr.K[i], tr.R[i], tr.tvec[i], float(tr.k1[i]))
            if z[0] <= 0.3:
                continue
            s_px = float(tr.K[i][0, 0] * 0.25 / z[0])
            u, vv = uv[0]
            if s_px < 18 or not (0.3 * s_px <= u <= tr.width - 0.3 * s_px and 0.3 * s_px <= vv <= tr.height - 0.3 * s_px):
                continue
            cam_rows.append({"cam": cam, "i": i, "t": round(t, 4), "k": k, "uv": [float(u), float(vv)], "z": float(z[0]),
                             "s_px": round(s_px, 1), **v, **mo.calm(t)})
        # image evidence only for the geometric survivors
        for r in cam_rows:
            m2, m1 = _mask(cam, r["i"], PID), _mask(cam, r["i"], LEAD_PID)
            rad = 0.42 * r["s_px"]
            r["frac_follow"] = round(_disc_frac(m2, np.array(r["uv"]), rad), 3)
            r["frac_lead"] = round(_disc_frac(m1, np.array(r["uv"]), rad), 3)
            g = cv2.imread(tr.files[r["i"]], cv2.IMREAD_GRAYSCALE)
            h = int(0.6 * r["s_px"])
            u, vv = int(r["uv"][0]), int(r["uv"][1])
            crop = g[max(0, vv - h):vv + h, max(0, u - h):u + h]
            r["sharp"] = float(cv2.Laplacian(cv2.resize(crop, (96, 96), interpolation=cv2.INTER_AREA),
                                             cv2.CV_32F).var()) if crop.size else 0.0
        if cam_rows:
            med = float(np.median([r["sharp"] for r in cam_rows])) or 1.0
            for r in cam_rows:
                f_face = float(np.clip((45.0 - r["facing_deg"]) / 25.0, 0, 1))
                f_size = min(1.0, r["s_px"] / 160.0) ** 0.75
                w = max(r["w_head"], 0.5 * r["w_torso"])
                f_rest = math.exp(-(w / 150.0) ** 2 - (r["v_head"] / 0.8) ** 2)
                f_vis = float(np.clip((r["frac_follow"] - 0.55) / 0.35, 0, 1) * np.clip(1 - 3 * r["frac_lead"], 0, 1))
                f_sharp = min(1.0, r["sharp"] / med) ** 0.5
                r.update(f_face=round(f_face, 3), f_size=round(f_size, 3), f_rest=round(f_rest, 3),
                         f_vis=round(f_vis, 3), f_sharp=round(f_sharp, 3),
                         q=round(f_face * f_size * f_rest * f_vis * f_sharp, 4))
        log(f"cam {cam}: {len(cam_rows)} frontal frames (stride {stride}), best q "
            f"{max([r['q'] for r in cam_rows], default=0):.3f}")
        rows += cam_rows
    return rows


def _greedy(rows: list[dict], n: int, min_dt: float) -> list[dict]:
    out = []
    for r in sorted(rows, key=lambda r: -r["q"]):
        if r["q"] <= 0:
            break
        if all(abs(r["t"] - p["t"]) >= min_dt for p in out):
            out.append(r)
        if len(out) == n:
            break
    return out


def _square_crop(img: np.ndarray, c: np.ndarray, side: float, out: int = CROP_PX) -> np.ndarray:
    """Square crop centred on c (zero padded outside the image), resized to out x out."""
    h = side / 2.0
    x0, y0 = int(round(c[0] - h)), int(round(c[1] - h))
    S = int(round(side))
    H, W = img.shape[:2]
    pad = np.zeros((S, S) + img.shape[2:], img.dtype)
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(W, x0 + S), min(H, y0 + S)
    if sx1 > sx0 and sy1 > sy0:
        pad[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    interp = cv2.INTER_CUBIC if S < out else cv2.INTER_AREA
    return cv2.resize(pad, (out, out), interpolation=interp)


def _crop_box(c: np.ndarray, side: float) -> list[int]:
    return [int(round(c[0] - side / 2)), int(round(c[1] - side / 2)), int(round(side))]


def _smplx_model(batch: int = 1):
    import smplx

    from ..mamma_dyn import SMPLX_MODELS
    return smplx.create(str(SMPLX_MODELS), model_type="smplx", gender="neutral", ext="npz", use_pca=False,
                        flat_hand_mean=True, num_betas=16, num_expression_coeffs=10, batch_size=batch)


def _posed(m, betas: np.ndarray, p: dict | None) -> np.ndarray:
    import torch
    f = lambda a: torch.tensor(np.asarray(a, np.float64).reshape(1, -1), dtype=torch.float32)  # noqa: E731
    z3 = np.zeros(3)
    kw = {"betas": f(betas)}
    if p is not None:
        kw.update(global_orient=f(p["global_orient"]), body_pose=f(p["body_pose"]), jaw_pose=f(p["jaw_pose"]),
                  leye_pose=f(z3), reye_pose=f(z3), left_hand_pose=f(p["left_hand_pose"]),
                  right_hand_pose=f(p["right_hand_pose"]), transl=f(p["transl"]))
    with torch.no_grad():
        return m(**kw).vertices[0].numpy().astype(np.float64)


def _template_and_head_ids(m) -> tuple[np.ndarray, np.ndarray, dict]:
    """DiffLocks' frame = SMPLX_NEUTRAL v_template (checked against smplx_base.ply: 0 mm on the head)."""
    tmpl = np.load(WORKSPACE / "smplx" / "SMPLX_NEUTRAL.npz", allow_pickle=True)["v_template"].astype(np.float64)
    zero = _posed(m, np.zeros(16), None)  # the locked-head model at zero shape/pose
    fl = np.load(WORKSPACE / "models" / "flame" / "SMPL-X__FLAME_vertex_ids.npy")
    ids = fl[tmpl[fl, 1] > 0.29]  # skull + upper face (no jaw / neck: jaw pose and the neck seam move them)
    info = {"template_vs_locked_head_max_mm": round(float(np.abs(tmpl[fl] - zero[fl]).max() * 1000), 3),
            "n_head_verts": int(len(ids))}
    return tmpl, ids, info


def _compare_cams(mo: _Motion, tracks: dict, t: float, own: str, k: int) -> list[dict]:
    """Other phones at the same moment, picked for view diversity (back and sides first) and visibility."""
    rows = []
    for cam, tr in tracks.items():
        j = int(np.argmin(np.abs(tr.t - t)))
        if abs(tr.t[j] - t) > 0.03 or not tr.ok[j]:
            continue
        C = -tr.R[j].T @ tr.tvec[j]
        v = _view(mo, k, C)
        fc = mo.J[k, HEAD_J] + 0.05 * mo.up[k]
        uv, z = project(fc[None], tr.K[j], tr.R[j], tr.tvec[j], float(tr.k1[j]))
        if z[0] <= 0.3:
            continue
        s_px = float(tr.K[j][0, 0] * 0.25 / z[0])
        u, vv = uv[0]
        if not (0 <= u < tr.width and 0 <= vv < tr.height):
            continue
        ff = _disc_frac(_mask(cam, j, PID), uv[0], 0.5 * s_px)
        hh = _disc_frac(_mask(cam, j, PID, "hair"), uv[0], 0.8 * s_px)
        fl = _disc_frac(_mask(cam, j, LEAD_PID), uv[0], 0.5 * s_px)
        rows.append({"cam": cam, "j": j, "t": round(float(tr.t[j]), 4), "uv": [float(u), float(vv)], "z": float(z[0]),
                     "s_px": round(s_px, 1), "frac_follow": round(ff, 3), "frac_hair": round(hh, 3),
                     "frac_lead": round(fl, 3), "own": cam == own, **v})
    ok = [r for r in rows if not r["own"] and r["frac_follow"] >= 0.35 and r["frac_lead"] <= 0.35]
    picked = []
    while ok and len(picked) < 3:
        def gain(r):
            ang = [abs((r["az_deg"] - p["az_deg"] + 180) % 360 - 180) for p in picked] or [abs(r["az_deg"])]
            return min(ang) / 180.0 * min(1.0, r["s_px"] / 40.0) ** 0.5 * (0.5 + r["frac_follow"])
        b = max(ok, key=gain)
        picked.append(b)
        ok.remove(b)
    own_row = [r for r in rows if r["own"]]
    return own_row + picked


def prep(n_moments: int = 3, with_portrait: bool = False) -> dict:
    from .. import mamma_dyn as md
    t0 = time.time()
    fit = _load_fit()
    mo = _Motion(fit)
    rows = scan()
    tracks = md.load_tracks(WORK, list(CAMS))
    if INPUT.exists():
        shutil.rmtree(INPUT)
    for d in ("crops", "real", "masks", "moments"):
        (INPUT / d).mkdir(parents=True)
    by_cam = {c: sorted([r for r in rows if r["cam"] == c], key=lambda r: -r["q"]) for c in CAMS}
    best = {c: (rs[0] if rs else None) for c, rs in by_cam.items()}
    picks = _greedy(by_cam[PREF_CAM], n_moments, 1.5)
    alt = max((b for c, b in best.items() if c != PREF_CAM and b), key=lambda r: r["q"], default=None)
    alt_note = None
    if alt and picks and alt["q"] > 1.5 * picks[0]["q"]:
        alt_note = f"phone {alt['cam']} is much better (q {alt['q']:.3f} vs {picks[0]['q']:.3f}): added as an extra moment"
        picks.append(alt)
    elif alt:
        alt_note = (f"best other phone {alt['cam']} q {alt['q']:.3f} (s_px {alt['s_px']}) vs phone 06 q "
                    f"{picks[0]['q'] if picks else 0:.3f} (s_px {picks[0]['s_px'] if picks else 0}): 06 kept")
    m = _smplx_model()
    tmpl, head_ids, tinfo = _template_and_head_ids(m)
    betas = np.asarray(fit["betas"], float)
    rest = _posed(m, betas, None)
    height = float(rest[:, 1].max() - rest[:, 1].min())
    np.save(INPUT / "smplx_faces.npy", np.asarray(m.faces, np.int32))
    inputs, moments = [], []
    for n, r in enumerate(picks):
        mid = f"m{n + 1}"
        cam, i, t = r["cam"], r["i"], r["t"]
        tr = tracks[cam]
        img = cv2.imread(tr.files[i])
        p = interp_params(fit, t)
        k = mo.k(t)
        fc = _face_center(mo, k)
        uv, z = project(fc[None], tr.K[i], tr.R[i], tr.tvec[i], float(tr.k1[i]))
        side = float(tr.K[i][0, 0] * CROP_M / z[0])
        raw = _square_crop(img, uv[0], side)
        m2 = _mask(cam, i, PID)
        mh = _mask(cam, i, PID, "hair")
        if m2 is None:
            m2 = np.ones(img.shape[:2], bool)
        fg = (m2 | mh) if mh is not None else m2
        rad = max(2, int(round(0.012 * side)))
        fg = cv2.dilate(fg.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rad + 1,) * 2)) > 0
        fgc = _square_crop((img * fg[..., None]).astype(np.uint8), uv[0], side)
        grey = _square_crop(np.where(fg[..., None], img, 128).astype(np.uint8), uv[0], side)
        stem = f"{mid}_{cam}_{i:06d}"
        for var, im in (("raw", raw), ("fg", fgc), ("gray", grey)):
            cv2.imwrite(str(INPUT / "crops" / f"{stem}_{var}.png"), im)
            # face_xy: her fitted face centre in the crop (normalised): difflocks_pod.py uses the MediaPipe detection
            # nearest to it (DiffLocks itself takes the largest 'face'; on phone 02 that was the print on her shirt)
            inputs.append({"name": f"{stem}_{var}", "img": _dc(INPUT / "crops" / f"{stem}_{var}.png"), "moment": mid,
                           "cam": cam, "i": i, "t": t, "variant": var, "seed": 5, "face_xy": [0.5, 0.5]})
        # alignment: template head -> her posed head at t
        posed = _posed(m, betas, p)
        s, R, T, rms = umeyama(tmpl[head_ids], posed[head_ids])
        np.save(INPUT / "moments" / f"{mid}_body.npy", posed.astype(np.float32))
        comps = _compare_cams(mo, tracks, t, cam, k)
        views = []
        for c in comps:
            trc = tracks[c["cam"]]
            j = c["j"]
            key = f"{c['cam']}_{j:06d}"
            shutil.copy2(trc.files[j], INPUT / "real" / f"{key}.jpg")
            for pid, kind, tag in ((PID, "masks", "f"), (LEAD_PID, "masks", "l"), (PID, "hair", "h")):
                src = WORK / "dancers" / c["cam"] / kind / f"mask_{j:04d}_{pid}.png"
                if src.exists():
                    shutil.copy2(src, INPUT / "masks" / f"{key}_{tag}.png")
            # crop: head + the hair down the back (crown 0.25 m above the head joint .. 0.75 m below), at head depth
            head = mo.J[k, HEAD_J]
            cc, zc = project((head - 0.20 * np.array([0, 1.0, 0]))[None], trc.K[j], trc.R[j], trc.tvec[j], float(trc.k1[j]))
            side_c = float(trc.K[j][0, 0] * 1.10 / zc[0])
            views.append({**c, "key": key, "K": np.asarray(trc.K[j]).tolist(), "R": np.asarray(trc.R[j]).tolist(),
                          "tvec": np.asarray(trc.tvec[j]).tolist(), "k1": float(trc.k1[j]), "width": trc.width,
                          "height": trc.height, "crop": _crop_box(cc[0], side_c),
                          "label": ("self" if c["own"] else _view_label(c["az_deg"]))})
        moments.append({"id": mid, "cam": cam, "i": i, "t": t, "score": r, "crop_box": _crop_box(uv[0], side),
                        "align": {"s": s, "R": R.tolist(), "T": T.tolist(), "rms_mm": round(rms * 1000, 2)},
                        "head_joint": mo.J[k, HEAD_J].tolist(), "up": mo.up[k].tolist(), "fwd": mo.fwd[k].tolist(),
                        "views": views})
        log(f"{mid}: cam {cam} frame {i} t {t:.3f} s facing {r['facing_deg']} deg, head {r['s_px']} px, calm w "
            f"{r['w_head']:.0f} deg/s v {r['v_head']:.2f} m/s, q {r['q']:.3f}; align s {s:.3f} rms {rms * 1000:.1f} mm; "
            f"compare {[(v['cam'], v['label']) for v in views]}")
    if with_portrait and PORTRAIT.exists():
        shutil.copy2(PORTRAIT, INPUT / "crops" / "portrait.png")
        inputs.append({"name": "portrait", "img": _dc(INPUT / "crops" / "portrait.png"), "moment": None, "cam": None,
                       "variant": "portrait", "seed": 5})
    if picks:  # a second seed on the best frame: how much of the result is the sampler's
        b = dict(inputs[0])
        b.update(name=b["name"] + "_s2", seed=17)
        inputs.insert(1, b)
    meta = {"take": TAKE, "role": ROLE, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "licence": LICENCE,
            "avatar_rest_height_m": round(height, 4), "template": tinfo, "alt_cam_note": alt_note,
            "best_per_cam": {c: ({k2: b[k2] for k2 in ("i", "t", "q", "s_px", "facing_deg", "w_head", "v_head",
                                                       "frac_follow", "frac_lead")} if b else None)
                             for c, b in best.items()},
            "crop_m": CROP_M, "crop_px": CROP_PX, "calm_window_s": CALM_WIN, "seconds": round(time.time() - t0, 1)}
    (INPUT / "inputs.json").write_text(json.dumps(inputs, indent=1))
    (INPUT / "moments.json").write_text(json.dumps({"meta": meta, "moments": moments}, indent=1))
    top = {c: [{k2: r[k2] for k2 in ("i", "t", "q", "s_px", "facing_deg", "w_head", "v_head", "frac_follow",
                                       "frac_lead", "sharp")} for r in rs[:12]] for c, rs in by_cam.items()}
    (OUT / "scan.json").write_text(json.dumps({"n_rows": len(rows), "top": top}, indent=1))
    _candidates_sheet(by_cam, tracks)
    log(f"{len(inputs)} inputs, {len(moments)} moments in {meta['seconds']} s; {alt_note}")
    return {"inputs": inputs, "meta": meta}


def _view_label(az: float) -> str:
    a = abs(az)
    if a < 45:
        return f"front az{az:+.0f}"
    if a > 135:
        return f"back az{az:+.0f}"
    return f"{'left' if az > 0 else 'right'} side az{az:+.0f}"


def _candidates_sheet(by_cam: dict, tracks: dict, n: int = 4, tile: int = 200) -> Path:
    rows = []
    for cam in CAMS:
        tiles = []
        for r in by_cam[cam][:n]:
            img = cv2.imread(tracks[cam].files[r["i"]])
            t = _square_crop(img, np.array(r["uv"]), 4 * r["s_px"], tile)
            cv2.putText(t, f"{cam} #{r['i']} q{r['q']:.2f}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1,
                        cv2.LINE_AA)
            cv2.putText(t, f"{r['s_px']:.0f}px f{r['facing_deg']:.0f} w{r['w_head']:.0f}", (4, tile - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
            tiles.append(t)
        tiles += [np.zeros((tile, tile, 3), np.uint8)] * (n - len(tiles))
        rows.append(np.concatenate(tiles, 1))
    sheet = np.concatenate(rows, 0)
    p = OUT / "candidates.jpg"
    cv2.imwrite(str(p), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return p


def select(names: list[str], portrait: bool = False, seed2: str | None = None) -> list[dict]:
    """Keep these inputs (in this order; moments without an input are dropped), optionally add the portrait and a
    second-seed copy of one input (after it)."""
    allin = {x["name"]: x for x in json.loads((INPUT / "inputs.json").read_text())}
    out = []
    for n in names:
        if n not in allin:
            raise SystemExit(f"no input {n!r}; have {sorted(allin)}")
        x = dict(allin[n])
        x.pop("_s2", None)
        out.append(x)
        if seed2 == n:
            y = dict(x)
            y.update(name=n + "_s2", seed=17)
            out.append(y)
    if portrait and PORTRAIT.exists():
        shutil.copy2(PORTRAIT, INPUT / "crops" / "portrait.png")
        out.append({"name": "portrait", "img": _dc(INPUT / "crops" / "portrait.png"), "moment": None, "cam": None,
                    "variant": "portrait", "seed": 5})
    (INPUT / "inputs.json").write_text(json.dumps(out, indent=1))
    mj = json.loads((INPUT / "moments.json").read_text())
    used = {x.get("moment") for x in out}
    mj["moments"] = [m for m in mj["moments"] if m["id"] in used]
    mj["meta"]["selected"] = [x["name"] for x in out]
    (INPUT / "moments.json").write_text(json.dumps(mj, indent=1))
    keep_keys = {v["key"] for m in mj["moments"] for v in m["views"]}
    for d in ("real", "masks"):
        for f in (INPUT / d).iterdir():
            if not any(f.name.startswith(k) for k in keep_keys):
                f.unlink()
    for f in (INPUT / "crops").iterdir():
        if f.stem not in {Path(x["img"]).stem for x in out}:
            f.unlink()
    for f in (INPUT / "moments").iterdir():
        if f.name.split("_")[0] not in used:
            f.unlink()
    log(f"inputs: {[x['name'] for x in out]}; moments {sorted(used - {None})}")
    return out


# ================================================================================================ pod job


def job(max_minutes: float = 75) -> Path:
    ins = [{"local": _ws(CKPT_EMA), "pod": f"{POD_CKPT}/difflocks_diffusion/{CKPT_EMA.name}"}]
    for rel in ("strand_vae/strand_codec.pt", "strand_vae/hyperparams.json", "rgb2material/rgb2material.pt",
                "rgb2material/hyperparams.json"):
        ins.append({"local": _ws(CKPT_SRC / rel), "pod": f"{POD_CKPT}/{rel}"})
    ins.append(_ws(INPUT) + "/")
    venv_py = "/workspace/difflocks_cand/venv/bin/python"
    repo = "/workspace/difflocks_cand/difflocks"
    d = {"schema": "dancecap.pod_job/1", "name": NAME, "take": TAKE, "max_minutes": max_minutes, "needs": [],
         "inputs": ins,
         # no MAMMA code, no cached CUDA wheels: the bundle is the 2.43 GB model_ema + ~15 MB
         "exclude": ["dancecap/work/runpod/pywheels", "dancecap/work/runpod/wheelhouse", "mamma-turing"],
         "env": {"DIFFLOCKS_COMMIT": DIFFLOCKS_COMMIT, "DINOV2_COMMIT": DINOV2_COMMIT,
                 "FLASH_ATTN_WHEEL_URL": FLASH_ATTN_WHEEL, "DIFFLOCKS_CKPT_DIR": f"/workspace/HeadMovement/{POD_CKPT}",
                 "TORCH_HOME": "/workspace/difflocks_cand/torch_home", "K_DIFFUSION_USE_COMPILE": "0",
                 "PYTHONUNBUFFERED": "1"},
         "commands": [
             {"name": "install", "timeout_min": 22, "cmd": "bash remote/cand_difflocks/install.sh"},
             {"name": "infer", "timeout_min": 30,
              "cmd": f"{venv_py} remote/cand_difflocks/difflocks_pod.py --inputs {_dc(INPUT / 'inputs.json')} "
                     f"--out {_dc(POD_OUT)} --repo {repo} --ckpt-dir /workspace/HeadMovement/{POD_CKPT}"},
             {"name": "render", "timeout_min": 15, "optional": True,
              "cmd": f"python -m dancecap.mono4d.cand_difflocks render --dir {_dc(POD_OUT)} "
                     f"--mesh {repo}/data_loader/difflocks_bodydata/smplx_base.ply"},
             {"name": "blender", "timeout_min": 14, "optional": True,
              "cmd": f"bash remote/cand_difflocks/blender_export.sh {_dc(POD_OUT)}"},
             {"name": "cleanup", "timeout_min": 2, "optional": True,
              "cmd": f"rm -rf /workspace/HeadMovement/{POD_CKPT} {repo}/checkpoints /workspace/difflocks_cand/tmp "
                     "/workspace/upload/bundle_*.tar.gz && echo '[cleanup] licensed checkpoints and the bundle "
                     "deleted from the pod' && ls -la /workspace/HeadMovement/models 2>&1 | head -5"}],
         "outputs": [_ws(POD_OUT) + "/"],
         "verify": [_ws(POD_OUT / "infer.json")],
         "after_download": [f"python -m dancecap.mono4d.{NAME} import"],
         "notes": [LICENCE, "inputs: the stripped model_ema checkpoint (2.43 GB of the 9.7 GB training file), strand VAE, "
                   "rgb2material, head crops of the follower + 2-3 real frames / masks per moment for the overlays",
                   "NATTEN: exact pure-PyTorch na2d_qk/na2d_av (no NATTEN binary wheels exist for torch 2.5 on PyPI; "
                   "the shi-labs wheel server's certificate had expired)"]}
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    p = JOB_DIR / "pod_job.json"
    p.write_text(json.dumps(d, indent=1), encoding="utf-8")
    log(f"wrote {p}")
    return p


# ================================================================================================ strands: stats


def read_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Binary little-endian PLY (Blender export: float xyz/normal/st [+ uchar rgba], uchar-count uint faces)."""
    b = Path(path).read_bytes()
    h = b.index(b"end_header\n") + len(b"end_header\n")
    hdr = b[:h].decode("ascii", "replace").splitlines()
    nv = int(next(x for x in hdr if x.startswith("element vertex")).split()[-1])
    nf = int(next(x for x in hdr if x.startswith("element face")).split()[-1])
    props, in_v = [], False
    for x in hdr:
        if x.startswith("element"):
            in_v = x.startswith("element vertex")
        elif x.startswith("property") and in_v and "list" not in x:
            _, typ, nm = x.split()
            props.append((nm, {"float": "<f4", "uchar": "u1", "double": "<f8", "int": "<i4"}[typ]))
    v = np.frombuffer(b, dtype=np.dtype(props), count=nv, offset=h)
    f = np.frombuffer(b, dtype=np.dtype([("n", "u1"), ("i", "<u4", (3,))]), count=nf, offset=h + v.nbytes)
    return np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64), f["i"].astype(np.int64)


def strand_stats(P: np.ndarray, scale: float = 1.0) -> dict:
    """P (N, n, 3) strands in the template frame (m). scale: template -> her world (alignment s)."""
    P = np.asarray(P, np.float32)
    seg = np.linalg.norm(np.diff(P, axis=1), axis=2)
    L = seg.sum(1)
    chord = np.linalg.norm(P[:, -1] - P[:, 0], axis=1)
    root, tip = P[:, 0], P[:, -1]
    crown = TEMPLATE_CROWN_Y
    drop = crown - tip[:, 1]
    back = root[:, 2] < -0.03
    pct = lambda a, q=(5, 25, 50, 75, 95): {f"p{x}": round(float(np.percentile(a, x)) * scale, 4) for x in q}  # noqa
    prof = []
    flat = P.reshape(-1, 3)
    for b in range(12):
        sel = (flat[:, 1] <= crown - 0.05 * b) & (flat[:, 1] > crown - 0.05 * (b + 1))
        prof.append(round(float(np.percentile(flat[sel, 0], 97) - np.percentile(flat[sel, 0], 3)) * scale, 4)
                    if sel.sum() > 100 else None)
    # parting: top-cap roots, lateral direction 2-4 cm along the strand
    cum = np.concatenate([np.zeros((len(P), 1), np.float32), np.cumsum(seg, 1)], 1)
    idx = np.clip((cum < 0.03).sum(1), 1, P.shape[1] - 1)
    dirx = P[np.arange(len(P)), idx, 0] - root[:, 0]
    top = root[:, 1] > crown - 0.06
    part = None
    if top.sum() > 200:
        rx, gx = root[top, 0], dirx[top] > 0
        xs = np.linspace(-0.07, 0.07, 141)
        err = [((rx < x) & gx).sum() + ((rx >= x) & ~gx).sum() for x in xs]
        x_best = float(xs[int(np.argmin(err))])
        part = {"x_m": round(x_best * scale, 4), "side": ("her left" if x_best > 0.01 else
                                                           "her right" if x_best < -0.01 else "centre"),
                "top_to_her_right": round(float((~gx).mean()), 3), "top_to_her_left": round(float(gx.mean()), 3),
                "misfit": round(float(min(err)) / len(rx), 3), "n_top": int(top.sum())}
    return {"n_strands": int(len(P)), "pts_per_strand": int(P.shape[1]), "scale": round(scale, 4),
            "length_m": {**pct(L), "max": round(float(L.max()) * scale, 4), "mean": round(float(L.mean()) * scale, 4)},
            "length_hist_m": np.histogram(L * scale, bins=np.arange(0, 1.05, 0.05))[0].tolist(),
            "tip_drop_below_crown_m": pct(drop), "back_rooted": {"n": int(back.sum()), "length_m": pct(L[back]),
                                                                  "tip_drop_m": pct(drop[back])},
            "straightness_chord_over_arc": pct(chord / np.maximum(L, 1e-6), (25, 50, 75)),
            "width_profile_m": prof, "part": part,
            "lowest_tip_y_template": round(float(tip[:, 1].min()), 4)}


def hair_albedo(mat: dict | None) -> tuple[np.ndarray, str]:
    """Rough sRGB albedo from DiffLocks' melanin / redness (Cycles Principled Hair melanin
    parametrisation and absorption coefficients). Linear RGB in [0, 1] and hex."""
    if not mat:
        lin = np.array([0.171, 0.025, 0.018])
    else:
        mel, red = float(mat.get("melanin", 0.5)), float(mat.get("redness", 0.5))
        q = -math.log(max(1e-4, 1.0 - min(0.9999, mel)))
        sig = q * (1 - red) * np.array([0.506, 0.841, 1.653]) + q * red * np.array([0.343, 0.733, 1.924])
        lin = np.exp(-sig * 4.0)  # Cycles Principled Hair coefficients; ~4 fibre-widths of path: a display colour
    srgb = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * np.power(lin, 1 / 2.4) - 0.055)
    return lin, "#" + "".join(f"{int(round(np.clip(c, 0, 1) * 255)):02X}" for c in srgb)


# ================================================================================================ rendering


def _densify(P: np.ndarray, px_per_m: float, max_step_px: float = 0.8, cap: int = 10) -> tuple[np.ndarray, np.ndarray,
                                                                                                  np.ndarray]:
    """Strands (N, n, 3) -> points along every segment at <= max_step_px, their unit tangents and strand ids."""
    seg = np.diff(P, axis=1)
    sl = np.linalg.norm(seg, axis=2)
    m = int(np.clip(math.ceil(float(np.percentile(sl, 99)) * px_per_m / max_step_px), 1, cap))
    a = (np.arange(m, dtype=np.float32) / m)[None, None, :, None]
    pts = (P[:, :-1, None, :] + a * seg[:, :, None, :]).reshape(len(P), -1, 3)
    tan = np.repeat(seg / np.maximum(sl, 1e-9)[..., None], m, axis=1)
    sid = np.repeat(np.arange(len(P)), pts.shape[1])
    par = np.tile(np.repeat(np.linspace(0, 1, P.shape[1] - 1, dtype=np.float32), m), len(P))
    return pts.reshape(-1, 3), tan.reshape(-1, 3), np.stack([sid, par], 1)


def _mesh_points(V: np.ndarray, F: np.ndarray, px_per_m: float, keep=None, spacing_px: float = 0.7, seed: int = 0):
    """Area-proportional surface samples of a triangle mesh (density from the pixel scale) + face normals."""
    rng = np.random.default_rng(seed)
    if keep is not None:
        F = F[keep(V[F].mean(1))]
    A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    n = np.cross(B - A, C - A)
    area = 0.5 * np.linalg.norm(n, axis=1)
    nrm = n / np.maximum(2 * area, 1e-12)[:, None]
    cnt = np.maximum(1, np.ceil(area * (px_per_m / spacing_px) ** 2)).astype(np.int64)
    fi = np.repeat(np.arange(len(F)), cnt)
    r1, r2 = rng.random(len(fi)), rng.random(len(fi))
    s = np.sqrt(r1)
    w = np.stack([1 - s, s * (1 - r2), s * r2], 1)
    P = w[:, :1] * A[fi] + w[:, 1:2] * B[fi] + w[:, 2:] * C[fi]
    return P.astype(np.float32), nrm[fi].astype(np.float32)


def _zbuffer(u: np.ndarray, v: np.ndarray, depth: np.ndarray, W: int, H: int) -> tuple[np.ndarray, np.ndarray]:
    """Nearest point per pixel: (flat pixel ids, index into the inputs)."""
    ui, vi = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
    ok = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H) & np.isfinite(depth)
    idx = np.flatnonzero(ok)
    pix = vi[idx] * W + ui[idx]
    order = np.lexsort((depth[idx], pix))
    ps = pix[order]
    first = np.r_[True, ps[1:] != ps[:-1]]
    return ps[first], idx[order[first]]


def render_ortho(strands: np.ndarray, mesh: tuple[np.ndarray, np.ndarray] | None, view: str, albedo_lin: np.ndarray,
                 size: int = 1024, center=(0.0, 0.08, 0.0), extent: float = 1.0, seed: int = 0) -> np.ndarray:
    """Orthographic render in the template frame. view: front | back | left | right | top (her left/right side)."""
    dirs = {"front": ([0, 0, -1.0], [0, 1.0, 0]), "back": ([0, 0, 1.0], [0, 1.0, 0]),
            "left": ([-1.0, 0, 0], [0, 1.0, 0]), "right": ([1.0, 0, 0], [0, 1.0, 0]), "top": ([0, -1.0, 0], [0, 0, -1.0])}
    d, up = (np.asarray(x, np.float32) for x in dirs[view])
    right = np.cross(d, up)
    ppm = size / extent
    c = np.asarray(center, np.float32)
    pts, tan, sp = _densify(np.asarray(strands, np.float32), ppm)
    rng = np.random.default_rng(seed)
    jit = rng.uniform(0.75, 1.2, len(strands)).astype(np.float32)[sp[:, 0].astype(np.int64)]
    L = d * -1.0 + up * 0.7 + right * -0.45
    L /= np.linalg.norm(L)
    Hv = L - d
    Hv /= np.linalg.norm(Hv)
    tl, th = tan @ L, tan @ Hv
    diff = np.sqrt(np.clip(1 - tl * tl, 0, 1))
    spec = np.sqrt(np.clip(1 - th * th, 0, 1)) ** 80
    rootdark = 0.75 + 0.25 * np.clip(sp[:, 1] * 4, 0, 1)
    col_h = (albedo_lin.astype(np.float32)[None] * (0.18 + 0.82 * diff)[:, None] * (jit * rootdark)[:, None]
             + 0.22 * spec[:, None]).astype(np.float32)
    allp, allc = [pts], [col_h]
    if mesh is not None:
        V, F = mesh
        mp_, mn = _mesh_points(V, F, ppm, keep=lambda cen: cen[:, 1] > c[1] - extent * 0.62)
        lam = np.clip(mn @ L, 0, 1) * 0.75 + 0.25
        skin = np.array([0.55, 0.47, 0.43], np.float32)
        allp.append(mp_)
        allc.append((skin[None] * lam[:, None]).astype(np.float32))
    P = np.concatenate(allp)
    Cc = np.concatenate(allc)
    is_body = np.r_[np.zeros(len(pts), bool), np.ones(len(P) - len(pts), bool)]
    rel = P - c
    u = rel @ right * ppm + size / 2
    v = -(rel @ up) * ppm + size / 2
    pix, idx = _zbuffer(u, v, rel @ d, size, size)
    img = np.empty((size * size, 3), np.float32)
    g = np.linspace(0.035, 0.012, size, dtype=np.float32)
    img[:] = np.repeat(g, size)[:, None]
    img[pix] = Cc[idx]
    img = img.reshape(size, size, 3)
    # fill the sampling holes of the body surface (not the gaps between strands)
    cov = np.zeros(size * size, np.float32)
    cov[pix] = 1.0
    body = np.zeros(size * size, np.float32)
    body[pix[is_body[idx]]] = 1.0
    cov, body = cov.reshape(size, size), body.reshape(size, size)
    den = cv2.blur(body, (3, 3))
    num = cv2.blur(img * body[..., None], (3, 3))
    holes = (cov == 0) & (den > 0.55)
    img[holes] = num[holes] / den[holes][:, None]
    srgb = np.where(img <= 0.0031308, 12.92 * img, 1.055 * np.power(np.clip(img, 0, None), 1 / 2.4) - 0.055)
    return (np.clip(srgb, 0, 1) * 255 + 0.5).astype(np.uint8)[:, :, ::-1]


def overlay_view(strands_t: np.ndarray, align: dict, body: np.ndarray, faces: np.ndarray, view: dict, real: np.ndarray,
                 masks: dict, out_px: int = 512) -> tuple[np.ndarray, np.ndarray, dict]:
    """Strands rigidly attached to her head at the moment, projected into a phone; z-tested against her posed body.
    Returns (overlay crop, real crop, metrics vs the phone's hair mask)."""
    s, R, T = float(align["s"]), np.asarray(align["R"]), np.asarray(align["T"])
    Pw = s * strands_t.reshape(-1, 3) @ R.T + T
    K, Rc, tc, k1 = np.asarray(view["K"]), np.asarray(view["R"]), np.asarray(view["tvec"]), float(view["k1"])
    W, H = int(view["width"]), int(view["height"])
    x0, y0, S = view["crop"]
    zc = float(np.median((Pw @ Rc.T + tc)[:, 2]))
    ppm = K[0, 0] / zc
    # densify in the phone's pixel scale
    Pw = Pw.reshape(strands_t.shape)
    pts, _, _ = _densify(Pw, ppm, max_step_px=0.7, cap=6)
    bp, _ = _mesh_points(body, faces, ppm, spacing_px=0.6)
    uv_h, z_h = project(pts, K, Rc, tc, k1)
    uv_b, z_b = project(bp, K, Rc, tc, k1)
    uv = np.concatenate([uv_h, uv_b])
    z = np.concatenate([z_h, z_b])
    lab = np.concatenate([np.ones(len(z_h), bool), np.zeros(len(z_b), bool)])
    pix, idx = _zbuffer(uv[:, 0], uv[:, 1], np.where(z > 0, z, np.nan), W, H)
    hair_vis = np.zeros(W * H, bool)
    hair_vis[pix[lab[idx]]] = True
    hair_vis = cv2.morphologyEx(hair_vis.reshape(H, W).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
    gt = masks.get("h")
    lead = masks.get("l")
    met = {}
    box = np.zeros((H, W), bool)
    box[max(0, y0):max(0, y0 + S), max(0, x0):max(0, x0 + S)] = True
    if gt is not None:
        valid = box & ~(lead if lead is not None else np.zeros_like(box))
        P_, G_ = hair_vis & valid, gt & valid
        inter, uni = float((P_ & G_).sum()), float((P_ | G_).sum())
        met = {"iou": round(inter / uni, 3) if uni else None,
               "precision": round(inter / P_.sum(), 3) if P_.sum() else None,
               "recall": round(inter / G_.sum(), 3) if G_.sum() else None,
               "pred_px": int(P_.sum()), "real_px": int(G_.sum())}
        if P_.any() and G_.any():
            vp, vg = np.nonzero(P_)[0].max(), np.nonzero(G_)[0].max()
            met["lowest_px_pred_minus_real"] = int(vp - vg)
            met["lowest_m_pred_minus_real"] = round(float(vp - vg) / ppm, 3)
            wp = np.nonzero(P_.any(0))[0]
            wg = np.nonzero(G_.any(0))[0]
            met["width_m_pred"], met["width_m_real"] = round(float(wp.max() - wp.min()) / ppm, 3), \
                round(float(wg.max() - wg.min()) / ppm, 3)
    img = real.astype(np.float32)
    tint = np.array([255, 230, 40], np.float32)  # cyan in BGR
    a = 0.55
    img[hair_vis] = (1 - a) * img[hair_vis] + a * tint
    ov = img.astype(np.uint8)
    if gt is not None:
        cnt, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(ov, cnt, -1, (0, 220, 255), max(1, int(round(S / 400))), cv2.LINE_AA)
    crop = lambda im: _square_crop(im, np.array([x0 + S / 2, y0 + S / 2]), S, out_px)  # noqa: E731
    return crop(ov), crop(real), met


def _render_one(job: tuple) -> tuple[str, dict | None]:
    inp, pod_out, mesh, mesh_src, m, faces, inputs_dir, n_render = job
    d = Path(pod_out) / inp["name"]
    f = d / "strands_f16.npz"
    if not f.exists():
        return inp["name"], None
    t0 = time.time()
    P = np.load(f)["positions"].astype(np.float32)
    mat = json.loads((d / "hair.json").read_text()) if (d / "hair.json").exists() else None
    lin, hexc = hair_albedo(mat)
    s = float(m["align"]["s"]) if m else 1.0
    st = strand_stats(P, s)
    st.update(albedo_hex=hexc, material=mat, input=inp, scale_note="lengths x s (template -> her head in the "
              "larissa_kadu_38 world, metric)")
    rng = np.random.default_rng(0)
    sub = P[np.sort(rng.choice(len(P), min(n_render, len(P)), replace=False))]
    rd = d / "renders"
    rd.mkdir(exist_ok=True)
    for v in ("front", "left", "right", "back", "top"):
        ext = 0.6 if v == "top" else 1.0
        ctr = (0.0, 0.30, 0.0) if v == "top" else (0.0, 0.08, 0.0)
        cv2.imwrite(str(rd / f"{v}.png"), render_ortho(sub, mesh, v, lin, center=ctr, extent=ext))
    ovm = {}
    if m:
        body = np.load(Path(inputs_dir) / "moments" / f"{m['id']}_body.npy").astype(np.float32)
        for vw in m["views"]:
            real = cv2.imread(str(Path(inputs_dir) / "real" / f"{vw['key']}.jpg"))
            if real is None:
                continue
            masks = {}
            for tag in ("f", "l", "h"):
                mp = Path(inputs_dir) / "masks" / f"{vw['key']}_{tag}.png"
                if mp.exists():
                    masks[tag] = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) > 127
            ov, rc, met = overlay_view(sub, m["align"], body, faces, vw, real, masks)
            cv2.imwrite(str(rd / f"ov_{vw['cam']}.jpg"), ov, [cv2.IMWRITE_JPEG_QUALITY, 90])
            cv2.imwrite(str(rd / f"real_{vw['cam']}.jpg"), rc, [cv2.IMWRITE_JPEG_QUALITY, 90])
            ovm[vw["cam"]] = {"label": vw["label"], "s_px": vw["s_px"], **met}
    st["overlays"] = ovm
    st["render"] = {"n_strands_drawn": int(len(sub)), "mesh": mesh_src, "seconds": round(time.time() - t0, 1)}
    (d / "stats.json").write_text(json.dumps(st, indent=1))
    summ = {"n": st["n_strands"], "len_p50": st["length_m"]["p50"], "len_p95": st["length_m"]["p95"],
            "drop_p50": st["tip_drop_below_crown_m"]["p50"], "albedo": hexc, "part": st["part"] and st["part"]["side"],
            "iou": {c: o.get("iou") for c, o in ovm.items()}}
    log(f"{inp['name']}: {summ}")
    return inp["name"], summ


def render_dir(pod_out: Path, mesh_ply: Path | None = None, n_render: int = 20000, inputs_dir: Path = INPUT,
               workers: int | None = None) -> dict:
    """For every DiffLocks output under pod_out: stats.json, renders/{front,left,right,back,top}.png and overlays into
    the moment's phones (renders/ov_<cam>.jpg, real crops renders/real_<cam>.jpg). Runs on the pod (one process per
    output) or locally (workers=1)."""
    import os
    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    mom = json.loads((inputs_dir / "moments.json").read_text())
    by_id = {m["id"]: m for m in mom["moments"]}
    faces = np.load(inputs_dir / "smplx_faces.npy")
    if mesh_ply and Path(mesh_ply).exists():
        mesh = read_ply(mesh_ply)
        mesh_src = str(mesh_ply)
    else:  # locally: the same geometry from the workspace SMPL-X template (T-pose arms; the head is identical)
        z = np.load(WORKSPACE / "smplx" / "SMPLX_NEUTRAL.npz", allow_pickle=True)
        mesh = (z["v_template"].astype(np.float64), z["f"].astype(np.int64))
        mesh_src = "smplx/SMPLX_NEUTRAL.npz v_template (T-pose)"
    inputs = json.loads((inputs_dir / "inputs.json").read_text())
    jobs = [(inp, str(pod_out), mesh, mesh_src, by_id.get(inp.get("moment")), faces, str(inputs_dir), n_render)
            for inp in inputs if (pod_out / inp["name"] / "strands_f16.npz").exists()]
    if workers is None:
        try:
            q, per = (int(x) if x != "max" else 0 for x in Path("/sys/fs/cgroup/cpu.max").read_text().split())
            cpus = max(1, -(-q // per)) if q else os.cpu_count() or 1
        except Exception:
            cpus = os.cpu_count() or 1
        workers = max(1, min(len(jobs), cpus // 2, 6))
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            res = list(ex.map(_render_one, jobs))
    else:
        res = [_render_one(j) for j in jobs]
    summary = {k: v for k, v in res if v is not None}
    (pod_out / "render.json").write_text(json.dumps({"summary": summary, "seconds": round(time.time() - t0, 1),
                                                     "mesh": mesh_src, "workers": workers}, indent=1))
    log(f"rendered {len(summary)} outputs in {time.time() - t0:.0f} s with {workers} workers")
    return summary


# ================================================================================================ import: sheet


def _tile(img: np.ndarray | None, size: int, label: str = "", sub: str = "") -> np.ndarray:
    if img is None:
        t = np.full((size, size, 3), 24, np.uint8)
    else:
        h, w = img.shape[:2]
        s = size / max(h, w)
        r = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
        t = np.full((size, size, 3), 12, np.uint8)
        y0, x0 = (size - r.shape[0]) // 2, (size - r.shape[1]) // 2
        t[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
    for k, (txt, y) in enumerate(((label, 16), (sub, size - 8))):
        if txt:
            cv2.putText(t, txt, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(t, txt, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1, cv2.LINE_AA)
    return t


def _groom_tiles(size: int) -> list[np.ndarray]:
    out = []
    for v in ("front", "side", "back"):
        p = GROOM_SHOTS / f"hair_calm_{v}.png"
        im = cv2.imread(str(p)) if p.exists() else None
        if im is not None:  # 1280 x 720 Unity game view: the dancer is central
            h, w = im.shape[:2]
            im = im[0:int(h * 0.75), int(w * 0.30):int(w * 0.70)]
        out.append(_tile(im, size, f"current groom {v}", "Unity 04, v2 groom" if im is not None else "none"))
    return out


RUN1_DIR = OUT / "pod_out_run1"  # session 20261019-165713 (before the face-selection fix)
RUN1_SHIRT = {"m4_02_001680_raw", "m4_02_001680_raw_s2", "m4_02_001680_fg"}  # MediaPipe picked her T-shirt print


def _runs() -> list[tuple[str, Path]]:
    return [(t, d) for t, d in (("r2", POD_OUT), ("r1", RUN1_DIR)) if (d / "infer.json").exists()]


def _run_inputs(d: Path) -> tuple[list[dict], dict]:
    ip = d / "inputs.json" if (d / "inputs.json").exists() else INPUT / "inputs.json"
    mp_ = d / "moments.json" if (d / "moments.json").exists() else INPUT / "moments.json"
    return json.loads(ip.read_text()), {m["id"]: m for m in json.loads(mp_.read_text())["moments"]}


def contact_sheet(size: int = 260) -> list[Path]:
    rows, rows_ov, bad, bad_ov = [], [], [], []
    groom = _groom_tiles(size)
    for tag, pod_out in _runs():
        inputs, mom = _run_inputs(pod_out)
        for inp in inputs:
            d = pod_out / inp["name"]
            if not (d / "stats.json").exists():
                continue
            st = json.loads((d / "stats.json").read_text())
            failed = tag == "r1" and inp["name"] in RUN1_SHIRT
            crop = cv2.imread(str(d / "rgb.png")) if (d / "rgb.png").exists() else None
            lbl = f"{tag} {inp['name']}"
            sub = f"{st['n_strands'] // 1000}K strands, L50 {st['length_m']['p50']:.2f} m"
            r = [_tile(crop, size, lbl, "FAILED: shirt-print crop" if failed else "DiffLocks' own 770 px crop")]
            for v in ("front", "right", "back"):
                r.append(_tile(cv2.imread(str(d / "renders" / f"{v}.png")), size, f"DiffLocks {v}",
                               sub if v == "front" else (st["albedo_hex"] if v == "right" else "")))
            r += groom
            m = mom.get(inp.get("moment"))
            ro = [r[0], r[1]]
            if m:
                for vw in [v for v in m["views"] if v["label"] != "self"][:3]:
                    r.append(_tile(cv2.imread(str(d / "renders" / f"real_{vw['cam']}.jpg")), size,
                                   f"phone {vw['cam']} {vw['label']}", f"same moment t {m['t']:.2f} s"))
                for vw in m["views"]:
                    o = st.get("overlays", {}).get(vw["cam"], {})
                    ro.append(_tile(cv2.imread(str(d / "renders" / f"ov_{vw['cam']}.jpg")), size,
                                    f"{vw['cam']} {vw['label']}",
                                    f"IoU {o.get('iou')} P {o.get('precision')} R {o.get('recall')}"))
            elif inp.get("variant") == "portrait":
                r.append(_tile(cv2.imread(str(PORTRAIT)), size, "portrait (input)", "larissa.jpg"))
            (bad if failed else rows).append(r)
            (bad_ov if failed else rows_ov).append(ro)
    paths = []
    for name, rr, title in (("contact_sheet.jpg", rows + bad, "DiffLocks' crop | DiffLocks strands (front, her right, "
                             "back) | current Unity groom (calm, v2) | real phones at the same moment"),
                            ("overlays.jpg", rows_ov + bad_ov, "DiffLocks strands rigidly on her fitted head, projected "
                             "into the phones (cyan = visible strands, yellow = the phone's hair mask)")):
        if not rr:
            continue
        n = max(len(x) for x in rr)
        body = np.concatenate([np.concatenate(x + [_tile(None, size)] * (n - len(x)), 1) for x in rr], 0)
        head = np.full((40, body.shape[1], 3), 0, np.uint8)
        cv2.putText(head, f"{TAKE} follower, DiffLocks @ {DIFFLOCKS_COMMIT[:7]}: {title}", (8, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        p = OUT / name
        cv2.imwrite(str(p), np.concatenate([head, body], 0), [cv2.IMWRITE_JPEG_QUALITY, 88])
        paths.append(p)
    return paths


def _session_meta(sid: str | None) -> dict:
    root = PACKAGE_ROOT / "work" / "runpod" / "sessions"
    if sid is None:
        c = sorted((p for p in root.glob("*") if (p / "result.json").exists()
                    and NAME in (p / "result.json").read_text(errors="ignore")), key=lambda p: p.stat().st_mtime)
        if not c:
            return {}
        sid = c[-1].name
    p = root / sid / "result.json"
    if not p.exists():
        return {"session": sid}
    r = json.loads(p.read_text())
    pod = r.get("pod") or {}
    return {"session": sid, "gpu": pod.get("gpu"), "cost_per_hr": pod.get("cost_per_hr"), "cost_usd": r.get("cost"),
            "minutes": r.get("minutes"), "job": (r.get("jobs") or {}).get(NAME, {}).get("status")}


def do_import(sid: str | None = None, rerender: bool = False, sid_run1: str = "20261019-165713-batch") -> dict:
    for f in ("inputs.json", "moments.json"):  # this run's inputs travel with its outputs
        if (POD_OUT / "infer.json").exists() and not (POD_OUT / f).exists():
            shutil.copy2(INPUT / f, POD_OUT / f)
    if rerender or not (POD_OUT / "render.json").exists():
        log("renders missing (or --rerender): rendering locally")
        render_dir(POD_OUT, None, workers=1)
    sheets = contact_sheet()
    runs = {}
    for tag, d0 in _runs():
        infer = json.loads((d0 / "infer.json").read_text())
        stats = {}
        for d in sorted(d0.iterdir()):
            if (d / "stats.json").exists():
                st = json.loads((d / "stats.json").read_text())
                res = json.loads((d / "result.json").read_text()) if (d / "result.json").exists() else {}
                stats[d.name] = {k: st.get(k) for k in ("n_strands", "length_m", "length_hist_m", "tip_drop_below_crown_m",
                                                         "back_rooted", "straightness_chord_over_arc", "width_profile_m",
                                                         "part", "albedo_hex", "material", "overlays", "scale")}
                stats[d.name].update(face=res.get("face"), seconds=res.get("seconds"), peak_gpu_gb=res.get("peak_gpu_gb"),
                                     shirt_crop_failure=(tag == "r1" and d.name in RUN1_SHIRT))
        bl = d0 / "blender" / "blender.json"
        runs[tag] = {"dir": str(d0), "session": _session_meta(sid if tag == "r2" else sid_run1), "env": infer.get("env"),
                     "stats": stats, "blender": json.loads(bl.read_text()) if bl.exists() else None}
    meta = json.loads((INPUT / "moments.json").read_text())["meta"]
    rep = {"name": NAME, "take": TAKE, "difflocks_commit": DIFFLOCKS_COMMIT, "runs": runs, "reference": REF,
           "avatar_rest_height_m": meta.get("avatar_rest_height_m"), "frame_pick": {k: meta.get(k) for k in
           ("alt_cam_note", "best_per_cam", "crop_m", "calm_window_s")}, "sheets": [str(p) for p in sheets],
           "licence": LICENCE}
    (OUT / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    log(f"report {OUT / 'report.json'}; sheets {[p.name for p in sheets]}")
    return rep


# ================================================================================================ CLI


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog=f"python -m dancecap.mono4d.{NAME}")
    ap.add_argument("action", choices=["strip-ckpt", "scan", "prep", "select", "job", "render", "sheet", "import"])
    ap.add_argument("--names", default="", help="select: comma list of input names to keep, in order")
    ap.add_argument("--seed2", default=None, help="select: add a second-seed copy of this input")
    ap.add_argument("--moments", type=int, default=3)
    ap.add_argument("--portrait", action="store_true", help="prep: also run the small portrait (larissa.jpg)")
    ap.add_argument("--max-minutes", type=float, default=75)
    ap.add_argument("--dir", default=str(POD_OUT))
    ap.add_argument("--mesh", default=None)
    ap.add_argument("--session")
    ap.add_argument("--rerender", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args(argv)
    if a.action == "strip-ckpt":
        strip_ckpt()
    elif a.action == "scan":
        rows = scan()
        print(json.dumps(sorted(rows, key=lambda r: -r["q"])[:15], indent=1))
    elif a.action == "prep":
        prep(a.moments, a.portrait)
    elif a.action == "select":
        select([x for x in a.names.split(",") if x], a.portrait, a.seed2)
    elif a.action == "job":
        job(a.max_minutes)
    elif a.action == "render":
        d = Path(a.dir)
        render_dir(d if d.is_absolute() else Path.cwd() / d, Path(a.mesh) if a.mesh else None,
                   workers=a.workers)
    elif a.action == "sheet":
        print(contact_sheet())
    else:
        do_import(a.session, a.rerender)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
