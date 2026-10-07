"""Hair reference for the follow: shape, motion and colour measured from the multi-view video and her portrait, plus
the original head-movement tip-bloom recipe -> acceptance criteria for the Unity hair (docs/HAIR_REFERENCE.md).

    python -m dancecap.hair_reference larissa_kadu --reuse-hull --device cpu   re-measure / re-render from hull.npz
    python -m dancecap.hair_reference larissa_kadu_38 --write-pod-job          long takes: hair_ref/pod_job.json for
        python -m dancecap remote run --take larissa_kadu_38 --jobs hair_ref   Runpod (hull + colour + MediaPipe), then
        python -m dancecap.hair_reference larissa_kadu_38 --reuse-hull --reuse-colour --device cpu   (local re-render)
    python -m dancecap.hair_reference larissa_kadu                full local run (the hull needs the GPU ~12 min)

Compute policy (the workstation is shared): the 3D hull is the only GPU step. Locally it runs in a CHILD process
(--only-hull) under dancecap.gpu.exclusive and torch.cuda.empty_cache(), so this process never holds a CUDA context
or cached VRAM while it does minutes of CPU work; --device cpu keeps everything on the CPU. Everything else (re-projection
check, shape / motion metrics, colour, review images) runs on the CPU with --threads (default 4) threads, and the
re-projection check (hair_ref/consistency.json) and colour samples (hair_ref/colour_samples.npz) are cached so a
re-render does not redo them. Colour uses at most --colour-views-per-cam views per camera (auto step >= 6).

Inputs (all per take, nothing take-specific in the code): smplx/follow.npz (+ avatar/head_fix.npz when adopted),
cameras/<cam>.npz + world_from_sfm.json + sync/offsets.json (mamma_dyn.load_tracks: sync-corrected world cameras),
frames/<cam>, dancers/<cam>/{masks,body,hair} (follow full / body / hair masks, lead full mask), dancers/roles.json,
takes/<take>.toml [dancers.follow] stature_m + name + pid, [dancers.lead] pid (portrait = <source_dir>/<name lower>.jpg
unless --portrait).

Method
  1. Hair hull (3D, per 30 Hz grid frame): a voxel grid around her head (1.5 cm) is carved by the follow's full
     masks in every camera's sync-nearest frame (|dt| <= 0.6 frame), occlusion-aware: pixels outside the image never
     carve, pixels of the lead's mask only carve what lies clearly in front of his rendered SMPL-X body. Masks are dilated by max(2 px, 1.5 cm at her head depth); each view is first
     registered to the fit by the 2D shift that maximises IoU(rendered SMPL-X follow silhouette, follow BODY mask)
     (absorbs the fit's few-pixel misregistration). Hull voxels on or outside the posed SMPL-X surface (signed via vertex
     normals, so hair lying flat on her back is kept) are hair when they project into the hair mask in >= 2 cameras
     and at least twice as often as CONFIDENTLY onto her body (her mask minus the hair, eroded by the tolerance, so
     rays grazing her silhouette cast no vote) among the cameras that see them un-occluded by her own body (exact z-buffer, avatar_raster) - then only the components touching her head (26-connected
     across gaps of up to 2 voxels) are kept. A visual hull is a superset (concavities fill, masks are dilated): widths
     and thicknesses are upper bounds, of the order of the dilation per side.
  2. Shape metrics on CALM frames (slow head: low yaw rate and speed): tip level = lowest hair voxel lying on her
     back (nearest posed vertex within 5 cm) mapped to the REST mesh height of that vertex (so it is independent of
     the pose), strand length = posterior-envelope arc length of the rest mesh from the crown to the tip level,
     silhouette width/thickness profile in 5 cm bands below the crown in her torso frame (front tip level = a LOWER
     bound: the lead and her arms hide the front). Motion on TURN frames: spread radius (horizontal distance of hair
     from the neck axis; an upper bound in turns, motion blur grows the hull) and the trailing angle of the hair-mass
     azimuth. The hair mass is not straight behind her head even when calm (it lies over one shoulder), so the
     trailing angle is measured against the CALM-frame offset, against the head and the torso heading, per turn
     direction, with a bootstrap CI; it is reported as resolved only when both references and both directions agree
     and it clears the noise floor (motion_metrics). Unresolved trailing / lag are left out of the acceptance.
  3. Colour (2D, per view): hair pixels = hair mask (eroded 1 px) AND MediaPipe SelfieMulticlass hair (run in
     .venv-face, hair_segment_mp.py; skipped with --no-mp), zoned root / mid / tips by the distance from her projected
     head centre in metres at head depth. White balance per view against her white top (low-saturation, unclipped
     body-mask pixels in the torso quad of the projected SMPL-X joints), scaled to an assumed top albedo of 0.80
     -> OBSERVED white-balanced linear sRGB: it includes the hair's own shading and self-shadowing, so it is an image
     statistic, NOT an albedo. Medians per camera, then across cameras (the spread is reported). Highlights and
     shadows = luminance bands of the mid+tip pixels; image_stats = the p10 / p50 / p90 luminance pixels, which the
     rendered hair must match under the demo lighting. Phone JPEGs are tone-mapped: "linear" = sRGB-decoded.
  4. Portrait: SelfieMulticlass hair + clothes masks; white balance on the white top; zones from the top of the hair
     to the lowest tips; length of the visible (front) section in head heights.
  5. Tip bloom: the original head-movement recipe parsed from the repo (git history of HairSimulation.cs, the shader
     graph, BloomMat, the URP assets and the default volume profile) - see tip_bloom_recipe().

World scale: lengths are measured in the fit's world and also reported stature-corrected (x stature+sole / SMPL-X
rest height of the follow; the larissa_kadu world is ~4 % large, CONTRACTS.md cameras/world_from_sfm.json).
Qualitative notes (parting, fringe, layering, texture) are read from hair_ref/observations.json when present.
Outputs (git-ignored): avatar/hair_reference.json, hair_ref/ (hull.npz, consistency.json, colour_samples.npz,
MediaPipe masks, motion plot, pod_job.json), review/hair_reference.jpg (reference sheet), review/hair_reference_hull.jpg
(3D hull re-projected into the videos).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .take import PACKAGE_ROOT, WORKSPACE, Take

ROLE = "follow"
DEFAULT_PID, DEFAULT_LEAD_PID = 2, 1  # pipeline convention; the take's [dancers.follow/lead] pid wins
J_PELVIS, J_LHIP, J_RHIP, J_SPINE1, J_SPINE2, J_SPINE3, J_NECK, J_HEAD, J_LSH, J_RSH = 0, 1, 2, 3, 6, 9, 12, 15, 16, 17
CHAIN_HEAD = (0, 3, 6, 9, 12, 15)  # SMPL-X kinematic chain pelvis -> head
FACE_VENV = WORKSPACE / ".venv-face" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
MP_DIR = PACKAGE_ROOT / "weights" / "mediapipe"
MP_MODELS = {"hair": MP_DIR / "hair_segmenter.tflite", "multiclass": MP_DIR / "selfie_multiclass_256x256.tflite"}
MP_VERSION = "1.1.0"  # mediapipe in .venv-face (pinned for the pod install too)
HM_REPO = WORKSPACE / "head-movement"
ORIGINAL_HAIR_COMMIT = "a4e493b"  # last commit before the 2026 capture-v3 edits of HairSimulation.cs (user's original)
# LineRenderer.BakeMesh vertex colours of the original 6-point strand (x 255), read in the running Unity 6000.6.3f1
# editor on 2026-10-06 (vertex attribute Color: UNorm8x4). Kept apart from the model (baked_vertex_colours), which
# reproduces them within one 8-bit step (18 vs 19, 115 vs 114: Unity quantises gamma and linear separately).
MEASURED_BAKED_8BIT = (0, 1, 2, 18, 115, 255)
MEASURED_STRAND_KEYS_T = (0.0, 0.5, 0.95, 1.0)  # gradient key times of the measured strand
URP_SOFT_KNEE = 0.5  # URP bloom: knee = 0.5 x threshold (hard-coded), pixels above threshold - knee start to bloom


@dataclass
class HairRefConfig:
    voxel_m: float = 0.015
    box_xz_m: float = 0.80        # half size of the voxel box around the head joint (horizontal)
    box_up_m: float = 0.40
    box_down_m: float = 1.15
    tol_px: float = 2.0           # minimum mask dilation (analysis px)
    tol_m: float = 0.015          # mask dilation as metres at her head depth
    max_side: int = 1000          # analysis resolution: longer image side (1080x1920 phones are halved)
    max_out_votes: int = 0        # cameras allowed to carve a hull voxel and still keep it
    min_in_votes: int = 3
    inside_m: float = 0.005       # hull voxels deeper than this inside the posed SMPL-X surface are body
    occl_m: float = 0.03          # a voxel this far behind her rendered body is hidden in that view
    lead_margin_m: float = 0.08   # a lead pixel carves only voxels this far in front of his rendered body (fit slack)
    hair_min_votes: int = 2
    hair_frac: float = 0.67       # hair votes >= 2 x non-hair votes
    on_skin_m: float = 0.03       # within this of her skin, one hair vote and no body vote suffice (hair lying on her)
    head_link_m: float = 0.22     # kept hair components must come this close to the head joint
    link_gap: int = 2             # voxels: hair pieces separated by gaps up to this join the head component
    arm_clear_m: float = 0.04     # voxels this close to her arm / hand skin are never hair
    shift_px: int = 12            # registration search range (analysis px)
    min_view_iou: float = 0.45    # views whose rendered body overlaps the body mask less than this only carve
    min_carve_views: int = 7      # frames carved by fewer cameras are left out of the shape / motion statistics
    min_frame_recall: float = 0.8  # ... and frames whose 3D hair covers less of the 2D hair masks (median view)
    step: int = 1                 # every n-th grid frame for the hull
    colour_step: int = 0          # every n-th camera frame for colour; 0 = auto (colour_views_per_cam, >= min step)
    colour_views_per_cam: int = 30
    colour_min_step: int = 6
    root_m: float = 0.11          # colour zones: distance from the projected head centre (m at head depth)
    mid_m: float = 0.18
    tip_frac: float = 0.80        # tips: d >= tip_frac x (p99 distance of the view), when that is >= min_len_m
    min_len_m: float = 0.30
    top_albedo: float = 0.80      # assumed albedo of her white top (white balance -> albedo-like colour)
    calm_yaw_dps: float = 90.0
    calm_speed_mps: float = 0.6
    turn_yaw_dps: float = 180.0
    trail_noise_deg: float = 15.0  # |offset-corrected trailing median| below this is not resolved (hair-mass azimuth noise)
    trail_min_per_dir: int = 6     # turn frames needed in each turn direction
    lag_max_rms_deg: float = 20.0  # a phase-lag fit with a larger residual RMS is not resolved
    bootstrap: int = 2000
    mp: bool = True
    device: str = "cuda"          # the hull only; everything else runs on the CPU
    isolate_gpu: bool = True      # local CUDA hull in a child process (its CUDA context and VRAM die with it)
    threads: int = 4              # CPU threads (torch, OpenCV, KD-trees): the workstation is shared
    review: bool = True           # write the review images (the pod run skips them, the local re-render draws them)
    cams: tuple = ()


# ------------------------------------------------------------------------------------------------ colour helpers

def srgb_to_linear(x):
    x = np.asarray(x, float)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    x = np.clip(np.asarray(x, float), 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


SRGB8_TO_LINEAR = srgb_to_linear(np.arange(256) / 255.0).astype(np.float32)  # 8-bit lookup: decode only the pixels used


def bgr8_to_linear_rgb(px_bgr: np.ndarray) -> np.ndarray:
    """(..., 3) uint8 BGR pixels -> float32 linear sRGB (RGB order) through the 8-bit lookup table."""
    return SRGB8_TO_LINEAR[np.asarray(px_bgr, np.uint8)[..., ::-1]]


def luminance(rgb):
    return np.asarray(rgb, float) @ np.array([0.2126, 0.7152, 0.0722])


def linear_to_lab(rgb):
    """Linear sRGB (D65) -> CIELAB."""
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = np.asarray(rgb, float) @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def delta_e76(a, b) -> float:
    return float(np.linalg.norm(linear_to_lab(a) - linear_to_lab(b)))


def colour_entry(rgb_lin, n: int | None = None) -> dict:
    """JSON colour record: linear sRGB, 8-bit sRGB, hex, HSV (of sRGB), CIELAB."""
    lin = np.clip(np.asarray(rgb_lin, float), 0, None)
    s8 = np.round(linear_to_srgb(np.clip(lin, 0, 1)) * 255).astype(int)
    hsv = cv2.cvtColor(s8.astype(np.uint8)[None, None], cv2.COLOR_RGB2HSV_FULL)[0, 0]
    out = {"linear_srgb": [round(float(v), 4) for v in lin], "srgb8": [int(v) for v in s8],
           "hex": "#%02X%02X%02X" % tuple(int(v) for v in s8),
           "hsv": [round(float(hsv[0]) * 360 / 256, 1), round(float(hsv[1]) / 255, 3), round(float(hsv[2]) / 255, 3)],
           "lab": [round(float(v), 1) for v in linear_to_lab(lin)], "clipped": bool((lin > 1).any())}
    if n is not None:
        out["n_px"] = int(n)
    return out


def white_balance(rgb_lin: np.ndarray, white_lin: np.ndarray, albedo: float) -> np.ndarray:
    """Per-channel gains that map the measured white to a neutral grey of the given albedo."""
    return np.asarray(rgb_lin, float) * (albedo / np.maximum(np.asarray(white_lin, float), 1e-4))


# ------------------------------------------------------------------------------------------------ geometry helpers

def rodrigues(rv: np.ndarray) -> np.ndarray:
    """(..., 3) axis-angle -> (..., 3, 3)."""
    rv = np.asarray(rv, float)
    th = np.linalg.norm(rv, axis=-1, keepdims=True)
    k = rv / np.maximum(th, 1e-12)
    K = np.zeros(rv.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -k[..., 2], k[..., 1], -k[..., 0]
    K[..., 1, 0], K[..., 2, 0], K[..., 2, 1] = k[..., 2], -k[..., 1], k[..., 0]
    th = th[..., None]
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def chain_rotations(global_orient: np.ndarray, body_pose: np.ndarray) -> dict[int, np.ndarray]:
    """Global rotations (T,3,3) of the SMPL-X joints on the pelvis -> head chain."""
    R = {0: rodrigues(global_orient)}
    prev = 0
    for j in CHAIN_HEAD[1:]:
        R[j] = R[prev] @ rodrigues(body_pose[:, 3 * (j - 1): 3 * j])
        prev = j
    return R


def azimuth(v: np.ndarray) -> np.ndarray:
    """Heading angle (rad) of world vectors about +Y: atan2(x, z)."""
    return np.arctan2(v[..., 0], v[..., 2])


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def phase_lag(phi_lead: np.ndarray, phi_follow: np.ndarray, dt: float, max_s: float = 0.4,
              offset: float = np.pi, mask: np.ndarray | None = None) -> tuple[float, float]:
    """Time shift tau >= 0 that best aligns phi_follow(t) with phi_lead(t - tau) + offset (circular RMS over the
    frames in mask); returns (tau seconds, rms degrees). A trailing hair mass has tau > 0."""
    n = len(phi_lead)
    best = (0.0, np.inf)
    m = np.ones(n, bool) if mask is None else np.asarray(mask, bool)
    for s in range(0, int(round(max_s / dt)) + 1):
        a, b = phi_lead[: n - s], phi_follow[s:]
        mm = m[s:] & np.isfinite(a) & np.isfinite(b)
        if mm.sum() < 5:
            continue
        r = float(np.sqrt(np.mean(wrap(b[mm] - a[mm] - offset) ** 2)))
        if r < best[1]:
            best = (s * dt, r)
    return best[0], float(np.degrees(best[1]))


def back_envelope_length(rest_v: np.ndarray, y_top: float, y_bottom: float, x_half: float = 0.05,
                         bin_m: float = 0.01, offset_m: float = 0.01) -> float:
    """Arc length (m) of the posterior envelope of the rest mesh at the back midline from y_top down to y_bottom: hair
    hanging straight down the back drapes over the occiput and the shoulder blades and bridges the hollows, so the
    profile is the convex (most posterior) hull of the midline section, offset outward by the hair layer."""
    sel = np.abs(rest_v[:, 0]) < x_half
    v = rest_v[sel]
    if y_bottom >= y_top or len(v) < 10:
        return float("nan")
    ys = np.arange(y_top - bin_m, y_bottom - bin_m, -bin_m)
    top = v[np.argmax(v[:, 1])]  # the crown vertex itself: back strands start on top of the skull
    pts = [(float(top[1]) + offset_m, float(top[2]))]
    for y in ys:
        b = v[np.abs(v[:, 1] - y) < bin_m]
        if len(b):
            pts.append((y, float(b[:, 2].min()) - offset_m))
    if len(pts) < 3:
        return float("nan")
    P = np.array(pts)  # (y, z) with y descending
    # lower convex hull in z (most posterior = smallest z) as a function of y: monotone chain
    hull = []
    for p in P:
        while len(hull) >= 2:
            (y1, z1), (y2, z2) = hull[-2], hull[-1]
            cross = (y2 - y1) * (p[1] - z1) - (z2 - z1) * (p[0] - y1)
            if cross >= 0:  # keeps the chain convex towards -z
                hull.pop()
            else:
                break
        hull.append(tuple(p))
    H = np.array(hull)
    return float(np.sum(np.linalg.norm(np.diff(H, axis=0), axis=1)))


# ------------------------------------------------------------------------------------------------ context

def follow_meta(take: Take) -> dict:
    ds = take.cfg.get("dancers", {})
    d = ds.get(ROLE, {})
    return {"name": d.get("name", ROLE), "stature_m": d.get("stature_m"), "description": d.get("description", ""),
            "pid": int(d.get("pid", DEFAULT_PID)), "lead_pid": int(ds.get("lead", {}).get("pid", DEFAULT_LEAD_PID))}


def face_python() -> tuple[str | None, dict]:
    """Interpreter for hair_segment_mp.py (MediaPipe lives outside the torch venv) + extra environment.
    DANCECAP_FACE_PYTHON (a path or a command on PATH) overrides .venv-face; DANCECAP_FACE_PYTHONPATH is prepended to
    its PYTHONPATH (the pod installs mediapipe with pip --target)."""
    env = {}
    extra = os.environ.get("DANCECAP_FACE_PYTHONPATH")
    if extra:
        env["PYTHONPATH"] = extra + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")
    exe = os.environ.get("DANCECAP_FACE_PYTHON")
    if exe:
        return (exe if Path(exe).exists() else shutil.which(exe)), env
    return (str(FACE_VENV) if FACE_VENV.exists() else None), env


def lower_priority() -> str:
    """Batch work yields to interactive programs on the shared workstation: BELOW_NORMAL priority class on Windows
    (inherited by the MediaPipe and hull child processes), nice +5 elsewhere (not on pods)."""
    try:
        if os.name == "nt":
            import ctypes
            k = ctypes.windll.kernel32
            k.SetPriorityClass(k.GetCurrentProcess(), 0x00004000)  # BELOW_NORMAL_PRIORITY_CLASS
            return "below_normal"
        if not (os.environ.get("RUNPOD_POD_ID") or os.environ.get("DANCECAP_POD")):
            os.nice(5)
            return "nice+5"
    except Exception as e:  # never fatal
        return f"unchanged ({type(e).__name__})"
    return "normal (pod)"


def set_threads(n: int) -> None:
    """Bound this process's CPU threads (torch, OpenCV); numpy's BLAS follows OMP/MKL_NUM_THREADS set by main()."""
    n = max(1, int(n))
    cv2.setNumThreads(n)
    try:
        import torch
        torch.set_num_threads(n)
    except Exception:
        pass


def default_portrait(take: Take) -> Path | None:
    name = follow_meta(take)["name"]
    root = (PACKAGE_ROOT / "takes" / take.cfg["source_dir"]).resolve()
    for ext in (".jpg", ".jpeg", ".png"):
        p = root / f"{name.lower()}{ext}"
        if p.exists():
            return p
    return None


@dataclass
class Context:
    take: Take
    work: Path
    cfg: HairRefConfig
    tracks: dict
    body: object                 # avatar_uv.Body (posed follow, head fix applied when adopted)
    lead_body: object            # avatar_uv.Body of the lead (his depth decides what his mask can carve)
    faces: np.ndarray
    rest_v: np.ndarray           # shaped T-pose vertices (sole at y = 0)
    rest_n: np.ndarray
    rest_j: np.ndarray
    rot: dict                    # chain rotations (T,3,3)
    arm_v: np.ndarray            # (V,) bool: arm / hand vertices (rest T-pose: lateral of the shoulder joints)
    scale: dict                  # world -> true metres
    swaps: dict
    pid: int = DEFAULT_PID       # mask ids of the follow / lead (takes/<take>.toml [dancers.*].pid)
    lead_pid: int = DEFAULT_LEAD_PID
    info: dict = field(default_factory=dict)


def load_context(take: Take, cfg: HairRefConfig) -> Context:
    from .avatar_texture import _roles_swap, load_bodies, load_smplx, model_dir_for
    from .avatar_uv import model_npz, vertex_normals
    from .cameras.world import SOLE_M
    from .mamma_dyn import load_tracks
    work = take.work
    tracks = load_tracks(work)
    if cfg.cams:
        tracks = {c: v for c, v in tracks.items() if c in cfg.cams}
    bodies, binfo = load_bodies(work, "cpu")
    body = bodies[ROLE]
    d = load_smplx(work, ROLE)
    m = np.load(model_npz(model_dir_for(d)), allow_pickle=True)
    faces = np.asarray(m["f"], np.int64)
    betas = np.asarray(d["betas"], float).reshape(-1)
    nb = len(betas)
    rest = np.asarray(m["v_template"], float) + np.asarray(m["shapedirs"], float)[:, :, :nb] @ betas
    rest_j = np.asarray(m["J_regressor"], float) @ rest
    y0 = rest[:, 1].min()
    rest[:, 1] -= y0
    rest_j[:, 1] -= y0
    bp = np.asarray(d["body_pose"], float).copy()
    hf = work / "avatar" / "head_fix.npz"
    if binfo[ROLE].get("head_fix") and hf.exists():
        with np.load(hf) as z:
            bp[:, 33:36] = z[f"{ROLE}_neck_pose"]
    rot = chain_rotations(np.asarray(d["global_orient"], float), bp)
    meta = follow_meta(take)
    H = float(rest[:, 1].max())
    target = (meta["stature_m"] + SOLE_M) if meta["stature_m"] else None
    scale = {"rest_height_world_m": round(H, 4), "stature_m": meta["stature_m"], "sole_m": SOLE_M,
             "target_m": target, "true_per_world": round(target / H, 5) if target else 1.0,
             "note": "true metres = world metres x true_per_world (stature + sole over the fitted SMPL-X rest height)"}
    arm_v = np.abs(rest[:, 0]) > np.abs(rest_j[[J_LSH, J_RSH], 0]).mean() + 0.04
    return Context(take, work, cfg, tracks, body, bodies["lead"], faces, rest, vertex_normals(rest, faces), rest_j, rot,
                   arm_v, scale, _roles_swap(work), meta["pid"], meta["lead_pid"], {"bodies": binfo})


def _mask_path(ctx: Context, cam: str, kind: str, i: int, pid: int | None = None) -> Path:
    """dancers/<cam>/<kind>/mask_<frame>_<pid>.png (kind = masks | body | hair); pid defaults to the follow."""
    return ctx.work / "dancers" / cam / kind / f"mask_{i:04d}_{(ctx.pid if pid is None else pid):02d}.png"


# ------------------------------------------------------------------------------------------------ views

def _read(p: Path) -> np.ndarray | None:
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    return None if m is None else m > 127


def nearest_frame(track, t: float) -> tuple[int, float] | None:
    i = int(np.argmin(np.abs(track.t - t)))
    dt = float(track.t[i] - t)
    return (i, dt) if abs(dt) <= 0.6 * track.frame_dt else None


def load_view(ctx: Context, cam: str, i: int) -> dict | None:
    """Masks of one camera frame at the analysis scale + scaled intrinsics."""
    tr = ctx.tracks[cam]
    full = _read(_mask_path(ctx, cam, "masks", i))
    if full is None or not full.any():
        return None
    lead = _read(_mask_path(ctx, cam, "masks", i, ctx.lead_pid))
    hair = _read(_mask_path(ctx, cam, "hair", i))
    bodym = _read(_mask_path(ctx, cam, "body", i))
    H, W = full.shape
    s = min(1.0, ctx.cfg.max_side / max(H, W))
    size = (int(round(W * s)), int(round(H * s)))

    def rs(m):
        if m is None:
            return np.zeros((size[1], size[0]), bool)
        return cv2.resize(m.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool) if s < 1 else m
    K = tr.K[i].copy()
    K[:2] *= s
    return {"cam": cam, "i": i, "s": s, "H": size[1], "W": size[0], "K": K, "R": tr.R[i], "t": tr.tvec[i],
            "k1": float(tr.k1[i]), "full": rs(full), "lead": rs(lead), "hair": rs(hair),
            "body": rs(bodym if bodym is not None else full), "swap": ctx.swaps.get(cam, {}).get(i, 0.0)}


def best_shift(sil: np.ndarray, target: np.ndarray, r: int) -> tuple[tuple[int, int], float, float]:
    """(dx, dy) maximising IoU(sil shifted by (dx, dy), target); returns shift, IoU there, IoU at (0, 0)."""
    ys, xs = np.nonzero(sil | target)
    if len(ys) == 0:
        return (0, 0), 0.0, 0.0
    y0, y1 = max(0, ys.min() - r), min(sil.shape[0], ys.max() + r + 1)
    x0, x1 = max(0, xs.min() - r), min(sil.shape[1], xs.max() + r + 1)
    a = sil[y0:y1, x0:x1]
    b = target[y0:y1, x0:x1]
    f = max(1, int(np.ceil((y1 - y0) / 200)))

    def iou_at(A, B, dx, dy):
        h, w = A.shape
        As = np.zeros_like(A)
        ys0, ys1 = max(0, dy), min(h, h + dy)
        xs0, xs1 = max(0, dx), min(w, w + dx)
        As[ys0:ys1, xs0:xs1] = A[ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx]
        inter = np.count_nonzero(As & B)
        uni = np.count_nonzero(As | B)
        return inter / uni if uni else 0.0
    A2, B2 = a[::f, ::f], b[::f, ::f]
    rr = max(1, r // f)
    best, bi = (0, 0), -1.0
    for dy in range(-rr, rr + 1):
        for dx in range(-rr, rr + 1):
            v = iou_at(A2, B2, dx, dy)
            if v > bi:
                best, bi = (dx, dy), v
    cx, cy = best[0] * f, best[1] * f
    bi = -1.0
    for dy in range(cy - f, cy + f + 1):
        for dx in range(cx - f, cx + f + 1):
            if abs(dx) > r or abs(dy) > r:
                continue
            v = iou_at(a, b, dx, dy)
            if v > bi:
                best, bi = (dx, dy), v
    return best, float(bi), float(iou_at(a, b, 0, 0))


def _dilate(m: np.ndarray, r: float) -> np.ndarray:
    k = 2 * int(np.ceil(r)) + 1
    return cv2.dilate(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(bool)


# ------------------------------------------------------------------------------------------------ 3D hair hull

def onbody_hits(ctx: Context, v: dict, fid: np.ndarray, shift: tuple[int, int]) -> dict:
    """Hair pixels of one view ray-cast onto her rendered body (single-view tip level): the REST-mesh heights of the
    hit faces, split into back-facing and front-facing body surface (rest normals). Pixels of the lead are skipped."""
    dx, dy = shift
    ys, xs = np.nonzero(v["hair"] & ~v["lead"])
    u, w = xs - dx, ys - dy
    ok = (u >= 0) & (u < v["W"]) & (w >= 0) & (w < v["H"])
    f = fid[w[ok], u[ok]]
    f = f[f >= 0]
    out = {"hair_px": int(len(ys)), "onbody_px": int(len(f))}
    if len(f) < 20:
        return out
    tri = ctx.faces[f]
    y = ctx.rest_v[tri, 1].mean(1)
    nz = ctx.rest_n[tri, 2].mean(1)
    below_neck = y < ctx.rest_j[J_NECK, 1]
    for name, sel in (("back", (nz < -0.3) & below_neck), ("front", (nz > 0.3) & below_neck)):
        if sel.sum() >= 20:
            out[f"{name}_px"] = int(sel.sum())
            out[f"{name}_rest_y_p2"] = round(float(np.percentile(y[sel], 2)), 4)
            out[f"{name}_rest_y_p50"] = round(float(np.percentile(y[sel], 50)), 4)
    return out


def hull_frame(ctx: Context, k: int, torch, dev, debug: bool = False) -> dict:
    """Hair voxels (world) of grid frame k + per-view diagnostics (see module doc, step 1)."""
    from scipy import ndimage
    from scipy.spatial import cKDTree
    from .avatar_raster import rasterize
    from .avatar_texture import project
    from .avatar_uv import vertex_normals
    cfg = ctx.cfg
    t = float(ctx.body.t[k])
    V = ctx.body.verts[k].astype(np.float64)
    head = ctx.body.joints[k, J_HEAD]
    vs = cfg.voxel_m
    xs = np.arange(-cfg.box_xz_m, cfg.box_xz_m + 1e-9, vs)
    ys = np.arange(-cfg.box_down_m, cfg.box_up_m + 1e-9, vs)
    shape = (len(xs), len(ys), len(xs))
    gx, gy, gz = np.meshgrid(xs + head[0], ys + head[1], xs + head[2], indexing="ij")
    G = torch.tensor(np.stack([gx.ravel(), gy.ravel(), gz.ravel()], 1), dtype=torch.float32, device=dev)
    Vt = torch.tensor(V, dtype=torch.float32, device=dev)
    VLt = torch.tensor(ctx.lead_body.verts[k], dtype=torch.float32, device=dev)
    Ft = torch.tensor(ctx.faces, device=dev)
    N = G.shape[0]
    out_v = torch.zeros(N, dtype=torch.int16, device=dev)
    in_v = torch.zeros(N, dtype=torch.int16, device=dev)
    views, diag = [], []
    n_carve = 0
    for cam, tr in ctx.tracks.items():
        nf = nearest_frame(tr, t)
        if nf is None:
            continue
        i, dt = nf
        if not bool(tr.ok[i]):
            diag.append({"cam": cam, "i": i, "skip": "camera not ok"})
            continue
        v = load_view(ctx, cam, i)
        if v is None:
            diag.append({"cam": cam, "i": i, "skip": "no follow mask"})
            continue
        if v["swap"] > 0.5:
            diag.append({"cam": cam, "i": i, "skip": f"swap_prob {v['swap']:.2f}"})
            continue
        K = torch.tensor(v["K"], dtype=torch.float32, device=dev)
        R = torch.tensor(v["R"], dtype=torch.float32, device=dev)
        tv = torch.tensor(v["t"], dtype=torch.float32, device=dev)
        pxv, zv, _ = project(torch, Vt, K, R, tv, v["k1"])
        zb, fid = rasterize(torch, pxv, zv, Ft, v["H"], v["W"])
        fid_np = fid.cpu().numpy()
        sil = fid_np >= 0
        # the rendered silhouette ignores the partner: compare only where the lead's mask is absent
        free = ~_dilate(v["lead"], 2.0)
        (dx, dy), iou, iou0 = best_shift(sil & free, v["body"] & free, cfg.shift_px)
        zh = float((v["R"] @ head + v["t"])[2])
        rpx = max(cfg.tol_px, cfg.tol_m * float(v["K"][0, 0]) / max(zh, 0.5))
        d = {"cam": cam, "i": i, "dt_ms": round(dt * 1000, 1), "shift_px": [dx, dy], "iou": round(iou, 3),
             "iou_unshifted": round(iou0, 3), "dilate_px": round(rpx, 2), "scale": round(v["s"], 3)}
        registered = iou >= cfg.min_view_iou
        if not registered:
            # mostly hidden behind the lead (or a poor fit): no reliable registration -> unshifted, used to carve
            # (background and lead-depth tests are still valid) but casts no hair votes
            dx = dy = 0
            d["shift_px"] = [0, 0]
            d["carve_only"] = f"silhouette IoU {iou:.2f} < {cfg.min_view_iou}"
        else:
            d.update(onbody_hits(ctx, v, fid_np, (dx, dy)))
        diag.append(d)
        Fd = torch.tensor(_dilate(v["full"], rpx), device=dev)
        Ld = torch.tensor(_dilate(v["lead"], rpx), device=dev)
        pxl, zl, _ = project(torch, VLt, K, R, tv, v["k1"])
        zbl, _ = rasterize(torch, pxl, zl, Ft, v["H"], v["W"])
        px, z, _ = project(torch, G, K, R, tv, v["k1"])
        u = torch.round(px[:, 0] + dx).long()
        w = torch.round(px[:, 1] + dy).long()
        inside = (z > 0.1) & (u >= 0) & (u < v["W"]) & (w >= 0) & (w < v["H"])
        uu, ww = u.clamp(0, v["W"] - 1), w.clamp(0, v["H"] - 1)
        u0 = torch.round(px[:, 0]).long().clamp(0, v["W"] - 1)
        w0 = torch.round(px[:, 1]).long().clamp(0, v["H"] - 1)
        f_in = Fd[ww, uu] & inside
        # a lead pixel hides only what is at or behind his (rendered) body: anything of hers clearly in front of him
        # would have shown up in her mask, so such voxels are carved
        l_in = Ld[ww, uu] & inside & ~(z < zbl[w0, u0] - cfg.lead_margin_m)
        out_v += (inside & ~f_in & ~l_in).to(torch.int16)
        in_v += f_in.to(torch.int16)
        if registered:
            views.append((v, K, R, tv, zb, (dx, dy), rpx))
        n_carve += 1
    hull = (out_v <= cfg.max_out_votes) & (in_v >= cfg.min_in_votes)
    hidx = torch.nonzero(hull).squeeze(1)
    res = {"k": k, "t": t, "views": diag, "n_views": len(views), "n_carve": n_carve, "n_hull": int(len(hidx)), "hair": np.zeros((0, 3), np.float32),
           "n_cand": 0}
    if len(hidx) == 0 or n_carve < cfg.min_in_votes:
        return res
    P = G[hidx].cpu().numpy().astype(np.float64)
    tree = cKDTree(V)
    dist, nn = tree.query(P, workers=cfg.threads)
    nrm = vertex_normals(V, ctx.faces)
    signed = np.einsum("ij,ij->i", P - V[nn], nrm[nn])
    # on or outside her skin (hair lying on her back is kept, the votes decide), not on her arms: the arm zones of the
    # hair masks protect her arms, but a hair vote from a view where an arm crosses her hair would glue them on
    cand = (signed > -cfg.inside_m) & ~(ctx.arm_v[nn] & (dist < cfg.arm_clear_m))
    cidx = hidx[torch.tensor(np.nonzero(cand)[0], device=dev)]
    res["n_cand"] = int(len(cidx))
    if len(cidx) == 0:
        return res
    Pc = G[cidx]
    hv = torch.zeros(len(cidx), dtype=torch.int16, device=dev)
    nv = torch.zeros(len(cidx), dtype=torch.int16, device=dev)
    for v, K, R, tv, zb, (dx, dy), rpx in views:
        px, z, _ = project(torch, Pc, K, R, tv, v["k1"])
        u0 = torch.round(px[:, 0]).long().clamp(0, v["W"] - 1)
        w0 = torch.round(px[:, 1]).long().clamp(0, v["H"] - 1)
        u = torch.round(px[:, 0] + dx).long()
        w = torch.round(px[:, 1] + dy).long()
        inside = (z > 0.1) & (u >= 0) & (u < v["W"]) & (w >= 0) & (w < v["H"])
        uu, ww = u.clamp(0, v["W"] - 1), w.clamp(0, v["H"] - 1)
        hidden = zb[w0, u0] < z - cfg.occl_m
        Hd = torch.tensor(_dilate(v["hair"], 1.0), device=dev)
        Fm = torch.tensor(v["full"], device=dev)
        Lm = torch.tensor(v["lead"], device=dev)
        # confident body: her mask minus the (dilated) hair, eroded by the tolerance - a ray grazing her silhouette
        # (thin hair layer on her back seen from the side) or the hair/body border casts no vote
        conf_body = ~_dilate(~(v["full"] & ~_dilate(v["hair"], rpx)), rpx)
        Bc = torch.tensor(conf_body, device=dev)
        ok = inside & ~hidden & ~(Lm[ww, uu] & ~Fm[ww, uu])
        h = ok & Hd[ww, uu]
        hv += h.to(torch.int16)
        nv += (ok & Bc[ww, uu] & ~Hd[ww, uu]).to(torch.int16)
    hvn, nvn = hv.cpu().numpy(), nv.cpu().numpy()
    sgn_c = signed[cand]
    is_hair = ((hvn >= cfg.hair_min_votes) & (hvn >= cfg.hair_frac * (hvn + nvn))) |         ((hvn >= 1) & (nvn == 0) & (sgn_c < cfg.on_skin_m))
    occ = np.zeros(N, bool)
    occ[cidx.cpu().numpy()[is_hair]] = True
    occ = occ.reshape(shape)
    if debug:
        Pall_dbg = np.stack([gx, gy, gz], -1).reshape(-1, 3)
        res["_dbg"] = {"G": Pall_dbg, "hull": hull.cpu().numpy(), "cand": cidx.cpu().numpy(), "hv": hvn, "nv": nvn,
                        "out": out_v.cpu().numpy(), "inn": in_v.cpu().numpy(), "signed": signed, "hidx": hidx.cpu().numpy(),
                        "pre_cc": occ.copy()}
    # components are linked across gaps of up to link_gap voxels (strands lying flat at the nape or hidden behind an
    # arm leave holes between the scalp and the hanging hair)
    bridged = ndimage.binary_dilation(occ, structure=np.ones((3, 3, 3)), iterations=cfg.link_gap) if cfg.link_gap else occ
    lab, n = ndimage.label(bridged, structure=np.ones((3, 3, 3)))
    if n:
        Pall = np.stack([gx, gy, gz], -1)
        near = np.linalg.norm(Pall - head, axis=-1) < cfg.head_link_m
        keep = np.unique(lab[near & occ & (lab > 0)])
        occ = occ & np.isin(lab, keep[keep > 0])
        res["hair"] = Pall[occ].astype(np.float32)
        res["n_components"] = int(n)
        res["kept_components"] = int(len(keep[keep > 0]))
    res["hair_votes_median"] = float(np.median(hvn[is_hair])) if is_hair.any() else 0.0
    res["recall"] = frame_recall(res["hair"], views, torch, dev, cfg.voxel_m)
    return res


def frame_recall(P: np.ndarray, views: list, torch, dev, voxel_m: float) -> float | None:
    """Median over the registered views of the share of hair-mask pixels (analysis scale) covered by the re-projected
    hair voxels (voxel-sized disks): does the 3D hair explain what each camera sees?"""
    from .avatar_texture import project
    if len(P) == 0 or not views:
        return None
    Pt = torch.tensor(np.asarray(P), dtype=torch.float32, device=dev)
    out = []
    for v, K, R, tv, zb, (dx, dy), rpx in views:
        hm = v["hair"]
        if hm.sum() < 30:
            continue
        px, z, _ = project(torch, Pt, K, R, tv, v["k1"])
        px = px.cpu().numpy() + np.array([dx, dy])
        z = z.cpu().numpy()
        ok = (z > 0.1) & (px[:, 0] >= 0) & (px[:, 0] < v["W"]) & (px[:, 1] >= 0) & (px[:, 1] < v["H"])
        if not ok.any():
            out.append(0.0)
            continue
        m = np.zeros(hm.shape, np.uint8)
        q = px[ok].astype(int)
        m[q[:, 1], q[:, 0]] = 1
        r = max(1, int(round(0.5 * voxel_m * float(K[0, 0].item()) / float(np.median(z[ok])))))
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))).astype(bool)
        out.append(float(np.count_nonzero(hm & m) / hm.sum()))
    return round(float(np.median(out)), 3) if out else None


@contextlib.contextmanager
def gpu_section(device: str, label: str, stats: dict | None = None):
    """Yields (torch, dev). On CUDA the block holds work/gpu.lock (dancecap.gpu.exclusive) and, before the lock is
    released, synchronises and returns PyTorch's cached VRAM (torch.cuda.empty_cache), recording the peak in stats;
    every tensor of the block must be dropped by then. device 'cpu' (or no CUDA) runs without the lock."""
    import torch
    dev = device if device == "cpu" or torch.cuda.is_available() else "cpu"
    st = stats if stats is not None else {}
    t0 = time.time()
    if dev == "cpu":
        with torch.no_grad():
            yield torch, "cpu"
        st.update({"device": "cpu", "seconds": round(time.time() - t0, 1)})
        return
    from .gpu import exclusive
    with exclusive(f"hair_reference {label}"):
        t1 = time.time()
        torch.cuda.reset_peak_memory_stats()
        try:
            with torch.no_grad():
                yield torch, dev
        finally:
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_reserved() / 2 ** 20
            torch.cuda.empty_cache()
            st.update({"device": dev, "seconds": round(time.time() - t1, 1), "lock_wait_s": round(t1 - t0, 1),
                       "peak_reserved_mib": round(peak), "reserved_after_release_mib":
                           round(torch.cuda.memory_reserved() / 2 ** 20)})


def run_hull(ctx: Context, frames: list[int], torch, dev) -> list[dict]:
    out = []
    t0 = time.time()
    for n, k in enumerate(frames):
        out.append(hull_frame(ctx, k, torch, dev))
        if n % 20 == 0:
            r = out[-1]
            print(f"[hair_ref] hull frame {k} ({n + 1}/{len(frames)}): {r['n_views']} views, hull "
                  f"{r['n_hull']}, hair {len(r['hair'])} voxels ({time.time() - t0:.0f} s)", flush=True)
    return out


def hull_stage(ctx: Context, hull_p: Path, cons_p: Path, frames: list[int]) -> dict:
    """The GPU part: hull of every listed grid frame + its re-projection check, in one gpu_section; writes hull.npz
    and consistency.json. Returns the compute record."""
    stats: dict = {}
    with gpu_section(ctx.cfg.device, "hull", stats) as (torch, dev):
        hulls = run_hull(ctx, frames, torch, dev)
        save_hull(hull_p, hulls, ctx.cfg)
        cons = projection_consistency(ctx, hulls, torch, dev)
        save_consistency(cons_p, hull_p, cons)
    stats["frames"] = len(frames)
    return stats


def hull_in_child(take: Take, cfg: HairRefConfig) -> dict:
    """Runs hull_stage in a child process (python -m dancecap.hair_reference <take> --only-hull): its CUDA context
    and every byte of VRAM are gone when it exits, while this process goes on with minutes of CPU work."""
    args = [sys.executable, "-m", "dancecap.hair_reference", take.name, "--only-hull", "--step", str(cfg.step),
            "--device", cfg.device, "--threads", str(cfg.threads)]
    if cfg.cams:
        args += ["--cams", *cfg.cams]
    t0 = time.time()
    r = subprocess.run(args, cwd=str(PACKAGE_ROOT))
    if r.returncode != 0:
        raise SystemExit(f"[hair_ref] hull child process failed (rc {r.returncode})")
    p = take.work / "hair_ref" / "hull_compute.json"
    st = json.loads(p.read_text()) if p.exists() else {}
    st["child_wall_s"] = round(time.time() - t0, 1)
    return st


def save_hull(path: Path, hulls: list[dict], cfg: HairRefConfig) -> None:
    ks = np.concatenate([np.full(len(h["hair"]), h["k"], np.int32) for h in hulls]) if hulls else np.zeros(0, np.int32)
    P = np.concatenate([h["hair"] for h in hulls]) if hulls else np.zeros((0, 3), np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, k=ks, P=P, frames=np.array([h["k"] for h in hulls]), voxel_m=cfg.voxel_m,
                        views=json.dumps([{k: v for k, v in h.items() if k != "hair"} for h in hulls]))


def load_hull(path: Path) -> list[dict]:
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["views"]))
        P, ks = z["P"], z["k"]
    out = []
    for m in meta:
        m["hair"] = P[ks == m["k"]]
        out.append(m)
    return out


def _file_key(p: Path) -> str:
    st = p.stat()
    return f"{p.name}|{st.st_size}|{st.st_mtime_ns}"


def save_consistency(path: Path, hull_p: Path, cons: dict) -> None:
    path.write_text(json.dumps({**cons, "hull_key": _file_key(hull_p)}, indent=1))


def load_consistency(path: Path, hull_p: Path) -> dict | None:
    """The cached re-projection check, only when it was computed from this very hull.npz."""
    if not (path.exists() and hull_p.exists()):
        return None
    d = json.loads(path.read_text())
    return d if d.get("hull_key") == _file_key(hull_p) else None


# ------------------------------------------------------------------------------------------------ shape + motion

def head_kinematics(ctx: Context) -> dict:
    """Per grid frame: head/torso facing azimuths, yaw rate (deg/s), head speed, head tilt."""
    t = ctx.body.t
    J = ctx.body.joints
    Rh, Rt = ctx.rot[J_HEAD], ctx.rot[J_SPINE3]
    fwd_h = Rh @ np.array([0, 0, 1.0])
    fwd_t = Rt @ np.array([0, 0, 1.0])
    up_h = Rh @ np.array([0, 1.0, 0])
    phi_h = np.unwrap(azimuth(fwd_h))
    phi_t = np.unwrap(azimuth(fwd_t))
    yaw = np.gradient(phi_h, t)
    yaw_t = np.gradient(phi_t, t)
    speed = np.linalg.norm(np.gradient(J[:, J_HEAD], t, axis=0), axis=1)
    tilt = np.degrees(np.arccos(np.clip(up_h[:, 1], -1, 1)))
    return {"t": t, "phi_head": phi_h, "phi_torso": phi_t, "yaw_dps": np.degrees(yaw), "torso_yaw_dps": np.degrees(yaw_t),
            "head_speed": speed, "head_tilt_deg": tilt}


def rest_landmarks(ctx: Context) -> dict:
    """Heights (world m above the sole, REST mesh) of the body landmarks used to describe the tip level."""
    v, j = ctx.rest_v, ctx.rest_j
    crown = float(v[:, 1].max())
    sh = float(np.mean(j[[J_LSH, J_RSH], 1]))
    torso = v[(np.abs(v[:, 0]) < 0.25)]
    # waist = smallest horizontal convex perimeter of the torso between the pelvis and spine3
    from scipy.spatial import ConvexHull
    best = (np.inf, None)
    for y in np.arange(j[J_PELVIS, 1] + 0.04, j[J_SPINE3, 1], 0.01):
        s = torso[np.abs(torso[:, 1] - y) < 0.006][:, [0, 2]]
        if len(s) < 8:
            continue
        try:
            hl = ConvexHull(s)
        except Exception:
            continue
        per = hl.area  # 2D hull: area attribute = perimeter
        if per < best[0]:
            best = (per, float(y))
    stat = crown
    lm = {"crown": crown, "neck_joint": float(j[J_NECK, 1]), "shoulder_joints": sh,
          "spine3_joint": float(j[J_SPINE3, 1]), "spine2_joint": float(j[J_SPINE2, 1]),
          "spine1_joint": float(j[J_SPINE1, 1]), "pelvis_joint": float(j[J_PELVIS, 1]),
          "waist_min_perimeter": best[1] if best[1] is not None else float(0.62 * stat)}
    # scapula inferior angle (~T7) is not a SMPL-X joint: shoulder joint height - 0.135 x (stature / 1.65 m)
    lm["shoulder_blade_lower_tip_approx"] = sh - 0.135 * stat / 1.65
    lm["mid_back_approx"] = 0.5 * (lm["shoulder_blade_lower_tip_approx"] + lm["waist_min_perimeter"])
    return {k: round(float(x), 4) for k, x in lm.items()}


def describe_level(y: float, lm: dict) -> str:
    """Plain-language position of a rest height between the landmarks."""
    marks = [("crown", lm["crown"]), ("base of the neck", lm["neck_joint"]), ("shoulder line", lm["shoulder_joints"]),
             ("lower tip of the shoulder blades", lm["shoulder_blade_lower_tip_approx"]),
             ("mid-back", lm["mid_back_approx"]), ("waist", lm["waist_min_perimeter"]),
             ("hips (pelvis joint)", lm["pelvis_joint"])]
    for (na, ya), (nb, yb) in zip(marks[:-1], marks[1:]):
        if yb <= y <= ya:
            f = (ya - y) / max(ya - yb, 1e-6)
            if f < 0.2:
                return f"at the {na} (slightly below)"
            if f > 0.8:
                return f"just above the {nb}"
            return f"between the {na} and the {nb} ({round(100 * f)} % of the way down)"
    return "above the crown" if y > marks[0][1] else "below the hips"


def shape_metrics(ctx: Context, hulls: list[dict], kin: dict) -> tuple[dict, dict]:
    """Length / tip level / silhouette profile (calm frames) and motion (turn frames). Returns (metrics, per-frame)."""
    from scipy.spatial import cKDTree
    cfg = ctx.cfg
    sc = ctx.scale["true_per_world"]
    lm = rest_landmarks(ctx)
    rest_y = ctx.rest_v[:, 1]
    back_v = (ctx.rest_n[:, 2] < -0.3) & (rest_y < lm["neck_joint"])
    front_v = (ctx.rest_n[:, 2] > 0.3) & (rest_y < lm["neck_joint"])
    top_vertex = int(np.argmax(rest_y))
    pf = {k: [] for k in ("k", "t", "n", "tip_back", "tip_front", "drop", "r95", "r99", "dhead99", "psi", "phi",
                          "phi_torso", "yaw", "yaw_torso", "centroid_r", "speed", "tilt", "calm", "turn", "trail_deg")}
    prof_lat, prof_fwd, prof_off, prof_bins = [], [], [], np.arange(0.0, 1.05, 0.05)
    calm_clouds, turn_clouds = [], []
    excluded = []
    for h in hulls:
        k = h["k"]
        P = np.asarray(h["hair"], float)
        if len(P) < 30:
            excluded.append({"k": int(k), "t": round(float(ctx.body.t[k]), 3), "reason": f"{len(P)} hair voxels"})
            continue
        rc = h.get("recall")
        nc = h.get("n_carve", h.get("n_views", 0))
        if nc < cfg.min_carve_views or (rc is not None and rc < cfg.min_frame_recall):
            excluded.append(_exclusion_reason(h, float(ctx.body.t[k]), cfg))
            continue
        V = ctx.body.verts[k].astype(np.float64)
        J = ctx.body.joints[k]
        neck = J[J_NECK]
        head = J[J_HEAD]
        crown = V[top_vertex]
        dist, nn = cKDTree(V).query(P, workers=cfg.threads)
        on = dist <= 0.05 + cfg.voxel_m
        tb = rest_y[nn[on & back_v[nn]]]
        tf = rest_y[nn[on & front_v[nn]]]
        rel = P - neck
        rh = np.linalg.norm(rel[:, [0, 2]], axis=1)
        c = P.mean(0) - neck
        psi = float(np.arctan2(c[0], c[2]))
        yaw = float(kin["yaw_dps"][k])
        calm = abs(yaw) < cfg.calm_yaw_dps and kin["head_speed"][k] < cfg.calm_speed_mps
        turn = abs(yaw) > cfg.turn_yaw_dps
        behind = kin["phi_head"][k] + np.pi
        trail = float(np.degrees(-np.sign(yaw) * wrap(psi - behind))) if turn else np.nan  # raw (no calm offset)
        for key, val in (("k", k), ("t", float(ctx.body.t[k])), ("n", len(P)),
                         ("tip_back", float(np.percentile(tb, 3)) if len(tb) > 20 else np.nan),
                         ("tip_front", float(np.percentile(tf, 3)) if len(tf) > 20 else np.nan),
                         ("drop", float(crown[1] - np.percentile(P[:, 1], 2))),
                         ("r95", float(np.percentile(rh, 95))), ("r99", float(np.percentile(rh, 99))),
                         ("dhead99", float(np.percentile(np.linalg.norm(P - head, axis=1), 99))),
                         ("psi", psi), ("phi", float(kin["phi_head"][k])), ("phi_torso", float(kin["phi_torso"][k])),
                         ("yaw", yaw), ("yaw_torso", float(kin["torso_yaw_dps"][k])),
                         ("centroid_r", float(np.hypot(c[0], c[2]))),
                         ("speed", float(kin["head_speed"][k])), ("tilt", float(kin["head_tilt_deg"][k])),
                         ("calm", calm), ("turn", turn), ("trail_deg", trail)):
            pf[key].append(val)
        # torso frame: lateral = her left (horizontal), forward = lateral x up, origin = neck joint
        lat = J[J_LSH] - J[J_RSH]
        lat[1] = 0
        lat /= max(np.linalg.norm(lat), 1e-9)
        up = np.array([0, 1.0, 0])
        fwd = np.cross(lat, up)
        loc = np.stack([rel @ lat, P[:, 1] - crown[1], rel @ fwd], 1)  # (lateral, height below crown (<0), forward)
        if calm:
            calm_clouds.append((k, loc, np.stack([(V - neck) @ lat, V[:, 1] - crown[1], (V - neck) @ fwd], 1)))
            lat_w, fwd_w, off_w = [], [], []
            behind = (loc[:, 2] < 0) & on
            for b0 in prof_bins:
                s = (-loc[:, 1] >= b0) & (-loc[:, 1] < b0 + 0.05)
                if s.sum() >= 15:
                    lat_w.append(float(np.percentile(loc[s, 0], 97) - np.percentile(loc[s, 0], 3)))
                    fwd_w.append(float(np.percentile(loc[s, 2], 97) - np.percentile(loc[s, 2], 3)))
                else:
                    lat_w.append(np.nan)
                    fwd_w.append(np.nan)
                sb = s & behind
                off_w.append(float(np.percentile(dist[sb], 90)) if sb.sum() >= 10 else np.nan)
            prof_lat.append(lat_w)
            prof_fwd.append(fwd_w)
            prof_off.append(off_w)
        if turn:
            # top view in the head's heading frame: +y = facing, rotated so the head faces up
            ph = kin["phi_head"][k]
            ca, sa = np.cos(ph), np.sin(ph)
            x_r = rel[:, 0] * ca - rel[:, 2] * sa
            z_r = rel[:, 0] * sa + rel[:, 2] * ca
            turn_clouds.append((k, np.stack([x_r * np.sign(yaw), z_r], 1)))  # mirrored so every turn spins the same way
    pf = {k: np.asarray(v) for k, v in pf.items()}
    calm = pf["calm"].astype(bool) if len(pf["k"]) else np.zeros(0, bool)
    turn = pf["turn"].astype(bool) if len(pf["k"]) else np.zeros(0, bool)
    if calm.sum() < 8 and len(pf["k"]):
        score = np.abs(pf["yaw"]) / cfg.calm_yaw_dps + pf["speed"] / cfg.calm_speed_mps
        calm = score <= np.percentile(score, 25)
        pf["calm"] = calm
    tipb = pf["tip_back"][calm]
    tipb = tipb[np.isfinite(tipb)]
    tipf = pf["tip_front"][calm]
    tipf = tipf[np.isfinite(tipf)]

    def q(a, p):
        return float(np.percentile(a, p)) if len(a) else float("nan")
    hull_med, hull_long = q(tipb, 50), q(tipb, 25)  # the lower quartile = hair hanging fully (longest reading)
    # single-view cross-check: hair pixels ray-cast onto her fitted body in every registered view (no hull needed);
    # per calm frame the lowest back hit over the views (biased low by mask false positives, the hull is biased high
    # where a camera's misregistration carves hanging hair) -> the reference is the mean of the two medians
    ray = []
    for h in hulls:
        if h["k"] in set(pf["k"][calm].tolist()):
            vals = [v["back_rest_y_p2"] for v in h["views"] if "back_rest_y_p2" in v]
            if vals:
                ray.append(min(vals))
    ray = np.array(ray)
    ray_med, ray_long = q(ray, 50), q(ray, 25)
    both = [x for x in (hull_med, ray_med) if np.isfinite(x)]
    tip_med = float(np.mean(both)) if both else float("nan")
    tip_long = float(np.mean([x for x in (hull_long, ray_long) if np.isfinite(x)])) if both else float("nan")
    L_med = back_envelope_length(ctx.rest_v, lm["crown"], tip_med)
    L_long = back_envelope_length(ctx.rest_v, lm["crown"], tip_long)
    L_rng = [back_envelope_length(ctx.rest_v, lm["crown"], x) for x in (hull_med, ray_med) if np.isfinite(x)]
    with warnings.catch_warnings():  # bands below the longest hair are all-NaN by design
        warnings.simplefilter("ignore", RuntimeWarning)
        prof_lat = np.nanmedian(np.array(prof_lat), 0) if prof_lat else np.full(len(prof_bins), np.nan)
        prof_fwd = np.nanmedian(np.array(prof_fwd), 0) if prof_fwd else np.full(len(prof_bins), np.nan)
        prof_off = np.nanmedian(np.array(prof_off), 0) if prof_off else np.full(len(prof_bins), np.nan)
    sh_drop = lm["crown"] - lm["shoulder_joints"]
    ear_drop = 0.12 * lm["crown"] / 1.75  # ear level ~ 12 cm below the crown for a 1.75 m avatar
    prof_at = lambda d: float(np.interp(d, prof_bins + 0.025, np.nan_to_num(prof_lat, nan=np.nan)))
    rv = ctx.rest_v
    sl = np.abs(rv[:, 1] - lm["shoulder_joints"]) < 0.01
    sl &= np.abs(rv[:, 0]) < np.abs(ctx.rest_j[J_LSH, 0]) + 0.08
    body_sh_width = float(rv[sl, 0].max() - rv[sl, 0].min()) if sl.any() else float("nan")
    hs = (rv[:, 1] > lm["crown"] - 0.14) & (rv[:, 1] < lm["crown"] - 0.10)
    head_width = float(rv[hs, 0].max() - rv[hs, 0].min()) if hs.any() else float("nan")
    kk = pf["k"]
    w = lambda x: round(float(x), 4) if x is not None and np.isfinite(x) else None
    wt = lambda x: round(float(x) * sc, 4) if x is not None and np.isfinite(x) else None
    reg = registration_stats(hulls, cfg)
    metrics = {
        "frames": {"hull": int(len(hulls)), "with_hair": int(len(kk)), "calm": int(calm.sum()), "turn": int(turn.sum()),
                   "left_out_unreliable": int(len(excluded)), "left_out": excluded,
                   "left_out_summary": summarise_exclusions(excluded),
                   "registration": reg,
                   "reliability_rule": f">= {cfg.min_carve_views} carving cameras and 3D hair covering >= "
                                       f"{cfg.min_frame_recall:.0%} of the 2D hair mask (median view)",
                   "calm_rule": f"|head yaw rate| < {cfg.calm_yaw_dps} deg/s and head speed < {cfg.calm_speed_mps} m/s",
                   "turn_rule": f"|head yaw rate| > {cfg.turn_yaw_dps} deg/s"},
        "landmarks_rest_m_above_sole_world": lm,
        "length": {
            "tip_level_back_rest_m": {"median": w(tip_med), "longest_quartile": w(tip_long),
                                      "hull": {"median": w(hull_med), "longest_quartile": w(hull_long),
                                               "p75": w(q(tipb, 75)), "n_frames": int(len(tipb))},
                                      "single_view_raycast": {"median": w(ray_med), "longest_quartile": w(ray_long),
                                                              "p75": w(q(ray, 75)), "n_frames": int(len(ray))},
                                      "reference": "mean of the hull and ray-cast medians (their difference is the "
                                                   "measurement uncertainty)",
                                      "fraction_of_rest_height": w(tip_med / lm["crown"])},
            "tip_level_front_rest_m": {"median": w(q(tipf, 50)), "longest_quartile": w(q(tipf, 25)),
                                       "n_frames": int(len(tipf)),
                                       "bound": "LOWER bound of the front length (the rest height is an UPPER bound): "
                                                "the lead's arms and body and her own arms hide the front sections "
                                                "(occlusion-aware carving keeps nothing it cannot see voted hair, "
                                                "and voxels within 4 cm of her arms are never hair)"},
            "tip_drop_below_crown_rest": {"world_m": w(lm["crown"] - tip_med), "true_m": wt(lm["crown"] - tip_med),
                                          "longest_world_m": w(lm["crown"] - tip_long),
                                          "longest_true_m": wt(lm["crown"] - tip_long),
                                          "fraction_of_height": w((lm["crown"] - tip_med) / lm["crown"])},
            "tips_fall": describe_level(tip_med, lm) if np.isfinite(tip_med) else None,
            "tips_fall_longest": describe_level(tip_long, lm) if np.isfinite(tip_long) else None,
            "strand_length_crown_to_tip_back": {"world_m": w(L_med), "true_m": wt(L_med), "longest_world_m": w(L_long),
                                                "longest_true_m": wt(L_long),
                                                "range_true_m": [wt(min(L_rng)), wt(max(L_rng))] if L_rng else None,
                                                "method": "posterior-envelope arc length of the rest mesh, crown -> tip "
                                                          "level, +1 cm hair layer"},
            "posed_vertical_extent_below_crown": {"calm_median_world_m": w(q(pf["drop"][calm], 50)),
                                                  "calm_median_true_m": wt(q(pf["drop"][calm], 50))},
            "max_reach_from_head_centre_turns": {"world_m": w(q(pf["dhead99"][turn], 90)) if turn.any() else None,
                                                 "true_m": wt(q(pf["dhead99"][turn], 90)) if turn.any() else None,
                                                 "note": "p90 over turn frames of each frame's p99 distance from the "
                                                         "SMPL-X head joint: a lower bound of strand length + ~9 cm"},
        },
        "volume": {
            "profile_below_crown_m": [round(float(b), 3) for b in prof_bins],
            "width_lateral_world_m": [w(x) for x in prof_lat],
            "front_back_extent_world_m": [w(x) for x in prof_fwd],
            "back_layer_standoff_world_m": [w(x) for x in prof_off],
            "definitions": "per 5 cm band below the crown, calm frames, her torso frame: width = lateral p3..p97 extent of "
                           "the hair; front_back_extent = forward p3..p97 extent (face-framing / front strands to the "
                           "back of the hair, includes the head at ear level); back_layer_standoff = p90 distance of "
                           "the hair behind her (within 5 cm of her body) from the SMPL-X skin = how far the hair layer "
                           "stands off her back",
            "width_at_ear_level": {"world_m": w(prof_at(ear_drop)), "true_m": wt(prof_at(ear_drop)),
                                   "head_width_world_m": w(head_width),
                                   "ratio_to_head": w(prof_at(ear_drop) / head_width)},
            "width_at_shoulder_level": {"world_m": w(prof_at(sh_drop)), "true_m": wt(prof_at(sh_drop)),
                                        "shoulder_width_world_m": w(body_sh_width),
                                        "ratio_to_shoulders": w(prof_at(sh_drop) / body_sh_width)},
            "front_back_extent_at_shoulder_level": {"world_m": w(float(np.interp(sh_drop, prof_bins + 0.025, prof_fwd))),
                                                    "true_m": wt(float(np.interp(sh_drop, prof_bins + 0.025, prof_fwd)))},
            "back_layer_standoff_below_shoulders": {
                "world_m": w(float(np.nanmedian(prof_off[(prof_bins + 0.025 > sh_drop)]))) if np.isfinite(
                    prof_off[(prof_bins + 0.025 > sh_drop)]).any() else None,
                "true_m": wt(float(np.nanmedian(prof_off[(prof_bins + 0.025 > sh_drop)]))) if np.isfinite(
                    prof_off[(prof_bins + 0.025 > sh_drop)]).any() else None},
            "bias": f"visual hull = upper bound: masks dilated by >= {cfg.tol_m * 100:.1f} cm per side, concavities fill",
        },
        "motion": motion_metrics(pf, calm, turn, cfg, sc, bool(ctx.rest_j[J_LSH, 0] > ctx.rest_j[J_RSH, 0])),
    }
    metrics["length"]["posed_vertical_extent_below_crown"]["note"] = (
        "the POSED hair, measured down from the posed crown in calm frames, ends higher than the rest-mapped tip level "
        "(posture: her head is tilted and her back curved while dancing); the rest height is the length target")
    return metrics, {"pf": pf, "calm_clouds": calm_clouds, "turn_clouds": turn_clouds, "landmarks": lm}


def _exclusion_reason(h: dict, t: float, cfg: HairRefConfig) -> dict:
    """Why a hull frame is left out of the statistics, from its per-view diagnostics."""
    v = h.get("views", [])
    rc = h.get("recall")
    nc = h.get("n_carve", h.get("n_views", 0))
    carve_only = [f"{d['cam']} (IoU {d.get('iou', 0):.2f})" for d in v if "carve_only" in d]
    skipped = [f"{d['cam']} ({d['skip']})" for d in v if "skip" in d]
    why = []
    if nc < cfg.min_carve_views:
        why.append(f"only {nc} carving cameras")
    if rc is not None and rc < cfg.min_frame_recall:
        why.append(f"3D hair covers {rc:.0%} of the 2D hair masks")
    if carve_only:
        why.append("silhouette did not register (carve-only): " + ", ".join(carve_only))
    if skipped:
        why.append("skipped: " + ", ".join(skipped))
    return {"k": int(h["k"]), "t": round(t, 3), "recall": rc, "n_carve": nc, "carve_only": [d["cam"] for d in v
                                                                                          if "carve_only" in d],
            "reason": "; ".join(why)}


def summarise_exclusions(ex: list[dict]) -> str:
    """One line: which frames were left out and why (grouped by the carve-only camera)."""
    if not ex:
        return "none"
    groups: dict[str, list[int]] = {}
    for e in ex:
        key = ("camera " + "+".join(e.get("carve_only") or []) + " silhouette not registered (carve-only), low recall"
               if e.get("carve_only") else "recall just below the threshold" if e.get("recall") is not None
               else e["reason"])
        groups.setdefault(key, []).append(e["k"])
    return "; ".join(f"{len(ks)} frame{'s' if len(ks) > 1 else ''} ({_ranges(ks)}): {key}" for key, ks in groups.items())


def _ranges(ks: list[int]) -> str:
    ks = sorted(ks)
    out, a = [], ks[0]
    for p, c in zip(ks, ks[1:] + [None]):
        if c != p + 1:
            out.append(f"{a}" if a == p else f"{a}-{p}")
            a = c
    return ", ".join(out)


def registration_stats(hulls: list[dict], cfg: HairRefConfig) -> dict:
    """Silhouette registration of the views: how many hit the +-shift_px search limit (per camera), how many failed."""
    reg = [d for h in hulls for d in h.get("views", []) if "shift_px" in d and "carve_only" not in d]
    hit = [d for d in reg if max(abs(d["shift_px"][0]), abs(d["shift_px"][1])) >= cfg.shift_px]
    per: dict[str, int] = {}
    for d in hit:
        per[d["cam"]] = per.get(d["cam"], 0) + 1
    co = [d for h in hulls for d in h.get("views", []) if "carve_only" in d]
    per_co: dict[str, int] = {}
    for d in co:
        per_co[d["cam"]] = per_co.get(d["cam"], 0) + 1
    return {"registered_views": len(reg), "shift_limit_px": cfg.shift_px, "at_shift_limit": len(hit),
            "at_shift_limit_per_camera": dict(sorted(per.items())), "carve_only_views": len(co),
            "carve_only_per_camera": dict(sorted(per_co.items())),
            "note": "views at the limit may be under-registered by a few px (their hair votes still count)"}


def _circ_median(a: np.ndarray) -> float:
    """Circular centre of angles (rad): circular mean, refined by the median of the wrapped residuals."""
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if not len(a):
        return float("nan")
    m = float(np.angle(np.mean(np.exp(1j * a))))
    return float(wrap(m + np.median(wrap(a - m))))


def trailing_angles(psi: np.ndarray, phi_ref: np.ndarray, yaw_sign: np.ndarray, calm: np.ndarray) -> dict:
    """Trailing angle (deg) of the hair-mass azimuth psi behind a reference heading phi_ref, corrected by the
    CALM-frame offset (the hair mass is rarely straight behind: it lies over a shoulder). Positive = the hair mass lags
    the turn (rotated against yaw_sign). Returns raw + corrected per frame and the calm offset (deg)."""
    rel = wrap(np.asarray(psi, float) - (np.asarray(phi_ref, float) + np.pi))
    off = _circ_median(rel[np.asarray(calm, bool)]) if np.asarray(calm, bool).any() else 0.0
    sg = -np.sign(yaw_sign)
    return {"offset_deg": float(np.degrees(off)), "raw": np.degrees(sg * rel),
            "corrected": np.degrees(sg * wrap(rel - off)),
            "calm_scatter_iqr_deg": [float(np.degrees(x)) for x in np.percentile(wrap(rel[np.asarray(calm, bool)] - off),
                                                                                [25, 75])] if np.asarray(calm, bool).any()
            else None}


def _bootstrap_median_ci(x: np.ndarray, n: int, seed: int = 0) -> list[float] | None:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return None
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(x, (n, len(x))), axis=1)
    return [float(v) for v in np.percentile(meds, [2.5, 97.5])]


def motion_metrics(pf: dict, calm: np.ndarray, turn: np.ndarray, cfg: HairRefConfig, sc: float,
                   left_is_plus_x: bool = True) -> dict:
    """Spread (upper bound in turns) + trailing angle / lag, each with a resolved / not-resolved verdict."""
    w = lambda x: round(float(x), 4) if x is not None and np.isfinite(x) else None
    wt = lambda x: round(float(x) * sc, 4) if x is not None and np.isfinite(x) else None
    q = lambda a, p: float(np.percentile(a, p)) if len(a) else float("nan")
    kk = pf["k"]
    n_calm, n_turn = pf["n"][calm], pf["n"][turn]
    vox_ratio = q(n_turn, 50) / q(n_calm, 50) if len(n_calm) and len(n_turn) else float("nan")
    out = {
        "spread_radius_r95_world_m": {"calm_median": w(q(pf["r95"][calm], 50)),
                                      "turn_median": w(q(pf["r95"][turn], 50)) if turn.any() else None,
                                      "turn_p90": w(q(pf["r95"][turn], 90)) if turn.any() else None,
                                      "max": w(np.nanmax(pf["r95"])) if len(kk) else None},
        "spread_radius_r95_true_m": {"calm_median": wt(q(pf["r95"][calm], 50)),
                                     "turn_median": wt(q(pf["r95"][turn], 50)) if turn.any() else None,
                                     "turn_p90": wt(q(pf["r95"][turn], 90)) if turn.any() else None},
        "spread_bound": f"UPPER bound in turns: turn-frame hulls hold {vox_ratio:.2f} x the calm-frame voxels "
                        "(median), partly real spread, partly motion blur widening the masks" if np.isfinite(vox_ratio)
        else "upper bound (visual hull)",
        "hull_voxels_turn_over_calm": w(vox_ratio),
        "head_yaw_rate_dps": {"p50_abs": w(q(np.abs(pf["yaw"]), 50)), "p90_abs": w(q(np.abs(pf["yaw"]), 90)),
                              "max_abs": w(np.max(np.abs(pf["yaw"]))) if len(kk) else None},
        "spread_vs_yaw_corr": w(float(np.corrcoef(np.abs(pf["yaw"]), pf["r95"])[0, 1])) if len(kk) > 5 else None,
    }
    if not turn.any() or not calm.any():
        out["trailing"] = {"status": "not_resolved", "why": "no turn or no calm frames"}
        return out
    tr = {}
    for ref, phi, yw in (("head", pf["phi"], pf["yaw"]), ("torso", pf["phi_torso"], pf["yaw_torso"])):
        a = trailing_angles(pf["psi"], phi, yw, calm)
        cor = a["corrected"][turn]
        raw = a["raw"][turn]
        dirs = {}
        for name, s in (("positive_yaw", 1), ("negative_yaw", -1)):
            m = np.sign(yw[turn]) == s
            dirs[name] = {"n": int(m.sum()), "raw_median": w(q(raw[m], 50)) if m.any() else None,
                          "corrected_median": w(q(cor[m], 50)) if m.any() else None}
        tr[ref] = {"calm_offset_deg": w(a["offset_deg"]), "calm_scatter_iqr_deg": [w(x) for x in a["calm_scatter_iqr_deg"]],
                   "raw_median": w(q(raw, 50)), "corrected_median": w(q(cor, 50)),
                   "corrected_iqr": [w(q(cor, 25)), w(q(cor, 75))], "corrected_range": [w(np.min(cor)), w(np.max(cor))],
                   "corrected_median_ci95": [w(x) for x in (_bootstrap_median_ci(cor, cfg.bootstrap) or [])],
                   "by_turn_direction": dirs}
    # verdict: trailing must show the same sign in both turn directions and for both references, clear the noise floor
    # and have a bootstrap CI that excludes zero
    checks = {}
    for ref, r in tr.items():
        d = r["by_turn_direction"]
        meds = [d[x]["corrected_median"] for x in d if d[x]["n"] >= cfg.trail_min_per_dir]
        ci = r["corrected_median_ci95"]
        checks[ref] = {"both_directions": len(meds) == 2,
                       "same_sign": len(meds) == 2 and np.sign(meds[0]) == np.sign(meds[1]) != 0,
                       "above_noise": r["corrected_median"] is not None and abs(r["corrected_median"]) >= cfg.trail_noise_deg,
                       "ci_excludes_zero": bool(ci) and (ci[0] > 0 or ci[1] < 0)}
    resolved = all(all(c.values()) for c in checks.values())
    side_deg = tr["head"]["calm_offset_deg"]
    # offset sign: positive = a positive (right-handed) rotation about +Y away from straight behind; facing +z, that
    # moves the hair mass toward -x, which is her RIGHT when her left shoulder is on +x (SMPL-X rest pose)
    pos_side = "her right" if left_is_plus_x else "her left"
    neg_side = "her left" if left_is_plus_x else "her right"
    side = None if side_deg is None or abs(side_deg) < 10 else (pos_side if side_deg > 0 else neg_side)
    dt = float(np.median(np.diff(pf["t"]))) if len(kk) > 1 else 1 / 30
    off_h = np.radians(tr["head"]["calm_offset_deg"] or 0.0)
    tau, rms = (phase_lag(pf["phi"], np.unwrap(pf["psi"]), dt, offset=np.pi + off_h) if len(kk) > 10
                else (np.nan, np.nan))
    lag_ok = resolved and np.isfinite(rms) and rms <= cfg.lag_max_rms_deg and tau > 0
    out["trailing"] = {
        "status": "resolved" if resolved else "not_resolved",
        "n_turn_frames": int(turn.sum()), "n_calm_frames": int(calm.sum()),
        "definition": "hair-mass azimuth about the neck axis minus (reference heading + 180 deg + the CALM-frame "
                      "offset); positive = the hair mass lags the turn; turn frames only",
        "rule": f"resolved only when, for the head AND the torso heading, the corrected median is >= "
                f"{cfg.trail_noise_deg:.0f} deg, both turn directions (>= {cfg.trail_min_per_dir} frames each) agree in "
                f"sign and the bootstrap 95 % CI of the median excludes 0",
        "checks": checks, "head": tr["head"], "torso": tr["torso"],
        "calm_hair_mass_side": (f"in calm frames the hair mass sits {abs(side_deg):.0f} deg toward {side} of straight "
                                f"behind her head (hair swept over that shoulder)") if side else None,
        "hair_mass_centroid_radius_m_world": {"median": w(q(pf["centroid_r"], 50)),
                                              "note": "the hair-mass centroid lies this close to the neck axis, so its "
                                                      "azimuth is noisy (1 cm of centroid noise ~ 9 deg at 6 cm)"},
        "raw_median_without_offset_deg": tr["head"]["raw_median"],
        "verdict": ("the hair mass trails the turn by about {:.0f} deg".format(tr["head"]["corrected_median"])
                    if resolved else
                    "NOT RESOLVED in this window: after the calm-frame offset the trailing angle is "
                    f"{tr['head']['corrected_median']:+.0f} deg vs the head and {tr['torso']['corrected_median']:+.0f} deg "
                    "vs the torso, the turn directions disagree, and the CI includes 0; the raw "
                    f"{tr['head']['raw_median']:.0f} deg came from the calm offset, not from trailing"),
    }
    # version-1 keys kept for consumers (hair_groom reads them): null unless resolved
    out["trailing_angle_deg_turns"] = {"median": tr["head"]["corrected_median"] if resolved else None,
                                       "status": out["trailing"]["status"], "see": "motion.trailing"}
    out["lag_time_s"] = {"status": "resolved" if lag_ok else "not_resolved",
                         "phase_alignment": w(tau) if lag_ok else None,
                         "phase_alignment_s": w(tau), "residual_rms_deg": w(rms),
                         "rule": f"resolved only with a resolved trailing angle and a fit residual <= "
                                 f"{cfg.lag_max_rms_deg:.0f} deg RMS",
                         "note": "phase alignment of the hair-mass azimuth to the head heading (+180 deg + calm "
                                 f"offset); one grid step = {dt:.3f} s"}
    return out


# ------------------------------------------------------------------------------------------------ MediaPipe

def run_mediapipe(jobs: list[dict], spec_path: Path, model: str = "multiclass") -> dict:
    """Runs hair_segment_mp.py in .venv-face (face_python()); jobs already carry their output paths, existing outputs
    are not redone."""
    exe, extra_env = face_python()
    if exe is None or not MP_MODELS[model].exists():
        return {"status": "unavailable", "python": exe or str(FACE_VENV), "model": str(MP_MODELS[model])}
    todo = [j for j in jobs if not Path(j["out"]).exists()]
    t0 = time.time()
    if todo:
        spec_path.parent.mkdir(parents=True, exist_ok=True)
        spec_path.write_text(json.dumps({"models": {model: str(MP_MODELS[model])},
                                         "jobs": [{**j, "model": model} for j in todo]}))
        script = Path(__file__).with_name("hair_segment_mp.py")
        env = {**os.environ, **extra_env}
        r = subprocess.run([exe, str(script), "--jobs", str(spec_path)], capture_output=True, text=True, env=env)
        if r.returncode != 0:
            return {"status": "failed", "stderr": r.stderr[-2000:]}
    return {"status": "ok", "model": MP_MODELS[model].name, "jobs": len(jobs), "run": len(todo),
            "seconds": round(time.time() - t0, 1)}


# ------------------------------------------------------------------------------------------------ colour (video)

def _torso_quad(px_j: dict, shrink: float = 0.2) -> np.ndarray:
    q = np.array([px_j[J_LSH], px_j[J_RSH], px_j[J_RHIP], px_j[J_LHIP]], float)
    c = q.mean(0)
    return (c + (1 - shrink) * (q - c)).astype(np.int32)


def colour_step_for(cfg: HairRefConfig, n_frames_in_window: int) -> int:
    """colour_step, or (0 = auto) the step that keeps at most colour_views_per_cam views per camera, >= colour_min_step:
    the colour median converges long before that, and every view costs a frame read + a MediaPipe crop."""
    if cfg.colour_step > 0:
        return int(cfg.colour_step)
    return max(cfg.colour_min_step, int(np.ceil(n_frames_in_window / max(cfg.colour_views_per_cam, 1))))


def colour_views_for(tracks: dict, t0: float, t1: float, cfg: HairRefConfig) -> list[tuple[str, int, float]]:
    """(cam, frame, corrected t) of every step-th camera frame inside [t0, t1] (step per camera, colour_step_for)."""
    out = []
    for cam, tr in tracks.items():
        idx = np.nonzero((tr.t >= t0) & (tr.t <= t1))[0]
        if not len(idx):
            continue
        step = colour_step_for(cfg, len(idx))
        for i in idx[idx % step == 0]:  # absolute multiples: stable picks, cached MediaPipe crops stay valid
            if bool(tr.ok[i]):
                out.append((cam, int(i), float(tr.t[i])))
    return out


def colour_views(ctx: Context) -> list[tuple[str, int, float]]:
    return colour_views_for(ctx.tracks, float(ctx.body.t[0]), float(ctx.body.t[-1]), ctx.cfg)


def head_crop_box(ctx: Context, cam: str, i: int, t: float, side_m: float = 1.6) -> tuple[list[int], dict] | None:
    from .face_common import project as proj_np
    tr = ctx.tracks[cam]
    k = int(np.clip(np.searchsorted(ctx.body.t, t), 0, len(ctx.body.t) - 1))
    J = ctx.body.joints[k]
    pts, z = proj_np(J[:22], tr.K[i], tr.R[i], tr.tvec[i], float(tr.k1[i]))
    if z[J_HEAD] <= 0.2:
        return None
    f = float(tr.K[i][0, 0])
    side = side_m * f / z[J_HEAD]
    cx, cy = pts[J_HEAD]
    cy += 0.3 * side_m * f / z[J_HEAD]  # her head sits in the upper part of the crop, the hair below
    box = [int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2)]
    return box, {"px": pts, "z": z, "f": f, "k": k}


def sample_colours(ctx: Context, out_dir: Path, reuse: bool = False) -> dict:
    """Per view: white-top reference + zoned hair pixels (see module doc, step 3). Only the pixels used are decoded to
    linear (8-bit lookup) and the white-top search is limited to the torso quad's box. Cached in
    hair_ref/colour_samples.npz (reused with reuse=True when it holds the same views)."""
    cfg = ctx.cfg
    views = colour_views(ctx)
    cache = out_dir / "colour_samples.npz"
    key = json.dumps({"views": [(c, i) for c, i, _ in views], "mp": cfg.mp, "root_m": cfg.root_m, "mid_m": cfg.mid_m,
                      "tip_frac": cfg.tip_frac, "min_len_m": cfg.min_len_m, "pid": ctx.pid})
    if reuse:
        got = load_colour_samples(cache, key)
        if got is not None:
            print(f"[hair_ref] colour: reusing {cache.name} ({len(got['samples'])} views)", flush=True)
            return got
    mp_dir = out_dir / "mp"
    jobs, meta = [], []
    for cam, i, t in views:
        r = head_crop_box(ctx, cam, i, t)
        if r is None:
            continue
        box, geo = r
        img_p = ctx.work / "frames" / cam / f"{i:06d}.jpg"
        out = mp_dir / cam / f"{i:06d}.png"
        jobs.append({"image": str(img_p), "box": box, "out": str(out)})
        meta.append((cam, i, t, box, geo, out))
    mp_status = run_mediapipe(jobs, out_dir / "mp_jobs.json") if cfg.mp else {"status": "off"}
    use_mp = mp_status.get("status") == "ok"
    samples = []
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    t0 = time.time()
    for cam, i, t, box, geo, mp_out in meta:
        img = cv2.imread(str(ctx.work / "frames" / cam / f"{i:06d}.jpg"))
        hair = _read(_mask_path(ctx, cam, "hair", i))
        bodym = _read(_mask_path(ctx, cam, "body", i))
        if img is None or hair is None or bodym is None or hair.sum() < 50:
            continue
        H, W = hair.shape
        hair_e = cv2.erode(hair.astype(np.uint8), k3).astype(bool)
        mp_agree = None
        if use_mp and Path(mp_out).exists():
            mpm = cv2.imread(str(mp_out), cv2.IMREAD_UNCHANGED)
            x0, y0, x1, y1 = max(0, box[0]), max(0, box[1]), min(W, box[2]), min(H, box[3])
            mph = np.zeros((H, W), bool)
            if mpm is not None and mpm.ndim == 3 and mpm.shape[:2] == (y1 - y0, x1 - x0):
                mph[y0:y1, x0:x1] = mpm[..., 0] > 127
            full = _read(_mask_path(ctx, cam, "masks", i))
            inter = np.count_nonzero(hair & mph)
            uni = np.count_nonzero(hair | (mph & full)) if full is not None else np.count_nonzero(hair | mph)
            mp_agree = {"iou": round(inter / max(uni, 1), 3), "hair_mask_in_mp": round(inter / max(hair.sum(), 1), 3)}
            hair_e &= mph
        if hair_e.sum() < 30:
            continue
        # white top reference, searched only inside the torso quad's bounding box
        quad = _torso_quad({j: geo["px"][j] for j in (J_LSH, J_RSH, J_LHIP, J_RHIP)})
        qx0, qy0 = (int(v) for v in np.clip(quad.min(0), 0, [W - 1, H - 1]))
        qx1, qy1 = (int(v) for v in np.clip(quad.max(0) + 1, 1, [W, H]))
        if qx1 - qx0 < 4 or qy1 - qy0 < 4:
            continue
        sub = img[qy0:qy1, qx0:qx1]
        tq = np.zeros(sub.shape[:2], np.uint8)
        cv2.fillConvexPoly(tq, (quad - np.array([qx0, qy0])).astype(np.int32), 1)
        hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
        body_c = cv2.erode(bodym[qy0:qy1, qx0:qx1].astype(np.uint8), k3, iterations=2).astype(bool)
        hair_c = cv2.dilate(hair[qy0:qy1, qx0:qx1].astype(np.uint8), k3, iterations=3).astype(bool)
        top = tq.astype(bool) & body_c & (hsv[..., 1] < 60) & (sub.max(2) < 250) & ~hair_c
        if top.sum() < 60:
            continue
        tv = hsv[..., 2][top]
        sel = top & (hsv[..., 2] >= np.percentile(tv, 40)) & (hsv[..., 2] <= np.percentile(tv, 95))
        white = np.median(bgr8_to_linear_rgb(sub[sel]), 0)
        if luminance(white) < 0.05:
            continue
        # along-strand zones by distance from the projected head centre (m at head depth)
        ys, xs = np.nonzero(hair_e)
        hc = geo["px"][J_HEAD]
        mpp = geo["z"][J_HEAD] / geo["f"]  # metres per px at head depth
        d = np.hypot(xs - hc[0], ys - hc[1]) * mpp
        L = float(np.percentile(d, 99))
        zone = np.full(len(d), -1, np.int8)
        zone[d <= cfg.root_m] = 0
        if L >= cfg.min_len_m:
            zone[(d >= cfg.mid_m) & (d < cfg.tip_frac * L)] = 1
            zone[d >= cfg.tip_frac * L] = 2
        else:
            zone[d >= cfg.mid_m] = 1
        px = img[ys, xs]
        samples.append({"cam": cam, "i": i, "t": t, "white": white, "bgr8": px, "rgb": bgr8_to_linear_rgb(px),
                        "zone": zone, "d": d.astype(np.float32), "L": L, "mp": mp_agree, "n_top": int(sel.sum())})
    res = {"samples": samples, "mp": mp_status, "n_views": len(meta), "seconds": round(time.time() - t0, 1)}
    save_colour_samples(cache, key, res)
    return res


def save_colour_samples(path: Path, key: str, res: dict) -> None:
    S = res["samples"]
    n = np.array([len(s["zone"]) for s in S], np.int64)
    mp = np.array([[s["mp"]["iou"], s["mp"]["hair_mask_in_mp"]] if s.get("mp") else [np.nan, np.nan] for s in S],
                  np.float32).reshape(-1, 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, key=key, meta=json.dumps({"mp": res["mp"], "n_views": res["n_views"]}),
        cam=np.array([s["cam"] for s in S]), i=np.array([s["i"] for s in S], np.int32),
        t=np.array([s["t"] for s in S]), white=np.array([s["white"] for s in S], np.float32).reshape(-1, 3),
        L=np.array([s["L"] for s in S], np.float32), n_top=np.array([s["n_top"] for s in S], np.int32), mp_q=mp, n=n,
        bgr8=np.concatenate([s["bgr8"] for s in S]) if S else np.zeros((0, 3), np.uint8),
        zone=np.concatenate([s["zone"] for s in S]) if S else np.zeros(0, np.int8),
        d=np.concatenate([s["d"] for s in S]).astype(np.float16) if S else np.zeros(0, np.float16))


def load_colour_samples(path: Path, key: str | None = None) -> dict | None:
    """colour_samples.npz -> sample_colours' result (None when missing or computed for other views / settings)."""
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as z:
        if key is not None and str(z["key"]) != key:
            return None
        meta = json.loads(str(z["meta"]))
        n = z["n"]
        off = np.concatenate([[0], np.cumsum(n)]).astype(np.int64)
        cams, ii, tt, white, L, n_top, mp_q = z["cam"], z["i"], z["t"], z["white"], z["L"], z["n_top"], z["mp_q"]
        bgr8, zone, d = z["bgr8"], z["zone"], z["d"]
    S = []
    for j in range(len(n)):
        a, b = off[j], off[j + 1]
        mq = mp_q[j]
        S.append({"cam": str(cams[j]), "i": int(ii[j]), "t": float(tt[j]), "white": white[j].astype(float),
                  "bgr8": bgr8[a:b], "rgb": bgr8_to_linear_rgb(bgr8[a:b]), "zone": zone[a:b],
                  "d": d[a:b].astype(np.float32), "L": float(L[j]), "n_top": int(n_top[j]),
                  "mp": None if np.isnan(mq[0]) else {"iou": float(mq[0]), "hair_mask_in_mp": float(mq[1])}})
    return {"samples": S, "mp": {**meta["mp"], "cached": True}, "n_views": meta["n_views"]}


IMAGE_STAT_BANDS = {"p10": (7.5, 12.5), "p50": (47.5, 52.5), "p90": (87.5, 92.5)}  # luminance percentile bands


def palette_from_samples(samples: list[dict], albedo: float) -> dict:
    """Root / mid / tip medians (white-balanced), highlight / shadow bands and the p10 / p50 / p90 image statistics,
    per camera and overall (median over cameras), + the cross-camera spread of hue, saturation and lightness.
    These are OBSERVED colours (white-balanced on her top): they include the hair's own shading, so they describe
    the image a render must reproduce, not an albedo."""
    per_cam: dict[str, dict] = {}
    for s in samples:
        wb = white_balance(s["rgb"], s["white"], albedo)
        pc = per_cam.setdefault(s["cam"], {"root": [], "mid": [], "tip": [], "all": [], "raw": [], "white": [], "n_views": 0})
        pc["n_views"] += 1
        pc["white"].append(s["white"])
        for z, name in ((0, "root"), (1, "mid"), (2, "tip")):
            sel = s["zone"] == z
            if sel.sum() >= 10:
                pc[name].append(wb[sel])
        pc["all"].append(wb[s["zone"] >= 1])
        pc["raw"].append(s["rgb"])
    out_cam, zones = {}, {"root": [], "mid": [], "tip": [], "highlight": [], "shadow": [], "raw": [],
                          **{"img_" + n: [] for n in IMAGE_STAT_BANDS}}
    for cam, pc in sorted(per_cam.items()):
        rec = {"n_views": pc["n_views"]}
        for name in ("root", "mid", "tip"):
            if pc[name]:
                a = np.concatenate(pc[name])
                rec[name] = np.median(a, 0)
                rec[name + "_n"] = len(a)
        if pc["all"]:
            a = np.concatenate(pc["all"])
            if len(a) > 50:
                Y = luminance(a)
                lo, hi = np.percentile(Y, [5, 15]), np.percentile(Y, [90, 99])
                rec["shadow"] = np.median(a[(Y >= lo[0]) & (Y <= lo[1])], 0)
                rec["highlight"] = np.median(a[(Y >= hi[0]) & (Y <= hi[1])], 0)
                for name, (pa, pb) in IMAGE_STAT_BANDS.items():
                    ya, yb = np.percentile(Y, [pa, pb])
                    rec["img_" + name] = np.median(a[(Y >= ya) & (Y <= yb)], 0)
        rec["raw"] = np.median(np.concatenate(pc["raw"]), 0)
        rec["white_ref"] = np.median(np.array(pc["white"]), 0)
        out_cam[cam] = rec
        for name in zones:
            if name in rec:
                zones[name].append(rec[name])
    hues = {c: colour_entry(r["mid"])["hsv"][0] for c, r in out_cam.items() if "mid" in r}
    light = {c: colour_entry(r["mid"])["lab"][0] for c, r in out_cam.items() if "mid" in r}
    sats = {c: colour_entry(r["mid"])["hsv"][1] for c, r in out_cam.items() if "mid" in r}
    overall = {}
    for name, vals in zones.items():
        if vals:
            A = np.array(vals)
            med = np.median(A, 0)
            d = [delta_e76(a, med) for a in A]
            overall[name] = {**colour_entry(med), "n_cams": len(A),
                             "camera_spread_dE76_median": round(float(np.median(d)), 1) if len(A) > 1 else 0.0,
                             "camera_spread_dE76_max": round(float(np.max(d)), 1) if len(A) > 1 else 0.0}
    image_stats = {n: overall.pop("img_" + n) for n in IMAGE_STAT_BANDS if "img_" + n in overall}
    Ls = np.array(list(light.values()), float)
    hs = np.array(list(hues.values()), float)
    hue_dev = np.degrees(wrap(np.radians(hs) - _circ_median(np.radians(hs)))) if len(hs) else np.zeros(0)
    Ss = np.array(list(sats.values()), float)
    spread = {
        "mid_L_range": [round(float(Ls.min()), 1), round(float(Ls.max()), 1)] if len(Ls) else None,
        "mid_L_sd": round(float(Ls.std(ddof=1)), 1) if len(Ls) > 1 else None,
        "mid_hue_dev_deg_max": round(float(np.abs(hue_dev).max()), 1) if len(hue_dev) else None,
        "mid_hue_sd_deg": round(float(hue_dev.std(ddof=1)), 1) if len(hue_dev) > 1 else None,
        "mid_saturation_range": [round(float(Ss.min()), 3), round(float(Ss.max()), 3)] if len(Ss) else None,
        "mid_saturation_sd": round(float(Ss.std(ddof=1)), 3) if len(Ss) > 1 else None,
        "mid_camera_dE76_to_consensus": {c: round(delta_e76(r["mid"], overall["mid"]["linear_srgb"]), 1)
                                         for c, r in out_cam.items() if "mid" in r and "mid" in overall},
        "note": "per-camera spread of the white-balanced mid colour: hue is consistent, lightness is not (phone tone "
                "curves and exposure); tolerances in the acceptance follow from these numbers"}
    return {"overall": overall, "image_stats": image_stats, "camera_spread": spread,
            "mid_hue_deg_per_camera": hues, "mid_lightness_L_per_camera": light, "mid_saturation_per_camera": sats,
            "what": "OBSERVED hair colours, white-balanced per view on her white top (scaled to an assumed top albedo "
                    "of %.2f), linear sRGB: they include the hair's own shading and self-shadowing, so they are image "
                    "statistics to match under the demo lighting, NOT an albedo (used as one, a lit shader would darken "
                    "the hair a second time)" % albedo,
            "note": "phone JPEGs are tone-mapped: dividing by the white top fixes the colour cast and most of the "
                    "exposure, but cameras whose top sits in the highlight shoulder (bright, compressed) render the "
                    "hair darker; the cross-camera median is the reference, the spread is reported per zone",
            "per_camera": {c: {k: (colour_entry(v, r.get(k + "_n")) if isinstance(v, np.ndarray) else v)
                               for k, v in r.items() if not k.endswith("_n")} for c, r in out_cam.items()}}


# ------------------------------------------------------------------------------------------------ portrait

def portrait_reference(path: Path, out_dir: Path, albedo: float, use_mp: bool = True) -> dict:
    """Portrait hair colour + visible-section length (module doc, step 4)."""
    img = cv2.imread(str(path))
    if img is None:
        return {"status": "missing", "path": str(path)}
    H, W = img.shape[:2]
    mpo = out_dir / "mp_portrait.png"
    st = run_mediapipe([{"image": str(path), "box": None, "out": str(mpo)}], out_dir / "mp_portrait_job.json") \
        if use_mp else {"status": "off"}
    if st.get("status") != "ok" or not mpo.exists():
        return {"status": "no segmentation", "mp": st}
    m = cv2.imread(str(mpo), cv2.IMREAD_UNCHANGED)
    hair, clothes, face = m[..., 0] > 127, m[..., 1] > 127, m[..., 2] > 127
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    rgb = srgb_to_linear(img[:, :, ::-1] / 255.0)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    hair_e = cv2.erode(hair.astype(np.uint8), k3).astype(bool)
    cl = cv2.erode(clothes.astype(np.uint8), k3, iterations=2).astype(bool) & (hsv[..., 1] < 50) & (img.max(2) < 250)
    if cl.sum() < 50 or hair_e.sum() < 200:
        return {"status": "too few pixels", "white_px": int(cl.sum()), "hair_px": int(hair_e.sum())}
    v = hsv[..., 2][cl]
    sel = cl & (hsv[..., 2] >= np.percentile(v, 40)) & (hsv[..., 2] <= np.percentile(v, 95))
    white = np.median(rgb[sel], 0)
    ys, xs = np.nonzero(hair_e)
    fy, fx = np.nonzero(face)
    crown_y = int(ys.min())
    chin_y = int(fy.max()) if len(fy) else crown_y + int(0.3 * H)
    head_h = max(chin_y - crown_y, 1)
    bottom = float(np.percentile(ys, 99.5))
    Lpx = bottom - crown_y
    zone = np.full(len(ys), -1)
    zone[ys < crown_y + 0.18 * head_h] = 0
    zone[(ys >= crown_y + 0.45 * head_h) & (ys < bottom - 0.25 * Lpx)] = 1
    zone[ys >= bottom - 0.15 * Lpx] = 2
    wb = white_balance(rgb[ys, xs], white, albedo)
    pal = {}
    for z, name in ((0, "root"), (1, "mid"), (2, "tip")):
        s = zone == z
        if s.sum() >= 20:
            pal[name] = colour_entry(np.median(wb[s], 0), int(s.sum()))
    Y = luminance(wb)
    lo, hi = np.percentile(Y, [5, 15]), np.percentile(Y, [90, 99])
    pal["shadow"] = colour_entry(np.median(wb[(Y >= lo[0]) & (Y <= lo[1])], 0))
    pal["highlight"] = colour_entry(np.median(wb[(Y >= hi[0]) & (Y <= hi[1])], 0))
    pal["raw"] = colour_entry(np.median(rgb[ys, xs], 0))
    return {"status": "ok", "path": str(path), "size": [W, H], "mp": st, "white_ref": colour_entry(white, int(sel.sum())),
            "hair_px": int(hair_e.sum()), "palette": pal,
            "geometry_px": {"crown_y": crown_y, "chin_y": chin_y, "hair_bottom_y": round(bottom, 1),
                            "head_height_px": head_h},
            "visible_length_head_heights": round(Lpx / head_h, 2),
            "_masks": {"hair": hair, "clothes_white": sel, "face": face, "zone": (ys, xs, zone)}}


# ------------------------------------------------------------------------------------------------ tip bloom recipe

def _git_show(rev_path: str, repo: Path = HM_REPO) -> str | None:
    try:
        r = subprocess.run(["git", "-C", str(repo), "show", rev_path], capture_output=True, text=True, timeout=30)
        return r.stdout if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _yaml_block_value(text: str, block: str, key: str) -> str | None:
    """Value of `key: m_Value: x` inside the volume component named `block` of a Unity YAML asset."""
    i = text.find(f"m_Name: {block}\n")
    if i < 0:
        return None
    j = text.find("--- !u!", i)
    seg = text[i: j if j > 0 else None]
    m = re.search(rf"\n  {key}:\n    m_OverrideState: (\d)\n    m_Value: ([^\n]+)", seg)
    return m.group(2).strip() if m else None


def parse_hair_cs(src: str) -> dict:
    """Strand geometry and colour gradient of the original HairSimulation.cs."""
    num = lambda name: float(re.search(rf"public \w+ {name} = ([-0-9.]+)f?;", src).group(1))
    keys = re.findall(r"new GradientColorKey\((Color\.\w+|new Color\([^)]*\)|\w+), ([0-9.]+)f\)", src)
    grad = []
    for col, pos in keys:
        col = col.strip()
        if col == "Color.black":
            c = [0.0, 0.0, 0.0]
        elif col == "Color.white":
            c = [1.0, 1.0, 1.0]
        elif col == "darkGrey":
            c = [float(x) for x in re.search(r"Color darkGrey = new\(([^)]*)\)", src).group(1).replace("f", "").split(",")]
        else:
            c = [float(x) for x in re.search(r"Color\(([^)]*)\)", col).group(1).replace("f", "").split(",")]
        grad.append({"t": float(pos), "rgb_gamma": c})
    card = re.search(r"Mathf\.Max\(([0-9.]+)f, cardioidScale \* \(1f - Mathf\.Cos\(alphaVal\)\)\)", src)
    return {"skullRadius": num("skullRadius"), "cardioidScale": num("cardioidScale"), "topHairCount": int(num("topHairCount")),
            "domeMaxAngle": num("domeMaxAngle"), "sideLayerCount": int(num("sideLayerCount")),
            "sideStrandsPerLayer": int(num("sideStrandsPerLayer")), "ringAngleSpan": num("ringAngleSpan"),
            "segmentsPerStrand": int(num("segmentsPerStrand")), "strandThickness": num("strandThickness"),
            "strandEndThickness": num("strandEndThickness"), "min_length_m": float(card.group(1)) if card else None,
            "gradient_keys": grad}


def gradient_eval(keys: list[dict], s: float) -> np.ndarray:
    ts = [k["t"] for k in keys]
    cs = np.array([k["rgb_gamma"] for k in keys], float)
    return np.array([np.interp(s, ts, cs[:, c]) for c in range(3)])


def baked_vertex_colours(keys: list[dict], n_vertices: int) -> np.ndarray:
    """What LineRenderer actually feeds the shader (verified in the running editor, 2026-10-06): the gradient
    evaluated per vertex (by length fraction), converted gamma -> linear (project colour space Linear) and stored as
    UNorm8x4 vertex colours, i.e. clamped to [0, 1] and quantised to 1/255."""
    s = np.linspace(0, 1, n_vertices)
    g = np.array([gradient_eval(keys, x) for x in s])
    lin = srgb_to_linear(np.clip(g, 0, 1))
    return np.round(lin * 255) / 255


def measured_baked(keys: list[dict], n_vertices: int) -> np.ndarray | None:
    """The editor-measured vertex colours when the strand is the measured original (6 points, keys 0/.5/.95/1)."""
    if n_vertices == len(MEASURED_BAKED_8BIT) and np.allclose([k["t"] for k in keys], MEASURED_STRAND_KEYS_T):
        return np.repeat(np.array(MEASURED_BAKED_8BIT, float)[:, None] / 255, 3, 1)
    return None


LIT_HAIR_MAX_LINEAR = 0.6  # lit hair body ceiling (max channel): bloom share < 1 % under URP's soft knee


def bloom_share(v: float, threshold: float = 1.0, knee: float = URP_SOFT_KNEE) -> float:
    """Share of a pixel of brightness v (max channel, linear) that URP's bloom prefilter passes on (soft knee)."""
    k = knee * threshold
    soft = min(max(v - threshold + k, 0.0), 2 * k) ** 2 / (4 * k + 1e-5)
    return max(soft, v - threshold) / max(v, 1e-5)


def emission_profile(recipe: dict, n: int = 101) -> tuple[np.ndarray, np.ndarray]:
    """Linear HDR output along the strand (s in [0, 1]): baked vertex colours interpolated linearly x shader gain."""
    vc = np.array(recipe["vertex_colours_linear_baked"])
    sv = np.linspace(0, 1, len(vc))
    s = np.linspace(0, 1, n)
    return s, np.interp(s, sv, vc[:, 0]) * recipe["shader"]["multiplier"]


def tip_bloom_recipe(repo: Path = HM_REPO) -> dict:
    """The original glowing-tip look, read from the head-movement repo (falls back to the values verified on
    2026-10-06 when the repo or git is unavailable)."""
    src = _git_show(f"{ORIGINAL_HAIR_COMMIT}:Assets/HairSimulation.cs", repo) if (repo / ".git").exists() else None
    hair = parse_hair_cs(src) if src else None
    sg_text = (repo / "Assets" / "bloom-shader.shadergraph").read_text() if (repo / "Assets" / "bloom-shader.shadergraph").exists() else ""
    mult = None
    if sg_text:
        objs = [json.loads(b) for b in re.split(r"\n\s*\n(?=\{)", sg_text) if b.strip().startswith("{")]
        byid = {o.get("m_ObjectId"): o for o in objs}
        for o in objs:
            if o.get("m_Type", "").endswith("Vector1Node"):
                for sl in o.get("m_Slots", []):
                    so = byid.get(sl["m_Id"], {})
                    if so.get("m_SlotType") == 0:
                        mult = float(so.get("m_Value"))
        unlit = any(o.get("m_Type", "").endswith("UniversalUnlitSubTarget") for o in objs)
        tgt = next((o for o in objs if o.get("m_Type", "").endswith("UniversalTarget")), {})
        vcol = any(o.get("m_Type", "").endswith("VertexColorNode") for o in objs)
    else:
        unlit, tgt, vcol = True, {}, True
    vp = (repo / "Assets" / "DefaultVolumeProfile.asset")
    vtext = vp.read_text() if vp.exists() else ""
    bloom = {k: _yaml_block_value(vtext, "Bloom", k) for k in ("threshold", "intensity", "scatter", "clamp", "tint",
                                                                "highQualityFiltering", "downscale", "maxIterations",
                                                                "skipIterations", "dirtIntensity")} if vtext else {}
    tonemap = _yaml_block_value(vtext, "Tonemapping", "mode") if vtext else None
    urp = (repo / "Assets" / "Bloom.asset")
    utext = urp.read_text() if urp.exists() else ""
    hdr = re.search(r"m_SupportsHDR: (\d)", utext)
    prec = re.search(r"m_HDRColorBufferPrecision: (\d)", utext)
    msaa = re.search(r"m_MSAA: (\d)", utext)
    gs = (repo / "ProjectSettings" / "GraphicsSettings.asset")
    gtext = gs.read_text() if gs.exists() else ""
    active = "Bloom.asset" if "b44afe424bf20cfaba438ab6431aa900" in gtext else "unknown"
    mat = (repo / "Assets" / "BloomMat.mat")
    mtext = mat.read_text() if mat.exists() else ""
    hair = hair or {"segmentsPerStrand": 6, "strandThickness": 0.01, "strandEndThickness": 0.001, "skullRadius": 0.07,
                    "cardioidScale": 0.16, "topHairCount": 30, "domeMaxAngle": 60.0, "sideLayerCount": 5,
                    "sideStrandsPerLayer": 30, "ringAngleSpan": 60.0, "min_length_m": 0.2,
                    "gradient_keys": [{"t": 0.0, "rgb_gamma": [0, 0, 0]}, {"t": 0.5, "rgb_gamma": [0.1, 0.1, 0.1]},
                                      {"t": 0.95, "rgb_gamma": [1, 1, 1]}, {"t": 1.0, "rgb_gamma": [4, 3.2, 1.5]}]}
    nseg = hair["segmentsPerStrand"]
    vc_model = baked_vertex_colours(hair["gradient_keys"], nseg)
    vc_meas = measured_baked(hair["gradient_keys"], nseg)
    vc = vc_meas if vc_meas is not None else vc_model  # what the GPU gets: the editor measurement when it applies
    rec = {
        "source": {"repo": str(repo), "hair_script": f"git show {ORIGINAL_HAIR_COMMIT}:Assets/HairSimulation.cs "
                                                     "(user's original; created 961b058 'parametric hair and line "
                                                     "renderers' 2024-12-23, bloom set up in 91da08c 'bloom' 2024-12-20)",
                   "shader_graph": "Assets/bloom-shader.shadergraph", "material": "Assets/BloomMat.mat",
                   "pipeline": "Assets/Bloom.asset + Assets/Bloom_Renderer.asset",
                   "volume": "Assets/DefaultVolumeProfile.asset (URP global default volume profile; no scene Volume)",
                   "parsed_from_repo": bool(src and sg_text and vtext)},
        "strands": {"count": hair["topHairCount"] + hair["sideLayerCount"] * hair["sideStrandsPerLayer"],
                    "dome": hair["topHairCount"], "ring": f"{hair['sideLayerCount']} layers x {hair['sideStrandsPerLayer']}",
                    "renderer": "one LineRenderer per strand (world space, camera-facing ribbon), material BloomMat",
                    "points_per_strand": nseg, "width_root_m": hair["strandThickness"], "width_tip_m": hair["strandEndThickness"],
                    "length_m": f"max({hair['min_length_m']}, {hair['cardioidScale']} x (1 - cos alpha)) = "
                                f"{hair['min_length_m']}..{2 * hair['cardioidScale']:.2f} m (cardioid: longest at the back)",
                    "skull_radius_m": hair["skullRadius"]},
        "gradient_authored": hair["gradient_keys"],
        "vertex_colours_linear_baked": [[round(float(x), 4) for x in r] for r in vc],
        "vertex_colours_source": "editor measurement (LineRenderer.BakeMesh, 2026-10-06)" if vc_meas is not None
        else "model (gamma -> linear, UNorm8)",
        "vertex_colours_8bit": {"model": [int(round(x * 255)) for x in vc_model[:, 0]],
                                "measured": list(MEASURED_BAKED_8BIT) if vc_meas is not None else None,
                                "max_step_difference": int(np.abs(np.round(vc_model[:, 0] * 255) - np.round(
                                    vc_meas[:, 0] * 255)).max()) if vc_meas is not None else None},
        "vertex_colour_facts": "LineRenderer bakes the gradient per vertex (by length fraction), gamma->linear (Linear "
                               "colour space) into UNorm8x4 vertex colours: the HDR tip key (4, 3.2, 1.5) is CLAMPED to "
                               "white (1, 1, 1) and the intermediate keys are only sampled at the 6 vertices "
                               "(verified 2026-10-06 in the running editor: LineRenderer.BakeMesh colours "
                               "0, .004, .008, .071, .451, 1.0 and vertex attribute Color:UNorm8x4)",
        "shader": {"graph": "URP Unlit, opaque, ZWrite on, front faces, casts shadows",
                   "unlit": unlit, "vertex_colour_input": vcol, "multiplier": mult if mult is not None else 3.1,
                   "base_colour": "VertexColor x {:.2f}".format(mult if mult is not None else 3.1),
                   "render_face": tgt.get("m_RenderFace"), "material_shader_guid_matches":
                       "0d67853f51e9ffe3b89dc3ba5c7fee5a" in mtext,
                   "unused_property": "_Color (HDR, 6.42 on BloomMat) is declared but not connected"},
        "bloom_volume": {**{k: v for k, v in bloom.items()}, "tonemapping_mode": tonemap,
                         "tonemapping": "None (mode 0): HDR values above 1 clip to white on screen, the bloom halo "
                                        "carries the glow"},
        "pipeline": {"asset": active, "hdr": bool(int(hdr.group(1))) if hdr else None,
                     "hdr_precision": "R11G11B10 (32-bit)" if prec and prec.group(1) == "0" else (prec.group(1) if prec else None),
                     "msaa": int(msaa.group(1)) if msaa else None, "renderer": "Forward (Bloom_Renderer)",
                     "camera": "post-processing on, HDR on"},
    }
    s, e = emission_profile(rec, 1001)
    thr = float(bloom.get("threshold") or 1.0)
    above = s[e > thr]
    knee = s[e > URP_SOFT_KNEE * thr]
    rec["emission_along_strand"] = {
        "s": [round(float(x), 2) for x in s[::100]], "linear_hdr": [round(float(x), 3) for x in e[::100]],
        "at_vertices": [round(float(x) * rec["shader"]["multiplier"], 3) for x in vc[:, 0]],
        "above_threshold_from_s": round(float(above.min()), 3) if len(above) else None,
        "soft_knee_from_s": round(float(knee.min()), 3) if len(knee) else None,
        "soft_knee": {"knee": URP_SOFT_KNEE, "starts_at_linear": round(URP_SOFT_KNEE * thr, 3),
                      "bloom_share_at_linear": {str(v): round(bloom_share(v, thr), 4) for v in (0.6, 0.7, 0.8, 1.0)},
                      "lit_hair_max_linear": LIT_HAIR_MAX_LINEAR,
                      "note": "URP bloom prefilter: knee = 0.5 x threshold (hard-coded), so pixels above 0.5 linear "
                              "(max channel) already bloom a little; keep the lit hair body <= %.1f linear so only the "
                              "tips glow" % LIT_HAIR_MAX_LINEAR},
        "tip_value": round(float(e[-1]), 3), "colour": "neutral white (tint white; the warm HDR key never reaches the GPU)",
        "glowing_length_m_for_0p32m_strand": round((1 - float(above.min())) * 2 * hair["cardioidScale"], 3) if len(above) else None,
    }
    return rec


# ------------------------------------------------------------------------------------------------ review images

def _put(img, text, org, scale=0.5, color=(255, 255, 255), thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _swatch(entry: dict | None, w: int, h: int, label: str, compact: bool = False) -> np.ndarray:
    tile = np.full((h, w, 3), 30, np.uint8)
    if entry:
        r, g, b = entry["srgb8"]
        tile[: h - 34] = (b, g, r)
        _put(tile, label, (4, h - 20), 0.42)
        _put(tile, entry["hex"] if compact else entry["hex"] + "  lin " + ",".join(f"{x:.2f}" for x in entry["linear_srgb"]),
             (4, h - 5), 0.36)
    else:
        _put(tile, label + ": n/a", (4, h - 8), 0.42)
    return tile


def render_bloom_strip(recipe: dict, base_rgb_lin: np.ndarray | None, w: int = 600, h: int = 110) -> np.ndarray:
    """Simulated look of one strand with the original emission + URP-like bloom (threshold/soft knee, blur pyramid)."""
    s, e = emission_profile(recipe, 400)
    thr = float(recipe["bloom_volume"].get("threshold") or 1)
    inten = float(recipe["bloom_volume"].get("intensity") or 1)
    img = np.zeros((h, w, 3), np.float32)
    x0, x1, yc = 30, w - 40, h // 2
    for n in range(len(s) - 1):
        xa = int(x0 + s[n] * (x1 - x0))
        width = recipe["strands"]["width_root_m"] + s[n] * (recipe["strands"]["width_tip_m"] - recipe["strands"]["width_root_m"])
        half = max(1, int(round(width / recipe["strands"]["width_root_m"] * 7)))
        col = np.full(3, e[n], np.float32)
        if base_rgb_lin is not None:  # target: the lit hair colour + the continuous white emission E(s)
            col = np.asarray(base_rgb_lin, np.float32) + np.full(3, e[n], np.float32)
        img[yc - half: yc + half + 1, xa: xa + 2] = col
    knee = thr * URP_SOFT_KNEE
    Y = img.max(2, keepdims=True)
    soft = np.clip(Y - thr + knee, 0, 2 * knee) ** 2 / (4 * knee + 1e-5)
    contrib = np.maximum(soft, Y - thr) / np.maximum(Y, 1e-5)
    bright = img * contrib
    glow = np.zeros_like(img)
    for sig in (2, 4, 8, 16, 32):
        glow += cv2.GaussianBlur(bright, (0, 0), sig)
    out = img + inten * glow / 2.0
    return (linear_to_srgb(np.clip(out, 0, 1))[:, :, ::-1] * 255).astype(np.uint8)


def _plot_motion(pf: dict, metrics: dict, path: Path) -> np.ndarray:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(7.2, 4.6), dpi=100, sharex=True)
    t = pf["t"]
    ax[0].plot(t, pf["r95"] * 100, color="#d0306a", lw=1.6, label="hair spread r95 about the neck axis (cm, world)")
    ax[0].plot(t, pf["drop"] * 100, color="#888", lw=1.0, label="vertical extent below crown (cm, world)")
    calm = pf["calm"].astype(bool)
    turn = pf["turn"].astype(bool)
    ax[0].scatter(t[calm], np.zeros(calm.sum()) + 2, s=6, color="#2a9d8f", label="calm frames")
    ax[0].scatter(t[turn], np.zeros(turn.sum()) + 6, s=6, color="#e76f51", label="turn frames")
    ax[0].legend(fontsize=7, loc="upper left", ncol=2)
    ax[0].set_ylabel("cm")
    ax[0].set_ylim(0, max(70, float(np.nanmax(pf["drop"]) * 100 + 10)))
    ax[1].plot(t, pf["yaw"], color="#264653", lw=1.2, label="head yaw rate (deg/s)")
    ax[1].axhline(0, color="#ccc", lw=0.8)
    ax[1].set_ylabel("deg/s")
    ax[1].set_xlabel("reference time (s)")
    a2 = ax[1].twinx()
    tr = metrics["motion"].get("trailing", {})
    off = (tr.get("head") or {}).get("calm_offset_deg")
    if off is not None:
        a_ = trailing_angles(pf["psi"], pf["phi"], pf["yaw"], calm)
        a2.plot(t[turn], a_["raw"][turn], "o", mfc="none", color="#bbbbbb", ms=3,
                label="raw trailing angle (no calm offset)")
        a2.plot(t[turn], a_["corrected"][turn], "o", color="#e76f51", ms=3,
                label=f"trailing angle minus the calm offset ({off:+.0f} deg)")
        a2.axhspan(-15, 15, color="#e76f51", alpha=0.08, lw=0)
    a2.axhline(0, color="#e76f51", lw=0.6, ls=":")
    a2.set_ylabel("trailing deg", color="#e76f51")
    a2.set_ylim(-90, 110)
    h1, l1 = ax[1].get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    ax[1].legend(h1 + h2, l1 + l2, fontsize=6.5, loc="upper left")
    st = tr.get("status", "n/a").replace("_", " ").upper()
    ax[1].set_title(f"hair trailing in turns: {st} (shaded = +-15 deg noise floor)", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return cv2.imread(str(path))


def _occupancy_panel(calm_clouds, lm_posed, metrics: dict, view: str, voxel_m: float, size=(360, 520)) -> np.ndarray:
    """How often each pixel is covered by hair over the calm frames (her torso frame; back view: lateral x height,
    side view: forward x height), contours at 25 % and 50 %, body silhouette (>= 30 % of frames) in grey, landmark
    lines at their REST heights below the crown."""
    W, H = size
    img = np.full((H, W, 3), 18, np.uint8)
    if not calm_clouds:
        return img
    rng_x, rng_y = (-0.45, 0.45), (-1.05, 0.12)
    sx, sy = W / (rng_x[1] - rng_x[0]), H / (rng_y[1] - rng_y[0])
    half = max(1, int(round(0.5 * voxel_m * sx)))
    freq = np.zeros((H, W), np.float32)
    bod = np.zeros((H, W), np.float32)
    ci = 0 if view == "back" else 2
    sign = -1 if view == "back" else 1
    for _, loc, body in calm_clouds:
        for pts, A, r in ((loc, freq, half), (body, bod, 2)):
            u = ((pts[:, ci] * sign - rng_x[0]) * sx).astype(int)
            v = ((rng_y[1] - pts[:, 1]) * sy).astype(int)
            ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
            m = np.zeros((H, W), np.uint8)
            m[v[ok], u[ok]] = 1
            m = cv2.dilate(m, np.ones((2 * r + 1, 2 * r + 1), np.uint8))
            A += m
    n = len(calm_clouds)
    freq /= n
    bod /= n
    img[bod >= 0.3] = (75, 75, 75)
    a = np.clip(freq, 0, 1)[..., None]
    img = (img * (1 - 0.85 * a) + np.array([50, 50, 235]) * 0.85 * a).astype(np.uint8)
    for lev, col in ((0.25, (200, 200, 255)), (0.5, (255, 255, 255))):
        cnts, _ = cv2.findContours((freq >= lev).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, cnts, -1, col, 1, cv2.LINE_AA)
    for name, dy, colr in lm_posed:
        y = int((rng_y[1] - dy) * sy)
        if name.startswith("POSED"):  # dashed: measured in this (posed) frame
            for x in range(0, W, 14):
                cv2.line(img, (x, y), (min(W, x + 8), y), colr, 2, cv2.LINE_AA)
            _put(img, name, (W - 150, y - 4), 0.36, colr)
            continue
        cv2.line(img, (0, y), (W, y), colr, 1, cv2.LINE_AA)
        _put(img, name, (4, y - 3), 0.36, colr)
    _put(img, "BACK view (her left = image right)" if view == "back" else "SIDE view (she faces right)", (4, 14), 0.42)
    _put(img, "hair frequency, contours 25 % / 50 %", (4, 30), 0.38)
    dy = {name.split()[0]: d for name, d, _ in lm_posed}
    if "POSED" in dy and "TIPS" in dy:
        _put(img, "dashed = POSED calm tip level (this image)", (4, H - 46), 0.36, (60, 160, 255))
        _put(img, f"solid TIPS (rest) = length target, {100 * (dy['POSED'] - dy['TIPS']):.0f} cm lower:",
             (4, H - 30), 0.36, (60, 60, 255))
        _put(img, "posture (head tilt, curved back) lifts the posed hair", (4, H - 14), 0.36, (60, 60, 255))
    return img


def _turn_panel(turn_clouds, trailing: dict | None = None, size=360) -> np.ndarray:
    img = np.full((size, size, 3), 18, np.uint8)
    c = size // 2
    sc = size / 1.6
    if turn_clouds:
        acc = np.zeros((size, size), np.float32)
        for _, xz in turn_clouds:
            u = (c + xz[:, 0] * sc).astype(int)
            v = (c - xz[:, 1] * sc).astype(int)
            ok = (u >= 0) & (u < size) & (v >= 0) & (v < size)
            m = np.zeros((size, size), np.uint8)
            m[v[ok], u[ok]] = 1
            acc += cv2.dilate(m, np.ones((5, 5), np.uint8))
        acc /= len(turn_clouds)
        a = np.clip(acc, 0, 1)[..., None]
        img = (img * (1 - 0.9 * a) + np.array([60, 60, 230]) * 0.9 * a).astype(np.uint8)
        cnts, _ = cv2.findContours((acc >= 0.25).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, cnts, -1, (200, 200, 255), 1, cv2.LINE_AA)
    for r in (0.2, 0.4, 0.6):
        cv2.circle(img, (c, c), int(r * sc), (90, 90, 90), 1, cv2.LINE_AA)
        _put(img, f"{int(r * 100)} cm", (c + int(r * sc * 0.72), c - int(r * sc * 0.72)), 0.35, (160, 160, 160))
    cv2.arrowedLine(img, (c, c), (c, c - int(0.22 * sc)), (80, 220, 80), 2, tipLength=0.25)
    # spin direction: anticlockwise seen from above (image), drawn as an arc at 0.5 m
    cv2.ellipse(img, (c, c), (int(0.5 * sc), int(0.5 * sc)), 0, -20, -80, (60, 200, 230), 2, cv2.LINE_AA)
    ang = np.radians(-80)
    tip = (int(c + 0.5 * sc * np.cos(ang)), int(c + 0.5 * sc * np.sin(ang)))
    cv2.arrowedLine(img, (tip[0] + 8, tip[1] + 2), tip, (60, 200, 230), 2, tipLength=1.0)
    cv2.line(img, (c, c), (c, c + int(0.45 * sc)), (120, 120, 120), 1, cv2.LINE_AA)  # straight behind
    if trailing:
        hd, to = trailing.get("head", {}), trailing.get("torso", {})
        if trailing.get("status") == "resolved":
            th = np.radians(90 + hd["corrected_median"])
            end = (int(c + 0.45 * sc * np.cos(th)), int(c + 0.45 * sc * np.sin(th)))
            cv2.line(img, (c, c), end, (60, 200, 230), 1, cv2.LINE_AA)
            _put(img, f"hair mass trails by {hd['corrected_median']:.0f} deg", (4, size - 10), 0.42, (60, 200, 230))
        elif hd.get("corrected_median") is not None:
            _put(img, "trailing NOT RESOLVED: after the calm offset", (4, size - 26), 0.4, (60, 200, 230))
            _put(img, f"{hd['corrected_median']:+.0f} deg vs head, {to.get('corrected_median', float('nan')):+.0f} vs "
                      "torso", (4, size - 10), 0.4, (60, 200, 230))
    _put(img, "TOP view, turn frames: hair frequency", (4, 14), 0.4)
    _put(img, "head faces up (green); turns mirrored to spin", (4, 30), 0.38)
    _put(img, "anticlockwise (yellow arc)", (4, 46), 0.38)
    return img


def reproject(ctx: Context, h: dict, v: dict, torch=None, dev: str = "cpu") -> dict | None:
    """Hair voxels of hull h drawn into view v (native pixels): visible (in front of her rendered body) and hidden
    disks, + the camera's hair / full / lead masks. None when the frame has no masks."""
    from .avatar_raster import rasterize
    from .avatar_texture import project
    from .face_common import project as proj_np
    cam, i = v["cam"], v["i"]
    tr = ctx.tracks[cam]
    hm = _read(_mask_path(ctx, cam, "hair", i))
    fm = _read(_mask_path(ctx, cam, "masks", i))
    if hm is None or fm is None:
        return None
    Hh, Ww = hm.shape
    s = v.get("scale", 1.0)
    P = np.asarray(h["hair"], float)
    px, z = proj_np(P, tr.K[i], tr.R[i], tr.tvec[i], float(tr.k1[i]))
    if torch is None:
        import torch
    Vt = torch.tensor(ctx.body.verts[h["k"]], dtype=torch.float32, device=dev)
    pxv, zv, _ = project(torch, Vt, torch.tensor(tr.K[i], dtype=torch.float32, device=dev),
                         torch.tensor(tr.R[i], dtype=torch.float32, device=dev),
                         torch.tensor(tr.tvec[i], dtype=torch.float32, device=dev), float(tr.k1[i]))
    zb, _ = rasterize(torch, pxv, zv, torch.tensor(ctx.faces, device=dev), Hh, Ww)
    zb = zb.cpu().numpy()
    ok = (z > 0.1) & (px[:, 0] >= 0) & (px[:, 0] < Ww) & (px[:, 1] >= 0) & (px[:, 1] < Hh)
    pi = px[ok].astype(int)
    hidden = zb[pi[:, 1], pi[:, 0]] < z[ok] - ctx.cfg.occl_m
    shift = np.array(v.get("shift_px", [0, 0])) / s
    rad = max(1, int(round(0.5 * ctx.cfg.voxel_m * tr.K[i][0, 0] / np.median(z[ok])))) if ok.any() else 1
    q = np.round(px[ok] + shift).astype(int)
    vis, hid = splat_disks(q[~hidden], (Hh, Ww), rad), splat_disks(q[hidden], (Hh, Ww), rad)
    ph, zh = proj_np(ctx.body.joints[h["k"], J_HEAD][None], tr.K[i], tr.R[i], tr.tvec[i], float(tr.k1[i]))
    return {"vis": vis, "hid": hid & ~vis, "hair": hm, "full": fm,
            "lead": _read(_mask_path(ctx, cam, "masks", i, ctx.lead_pid)),
            "head_px": ph[0] + shift, "head_z": float(zh[0]), "f": float(tr.K[i][0, 0])}


def splat_disks(q: np.ndarray, shape: tuple[int, int], rad: int) -> np.ndarray:
    """Union of filled disks of radius rad (px) centred on the integer points q (N,2) x,y: one splat + one dilation
    instead of a cv2.circle per voxel."""
    m = np.zeros(shape, np.uint8)
    if len(q):
        ok = (q[:, 0] >= 0) & (q[:, 0] < shape[1]) & (q[:, 1] >= 0) & (q[:, 1] < shape[0])
        m[q[ok, 1], q[ok, 0]] = 1
        if rad > 0:
            k = np.zeros((2 * rad + 1, 2 * rad + 1), np.uint8)
            cv2.circle(k, (rad, rad), rad, 1, -1)  # the very disk cv2.circle draws
            m = cv2.dilate(m, k)
    return m.astype(bool)


def _rp_scores(r: dict) -> tuple[float, float] | None:
    """recall = hair-mask pixels covered by re-projected voxels (visible or hidden); precision = visible-voxel pixels
    inside her hair mask, not counting pixels of the lead's mask (he may be in front)."""
    hm, vis = r["hair"], r["vis"]
    if hm.sum() < 50 or vis.sum() < 50:
        return None
    lead = r["lead"] if r["lead"] is not None else np.zeros_like(hm)
    vv = vis & ~(lead & ~r["full"])
    return np.count_nonzero(hm & (vis | r["hid"])) / hm.sum(), np.count_nonzero(hm & vv) / max(vv.sum(), 1)


def hull_review(ctx: Context, hulls: list[dict], path: Path, n_rows: int = 4, tile_h: int = 330) -> dict:
    """3D hair hull re-projected into the videos: hair mask (red) + visible voxels (cyan outline) + voxels hidden
    behind her body (orange outline), upper-body crops of every camera. CPU only (32 views)."""
    import torch
    dev = "cpu"
    good = [h for h in hulls if len(h["hair"]) > 200]
    if not good:
        return {"status": "no hull"}
    picks = [good[int(x)] for x in np.linspace(0, len(good) - 1, min(n_rows, len(good)))]
    rows, rec, prec = [], [], []
    for h in picks:
        tiles = []
        for v in h["views"]:
            r = reproject(ctx, h, v, torch, dev)
            img = cv2.imread(str(ctx.work / "frames" / v["cam"] / f"{v['i']:06d}.jpg"))
            if r is None or img is None:
                continue
            sc = _rp_scores(r)
            if sc and "skip" not in v and "carve_only" not in v:
                rec.append(sc[0])
                prec.append(sc[1])
            ov = img.copy()
            ov[r["hair"]] = (0.45 * ov[r["hair"]] + 0.55 * np.array([40, 40, 235])).astype(np.uint8)
            th = max(1, img.shape[1] // 450)
            cnts, _ = cv2.findContours(r["hid"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(ov, cnts, -1, (0, 160, 255), th)
            cnts, _ = cv2.findContours(r["vis"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(ov, cnts, -1, (255, 255, 0), th + 1)
            hx, hy = r["head_px"]
            side = int(1.5 * r["f"] / max(r["head_z"], 0.5))
            x0, y0 = int(hx - side / 2), int(hy - 0.3 * side)
            crop = np.zeros((side, side, 3), np.uint8)
            xs0, ys0 = max(0, x0), max(0, y0)
            xs1, ys1 = min(img.shape[1], x0 + side), min(img.shape[0], y0 + side)
            if xs1 > xs0 and ys1 > ys0:
                crop[ys0 - y0: ys1 - y0, xs0 - x0: xs1 - x0] = ov[ys0:ys1, xs0:xs1]
            crop = cv2.resize(crop, (tile_h, tile_h))
            lab = f"cam {v['cam']} #{v['i']}" + (" SKIPPED" if "skip" in v else (" carve-only" if "carve_only" in v
                                                                                   else ""))
            if sc:
                lab += f"  R {sc[0]:.2f} P {sc[1]:.2f}"
            _put(crop, lab, (4, 16), 0.45)
            tiles.append(crop)
        if tiles:
            row = np.hstack(tiles)
            bar = np.full((22, row.shape[1], 3), 20, np.uint8)
            _put(bar, f"t = {h['t']:.2f} s  grid frame {h['k']}  hair voxels {len(h['hair'])}  views {h['n_views']}",
                 (6, 16), 0.5)
            rows.append(np.vstack([bar, row]))
    Wmax = max(r.shape[1] for r in rows)
    rows = [np.hstack([r, np.full((r.shape[0], Wmax - r.shape[1], 3), 20, np.uint8)]) for r in rows]
    head = np.full((44, Wmax, 3), 20, np.uint8)
    _put(head, "HAIR HULL CHECK: red = 2D hair mask, cyan = 3D hair voxels re-projected (visible), orange = voxels "
               "hidden behind her own body in that view (not a miss); R = recall, P = precision", (8, 18), 0.5)
    _put(head, f"median over shown views: hair-mask recall {np.median(rec) if rec else float('nan'):.2f}, "
               f"visible-voxel precision {np.median(prec) if prec else float('nan'):.2f}", (8, 38), 0.5)
    img = np.vstack([head] + rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return {"recall_median": round(float(np.median(rec)), 3) if rec else None,
            "precision_median": round(float(np.median(prec)), 3) if prec else None, "n_views": len(rec)}


def projection_consistency(ctx: Context, hulls: list[dict], torch, dev: str, step: int = 4) -> dict:
    """Recall / precision of the re-projected hull vs the 2D hair masks over many frames (numbers for the json).
    Runs on the caller's device: inside the hull's gpu_section, or on the CPU when the hull is reused."""
    rec, prec = {}, {}
    t0 = time.time()
    for h in hulls[::step]:
        if len(h["hair"]) < 50:
            continue
        for v in h["views"]:
            if "skip" in v or "carve_only" in v:
                continue
            r = reproject(ctx, h, v, torch, dev)
            sc = _rp_scores(r) if r is not None else None
            if sc:
                rec.setdefault(v["cam"], []).append(sc[0])
                prec.setdefault(v["cam"], []).append(sc[1])
    per = {c: {"recall": round(float(np.median(rec[c])), 3), "precision": round(float(np.median(prec[c])), 3),
               "n": len(rec[c])} for c in sorted(rec)}
    allr = np.concatenate([rec[c] for c in rec]) if rec else np.zeros(0)
    allp = np.concatenate([prec[c] for c in prec]) if prec else np.zeros(0)
    return {"per_camera": per, "recall_median": round(float(np.median(allr)), 3) if len(allr) else None,
            "precision_median": round(float(np.median(allp)), 3) if len(allp) else None,
            "every_nth_hull_frame": step, "device": str(dev), "seconds": round(time.time() - t0, 1),
            "definition": "recall = share of 2D hair-mask pixels covered by the re-projected 3D hair voxels; precision "
                          "= share of re-projected VISIBLE voxel pixels (not behind her rendered body, not where the "
                          "lead's mask is) inside the hair mask; voxels are drawn as 1.5 cm disks, so precision is "
                          "lowered by the disk overhang at the hair outline"}


def reference_sheet(ctx: Context, path: Path, metrics: dict, extra: dict, palette: dict, portrait: dict,
                    recipe: dict, colour: dict, numbers: list[str]) -> None:
    """review/hair_reference.jpg: portrait | video crops | 3D occupancy (back, side) | turns | palette | bloom."""
    W = 2200
    rows = []
    # --- row 1: portrait (mask + zones) and video crops with hair mask + tip level ---
    tiles = []
    if portrait.get("status") == "ok":
        im = cv2.imread(portrait["path"])
        ms = portrait["_masks"]
        ov = im.copy()
        ys, xs, zone = ms["zone"]
        cols = {0: (60, 200, 255), 1: (60, 255, 60), 2: (255, 120, 255)}
        for z, c in cols.items():
            s = zone == z
            ov[ys[s], xs[s]] = (0.45 * ov[ys[s], xs[s]] + 0.55 * np.array(c)).astype(np.uint8)
        cnts, _ = cv2.findContours(ms["hair"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(ov, cnts, -1, (255, 255, 255), 1)
        both = np.hstack([im, ov])
        both = cv2.resize(both, (int(both.shape[1] * 420 / both.shape[0]), 420))
        _put(both, "portrait | hair (MediaPipe) zones: root yellow, mid green, tips pink", (6, 16), 0.45)
        tiles.append(both)
    vid = extra.get("video_crops")
    if vid is not None:
        tiles.append(cv2.resize(vid, (int(vid.shape[1] * 420 / vid.shape[0]), 420)))
    if tiles:
        r = np.hstack(tiles)
        if r.shape[1] > W:
            r = cv2.resize(r, (W, int(r.shape[0] * W / r.shape[1])))
        rows.append(r)
    # --- row 2: occupancy back/side, turn top view, motion plot ---
    lm = extra["landmarks"]
    crown = lm["crown"]
    lines = [("crown", 0.0, (255, 255, 255)), ("neck", lm["neck_joint"] - crown, (180, 180, 180)),
             ("shoulders", lm["shoulder_joints"] - crown, (120, 200, 255)),
             ("shoulder-blade tips", lm["shoulder_blade_lower_tip_approx"] - crown, (120, 255, 200)),
             ("mid-back", lm["mid_back_approx"] - crown, (200, 200, 120)),
             ("waist", lm["waist_min_perimeter"] - crown, (200, 160, 255))]
    tip = metrics["length"]["tip_level_back_rest_m"]["median"]
    if tip is not None:
        lines.append(("TIPS (rest, median)", tip - crown, (60, 60, 255)))
    posed = metrics["length"]["posed_vertical_extent_below_crown"]["calm_median_world_m"]
    if posed is not None:
        lines.append(("POSED tips (calm)", -posed, (60, 160, 255)))
    back = _occupancy_panel(extra["calm_clouds"], lines, metrics, "back", ctx.cfg.voxel_m)
    side = _occupancy_panel(extra["calm_clouds"], lines, metrics, "side", ctx.cfg.voxel_m)
    top = _turn_panel(extra["turn_clouds"], metrics["motion"].get("trailing"), 360)
    top = np.vstack([top, np.full((520 - 360, 360, 3), 18, np.uint8)])
    mot = extra.get("motion_plot")
    r2 = [back, side, top]
    if mot is not None:
        r2.append(cv2.resize(mot, (int(mot.shape[1] * 520 / mot.shape[0]), 520)))
    r2 = np.hstack(r2)
    r2 = np.vstack([r2, np.full((22, r2.shape[1], 3), 18, np.uint8)])
    _put(r2, "calm frames (hair frequency red, body grey); solid lines = REST-body heights below the crown, dashed = "
             "posed calm tip level", (6, r2.shape[0] - 6), 0.42)
    rows.append(r2)
    # --- row 3: palettes + bloom strips ---
    sw = 200
    vrow = [_swatch(palette["overall"].get(n), sw, 110, f"video {n}") for n in ("root", "mid", "tip", "highlight", "shadow")]
    prow = [_swatch(portrait.get("palette", {}).get(n), sw, 110, f"portrait {n}") for n in ("root", "mid", "tip", "highlight", "shadow")]
    raw = [_swatch(palette["overall"].get("raw"), sw, 110, "video raw (no WB)"),
           _swatch(portrait.get("palette", {}).get("raw"), sw, 110, "portrait raw (no WB)")]
    ist = palette.get("image_stats", {})
    irow = [_swatch(ist.get(n), sw, 110, f"video image {n}") for n in ("p10", "p50", "p90")]
    note = np.full((110, 3 * sw, 3), 30, np.uint8)
    for n_, line in enumerate(["OBSERVED colours (white-balanced on her top),",
                               "they include the hair's own shading: NOT an albedo.",
                               "Acceptance: the rendered hair's p10 / p50 / p90",
                               "pixels under the demo lighting match these."]):
        _put(note, line, (6, 22 + 22 * n_), 0.45)
    pal = np.vstack([np.hstack(vrow + [raw[0]]), np.hstack(prow + [raw[1]]), np.hstack(irow + [note])])
    # per-camera mid colours
    pc = palette["per_camera"]
    cam_tiles = [_swatch(pc[c].get("mid"), 75, 70, f"cam {c}", compact=True) for c in sorted(pc)]
    cam_strip = np.hstack(cam_tiles) if cam_tiles else np.zeros((70, 10, 3), np.uint8)
    base = np.array(palette["overall"]["tip"]["linear_srgb"]) if "tip" in palette["overall"] else None
    s1 = render_bloom_strip(recipe, None)
    s2 = render_bloom_strip(recipe, base)
    _put(s1, "ORIGINAL strand (6 pts, VertexColor x 3.1, URP bloom thr 1 / int 1 / scatter 0.7)", (6, 14), 0.4)
    _put(s2, "target: observed tip colour + the same continuous white emission E(s)", (6, 14), 0.4)
    bl = np.vstack([s1, s2])
    right = np.vstack([cam_strip[:, :min(cam_strip.shape[1], bl.shape[1])] if cam_strip.shape[1] else cam_strip, bl])
    right = np.hstack([right, np.full((right.shape[0], max(0, 600 - right.shape[1]), 3), 30, np.uint8)])
    hgt = max(pal.shape[0], right.shape[0])
    pad = lambda a: np.vstack([a, np.full((hgt - a.shape[0], a.shape[1], 3), 20, np.uint8)])
    rows.append(np.hstack([pad(pal), pad(right)]))
    # --- row 4: numbers (wrapped to the sheet width) ---
    import textwrap
    wrapped = []
    for line in numbers:
        parts = textwrap.wrap(line, 175)
        wrapped += parts[:1] + ["    " + q for q in parts[1:]]
    txt = np.full((22 * (len(wrapped) + 1) + 10, W, 3), 20, np.uint8)
    for n, line in enumerate(wrapped):
        _put(txt, line, (10, 24 + 22 * n), 0.52)
    rows.append(txt)
    Wmax = max(r.shape[1] for r in rows)
    rows = [np.hstack([r, np.full((r.shape[0], Wmax - r.shape[1], 3), 20, np.uint8)]) for r in rows]
    title = np.full((36, Wmax, 3), 20, np.uint8)
    _put(title, f"HAIR REFERENCE - {follow_meta(ctx.take)['name']} ({ctx.take.name}) - likeness data, local only", (8, 25), 0.7)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.vstack([title] + rows), [cv2.IMWRITE_JPEG_QUALITY, 90])


def video_crops(ctx: Context, hulls: list[dict], extra: dict, h: int = 420) -> np.ndarray | None:
    """One calm frame per camera, chosen where most of her back is visible: crop of her upper body with the hair mask
    outline and the measured median tip level drawn on her back (rest-height iso-line of the fitted body, only where
    that part of her back faces the camera and the lead does not cover it)."""
    from .avatar_uv import vertex_normals
    from .face_common import project as proj_np
    pf = extra["pf"]
    tip = extra.get("tip_rest")
    calm_k = pf["k"][pf["calm"].astype(bool)] if len(pf["k"]) else np.zeros(0, int)
    if not len(calm_k) or tip is None:
        return None
    rest_y = ctx.rest_v[:, 1]
    band = np.nonzero((np.abs(rest_y - tip) < 0.012) & (ctx.rest_n[:, 2] < -0.2) & ~ctx.arm_v)[0]
    tiles = []
    for cam, tr in ctx.tracks.items():
        best = None
        for k in calm_k[:: max(1, len(calm_k) // 30)]:
            nf = nearest_frame(tr, float(ctx.body.t[k]))
            if nf is None:
                continue
            i = nf[0]
            lead = _read(_mask_path(ctx, cam, "masks", i, ctx.lead_pid))
            V = ctx.body.verts[k].astype(float)
            C = -tr.R[i].T @ tr.tvec[i]
            n = vertex_normals(V, ctx.faces)[band]
            facing = np.einsum("ij,ij->i", n, C - V[band]) > 0
            px, z = proj_np(V[band], tr.K[i], tr.R[i], tr.tvec[i], float(tr.k1[i]))
            q = px.astype(int)
            Hh, Ww = (lead.shape if lead is not None else (10 ** 6, 10 ** 6))
            inside = (q[:, 0] >= 0) & (q[:, 0] < Ww) & (q[:, 1] >= 0) & (q[:, 1] < Hh)
            free = inside.copy()
            if lead is not None:
                free[inside] = ~lead[q[inside, 1], q[inside, 0]]
            vis = facing & free
            if best is None or vis.sum() > best[2]:
                best = (k, i, int(vis.sum()), q[vis])
        if best is None:
            continue
        k, i, nvis, q = best
        img = cv2.imread(str(ctx.work / "frames" / cam / f"{i:06d}.jpg"))
        hm = _read(_mask_path(ctx, cam, "hair", i))
        fm = _read(_mask_path(ctx, cam, "masks", i))
        if img is None or fm is None or not fm.any():
            continue
        if hm is not None:
            cnts, _ = cv2.findContours(hm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cnts, -1, (255, 255, 0), max(1, img.shape[1] // 500))
        for pt in q:
            cv2.circle(img, (int(pt[0]), int(pt[1])), max(1, img.shape[1] // 350), (60, 60, 255), -1)
        yy, xx = np.nonzero(fm)
        y0, y1 = yy.min(), yy.max()
        hh = int(0.65 * (y1 - y0))
        cx = int(np.median(xx[yy < y0 + hh]))
        half = int(hh * 0.38)
        crop = img[max(0, y0 - 8): y0 + hh, max(0, cx - half): cx + half]
        crop = cv2.resize(crop, (int(crop.shape[1] * h / max(crop.shape[0], 1)), h))
        _put(crop, f"cam {cam} t={ctx.body.t[k]:.2f}", (4, 16), 0.42)
        tiles.append(crop)
    if not tiles:
        return None
    out = np.hstack(tiles)
    _put(out, "video (calm frames, back most visible): cyan = hair mask, red dots = measured median tip level on her "
              "back (fitted body)", (6, h - 8), 0.45)
    return out


def colour_tolerances(palette: dict) -> dict:
    """Acceptance tolerances from the measured cross-camera spread: hue 10 deg (the hue is consistent across phones),
    saturation max(0.10, 1.5 SD), lightness 1.5 SD of the per-camera mid L* clamped to 10..14 L*."""
    sp = palette.get("camera_spread", {})
    lsd = sp.get("mid_L_sd") or 8.0
    ssd = sp.get("mid_saturation_sd") or 0.05
    return {"hue_deg": 10.0, "saturation": round(max(0.10, 1.5 * ssd), 2),
            "lightness_L": round(float(np.clip(1.5 * lsd, 10.0, 14.0)), 1)}


def acceptance(metrics: dict, palette: dict, recipe: dict, scale: dict) -> dict:
    L = metrics["length"]
    V = metrics["volume"]
    M = metrics["motion"]
    pal = palette["overall"]
    ist = palette.get("image_stats", {})
    tol = colour_tolerances(palette)
    emi = recipe["emission_along_strand"]
    sk = emi.get("soft_knee", {})
    sp = M["spread_radius_r95_world_m"]
    trail = M.get("trailing", {})
    out = {
        "length": {"tip_level_rest_m": L["tip_level_back_rest_m"]["median"], "tolerance_m": 0.04,
                   "text": f"back tips {L['tips_fall']}; tip level within +-4 cm (avatar rest heights, world m) of "
                           f"{L['tip_level_back_rest_m']['median']} m above the sole"},
        "strand_length_true_m": L["strand_length_crown_to_tip_back"]["true_m"],
        "front_tip_level_rest_m": {"at_most": L["tip_level_front_rest_m"]["median"],
                                   "text": "front sections reach at least this rest height (world m; lower bound of "
                                           "the front length: the hull misses occluded front hair)"},
        "width_at_shoulder_level_world_m": {"max": V["width_at_shoulder_level"]["world_m"],
                                            "text": "upper bound (visual hull); the hair silhouette at shoulder level "
                                                    "must not exceed it and should be >= 75 % of it"},
        "spread_in_turns_world_m": {"calm_r95": sp["calm_median"], "turn_r95_upper_bound": sp["turn_median"],
                                    "text": "in turns faster than 180 deg/s the hair spreads beyond its calm r95 "
                                            f"({sp['calm_median']} m world) but not beyond the turn upper bound "
                                            f"({sp['turn_median']} m world; motion blur inflates the turn hull)"},
        "colour": {"image_stats": {n: ist.get(n, {}).get("hex") for n in ("p10", "p50", "p90")},
                   "image_stats_linear": {n: ist.get(n, {}).get("linear_srgb") for n in ("p10", "p50", "p90")},
                   "tolerance": tol,
                   "observed_zones": {z: pal.get(z, {}).get("hex") for z in ("root", "mid", "tip", "highlight", "shadow")},
                   "uniform_root_to_tip": True,
                   "text": "render the hair under the demo lighting, white-balance the render on her white top the "
                           "same way, then the hair pixels' p10 / p50 / p90 (by luminance) must match "
                           f"{ist.get('p10', {}).get('hex')} / {ist.get('p50', {}).get('hex')} / "
                           f"{ist.get('p90', {}).get('hex')}: hue within {tol['hue_deg']:.0f} deg, HSV saturation "
                           f"within {tol['saturation']:.2f}, lightness within {tol['lightness_L']:.0f} L*; root, mid and "
                           "tip medians within dE76 6 of each other (no darker roots). The palette is OBSERVED "
                           "(it includes the hair's own shading): do not use it as the albedo"},
        "tip_bloom": {"emission_linear_at_tip": emi["tip_value"], "glow_from_strand_fraction": emi["above_threshold_from_s"],
                      "colour": "white", "bloom": {k: recipe["bloom_volume"].get(k) for k in
                                                   ("threshold", "intensity", "scatter")},
                      "lit_hair_max_linear": sk.get("lit_hair_max_linear", LIT_HAIR_MAX_LINEAR),
                      "text": f"only the last ~{100 * (1 - emi['above_threshold_from_s']):.0f} % of every strand "
                              "exceeds the bloom threshold; URP's soft knee starts at "
                              f"{sk.get('starts_at_linear', 0.5)} linear, so the lit hair body must stay <= "
                              f"{sk.get('lit_hair_max_linear', LIT_HAIR_MAX_LINEAR)} linear (max channel) to keep the "
                              "rest of the hair from glowing"},
        "not_checked": {},
    }
    if trail.get("status") == "resolved":
        out["trailing_angle_deg"] = {"median": trail["head"]["corrected_median"], "tolerance_deg": 15}
    else:
        out["not_checked"]["trailing_angle_and_lag"] = (
            "not resolved in this window (offset-corrected trailing within noise, turn directions disagree); "
            "re-measure on a longer take before using it as a target")
    return out


def unity_notes(metrics: dict, palette: dict, recipe: dict, scale: dict) -> list[str]:
    """Plain instructions for the Unity hair, generated from the numbers (also in docs/HAIR_REFERENCE.md, generic)."""
    L, V, M = metrics["length"], metrics["volume"], metrics["motion"]
    acc = acceptance(metrics, palette, recipe, scale)
    pal, ist = palette["overall"], palette.get("image_stats", {})
    emi = recipe["emission_along_strand"]
    sk = emi.get("soft_knee", {})
    tol = acc["colour"]["tolerance"]
    s_v = list(zip(emi["s"][::2], emi["linear_hdr"][::2]))
    sp = M["spread_radius_r95_world_m"]
    crown = metrics["landmarks_rest_m_above_sole_world"]["crown"]
    notes = [
        f"Colour target = image statistics, not albedo: under the demo lighting the rendered hair's p10 / p50 / p90 "
        f"pixels (white-balanced on her top) should read {ist.get('p10', {}).get('hex')} / "
        f"{ist.get('p50', {}).get('hex')} / {ist.get('p90', {}).get('hex')} (hue +-{tol['hue_deg']:.0f} deg, saturation "
        f"+-{tol['saturation']:.2f}, lightness +-{tol['lightness_L']:.0f} L*). The observed mid "
        f"{pal.get('mid', {}).get('hex')} already contains the hair's own shading: start the albedo lighter than it "
        f"(same hue) and tune until the statistics match. Highlight lobe toward {pal.get('highlight', {}).get('hex')}, "
        f"shadows toward {pal.get('shadow', {}).get('hex')}; uniform from root to tip (no darker roots).",
        f"Keep the lit hair body <= {sk.get('lit_hair_max_linear', LIT_HAIR_MAX_LINEAR)} linear (max channel): URP's "
        f"bloom soft knee starts at {sk.get('starts_at_linear', 0.5)} (0.6 -> "
        f"{100 * sk.get('bloom_share_at_linear', {}).get('0.6', 0):.1f} % bloom, 1.0 -> "
        f"{100 * sk.get('bloom_share_at_linear', {}).get('1.0', 0):.1f} %).",
        f"Shape: back tips at rest height {L['tip_level_back_rest_m']['median']} +-0.04 m on the current avatar "
        f"({L['tip_level_back_rest_m']['fraction_of_rest_height']} x avatar height; {L['tips_fall']}); back strands "
        f"{L['strand_length_crown_to_tip_back']['world_m'] / crown:.3f} x avatar height from the crown; front sections "
        f"reach at least rest height {L['tip_level_front_rest_m']['median']} m (lower bound; the portrait shows them "
        f"longer). Silhouette widest {V['width_at_shoulder_level']['world_m']} m (world, upper bound) at the shoulder "
        f"line, >= 75 % of it.",
        f"Motion: in turns faster than 180 deg/s the hair spreads beyond its calm r95 ({sp['calm_median']} m world) up "
        f"to at most {sp['turn_median']} m (upper bound). No trailing-angle or lag target: "
        + ("the trailing angle is resolved: " + str(M['trailing']['head']['corrected_median']) + " deg"
           if M.get("trailing", {}).get("status") == "resolved" else
           "not resolved in this window (let the simulation decide)."),
        "Tip bloom: add WHITE emission along the normalised strand arc length s, piecewise linear through "
        + ", ".join(f"({a:.1f}, {b:.3f})" for a, b in s_v)
        + f"; above the bloom threshold from s = {emi['above_threshold_from_s']}. Do not re-evaluate the authored "
          "gradient and do not use its (4, 3.2, 1.5) key (Unity always clamped it to white). Keep the 10 mm -> 1 mm "
          "taper, HDR on, Bloom threshold 1 / intensity 1 / scatter 0.7, Tonemapping None; do not scale the emission "
          "by the avatar opacity unless asked; re-check URP bloom cost on the Quest tier.",
    ]
    return notes


def numbers_lines(metrics: dict, palette: dict, portrait: dict, recipe: dict, scale: dict, cons: dict) -> list[str]:
    L, V, M = metrics["length"], metrics["volume"], metrics["motion"]
    fr = metrics["frames"]
    f = lambda x, n=2: "n/a" if x is None else f"{x:.{n}f}"
    tr = M.get("trailing", {})
    hd, to = tr.get("head", {}), tr.get("torso", {})
    reg = fr.get("registration", {})
    lines = [
        f"frames: hull {fr['hull']}, calm {fr['calm']}, turn {fr['turn']}, left out {fr['left_out_unreliable']} "
        f"({fr.get('left_out_summary', '')})  |  world scale: true = world x {scale['true_per_world']} (SMPL-X rest "
        f"height {scale['rest_height_world_m']} m vs stature {scale['stature_m']} + sole)",
        f"LENGTH: back tips {L['tips_fall']}: rest height {f(L['tip_level_back_rest_m']['median'])} m world (3D hull "
        f"{f(L['tip_level_back_rest_m']['hull']['median'])}, single-view ray-cast "
        f"{f(L['tip_level_back_rest_m']['single_view_raycast']['median'])}) = {f(L['tip_drop_below_crown_rest']['true_m'])} m "
        f"TRUE below the crown ({f(100 * L['tip_drop_below_crown_rest']['fraction_of_height'], 1)} % of height); strand "
        f"crown->tip along the back {f(L['strand_length_crown_to_tip_back']['true_m'])} m TRUE (longest quartile "
        f"{f(L['strand_length_crown_to_tip_back']['longest_true_m'])}); front tips at rest "
        f"{f(L['tip_level_front_rest_m']['median'])} m world or lower (lower bound of the front length)",
        f"VOLUME (calm, upper bounds, TRUE m): width at ear level {f(V['width_at_ear_level']['true_m'])} "
        f"({f(V['width_at_ear_level']['ratio_to_head'])} x head), at shoulder level "
        f"{f(V['width_at_shoulder_level']['true_m'])} ({f(V['width_at_shoulder_level']['ratio_to_shoulders'])} x "
        f"shoulders), hair layer stands off her back by {f(V['back_layer_standoff_below_shoulders']['true_m'])} (p90)",
        f"MOTION (TRUE m): spread r95 calm {f(M['spread_radius_r95_true_m']['calm_median'])} -> turns "
        f"{f(M['spread_radius_r95_true_m']['turn_median'])} (p90 {f(M['spread_radius_r95_true_m']['turn_p90'])}; turn "
        f"values are upper bounds, {f(M.get('hull_voxels_turn_over_calm'))} x the calm voxels); head yaw p90 "
        f"{f(M['head_yaw_rate_dps']['p90_abs'], 0)} deg/s",
        f"TRAILING in turns: {tr.get('status', 'n/a').replace('_', ' ').upper()} - raw {f(hd.get('raw_median'), 0)} deg "
        f"is the calm offset ({f(hd.get('calm_offset_deg'), 0)} deg, hair mass toward one shoulder); corrected "
        f"{f(hd.get('corrected_median'), 0)} deg vs head (CI {', '.join(f(x, 0) for x in hd.get('corrected_median_ci95') or [])}), "
        f"{f(to.get('corrected_median'), 0)} deg vs torso; lag {M.get('lag_time_s', {}).get('status', 'n/a').replace('_', ' ')} "
        f"(fit residual {f(M.get('lag_time_s', {}).get('residual_rms_deg'), 0)} deg RMS)",
        f"3D hull vs 2D hair masks: recall {cons.get('recall_median')}, precision {cons.get('precision_median')}; "
        f"registration at the +-{reg.get('shift_limit_px')} px search limit in {reg.get('at_shift_limit')} of "
        f"{reg.get('registered_views')} views {reg.get('at_shift_limit_per_camera')}",
    ]
    po = portrait.get("palette", {})
    pv = palette["overall"]
    ist = palette.get("image_stats", {})
    sp = palette.get("camera_spread", {})
    lines.append("COLOUR (OBSERVED, white-balanced on her white top, linear sRGB; includes shading, not an albedo): "
                 f"video root {pv.get('root', {}).get('hex')} mid {pv.get('mid', {}).get('hex')} tip "
                 f"{pv.get('tip', {}).get('hex')} highlight {pv.get('highlight', {}).get('hex')} shadow "
                 f"{pv.get('shadow', {}).get('hex')}; image p10/p50/p90 {ist.get('p10', {}).get('hex')} / "
                 f"{ist.get('p50', {}).get('hex')} / {ist.get('p90', {}).get('hex')}; per-camera mid L* "
                 f"{sp.get('mid_L_range')} (SD {sp.get('mid_L_sd')}) | portrait root {po.get('root', {}).get('hex')} "
                 f"mid {po.get('mid', {}).get('hex')} tip {po.get('tip', {}).get('hex')}")
    e = recipe["emission_along_strand"]
    lines.append(f"TIP BLOOM (original): strand colour = baked vertex colour x {recipe['shader']['multiplier']:.2f} (unlit); "
                 f"linear output at the 6 points {e['at_vertices']}; above the bloom threshold from s = "
                 f"{e['above_threshold_from_s']} (last {100 * (1 - e['above_threshold_from_s']):.0f} %); soft knee from "
                 f"s = {e['soft_knee_from_s']}: keep the lit hair <= {e.get('soft_knee', {}).get('lit_hair_max_linear')} linear")
    lines.append(f"    white glow; URP bloom threshold {recipe['bloom_volume'].get('threshold')} intensity "
                 f"{recipe['bloom_volume'].get('intensity')} scatter {recipe['bloom_volume'].get('scatter')}, tonemapping off, "
                 f"HDR on (Bloom.asset); 180 strands x 6 points, width 10 -> 1 mm")
    return lines


def load_observations(work: Path) -> dict | None:
    """Optional by-eye notes (parting, fringe, layering, texture, look differences) for the take, kept next to the
    measurements in hair_ref/observations.json - qualitative, so they are written by a person, not computed."""
    p = work / "hair_ref" / "observations.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def run(take: Take, cfg: HairRefConfig, portrait: Path | None = None, reuse_hull: bool = False,
        reuse_colour: bool = False) -> dict:
    work = take.work
    out_dir = work / "hair_ref"
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    set_threads(cfg.threads)
    timings: dict = {}
    hull_p, cons_p = out_dir / "hull.npz", out_dir / "consistency.json"
    compute: dict = {"threads": cfg.threads, "device_requested": cfg.device}
    need_hull = not (reuse_hull and hull_p.exists())
    if need_hull and cfg.device != "cpu" and cfg.isolate_gpu:
        # the GPU part in a child process: this process never creates a CUDA context
        compute["hull"] = hull_in_child(take, cfg)
        need_hull = False
    t0 = time.time()
    ctx = load_context(take, cfg)
    timings["load_context_s"] = round(time.time() - t0, 1)
    print(f"[hair_ref] {take.name}: {len(ctx.tracks)} cameras, {len(ctx.body.t)} grid frames, scale "
          f"{ctx.scale['true_per_world']}", flush=True)
    frames = list(range(0, len(ctx.body.t), max(1, cfg.step)))
    if need_hull:
        compute["hull"] = hull_stage(ctx, hull_p, cons_p, frames)
    hulls = load_hull(hull_p)
    cons = load_consistency(cons_p, hull_p)
    if cons is None:
        t0 = time.time()
        with gpu_section("cpu", "consistency") as (torch, dev):
            cons = projection_consistency(ctx, hulls, torch, dev)
        save_consistency(cons_p, hull_p, cons)
        timings["consistency_s"] = round(time.time() - t0, 1)
    else:
        cons.pop("hull_key", None)
        compute["consistency"] = "reused hair_ref/consistency.json (same hull.npz)"
    t0 = time.time()
    kin = head_kinematics(ctx)
    metrics, extra = shape_metrics(ctx, hulls, kin)
    extra["tip_rest"] = metrics["length"]["tip_level_back_rest_m"]["median"]
    timings["shape_motion_s"] = round(time.time() - t0, 1)
    print(f"[hair_ref] shape: tips {metrics['length']['tips_fall']}; hull/mask recall {cons['recall_median']} "
          f"precision {cons['precision_median']}; trailing {metrics['motion']['trailing']['status']}", flush=True)
    t0 = time.time()
    colour = sample_colours(ctx, out_dir, reuse=reuse_colour)
    palette = palette_from_samples(colour["samples"], cfg.top_albedo)
    timings["colour_s"] = round(time.time() - t0, 1)
    portrait = portrait or default_portrait(take)
    pr = portrait_reference(portrait, out_dir, cfg.top_albedo, cfg.mp) if portrait else {"status": "none"}
    recipe = tip_bloom_recipe()
    review = work / "review"
    nums = numbers_lines(metrics, palette, pr, recipe, ctx.scale, cons)
    hull_qa = {"status": "skipped (--no-review)"}
    if cfg.review:
        t0 = time.time()
        extra["motion_plot"] = _plot_motion(extra["pf"], metrics, out_dir / "motion.png") if len(extra["pf"]["k"]) else None
        extra["video_crops"] = video_crops(ctx, hulls, extra)
        hull_qa = hull_review(ctx, hulls, review / "hair_reference_hull.jpg")
        reference_sheet(ctx, review / "hair_reference.jpg", metrics, extra, palette, pr, recipe, colour, nums)
        timings["review_s"] = round(time.time() - t0, 1)
    mp_q = [s["mp"] for s in colour["samples"] if s.get("mp")]
    pv, po = palette["overall"], pr.get("palette", {})
    look = {}
    for z in ("root", "mid", "tip"):
        if z in pv and z in po:
            look[z] = {"dE76_video_vs_portrait": round(delta_e76(pv[z]["linear_srgb"], po[z]["linear_srgb"]), 1),
                       "hue_deg_video": pv[z]["hsv"][0], "hue_deg_portrait": po[z]["hsv"][0],
                       "lightness_L_video": pv[z]["lab"][0], "lightness_L_portrait": po[z]["lab"][0]}
    trend = {}
    for src, pal in (("video", pv), ("portrait", po)):
        if all(z in pal for z in ("root", "mid", "tip")):
            L = {z: pal[z]["lab"][0] for z in ("root", "mid", "tip")}
            trend[src] = {"L_root_minus_mid": round(L["root"] - L["mid"], 1), "L_tip_minus_mid": round(L["tip"] - L["mid"], 1),
                          "dE76_root_mid": round(delta_e76(pal["root"]["linear_srgb"], pal["mid"]["linear_srgb"]), 1),
                          "dE76_tip_mid": round(delta_e76(pal["tip"]["linear_srgb"], pal["mid"]["linear_srgb"]), 1),
                          "darker_roots": bool(L["root"] < L["mid"] - 3)}
    timings["total_s"] = round(time.time() - t_start, 1)
    compute["timings"] = timings
    compute["colour_views"] = {"views": colour["n_views"], "used": len(colour["samples"]),
                               "per_camera_cap": cfg.colour_views_per_cam, "min_step": cfg.colour_min_step,
                               "mediapipe": colour["mp"]}
    result = {
        "version": 2, "take": take.name, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "dancer": follow_meta(take),
        "likeness": "measurements of a consenting dancer for a non-commercial demo; keep this file and the review "
                    "images local (work/ is git-ignored)",
        "config": asdict(cfg), "scale": ctx.scale, "compute": compute,
        "inputs": {"smplx": "smplx/follow.npz", "head_fix": ctx.info["bodies"][ROLE].get("head_fix"),
                   "cameras": sorted(ctx.tracks), "sync": "sync/offsets.json (mamma_dyn.load_tracks)",
                   "portrait": str(portrait) if portrait else None, "pids": {"follow": ctx.pid, "lead": ctx.lead_pid}},
        "shape": metrics,
        "hull_consistency": cons, "hull_review": hull_qa,
        "colour": {"method": "hair mask (eroded 1 px) AND MediaPipe SelfieMulticlass hair; white balance per view on "
                             f"her white top scaled to {cfg.top_albedo}; zones by distance from the projected head "
                             f"centre: root <= {cfg.root_m} m, mid {cfg.mid_m} m .. {cfg.tip_frac} L, tips >= "
                             f"{cfg.tip_frac} L (L = p99 distance of the view)",
                   "what": palette["what"],
                   "views": len(colour["samples"]), "mediapipe": colour["mp"],
                   "mediapipe_vs_hair_mask": {"iou_median": round(float(np.median([m["iou"] for m in mp_q])), 3) if mp_q else None,
                                              "hair_mask_share_confirmed_median": round(float(np.median(
                                                  [m["hair_mask_in_mp"] for m in mp_q])), 3) if mp_q else None},
                   "video": palette, "portrait": {k: v for k, v in pr.items() if not k.startswith("_")},
                   "video_vs_portrait": look, "root_mid_tip_trend": trend},
        "observations": load_observations(work),
        "tip_bloom": recipe,
        "acceptance": acceptance(metrics, palette, recipe, ctx.scale),
        "unity_notes": unity_notes(metrics, palette, recipe, ctx.scale),
        "review": {"sheet": "review/hair_reference.jpg", "hull": "review/hair_reference_hull.jpg"} if cfg.review else None,
        "runtime_s": round(time.time() - t_start, 1),
    }
    out = work / "avatar" / "hair_reference.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else
                              (o.item() if isinstance(o, np.generic) else str(o))))
    print(f"[hair_ref] wrote {out}{' and review/hair_reference*.jpg' if cfg.review else ''} ({result['runtime_s']} s)",
          flush=True)
    for line in nums:
        print("  " + line)
    return result


def only_hull(take: Take, cfg: HairRefConfig) -> dict:
    """--only-hull (the child process of hull_in_child, or a pod step): hull.npz + consistency.json + hull_compute.json."""
    set_threads(cfg.threads)
    ctx = load_context(take, cfg)
    out_dir = take.work / "hair_ref"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = list(range(0, len(ctx.body.t), max(1, cfg.step)))
    st = hull_stage(ctx, out_dir / "hull.npz", out_dir / "consistency.json", frames)
    (out_dir / "hull_compute.json").write_text(json.dumps(st, indent=1))
    print(f"[hair_ref] hull: {st}", flush=True)
    return st


# ------------------------------------------------------------------------------------------------ Runpod job

def pod_job_spec(take: Take, cfg: HairRefConfig, max_minutes: float | None = None) -> tuple[dict, list[str]]:
    """dancecap.pod_job/1 spec for the heavy part on a Runpod GPU: hull (+ re-projection check), colour samples and
    MediaPipe (pip-installed into /workspace/mp_site on the pod), JSON; review images are drawn locally afterwards
    from the downloaded caches. Uploads only what the stage reads: the per-camera calibration files (not cameras/cache),
    the follow/lead/body/hair masks (not the SAM cache), the fit, and just the frames the colour step samples.
    Returns (spec, warnings): a missing fit or masks is expected only when the mamma / dancers jobs run first in the
    same batch."""
    from .mamma_dyn import load_tracks
    work = take.work
    wr = f"dancecap/work/{take.name}"
    rel = lambda p: Path(p).resolve().relative_to(WORKSPACE).as_posix()
    warn, inputs = [], []
    align = (PACKAGE_ROOT / "takes" / take.cfg["alignment"]).resolve()
    if align.exists():
        inputs.append(rel(align))
    cams = [s.id for s in take.sources]
    missing: dict[str, list[str]] = {}
    for cam in cams:
        for p in (work / "cameras" / f"{cam}.npz", work / "frames" / cam / "frames.json"):
            if p.exists():
                inputs.append(f"{wr}/{p.relative_to(work).as_posix()}")
            else:
                missing.setdefault(p.parent.name if p.suffix == ".npz" else "frames.json", []).append(cam)
        for kind in ("masks", "body", "hair"):
            d = work / "dancers" / cam / kind
            if d.exists():
                inputs.append(f"{wr}/dancers/{cam}/{kind}/")
            else:
                missing.setdefault(f"dancers/<cam>/{kind}/", []).append(cam)
    for what, cs in missing.items():
        warn.append(f"{what} missing locally for cameras {','.join(cs)}" + (": the 'dancers' job must run first in "
                                                                            "the batch" if "dancers" in what else ""))
    for p, why in ((work / "cameras" / "world_from_sfm.json", "cameras world step"), (work / "sync" / "offsets.json",
                   "python -m dancecap sync"), (work / "dancers" / "roles.json", "'dancers' job"),
                   (work / "smplx" / "follow.npz", "'mamma' job"), (work / "smplx" / "lead.npz", "'mamma' job")):
        if p.exists():
            inputs.append(f"{wr}/{p.relative_to(work).as_posix()}")
        else:
            warn.append(f"{p.relative_to(work).as_posix()} missing locally: needs the {why} first")
    for p in (work / "avatar" / "head_fix.npz", work / "hair_ref" / "observations.json"):
        if p.exists():
            inputs.append(f"{wr}/{p.relative_to(work).as_posix()}")
    por = default_portrait(take)
    if por is not None:
        inputs.append(rel(por))
    inputs += [rel(m) for m in MP_MODELS.values() if m.exists()]
    # frames: only those the colour step reads (all of frames/ when the window / cameras are not known yet)
    n_frames = None
    try:
        tracks = load_tracks(work)
        t0, t1 = (float(x) for x in take.cfg["window"]) if "window" in take.cfg else (-np.inf, np.inf)
        cv = colour_views_for(tracks, t0 - 0.1, t1 + 0.1, cfg)
        inputs += [f"{wr}/frames/{c}/{i:06d}.jpg" for c, i, _ in cv if (work / "frames" / c / f"{i:06d}.jpg").exists()]
        n_frames = len(cv)
    except (SystemExit, Exception) as e:  # no cameras / sync yet: the whole frames/ (the dancers job uploads it anyway)
        warn.append(f"colour frames not selectable yet ({type(e).__name__}: {str(e)[:120]}): uploading frames/")
        inputs.append(f"{wr}/frames/")
    n_grid = int(round((float(take.cfg["window"][1]) - float(take.cfg["window"][0])) * 30)) if "window" in take.cfg else 1200
    hull_min = n_grid / max(cfg.step, 1) * 1.0 / 60  # guess: ~1 s per grid frame on a pod (1.5 s on the 2080 Ti; the
    # mask decoding and registration are CPU-bound)
    mm = float(max_minutes or round(15 + 1.5 * hull_min + 10))
    cmd = (f"python -m dancecap.hair_reference {take.name} --device cuda --step {cfg.step} --threads 8 --no-review "
           f"--no-isolate-gpu")
    spec = {
        "schema": "dancecap.pod_job/1", "name": "hair_ref", "take": take.name, "max_minutes": mm,
        "needs": ["texture"],  # no extra bootstrap; 'texture' puts the SMPL-X body models into the bundle
        "after": ["dancers", "mamma", "mamma_loo"],
        "inputs": sorted(dict.fromkeys(inputs)),
        "exclude": ["*.mp4", f"{wr}/dancers/cache", f"{wr}/dancers/qa"],
        "env": {"DANCECAP_FACE_PYTHON": "python", "DANCECAP_FACE_PYTHONPATH": "/workspace/mp_site"},
        # MediaPipe only confirms the hair pixels: a failed install must not stop the hull (the colour then uses the
        # dancers hair masks alone and says so in colour.mediapipe)
        "setup": [f"python -m pip install -q --target /workspace/mp_site mediapipe=={MP_VERSION} numpy==1.26.4 || "
                  "echo 'mediapipe install FAILED: colour without the MediaPipe confirmation'"],
        "commands": [{"name": "mediapipe_check", "timeout_min": 10, "optional": True,
                      "cmd": "PYTHONPATH=/workspace/mp_site python -c \"import mediapipe, cv2; print('mediapipe', "
                             "mediapipe.__version__, 'cv2', cv2.__version__)\""},
                     {"name": "hair_reference", "timeout_min": mm - 5, "cmd": cmd}],
        "outputs": [f"{wr}/hair_ref/hull.npz", f"{wr}/hair_ref/consistency.json", f"{wr}/hair_ref/colour_samples.npz",
                    f"{wr}/hair_ref/mp/", f"{wr}/hair_ref/mp_portrait.png", f"{wr}/avatar/hair_reference.json"],
        "verify": [f"{wr}/hair_ref/hull.npz", f"{wr}/hair_ref/consistency.json", f"{wr}/hair_ref/colour_samples.npz",
                   f"{wr}/avatar/hair_reference.json"],
        "after_download": [f"python -m dancecap.hair_reference {take.name} --reuse-hull --reuse-colour --device cpu   "
                           "(local CPU, a few minutes: re-parses the tip bloom from head-movement, draws the review "
                           "images)"],
        "notes": ["hair_reference on the pod: the 3D hull (GPU + CPU, ~1 s per grid frame assumed), the re-projection check, "
                  f"colour sampling of {n_frames if n_frames is not None else 'all'} frames and their MediaPipe crops "
                  "(CPU); keeps all of it off the shared workstation",
                  "mediapipe is installed with pip --target into /workspace/mp_site and run as a subprocess "
                  "(DANCECAP_FACE_PYTHON / _PYTHONPATH), so the torch environment is untouched",
                  "the head-movement repo is not uploaded: the pod JSON carries the verified tip-bloom fallback "
                  "(parsed_from_repo false) until the local re-render"],
    }
    return spec, warn


def write_pod_job(take: Take, cfg: HairRefConfig, out: Path | None = None, max_minutes: float | None = None) -> Path:
    from .remote_jobs import parse_job
    spec, warn = pod_job_spec(take, cfg, max_minutes)
    parse_job(spec, "hair_ref")  # schema check
    p = out or take.work / "hair_ref" / "pod_job.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({**spec, "warnings": warn}, indent=1))
    for w_ in warn:
        print(f"[hair_ref] pod job: {w_}")
    return p


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m dancecap.hair_reference", description=__doc__.split("\n")[0])
    ap.add_argument("take")
    ap.add_argument("--portrait", type=Path)
    ap.add_argument("--step", type=int, default=1, help="hull every n-th grid frame")
    ap.add_argument("--colour-step", type=int, default=0, help="0 = auto (--colour-views-per-cam, step >= 6)")
    ap.add_argument("--colour-views-per-cam", type=int, default=30)
    ap.add_argument("--cams", nargs="*", default=[])
    ap.add_argument("--device", default="cuda", help="device of the 3D hull (everything else runs on the CPU)")
    ap.add_argument("--threads", type=int, default=4, help="CPU threads (torch, OpenCV, KD-trees)")
    ap.add_argument("--no-isolate-gpu", action="store_true", help="run the CUDA hull in this process (pods)")
    ap.add_argument("--no-mp", action="store_true", help="skip the MediaPipe hair confirmation (and the portrait)")
    ap.add_argument("--no-review", action="store_true", help="skip the review images (pod runs)")
    ap.add_argument("--reuse-hull", action="store_true", help="reuse hair_ref/hull.npz (+ consistency.json)")
    ap.add_argument("--reuse-colour", action="store_true", help="reuse hair_ref/colour_samples.npz when it matches")
    ap.add_argument("--only-hull", action="store_true", help="hull + re-projection check only (GPU child / pod step)")
    ap.add_argument("--write-pod-job", action="store_true", help="write hair_ref/pod_job.json and exit")
    ap.add_argument("--pod-job-out", type=Path, help="write the pod job here instead (not discovered by remote run)")
    ap.add_argument("--max-minutes", type=float)
    ap.add_argument("--normal-priority", action="store_true", help="keep the normal process priority")
    a = ap.parse_args(argv)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(k, str(a.threads))
    prio = "normal" if a.normal_priority else lower_priority()
    print(f"[hair_ref] priority {prio}, {a.threads} CPU threads", flush=True)
    cfg = HairRefConfig(step=a.step, colour_step=a.colour_step, colour_views_per_cam=a.colour_views_per_cam,
                        cams=tuple(a.cams), device=a.device, mp=not a.no_mp, threads=a.threads,
                        isolate_gpu=not a.no_isolate_gpu, review=not a.no_review)
    take = Take.load(a.take)
    if a.write_pod_job:
        print(write_pod_job(take, cfg, out=a.pod_job_out, max_minutes=a.max_minutes))
    elif a.only_hull:
        only_hull(take, cfg)
    else:
        run(take, cfg, a.portrait, a.reuse_hull, a.reuse_colour)


if __name__ == "__main__":
    main()
