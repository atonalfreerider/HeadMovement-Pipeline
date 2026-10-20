r"""mono4d track "fitb" (2026-10-19): the ONE-camera MAMMA baseline ("fit B") of phone 06, larissa_kadu_38, 58-62 s - the
yardstick every one-camera candidate (mono4d/candidates.py) must beat.

It is john_lula's single-camera path (dancecap.jl_front / jl_mono / farm.jl_profile: MAMMA_MONO=1, mamma-turing
optimization/utils/mono.py) applied to one phone of the 8-phone take, WITHOUT touching larissa_kadu_38's files:

    derive     work/larissa_kadu_38_mono06/ = junctions to larissa_kadu_38's frames/06 and dancers/06 (read only), copies of
               dancers/roles.json, cameras/world_from_sfm.json, mamma/ma_2d_native/06.{npz,_boxes.npy,manifest.json} and
               cameras/06.npz with its t moved onto larissa_kadu_38's sync-corrected clock (06's measured video offset,
               sync/offsets.json delta -30.04 ms, applied ONCE in the copy; the original is kept as t_raw). The 8-phone fit
               used exactly these times for phone 06, so the derived grid (anchored on the reference fit's t[0]) carries
               the same per-instant observations of phone 06 as the reference fit did ('check' proves it).
    prepare    (CLI)  MAMMA_MONO=1 python -m dancecap mamma prepare larissa_kadu_38_mono06 --set m06=06 --set p06=06
               --skip-2d --no-sync --grid-anchor <ref t0> --grid-k=836:955     (120 grid frames, 58.02..61.98 s on the lattice
               t_k = 30.151176844227862 + k/30; MAMMA_MONO=1 at prepare time puts MA3D_MONO_OVERRIDES in ma3d_config.yaml)
    check      the derived cam06 inputs (ma_2d landmarks, cam_int, cam_ext) equal the reference c8 fit's cam06 rows
    mono-init  mamma/mono_init.npz {pos (T, 2, 3)} = jl_mono.pelvis_positions on COCO-17 keypoints (people/06.npz rows
               matched to the lead / follow boxes of roles.json; COCO-WholeBody's first 17 joints are COCO-17) + a
               skeleton-on-partner flag from the dancer masks; mamma/stature.json = jl_mono.stature_targets (stature + sole)
    pin        mamma/pin_betas_ref.json = the 8-phone fit's betas (ONLY for the leak variant p06, never for m06)
    job        the pod job (dancecap/remote/cand_fitb/pod_job.json): fit m06 = free betas held to the owner's statures (the
               shape + pose recipe of jl_profile's shape pass) and, optional, fit p06 = the same with the reference betas
               pinned (MAMMA_FREEZE_BETAS / MAMMA_INIT_BETAS: LEAKS shape from the multi-view fit; a diagnostic only)
    import     fits/<tag>/smplx/{lead,follow}.npz -> candidate (frame 'world', joints = smplx joints[:, :22]) + meta

    python -m dancecap.mono4d.cand_fitb derive|check|mono-init|pin|job|import [--tag m06 --name mamma_mono06 --session SID]

Variant "3db" (2026-10-19): fit B + SAM 3D Body (cand_sam3d_body, candidate sam3d_body = pod_out/smplx_mask255.npz) as a
per-frame 3D pose PRIOR inside MAMMA_MONO (mamma-turing utils/mono.py pose_prior_loss, env MAMMA_POSE_PRIOR):
    prior-3db  mamma/prior_3db.npz: SAM 3D Body's SMPL-X body joints and global orientation moved from phone 06's camera
               frame to the world with that phone's track (mamma_dyn.load_tracks, the sync-corrected clock; SAM 3D Body's
               frame times through candidates.frame_to_ref), put on the fit grid (nearest frame, <= 5 ms), the
               root-relative joints and the rotations smoothed over time (Gaussian, PRIOR_SMOOTH frames: per-frame 3DB
               jitters ~6x the reference); conf per frame = the dancer's visible mask area against its median (an occluded
               dancer is where 3DB errs most); body pose + world orientation also seed the fit (instead of the heading search)
    configs-3db  mamma/ma3d_config_<tag>.yaml = fit B's config + pose_prior_loss (weight w) in all four stages, and sets
               <tag> in prepare.json (fit B's m06 inputs, own config; mamma_dyn.fit reads a set's "config")
    job-3db    dancecap/remote/cand_fitb/pod_job_3db.json: a smoke fit (2 iterations a stage), then the weights in parallel
    import-3db fits/<tag> -> candidates mamma_mono06_3db (w 2, pre-registered) / _w05 / _w8
    python -m dancecap.mono4d.cand_fitb prior-3db|configs-3db|job-3db|import-3db [--session SID]
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import time
import tomllib
from pathlib import Path

import numpy as np

from ..take import PACKAGE_ROOT

REF = "larissa_kadu_38"
TAKE = "larissa_kadu_38_mono06"
CAM = "06"
WINDOW = (58.0, 62.0)
GRID_K = (836, 955)                     # ref lattice indices inside 58.0..62.0 (t_k = 30.151176844227862 + k/30)
REF_WORK = PACKAGE_ROOT / "work" / REF
WORK = PACKAGE_ROOT / "work" / TAKE
JOB_DIR = PACKAGE_ROOT / "remote" / "cand_fitb"
POD_WS = "/workspace/HeadMovement"
SCORE_MIN = 0.35
ROLES = ("lead", "follow")
# variant 3db: SAM 3D Body as a per-frame 3D pose prior
PRIOR_SRC = REF_WORK / "mono4d" / "sam3d_body" / "pod_out" / "smplx_mask255.npz"   # = candidate sam3d_body
PRIOR_META = REF_WORK / "mono4d" / "sam3d_body" / "input" / "meta.json"
PRIOR_NPZ = WORK / "mamma" / "prior_3db.npz"
PRIOR_SMOOTH = 1.5        # frames: Gaussian sigma over the prior's root-relative joints and rotations
PRIOR_DT_MAX = 5e-3       # s: a grid instant takes the 3DB frame nearest to it within this (else no prior there)
PRIOR_SIGMA_M = 0.10      # Geman-McClure scale of pose_prior_loss (mamma-turing utils/mono.py)
PRIOR_RUNS = ("first_run", "second_run", "third_run", "fourth_run")
PRIOR_VARIANTS = {"3db_w05": (0.5, "mamma_mono06_3db_w05"), "3db": (2.0, "mamma_mono06_3db"),   # tag: (weight, candidate)
                  "3db_w8": (8.0, "mamma_mono06_3db_w8")}                                     # "3db" = pre-registered
PRIOR_SMOKE = "3db_smoke"
LR_PAIRS = ((1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21))       # SMPL-X body left / right


def log(msg: str) -> None:
    print(f"[cand_fitb {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ref_t0() -> float:
    return float(np.load(REF_WORK / "smplx" / "lead.npz")["t"][0])


# ------------------------------------------------------------------------------------------------ derive


def derive() -> dict:
    from .. import mamma_dyn as md
    rep = {"from": REF, "cam": CAM, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "items": {}}
    for sub in ("frames", "dancers"):
        rep["items"][f"{sub}/{CAM}"] = md.link_dir(WORK / sub / CAM, REF_WORK / sub / CAM)
    for rel in ("dancers/roles.json", "cameras/world_from_sfm.json", f"mamma/ma_2d_native/{CAM}.npz",
                f"mamma/ma_2d_native/{CAM}_boxes.npy", f"mamma/ma_2d_native/{CAM}.manifest.json"):
        (WORK / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REF_WORK / rel, WORK / rel)
        rep["items"][rel] = "copy"
    off = md.read_sync_offsets(REF_WORK)[CAM]
    z = dict(np.load(REF_WORK / "cameras" / f"{CAM}.npz", allow_pickle=True))
    z["t_raw"] = np.asarray(z["t"], float)
    z["t"] = md.apply_offset(z["t_raw"], off)
    z["clock_note"] = np.array(f"t = t_raw + {REF}'s sync delta for cam {CAM} ({1000 * off['delta']:+.2f} ms, rate "
                               f"{off['rate']}): the reference fit's clock")
    np.savez(WORK / "cameras" / f"{CAM}.npz", **z)
    rep["items"][f"cameras/{CAM}.npz"] = "copy, t shifted"
    rep["sync_delta_s"] = off["delta"]
    rep["sync_rate"] = off["rate"]
    (WORK / "derived.json").write_text(json.dumps(rep, indent=1))
    log(f"derived {WORK}: {rep['items']}; cam {CAM} t shifted by {1000 * off['delta']:+.2f} ms")
    return rep


def prepare_cmd() -> str:
    return (f"MAMMA_MONO=1 python -m dancecap mamma prepare {TAKE} --set m06={CAM} --set p06={CAM} --skip-2d --no-sync "
            f"--grid-anchor {ref_t0()!r} --grid-k={GRID_K[0]}:{GRID_K[1]}")


# ------------------------------------------------------------------------------------------------ check


def check() -> dict:
    """The derived camera-06 fit inputs against the reference c8 fit's camera-06 rows at the same grid instants."""
    prep = json.loads((WORK / "mamma" / "prepare.json").read_text())
    rprep = json.loads((REF_WORK / "mamma" / "prepare.json").read_text())
    g, rg = np.asarray(prep["grid"]["t"]), np.asarray(rprep["grid"]["t"])
    ri = np.array([int(np.argmin(np.abs(rg - x))) for x in g])
    out = {"T": int(len(g)), "grid": [float(g[0]), float(g[-1])], "max_grid_dt_s": float(np.abs(rg[ri] - g).max())}
    seq, rseq = prep["sets"]["m06"]["seq"], rprep["sets"]["c8"]["seq"]
    a = np.load(WORK / "mamma" / "ma_2d" / seq / f"cam{CAM}.npz")
    b = np.load(REF_WORK / "mamma" / "ma_2d" / rseq / f"cam{CAM}.npz")
    out["landmarks_xy_max_abs_diff_px"] = float(np.abs(a["landmarks"][..., :2] - b["landmarks"][ri][..., :2]).max())
    # logvar may differ by exactly ma_3d's sigma-halving compensation (build_inputs pre-scales a camera-person whose smallest
    # crop sigma in the LOADED window is > 15 px; the window is 120 frames here, 1148 in the reference). Positions are
    # identical, the weights are not: in this short window ma_3d halves the follow's sigmas (the reference does not),
    # so fit B weights the follow's 2D term 4x (variance) relative to the reference fit; the lead's is unchanged
    out["landmarks_logvar_max_abs_diff"] = float(np.abs(a["landmarks"][..., 2] - b["landmarks"][ri][..., 2]).max())
    out["sigma_halved_by_ma3d"] = {"derived": prep["sets"]["m06"]["inputs"][CAM]["sigma_halved_by_ma3d"],
                                   "reference": rprep["sets"]["c8"]["inputs"][CAM]["sigma_halved_by_ma3d"]}
    out["vis_max_abs_diff"] = float(np.abs(a["visibilities"] - b["visibilities"][ri]).max())
    ca = np.load(WORK / "mamma" / "ma_cap" / seq / "gt" / f"cam{CAM}.npz")
    cb = np.load(REF_WORK / "mamma" / "ma_cap" / rseq / "gt" / f"cam{CAM}.npz")
    out["cam_int_max_abs_diff"] = float(np.abs(ca["cam_int"] - cb["cam_int"][ri]).max())
    out["cam_ext_max_abs_diff"] = float(np.abs(ca["cam_ext"] - cb["cam_ext"][ri]).max())
    out["visible_frac"] = {r: float((a["visibilities"][:, p] > 0).any(-1).mean()) for p, r in enumerate(ROLES)}
    out["identical"] = all(out[k] < 1e-4 for k in ("landmarks_xy_max_abs_diff_px", "vis_max_abs_diff", "cam_ext_max_abs_diff")) \
        and out["cam_int_max_abs_diff"] < 1e-2
    (WORK / "mamma" / "check_vs_ref.json").write_text(json.dumps(out, indent=1))
    log(f"check: {out}")
    return out


# ------------------------------------------------------------------------------------------------ mono init


def _iou(a, b) -> float:
    from .holdout import box_iou
    return box_iou(np.asarray(a, float), np.asarray(b, float))


def _mask(i: int, pid: int):
    import cv2
    m = cv2.imread(str(WORK / "dancers" / CAM / "masks" / f"mask_{i:04d}_{pid:02d}.png"), cv2.IMREAD_GRAYSCALE)
    return None if m is None else m > 127


def role_keypoints(frames: np.ndarray) -> dict:
    """{role: kp (T, 17, 3), iou (T,), in_partner (T,)}: the people/06.npz row whose box best matches the role's
    roles.json box (IoU >= 0.5, else no measurement); in_partner = share of its confident keypoints that sit in the
    partner's mask and not in its own (a skeleton lying on the partner)."""
    pp = np.load(REF_WORK / "people" / f"{CAM}.npz", allow_pickle=True)
    roles = json.loads((WORK / "dancers" / "roles.json").read_text())["cams"][CAM]["frames"]
    T = len(frames)
    out = {r: {"kp": np.zeros((T, 17, 3)), "iou": np.zeros(T), "in_partner": np.zeros(T)} for r in ROLES}
    for k, i in enumerate(frames):
        valid = np.flatnonzero(pp["valid"][i])
        masks = {r: _mask(int(i), p) for r, p in (("lead", 1), ("follow", 2))}
        for r in ROLES:
            box = (roles[i].get(r) or {}).get("box")
            if box is None or not len(valid):
                continue
            ious = [_iou(box, pp["boxes"][i, j]) for j in valid]
            j = int(valid[int(np.argmax(ious))])
            if max(ious) < 0.5:
                continue
            kp = np.asarray(pp["kpts"][i, j], float)
            out[r]["kp"][k], out[r]["iou"][k] = kp, max(ious)
            own, other = masks[r], masks["follow" if r == "lead" else "lead"]
            conf = np.flatnonzero(kp[:, 2] >= SCORE_MIN)
            if own is not None and other is not None and len(conf):
                H, W = own.shape
                u = np.clip(kp[conf, 0].round().astype(int), 0, W - 1)
                v = np.clip(kp[conf, 1].round().astype(int), 0, H - 1)
                out[r]["in_partner"][k] = float((other[v, u] & ~own[v, u]).mean())
    return out


def mono_init() -> dict:
    from .. import jl_mono
    from .. import mamma_dyn as md
    prep = json.loads((WORK / "mamma" / "prepare.json").read_text(encoding="utf-8"))
    tg = np.asarray(prep["grid"]["t"], float)
    tr = md.load_tracks(WORK, [CAM], offsets={})[CAM]
    j = md.nearest_index(tr.t, tg)
    cfg = tomllib.loads((PACKAGE_ROOT / "takes" / f"{TAKE}.toml").read_text(encoding="utf-8"))
    stat = [float(cfg["dancers"][r]["stature_m"]) for r in ROLES]
    kps = role_keypoints(j)
    T = len(tg)
    pos, conf, depth, src = np.zeros((T, 2, 3)), np.zeros((T, 2)), np.zeros((T, 2)), np.zeros((T, 2), int)
    for b, r in enumerate(ROLES):
        res = jl_mono.pelvis_positions(kps[r]["kp"], tr.K[j], tr.R[j], tr.tvec[j], tr.k1[j], stat[b], tr.ok[j],
                                       kps[r]["in_partner"])
        pos[:, b], conf[:, b], depth[:, b], src[:, b] = res["pos"], res["conf"], res["depth"], res["source"]
    if not np.isfinite(pos).all():
        raise RuntimeError(f"mono-init: no position for {int((~np.isfinite(pos).all(-1)).sum())} (frame, dancer) pairs")
    np.savez_compressed(WORK / "mamma" / "mono_init.npz", pos=pos.astype(np.float32), conf=conf, depth=depth, source=src, t=tg)
    soles = {r: float(cfg["dancers"][r].get("shoe_sole_m", 0.02)) for r in ROLES}
    (WORK / "mamma" / "stature.json").write_text(json.dumps(jl_mono.stature_targets(cfg, soles)))
    # diagnostic only (never an input): the start's pelvis against the 8-phone fit's pelvis
    from .holdout import load_fit, match_times
    diag = {}
    for b, r in enumerate(ROLES):
        f = load_fit(REF_WORK / "smplx", r)
        ir, it = match_times(f["t"], tg)
        e = pos[it, b] - f["joints"][ir, 0]
        C = -np.einsum("fji,fj->fi", tr.R[j[it]], tr.tvec[j[it]])
        u = f["joints"][ir, 0] - C
        u /= np.linalg.norm(u, axis=-1, keepdims=True)
        along = (e * u).sum(-1)
        diag[r] = {"pelvis_err_m_median": round(float(np.median(np.linalg.norm(e, axis=-1))), 3),
                   "along_ray_m_median": round(float(np.median(along)), 3),
                   "across_ray_m_median": round(float(np.median(np.linalg.norm(e - along[:, None] * u, axis=-1))), 3),
                   "height_err_m_median": round(float(np.median(e[:, 1])), 3)}
    rep = {"T": T, "grid": [float(tg[0]), float(tg[-1])],
           "measured": {r: float(conf[:, b].mean()) for b, r in enumerate(ROLES)},
           "from_feet": {r: float((src[:, b] == 1).mean()) for b, r in enumerate(ROLES)},
           "from_torso": {r: float((src[:, b] == 2).mean()) for b, r in enumerate(ROLES)},
           "box_iou_median": {r: float(np.median(kps[r]["iou"])) for r in ROLES},
           "skeleton_on_partner_frames": {r: int((kps[r]["in_partner"] > 0.5).sum()) for r in ROLES},
           "range_m_median": {r: float(np.median(depth[:, b])) for b, r in enumerate(ROLES)},
           "statures": json.loads((WORK / "mamma" / "stature.json").read_text()),
           "vs_reference_pelvis (diagnostic only)": diag}
    (WORK / "mamma" / "mono_init.json").write_text(json.dumps(rep, indent=1))
    log(f"mono-init: {json.dumps(rep)}")
    return rep


def pin() -> Path:
    """The 8-phone fit's betas as a MAMMA_INIT_BETAS file (body 0 = lead, 1 = follow): for the LEAK variant p06 only."""
    d = {str(b): [float(x) for x in np.load(REF_WORK / "smplx" / f"{r}.npz")["betas"]] for b, r in enumerate(ROLES)}
    p = WORK / "mamma" / "pin_betas_ref.json"
    p.write_text(json.dumps(d))
    (WORK / "mamma" / "pin_betas_ref.source.json").write_text(json.dumps(
        {"what": f"betas of {REF}/smplx (the 8-phone fit): LEAKS multi-view shape into a one-camera fit (variant p06 only)"}))
    return p


# ------------------------------------------------------------------------------------------------ pod job


def job() -> Path:
    w = f"dancecap/work/{TAKE}"
    pw = f"{POD_WS}/{w}"
    prep = json.loads((WORK / "mamma" / "prepare.json").read_text())
    seqs = [prep["sets"][k]["seq"] for k in ("m06", "p06")]
    fit = "python -m dancecap mamma fit " + TAKE + " --tag {tag}"
    pinned = (f"unset MAMMA_STATURE; export MAMMA_FREEZE_BETAS=1 MAMMA_INIT_BETAS={pw}/mamma/pin_betas_ref.json; "
              + fit.format(tag="p06"))
    d = {"schema": "dancecap.pod_job/1", "name": "cand_fitb", "take": TAKE, "max_minutes": 40, "needs": ["mamma"],
         "inputs": [f"{w}/mamma/prepare.json", f"{w}/mamma/ma3d_config.yaml", f"{w}/mamma/mono_init.npz",
                    f"{w}/mamma/stature.json", f"{w}/mamma/pin_betas_ref.json"]
                   + [f"{w}/mamma/ma_2d/{s}/" for s in seqs] + [f"{w}/mamma/ma_cap/{s}/" for s in seqs],
         "env": {"DANCECAP_POD": "1", "PYOPENGL_PLATFORM": "egl", "MAMMA_MONO": "1",
                 "MAMMA_MONO_INIT": f"{pw}/mamma/mono_init.npz", "MAMMA_STATURE": f"{pw}/mamma/stature.json"},
         "cwd": "dancecap",
         # the bundled PyPI wheels (210 MB) are left out: bootstrap.sh's pipq falls back to PyPI on the pod. Attempt 1
         # (session 20261019-134044-batch) lost its 343 MB upload to two scp timeouts (< 0.5 MB/s to that pod) and a
         # concurrent session's prune_bundles(keep=3) deleting the bundle between retries
         "exclude": ["dancecap/work/runpod/pywheels",
                     "mamma/data/body_models/smplx_locked_head/smplx/SMPLX_MALE.npz",
                     "mamma/data/body_models/smplx_locked_head/smplx/SMPLX_FEMALE.npz"],
         "setup": ["for g in MALE FEMALE; do ln -sf SMPLX_NEUTRAL.npz "
                   f"{POD_WS}/mamma/data/body_models/smplx_locked_head/smplx/SMPLX_$g.npz; done; "
                   "nvidia-smi --query-gpu=name,memory.total --format=csv"],
         "commands": [{"name": "fit_m06", "cmd": fit.format(tag="m06"), "timeout_min": 20},
                      {"name": "fit_p06", "cmd": pinned, "timeout_min": 15, "optional": True}],
         "outputs": [f"{w}/mamma/fits/", f"{w}/mamma/logs/"],
         "verify": [f"{w}/mamma/fits/m06/fit.json", f"{w}/mamma/fits/m06/smplx/lead.npz",
                    f"{w}/mamma/fits/m06/smplx/follow.npz"],
         "after_download": ["python -m dancecap.mono4d.cand_fitb import --tag m06 --name mamma_mono06 --session <sid>",
                            "python -m dancecap.mono4d.cand_fitb import --tag p06 --name mamma_mono06_pinbetas --session <sid>"],
         "notes": ["mono4d fit B: MAMMA_MONO=1 single-camera ma_3d of phone 06, 58-62 s (120 grid frames)",
                   "m06: free betas held to the owner's statures (MAMMA_STATURE: stature + 2.5 cm sole) - the honest one-camera fit",
                   "p06 (optional): betas pinned to the 8-phone fit's (LEAKS shape from the multi-view fit) - diagnostic only"]}
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    p = JOB_DIR / "pod_job.json"
    p.write_text(json.dumps(d, indent=1))
    log(f"pod job -> {p}")
    return p


# ------------------------------------------------------------------------------------------------ import


def _session_meta(sid: str | None) -> dict:
    if not sid:
        return {}
    p = PACKAGE_ROOT / "work" / "runpod" / "sessions" / sid / "result.json"
    if not p.exists():
        return {"session": sid}
    r = json.loads(p.read_text())
    pod = r.get("pod") or {}
    log = (p.parent / "runner.log").read_text(encoding="utf-8", errors="replace") if (p.parent / "runner.log").exists() else ""
    cloud = next((c for c in ("SECURE", "COMMUNITY") if f"created {pod.get('id')}" in log and f"({c})" in
                  log.split(f"created {pod.get('id')}", 1)[1][:200]), None)
    return {"session": sid, "gpu": pod.get("gpu"), "cloud": cloud, "cost_per_hr": pod.get("cost_per_hr"),
            "session_cost_usd": pod.get("ledger_cost"), "pod_minutes": round(float(pod.get("lifetime_s") or 0) / 60, 1),
            "commands_rc": {k: v.get("commands") for k, v in (r.get("jobs") or {}).items()}}


def import_fit(tag: str, name: str, sid: str | None = None, extra: dict | None = None) -> Path:
    from .candidates import cand_dir, save
    d = WORK / "mamma" / "fits" / tag
    fj = json.loads((d / "fit.json").read_text())
    m = {"method": "MAMMA ma_3d single-camera mode (MAMMA_MONO=1, mamma-turing utils/mono.py) on phone 06 alone",
         "input_cam": CAM, "source": str(d), "licence": "MAMMA (research code, mamma-turing local branch); SMPL-X locked-head (non-commercial)",
         "version": "mamma-turing 52df222 (turing-dynamic-cams + single-camera mode); dancecap mamma_dyn MA3D_MONO_OVERRIDES",
         "shape": ("betas PINNED to the 8-phone fit's (MAMMA_FREEZE_BETAS): LEAKS shape from the multi-view fit"
                   if tag == "p06" else "free betas held to the owner's statures (MAMMA_STATURE)"),
         "inputs": "the 8-phone fit's own phone-06 inputs: ma_2d 512 landmarks (SAM masks), camera track 06 (metric world from "
                   "the multi-phone SfM), pelvis start from ankle rays (jl_mono) on COCO-17 people/06.npz",
         "ma_3d": {k: fj["ma_3d"].get(k) for k in ("seconds", "gpu_used_peak_mib", "returncode", "torch_peak")},
         "smplx": {r: {k: v.get(k) for k in ("median_reproj_px", "quality_median", "betas0")} for r, v in fj["smplx"].items()},
         **_session_meta(sid), **(extra or {})}
    for r in ROLES:
        z = np.load(d / "smplx" / f"{r}.npz", allow_pickle=True)
        save(REF, name, r, np.asarray(z["t"], float), np.asarray(z["joints"], float)[:, :22],
             "world", m)
    log(f"imported {d} -> {cand_dir(REF, name)}")
    return cand_dir(REF, name)


# ------------------------------------------------------------------------------------------------ variant 3db: SAM 3D Body prior


def mask_conf(mask_px: np.ndarray, lo: float = 0.5, hi: float = 0.95, floor: float = 0.25) -> np.ndarray:
    """(F,) visible mask pixels of one dancer -> prior confidence: 1 from hi x its median area up, falling linearly to floor
    at lo x (larissa_kadu_38, lead: 3DB's per-frame PA-MPJPE is 31 mm at >= 0.9 x the median area, 51-63 mm below)."""
    rel = np.asarray(mask_px, float) / max(float(np.median(mask_px)), 1.0)
    return np.clip(floor + (1.0 - floor) * (rel - lo) / (hi - lo), floor, 1.0)


def grid_index(t_src: np.ndarray, t_grid: np.ndarray, tol: float = PRIOR_DT_MAX) -> tuple[np.ndarray, np.ndarray]:
    """For each grid instant the nearest source sample and whether it is within tol seconds."""
    from ..mamma_dyn import nearest_index
    t_src, t_grid = np.asarray(t_src, float), np.asarray(t_grid, float)
    k = nearest_index(t_src, t_grid)
    return k, np.abs(t_src[k] - t_grid) <= tol


def _gauss_weights(T: int, sigma: float, valid: np.ndarray) -> np.ndarray:
    i = np.arange(T)
    W = np.exp(-0.5 * ((i[:, None] - i[None, :]) / max(sigma, 1e-6)) ** 2) * np.asarray(valid, float)[None]
    return W / np.maximum(W.sum(1, keepdims=True), 1e-12)


def smooth_series(X: np.ndarray, sigma: float, valid: np.ndarray) -> np.ndarray:
    """Normalised Gaussian convolution over axis 0 that only reads the valid samples (sigma in samples; <= 0: a copy)."""
    X = np.asarray(X, float)
    if sigma <= 0:
        return X.copy()
    W = _gauss_weights(len(X), sigma, valid)
    return np.einsum("ij,j...->i...", W, np.nan_to_num(X))


def smooth_rotvec(aa: np.ndarray, sigma: float, valid: np.ndarray) -> np.ndarray:
    """(T, ..., 3) axis-angle -> the same, smoothed over time: per element the normalised Gaussian-weighted sum of the valid
    frames' quaternions (each flipped into the hemisphere of the frame's own), so a 360-degree wrap never averages to
    nonsense. Invalid frames come out as the weighted mean of their valid neighbours."""
    from scipy.spatial.transform import Rotation
    aa = np.asarray(aa, float)
    T, shp = aa.shape[0], aa.shape[1:-1]
    q = Rotation.from_rotvec(aa.reshape(-1, 3)).as_quat().reshape(T, -1, 4)        # (T, N, 4)
    if sigma <= 0:
        return aa.copy()
    W = _gauss_weights(T, sigma, valid)
    out = np.empty_like(q)
    for i in range(T):
        sgn = np.sign((q * q[i][None]).sum(-1, keepdims=True))
        sgn[sgn == 0] = 1.0
        s = np.einsum("j,jnk->nk", W[i], q * sgn)
        out[i] = s / np.maximum(np.linalg.norm(s, axis=-1, keepdims=True), 1e-12)
    return Rotation.from_quat(out.reshape(-1, 4)).as_rotvec().reshape(T, *shp, 3)


def orient_to_world(go_cam: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Global orientation (F, 3) axis-angle in the camera frame (x_cam = R X + t) -> world: R^T R(go)."""
    from scipy.spatial.transform import Rotation
    M = Rotation.from_rotvec(np.asarray(go_cam, float)).as_matrix()
    return Rotation.from_matrix(np.einsum("fji,fjk->fik", R, M)).as_rotvec()


def _fk_joints(betas, go, bp, transl) -> np.ndarray:
    """SMPL-X 2020 neutral (10 betas, flat hands) body joints (F, 22, 3) of SAM 3D Body's converted parameters."""
    import torch
    import smplx
    from ..take import WORKSPACE
    m = smplx.SMPLX(model_path=str(WORKSPACE / "smplx" / "SMPLX_NEUTRAL.npz"), gender="neutral", num_betas=10,
                    use_pca=False, flat_hand_mean=True, batch_size=len(go))
    f = lambda x: torch.as_tensor(np.asarray(x, np.float32))  # noqa: E731
    with torch.no_grad():
        return m(betas=f(betas), global_orient=f(go), body_pose=f(bp), transl=f(transl)).joints[:, :22].numpy().astype(float)


def prior_3db(src: Path = PRIOR_SRC, smooth: float = PRIOR_SMOOTH) -> dict:
    """SAM 3D Body (camera frame, per frame) -> mamma/prior_3db.npz on the fit grid, world frame (see the module doc)."""
    from .. import mamma_dyn as md
    from . import candidates as C
    from .holdout import load_fit
    z = np.load(src, allow_pickle=True)
    rows = json.loads(PRIOR_META.read_text())["frames"]
    fi = np.asarray(z["frame_i"], int)
    if [r["i"] for r in rows] != fi.tolist():
        raise SystemExit(f"{src}: frame indices differ from {PRIOR_META}")
    t_ref = C.frame_to_ref(REF, CAM, np.asarray(z["t"], float))   # 3DB rows carry phone 06's raw frames.json times
    tr = md.load_tracks(REF_WORK, [CAM])[CAM]                      # sync-corrected clock (sync/offsets.json)
    clock_err = float(np.abs(tr.t[fi] - t_ref).max())
    if clock_err > 1e-6:
        raise SystemExit(f"3DB frame times on the reference clock differ from the track's by {clock_err:.2e} s")
    prep = json.loads((WORK / "mamma" / "prepare.json").read_text())
    tg = np.asarray(prep["grid"]["t"], float)
    k, ok = grid_index(t_ref, tg)
    T = len(tg)
    R, tv = tr.R[fi], tr.tvec[fi]
    joints, conf = np.full((T, 2, 22, 3), np.nan), np.zeros((T, 2))
    go, bp = np.zeros((T, 2, 3)), np.zeros((T, 2, 63))
    checks = {}
    for b, role in enumerate(ROLES):
        Jc = np.asarray(z[f"{role}_joints"], float)[:, :22]
        Jw = C.to_world(Jc, R, tv)
        gcam = np.asarray(z[f"{role}_smplx_global_orient"], float)
        gw = orient_to_world(gcam, R)
        body = np.asarray(z[f"{role}_smplx_body_pose"], float)
        # the converter's parameters must reproduce its joints (camera frame), and the world orientation the world joints
        betas, transl = np.asarray(z[f"{role}_smplx_betas"], float), np.asarray(z[f"{role}_smplx_transl"], float)
        fk_cam = _fk_joints(betas, gcam, body, transl)
        fk_w = _fk_joints(betas, gw, body, np.zeros_like(transl))
        checks[role] = {"fk_vs_joints_camera_mm_max": round(1e3 * float(np.abs(fk_cam - Jc).max()), 3),
                        "fk_world_orient_vs_world_joints_rr_mm_max": round(1e3 * float(np.abs(
                            (fk_w - fk_w[:, :1]) - (Jw - Jw[:, :1])).max()), 3)}
        Jg = Jw[k]
        rr = smooth_series(Jg - Jg[:, :1], smooth, ok)
        joints[:, b] = np.where(ok[:, None, None], rr + Jg[:, :1], np.nan)
        go[:, b] = smooth_rotvec(gw[k], smooth, ok)
        bp[:, b] = smooth_rotvec(body[k].reshape(T, 21, 3), smooth, ok).reshape(T, 63)
        conf[:, b] = np.where(ok, mask_conf(np.array([r["mask_px"][role] for r in rows], float))[k], 0.0)
    # MAMMA's own camera of each grid frame (ma_cap, resampled to the grid instant) against the track's at the 3DB frame
    seq = prep["sets"]["m06"]["seq"]
    ce = np.load(WORK / "mamma" / "ma_cap" / seq / "gt" / f"cam{CAM}.npz")["cam_ext"]
    checks["ma_cap_vs_track"] = {"R_max_abs_diff": float(np.abs(ce[ok, :3, :3] - tr.R[fi[k[ok]]]).max()),
                                 "t_max_abs_diff_m": float(np.abs(ce[ok, :3, 3] - tr.tvec[fi[k[ok]]]).max())}
    checks["clock_max_err_s"] = clock_err
    checks["grid_dt_ms_max_used"] = round(1e3 * float(np.abs(t_ref[k[ok]] - tg[ok]).max()), 3)
    np.savez_compressed(PRIOR_NPZ, joints=joints.astype(np.float32), conf=conf.astype(np.float32),
                        global_orient=go.astype(np.float32), body_pose=bp.astype(np.float32), sigma_m=PRIOR_SIGMA_M,
                        rotate=False, scale=True, t=tg, frame_i=np.where(ok, fi[k], -1), smooth_frames=smooth,
                        source=str(src))
    # diagnostic only (never an input): the prior itself against the 8-phone fit
    diag = {}
    for b, role in enumerate(ROLES):
        ref = load_fit(REF_WORK / "smplx", role)
        s = C.score_arrays(tg[ok], joints[ok, b], "world", ref["t"], ref["joints"], WINDOW)
        diag[role] = {m: round(s[m]["median"], 1) for m in ("pa22_mm", "rr_mm", "pelvis_mm", "limb_deg", "accel_rr",
                                                            "accel_rr_ref")}
    rep = {"T": T, "frames_with_prior": int(ok.sum()), "missing_grid_frames": np.flatnonzero(~ok).tolist(),
           "source": str(src), "smooth_frames": smooth, "sigma_m": PRIOR_SIGMA_M,
           "conf_mean": {r: round(float(conf[ok, b].mean()), 3) for b, r in enumerate(ROLES)},
           "conf_below_0.5": {r: int((conf[ok, b] < 0.5).sum()) for b, r in enumerate(ROLES)},
           "checks": checks, "prior_vs_reference (diagnostic only)": diag}
    (WORK / "mamma" / "prior_3db.json").write_text(json.dumps(rep, indent=1))
    # the control: the prior on its own (3DB smoothed over time, nothing from MAMMA) as a candidate of its own
    for b, role in enumerate(ROLES):
        C.save(REF, "sam3d_body_smooth", role, tg[ok], joints[ok, b], "world",
               {"method": f"SAM 3D Body (candidate sam3d_body) in the world frame, root-relative joints smoothed over time "
                          f"(Gaussian sigma {smooth:g} frames): the prior of mamma_mono06_3db on its own (a control)",
                "input_cam": CAM, "source": str(PRIOR_NPZ), "cost_usd": 0.0,
                "time": "fit grid instants (reference seconds); each takes the 3DB frame within 5 ms"})
    log(f"prior-3db -> {PRIOR_NPZ}: {json.dumps(rep)}")
    return rep


def configs_3db() -> dict:
    """mamma/ma3d_config_<tag>.yaml (fit B's config + pose_prior_loss) and sets <tag> in prepare.json."""
    import yaml
    root = WORK / "mamma"
    prep = json.loads((root / "prepare.json").read_text())
    base_name = prep["config"]
    base = yaml.safe_load((root / base_name).read_text())
    todo = {**{t: w for t, (w, _) in PRIOR_VARIANTS.items()}, PRIOR_SMOKE: PRIOR_VARIANTS["3db"][0]}
    for tag, w in todo.items():
        cfg = copy.deepcopy(base)
        for run in PRIOR_RUNS:
            cfg["optim"][run]["losses"]["pose_prior_loss"] = {"weight": float(w)}
            if tag == PRIOR_SMOKE:
                cfg["optim"][run]["num_iters"] = 2
        fn = f"ma3d_config_{tag}.yaml"
        (root / fn).write_text(f"# dancecap mono4d.cand_fitb: {base_name} + pose_prior_loss weight {w:g} in "
                               f"{', '.join(PRIOR_RUNS)}{' (SMOKE: 2 iterations a stage)' if tag == PRIOR_SMOKE else ''} "
                               "(generated, do not edit)\n" + yaml.safe_dump(cfg, sort_keys=False), newline="\n")
        prep["sets"][tag] = {**{k: v for k, v in prep["sets"]["m06"].items() if k != "config"}, "config": fn,
                             "pose_prior": {"weight": float(w), "npz": PRIOR_NPZ.name,
                                            "candidate": PRIOR_VARIANTS.get(tag, (None, None))[1]}}
    (root / "prepare.json").write_text(json.dumps(prep, indent=1))
    log(f"configs-3db: sets {list(todo)} -> {root}")
    return {t: prep["sets"][t]["config"] for t in todo}


def job_3db(max_minutes: float = 55) -> Path:
    w = f"dancecap/work/{TAKE}"
    pw = f"{POD_WS}/{w}"
    prep = json.loads((WORK / "mamma" / "prepare.json").read_text())
    seq = prep["sets"]["m06"]["seq"]
    tags = list(PRIOR_VARIANTS)
    if any(prep["sets"].get(t, {}).get("seq") != seq for t in tags + [PRIOR_SMOKE]):
        raise SystemExit("run configs-3db first")
    fit = f"python -m dancecap mamma fit {TAKE} --tag"
    lg = f"{pw}/mamma/logs"
    # the three weights share the GPU (fit B peaked at 1.5 GiB on an 80 GB card). Measured in session
    # 20261019-193207-batch: no gain (stages 1-3 at 3x fit B's time per iteration, the SDF stage at 6-7x), and the
    # slowest weight missed the timeout - for a rerun, run the fits one after another or on separate pods
    par = (f"mkdir -p {lg}; for t in {' '.join(tags)}; do ({fit} $t > {lg}/fit_$t.out 2>&1; echo $? > {lg}/fit_$t.rc) & "
           f"sleep 20; done; wait; bad=0; for t in {' '.join(tags)}; do rc=$(cat {lg}/fit_$t.rc 2>/dev/null); "
           f"echo \"[3db] fit $t rc=$rc\"; tail -n 4 {lg}/fit_$t.out; [ \"$rc\" = 0 ] || bad=1; done; exit $bad")
    d = {"schema": "dancecap.pod_job/1", "name": "cand_fitb_3db", "take": TAKE, "max_minutes": max_minutes, "needs": ["mamma"],
         "inputs": [f"{w}/mamma/prepare.json", f"{w}/mamma/mono_init.npz", f"{w}/mamma/stature.json",
                    f"{w}/mamma/{PRIOR_NPZ.name}"]
                   + [f"{w}/mamma/{prep['sets'][t]['config']}" for t in tags + [PRIOR_SMOKE]]
                   + [f"{w}/mamma/ma_2d/{seq}/", f"{w}/mamma/ma_cap/{seq}/"],
         "env": {"DANCECAP_POD": "1", "PYOPENGL_PLATFORM": "egl", "MAMMA_MONO": "1",
                 "MAMMA_MONO_INIT": f"{pw}/mamma/mono_init.npz", "MAMMA_STATURE": f"{pw}/mamma/stature.json",
                 "MAMMA_POSE_PRIOR": f"{pw}/mamma/{PRIOR_NPZ.name}",
                 "DANCECAP_GPU_SHARED": "1"},  # side by side on the pod GPU (dancecap.gpu.exclusive would serialise them)
         "cwd": "dancecap",
         "exclude": ["dancecap/work/runpod/pywheels",
                     "mamma/data/body_models/smplx_locked_head/smplx/SMPLX_MALE.npz",
                     "mamma/data/body_models/smplx_locked_head/smplx/SMPLX_FEMALE.npz"],
         "setup": ["for g in MALE FEMALE; do ln -sf SMPLX_NEUTRAL.npz "
                   f"{POD_WS}/mamma/data/body_models/smplx_locked_head/smplx/SMPLX_$g.npz; done; "
                   "nvidia-smi --query-gpu=name,memory.total --format=csv"],
         "commands": [{"name": "fit_smoke", "cmd": f"{fit} {PRIOR_SMOKE}", "timeout_min": 10},
                      {"name": "fit_3db_parallel", "cmd": par, "timeout_min": 42}],
         "outputs": [f"{w}/mamma/fits/", f"{w}/mamma/logs/"],
         "verify": [f"{w}/mamma/fits/{t}/{f}" for t in tags for f in ("fit.json", "smplx/lead.npz", "smplx/follow.npz")],
         "after_download": ["python -m dancecap.mono4d.cand_fitb import-3db --session <sid>"],
         "notes": ["mono4d fit B + SAM 3D Body prior: MAMMA_MONO=1 single-camera ma_3d of phone 06, 58-62 s, with "
                   "pose_prior_loss (mamma-turing utils/mono.py, MAMMA_POSE_PRIOR) at weights "
                   + ", ".join(f"{t}={w:g}" for t, (w, _) in PRIOR_VARIANTS.items()) + "; '3db' is pre-registered",
                   "the prior: SAM 3D Body's own per-frame SMPL-X (candidate sam3d_body) moved to the world with phone "
                   "06's track; no multi-view data enters except the camera track and sync, exactly as for fit B",
                   f"{PRIOR_SMOKE}: 2 iterations a stage, a crash check before the three 15-min fits"]}
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    p = JOB_DIR / "pod_job_3db.json"
    p.write_text(json.dumps(d, indent=1), newline="\n")
    log(f"pod job -> {p}")
    return p


def _mamma_version() -> str:
    from .. import env
    root = env.mamma_root()
    try:
        h = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--", "optimization"], capture_output=True,
                               text=True).stdout.strip()
        return f"mamma-turing {h}{' + uncommitted changes' if dirty else ''} (turing-dynamic-cams)"
    except OSError:
        return "mamma-turing (version unknown)"


def import_3db(sid: str | None = None, score: bool = True) -> dict:
    from .candidates import score as cscore
    prior = json.loads((WORK / "mamma" / "prior_3db.json").read_text())
    out = {}
    for tag, (w, name) in PRIOR_VARIANTS.items():
        d = WORK / "mamma" / "fits" / tag
        if not (d / "fit.json").exists():
            log(f"{tag}: no fit ({d}) - {name} not imported")
            out[name] = {"missing": str(d)}
            continue
        extra = {"method": f"MAMMA ma_3d single-camera mode (MAMMA_MONO=1) on phone 06 alone + SAM 3D Body as a per-frame 3D "
                           f"pose prior (pose_prior_loss weight {w:g} in all four stages; prior also the starting pose)",
                 "version": _mamma_version() + "; dancecap mamma_dyn MA3D_MONO_OVERRIDES + cand_fitb configs-3db",
                 "variant": tag, "prior_weight": w, "pre_registered": tag == "3db",
                 "prior": {"source": prior["source"], "smooth_frames": prior["smooth_frames"], "sigma_m": prior["sigma_m"],
                           "loss": "Geman-McClure on root-relative world joints (21 below the pelvis), per-frame scale of "
                                   "the prior fitted without gradient, no rotation alignment; conf = visible mask area",
                           "frames_with_prior": prior["frames_with_prior"], "checks": prior["checks"]},
                 "inputs": "fit B's inputs (ma_2d 512 landmarks of phone 06, camera track 06, ankle-ray pelvis start, the "
                           "owner's statures) + SAM 3D Body's per-frame SMPL-X of phone 06 (our masks + K)",
                 "run_note": "session 20261019-193207-batch: the three weights ran side by side on one A100; its SDF "
                             "stage ran at ~65-70 s/iteration, so 3db_w8 (which could not finish inside the command "
                             "timeout) was killed by hand on the pod: fit_3db_parallel rc 1 is that kill; 3db and "
                             "3db_w05 finished with rc 0. Cost and minutes are the whole session (both weights)"}
        import_fit(tag, name, sid, extra)
        out[name] = cscore(REF, name, WINDOW)["roles"] if score else {}
    (WORK / "mamma" / "import_3db.json").write_text(json.dumps(out, indent=1, default=str))
    return out


def audit_3db(names: list[str] | None = None) -> dict:
    """Independent checks of the 3db candidates (and the prior, fit B, sam3d_body for comparison): left/right and role
    swaps must score far worse than the candidate as saved; PA-MPJPE over time shifts of -2..+2 frames must be lowest at 0;
    limb lengths against the reference; the clock check; a re-score with an own Procrustes."""
    from . import candidates as C
    from .holdout import load_fit, match_times
    names = names or ["mamma_mono06", "sam3d_body"] + [n for _, n in PRIOR_VARIANTS.values()]
    refs = {r: load_fit(REF_WORK / "smplx", r) for r in ROLES}
    swap = list(range(22))
    for a, b in LR_PAIRS:
        swap[a], swap[b] = b, a
    limbs = [(1, 4), (4, 7), (2, 5), (5, 8), (16, 18), (18, 20), (17, 19), (19, 21)]

    def pa(A, B):
        A, B = A - A.mean(0), B - B.mean(0)
        U, S, Vt = np.linalg.svd(A.T @ B)
        dd = np.sign(np.linalg.det(U @ Vt))
        Rm = U @ np.diag([1, 1, dd]) @ Vt
        s = (S * [1, 1, dd]).sum() / (A ** 2).sum()
        return 1e3 * np.linalg.norm(s * A @ Rm - B, axis=-1).mean()

    def world(name, role):
        d = C.cand_dir(REF, name)
        z = np.load(d / f"{role}.npz", allow_pickle=True)
        t, J = np.asarray(z["t"], float), np.asarray(z["joints"], float)[:, :22]
        if str(z["frame"]) == "camera":
            from .. import mamma_dyn as md
            tr = md.load_tracks(REF_WORK, [CAM])[CAM]
            kk = np.array([int(np.argmin(np.abs(tr.t - x))) for x in t])
            J = C.to_world(J, tr.R[kk], tr.tvec[kk])
        return t, J

    out = {}
    for name in names:
        if not (C.cand_dir(REF, name) / "lead.npz").exists():
            continue
        res = {}
        for role in ROLES:
            t, J = world(name, role)
            ref = refs[role]
            other = refs["follow" if role == "lead" else "lead"]
            win = (t >= WINDOW[0] - 0.02) & (t <= WINDOW[1] + 0.02)
            t, J = t[win], J[win]
            ir, it = match_times(ref["t"], t)
            R = ref["joints"][ir][:, :22]
            A = J[it]
            rr = lambda X, Y: float(np.median(1e3 * np.linalg.norm((X - X[:, :1]) - (Y - Y[:, :1]), axis=-1).mean(-1)))  # noqa: E731
            shifts = {}
            for sh in (-2, -1, 0, 1, 2):
                i2, t2 = match_times(ref["t"], t + sh / 30.0)
                shifts[sh] = round(float(np.median([pa(J[b], ref["joints"][a][:22]) for a, b in zip(i2, t2)])), 1)
            io, ito = match_times(other["t"], t)
            ratio = [float(np.median(np.linalg.norm(A[:, c] - A[:, p], axis=-1) / np.linalg.norm(R[:, c] - R[:, p], axis=-1)))
                     for p, c in limbs]
            res[role] = {"frames": int(len(it)), "clock": C.clock_check(REF, CAM, t, ref_grid=ref["t"]).get("clock"),
                         "pa22_own_mm": round(float(np.median([pa(a, b) for a, b in zip(A, R)])), 1),
                         "rr_mm": round(rr(A, R), 1), "rr_lr_swapped_mm": round(rr(A[:, swap], R), 1),
                         "rr_vs_partner_mm": round(rr(J[ito], other["joints"][io][:, :22]), 1),
                         "pa22_by_shift_frames": shifts, "best_shift_frames": min(shifts, key=shifts.get),
                         "limb_length_ratio_median": round(float(np.median(ratio)), 3),
                         "limb_length_ratio_range": [round(min(ratio), 3), round(max(ratio), 3)]}
        out[name] = res
    p = WORK / "mamma" / "audit_3db.json"
    p.write_text(json.dumps(out, indent=1))
    log(f"audit -> {p}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.mono4d.cand_fitb", description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("derive", "prepare-cmd", "check", "mono-init", "pin", "job", "import",
                                    "prior-3db", "configs-3db", "job-3db", "import-3db", "audit-3db"))
    ap.add_argument("--tag", default="m06")
    ap.add_argument("--name", default="mamma_mono06")
    ap.add_argument("--session")
    ap.add_argument("--smooth", type=float, default=PRIOR_SMOOTH, help="prior-3db: Gaussian sigma (frames)")
    a = ap.parse_args(argv)
    if a.cmd == "prior-3db":
        prior_3db(smooth=a.smooth)
    elif a.cmd == "configs-3db":
        print(json.dumps(configs_3db(), indent=1))
    elif a.cmd == "job-3db":
        print(job_3db())
    elif a.cmd == "import-3db":
        print(json.dumps(import_3db(a.session), indent=1, default=str))
    elif a.cmd == "audit-3db":
        print(json.dumps(audit_3db(), indent=1))
    elif a.cmd == "derive":
        derive()
    elif a.cmd == "prepare-cmd":
        print(prepare_cmd())
    elif a.cmd == "check":
        check()
    elif a.cmd == "mono-init":
        mono_init()
    elif a.cmd == "pin":
        print(pin())
    elif a.cmd == "job":
        print(job())
    else:
        import_fit(a.tag, a.name, a.session)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
