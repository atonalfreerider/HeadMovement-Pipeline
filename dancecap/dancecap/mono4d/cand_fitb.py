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
"""
from __future__ import annotations

import argparse
import json
import shutil
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
         "shape": ("free betas held to the owner's statures (MAMMA_STATURE)" if tag == "m06" else
                   "betas PINNED to the 8-phone fit's (MAMMA_FREEZE_BETAS): LEAKS shape from the multi-view fit"),
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.mono4d.cand_fitb", description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("derive", "prepare-cmd", "check", "mono-init", "pin", "job", "import"))
    ap.add_argument("--tag", default="m06")
    ap.add_argument("--name", default="mamma_mono06")
    ap.add_argument("--session")
    a = ap.parse_args(argv)
    if a.cmd == "derive":
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
