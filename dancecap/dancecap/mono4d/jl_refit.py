r"""john_lula realism, stage 2 (2026-10-20): the whole-take REFIT = this take's one-camera MAMMA recipe (J3b: NOTES_J3b.md,
farm/jl_profile.py) PLUS the SAM 3D Body per-frame pose prior (fit B + prior, docs/MONO4D_NOVEL_VIEWS.md 10.6: mamma-turing
58ff98a MAMMA_POSE_PRIOR, pose_prior_loss weight 0.5 = the recommended default, prior smoothed as cand_fitb 3db, per-frame
conf from the visible mask area, prior as the starting pose).

Owner, 2026-10-19: "Yes: use skinned LHM++ splats posed by the MAMMA + SAM 3D Body fit for the john_lula tutorial video. Run
the full john_lula process now (SAM 3D Body on all frames, the prior refit, LHM++ with my whole photo mask, renders)".

Same inputs, chunks and seams as the promoted fit (smplx/ = the second stitch john_lula_t2 of jlg0000 | jlx1158 | jlg1322):
  - chunks   jlg0000 k 0..1381, jlx1158 k 1158..1518 (the stitcher's boundary refit), jlg1322 k 1262..2642 (t = k / 30)
  - inputs   per chunk the J3b front's own grid inputs, copied from work/john_lula_<chunk>/mamma/: prepare.json (grid, set c1),
             ma_2d/<seq>/cam01.npz (512 landmarks AFTER qa_mask), mono_init.npz (ankle-ray pelvis start), stature.json;
             ma_cap/<seq>/gt/cam01.npz + global.npz (never pulled from the farm) are REBUILT from cameras/01.npz with
             mamma_dyn.resample_cameras (= build_inputs) and checked: the old fit's residuals recomputed with them equal its
             fit.json (build-check); ma3d_config.yaml = mamma_dyn.write_ma3d_config under MAMMA_MONO=1 (the J3b config)
  - shape    betas pinned and frozen from the shape pass (work/john_lula_jlshape/mamma_pin_betas.json, MAMMA_FREEZE_BETAS)
  - prior    realism/sam3db/prior_3db_full.npz (stage 1) cut to the chunk grid (jl_sam3db.prior_slice); config
             ma3d_config_3db.yaml = ma3d_config.yaml + pose_prior_loss weight 0.5 in all four stages
  - per pod  ONE fit (never two MAMMA fits on one GPU): mamma fit (set c1, its config = the 3db one) -> mamma select ->
             pose_refine smplx -> smplx_rf2_noslip (the J3b chunk recipe, RP.refine_args_recipe)
  - stitch   dancecap.stitch with the second stitch's arguments (--partial --force --select-weights arm=0.3,fingers=0.3)
             into realism/refit/stitch/, then jl_repair (depth spikes) on both folders; smplx_3db/{lead,follow}.npz = the
             repaired raw stitch (the contract of smplx/); the refined stitch stays in realism/refit/stitch/smplx_rf2_noslip/
New chunk work dirs: work/john_lula/realism/refit/chunks/john_lula_<chunk>/ (take name "john_lula/realism/refit/chunks/
john_lula_<chunk>": mamma_dyn.take_work and pose_refine join it under work/; the dir name keeps the seq john_lula_<chunk>_c1).
Never touched: work/john_lula/{smplx, smplx_rf2_noslip, avatar, export}/ and the J3b chunk takes work/john_lula_<chunk>/.

    python -m dancecap.mono4d.jl_refit build          # chunk dirs + rebuilt ma_cap + configs + prior slices + build-check
    python -m dancecap.mono4d.jl_refit job            # remote/jl_refit/pod_job_<chunk>.json (one per chunk = one pod each)
    (DANCECAP_RUNPOD_CAP=... python -m dancecap remote run --jobs remote/jl_refit/pod_job_<chunk>.json --max-cost .. --gpus ..)
    python -m dancecap.mono4d.jl_refit stitch         # realism/refit/stitch/ + jl_repair + smplx_3db/
    python -m dancecap.mono4d.jl_refit compare        # realism/refit/compare.json (current smplx/ vs smplx_3db/)
    python -m dancecap.mono4d.jl_refit sheets|clip    # overlays (12 frames) and a 10-s side-by-side clip
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from ..take import PACKAGE_ROOT, WORKSPACE

TAKE, CAM = "john_lula", "01"
ROLES = ("lead", "follow")
NAME = {"lead": "John", "follow": "Lula"}
WORK = PACKAGE_ROOT / "work" / TAKE
REAL = WORK / "realism"
REFIT = REAL / "refit"
CHUNK_ROOT = REFIT / "chunks"
STITCH_DIR = REFIT / "stitch"
OUT_3DB = WORK / "smplx_3db"
PRIOR_FULL = REAL / "sam3db" / "prior_3db_full.npz"
PIN = PACKAGE_ROOT / "work" / "john_lula_jlshape" / "mamma_pin_betas.json"
JOB_DIR = PACKAGE_ROOT / "remote" / "jl_refit"
POD_WS = "/workspace/HeadMovement"
CHUNKS = {"jlg0000": (0, 1381), "jlx1158": (1158, 1518), "jlg1322": (1262, 2642)}   # stitch_list.json of the second stitch
SET = "c1"
PRIOR_W = 0.5
PRIOR_RUNS = ("first_run", "second_run", "third_run", "fourth_run")
CFG_3DB = "ma3d_config_3db.yaml"
PROTECTED = [WORK / d for d in ("smplx", "smplx_rf2_noslip", "avatar", "export")]
STITCH_ARGS = ["--partial", "--force", "--select-weights", "arm=0.3,fingers=0.3"]


def log(msg: str) -> None:
    print(f"[jl_refit {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _ws(p: Path) -> str:
    return Path(p).resolve().relative_to(WORKSPACE.resolve()).as_posix()


def take_name(cid: str) -> str:
    """mamma_dyn / pose_refine take name of the new chunk dir (joined under work/)."""
    return (CHUNK_ROOT / f"{TAKE}_{cid}").resolve().relative_to((PACKAGE_ROOT / "work").resolve()).as_posix()


def chunk_dir(cid: str) -> Path:
    return CHUNK_ROOT / f"{TAKE}_{cid}"


def old_dir(cid: str) -> Path:
    return PACKAGE_ROOT / "work" / f"{TAKE}_{cid}"


def tree_digest(paths: list[Path]) -> dict:
    """sha1 of every file under the protected folders (to prove nothing there changed)."""
    out = {}
    for d in paths:
        for f in sorted(Path(d).rglob("*")):
            if f.is_file():
                out[str(f.relative_to(WORK))] = hashlib.sha1(f.read_bytes()).hexdigest()
    return out


# ================================================================================================ build
def _rebuild_ma_cap(cid: str, prep: dict, seq: str, dst_root: Path) -> dict:
    """ma_cap/<seq>/gt/{cam01,global}.npz from the take's camera track, exactly as mamma_dyn.build_inputs writes them
    (resample_cameras on the chunk grid; identity clock, offsets 0)."""
    from .. import mamma_dyn as md
    tr = md.load_tracks(WORK, [CAM], offsets={})[CAM]
    grid = np.asarray(prep["grid"]["t"], float)
    # the J3b chunk's camera slice (recap_front.step_cameras: frames within the chunk window +- the front margin); its
    # median frame step is what build_inputs passed as frame_dt
    t0, t1 = prep["grid"]["trim"]["window"]
    idx = np.flatnonzero((tr.t >= t0 - 1e-9) & (tr.t <= t1 + 1e-9))
    fdt = float(np.median(np.diff(tr.t[idx])))
    cg = md.resample_cameras(tr.t[idx], tr.K[idx], tr.R[idx], tr.tvec[idx], tr.ok[idx], grid, fdt)
    gt = dst_root / "ma_cap" / seq / "gt"
    if gt.exists():
        shutil.rmtree(gt)
    gt.mkdir(parents=True)
    nearest = md.nearest_index(tr.t[idx], grid)
    files = [tr.files[i] for i in idx[nearest]]
    np.savez(gt / f"cam{CAM}.npz", img_abs_path=np.array(files), img_rel_path=np.array(
        [Path(p).relative_to(WORK).as_posix() for p in files]), cam_name=np.array(f"cam{CAM}"),
             cam_int=cg["K"].astype(np.float32), cam_ext=md.extrinsics(cg["R"], cg["t"]).astype(np.float32),
             cam_img_w=int(tr.width), cam_img_h=int(tr.height), cam_portrait=bool(tr.height > tr.width), frame_start=0,
             frame_end=len(grid), video_path=np.array(""), distortion_model=np.array("radtan"), distortion_coeffs=np.zeros(4),
             is_body_in_img=np.ones(len(grid), bool), t=grid, native_index=nearest)
    np.savez(gt / "global.npz", params_paths=np.array([]), ioi_dir="", seq_name=seq, frames_len=int(len(grid)),
             fps=int(round(md.GRID_HZ)), frame_start=0, frame_end=int(len(grid)), people_len=len(md.PIDS), t=grid)
    # consistency with the copied landmarks: build_inputs zeroed every landmark of a grid frame whose camera is not ok
    lm = np.load(dst_root / "ma_2d" / seq / f"cam{CAM}.npz")
    nonzero_on_bad = int((np.abs(lm["landmarks"][~cg["ok"]]).sum((1, 2, 3)) > 0).sum()) if (~cg["ok"]).any() else 0
    inp = prep["sets"][SET]["inputs"][CAM]
    if len(idx) != inp["native_frames"] or abs(float(cg["ok"].mean()) - inp["grid_cam_ok"]) > 1e-9 or nonzero_on_bad:
        raise SystemExit(f"{cid}: rebuilt cameras differ from the J3b front (native {len(idx)} vs {inp['native_frames']}, "
                         f"cam ok {cg['ok'].mean()} vs {inp['grid_cam_ok']}, landmarks on not-ok frames {nonzero_on_bad})")
    return {"native_frames": int(len(idx)), "frame_dt": fdt, "grid_cam_ok": float(cg["ok"].mean()),
            "grid_cam_ok_in_prepare": prep["sets"][SET]["inputs"][CAM]["grid_cam_ok"],
            "native_frames_in_prepare": prep["sets"][SET]["inputs"][CAM]["native_frames"],
            "landmark_frames_nonzero_where_cam_not_ok": nonzero_on_bad}


def _configs(dst_root: Path) -> dict:
    import yaml
    from .. import mamma_dyn as md
    old = os.environ.get("MAMMA_MONO")
    os.environ["MAMMA_MONO"] = "1"
    try:
        base = md.write_ma3d_config(dst_root / "ma3d_config.yaml")
    finally:
        if old is None:
            os.environ.pop("MAMMA_MONO", None)
        else:
            os.environ["MAMMA_MONO"] = old
    cfg = yaml.safe_load(base.read_text())
    c3 = copy.deepcopy(cfg)
    for run in PRIOR_RUNS:
        c3["optim"][run]["losses"]["pose_prior_loss"] = {"weight": float(PRIOR_W)}
    (dst_root / CFG_3DB).write_text(f"# dancecap mono4d.jl_refit: ma3d_config.yaml (MAMMA_MONO=1, the J3b chunk config) + "
                                    f"pose_prior_loss weight {PRIOR_W:g} in {', '.join(PRIOR_RUNS)} (generated, do not edit)\n"
                                    + yaml.safe_dump(c3, sort_keys=False), newline="\n")
    return {"base": base.name, "prior": CFG_3DB,
            "losses_by_run": {r: {k: v.get("weight") for k, v in c3["optim"][r]["losses"].items()} for r in PRIOR_RUNS}}


def build_chunk(cid: str, check: bool = True) -> dict:
    from . import jl_sam3db
    k0, k1 = CHUNKS[cid]
    src, dst = old_dir(cid) / "mamma", chunk_dir(cid) / "mamma"
    prep = json.loads((src / "prepare.json").read_text())
    seq = prep["sets"][SET]["seq"]
    gk = prep["grid"]["trim"]["grid_k"]
    if [int(gk[0]), int(gk[1])] != [k0, k1] or prep["grid"]["T"] != k1 - k0 + 1:
        raise SystemExit(f"{cid}: prepare grid k {gk} differs from the stitch list's {k0}..{k1}")
    tg = np.asarray(prep["grid"]["t"], float)
    if np.abs(tg - np.arange(k0, k1 + 1) / 30.0).max() > 1e-9:
        raise SystemExit(f"{cid}: grid t is not k / 30")
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "ma_2d" / seq).mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / "ma_2d" / seq / f"cam{CAM}.npz", dst / "ma_2d" / seq / f"cam{CAM}.npz")
    for f in ("mono_init.npz", "mono_init.json", "stature.json", "qa_mask.json"):
        shutil.copy2(src / f, dst / f)
    rep = {"chunk": cid, "k": [k0, k1], "T": int(len(tg)), "seq": seq, "from": str(src),
           "ma_cap": _rebuild_ma_cap(cid, prep, seq, dst), "configs": _configs(dst)}
    p2 = copy.deepcopy(prep)
    p2["sets"][SET]["config"] = CFG_3DB
    p2["sets"][SET]["pose_prior"] = {"weight": PRIOR_W, "npz": "prior_3db.npz", "source": str(PRIOR_FULL)}
    p2["config"] = "ma3d_config.yaml"
    p2["refit"] = {"by": "dancecap.mono4d.jl_refit (realism stage 2)", "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   "from": str(src / "prepare.json"),
                   "note": "J3b chunk inputs (ma_2d after qa_mask, mono_init, grid) + rebuilt ma_cap; set c1 runs the 3db config"}
    (dst / "prepare.json").write_text(json.dumps(p2, indent=1))
    jl_sam3db.prior_slice(k0, k1, dst / "prior_3db.npz")
    pz = np.load(dst / "prior_3db.npz", allow_pickle=True)
    if np.abs(np.asarray(pz["t"], float) - tg).max() > 1e-9 or pz["joints"].shape[:2] != (len(tg), 2):
        raise SystemExit(f"{cid}: prior slice does not match the grid")
    mi = np.load(dst / "mono_init.npz")
    if mi["pos"].shape[:2] != (len(tg), 2):
        raise SystemExit(f"{cid}: mono_init has {mi['pos'].shape[0]} frames, the grid {len(tg)}")
    rep["prior"] = {"frames_with_prior": [int((pz["conf"][:, b] > 0).sum()) for b in range(2)],
                    "conf_mean": [round(float(pz["conf"][pz["conf"][:, b] > 0, b].mean()), 3) for b in range(2)]}
    if check:
        rep["check"] = build_check(cid)
    (chunk_dir(cid) / "build.json").write_text(json.dumps(rep, indent=1, default=str))
    log(f"{cid}: {json.dumps({k: rep[k] for k in ('T', 'ma_cap', 'prior')})}")
    return rep


def build_check(cid: str) -> dict:
    """The OLD chunk fit (work/john_lula_<chunk>/mamma/fits/c1) re-scored with the rebuilt ma_cap and the copied ma_2d:
    its residuals must equal its own fit.json, and its per-frame reprojection its npz reproj_px (frames not repaired)."""
    from .. import mamma_dyn as md
    old = old_dir(cid) / "mamma"
    fj = json.loads((old / "fits" / SET / "fit.json").read_text())
    prep = json.loads((chunk_dir(cid) / "mamma" / "prepare.json").read_text())
    seq = prep["sets"][SET]["seq"]
    gt, lm = chunk_dir(cid) / "mamma" / "ma_cap" / seq / "gt", chunk_dir(cid) / "mamma" / "ma_2d" / seq
    res = md.camera_residuals(old / "fits" / SET / "smplx", gt, lm, [CAM])
    out = {}
    D = md._load_verts512()
    for p, role in enumerate(ROLES):
        a, b = res[CAM][role], fj["residuals"][CAM][role]
        z = np.load(old / "fits" / SET / "smplx" / f"{role}.npz")
        pose = np.concatenate([z[k] for k in ("global_orient", "body_pose", "jaw_pose", "leye_pose", "reye_pose",
                                              "left_hand_pose", "right_hand_pose")], 1)
        verts, _ = md._forward(pose, z["transl"], z["betas"], True)
        rp = md.reprojection_parts(np.matmul(D, verts), [f"cam{CAM}"], gt, lm, p)["all"]
        keep = ~np.asarray(z["repaired"], bool) & np.isfinite(rp) & np.isfinite(z["reproj_px"])
        d = np.abs(rp[keep] - np.asarray(z["reproj_px"], float)[keep])
        out[role] = {"median_px_rebuilt": a["median_px"], "median_px_fit_json": b["median_px"],
                     "p90_px_rebuilt": a["p90_px"], "p90_px_fit_json": b["p90_px"], "frames": [a["frames"], b["frames"]],
                     "per_frame_vs_npz_reproj_px": {"frames": int(keep.sum()), "median_abs_px": float(np.median(d)),
                                                    "p99_abs_px": float(np.percentile(d, 99)), "max_abs_px": float(d.max())}}
    ok = all(abs(out[r]["median_px_rebuilt"] - out[r]["median_px_fit_json"]) < 0.05
             and abs(out[r]["p90_px_rebuilt"] - out[r]["p90_px_fit_json"]) < 0.1 and out[r]["frames"][0] == out[r]["frames"][1]
             for r in ROLES)
    out["ok"] = bool(ok)
    log(f"build-check {cid}: {json.dumps(out)}")
    if not ok:
        raise SystemExit(f"{cid}: the rebuilt ma_cap does not reproduce the old fit's residuals: {out}")
    return out


def build() -> dict:
    if not PRIOR_FULL.exists():
        raise SystemExit(f"{PRIOR_FULL} missing (stage 1)")
    REFIT.mkdir(parents=True, exist_ok=True)
    dig = REFIT / "protected_sha1_before.json"
    if not dig.exists():
        dig.write_text(json.dumps(tree_digest(PROTECTED), indent=0))
    return {cid: build_chunk(cid) for cid in CHUNKS}


# ================================================================================================ pod jobs
def job(cid: str, max_minutes: float | None = None) -> Path:
    from ..farm import recap_profile as RP
    k0, k1 = CHUNKS[cid]
    T = k1 - k0 + 1
    tk = take_name(cid)
    w = _ws(chunk_dir(cid))
    pw = f"{POD_WS}/{w}"
    prep = json.loads((chunk_dir(cid) / "mamma" / "prepare.json").read_text())
    seq = prep["sets"][SET]["seq"]
    # J3b: 32-33 min for ~1380 frames on a 4090 (jlx1158: 11 min for 361); timeout ~2x that; + bootstrap, select, refine
    fit_min = float(round(0.04 * T + 10))
    mm = max_minutes or fit_min + 25
    refine = RP.refine_args_recipe()
    d = {"schema": "dancecap.pod_job/1", "name": f"jl_refit_{cid}", "take": TAKE, "max_minutes": mm, "needs": ["mamma"],
         "inputs": [f"{w}/mamma/prepare.json", f"{w}/mamma/ma3d_config.yaml", f"{w}/mamma/{CFG_3DB}",
                    f"{w}/mamma/mono_init.npz", f"{w}/mamma/stature.json", f"{w}/mamma/prior_3db.npz",
                    f"{w}/mamma/ma_2d/{seq}/", f"{w}/mamma/ma_cap/{seq}/", _ws(PIN)],
         "env": {"DANCECAP_POD": "1", "PYOPENGL_PLATFORM": "egl", "MAMMA_MONO": "1",
                 "MAMMA_MONO_INIT": f"{pw}/mamma/mono_init.npz",
                 "MAMMA_FREEZE_BETAS": "1", "MAMMA_INIT_BETAS": f"{POD_WS}/{_ws(PIN)}",
                 "MAMMA_POSE_PRIOR": f"{pw}/mamma/prior_3db.npz"},
         "cwd": "dancecap",
         "exclude": ["dancecap/work/runpod/pywheels",
                     "mamma/data/body_models/smplx_locked_head/smplx/SMPLX_MALE.npz",
                     "mamma/data/body_models/smplx_locked_head/smplx/SMPLX_FEMALE.npz"],
         "setup": ["for g in MALE FEMALE; do ln -sf SMPLX_NEUTRAL.npz "
                   f"{POD_WS}/mamma/data/body_models/smplx_locked_head/smplx/SMPLX_$g.npz; done; "
                   "nvidia-smi --query-gpu=name,memory.total --format=csv"],
         "commands": [{"name": "fit", "cmd": f"python -m dancecap mamma fit {tk} --tag {SET}", "timeout_min": fit_min},
                      {"name": "select", "cmd": f"python -m dancecap mamma select {tk} --no-qa", "timeout_min": 5},
                      {"name": "refine", "cmd": f"python -m dancecap.pose_refine {tk} --src smplx --dst smplx_rf2_noslip {refine}",
                       "timeout_min": 12, "optional": True}],
         "outputs": [f"{w}/mamma/fits/", f"{w}/mamma/logs/", f"{w}/mamma/select.json", f"{w}/mamma/report.json",
                     f"{w}/smplx/", f"{w}/smplx_rf2_noslip/"],
         "verify": [f"{w}/mamma/fits/{SET}/fit.json", f"{w}/smplx/lead.npz", f"{w}/smplx/follow.npz"],
         "after_download": ["python -m dancecap.mono4d.jl_refit stitch (after every chunk is home)"],
         "notes": [f"john_lula realism stage 2: chunk {cid} (k {k0}..{k1}, {T} grid frames) of the one-camera MAMMA fit "
                   "(MAMMA_MONO=1, J3b's inputs, betas pinned from jlshape) + SAM 3D Body pose prior (MAMMA_POSE_PRIOR, "
                   f"pose_prior_loss weight {PRIOR_W:g} in all four stages, prior also the start); ONE fit on this pod",
                   "then mamma select (fits/c1 -> smplx/) and pose_refine smplx -> smplx_rf2_noslip (J3b chunk recipe)",
                   "likeness-derived inputs only (landmarks, cameras, joints); no frames leave the PC"]}
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    p = JOB_DIR / f"pod_job_{cid}.json"
    p.write_text(json.dumps(d, indent=1), newline="\n")
    log(f"pod job -> {p} (max {mm} min, fit timeout {fit_min} min)")
    return p


# ================================================================================================ stitch
def stitch_list() -> Path:
    """The second stitch's chunk list (work/john_lula/farm/stitch_list.json) with the refit chunk folders."""
    base = json.loads((WORK / "farm" / "stitch_list.json").read_text())
    d = copy.deepcopy(base)
    wr = (PACKAGE_ROOT / "work").resolve()
    if [c["name"] for c in d["chunks"]] != ["jlg0000", "jlx1158", "jlg1322"]:
        raise SystemExit(f"unexpected chunk list {[c['name'] for c in d['chunks']]}")
    for c in d["chunks"]:
        if list(c["k"]) != list(CHUNKS[c["name"]]):
            raise SystemExit(f"{c['name']}: k {c['k']} != {CHUNKS[c['name']]}")
        cd = chunk_dir(c["name"]).resolve().relative_to(wr).as_posix()
        c.update(src=f"{cd}/smplx", refined=f"{cd}/smplx_rf2_noslip", prepare=f"{cd}/mamma/prepare.json",
                 ma_2d=f"{cd}/mamma/ma_2d/{TAKE}_{c['name']}_c1")
    d["note"] = "jl_refit: work/john_lula/farm/stitch_list.json (the second stitch, john_lula_t2) with the 3db refit chunks"
    p = REFIT / "stitch_list.json"
    p.write_text(json.dumps(d, indent=1))
    return p


def do_stitch() -> dict:
    before = tree_digest(PROTECTED)
    for cid in CHUNKS:
        for sub in ("smplx", "smplx_rf2_noslip"):
            for r in ROLES:
                if not (chunk_dir(cid) / sub / f"{r}.npz").exists():
                    raise SystemExit(f"{cid}: {sub}/{r}.npz missing (pod job not home?)")
    p = stitch_list()
    out_rel = STITCH_DIR.resolve().relative_to((PACKAGE_ROOT / "work").resolve()).as_posix()
    cmd = [sys.executable, "-m", "dancecap.stitch", TAKE, "--chunks", str(p), "--out", out_rel, *STITCH_ARGS]
    log(" ".join(cmd))
    r = subprocess.run(cmd, cwd=PACKAGE_ROOT, capture_output=True, text=True)
    (REFIT / "stitch_stdout.log").write_text(r.stdout + "\n--- stderr\n" + r.stderr)
    print("\n".join(r.stdout.splitlines()[-14:]))
    if r.returncode not in (0, 2):
        raise SystemExit(f"stitch rc {r.returncode}: {r.stderr[-2000:]}")
    rr = subprocess.run([sys.executable, "-m", "dancecap.jl_repair", str(STITCH_DIR / "smplx"), str(STITCH_DIR / "smplx_rf2_noslip")],
                        cwd=PACKAGE_ROOT, capture_output=True, text=True)
    (REFIT / "repair_stdout.log").write_text(rr.stdout + "\n--- stderr\n" + rr.stderr)
    print(rr.stdout[-1500:])
    if rr.returncode != 0:
        raise SystemExit(f"jl_repair rc {rr.returncode}: {rr.stderr[-2000:]}")
    OUT_3DB.mkdir(parents=True, exist_ok=True)
    for f in ("lead.npz", "follow.npz", "depth_spike_repair.json"):
        if (STITCH_DIR / "smplx" / f).exists():
            shutil.copy2(STITCH_DIR / "smplx" / f, OUT_3DB / f)
    st = json.loads((STITCH_DIR / "stitch.json").read_text())
    prov = {"what": "john_lula one-camera MAMMA fit + SAM 3D Body pose prior (realism stage 2), raw stitch after jl_repair; "
                    "same contract as smplx/ (the promoted J3b fit); NOT promoted (the owner promotes)",
            "made_by": "python -m dancecap.mono4d.jl_refit stitch", "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "chunks": {c: str(chunk_dir(c)) for c in CHUNKS}, "stitch": str(STITCH_DIR / "stitch.json"),
            "refined_stitch": str(STITCH_DIR / "smplx_rf2_noslip"),
            "gate": st.get("gate"), "stitch_args": STITCH_ARGS,
            "prior": {"npz": str(PRIOR_FULL), "weight": PRIOR_W, "runs": PRIOR_RUNS},
            "sha1": {f: hashlib.sha1((OUT_3DB / f).read_bytes()).hexdigest() for f in ("lead.npz", "follow.npz")}}
    (OUT_3DB / "provenance.json").write_text(json.dumps(prov, indent=1, default=str))
    after = tree_digest(PROTECTED)
    if after != before:
        changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
        raise SystemExit(f"PROTECTED FILES CHANGED: {changed[:10]}")
    ref = json.loads((REFIT / "protected_sha1_before.json").read_text())
    prov["protected_unchanged_since_build"] = after == ref
    (OUT_3DB / "provenance.json").write_text(json.dumps(prov, indent=1, default=str))
    log(f"smplx_3db -> {OUT_3DB}; gate {st.get('gate', {}).get('passed')}; protected unchanged: {after == ref}")
    return prov


# ================================================================================================ compare
FITS = {"current": WORK / "smplx", "new": OUT_3DB, "current_rf2": WORK / "smplx_rf2_noslip",
        "new_rf2": STITCH_DIR / "smplx_rf2_noslip"}
POPS = {"lead": 10.87, "follow": 12.77}          # the LHM++ preview's two pose pops (~110 mm pelvis jumps) in smplx/


def load_fit(d: Path) -> dict:
    return {r: dict(np.load(Path(d) / f"{r}.npz", allow_pickle=True)) for r in ROLES}


def _pose165(z: dict) -> np.ndarray:
    return np.concatenate([np.asarray(z[k], float).reshape(len(z["t"]), -1) for k in
                           ("global_orient", "body_pose", "jaw_pose", "leye_pose", "reye_pose", "left_hand_pose",
                            "right_hand_pose")], 1)


def _stats(x, nd: int = 2) -> dict:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0}
    return {"n": int(len(x)), "median": round(float(np.median(x)), nd), "mean": round(float(x.mean()), nd),
            "p90": round(float(np.percentile(x, 90)), nd), "p99": round(float(np.percentile(x, 99)), nd),
            "max": round(float(x.max()), nd)}


def reproj512(F: dict, owner: np.ndarray) -> dict:
    """MAMMA's own 2D measure: per grid frame the median px error of the fit's 512 landmark vertices (FK of the saved
    parameters, MAMMA's locked-head model) against the ma_2d landmarks of the chunk that owns the frame (the refit chunk
    dirs hold identical copies of the J3b landmarks and the rebuilt, verified cameras)."""
    from .. import mamma_dyn as md
    D = md._load_verts512()
    T = len(F["lead"]["t"])
    out = {r: np.full(T, np.nan) for r in ROLES}
    for cid, (k0, k1) in CHUNKS.items():
        ks = np.flatnonzero(owner == cid)
        if not len(ks):
            continue
        seq = f"{TAKE}_{cid}_{SET}"
        g = np.load(chunk_dir(cid) / "mamma" / "ma_cap" / seq / "gt" / f"cam{CAM}.npz")
        lm = np.load(chunk_dir(cid) / "mamma" / "ma_2d" / seq / f"cam{CAM}.npz")
        E, K = np.asarray(g["cam_ext"], float), np.asarray(g["cam_int"], float)
        for p, r in enumerate(ROLES):
            z = F[r]
            pose, tr, b = _pose165(z), np.asarray(z["transl"], float), np.asarray(z["betas"], float)
            for s0 in range(0, len(ks), 400):
                kk = ks[s0:s0 + 400]
                v, _ = md._forward(pose[kk], tr[kk], b, True)
                pts = np.matmul(D, v)
                li = kk - k0
                px, zc = md.project(pts, K[li], E[li, :3, :3], E[li, :3, 3])
                obs, vis = lm["landmarks"][li, p, :, :2], lm["visibilities"][li, p]
                ok = (vis > 0.5) & (np.abs(obs).sum(-1) > 0) & (zc > 0)
                e = np.linalg.norm(px - obs, axis=-1)
                out[r][kk] = [np.median(e[i][ok[i]]) if ok[i].sum() >= 3 else np.nan for i in range(len(kk))]
    return out


def facing_world(J: np.ndarray) -> np.ndarray:
    from .jl_sam3db import facing
    return facing(J)


def _frames_of_grid(t: np.ndarray, tc: np.ndarray) -> np.ndarray:
    j = np.clip(np.searchsorted(tc, t), 1, len(tc) - 1)
    return np.where(np.abs(tc[j - 1] - t) < np.abs(tc[j] - t), j - 1, j)


def _interp_to(tg: np.ndarray, X: np.ndarray, tf: np.ndarray) -> np.ndarray:
    a = np.clip(np.searchsorted(tg, tf, side="right") - 1, 0, len(tg) - 2)
    w = np.clip((tf - tg[a]) / (tg[a + 1] - tg[a]), 0, 1).reshape((-1,) + (1,) * (X.ndim - 1))
    return (1 - w) * X[a] + w * X[a + 1]


def sole_metrics(z: dict, contact_m: float = 0.03) -> dict:
    """Floor contact and foot skating of one dancer: per foot the lowest sole vertex height and the horizontal motion of
    the sole centroid between consecutive frames in which that foot is down (lowest sole point < contact_m in both)."""
    from .. import physics_body as pb
    body = pb.load_body(z)
    pose = pb.full_pose(z)
    R, p = pb.forward_kinematics(body, pose, np.asarray(z["transl"], float).reshape(len(pose), 3))
    out, low_all = {}, []
    skate_all, cont_all = [], []
    for foot in ("left", "right"):
        idx = body.sole_vertices(foot, 0.02)
        S = np.concatenate([pb.lbs_points(body, R[s:s + 256], p[s:s + 256], idx) for s in range(0, len(R), 256)])
        h = S[..., 1].min(1)
        c = S.mean(1)[:, [0, 2]]
        down = (h[:-1] < contact_m) & (h[1:] < contact_m)
        sk = 1e3 * np.linalg.norm(np.diff(c, axis=0), axis=1)
        out[foot] = {"down_share": round(float(down.mean()), 3), "skate_mm_per_frame": _stats(sk[down], 1),
                     "skate_over_10mm_share_of_down": round(float((sk[down] > 10).mean()), 4) if down.any() else None,
                     "sole_height_m": _stats(h, 4)}
        low_all.append(h)
        skate_all.append(sk[down])
        cont_all.append(down)
    low = np.minimum(*low_all)
    sk = np.concatenate(skate_all)
    out["both_feet"] = {"skate_mm_per_frame": _stats(sk, 1), "skate_over_10mm_share": round(float((sk > 10).mean()), 4),
                        "frames_no_foot_down_share": round(float((~(cont_all[0] | cont_all[1])).mean()), 4),
                        "lowest_sole_m": {"p1": round(float(np.percentile(low, 1)), 4), "median": round(float(np.median(low)), 4),
                                          "p99": round(float(np.percentile(low, 99)), 4),
                                          "share_below_-1.5cm": round(float((low < -0.015).mean()), 4),
                                          "share_above_5cm": round(float((low > 0.05).mean()), 4)}}
    return out


def mask_agreement(keys=("current", "new", "current_rf2", "new_rf2"), dilate_px: int = 7) -> dict:
    """The projected 22 body joints of each fit against the SAM 2.1 masks of BOTH dancers (people/masks_01.npz) at the
    nearest video frame: share in the dancer's own (dilated) mask, on the partner's mask only (legitimate when hidden
    behind the partner), and outside both (background: wrong). Split by the dancer's visible share (teachers `visible`)."""
    import cv2
    from .. import jl_fitcheck as FC
    from .jl_sam3db import WHO, _decode
    fits = {k: load_fit(FITS[k]) for k in keys if (Path(FITS[k]) / "lead.npz").exists()}
    cam = np.load(WORK / "cameras" / f"{CAM}.npz", allow_pickle=True)
    tc = np.asarray(cam["t"], float)
    t = np.asarray(fits["current"]["lead"]["t"], float)
    jf = _frames_of_grid(t, tc)
    mz = np.load(WORK / "people" / f"masks_{CAM}.npz")
    vis = np.asarray(np.load(WORK / "people" / f"teachers_{CAM}.npz", allow_pickle=True)["visible"], float)
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
    T = len(t)
    S = {k: {r: np.full((T, 3), np.nan) for r in ROLES} for k in fits}
    for i in range(T):
        f = int(jf[i])
        if abs(tc[f] - t[i]) > 0.04 or not cam["ok"][f]:
            continue
        ms = {r: _decode(mz, f"m_{f}_{WHO[r]}") for r in ROLES}
        if any(m is None for m in ms.values()):
            continue
        dil = {r: cv2.dilate(ms[r].astype(np.uint8), kern) > 0 for r in ROLES}
        H, W = ms["lead"].shape
        for k, F in fits.items():
            for r in ROLES:
                o = "follow" if r == "lead" else "lead"
                uv, z = FC.project(np.asarray(F[r]["joints"], float)[i, :22], np.asarray(cam["R"][f], float),
                                   np.asarray(cam["tvec"][f], float), np.asarray(cam["K"][f], float),
                                   float(np.asarray(cam["dist"][f]).reshape(-1)[0]))
                ins = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
                if ins.sum() < 6:
                    continue
                u, v = uv[ins, 0].astype(int), uv[ins, 1].astype(int)
                own = dil[r][v, u]
                par = dil[o][v, u] & ~own
                S[k][r][i] = [own.mean(), par.mean(), (~own & ~par).mean()]
    out = {}
    bins = ((0.0, 0.3), (0.3, 0.6), (0.6, 9.0))
    for r in ROLES:
        vv = vis[jf, WHO[r]]
        out[r] = {}
        for k in fits:
            X = S[k][r]
            d = {"all": {"frames": int(np.isfinite(X[:, 0]).sum()), "own": round(float(np.nanmean(X[:, 0])), 4),
                         "partner_only": round(float(np.nanmean(X[:, 1])), 4), "outside_both": round(float(np.nanmean(X[:, 2])), 4)}}
            for lo, hi in bins:
                m = (vv >= lo) & (vv < hi) & np.isfinite(X[:, 0])
                d[f"visible_{lo}-{hi}"] = {"frames": int(m.sum()), "own": round(float(X[m, 0].mean()), 4) if m.any() else None,
                                           "partner_only": round(float(X[m, 1].mean()), 4) if m.any() else None,
                                           "outside_both": round(float(X[m, 2].mean()), 4) if m.any() else None}
            out[r][k] = d
    np.savez_compressed(REFIT / "mask_agreement_frames.npz", t=t, **{f"{r}_{k}": S[k][r] for k in fits for r in ROLES})
    (REFIT / "mask_agreement.json").write_text(json.dumps(out, indent=1))
    log(f"mask agreement -> {REFIT / 'mask_agreement.json'}")
    return out


def compare(do_sanity: bool = True) -> dict:
    from . import candidates as C
    from .. import jl_fitcheck as FC
    from .. import mamma_dyn as md
    from .jl_sam3db import _pa
    REFIT.mkdir(parents=True, exist_ok=True)
    fits = {k: load_fit(v) for k, v in FITS.items() if (Path(v) / "lead.npz").exists()}
    cur, new = fits["current"], fits["new"]
    t = np.asarray(cur["lead"]["t"], float)
    T = len(t)
    for k, F in fits.items():
        if len(F["lead"]["t"]) != T or np.abs(np.asarray(F["lead"]["t"], float) - t).max() > 1e-9:
            raise SystemExit(f"{k}: grid differs from smplx/")
    owner = np.asarray(cur["lead"]["segment_owner"]).astype(str)
    cam = np.load(WORK / "cameras" / f"{CAM}.npz", allow_pickle=True)
    tc = np.asarray(cam["t"], float)
    jf = _frames_of_grid(t, tc)
    tt = np.load(WORK / "people" / f"teachers_{CAM}.npz", allow_pickle=True)
    rep = {"grid": {"T": T, "t": [float(t[0]), float(t[-1])]}, "fits": {k: str(v) for k, v in FITS.items()},
           "note": "no ground truth on this take: every number is a consistency / plausibility measure", "roles": {}}
    # ---- 2D: MAMMA's 512 landmarks (px) and the RTMW teacher (percent of the person's height)
    r512 = {k: reproj512(fits[k], owner) for k in ("current", "new")}
    wr = (PACKAGE_ROOT / "work").resolve()
    ts = {k: (Path(FITS[k]).resolve().parent.relative_to(wr).as_posix(), Path(FITS[k]).name) for k in fits}
    teach = {k: FC.reproj(*ts[k]) for k in fits}
    sanity = {}
    if do_sanity:
        for k in fits:
            s = FC.sanity(*ts[k])
            sanity[k] = {kk: v for kk, v in s.items() if kk not in ("_arrays", "pair_distance_by_frame_m_every10")}
    rep["sanity_jl_fitcheck"] = sanity
    s3 = np.load(REAL / "sam3db" / "smplx_3db.npz", allow_pickle=True)
    t3 = np.asarray(s3["t"], float)
    per_frame = {}
    for b, r in enumerate(ROLES):
        o = 1 - b
        R = {}
        absent = np.asarray(cur[r].get("absent", np.zeros(T, bool)), bool)
        R["absent_frames_current"] = int(absent.sum())
        R["reproj512_px"] = {k: _stats(v[r][~absent]) for k, v in r512.items()}
        R["reproj512_px_frames_new_worse_by_2px"] = int((r512["new"][r] - r512["current"][r] > 2).sum())
        R["reproj512_px_frames_new_better_by_2px"] = int((r512["current"][r] - r512["new"][r] > 2).sum())
        R["teacher_reproj_pct_height"] = {k: {kk: v["summary"][r][kk] for kk in ("median", "p90", "frames", "median_wide",
                                                                                   "median_tight", "bad_seconds")}
                                          for k, v in teach.items()}
        # jitter
        R["jitter_accel_rr_mm_per_frame2"] = {k: _stats(C.accel_rr_mm(t, np.asarray(F[r]["joints"], float)[:, :22]))
                                              for k, F in fits.items()}
        # pelvis pops: per-frame step and second difference of the pelvis
        pops = {}
        for k, F in fits.items():
            P = np.asarray(F[r]["joints"], float)[:, 0]
            step = 1e3 * np.linalg.norm(np.diff(P, axis=0), axis=1)
            acc = np.full(T, np.nan)
            acc[1:-1] = 1e3 * np.linalg.norm(P[2:] - 2 * P[1:-1] + P[:-2], axis=1)
            w = np.abs(t - POPS[r]) <= 0.3
            ws = np.abs(t[:-1] - POPS[r]) <= 0.3
            top = np.argsort(-np.nan_to_num(acc))[:8]
            pops[k] = {"at_preview_pop": {"t": POPS[r], "max_pelvis_step_mm": round(float(step[ws].max()), 1),
                                          "max_pelvis_accel_mm_f2": round(float(np.nanmax(acc[w])), 1),
                                          "t_of_max_step": round(float(t[:-1][ws][np.argmax(step[ws])]), 3)},
                       "pelvis_step_mm": _stats(step, 1), "pelvis_accel_mm_f2": _stats(acc, 1),
                       "frames_pelvis_accel_over_40": int((np.nan_to_num(acc) > 40).sum()),
                       "frames_pelvis_step_over_80mm": int((step > 80).sum()),
                       "top_accel": [[round(float(t[i]), 3), round(float(acc[i]), 1)] for i in top]}
        R["pelvis_pops"] = pops
        # floor contact and foot skating (sole vertices; the raw fits float / sink by cm, the refinement puts them down)
        R["feet"] = {k: sole_metrics(F[r]) for k, F in fits.items()}
        # the change between the two fits
        Jc, Jn = np.asarray(cur[r]["joints"], float)[:, :22], np.asarray(new[r]["joints"], float)[:, :22]
        d_rr = 1e3 * np.linalg.norm((Jn - Jn[:, :1]) - (Jc - Jc[:, :1]), axis=-1)[:, 1:].mean(1)
        d_pel = 1e3 * np.linalg.norm(Jn[:, 0] - Jc[:, 0], axis=1)
        fc, fn = facing_world(Jc), facing_world(Jn)
        d_face = np.degrees(np.arccos(np.clip((fc * fn).sum(1), -1, 1)))
        R["new_vs_current"] = {"root_relative_mm": _stats(d_rr, 1), "pelvis_mm": _stats(d_pel, 1),
                               "facing_deg": _stats(d_face, 1), "facing_over_90deg_frames": int((d_face > 90).sum()),
                               "facing_over_90deg_t": [round(float(x), 2) for x in t[d_face > 90][:60]]}
        per_frame[f"{r}_d_rr"], per_frame[f"{r}_d_pelvis"], per_frame[f"{r}_d_face"] = d_rr, d_pel, d_face
        # agreement with SAM 3D Body (NOT independent for the new fit: it is the new fit's prior)
        ok3 = np.asarray(s3[f"{r}_qa_ok"], bool)
        J3 = np.asarray(s3[f"{r}_joints_world"], float)[:, :22]
        agr = {}
        for k in ("current", "new"):
            Jf = _interp_to(t, np.asarray(fits[k][r]["joints"], float)[:, :22], t3)
            ab = _interp_to(t, np.asarray(fits[k][r].get("absent", np.zeros(T)), float), t3) > 0
            use = ok3 & ~ab & np.isfinite(J3).all((1, 2))
            rr = 1e3 * np.linalg.norm((Jf[use] - Jf[use, :1]) - (J3[use] - J3[use, :1]), axis=-1)[:, 1:].mean(1)
            pa = [_pa(J3[i], Jf[i])[0] for i in np.flatnonzero(use)]
            agr[k] = {"frames": int(use.sum()), "rr_mm": _stats(rr, 1), "pa22_mm": _stats(pa, 1)}
        R["vs_sam3db (not independent for the new fit)"] = agr
        # facing in the picture: the order of the left / right shoulder + hip in x, fit vs the RTMW teacher
        lr = {}
        kp = np.asarray(tt["kp"], float)[:, b]                       # (F, 17, 3) COCO-17 of this role
        box_h = np.maximum(np.asarray(tt["box"], float)[:, b, 3] - np.asarray(tt["box"], float)[:, b, 1], 50.0)
        sip = np.asarray(tt["skel_in_partner"], float)[:, b]
        conf = np.asarray(tt["conf"], float)[:, b]
        for k in ("current", "new"):
            J = np.asarray(fits[k][r]["joints"], float)
            agree = np.full(T, np.nan)
            for i in range(T):
                f = int(jf[i])
                if abs(tc[f] - t[i]) > 0.04 or conf[f] < 0.5 or sip[f] > 0.3 or not cam["ok"][f]:
                    continue
                pairs = [(5, 6, 16, 17), (11, 12, 1, 2)]
                dT, dF = [], []
                uv, _ = FC.project(J[i, [16, 17, 1, 2]], np.asarray(cam["R"][f], float), np.asarray(cam["tvec"][f], float),
                                   np.asarray(cam["K"][f], float), float(np.asarray(cam["dist"][f]).reshape(-1)[0]))
                q = {16: uv[0], 17: uv[1], 1: uv[2], 2: uv[3]}
                for a_, b_, sa, sb in pairs:
                    if kp[f, a_, 2] >= 0.5 and kp[f, b_, 2] >= 0.5:
                        dT.append((kp[f, a_, 0] - kp[f, b_, 0]) / box_h[f])
                        dF.append((q[sa][0] - q[sb][0]) / box_h[f])
                if not dT:
                    continue
                dT, dF = float(np.mean(dT)), float(np.mean(dF))
                if abs(dT) < 0.06 or abs(dF) < 0.03:              # side-on in the teacher or in the fit: no evidence
                    continue
                agree[i] = float(np.sign(dT) == np.sign(dF))
            lr[k] = {"frames": int(np.isfinite(agree).sum()), "agree_share": round(float(np.nanmean(agree)), 4),
                     "disagree_frames": int((agree == 0).sum()),
                     "disagree_t": [round(float(x), 2) for x in t[agree == 0][:80]]}
            per_frame[f"{r}_lr_agree_{k}"] = agree
        R["facing_lr_vs_teacher"] = lr
        rep["roles"][r] = R
    # ---- the couple: who is nearer the camera
    order = {}
    sipL, sipF = np.asarray(tt["skel_in_partner"], float)[:, 0], np.asarray(tt["skel_in_partner"], float)[:, 1]
    ev = np.zeros(T)                                            # +1 lead behind follow, -1 follow behind lead, 0 none
    for i in range(T):
        f = int(jf[i])
        if sipL[f] >= 0.3 and sipF[f] <= 0.1:
            ev[i] = 1
        elif sipF[f] >= 0.3 and sipL[f] <= 0.1:
            ev[i] = -1
    zz = {}
    for k in ("current", "new", "current_rf2", "new_rf2"):
        if k not in fits:
            continue
        z = {}
        for r in ROLES:
            P = np.asarray(fits[k][r]["joints"], float)[:, 0]
            z[r] = np.einsum("tij,tj->ti", np.asarray(cam["R"], float)[jf], P)[:, 2] + np.asarray(cam["tvec"], float)[jf][:, 2]
        zz[k] = z
        dz = z["lead"] - z["follow"]                            # > 0: lead farther = lead behind
        m = ev != 0
        order[k] = {"evidence_frames": int(m.sum()), "agree_with_occlusion_evidence": round(float((np.sign(dz[m]) == ev[m]).mean()), 4),
                    "depth_gap_m": _stats(np.abs(dz), 3), "lead_nearer_share": round(float((dz < 0).mean()), 4)}
    # SAM 3D Body's own order (camera-frame pelvis depth, per frame where both are valid)
    v3 = np.asarray(s3["lead_valid"], bool) & np.asarray(s3["follow_valid"], bool)
    dz3 = np.asarray(s3["lead_joints"], float)[:, 0, 2] - np.asarray(s3["follow_joints"], float)[:, 0, 2]
    ev3 = np.zeros(len(t3))
    ev3[(np.asarray(tt["skel_in_partner"], float)[:, 0] >= 0.3) & (np.asarray(tt["skel_in_partner"], float)[:, 1] <= 0.1)] = 1
    ev3[(np.asarray(tt["skel_in_partner"], float)[:, 1] >= 0.3) & (np.asarray(tt["skel_in_partner"], float)[:, 0] <= 0.1)] = -1
    m3 = v3 & (ev3 != 0)
    order["sam3db_per_frame"] = {"evidence_frames": int(m3.sum()), "agree_with_occlusion_evidence": round(float((np.sign(dz3[m3]) == ev3[m3]).mean()), 4)}
    dzc, dzn = zz["current"]["lead"] - zz["current"]["follow"], zz["new"]["lead"] - zz["new"]["follow"]
    big = (np.abs(dzc) > 0.15) | (np.abs(dzn) > 0.15)
    order["new_vs_current_same_order_share"] = round(float((np.sign(dzc) == np.sign(dzn)).mean()), 4)
    order["new_vs_current_same_order_share_gap_over_15cm"] = round(float((np.sign(dzc[big]) == np.sign(dzn[big])).mean()), 4)
    order["new_vs_current_order_flips_t"] = [round(float(x), 2) for x in t[big & (np.sign(dzc) != np.sign(dzn))][:60]]
    order["evidence_rule"] = ("RTMW teacher skeleton of A lies on B's SAM mask (skel_in_partner >= 0.3) while B's does not "
                              "(<= 0.1): A is behind B")
    rep["couple_depth_order"] = order
    # seams of both stitches
    def seams(p):
        if not Path(p).exists():
            return None
        s = json.loads(Path(p).read_text())
        return {"gate": s.get("gate"), "seams": [{"name": x.get("name"), "passed": x.get("gate", {}).get("passed"),
                                                   "window_t": x.get("window_t") or x.get("blend", {}).get("t"),
                                                   "roles": {r: x["roles"][r]["checks"] for r in x.get("roles", {})}}
                                                  for x in s.get("seams", [])]}
    rep["stitch"] = {"current": seams(WORK / "stitch.json"), "new": seams(STITCH_DIR / "stitch.json")}
    np.savez_compressed(REFIT / "compare_frames.npz", t=t, **per_frame,
                        **{f"{r}_reproj512_{k}": v[r] for k, v in r512.items() for r in ROLES},
                        **{f"{r}_z_{k}": zz[k][r] for k in zz for r in ROLES}, occlusion_evidence=ev)
    (REFIT / "compare.json").write_text(json.dumps(rep, indent=1, default=str))
    log(f"compare -> {REFIT / 'compare.json'}")
    return rep


# ================================================================================================ overlays + clip
BONES = [(0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8), (6, 9), (7, 10), (8, 11), (9, 12), (12, 15),
         (9, 13), (9, 14), (13, 16), (14, 17), (16, 18), (17, 19), (18, 20), (19, 21)]
LEFT = {1, 4, 7, 10, 13, 16, 18, 20}                            # SMPL-X left-side joints (a bone ending there is "left")
COL = {"lead": (0, 140, 255), "follow": (255, 220, 0)}           # BGR: John orange, Lula cyan
SHEETS = REFIT / "sheets"
VIDEOS = REFIT / "videos"


class _Overlay:
    def __init__(self, keys=("current", "new")):
        self.cam = np.load(WORK / "cameras" / f"{CAM}.npz", allow_pickle=True)
        self.tc = np.asarray(self.cam["t"], float)
        self.F = {k: load_fit(FITS[k]) for k in keys}
        self.t = np.asarray(self.F[keys[0]]["lead"]["t"], float)
        self.tt = np.load(WORK / "people" / f"teachers_{CAM}.npz", allow_pickle=True)
        fj = json.loads((WORK / "frames" / CAM / "frames.json").read_text())
        self.files = [WORK / "frames" / CAM / f["file"] for f in fj["frames"]]

    def uv(self, key: str, role: str, f: int) -> np.ndarray:
        from .. import jl_fitcheck as FC
        J = _interp_to(self.t, np.asarray(self.F[key][role]["joints"], float)[:, :25], np.array([self.tc[f]]))[0]
        c = self.cam
        uv, z = FC.project(J, np.asarray(c["R"][f], float), np.asarray(c["tvec"][f], float), np.asarray(c["K"][f], float),
                           float(np.asarray(c["dist"][f]).reshape(-1)[0]))
        return np.where((z > 0)[:, None], uv, np.nan)

    def absent(self, key: str, role: str, f: int) -> bool:
        z = self.F[key][role]
        if "absent" not in z:
            return False
        i = int(np.argmin(np.abs(self.t - self.tc[f])))
        return bool(np.asarray(z["absent"])[i])

    def draw(self, img, key: str, f: int, S: float = 1.0, x0: float = 0.0, y0: float = 0.0, lw: int = 2) -> None:
        import cv2
        for role in ROLES:
            uv = (self.uv(key, role, f) - [x0, y0]) * S
            col = COL[role]
            light = tuple(int(0.45 * c + 0.55 * 255) for c in col)
            dash = self.absent(key, role, f)
            for a, b in BONES:
                if np.isfinite(uv[[a, b]]).all():
                    cv2.line(img, tuple(int(v) for v in uv[a]), tuple(int(v) for v in uv[b]),
                             light if b in LEFT else col, lw if not dash else 1, cv2.LINE_AA)
            for q in range(22):
                if np.isfinite(uv[q]).all():
                    cv2.circle(img, tuple(int(v) for v in uv[q]), max(2, lw), (255, 255, 255), -1, cv2.LINE_AA)
            for q in (23, 24):                                   # the eyes: which way the face points
                if np.isfinite(uv[q]).all():
                    cv2.circle(img, tuple(int(v) for v in uv[q]), max(2, lw), (255, 0, 255), -1, cv2.LINE_AA)

    def crop_box(self, f: int, aspect: float = 0.75) -> tuple[float, float, float, float]:
        bx = np.asarray(self.tt["box"], float)[f]
        pts = [bx[b] for b in range(2) if np.isfinite(bx[b]).all() and bx[b, 3] > bx[b, 1]]
        for key in self.F:
            for role in ROLES:
                u = self.uv(key, role, f)[:22]
                u = u[np.isfinite(u).all(1)]
                if len(u):
                    pts.append(np.r_[u.min(0), u.max(0)])
        P = np.array(pts)
        x0, y0, x1, y1 = P[:, 0].min(), P[:, 1].min(), P[:, 2].max(), P[:, 3].max()
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        h = max(y1 - y0, (x1 - x0) / aspect) * 1.12
        w = h * aspect
        return cx - w / 2, cy - h / 2, w, h

    def tile(self, f: int, key: str, size=(360, 480), label: str = "", box=None) -> np.ndarray:
        import cv2
        img = cv2.imread(str(self.files[f]))
        x0, y0, w, h = box or self.crop_box(f, size[0] / size[1])
        S = size[1] / h
        M = np.array([[S, 0, -x0 * S], [0, S, -y0 * S]], float)
        out = cv2.warpAffine(img, M, size, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(40, 40, 40))
        self.draw(out, key, f, S, x0, y0, lw=2)
        for k, line in enumerate(label.split("\n")):
            cv2.putText(out, line, (6, 18 + 18 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, line, (6, 18 + 18 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return out


def _frame_at(ov: _Overlay, tsec: float) -> int:
    return int(np.argmin(np.abs(ov.tc - tsec)))


def sheets(extra: list[float] | None = None) -> list[Path]:
    """sheets/overlay_{1,2,3}.jpg: 12 instants spread over the dance, each as a pair (current smplx/ | new smplx_3db/);
    sheets/focus_*.jpg: the preview's two pops, the largest changes between the fits and the facing disagreements."""
    import cv2
    ov = _Overlay()
    SHEETS.mkdir(parents=True, exist_ok=True)
    cmpf = np.load(REFIT / "compare_frames.npz") if (REFIT / "compare_frames.npz").exists() else None

    def pair(f, note=""):
        box = ov.crop_box(f, 360 / 480)
        lab = f"f{f} t={ov.tc[f]:.2f}s{note}"
        a = ov.tile(f, "current", label=lab + "\ncurrent smplx/", box=box)
        b = ov.tile(f, "new", label=lab + "\nnew smplx_3db/", box=box)
        sep = np.full((480, 4, 3), 255, np.uint8)
        return np.concatenate([a, sep, b], 1)

    def write(name, pairs, cols=2):
        rows = []
        for k in range(0, len(pairs), cols):
            r = pairs[k:k + cols]
            r += [np.zeros_like(pairs[0])] * (cols - len(r))
            rows.append(np.concatenate([np.concatenate([x, np.full((480, 8, 3), 0, np.uint8)], 1) for x in r], 1))
        img = np.concatenate([np.concatenate([r, np.zeros((8, r.shape[1], 3), np.uint8)], 0) for r in rows], 0)
        p = SHEETS / name
        cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        return p

    out = []
    times = np.linspace(3.0, 85.0, 12)
    frames = [_frame_at(ov, x) for x in times]
    for s in range(3):
        out.append(write(f"overlay_{s + 1}.jpg", [pair(f) for f in frames[4 * s:4 * s + 4]]))
    focus = [(POPS["lead"], " John pop?"), (POPS["follow"], " Lula pop?")]
    if cmpf is not None:
        t = cmpf["t"]
        for r in ROLES:
            d = np.nan_to_num(cmpf[f"{r}_d_rr"])
            taken = []
            for i in np.argsort(-d):
                if all(abs(t[i] - x) > 2.0 for x in taken):
                    taken.append(float(t[i]))
                if len(taken) == 2:
                    break
            focus += [(x, f" {NAME[r]} biggest change") for x in taken]
            for kk, lab in ((f"{r}_lr_agree_new", f" {NAME[r]} new L/R != RTMW"), (f"{r}_lr_agree_current", f" {NAME[r]} cur L/R != RTMW")):
                a = cmpf[kk]
                o = cmpf[f"{r}_lr_agree_{'current' if kk.endswith('new') else 'new'}"]
                m = np.flatnonzero((a == 0) & (o == 1))
                if len(m):
                    focus.append((float(t[m[len(m) // 2]]), lab))
    for x in extra or []:
        focus.append((x, ""))
    pairs = [pair(_frame_at(ov, x), note) for x, note in focus]
    for s in range(0, len(pairs), 4):
        out.append(write(f"focus_{s // 4 + 1}.jpg", pairs[s:s + 4]))
    (SHEETS / "focus.json").write_text(json.dumps([{"t": x, "frame": _frame_at(ov, x), "why": n.strip()} for x, n in focus], indent=1))
    log(f"sheets: {[str(p) for p in out]}")
    return out


def clip(t0: float = 4.0, t1: float = 14.0, name: str = "sbs_current_vs_new_4-14s.mp4") -> Path:
    """A side-by-side clip (current smplx/ | new smplx_3db/) on the phone's own frames (538 x 960 each, 29.994 fps)."""
    import cv2
    ov = _Overlay()
    VIDEOS.mkdir(parents=True, exist_ok=True)
    fr = [f for f in range(len(ov.tc)) if t0 <= ov.tc[f] < t1]
    fps = 1.0 / float(np.median(np.diff(ov.tc)))
    tmp = VIDEOS / (name + ".avi")
    W, H = 538 * 2 + 4, 960 + 30
    W += W % 2
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"MJPG"), fps, (W, H))
    for f in fr:
        img = cv2.imread(str(ov.files[f]))
        a, b = img.copy(), img.copy()
        ov.draw(a, "current", f)
        ov.draw(b, "new", f)
        fr_img = np.zeros((H, W, 3), np.uint8)
        fr_img[30:, :538] = a
        fr_img[30:, 542:542 + 538] = b
        for x, txt in ((6, f"current smplx/   t={ov.tc[f]:.2f}s"), (548, "new smplx_3db/ (MAMMA + SAM 3D Body prior)")):
            cv2.putText(fr_img, txt, (x, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        vw.write(fr_img)
    vw.release()
    out = VIDEOS / name
    ff = shutil.which("ffmpeg") or r"C:\ffmpeg-8.1.1-essentials_build\bin\ffmpeg.exe"
    r = subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                        "-preset", "medium", "-r", f"{fps:.4f}", str(out)], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"ffmpeg: {r.stderr[-800:]}")
    tmp.unlink(missing_ok=True)
    log(f"clip {len(fr)} frames ({t0}-{t1} s, {W}x{H}) -> {out}")
    return out


# ================================================================================================ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.mono4d.jl_refit")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--chunk", choices=list(CHUNKS))
    sub.add_parser("build-check").add_argument("--chunk", choices=list(CHUNKS), required=True)
    j = sub.add_parser("job")
    j.add_argument("--chunk", choices=list(CHUNKS))
    j.add_argument("--max-minutes", type=float)
    sub.add_parser("stitch")
    sub.add_parser("masks")
    c = sub.add_parser("compare")
    c.add_argument("--no-sanity", action="store_true")
    sh = sub.add_parser("sheets")
    sh.add_argument("--extra", type=float, nargs="*")
    cl = sub.add_parser("clip")
    cl.add_argument("--t0", type=float, default=4.0)
    cl.add_argument("--t1", type=float, default=14.0)
    a = ap.parse_args(argv)
    if a.cmd == "stitch":
        print(json.dumps(do_stitch(), indent=1, default=str)[:3000])
    elif a.cmd == "masks":
        print(json.dumps(mask_agreement(), indent=1))
    elif a.cmd == "compare":
        rep = compare(not a.no_sanity)
        print(json.dumps(rep["roles"], indent=1, default=str)[:12000])
    elif a.cmd == "sheets":
        sheets(a.extra)
    elif a.cmd == "clip":
        clip(a.t0, a.t1, f"sbs_current_vs_new_{a.t0:g}-{a.t1:g}s.mp4")
    if a.cmd == "build":
        print(json.dumps(build_chunk(a.chunk) if a.chunk else build(), indent=1, default=str)[:4000])
    elif a.cmd == "build-check":
        print(json.dumps(build_check(a.chunk), indent=1))
    elif a.cmd == "job":
        for cid in ([a.chunk] if a.chunk else list(CHUNKS)):
            print(job(cid, a.max_minutes))
    return 0


if __name__ == "__main__":
    sys.exit(main())
