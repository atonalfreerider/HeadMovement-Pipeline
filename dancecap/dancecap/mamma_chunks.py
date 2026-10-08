"""Chunked MAMMA fit of a long take on ONE 80 GB GPU: overlapping grid chunks fitted in parallel, merged into one fit.

    python -m dancecap.mamma_chunks plan larissa_kadu_38 [--tag c8] [--chunks 6] [--overlap-s 1.2]   # local
                                    [--holdout-cam 03 --holdout-chunk 4]   # + one held-out check (below)
        -> work/<take>/mamma/chunks.json + work/<take>/mamma_chunks/pod_job.json (job "mamma", dancecap.pod_job/1)
    python -m dancecap.mamma_chunks fit larissa_kadu_38 [--par 6]     # pod: every chunk = `mamma fit --fit-frames a:b`
    python -m dancecap.mamma_chunks merge larissa_kadu_38            # pod or local: chunks -> fits/<tag>/ (selectable)
    python -m dancecap.mamma_chunks gate larissa_kadu_38 [--fallback full]   # pod: quality gate on the merge (below)
    python -m dancecap mamma select larissa_kadu_38 --tags c8 --no-qa   # promote to smplx/ (pod; again locally for QA)
    python -m dancecap.mamma_chunks recheck larissa_kadu_38          # local: seams/held-out/gate again from the chunk
                                                                     # fits with the current checks -> chunks_recheck.json

Why chunks: ma_3d batches every frame of a sequence in each of its ~675 Adam iterations (first 50 / second 500 / third
100 / fourth 25). Per iteration it costs ~0.40 s of fixed overhead (python, launches) + ~4.8 ms per frame on an A100
(larissa_kadu: 3 frames 297 s, 176 frames 858 s), so one 1151-frame fit would take ~65 min while six ~7.4 s chunks run
side by side under CUDA MPS in ~30-35 min (7 parallel 176-frame LOO fits took 2.0x their solo time, 2026-10-06).

Chunks: `n` equal chunks of the prepared grid with `overlap` frames shared by neighbours (default 36 = 1.2 s at 30 Hz).
They all use the SAME prepared inputs (ma_2d/<seq>, ma_cap/<seq>: ma_3d's --start_frame/--end_frame), so nothing extra
is prepared or uploaded; each writes fits/<tag>_f<a>-<b>/ (partial, never selectable on its own).

Merge (per dancer):
  betas    one body shape: the frame-weighted mean of the chunks' betas (MAMMA fits betas per sequence; chunks of the same
           person differ a little, see report betas_sd). Each chunk is re-expressed with the common betas by keeping its
           pose parameters and moving its translation so the PELVIS world trajectory is unchanged (transl += J0(b_chunk)
           - J0(b_common)); the joint displacement this causes is reported (betas_swap_mm: median / p95 over joints and
           frames, per chunk).
  seams    inside each overlap the two chunks are cross-faded with a cosine weight over its central half (the outer
           quarters keep the chunk whose interior they are): per-joint quaternion slerp, linear translation and contacts.
           The disagreement of the two chunks there is reported (seam_mm: joint-position median / p95 per seam) - a
           large value means the chunks found different solutions (a re-fit with more overlap is then due).
  export   FK with the common betas -> joints; reprojection per frame and part against every camera of the eval set
           (as mamma_dyn.export_smplx), outlier frames re-filled the same way; quality/quality_parts; smplx contract
           keys + chunks provenance -> fits/<tag>/smplx/{lead,follow}.npz and fits/<tag>/fit.json (residuals per camera,
           partial false) -> `mamma select` treats it like a full fit.
Report: mamma/chunks_merge.json (+ mamma/chunks.json plan, per-chunk timings in mamma/chunks_fit.json).

Held-out check (optional, plan --holdout-cam C): the fit set equals the evaluation set, so nothing tests the fit on a
camera it did not see. One chunk is fitted a second time, side by side with the others, from the prepared set h<C>
(the fit set without camera C: `mamma prepare --set h<C>=...`); merge reports camera C's reprojection from that fit
(held out) next to its reprojection from the merged fit over the same frames (in the fit): chunks_merge.json
"holdout". Informational, not part of the gate; its failure never fails the job.

Gate (chunks_merge.json "gate", checked by the `gate` command BEFORE select, so a bad merge is never promoted and the
avatar bake - a job that runs `after` mamma - is skipped): per role and seam, tested PER JOINT over the blend window
(seam_stats; each joint's own p95, so one limb placed differently by the two chunks fails): every body joint 0-17
<= 60 mm, every elbow/wrist <= 75 mm, every finger joint measured relative to its wrist <= 80 mm (curl differences
are weakly observed and stay below that: 12-71 mm on larissa_kadu_38; a flipped hand is ~15 cm), pelvis median
<= 40 mm (a chunk that found a different solution, not the ~1-3 cm noise of two fits of the same frames). Merge
reports without per-joint stats fall back to the pooled all-joint p95 <= 60 mm. Also every chunk's betas swap
p95 <= 40 mm, the merged median reprojection <= 1.25 x the chunks' frame-weighted
median + 0.5 px, and no chunk's own median reprojection above 2 x the chunks' median (a chunk that failed outright).
On failure `gate --fallback full` re-fits the whole grid in one ma_3d run (`mamma fit <take> --tag <tag>`, ~65 min on
an A100 for 1148 frames) when the pod's kill switch leaves time for it (/workspace/.dancecap_kill_at, written by
remote.arm_killswitch) and writes mamma/chunks_gate.json; otherwise it exits 3 (select and the bake do not run; the
downloaded chunk fits and chunks_merge.json say why).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import mamma_dyn as md
from . import progress as _progress  # dancecap.progress/1 (stdlib; no-op unless DANCECAP_PROGRESS)

POSE_KEYS = ("global_orient", "body_pose", "jaw_pose", "leye_pose", "reye_pose", "left_hand_pose", "right_hand_pose")
# measured on A100 pods 2026-10-06 (larissa_kadu): per ma_3d iteration a + b x frames, 675 iterations
ITER_FIXED_S, ITER_PER_FRAME_S, N_ITERS = 0.40, 0.0048, 675
PARALLEL_SLOWDOWN = 2.1  # 7 x 176-frame fits side by side under MPS took 2.03x their solo time
# merge quality gate (see the module docstring); gross-failure thresholds, not the noise of two fits of the same frames
GATE = {"seam_p95_mm": 60.0, "seam_arm_p95_mm": 75.0, "seam_fingers_rel_p95_mm": 80.0,
        "seam_pelvis_median_mm": 40.0, "betas_swap_p95_mm": 40.0, "reproj_ratio": 1.25,
        "reproj_slack_px": 0.5, "chunk_reproj_vs_median": 2.0}
# SMPL-X joints 0-21 (pelvis..wrists); the seam gate tests EACH of them (its own p95 over the seam's blend window), so
# one limb the two chunks put in different places fails the merge - a p95 pooled over 22 joints cannot see one joint
# (2026-10-07 larissa_kadu_38 follow seam 2: pooled body p95 44 mm, her left wrist 102 mm through the blend window)
BODY_JOINTS = ("pelvis", "l_hip", "r_hip", "spine1", "l_knee", "r_knee", "spine2", "l_ankle", "r_ankle", "spine3",
               "l_foot", "r_foot", "neck", "l_collar", "r_collar", "head", "l_shoulder", "r_shoulder", "l_elbow",
               "r_elbow", "l_wrist", "r_wrist")
ARM_JOINTS = (18, 19, 20, 21)  # elbows + wrists: own limit seam_arm_p95_mm (they move fastest and are least observed)
FINGERS = ((20, range(25, 40)), (21, range(40, 55)))  # (wrist, its 15 finger joints); measured relative to the wrist
KILL_AT = Path("/workspace/.dancecap_kill_at")  # epoch seconds of the on-pod kill switch (remote.arm_killswitch)
FALLBACK_MARGIN_MIN = 15.0  # select + pack + download of the mamma job after a fallback fit


# ------------------------------------------------------------------------------------------------ plan


def plan_chunks(T: int, n: int, overlap: int) -> list[tuple[int, int]]:
    """n chunks [a, b) of equal length covering 0..T, neighbours sharing `overlap` frames."""
    if n <= 1:
        return [(0, T)]
    L = int(math.ceil((T + (n - 1) * overlap) / n))
    stride = L - overlap
    out = []
    for k in range(n):
        a = min(k * stride, T - L)
        out.append((a, min(T, a + L)))
    out[-1] = (T - L, T)
    return out


def solo_minutes(frames: int) -> float:
    return (ITER_FIXED_S + ITER_PER_FRAME_S * frames) * N_ITERS / 60


def plan(take: str, tag: str | None = None, n: int = 6, overlap: int = 36, holdout_cam: str | None = None,
         holdout_chunk: int | None = None) -> dict:
    work = md.take_work(take)
    prep = json.loads((work / "mamma" / "prepare.json").read_text())
    tag = tag or max(prep["sets"], key=lambda k: len(prep["sets"][k]["cams"]))
    T = int(prep["grid"]["T"])
    chunks = plan_chunks(T, n, overlap)
    est_chunk = max(solo_minutes(b - a) for a, b in chunks) * (PARALLEL_SLOWDOWN if n > 1 else 1.0)
    doc = {"take": take, "tag": tag, "seq": prep["sets"][tag]["seq"], "cams": prep["sets"][tag]["cams"], "T": T,
           "grid_t0": prep["grid"]["t0"], "hz": prep["grid"]["hz"], "overlap": overlap,
           "chunks": [{"a": a, "b": b, "run_tag": f"{tag}_f{a}-{b}", "t": [prep["grid"]["t"][a], prep["grid"]["t"][b - 1]]}
                      for a, b in chunks],
           "estimate_min": {"chunk_parallel": round(est_chunk, 1), "single_fit_solo": round(solo_minutes(T), 1),
                            "basis": f"per iteration {ITER_FIXED_S} s + {ITER_PER_FRAME_S * 1000:.1f} ms/frame x "
                                     f"{N_ITERS} iterations (A100, larissa_kadu); x{PARALLEL_SLOWDOWN} side by side"},
           "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if holdout_cam:
        htag = f"h{holdout_cam}"
        hs = prep["sets"].get(htag)
        if hs is None or holdout_cam in hs["cams"] or holdout_cam not in prep["eval"]["cams"]:
            raise SystemExit(f"[chunks] held-out check needs a prepared set {htag} without camera {holdout_cam} "
                             f"(mamma prepare --set {htag}=<the {tag} cameras minus {holdout_cam}>) and {holdout_cam} "
                             "in the evaluation set")
        k = len(chunks) // 2 if holdout_chunk is None else int(holdout_chunk)
        a, b = chunks[k]
        doc["holdout"] = {"tag": htag, "cam": holdout_cam, "chunk": k, "a": a, "b": b, "run_tag": f"{htag}_f{a}-{b}",
                          "t": [prep["grid"]["t"][a], prep["grid"]["t"][b - 1]], "seq": hs["seq"]}
        # one more process side by side: budget it as if the GPU time grew with the work, (n + 1) / n
        doc["estimate_min"]["chunk_parallel"] = round(est_chunk * (len(chunks) + 1) / len(chunks), 1)
        doc["estimate_min"]["basis"] += f"; x{(len(chunks) + 1) / len(chunks):.2f} for the held-out run"
    (work / "mamma" / "chunks.json").write_text(json.dumps(doc, indent=1))
    return doc


# ------------------------------------------------------------------------------------------------ pod fit


def _mps(env: dict) -> dict | None:
    if not shutil.which("nvidia-cuda-mps-control"):
        return None
    pipe, logd = Path("/tmp/dancecap_mps_chunks"), Path("/tmp/dancecap_mps_chunks_log")
    pipe.mkdir(exist_ok=True)
    logd.mkdir(exist_ok=True)
    menv = dict(os.environ, CUDA_MPS_PIPE_DIRECTORY=str(pipe), CUDA_MPS_LOG_DIRECTORY=str(logd))
    if subprocess.run(["nvidia-cuda-mps-control", "-d"], env=menv).returncode != 0:
        return None
    env.update(CUDA_MPS_PIPE_DIRECTORY=str(pipe), CUDA_MPS_LOG_DIRECTORY=str(logd))
    return menv


def fit_chunks(take: str, par: int = 0, timeout_min: float = 80) -> dict:
    work = md.take_work(take)
    root = work / "mamma"
    doc = json.loads((root / "chunks.json").read_text())
    par = par or len(doc["chunks"])
    runs = doc["chunks"] + ([doc["holdout"]] if doc.get("holdout") else [])  # + the held-out check, side by side
    par = max(par, len(runs)) if par == len(doc["chunks"]) else par
    # pods can have a small CPU quota (8.5 CPUs on the 2026-10-06 A100 PCIe pod): one python driver per chunk,
    # 2 BLAS/OpenMP threads each
    env = dict(os.environ, DANCECAP_GPU_SHARED="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
               OPENBLAS_NUM_THREADS="2")
    mps = _mps(env)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    print(f"[chunks] {len(doc['chunks'])} chunks of {doc['tag']}" + (f" + held-out {doc['holdout']['run_tag']}"
          if doc.get("holdout") else "") + f", {par} in parallel, CUDA MPS {'on' if mps else 'off'}", flush=True)
    _progress.plan([{"step": "fit_chunks", "lane": c["run_tag"], "total": None, "unit": "iters",
                     "frames": c["b"] - c["a"]} for c in runs], step="fit_chunks", lanes=len(runs), parallel=par,
                   mps=bool(mps), tag=doc["tag"], est_min=doc.get("estimate_min", {}).get("chunk_parallel"))

    def one(c):
        t0 = time.time()
        # in-lane progress: MAMMA's own 'Iter. N of M' lines (ma_3d log), relayed by a thread (no-op when disabled)
        _progress.start("fit_chunks", lane=c["run_tag"], unit="iters")
        stop_watch = _progress.watch_log(root / "logs" / f"ma_3d_{c['run_tag']}.log", "mamma_iter",
                                         step="fit_chunks", lane=c["run_tag"], from_start=False)
        with open(root / "logs" / f"chunk_{c['run_tag']}.out", "w") as fh:
            pr = subprocess.Popen([sys.executable, "-m", "dancecap", "mamma", "fit", take, "--tag", c.get("tag", doc["tag"]),
                                   "--fit-frames", f"{c['a']}:{c['b']}"], stdout=fh, stderr=subprocess.STDOUT, env=env,
                                  start_new_session=True, cwd=str(md.PACKAGE_ROOT))
            try:
                rc = pr.wait(timeout=timeout_min * 60)
            except subprocess.TimeoutExpired:
                os.killpg(pr.pid, signal.SIGKILL)
                pr.wait()
                rc = "timeout"
        s = round(time.time() - t0, 1)
        stop_watch()
        (_progress.end if rc == 0 else _progress.fail)("fit_chunks", lane=c["run_tag"], **(
            {"seconds": s, "frames": c["b"] - c["a"]} if rc == 0 else {"rc": rc}))
        print(f"[chunks] {c['run_tag']}: rc={rc} in {s:.0f}s", flush=True)
        return c["run_tag"], {"rc": rc, "seconds": s}
    try:
        with ThreadPoolExecutor(par) as ex:
            res = dict(ex.map(one, runs))
    finally:
        if mps:
            subprocess.run(["nvidia-cuda-mps-control"], input=b"quit\n", env=mps)
    out = {"par": par, "mps": bool(mps), "chunks": res, "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (root / "chunks_fit.json").write_text(json.dumps(out, indent=1))
    bad = [k for k, v in res.items() if v["rc"] != 0 and k != (doc.get("holdout") or {}).get("run_tag")]
    if bad:
        raise SystemExit(f"[chunks] failed: {bad}")
    return out


# ------------------------------------------------------------------------------------------------ merge


def _rest_joints(betas: np.ndarray) -> np.ndarray:
    """Rest-pose joints (55, 3) of MAMMA's locked-head neutral model for these betas (no translation)."""
    T = 1
    pose = np.zeros((T, 165))
    _, j = md._forward(pose, np.zeros((T, 3)), betas, True)
    return j[0]


def _quat(aa: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    return Rotation.from_rotvec(aa.reshape(-1, 3)).as_quat().reshape(*aa.shape[:-1], 4)


def _slerp_aa(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Per-element slerp of axis-angle arrays (..., 3) from a (w=0) to b (w=1); w broadcast over the last axis."""
    from scipy.spatial.transform import Rotation
    qa, qb = _quat(a), _quat(b)
    dot = (qa * qb).sum(-1, keepdims=True)
    qb = np.where(dot < 0, -qb, qb)
    dot = np.abs(dot).clip(-1, 1)
    th = np.arccos(dot)
    s = np.sin(th)
    ww = w.reshape(w.shape + (1,) * (qa.ndim - w.ndim))
    small = s < 1e-6
    k0 = np.where(small, 1 - ww, np.sin((1 - ww) * th) / np.where(small, 1, s))
    k1 = np.where(small, ww, np.sin(ww * th) / np.where(small, 1, s))
    q = k0 * qa + k1 * qb
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    return Rotation.from_quat(q.reshape(-1, 4)).as_rotvec().reshape(a.shape)


def _seam_weights(n_ov: int) -> np.ndarray:
    """Weight of the LATER chunk over an overlap of n_ov frames: 0 in the first quarter, cosine ramp over the central
    half, 1 in the last quarter."""
    x = (np.arange(n_ov) + 0.5) / n_ov
    r = np.clip((x - 0.25) / 0.5, 0, 1)
    return 0.5 - 0.5 * np.cos(np.pi * r)


def _joint_p95(d: np.ndarray) -> dict:
    """d (frames, 55) mm -> per-joint p95 of the body joints, their worst (non-arm and arm separately)."""
    pj = np.percentile(d[:, :22], 95, axis=0)
    rest = [j for j in range(22) if j not in ARM_JOINTS]
    jr = max(rest, key=lambda j: pj[j])
    ja = max(ARM_JOINTS, key=lambda j: pj[j])
    return {"joint_p95_max": round(float(pj[jr]), 2), "joint": BODY_JOINTS[jr],
            "arm_p95_max": round(float(pj[ja]), 2), "arm_joint": BODY_JOINTS[ja],
            "per_joint_p95": {BODY_JOINTS[j]: round(float(pj[j]), 1) for j in range(22)}}


def seam_stats(jp: np.ndarray, jn: np.ndarray, w: np.ndarray | None = None) -> dict:
    """Disagreement of two chunks over their shared frames. jp, jn (n, 55, 3) metres: the earlier / later chunk's joints
    (both with the common betas); w (n,) the later chunk's merge weight (default the cosine seam). Reports the old pooled
    numbers (median, p95 over all 55 joints, p95_body over joints 0-21, p95_fingers, pelvis_median) and per-joint p95s
    over the BLEND window (0 < w < 1, the frames the cross-fade mixes; gated) and over the whole overlap (informational):
    the worst non-arm body joint, the worst elbow/wrist, and the worst finger joint measured relative to its wrist."""
    n = len(jp)
    w = _seam_weights(n) if w is None else np.asarray(w, float)
    d = np.linalg.norm(jp - jn, axis=-1) * 1000
    rel = np.concatenate([np.linalg.norm((jp[:, list(f)] - jp[:, [wr]]) - (jn[:, list(f)] - jn[:, [wr]]), axis=-1)
                          for wr, f in FINGERS], 1) * 1000
    out = {"median": round(float(np.median(d)), 2), "p95": round(float(np.percentile(d, 95)), 2),
           "p95_body": round(float(np.percentile(d[:, :22], 95)), 2),
           "p95_fingers": round(float(np.percentile(d[:, 25:], 95)), 2),
           "pelvis_median": round(float(np.median(d[:, 0])), 2)}
    for name, m in (("blend", (w > 0) & (w < 1)), ("overlap", np.ones(n, bool))):
        if m.sum() < 2:
            m = np.ones(n, bool)
        s = _joint_p95(d[m])
        s["fingers_rel_wrist_p95_max"] = round(float(np.percentile(rel[m], 95, axis=0).max()), 2)
        s["frames"] = int(m.sum())
        out[name] = s
    return out


def _common_betas(root: Path, chunks: list[dict], role: str):
    """Load the chunk fits of one role and re-express each with the frame-weighted common betas (pelvis trajectory
    kept) -> fits, betas, report stub {betas_*, seam_mm []}, poses, translations, joints (frames, 55, 3) per chunk."""
    fits = [dict(np.load(root / "fits" / c["run_tag"] / "smplx" / f"{role}.npz", allow_pickle=True)) for c in chunks]
    for c, f in zip(chunks, fits):
        if len(f["t"]) != c["b"] - c["a"]:
            raise ValueError(f"{c['run_tag']} {role}: {len(f['t'])} frames, plan {c['b'] - c['a']}")
    n = np.array([c["b"] - c["a"] for c in chunks], float)
    B = np.stack([np.asarray(f["betas"], float) for f in fits])
    betas = (B * n[:, None]).sum(0) / n.sum()
    J0c = _rest_joints(betas)[0]
    rr = {"betas_chunks": B[:, :6].round(4).tolist(), "betas_sd": B.std(0)[:6].round(4).tolist(),
          "betas": betas[:6].round(4).tolist(), "betas_swap_mm": [], "seam_mm": []}
    pose_c, transl_c, joints_own = [], [], []
    for c, f in zip(chunks, fits):
        pose = np.concatenate([np.asarray(f[k], float) for k in POSE_KEYS], 1)
        dJ = _rest_joints(np.asarray(f["betas"], float))[0] - J0c
        tr = np.asarray(f["transl"], float) + dJ
        _, j_new = md._forward(pose, tr, betas, True)
        d = np.linalg.norm(j_new - np.asarray(f["joints"], float)[:, :55], axis=-1) * 1000
        rr["betas_swap_mm"].append({"chunk": c["run_tag"], "median": round(float(np.median(d)), 2),
                                    "p95": round(float(np.percentile(d, 95)), 2)})
        pose_c.append(pose)
        transl_c.append(tr)
        joints_own.append(j_new)
    return fits, betas, rr, pose_c, transl_c, joints_own


def merge(take: str) -> dict:
    work = md.take_work(take)
    root = work / "mamma"
    doc = json.loads((root / "chunks.json").read_text())
    prep = json.loads((root / "prepare.json").read_text())
    tag, T = doc["tag"], doc["T"]
    grid = np.asarray(prep["grid"]["t"], float)
    ev = prep["eval"]
    gt_dir, lm_dir = root / "ma_cap" / ev["seq"] / "gt", root / "ma_2d" / ev["seq"]
    cams_ev = sorted(p.stem for p in gt_dir.glob("cam*.npz"))
    D = md._load_verts512()
    parts = md.part_index()
    chunks = doc["chunks"]
    out_dir = root / "fits" / tag
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "smplx").mkdir(parents=True)
    report = {"tag": tag, "chunks": [c["run_tag"] for c in chunks], "overlap": doc["overlap"], "roles": {}}
    summary = {}
    for p, pid in enumerate(md.PIDS):
        role = md.ROLES[pid]
        fits, betas, rr, pose_c, transl_c, joints_own = _common_betas(root, chunks, role)
        # stitch
        pose_all = np.zeros((T, 165))
        transl_all = np.zeros((T, 3))
        contact = np.zeros((T, md.N_LMK), np.float32)
        floor = np.zeros((T, md.N_LMK), np.float32)
        weight = np.zeros(T)
        owner = np.full(T, -1)
        for k, (c, f) in enumerate(zip(chunks, fits)):
            a, b = c["a"], c["b"]
            w = np.ones(b - a)
            if k > 0:  # overlap with the previous chunk at the start of this one
                ov = chunks[k - 1]["b"] - a
                if ov > 0:
                    w[:ov] = _seam_weights(ov)
            if k < len(chunks) - 1:
                ov = b - chunks[k + 1]["a"]
                if ov > 0:
                    w[-ov:] = np.minimum(w[-ov:], 1 - _seam_weights(ov))
            for i in range(b - a):
                g = a + i
                if weight[g] == 0:
                    pose_all[g], transl_all[g] = pose_c[k][i], transl_c[k][i]
                    contact[g], floor[g] = f["contact"][i], f["floor_contact"][i]
                    weight[g] = w[i]
                else:  # blend the earlier value (weight[g]) with this chunk (w[i]); weights sum to 1 by design
                    wt = w[i] / max(weight[g] + w[i], 1e-9)
                    pose_all[g] = _slerp_aa(pose_all[g].reshape(-1, 3), pose_c[k][i].reshape(-1, 3),
                                            np.full(55, wt)).reshape(-1)
                    transl_all[g] = (1 - wt) * transl_all[g] + wt * transl_c[k][i]
                    contact[g] = (1 - wt) * contact[g] + wt * np.asarray(f["contact"][i], np.float32)
                    floor[g] = (1 - wt) * floor[g] + wt * np.asarray(f["floor_contact"][i], np.float32)
                    weight[g] += w[i]
                owner[g] = k if w[i] >= 0.5 else owner[g]
            if k > 0:
                ov = chunks[k - 1]["b"] - a
                if ov > 0:
                    rr["seam_mm"].append({"between": [chunks[k - 1]["run_tag"], c["run_tag"]],
                                          "frames": [a, a + ov],
                                          **seam_stats(joints_own[k - 1][-ov:], joints_own[k][:ov],
                                                       _seam_weights(ov))})
        if (weight == 0).any():
            raise ValueError(f"{role}: grid frames without a chunk: {np.flatnonzero(weight == 0)[:10].tolist()}")
        verts, joints = md._forward(pose_all, transl_all, betas, True)
        rp = md.reprojection_parts(np.matmul(D, verts), cams_ev, gt_dir, lm_dir, p, parts)
        reproj_raw = rp["all"].copy()
        bad, suspect = md.outlier_frames(rp["all"], return_suspect=True)
        if bad.any() and (~bad).sum() >= 2:
            pose_all, transl_all = md._interpolate_frames(pose_all, transl_all, bad)
            verts, joints = md._forward(pose_all, transl_all, betas, True)
            rp = md.reprojection_parts(np.matmul(D, verts), cams_ev, gt_dir, lm_dir, p, parts)
            print(f"[chunks] {role}: repaired {int(bad.sum())} outlier frame(s) {np.flatnonzero(bad).tolist()[:20]}")
        reproj = rp["all"]
        part_px = np.stack([rp[k] for k in md.PARTS], 1)
        core = [k for k in md.PARTS if "hand" not in k]
        cp = np.stack([rp[k] for k in core], 1)
        worst = np.nanmax(np.where(np.isfinite(cp), cp, -np.inf), 1)
        worst = np.where(np.isfinite(worst), np.maximum(worst, np.nan_to_num(reproj, nan=0.0)), reproj)
        quality = np.exp(-np.nan_to_num(worst, nan=1e3) / 10.0)
        quality[bad] = np.minimum(quality[bad], 0.1)
        rep0 = [bool(np.asarray(f["repaired"]).any()) for f in fits]
        np.savez_compressed(
            out_dir / "smplx" / f"{role}.npz", t=grid[:T], global_orient=pose_all[:, :3], body_pose=pose_all[:, 3:66],
            jaw_pose=pose_all[:, 66:69], leye_pose=pose_all[:, 69:72], reye_pose=pose_all[:, 72:75],
            left_hand_pose=pose_all[:, 75:120], right_hand_pose=pose_all[:, 120:165], transl=transl_all, betas=betas,
            joints=joints, repaired=bad, suspect=suspect & ~bad, reproj_px_raw=reproj_raw, contact=contact,
            floor_contact=floor, quality=quality, quality_parts=np.exp(-np.nan_to_num(part_px, nan=1e3) / 10.0)
            .astype(np.float32), reproj_parts_px=part_px.astype(np.float32), parts=np.array(md.PARTS),
            reproj_px=reproj, gender="neutral", model="smplx_locked_head (MAMMA)", flat_hand_mean=True,
            num_betas=len(betas), source=f"mamma ma_3d, {len(chunks)} chunks merged (dancecap.mamma_chunks)",
            chunk_owner=owner.astype(np.int16), chunk_ranges=np.array([[c["a"], c["b"]] for c in chunks], np.int32))
        rr["chunks_had_repairs"] = rep0
        report["roles"][role] = rr
        summary[role] = {"frames": int(T), "median_reproj_px": float(np.nanmedian(reproj)),
                         "median_reproj_parts_px": {k: float(np.nanmedian(rp[k])) if np.isfinite(rp[k]).any() else None
                                                    for k in md.PARTS},
                         "quality_median": float(np.median(quality)), "repaired_frames": np.flatnonzero(bad).tolist(),
                         "suspect_not_repaired": np.flatnonzero(suspect & ~bad).tolist(), "betas0": float(betas[0])}
    resid = md.camera_residuals(out_dir / "smplx", gt_dir, lm_dir, ev["cams"])
    for cam in resid:
        resid[cam]["in_fit"] = cam in doc["cams"]
    chunk_fits = {c["run_tag"]: json.loads((root / "fits" / c["run_tag"] / "fit.json").read_text()) for c in chunks}
    fit_rep = {"tag": tag, "run_tag": tag, "cams": doc["cams"], "frames": None, "partial": False,
               "world_frame": prep["world_frame"], "chunked": True,
               "ma_3d": {"chunks": {k: v["ma_3d"] for k, v in chunk_fits.items()}},
               "smplx": summary, "residuals": resid, "eval_seq": ev["seq"], "host": "pod" if md.env.is_pod() else "local",
               "chunks": {k: {r: v["smplx"][r]["median_reproj_px"] for r in ("lead", "follow")}
                          for k, v in chunk_fits.items()},
               "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (out_dir / "fit.json").write_text(json.dumps(fit_rep, indent=1, default=str))
    report["smplx"] = summary
    report["chunk_median_reproj_px"] = fit_rep["chunks"]
    report["chunk_frames"] = {c["run_tag"]: int(c["b"] - c["a"]) for c in chunks}
    report["holdout"] = _holdout_report(root, doc, out_dir, gt_dir, lm_dir)
    report["gate"] = gate(report)
    (root / "chunks_merge.json").write_text(json.dumps(report, indent=1, default=str))
    print(f"[chunks] gate {'PASSED' if report['gate']['passed'] else 'FAILED: ' + '; '.join(report['gate']['why'])}")
    print(f"[chunks] merged {tag}: " + ", ".join(f"{r} median {v['median_reproj_px']:.2f} px" for r, v in summary.items())
          + "; seams " + "; ".join(f"{r} p95 " + ",".join(f"{s['p95']:.0f}" for s in report['roles'][r]['seam_mm'])
                                   + " mm" for r in report["roles"]) + f" -> {out_dir}")
    return report


def _holdout_report(root: Path, doc: dict, merged_dir: Path, gt_dir: Path, lm_dir: Path) -> dict | None:
    """Held-out camera C: its reprojection from the chunk fitted WITHOUT it vs from the merged fit (C in the fit) over
    the same grid frames."""
    h = doc.get("holdout")
    if not h:
        return None
    fj = root / "fits" / h["run_tag"] / "fit.json"
    if not fj.exists():
        return {"cam": h["cam"], "run_tag": h["run_tag"], "status": "missing (held-out fit failed or did not run)"}
    held = json.loads(fj.read_text())["residuals"].get(h["cam"], {})
    tmp = root / "fits" / f"_holdout_slice_{h['run_tag']}"
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        for role in ("lead", "follow"):
            z = dict(np.load(merged_dir / "smplx" / f"{role}.npz", allow_pickle=True))
            T = len(z["t"])
            np.savez(tmp / f"{role}.npz", **{k: (v[h["a"]:h["b"]] if isinstance(v, np.ndarray) and v.ndim and len(v) == T
                                                 else v) for k, v in z.items()})
        inf = md.camera_residuals(tmp, gt_dir, lm_dir, [h["cam"]], frame0=h["a"]).get(h["cam"], {})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # baseline: the chunk fitted over the SAME frames with camera C (same length and context, only C differs); the
    # merged fit over those frames is kept for reference (2026-10-07 review: 1.20/1.13 vs 1.18/1.17 from the merge)
    same = next((c for c in doc["chunks"] if c["a"] == h["a"] and c["b"] == h["b"]), None)
    same_res = {}
    if same is not None and (root / "fits" / same["run_tag"] / "fit.json").exists():
        same_res = json.loads((root / "fits" / same["run_tag"] / "fit.json").read_text())["residuals"].get(h["cam"], {})
    out = {"cam": h["cam"], "run_tag": h["run_tag"], "frames": [h["a"], h["b"]], "t": h["t"], "status": "ok",
           "baseline": same["run_tag"] if same_res else f"merged {merged_dir.name}"}
    for role in ("lead", "follow"):
        a = (held.get(role) or {}).get("median_px")
        m = (inf.get(role) or {}).get("median_px")
        s = (same_res.get(role) or {}).get("median_px")
        b = s if s is not None else m
        out[role] = {"held_out_median_px": a, "in_fit_median_px": b, "in_fit_same_chunk_median_px": s,
                     "in_fit_merged_median_px": m, "ratio": round(a / b, 3) if a is not None and b else None}
    print(f"[chunks] held-out cam{h['cam']} ({h['run_tag']} vs {out['baseline']}): " + ", ".join(
        f"{r} {out[r]['held_out_median_px'] or float('nan'):.1f} px held out vs {out[r]['in_fit_median_px'] or float('nan'):.1f} "
        "px in the fit" for r in ("lead", "follow")))
    return out


# ------------------------------------------------------------------------------------------------ gate


def gate(report: dict, th: dict | None = None) -> dict:
    """Quality gate of a merge report (chunks_merge.json) -> {passed, why [...], checks {...}, thresholds}."""
    th = {**GATE, **(th or {})}
    why, checks = [], {}
    frames = report.get("chunk_frames") or {}
    for role, rr in (report.get("roles") or {}).items():
        seams = rr.get("seam_mm", [])
        # per joint over the blend window (seam_stats): worst non-arm body joint, worst elbow/wrist, worst finger
        # relative to its wrist. Reports from before per-joint stats keep the pooled all-joint p95 (stricter there).
        per = [s["blend"] for s in seams if "blend" in s]
        if per and len(per) == len(seams):
            seam_p95 = max(s["joint_p95_max"] for s in per)
            arm_p95 = max(s["arm_p95_max"] for s in per)
            fing_p95 = max(s["fingers_rel_wrist_p95_max"] for s in per)
            worst = max(per, key=lambda s: s["arm_p95_max"])
            arm_at = f"{worst['arm_joint']} at seam {per.index(worst) + 1}"
        else:
            seam_p95 = max([s["p95"] for s in seams] or [0.0])
            arm_p95 = fing_p95 = None
            arm_at = None
        seam_pel = max([s.get("pelvis_median", 0.0) for s in seams] or [0.0])
        swap_p95 = max([s["p95"] for s in rr.get("betas_swap_mm", [])] or [0.0])
        chunk_px = {k: v[role] for k, v in (report.get("chunk_median_reproj_px") or {}).items()
                    if v.get(role) is not None}
        if chunk_px:
            w = np.array([frames.get(k, 1) for k in chunk_px], float)
            ref = float(np.average(list(chunk_px.values()), weights=w))
            med = float(np.median(list(chunk_px.values())))
        else:
            ref = med = float("nan")
        merged = float((report.get("smplx") or {}).get(role, {}).get("median_reproj_px", float("nan")))
        lim_px = th["reproj_ratio"] * ref + th["reproj_slack_px"]
        bad_chunks = sorted(k for k, v in chunk_px.items() if v > th["chunk_reproj_vs_median"] * med)
        checks[role] = {"seam_p95_mm": seam_p95, "seam_arm_p95_mm": arm_p95, "seam_arm_worst": arm_at,
                        "seam_fingers_rel_p95_mm": fing_p95, "seam_pelvis_median_mm": seam_pel,
                        "betas_swap_p95_mm": swap_p95,
                        "merged_reproj_px": round(merged, 3), "chunks_reproj_px": round(ref, 3),
                        "merged_reproj_limit_px": round(lim_px, 3), "failed_chunks": bad_chunks}
        if seam_p95 > th["seam_p95_mm"]:
            why.append(f"{role}: seam p95 {seam_p95:.0f} mm > {th['seam_p95_mm']:.0f}"
                       + (" (worst body joint)" if arm_p95 is not None else " (all joints pooled, old report)"))
        if arm_p95 is not None and arm_p95 > th["seam_arm_p95_mm"]:
            why.append(f"{role}: seam {arm_at} p95 {arm_p95:.0f} mm > {th['seam_arm_p95_mm']:.0f}")
        if fing_p95 is not None and fing_p95 > th["seam_fingers_rel_p95_mm"]:
            why.append(f"{role}: seam fingers (relative to the wrist) p95 {fing_p95:.0f} mm > "
                       f"{th['seam_fingers_rel_p95_mm']:.0f}")
        if seam_pel > th["seam_pelvis_median_mm"]:
            why.append(f"{role}: seam pelvis median {seam_pel:.0f} mm > {th['seam_pelvis_median_mm']:.0f}")
        if swap_p95 > th["betas_swap_p95_mm"]:
            why.append(f"{role}: betas swap p95 {swap_p95:.0f} mm > {th['betas_swap_p95_mm']:.0f}")
        if not (merged <= lim_px):  # NaN fails too
            why.append(f"{role}: merged reprojection {merged:.2f} px > {lim_px:.2f} (chunks {ref:.2f})")
        if bad_chunks:
            why.append(f"{role}: chunk(s) {bad_chunks} reproject > {th['chunk_reproj_vs_median']}x the chunks' median")
    if not checks:
        why.append("no roles in the merge report")
    return {"passed": not why, "why": why, "checks": checks, "thresholds": th}


def recheck(take: str, merged: str | None = None) -> dict:
    """Re-run the merge checks from the chunk fits without re-merging (local, CPU, ~1 min): seam stats with the
    current seam_stats (per joint), the held-out comparison against the same-frames chunk, and the current gate on the
    merge's stored reprojection numbers. -> mamma/chunks_recheck.json (chunks_merge.json, the pod's record, is kept)."""
    root = md.take_work(take) / "mamma"
    doc = json.loads((root / "chunks.json").read_text())
    rep = json.loads((root / "chunks_merge.json").read_text())
    chunks = doc["chunks"]
    roles = {}
    for role in ("lead", "follow"):
        _, _, rr, _, _, joints_own = _common_betas(root, chunks, role)
        for k in range(1, len(chunks)):
            a, ov = chunks[k]["a"], chunks[k - 1]["b"] - chunks[k]["a"]
            if ov > 0:
                rr["seam_mm"].append({"between": [chunks[k - 1]["run_tag"], chunks[k]["run_tag"]], "frames": [a, a + ov],
                                      **seam_stats(joints_own[k - 1][-ov:], joints_own[k][:ov], _seam_weights(ov))})
        roles[role] = rr
    new = {**{k: v for k, v in rep.items() if k not in ("roles", "gate", "holdout")}, "roles": roles}
    mdir = root / "fits" / (merged or f"{doc['tag']}_merged_rejected")
    if not (mdir / "smplx").exists():
        mdir = root / "fits" / doc["tag"]
    prep = json.loads((root / "prepare.json").read_text())
    ev = prep["eval"]
    new["holdout"] = _holdout_report(root, doc, mdir, root / "ma_cap" / ev["seq"] / "gt", root / "ma_2d" / ev["seq"])
    new["gate"] = gate(new)
    new["gate_at_merge"] = rep.get("gate")
    new["rechecked"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (root / "chunks_recheck.json").write_text(json.dumps(new, indent=1, default=str))
    print(f"[chunks] recheck gate {'PASSED' if new['gate']['passed'] else 'FAILED: ' + '; '.join(new['gate']['why'])}")
    for role, rr in roles.items():
        for i, s in enumerate(rr["seam_mm"], 1):
            b = s["blend"]
            print(f"[chunks] {role} seam {i} {s['between']}: blend {b['frames']} fr: worst body {b['joint']} "
                  f"{b['joint_p95_max']:.0f} mm, worst arm {b['arm_joint']} {b['arm_p95_max']:.0f} mm, fingers rel "
                  f"{b['fingers_rel_wrist_p95_max']:.0f} mm | overlap arm {s['overlap']['arm_joint']} "
                  f"{s['overlap']['arm_p95_max']:.0f} mm | pooled body {s['p95_body']:.0f} mm")
    return new


def kill_minutes_left(path: Path | None = None) -> float | None:
    """Minutes until the on-pod kill switch fires (None when unknown, e.g. locally)."""
    try:
        return (float((path or KILL_AT).read_text().split()[0]) - time.time()) / 60.0
    except (OSError, ValueError, IndexError):
        return None


def _retag_rejected(rejected: Path) -> None:
    """The moved merge keeps fit.json tag <tag>; `mamma select` keys fits by that tag, so the rejected merge would
    shadow the fallback fit's report in select.json/report.json (2026-10-07, larissa_kadu_38). Give it its own tag."""
    fj = rejected / "fit.json"
    if fj.exists():
        r = json.loads(fj.read_text())
        r.update(tag=rejected.name, run_tag=rejected.name, rejected=True, partial=True)
        fj.write_text(json.dumps(r, indent=1, default=str))


def gate_or_fallback(take: str, fallback: str = "none", fit_timeout_min: float | None = None) -> int:
    """Pod command between merge and select. Gate passed -> 0. Failed: fallback "full" re-fits the whole grid in one
    run when the kill switch leaves time (-> 0 if that fit finished), else 3. Writes mamma/chunks_gate.json."""
    work = md.take_work(take)
    root = work / "mamma"
    rep = json.loads((root / "chunks_merge.json").read_text())
    g = rep.get("gate") or gate(rep)
    doc = json.loads((root / "chunks.json").read_text())
    out = {"take": take, "gate": g, "fallback": None, "finished": None}
    rc = 0
    if g["passed"]:
        print("[chunks] gate passed: " + json.dumps(g["checks"]))
        _progress.note("gate passed", step="gate", branch="pass", gate=True)
    else:
        print("[chunks] gate FAILED: " + "; ".join(g["why"]))
        _progress.note("gate FAILED: " + "; ".join(g["why"]), step="gate", level="warn", branch="fallback?",
                       gate=False)
        need = solo_minutes(doc["T"]) * 1.25 + FALLBACK_MARGIN_MIN
        left = kill_minutes_left()
        out["fallback"] = {"mode": fallback, "need_min": round(need, 1),
                           "kill_minutes_left": None if left is None else round(left, 1)}
        if fallback != "full":
            rc = 3
        elif left is not None and left < need:
            print(f"[chunks] no time for the full-fit fallback ({left:.0f} min left, {need:.0f} needed)")
            out["fallback"]["result"] = "skipped: not enough pod time"
            rc = 3
        else:
            rejected = root / "fits" / f"{doc['tag']}_merged_rejected"
            if rejected.exists():
                shutil.rmtree(rejected)
            shutil.move(str(root / "fits" / doc["tag"]), str(rejected))
            _retag_rejected(rejected)
            print(f"[chunks] fallback: one full fit of {doc['tag']} ({doc['T']} frames, ~{solo_minutes(doc['T']):.0f} "
                  f"min); the rejected merge is kept in {rejected}", flush=True)
            t0 = time.time()
            env = {k: v for k, v in os.environ.items() if k != "DANCECAP_GPU_SHARED"}
            (root / "logs").mkdir(parents=True, exist_ok=True)
            _progress.plan([{"step": "gate/fallback_fit", "total": None, "unit": "iters", "frames": doc["T"]}],
                           step="gate", branch="fallback", eta_s=round(solo_minutes(doc["T"]) * 60))
            stop_watch = _progress.watch_log(root / "logs" / f"ma_3d_{doc['tag']}.log", "mamma_iter",
                                             step="gate/fallback_fit", lane="full", from_start=False)
            try:
                with open(root / "logs" / "fallback_full_fit.out", "w") as fh:
                    pr = subprocess.run([sys.executable, "-m", "dancecap", "mamma", "fit", take, "--tag", doc["tag"]],
                                        stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=str(md.PACKAGE_ROOT),
                                        timeout=None if fit_timeout_min is None else fit_timeout_min * 60)
                prc = pr.returncode
            except subprocess.TimeoutExpired:
                prc = "timeout"
            stop_watch()
            ok = prc == 0 and (root / "fits" / doc["tag"] / "fit.json").exists()
            (_progress.end if ok else _progress.fail)("gate/fallback_fit", lane="full",
                                                      **({"seconds": round(time.time() - t0, 1)} if ok else {"rc": prc}))
            out["fallback"].update(result="full fit ok" if ok else f"full fit failed rc={prc}",
                                   seconds=round(time.time() - t0, 1))
            rc = 0 if ok else 3
    out["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (root / "chunks_gate.json").write_text(json.dumps(out, indent=1, default=str))
    return rc


# ------------------------------------------------------------------------------------------------ job spec


def write_job(take: str, par: int = 0, extra_minutes: float = 12, fallback: str = "full") -> Path:
    """work/<take>/mamma_chunks/pod_job.json (job name "mamma", so the avatar bake job runs after it). fallback
    "full": the job's time budget includes one full re-fit, run only when the merge gate fails."""
    work = md.take_work(take)
    doc = json.loads((work / "mamma" / "chunks.json").read_text())
    prep = json.loads((work / "mamma" / "prepare.json").read_text())
    w = f"dancecap/work/{take}"
    seqs = sorted({prep["sets"][doc["tag"]]["seq"], prep["eval"]["seq"]}
                  | ({doc["holdout"]["seq"]} if doc.get("holdout") else set()))
    fit_min = doc["estimate_min"]["chunk_parallel"]
    fb_min = solo_minutes(doc["T"]) * 1.25 + 5 if fallback == "full" else 0.0  # full re-fit only if the gate fails
    mm = int(math.ceil(fit_min * 1.4 + extra_minutes + fb_min))
    job = {
        "schema": "dancecap.pod_job/1", "name": "mamma", "take": take, "max_minutes": mm, "needs": ["mamma"],
        "inputs": [f"{w}/mamma/prepare.json", f"{w}/mamma/ma3d_config.yaml", f"{w}/mamma/chunks.json",
                   *[f"{w}/mamma/ma_2d/{s}/" for s in seqs], *[f"{w}/mamma/ma_cap/{s}/" for s in seqs]],
        "env": {"PYOPENGL_PLATFORM": "egl"},
        "commands": [
            {"name": "fit_chunks", "timeout_min": int(math.ceil(fit_min * 1.4)),
             "cmd": f"python -m dancecap.mamma_chunks fit {take} --par {par or len(doc['chunks'])} "
                    f"--timeout-min {int(math.ceil(fit_min * 1.35))}"},
            {"name": "merge", "timeout_min": 15, "cmd": f"python -m dancecap.mamma_chunks merge {take}"},
            {"name": "gate", "timeout_min": int(math.ceil(max(fb_min, 2))),
             "cmd": f"python -m dancecap.mamma_chunks gate {take} --fallback {fallback}"
                    + (f" --fit-timeout-min {int(math.ceil(solo_minutes(doc['T']) * 1.25))}" if fallback == "full" else "")},
            {"name": "select", "timeout_min": 10, "cmd": f"python -m dancecap mamma select {take} --tags {doc['tag']} "
                                                         "--no-qa"},
        ],
        "outputs": [f"{w}/mamma/fits/", f"{w}/mamma/logs/", f"{w}/mamma/select.json", f"{w}/mamma/report.json",
                    f"{w}/mamma/chunks_fit.json", f"{w}/mamma/chunks_merge.json", f"{w}/mamma/chunks_gate.json",
                    f"{w}/smplx/"],
        "verify": [f"{w}/mamma/fits/{doc['tag']}/fit.json", f"{w}/smplx/lead.npz", f"{w}/smplx/follow.npz",
                   f"{w}/mamma/chunks_merge.json"],
        "after_download": [f"python -m dancecap mamma select {take} --tags {doc['tag']}   # local QA sheet "
                           f"review/pose_mamma_{doc['tag']}.jpg",
                           f"python -m dancecap.cameras.world_scale check {take} --out "
                           f"work/{take}/scale/check_after_fit.json",
                           f"python -m dancecap physics {take} --height-lead 1.727 --height-follow 1.651"],
        "notes": [f"{len(doc['chunks'])} chunks x ~{doc['chunks'][0]['b'] - doc['chunks'][0]['a']} grid frames, overlap "
                  f"{doc['overlap']} ({doc['overlap'] / doc['hz']:.1f} s), fitted side by side under CUDA MPS "
                  "(dancecap.mamma_chunks); merged with common betas and cosine seams; gates in chunks_merge.json: "
                  "seam_mm (chunk disagreement), betas_swap_mm (shape unification), merged vs chunk reprojection",
                  f"gate before select (thresholds {GATE}); fallback {fallback!r}"
                  + (f": one full re-fit (~{solo_minutes(doc['T']):.0f} min) when the gate fails, budgeted in "
                     "max_minutes" if fallback == "full" else ": none (a failed gate stops select and the bake)"),
                  f"estimate: {fit_min:.0f} min of fitting (single full fit ~{doc['estimate_min']['single_fit_solo']:.0f} "
                  "min)",
                  (f"held-out check: chunk {doc['holdout']['chunk']} also fitted without cam{doc['holdout']['cam']} "
                   f"(set {doc['holdout']['tag']}, run {doc['holdout']['run_tag']}) -> chunks_merge.json holdout"
                   if doc.get("holdout") else "no held-out check")],
    }
    p = work / "mamma_chunks" / "pod_job.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(job, indent=1))
    print(f"[chunks] -> {p} (max {mm} min)")
    return p


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m dancecap.mamma_chunks", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["plan", "fit", "merge", "gate", "write-job", "recheck"])
    ap.add_argument("--merged", help="recheck: merged fit dir under mamma/fits/ for the held-out reference "
                                     "(default <tag>_merged_rejected, else <tag>)")
    ap.add_argument("take")
    ap.add_argument("--tag")
    ap.add_argument("--chunks", type=int, default=6)
    ap.add_argument("--overlap-s", type=float, default=1.2)
    ap.add_argument("--par", type=int, default=0)
    ap.add_argument("--timeout-min", type=float, default=80)
    ap.add_argument("--fallback", choices=["full", "none"], default="full",
                    help="gate: what to do when the merge gate fails; plan/write-job: budget the full re-fit or not")
    ap.add_argument("--fit-timeout-min", type=float, default=None, help="gate: timeout of the fallback full fit")
    ap.add_argument("--holdout-cam", help="plan: refit one chunk without this camera (prepared set h<cam>) as a check")
    ap.add_argument("--holdout-chunk", type=int, default=None, help="plan: which chunk (default the middle one)")
    a = ap.parse_args(argv)
    if a.action == "plan":
        d = plan(a.take, a.tag, a.chunks, int(round(a.overlap_s * md.GRID_HZ)), a.holdout_cam, a.holdout_chunk)
        print(json.dumps({k: d[k] for k in ("tag", "T", "overlap", "estimate_min")}, indent=1))
        for c in d["chunks"]:
            print(f"[chunks] {c['run_tag']}: {c['b'] - c['a']} frames, t {c['t'][0]:.3f}-{c['t'][1]:.3f} s")
        write_job(a.take, a.par, fallback=a.fallback)
    elif a.action == "write-job":
        write_job(a.take, a.par, fallback=a.fallback)
    elif a.action == "gate":
        raise SystemExit(gate_or_fallback(a.take, a.fallback, a.fit_timeout_min))
    elif a.action == "fit":
        fit_chunks(a.take, a.par, a.timeout_min)
    elif a.action == "recheck":
        recheck(a.take, a.merged)
    else:
        merge(a.take)


if __name__ == "__main__":
    main()
