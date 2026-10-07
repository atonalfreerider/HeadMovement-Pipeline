"""One Runpod session for every heavy hand step of a take (Runpod-first, user 2026-10-06: the PC has 4 logical CPUs and
a shared 2080 Ti, so detectors / MANO regressors / multi-view fits run on a pod; locally only orchestration + review).

    python -m dancecap.hands_pod spec <take> [--no-hamer] [--no-contact]   -> work/<take>/hands/pod_job.json (+ summary)
    python -m dancecap.hands_pod run  <take> [--max-cost 3] [--no-hamer] [--dry-run] [--session SID] [--force]
    python -m dancecap.hands_pod run  <take> --build [--max-cost 2]    the detail build (hands_detail): wrist gate,
                                           fingers (+ parallel LOO), guarded contact, smplx_hands/, smplx_detail/,
                                           metrics, the mesh review images (GPU for the exact contact geometry) - on the
                                           existing hand evidence; no frames, no MANO file in the bundle (SECURE cloud
                                           for reliability); then locally the video crops:
                                           python -m dancecap.refine_hands review <take> --video-only

The job (dancecap.pod_job/1, run by dancecap.remote_jobs: one bundle upload, bootstrap base, resumable commands with
.done markers, outputs downloaded + whitelisted) does, in order:
  survey      (only when hands/jobs.json is missing, e.g. a new take)  refine_hands survey
  kp2d        hand_worker --device cuda: RTMW-x whole-body, RTMPose-m hand, MediaPipe on every hand job
  wilor       hand_mano_worker (WiLoR + its YOLO hand detector)
  detectors   refine_hands detectors (keypoint detectors + WiLoR joints: detection rates, triangulation, LOO)
  mano        refine_hands mano (fuse WiLoR across views + time, LOO)
  fingers_ik  refine_hands fingers (keypoint IK, parallel LOO)            -> hands/smplx_fingers_ik/
  fingers_hy  refine_hands fingers --mano wilor --tag hybrid               -> hands/smplx_fingers_hybrid/
  contact     refine_hands contact --smplx-dir <hybrid, else ik, else smplx>  -> hands/smplx/ (fingers + contact)
  compare     refine_hands compare
  hamer...    HaMeR (optional, last: 6 GB download started in the background at the beginning), then detectors / mano /
              compare again so HaMeR joins the comparison.
Every step after kp2d is optional (a failure is reported, later steps still run; logs come back either way).

Licences: the bundle carries the user's MANO_RIGHT.pkl (plain-numpy copy, models/mano/nochumpy/; MANO licence: non-
commercial, no redistribution), so the pod is created in Runpod's SECURE cloud only and deleted at the end. Open weights
(RTMW / RTMPose / MediaPipe: Apache-2.0; WiLoR: CC-BY-NC-ND 4.0; HaMeR checkpoint: MANO-derived, non-commercial) are
downloaded on the pod from the authors' official locations (remote/hands_setup.sh). Budget: the global $30 cap and the
per-session --max-cost are enforced by dancecap.remote before the pod exists.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from . import env
from .take import PACKAGE_ROOT, WORKSPACE

# 48 GB-class GPUs are plenty (ViT-H at batch 48 < 12 GB); cheapest secure first, 80 GB as a fallback
GPUS = ["NVIDIA A40", "NVIDIA RTX A6000", "NVIDIA RTX 6000 Ada Generation", "NVIDIA L40S", "NVIDIA A100 80GB PCIe",
        "NVIDIA A100-SXM4-80GB"]
CLOUDS = ["SECURE"]
MANO_PLAIN = WORKSPACE / "models" / "mano" / "nochumpy" / "MANO_RIGHT.pkl"
MAX_MINUTES = 150


def _rel(p: Path) -> str:
    return p.resolve().relative_to(WORKSPACE.resolve()).as_posix()


def spec(take: str, hamer: bool = True, contact: bool = True, analysis_only: bool = False,
         hamer_only: bool = False) -> dict:
    """analysis_only: the 2D detectors / WiLoR already ran (hands/detections.npz, mano_wilor.npz exist locally) - upload
    hands/ and run only the fusion / IK / contact steps (+ HaMeR when hands/mano_hamer.npz is still missing)."""
    from .refine_hands import fit_cams
    work = PACKAGE_ROOT / "work" / take
    w = f"dancecap/work/{take}"
    cams = fit_cams(work)
    need_survey = not (work / "hands" / "jobs.json").exists()
    sel = work / "mamma" / "select.json"
    tag = json.loads(sel.read_text()).get("chosen") if sel.exists() else None
    md = env.mamma_data() / "body_models"
    inputs = [f"{w}/smplx/lead.npz", f"{w}/smplx/follow.npz", f"{w}/sync/offsets.json", f"{w}/frames/",
              f"{w}/cameras/world_from_sfm.json", *[f"{w}/cameras/{c}.npz" for c in cams],
              *[f"{w}/dancers/{c}/masks/" for c in cams],
              {"local": str(md / "smplx_locked_head" / "smplx" / "SMPLX_NEUTRAL.npz"),
               "pod": "mamma/data/body_models/smplx_locked_head/smplx/SMPLX_NEUTRAL.npz"},
              {"local": str(md / "downsampled_verts" / "verts_512.pkl"),
               "pod": "mamma/data/body_models/downsampled_verts/verts_512.pkl"},
              {"local": str(md / "downsampled_verts" / "smplx_512_body_parts.npz"),
               "pod": "mamma/data/body_models/downsampled_verts/smplx_512_body_parts.npz"},
              _rel(MANO_PLAIN)]
    if sel.exists():
        inputs.append(f"{w}/mamma/select.json")
        if tag and (work / "mamma" / "ma_2d" / f"{take}_{tag}").exists():
            inputs.append(f"{w}/mamma/ma_2d/{take}_{tag}/")
    analysis_only = analysis_only or hamer_only
    if analysis_only:
        if not (work / "hands" / "detections.npz").exists():
            raise SystemExit("analysis_only needs hands/detections.npz (run the full session first)")
        need_survey = False
        hamer = hamer and not (work / "hands" / "mano_hamer.npz").exists()
        inputs += [f"{w}/hands/"]
    elif not need_survey:
        inputs += [f"{w}/hands/jobs.json", f"{w}/hands/survey.npz", f"{w}/hands/survey.json"]
    ncpu = "N=$(bash remote/hands_setup.sh ncpu); W=$(( N > 13 ? 12 : (N > 3 ? N - 1 : 2) )); "
    rh = "python -m dancecap.refine_hands"
    cmds = []
    if hamer:
        cmds.append({"name": "hamer_fetch", "cmd": "bash remote/hands_setup.sh hamer_fetch", "timeout_min": 2,
                     "optional": True})
    if need_survey:
        cmds.append({"name": "survey", "cmd": f"{rh} survey {take}", "timeout_min": 30})
    if hamer_only:   # HaMeR joins the comparison; fingers / contact results already exist and are kept
        cmds = [{"name": "setup_hamer", "cmd": "bash remote/hands_setup.sh hamer", "timeout_min": 35},
                {"name": "hamer", "cmd": f"python -m dancecap.hand_mano_worker run {take} --model hamer "
                                         f"--repo /workspace/ext/hamer", "timeout_min": 30},
                {"name": "detectors", "cmd": f"{rh} detectors {take}", "timeout_min": 15, "optional": True},
                {"name": "mano", "cmd": f"{rh} mano {take}", "timeout_min": 20, "optional": True},
                {"name": "compare", "cmd": f"{rh} compare {take}", "timeout_min": 5, "optional": True}]
        hamer = contact = False
    if not analysis_only:
        cmds += [
            {"name": "setup_detectors", "cmd": "bash remote/hands_setup.sh detectors", "timeout_min": 15},
            {"name": "kp2d", "cmd": f"python -m dancecap.hand_worker {take} --device cuda", "timeout_min": 40},
            {"name": "setup_wilor", "cmd": "bash remote/hands_setup.sh wilor", "timeout_min": 20, "optional": True},
            {"name": "wilor", "cmd": f"python -m dancecap.hand_mano_worker run {take} --model wilor "
                                     f"--repo /workspace/ext/WiLoR", "timeout_min": 25, "optional": True}]
    cmds += [] if hamer_only else [
        {"name": "detectors", "cmd": f"{rh} detectors {take}", "timeout_min": 15, "optional": True},
        {"name": "mano", "cmd": f"{rh} mano {take}", "timeout_min": 20, "optional": True},
        {"name": "fingers_ik", "cmd": ncpu + f"{rh} fingers {take} --workers $W", "timeout_min": 30, "optional": True},
        {"name": "fingers_hybrid", "cmd": ncpu + f"{rh} fingers {take} --mano wilor --tag hybrid --workers $W",
         "timeout_min": 30, "optional": True},
    ]
    if contact and not hamer_only:
        cmds.append({"name": "contact", "timeout_min": 45, "optional": True, "cmd": (
            f"N=$(bash remote/hands_setup.sh ncpu); export OMP_NUM_THREADS=$N MKL_NUM_THREADS=$N; "
            f"d=work/{take}/hands/smplx_fingers_hybrid; [ -f $d/lead.npz ] || d=work/{take}/hands/smplx_fingers_ik; "
            f"[ -f $d/lead.npz ] || d=work/{take}/smplx; echo contact from $d; {rh} contact {take} --smplx-dir $d")})
    if not hamer_only:
        cmds.append({"name": "compare", "cmd": f"{rh} compare {take}", "timeout_min": 5, "optional": True})
    if hamer:
        cmds += [
            {"name": "setup_hamer", "cmd": "bash remote/hands_setup.sh hamer", "timeout_min": 35, "optional": True},
            {"name": "hamer", "cmd": f"python -m dancecap.hand_mano_worker run {take} --model hamer --repo /workspace/ext/hamer",
             "timeout_min": 30, "optional": True},
            {"name": "detectors2", "cmd": f"{rh} detectors {take}", "timeout_min": 15, "optional": True},
            {"name": "mano2", "cmd": f"{rh} mano {take}", "timeout_min": 20, "optional": True},
            {"name": "compare2", "cmd": f"{rh} compare {take}", "timeout_min": 5, "optional": True},
        ]
    return {
        "schema": "dancecap.pod_job/1", "name": "hands", "take": take,
        "max_minutes": sum(c["timeout_min"] for c in cmds),
        "needs": [], "inputs": inputs, "commands": cmds,
        "exclude": [f"{w}/hands/logs/*", f"{w}/hands/pod_job.json"],   # local logs may be open (Windows locks)
        "env": {"PYOPENGL_PLATFORM": "egl", "PYTHONUNBUFFERED": "1"},
        "outputs": [f"{w}/hands/", f"{w}/review/"],
        "verify": [f"{w}/hands/detections.npz"] + ([f"{w}/hands/mano_hamer.npz"] if hamer_only else
                                                   [f"{w}/hands/fingers.json"] if analysis_only else []),
        "notes": ["bundle holds the user's MANO_RIGHT.pkl (plain-numpy copy): SECURE cloud only, never public",
                  "weights downloaded on the pod: RTMW-x / RTMPose-m hand (download.openmmlab.com), MediaPipe hand "
                  "landmarker (storage.googleapis.com), WiLoR (huggingface.co/spaces/rolpotamias/WiLoR), HaMeR "
                  "(cs.utexas.edu/~pavlakos/hamer)"],
    }


def build_minutes(frames: int) -> int:
    """Pod minutes for the build: larissa_kadu (176 grid frames) took 7.6 min on an A40 before the 2026-10-07 review
    fixes; the exact contact geometry (GPU winding numbers + closest points every 20 iterations, the guard's finger
    crossing / arm penetration checks, the contact-aware wrist gate, metrics over three stages) brings the estimate to
    ~35 min with the guard's retries. Everything scales with the frames: 2x margin + 15 min, capped to the session
    maximum."""
    return int(min(MAX_MINUTES - 20, 15 + 2 * 35 * frames / 176))


def build_spec(take: str, loo: bool = True) -> dict:
    """Pod job for `refine_hands build <take> --no-review`: CPU-heavy IK (fingers + 32 held-out fits, contact episodes
    with the arm guard's retries) and the metrics; the review images are drawn locally afterwards (frames stay home).
    Inputs are the evidence files of the earlier hand session (hands/), the fit, cameras (+ frames.json only), dancer
    masks, MAMMA's grid landmarks and the face expression fits (for smplx_detail)."""
    from .refine_hands import fit_cams
    work = PACKAGE_ROOT / "work" / take
    w = f"dancecap/work/{take}"
    cams = fit_cams(work)
    need = [work / "hands" / f for f in ("grid_obs.npz", "jobs.json", "detections.npz")]
    missing = [str(p) for p in need if not p.exists()]
    if missing:
        raise SystemExit(f"build needs the hand evidence of `hands_pod run {take}` first; missing {missing}")
    sel = work / "mamma" / "select.json"
    tag = json.loads(sel.read_text()).get("chosen") if sel.exists() else None
    md_ = env.mamma_data() / "body_models"
    inputs = [f"{w}/smplx/lead.npz", f"{w}/smplx/follow.npz", f"{w}/sync/offsets.json",
              f"{w}/cameras/world_from_sfm.json", *[f"{w}/cameras/{c}.npz" for c in cams],
              *[f"{w}/frames/{c}/frames.json" for c in cams], *[f"{w}/dancers/{c}/masks/" for c in cams],
              f"{w}/hands/",
              {"local": str(md_ / "smplx_locked_head" / "smplx" / "SMPLX_NEUTRAL.npz"),
               "pod": "mamma/data/body_models/smplx_locked_head/smplx/SMPLX_NEUTRAL.npz"},
              {"local": str(md_ / "downsampled_verts" / "verts_512.pkl"),
               "pod": "mamma/data/body_models/downsampled_verts/verts_512.pkl"},
              {"local": str(md_ / "downsampled_verts" / "smplx_512_body_parts.npz"),
               "pod": "mamma/data/body_models/downsampled_verts/smplx_512_body_parts.npz"}]
    if sel.exists():
        inputs.append(f"{w}/mamma/select.json")
        if tag and (work / "mamma" / "ma_2d" / f"{take}_{tag}").exists():
            inputs.append(f"{w}/mamma/ma_2d/{take}_{tag}/")
    for r in ("lead", "follow"):
        if (work / "face" / f"expr_{r}.npz").exists():
            inputs.append(f"{w}/face/expr_{r}.npz")
        if (work / "smplx_face" / f"{r}.npz").exists():           # hands_detail merges this face block (2026-10-07)
            inputs.append(f"{w}/smplx_face/{r}.npz")
    import numpy as np
    with np.load(work / "smplx" / "lead.npz", allow_pickle=True) as z:
        T = len(z["t"])
    work_min = build_minutes(T)
    # threads: the container's CPU quota (cgroup v1 or v2), at most 8 per process, passive OpenMP waits (the
    # refinement runs many small tensor ops; spinning idle threads ate the quota, 2026-10-07)
    cmd = ("N=$(bash remote/hands_setup.sh ncpu); W=$(( N > 13 ? 12 : (N > 3 ? N - 1 : 2) )); "
           "TH=$(( N > 8 ? 8 : N )); export OMP_NUM_THREADS=$TH MKL_NUM_THREADS=$TH OMP_WAIT_POLICY=PASSIVE; "
           f"python -m dancecap.refine_hands build {take} --workers $W --no-review" + ("" if loo else " --no-loo"))
    return {
        "schema": "dancecap.pod_job/1", "name": "hands_build", "take": take, "max_minutes": work_min,
        "needs": [], "inputs": inputs,
        "commands": [{"name": "build", "cmd": cmd, "timeout_min": work_min}],
        # only the evidence goes up: earlier sessions' logs / bundles / fits are rebuilt on the pod
        "exclude": [f"{w}/hands/logs/*", f"{w}/hands/pod_job*.json", f"{w}/hands/smplx/*",
                    f"{w}/hands/smplx_fingers*/*", f"{w}/hands/fingers*", f"{w}/hands/contact*",
                    f"{w}/hands/metrics*", f"{w}/hands/build.json", f"{w}/hands/wrist_gate*"],
        # DANCECAP_POD / _CONTACT_DEVICE: the exact contact geometry (mesh_contact) runs brute force on the pod's GPU
        "env": {"PYTHONUNBUFFERED": "1", "DANCECAP_POD": "1", "DANCECAP_CONTACT_DEVICE": "cuda"},
        "outputs": [f"{w}/hands/", f"{w}/smplx_hands/", f"{w}/smplx_detail/", f"{w}/review/"],
        "verify": [f"{w}/smplx_hands/lead.npz", f"{w}/smplx_hands/follow.npz", f"{w}/smplx_detail/lead.npz",
                   f"{w}/smplx_detail/follow.npz", f"{w}/hands/metrics.json", f"{w}/hands/build.json"],
        "notes": ["no MANO / FLAME model file in this bundle (hand evidence = derived outputs); SECURE for reliability",
                  "afterwards locally: python -m dancecap.refine_hands review " + take + " --video-only"],
    }


def write_spec(take: str, build: bool = False, **kw) -> Path:
    if build:
        d = build_spec(take, loo=kw.get("loo", True))
        p = PACKAGE_ROOT / "work" / take / "hands" / "pod_job_build.json"
        p.write_text(json.dumps(d, indent=1))
        return p
    d = spec(take, **kw)
    p = PACKAGE_ROOT / "work" / take / "hands" / "pod_job.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, indent=1))
    return p


def run(take: str, max_cost: float = 3.0, hamer: bool = True, contact: bool = True, dry_run: bool = False,
        sid: str | None = None, force: bool = False, analysis_only: bool = False, hamer_only: bool = False,
        build: bool = False) -> dict:
    from . import remote_jobs as rj
    if build:
        p = write_spec(take, build=True)
        job = rj.load_job(p)
        mm = min(MAX_MINUTES, rj.estimate_minutes([job], 0)["bootstrap"] + job.max_minutes + _upload_minutes(job))
        # SECURE although the bundle holds no MANO file: a COMMUNITY L40S host failed CUDA init in the bootstrap
        # check (2026-10-06, $0.38 lost), the SECURE A40s of the earlier hand sessions were reliable
        return rj.run_session([job], sid=sid, label=f"hands-build-{take}", max_cost=max_cost, gpus=GPUS,
                              clouds=CLOUDS, dry_run=dry_run, force=force, max_minutes=mm)
    if not MANO_PLAIN.exists():
        from .hand_mano_worker import convert_mano
        convert_mano()
    p = write_spec(take, hamer=hamer, contact=contact, analysis_only=analysis_only, hamer_only=hamer_only)
    job = rj.load_job(p)
    # pod time: setup/bootstrap + the commands' realistic run time, not the sum of their timeouts
    work_min = 30 if hamer_only else 75 if analysis_only else 105
    mm = min(MAX_MINUTES, max(60.0, rj.estimate_minutes([job], 0)["bootstrap"] + work_min + (25 if hamer else 0)))
    return rj.run_session([job], sid=sid, label=f"hands-{take}", max_cost=max_cost, gpus=GPUS, clouds=CLOUDS,
                          dry_run=dry_run, force=force, max_minutes=mm + _upload_minutes(job))


def _upload_minutes(job) -> float:
    from . import remote_jobs as rj
    files = rj.bundle_files([job])
    return round(sum(f.size for f in files) / 1e6 / rj.UPLOAD_MBPS / 60, 1)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["spec", "run"])
    ap.add_argument("take")
    ap.add_argument("--max-cost", type=float, default=3.0)
    ap.add_argument("--no-hamer", action="store_true")
    ap.add_argument("--no-contact", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--session")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--analysis-only", action="store_true", help="detections/WiLoR exist: fusion, IK, contact (+HaMeR)")
    ap.add_argument("--hamer-only", action="store_true", help="add HaMeR to an analysed take (+ detectors/mano/compare)")
    ap.add_argument("--build", action="store_true", help="the detail build (refine_hands build) on existing evidence")
    a = ap.parse_args(argv)
    if a.cmd == "spec":
        p = write_spec(a.take, build=a.build, hamer=not a.no_hamer, contact=not a.no_contact,
                       analysis_only=a.analysis_only, hamer_only=a.hamer_only)
        from . import remote_jobs as rj
        job = rj.load_job(p)
        files = rj.bundle_files([job])
        print(f"{p}: {len(job.commands)} commands, {len(files)} bundle files, "
              f"{sum(f.size for f in files) / 1e6:.0f} MB raw (upload ~{_upload_minutes(job):.0f} min)")
        return 0
    res = run(a.take, a.max_cost, not a.no_hamer, not a.no_contact, a.dry_run, a.session, a.force, a.analysis_only,
              a.hamer_only, a.build)
    print(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
