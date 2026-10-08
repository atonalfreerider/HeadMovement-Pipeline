"""The dancer-splat pod job (dancecap.pod_job/1, run by dancecap.remote_jobs: one bundle upload, bootstrap base +
gsplat, resumable commands with .done markers, outputs packed / downloaded / verified, the pod always deleted).

    python -m dancecap.dancer_splats <take> write-pod-job     -> work/<take>/dancer_splats/pod_job.json
    python -m dancecap remote run --take <take> --jobs dancer_splats --max-cost 1.5 --max-minutes 55

Budget (user approval 2026-10-08): one session capped at $1.50 (A100 SXM community $1.39/h x (55 + 6 start) min =
$1.41 worst case), at most $2 including one retry: a retry runs with --retry (uploads what the failed session brought
back: dancer_splats/cache/ckpt.pt is packed even on failure, model.pt + pose_delta.npz when only eval failed - train
then resumes from the checkpoint or is skipped) and --max-cost = 2.00 - the first session's ledger cost
(retry_budget()). Training stops itself inside --time-budget-min (the schedule
shrinks), so the kill switch never takes the outputs. The upload is the bundle only (crops, labels, bodies, init
samples, groom strands, avatar reference renders) + the dancecap code + the cached gsplat wheel and the base PyPI
wheels; MAMMA, the body models and the mamma/dancers/cameras wheels are excluded.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from ..take import PACKAGE_ROOT

JOB_NAME = "dancer_splats"
MAX_COST = 1.50
SESSION_MAX_MINUTES = 55
TRAIN_BUDGET_MIN = 24.0
TRAIN_TIMEOUT_MIN = 30
EVAL_TIMEOUT_MIN = 10
# bundle exclusions: code and wheels this job never uses (patterns: remote_jobs._excluded)
EXCLUDE = ["mamma-turing", "pytorch_sdf-*.whl", "opencv_python-4*.whl", "polars*.whl", "imageio_ffmpeg-*.whl",
           "pandas-*.whl", "scikit_learn-*.whl", "plotly-*.whl", "kornia*.whl", "ultralytics-*.whl",
           "pyrender-*.whl", "pyglet-*.whl", "PyOpenGL-*", "scenepic-*.whl", "freetype_py-*.whl", "narwhals-*.whl",
           "networkx-*.whl", "tzdata-*.whl"]


APPROVED_TOTAL = 2.00  # $ for the prototype including one retry (user, 2026-10-08)


def retry_inputs(take: str) -> list[str]:
    """What a failed session brought back that lets a retry continue instead of starting over (workspace-relative):
    model.pt + pose_delta.npz (+ report.json) when training finished, else the trainer's checkpoint."""
    loc = PACKAGE_ROOT / "work" / take / "dancer_splats"
    d = f"dancecap/work/{take}/dancer_splats"
    if (loc / "model.pt").exists() and (loc / "pose_delta.npz").exists():
        return [f"{d}/{f}" for f in ("model.pt", "pose_delta.npz", "report.json") if (loc / f).exists()]
    if (loc / "cache" / "ckpt.pt").exists():
        return [f"{d}/cache/ckpt.pt"]
    return []


def retry_budget(sids: list[str] | None = None) -> tuple[float, float]:
    """(max_cost $, max_minutes) left for a retry inside APPROVED_TOTAL: the ledger cost of this job's earlier
    sessions (pods named dancecap-splats-*) is subtracted; minutes = what that buys at $1.39/h minus 6 min start-up."""
    from .. import remote
    used = 0.0
    for r in remote.ledger():
        if not str(r.get("name", "")).startswith("splats-") or (sids and r.get("session") not in sids):
            continue
        end = r.get("deleted_ts") or time.time()
        run_s = max(0.0, end - float(r.get("created_ts") or end))
        used += float(r["cost"]) if r.get("cost") is not None else float(r.get("cost_per_hr") or 0.0) * run_s / 3600
    left = max(0.0, APPROVED_TOTAL - used)
    return round(min(MAX_COST, left), 2), max(0.0, min(SESSION_MAX_MINUTES, left / 1.39 * 60 - 6))


def spec(take: str, preset: str = "pod", budget_min: float = TRAIN_BUDGET_MIN, retry: bool = False) -> dict:
    w = f"dancecap/work/{take}"
    d = f"{w}/dancer_splats"
    bundle = PACKAGE_ROOT / "work" / take / "dancer_splats" / "bundle"
    if not (bundle / "manifest.json").exists():
        raise SystemExit(f"dancer_splats: no bundle at {bundle}: run `python -m dancecap.dancer_splats {take} prepare`")
    extra_in = retry_inputs(take) if retry else []
    cmds = [{"name": "train", "timeout_min": TRAIN_TIMEOUT_MIN,
             "cmd": f"python -m dancecap.dancer_splats {take} train --preset {preset} --time-budget-min {budget_min:g}"},
            {"name": "eval", "timeout_min": EVAL_TIMEOUT_MIN,
             "cmd": f"python -m dancecap.dancer_splats {take} eval --preset {preset}"}]
    return {
        "schema": "dancecap.pod_job/1", "name": JOB_NAME, "take": take,
        "max_minutes": TRAIN_TIMEOUT_MIN + EVAL_TIMEOUT_MIN, "needs": ["gsplat"],
        "inputs": [f"{d}/bundle/", *extra_in],
        "setup": ["pip install -q --no-deps lpips==0.1.4 || echo 'lpips unavailable: LPIPS is skipped'"],
        "commands": cmds,
        "exclude": EXCLUDE,
        "env": {"PYTHONUNBUFFERED": "1"},
        "outputs": [f"{d}/model.pt", f"{d}/pose_delta.npz", f"{d}/metrics.json", f"{d}/report.json",
                    f"{d}/progress.json", f"{d}/train.log", f"{w}/review/splats/",
                    f"{d}/cache/"],  # the checkpoint (gone after a finished train): a retry resumes from it
        "verify": [f"{d}/model.pt", f"{d}/pose_delta.npz", f"{d}/metrics.json"],
        "after_download": [f"python -m dancecap.dancer_splats {take} export",
                           f"python -m dancecap.dancer_splats {take} review"],
        "max_cost_usd": MAX_COST, "session_max_minutes": SESSION_MAX_MINUTES,
        "run": f"python -m dancecap remote run --take {take} --jobs {JOB_NAME} --max-cost {MAX_COST:.2f} "
               f"--max-minutes {SESSION_MAX_MINUTES}",
        "notes": ["dancers only: every Gaussian is bound to a SMPL-X triangle or a groom strand (no room / floor / "
                  "spectators)", "camera 03 + every 10th instant held out; metrics.json gates vs the textured avatar",
                  "per-frame pose corrections -> dancer_splats/pose_delta.npz (smplx/ is never written)",
                  "train stops inside --time-budget-min; eval renders review/splats/*.jpg on the pod",
                  "LPIPS (lpips 0.1.4, alexnet weights from download.pytorch.org) is optional",
                  "train stops hard at its budget / on SIGTERM and still writes model.pt + pose_delta.npz",
                  *(["retry: uploads " + ", ".join(Path(x).name for x in extra_in)] if extra_in else []),
                  "likeness data: work/ only; nothing goes to the head-movement repo"],
    }


def write_pod_job(take: str, **kw) -> Path:
    p = PACKAGE_ROOT / "work" / take / "dancer_splats" / "pod_job.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(spec(take, **kw), indent=1))
    return p


def uploads_in_flight(exclude_sid: str | None = None) -> list[str]:
    """Sessions whose runner is alive and whose upload has started but not finished (the home link is shared: a
    second upload halves both). Read-only: runner.json pid + runner.log."""
    from ..remote_jobs import SESSIONS
    busy = []
    if not SESSIONS.exists():
        return busy
    for d in sorted(SESSIONS.iterdir(), reverse=True)[:12]:
        if d.name == exclude_sid or not (d / "runner.log").exists():
            continue
        try:
            info = json.loads((d / "runner.json").read_text()) if (d / "runner.json").exists() else {}
            pid = int(info.get("pid") or 0)
            if not pid or not _alive(pid):
                continue
            text = (d / "runner.log").read_text(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            continue
        starts = [m.start() for m in re.finditer(r"\] upload \.\.\.", text)]
        if starts and not re.search(r"\] upload (done|FAILED)", text[starts[-1]:]):
            busy.append(d.name)
    return busy


def _alive(pid: int) -> bool:
    try:
        import psutil
        return psutil.pid_exists(pid)
    except ImportError:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


def run(take: str, max_cost: float = MAX_COST, max_minutes: float = SESSION_MAX_MINUTES, dry_run: bool = False,
        sid: str | None = None, force: bool = False, wait_upload: bool = True, retry: bool = False,
        after: bool = True, log=print) -> dict:
    """One pod session for this job alone (the pipeline launches the same through `remote run --jobs`). Waits while
    another session is uploading (shared home link). retry: upload what the last session brought back and keep the
    whole prototype inside APPROVED_TOTAL."""
    from .. import remote_jobs as rj
    if retry:
        left, mins = retry_budget()
        max_cost, max_minutes = min(max_cost, left), min(max_minutes, mins)
        log(f"[dancer_splats] retry: ${max_cost:.2f} / {max_minutes:.0f} min left of the approved "
            f"${APPROVED_TOTAL:.2f}; uploads {retry_inputs(take) or 'nothing extra'}")
        if max_minutes < 20:
            raise SystemExit("dancer_splats: not enough of the approved budget left for a useful retry")
    job = rj.parse_job(spec(take, retry=retry), f"dancer_splats:{take}")
    while wait_upload and not dry_run:
        busy = uploads_in_flight(sid)
        if not busy:
            break
        log(f"[dancer_splats] waiting: session(s) {busy} are uploading over the same link")
        time.sleep(60)
    res = rj.run_session([job], sid=sid, label=f"splats-{take}", max_cost=max_cost, dry_run=dry_run, force=force,
                         max_minutes=max_minutes)
    if after and not dry_run and (res.get("jobs", {}).get(JOB_NAME) or {}).get("status") == "ok":
        res["after_download"] = after_download(take, log)
    return res


def after_download(take: str, log=print) -> dict:
    """The job's local follow-ups (the runner does not run them): PLY export + the review sheets. Never raises."""
    import subprocess

    from ..env import cuda_wrapper  # gsplat's CUDA extension needs the MSVC + CUDA env on Windows
    out = {}
    for action in ("export", "review"):
        t0 = time.time()
        try:
            rc = subprocess.call(cuda_wrapper(["-m", "dancecap.dancer_splats", take, action]), cwd=str(PACKAGE_ROOT))
        except OSError as e:
            rc = f"{type(e).__name__}: {e}"
        out[action] = {"rc": rc, "s": round(time.time() - t0, 1)}
        log(f"[dancer_splats] after download: {action} rc={rc} in {out[action]['s']:.0f}s")
    return out
