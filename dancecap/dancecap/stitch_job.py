"""The stitch as a Runpod job: the farm 'stitch' step template, a job-spec builder and the pod-side chunk staging.

All analysis runs on Runpod (FARM.md section 0.10), never on the 4-CPU workstation. `dancecap.stitch` is numpy + scipy only
and headless, so a CPU pod runs it from the volume's environment; this module is the ready-made job around it. It imports
nothing from dancecap.farm (those files belong to the farm workflow) and is stdlib + dancecap.stitch (lazily, for the
take's chunk list). The farm can adopt the template without editing this file:

    from dancecap import stitch_job
    stitch_job.register(chunks.STEP_TEMPLATES)        # adds "stitch" and "stitch_check" (never overwrites an existing key)
    job = stitch_job.build_job("larissa_kadu_full", stitch_job.chunk_list_from_take("larissa_kadu_full"), world_id=...)
    stitch_job.write_job(job, chunk_list)             # farm/jobs/<id>.json + the chunk-list input file

Command line (workstation, orchestration only):

    python -m dancecap.stitch_job build larissa_kadu_full [--mode stitch|check] [--chunks chunks.json] [--partial]
            [--world-id W --pose-id P --world-sha256 H] [--tar a=chunks/a/out/result.tar ...] [--job-id ID]
        -> work/<dance>/farm/jobs/<id>.json  +  work/<dance>/farm/stitch/<id>.chunks.json (the job's input file)
    python -m dancecap.stitch_job stage --chunks FILE [--volume /workspace/hm] [--dest DIR]       # pod side

The job (schema dancecap.farm.job/1, kind `analyse`, needs_gpu false, cwd dancecap/): steps run in order from the pod
workspace's dancecap/ folder with DANCECAP_WORK_ROOT = the workspace's dancecap/work:

  stage        (only when the chunk list names result tars) extract chunks/<id>/out/result.tar of the volume into the
               workspace, sha256-checked: the fits and refined folders then sit at the workspace-relative paths the chunk
               list names. Chunks that already sit on the volume or in the workspace are read in place.
  stitch       python -m dancecap.stitch <dance> --chunks <list> --work-root work --out <dance> --gate-exit report
               gates + bit identity + verification of the written files + both review sheets; a failing seam is a RESULT
               (stitch.json: which gate, the centred 12 s refit window; the motion goes to stitch_rejected/), not a job
               failure, so the watcher pulls it. Only real errors (missing fit, world mismatch, ...) fail the step.
  stitch_check (mode check) the same analysis with --check: stitch_check.json + review sheets, no motion written (verify
               a candidate chunk list, or a partial one with --partial, at full length without touching smplx/).

Outputs (workspace-relative, packed into out/result.tar by the worker): dancecap/work/<dance>/{stitch.json, smplx/,
<refined>/, review/, stitch_rejected/, stitch_partial/} (mode check: dancecap/work/<dance>_check/{stitch_check.json,
review/}). A stitch of the 188 s dance is about 1-2 minutes on 4 vCPU and under 4 GB RAM: any CPU pod, no GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import posixpath
import subprocess
import tarfile
import time
from pathlib import Path

SCHEMA = "dancecap.farm.job/1"
WORK_WS = "dancecap/work"            # the work root, workspace-relative (pod: /opt/hm/ws/dancecap/work)
POD_ENV = {"DANCECAP_POD": "1", "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
           "MPLBACKEND": "Agg"}
STITCH_FLAGS = "--gate-exit report"


def _step(name: str, cmd: str, timeout_min, optional: bool = False, env: dict | None = None) -> dict:
    """The same shape as dancecap.farm.chunks._step."""
    d = {"name": name, "cmd": cmd, "timeout_min": timeout_min if isinstance(timeout_min, str) else round(float(timeout_min), 1),
         "optional": optional}
    if env:
        d["env"] = env
    return d


# THE STEP TEMPLATES (farm shape: {name, cmd, timeout_min, optional}). {py} {dance} {chunks} {work} {out} {flags} are filled
# by build_job(); {py} is `python` on the pod (the worker puts the volume's venv first on PATH).
STEP_TEMPLATES: dict[str, list[dict]] = {
    "stitch": [
        _step("stage", "{py} -m dancecap.stitch_job stage --chunks {work}/{chunks} --work-root {work}", 20),
        _step("stitch", "{py} -m dancecap.stitch {dance} --chunks {work}/{chunks} --work-root {work} --out {out} {flags}",
              "{stitch_min}"),
    ],
    "stitch_check": [
        _step("stage", "{py} -m dancecap.stitch_job stage --chunks {work}/{chunks} --work-root {work}", 20),
        _step("stitch_check", "{py} -m dancecap.stitch {dance} --chunks {work}/{chunks} --work-root {work} --out {out} "
                              "--check {flags}", "{stitch_min}"),
    ],
}


def register(templates: dict) -> dict:
    """Add the stitch templates to a STEP_TEMPLATES dict (the farm's), never replacing an existing key. -> templates."""
    for k, v in STEP_TEMPLATES.items():
        templates.setdefault(k, [dict(s) for s in v])
    return templates


class _Fmt(dict):
    def __missing__(self, key):
        return "{" + key + "}"


def fmt(template: str, **ctx) -> str:
    return template.format_map(_Fmt(**ctx))


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def code_info() -> dict:
    """{git_sha, branch, dirty, bundle_sha256: None}: the code the job was written for (as farm.chunks.code_info)."""
    here = Path(__file__).resolve().parents[2]

    def git(*a):
        try:
            return subprocess.run(["git", *a], cwd=here, capture_output=True, text=True, timeout=60).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""
    return {"git_sha": git("rev-parse", "HEAD") or None, "branch": git("rev-parse", "--abbrev-ref", "HEAD") or None,
            "dirty": bool(git("status", "--porcelain")), "bundle_sha256": None}


def chunk_list_from_take(dance: str, tag: str | None = None) -> dict:
    """The chunk list (the stitch's --chunks JSON) of a take: its [segments.*] fits + the reused [reuse] take, paths relative
    to the work root. Needs only the take TOML (+ prepare.json / select.json where they exist)."""
    from . import stitch
    return stitch.chunk_list(stitch.spec_from_take(dance, tag))


def build_job(dance: str, chunk_list: dict, *, mode: str = "stitch", job_id: str | None = None, world_id: str | None = None,
              world_sha256: str | None = None, pose_id: str | None = None, partial: bool = False,
              tars: dict | None = None, py: str = "python", max_minutes: float = 60.0, priority: float = 1000.0,
              code: dict | None = None, extra_flags: str = "", work_ws: str = WORK_WS) -> dict:
    """The job spec of one stitch (or one stitch check) of `chunk_list`.

    tars: {chunk name: {"path": "chunks/<id>/out/result.tar" (volume-relative), "sha256": optional}} - chunk results the
    `stage` step extracts into the workspace before stitching (written into the job's copy of the chunk list).
    The chunk list is embedded (`stitch.chunk_list`) and is the job's one input file (inputs[0]); write_job() writes it."""
    if mode not in ("stitch", "check"):
        raise ValueError(f"mode {mode!r} not in ('stitch', 'check')")
    cl = json.loads(json.dumps(chunk_list))
    cl["take"] = dance
    if world_id:
        cl.setdefault("world_id", world_id)
    if pose_id:
        cl.setdefault("pose_id", pose_id)
    for c in cl["chunks"]:
        if tars and c["name"] in tars:
            c["tar"] = tars[c["name"]]
    ks = [c["k"] for c in cl["chunks"] if c.get("k")]
    k0, k1 = (min(k[0] for k in ks), max(k[1] for k in ks)) if ks else (None, None)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    jid = job_id or f"stitch-{dance}-{stamp}" + ("-check" if mode == "check" else "")
    work_cwd = posixpath.relpath(work_ws, "dancecap")            # the work root as the steps (cwd dancecap/) see it
    chunks_rel = f"{dance}/farm/stitch/{jid}.chunks.json"        # under the work root
    text = json.dumps(cl, indent=1)
    out_rel = dance if mode == "stitch" else f"{dance}_check"
    ctx = {"py": py, "dance": dance, "chunks": chunks_rel, "work": work_cwd, "out": out_rel,
           "flags": " ".join(x for x in (STITCH_FLAGS, "--partial" if partial else "", extra_flags) if x),
           "stitch_min": max(10, int(max_minutes) - 25)}
    template = STEP_TEMPLATES["stitch" if mode == "stitch" else "stitch_check"]
    steps = []
    for t in template:
        s = dict(t)
        if s["name"] == "stage" and not any(c.get("tar") for c in cl["chunks"]):
            continue                                              # nothing to stage: everything is read in place
        s["cmd"] = fmt(t["cmd"], **ctx)
        tm = t["timeout_min"]
        s["timeout_min"] = round(float(fmt(tm, **ctx)) if isinstance(tm, str) else float(tm), 1)
        s["stage"] = "stitch"
        s["env"] = dict(POD_ENV, **(t.get("env") or {}))
        steps.append(s)
    out_ws = f"{work_ws}/{out_rel}"
    if mode == "stitch":
        outputs = [{"path": f"{out_ws}/stitch.json"}, {"path": f"{out_ws}/smplx/", "optional": True},
                   {"path": f"{out_ws}/{cl.get('out_refined') or 'smplx_rf2_noslip'}/", "optional": True},
                   {"path": f"{out_ws}/review/", "optional": True}, {"path": f"{out_ws}/stitch_rejected/", "optional": True},
                   {"path": f"{out_ws}/stitch_partial/", "optional": True}]
    else:
        outputs = [{"path": f"{out_ws}/stitch_check.json"}, {"path": f"{out_ws}/review/", "optional": True}]
    return {
        "schema": SCHEMA, "id": jid, "dance": dance, "kind": "analyse", "stage": "stitch", "mode": mode, "take": dance,
        "chunk": {"id": jid, "kind": "analyse", "k0": k0, "k1": k1, "core_k0": k0, "core_k1": k1, "cams": [],
                  "world_id": world_id, "status": "planned"},
        "cams": [], "world_id": world_id, "world_sha256": world_sha256, "pose_id": pose_id,
        "code": code if code is not None else code_info(), "sources": {},
        "inputs": [{"path": f"{work_ws}/{chunks_rel}", "sha256": _sha256_bytes(text.encode("utf-8")),
                    "bytes": len(text.encode("utf-8")), "kind": "file"}],
        "steps": steps, "outputs": outputs, "max_minutes": float(max_minutes), "priority": float(priority),
        "needs_gpu": False, "cwd": "dancecap", "retries": 1,
        "env": {"DANCECAP_WORK_ROOT": "{WS}/" + work_ws},
        "resources": {"cpu_only": True, "min_vcpu": 4, "min_ram_gb": 16, "est_minutes": 3},
        "estimate": {"total_min": 3.0, "note": "numpy/scipy only; ~1-2 min on 4 vCPU for the 188 s dance, plus staging"},
        "stitch": {"chunk_list_path": f"{work_ws}/{chunks_rel}", "chunk_list_sha256": _sha256_bytes(text.encode("utf-8")),
                   "chunks": [c["name"] for c in cl["chunks"]], "partial": bool(partial)},
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "_chunk_list_text": text}


def write_job(job: dict, local_root: str | Path | None = None, jobs_dir: str | Path | None = None) -> tuple[Path, Path]:
    """Write the job spec (farm/jobs/<id>.json) and its input file (the chunk list at the workspace-relative path the spec
    names, under local_root = the workspace root, default the repo root). -> (spec path, chunk-list path)."""
    root = Path(local_root) if local_root else Path(__file__).resolve().parents[2]
    spec = {k: v for k, v in job.items() if k != "_chunk_list_text"}
    cl_path = root / job["stitch"]["chunk_list_path"]
    cl_path.parent.mkdir(parents=True, exist_ok=True)
    cl_path.write_bytes(job["_chunk_list_text"].encode("utf-8"))      # bytes: the sha256 in the spec is of these bytes
    jd = Path(jobs_dir) if jobs_dir else root / WORK_WS / job["dance"] / "farm" / "jobs"
    jd.mkdir(parents=True, exist_ok=True)
    sp = jd / f"{job['id']}.json"
    tmp = sp.with_suffix(".json.part")
    tmp.write_text(json.dumps(spec, indent=1), encoding="utf-8")
    os.replace(tmp, sp)
    return sp, cl_path


# ------------------------------------------------------------------------------------------------ pod side


def _safe_extract(tar: tarfile.TarFile, dest: Path) -> int:
    base = dest.resolve()
    n = 0
    for m in tar.getmembers():
        t = (dest / m.name).resolve()
        if base != t and base not in t.parents:
            raise ValueError(f"unsafe path in tar: {m.name}")
        if m.issym() or m.islnk():
            raise ValueError(f"link in tar: {m.name}")
        n += 1
    tar.extractall(dest)
    return n


def stage(chunks_json: str | Path, volume: str | Path | None = None, dest: str | Path | None = None,
          work_root: str | Path | None = None) -> dict:
    """Pod side: extract every `tar` of the chunk list (a volume-relative result.tar, sha256-checked when named) into the
    workspace (dest; default $HM_WS, else the parent of the current dancecap/ folder). Idempotent. -> report."""
    cl = json.loads(Path(chunks_json).read_text(encoding="utf-8"))
    vol = Path(volume or os.environ.get("HM_ROOT") or "/workspace/hm")
    ws = Path(dest or os.environ.get("HM_WS") or Path.cwd().parent)
    rep = {"volume": str(vol), "dest": str(ws), "tars": []}
    for c in cl.get("chunks", []):
        t = c.get("tar")
        if not t:
            continue
        spec = {"path": t} if isinstance(t, str) else dict(t)
        src = Path(spec["path"]) if Path(spec["path"]).is_absolute() else vol / spec["path"]
        if not src.is_file():
            raise FileNotFoundError(f"chunk {c['name']}: {src} is not on the volume")
        if spec.get("sha256"):
            h = hashlib.sha256()
            with open(src, "rb") as f:
                for b in iter(lambda: f.read(1 << 22), b""):
                    h.update(b)
            if h.hexdigest() != spec["sha256"]:
                raise ValueError(f"chunk {c['name']}: {src} sha256 {h.hexdigest()} != the job's {spec['sha256']}")
        with tarfile.open(src, "r:*") as tar:
            n = _safe_extract(tar, ws)
        rep["tars"].append({"chunk": c["name"], "tar": str(src), "files": n})
        print(f"[stitch_job] staged {c['name']}: {n} files from {src}", flush=True)
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.stitch_job", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="write the job spec + its chunk-list input file")
    b.add_argument("dance")
    b.add_argument("--chunks", help="chunk list JSON (default: the take's [segments]/[reuse] via dancecap.stitch)")
    b.add_argument("--mode", choices=("stitch", "check"), default="stitch")
    b.add_argument("--partial", action="store_true")
    b.add_argument("--world-id")
    b.add_argument("--pose-id")
    b.add_argument("--world-sha256")
    b.add_argument("--tar", action="append", default=[], metavar="CHUNK=PATH",
                   help="a chunk whose result.tar the pod extracts first (volume-relative path)")
    b.add_argument("--job-id")
    b.add_argument("--max-minutes", type=float, default=60.0)
    b.add_argument("--tag")
    s = sub.add_parser("stage", help="pod side: extract the chunk results named in the chunk list")
    s.add_argument("--chunks", required=True)
    s.add_argument("--volume")
    s.add_argument("--dest")
    s.add_argument("--work-root", help="(accepted for symmetry with dancecap.stitch; staging goes to --dest / $HM_WS)")
    a = ap.parse_args(argv)
    if a.cmd == "stage":
        stage(a.chunks, a.volume, a.dest, a.work_root)
        return 0
    cl = json.loads(Path(a.chunks).read_text(encoding="utf-8")) if a.chunks else chunk_list_from_take(a.dance, a.tag)
    tars = {}
    for t in a.tar:
        name, _, path = t.partition("=")
        tars[name] = {"path": path}
    job = build_job(a.dance, cl, mode=a.mode, job_id=a.job_id, world_id=a.world_id, world_sha256=a.world_sha256,
                    pose_id=a.pose_id, partial=a.partial, tars=tars, max_minutes=a.max_minutes)
    sp, cp = write_job(job)
    print(f"[stitch_job] {job['id']}: {len(job['steps'])} steps -> {sp}\n[stitch_job] input {cp}")
    for st in job["steps"]:
        print(f"[stitch_job]   {st['name']}: {st['cmd']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
