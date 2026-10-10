"""Farm worker (runs ON a pod that mounts the hm-farm volume): claim a chunk job, run its steps, publish the result.

    python -m dancecap.farm.worker --volume /workspace/hm [--kinds fit,refine] [--max-jobs N] [--idle-exit-min 5]
                                   [--pod-id ID] [--poll-s 10]

Protocol (FARM.md section 5; paths are relative to --volume):

  queue/<id>.json                      job spec (schema dancecap.farm.job/1), written by `farm queue`
  chunks/<id>/claim.lock/owner.json    atomic `mkdir` = the claim. A claim whose heartbeat is older than --stale-s (a dead
                                       pod) is taken over by renaming the lock aside (rename is atomic: one taker wins)
  chunks/<id>/heartbeat                every 30 s: step, progress, GPU util / memory, load, disk, seconds to the pod's kill
  chunks/<id>/steps/<name>.json        per-step done marker (rc, seconds); chunks/<id>/ckpt/<name>.tar = the step's `save`
                                       paths, so a restarted worker (any pod) resumes after the last finished step
  chunks/<id>/in/<ws path>             the job's inputs, sha256-checked against the spec before they are used
  chunks/<id>/log/<step>.log           step output (appended live)
  chunks/<id>/out/result.tar           the declared outputs at their workspace-relative paths, + out/manifest.json with
                                       the sha256 of every member and of the tar (the local puller verifies both)
  chunks/<id>/done.json | failed.json  final state: status, steps, seconds, gpu, price, world_id, code, pod, cost

The volume is a network filesystem (~24 ms per small file created): the worker runs on the pod's LOCAL disk (workspace
/opt/hm/ws, scratch /opt/hm/work = dancecap/work) and touches the volume with a handful of large files only.

Job kinds: GPU jobs `front`, `fit`, `refine`, `full`, `asr` (narration, faster-whisper); CPU jobs `analyse` (prepare, stitch,
pose_quality, timing, physics, counterbalance, floorcraft, analysis_review, camera-gate evidence: see ANALYSIS below) which
run on cheap CPU pods that mount the same volume (pods.launch_cpu). A CPU pod only claims jobs that need no GPU, a GPU
pod claims everything its --kinds allows. `after` chains jobs: a job is claimable once every id in `after` has done.json
(stitch after the fit chunks, analysis after the stitch); the upstream outputs arrive as inputs of kind
{"path": "chunks/<upstream id>/out/result.tar", "archive": true} (extracted into this job's workspace).

Job spec (schema dancecap.farm.job/1):
  {"id", "dance", "kind": "front|fit|refine|full|analyse|asr", "chunk": {...}, "world_id", "world_sha256", "after": [ids],
   "code": {"git_sha", "dirty", "bundle_sha256"}, "inputs": [{"path", "sha256", "bytes", "archive"?}],
   "steps": [{"name", "cmd", "timeout_min", "optional"?, "save"?: [ws paths], "live"?: [ws dirs], "env"?: {}}],
   "outputs": [{"path", "optional"?}], "max_minutes", "priority" (higher first), "env"?, "cwd"? ("dancecap"),
   "retries"? (attempts, default 2), "needs_gpu"? (default true unless kind == analyse), "min_vram_gb"?}

Input resolution: `path` is the workspace-relative DESTINATION. The source is volume `<path>` when it starts with worlds/,
src/, weights/, models/ or chunks/ (shared, content-addressed or another job's output), else `chunks/<id>/in/<path>`; an
explicit "src" (volume-relative) overrides both. `archive: true` (or a .tar/.tar.gz name)
extracts the source into the workspace root instead of copying it.

Step environment: DANCECAP_POD=1, PYOPENGL_PLATFORM=egl, TORCH_HOME, PYTHONUNBUFFERED, YOLO_CONFIG_DIR, HM_ROOT, HM_CHUNK,
HM_WORLD_ID, HM_WORLD_DIR (the world package copied to the pod's local disk), DANCECAP_PROGRESS (jsonl read for the
heartbeat), plus the spec's `env`. Steps run in dancecap/ (the spec's `cwd`) with the volume's venv first on PATH.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import threading
import time
import uuid
from pathlib import Path

SCHEMA = "dancecap.farm.job/1"
HEARTBEAT_S = 30
JOB_LIMIT_FACTOR, JOB_LIMIT_SLACK_MIN = 1.25, 5.0   # a job is killed after max_minutes x 1.25 + 5 min of wall clock
SHARED_PREFIXES = ("worlds/", "src/", "weights/", "models/", "chunks/")
GPU_KINDS = {"front", "fit", "refine", "full", "asr"}   # everything else (analyse, stitch...) needs no GPU unless the spec says so


# ------------------------------------------------------------------------------------------------ small helpers


def now() -> float:
    return time.time()


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def step_fingerprint(spec: dict, step: dict) -> str:
    """What a finished step's marker (and its checkpoint) is valid FOR: the command and environment, the job's inputs (their sha256),
    the world and the pinned code. A marker whose fingerprint differs is ignored, so a job re-planned with a different geometry or world
    never resumes from the OLD step result (found by the survivability review: it did, and the new plan's output was the old one)."""
    code = spec.get("code") or {}
    doc = {"name": step.get("name"), "cmd": step.get("cmd"), "env": {**(spec.get("env") or {}), **(step.get("env") or {})},
           "save": step.get("save") or [], "inputs": sorted([i.get("path"), i.get("sha256"), i.get("bytes")] for i in spec.get("inputs", [])
                                                         if i.get("sha256") or i.get("bytes") is not None),
           "world": spec.get("world_sha256"), "bundle": code.get("bundle_sha256"), "outputs": [o.get("path") for o in spec.get("outputs", [])]}
    return hashlib.sha256(json.dumps(doc, sort_keys=True, default=str).encode()).hexdigest()[:20]


def job_fingerprint(spec: dict) -> str:
    """One hash over every step's fingerprint: equal for a re-queue of the same plan, different for a re-plan."""
    return hashlib.sha256("|".join(step_fingerprint(spec, s) for s in spec.get("steps", [])).encode()).hexdigest()[:20]


def write_json(path: Path, obj) -> None:
    """Atomic JSON write (temp + rename) so readers over the network never see a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def gpu_info() -> dict:
    """{name, mem_total_mb, mem_used_mb, util} of the first GPU, {} without one."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15).stdout.strip()
        name, total, used, util = [x.strip() for x in out.splitlines()[0].split(",")]
        return {"name": name, "mem_total_mb": int(total), "mem_used_mb": int(used), "util": int(util)}
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return {}


def last_progress(path: Path) -> dict:
    """Last informative dancecap.progress/1 event of a jsonl file (cheap tail read)."""
    try:
        size = path.stat().st_size
        with open(path, "rb") as fh:
            fh.seek(max(0, size - 8192))
            lines = fh.read().decode("utf-8", "replace").splitlines()
    except OSError:
        return {}
    for line in reversed(lines):
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("kind") in ("alive",):
            continue
        return {k: ev[k] for k in ("step", "kind", "done", "total", "unit", "msg", "ts", "lane") if k in ev}
    return {}


# ------------------------------------------------------------------------------------------------ the worker


class Worker:
    def __init__(self, volume: Path, *, pod_id: str | None = None, kinds: set[str] | None = None, ws: Path | None = None,
                 work: Path | None = None, state: Path | None = None, stale_s: float = 240.0, poll_s: float = 10.0,
                 gpu_required: bool | None = None):
        self.vol = Path(volume)
        self.pod_id = pod_id or os.environ.get("RUNPOD_POD_ID") or f"{socket.gethostname()}-{os.getpid()}"
        self.kinds = kinds
        local = Path(os.environ.get("HM_LOCAL", "/opt/hm"))
        self.ws = Path(ws or os.environ.get("HM_WS", local / "ws"))
        self.work = Path(work or os.environ.get("HM_WORK", local / "work"))
        self.state = Path(state or local / "state")
        self.stale_s = stale_s
        self.poll_s = poll_s
        self.gpu_required = gpu_required  # None: per job (needs_gpu / kind)
        self.price_hr = float(os.environ.get("HM_PRICE_HR") or 0)
        self.failed_here: set[str] = set()
        self.pod_dir = self.vol / "pods" / self.pod_id
        self.cur: dict = {}
        self.claims: dict[str, str] = {}      # job id -> the token written into claim.lock/owner.json by THIS worker
        self.lost_claim = False               # set by the heartbeat when the claim on the running job is no longer ours
        self.job_deadline: float | None = None
        self._claim_miss = 0
        self._hb_stop = threading.Event()
        self._hb_thread: threading.Thread | None = None
        self.gpu0 = gpu_info()

    # -- logging
    def log(self, msg: str) -> None:
        line = f"[worker {time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)

    def event(self, what: str) -> None:
        try:
            self.pod_dir.mkdir(parents=True, exist_ok=True)
            with open(self.pod_dir / "events.log", "a", encoding="utf-8") as fh:
                fh.write(f"{int(now())} {what}\n")
        except OSError:
            pass

    # -- queue
    def specs(self) -> list[dict]:
        out = []
        for p in sorted((self.vol / "queue").glob("*.json")):
            spec = read_json(p)
            if isinstance(spec, dict) and spec.get("id") and spec.get("schema", SCHEMA) == SCHEMA:
                out.append(spec)
        out.sort(key=lambda s: (-float(s.get("priority", 0)), str(s["id"])))
        return out

    def chunk_dir(self, jid: str) -> Path:
        return self.vol / "chunks" / jid

    def claimable(self, spec: dict) -> bool:
        jid = spec["id"]
        cd = self.chunk_dir(jid)
        if jid in self.failed_here or (cd / "done.json").exists() or (cd / "failed.json").exists():
            return False
        if needs_gpu(spec) and not self.gpu0 and self.gpu_required is not False:
            return False  # a CPU pod leaves GPU jobs to the GPU pods
        for dep in spec.get("after") or []:
            if not (self.vol / "chunks" / str(dep) / "done.json").exists():
                return False
        if self.kinds and spec.get("kind") not in self.kinds and not (
                spec.get("kind") == "full" and self.kinds & {"front", "fit", "refine"}):
            return False
        return True

    def kill_remaining_s(self) -> float | None:
        try:
            dl = int((self.pod_dir / "kill_at").read_text().strip())
            return dl - now()
        except (OSError, ValueError):
            return None

    def enough_time(self, spec: dict) -> bool:
        rem = self.kill_remaining_s()
        if rem is None:
            return True
        need = min(float(spec.get("max_minutes", 60)) * 0.5, 30) * 60
        return rem >= need

    def try_claim(self, spec: dict) -> bool:
        """Atomic `mkdir claim.lock`. A stale claim is taken over by renaming the lock aside; the rename happens first and the
        takeover is then VERIFIED against what was judged stale (a rename can move a lock that a faster taker has just made fresh: it is
        put back and this worker walks away)."""
        jid = spec["id"]
        cd = self.chunk_dir(jid)
        cd.mkdir(parents=True, exist_ok=True)
        lock = cd / "claim.lock"
        token = uuid.uuid4().hex
        try:
            lock.mkdir()
        except FileExistsError:
            seen = read_json(lock / "owner.json") or {}
            if not self._stale(cd):
                return False
            aside = cd / f"claim.stale.{int(now())}.{self.pod_id}.{token[:6]}"
            try:
                os.rename(lock, aside)  # atomic: of several takers exactly one succeeds
            except OSError:
                return False
            moved = read_json(aside / "owner.json") or {}
            if (moved.get("token"), moved.get("pod_id"), moved.get("t")) != (seen.get("token"), seen.get("pod_id"), seen.get("t")):
                with contextlib.suppress(OSError):   # we moved somebody's FRESH lock (another worker took over first): give it back
                    os.rename(aside, lock)
                return False
            try:
                lock.mkdir()
            except FileExistsError:
                return False
            self.log(f"took over the stale claim of {jid}")
        write_json(lock / "owner.json", {"pod_id": self.pod_id, "t": now(), "host": socket.gethostname(), "pid": os.getpid(), "token": token})
        self.claims[jid] = token
        return True

    def _stale(self, cd: Path) -> bool:
        hb = read_json(cd / "heartbeat")
        owner = read_json(cd / "claim.lock" / "owner.json") or {}
        if owner.get("pod_id") == self.pod_id and (owner.get("token") not in self.claims.values() or not self.cur):
            return True  # our own claim from a previous worker process on this pod
        last = max(float((hb or {}).get("t", 0)), float(owner.get("t", 0)))
        if not last:   # the lock dir exists but its owner.json is not written yet (the winner is between mkdir and write_json):
            try:       # that is a claim being made, not a dead one: age it by the directory itself (found by the drill / a threaded test)
                last = (cd / "claim.lock").stat().st_mtime
            except OSError:
                last = now()
        return now() - last > self.stale_s

    def owns_claim(self, jid: str) -> bool | None:
        """Is claim.lock still the one this worker made? None: cannot tell (unreadable)."""
        tok = self.claims.get(jid)
        lock = self.chunk_dir(jid) / "claim.lock"
        if not lock.is_dir():
            return False
        try:
            o = json.loads((lock / "owner.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return o.get("token") == tok and o.get("pod_id") == self.pod_id

    def release(self, jid: str) -> None:
        """Give the claim back - only if it is still OURS. A slow worker whose claim was taken over (a volume stall longer than the stale
        time) used to rmtree the NEW owner's live lock when it finished, and a third pod then claimed the running job. The lock is
        renamed to a token-named path first (atomic), looked at, and only deleted when it is ours; otherwise it is put back."""
        cd = self.chunk_dir(jid)
        tok = self.claims.pop(jid, None)
        lock = cd / "claim.lock"
        aside = cd / f"claim.released.{tok[:8] if tok else 'x'}.{int(now())}"
        try:
            os.rename(lock, aside)
        except OSError:
            return
        try:
            o = read_json(aside / "owner.json") or {}
            if tok and o.get("token") == tok:
                shutil.rmtree(aside, ignore_errors=True)
                return
            self.log(f"{jid}: the claim is no longer ours (owner {o.get('pod_id')}); left alone")
        except OSError:
            pass
        with contextlib.suppress(OSError):   # not ours: put it back for its owner
            os.rename(aside, lock)

    # -- heartbeat
    def _heartbeat_loop(self) -> None:
        while not self._hb_stop.wait(HEARTBEAT_S):
            self.beat()

    def beat(self, **extra) -> None:
        c = self.cur
        if not c:
            return
        try:
            prog = last_progress(Path(c["progress_file"])) if c.get("progress_file") else {}
            g = gpu_info()
            du = shutil.disk_usage(self.work if self.work.exists() else "/")
            rem = self.kill_remaining_s()
            hb = {"t": now(), "pod_id": self.pod_id, "id": c["id"], "step": c.get("step"), "step_i": c.get("step_i"),
                  "steps_n": c.get("steps_n"), "step_started": c.get("step_started"), "started": c.get("started"),
                  "attempt": c.get("attempt"), "progress": prog, "gpu": g, "disk_free_gb": round(du.free / 1e9, 1),
                  "load1": round(os.getloadavg()[0], 2) if hasattr(os, "getloadavg") else None,
                  "kill_in_s": round(rem) if rem is not None else None, **extra}
            own = self.owns_claim(c["id"]) if c["id"] in self.claims else True
            if own is False:   # taken over (a volume stall longer than the stale time): stop, do not race the new owner's output
                self._claim_miss += 1
                if self._claim_miss >= 2:
                    if not self.lost_claim:
                        self.log(f"{c['id']}: the claim is no longer ours - aborting the job")
                    self.lost_claim = True
                    return
            else:
                self._claim_miss = 0
            write_json(self.chunk_dir(c["id"]) / "heartbeat", hb)
            write_json(self.pod_dir / "heartbeat", {"t": now(), "state": "running", "job": c["id"], "step": c.get("step")})
        except Exception as e:  # noqa: BLE001 - a failing heartbeat must never kill the job
            self.log(f"heartbeat failed: {type(e).__name__}: {e}")

    # -- world
    def verify_world(self, spec: dict) -> dict:
        """Refuse a job whose world on the volume is not the one it names (FARM.md section 3). The manifest at
        worlds/[<dance>/]<world_id>/manifest.json must carry EXACTLY the job's world_sha256 (the full 64 hex digits, not a prefix), be
        self-consistent (sha256 over the sorted 'member NUL sha' lines of its members = its world_sha256), and EVERY file it lists
        must exist with its size; once per pod the files are re-hashed too. Returns {dir, id, manifest_sha256, members, files}."""
        wid = spec.get("world_id")
        if not wid:
            return {}
        cands = [self.vol / "worlds" / str(spec.get("dance", "")) / wid, self.vol / "worlds" / wid]
        wd = next((c for c in cands if (c / "manifest.json").exists()), None)
        if wd is None:
            raise JobError(f"world {wid} not on the volume (looked in {[str(c) for c in cands]})")
        man_p = wd / "manifest.json"
        man_sha = sha256_file(man_p)
        man = read_json(man_p, {}) or {}
        want = spec.get("world_sha256")
        have = man.get("world_sha256")
        if man.get("world_id") != wid:
            raise JobError(f"world {wid}: the manifest on the volume is world {man.get('world_id')}")
        members = man.get("members") or {}
        digest = hashlib.sha256("".join(f"{n}\0{members[n].get('sha256')}\n" for n in sorted(members)).encode()).hexdigest() if members else None
        if not have or digest != have:
            raise JobError(f"world {wid}: the manifest is not self-consistent (members digest {str(digest)[:12]} != world_sha256 {str(have)[:12]})")
        if want and have != want:
            raise JobError(f"world {wid}: sha256 on the volume {have[:12]} differs from the job's world_sha256 {str(want)[:12]}")
        files = [f for f in (man.get("files") or []) if isinstance(f, dict) and f.get("path")]
        if not files:
            raise JobError(f"world {wid}: the manifest lists no files")
        marker = self.state / f"world_{wid}.verified"
        if not (marker.exists() and marker.read_text() == man_sha):
            missing, bad = [], []
            for f in files:
                p = wd / f["path"]
                if not p.is_file():
                    missing.append(f["path"])
                elif f.get("bytes") is not None and p.stat().st_size != int(f["bytes"]):
                    bad.append(f["path"])
                elif f.get("sha256") and sha256_file(p) != f["sha256"]:
                    bad.append(f["path"])
            if missing:
                raise JobError(f"world {wid}: {len(missing)} file(s) listed in the manifest are missing on the volume, first {missing[0]}")
            if bad:
                raise JobError(f"world {wid}: {len(bad)} member files differ from the manifest, first {bad[0]}")
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(man_sha)
        return {"dir": str(wd), "id": wid, "manifest_sha256": man_sha, "members": len(members), "files": len(files)}

    def install_world(self, spec: dict, info: dict) -> Path | None:
        """Copy the world package to the pod's local disk once (tar stream, sequential reads) and return the local dir."""
        if not info:
            return None
        dest = self.ws / "dancecap" / "work" / "worlds" / str(spec.get("dance", "_")) / info["id"]
        marker = dest / ".installed"
        if marker.exists() and marker.read_text() == info["manifest_sha256"]:
            return dest
        dest.mkdir(parents=True, exist_ok=True)
        src = Path(info["dir"])
        n = 0
        for f in sorted(src.rglob("*")):
            if f.is_file():
                t = dest / f.relative_to(src)
                t.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(f, t)
                n += 1
        marker.write_text(info["manifest_sha256"])
        self.log(f"world {info['id']}: {n} files installed at {dest}")
        return dest

    # -- inputs
    def input_source(self, jid: str, path: str, src: str | None = None) -> Path:
        p = (src or path).replace("\\", "/").lstrip("/")
        if ".." in p.split("/"):
            raise JobError(f"bad input path {path}")
        return self.vol / p if p.startswith(SHARED_PREFIXES) else self.chunk_dir(jid) / "in" / p

    def materialize_inputs(self, spec: dict) -> dict:
        jid = spec["id"]
        t0 = now()
        n = size = 0
        for inp in spec.get("inputs", []):
            path = inp["path"]
            src = self.input_source(jid, path, inp.get("src"))
            if not src.is_file():
                raise JobError(f"input {path} missing on the volume ({src})")
            if inp.get("bytes") is not None and src.stat().st_size != int(inp["bytes"]):
                raise JobError(f"input {path}: {src.stat().st_size} bytes != spec {inp['bytes']}")
            if inp.get("sha256") and sha256_file(src) != inp["sha256"]:
                raise JobError(f"input {path}: sha256 differs from the spec")
            if inp.get("archive") or path.endswith((".tar", ".tar.gz", ".tgz")):
                with tarfile.open(src, "r:*") as tar:
                    _safe_extract(tar, self.ws)
            elif not (inp.get("src") or path).lstrip("/").startswith(SHARED_PREFIXES):
                dst = self.ws / path
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
            n += 1
            size += src.stat().st_size
        return {"n": n, "bytes": size, "seconds": round(now() - t0, 1)}

    # -- steps
    def step_env(self, spec: dict, step: dict, world_dir: Path | None, progress_file: Path) -> dict:
        env = dict(os.environ)
        venv = os.environ.get("HM_ENV_LOCAL", "/opt/hm/env")
        env.update({"DANCECAP_POD": "1", "PYOPENGL_PLATFORM": "egl", "PYTHONUNBUFFERED": "1", "HM_ROOT": str(self.vol),
                    "HM_CHUNK": spec["id"], "HM_POD_ID": self.pod_id, "YOLO_CONFIG_DIR": "/tmp/Ultralytics",
                    "TORCH_HOME": str(self.vol / "weights" / "torch_home"),
                    "DANCECAP_PROGRESS": str(progress_file), "DANCECAP_PROGRESS_JOB": spec["id"],
                    "DANCECAP_PROGRESS_CMD": step["name"], "DANCECAP_TAKE": str(spec.get("dance", ""))})
        if (Path(venv) / "bin").exists():
            env["PATH"] = f"{venv}/bin{os.pathsep}{env.get('PATH', '')}"
        env.update({"MPLBACKEND": "Agg", "HF_HOME": str(self.vol / "weights" / "hf"), "HF_HUB_DISABLE_TELEMETRY": "1",
                    "DANCECAP_ASR_PYTHON": f"{venv}/bin/python"})
        libs = nvidia_lib_dirs(venv)
        if libs:
            env["LD_LIBRARY_PATH"] = os.pathsep.join(libs + ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else []))
        if spec.get("world_id"):
            env["HM_WORLD_ID"] = spec["world_id"]
        if world_dir:
            env["HM_WORLD_DIR"] = str(world_dir)
        env["HM_WS"] = str(self.ws)
        for src in (spec.get("env") or {}, step.get("env") or {}):  # "{WS}" in a value = this pod's local workspace
            env.update({k: str(v).replace("{WS}", str(self.ws)) for k, v in src.items()})
        return env

    def run_command(self, spec: dict, step: dict, env: dict, log_path: Path) -> int:
        """One step: `bash -c cmd` (cmd.exe-free: the Linux pod) in a new session; python-side timeout; abort file."""
        cwd = self.ws / (spec.get("cwd") or "dancecap")
        cwd.mkdir(parents=True, exist_ok=True)
        timeout = float(step.get("timeout_min", spec.get("max_minutes", 60))) * 60
        log_path.parent.mkdir(parents=True, exist_ok=True)
        abort = self.chunk_dir(spec["id"]) / "abort"
        argv = ["bash", "-c", step["cmd"]] if os.name != "nt" else step["cmd"]
        with open(log_path, "ab", buffering=0) as lf:
            lf.write(f"[worker {time.strftime('%FT%T')}] start {step['name']} on {self.pod_id}\n".encode())
            proc = subprocess.Popen(argv, cwd=cwd, env=env, stdout=lf, stderr=subprocess.STDOUT, shell=(os.name == "nt"),
                                    start_new_session=(os.name != "nt"))
            t0 = now()
            rc = None
            while rc is None:
                try:
                    rc = proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    if now() - t0 > timeout:
                        self.log(f"step {step['name']}: timeout after {timeout / 60:.0f} min, killing")
                        _kill(proc)
                        rc = 124
                    elif abort.exists():
                        self.log(f"step {step['name']}: abort requested, killing")
                        _kill(proc)
                        rc = 130
                    elif self.lost_claim:
                        self.log(f"step {step['name']}: claim lost, killing")
                        _kill(proc)
                        rc = 131
                    elif self.job_deadline and now() > self.job_deadline:
                        self.log(f"step {step['name']}: the job's max_minutes ran out, killing")
                        _kill(proc)
                        rc = 125
            lf.write(f"[worker {time.strftime('%FT%T')}] end {step['name']} rc={rc} in {now() - t0:.0f}s\n".encode())
        return int(rc)

    def save_paths(self, cd: Path, name: str, paths: list[str], tag: str = "") -> None:
        """tar the listed workspace paths into chunks/<id>/ckpt/<name>.tar (atomic)."""
        ck = cd / "ckpt"
        ck.mkdir(parents=True, exist_ok=True)
        dst = ck / f"{tag}{name}.tar"
        tmp = dst.with_suffix(".tar.part")
        with tarfile.open(tmp, "w") as tar:
            for p in paths:
                f = self.ws / p
                if f.exists():
                    tar.add(f, arcname=p.rstrip("/"))
        os.replace(tmp, dst)

    def restore(self, cd: Path, name: str, tag: str = "") -> bool:
        f = cd / "ckpt" / f"{tag}{name}.tar"
        if not f.exists():
            return False
        with tarfile.open(f, "r") as tar:
            _safe_extract(tar, self.ws)
        return True

    def run_steps(self, spec: dict, world_dir: Path | None) -> dict:
        jid = spec["id"]
        cd = self.chunk_dir(jid)
        loc = self.state / jid
        loc.mkdir(parents=True, exist_ok=True)
        progress_file = loc / "progress.jsonl"
        self.cur["progress_file"] = str(progress_file)
        steps = spec.get("steps", [])
        self.cur["steps_n"] = len(steps)
        results: dict[str, dict] = {}
        rerun = False
        for i, step in enumerate(steps):
            name = step["name"]
            marker = cd / "steps" / f"{name}.json"
            lok = loc / f"{name}.ok"
            fp = step_fingerprint(spec, step)
            m = read_json(marker)
            if not rerun and m and m.get("rc") == 0:
                if m.get("fp") != fp:
                    self.log(f"step {name}: its marker is from a different plan (command / inputs / world / code changed), rerunning from here")
                elif (lok.exists() and lok.read_text() == fp) or self.restore(cd, name):
                    lok.write_text(fp)
                    results[name] = {"rc": 0, "seconds": m.get("seconds"), "skipped": True}
                    self.log(f"step {name}: already done (marker), skipped")
                    continue
                else:
                    self.log(f"step {name}: marker without local state or checkpoint, rerunning from here")
            rerun = True
            self.cur.update(step=name, step_i=i, step_started=now())
            if step.get("live") and self.restore(cd, name, tag="live_"):
                self.log(f"step {name}: restored its live snapshot")
            live_stop = threading.Event()
            if step.get("live"):
                threading.Thread(target=self._live_loop, args=(cd, name, step["live"], float(step.get("live_min", 8)), live_stop),
                                 daemon=True).start()
            t0 = now()
            self.beat()
            try:
                rc = self.run_command(spec, step, self.step_env(spec, step, world_dir, progress_file), cd / "log" / f"{name}.log")
            finally:
                live_stop.set()
            dt = round(now() - t0, 1)
            results[name] = {"rc": rc, "seconds": dt}
            if rc == 0:
                if step.get("save"):
                    self.save_paths(cd, name, step["save"])
                write_json(marker, {"rc": 0, "seconds": dt, "finished": now(), "pod_id": self.pod_id, "gpu": self.gpu0.get("name"), "fp": fp})
                lok.write_text(fp)
            elif step.get("optional"):
                self.log(f"step {name}: rc={rc} (optional, continuing)")
            else:
                write_json(marker, {"rc": rc, "seconds": dt, "finished": now(), "pod_id": self.pod_id, "fp": fp})
                raise StepFailed(name, rc, results)
        return results

    def _live_loop(self, cd: Path, name: str, paths: list[str], minutes: float, stop: threading.Event) -> None:
        while not stop.wait(minutes * 60):
            try:
                self.save_paths(cd, name, paths, tag="live_")
            except Exception as e:  # noqa: BLE001
                self.log(f"live snapshot of {name} failed: {e}")

    # -- outputs
    def pack_outputs(self, spec: dict, steps: dict, status: str) -> dict:
        """out/result.tar (the declared outputs at their workspace-relative paths) + out/manifest.json (sha256 per file)."""
        jid = spec["id"]
        out = self.chunk_dir(jid) / "out"
        out.mkdir(parents=True, exist_ok=True)
        files: list[dict] = []
        missing: list[str] = []
        tmp = out / "result.tar.part"
        with tarfile.open(tmp, "w") as tar:
            for o in spec.get("outputs", []):
                path = o["path"].replace("\\", "/").lstrip("/")
                src = self.ws / path
                if not src.exists():
                    if not o.get("optional"):
                        missing.append(path)
                    continue
                for f in ([src] if src.is_file() else sorted(p for p in src.rglob("*") if p.is_file())):
                    rel = f.relative_to(self.ws).as_posix()
                    files.append({"path": rel, "bytes": f.stat().st_size, "sha256": sha256_file(f)})
                    tar.add(f, arcname=rel, recursive=False)
        tar_sha = sha256_file(tmp)
        os.replace(tmp, out / "result.tar")
        man = {"schema": "dancecap.farm.out/1", "id": jid, "dance": spec.get("dance"), "kind": spec.get("kind"),
               "chunk": spec.get("chunk"), "world_id": spec.get("world_id"), "world_sha256": spec.get("world_sha256"),
               "pose_id": spec.get("pose_id"), "status": status, "missing": missing,
               "files": files, "tar": {"name": "result.tar", "bytes": (out / "result.tar").stat().st_size, "sha256": tar_sha},
               "created": now(), "pod_id": self.pod_id}
        write_json(out / "manifest.json", man)
        return man

    # -- one job
    def run_job(self, spec: dict) -> str:
        jid = spec["id"]
        cd = self.chunk_dir(jid)
        att = read_json(cd / "attempts.json", {"n": 0})
        att["n"] = int(att.get("n", 0)) + 1
        write_json(cd / "attempts.json", att)
        t_start = now()
        self.lost_claim = False
        self._claim_miss = 0
        # wall-clock limit of the whole job: a step that hangs with a live heartbeat (a GPU deadlock; the heartbeat is a thread) is
        # otherwise only stopped by its own timeout_min. 25 % + 5 min over the spec's max_minutes.
        mm = float(spec.get("max_minutes") or 0)
        self.job_deadline = t_start + (mm * JOB_LIMIT_FACTOR + JOB_LIMIT_SLACK_MIN) * 60 if mm > 0 else None
        self.cur = {"id": jid, "started": t_start, "attempt": att["n"]}
        self.event(f"job {jid} start attempt {att['n']}")
        self._hb_stop.clear()
        self._hb_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._hb_thread.start()
        self.beat(phase="prepare")
        steps: dict = {}
        status, err = "ok", None
        try:
            self.preflight(spec)
            winfo = self.verify_world(spec)
            wdir = self.install_world(spec, winfo)
            self.use_code(spec)
            inp = self.materialize_inputs(spec)
            self.log(f"{jid}: {inp['n']} inputs ({inp['bytes'] / 1e6:.0f} MB) in {inp['seconds']} s")
            self.event(f"job {jid} first step")
            steps = self.run_steps(spec, wdir)
        except StepFailed as e:
            status, err, steps = "failed", f"step {e.step} rc={e.rc}" + (" (job max_minutes exceeded)" if e.rc == 125 else ""), e.results
        except JobError as e:
            status, err = "failed", str(e)
        except Exception as e:  # noqa: BLE001
            status, err = "failed", f"{type(e).__name__}: {e}"
        if self.lost_claim:   # somebody else owns the job now: write nothing (done / failed / out would race the new owner), release nothing
            self._hb_stop.set()
            self.failed_here.add(jid)
            self.claims.pop(jid, None)
            self.cur = {}
            self.event(f"job {jid} claim lost, abandoned")
            self.log(f"{jid}: claim lost, abandoned without writing a result")
            return "lost"
        try:
            man = self.pack_outputs(spec, steps, status if status == "ok" else "partial")
            if status == "ok" and man["missing"]:
                status, err = "failed", f"required outputs missing: {man['missing'][:5]}"
        except Exception as e:  # noqa: BLE001
            man = {"files": [], "missing": ["pack failed"]}
            status, err = "failed", f"{err or ''} pack failed: {type(e).__name__}: {e}".strip()
        self._hb_stop.set()
        seconds = round(now() - t_start, 1)
        info = {"id": jid, "status": status, "error": err, "steps": steps, "seconds": seconds, "gpu": self.gpu0,
                "price_hr": self.price_hr, "cost_usd": round(self.price_hr * seconds / 3600, 4),
                "world_id": spec.get("world_id"), "pose_id": spec.get("pose_id"), "world_sha256": spec.get("world_sha256"),
                "code": spec.get("code"), "pod_id": self.pod_id,
                "started": t_start, "finished": now(), "attempt": att["n"], "out_files": len(man.get("files", [])),
                "out_bytes": sum(f["bytes"] for f in man.get("files", []))}
        if status == "ok":
            write_json(cd / "done.json", info)
            (cd / "failed.json").unlink(missing_ok=True)
            self.log(f"{jid}: done in {seconds:.0f} s, {info['out_files']} files {info['out_bytes'] / 1e6:.0f} MB")
        else:
            maxn = int(spec.get("retries", 2))
            self.failed_here.add(jid)
            self.log(f"{jid}: FAILED ({err}) attempt {att['n']}/{maxn}")
            if att["n"] >= maxn:
                write_json(cd / "failed.json", info)
            else:
                write_json(cd / "last_failure.json", info)
        self.release(jid)
        self.cur = {}
        write_json(self.pod_dir / "heartbeat", {"t": now(), "state": "idle", "job": None})
        self.event(f"job {jid} {status}")
        return status

    def preflight(self, spec: dict) -> None:
        need_gpu = needs_gpu(spec) if self.gpu_required is None else self.gpu_required
        if need_gpu and not self.gpu0:
            raise JobError("job needs a GPU but nvidia-smi shows none on this pod")
        mv = spec.get("min_vram_gb")
        if mv and self.gpu0 and self.gpu0["mem_total_mb"] < float(mv) * 1024 * 0.95:
            raise JobError(f"job needs {mv} GB VRAM, this GPU has {self.gpu0['mem_total_mb'] / 1024:.0f} GB")
        self.work.mkdir(parents=True, exist_ok=True)
        link = self.ws / "dancecap" / "work"
        if self.ws.exists() and not link.exists() and not link.is_symlink():
            try:
                link.symlink_to(self.work)
            except OSError:
                pass

    def use_code(self, spec: dict) -> None:
        """Pin the code version: when the spec names a bundle sha256, make sure that bundle is what /opt/hm/ws holds."""
        want = (spec.get("code") or {}).get("bundle_sha256")
        if not want:
            return
        cur = (self.ws / ".code_sha256").read_text().strip() if (self.ws / ".code_sha256").exists() else None
        if cur == want:
            return
        for meta in sorted((self.vol / "code").glob("*.json")):
            m = read_json(meta, {})
            if m.get("bundle_sha256") == want:
                bundle = self.vol / "code" / f"{m['key']}.tar.gz"
                self.log(f"code: unpacking pinned bundle {m['key']}")
                for d in ("dancecap/dancecap", "dancecap/takes", "dancecap/remote", "mamma-turing"):
                    shutil.rmtree(self.ws / d, ignore_errors=True)
                with tarfile.open(bundle, "r:gz") as tar:
                    _safe_extract(tar, self.ws)
                (self.ws / ".code_sha256").write_text(want)
                (self.ws / ".code_key").write_text(m["key"])
                return
        raise JobError(f"code bundle {want[:12]} is not on the volume (code/*.json)")

    # -- main loop
    def serve(self, max_jobs: int | None = None, idle_exit_min: float = 5.0, once: bool = False) -> int:
        self.event(f"worker up pod={self.pod_id} gpu={self.gpu0.get('name')} kinds={sorted(self.kinds) if self.kinds else 'all'}")
        done = 0
        idle_since = now()
        while True:
            write_json(self.pod_dir / "heartbeat", {"t": now(), "state": "idle", "job": None})
            job = None
            for spec in self.specs():
                if not self.claimable(spec) or not self.enough_time(spec):
                    continue
                if self.try_claim(spec):
                    job = spec
                    break
            if job:
                idle_since = now()
                self.run_job(job)
                done += 1
                if max_jobs and done >= max_jobs:
                    self.log(f"max jobs ({max_jobs}) reached")
                    return 0
                if once:
                    return 0
                continue
            if now() - idle_since > idle_exit_min * 60:
                self.log(f"queue empty for {idle_exit_min:g} min: exiting")
                write_json(self.pod_dir / "heartbeat", {"t": now(), "state": "exited", "job": None, "jobs_done": done})
                return 0
            time.sleep(self.poll_s)


def needs_gpu(spec: dict) -> bool:
    """Spec's explicit needs_gpu, else by kind (GPU_KINDS)."""
    if "needs_gpu" in spec:
        return bool(spec["needs_gpu"])
    return spec.get("kind") in GPU_KINDS


def nvidia_lib_dirs(venv: str | None = None) -> list[str]:
    """site-packages/nvidia/*/lib of the volume env: ctranslate2 (faster-whisper) loads cuBLAS / cuDNN by dlopen and does not
    import torch first, so the pip CUDA libraries must be on LD_LIBRARY_PATH."""
    root = Path(venv or os.environ.get("HM_ENV_LOCAL", "/opt/hm/env")) / "lib"
    return sorted(str(p) for p in root.glob("python3*/site-packages/nvidia/*/lib") if p.is_dir())


class JobError(RuntimeError):
    pass


class StepFailed(RuntimeError):
    def __init__(self, step: str, rc: int, results: dict):
        super().__init__(f"step {step} rc={rc}")
        self.step, self.rc, self.results = step, rc, results


def _kill(proc: subprocess.Popen) -> None:
    try:
        if os.name == "nt":
            proc.kill()
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (OSError, ProcessLookupError):
        pass


def _safe_extract(tar: tarfile.TarFile, dest: Path) -> None:
    """Extract regular files and directories only; no absolute paths, no '..', no links or devices."""
    # LEXICAL containment check: on a pod `<ws>/dancecap/work` is a symlink to the local scratch disk (pod_prepare.sh / preflight), so
    # resolve()-ing the target left `dest` and every member under dancecap/work was SKIPPED SILENTLY (prepared inputs never arrived:
    # the drill's fit died with "prepare.json not found"). Members are already refused when absolute / '..' / a link.
    dest = Path(os.path.abspath(dest))
    for m in tar.getmembers():
        name = m.name[2:] if m.name.startswith("./") else m.name
        if m.issym() or m.islnk() or m.isdev() or name.startswith("/") or ".." in Path(name).parts:
            continue
        target = Path(os.path.normpath(dest / name))
        if dest not in target.parents and target != dest:
            continue
        if m.isdir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        src = tar.extractfile(m)
        if src is None:
            continue
        with open(target, "wb") as fh:
            shutil.copyfileobj(src, fh, 1 << 22)
        os.utime(target, (m.mtime, m.mtime))


# ------------------------------------------------------------------------------------------------ step templates


def fit_steps(take: str, tag: str, pin_json: str | None = None, *, minutes: float = 100, preflight: bool = True) -> list[dict]:
    """Steps of the MAMMA fit as run for larissa_kadu_full a-d (dancecap.fit_full.job_specs): preflight (GPU, imports,
    inputs, a 20-frame smoke fit with the betas pinned and checked), the full ma_3d fit of the prepared camera set with the
    betas pinned and frozen to the world's, select (promotes mamma/fits/<tag> to smplx/), verify. `pin_json` = the world's
    mamma_pin_betas.json (workspace-relative). Outputs to declare: work/<take>/{mamma/fits, mamma/ma_3d, mamma/logs,
    mamma/select.json, mamma/report.json, smplx}."""
    env = {"MAMMA_FREEZE_BETAS": "1"}
    if pin_json:
        env["MAMMA_INIT_BETAS"] = pin_json if pin_json.startswith("/") else "{WS}/" + pin_json
    steps = []
    if preflight:
        steps.append({"name": "preflight", "cmd": f"python -m dancecap.fit_full preflight {take}", "timeout_min": 12, "env": env})
    steps += [
        {"name": f"fit_{tag}", "cmd": f"python -m dancecap mamma fit {take} --tag {tag}", "timeout_min": minutes, "env": env,
         "live": [f"dancecap/work/{take}/mamma/ma_3d/"], "live_min": 10,
         "save": [f"dancecap/work/{take}/mamma/fits/", f"dancecap/work/{take}/mamma/ma_3d/"]},
        {"name": "select", "cmd": f"python -m dancecap mamma select {take} --no-qa", "timeout_min": 8, "env": env,
         "save": [f"dancecap/work/{take}/smplx/", f"dancecap/work/{take}/mamma/select.json", f"dancecap/work/{take}/mamma/report.json"]},
        {"name": "verify", "cmd": f"python -m dancecap.fit_full verify {take}", "timeout_min": 6, "optional": True, "env": env},
    ]
    return steps


def fit_outputs(take: str) -> list[dict]:
    w = f"dancecap/work/{take}"
    return [{"path": f"{w}/mamma/fits/"}, {"path": f"{w}/mamma/ma_3d/", "optional": True}, {"path": f"{w}/mamma/logs/", "optional": True},
            {"path": f"{w}/mamma/select.json"}, {"path": f"{w}/mamma/report.json", "optional": True},
            {"path": f"{w}/mamma/preflight.json", "optional": True}, {"path": f"{w}/mamma/fit_check.json", "optional": True},
            {"path": f"{w}/smplx/"}]


def refine_steps(take: str, dst: str = "smplx_rf2_noslip", *, minutes: float = 20) -> list[dict]:
    """pose_refine with the larissa_kadu_38 recipe (smplx_rf2_noslip = the defaults: root 5 Hz, rotation 8 Hz, image-aware root
    height + legs with the floor term, 400 iterations)."""
    return [{"name": "refine", "cmd": f"python -m dancecap.pose_refine {take} --src smplx --dst {dst} --device cuda",
             "timeout_min": minutes, "save": [f"dancecap/work/{take}/{dst}/"]}]


def refine_outputs(take: str, dst: str = "smplx_rf2_noslip") -> list[dict]:
    return [{"path": f"dancecap/work/{take}/{dst}/"}]


def front_steps(take: str, *, minutes: float = 150) -> list[dict]:
    """The S1 front end of one take on a pod (dancecap.front_pod, as front_pod.write_job): public weights are already on the
    volume (this only verifies their sha256), environment check, the supervised driver (restarted twice if the process
    itself dies; it adopts live workers and resumes from its task markers), status."""
    return [
        {"name": "weights", "cmd": "python -m dancecap.front_full fetch-weights", "timeout_min": 12},
        {"name": "check", "cmd": f"python -m dancecap.front_pod check {take}", "timeout_min": 12},
        *[{"name": f"front{'' if k == 0 else f'_{k + 1}'}", "optional": True, "timeout_min": minutes,
           "cmd": f"python -m dancecap.front_pod supervise {take}"} for k in range(3)],
        {"name": "status", "cmd": f"python -m dancecap.front_pod status {take}", "timeout_min": 2},
    ]


def smoke_spec(job_id: str = "smoke-gpu", frames: int = 60, gpu: bool = True) -> dict:
    """Self-test job (no world, tiny inputs): GPU + imports + MAMMA end-to-end on the synthetic couple with ground truth
    (`python -m dancecap mamma --validate-synthetic`, as remote_jobs.validation_job). gpu=False: a protocol test that only
    imports the stack (needs_gpu false). Inputs: work/synthetic_mamma/gt120/smplx/{lead,follow}.npz (submit_job pushes them)."""
    imports = ("python -c \"import dancecap.mamma_dyn, dancecap.room, dancecap.avatar_texture, dancecap.env as e, torch; "
               "print('dancecap imports ok; is_pod', e.is_pod(), 'torch', torch.__version__, 'cuda', torch.cuda.is_available())\"")
    steps = [{"name": "imports", "cmd": imports, "timeout_min": 5}]
    out = "dancecap/work/synthetic_mamma/noise0"
    if gpu:
        steps = [{"name": "gpu_check", "timeout_min": 3,
                  "cmd": "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader && python -c "
                         "\"import torch; x=torch.randn(2048,2048,device='cuda'); print('matmul ok', float((x@x).sum()!=0)); "
                         "from pytorch_sdf import sdf; import gsplat; print('pytorch_sdf + gsplat import ok')\""}] + steps
        steps.append({"name": "mamma_synthetic", "timeout_min": 25,
                      "cmd": f"python -m dancecap mamma --validate-synthetic --frames {frames}",
                      "save": [out + "/"]})
    return {"schema": SCHEMA, "id": job_id, "dance": "synthetic_mamma", "kind": "fit" if gpu else "analyse",
            "chunk": {"id": job_id, "kind": "smoke"}, "world_id": None, "world_sha256": None, "code": {},
            "inputs": [], "steps": steps,
            "outputs": [{"path": f"{out}/report.json", "optional": not gpu}, {"path": f"{out}/smplx/", "optional": not gpu}],
            "max_minutes": 30, "priority": 1000, "needs_gpu": gpu, "retries": 1}


# ANALYSIS: everything that analyses a result runs on a pod (user directive 2026-10-08, FARM.md section 0 item 10), kind
# "analyse" on a CPU pod (pods.launch_cpu, about $0.035 per vCPU-hour, the volume mounted) unless it needs a GPU (asr).
# name -> (command with {take}, default timeout minutes, python -m target for the docs). The command is run in dancecap/ with
# the volume env first on PATH; `args` are appended. Inputs come from the volume (the stitched result as an archive input
# {"path": "chunks/<stitch id>/out/result.tar", "archive": true}); results are the declared outputs of the job.
ANALYSIS = {
    "prepare":         ("python -m dancecap mamma prepare {take} --skip-2d", 30),
    "stitch":          ("python -m dancecap.stitch {take}", 30),
    "pose_quality":    ("python -m dancecap.pose_quality {take}", 20),
    "timing":          ("python -m dancecap timing {take}", 40),
    "timing_free":     ("python -m dancecap.timing_free {take}", 40),   # no beat grid (the class recap); farm/recap_analysis.py
    "physics":         ("python -m dancecap physics {take}", 40),
    "counterbalance":  ("python -m dancecap.counterbalance detect --take {take}", 30),
    "floorcraft":      ("python -m dancecap.floorcraft {take} --no-review", 30),
    "analysis_review": ("python -m dancecap.analysis_review {take}", 30),
    # camera-gate evidence (independent measurements from the source videos; cameras/gate_cameras.py doc)
    "video_zoom":      ("python -m dancecap.cameras.video_zoom {take} {cam}", 60),
    "video_motion":    ("python -m dancecap.cameras.video_motion", 60),
    "gate_cameras":    ("python -m dancecap.cameras.gate_cameras table", 40),
    "gate_final":      ("python -m dancecap.cameras.gate_final run", 30),
    "attach_global":   ("python -m dancecap.cameras.attach_global", 60),
}


def analysis_steps(what: str, take: str = "", args: str = "", *, cam: str = "", minutes: float | None = None,
                   name: str | None = None, save: list[str] | None = None) -> list[dict]:
    """One analysis step from ANALYSIS (`what` in ANALYSIS), e.g. analysis_steps("timing", "larissa_kadu_full", "--mc 200").
    Several can be concatenated into one job's steps (each has its own done marker, so a rerun skips finished ones). `save`
    = workspace-relative paths snapshotted to the volume when the step succeeds (resume on another pod)."""
    if what not in ANALYSIS:
        raise KeyError(f"unknown analysis {what!r}; known: {sorted(ANALYSIS)}")
    cmd, default_min = ANALYSIS[what]
    cmd = cmd.format(take=take, cam=cam).strip()
    if args:
        cmd += " " + args
    step = {"name": name or (f"{what}_{cam}" if cam else what), "cmd": cmd, "timeout_min": float(minutes or default_min)}
    if save:
        step["save"] = save
    return [step]


def asr_steps(wav: str, out: str, *, model: str = "large-v3", passes: str = "en,pt", device: str = "cuda",
              compute_type: str | None = None, extra: str = "", minutes: float = 30, name: str = "asr") -> list[dict]:
    """Narration ASR with the PC's own worker (dancecap/moves/annotate/asr_worker.py: faster-whisper large-v3, Silero VAD,
    word timestamps, two passes en+pt fused later by dancecap.moves.annotate.asr). The models are on the volume
    (HF_HOME=<volume>/weights/hf, loaded offline); the CUDA libraries come from the volume env (LD_LIBRARY_PATH, set by the
    worker). `wav`/`out` are workspace-relative. device=cpu runs int8 (no GPU needed; give the job needs_gpu false)."""
    ct = compute_type or ("float16" if device == "cuda" else "int8")
    cmd = (f"python dancecap/moves/annotate/asr_worker.py --wav {wav} --out {out} --model {model} --device {device} "
           f"--compute-type {ct} --passes {passes}" + (f" {extra}" if extra else ""))
    return [{"name": name, "cmd": cmd, "timeout_min": minutes, "save": [out]}]


def analyse_spec(job_id: str, dance: str, steps: list[dict], outputs: list[dict], *, after: list[str] | None = None,
                 inputs: list[dict] | None = None, world_id: str | None = None, world_sha256: str | None = None,
                 max_minutes: float = 60, priority: float = 0, gpu: bool = False, kind: str = "analyse") -> dict:
    """A dancecap.farm.job/1 spec for analysis steps (CPU by default). Chain it behind the fits with after=[chunk ids] and the
    archive inputs of those chunks."""
    return {"schema": SCHEMA, "id": job_id, "dance": dance, "kind": kind if not gpu else "asr" if kind == "analyse" else kind,
            "chunk": {"id": job_id, "kind": kind}, "world_id": world_id, "world_sha256": world_sha256, "code": {},
            "after": list(after or []), "inputs": list(inputs or []), "steps": steps, "outputs": outputs,
            "max_minutes": max_minutes, "priority": priority, "needs_gpu": bool(gpu), "retries": 2}


STEP_TEMPLATES = {"fit": fit_steps, "refine": refine_steps, "front": front_steps, "analyse": analysis_steps, "asr": asr_steps}


# ------------------------------------------------------------------------------------------------ CLI


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.farm.worker", description=__doc__.split("\n")[0])
    ap.add_argument("--volume", default=os.environ.get("HM_ROOT", "/workspace/hm"))
    ap.add_argument("--kinds", help="comma list of job kinds this worker takes (default: all)")
    ap.add_argument("--max-jobs", type=int)
    ap.add_argument("--idle-exit-min", type=float, default=5.0)
    ap.add_argument("--pod-id")
    ap.add_argument("--poll-s", type=float, default=10.0)
    ap.add_argument("--stale-s", type=float, default=240.0)
    ap.add_argument("--once", action="store_true", help="take at most one job")
    ap.add_argument("--ws", help="local workspace (default /opt/hm/ws)")
    ap.add_argument("--work", help="local scratch = dancecap/work (default /opt/hm/work)")
    ap.add_argument("--no-gpu-check", action="store_true")
    a = ap.parse_args(argv)
    kinds = {k.strip() for k in a.kinds.split(",") if k.strip()} if a.kinds else None
    w = Worker(Path(a.volume), pod_id=a.pod_id, kinds=kinds, ws=Path(a.ws) if a.ws else None, work=Path(a.work) if a.work else None,
               stale_s=a.stale_s, poll_s=a.poll_s, gpu_required=False if a.no_gpu_check else None)
    return w.serve(max_jobs=a.max_jobs, idle_exit_min=a.idle_exit_min, once=a.once)


if __name__ == "__main__":
    sys.exit(main())
