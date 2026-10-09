"""Reading the volume and bringing finished jobs home (FARM.md section 6): the watcher's I/O layer.

Everything here talks to the volume through a `volume.Link` (an ssh handle of ANY pod that mounts it; the volume is shared, so a
finished job can be pulled through a different pod than the one that produced it) and is written to survive what the home
connection does to it: a dropped ssh stream, a half-written file, a killed process.

    snap = take_snapshot(link, job_ids, pod_ids)       # ONE ssh call: done/failed/heartbeat/claim of every job, kill_at of every pod
    res  = pull_job(link, "g0000", local_dir, dest)    # sha256-verified, resumable, idempotent; extracts at workspace paths
    release_claim(link, "g0000", "podX")               # give a dead pod's job back (guarded: only if podX still owns it)
    requeue_failed(link, "g0000")                      # move failed.json aside and reset the attempts so a worker takes it again

A pull is verified twice: the tar against the sha256 the worker recorded in out/manifest.json (size first, then hash), and every
member against the manifest's per-file sha256 while it is written (temp file + atomic replace, so a killed watcher never leaves
a half file behind). `pull.json` in the job's local directory records the verified pull; a restarted watcher trusts it only when
the tar sha256 on the volume is still the one it recorded.
"""
from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import tarfile
import time
from pathlib import Path

from .state import atomic_write_json, read_json
from .worker import sha256_file


class PullError(RuntimeError):
    """transient=True: try again later (ssh reset, hash mismatch of a download); False: needs a person (unsafe archive, ...)."""

    def __init__(self, msg: str, transient: bool = True):
        super().__init__(msg)
        self.transient = transient


def pid_alive(pid: int) -> bool:
    if not pid or int(pid) <= 0:
        return False
    if os.name == "nt":
        import ctypes
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x1000, False, int(pid))   # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        try:
            code = ctypes.c_ulong()
            return bool(k32.GetExitCodeProcess(h, ctypes.byref(code))) and code.value == 259   # STILL_ACTIVE
        finally:
            k32.CloseHandle(h)
    try:
        os.kill(int(pid), 0)
        return True
    except PermissionError:
        return True
    except OSError:
        return False


class PullLock:
    """One puller per job at a time (the watcher, `farm pull`, `farm down` may all try): a lock file holding the owner's pid. A lock
    whose process is gone (a killed watcher) is taken over at once; any other is respected for up to `stale` seconds."""

    def __init__(self, path: Path, stale: float = 7200.0):
        self.path, self.stale = Path(path), stale

    def acquire(self) -> None:
        import socket
        self.path.parent.mkdir(parents=True, exist_ok=True)
        me = {"pid": os.getpid(), "host": socket.gethostname(), "t": time.time()}
        for _ in range(3):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                held = read_json(self.path, None) or {}
                dead = held.get("host") == me["host"] and not pid_alive(held.get("pid") or 0)
                old = time.time() - float(held.get("t") or 0) > self.stale
                if dead or old or not held:
                    try:
                        os.unlink(self.path)
                    except OSError:
                        pass
                    continue
                raise PullError(f"{self.path.parent.name}: being pulled by pid {held.get('pid')}", transient=True)
            with os.fdopen(fd, "w") as fh:
                fh.write(json.dumps(me))
            return
        raise PullError(f"could not take {self.path}", transient=True)

    def release(self) -> None:
        try:
            held = read_json(self.path, None) or {}
            if held.get("pid") == os.getpid():
                os.unlink(self.path)
        except OSError:
            pass


# ------------------------------------------------------------------------------------------------ snapshot

# Runs ON the pod with the system python3 (stdlib only): one pass over the small state files of the watched jobs and pods, one JSON
# line out. The volume is slow for many small files (~12 ms each), so only the ids asked for are read.
SNAPSHOT_PY = r'''
import json, os, sys, time
ROOT = sys.argv[1]
jobs, pods, deps = [], [], []
for a in sys.argv[2:]:
    (jobs if a.startswith("j:") else deps if a.startswith("d:") else pods).append(a[2:])
def rj(p):
    try:
        with open(p, "rb") as f:
            return json.loads(f.read().decode("utf-8", "replace"))
    except Exception:
        return None
def rt(p):
    try:
        with open(p, "rb") as f:
            return f.read().decode("utf-8", "replace").strip()
    except Exception:
        return None
def rj_err(p):
    # (data, err): err is True when the file EXISTS but could not be read or parsed (EIO / ESTALE on the network volume); a file that is
    # simply absent is (None, False). The watcher must not read "unreadable" as "not there".
    try:
        with open(p, "rb") as f:
            raw = f.read()
    except FileNotFoundError:
        return None, False
    except Exception:
        return None, True
    try:
        return json.loads(raw.decode("utf-8", "replace")), False
    except Exception:
        return None, True
def pick(d, keys):
    return {k: d[k] for k in keys if isinstance(d, dict) and k in d}
def tail(p, n=8, size=4096):
    try:
        with open(p, "rb") as f:
            f.seek(0, 2); sz = f.tell(); f.seek(max(0, sz - size))
            return f.read().decode("utf-8", "replace").splitlines()[-n:]
    except Exception:
        return []
def num(s):
    try:
        return int("".join(c for c in (s or "") if c.isdigit()))
    except Exception:
        return None
out = {"schema": "dancecap.farm.snap/1", "t": time.time(), "root": ROOT, "jobs": {}, "pods": {}}
DONE = ("status", "error", "seconds", "gpu", "price_hr", "cost_usd", "world_id", "code", "pod_id", "started", "finished",
        "attempt", "out_files", "out_bytes", "steps")
FAIL = ("status", "error", "seconds", "pod_id", "started", "finished", "attempt", "steps")
HB = ("t", "pod_id", "id", "step", "step_i", "steps_n", "step_started", "started", "attempt", "progress", "gpu",
      "disk_free_gb", "load1", "kill_in_s", "phase")
for jid in jobs:
    cd = ROOT + "/chunks/" + jid
    e = {"in_queue": os.path.exists(ROOT + "/queue/" + jid + ".json"), "dir": os.path.isdir(cd)}
    if e["dir"]:
        d, derr = rj_err(cd + "/done.json")
        if isinstance(d, dict):
            e["done"] = pick(d, DONE)
        f, ferr = rj_err(cd + "/failed.json")
        if isinstance(f, dict):
            e["failed"] = pick(f, FAIL)
        if derr or ferr:
            e["read_err"] = True
        lf = rj(cd + "/last_failure.json")
        if isinstance(lf, dict):
            e["last_failure"] = pick(lf, FAIL)
        hb = rj(cd + "/heartbeat")
        if isinstance(hb, dict):
            e["heartbeat"] = pick(hb, HB)
        at = rj(cd + "/attempts.json")
        if isinstance(at, dict):
            e["attempts"] = at.get("n")
        if os.path.isdir(cd + "/claim.lock"):
            ow = rj(cd + "/claim.lock/owner.json")
            e["claim"] = pick(ow, ("pod_id", "t")) if isinstance(ow, dict) else {"pod_id": None, "t": None}
        if e.get("done") or e.get("failed"):
            m = rj(cd + "/out/manifest.json")
            if isinstance(m, dict):
                e["manifest"] = {"status": m.get("status"), "missing": m.get("missing") or [], "tar": m.get("tar"),
                                 "world_id": m.get("world_id"), "n_files": len(m.get("files") or []), "pod_id": m.get("pod_id"),
                                 "created": m.get("created"), "chunk": pick(m.get("chunk"), ("id", "take", "k0", "k1", "core_k0", "core_k1"))}
    out["jobs"][jid] = e
for jid in deps:   # only "is it done": a chunk some watched job runs after (its manifest is not needed again)
    if jid not in out["jobs"]:
        d = rj(ROOT + "/chunks/" + jid + "/done.json")
        out["jobs"][jid] = {"dir": True, "done": pick(d, ("status", "pod_id", "world_id"))} if isinstance(d, dict) else {"dir": False}
for pid in pods:
    pd = ROOT + "/pods/" + pid
    hbd, hberr = rj_err(pd + "/heartbeat")
    out["pods"][pid] = {"heartbeat": hbd, "heartbeat_err": hberr, "kill_at": num(rt(pd + "/kill_at")), "worker_exit": rt(pd + "/worker_exit"),
                        "events": tail(pd + "/events.log")}
print(json.dumps(out))
'''


def take_snapshot(link, jobs, pods, *, deps=(), python: str | None = None, timeout: float = 120) -> dict:
    """The state of `jobs` (chunk job ids) and `pods` (pod ids) as the volume holds it right now, in one ssh round trip. `deps` are job
    ids that only need the "done?" answer (jobs a watched job runs after).
    Times inside (`t`, heartbeats, `kill_at`) are the POD's clock. Raises RuntimeError / TimeoutExpired / OSError when the link is
    down; ValueError when the answer is not a snapshot."""
    py = python or getattr(link, "python", None) or "python3"
    args = " ".join(shlex.quote(x) for x in [link.root, *[f"j:{j}" for j in jobs], *[f"d:{d}" for d in deps if d not in set(jobs)],
                                              *[f"p:{p}" for p in pods]])
    raw = link.sh(f"{shlex.quote(py)} - {args} <<'HM_SNAP'\n{SNAPSHOT_PY}\nHM_SNAP\n", timeout)
    for line in reversed(raw.strip().splitlines()):   # tolerate a login banner before the JSON line
        line = line.strip()
        if line.startswith("{"):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(d, dict) and d.get("schema") == "dancecap.farm.snap/1":
                return d
    raise ValueError(f"not a snapshot: {raw[-300:]!r}")


# ------------------------------------------------------------------------------------------------ small remote operations


def release_claim(link, jid: str, pod_id: str) -> bool:
    """Rename chunks/<jid>/claim.lock aside (atomic, as the worker's own takeover does) so a live worker can take the job at once
    instead of waiting for the heartbeat to go stale. Only if `pod_id` still owns the claim (a worker of another pod may have taken
    it over meanwhile). Returns True when the claim was released."""
    cd = shlex.quote(link.abs(f"chunks/{jid}"))
    pat = shlex.quote('"pod_id": ' + json.dumps(pod_id))   # owner.json is written by worker.write_json (indent=1)
    out = link.sh(
        f"cd {cd} 2>/dev/null || {{ echo gone; exit 0; }}; "
        f"if [ -d claim.lock ] && grep -q {pat} claim.lock/owner.json 2>/dev/null; then "
        f"mv claim.lock claim.stale.$(date +%s).watch && echo released; else echo kept; fi", 60)
    return "released" in out


def requeue_failed(link, jid: str) -> None:
    """A job the worker gave up on (failed.json after its retries): keep the evidence (failed.<ts>.json), reset attempts.json so the
    next worker that takes it starts at attempt 1. Finished step markers (and their checkpoints) stay: the retry resumes after
    the last good step."""
    cd = shlex.quote(link.abs(f"chunks/{jid}"))
    link.sh(f"cd {cd} && ts=$(date +%s) && "
            "{ [ -f failed.json ] && mv failed.json failed.$ts.json; true; } && "
            "{ [ -f last_failure.json ] && mv last_failure.json last_failure.$ts.json; true; } && "
            "rm -f abort && echo '{\"n\": 0}' > attempts.json.tmp && mv attempts.json.tmp attempts.json", 60)


def quarantine_result(link, jid: str) -> None:
    """The result on the volume is corrupt (its tar never matches the sha256 the worker recorded): move out/, done.json and the step
    markers aside (evidence kept as *.corrupt.<ts>) and reset attempts.json, so the next worker re-runs the job and packs a good one."""
    cd = shlex.quote(link.abs(f"chunks/{jid}"))
    link.sh(f"cd {cd} && ts=$(date +%s) && "
            "{ [ -d out ] && mv out out.corrupt.$ts; true; } && "
            "{ [ -f done.json ] && mv done.json done.corrupt.$ts.json; true; } && "
            "{ [ -d steps ] && mv steps steps.corrupt.$ts; true; } && "
            "rm -f abort && echo '{\"n\": 0}' > attempts.json.tmp && mv attempts.json.tmp attempts.json", 120)


# ------------------------------------------------------------------------------------------------ extraction


def _lp(p) -> str:
    """Windows long-path form (deep MAMMA output trees exceed MAX_PATH); plain path elsewhere."""
    s = os.path.abspath(str(p))
    if os.name == "nt" and not s.startswith("\\\\?\\"):
        s = "\\\\?\\UNC\\" + s[2:] if s.startswith("\\\\") else "\\\\?\\" + s
    return s


def _replace(src: str, dst: str, tries: int = 40) -> None:
    for i in range(tries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:   # Windows: a reader (monitor, indexer) has the target open for a moment
            if i == tries - 1:
                raise
            time.sleep(0.05)


def extract_verified(tar_path: Path, dest_root: Path, files: list[dict]) -> dict:
    """Extract result.tar under `dest_root` (members are workspace-relative paths) checking every member against the manifest's
    sha256 while it is written. Files already in place with the right hash are left alone (idempotent re-extraction after a
    crash). Raises PullError when a member's hash differs, the archive escapes `dest_root`, or a manifest file is missing."""
    want = {f["path"].replace("\\", "/").lstrip("/"): f for f in files}
    done: set[str] = set()
    skipped, nbytes, extra = 0, 0, []
    dest_root = Path(dest_root)
    with tarfile.open(tar_path, "r:") as tar:
        for m in tar:
            name = m.name[2:] if m.name.startswith("./") else m.name
            if not m.isfile():
                continue
            parts = Path(name).parts
            if name.startswith(("/", "\\")) or ".." in parts or (parts and ":" in parts[0]):
                raise PullError(f"unsafe archive member {name!r}", transient=False)
            w = want.get(name)
            if w is None:
                extra.append(name)
                continue
            target = dest_root / name
            if target.is_file() and target.stat().st_size == m.size and sha256_file(target) == w["sha256"]:
                done.add(name)
                skipped += 1
                continue
            os.makedirs(_lp(target.parent), exist_ok=True)
            tmp = _lp(target) + ".part"
            h = hashlib.sha256()
            n = 0
            src = tar.extractfile(m)
            with open(tmp, "wb") as out:
                for blk in iter(lambda: src.read(1 << 22), b""):
                    h.update(blk)
                    out.write(blk)
                    n += len(blk)
            if h.hexdigest() != w["sha256"] or (w.get("bytes") is not None and n != int(w["bytes"])):
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise PullError(f"{name}: sha256 {h.hexdigest()[:12]} != manifest {str(w['sha256'])[:12]}")
            _replace(tmp, _lp(target))
            try:
                os.utime(_lp(target), (m.mtime, m.mtime))
            except OSError:
                pass
            done.add(name)
            nbytes += n
    missing = sorted(set(want) - done)
    if missing:
        raise PullError(f"{len(missing)} manifest files not in the archive, first {missing[0]}")
    return {"files": len(done), "skipped": skipped, "bytes": nbytes, "extra": extra[:5]}


# ------------------------------------------------------------------------------------------------ the pull


def pull_job(link, jid: str, local_dir: Path, dest_root: Path, *, take: str | None = None, keep_tar: bool = False,
             progress=None, disk_factor: float = 2.5) -> dict:
    """Bring chunks/<jid>/out/ home: result.tar (resumable, size + sha256 checked against out/manifest.json) extracted at its
    workspace-relative paths under `dest_root` (every file verified), done.json / manifest.json mirrored in `local_dir`, and
    `pull.json` written LAST (it is the proof of a verified pull). The chunk's work dir also gets farm/pulled.json with the
    world_id the pod recorded. Idempotent: a pull whose tar sha256 matches the recorded one returns at once.

    Raises PullError (transient flag) or the link's own errors; nothing partial is ever marked verified."""
    local_dir, dest_root = Path(local_dir), Path(dest_root)
    man = link.read_json(f"chunks/{jid}/out/manifest.json")
    if not isinstance(man, dict) or "tar" not in man:
        raise PullError(f"{jid}: out/manifest.json not readable on the volume")
    tar = man["tar"]
    prev = read_json(local_dir / "pull.json", None)
    if isinstance(prev, dict) and prev.get("verified") and prev.get("tar_sha256") == tar["sha256"]:
        return {**prev, "already": True}
    lock = PullLock(local_dir / "pull.lock")
    lock.acquire()
    try:
        return _pull_locked(link, jid, man, local_dir, dest_root, take, keep_tar, progress, disk_factor)
    finally:
        lock.release()


def _pull_locked(link, jid: str, man: dict, local_dir: Path, dest_root: Path, take, keep_tar: bool, progress, disk_factor: float) -> dict:
    tar = man["tar"]
    free = shutil.disk_usage(dest_root if dest_root.exists() else dest_root.parent).free
    if free < tar["bytes"] * disk_factor + (256 << 20):
        raise PullError(f"{jid}: {free / 1e9:.1f} GB free is not enough for a {tar['bytes'] / 1e9:.2f} GB result", transient=True)
    t0 = time.time()
    local_dir.mkdir(parents=True, exist_ok=True)
    tar_local = local_dir / "result.tar"
    got = link.pull_file(f"chunks/{jid}/out/{tar['name']}", tar_local, size=int(tar["bytes"]), sha256=tar["sha256"], progress=progress)
    ex = extract_verified(tar_local, dest_root, man.get("files") or [])
    done = link.read_json(f"chunks/{jid}/done.json") or {}
    if not keep_tar:
        tar_local.unlink(missing_ok=True)
    take = take or (man.get("chunk") or {}).get("take") or (man.get("chunk") or {}).get("id")
    doc = {"verified": True, "id": jid, "take": take, "world_id": man.get("world_id"), "status": man.get("status"),
           "missing": man.get("missing") or [], "tar_sha256": tar["sha256"], "tar_bytes": tar["bytes"], "files": ex["files"],
           "files_skipped": ex["skipped"], "bytes": ex["bytes"], "download_s": got.get("seconds"), "seconds": round(time.time() - t0, 1),
           "pulled": time.strftime("%Y-%m-%dT%H:%M:%S"), "pod_id": man.get("pod_id") or done.get("pod_id"),
           "gpu": (done.get("gpu") or {}).get("name"), "code": done.get("code"), "done_seconds": done.get("seconds"),
           "cost_usd": done.get("cost_usd"), "dest": str(dest_root)}
    atomic_write_json(local_dir / "manifest.json", man)
    if done:
        atomic_write_json(local_dir / "done.json", done)
    if take:
        atomic_write_json(dest_root / "dancecap" / "work" / str(take) / "farm" / "pulled.json",
                          {k: doc[k] for k in ("id", "take", "world_id", "status", "tar_sha256", "files", "pulled", "pod_id", "gpu", "code")})
    atomic_write_json(local_dir / "pull.json", doc)   # last: the proof
    return doc
