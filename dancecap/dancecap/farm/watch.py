"""The farm watcher: one DETACHED local process per dance that keeps a run alive and brings its results home (FARM.md section 6).

    python -m dancecap.farm watch [start]     start it detached (idempotent: a running watcher is reported, never doubled)
    python -m dancecap.farm watch run         the same loop in the foreground (what `start` spawns; logs to farm/watcher.log)
    python -m dancecap.farm watch status      ONE json line: watcher health, jobs, pods, coverage, spend (for monitor agents)
    python -m dancecap.farm watch stop        ask it to stop after the current cycle (pods are left to their kill switches)
    python -m dancecap.farm watch ensure      start one if there is work and none runs (reboot / crash), restart a hung one
    python -m dancecap.farm wait N            block up to N s, print ONE json line when something happened (or on timeout)
    python -m dancecap.farm launch | up | pull | down        see launch.py

SURVIVABILITY. Nothing the watcher knows lives only in memory: it is rebuilt every cycle from (a) the volume (one ssh call reads
done.json / failed.json / heartbeat / claim of every job and kill_at / heartbeat of every pod), (b) Runpod's pod list, (c) the
verified-pull markers in work/<dance>/farm/chunks/<id>/pull.json and (d) state.json. Kill it, reboot the PC, start it again: it
carries on. A single-instance guard is an OS file lock (released by the OS when the process dies, so no stale-pid guessing).

Each cycle (every `poll_s`, 30 s):
    1. Runpod pod list (every 60 s) and the volume snapshot through the first pod whose ssh answers (the volume is shared, any
       pod will do; a lost link is retried on the next pod);
    2. pods: starting / running / idle / exited / lost; a pod missing from Runpod's list twice in a row whose heartbeat is also
       stale is LOST: its claims are released at once (another pod takes the job, resuming after the last finished step from
       the volume checkpoints), the ledger row is closed;
    3. jobs: queued -> running (progress + ETA from the heartbeat) -> done; a job the worker gave up on (failed.json) is put back
       in the queue up to `max_requeues` times, the pod that failed it is recycled; a job whose final dependency failed is
       marked failed (`blocked`) instead of waiting forever;
    4. PULL every done job at once (2 at a time): result.tar resumable over ssh, sha256-verified, extracted at workspace paths,
       every member hash-checked, pull.json written last; retries with backoff, never marks anything pulled that is not verified;
    5. fleet: extend kill switches of healthy busy pods (never past the hard ceiling or the budget gates), launch replacement /
       CPU analysis pods when work is waiting (budget gates, launch cap, cool-down), delete a pod only when it is idle AND every
       job it finished is pulled and verified;
    6. state.json (every cycle) and coverage.json (when something changed, at least every 5 min), local stitch if asked
       (`--stitch local`; by default the stitch is the `final` pod job), result.json when everything is terminal.
Events go to state.json (`farm events`) and the log.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from dataclasses import fields as dc_fields
from pathlib import Path

from . import DEFAULT_DANCE, chunks_dir, farm_dir, jobs_dir
from . import pull as P
from .state import State, add_event, atomic_write_json, now_iso, read_json

SCHEMA_RESULT = "dancecap.farm.result/1"
GPU_KINDS = "front,fit,refine,full,asr"   # GPU pods never claim the CPU analysis jobs (kind analyse): those run on CPU pods
CPU_KINDS = "analyse"
ACTIVE = ("queued", "running", "done")    # job states the watcher works on; planned / pulled / stitched / failed / cancelled are not
TERMINAL = ("pulled", "stitched", "failed", "cancelled")
POD_GONE_STATUSES = ("EXITED", "TERMINATED", "STOPPED", "DELETED")
GPU_ALIASES = {"4090": "NVIDIA GeForce RTX 4090", "3090": "NVIDIA GeForce RTX 3090", "l4": "NVIDIA L4", "l40": "NVIDIA L40",
               "l40s": "NVIDIA L40S", "a4500": "NVIDIA RTX A4500", "a5000": "NVIDIA RTX A5000", "a6000": "NVIDIA RTX A6000",
               "a40": "NVIDIA A40", "a4000": "NVIDIA RTX A4000", "6000ada": "NVIDIA RTX 6000 Ada Generation",
               "4000ada": "NVIDIA RTX 4000 Ada Generation", "a100": "NVIDIA A100 80GB PCIe", "a100sxm": "NVIDIA A100-SXM4-80GB",
               "h100": "NVIDIA H100 80GB HBM3", "h100pcie": "NVIDIA H100 PCIe", "h100nvl": "NVIDIA H100 NVL", "h200": "NVIDIA H200"}


# ------------------------------------------------------------------------------------------------ paths and the instance lock


def pid_path(dance: str) -> Path:
    return farm_dir(dance) / "watcher.pid"


def lock_path(dance: str) -> Path:
    return farm_dir(dance) / "watcher.lock"


def log_path(dance: str) -> Path:
    return farm_dir(dance) / "watcher.log"


def stop_path(dance: str) -> Path:
    return farm_dir(dance) / "watcher.stop"


def result_path(dance: str) -> Path:
    return farm_dir(dance) / "result.json"


def workspace_root() -> Path:
    import dancecap.farm as F   # read at call time: tests re-point F.PACKAGE_ROOT
    return F.PACKAGE_ROOT.parent


def dance_tag(dance: str) -> str:
    """Pod-name tag of a dance: hm-farm-<tag>-... (lets a restarted watcher recognise pods it launched and lost track of). The WHOLE
    name, not a prefix of it: a 10-character cut made larissa_kadu_full, larissa_kadu_38 and larissa_kadu one tag, and one watcher
    adopted and deleted another dance's running pod."""
    return re.sub(r"[^a-z0-9]+", "", dance.lower()) or "dance"


def pod_name_belongs(name: str | None, dance: str) -> bool:
    """Does a pod name look like one THIS dance's watcher launched (`hm-farm-<tag>-<stamp>-<i>`, `...-cpu-...`, `...-gateway`)? The tag must
    be followed by a delimiter, so 'larissakadu' never matches 'larissakadufull', and 't' never matches 't2'."""
    return bool(re.match(rf"^hm-farm-{re.escape(dance_tag(dance))}-(?:\d|cpu-|gateway)", name or ""))


def owner_of(dance: str) -> str:
    """The ledger `owner` of every pod a farm run for this dance creates."""
    return f"farm:{dance}"


class InstanceLock:
    """Exclusive, non-blocking OS lock on a file: held for the life of the process, released by the OS when it dies (even when
    killed), so there is never a stale lock and never a pid-reuse guess. A second holder (any process, any handle) is refused."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.fh = None

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.path, "a+b")
        try:
            fh.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.close()
            return False
        self.fh = fh
        return True

    def release(self) -> None:
        fh, self.fh = self.fh, None
        if fh is None:
            return
        with contextlib.suppress(OSError):
            fh.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        with contextlib.suppress(OSError):
            fh.close()

    def __enter__(self):
        if not self.acquire():
            raise AlreadyRunning(str(self.path))
        return self

    def __exit__(self, *exc):
        self.release()


class AlreadyRunning(RuntimeError):
    pass


def health(dance: str) -> dict:
    """Is a watcher running for this dance? Authoritative by the OS lock; the pid file adds who / since when / last heartbeat."""
    probe = InstanceLock(lock_path(dance))
    free = probe.acquire()
    if free:
        probe.release()
    info = read_json(pid_path(dance), None) or {}
    hb = float(info.get("heartbeat") or 0)
    age = round(time.time() - hb, 1) if hb else None
    running = not free
    return {"running": running, "pid": info.get("pid") if running else None, "since": info.get("started") if running else None,
            "state": info.get("state") if running else "stopped", "heartbeat_age_s": age if running else None,
            "hung": bool(running and age is not None and age > max(300.0, 6 * float(info.get("poll_s") or 30))),
            "cycle": info.get("cycle"), "log": str(log_path(dance)), "last_pid": info.get("pid")}


# ------------------------------------------------------------------------------------------------ policy


@dataclass
class Policy:
    """How the fleet behaves; persisted in state.json["policy"] so a restarted watcher (or a CLI flag given to `launch`) keeps it."""
    target_gpu_pods: int = 1          # how many GPU pods the watcher keeps busy (never more than there are waiting jobs)
    gpus: list = field(default_factory=list)   # GPU priority list (empty: pods.GPU_PRIORITY)
    minutes: float = 100.0            # initial kill-switch lead of a GPU pod (extended while healthy)
    hard_minutes: float = 240.0       # hard ceiling of a GPU pod: the kill switch is never extended past it
    cpu_vcpu: int = 8
    cpu_minutes: float = 60.0
    cpu_hard_minutes: float = 150.0
    extend_minutes: float = 30.0      # one extension step
    extend_below_min: float = 25.0    # extend when less than this is left on the kill switch
    budget_usd: float | None = None   # ceiling of THIS farm run's worst case (spent + committed + new pod); None: only the global gates
    max_requeues: int = 2             # watcher-level re-queues of a job its worker gave up on
    max_pod_launches: int = 12        # launches by the watcher per run (loop guard)
    idle_grace_s: float = 150.0       # an idle pod with everything pulled is deleted after this
    stale_hb_s: float = 600.0         # a claimed job whose heartbeat is this old is stalled: claim released, pod recycled
    stagnant_min: float = 15.0        # a GPU step (fit / refine) whose GPU sits under 5 % with no progress this long is stalled too (a hung step
                                      # keeps a fresh heartbeat: it is written by a thread); at half of it the kill switch is no longer extended
    start_timeout_s: float = 780.0    # a pod that wrote no heartbeat by then never started its worker
    auto_launch: bool = True          # the watcher may create pods (replacements, CPU analysis pod, gateway)
    keep_pods: bool = False           # finish without deleting the pods (they die at their kill switch)
    stitch: str = "pod"               # pod: the `final` job stitches on a CPU pod | local: dancecap.stitch here | none
    interim_stitch: bool = False      # pod mode: also stitch what has landed so far (stitch-N jobs on a CPU pod) while the rest still runs
    hold: bool = False                # queued with --hold: release jobs as their `after` chunks finish
    pull_workers: int = 2
    poll_s: float = 30.0
    runpod_poll_s: float = 60.0
    state_s: float = 60.0
    coverage_s: float = 60.0          # coverage.json is recomputed at least this often (about 1 s of CPU), at once after a pull
    launch_cooldown_s: float = 120.0  # after a failed launch (doubles per failure, max 15 min)
    launch_gap_s: float = 15.0        # between two successful launches
    idle_exit_s: float = 300.0        # nothing queued, no pods: the watcher exits after this

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None, base: "Policy | None" = None) -> "Policy":
        p = base or cls()
        names = {f.name for f in dc_fields(cls)}
        return cls(**{**asdict(p), **{k: v for k, v in (d or {}).items() if k in names}})

    @classmethod
    def load(cls, dance: str) -> "Policy":
        return cls.from_dict(State(dance).load().get("policy"))

    def save(self, dance: str) -> None:
        State(dance).update(lambda s: s.__setitem__("policy", self.to_dict()))


# ------------------------------------------------------------------------------------------------ backends


_LEDGER_LOCK = threading.RLock()


def install_ledger_lock() -> None:
    """remote.py updates work/runpod/ledger.json by read-modify-write without a lock. The watcher creates pods on a worker thread while
    its main thread deletes / closes others, so within THIS process those updates are serialised (a lost row would hide a live pod from
    the budget gate). Idempotent; does not (cannot) protect against other processes writing the ledger at the same instant."""
    from .. import remote
    if getattr(remote, "_farm_ledger_locked", False):
        return
    for name in ("_record", "_delete"):
        orig = getattr(remote, name)

        def wrapped(*a, __orig=orig, **k):
            with _LEDGER_LOCK:
                return __orig(*a, **k)
        wrapped.__name__ = name
        setattr(remote, name, wrapped)
    remote._farm_ledger_locked = True


class _PodRef:
    """What pods.extend_killswitch needs of the target pod: its id (the deadline file is on the shared volume)."""

    def __init__(self, pod_id: str):
        self.id = pod_id


class RunpodBackend:
    """Everything the watcher asks of Runpod and ssh. Imports are lazy so `import dancecap.farm.watch` stays light (the CLI imports
    every plug-in on every non-core command). Tests replace this with a fake."""

    def __init__(self, dance: str = DEFAULT_DANCE):
        self.dance = dance
        install_ledger_lock()

    def runpod_pods(self) -> dict[str, dict]:
        from .. import remote
        data = remote._ctl("pod", "list", check=False, timeout=90)
        rows = data if isinstance(data, list) else (data.get("pods") if isinstance(data, dict) else None)
        if not isinstance(rows, list):   # runpodctl failed or printed an error object: do NOT read that as "no pods"
            raise RuntimeError(f"pod list returned {str(data)[:200]}")
        owners = {}
        with contextlib.suppress(Exception):
            owners = {r["pod_id"]: r.get("owner") for r in remote.ledger()}
        return {p["id"]: {"id": p["id"], "name": p.get("name"), "status": p.get("desiredStatus") or p.get("status"),
                          "cost_per_hr": p.get("costPerHr"), "uptime_s": p.get("uptimeSeconds"), "owner": owners.get(p["id"])}
                for p in rows if p.get("id")}

    def make_link(self, pod: dict):
        from .. import remote
        from . import volume
        return volume.Link(remote.Pod(pod["id"], pod.get("name", ""), pod.get("gpu", ""), float(pod.get("price_hr") or 0),
                                      pod["host"], int(pod["port"])))

    def attach(self, pod_id: str) -> dict | None:
        from . import pods
        try:
            h = pods.attach(pod_id)
        except Exception:  # noqa: BLE001
            return None
        return {"host": h.host, "port": h.port}

    def launch_gpu(self, n: int, *, gpus=None, minutes=100, hard_minutes=None, max_cost=None, kinds=GPU_KINDS, prefix="hm-farm",
                   idle_min=5, dry_run=False, **kw) -> list[dict]:
        from . import pods
        return pods.launch(n, gpus=gpus or None, minutes=minutes, hard_minutes=hard_minutes, max_cost=max_cost, kinds=kinds,
                           idle_min=idle_min, prefix=prefix, dry_run=dry_run, owner=owner_of(self.dance))

    def launch_cpu(self, n: int, *, vcpu=8, minutes=60, hard_minutes=None, max_cost=None, kinds=CPU_KINDS, prefix="hm-farm-cpu",
                   dry_run=False, **kw) -> list[dict]:
        from . import pods
        return pods.launch_cpu(n, vcpu=vcpu, minutes=minutes, hard_minutes=hard_minutes, max_cost=max_cost, kinds=kinds, prefix=prefix,
                               dry_run=dry_run, owner=owner_of(self.dance))

    def offers(self, gpus=None) -> list[dict]:
        from . import pods
        return pods.gpu_offers(priority=gpus or None)

    def gateway(self, minutes: float = 30, vcpu: int = 2, reuse: bool = True, name: str | None = None):
        """(info dict, created): a small CPU pod (no worker) that mounts the volume, for pulls / uploads / `ls` while no worker pod is
        up. reuse=True returns a running farm pod instead ({'borrowed': True}, never deleted by us). A new one passes the budget gate
        first (pods.gateway itself does not)."""
        from . import pods
        if reuse:
            for q in pods.list_farm_pods():
                if str(q.get("status")).upper() == "RUNNING":
                    try:
                        h = pods.attach(q["id"])
                    except Exception:  # noqa: BLE001
                        continue
                    return {"pod_id": h.id, "name": h.name, "gpu": h.gpu, "cost_per_hr": h.cost_per_hr, "host": h.host, "port": h.port,
                            "borrowed": True}, False
        price = pods.cpu_price("cpu5c", vcpu)
        pods.gate(price * (minutes * 1.5 + pods.START_TIMEOUT_MIN) / 60)
        info = pods.create(name or f"{pods.POD_PREFIX}-gateway", cpu="cpu5c", vcpu=vcpu, minutes=minutes, hard_minutes=minutes * 1.5,
                           entry=True, autostart=False, disk_gb=20, price=price, wait=True, owner=owner_of(self.dance))
        return info, True

    def delete(self, pod_id: str, reason: str) -> None:
        from . import pods
        pods.delete(pod_id, reason=reason)

    def mark_gone(self, pod_id: str, reason: str) -> None:
        """Close the ledger row of a pod that no longer exists WITHOUT calling Runpod (nothing to delete; a delete call could only
        hit a pod that is in fact still there)."""
        from .. import remote
        with _LEDGER_LOCK, remote.ledger_lock():
            rows = remote.ledger()
            for r in rows:
                if r["pod_id"] == pod_id and not r.get("deleted_ts"):
                    r["deleted_ts"] = time.time()
                    r["deleted_reason"] = reason
                    r["cost"] = round(r["cost_per_hr"] * (r["deleted_ts"] - r["created_ts"]) / 3600, 4)
            remote._save_ledger(rows)

    def reconcile(self) -> dict:
        """Close ledger rows of pods Runpod no longer lists (two looks), stamp the others alive. Never deletes a pod."""
        from .. import remote
        return remote.reconcile_ledger(confirm_s=90.0)

    def ledger_cost(self, pod_id: str) -> float | None:
        from .. import remote
        for r in remote.ledger():
            if r["pod_id"] == pod_id:
                return float(r["cost"]) if r.get("deleted_ts") and r.get("cost") is not None else None
        return None

    def hard_ceiling(self, pod: dict) -> float | None:
        """created + max_minutes of the ledger row = the on-pod HM_HARD_CEIL (pods.create writes both)."""
        from .. import remote
        for r in remote.ledger():
            if r["pod_id"] == pod["id"]:
                return float(r["created_ts"]) + float(r["max_minutes"]) * 60
        return None

    def balance(self) -> float:
        from . import pods
        return pods.balance()

    def remaining(self) -> float:
        from .. import remote
        return remote.remaining()

    def gate(self, cost: float) -> dict:
        from . import pods
        return pods.gate(cost)

    def extend(self, pod: dict, add_minutes: float, hard_epoch: int | None, link) -> int:
        from . import pods
        return pods.extend_killswitch(_PodRef(pod["id"]), add_minutes=add_minutes, hard_epoch=hard_epoch, via=link.pod)


# ------------------------------------------------------------------------------------------------ helpers


def short_err(e: BaseException, n: int = 160) -> str:
    """An exception for a log line: the start and the end (a pod command error carries the whole command in the middle)."""
    t = " ".join(str(e).split())
    return t if len(t) <= n else t[:40] + " ... " + t[-(n - 45):]


def is_mine(rec: dict) -> bool:
    """False for a record another session owns (owner fit_full ...): the farm's watcher never touches it."""
    return (rec.get("owner") or "farm") == "farm"


def iso(t: float | None = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t if t is not None else time.time()))


def load_spec(dance: str, jid: str) -> dict:
    return read_json(jobs_dir(dance) / f"{jid}.json", None) or {}


def needs_gpu(spec: dict) -> bool:
    from .worker import needs_gpu as ng
    return ng(spec) if spec else True


def job_progress(spec: dict, hb: dict, pod_now: float) -> tuple[float, float | None]:
    """(fraction 0..1, ETA in minutes or None) of a running job from its heartbeat: steps weighted by their timeouts, the running
    step by its own progress events when it emits them, else by elapsed time against half its timeout (a guide, not a promise)."""
    steps = spec.get("steps") or []
    n = int(hb.get("steps_n") or len(steps) or 1)
    i = int(hb.get("step_i") or 0)
    w = [max(1.0, float(s.get("timeout_min") or 1)) for s in steps] if len(steps) == n else [1.0] * n
    p = hb.get("progress") or {}
    sub = None
    if p.get("total"):
        sub = min(1.0, max(0.0, float(p.get("done") or 0) / float(p["total"])))
    cur = w[i] if i < n else 0.0
    if sub is None and hb.get("step_started"):
        sub = min(0.95, ((pod_now - float(hb["step_started"])) / 60.0) / max(1.0, cur * 0.5))
    frac = min(0.999, (sum(w[:i]) + cur * (sub or 0.0)) / max(1.0, sum(w)))
    eta = None
    est = (spec.get("estimate") or {}).get("total_min")
    if hb.get("started"):
        el = max(0.0, (pod_now - float(hb["started"])) / 60.0)
        if frac > 0.05:
            eta = el * (1 - frac) / frac
        elif est:
            eta = max(0.0, 1.3 * float(est) - el)
    return round(frac, 3), (round(eta, 1) if eta is not None else None)


def compact_done(d: dict) -> dict:
    steps = {k: {"rc": v.get("rc"), "s": v.get("seconds")} for k, v in (d.get("steps") or {}).items()}
    return {"seconds": d.get("seconds"), "cost_usd": d.get("cost_usd"), "price_hr": d.get("price_hr"),
            "gpu": (d.get("gpu") or {}).get("name"), "pod": d.get("pod_id"), "attempt": d.get("attempt"),
            "out_files": d.get("out_files"), "out_bytes": d.get("out_bytes"), "steps": steps, "world_id": d.get("world_id"),
            "finished": iso(d["finished"]) if d.get("finished") else None}


def record_pod(state: State, r: dict, role: str, by: str = "watch", now: float | None = None) -> dict:
    """Put a pod that pods.create / pods.launch just returned into state.json (the watcher manages exactly the pods recorded here)."""
    now = now if now is not None else time.time()
    fields = dict(id=r["pod_id"], name=r.get("name"), role=role, gpu=r.get("gpu"), price_hr=float(r.get("cost_per_hr") or 0),
                  state="starting", host=r.get("host"), port=r.get("port"), created_ts=r.get("t_create") or now,
                  created=iso(r.get("t_create") or now), kill_at=r.get("kill_at"), hard_ceil=r.get("hard_ceil"), minutes=r.get("minutes"),
                  hard_minutes=r.get("hard_minutes"), dc=r.get("dc"), launched_by=by, extensions=0, jobs=[], idle_since=None,
                  borrowed=bool(r.get("borrowed")))
    p = state.upsert_pod(r["pod_id"], **fields)
    state.event("launch", f"{role} pod {r['pod_id']} {r.get('gpu')} ${float(r.get('cost_per_hr') or 0):.2f}/h up "
                          f"(kill switch +{r.get('minutes')} min, ceiling +{r.get('hard_minutes')} min, by {by})", pod=r["pod_id"])
    return p


def mark_pod_deleted(state: State, pid: str, reason: str, cost: float | None, now: float | None = None) -> None:
    now = now if now is not None else time.time()

    def f(s):
        q = s["pods"].get(pid)
        if q is None:
            return
        q["state"] = "deleted"
        q["deleted"] = now_iso()
        q["deleted_reason"] = reason
        q["cost_usd"] = cost if cost is not None else round(float(q.get("price_hr") or 0) * (now - float(q.get("created_ts") or now)) / 3600, 3)
        q["jobs"] = []
        add_event(s, "pod", f"pod {pid} deleted ({reason}), ${q['cost_usd']}", pod=pid)
    state.update(f)


def farm_cost(st: dict, now: float, hard_of=lambda p: p.get("hard_ceil")) -> tuple[float, float]:
    """(spent so far by this farm's pods, worst case still committed on its live pods) from state.json."""
    spent = committed = 0.0
    for p in st["pods"].values():
        price = float(p.get("price_hr") or 0)
        t0 = float(p.get("created_ts") or now)
        if p.get("state") in ("deleted", "gone", "lost"):
            spent += float(p.get("cost_usd") or 0)
        else:
            spent += price * (now - t0) / 3600
            hard = hard_of(p) or (t0 + float(p.get("hard_minutes") or 0) * 60)
            committed += price * max(0.0, float(hard) - now) / 3600
    return spent, committed


@dataclass
class View:
    """One cycle's picture of the world."""
    now: float
    st: dict
    snap: dict | None
    rp: dict | None
    rp_fresh: bool
    tracked: dict
    skew: float = 0.0        # pod clock - local clock

    def pod_now(self) -> float:
        return float(self.snap["t"]) if self.snap else self.now + self.skew


# ------------------------------------------------------------------------------------------------ the watcher


class Recorder:
    """What the watcher, `farm pull` and `farm down` share: events, the job-status bookkeeping in state.json, and what happens when a
    pull has been verified. Light: no Runpod, no threads."""

    def __init__(self, dance: str, clock=time.time, log=None):
        self.dance = dance
        self.state = State(dance)
        self.clock = clock
        self._log = log or (lambda m: print(f"[watch {time.strftime('%H:%M:%S')}] {m}", flush=True))
        self.warned: set[str] = set()
        self.specs: dict[str, tuple[float, dict]] = {}
        self.cov_dirty = True
        self.stats = {"pulls": 0, "pull_bytes": 0, "pull_failures": 0, "requeues": 0, "pods_launched": 0, "pods_deleted": 0,
                      "pods_lost": 0, "extensions": 0, "errors": 0}

    def log(self, msg: str) -> None:
        with contextlib.suppress(Exception):
            self._log(msg)

    def event(self, kind: str, msg: str, **data) -> None:
        self.log(f"{kind}: {msg}")
        with contextlib.suppress(Exception):
            self.state.event(kind, msg, **data)

    def warn_once(self, key: str, msg: str, **data) -> None:
        if key not in self.warned:
            self.warned.add(key)
            self.event("warn", msg, **data)

    def spec(self, jid: str) -> dict:
        p = jobs_dir(self.dance) / f"{jid}.json"
        try:
            m = p.stat().st_mtime
        except OSError:
            return {}
        hit = self.specs.get(jid)
        if hit and hit[0] == m:
            return hit[1]
        d = read_json(p, None) or {}
        self.specs[jid] = (m, d)
        return d

    def rec_of(self, st: dict, jid: str) -> dict | None:
        return st["chunks"].get(jid) or st["jobs"].get(jid)

    def apply(self, s: dict, jid: str, status: str | None = None, **fields) -> None:
        """Set a job's status / fields inside an open state dict (chunk record and the job mirror)."""
        c = s["chunks"].get(jid)
        j = s["jobs"].setdefault(jid, {"id": jid})
        rec = c if c is not None else j
        if status and status != rec.get("status"):
            rec["status_ts"] = now_iso()
            add_event(s, "chunk" if c is not None else "job", f"{jid}: {rec.get('status', 'new')} -> {status}", chunk=jid, status=status)
        if status:
            rec["status"] = status
            j["status"] = status
        rec.update(fields)
        if c is not None:
            j.update({k: v for k, v in fields.items() if k in ("pod", "error", "pulled", "progress", "eta_min")})

    def set_status(self, jid: str, status: str | None = None, **fields) -> None:
        self.state.update(lambda s: self.apply(s, jid, status, **fields))

    def pull_done(self, jid: str, res: dict, quiet: bool = False) -> None:
        self.stats["pulls"] += 0 if res.get("already") else 1
        self.stats["pull_bytes"] += 0 if res.get("already") else int(res.get("tar_bytes") or 0)
        world = res.get("world_id")
        info = {"state": "ok", "files": res.get("files"), "bytes": res.get("tar_bytes"), "seconds": res.get("seconds"),
                "pulled": res.get("pulled"), "attempts": 0, "error": None, "world_id": world}
        self.set_status(jid, "pulled", pulled=res.get("pulled") or now_iso(), pull=info, world_id=world, progress=1.0, eta_min=0)
        if not quiet:
            self.event("pull", f"{jid} pulled and verified: {res.get('files')} files, {int(res.get('tar_bytes') or 0) / 1e6:.1f} MB "
                               f"in {res.get('seconds')} s (world {world})", chunk=jid)
        self.cov_dirty = True
        self.after_pull(jid, res)

    def after_pull(self, jid: str, res: dict) -> None:
        spec = self.spec(jid)
        if spec.get("what") == "stitch_analyse" or jid == "final" or spec.get("stitch_chunks"):
            ids = [c for c in (spec.get("chunks") or []) if (self.rec_of(self.state.load(), c) or {}).get("status") == "pulled"]
            if ids:
                def f(s):
                    for c in ids:
                        self.apply(s, c, "stitched")
                self.state.update(f)
                self.event("stitch", f"{jid} stitched {len(ids)} chunk(s): {', '.join(ids)}")
        # world consistency: a chunk of another pose id must never reach the stitch
        try:
            from . import world as W
            cur = W.current(self.dance)
            if cur and res.get("world_id") and res["world_id"] != cur.get("world_id"):
                W.check_same_world([{"world_id": res["world_id"], "id": jid}, {"world_id": cur["world_id"], "id": "current"}], self.dance)
        except Exception as e:  # noqa: BLE001  WorldMismatch
            self.set_status(jid, None, world_mismatch=True, error=f"world mismatch: {str(e)[:200]}")
            self.event("warn", f"{jid}: {str(e)[:200]}", chunk=jid)


class Watcher(Recorder):
    def __init__(self, dance: str, backend=None, *, policy: Policy | None = None, dest_root: Path | None = None, clock=time.time,
                 once: bool = False, max_hours: float | None = None, log=None):
        super().__init__(dance, clock, log)
        self.backend = backend or RunpodBackend(dance)
        self.dest = Path(dest_root) if dest_root else workspace_root()
        self.once = once
        self.max_hours = max_hours
        if policy is not None:
            policy.save(dance)
        self.policy = Policy.load(dance)
        self.pool = ThreadPoolExecutor(max_workers=max(2, self.policy.pull_workers) + 2, thread_name_prefix="farm-watch")
        self.pulls: dict[str, Future] = {}
        self.pull_prog: dict[str, dict] = {}
        self.actions: dict[str, Future] = {}
        self.link = None
        self.link_pod: str | None = None
        self.link_fail: dict[str, int] = {}
        self.rp: dict | None = None
        self.rp_t = 0.0
        self.rp_fail = 0
        self.missing: dict[str, int] = {}
        self.recycle: set[str] = set()
        self.cycles = 0
        self.t_start = clock()
        self.seen_active = False
        self.idle_since: float | None = None
        self.finished = False
        self.finish_reason = ""
        self.cov_t = 0.0
        self.state_t = 0.0
        self.spend_t = 0.0
        self.next_launch_at = 0.0
        self.launch_failures = 0
        self.launches = 0
        self.gw_launches = 0
        self.skew = 0.0
        self.last_stitch_sig = ""
        self.interim_retry_at = 0.0
        self.stuck_pods = 0
        self.consec_errors = 0
        self.hb_miss: dict[str, int] = {}
        self.stag: dict[str, dict] = {}        # job id -> stagnation tracker (see sync_jobs)
        self.delete_fail: dict[str, tuple[int, float]] = {}   # pod id -> (failures, next try)
        self.reconcile_t = 0.0
        self.warned_unknown: set[str] = set()
        self.relisted: dict[str, int] = {}

    # ---------------------------------------------------------------------------------------------- tracked work
    def tracked(self, st: dict) -> dict[str, dict]:
        """The jobs this farm works on: queued / running / done and not owned by another session (the monitor marks the chunks a live
        fit_full session is fitting `running` with owner fit_full: they are none of the watcher's business)."""
        out: dict[str, dict] = {}
        for cid, c in st["chunks"].items():
            if c.get("status") in ACTIVE and is_mine(c):
                out[cid] = {"kind": "chunk", "rec": c}
        for jid, j in st["jobs"].items():
            if jid not in st["chunks"] and j.get("status") in ACTIVE and is_mine(j):
                out[jid] = {"kind": "job", "rec": j}
        return out

    def alive_pods(self, st: dict) -> list[dict]:
        return [p for p in st["pods"].values() if p.get("state") not in ("deleted", "gone", "lost")]

    # ---------------------------------------------------------------------------------------------- 1. runpod + link + snapshot
    def poll_runpod(self, now: float) -> tuple[dict | None, bool]:
        if self.rp is not None and now - self.rp_t < self.policy.runpod_poll_s:
            return self.rp, False
        try:
            self.rp = self.backend.runpod_pods()
            self.rp_t = now
            if self.rp_fail >= 3:
                self.event("fleet", "Runpod API reachable again")
            self.rp_fail = 0
            return self.rp, True
        except Exception as e:  # noqa: BLE001
            self.rp_fail += 1
            self.log(f"runpod pod list failed ({self.rp_fail}): {type(e).__name__}: {str(e)[:160]}")
            if self.rp_fail == 3:
                self.event("warn", f"Runpod API unreachable for {self.rp_fail} polls; pod states are frozen until it answers")
            return None, False

    def link_candidates(self, st: dict) -> list[dict]:
        pods = [p for p in self.alive_pods(st) if p.get("host") and p.get("port") and p.get("state") != "stuck"]
        pods.sort(key=lambda p: (p["id"] != self.link_pod, self.link_fail.get(p["id"], 0), p.get("created_ts") or 0))
        return pods

    def snapshot(self, st: dict, tracked: dict) -> dict | None:
        ids = list(dict.fromkeys([*tracked, *self.pulls]))
        deps = list(dict.fromkeys(d for jid in tracked for d in (self.spec(jid).get("after") or []) if d not in ids))
        pod_ids = [p["id"] for p in self.alive_pods(st)]
        cands = self.link_candidates(st)
        for p in cands[:2]:   # at most two ssh attempts per cycle (45 s each): the heartbeat the monitor watches must stay fresh
            t0 = self.clock()
            try:
                lk = self.link if (self.link is not None and self.link_pod == p["id"]) else self.backend.make_link(p)
                snap = P.take_snapshot(lk, ids, pod_ids, deps=deps, timeout=45)
            except Exception as e:  # noqa: BLE001
                self.link_fail[p["id"]] = self.link_fail.get(p["id"], 0) + 1
                self.log(f"snapshot via {p['id']} failed ({self.link_fail[p['id']]}): {type(e).__name__}: {short_err(e)}")
                if self.link_pod == p["id"]:
                    self.link, self.link_pod = None, None
                if self.link_fail[p["id"]] == 3:   # maybe the host/port moved
                    fresh = self.backend.attach(p["id"])
                    if fresh and (fresh["host"], fresh["port"]) != (p.get("host"), p.get("port")):
                        self.state.upsert_pod(p["id"], host=fresh["host"], port=fresh["port"])
                        self.event("fleet", f"pod {p['id']} ssh endpoint changed to {fresh['host']}:{fresh['port']}")
                continue
            if self.link_pod != p["id"]:
                self.log(f"volume link via pod {p['id']}")
            self.link, self.link_pod = lk, p["id"]
            self.link_fail[p["id"]] = 0
            self.skew = float(snap["t"]) - (t0 + self.clock()) / 2
            return snap
        self.link, self.link_pod = None, None
        return None

    # ---------------------------------------------------------------------------------------------- 2. pods
    def sync_pods(self, v: View) -> None:
        upd: dict[str, dict] = {}
        lost: list[str] = []
        for pid, p in v.st["pods"].items():
            if p.get("state") in ("deleted", "gone", "lost"):
                continue
            ps = v.snap["pods"].get(pid) if v.snap else None
            hb = (ps or {}).get("heartbeat")
            fields: dict = {}
            age = v.now - float(p.get("created_ts") or v.now)
            fields["age_min"] = round(age / 60, 1)
            fields["spend_usd"] = round(float(p.get("price_hr") or 0) * age / 3600, 3)
            R = v.rp.get(pid) if v.rp is not None else None
            if v.rp is not None and v.rp_fresh:
                gone = R is None or str(R.get("status") or "").upper() in POD_GONE_STATUSES
                self.missing[pid] = self.missing.get(pid, 0) + 1 if gone else 0
            if R is not None:
                fields["runpod_status"] = R.get("status")
            if ps is not None:
                hb_err = bool(ps.get("heartbeat_err"))   # the file exists but could not be read: unknown, NOT absent
                if ps.get("kill_at"):
                    fields["kill_at"] = ps["kill_at"]
                    fields["kill_in_min"] = round((ps["kill_at"] - v.pod_now()) / 60, 1)
                fresh_hb = bool(hb) and v.pod_now() - float(hb.get("t") or 0) < 120
                if hb:
                    fields["hb_age_s"] = round(v.pod_now() - float(hb.get("t") or 0), 1)
                    fields["worker"] = hb.get("state")
                    fields["hb_seen"] = True        # this pod has run a worker: it can never be judged "never started" again
                    self.hb_miss.pop(pid, None)
                if ps.get("events"):
                    fields["last_event"] = str(ps["events"][-1])[:160]
                if self.missing.get(pid, 0) >= 2 and not fresh_hb:
                    lost.append(pid)
                    continue
                if p.get("role") == "gateway":
                    fields["state"] = "idle"
                elif ps.get("worker_exit") is not None and (hb or {}).get("state") != "running":
                    fields["state"] = "exited"
                elif not hb:
                    # No heartbeat. "stuck" (= the worker never started: the pod is deleted) is only for a pod that has NEVER written one,
                    # and only after several silent snapshots in a row: one failed read (EIO / ESTALE on the network volume) in the last
                    # minutes of a 17-minute job used to destroy the running fit.
                    if hb_err:
                        self.warn_once(f"hb-err:{pid}", f"pod {pid}: heartbeat file unreadable on the volume; state left as it was", pod=pid)
                    else:
                        miss = self.hb_miss[pid] = self.hb_miss.get(pid, 0) + 1
                        if not p.get("hb_seen"):
                            fields["state"] = "stuck" if (age > self.policy.start_timeout_s and miss >= 3) else "starting"
                        else:
                            fields["hb_missing"] = miss
                else:   # a long heartbeat gap of a running worker is judged per job (stalled), not per pod
                    st_ = hb.get("state")
                    fields["state"] = "running" if st_ == "running" else "exited" if st_ == "exited" else "idle"
                    if p.get("state") == "starting" and fields["state"] in ("idle", "running"):
                        self.stuck_pods = 0   # a pod that started its worker clears the strikes
            elif self.missing.get(pid, 0) >= 2 and v.snap is None and self.link is None:
                lost.append(pid)   # no volume view at all and Runpod says it is gone twice
                continue
            upd[pid] = fields
        # jobs that run on each pod, idle tracking
        running_on: dict[str, list[str]] = {}
        for jid, t in v.tracked.items():
            if t["rec"].get("status") == "running" and t["rec"].get("pod"):
                running_on.setdefault(t["rec"]["pod"], []).append(jid)

        def f(s):
            for pid, fields in upd.items():
                p = s["pods"].get(pid)
                if p is None:
                    continue
                if "state" in fields and fields["state"] != p.get("state"):
                    add_event(s, "pod", f"pod {pid}: {p.get('state', 'new')} -> {fields['state']}", pod=pid)
                p.update(fields)
                p["jobs"] = running_on.get(pid, [])
                busy = bool(p["jobs"]) or p.get("state") in ("starting",)
                if busy:
                    p["idle_since"] = None
                elif not p.get("idle_since"):
                    p["idle_since"] = v.now
        self.state.update(f)
        for pid in lost:
            self.on_pod_lost(v, pid)

    def on_pod_lost(self, v: View, pid: str) -> None:
        """A pod that Runpod no longer lists (kill switch, preemption, deleted by hand) and that stopped writing heartbeats."""
        self.stats["pods_lost"] += 1
        self.event("pod", f"pod {pid} is LOST (not in Runpod's list, heartbeat stale)", pod=pid)
        for jid, t in v.tracked.items():
            rec = t["rec"]
            E = v.snap["jobs"].get(jid) if v.snap else None
            owner = ((E or {}).get("claim") or {}).get("pod_id") or rec.get("pod")
            if owner == pid and rec.get("status") == "running":
                released = False
                if self.link is not None:
                    with contextlib.suppress(Exception):
                        released = P.release_claim(self.link, jid, pid)
                self.set_status(jid, "queued", pod=None, lost_count=int(rec.get("lost_count") or 0) + 1, progress=rec.get("progress"),
                                note=f"pod {pid} lost; claim {'released' if released else 'left to go stale'}")
        R = v.rp.get(pid) if v.rp else None
        if R is not None:   # Runpod still lists it (EXITED / TERMINATED): it lingers in the account until it is deleted
            with contextlib.suppress(Exception):
                self.backend.delete(pid, f"lost: Runpod lists it as {R.get('status')}")
        else:
            with contextlib.suppress(Exception):
                self.backend.mark_gone(pid, "lost (not in Runpod's list)")
        cost = None
        with contextlib.suppress(Exception):
            cost = self.backend.ledger_cost(pid)

        def f(s):
            q = s["pods"][pid]
            q["state"] = "lost"
            q["deleted"] = now_iso()
            q["cost_usd"] = cost if cost is not None else round(float(q.get("price_hr") or 0) * (v.now - float(q.get("created_ts") or v.now)) / 3600, 3)
            add_event(s, "pod", f"pod {pid}: lost", pod=pid)
        self.state.update(f)
        if self.link_pod == pid:
            self.link, self.link_pod = None, None

    # ---------------------------------------------------------------------------------------------- 3. jobs
    def sync_jobs(self, v: View) -> None:
        if not v.snap:
            return
        pod_now = v.pod_now()
        changes: list[tuple] = []   # (jid, status|None, fields)
        for jid, t in v.tracked.items():
            rec = t["rec"]
            s = rec.get("status")
            E = v.snap["jobs"].get(jid)
            if E is None or E.get("read_err"):   # a state file that exists but could not be read this time: judge it next cycle
                continue
            if not E.get("in_queue") and not E.get("dir") and s in ("queued", "running"):
                self.warn_once(f"gone:{jid}", f"job {jid} is not on the volume (no queue/{jid}.json, no chunks/{jid}/): "
                                              "queue it again (farm queue)", chunk=jid)
                continue
            if E.get("done"):
                if s != "done":
                    d = E["done"]
                    changes.append((jid, "done", {"done": compact_done(d), "pod": d.get("pod_id"), "progress": 1.0, "eta_min": 0,
                                                  "step": None, "world_id": d.get("world_id") or rec.get("world_id"),
                                                  "finished": now_iso(), "error": None}))
                    self.cov_dirty = True
                continue
            if E.get("failed"):
                continue   # handle_failed
            hb = E.get("heartbeat")
            claim = E.get("claim")
            if claim is not None and hb:
                age = pod_now - float(hb.get("t") or 0)
                frac, eta = job_progress(self.spec(jid), hb, pod_now)
                pod = claim.get("pod_id") or hb.get("pod_id")
                f = {"pod": pod, "progress": frac, "eta_min": eta, "step": hb.get("step"), "step_i": hb.get("step_i"),
                     "steps_n": hb.get("steps_n"), "attempt": hb.get("attempt"), "hb_age_s": round(age, 1),
                     "gpu_util": (hb.get("gpu") or {}).get("util"), "gpu": (hb.get("gpu") or {}).get("name"),
                     "stalled": age > self.policy.stale_hb_s}
                stag_min = self.track_stagnation(jid, self.spec(jid), hb)
                f["stagnant_min"] = stag_min
                if stag_min >= self.policy.stagnant_min:   # alive heartbeat (a thread writes it) but the step does nothing
                    f["stalled"] = True
                    f["stall_reason"] = f"step {hb.get('step')} idle for {stag_min:.0f} min (GPU < 5 %, no progress)"
                if s != "running":
                    f["started"] = now_iso()
                changes.append((jid, "running" if s != "running" else None, f))
            elif claim is not None:
                changes.append((jid, "running" if s != "running" else None, {"pod": claim.get("pod_id")}))
            elif s == "running":
                changes.append((jid, "queued", {"pod": None, "progress": rec.get("progress"), "step": None, "note": "released"}))
            lf = E.get("last_failure")
            if lf and claim is None and not E.get("done"):
                if lf.get("pod_id"):
                    self.recycle.add(lf["pod_id"])   # its worker will not retry a job it failed itself: replace the pod
                changes.append((jid, None, {"last_error": lf.get("error"), "last_failed_on": lf.get("pod_id")}))
        # a queued job whose dependency can never finish
        for jid, t in v.tracked.items():
            rec = t["rec"]
            if rec.get("status") != "queued":
                continue
            deps = self.spec(jid).get("after") or []
            bad = [d for d in deps if (self.rec_of(v.st, d) or {}).get("status") in ("failed", "cancelled")]
            if bad:
                changes.append((jid, "failed", {"error": f"blocked: upstream {', '.join(bad)} failed", "blocked": True}))
                continue
            orphan = [d for d in deps if d not in v.tracked and not (v.snap["jobs"].get(d) or {}).get("done")
                      and ((self.rec_of(v.st, d) or {}).get("status") in (None, "planned") or not is_mine(self.rec_of(v.st, d) or {}))]
            if orphan:   # e.g. a stitch job that waits for chunks fitted outside the farm: it needs their motion pulled locally + a re-plan
                self.warn_once(f"orphan:{jid}:{','.join(orphan)}", f"job {jid} runs after {', '.join(orphan)}, which no farm job will produce "
                               "(planned only / owned by another session): re-plan it once their results are local (farm plan, farm queue)", chunk=jid)
        if not changes:
            return

        def f(s):
            for jid, status, fields in changes:
                self.apply(s, jid, status, **fields)
        self.state.update(f)
        for jid, status, fields in changes:
            if status == "failed" and fields.get("blocked"):
                self.event("chunk", f"{jid}: {fields['error']}", chunk=jid)

    GPU_STEPS = ("fit", "refine")

    def track_stagnation(self, jid: str, spec: dict, hb: dict) -> float:
        """Minutes the running GPU step has been idle: GPU utilisation under 5 % in every observation since `since`, and the step's
        progress counter has not moved. Sparse observations (30 s) of a healthy fit (94 % util) cannot produce a streak, a deadlocked or
        hung one does. 0 for steps that are not GPU-bound or when anything moved."""
        step = str(hb.get("step") or "")
        util = (hb.get("gpu") or {}).get("util")
        prog = hb.get("progress") or {}
        sig = (step, hb.get("step_started"), prog.get("done"), prog.get("ts"))
        tr = self.stag.get(jid)
        low = util is not None and util < 5 and needs_gpu(spec) and step.split("_")[0] in self.GPU_STEPS
        now = self.clock()
        if tr is None or tr["sig"] != sig or not low:
            self.stag[jid] = {"sig": sig, "since": now if low else None}
            return 0.0
        if tr["since"] is None:
            tr["since"] = now
        return round((now - tr["since"]) / 60.0, 3)

    def handle_failed(self, v: View) -> None:
        if not v.snap or self.link is None:
            return
        for jid, t in v.tracked.items():
            rec = t["rec"]
            E = v.snap["jobs"].get(jid)
            if not E or not E.get("failed") or E.get("done"):
                continue
            n = int(rec.get("requeues") or 0)
            err = (E["failed"].get("error") or "")[:300]
            fpod = E["failed"].get("pod_id")
            # circuit breaker: the same error on two different pods is the job, not the pod. Re-queueing it again only recycles pods
            # (about $0.02 each) and re-runs a fit that fails the same way (about $0.25 each).
            sig = re.sub(r"0x[0-9a-f]+|\d+", "#", err.lower())[:120]
            seen = list(rec.get("fail_sigs") or [])
            if sig and not any(x.get("pod") == fpod and x.get("sig") == sig for x in seen):
                seen.append({"pod": fpod, "sig": sig})
            same = {x.get("pod") for x in seen if x.get("sig") == sig}
            if len(same) >= 2:
                self.set_status(jid, "failed", error=f"deterministic failure, the same error on {len(same)} pods: {err[:200]}", pod=None, fail_sigs=seen)
                self.event("chunk", f"{jid} FAILED for good: the same error on {len(same)} different pods ({err[:100]}); not re-queued", chunk=jid)
                self.cov_dirty = True
                if fpod:
                    self.recycle.add(fpod)
                continue
            if n < self.policy.max_requeues:
                try:
                    P.requeue_failed(self.link, jid)
                except Exception as e:  # noqa: BLE001
                    self.log(f"requeue of {jid} failed: {e}")
                    continue
                self.stats["requeues"] += 1
                if fpod:
                    self.recycle.add(fpod)
                self.set_status(jid, "queued", requeues=n + 1, last_error=err, last_failed_on=fpod, pod=None, progress=0.0, step=None,
                                note=f"re-queued after: {err[:120]}", fail_sigs=seen)
                self.event("chunk", f"{jid} failed on {fpod} ({err[:100]}); re-queued {n + 1}/{self.policy.max_requeues}", chunk=jid)
            else:
                self.set_status(jid, "failed", error=err, pod=None)
                self.event("chunk", f"{jid} FAILED for good after {n} re-queues: {err[:140]}", chunk=jid)
                self.cov_dirty = True
        # stalled: claimed, pod alive per Runpod, heartbeat dead for stale_hb_s
        for jid, t in v.tracked.items():
            rec = t["rec"]
            if rec.get("status") == "running" and rec.get("stalled") and rec.get("pod"):
                pid = rec["pod"]
                n = int(rec.get("stall_count") or 0) + 1
                self.event("chunk", f"{jid} stalled on {pid} ({rec.get('stall_reason') or 'no heartbeat for %.0f s' % self.policy.stale_hb_s}): "
                                    "claim released, pod recycled", chunk=jid)
                with contextlib.suppress(Exception):
                    P.release_claim(self.link, jid, pid)
                self.set_status(jid, "queued" if n <= 2 else "failed", stall_count=n, pod=None, stalled=False, stall_reason=None,
                                stagnant_min=0, error=None if n <= 2 else "stalled three times")
                self.stag.pop(jid, None)
                self.recycle.add(pid)
                self.delete_pod(pid, "stalled job", force=True)

    # ---------------------------------------------------------------------------------------------- 4. pulls
    def start_pulls(self, v: View) -> None:
        if self.link is None:
            return
        for jid, t in v.tracked.items():
            rec = t["rec"]
            if rec.get("status") != "done" or jid in self.pulls:
                continue
            if len(self.pulls) >= self.policy.pull_workers:
                break
            if self.clock() < float((rec.get("pull") or {}).get("next_at") or 0):
                continue
            E = (v.snap or {}).get("jobs", {}).get(jid) or {}
            tar = ((E.get("manifest") or {}).get("tar") or {})
            local = chunks_dir(self.dance) / jid
            prev = read_json(local / "pull.json", None)
            if isinstance(prev, dict) and prev.get("verified") and tar and prev.get("tar_sha256") == tar.get("sha256"):
                self.pull_done(jid, {**prev, "already": True}, quiet=True)   # verified before a restart
                continue
            take = (self.spec(jid).get("take")) or None
            self.pulls[jid] = self.pool.submit(self._pull_task, jid, self.link, local, take)
            self.set_status(jid, None, pull={**(rec.get("pull") or {}), "state": "pulling", "started": now_iso()})
            self.log(f"pulling {jid} via {self.link_pod}")

    def _pull_task(self, jid: str, link, local: Path, take: str | None) -> dict:
        def prog(rel, pos, size):
            self.pull_prog[jid] = {"pos": pos, "size": size, "t": time.time()}
        return P.pull_job(link, jid, local, self.dest, take=take, progress=prog)

    def publish_pull_progress(self) -> None:
        rows = {jid: dict(self.pull_prog[jid]) for jid in self.pulls if jid in self.pull_prog}
        if not rows:
            return

        def f(s):
            for jid, pr in rows.items():
                rec = s["chunks"].get(jid) or s["jobs"].get(jid)
                if rec is not None:
                    pl = dict(rec.get("pull") or {})
                    pl.update(state="pulling", bytes=pr["pos"], total=pr["size"])
                    rec["pull"] = pl
        self.state.update(f)

    def collect(self) -> None:
        for jid, fut in list(self.pulls.items()):
            if not fut.done():
                continue
            del self.pulls[jid]
            self.pull_prog.pop(jid, None)
            try:
                res = fut.result()
            except BaseException as e:  # noqa: BLE001
                self.pull_failed(jid, e)
            else:
                self.pull_done(jid, res)
        for name, fut in list(self.actions.items()):
            if not fut.done():
                continue
            del self.actions[name]
            try:
                fut.result()
            except BaseException as e:  # noqa: BLE001
                self.event("error", f"background action {name} failed: {type(e).__name__}: {str(e)[:200]}")

    def pull_failed(self, jid: str, e: BaseException) -> None:
        self.stats["pull_failures"] += 1
        rec = self.rec_of(self.state.load(), jid) or {}
        pl = dict(rec.get("pull") or {})
        n = int(pl.get("attempts") or 0) + 1
        transient = getattr(e, "transient", True)
        delay = min(300.0, 10.0 * 2 ** (n - 1)) if transient else 600.0
        bad_sha = "sha256 mismatch" in str(e) or "sha256" in str(e) and "manifest" in str(e)
        pl["bad_sha"] = int(pl.get("bad_sha") or 0) + 1 if bad_sha else 0
        pl.update(state="error", attempts=n, error=f"{type(e).__name__}: {str(e)[:200]}", next_at=self.clock() + delay)
        self.set_status(jid, None, pull=pl)
        if pl["bad_sha"] >= 3:
            self.quarantine_corrupt(jid, rec, str(e))
            return
        if n in (1, 3, 6) or n % 10 == 0 or not transient:
            self.event("pull", f"{jid}: pull attempt {n} failed ({type(e).__name__}: {str(e)[:140]}); retry in {delay:.0f} s", chunk=jid)

    def quarantine_corrupt(self, jid: str, rec: dict, why: str) -> None:
        """Three pulls in a row (each already six downloads) delivered bytes that do not match the sha256 the worker recorded: the copy on
        the volume itself is bad and no retry will fix it. Move it aside and re-run the job (counts against max_requeues), or fail it."""
        n = int(rec.get("requeues") or 0)
        if self.link is None:
            return
        if n >= self.policy.max_requeues:
            self.set_status(jid, "failed", error=f"result on the volume is corrupt (sha256 never matches): {why[:160]}", pod=None)
            self.event("chunk", f"{jid} FAILED: its result on the volume is corrupt and was already re-run {n}x", chunk=jid)
            self.cov_dirty = True
            return
        try:
            P.quarantine_result(self.link, jid)
        except Exception as e:  # noqa: BLE001
            self.log(f"quarantine of {jid} failed: {short_err(e)}")
            return
        self.stats["requeues"] += 1
        self.set_status(jid, "queued", requeues=n + 1, pod=None, progress=0.0, step=None, pull={"state": "idle", "attempts": 0, "bad_sha": 0},
                        note="result corrupt on the volume: moved aside, job re-run")
        self.event("chunk", f"{jid}: result on the volume is corrupt ({why[:100]}); moved aside, job re-queued {n + 1}/{self.policy.max_requeues}",
                   chunk=jid)

    # ---------------------------------------------------------------------------------------------- 5. fleet
    def hard_ceiling(self, p: dict) -> float | None:
        if p.get("hard_ceil"):
            return float(p["hard_ceil"])
        with contextlib.suppress(Exception):
            return self.backend.hard_ceiling(p)
        return None

    def farm_cost(self, st: dict, now: float) -> tuple[float, float]:
        return farm_cost(st, now, self.hard_ceiling)

    def extend_killswitches(self, v: View) -> None:
        if not v.snap or self.link is None:
            return
        granted = 0.0   # what this cycle's extensions already add up to: each is checked against the room that is LEFT, not against all of it
        for p in self.alive_pods(v.st):
            ps = v.snap["pods"].get(p["id"]) or {}
            kill_at = ps.get("kill_at")
            if not kill_at:
                continue
            busy = bool(p.get("jobs")) or (p.get("state") == "idle" and p.get("role") in ("gpu", "cpu") and self.wanted_by(v, p)) \
                or (p["id"] == self.link_pod and bool(v.tracked))   # the pod we read / pull through must outlive the work
            if not busy:
                continue
            left = kill_at - v.pod_now()
            if left > self.policy.extend_below_min * 60:
                continue
            hard = self.hard_ceiling(p)
            hard_pod = (hard + self.skew) if hard else None
            if hard_pod is not None and hard_pod - kill_at <= 60:
                self.warn_once(f"ceiling:{p['id']}", f"pod {p['id']}: kill switch is at its hard ceiling "
                                                      f"({(hard_pod - v.pod_now()) / 60:.0f} min left); not extended", pod=p["id"])
                continue
            hb = (ps.get("heartbeat") or {})
            if any(float((self.rec_of(v.st, j) or {}).get("stagnant_min") or 0) >= self.policy.stagnant_min / 2 for j in p.get("jobs") or []):
                self.warn_once(f"stagnant:{p['id']}", f"pod {p['id']}: its step makes no progress (GPU idle); kill switch not extended", pod=p["id"])
                continue
            if p.get("jobs") and v.pod_now() - float(hb.get("t") or 0) > 300:
                self.warn_once(f"unhealthy:{p['id']}", f"pod {p['id']}: worker heartbeat is {v.pod_now() - float(hb.get('t') or 0):.0f} s old; "
                                                       "kill switch not extended", pod=p["id"])
                continue
            add = self.policy.extend_minutes
            price = float(p.get("price_hr") or 0)
            try:
                # the extension stays inside the pod's own hard ceiling, which its creation already reserved in the ledger: what is
                # checked is that the money for the next step really exists (cap left, balance - $2) and the farm's own budget holds
                need = price * add / 60.0
                room = min(self.backend.remaining(), self.backend.balance() - 2.0) - granted
                if need > room:
                    raise RuntimeError(f"${need:.2f} for {add:.0f} more minutes > min(cap left, balance - $2) = ${room:.2f}"
                                       + (f" after ${granted:.2f} of other extensions this cycle" if granted else ""))
                if self.policy.budget_usd is not None:
                    spent, _ = self.farm_cost(v.st, v.now)
                    if spent + need > self.policy.budget_usd + 1e-6:
                        raise RuntimeError(f"farm budget ${self.policy.budget_usd:.2f} would be exceeded (spent ${spent:.2f} + ${need:.2f})")
            except Exception as e:  # noqa: BLE001  BudgetError and API errors alike: no money check, no extension
                self.warn_once(f"budget-ext:{p['id']}", f"pod {p['id']}: kill switch not extended: {str(e)[:200]}", pod=p["id"])
                continue
            try:
                new = self.backend.extend(p, add, int(hard_pod) if hard_pod else None, self.link)
            except Exception as e:  # noqa: BLE001
                self.log(f"extend {p['id']} failed: {type(e).__name__}: {str(e)[:160]}")
                continue
            if new and new > kill_at:
                granted += need
                self.stats["extensions"] += 1
                self.state.upsert_pod(p["id"], kill_at=new, kill_in_min=round((new - v.pod_now()) / 60, 1),
                                      extensions=int(p.get("extensions") or 0) + 1)
                self.event("fleet", f"pod {p['id']}: kill switch extended to +{(new - v.pod_now()) / 60:.0f} min "
                                    f"(ceiling +{((hard_pod or new) - v.pod_now()) / 60:.0f} min, ${price:.2f}/h)", pod=p["id"])

    STALE_CLAIM_S = 240.0   # worker.Worker.stale_s: a claim this long without a heartbeat is taken over by the next worker that looks

    def claimable_now(self, v: View, jid: str, rec: dict) -> bool:
        """Could a worker that starts now take this job? Queued on the volume, every `after` done, and either unclaimed or claimed
        by a pod that is gone / silent (the next worker takes such a claim over by itself)."""
        E = (v.snap or {}).get("jobs", {}).get(jid) or {}
        if rec.get("status") != "queued" or not E.get("in_queue") or E.get("done"):
            return False
        claim = E.get("claim")
        if claim is not None:
            owner = claim.get("pod_id")
            hb = E.get("heartbeat") or {}
            silent = v.pod_now() - float(hb.get("t") or claim.get("t") or 0) > self.STALE_CLAIM_S
            owner_alive = owner in {p["id"] for p in self.alive_pods(v.st)}
            if owner_alive and not silent:
                return False
        for d in self.spec(jid).get("after") or []:
            if not ((v.snap or {}).get("jobs", {}).get(d) or {}).get("done"):
                return False
        return True

    def wanted_by(self, v: View, p: dict) -> bool:
        """Is there queued work this pod could take right now? (GPU pods take GPU jobs, CPU pods the analysis jobs.)"""
        gpu = p.get("role") == "gpu"
        return any(self.claimable_now(v, jid, t["rec"]) and needs_gpu(self.spec(jid)) == gpu for jid, t in v.tracked.items())

    def can_delete(self, v: View, p: dict) -> tuple[bool, str]:
        if p.get("borrowed"):
            return False, "borrowed pod"
        for jid, t in v.tracked.items():
            r = t["rec"]
            if r.get("pod") == p["id"] and r.get("status") in ("running", "done"):
                return False, f"{jid} is {r['status']} here"
        for jid in self.pulls:
            if (self.rec_of(v.st, jid) or {}).get("pod") == p["id"]:
                return False, f"{jid} is being pulled"
        # the volume is reachable only through a pod: never delete the pod the watcher reads / pulls through while a pull is in
        # flight or a finished job is waiting for one (unless another pod can take over)
        others = [q for q in self.alive_pods(v.st) if q["id"] != p["id"] and q.get("host") and q.get("state") != "stuck"]
        if p["id"] == self.link_pod and not others and (self.pulls or any(t["rec"].get("status") == "done" for t in v.tracked.values())):
            return False, "it is the only link to the volume while results are being pulled"
        return True, "all its jobs are pulled"

    def delete_pod(self, pid: str, reason: str, force: bool = False) -> bool:
        st = self.state.load()
        p = st["pods"].get(pid)
        if p is None or p.get("state") in ("deleted", "gone", "lost"):
            return False
        if p.get("borrowed") and not force:
            return False
        fails, next_try = self.delete_fail.get(pid, (0, 0.0))
        if self.clock() < next_try:
            return False
        try:
            self.backend.delete(pid, reason)
        except Exception as e:  # noqa: BLE001   a delete that Runpod did not confirm: the pod stays in state.json and the ledger, billing
            fails += 1
            wait = min(900.0, 60.0 * 2 ** (fails - 1))
            self.delete_fail[pid] = (fails, self.clock() + wait)
            self.stats["errors"] += 1
            self.event("error", f"delete of pod {pid} NOT confirmed ({fails}x; next try in {wait:.0f} s): {type(e).__name__}: {str(e)[:160]}", pod=pid)
            return False
        self.delete_fail.pop(pid, None)
        cost = None
        with contextlib.suppress(Exception):
            cost = self.backend.ledger_cost(pid)
        mark_pod_deleted(self.state, pid, reason, cost, self.clock())
        self.stats["pods_deleted"] += 1
        self.missing.pop(pid, None)
        self.recycle.discard(pid)
        if self.link_pod == pid:
            self.link, self.link_pod = None, None
        self.log(f"deleted pod {pid}: {reason}")
        return True

    def delete_idle_pods(self, v: View) -> None:
        if v.snap is None:
            return   # no fresh view of the volume: a result that finished a moment ago would not be known to be unpulled
        for p in list(self.alive_pods(v.st)):
            if p.get("state") not in ("idle", "exited", "stuck") and p["id"] not in self.recycle:
                continue
            if p.get("state") == "stuck":
                ok_, why_ = self.can_delete(v, p)
                if not ok_ or p.get("hb_seen"):   # never delete a pod that holds a job or has ever run a worker on a "stuck" guess
                    self.warn_once(f"stuck-kept:{p['id']}", f"pod {p['id']} looks stuck but is kept: {why_ if not ok_ else 'its worker ran before'}",
                                   pod=p["id"])
                    continue
                self.event("pod", f"pod {p['id']} never started its worker (no heartbeat after {self.policy.start_timeout_s:.0f} s; "
                                  f"last event: {p.get('last_event') or '-'}): deleting", pod=p["id"])
                self.delete_pod(p["id"], "worker never started", force=True)
                self.launch_failures += 1
                self.stuck_pods += 1
                if self.stuck_pods >= 2:
                    self.warn_once("halt", "two pods in a row never started their worker: the watcher stops creating pods "
                                           "(look at pods/<id>/events.log and prepare.log on the volume, then restart the watcher)")
                continue
            idle_for = v.now - float(p.get("idle_since") or v.now)
            ok, why = self.can_delete(v, p)
            if not ok:
                continue
            exited = p.get("state") == "exited"
            if p.get("role") == "gateway":
                if len(self.alive_pods(v.st)) > 1 or not v.tracked:
                    self.delete_pod(p["id"], "gateway no longer needed")
                continue
            if self.wanted_by(v, p) and idle_for < 2 * self.policy.idle_grace_s and p["id"] not in self.recycle:
                continue
            if exited or idle_for >= self.policy.idle_grace_s or p["id"] in self.recycle:
                self.delete_pod(p["id"], "recycled after a failed job" if p["id"] in self.recycle else
                                f"{'worker exited' if exited else 'idle'}, {why}")

    # ---- launching
    def room_for_launch(self, v: View) -> float | None:
        """Worst-case budget left for new pods under the farm's own ceiling (None: unlimited by it)."""
        if self.policy.budget_usd is None:
            return None
        spent, committed = self.farm_cost(v.st, v.now)
        return max(0.0, self.policy.budget_usd - spent - committed)

    def launch_if_needed(self, v: View) -> None:
        if not self.policy.auto_launch or "launch" in self.actions or self.clock() < self.next_launch_at or self.stuck_pods >= 2:
            return
        if self.launches >= self.policy.max_pod_launches:
            self.warn_once("max-launches", f"launch cap reached ({self.policy.max_pod_launches}); no more pods will be created by the watcher")
            return
        alive = [p for p in self.alive_pods(v.st) if p["id"] not in self.recycle]
        gpu_ready = [j for j, t in v.tracked.items() if needs_gpu(self.spec(j)) and self.claimable_now(v, j, t["rec"])]
        cpu_ready = [j for j, t in v.tracked.items() if not needs_gpu(self.spec(j)) and self.claimable_now(v, j, t["rec"])]
        gpu_alive = [p for p in alive if p.get("role") == "gpu"]
        cpu_alive = [p for p in alive if p.get("role") == "cpu"]
        free_gpu = [p for p in gpu_alive if p.get("state") in ("idle", "starting")]
        n_gpu = min(len(gpu_ready) - len(free_gpu), self.policy.target_gpu_pods - len(gpu_alive))
        n_cpu = 1 if cpu_ready and not [p for p in cpu_alive if p.get("state") in ("idle", "starting", "running")] else 0
        # the volume is reachable only through a pod: with work outstanding and no pod at all, a small CPU gateway restores the view
        need_gw = v.snap is None and bool(v.tracked) and not alive and self.gw_launches < 2
        if n_gpu <= 0 and n_cpu <= 0 and not need_gw:
            return
        room = self.room_for_launch(v)
        if room is not None and room <= 0:
            self.warn_once("farm-budget", f"farm budget ${self.policy.budget_usd:.2f} is fully committed; no new pods")
            return
        role = "gateway" if need_gw else "gpu" if n_gpu > 0 else "cpu"
        if role == "gateway":
            self.gw_launches += 1
        self.actions["launch"] = self.pool.submit(self._launch_task, role, 1, room)   # one at a time: each launch re-checks offers, gates, stock
        self.launches += 1
        self.log(f"launching a {role} pod: gpu_ready={gpu_ready} cpu_ready={cpu_ready} alive_gpu={len(gpu_alive)}")

    def _launch_task(self, role: str, n: int, room: float | None) -> list[dict]:
        pol = self.policy
        prefix = f"hm-farm-{dance_tag(self.dance)}"
        if role == "gpu":
            res = self.backend.launch_gpu(n, gpus=pol.gpus or None, minutes=pol.minutes, hard_minutes=pol.hard_minutes, max_cost=room,
                                          kinds=GPU_KINDS, prefix=prefix)
        elif role == "cpu":
            res = self.backend.launch_cpu(n, vcpu=pol.cpu_vcpu, minutes=pol.cpu_minutes, hard_minutes=pol.cpu_hard_minutes, max_cost=room,
                                          kinds=CPU_KINDS, prefix=f"{prefix}-cpu")
        else:
            info, created = self.backend.gateway(minutes=30, vcpu=2, reuse=False, name=f"{prefix}-gateway")
            res = [info] if created else []
        ok = [r for r in res if r.get("pod_id")]
        for r in ok:
            self.register_pod(r, role, by="watch")
        bad = [r for r in res if not r.get("pod_id")]
        if bad and not ok:
            self.launch_failures += 1
            delay = min(900.0, self.policy.launch_cooldown_s * 2 ** min(4, self.launch_failures - 1))
            self.next_launch_at = self.clock() + delay
            self.event("launch", f"{role} launch failed: {str(bad[0].get('error'))[:200]}; next try in {delay:.0f} s")
        elif ok:
            self.launch_failures = 0
            self.next_launch_at = self.clock() + self.policy.launch_gap_s
        return res

    def register_pod(self, r: dict, role: str, by: str = "watch") -> None:
        record_pod(self.state, r, role, by, now=self.clock())
        self.stats["pods_launched"] += 1
        self.log(f"registered {role} pod {r['pod_id']}")

    def adopt_orphans(self, st: dict) -> None:
        """Pods that carry THIS dance's name (hm-farm-<whole tag>-<stamp>|cpu-|gateway) and the ledger owner farm:<dance> (or no owner row at
        all), that Runpod lists but state.json does not know (the watcher died while launching). Another dance's pods, and any pod this
        watcher did not launch, are never adopted: finish() deletes what it manages. hm-farm pods nobody claims are reported, not touched."""
        if self.rp is None:
            return
        mine = owner_of(self.dance)
        for pid, r in self.rp.items():
            nm = r.get("name") or ""
            if pid in st["pods"] or not nm.startswith("hm-farm"):
                continue
            running = str(r.get("status")).upper() == "RUNNING"
            if pod_name_belongs(nm, self.dance) and running and r.get("owner") in (None, mine):
                role = "cpu" if "-cpu-" in nm else "gateway" if nm.endswith("-gateway") else "gpu"
                info = self.backend.attach(pid)
                self.state.upsert_pod(pid, id=pid, name=nm, role=role, gpu=r.get("gpu"), price_hr=float(r.get("cost_per_hr") or 0),
                                      state="starting", host=(info or {}).get("host"), port=(info or {}).get("port"),
                                      created_ts=self.clock() - float(r.get("uptime_s") or 0), launched_by="adopted", jobs=[], extensions=0)
                self.event("fleet", f"adopted orphan pod {pid} ({nm})", pod=pid)
            elif pid not in self.warned_unknown:
                self.warned_unknown.add(pid)
                self.event("warn", f"hm-farm pod {pid} ({nm}, owner {r.get('owner') or '-'}) is not this watcher's: left alone", pod=pid)

    def recheck_deleted(self, v: View) -> None:
        """A pod this watcher marked deleted / lost that Runpod still lists (a delete that did not take, or a lost pod that came back) is
        deleted again: state.json and the ledger must never say a billing pod is gone."""
        if not v.rp or not v.rp_fresh:
            return
        for pid, p in v.st["pods"].items():
            if p.get("state") not in ("deleted", "lost", "gone") or p.get("borrowed"):
                continue
            r = v.rp.get(pid)
            if r is None:
                self.relisted.pop(pid, None)
                continue
            n = self.relisted[pid] = self.relisted.get(pid, 0) + 1
            if n < 2:
                continue   # one look can be a stale list
            fails, next_try = self.delete_fail.get(pid, (0, 0.0))
            if self.clock() < next_try:
                continue
            try:
                self.backend.delete(pid, "re-delete: still listed by Runpod after it was marked deleted")
            except Exception as e:  # noqa: BLE001
                fails += 1
                self.delete_fail[pid] = (fails, self.clock() + min(900.0, 60.0 * 2 ** (fails - 1)))
                self.event("error", f"pod {pid} is still listed by Runpod ({r.get('status')}) and the re-delete failed: {short_err(e)}", pod=pid)
                continue
            self.delete_fail.pop(pid, None)
            self.relisted.pop(pid, None)
            self.event("fleet", f"pod {pid} was still listed by Runpod after being marked {p.get('state')}: deleted again", pod=pid)

    def reconcile_ledger(self, now: float) -> None:
        if now - self.reconcile_t < 300.0:
            return
        self.reconcile_t = now
        try:
            res = self.backend.reconcile()
        except Exception as e:  # noqa: BLE001
            self.log(f"ledger reconcile failed: {short_err(e)}")
            return
        if res.get("closed"):
            self.event("fleet", f"ledger: closed {len(res['closed'])} row(s) of pods Runpod no longer lists: {', '.join(res['closed'])}")

    # ---------------------------------------------------------------------------------------------- interim stitch (pod)
    def interim_stitch(self, v: View) -> None:
        """Pod mode, opt-in: whenever the set of chunks the farm has finished changes and the final job is not yet runnable, queue a
        partial stitch (job stitch-N: the reference span + every finished chunk, `dancecap.stitch --partial`, evidence for coverage) on a
        CPU pod. Seam gates fail early instead of after the last chunk, and coverage turns green as results land."""
        pol = self.policy
        if not pol.interim_stitch or pol.stitch != "pod" or "interim" in self.actions or self.clock() < self.interim_retry_at:
            return
        st = v.st
        jobs = st["jobs"]
        if any(j.startswith("stitch-") and r.get("status") in ACTIVE for j, r in jobs.items()):
            return
        farm_done = sorted(c for c, r in st["chunks"].items()
                           if r.get("status") in ("done", "pulled", "stitched") and r.get("core_k0") is not None and r.get("pod"))
        if not farm_done:
            return
        final = self.spec("final") if "final" in jobs else {}
        if final and all((self.rec_of(st, d) or {}).get("status") in ("done", "pulled", "stitched") for d in final.get("after") or []):
            return   # the final job can run now: it stitches everything
        sig = ",".join(farm_done)
        nums = [int(j.split("-", 1)[1]) for j in jobs if j.startswith("stitch-") and j.split("-", 1)[1].isdigit()]
        if sig in {r.get("sig") for j, r in jobs.items() if j.startswith("stitch-")}:
            return
        n = max(nums or [0]) + 1
        self.actions["interim"] = self.pool.submit(self._interim_task, farm_done, n, sig)

    def _interim_task(self, ids: list[str], n: int, sig: str) -> str:
        try:
            return self._interim_build_and_queue(ids, n, sig)
        except BaseException:
            self.interim_retry_at = self.clock() + 300.0   # a broken build is not retried every cycle
            raise

    def _interim_build_and_queue(self, ids: list[str], n: int, sig: str) -> str:
        from . import analysis as A
        from . import chunks as C
        from . import launch
        from . import world as W
        from .timeline import DanceSpec
        jid = f"stitch-{n}"
        spec = DanceSpec.load(self.dance)
        try:
            world = W.load(self.dance)
        except W.WorldError:
            world = None
        st = self.state.load()
        job = A.build_final_job(spec, [st["chunks"][c] for c in ids], world, analyses=(), jid=jid, partial=True, hash_inputs=True,
                                code=C.code_info())
        A.write_job(job)
        self.state.set_job(jid, kind="stitch", status="planned", after=job.get("after"), chunks=ids, sig=sig, written=now_iso(),
                           spec=str(jobs_dir(self.dance) / f"{jid}.json"))
        res = launch.queue_upload(self.dance, [job], None)
        r = res.get(jid) or {}
        if not r.get("ok"):
            self.state.set_job(jid, status="failed", error=str(r.get("error"))[:300])
            raise RuntimeError(f"interim stitch {jid} could not be queued: {r.get('error')}")
        self.state.set_job(jid, status="queued", queued=now_iso())
        self.event("stitch", f"interim stitch {jid} queued for {len(ids)} finished chunk(s): {', '.join(ids)}")
        return jid

    # ---------------------------------------------------------------------------------------------- local stitch
    def local_stitch(self, v: View) -> None:
        if self.policy.stitch != "local" or "stitch" in self.actions:
            return
        pulled = sorted(c for c, r in v.st["chunks"].items() if r.get("status") in ("pulled", "stitched") and r.get("core_k0") is not None)
        sig = ",".join(pulled)
        if not pulled or sig == self.last_stitch_sig:
            return
        if any(r.get("status") in ("queued", "running") for r in v.st["chunks"].values()) and len(pulled) < 2:
            return
        self.last_stitch_sig = sig
        self.actions["stitch"] = self.pool.submit(self._stitch_task, pulled)

    def _stitch_task(self, ids: list[str]) -> dict:
        """dancecap.stitch on this machine (only when asked: FARM.md 0.10 puts the stitch on a pod). Single thread, normal priority."""
        import importlib.util
        if importlib.util.find_spec("dancecap.stitch") is None:
            self.event("stitch", "dancecap.stitch is not present; local stitch skipped")
            return {"skipped": True}
        from . import analysis as A
        from .timeline import DanceSpec
        import dancecap.farm as F
        spec = DanceSpec.load(self.dance)
        st = self.state.load()
        chunks = [st["chunks"][c] for c in ids if c in st["chunks"] and "take" in st["chunks"][c]]
        doc = A.stitch_list(spec, chunks, partial=True)
        lst = farm_dir(self.dance) / "stitch_list.json"
        atomic_write_json(lst, doc)
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONUTF8="1")
        t0 = time.time()
        p = subprocess.run([sys.executable, "-m", "dancecap.stitch", self.dance, "--chunks", f"work/{self.dance}/farm/stitch_list.json",
                            "--partial"], cwd=str(F.PACKAGE_ROOT), env=env, capture_output=True, text=True, timeout=3600)
        (farm_dir(self.dance) / "stitch_local.log").write_text((p.stdout or "") + "\n" + (p.stderr or ""), encoding="utf-8")
        verdict = {0: "ok", 2: "a seam gate failed (see stitch.json)"}.get(p.returncode, f"rc={p.returncode}")
        self.event("stitch", f"local stitch of {len(ids)} chunk(s): {verdict} in {time.time() - t0:.0f} s")
        if p.returncode in (0, 2):
            def f(s):
                for c in ids:
                    self.apply(s, c, "stitched")
            self.state.update(f)
        self.cov_dirty = True
        return {"rc": p.returncode}

    # ---------------------------------------------------------------------------------------------- housekeeping
    def beat_pidfile(self, state: str = "running") -> None:
        atomic_write_json(pid_path(self.dance), {"pid": os.getpid(), "ppid": os.getppid(), "host": socket.gethostname(),
                                                 "started": iso(self.t_start), "heartbeat": time.time(), "heartbeat_iso": now_iso(),
                                                 "cycle": self.cycles, "state": state, "poll_s": self.policy.poll_s,
                                                 "argv": sys.argv[1:], "python": sys.executable}, indent=1)

    def refresh_coverage(self) -> None:
        try:
            from . import cli
            cli.refresh_coverage(self.dance)
            self.cov_dirty = False
            self.cov_t = self.clock()
        except Exception as e:  # noqa: BLE001
            self.cov_t = self.clock()
            self.warn_once(f"cov:{type(e).__name__}", f"coverage refresh failed: {type(e).__name__}: {str(e)[:200]}")

    def housekeeping(self, v: View) -> None:
        now = self.clock()
        self.beat_pidfile()
        if self.cov_dirty and now - self.cov_t >= 20 or now - self.cov_t >= self.policy.coverage_s:
            self.refresh_coverage()
        self.reconcile_ledger(now)
        if now - self.state_t < self.policy.state_s:
            return
        self.state_t = now
        st = self.state.load()
        spent, committed = self.farm_cost(st, now)
        rate = sum(float(p.get("price_hr") or 0) for p in self.alive_pods(st))
        spend = {"farm_usd": round(spent, 3), "farm_committed_usd": round(committed, 2), "rate_hr": round(rate, 3),
                 "budget_usd": self.policy.budget_usd}
        if now - self.spend_t >= 600 or not st.get("spend", {}).get("balance"):
            self.spend_t = now
            with contextlib.suppress(Exception):
                spend["balance"] = round(self.backend.balance(), 2)
            with contextlib.suppress(Exception):
                spend["cap_remaining"] = round(self.backend.remaining(), 2)
        else:
            for k in ("balance", "cap_remaining"):
                if k in st.get("spend", {}):
                    spend[k] = st["spend"][k]
        by: dict[str, int] = {}
        for c in st["chunks"].values():
            by[c.get("status", "?")] = by.get(c.get("status", "?"), 0) + 1

        def f(s):
            s["spend"].update(spend)
            s["spend"]["updated"] = now_iso()
            w = s["watcher"]
            w.update(pid=os.getpid(), host=socket.gethostname(), started=iso(self.t_start), heartbeat=now_iso(), heartbeat_epoch=time.time(),
                     cycle=self.cycles, state="running", poll_s=self.policy.poll_s, link_pod=self.link_pod, stats=dict(self.stats),
                     chunks=by, log=str(log_path(self.dance)))
        self.state.update(f)

    def check_finished(self, v: View) -> None:
        st = self.state.load()
        tracked = self.tracked(st)
        if tracked:
            self.seen_active = True
            self.idle_since = None
            return
        if self.pulls or self.actions:
            return
        if not self.seen_active and not any(c.get("status") in TERMINAL for c in st["chunks"].values()):
            self.idle_since = self.idle_since or self.clock()
            if self.clock() - self.idle_since >= self.policy.idle_exit_s:
                self.finish("nothing queued")
            return
        self.finish("all jobs terminal")

    def finish(self, reason: str) -> None:
        st = self.state.load()
        left: list[str] = []
        self.delete_fail.clear()
        if not self.policy.keep_pods:
            for p in self.alive_pods(st):
                if not p.get("borrowed") and not self.delete_pod(p["id"], f"watcher finished: {reason}", force=True):
                    left.append(p["id"])
        if left:   # a delete Runpod did not confirm: the watcher says so loudly instead of finishing as if everything were clean
            self.event("error", f"watcher finished but pod(s) {', '.join(left)} could NOT be deleted: they may still bill until their kill switch")
        self.refresh_coverage()
        res = self.build_result(reason)
        res["pods_left"] = left
        atomic_write_json(result_path(self.dance), res)
        self.state.set_watcher(state="finished", finished=now_iso(), result=str(result_path(self.dance)), reason=reason)
        self.event("finish", f"finished: {reason}; result {res['status']}")
        self.finished = True
        self.finish_reason = reason

    def build_result(self, reason: str) -> dict:
        st = self.state.load()
        chunks = {}
        for cid, c in st["chunks"].items():
            d = c.get("done") or {}
            chunks[cid] = {"status": c.get("status"), "kind": c.get("kind"), "core_s": c.get("core_s"), "pod": c.get("pod"),
                           "seconds": d.get("seconds"), "cost_usd": d.get("cost_usd"), "gpu": d.get("gpu"), "world_id": c.get("world_id"),
                           "pulled": c.get("pulled"), "requeues": c.get("requeues", 0), "error": c.get("error") or c.get("last_error")}
        jobs = {jid: {"status": j.get("status"), "pod": j.get("pod"), "error": j.get("error")} for jid, j in st["jobs"].items()
                if jid not in st["chunks"]}
        failed = [c for c, x in chunks.items() if x["status"] == "failed"] + [j for j, x in jobs.items() if x["status"] == "failed"]
        got = [c for c, x in chunks.items() if x["status"] in ("pulled", "stitched")]
        status = "failed" if not got and failed else "partial" if failed else "complete" if got else "empty"
        cov = None
        with contextlib.suppress(Exception):
            from .timeline import Coverage
            cv = Coverage.load(self.dance, with_frames=False)
            if cv:
                cov = {"counts": cv.counts(), "percent": cv.percent(), "seconds": cv.seconds(), "updated": cv.updated}
        spent, _ = self.farm_cost(st, self.clock())
        pods = [{"id": p["id"], "role": p.get("role"), "gpu": p.get("gpu"), "price_hr": p.get("price_hr"), "state": p.get("state"),
                 "cost_usd": p.get("cost_usd"), "age_min": p.get("age_min"), "reason": p.get("deleted_reason"), "jobs": p.get("jobs")}
                for p in st["pods"].values()]
        return {"schema": SCHEMA_RESULT, "dance": self.dance, "finished": now_iso(), "reason": reason, "status": status,
                "elapsed_min": round((self.clock() - self.t_start) / 60, 1), "world": st.get("world"), "chunks": chunks, "jobs": jobs,
                "failed": failed, "coverage": cov, "pods": pods, "spend": {**st.get("spend", {}), "farm_usd": round(spent, 3)},
                "stats": self.stats, "events": st["events"][-25:]}

    # ---------------------------------------------------------------------------------------------- the cycle and the loop
    def cycle(self) -> None:
        now = self.clock()
        self.cycles += 1
        self.collect()
        st = self.state.load()
        pol = Policy.from_dict(st.get("policy"))
        if pol != self.policy:
            self.policy = pol
        tracked = self.tracked(st)
        rp, fresh = self.poll_runpod(now)
        if fresh and self.cycles <= 2:
            self.adopt_orphans(st)
            st = self.state.load()
            tracked = self.tracked(st)
        snap = self.snapshot(st, tracked)
        self.beat_pidfile()
        v = View(now=now, st=st, snap=snap, rp=rp, rp_fresh=fresh, tracked=tracked, skew=self.skew)
        self.sync_jobs(v)
        v.st = self.state.load()
        v.tracked = self.tracked(v.st)
        self.sync_pods(v)
        v.st = self.state.load()
        v.tracked = self.tracked(v.st)
        self.recheck_deleted(v)
        self.handle_failed(v)
        if self.policy.hold:
            self.release_held()
        v.st = self.state.load()
        v.tracked = self.tracked(v.st)
        self.start_pulls(v)
        self.publish_pull_progress()
        self.collect()
        v.st = self.state.load()
        v.tracked = self.tracked(v.st)
        self.extend_killswitches(v)
        self.delete_idle_pods(v)
        v.st = self.state.load()
        self.launch_if_needed(v)
        self.interim_stitch(v)
        self.local_stitch(v)
        self.housekeeping(v)
        self.check_finished(v)

    def release_held(self) -> None:
        try:
            from . import cli
            cli.release_ready(self.dance)
        except Exception as e:  # noqa: BLE001
            self.warn_once("hold", f"releasing held jobs failed: {type(e).__name__}: {str(e)[:160]}")

    def sleep(self, seconds: float) -> None:
        end = time.time() + seconds
        while time.time() < end and not stop_path(self.dance).exists():
            time.sleep(min(1.0, max(0.0, end - time.time())))

    def run(self) -> int:
        lock = InstanceLock(lock_path(self.dance))
        for attempt in range(8):   # health() probes the same lock for a few ms: do not mistake a probe for a running watcher
            if lock.acquire():
                break
            time.sleep(0.25)
        else:
            self.log(f"another watcher already runs for {self.dance}; refusing to start a second one")
            return 3
        with contextlib.suppress(OSError):
            stop_path(self.dance).unlink()
        try:
            self.state.set_watcher(pid=os.getpid(), host=socket.gethostname(), started=iso(self.t_start), state="running", finished=None,
                                   reason=None)
            self.event("watch", f"watcher started (pid {os.getpid()}, poll {self.policy.poll_s:.0f} s, stitch {self.policy.stitch})")
            self.beat_pidfile()
            if self.policy.stitch == "pod" and "final" not in self.state.load()["jobs"]:
                self.event("watch", "no `final` stitch job is planned (farm plan without --no-final): chunks will be pulled but not stitched")
            while not self.finished:
                if stop_path(self.dance).exists():
                    self.event("watch", "stop requested: leaving pods to their kill switches")
                    break
                if self.max_hours and (self.clock() - self.t_start) > self.max_hours * 3600:
                    self.event("watch", f"max hours ({self.max_hours}) reached: stopping")
                    break
                try:
                    self.cycle()
                    self.consec_errors = 0
                except Exception as e:  # noqa: BLE001  the loop never dies of one bad cycle
                    self.stats["errors"] += 1
                    self.consec_errors += 1
                    self.log(f"cycle {self.cycles} failed: {type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}")
                    if self.consec_errors in (1, 5, 20):
                        self.event("error", f"watcher cycle failed ({self.consec_errors}x): {type(e).__name__}: {str(e)[:200]}")
                if self.once or self.finished:
                    break
                self.sleep(self.policy.poll_s * (4 if self.consec_errors > 10 else 1))
        finally:
            fin = self.finished
            with contextlib.suppress(Exception):
                self.pool.shutdown(wait=False, cancel_futures=False)
            with contextlib.suppress(Exception):
                self.state.set_watcher(state="finished" if fin else "stopped", stopped=now_iso())
            with contextlib.suppress(Exception):
                self.beat_pidfile("finished" if fin else "stopped")
            lock.release()
        return 0


# ------------------------------------------------------------------------------------------------ status / wait / start / stop


def summary(dance: str) -> dict:
    """ONE compact dict for monitor agents: watcher health, jobs, pods, coverage, spend, newest events."""
    st = State(dance).load()
    h = health(dance)
    by: dict[str, int] = {}
    running = []
    for c in st["chunks"].values():
        by[c.get("status", "?")] = by.get(c.get("status", "?"), 0) + 1
        if c.get("status") == "running":
            running.append({"id": c["id"], "pod": c.get("pod"), "step": c.get("step"), "progress": c.get("progress"), "eta_min": c.get("eta_min")})
    jobs_other = {jid: j.get("status") for jid, j in st["jobs"].items() if jid not in st["chunks"]}
    pods = [{"id": p["id"], "role": p.get("role"), "gpu": p.get("gpu"), "state": p.get("state"), "price_hr": p.get("price_hr"),
             "age_min": p.get("age_min"), "kill_in_min": p.get("kill_in_min"), "jobs": p.get("jobs"), "spend_usd": p.get("spend_usd")}
            for p in st["pods"].values() if p.get("state") not in ("deleted", "gone", "lost")]
    cov = None
    with contextlib.suppress(Exception):
        d = read_json(farm_dir(dance) / "coverage.json", None)
        if d:
            cov = {"percent": d.get("percent"), "seconds": d.get("seconds"), "counts": d.get("counts"), "updated": d.get("updated")}
    res = read_json(result_path(dance), None)
    return {"dance": dance, "time": now_iso(), "watcher": h, "finished": bool(res) and (st.get("watcher") or {}).get("state") == "finished",
            "chunks": {"by_status": by, "running": running}, "jobs": jobs_other, "pods": pods, "coverage": cov,
            "spend": {k: st.get("spend", {}).get(k) for k in ("farm_usd", "rate_hr", "balance", "cap_remaining", "budget_usd")},
            "result": None if not res else {"status": res.get("status"), "failed": res.get("failed")},
            "last_events": [{"ts": e["ts"], "kind": e["kind"], "msg": e["msg"]} for e in st["events"][-4:]]}



def wait(dance: str, seconds: float, until: str = "any", poll: float = 2.0) -> dict:
    """Block up to `seconds` or until something worth telling happened since the call began (a job done / pulled / failed, a pod
    lost / launched / deleted, the watcher gone or finished). Returns summary() + why + the new events."""
    t_end = time.time() + max(0.0, seconds)
    ev0 = (State(dance).load()["events"] or [{}])[-1].get("t", 0)
    why, new = "timeout", []
    while True:
        st = State(dance).load()
        new = [e for e in st["events"] if e.get("t", 0) > ev0]
        h = health(dance)
        w = st.get("watcher") or {}
        if w.get("state") == "finished" and h["running"] is False:
            why = "finished"
            break
        if not h["running"] and w.get("state") in ("running",):
            why = "watcher_down"
            break
        if not h["running"] and any(c.get("status") in ACTIVE for c in st["chunks"].values()):
            why = "no_watcher"   # jobs are active but nobody is watching them: waiting would only burn the time
            break
        hit = None
        for e in new:
            k, st_ = e.get("kind"), e.get("status")
            if until in ("pulled", "any") and k == "pull":
                hit = "pulled"
            elif until in ("done", "any") and k == "chunk" and st_ in ("done", "failed"):
                hit = f"chunk_{st_}"
            elif until == "any" and (k in ("pod", "launch", "error", "warn", "stitch")):
                hit = k
        if hit:
            why = hit
            break
        if time.time() >= t_end:
            break
        time.sleep(max(0.2, min(poll, t_end - time.time())))
    out = summary(dance)
    out.update(why=why, waited_s=round(seconds - max(0.0, t_end - time.time()), 1),
               new_events=[{"ts": e["ts"], "kind": e["kind"], "msg": e["msg"]} for e in new[-10:]])
    return out


def _harden_children() -> None:
    """Detached on Windows (no console): every console child (ssh, runpodctl, bash) would open its own window. Give each its own hidden
    console instead (CREATE_NO_WINDOW), for everything this process spawns."""
    if os.name != "nt" or getattr(subprocess.Popen, "_farm_hardened", False):
        return
    orig = subprocess.Popen.__init__

    def init(self, *a, **k):
        k["creationflags"] = int(k.get("creationflags") or 0) | 0x08000000
        orig(self, *a, **k)
    subprocess.Popen.__init__ = init
    subprocess.Popen._farm_hardened = True


def start_detached(dance: str, extra: list[str] | None = None, wait_s: float = 25.0) -> dict:
    """Spawn `watch run` as a detached process (Windows: DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP, off the parent's job object
    when allowed; POSIX: new session). Idempotent: a running watcher is reported, not doubled."""
    h = health(dance)
    if h["running"]:
        return {"started": False, "already_running": True, **h}
    import dancecap.farm as F
    farm_dir(dance).mkdir(parents=True, exist_ok=True)
    lp = log_path(dance)
    with contextlib.suppress(OSError):
        if lp.exists() and lp.stat().st_size > 5_000_000:
            os.replace(lp, lp.with_suffix(".log.1"))
    cmd = [sys.executable, "-m", "dancecap.farm", "--dance", dance, "watch", "run", *(extra or [])]
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONUTF8="1")
    t0 = time.time()
    fh = open(lp, "ab")
    fh.write(f"\n=== {now_iso()} starting watcher: {' '.join(cmd)}\n".encode())
    fh.flush()
    proc = None
    if os.name == "nt":
        base = 0x00000008 | 0x00000200   # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
        for flags in (base | 0x01000000, base):   # CREATE_BREAKAWAY_FROM_JOB first; refused when the job forbids it
            try:
                proc = subprocess.Popen(cmd, cwd=str(F.PACKAGE_ROOT), stdin=subprocess.DEVNULL, stdout=fh, stderr=subprocess.STDOUT,
                                        env=env, creationflags=flags, close_fds=True)
                break
            except OSError:
                continue
    else:
        proc = subprocess.Popen(cmd, cwd=str(F.PACKAGE_ROOT), stdin=subprocess.DEVNULL, stdout=fh, stderr=subprocess.STDOUT, env=env,
                                start_new_session=True, close_fds=True)
    fh.close()
    if proc is None:
        return {"started": False, "error": "could not spawn the watcher process"}
    while time.time() - t0 < wait_s:
        time.sleep(0.5)
        h = health(dance)
        if h["running"]:
            return {"started": True, "spawn_pid": proc.pid, **h}
        if proc.poll() is not None:
            break
    h = health(dance)
    tail = ""
    with contextlib.suppress(OSError):
        tail = lp.read_text(encoding="utf-8", errors="replace")[-600:]
    return {"started": bool(h["running"]), "spawn_pid": proc.pid, "exit_code": proc.poll(), "log_tail": tail, **h}


def stop(dance: str, force: bool = False, wait_s: float = 60.0) -> dict:
    """Graceful: write watcher.stop (the loop checks it every second). force: also kill the process when it does not leave in time."""
    h = health(dance)
    if not h["running"]:
        return {"stopped": True, "was_running": False}
    stop_path(dance).write_text(now_iso())
    t0 = time.time()
    while time.time() - t0 < wait_s:
        if not health(dance)["running"]:
            return {"stopped": True, "was_running": True, "graceful": True}
        time.sleep(0.5)
    if force and h.get("pid"):
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(h["pid"]), "/T", "/F"], capture_output=True)
        else:
            with contextlib.suppress(OSError):
                os.kill(int(h["pid"]), 15)
        time.sleep(1.0)
    return {"stopped": not health(dance)["running"], "was_running": True, "graceful": False}


def has_work(dance: str) -> bool:
    """Is there anything a watcher should be watching: a queued / running / done-but-unpulled job of the farm, or a pod it manages?"""
    st = State(dance).load()
    if any(c.get("status") in ACTIVE and is_mine(c) for c in [*st["chunks"].values(), *st["jobs"].values()]):
        return True
    return any(p.get("state") not in ("deleted", "gone", "lost") and not p.get("borrowed") for p in st["pods"].values())


def ensure(dance: str, restart_hung: bool = True) -> dict:
    """Make sure a watcher runs when there is work for it: start one if none runs (a rebooted PC, a crash), restart one whose heartbeat
    stopped (`health()['hung']`). Idempotent and cheap; the Pipeline Monitor calls it while it is open. Nothing to watch: does nothing."""
    h = health(dance)
    if h["running"] and not h["hung"]:
        return {"action": "none", "reason": "running", **h}
    if h["running"] and h["hung"]:
        if not restart_hung:
            return {"action": "none", "reason": "hung", **h}
        r = stop(dance, force=True, wait_s=20.0)
        if not r.get("stopped"):
            return {"action": "failed", "reason": "hung watcher would not stop", "stop": r}
    if not has_work(dance):
        return {"action": "none", "reason": "nothing to watch", **health(dance)}
    r = start_detached(dance)
    return {"action": "started", "reason": "hung" if h["running"] else "not running", **r}


# ------------------------------------------------------------------------------------------------ CLI


def policy_from_args(a, base: Policy) -> Policy:
    """Overlay the flags the user actually gave on the stored policy."""
    d: dict = {}

    def put(key, val):
        if val is not None:
            d[key] = val
    put("target_gpu_pods", getattr(a, "pods", None))
    gp = getattr(a, "gpus", None)
    if gp:
        d["gpus"] = [GPU_ALIASES.get(x.strip().lower(), x.strip()) for x in gp.split(",") if x.strip()]
    for k in ("minutes", "hard_minutes", "poll_s", "pull_workers", "max_requeues", "extend_minutes", "cpu_vcpu"):
        put(k, getattr(a, k, None))
    put("budget_usd", getattr(a, "budget", None))
    put("stitch", getattr(a, "stitch", None))
    if getattr(a, "no_launch", False):
        d["auto_launch"] = False
    if getattr(a, "keep_pods", False):
        d["keep_pods"] = True
    if getattr(a, "hold", False):
        d["hold"] = True
    if getattr(a, "interim_stitch", False):
        d["interim_stitch"] = True
    return Policy.from_dict(d, base)


def add_policy_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--pods", type=int, help="GPU pods the watcher keeps busy (default 1; never more than there are waiting jobs)")
    p.add_argument("--gpus", help="GPU priority list, e.g. 4090,l4,a4500 or full Runpod ids (default: pods.GPU_PRIORITY)")
    p.add_argument("--minutes", type=float, help="initial kill-switch lead of a GPU pod (default 100; extended while healthy)")
    p.add_argument("--hard-minutes", dest="hard_minutes", type=float, help="hard ceiling of a GPU pod (default 240)")
    p.add_argument("--budget", type=float, help="ceiling in USD for this run's worst case (spent + committed + new pods)")
    p.add_argument("--stitch", choices=("pod", "local", "none"), help="pod: the `final` job stitches on a CPU pod (default); local: dancecap.stitch here")
    p.add_argument("--hold", action="store_true", help="release queued jobs as their `after` chunks finish")
    p.add_argument("--interim-stitch", dest="interim_stitch", action="store_true",
                   help="stitch what has landed so far on a CPU pod while the rest runs (stitch-N jobs; the final job still stitches everything)")
    p.add_argument("--no-launch", action="store_true", help="the watcher never creates pods")
    p.add_argument("--keep-pods", action="store_true", help="do not delete the pods when everything is done")
    p.add_argument("--max-requeues", dest="max_requeues", type=int)
    p.add_argument("--pull-workers", dest="pull_workers", type=int)


def cmd_watch(a) -> int:
    act = a.action or "start"
    dance = a.dance
    if act == "status":
        print(json.dumps(summary(dance), default=str))
        return 0
    if act == "ensure":
        r = ensure(dance)
        print(json.dumps(r, default=str))
        return 0 if r.get("action") != "failed" else 1
    if act == "stop":
        r = stop(dance, force=a.force)
        print(json.dumps(r))
        return 0 if r["stopped"] else 1
    if act == "wait":
        print(json.dumps(wait(dance, a.seconds if a.seconds is not None else 60.0, a.until), default=str))
        return 0
    base = Policy.load(dance)
    pol = policy_from_args(a, base)
    if act == "start":
        if pol != base:
            pol.save(dance)
        extra = []
        for flag, val in (("--poll", a.poll_s), ("--max-hours", a.max_hours)):
            if val is not None:
                extra += [flag, str(val)]
        r = start_detached(dance, extra)
        if a.json:
            print(json.dumps(r, default=str))
        elif r.get("already_running"):
            print(f"watcher already running for {dance}: pid {r.get('pid')}, cycle {r.get('cycle')}, heartbeat {r.get('heartbeat_age_s')} s ago, log {r['log']}")
        elif r.get("started"):
            print(f"watcher started for {dance}: pid {r.get('pid')}, log {r['log']}")
        else:
            print(f"watcher did not start: {r.get('error') or r.get('log_tail')}", file=sys.stderr)
        return 0 if (r.get("started") or r.get("already_running")) else 1
    if act == "run":
        if os.name == "nt":
            _harden_children()
        w = Watcher(dance, policy=pol if pol != base else None, once=a.once, max_hours=a.max_hours)
        return w.run()
    return 2


def cmd_wait(a) -> int:
    print(json.dumps(wait(a.dance, a.seconds, a.until), default=str))
    return 0


def register(sub) -> None:
    """Plug-in entry of cli.py: the watch / wait commands here, launch / up / pull / down from launch.py (imported lazily)."""
    p = sub.add_parser("watch", help="the detached watcher: start | run | status | stop | wait")
    p.add_argument("action", nargs="?", default="start", choices=("start", "run", "status", "stop", "wait", "ensure"))
    p.add_argument("--poll", dest="poll_s", type=float, help="seconds between cycles (default 30)")
    p.add_argument("--max-hours", dest="max_hours", type=float, help="run: stop after this long (pods keep their kill switches)")
    p.add_argument("--once", action="store_true", help="run: one cycle, then exit")
    p.add_argument("--force", action="store_true", help="stop: kill the process if it does not leave in 60 s")
    p.add_argument("--seconds", type=float, help="wait: how long")
    p.add_argument("--until", choices=("any", "pulled", "done"), default="any")
    add_policy_flags(p)
    p.set_defaults(func=cmd_watch)
    p = sub.add_parser("wait", help="block up to N seconds; print one JSON line when a job finished / was pulled / a pod changed")
    p.add_argument("seconds", type=float)
    p.add_argument("--until", choices=("any", "pulled", "done"), default="any")
    p.set_defaults(func=cmd_wait)
    from . import launch
    launch.register_commands(sub)
    from .cli import register_hook
    register_hook("queue_upload", launch.queue_upload)
