"""Fakes for the watcher tests (tests/test_farm_watch*.py): a local bash plays the pod, a temp dir plays the volume, a
FakeBackend plays Runpod, and a real `worker.Worker` produces genuine done.json / out/result.tar / manifest.json files.

Nothing here touches the network, Runpod or the real work tree.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from dancecap import remote
from dancecap.farm import volume, worker

BASH = shutil.which("bash")
PY = Path(sys.executable).as_posix()


def sh(code: str) -> str:
    """A portable step command: the current interpreter running a one-line program (no quotes needed in `code`)."""
    return f'{sys.executable} -c "{code}"'


class LocalPod:
    """Stands in for remote.Pod: `output()` and the ssh argv run through a local bash."""

    def __init__(self, pod_id: str = "localpod"):
        self.id, self.name, self.gpu, self.cost_per_hr, self.host, self.port = pod_id, "local", "", 0.0, "localhost", 22

    def output(self, cmd: str, timeout: float = 300) -> str:
        p = subprocess.run([BASH, "-c", cmd], capture_output=True, text=True, timeout=timeout)
        if p.returncode != 0:
            raise RuntimeError(f"local command failed ({p.returncode}): {cmd[:120]}: {p.stderr[-200:]}")
        return p.stdout

    def run(self, cmd, check=True, timeout=None, log=None, echo=True):
        return subprocess.run([BASH, "-c", cmd]).returncode


class LocalLink(volume.Link):
    """volume.Link whose 'ssh' is a local bash; python = this interpreter for the snapshot script. The snapshot command itself runs
    in-process by default (`fast_snapshot`: same script, same arguments, no process start-up) so the cycle tests stay quick; one test
    switches it off to run the real heredoc through bash."""
    python = PY
    fast_snapshot = True
    cmds: list = []

    def sh(self, cmd: str, timeout: float = 300) -> str:
        if self.fast_snapshot and "<<'HM_SNAP'" in cmd:
            import contextlib
            import io
            import shlex as _shlex
            head = cmd.split(chr(10), 1)[0]
            argv = _shlex.split(head.split("<<")[0])
            buf = io.StringIO()
            old = sys.argv
            sys.argv = ["-", *argv[2:]]
            try:
                with contextlib.redirect_stdout(buf):
                    exec(compile(snapshot_source(), "<snapshot>", "exec"), {"__name__": "__main__"})
            finally:
                sys.argv = old
            return buf.getvalue()
        return super().sh(cmd, timeout)

    def _ssh(self, remote_cmd: str, compress: bool) -> list[str]:
        LocalLink.cmds.append(remote_cmd)
        return [BASH, "-c", remote_cmd]


def snapshot_source() -> str:
    from dancecap.farm import pull
    return pull.SNAPSHOT_PY


class FlakyLink(LocalLink):
    """The first `drops` downloads are cut after `cut` bytes and exit 255 (an ssh reset in the middle of the stream); `down`
    makes every ssh call fail while True (the whole connection is gone)."""
    drops = 0
    cut = 64
    down = False
    calls = 0

    def sh(self, cmd: str, timeout: float = 300) -> str:
        if self.down:
            raise RuntimeError("ssh: connection reset by peer")
        return super().sh(cmd, timeout)

    def _ssh(self, remote_cmd: str, compress: bool) -> list[str]:
        LocalLink.cmds.append(remote_cmd)
        if self.down:
            return [BASH, "-c", "exit 255"]
        if remote_cmd.startswith("tail -c +"):
            FlakyLink.calls += 1
            if FlakyLink.drops > 0:
                FlakyLink.drops -= 1
                return [BASH, "-c", f"{remote_cmd} | head -c {FlakyLink.cut}; exit 255"]
        return [BASH, "-c", remote_cmd]


class FakeBackend:
    """Everything the watcher asks of Runpod, on a local 'volume'. Pods are dicts; the test plays the pods' workers by calling
    `run_job` / `claim` / `crash_pod` etc."""

    def __init__(self, vol: Path, tmp: Path, link_cls=LocalLink):
        self.vol, self.tmp, self.link_cls = Path(vol), Path(tmp), link_cls
        self.pods: dict[str, dict] = {}
        self.calls: list[tuple] = []
        self.balance_usd = 100.0
        self.cap_left = 50.0
        self.committed_usd = 0.0
        self.launch_errors: list[str] = []   # popped one per launch attempt: an error string = that attempt fails
        self.n = 0
        self.links: dict[str, object] = {}
        self.api_down = False
        self.order: list[str] = []            # 'delete:<id>' / 'pull:<id>' markers for ordering assertions
        self.workers: dict[str, worker.Worker] = {}
        self.auto_workers = False             # True: every launched gpu / cpu pod runs a real worker loop in a thread until it is deleted
        self.threads: list[threading.Thread] = []

    # -- Runpod listing
    def runpod_pods(self) -> dict[str, dict]:
        if self.api_down:
            raise RuntimeError("runpodctl: api down")
        return {pid: {"id": pid, "name": p["name"], "status": p["status"], "cost_per_hr": p["price"], "owner": p.get("owner")}
                for pid, p in self.pods.items()}

    def make_link(self, pod: dict):
        lk = self.links.get(pod["id"]) or self.link_cls(LocalPod(pod["id"]), root=self.vol.as_posix())
        self.links[pod["id"]] = lk
        return lk

    def attach(self, pod_id: str):
        return {"host": "localhost", "port": 22} if pod_id in self.pods else None

    # -- creation
    def _worker_loop(self, pid: str, kinds, gpu: bool) -> None:
        w = self.worker_for(pid, kinds=set(kinds.split(",")) if kinds else None)
        w.gpu0 = {"name": "FAKE GPU"} if gpu else {}
        while pid in self.pods:
            try:   # on Windows os.replace over a file the watcher is reading can fail for a moment (not on the pods' Linux)
                worker.write_json(self.vol / "pods" / pid / "heartbeat", {"t": time.time(), "state": "idle", "job": None})
                for spec in w.specs():
                    if pid in self.pods and w.claimable(spec) and w.enough_time(spec) and w.try_claim(spec):
                        w.run_job(spec)
                        break
            except OSError:
                pass
            time.sleep(0.05)

    def _new(self, kind: str, price: float, minutes: float, hard_minutes: float, name: str, kinds: str | None = None) -> dict:
        self.n += 1
        pid = f"{kind}pod{self.n}"
        now = time.time()
        self.pods[pid] = {"id": pid, "name": name, "status": "RUNNING", "price": price, "created": now}
        pd = self.vol / "pods" / pid
        pd.mkdir(parents=True, exist_ok=True)
        (pd / "kill_at").write_text(str(int(now + minutes * 60)))
        worker.write_json(pd / "heartbeat", {"t": now, "state": "idle", "job": None})
        if self.auto_workers and kind in ("gpu", "cpu"):
            th = threading.Thread(target=self._worker_loop, args=(pid, kinds, kind == "gpu"), daemon=True)
            th.start()
            self.threads.append(th)
        return {"pod_id": pid, "name": name, "gpu": "NVIDIA GeForce RTX 4090" if kind == "gpu" else "cpu:cpu5cx8", "cost_per_hr": price,
                "host": "localhost", "port": 22, "t_create": now, "t_ssh": now, "minutes": minutes, "hard_minutes": hard_minutes,
                "kill_at": int(now + minutes * 60), "hard_ceil": int(now + hard_minutes * 60), "dc": "EU-RO-1", "volume": "vol"}

    def offers(self, gpus=None):
        return [{"gpu": "NVIDIA GeForce RTX 4090", "price": 0.69, "stock": "Low", "memory_gb": 24},
                {"gpu": "NVIDIA L4", "price": 0.43, "stock": "Low", "memory_gb": 24}]

    def launch_gpu(self, n, *, gpus=None, minutes=90, hard_minutes=None, max_cost=None, kinds=None, prefix="hm-farm", idle_min=5,
                   dry_run=False, **kw):
        out = []
        if dry_run:
            return [{"dry_run": True, "gpu": "NVIDIA GeForce RTX 4090", "price": 0.69, "worst": round(0.69 * ((hard_minutes or minutes) + 8) / 60, 3)}
                    for _ in range(n)]
        for i in range(n):
            self.calls.append(("launch_gpu", n, minutes, hard_minutes, kinds))
            if self.launch_errors:
                err = self.launch_errors.pop(0)
                if err:
                    out.append({"error": err})
                    continue
            out.append(self._new("gpu", 0.69, minutes, hard_minutes or minutes * 1.5, f"{prefix}-{self.n + 1}", kinds))
        return out

    def launch_cpu(self, n, *, vcpu=8, minutes=60, hard_minutes=None, max_cost=None, kinds="analyse", prefix="hm-farm-cpu", dry_run=False,
                   **kw):
        out = []
        if dry_run:
            return [{"dry_run": True, "flavor": "cpu5c", "vcpu": vcpu, "price": 0.28, "worst": 0.4} for _ in range(n)]
        for i in range(n):
            self.calls.append(("launch_cpu", n, minutes, hard_minutes, kinds))
            out.append(self._new("cpu", 0.28, minutes, hard_minutes or minutes * 1.5, f"{prefix}-{self.n + 1}", kinds))
        return out

    def gateway(self, minutes=30, vcpu=2, reuse=True, name=None):
        self.calls.append(("gateway", minutes))
        return self._new("gw", 0.07, minutes, minutes * 1.5, "hm-farm-gateway"), True

    # -- deletion / budget
    def delete(self, pod_id: str, reason: str) -> None:
        self.calls.append(("delete", pod_id, reason))
        self.order.append(f"delete:{pod_id}")
        self.pods.pop(pod_id, None)

    def mark_gone(self, pod_id: str, reason: str) -> None:
        self.calls.append(("mark_gone", pod_id, reason))

    def ledger_cost(self, pod_id: str):
        return None

    def reconcile(self) -> dict:
        self.calls.append(("reconcile",))
        return {"closed": [], "flagged": [], "alive": list(self.pods)}

    def balance(self) -> float:
        return self.balance_usd

    def remaining(self) -> float:
        return self.cap_left

    def gate(self, cost: float) -> dict:
        limit = min(self.cap_left, self.balance_usd - 2.0) - self.committed_usd
        if cost > limit + 1e-9:
            raise remote.BudgetError(f"worst case ${cost:.2f} > available ${limit:.2f}")
        return {"available": limit}

    def hard_ceiling(self, pod: dict):
        return None

    # -- kill switch (the real one is pods.extend_killswitch: never past hard_epoch)
    def extend(self, pod: dict, add_minutes: float, hard_epoch: int | None, link) -> int:
        kf = self.vol / "pods" / pod["id"] / "kill_at"
        now = int(time.time())
        old = int(kf.read_text().strip() or 0) if kf.exists() else now
        new = max(now, old) + int(add_minutes * 60)
        if hard_epoch:
            new = min(new, int(hard_epoch))
        if new <= old:
            return old
        kf.write_text(str(new))
        self.calls.append(("extend", pod["id"], new))
        return new

    # -- helpers for the tests: the 'pods' behave through real workers
    def worker_for(self, pod_id: str, ws: Path | None = None, **kw) -> worker.Worker:
        if pod_id not in self.workers:
            w = worker.Worker(self.vol, pod_id=pod_id, ws=ws or self.tmp / f"ws_{pod_id}", work=self.tmp / f"work_{pod_id}",
                              state=self.tmp / f"state_{pod_id}", poll_s=0.05, **kw)
            w.gpu0 = {"name": "FAKE GPU"}
            (w.ws / "dancecap").mkdir(parents=True, exist_ok=True)
            self.workers[pod_id] = w
        return self.workers[pod_id]


def queue_job(vol: Path, spec: dict) -> dict:
    """Write queue/<id>.json the way `farm queue` does (a minimal but complete dancecap.farm.job/1 spec)."""
    spec = {"schema": worker.SCHEMA, "dance": "t", "kind": "fit", "world_id": None, "inputs": [], "steps": [], "outputs": [],
            "max_minutes": 5, "needs_gpu": True, **spec}
    (Path(vol) / "queue").mkdir(parents=True, exist_ok=True)
    (Path(vol) / "queue" / f"{spec['id']}.json").write_text(json.dumps(spec))
    return spec


def job_spec(jid: str, take: str | None = None, payload: bytes | int = b"motion", **kw) -> dict:
    """A job whose single step writes dancecap/work/<take>/smplx/lead.npz (bytes `payload`, or `payload` bytes of a repeating
    pattern when it is an int: command lines are short) and the evidence file."""
    take = take or f"t_{jid}"
    data = f"(bytes(range(256)) * ({payload} // 256 + 1))[:{payload}]" if isinstance(payload, int) else repr(payload)
    code = (f"import pathlib; d = pathlib.Path('work/{take}/smplx'); d.mkdir(parents=True, exist_ok=True); "
            f"(d / 'lead.npz').write_bytes({data}); e = pathlib.Path('work/{take}/farm'); e.mkdir(parents=True, exist_ok=True); "
            f"(e / 'evidence.npz').write_bytes(b'ev')")
    spec = {"id": jid, "take": take, "chunk": {"id": jid, "take": take, "k0": 0, "k1": 899, "core_k0": 0, "core_k1": 899},
            "steps": [{"name": "make", "cmd": sh(code), "timeout_min": 1}],
            "outputs": [{"path": f"dancecap/work/{take}/smplx/"}, {"path": f"dancecap/work/{take}/farm/evidence.npz"}], **kw}
    return spec


def finish_job(be: FakeBackend, pod_id: str, spec: dict) -> str:
    """The pod's worker claims and runs `spec` for real (done.json + out/ on the volume)."""
    w = be.worker_for(pod_id)
    w.failed_here.discard(spec["id"])
    assert w.try_claim(spec), f"claim of {spec['id']} by {pod_id} failed"
    return w.run_job(spec)


def start_job(be: FakeBackend, pod_id: str, spec: dict, step: str = "make", progress: dict | None = None) -> None:
    """Claim the job and write a live heartbeat, leaving it 'running' (the step is not executed)."""
    w = be.worker_for(pod_id)
    assert w.try_claim(spec)
    now = time.time()
    worker.write_json(be.vol / "chunks" / spec["id"] / "heartbeat",
                      {"t": now, "pod_id": pod_id, "id": spec["id"], "step": step, "step_i": 0, "steps_n": max(1, len(spec.get("steps", []))),
                       "step_started": now, "started": now, "attempt": 1, "progress": progress or {}, "gpu": {"name": "FAKE GPU", "util": 90}})
    worker.write_json(be.vol / "pods" / pod_id / "heartbeat", {"t": now, "state": "running", "job": spec["id"], "step": step})
