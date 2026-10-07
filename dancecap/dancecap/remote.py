"""Runpod offload for GPU-heavy stages (MAMMA ma_3d, gsplat room training, texture bakes, large models).

Basic GPU work (YOLO, ma_2d inference, beats, Unity) stays local; anything that would hold the local
2080 Ti for a long time runs on a rented 80 GB GPU here.

Safety and cost rules (user decisions, 2026-10-06):
- Spending cap (default $30, override with DANCECAP_RUNPOD_CAP) enforced from work/runpod/ledger.json BEFORE a pod
  is created: remaining budget must cover cost_per_hour x max_minutes.
- Every pod gets two kill switches: an on-pod timer (`sleep N; runpodctl remove pod $RUNPOD_POD_ID`, using the
  pod-scoped key Runpod injects) and a local `finally:` delete. The ledger is updated on delete.
- API key from RUNPOD_API_KEY or the file in DANCECAP_RUNPOD_KEY_FILE (default ~/Desktop/runpod-key.txt); it is
  passed to runpodctl through the environment only and never printed or written.
- Transfers go over SSH with a dedicated key (~/.ssh/runpod_dancecap), bundled as tar files.

    python -m dancecap remote status          # auth check, ssh key, ledger and remaining budget
    python -m dancecap remote gpus            # 80 GB GPU availability and prices
"""
from __future__ import annotations

import contextlib
import json
import os
import shlex
import subprocess
import sys
import tarfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .take import PACKAGE_ROOT

LEDGER = PACKAGE_ROOT / "work" / "runpod" / "ledger.json"
SSH_KEY = Path.home() / ".ssh" / "runpod_dancecap"
# pod host keys change every session: a throwaway known_hosts under work/runpod (git-ignored). 'UserKnownHostsFile=NUL'
# only works for Windows' own OpenSSH; Git Bash's ssh wrote a file literally named NUL into the cwd (dancecap/NUL).
KNOWN_HOSTS = PACKAGE_ROOT / "work" / "runpod" / "known_hosts"
DEFAULT_CAP = 30.0  # raised by the user 2026-10-06 (was $10)
# preferred 80 GB GPUs, cheapest first; availability changes, so status/gpus shows what's live
GPU_PREFERENCE = ["NVIDIA A100-SXM4-80GB", "NVIDIA A100 80GB PCIe", "NVIDIA H100 80GB HBM3", "NVIDIA H100 NVL",
                  "NVIDIA H100 PCIe"]
CLOUDS = ["COMMUNITY", "SECURE"]  # community is cheaper; secure when no community host with a public IP is free
DEFAULT_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"


class BudgetError(RuntimeError):
    pass


def _api_key() -> str:
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        f = Path(os.environ.get("DANCECAP_RUNPOD_KEY_FILE", Path.home() / "Desktop" / "runpod-key.txt"))
        if not f.exists():
            raise RuntimeError(f"no Runpod API key: set RUNPOD_API_KEY or create {f}")
        key = f.read_text().strip()
    return key


def _ctl(*args: str, check: bool = True, timeout: float = 900) -> dict | list | str:
    env = dict(os.environ, RUNPOD_API_KEY=_api_key())
    p = subprocess.run(["runpodctl", *args, "-o", "json"], capture_output=True, text=True, env=env, timeout=timeout)
    out = p.stdout.strip()
    if check and p.returncode != 0:
        # runpodctl errors are JSON on stdout; never echo the environment
        raise RuntimeError(f"runpodctl {' '.join(args[:2])} failed ({p.returncode}): {out[:800] or p.stderr[:800]}")
    try:
        return json.loads(out) if out else {}
    except json.JSONDecodeError:
        return out


# ---------------------------------------------------------------------------
# ledger / budget
# ---------------------------------------------------------------------------

def cap() -> float:
    return float(os.environ.get("DANCECAP_RUNPOD_CAP", DEFAULT_CAP))


def ledger() -> list[dict]:
    return json.loads(LEDGER.read_text()) if LEDGER.exists() else []


def _save_ledger(rows: list[dict]) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    LEDGER.write_text(json.dumps(rows, indent=1))


def spent() -> float:
    """Spend so far: a closed row's recorded/reconciled cost if present, else price x lifetime (open rows keep accruing)."""
    now = time.time()
    total = 0.0
    for r in ledger():
        if r.get("deleted_ts") and r.get("cost") is not None:
            total += float(r["cost"])
            continue
        end = r.get("deleted_ts") or now  # a pod still alive keeps costing
        total += r["cost_per_hr"] * max(0.0, end - r["created_ts"]) / 3600
    return total


def remaining() -> float:
    return cap() - spent()


# ---------------------------------------------------------------------------
# ssh key
# ---------------------------------------------------------------------------

def ensure_ssh_key() -> Path:
    if not SSH_KEY.exists():
        SSH_KEY.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ssh-keygen", "-t", "ed25519", "-N", "", "-C", "dancecap-runpod", "-f", str(SSH_KEY)],
                       check=True, capture_output=True)
    pub = SSH_KEY.with_suffix(".pub").read_text().strip()
    keys = _ctl("ssh", "list-keys")
    listed = json.dumps(keys)
    if pub.split()[1] not in listed:
        _ctl("ssh", "add-key", "--key-file", str(SSH_KEY.with_suffix(".pub")))
    return SSH_KEY


# ---------------------------------------------------------------------------
# pods
# ---------------------------------------------------------------------------

def _echo(s: str) -> None:
    """print() that never raises: a cp1252 console (stdout redirected to a file on Windows) cannot encode the emoji
    some pod tools print (ultralytics' first import), and that UnicodeEncodeError ended a session and deleted its pod
    mid-job (2026-10-06, detect09)."""
    try:
        print(s, end="", flush=True)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "ascii"
        print(s.encode(enc, "replace").decode(enc, "replace"), end="", flush=True)


def _known_hosts_opt() -> str:
    if os.name != "nt":
        return "UserKnownHostsFile=/dev/null"
    KNOWN_HOSTS.parent.mkdir(parents=True, exist_ok=True)
    return f"UserKnownHostsFile={KNOWN_HOSTS.as_posix()}"


@dataclass
class Pod:
    id: str
    name: str
    gpu: str
    cost_per_hr: float
    host: str
    port: int
    user: str = "root"
    killswitch: dict | None = None

    def _ssh_base(self) -> list[str]:
        return ["ssh", "-i", str(SSH_KEY), "-p", str(self.port), "-o", "StrictHostKeyChecking=no",
                "-o", _known_hosts_opt(),
                "-o", "ServerAliveInterval=30", "-o", "LogLevel=ERROR", f"{self.user}@{self.host}"]

    def run(self, cmd: str, check: bool = True, timeout: float | None = None, log: Path | None = None,
            echo: bool = True) -> int:
        """Run a shell command on the pod (bash -lc), streaming output to the console and optional log file.

        timeout (s) is enforced by a watchdog that kills the local ssh client even when the command is silent.
        """
        full = self._ssh_base() + [f"bash -lc {shlex.quote(cmd)}"]
        timed_out = threading.Event()
        with contextlib.ExitStack() as stack:
            fh = stack.enter_context(open(log, "a", encoding="utf-8")) if log else None
            proc = subprocess.Popen(full, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                    encoding="utf-8", errors="replace")
            dog = None
            if timeout:
                def _kill():
                    timed_out.set()
                    proc.kill()
                dog = threading.Timer(timeout, _kill)
                dog.daemon = True
                dog.start()
            try:
                for line in proc.stdout:
                    if echo:
                        _echo(f"[pod] {line}")
                    if fh:
                        fh.write(line)
                        fh.flush()
                rc = proc.wait()
            finally:
                if dog:
                    dog.cancel()
        if timed_out.is_set():
            raise TimeoutError(f"pod command exceeded {timeout:.0f}s: {cmd[:120]}")
        if check and rc != 0:
            raise RuntimeError(f"pod command failed ({rc}): {cmd[:200]}")
        return rc

    def output(self, cmd: str, timeout: float = 300) -> str:
        """stdout of a short pod command (raises on failure)."""
        p = subprocess.run(self._ssh_base() + [f"bash -lc {shlex.quote(cmd)}"], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout)
        if p.returncode != 0:
            raise RuntimeError(f"pod command failed ({p.returncode}): {cmd[:200]}: {(p.stderr or p.stdout)[-400:]}")
        return p.stdout

    def _scp(self, src: str, dst: str, timeout: float | None) -> None:
        subprocess.run(["scp", "-q", "-i", str(SSH_KEY), "-P", str(self.port), "-o", "StrictHostKeyChecking=no",
                        "-o", _known_hosts_opt(),
                        "-o", "ServerAliveInterval=30", "-o", "LogLevel=ERROR", src, dst], check=True, timeout=timeout)

    def put(self, local: Path, remote: str, timeout: float | None = None) -> None:
        self._scp(str(local), f"{self.user}@{self.host}:{remote}", timeout)

    def get(self, remote: str, local: Path, timeout: float | None = None) -> None:
        local.parent.mkdir(parents=True, exist_ok=True)
        self._scp(f"{self.user}@{self.host}:{remote}", str(local), timeout)


def _pod_ssh(pod_json: dict) -> tuple[str, int]:
    """Public IP + mapped port 22 from a pod record."""
    found = _direct_ssh(pod_json)
    if found:
        return found
    info = _ctl("ssh", "info", pod_json["id"])
    host = info.get("ip") or info.get("host")
    port = info.get("port")
    if not host or not port:
        raise RuntimeError(f"no public ssh endpoint for pod {pod_json['id']}: {json.dumps(info)[:400]}")
    return host, int(port)


@contextlib.contextmanager
def pod_session(name: str, max_minutes: float, gpus: list[str] | None = None, image: str = DEFAULT_IMAGE,
                disk_gb: int = 80, clouds: list[str] | None = None, start_timeout_min: float = 6,
                max_start_attempts: int = 2, max_cost: float | None = None):
    """Create a pod, yield it, and always delete it. Refuses to start if the budget can't cover max_minutes.

    max_cost: a per-session ceiling ($) on top of the global cap: GPU/cloud options whose listed price x
    (max_minutes + start timeout) exceeds it are skipped (the on-pod kill switch makes that the worst case).
    """
    ensure_ssh_key()
    last_err = None
    pod = None
    attempts = 0
    for gpu, cloud in [(g, c) for c in (clouds or CLOUDS) for g in (gpus or GPU_PREFERENCE)]:
        price = gpu_price(gpu, cloud)
        if price is None:
            print(f"[runpod] {gpu} ({cloud}): not offered / no stock right now")
            continue
        # budget check uses the listed price BEFORE anything is created
        need = price * (max_minutes + start_timeout_min) / 60
        if max_cost is not None and need > max_cost:
            print(f"[runpod] skip {gpu} ({cloud}) ${price:.2f}/h: worst case ${need:.2f} > session max ${max_cost:.2f}")
            last_err = BudgetError(f"every option exceeds the session max ${max_cost:.2f}")
            continue
        if need > remaining():
            raise BudgetError(f"{gpu}: {max_minutes + start_timeout_min:.0f} min at ${price:.2f}/h = ${need:.2f} > "
                              f"remaining ${remaining():.2f} of the ${cap():.2f} cap")
        # 22/tcp must be requested at creation: without it Runpod gives only its ssh proxy, which can't scp
        args = ["pod", "create", "--name", f"dancecap-{name}", "--image", image, "--gpu-id", gpu,
                "--container-disk-in-gb", str(disk_gb), "--cloud-type", cloud, "--ports", "22/tcp"]
        if cloud == "COMMUNITY":
            args.append("--public-ip")
        try:
            created = _ctl(*args, timeout=180)  # no --wait: record the pod in the ledger the moment it exists
        except Exception as e:  # no capacity for this GPU/cloud: try the next one
            last_err = e
            pid = _id_from_error(str(e))
            if pid:
                _record(pid, name, gpu, cloud, price, max_minutes)
                _delete(pid, reason="create errored")
            continue
        pod_id = created["id"]
        cost = float(created.get("costPerHr") or price)
        _record(pod_id, name, gpu, cloud, cost, max_minutes)
        print(f"[runpod] created {pod_id}: {gpu} ({cloud}) ${cost:.2f}/h; waiting up to {start_timeout_min:.0f} min "
              f"for the container")
        try:
            pod = _wait_running(pod_id, start_timeout_min * 60)
        except BaseException:  # Ctrl-C / crash while waiting: never leave the pod behind
            _delete(pod_id, reason="interrupted while starting")
            raise
        if pod:
            break
        last_err = RuntimeError(f"{gpu} ({cloud}) not reachable over direct ssh in {start_timeout_min:.0f} min")
        _delete(pod_id, reason="not reachable over direct ssh")
        attempts += 1
        if attempts >= max_start_attempts:  # don't walk the whole GPU list paying for start-ups
            break
    if not pod:
        raise RuntimeError(f"no 80 GB GPU pod could be started: {last_err}")

    try:
        host, port = _pod_ssh(pod)
        p = Pod(pod_id, name, gpu, cost, host, port)
        for _ in range(30):  # sshd answers before keys are installed
            try:
                if p.run("true", check=False, timeout=40) == 0:
                    break
            except TimeoutError:
                pass
            time.sleep(5)
        # on-pod kill switch: Runpod injects a pod-scoped RUNPOD_API_KEY and runpodctl into pods. ssh sessions do
        # not always inherit the container environment, so take the two variables from PID 1.
        p.killswitch = arm_killswitch(p, max_minutes)
        print(f"[runpod] pod {pod_id} {gpu} ${cost:.2f}/h, ssh {host}:{port}, auto-delete in {max_minutes:.0f} min; "
              f"budget remaining ${remaining():.2f}")
        yield p
    finally:
        _delete(pod_id, reason="session end")


def arm_killswitch(p: "Pod", max_minutes: float) -> dict:
    """Start the on-pod timer that deletes the pod after max_minutes; returns what was verified about it."""
    secs = int(max_minutes * 60)
    # the deadline is also written to /workspace/.dancecap_kill_at (epoch s): long jobs read it to decide whether an
    # optional step still fits (mamma_chunks gate --fallback full)
    ks = (f"echo $(( $(date +%s) + {secs} )) > /workspace/.dancecap_kill_at; "
          r"set -a; . <(tr '\0' '\n' < /proc/1/environ | grep -E '^RUNPOD_(API_KEY|POD_ID)='); set +a; "
          f"sleep {secs}; runpodctl pod delete $RUNPOD_POD_ID || runpodctl remove pod $RUNPOD_POD_ID")
    p.run(f"nohup bash -c {shlex.quote(ks)} >/tmp/killswitch.log 2>&1 < /dev/null &", check=False, timeout=60)
    try:
        out = p.output(f"sleep 1; pgrep -a -x sleep | grep -q ' {secs}$' && echo timer=1 || echo timer=0; "
                       "command -v runpodctl >/dev/null && echo ctl=1 || echo ctl=0; "
                       r"tr '\0' '\n' < /proc/1/environ | grep -c -E '^RUNPOD_(API_KEY|POD_ID)=' || true", timeout=60)
        lines = out.split()
        info = {"timer": "timer=1" in lines, "runpodctl": "ctl=1" in lines,
                "env_vars": int(lines[-1]) if lines and lines[-1].isdigit() else 0, "after_min": round(max_minutes, 1)}
    except (RuntimeError, subprocess.SubprocessError) as e:
        info = {"error": str(e)[:200]}
    info["armed"] = bool(info.get("timer") and info.get("runpodctl") and info.get("env_vars") == 2)
    if not info["armed"]:
        print(f"[runpod] WARNING: on-pod kill switch not verified ({info}); the local delete is the only guard")
    return info


def _record(pod_id: str, name: str, gpu: str, cloud: str, cost: float, max_minutes: float) -> None:
    rows = ledger()
    if not any(r["pod_id"] == pod_id for r in rows):
        rows.append({"pod_id": pod_id, "name": name, "gpu": gpu, "cloud": cloud, "cost_per_hr": cost,
                     "created": datetime.now(timezone.utc).isoformat(), "created_ts": time.time(),
                     "max_minutes": max_minutes})
        _save_ledger(rows)


def _wait_running(pod_id: str, timeout_s: float) -> dict | None:
    """Poll until the container runs and port 22 is publicly mapped; None on timeout (caller deletes the pod)."""
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        p = _ctl("pod", "get", pod_id, check=False)
        if isinstance(p, dict):
            last = p
            if _direct_ssh(p):
                return p
        time.sleep(10)
    if last:  # say why, so a running-but-unreachable pod isn't mistaken for one that never started
        rt = last.get("runtime") or {}
        print(f"[runpod] {pod_id}: status={last.get('status') or last.get('desiredStatus')} "
              f"uptime={rt.get('uptime')} ports={json.dumps(rt.get('ports'))[:300]} ssh={json.dumps(last.get('ssh'))[:200]}")
    return None


def _direct_ssh(pod_json: dict) -> tuple[str, int] | None:
    """(public ip, port) mapped to the container's port 22, from either REST shape, or None."""
    ssh = pod_json.get("ssh") or {}
    direct = ssh.get("direct")  # REST v2 / MCP shape
    if isinstance(direct, dict) and direct.get("host") and direct.get("port"):
        return direct["host"], int(direct["port"])
    if ssh.get("ip") and ssh.get("port") and not str(ssh["ip"]).startswith("100."):  # runpodctl pod get shape
        return ssh["ip"], int(ssh["port"])
    for x in (pod_json.get("runtime") or {}).get("ports") or []:
        if not isinstance(x, dict):
            continue
        private = x.get("private", x.get("privatePort"))
        public = x.get("public", x.get("publicPort"))
        is_public = x.get("isIpPublic", str(x.get("type", "")).lower() == "tcp")
        if private is not None and int(private) == 22 and public and is_public:
            return x["ip"], int(public)
    return None


def _id_from_error(msg: str) -> str | None:
    try:
        start = msg.index("{")
        return json.loads(msg[start:]).get("id")
    except (ValueError, json.JSONDecodeError, AttributeError):
        return None


def _delete(pod_id: str, reason: str) -> None:
    try:
        _ctl("pod", "delete", pod_id, check=False)
    finally:
        rows = ledger()
        for r in rows:
            if r["pod_id"] == pod_id and not r.get("deleted_ts"):
                r["deleted_ts"] = time.time()
                r["deleted_reason"] = reason
                r["cost"] = round(r["cost_per_hr"] * (r["deleted_ts"] - r["created_ts"]) / 3600, 4)
        _save_ledger(rows)
        print(f"[runpod] pod {pod_id} deleted ({reason}); spent ${spent():.2f} of ${cap():.2f}")


def gpu_price(gpu: str, cloud: str = "COMMUNITY") -> float | None:
    """On-demand $/h for a GPU type in a cloud, or None if that cloud doesn't offer it / has no stock."""
    for g in gpu_list():
        if g.get("gpuId") == gpu or g.get("displayName") == gpu:
            if not g.get("communityCloud" if cloud == "COMMUNITY" else "secureCloud"):
                return None
            v = g.get("communityPricePerHr" if cloud == "COMMUNITY" else "securePricePerHr")
            return float(v) if v else None
    return None


def gpu_table(min_gb: float = 80) -> list[dict]:
    rows = []
    for g in gpu_list():
        if (g.get("memoryInGb") or 0) < min_gb:
            continue
        stocked = [d["dataCenterId"] for d in g.get("dataCenterAvailability", []) if d.get("stockStatus") not in (None, "none")]
        rows.append({"gpu": g["gpuId"], "gb": g["memoryInGb"], "community": g.get("communityPricePerHr"),
                     "secure": g.get("securePricePerHr"), "stock": g.get("stockStatus"), "datacenters": stocked})
    return sorted(rows, key=lambda r: r["community"] or r["secure"] or 99)


def gpu_list() -> list[dict]:
    data = _ctl("gpu", "list")
    return data if isinstance(data, list) else data.get("gpus", data.get("items", []))


def cleanup_orphans() -> list[str]:
    """Delete any dancecap-* pod still running (e.g. after a crash) and close its ledger row."""
    pods = _ctl("pod", "list")
    pods = pods if isinstance(pods, list) else pods.get("pods", [])
    gone = []
    for p in pods:
        if str(p.get("name", "")).startswith("dancecap-"):
            _delete(p["id"], reason="orphan cleanup")
            gone.append(p["id"])
    return gone


# ---------------------------------------------------------------------------
# bundles
# ---------------------------------------------------------------------------

def make_bundle(out: Path, items: list[tuple[Path, str]], exclude: tuple[str, ...] = ()) -> Path:
    """tar.gz of (local path, archive name) pairs, skipping any path containing an excluded fragment."""
    out.parent.mkdir(parents=True, exist_ok=True)

    def filt(ti: tarfile.TarInfo):
        return None if any(x in ti.name for x in exclude) else ti

    with tarfile.open(out, "w:gz", compresslevel=3) as tar:
        for local, arc in items:
            tar.add(str(local), arcname=arc, filter=filt)
    return out


def status() -> dict:
    out = {"cap": cap(), "spent": round(spent(), 4), "remaining": round(remaining(), 4),
           "ssh_key": SSH_KEY.exists(), "ledger_rows": len(ledger())}
    try:
        _api_key()
        out["api_key"] = "present"
        me = _ctl("user", check=False)
        if isinstance(me, dict):  # keep only non-sensitive account fields
            out["account"] = {k: v for k, v in me.items() if any(s in k.lower() for s in ("balance", "spend", "limit"))}
        else:
            out["account"] = str(me)[:200]
    except RuntimeError as e:
        out["api_key"] = str(e)
    return out
