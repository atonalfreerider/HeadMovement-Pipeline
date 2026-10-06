"""Runpod offload for GPU-heavy stages (MAMMA ma_3d, gsplat room training, texture bakes, large models).

Basic GPU work (YOLO, ma_2d inference, beats, Unity) stays local; anything that would hold the local
2080 Ti for a long time runs on a rented 80 GB GPU here.

Safety and cost rules (user decisions, 2026-10-06):
- Spending cap (default $10, override with DANCECAP_RUNPOD_CAP) enforced from work/runpod/ledger.json BEFORE a pod
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
import tarfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .take import PACKAGE_ROOT

LEDGER = PACKAGE_ROOT / "work" / "runpod" / "ledger.json"
SSH_KEY = Path.home() / ".ssh" / "runpod_dancecap"
DEFAULT_CAP = 10.0
# preferred 80 GB GPUs, cheapest first; availability changes, so status/gpus shows what's live
GPU_PREFERENCE = ["NVIDIA A100 80GB PCIe", "NVIDIA A100-SXM4-80GB", "NVIDIA H100 PCIe", "NVIDIA H100 80GB HBM3",
                  "NVIDIA H100 NVL"]
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
    now = time.time()
    total = 0.0
    for r in ledger():
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

@dataclass
class Pod:
    id: str
    name: str
    gpu: str
    cost_per_hr: float
    host: str
    port: int
    user: str = "root"

    def _ssh_base(self) -> list[str]:
        return ["ssh", "-i", str(SSH_KEY), "-p", str(self.port), "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=NUL" if os.name == "nt" else "UserKnownHostsFile=/dev/null",
                "-o", "ServerAliveInterval=30", "-o", "LogLevel=ERROR", f"{self.user}@{self.host}"]

    def run(self, cmd: str, check: bool = True, timeout: float | None = None, log: Path | None = None) -> int:
        """Run a shell command on the pod (bash -lc), streaming output to the console and optional log file."""
        full = self._ssh_base() + [f"bash -lc {shlex.quote(cmd)}"]
        with contextlib.ExitStack() as stack:
            fh = stack.enter_context(open(log, "a", encoding="utf-8")) if log else None
            proc = subprocess.Popen(full, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                    encoding="utf-8", errors="replace")
            start = time.time()
            for line in proc.stdout:
                print(f"[pod] {line}", end="")
                if fh:
                    fh.write(line)
                if timeout and time.time() - start > timeout:
                    proc.kill()
                    raise TimeoutError(f"pod command exceeded {timeout:.0f}s: {cmd[:120]}")
            rc = proc.wait()
        if check and rc != 0:
            raise RuntimeError(f"pod command failed ({rc}): {cmd[:200]}")
        return rc

    def put(self, local: Path, remote: str) -> None:
        subprocess.run(["scp", "-i", str(SSH_KEY), "-P", str(self.port), "-o", "StrictHostKeyChecking=no",
                        "-o", "LogLevel=ERROR", str(local), f"{self.user}@{self.host}:{remote}"], check=True)

    def get(self, remote: str, local: Path) -> None:
        local.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["scp", "-i", str(SSH_KEY), "-P", str(self.port), "-o", "StrictHostKeyChecking=no",
                        "-o", "LogLevel=ERROR", f"{self.user}@{self.host}:{remote}", str(local)], check=True)


def _pod_ssh(pod_json: dict) -> tuple[str, int]:
    """Public IP + mapped port 22 from a pod record."""
    for p in (pod_json.get("runtime") or {}).get("ports") or pod_json.get("ports") or []:
        if isinstance(p, dict) and int(p.get("privatePort", 0)) == 22 and p.get("isIpPublic", True):
            return p["ip"], int(p["publicPort"])
    info = _ctl("ssh", "info", pod_json["id"])
    host = info.get("ip") or info.get("host")
    port = info.get("port")
    if not host or not port:
        raise RuntimeError(f"no public ssh endpoint for pod {pod_json['id']}: {json.dumps(info)[:400]}")
    return host, int(port)


@contextlib.contextmanager
def pod_session(name: str, max_minutes: float, gpus: list[str] | None = None, image: str = DEFAULT_IMAGE,
                disk_gb: int = 80, cloud: str = "COMMUNITY"):
    """Create a pod, yield it, and always delete it. Refuses to start if the budget can't cover max_minutes."""
    ensure_ssh_key()
    last_err = None
    for gpu in gpus or GPU_PREFERENCE:
        args = ["pod", "create", "--name", f"dancecap-{name}", "--image", image, "--gpu-id", gpu,
                "--container-disk-in-gb", str(disk_gb), "--cloud-type", cloud, "--wait", "--wait-timeout", "15m"]
        if cloud == "COMMUNITY":
            args.append("--public-ip")
        try:
            # budget check uses the GPU's listed price before anything is created
            price = gpu_price(gpu, cloud)
            if price is None:
                continue
            need = price * max_minutes / 60
            if need > remaining():
                raise BudgetError(f"{gpu}: {max_minutes:.0f} min at ${price:.2f}/h = ${need:.2f} > remaining "
                                  f"${remaining():.2f} of the ${cap():.2f} cap")
            pod = _ctl(*args, timeout=1200)
            break
        except BudgetError:
            raise
        except Exception as e:  # unavailable GPU / capacity: try the next one
            last_err = e
            pod = None
            # a create that timed out may still exist: the error carries its id
            pid = _id_from_error(str(e))
            if pid:
                _delete(pid, reason="create failed")
    if not pod:
        raise RuntimeError(f"no 80 GB GPU pod could be created: {last_err}")

    pod_id = pod["id"]
    cost = float(pod.get("costPerHr") or price)
    rows = ledger()
    rows.append({"pod_id": pod_id, "name": name, "gpu": gpu, "cloud": cloud, "cost_per_hr": cost,
                 "created": datetime.now(timezone.utc).isoformat(), "created_ts": time.time(),
                 "max_minutes": max_minutes})
    _save_ledger(rows)
    try:
        host, port = _pod_ssh(pod)
        p = Pod(pod_id, name, gpu, cost, host, port)
        for _ in range(30):  # sshd answers before keys are installed
            if p.run("true", check=False) == 0:
                break
            time.sleep(5)
        # on-pod kill switch: Runpod injects a pod-scoped RUNPOD_API_KEY and runpodctl into pods
        p.run(f"nohup bash -c 'sleep {int(max_minutes * 60)}; runpodctl pod delete $RUNPOD_POD_ID "
              f"|| runpodctl remove pod $RUNPOD_POD_ID' >/tmp/killswitch.log 2>&1 &", check=False)
        print(f"[runpod] pod {pod_id} {gpu} ${cost:.2f}/h, ssh {host}:{port}, auto-delete in {max_minutes:.0f} min; "
              f"budget remaining ${remaining():.2f}")
        yield p
    finally:
        _delete(pod_id, reason="session end")


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
    for g in gpu_list():
        if g.get("id") == gpu or g.get("displayName") == gpu:
            key = "communityPrice" if cloud == "COMMUNITY" else "securePrice"
            v = g.get(key) or (g.get("lowestPrice") or {}).get("uninterruptablePrice")
            return float(v) if v else None
    return None


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
