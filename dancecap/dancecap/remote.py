"""Runpod offload for GPU-heavy stages (MAMMA ma_3d, gsplat room training, texture bakes, large models).

Basic GPU work (YOLO, ma_2d inference, beats, Unity) stays local; anything that would hold the local
2080 Ti for a long time runs on a rented GPU here: the fastest card with enough memory first (GPU_LADDER, 4090 first),
each card in every cloud before the next, (gpu, cloud) pairs that led to a broken host last (pod_session).

Safety and cost rules (user decisions, 2026-10-06):
- Spending cap enforced from work/runpod/ledger.json BEFORE a pod is created: remaining budget must cover
  cost_per_hour x max_minutes. It follows the Runpod account (ledger spend + live balance - $2, owner 2026-10-21);
  DANCECAP_RUNPOD_CAP=<dollars> overrides it.
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
import re
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
DEFAULT_CAP = 106.7  # J3b 2026-10-11: spent 78.60 + (balance 36.18 - 8) = 106.78, rounded down (john_lula single-camera fits); was 101.7 (F1 2026-10-09)
# the 80 GB GPUs for a caller that passes no list (front_full; jobs needing 80 GB get the same order from GPU_LADDER), FASTEST
# first (owner 2026-10-21: speed over cost; fit_full's tiers already put the H100 first); availability changes, `remote gpus`
GPU_PREFERENCE = ["NVIDIA H100 80GB HBM3", "NVIDIA H100 NVL", "NVIDIA H100 PCIe", "NVIDIA A100-SXM4-80GB",
                  "NVIDIA A100 80GB PCIe"]
# the clouds tried for EACH GPU before the next (slower) GPU: community is cheaper, secure has the same card when no
# community host with a public IP is free (pod_session loops GPUs outside, clouds inside)
CLOUDS = ["COMMUNITY", "SECURE"]
# GPUs by memory, fastest first for the measured workloads (owner 2026-10-21: speed over cost): a 4090 ran SAM 3D Body at 0.355 s
# per person-frame against 2.0 s on an A100-SXM (trial 2026-10-21) and fits MAMMA 2.4x faster than an A100 (farm drill); the
# Ada 48 GB cards next, then H100 before A100, the Ampere 48 GB cards and the 3090 last (LHM++ ran out of memory on a 3090 where
# the 4090 peaked at 8 GB). Blackwell cards (RTX 5090, RTX PRO) are left out: the bootstrap's torch 2.5.1 / cu124 has no sm_120
# kernels. remote_jobs.job_vram_gb gives every runner job a memory need (24 GB unless measured otherwise), so this ladder is the
# default order of `remote run`.
GPU_LADDER = [("NVIDIA GeForce RTX 4090", 24), ("NVIDIA L40S", 48), ("NVIDIA RTX 6000 Ada Generation", 48), ("NVIDIA L40", 48),
              ("NVIDIA H100 80GB HBM3", 80), ("NVIDIA H100 NVL", 94), ("NVIDIA H100 PCIe", 80),
              ("NVIDIA A100-SXM4-80GB", 80), ("NVIDIA A100 80GB PCIe", 80),
              ("NVIDIA A40", 48), ("NVIDIA RTX A6000", 48), ("NVIDIA GeForce RTX 3090", 24)]


def gpus_for(min_vram: list) -> list[str] | None:
    """The GPU list for a batch whose every job declares a memory need (the largest wins), fastest first; None when a job
    declares none (remote_jobs.job_vram_gb always declares one) or nothing in the ladder is big enough (= GPU_PREFERENCE)."""
    vals = [float(v) for v in min_vram if v is not None]
    if not min_vram or len(vals) != len(min_vram):
        return None
    need = max(vals)
    return [g for g, gb in GPU_LADDER if gb >= need] or None


def short_gpu(g: str | None) -> str:
    return str(g or "?").replace("NVIDIA ", "").replace("GeForce ", "")
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


def _ctl_rc(*args: str, timeout: float = 900) -> tuple[int, dict | list | str]:
    """Like `_ctl(check=False)` but also returns the exit code: a failed `pod delete` prints a JSON error object and exits 1, and
    that must not be read as success."""
    env = dict(os.environ, RUNPOD_API_KEY=_api_key())
    p = subprocess.run(["runpodctl", *args, "-o", "json"], capture_output=True, text=True, env=env, timeout=timeout)
    out = p.stdout.strip()
    try:
        return p.returncode, (json.loads(out) if out else {})
    except json.JSONDecodeError:
        return p.returncode, out or p.stderr.strip()


# ---------------------------------------------------------------------------
# ledger / budget
# ---------------------------------------------------------------------------

def cap() -> float:
    """The spend cap every budget gate uses: DANCECAP_RUNPOD_CAP=<dollars> as given; unset (or "balance") it follows the Runpod
    account (balance_cap()); DEFAULT_CAP only when the balance has never been read."""
    env = os.environ.get("DANCECAP_RUNPOD_CAP", "").strip()
    if env and env.lower() != "balance":
        return float(env)
    c = balance_cap()
    return float(DEFAULT_CAP) if c is None else c


BALANCE_RESERVE = 2.0   # = farm.pods.MIN_BALANCE_RESERVE: a pod is never cut off at a zero balance
BALANCE_CAP = PACKAGE_ROOT / "work" / "runpod" / "balance_cap.json"
BALANCE_TTL_S = 60.0
_balance_memo: dict = {}


def balance_cap(max_age: float = BALANCE_TTL_S) -> float | None:
    """The cap that follows the account (owner 2026-10-21, after adding $50: "use the runpod amount to set the spending cap"):
    ledger spend + the live clientBalance - BALANCE_RESERVE, so remaining() = balance - reserve at the read. Runpod is asked at
    most every `max_age` s; the last good value is kept in work/runpod/balance_cap.json and used when Runpod can't be reached,
    so an API hiccup never drops the cap under what is already spent (farm watchers read remaining()). None when it was never
    read, or with DANCECAP_RUNPOD_BALANCE=off (the tests)."""
    if os.environ.get("DANCECAP_RUNPOD_BALANCE", "").strip().lower() in ("off", "0", "no", "false"):
        return None
    now = time.time()
    if _balance_memo and now - _balance_memo["ts"] < max_age:
        return _balance_memo["cap"]
    c = None
    try:
        me = _ctl("user", check=False, timeout=30)
        if isinstance(me, dict) and me.get("clientBalance") is not None:
            bal, sp = float(me["clientBalance"]), spent()
            c = round(sp + bal - BALANCE_RESERVE, 2)
            BALANCE_CAP.parent.mkdir(parents=True, exist_ok=True)
            tmp = BALANCE_CAP.with_name(f"{BALANCE_CAP.name}.{os.getpid()}.tmp")
            tmp.write_text(json.dumps({"cap": c, "balance": round(bal, 4), "spent": round(sp, 4), "ts": now}))
            os.replace(tmp, BALANCE_CAP)
    except Exception:  # noqa: BLE001 - no key, no runpodctl, timeout, ledger busy: fall back to the last good read
        pass
    if c is None:
        try:
            c = float(json.loads(BALANCE_CAP.read_text())["cap"])
        except (OSError, ValueError, TypeError, KeyError):
            return None
    _balance_memo.update(ts=now, cap=c)
    return c


CAP_IN_USE = PACKAGE_ROOT / "work" / "runpod" / "cap_in_use.json"
CAP_IN_USE_FRESH_S = 3 * 24 * 3600


def record_cap_in_use(sid: str | None = None) -> None:
    """`remote run` writes the cap it enforces (DANCECAP_RUNPOD_CAP of the workflow that started it), so a process started
    without that environment (the monitor) can SHOW the same cap. Display only: cap() and every budget gate still read the
    environment / the balance. Never fatal."""
    try:
        env = os.environ.get("DANCECAP_RUNPOD_CAP", "").strip()
        src = "env" if env and env.lower() != "balance" else ("balance" if balance_cap() is not None else "default")
        CAP_IN_USE.parent.mkdir(parents=True, exist_ok=True)
        tmp = CAP_IN_USE.with_name(f"{CAP_IN_USE.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps({"cap": cap(), "source": src, "ts": time.time(), "sid": sid, "pid": os.getpid()}))
        os.replace(tmp, CAP_IN_USE)
    except Exception:  # noqa: BLE001
        pass


def effective_cap(now: float | None = None) -> dict:
    """{cap, source}: DANCECAP_RUNPOD_CAP of this process, else an explicit cap the last runner recorded (< 3 days old), else
    the balance cap, else DEFAULT_CAP. For display (the monitor, progress_eta.budget); the runners' own gate is cap()."""
    env = os.environ.get("DANCECAP_RUNPOD_CAP", "").strip()
    if env and env.lower() != "balance":
        return {"cap": cap(), "source": "DANCECAP_RUNPOD_CAP"}
    now = time.time() if now is None else now
    try:
        d = json.loads(CAP_IN_USE.read_text())
        # a runner that follows the balance recorded a value that is stale by now: read the balance instead
        if float(d.get("cap") or 0) > 0 and now - float(d.get("ts") or 0) < CAP_IN_USE_FRESH_S and d.get("source") != "balance":
            return {"cap": float(d["cap"]), "source": f"the last pod runner ({d.get('source')}, session {d.get('sid')})",
                    "ts": d.get("ts")}
    except (OSError, ValueError, TypeError):
        pass
    c = balance_cap()
    if c is not None:
        return {"cap": c, "source": f"the Runpod balance (spent + balance - ${BALANCE_RESERVE:.0f})"}
    return {"cap": float(DEFAULT_CAP), "source": "remote.DEFAULT_CAP"}


def ledger() -> list[dict]:
    """The ledger rows. Writers replace the file atomically, but an older process (a live fit_full session started before that
    change) may still write it in place: a read that fails to parse is retried for a few seconds, and then RAISES - a budget gate
    must never read a half-written ledger as an empty one."""
    last: Exception | None = None
    for _ in range(40):
        try:
            return json.loads(LEDGER.read_text()) if LEDGER.exists() else []
        except (json.JSONDecodeError, PermissionError, OSError) as e:   # Windows: sharing violation while a writer replaces it
            last = e
            time.sleep(0.1)
    raise RuntimeError(f"ledger {LEDGER} unreadable: {last}")


def _save_ledger(rows: list[dict]) -> None:
    """Atomic: temp file in the same directory, then os.replace (retried on a Windows sharing violation)."""
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    tmp = LEDGER.with_name(f"{LEDGER.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(rows, indent=1))
    for i in range(60):
        try:
            os.replace(tmp, LEDGER)
            return
        except PermissionError:
            if i == 59:
                raise
            time.sleep(0.05)


_LEDGER_MUTEX = threading.RLock()
_LEDGER_DEPTH = threading.local()


@contextlib.contextmanager
def ledger_lock(timeout: float = 60.0, stale: float = 30.0):
    """Cross-process (and cross-thread) mutex around every ledger read-modify-write: an O_EXCL lock file next to the ledger, taken over
    after `stale` seconds. Re-entrant. (A separate file from fit_full's own `ledger.lock` directory so the two never deadlock: a
    session that wraps remote._record in that lock calls into this one inside it.)"""
    with _LEDGER_MUTEX:
        depth = getattr(_LEDGER_DEPTH, "n", 0)
        lock = LEDGER.parent / "ledger.rw.lock"
        if depth == 0:
            LEDGER.parent.mkdir(parents=True, exist_ok=True)
            t_end = time.time() + timeout
            while True:
                try:
                    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    os.write(fd, json.dumps({"pid": os.getpid(), "t": time.time()}).encode())
                    os.close(fd)
                    break
                except FileExistsError:
                    try:
                        if time.time() - lock.stat().st_mtime > stale:
                            lock.unlink()
                            continue
                    except OSError:
                        pass
                except PermissionError:   # Windows: the holder is unlinking it right now
                    pass
                if time.time() > t_end:
                    raise TimeoutError(f"ledger lock {lock} busy for more than {timeout:.0f} s")
                time.sleep(0.02)
        _LEDGER_DEPTH.n = depth + 1
        try:
            yield
        finally:
            _LEDGER_DEPTH.n = depth
            if depth == 0:
                try:
                    lock.unlink()
                except OSError:
                    pass


OPEN_ROW_GRACE_MIN = 10.0   # an open row stops accruing this long after its max_minutes unless Runpod still lists the pod


def _row_end(r: dict, now: float) -> float:
    """Where an OPEN row stops accruing. Every pod has a kill switch at `max_minutes` (the farm's hard ceiling), so a row older than
    that plus a grace is a pod that is gone but whose closing write was lost; it counted forever (a 16 h phantom of $22.92 shrank
    the budget by $1.5 an hour). `reconcile_ledger` stamps `alive_ts` for a pod Runpod still lists, which keeps it accruing."""
    limit = float(r["created_ts"]) + (float(r.get("max_minutes") or 0) + OPEN_ROW_GRACE_MIN) * 60
    if not r.get("max_minutes") or now <= limit:
        return now
    return min(now, max(limit, float(r.get("alive_ts") or 0)))


def spent() -> float:
    """Spend so far: a closed row's recorded/reconciled cost if present, else price x lifetime (open rows keep accruing, up to the
    pod's own kill switch unless Runpod still lists it)."""
    now = time.time()
    total = 0.0
    for r in ledger():
        if r.get("deleted_ts") and r.get("cost") is not None:
            total += float(r["cost"])
            continue
        end = r.get("deleted_ts") or _row_end(r, now)
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
    choice: dict | None = None   # how pod_session picked this GPU / cloud (options tried, max-cost exclusions)

    def _ssh_base(self) -> list[str]:
        return ["ssh", "-i", str(SSH_KEY), "-p", str(self.port), "-o", "StrictHostKeyChecking=no",
                "-o", _known_hosts_opt(),
                "-o", "ServerAliveInterval=30", "-o", "LogLevel=ERROR", f"{self.user}@{self.host}"]

    def run(self, cmd: str, check: bool = True, timeout: float | None = None, log: Path | None = None,
            echo: bool = True, on_line=None) -> int:
        """Run a shell command on the pod (bash -lc), streaming output to the console and optional log file.

        timeout (s) is enforced by a watchdog that kills the local ssh client even when the command is silent.
        on_line(line) sees every output line as it arrives (the runner's frame counter); its errors are swallowed.
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
                    if on_line is not None:
                        try:
                            on_line(line)
                        except Exception:  # noqa: BLE001 - progress must never end a pod command
                            pass
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

    def put_secret(self, name: str, value: str, timeout: float = 60) -> None:
        """Write a secret to /workspace/.secrets/<name> (mode 600) through ssh STDIN: the value never appears in a command
        line, a log, a bundle or a session file."""
        if not name.replace("_", "").isalnum() or not name[0].isalpha():
            raise ValueError(f"bad secret name {name!r}")
        p = subprocess.run(self._ssh_base() + [f"bash -lc 'umask 077; mkdir -p /workspace/.secrets && cat > /workspace/.secrets/{name}'"],
                           input=value, capture_output=True, text=True, timeout=timeout)
        if p.returncode != 0:
            raise RuntimeError(f"could not write secret {name} on the pod (rc {p.returncode})")

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


# ---------------------------------------------------------------------------
# broken hosts (2026-10-21): community host 60.249.37.148 failed with "CUDA unknown error" 6 times out of 6 in the
# 2026-10-19..21 round, each try costing 3-11 min of wall time before the next one. Every new pod now gets a CUDA check
# (nvidia-smi + one torch matmul) before anything is uploaded; a failing host is denied for a day (for good after 3
# failures without a success), and a pod landing on a denied host is deleted at once.
# ---------------------------------------------------------------------------

BAD_HOSTS = PACKAGE_ROOT / "work" / "runpod" / "bad_hosts.json"
KNOWN_BAD_HOSTS = {"60.249.37.148": {"fails": 6, "ok": 0, "reason": "CUDA unknown error, 6 of 6 sessions 2026-10-19..21 "
                                                                   "(docs/PIPELINE_EFFICIENCY.md)",
                                     "gpu": "NVIDIA L40S", "cloud": "COMMUNITY"}}   # the ledger's 5 deletions 2026-10-21
BAD_HOST_DENY_S = 24 * 3600
PREFLIGHT_TIMEOUT_S = 150
PREFLIGHT_CMD = ("nvidia-smi -L >/dev/null 2>&1 || { echo PREFLIGHT_NVSMI_FAIL; exit 3; }; "
                 "if python -c 'import torch' 2>/dev/null; then python -c \"import torch; assert torch.cuda.is_available(), "
                 "'cuda not available'; x = torch.randn(512, 512, device='cuda'); v = float((x @ x).sum()); "
                 "print('PREFLIGHT_CUDA_OK', torch.cuda.get_device_name(0))\" || { echo PREFLIGHT_CUDA_FAIL; exit 4; }; fi; "
                 "echo PREFLIGHT_OK")


def bad_hosts() -> dict:
    """{host: {fails, ok, last, deny_until, reason}} = the seed above merged with work/runpod/bad_hosts.json."""
    out = {h: dict(v) for h, v in KNOWN_BAD_HOSTS.items()}
    try:
        d = json.loads(BAD_HOSTS.read_text())
        for h, v in (d.get("hosts") or {}).items():
            if isinstance(v, dict):
                out[h] = {**out.get(h, {}), **v}
    except (OSError, ValueError):
        pass
    return out


def _deny_reason(v: dict | None, now: float) -> str | None:
    if not v:
        return None
    if float(v.get("deny_until") or 0) > now:
        return f"failed the CUDA check {v.get('fails', 1)} time(s), last {v.get('reason') or ''}".strip()
    if int(v.get("fails") or 0) >= 3 and int(v.get("ok") or 0) == 0:
        return f"{v.get('fails')} failures and no success: {v.get('reason') or 'broken host'}"
    return None


def host_denied(host: str | None, now: float | None = None) -> str | None:
    """Why `host` is denied now, or None."""
    if not host:
        return None
    return _deny_reason(bad_hosts().get(str(host)), time.time() if now is None else now)


# (gpu, cloud) -> host for every broken host THIS process met (pod_session landed on it, or its CUDA check failed): Runpod
# cannot be told to avoid a host, and its scheduler hands the same (gpu, cloud) the same machine again (2026-10-21: the L40S
# community host 60.249.37.148 came up first in 4 of 11 creations), so pod_session tries such pairs last for the rest of the
# session. The persisted list (bad_hosts.json gpu/cloud of a denied host) does the same for later processes.
_BROKEN_PAIRS: dict[tuple[str, str], str] = {}
_BROKEN_LOCK = threading.Lock()


def remember_broken(host: str | None, gpu: str | None, cloud: str | None) -> None:
    """Remember which (gpu, cloud) led to a broken host, in this process and in bad_hosts.json. Never fatal."""
    if not (host and gpu and cloud):
        return
    with _BROKEN_LOCK:
        _BROKEN_PAIRS[(gpu, cloud)] = str(host)
    _update_host(host, lambda v: v.update(gpu=gpu, cloud=cloud, last=time.time()))   # met again: the demotion renews


BROKEN_PAIR_TTL_S = 3 * 86400   # a remembered (gpu, cloud) demotion lapses this long after its host last failed / was met
KNOWN_BAD_SINCE = 1792620000.0  # 2026-10-21: when the seed host last failed (its entries carry no `last`)


def broken_pairs(now: float | None = None) -> dict[tuple[str, str], str]:
    """(gpu, cloud) -> host to try later: hosts denied right now whose GPU / cloud is known (seed + bad_hosts.json) and
    met within BROKEN_PAIR_TTL_S, and every broken host met by this process. The host itself stays denied (a pod that
    lands on it is deleted at once); only the pair's demotion lapses, since Runpod moves hosts between pools."""
    now = time.time() if now is None else now
    out = {(v["gpu"], v["cloud"]): h for h, v in bad_hosts().items()
           if v.get("gpu") and v.get("cloud") and _deny_reason(v, now)
           and now - float(v.get("last") or KNOWN_BAD_SINCE) < BROKEN_PAIR_TTL_S}
    with _BROKEN_LOCK:
        out.update(_BROKEN_PAIRS)
    return out


def _update_host(host: str, fn) -> None:
    """Read-modify-write one bad_hosts.json entry under the ledger lock. Never fatal."""
    try:
        with ledger_lock():
            try:
                d = json.loads(BAD_HOSTS.read_text())
            except (OSError, ValueError):
                d = {}
            v = d.setdefault("hosts", {}).setdefault(str(host), {"fails": 0, "ok": 0})
            fn(v)
            BAD_HOSTS.parent.mkdir(parents=True, exist_ok=True)
            tmp = BAD_HOSTS.with_name(f"{BAD_HOSTS.name}.{os.getpid()}.{threading.get_ident()}.tmp")
            tmp.write_text(json.dumps(d, indent=1))
            os.replace(tmp, BAD_HOSTS)
    except Exception:  # noqa: BLE001
        pass


def note_host(host: str | None, ok: bool, reason: str = "", gpu: str | None = None, cloud: str | None = None) -> None:
    """Count a CUDA check result for a host (the deny list learns); a failure also remembers its (gpu, cloud). Never fatal."""
    if not host:
        return
    now = time.time()

    def upd(v):
        if ok:
            v["ok"] = int(v.get("ok") or 0) + 1
            v.pop("deny_until", None)
        else:
            v["fails"] = int(v.get("fails") or 0) + 1
            v["deny_until"] = now + BAD_HOST_DENY_S
            v["reason"] = reason[:200]
            if gpu and cloud:
                v.update(gpu=gpu, cloud=cloud)
        v["last"] = now
    _update_host(host, upd)
    if not ok and gpu and cloud:
        with _BROKEN_LOCK:
            _BROKEN_PAIRS[(gpu, cloud)] = str(host)


def preflight(p: "Pod") -> tuple[bool, str]:
    """(ok, detail): nvidia-smi and one CUDA matmul with the image's torch, before anything is uploaded. An ssh problem is
    reported as ok=None-like (True, 'unchecked') so a slow sshd never condemns a host."""
    try:
        out = p.output(PREFLIGHT_CMD, timeout=PREFLIGHT_TIMEOUT_S)
        lines = out.strip().splitlines() or ["ok"]
        return True, next((x for x in lines if x.startswith("PREFLIGHT_CUDA_OK")), lines[-1])[:160]
    except subprocess.TimeoutExpired:
        return False, "CUDA check timed out"
    except RuntimeError as e:
        msg = str(e)
        # Pod.output's message quotes the command (which names the PREFLIGHT_ markers): only the exit code tells
        if re.search(r"pod command failed \((3|4)\)", msg):
            return False, ("nvidia-smi failed" if "failed (3)" in msg else "CUDA failed") + ": " + msg[-160:]
        return True, f"unchecked: {msg[-120:]}"


def option_order(gpus: list[str], clouds: list[str], broken: dict | None = None) -> tuple[list[tuple[str, str]], list]:
    """(gpu, cloud) pairs in the order pod_session tries them: GPU by GPU (fastest first), each GPU in every cloud before the
    next, slower GPU (the 2026-10-21 trial walked every community GPU first and ended on slower cards and the broken host).
    A pair that led to a broken host (`broken`, remote.broken_pairs) moves behind the NEXT GPU type, not behind every slower
    card: landing on the broken host again costs ~30-50 s and $0.006 (the CUDA check deletes it), a 3090 instead of an L40S
    costs far more time (review 2026-10-21: speed over cost). -> (order, the demoted pairs)."""
    bad = [(g, c) for g in gpus for c in clouds if (g, c) in (broken or {})]
    out, carry = [], []
    for g in gpus:
        out += [(g, c) for c in clouds if (g, c) not in bad]
        out += carry                      # the previous GPU's broken pairs, after this GPU's good ones
        carry = [(g, c) for c in clouds if (g, c) in bad]
    return out + carry, bad


_GPU_MEMO = threading.local()   # pod_session reads the GPU catalog once per round, not once per (gpu, cloud) (~0.7 s each)


@contextlib.contextmanager
def pod_session(name: str, max_minutes: float, gpus: list[str] | None = None, image: str = DEFAULT_IMAGE,
                disk_gb: int = 80, clouds: list[str] | None = None, start_timeout_min: float = 6,
                max_start_attempts: int = 2, max_cost: float | None = None):
    """Create a pod, yield it, and always delete it. Refuses to start if the budget can't cover max_minutes.

    max_cost: a per-session ceiling ($) on top of the global cap: GPU/cloud options whose listed price x
    (max_minutes + start timeout) exceeds it are skipped (the on-pod kill switch makes that the worst case).
    Options are tried fastest GPU first (option_order). The yielded Pod carries `choice` = what was tried and why; when the
    session max or the remaining budget ruled out a FASTER GPU than the one picked, choice["warning"] says so (also printed)
    so a slower card is never chosen silently. A failure raises with the same `choice` attached (e.choice).
    """
    ensure_ssh_key()
    last_err = None
    pod = None
    attempts = 0
    broken = broken_pairs()
    order, demoted = option_order(gpus or GPU_PREFERENCE, clouds or CLOUDS, broken)
    choice: dict = {"order": [f"{g} ({c})" for g, c in order], "max_cost": max_cost, "max_minutes": max_minutes,
                    "tried": [], "demoted": [{"gpu": g, "cloud": c, "host": broken[(g, c)]} for g, c in demoted]}
    for g, c in demoted:
        print(f"[runpod] {short_gpu(g)} ({c}) is tried after the next GPU type: it led to the broken host {broken[(g, c)]}")
    started = False   # a create was attempted (a pure budget refusal stays a BudgetError, as before)
    _GPU_MEMO.on, _GPU_MEMO.rows = True, None
    try:
        for gpu, cloud in order:
            row: dict = {"gpu": gpu, "cloud": cloud}
            choice["tried"].append(row)
            price = gpu_price(gpu, cloud)
            row["price"] = price
            if price is None:
                row["result"] = "no stock"
                print(f"[runpod] {gpu} ({cloud}): not offered / no stock right now")
                continue
            # budget check uses the listed price BEFORE anything is created
            need = price * (max_minutes + start_timeout_min) / 60
            row["need"] = round(need, 2)
            if max_cost is not None and need > max_cost:
                row["result"] = "over the session max"
                print(f"[runpod] skip {gpu} ({cloud}) ${price:.2f}/h: worst case ${need:.2f} > session max ${max_cost:.2f}")
                last_err = BudgetError(f"every option exceeds the session max ${max_cost:.2f}")
                continue
            rem = remaining()
            if need > rem:   # a cheaper (slower) option may still fit: try it rather than stop here
                row["result"] = "over the remaining budget"
                last_err = BudgetError(f"{gpu}: {max_minutes + start_timeout_min:.0f} min at ${price:.2f}/h = ${need:.2f} > "
                                       f"remaining ${rem:.2f} of the ${cap():.2f} cap")
                print(f"[runpod] skip {gpu} ({cloud}): {last_err}")
                continue
            # 22/tcp must be requested at creation: without it Runpod gives only its ssh proxy, which can't scp
            args = ["pod", "create", "--name", f"dancecap-{name}", "--image", image, "--gpu-id", gpu,
                    "--container-disk-in-gb", str(disk_gb), "--cloud-type", cloud, "--ports", "22/tcp"]
            if cloud == "COMMUNITY":
                args.append("--public-ip")
            started = True
            _GPU_MEMO.rows = None   # stock moves while a pod starts: the next option reads the catalog again
            try:
                created = _ctl(*args, timeout=180)  # no --wait: record the pod in the ledger the moment it exists
            except Exception as e:  # no capacity for this GPU/cloud: try the next one
                row.update(result="create failed", error=str(e)[:200])
                print(f"[runpod] {gpu} ({cloud}): create failed: {str(e)[:200]}")
                last_err = e
                pid = _id_from_error(str(e))
                if pid:
                    _record(pid, name, gpu, cloud, price, max_minutes)
                    row["pod_id"] = pid   # analyze.run_pods charges every pod a session made to its run, exactly
                    _delete(pid, reason="create errored")
                continue
            pod_id = created["id"]
            cost = float(created.get("costPerHr") or price)
            _record(pod_id, name, gpu, cloud, cost, max_minutes)
            row["pod_id"] = pod_id
            print(f"[runpod] created {pod_id}: {gpu} ({cloud}) ${cost:.2f}/h; waiting up to {start_timeout_min:.0f} min "
                  f"for the container")
            try:
                pod = _wait_running(pod_id, start_timeout_min * 60)
                why_bad = None
                if pod:
                    host, port = _pod_ssh(pod)
                    denied = host_denied(host)
                    if denied:
                        why_bad = f"host {host} is on the broken-host list ({denied})"
                        remember_broken(host, gpu, cloud)
                    else:
                        p = Pod(pod_id, name, gpu, cost, host, port)
                        for _ in range(30):  # sshd answers before keys are installed
                            try:
                                if p.run("true", check=False, timeout=40) == 0:
                                    break
                            except TimeoutError:
                                pass
                            time.sleep(5)
                        ok, detail = preflight(p)
                        if detail.startswith("unchecked") or not ok:
                            print(f"[runpod] CUDA check on {host}: {detail}")
                        if ok:
                            if not detail.startswith("unchecked"):
                                note_host(host, True)
                                print(f"[runpod] CUDA check ok on {host}: {detail}")
                        else:
                            note_host(host, False, detail, gpu=gpu, cloud=cloud)
                            why_bad = f"host {host} failed the CUDA check: {detail}"
            except BaseException:  # Ctrl-C / crash while waiting: never leave the pod behind
                _delete(pod_id, reason="interrupted while starting")
                raise
            if pod and not why_bad:
                row["result"] = "ok"
                break
            if why_bad:
                row.update(result="broken host", host=why_bad.split()[1])
                last_err = RuntimeError(why_bad)
                print(f"[runpod] {why_bad}: deleting {pod_id} and trying the next option")
                _delete(pod_id, reason=why_bad[:120])
                pod = None
                attempts += 1
                if attempts >= max_start_attempts + 1:  # one extra try for a broken host
                    break
                continue
            row["result"] = "unreachable"
            last_err = RuntimeError(f"{gpu} ({cloud}) not reachable over direct ssh in {start_timeout_min:.0f} min")
            _delete(pod_id, reason="not reachable over direct ssh")
            attempts += 1
            if attempts >= max_start_attempts:  # don't walk the whole GPU list paying for start-ups
                break
    finally:
        _GPU_MEMO.on, _GPU_MEMO.rows = False, None
    if not pod:
        raise _no_pod_error(last_err, choice, started)

    try:
        try:   # bookkeeping only: it must never keep the pod from its delete below
            choice["chosen"] = {"gpu": gpu, "cloud": cloud, "price": cost, "rank": order.index((gpu, cloud))}
            warn = excluded_faster(choice)
            if warn:
                choice.update(warn)
                print(f"[runpod] NOTE: {warn['warning']}")
        except Exception as e:  # noqa: BLE001
            choice["error"] = f"{type(e).__name__}: {e}"[:200]
        p.choice = choice
        # on-pod kill switch: Runpod injects a pod-scoped RUNPOD_API_KEY and runpodctl into pods. ssh sessions do
        # not always inherit the container environment, so take the two variables from PID 1.
        p.killswitch = arm_killswitch(p, max_minutes)
        print(f"[runpod] pod {pod_id} {gpu} ${cost:.2f}/h, ssh {host}:{port}, auto-delete in {max_minutes:.0f} min; "
              f"budget remaining ${remaining():.2f}")
        yield p
    finally:
        try:
            _delete(pod_id, reason="session end")
        except PodDeleteError as e:   # already reported loudly; do not hide the exception that ended the session
            print(f"[runpod] {e}", flush=True)


_BUDGET_SKIPS = ("over the session max", "over the remaining budget")


def excluded_faster(choice: dict) -> dict | None:
    """{warning, excluded_faster: [...]} when the session max or the remaining budget ruled out a FASTER GPU type than the one
    picked (an option tried before it in the speed order); None when the fastest affordable card got the pod. 2026-10-21 trial:
    the session max put a fit on an A40 without a word, and the agent only noticed from the run time."""
    chosen = choice.get("chosen") or {}
    rows = choice.get("tried") or []
    k = next((i for i, r in enumerate(rows) if r.get("result") == "ok"), len(rows))
    faster = [r for r in rows[:k] if r.get("result") in _BUDGET_SKIPS and r.get("gpu") != chosen.get("gpu")]
    if not faster:
        return None
    best = faster[0]   # the fastest one that was ruled out
    by = (f"the session max ${choice['max_cost']:.2f}" if best["result"] == "over the session max"
          else "the remaining Runpod budget")
    fix = (f"--max-cost {best['need']:.2f} or a shorter --max-minutes would allow it"
           if best["result"] == "over the session max" else "the account balance would have to grow")
    return {"excluded_faster": [{x: r.get(x) for x in ("gpu", "cloud", "price", "need", "result")} for r in faster],
            "warning": (f"{by} ruled out the faster {short_gpu(best['gpu'])} ({best['cloud']}, ${best['price']:.2f}/h, "
                        f"worst case ${best['need']:.2f} for {choice.get('max_minutes', 0):.0f} min); this session runs "
                        f"on the slower {short_gpu(chosen.get('gpu'))}: {fix}")}


def _no_pod_error(last_err, choice: dict, started: bool) -> Exception:
    """The exception for 'no pod': a pure refusal of the remaining budget stays a BudgetError (callers such as fit_full catch
    that type); otherwise RuntimeError('no GPU pod could be started: ...'), naming the max-cost a session max would need."""
    rows = choice.get("tried") or []
    priced = [r for r in rows if r.get("price") is not None]
    if not started and priced and all(r.get("result") == "over the remaining budget" for r in priced):
        err: Exception = BudgetError(str(last_err))
    else:
        msg = f"no GPU pod could be started: {last_err}"
        over = [r for r in priced if r.get("result") == "over the session max"]
        if over and not started:
            cheapest = min(over, key=lambda r: r["need"])
            msg += (f" (the cheapest, {short_gpu(cheapest['gpu'])} ({cheapest['cloud']}), needs ${cheapest['need']:.2f}; "
                    f"the fastest, {short_gpu(over[0]['gpu'])} ({over[0]['cloud']}), ${over[0]['need']:.2f})")
        if not priced:
            msg += f" (no stock for any of {len(rows)} GPU/cloud options)"
        err = RuntimeError(msg)
    err.choice = choice   # type: ignore[attr-defined]  - remote_jobs puts it in the session state
    return err


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


def _record(pod_id: str, name: str, gpu: str, cloud: str, cost: float, max_minutes: float, owner: str | None = None) -> None:
    """Open a ledger row the moment a pod exists. `owner` (optional, e.g. "farm:<dance>") says who may adopt / delete it."""
    with ledger_lock():
        rows = ledger()
        if not any(r["pod_id"] == pod_id for r in rows):
            row = {"pod_id": pod_id, "name": name, "gpu": gpu, "cloud": cloud, "cost_per_hr": cost,
                   "created": datetime.now(timezone.utc).isoformat(), "created_ts": time.time(), "max_minutes": max_minutes}
            if owner:
                row["owner"] = owner
            rows.append(row)
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


class PodDeleteError(RuntimeError):
    """A pod delete that could not be confirmed. The ledger row stays OPEN and the pod must be treated as still billing."""


DELETE_WAITS = (3.0, 8.0, 20.0)   # pause before attempt 2, 3, 4
DELETE_CONFIRM_WAIT = 4.0         # between the 3 looks at the pod list that confirm a delete
DELETE_CALL_TIMEOUT = 90.0        # one `runpodctl pod delete` / `pod list` call (the old default of 900 s could freeze a watcher cycle)
GONE_STATUSES = ("EXITED", "TERMINATED", "STOPPED", "DELETED")
_NOT_FOUND = ("not found", "no such", "does not exist", "404", "unknown pod")


def listed_pods() -> dict[str, dict] | None:
    """{pod id: record} of Runpod's pod list, or None when it could not be read (an error object is NOT an empty list)."""
    try:
        rc, data = _ctl_rc("pod", "list", timeout=DELETE_CALL_TIMEOUT)
    except Exception:  # noqa: BLE001 - subprocess timeout, no key, runpodctl missing
        return None
    rows = data if isinstance(data, list) else (data.get("pods") if isinstance(data, dict) else None)
    if rc != 0 or not isinstance(rows, list):
        return None
    return {p["id"]: p for p in rows if isinstance(p, dict) and p.get("id")}


def _pod_present(rec: dict | None) -> bool:
    return rec is not None and str(rec.get("desiredStatus") or rec.get("status") or "").upper() not in GONE_STATUSES


def _delete(pod_id: str, reason: str) -> None:
    """Delete a pod, CONFIRM it is gone (absent from Runpod's pod list), retry with backoff, and only then close the ledger row.
    `runpodctl pod delete` prints a JSON error and exits 1 on a rejected call (401, 5xx, timeout): the old code ignored that,
    closed the row and let a billing pod disappear from the budget. Raises PodDeleteError after the last attempt."""
    err = "?"
    confirmed = unconfirmed = False
    for i in range(len(DELETE_WAITS) + 1):
        try:
            rc, out = _ctl_rc("pod", "delete", pod_id, timeout=DELETE_CALL_TIMEOUT)
        except Exception as e:  # noqa: BLE001
            rc, out = -1, f"{type(e).__name__}: {e}"
        text = json.dumps(out) if not isinstance(out, str) else out
        already_gone = rc != 0 and any(x in text.lower() for x in _NOT_FOUND)
        call_ok = (rc == 0 and not (isinstance(out, dict) and out.get("error"))) or already_gone
        if call_ok:
            for _ in range(3):
                live = listed_pods()
                if live is not None and not _pod_present(live.get(pod_id)):
                    confirmed = True
                    break
                err = "pod list unreadable after the delete" if live is None else "pod still listed after the delete"
                time.sleep(DELETE_CONFIRM_WAIT)
            if not confirmed and err.startswith("pod list unreadable") and rc == 0:
                confirmed = unconfirmed = True   # the API accepted the delete but cannot be asked: reconcile_ledger looks again later
            if confirmed:
                break
        else:
            err = f"rc={rc}: {text[:200]}"
        print(f"[runpod] delete of {pod_id} not confirmed (attempt {i + 1}/{len(DELETE_WAITS) + 1}): {err}", flush=True)
        if i < len(DELETE_WAITS):
            time.sleep(DELETE_WAITS[i])
    with ledger_lock():
        rows = ledger()
        for r in rows:
            if r["pod_id"] != pod_id or r.get("deleted_ts"):
                continue
            if confirmed:
                r["deleted_ts"] = time.time()
                r["deleted_reason"] = reason + (" (delete accepted, not re-listed)" if unconfirmed else "")
                r["cost"] = round(r["cost_per_hr"] * (r["deleted_ts"] - r["created_ts"]) / 3600, 4)
                r.pop("delete_error", None)
                if unconfirmed:
                    r["delete_unconfirmed"] = True
            else:
                r["delete_error"] = err[:300]
                r["delete_failed_ts"] = time.time()
        _save_ledger(rows)
    if not confirmed:
        print(f"[runpod] WARNING: pod {pod_id} could NOT be deleted ({err}); its ledger row stays open and it may still bill. "
              f"Delete it by hand, or let its on-pod kill switch do it.", flush=True)
        raise PodDeleteError(f"pod {pod_id}: delete not confirmed: {err}")
    print(f"[runpod] pod {pod_id} deleted ({reason}); spent ${spent():.2f} of ${cap():.2f}")


def reconcile_ledger(confirm_s: float = 60.0, min_age_s: float = 300.0) -> dict:
    """Compare the OPEN ledger rows with Runpod's pod list. A row whose pod is listed is stamped alive (it keeps accruing past its
    max_minutes); a row whose pod is absent is flagged `missing_ts`, and closed when it is still absent `confirm_s` later (two looks,
    never one). Rows younger than `min_age_s` are left alone (a pod may not be listed yet). The closed row's cost is capped by the
    pod's kill switch (max_minutes). Never calls Runpod to delete anything."""
    live = listed_pods()
    if live is None:
        return {"error": "pod list unreadable", "closed": [], "flagged": [], "alive": []}
    now = time.time()
    closed, flagged, alive = [], [], []
    with ledger_lock():
        rows = ledger()
        for r in rows:
            if r.get("deleted_ts"):
                continue
            pid = r["pod_id"]
            if _pod_present(live.get(pid)):
                r["alive_ts"] = now
                r.pop("missing_ts", None)
                alive.append(pid)
            elif now - float(r["created_ts"]) < min_age_s:
                continue
            elif not r.get("missing_ts"):
                r["missing_ts"] = now
                flagged.append(pid)
            elif now - float(r["missing_ts"]) >= confirm_s:
                kill = float(r["created_ts"]) + float(r.get("max_minutes") or 0) * 60
                end = min(float(r["missing_ts"]), max(float(r["alive_ts"]), kill) if r.get("alive_ts") else kill)
                end = max(end, float(r["created_ts"]))
                r["deleted_ts"] = end
                r["deleted_reason"] = "reconciled: pod absent from Runpod's list twice"
                r["cost"] = round(r["cost_per_hr"] * (end - r["created_ts"]) / 3600, 4)
                r["reconciled_ts"] = now
                closed.append(pid)
        _save_ledger(rows)
    return {"closed": closed, "flagged": flagged, "alive": alive, "spent": round(spent(), 2), "remaining": round(remaining(), 2)}


def gpu_price(gpu: str, cloud: str = "COMMUNITY") -> float | None:
    """On-demand $/h for a GPU type in a cloud, or None if that cloud doesn't offer it / has no stock."""
    for g in gpu_list():
        if g.get("gpuId") == gpu or g.get("displayName") == gpu:
            if not g.get("communityCloud" if cloud == "COMMUNITY" else "secureCloud"):
                return None
            v = g.get("communityPricePerHr" if cloud == "COMMUNITY" else "securePricePerHr")
            return float(v) if v else None
    return None


def gpu_table(min_gb: float | None = None) -> list[dict]:
    """The GPU catalog for people (`remote gpus`, the monitor's /api/runpod/gpus). Default: the GPU_LADDER cards in the
    order the runner tries them (fastest first), each with its memory, prices, stock and data centers - the cards it picks
    first (4090, the Ada 48 GB cards) used to be missing from a list of 80 GB cards sorted by price. min_gb: the old
    table, every catalog GPU with at least that much memory, cheapest first."""
    if min_gb is None:
        cat = {g.get("gpuId"): g for g in gpu_list()}
        out = []
        for i, (gid, gb) in enumerate(GPU_LADDER):
            g = cat.get(gid) or {}
            stocked = [d["dataCenterId"] for d in g.get("dataCenterAvailability", []) or []
                       if d.get("stockStatus") not in (None, "none")]
            out.append({"rank": i + 1, "gpu": gid, "gb": g.get("memoryInGb") or gb,
                        "community": g.get("communityPricePerHr") if g.get("communityCloud", True) else None,
                        "secure": g.get("securePricePerHr") if g.get("secureCloud", True) else None,
                        "stock": g.get("stockStatus") if g else "not offered", "datacenters": stocked})
        return out
    rows = []
    for g in gpu_list():
        if (g.get("memoryInGb") or 0) < min_gb:
            continue
        stocked = [d["dataCenterId"] for d in g.get("dataCenterAvailability", []) if d.get("stockStatus") not in (None, "none")]
        rows.append({"gpu": g["gpuId"], "gb": g["memoryInGb"], "community": g.get("communityPricePerHr"),
                     "secure": g.get("securePricePerHr"), "stock": g.get("stockStatus"), "datacenters": stocked})
    return sorted(rows, key=lambda r: r["community"] or r["secure"] or 99)


def gpu_list() -> list[dict]:
    rows = getattr(_GPU_MEMO, "rows", None)   # inside pod_session's option loop only (thread-local)
    if rows is not None:
        return rows
    data = _ctl("gpu", "list")
    rows = data if isinstance(data, list) else data.get("gpus", data.get("items", []))
    if getattr(_GPU_MEMO, "on", False):
        _GPU_MEMO.rows = rows
    return rows


def cleanup_orphans(delete: bool = False, only: list[str] | None = None) -> list[str]:
    """dancecap-* pods that Runpod lists. A DRY RUN unless delete=True: the live fit_full sessions run in pods named
    dancecap-fit-<seg>, so a blanket delete of the prefix would kill work in progress. `only` limits it to those pod ids."""
    pods = _ctl("pod", "list")
    pods = pods if isinstance(pods, list) else pods.get("pods", [])
    found = [p["id"] for p in pods if str(p.get("name", "")).startswith("dancecap-") and (not only or p["id"] in only)]
    if delete:
        for pid in found:
            _delete(pid, reason="orphan cleanup")
    return found


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
