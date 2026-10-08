"""Small shared helpers: tolerant JSON reads, atomic writes, redaction, input validation, process checks.

Windows rules (PIPELINE_MONITOR 3.6, PIPELINE_PROGRESS 5.2): open, read, close at once (a held handle makes another
writer's os.replace fail); our own writes go to a temp file and os.replace with retries on PermissionError.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

TAKE_RE = re.compile(r"^[a-z0-9][a-z0-9_]{0,47}$")
CAM_RE = re.compile(r"^[0-9]{2}$")
CAPTURE_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
STAGE_RE = re.compile(r"^[a-z0-9_]{1,40}$")
RUN_RE = re.compile(r"^r-\d{8}-\d{6}-[0-9a-f]{4}$")
SID_RE = re.compile(r"^[0-9]{8}-[0-9]{6}-[A-Za-z0-9_.-]{1,80}$")
POD_ID_RE = re.compile(r"^[a-z0-9]{6,32}$")

_RPA = re.compile(r"rpa_[A-Za-z0-9]+")
_LONG = re.compile(r"[A-Za-z0-9_-]{32,}")
_HEX = re.compile(r"^[0-9a-f]+$")


def redact(text: str) -> str:
    """Mask Runpod keys and token-like runs; keep pure lowercase hex (sha256, bundle keys) readable."""
    if not text:
        return text
    text = _RPA.sub("rpa_***", text)
    return _LONG.sub(lambda m: m.group(0) if _HEX.match(m.group(0)) else m.group(0)[:4] + "***", text)


def read_text(path: Path, default: str | None = None, encoding: str = "utf-8-sig") -> str | None:
    try:
        with open(path, "r", encoding=encoding, errors="replace") as fh:
            return fh.read()
    except OSError:
        return default


def read_json(path: Path, default=None, retries: int = 3):
    """Tolerant read: a file being truncate-written may be partial; retry, then give up with `default`."""
    for i in range(retries):
        try:
            with open(path, "r", encoding="utf-8-sig") as fh:
                return json.load(fh)
        except FileNotFoundError:
            return default
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            if i == retries - 1:
                return default
            time.sleep(0.1)
    return default


def write_text_atomic(path: Path, text: str, encoding: str = "utf-8") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding=encoding, newline="\n") as fh:
        fh.write(text)
    for i in range(4):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if i == 3:
                try:
                    tmp.unlink()
                except OSError:
                    pass
                raise
            time.sleep(0.15 * (i + 1))


def write_json_atomic(path: Path, obj, indent: int | None = 1) -> None:
    write_text_atomic(path, json.dumps(obj, indent=indent, ensure_ascii=False, default=str) + "\n")


def append_jsonl(path: Path, obj) -> None:
    """One line, one write call (never fatal for the caller's work: errors are swallowed)."""
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(obj, ensure_ascii=False, default=str) + "\n"
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line)
    except OSError:
        pass


def read_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    out = []
    txt = read_text(path, "")
    for line in (txt or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out[-limit:] if limit else out


def sha1_bytes(b: bytes) -> str:
    return "sha1:" + hashlib.sha1(b).hexdigest()


def sha256_16(path: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()[:16]
    except OSError:
        return None


def mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def pid_alive(pid: int | None, create_time: float | None = None) -> bool:
    """True when the process exists (and, when given, started at create_time: pids get reused)."""
    if not pid:
        return False
    try:
        import psutil
        p = psutil.Process(int(pid))
        if create_time is not None and abs(p.create_time() - float(create_time)) > 2.0:
            return False
        return p.is_running() and p.status() != psutil.STATUS_ZOMBIE
    except ImportError:
        pass
    except Exception:
        return False
    if os.name == "nt":
        import ctypes
        h = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
        if not h:
            return False
        code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(h)
        return code.value == 259
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def kill_tree(pid: int, include_parent: bool = True, timeout: float = 5.0) -> int:
    """Terminate a process tree (psutil). Returns the number of processes signalled."""
    try:
        import psutil
    except ImportError:
        return 0
    try:
        parent = psutil.Process(int(pid))
    except Exception:
        return 0
    procs = parent.children(recursive=True)
    if include_parent:
        procs.append(parent)
    for p in procs:
        try:
            p.terminate()
        except Exception:
            pass
    _, alive = psutil.wait_procs(procs, timeout=timeout)
    for p in alive:
        try:
            p.kill()
        except Exception:
            pass
    return len(procs)


def fmt_min(s: float | None) -> str:
    if s is None:
        return "-"
    m = s / 60
    return f"{m / 60:.1f} h" if m >= 90 else f"{m:.0f} min"
