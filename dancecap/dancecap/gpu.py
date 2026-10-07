"""GPU policy for a single 11 GB Turing card (RTX 2080 Ti, sm_75).

torch.cuda.is_bf16_supported() returns True on Turing because bf16 is *emulated* (measured ~18x slower
matmuls), so models that pick bf16 automatically must be overridden: use amp_dtype() everywhere.
Heavy jobs take an exclusive file lock so two stages never fight for VRAM.
"""
from __future__ import annotations

import contextlib
import os
import time
from pathlib import Path

LOCK = Path(__file__).resolve().parents[1] / "work" / "gpu.lock"


def amp_dtype():
    """fp16 on pre-Ampere GPUs (no native bf16), bf16 on Ampere+."""
    import torch
    if not torch.cuda.is_available():
        return torch.float32
    major, _ = torch.cuda.get_device_capability(0)
    return torch.bfloat16 if major >= 8 else torch.float16


@contextlib.contextmanager
def exclusive(label: str, timeout_s: float = 6 * 3600, poll_s: float = 5.0):
    """Hold work/gpu.lock for the duration of a heavy GPU job. Stale locks (dead pid) are taken over.

    DANCECAP_GPU_SHARED=1 skips the lock: only for an 80 GB pod GPU running several small jobs side by side
    (e.g. leave-one-camera-out MAMMA fits, ~2-3 GB each); never set it on the 11 GB workstation."""
    if os.environ.get("DANCECAP_GPU_SHARED") == "1":
        yield
        return
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + timeout_s
    while True:
        try:
            fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()} {label}".encode())
            os.close(fd)
            break
        except FileExistsError:
            try:
                pid = int(LOCK.read_text().split()[0])
                if not _alive(pid):
                    LOCK.unlink(missing_ok=True)
                    continue
            except (ValueError, IndexError, FileNotFoundError):
                LOCK.unlink(missing_ok=True)
                continue
            if time.time() > deadline:
                raise TimeoutError(f"GPU busy: {LOCK.read_text()}")
            time.sleep(poll_s)
    try:
        yield
    finally:
        _return_cached_vram()
        LOCK.unlink(missing_ok=True)


def _return_cached_vram() -> None:
    """Give PyTorch's cached (unused) VRAM back to the driver before the lock goes: a process that goes on with CPU
    work after its GPU block otherwise keeps it reserved, and the next stage on the 11 GB card gets less (on Windows
    the driver may then page to system RAM instead of failing: many times slower). Live tensors are not touched."""
    import sys
    torch = sys.modules.get("torch")
    try:
        if torch is not None and torch.cuda.is_initialized():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:  # never let cleanup mask the block's own error
        pass


def _alive(pid: int) -> bool:
    if os.name == "nt":
        import ctypes
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(handle)
        return code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False
