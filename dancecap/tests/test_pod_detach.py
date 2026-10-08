"""pod_detach: a detached command survives a lost attach (BrokenPipe = the ssh connection) and is re-attached to, never
started twice; a finished command is never re-run; rc 255 is reserved for "connection lost". Needs a POSIX-like bash
with `timeout` (Git Bash on the workstation, bash on a pod)."""
from __future__ import annotations

import io
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from dancecap import pod_detach as pd


def _bash_ok() -> bool:
    b = shutil.which("bash")
    if not b or "system32" in b.lower():  # WSL's bash.exe is not what the pods run (and is banned here)
        return False
    try:
        return subprocess.run([b, "-c", "timeout 5 true"], capture_output=True, timeout=20).returncode == 0
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(not _bash_ok(), reason="needs a POSIX bash with `timeout`")


class _Broken(io.StringIO):
    """A stdout whose reader (the ssh connection) is gone after `n` writes."""

    def __init__(self, n: int):
        super().__init__()
        self.n = n

    def write(self, s):
        self.n -= 1
        if self.n < 0:
            raise BrokenPipeError
        return super().write(s)


def test_finished_command_rc_and_not_rerun(tmp_path: Path):
    st, log = tmp_path / "st", tmp_path / "x.log"
    out = io.StringIO()
    rc = pd.run("a", "echo hello; exit 3", log, 1, st, out=out)
    assert rc == 3 and "hello" in out.getvalue()
    marker = tmp_path / "ran_again"
    out2 = io.StringIO()
    rc2 = pd.run("a", f"touch {marker.as_posix()}; exit 0", log, 1, st, out=out2)
    assert rc2 == 3 and not marker.exists()  # the recorded rc is returned, the command is not started again


def test_255_is_mapped_away(tmp_path: Path):
    rc = pd.run("b", "exit 255", tmp_path / "b.log", 1, tmp_path / "st", out=io.StringIO())
    assert rc == 254 != pd.RC_SSH_LOST


def test_connection_loss_then_reattach_continues_same_child(tmp_path: Path):
    st, log = tmp_path / "st", tmp_path / "c.log"
    cmd = "for i in 1 2 3 4 5 6; do echo tick $i; sleep 1; done; echo finished-ok"
    rc = pd.run("c", cmd, log, 1, st, out=_Broken(1))  # the ssh pipe dies after the first write
    assert rc == pd.RC_SSH_LOST
    pid1 = (st / "c.pid").read_text()
    assert pd.status("c", st)["alive"] and not (st / "c.rc").exists()  # the child kept running
    out = io.StringIO()
    rc = pd.run("c", cmd, log, 1, st, out=out)  # same command again = attach
    assert rc == 0 and "already running" in out.getvalue() and "finished-ok" in out.getvalue()
    assert (st / "c.pid").read_text() == pid1  # no second child was started
    assert log.read_text().count("start c") == 1


def test_vanished_child_reports_250(tmp_path: Path):
    st, log = tmp_path / "st", tmp_path / "d.log"
    st.mkdir()
    (st / "d.pid").write_text("999999")  # a pid that does not exist, no rc
    t0 = time.time()
    rc = pd.attach("d", log, st, poll_s=0.5, out=io.StringIO())
    assert rc == pd.RC_VANISHED and time.time() - t0 < 30


def test_timeout_kills_child(tmp_path: Path):
    rc = pd.run("e", "sleep 60", tmp_path / "e.log", 0.02, tmp_path / "st", out=io.StringIO())  # 1.2 s
    assert rc == pd.RC_TIMEOUT


def test_detached_command_string_is_shell_safe():
    s = pd.detached("fit_c8", "cd /x && python -m dancecap mamma fit t --tag c8", "fit", 95)
    assert s.startswith("python -m dancecap.pod_detach run --name fit_c8 --log \"")
    assert "$DANCECAP_SID" in s and "pod_logs/fit/fit_c8.child.log" in s and "--timeout-min 95" in s
    assert pd.MARK in s


class _FakePod:
    id = "p1"

    def __init__(self, rcs):
        self.rcs = list(rcs)
        self.calls = []


def test_resilient_run_reattaches_while_connection_is_down(monkeypatch, tmp_path: Path):
    from dancecap import remote

    calls = []
    rcs = [255, 255, 0]

    def base_run(self, cmd, check=True, timeout=None, log=None, echo=True):
        calls.append(cmd)
        if cmd == "true":
            return 0
        return rcs.pop(0)

    monkeypatch.setattr(remote.Pod, "run", base_run)
    monkeypatch.setattr(pd.time, "sleep", lambda s: None)
    pod = remote.Pod("p1", "n", "gpu", 1.0, "h", 22)
    pd.make_resilient(pod)
    log = tmp_path / "r.log"
    rc = pod.run("python -m dancecap.pod_detach run --name x -- 'sleep 1'", check=False, timeout=600, log=log)
    assert rc == 0
    assert sum(1 for c in calls if "pod_detach" in c) == 3  # attach, attach, attach (the third one reaches the rc)
    assert "ssh to pod p1 lost" in log.read_text()
    # a command that is not detached is passed through untouched (a real 255 stays 255)
    rcs[:] = [255]
    assert pod.run("echo hi", check=False) == 255


def test_resilient_run_gives_up_after_max_lost(monkeypatch, tmp_path: Path):
    from dancecap import remote

    def base_run(self, cmd, check=True, timeout=None, log=None, echo=True):
        return 255 if cmd != "true" else 1  # never reconnects

    monkeypatch.setattr(remote.Pod, "run", base_run)
    monkeypatch.setattr(pd.time, "sleep", lambda s: None)
    t = [1000.0]
    monkeypatch.setattr(pd.time, "time", lambda: t.__setitem__(0, t[0] + 600) or t[0])  # 10 min per call
    monkeypatch.setattr(remote, "_ctl", lambda *a, **k: {"id": "p1", "desiredStatus": "RUNNING"})
    pod = remote.Pod("p1", "n", "gpu", 1.0, "h", 22)
    pd.make_resilient(pod, max_lost_min=30)
    assert pod.run("python -m dancecap.pod_detach run --name x -- 'sleep 1'", check=False, timeout=10 ** 9) == 255
