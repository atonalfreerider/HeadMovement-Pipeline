"""Offline tests for dancecap.remote: pod-record parsing and budget accounting (no Runpod calls)."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dancecap import remote  # noqa: E402

# Real shapes observed on 2026-10-06 (ids/IPs replaced by documentation values; no secrets).
MCP_SHAPE = {  # REST v2 via the Runpod MCP server, pod created WITH --ports 22/tcp
    "id": "podidexample01", "status": "RUNNING",
    "runtime": {"ports": [{"ip": "100.64.0.10", "private": 19123, "public": 60758, "type": "http"},
                          {"ip": "203.0.113.10", "private": 22, "public": 11793, "type": "tcp"}], "uptime": 226},
    "ssh": {"direct": {"command": "ssh root@203.0.113.10 -p 11793", "host": "203.0.113.10", "port": 11793,
                       "username": "root"},
            "proxy": {"host": "ssh.runpod.io", "port": 22, "username": "podidexample01-0a1b2c3d"}},
}
RUNPODCTL_SHAPE = {  # runpodctl 2.14 `pod get`: ssh block flattened, runtime ports null
    "id": "podidexample02", "desiredStatus": "RUNNING", "runtime": {"ports": None, "uptime": None},
    "ssh": {"id": "podidexample02", "ip": "198.51.100.20", "port": 23017,
            "ssh_command": "ssh root@198.51.100.20 -p 23017"},
}
NO_TCP_SHAPE = {  # created WITHOUT --ports 22/tcp: only an internal http port + the proxy
    "id": "podidexample03", "status": "RUNNING",
    "runtime": {"ports": [{"ip": "100.64.0.10", "private": 19123, "public": 60757, "type": "http"}], "uptime": 223},
    "ssh": {"direct": None, "proxy": {"host": "ssh.runpod.io", "port": 22}},
}
LEGACY_SHAPE = {"id": "x", "runtime": {"ports": [{"ip": "1.2.3.4", "privatePort": 22, "publicPort": 40022,
                                                   "isIpPublic": True, "type": "tcp"}]}}


def test_direct_ssh_all_shapes():
    assert remote._direct_ssh(MCP_SHAPE) == ("203.0.113.10", 11793)
    assert remote._direct_ssh(RUNPODCTL_SHAPE) == ("198.51.100.20", 23017)
    assert remote._direct_ssh(LEGACY_SHAPE) == ("1.2.3.4", 40022)
    assert remote._direct_ssh(NO_TCP_SHAPE) is None


def test_internal_ip_is_not_direct_ssh():
    shape = {"ssh": {"ip": "100.64.0.10", "port": 60757}}
    assert remote._direct_ssh(shape) is None


def test_spent_counts_open_and_closed_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(remote, "LEDGER", tmp_path / "ledger.json")
    now = time.time()
    rows = [{"pod_id": "a", "cost_per_hr": 3.6, "created_ts": now - 3600, "deleted_ts": now - 3000},  # 10 min
            {"pod_id": "b", "cost_per_hr": 1.2, "created_ts": now - 600}]  # still open: 10 min so far
    (tmp_path / "ledger.json").write_text(json.dumps(rows))
    assert abs(remote.spent() - (0.6 + 0.2)) < 0.01
    monkeypatch.setenv("DANCECAP_RUNPOD_CAP", "1.0")
    assert abs(remote.remaining() - 0.2) < 0.01


def test_default_cap_is_30(monkeypatch):
    monkeypatch.delenv("DANCECAP_RUNPOD_CAP", raising=False)
    assert remote.DEFAULT_CAP == 30.0
    assert remote.cap() == 30.0


def test_create_args_request_direct_ssh(monkeypatch):
    calls = []

    def fake_ctl(*args, **kw):
        calls.append(args)
        raise RuntimeError("stop after create args")

    monkeypatch.setattr(remote, "_ctl", fake_ctl)
    monkeypatch.setattr(remote, "ensure_ssh_key", lambda: None)
    monkeypatch.setattr(remote, "gpu_price", lambda g, c: 1.0)
    monkeypatch.setattr(remote, "remaining", lambda: 100.0)
    try:
        with remote.pod_session("t", 5, gpus=["G"], clouds=["COMMUNITY"]):
            pass
    except RuntimeError:
        pass
    create = next(a for a in calls if a[:2] == ("pod", "create"))
    assert "--ports" in create and create[create.index("--ports") + 1] == "22/tcp"
    assert "--public-ip" in create
