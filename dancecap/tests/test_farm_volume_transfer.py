"""Farm volume transfers and pod creation (mocked: a local bash plays the pod, a temp dir plays the volume).

Covers what test_farm_infra.py does not: the hash-diffed, striped, resumable upload (`Link.push_file`), resumable verified
downloads, small-file batches, the submit -> worker -> pull round trip, and the arguments `pods.create` sends to Runpod.

    ..\\.venv-nopo4d\\Scripts\\python.exe -m pytest tests/test_farm_volume_transfer.py -q      (from dancecap/)
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from dancecap import remote
from dancecap.farm import pods, volume, worker

BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(BASH is None, reason="needs bash to play the pod")


class LocalPod:
    """Stands in for remote.Pod: `output()` and the ssh argv run through a local bash, in a temp 'volume'."""
    id, name, gpu, cost_per_hr, host, port = "localpod", "local", "", 0.0, "localhost", 22

    def output(self, cmd: str, timeout: float = 300) -> str:
        p = subprocess.run([BASH, "-c", cmd], capture_output=True, text=True, timeout=timeout)
        if p.returncode != 0:
            raise RuntimeError(f"local command failed ({p.returncode}): {cmd[:120]}: {p.stderr[-200:]}")
        return p.stdout

    def run(self, cmd, check=True, timeout=None, log=None, echo=True):
        return subprocess.run([BASH, "-c", cmd]).returncode


class LocalLink(volume.Link):
    def _ssh(self, remote_cmd: str, compress: bool) -> list[str]:   # no ssh: run the remote command in a local bash
        return [BASH, "-c", remote_cmd]


@pytest.fixture
def link(tmp_path):
    root = (tmp_path / "vol").as_posix()
    os.makedirs(root)
    return LocalLink(LocalPod(), root=root)


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def blob(n: int, seed: int = 0) -> bytes:
    return (bytes(range(256)) * (n // 256 + 1))[seed:n + seed]


def test_push_file_skips_identical_and_resends_changed(tmp_path, link):
    f = tmp_path / "a.bin"
    f.write_bytes(blob(200_000))
    r = link.push_file(f, "models/a.bin")
    assert r["skipped"] is False and r["sent"] == 200_000
    assert Path(link.abs("models/a.bin")).read_bytes() == f.read_bytes()
    assert link.push_file(f, "models/a.bin")["skipped"] is True
    f.write_bytes(blob(200_000, seed=7))
    r = link.push_file(f, "models/a.bin")
    assert r["skipped"] is False and Path(link.abs("models/a.bin")).read_bytes() == f.read_bytes()
    assert not Path(link.abs("models/a.bin.part")).exists()


def test_striped_push_resumes_verified_prefixes_and_rejects_stale_parts(tmp_path, link, monkeypatch):
    monkeypatch.setattr(volume, "STRIPE_MIN", 1000)
    data = blob(3 * volume.CHUNK + 1234)                 # 3 stripes of 1 MiB + a short last one -> streams=4
    f = tmp_path / "big.bin"
    f.write_bytes(data)
    seg = volume.CHUNK
    dst = Path(link.abs("weights/big.bin"))
    dst.parent.mkdir(parents=True)
    # stripe 0: a correct 600 kB prefix from an interrupted upload; stripe 1: leftovers of a DIFFERENT file; stripe 2: full
    (dst.parent / "big.bin.part4.0").write_bytes(data[:600_000])
    (dst.parent / "big.bin.part4.1").write_bytes(b"\x07" * 400_000)
    (dst.parent / "big.bin.part4.2").write_bytes(data[2 * seg:3 * seg])
    r = link.push_file(f, "weights/big.bin", streams=4)
    assert r["streams"] == 4 and not r["skipped"]
    assert dst.read_bytes() == data
    assert r["sent"] == len(data) - 600_000 - seg         # resumed stripe 0, reused stripe 2, resent stripe 1 completely
    assert not list(dst.parent.glob("big.bin.part*"))


def test_push_file_detects_corruption_and_retries(tmp_path, link, monkeypatch):
    f = tmp_path / "c.bin"
    f.write_bytes(blob(50_000))
    real = volume.Link._upload_range
    calls = {"n": 0}

    def flaky(self, local, start, length, part, compress, report=None):
        calls["n"] += 1
        sent = real(self, local, start, length, part, compress, report)
        if calls["n"] == 1:                              # flip a byte on the 'remote' side after the first attempt
            with open(part, "r+b") as fh:
                fh.seek(10)
                fh.write(b"\xff\xfe")
        return sent

    monkeypatch.setattr(volume.Link, "_upload_range", flaky)
    monkeypatch.setattr(volume.time, "sleep", lambda s: None)
    r = link.push_file(f, "x/c.bin")
    assert calls["n"] == 2 and Path(link.abs("x/c.bin")).read_bytes() == f.read_bytes() and not r["skipped"]


def test_state_hashes_only_files_of_equal_size(tmp_path, link):
    for name, data in (("p/one.bin", blob(1000)), ("p/two.bin", blob(2000))):
        Path(link.abs(name)).parent.mkdir(parents=True, exist_ok=True)
        Path(link.abs(name)).write_bytes(data)
    st = link.state(["p/one.bin", "p/two.bin", "p/none.bin"], {"p/one.bin": 1000, "p/two.bin": 5})
    assert st["p/one.bin"] == (1000, sha(blob(1000)))
    assert st["p/two.bin"] == (2000, None)               # size differs from the local file: no point reading it
    assert "p/none.bin" not in st


def test_push_files_batches_small_files_and_skips_on_rerun(tmp_path, link, monkeypatch):
    monkeypatch.setattr(volume, "STRIPE_MIN", 10 ** 9)
    items = []
    for i in range(5):
        p = tmp_path / f"s{i}.bin"
        p.write_bytes(blob(500 + i))
        items.append((p, f"in/s{i}.bin"))
    big = tmp_path / "big.bin"
    big.write_bytes(blob(volume.SMALL + 10))
    items.append((big, "in/big.bin"))
    r = link.push_files(items)
    assert r["files"] == 6 and r["sent_files"] == 6 and r["skipped"] == 0
    for p, rel in items:
        assert Path(link.abs(rel)).read_bytes() == p.read_bytes()
    r2 = link.push_files(items)
    assert r2["skipped"] == 6 and r2["sent_files"] == 0 and r2["sent_bytes"] == 0


def test_pull_file_resumes_and_verifies(tmp_path, link, monkeypatch):
    data = blob(3_000_000)
    src = Path(link.abs("chunks/c1/out/result.tar"))
    src.parent.mkdir(parents=True)
    src.write_bytes(data)
    dest = tmp_path / "got" / "r.tar"
    dest.parent.mkdir()
    (dest.parent / "r.tar.part").write_bytes(data[:1_000_000])      # an interrupted earlier pull
    r = link.pull_file("chunks/c1/out/result.tar", dest, size=len(data), sha256=sha(data))
    assert r["bytes"] == len(data) and dest.read_bytes() == data and not (dest.parent / "r.tar.part").exists()
    assert link.pull_file("chunks/c1/out/result.tar", dest, size=len(data), sha256=sha(data))["skipped"] is True
    monkeypatch.setattr(volume.time, "sleep", lambda s: None)
    bad = tmp_path / "got" / "bad.tar"
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        link.pull_file("chunks/c1/out/result.tar", bad, size=len(data), sha256="0" * 64, retries=1)
    assert not bad.exists()


def test_submit_worker_pull_round_trip(tmp_path, link):
    """submit_job -> a Worker on the same 'volume' -> done.json + out/ -> pull_result into a local workspace, verified."""
    inp = tmp_path / "in.txt"
    inp.write_text("hello farm")
    py = sys.executable.replace("\\", "/")
    spec = {"schema": worker.SCHEMA, "id": "rt1", "dance": "t", "kind": "analyse", "world_id": None, "max_minutes": 5,
            "steps": [{"name": "copy", "cmd": f'"{py}" -c "import pathlib; pathlib.Path(\'out\').mkdir(exist_ok=True); '
                                              f'pathlib.Path(\'out/r.txt\').write_text(pathlib.Path(\'../in.txt\').read_text().upper())"',
                       "timeout_min": 2}],
            "outputs": [{"path": "dancecap/out/r.txt"}]}
    q = volume.submit_job(link, spec, {"in.txt": inp})
    assert q["inputs"] == 1 and Path(link.abs("queue/rt1.json")).exists()
    assert Path(link.abs("chunks/rt1/in/in.txt")).read_text() == "hello farm"
    w = worker.Worker(Path(link.root), pod_id="podX", ws=tmp_path / "ws", work=tmp_path / "work", state=tmp_path / "state",
                      poll_s=0.05, gpu_required=False)
    assert w.serve(max_jobs=1, idle_exit_min=0.01) == 0
    st = volume.job_status(link, "rt1")
    assert st["done"]["status"] == "ok" and st["manifest"]["files"][0]["path"] == "dancecap/out/r.txt"
    dest = tmp_path / "local_ws"
    r = volume.pull_result(link, "rt1", dest)
    assert r["files"] == 1 and (dest / "dancecap" / "out" / "r.txt").read_text() == "HELLO FARM"


def test_code_bundle_key_follows_content(tmp_path, monkeypatch):
    a, b = tmp_path / "a.py", tmp_path / "b.sh"
    a.write_text("x = 1\n")
    b.write_bytes(b"echo hi\r\n")
    monkeypatch.setattr(volume, "code_files", lambda: [(a, "dancecap/dancecap/a.py"), (b, "dancecap/remote/b.sh")])
    monkeypatch.setattr(volume, "_git", lambda args, cwd: "")
    monkeypatch.setattr(volume, "SHA_CACHE", tmp_path / "cache.json")
    k1 = volume.code_info()["key"]
    assert volume.code_info()["key"] == k1
    out, info = volume.build_code_bundle(tmp_path / "bundles")
    import tarfile
    with tarfile.open(out) as tar:
        assert tar.extractfile("dancecap/remote/b.sh").read() == b"echo hi\n"       # LF endings, executable
        assert tar.getmember("dancecap/remote/b.sh").mode == 0o755
    a.write_text("x = 2\n")
    assert volume.code_info()["key"] != k1


# ------------------------------------------------------------------------------------------------ pods


@pytest.fixture
def fake_runpod(tmp_path, monkeypatch):
    vol = tmp_path / "farm_volume.json"
    vol.write_text(json.dumps({"id": "vol123", "dc": "EU-RO-1", "mount": "/workspace", "size_gb": 150}))
    monkeypatch.setattr(pods, "VOLUME_JSON", vol)
    calls = {"ctl": [], "rest": [], "recorded": []}
    monkeypatch.setattr(remote, "ensure_ssh_key", lambda: None)
    monkeypatch.setattr(remote, "_ctl", lambda *a, **k: (calls["ctl"].append(a), {"id": "pod1", "costPerHr": 0.59})[1])
    monkeypatch.setattr(remote, "_record", lambda *a: calls["recorded"].append(a))
    monkeypatch.setattr(remote, "_wait_running", lambda pid, t: {"id": pid, "ssh": {"direct": {"host": "1.2.3.4", "port": 2222}}})
    monkeypatch.setattr(remote, "_pod_ssh", lambda p: ("1.2.3.4", 2222))
    monkeypatch.setattr(pods, "_rest", lambda m, p, b=None, t=60: (calls["rest"].append((m, p, b)), {"id": "cpu1", "costPerHr": 0.07})[1])
    return calls


def test_create_gpu_pod_mounts_the_volume_and_arms_the_deadlines(fake_runpod):
    info = pods.create("hm-farm-t", gpu="NVIDIA L4", minutes=30, hard_minutes=40, kinds="fit", idle_min=2, price=0.59)
    args = list(fake_runpod["ctl"][0])
    assert args[:2] == ["pod", "create"]
    for flag, val in (("--network-volume-id", "vol123"), ("--data-center-ids", "EU-RO-1"), ("--volume-mount-path", "/workspace"),
                      ("--cloud-type", "SECURE"), ("--gpu-id", "NVIDIA L4"), ("--docker-args", pods.ENTRY_CMD)):
        assert args[args.index(flag) + 1] == val
    env = json.loads(args[args.index("--env") + 1])
    assert env["HM_KINDS"] == "fit" and env["HM_IDLE_MIN"] == "2" and env["HM_PRICE_HR"] == "0.59" and env["HM_AUTOSTART"] == "1"
    assert int(env["HM_HARD_CEIL"]) - int(env["HM_KILL_AT"]) == 600                  # 40 min hard vs 30 min deadline
    assert info["host"] == "1.2.3.4" and info["port"] == 2222
    assert fake_runpod["recorded"][0][0] == "pod1" and fake_runpod["recorded"][0][-1] == 40 + pods.START_TIMEOUT_MIN


def test_create_cpu_pod_uses_rest_with_the_volume_and_no_gpu(fake_runpod):
    info = pods.create("hm-farm-gw", cpu="cpu5c", vcpu=2, minutes=20, entry=True, autostart=False, price=0.07)
    method, path, body = fake_runpod["rest"][0]
    assert (method, path) == ("POST", "/pods")
    assert body["computeType"] == "CPU" and body["cpuFlavorIds"] == ["cpu5c"] and body["vcpuCount"] == 2
    assert body["networkVolumeId"] == "vol123" and body["volumeMountPath"] == "/workspace" and body["dataCenterIds"] == ["EU-RO-1"]
    assert body["dockerStartCmd"] == ["bash", "/workspace/hm/bin/pod_entry.sh"] and body["env"]["HM_AUTOSTART"] == "0"
    assert "gpuTypeIds" not in body and info["pod_id"] == "cpu1"


def test_create_cpu_retries_5xx_and_adopts_a_pod_created_despite_the_error(monkeypatch):
    monkeypatch.setattr(pods.time, "sleep", lambda s: None)
    seq = iter([pods.PodError("POST /pods: HTTP 500: boom"), pods.PodError("POST /pods: HTTP 500: boom")])

    def rest(m, p, b=None, t=60):
        raise next(seq)
    monkeypatch.setattr(pods, "_rest", rest)
    listings = iter([[], [{"id": "ghost1", "name": "hm-farm-x", "cost_per_hr": 0.07}]])
    monkeypatch.setattr(pods, "list_farm_pods", lambda include_all=False: next(listings))
    got = pods._create_cpu("hm-farm-x", {"name": "hm-farm-x"})
    assert got["id"] == "ghost1"                         # adopted, not created a second time
    monkeypatch.setattr(pods, "_rest", lambda *a, **k: (_ for _ in ()).throw(pods.PodError("POST /pods: HTTP 400: bad")))
    with pytest.raises(pods.PodError, match="HTTP 400"):
        pods._create_cpu("hm-farm-y", {})


def test_gateway_reuses_a_running_farm_pod_else_creates_a_cpu_pod(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(pods, "list_farm_pods", lambda include_all=False: [{"id": "p1", "status": "RUNNING"}])
    monkeypatch.setattr(pods, "attach", lambda pid: sentinel)
    assert pods.gateway() == (sentinel, False)
    monkeypatch.setattr(pods, "list_farm_pods", lambda include_all=False: [])
    seen = {}
    monkeypatch.setattr(pods, "create", lambda name, **kw: (seen.update(kw), {"pod_id": "g", "host": "h", "port": 1, "cost_per_hr": 0.07})[1])
    pod, created = pods.gateway(minutes=10)
    assert created and seen["cpu"] == "cpu5c" and seen["autostart"] is False and seen["entry"] is True
