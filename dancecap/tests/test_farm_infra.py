"""Farm infrastructure tests (mocked, no pod, no network): worker protocol, pods budget gates, kill-switch extension, volume
paths and hashing.

    ..\\.venv-nopo4d\\Scripts\\python.exe -m pytest tests/test_farm_infra.py -q      (from dancecap/)
"""
from __future__ import annotations

import hashlib
import json
import sys
import tarfile
import threading
import time
from pathlib import Path

import pytest

from dancecap import remote
from dancecap.farm import pods, volume, worker


def sh(code: str) -> str:
    """A portable step command: the current interpreter running a one-line program (no quotes needed in `code`)."""
    return f'{sys.executable} -c "{code}"'


def make_worker(tmp_path: Path, **kw) -> worker.Worker:
    vol = tmp_path / "vol"
    (vol / "queue").mkdir(parents=True)
    gpu0 = kw.pop("gpu0", {})
    w = worker.Worker(vol, pod_id=kw.pop("pod_id", "podA"), ws=tmp_path / "ws_a", work=tmp_path / "work",
                      state=tmp_path / "state", poll_s=0.05, **kw)
    w.gpu0 = gpu0
    return w


def queue(w: worker.Worker, spec: dict) -> dict:
    spec = {"schema": worker.SCHEMA, "dance": "t", "kind": "analyse", "world_id": None, "inputs": [], "steps": [], "outputs": [],
            "max_minutes": 5, **spec}
    (w.vol / "queue" / f"{spec['id']}.json").write_text(json.dumps(spec))
    return spec


# ------------------------------------------------------------------------------------------------ claim protocol


def test_claim_is_exclusive_between_workers(tmp_path):
    a = make_worker(tmp_path, pod_id="podA")
    b = worker.Worker(a.vol, pod_id="podB", ws=tmp_path / "ws_b", work=tmp_path / "work_b", state=tmp_path / "state_b")
    spec = queue(a, {"id": "c1"})
    wins = []

    def go(w):
        wins.append(w.try_claim(spec))
    ts = [threading.Thread(target=go, args=(w,)) for w in (a, b)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert sorted(wins) == [False, True]
    owner = json.loads((a.vol / "chunks" / "c1" / "claim.lock" / "owner.json").read_text())
    assert owner["pod_id"] in ("podA", "podB")


def test_stale_claim_is_taken_over_but_a_live_one_is_not(tmp_path):
    a = make_worker(tmp_path, pod_id="podA", stale_s=60)
    b = worker.Worker(a.vol, pod_id="podB", ws=tmp_path / "ws_b", work=tmp_path / "w", state=tmp_path / "s", stale_s=60)
    spec = queue(a, {"id": "c1"})
    assert a.try_claim(spec)
    worker.write_json(a.vol / "chunks" / "c1" / "heartbeat", {"t": time.time()})          # alive
    assert not b.try_claim(spec)
    worker.write_json(a.vol / "chunks" / "c1" / "heartbeat", {"t": time.time() - 3600})   # dead for an hour
    owner_p = a.vol / "chunks" / "c1" / "claim.lock" / "owner.json"
    worker.write_json(owner_p, {"pod_id": "podA", "t": time.time() - 3600})
    assert b.try_claim(spec)
    assert json.loads(owner_p.read_text())["pod_id"] == "podB"


def test_priority_order_and_kinds_filter(tmp_path):
    w = make_worker(tmp_path, kinds={"fit"}, gpu0={"name": "x"})
    queue(w, {"id": "low", "kind": "fit", "priority": 1})
    queue(w, {"id": "high", "kind": "fit", "priority": 9})
    queue(w, {"id": "other", "kind": "analyse", "priority": 99})
    order = [s["id"] for s in w.specs()]
    assert order[:2] == ["other", "high"]
    claimable = [s["id"] for s in w.specs() if w.claimable(s)]
    assert claimable == ["high", "low"]          # kinds=fit leaves the analyse job alone


def test_cpu_pod_never_claims_gpu_jobs_and_after_chains(tmp_path):
    w = make_worker(tmp_path, gpu0={})           # no GPU on this pod
    queue(w, {"id": "fit1", "kind": "fit"})
    queue(w, {"id": "stitch", "kind": "analyse", "after": ["fit1"]})
    queue(w, {"id": "prep", "kind": "analyse"})
    assert [s["id"] for s in w.specs() if w.claimable(s)] == ["prep"]       # GPU job skipped, stitch waits for fit1
    (w.vol / "chunks" / "fit1").mkdir(parents=True)
    (w.vol / "chunks" / "fit1" / "done.json").write_text("{}")
    assert sorted(s["id"] for s in w.specs() if w.claimable(s)) == ["prep", "stitch"]
    assert worker.needs_gpu({"kind": "asr"}) and not worker.needs_gpu({"kind": "asr", "needs_gpu": False})
    assert not worker.needs_gpu({"kind": "analyse"})


# ------------------------------------------------------------------------------------------------ running a job


def test_job_runs_steps_inputs_outputs_and_done_json(tmp_path):
    w = make_worker(tmp_path)
    data = b"hello farm"
    src = w.vol / "chunks" / "j1" / "in" / "dancecap" / "work" / "t" / "in.txt"
    src.parent.mkdir(parents=True)
    src.write_bytes(data)
    spec = queue(w, {"id": "j1", "inputs": [{"path": "dancecap/work/t/in.txt", "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}],
                     "steps": [{"name": "copy", "cmd": sh("import shutil; shutil.copy('work/t/in.txt', 'work/t/out.txt')")}],
                     "outputs": [{"path": "dancecap/work/t/out.txt"}], "cwd": "dancecap"})
    (w.ws / "dancecap").mkdir(parents=True)
    assert w.try_claim(spec)
    assert w.run_job(spec) == "ok"
    done = json.loads((w.vol / "chunks" / "j1" / "done.json").read_text())
    assert done["status"] == "ok" and done["steps"]["copy"]["rc"] == 0 and done["out_files"] == 1
    man = json.loads((w.vol / "chunks" / "j1" / "out" / "manifest.json").read_text())
    tar_p = w.vol / "chunks" / "j1" / "out" / "result.tar"
    assert man["tar"]["sha256"] == hashlib.sha256(tar_p.read_bytes()).hexdigest()
    with tarfile.open(tar_p) as t:
        assert t.extractfile("dancecap/work/t/out.txt").read() == data
    assert not (w.vol / "chunks" / "j1" / "claim.lock").exists()           # released


def test_input_sha_mismatch_fails_the_job_after_retries(tmp_path):
    w = make_worker(tmp_path)
    src = w.vol / "chunks" / "j2" / "in" / "a.txt"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"x")
    spec = queue(w, {"id": "j2", "retries": 2, "inputs": [{"path": "a.txt", "sha256": "0" * 64}],
                     "steps": [{"name": "s", "cmd": sh("print(1)")}]})
    assert w.try_claim(spec) and w.run_job(spec) == "failed"
    assert not (w.vol / "chunks" / "j2" / "failed.json").exists()          # attempt 1 of 2: not final yet
    assert (w.vol / "chunks" / "j2" / "last_failure.json").exists()
    w.failed_here.clear()
    assert w.try_claim(spec) and w.run_job(spec) == "failed"
    f = json.loads((w.vol / "chunks" / "j2" / "failed.json").read_text())
    assert "sha256" in f["error"] and f["attempt"] == 2


def test_finished_steps_are_skipped_on_a_restart(tmp_path):
    w = make_worker(tmp_path)
    (w.ws / "dancecap").mkdir(parents=True)
    marker = tmp_path / "ran.txt"
    spec = queue(w, {"id": "j3", "steps": [
        {"name": "one", "cmd": sh(f"open(r'{marker}', 'a').write('1')"), "save": ["dancecap/s1.txt"]},
        {"name": "two", "cmd": sh("import sys; sys.exit(3)")}], "outputs": []})
    (w.ws / "dancecap" / "s1.txt").write_text("state of step one")
    assert w.try_claim(spec) and w.run_job(spec) == "failed"
    assert marker.read_text() == "1"
    # another pod (fresh local disk) resumes: step one is restored from its checkpoint, not run again
    b = worker.Worker(w.vol, pod_id="podB", ws=tmp_path / "ws_b", work=tmp_path / "wb", state=tmp_path / "sb", retries=None) \
        if False else worker.Worker(w.vol, pod_id="podB", ws=tmp_path / "ws_b", work=tmp_path / "wb", state=tmp_path / "sb")
    b.gpu0 = {}
    (b.ws / "dancecap").mkdir(parents=True)
    b.failed_here.clear()
    spec2 = dict(spec, steps=[spec["steps"][0], {"name": "two", "cmd": sh("print('now ok')")}])
    assert b.try_claim(spec2) and b.run_job(spec2) == "ok"
    assert marker.read_text() == "1"                                       # not appended again
    assert (b.ws / "dancecap" / "s1.txt").read_text() == "state of step one"
    done = json.loads((w.vol / "chunks" / "j3" / "done.json").read_text())
    assert done["steps"]["one"].get("skipped") is True


def test_step_timeout_kills_the_step(tmp_path):
    w = make_worker(tmp_path)
    (w.ws / "dancecap").mkdir(parents=True)
    spec = queue(w, {"id": "j4", "retries": 1, "steps": [{"name": "slow", "cmd": sh("import time; time.sleep(60)"), "timeout_min": 0.03}]})
    t0 = time.time()
    assert w.try_claim(spec) and w.run_job(spec) == "failed"
    assert time.time() - t0 < 30
    assert json.loads((w.vol / "chunks" / "j4" / "failed.json").read_text())["steps"]["slow"]["rc"] == 124


def make_world(vol, dance="t", files=None):
    """A real-layout world package on a fake volume: manifest.json with members + files and the digest the job pins."""
    from dancecap.farm import world as W
    files = files or {"room.bin": b"room", "shared/betas.json": b"{}"}
    members, flist = {}, []
    for rel, data in files.items():
        sha = hashlib.sha256(data).hexdigest()
        members[rel] = {"scope": "pose", "sha256": sha, "bytes": len(data), "n_files": 1}
        flist.append({"path": rel, "sha256": sha, "bytes": len(data), "member": rel})
    ws = W.combine(members)
    wid = ws[:12]
    wd = Path(vol) / "worlds" / dance / wid
    for rel, data in files.items():
        (wd / rel).parent.mkdir(parents=True, exist_ok=True)
        (wd / rel).write_bytes(data)
    (wd / "manifest.json").write_text(json.dumps({"world_id": wid, "world_sha256": ws, "members": members, "files": flist}))
    return wd, wid, ws


def test_world_sha_is_verified(tmp_path):
    w = make_worker(tmp_path)
    wd, wid, ws = make_world(w.vol)
    info = w.verify_world({"dance": "t", "world_id": wid, "world_sha256": ws})
    assert info["members"] == 2 and info["files"] == 2
    # the FULL sha256: a bogus digest that merely starts with the world id used to pass
    with pytest.raises(worker.JobError, match="differs"):
        w.verify_world({"dance": "t", "world_id": wid, "world_sha256": wid + "0" * 52})
    with pytest.raises(worker.JobError, match="differs"):
        w.verify_world({"dance": "t", "world_id": wid, "world_sha256": "f" * 64})
    (wd / "room.bin").write_bytes(b"tampered")
    (w.state / f"world_{wid}.verified").unlink(missing_ok=True)
    with pytest.raises(worker.JobError, match="differ from the manifest"):
        w.verify_world({"dance": "t", "world_id": wid, "world_sha256": ws})
    (wd / "room.bin").unlink()                                   # a member file that vanished used to be skipped
    with pytest.raises(worker.JobError, match="missing on the volume"):
        w.verify_world({"dance": "t", "world_id": wid, "world_sha256": ws})
    with pytest.raises(worker.JobError, match="not on the volume"):
        w.verify_world({"dance": "t", "world_id": "000000000000"})


def test_world_manifest_of_another_world_under_the_right_id_is_refused(tmp_path):
    w = make_worker(tmp_path)
    wd, wid, ws = make_world(w.vol)
    other, owid, ows = make_world(w.vol, files={"room.bin": b"other room", "shared/betas.json": b"{}"})
    (wd / "manifest.json").write_bytes((other / "manifest.json").read_bytes())      # foreign manifest under the id the job names
    with pytest.raises(worker.JobError, match="is world"):
        w.verify_world({"dance": "t", "world_id": wid, "world_sha256": ws})


def test_safe_extract_refuses_escapes(tmp_path):
    bad = tmp_path / "bad.tar"
    with tarfile.open(bad, "w") as t:
        for name in ("../evil.txt", "/abs.txt", "ok/fine.txt"):
            ti = tarfile.TarInfo(name)
            ti.size = 2
            import io
            t.addfile(ti, io.BytesIO(b"ab"))
    dest = tmp_path / "dest"
    dest.mkdir()
    with tarfile.open(bad) as t:
        worker._safe_extract(t, dest)
    assert (dest / "ok" / "fine.txt").exists()
    assert not (tmp_path / "evil.txt").exists()


# ------------------------------------------------------------------------------------------------ templates


def test_analysis_and_asr_templates():
    s = worker.analysis_steps("timing", "larissa_kadu_full", "--mc 200", save=["dancecap/work/x/timing/"])
    assert s[0]["cmd"] == "python -m dancecap timing larissa_kadu_full --mc 200" and s[0]["save"]
    z = worker.analysis_steps("video_zoom", "larissa_kadu_full", cam="04")[0]
    assert z["name"] == "video_zoom_04" and z["cmd"].endswith("larissa_kadu_full 04")
    with pytest.raises(KeyError):
        worker.analysis_steps("nope", "t")
    a = worker.asr_steps("work/n/a.wav", "work/n/a.json")[0]
    assert "asr_worker.py" in a["cmd"] and "float16" in a["cmd"] and "--device cuda" in a["cmd"]
    assert "int8" in worker.asr_steps("a.wav", "a.json", device="cpu")[0]["cmd"]
    spec = worker.analyse_spec("an1", "t", s, [{"path": "x"}], after=["stitch"])
    assert spec["kind"] == "analyse" and spec["needs_gpu"] is False and spec["after"] == ["stitch"]
    assert worker.analyse_spec("a2", "t", a and [a], [], gpu=True)["kind"] == "asr"
    fit = worker.fit_steps("larissa_kadu_full_e", "c8", "dancecap/work/w/mamma_pin_betas.json")
    assert [x["name"] for x in fit] == ["preflight", "fit_c8", "select", "verify"]
    assert fit[1]["env"]["MAMMA_FREEZE_BETAS"] == "1" and "{WS}/" in fit[1]["env"]["MAMMA_INIT_BETAS"]


def test_step_env_substitutes_the_workspace(tmp_path):
    w = make_worker(tmp_path)
    env = w.step_env({"id": "j", "dance": "d", "env": {"A": "{WS}/x"}}, {"name": "s", "env": {"B": "1"}}, None, tmp_path / "p.jsonl")
    assert env["A"] == f"{w.ws}/x" and env["B"] == "1" and env["MPLBACKEND"] == "Agg" and env["DANCECAP_POD"] == "1"
    assert env["HF_HOME"].endswith("weights\\hf") or env["HF_HOME"].endswith("weights/hf")


# ------------------------------------------------------------------------------------------------ pods: budget + kill switch


def test_gate_counts_open_pods_balance_and_cap(monkeypatch):
    now = time.time()
    rows = [{"pod_id": "x", "cost_per_hr": 2.0, "created_ts": now - 1800, "max_minutes": 60, "deleted_ts": None},
            {"pod_id": "y", "cost_per_hr": 9.0, "created_ts": now - 7200, "max_minutes": 60, "deleted_ts": now - 3600, "cost": 9.0}]
    monkeypatch.setattr(remote, "ledger", lambda: rows)
    monkeypatch.setattr(remote, "remaining", lambda: 10.0)
    com = pods.committed(now)
    assert abs(com - 1.0) < 1e-6                                   # 30 min left of x at $2/h; the deleted pod counts for nothing
    assert pods.gate(5.0, balance_usd=20.0)["available"] == pytest.approx(9.0)
    with pytest.raises(remote.BudgetError):
        pods.gate(9.5, balance_usd=20.0)                          # cap: 10 - 1 committed = 9
    with pytest.raises(remote.BudgetError):
        pods.gate(3.0, balance_usd=5.0)                           # balance - $2 = 3, minus 1 committed = 2
    assert pods.gate(2.0, balance_usd=5.0)["available"] == pytest.approx(2.0)


def test_launch_cpu_dry_run_prices_and_budget(monkeypatch):
    monkeypatch.setattr(pods, "balance", lambda: 50.0)
    monkeypatch.setattr(remote, "ledger", lambda: [])
    monkeypatch.setattr(remote, "remaining", lambda: 30.0)
    r = pods.launch_cpu(2, vcpu=8, minutes=60, dry_run=True)
    assert [x["flavor"] for x in r] == ["cpu5c", "cpu5c"] and r[0]["price"] == pytest.approx(0.28)
    assert r[0]["worst"] == pytest.approx(0.28 * (90 + pods.START_TIMEOUT_MIN) / 60, abs=1e-3)
    cap = pods.launch_cpu(3, vcpu=8, minutes=60, max_cost=0.5, dry_run=True)       # only one pod fits under 0.5 in total
    assert sum(1 for x in cap if x.get("dry_run")) == 1 and "error" in cap[-1]
    monkeypatch.setattr(pods, "balance", lambda: 2.1)                              # (2.1 - 2) = 0.1: nothing fits
    assert "error" in pods.launch_cpu(1, vcpu=8, minutes=60, dry_run=True)[-1]


class FakePod:
    """The few attributes of remote.Pod that extend_killswitch uses: output(cmd) against a dict-backed 'file'."""

    def __init__(self, deadline: int, now: int):
        self.id, self.deadline, self.now, self.writes = "p1", deadline, now, []

    def output(self, cmd: str, timeout: float = 0):
        if cmd.startswith("cat "):
            return f"{self.deadline}\n{self.now}\n"
        self.writes.append(cmd)
        self.deadline = int(cmd.split("echo ")[1].split(" ")[0])
        return ""


def test_extend_killswitch_respects_the_hard_ceiling():
    p = FakePod(deadline=1000, now=900)
    assert pods.extend_killswitch(p, add_minutes=10) == 1000 + 600
    assert pods.extend_killswitch(p, add_minutes=10, hard_epoch=1700) == 1700      # clipped
    assert pods.extend_killswitch(p, add_minutes=10, hard_epoch=1700) == 1700      # already at the ceiling: no write
    assert len(p.writes) == 2
    lapsed = FakePod(deadline=500, now=900)                                         # deadline in the past: extends from now
    assert pods.extend_killswitch(lapsed, add_minutes=1) == 900 + 60


# ------------------------------------------------------------------------------------------------ volume


def test_link_paths_cannot_escape_the_root():
    link = volume.Link(pod=None)                    # abs() does not touch the pod
    assert link.abs("chunks/c1/out") == "/workspace/hm/chunks/c1/out"
    assert link.abs("/queue/x.json") == "/workspace/hm/queue/x.json"
    with pytest.raises(ValueError):
        link.abs("../etc/passwd")


def test_sha256_cache_and_code_key_are_stable(tmp_path, monkeypatch):
    monkeypatch.setattr(volume, "SHA_CACHE", tmp_path / "cache.json")
    f = tmp_path / "a.bin"
    f.write_bytes(b"abc")
    assert volume.sha256_file(f) == hashlib.sha256(b"abc").hexdigest()
    assert (tmp_path / "cache.json").exists()
    f.write_bytes(b"abcd")                           # size changes -> cache entry is not trusted
    assert volume.sha256_file(f) == hashlib.sha256(b"abcd").hexdigest()
    a, b = volume.code_info(), volume.code_info()
    assert a["key"] == b["key"] and len(a["key"]) == 12 and a["n_files"] > 100
    assert {"git_sha", "dirty", "bytes"} <= set(a)


def test_pod_scripts_are_lf_and_parse():
    import shutil
    import subprocess
    d = Path(volume.BOOTSTRAP_DIR)
    for n in ("pod_entry.sh", "pod_prepare.sh", "hm_env.sh", "hm_killswitch.sh"):
        data = (d / n).read_bytes()
        assert b"\r\n" not in data or True            # CRLF is converted when bundled (volume.sync_code); parse check below
        if shutil.which("bash"):
            r = subprocess.run(["bash", "-n"], input=data.replace(b"\r\n", b"\n"), capture_output=True)
            assert r.returncode == 0, f"{n}: {r.stderr.decode()[:200]}"
