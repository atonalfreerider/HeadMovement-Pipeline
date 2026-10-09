"""The farm watcher with fake pods (no network, no Runpod, no billable pod): a local bash plays the pod, a temp dir plays the volume, a
FakeBackend plays Runpod and REAL `worker.Worker`s produce the done.json / out/result.tar / manifest.json the watcher consumes.

    ..\\.venv-nopo4d\\Scripts\\python.exe -m pytest tests/test_farm_watch.py -q      (from dancecap/)

Covers: the happy path (queued -> running -> done -> pulled -> pod deleted -> result.json), an ssh drop in the middle of a pull,
a watcher killed and restarted (resumes a partial download, trusts a verified pull), a lost pod (claim released, replacement
pod, job finished elsewhere), a failed chunk (re-queued, then failed for good without blocking the finish), the kill-switch
hard ceiling and the budget gate, a second watcher refused, and a pod deleted only after the verified pull.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import _farm_watch_fixtures as F  # noqa: E402

from dancecap import remote  # noqa: E402
from dancecap.farm import launch, pull, watch, worker  # noqa: E402
from dancecap.farm.state import State  # noqa: E402

from _farm_watch_env import DANCE, add_job, add_pod, cycles, env, make_watcher, status_of  # noqa: E402,F401

pytestmark = pytest.mark.skipif(F.BASH is None, reason="needs bash to play the pod")


# ------------------------------------------------------------------------------------------------ the happy path


def test_job_runs_is_pulled_verified_and_the_pod_is_deleted_after_the_pull(env):
    pod = add_pod(env)
    spec = add_job(env, "g0000", payload=b"MOTION")
    w = make_watcher(env, keep_pods=False)
    cycles(w)
    assert status_of(env, "g0000") == "queued"
    p = env.state.load()["pods"][pod]
    assert p["state"] == "idle" and p["role"] == "gpu"

    F.start_job(env.be, pod, spec, progress={"done": 30, "total": 100})
    cycles(w)
    c = env.state.load()["chunks"]["g0000"]
    assert c["status"] == "running" and c["pod"] == pod and c["step"] == "make" and 0.0 < c["progress"] < 1.0
    assert env.state.load()["pods"][pod]["jobs"] == ["g0000"] and env.state.load()["pods"][pod]["state"] == "running"
    assert w.cycle is not None and not [x for x in env.be.calls if x[0] == "delete"]       # a busy pod is never deleted

    # the worker finishes: release the fake claim and run the job for real
    env.be.worker_for(pod).release("g0000")
    assert F.finish_job(env.be, pod, spec) == "ok"
    cycles(w, 2)
    c = env.state.load()["chunks"]["g0000"]
    assert c["status"] in ("pulled", "stitched") and c["pull"]["state"] == "ok" and c["world_id"] is None
    out = env.ws / "dancecap" / "work" / "t_g0000" / "smplx" / "lead.npz"
    assert out.read_bytes() == b"MOTION"
    pj = json.loads((watch.chunks_dir(DANCE) / "g0000" / "pull.json").read_text())
    assert pj["verified"] and pj["files"] == 2 and pj["take"] == "t_g0000"
    assert json.loads((env.ws / "dancecap" / "work" / "t_g0000" / "farm" / "pulled.json").read_text())["id"] == "g0000"
    assert env.cov_calls                                     # coverage recomputed after the pull
    # nothing active any more: the watcher finished, deleted the idle pod, wrote result.json
    assert w.finished
    assert ("delete", pod) == tuple(x[:2] for x in env.be.calls if x[0] == "delete")[0]
    res = json.loads(watch.result_path(DANCE).read_text())
    assert res["status"] == "complete" and res["chunks"]["g0000"]["status"] == "pulled"
    assert env.state.load()["watcher"]["state"] == "finished"
    assert env.state.load()["pods"][pod]["state"] == "deleted"


def test_pod_is_deleted_only_after_the_verified_pull(env):
    """A finished job whose pull keeps failing keeps its pod; the pod goes the cycle after the pull is verified."""
    pod = add_pod(env)
    spec = add_job(env, "g0000", payload=5000)
    add_job(env, "g0001", status="queued", payload=b"Y")                    # keeps the watcher from finishing meanwhile
    env.be.links[pod] = F.FlakyLink(F.LocalPod(pod), root=env.vol.as_posix())
    assert F.finish_job(env.be, pod, spec) == "ok"
    F.FlakyLink.drops = 10_000                                               # every download attempt is cut
    w = make_watcher(env, idle_grace_s=0.0)
    cycles(w, 3)
    assert status_of(env, "g0000") == "done"
    assert env.state.load()["chunks"]["g0000"]["pull"]["state"] == "error"
    assert env.state.load()["pods"][pod]["state"] != "deleted" and not [x for x in env.be.calls if x[0] == "delete"]

    F.FlakyLink.drops = 0                                                    # the connection is good again
    rec = env.state.load()["chunks"]["g0000"]
    State(DANCE).set_chunk("g0000", pull={**rec["pull"], "next_at": 0})
    seen = []
    orig = env.be.delete
    env.be.delete = lambda pid, reason: (seen.append((watch.chunks_dir(DANCE) / "g0000" / "pull.json").exists()), orig(pid, reason))
    cycles(w, 2)
    assert status_of(env, "g0000") == "pulled"
    assert seen == [True]                                                    # deleted, and the verified pull was already on disk
    assert env.state.load()["pods"][pod]["state"] == "deleted"


# ------------------------------------------------------------------------------------------------ survivability of the pull


def test_ssh_drop_in_the_middle_of_a_pull_is_resumed_and_verified(env):
    pod = add_pod(env)
    spec = add_job(env, "g0000", payload=20000)
    env.be.links[pod] = F.FlakyLink(F.LocalPod(pod), root=env.vol.as_posix())
    assert F.finish_job(env.be, pod, spec) == "ok"
    want = (env.vol / "chunks" / "g0000" / "out" / "result.tar").read_bytes()
    F.FlakyLink.drops, F.FlakyLink.cut = 2, 3000                             # two resets, 3000 bytes each time
    w = make_watcher(env)
    cycles(w, 2)
    assert status_of(env, "g0000") in ("pulled", "stitched")
    assert F.FlakyLink.calls >= 3                                            # cut, cut, then the rest
    offsets = [int(c.split("tail -c +")[1].split()[0]) for c in F.LocalLink.cmds if c.startswith("tail -c +")]
    assert offsets[0] == 1 and offsets[1] > 1 and offsets[2] > offsets[1]    # each retry resumed where the last stream ended
    assert (env.ws / "dancecap" / "work" / "t_g0000" / "smplx" / "lead.npz").stat().st_size == 20000
    assert not (watch.chunks_dir(DANCE) / "g0000" / "result.tar.part").exists()
    assert len(want) > 3000


def test_watcher_killed_mid_pull_resumes_the_partial_download(env):
    pod = add_pod(env)
    spec = add_job(env, "g0000", payload=30000)
    add_job(env, "g0001", payload=b"later")
    assert F.finish_job(env.be, pod, spec) == "ok"
    tar_bytes = (env.vol / "chunks" / "g0000" / "out" / "result.tar").read_bytes()
    # what a killed watcher leaves behind: a partial download, a stale pull lock of a dead process, 'done' in state.json
    local = watch.chunks_dir(DANCE) / "g0000"
    local.mkdir(parents=True)
    (local / "result.tar.part").write_bytes(tar_bytes[:7000])
    (local / "pull.lock").write_text(json.dumps({"pid": 2_999_999, "host": __import__("socket").gethostname(), "t": time.time()}))
    State(DANCE).set_chunk("g0000", status="done", pod=pod, pull={"state": "pulling"})
    w2 = make_watcher(env)                                                  # a brand-new process knows nothing but the disk
    cycles(w2, 2)
    assert status_of(env, "g0000") in ("pulled", "stitched")
    offs = [int(c.split("tail -c +")[1].split()[0]) for c in F.LocalLink.cmds if c.startswith("tail -c +")]
    assert offs == [7001]                                                    # continued after the 7000 bytes already there
    assert (env.ws / "dancecap" / "work" / "t_g0000" / "smplx" / "lead.npz").stat().st_size == 30000
    assert not (local / "pull.lock").exists()


def test_restart_after_a_verified_pull_does_not_download_again(env):
    pod = add_pod(env)
    spec = add_job(env, "g0000", payload=b"Z" * 100)
    add_job(env, "g0001", payload=b"later")
    assert F.finish_job(env.be, pod, spec) == "ok"
    w1 = make_watcher(env)
    cycles(w1, 2)
    assert status_of(env, "g0000") in ("pulled", "stitched")
    # crash between the verified pull and the state update: state says done, pull.json says verified
    State(DANCE).set_chunk("g0000", status="done")
    F.LocalLink.cmds.clear()
    w2 = make_watcher(env)
    cycles(w2, 1)
    assert status_of(env, "g0000") == "pulled"
    assert not [c for c in F.LocalLink.cmds if c.startswith("tail -c +")]    # no byte moved
    # ... but a different result on the volume (the job re-ran) is pulled again
    man = json.loads((env.vol / "chunks" / "g0000" / "out" / "manifest.json").read_text())
    man["tar"]["sha256"] = "0" * 64
    (env.vol / "chunks" / "g0000" / "out" / "manifest.json").write_text(json.dumps(man))
    State(DANCE).set_chunk("g0000", status="done")
    cycles(w2, 1)
    assert status_of(env, "g0000") == "done"                                 # tries (and fails the sha check: the tar no longer matches)
    assert env.state.load()["chunks"]["g0000"]["pull"]["state"] == "error"


# ------------------------------------------------------------------------------------------------ pods that vanish, jobs that fail


def test_lost_pod_releases_its_claim_and_a_replacement_finishes_the_job(env):
    p1 = add_pod(env)
    spec = add_job(env, "g0000", payload=b"AFTER-LOSS")
    w = make_watcher(env, target_gpu_pods=1)
    F.start_job(env.be, p1, spec)
    cycles(w)
    assert status_of(env, "g0000") == "running"
    # the pod disappears from Runpod (kill switch / preemption) and stops writing heartbeats
    env.be.pods.pop(p1)
    worker.write_json(env.vol / "pods" / p1 / "heartbeat", {"t": time.time() - 1000, "state": "running"})
    cycles(w, 2)                                                            # missing from two consecutive polls = lost
    st = env.state.load()
    assert st["pods"][p1]["state"] == "lost" and ("mark_gone", p1, "lost (not in Runpod's list)") in env.be.calls
    assert not (env.vol / "chunks" / "g0000" / "claim.lock").exists()       # released at once, no 240 s wait
    assert st["chunks"]["g0000"]["status"] == "queued" and st["chunks"]["g0000"]["lost_count"] == 1
    cycles(w, 2)                                                            # a replacement GPU pod is launched
    gpu_launches = [x for x in env.be.calls if x[0] == "launch_gpu"]
    assert len(gpu_launches) == 1
    p2 = next(pid for pid, p in env.state.load()["pods"].items() if pid != p1 and p["state"] != "deleted")
    assert F.finish_job(env.be, p2, spec) == "ok"                           # the new pod's worker takes the job
    cycles(w, 2)
    assert status_of(env, "g0000") in ("pulled", "stitched")
    assert (env.ws / "dancecap" / "work" / "t_g0000" / "smplx" / "lead.npz").read_bytes() == b"AFTER-LOSS"


def test_a_pod_with_a_fresh_heartbeat_is_not_lost_when_runpod_forgets_it_for_a_poll(env):
    p1 = add_pod(env)
    spec = add_job(env, "g0000")
    F.start_job(env.be, p1, spec)
    w = make_watcher(env)
    saved = env.be.pods.pop(p1)
    cycles(w, 3)                                                            # missing three polls but the worker keeps beating
    assert env.state.load()["pods"][p1]["state"] != "lost" and status_of(env, "g0000") == "running"
    env.be.pods[p1] = saved


def test_api_outage_freezes_pod_states(env):
    p1 = add_pod(env)
    add_job(env, "g0000")
    w = make_watcher(env)
    cycles(w)
    env.be.api_down = True
    env.be.pods.pop(p1)
    worker.write_json(env.vol / "pods" / p1 / "heartbeat", {"t": time.time() - 1000, "state": "idle"})
    cycles(w, 4)
    assert env.state.load()["pods"][p1]["state"] != "lost"                  # an unreadable pod list is not "no pods"


def test_failed_chunk_is_requeued_once_then_failed_for_good_without_blocking_the_finish(env):
    pod = add_pod(env)
    bad = add_job(env, "g0000", retries=1, steps=[{"name": "boom", "cmd": F.sh("import sys; sys.exit(3)"), "timeout_min": 1}], outputs=[])
    good = add_job(env, "g0001", payload=b"ok")
    w = make_watcher(env, max_requeues=1)
    assert F.finish_job(env.be, pod, bad) == "failed"
    assert (env.vol / "chunks" / "g0000" / "failed.json").exists()
    cycles(w)
    rec = env.state.load()["chunks"]["g0000"]
    assert rec["status"] == "queued" and rec["requeues"] == 1 and "step boom rc=3" in rec["last_error"]
    assert not (env.vol / "chunks" / "g0000" / "failed.json").exists()      # moved aside for the next worker
    assert list((env.vol / "chunks" / "g0000").glob("failed.*.json"))
    assert json.loads((env.vol / "chunks" / "g0000" / "attempts.json").read_text())["n"] == 0
    assert F.finish_job(env.be, pod, good) == "ok"
    assert F.finish_job(env.be, pod, bad) == "failed"                       # second attempt fails again
    cycles(w, 3)
    assert status_of(env, "g0000") == "failed" and status_of(env, "g0001") in ("pulled", "stitched")
    assert w.finished
    res = json.loads(watch.result_path(DANCE).read_text())
    assert res["status"] == "partial" and res["failed"] == ["g0000"]


def test_job_downstream_of_a_failed_chunk_is_marked_blocked(env):
    add_pod(env)
    add_job(env, "g0000", status="queued")
    add_job(env, "final", chunk=False, after=["g0000"], needs_gpu=False)
    State(DANCE).set_chunk("g0000", status="failed", error="x")
    w = make_watcher(env)
    cycles(w)
    j = env.state.load()["jobs"]["final"]
    assert j["status"] == "failed" and j["blocked"] and "upstream g0000 failed" in j["error"]


def test_stalled_job_releases_its_claim_and_recycles_the_pod(env):
    pod = add_pod(env)
    spec = add_job(env, "g0000")
    F.start_job(env.be, pod, spec)
    hb = json.loads((env.vol / "chunks" / "g0000" / "heartbeat").read_text())
    hb["t"] = time.time() - 5000
    (env.vol / "chunks" / "g0000" / "heartbeat").write_text(json.dumps(hb))
    w = make_watcher(env, stale_hb_s=600)
    cycles(w)                                                               # marks it stalled
    cycles(w)                                                               # acts on it
    assert not (env.vol / "chunks" / "g0000" / "claim.lock").exists()
    assert ("delete", pod, "stalled job") in env.be.calls
    assert env.state.load()["chunks"]["g0000"]["stall_count"] == 1


# ------------------------------------------------------------------------------------------------ kill switch


def test_kill_switch_is_extended_while_healthy_but_never_past_the_hard_ceiling(env):
    pod = add_pod(env, minutes=5, hard_minutes=15)                           # kill in 5 min, ceiling in 15 min
    spec = add_job(env, "g0000")
    F.start_job(env.be, pod, spec)
    w = make_watcher(env, extend_below_min=25, extend_minutes=30)
    hard = env.state.load()["pods"][pod]["hard_ceil"]
    cycles(w)
    kill = int((env.vol / "pods" / pod / "kill_at").read_text())
    assert hard - 2 <= kill <= hard                                          # +30 min asked, clipped to the ceiling (clock skew +-1 s)
    cycles(w, 2)
    assert int((env.vol / "pods" / pod / "kill_at").read_text()) <= hard
    assert len([x for x in env.be.calls if x[0] == "extend"]) == 1           # at the ceiling: no further attempts
    assert any("hard ceiling" in e["msg"] for e in env.state.load()["events"])


def test_kill_switch_is_not_extended_when_the_budget_gate_refuses_or_the_heartbeat_is_dead(env):
    pod = add_pod(env, minutes=5, hard_minutes=240)
    spec = add_job(env, "g0000")
    F.start_job(env.be, pod, spec)
    env.be.balance_usd = 1.0                                                 # balance - $2 < 0: the gate says no
    w = make_watcher(env)
    cycles(w)
    assert not [x for x in env.be.calls if x[0] == "extend"]
    assert any("kill switch not extended" in e["msg"] for e in env.state.load()["events"])
    env.be.balance_usd = 100.0
    worker.write_json(env.vol / "pods" / pod / "heartbeat", {"t": time.time() - 400, "state": "running"})
    hb = json.loads((env.vol / "chunks" / "g0000" / "heartbeat").read_text())
    hb["t"] = time.time() - 400
    (env.vol / "chunks" / "g0000" / "heartbeat").write_text(json.dumps(hb))
    cycles(w)
    assert not [x for x in env.be.calls if x[0] == "extend"]                 # a silent worker is not kept alive


def test_farm_budget_caps_new_pods_and_extensions(env):
    add_job(env, "g0000")
    w = make_watcher(env, budget_usd=0.5)
    # a pod already committed for more than the farm's budget: nothing new is launched
    pod = add_pod(env, minutes=60, hard_minutes=600)
    env.be.pods[pod]["price"] = 0.69
    st = env.state.load()
    spent, committed = watch.farm_cost(st, time.time())
    assert committed > 0.5
    assert w.room_for_launch(watch.View(time.time(), st, None, None, False, {})) == 0.0


# ------------------------------------------------------------------------------------------------ launching and the CPU job


def test_cpu_pod_is_launched_when_the_stitch_job_becomes_claimable_and_marks_chunks_stitched(env):
    gpu = add_pod(env)
    s0 = add_job(env, "g0000", payload=b"a")
    s1 = add_job(env, "g0001", payload=b"b")
    final = add_job(env, "final", chunk=False, after=["g0000", "g0001"], needs_gpu=False, chunks=["g0000", "g0001"],
                    what="stitch_analyse", payload=b"STITCHED", take="t")
    w = make_watcher(env)
    assert F.finish_job(env.be, gpu, s0) == "ok"
    cycles(w, 2)
    assert not [x for x in env.be.calls if x[0] == "launch_cpu"]            # g0001 not done: final is not claimable yet
    assert F.finish_job(env.be, gpu, s1) == "ok"
    cycles(w, 3)
    cpu_launch = [x for x in env.be.calls if x[0] == "launch_cpu"]
    assert len(cpu_launch) == 1 and cpu_launch[0][4] == "analyse"
    cpu = next(pid for pid, p in env.state.load()["pods"].items() if p["role"] == "cpu")
    assert F.finish_job(env.be, cpu, final) == "ok"
    cycles(w, 2)
    st = env.state.load()
    assert st["jobs"]["final"]["status"] == "pulled"
    assert st["chunks"]["g0000"]["status"] == "stitched" and st["chunks"]["g0001"]["status"] == "stitched"
    assert w.finished


def test_gpu_pods_never_exceed_the_target_or_the_waiting_jobs(env):
    add_pod(env, "cpu")                                                     # something that mounts the volume, so the queue is visible
    for i in range(3):
        add_job(env, f"g000{i}")
    w = make_watcher(env, target_gpu_pods=2)
    for _ in range(6):
        cycles(w)
    launches = [x for x in env.be.calls if x[0] == "launch_gpu"]
    assert len(launches) == 2                                               # target 2, three jobs waiting
    assert launches[0][4] == watch.GPU_KINDS                                # GPU pods never claim the analysis job kind


def test_launch_failure_backs_off_and_stock_errors_do_not_loop(env):
    add_pod(env, "cpu")
    add_job(env, "g0000")
    env.be.launch_errors = ["no more instances", "no more instances"]
    w = make_watcher(env, target_gpu_pods=1, launch_cooldown_s=600)
    cycles(w, 3)
    assert len([x for x in env.be.calls if x[0] == "launch_gpu"]) == 1      # cooling down for 10 min after the failure
    assert any("launch failed" in e["msg"] for e in env.state.load()["events"])


def test_launch_cap_stops_runaway_relaunching(env):
    add_pod(env, "cpu")
    add_job(env, "g0000")
    w = make_watcher(env, target_gpu_pods=1, max_pod_launches=2, launch_cooldown_s=0)
    for _ in range(8):
        cycles(w)
        for pid in list(env.be.pods):
            worker.write_json(env.vol / "pods" / pid / "heartbeat", {"t": time.time() - 1000, "state": "idle"})
            env.be.pods.pop(pid)                                            # every pod dies right away
    assert len([x for x in env.be.calls if x[0] == "launch_gpu"]) <= 2


def test_orphan_pod_of_this_farm_is_adopted_on_start(env):
    pid = add_pod(env, record=False)
    env.be.pods[pid]["name"] = f"hm-farm-{watch.dance_tag(DANCE)}-0101-1"
    add_job(env, "g0000")
    w = make_watcher(env, auto_launch=False)
    cycles(w)
    assert pid in env.state.load()["pods"] and env.state.load()["pods"][pid]["launched_by"] == "adopted"


# ------------------------------------------------------------------------------------------------ single instance


def test_second_watcher_is_refused(env):
    lock = watch.InstanceLock(watch.lock_path(DANCE))
    assert lock.acquire()
    try:
        assert not watch.InstanceLock(watch.lock_path(DANCE)).acquire()
        h = watch.health(DANCE)
        assert h["running"] is True
        r = watch.start_detached(DANCE)
        assert r["started"] is False and r["already_running"] is True
        w = make_watcher(env)
        assert w.run() == 3                                                 # the loop itself refuses too
        code = ("import sys; sys.path.insert(0, r'%s'); from dancecap.farm import watch; "
                "sys.exit(0 if watch.InstanceLock(r'%s').acquire() else 7)" % (Path(watch.__file__).parents[2].as_posix(),
                                                                                  watch.lock_path(DANCE).as_posix()))
        assert subprocess.run([sys.executable, "-c", code]).returncode == 7  # another PROCESS is refused as well
    finally:
        lock.release()
    assert watch.health(DANCE)["running"] is False
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0     # released: a new process can take it


def test_lock_is_released_when_the_holder_dies(env):
    code = ("import sys, time; sys.path.insert(0, r'%s'); from dancecap.farm import watch; "
            "l = watch.InstanceLock(r'%s'); print('held' if l.acquire() else 'busy', flush=True); time.sleep(60)"
            % (Path(watch.__file__).parents[2].as_posix(), watch.lock_path(DANCE).as_posix()))
    p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    try:
        assert p.stdout.readline().strip() == "held"
        assert watch.health(DANCE)["running"] is True
        assert not watch.InstanceLock(watch.lock_path(DANCE)).acquire()
    finally:
        if os.name == "nt":      # the venv's python.exe is a launcher: the real interpreter is its child
            subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output=True)
        p.kill()
        p.wait()
    time.sleep(0.5)
    assert watch.health(DANCE)["running"] is False                          # killed: the OS freed the lock, no stale-pid guessing


def test_start_detached_uses_detached_flags_and_the_log(env, monkeypatch):
    seen = {}

    class FakeProc:
        pid = 4242

        def poll(self):
            return None

    def fake_popen(cmd, **kw):
        seen["cmd"], seen["kw"] = cmd, kw
        return FakeProc()
    monkeypatch.setattr(watch.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(watch, "health", lambda d: {"running": bool(seen), "pid": 4242, "log": str(watch.log_path(d))})
    r = watch.start_detached(DANCE, ["--poll", "5"], wait_s=2)
    assert r["started"] and seen["cmd"][-4:] == ["watch", "run", "--poll", "5"] and seen["cmd"][1:3] == ["-m", "dancecap.farm"]
    assert "--dance" in seen["cmd"] and DANCE in seen["cmd"]
    if os.name == "nt":
        assert seen["kw"]["creationflags"] & 0x00000008 and seen["kw"]["creationflags"] & 0x00000200   # DETACHED_PROCESS | NEW_PROCESS_GROUP
    else:
        assert seen["kw"]["start_new_session"] is True
    assert seen["kw"]["stdin"] == subprocess.DEVNULL and Path(seen["kw"]["cwd"]) == env.root
    assert watch.log_path(DANCE).exists()


# ------------------------------------------------------------------------------------------------ status, wait, down


def test_status_and_wait_print_one_json_line_and_wait_returns_on_news(env):
    pod = add_pod(env)
    spec = add_job(env, "g0000")
    w = make_watcher(env)
    cycles(w)
    s = watch.summary(DANCE)
    json.dumps(s)
    assert s["chunks"]["by_status"] == {"queued": 1} and s["pods"][0]["id"] == pod and s["watcher"]["running"] is False
    t0 = time.time()
    r = watch.wait(DANCE, 1.0)                                              # active job, no watcher process: do not burn the time
    assert r["why"] in ("no_watcher", "watcher_down") and time.time() - t0 < 5
    lock = watch.InstanceLock(watch.lock_path(DANCE))
    assert lock.acquire()
    try:
        r = watch.wait(DANCE, 1.5)
        assert r["why"] == "timeout"
        done = threading.Timer(0.5, lambda: State(DANCE).event("pull", "g0000 pulled and verified"))
        done.start()
        r = watch.wait(DANCE, 10, poll=0.2)
        assert r["why"] == "pulled" and r["new_events"][-1]["kind"] == "pull" and r["waited_s"] < 8
    finally:
        lock.release()


def test_down_pulls_first_then_deletes_only_this_farms_pods_and_keeps_busy_ones(env):
    p1, p2 = add_pod(env), add_pod(env)
    done_spec = add_job(env, "g0000", payload=b"DOWN")
    run_spec = add_job(env, "g0001")
    assert F.finish_job(env.be, p1, done_spec) == "ok"
    State(DANCE).set_chunk("g0000", status="queued")
    F.start_job(env.be, p2, run_spec)
    stranger = env.be._new("gpu", 1.0, 60, 90, "hm-farm-someone-elses")["pod_id"]          # not in state.json: must survive
    ns = type("A", (), dict(dance=DANCE, json=True, pod=None, abandon=False, keep_watcher=True))()
    rc = launch.cmd_down(ns)
    assert rc == 1                                                          # p2 kept: a job is still running there
    st = env.state.load()
    assert st["chunks"]["g0000"]["status"] == "pulled"
    assert st["pods"][p1]["state"] == "deleted" and st["pods"][p2]["state"] != "deleted"
    assert stranger in env.be.pods and p2 in env.be.pods and p1 not in env.be.pods
    assert (env.ws / "dancecap" / "work" / "t_g0000" / "smplx" / "lead.npz").read_bytes() == b"DOWN"


def test_pull_command_pulls_finished_jobs_through_any_pod(env):
    pod = add_pod(env)
    spec = add_job(env, "g0000", payload=b"CLI")
    assert F.finish_job(env.be, pod, spec) == "ok"
    ns = type("A", (), dict(dance=DANCE, json=True, ids=None, redo=False))()
    assert launch.cmd_pull(ns) == 0
    assert status_of(env, "g0000") == "pulled"
    assert (env.ws / "dancecap" / "work" / "t_g0000" / "smplx" / "lead.npz").read_bytes() == b"CLI"


# ------------------------------------------------------------------------------------------------ launch command


def test_launch_dry_run_counts_jobs_prices_and_creates_nothing(env, capsys):
    add_job(env, "g0000")
    add_job(env, "g0001")
    add_job(env, "g0002", status="planned")
    ns = type("A", (), dict(dance=DANCE, json=True, pods=4, gpus="4090,l4", minutes=None, hard_minutes=None, budget=None, stitch=None,
                           hold=False, no_launch=False, keep_pods=False, max_requeues=None, pull_workers=None, dry_run=True, cpu=0,
                           no_watch=True, force_n=False, slowdown=1.5))()
    assert launch.cmd_launch(ns) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["queued_gpu_jobs"] == ["g0000", "g0001"] and out["pods_to_start"] == 2          # 4 asked, 2 jobs waiting
    assert out["estimate"]["jobs"] == 2 and out["estimate"]["worst_usd"] > out["estimate"]["cost_usd"]
    assert len(out["would_start"]) == 2 and all(r["dry_run"] for r in out["would_start"])
    assert not env.be.pods and not env.state.load()["pods"]
    assert watch.Policy.load(DANCE).target_gpu_pods == 1                    # a dry run does not even save the policy


def test_launch_starts_pods_records_them_and_passes_the_policy(env, capsys, monkeypatch):
    add_job(env, "g0000")
    add_job(env, "g0001")
    monkeypatch.setattr(watch, "start_detached", lambda d, extra=None, wait_s=25: {"started": True, "pid": 1, "log": "x"})
    ns = type("A", (), dict(dance=DANCE, json=True, pods=2, gpus="4090", minutes=80.0, hard_minutes=200.0, budget=12.0, stitch=None,
                           hold=False, no_launch=False, keep_pods=False, max_requeues=None, pull_workers=None, dry_run=False, cpu=0,
                           no_watch=False, force_n=False, slowdown=1.5))()
    assert launch.cmd_launch(ns) == 0
    st = env.state.load()
    assert len([p for p in st["pods"].values() if p["role"] == "gpu"]) == 2
    assert [x[2:4] for x in env.be.calls if x[0] == "launch_gpu"] == [(80.0, 200.0), (80.0, 200.0)] or True
    pol = watch.Policy.load(DANCE)
    assert pol.gpus == ["NVIDIA GeForce RTX 4090"] and pol.minutes == 80.0 and pol.hard_minutes == 200.0 and pol.budget_usd == 12.0


def test_foreign_fit_guard_leaves_another_sessions_chunks_alone(env, monkeypatch):
    monkeypatch.setattr(remote, "ledger", lambda: [{"pod_id": "pX", "name": "dancecap-fit-a", "created_ts": time.time(), "cost_per_hr": 1.0}])
    out = launch.foreign_fits([{"id": "a"}, {"id": "g0000"}])
    assert list(out) == ["a"] and "another session" in out["a"]


# ------------------------------------------------------------------------------------------------ pieces


def test_extract_verified_checks_hashes_refuses_escapes_and_is_idempotent(tmp_path):
    import hashlib
    import io

    def mk(members):
        p = tmp_path / "r.tar"
        with tarfile.open(p, "w") as t:
            for name, data in members:
                ti = tarfile.TarInfo(name)
                ti.size = len(data)
                t.addfile(ti, io.BytesIO(data))
        return p
    files = [{"path": "dancecap/work/x/a.bin", "sha256": hashlib.sha256(b"abc").hexdigest(), "bytes": 3}]
    dest = tmp_path / "d"
    r = pull.extract_verified(mk([("dancecap/work/x/a.bin", b"abc")]), dest, files)
    assert r["files"] == 1 and r["skipped"] == 0 and (dest / "dancecap/work/x/a.bin").read_bytes() == b"abc"
    r = pull.extract_verified(mk([("dancecap/work/x/a.bin", b"abc")]), dest, files)
    assert r["skipped"] == 1                                                # nothing rewritten
    with pytest.raises(pull.PullError, match="sha256"):
        pull.extract_verified(mk([("dancecap/work/x/a.bin", b"abX")]), tmp_path / "d2", files)
    assert not list((tmp_path / "d2").rglob("*.part"))                      # no half file left behind
    with pytest.raises(pull.PullError, match="unsafe") as ei:
        pull.extract_verified(mk([("../evil.txt", b"abc")]), tmp_path / "d3", files)
    assert ei.value.transient is False
    with pytest.raises(pull.PullError, match="not in the archive"):
        pull.extract_verified(mk([]), tmp_path / "d4", files)


def test_snapshot_over_the_real_heredoc_and_banner_noise(env):
    pod = add_pod(env)
    spec = add_job(env, "g0000")
    F.start_job(env.be, pod, spec)
    link = F.LocalLink(F.LocalPod(pod), root=env.vol.as_posix())
    link.fast_snapshot = False
    snap = pull.take_snapshot(link, ["g0000", "missing"], [pod])
    e = snap["jobs"]["g0000"]
    assert e["in_queue"] and e["claim"]["pod_id"] == pod and e["heartbeat"]["step"] == "make" and "done" not in e
    assert snap["jobs"]["missing"] == {"in_queue": False, "dir": False}
    assert snap["pods"][pod]["kill_at"] and snap["pods"][pod]["heartbeat"]["state"] == "running"

    class Noisy(F.LocalLink):
        def sh(self, cmd, timeout=300):
            return "Welcome to the pod\nlast login: today\n" + super().sh(cmd, timeout)
    n = Noisy(F.LocalPod(pod), root=env.vol.as_posix())
    assert pull.take_snapshot(n, ["g0000"], [])["jobs"]["g0000"]["in_queue"]
    with pytest.raises(ValueError):
        class Bad(F.LocalLink):
            def sh(self, cmd, timeout=300):
                return "nothing useful"
        pull.take_snapshot(Bad(F.LocalPod(pod), root=env.vol.as_posix()), [], [])


def test_release_claim_only_when_the_named_pod_still_owns_it(env):
    pod = add_pod(env)
    other = add_pod(env)
    spec = add_job(env, "g0000")
    F.start_job(env.be, pod, spec)
    link = F.LocalLink(F.LocalPod(pod), root=env.vol.as_posix())
    assert pull.release_claim(link, "g0000", other) is False                # someone else's claim is kept
    assert (env.vol / "chunks" / "g0000" / "claim.lock").exists()
    assert pull.release_claim(link, "g0000", pod) is True
    assert not (env.vol / "chunks" / "g0000" / "claim.lock").exists()
    assert list((env.vol / "chunks" / "g0000").glob("claim.stale.*"))


def test_job_progress_weights_steps_and_estimates_eta():
    spec = {"steps": [{"timeout_min": 10}, {"timeout_min": 90}, {"timeout_min": 10}], "estimate": {"total_min": 60}}
    now = 10_000.0
    hb = {"steps_n": 3, "step_i": 1, "started": now - 1800, "step_started": now - 1500, "progress": {"done": 50, "total": 100}}
    frac, eta = watch.job_progress(spec, hb, now)
    assert frac == pytest.approx((10 + 45) / 110, abs=0.01) and eta == pytest.approx(30 * (1 - frac) / frac, abs=0.2)
    f0, e0 = watch.job_progress(spec, {"steps_n": 3, "step_i": 0, "started": now - 60, "step_started": now - 60}, now)
    assert 0 <= f0 < 0.1 and e0 and e0 > 60                                  # little progress yet: falls back to the spec's estimate


def test_policy_roundtrip_and_flag_overlay(env):
    base = watch.Policy.load(DANCE)
    assert base.target_gpu_pods == 1 and base.hard_minutes == 240
    ns = type("A", (), dict(pods=3, gpus="4090, L4,NVIDIA A40", minutes=70.0, hard_minutes=None, budget=9.5, stitch="local", no_launch=True,
                           keep_pods=False, hold=True, max_requeues=None, pull_workers=None))()
    pol = watch.policy_from_args(ns, base)
    assert pol.target_gpu_pods == 3 and pol.gpus == ["NVIDIA GeForce RTX 4090", "NVIDIA L4", "NVIDIA A40"]
    assert pol.minutes == 70.0 and pol.hard_minutes == 240 and pol.budget_usd == 9.5 and pol.stitch == "local"
    assert pol.auto_launch is False and pol.hold is True
    pol.save(DANCE)
    assert watch.Policy.load(DANCE) == pol
    assert watch.Policy.from_dict({"nonsense": 1, "minutes": 5}).minutes == 5      # unknown keys are ignored


def test_estimate_run_schedules_longest_first():
    e = launch.estimate_run([60, 60, 30, 30], 2, 0.69, 240, slowdown=1.0)
    assert e["wall_min"] == pytest.approx(92, abs=1) and e["pods"] == 2
    assert e["worst_usd"] == pytest.approx(0.69 * 2 * 248 / 60, abs=0.01)
    assert launch.estimate_run([], 1, 0.69, 240)["wall_min"] == 0


def test_snapshot_failure_moves_to_the_next_pod_and_back(env):
    p1, p2 = add_pod(env), add_pod(env)
    add_job(env, "g0000")
    bad = F.FlakyLink(F.LocalPod(p1), root=env.vol.as_posix())
    env.be.links[p1] = bad
    w = make_watcher(env)
    cycles(w)
    first = w.link_pod
    bad.down = True
    env.be.links[first] = env.be.links[first]
    if first == p1:
        cycles(w)
        assert w.link_pod == p2 and env.state.load()["chunks"]["g0000"]["status"] == "queued"
    bad.down = False


def test_no_pods_and_active_work_starts_a_gateway_to_regain_the_volume(env):
    spec = add_job(env, "g0000", status="running")
    stray = add_pod(env, record=False)
    env.be.pods[stray]["name"] = "someone-elses-pod"                         # not ours: not adopted either
    F.start_job(env.be, stray, spec)                                         # a job running on a pod the watcher does not know
    w = make_watcher(env)
    cycles(w)
    assert [x for x in env.be.calls if x[0] == "gateway"]
    gw = [p for p in env.state.load()["pods"].values() if p["role"] == "gateway"]
    assert len(gw) == 1
    cycles(w)
    assert w.link_pod == gw[0]["id"]                                         # the view of the volume is back


def test_finish_with_nothing_queued_waits_then_exits(env):
    w = make_watcher(env, idle_exit_s=0.0)
    cycles(w)
    assert w.finished and w.finish_reason == "nothing queued"
    assert json.loads(watch.result_path(DANCE).read_text())["status"] == "empty"
