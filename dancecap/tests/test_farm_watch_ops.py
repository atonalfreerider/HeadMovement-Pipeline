"""Farm watcher, part 2: the volume uploader behind `farm queue`, the optional local stitch, held jobs, the run loop itself (a bad
cycle, the stop file, the pid file) and the light snapshot of dependencies. Fake pods only: no network, nothing billable.

    ..\\.venv-nopo4d\\Scripts\\python.exe -m pytest tests/test_farm_watch_ops.py -q      (from dancecap/)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
from pathlib import Path

import pytest

import _farm_fixtures as FX
import _farm_watch_fixtures as F
from _farm_watch_env import DANCE, add_job, add_pod, cycles, env, make_watcher, status_of  # noqa: F401

from dancecap import remote
from dancecap.farm import cli, launch, pull, watch
from dancecap.farm.state import State, atomic_write_json

pytestmark = pytest.mark.skipif(F.BASH is None, reason="needs bash to play the pod")


def test_queue_upload_pushes_inputs_pins_the_code_archives_stale_runs_and_honors_only(env, monkeypatch):
    pod = add_pod(env)
    monkeypatch.setattr(launch, "ensure_synced", lambda link, dance, specs: ({"bundle_sha256": "abc", "key": "k1", "git_sha": "g1",
                                                                              "dirty": False}, {}))
    monkeypatch.setattr(remote, "ledger", lambda: [])
    data = b"PREPARED" * 100
    inp = env.ws / "dancecap" / "work" / "t" / "farm" / "in" / "g1" / "prepared.tar"
    inp.parent.mkdir(parents=True)
    inp.write_bytes(data)
    spec = F.job_spec("g1")
    spec["inputs"] = [{"path": "dancecap/work/t_g1/mamma/prepared.tar", "archive": True, "sha256": hashlib.sha256(data).hexdigest(),
                       "bytes": len(data), "local": "dancecap/work/t/farm/in/g1/prepared.tar"}]
    spec["code"] = {"git_sha": "old", "dirty": True, "bundle_sha256": None}
    plain = F.job_spec("g2")
    stale, claimed, other = F.job_spec("g3"), F.job_spec("g4"), F.job_spec("g5")
    for jid in ("g3", "g4"):
        (env.vol / "chunks" / jid).mkdir(parents=True)
        (env.vol / "chunks" / jid / "done.json").write_text("{}")
    (env.vol / "chunks" / "g4" / "claim.lock").mkdir()
    ns = argparse.Namespace(only="g1,g2,g3,g4", exclude="g2", keep_gateway=False)
    res = launch.queue_upload(DANCE, [spec, plain, stale, claimed, other], ns)
    assert set(res) == {"g1", "g3", "g4"}                                   # g2 excluded, g5 not in --only
    assert res["g1"]["ok"] and res["g3"]["ok"] and not res["g4"]["ok"] and "claim" in res["g4"]["error"]
    q = json.loads((env.vol / "queue" / "g1.json").read_text())
    assert q["code"]["bundle_sha256"] == "abc" and q["code"]["key"] == "k1" and q["inputs"][0]["bytes"] == len(data)
    assert "local" not in q["inputs"][0]
    assert (env.vol / "chunks" / "g1" / "in" / "dancecap/work/t_g1/mamma/prepared.tar").read_bytes() == data
    assert not (env.vol / "chunks" / "g3" / "done.json").exists()           # the old run was moved aside, not deleted
    assert list((env.vol / "chunks").glob("g3.old.*"))
    assert (env.vol / "chunks" / "g4" / "done.json").exists()               # a claimed job is never touched
    assert not (env.vol / "queue" / "g2.json").exists()
    assert pod in env.state.load()["pods"]


def test_queue_upload_skips_chunks_another_session_is_fitting(env, monkeypatch):
    add_pod(env)
    monkeypatch.setattr(launch, "ensure_synced", lambda link, dance, specs: ({"bundle_sha256": "abc", "key": "k"}, {}))
    monkeypatch.setattr(remote, "ledger", lambda: [{"pod_id": "pX", "name": "dancecap-fit-a", "created_ts": time.time(), "cost_per_hr": 1.0}])
    res = launch.queue_upload(DANCE, [F.job_spec("a"), F.job_spec("g0000")], None)
    assert res["a"]["ok"] is False and "another session" in res["a"]["error"]
    assert res["g0000"]["ok"] and (env.vol / "queue" / "g0000.json").exists() and not (env.vol / "queue" / "a.json").exists()


def test_queue_upload_reports_a_world_that_cannot_be_synced(env, monkeypatch):
    add_pod(env)
    monkeypatch.setattr(launch, "ensure_synced", lambda link, dance, specs: ({"bundle_sha256": "abc", "key": "k"}, {"w1": {"error": "no data tree"}}))
    monkeypatch.setattr(remote, "ledger", lambda: [])
    res = launch.queue_upload(DANCE, [F.job_spec("g0000", world_id="w1"), F.job_spec("g0001")], None)
    assert not res["g0000"]["ok"] and "world w1" in res["g0000"]["error"] and res["g0001"]["ok"]


def test_cli_loads_the_plugin_for_queue_and_registers_every_command(env, monkeypatch):
    """`farm queue` is a core command; it must still find the volume uploader (cli.HOOK_COMMANDS), and register() adds the commands."""
    monkeypatch.setattr(cli, "HOOKS", {})
    monkeypatch.setattr(cli, "PLUGINS", {})
    assert "queue" in cli.HOOK_COMMANDS
    ap = cli.build_parser()
    sub = next(a for a in ap._actions if isinstance(a, argparse._SubParsersAction))
    cli.load_plugins(sub)
    assert cli.PLUGINS["watch"]["registered"] and cli.PLUGINS["watch"]["error"] is None
    assert cli.HOOKS["queue_upload"]
    for name in ("watch", "wait", "launch", "up", "pull", "down"):
        assert name in sub.choices


def test_local_stitch_runs_dancecap_stitch_on_the_pulled_chunks(env, monkeypatch):
    from dancecap.farm import timeline as tl
    pod = add_pod(env)
    spec = add_job(env, "g0000", payload=b"a")
    State(DANCE).set_chunk("g0000", core_k0=0, core_k1=899, origin="planned")
    assert F.finish_job(env.be, pod, spec) == "ok"
    calls = []

    class Done:
        returncode, stdout, stderr = 2, "gate failed", ""
    real_run = watch.subprocess.run

    def fake_run(cmd, **kw):
        if "dancecap.stitch" in cmd:
            calls.append((cmd, kw))
            return Done()
        return real_run(cmd, **kw)                                          # the fake pod's bash still runs for real
    monkeypatch.setattr(watch.subprocess, "run", fake_run)
    monkeypatch.setattr(tl.DanceSpec, "load", classmethod(lambda cls, dance=None, **k: FX.make_spec(DANCE)))
    w = make_watcher(env, stitch="local")
    cycles(w, 3)
    assert calls and calls[0][0][1:4] == ["-m", "dancecap.stitch", DANCE] and "--partial" in calls[0][0]
    assert calls[0][1]["env"]["OMP_NUM_THREADS"] == "1" and Path(calls[0][1]["cwd"]) == env.root
    lst = json.loads((watch.farm_dir(DANCE) / "stitch_list.json").read_text())
    assert [c["name"] for c in lst["chunks"]] == ["g0000"] and lst["partial"] is True
    assert status_of(env, "g0000") == "stitched"                            # rc 2 = a seam gate failed: a verdict, not a failure
    assert any("seam gate failed" in e["msg"] for e in env.state.load()["events"])


def test_held_jobs_are_released_every_cycle(env, monkeypatch):
    add_pod(env)
    add_job(env, "g0000")
    seen = []
    monkeypatch.setattr(cli, "release_ready", lambda dance, args=None: seen.append(dance))
    w = make_watcher(env, hold=True)
    cycles(w, 2)
    assert seen == [DANCE, DANCE]


def test_run_loop_survives_a_bad_cycle_stops_on_the_stop_file_and_writes_its_pidfile(env):
    add_pod(env)
    add_job(env, "g0000")
    w = make_watcher(env)
    boom = {"n": 0}
    orig = w.cycle

    def flaky():
        boom["n"] += 1
        if boom["n"] == 1:
            raise RuntimeError("synthetic failure")
        orig()
    w.cycle = flaky
    t = threading.Thread(target=w.run, daemon=True)
    t.start()
    try:
        for _ in range(100):
            time.sleep(0.1)
            if boom["n"] >= 3:
                break
        assert boom["n"] >= 3 and t.is_alive()                              # one bad cycle did not end the loop
        h = watch.health(DANCE)
        assert h["running"] and h["pid"] == os.getpid() and h["heartbeat_age_s"] < 10
        assert w.stats["errors"] == 1 and any(e["kind"] == "error" for e in env.state.load()["events"])
    finally:
        watch.stop_path(DANCE).write_text("stop")                           # never leave the loop running into the next test
        t.join(15)
    assert not t.is_alive() and not watch.health(DANCE)["running"]
    assert env.state.load()["watcher"]["state"] == "stopped"
    assert json.loads(watch.pid_path(DANCE).read_text())["state"] == "stopped"


def test_whole_run_through_the_loop_ends_with_result_json_and_no_pods(env):
    pod = add_pod(env)
    spec = add_job(env, "g0000", payload=b"LOOP")
    assert F.finish_job(env.be, pod, spec) == "ok"
    w = make_watcher(env)
    t = threading.Thread(target=w.run, daemon=True)
    t.start()
    t.join(60)
    assert not t.is_alive() and w.finished
    res = json.loads(watch.result_path(DANCE).read_text())
    assert res["status"] == "complete" and res["pods"][0]["state"] == "deleted" and res["coverage"] is None
    assert env.state.load()["watcher"]["state"] == "finished"
    assert watch.summary(DANCE)["finished"] is True


def test_snapshot_dependencies_are_answered_light(env):
    pod = add_pod(env)
    s0 = add_job(env, "g0000", payload=b"a")
    assert F.finish_job(env.be, pod, s0) == "ok"
    link = F.LocalLink(F.LocalPod(pod), root=env.vol.as_posix())
    snap = pull.take_snapshot(link, ["final"], [pod], deps=["g0000", "nope"])
    assert snap["jobs"]["g0000"]["done"]["status"] == "ok" and "manifest" not in snap["jobs"]["g0000"]
    assert snap["jobs"]["nope"] == {"dir": False}


def test_job_missing_from_the_volume_is_reported_once_not_looped_on(env):
    add_pod(env)
    add_job(env, "g0000")
    (env.vol / "queue" / "g0000.json").unlink()
    w = make_watcher(env)
    cycles(w, 3)
    msgs = [e["msg"] for e in env.state.load()["events"] if "not on the volume" in e["msg"]]
    assert len(msgs) == 1 and status_of(env, "g0000") == "queued"


def test_up_queues_the_planned_jobs_launches_pods_for_them_and_starts_the_watcher(env, monkeypatch, capsys):
    """`farm up` end to end with fakes: planned specs -> uploader hook -> state 'queued' -> pods launched and recorded -> watcher."""
    monkeypatch.setattr(launch, "ensure_synced", lambda link, dance, specs: ({"bundle_sha256": "abc", "key": "k1", "git_sha": "g"}, {}))
    monkeypatch.setattr(remote, "ledger", lambda: [])
    monkeypatch.setattr(cli, "cmd_plan", lambda a: 0)                       # planning is core's (tested there): specs are written below
    started = []
    monkeypatch.setattr(watch, "start_detached", lambda d, extra=None, wait_s=25: (started.append(d), {"started": True, "pid": 7, "log": "x"})[1])
    monkeypatch.setattr(cli, "HOOKS", {"queue_upload": [launch.queue_upload]})
    add_pod(env, "cpu")                                                     # the link to the volume (a gateway / any pod)
    for jid in ("g0000", "g0001", "a"):
        atomic_write_json(watch.jobs_dir(DANCE) / f"{jid}.json", F.job_spec(jid, kind="fit", needs_gpu=True))
        env.state.put_chunks([{"id": jid, "k0": 0, "k1": 9, "core_k0": 0, "core_k1": 9, "take": f"t_{jid}", "status": "planned", "kind": "fit"}],
                             replace_planned=False)
        env.state.set_job(jid, status="planned")
    ns = argparse.Namespace(dance=DANCE, json=True, dry_run=False, no_plan=False, no_final=False, only=None, exclude="a", pods=2, gpus="4090",
                            minutes=None, hard_minutes=None, budget=None, stitch=None, hold=False, no_launch=False, keep_pods=False,
                            max_requeues=None, pull_workers=None, slowdown=1.5, force_n=False)
    monkeypatch.setattr(remote, "ledger", lambda: [])
    assert launch.cmd_up(ns) == 0
    out = json.loads(capsys.readouterr().out)
    st = env.state.load()
    assert sorted(out["steps"]["queue"]["queued"]) == ["g0000", "g0001"] and st["chunks"]["a"]["status"] == "planned"   # a: excluded
    assert (env.vol / "queue" / "g0000.json").exists() and not (env.vol / "queue" / "a.json").exists()
    assert st["chunks"]["g0000"]["status"] == "queued"
    gpus = [p for p in st["pods"].values() if p["role"] == "gpu"]
    assert len(gpus) == 2 and gpus[0]["launched_by"] == "launch" and gpus[0]["hard_ceil"]
    assert started == [DANCE]
    assert watch.Policy.load(DANCE).gpus == ["NVIDIA GeForce RTX 4090"]


def test_up_dry_run_changes_nothing_and_lists_what_it_would_do(env, monkeypatch, capsys):
    add_pod(env, "cpu")
    monkeypatch.setattr(cli, "cmd_plan", lambda a: 0)
    for jid in ("g0000", "a"):
        atomic_write_json(watch.jobs_dir(DANCE) / f"{jid}.json", F.job_spec(jid, kind="fit", needs_gpu=True))
        env.state.put_chunks([{"id": jid, "k0": 0, "k1": 9, "core_k0": 0, "core_k1": 9, "take": f"t_{jid}", "status": "planned", "kind": "fit"}],
                             replace_planned=False)
        env.state.set_job(jid, status="planned")
    before = json.dumps(env.state.load()["chunks"], sort_keys=True)
    ns = argparse.Namespace(dance=DANCE, json=True, dry_run=True, no_plan=False, no_final=False, only=None, exclude="a", pods=None, gpus=None,
                            minutes=None, hard_minutes=None, budget=None, stitch=None, hold=False, no_launch=False, keep_pods=False,
                            max_requeues=None, pull_workers=None, slowdown=1.5, force_n=False)
    assert launch.cmd_up(ns) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["dry_run"] and out["steps"]["queue"]["ready"] == ["g0000"]
    assert json.dumps(env.state.load()["chunks"], sort_keys=True) == before and not (env.vol / "queue").exists()


def test_prefer_volume_results_swaps_uploads_for_results_the_volume_already_holds(env):
    pod = add_pod(env)
    done_spec = add_job(env, "g0000", payload=b"x")
    assert F.finish_job(env.be, pod, done_spec) == "ok"                     # the farm produced g0000: result.tar is on the volume
    link = F.LocalLink(F.LocalPod(pod), root=env.vol.as_posix())
    spec = {"id": "final", "after": ["a"], "local_chunks": ["g0000", "a"],
            "stitch_chunks": [{"id": "g0000", "take": "t_g0000"}, {"id": "a", "take": "t_a"}],
            "inputs": [{"path": "dancecap/work/t_g0000/motion.tar", "local": "x", "sha256": "1"},
                       {"path": "dancecap/work/t_a/motion.tar", "local": "y", "sha256": "2"},
                       {"path": "dancecap/work/t/reference.tar", "local": "z", "sha256": "3"}]}
    assert launch.prefer_volume_results(link, spec) == ["g0000"]            # 'a' exists only on this PC: it is still uploaded
    paths = [i["path"] for i in spec["inputs"]]
    assert "chunks/g0000/out/result.tar" in paths and "dancecap/work/t_g0000/motion.tar" not in paths
    assert "dancecap/work/t_a/motion.tar" in paths and "dancecap/work/t/reference.tar" in paths
    assert spec["after"] == ["a", "g0000"] and spec["local_chunks"] == ["a"]


def test_interim_stitch_is_queued_as_chunks_land_but_not_when_the_final_job_can_run(env, monkeypatch):
    from dancecap.farm import analysis as A
    from dancecap.farm import timeline as tl
    from dancecap.farm import world as W
    monkeypatch.setattr(launch, "ensure_synced", lambda link, dance, specs: ({"bundle_sha256": None, "key": "k"}, {}))
    monkeypatch.setattr(tl.DanceSpec, "load", classmethod(lambda cls, dance=None, **k: FX.make_spec(DANCE)))
    monkeypatch.setattr(W, "load", lambda *a, **k: (_ for _ in ()).throw(W.WorldError("no world in this test")))
    built = []

    def fake_build(spec, chunks, world=None, *, analyses=(), jid="final", partial=True, hash_inputs=True, code=None, **k):
        ids = [c["id"] for c in chunks]
        built.append((jid, ids, tuple(analyses), partial))
        return {**F.job_spec(jid, take=DANCE, after=ids, needs_gpu=False, kind="analyse"), "dance": DANCE, "what": "stitch_analyse", "chunks": ids,
                "local_chunks": [], "stitch_chunks": list(chunks)}
    monkeypatch.setattr(A, "build_final_job", fake_build)
    gpu = add_pod(env)
    s0, s1 = add_job(env, "g0000", payload=b"a"), add_job(env, "g0001", payload=b"b")
    add_job(env, "final", chunk=False, after=["g0000", "g0001"], needs_gpu=False, chunks=["g0000", "g0001"], what="stitch_analyse", take=DANCE)
    for jid in ("g0000", "g0001"):
        State(DANCE).set_chunk(jid, core_k0=0, core_k1=9)
    w = make_watcher(env, interim_stitch=True)
    assert F.finish_job(env.be, gpu, s0) == "ok"
    cycles(w, 3)
    assert built == [("stitch-1", ["g0000"], (), True)]                     # the reference + the one chunk that has landed
    assert (env.vol / "queue" / "stitch-1.json").exists()
    j = env.state.load()["jobs"]["stitch-1"]
    assert j["status"] in ("queued", "running", "done", "pulled") and j["sig"] == "g0000"
    cpu = next(pid for pid, p in env.state.load()["pods"].items() if p["role"] == "cpu")                 # claimable at once: CPU pod
    assert F.finish_job(env.be, cpu, json.loads((env.vol / "queue" / "stitch-1.json").read_text())) == "ok"
    cycles(w, 3)
    assert env.state.load()["jobs"]["stitch-1"]["status"] == "pulled" and status_of(env, "g0000") == "stitched"
    cycles(w, 2)
    assert len(built) == 1                                                  # nothing new landed: no second interim job
    assert F.finish_job(env.be, gpu, s1) == "ok"
    cycles(w, 3)
    assert len(built) == 1                                                  # the final job can run now: no interim for it
    assert not env.state.load()["jobs"].get("stitch-2")


def _launch_ns(**kw):
    base = dict(dance=DANCE, json=True, pods=None, gpus=None, minutes=None, hard_minutes=None, budget=None, stitch=None, hold=False,
                no_launch=False, keep_pods=False, max_requeues=None, pull_workers=None, dry_run=False, cpu=0, no_watch=True, force_n=False,
                slowdown=1.5, interim_stitch=False)
    base.update(kw)
    return argparse.Namespace(**base)


def test_second_launch_tops_up_to_the_target_and_never_doubles_the_pods(env, capsys):
    add_job(env, "g0000")
    add_job(env, "g0001")
    add_job(env, "g0002")
    assert launch.cmd_launch(_launch_ns(pods=2)) == 0
    capsys.readouterr()
    assert len([p for p in env.state.load()["pods"].values() if p["role"] == "gpu"]) == 2
    assert launch.cmd_launch(_launch_ns(pods=2)) == 2                       # nothing to add: reported, not an error of the farm
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["pods_to_start"] == 0 and len(out["pods_running"]) == 2 and "nothing to launch" in out["error"]
    assert launch.cmd_launch(_launch_ns(pods=3)) == 0                       # a third pod for the third waiting job
    capsys.readouterr()
    assert len([p for p in env.state.load()["pods"].values() if p["role"] == "gpu"]) == 3
    assert len([x for x in env.be.calls if x[0] == "launch_gpu"]) == 3


def test_down_abandon_deletes_a_busy_pod_but_never_one_with_unpulled_results(env, capsys):
    busy, done = add_pod(env), add_pod(env)
    run_spec, done_spec = add_job(env, "g0001"), add_job(env, "g0000", payload=b"D")
    F.start_job(env.be, busy, run_spec)
    assert F.finish_job(env.be, done, done_spec) == "ok"
    State(DANCE).set_chunk("g0000", status="queued")
    # the volume becomes unreachable for pulling: the pod that holds an unpulled result must stay
    env.be.links[done] = F.FlakyLink(F.LocalPod(done), root=env.vol.as_posix())
    # cut too: test_farm_watch.py leaves the class at cut = 3000, more than this 1-byte payload, and the pull then
    # succeeded whenever that file ran first (the "flaky" failure of the multi-file runs since round 1)
    F.FlakyLink.drops, F.FlakyLink.cut = 10_000, 64
    env.be.links[busy] = env.be.links[done]
    ns = argparse.Namespace(dance=DANCE, json=True, pod=None, abandon=True, keep_watcher=True)
    rc = launch.cmd_down(ns)
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert rc == 1 and done in out["kept"] and "not pulled" in out["kept"][done]
    assert busy in out["deleted"] and busy not in env.be.pods               # --abandon: a job still running there does not keep it
    assert env.state.load()["pods"][done]["state"] != "deleted"


def test_nothing_is_deleted_while_the_volume_cannot_be_read(env):
    pod = add_pod(env)
    spec = add_job(env, "g0000", payload=b"x")
    add_job(env, "g0001")
    assert F.finish_job(env.be, pod, spec) == "ok"                           # finished, not yet seen by the watcher
    bad = F.FlakyLink(F.LocalPod(pod), root=env.vol.as_posix())
    bad.down = True
    env.be.links[pod] = bad
    w = make_watcher(env, idle_grace_s=0.0)
    State(DANCE).upsert_pod(pod, state="idle", idle_since=time.time() - 3600)
    cycles(w, 3)
    assert not [x for x in env.be.calls if x[0] == "delete"] and pod in env.be.pods       # blind: nothing is deleted
    bad.down = False
    cycles(w, 3)
    assert status_of(env, "g0000") in ("pulled", "stitched")


def test_ensure_synced_puts_code_and_the_world_on_the_volume_once(env, monkeypatch):
    from dancecap.farm import volume
    monkeypatch.setattr(volume, "PACKAGE_ROOT", env.root)
    monkeypatch.setattr(volume, "sync_code", lambda link: {"key": "k9", "git_sha": "abcdef123456", "dirty": False, "bundle_sha256": "s" * 64})
    data = env.root / "work" / "worlds" / DANCE / "w1"
    (data / "shared").mkdir(parents=True)
    (data / "shared" / "betas.json").write_text("[1, 2]")
    (data / "big.bin").write_bytes(bytes(range(256)) * 20000)                # 5 MB: goes through the striped path
    (data / "manifest.json").write_text('{"world_id": "w1"}')
    pod = add_pod(env)
    link = F.LocalLink(F.LocalPod(pod), root=env.vol.as_posix())
    code, worlds = launch.ensure_synced(link, DANCE, [{"world_id": "w1"}, {"world_id": "w1"}, {"world_id": None}])
    assert code["key"] == "k9" and list(worlds) == ["w1"] and worlds["w1"]["skipped"] is not True
    assert (env.vol / "worlds" / "w1" / "shared" / "betas.json").read_text() == "[1, 2]"
    assert (env.vol / "worlds" / "w1" / "big.bin").stat().st_size == 5_120_000
    assert (env.vol / "worlds" / "w1" / "manifest.json").exists()
    _, again = launch.ensure_synced(link, DANCE, [{"world_id": "w1"}])
    assert again["w1"]["skipped"] is True                                   # same manifest on the volume: nothing sent
    _, missing = launch.ensure_synced(link, DANCE, [{"world_id": "nope"}])
    assert "error" in missing["nope"]


def test_pods_that_never_start_a_worker_are_deleted_and_launching_halts_after_two(env):
    add_pod(env, "cpu")
    add_job(env, "g0000")
    w = make_watcher(env, target_gpu_pods=1, start_timeout_s=0.0, launch_cooldown_s=0)
    # fake pods whose worker never writes a heartbeat (bootstrap failed): remove the heartbeat the fake writes at creation
    orig = env.be.launch_gpu

    def broken(*a, **k):
        res = orig(*a, **k)
        for r in res:
            if r.get("pod_id"):
                (env.vol / "pods" / r["pod_id"] / "heartbeat").unlink()
                (env.vol / "pods" / r["pod_id"] / "events.log").write_text("1 NO WORKER: env=no code=yes\n")
        return res
    env.be.launch_gpu = broken
    for _ in range(14):
        cycles(w)
    launches = [x for x in env.be.calls if x[0] == "launch_gpu"]
    deleted = [x for x in env.be.calls if x[0] == "delete" and x[2] == "worker never started"]
    assert len(launches) == 2 and len(deleted) == 2                         # two strikes, then no third pod
    ev = [e["msg"] for e in env.state.load()["events"]]
    assert any("never started its worker" in m and "NO WORKER" in m for m in ev) and any("stops creating pods" in m for m in ev)


def test_storyline_gateway_to_gpu_pods_to_final_stitch_to_result_with_real_worker_loops(env):
    """The whole autonomous life of a run, no helper plays any role: no pod exists, three GPU chunks and the final job are queued. The
    watcher starts a gateway to see the volume, launches GPU pods (workers claim jobs on their own), pulls every chunk as it lands,
    starts one CPU pod for the final job when the last chunk is done, pulls that, deletes every pod and writes result.json."""
    env.be.auto_workers = True
    ids = ["g0000", "g0001", "g0002"]
    for i, jid in enumerate(ids):
        add_job(env, jid, payload=1000 * (i + 1))
    add_job(env, "final", chunk=False, after=ids, needs_gpu=False, chunks=ids, what="stitch_analyse", take="t_final", payload=77)
    w = make_watcher(env, target_gpu_pods=2, idle_grace_s=0.5, launch_gap_s=0.0)
    t = threading.Thread(target=w.run, daemon=True)
    t.start()
    try:
        t.join(180)
    finally:
        watch.stop_path(DANCE).write_text("stop")
        t.join(20)
        env.be.pods.clear()                                                  # lets the fake worker threads end
    assert w.finished, [e["msg"] for e in env.state.load()["events"][-15:]]
    st = env.state.load()
    assert all(st["chunks"][c]["status"] == "stitched" for c in ids) and st["jobs"]["final"]["status"] == "pulled"
    res = json.loads(watch.result_path(DANCE).read_text())
    assert res["status"] == "complete" and all(p["state"] == "deleted" for p in res["pods"])
    roles = sorted(p["role"] for p in st["pods"].values())
    assert roles.count("gateway") == 1 and roles.count("cpu") == 1 and 1 <= roles.count("gpu") <= 2
    for i, jid in enumerate(ids):
        assert (env.ws / "dancecap" / "work" / f"t_{jid}" / "smplx" / "lead.npz").stat().st_size == 1000 * (i + 1)
    assert (env.ws / "dancecap" / "work" / "t_final" / "smplx" / "lead.npz").stat().st_size == 77
    deleted = [x[1] for x in env.be.calls if x[0] == "delete"]
    assert sorted(deleted) == sorted(p["id"] for p in st["pods"].values())     # every pod this farm created, nothing else
