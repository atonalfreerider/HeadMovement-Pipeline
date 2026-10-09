"""Hardening of the farm after the 2026-10-08 reviews (survivability, world consistency): every defect the reviewers reproduced on fake
pods is reproduced here and must stay fixed. No network, no Runpod, no billable pod.

    ..\\.venv-nopo4d\\Scripts\\python.exe -m pytest tests/test_farm_hardening.py -q      (from dancecap/)

remote.py   a failed `pod delete` is never reported as success; the ledger survives concurrent writers; phantom rows stop accruing;
            reconcile closes rows of pods Runpod no longer lists (two looks); `remote cleanup` is a dry run
watcher     one unreadable heartbeat does not delete a running pod; a pod that ever ran is never "stuck"; pod ownership is by the whole
            dance name; a delete Runpod did not confirm keeps the pod in state.json and is retried; a pod still listed after it was
            marked deleted is deleted again; the same error on two pods is not re-queued again; a result corrupt on the volume is moved
            aside and re-run; a hung step with a live heartbeat is stalled; extensions share the budget
worker      markers of a different plan are ignored; a stolen claim is never deleted by its old owner; a stale-claim takeover that moved a
            fresh lock puts it back; a worker that lost its claim aborts without writing a result; max_minutes is enforced
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import _farm_watch_fixtures as F  # noqa: E402

from dancecap import remote  # noqa: E402
from dancecap.farm import pull, watch, worker  # noqa: E402
from dancecap.farm.state import State  # noqa: E402

from _farm_watch_env import DANCE, add_job, add_pod, cycles, env, make_watcher, status_of  # noqa: E402,F401

needs_bash = pytest.mark.skipif(F.BASH is None, reason="needs bash to play the pod")


# ------------------------------------------------------------------------------------------------ remote.py: ledger and delete


class FakeRunpod:
    """`runpodctl` as `remote._ctl_rc` sees it: pod list / pod delete with switchable failure modes."""

    def __init__(self, ids=()):
        self.pods = {i: {"id": i, "desiredStatus": "RUNNING"} for i in ids}
        self.delete_rc = 0            # 1: the call is rejected (401 JSON error on stdout, exit 1)
        self.delete_effective = True  # False: the call "succeeds" but the pod stays
        self.list_ok = True
        self.calls = []

    def __call__(self, *args, timeout=900):
        self.calls.append(args)
        if args[:2] == ("pod", "list"):
            return (0, list(self.pods.values())) if self.list_ok else (1, {"error": "boom", "status": 503})
        if args[:2] == ("pod", "delete"):
            if self.delete_rc:
                return 1, {"error": "failed to delete pod", "status": 401}
            if self.delete_effective:
                self.pods.pop(args[2], None)
            return 0, {"deleted": args[2]}
        return 1, {"error": "unexpected"}


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(remote, "LEDGER", tmp_path / "runpod" / "ledger.json")
    monkeypatch.setattr(remote, "DELETE_WAITS", (0.0, 0.0, 0.0))
    monkeypatch.setattr(remote, "DELETE_CONFIRM_WAIT", 0.0)
    rp = FakeRunpod()
    monkeypatch.setattr(remote, "_ctl_rc", rp)
    return rp


def open_row(pid="p1", price=1.0, minutes=60, age_min=10):
    remote._record(pid, "n", "gpu", "SECURE", price, minutes)
    rows = remote.ledger()
    for r in rows:
        if r["pod_id"] == pid:
            r["created_ts"] = time.time() - age_min * 60
    remote._save_ledger(rows)


def row(pid="p1"):
    return next(r for r in remote.ledger() if r["pod_id"] == pid)


def test_a_confirmed_delete_closes_the_row(ledger):
    ledger.pods["p1"] = {"id": "p1", "desiredStatus": "RUNNING"}
    open_row()
    remote._delete("p1", "done")
    assert row()["deleted_ts"] and row()["cost"] > 0 and "p1" not in ledger.pods


def test_a_rejected_delete_keeps_the_row_open_and_raises(ledger):
    """The review's D6: runpodctl printed {"error":..., "status":401} and exited 1; the old code closed the row and the pod billed on."""
    ledger.pods["p1"] = {"id": "p1", "desiredStatus": "RUNNING"}
    ledger.delete_rc = 1
    open_row()
    with pytest.raises(remote.PodDeleteError):
        remote._delete("p1", "done")
    r = row()
    assert not r.get("deleted_ts") and "401" in r["delete_error"] and r["delete_failed_ts"]
    assert remote.spent() > 0                                   # still accruing: the budget gate sees the pod
    assert len([c for c in ledger.calls if c[:2] == ("pod", "delete")]) == len(remote.DELETE_WAITS) + 1     # retried with backoff


def test_a_delete_that_does_not_take_is_retried_until_the_pod_is_gone(ledger):
    ledger.pods["p1"] = {"id": "p1", "desiredStatus": "RUNNING"}
    ledger.delete_effective = False
    open_row()
    orig = ledger.__call__

    def flip(*a, **k):
        if a[:2] == ("pod", "delete") and sum(1 for c in ledger.calls if c[:2] == ("pod", "delete")) >= 1:
            ledger.delete_effective = True                      # the second call works
        return orig(*a, **k)
    remote._ctl_rc = flip
    remote._delete("p1", "done")
    assert row()["deleted_ts"] and "p1" not in ledger.pods


def test_deleting_a_pod_that_is_already_gone_is_success(ledger):
    open_row()                                                  # the pod was never listed: nothing to delete is a clean outcome
    remote._delete("p1", "done")
    assert row()["deleted_ts"]


def test_an_accepted_delete_with_an_unreadable_pod_list_is_marked_unconfirmed(ledger):
    ledger.pods["p1"] = {"id": "p1", "desiredStatus": "RUNNING"}
    ledger.list_ok = False
    open_row()
    remote._delete("p1", "done")
    r = row()
    assert r["deleted_ts"] and r["delete_unconfirmed"] is True


def test_ledger_is_not_corrupted_by_concurrent_processes(tmp_path):
    """The review's stress: 4 writers x 40 _record calls lost 76 of 160 rows and crashed two writers. Atomic replace + a cross-process lock."""
    led = tmp_path / "runpod" / "ledger.json"
    code = (
        "import sys, pathlib; from dancecap import remote; remote.LEDGER = pathlib.Path(sys.argv[1]); "
        "k = sys.argv[2]\n"
        "for i in range(25): remote._record(f'{k}-{i}', 'n', 'g', 'SECURE', 1.0, 10)\n")
    root = Path(__file__).resolve().parents[1]
    procs = [subprocess.Popen([sys.executable, "-c", code, str(led), f"w{n}"], cwd=str(root), stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE) for n in range(4)]
    for p in procs:
        out, err = p.communicate(timeout=240)
        assert p.returncode == 0, err.decode()[-600:]
    rows = json.loads(led.read_text())
    assert len(rows) == 100 and len({r["pod_id"] for r in rows}) == 100
    assert not list(led.parent.glob("*.tmp")) and not (led.parent / "ledger.rw.lock").exists()


def test_a_phantom_open_row_stops_accruing_at_its_kill_switch(ledger):
    """resync-full: $22.92 at 16.5 h for a pod whose closing write was lost, shrinking the budget by $1.5 an hour for ever."""
    open_row("ghost", price=1.5, minutes=75, age_min=16 * 60)
    assert remote.spent() == pytest.approx(1.5 * (75 + remote.OPEN_ROW_GRACE_MIN) / 60, abs=0.01)
    open_row("live", price=1.0, minutes=75, age_min=30)         # inside its window: accrues normally
    assert remote.spent() == pytest.approx(1.5 * 85 / 60 + 0.5, abs=0.02)
    rows = remote.ledger()
    for r in rows:
        if r["pod_id"] == "ghost":
            r["alive_ts"] = time.time()                         # Runpod still lists it: it keeps accruing
    remote._save_ledger(rows)
    assert remote.spent() > 20


def test_reconcile_closes_rows_of_absent_pods_only_after_two_looks(ledger):
    ledger.pods["alive"] = {"id": "alive", "desiredStatus": "RUNNING"}
    open_row("alive", age_min=600, minutes=30)
    open_row("ghost", age_min=600, minutes=30)
    open_row("young", age_min=1, minutes=30)                    # too young to judge: Runpod may not list it yet
    r1 = remote.reconcile_ledger(confirm_s=0.2)
    assert r1["flagged"] == ["ghost"] and r1["closed"] == [] and "alive" in r1["alive"]
    time.sleep(0.3)
    r2 = remote.reconcile_ledger(confirm_s=0.2)
    assert r2["closed"] == ["ghost"]
    g = row("ghost")
    assert g["deleted_ts"] and g["cost"] == pytest.approx(0.5, abs=0.01)    # capped by its 30-min kill switch, not 10 h
    assert not row("alive").get("deleted_ts") and not row("young").get("deleted_ts")
    ledger.list_ok = False
    assert remote.reconcile_ledger()["error"]                   # an unreadable list closes nothing


def test_cleanup_orphans_is_a_dry_run_unless_told_otherwise(monkeypatch):
    monkeypatch.setattr(remote, "_ctl", lambda *a, **k: [{"id": "a", "name": "dancecap-fit-b"}, {"id": "b", "name": "other"}])
    gone = []
    monkeypatch.setattr(remote, "_delete", lambda pid, reason: gone.append(pid))
    assert remote.cleanup_orphans() == ["a"] and gone == []
    assert remote.cleanup_orphans(delete=True, only=["zzz"]) == [] and gone == []
    assert remote.cleanup_orphans(delete=True) == ["a"] and gone == ["a"]


# ------------------------------------------------------------------------------------------------ the watcher


def test_one_unreadable_heartbeat_does_not_destroy_a_running_pod(env):
    """The review's D5: an EIO on pods/<id>/heartbeat in the last minutes of a job turned the pod `stuck` and it was deleted."""
    pod = add_pod(env)
    spec = add_job(env, "g0000")
    F.start_job(env.be, pod, spec)
    w = make_watcher(env, start_timeout_s=0.0)
    cycles(w)
    assert env.state.load()["pods"][pod]["state"] == "running" and env.state.load()["pods"][pod]["hb_seen"] is True
    hb = env.vol / "pods" / pod / "heartbeat"
    hb.unlink()
    hb.mkdir()                                                  # exists but cannot be read as a file: the snapshot reports heartbeat_err
    cycles(w, 4)
    assert env.state.load()["pods"][pod]["state"] == "running"
    assert not [x for x in env.be.calls if x[0] == "delete"]
    assert status_of(env, "g0000") == "running"


def test_a_pod_that_ran_a_worker_is_never_stuck_even_without_a_heartbeat_file(env):
    pod = add_pod(env)
    spec = add_job(env, "g0000")
    F.start_job(env.be, pod, spec)
    w = make_watcher(env, start_timeout_s=0.0)
    cycles(w)
    (env.vol / "pods" / pod / "heartbeat").unlink()
    cycles(w, 5)
    assert env.state.load()["pods"][pod]["state"] == "running"
    assert not [x for x in env.be.calls if x[0] == "delete"]


def test_a_never_started_pod_needs_three_silent_snapshots_before_it_is_stuck(env):
    pod = add_pod(env)
    (env.vol / "pods" / pod / "heartbeat").unlink()             # the worker never started
    add_job(env, "g0000")
    w = make_watcher(env, start_timeout_s=0.0, auto_launch=False)
    cycles(w)
    assert env.state.load()["pods"][pod]["state"] == "starting"
    cycles(w)
    assert env.state.load()["pods"][pod]["state"] == "starting"
    cycles(w)
    assert [x for x in env.be.calls if x[0] == "delete" and x[2] == "worker never started"]


def test_dance_names_are_matched_whole_not_by_a_ten_character_prefix():
    assert watch.dance_tag("larissa_kadu_full") != watch.dance_tag("larissa_kadu_38") != watch.dance_tag("larissa_kadu")
    assert watch.pod_name_belongs("hm-farm-larissakadufull-1008-1530-1", "larissa_kadu_full")
    assert watch.pod_name_belongs("hm-farm-larissakadufull-cpu-1008-1530-1", "larissa_kadu_full")
    assert watch.pod_name_belongs("hm-farm-larissakadufull-gateway", "larissa_kadu_full")
    assert not watch.pod_name_belongs("hm-farm-larissakadufull-1008-1530-1", "larissa_kadu")
    assert not watch.pod_name_belongs("hm-farm-larissakadu38-1008-1530-1", "larissa_kadu_full")
    assert not watch.pod_name_belongs("hm-farm-t2-0101-1", "t") and watch.pod_name_belongs("hm-farm-t-0101-1", "t")
    assert not watch.pod_name_belongs("hm-farm-farmdrill-0101-1", "farm")


def test_pods_of_another_dance_or_owner_are_never_adopted_or_deleted(env):
    mine = add_pod(env, record=False)
    env.be.pods[mine]["name"] = f"hm-farm-{watch.dance_tag(DANCE)}-0101-1"
    other_dance = add_pod(env, record=False)
    env.be.pods[other_dance]["name"] = f"hm-farm-{watch.dance_tag(DANCE)}2-0101-1"           # dance "t2" next to dance "t"
    foreign_owner = add_pod(env, record=False)
    env.be.pods[foreign_owner]["name"] = f"hm-farm-{watch.dance_tag(DANCE)}-0101-2"
    env.be.pods[foreign_owner]["owner"] = "farm:someone-else"
    add_job(env, "g0000")
    w = make_watcher(env, auto_launch=False)
    cycles(w)
    pods = env.state.load()["pods"]
    assert mine in pods and other_dance not in pods and foreign_owner not in pods
    w.finish("test")
    deleted = {x[1] for x in env.be.calls if x[0] == "delete"}
    assert deleted == {mine}                                    # finish() touches only what this watcher manages
    assert other_dance in env.be.pods and foreign_owner in env.be.pods
    assert any("not this watcher's" in e["msg"] for e in env.state.load()["events"])


def test_a_delete_runpod_did_not_confirm_keeps_the_pod_and_is_retried_later(env):
    pod = add_pod(env)
    add_job(env, "g0000", status="pulled")
    env.state.set_job("g0000", status="pulled")
    add_job(env, "g0001")                                        # still queued: the watcher keeps running (finish() would retry at once)
    w = make_watcher(env, idle_grace_s=0.0, auto_launch=False)
    real = env.be.delete
    attempts = []

    def failing(pid, reason):
        attempts.append(pid)
        raise remote.PodDeleteError(f"pod {pid}: delete not confirmed: rc=1")
    env.be.delete = failing
    cycles(w)
    assert attempts == [pod] and env.state.load()["pods"][pod]["state"] != "deleted"
    assert any("NOT confirmed" in e["msg"] for e in env.state.load()["events"])
    cycles(w)
    assert attempts == [pod]                                    # backing off: no hammering every cycle
    w.delete_fail[pod] = (1, 0.0)
    env.be.delete = real
    cycles(w)
    assert env.state.load()["pods"][pod]["state"] == "deleted" and pod not in env.be.pods


def test_a_pod_runpod_still_lists_after_it_was_marked_deleted_is_deleted_again(env):
    pod = add_pod(env)
    add_job(env, "g0000", status="pulled")
    env.state.set_job("g0000", status="pulled")
    watch.mark_pod_deleted(env.state, pod, "test", 0.0)         # state.json says gone, Runpod still lists it
    w = make_watcher(env, auto_launch=False)
    cycles(w, 3)
    assert ("delete", pod, "re-delete: still listed by Runpod after it was marked deleted") in env.be.calls
    assert pod not in env.be.pods


def test_the_same_error_on_two_pods_fails_the_job_instead_of_requeueing_again(env):
    """Deterministic failures recycled a pod per re-queue (up to 6 attempts per chunk)."""
    p1, p2 = add_pod(env), add_pod(env)
    spec = add_job(env, "g0000", steps=[{"name": "make", "cmd": F.sh("import sys; sys.exit(3)"), "timeout_min": 1}], retries=1)
    w = make_watcher(env, max_requeues=5, auto_launch=False)
    assert F.finish_job(env.be, p1, spec) == "failed"
    cycles(w)
    assert status_of(env, "g0000") == "queued" and env.state.load()["chunks"]["g0000"]["requeues"] == 1
    assert F.finish_job(env.be, p2, spec) == "failed"
    cycles(w)
    rec = env.state.load()["chunks"]["g0000"]
    assert rec["status"] == "failed" and "deterministic" in rec["error"] and rec["requeues"] == 1


@needs_bash
def test_a_result_corrupt_on_the_volume_is_moved_aside_and_the_job_rerun(env):
    pod = add_pod(env)
    spec = add_job(env, "g0000", payload=4000)
    add_job(env, "g0001", payload=b"keeps the watcher alive")
    assert F.finish_job(env.be, pod, spec) == "ok"
    tar = env.vol / "chunks" / "g0000" / "out" / "result.tar"
    raw = bytearray(tar.read_bytes())
    raw[len(raw) // 2] ^= 0xFF                                  # a bit flip on the volume: every download mismatches the recorded sha256
    tar.write_bytes(bytes(raw))
    w = make_watcher(env, max_requeues=2)
    for _ in range(3):
        cycles(w)
        rec = env.state.load()["chunks"]["g0000"]
        State(DANCE).set_chunk("g0000", pull={**(rec.get("pull") or {}), "next_at": 0})      # skip the retry back-off
    rec = env.state.load()["chunks"]["g0000"]
    assert rec["status"] == "queued" and rec["requeues"] == 1
    assert list((env.vol / "chunks" / "g0000").glob("out.corrupt.*")) and not (env.vol / "chunks" / "g0000" / "done.json").exists()
    assert json.loads((env.vol / "chunks" / "g0000" / "attempts.json").read_text()) == {"n": 0}
    assert F.finish_job(env.be, pod, spec) == "ok"             # re-run for real, packs a good result
    cycles(w, 2)
    assert status_of(env, "g0000") in ("pulled", "stitched")


def test_a_hung_gpu_step_with_a_live_heartbeat_is_stalled_and_not_extended(env):
    """The review's D8: a deadlocked fit keeps a fresh heartbeat (a thread writes it) and was extended for ever."""
    pod = add_pod(env, minutes=5, hard_minutes=240)
    spec = add_job(env, "g0000")
    F.start_job(env.be, pod, spec, step="fit")
    hbp = env.vol / "chunks" / "g0000" / "heartbeat"

    def beat(util):
        hb = json.loads(hbp.read_text())
        hb.update(t=time.time(), gpu={"name": "FAKE GPU", "util": util})
        hbp.write_text(json.dumps(hb))
    w = make_watcher(env, stagnant_min=0.004, extend_below_min=1e6)       # 0.24 s of idle GPU = stalled; always eligible to be extended
    beat(0)
    cycles(w)                                                   # first sighting of an idle GPU is not yet a streak: extended as usual
    n_ext = len([x for x in env.be.calls if x[0] == "extend"])
    assert n_ext == 1
    time.sleep(0.15)
    beat(0)
    cycles(w)                                                   # over half the limit: no extension any more
    assert len([x for x in env.be.calls if x[0] == "extend"]) == n_ext
    time.sleep(0.2)
    beat(0)
    cycles(w)
    assert any("stalled" in e["msg"] and "idle" in e["msg"] for e in env.state.load()["events"])
    assert status_of(env, "g0000") == "queued"                  # claim released, pod recycled
    # a healthy fit (high utilisation) never builds a streak
    spec2 = add_job(env, "g0001")
    pod2 = add_pod(env)
    F.start_job(env.be, pod2, spec2, step="fit")
    for _ in range(3):
        time.sleep(0.1)
        cycles(w)
    assert status_of(env, "g0001") == "running"


def test_extensions_in_one_cycle_share_the_room_left(env):
    pods = [add_pod(env, minutes=5, hard_minutes=240) for _ in range(3)]
    for i, p in enumerate(pods):
        F.start_job(env.be, p, add_job(env, f"g000{i}"))
    env.be.cap_left = 0.6                                       # 30 min of one 0.69 $/h pod = 0.345: room for exactly one
    w = make_watcher(env, extend_below_min=25, extend_minutes=30)
    cycles(w)
    assert len([x for x in env.be.calls if x[0] == "extend"]) == 1
    assert any("after $" in e["msg"] for e in env.state.load()["events"])


def test_ensure_starts_a_watcher_when_there_is_work_and_none_runs(env, monkeypatch):
    started = []
    monkeypatch.setattr(watch, "start_detached", lambda dance, *a, **k: started.append(dance) or {"started": True})
    assert watch.ensure(DANCE)["reason"] == "nothing to watch" and not started
    add_job(env, "g0000")
    r = watch.ensure(DANCE)
    assert r["action"] == "started" and started == [DANCE]
    lock = watch.InstanceLock(watch.lock_path(DANCE))
    assert lock.acquire()
    try:
        assert watch.ensure(DANCE)["action"] == "none"          # running and not hung: left alone
    finally:
        lock.release()


def test_snapshot_reports_a_read_error_apart_from_a_missing_file(env):
    pod = add_pod(env)
    link = env.be.make_link({"id": pod})
    (env.vol / "pods" / pod / "heartbeat").unlink()
    s = pull.take_snapshot(link, [], [pod])
    assert s["pods"][pod]["heartbeat"] is None and s["pods"][pod]["heartbeat_err"] is False
    (env.vol / "pods" / pod / "heartbeat").mkdir()
    s = pull.take_snapshot(link, [], [pod])
    assert s["pods"][pod]["heartbeat"] is None and s["pods"][pod]["heartbeat_err"] is True


# ------------------------------------------------------------------------------------------------ the worker


def make_worker(tmp_path, pod="podA", **kw):
    vol = tmp_path / "vol"
    vol.mkdir(exist_ok=True)
    w = worker.Worker(vol, pod_id=pod, ws=tmp_path / f"ws_{pod}", work=tmp_path / f"work_{pod}", state=tmp_path / f"state_{pod}",
                      poll_s=0.05, **kw)
    w.gpu0 = {"name": "FAKE"}
    (w.ws / "dancecap").mkdir(parents=True, exist_ok=True)
    return w


def spec_writing(jid, text, take="t_x", **kw):
    code = f"import pathlib; d = pathlib.Path('work/{take}/smplx'); d.mkdir(parents=True, exist_ok=True); (d / 'lead.npz').write_text('{text}')"
    return {"schema": worker.SCHEMA, "id": jid, "dance": "t", "kind": "fit", "world_id": None, "inputs": [], "needs_gpu": True,
            "steps": [{"name": "make", "cmd": F.sh(code), "timeout_min": 1}], "outputs": [{"path": f"dancecap/work/{take}/smplx/"}],
            "max_minutes": 5, **kw}


def test_a_step_marker_of_a_different_plan_is_not_reused(tmp_path):
    """The review's D4: a re-planned job with a different command resumed with the OLD step result."""
    w = make_worker(tmp_path)
    a = spec_writing("j1", "OLD-GEOMETRY")
    assert w.try_claim(a) and w.run_job(a) == "ok"
    (w.vol / "chunks" / "j1" / "done.json").unlink()
    shutil.rmtree(w.vol / "chunks" / "j1" / "out")
    b = spec_writing("j1", "NEW-GEOMETRY")
    w.failed_here.clear()
    assert w.try_claim(b) and w.run_job(b) == "ok"
    out = w.ws / "dancecap" / "work" / "t_x" / "smplx" / "lead.npz"
    assert out.read_text() == "NEW-GEOMETRY"
    # the SAME plan (a resume after a failure) still skips the finished step
    (w.vol / "chunks" / "j1" / "done.json").unlink()
    w.failed_here.clear()
    assert w.try_claim(b) and w.run_job(b) == "ok"
    log = (w.vol / "chunks" / "j1" / "log" / "make.log").read_text()
    assert log.count("start make") == 2                         # run once for OLD, once for NEW, not a third time
    # a changed input sha256 or world invalidates it as well
    f1 = worker.step_fingerprint(b, b["steps"][0])
    assert f1 != worker.step_fingerprint({**b, "inputs": [{"path": "x", "sha256": "1" * 64}]}, b["steps"][0])
    assert f1 != worker.step_fingerprint({**b, "world_sha256": "2" * 64}, b["steps"][0])
    assert f1 == worker.step_fingerprint(dict(b), b["steps"][0])


def test_the_old_owner_of_a_taken_over_claim_does_not_delete_the_new_owners_lock(tmp_path):
    a, b = make_worker(tmp_path, "podA", stale_s=0.0), make_worker(tmp_path, "podB", stale_s=0.0)
    spec = spec_writing("j1", "x")
    assert a.try_claim(spec)
    time.sleep(0.05)
    assert b.try_claim(spec)                                    # A's claim is stale: B takes it over
    a.release("j1")                                             # A finishes late and "releases" - must not touch B's lock
    lock = a.vol / "chunks" / "j1" / "claim.lock"
    assert lock.is_dir() and json.loads((lock / "owner.json").read_text())["pod_id"] == "podB"
    c = make_worker(tmp_path, "podC", stale_s=1e9)
    assert not c.try_claim(spec)                                # a third pod cannot claim the running job
    b.release("j1")
    assert not lock.exists()


def test_a_takeover_that_moved_a_fresh_lock_puts_it_back(tmp_path, monkeypatch):
    """try_claim's check-then-rename gap: B decided the claim was stale, A (a faster taker) made it fresh, B's rename moved A's lock."""
    a = make_worker(tmp_path, "podA", stale_s=1e9)
    b = make_worker(tmp_path, "podB", stale_s=1e9)
    spec = spec_writing("j1", "x")
    assert a.try_claim(spec)
    old = json.loads((a.vol / "chunks" / "j1" / "claim.lock" / "owner.json").read_text())
    monkeypatch.setattr(b, "_stale", lambda cd: True)           # B judged the (old) claim stale ...
    # ... but between its look and its rename a third worker replaced the lock: simulate by making B's snapshot of the owner outdated
    real_read = worker.read_json
    calls = {"n": 0}

    def outdated(path, default=None):
        d = real_read(path, default)
        if str(path).endswith("owner.json") and calls["n"] == 0:
            calls["n"] += 1
            return {**old, "t": old["t"] - 999, "token": "stale-token"}      # what B saw when it decided
        return d
    monkeypatch.setattr(worker, "read_json", outdated)
    assert not b.try_claim(spec)
    monkeypatch.setattr(worker, "read_json", real_read)
    lock = a.vol / "chunks" / "j1" / "claim.lock"
    assert lock.is_dir() and json.loads((lock / "owner.json").read_text())["token"] == old["token"]      # A's live lock is back
    assert a.owns_claim("j1") is True


def test_a_worker_that_lost_its_claim_aborts_and_writes_no_result(tmp_path):
    w = make_worker(tmp_path, "podA")
    slow = {"schema": worker.SCHEMA, "id": "j1", "dance": "t", "kind": "fit", "world_id": None, "inputs": [], "needs_gpu": True,
            "steps": [{"name": "wait", "cmd": F.sh("import time; time.sleep(60)"), "timeout_min": 2}], "outputs": [], "max_minutes": 5}
    assert w.try_claim(slow)
    import threading

    def steal():
        time.sleep(1.0)
        lock = w.vol / "chunks" / "j1" / "claim.lock"
        (lock / "owner.json").write_text(json.dumps({"pod_id": "podB", "t": time.time(), "token": "other"}))
    threading.Thread(target=steal, daemon=True).start()
    worker.HEARTBEAT_S, old = 0.3, worker.HEARTBEAT_S
    try:
        t0 = time.time()
        assert w.run_job(slow) == "lost"
        assert time.time() - t0 < 30
    finally:
        worker.HEARTBEAT_S = old
    cd = w.vol / "chunks" / "j1"
    assert not (cd / "done.json").exists() and not (cd / "failed.json").exists() and not (cd / "out").exists()
    assert json.loads((cd / "claim.lock" / "owner.json").read_text())["pod_id"] == "podB"      # the new owner's claim untouched


def test_the_job_wall_clock_limit_stops_a_hung_step(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "JOB_LIMIT_SLACK_MIN", 0.0)
    w = make_worker(tmp_path)
    hung = {"schema": worker.SCHEMA, "id": "j1", "dance": "t", "kind": "fit", "world_id": None, "inputs": [], "needs_gpu": True,
            "steps": [{"name": "hang", "cmd": F.sh("import time; time.sleep(120)"), "timeout_min": 30}], "outputs": [],
            "max_minutes": 0.02, "retries": 1}                  # 0.02 min x 1.25 = 1.5 s, while the step's own timeout is 30 min
    assert w.try_claim(hung)
    t0 = time.time()
    assert w.run_job(hung) == "failed"
    assert time.time() - t0 < 60
    err = json.loads((w.vol / "chunks" / "j1" / "failed.json").read_text())["error"]
    assert "max_minutes" in err
