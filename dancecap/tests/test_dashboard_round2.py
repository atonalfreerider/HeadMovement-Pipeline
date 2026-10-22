"""Fix round 2 (2026-10-22), dashboard group: the analysis map fills by measured progress (not elapsed time), reused steps,
ended sessions without a stale upload, sessions without a pod, frame counters and slow-GPU / overrun / stall alerts, the
strip's choice of runs, every recent session, farm-run snapshots, lease pruning, the stale-ledger reconcile, `monitor
--launch --replace`, the bootstrap-retry Gantt, the no-history ETA prior, and the Farm page's runner analyses.
No pod, no Runpod call, no server process, no Edge: temp package roots and fakes only.

    ..\\.venv-nopo4d\\Scripts\\python.exe -m pytest tests/test_dashboard_round2.py -q        (from dancecap/ or the workspace)
"""
import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _pipeline_fixtures import FakePoller, make_root  # noqa: E402
from dancecap import progress_eta as E  # noqa: E402
from dancecap.pipeline import activity, analyze as A, farm_view, monitor, paths, snapshots  # noqa: E402

BASE = "http://127.0.0.1:8790"


@pytest.fixture()
def root(tmp_path):
    r = make_root(tmp_path)
    yield r
    paths.set_root(None)


def _doc(run: str, take: str, steps: list[dict], status: str = "planned") -> dict:
    for s in steps:
        s.setdefault("status", "pending")
    return {"schema": A.SCHEMA, "run": run, "take": take, "created": time.time() - 600, "updated": time.time(),
            "status": status, "input": {"path": "x", "route": "mono", "kind": "single", "videos": []},
            "facts": {"n": 1, "duration_s": 10, "size_mb": 1, "fps": 30, "frames": 300}, "steps": steps,
            "plan": A.totals(steps), "cost_so_far_usd": 0.0}


def _session(root: Path, sid: str, state: dict, result: bool = True) -> Path:
    d = root / "work" / "runpod" / "sessions" / sid
    d.mkdir(parents=True, exist_ok=True)
    (d / "state.json").write_text(json.dumps(state))
    (d / "runner.log").write_text("[pod-runner 10:00:00] bundle_scan ...\n")
    if result:
        time.sleep(0.01)
        (d / "result.json").write_text("{}")
    return d


def _ledger(root: Path, rows: list[dict]) -> None:
    led = root / "work" / "runpod" / "ledger.json"
    led.parent.mkdir(parents=True, exist_ok=True)
    led.write_text(json.dumps(rows))


# ------------------------------------------------------------------------------------------------ the analysis map

def test_squares_fill_by_measured_progress_and_draw_elapsed_time_apart():
    now = time.time()
    st = {"id": "fits", "where": "pod-gpu", "lanes": 1, "p50_min": 24, "p90_min": 40, "status": "running",
          "started": now - 22 * 60}
    # no progress figure: 22 of 24 min passed is NOT "11 of 12 done" (the 2026-10-21 review): striped time squares
    k = snapshots.step_blocks(st, now)
    assert len(k) == 12 and set(k) == {"time", "left"} and k.count("time") == 11
    # a measured figure (a live pod session's phases, a byte / frame counter) fills "run" squares
    k = snapshots.step_blocks({**st, "progress_frac": {"frac": 0.25, "basis": "pod session phases"}}, now)
    assert k.count("run") == 3 and k.count("left") == 9 and "time" not in k
    k = snapshots.step_blocks({**st, "progress": {"done": 50, "total": 100, "unit": "bytes"}}, now)
    assert k.count("run") == 6
    # past the plan: every plan square elapsed, plus "over" squares for the overrun (49 min of ~23 in the trial)
    k = snapshots.step_blocks({**st, "p50_min": 23, "started": now - 49 * 60}, now)
    assert k.count("time") == 12 and k.count("over") >= 12 and "left" not in k
    # a reused step: one square, no pods; a step the run's end stopped
    assert snapshots.step_blocks({**st, "status": "done", "reused": True, "lanes": 6}, now) == ["reused"]
    assert set(snapshots.step_blocks({**st, "status": "stopped"}, now)) == {"stop"}
    legend = snapshots._legend()
    assert "done so far" not in legend and "time elapsed vs the plan" in legend and "reused" in legend


def test_reused_steps_cost_nothing_and_cancelled_runs_stop_their_steps(root):
    run = "20261021-145414-trialx"
    steps = [A._step("intake", "i", "local", [], 0.5, 1, executor="builtin", price="local"),
             A._step("camera", "c", "pod-gpu", ["intake"], 9, 15, lanes=4),
             A._step("people", "p", "pod-gpu", ["camera"], 6, 10, lanes=6)]
    d = _doc(run, "trialx", steps, status="cancelled")
    d["steps"][0].update(status="done", started=time.time() - 500, ended=time.time() - 490)
    # a run from before the `reused` flag: only its message says so
    d["steps"][1].update(status="done", started=time.time() - 480, ended=time.time() - 420,
                         message="REUSED, not computed: the parent video's single-camera solve")
    d["steps"][2].update(status="running", started=time.time() - 400)
    A.run_dir(run).mkdir(parents=True)
    A.save(d)
    card = activity.analysis_card(A.load(run), time.time())
    by = {s["id"]: s for s in card["steps"]}
    assert by["camera"]["reused"] is True and by["camera"]["cost_p50_usd"] == 0.0
    assert snapshots.step_blocks(by["camera"], time.time()) == ["reused"]
    assert snapshots.step_line(by["camera"], time.time()) == "reused, not computed"
    assert by["people"]["status"] == "stopped" and by["people"]["progress_frac"] is None
    assert activity.is_reused({"status": "done", "message": "take exists: reused"}) is False   # lower case: not the mark
    assert activity.is_reused({"reused": False, "message": "REUSED"}) is False                 # the flag wins


# ------------------------------------------------------------------------------------------------ sessions

def test_ended_sessions_show_where_the_upload_stopped_and_no_pod_errors(root):
    d = _session(root, "20261021-161832-batch", {"bundle": {"bytes": 146e6}, "pod": {"id": "p1"}, "timings": []})
    (d / "progress").mkdir()
    (d / "progress" / "upload.json").write_text(json.dumps(
        {"total": 145590923, "sent": 65011712, "t0": time.time() - 3600, "ts": time.time() - 3570, "streams": 4}))
    _ledger(root, [{"pod_id": "p1", "cost_per_hr": 0.89, "created_ts": time.time() - 3700,
                    "deleted_ts": time.time() - 3500, "cost": 0.02, "max_minutes": 30}])
    c = activity.session_card("20261021-161832-batch")
    up = c["upload"]
    assert c["state"] not in activity.LIVE and up["stopped"] is True and up["eta_s"] is None and not up["done"]
    assert "stopped at 65 of 146 MB" in snapshots.upload_text(up)
    # a session that never got a pod: its error, not "44.2 MB to upload"
    _session(root, "20261021-145758-batch", {
        "bundle": {"bytes": 44200683}, "timings": [], "jobs": {"s": {"status": "pending"}},
        "error": "RuntimeError: no GPU pod could be started: every option exceeds the session max $1.00",
        "gpu_choice": {"tried": [{"gpu": "NVIDIA A100-SXM4-80GB", "cloud": "SECURE", "price": 1.39, "need": 2.1,
                                  "result": "over the session max"}]}})
    c = activity.session_card("20261021-145758-batch")
    assert c["no_pod"] is True and c["upload"]["no_pod"] is True and "session max" in c["error"]
    assert c["choice"]["no_pod"] is True and c["choice"]["tried"][0]["result"] == "over the session max"
    page = snapshots._session_rows([c], time.time())
    assert "no pod started" in page and "every option exceeds the session max" in page and "44 MB" not in page


def test_frame_counters_flag_a_slow_gpu_and_runner_warnings_are_read(root):
    d = _session(root, "20261021-151209-batch", {"pod": {"id": "p2", "gpu": "NVIDIA A100-SXM4-80GB"}, "timings": [],
                                                 "warnings": [{"kind": "slow_gpu", "level": "warn", "msg": "x is slow"}]},
                 result=False)
    (d / "progress").mkdir()
    (d / "progress" / "frames.json").write_text(json.dumps({"v": 1, "gpu": "A100", "jobs": {"s/infer": {
        "job": "s", "cmd": "infer", "step": "sam3db/mask255", "unit": "frames", "done": 200, "total": 599,
        "rate_s": 2.0, "rate_unit": "person-frame", "eta_s": 800, "state": "running", "ts": time.time(),
        "slow_x": 5.63, "ref": {"rate_s": 0.355, "rate_unit": "person-frame", "gpu": "NVIDIA GeForce RTX 4090"}}}}))
    fr = activity.frames_info("20261021-151209-batch")
    assert fr[0]["slow"] is True and fr[0]["eta_s"] == 800 and fr[0]["key"] == "s/infer"
    assert "5.6x slower than on GeForce RTX 4090" in snapshots.frames_text(fr[0])
    assert activity.frames_info("20261021-151209-batch", ended=True)[0]["eta_s"] is None
    assert activity._state_warnings(json.loads((d / "state.json").read_text()))[0]["kind"] == "slow_gpu"


def test_alerts_collect_stalls_slow_gpus_overruns_failures_and_podless_sessions():
    now = time.time()
    card = {"run": "20261021-145414-t", "steps": [
        {"id": "fits", "status": "running", "started": now - 50 * 60, "p50_min": 23, "p90_min": 40},
        {"id": "post", "status": "failed", "message": "stitch failed"},
        {"id": "sam3d_body", "status": "pending"}],
        "sessions": [
            {"sid": "20261021-150409-batch", "state": "running", "analysis": {"step": "fits"},
             "warnings": ["no progress in fits/front (front) for 12 min at 3/9", "kill-switch margin at p90 is 9 min"],
             "frames": [{"key": "s/infer", "slow": True, "state": "running", "slow_x": 5.6, "rate_s": 2.0,
                         "rate_unit": "person-frame", "ref": {"rate_s": 0.355, "gpu": "NVIDIA GeForce RTX 4090"}}]},
            {"sid": "20261021-145758-batch", "state": "failed", "no_pod": True, "analysis": {"step": "sam3d_body"},
             "error": "every option exceeds the session max $1.00", "ended": now - 60}]}
    al = activity.analysis_alerts(card, now)
    ids = [a["id"].split(":")[0] for a in al]
    assert al[0]["level"] == "red" and al[0]["id"] == "failed:post"
    assert {"warn", "slow", "overrun", "nopod"} <= set(ids)
    assert not any("kill-switch margin" in a["message"] for a in al)       # routine margins stay in the session row
    assert any("past its plan's p90 of 40 min" in a["message"] for a in al)


# ------------------------------------------------------------------------------------------------ the strip, recent sessions

def test_the_strip_shows_the_most_recently_finished_runs_first():
    now = time.time()
    # the 2026-10-21 day: the multi-view trial was created after the mono trial and cancelled before it finished
    mono = {"run": "20261021-145414-mono", "status": "done", "created": now - 9000, "updated": now - 5000,
            "ended": now - 5000}
    mv = {"run": "20261021-155346-mv", "status": "cancelled", "created": now - 8000, "updated": now - 6000,
          "ended": now - 6000}
    old = {"run": "20261021-142636-jl", "status": "cancelled", "created": now - 12000, "updated": now - 11000}
    picked = activity.pick_runs([mv, mono, old], now)         # list_runs order: newest created first
    assert [d["run"] for d, _ in picked] == [mono["run"], mv["run"], old["run"]] and not any(a for _, a in picked)
    live = {"run": "20261021-170000-new", "status": "running", "created": now - 60, "updated": now - 5}
    picked = activity.pick_runs([live, mv, mono, old], now)
    assert picked[0] == (live, True) and len(picked) == activity.FINISHED_SHOWN


def test_recent_sessions_lists_every_session_of_the_last_day(root):
    for i in range(17):
        _session(root, f"20261021-10{i:02d}00-batch", {"pod": {"id": f"p{i}"}, "timings": []})
    _ledger(root, [{"pod_id": f"p{i}", "cost_per_hr": 0.5, "created_ts": time.time() - 3600, "deleted_ts": time.time(),
                    "cost": 0.1, "max_minutes": 30} for i in range(17)])
    import datetime as dt
    ts = dt.datetime.strptime("20261021-100000", "%Y%m%d-%H%M%S").timestamp()
    assert len(activity.recent_sessions(ts + 3600)) == 17               # it stopped at 12


def test_budget_keeps_ledger_rows_and_runpod_pods_apart():
    bud = {"open_pods": 7, "pods": [{"pod_id": f"x{i}", "phantom_cost": 0.1} for i in range(6)] + [{"pod_id": "live"}]}
    rp = {"connection": "ok", "pods": [{"id": "live", "managed": True}], "checked": 1.0}
    v = activity.ledger_vs_runpod(bud, rp)
    assert v["ledger_open"] == 7 and v["ledger_stale"] == 6 and v["runpod_listed"] == 1 and v["runpod_ours"] == 1
    assert activity.ledger_vs_runpod(bud, {"connection": "api_error"})["runpod_listed"] is None
    fb = farm_view._budget({"budget": bud, "connection": "ok", "pods": rp["pods"]}, {})
    assert fb["ledger_open"] == 7 and fb["ledger_stale"] == 6 and fb["runpod_listed"] == 1


# ------------------------------------------------------------------------------------------------ the farm page

def test_farm_page_knows_runner_analyses(root):
    take = "trialy"
    (root / "work" / take / "smplx_rf2_noslip").mkdir(parents=True)
    (root / "work" / take / "physics").mkdir()
    steps = [A._step("intake", "i", "local", [], 0.5, 1, executor="builtin", price="local")]
    done = _doc("20261021-145414-trialy", take, [dict(s) for s in steps], status="done")
    done["steps"][0].update(status="done", started=time.time() - 100, ended=time.time() - 90)
    later = _doc("20261021-172913-trialy", take, [dict(s) for s in steps], status="cancelled")
    for d in (done, later):
        A.run_dir(d["run"]).mkdir(parents=True)
        A.save(d)
    an = farm_view.analyses_by_take()[take]
    assert an["run"] == "20261021-145414-trialy" and an["done"] == an["total"] == 1   # the 12/12 run, not the later 4/12
    assert an["outputs"] == ["smplx_rf2_noslip", "physics"]
    ds = {d["id"]: d for d in farm_view.list_dances()}
    assert ds[take]["analysis"]["status"] == "done" and ds[take]["farm"] is False and ds[take]["complete_pct"] is None
    b = farm_view.build(take)
    assert b["analysis"]["run"] == an["run"] and not any(a["id"] == "no-coverage" for a in b["alerts"])
    # a dance with farm files stays the page's default over a runner-only take
    (root / "work" / "farmy" / "farm").mkdir(parents=True)
    (root / "work" / "farmy" / "farm" / "state.json").write_text(json.dumps({"dance": "farmy", "chunks": {}}))
    import os
    os.utime(root / "work" / "farmy" / "farm" / "state.json", (time.time() - 9e5, time.time() - 9e5))
    assert farm_view.default_dance() == "farmy"


# ------------------------------------------------------------------------------------------------ snapshots

def _bump(p: Path, st: dict) -> None:
    """Rewrite a farm state.json with a newer mtime (the snapper looks at changed files only)."""
    p.write_text(json.dumps(st))
    t = p.stat().st_mtime + 1 + _bump.n
    _bump.n += 1
    import os
    os.utime(p, (t, t))


_bump.n = 0


def _farm_state(root: Path, dance: str, events: list[dict], chunks: dict | None = None) -> Path:
    fd = root / "work" / dance / "farm"
    fd.mkdir(parents=True, exist_ok=True)
    p = fd / "state.json"
    p.write_text(json.dumps({"dance": dance, "events": events, "chunks": chunks or {}, "pods": {}}))
    return p


def test_farm_runs_take_snapshots_at_launch_milestones_and_end(root):
    t0 = time.time() - 3000
    sp = _farm_state(root, "jl", [{"t": t0 - 9000, "kind": "finish", "msg": "an older run"}])
    calls = []

    def req(run, event, label="", final=False, sid=None):
        calls.append((run, event, final))
        return True
    fs = monitor.FarmSnapper()
    assert fs.tick(t0, request=req) == [] and calls == []              # first sight: history is never re-snapshotted
    _bump(sp, {"dance": "jl", "chunks": {"c1": {"status": "running"}}, "events": [
        {"t": t0 - 9000, "kind": "finish", "msg": "an older run"}, {"t": t0 + 10, "kind": "plan", "msg": "2 chunks"},
        {"t": t0 + 20, "kind": "launch", "msg": "pod a up"}, {"t": t0 + 21, "kind": "pod", "msg": "noise"}]})
    fs.tick(t0 + 30, request=req)
    run = snapshots.farm_run_id("jl", t0 + 10)
    assert calls == [(run, "farm_start", False)]
    st = json.loads(sp.read_text())
    st["events"] += [{"t": t0 + 40, "kind": "pull", "msg": "c1 pulled"}]
    _bump(sp, st)
    fs.tick(t0 + 50, request=req)                                      # inside the 60 s gap: not yet
    assert len(calls) == 1
    st["events"] += [{"t": t0 + 200, "kind": "job", "msg": "c2: running -> done"}]
    _bump(sp, st)
    fs.tick(t0 + 210, request=req)
    assert calls[-1] == (run, "farm_chunk_pulled", False)               # the higher-ranked event of the batch
    fs.tick(t0 + 210 + monitor.FARM_HEARTBEAT_S + 1, request=req)         # 15 min with chunks running: a heartbeat
    assert calls[-1] == (run, "farm_heartbeat", False)
    st["events"] += [{"t": t0 + 2000, "kind": "finish", "msg": "finished: all jobs terminal"}]
    _bump(sp, st)
    fs.tick(t0 + 2010, request=req)
    assert calls[-1] == (run, "farm_end", True)
    assert json.loads((paths.monitor_dir() / monitor.FARM_SNAP_STATE).read_text())["jl"]["finished"] is True
    assert snapshots.FARM_RUN_RE.match(run).group("dance") == "jl"
    # an abandoned farm (launched 2 weeks ago, never finished, state untouched since) is not a run in flight: no heartbeats
    old = _farm_state(root, "lk", [{"t": t0 - 14 * 86400, "kind": "launch", "msg": "pod up"}], {"c": {"status": "queued"}})
    import os
    os.utime(old, (t0 - 14 * 86400, t0 - 14 * 86400))
    n = len(calls)
    fs.tick(t0 + 3000, request=req)
    fs.tick(t0 + 3000 + 2 * monitor.FARM_HEARTBEAT_S, request=req)
    assert len(calls) == n and fs.seen["lk"]["run"] is None


def test_farm_snapshot_page_and_analysis_snapshot_count(root, monkeypatch):
    _farm_state(root, "jl", [{"t": time.time(), "ts": "2026-10-22T10:00:00", "kind": "launch", "msg": "pod up " + "x" * 400}],
                {"c1": {"status": "running", "k0": 0, "k1": 899}})
    run = snapshots.farm_run_id("jl", time.time() - 60)
    calls = []

    def fake_png(hp, pp, height):
        calls.append(hp.read_text(encoding="utf-8"))
        pp.write_bytes(b"\x89PNG" + b"0" * 2000)
        return "edge", 5, None
    monkeypatch.setattr(snapshots, "render_png", fake_png)
    shot = snapshots.take(run, "farm_launch", "farm jl: pod up")
    assert shot["kind"] == "farm" and shot["chunks"] == 1 and "volume farm" in calls[0] and "x" * 400 not in calls[0]
    assert snapshots.clip("a " * 200, 50).endswith(" …") and len(snapshots.clip("word " * 100, 60)) <= 62
    # the analysis page's button counts every snapshot, not the 60 it lists
    arun = "20261021-120000-snapc"
    d = _doc(arun, "snapc", [A._step("intake", "i", "local", [], 0.5, 1, executor="builtin", price="local")])
    A.run_dir(arun).mkdir(parents=True)
    A.save(d)
    sd = snapshots.snap_dir(arun)
    sd.mkdir(parents=True)
    (sd / "index.json").write_text(json.dumps({"run": arun, "shots": [{"i": i, "event": "x"} for i in range(80)]}))
    det = activity.analysis_detail(arun)
    assert det["snapshots_n"] == 80 and len(det["snapshots"]) == 60


# ------------------------------------------------------------------------------------------------ monitor housekeeping

def test_stale_ledger_rows_are_reconciled_through_the_repo_path(root, monkeypatch):
    now = time.time()
    _ledger(root, [{"pod_id": "old1", "cost_per_hr": 1, "created_ts": now - 10 * 86400, "max_minutes": 150},
                   {"pod_id": "fresh", "cost_per_hr": 1, "created_ts": now - 600, "max_minutes": 60},
                   {"pod_id": "done", "cost_per_hr": 1, "created_ts": now - 9e5, "deleted_ts": now - 8e5, "cost": 1,
                    "max_minutes": 10}])
    rows = json.loads((root / "work" / "runpod" / "ledger.json").read_text())
    assert [r["pod_id"] for r in monitor.stale_ledger_rows(rows, now)] == ["old1"]
    from dancecap import remote
    looks = []

    def fake_reconcile(confirm_s=60.0, min_age_s=300.0):
        looks.append(min_age_s)
        return {"closed": [] if len(looks) == 1 else ["old1"], "flagged": ["old1"] if len(looks) == 1 else []}
    monkeypatch.setattr(remote, "reconcile_ledger", fake_reconcile)
    r = monitor.reconcile_stale(now, look_gap_s=0.0)
    assert r["closed"] == ["old1"] and len(looks) == 2 and looks[0] > 9 * 86400     # fresh rows are never judged
    _ledger(root, [])
    assert monitor.reconcile_stale(now) is None


def test_prune_leases_and_replace_stops_only_our_server(root, monkeypatch):
    f = monitor.write_lease("unit")
    dead = f.with_name("999998.json")
    dead.write_text(json.dumps({"pid": 999998, "create_time": 1.0, "reason": "gone"}))
    assert monitor.prune_leases() == 1 and not dead.exists() and f.exists()
    assert monitor.stop_running() is None                               # no server.json: nothing to stop
    alive = {"v": True}
    monkeypatch.setattr(monitor, "_running_server", lambda: {"pid": 4242, "port": 8799, "instance": "abc"})
    monkeypatch.setattr(monitor, "_post_quit", lambda port, timeout=5.0: alive.update(v=False) or True)
    from dancecap.pipeline import util
    monkeypatch.setattr(util, "pid_alive", lambda pid, ct=None: alive["v"])
    assert monitor.stop_running(timeout=2.0) == {"pid": 4242, "how": "quit"}


# ------------------------------------------------------------------------------------------------ progress_eta

def _hh(t):
    return time.strftime("%H:%M:%S", time.localtime(t))


def test_a_failed_first_bootstrap_stays_in_the_gantt(tmp_path):
    t0 = time.time() - 2000
    lines = [f"[pod-runner {_hh(t0)}] bundle_scan ...", f"[pod-runner {_hh(t0 + 1)}] bundle_scan done in 1.0s",
             f"[pod-runner {_hh(t0 + 60)}] bootstrap ...", f"[pod-runner {_hh(t0 + 920)}] bootstrap done in 860.0s",
             f"[pod-runner {_hh(t0 + 920)}] bootstrap rc=1: retrying once on this pod (finished features are skipped)",
             f"[pod-runner {_hh(t0 + 920)}] bootstrap_retry ...", f"[pod-runner {_hh(t0 + 968)}] bootstrap_retry done in 47.6s"]
    d = tmp_path / "20261021-132809-batch"
    d.mkdir()
    (d / "runner.log").write_text("\n".join(lines) + "\n")
    (d / "state.json").write_text(json.dumps({"bootstrap_rc": 1, "jobs": {}, "pod": {"id": "pz"}}))
    (d / "result.json").write_text("{}")
    snap = E.snapshot(d, now=t0 + 1000, rates={"keys": {}}, ledger_rows=[
        {"pod_id": "pz", "cost_per_hr": 0.79, "created_ts": t0 + 2, "deleted_ts": t0 + 990, "cost": 0.2}])
    ph = {p["name"]: p for p in snap["phases"]}
    assert ph["bootstrap_try1"]["status"] == "failed" and ph["bootstrap_try1"]["elapsed_s"] == pytest.approx(860, abs=2)
    assert ph["bootstrap"]["status"] == "failed" and ph["bootstrap"]["elapsed_s"] == pytest.approx(47.6, abs=2)
    names = [p["name"] for p in snap["phases"]]
    assert names.index("bootstrap_try1") == names.index("bootstrap") - 1


def _running_cmd_session(tmp_path, sid, rates_keys=None, frames=None, warnings=None):
    now = time.time()
    lines = [f"[pod-runner {_hh(now - 700)}] bundle_scan ...", f"[pod-runner {_hh(now - 699)}] bundle_scan done in 1.0s",
             f"[pod-runner {_hh(now - 600)}] j/run ..."]
    d = tmp_path / sid
    d.mkdir()
    (d / "runner.log").write_text("\n".join(lines) + "\n")
    st = {"jobs": {"j": {"status": "running", "spec": {"name": "j", "commands": [
        {"name": "run", "cmd": "python work/x/probe.py", "timeout_min": 60}]}}},
        "pod": {"id": "pr", "gpu": "NVIDIA GeForce RTX 4090", "cost_per_hr": 0.89, "started": now - 650,
                "killswitch": {"armed": True, "after_min": 90}}, "max_minutes": 90, "warnings": warnings or []}
    (d / "state.json").write_text(json.dumps(st))
    if frames:
        (d / "progress").mkdir()
        (d / "progress" / "frames.json").write_text(json.dumps({"v": 1, "jobs": {"j/run": frames}}))
    rows = [{"pod_id": "pr", "cost_per_hr": 0.89, "created_ts": now - 660, "max_minutes": 90}]
    return E.snapshot(d, now=now, rates={"keys": rates_keys or {}}, ledger_rows=rows)


def test_no_history_eta_is_a_prior_not_the_timeout_and_history_labels(tmp_path):
    snap = _running_cmd_session(tmp_path, "20261021-160000-batch", rates_keys={
        "cmd/a.b/s": {"n": 3, "p10": 100, "p50": 180, "p90": 300}, "cmd/c.d/s": {"n": 2, "p10": 50, "p50": 60, "p90": 90}})
    ph = {p["name"]: p for p in snap["phases"]}["j/run"]
    assert "timeout" not in (ph["detail"] or "") and "generic prior" in ph["detail"]
    assert ph["remaining"]["p50"] < 0.3 * 3600                     # the 60-min timeout said 18 min of 0.3 x timeout
    snap = _running_cmd_session(tmp_path, "20261021-160100-batch", rates_keys={
        "cmd/j.run/s": {"n": 4, "p10": 50, "p50": 60, "p90": 80, "sigma": 0.2}})
    ph = {p["name"]: p for p in snap["phases"]}["j/run"]
    assert ph["detail"].startswith("history n=4") and "history history" not in ph["detail"]
    # 600 s against a 60-s history (p90 80 s): far slower than before, a warning
    assert any(w.get("kind") == "slow_history" for w in snap["warnings"])


def test_frame_counter_gives_the_remaining_time_and_runner_warnings_surface(tmp_path):
    fr = {"job": "j", "cmd": "run", "src": "sam3db", "step": "sam3db/mask255", "unit": "frames", "done": 300,
          "total": 600, "rate_s": 0.7, "rate_unit": "person-frame", "eta_s": 210.0, "state": "running",
          "ts": time.time()}
    snap = _running_cmd_session(tmp_path, "20261021-160200-batch", frames=fr,
                                warnings=[{"kind": "gpu_choice", "level": "warn", "msg": "the session max ruled out a 4090"}])
    ph = {p["name"]: p for p in snap["phases"]}["j/run"]
    assert "frame counter" in ph["detail"] and ph["fraction"] == pytest.approx(0.5)
    assert 150 < ph["remaining"]["p50"] <= 210
    assert any(w.get("kind") == "gpu_choice" for w in snap["warnings"])
    # MAMMA's iterations stop early: no remaining time from them
    snap = _running_cmd_session(tmp_path, "20261021-160300-batch", frames={**fr, "src": "mamma_iter"})
    assert "frame counter" not in ({p["name"]: p for p in snap["phases"]}["j/run"]["detail"] or "")
    assert E.BOOT_SEED_MIN["ffmpeg"] and E.BOOT_SEED_MIN["open3d"]


# ------------------------------------------------------------------------------------------------ the API

def test_activity_and_analysis_api_carry_the_new_fields(root):
    from dancecap.pipeline.monitor import server
    run = "20261021-120002-apix"
    steps = [A._step("intake", "i", "local", [], 0.5, 1, executor="builtin", price="local"),
             A._step("people", "p", "pod-gpu", ["intake"], 10, 15, lanes=3)]
    d = _doc(run, "apix", steps, status="waiting")
    d["steps"][0].update(status="done", started=time.time() - 60, ended=time.time() - 30)
    d["steps"][1].update(status="failed", message="boom")
    A.run_dir(run).mkdir(parents=True)
    A.save(d)
    m = server.Monitor(port=8790, poller=FakePoller())
    app = server.create_app(m)
    c = app.test_client()
    c.get("/", base_url=BASE)
    a = c.get("/api/activity", base_url=BASE).get_json()
    assert a["monitor"]["instance"] == m.instance and "ledger_open" in a["budget"]
    assert any(x["id"] == "failed:people" for x in a["alerts"])
    det = c.get(f"/api/analyses/{run}", base_url=BASE).get_json()
    assert det["alerts"][0]["level"] == "red" and det["snapshots_n"] == 0 and det["monitor"]["instance"] == m.instance
    m.helper.shutdown(wait=True)


# ------------------------------------------------------------------------------------------------ integration (round 2 review)

def test_farm_snapshots_read_the_farms_real_chunk_events(root):
    """Review: the farm writes chunk status changes and failures as kind "chunk" (watch.py), which the snapper ignored; a
    pull failure was taken for a pulled chunk and a failed launch started a new run."""
    c = monitor.FarmSnapper.classify
    assert c({"kind": "chunk", "msg": "jlg0000: running -> done"}) == "chunk_done"
    assert c({"kind": "chunk", "msg": "jlshape: done -> pulled"}) == "chunk_done"
    assert c({"kind": "chunk", "msg": "jlsynth failed on 2pbo6cykjqqkdv (step synth rc=1); re-queued 1/2"}) == "chunk_failed"
    assert c({"kind": "chunk", "msg": "cag0000 FAILED for good: the same error on 2 different pods"}) == "chunk_failed"
    assert c({"kind": "chunk", "msg": "jlg0000: planned -> queued"}) is None
    assert c({"kind": "job", "msg": "jlan-physics: done -> pulled"}) == "chunk_done"
    assert c({"kind": "pull", "msg": "g1588: pull attempt 1 failed (PullError: ...); retry in 10 s"}) == "pull_failed"
    assert c({"kind": "pull", "msg": "g1588 pulled and verified: 31 files"}) == "chunk_pulled"
    assert c({"kind": "launch", "msg": "launch failed: no GPU"}) == "launch_failed"
    t0 = time.time() - 4000
    sp = _farm_state(root, "jl", [{"t": t0 - 9000, "kind": "finish", "msg": "an older run"}])
    calls = []

    def req(run, event, label="", final=False, sid=None):
        calls.append((run, event, label))
        return True
    fs = monitor.FarmSnapper()
    fs.tick(t0, request=req)
    evs = [{"t": t0 - 9000, "kind": "finish", "msg": "an older run"}, {"t": t0 + 10, "kind": "plan", "msg": "2 chunks"}]
    _bump(sp, {"dance": "jl", "chunks": {"c1": {"status": "running"}}, "events": evs})
    fs.tick(t0 + 20, request=req)
    run = snapshots.farm_run_id("jl", t0 + 10)
    assert calls[-1][:2] == (run, "farm_start")
    evs += [{"t": t0 + 100, "kind": "chunk", "msg": "jlg0000: running -> done"}]
    _bump(sp, {"dance": "jl", "chunks": {"c1": {"status": "running"}}, "events": evs})
    fs.tick(t0 + 110, request=req)
    assert calls[-1][:2] == (run, "farm_chunk_done")
    evs += [{"t": t0 + 300, "kind": "chunk", "msg": "jlg1322 failed on podx (step fit rc=1); re-queued 1/2"},
            {"t": t0 + 301, "kind": "pull", "msg": "jlg0000 pulled and verified: 25 files"}]
    _bump(sp, {"dance": "jl", "chunks": {"c1": {"status": "running"}}, "events": evs})
    fs.tick(t0 + 310, request=req)
    assert calls[-1][:2] == (run, "farm_chunk_failed")             # a failure outranks a pull in one batch
    n = len(calls)
    evs += [{"t": t0 + 500, "kind": "launch", "msg": "launch failed: no GPU in stock"}]
    _bump(sp, {"dance": "jl", "chunks": {"c1": {"status": "running"}}, "events": evs})
    fs.tick(t0 + 510, request=req)
    assert calls[n][:2] == (run, "farm_launch_failed") and fs.seen["jl"]["run"] == run   # never a new run


def test_an_unfinished_farm_record_of_an_old_server_is_rebaselined(root):
    t = time.time()
    sp = _farm_state(root, "lkfull", [{"t": t - 13 * 86400, "kind": "plan", "msg": "7 chunks"}], {"c": {"status": "queued"}})
    import os
    os.utime(sp, (t - 13 * 86400, t - 13 * 86400))
    (paths.monitor_dir()).mkdir(parents=True, exist_ok=True)
    (paths.monitor_dir() / monitor.FARM_SNAP_STATE).write_text(json.dumps(
        {"lkfull": {"t": t - 13 * 86400, "run": "20261008-130529-farm_lkfull", "finished": False, "hb": t - 3600}}))
    fs = monitor.FarmSnapper()
    assert fs._heartbeat_due("lkfull", t)
    fs.tick(t, request=lambda *a, **k: True)
    rec = json.loads((paths.monitor_dir() / monitor.FARM_SNAP_STATE).read_text())["lkfull"]
    assert rec["finished"] is True and rec["run"] is None and not fs._heartbeat_due("lkfull", t + 3600)


def test_a_parallel_session_still_bundling_is_not_a_session_without_a_pod(root, monkeypatch):
    """Review: every -p1..p(n-1) session of `remote run --parallel` is 'unknown' while it bundles (not the newest dir),
    and was drawn as a crit "no pod" with an amber alert."""
    from dancecap.pipeline import runpod_view
    _ledger(root, [])
    tag = {"run": "20261022-100000-synth", "step": "fits"}
    for sid in ("20261022-100000-batch-p1", "20261022-100000-batch-p2"):
        _session(root, sid, {"bundle": {"bytes": 1.8e9}, "timings": [], "analysis": tag}, result=False)
    monkeypatch.setattr(runpod_view, "runner_alive_of", lambda d: None)        # no runner.json: the old rule only
    c1 = activity.session_card("20261022-100000-batch-p1")
    assert c1["state"] == "unknown" and c1["no_pod"] is False and c1["preparing"] is True
    assert not (c1["upload"] or {}).get("no_pod")
    card = {"run": tag["run"], "steps": [{"id": "fits", "status": "running"}], "sessions": [c1]}
    assert not [a for a in activity.analysis_alerts(card, time.time()) if a["id"].startswith("nopod:")]
    monkeypatch.setattr(runpod_view, "runner_alive_of", lambda d: True)        # its runner lives: preparing
    runpod_view._sum_cache.clear()
    assert activity.session_card("20261022-100000-batch-p1")["state"] == "preparing"
    # the same session once the runner wrote that no GPU could be started: no pod, said once
    monkeypatch.setattr(runpod_view, "runner_alive_of", lambda d: False)
    _session(root, "20261022-100000-batch-p1", {"bundle": {"bytes": 1.8e9}, "timings": [], "analysis": tag,
                                                "error": "RuntimeError: no GPU pod could be started"}, result=True)
    runpod_view._sum_cache.clear()
    c1 = activity.session_card("20261022-100000-batch-p1")
    assert c1["no_pod"] is True and c1["upload"]["no_pod"] is True
    # a dry run: neither "no pod" nor an upload
    _session(root, "20261022-100500-batch", {"bundle": {"bytes": 5e6}, "timings": []}, result=True)
    runpod_view._sum_cache.clear()
    c = activity.session_card("20261022-100500-batch")
    assert c["state"] == "dry_run" and c["no_pod"] is False and c["upload"] is None


def test_one_slow_gpu_is_one_alert():
    now = time.time()
    msg = "trial_sam3d_body/infer: 2.000 s per person-frame on A100-SXM4-80GB, 5.6x the 0.355 s measured on RTX 4090"
    card = {"run": "20261021-145414-t", "steps": [{"id": "sam3d_body", "status": "running", "started": now - 60,
                                                   "p50_min": 10, "p90_min": 20}],
            "sessions": [{"sid": "20261021-151934-batch", "state": "running", "analysis": {"step": "sam3d_body"},
                          "warnings": [], "state_warnings": [{"kind": "slow_gpu", "level": "warn", "msg": msg}],
                          "frames": [{"key": "trial_sam3d_body/infer", "slow": True, "state": "running", "slow_x": 5.6,
                                      "rate_s": 2.0, "rate_unit": "person-frame",
                                      "ref": {"rate_s": 0.355, "gpu": "NVIDIA GeForce RTX 4090"}}]}]}
    al = activity.analysis_alerts(card, now)
    assert len([a for a in al if "slower" in a["message"] or "5.6x" in a["message"]]) == 1


def test_a_dry_run_session_has_nothing_running(tmp_path):
    """Review: GET /api/sessions/<dry run> said pod_start 'running' with 851 s left, an hour after it ended."""
    d = tmp_path / "20261021-174531-batch"
    (d / "progress").mkdir(parents=True)
    now = time.time()
    t = now - 3400
    (d / "runner.log").write_text(f"[pod-runner {_hh(t)}] bundle_scan ...\n[pod-runner {_hh(t)}] bundle_scan done in 0.3s\n"
                                  f"[pod-runner {_hh(t + 20)}] dry run: no pod\n")
    sid = d.name
    evs = [{"v": 1, "ts": t, "job": "_session", "step": "bundle_scan", "kind": "start", "sid": sid},
           {"v": 1, "ts": t + 0.3, "job": "_session", "step": "bundle_scan", "kind": "end", "extra": {"seconds": 0.3},
            "sid": sid},
           {"v": 1, "ts": t + 20, "job": "_session", "step": "session", "kind": "end", "msg": "session ended: ok",
            "sid": sid}]
    (d / "progress" / "runner.jsonl").write_text("".join(json.dumps(e) + "\n" for e in evs))
    (d / "state.json").write_text(json.dumps({"jobs": {"j": {"status": "pending", "spec": {"name": "j", "commands": [
        {"name": "run", "cmd": "true", "timeout_min": 10}]}}}, "max_minutes": 25,
        "timings": [{"step": "bundle_scan", "seconds": 0.3, "ok": True, "start": t}]}))
    (d / "result.json").write_text("{}")
    snap = E.snapshot(d, now=now, rates={"keys": {}}, ledger_rows=[])
    assert snap["state"] == "dry_run" and snap.get("eta") is None
    assert not [p["name"] for p in snap["phases"] if p.get("status") in ("running", "pending")]


def test_a_live_session_row_lists_the_runners_warning_once(root, monkeypatch):
    """Live probe 2026-10-21 19:51: the Farm strip's session row showed the runner's slow-GPU warning twice, once from
    state.json and once from the session snapshot, which runpod_view flattens to plain text (no `kind` to filter on)."""
    from dancecap.pipeline import runpod_view
    msg = "runner_probe/progress: 0.900 s per person-frame on RTX 4090, 2.5x the 0.355 s measured on RTX 4090"
    sid = "20261021-194737-batch"
    _session(root, sid, {"pod": {"id": "p9", "gpu": "NVIDIA GeForce RTX 4090"}, "timings": [],
                         "warnings": [{"kind": "slow_gpu", "level": "warn", "msg": msg}]}, result=False)
    monkeypatch.setattr(runpod_view, "session_summary", lambda s, ledger=None: {
        "sid": sid, "state": "running", "take": "runner_probe", "jobs": ["runner_probe"], "started": time.time() - 200})
    monkeypatch.setattr(runpod_view, "session_snapshot", lambda s: {
        "phases": [], "warnings": [msg, "no progress for 6 min (last: bootstrap)"]})
    c = activity.session_card(sid)
    assert c["warnings"] == ["no progress for 6 min (last: bootstrap)"]
    assert [w["msg"] for w in c["state_warnings"]] == [msg]
