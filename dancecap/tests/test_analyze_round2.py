"""2026-10-22 fix round, group "plan": honest plans, ETAs, elapsed and run cost; one driver per run, snapshots only at
progress points, the run's heartbeat; reused steps; one-video intake defaults; the run's pod limit (--max-cost).
Synthetic data only: temp package roots, fake clocks and fakes. No pod, no Runpod call, no snapshot process.

    ..\\.venv-nopo4d\\Scripts\\python.exe -m pytest tests/test_analyze_round2.py -q        (from dancecap/ or the workspace)
"""
import json
import os
import sys
import time
import tomllib
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _pipeline_fixtures import fake_probe, make_root  # noqa: E402
from dancecap.pipeline import analyze as A, paths, snapshots  # noqa: E402

TRIAL = {"n": 1, "duration_s": 20.0, "size_mb": 9.3, "fps": 29.994, "frames": 600, "probed": True}


@pytest.fixture()
def root(tmp_path):
    r = make_root(tmp_path)
    yield r
    paths.set_root(None)


def _doc(run: str, steps: list[dict], **kw) -> dict:
    for s in steps:
        s.setdefault("status", "pending")
    d = {"schema": A.SCHEMA, "run": run, "take": "synth", "created": time.time(), "updated": time.time(),
         "agent_hint": "", "status": "planned", "input": {"path": "x", "route": "mono", "kind": "single", "videos": []},
         "facts": dict(TRIAL), "steps": steps, "plan": A.totals(steps), "cost_so_far_usd": 0.0}
    d.update(kw)
    A.run_dir(run).mkdir(parents=True, exist_ok=True)
    A.save(d)
    return d


def _steps(*ids_deps):
    return [A._step(i, i, "pod-gpu", deps, 10, 20) for i, deps in ids_deps]


# ------------------------------------------------------------------------------------------------ the plan

def test_mono_plan_is_what_is_built_plus_session_overhead():
    by = {s["id"]: s for s in A.route_plan("mono", TRIAL, {})}
    for sid in ("camera", "people", "sam3d_body"):        # the time shards (4 / 6 / 4) are not built: one pod each
        assert by[sid]["lanes"] == 1, sid
    for sid in ("beats", "camera", "people", "sam3d_body", "shape", "fits", "analyses", "avatars"):
        s = by[sid]
        assert s["overhead_min"]["kind"] == "session"
        assert s["p50_min"] == pytest.approx(s["compute_min"]["p50"] + s["overhead_min"]["p50"], abs=0.11)
        assert s["overhead_min"]["p50"] >= A.SESSION_OVERHEAD_MIN["p50"] and s["p90_min"] > s["p50_min"]
    # every fit session uploads the ma_2d checkpoint's 540 MB state_dict copy (the trial: the 1636 MB file in 600 s)
    assert by["fits"]["overhead_min"]["p50"] == pytest.approx(A.SESSION_OVERHEAD_MIN["p50"] + A.MA2D_CKPT_MB / 2.1 / 60, abs=0.11)
    # SAM 3D Body for 600 frames on one 4090: about 8 min of compute + a session (the trial's good session: 14.3 min, 7.4
    # of it pod-ready to compute; 2026-10-22: 1.8 min, the session 3.7 min in all for 100 frames)
    assert 10 < by["sam3d_body"]["p50_min"] < 13
    assert by["post"]["overhead_min"]["kind"] == "same_pod"          # one chunk: on the fit pod (trial: 2 s)
    assert "beats" in by["analyses"]["deps"]                          # timing reads the beat grid
    t = A.totals(list(by.values()))
    # the 20-s trial: ~1.6 h p50 with the 2026-10-22 session start-up (2.0 h with round 2's 6.3-min sessions)
    assert 80 < t["wall_min"]["p50"] < 120 and t["wall_min"]["p90"] > 1.6 * t["wall_min"]["p50"]
    assert t["critical"][:3] == ["intake", "upload", "people"]
    long = {s["id"]: s for s in A.route_plan("mono", {**TRIAL, "duration_s": 88.1, "frames": 2643}, {})}
    assert long["fits"]["lanes"] == 5 and long["post"]["overhead_min"]["kind"] == "session"
    multi = {s["id"]: s for s in A.route_plan("multi", {"n": 6, "duration_s": 360.0, "size_mb": 1230.0, "fps": 30.0,
                                                         "frames": 10800}, {"room_known": True})}
    assert multi["cameras"]["lanes"] == 6 and multi["cameras"]["overhead_min"]["kind"] == "session"
    assert all(multi[s]["overhead_min"]["kind"] == "farm" for s in ("fronts", "shape", "fits", "stitch", "analyses"))
    assert "6 ssh" in multi["upload"]["label"] and "4 files" in multi["upload"]["label"]


# ------------------------------------------------------------------------------------------------ ETA

def test_residual_pushes_the_eta_out_past_the_estimate():
    assert A.residual_min(23, 40, 0) == (23, 40)
    prev = 0.0
    for el in range(0, 240, 3):
        l50, l90 = A.residual_min(23, 40, el)
        assert l50 >= 0 and l90 >= l50
        assert el + l50 >= prev - 1e-6                               # the finish never moves earlier as time passes
        prev = el + l50
    l50, l90 = A.residual_min(23, 40, 49)              # the trial: "running 49 min of ~23" showed almost nothing left
    assert l50 > 10 and l90 > 2 * l50
    l50, l90 = A.residual_min(23, 40, 120)             # far past p90: at least 0.3 x / 1 x the elapsed time
    assert l50 >= 0.3 * 120 - 1e-6 and l90 >= 120 - 1e-6
    assert A.residual_min(23, 40, 1e9)[0] > 0                         # no underflow
    l50, _ = A.residual_min(30, 50, 10)
    assert l50 == pytest.approx(20, abs=0.2)                          # well before p50 it is the plan's remainder


def test_schedule_takes_a_session_eta_only_to_push_out():
    now = 1_000_000.0
    st = {"id": "fits", "deps": [], "p50_min": 23, "p90_min": 40, "status": "running", "started": now - 7 * 60}
    model = A.schedule([st], now)["left"]["fits"]
    assert model == pytest.approx(16 * 60, abs=30)
    assert A.schedule([{**st, "eta_s": 300}], now)["left"]["fits"] == pytest.approx(model)    # a front session's 5 min
    assert A.schedule([{**st, "eta_s": 3000}], now)["left"]["fits"] == 3000
    assert A.schedule([{**st, "eta_s": 3000, "eta_basis": "no history: timeout-based"}], now)["left"]["fits"] == model
    assert A.schedule([{**st, "eta_s": 300, "eta_s_p90": 9000}], now, "p90")["left"]["fits"] == 9000


def test_elapsed_stops_when_the_run_ends_and_a_finished_run_projects_nothing():
    steps = [A._step("a", "a", "pod-gpu", [], 10, 20), A._step("film", "f", "local", ["a"], 1, 2, optional=True,
                                                                 price="local")]
    t0 = 1_000_000.0
    steps[0].update(status="done", started=t0 + 10, ended=t0 + 600)
    steps[1].update(status="pending")
    doc = {"steps": steps, "created": t0, "status": "done", "ended": t0 + 700, "cost_so_far_usd": 1.0}
    for now in (t0 + 800, t0 + 99999):
        lv = A.live(doc, now)
        assert lv["elapsed_s"] == 700 and lv["eta_s"]["p50"] == 0 and lv["cost_usd"]["p90"] == 1.0
    old = {**doc, "ended": None, "driver": {"ended": t0 + 650}}       # runs from before the fix: the last step / driver
    assert A.live(old, t0 + 5000)["elapsed_s"] == 650
    assert A.live({**doc, "status": "cancelled", "ended": None, "updated": t0 + 300}, t0 + 5000)["elapsed_s"] == 600
    running = {**doc, "status": "waiting", "ended": None}
    assert A.live(running, t0 + 5000)["elapsed_s"] == 5000 and A.live(running, t0 + 5000)["ended"] is None


# ------------------------------------------------------------------------------------------------ run cost

def _sess(root: Path, sid: str, state: dict, label: str = "batch", result: bool = True) -> None:
    d = root / "work" / "runpod" / "sessions" / sid
    d.mkdir(parents=True, exist_ok=True)
    (d / "state.json").write_text(json.dumps(state))
    (d / "runner.json").write_text(json.dumps({"label": label, "started": A._sid_epoch(sid) + 1}))
    if result:
        (d / "result.json").write_text("{}")


def test_run_cost_counts_every_pod_its_sessions_created(root):
    run = "20261021-150000-synth"
    _doc(run, _steps(("fits", []), ("analyses", ["fits"])), created=A._sid_epoch(run))
    t = A._sid_epoch("20261021-150100-batch")
    led = [  # (pod, name, max_minutes, created offset, cost, owner)
        {"pod_id": "gw1", "name": "hm-farm-gateway", "owner": f"analysis:{run}", "created_ts": t - 30},
        {"pod_id": "deny1", "name": "batch", "max_minutes": 57.2, "created_ts": t + 30},      # deleted inside session A
        {"pod_id": "podA", "name": "batch", "max_minutes": 57.2, "created_ts": t + 80},       # session A's pod
        {"pod_id": "deny2", "name": "batch", "max_minutes": 35.0, "created_ts": t + 70},      # session B's (other work)
        {"pod_id": "podB", "name": "batch", "max_minutes": 35.0, "created_ts": t + 90},
        {"pod_id": "old", "name": "batch", "max_minutes": 57.2, "created_ts": t - 7200},      # before the run
        {"pod_id": "farm1", "name": "batch", "max_minutes": 57.2, "created_ts": t + 40, "owner": "farm:x"},
        # sessions C / D list what they created (the runner's GPU choice / pods_created): read exactly, even when the
        # row does not look like theirs (another kill time here)
        {"pod_id": "deny3", "name": "batch-p1", "max_minutes": 25.0, "created_ts": t + 200},
        {"pod_id": "podC", "name": "batch-p1", "max_minutes": 20.0, "created_ts": t + 230},
        {"pod_id": "deny4", "name": "batch-p2", "max_minutes": 26.0, "created_ts": t + 210},
        {"pod_id": "podD", "name": "batch-p2", "max_minutes": 20.0, "created_ts": t + 240}]
    for i, r in enumerate(led):
        r.update(gpu="g", cloud="SECURE", cost_per_hr=1.0, deleted_ts=r["created_ts"] + 60, cost=round(0.01 * (i + 1), 4))
    (root / "work" / "runpod").mkdir(parents=True, exist_ok=True)
    (root / "work" / "runpod" / "ledger.json").write_text(json.dumps(led))
    _sess(root, "20261021-150100-batch", {"analysis": {"run": run, "step": "fits"}, "pod": {"id": "podA"},
                                          "max_minutes": 57.2})
    _sess(root, "20261021-150101-batch", {"pod": {"id": "podB"}, "max_minutes": 35.0})          # not this run's
    _sess(root, "20261021-150300-batch-p1", {"analysis": {"run": run, "step": "analyses"}, "max_minutes": 20.0,
                                             "pod": {"id": "podC", "choice": {"tried": [
                                                 {"gpu": "g", "result": "no stock"},
                                                 {"gpu": "g", "pod_id": "deny3", "result": "broken host"},
                                                 {"gpu": "g", "pod_id": "podC", "result": "ok"}]}}}, label="batch-p1")
    _sess(root, "20261021-150300-batch-p2", {"analysis": {"run": run, "step": "analyses"}, "pod": {"id": "podD"},
                                             "max_minutes": 20.0, "pods_created": ["deny4", "podD"]}, label="batch-p2")
    det = A.cost_detail(run)
    got = {p["pod_id"]: p for p in det["pods"]}
    assert set(got) == {"gw1", "deny1", "podA", "deny3", "podC", "deny4", "podD"}
    assert got["deny1"]["sid"] == "20261021-150100-batch" and got["deny1"]["how"] == "deleted inside its session"
    assert got["gw1"]["step"] == "upload" and got["podC"]["step"] == "analyses"
    want = sum(r["cost"] for r in led if r["pod_id"] in got)
    assert det["total"] == pytest.approx(want) and A.refresh_cost(run) == pytest.approx(want)
    assert det["by_step"]["fits"] == pytest.approx(0.02 + 0.03)
    # a session an agent named with --session (untagged) counts too
    with A._RunLock(run):
        d = A.load(run)
        d["steps"][1]["sessions"] = ["20261021-150101-batch"]
        A.save(d)
    assert "podB" in {p["pod_id"] for p in A.run_pods(run)}


# ------------------------------------------------------------------------------------------------ the driver

def _snaps(monkeypatch) -> list:
    got = []
    monkeypatch.setattr(snapshots, "request", lambda run, ev, label="", sid=None, final=False: got.append((ev, final)))
    return got


def _probe_steps():
    return [A._step("probe", "generic", "local", [], 0.1, 0.2, executor="cmd", price="local", cmd=["-c", "pass"]),
            A._step("people", "agent step", "pod-gpu", ["probe"], 10, 15),
            A._step("film", "film", "local", ["people"], 1, 2, price="local", optional=True)]


def test_one_driver_per_run_and_snapshots_only_at_progress_points(root, monkeypatch):
    import psutil
    shots = _snaps(monkeypatch)
    run = "20261021-120000-synth"
    _doc(run, _probe_steps())
    parent = psutil.Process().parent()                 # a live process that is not this one
    with A._RunLock(run):
        d = A.load(run)
        d["driver"] = {"pid": parent.pid, "create_time": parent.create_time(), "started": time.time()}
        A.save(d)
    assert A.Driver(run, poll_s=0.05).loop() == 0      # another live driver holds the run: this one leaves at once
    assert [s["status"] for s in A.load(run)["steps"]] == ["pending"] * 3 and shots == []
    with A._RunLock(run):
        d = A.load(run)
        d.pop("driver")
        A.save(d)
    assert A.Driver(run, poll_s=0.05).loop() == 0      # the run's first driver
    assert [e for e, _ in shots] == ["start", "probe_done"]
    assert A.load(run)["status"] == "waiting" and A.load(run).get("ended") is None
    assert A.Driver(run, poll_s=0.05).loop() == 0      # resumed with nothing new: no snapshot, no second driver_end
    assert [e for e, _ in shots] == ["start", "probe_done"]
    assert [e["kind"] for e in A.read_jsonl(A.run_dir(run) / "events.jsonl")].count("driver_end") == 1
    spawned = []
    monkeypatch.setattr(A, "spawn_driver", lambda r: spawned.append(r) or 0)
    assert A.cmd_analysis(["step", run, "people", "done", "--note", "pulled"]) == 0 and spawned == [run]
    shots.clear()
    assert A.Driver(run, poll_s=0.05).loop() == 0      # a resumed driver: no "resume" snapshot, the final one only
    assert shots == [("final", True)]
    d = A.load(run)
    assert d["status"] == "done" and d["ended"] >= d["created"]
    kinds = [e["kind"] for e in A.read_jsonl(A.run_dir(run) / "events.jsonl")]
    assert kinds.count("driver_end") == 2              # waiting (a status change), then done
    A.cmd_analysis(["cancel", run])
    assert A.Driver(run, poll_s=0.05).loop() == 0 and A.load(run)["status"] == "cancelled"   # never revives a cancel


def test_a_mark_landing_while_the_driver_stops_is_not_lost(root, monkeypatch):
    _snaps(monkeypatch)
    run = "20261021-120001-synth"
    _doc(run, _probe_steps())
    real = A._has_work
    calls = []

    def has_work(doc):
        calls.append(1)
        if len(calls) == 1:          # an agent's mark lands just before the driver takes the lock to stop
            next(s for s in doc["steps"] if s["id"] == "people").update(status="done", ended=time.time())
            A.save(doc)
        return real(doc)
    monkeypatch.setattr(A, "_has_work", has_work)
    assert A.Driver(run, poll_s=0.05).loop() == 0
    got = {s["id"]: s["status"] for s in A.load(run)["steps"]}
    assert got == {"probe": "done", "people": "done", "film": "waiting_agent"} and A.load(run)["status"] == "done"


def test_a_failed_step_redone_by_an_agent_unblocks_what_waited_on_it(root, monkeypatch):
    shots = _snaps(monkeypatch)
    monkeypatch.setattr(A, "spawn_driver", lambda r: 0)
    run = "20261021-120006-synth"
    _doc(run, [A._step("bad", "fails", "local", [], 0.1, 0.2, executor="cmd", price="local",
                       cmd=["-c", "raise SystemExit(3)"]),
               A._step("after", "needs bad", "local", ["bad"], 0.1, 0.2, executor="cmd", price="local", cmd=["-c", "pass"])])
    assert A.Driver(run, poll_s=0.05).loop() == 1
    d = A.load(run)
    assert d["status"] == "failed" and d["ended"] and [s["status"] for s in d["steps"]] == ["failed", "blocked"]
    assert ("failed", True) in shots
    assert A.cmd_analysis(["step", run, "bad", "done", "--note", "fixed by hand"]) == 0
    d = A.load(run)
    assert d["status"] == "running" and "ended" not in d and d["steps"][1]["status"] == "pending"
    assert A.Driver(run, poll_s=0.05).loop() == 0
    assert A.load(run)["status"] == "done" and A.load(run)["steps"][1]["status"] == "done"


def test_spawn_driver_starts_one_driver(root, monkeypatch):
    import psutil
    run = "20261021-120002-synth"
    _doc(run, _probe_steps())
    pops = []

    class FakePopen:
        def __init__(self, argv, **kw):
            pops.append(argv)
            self.pid = os.getpid()                     # alive: "a driver is starting"
    monkeypatch.setattr(A.subprocess, "Popen", FakePopen)
    A.spawn_driver(run)
    A.spawn_driver(run)                                # the first has not registered yet: no second one
    assert len(pops) == 1 and pops[0][-2:] == ["resume", run]
    me = psutil.Process()
    with A._RunLock(run):
        d = A.load(run)
        d["driver"] = {"pid": me.pid, "create_time": me.create_time(), "started": time.time() + 1}
        A.save(d)
    A.spawn_driver(run)                                # it registered and runs: it sees every mark itself
    assert len(pops) == 1
    with A._RunLock(run):
        d = A.load(run)
        d["driver"]["ended"] = time.time() + 2
        A.save(d)
    A.spawn_driver(run)                                # it ended: the next mark starts a new one
    assert len(pops) == 2


def test_heartbeat_snapshots_every_15_min_while_the_run_moves(root, monkeypatch):
    import psutil
    run = "20261021-120003-synth"
    steps = _steps(("fits", []))
    steps[0].update(status="running", started=time.time())
    _doc(run, steps, status="waiting")
    clock = [time.time()]
    asked = []

    def request(r, ev, label):
        asked.append((round(clock[0] - t0), ev))
        sd = snapshots.snap_dir(r)
        sd.mkdir(parents=True, exist_ok=True)
        (sd / "requested.json").write_text(json.dumps({"ts": clock[0], "event": ev}))

    def sleep(s):
        clock[0] += 300
    t0 = clock[0]
    why = A.heartbeat_loop(run, max_loops=10, clock=lambda: clock[0], sleep=sleep, request=request)
    assert [t for t, _ in asked] == [0, 900, 1800, 2700] and {e for _, e in asked} == {"heartbeat"}
    hb = json.loads((A.run_dir(run) / "heartbeat.json").read_text())
    assert hb["pid"] == os.getpid() and hb["ended"] and why == "stopped"
    with A._RunLock(run):                              # nothing runs, nothing written for 7 h: it ends (idle)
        d = A.load(run)
        d["steps"][0]["status"] = "waiting_agent"
        A.save(d)
    clock[0] += 7 * 3600
    assert A.heartbeat_loop(run, clock=lambda: clock[0], sleep=sleep, request=request).startswith("nothing moved")
    A.cmd_analysis(["cancel", run])
    assert A.heartbeat_loop(run, clock=lambda: clock[0], sleep=sleep, request=request) == "the run is cancelled"
    parent = psutil.Process().parent()
    (A.run_dir(run) / "heartbeat.json").write_text(json.dumps({"pid": parent.pid, "create_time": parent.create_time()}))
    assert A.heartbeat_loop(run, request=request) == "another heartbeat runs"
    assert A.ensure_heartbeat(run) is False            # never spawned from a test / a temp root


# ------------------------------------------------------------------------------------------------ reused steps

def test_reused_steps_cost_nothing_and_are_marked(root, monkeypatch):
    _snaps(monkeypatch)
    monkeypatch.setattr(A, "spawn_driver", lambda r: 0)
    run = "20261021-120004-synth"
    _doc(run, _steps(("camera", []), ("fits", ["camera"])), status="waiting")
    before = A.live(A.load(run))["cost_usd"]["p50"]
    assert A.cmd_analysis(["step", run, "camera", "done", "--reused", "--note", "parent's solve cut"]) == 0
    d = A.load(run)
    cam = d["steps"][0]
    assert cam["reused"] is True and A.step_cost(cam) == 0.0
    lv = A.live(d)
    assert lv["reused"] == ["camera"] and lv["cost_usd"]["p50"] == pytest.approx(before - A.step_cost(
        {**cam, "reused": False}), abs=0.01)
    ev = A.read_jsonl(A.run_dir(run) / "events.jsonl")[-1]
    assert ev["kind"] == "done" and ev["reused"] is True
    assert A.cmd_analysis(["step", run, "camera", "start"]) == 0 and A.load(run)["steps"][0]["reused"] is False
    assert A.cmd_analysis(["step", run, "camera", "skip", "--reused"]) == 2


# ------------------------------------------------------------------------------------------------ intake defaults

def test_one_video_intake_writes_window_timeline_stitch_export(root, tmp_path, monkeypatch):
    from dancecap.pipeline import intake, takes
    vid = tmp_path / "clip" / "trial_clip.mp4"
    vid.parent.mkdir(parents=True)
    vid.write_bytes(b"0" * 1000)
    spec = {"dancers": {"lead": {"name": "John", "stature_m": 1.803}, "follow": {"name": "Lula", "stature_m": 1.702}},
            "dance": {"level": "professional", "category": "jack_and_jill"}}
    orig = intake.create
    monkeypatch.setattr(intake, "create", lambda req, **kw: orig(req, probe_fn=fake_probe(seconds=20.0, fps=29.994)))
    doc = {"take": "trial_clip", "facts": dict(TRIAL), "input": {"videos": [{"cam": "01", "path": str(vid)}],
                                                                  "soundtrack": None}}
    msg = A._intake(doc, spec)
    assert "single-view" in msg and "window" in msg and "[export]" in msg
    text = paths.take_toml("trial_clip").read_text(encoding="utf-8")
    cfg = tomllib.loads(text)
    assert cfg["window"] == [0.0, 19.97]                         # (600 - 1) / 29.994, the trial's hand-written value
    assert cfg["timeline"] == {"anchor": 0.0, "hz": 30.0} and cfg["ingest"]["max_fps"] == 30
    assert cfg["stitch"]["mono"] == [[0.0, 19.97]] and cfg["export"]["smplx_dir"] == "smplx_rf2_noslip"
    assert cfg["cameras"]["01"]["file"] == "trial_clip.mp4" and cfg["dancers"]["lead"]["name"] == "John"
    assert takes.window(cfg) == (0.0, 19.97) or list(takes.window(cfg)) == [0.0, 19.97]
    assert A.single_view_defaults(text, 20.0, 29.994) == (text, [])            # a second pass adds nothing
    # an existing window and tables are kept; no duration: no window, the rest still written
    t2, added = A.single_view_defaults('name = "x"\nwindow = [1.0, 5.0]\n\n[dance]\nlevel = "novice"\n', None, None)
    c2 = tomllib.loads(t2)
    assert c2["window"] == [1.0, 5.0] and c2["stitch"]["mono"] == [[1.0, 5.0]] and "window" not in added
    t3, added3 = A.single_view_defaults('name = "x"\n\n[dance]\nlevel = "novice"\n', None, None)
    assert "window" not in tomllib.loads(t3) and "stitch" not in tomllib.loads(t3) and "[export]" in added3


# ------------------------------------------------------------------------------------------------ the run's pod limit

def test_max_cost_is_the_runs_pod_limit(root, tmp_path, monkeypatch, capsys):
    vid = tmp_path / "v" / "clip.mp4"
    vid.parent.mkdir(parents=True)
    vid.write_bytes(b"0" * 2_000_000)
    doc = A.make_run(str(vid), probe=False, max_cost=3.5)
    assert doc["max_cost_usd"] == 3.5 and "analysis budget" in doc["agent_hint"]
    with pytest.raises(ValueError):
        A.make_run(str(vid), probe=False, max_cost=-1)
    assert A.cmd_analyze([str(vid), "--no-probe", "--dry-run", "--max-cost", "0.5"]) == 0
    out = capsys.readouterr().out
    assert "run pod limit: $0.50" in out and "below the p90" in out and not A.analyses_dir().exists()
    run = "20261021-120005-synth"
    up = A._step("upload", "u", "pod-cpu", [], 1, 2, executor="builtin", price="cpu_gw")
    _doc(run, [up], max_cost_usd=0.01, allow_pods=True)
    led = [{"pod_id": "gw0", "name": "hm-farm-gateway", "owner": f"analysis:{run}", "gpu": "cpu", "cloud": "SECURE",
            "cost_per_hr": 0.07, "created_ts": A.load(run)["created"] + 5, "deleted_ts": A.load(run)["created"] + 65,
            "cost": 0.02, "max_minutes": 45}]
    (root / "work" / "runpod").mkdir(parents=True, exist_ok=True)
    (root / "work" / "runpod" / "ledger.json").write_text(json.dumps(led))
    assert A.budget_left(run) == pytest.approx(-0.01)

    def no_pod(*a, **k):
        raise AssertionError("a pod step started past the run's limit")
    monkeypatch.setattr(A, "_upload", no_pod)
    _snaps(monkeypatch)
    assert A.Driver(run, allow_pods=True, poll_s=0.05).loop() == 0
    st = A.load(run)["steps"][0]
    assert st["status"] == "waiting_agent" and "pod limit" in st["message"]
    capsys.readouterr()
    assert A.cmd_analysis(["budget", run]) == 0 and capsys.readouterr().out.strip() == "0.00"
    assert A.cmd_analysis(["limit", run, "2"]) == 0 and A.budget_left(run) == pytest.approx(1.98)
    assert A.cmd_analysis(["limit", run, "none"]) == 0 and A.budget_left(run) is None
    capsys.readouterr()
    assert A.cmd_analysis(["budget", run]) == 0 and capsys.readouterr().out.strip() == "none"
    assert A.cmd_analysis(["limit", run, "-3"]) == 2
    assert A.live(A.load(run))["cost_usd"].get("limit") is None


# ------------------------------------------------------------------------------------------------ integration (round 2 review)

def test_parallel_lanes_share_the_uplink_and_run_in_rounds():
    """Review: 5 fit lanes each uploading the 1.6 GB checkpoint were planned as if each had the home uplink to itself (13
    min where the physical floor is 63), and 8 phones as 8 pods at once where `remote run --parallel` starts 6."""
    long = {s["id"]: s for s in A.route_plan("mono", {**TRIAL, "duration_s": 88.1, "frames": 2643}, {})}
    fits = long["fits"]
    assert fits["lanes"] == 5 and fits["overhead_min"]["p50"] == pytest.approx(A.SESSION_OVERHEAD_MIN["p50"] + 5 * A.MA2D_CKPT_MB / 2.1 / 60, abs=0.11)
    assert fits["overhead_min"]["p90"] == pytest.approx(A.SESSION_OVERHEAD_MIN["p90"] + 5 * A.MA2D_CKPT_MB / 0.73 / 60, abs=0.11)
    one = {s["id"]: s for s in A.route_plan("mono", TRIAL, {})}["fits"]
    assert one["lanes"] == 1 and one["overhead_min"]["p50"] == pytest.approx(A.SESSION_OVERHEAD_MIN["p50"] + A.MA2D_CKPT_MB / 2.1 / 60, abs=0.11)
    t = A.totals(list(long.values()))
    # ~3.3 h for 88 s: 5 fit lanes still push 5 x the 540 MB checkpoint copy through one uplink (4.3 h with the 1.6 GB
    # file and the 6.3-min sessions of round 2)
    assert t["wall_min"]["p50"] > 180
    multi = {s["id"]: s for s in A.route_plan("multi", {"n": 8, "duration_s": 60.0, "size_mb": 400.0, "fps": 30.0,
                                                         "frames": 1800}, {"room_known": True})}
    cam = multi["cameras"]
    assert (cam["lanes"], cam["rounds"], cam["pods_at_once"]) == (8, 2, 6)
    per_round = cam["compute_min"]["p50"] + cam["overhead_min"]["p50"]
    assert cam["p50_min"] == pytest.approx(2 * per_round, abs=0.11)
    assert A.step_cost(cam) == pytest.approx(cam["price_hr"] * 8 * per_round / 60, abs=0.01)   # each phone pays once
    assert "fronts" in multi and "rounds" not in multi["fronts"]                                 # farm lanes: no rounds


def test_pods_at_once_counts_steps_that_overlap():
    mono = A.route_plan("mono", TRIAL, {})
    t = A.totals(mono)
    assert t["pods_max"] >= 3                                   # beats, camera and people all start after the upload
    assert A.pods_at_once([A._step("a", "a", "pod-gpu", [], 10, 20, lanes=2),
                           A._step("b", "b", "pod-gpu", ["a"], 10, 20, lanes=3)]) == 3


def test_a_builtin_failure_beside_a_running_agent_step_keeps_the_run_open(root, monkeypatch):
    _snaps(monkeypatch)
    run = "20261021-120010-synth"
    steps = [A._step("room", "room", "pod-gpu", [], 20, 35),
             A._step("clocks", "clocks", "local", [], 1, 2, executor="cmd", price="local",
                     cmd=["-c", "import sys; sys.exit(2)"]),
             A._step("cameras", "cameras", "pod-gpu", ["clocks", "room"], 7, 12)]
    steps[0].update(status="running", started=time.time())
    _doc(run, steps)
    A.Driver(run, poll_s=0.05).loop()
    d = A.load(run)
    assert [s["status"] for s in d["steps"]] == ["running", "failed", "blocked"]
    assert d["status"] == "waiting" and not d.get("ended")       # the agent still works: no end, ETA and ELAPSED go on
    assert A.live(d)["eta_s"]["p50"] > 0


def test_finished_runs_keep_their_end(root, monkeypatch):
    _snaps(monkeypatch)
    monkeypatch.setattr(A, "spawn_driver", lambda r: 0)
    run = "20261021-120011-synth"
    t_end = time.time() - 3600
    _doc(run, _steps(("a", []), ("opt", ["a"])), status="done", ended=t_end, created=t_end - 700)
    with A._RunLock(run):
        d = A.load(run)
        d["steps"][0].update(status="done", ended=t_end)
        d["steps"][1].update(status="pending", optional=True)
        A.save(d)
    assert A.cmd_analysis(["step", run, "opt", "skip"]) == 0           # a late skip of an optional step: still done
    d = A.load(run)
    assert d["status"] == "done" and d["ended"] == t_end and A.live(d)["elapsed_s"] == 700
    assert A.cmd_analysis(["resume", run]) == 0 and A.load(run)["ended"] == t_end    # nothing to resume
    assert A.cmd_analysis(["cancel", run]) == 0
    d = A.load(run)
    assert d["status"] == "cancelled" and d["ended"] == t_end      # cancel keeps the end it had
    run2 = "20261021-120012-synth"
    _doc(run2, _steps(("a", [])), status="done", ended=t_end)
    assert A.cmd_analysis(["step", run2, "a", "start"]) == 0       # a step that STARTS reopens a done run
    d = A.load(run2)
    assert d["status"] == "running" and "ended" not in d and A.live(d)["eta_s"]["p50"] > 0


def test_fail_and_skip_clear_the_reused_flag(root, monkeypatch):
    _snaps(monkeypatch)
    monkeypatch.setattr(A, "spawn_driver", lambda r: 0)
    run = "20261021-120013-synth"
    _doc(run, _steps(("camera", [])), status="waiting")
    A.cmd_analysis(["step", run, "camera", "done", "--reused"])
    assert A.load(run)["steps"][0]["reused"] is True
    A.cmd_analysis(["step", run, "camera", "fail"])
    st = A.load(run)["steps"][0]
    assert st["status"] == "failed" and st["reused"] is False and A.step_cost(st) > 0


def test_raising_the_limit_unparks_the_step_it_held(root, monkeypatch):
    _snaps(monkeypatch)
    run = "20261021-120014-synth"
    up = A._step("upload", "u", "pod-cpu", [], 1, 2, executor="builtin", price="cpu_gw")
    _doc(run, [up], max_cost_usd=0.01, allow_pods=True)
    led = [{"pod_id": "gw0", "name": "hm-farm-gateway", "owner": f"analysis:{run}", "gpu": "cpu", "cloud": "SECURE",
            "cost_per_hr": 0.07, "created_ts": A.load(run)["created"] + 5, "deleted_ts": A.load(run)["created"] + 65,
            "cost": 0.02, "max_minutes": 45}]
    (root / "work" / "runpod").mkdir(parents=True, exist_ok=True)
    (root / "work" / "runpod" / "ledger.json").write_text(json.dumps(led))
    ran = []
    monkeypatch.setattr(A, "_upload", lambda doc, report: ran.append(1) or "sent")
    A.Driver(run, allow_pods=True, poll_s=0.05).loop()
    st = A.load(run)["steps"][0]
    assert st["status"] == "waiting_agent" and st["parked"] == "limit" and not ran
    assert A.cmd_analysis(["limit", run, "2"]) == 0                 # (spawns the driver for real only outside tests)
    A.Driver(run, allow_pods=True, poll_s=0.05).loop()               # what that spawned driver does
    st = A.load(run)["steps"][0]
    assert ran and st["status"] == "done" and not st.get("parked")
    # the 'needs --yes' park comes back with `analysis resume RUN --yes`
    run2 = "20261021-120015-synth"
    _doc(run2, [A._step("upload", "u", "pod-cpu", [], 1, 2, executor="builtin", price="cpu_gw")])
    A.Driver(run2, allow_pods=False, poll_s=0.05).loop()
    assert A.load(run2)["steps"][0]["parked"] == "yes"
    assert A.cmd_analysis(["resume", run2, "--yes"]) == 0
    assert A.load(run2)["steps"][0]["status"] == "done" and A.load(run2)["allow_pods"] is True


def test_an_abandoned_running_step_does_not_keep_the_heartbeat_for_ever(root):
    run = "20261021-120016-synth"
    steps = _steps(("fits", []))                                    # p90 20 min: busy for at most 3 x 20 min
    steps[0].update(status="running", started=time.time())
    _doc(run, steps, status="waiting")
    clock = [time.time()]
    asked = []

    def request(r, ev, label):
        asked.append(clock[0])
        sd = snapshots.snap_dir(r)
        sd.mkdir(parents=True, exist_ok=True)
        (sd / "requested.json").write_text(json.dumps({"ts": clock[0], "event": ev}))

    def sleep(s):
        clock[0] += 3600
    why = A.heartbeat_loop(run, max_loops=72, clock=lambda: clock[0], sleep=sleep, request=request)
    assert why.startswith("nothing moved") and len(asked) <= 2     # the review: 72 snapshots in 72 h, never ending


def test_the_older_estimator_pushes_out_past_its_estimate():
    from dancecap.pipeline import estimate
    r = estimate.live_blend(23 * 60, 40 * 60, None, 49 * 60)
    l50, l90 = A.residual_min(23, 40, 49)
    assert r["p50_s"] == round(l50 * 60) > 0 and r["p90_s"] >= r["p50_s"]   # it said 0 s left


def test_an_existing_one_video_take_gets_the_defaults_too(root, tmp_path, monkeypatch):
    from dancecap.pipeline import intake
    vid = tmp_path / "clip" / "wiz_clip.mp4"
    vid.parent.mkdir(parents=True)
    vid.write_bytes(b"0" * 1000)
    spec = {"dancers": {"lead": {"name": "John", "stature_m": 1.803}, "follow": {"name": "Lula", "stature_m": 1.702}},
            "dance": {"level": "professional", "category": "jack_and_jill"}}
    orig = intake.create
    intake_req = {"take": "wiz_clip", "source_mode": "reference", "videos": [{"cam": "01", "path": str(vid)}], **spec}
    orig(intake_req, probe_fn=fake_probe(seconds=20.0, fps=29.994))     # made elsewhere (the monitor's wizard)
    assert "window" not in tomllib.loads(paths.take_toml("wiz_clip").read_text(encoding="utf-8"))
    doc = {"take": "wiz_clip", "facts": dict(TRIAL), "input": {"videos": [{"cam": "01", "path": str(vid)}],
                                                                "soundtrack": None}}
    msg = A._intake(doc, None)
    cfg = tomllib.loads(paths.take_toml("wiz_clip").read_text(encoding="utf-8"))
    assert "reused" in msg and "window" in msg and cfg["window"] == [0.0, 19.97] and cfg["stitch"]["mono"]


def test_a_finished_run_keeps_its_cost(root):
    run = "20261021-150500-synth"
    _doc(run, _steps(("fits", [])), created=A._sid_epoch(run), status="done", ended=time.time())
    t = A._sid_epoch("20261021-150600-batch")
    led = [{"pod_id": "podA", "name": "batch", "max_minutes": 30.0, "created_ts": t + 20, "deleted_ts": t + 80,
            "gpu": "g", "cloud": "SECURE", "cost_per_hr": 1.0, "cost": 0.05}]
    (root / "work" / "runpod").mkdir(parents=True, exist_ok=True)
    (root / "work" / "runpod" / "ledger.json").write_text(json.dumps(led))
    _sess(root, "20261021-150600-batch", {"analysis": {"run": run, "step": "fits"}, "pod": {"id": "podA"},
                                          "max_minutes": 30.0})
    assert A.cost_detail(run)["total"] == pytest.approx(0.05)
    assert A.load(run)["cost_final"]["total"] == pytest.approx(0.05)
    import shutil
    shutil.rmtree(root / "work" / "runpod" / "sessions")            # its session dirs fall out of the newest-400 window
    assert A.cost_detail(run)["total"] == pytest.approx(0.05)


def test_retry_reruns_a_failed_builtin_step_and_reopens_the_run(root, tmp_path, monkeypatch):
    # live check 20261021-202809: the upload's gateway create failed on stock, the run ended failed, no mark restarted it
    shots = _snaps(monkeypatch)
    spawned = []
    monkeypatch.setattr(A, "spawn_driver", lambda r: spawned.append(r) or 0)
    flag = tmp_path / "stock_back"
    run = "20261021-120020-synth"
    _doc(run, [A._step("up", "gateway", "local", [], 0.1, 0.2, executor="cmd", price="local",
                       cmd=["-c", f"import os, sys; sys.exit(0 if os.path.exists({str(flag)!r}) else 3)"]),
               A._step("after", "needs up", "local", ["up"], 0.1, 0.2, executor="cmd", price="local", cmd=["-c", "pass"])])
    assert A.Driver(run, poll_s=0.05).loop() == 1
    assert A.load(run)["status"] == "failed" and [s["status"] for s in A.load(run)["steps"]] == ["failed", "blocked"]
    assert A.cmd_analysis(["step", run, "after", "retry"]) == 2          # blocked behind a failed step: retry that one
    assert A.cmd_analysis(["step", run, "up", "done", "--reused"]) == 0
    assert A.cmd_analysis(["step", run, "up", "retry"]) == 2             # only a failed step
    A.update_step(run, "up", status="failed")
    A.update_step(run, "after", status="blocked")
    flag.write_text("1")
    assert A.cmd_analysis(["step", run, "up", "retry", "--note", "stock is back"]) == 0 and spawned[-1] == run
    d = A.load(run)
    assert d["status"] == "running" and "ended" not in d
    assert [(s["status"], s.get("ended"), s.get("reused")) for s in d["steps"]] == [("pending", None, False),
                                                                                   ("pending", None, None)]
    assert A.Driver(run, poll_s=0.05).loop() == 0
    assert A.load(run)["status"] == "done" and ("final", True) in shots
