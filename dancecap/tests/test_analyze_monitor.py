"""2026-10-21 pipeline round: `pipeline analyze` (classify, route plans, schedule / ETA / cost, the driver), the monitor's
activity bundle and analysis API, snapshots, auto-start leases, the budget fixes, the slim bundles, parallel sessions and
the broken-host list. No pod, no Runpod call, no server, no Edge: temp package roots and fakes only.

    ..\\.venv-nopo4d\\Scripts\\python.exe -m pytest tests/test_analyze_monitor.py -q        (from dancecap/ or the workspace)
"""
import json
import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _pipeline_fixtures import FakePoller, make_root, make_take  # noqa: E402
from dancecap import remote, remote_jobs as rj  # noqa: E402
from dancecap.pipeline import activity, analyze as A, paths, runpod_view, snapshots  # noqa: E402
from dancecap.pipeline import monitor  # noqa: E402

BASE = "http://127.0.0.1:8790"


@pytest.fixture()
def root(tmp_path):
    r = make_root(tmp_path)
    yield r
    paths.set_root(None)


def _files(d: Path, names) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    for n in names:
        (d / n).write_bytes(b"x" * 2048)
    return d


# ------------------------------------------------------------------------------------------------ classify / plan (pure)

def test_classify_folder_video_and_slug(tmp_path):
    d = _files(tmp_path / "Larissa Kadu Recap!", ["b-phone.MOV", "a-phone.mp4", "c.mp4", ".hidden.mp4", "song.m4a",
                                                  "notes.txt"])
    c = A.classify(d)
    assert c["kind"] == "multi" and c["route"] == "multi"
    assert [v["cam"] for v in c["videos"]] == ["01", "02", "03"]
    assert [v["name"] for v in c["videos"]] == ["a-phone.mp4", "b-phone.MOV", "c.mp4"]
    assert c["soundtrack"].endswith("song.m4a") and c["take"] == "larissa_kadu_recap"
    one = A.classify(d / "c.mp4")
    assert one["kind"] == "single" and one["route"] == "mono" and one["take"] == "c"
    two = A.classify(_files(tmp_path / "pair", ["x.mp4", "y.mp4"]))
    assert two["kind"] == "two" and two["route"] == "multi"
    with pytest.raises(ValueError):
        A.classify(_files(tmp_path / "empty", ["a.txt"]))
    with pytest.raises(ValueError):
        A.classify(d / "notes.txt")
    assert A.slug("John & Lula - JnJ (2026)") == "john_lula_jnj_2026"
    assert A.slug("__") == "take" or A.slug("__").startswith("take")


def test_facts_use_the_median_phone_for_multi_view():
    cls = {"videos": [{"cam": f"{i:02d}", "size_bytes": 100e6} for i in range(1, 8)]}
    durs = [369.6, 360.5, 358.4, 300.3, 77.3, 353.1, 331.1]          # the class recap: one phone filmed 77 s
    f = A.facts_of(cls, {f"{i + 1:02d}": {"duration_s": d, "fps": 30.0} for i, d in enumerate(durs)})
    assert f["duration_s"] == pytest.approx(353.1) and f["n"] == 7 and f["frames"] == round(353.1 * 30)
    mono = A.facts_of({"videos": [{"cam": "01", "size_bytes": 13e6}]}, {"01": {"duration_s": 88.1, "fps": 29.994}})
    assert mono["duration_s"] == pytest.approx(88.1) and mono["size_mb"] == pytest.approx(13.0)


@pytest.mark.parametrize("route,n,dur", [("mono", 1, 88.1), ("multi", 6, 360.0)])
def test_route_plans_put_every_heavy_step_on_runpod(route, n, dur):
    f = {"n": n, "duration_s": dur, "size_mb": 13.0 if n == 1 else 1230.0, "fps": 30.0, "frames": int(dur * 30)}
    steps = A.route_plan(route, f, {"room_known": True})
    ids = [s["id"] for s in steps]
    by = {s["id"]: s for s in steps}
    assert ids[0] == "intake" and "upload" in ids and ids[-1] == "film"
    local = {s["id"] for s in steps if s["where"] == "local"}
    assert local <= {"intake", "clocks", "export", "film"}          # this PC: light work only (owner rule)
    for heavy in ("people", "fits", "analyses", "avatars", "shape"):
        assert by[heavy]["where"].startswith("pod"), heavy
    for s in steps:
        assert set(s["deps"]) <= set(ids) and s["p50_min"] <= s["p90_min"]
    t = A.totals(steps)
    assert t["critical"][0] == "intake" and t["critical"][-1] == "film"
    # 2026-10-22: planned from what is built (one pod per mono step, every session's start-up): the 88-s mono plan is
    # about 3.4 h, as john_lula's first pass measured 3.2 h of compute on its critical path (PIPELINE_EFFICIENCY.md 4.1)
    lo, hi = (150, 260) if route == "mono" else (40, 150)
    assert lo < t["wall_min"]["p50"] < hi and t["wall_min"]["p90"] > t["wall_min"]["p50"]
    assert t["cost_usd"]["p90"] >= t["cost_usd"]["p50"] > 0
    if route == "mono":
        assert by["people"]["lanes"] == 1 and by["sam3d_body"]["deps"] == ["people"]     # time shards are not built
        assert "sam3d_body" in by["fits"]["deps"]
        no_real = {s["id"]: s for s in A.route_plan("mono", f, {"realism": False})}
        assert "sam3d_body" not in no_real["fits"]["deps"] and no_real["sam3d_body"]["optional"]
    else:
        assert by["fits"]["lanes"] == 12 and by["cameras"]["lanes"] == n and "room" not in by
        assert "room" in {s["id"] for s in A.route_plan("multi", f, {})}


def test_schedule_and_live_follow_what_finished():
    steps = [{"id": "a", "deps": [], "p50_min": 10, "p90_min": 20, "where": "pod-gpu", "lanes": 2, "price_hr": 0.6},
             {"id": "b", "deps": ["a"], "p50_min": 30, "p90_min": 50, "where": "pod-gpu", "lanes": 1, "price_hr": 0.6},
             {"id": "c", "deps": ["a"], "p50_min": 5, "p90_min": 9, "where": "pod-cpu", "lanes": 1, "price_hr": 0.3},
             {"id": "d", "deps": ["b", "c"], "p50_min": 2, "p90_min": 4, "where": "local", "lanes": 1, "price_hr": 0.0}]
    sch = A.schedule([{**s, "status": "pending"} for s in steps], 0.0)
    assert sch["total_min"] == pytest.approx(42) and sch["critical"] == ["a", "b", "d"]
    now = 1_000_000.0
    run = [{**steps[0], "status": "done", "started": now - 900, "ended": now - 300},
           {**steps[1], "status": "running", "started": now - 600},
           {**steps[2], "status": "done", "started": now - 300, "ended": now - 60},
           {**steps[3], "status": "pending"}]
    lv = A.live({"steps": run, "created": now - 900, "cost_so_far_usd": 0.5}, now)
    # b has about 20 min left (its residual given 10 min run, well before its p50), then d
    assert lv["eta_s"]["p50"] == pytest.approx((30 - 10) * 60 + 2 * 60, abs=10)
    assert lv["done"] == 2 and lv["total"] == 4 and lv["running"] == ["b"]
    # a running step costs its residual minutes (2026-10-22), not its whole p50 again
    assert lv["cost_usd"]["so_far"] == 0.5 and lv["cost_usd"]["p50"] == pytest.approx(0.5 + 0.6 * 20 / 60, abs=0.01)
    finished = A.live({"steps": [{**s, "status": "done", "ended": now} for s in steps], "created": now - 60}, now)
    assert finished["eta_s"]["p50"] == 0 and finished["finish_at"]["p50"] is None


# ------------------------------------------------------------------------------------------------ the driver

def _doc(run: str, take: str, steps: list[dict]) -> dict:
    for s in steps:
        s.setdefault("status", "pending")
    return {"schema": A.SCHEMA, "run": run, "take": take, "created": time.time(), "updated": time.time(),
            "agent_hint": f"DANCECAP_ANALYSIS={run} DANCECAP_ANALYSIS_STEP=<step>",
            "status": "planned", "input": {"path": "x", "route": "mono", "kind": "single", "videos": []},
            "facts": {"n": 1, "duration_s": 10, "size_mb": 1, "fps": 30, "frames": 300}, "steps": steps,
            "plan": A.totals(steps), "cost_so_far_usd": 0.0}


def test_driver_runs_generic_steps_and_waits_for_agents(root, monkeypatch):
    run = "20261021-120000-synth"
    steps = [A._step("probe", "a generic command", "local", [], 0.1, 0.2, executor="cmd", price="local",
                     cmd=["-c", "print('hello {take}')"]),
             A._step("people", "take-specific today", "pod-gpu", ["probe"], 10, 15, lanes=3),
             A._step("film", "film", "local", ["people"], 1, 2, price="local", optional=True)]
    d = _doc(run, "synth", steps)
    A.run_dir(run).mkdir(parents=True)
    A.save(d)
    assert A.Driver(run, poll_s=0.05).loop() == 0
    got = {s["id"]: s for s in A.load(run)["steps"]}
    assert got["probe"]["status"] == "done" and "rc 0" in got["probe"]["message"]
    assert got["people"]["status"] == "waiting_agent" and "needs an agent" in got["people"]["message"]
    assert got["film"]["status"] == "pending" and A.load(run)["status"] == "waiting"
    assert "hello synth" in (A.run_dir(run) / "logs" / "probe.log").read_text()
    spawned = []
    monkeypatch.setattr(A, "spawn_driver", lambda r: spawned.append(r) or 0)
    assert A.cmd_analysis(["step", run, "people", "done", "--note", "fits pulled", "--session", "20261021-121212-batch"]) == 0
    assert spawned == [run]
    assert A.Driver(run, poll_s=0.05).loop() == 0
    d2 = A.load(run)
    got = {s["id"]: s for s in d2["steps"]}
    assert got["people"]["sessions"] == ["20261021-121212-batch"] and got["film"]["status"] == "waiting_agent"
    assert d2["status"] == "done"             # film is optional: the required steps are done
    ev = [e["kind"] for e in A.read_jsonl(A.run_dir(run) / "events.jsonl")]
    assert "waiting_agent" in ev and "done" in ev
    assert A.cmd_analysis(["cancel", run]) == 0 and A.load(run)["status"] == "cancelled"


def test_driver_upload_needs_yes_and_failed_steps_block_their_dependents(root):
    run = "20261021-120001-synth"
    steps = [A._step("upload", "u", "pod-cpu", [], 1, 2, executor="builtin", price="cpu_gw"),
             A._step("bad", "fails", "local", [], 0.1, 0.2, executor="cmd", price="local", cmd=["-c", "raise SystemExit(3)"]),
             A._step("after", "needs bad", "local", ["bad"], 1, 2, executor="cmd", price="local", cmd=["-c", "pass"])]
    A.run_dir(run).mkdir(parents=True)
    A.save(_doc(run, "synth", steps))
    assert A.Driver(run, allow_pods=False, poll_s=0.05).loop() == 0       # waiting (upload needs --yes)
    got = {s["id"]: s for s in A.load(run)["steps"]}
    assert got["upload"]["status"] == "waiting_agent" and "--yes" in got["upload"]["message"]
    assert got["bad"]["status"] == "failed" and "rc=3" in got["bad"]["message"]
    assert got["after"]["status"] == "blocked"


def test_intake_reuses_an_existing_take_and_asks_for_dancer_facts(root, tmp_path):
    make_take(root, "synth", cams=("01",), video_dir=tmp_path / "vids")
    vid = tmp_path / "vids" / "01-a.mp4"
    vid.write_bytes(b"0" * 100)
    d = {"take": "synth", "input": {"videos": [{"cam": "01", "path": str(vid), "name": vid.name}]}}
    assert "reused" in A._intake(d, None)
    other = tmp_path / "elsewhere.mp4"
    other.write_bytes(b"0")
    with pytest.raises(RuntimeError, match="other videos"):
        A._intake({"take": "synth", "input": {"videos": [{"cam": "01", "path": str(other)}]}}, None)
    with pytest.raises(RuntimeError, match="--lead"):
        A._intake({"take": "newtake", "input": {"videos": [{"cam": "01", "path": str(other)}], "soundtrack": None}}, None)


# ------------------------------------------------------------------------------------------------ sessions, activity

def _session(root: Path, sid: str, state: dict, result: bool = True, runner_pid: int | None = None) -> Path:
    d = root / "work" / "runpod" / "sessions" / sid
    d.mkdir(parents=True, exist_ok=True)
    (d / "state.json").write_text(json.dumps(state))
    (d / "runner.log").write_text("[pod-runner 10:00:00] bundle_scan ...\n")
    if result:
        time.sleep(0.01)
        (d / "result.json").write_text("{}")
    if runner_pid:
        (d / "runner.json").write_text(json.dumps({"pid": runner_pid, "started": time.time() + 10}))
    return d


def test_session_states_are_honest(root):
    led = root / "work" / "runpod" / "ledger.json"
    led.parent.mkdir(parents=True, exist_ok=True)
    now = time.time()
    led.write_text(json.dumps([
        {"pod_id": "podskip01", "cost_per_hr": 0.79, "created_ts": now - 1000, "deleted_ts": now - 100, "cost": 0.2,
         "max_minutes": 20},
        {"pod_id": "podorph01", "cost_per_hr": 0.5, "created_ts": now - 300, "max_minutes": 60}]))
    _session(root, "20261021-132809-batch", {"pod": {"id": "podskip01"}, "bootstrap_rc": 1,
                                             "jobs": {"probe": {"status": "skipped: bootstrap feature failed ['base']"}},
                                             "timings": [{"step": "bootstrap", "ok": True, "seconds": 860},
                                                         {"step": "bootstrap_retry", "ok": True, "seconds": 48}]})
    s = runpod_view.session_summary("20261021-132809-batch")
    assert s["state"] == "failed" and "probe" in s["jobs_bad"]
    ph = {p["name"]: p["status"] for p in activity.ended_phases(json.loads(
        (root / "work/runpod/sessions/20261021-132809-batch/state.json").read_text()))}
    assert ph == {"bootstrap": "failed", "bootstrap_retry": "failed"}
    # an open pod whose runner died (a pid that started after runner.json was written cannot be the runner)
    _session(root, "20261021-140000-batch", {"pod": {"id": "podorph01"}, "jobs": {"j": {"status": "running"}}},
             result=False, runner_pid=os.getpid())
    (root / "work/runpod/sessions/20261021-140000-batch/runner.json").write_text(
        json.dumps({"pid": os.getpid(), "started": time.time() - 3600 * 24 * 365}))
    assert runpod_view.session_summary("20261021-140000-batch")["state"] == "orphaned"


def test_upload_info_reads_the_striped_uploaders_counter(root):
    d = _session(root, "20261021-150000-batch", {"bundle": {"bytes": 300e6}, "timings": []}, result=False)
    up = rj.UploadProgress(d, 300_000_000, 4)
    up("bundle.tar.gz", 100_000_000, 300_000_000)
    info = activity.upload_info("20261021-150000-batch")
    assert info["source"] == "uploader" and info["sent"] == 100_000_000 and info["streams"] == 4 and not info["done"]
    assert info["mbps"] > 0 and info["eta_s"] is not None
    up("bundle.tar.gz", 300_000_000, 300_000_000)
    assert activity.upload_info("20261021-150000-batch")["done"] is True


def test_activity_bundle_and_analysis_api(root, monkeypatch):
    from dancecap.pipeline.monitor import server
    run = "20261021-120002-synth"
    steps = [A._step("intake", "i", "local", [], 0.5, 1, executor="builtin", price="local"),
             A._step("people", "p", "pod-gpu", ["intake"], 10, 15, lanes=3)]
    d = _doc(run, "synth", steps)
    d["steps"][0].update(status="done", started=time.time() - 60, ended=time.time() - 30)
    d["steps"][1].update(status="waiting_agent", message="needs an agent")
    d["status"] = "waiting"
    A.run_dir(run).mkdir(parents=True)
    A.save(d)
    _session(root, "20261021-120500-batch", {"analysis": {"run": run, "step": "people"}, "pod": {"id": "pp1"},
                                             "jobs": {"people": {"status": "ok"}}, "timings": []})
    b = activity.bundle(rp=FakePoller().summary())
    assert b["focus"] == run and b["analyses"][0]["live"]["waiting_agent"] == ["people"]
    card = b["analyses"][0]
    assert card["steps"][1]["sessions"] == ["20261021-120500-batch"] and card["sessions"][0]["analysis"]["step"] == "people"
    # an agent's tagged session running now makes its waiting step "running" (map, counts, ETA)
    led = root / "work" / "runpod" / "ledger.json"
    led.write_text(json.dumps([{"pod_id": "pp2", "cost_per_hr": 0.5, "created_ts": time.time() - 120, "max_minutes": 30}]))
    d2 = _session(root, "20261021-121000-batch", {"analysis": {"run": run, "step": "people"}, "pod": {"id": "pp2"},
                                                  "jobs": {"p": {"status": "running"}}, "timings": []}, result=False)
    import psutil
    (d2 / "runner.json").write_text(json.dumps({"pid": os.getpid(), "started": psutil.Process().create_time() + 1}))
    card = activity.analysis_card(A.load(run), time.time())
    st = {s["id"]: s for s in card["steps"]}["people"]
    assert st["status"] == "running" and "driven by an agent" in st["message"] and card["live"]["running"] == ["people"]
    led.write_text("[]")
    m = server.Monitor(port=8790, poller=FakePoller())
    app = server.create_app(m)
    c = app.test_client()
    c.get("/", base_url=BASE)
    r = c.get("/api/activity", base_url=BASE)
    assert r.status_code == 200 and r.get_json()["focus"] == run
    r = c.get(f"/api/analyses/{run}", base_url=BASE)
    assert r.status_code == 200 and r.get_json()["steps"][0]["status"] == "done"
    assert c.get("/api/analyses/..%2Fx", base_url=BASE).status_code == 404
    assert c.get(f"/api/snapshots/{run}/../../ledger.json", base_url=BASE).status_code == 404
    assert app.test_client().get("/api/activity", base_url=BASE).status_code == 403     # cookie guard
    m.helper.shutdown(wait=True)


def test_snapshot_renders_a_static_page_and_indexes_it(root, monkeypatch):
    run = "20261021-120003-synth"
    steps = [A._step("intake", "i", "local", [], 0.5, 1, executor="builtin", price="local"),
             A._step("people", "people, 6 shards", "pod-gpu", ["intake"], 17, 27, lanes=6)]
    d = _doc(run, "synth", steps)
    d["steps"][0].update(status="done", started=time.time() - 60, ended=time.time() - 30)
    d["steps"][1].update(status="running", started=time.time() - 300)
    A.run_dir(run).mkdir(parents=True)
    A.save(d)
    kinds = snapshots.step_blocks(d["steps"][1], time.time())
    # no progress figure: striped "time" squares (elapsed against the plan), never "run" (= measured progress)
    assert len(kinds) == 6 * round(17 / 2) and kinds[0] == "time" and kinds[-1] == "left"   # 6 pods x 17 min / 2-min squares
    calls = []

    def fake_png(hp, pp, height):
        calls.append((hp.read_text(encoding="utf-8"), height))
        pp.write_bytes(b"\x89PNG" + b"0" * 2000)
        return "edge", 12, None
    monkeypatch.setattr(snapshots, "render_png", fake_png)
    shot = snapshots.take(run, "people_start", "people started")
    page, height = calls[0]
    assert "<script" not in page and "people, 6 shards" in page and 'class="b run"' in page and height > 400
    assert shot["png"] == "01_people_start.png" and shot["done"] == 1 and shot["total"] == 2
    assert snapshots.index(run)["shots"][0]["event"] == "people_start"
    assert snapshots.list_runs()[0]["run"] == run
    assert snapshots.request(run, "x") is False        # never spawns from a test / a temp root


# ------------------------------------------------------------------------------------------------ monitor auto-start

def test_leases_and_activity_hold(root):
    assert monitor.ensure_running("test") == {"state": "skipped"}      # temp root / pytest: never starts a server
    f = monitor.write_lease("unit test")
    assert f is not None and [x["reason"] for x in monitor.live_leases()] == ["unit test"]
    assert monitor.activity_hold().startswith("lease")
    dead = f.with_name("999999.json")
    dead.write_text(json.dumps({"pid": 999999, "create_time": 1.0, "reason": "gone"}))
    f.unlink()
    assert monitor.live_leases() == [] and not dead.exists()            # a dead process's lease is pruned
    assert monitor.activity_hold() is None
    led = root / "work" / "runpod" / "ledger.json"
    led.parent.mkdir(parents=True, exist_ok=True)
    led.write_text(json.dumps([{"pod_id": "x1", "cost_per_hr": 1, "created_ts": time.time() - 60, "max_minutes": 30}]))
    assert "pod x1 open" in monitor.activity_hold()


# ------------------------------------------------------------------------------------------------ runner speedups

def _job(name, needs=(), after=(), **kw):
    return rj.parse_job({"schema": rj.SCHEMA, "name": name, "take": "t", "max_minutes": 10, "commands": ["true"],
                         "outputs": [f"dancecap/work/t/{name}/"], "needs": list(needs), "after": list(after), **kw})


def test_slim_bundles_by_bootstrap_features(monkeypatch):
    monkeypatch.delenv("DANCECAP_BUNDLE_PYWHEELS", raising=False)
    tool = _job("tool")
    fit = _job("fit", needs=["mamma"])
    assert rj.wheel_policy([tool]) == {"pywheels": False, "wheelhouse": False,
                                       "why": "base/check only: the pod uses PyPI"}
    assert rj.wheel_policy([tool, fit])["pywheels"] and rj.wheel_policy([fit])["wheelhouse"]
    assert rj.wheel_policy(None)["pywheels"]                       # other modules' callers: unchanged
    assert rj.wheel_policy([_job("x", bundle_pywheels=True)])["pywheels"]
    monkeypatch.setenv("DANCECAP_BUNDLE_PYWHEELS", "0")
    assert not rj.wheel_policy([fit])["pywheels"]
    monkeypatch.delenv("DANCECAP_BUNDLE_PYWHEELS")
    arcs = [a for _, a in rj.code_items([tool])]
    assert not any("pywheels" in a or "wheelhouse" in a for a in arcs) and "dancecap/dancecap" in arcs


def test_components_split_independent_jobs():
    jobs = [_job("a"), _job("b", after=["a"]), _job("c"), _job("d", after=["zz"])]
    groups = sorted([[j.name for j in g] for g in rj.components(jobs)])
    assert groups == [["a", "b"], ["c"], ["d"]]
    assert rj.upload_streams(10 << 20) == 1 and rj.upload_streams(300 << 20) == 4


def test_gpu_ladder_and_broken_hosts(tmp_path, monkeypatch):
    assert remote.gpus_for([None, 24]) is None and remote.gpus_for([]) is None
    l24 = remote.gpus_for([24, 16])
    assert l24[0] == "NVIDIA GeForce RTX 4090" and not any("5090" in g or "PRO" in g for g in l24)
    assert all(gb >= 48 for g, gb in remote.GPU_LADDER if g in remote.gpus_for([40]))
    monkeypatch.setattr(remote, "BAD_HOSTS", tmp_path / "bad_hosts.json")
    monkeypatch.setattr(remote, "LEDGER", tmp_path / "ledger.json")
    assert remote.host_denied("60.249.37.148")                       # the measured broken host is seeded
    assert remote.host_denied("10.0.0.1") is None
    remote.note_host("10.0.0.1", False, "CUDA failed")
    assert remote.host_denied("10.0.0.1")
    remote.note_host("10.0.0.1", True)
    assert remote.host_denied("10.0.0.1") is None

    class FakePod:
        def __init__(self, exc=None, out="PREFLIGHT_CUDA_OK NVIDIA L40S\nPREFLIGHT_OK\n"):
            self.exc, self.out = exc, out

        def output(self, cmd, timeout=0):
            assert "PREFLIGHT_" in cmd
            if self.exc:
                raise self.exc
            return self.out
    assert remote.preflight(FakePod()) == (True, "PREFLIGHT_CUDA_OK NVIDIA L40S")
    ok, why = remote.preflight(FakePod(RuntimeError(f"pod command failed (4): {remote.PREFLIGHT_CMD[:200]}: x")))
    assert not ok and why.startswith("CUDA failed")
    ok, why = remote.preflight(FakePod(RuntimeError(f"pod command failed (255): {remote.PREFLIGHT_CMD[:200]}: ssh")))
    assert ok and why.startswith("unchecked")                         # an ssh problem never condemns a host


def test_effective_cap_prefers_env_then_the_runners_record(tmp_path, monkeypatch):
    monkeypatch.setattr(remote, "CAP_IN_USE", tmp_path / "cap_in_use.json")
    monkeypatch.setenv("DANCECAP_RUNPOD_CAP", "112.54")
    remote.record_cap_in_use("20261021-130000-batch")
    assert remote.effective_cap() == {"cap": 112.54, "source": "DANCECAP_RUNPOD_CAP"}
    monkeypatch.delenv("DANCECAP_RUNPOD_CAP")
    eff = remote.effective_cap()
    assert eff["cap"] == 112.54 and "20261021-130000-batch" in eff["source"]
    assert remote.cap() == remote.DEFAULT_CAP                          # the runners' own gate is unchanged
    assert remote.effective_cap(now=time.time() + 4 * 86400)["cap"] == remote.DEFAULT_CAP


def test_effective_cap_shows_the_balance_unless_a_runner_set_one(tmp_path, monkeypatch):
    monkeypatch.setattr(remote, "CAP_IN_USE", tmp_path / "cap_in_use.json")
    monkeypatch.setattr(remote, "BALANCE_CAP", tmp_path / "balance_cap.json")
    monkeypatch.setattr(remote, "spent", lambda: 100.0)
    monkeypatch.setattr(remote, "_ctl", lambda *a, **k: {"clientBalance": 62.7})
    monkeypatch.delenv("DANCECAP_RUNPOD_CAP", raising=False)
    monkeypatch.delenv("DANCECAP_RUNPOD_BALANCE")
    remote.record_cap_in_use("20261021-170000-batch")                   # a runner that follows the balance
    assert json.loads((tmp_path / "cap_in_use.json").read_text())["source"] == "balance"
    eff = remote.effective_cap()
    assert eff["cap"] == 160.7 and eff["source"].startswith("the Runpod balance")
    monkeypatch.setenv("DANCECAP_RUNPOD_CAP", "112.54")
    remote.record_cap_in_use("20261021-170500-batch")                   # an explicit cap a live workflow still enforces
    monkeypatch.delenv("DANCECAP_RUNPOD_CAP")
    assert remote.effective_cap()["cap"] == 112.54
