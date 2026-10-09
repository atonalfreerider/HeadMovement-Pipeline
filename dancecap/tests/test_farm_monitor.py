"""The Pipeline Monitor's farm view (docs/FARM.md section 7): the data layer (coverage cells, chunks, jobs, pods, budget,
volume), the read-only adapter for live `fit_full` sessions, the REST routes through the Flask test client, the actions'
safety rules (plan -> approve, never queue a live fit_full chunk, only farm pods are stopped), the legacy archive and the
synthetic demo. No server, no pod, no Runpod call: everything runs on a temp package root."""
import json
import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pipeline_fixtures import FakePoller, make_root, touch  # noqa: E402
from dancecap.pipeline import farm_actions, farm_demo, farm_view, fit_sessions, legacy_archive, paths  # noqa: E402
from dancecap.pipeline.monitor import server  # noqa: E402

BASE = "http://127.0.0.1:8790"


@pytest.fixture()
def root(tmp_path):
    r = make_root(tmp_path)
    farm_view._cov_cache.clear()
    yield r
    paths.set_root(None)


class Client:
    def __init__(self, app, token):
        self.c = app.test_client()
        self.token = token
        self.c.get("/", base_url=BASE)

    def get(self, url, **kw):
        return self.c.get(url, base_url=BASE, **kw)

    def post(self, url, body=None, token=True):
        h = {"X-Monitor-Token": self.token} if token else {}
        return self.c.post(url, base_url=BASE, json=body if body is not None else {}, headers=h)


@pytest.fixture()
def client(root):
    m = server.Monitor(port=8790, poller=FakePoller())
    app = server.create_app(m)
    yield Client(app, m.token)
    m.helper.shutdown(wait=True)


def _session(root: Path, sid: str, seg: str, jobs: dict, pod: dict | None = None, take="dance_x", age_s=0.0, result=False):
    d = paths.sessions_dir() / sid
    st = {"sid": sid, "created": "2026-10-08T13:00:00", "timings": [{"step": "upload", "seconds": 3, "ok": True,
                                                                     "start": time.time() - 600}],
          "jobs": {k: {"status": v[0], "current": v[1], "spec": {"take": f"{take}_{seg}"}} for k, v in jobs.items()},
          "pod": pod or {}, "max_minutes": 110.0, "estimate_min": {"total": 131.0}}
    touch(d / "state.json", time.time() - age_s, json.dumps(st))
    if result:
        touch(d / "result.json", time.time() + 5, "{}")
    return d


# ------------------------------------------------------------------------------------------------ demo + bundle

def test_demo_bundle_cells_chunks_and_rev(root):
    farm_demo.write_demo("farm_demo", "mostly_partial")
    assert [d["id"] for d in farm_view.list_dances()] == []                       # demo dances are not listed by default
    assert [d["id"] for d in farm_view.list_dances(show_demo=True)] == ["farm_demo"]
    b = farm_view.build("farm_demo", rp={"connection": "ok", "pods": [{"id": "realpod123456", "name": "x"}]})
    cov = b["coverage"]
    assert cov["exists"] and cov["n_cells"] == 190 and cov["counts"] == {"complete": 19, "partial": 114, "none": 57}
    assert cov["percent"]["complete"] == 10.0 and cov["seconds"]["none"] == 57.0
    cells = cov["cells"]
    assert cells[0]["s"] == "complete" and cells[19]["s"] == "partial" and cells[19]["g"] == "front" and cells[189]["s"] == "none"
    assert cells[30]["c"] == "g570" and cells[30]["live"] is True              # a running chunk's cells animate
    assert cells[5]["live"] is False and cells[5]["c"] == "g0"
    assert any(c["o"] for c in cells)                                           # overlaps of neighbouring chunks are marked
    by = {c["id"]: c for c in b["chunks"]}
    assert by["g570"]["status"] == "running" and by["g3000"]["status"] == "queued" and by["g0"]["status"] == "stitched"
    assert by["g570"]["t0"] == pytest.approx(cells[19]["t0"], abs=0.01)
    assert {j["id"] for j in b["jobs"] if j["status"] == "running"} == {"g570", "g1590"}
    job = next(j for j in b["jobs"] if j["id"] == "g570")
    assert job["progress"]["frac"] == pytest.approx(0.46) and job["eta"]["p50"] == pytest.approx(24 * 60) \
        and job["heartbeat_age_s"] is not None and [s["name"] for s in job["steps"]] == ["front", "fit", "refine"]
    assert {q["id"] for q in b["queue"]} >= {"g3000", "final"}
    assert [p["id"] for p in b["pods"]] == ["demo4090a", "demo4090b"]            # the account's real pods are not mixed into a demo
    assert b["budget"]["cap"] == 80.0 and b["budget"]["available_usd"] == 24.1
    assert b["watcher"]["running"] is True and b["world"]["sealed"] is True
    r1 = farm_view.rev_of(b)
    assert farm_view.rev_of(dict(b, now=b["now"] + 5)) == r1                    # the clock alone never changes the ETag
    farm_demo.write_demo("farm_demo", "midrun")
    assert farm_view.rev_of(farm_view.build("farm_demo")) != r1


def test_missing_coverage_and_defaults(root):
    (paths.work() / "empty_dance" / "farm").mkdir(parents=True)
    touch(paths.work() / "empty_dance" / "farm" / "state.json", None, json.dumps({"schema": "x", "chunks": {}}))
    b = farm_view.build("empty_dance")
    assert b["coverage"]["exists"] is False and b["coverage"]["cells"] == [] and b["chunks"] == []
    assert any(a["id"] == "no-coverage" for a in b["alerts"])
    # another workflow's coverage_cells.json (same cell format) is accepted when coverage.json is missing
    touch(paths.work() / "empty_dance" / "coverage_cells.json", None, json.dumps(
        {"cells": [{"i": 0, "t0": 0, "t1": 1, "state": "complete", "stage": "refined"},
                   {"i": 1, "t0": 1, "t1": 2, "state": "bogus", "stage": "weird"}]}))
    farm_view._cov_cache.clear()
    b = farm_view.build("empty_dance")
    assert [c["s"] for c in b["coverage"]["cells"]] == ["complete", "none"] and b["coverage"]["source"] == "coverage_cells.json"
    assert farm_view.default_dance() == "empty_dance"


def test_landing_detection_marks_the_map_stale(root):
    farm_demo.write_demo("farm_demo", "midrun")
    cov_m = (paths.work() / "farm_demo" / "farm" / "coverage.json").stat().st_mtime
    assert farm_view.build("farm_demo")["outputs"]["stale"] is False
    out = paths.work() / "farm_demo_g1590" / "smplx_rf2_noslip"            # a refined chunk lands next to the coverage
    out.mkdir(parents=True)
    os.utime(out, (cov_m + 60, cov_m + 60))
    state = json.loads((paths.work() / "farm_demo" / "farm" / "state.json").read_text())
    state["chunks"]["g1590"]["take"] = "farm_demo_g1590"
    touch(paths.work() / "farm_demo" / "farm" / "state.json", cov_m - 1, json.dumps(state))
    b = farm_view.build("farm_demo")
    assert b["outputs"]["stale"] is True and any(a["id"] == "stale" for a in b["alerts"])


# ------------------------------------------------------------------------------------------------ fit_full adapter

def test_fit_sessions_adapter_is_read_only_and_picks_the_newest(root):
    fitdir = paths.work() / "dance_x" / "fit"
    touch(fitdir / "runner_a.json", None, json.dumps({"seg": "a", "pid": os.getpid(), "started": "2026-10-08T13:16:38", "order": 2}))
    touch(fitdir / "commitments.json", None, json.dumps({"a": {"worst": 3.6, "pid": os.getpid()}}))
    pod = {"id": "podaaaaaa111", "name": "fit-a", "gpu": "NVIDIA A100-SXM4-80GB", "cost_per_hr": 1.39, "created_ts": time.time() - 1200}
    _session(root, "20261008-120000-fit-a", "a", {"fit": ("failed: preflight rc=2", None), "refine": ("skipped: x", None)}, pod)
    new = _session(root, "20261008-131657-fit-a", "a", {"fit": ("running", "fit_c8"), "refine": ("pending", None),
                                                         "quality": ("pending", None)}, pod)
    _session(root, "20261008-131606-fit-b", "b", {"fit": ("ok", None), "refine": ("ok", None), "quality": ("ok", None)},
             dict(pod, id="podbbbbbb222", name="fit-b"))
    _session(root, "20261001-000000-fit-c", "c", {"fit": ("running", None)}, dict(pod, id="podcccccc333"), age_s=30 * 86400)   # too old
    _session(root, "20261008-100000-batch", "z", {"x": ("ok", None)})                                                         # not a fit
    touch(paths.sessions_dir() / "20261008-131657-fit-a" / "progress" / "puller.json", None,
          json.dumps({"last_ok": time.time() - 7, "gpu": [{"util": 91.0}]}))
    before = sorted(str(p) + str(p.stat().st_mtime_ns) for p in paths.work().rglob("*") if p.is_file())
    live = fit_sessions.live_fits("dance_x")
    after = sorted(str(p) + str(p.stat().st_mtime_ns) for p in paths.work().rglob("*") if p.is_file())
    assert before == after                                                       # nothing written, nothing touched
    assert sorted(live["segments"]) == ["a", "b"] and live["found"] == 2
    a = live["segments"]["a"]
    assert a["sid"] == "20261008-131657-fit-a" and a["status"] == "running" and a["dance"] == "dance_x"
    assert [(j["name"], j["status"]) for j in a["jobs"]] == [("fit", "running"), ("refine", "pending"), ("quality", "pending")]
    assert a["runner"]["alive"] is True and a["worst_usd"] == 3.6 and a["heartbeat_age_s"] in range(6, 12)
    assert a["pod"]["id"] == "podaaaaaa111" and a["pod"]["price_hr"] == 1.39 and a["pod"]["gpu_util"] == 91.0
    assert live["segments"]["b"]["status"] == "done" and live["segments"]["b"]["progress"]["frac"] == 1.0
    assert {p["id"] for p in live["pods"]} == {"podaaaaaa111", "podbbbbbb222"}
    assert fit_sessions.live_fits("another_dance")["found"] == 0
    assert new.exists()


def test_live_fits_show_up_as_pods_and_running_chunks(root):
    # a farm state with S1 chunks a-b 'planned' (as core writes them) + live fit_full sessions = running chunks, animated cells
    farm_demo.write_demo("dance_x", "empty")
    fd = paths.work() / "dance_x" / "farm"
    st = json.loads((fd / "state.json").read_text())
    st["demo"] = False
    st["chunks"] = {"a": {"id": "a", "k0": -90, "k1": 600, "core_k0": -90, "core_k1": 509, "status": "planned", "origin": "s1", "kind": "full"},
                    "b": {"id": "b", "k0": 400, "k1": 1200, "core_k0": 510, "core_k1": 1109, "status": "planned", "origin": "s1", "kind": "full"}}
    st["jobs"] = {}
    touch(fd / "state.json", None, json.dumps(st))
    pod = {"id": "podaaaaaa111", "name": "fit-a", "gpu": "NVIDIA A100-SXM4-80GB", "cost_per_hr": 1.39, "created_ts": time.time() - 600}
    _session(root, "20261008-131657-fit-a", "a", {"fit": ("running", "fit_c8"), "refine": ("pending", None), "quality": ("pending", None)}, pod)
    b = farm_view.build("dance_x", rp={"connection": "ok", "pods": []})
    by = {c["id"]: c for c in b["chunks"]}
    assert by["a"]["status"] == "running" and by["a"]["source"] == "fit_full" and by["b"]["status"] == "planned"
    assert [j["id"] for j in b["jobs"]] == ["a"] and b["jobs"][0]["source"] == "fit_full" and b["jobs"][0]["pod"] == "podaaaaaa111"
    assert [q["id"] for q in b["queue"]] == ["b"]
    assert [p["source"] for p in b["pods"]] == ["fit_full"] and b["pods"][0]["stoppable"] is False
    assert any(c["live"] for c in b["coverage"]["cells"][:15]) and not any(c["live"] for c in b["coverage"]["cells"][40:])


# ------------------------------------------------------------------------------------------------ routes

def test_farm_routes_security_etag_and_validation(client, root, monkeypatch):
    from dancecap.pipeline.monitor import farm_routes
    monkeypatch.setattr(farm_routes, "BUNDLE_TTL_S", 60.0)     # two polls inside the cache window answer the same bundle
    farm_demo.write_demo("farm_demo", "mostly_partial")
    assert client.c.get("/api/farm", base_url="http://evil.example:8790").status_code == 403
    r = client.get("/api/farm?dance=farm_demo")
    assert r.status_code == 200
    d = r.get_json()
    assert d["schema"] == "dancecap.monitor.farm/1" and d["dance"] == "farm_demo" and d["coverage"]["n_cells"] == 190
    assert any(x["id"] == "farm_demo" for x in d["dances"]) and d["monitor"]["port"] == 8790 and "rev" in d
    assert d["commands"]["queue"] is True
    tag = r.headers["ETag"]
    assert client.get("/api/farm?dance=farm_demo", headers={"If-None-Match": tag}).status_code == 304
    assert client.get("/api/farm?dance=nope_nothing").status_code == 404
    assert client.get("/api/farm?dance=../x").status_code in (400, 404)
    assert client.post("/api/farm/pull", {"dance": "farm_demo"}, token=False).status_code == 403     # writes need the token
    assert client.get("/api/farm/log?dance=farm_demo").get_json()["name"] is None
    touch(paths.work() / "farm_demo" / "farm" / "logs" / "watch.log", None, "line one\nline two key rpa_ABCdef123456\n")
    lg = client.get("/api/farm/log?dance=farm_demo").get_json()
    assert lg["name"] == "watch.log" and "line two" in lg["text"] and "rpa_ABCdef" not in lg["text"]
    assert client.get("/api/farm/log?dance=farm_demo&name=../../state.json").status_code == 400


def test_map_png_export(client, root):
    farm_demo.write_demo("farm_demo", "midrun")
    import base64
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
    r = client.post("/api/farm/map-png", {"dance": "farm_demo", "png_b64": base64.b64encode(png).decode()})
    assert r.status_code == 200
    out = list((paths.work() / "farm_demo" / "farm" / "review").glob("monitor_map_*.png"))
    assert len(out) == 1 and out[0].read_bytes() == png
    assert client.post("/api/farm/map-png", {"dance": "farm_demo", "png_b64": base64.b64encode(b"not a png").decode()}).status_code == 400
    assert client.post("/api/farm/map-png", {"dance": "farm_demo", "png_b64": "***"}).status_code == 400


def test_commands_not_installed_are_refused_politely(client, root, monkeypatch):
    farm_demo.write_demo("farm_demo", "midrun")
    monkeypatch.setattr(farm_actions, "available_commands", lambda: {"plan": "farm", "queue": "farm", "coverage": "farm"})
    r = client.post("/api/farm/pull", {"dance": "farm_demo"})
    assert r.status_code == 409 and r.get_json()["error"]["code"] == "not_installed"
    r = client.post("/api/farm/watcher/restart", {"dance": "farm_demo"})
    assert r.status_code == 409 and "farm watch" in r.get_json()["error"]["message"]


# ------------------------------------------------------------------------------------------------ actions

def test_stop_pods_only_touches_this_farms_pods(client, root, monkeypatch):
    farm_demo.write_demo("farm_demo", "midrun")
    started = []
    monkeypatch.setattr(farm_actions, "available_commands", lambda: {})                  # no `down`: the plain pod delete fallback
    monkeypatch.setattr(farm_actions, "start", lambda label, dance, steps, meta=None: started.append((label, steps, meta)) or {"id": "a-x"})
    assert client.post("/api/farm/pods/stop", {"dance": "farm_demo", "pods": ["demo4090a"], "confirm": "stop"}).status_code == 400
    assert client.post("/api/farm/pods/stop", {"dance": "farm_demo", "pods": ["otherpod12345"], "confirm": "STOP"}).get_json()[
        "error"]["code"] == "not_farm_pod"
    # a fit_full pod of the same dance is refused even when it is named
    fd = paths.work() / "farm_demo" / "farm"
    st = json.loads((fd / "state.json").read_text())
    st["pods"]["podfitfull001"] = {"id": "podfitfull001", "state": "running"}
    st["demo"] = False
    touch(fd / "state.json", None, json.dumps(st))
    _session(root, "20261008-131657-fit-a", "a", {"fit": ("running", None)},
             {"id": "podfitfull001", "name": "fit-a", "gpu": "A100", "cost_per_hr": 1.0, "created_ts": time.time()}, take="farm_demo")
    r = client.post("/api/farm/pods/stop", {"dance": "farm_demo", "pods": ["podfitfull001"], "confirm": "STOP"})
    assert r.status_code == 409 and r.get_json()["error"]["code"] == "fit_full_pod" and not started
    r = client.post("/api/farm/pods/stop", {"dance": "farm_demo", "pods": ["demo4090a", "demo4090b"], "confirm": "STOP"})
    assert r.status_code == 202 and started[0][0] == "stop_pods"
    assert [s["argv"][-2:] for s in started[0][1]] == [["delete", "demo4090a"], ["delete", "demo4090b"]]
    # with the watch stage's `down` the stop is a verified pull first, restricted to the named pods
    monkeypatch.setattr(farm_actions, "available_commands", lambda: {"down": "farm"})
    started.clear()
    assert client.post("/api/farm/pods/stop", {"dance": "farm_demo", "pods": ["demo4090a"], "confirm": "STOP"}).status_code == 202
    argv = started[0][1][0]["argv"]
    assert argv[-5:] == ["down", "--pod", "demo4090a", "--abandon"][-5:] or argv[-4:] == ["down", "--pod", "demo4090a", "--abandon"]


def test_fill_gaps_plan_then_approve(client, root, monkeypatch):
    farm_demo.write_demo("farm_demo", "mostly_partial")
    plan = client.post("/api/farm/fill/plan", {"dance": "farm_demo", "pods": 3}).get_json()
    assert plan["demo"] and plan["estimate"]["pods"] == 2 and len(plan["chunks"]) == 2          # 57 empty cells = 2 chunks of ~28 s
    assert all(25 <= c["core_s"] <= 30 for c in plan["chunks"]) and plan["estimate"]["worst_cost_usd"] > plan["estimate"]["expected_cost_usd"] > 0
    assert plan["estimate"]["wall_min"] >= plan["estimate"]["makespan_min"] and plan["plan_hash"]
    assert (farm_actions.actions_dir() / f"plan-{plan['plan_hash']}.json").exists()
    r = client.post("/api/farm/fill/run", {"dance": "farm_demo", "plan_hash": plan["plan_hash"], "approve": {"max_cost_usd": 99}})
    assert r.status_code == 409 and r.get_json()["error"]["code"] == "demo"                      # a demo never launches anything
    # a real (non-demo) dance: the approval must cover the worst case, the plan expires, missing commands are named
    fd = paths.work() / "farm_demo" / "farm"
    st = json.loads((fd / "state.json").read_text())
    st["demo"] = False
    touch(fd / "state.json", None, json.dumps(st))
    started = []
    monkeypatch.setattr(farm_actions, "start", lambda label, dance, steps, meta=None: started.append((label, steps, meta)) or
                        {"id": "a-1", "state": "running"})
    monkeypatch.setattr(farm_actions, "available_commands", lambda: {"queue": "farm", "pods:launch": "pods", "pods:launch-cpu": "pods"})
    pj = farm_actions.actions_dir() / f"plan-{plan['plan_hash']}.json"
    doc = json.loads(pj.read_text())
    doc["demo"] = False
    doc["budget"]["gate"] = "ok"
    pj.write_text(json.dumps(doc))
    low = client.post("/api/farm/fill/run", {"dance": "farm_demo", "plan_hash": plan["plan_hash"], "approve": {"max_cost_usd": 0.5}})
    assert low.status_code == 409 and low.get_json()["error"]["code"] == "approval_too_low"
    ok = client.post("/api/farm/fill/run", {"dance": "farm_demo", "plan_hash": plan["plan_hash"],
                                            "approve": {"max_cost_usd": doc["estimate"]["worst_cost_usd"]}})
    assert ok.status_code == 202, ok.get_json()
    label, steps, meta = started[0]
    names = [s["label"] for s in steps]
    assert label == "fill" and names[:3] == ["claim live fit_full chunks", "farm plan", "farm queue"] and "launch GPU pods" in names
    launch = next(s for s in steps if s["label"] == "launch GPU pods")["argv"]
    assert launch[1:5] == ["-m", "dancecap.farm.pods", "launch", "-n"] and "--max-cost" in launch
    monkeypatch.setattr(farm_actions, "available_commands", lambda: {"up": "farm", "queue": "farm"})
    started.clear()
    up = client.post("/api/farm/fill/run", {"dance": "farm_demo", "plan_hash": plan["plan_hash"],
                                            "approve": {"max_cost_usd": doc["estimate"]["worst_cost_usd"]}})
    assert up.status_code == 202
    names = [s["label"] for s in started[0][1]]
    assert names == ["claim live fit_full chunks", "farm up (plan, queue, launch, watcher)"]
    argv = started[0][1][1]["argv"]
    assert argv[argv.index("up") + 1:argv.index("up") + 3] == ["--pods", "2"] and "--only" in argv
    assert set(argv[argv.index("--only") + 1].split(",")) == {c["id"] for c in doc["chunks"]} | {"final"}
    assert "--budget" in argv and "--minutes" in argv and "--hard-minutes" in argv
    doc["created"] = time.time() - 3600
    pj.write_text(json.dumps(doc))
    assert client.post("/api/farm/fill/run", {"dance": "farm_demo", "plan_hash": plan["plan_hash"],
                                              "approve": {"max_cost_usd": 99}}).get_json()["error"]["code"] == "plan_expired"
    assert client.post("/api/farm/fill/run", {"dance": "farm_demo", "plan_hash": "0" * 16, "approve": {}}).status_code == 404


def test_fill_plan_excludes_live_fit_full_chunks(root, monkeypatch):
    """The four S1 segments are being fitted by fit_full sessions: Fill gaps must not plan, queue or launch them."""
    farm_demo.write_demo("dance_x", "empty")
    fd = paths.work() / "dance_x" / "farm"
    st = json.loads((fd / "state.json").read_text())
    st["demo"] = False
    st["chunks"] = {"a": {"id": "a", "status": "planned", "core_k0": 0, "core_k1": 899, "core_s": 30},
                    "b": {"id": "b", "status": "planned", "core_k0": 900, "core_k1": 1799, "core_s": 30}}
    touch(fd / "state.json", None, json.dumps(st))
    _session(root, "20261008-131657-fit-a", "a", {"fit": ("running", "fit_c8")},
             {"id": "podaaaaaa111", "name": "fit-a", "gpu": "A100", "cost_per_hr": 1.39, "created_ts": time.time()})
    core = {"report": {"gaps": [], "gap_s": []}, "chunks": [],
            "s1": [{"id": "a", "T": 1119, "core_s": 33.3, "kind": "full", "estimate": {"total_min": 50.8}},
                   {"id": "b", "T": 1286, "core_s": 36.8, "kind": "full", "estimate": {"total_min": 57.8}}],
            "jobs": [{"id": "final"}]}
    monkeypatch.setattr(farm_actions, "_run_json", lambda argv, timeout: (0, core, ""))
    monkeypatch.setattr(farm_actions, "_gpu_offers", lambda: ([{"gpu": "NVIDIA GeForce RTX 4090", "price": 0.69, "stock": "Low"}], None))
    monkeypatch.setattr(farm_actions, "_rp_cached", lambda: None)
    plan = farm_actions.fill_plan("dance_x")
    assert [c["id"] for c in plan["chunks"]] == ["b"]
    assert plan["excluded"] == [{"id": "a", "why": "a fit_full session is fitting it right now", "status": "running"}]
    assert plan["live_fit_full"] == ["a"] and plan["gpu"]["gpu"].endswith("4090")
    assert plan["commands"][0].endswith("_claim_live dance_x")                     # the live chunk is claimed before queueing
    # when everything is claimed there is nothing to launch
    core["s1"] = core["s1"][:1]
    plan = farm_actions.fill_plan("dance_x")
    assert plan["nothing_to_do"] and plan["chunks"] == [] and "Nothing to launch" in plan["message"]


def test_fill_plan_finished_chunks_leave_only_the_cpu_stitch_job(root, monkeypatch):
    """The fits of a-d finished and were pulled: nothing for a GPU pod (core: no step left), but the dance's stitch + analysis
    job (one CPU pod) is still open. While a fit_full session is still fitting, that job is not planned yet."""
    farm_demo.write_demo("dance_x", "empty")
    fd = paths.work() / "dance_x" / "farm"
    st = json.loads((fd / "state.json").read_text())
    st["demo"] = False
    st["chunks"] = {k: {"id": k, "status": "planned", "core_s": 30} for k in "abc"}
    st["jobs"] = {"final": {"id": "final", "status": "planned", "after": []}}
    touch(fd / "state.json", None, json.dumps(st))
    core = {"report": {"gaps": [], "gap_s": []}, "chunks": [],
            "s1": [{"id": k, "T": 1000, "core_s": 30, "kind": "full", "estimate": {"total_min": 0}} for k in "abc"],
            "jobs": [{"id": "final", "kind": "stitch_analyse"}]}
    monkeypatch.setattr(farm_actions, "_run_json", lambda argv, timeout: (0, core, ""))
    monkeypatch.setattr(farm_actions, "_gpu_offers", lambda: ([{"gpu": "NVIDIA GeForce RTX 4090", "price": 0.69, "stock": "Low"}], None))
    monkeypatch.setattr(farm_actions, "_rp_cached", lambda: None)
    monkeypatch.setattr(farm_actions, "available_commands", lambda: {"up": "farm", "queue": "farm", "watch": "farm"})
    plan = farm_actions.fill_plan("dance_x")
    assert plan["chunks"] == [] and plan["final_cpu"] is True and plan["nothing_to_do"] is False
    assert {x["id"] for x in plan["excluded"]} == {"a", "b", "c"} and plan["estimate"]["pods"] == 0
    assert 0 < plan["estimate"]["worst_cost_usd"] <= 1.0 and plan["estimate"]["wall_min"] < 30          # one small CPU pod, no GPU
    assert plan["commands"][-1].endswith("up --pods 1 --budget %s --only final --exclude " % plan["estimate"]["worst_cost_usd"])         or "up --pods 1" in plan["commands"][-1] and "--only final" in plan["commands"][-1]
    assert not any("--minutes" in c for c in plan["commands"])                        # no GPU pod: no GPU kill-switch flags
    monkeypatch.setattr(farm_actions, "available_commands", lambda: {"queue": "farm", "pods:launch-cpu": "pods"})   # an older tree without `up`
    old = farm_actions.fill_plan("dance_x")
    assert any("launch-cpu" in c for c in old["commands"]) and not any("pods launch -n" in c for c in old["commands"])
    monkeypatch.setattr(farm_actions, "available_commands", lambda: {"up": "farm", "queue": "farm", "watch": "farm"})
    _session(root, "20261008-131657-fit-a", "a", {"fit": ("running", "fit_c8")},
             {"id": "podaaaaaa111", "name": "fit-a", "gpu": "A100", "cost_per_hr": 1.39, "created_ts": time.time()}, take="dance_x")
    plan = farm_actions.fill_plan("dance_x")
    assert plan["final_cpu"] is False and plan["final_blocked"] is True and plan["nothing_to_do"] is True


def test_estimate_plan_and_lpt():
    make, loads = farm_actions.lpt_makespan([45, 40, 40, 38], 2)
    assert make == 83 and sorted(sum(l) for l in loads) == [80, 83]
    e = farm_actions.estimate_plan([{"id": "a", "est_min": 40}, {"id": "b", "est_min": 40}], pods=5, price_hr=0.7, final_cpu=True)
    assert e["pods"] == 2 and e["makespan_min"] == 40 and e["wall_min"] == 40 + farm_actions.START_OVERHEAD_MIN + farm_actions.FINAL_CPU_MIN
    assert e["expected_cost_usd"] == pytest.approx(80 * 0.7 / 60 * 1.08 + 0.07, abs=0.01) and e["worst_cost_usd"] > 2 * 0.7 * 60 / 60


def test_available_commands_reads_code_not_docstrings():
    names = farm_actions.available_commands()
    assert "coverage" in names and "plan" in names and "pods:launch" in names
    assert "launch" not in names or names["launch"] == "farm"      # cli.py's docstring example is not a command
    src = 'def f(sub):\n    """sub.add_parser("ghost")"""\n    sub.add_parser("real")\n'
    assert farm_actions._parser_names(src) == (["real"], False)


def test_coverage_refresh_is_throttled_and_never_doubles(root, monkeypatch):
    farm_demo.write_demo("farm_demo", "midrun")
    calls = []
    monkeypatch.setattr(farm_actions, "start", lambda label, dance, steps, meta=None: calls.append(steps) or {"id": "a-1"})
    farm_actions._last_coverage_try.clear()
    assert farm_actions.refresh_coverage("farm_demo") is not None
    assert farm_actions.refresh_coverage("farm_demo") is None                 # inside the 2 min throttle
    assert farm_actions.refresh_coverage("farm_demo", force=True) is not None
    assert calls[0][0]["argv"][-3:] == ["--json", "coverage"] or calls[0][0]["argv"][-2:] == ["--json", "coverage"]


def test_exec_wrapper_runs_steps_and_logs(root):
    farm_demo.write_demo("farm_demo", "midrun")
    rec = {"id": "a-20261008-120000-abcd", "label": "t", "dance": "farm_demo", "created": time.time(), "started": None, "ended": None, "rc": None,
           "step_idx": None, "log": str(paths.work() / "farm_demo" / "farm" / "logs" / "a-20261008-120000-abcd.log"),
           "log_name": "a-20261008-120000-abcd.log", "meta": {},
           "steps": [{"label": "one", "argv": [sys.executable, "-c", "print('hello rpa_SECRET123')"]},
                     {"label": "two", "argv": [sys.executable, "-c", "import sys; sys.exit(3)"]},
                     {"label": "never", "argv": [sys.executable, "-c", "print('no')"]}]}
    p = farm_actions.actions_dir() / f"{rec['id']}.json"
    touch(p, None, json.dumps(rec))
    assert farm_actions._exec(p) == 3
    done = json.loads(p.read_text())
    assert done["rc"] == 3 and done["step_idx"] == 1 and "step 2" in done["error"]
    text = Path(rec["log"]).read_text()
    assert "hello" in text and "never" not in text
    assert farm_actions.describe(done)["state"] == "failed"
    assert farm_actions.recent("farm_demo")[0]["id"] == rec["id"]


# ------------------------------------------------------------------------------------------------ legacy archive

def test_archive_legacy_moves_and_keeps_live_runs(root):
    base = paths.work() / "pipeline"
    for rid, rc in (("r-20261007-142829-07f0", 0), ("r-20261007-173143-f143", 1)):
        touch(base / "runs" / rid / "run.json", None, "{}")
        touch(base / "runs" / rid / "rc.json", None, json.dumps({"rc": rc, "ended": 1}))
    live = base / "runs" / "r-20261007-151009-6d2f"
    touch(live / "run.json", None, "{}")
    import psutil
    touch(live / "pid.json", None, json.dumps({"pid": os.getpid(), "create_time": psutil.Process().create_time()}))
    touch(base / "plans" / "abc.json", None, "{}")
    touch(base / "runs_archive" / "202610" / "r-old" / "run.json", None, "{}")
    touch(paths.work() / "t1" / "pipeline" / "events.jsonl", None, "{}\n")
    touch(paths.work() / "t1" / "pipeline" / "gates.json", None, "{}")                        # stage logic reads it: stays
    touch(paths.runpod_dir() / "sessions" / "20261008-131657-fit-a" / "state.json", None, "{}")  # live sessions: never moved
    dry = legacy_archive.archive_legacy("2026-10-08", dry_run=True)
    assert dry["counts"] == {"runs": 2, "plans": 1, "runs_archive": 1, "takes": 1} and (base / "plans" / "abc.json").exists()
    rep = legacy_archive.archive_legacy("2026-10-08")
    dest = base / "archive" / "2026-10-08"
    assert rep["counts"] == {"runs": 2, "plans": 1, "runs_archive": 1, "takes": 1}
    assert (dest / "runs" / "r-20261007-142829-07f0" / "rc.json").exists() and (dest / "plans" / "abc.json").exists()
    assert (dest / "takes" / "t1" / "events.jsonl").exists() and (dest / "MANIFEST.json").exists()
    assert live.exists() and rep["skipped"] == [{"run": "r-20261007-151009-6d2f", "why": "still running"}]
    assert (paths.work() / "t1" / "pipeline" / "gates.json").exists()
    assert (paths.runpod_dir() / "sessions" / "20261008-131657-fit-a" / "state.json").exists()
    assert not list((base / "runs").glob("r-20261007-1[47]*"))
    # a second archive on the same day never overwrites
    touch(base / "plans" / "abc.json", None, '{"second": 1}')
    legacy_archive.archive_legacy("2026-10-08")
    assert json.loads((dest / "plans" / "abc.json").read_text()) == {} and (dest / "plans" / "abc.json.1").exists()
