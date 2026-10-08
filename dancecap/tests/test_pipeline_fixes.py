"""Regression tests for the review of the pipeline backend (2026-10-07): run states, camera types vs the SfM map,
safe stop, wizard takes past intake (audio_sync, single-view identity clock), dry runs, honest pod counts, review
gates, window validation, cross-process locks, applied camera types from report.json, trigger layer ids, orphan-only
cleanup, approval replay, status cost with many runs, and the smaller API input checks.
No pods, no Runpod calls, no real take touched (synthetic package root in a temp dir)."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pipeline_fixtures import (FakePoller, have_ffmpeg, make_root, make_take, make_video, touch,  # noqa: E402
                                wait_for, NO_WINDOW)
from dancecap.pipeline import (audio_sync, camtypes, gates, intake, locks, paths, plan as planmod,  # noqa: E402
                               registry, runs, status, takes, triggers)
from dancecap.pipeline.orchestrator import Orchestrator, load_plan_for_run  # noqa: E402

BASE = "http://127.0.0.1:8790"

FAKE_STAGE = '''
import os, sys, time
from pathlib import Path
root = Path(os.environ["DANCECAP_PIPELINE_ROOT"])
args = sys.argv[1:]
take = args[0]
outs = [a for a in args[1:] if not a.startswith("--") and "=" not in a]
opt = dict(a[2:].split("=", 1) for a in args[1:] if a.startswith("--") and "=" in a)
rc = int(opt.get("rc", 0))
try:
    print(f"[fake] {take} start", flush=True)
    t0 = time.time()
    while time.time() - t0 < float(opt.get("sleep", 0)):
        time.sleep(0.05)
    if rc == 0:
        for o in outs:
            p = root / "work" / take / o
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("{}")
finally:
    if opt.get("marker"):
        Path(opt["marker"]).write_text("finally ran")
sys.exit(rc)
'''

# the real orchestrator inside a run of kind "pipeline", with fake stage commands (registry overlay patched here)
FAKE_PIPE = '''
import json, sys
from pathlib import Path
from dancecap.pipeline import registry
from dancecap.pipeline.orchestrator import Orchestrator
plan = json.loads(Path(sys.argv[2]).read_text())
for sid, argv in json.loads(Path(sys.argv[3]).read_text()).items():
    registry.OVERLAY[sid]["commands"] = [argv]
sys.exit(Orchestrator(sys.argv[1], plan, poll_s=0.1).run())
'''

# a stage that starts a pod-runner-looking child and a plain child, then waits on the plain one
FAKE_SPAWN = '''
import json, subprocess, sys
from pathlib import Path
sess = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)", "dancecap", "remote", "run"])
plain = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
Path(sys.argv[1]).write_text(json.dumps({"sess": sess.pid, "plain": plain.pid}))
plain.wait()
'''


@pytest.fixture()
def root(tmp_path, monkeypatch):
    r = make_root(tmp_path)
    mods = tmp_path / "fakemods"
    mods.mkdir()
    for name, src in (("fake_stage", FAKE_STAGE), ("fake_pipe", FAKE_PIPE), ("fake_spawn", FAKE_SPAWN)):
        (mods / f"{name}.py").write_text(src, encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(mods))
    monkeypatch.setenv("DANCECAP_STOP_GRACE_S", "10")
    make_take(r)
    yield r
    paths.set_root(None)


def fake_run_dir(kind: str, take: str, stages: list[str], rc: int | None, stop_at: float | None = None,
                 ended: float | None = None, created: float | None = None) -> str:
    while True:
        rid = runs.new_id()
        d = paths.runs_dir() / rid
        if not d.exists():
            break
    d.mkdir(parents=True)
    now = time.time()
    (d / "run.json").write_text(json.dumps({"id": rid, "kind": kind, "take": take, "stages": stages,
                                            "created": created or now - 30}))
    (d / "pid.json").write_text(json.dumps({"pid": 999999, "create_time": None, "started": now - 30}))
    if stop_at is not None:
        (d / "stop.json").write_text(json.dumps({"mode": "now", "at": stop_at}))
    if rc is not None:
        (d / "rc.json").write_text(json.dumps({"rc": rc, "ended": ended or now - 1, "error": None}))
    return rid


def st_of(take):
    snap = status.snapshot(take, with_activity=False)
    return {x["id"]: x for x in snap["stages"]}, snap


def confirm_all(take, cams=("01", "02", "03"), mount="handheld", zoom=True):
    for c in cams:
        _, etag = camtypes.load(take, [c])
        camtypes.set_type(take, c, mount, zoom, etag)


# ------------------------------------------------------------------------------------------- 1 / 5: run states

def test_ended_runs_are_never_active_and_dry_runs_never_fail_a_stage(root):
    rid = fake_run_dir("pipeline", "synth", ["probe", "ingest"], rc=3)
    r = runs.describe(rid)
    assert r["state"] == "waiting_human" and r["stop_modes"] == []
    assert runs.list_runs(take="synth", active=True) == []
    dry = fake_run_dir("dry_run", "synth", ["probe", "ingest"], rc=3)
    assert runs.describe(dry)["state"] == "waiting_human"
    s, _ = st_of("synth")
    assert s["ingest"]["status"] == "ready" and s["probe"]["status"] == "ready"  # never "failed" by a dry run
    # a tool stopped by the user that swallowed the interrupt (rc 0) reads as stopped
    t = fake_run_dir("tool", "synth", ["narrate"], rc=0, stop_at=time.time() - 5, ended=time.time() - 4)
    assert runs.describe(t)["state"] == "stopped"
    # a running pipeline whose current step is a pod session offers no 'now'
    assert runs.stop_modes({"kind": "pipeline"}, "running", {"session_active": True}) == ["after_step"]
    assert runs.stop_modes({"kind": "pipeline"}, "running", {}) == ["after_step", "now"]
    assert runs.stop_modes({"kind": "session"}, "running", {}) == []
    assert runs.stop_modes({"kind": "pipeline"}, "stopping", {}, "after_step") == ["now"]  # escalate a slow step
    assert runs.stop_modes({"kind": "pipeline"}, "stopping", {"session_active": True}, "after_step") == []
    assert runs.stop_modes({"kind": "pipeline"}, "stopping", {}, "now") == []


# ------------------------------------------------------------------------------------------- 2: cam_map

def test_camera_types_never_touch_the_sfm_map_and_the_map_is_kept_unless_opted_in(root):
    w = root / "work" / "synth"
    t0 = time.time() - 1000
    for f in ("frames/01/frames.json", "keyframes/01/keyframes.json", "people/01.npz", "people/report.json"):
        touch(w / f, t0 - 100)
    for f in ("cameras/sfm/images.bin", "cameras/sfm_work/map.json", "cameras/cache/loc_01.npz",
              "cameras/world_from_sfm.json", "cameras/report.json"):
        touch(w / f, t0)
    confirm_all("synth")
    s, _ = st_of("synth")
    stat = {k: v["status"] for k, v in s.items()}
    vst = registry.variant_stages("multi")
    inval = camtypes.invalidated_by_change("synth", "01", "tripod", False, stat,
                                           lambda roots: registry.descendants(vst, roots))
    assert "cam_loc" in inval and "cam_world" in inval and "cam_map" not in inval
    _, etag = camtypes.load("synth", ["01"])
    camtypes.set_type("synth", "01", "tripod", False, etag)
    s, _ = st_of("synth")
    assert s["cam_map"]["status"] == "done" and s["cam_loc"]["status"] == "stale"
    # the map is stale by mtime (detect rewritten after it): a plan keeps it unless rebuild_map is set
    touch(w / "people" / "report.json")
    s, _ = st_of("synth")
    assert s["cam_map"]["status"] == "stale"
    p = planmod.make_plan("synth", {"target": "cam_map", "scope": "only"})
    (step,) = p["steps"]
    assert step["action"] == "skip_done" and step["reason"].startswith("kept:")
    assert any("rebuild_map" in w_ for w_ in p["warnings"])
    p = planmod.make_plan("synth", {"target": "cam_map", "scope": "only", "mode": "force"})
    assert p["steps"][0]["action"] == "skip_done"  # force alone never rebuilds it
    p = planmod.make_plan("synth", {"target": "cam_map", "scope": "only", "options": {"rebuild_map": True}})
    assert p["steps"][0]["action"] == "rerun_stale"
    with pytest.raises(planmod.PlanError):
        planmod.make_plan("synth", {"target": "cam_map", "options": {"rebuild_map": "yes"}})
    # a reused map is rebuilt by reusing it again, never by a fresh SfM
    assert registry.commands("cam_map", {"take": "synth"}) == [["-m", "dancecap", "cameras", "synth", "--stages", "map"]]
    touch(w / "cameras" / "sfm_work" / "map_images.json", text=json.dumps({"reused_from": "base_take"}))
    assert registry.commands("cam_map", {"take": "synth"}) == [
        ["-m", "dancecap.cameras.reuse", "map", "synth", "--from", "base_take"]]


def test_applied_types_come_from_the_cameras_report(root):
    w = root / "work" / "synth"
    t0 = time.time() - 1000
    for f in ("cameras/sfm/images.bin", "cameras/sfm_work/map.json", "cameras/cache/loc_01.npz",
              "cameras/world_from_sfm.json"):
        touch(w / f, t0)
    touch(w / "cameras" / "report.json", t0, text=json.dumps({}))
    _, etag = camtypes.load("synth", ["01"])
    camtypes.set_type("synth", "01", "tripod", False, etag)
    s, _ = st_of("synth")
    assert s["cam_loc"]["status"] == "stale"
    # the cameras stage ran again from the CLI (not the orchestrator): its report records what it applied
    rep = {"camera_types": {"policy": "user", "applied": {
        "01": {"cam": "01", "mount": "tripod", "zoom": False, "class": "tripod"},
        "02": {"cam": "02", "mount": "handheld", "zoom": True, "class": "handheld_zoom"},
        "03": {"cam": "03", "mount": "handheld", "zoom": True, "class": "handheld_zoom"}}}}
    touch(w / "cameras" / "report.json", text=json.dumps(rep))
    s, _ = st_of("synth")
    assert not any("camera type" in x for x in s["cam_loc"]["outputs"]["stale_because"])
    assert not any("camera type" in x for x in s["cam_world"]["outputs"]["stale_because"])


def test_the_adapter_runs_the_shared_camera_types_module_on_a_test_tree(root):
    ct = camtypes._shared_module()
    assert ct is not None, "dancecap.camera_types must be importable"
    before = Path(ct.root()).resolve()
    _, etag = camtypes.load("synth", ["02"])
    t, _ = camtypes.set_type("synth", "02", "tripod", True, etag)
    assert t["state"] == "confirmed" and t["effective"]["class"] == "tripod_zoom"
    assert Path(ct.root()).resolve() == before  # the shared module's root is restored after each call
    with pytest.raises(camtypes.StaleEtag):
        camtypes.set_type("synth", "02", "handheld", True, etag)


# ------------------------------------------------------------------------------------------- 3: safe stop

def test_stop_now_of_a_pipeline_lets_the_orchestrator_stop_its_children(root, tmp_path):
    marker = tmp_path / "marker.txt"
    p = planmod.make_plan("synth", {"target": "ingest"})
    assert [s["stage"] for s in p["steps"] if s["action"] == "run"] == ["ingest"]
    pf = tmp_path / "plan.json"
    pf.write_text(json.dumps(p))
    cf = tmp_path / "cmds.json"
    cf.write_text(json.dumps({"ingest": ["-m", "fake_stage", "{take}", "frames/01/frames.json", "--sleep=60",
                                         f"--marker={marker}"]}))
    pipe = runs.start("pipeline", ["-m", "fake_pipe", "synth", str(pf), str(cf)], take="synth", stages=["ingest"])

    def child():
        rr = [r for r in runs.list_runs(take="synth", kind="stage") if r.get("parent") == pipe["id"]]
        return rr[0] if rr and rr[0]["state"] == "running" else None
    c = wait_for(child, 60)
    assert c, runs.log_chunk(runs.run_dir(pipe["id"]) / "output.log")["lines"]
    assert locks.holder(locks.take_lock("synth")) is not None  # the orchestrator holds the take
    runs.request_stop(pipe["id"], "now")
    final = wait_for(lambda: runs.describe(pipe["id"], fresh=True)["state"] in ("stopped", "failed", "done")
                     and runs.describe(pipe["id"], fresh=True), 90)
    assert final and final["state"] == "stopped" and final["rc"] == 130
    cf_ = runs.describe(c["id"], fresh=True)
    assert cf_["state"] == "stopped" and cf_["rc"] == 130          # the child wrote its own rc.json
    assert marker.read_text() == "finally ran"                      # its cleanup ran (never hard-killed first)
    assert locks.holder(locks.take_lock("synth")) is None           # the take is free again
    st = json.loads((paths.pipeline_dir("synth") / "state.json").read_text())
    assert st["status"] == "stopped" and st["stages"]["ingest"]["state"] == "stopped"


def test_pod_runners_are_never_killed_by_a_stop(root, tmp_path):
    info = tmp_path / "pids.json"
    r = runs.start("stage", ["-m", "fake_spawn", str(info)], take="synth", stages=["ingest"])
    assert wait_for(lambda: info.exists() and runs.describe(r["id"], fresh=True)["state"] == "running", 60)
    pids = json.loads(info.read_text())
    import psutil
    try:
        runs.request_stop(r["id"], "now")
        final = wait_for(lambda: runs.describe(r["id"], fresh=True)["state"] in ("stopped", "failed")
                         and runs.describe(r["id"], fresh=True), 60)
        assert final and final["state"] == "stopped"
        assert not psutil.pid_exists(pids["plain"]) or psutil.Process(pids["plain"]).status() == psutil.STATUS_ZOMBIE
        assert psutil.pid_exists(pids["sess"]), "a pod-runner process must survive a stop"
    finally:
        from dancecap.pipeline.util import kill_tree
        kill_tree(pids["sess"])
    # a run of kind session ignores stop.json altogether (only its pod's typed delete stops it)
    s = runs.start("session", ["-m", "fake_stage", "synth", "--sleep=8"], take="synth", stages=["texture"])
    assert wait_for(lambda: runs.describe(s["id"], fresh=True)["state"] == "running", 60)
    assert runs.describe(s["id"], fresh=True)["stop_modes"] == []
    (runs.run_dir(s["id"]) / "stop.json").write_text(json.dumps({"mode": "now", "at": time.time()}))
    time.sleep(3)
    assert runs.describe(s["id"], fresh=True)["state"] in ("stopping", "running")
    assert wait_for(lambda: (runs.run_dir(s["id"]) / "rc.json").exists(), 30)
    assert json.loads((runs.run_dir(s["id"]) / "rc.json").read_text())["rc"] == 0  # it ran to its end


# ------------------------------------------------------------------------------------------- 4: wizard takes

def _req(vids, **kw):
    r = {"take": "solo_take", "videos": [{"cam": f"{i + 1:02d}", "path": str(p)} for i, p in enumerate(vids)],
         "dancers": {"lead": {"name": "Kai", "stature_m": "180 cm"}, "follow": {"name": "Lia", "stature_m": "165 cm"}},
         "dance": {"level": "novice", "category": "lesson"}, "consent": {}}
    r.update(kw)
    return r


def test_single_view_wizard_take_gets_its_identity_clock(root, tmp_path):
    from _pipeline_fixtures import fake_probe
    v = tmp_path / "src" / "solo" / "01-front.mp4"
    v.parent.mkdir(parents=True)
    v.write_bytes(b"x")
    sug = {"mount": "tripod", "zoom": False, "confidence": 0.8, "by": "probe_a", "evidence": {"parallax_px": 0.2}}
    req = _req([v])
    req["videos"][0]["suggestion"] = sug
    res = intake.create(req, probe_fn=fake_probe(seconds=42.0))
    assert res["variant"] == "single"
    al = json.loads((paths.take_work("solo_take") / "clock" / "alignment.json").read_text())
    assert al["sources"][0]["id"] == "01" and al["sources"][0]["slope"] == 1.0 and al["sources"][0]["offset"] == 0.0
    assert al["sources"][0]["name"] == "01-front.mp4" and al["maximum_overlap"][0]["end"] == pytest.approx(42.0)
    # the staged probe-A suggestion is in the TOML (pre-filled, not confirmed)
    t = camtypes.load("solo_take", ["01"])[0][0]
    assert t["state"] == "suggested" and t["mount"] == "tripod" and t["effective"]["class"] == "handheld_zoom"
    s, snap = st_of("solo_take")
    assert s["window"]["status"] == "blocked" and "0.00-42.00" in s["window"]["status_reason"]
    assert s["ingest"]["status"] == "waiting"
    # once the window is set, the take loads with its clock exactly as every dancecap stage loads it
    from dancecap.pipeline import tomledit
    from dancecap.take import Take
    tomledit.write("solo_take", [(None, {"window": [1.0, 40.0]})], None)
    s, _ = st_of("solo_take")
    assert s["window"]["status"] == "done" and s["ingest"]["status"] == "ready"
    tk = Take.load(paths.take_toml("solo_take"))
    assert [x.id for x in tk.sources] == ["01"] and tk.sources[0].slope == 1.0 and tk.window == (1.0, 40.0)
    assert tk.sources[0].path.resolve() == v.resolve()


@pytest.mark.skipif(not have_ffmpeg(), reason="ffmpeg not installed")
def test_audio_sync_recovers_the_clocks_of_real_videos(root, tmp_path):
    d = tmp_path / "src" / "trio"
    d.mkdir(parents=True)
    master = d / "master.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    "anoisesrc=d=40:c=pink:r=16000:a=0.5:seed=7", str(master)],
                   check=True, capture_output=True, creationflags=NO_WINDOW)
    starts = {"01": 0.0, "02": 3.25, "03": 7.5}  # source t = 0 is master t = start: reference = longest (01)
    for cam, st in starts.items():
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=96x64:rate=15",
                        "-ss", f"{st}", "-i", str(master), "-t", f"{30 - st:.2f}", "-map", "0:v", "-map", "1:a",
                        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac",
                        str(d / f"{cam}-cam.mp4")], check=True, capture_output=True, creationflags=NO_WINDOW)
    from dancecap.pipeline import probe as probemod
    req = _req([d / f"{c}-cam.mp4" for c in starts], take="trio_take")
    res = intake.create(req, probe_fn=probemod.probe_file)
    assert res["variant"] == "multi"
    assert not (paths.take_work("trio_take") / "clock" / "alignment.json").exists()  # multi: audio_sync writes it
    s, _ = st_of("trio_take")
    assert s["audio_sync"]["status"] == "ready" and s["ingest"]["status"] == "waiting"
    doc = audio_sync.run("trio_take", log=lambda *_: None)
    rows = {r["id"]: r for r in doc["sources"]}
    assert doc["reference"] == "01-cam.mp4" and all(r["accepted"] for r in rows.values())
    for cam, st in starts.items():
        assert rows[cam]["offset"] == pytest.approx(st, abs=0.01), cam   # ref_time = source_time + start
        assert rows[cam]["slope"] == pytest.approx(1.0, abs=1e-3)
    assert doc["maximum_overlap"][0]["count"] == 3
    s, _ = st_of("trio_take")
    assert s["audio_sync"]["status"] == "done" and s["window"]["status"] == "blocked"
    assert "suggested: the maximum overlap 7.50" in s["window"]["status_reason"]
    # an existing clock file outside work/<take>/ is never overwritten
    with pytest.raises(SystemExit):
        audio_sync.run("synth", log=lambda *_: None)


# ------------------------------------------------------------------------------------------- 6: pod counts

def test_s1_prices_what_the_job_runs(root):
    # S1: the front job re-runs every stage without resume markers, whatever was selected
    w = root / "work" / "synth"
    for c in ("01", "02", "03"):
        touch(w / "cameras" / "cache" / f"loc_{c}.npz")
        touch(w / "cameras" / f"{c}.npz")
    touch(w / "mamma" / "ma_2d_native" / "01.manifest.json")
    rr = registry.session_job_reruns("S1_front", "synth", ["01", "02", "03"])
    assert rr["dancers_sam"] == 1.0 and "cam_loc" not in rr and rr["ma2d"] == pytest.approx(2 / 3, abs=1e-3)
    assert rr["cam_world"] == rr["sync"] == rr["scale_chain"] == 1.0


def test_plans_and_estimates_have_no_hands_stage(root):
    """The hands steps (hands_survey, hands_pod, hands_build; pod session S3_hands) were retired by the user on
    2026-10-07: a plan of the whole DAG has no hands step and no secure-cloud pod, and neither the plan totals nor
    the take's remaining time / pod cost carry their estimates (~10 h + ~8 h on larissa_kadu_full)."""
    retired = {"hands_survey", "hands_pod", "hands_build"}
    w = root / "work" / "synth"
    for f in ("smplx/lead.npz", "smplx/follow.npz", "mamma/select.json"):
        touch(w / f, time.time() - 500)  # a fitted body: the detail stages are next
    for req in ({"scope": "all"}, {"target": "demo_export", "scope": "upto"}):
        p = planmod.make_plan("synth", req)
        assert not retired & {st["stage"] for st in p["steps"]}, req
        assert all(s["group"] != "S3_hands" and s["cloud"] != "SECURE" for s in p["sessions"]), req
        assert all(st["runs_on"] != "pod_secure" for st in p["steps"]), req
    s, snap = st_of("synth")
    assert not retired & set(s) and not retired & set(snap["critical_path"])
    assert "S3_hands" not in [g["id"] for g in snap["groups"]]
    assert not any(b["stage"] in retired for b in snap["branches"])  # no 'HaMeR 6 GB download' risk branch
    todo = [x for x in snap["stages"] if x["status"] not in ("done", "skipped", "n/a", "planned")]
    assert snap["pod_cost_remaining_usd"]["p50"] == pytest.approx(sum(x["cost"]["p50_usd"] for x in todo), abs=0.02)
    # soft deps that pointed at the hands build are gone: texture, timing, counterbalance, the export, shoe_size
    for sid in ("texture", "timing", "counterbalance", "demo_export", "shoe_size"):
        assert not retired & set(s[sid]["deps"] + s[sid]["soft_deps"]), sid


# ------------------------------------------------------------------------------------------- 7: review gates

def _gate_take(root):
    w = root / "work" / "synth"
    t0 = time.time() - 500
    for f in ("smplx/lead.npz", "smplx/follow.npz", "mamma/select.json"):
        touch(w / f, t0)
    confirm_all("synth")


def test_fit_qa_is_a_real_review_gate(root, monkeypatch):
    _gate_take(root)
    cmds = registry.OVERLAY["fit_qa"]["commands"]
    assert ["-m", "dancecap.cameras.world_scale", "check", "{take}", "--strict"] in cmds
    s, _ = st_of("synth")
    assert s["fit_qa"]["status"] == "ready" and s["texture"]["status"] == "waiting"
    # a plan through the gate halts there: nothing after it is planned, priced or approved
    p = planmod.make_plan("synth", {"target": "texture"})
    acts = {x["stage"]: x for x in p["steps"]}
    assert acts["fit_qa"]["action"] == "run" and acts["texture"]["action"] == "blocked"
    assert "review gate at fit_qa" in acts["texture"]["reason"]
    assert p["review_gates"] == ["fit_qa"] and "fit_qa" in p["halts_at"] and p["sessions"] == []
    # the orchestrator runs fit_qa, opens the gate and stops before its descendants (rc 3)
    monkeypatch.setitem(registry.OVERLAY["fit_qa"], "commands", [["-m", "fake_stage", "{take}",
                                                                   "mamma/chunks_seams.json"]])
    child = next(c for c in registry.descendants(registry.variant_stages("multi"), ["fit_qa"])
                 if registry.get(c).get("runnable") == "local" and "fit_qa" in registry.get(c)["deps"])
    monkeypatch.setitem(registry.OVERLAY[child], "commands", [["-m", "fake_stage", "{take}", "x.json"]])
    step = {"est": {"p50_s": 1, "p90_s": 1}, "cost": {"p50_usd": 0, "p90_usd": 0}, "session": None, "status": "ready"}
    plan = dict(p, steps=[dict(step, stage="fit_qa", action="run", runs_on="local"),
                          dict(step, stage=child, action="run", runs_on="local")], sessions=[])
    o = Orchestrator("synth", plan, poll_s=0.1)
    assert o.run() == 3
    assert o.finished["fit_qa"] == "gate" and o.finished[child] == "skipped"
    st = json.loads((paths.pipeline_dir("synth") / "state.json").read_text())
    assert st["status"] == "waiting_human" and any(g["stage"] == "fit_qa" and g["review"] for g in st["gate"])
    assert "review of fit_qa" in st["stages"][child]["reason"]
    s, _ = st_of("synth")
    assert s["fit_qa"]["status"] == "blocked" and "confirm_gate" in s["fit_qa"]["actions"]
    p2 = planmod.make_plan("synth", {"target": "texture"})
    assert {x["stage"]: x["action"] for x in p2["steps"]}["texture"] == "blocked"
    # reject: the stage is stale (re-run it); ok: it is done and the stages after it can be planned
    gates.confirm("synth", "fit_qa", "reject")
    s, _ = st_of("synth")
    assert s["fit_qa"]["status"] == "stale" and "accept_inputs" not in s["fit_qa"]["actions"]
    gates.open_gate("synth", "fit_qa", "look")
    gates.confirm("synth", "fit_qa", "ok")
    s, _ = st_of("synth")
    assert s["fit_qa"]["status"] == "done"


# ------------------------------------------------------------------------------------------- 9: locks

def test_cross_process_locks(root):
    assert locks.acquire(locks.take_lock("synth"), run="r-test")
    assert not locks.acquire(locks.take_lock("synth"))  # not re-entrant, even for this process
    p = planmod.make_plan("synth", {"target": "ingest"})
    assert Orchestrator("synth", p, dry_run=True).run() == 2  # a second orchestrator of the take refuses
    locks.release(locks.take_lock("synth"))
    # a stale lock (dead holder) is taken over
    touch(locks.take_lock("synth"), text=json.dumps({"pid": 999999, "create_time": 1.0}))
    assert locks.holder(locks.take_lock("synth")) is None and locks.acquire(locks.take_lock("synth"))
    locks.release(locks.take_lock("synth"))
    # local slots: 2 at a time, a heavy stage needs both, one GPU stage
    a, _ = locks.acquire_local({"take": "x", "stage": "a"})
    b, _ = locks.acquire_local({"take": "y", "stage": "b"})
    c, why = locks.acquire_local({"take": "z", "stage": "c"})
    assert a and b and c is None and "2 local stages" in why
    locks.release_all(b)
    h, why = locks.acquire_local({"take": "z", "stage": "heavy"}, heavy=True)
    assert h is None and "heavy" in why
    g1, _ = locks.acquire_local({"take": "z", "stage": "g1"}, gpu=True)
    locks.release_all(a)
    g2, why = locks.acquire_local({"take": "q", "stage": "g2"}, gpu=True)
    assert g1 and g2 is None and "GPU" in why
    locks.release_all(g1)
    # the pod-session lock held by another run queues a session start
    assert locks.acquire(locks.pod_lock(), take="other", session="S2_fit")
    o = Orchestrator("synth", dict(p, steps=[]), dry_run=False)
    allow, why = o.limits_allow("dancers_sam")
    assert allow is False and "one pod session" in why
    locks.release(locks.pod_lock())


# ------------------------------------------------------------------------------------------- 13: plan replay

def test_plan_files_cannot_be_replayed_with_an_approval(root):
    p = planmod.make_plan("synth", {"target": "ingest"})
    planmod.save(p)
    f = planmod.plan_path(p["plan_hash"])
    got = load_plan_for_run("synth", str(f), "r-20261007-120000-aaaa")
    assert got["plan_hash"] == p["plan_hash"] and "approved" not in got
    appr = dict(p, approved={"pods": 1, "max_cost_usd": 1.0, "at": time.time(), "run_id": "r-20261007-120000-bbbb"})
    f.write_text(json.dumps(appr))
    assert "approved" not in load_plan_for_run("synth", str(f), "r-20261007-120000-cccc")  # another run: stripped
    assert load_plan_for_run("synth", str(f), "r-20261007-120000-bbbb")["approved"]["pods"] == 1  # the bound run
    assert json.loads(f.read_text())["approved"]["consumed_by"] == "r-20261007-120000-bbbb"
    assert "approved" not in load_plan_for_run("synth", str(f), "r-20261007-120000-bbbb")  # one shot
    old = dict(p, created=time.time() - planmod.VALID_S - 5)
    f.write_text(json.dumps(old))
    with pytest.raises(planmod.PlanError):
        load_plan_for_run("synth", str(f), None)
    touch(root / "work" / "synth" / "frames" / "01" / "frames.json")  # inputs changed since the plan was made
    f.write_text(json.dumps(p))
    with pytest.raises(planmod.PlanError):
        load_plan_for_run("synth", str(f), None)


# ------------------------------------------------------------------------------------------- 14: status cost

def test_status_cost_does_not_grow_with_finished_runs(root, monkeypatch):
    for i in range(400):
        fake_run_dir("stage", "other" if i % 2 else "synth", ["ingest"], rc=0)
    status.snapshot("synth", with_activity=False)  # warms the caches
    calls = []
    real = runs._state
    monkeypatch.setattr(runs, "_state", lambda d, info: calls.append(d) or real(d, info))
    t0 = time.time()
    for _ in range(3):
        status.snapshot("synth", with_activity=False)
    assert calls == [] and time.time() - t0 < 3.0
    moved = runs.archive(max_age_days=0.0, keep=10, now=time.time() + 60)
    assert moved > 300 and len(list(paths.runs_dir().iterdir())) < 100
    assert runs.list_runs(take="synth", kind="stage")  # the newest run of each take/stage stays


# ------------------------------------------------------------------------------------------- 11: triggers

def test_trigger_layers_use_the_viewer_ids(root):
    rules, _ = triggers.load_rules()
    viewer = set(rules["viewer"]["layers"])
    cs = Path(__file__).resolve().parents[2] / "head-movement" / "Assets" / "Tour" / "DanceLayers.cs"
    if cs.exists():
        import re
        line = next(x for x in cs.read_text(encoding="utf-8").splitlines() if "bool Handles(string layer)" in x)
        assert viewer == set(re.findall(r'"([a-z_]+)"', line))
    for k in rules["kinds"].values():
        assert triggers.viewer_layers(k["layers"], rules), k
    w = root / "work" / "synth"
    touch(w / "counterbalance" / "counterbalance.json", text=json.dumps(
        {"intervals": [{"t0": 1, "t1": 2, "kind": "pivot", "confidence": 0.4, "pivot": [0, 0, 0]}]}))
    doc = triggers.collect("synth")
    (t,) = doc["triggers"]
    assert set(t["viewer_layers"]) <= viewer and "counterbalance" in t["viewer_layers"] and t["style"] == "dashed"
    assert ["-m", "dancecap.pipeline", "triggers", "{take}", "--export-capture", "{capture}"] in \
        registry.OVERLAY["demo_export"]["commands"]


# ------------------------------------------------------------------------------------------- API

@pytest.fixture()
def env(root, monkeypatch):
    from dancecap.pipeline.monitor import server
    started = []

    def fake_start(kind, argv, **kw):
        started.append({"kind": kind, "argv": argv, **kw})
        return {"id": kw.get("run_id") or f"r-20261007-120000-{len(started):04x}", "kind": kind, "state": "starting",
                "take": kw.get("take"), "stages": kw.get("stages")}
    monkeypatch.setattr(server.runs, "start", fake_start)
    poller = FakePoller()
    m = server.Monitor(port=8790, poller=poller)
    app = server.create_app(m)
    c = app.test_client()
    c.get("/", base_url=BASE)
    tok = {"X-Monitor-Token": m.token}

    def send(method, url, body=None, headers=None, data=None):
        h = dict(tok, **(headers or {}))
        if data is not None:
            return getattr(c, method)(url, base_url=BASE, data=data, headers=dict(h, **{"Content-Type": "application/json"}))
        return getattr(c, method)(url, base_url=BASE, json=body, headers=h)
    yield {"root": root, "m": m, "c": c, "send": send, "started": started, "poller": poller, "server": server}
    m.helper.shutdown(wait=True)


def test_api_runs_after_a_gate_and_the_take_lock(env):
    c, send = env["c"], env["send"]
    fake_run_dir("pipeline", "synth", ["ingest"], rc=3)  # ended at a human gate
    p = send("post", "/api/takes/synth/plan", {"target": "ingest"}).get_json()
    r = send("post", "/api/takes/synth/runs", {"plan_hash": p["plan_hash"], "dry_run": True})
    assert r.status_code == 202, r.get_json()
    assert env["started"][-1]["run_id"] == r.get_json()["run"]["id"]  # the run id the approval is bound to
    assert locks.acquire(locks.take_lock("synth"), run=None)  # a CLI orchestrator of this take
    p = send("post", "/api/takes/synth/plan", {"target": "ingest"}).get_json()
    r = send("post", "/api/takes/synth/runs", {"plan_hash": p["plan_hash"], "dry_run": True})
    assert r.status_code == 409 and "CLI" in r.get_json()["error"]["message"]
    locks.release(locks.take_lock("synth"))


def test_api_review_gate_endpoint(env):
    send = env["send"]
    assert send("post", "/api/takes/synth/gates/fit_qa", {}).status_code == 409  # no open gate
    gates.open_gate("synth", "geoveto", "look")
    assert send("post", "/api/takes/synth/gates/geoveto", {"decision": "maybe"}).status_code == 422
    r = send("post", "/api/takes/synth/gates/geoveto", {"decision": "apply"})
    assert r.status_code == 200 and env["started"][-1]["argv"] == ["-m", "dancecap.dancers_geoveto", "synth", "--apply"]
    assert not gates.is_open("synth", "geoveto")
    assert send("post", "/api/takes/synth/gates/labels_review", {}).status_code == 200


def test_api_window_validation(env):
    c, send, root = env["c"], env["send"], env["root"]
    touch(root / "work" / "synth" / "probe" / "probe.json", text=json.dumps(
        {"cameras": {c_: {"duration_s": 60.0} for c_ in ("01", "02", "03")}}))
    etag = c.get("/api/takes/synth", base_url=BASE).get_json()["toml_etag"]
    h = {"If-Match": etag}
    assert send("patch", "/api/takes/synth", data='{"window": [1, Infinity]}', headers=h).status_code == 422
    assert send("patch", "/api/takes/synth", {"window": [False, True]}, headers=h).status_code == 422
    r = send("patch", "/api/takes/synth", {"window": [10, 70]}, headers=h)  # the videos end at 62 s (offset 2)
    assert r.status_code == 422 and "beyond the videos" in r.get_json()["error"]["details"]["problems"][0]["message"]
    assert send("patch", "/api/takes/synth", {"window": [10, 50]}, headers=h).status_code == 200
    # a TOML that already holds an infinite window reads as "no window" (no 500 anywhere)
    tp = root / "takes" / "synth.toml"
    tp.write_text(tp.read_text(encoding="utf-8").replace("window = [10.0, 50.0]", "window = [1.0, inf]"),
                  encoding="utf-8")
    env["m"].invalidate()
    for url in ("/api/takes/synth", "/api/takes/synth/pipeline"):
        assert c.get(url, base_url=BASE).status_code == 200, url
    assert send("post", "/api/takes/synth/plan", {"target": "ingest"}).status_code == 200


def test_api_orphan_cleanup_only_deletes_orphans(env, monkeypatch):
    send, poller = env["send"], env["poller"]
    poller.pods = [{"id": "orphan0001", "name": "dancecap-a", "managed": True, "flags": ["orphan"], "session": None},
                   {"id": "active0002", "name": "dancecap-b", "managed": True, "flags": [],
                    "session": {"sid": "20261007-123801-batch", "take": "synth"}}]
    from dancecap import remote
    deleted = []
    monkeypatch.setattr(remote, "_delete", lambda pid, reason: deleted.append(pid))
    assert env["c"].get("/api/runpod/orphans", base_url=BASE).get_json()["orphans"][0]["id"] == "orphan0001"
    assert send("post", "/api/runpod/cleanup", {"confirm": "nope"}).status_code == 403
    r = send("post", "/api/runpod/cleanup", {"confirm": "DELETE", "pods": ["active0002"]})
    assert r.status_code == 409 and deleted == []
    r = send("post", "/api/runpod/cleanup", {"confirm": "DELETE"})
    assert r.status_code == 200 and r.get_json()["deleted"] == ["orphan0001"] and deleted == ["orphan0001"]


def test_api_bad_inputs_are_4xx_and_small_fixes(env):
    c, send, root = env["c"], env["send"], env["root"]
    g = lambda u: c.get(u, base_url=BASE)  # noqa: E731
    assert g("/api/runs?limit=abc").status_code == 400
    assert send("post", "/api/takes/synth/plan", {"target": "ingest", "options": "x"}).status_code == 422
    assert send("post", "/api/takes/synth/plan", {"target": "ingest", "options": {"max_cost_usd": "abc"}}).status_code == 422
    assert send("post", "/api/takes/synth/plan", {"target": "ingest", "include_optional": "room"}).status_code == 422
    etag = g("/api/takes/synth").get_json()["toml_etag"]
    assert send("patch", "/api/takes/synth", {"dance": "x"}, headers={"If-Match": etag}).status_code == 422
    assert send("patch", "/api/takes/synth", {"dancers": {"lead": "x"}}, headers={"If-Match": etag}).status_code == 422
    assert send("post", "/api/takes/synth/camera-types/accept", {"min_confidence": "abc"},
                headers={"If-Match": etag}).status_code == 422
    assert send("post", "/api/takes", {"take": "x", "videos": "abc"}).status_code == 400
    assert send("put", "/api/settings", {"poll_ms": True}).status_code == 422
    assert send("post", "/api/takes/synth/plan", data="{" + '"a": "' + "x" * (5 << 20) + '"}').status_code == 413
    # consent: one scope at a time never resets the others
    r = send("patch", "/api/takes/synth", {"consent": {"lead": {"cloud_processing": True}}}, headers={"If-Match": etag})
    etag = r.get_json()["toml_etag"]
    r = send("patch", "/api/takes/synth", {"consent": {"lead": {"publication": False}}}, headers={"If-Match": etag})
    assert r.status_code == 200
    d = g("/api/takes/synth").get_json()["consent"]["lead"]
    assert d["cloud_processing"] is True and d["publication"] is False
    # Unity load refuses while another workflow holds work/unity.lock
    (paths.work() / "unity.lock").mkdir()
    assert send("post", "/api/unity/load", {"capture": "05_X"}).status_code == 423
    # /index.html is the page with its token, like /
    assert f'content="{env["m"].token}"' in g("/index.html").get_data(as_text=True)


def test_soundtrack_must_be_an_audio_file(root, tmp_path):
    from _pipeline_fixtures import fake_probe
    v = tmp_path / "snd" / "01-cam.mp4"
    v.parent.mkdir(parents=True)
    v.write_bytes(b"x")
    _, probs, _ = intake.validate(_req([v], take="snd_take", soundtrack={"path": str(v)}), probe_fn=fake_probe())
    assert any(p["field"] == "soundtrack" and "audio file" in p["message"] for p in probs)
