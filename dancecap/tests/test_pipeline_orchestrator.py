"""dancecap.pipeline plan / orchestrator / runs / runwrap: plans with gates, DAG-order runs with limits, resumable
state, dry runs, stop. Stage commands are swapped for a tiny fake module; no pod is ever created (sessions stay
blocked without an approval and only dry-run otherwise)."""
import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pipeline_fixtures import make_root, make_take, touch, wait_for  # noqa: E402
from dancecap.pipeline import paths, plan as planmod, registry, runs  # noqa: E402
from dancecap.pipeline.orchestrator import Orchestrator  # noqa: E402

FAKE = '''
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
    print(f"[fake] {take} end rc={rc}", flush=True)
finally:
    if opt.get("marker"):
        Path(opt["marker"]).write_text("finally ran")
sys.exit(rc)
'''


@pytest.fixture()
def root(tmp_path, monkeypatch):
    r = make_root(tmp_path)
    mods = tmp_path / "fakemods"
    mods.mkdir()
    (mods / "fake_stage.py").write_text(FAKE, encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(mods))
    monkeypatch.setenv("DANCECAP_STOP_GRACE_S", "10")
    make_take(r)
    yield r
    paths.set_root(None)


def fake_cmd(monkeypatch, stage, outs, **opt):
    argv = ["-m", "fake_stage", "{take}", *outs] + [f"--{k}={v}" for k, v in opt.items()]
    monkeypatch.setitem(registry.OVERLAY[stage], "commands", [argv])


def test_plan_actions_hash_and_gates(root, monkeypatch):
    p = planmod.make_plan("synth", {"target": "ingest"})
    acts = {s["stage"]: s["action"] for s in p["steps"]}
    assert acts == {"intake": "skip_done", "audio_sync": "skip_done", "window": "skip_done", "ingest": "run"}
    assert p["sessions"] == [] and p["budget"]["gate"] == "ok" and p["plan_hash"].startswith("sha1:")
    assert planmod.make_plan("synth", {"target": "ingest"})["plan_hash"] == p["plan_hash"]  # deterministic
    full = planmod.make_plan("synth", {"scope": "all"})
    fa = {s["stage"]: s for s in full["steps"]}
    assert fa["camera_types"]["action"] == "blocked" and fa["cam_map"]["action"] == "blocked"
    assert fa["scale_probe"]["action"] == "blocked" and "not built" in fa["scale_probe"]["reason"]
    assert "camera_types" in full["halts_at"]
    planmod.save(p)
    touch(root / "work" / "synth" / "frames" / "01" / "frames.json")  # inputs changed -> the plan is stale
    with pytest.raises(planmod.StalePlan):
        planmod.revalidate("synth", p["plan_hash"], None, dry_run=False)


def _ready_for_front(root):
    w = root / "work" / "synth"
    t = time.time() - 100
    for f in ("frames/01/frames.json", "keyframes/01/keyframes.json", "people/01.npz", "people/report.json",
              "dancers/labels.json"):
        touch(w / f, t)
    touch(w / "intake.json", text=json.dumps({"labels_reviewed_at": time.time()}))


def test_pod_sessions_need_approval_consent_and_budget(root, monkeypatch):
    _ready_for_front(root)
    p = planmod.make_plan("synth", {"target": "dancers_sam", "scope": "only"})
    step = p["steps"][0]
    assert step["action"] == "run" and step["session"] == "S1_front"
    # the front pod job cannot be narrowed to dancers_sam: what it really runs is in the plan (ma2d: its inputs
    # will be there) or named in a warning (the camera tasks: their inputs are not ready)
    assert [s["stage"] for s in p["steps"]] == ["dancers_sam", "ma2d"]
    assert "outside the requested scope" in p["steps"][1]["reason"]
    assert any("also attempts" in w and "cam_world" in w for w in p["warnings"])
    (sess,) = p["sessions"]
    assert sess["stages"] == ["dancers_sam", "ma2d"] and p["pods"] == 1
    assert sess["id"] == "S1_front" and sess["max_cost_usd"] > 0 and sess["est_min"]["p90"] >= sess["est_min"]["p50"]
    planmod.save(p)
    with pytest.raises(planmod.ApprovalError) as e:
        planmod.revalidate("synth", p["plan_hash"], None, dry_run=False)
    assert e.value.code == "conflict"
    ok_approve = {"pods": 1, "max_cost_usd": sess["max_cost_usd"]}
    with pytest.raises(planmod.ApprovalError) as e:  # the synthetic take records no cloud consent
        planmod.revalidate("synth", p["plan_hash"], ok_approve, dry_run=False)
    assert e.value.code == "consent_gate"
    dry = planmod.revalidate("synth", p["plan_hash"], None, dry_run=True)  # a dry run needs no approval
    assert dry["dry_run"] is True and "approved" not in dry
    # without the approval the orchestrator never starts a session: blocked, rc 3, no process at all
    rc = Orchestrator("synth", p, run_id=None, poll_s=0.05).run()
    assert rc == 3
    assert not paths.runs_dir().exists() or not list(paths.runs_dir().iterdir())
    st = json.loads((paths.pipeline_dir("synth") / "state.json").read_text())
    assert st["stages"]["dancers_sam"]["state"] == "blocked" and st["status"] == "waiting_human"


def test_budget_gate_refuses_over_cap(root, monkeypatch):
    _ready_for_front(root)
    monkeypatch.setenv("DANCECAP_RUNPOD_CAP", "30")
    touch(root / "work" / "runpod" / "ledger.json", text=json.dumps(
        [{"pod_id": "zzzz9999", "cost_per_hr": 1.0, "created_ts": 0, "deleted_ts": 1, "cost": 29.9}]))
    p = planmod.make_plan("synth", {"target": "dancers_sam", "scope": "only"})
    assert p["budget"]["gate"] == "over_cap"
    cfg_path = root / "takes" / "synth.toml"
    cfg_path.write_text(cfg_path.read_text() + '\n[consent]\nlead = { cloud_processing = true }\n'
                                               'follow = { cloud_processing = true }\n')
    p = planmod.make_plan("synth", {"target": "dancers_sam", "scope": "only"})
    planmod.save(p)
    with pytest.raises(planmod.ApprovalError) as e:
        planmod.revalidate("synth", p["plan_hash"], {"pods": 1, "max_cost_usd": 99}, dry_run=False)
    assert e.value.code == "budget_gate"


def test_dry_run_orders_stages_and_starts_nothing(root, monkeypatch):
    _ready_for_front(root)
    p = planmod.make_plan("synth", {"target": "dancers_sam", "scope": "only"})
    rc = Orchestrator("synth", p, dry_run=True, poll_s=0.02).run()
    assert rc == 0
    evs = [json.loads(x) for x in (paths.pipeline_dir("synth") / "events.jsonl").read_text().splitlines()]
    would = [e for e in evs if e["kind"] == "would_run"]
    assert would and any("remote" in " ".join(e["argv"]) and "--max-cost" in e["argv"] for e in would)
    assert any(e["argv"][2:4] == ["dancecap.front_pod", "write-job"] for e in would)
    assert not paths.runs_dir().exists() or not list(paths.runs_dir().iterdir())


def test_local_run_in_dag_order_with_state_and_resume(root, monkeypatch):
    fake_cmd(monkeypatch, "ingest", ["frames/01/frames.json", "keyframes/01/keyframes.json"], sleep=0.2)
    fake_cmd(monkeypatch, "probe", ["probe/probe.json"])
    p = planmod.make_plan("synth", {"scope": "all"})
    todo = [s["stage"] for s in p["steps"] if s["action"] in ("run", "rerun_stale")]
    assert {"probe", "ingest"} <= set(todo)
    # only the two fake stages run here; everything else waits on the camera-types gate / planned stages
    p["steps"] = [s for s in p["steps"] if s["stage"] in ("intake", "audio_sync", "window", "probe", "ingest")
                  or s["action"] not in ("run", "rerun_stale")]
    o = Orchestrator("synth", p, poll_s=0.1)
    rc = o.run()
    assert o.finished.get("ingest") == "ok" and o.finished.get("probe") == "ok"
    assert rc == 3  # finished what could run; human gates (camera types) halt the rest
    st = json.loads((paths.pipeline_dir("synth") / "state.json").read_text())
    assert st["stages"]["ingest"]["state"] == "done" and st["status"] == "waiting_human"
    rr = runs.list_runs(take="synth")
    assert {tuple(r["stages"]) for r in rr} >= {("ingest",), ("probe",)}
    assert all(r["state"] == "done" and r["rc"] == 0 for r in rr)
    log = runs.log_chunk(runs.run_dir(rr[0]["id"]) / "output.log")
    assert any("[fake]" in line for line in log["lines"])
    # resume: the same request now skips what is done
    p2 = planmod.make_plan("synth", {"target": "ingest"})
    assert {s["stage"]: s["action"] for s in p2["steps"]}["ingest"] == "skip_done"


def test_failed_stage_skips_its_descendants(root, monkeypatch):
    fake_cmd(monkeypatch, "ingest", ["frames/01/frames.json"], rc=1)
    p = planmod.make_plan("synth", {"target": "ingest"})
    p["steps"].append({"stage": "detect", "action": "run", "runs_on": "local_gpu", "session": None, "status": "ready",
                       "est": {"p50_s": 1, "p90_s": 1}, "cost": {"p50_usd": 0, "p90_usd": 0}})
    o = Orchestrator("synth", p, poll_s=0.1)
    assert o.run() == 1
    assert o.finished["ingest"] == "failed" and o.finished["detect"] == "skipped"
    assert runs.list_runs(take="synth")[0]["state"] == "failed"


def test_limits_two_local_heavy_alone_one_gpu_one_session(root):
    p = planmod.make_plan("synth", {"scope": "all"})
    o = Orchestrator("synth", p, dry_run=True)
    o.active = {"counterbalance": {"members": ["counterbalance"]}}
    assert o.limits_allow("timing")[0] is True
    o.active["physics"] = {"members": ["physics"]}
    assert o.limits_allow("timing") == (False, "2 local stages at a time")
    o.active = {"ingest": {"members": ["ingest"]}}  # heavy
    assert o.limits_allow("timing")[0] is False
    o.active = {"timing": {"members": ["timing"]}}
    assert o.limits_allow("cam_map")[0] is False  # heavy waits for the others
    o.active = {"beats": {"members": ["beats"]}}
    assert o.limits_allow("detect")[0] is False  # one local GPU stage
    o.active = {"dancers_sam": {"members": ["dancers_sam"]}}
    assert o.limits_allow("mamma_fit")[0] is False and o.limits_allow("timing")[0] is True
    (paths.work() / "unity.lock").mkdir()
    o.active = {}
    assert o.limits_allow("demo_export")[0] is True  # a dry run only notes the lock
    real = Orchestrator("synth", p, dry_run=False)
    assert real.limits_allow("demo_export")[0] is None  # nothing else running: blocked, never spins on the lock
    real.active = {"timing": {"members": ["timing"]}}
    assert real.limits_allow("demo_export")[0] is False  # queued while other work runs


def test_stop_now_kills_the_run_and_finally_runs(root, tmp_path):
    marker = tmp_path / "marker.txt"
    r = runs.start("stage", ["-m", "fake_stage", "synth", "x.json", "--sleep=60", f"--marker={marker}"], take="synth",
                   stages=["ingest"])
    assert wait_for(lambda: runs.describe(r["id"])["state"] == "running", 30)
    assert runs.describe(r["id"])["stop_modes"] == ["now"]
    with pytest.raises(PermissionError):
        runs.request_stop(r["id"], "after_step")
    runs.request_stop(r["id"], "now")
    final = wait_for(lambda: (runs.describe(r["id"]) or {}).get("state") in ("stopped", "failed", "done")
                     and runs.describe(r["id"]), 30)
    assert final and final["state"] == "stopped" and final["rc"] == 130
    assert marker.read_text() == "finally ran"
    assert not (paths.take_work("synth") / "x.json").exists()


def test_runs_refuse_non_module_argv(root):
    with pytest.raises(ValueError):
        runs.start("stage", ["python", "-c", "print(1)"])
    with pytest.raises(ValueError):
        runs.run_dir("../../etc")


def test_session_waves_never_deadlock(root):
    """texture (S2) needs fit_qa (local), which needs mamma_fit (S2): two S2 sessions, priced and run in order."""
    def step(sid, sess):
        return {"stage": sid, "action": "run", "runs_on": "pod" if sess else "local", "session": sess,
                "status": "ready", "est": {"p50_s": 600, "p90_s": 900}, "cost": {"p50_usd": 0.2, "p90_usd": 0.3}}
    steps = [step("mamma_fit", "S2_fit"), step("fit_qa", None), step("texture", "S2_fit"), step("face_pod", "S2_fit")]
    vst = registry.variant_stages("multi")
    waves = planmod._session_waves(steps, vst, {s["id"]: s for s in vst}, {s["stage"] for s in steps})
    assert [(w, [s["stage"] for s in m]) for w, m in waves] == [("S2_fit", ["mamma_fit"]),
                                                                  ("S2_fit#2", ["texture", "face_pod"])]
    p = planmod.make_plan("synth", {"target": "ingest"})
    p["steps"] = steps
    p["sessions"] = [{"id": w, "stages": [s["stage"] for s in m], "max_cost_usd": 2.0, "max_minutes": 60}
                     for w, m in waves]
    o = Orchestrator("synth", p, dry_run=True, poll_s=0.02)
    assert o.run() == 0
    evs = [json.loads(x) for x in (paths.pipeline_dir("synth") / "events.jsonl").read_text().splitlines()]
    starts = [e for e in evs if e["kind"] == "start"]
    assert [e["stage"] for e in starts][:2] == ["mamma_fit", "fit_qa"] and len(starts) == 3
    assert sorted(starts[2]["members"]) == ["face_pod", "texture"]  # one pod for the second wave
