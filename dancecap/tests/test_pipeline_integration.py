"""Integration fixes of the Pipeline Monitor (stage 'integrate', 2026-10-07), on a synthetic package root:

- one pod per job: the S1_front pod (front_pod has no per-stage selection) is never planned while one of its own
  stages still waits for a human gate (labels_review): no half-failing pod, no second upload later;
- what-if plans (options.assume_reviews): dry-run only, human steps become `pause`, the review gates do not block
  their descendants, so a new capture's whole order / time / pod cost can be previewed; a real run is refused;
- the orchestrator waits for SOFT deps that run in the same run (demo_export after hair_groom / triggers);
- the plan carries the camera types it applies (per camera and per camera step);
- `python -m dancecap.pipeline export --dry-run` and its own monitor route (writes nothing; an older server answers
  404 instead of exporting);
- probe A is not measured twice: the wizard's staged suggestion (file signature) is reused by the probe stage.
No pods, no Runpod calls, no real take touched."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pipeline_fixtures import FakePoller, make_root, make_take, touch  # noqa: E402
from dancecap.pipeline import camtypes, intake, orchestrator, paths, plan as planmod  # noqa: E402

CONSENT = ('\n[consent]\nlead = { capture = true, fit_3d = true, ml_analysis = true, cloud_processing = true }\n'
           'follow = { capture = true, fit_3d = true, ml_analysis = true, cloud_processing = true }\n')


@pytest.fixture()
def root(tmp_path):
    r = make_root(tmp_path)
    make_take(r, extra=CONSENT)
    yield r
    paths.set_root(None)


def steps_of(p):
    return {s["stage"]: s for s in p["steps"]}


# ------------------------------------------------------------------------------------------- one pod per job

def confirm_types(take="synth", cams=("01", "02", "03")):
    for c in cams:
        _, etag = camtypes.load(take, [c])
        camtypes.set_type(take, c, "handheld", False, etag)


def test_s1_pod_waits_for_the_labels_review_of_its_own_job(root):
    confirm_types()
    p = planmod.make_plan("synth", {"scope": "all"})
    st = steps_of(p)
    assert st["labels_review"]["action"] == "blocked"
    assert st["dancers_sam"]["action"] == "blocked"
    # cam_loc alone would be runnable, but the S1_front job would also attempt SAM before the review: held back
    assert st["cam_loc"]["action"] == "blocked" and "S1_front pod job" in st["cam_loc"]["reason"]
    assert not [s for s in p["sessions"] if s["group"] == "S1_front"]


def test_s1_is_one_pod_with_all_its_stages_once_the_review_is_assumed(root):
    p0 = planmod.make_plan("synth", {"scope": "all", "options": {"assume_reviews": True}})
    s1 = [s for s in p0["sessions"] if s["group"] == "S1_front"]
    assert len(s1) == 1 and {"dancers_sam", "cam_loc", "ma2d", "cam_world", "sync"} <= set(s1[0]["stages"])


# ------------------------------------------------------------------------------------------- what-if

def test_what_if_plan_covers_the_whole_pipeline_and_is_dry_run_only(root):
    p = planmod.make_plan("synth", {"scope": "all", "options": {"assume_reviews": True, "capture": "07_Synth"}})
    st = steps_of(p)
    assert p["what_if"] is True
    assert st["labels_review"]["action"] == "pause" and st["narrate"]["action"] == "pause"
    assert st["dancers_sam"]["action"] == "run" and st["mamma_fit"]["action"] == "run"
    assert st["texture"]["action"] == "run" and st["demo_export"]["action"] == "run"  # beyond the review gates
    assert {"geoveto", "fit_qa"} <= set(p["review_gates"]) and "labels_review" in p["halts_at"]
    groups = [s["group"] for s in p["sessions"]]
    assert "S1_front" in groups and "S2_fit" in groups
    assert p["warnings"][0].startswith("what-if plan (dry-run only)")
    planmod.save(p)
    with pytest.raises(planmod.PlanError):
        planmod.revalidate("synth", p["plan_hash"], {"pods": len(p["sessions"]), "max_cost_usd": 999}, dry_run=False)
    ok = planmod.revalidate("synth", p["plan_hash"], None, dry_run=True)
    assert ok["dry_run"] is True and "approved" not in ok
    # the normal plan of the same take still halts at the first human gate
    n = planmod.make_plan("synth", {"scope": "all"})
    assert not n["what_if"] and steps_of(n)["demo_export"]["action"] == "blocked"


def test_what_if_dry_run_pauses_and_waits_for_soft_deps(root, capsys):
    p = planmod.make_plan("synth", {"scope": "all", "options": {"assume_reviews": True, "capture": "07_Synth"}})
    o = orchestrator.Orchestrator("synth", p, dry_run=True, poll_s=0.01)
    assert o.run() == 0
    evs = [json.loads(x) for x in (paths.pipeline_dir("synth") / "events.jsonl").read_text().splitlines()]
    order = [e["stage"] for e in evs if e["kind"] in ("start", "would_pause")]
    assert {"labels_review", "narrate"} <= {e["stage"] for e in evs if e["kind"] == "would_pause"}
    # soft deps that run in the same run come first: the export carries the groom and the triggers
    for soft in ("hair_groom", "triggers", "counterbalance"):
        assert order.index(soft) < order.index("demo_export"), soft
    assert order.index("moves_proposals") < order.index("narrate")  # the narration shows the proposals
    fin = [e for e in evs if e["kind"] == "finish"][-1]
    assert "what-if: a real run pauses for you at" in fin["msg"]
    # a what-if plan is never run for real from the CLI either
    planmod.save(p)
    rel = Path("work") / "pipeline" / "plans" / planmod.plan_path(p["plan_hash"]).name
    assert orchestrator.main(["synth", "--plan", rel.as_posix()]) == 2


def test_soft_deps_never_deadlock(root):
    """A soft dep that is also a descendant is ignored (no wait cycle)."""
    p = planmod.make_plan("synth", {"scope": "all", "options": {"assume_reviews": True, "capture": "07_Synth"}})
    o = orchestrator.Orchestrator("synth", p, dry_run=True, poll_s=0.01)
    for sid in o.todo:
        from dancecap.pipeline import registry
        desc = set(registry.descendants(list(o.vby.values()), [sid]))
        assert not set(o.soft_wait(sid)) & desc


# ------------------------------------------------------------------------------------------- camera types in the plan

def test_plan_lists_the_camera_types_it_applies(root):
    _, etag = camtypes.load("synth", ["01"])
    camtypes.set_type("synth", "01", "tripod", False, etag)
    p = planmod.make_plan("synth", {"scope": "all", "options": {"assume_reviews": True}})
    by = {c["cam"]: c for c in p["camera_types"]}
    assert by["01"]["class"] == "tripod" and by["01"]["state"] == "confirmed" and by["01"]["effects"]
    assert by["02"]["class"] == "handheld_zoom" and by["02"]["state"] != "confirmed"  # unanswered: general model
    loc = steps_of(p)["cam_loc"]
    assert loc["camera_types"]["tripod"] == ["01"] and "02" in loc["camera_types"]["handheld_zoom"]
    assert "camera_types" not in steps_of(p)["ingest"]


# ------------------------------------------------------------------------------------------- export dry run

def test_export_dry_run_writes_nothing(root, capsys):
    from dancecap.pipeline import export
    w = root / "work" / "synth"
    before = sorted(str(x) for x in root.rglob("*"))
    assert export.main(["synth", "--capture", "07_Synth", "--dry-run"]) == 1  # no motion yet
    out = capsys.readouterr().out
    assert "DRY RUN" in out and "MISSING motion (smplx/)" in out and "would create" in out
    for f in ("smplx/lead.npz", "smplx/follow.npz", "cameras/world_from_sfm.json", "audio/audio_overlap.wav",
              "audio/audio_overlap.json"):
        touch(w / f)
    before = sorted(str(x) for x in root.rglob("*"))
    assert export.main(["synth", "--capture", "07_Synth", "--dry-run"]) == 0
    assert sorted(str(x) for x in root.rglob("*")) == before  # nothing written
    assert "DRY RUN OK" in capsys.readouterr().out


def test_export_dry_run_route(root, monkeypatch):
    from dancecap.pipeline.monitor import server
    started = []

    def fake_start(kind, argv, **kw):
        started.append({"kind": kind, "argv": argv, **kw})
        return {"id": "r-20261007-120000-0001", "kind": kind, "state": "starting"}

    monkeypatch.setattr(server.runs, "start", fake_start)
    m = server.Monitor(port=8790, poller=FakePoller())
    app = server.create_app(m)
    c = app.test_client()
    base = "http://127.0.0.1:8790"
    c.get("/", base_url=base)
    h = {"X-Monitor-Token": m.token}
    r = c.post("/api/takes/synth/export-dry-run", base_url=base, json={"capture": "07_Synth"}, headers=h)
    assert r.status_code == 202 and r.get_json()["dry_run"] is True
    assert started[-1]["kind"] == "dry_run" and started[-1]["argv"][-1] == "--dry-run"
    assert started[-1]["argv"][:4] == ["-m", "dancecap.pipeline", "export", "synth"]
    # the real export route ignores a dry_run flag in the body: it never decides a dry run by itself
    r = c.post("/api/takes/synth/export", base_url=base, json={"capture": "07_Synth", "dry_run": True}, headers=h)
    assert r.status_code in (202, 423) and (r.status_code == 423 or "--dry-run" not in started[-1]["argv"])
    m.helper.shutdown(wait=True)


# ------------------------------------------------------------------------------------------- probe A reuse

def test_staged_probe_a_suggestion_is_reused_by_the_probe_stage(root, tmp_path, monkeypatch):
    from _pipeline_fixtures import fake_probe
    from dancecap import camera_types as ct
    v = tmp_path / "src" / "pair" / "01-front.mp4"
    v.parent.mkdir(parents=True)
    v.write_bytes(b"\0" * 2048)
    v2 = v.parent / "02-side.mp4"
    v2.write_bytes(b"\0" * 4096)
    sig = ct._file_sig(v)
    staged = {"mount": "tripod", "zoom": False, "confidence": 0.8, "by": "probe_a", "version": ct.PROBE_A_VERSION,
              "evidence": {"parallax_px": 0.0}, "file": sig}
    req = {"take": "pair_take", "videos": [{"cam": "01", "path": str(v), "suggestion": staged},
                                           {"cam": "02", "path": str(v2)}],
           "dancers": {"lead": {"name": "A", "stature_m": "180 cm"}, "follow": {"name": "B", "stature_m": "165 cm"}},
           "dance": {"level": "novice", "category": "lesson"}, "consent": {}}
    intake.create(req, probe_fn=fake_probe())
    doc = json.loads((paths.take_work("pair_take") / "probe" / "camera_types_suggested.json").read_text())
    assert doc["cameras"]["01"]["file"] == sig and "02" not in doc["cameras"]

    def boom(*a, **k):
        raise AssertionError("probe A decoded the video again")
    monkeypatch.setattr(ct, "suggest_from_video", boom)
    s = camtypes.suggest_for_video(v, {"duration_s": 60})
    assert s["mount"] == "tripod" and s["reused_from"] == "pair_take"
    # another file (no staged result): measured, and the result carries its signature for the next reuse
    monkeypatch.setattr(ct, "suggest_from_video", lambda video, probe=None: {"mount": "handheld", "zoom": True})
    s2 = camtypes.suggest_for_video(v2, {"duration_s": 60})
    assert s2["file"] == ct._file_sig(v2)


# ------------------------------------------------------------------------------------------- failing pod jobs

def test_failed_pod_tasks_are_flagged(root):
    from dancecap.pipeline import runpod_view
    log = "\r\n".join([
        "[pod-runner 17:51:04] front_full/front ...",
        "[front] 00:51:52 start b.ingest:02",
        "[front] 00:51:52 b.ingest:02: failed rc=1 in 3s (rss peak 0.0 GB)",
        "[front] b.loc:01:0: skipped (dependency failed: b.detect:01)",
        "[front] d.sam:01: skipped (dependency failed: d.detect:01)",
        "[front] 00:52:33 start a.merge",
        "[front] 00:52:40 a.merge: ok in 7s"])
    tf = runpod_view.task_failures(log)
    assert tf["failed"] == 1 and tf["skipped"] == 2 and tf["tasks"] == ["b.ingest:02"]
    out = runpod_view._with_failures({"warnings": ["older"]}, log)
    assert out["warnings"][0].startswith("1 pod task(s) failed and 2 were skipped") and out["warnings"][1] == "older"
    healthy = runpod_view._with_failures({"warnings": []}, "[front] 00:52:40 a.merge: ok in 7s")
    assert healthy["warnings"] == [] and healthy["task_failures"]["failed"] == 0


def test_a_job_failed_with_a_reason_fails_its_session(root):
    """remote_jobs writes 'failed: <reason>' as the job status: the session is failed (was shown 'ok'), and an ended
    session shows no stale 'Now' line."""
    from dancecap.pipeline import runpod_view
    d = paths.sessions_dir() / "20261007-172928-front-full"
    d.mkdir(parents=True)
    (d / "state.json").write_text(json.dumps({"sid": d.name, "pod": {"id": "abc123def456gh", "cost_per_hr": 1.39},
                                              "jobs": {"front_full": {"status": "failed: status rc=1",
                                                                      "spec": {"take": "synth"}}}}))
    (d / "runner.log").write_text("[front] 00:52:32 start a.merge\n[pod-runner 17:52:57] front_full: failed: status "
                                  "rc=1 (113s)\n")
    import os
    import time as _t
    os.utime(d / "state.json", (_t.time() - 60, _t.time() - 60))
    os.utime(d / "runner.log", (_t.time() - 60, _t.time() - 60))
    (d / "result.json").write_text("{}")
    s = runpod_view.session_summary(d.name, ledger=[])
    assert s["state"] == "failed"
    sn = runpod_view.session_snapshot(d.name)
    assert sn["progress"] is None and sn["current"] is None
