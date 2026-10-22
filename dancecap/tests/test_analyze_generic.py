"""2026-10-22 generic round: `analyze <video>` runs every mono step with no agent (pipeline/mono_route.py), and the round-2
review's dashboard leftovers (a run ETA that only pushed out, snapshots drawn at render time, "waiting" while a step runs,
pod-cpu steps the runner cannot place). Synthetic data only: temp package roots, fake clocks, fake sessions. No pod, no Runpod
call, no snapshot process.

    ..\\.venv-nopo4d\\Scripts\\python.exe -m pytest tests/test_analyze_generic.py -q        (from dancecap/ or the workspace)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _pipeline_fixtures import make_root  # noqa: E402
from dancecap.pipeline import activity, analyze as A, mono_route as M, paths, snapshots  # noqa: E402

TRIAL = {"n": 1, "duration_s": 19.97, "size_mb": 11.2, "fps": 30.0, "frames": 599, "probed": True}


@pytest.fixture()
def root(tmp_path):
    r = make_root(tmp_path)
    yield r
    paths.set_root(None)


def _doc(run: str, steps: list[dict], **kw) -> dict:
    for s in steps:
        s.setdefault("status", "pending")
    d = {"schema": A.SCHEMA, "run": run, "take": "synth", "created": time.time(), "updated": time.time(), "agent_hint": "",
         "status": "planned", "input": {"path": "x", "route": "mono", "kind": "single", "videos": []}, "facts": dict(TRIAL),
         "steps": steps, "plan": A.totals(steps), "cost_so_far_usd": 0.0}
    d.update(kw)
    A.run_dir(run).mkdir(parents=True, exist_ok=True)
    A.save(d)
    return d


# ------------------------------------------------------------------------------------------------ ETA: honest both ways

def _analyses_case(now, basis, whole=True, live_lanes=1):
    """The review's 20:59:02 case: analyses (plan 8.3 / 19.1 min) has run 7.4 min, its one session says 91.5 s (p90 110 s)
    from measured phases; export (2 / 4 min) follows."""
    an = A._step("analyses", "analyses", "pod-gpu", [], 8.3, 19.1)
    an.update(status="running", started=now - 444, eta_s=91.5, eta_s_p90=110.0, eta_basis=basis, live_lanes=live_lanes,
              whole_step=whole)
    ex = A._step("export", "export", "local", ["analyses"], 2, 4, price="local")
    return [an, ex]


def test_a_measured_whole_step_eta_pulls_the_run_eta_in():
    now = 1_800_000_000.0
    s50 = A.schedule(_analyses_case(now, "measured"), now, "p50")
    s90 = A.schedule(_analyses_case(now, "measured"), now, "p90")
    assert s50["left"]["analyses"] == pytest.approx(91.5) and s90["left"]["analyses"] == pytest.approx(110.0)
    assert s50["end"] - now == pytest.approx(91.5 + 120) and s90["end"] - now == pytest.approx(110 + 240)   # 212 / 350 s
    # not measured, or a session covering part of the step, or fewer live sessions than lanes: push out only (the overrun
    # floor of residual_min stays: 277 s p50 here, as the round-2 rule computed)
    for steps in (_analyses_case(now, "session"), _analyses_case(now, "measured", whole=False),
                  _analyses_case(now, "measured", live_lanes=0)):
        s = A.schedule(steps, now, "p50")
        l50, _ = A.residual_min(8.3, 19.1, 7.4)
        assert s["left"]["analyses"] == pytest.approx(max(l50 * 60, 91.5))
    # a session ETA made from a job's timeout is never used (it said 21 min for a 3-min job)
    st = _analyses_case(now, "timeout")
    st[0]["eta_s"] = 9999.0
    assert A.schedule(st, now, "p50")["left"]["analyses"] < 9999.0
    # a measured ETA that is LONGER than the plan pushes out too (in or out)
    st = _analyses_case(now, "measured")
    st[0]["eta_s"] = st[0]["eta_s_p90"] = 3000.0
    assert A.schedule(st, now, "p50")["left"]["analyses"] == pytest.approx(3000.0)


def test_session_eta_basis_rules():
    good = {"eta": {"p50": 90, "basis": "progress_eta L2"}, "phase_detail": "history n=4", "warnings": []}
    assert activity.session_eta_basis([good]) == "measured"
    assert activity.session_eta_basis([good, {**good, "eta": {"p50": 1, "basis": "session estimate - elapsed"}}]) == "session"
    assert activity.session_eta_basis([{**good, "phase_detail": "(no history: generic prior)"}]) == "session"
    assert activity.session_eta_basis([{**good, "phase_detail": "0.3 x timeout"}]) == "timeout"
    assert activity.session_eta_basis([{**good, "warnings": [{"msg": "no progress for 6 min"}]}]) == "session"


def test_step_row_and_run_kpi_show_the_same_time_left(root, monkeypatch):
    run = "20261022-100000-synth"
    now = time.time()
    steps = _analyses_case(now, None)
    for s in steps:
        for k in ("eta_s", "eta_s_p90", "eta_basis", "live_lanes", "whole_step"):
            s.pop(k, None)
    d = _doc(run, steps, status="waiting")
    card = {"sid": "20261022-100100-batch", "state": "running", "started": now - 400, "phase_detail": "history n=3",
            "eta": {"p50": 91.5, "p90": 110.0, "basis": "progress_eta L2"}, "warnings": []}
    monkeypatch.setattr(A, "sessions_of", lambda r, limit_dirs=400: [{"sid": card["sid"], "step": "analyses"}])
    monkeypatch.setattr(activity, "session_card", lambda sid: dict(card))
    monkeypatch.setattr(activity, "_run_cost", lambda r: {"total": 0.0, "by_step": {}, "pods": []})
    c = activity.analysis_card(A.load(run), now)
    row = next(s for s in c["steps"] if s["id"] == "analyses")
    assert row["eta_basis"] == "measured" and row["eta_session_s"] == pytest.approx(91.5)
    assert row["eta_s"] == pytest.approx(c["live"]["left"]["analyses"]["p50"], abs=1)
    assert c["live"]["eta_s"]["p50"] == pytest.approx(row["eta_s"] + 120, abs=2)
    # the run is "waiting" on disk, but a step runs: the badge says running (2026-10-22 review)
    assert d["status"] == "waiting" and c["status"] == "running"


# ------------------------------------------------------------------------------------------------ badge

def test_driver_ends_running_not_waiting_while_a_step_runs(root, monkeypatch):
    monkeypatch.setattr(snapshots, "request", lambda *a, **k: False)
    run = "20261022-100001-synth"
    steps = [A._step("people", "people", "pod-gpu", [], 10, 20), A._step("world", "world", "pod-gpu", ["people"], 1, 2)]
    steps[0].update(status="running", started=time.time())
    _doc(run, steps)
    assert A.Driver(run, poll_s=0.05).loop() == 0
    assert A.load(run)["status"] == "running"
    assert A._final_status({"steps": [{"status": "waiting_agent"}, {"status": "done"}]}) == "waiting"
    assert A._final_status({"steps": [{"status": "failed"}, {"status": "blocked"}]}) == "failed"


# ------------------------------------------------------------------------------------------------ snapshots at their event

def test_a_snapshot_draws_the_state_of_its_event_not_of_its_render(root, monkeypatch):
    run = "20261022-100002-synth"
    steps = [A._step("camera", "camera", "pod-gpu", [], 10, 20)]
    _doc(run, steps, status="running")
    A.update_step(run, "camera", status="running", started=time.time())
    frozen = snapshots.freeze(run, "camera_start", None, time.time())
    assert frozen and frozen.exists() and frozen.parent.name == "pending"
    A.update_step(run, "camera", status="done", ended=time.time())       # the run moves on before the render
    pages = []
    monkeypatch.setattr(snapshots, "render_png", lambda hp, pp, h: (pages.append(hp.read_text(encoding="utf-8")) or
                                                                     ("edge", 1, None)))
    shot = snapshots.take(run, "camera_start", "camera started", data_file=frozen)
    assert shot["frozen"] is True and shot["ts"] <= shot["rendered"]
    assert 'pill info">running' in pages[0] and 'pill good">done' not in pages[0]
    assert not frozen.exists()                                           # consumed
    shot2 = snapshots.take(run, "later", "later")                       # no frozen file: gathered now
    assert shot2["frozen"] is False and 'pill good">done' in pages[1]
    assert snapshots.load_frozen(frozen, run) is None


# ------------------------------------------------------------------------------------------------ the generic plan + driver

def test_every_mono_step_has_a_generic_executor_and_pod_cpu_is_gone():
    steps = A.route_plan("mono", TRIAL, {})
    for s in steps:
        assert s["executor"] in ("builtin", "cmd"), s["id"]
        assert not (s["where"] == "pod-cpu" and s["id"] != "upload"), s["id"]    # the runner has no CPU route
    by = {s["id"]: s for s in steps}
    assert by["fits"]["deps"] == ["world", "sam3d_body"] and by["export"]["deps"] == ["analyses"]
    assert by["avatars"]["optional"] and by["film"]["optional"]
    assert A.mono_chunks(19.97, 30.0) == 1 and A.mono_chunks(88.1, 29.994) == 4
    for dur, fps in ((19.97, 30.0), (26.5, 30.0), (50.0, 30.0), (88.1, 29.994), (300.0, 30.0)):
        assert A.mono_chunks(dur, fps) == len(M.plan_chunks("x", M.grid_frames(dur, fps))), dur


def test_cmd_steps_report_their_result_skip_or_failure(root, monkeypatch):
    monkeypatch.setattr(snapshots, "request", lambda *a, **k: False)
    run = "20261022-100003-synth"
    code = ("import sys; t, r, k = sys.argv[1:4]; print('noise'); "
            "print('RESULT: made it for ' + t + ' in ' + r) if k == 'ok' else None; "
            "(print('SKIP: not built here'), sys.exit(3)) if k == 'skip' else None; "
            "(print('Traceback: boom'), sys.exit(1)) if k == 'fail' else None")
    steps = [A._step("a", "a", "local", [], 0.1, 0.2, executor="cmd", price="local", cmd=["-c", code, "{take}", "{run}", "ok"]),
             A._step("b", "b", "local", ["a"], 0.1, 0.2, executor="cmd", price="local", optional=True,
                     cmd=["-c", code, "{take}", "{run}", "skip"]),
             A._step("c", "c", "local", ["a"], 0.1, 0.2, executor="cmd", price="local", cmd=["-c", code, "{take}", "{run}", "fail"])]
    _doc(run, steps)
    A.Driver(run, poll_s=0.05).loop()
    by = {s["id"]: s for s in A.load(run)["steps"]}
    assert by["a"]["status"] == "done" and by["a"]["message"].startswith(f"made it for synth in {run}")
    assert by["b"]["status"] == "skipped" and by["b"]["message"].startswith("not built here")
    assert by["c"]["status"] == "failed" and "Traceback: boom" in by["c"]["message"] and "rc=1" in by["c"]["message"]


def test_pod_cmd_steps_wait_for_yes(root, monkeypatch):
    monkeypatch.setattr(snapshots, "request", lambda *a, **k: False)
    run = "20261022-100004-synth"
    st = A._pod_step("camera", "camera", "pod-gpu", [], 5, 8, executor="cmd")
    st["cmd"] = ["-c", "print('RESULT: x')"]
    _doc(run, [st])
    A.Driver(run, poll_s=0.05, allow_pods=False).loop()
    s = A.load(run)["steps"][0]
    assert s["status"] == "waiting_agent" and s["parked"] == "yes" and "--yes" in s["message"]
    A.Driver(run, poll_s=0.05, allow_pods=True).loop()
    assert A.load(run)["steps"][0]["status"] == "done"


# ------------------------------------------------------------------------------------------------ mono_route pure parts

def test_chunk_plan_and_stitch_list():
    one = M.plan_chunks("t", M.grid_frames(19.97, 30.0))
    assert len(one) == 1 and one[0]["k0"] == 0 and one[0]["k1"] == 598 and one[0]["take"] == "t_g0000"
    four = M.plan_chunks("t", M.grid_frames(88.1, 29.994))
    assert [c["id"] for c in four][0] == "g0000" and all(c["T"] <= M.CHUNK_MAX_S * 30 for c in four)
    assert four[-1]["k1"] == M.grid_frames(88.1, 29.994) - 1
    for a, b in zip(four, four[1:]):
        assert b["k0"] < a["k1"] and b["core_k0"] == a["core_k1"] + 1          # overlapping windows, cores tile the take
    sl = M.stitch_list_doc("t", four, 88.0)
    assert sl["partial"] and sl["mono"] == [[0.0, 88.0]] and [c["k"] for c in sl["chunks"]] == [[c["k0"], c["k1"]] for c in four]
    assert sl["chunks"][0]["ma_2d"] == "t_g0000/mamma/ma_2d/t_g0000_c1"


def test_beta0_solves_the_rest_height():
    rng = np.random.default_rng(0)
    V = 200
    vt = np.c_[rng.normal(0, 0.1, V), np.linspace(-0.9, 0.82, V), rng.normal(0, 0.1, V)]
    sd = np.zeros((V, 3, 16))
    sd[:, 1, 0] = np.linspace(-0.04, 0.04, V)          # beta0 stretches the body vertically (8 cm per unit)
    for target in (1.60, 1.72, 1.83):
        b = M.beta0_for_height(vt, sd, target)
        assert len(b) == M.BETAS_N and np.all(b[1:] == 0)
        assert M.rest_height(vt, sd, b) == pytest.approx(target, abs=1e-4)


def test_frames_json_and_session_failure_rules():
    t = np.arange(5) / 30.0
    d = M.frames_json_doc(t, t, (480, 848), "v.mp4")
    assert d["frame_count"] == 5 and d["frames"][2]["file"] == "000002.jpg" and d["width"] == 480
    assert M.session_failure([], 1)["retry"]
    assert M.session_failure([{"sid": "s", "error": "PodError: no GPU"}], 0)["retry"]
    assert M.session_failure([{"sid": "s", "bootstrap_rc": 2, "jobs": {}}], 0)["retry"]
    real = M.session_failure([{"sid": "s", "jobs": {"j": {"status": "failed: [fit]", "commands": {"build": 0, "fit": 1}}}}], 0)
    assert not real["retry"] and "command fit" in real["msg"]
    net = M.session_failure([{"sid": "s", "jobs": {"j": {"status": "failed", "commands": {"install": 1}}}}], 0)
    assert net["retry"]
    assert M.session_max_cost(30) == pytest.approx(0.9) and M.session_max_cost(5) == 0.6


def _mono_take(tmp_path: Path, monkeypatch, name: str = "lk_mono") -> Path:
    """A synthetic one-video take in a temp package root (takes/, work/, the video, cameras/01.npz)."""
    import dancecap.take as TK
    pkg = tmp_path / "ws" / "dancecap"
    (pkg / "takes").mkdir(parents=True)
    vdir = tmp_path / "ws" / "dancecap" / "work" / "trial_src" / name
    vdir.mkdir(parents=True)
    (vdir / f"{name}.mp4").write_bytes(b"0" * 1000)
    al = pkg / "work" / name / "clock" / "alignment.json"
    al.parent.mkdir(parents=True)
    al.write_text(json.dumps({"reference": f"{name}.mp4", "sources": [{"id": "01", "name": f"{name}.mp4", "slope": 1.0,
                                                                      "offset": 0.0}]}))
    (pkg / "takes" / f"{name}.toml").write_text(
        f'name = "{name}"\nsource_dir = "../work/trial_src/{name}"\nalignment = "../work/{name}/clock/alignment.json"\n'
        'window = [0.0, 19.93]\n[timeline]\nanchor = 0.0\nhz = 30.0\n'
        '[dancers.lead]\nname = "Kadu"\nstature_m = 1.727\nshoe_sole_m = 0.025\n'
        '[dancers.follow]\nname = "Larissa"\nstature_m = 1.651\nshoe_sole_m = 0.025\n', encoding="utf-8")
    cam = pkg / "work" / name / "cameras"
    cam.mkdir(parents=True)
    t = np.arange(599) / 30.0
    np.savez(cam / "01.npz", t=t, pts=t, frame_index=np.arange(599), K=np.tile(np.eye(3), (599, 1, 1)),
             native_size=np.array([480, 848]))
    for mod in (TK, M):
        monkeypatch.setattr(mod, "PACKAGE_ROOT", pkg)
    monkeypatch.setattr(M, "WORKSPACE", pkg.parent)
    return pkg


def test_job_writers_map_the_video_and_never_name_john_lula(tmp_path, monkeypatch):
    pkg = _mono_take(tmp_path, monkeypatch)
    ctx = M.Ctx("lk_mono", run="20261022-100005-lk_mono", step="camera", dry_run=True)
    assert ctx.video_pod == "dances/lk_mono/lk_mono.mp4" and ctx.video_arg == "../dances/lk_mono/lk_mono.mp4"
    calls = []
    monkeypatch.setattr(M.Ctx, "remote", lambda self, jobs, **kw: calls.append((jobs, kw)) or [])
    M.step_camera(ctx)
    M.step_people(ctx)
    (pkg / "work" / "lk_mono" / "song").mkdir(parents=True)
    (pkg / "work" / "lk_mono" / "song" / "segments.json").write_text(json.dumps({"no_beat": [[0, 3.2], [18.5, 19.9]]}))
    M.step_analyses(ctx)
    jobs = [json.loads(p.read_text(encoding="utf-8")) for c in calls for p in c[0]]
    names = [j["name"] for j in jobs]
    assert names == ["lk_mono_camera", "lk_mono_people", "lk_mono_analyses"]
    cam, ppl, ana = jobs
    assert cam["inputs"][0]["pod"] == "dances/lk_mono/lk_mono.mp4" and "cameras" in cam["needs"]
    assert any("mono_cam skeleton" in c["cmd"] for c in cam["commands"]) and cam["min_vram_gb"] == 24
    assert ppl["needs"] == ["dancers"] and [c["name"] for c in ppl["commands"]][:2] == ["fetch_sam2", "detect"]
    assert "--no-beat 0:3.2,18.5:19.9" in ana["commands"][0]["cmd"] and ana["needs"] == []
    assert [c["name"] for c in ana["commands"]] == ["timing", "physics", "counterbalance", "floorcraft"]
    blob = json.dumps(jobs)
    assert "john_lula" not in blob and "trial_mono_jl20" not in blob


def test_pinned_betas_and_chunk_jobs(tmp_path, monkeypatch):
    pkg = _mono_take(tmp_path, monkeypatch)
    V = 120
    model = pkg.parent / "mamma" / "data" / "body_models" / "smplx_locked_head" / "smplx" / "SMPLX_NEUTRAL.npz"
    model.parent.mkdir(parents=True)
    sd = np.zeros((V, 3, 16))
    sd[:, 1, 0] = np.linspace(-0.04, 0.04, V)
    np.savez(model, v_template=np.c_[np.zeros(V), np.linspace(-0.9, 0.82, V), np.zeros(V)], shapedirs=sd)
    ctx = M.Ctx("lk_mono", run="20261022-100006-lk_mono", step="fits", dry_run=True)
    pin = M.write_pinned_betas(ctx)
    doc = json.loads(pin.read_text())
    src = json.loads(pin.with_name("mamma_pin_betas.source.json").read_text())
    assert set(doc) == {"0", "1"} and src["0"]["role"] == "lead" and src["0"]["target_rest_height_m"] == pytest.approx(1.752)
    assert src["1"]["rest_height_m"] == pytest.approx(1.676, abs=1e-3) and doc["0"][0] > doc["1"][0]
    c = M.plan_chunks("lk_mono", 598)[0]
    fake = {"job": pkg / "work" / "lk_mono" / "farm" / "jobs" / f"{c['id']}.json", "prior": pkg / "work" / "lk_mono" / "p.npz"}
    import dancecap.front_pod as FP
    monkeypatch.setattr(FP, "MA2D_STRIPPED", "dancecap/ckpt.ckpt")
    front_p, fit_p = M.chunk_jobs(ctx, c, fake, pin)
    front, fit = json.loads(front_p.read_text()), json.loads(fit_p.read_text())
    assert fit["after"] == [front["name"]] and front["env"]["MAMMA_MONO"] == "1" and "MAMMA_POSE_PRIOR" not in front["env"]
    assert fit["env"]["MAMMA_FREEZE_BETAS"] == "1" and fit["env"]["HM_FRONT_DANCE"] == "lk_mono"
    assert [x["name"] for x in front["commands"]] == ["frames", "cameras", "masks", "ma2d", "sync", "prepare", "mono_init", "qa_mask"]
    assert [x["name"] for x in fit["commands"]] == ["build", "fit", "select", "refine"]
    assert any(isinstance(i, dict) and i["pod"] == M.MA2D_POD for i in front["inputs"])
    # the stripped checkpoint is mapped to that arc: an exclude of it dropped the file from the bundle (first fits check)
    from dancecap.remote_jobs import _excluded
    assert not any(_excluded(M.MA2D_POD, j["exclude"]) for j in (front, fit))
    assert f"--grid-k={c['k0']}:{c['k1']}" in front["commands"][5]["cmd"]


# ------------------------------------------------------------------------------------------------ take-specific code made generic

def test_sam3db_configure_and_size_rules():
    from dancecap.mono4d import jl_sam3db as S
    try:
        S.configure("lk_mono_x")
        assert S.WORK.name == "lk_mono_x" and S.JOB.parent == S.WORK / "jobs" and S.JOB_NAME == "lk_mono_x_sam3d_body"
        assert S.OUT == S.WORK / "realism" / "sam3db" and not S.has_fit()
        assert S.size_rules(538, 960) == (1500, 80.0)
        px, bh = S.size_rules(480, 848)
        assert px == round(1500 * 480 * 848 / (538 * 960)) and bh == pytest.approx(80 * 848 / 960)
    finally:
        S.configure("john_lula")


def test_export_and_front_follow_the_take(monkeypatch):
    from dancecap import jl_export as JE, jl_front as JF
    assert JE.default_capture("lk03_mono20") == "Mono_Lk03Mono20"
    monkeypatch.setattr(JF.RF, "DANCE", "lk03_mono20")
    assert JF.people_dir().parts[-2:] == ("lk03_mono20", "people")
