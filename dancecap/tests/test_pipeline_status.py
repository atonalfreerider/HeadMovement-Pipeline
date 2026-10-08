"""dancecap.pipeline.status / camtypes / estimate on a synthetic take: real status from files, staleness rules,
camera types (owned-block TOML writes, states, invalidation), estimates and budget. No pods, no real takes."""
import json
import os
import shutil
import sys
import time
import tomllib
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pipeline_fixtures import make_root, make_take, touch  # noqa: E402
from dancecap.pipeline import camtypes, estimate, paths, registry, runs, status, takes  # noqa: E402

REAL_TAKES = Path(__file__).resolve().parents[1] / "takes"


@pytest.fixture()
def root(tmp_path):
    r = make_root(tmp_path)
    yield r
    paths.set_root(None)


def st_of(take):
    snap = status.snapshot(take, with_activity=False)
    return {x["id"]: x for x in snap["stages"]}, snap


def test_fresh_take_status(root):
    make_take(root)
    s, snap = st_of("synth")
    assert snap["variant"] == "multi"
    assert s["intake"]["status"] == "done"
    assert s["audio_sync"]["status"] == "done"          # the TOML's alignment file exists
    assert s["window"]["status"] == "done"
    assert s["probe"]["status"] == "ready"
    assert s["camera_types"]["status"] == "blocked"     # no types, no suggestion, cameras not solved yet
    assert s["ingest"]["status"] == "ready"
    assert s["cam_map"]["status"] == "waiting"
    assert s["scale_probe"]["status"] == "planned" and s["pair_calib"]["status"] if "pair_calib" in s else True
    assert s["room"]["status"] == "skipped"              # optional
    assert s["song_align"]["status"] == "skipped"        # no soundtrack
    assert s["unity_check"]["status"] == "n/a"
    assert snap["remaining"]["p50_s"] > 0 and snap["critical_path"]
    summ = status.summary("synth", snap)
    assert summ["n_used"] == 3 and summ["camera_types"]["unset"] == 3


def test_done_stale_accept_and_implied(root):
    make_take(root)
    w = root / "work" / "synth"
    now = time.time()
    for c in ("01", "02", "03"):
        touch(w / "frames" / c / "frames.json", now - 300, '{"frame_count": 600}')
        touch(w / "keyframes" / c / "keyframes.json", now - 300)
        touch(w / "people" / f"{c}.npz", now - 400)        # detect older than ingest -> stale
    touch(w / "people" / "report.json", now - 400)
    s, _ = st_of("synth")
    assert s["ingest"]["status"] == "done"
    assert s["detect"]["status"] == "stale" and "ingest is newer" in s["detect"]["outputs"]["stale_because"]
    assert "accept_inputs" in s["detect"]["actions"]
    status.accept_inputs("synth", "detect")
    s, _ = st_of("synth")
    assert s["detect"]["status"] == "done"
    touch(w / "frames" / "01" / "frames.json", time.time() + 5, '{"frame_count": 600}')  # a real change again
    s, _ = st_of("synth")
    assert s["detect"]["status"] == "stale"
    # MAMMA output without the chunk plan: mamma_prepare is implied done, scale_probe (not built) skipped
    for f in ("smplx/lead.npz", "smplx/follow.npz", "mamma/select.json"):
        touch(w / f, time.time() + 10)
    s, _ = st_of("synth")
    assert s["mamma_fit"]["status"] == "done"
    assert s["mamma_prepare"]["status"] == "done" and "implied" in s["mamma_prepare"]["status_reason"]
    assert s["scale_probe"]["status"] == "planned"   # soft input of mamma_fit: shown as not built yet
    assert s["cam_refine_body"]["status"] == "planned"


def test_running_and_failed_from_runs(root):
    make_take(root)
    rid = runs.new_id()
    d = paths.runs_dir() / rid
    d.mkdir(parents=True)
    (d / "run.json").write_text(json.dumps({"id": rid, "kind": "stage", "take": "synth", "stages": ["ingest"],
                                            "created": time.time()}))
    import psutil
    (d / "pid.json").write_text(json.dumps({"pid": os.getpid(), "create_time": psutil.Process().create_time(),
                                            "started": time.time()}))
    s, _ = st_of("synth")
    assert s["ingest"]["status"] == "running" and rid in s["ingest"]["status_reason"]
    (d / "rc.json").write_text(json.dumps({"rc": 2, "ended": time.time()}))
    s, _ = st_of("synth")
    assert s["ingest"]["status"] == "failed"
    assert runs.describe(rid)["state"] == "failed"


def test_two_and_single_view_variants(root):
    make_take(root, "pair", cams=("01", "02"))
    s, snap = st_of("pair")
    assert snap["variant"] == "two" and s["pair_calib"]["status"] == "planned" and "sync" not in s
    make_take(root, "solo", cams=("01",))
    s, snap = st_of("solo")
    assert snap["variant"] == "single"
    assert s["mono_camera"]["status"] == "planned" and "cam_map" not in s
    assert "planned" in (s["mono_hmr"]["status"],)


# ------------------------------------------------------------------------------------------------ camera types

def test_camera_type_states_and_effects(root):
    make_take(root)
    types, etag = camtypes.load("synth", ["01", "02"])
    assert [t["state"] for t in types] == ["unset", "unset"]
    assert types[0]["effective"]["class"] == "handheld_zoom"  # the default = today's solver
    t, etag2 = camtypes.set_type("synth", "01", "tripod", False, etag)
    assert t["state"] == "confirmed" and t["effective"]["class"] == "tripod" and t["source"] == "user"
    assert etag2 != etag
    with pytest.raises(camtypes.StaleEtag):
        camtypes.set_type("synth", "02", "handheld", True, etag)  # old etag
    cfg = tomllib.loads((root / "takes" / "synth.toml").read_text(encoding="utf-8"))
    assert cfg["cameras"]["01"]["mount"] == "tripod" and cfg["cameras"]["01"]["zoom"] is False
    assert cfg["cameras"]["scale_rule"] == "chain" and cfg["cameras"]["loc"] == {"kf_by_time": True}
    assert camtypes.loc_factor(t) == pytest.approx(0.1)  # a locked tripod: keyframes only (cameras/typed.py)
    assert camtypes.loc_factor(t, conservative=True) == pytest.approx(1.1)  # keyframes refused: localised twice
    # an automatic suggestion never overrides the user; on an unset camera it pre-fills the values
    camtypes.write_suggestion("synth", "01", {"mount": "handheld", "zoom": True, "confidence": 0.9, "by": "probe_a"})
    camtypes.write_suggestion("synth", "02", {"mount": "tripod", "zoom": False, "confidence": 0.8, "by": "probe_a"})
    types, etag3 = camtypes.load("synth", ["01", "02", "03"])
    assert types[0]["state"] == "confirmed" and types[0]["mount"] == "tripod"
    # a suggestion pre-fills the review screen; the camera stages apply only the user's answer (shared module)
    assert types[1]["state"] == "suggested" and types[1]["mount"] == "tripod"
    assert types[1]["effective"]["class"] == "handheld_zoom"
    assert camtypes.loc_factor(types[1], conservative=True) == 1.0  # p90 prices unconfirmed as handheld + zoom
    assert camtypes.needs_user(types[2]) and not camtypes.needs_user(types[1])
    types, _ = camtypes.accept("synth", ["02", "03"], 0.7, etag3)
    assert types[0]["state"] == "confirmed" and types[0]["mount"] == "tripod" and types[1]["state"] == "unset"
    # probe B contradicts the user -> conflict (never an override)
    touch(root / "work" / "synth" / "probe" / "camera_types_probe_b.json",
          text=json.dumps({"cameras": {"01": {"mount": "handheld", "zoom": False, "confidence": 0.95, "by": "probe_b",
                                              "at": time.time() + 5, "evidence": {"centre_p95_m": 0.14}}}}))
    types, _ = camtypes.load("synth", ["01"])
    assert types[0]["state"] == "conflict" and types[0]["mount"] == "tripod"
    assert types[0]["conflict"]["field"] == "mount" and camtypes.needs_user(types[0])


def test_owned_block_writer_keeps_every_other_byte_of_the_real_takes(root):
    for name in ("larissa_kadu", "larissa_kadu_38"):
        src = REAL_TAKES / f"{name}.toml"
        if not src.exists():
            pytest.skip("real take TOMLs not present")
        dst = root / "takes" / f"{name}.toml"
        shutil.copyfile(src, dst)
        before = dst.read_text(encoding="utf-8")
        _, etag = camtypes.load(name, ["01"])
        camtypes.set_type(name, "05", "tripod", True, etag)
        _, etag = camtypes.load(name, ["05"])
        camtypes.set_type(name, "05", "handheld", False, etag)  # rewrite the same block
        _, etag = camtypes.load(name, ["01"])
        camtypes.set_type(name, "01", "handheld", True, etag)
        after = dst.read_text(encoding="utf-8")
        assert after.startswith(before.rstrip("\n"))  # every original byte kept; blocks appended
        cfg_a, cfg_b = tomllib.loads(before), tomllib.loads(after)
        for k in cfg_a:
            if k != "cameras":
                assert cfg_a[k] == cfg_b[k], k
        for k in ("scale_reference", "loc", "scale_rule"):
            assert cfg_a.get("cameras", {}).get(k) == cfg_b["cameras"].get(k)
        assert cfg_b["cameras"]["05"]["mount"] == "handheld" and cfg_b["cameras"]["05"]["zoom"] is False
        assert after.count('[cameras."05"]') == 1
        assert list((root / "work" / name / "monitor" / "toml_backups").glob("*.toml"))


def test_camera_type_change_makes_camera_stages_and_descendants_stale(root):
    make_take(root)
    w = root / "work" / "synth"
    t0 = time.time() - 1000
    for f in ("cameras/sfm/images.bin", "cameras/sfm_work/map.json", "cameras/cache/loc_01.npz",
              "cameras/world_from_sfm.json", "cameras/report.json"):
        touch(w / f, t0)
    s, _ = st_of("synth")
    assert s["cam_loc"]["status"] == "done"
    _, etag = camtypes.load("synth", ["01"])
    stat = {k: v["status"] for k, v in s.items()}
    vst = registry.variant_stages("multi")
    inval = camtypes.invalidated_by_change("synth", "01", "tripod", False, stat,
                                           lambda roots: registry.descendants(vst, roots))
    assert "cam_loc" in inval and "cam_world" in inval
    assert camtypes.invalidated_by_change("synth", "01", "handheld", True, stat,
                                          lambda roots: registry.descendants(vst, roots)) == []
    camtypes.set_type("synth", "01", "tripod", False, etag)
    s, _ = st_of("synth")
    assert s["cam_loc"]["status"] == "stale"
    assert any("camera type 01 changed" in x for x in s["cam_loc"]["outputs"]["stale_because"])
    assert s["cam_world"]["status"] == "stale"            # the semantic change propagates downstream
    assert "accept_inputs" not in s["cam_loc"]["actions"]


def test_type_staleness_uses_the_recorded_policy_and_refused_types_warn(root):
    """The stale check compares with the class under the policy the cameras run used (report.json camera_types.policy):
    a run with --camera-types auto is not stale forever. A confirmed type the cameras stage refused is a warning (the
    camera_types step's reason, summary 'refused', a monitor alert), not a block."""
    make_take(root)
    w = root / "work" / "synth"
    t0 = time.time() - 1000
    for f in ("cameras/sfm/images.bin", "cameras/sfm_work/map.json", "cameras/cache/loc_01.npz",
              "cameras/world_from_sfm.json"):
        touch(w / f, t0)
    camtypes.write_suggestion("synth", "01", {"mount": "tripod", "zoom": False, "confidence": 0.9, "by": "probe_a"})
    rep = {"camera_types": {"policy": "auto", "applied": {"01": {"class": "tripod", "source": "auto"}}}}
    touch(w / "cameras" / "report.json", t0 + 1, text=json.dumps(rep))
    s, _ = st_of("synth")
    assert s["cam_loc"]["status"] == "done", s["cam_loc"]["outputs"]["stale_because"]
    rep["camera_types"]["policy"] = "user"  # the same record under the default policy: the suggestion is not applied
    touch(w / "cameras" / "report.json", t0 + 2, text=json.dumps(rep))
    s, _ = st_of("synth")
    assert s["cam_loc"]["status"] == "stale"
    assert any("tripod -> handheld_zoom" in x for x in s["cam_loc"]["outputs"]["stale_because"])
    _, etag = camtypes.load("synth", ["02"])
    camtypes.set_type("synth", "02", "tripod", False, etag)
    rep = {"camera_types": {"policy": "user", "applied": {"02": {"class": "tripod", "source": "user"}}},
           "cams": {"02": {"camera_type": {"class": "tripod", "model": "fixed_focal", "at": "2026-10-07T12:00:00Z",
                                           "check": {"passed": False, "why": "per-frame poses needed: the camera "
                                                                             "moves (not a locked tripod)"}}}}}
    touch(w / "cameras" / "report.json", t0 + 3, text=json.dumps(rep))
    s, snap = st_of("synth")
    assert s["camera_types"]["status"] == "done"  # a warning, not a block
    assert "could not use your type for 02" in (s["camera_types"]["status_reason"] or "")
    assert status.summary("synth", snap)["camera_types"]["refused"] == 1


# ------------------------------------------------------------------------------------------------ estimates, budget

def test_estimates_use_camera_types_and_risk(root):
    make_take(root)
    cfg, _ = takes.load_cfg("synth")
    wl = estimate.workload("synth", cfg)
    assert wl["C"] == 3 and wl["V"] == pytest.approx(10 / 60, abs=1e-3)
    loc = registry.get("cam_loc")
    types, etag = camtypes.load("synth", ["01", "02", "03"])
    e_def = estimate.stage_estimate(loc, wl, types)
    for c in ("01", "02", "03"):
        _, etag = camtypes.load("synth", [c])
        camtypes.set_type("synth", c, "tripod", False, etag)
    types, _ = camtypes.load("synth", ["01", "02", "03"])
    e_tri = estimate.stage_estimate(loc, wl, types)
    assert e_tri["p50_s"] < 0.2 * e_def["p50_s"] and e_tri["billed"]
    mf = estimate.stage_estimate(registry.get("mamma_fit"), wl)
    assert mf["p90_s"] > mf["p50_s"] and mf["branches"][1]["eta_p50_s"] > mf["branches"][0]["eta_p50_s"]
    lb = estimate.live_blend(1000, 1600, 0.5, 300)
    assert 250 < lb["p50_s"] < 600 and lb["p10_s"] <= lb["p50_s"] <= lb["p90_s"]


def test_budget_and_launch_gate_from_a_fake_ledger(root, monkeypatch):
    monkeypatch.setenv("DANCECAP_RUNPOD_CAP", "30")
    now = time.time()
    led = [{"pod_id": "aaaa1111", "cost_per_hr": 1.2, "created_ts": now - 7200, "deleted_ts": now - 3600, "cost": 1.2},
           {"pod_id": "bbbb2222", "cost_per_hr": 1.5, "created_ts": now - 1800, "max_minutes": 60}]
    touch(root / "work" / "runpod" / "ledger.json", text=json.dumps(led))
    b = estimate.budget(now=now)
    assert b["spent"] == pytest.approx(1.2 + 0.75, abs=0.01)
    assert b["running"]["worst_usd"] == pytest.approx(0.75, abs=0.02) and b["open_pods"] == 1
    g = estimate.launch_gate(5.0, now=now)
    assert g["gate"] == "ok"
    assert estimate.launch_gate(26.5, now=now)["gate"] == "over_reserve"  # 1.95 + 0.75 + 26.5 > 30 - 2
    assert estimate.launch_gate(40.0, now=now)["gate"] == "over_cap"


def test_running_stage_progress_from_progress_events(root):
    """An instrumented stage (dancecap.progress, L2) reports done/total: the snapshot shows it with a live ETA."""
    make_take(root)
    rid = runs.new_id()
    d = paths.runs_dir() / rid
    d.mkdir(parents=True)
    (d / "run.json").write_text(json.dumps({"id": rid, "kind": "stage", "take": "synth", "stages": ["ingest"],
                                            "created": time.time() - 60}))
    import psutil
    (d / "pid.json").write_text(json.dumps({"pid": os.getpid(), "create_time": psutil.Process().create_time(),
                                            "started": time.time() - 60}))
    pf = root / "work" / "synth" / "progress" / f"ingest.{rid}.4242.jsonl"
    evs = [{"v": 1, "ts": time.time() - 50, "job": "ingest", "step": "ingest/01", "kind": "start", "total": 600,
            "unit": "frames"},
           {"v": 1, "ts": time.time() - 5, "job": "ingest", "step": "ingest/01", "kind": "tick", "done": 300,
            "total": 600, "unit": "frames", "msg": "cam 01"}]
    touch(pf, text="\n".join(json.dumps(e) for e in evs) + "\n")
    prog, steps = status.progress_events("synth", "ingest", rid)
    assert prog["frac"] == 0.5 and steps[0]["step"] == "ingest/01"
    s, _ = st_of("synth")
    x = s["ingest"]
    assert x["status"] == "running" and x["progress"]["frac"] == 0.5
    assert x["eta"]["basis"].startswith("live") and 0 < x["eta"]["p50_s"] < 600
