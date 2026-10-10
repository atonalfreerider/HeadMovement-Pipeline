"""Unit tests for the recap chunk planner (farm/chunks.py equal windows, farm/recap_profile.py) and dancecap.recap_front's pure parts.

No pod, no footage: synthetic teacher coverage, a synthetic dance spec, temporary work directories.

    .venv-nopo4d\\Scripts\\python.exe -m pytest dancecap/tests/test_recap_plan.py -q      (from the workspace root)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dancecap import recap_front as rf  # noqa: E402
from dancecap.farm import chunks as C  # noqa: E402
from dancecap.farm import recap_profile as RP  # noqa: E402
from dancecap.farm.timeline import DanceSpec, Grid  # noqa: E402


def recap_spec(profile="recap", **chunks):
    cfg = {"name": "larissa_kadu_recap", "chunks": {"chunk_s": 60.0, "overlap_s": 4.0, "mode": "windows", "profile": profile, **chunks}}
    return DanceSpec(name="larissa_kadu_recap", grid=Grid(0.0, 0, 10814, 30.0), cams=["00", "01", "02", "03", "05", "06"], cfg=cfg)


# ------------------------------------------------------------------------------------------------ the arithmetic


@pytest.mark.parametrize("overlap_s", [3.0, 4.0, 5.0, 6.0])
def test_equal_windows_cover_the_class_with_equal_windows_of_at_most_60_s(overlap_s):
    N = 10815
    e = C.equal_windows(N, 30.0, 60.0, overlap_s)
    assert e["n"] == 7                                       # 6 windows would need 62.6-65.6 s
    wins, cores, o = e["windows"], e["cores"], e["overlap_k"]
    assert wins[0][0] == 0 and wins[-1][1] == N - 1
    assert max(b - a + 1 for a, b in wins) <= 60 * 30 and max(b - a + 1 for a, b in wins) - min(b - a + 1 for a, b in wins) <= 1
    for (a0, b0), (a1, b1) in zip(wins, wins[1:]):
        assert abs((b0 - a1 + 1) - o) <= 1                   # neighbours overlap by overlap_s (+-1 frame)
    assert cores[0][0] == 0 and cores[-1][1] == N - 1
    assert all(c1[0] == c0[1] + 1 for c0, c1 in zip(cores, cores[1:]))          # the cores tile the timeline
    for (a, b), (ca, cb) in zip(wins, cores):
        assert a <= ca <= cb <= b
    for j in range(6):                                       # a seam is cut in the middle of its overlap
        mid = (wins[j + 1][0] + wins[j][1] + 1) / 2
        assert abs(cores[j + 1][0] - mid) <= 1


def test_the_naive_60_s_stepping_would_leave_a_tiny_last_chunk_and_the_planner_avoids_it():
    N, w, o = 10815, 1800, 120
    starts = list(range(0, N - o, w - o))
    assert (N - starts[-1]) / 30 < 30                        # 24.5 s
    e = C.equal_windows(N, 30.0, 60.0, 4.0)
    assert min(e["window_s"]) > 54 and [t["n"] for t in e["tried"]][-1] == 7 and e["tried"][-2]["window_s"] > 60


def test_a_short_timeline_is_one_chunk_and_the_params_come_from_the_toml():
    assert C.equal_windows(900, 30.0, 60.0, 4.0)["n"] == 1
    assert C.chunk_params(recap_spec()) == {"chunk_s": 60.0, "overlap_s": 4.0, "mode": "windows", "profile": "recap"}
    demo = DanceSpec(name="x", grid=Grid(0.0, 0, 99), cfg={})
    p = C.chunk_params(demo)
    assert p["profile"] == "" and p["mode"] == "cores" and p["chunk_s"] == C.CORE_TARGET_S


def test_plan_gaps_core_target_changes_the_split():
    cores = C.split_gap(10815, 30.0, 60.0, 48.0, 72.0)
    assert len(cores) == 6 and sum(cores) == 10815 and set(cores) <= {1802, 1803}
    assert len(C.split_gap(10815, 30.0)) == 12               # the demo's 30 s default is unchanged


def test_plan_windows_ids_cams_take_names_and_claimed_skip():
    spec = recap_spec()
    spans = {c: (0.0, 360.0) for c in spec.cams}
    ch, rep = C.plan_windows(spec, spans=spans, cams_fn=lambda k0, k1: ["01", "02"])
    assert len(ch) == 7 and ch[0].id == "g0000" and ch[0].take == "larissa_kadu_recap_g0000" and ch[0].cams == ["01", "02"]
    assert ch[0].k0 == 0 and ch[-1].k1 == 10814 and ch[0].core_k1 + 1 == ch[1].core_k0
    assert rep["window_s"] == [round(c.T / 30, 2) for c in ch] and rep["overlap_s"] == 4.0
    ch2, rep2 = C.plan_windows(spec, spans=spans, claimed_ids={ch[1].id, ch[3].id}, cams_fn=lambda a, b: [])
    assert [c.id for c in ch2] == [c.id for i, c in enumerate(ch) if i not in (1, 3)] and len(rep2["claimed_skipped"]) == 2


# ------------------------------------------------------------------------------------------------ the camera rule and the shape window


def coverage(spans):
    """{cam: teacher arrays} on a 30 Hz clock: spans = {cam: (t0, t1, lead, follow, both)}: shares of frames with the lead / the follow / both, spread
    evenly over time (a repeating pattern of 100 frames)."""
    out = {}
    for c, (t0, t1, lead, follow, both) in spans.items():
        t = np.arange(t0, t1, 1 / 30)
        j = np.arange(len(t)) % 100
        nb, nl, nf = int(round(100 * both)), int(round(100 * (lead - both))), int(round(100 * (follow - both)))
        conf = np.zeros((len(t), 2))
        conf[j < nb] = 0.9
        conf[(j >= nb) & (j < nb + nl), 0] = 0.9
        conf[(j >= nb + nl) & (j < nb + nl + nf), 1] = 0.9
        out[c] = {"t": t, "ok": np.ones(len(t), bool), "conf": conf}
    return out


def test_window_stats_and_camera_rule():
    tt = coverage({"01": (0, 360, 0.9, 0.9, 0.85), "02": (0, 360, 0.5, 0.2, 0.1), "03": (60, 360, 0.9, 0.9, 0.9), "05": (7, 360, 0.7, 0.7, 0.4)})
    st = RP.window_stats(tt, 30.0, 70.0)
    assert st["03"]["frames"] > 0 and st["03"]["record_share"] < RP.MIN_RECORD_SHARE      # records only the last 10 s of the 40 s window
    cams, why = RP.choose_cams(st)
    assert cams == ["01", "05"] and "records" in why["03"] and "only" in why["02"] and "attached" in why["05"]
    cams2, _ = RP.choose_cams(RP.window_stats(tt, 100.0, 155.0))
    assert cams2 == ["01", "03", "05"]                                       # 02 sees both teachers in 10 % of its frames only; lead 0.5 < 0.6: out


def test_one_teacher_in_most_frames_is_enough_when_the_other_is_sometimes_there():
    st = {"01": {"frames": 1600, "record_share": 0.97, "lead": 0.8, "follow": 0.2, "both": 0.1}}
    assert RP.choose_cams(st)[0] == ["01"]
    st["01"].update(follow=0.05)
    assert RP.choose_cams(st)[0] == []


def test_shape_window_is_the_best_covered_30_s():
    tt = {}
    for c in ("00", "01", "02"):
        t = np.arange(0, 360, 1 / 30)
        conf = np.full((len(t), 2), 0.1)
        conf[(t >= 150) & (t < 190)] = 0.9                                   # all three phones see both teachers only here
        tt[c] = {"t": t, "ok": np.ones(len(t), bool), "conf": conf}
    sw = RP.shape_window(tt)
    assert 150 <= sw["t0"] <= 160 and sw["k1"] - sw["k0"] + 1 == 900


# ------------------------------------------------------------------------------------------------ estimates and job specs


def test_estimates_follow_the_drill_model_and_the_cost_table_adds_up():
    spec = recap_spec()
    ch, _ = C.plan_windows(spec, spans={c: (0.0, 360.0) for c in spec.cams}, cams_fn=lambda a, b: ["00", "01", "02", "03", "05", "06"])
    e = RP.estimate_chunk(ch[2], cam_frames=ch[2].T * 6)
    assert e["fit_min"] == pytest.approx((74 + 0.92 * ch[2].T) / 60, abs=0.06)
    assert e["total_min_4090"] == pytest.approx(e["front_min"] + e["fit_min"] + e["refine_min"] + e["quality_min"] + e["boot_min"], abs=0.2)
    assert e["usd_4090"] == pytest.approx(e["total_min_4090"] / 60 * 0.89, abs=0.01) and e["total_min"] == pytest.approx(e["total_min_4090"] / 0.45, abs=0.2)
    shape = RP.estimate_chunk(C.Chunk(id="shape", k0=0, k1=899, core_k0=0, core_k1=899, cams=["00", "01"]), shape=True)
    assert shape["refine_min"] == 0 and shape["quality_min"] == 0
    jobs = []
    for c in ch:
        c.estimate = RP.estimate_chunk(c, cam_frames=c.T * 6)
        jobs.append({"id": c.id, "what": "chunk", "cams": c.cams, "estimate": c.estimate})
    jobs.append({"id": "shape", "what": "shape", "cams": ["00"], "estimate": shape})
    tab = RP.cost_table(jobs)
    assert tab["usd"] == pytest.approx(sum(j["estimate"]["usd_4090"] for j in jobs), abs=0.02) and len(tab["jobs"]) == 8


def test_chunk_job_pins_the_shape_betas_waits_for_them_in_a_step_and_has_the_step_order():
    spec = recap_spec()
    ch, _ = C.plan_windows(spec, spans={c: (0.0, 360.0) for c in spec.cams}, cams_fn=lambda a, b: ["00", "01", "02"])
    ch[3].estimate = RP.estimate_chunk(ch[3], cam_frames=ch[3].T * 3)
    j = RP.build_job(ch[3], spec, hash_inputs=False, code={})
    # F1 2026-10-09: claimable at once; the front runs in parallel with the shape pass and only the fit waits for its betas (step await_pin)
    assert [s["name"] for s in j["steps"]] == ["chunk_take", "front", "await_pin", "fit", "refine", "quality", "evidence"]
    assert next(s for s in j["steps"] if s["name"] == "await_pin")["cmd"].startswith("python -m dancecap.recap_front await-pin ")
    fit = next(s for s in j["steps"] if s["name"] == "fit")
    assert fit["env"]["MAMMA_FREEZE_BETAS"] == "1" and fit["env"]["MAMMA_INIT_BETAS"].endswith("larissa_kadu_recap_shape/mamma_pin_betas.json") and fit["cmd"].endswith("--pin")
    assert j["after"] == [] and j["world_id"] is None and j["needs_gpu"] and j["kind"] == "full"
    assert not any(i["path"] == "chunks/shape/out/result.tar" for i in j["inputs"])
    assert any(o["path"].endswith(f"{ch[3].take}/smplx_rf2_noslip/") for o in j["outputs"])
    assert next(s for s in j["steps"] if s["name"] == "refine")["cmd"].startswith("python -m dancecap.pose_refine larissa_kadu_recap_g")


def test_shape_job_fits_free_and_writes_the_pin_instead_of_refining():
    spec = recap_spec()
    shape = C.Chunk(id="shape", k0=3000, k1=3899, core_k0=3000, core_k1=3899, overlap_k=0, cams=["00", "01", "02", "03"], take="larissa_kadu_recap_shape")
    shape.estimate = RP.estimate_chunk(shape, shape=True)
    j = RP.build_job(shape, spec, hash_inputs=False, code={}, shape=True)
    assert [s["name"] for s in j["steps"]] == ["chunk_take", "front", "fit", "pin", "evidence"]
    fit = next(s for s in j["steps"] if s["name"] == "fit")
    assert "env" not in fit or not fit.get("env") and "--pin" not in fit["cmd"]
    assert j["after"] == [] and j["what"] == "shape"
    assert any(o["path"].endswith("larissa_kadu_recap_shape/mamma_pin_betas.json") for o in j["outputs"])


def test_build_job_hook_routes_a_recap_take_to_the_profile_and_leaves_other_takes_alone(monkeypatch):
    spec = recap_spec()
    ch, _ = C.plan_windows(spec, spans={c: (0.0, 360.0) for c in spec.cams}, cams_fn=lambda a, b: ["00", "01"])
    ch[0].estimate = RP.estimate_chunk(ch[0], cam_frames=ch[0].T * 2)
    j = C.build_job(ch[0], spec, hash_inputs=False, code={})
    assert j.get("profile") == "recap" and j["steps"][1]["cmd"].startswith("python -m dancecap.recap_front run")
    other = recap_spec(profile="")
    assert C.chunk_params(other)["profile"] == ""


def test_final_job_stitches_all_chunks_without_analyses_and_without_the_world_check(monkeypatch):
    monkeypatch.setattr(RP.A, "_local_motion", lambda spec, chunk: False)       # hermetic: the real work dir fills up as the farm pulls chunks (they would travel as inputs)
    spec = recap_spec()
    ch, _ = C.plan_windows(spec, spans={c: (0.0, 360.0) for c in spec.cams}, cams_fn=lambda a, b: ["00", "01"])
    f = RP.build_final(ch, spec, code={})
    assert f["after"] == [c.id for c in ch] and f["analyses"] == [] and f["needs_gpu"] is False
    stitch = next(s for s in f["steps"] if s["name"] == "stitch")
    assert "--require-world" not in stitch["cmd"] and "--partial" in stitch["cmd"]
    assert [i["path"] for i in f["inputs"]] == [f"chunks/{c.id}/out/result.tar" for c in ch]


def validation_spec():
    """recap_spec() plus the keys chunks.chunk_take_toml reads (source_dir, alignment, dance, dancers): enough for validate_plan on a synthetic plan."""
    spec = recap_spec()
    spec.cfg.update({"source_dir": "../../dances/x", "alignment": "../../prototype/x.json", "dance": {"level": "professional", "category": "lesson"},
                     "dancers": {"lead": {"name": "A", "pid": 1, "stature_m": 1.7, "description": "black t-shirt"},
                                 "follow": {"name": "B", "pid": 2, "stature_m": 1.6, "description": "red hair"}}})
    return spec


def synthetic_plan(monkeypatch, claimed=()):
    monkeypatch.setattr(RP.A, "_local_motion", lambda spec, chunk: False)       # hermetic: pulled chunks of the real work dir would leave the final job's `after`
    monkeypatch.setattr(RP, "load_teachers", lambda: coverage({c: (0, 361, 0.9, 0.9, 0.85) for c in ("00", "01", "02", "03", "05", "06")}))
    spec = validation_spec()
    return spec, RP.make_plan(spec, hash_inputs=False, code={}, claimed_ids=set(claimed))


def test_validate_plan_passes_on_the_full_plan_and_when_chunks_are_already_claimed(monkeypatch):
    """Regression (audit 2026-10-09): `farm plan --dry-run` passes the queued / running chunk ids as claimed_ids, so plan['_chunks'] can be partial or
    EMPTY; validate_plan used to raise IndexError (cores[0][0]) and its tiling check judged the gaps. It now checks the full window arithmetic."""
    spec, full = synthetic_plan(monkeypatch)
    ids = [c.id for c in full["_chunks"]]
    assert len(ids) == 7
    v = RP.validate_plan(full, spec)
    assert v["ok"], {k: c for k, c in v["checks"].items() if not c["ok"]}
    for claimed in (set(ids), {ids[1], ids[3]}, {ids[0]}, set(ids[:-1])):
        spec, plan = synthetic_plan(monkeypatch, claimed)
        assert [c.id for c in plan["_chunks"]] == [i for i in ids if i not in claimed]
        v = RP.validate_plan(plan, spec)                                   # must not raise
        assert v["ok"], (claimed, {k: c for k, c in v["checks"].items() if not c["ok"]})
        tile = v["checks"]["windows_tile_the_timeline_with_the_overlap"]
        assert tile["windows"] == 7 and tile["chunks_planned"] == 7 - len(claimed) and tile["longest_window_s"] <= 60
        assert v["checks"]["worker_queue_and_claim_rules"]["chunks_planned"] == 7 - len(claimed)


def test_validate_plan_still_catches_a_broken_tiling_and_a_foreign_chunk(monkeypatch):
    spec, plan = synthetic_plan(monkeypatch, claimed={"g0000"})
    plan["arithmetic"]["cores"][2][1] += 5                                 # a hole between two cores
    assert not RP.validate_plan(plan, spec)["checks"]["windows_tile_the_timeline_with_the_overlap"]["ok"]
    spec, plan = synthetic_plan(monkeypatch)
    plan["_chunks"][3].k1 += 1                                             # a planned chunk that is none of the arithmetic's windows
    assert not RP.validate_plan(plan, spec)["checks"]["windows_tile_the_timeline_with_the_overlap"]["ok"]


# ------------------------------------------------------------------------------------------------ recap_front: pure parts


def test_frame_range_slice_and_people_arrays(tmp_path):
    t = np.arange(0, 10, 0.1)
    idx = rf.frame_range(t, 2.0, 3.0)
    assert idx[0] == 20 and idx[-1] == 30
    p = tmp_path / "01.npz"
    np.savez(p, t=t, K=np.zeros((100, 3, 3)), ok=np.ones(100, bool), attached=np.array(False), camera_class=np.array("handheld"), native_size=np.array([480, 864]))
    s = rf.slice_npz(p, idx)
    assert len(s["t"]) == 11 and s["K"].shape == (11, 3, 3) and s["native_size"].shape == (2,) and not bool(s["attached"])
    kp = np.full((5, 3, 17, 3), np.nan, np.float32)
    n = np.array([2, 1, 0, 3, 2], np.int16)
    for r in range(5):
        for j in range(n[r]):
            kp[r, j, :, 0] = np.linspace(100 + 10 * j, 160 + 10 * j, 17)
            kp[r, j, :, 1] = np.linspace(50, 300, 17)
            kp[r, j, :, 2] = 0.9
    pa = rf.people_arrays(kp, n, np.array([0, 2, 3]))
    assert pa["boxes"].shape == (3, 3, 4) and pa["valid"].sum(1).tolist() == [2, 0, 3] and (pa["scores"][pa["valid"]] > 0.8).all()
    assert np.isfinite(pa["kpts"]).all() and (pa["boxes"][0, 0, 3] > pa["boxes"][0, 0, 1])


def test_teacher_slots_drop_the_doubtful_and_keep_lead_as_slot_a():
    tt = {"row": np.array([[3, 5], [4, -1], [2, 1]], np.int16), "conf": np.array([[0.9, 0.2], [0.8, 0.0], [0.6, 0.7]], np.float32),
          "box": np.ones((3, 2, 4), np.float32)}
    s = rf.teacher_slots(tt, np.array([0, 1, 2]))
    assert s["idx"].tolist() == [[3, -1], [4, -1], [2, 1]] and np.isnan(s["boxes"][0, 1]).all() and (s["boxes"][0, 0] == 1).all()


def test_regauge_makes_the_master_camera_zero_and_keeps_the_differences(tmp_path, monkeypatch):
    monkeypatch.setattr(rf, "take_work", lambda cid: tmp_path)
    (tmp_path / "sync").mkdir()
    doc = {"offsets": {"01": {"delta": 0.040}, "02": {"delta": 0.010}, "03": {"delta": -0.020}}, "gauge": "median"}
    (tmp_path / "sync" / "offsets.json").write_text(json.dumps(doc))
    r = rf.regauge_master("g0000")
    d = json.loads((tmp_path / "sync" / "offsets.json").read_text())["offsets"]
    assert r["regauged"] and d["01"]["delta"] == 0.0 and d["02"]["delta"] == pytest.approx(-0.030) and d["03"]["delta"] == pytest.approx(-0.060)
    assert d["02"]["delta_ms"] == pytest.approx(-30.0)


def test_front_dag_has_one_sam_and_ma2d_task_per_camera_and_prepares_the_chunk_window(tmp_path, monkeypatch):
    work = tmp_path / "dance"
    (work / "cameras").mkdir(parents=True)
    for c, (w, h) in {"01": (1080, 1920), "02": (848, 480)}.items():
        np.savez(work / "cameras" / f"{c}.npz", t=np.arange(0, 120, 1 / 30), native_size=np.array([w, h]))
    monkeypatch.setattr(rf, "dance_work", lambda: work)
    monkeypatch.setattr(rf, "take_work", lambda cid: tmp_path / "take")
    chunk = {"id": "g1588", "k0": 1528, "k1": 3175, "cams": ["01", "02"], "t0": 49.0, "t1": 112.0}
    added = []

    class FakeRunner:
        def add(self, t):
            added.append(t)

    rf.build_tasks(FakeRunner(), chunk)
    names = [t.name for t in added]
    assert names == ["frames", "cameras", "people", "sam:01", "ma2d:01", "sam:02", "ma2d:02", "merge", "sync", "prepare"]
    by = {t.name: t for t in added}
    assert by["sam:01"].deps == ["people", "cameras"] and by["ma2d:02"].deps == ["sam:02"] and set(by["sync"].after) == {"ma2d:01", "ma2d:02"}
    assert by["sam:01"].mem_gb > 20 > by["sam:02"].mem_gb                    # the 1080x1920 phone holds ~30 GB, the 848x480 one far less
    prep = by["prepare"].cmd
    assert "--grid-anchor" in prep and "--grid-k=1528:3175" in prep and "c2=01,02" in prep and "--skip-2d" in prep and prep[prep.index("--coverage") + 1] == "union"


def test_await_pin_extracts_the_shape_betas_from_the_volume_and_is_idempotent(tmp_path, monkeypatch):
    import io
    import tarfile

    from dancecap import recap_front as RF
    pkg = tmp_path / "pkg"
    vol = tmp_path / "vol"
    out_dir = vol / "chunks" / "shape" / "out"
    out_dir.mkdir(parents=True)
    with tarfile.open(out_dir / "result.tar", "w") as tar:
        for name, data in (("dancecap/work/larissa_kadu_recap_shape/mamma_pin_betas.json", b'{"0": [0.1], "1": [0.2]}'),
                           ("dancecap/work/larissa_kadu_recap_shape/mamma_pin_betas.source.json", b"{}")):
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            tar.addfile(ti, io.BytesIO(data))
    monkeypatch.setattr(RF, "PACKAGE_ROOT", pkg)
    monkeypatch.setenv("HM_ROOT", str(vol))
    with pytest.raises(TimeoutError):     # no done.json yet: it waits, then gives up
        RF.step_await_pin("g0000", timeout_min=0.01, poll_s=0.05)
    (vol / "chunks" / "shape" / "done.json").write_text("{}")
    p = RF.step_await_pin("g0000", timeout_min=1, poll_s=0.05)
    assert p == pkg / "work" / "larissa_kadu_recap_shape" / "mamma_pin_betas.json" and p.read_text() == '{"0": [0.1], "1": [0.2]}'
    assert p.with_name("mamma_pin_betas.source.json").exists()
    (out_dir / "result.tar").unlink()
    assert RF.step_await_pin("g0000", timeout_min=0.01) == p     # already in place: no volume needed


def test_sync_policy_replaces_only_the_unreliable_window_estimates():
    import copy

    def cam(delta, res, gap=2.0, edge=False, use=True):
        return {"delta": delta / 1000, "delta_ms": delta, "use": use, "at_search_edge": edge, "estimator_gap_ms": gap,
                "residual_px": {"metric": "along_px", "at_delta": res, "at_zero": res + 1}}
    good = {"offsets": {"00": cam(125.0, 2.4), "01": cam(0.0, 4.9), "02": cam(-7.3, 2.4), "05": cam(-25.7, 3.2, gap=26.1)}}
    before = copy.deepcopy(good)
    assert rf.sync_policy(good) == {} and good == before                       # a reliable window keeps its own numbers
    bad = {"offsets": {"00": cam(-185.6, 20.7, gap=299.0), "01": cam(0.0, 30.6), "02": cam(-310.3, 71.2, gap=299.1, edge=True, use=False),
                       "05": cam(-20.0, 15.7, gap=13.9)}}
    done = rf.sync_policy(bad)
    assert set(done) == {"00", "02", "05"}                                     # the master is never touched
    assert bad["offsets"]["01"]["delta_ms"] == 0.0 and "f1_policy" not in bad["offsets"]["01"]
    assert bad["offsets"]["00"]["delta_ms"] == rf.REF_OFFSETS_MS["00"] and bad["offsets"]["00"]["delta"] == rf.REF_OFFSETS_MS["00"] / 1000
    assert bad["offsets"]["02"]["use"] is True and "search edge" in done["02"]["why"] and "unusable" in done["02"]["why"]
    assert bad["f1_policy"]["cams"]["00"]["own_ms"] == -185.6


def test_mask_frames_and_step_mask_zero_one_camera_for_a_stretch(tmp_path, monkeypatch):
    import json

    import numpy as np
    assert rf.parse_masks("06:152:208, 02:242.5:258") == [("06", 152.0, 208.0), ("02", 242.5, 258.0)]
    assert rf.mask_frames(200.0, 30.0, 300, 202.0, 204.0) == (60, 120) and rf.mask_frames(200.0, 30.0, 300, 190.0, 195.0) is None
    assert rf.mask_frames(200.0, 30.0, 300, 205.0, 400.0) == (150, 300)       # clipped to the chunk
    pkg = tmp_path / "pkg"
    work = pkg / "work" / "larissa_kadu_recap_g9999"
    seq = work / "mamma" / "ma_2d" / "larissa_kadu_recap_g9999_c2"
    seq.mkdir(parents=True)
    (work / "mamma" / "prepare.json").write_text(json.dumps({"grid": {"t0": 200.0, "hz": 30.0, "T": 300}, "sets": {"c2": {"cams": ["00", "06"]}}}))
    for c in ("00", "06"):
        np.savez_compressed(seq / f"cam{c}.npz", landmarks=np.ones((300, 2, 4, 3), np.float32), visibilities=np.ones((300, 2, 4), np.float32),
                            contacts=np.ones((300, 2, 4), np.float32), floor_contacts=np.ones((300, 2, 4), np.float32))
    monkeypatch.setattr(rf, "PACKAGE_ROOT", pkg)
    res = rf.step_mask("g9999", "06:202:204,07:202:204")
    assert res["masked"][0]["frames"] == [60, 120] and res["masked"][1]["frames"] is None
    z6, z0 = np.load(seq / "cam06.npz"), np.load(seq / "cam00.npz")
    assert float(z6["visibilities"][60:120].sum()) == 0.0 and float(z6["landmarks"][59].sum()) > 0 and float(z6["visibilities"][120].sum()) > 0
    assert float(z6["contacts"][60:120].sum()) == 0.0 and float(z0["visibilities"].min()) == 1.0     # the other camera is untouched
    assert json.loads((work / "mamma" / "mask.json").read_text())["masked"][0]["cam"] == "06"
    z6.close()
    z0.close()                                                                  # (Windows cannot replace an open file)
    rf.step_mask("g9999", "06:202:204")                                         # idempotent
    with np.load(seq / "cam06.npz") as z:
        assert float(z["visibilities"][60:120].sum()) == 0.0


def test_refit_spec_drops_the_front_and_adds_mask_and_the_checkpoint_input():
    from dancecap.farm import recap_refit as RR
    names = ["chunk_take", "front", "await_pin", "fit", "refine", "quality", "evidence"]
    base = {"id": "g4644", "steps": [{"name": n, "cmd": f"python -m x {n}", "timeout_min": 5, "optional": n == "quality"} for n in names],
            "inputs": [{"path": "src/Larissa-Kadu-recap/00-Recap-Ali.mp4", "sha256": "a"}, {"path": "src/Larissa-Kadu-recap/recap_in.tar", "archive": True}],
            "estimate": {"fit_min": 26.5, "refine_min": 1.0, "quality_min": 3.0}, "code": {"key": "abc"}}
    new = RR.refit_spec(base, [("06", 152.0, 208.0)], "src/Larissa-Kadu-recap/refit/g4644.r1.front.tar", "r1")
    assert [s["name"] for s in new["steps"]] == ["chunk_take", "mask", "await_pin", "fit", "refine", "quality", "evidence"]
    assert new["steps"][1]["cmd"] == "python -m dancecap.recap_front mask g4644 --mask 06:152:208"
    assert [i["path"] for i in new["inputs"]] == ["src/Larissa-Kadu-recap/recap_in.tar", "src/Larissa-Kadu-recap/refit/g4644.r1.front.tar"]
    assert new["inputs"][-1]["archive"] is True and new["refit"]["masks"] == [{"cam": "06", "t0": 152.0, "t1": 208.0}]
    assert [s["name"] for s in base["steps"]] == names and len(base["inputs"]) == 2        # the base spec is not modified
