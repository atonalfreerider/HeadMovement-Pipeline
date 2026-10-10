"""Unit tests of the pose-polish pieces (agent T1): recap_front.qa_mask_camera / apply_fixed_offsets and farm/recap_fix's pure job builders.

No pod, no footage: synthetic landmarks, a synthetic template job spec.

    .venv-nopo4d\\Scripts\\python.exe -m pytest dancecap/tests/test_recap_fix.py -q      (from the workspace root)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dancecap import recap_front as rf  # noqa: E402
from dancecap.farm import recap_fix as fx  # noqa: E402


# ------------------------------------------------------------------------------------------------ qa mask


def person(cx, cy, h=200.0, n=512):
    """512 landmarks spread over a person of height h centred on (cx, cy) and the 17 COCO joints of the same person."""
    rng = np.random.default_rng(1)
    L = np.zeros((n, 3), np.float32)
    L[:, 0] = cx + rng.uniform(-0.15, 0.15, n) * h
    L[:, 1] = cy + rng.uniform(-0.5, 0.5, n) * h
    V = np.ones(n, np.float32)
    kp = np.zeros((17, 3), np.float32)
    kp[:, 0] = cx + rng.uniform(-0.12, 0.12, 17) * h
    kp[:, 1] = np.linspace(cy - 0.5 * h, cy + 0.5 * h, 17)
    kp[:, 2] = 0.9
    return L, V, kp


def test_extent_error_is_small_on_the_same_person_and_large_on_another():
    L, V, kp = person(300, 400)
    assert rf.extent_error(L, V, kp) < 0.05
    L2, V2, _ = person(500, 400)                       # the landmarks of somebody 200 px (one person height) away
    assert rf.extent_error(L2, V2, kp) > 0.8
    V0 = np.zeros_like(V)
    assert np.isnan(rf.extent_error(L, V0, kp))        # nothing visible
    kp0 = kp.copy()
    kp0[:, 2] = 0.0
    assert np.isnan(rf.extent_error(L, V, kp0))        # no teacher joints


def test_qa_ranges_find_a_wrong_stretch_widen_it_and_ignore_blips_and_gaps():
    d = np.full(300, 0.03)
    d[100:180] = 0.6                                    # 80 frames on somebody else
    d[40] = 0.9                                         # a one-frame blip: the rolling median ignores it
    d[220:240] = np.nan                                 # unknown frames are never masked
    rg = rf.qa_ranges(d)
    assert len(rg) == 1
    a, b = rg[0]
    assert 80 <= a <= 100 and 180 <= b <= 200           # the stretch widened by about the pad
    assert rf.qa_ranges(np.full(100, 0.05)) == []


def synthetic_cam(T=240, wrong=(60, 140)):
    """ma_2d arrays of one camera: slot 0 (lead) on the teacher all the time, slot 1 (follow) on somebody else in `wrong`."""
    Ls = np.zeros((T, 2, 512, 3), np.float32)
    Vs = np.ones((T, 2, 512), np.float32)
    kp = np.zeros((T, 2, 17, 3), np.float32)
    for t in range(T):
        L0, _, k0 = person(300, 400)
        L1, _, k1 = person(150, 420)
        Ls[t, 0], kp[t, 0] = L0, k0
        Ls[t, 1], kp[t, 1] = (person(450, 420)[0] if wrong[0] <= t < wrong[1] else L1), k1
    z = {"landmarks": Ls, "visibilities": Vs, "contacts": np.ones((T, 2, 512), np.float32), "floor_contacts": np.ones((T, 2, 512), np.float32)}
    return z, kp


def test_qa_mask_camera_zeroes_only_the_wrong_dancer_in_the_wrong_stretch():
    z, kp = synthetic_cam()
    T = z["landmarks"].shape[0]
    j = np.arange(T)
    r = rf.qa_mask_camera(z, kp, j, np.ones(T, bool))
    assert r["lead"] == [] and r["suspect"] == []
    assert len(r["follow"]) == 1
    a, b = r["follow"][0]
    assert a <= 60 and b >= 140
    assert not z["landmarks"][a:b, 1].any() and not z["visibilities"][a:b, 1].any() and not z["contacts"][a:b, 1].any()
    xy = z["landmarks"][..., :2]
    assert (xy[:, 0] != 0).all()                        # the lead's landmarks are untouched everywhere
    assert (xy[0:a - 1, 1] != 0).all() and (xy[b + 1:, 1] != 0).all()


def test_qa_mask_camera_distrusts_itself_when_both_dancers_are_wrong_most_of_the_time():
    z, kp = synthetic_cam(wrong=(0, 240))
    T = z["landmarks"].shape[0]
    for t in range(T):                                  # the lead too is on somebody else in every frame: a clock / coordinate bug, not two SAM failures
        z["landmarks"][t, 0] = person(500, 400)[0]
    r = rf.qa_mask_camera(z, kp, np.arange(T), np.ones(T, bool))
    assert r["suspect"] == ["lead", "follow"] and r["lead"] == [] and r["follow"] == []
    assert (z["landmarks"][..., :2][:, 1] != 0).all() and (z["landmarks"][..., :2][:, 0] != 0).all()   # nothing was masked


def test_qa_mask_camera_masks_a_dancer_who_is_wrong_for_the_whole_chunk():
    z, kp = synthetic_cam(wrong=(0, 240))               # phone 06's follow in 150-208 s: the SAM mask is on somebody else from the first to the last frame
    T = z["landmarks"].shape[0]
    r = rf.qa_mask_camera(z, kp, np.arange(T), np.ones(T, bool))
    assert r["suspect"] == [] and r["lead"] == [] and r["follow"] == [(0, T)]
    assert not z["landmarks"][:, 1].any() and (z["landmarks"][..., :2][:, 0] != 0).all()


def test_qa_mask_camera_skips_frames_without_a_teacher():
    z, kp = synthetic_cam(wrong=(60, 140))
    T = z["landmarks"].shape[0]
    good = np.ones(T, bool)
    good[60:140] = False                                # the teacher is not tracked where the landmarks are wrong: unknown, not wrong
    r = rf.qa_mask_camera(z, kp, np.arange(T), good)
    assert r["follow"] == []


# ------------------------------------------------------------------------------------------------ fixed offsets


def sync_doc():
    return {"offsets": {c: {"delta": 0.0, "delta_ms": 0.0, "use": True} for c in ("00", "01", "02", "05")}}


def test_apply_fixed_offsets_sets_the_cameras_and_never_the_master():
    doc = sync_doc()
    done = rf.apply_fixed_offsets(doc, {"offsets_ms": {"00": 113.1, "01": 50.0, "05": -25.7, "99": 1.0}})
    assert set(done) == {"00", "05"}                    # 01 is the master, 99 is no camera of the window
    assert doc["offsets"]["00"]["delta"] == 0.1131 and doc["offsets"]["00"]["delta_ms"] == 113.1
    assert doc["offsets"]["01"]["delta_ms"] == 0.0
    assert doc["offsets"]["02"]["delta_ms"] == 0.0
    assert doc["offsets"]["00"]["use"] is True and doc["offsets"]["00"]["use_rate"] is False
    assert rf.apply_fixed_offsets(sync_doc(), None) == {}


# ------------------------------------------------------------------------------------------------ the fix jobs


def template_job():
    take = "larissa_kadu_recap_g1588"
    return {"id": "g1588", "take": take, "what": "chunk",
            "steps": [{"name": "chunk_take", "cmd": "python -m dancecap.farm chunk-take g1588", "timeout_min": 3},
                      {"name": "front", "cmd": "python -m dancecap.recap_front run g1588", "timeout_min": 89, "save": [f"dancecap/work/{take}/mamma/prepare.json"]},
                      {"name": "await_pin", "cmd": "python -m dancecap.recap_front await-pin g1588", "timeout_min": 130},
                      {"name": "fit", "cmd": "python -m dancecap.recap_front fit g1588 --pin", "timeout_min": 58},
                      {"name": "refine", "cmd": f"python -m dancecap.pose_refine {take} --src smplx --dst smplx_rf2_noslip", "timeout_min": 10},
                      {"name": "evidence", "cmd": "python -m dancecap.farm chunk-evidence g1588 --stage refined", "timeout_min": 10}],
            "outputs": [{"path": f"dancecap/work/{take}/smplx/"}, {"path": f"dancecap/work/{take}/quality/", "optional": True}],
            "inputs": [{"path": "src/x/00.mp4"}], "env": {}, "estimate": {}}


EST = {"front_min": 12.0, "fit_min": 10.0, "refine_min": 1.0, "quality_min": 3.0, "total_min_4090": 33.0, "usd_4090": 0.49}


def test_mean_offsets_averages_neighbours_and_keeps_the_master_at_zero():
    m = fx.mean_offsets([{"00": 100.0, "01": 0.0, "02": 20.0}, {"00": 110.0, "01": 0.0, "03": 4.0}])
    assert m == {"00": 105.0, "01": 0.0, "02": 20.0, "03": 4.0}
    assert fx.chunk_offsets("g0000_override") == fx.G0000_OVERRIDE


def test_window_cams_takes_every_recording_phone_that_sees_a_teacher_in_a_useful_share():
    st = {"00": {"record_share": 1.0, "lead": 0.34, "follow": 0.34}, "03": {"record_share": 0.0, "lead": 0.0, "follow": 0.0},
          "05": {"record_share": 1.0, "lead": 0.3, "follow": 0.28}, "06": {"record_share": 0.99, "lead": 0.1, "follow": 0.05},
          "02": {"record_share": 0.2, "lead": 0.9, "follow": 0.9}}
    assert fx.window_cams(st) == ["00", "05"]            # 03 does not record, 06 sees nobody, 02 records only 20 % of the window


def test_fix_job_renames_everything_and_adds_the_mask_steps_after_the_front():
    tpl = template_job()
    job = fx.build_fix_job(tpl, "h0000", 0, 1647, ["00", "01", "02", "05"], {**EST, "front_min": 20.0}, {"offsets_ms": {"00": 113.1}}, [("01", 9.0, 30.0)], "why",
                           code={"key": "k"})
    names = [s["name"] for s in job["steps"]]
    assert names == ["chunk_take", "front", "mask", "qa_mask", "await_pin", "fit", "refine", "evidence"]
    txt = str(job["steps"]) + str(job["outputs"])
    assert "g1588" not in txt and "larissa_kadu_recap_h0000" in txt
    mask = next(s for s in job["steps"] if s["name"] == "mask")
    assert mask["cmd"].endswith("mask h0000 --mask 01:9:30")
    assert next(s for s in job["steps"] if s["name"] == "qa_mask")["cmd"].endswith("qa-mask h0000")
    fit = next(s for s in job["steps"] if s["name"] == "fit")
    assert fit["timeout_min"] >= 1.8 * 1648 / 60                # a 55 s chunk needs about 50 minutes of fit
    c = job["chunk"]
    assert (c["k0"], c["k1"], c["core_k0"], c["core_k1"], c["T"], c["take"]) == (0, 1647, 0, 1647, 1648, "larissa_kadu_recap_h0000")
    assert c["sync_fixed"]["offsets_ms"]["00"] == 113.1 and job["cams"] == ["00", "01", "02", "05"]
    assert any(o["path"].endswith("mamma/qa_mask.json") for o in job["outputs"])
    assert tpl["id"] == "g1588" and "mask" not in [s["name"] for s in tpl["steps"]]      # the template is not modified


def test_a_boundary_job_without_manual_masks_has_only_the_qa_step_and_a_short_fit_timeout():
    job = fx.build_fix_job(template_job(), "y1384", 1384, 1744, ["00", "01", "02", "05", "06"], EST, {"offsets_ms": {}}, [], "why")
    names = [s["name"] for s in job["steps"]]
    assert names[:4] == ["chunk_take", "front", "qa_mask", "await_pin"]
    assert next(s for s in job["steps"] if s["name"] == "fit")["timeout_min"] < 40
    assert job["max_minutes"] > EST["total_min_4090"]


def final_template():
    return {"id": "final", "chunk": {"id": "final", "chunks": ["g0000"]}, "after": ["g0000"], "chunks": ["g0000"], "inputs": [], "local_chunks": [], "stitch_chunks": [],
            "chunk_worlds": {}, "outputs": [],
            "steps": [{"name": "stitch_prep", "cmd": "python -m dancecap.farm stitch-prep final"},
                      {"name": "stitch", "cmd": "python -m dancecap.stitch larissa_kadu_recap --chunks work/larissa_kadu_recap/farm/stitch_list.json --partial --force; rc=$?"},
                      {"name": "evidence", "cmd": "python -m dancecap.farm chunk-evidence final --stage final"}]}


def test_final2_lists_the_fixed_set_in_order_writes_elsewhere_and_gives_the_short_windows_a_bigger_margin():
    chunks = [{"id": "g1588", "k0": 1528, "k1": 3175}, {"id": "y1384", "k0": 1384, "k1": 1744}, {"id": "h0000", "k0": 0, "k1": 1647}]
    job = fx.final2_job(final_template(), chunks)
    assert job["id"] == "final2" and job["after"] == ["g1588", "y1384", "h0000"]
    assert [c["id"] for c in job["stitch_chunks"]] == ["h0000", "y1384", "g1588"]            # sorted by the first frame
    assert {c["id"]: c["margin_s"] for c in job["stitch_chunks"]} == {"h0000": 0.2, "y1384": 0.5, "g1588": 0.2}
    assert [i["path"] for i in job["inputs"]] == [f"chunks/{c}/out/result.tar" for c in ("g1588", "y1384", "h0000")]
    cmd = next(s["cmd"] for s in job["steps"] if s["name"] == "stitch")
    assert "--out larissa_kadu_recap_t1 --partial --force --select-weights arm=0.3,fingers=0.3" in cmd
    assert next(s["cmd"] for s in job["steps"] if s["name"] == "stitch_prep").endswith("stitch-prep final2")
    assert "chunk-evidence final2 --stage" in next(s["cmd"] for s in job["steps"] if s["name"] == "evidence")
    assert all(o["path"].startswith("dancecap/work/larissa_kadu_recap_t1/") or o["path"].endswith("stitch_list.json") for o in job["outputs"])
    assert fx.REPLACED == {"g0000": "h0000"}


def test_analysis2_jobs_follow_final2_stage_the_motion_and_time_the_refined_fit():
    base = {"id": "an-timing-w", "chunk": {"id": "an-timing-w"}, "after": ["final"], "inputs": [{"path": "chunks/final/out/result.tar", "archive": True}],
            "steps": [{"name": "timing_free", "cmd": "python -m dancecap.timing_free larissa_kadu_recap --mc 60"}], "kind": "analyse"}
    mem = {**base, "id": "an-physics-m", "kind": "analyse_mem", "steps": [{"name": "physics", "cmd": "python -m dancecap physics larissa_kadu_recap"}]}
    jobs = fx.analysis2_jobs([base, mem])
    assert [j["id"] for j in jobs] == ["an2-timing", "an2-physics"] and all(j["after"] == ["final2"] for j in jobs)
    assert jobs[0]["inputs"] == [{"path": "chunks/final2/out/result.tar", "archive": True}]
    assert [s["name"] for s in jobs[0]["steps"]] == ["stage_motion", "timing_free"]
    assert jobs[0]["steps"][0]["cmd"] == "mkdir -p work/larissa_kadu_recap && cp -r work/larissa_kadu_recap_t1/. work/larissa_kadu_recap/"
    assert jobs[0]["steps"][1]["cmd"].endswith("--smplx-dir smplx_rf2_noslip")
    assert "--smplx-dir" not in jobs[1]["steps"][1]["cmd"] and jobs[1]["kind"] == "analyse_mem"      # physics keeps the raw fit; the memory kind survives
    assert base["after"] == ["final"] and len(base["steps"]) == 1                                    # the input is not modified


def refit_base():
    take = "larissa_kadu_recap_g6171"
    names = ("chunk_take", "front", "await_pin", "fit", "refine", "quality", "evidence")
    return {"id": "g6171", "take": take, "what": "chunk", "estimate": {"fit_min": 26.5, "refine_min": 1.1, "quality_min": 3.0},
            "steps": [{"name": n, "cmd": f"python -m x {n} g6171", "timeout_min": 10, "optional": False, "stage": "front"} for n in names],
            "inputs": [{"path": "src/Larissa-Kadu-recap/00.mp4"}, {"path": "src/Larissa-Kadu-recap/recap_in.tar", "archive": True}], "outputs": []}


def test_qa_refit_spec_keeps_the_id_drops_the_front_and_adds_the_qa_step():
    spec = fx.qa_refit_spec(refit_base(), [("02", 242.0, 258.7)], "src/Larissa-Kadu-recap/refit/g6171.r1.front.tar", "r2", 0.12)
    names = [s["name"] for s in spec["steps"]]
    assert names == ["chunk_take", "mask", "qa_mask", "await_pin", "fit", "refine", "quality", "evidence"] and spec["id"] == "g6171"
    assert next(s["cmd"] for s in spec["steps"] if s["name"] == "mask").endswith("mask g6171 --mask 02:242:258.7")
    assert next(s["cmd"] for s in spec["steps"] if s["name"] == "qa_mask").endswith("qa-mask g6171 --thr 0.12")
    assert [i["path"] for i in spec["inputs"]] == ["src/Larissa-Kadu-recap/recap_in.tar", "src/Larissa-Kadu-recap/refit/g6171.r1.front.tar"]    # no videos
    assert spec["refit"]["tag"] == "r2" and spec["refit"]["qa_thr"] == 0.12 and spec["refit"]["masks"] == [{"cam": "02", "t0": 242.0, "t1": 258.7}]
    none = fx.qa_refit_spec(refit_base(), [], "x.tar", "r2")
    assert [s["name"] for s in none["steps"]] == ["chunk_take", "qa_mask", "await_pin", "fit", "refine", "quality", "evidence"]      # no empty `mask --mask`
    assert none["refit"]["masks"] == []


def test_the_stitcher_takes_a_margin_pair_per_chunk():
    from dancecap import stitch as stc
    assert stc._margin(0.5) == 0.5 and stc._margin([0.2, 1.5]) == [0.2, 1.5] and stc._margin((1, 2)) == [1.0, 2.0]
    from dancecap.farm import analysis as A
    assert A._margin_s([0.2, 1.5]) == [0.2, 1.5] and A._margin_s(0.2) == 0.2


def test_stitch_spec_reads_select_weights_and_margin_pairs_from_the_chunk_list(tmp_path):
    import json as _json
    from dancecap import stitch as stc
    doc = {"take": "t", "select_weights": {"arm": 0.3, "fingers": 0.3}, "edge_margin_s": 0.2,
           "chunks": [{"name": "a", "src": "a/smplx", "k": [0, 99], "margin_s": [0.2, 1.5]}, {"name": "b", "src": "b/smplx", "k": [60, 199], "margin_s": 0.5}]}
    p = tmp_path / "list.json"
    p.write_text(_json.dumps(doc))
    spec = stc.spec_from_json(p)
    assert spec.select_weights == {"arm": 0.3, "fingers": 0.3}
    assert [c.margin_s for c in spec.chunks] == [[0.2, 1.5], 0.5]
    assert stc.chunk_list(spec)["select_weights"] == {"arm": 0.3, "fingers": 0.3}


def test_provisional_md_has_a_before_now_table_and_a_custom_improvement_text():
    from dancecap import stitch_provisional as sp
    chk = {"body_p95_mm": 70.0, "elbow_wrist_p95_mm": 200.0, "fingers_rel_wrist_p95_mm": 150.0, "pelvis_median_mm": 10.0}
    seam = {"name": "a|b", "window_t": [1.0, 3.0], "gate": {"passed": False, "why": ["x"]}, "roles": {"lead": {"checks": chk}, "follow": {"checks": {**chk, "body_p95_mm": 90.0}}}}
    now = {"name": "a|c", "window_t": [2.0, 4.0], "gate": {"passed": True, "why": []}, "roles": {"lead": {"checks": {**chk, "body_p95_mm": 10.0}}, "follow": {"checks": {**chk, "body_p95_mm": 12.0}}}}
    st = {"grid": {"T": 10, "hz": 30.0, "t": [0, 1]}, "seams": [now], "thresholds": {}}
    rows = sp.seam_rows(st)
    md = sp.render("t", st, rows, {}, "why", [], "now", prev={"seams": [seam]}, improvement="do the thing")
    assert "## Before this stitch (status per seam)" in md and "| a|b | 1.00 to 3.00 | FAIL | 90 mm | 200 mm | 150 mm | 10 mm |" in md
    assert "| a|c | pass | 12 mm |" in md and "do the thing" in md and "12-s boundary refits" not in md
