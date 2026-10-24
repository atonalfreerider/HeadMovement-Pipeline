r"""dancecap.mono4d.recap_splatav take profiles (2026-10-23: the recap route generalised to carlos_aline).

Regression: the class recap's profile (no --take) reproduces the pre-profile module exactly: its constants, the SAM
candidate selection on the saved survey (work data; skipped when absent) and both pod job jsons byte for byte against the
tracked remote/cand_lhmpp/pod_job_recap_*.json. New: the `_world.npz` loader that merges pts / frame_index, the take's
constant sync offsets, the watermark rejection (survey box and SAM mask), the white-top colour helpers, generic A2
camera keys, the export hair rule, and that a new take never writes its job jsons over the recap's tracked files.

    .venv-nopo4d\Scripts\python.exe -m pytest dancecap/tests/test_recap_splatav_profile.py -q -p no:cacheprovider
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dancecap.mono4d import recap_splatav as RS  # noqa: E402

PKG = Path(__file__).resolve().parents[1]
RECAP_LHMPP = PKG / "work" / "larissa_kadu_recap" / "lhmpp"
CA_WORK = PKG / "work" / "carlos_aline"


@pytest.fixture(autouse=True)
def _recap_after():
    RS.use(None)
    yield
    RS.use(None)


# ------------------------------------------------------------------------------------------------ recap regression
def test_recap_profile_is_the_pre_profile_module():
    """The values of the module constants before the profile refactor (git HEAD b9844ad)."""
    p = RS.use(None)
    assert p.take == "larissa_kadu_recap" and p.capture == "06_LarissaKaduRecap"
    assert p.cams == ("00", "01", "02", "03", "05", "06") and p.hires == ("01", "06")
    assert p.names == {"lead": "Kadu", "follow": "Larissa"}
    assert p.portrait_cams == ("05", "06")
    assert p.offsets_file is None and p.students and p.window is None and p.ignore_rects == {} and p.gate is None
    assert p.sel is None and p.front_extra == 0 and p.front_gate is None and not p.hires_in_order
    assert p.per_cam == 10
    assert list(p.caps.items()) == [("01", 4), ("06", 4), ("00", 2), ("02", 2), ("03", 2), ("05", 2)] and p.cap_default == 2
    assert p.min_sharp == 200.0 and p.preview_window == (345.0, 355.0)
    assert p.a2_cams == ("01", "06") and p.prev_cams == ("01", "06") and p.a2_span == (30.0, 330.0)
    assert p.hair_rule == "red" and p.colour == "ref:01" and p.legacy_jobs
    assert set(p.veto) == {("lead", "03_04680"), ("lead", "05_00368"), ("follow", "03_05552"), ("follow", "03_01992"),
                           ("follow", "05_03520"), ("lead", "06_02824")}
    assert set(p.face_ok) == {("follow", "01_03872")}
    assert p.work == PKG / "work" / "larissa_kadu_recap" and p.out == p.work / "lhmpp"
    assert p.fit_dir == p.work / "smplx_rf2_noslip"
    assert p.unity_dir.parts[-3:] == ("StreamingAssets", "06_LarissaKaduRecap", "splats")
    assert p.sam_job == RS.JOB_DIR / "pod_job_recap_sam.json" and p.lhm_job == RS.JOB_DIR / "pod_job_recap_lhmpp.json"
    assert RS.GATE == {"inimg": 1.0, "occ_partner": 0.01, "occ_student": 0.005, "v_med": 0.45, "min_h_frac": 0.22}
    assert (RS.YAW_BINS, RS.MIN_DT, RS.STRIDE, RS.VSTEP) == (8, 3.0, 8, 13)
    assert RS.SEL == {"n": 12, "per_cam": 3, "min_dt": 2.0, "white_frac": 0.015, "bad_hole_frac": 0.002,
                      "outside_frac": 0.15, "sam_score": 0.80, "front_deg": 40.0}


def test_recap_sam_job_json_byte_identical(tmp_path):
    RS.use(None)
    p = RS.sam_job(tmp_path / "sam.json")
    # text compare: write_text gives the platform's line ends, git may check the tracked file out with LF
    assert p.read_text() == (RS.JOB_DIR / "pod_job_recap_sam.json").read_text()


def test_recap_lhm_job_json_byte_identical():
    RS.use(None)
    assert json.dumps(RS.lhm_job_spec(40.0), indent=1) == (RS.JOB_DIR / "pod_job_recap_lhmpp.json").read_text()


@pytest.mark.skipif(not (RECAP_LHMPP / "survey.json").exists() or not (RECAP_LHMPP / "sam_candidates.json").exists(),
                    reason="recap work data (likeness, git-ignored) not present")
def test_recap_sam_candidates_from_the_saved_survey_unchanged():
    RS.use(None)
    rows = json.loads((RECAP_LHMPP / "survey.json").read_text())["rows"]
    saved = json.loads((RECAP_LHMPP / "sam_candidates.json").read_text())
    got = RS.pick_sam_candidates(rows)
    assert json.loads(json.dumps(got)) == saved["candidates"]
    assert saved["gate"] == RS.GATE and saved["per_cam"] == RS.P.per_cam and saved["yaw_bins"] == RS.YAW_BINS


def test_no_take_and_recap_alias_are_the_same_profile():
    assert RS.profile(None) == RS.profile("larissa_kadu_recap") == RS.profile("recap")
    with pytest.raises(SystemExit):
        RS.profile("no_such_take")


# ------------------------------------------------------------------------------------------------ carlos_aline profile
def test_carlos_aline_profile():
    p = RS.use("carlos_aline")
    assert p.capture == "07_CarlosAline" and p.cams == ("01", "02") and p.hires == ("01", "02") and p.hires_in_order and p.front_extra == 8
    assert p.names == {"lead": "Carlos", "follow": "Aline"}            # from takes/carlos_aline.toml [dancers.*].name
    assert p.window == (14.82, 196.5) and not p.students
    assert p.ignore_rects == {"01": [(1440.0, 800.0, 1920.0, 1080.0)]}
    assert p.caps == {"01": 8, "02": 6} and p.per_cam == 20
    assert p.gate == {**RS.GATE, "occ_partner": 0.05, "v_med": 0.8}
    assert p.a2_cams == ("01", "02") and p.prev_cams == ("01", "02")
    assert p.hair_rule == "none" and p.white_role == "lead" and p.colour == "auto"
    assert p.unity_dir.parts[-3:] == ("StreamingAssets", "07_CarlosAline", "splats")


def test_new_take_never_writes_over_the_recap_job_jsons():
    p = RS.use("carlos_aline")
    assert p.sam_job == PKG / "work" / "carlos_aline" / "lhmpp" / "pod_job_sam.json"
    assert p.lhm_job == PKG / "work" / "carlos_aline" / "lhmpp" / "pod_job_lhmpp.json"
    assert RS.JOB_DIR not in (p.sam_job.parent, p.lhm_job.parent)
    s = RS.sam_job_spec()
    assert s["take"] == "carlos_aline" and s["inputs"] == ["dancecap/work/carlos_aline/lhmpp/sam_in/"]
    assert "--input work/carlos_aline/lhmpp/sam_in --out work/carlos_aline/lhmpp/sam_out" in s["commands"][0]["cmd"]
    assert s["after_download"] == ["python -m dancecap.mono4d.recap_splatav select --take carlos_aline"]
    j = RS.lhm_job_spec(40.0)
    assert j["inputs"] == ["dancecap/work/carlos_aline/lhmpp/input/"] and j["take"] == "carlos_aline"
    assert j["after_download"] == ["python -m dancecap.mono4d.recap_splatav import --take carlos_aline --session <sid>"]
    assert "Carlos" in j["notes"][0] and "Kadu" not in j["notes"][0]


def test_cli_take_flag_writes_into_the_take_folder(tmp_path):
    RS.main(["sam-job", "--take", "carlos_aline", "--out-dir", str(tmp_path)])
    d = json.loads((tmp_path / "pod_job_sam.json").read_text())
    assert d["take"] == "carlos_aline" and d["name"] == "carlos_aline_lhmpp_sam"


# ------------------------------------------------------------------------------------------------ loader, offsets
def test_merge_cam_rows_adds_pts_and_frame_index():
    t = np.array([14.8158, 14.84913, 14.88247])
    world = {"t": t, "R": np.zeros((3, 3, 3)), "ok": np.ones(3, bool)}
    plain = {"t": t.copy(), "pts": t - 14.8158, "frame_index": np.arange(3)}
    d = RS.merge_cam_rows(world, plain, "02")
    assert np.allclose(d["pts"], [0, 0.03333, 0.06667], atol=1e-4) and list(d["frame_index"]) == [0, 1, 2]
    assert d["R"] is world["R"]
    with pytest.raises(ValueError):
        RS.merge_cam_rows(world, {"t": t[:2], "pts": t[:2]}, "02")
    with pytest.raises(ValueError):
        RS.merge_cam_rows(world, {"t": t + 0.01, "pts": t}, "02")


def test_merge_keeps_existing_pts():
    t = np.array([1.0, 2.0])
    d = RS.merge_cam_rows({"t": t, "pts": np.array([5.0, 6.0])}, {"t": t, "pts": np.array([0.0, 0.0])})
    assert list(d["pts"]) == [5.0, 6.0]


def test_read_offsets_sync_layout(tmp_path):
    p = tmp_path / "offsets.json"
    p.write_text(json.dumps({"offsets": {"01": {"delta": 0.0, "use": True}, "02": {"delta": -0.062, "use": True,
                                                                                  "use_rate": False, "rate": 0.5},
                                         "03": {"delta": 0.2, "use": False},
                                         "04": {"delta": 0.1, "use_rate": True, "rate": 0.001, "t_ref": 10.0}}}))
    o = RS.read_offsets(p)
    assert o["01"]["delta"] == 0.0 and o["02"] == {"delta": -0.062, "rate": 0.0, "t_ref": 0.0}
    assert o["03"]["delta"] == 0.0 and o["04"] == {"delta": 0.1, "rate": 0.001, "t_ref": 10.0}
    p.write_text(json.dumps({"01": 0.0, "02": -0.05}))
    assert RS.read_offsets(p)["02"]["delta"] == -0.05


@pytest.mark.skipif(not (PKG / "work" / "carlos_aline_cag0000" / "sync" / "offsets.json").exists(),
                    reason="carlos_aline sync data not present")
def test_carlos_aline_video_delta_is_the_constant_sync_offset():
    RS.use("carlos_aline")
    t = np.array([20.0, 100.0, 190.0])
    assert np.allclose(RS.video_delta("02", t), -0.062) and np.allclose(RS.video_delta("01", t), 0.0)
    assert float(RS.video_delta("02", 50.0)) == pytest.approx(-0.062)


@pytest.mark.skipif(not (CA_WORK / "cameras" / "02_world.npz").exists(), reason="carlos_aline cameras not present")
def test_carlos_aline_load_cam_merges_pts():
    RS.use("carlos_aline")
    c = RS.load_cam("02")
    assert (c["W"], c["H"]) == (1920, 906)
    assert abs(float(c["pts"][0])) < 1e-6 and abs(float(c["t"][0] - c["pts"][0]) - 14.8158) < 1e-3
    c1 = RS.load_cam("01")
    assert (c1["W"], c1["H"]) == (1920, 1080) and np.allclose(c1["pts"], c1["t"], atol=1e-5)


# ------------------------------------------------------------------------------------------------ watermark
def _row(cam, box, H=1080, occ=0.0, v=0.1):
    return {"cam": cam, "box": box, "inimg": 1.0, "occ_partner": occ, "occ_student": 0.0, "v_med": v,
            "suspect": False, "h_px": box[3] - box[1], "H": H}


def test_box_touches_and_pad():
    rect = [(1440.0, 800.0, 1920.0, 1080.0)]
    assert RS.box_touches((1500, 300, 1700, 1000), rect)
    assert not RS.box_touches((900, 200, 1200, 1000), rect)
    assert not RS.box_touches((1300, 100, 1420, 700), rect)
    assert not RS.box_touches((1300, 200, 1420, 790), rect)
    assert RS.box_touches((1300, 200, 1420, 790), rect, pad=0.05)       # grown by 29.5 px: x1 1449.5, y1 819.5
    assert not RS.box_touches((1000, 100, 1300, 700), rect, pad=0.05)


def test_watermark_gate_only_on_its_camera_and_take():
    RS.use("carlos_aline")
    corner = _row("01", [1480.0, 200.0, 1760.0, 1010.0])
    assert not RS._gate(corner)
    assert RS._gate(_row("01", [700.0, 150.0, 980.0, 990.0]))
    assert RS._gate(_row("02", [1480.0, 100.0, 1760.0, 890.0], H=906))   # camera 02 has no watermark
    mid = _row("02", [700.0, 150.0, 980.0, 800.0], H=906, occ=0.04, v=0.7)
    assert RS._gate(mid)                                                  # the heat's wider gate
    RS.use(None)
    assert RS._gate(corner)                                               # the recap has no ignore rectangles
    assert not RS._gate(mid)                                              # ... and its own gate


def test_mask_touches():
    m = np.zeros((1080, 1920), bool)
    m[300:790, 1500:1700] = True
    rect = [(1440.0, 800.0, 1920.0, 1080.0)]
    assert not RS.mask_touches(m, rect)
    m[800, 1440] = True
    assert RS.mask_touches(m, rect)


# ------------------------------------------------------------------------------------------------ colour
def test_white_pixels_takes_the_bright_top_band():
    img = np.zeros((100, 40, 3), np.uint8)
    m = np.zeros((100, 40), bool)
    m[0:100, 10:30] = True
    img[25:45, 10:30] = (230, 220, 240)          # the white top (purple cast)
    img[45:100, 10:30] = (20, 20, 25)            # dark trousers
    img[0:25, 10:30] = (120, 80, 60)             # head / skin
    px = RS.white_pixels(img, m)
    assert len(px) > 0 and (px == np.array([230, 220, 240], np.uint8)).all(1).all()


def test_wb_gains_make_the_top_grey():
    px = np.tile(np.array([[230, 214, 242]], np.uint8), (500, 1))
    ab = RS.wb_gains(px)
    out = RS.apply_colour(px[None], ab)[0].astype(float)
    assert np.ptp(out.mean(0)) < 1.5
    lab = RS.lab_mean(out.astype(np.uint8))
    assert np.hypot(lab[1], lab[2]) < 1.0


# ------------------------------------------------------------------------------------------------ A2 keys, hair, preview
def test_a2_instants_use_the_profile_cameras(tmp_path):
    RS.use("carlos_aline", out_dir=str(tmp_path))
    rows = []
    for k in range(60):
        tb = 20.0 + 3.0 * k
        for cam, i, dt in (("01", 100 + k, 0.0), ("02", 400 + k, 0.05)):
            for role in ("lead", "follow"):
                rows.append({"cam": cam, "i": i, "tb": tb + dt, "role": role, "inimg": 1.0})
    (tmp_path / "survey.json").write_text(json.dumps({"rows": rows}))
    inst = RS.a2_instants()
    assert len(inst) == 5                                   # the preview instant (105 s) falls on a spread one: kept once
    assert all(set(p) == {"01", "02", "tb_01", "tb_02"} for p in inst)
    assert len({p["01"] for p in inst}) == 5
    assert any(abs(p["tb_01"] - 105.0) <= 1.5 for p in inst)    # one inside the preview window (100-110 s)


def test_export_hair_rule():
    ex = {"mean0": np.zeros((7, 3))}
    RS.use("carlos_aline")
    for role in ("lead", "follow"):
        h, info = RS.export_hair(role, ex)
        assert not h.any() and info["hair"] == 0 and "note" in info
    assert "red" in RS.export_hair("follow", ex)[1]["note"]
    RS.use(None)
    h, info = RS.export_hair("lead", ex)
    assert info == {"hair": 0, "note": "Kadu: shaved head, no hair Gaussians (by design)"}


def test_fit_into_keeps_same_aspect_and_letterboxes_others():
    import cv2
    im = (np.arange(1920 * 1080 * 3) % 251).astype(np.uint8).reshape(1920, 1080, 3)
    assert np.array_equal(RS._fit_into(im, 540, 960), cv2.resize(im, (540, 960), interpolation=cv2.INTER_AREA))
    out = RS._fit_into(im, 960, 540)
    assert out.shape == (540, 960, 3) and (out[:, :100] == 30).all() and (out[:, -100:] == 30).all()


def test_flagged_counts_absent():
    d = {"suspect": np.array([False, False, False]), "absent": np.array([False, True, False])}
    assert not RS.flagged({"suspect": d["suspect"]}, 0)
    assert RS.flagged(d, 0) and RS.flagged(d, 1)
    assert not RS.flagged({"suspect": d["suspect"], "absent": np.zeros(3, bool)}, 1)
