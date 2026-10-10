"""Unit tests for dancecap.cameras.newcam / newcam_report / recap_cameras (synthetic geometry; no footage, no GPU, no pod).

    .venv-nopo4d\\Scripts\\python.exe -m pytest dancecap/tests/test_newcam.py -q      (from the workspace root)
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Ro

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dancecap import recap_cameras as rc  # noqa: E402
from dancecap.cameras import newcam as nc  # noqa: E402
from dancecap.cameras import newcam_report as rep  # noqa: E402
from dancecap.cameras import typed as T  # noqa: E402


# ------------------------------------------------------------------------------------------------ synthetic world


def make_cam(seed=0, w=728, h=1296, f=1090.0, k1=0.006):
    rng = np.random.default_rng(seed)
    X = np.c_[rng.uniform(-6, 6, 3000), rng.uniform(0.0, 5, 3000), rng.uniform(4, 16, 3000)]
    R = Ro.from_euler("xyz", [3, -8, 1], degrees=True).as_matrix()
    C = np.array([0.3, 1.5, -1.0])
    return X, R, -R @ C, f, k1, w, h


def project(X, R, t, f, k1, w, h):
    Xc = X @ R.T + t
    x, y = Xc[:, 0] / Xc[:, 2], Xc[:, 1] / Xc[:, 2]
    d = 1 + k1 * (x * x + y * y)
    return np.c_[f * x * d + w / 2 - 0.5, f * y * d + h / 2 - 0.5]


class FakeEngine:
    """match(a, b) = identical keypoint indices (the synthetic frames see the same points under the same index)."""

    def match(self, a, b):
        n = min(len(a), len(b))
        return np.c_[np.arange(n), np.arange(n)].astype(np.int64)


def feat_of(uv, w, h):
    return nc.Feat(uv.astype(np.float32), np.zeros((len(uv), 128), np.float16), np.array([w, h], np.int32))


def synthetic_pose(X, R, t, f, k1, w, h, noise=0.4, seed=1):
    rng = np.random.default_rng(seed)
    uv = project(X, R, t, f, k1, w, h) + rng.normal(0, noise, (len(X), 2))
    return nc.Feat(uv.astype(np.float32), np.zeros((len(uv), 128), np.float16), np.array([w, h], np.int32))


# ------------------------------------------------------------------------------------------------ frame table / chunks


def test_frame_table_decimates_60fps_and_windows():
    pts = np.arange(0, 20, 1 / 59.94)
    tab = nc.frame_table(pts, 1.0, 0.0, 59.94, window=(1.0, 5.0))
    assert tab["every"] == 2
    assert np.all(tab["n"] % 2 == 0) and tab["t"].min() >= 1.0 and tab["t"].max() <= 5.0
    assert abs(len(tab["n"]) - 4 * 29.97) < 3


def test_frame_table_offset_and_slope():
    pts = np.arange(0, 100, 1 / 30)
    tab = nc.frame_table(pts, 1.0001, 60.9, 30.0, window=(0.0, 100.0))
    assert tab["t"][0] == pytest.approx(60.9) and tab["t"][-1] <= 100.0
    assert tab["every"] == 1


def test_frame_table_without_clock_keeps_everything():
    pts = np.arange(0, 10, 1 / 30)
    tab = nc.frame_table(pts, None, None, 30.0)
    assert len(tab["n"]) == len(pts) and np.isnan(tab["t"]).all()


def test_chunk_windows_cover_the_window_with_overlap():
    w = nc.chunk_windows()
    assert len(w) == 6
    assert w[0]["core"][0] == 0.0 and w[-1]["core"][1] == pytest.approx(360.48)
    for a, b in zip(w[:-1], w[1:]):
        assert a["core"][1] == pytest.approx(b["core"][0])
        assert a["window"][1] == pytest.approx(a["core"][1] + nc.OVERLAP_S)
        assert b["window"][0] == pytest.approx(b["core"][0] - nc.OVERLAP_S)
    assert w[0]["window"][0] == 0.0 and w[-1]["window"][1] == pytest.approx(360.48)
    assert all(abs((c["core"][1] - c["core"][0]) - 60.08) < 0.01 for c in w)


def test_work_size():
    assert nc.work_size(1080, 1920, 1296) == (728, 1296, pytest.approx(0.675))
    assert nc.work_size(480, 864, 1296) == (480, 864, 1.0)
    w, h, s = nc.work_size(848, 480, 1296)
    assert (w, h, s) == (848, 480, 1.0)


def test_anchor_frames_every_4s():
    a = nc.anchor_frames(900, 30.0, 4.0)
    assert np.all(np.diff(a) == 120) and a[0] == 60


# ------------------------------------------------------------------------------------------------ people masks / pose json


def test_people_mask_covers_keypoints_and_grows():
    p = np.zeros((1, 17, 3), np.float32)
    p[0, :, 0] = np.linspace(100, 140, 17)
    p[0, :, 1] = np.linspace(50, 250, 17)
    p[0, :, 2] = 0.9
    m = nc.people_mask(p, 0.5, (200, 200), margin=0.10)
    assert m.shape == (200, 200) and m.dtype == np.uint8
    xs, ys = p[0, :, 0] * 0.5, p[0, :, 1] * 0.5
    assert all(m[int(y), int(x)] == 255 for x, y in zip(xs, ys))
    assert (m[100, :] > 0).sum() >= 15                    # a line of keypoints still gets a stroke around it, not a hairline
    assert nc.people_mask(np.zeros((0, 17, 3), np.float32), 1.0, (50, 50)).sum() == 0


def test_load_pose_frames_json_and_gz(tmp_path):
    kp = [{"X": 1.0 + i, "Y": 2.0, "Z": 0.9} for i in range(17)]
    doc = [[kp, kp], [], [kp]]
    for name, op in (("a.json", open), ("b.json.gz", gzip.open)):
        with op(tmp_path / name, "wt") as f:
            json.dump(doc, f)
        fr = nc.load_pose_frames(tmp_path / name)
        assert [len(x) for x in fr] == [2, 0, 1] and fr[0].shape == (2, 17, 3)
        assert fr[0][0, 3, 0] == pytest.approx(4.0)


# ------------------------------------------------------------------------------------------------ pose solves


def test_localize_recovers_pose_and_fixed_focal():
    X, R, t, f, k1, w, h = make_cam()
    q = synthetic_pose(X, R, t, f, k1, w, h)
    ref = nc.QRef("m", feat_of(np.zeros((len(X), 2)), w, h), np.arange(len(X)), X, np.arange(len(X)))
    from dancecap.cameras.localize import LocSettings
    # calibrated camera: P3P with the true focal and k1
    p = nc.localize(FakeEngine(), q, [ref], w, h, f, k1, LocSettings(fix_focal=True, min_inliers=40))
    assert p.ok and p.n_inliers > 2500
    assert np.linalg.norm(-p.R.T @ p.t - (-R.T @ t)) < 0.02
    assert np.degrees(np.linalg.norm(Ro.from_matrix(p.R @ R.T).as_rotvec())) < 0.1
    assert p.reproj_px < 0.8
    # uncalibrated camera: P4Pf finds the focal from a wrong guess
    p2 = nc.localize(FakeEngine(), q, [ref], w, h, 0.8 * f, 0.0, LocSettings(max_focal_change=2.0, min_inliers=40))
    assert p2.ok and abs(p2.f / f - 1) < 0.02


def test_fit_intrinsics_recovers_one_focal_and_k1():
    X, _, _, f, k1, w, h = make_cam(seed=5)
    poses = []
    rng = np.random.default_rng(3)
    for i in range(8):
        R = Ro.from_euler("xyz", [3 + i, -8 + 2 * i, 1], degrees=True).as_matrix()
        C = np.array([0.3 + 0.2 * i, 1.5, -1.0 - 0.1 * i])
        t = -R @ C
        feat = synthetic_pose(X, R, t, f, k1, w, h, noise=0.5, seed=i)
        # start every anchor from a slightly wrong pose and focal (what P4Pf gives)
        Rg = Ro.from_rotvec(rng.normal(0, 0.002, 3)).as_matrix() @ R
        poses.append(nc.Pose(ok=True, R=Rg, t=-Rg @ (C + rng.normal(0, 0.01, 3)), f=f * (1 + rng.normal(0, 0.02)), n_inliers=len(X),
                             ref=nc.QRef("a", feat, np.arange(len(X)), X, np.arange(len(X)))))
    r = nc.fit_intrinsics(poses, w, h, 0.85 * max(w, h))
    assert r["ok"] and abs(r["f"] / f - 1) < 0.004 and abs(r["k1"] - k1) < 0.004
    assert r["anchor_reproj_med_px"] < 1.0


def test_refine_fixed_reduces_the_error():
    X, R, t, f, k1, w, h = make_cam(seed=7)
    q = synthetic_pose(X, R, t, f, k1, w, h, noise=0.6)
    Rg = Ro.from_rotvec([0.003, -0.002, 0.001]).as_matrix() @ R
    C = -R.T @ t
    p = nc.Pose(ok=True, R=Rg, t=-Rg @ (C + [0.02, -0.01, 0.015]), f=f, k1=k1, n_inliers=len(X),
                ref=nc.QRef("a", q, np.arange(len(X)), X, np.arange(len(X))))
    nc.refine_fixed([p], w, h, f, k1)
    assert np.linalg.norm(-p.R.T @ p.t - C) < 0.01
    assert np.degrees(np.linalg.norm(Ro.from_matrix(p.R @ R.T).as_rotvec())) < 0.05
    assert p.reproj_px < 0.9


# ------------------------------------------------------------------------------------------------ interpolation / merge


def test_interpolate_fills_the_skipped_frames_and_refuses_long_gaps():
    n_all = np.arange(0, 41)
    n_loc = np.arange(0, 41, 2)
    ang = np.radians(np.linspace(0, 20, len(n_loc)))
    R = Ro.from_rotvec(np.c_[np.zeros(len(n_loc)), ang, np.zeros(len(n_loc))]).as_matrix()
    C = np.c_[np.linspace(0, 1, len(n_loc)), np.zeros(len(n_loc)), np.zeros(len(n_loc))]
    ok = np.ones(len(n_loc), bool)
    ok[8:12] = False                       # frames 16..22 failed: a gap of 10 video frames between ok neighbours 14 and 24
    Ro_, Co, oko, interp = nc.interpolate(n_all, n_loc, R, C, ok, max_gap=4)
    assert oko[0] and oko[14] and not oko[16] and not oko[20] and oko[24] and oko[26]
    assert interp[1] and interp[13] and not interp[14] and not interp[17]
    assert Co[1, 0] == pytest.approx(0.025)
    assert np.degrees(Ro.from_matrix(Ro_[1]).as_rotvec()[1]) == pytest.approx(0.5, abs=0.01)
    Ro2, Co2, ok2, i2 = nc.interpolate(n_all, n_loc, R, C, ok, max_gap=12)
    assert ok2[20] and i2[20]


def _chunk(k, n, t, core, R, C, ok=None, loc=None):
    F = len(n)
    return {"chunk": k, "n": np.asarray(n), "t": np.asarray(t, float), "pts": np.asarray(t, float), "core": np.array(core),
            "ok": np.ones(F, bool) if ok is None else ok, "R": R, "tvec": -np.einsum("fij,fj->fi", R, C), "C": C,
            "n_inliers": np.full(F, 500), "reproj_px": np.full(F, 1.0), "source": np.array(["track"] * F),
            "interp": np.zeros(F, bool), "localised": np.ones(F, bool) if loc is None else loc}


def test_merge_chunks_owner_by_core_and_overlap_agreement():
    t = np.arange(0, 70, 0.5)
    n = np.arange(len(t)) * 15
    R = np.tile(np.eye(3), (len(t), 1, 1))
    C = np.c_[t * 0.01, np.zeros(len(t)), np.zeros(len(t))]
    a = _chunk(0, n[t <= 34], t[t <= 34], [0, 30], R[t <= 34], C[t <= 34])
    Cb = C[t >= 26].copy()
    Cb[:, 1] += 0.003                               # the second chunk disagrees by 3 mm in the 4 s overlap
    b = _chunk(1, n[t >= 26], t[t >= 26], [30, 60], R[t >= 26], Cb)
    m = rep.merge_chunks([b, a])
    assert np.all(np.diff(m["n"]) > 0) and len(m["n"]) == len(set(m["n"]))
    assert set(m["chunk"][m["t"] < 30]) == {0} and set(m["chunk"][(m["t"] >= 30) & (m["t"] < 60)]) == {1}
    assert len(m["overlap_dC"]) > 0 and np.median(m["overlap_dC"]) == pytest.approx(0.003, abs=1e-6)
    assert np.median(m["overlap_dR_deg"]) == pytest.approx(0.0, abs=1e-6)


def test_export_track_schema_and_world_frame(tmp_path):
    t = np.arange(0, 3, 1 / 30)
    F = len(t)
    R = np.tile(Ro.from_euler("y", 20, degrees=True).as_matrix(), (F, 1, 1))
    C = np.tile([1.0, 1.5, -3.0], (F, 1))
    ch = _chunk(0, np.arange(F), t, [0, 60], R, C)
    m = rep.merge_chunks([ch])
    calib = {"f_work": 1090.0, "k1": 0.006, "scale_to_native": 1.5, "native_size": [1080, 1920], "work_size": [720, 1280]}
    cam = {"id": "01", "file": "x.mp4"}
    Rw = Ro.from_euler("y", 90, degrees=True).as_matrix()
    world = {"s": 2.0, "R": Rw.tolist(), "t": [0.5, 0.0, 0.0]}
    rep.export_track(cam, calib, m, tmp_path, world)
    z = np.load(tmp_path / "01.npz", allow_pickle=True)
    for k in ("t", "K", "R", "tvec", "dist", "ok", "reproj_px", "n_inliers", "attached", "axis_sd", "recentred", "camera_class",
              "type_model", "frame_index", "pts"):
        assert k in z.files
    assert z["K"][0, 0, 0] == pytest.approx(1635.0) and z["K"][0, 0, 2] == pytest.approx(539.5) and z["K"][0, 1, 2] == pytest.approx(959.5)
    assert z["reproj_px"][0] == pytest.approx(1.5) and not bool(z["attached"]) and str(z["type_model"]) == "fixed_focal"
    w = np.load(tmp_path / "01_world.npz", allow_pickle=True)
    assert np.allclose(w["C"][0], 2.0 * Rw @ C[0] + [0.5, 0, 0])
    X_sfm = np.array([0.4, 1.0, 2.0])
    X_w = 2.0 * Rw @ X_sfm + [0.5, 0, 0]
    # the same 3D point is in front of the camera in both frames, and the metric depth is s times the SfM depth
    zs = (z["R"][0] @ X_sfm + z["tvec"][0])[2]
    zw = (w["R"][0] @ X_w + w["tvec"][0])[2]
    assert zw == pytest.approx(2.0 * zs)


def test_runs_and_quality_verdict():
    t = np.arange(0, 30, 1 / 30)
    ok = np.ones(len(t), bool)
    ok[300:360] = False                    # 2 s gap
    assert rep.runs(~ok, t, 0.5) == [[10.0, pytest.approx(11.967, abs=0.01)]]
    F = len(t)
    R = np.tile(np.eye(3), (F, 1, 1))
    C = np.c_[np.sin(t), np.zeros(F), np.zeros(F)] * 0.01
    ch = _chunk(0, np.arange(F), t, [0, 60], R, C, ok=ok)
    m = rep.merge_chunks([ch])
    calib = {"f_work": 1000.0, "k1": 0.0, "scale_to_native": 1.0, "native_size": [720, 1280], "work_size": [720, 1280],
             "f_over_maxdim": 0.78}
    q = rep.camera_quality({"id": "00", "file": "a.mp4"}, calib, m, scale_m=1.0)
    assert q["frames_ok"] == int(ok.sum()) and q["longest_gap_s"] == pytest.approx(1.97, abs=0.05)
    assert rep.verdict(q)["verdict"] == "usable"
    assert rep.verdict(q, in_world=False)["verdict"] == "partial"
    assert rep.verdict({**q, "ok_fraction": 0.1})["verdict"] == "unusable"
    assert rep.verdict({**q, "frames_ok": 0})["verdict"] == "unusable"


# ------------------------------------------------------------------------------------------------ review overlay / epipolar


def test_project_points_matches_the_synthetic_camera():
    X, R, t, f, k1, w, h = make_cam()
    K = np.array([[f, 0, w / 2 - 0.5], [0, f, h / 2 - 0.5], [0, 0, 1.0]])
    uv, z = rep.project_points(X, R, t, K, k1)
    assert np.allclose(uv, project(X, R, t, f, k1, w, h)) and (z > 0).all()


def test_epipolar_check_separates_the_true_pose_from_a_rotated_one():
    rng = np.random.default_rng(2)
    w, h, f = 720, 1280, 1000.0
    K = np.array([[f, 0, w / 2 - 0.5], [0, f, h / 2 - 0.5], [0, 0, 1.0]])
    Ra = np.eye(3)
    Ca = np.array([0.0, 1.5, -6.0])
    Rb = Ro.from_euler("y", -50, degrees=True).as_matrix()
    Cb = np.array([6.0, 1.5, 0.0])

    def cam(R, C, t):
        F = 3
        return {"t": np.arange(F) / 30.0, "K": np.tile(K, (F, 1, 1)), "R": np.tile(R, (F, 1, 1)), "tvec": np.tile(-R @ C, (F, 1)),
                "dist": np.zeros(F), "ok": np.ones(F, bool), "frame_index": np.arange(F)}
    A, B = cam(Ra, Ca, 0), cam(Rb, Cb, 0)
    # three people standing around the origin, 17 joints each (a vertical stick figure)
    people = []
    for _ in range(3):
        base = np.array([rng.uniform(-1, 1), 0.0, rng.uniform(-1, 1)])
        joints = np.array([base + [0.15 * np.sin(j), 0.1 * j + 0.1, 0.1 * np.cos(j)] for j in range(17)])
        people.append(joints)

    def view(R, C):
        out = []
        for P in people:
            uv = project(P, R, -R @ C, f, 0.0, w, h)
            out.append(np.c_[uv, np.full(17, 0.9)])
        return np.array(out, np.float32)
    pa = [view(Ra, Ca)] * 3
    pb = [view(Rb, Cb)] * 3
    r = rep.epipolar_check(A, B, pa, pb, np.array([0.0, 1 / 30]), perturb_deg=(1.0, 3.0))
    assert r["real"]["n_persons"] >= 4 and r["real"]["median_px"] < 0.5 and r["real"]["under_3px"] == 1.0
    assert r["pitch_3.0deg"]["median_px"] > 10 * max(r["real"]["median_px"], 0.05)
    assert r["real"]["median_px"] < r["pitch_1.0deg"]["median_px"] < r["pitch_3.0deg"]["median_px"]
    assert r["roll_3.0deg"]["median_px"] > 5 * max(r["real"]["median_px"], 0.05)


# ------------------------------------------------------------------------------------------------ orchestration


CAMS = {"00": {"id": "00", "file": "00.mp4", "width": 480, "height": 864, "rotation": 0, "avg_fps": 30.0, "duration": 369.58,
               "slope": 1.0000145, "offset": -8.9023, "accepted": True},
        "03": {"id": "03", "file": "03.mp4", "width": 480, "height": 848, "rotation": 0, "avg_fps": 30.0, "duration": 300.34,
               "slope": 0.99998, "offset": 60.935, "accepted": True},
        "05": {"id": "05", "file": "05.mp4", "width": 480, "height": 640, "rotation": -90, "avg_fps": 30.0, "duration": 353.1,
               "slope": 1.0, "offset": 7.4326, "accepted": True},
        "04": {"id": "04", "file": "04.mp4", "width": 400, "height": 880, "rotation": 0, "avg_fps": 30.0, "duration": 77.28,
               "slope": 1.0, "offset": None, "accepted": False}}


def test_camera_chunks_follow_the_camera_coverage():
    w = nc.chunk_windows()
    assert rc.camera_chunks(CAMS["00"], w) == [0, 1, 2, 3, 4, 5]
    assert rc.camera_chunks(CAMS["03"], w) == [1, 2, 3, 4, 5]         # starts at master 60.9 s: chunk 0 has nothing
    assert rc.camera_chunks(CAMS["04"], w) == [0]                     # no clock: one chunk, its own video time


def test_frames_per_camera_and_estimate_arithmetic(tmp_path, monkeypatch):
    assert rc.n_frames(CAMS["00"], 2) == int(360.48 * 30 / 2)
    assert rc.n_frames(CAMS["03"], 2) == int((360.48 - 60.935) * 30 / 2)
    assert rc.n_frames(CAMS["04"], 2) == int(77.28 * 30 / 2)
    p = tmp_path / "cams.json"
    p.write_text(json.dumps({"cams": CAMS}))
    monkeypatch.setattr(rc, "CAMS_JSON", p)
    e = rc.estimate(["00", "03"], 2, s_per_frame=0.2, price_hr=1.0, calib_min=2.0, overhead_s=30.0, procs=2.0)
    assert e["jobs"] == 2 and e["chunks"] == 6 + 5
    frames = rc.n_frames(CAMS["00"], 2) + rc.n_frames(CAMS["03"], 2)
    assert e["gpu_hours"] == pytest.approx(round((frames * 0.2 / 2.0 + 2 * 120 + 2 * 30) / 3600, 2))


def test_job_specs_have_the_worker_contract(tmp_path, monkeypatch):
    p = tmp_path / "cams.json"
    p.write_text(json.dumps({"cams": CAMS}))
    monkeypatch.setattr(rc, "CAMS_JSON", p)
    monkeypatch.setattr(rc, "world_info", lambda: {"world_id": "w" * 12, "world_sha256": "a" * 64, "pose_id": "p" * 12})
    specs = rc.job_specs(["03", "04"], 2, code={"bundle_sha256": "b" * 64}, mode="chunk")
    ids = [s["id"] for s in specs]
    assert ids == ["rc-calib-03", "rc-trk-03-1", "rc-trk-03-2", "rc-trk-03-3", "rc-trk-03-4", "rc-trk-03-5", "rc-calib-04", "rc-trk-04-0"]
    cal, trk = specs[0], specs[1]
    assert cal["schema"] == "dancecap.farm.job/1" and cal["kind"] == "front" and cal["needs_gpu"] is True
    assert cal["world_id"] == "w" * 12 and cal["code"]["bundle_sha256"] == "b" * 64
    assert trk["after"] == ["rc-calib-03"] and any(i.get("archive") for i in trk["inputs"])
    assert trk["outputs"] == [{"path": "dancecap/work/larissa_kadu_recap/cameras/chunks/03_1.npz"}]
    assert "track 03 1 --model main" in trk["steps"][0]["cmd"] and "--stride 2" in trk["steps"][0]["cmd"]
    assert specs[-1]["steps"][0]["cmd"].startswith("python -m dancecap.cameras.newcam track 04 0")
    only = rc.job_specs(["03"], 2, only_chunks=[2], mode="chunk")
    assert [s["id"] for s in only] == ["rc-calib-03", "rc-trk-03-2"]
    assert [s["id"] for s in rc.job_specs(["00"], 2, calib_only=True, mode="chunk")] == ["rc-calib-00"]
    s5 = rc.job_specs(["05"], 2, mode="chunk")
    assert "--model sub_05" in s5[0]["steps"][0]["cmd"]
    cam = rc.job_specs(["03", "05"], 2, code={"bundle_sha256": "b" * 64}, procs=4)
    assert [x["id"] for x in cam] == ["rc-cam-03", "rc-cam-05"]
    c3 = cam[0]
    assert "newcam run 03 --model main" in c3["steps"][0]["cmd"] and "--procs 4" in c3["steps"][0]["cmd"] and "--chunks 1,2,3,4,5" in c3["steps"][0]["cmd"]
    assert [o["path"] for o in c3["outputs"]][0].endswith("calib/03.json") and len(c3["outputs"]) == 6 and not c3.get("after")
    assert c3["kind"] == "front" and c3["needs_gpu"] is True and "--model sub_05" in cam[1]["steps"][0]["cmd"]


def test_state_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(rc, "STATE", tmp_path / "s.json")
    assert rc.load_state()["pulled"] == []
    rc.save_state({"pods": {"x": 1}, "jobs": {}, "pulled": ["a"]})
    assert rc.load_state()["pods"] == {"x": 1}


def test_poll_parses_one_ssh_call():
    class L:
        def sh(self, cmd, timeout=0):
            assert cmd.count("ssh") == 0 and "rc-calib-01" in cmd
            return "rc-calib-01 done\nrc-trk-01-0 running\nrc-trk-01-1 queued\n"
    assert rc.poll(L(), ["rc-calib-01", "rc-trk-01-0", "rc-trk-01-1"]) == {"rc-calib-01": "done", "rc-trk-01-0": "running", "rc-trk-01-1": "queued"}
    assert rc.poll(L(), []) == {}


# ------------------------------------------------------------------------------------------------ attach by people (synthetic room)


from dancecap.cameras import attach_people as ap  # noqa: E402


def _skeleton(foot_xz, rng):
    """17 COCO joints of a standing person (world metres, Y up): nose 0 at 1.65 m, ankles 15 / 16 at 0.05 m."""
    y = np.array([1.65, 1.7, 1.7, 1.68, 1.68, 1.45, 1.45, 1.2, 1.2, 1.0, 1.0, 0.95, 0.95, 0.5, 0.5, 0.05, 0.05])
    x = np.array([0, -.03, .03, -.07, .07, -.18, .18, -.22, .22, -.2, .2, -.1, .1, -.1, .1, -.1, .1])
    z = rng.normal(0, 0.03, 17)
    return np.c_[x + foot_xz[0], y, z + foot_xz[1]]


def _look_at(C, target):
    zc = target - C
    zc = zc / np.linalg.norm(zc)
    xc = np.cross(zc, [0, 1.0, 0])
    xc /= np.linalg.norm(xc)
    yc = np.cross(zc, xc)
    return np.stack([xc, -yc, zc])         # rows = camera axes in the world; image y points DOWN


def _room(n_times=6, seed=0):
    rng = np.random.default_rng(seed)
    K = np.array([[1000.0, 0, 539.5], [0, 1000.0, 959.5], [0, 0, 1.0]])
    mains = {}
    for name, ang in (("a", 20), ("b", 55), ("c", 95)):
        C = np.array([6 * np.sin(np.radians(ang)), 1.5, 6 * np.cos(np.radians(ang))])
        R = _look_at(C, np.array([0, 1.0, 0]))
        mains[name] = {"t": np.arange(n_times) * 1.0, "K": np.tile(K, (n_times, 1, 1)), "R": np.tile(R, (n_times, 1, 1)),
                       "tvec": np.tile(-R @ C, (n_times, 1)), "C": np.tile(C, (n_times, 1)), "dist": np.zeros(n_times),
                       "ok": np.ones(n_times, bool), "frame_index": np.arange(n_times)}
    persons = [[_skeleton(rng.uniform(-2.5, 2.5, 2), rng) for _ in range(7)] for _ in range(n_times)]
    return rng, K, mains, persons


def _view_people(persons_t, R, C, K, w=1080, h=1920):
    out = []
    for X in persons_t:
        uv, z = ap.project_world(X, R, C, K, 0.0)
        out.append(np.c_[uv, np.where((z > 0.3) & (uv[:, 0] > 0) & (uv[:, 0] < w) & (uv[:, 1] > 0) & (uv[:, 1] < h), 0.95, 0.0)])
    return np.array(out, np.float32)


def test_triangulate_people_stand_on_the_floor_and_dedupe():
    rng, K, mains, persons = _room()
    pp = {c: [_view_people(persons[i], T["R"][i], T["C"][i], K) for i in range(len(T["t"]))] for c, T in mains.items()}
    p3 = ap.triangulate_people(mains, pp, np.array([0.0, 2.0, 4.0]))
    assert len(p3) == 3
    for q in p3:
        assert 4 <= len(q.X) <= 7                        # persons seen by two or three cameras, each kept once
        ank = np.nanmin(q.X[:, [15, 16], 1], axis=1)
        assert np.all(np.abs(ank - 0.05) < 0.03)
        assert np.all(np.abs(np.nanmax(q.X[:, 0, 1]) - 1.65) < 0.05)


def test_people_hits_peak_at_the_true_attach_and_refine_recovers_it():
    rng, K, mains, persons = _room(n_times=8)
    pp = {c: [_view_people(persons[i], T["R"][i], T["C"][i], K) for i in range(8)] for c, T in mains.items()}
    p3 = ap.triangulate_people(mains, pp, np.arange(0.0, 8.0, 1.0))
    # the attached camera: world pose looking at the dancers from the 4th side; its sub frame = a rotation Rt away from the world
    Ct = np.array([-5.0, 1.4, -3.0])
    Rt_true = Ro.from_euler("yxz", [40, 3, 175], degrees=True).as_matrix()
    R_wc = _look_at(Ct, np.array([0, 1.0, 0]))
    S = {"t": np.arange(8) * 1.0, "K": np.tile(K, (8, 1, 1)), "dist": np.zeros(8), "ok": np.ones(8, bool), "frame_index": np.arange(8),
         "R": np.tile(R_wc @ Rt_true, (8, 1, 1)), "native_size": np.array([1080, 1920])}
    S_people = [_view_people(persons[i], R_wc, Ct, K) for i in range(8)]
    views = ap.build_views(S, S_people, p3)
    assert len(views) == len(p3)
    thr = (10.0, 20.0)
    good = ap.people_hits(Rt_true, Ct, views, thr)
    assert good["visible"] >= 20 and good["median_px"] < 1.0 and good["share_under_20px"] > 0.95
    for name, (dR, dC) in {"centre": (np.eye(3), [0.5, 0, 0]), "yaw": (Ro.from_euler("y", 3, degrees=True).as_matrix(), [0, 0, 0]),
                           "pitch": (Ro.from_euler("x", 3, degrees=True).as_matrix(), [0, 0, 0])}.items():
        bad = ap.people_hits(dR @ Rt_true, Ct + np.array(dC), views, thr)
        assert bad["median_px"] > 15 and bad.get("share_under_10px", 0) < 0.25, (name, bad)
    # refinement from a prior that is 1.5 deg / 0.25 m off finds the truth again
    prior_R = Ro.from_rotvec(np.radians([1.0, -0.8, 0.7])).as_matrix() @ Rt_true
    prior_C = Ct + [0.2, 0.0, -0.15]
    before = ap.people_hits(prior_R, prior_C, views, thr)["median_px"]
    Rn, Cn, score = ap.kernel_refine(views, prior_R, prior_C, 1080)
    after = ap.people_hits(Rn, Cn, views, thr)
    assert after["median_px"] < 0.5 * before and np.linalg.norm(Cn - Ct) < np.linalg.norm(prior_C - Ct)


def test_pair_costs_zero_for_true_correspondences():
    rng, K, mains, persons = _room(n_times=1)
    A, B = mains["a"], mains["b"]
    pa = _view_people(persons[0], A["R"][0], A["C"][0], K)
    pb = _view_people(persons[0], B["R"][0], B["C"][0], K)
    fa, fb = ap.make_frame(pa, K, 0.0), ap.make_frame(pb, K, 0.0)
    Rab = B["R"][0] @ A["R"][0].T
    E = ap.skew(B["tvec"][0] - Rab @ A["tvec"][0]) @ Rab
    c = ap.pair_costs(E, fa, fb)
    assert np.allclose(np.diag(c)[np.isfinite(np.diag(c))], 0.0, atol=1e-3)
