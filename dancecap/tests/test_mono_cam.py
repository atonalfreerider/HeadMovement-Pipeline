"""dancecap.cameras.mono_cam: the world solve of a fixed-centre camera, the axis rule, the camera file contract, the frame checks (synthetic only)."""
import json
import re
from pathlib import Path

import numpy as np
import pytest

from dancecap.cameras import mono_cam as MC
from dancecap.cameras import ptz as P
from dancecap.cameras import single_view as SV

W_IMG, H_IMG = 540, 960
STAT = {1: 1.80, 2: 1.68}
SOLE = {1: 0.025, 2: 0.02}


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _look_at(C, target, roll_deg=0.0):
    """world -> camera rotation (OpenCV axes: x right, y down, z forward) of a level camera at C looking at target, then rolled."""
    z = _unit(np.asarray(target, float) - C)
    x = _unit(np.cross(z, [0.0, 1.0, 0.0]))
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    a = np.radians(roll_deg)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])
    return Rz @ R


def _scene(n=40, h_cam=1.45, C=(1.1, None, 4.3)):
    """A phone at a fixed centre panning / tilting after a couple, 1x (800 px) and a 2x stretch (frames 20-29), COCO keypoints of both."""
    rng = np.random.default_rng(3)
    C = np.array([C[0], h_cam, C[2]])
    t = np.arange(n)
    lead = np.c_[-0.30 + 0.25 * np.sin(t / 6), np.zeros(n), 0.10 + 0.2 * np.cos(t / 7)]
    foll = np.c_[0.35 + 0.2 * np.sin(t / 5 + 1), np.zeros(n), -0.10 + 0.15 * np.cos(t / 8)]
    f = np.where((t >= 20) & (t < 30), 1600.0, 800.0)
    Rw = np.stack([_look_at(C, 0.5 * (lead[i] + foll[i]) + [0.2 * np.sin(i / 4), 0.9 + 0.1 * np.cos(i / 5), 0], rng.uniform(-2, 2))
                   for i in range(n)])
    R_ref = np.einsum("nab,cb->nac", Rw, Rw[0])                 # reference = camera 0: R_i R_0^T
    lens = P.Lens(W_IMG, H_IMG)
    kp = np.zeros((n, 2, 17, 3))
    box = np.zeros((n, 2, 4))
    for r, pos in ((0, lead), (1, foll)):
        rid = r + 1
        S, sole = STAT[rid], SOLE[rid]
        hts = {"ank": MC.ANKLE_OVER_STATURE * S + sole, "eye": 0.936 * S + sole, "sh": 0.82 * S + sole, "hip": 0.53 * S + sole}
        for i in range(n):
            px = {}
            for k, hh in hts.items():
                Xc = Rw[i] @ (pos[i] + [0, hh, 0] - C)
                px[k] = Xc[:2] / Xc[2] * f[i] + lens.c
            for j, k in ((SV.LANK, "ank"), (SV.RANK, "ank"), (SV.LEYE, "eye"), (SV.REYE, "eye"), (SV.NOSE, "eye"), (SV.LSH, "sh"),
                         (SV.RSH, "sh"), (SV.LHIP, "hip"), (SV.RHIP, "hip")):
                kp[i, r, j] = [*px[k], 0.9]
            ys = [px[k][1] for k in px]
            box[i, r] = [px["eye"][0] - 40, min(ys) - 30, px["eye"][0] + 40, max(ys) + 5]
    return {"C": C, "Rw": Rw, "R_ref": R_ref, "f": f, "lens": lens, "kp": kp, "box": box, "lead": lead, "foll": foll,
            "up_ref": Rw[0] @ np.array([0, 1.0, 0]), "two_x": f > 1000}


def _solve(sc, **kw):
    dets = MC.dancer_detections(sc["kp"], np.full(sc["kp"].shape[:2], 0.9), np.arange(len(sc["f"])), (W_IMG, H_IMG), sc["box"])
    eye = {r: 0.936 * STAT[r] + SOLE[r] for r in (1, 2)}
    ank = {r: MC.ANKLE_OVER_STATURE * STAT[r] + SOLE[r] for r in (1, 2)}
    return dets, MC.world_solve(sc["R_ref"], sc["f"], sc["lens"], sc["up_ref"], dets, eye, ank, sc["two_x"], **kw)


def test_world_solve_recovers_height_and_origin():
    sc = _scene()
    dets, ws = _solve(sc, min_upright=30)
    assert dets["height_ok"].all() and ws["upright"].all()
    assert abs(ws["camera_height_m"] - 1.45) < 2e-3
    wide = ~sc["two_x"]
    origin_true = np.concatenate([sc["lead"][wide], sc["foll"][wide]]).mean(0)
    # true world -> solved world through the shared reference frame: X_rec = Wm^T R_0 (X - C) + C_rec
    to_rec = lambda X: ws["Wm"].T @ (sc["Rw"][0] @ (X - sc["C"])) + ws["C"]          # noqa: E731
    assert np.linalg.norm(to_rec(origin_true)) < 0.01
    assert abs(ws["distance_m"] - np.hypot(*(sc["C"] - origin_true)[[0, 2]])) < 0.01
    assert np.allclose(ws["Wm"].T @ (sc["Rw"][0] @ [0, 1.0, 0]), [0, 1, 0], atol=1e-9)        # Y = up
    fw = ws["floor_world"]
    assert np.isfinite(fw).all() and np.allclose(fw[:, 1], 0)
    assert ws["origin_feet_samples"]["lead"] == wide.sum() and ws["origin_feet_samples"]["follow"] == wide.sum()


def test_axis_rule_z_points_to_the_camera():
    sc = _scene()
    _, ws = _solve(sc)
    C, Wm = ws["C"], ws["Wm"]
    assert abs(C[0]) < 1e-9 and C[2] > 3.0 and abs(C[1] - ws["camera_height_m"]) < 1e-12
    assert np.isclose(np.linalg.det(Wm), 1.0) and np.allclose(Wm.T @ Wm, np.eye(3), atol=1e-12)          # right handed, orthonormal
    assert np.allclose(np.cross(Wm[:, 1], Wm[:, 2]), Wm[:, 0])                                           # X = Y x Z
    R = np.einsum("nab,bc->nac", sc["R_ref"], Wm)
    fwd = R[:, 2, :]                                                                                     # camera z axis in world coordinates
    assert (fwd[:, 2] < -0.8).all()                                                                      # the camera looks along about -Z
    # the sign of the given gravity does not matter
    sc2 = dict(sc, up_ref=-sc["up_ref"])
    _, ws2 = _solve(sc2)
    assert np.allclose(ws2["Wm"], Wm) and abs(ws2["camera_height_m"] - ws["camera_height_m"]) < 1e-12


def test_height_fallback_when_few_upright_samples():
    h = np.array([1.40, 1.42, 1.44, 1.46, 1.38, 1.41])
    role = np.array([1, 1, 1, 2, 2, 2])
    hs = MC.camera_height_from_samples(h, np.ones(6, bool), np.zeros(6, bool), role, np.zeros(6, bool), min_upright=30)
    assert hs["fallback_all_samples"] and hs["roles"]["lead"]["used"] == "all"
    assert abs(hs["camera_height_m"] - np.median([1.42, 1.41])) < 1e-12


def test_focal_from_moge_uses_wide_frames_only():
    sk_f = np.r_[np.full(10, 1000.0), np.full(10, 2000.0), np.full(10, 1010.0)]
    fr = np.arange(0, 30, 2) * 15
    f_at = sk_f[fr // 15]
    moge = np.where(f_at > 1500, 0.6, 0.8) * f_at
    got = MC.focal_from_moge(fr, moge, sk_f, 15)
    assert abs(got["s"] - 0.8) < 1e-9 and got["n_wide"] == 10
    pick = MC.pick_moge_frames(np.arange(30) * 15, sk_f, n=8)
    assert len(pick) == 8 and all(sk_f[k // 15] < 1100 for k in pick)


def _cams(n=12):
    rng = np.random.default_rng(0)
    rv = rng.normal(scale=0.05, size=(n, 3))
    f = np.full(n, 800.0)
    Wm = np.eye(3)
    C = np.array([0.0, 1.4, 4.0])
    R, tvec, K, dist, _ = SV.camera_arrays(rv, f, Wm, C, (W_IMG, H_IMG), -0.01)
    pts = np.arange(n) / 29.97
    return MC.camera_npz(R, tvec, K, -0.01, C, pts, np.ones(n, bool), np.full(n, 1.2), np.full(n, 300), np.zeros(n, bool), f,
                         np.zeros(n, bool), (W_IMG, H_IMG), 1.4)


def test_camera_file_contract(tmp_path):
    n = 12
    cams = _cams(n)
    pr = {"width": W_IMG, "height": H_IMG, "rotation": 0, "avg_fps": 29.97, "fps": 29.97, "duration_s": 0.4}
    MC.write_outputs(tmp_path, "t1", "clip.mp4", pr, cams, {"camera_height_m": 1.4})
    z = np.load(tmp_path / "01.npz")
    want = {"t": (n,), "pts": (n,), "frame_index": (n,), "K": (n, 3, 3), "R": (n, 3, 3), "tvec": (n, 3), "dist": (n,), "ok": (n,),
            "reproj_px": (n,), "n_inliers": (n,), "interpolated": (n,), "focal_px": (n,), "zoom_2x": (n,), "centre_world": (3,), "pp_px": (2,),
            "camera_height_m": (), "native_size": (2,), "work_size": (2,), "camera_class": (), "type_model": (), "attached": (), "t_clock": ()}
    assert set(z.files) == set(want)
    for k, s in want.items():
        assert z[k].shape == s, k
    assert str(z["camera_class"]) == "handheld_zoom" and str(z["type_model"]) == "fixed_centre" and not bool(z["attached"])
    assert np.array_equal(z["frame_index"], np.arange(n)) and np.array_equal(z["t"], z["pts"])
    assert np.allclose(np.einsum("nab,b->na", z["R"], z["centre_world"]) + z["tvec"], 0)               # tvec = -R C
    w = np.load(tmp_path / "01_world.npz")
    assert set(w.files) == {"t", "R", "tvec", "C", "K", "dist", "ok"} and w["C"].shape == (n, 3)
    rc = json.loads((tmp_path / "recap_cams.json").read_text())
    row = rc["cams"]["01"]
    assert {"file", "width", "height", "rotation", "avg_fps", "slope", "offset"} <= set(row)
    assert (row["file"], row["width"], row["height"], row["slope"], row["offset"]) == ("clip.mp4", W_IMG, H_IMG, 1.0, 0.0)
    sim = json.loads((tmp_path / "world_from_sfm.json").read_text())
    assert sim["s"] == 1.0 and np.allclose(sim["R"], np.eye(3)) and sim["t"] == [0.0, 0.0, 0.0]


def test_frame_count_mismatch_raises():
    MC.check_frames(10, 10, np.arange(10))
    with pytest.raises(ValueError):
        MC.check_frames(10, n_decoded=9)
    with pytest.raises(ValueError):
        MC.check_frames(10, track_frames=np.r_[0:5, 6:10])
    with pytest.raises(ValueError):
        MC.check_frames(10, track_frames=np.arange(11))
    c = _cams(12)
    with pytest.raises(ValueError):
        MC.camera_npz(c["R"], c["tvec"], c["K"], 0.0, c["centre_world"], np.arange(13) / 30.0, c["ok"], c["reproj_px"], c["n_inliers"],
                      c["interpolated"], c["focal_px"], c["zoom_2x"], (W_IMG, H_IMG), 1.4)


def test_rotated_stream_is_refused():
    st = {"coded_width": 1920, "coded_height": 1080, "width_display": 1080, "height_display": 1920, "rotation": -90}
    # OpenCV rotates: ptz_run's coded size is wrong
    assert len(MC.size_problems(st, {"width": 1080, "height": 1920, "orientation_auto": True})) == 1
    assert len(MC.size_problems(st, {"width": 1920, "height": 1080, "orientation_auto": False})) == 2       # OpenCV does not: frames sideways
    ok = {"coded_width": 538, "coded_height": 960, "width_display": 538, "height_display": 960, "rotation": 0}
    assert MC.size_problems(ok, {"width": 538, "height": 960, "orientation_auto": False}) == []


def test_moge_pins_match_single_cam_pod():
    src = (Path(MC.__file__).parent / "single_cam_pod.py").read_text(encoding="utf-8")
    assert re.search(r'MOGE_COMMIT = "([0-9a-f]+)"', src).group(1) == MC.MOGE_COMMIT
    assert re.search(r'UTILS3D_COMMIT = "([0-9a-f]+)"', src).group(1) == MC.UTILS3D_COMMIT
