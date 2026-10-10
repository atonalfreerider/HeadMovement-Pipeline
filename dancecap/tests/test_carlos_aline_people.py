"""Unit tests for the pure parts of dancecap.carlos_aline_people, carlos_aline_inputs and farm/ca_profile (synthetic numbers; no footage, no pod).

    .venv-nopo4d\\Scripts\\python.exe -m pytest dancecap/tests/test_carlos_aline_people.py -q      (from the workspace root)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dancecap import carlos_aline_inputs as ci  # noqa: E402
from dancecap import carlos_aline_people as cp  # noqa: E402
from dancecap.farm import ca_profile as cap  # noqa: E402


def test_redness_separates_red_from_white_and_grey():
    r = cp.redness(np.array([[200, 40, 50], [235, 235, 235], [20, 20, 20], [150, 140, 135]], float))
    assert r[0] > 0.2 and abs(r[1]) < 0.02 and abs(r[2]) < 0.05 and r[0] > r[3] + 0.15


def test_colour_model_llr_sign():
    m = cp.fit_colour_model(np.array([0.30, 0.34, 0.28, 0.33]), np.array([0.01, 0.03, 0.0, 0.05]))
    ll = cp.llr_aline(np.array([0.32, 0.02, np.nan]), m)
    assert ll[0] > 1 and ll[1] < -1 and ll[2] == 0.0
    assert np.all(np.abs(ll) <= cp.LLR_CLIP)


def test_box_iou_and_pair_states():
    a = np.array([0, 0, 10, 10.0])
    assert cp.box_iou(a, a) == 1.0 and cp.box_iou(a, np.array([20, 20, 30, 30.0])) == 0.0
    st = cp.pair_states(2)
    assert len(st) == 7 and not any(c == a_ != 0 for c, a_ in st)       # (0,0) (0,1) (0,2) (1,0) (1,2) (2,0) (2,1)


def frame(centres, h, llr):
    return {"c": np.array(centres, float).reshape(-1, 2), "h": np.array(h, float), "llr": np.array(llr, float)}


def test_tracker_holds_identities_through_a_crossing_with_weak_colour():
    """Carlos (llr -3) left, Aline (llr +3) right; for 20 frames the colour of both is unknown (0) while they cross: the motion continuity keeps the roles."""
    frames = []
    for t in range(60):
        xc = 100.0 + 8 * t                      # Carlos moves right ...
        xa = 700.0 - 8 * t                      # ... Aline moves left; they pass each other at t = 37
        amb = 20 <= t < 50
        frames.append(frame([[xc, 300], [xa, 300]], [400, 400], [0.0 if amb else -3.0, 0.0 if amb else 3.0]))
    res = cp.track_pair(frames)
    assert (res["cand"][:, 0] == 0).all() and (res["cand"][:, 1] == 1).all()
    assert res["p"][0].min() > 0.9


def test_tracker_colour_decides_when_the_order_is_swapped_by_the_detector():
    """The detector lists Aline first in every frame: the colour (not the list order) names the roles."""
    frames = [frame([[500, 300], [300, 300]], [400, 400], [3.0, -3.0]) for _ in range(10)]
    res = cp.track_pair(frames)
    assert (res["cand"][:, 0] == 1).all() and (res["cand"][:, 1] == 0).all()


def test_tracker_keeps_both_roles_around_a_short_gap_with_a_far_bystander_present():
    """One dancer pair plus a bystander far away in the image: Aline's slot is untouched by Carlos' three missing frames and Carlos is back on his own person after."""
    frames = []
    for t in range(40):
        c = [[300, 300], [350, 300]] + [[1500, 300]]
        llr = [-3.0, 3.0, -0.5]
        if 15 <= t < 18:                          # Carlos' detection is missing for three frames
            c, llr = [[350, 300], [1500, 300]], [3.0, -0.5]
        frames.append(frame(c, [400] * len(c), llr))
    res = cp.track_pair(frames)
    assert (res["cand"][:15, 0] == 0).all() and (res["cand"][18:, 0] == 0).all() and (res["cand"][:15, 1] == 1).all()
    assert res["cand"][15:18, 1].tolist() == [0, 0, 0]                                         # her index in the shortened frames


def test_move_cost_shapes_and_cap():
    T = cp.move_cost(np.array([[100.0, 100.0]]), np.array([400.0]), np.array([[110.0, 100.0], [900.0, 100.0]]), np.array([400.0, 400.0]))
    assert T.shape == (2, 3) and T[1, 1] < 0.1 and T[1, 2] == cp.MOVE_CAP and T[0, 1] == cp.REACQ_COST and T[1, 0] == cp.LOSE_COST


def test_static_cells_and_veto():
    dirs = np.array([[35.0, -6.4]] * 2000 + [[40.0, -9.0]] * 50)
    counts = cp.static_cells(dirs)
    assert cp.is_static(np.array([35.05, -6.38]), counts)
    assert not cp.is_static(np.array([40.0, -9.0]), counts)


def test_foot_direction_is_pan_independent():
    K = np.array([[3000.0, 0, 960], [0, 3000.0, 540], [0, 0, 1]])
    R0 = np.eye(3)
    a = np.radians(5.0)
    Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])         # camera panned by 5 degrees
    box = np.array([900, 300, 1000, 800.0])
    d0 = cp.foot_direction(box, K, R0, 0.0)
    # a fixed world point seen after the pan sits at a different pixel but gives the same world direction
    ray = np.r_[0.0, 0.1, 1.0]
    pix0 = K @ ray
    pix0 = pix0[:2] / pix0[2]
    ray1 = Ry @ ray
    pix1 = K @ ray1
    pix1 = pix1[:2] / pix1[2]
    b0 = np.array([pix0[0] - 50, 0, pix0[0] + 50, pix0[1]])
    b1 = np.array([pix1[0] - 50, 0, pix1[0] + 50, pix1[1]])
    a0, a1 = cp.foot_direction(b0, K, R0, 0.0), cp.foot_direction(b1, K, Ry, 0.0)
    assert abs(((a0[0] - a1[0] + 180) % 360) - 180) < 1e-6 and abs(a0[1] - a1[1]) < 1e-6        # azimuth compared modulo 360 (this synthetic camera looks along +Z)
    assert d0.shape == (2,)


def test_floor_point_of_a_standing_person():
    """A camera 1.5 m high, 10 m from a person standing at the origin, ankles seen: the floor point comes back at the origin."""
    C = np.array([0.0, 1.5, 10.0])
    R = np.diag([1.0, -1.0, -1.0])
    K = np.array([[3000.0, 0, 960], [0, 0 + 3000.0, 540], [0, 0, 1]])
    ankle = np.array([0.1, 0.0, 0.0])                                         # floor_point treats the ankle joint as ON the floor plane (documented bias ~0.09 m / camera height of the depth)
    Xc = R @ (ankle - C)
    uv = (K @ Xc)[:2] / Xc[2]
    kp = np.zeros((17, 2))
    conf = np.zeros(17)
    kp[15] = kp[16] = uv
    conf[15] = conf[16] = 1.0
    f = cp.floor_point(kp, conf, K, R, C, 0.0)
    assert np.allclose(f, [0.1, 0.0], atol=0.02)


def test_pose_table_orders_by_confidence_and_pads():
    z = {"frame": np.array([0, 0, 2]), "box": np.zeros((3, 4), np.float32), "conf": np.array([0.3, 0.9, 0.5], np.float32), "kp": np.zeros((3, 17, 3), np.float32),
         "rgb": np.ones((3, 3), np.float32), "n_frames": np.array(4)}
    z["kp"][0, 0, 0], z["kp"][1, 0, 0] = 1.0, 2.0
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "pose.npz"
        np.savez(p, **z)
        t = ci.pose_table(p)
    assert list(t["n"]) == [2, 0, 1, 0] and t["kp"].shape[:2] == (4, 2)
    assert t["kp"][0, 0, 0, 0] == 2.0 and np.isnan(t["kp"][1, 0, 0, 0]) and t["det_conf"][0, 0] == np.float32(0.9)


def test_ca_profile_ids_never_meet_the_recap():
    assert cap.SHAPE_ID.startswith(cap.ID_PREFIX) and cap.FINAL_ID.startswith(cap.ID_PREFIX)
    assert cap.JOB_ENV["HM_SHAPE_JOB"] == cap.SHAPE_ID and cap.SHAPE_TAKE == f"carlos_aline_{cap.SHAPE_ID}"
    assert cap.PIN_REL.endswith("mamma_pin_betas.json") and cap.SHAPE_TAKE in cap.PIN_REL


def test_shape_window_prefers_both_cameras_apart():
    t = np.arange(0.0, 200.0, 1 / 30.0)
    def cam(good_from, good_to):
        conf = np.zeros((len(t), 2))
        conf[(t >= good_from) & (t < good_to)] = 0.9
        fl = np.zeros((len(t), 2, 2))
        fl[:, 1, 0] = 1.5                                                    # 1.5 m apart everywhere
        return {"t": t, "ok": np.ones(len(t), bool), "conf": conf, "merged": np.zeros((len(t), 2), bool), "floor": fl}
    tt = {"01": cam(60, 100), "02": cam(60, 100)}
    sw = cap.shape_window(tt)
    assert 60 <= sw["t0"] <= 70 and sw["k1"] - sw["k0"] + 1 == 900 and sw["stats"]["both_cameras_both_dancers"] > 0.95


def test_fixed_sync_doc_is_read_by_the_mamma_offset_reader(tmp_path):
    """recap_front.fixed_sync_doc writes what dancecap.sync would: corrected time = t + delta, master 0, the clocks the frames use recorded."""
    import json

    from dancecap import mamma_dyn as md
    from dancecap import recap_front as rf
    work = tmp_path / "carlos_aline_cagtest"
    for cam, (s, o) in {"01": (1.0, 0.0), "02": (1.0, 14.815753692354448)}.items():
        (work / "frames" / cam).mkdir(parents=True)
        (work / "frames" / cam / "frames.json").write_text(json.dumps({"slope": s, "offset": o, "frames": []}))
    doc = rf.fixed_sync_doc(work, {"01": 5.0, "02": -62.0}, "test", ["01", "02"])
    assert doc["offsets"]["01"]["delta_ms"] == 0.0 and doc["offsets"]["02"]["delta"] == -0.062          # the master is never moved
    assert doc["clocks"]["sources"]["02"] == [1.0, 14.815753692354448]
    (work / "sync").mkdir()
    (work / "sync" / "offsets.json").write_text(json.dumps(doc))
    off = md.read_sync_offsets(work)
    assert off["02"]["delta"] == -0.062 and off["01"]["delta"] == 0.0 and off["02"]["rate"] == 0.0
