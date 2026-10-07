"""Hands detail build (hands_detail + the refine_hands contact guard): tiny synthetic checks, no take data needed.

    .venv-nopo4d\\Scripts\\python.exe -m pytest dancecap/tests/test_hands_detail.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dancecap import face_common as fc  # noqa: E402
from dancecap import hands_detail as hd  # noqa: E402
from dancecap import refine_hands as rh  # noqa: E402

needs_model = pytest.mark.skipif(not fc.smplx_model_npz().exists(), reason="SMPL-X model not installed")


# ------------------------------------------------------------------------------------------------ gate helpers

def test_runs_and_clean_runs():
    g = np.zeros(40, bool)
    g[2:5] = True          # short run (3) -> dropped
    g[10:20] = True
    g[23:30] = True        # 3-frame gap to the previous run -> bridged
    assert hd.runs_of(g) == [(2, 4), (10, 19), (23, 29)]
    c = hd.clean_runs(g, fill_gap=4, min_run=10)
    assert hd.runs_of(c) == [(10, 29)]
    assert not hd.clean_runs(g, fill_gap=0, min_run=11).any()


def test_ramp_inside_never_leaves_the_run():
    g = np.zeros(30, bool)
    g[5:20] = True
    w = hd.ramp_inside(g, 4)
    assert np.all(w[:5] == 0) and np.all(w[20:] == 0)
    assert np.allclose(w[9:16], 1.0) and 0 < w[5] < 0.2 and 0 < w[19] < 0.2
    assert np.all(np.diff(w[5:10]) > 0)
    g2 = np.zeros(10, bool)
    g2[:6] = True          # touches the take's start: no ramp there
    assert hd.ramp_inside(g2, 3)[0] == 1.0


def test_blend_rotvecs_is_geodesic():
    r0 = np.zeros((3, 3))
    r1 = np.tile([0.0, 0.0, np.pi / 2], (3, 1))
    out = hd.blend_rotvecs(r0, r1, np.array([0.0, 0.5, 1.0]))
    assert np.allclose(out[0], 0) and np.allclose(out[2], r1[2])
    assert np.allclose(out[1], [0, 0, np.pi / 4])
    # NaN target where the weight is 0 is fine
    out = hd.blend_rotvecs(r0[:1], np.full((1, 3), np.nan), np.zeros(1))
    assert np.allclose(out, 0)


def test_gate_frames_needs_agreement_detections_and_no_hold():
    from dancecap.hands_detail import GateConfig
    T = 12
    dis = np.full(T, 10.0)
    dis[0] = 50.0                                   # regressors disagree
    dis[1] = np.nan                                 # one regressor missing
    dv = np.full(T, 3)
    dv[6:] = 0                                      # own detector misses ...
    dv[9] = 2                                       # ... except here (+-2 frames window)
    hh = np.zeros(T, bool)
    hh[4] = True                                    # in a hand-hand hold
    g = hd.gate_frames(dis, dv, hh, GateConfig(det_window=2))
    assert not g[0] and not g[1] and not g[4]
    assert g[2] and g[3] and g[5]
    assert g[7] and g[8] and g[11] and g[6]         # within 2 frames of the frame-9 detection or of frame 5
    dv2 = np.zeros(T, int)
    assert not hd.gate_frames(dis, dv2, hh, GateConfig()).any()


def test_handhand_frames_marks_both_hands():
    eps = [{"role": "follow", "side": "right", "region": "left hand", "kind": "hold", "hand_hand": True,
            "i0": 10, "i1": 20},
           {"role": "lead", "side": "right", "region": "upper back", "kind": "hold", "hand_hand": False,
            "i0": 0, "i1": 30}]
    m = hd.handhand_frames(eps, 40, margin=3)
    assert m[("follow", "right")][7:24].all() and not m[("follow", "right")][6]
    assert m[("lead", "left")][7:24].all()          # the partner's hand of the hold
    assert not m[("lead", "right")].any()           # hand-body hold is not excluded


def test_gated_target_drops_isolated_flip_and_bridges():
    T = 40
    base = Rot.from_rotvec([0.0, 0.0, 0.6]).as_matrix()
    loc = np.tile(base, (T, 1, 1))
    loc[12] = Rot.from_rotvec([0.0, 2.5, 0.0]).as_matrix()      # an isolated flip of a hidden hand
    g_raw = np.zeros(T, bool)
    g_raw[5:15] = True
    g_raw[20:30] = True
    g = hd.clean_runs(g_raw, fill_gap=6, min_run=8)
    tgt = hd.gated_target(loc, g_raw, g, sigma=2.0)
    assert np.isfinite(tgt[5:30]).all() and np.isnan(tgt[:5]).all() and np.isnan(tgt[30:]).all()
    ang = np.degrees(np.linalg.norm((Rot.from_matrix(base).inv() * Rot.from_rotvec(tgt[5:30])).as_rotvec(), axis=1))
    assert ang.max() < 2.0                            # the flip did not leak, the gap 15-19 is bridged


def test_judge_errors_reanchors_the_wrist():
    K = np.array([[800.0, 0, 400], [0, 800, 300], [0, 0, 1]])
    Ps = [K @ np.concatenate([np.eye(3), [[0.0], [0.0], [3.0]]], 1)]
    gc = {"01": {"K": K[None], "R": np.eye(3)[None], "t": np.array([[0, 0, 3.0]]), "ok": np.array([True])}}
    rng = np.random.default_rng(0)
    J = rng.normal(scale=0.05, size=(1, 21, 3))
    J[0, 12] = J[0, 0] + [0.18, 0, 0]
    kp = rh._proj(Ps[0], J[0])[None]
    obs = {"01": (kp + 25.0, np.ones((1, 21)))}       # detections shifted: same shape
    e = hd.judge_errors(J, obs, gc, J, np.ones(1, bool))
    assert e and abs(e[(0, "01")]) < 1e-9


# ------------------------------------------------------------------------------------------------ contact guard

def test_attract_mask_and_runs():
    ep = {"i0": 5, "i1": 30}
    assert rh.attract_mask(ep, 40).sum() == 26
    ep["attract_runs"] = [[5, 10], [20, 25]]
    m = rh.attract_mask(ep, 40)
    assert m.sum() == 12 and m[5] and m[10] and not m[11] and m[20] and not m[26]


def test_guard_ok_and_excess():
    movers = [("lead", "right")]
    rec = {"lead_right_rep_px_before": 13.68, "lead_right_rep_px_after": 16.49, "lead_right_arm_change_deg_max": 41.1}
    ok, why = rh.guard_ok(rec, movers)
    assert not ok and len(why) == 2
    rec2 = {"lead_right_rep_px_before": 13.68, "lead_right_rep_px_after": 14.5, "lead_right_arm_change_deg_max": 12.0}
    assert rh.guard_ok(rec2, movers)[0]                 # +0.8 px < max(1 px, 8 %)
    rec3 = {"lead_right_rep_px_before": float("nan"), "lead_right_rep_px_after": float("nan"),
            "lead_right_arm_change_deg_max": 30.0}
    assert not rh.guard_ok(rec3, movers)[0]             # no landmarks: judged on the arm change only
    assert rh._guard_excess(rec2, movers) == 0 and rh._guard_excess(rec, movers) > rh._guard_excess(rec3, movers) > 0


def test_guard_reference_is_the_worse_of_input_and_mamma():
    movers = [("lead", "right")]
    # the wrist gate improved the arm landmarks (13.7 -> 11.4); the contact fix may give that back, not more
    rec = {"lead_right_rep_px_before": 11.4, "lead_right_rep_px_base": 13.7, "lead_right_rep_px_after": 13.8,
           "lead_right_arm_change_deg_max": 17.0}
    assert rh._guard_ref(rec, "lead", "right") == 13.7 and rh.guard_ok(rec, movers)[0]
    assert not rh.guard_ok(dict(rec, lead_right_rep_px_after=15.3), movers)[0]
    rec2 = {"lead_right_rep_px_before": 9.0, "lead_right_rep_px_base": 8.0}
    assert rh._guard_ref(rec2, "lead", "right") == 9.0


def test_smooth_basis_rows_and_smoothness():
    B = rh.smooth_basis(40, 2, 2.0)
    assert B.shape == (40, 21) and np.allclose(B.sum(1), 1.0)
    rng = np.random.default_rng(4)
    d = B @ rng.normal(size=21)                         # any key values -> a smooth per-frame curve
    assert np.abs(np.diff(d, 2)).max() < 0.6 * np.abs(np.diff(rng.normal(size=40), 2)).max()
    assert np.allclose(rh.smooth_basis(5, 2, 0.0), np.eye(5))


def test_guard_steps_get_more_conservative():
    arm = [s[0] for s in rh.GUARD_STEPS]
    rep = [s[1] for s in rh.GUARD_STEPS]
    con = [s[2] for s in rh.GUARD_STEPS]
    assert arm == sorted(arm) and rep == sorted(rep) and con == sorted(con, reverse=True) and rh.GUARD_STEPS[0] == (1, 1, 1)


# ------------------------------------------------------------------------------------------------ outputs + metrics

def _contract(T=6):
    t = 56.0 + np.arange(T) / 30
    return {"t": t, "global_orient": np.zeros((T, 3)), "body_pose": np.zeros((T, 63)), "jaw_pose": np.zeros((T, 3)),
            "left_hand_pose": np.zeros((T, 45)), "right_hand_pose": np.zeros((T, 45)), "transl": np.zeros((T, 3)),
            "betas": np.zeros(16), "joints": np.zeros((T, 55, 3)), "source": np.array("test")}


def test_merge_face_takes_jaw_and_expression_not_the_neck():
    d = _contract()
    z = {"t": d["t"].copy(), "jaw_pose": np.full((6, 3), 0.1), "expression": np.ones((6, 10), np.float32),
         "expression_rest": np.zeros(10, np.float32), "n_expr": np.array(10), "confidence": np.full(6, 0.5),
         "neck_pose": np.full((6, 3), 0.3)}
    out, note = hd.merge_face(d, z)
    assert np.allclose(out["jaw_pose"], 0.1) and out["expression"].shape == (6, 10)
    assert np.allclose(out["body_pose"], 0) and not bool(out["face_neck_adopted"]) and note.startswith("face_expr")
    out2, _ = hd.merge_face(d, dict(z, adopt_neck=np.array(True)))
    assert np.allclose(out2["body_pose"][:, 33:36], 0.3) and bool(out2["face_neck_adopted"])
    assert np.allclose(d["jaw_pose"], 0)                                # input untouched
    bad, note = hd.merge_face(d, dict(z, t=d["t"] + 0.5))
    assert "not merged" in note and np.allclose(bad["jaw_pose"], 0)


def test_contact_state_hold_wins_over_touch():
    eps = [{"role": "lead", "side": "left", "kind": "touch", "i0": 0, "i1": 5},
           {"role": "lead", "side": "left", "kind": "hold", "i0": 3, "i1": 8},
           {"role": "lead", "side": "right", "kind": "near-miss", "i0": 0, "i1": 9},
           {"role": "follow", "side": "right", "kind": "hold", "i0": 0, "i1": 9}]
    st = hd.contact_state(eps, "lead", 10)
    assert st.dtype == np.int8 and st.shape == (10, 2)
    assert list(st[:, 0]) == [1, 1, 1, 2, 2, 2, 2, 2, 2, 0] and not st[:, 1].any()


def test_gap_stats():
    g = np.array([-0.03, -0.005, 0.0, 0.004, 0.009, 0.03])
    s = hd.gap_stats(g)
    assert s["n"] == 6 and s["penetration_cm_max"] == 3.0
    assert s["touching_within_1cm"] == round(4 / 6, 3) and s["apart_gt_2cm"] == round(1 / 6, 3)
    assert s["sunk_gt_1cm"] == round(1 / 6, 3) and s["p50"] == 0.2
    assert hd.gap_stats(np.array([np.nan]))["n"] == 0


def test_hf_rms_sees_jitter_not_motion():
    t = np.arange(120) / 30
    smooth = np.stack([0.3 * np.sin(2 * np.pi * 0.5 * t), 0.1 * t, np.zeros_like(t)], 1)[:, None]
    rng = np.random.default_rng(1)
    noisy = smooth + rng.normal(scale=0.002, size=smooth.shape)
    assert hd.hf_rms(smooth) < 1e-4
    assert 1.5e-3 < hd.hf_rms(noisy) < 4e-3


def test_hands_pod_build_spec_has_no_mano_and_writes_the_detail(tmp_path, monkeypatch):
    from dancecap import hands_pod
    work = tmp_path / "dancecap" / "work" / "tk"
    for p in ("hands/grid_obs.npz", "hands/jobs.json", "hands/detections.npz", "smplx/lead.npz", "face/expr_lead.npz",
              "mamma/ma_2d/tk_c8/cam01.npz"):
        (work / p).parent.mkdir(parents=True, exist_ok=True)
        (work / p).write_bytes(b"x")
    (work / "mamma" / "select.json").write_text(json.dumps({"chosen": "c8", "cams": ["01", "02"]}))
    np.savez(work / "smplx" / "lead.npz", t=np.arange(352) / 30.0)
    monkeypatch.setattr(hands_pod, "PACKAGE_ROOT", tmp_path / "dancecap")
    d = hands_pod.build_spec("tk")
    assert d["max_minutes"] == hands_pod.build_minutes(352) > hands_pod.build_minutes(176)
    flat = json.dumps(d["inputs"])
    assert "mano" not in flat.lower().replace("mamma", "")
    assert "dancecap/work/tk/frames/01/frames.json" in d["inputs"] and "dancecap/work/tk/frames/" not in d["inputs"]
    assert "dancecap/work/tk/face/expr_lead.npz" in d["inputs"] and "dancecap/work/tk/mamma/ma_2d/tk_c8/" in d["inputs"]
    assert "dancecap/work/tk/smplx_detail/" in d["outputs"] and "dancecap/work/tk/smplx_hands/lead.npz" in d["verify"]
    assert "--no-review" in d["commands"][0]["cmd"] and "refine_hands build tk" in d["commands"][0]["cmd"]
    (work / "hands" / "grid_obs.npz").unlink()
    with pytest.raises(SystemExit):
        hands_pod.build_spec("tk")


# ------------------------------------------------------------------------------------------------ model-dependent

@needs_model
def test_set_dancer_pose_updates_fk_and_drops_the_mesh_cache():
    import torch
    from dancecap.hand_kin import SmplxKin
    rng = np.random.default_rng(3)
    T = 3
    kin = SmplxKin(fc.smplx_model_npz().parent.parent, np.zeros(10))

    class D:
        pass
    dn = D()
    dn.kin, dn.transl = kin, torch.zeros((T, 3), dtype=torch.float64)
    dn.d = {"t": np.arange(T) / 30.0}
    dn._verts = "stale"
    P = rng.normal(scale=0.2, size=(T, 55, 3))
    rh.set_dancer_pose(dn, P)
    assert dn._verts is None and np.allclose(dn.d["body_pose"], P[:, 1:22].reshape(T, 63))
    assert np.allclose(dn.d["right_hand_pose"], P[:, 40:55].reshape(T, 45))
    ref = kin.forward(torch.tensor(P), dn.transl)
    assert np.allclose(dn.joints, ref["joints"].numpy())
    assert np.allclose(dn.hands["left"], kin.hand21(ref, "left").numpy())


@needs_model
def test_fit_hand_starts_from_the_gated_wrist(tmp_path):
    """FingerConfig.wrist_init replaces MAMMA's wrist as the IK's start / prior (no observations: returned as is)."""
    import torch
    from dancecap.hand_kin import SmplxKin
    T = 4
    kin = SmplxKin(fc.smplx_model_npz().parent.parent, np.zeros(10))

    class D:
        pass
    dn = D()
    dn.role, dn.t, dn.kin, dn.mdir = "lead", np.arange(T) / 30.0, kin, fc.smplx_model_npz().parent.parent
    dn.pose = torch.zeros((T, 55, 3), dtype=torch.float64)
    dn.out = kin.forward(dn.pose, torch.zeros((T, 3), dtype=torch.float64))
    dn.hands = {s: kin.hand21(dn.out, s).numpy() for s in ("left", "right")}
    wi = np.tile([0.1, -0.2, 0.3], (T, 1))
    np.savez(tmp_path / "wg.npz", lead_right_wrist=wi)
    cfg = rh.FingerConfig(wrist_init=str(tmp_path / "wg.npz"))
    f = rh.fit_hand(dn, "right", {}, {}, [], cfg)
    assert np.allclose(f["wrist"], wi)
    f = rh.fit_hand(dn, "left", {}, {}, [], cfg)                        # no entry for this hand: MAMMA's wrist
    assert np.allclose(f["wrist"], 0)
