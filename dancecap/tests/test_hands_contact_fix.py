"""Fixes after the adversarial review of the hands build (2026-10-07): hold-gap bridging, one shared contact state per
hold, the stricter guard (per camera, jitter, palm displacement, finger crossings), keeping the input when every
attempt fails, finger spacing thresholds, the contact-aware wrist gate. Tiny synthetic inputs only.

    .venv-nopo4d\\Scripts\\python.exe -m pytest dancecap/tests/test_hands_contact_fix.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dancecap import hands_detail as hd  # noqa: E402
from dancecap import mesh_contact as mc  # noqa: E402
from dancecap import refine_hands as rh  # noqa: E402


# ------------------------------------------------------------------------------------------------ hand-hand holds

def test_bridge_runs_needs_short_gap_and_strong_evidence():
    ev = np.full(200, 1.0)
    runs, gaps = rh.bridge_runs([[1, 54], [69, 128], [134, 175]], ev)
    assert runs == [[1, 175]] and gaps == [[55, 68], [129, 133]]           # larissa_kadu's two drifting gaps
    ev2 = ev.copy()
    ev2[60] = 0.5                                                           # one frame below min_ev: no bridge
    runs, gaps = rh.bridge_runs([[1, 54], [69, 128]], ev2)
    assert runs == [[1, 54], [69, 128]] and gaps == []
    ev3 = ev.copy()
    ev3[55:69] = 0.65                                                       # median below med_ev
    assert rh.bridge_runs([[1, 54], [69, 128]], ev3)[1] == []
    assert rh.bridge_runs([[1, 30], [50, 80]], ev)[1] == []                 # 19-frame gap > max_gap
    assert rh.bridge_runs([], ev) == ([], [])


def _hold_records():
    T = 40
    eps = [{"role": "follow", "side": "right", "region": "left hand", "kind": "hold", "hand_hand": True,
            "i0": 0, "i1": 39},
           {"role": "lead", "side": "left", "region": "right hand", "kind": "hold", "hand_hand": True,
            "i0": 0, "i1": 15},
           {"role": "lead", "side": "left", "region": "right hand", "kind": "hold", "hand_hand": True,
            "i0": 22, "i1": 39}]
    series = {("follow", "right"): {"evidence": np.ones(T)}, ("lead", "left"): {"evidence": np.full(T, 0.3)}}
    return T, eps, series


def test_merge_hold_records_bridges_with_the_better_record():
    T, eps, series = _hold_records()
    merged, twin = rh.merge_hold_records(eps[0], eps, series, ("hold", "touch"), rh.ContactConfig())
    assert len(twin) == 2 and merged["union"] == [0, 39] and merged["partner_side"] == "left"
    assert merged["attract_runs"] == [[0, 39]] and merged["bridged_gaps"] == [[16, 21]]
    series[("follow", "right")]["evidence"][16:22] = 0.4                    # nobody sees the hold there
    merged, _ = rh.merge_hold_records(eps[0], eps, series, ("hold", "touch"), rh.ContactConfig())
    assert merged["attract_runs"] == [[0, 15], [22, 39]] and merged["bridged_gaps"] == []


def test_contact_state_is_shared_by_both_hands_of_a_hold():
    T, eps, series = _hold_records()
    series[("follow", "right")]["evidence"][16:22] = 0.4
    merged, twin = rh.merge_hold_records(eps[0], eps, series, ("hold", "touch"), rh.ContactConfig())
    for k in ("attract_runs", "bridged_gaps", "union", "partner_side"):
        eps[0][k] = merged[k]
    for e in twin:
        e["merged_into"] = 0
    lead = hd.contact_state(eps, "lead", T)
    follow = hd.contact_state(eps, "follow", T)
    assert np.array_equal(lead[:, 0], follow[:, 1])                          # lead left == follow right, every frame
    assert lead[5, 0] == 2 and lead[18, 0] == 1 and not lead[:, 1].any()     # hold / touch in the un-bridged gap


# ------------------------------------------------------------------------------------------------ guard + policy

def _rec(**kw):
    base = {"lead_right_rep_px_before": 12.0, "lead_right_rep_px_after": 12.3, "lead_right_arm_change_deg_max": 8.0,
            "lead_right_cam_px_before": {"02": (16.5, 300), "06": (30.2, 300)},
            "lead_right_cam_px_after": {"02": (17.0, 300), "06": (31.0, 300)},
            "lead_right_wrist_jitter_mm_before": 2.2, "lead_right_wrist_jitter_mm_after": 2.5,
            "lead_right_elbow_jitter_mm_before": 1.8, "lead_right_elbow_jitter_mm_after": 1.9,
            "lead_right_palm_disp_cm_p95_good": 2.5, "lead_right_disp_cap_cm": 3.0,
            "lead_right_cross_frames_before": 0, "lead_right_cross_frames_after": 1, "window_frames": 176}
    base.update(kw)
    return base


def test_guard_catches_per_camera_jitter_displacement_and_crossings():
    m = [("lead", "right")]
    assert rh.guard_ok(_rec(), m)[0]
    # the review's numbers: median over all cameras fine, cam 02 16.5 -> 38.7 px
    ok, why = rh.guard_ok(_rec(lead_right_cam_px_after={"02": (38.7, 300), "06": (31.0, 300)}), m)
    assert not ok and "cam 02" in why[0]
    assert rh.guard_ok(_rec(lead_right_cam_px_after={"02": (38.7, 5), "06": (31.0, 300)}), m)[0]   # too few obs
    ok, why = rh.guard_ok(_rec(lead_right_wrist_jitter_mm_after=4.9), m)      # 2.2 -> 4.9 mm (review)
    assert not ok and "jitter" in why[0]
    assert not rh.guard_ok(_rec(lead_right_palm_disp_cm_p95_good=6.6), m)[0]
    ok, why = rh.guard_ok(_rec(lead_right_cross_frames_after=45), m)          # 0 -> 45 frames (review)
    assert not ok and "crossings" in why[0]
    assert rh.guard_ok({"lead_right_rep_px_before": 1.0, "lead_right_rep_px_after": 1.5}, m)[0]   # missing keys pass


def test_choose_attempt_keeps_the_input_when_every_attempt_fails():
    m = [("lead", "right")]
    a0 = (_rec(lead_right_rep_px_after=15.0, guard_ok=False), "res0")
    a1 = (_rec(lead_right_rep_px_after=14.0, guard_ok=False), "res1")
    adopt, closest, lab = rh.choose_attempt([a0, a1], m)
    assert adopt is None and closest is a1 and "input pose kept" in lab
    adopt, _, lab = rh.choose_attempt([a0, a1], m, keep_failed=True)
    assert adopt is a1 and "closest" in lab
    a2 = (_rec(guard_ok=True), "res2")
    assert rh.choose_attempt([a0, a2], m)[0] is a2


def test_guard_steps_end_sink_only():
    assert rh.GUARD_STEPS[-1][2] == 0 and all(s[2] > 0 for s in rh.GUARD_STEPS[:-1])


# ------------------------------------------------------------------------------------------------ finger spacing

def test_self_thresholds_never_closer_than_the_input():
    lay = {"radius": np.array([0.008, 0.008, 0.008]), "pair": np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], bool)}
    D0 = np.array([[[0, 0.030, 0.014], [0.030, 0, 0.010], [0.014, 0.010, 0]]])
    thr = rh.self_thresholds(D0, lay, floor=0.95, slack=0.0)
    assert thr[0, 0, 0] == -1 and abs(thr[0, 0, 1] - 0.016) < 1e-12      # apart: r_a + r_b
    assert abs(thr[0, 0, 2] - 0.0152) < 1e-12                              # 14 mm overlap -> pushed to 0.95 x 16 mm
    assert abs(thr[0, 1, 2] - 0.0152) < 1e-12
    D1 = D0.copy()
    D1[0, 0, 2] = D1[0, 2, 0] = 0.0155
    assert abs(rh.self_thresholds(D1, lay, 0.95, 0.0)[0, 0, 2] - 0.0155) < 1e-12   # no closer than the input


def test_spacing_penalty_after_a_contact_like_squeeze():
    """Fingers squeezed together by a contact step: the capsule term is positive exactly for the squeezed pair."""
    import torch
    radii = {f: np.full(3 if f != "thumb" else 2, 0.008) for f in mc.CHAINS21}
    lay = mc.capsule_layout(radii, samples=3)
    J = np.zeros((1, 21, 3))
    for i, (f, ch) in enumerate(mc.CHAINS21.items()):
        for n, k in enumerate(ch):
            J[0, k] = [0.02 * i, 0.0, 0.03 * n]                             # 20 mm spacing: apart
    X0 = mc.capsule_points(J, lay)
    thr = torch.tensor(rh.self_thresholds(np.linalg.norm(X0[:, :, None] - X0[:, None], axis=-1), lay, 0.95))
    Jt = torch.tensor(J)

    def pen(Jx):
        X = mc.capsule_points(Jx, lay)
        return float((torch.relu(thr - (X[:, :, None] - X[:, None]).norm(dim=-1)) ** 2).sum())
    assert pen(Jt) == 0
    J2 = Jt.clone()
    J2[0, list(mc.CHAINS21["middle"]), 0] -= 0.008                          # middle pushed 8 mm onto the index
    assert pen(J2) > 0


# ------------------------------------------------------------------------------------------------ wrist gate

def test_contact_limit_cuts_the_gate_where_the_hand_would_sink(monkeypatch):
    T = 40
    wg = np.zeros(T)
    wg[5:35] = 1.0
    calls = {}

    def fake_sink(dn, side, wrists, frames, Vo, device=None, topo=None):
        n = len(wrists)
        calls["n"] = n
        out = np.zeros((n, len(frames)))
        deep = (frames >= 18) & (frames <= 22)
        if n == 1:                                       # the final check
            return out
        for i in range(1, n):                            # candidates 1.0, 0.75, 0.5, 0.25 of the gate
            c = (1.0, 0.75, 0.5, 0.25)[i - 1]
            out[i] = np.where(deep, 0.11 * c, 0.0)       # the full gate sinks 11 cm on frames 18-22
        out[0] = np.where(deep, 0.02, 0.0)               # MAMMA's hand sinks 2 cm there
        return out
    monkeypatch.setattr(hd, "hand_sink", fake_sink)
    w, info = hd.contact_limit(None, "right", np.zeros((T, 3)), np.zeros((T, 3)), wg, None, hd.GateConfig())
    # allowed: max(2 cm, 1 cm) + 1 cm = 3 cm -> 0.25 x 11 = 2.75 cm passes, 0.5 x 11 = 5.5 does not
    assert np.all(w[18:23] <= 0.25 + 0.1) and np.all(w[:5] == 0) and np.all(w[35:] == 0)
    assert np.allclose(w[5:9], 1.0, atol=0.01) and np.allclose(w[31:35], 1.0, atol=0.01)   # cut stays local
    assert info["frames_cut"] > 0 and info["sink_cm_max_full_gate"] == 11.0 and info["sink_cm_max_mamma"] == 2.0


def test_gap_stats_reports_floating():
    g = np.array([-0.03, -0.005, 0.0, 0.004, 0.015, 0.03])
    s = hd.gap_stats(g)
    assert s["float_gt_1cm"] == round(2 / 6, 3) and s["apart_gt_2cm"] == round(1 / 6, 3)
    assert s["float_cm_max"] == 3.0 and s["penetration_cm_max"] == 3.0


def test_jitter_numbers_rise_with_noise():
    class D:
        pass
    import torch
    T = 60
    t = np.arange(T) / 30
    rng = np.random.default_rng(2)
    H = np.zeros((T, 21, 3))
    H[:] = np.stack([0.2 * np.sin(t), 0.1 * t, np.zeros(T)], 1)[:, None] + rng.normal(scale=0.01, size=(1, 21, 3))
    dn = D()
    dn.hands = {"right": H}
    dn.joints = np.zeros((T, 55, 3))
    dn.pose = torch.zeros((T, 55, 3), dtype=torch.float64)
    a = hd.jitter(dn, "right")
    H2 = H + rng.normal(scale=0.003, size=H.shape)
    dn.hands = {"right": H2}
    dn.pose = torch.tensor(rng.normal(scale=0.02, size=(T, 55, 3)))
    b = hd.jitter(dn, "right")
    assert b["wrist_mm"] > a["wrist_mm"] and b["wrist_acc_mm_p95"] > a["wrist_acc_mm_p95"]
    assert b["elbow_ang_acc_deg_p95"] > a["elbow_ang_acc_deg_p95"] == 0


def test_guard_wrist_may_only_turn_back_towards_mamma():
    m = [("lead", "right")]
    rec = _rec(lead_right_arm_change_deg_max=42.7, lead_right_shoulder_elbow_change_deg_max=18.7,
               lead_right_wrist_excess_deg_max=0.0, lead_right_wrist_toward_mamma_frames=30)
    assert rh.guard_ok(rec, m)[0]                     # 42.7 deg at the wrist, but back towards MAMMA's wrist
    ok, why = rh.guard_ok(dict(rec, lead_right_wrist_excess_deg_max=12.0), m)
    assert not ok and "wrist" in why[0]
    ok, why = rh.guard_ok(dict(rec, lead_right_shoulder_elbow_change_deg_max=26.0), m)
    assert not ok and "shoulder / elbow" in why[0]
    assert not rh.guard_ok(_rec(lead_right_arm_change_deg_max=30.0), m)[0]   # older records: all three joints
