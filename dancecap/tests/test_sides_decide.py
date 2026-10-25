"""sides.hmm / sides.decide (DESIGN 8): dead-zone noise stays undecided, a direct flip inside the entangled band costs ~14 nats and needs
>= 0.5 s of strong contrary evidence, the prior alone never decides at 0.5 nats (and does at 1.5 only where the cameras are blind),
minimum runs, the Schmitt exit, the MNAR sign, a planted view reliability, the R4 generative visibility test on a synthetic scene, and the
yaw Viterbi on 2.6 turns of noisy half-circle bits.

    .venv-nopo4d\\Scripts\\python.exe -m pytest dancecap/tests/test_sides_decide.py -q -p no:cacheprovider     (from the workspace root)
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from dancecap.sides import decide, geom, hmm
from dancecap.sides.config import SidesCfg
from dancecap.sides.evidence import cue_reliability, missing_term

CFG = SidesCfg()


def test_dead_zone_noise_is_undecided_and_never_flips():
    rng = np.random.default_rng(1)
    m = rng.normal(0.0, 0.01, 600)                          # elbows level within 1 cm: inside the 4-cm dead zone
    L = decide.l3d(m, np.full(600, 0.02), 0.04, 4.0, 0.004)
    assert np.all(L == 0)
    d = hmm.decide_sign(L, np.ones(600, bool), np.zeros(600, bool), CFG)
    assert np.all(d["s"] == 0)


def test_direct_flip_in_band_costs_about_14_nats():
    out_m, band_m = hmm.sign_transitions(CFG, 3)
    assert -math.log(band_m[1, 2]) >= 13.8 and -math.log(band_m[2, 1]) >= 13.8
    assert -math.log(out_m[1, 2]) < 6.0
    assert np.allclose(band_m.sum(1), 1) and np.allclose(out_m.sum(1), 1)


def test_short_contrary_burst_in_band_does_not_flip_but_half_a_second_does():
    n = 120
    band = np.ones(n, bool)
    gate = np.ones(n, bool)
    L = np.full(n, 4.0)
    L[60:63] = -4.0                                          # 0.1 s of contrary evidence while entangled
    d = hmm.decide_sign(L, gate, band, CFG)
    assert not (d["s"] == -1).any() and (d["s"][:55] == 1).all()
    L2 = np.full(n, 4.0)
    L2[60:] = -4.0                                           # >= 0.5 s of strong contrary evidence
    d2 = hmm.decide_sign(L2, gate, band, CFG)
    assert (d2["s"][95:] == -1).all()


def test_prior_alone_never_decides_at_half_a_nat_and_does_at_1p5_only_where_blind():
    n = 150
    fcls = np.full(n, "face")
    gate = np.ones(n, bool)
    band = np.zeros(n, bool)
    Lp = decide.r4_prior(fcls, np.zeros(n, bool), CFG)
    assert np.abs(Lp).max() <= CFG.prior_nats + 1e-12
    d = hmm.four_state_r4(Lp, gate, band, CFG)
    assert (d["cls"] == 0).all()
    blind = np.zeros(n, bool)
    blind[50:] = True
    Lp2 = decide.r4_prior(fcls, blind, CFG)
    assert np.abs(Lp2[50:]).max() <= CFG.blind_prior_nats + 1e-12
    d2 = hmm.four_state_r4(Lp2, gate, band, CFG)
    assert (d2["cls"][:45] == 0).all()
    assert (d2["cls"][80:] == 3).all()                      # camera-blind: "usually on her upper-left back" decides (D1 (b))
    # decision (a): blind_prior_nats = prior_nats -> never decides
    from dataclasses import replace
    c_a = replace(CFG, blind_prior_nats=CFG.prior_nats)
    d3 = hmm.four_state_r4(decide.r4_prior(fcls, blind, c_a), gate, band, c_a)
    assert (d3["cls"] == 0).all()


def test_min_run_and_schmitt_exit():
    lab = np.zeros(60, int)
    lab[10:15] = 1                                           # 5 frames < the 0.5-s minimum run
    lab[30:50] = 2
    out = hmm.min_runs(lab, 15)
    assert (out[10:15] == 0).all() and (out[30:50] == 2).all()
    path = np.full(10, 1)
    post = np.zeros((10, 3))
    post[:, 1] = [0.5, 0.95, 0.85, 0.75, 0.72, 0.65, 0.95, 0.8, 0.69, 0.9]
    st = hmm.schmitt(path, post, 0.9, 0.7)
    assert st.tolist() == [0, 1, 1, 1, 1, 0, 1, 1, 0, 1]


def test_mnar_sign():
    det = np.array([[False, False]])
    vis_pred = np.array([[True, False]])
    l = missing_term(det, vis_pred, np.array([0.9, 0.9]), np.array([0.2, 0.2]))
    # no detection where the hypothesis says visible is evidence against it
    assert l[0, 0] < l[0, 1]
    l2 = missing_term(~det, vis_pred, np.array([0.9, 0.9]), np.array([0.2, 0.2]))
    assert l2[0, 0] > l2[0, 1]


def test_view_reliability_recovers_a_planted_q():
    rng = np.random.default_rng(3)
    n = 4000
    lab = rng.choice([-1, 1], n)
    qs = np.array([0.9, 0.75, 0.5, 0.3])
    votes = np.stack([np.where(rng.random(n) < q, lab, -lab) for q in qs], 1)
    votes[rng.random((n, 4)) < 0.5] = 0                     # views often see no crossing
    q, cnt = cue_reliability(votes, lab)
    assert np.allclose(q, qs, atol=0.03)
    w, _, _ = decide.vote_weights(votes, lab, CFG)
    assert w[3] == 0 and w[2] == 0 and w[0] > w[1] > 0       # q < 0.55 -> no weight


def _cam(C, fwd, down=(0.0, -1.0, 0.0)):
    z = np.asarray(fwd, float) / np.linalg.norm(fwd)
    y = np.asarray(down, float)
    x = np.cross(y, z)
    R = np.stack([x, y, z])
    return R, -R @ np.asarray(C, float)


def test_r4_generative_test_picks_flank_when_seen_from_front_and_behind():
    n = 3
    K = np.array([[1000.0, 0, 960], [0, 1000, 540], [0, 0, 1]])
    cams = ["F", "B"]
    obs = {}
    for c, C, fwd in (("F", (0, 0, 3.0), (0, 0, -1.0)), ("B", (0, 0, -3.0), (0, 0, 1.0))):
        R, t = _cam(C, fwd)
        assert abs(np.linalg.det(R) - 1) < 1e-9
        obs[c] = {"K": np.repeat(K[None], n, 0), "R": np.repeat(R[None], n, 0), "t": np.repeat(t[None], n, 0), "ok": np.ones(n, bool),
                  "C": np.repeat(np.asarray(C, float)[None], n, 0), "f": np.ones(n)}
    # her torso: a vertical capsule r = 11 cm on the y axis; she faces +z, her left is +x
    OA = np.repeat(np.array([[[0.0, -0.3, 0.0]]]), n, 0)
    OB = np.repeat(np.array([[[0.0, 0.3, 0.0]]]), n, 0)
    occ = (OA, OB, np.array([0.11]))
    sites = {"chest": {"joint": 9, "p": [0.0, 0.0, 0.12], "n": [0.0, 0.0, 1.0]},
             "side_L": {"joint": 9, "p": [0.15, 0.0, 0.0], "n": [1.0, 0.0, 0.0]},
             "upper_back": {"joint": 9, "p": [0.0, 0.0, -0.12], "n": [0.0, 0.0, -1.0]}}
    Jf = np.zeros((n, 55, 3))
    Rf = np.repeat(np.eye(3)[None, None], 55, 1).repeat(n, 0)
    palm = np.repeat(np.array([[0.17, 0.0, 0.01]]), n, 0)
    elbow = np.repeat(np.array([[0.40, 0.0, 0.05]]), n, 0)
    # what the views observe: his hand and wrist at her flank, seen by both
    flank_palm = np.array([0.165, 0.0, 0.0])
    wr = flank_palm - 0.08 * (flank_palm - elbow[0]) / np.linalg.norm(flank_palm - elbow[0])
    views = {}
    for c in cams:
        o = obs[c]
        h = K @ (o["R"][0] @ wr + o["t"][0])
        uv = np.zeros((n, 2, 17, 2), np.float32)
        uv[:, 0, 10] = h[:2] / h[2]
        conf = np.zeros((n, 2, 17), np.float32)
        conf[:, 0, 10] = 0.9
        pv = np.zeros((n, 2, 6), np.float32)
        pv[:, 0, 2] = 0.9
        views[c] = {"uv": uv, "conf": conf, "det": np.full((n, 2), 0.9, np.float32), "part_vis": pv}
    rates = {c: {"hand_vis": 0.9, "hand_hid": 0.1, "wr_vis": 0.9, "wr_hid": 0.3} for c in cams}
    vt = decide.r4_visibility_test(np.arange(n), palm, elbow, sites, Jf, Rf, occ, obs, views, cams, rates, CFG)
    l = vt["l"]
    assert (np.argmax(l, 1) == 1).all()                      # flank
    # the front palm is hidden from the back camera, the back palm from the front camera, the flank palm from neither
    assert vt["vis"][0, 0].tolist() == [True, False] and vt["vis"][0, 2].tolist() == [False, True] and vt["vis"][0, 1].all()
    # seen only from the front: front and flank both fit; from behind only: back and flank
    views_f = {"F": views["F"], "B": {**views["B"], "conf": np.zeros_like(views["B"]["conf"]), "part_vis": np.zeros_like(views["B"]["part_vis"])}}
    l2 = decide.r4_visibility_test(np.arange(n), palm, elbow, sites, Jf, Rf, occ, obs, views_f, cams, rates, CFG)["l"]
    assert l2[0, 0] > l2[0, 2]


def test_yaw_viterbi_recovers_2p6_turns_from_noisy_half_circle_bits():
    rng = np.random.default_rng(7)
    hz = 30.0
    n = 120
    psi = np.linspace(0.0, 2.6 * 360.0, n)                 # 2.6 turns in 4 s (234 deg/s)
    az = np.arange(0, 360, 60.0)
    B = CFG.yaw_bins
    cb = np.arange(B) * 360.0 / B
    LL = np.zeros((n, B))
    for a in az:
        bit = np.where(np.abs(geom.wrap_deg(psi - a)) < 90, 1, -1)
        flip = rng.random(n) < 0.1
        bit = np.where(flip, -bit, bit)
        LL += decide._halfplane_log(geom.wrap_deg(cb[None] - a), bit, 0.85, CFG.yaw_soft_deg)
    y = hmm.yaw_viterbi(LL, np.zeros(n), CFG, hz, psi0=0.0)
    turns = (y["psi"][-1] - y["psi"][0]) / 360.0
    assert abs(turns - 2.6) < 0.1
    assert np.percentile(np.abs(geom.wrap_deg(y["psi"] - psi)), 90) < 35


def test_crop_terms_contract(tmp_path):
    import json
    p = tmp_path / "crops.json"
    p.write_text(json.dumps({"items": [
        {"t": 1.0, "cam": "00", "relation": "R4", "p": {"front": 0.05, "flank": 0.15, "back": 0.8}, "score": 0.9, "source": "jev"},
        {"t": 2.0, "cam": "01", "relation": "R9_RL", "p": {"+": 0.9}, "score": 1.0, "source": "agent"},
        {"t": 3.0, "cam": "01", "relation": "R9_RL", "p": {"+": 0.1}, "score": 0.5, "source": "jev"}]}))   # below crop_min_score
    t = np.arange(0, 4.0, 1 / 30)
    ct = decide.crop_terms(decide.load_crops(p), t, CFG)
    k1, k2, k3 = 30, 60, 90
    assert np.argmax(ct["R4"][k1]) == 2 and abs(ct["R4"][k1].sum()) < 1e-9          # centred, back wins
    assert ct["R4"][k1 + 20].max() == 0                                              # beyond +-0.5 s: no term
    assert ct["R9_RL"][k2] > 2.0 and ct["R9_RL"][k3] == 0                           # the low-score item is ignored
    assert np.abs(ct["R9_RL"]).max() <= CFG.clip_crop
    assert decide.load_crops(tmp_path / "missing.json") == []


def test_crop_look_holds_along_its_band_run_only():
    """a look at t0 inside an entangled run holds (full weight 1 s, fading to 0 at 2.5 s) and never leaks outside the run"""
    t = np.arange(300) / 30.0
    band = np.zeros(300, bool)
    band[60:240] = True                                    # 2.0-8.0 s
    w = decide.hold_weight(t, 5.0, band, CFG)
    assert w[150] == 1.0 and w[int(5.9 * 30)] == 1.0
    assert 0 < w[int(6.8 * 30)] < 1 and w[int(7.6 * 30)] == 0.0
    assert w[:60].max() == 0 and w[240:].max() == 0
    assert decide.hold_weight(t, 9.0, band, CFG).max() == 0.0          # a look outside the band: no hold
    items = [{"t": 5.0, "relation": "R9_RL", "p": {"+": 0.05}, "score": 1.0}]
    ct = decide.crop_terms(items, t, CFG, {"R9_RL": band})
    assert ct["R9_RL"][int(4.5 * 30)] < -2.5 and ct["R9_RL"][int(1.5 * 30)] == 0.0


def test_r4_cue_calibration_neutralises_a_cue_the_looks_contradict():
    """a cue that says "front" on every frame the looks label "flank": its calibrated front vote no longer favours front over flank"""
    T = 100
    L = np.tile(np.array([1.0, -0.5, -0.5]), (T, 1)) * 2.0           # always front, decisively
    labels = [(k, 1) for k in range(0, 100, 3)]                       # truth: flank (class index 1)
    rep = decide.r4_cue_reliability(labels, {"l3d": L}, CFG)["l3d"]
    assert rep["acc"] == 0.0 and rep["n"] == len(labels)
    Lc = decide.calibrated_cue(L, rep, CFG)
    assert Lc[0, 1] >= Lc[0, 0] - 1e-9                                # flank >= front now
    # a cue without enough labels is left as it is
    rep2 = decide.r4_cue_reliability(labels[:3], {"l3d": L}, CFG)["l3d"]
    assert np.array_equal(decide.calibrated_cue(L, rep2, CFG), L)
