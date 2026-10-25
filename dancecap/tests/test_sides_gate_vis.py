"""dancecap.sides.gate, build C: the visibility-consistent gate for a DECIDED R4 class (owner rule 2026-10-24 + D1). On a decided frame
where the candidate's palm is on the decided class, a view where the candidate predicts his right hand / distal forearm hidden (behind
her back, under her arm) contributes no hand / wrist / distal-forearm detection to G1 / G10 and no triangulated wrist to G2: those
detections are hallucinations of a hidden hand. The rest of the arm and the body stay counted; off-class and undecided frames keep
every detection. The synthetic couple of test_sides_gate (two SMPL-X bodies face to face, four cameras)."""
import dataclasses

import numpy as np
import pytest

from test_sides_gate import (ARM, MODEL, T, _look_at, _observed_from, _shoulder, _win, _with_pose,  # noqa: F401
                             scene)

pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="no SMPL-X model")
SKIP = ("G3", "G4", "G5", "G6", "G7", "G8", "G9", "trust")


def _two_cams():
    """a camera in front of her / behind him (F, z = -4) and one behind her back (B, z = +5), at her chest height (the synthetic bodies'
    pelvis is at y = -0.35, her spine3 at y = -0.06)"""
    K = np.array([[1000.0, 0, 960], [0, 1000.0, 540], [0, 0, 1]])
    obs = {}
    for c, C in (("F", np.array([0.0, 0.2, -4.0])), ("B", np.array([0.0, 0.2, 5.0]))):
        R, t = _look_at(C, (0, -0.1, 0.5))
        obs[c] = {"K": np.repeat(K[None], T, 0), "R": np.repeat(R[None], T, 0), "t": np.repeat(t[None], T, 0), "ok": np.ones(T, bool),
                  "C": np.repeat(C[None], T, 0), "f": np.ones(T)}
    return obs


def _hand_behind_her(scene):
    """joints-only lead fit: his right wrist (and the hand joints with it) behind her back at her spine3 height, his elbow at her side"""
    f = dict(scene.base["lead"])
    J = np.array(f["joints"], copy=True)
    off = np.array([-0.05, -0.05, 1.12]) - J[:, 21]
    J[:, [21, *range(40, 55)]] += off[:, None]
    J[:, 19] = [-0.27, -0.02, 0.88]
    f["joints"] = J
    return {"lead": f, "follow": scene.base["follow"]}


def test_r_hand_hidden_two_body_ray_test(scene):
    """his hand behind her back: hidden from the camera in front of her (her torso blocks the ray), seen from the one behind her; a
    camera without the frame is never 'hidden' (it has no detection to drop)"""
    import copy
    from dancecap.sides.gate import r_hand_hidden
    c2 = copy.copy(scene)
    c2.obs, c2.cams = _two_cams(), ["F", "B"]
    fits = _hand_behind_her(scene)
    h = r_hand_hidden(c2, fits, np.arange(T))
    assert h["hand"][:, 0].all() and not h["hand"][:, 1].any()
    hb = r_hand_hidden(c2, scene.base, np.arange(T))              # the base's hand between them: her body hides it from B
    assert hb["hand"][:, 1].all()
    c2.obs["F"] = dict(c2.obs["F"], ok=np.zeros(T, bool))
    assert not r_hand_hidden(c2, fits, np.arange(T))["hand"][:, 0].any()


def test_r4_proxy_hidden_places_the_decided_class_on_her_surface(scene):
    """the solver's rule (solve_r4_vis): a decided 'back' frame hides the hand from the camera in front of her, not from the one behind
    her; undecided frames hide nothing"""
    from dancecap import zouk_contacts as zc
    from dancecap.sides.config import SidesCfg
    from dancecap.sides.gate import r4_proxy_hidden
    sites = zc.fit_sites(np.zeros(16))
    cls = np.zeros(T, np.int8)
    cls[10:30] = 3
    h = r4_proxy_hidden(scene.base, scene.caps, sites, _two_cams(), ["F", "B"], SidesCfg(), np.arange(T), cls)
    assert h["hand"][10:30, 0].all() and not h["hand"][:, 1].any()
    assert not h["hand"][:10].any() and not h["hand"][30:].any()


def _hallucinated(scene, truth, hidden_cams, base):
    """the observed scene of `truth`, except that in `hidden_cams` the detector puts his right hand / distal forearm (MAMMA) and his
    COCO right wrist where the BASE has them (a hallucinated hidden hand); the triangulated wrist follows the base too (all inliers)"""
    from dancecap.sides import body as bd
    from dancecap.sides.gate import r_arm_groups
    obs_ctx = _observed_from(scene, truth)
    o = bd.forward(truth["lead"])
    hand, fore = r_arm_groups(scene.Bw["lead"], o["pts"], o["J"])
    pb = bd.forward(base["lead"])["pts"]
    Jb = np.asarray(base["lead"]["joints"])
    for c in hidden_cams:
        ob = scene.obs[c]
        R, t = ob["R"][0], ob["t"][0]
        Xc = np.einsum("ij,tpj->tpi", R, pb) + t
        L = obs_ctx._lm[c].copy()
        L[:, 0, hand | fore] = ((Xc[..., :2] / Xc[..., 2:]) * 1000 + np.array([960, 540]))[:, hand | fore]
        obs_ctx._lm[c] = L
        uv = obs_ctx.views[c]["uv"].copy()
        Xw = (R @ Jb[:, 21].T).T + t
        uv[:, 0, 10] = (Xw[:, :2] / Xw[:, 2:]) * 1000 + np.array([960, 540])
        obs_ctx.views[c] = dict(obs_ctx.views[c], uv=uv)
    tr = dict(obs_ctx.tri["lead"])
    X = np.array(tr["Xd"], copy=True)
    X[:, 10] = Jb[:, 21]
    tr.update(Xd=X, X=X, inl=np.ones((T, 17, len(scene.cams)), bool))
    obs_ctx.tri = dict(obs_ctx.tri, lead=tr)
    obs_ctx.tri_ho = obs_ctx.tri
    return obs_ctx


def _patch(monkeypatch, G, scene, cls_value):
    hid = np.zeros((T, len(scene.cams)), bool)
    hid[:, :2] = True                                            # phones 00 / 01 cannot see the hand where the candidate puts it
    monkeypatch.setattr(G, "r_hand_hidden", lambda ctx, fits, ks: {"hand": hid[ks], "forearm": hid[ks]})
    monkeypatch.setattr(G, "r4_classes", lambda ctx, fits, ks: np.full(len(ks), cls_value, np.int8))


def test_r_arm_groups_split_hand_and_distal_forearm(scene):
    from dancecap.sides import body as bd
    from dancecap.sides.gate import r_arm_groups
    o = bd.forward(scene.base["lead"])
    hand, fore = r_arm_groups(scene.Bw["lead"], o["pts"], o["J"])
    dj = np.argmax(scene.Bw["lead"], 1)
    assert hand.sum() > 5 and fore.sum() > 3 and not (hand & fore).any()
    assert set(dj[fore].tolist()) == {19} and (dj == 19).sum() > fore.sum()       # the proximal half of the forearm stays counted


def test_vis_r4_drops_hidden_hand_detections_on_decided_on_class_frames(scene, monkeypatch):
    """the candidate (= the truth) has his hand where the decided class puts it, hidden from phones 00 / 01, whose detector hallucinates
    the hand where the base has it. The old gate rejects the truth (G1 / G2 / G10); the visibility-consistent gate drops exactly those
    detections, accepts it and records the dropped view-frames; the same candidate OFF the decided class, or on undecided frames, gets
    no exemption"""
    from dancecap.sides import gate as G
    from dancecap.sides.config import SidesCfg
    truth = _with_pose(scene, "lead", _shoulder(14.0, (1, 0, 0)), np.arange(4, 36))
    obs_ctx = _hallucinated(scene, truth, ("00", "01"), scene.base)
    rel = dict(scene.rel, R4__cls=np.full(T, 3, np.int8))
    obs_ctx.rel = rel
    _patch(monkeypatch, G, scene, 3)
    win = _win({"lead": ARM})
    obs_ctx.cfg = dataclasses.replace(SidesCfg(), gate_vis_r4=False)
    old = G.window_gates(obs_ctx, truth, win, skip=SKIP)
    assert {"G1", "G2", "G10"} <= set(old["failed"]), old["failed"]
    assert "vis_r4" not in old["gates"]["G1"]
    obs_ctx.cfg = SidesCfg()
    assert obs_ctx.cfg.gate_vis_r4
    new = G.window_gates(obs_ctx, truth, win, skip=SKIP)
    assert new["pass"], (new["failed"], new["gates"]["G1"]["failed_chains"])
    v = new["gates"]["G1"]["vis_r4"]
    n = win["r1"] - win["r0"] + 1
    assert v["decided_frames"] == n and v["cand_on_class_frames"] == n
    assert v["hand_hidden_view_frames"]["cand"] == 2 * n
    assert v["dropped_detections"]["mamma_view_frames"] > 0 and set(v["dropped_detections"]["per_cam"]) <= {"00", "01"}
    assert new["gates"]["G2"]["vis_r4_wrist_dropped"] > 0
    # off the decided class: no exemption, the hallucinated detections count again
    monkeypatch.setattr(G, "r4_classes", lambda ctx, fits, ks: np.full(len(ks), 1, np.int8))
    off = G.window_gates(obs_ctx, truth, win, skip=SKIP)
    assert "G1" in off["failed"] and off["gates"]["G1"]["vis_r4"]["cand_on_class_frames"] == 0
    # undecided frames: the old gate
    obs_ctx.rel = dict(rel, R4__cls=np.zeros(T, np.int8))
    und = G.window_gates(obs_ctx, truth, win, skip=SKIP)
    assert "G1" in und["failed"] and "vis_r4" not in und["gates"]["G1"]


def test_vis_r4_keeps_the_rest_of_the_arm_counted(scene, monkeypatch):
    """the exemption covers his hand / wrist / distal forearm only: a candidate whose arm swings the other way (elbow, upper arm off
    where every phone sees them) still fails G1 and G2 on a decided, 'on-class' window"""
    from dancecap.sides import gate as G
    from dancecap.sides.config import SidesCfg
    truth = _with_pose(scene, "lead", _shoulder(14.0, (1, 0, 0)), np.arange(4, 36))
    obs_ctx = _hallucinated(scene, truth, ("00", "01"), scene.base)
    obs_ctx.rel = dict(scene.rel, R4__cls=np.full(T, 3, np.int8))
    obs_ctx.cfg = SidesCfg()
    _patch(monkeypatch, G, scene, 3)
    wrong = _with_pose(scene, "lead", _shoulder(-25.0, (1, 0, 0)), np.arange(4, 36))
    r = G.window_gates(obs_ctx, wrong, _win({"lead": ARM}), skip=SKIP)
    assert "G1" in r["failed"] and "G2" in r["failed"], r["failed"]


def test_solver_r4_weights_default_to_the_old_objective():
    """the build-C solver knobs are off by default (w_r4_mult 1, r4_keep_scale off, solve_r4_vis off): the old ladder is unchanged"""
    from dancecap.sides.config import SidesCfg
    c = SidesCfg()
    assert c.w_r4_mult == 1.0 and not c.r4_keep_scale and not c.solve_r4_vis
    assert all("r4" not in a for a in c.solve_attempts)
