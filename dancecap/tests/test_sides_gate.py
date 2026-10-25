"""dancecap.sides.gate on a synthetic couple (two SMPL-X bodies, four cameras; DESIGN 8): each of G1-G10 and the trust gate fires on a
crafted failure, a no-op passes, a failing window is reverted by the merge (frames byte-identical), overlapping windows merge
continuously, and the output is deterministic."""
import json
import math

import numpy as np
import pytest

from dancecap.take import WORKSPACE

MODEL = WORKSPACE / "mamma" / "data" / "body_models" / "smplx_locked_head" / "smplx" / "SMPLX_NEUTRAL.npz"
pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="no SMPL-X model")

T = 40
HZ = 30.0


def _fit(transl, yaw_deg: float, arm_forward: bool = False):
    from dancecap.sides import body as bd
    fit = {"betas": np.zeros(16), "t": np.arange(T) / HZ, "transl": np.tile(np.asarray(transl, float), (T, 1)),
           "model": np.array("smplx_locked_head"), "gender": np.array("neutral"), "stance": np.ones((T, 2), bool)}
    for k, (a, b) in bd.POSE_SLICES.items():
        fit[k] = np.zeros((T, (b - a) * 3))
    fit["global_orient"][:] = np.array([0.0, math.radians(yaw_deg), 0.0])
    bp = fit["body_pose"].reshape(T, 21, 3)
    bp[:, 15] = [0, 0, -1.2]            # left shoulder (16): arm down
    bp[:, 16] = [0, 0, 1.2]             # right shoulder (17): arm down
    if arm_forward:
        bp[:, 16] = [0, 1.45, 0.0]      # right arm forward (+z)
        bp[:, 18] = [0, 0.3, 0]         # right elbow slightly bent (flexion > 0)
    fit["body_pose"] = bp.reshape(T, 63)
    fit["joints"] = np.zeros((T, 55, 3))
    fit["joints"][:] = bd.forward(fit, points=False)["J"]
    return fit


def _look_at(C, target):
    z = np.asarray(target, float) - C
    z /= np.linalg.norm(z)
    x = np.cross(np.array([0.0, -1.0, 0.0]), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return R, -R @ C


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    from dancecap.sides import body as bd
    from dancecap.sides import geom
    from dancecap.sides.config import SidesCfg
    from dancecap.sides.gate import GateCtx
    from dancecap import zouk_contacts as zc
    lead = _fit([0.0, 0.0, 0.0], 0.0, arm_forward=True)
    follow = _fit([0.0, 0.0, 0.85], 180.0)
    cams = ["00", "01", "02", "03"]
    K = np.array([[1000.0, 0, 960], [0, 1000.0, 540], [0, 0, 1]])
    obs, views, lm = {}, {}, {}
    pts = {r: bd.forward(f)["pts"] for r, f in (("lead", lead), ("follow", follow))}
    for i, c in enumerate(cams):
        a = math.radians(45 + 90 * i)
        C = np.array([4 * math.sin(a), 1.4, 0.4 + 4 * math.cos(a)])
        R, t = _look_at(C, (0, 1.0, 0.4))
        o = {"K": np.repeat(K[None], T, 0), "R": np.repeat(R[None], T, 0), "t": np.repeat(t[None], T, 0), "ok": np.ones(T, bool),
             "C": np.repeat(C[None], T, 0), "f": np.ones(T), "vis": np.ones((T, 2, 512), np.float16), "size": np.array([1920, 1080])}
        obs[c] = o
        L = np.zeros((T, 2, 512, 2), np.float32)
        uv = np.zeros((T, 2, 17, 2), np.float32)
        for p, (r, f) in enumerate((("lead", lead), ("follow", follow))):
            Xc = np.einsum("ij,tpj->tpi", R, pts[r]) + t
            L[:, p] = (Xc[..., :2] / Xc[..., 2:]) * 1000 + np.array([960, 540])
            Xj = np.asarray(f["joints"])[:, list(geom.C2S)]
            Xc = np.einsum("ij,tpj->tpi", R, Xj) + t
            uv[:, p] = (Xc[..., :2] / Xc[..., 2:]) * 1000 + np.array([960, 540])
        lm[c] = L
        views[c] = {"uv": uv, "conf": np.ones((T, 2, 17), np.float16), "det": np.ones((T, 2), np.float16), "face": np.zeros((T, 2), np.int8),
                    "vis_pred": np.ones((T, 2, 17), bool), "ok": np.ones(T, bool)}
    tri = {}
    for r, f in (("lead", lead), ("follow", follow)):
        X = np.asarray(f["joints"])[:, list(geom.C2S)].astype(np.float32)
        tri[r] = {"Xd": X, "X": X, "nv": np.full((T, 17), 4, np.int8), "S": np.tile(np.eye(3) * 0.02 ** 2, (T, 17, 1, 1)).astype(np.float32),
                  "ho": np.zeros(T, np.int8)}
    rel = {}
    for rid in geom.REL_TABLE:
        rel[f"{rid}__s"] = np.zeros(T, np.int8)
        rel[f"{rid}__post"] = np.zeros((T, 3), np.float32)
    rel["R4__cls"] = np.zeros(T, np.int8)
    ctx = object.__new__(GateCtx)
    ctx.take, ctx.cfg = "synthetic", SidesCfg()
    ctx.T, ctx.hz, ctx.cams = T, HZ, cams
    ctx.input = {"lead": lead, "follow": follow}
    ctx.base = {"lead": lead, "follow": follow}
    ctx.tri, ctx.tri_ho, ctx.views, ctx.obs, ctx.rel = tri, tri, views, obs, rel
    ctx.yaw = {f"{r}__psi_star": geom.unwrap_deg(geom.line_heading(f["joints"][:, 1], f["joints"][:, 2])) for r, f in ctx.input.items()}
    ctx.yaw.update({f"{r}__conf": np.ones(T) for r in ctx.input})
    ctx.bias = {r: np.zeros((17, 3)) for r in ctx.input}
    ctx.caps = {r: zc.fit_capsules(np.zeros(16)) for r in ctx.input}
    ctx.rest = {r: bd.model(lead)["j_rest"] for r in ctx.input}
    ctx.parents = bd.model(lead)["parents"]
    ctx.Bw = {r: bd.model(lead)["Bw"] for r in ctx.input}
    ctx._lm = lm
    ctx._speed_ref = None
    ctx._hw = 0.14
    ctx.exact_cache = {}
    ctx.sd = tmp_path_factory.mktemp("sides")
    return ctx


def _win(dancers, k0=8, k1=31, kind="arm", reasons=("inside",)):
    from dancecap.sides.plan import trust_of
    from dancecap.sides.config import SidesCfg
    cfg = SidesCfg()
    return {"id": 0, "k0": k0, "k1": k1, "r0": k0 - 4, "r1": k1 + 4, "ramp_in": 4, "ramp_out": 4, "t0": k0 / HZ, "t1": k1 / HZ,
            "reasons": list(reasons), "dancers": {r: {"joints": js, "kind": kind, "reasons": list(reasons), "real_items": [],
                                                      "trust_deg": {str(j): trust_of(j, kind, cfg) for j in js}, "root_m": 0.03 if kind == "spin" else 0.0}
                                                  for r, js in dancers.items()}}


def _with_pose(ctx, role, fn, ks=None):
    from dancecap.sides import body as bd
    from dancecap.sides.gate import fit_window
    ks = np.arange(T) if ks is None else ks
    P = bd.pose55(ctx.base[role], ks)
    P = fn(P)
    out = dict(ctx.base)
    out[role] = fit_window(ctx.base[role], P, None, ks)
    return out


ARM = [9, 14, 17, 19, 21]


def test_noop_passes_and_is_deterministic(scene):
    from dancecap.sides.gate import window_gates
    w = _win({"lead": ARM})
    ex = {"base": None, "cand": None}
    r1 = window_gates(scene, scene.base, w, skip=("G9",))
    r2 = window_gates(scene, scene.base, w, skip=("G9",))
    assert r1["pass"], r1["failed"]
    a = json.dumps({k: v for k, v in r1.items() if k != "seconds"}, sort_keys=True, default=str)
    b = json.dumps({k: v for k, v in r2.items() if k != "seconds"}, sort_keys=True, default=str)
    assert a == b


def _shoulder(deg, axis=(0, 0, 1)):
    from scipy.spatial.transform import Rotation as Rot
    def fn(P):
        R = Rot.from_rotvec(np.asarray(axis, float) * math.radians(deg)) * Rot.from_rotvec(P[:, 17])
        P = P.copy()
        P[:, 17] = R.as_rotvec()
        return P
    return fn


def test_g1_g2_g10_and_trust_fire_on_a_moved_arm(scene):
    from dancecap.sides.gate import window_gates
    cand = _with_pose(scene, "lead", _shoulder(30.0, (1, 0, 0)), np.arange(4, 36))
    r = window_gates(scene, cand, _win({"lead": ARM}), skip=("G9",))
    for g in ("G1", "G2", "G10", "trust"):
        assert g in r["failed"], (g, r["failed"])


def test_g3_fires_on_a_newly_violated_decided_relation(scene):
    from dancecap.sides.gate import g3
    scene.rel["R1_lead__s"][:] = 1                   # decided: his torso toward her (the base is)
    try:
        cand = dict(scene.base)
        f = dict(scene.base["lead"])
        f["joints"] = np.array(f["joints"], copy=True)
        c = f["joints"][:, :, [0, 2]].mean(1, keepdims=True)
        f["joints"][10:30, :, [0, 2]] = 2 * c[10:30] - f["joints"][10:30, :, [0, 2]]     # turned 180 deg about his vertical
        cand["lead"] = f
        r = g3(scene, _win({"lead": [0]}), {"base": scene.base, "cand": cand}, np.arange(4, 36))
        assert not r["ok"] and "R1_lead" in r["newly_violated"]
    finally:
        scene.rel["R1_lead__s"][:] = 0


def _crossing_fits(scene, through_frames=None):
    """joints-only fits whose right / left forearms cross in their interiors 2 cm apart; through_frames pushed through"""
    base = {r: dict(f) for r, f in scene.base.items()}
    JL = np.array(base["lead"]["joints"], copy=True)
    JF = np.array(base["follow"]["joints"], copy=True)
    for k in range(T):
        JL[k, [17, 19, 21]] = [[-0.4, 1.31, 0.45], [-0.12, 1.31, 0.42], [0.2, 1.31, 0.37]]
        JF[k, [16, 18, 20]] = [[0.05, 1.29, 0.0], [0.02, 1.29, 0.28], [-0.03, 1.29, 0.6]]
    base["lead"]["joints"], base["follow"]["joints"] = JL, JF
    cand = {r: dict(f) for r, f in base.items()}
    J2 = JL.copy()
    if through_frames is not None:
        J2[through_frames][:, [17, 19, 21], 1] -= 0.04
        J2[through_frames, 17, 1] -= 0.04
        J2[through_frames, 19, 1] -= 0.04
        J2[through_frames, 21, 1] -= 0.04
    cand["lead"] = dict(cand["lead"])
    cand["lead"]["joints"] = J2
    return base, cand


def test_g4_fires_on_a_new_passthrough(scene):
    from dancecap.sides import geom
    from dancecap.sides.gate import g4, gli_series
    base, cand = _crossing_fits(scene, np.arange(15, 25))
    ks = np.arange(T)
    assert len(geom.passthrough_events(gli_series(base, ks)["R9_RL"])) == 0
    r = g4(scene, _win({"lead": ARM}), {"base": base, "cand": cand}, ks)
    assert not r["ok"] and r["R9_RL"]["cand"] >= 2


def test_g5_fires_on_a_planted_swing(scene):
    from dancecap.sides.gate import g5
    cand = {r: dict(f) for r, f in scene.base.items()}
    J = np.array(scene.base["lead"]["joints"], copy=True)
    tt = np.arange(T) / HZ
    sw = 0.04 * np.sin(2 * np.pi * 2.0 * tt)
    J[:, 19, 1] += sw
    J[:, 21, 1] += 2 * sw
    cand["lead"] = dict(cand["lead"])
    cand["lead"]["joints"] = J
    ks = np.arange(T)
    r = g5(scene, _win({"lead": ARM}), {"base": scene.base, "cand": cand}, ks, {"lead": ARM}, np.arange(4, 36))
    assert not r["ok"]


def test_g6_fires_on_a_new_elbow_limit(scene):
    from scipy.spatial.transform import Rotation as Rot
    from dancecap.sides.gate import g6

    def fn(P):
        P = P.copy()
        P[10:30, 19] = (Rot.from_rotvec([0, 0, math.radians(70)]) * Rot.from_rotvec(P[10:30, 19])).as_rotvec()   # off the hinge
        return P
    cand = _with_pose(scene, "lead", fn)
    r = g6(scene, _win({"lead": ARM}), {"base": scene.base, "cand": cand}, np.arange(T), {"lead": ARM})
    assert not r["ok"] and r["lead"]["new_frames"]


def test_g7_fires_on_a_jump(scene):
    from dancecap.sides.gate import g7
    cand = {r: dict(f) for r, f in scene.base.items()}
    J = np.array(scene.base["lead"]["joints"], copy=True)
    J[20, 21] += [0.15, 0, 0]
    J[20, 19] += [0.08, 0, 0]
    cand["lead"] = dict(cand["lead"])
    cand["lead"]["joints"] = J
    r = g7(scene, _win({"lead": ARM}), {"base": scene.base, "cand": cand}, np.arange(T), {"lead": ARM})
    assert not r["ok"]


def test_g8_fires_on_a_yaw_off_the_camera_track(scene):
    from dancecap.sides.gate import g8
    cand = {r: dict(f) for r, f in scene.base.items()}
    J = np.array(scene.base["follow"]["joints"], copy=True)
    c = J[:, 0:1].copy()
    a = math.radians(100.0)
    Rz = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    J[10:30] = np.einsum("ij,tpj->tpi", Rz, J[10:30] - c[10:30]) + c[10:30]
    cand["follow"] = dict(cand["follow"])
    cand["follow"]["joints"] = J
    r = g8(scene, _win({"follow": [0]}, kind="spin", reasons=("spin",)), {"base": scene.base, "cand": cand}, np.arange(T), ["follow"])
    assert not r["ok"]


def test_g9_fires_when_his_hand_goes_into_her(scene):
    from dancecap.sides.gate import g9, fit_window
    from dancecap.sides import body as bd
    ks = np.arange(12, 20)
    cand = dict(scene.base)
    f = scene.base["follow"]
    tr = np.array(f["transl"], copy=True)[ks]
    tr[:, 2] -= 0.42                          # she steps into his forward arm
    cand["follow"] = fit_window(f, bd.pose55(f, ks), tr, ks)
    r = g9(scene, _win({"follow": [0]}), {"base": scene.base, "cand": cand}, ks)
    assert r["frames_gt3"][0] == 0 and r["frames_gt3"][1] > 0
    assert not r["ok"]


def test_merge_reverts_a_failed_window_and_blends_overlaps(scene):
    from dancecap.sides import body as bd
    from dancecap.sides.merge import merge_fits
    base = scene.base
    # nothing adopted: byte-identical
    m, src = merge_fits(base, [], {})
    for r in base:
        assert np.array_equal(bd.pose55(m[r]), bd.pose55(base[r]))
        assert (src[r] == 0).all()
    # two overlapping windows that moved his right shoulder differently: the merge is continuous (no frame step > the windows' own)
    w1 = _win({"lead": ARM}, 6, 20)
    w1["id"] = 1
    w2 = _win({"lead": ARM}, 16, 32)
    w2["id"] = 2
    P = bd.pose55(base["lead"])
    c = {}
    for w, deg in ((w1, 10.0), (w2, -10.0)):
        ks = np.arange(w["r0"], w["r1"] + 1)
        from dancecap.sides.merge import ramp_weights
        ww = ramp_weights(w)
        Pk = P[ks].copy()
        from scipy.spatial.transform import Rotation as Rot
        Pk[:, 17] = (Rot.from_rotvec(np.array([1.0, 0, 0])[None] * (math.radians(deg) * ww)[:, None]) * Rot.from_rotvec(Pk[:, 17])).as_rotvec()
        c[w["id"]] = {"lead": {"P": Pk, "transl": np.asarray(base["lead"]["transl"])[ks]}}
    m, src = merge_fits(base, [w1, w2], c)
    Pm = bd.pose55(m["lead"])
    step = np.abs(np.diff(Pm[:, 17], axis=0)).max()
    assert step < math.radians(4.0)
    assert np.array_equal(Pm[:2], P[:2]) and np.array_equal(Pm[-2:], P[-2:])
    assert (src["lead"][w1["k0"]:w2["k1"] + 1] == 2).all()


def test_merge_reverts_one_window_per_overlapping_cluster():
    """two neighbours that fail together after the merge lose only one of them (the one failing more gates; on a tie the non-owner
    window); a window failing alone is reverted"""
    from dancecap.sides.merge import cluster_drops
    spans = {38: (100, 260), 39: (230, 400), 16: (900, 950)}
    assert sorted(cluster_drops(spans, {38: 1, 39: 3, 16: 1}, {})) == [16, 39]
    assert cluster_drops({1: (0, 50), 2: (40, 90)}, {1: 2, 2: 2}, {1: 2, 2: 1}) == [1]
    assert cluster_drops({}, {}, {}) == []


def _observed_from(ctx, fits):
    """a copy of the synthetic scene whose MAMMA landmarks, COCO detections and triangulated joints are those of `fits` (the truth)"""
    import copy
    from dancecap.sides import body as bd
    from dancecap.sides import geom
    c2 = copy.copy(ctx)
    c2._lm, c2.views, c2.tri = {}, {}, {}
    pts = {r: bd.forward(f)["pts"] for r, f in fits.items()}
    for c in ctx.cams:
        o = ctx.obs[c]
        R, t = o["R"][0], o["t"][0]
        L = np.zeros_like(ctx._lm[c])
        uv = np.zeros_like(ctx.views[c]["uv"])
        for p, r in enumerate(("lead", "follow")):
            Xc = np.einsum("ij,tpj->tpi", R, pts[r]) + t
            L[:, p] = (Xc[..., :2] / Xc[..., 2:]) * 1000 + np.array([960, 540])
            Xj = np.asarray(fits[r]["joints"])[:, list(geom.C2S)]
            Xc = np.einsum("ij,tpj->tpi", R, Xj) + t
            uv[:, p] = (Xc[..., :2] / Xc[..., 2:]) * 1000 + np.array([960, 540])
        c2._lm[c] = L
        c2.views[c] = dict(ctx.views[c], uv=uv)
    for r, f in fits.items():
        X = np.asarray(f["joints"])[:, list(geom.C2S)].astype(np.float32)
        c2.tri[r] = dict(ctx.tri[r], Xd=X, X=X)
    c2.tri_ho = c2.tri
    return c2


def test_g1_g2_per_chain_catch_one_arm_hidden_by_the_pooled_median(scene):
    """fix round 1 (verify_r0 major 2): the truth bends his spine 10 deg; the candidate gets the torso, head and left arm right but
    swings his right arm 30 deg off. The pooled G1 / G2 medians over every moved point pass (most moved points improve); the per-chain
    G1 / G2 fail on his right arm"""
    import dataclasses
    from dancecap.sides.config import SidesCfg
    from dancecap.sides.gate import window_gates, point_chains, coco_chains, CHAINS
    from scipy.spatial.transform import Rotation as Rot

    def bend(P):
        P = P.copy()
        P[:, 6] = (Rot.from_rotvec([math.radians(10.0), 0, 0]) * Rot.from_rotvec(P[:, 6])).as_rotvec()
        return P
    truth = _with_pose(scene, "lead", bend)
    obs_ctx = _observed_from(scene, truth)
    class _Truth:
        base = truth
    cand = _with_pose(_Truth, "lead", _shoulder(30.0, (1, 0, 0)), np.arange(4, 36))
    win = _win({"lead": [6, 9, 14, 17, 19, 21]})
    skip = ("G3", "G4", "G5", "G6", "G7", "G8", "G9", "G10", "trust")
    obs_ctx.cfg = dataclasses.replace(SidesCfg(), gate_per_chain=False)
    pooled = window_gates(obs_ctx, cand, win, skip=skip)
    assert pooled["pass"], pooled["failed"]
    assert any(x.startswith("lead_arm_R:") for x in pooled["gates"]["G1"]["failed_chains"])     # reported, not enforced
    assert "lead_arm_R" in pooled["gates"]["G2"]["failed_chains"]
    obs_ctx.cfg = SidesCfg()
    assert obs_ctx.cfg.gate_per_chain
    chain = window_gates(obs_ctx, cand, win, skip=skip)
    assert "G1" in chain["failed"] and "G2" in chain["failed"], chain["failed"]
    assert [x.split(":")[0] for x in chain["gates"]["G1"]["failed_chains"]] == ["lead_arm_R"], chain["gates"]["G1"]["failed_chains"]
    assert len(chain["gates"]["G1"]["failed_chains"][0].split(":")[1].split(",")) >= SidesCfg().gate_chain_min_cams
    # the chain maps: the MAMMA points of each arm / leg exist, the COCO wrists belong to their arm
    pc = point_chains(scene.Bw["lead"])
    assert all((pc == k).sum() > 10 for k in range(len(CHAINS)))
    assert CHAINS[coco_chains()[10]] == "arm_R" and CHAINS[coco_chains()[9]] == "arm_L" and CHAINS[coco_chains()[0]] == "torso"


def test_chain_camera_verdict_needs_an_uncontradicted_median_rise():
    """a chain is worse in a camera only when a detector's median rises beyond max(2 px, 10 %) and the other detector's median does
    not fall beyond it; p90 changes alone never count (the pooled check keeps them)"""
    from dancecap.sides.config import SidesCfg
    from dancecap.sides.gate import chain_cam_worse
    cfg = SidesCfg()
    up = {"mamma": {"base": [10.0, 20.0], "cand": [12.5, 21.0]}}
    assert chain_cam_worse(up, cfg)
    assert not chain_cam_worse({"mamma": {"base": [10.0, 20.0], "cand": [10.5, 40.0]}}, cfg)          # p90 only
    assert not chain_cam_worse({**up, "coco": {"base": [10.0, 20.0], "cand": [7.5, 20.0]}}, cfg)       # contradicted
    assert not chain_cam_worse({"mamma": {"base": [10.0, 20.0], "cand": [11.5, 21.0]}}, cfg)          # within the 2-px chain floor
    assert chain_cam_worse({**up, "coco": {"base": [10.0, 20.0], "cand": [9.5, 20.0]}}, cfg)           # not beyond tolerance
    assert not chain_cam_worse({"coco": {"base": [30.0, 60.0], "cand": [32.5, 60.0]}}, cfg)            # 10 % of a 30-px base
