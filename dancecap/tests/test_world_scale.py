"""World metric scale: the pose-invariant 'chain' rule (cameras/world.py) and the post-fit check / rescale tools
(cameras/world_scale.py). Synthetic, no GPU; one contract check on larissa_kadu when its fit is present.

    .venv-nopo4d\\Scripts\\python.exe -m pytest tests/test_world_scale.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dancecap.cameras.world import (CHAIN_FRAC, SCALE_SYS_REL, apply, bootstrap, chain_measure,  # noqa: E402
                                    rescale, roles_coverage, scale_checks, scale_rule_from_take, sole_from_take)
from dancecap.cameras import world_scale as wsc  # noqa: E402
from dancecap.take import PACKAGE_ROOT, WORKSPACE  # noqa: E402

STAT = {"lead": 1.727, "follow": 1.651}
LOCKED = WORKSPACE / "mamma" / "data" / "body_models" / "smplx_locked_head" / "smplx" / "SMPLX_NEUTRAL.npz"


def _look_at(C, target, up=np.array([0, 1.0, 0])):
    z = target - C
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return R, -R @ C


def _unit(a_deg, b_deg=0.0):
    """Direction tilted a_deg from +Y towards +Z (forward), rotated b_deg about Y."""
    a, b = np.radians(a_deg), np.radians(b_deg)
    return np.stack([np.sin(a) * np.sin(b), np.cos(a), np.sin(a) * np.cos(b)], -1)


def crouched_dancer(H: float, T: int, rng, noise: float = 0.01, outliers: float = 0.02) -> dict[int, np.ndarray]:
    """COCO keypoints (world, metres) of a dancer who NEVER stands upright: knees bent 20-50 deg, torso leaning
    5-20 deg, head tilted 30-60 deg, heels sometimes raised. Segment lengths: leg 0.47 H, trunk 0.30 H,
    shoulder->mid-eye (CHAIN_FRAC - 0.77) H, i.e. the unfolded chain is exactly CHAIN_FRAC x H."""
    thigh, shank, trunk, neck = 0.245 * H, 0.225 * H, 0.30 * H, (CHAIN_FRAC - 0.77) * H
    base = np.c_[rng.normal(0, 0.5, T), np.zeros(T), rng.normal(0, 0.5, T)]
    yaw = rng.uniform(0, 360, T)
    side = np.stack([np.cos(np.radians(yaw)), np.zeros(T), -np.sin(np.radians(yaw))], 1)  # lateral axis
    kp = {}
    hips = []
    for sgn, (a_i, k_i, h_i) in ((+1, (15, 13, 11)), (-1, (16, 14, 12))):
        heel = np.where(rng.random(T) < 0.3, rng.uniform(0, 0.06, T), 0.0)
        ank = base + 0.09 * sgn * side + np.c_[np.zeros(T), 0.07 + heel, np.zeros(T)]
        flex = rng.uniform(20, 50, T)  # knee flexion: knee angle 130-160 deg
        shin_tilt = flex * 0.5
        knee = ank + shank * _unit(shin_tilt, yaw)
        hip = knee + thigh * _unit(-(flex - shin_tilt), yaw)
        kp[a_i], kp[k_i], kp[h_i] = ank, knee, hip
        hips.append(hip)
    hipc = 0.5 * (hips[0] + hips[1])
    shc = hipc + trunk * _unit(rng.uniform(5, 20, T), yaw)
    kp[5], kp[6] = shc + 0.18 * side, shc - 0.18 * side
    eyec = shc + neck * _unit(rng.uniform(30, 60, T), yaw)
    kp[1], kp[2] = eyec + 0.03 * side, eyec - 0.03 * side
    for k in kp:
        kp[k] = kp[k] + rng.normal(0, noise, (T, 3))
        bad = rng.random(T) < outliers
        kp[k][bad] += rng.normal(0, 0.4, (int(bad.sum()), 3))  # mis-triangulations
    return kp


def _scene(seed: int = 3, T: int = 400, statures: dict | None = None):
    from scipy.spatial.transform import Rotation
    rng = np.random.default_rng(seed)
    dancers_w = {r: crouched_dancer(H, T, rng) for r, H in (statures or STAT).items()}
    floor = np.c_[rng.uniform(-3, 3, 3000), np.zeros(3000), rng.uniform(-3, 3, 3000)]
    s_true = 2.7
    Rq = Rotation.from_rotvec(rng.normal(0, 1, 3)).as_matrix()
    tq = rng.normal(0, 2, 3)
    to_sfm = lambda X: s_true * (X @ Rq.T) + tq  # noqa: E731
    tracks = {}
    for k, cam in enumerate(["06", "01", "02"]):
        a = 2 * np.pi * k / 3
        Rw, tw = _look_at(np.array([6 * np.cos(a), 1.4, 6 * np.sin(a)]), np.array([0, 1.0, 0]))
        R_s = Rw @ Rq.T
        tracks[cam] = {"R": np.repeat(R_s[None], 10, 0), "tvec": np.repeat((s_true * tw - R_s @ tq)[None], 10, 0),
                       "ok": np.ones(10, bool)}
    dancers_s = {r: {k: to_sfm(v) for k, v in d.items()} for r, d in dancers_w.items()}
    return tracks, dancers_s, to_sfm(floor), s_true, dancers_w


def test_chain_measure_is_pose_invariant():
    """POSTURE invariance only: the synthetic dancer's chain is CHAIN_FRAC x H by construction, so this says
    nothing about CHAIN_FRAC's absolute accuracy (that is a calibration on larissa_kadu's betas)."""
    rng = np.random.default_rng(1)
    kp = crouched_dancer(1.70, 300, rng)
    cm = chain_measure(kp)
    assert cm is not None and abs(cm["chain"] / (CHAIN_FRAC * 1.70) - 1) < 0.01, cm
    assert 0.001 <= cm["rel_se"] < 0.01
    assert cm["halves_rel_diff"] is not None and cm["halves_rel_diff"] < 0.01  # stationary dancer: halves agree
    assert set(cm["pct"]) == {"leg", "trunk", "shoulder_eye"} and all(a < b for a, b in cm["pct"].values())
    assert abs(cm["leg_lr"][0] / cm["leg_lr"][1] - 1) < 0.01 and cm["shoulder_eye_eyes"] == "any"
    assert chain_measure({k: v for k, v in kp.items() if k != 13}) is None  # a knee missing -> no chain


def test_chain_measure_eyes_and_halves():
    """One-eye frames are recorded separately; eyes="both" uses both-eye frames only (and falls back to "any" with
    too few); a dancer whose segments change between the halves of the window is visible in halves_rel_diff."""
    rng = np.random.default_rng(11)
    kp = crouched_dancer(1.70, 400, rng, outliers=0.0)
    one = rng.random(400) < 0.5
    kp[2] = kp[2].copy()
    kp[2][one] = np.nan  # right eye missing in half of the frames ...
    kp[1] = kp[1].copy()
    kp[1][one] += np.array([0.0, 0.03, 0.0])  # ... and the left eye then reads 3 cm high
    any_ = chain_measure(kp)
    both = chain_measure(kp, eyes="both")
    assert both["shoulder_eye_eyes"] == "both" and both["shoulder_eye"] < any_["shoulder_eye"]
    assert any_["shoulder_eye_one_eye"]["n"] == int(one.sum()) and both["shoulder_eye_both"]["n"] == int((~one).sum())
    assert any_["shoulder_eye_one_eye"]["median"] > any_["shoulder_eye_both"]["median"]
    few = {k: v.copy() for k, v in kp.items()}
    few[2][15:] = np.nan
    assert chain_measure(few, eyes="both")["shoulder_eye_eyes"] == "any"  # < 20 both-eye frames -> fallback
    grow = {k: v.copy() for k, v in kp.items()}
    for k in (13, 15):  # left knee + ankle drift: the leg reads 6 % longer in the second half
        grow[k][200:] += (grow[k][200:] - grow[11][200:]) * 0.06
    g = chain_measure(grow)
    assert g["halves_rel_diff"] > 0.01 and g["halves"][1] > g["halves"][0]
    with pytest.raises(ValueError):
        chain_measure(kp, eyes="one")


def test_chain_rule_recovers_scale_when_never_upright():
    """The old eye-p90 rule over-scales a crouched couple; the chain rule recovers the scale within 1 %."""
    tracks, dancers, floor, s_true, _ = _scene()
    new = bootstrap(tracks, dancers, floor, STAT, "test")  # default rule = chain
    old = bootstrap(tracks, dancers, floor, STAT, "test", scale_rule="eye_p90")
    assert new["scale_rule"] == "chain" and old["scale_rule"] == "eye_p90"
    assert abs(new["s"] * s_true - 1) < 0.01, new["s"] * s_true
    assert old["s"] * s_true - 1 > 0.03, old["s"] * s_true  # never upright -> world too large
    d = new["diagnostics"]
    assert d["scale_agreement_ok"] and d["scale_disagreement"] < 0.015
    assert set(d["scale_chain"]) == {"lead", "follow"} and set(d["scale_eye_p90"]) == {"lead", "follow"}
    assert abs(sum(d["scale_weights"].values()) - 1) < 1e-9
    # weights include the systematic term: statistical SEs of < 1 % cannot make them lopsided
    assert max(d["scale_weights"].values()) < 0.6 and d["scale_sys_rel"] == SCALE_SYS_REL
    assert d["scale_rel_se_combined"] >= SCALE_SYS_REL / np.sqrt(2) - 1e-12
    assert abs(d["scale_s_equal_weight"] / new["s"] - 1) < 0.005
    assert d["scale_rule_requested"] == "chain" and d["scale_abs_uncertainty_rel"] == 0.015
    assert abs(d["chain_frac"]["s_with_prior"] / new["s"] - 0.895 / CHAIN_FRAC) < 1e-12
    fl = apply(new, floor)
    assert np.abs(np.median(fl[:, 1])) < 0.02  # floor still at y = 0


def test_chain_rule_flags_inconsistent_statures():
    tracks, dancers, floor, s_true, _ = _scene(seed=4)
    wrong = {"lead": STAT["lead"], "follow": STAT["follow"] * 1.04}
    sim3 = bootstrap(tracks, dancers, floor, wrong, "test")
    d = sim3["diagnostics"]
    assert abs(d["scale_disagreement"] - 0.04) < 0.012 and not d["scale_agreement_ok"]


def test_scale_rule_fallback_and_take_setting():
    tracks, dancers, floor, s_true, _ = _scene(seed=5)
    eyes_ankles = {r: {k: v for k, v in d.items() if k in (1, 2, 15, 16)} for r, d in dancers.items()}
    sim3 = bootstrap(tracks, eyes_ankles, floor, STAT, "test")
    assert sim3["scale_rule"] == "eye_p90" and "scale_rule_fallback" in sim3["diagnostics"]
    assert scale_rule_from_take({}) == "chain"
    assert scale_rule_from_take({"cameras": {"scale_rule": "eye_p90"}}) == "eye_p90"
    with pytest.raises(ValueError):
        scale_rule_from_take({"cameras": {"scale_rule": "magic"}})
    with pytest.raises(ValueError):
        bootstrap(tracks, dancers, floor, STAT, "test", scale_rule="magic")


def test_rescale_keeps_floor_axes_and_image_projections():
    from scipy.spatial.transform import Rotation
    from dancecap.mamma_dyn import world_cameras
    rng = np.random.default_rng(7)
    sim3 = {"s": 1.3, "R": Rotation.from_rotvec(rng.normal(0, 1, 3)).as_matrix().tolist(),
            "t": rng.normal(0, 1, 3).tolist(), "scale_source": "test", "diagnostics": {}}
    f = 0.9545
    new = rescale(sim3, f, "unit test")
    X = rng.normal(0, 3, (50, 3))
    assert np.allclose(apply(new, X), f * apply(sim3, X))
    Rw = np.asarray(sim3["R"])
    on_floor = (np.c_[rng.normal(0, 2, 20), np.zeros(20), rng.normal(0, 2, 20)] - sim3["t"]) @ Rw / sim3["s"]
    assert np.abs(apply(new, on_floor)[:, 1]).max() < 1e-9  # y = 0 stays the floor
    assert np.allclose(new["R"], sim3["R"]) and new["diagnostics"]["rescaled"][0]["factor"] == f
    assert sim3["s"] == 1.3 and not sim3["diagnostics"]  # input untouched
    # a camera in the SfM frame projects a scene point to the same pixel through either world frame
    Rc = Rotation.from_rotvec(rng.normal(0, 0.3, 3)).as_matrix()[None]
    tc = np.array([[0.1, -0.2, 6.0]])
    for sim in (sim3, new):
        Rw_c, tw_c = world_cameras(Rc, tc, sim)
        xc = apply(sim, X) @ Rw_c[0].T + tw_c[0]
        uv = xc[:, :2] / xc[:, 2:]
        if sim is sim3:
            uv0 = uv
    assert np.allclose(uv, uv0, atol=1e-10)
    with pytest.raises(ValueError):
        rescale(sim3, 0.0, "bad")


def _toy_model(V: int = 400, seed: int = 0) -> dict:
    """Linear 'body': beta 0 scales the whole template about the origin by +10 %."""
    rng = np.random.default_rng(seed)
    vt = np.c_[rng.uniform(-0.2, 0.2, V), rng.uniform(0, 1.70, V), rng.uniform(-0.1, 0.1, V)]
    vt[0, 1], vt[1, 1] = 0.0, 1.70
    S = np.zeros((V, 3, 2))
    S[:, :, 0] = 0.1 * vt
    S[:, 0, 1] = rng.normal(0, 0.01, V)
    Jr = np.zeros((55, V))
    Jr[:, :10] = 0.1
    return {"v_template": vt, "shapedirs": S, "J_regressor": Jr, "path": "toy"}


def test_betas_check_verdicts():
    m = _toy_model()
    models = {"lead": m, "follow": m}
    st = {"lead": 1.70 - 0.025, "follow": 1.70 * 1.02 - 0.025}  # rest heights 1.70 and 1.734 m
    ok = wsc.betas_check({"lead": {"betas": np.zeros(2)}, "follow": {"betas": np.array([0.2, 0])}}, st, models=models)
    assert ok["verdict"] == "ok" and abs(ok["factor"] - 1) < 1e-9 and ok["disagreement"] < 1e-9
    big = wsc.betas_check({"lead": {"betas": np.array([0.5, 0])}, "follow": {"betas": np.array([0.71, 0])}}, st,
                          models=models)
    assert big["verdict"] == "rescale" and abs(big["factor"] - 1 / 1.05) < 1e-6
    bad = wsc.betas_check({"lead": {"betas": np.array([0.5, 0])}, "follow": {"betas": np.array([0.2, 0])}}, st,
                          models=models)
    assert bad["verdict"] == "inconsistent" and bad["disagreement"] > 0.015
    assert set(big["sole_sensitivity"]) == {"0.000", "0.020", "0.025", "0.030"}
    # mean within tol but one dancer 1.6 % off (dancers agree within 1.4 %) -> rescale, not ok
    one = wsc.betas_check({"lead": {"betas": np.array([0.16, 0])}, "follow": {"betas": np.array([0.2204, 0])}}, st,
                          models=models)
    assert abs(1 / one["factor"] - 1) < 0.015 and one["disagreement"] < 0.015
    assert not one["per_dancer"]["lead"]["ok"] and one["verdict"] == "rescale"
    # per-dancer soles: the follow measured in her shoes (sole 0) -> her target drops by 2.5 cm
    ps = wsc.betas_check({"lead": {"betas": np.zeros(2)}, "follow": {"betas": np.array([0.2, 0])}}, st,
                         sole_m={"lead": 0.025, "follow": 0.0}, models=models)
    assert ps["per_dancer"]["follow"]["target_m"] == pytest.approx(st["follow"])
    assert ps["per_dancer"]["follow"]["k"] > ok["per_dancer"]["follow"]["k"]


@pytest.mark.skipif(not LOCKED.exists(), reason="MAMMA locked-head SMPL-X model not present")
def test_rescale_fit_on_smplx():
    model = wsc.body_model(LOCKED)
    rng = np.random.default_rng(2)
    b = rng.normal(0, 1, 16)
    T = 5
    joints = rng.normal(0, 0.3, (T, 55, 3)) + [0, 1.0, 0]
    npz = {"t": np.arange(T) / 30, "betas": b.astype(np.float32), "transl": rng.normal(0, 1, (T, 3)), "joints": joints}
    f = 0.95
    new, info = wsc.rescale_fit(npz, f, model)
    assert abs(info["rest_height_after_m"] / info["rest_height_before_m"] - f) < 0.002
    assert info["vertex_fit_rms_mm"] < 2.0 and info["joint_fit_max_mm"] < 3.0
    Jr0 = model["J_regressor"][0]
    j0_old = Jr0 @ wsc.rest_vertices(b, model)
    j0_new = Jr0 @ wsc.rest_vertices(new["betas"], model)
    assert np.allclose(j0_new + new["transl"], f * (j0_old + npz["transl"]), atol=1e-6)  # pelvis at f x old place
    assert np.allclose(new["joints"], f * joints) and new["betas"].dtype == np.float32


def test_rescale_world_files(tmp_path):
    work = tmp_path / "take"
    (work / "cameras").mkdir(parents=True)
    (work / "room").mkdir()
    (work / "smplx").mkdir()
    sim3 = {"s": 1.05, "R": np.eye(3).tolist(), "t": [0.5, 1.0, -2.0], "scale_source": "x", "diagnostics": {}}
    (work / "cameras" / "world_from_sfm.json").write_text(json.dumps(sim3))
    tr = {"world_from_sfm": {k: sim3[k] for k in ("s", "R", "t")},
          "floor_plane_sfm": {"world_height_at_origin_m": 0.004},
          "mirror": {"top_m": 2.7, "cams_dist_m": [7.0, 8.0],
                     "world": {"n": [1, 0, 0], "d": 3.8, "along_range": [-7.5, 5.6], "height_range_m": [0.0, 2.7]}}}
    (work / "room" / "transform.json").write_text(json.dumps(tr))
    dry = wsc.rescale_world(work, 0.95, "test", dry_run=True)
    assert dry["dry_run"] and json.loads((work / "cameras" / "world_from_sfm.json").read_text())["s"] == 1.05
    rep = wsc.rescale_world(work, 0.95, "test")
    w = json.loads((work / "cameras" / "world_from_sfm.json").read_text())
    assert abs(w["s"] - 0.9975) < 1e-12 and np.allclose(w["t"], [0.475, 0.95, -1.9])
    assert json.loads((work / "cameras" / "world_from_sfm.prev.json").read_text())["s"] == 1.05
    t2 = json.loads((work / "room" / "transform.json").read_text())
    assert t2["world_from_sfm"]["s"] == w["s"] and abs(t2["mirror"]["world"]["d"] - 3.61) < 1e-12
    assert np.allclose(t2["mirror"]["world"]["along_range"], [-7.125, 5.32]) and abs(t2["mirror"]["top_m"] - 2.565) < 1e-12
    assert (work / "room" / "transform.prev.json").exists() and rep["room_synced"] is True
    assert "smplx" in rep["stale"]


def _fake_take(work: Path, C_by_cam: dict, pts_by_cam: dict, attached: set = frozenset()):
    for cam, C in C_by_cam.items():
        (work / "cameras").mkdir(parents=True, exist_ok=True)
        (work / "frames" / cam).mkdir(parents=True, exist_ok=True)
        F = len(C)
        R = np.repeat(np.eye(3)[None], F, 0)
        np.savez(work / "cameras" / f"{cam}.npz", R=R, tvec=-C, ok=np.ones(F, bool), attached=np.bool_(cam in attached))
        (work / "frames" / cam / "frames.json").write_text(json.dumps(
            {"frames": [{"file": f"{i:06d}.jpg", "pts": float(p), "t": float(p)} for i, p in enumerate(pts_by_cam[cam])]}))


def test_sfm_similarity_and_inherited_world(tmp_path):
    from scipy.spatial.transform import Rotation
    rng = np.random.default_rng(9)
    s, R, t = 0.42, Rotation.from_rotvec([0.3, -1.1, 0.4]).as_matrix(), np.array([1.0, -2.0, 0.5])
    Ca, Cb, pa, pb = {}, {}, {}, {}
    for k, cam in enumerate(["01", "02", "03", "05"]):
        a = 2 * np.pi * k / 4
        F = 60
        Cb[cam] = np.array([5 * np.cos(a), 1.5, 5 * np.sin(a)]) + rng.normal(0, 0.05, (F, 3))
        pb[cam] = 10 + np.arange(F) / 30
        Ca[cam] = s * Cb[cam] @ R.T + t
        pa[cam] = pb[cam].copy()
        if cam == "02":  # take a starts later: only part of the frames are common
            Ca[cam], pa[cam] = Ca[cam][20:], pa[cam][20:]
    Ca["05"] = Ca["05"] + 3.0  # an attached camera placed elsewhere: must be ignored
    Ca["01"][::10] += 0.5  # badly localised frames: trimmed
    _fake_take(tmp_path / "a", Ca, pa, attached={"05"})
    _fake_take(tmp_path / "b", Cb, pb, attached={"05"})
    sim = wsc.sfm_similarity(tmp_path / "a", tmp_path / "b")
    assert abs(sim["s"] / s - 1) < 1e-3 and np.allclose(sim["R"], R, atol=2e-3)
    assert "05" not in sim["per_cam"] and sim["per_cam"]["02"] == 40
    ref = {"s": 1.006, "R": Rotation.from_rotvec([0.1, 0.2, -0.3]).as_matrix().tolist(), "t": [0.3, 1.2, -0.7]}
    inh = wsc.inherit_world(ref, sim)
    Xb = rng.normal(0, 2, (10, 3))
    Xa = sim["s"] * Xb @ np.asarray(sim["R"]).T + sim["t"]
    assert np.allclose(apply(inh, Xb), apply(ref, Xa), atol=1e-6)
    cmp_ = wsc.compare_reference(tmp_path / "b", tmp_path / "a", ref)
    assert abs(cmp_["s_expected"] - ref["s"] * sim["s"]) < 1e-12 and "this" not in cmp_


@pytest.mark.skipif(not (PACKAGE_ROOT / "work" / "larissa_kadu" / "smplx" / "lead.npz").exists() or not LOCKED.exists(),
                    reason="larissa_kadu fit not present")
def test_larissa_kadu_betas_agree():
    """Contract on the real take: the two dancers' rest heights imply the same world scale within 1.5 %."""
    res = wsc.check_take("larissa_kadu")
    assert res["disagreement"] < 0.015 and res["verdict"] in ("ok", "rescale")


def test_sole_from_take_and_roles_coverage(tmp_path):
    assert sole_from_take({}) == ({"lead": 0.025, "follow": 0.025}, "assumed 2.5 cm sneaker sole")
    soles, src = sole_from_take({"dancers": {"shoe_sole_m": 0.02, "follow": {"shoe_sole_m": 0.0}}})
    assert soles == {"lead": 0.02, "follow": 0.0} and "toml" in src
    with pytest.raises(ValueError):
        sole_from_take({"dancers": {"lead": {"shoe_sole_m": 0.5}}})
    (tmp_path / "dancers").mkdir()
    box = {"box": [0, 0, 1, 1]}
    (tmp_path / "dancers" / "roles.json").write_text(json.dumps({"cams": {
        "01": {"frames": [{"lead": box, "follow": box}, {"lead": box, "follow": None}]}}}))
    cov = roles_coverage(tmp_path, ["01", "02"])
    assert cov["01"] == {"source": "roles.json", "frames": 2, "lead": 2, "follow": 1}
    assert cov["02"]["source"].startswith("fallback")


def _chain_sim3(seed: int = 3, T: int = 1200, cov_ok: bool = True):
    tracks, dancers, floor, s_true, _ = _scene(seed=seed, T=T)
    sim3 = bootstrap(tracks, dancers, floor, STAT, "test")
    sim3["diagnostics"]["dancer_roles"] = {c: ({"source": "roles.json", "frames": 10, "lead": 10, "follow": 10}
                                               if cov_ok or c != "02" else {"source": "fallback: two largest boxes"})
                                           for c in tracks}
    return sim3


def test_scale_checks_gates():
    sim3 = _chain_sim3()
    g = scale_checks(sim3)
    assert g["world_scale_chain"]["passed"], g["world_scale_chain"]
    assert g["world_scale"]["passed"] and g["world_scale"]["value"]["adopted"] == "chain"
    assert not scale_checks(_chain_sim3(cov_ok=False))["world_scale"]["passed"]  # a camera without roles.json
    bad = json.loads(json.dumps(sim3))
    bad["diagnostics"]["scale_chain"]["follow"]["halves_rel_diff"] = 0.03
    gb = scale_checks(bad)
    assert not gb["world_scale_chain"]["passed"] and not gb["world_scale"]["passed"]
    assert not gb["world_scale_chain"]["checks"]["halves"]["passed"] and gb["world_scale_chain"]["checks"]["rel_se"]["passed"]
    fb = json.loads(json.dumps(sim3))
    fb["scale_rule"], fb["diagnostics"]["scale_rule_fallback"] = "eye_p90", "chain keypoints missing"
    assert not scale_checks(fb)["world_scale"]["checks"]["rule_as_requested"]["passed"]
    # with an adopted reference the chain is a cross-check: halves failing does not fail world_scale, a > 2 % chain
    # vs reference does; a reference that was not adopted fails it
    ref = json.loads(json.dumps(bad))
    ref["diagnostics"]["scale_reference"] = {"adopted": True, "chain_vs_reference": -0.004, "why": "ok"}
    gr = scale_checks(ref)
    assert gr["world_scale"]["passed"] and not gr["world_scale_chain"]["passed"]
    assert gr["world_scale"]["value"]["adopted"] == "reference"
    ref["diagnostics"]["scale_reference"]["chain_vs_reference"] = 0.025
    assert not scale_checks(ref)["world_scale"]["passed"]
    ref["diagnostics"]["scale_reference"] = {"adopted": False, "chain_vs_reference": 0.001, "why": "2 cameras"}
    assert not scale_checks(ref)["world_scale"]["passed"]
    # report.gates carries them into cameras/report.json gates_G2a
    from dancecap.cameras.report import gates
    rep = {"cams": {"01": {"in_main_model": True, "hd": False, "map_reproj_px": 0.5, "frames": 10, "frames_ok": 10,
                           "focal": {"plausible": True}}}}
    out = gates(rep, 1, sim3)
    assert out["world_scale"]["passed"] and "world_scale_chain" in out and "world_scale" not in gates(rep, 1)


def test_adopt_reference(tmp_path):
    """Take b inherits take a's (corrected) scale through the camera-frame similarity when the geometry gates pass;
    gravity mismatch, too few cameras or a reference built with other soles keep b's own (chain) scale."""
    from scipy.spatial.transform import Rotation
    rng = np.random.default_rng(21)
    s, R, t = 0.8, Rotation.from_rotvec([0.2, 0.7, -0.1]).as_matrix(), np.array([0.5, 1.0, -1.5])
    Ca, Cb, pa, pb = {}, {}, {}, {}
    for k, cam in enumerate(["01", "02", "03", "04", "07"]):
        a = 2 * np.pi * k / 5
        Cb[cam] = np.array([5 * np.cos(a), 1.5, 5 * np.sin(a)]) + rng.normal(0, 0.05, (90, 3))
        pb[cam] = 20 + np.arange(90) / 30
        Ca[cam], pa[cam] = s * Cb[cam] @ R.T + t, pb[cam].copy()
    _fake_take(tmp_path / "a", Ca, pa)
    _fake_take(tmp_path / "b", Cb, pb)
    ref = {"s": 1.006, "R": Rotation.from_rotvec([0.1, 0.2, -0.3]).as_matrix().tolist(), "t": [0.3, 1.2, -0.7],
           "reference": {"sole_m": {"lead": 0.025, "follow": 0.025}}}
    Rinh = np.asarray(ref["R"]) @ R
    sim3_b = {"s": ref["s"] * s * 1.012, "R": Rinh.tolist(), "t": [0.0, 0.1, 0.0], "scale_rule": "chain",
              "scale_source": "chain", "diagnostics": {}}  # chain 1.2 % above the reference, same gravity
    soles = {"lead": 0.025, "follow": 0.025}
    new, info = wsc.adopt_reference(tmp_path / "b", sim3_b, tmp_path / "a", ref, soles, ref_name="a")
    assert info["adopted"] and info["n_cams"] == 5 and info["resid_rel"] < 1e-6
    assert new["s"] == pytest.approx(ref["s"] * s, rel=1e-9) and info["chain_vs_reference"] == pytest.approx(0.012, abs=1e-6)
    assert np.allclose(new["R"], sim3_b["R"]) and np.allclose(new["t"], np.asarray(sim3_b["t"]) / 1.012)
    assert new["diagnostics"]["scale_reference"]["adopted"] and new["diagnostics"]["rescaled"][0]["factor"] < 1
    assert sim3_b["s"] == pytest.approx(ref["s"] * s * 1.012) and not sim3_b["diagnostics"]  # input untouched
    tilt = Rotation.from_rotvec([np.radians(1.5), 0, 0]).as_matrix()  # this take's gravity 1.5 deg off
    tilted = dict(sim3_b, R=(tilt @ Rinh).tolist())
    keep, info2 = wsc.adopt_reference(tmp_path / "b", tilted, tmp_path / "a", ref, soles)
    assert not info2["adopted"] and "gravity" in info2["why"] and keep["s"] == tilted["s"]
    _, info3 = wsc.adopt_reference(tmp_path / "b", sim3_b, tmp_path / "a", ref, {"lead": 0.0, "follow": 0.0})
    assert not info3["adopted"] and "soles" in info3["why"]
    for cam in ("03", "04", "07"):  # only 2 common cameras left
        (tmp_path / "a" / "cameras" / f"{cam}.npz").unlink()
    _, info4 = wsc.adopt_reference(tmp_path / "b", sim3_b, tmp_path / "a", ref, soles)
    assert not info4["adopted"] and "only 2 cameras" in info4["why"]


@pytest.mark.skipif(not (PACKAGE_ROOT / "work" / "larissa_kadu" / "smplx" / "lead.npz").exists() or not LOCKED.exists(),
                    reason="larissa_kadu fit not present")
def test_larissa_kadu_reference():
    """make_reference = the published world x the betas factor, with the record adopt_reference checks."""
    ref = wsc.make_reference("larissa_kadu")
    w = json.loads((PACKAGE_ROOT / "work" / "larissa_kadu" / "cameras" / "world_from_sfm.json").read_text())
    rec = ref["reference"]
    assert ref["s"] == pytest.approx(w["s"] * rec["factor"]) and np.allclose(ref["R"], w["R"])
    assert rec["sole_m"] == {"lead": 0.025, "follow": 0.025} and rec["verdict"] == "rescale"
    assert 0.94 < rec["factor"] < 0.97


# ------------------------------------------------------------------ scale review fixes, round 2 (2026-10-06 evening)

def test_chain_rule_sole_coupling():
    """CHAIN_FRAC was calibrated with target = stature + 2.5 cm: with another shoe sole the chain rule moves exactly
    like the betas scale (chain_stature = stature + sole - 2.5 cm); at the default sole nothing changes."""
    from dancecap.cameras.world import CHAIN_FRAC_SOLE_M, chain_scale, chain_stature
    assert chain_stature(1.727) == pytest.approx(1.727) and chain_stature(1.727, 0.0) == pytest.approx(1.702)
    tracks, dancers, floor, s_true, _ = _scene(seed=5, T=800)
    ch0, c0 = chain_scale(dancers, STAT)
    ch1, c1 = chain_scale(dancers, STAT, {"lead": 0.0, "follow": 0.0})
    for r in STAT:
        assert ch1[r]["s"] / ch0[r]["s"] == pytest.approx((STAT[r] - CHAIN_FRAC_SOLE_M) / STAT[r], rel=1e-12)
    # the betas factor for sole 0 vs 2.5 cm moves the same way (larissa_kadu: 0.9405 / 0.9545 = 0.9854)
    assert c1["s"] / c0["s"] == pytest.approx(np.mean([(v - 0.025) / v for v in STAT.values()]), rel=2e-4)
    assert c0["chain_frac"]["calibration_sole_m"] == CHAIN_FRAC_SOLE_M and "barefoot" in c0["chain_frac"]["note"]
    # bootstrap takes per-dancer soles, uses their mean for the floor, and records both
    w0 = bootstrap(tracks, dancers, floor, STAT, "test")
    w1 = bootstrap(tracks, dancers, floor, STAT, "test", sole_m={"lead": 0.0, "follow": 0.0})
    assert w0["s"] == pytest.approx(c0["s"]) and w1["s"] == pytest.approx(c1["s"])
    assert w1["diagnostics"]["soles_m"] == {"lead": 0.0, "follow": 0.0} and w1["diagnostics"]["sole_m"] == 0.0
    assert w0["diagnostics"]["scale_abs_verified"] is False
    g = scale_checks(_chain_sim3())
    assert g["world_scale"]["value"]["abs_verified"] is False and "SELF-CONSISTENT" in g["world_scale"]["note"]


def _ref_pair(tmp_path, seed=21):
    from scipy.spatial.transform import Rotation
    rng = np.random.default_rng(seed)
    s, R, t = 0.8, Rotation.from_rotvec([0.2, 0.7, -0.1]).as_matrix(), np.array([0.5, 1.0, -1.5])
    Ca, Cb, pa, pb = {}, {}, {}, {}
    for k, cam in enumerate(["01", "02", "03", "04", "07"]):
        a = 2 * np.pi * k / 5
        Cb[cam] = np.array([5 * np.cos(a), 1.5, 5 * np.sin(a)]) + rng.normal(0, 0.05, (90, 3))
        pb[cam] = 20 + np.arange(90) / 30
        Ca[cam], pa[cam] = s * Cb[cam] @ R.T + t, pb[cam].copy()
    _fake_take(tmp_path / "a", Ca, pa)
    _fake_take(tmp_path / "b", Cb, pb)
    (tmp_path / "a" / "cameras" / "sfm").mkdir()
    (tmp_path / "a" / "cameras" / "sfm" / "images.bin").write_bytes(b"model v1")
    ref = {"s": 1.006, "R": Rotation.from_rotvec([0.1, 0.2, -0.3]).as_matrix().tolist(), "t": [0.3, 1.2, -0.7]}
    sim3_b = {"s": ref["s"] * s * 1.012, "R": (np.asarray(ref["R"]) @ R).tolist(), "t": [0.0, 0.1, 0.0],
              "scale_rule": "chain", "scale_source": "chain", "diagnostics": {}}
    return ref, sim3_b


def test_adopt_reference_checks_statures_and_inputs(tmp_path):
    """A reference is refused when it was built with other statures (a corrected stature must rebuild it) or when
    the reference take's SfM model / camera tracks / frame pts changed since; a record without a fingerprint is
    adopted with a warning."""
    ref, sim3_b = _ref_pair(tmp_path)
    soles = {"lead": 0.025, "follow": 0.025}
    fp = wsc.inputs_fingerprint(tmp_path / "a")
    cams = ("01", "02", "03", "04", "07")
    assert set(fp["files"]) == {"cameras/sfm/images.bin", *(f"cameras/{c}.npz" for c in cams),
                                *(f"frames/{c}/frames.json" for c in cams)}
    ref["reference"] = {"sole_m": soles, "statures": dict(STAT), "inputs": fp}
    new, info = wsc.adopt_reference(tmp_path / "b", sim3_b, tmp_path / "a", ref, soles, "a", statures=dict(STAT))
    assert info["adopted"] and info["ref_inputs_verified"] is True and info["statures_checked"]
    assert info["ref_record"]["inputs"] == {"sha1": fp["sha1"], "n_files": len(fp["files"])}  # small diagnostics
    _, i2 = wsc.adopt_reference(tmp_path / "b", sim3_b, tmp_path / "a", ref, soles, "a",
                                statures={"lead": 1.727, "follow": 1.676})  # follow corrected by an inch
    assert not i2["adopted"] and "statures" in i2["why"]
    (tmp_path / "a" / "cameras" / "sfm" / "images.bin").write_bytes(b"model v2: re-mapped")
    _, i3 = wsc.adopt_reference(tmp_path / "b", sim3_b, tmp_path / "a", ref, soles, "a", statures=dict(STAT))
    assert not i3["adopted"] and i3["ref_inputs_verified"] is False and "images.bin" in i3["why"]
    old = {**ref, "reference": {"sole_m": soles, "statures": dict(STAT)}}  # record made before fingerprints
    new4, i4 = wsc.adopt_reference(tmp_path / "b", sim3_b, tmp_path / "a", old, soles, "a", statures=dict(STAT))
    assert i4["adopted"] and i4["ref_inputs_verified"] is None and "fingerprint" in i4["warning"]
    _, i5 = wsc.adopt_reference(tmp_path / "b", sim3_b, tmp_path / "a", {**ref, "reference": {"sole_m": soles}},
                                soles, "a", statures=dict(STAT))
    assert not i5["adopted"] and "no statures" in i5["why"]


def test_chain_check_applies_sync_offsets(tmp_path, monkeypatch):
    """world_scale chain: main cameras only, frame times shifted by the sync offsets, chain vs the world's scale."""
    from dancecap.cameras import world as wmod
    tracks, dancers, floor, s_true, _ = _scene(seed=3, T=1200)
    sim3 = bootstrap(tracks, dancers, floor, STAT, "test")
    (tmp_path / "cameras").mkdir()
    for cam, tr in {**tracks, "05": tracks["06"]}.items():
        F = len(tr["R"])
        np.savez(tmp_path / "cameras" / f"{cam}.npz", t=10 + np.arange(F) / 30, K=np.repeat(np.eye(3)[None], F, 0),
                 R=tr["R"], tvec=tr["tvec"], dist=np.zeros(F), ok=tr["ok"], attached=np.bool_(cam == "05"))
    seen = {}

    def fake_tri(work, trk, **kw):
        seen.update(trk)
        return dancers
    monkeypatch.setattr(wmod, "triangulate_dancers", fake_tri)
    offs = {"01": {"delta": 0.05, "rate": 0.0, "t_ref": 0.0}}
    res = wsc.chain_check_work(tmp_path, sim3, STAT, {"lead": 0.025, "follow": 0.025}, offs)
    assert sorted(seen) == ["01", "02", "06"]  # attached camera 05 excluded
    assert seen["01"]["t"][0] == pytest.approx(10.05) and seen["02"]["t"][0] == pytest.approx(10.0)
    assert res["chain_vs_world"] == pytest.approx(0.0, abs=1e-9) and res["passed"] and res["sync_applied"]
    assert res["offsets_ms"] == {"01": 50.0} and res["sync_effect_rel"] == pytest.approx(0.0, abs=1e-9)
    w2 = wmod.rescale(sim3, 0.97, "test")  # a world 3 % smaller than the chain: fails the 2 % cross-check
    assert not wsc.chain_check_work(tmp_path, w2, STAT, {"lead": 0.025, "follow": 0.025}, offs)["passed"]


@pytest.mark.skipif(not (PACKAGE_ROOT / "work" / "larissa_kadu" / "smplx" / "lead.npz").exists() or not LOCKED.exists(),
                    reason="larissa_kadu fit not present")
def test_floor_stats_on_larissa_kadu():
    """Posed lowest vertex over all frames; lifting the fit by 5 cm lifts every statistic by 50 mm."""
    z = dict(np.load(PACKAGE_ROOT / "work" / "larissa_kadu" / "smplx" / "lead.npz", allow_pickle=True))
    fs = wsc.floor_stats(z)
    assert fs["n_frames"] == len(z["t"]) and fs["p5_mm"] <= fs["median_mm"] <= fs["p95_mm"]
    assert -60 < fs["median_mm"] < 30 and fs["n_below_15mm"] == int((fs["lowest_vertex_m"] < -0.015).sum())
    up = dict(z, transl=np.asarray(z["transl"], float) + [0, 0.05, 0])
    fu = wsc.floor_stats(up)
    assert fu["median_mm"] == pytest.approx(fs["median_mm"] + 50, abs=1e-6) and fu["n_below_15mm"] <= fs["n_below_15mm"]


SYN = PACKAGE_ROOT / "work" / "synthetic"


@pytest.mark.skipif(not (SYN / "smplx" / "lead.npz").exists(), reason="run: python -m dancecap synthetic")
def test_physics_stature_warning_uses_scale_tolerance():
    """physics warns at the world_scale tolerance (1.5 %), not 5 % (which missed Larissa's +4.25 %)."""
    from dancecap import physics as ph
    from dancecap import physics_body as pb
    from dancecap.cameras.world import SCALE_AGREE_REL
    z = ph.load_npz(SYN / "smplx" / "lead.npz")
    H = pb.load_body(z).stature
    cfg = ph.PhysicsConfig()
    hit = ph.analyze_dancer(z, cfg, stature=H / 1.0425 - 0.025, sole_m=0.025)["info"]
    ok = ph.analyze_dancer(z, cfg, stature=H / 1.01 - 0.025, sole_m=0.025)["info"]
    assert hit["stature_warning"] and hit["stature_tol"] == SCALE_AGREE_REL and not ok["stature_warning"]
