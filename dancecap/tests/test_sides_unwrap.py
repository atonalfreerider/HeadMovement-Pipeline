"""dancecap.sides.unwrap: R16 chain wraps found and removed without moving joints, events that are not pure wraps refused, the L0
spine split, and (with local data) the Carlos & Aline regression windows 126.13-127.87 / 154.47-156.50 s."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rot

from dancecap.sides import unwrap as uw
from dancecap.take import PACKAGE_ROOT

# a plausible SMPL-X-like rest skeleton (m; +x = the body's left, +y up, +z forward)
REST = np.array([
    [0, 0, 0], [0.06, -0.09, 0], [-0.06, -0.09, 0], [0, 0.11, -0.01], [0.10, -0.47, 0], [-0.10, -0.47, 0],
    [0, 0.24, 0.01], [0.09, -0.87, -0.04], [-0.09, -0.87, -0.04], [0, 0.29, 0.03], [0.12, -0.92, 0.08], [-0.12, -0.92, 0.08],
    [0, 0.50, 0], [0.07, 0.41, 0], [-0.07, 0.41, 0], [0, 0.60, 0.05], [0.17, 0.45, -0.01], [-0.17, 0.45, -0.01],
    [0.43, 0.43, -0.03], [-0.43, 0.43, -0.03], [0.68, 0.44, -0.03], [-0.68, 0.44, -0.03]], float)
T = 60


def _clean_pose(seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(T) / 30.0
    P = np.zeros((T, 22, 3))
    for j in range(22):
        a, f, ph = rng.normal(0, 0.12, 3), rng.uniform(0.3, 1.2, 3), rng.uniform(0, 6.28, 3)
        P[:, j] = a[None] * np.sin(2 * np.pi * f[None] * t[:, None] + ph[None])
    P[:, 0, 1] += np.linspace(0, 1.0, T)        # the pelvis turns ~57 deg over the clip
    return P


def _fit(P, transl=None):
    tr = np.zeros((len(P), 3)) if transl is None else transl
    _, X = uw.fk22(uw.rotmats(P), REST, tr)
    return {"global_orient": P[:, 0].copy(), "body_pose": P[:, 1:22].reshape(len(P), 63).copy(), "joints": X, "transl": tr,
            "betas": np.zeros(10)}


def _fk(new, idx):
    P = uw.pose_from_fit(new)[idx]
    return uw.fk22(uw.rotmats(P), REST, np.asarray(new["transl"], float)[idx])[1]


def _inject_wrap(P, phi_deg, j1=uw.SPINE1, j2=uw.SPINE2, axis_pair=(uw.SPINE1, uw.SPINE2)):
    """turn joint j1's GLOBAL frame by phi about its bone and undo it in j2's local rotation: the joints stay, the twists wrap"""
    a = uw._unit(REST[axis_pair[1]] - REST[axis_pair[0]])
    R = uw.rotmats(P)
    Tp = uw.axis_rot(a, phi_deg)
    R[:, j1] = R[:, j1] @ Tp
    R[:, j2] = np.swapaxes(Tp, 1, 2) @ R[:, j2]
    out = P.copy()
    out[:, j1], out[:, j2] = uw.rotvecs(R[:, j1]), uw.rotvecs(R[:, j2])
    return out


def _phi_bump(peak, a=15, b=45):
    phi = np.zeros(T)
    x = np.clip((np.arange(T) - a) / (b - a), 0, 1)
    phi[:] = peak * np.sin(np.pi * x) ** 2
    return phi


def test_twist_round_trip():
    rng = np.random.default_rng(3)
    a = uw._unit(np.array([0.1, 1.0, -0.2]))
    for th in (-170.0, -45.0, 0.0, 30.0, 179.0):
        u = uw._unit(np.cross(a, rng.normal(size=3)))           # a swing axis perpendicular to a
        S = Rot.from_rotvec(u * np.radians(rng.uniform(0, 80))).as_matrix()
        R = S @ uw.axis_rot(a, np.array([th]))[0]
        assert abs(uw._wrap(uw.twist_deg(R[None], a)[0] - th)) < 1e-9


def test_fk22_rest_and_translation():
    P = np.zeros((2, 22, 3))
    tr = np.array([[0, 0, 0], [1.0, 2.0, 3.0]])
    _, X = uw.fk22(uw.rotmats(P), REST, tr)
    assert np.allclose(X[0], REST) and np.allclose(X[1], REST + tr[1])


def test_synthetic_spine_wrap_found_and_unwrapped():
    clean = _clean_pose()
    P = _inject_wrap(clean, _phi_bump(178.0))
    fit = _fit(P)
    assert np.abs(fit["joints"] - _fit(clean)["joints"]).max() < 1e-9      # the wrap moves no joint
    d = uw.chain_wraps(fit, rest=REST)["spine"]
    assert d["wrapped"].any() and d["candy"].any()
    assert (np.abs(d["w"][d["wrapped"]]) >= 300).all()
    new, rep = uw.unwrap_fit(fit, rest=REST, fk=_fk, split=False)
    assert rep["max_joint_move_m"] < 1e-3                                  # within 1 mm (it is float round-off)
    after = uw.chain_wraps(new, rest=REST)["spine"]
    assert not after["wrapped"].any() and not after["candy"].any()
    assert rep["chains"]["spine"]["wrapped_frames_after"] == 0 and not rep["chains"]["spine"]["refused"]
    # spine3's world rotation (everything above it) is kept exactly
    G0, _ = uw.fk22(uw.rotmats(P), REST)
    G1, _ = uw.fk22(uw.rotmats(uw.pose_from_fit(new)), REST)
    assert np.abs(G0[:, uw.SPINE3] - G1[:, uw.SPINE3]).max() < 1e-9
    assert np.abs(G0[:, uw.HEAD] - G1[:, uw.HEAD]).max() < 1e-9
    # the twists end near the clean motion's (a redistribution of its total)
    cl = uw.chain_wraps(_fit(clean), rest=REST)["spine"]
    assert np.abs(after["twists"]).max() <= np.abs(cl["sum"]).max() + 5.0


def test_unwrap_has_no_skin_pop():
    clean = _clean_pose(1)
    P = _inject_wrap(clean, _phi_bump(175.0, 10, 50))
    new, _ = uw.unwrap_fit(_fit(P), rest=REST, fk=_fk, split=False)
    Pn = uw.pose_from_fit(new)

    def step(Q, j):
        R = Rot.from_rotvec(Q[:, j])
        return np.degrees((R[:-1].inv() * R[1:]).magnitude())
    for j in (uw.SPINE1, uw.SPINE2, uw.SPINE3):
        # no pop: at most the ramp's few degrees per frame over the clean motion (the wrapped input moves up to ~15 deg per frame)
        assert step(Pn, j).max() <= step(clean, j).max() + 3.0


def test_leg_wrap_unwrapped():
    clean = _clean_pose(2)
    P = _inject_wrap(clean, _phi_bump(-176.0), uw.L_HIP, uw.L_KNEE, (uw.L_HIP, uw.L_KNEE))
    fit = _fit(P)
    d = uw.chain_wraps(fit, rest=REST)["leg_l"]
    assert d["candy"].any()
    new, rep = uw.unwrap_fit(fit, rest=REST, fk=_fk, split=False)
    assert rep["max_joint_move_m"] < 1e-3
    after = uw.chain_wraps(new, rest=REST)["leg_l"]
    assert not after["wrapped"].any() and not after["candy"].any()


def test_real_torso_turn_is_refused_not_popped():
    """a torso that really turns 270 deg against the pelvis (the joints show it) is not a wrap: the re-expression would leave a
    wrap, so the event is refused and the pose is left exactly as it was"""
    clean = _clean_pose(4)
    R = uw.rotmats(clean)
    a = uw._unit(REST[uw.SPINE2] - REST[uw.SPINE1])
    R[:, uw.SPINE1] = R[:, uw.SPINE1] @ uw.axis_rot(a, _phi_bump(270.0))   # a real rotation: shoulders follow
    P = clean.copy()
    P[:, uw.SPINE1] = uw.rotvecs(R[:, uw.SPINE1])
    fit = _fit(P)
    new, rep = uw.unwrap_fit(fit, rest=REST, fk=_fk, split=False)
    assert rep["chains"]["spine"]["refused"], rep["chains"]["spine"]
    assert np.array_equal(uw.pose_from_fit(new)[:, uw.SPINE1], P[:, uw.SPINE1])


def test_guard_flags_not_pure_frames_and_gate_mode_drops_them():
    clean = _clean_pose(5)
    P = _inject_wrap(clean, _phi_bump(178.0))
    fit = _fit(P)
    d = uw.chain_wraps(fit, rest=REST)["spine"]
    bad = d["wrapped"] | d["candy"]
    assert bad.any()
    # the guard holds on a pure wrap (Sigma* = the clean total, close to the shoulder-minus-hip heading)
    assert d["guard_ok"][bad].mean() > 0.9
    # a forged relative heading 90 deg off makes the same frames not-pure; gate mode then re-expresses none of them
    J = fit["joints"].copy()
    ang = np.radians(90.0)
    Ry = np.array([[np.cos(ang), 0, np.sin(ang)], [0, 1, 0], [-np.sin(ang), 0, np.cos(ang)]])
    mid = 0.5 * (J[:, 16] + J[:, 17])
    J[:, 16], J[:, 17] = mid + (J[:, 16] - mid) @ Ry.T, mid + (J[:, 17] - mid) @ Ry.T
    d2 = uw.chain_wraps(pose=P, rest=REST, joints=J)["spine"]
    bad2 = d2["wrapped"] | d2["candy"]
    assert not d2["guard_ok"][bad2].any()
    lab_flag = uw.unwrap_runs(d2, uw.UnwrapCfg())
    lab_gate = uw.unwrap_runs(d2, uw.UnwrapCfg(guard_mode="gate"))
    assert lab_flag[bad2].all() and not lab_gate[bad2].any()


def test_split_spine_keeps_spine3_world_and_checks_limits():
    clean = _clean_pose(6) * 0.2
    P = clean.copy()
    a = uw._unit(REST[uw.SPINE2] - REST[uw.SPINE1])
    R = uw.rotmats(P)
    R[20:40, uw.SPINE1] = R[20:40, uw.SPINE1] @ uw.axis_rot(a, np.full(20, 50.0))
    P[:, uw.SPINE1] = uw.rotvecs(R[:, uw.SPINE1])
    core = np.zeros(T, bool)
    core[20:40] = True
    out, rep = uw.split_spine(P, REST, None, core)
    assert rep["runs"][0]["accepted"], rep
    G0, X0 = uw.fk22(uw.rotmats(P), REST)
    G1, X1 = uw.fk22(uw.rotmats(out), REST)
    assert np.abs(G0[:, uw.SPINE3] - G1[:, uw.SPINE3]).max() < 1e-9
    assert np.linalg.norm(X1 - X0, axis=-1).max() <= 0.03
    tw = uw.chain_twists(uw.rotmats(out), REST, uw.CHAINS["spine"])
    assert np.abs(tw[20:40]).max() <= 35.0
    # 150 deg of real over-twist cannot be split under 35 deg per joint: refused, unchanged
    R2 = uw.rotmats(clean)
    R2[20:40, uw.SPINE1] = R2[20:40, uw.SPINE1] @ uw.axis_rot(a, np.full(20, 150.0))
    P2 = clean.copy()
    P2[:, uw.SPINE1] = uw.rotvecs(R2[:, uw.SPINE1])
    out2, rep2 = uw.split_spine(P2, REST, None, core)
    assert not rep2["runs"][0]["accepted"] and np.array_equal(out2, P2)


CA = PACKAGE_ROOT / "work" / "carlos_aline" / "smplx_rf2_noslip" / "follow.npz"
MODEL = PACKAGE_ROOT.parent / "mamma" / "data" / "body_models" / "smplx_locked_head"


@pytest.mark.skipif(not (CA.exists() and MODEL.exists()), reason="needs work/carlos_aline + the SMPL-X model (local data)")
def test_carlos_aline_wraps_regression():
    """R16 on Aline at 126.13-127.87 and 154.47-156.50 s (DESIGN 6.3 / 8): joints within 1 cm, wraps 0 after the unwrap"""
    from dancecap import stitch
    with np.load(CA, allow_pickle=True) as z:
        fit = {k: z[k] for k in z.files}
    rest = stitch.rest_joints(fit["betas"])
    before = uw.chain_wraps(fit, rest=rest)
    new, rep = uw.unwrap_fit(fit, rest=rest, split=False)
    after = uw.chain_wraps(new, rest=rest)
    t = fit["t"]
    for a, b in ((126.13, 127.87), (154.47, 156.50)):
        m = (t >= a - 1e-6) & (t <= b + 1e-6)
        assert before["spine"]["wrapped"][m].sum() >= 5
        assert np.linalg.norm(new["joints"][m] - fit["joints"][m], axis=-1).max() < 0.01
        assert sum(int(after[c]["wrapped"][m].sum()) for c in after) == 0
        assert np.abs(after["spine"]["twists"][m]).max() < 60.0
    assert rep["chains"]["spine"]["wrapped_frames_after"] == 0
