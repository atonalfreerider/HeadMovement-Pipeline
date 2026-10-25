"""dancecap.sides.splice: the entry offset ramp is C1 and lands on the base, the exit slerp takes the nearest rotation branch, the
exit rule picks the first ALL-segment agreement (not the pelvis one), defects / scores, and (with local data) the recap's 199-s exit
at 200.10 +- 0.1 s."""
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rot

from dancecap.sides import splice as sp
from dancecap.sides import unwrap as uw
from dancecap.take import PACKAGE_ROOT

REST = np.array([
    [0, 0.9, 0], [0.06, 0.81, 0], [-0.06, 0.81, 0], [0, 1.01, -0.01], [0.10, 0.43, 0], [-0.10, 0.43, 0],
    [0, 1.14, 0.01], [0.09, 0.03, -0.04], [-0.09, 0.03, -0.04], [0, 1.19, 0.03], [0.12, -0.02, 0.08], [-0.12, -0.02, 0.08],
    [0, 1.40, 0], [0.07, 1.31, 0], [-0.07, 1.31, 0], [0, 1.50, 0.05], [0.17, 1.35, -0.01], [-0.17, 1.35, -0.01],
    [0.43, 1.33, -0.03], [-0.43, 1.33, -0.03], [0.68, 1.34, -0.03], [-0.68, 1.34, -0.03]], float)
SIZES = {"global_orient": 1, "body_pose": 21, "jaw_pose": 1, "leye_pose": 1, "reye_pose": 1, "left_hand_pose": 15,
         "right_hand_pose": 15}


def _fk(pose165, transl, betas):
    P = np.asarray(pose165, float).reshape(len(pose165), 55, 3)[:, :22]
    return uw.fk22(uw.rotmats(P), REST, transl)[1]


def _fit(P22: np.ndarray, transl: np.ndarray, hands=None) -> dict:
    T = len(P22)
    d = {"global_orient": P22[:, 0].copy(), "body_pose": P22[:, 1:22].reshape(T, 63).copy(), "transl": transl.copy(),
         "betas": np.zeros(10), "t": np.arange(T) / 30.0}
    for k in ("jaw_pose", "leye_pose", "reye_pose"):
        d[k] = np.zeros((T, 3))
    for k in ("left_hand_pose", "right_hand_pose"):
        d[k] = np.zeros((T, 45)) if hands is None else hands.copy()
    d["stance"] = np.zeros((T, 2), bool)
    d["joints"] = uw.fk22(uw.rotmats(P22), REST, transl)[1]
    return d


def _motion(T=90, spin_deg_s=360.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(T) / 30.0
    P = np.zeros((T, 22, 3))
    for j in range(1, 22):
        a, f, ph = rng.normal(0, 0.1, 3), rng.uniform(0.3, 1.0, 3), rng.uniform(0, 6.28, 3)
        P[:, j] = a[None] * np.sin(2 * np.pi * f[None] * t[:, None] + ph[None])
    yaw = np.radians(spin_deg_s) * t
    P[:, 0] = (Rot.from_rotvec(np.c_[np.zeros(T), yaw, np.zeros(T)]) * Rot.from_rotvec([0.05, 0, 0.03])).as_rotvec()
    tr = np.c_[0.2 * np.sin(t), np.zeros(T), 0.1 * t]
    return P, tr


def _yawed(P, tr, deg):
    """the same body turned about the vertical through its pelvis by deg"""
    out = P.copy()
    out[:, 0] = (Rot.from_rotvec([0, np.radians(deg), 0]) * Rot.from_rotvec(P[:, 0])).as_rotvec()
    return out


def test_entry_ramp_is_c1_and_spans_base_to_alt():
    r = sp.entry_ramp(7)
    assert r[0] == 1.0 and abs(r[-1]) < 1e-12 and np.all(np.diff(r) < 0)
    fine = sp.entry_ramp(2001)
    d = np.diff(fine) * 2000                     # derivative per unit ramp length
    assert abs(d[0]) < 2e-3 and abs(d[-1]) < 2e-3 and abs(d).max() > 1.5   # zero slope at both ends (C1), pi/2 in the middle


def test_entry_offset_ramp_on_a_splice():
    P, tr = _motion()
    base = {"follow": _fit(P, tr)}
    off = 20                                       # the chunk starts at grid frame 20
    Pa, tra = _yawed(P, tr, -20.0)[off:], tr[off:] + np.array([0.01, 0, -0.02])
    alt = {"follow": _fit(Pa, tra)}
    cfg = sp.SpliceCfg()
    out, info = sp.splice(base, alt, off, off, 60, cfg, fk=_fk)
    o = out["follow"]
    yb = sp.yaw_of(Rot.from_rotvec(base["follow"]["global_orient"]).as_matrix())
    yo = sp.yaw_of(Rot.from_rotvec(o["global_orient"]).as_matrix())
    dy = sp._wrap(yo - yb)
    # the entry frame is the base exactly, then the offset eases out with zero slope at both ends
    assert np.abs(o["joints"][off] - base["follow"]["joints"][off]).max() < 1e-9
    assert abs(dy[off]) < 1e-6 and abs(dy[off + 6] + 20.0) < 0.5
    steps = np.abs(np.diff(dy[off - 1:off + 8]))
    assert steps[0] < 1e-6 and steps[1] < 0.07 * 20 + 0.1        # no 20-deg switch at the seam: the first step is the ramp's smallest
    assert steps.max() < 0.26 * 20 + 0.2                          # the cosine's largest step (pi/2 / 6 of the offset)
    # the pure-alternative frames are the chunk's own (joints included); nothing outside the splice changed
    assert np.abs(o["joints"][off + 7:60] - alt["follow"]["joints"][7:40]).max() < 1e-9
    assert np.array_equal(o["global_orient"][:off], base["follow"]["global_orient"][:off])
    assert np.array_equal(o["global_orient"][66:], base["follow"]["global_orient"][66:])
    assert info["src"][0] == sp.SRC_ENTRY and info["src"][10] == sp.SRC_ALT and info["src"][-1] == sp.SRC_EXIT
    assert abs(info["roles"]["follow"]["entry_offset_yaw_deg"] - 20.0) < 0.5


def test_window_round_trip():
    """the saved window (spin_*.npz layout) written back into the base reproduces the splice exactly"""
    P, tr = _motion(seed=3)
    base = {r: _fit(P, tr) for r in ("lead", "follow")}
    alt = {r: _fit(_yawed(P, tr, 15.0)[10:], tr[10:]) for r in ("lead", "follow")}
    out, info = sp.splice(base, alt, 10, 10, 50, sp.SpliceCfg(), fk=_fk, alt_name="x0010")
    z = sp.splice_window_arrays(out, info)
    z["alt_name"] = np.array("x0010")
    for r in ("lead", "follow"):
        back = sp.apply_window(base[r], z, r)
        for k in (*sp.POSE_KEYS, "transl", "joints"):
            assert np.array_equal(back[k], out[r][k]), k


def test_exit_slerp_takes_the_nearest_branch():
    """the alternative ends 10 deg off the base, written on the 'other' axis-angle branch (angle - 2 pi) and one full turn ahead in
    its unwrapped heading: the exit slerp must turn the 10 deg, never a full turn"""
    P, tr = _motion(spin_deg_s=0.0)
    base = {"follow": _fit(P, tr)}
    Pa = _yawed(P, tr, 10.0)
    go = Pa[:, 0]
    ang = np.linalg.norm(go, axis=1, keepdims=True)
    Pa[:, 0] = go / ang * (ang - 2 * np.pi)                      # the same rotations, the far branch
    alt = {"follow": _fit(Pa, tr)}
    cfg = sp.SpliceCfg()
    out, info = sp.splice(base, alt, 0, 10, 50, cfg, fk=_fk)
    Rb = Rot.from_rotvec(base["follow"]["global_orient"])
    Ro = Rot.from_rotvec(out["follow"]["global_orient"])
    gap = np.degrees((Rb.inv() * Ro).magnitude())
    assert gap[49] > 9.0 and gap[56:].max() < 1e-9
    ex = gap[50:56]
    assert np.all(np.diff(ex) < 0) and ex.max() < 10.0 + 1e-6      # monotone 10 -> 0, never round the long way
    # the parameter track has no 2 pi jump either
    v = out["follow"]["global_orient"]
    assert np.linalg.norm(np.diff(v[5:60], axis=0), axis=1).max() < 0.5


def test_exit_rule_picks_all_segment_agreement_not_pelvis():
    T = 80
    P = np.zeros((T, 22, 3))
    tr = np.zeros((T, 3))
    base = _fit(P, tr)["joints"]
    # alternative: pelvis off by 40 deg until frame 20, the left foot wound 60 deg (relative to the pelvis) until frame 32
    Pa = P.copy()
    off_pel = np.where(np.arange(T) < 20, 40.0, 0.0)
    off_foot = np.where(np.arange(T) < 32, 60.0, 0.0)
    Pa[:, 0] = np.c_[np.zeros(T), np.radians(off_pel), np.zeros(T)]
    # turn the left ankle about the vertical (its local frame: the leg is straight, so local y = world y up to the pelvis yaw)
    Pa[:, 7] = np.c_[np.zeros(T), np.radians(off_foot), np.zeros(T)]
    alt = _fit(Pa, tr)["joints"]
    alt[:5] = np.nan                                             # the chunk starts at frame 5
    cfg = sp.SpliceCfg()
    ex = sp.choose_exit_all_segments({"follow": base}, {"follow": alt}, 5, cfg)
    assert ex["first_pelvis_only"] == 20
    assert ex["k_exit"] == 32
    # a second dancer that agrees late holds the exit back further
    alt2 = alt.copy()
    alt2[:40, :22] += np.array([0.08, 0, 0])                     # 8 cm off until frame 40
    ex2 = sp.choose_exit_all_segments({"follow": base, "lead": base}, {"follow": alt, "lead": alt2}, 5, cfg)
    assert ex2["k_exit"] == 40


def test_spin_defects_and_scores():
    T = 60
    psi_b = np.zeros(T)
    psi_s = np.zeros(T)
    psi_s[10:12] = 90.0                                          # 2 frames: too short
    psi_s[30:36] = 120.0                                         # a real defect
    face = np.zeros(T, int)
    face[45] = 2
    d = sp.spin_defects(psi_b, psi_s, face)
    assert d["runs"] == [(30, 35), (45, 45)]
    # face bits: 4 cameras around her; she faces camera 0 (azimuth 0); the base says she faces away (180)
    az = np.tile(np.array([0.0, 90.0, 180.0, -90.0]), (T, 1))
    lab = np.tile(np.array([1, 0, -1, 0]), (T, 1))
    cands = {"base": {"head": np.full(T, 180.0)}, "alt": {"head": np.zeros(T)}}
    res = sp.score_alternatives(cands, [("head", sp.face_bits_ll(lab, az, 0.9), 1.0)], runs=[(30, 35)])
    assert res["runs"][0]["best"] == "alt" and res["runs"][0]["margin"] > 10
    assert (res["best"] == 1).all()
    assert (sp.face_contradictions(lab, az, np.full(T, 180.0)) == 2).all()
    assert (sp.face_contradictions(lab, az, np.zeros(T)) == 0).all()
    vm = sp.von_mises_ll(np.full(T, 10.0), 5.0)
    assert vm(np.full(T, 10.0)).max() == 0.0 and vm(np.full(T, 190.0)).max() < -9.9


REC = PACKAGE_ROOT / "work" / "larissa_kadu_recap"
X5966 = PACKAGE_ROOT / "work" / "larissa_kadu_recap_x5966" / "smplx_rf2_noslip"


@pytest.mark.skipif(not ((REC / "smplx_rf2_noslip" / "follow.npz").exists() and (X5966 / "follow.npz").exists()),
                    reason="needs the recap fit + chunk x5966 (local data)")
def test_recap_199_exit_regression():
    """F1: the x5966 splice hands back at 200.10 +- 0.1 s (all segments), not at the pelvis agreement (199.6-199.85 s)"""
    J, A = {}, {}
    st = json.loads((REC / "stitch.json").read_text(encoding="utf-8"))
    off = int(st["grid"]["frame_offset"]["x5966"])
    for r in ("lead", "follow"):
        with np.load(REC / "smplx_rf2_noslip" / f"{r}.npz") as z:
            J[r] = np.asarray(z["joints"], float)
        with np.load(X5966 / f"{r}.npz") as z:
            a = np.full_like(J[r], np.nan)
            ja = np.asarray(z["joints"], float)
            a[off:off + len(ja)] = ja
            A[r] = a
    ex = sp.choose_exit_all_segments(J, A, int(round(199.57 * 30)), sp.SpliceCfg())
    assert abs(ex["k_exit"] / 30.0 - 200.10) <= 0.1 + 1e-9
    assert ex["first_pelvis_only"] / 30.0 < 199.9
