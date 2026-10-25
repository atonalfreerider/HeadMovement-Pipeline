"""dancecap.magnet: the sticky state machine (engage after T_on within R_in, stay within R_out, switch by a margin, release after T_off,
dwell, no site shared by both hands), the linear weights (ramps and a cross-fade summing to 1), the two-bone IK (the palm on the target,
the bones keep their lengths, the elbow on the circle nearest the original elbow, w = 0 is the original, out of reach w is scaled down),
and the install script (LF-only bash; --check changes nothing; the first install backs up once; refused while work/unity.lock is held)."""
import hashlib
import json
import shutil
import subprocess

import numpy as np
import pytest

from dancecap import magnet as mg

CFG = mg.MagnetCfg(r_in=0.12, r_out=0.22, t_on=0.1, t_off=0.1, dwell=0.2, margin=0.04, t_blend=0.2)   # 3 / 3 / 6 / 6 frames at 30 Hz


def _D(T, S, far=1.0):
    return np.full((T, S), far)


def test_engage_hold_release():
    D = _D(60, 2)
    D[10:40, 0] = 0.05          # within R_in from frame 10
    D[40:60, 0] = 0.30          # beyond R_out from frame 40
    att = mg.sticky_timeline([D], CFG)[0]
    assert len(att) == 1 and att[0]["site"] == 0 and att[0]["via"] == "engage"
    assert att[0]["k0"] == 12                                     # engaged after 3 frames (t_on) within R_in
    assert att[0]["k1"] == 40                                     # w back to 0 at the first frame beyond R_out
    D[40:60, 0] = 0.18                                            # between R_in and R_out: stays attached (hysteresis)
    att = mg.sticky_timeline([D], CFG)[0]
    assert len(att) == 1 and att[0]["k1"] == 59


def test_short_visit_never_engages_and_dwell_holds():
    D = _D(40, 1)
    D[5:7, 0] = 0.05                                              # 2 frames < t_on
    assert mg.sticky_timeline([D], CFG)[0] == []
    D = _D(40, 1)
    D[5:10, 0] = 0.05                                             # engaged at 7, then gone at 10: the dwell keeps it until 7 + 6
    D[10:40, 0] = 0.25                                            # (beyond r_out but within r_out + dwell_pull_m)
    att = mg.sticky_timeline([D], CFG)[0]
    assert att[0]["k0"] == 7 and att[0]["k1"] == 13                # held the dwell (6 frames), then released
    D[10:40, 0] = 0.40                                            # far beyond r_out + dwell_pull_m: released without waiting for the dwell
    att = mg.sticky_timeline([D], CFG)[0]
    assert att[0]["k1"] == 10 and len(att) == 1


def test_switch_needs_margin_and_crossfades():
    D = _D(80, 2)
    D[0:80, 0] = 0.06
    D[30:80, 1] = 0.04                                            # nearer, but by less than the margin: no switch
    att = mg.sticky_timeline([D], CFG)[0]
    assert [a["site"] for a in att] == [0]
    D[30:80, 1] = 0.01                                            # nearer by 5 cm > margin: switch after t_on
    att = mg.sticky_timeline([D], CFG)[0]
    assert [a["site"] for a in att] == [0, 1] and att[1]["via"] == "switch"
    W = mg.attachment_weights(att, 80, 6)
    ks = att[1]["k0"]
    assert np.allclose(W[ks:ks + 7].sum(1), 1.0)                  # the cross-fade sums to 1 ...
    assert np.allclose(np.diff(W[ks:ks + 7, 1]), 1 / 6)           # ... and is linear


def test_two_hands_never_share_a_site():
    D = _D(40, 2)
    D[:, 0] = 0.05
    D2 = D.copy()
    D2[:, 1] = 0.10
    a1, a2 = mg.sticky_timeline([D, D2], CFG)
    assert [a["site"] for a in a1] == [0] and [a["site"] for a in a2] == [1]


def test_weights_linear_ramps():
    W = mg.attachment_weights([{"site": 0, "k0": 10, "k1": 40}], 50, 6)[:, 0]
    assert W[10] == 0 and W[16] == 1 and W[34] == 1 and W[40] == 0
    assert np.allclose(np.diff(W[10:17]), 1 / 6) and np.allclose(np.diff(W[34:41]), -1 / 6)
    W = mg.attachment_weights([{"site": 0, "k0": 0, "k1": 6}], 10, 6)[:, 0]   # shorter than two ramps: a triangle
    assert W.max() == pytest.approx(0.5)


def _arm(n=1, bend_deg=60.0, side="right"):
    """a straight-down upper arm, the forearm bent forward; identity collar / wrist frames"""
    L1, L2 = 0.28, 0.25
    S = np.tile([0.0, 1.4, 0.0], (n, 1))
    E0 = S + [0, -L1, 0]
    b = np.radians(bend_deg)
    W0 = E0 + L2 * np.array([0, -np.cos(b), np.sin(b)])
    G14 = np.tile(np.eye(3), (n, 1, 1))
    up = np.array([0, -1.0, 0])
    f = (W0[0] - E0[0]) / L2
    ax, ang = mg.rot_between(up[None], f[None])
    G17 = np.tile(np.eye(3), (n, 1, 1)) @ mg.rot_axis_angle(np.array([[0, 0, 1.0]]), np.array([np.pi / 2]))
    G19 = mg.rot_axis_angle(ax, ang) @ G17
    G21 = G19.copy()
    pc0 = W0 + [0, -0.08, 0.02]
    pn0 = np.tile([0, 0, 1.0], (n, 1))
    return S, E0, W0, G14, G17, G19, G21, pc0, pn0, L1, L2


def test_ik_reaches_keeps_bones_and_swivel():
    S, E0, W0, G14, G17, G19, G21, pc0, pn0, L1, L2 = _arm()
    X = pc0 + [0.05, 0.06, 0.03]
    n_site = np.array([[0, 0, -1.0]])
    ik = mg.arm_ik(S, E0, W0, G14, G17, G19, G21, pc0, pn0, X, n_site, np.ones(1), "right", mg.MagnetCfg())
    assert ik["scale"][0] == 1.0
    assert np.linalg.norm(ik["E"] - S, axis=1)[0] == pytest.approx(L1, abs=1e-9)
    assert np.linalg.norm(ik["W"] - ik["E"], axis=1)[0] == pytest.approx(L2, abs=1e-9)
    G21n = ik["R21"]
    G17n = G14 @ ik["R17"]
    G19n = G17n @ ik["R19"]
    c_loc = np.einsum("nji,nj->ni", G21, pc0 - W0)
    pc_new = ik["W"] + np.einsum("nij,nj->ni", G19n @ G21n, c_loc)
    assert np.linalg.norm(pc_new - X) < 1e-9                      # the palm centre on the target
    # the elbow on the IK circle nearest the original elbow: no other point of the circle is nearer
    u = mg._unit(ik["W"] - S)[0]
    c = S[0] + ((ik["E"] - S)[0] @ u) * u
    r = np.linalg.norm(ik["E"][0] - c)
    e1 = mg._unit(np.cross(u, [1.0, 0, 0]))
    e2 = np.cross(u, e1)
    circ = c + r * (np.cos(np.linspace(0, 2 * np.pi, 360))[:, None] * e1 + np.sin(np.linspace(0, 2 * np.pi, 360))[:, None] * e2)
    assert np.linalg.norm(ik["E"][0] - E0[0]) <= np.linalg.norm(circ - E0[0], axis=1).min() + 1e-6


def test_ik_zero_weight_is_the_original_and_out_of_reach_scales_w():
    S, E0, W0, G14, G17, G19, G21, pc0, pn0, L1, L2 = _arm()
    ik = mg.arm_ik(S, E0, W0, G14, G17, G19, G21, pc0, pn0, pc0.copy(), np.array([[0, 0, -1.0]]), np.zeros(1), "right",
                   mg.MagnetCfg())                                # w = 0: the target is the original palm (X is already weighted)
    assert np.allclose(ik["E"], E0, atol=1e-9) and np.allclose(ik["W"], W0, atol=1e-9)
    assert np.allclose(G14 @ ik["R17"], G17, atol=1e-9) and np.allclose(G14 @ ik["R17"] @ ik["R19"], G19, atol=1e-9)
    far = S + [0, 0, 1.5]                                          # 1.5 m away: unreachable
    ik = mg.arm_ik(S, E0, W0, G14, G17, G19, G21, pc0, pn0, far, np.array([[0, 0, -1.0]]), np.ones(1), "right", mg.MagnetCfg())
    assert 0.0 <= ik["scale"][0] < 1.0
    assert ik["flex_deg"][0] >= mg.MagnetCfg().flex_min_deg - 1e-6   # no hyperextension


def test_film_time():
    assert mg.film_of(90.0) == 90.0 and mg.film_of(282.8) == pytest.approx(290.0)
    assert mg.dance_of(290.0) == pytest.approx(282.8) and mg.dance_of(32.0) == 32.0


@pytest.mark.skipif(shutil.which("bash") is None, reason="no bash")
def test_install_script_check_backup_lock(tmp_path):
    files = mg.MOTION_FILES
    cap = tmp_path / "cap"
    cap.mkdir()
    for f in files:
        (cap / f).write_bytes(b"OLD" + f.encode() * 40)
    (cap / "capture.json").write_text("{}", encoding="utf-8")
    u = tmp_path / "unity_motion"
    u.mkdir()
    for f in files:
        (u / f).write_bytes(b"NEW" + f.encode() * 40)
    (u / "unity_motion.json").write_text(json.dumps({"sha256": {f: hashlib.sha256((u / f).read_bytes()).hexdigest() for f in files}}),
                                         encoding="utf-8")
    p = mg.write_install_script(u)
    assert b"\r" not in p.read_bytes() and b"dancecap.magnet" in p.read_bytes()
    env = {"HM_CAPTURE_DIR": str(cap).replace("\\", "/"), "HM_UNITY_LOCK": str(tmp_path / "unity.lock").replace("\\", "/"),
           "PATH": __import__("os").environ["PATH"]}                # forward slashes: sha256sum escapes a name with a backslash
    old = {f: (cap / f).read_bytes() for f in files}
    r = subprocess.run(["bash", str(p).replace("\\", "/"), "--check"], env=env, capture_output=True, text=True)
    assert r.returncode == 0 and "check ok" in r.stdout and {f: (cap / f).read_bytes() for f in files} == old
    (tmp_path / "unity.lock").mkdir()
    (tmp_path / "unity.lock" / "owner.txt").write_text("someone recording\n", encoding="utf-8")
    r = subprocess.run(["bash", str(p).replace("\\", "/")], env=env, capture_output=True, text=True)
    assert r.returncode != 0 and "held" in r.stdout and {f: (cap / f).read_bytes() for f in files} == old
    shutil.rmtree(tmp_path / "unity.lock")
    r = subprocess.run(["bash", str(p).replace("\\", "/")], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert all((cap / f).read_bytes() == (u / f).read_bytes() for f in files)
    assert all((u / "backup_installed_motion" / f).read_bytes() == old[f] for f in files)
    r = subprocess.run(["bash", str(p).replace("\\", "/"), "--restore"], env=env, capture_output=True, text=True)
    assert r.returncode == 0 and {f: (cap / f).read_bytes() for f in files} == old


# ---------------------------------------------------------------------------------------------- v2: strength, target clamp, blips, elbow

def test_strength_makes_a_far_strong_site_win_and_reach_caps_it():
    cfg = mg.MagnetCfg(r_in=0.10, r_out=0.22, t_on=0.1, t_off=0.1, dwell=0.2, margin=0.04, t_blend=0.2)
    names = ["upper_back_L", "forearm_L"]
    st = mg.strength_vector(cfg, "right", names)
    assert st.tolist() == [3.0, 0.8] and mg.strength_vector(cfg, "left", ["hand_R", "tricep_L"]).tolist() == [2.5, 1.0]
    D = _D(40, 2)
    D[:, 0] = 0.24                                                # her back 24 cm: 8 cm effective, within R_in
    D[:, 1] = 0.10                                                # her forearm 10 cm: 12.5 cm effective, not within R_in
    att = mg.sticky_timeline([mg.effective_distances(D, st, cfg, "right")], cfg)[0]
    assert [a["site"] for a in att] == [0]                        # the closed hold goes to her upper-left back
    D[20:, 0] = 0.45                                              # beyond reach_max (right 0.40 m): released although 15 cm effective
    att = mg.sticky_timeline([mg.effective_distances(D, st, cfg, "right")], cfg)[0]
    assert att[0]["k1"] == 20


def test_target_clamp_holds_a_glitch_and_limits_speed():
    cfg = mg.MagnetCfg()
    T = 60
    pc0 = np.zeros((T, 3))
    pc0[:, 0] = np.arange(T) * 0.01                               # his palm moves 1 cm / frame
    tgt = pc0 + [0, 0.05, 0]
    tgt[30:33] += [0, 0.40, 0]                                    # her site jumps 40 cm for 3 frames (tracking glitch)
    nrm = np.tile([0, 0, 1.0], (T, 1))
    out, nout, info = mg.clamp_target(tgt, nrm, pc0, 0, T - 1, cfg)
    assert info["frames_held"] > 0
    assert np.linalg.norm(out - pc0, axis=1).max() < 0.05 + 1e-9  # never copied: the offset to his palm is held through the glitch
    assert np.allclose(out[45:], tgt[45:])                        # settled: back on the site
    tgt2 = pc0.copy()
    tgt2[20:] += [0, 0.15, 0]                                     # a 15 cm step (below jump_m): followed at <= his step + 0.6 m/s
    out2, _, info2 = mg.clamp_target(tgt2, nrm, pc0, 0, T - 1, cfg)
    step = np.linalg.norm(np.diff(out2, axis=0), axis=1)
    lim = np.maximum(np.linalg.norm(np.diff(pc0, axis=0), axis=1) + 0.6 / 30, cfg.target_floor_m)
    assert info2["frames_clamped"] > 0 and (step <= lim + 1e-9).all()


def test_blips_are_dropped_but_switch_chains_kept():
    cfg = mg.MagnetCfg(min_hold_s=1.5)
    att = [{"site": 0, "k0": 0, "k1": 30, "via": "engage"},           # 1.0 s alone: dropped
           {"site": 1, "k0": 100, "k1": 125, "via": "engage"},        # 0.83 s, but switched out of: kept
           {"site": 2, "k0": 120, "k1": 300, "via": "switch"},
           {"site": 0, "k0": 400, "k1": 460, "via": "engage"}]        # 2 s: kept
    assert [a["site"] for a in mg.drop_blips(att, cfg)] == [1, 2, 0]


def test_elbow_choice_keeps_its_side_where_the_original_is_on_the_axis():
    n = 5
    u = np.tile([0, -1.0, 0], (n, 1))
    C0 = np.zeros((n, 3))
    rr = np.full(n, 0.1)
    q = np.array([[1.0, 0, 0]] * 2 + [[-1.0, 0, 0]] + [[1.0, 0, 0]] * 2)   # the per-frame choice flips at frame 2
    E0 = np.zeros((n, 3))                                          # the original elbow on the shoulder-wrist line: ambiguous
    E0[:2] = [0.1, 0, 0]
    out = mg._elbow_dirs(q, u, C0, rr, E0, np.r_[False, np.ones(n - 1, bool)], None)
    assert np.allclose(out, [1.0, 0, 0])                           # continuity: no flip
