"""Physics stage tests: spline derivatives on analytic signals, body model, impulses, polygon geometry, the
partner-force LP on constructed scenes, and gate G6 on work/synthetic.

    .venv-nopo4d\\Scripts\\python.exe -m pytest tests/test_physics.py -q      (from dancecap/)
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from dancecap import physics as ph
from dancecap import physics_body as pb
from dancecap.physics_spline import Smoother

SYN = Path(__file__).resolve().parents[1] / "work" / "synthetic"
needs_syn = pytest.mark.skipif(not (SYN / "smplx" / "lead.npz").exists(), reason="run: python -m dancecap synthetic")


# ----------------------------------------------------------------------------------------------- spline

def _jittered(n=600, fps=30.0, jit=0.004, seed=0):
    rng = np.random.default_rng(seed)
    return np.sort(np.arange(n) / fps + rng.uniform(-jit, jit, n))


def test_spline_reproduces_quadratics_exactly():
    t = _jittered()
    S = Smoother(t, 5.0)
    y = 3 + 0.5 * t - 4.905 * t ** 2  # ballistic COM: a = -g everywhere, including the ends
    assert np.abs(S(y, 0) - y).max() < 1e-6
    assert np.abs(S(y, 1) - (0.5 - 9.81 * t)).max() < 1e-5
    assert np.abs(S(y, 2) + 9.81).max() < 1e-4


def test_spline_derivatives_of_slow_sine():
    t = _jittered()
    S = Smoother(t, 5.0)
    w = 2 * math.pi * 1.0
    y = np.sin(w * t)
    inner = slice(15, -15)
    assert np.abs(S(y, 1) - w * np.cos(w * t))[inner].max() < 0.005 * w
    assert np.abs(S(y, 2) + w * w * y)[inner].max() < 0.005 * w * w
    assert np.abs(S(y, 3) + w ** 3 * np.cos(w * t))[inner].max() < 0.02 * w ** 3


def test_spline_cutoff_response():
    t = np.arange(900) / 30.0
    S = Smoother(t, 5.0)
    for f, lo, hi in ((1.0, 0.99, 1.01), (5.0, 0.45, 0.55), (10.0, 0.0, 0.05)):
        y = np.sin(2 * math.pi * f * t)
        X = np.stack([np.sin(2 * math.pi * f * t), np.cos(2 * math.pi * f * t)], 1)[200:-200]
        g = float(np.linalg.norm(np.linalg.lstsq(X, S(y)[200:-200], rcond=None)[0]))  # output amplitude
        assert lo <= g <= hi, (f, g)
        assert abs(float(S.gain(f)) - g) < 0.05


def test_spline_noise_gain_matches_monte_carlo():
    t = _jittered(300)
    S = Smoother(t, 5.0)
    rng = np.random.default_rng(1)
    e = rng.normal(0, 1, (300, 400))
    emp = S(e, 2).std(1)
    assert np.allclose(emp[30:-30], S.noise_gain(2)[30:-30], rtol=0.15)


def test_spline_evaluates_between_samples():
    t = _jittered(200)
    S = Smoother(t, 5.0)
    y = 2.0 * t ** 2
    x = np.linspace(t[0], t[-1], 57)
    assert np.allclose(S(y, 1, x), 4.0 * x, atol=1e-5)


# ----------------------------------------------------------------------------------------------- geometry

def test_mesh_volume_com_cube():
    v = np.array([[x, y, z] for x in (0, 1) for y in (0, 2) for z in (0, 3)], float) + [1, 1, 1]
    from scipy.spatial import ConvexHull
    h = ConvexHull(v)
    f = h.simplices.copy()
    c0 = v.mean(0)
    for k, tri in enumerate(f):  # orient outward
        n = np.cross(v[tri[1]] - v[tri[0]], v[tri[2]] - v[tri[0]])
        if n @ (v[tri[0]] - c0) < 0:
            f[k] = tri[[0, 2, 1]]
    vol, com = pb.mesh_volume_com(v, f)
    assert abs(vol - 6.0) < 1e-9 and np.allclose(com, [1.5, 2.0, 2.5])


def test_signed_distance_polygon():
    sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1.0]])
    assert abs(ph.signed_distance_polygon(np.array([0.5, 0.5]), sq) - 0.5) < 1e-12
    assert abs(ph.signed_distance_polygon(np.array([1.5, 0.5]), sq) + 0.5) < 1e-12
    assert abs(ph.signed_distance_polygon(np.array([0.5, 0.9]), sq[::-1]) - 0.1) < 1e-12
    pad = np.vstack([sq, np.full((3, 2), np.nan)])
    assert abs(ph.signed_distance_polygon(np.array([0.5, 0.2]), pad) - 0.2) < 1e-12
    assert math.isnan(ph.signed_distance_polygon(np.zeros(2), np.full((4, 2), np.nan)))
    h = ph.hull2d(np.random.default_rng(0).uniform(-1, 1, (500, 2)), 8)
    assert len(h) <= 8


def test_axis_angle_matches_scipy():
    from scipy.spatial.transform import Rotation
    rv = np.random.default_rng(3).normal(0, 1.2, (50, 3))
    assert np.allclose(pb.axis_angle_to_matrix(rv), Rotation.from_rotvec(rv).as_matrix(), atol=1e-12)


# ----------------------------------------------------------------------------------------------- body model

@needs_syn
def test_body_model_on_synthetic_lead():
    d = ph.load_npz(SYN / "smplx" / "lead.npz")
    body = pb.segment_masses(pb.load_body(d))
    assert 60 <= body.mass <= 95
    assert abs(body.voxel_volume / body.volume - 1) < 0.01          # voxels fill the closed mesh
    assert abs(body.elem_mass.sum() - body.mass) < 1e-6
    R, p = pb.forward_kinematics(body, pb.full_pose(d), d["transl"])
    J = np.einsum("tjab,jb->tja", R, body.j_rest) + p
    assert np.abs(J - d["joints"]).max() < 1e-5                    # FK == smplx joints
    c, _ = pb.element_world(body, R, p)
    com = (body.elem_mass[None, :, None] * c).sum(1) / body.mass
    mesh = pb.posed_mesh_com(body, R[::60], p[::60])
    assert np.linalg.norm(mesh - com[::60], axis=1).max() < 0.01   # rigid elements vs posed mesh
    dl = body.de_leva_check()["segments"]
    for k in ("head", "upper_arm", "forearm", "shank"):           # limb/head fractions near de Leva
        assert 0.75 < dl[k]["ratio"] < 1.25, (k, dl[k])
    fr = body.group_fractions()
    assert abs(fr["thigh_l"] - fr["thigh_r"]) < 0.01
    assert pb.load_body(d).mass == 0.0 and pb.segment_masses(pb.load_body(d), mass=70.0).mass == 70.0


# ----------------------------------------------------------------------------------------------- impulses

def _fake_dancer(t, y, mass):
    S = Smoother(t, 5.0)
    com_raw = np.stack([0.1 * t, y, np.zeros_like(t)], 1)
    return {"t": t, "S": S, "com_raw": com_raw, "com": S(com_raw), "mass": mass, "sd_pos": np.zeros(3)}


def test_beat_impulses_periodic_bounce():
    t = np.arange(0, 12, 1 / 30)
    f = 88 / 60
    res = {"lead": _fake_dancer(t, 1.0 + 0.03 * np.sin(2 * math.pi * f * t), 80.0),
           "follow": _fake_dancer(t, 0.9 + 0.02 * np.sin(2 * math.pi * f * t + 1), 60.0)}
    beats = np.arange(0, 12, 0.5 / f)  # eighths: 2 per bounce period
    cfg = ph.PhysicsConfig(n_boot=20)
    imp = ph.beat_impulses(res, beats, cfg)
    # over a whole bounce period the impulse is exactly m g dt (velocity returns)
    J = imp["impulse_lead"][:, 1]
    per = J[0::2][:-1] + J[1::2][:len(J[0::2]) - 1]
    dt2 = 1 / f
    assert np.allclose(per, 80 * ph.G * dt2, rtol=2e-3)
    assert np.allclose(imp["impulse_lead"][:, 0], 0, atol=1e-3)     # constant horizontal velocity
    cyc = ph.cycle_grf(imp, beats, 140.0, per=8)
    assert cyc["n_cycles"] >= 3 and cyc["max_abs_dev"] < 2e-3


# ----------------------------------------------------------------------------------------------- partner LP

def _scene(follow_in_flight: bool, tension: float, T: int = 3):
    """Lead at the origin (double support), follow 0.6 m away along +x, lead L hand on follow L hand."""
    mL, mF = 80.0, 60.0
    h = np.array([0.31, 1.2, 0.0])

    def joints(base, hand_l, far):
        J = np.tile(base, (T, 55, 1)).astype(float)
        J[:, [20] + list(range(25, 40))] = hand_l
        J[:, [21] + list(range(40, 55))] = far
        return J

    jl = joints([0.0, 1.0, 0.0], [0.30, 1.2, 0.0], [0.0, 1.0, -5.0])
    jl[:, 9] = [0.0, 1.3, 0.0]
    jf = joints([0.6, 1.2, 0.0], [0.32, 1.2, 0.0], [0.6, 1.2, 3.0])
    GL, GF = np.array([0.0, 1.0, 0.0]), np.array([0.6, 1.0, 0.0])
    feet = lambda cx: np.array([[cx + x, 0.0, z] for x in (-0.1, 0.1) for z in (-0.15, -0.05, 0.05, 0.15)])
    F_on_f = np.array([-tension, 0.3 * mF * ph.G if follow_in_flight else 20.0, 0.0])
    grf_f = np.zeros(3) if follow_in_flight else np.array([0.0, mF * ph.G - F_on_f[1], 0.0]) + [tension * 0.5, 0, 0]
    fnet_f = F_on_f + grf_f
    dl_f = np.cross(h - GF, F_on_f) + (np.cross(np.array([0.6, 0, 0]) - GF, grf_f) if not follow_in_flight else 0)
    grf_l = np.array([0.0, mL * ph.G + F_on_f[1], 0.0])
    fnet_l = grf_l - F_on_f
    dl_l = np.cross(np.zeros(3) - GL, grf_l) + np.cross(h - GL, -F_on_f)

    def dancer(G, J, fnet, dl, corners, contact):
        return {"t": np.arange(T) / 30.0, "joints": J, "com": np.tile(G, (T, 1)), "f_net": np.tile(fnet, (T, 1)),
                "f_net_band": np.full((T, 3), 3.0), "dL": np.tile(dl, (T, 1)), "dL_band": np.full((T, 3), 1.0),
                "foot_corners": [corners] * T, "contact": np.tile(contact, (T, 1))}

    return {"lead": dancer(GL, jl, fnet_l, dl_l, feet(0.0), [True, True]),
            "follow": dancer(GF, jf, fnet_f, dl_f, np.zeros((0, 3)) if follow_in_flight else feet(0.6),
                             [False, False] if follow_in_flight else [True, True])}


def test_partner_identifiable_when_follow_is_airborne():
    res = _scene(True, tension=50.0)
    p = ph.partner_interval(res, ph.PhysicsConfig())
    assert np.all(p["status"] == 1), p["status"]
    assert np.all(p["lo_robust"] <= 50.0) and np.all(p["hi_robust"] >= 50.0)
    assert np.all(p["hi_robust"] - p["lo_robust"] <= 40.0)


def test_partner_not_identifiable_in_double_support():
    res = _scene(False, tension=30.0)
    p = ph.partner_interval(res, ph.PhysicsConfig())
    assert np.all(p["status"] == 0), p["status"]
    assert np.all(p["lo"] <= 30.0) and np.all(p["hi"] >= 30.0)
    assert np.all(p["width"] > 100.0)  # statically indeterminate: wide


def test_partner_cap_bound_flight_is_never_identifiable():
    """Reviewer repro: a (false) flight frame whose F_net needs more partner force than the LP's box cap gets a
    narrow interval from the cap alone. It must come out cap-bound and 'inconsistent', with unbounded lp_range."""
    res = _scene(True, tension=50.0)
    res["follow"]["f_net"][:, 1] = 1000.0
    res["follow"]["f_net_band"][:] = 300.0  # noisy bands: feasible only with the partner force at the cap
    res["follow"]["dL_band"][:] = 200.0
    cfg = ph.PhysicsConfig()
    p = ph.partner_interval(res, cfg)
    assert np.all(p["status"] != 1), p["status"]
    assert np.all(p["cap_bound"]) and np.all(p["status"] == 3)
    assert not np.all(np.isfinite(np.stack([p["lo"], p["hi"]])))
    # the honest scene stays identifiable and is not cap-bound
    q = ph.partner_interval(_scene(True, tension=50.0), cfg)
    assert np.all(q["status"] == 1) and not q["cap_bound"].any()


def test_partner_no_contact():
    res = _scene(True, tension=0.0)
    res["follow"]["joints"][:, [20] + list(range(25, 40))] = [3.0, 1.2, 0.0]
    p = ph.partner_interval(res, ph.PhysicsConfig())
    assert np.all(p["status"] == 2) and np.all(p["lo"] == 0)


# ----------------------------------------------------------------------------------------------- gate G6

@needs_syn
def test_gate_g6_synthetic():
    npzs = {r: ph.load_npz(SYN / "smplx" / f"{r}.npz") for r in ph.ROLES}
    beats = ph.load_beat_times(ph.find_beats(SYN))
    res = ph.analyze(npzs, ph.PhysicsConfig(n_boot=40), beats, partner=False)
    g6 = ph.gate_g6(res)
    assert g6["passed"] and g6["n_cycles"] >= 5, g6
    for role in ph.ROLES:
        r = res[role]
        assert 60 <= r["mass"] <= 90
        assert 0.5 < np.mean(r["com"][:, 1]) / r["info"]["stature_m"] < 0.62
        inner = slice(15, -15)
        assert abs(np.mean(r["f_net"][inner, 1]) / (r["mass"] * ph.G) - 1) < 0.01
        assert r["info"]["fk_max_err_mm"] < 0.01


# ----------------------------------------------------------------------------------------------- robustness / io

@needs_syn
def test_nan_frames_are_masked_not_propagated():
    """One NaN transl frame and a 0.4 s NaN pose run: everything else stays finite, the frames are flagged, and
    contacts are still found (before: 0/720 finite f_net, contact.any() False, silent)."""
    d = ph.load_npz(SYN / "smplx" / "lead.npz")
    d["transl"] = d["transl"].copy()
    d["transl"][300] = np.nan
    d["body_pose"] = d["body_pose"].copy()
    d["body_pose"][500:512, 3] = np.nan
    cfg = ph.PhysicsConfig(n_boot=20)
    r = ph.analyze_dancer(d, cfg)
    f = r["f_net"][:, 1]
    assert np.isfinite(f).sum() >= len(f) - 12
    assert np.all(np.isnan(f[500:512])) and np.isfinite(f[300])
    assert r["flags"][300] & ph.FLAG_BITS["interpolated"] and r["flags"][505] & ph.FLAG_BITS["gap"]
    assert r["contact"].any(axis=0).all() and r["info"]["n_bad_frames"] == 13
    clean = ph.analyze_dancer(ph.load_npz(SYN / "smplx" / "lead.npz"), cfg)
    far = np.r_[0:280, 320:480, 540:len(f)]
    assert np.nanmax(np.abs(f[far] - clean["f_net"][far, 1])) < 5.0  # far from the bad frames: unchanged
    d["betas"] = np.full_like(d["betas"], np.nan)
    with pytest.raises(ValueError):
        ph.analyze_dancer(d, cfg)


def test_beats_sources(tmp_path):
    """timing/timing.json (dict, refined grid) is preferred and parsed; [[t, type]] still works; the audio-clock
    zouk-time-analysis.json is never used; physics.json records the beats file as beats_source."""
    (tmp_path / "zouk-time-analysis.json").write_text("[[100.0, 1]]")
    assert ph.find_beats(tmp_path) is None
    (tmp_path / "beats.json").write_text("[[1.0, 1], [1.5, 3]]")
    assert ph.find_beats(tmp_path) == tmp_path / "beats.json"
    assert np.allclose(ph.load_beat_times(tmp_path / "beats.json"), [1.0, 1.5])
    (tmp_path / "timing").mkdir()
    (tmp_path / "timing" / "timing.json").write_text(
        '{"beats": {"grid": [[1.01, 1, 0, 0], [1.49, 3, 0, 1]], "grid_columns": ["t", "type", "measure", "count"]}}')
    assert ph.find_beats(tmp_path) == tmp_path / "timing" / "timing.json"
    assert np.allclose(ph.load_beat_times(ph.find_beats(tmp_path)), [1.01, 1.49])


@needs_syn
def test_short_window_run_writes_beats_source(tmp_path):
    """A 6 s window (like the real take) through run(): physics.json strict-parses, records beats_source (no longer
    overwritten by the beats list), and G6 on the 1-2 whole measures inside stays within 3 %."""
    import json
    from dancecap import synthetic as syn
    w = syn.window(SYN, 6.0, 12.0)
    res = ph.run(SYN, ph.PhysicsConfig(n_boot=20), beats_path=SYN / "beats.json", out_dir=tmp_path, npzs=w["npz"],
                 plot=False)
    js = json.loads((tmp_path / "physics.json").read_text())
    assert js["beats_source"] == str(SYN / "beats.json") and isinstance(js["beats"], list) and js["beats"]
    g6 = ph.gate_g6(res)
    assert g6["n_cycles"] >= 1 and g6["passed"], g6


REAL = Path(__file__).resolve().parents[1] / "work" / "larissa_kadu"


@pytest.mark.skipif(not (REAL / "smplx" / "lead.npz").exists(), reason="needs the larissa_kadu MAMMA fit")
def test_mamma_floor_contact_scores_feet():
    """MAMMA floor_contact (T, 512) maps onto the two feet; it is high in stance and drops in the lead's kick."""
    d = ph.load_npz(REAL / "smplx" / "lead.npz")
    sc = ph.mamma_foot_contact(d, pb.load_body(d))
    if sc is None:
        pytest.skip("mamma verts_512 not available")
    assert sc.shape == (len(d["t"]), 2) and np.median(sc) > 0.5
    kick = (d["t"] > 58.7) & (d["t"] < 58.95)
    assert sc[kick, 1].min() < 0.5  # right foot in the air
