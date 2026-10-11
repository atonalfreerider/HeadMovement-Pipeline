r"""Unit tests for dancecap.jl_mono, farm/jl_profile (pure parts) and the heading Viterbi of mamma-turing's utils/mono.py (synthetic numbers; no footage, no pod).

    .venv-nopo4d\Scripts\python.exe -m pytest dancecap/tests/test_jl_mono.py -q      (from the workspace root)
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dancecap import jl_mono as jm  # noqa: E402
from dancecap.farm import jl_profile as jlp  # noqa: E402


def camera(f=808.0, height=1.43, pitch_deg=6.0, dist=4.2):
    """A camera at (0, height, dist) looking along -Z, pitched down: world -> camera (OpenCV, +Z forward, +Y down)."""
    p = np.deg2rad(pitch_deg)
    # camera axes in the world: x right (+X), z forward (-Z pitched down), y down
    z = np.array([0.0, -np.sin(p), -np.cos(p)])
    x = np.array([1.0, 0.0, 0.0])
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    C = np.array([0.0, height, dist])
    K = np.array([[f, 0, 269.0], [0, f, 480.0], [0, 0, 1.0]])
    return K, R, -R @ C


def project(X, K, R, t):
    Xc = R @ X + t
    uv = K @ Xc
    return uv[:2] / uv[2]


def make_kp(x, z, K, R, t, stature=1.8, ankles=True):
    kp = np.zeros((133, 3))
    half = 0.1
    pts = {jm.L_ANK: (x - 0.1, jm.ANKLE_ABOVE_FLOOR_M, z), jm.R_ANK: (x + 0.1, jm.ANKLE_ABOVE_FLOOR_M, z),
           jm.L_HIP: (x - half, 0.53 * stature, z), jm.R_HIP: (x + half, 0.53 * stature, z),
           jm.L_SHO: (x - 0.2, (0.53 + jm.TORSO_OVER_STATURE) * stature, z), jm.R_SHO: (x + 0.2, (0.53 + jm.TORSO_OVER_STATURE) * stature, z)}
    for j, X in pts.items():
        if not ankles and j in (jm.L_ANK, jm.R_ANK):
            continue
        kp[j, :2] = project(np.array(X), K, R, t)
        kp[j, 2] = 0.9
    return kp


def test_floor_point_inverts_the_projection():
    K, R, t = camera()
    X = np.array([0.7, 0.09, -0.3])
    uv = project(X, K, R, t)
    got, z = jm.floor_point(uv, K, R, t, 0.0, 0.09)
    assert np.allclose(got, X, atol=1e-6) and z > 0


def test_floor_point_above_the_horizon_is_none():
    K, R, t = camera(pitch_deg=0.0)
    assert jm.floor_point(np.array([269.0, 100.0]), K, R, t, 0.0, 0.0) is None


def test_point_at_depth_roundtrip():
    K, R, t = camera()
    X = np.array([0.4, 1.0, -1.0])
    uv = project(X, K, R, t)
    depth = (R @ X + t)[2]
    assert np.allclose(jm.point_at_depth(uv, K, R, t, depth), X, atol=1e-6)


def test_undistort_matches_the_forward_model():
    from dancecap import mamma_dyn as md
    K, _, _ = camera()
    px = np.array([[100.0, 200.0], [500.0, 800.0]])
    d = md.distort_simple_radial(px, K, -0.012)
    assert np.allclose(jm.undistort_px(d[0], K, -0.012), px[0], atol=1e-3)


def test_pelvis_from_feet_recovers_the_position():
    K, R, t = camera()
    T = 12
    pos_true = np.array([[0.1 * i, 0.0, -0.5 + 0.05 * i] for i in range(T)])
    kp = np.stack([make_kp(p[0], p[2], K, R, t) for p in pos_true])
    out = jm.pelvis_positions(kp, np.repeat(K[None], T, 0), np.repeat(R[None], T, 0), np.repeat(t[None], T, 0), np.zeros(T), 1.8)
    assert (out["source"] == 1).all() and out["conf"].all()
    # the pelvis is the hip-mid at 0.53 stature above the floor, at the ankles' depth
    i = slice(2, -2)                                         # the median filter over 5 frames is biased on a ramp's two end frames
    assert np.allclose(out["pos"][i, 0], pos_true[i, 0], atol=0.03)
    assert np.allclose(out["pos"][:, 1], 0.53 * 1.8, atol=0.05)
    assert np.allclose(out["pos"][i, 2], pos_true[i, 2], atol=0.05)


def test_pelvis_without_ankles_uses_the_torso_length_and_fills_gaps():
    K, R, t = camera()
    T = 9
    kp = np.stack([make_kp(0.2, -0.4, K, R, t, ankles=False) for _ in range(T)])
    kp[4] = 0.0                                              # a frame with nothing: filled from the neighbours
    out = jm.pelvis_positions(kp, np.repeat(K[None], T, 0), np.repeat(R[None], T, 0), np.repeat(t[None], T, 0), np.zeros(T), 1.8)
    assert (out["source"][[0, 1, 2, 3, 5, 6, 7, 8]] == 2).all() and out["source"][4] == 0 and out["conf"][4] == 0
    assert np.allclose(out["pos"][:, 2], -0.4, atol=0.45)    # torso length: depth within ~10 % of 4.6 m
    assert np.isfinite(out["pos"]).all()


def test_partner_skeleton_and_bad_camera_frames_carry_no_measurement():
    K, R, t = camera()
    T = 6
    kp = np.stack([make_kp(0.0, 0.0, K, R, t) for _ in range(T)])
    sp = np.zeros(T)
    sp[2] = 0.9
    ok = np.ones(T, bool)
    ok[3] = False
    out = jm.pelvis_positions(kp, np.repeat(K[None], T, 0), np.repeat(R[None], T, 0), np.repeat(t[None], T, 0), np.zeros(T), 1.8, ok, sp)
    assert out["conf"].tolist() == [1, 1, 0, 0, 1, 1]
    assert np.isfinite(out["pos"]).all()


def test_stature_targets():
    cfg = {"dancers": {"lead": {"stature_m": 1.803}, "follow": {"stature_m": 1.702}}}
    assert jm.stature_targets(cfg, {"lead": 0.025, "follow": 0.02}) == {"0": pytest.approx(1.828), "1": pytest.approx(1.722)}


def _mono_module():
    from dancecap import env
    p = Path(env.mamma_root()) / "optimization" / "utils" / "mono.py"
    if not p.exists():
        pytest.skip("mamma-turing checkout not present")
    pytest.importorskip("torch")
    spec = importlib.util.spec_from_file_location("jl_mamma_mono", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_viterbi_heading_smooths_a_flipped_frame_and_follows_a_spin():
    m = _mono_module()
    J, T = 24, 60
    truth = (np.arange(T) * 0.5).astype(int) % J               # a spin of 0.5 steps / frame
    cost = np.ones((T, J))
    cost[np.arange(T), truth] = 0.0
    wrong = (truth[30] + 12) % J                               # one frame prefers the opposite heading
    cost[30, :] = 1.0
    cost[30, wrong] = 0.0
    cost[30, truth[30]] = 0.2
    valid = np.ones(T, bool)
    valid[10:13] = False                                       # no evidence: interpolated
    path = m.viterbi_yaw(cost, valid)
    assert path[30] == truth[30]
    d = np.abs(path - truth)
    d = np.minimum(d, J - d)
    assert (d <= 1).all()


def test_shape_window_prefers_the_clean_stretch():
    T = 3000
    t = np.arange(T) / 30.0
    tt = {"t": t, "conf": np.full((T, 2), 0.9), "merged": np.zeros((T, 2), bool), "edge": np.zeros((T, 2), bool), "skel_in_partner": np.zeros((T, 2)),
          "role_conf": np.full((T, 2), 0.95), "visible": np.ones((T, 2))}
    tt["merged"][:1200] = True                                 # the first 40 s are clinches
    w = jlp.shape_window(tt)
    assert w["t0"] >= 40.0 - 1e-6 and w["k1"] - w["k0"] == 899 and w["score"] == 1.0


def test_estimate_chunk_adds_up():
    from dancecap.farm import chunks as C
    ch = C.Chunk(id="jlx", k0=0, k1=1381, core_k0=0, core_k1=1321, overlap_k=120, kind="full", cams=["01"], world_id=None, origin="planned", take="john_lula_jlx")
    e = jlp.estimate_chunk(ch, cam_frames=ch.T)
    assert e["fit_min"] > 10 and e["total_min_4090"] > e["fit_min"] and e["usd_4090"] > 0


def test_depth_compare_scales_the_range_and_keeps_the_pose():
    from dancecap import jl_depth as jd
    rng = np.random.default_rng(0)
    N = 40
    pose = rng.normal(0, 0.3, (22, 3))
    centre = np.array([1.7, 1.43, 3.85])
    base = np.zeros((N, 55, 3))
    base[:, :22] = pose[None]
    base[:, :, 1] += 1.0
    base[:, :, 0] += np.linspace(0, 1, N)[:, None]
    base[:, :, 2] -= 0.4
    # the variant: the same dancer 10 % further from the camera on the floor, pose and heights unchanged
    var = base.copy()
    d = base[:, 0, [0, 2]] - centre[[0, 2]]
    shift = 0.1 * d
    var[:, :, 0] += shift[:, 0:1]
    var[:, :, 2] += shift[:, 1:2]
    r = jd.compare_arrays(base, var, centre, 1 / 30.0)
    assert abs(r["range_ratio_median"] - 1.1) < 1e-3
    assert r["pose_relative_to_pelvis_mpjpe_mm"]["median"] < 1e-3
    assert abs(r["pelvis_height_m"]["base"] - r["pelvis_height_m"]["variant"]) < 1e-9


def test_depth_spike_is_found_and_interpolated_and_nothing_else_moves():
    from dancecap import jl_repair as jr
    T = 120
    t = np.arange(T) / 30.0
    P = np.stack([0.5 * np.sin(t), 0.95 + 0 * t, -1.0 + 0.3 * t], 1)
    Q = P.copy()
    Q[60:63, 2] -= 1.5                                       # three frames 1.5 m too deep
    bad = jr.find_spikes(Q)
    assert bad[60:63].all() and bad.sum() <= 5 and not bad[:57].any() and not bad[66:].any()
    off = jr.repair_pelvis(Q, bad)
    fixed = Q + off
    assert np.abs(fixed - P)[bad].max() < 0.05               # a straight line over a smooth path
    assert np.abs(off[~bad]).max() == 0.0


def test_repair_folder_roundtrip(tmp_path):
    from dancecap import jl_repair as jr
    T = 90
    for role in ("lead", "follow"):
        P = np.stack([0.01 * np.arange(T), 0.95 + np.zeros(T), np.zeros(T)], 1)
        if role == "lead":
            P[40:42, 2] += 1.2
        J = np.repeat(P[:, None, :], 55, 1)
        np.savez(tmp_path / f"{role}.npz", t=np.arange(T) / 30.0, joints=J, transl=P.copy(), repaired=np.zeros(T, bool))
    rep = jr.repair_folder(tmp_path)
    assert rep["lead"]["frames"] >= 2 and rep["follow"]["frames"] == 0
    z = np.load(tmp_path / "lead.npz")
    assert abs(z["joints"][40, 0, 2]) < 0.05 and abs(z["transl"][41, 2]) < 0.05 and z["depth_spike_repaired"][40]


def test_worker_claims_only_specs_of_its_own_owner(tmp_path, monkeypatch):
    from dancecap.farm import worker as W
    (tmp_path / "queue").mkdir()
    spec = {"id": "x1", "kind": "analyse", "needs_gpu": False, "steps": []}
    mine = dict(spec, id="x2", owner="J3b")
    monkeypatch.delenv("HM_OWNER", raising=False)
    w = W.Worker(tmp_path, pod_id="p", gpu_required=False)
    assert w.claimable(spec) and not w.claimable(mine)          # an ownerless worker never takes an owned job
    monkeypatch.setenv("HM_OWNER", "J3b")
    assert w.claimable(mine) and not w.claimable(spec)          # and an owned worker takes only its own
    monkeypatch.setenv("HM_OWNER", "other")
    assert not w.claimable(mine)
