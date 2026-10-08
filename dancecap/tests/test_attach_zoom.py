"""Unit tests: per-frame focal of an attached zooming camera (cameras/attach_zoom.py), the video-zoom estimator
(cameras/video_zoom.py) and the camera-tracking gate helpers (cameras/gate_cameras.py). Synthetic geometry, no GPU.

    .venv-nopo4d\\Scripts\\python.exe -m pytest tests/test_attach_zoom.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dancecap.cameras import gate_cameras as gc  # noqa: E402
from dancecap.cameras import video_zoom as vz  # noqa: E402
from dancecap.cameras.attach_zoom import ZoomSettings, resolve_zoom  # noqa: E402
from dancecap.cameras.localize import FramePose  # noqa: E402
from dancecap.cameras.pipeline import _cached, _norm_key  # noqa: E402

W, H = 480, 640
CX, CY = W / 2 - 0.5, H / 2 - 0.5


def _look(z):
    z = z / np.linalg.norm(z)
    x = np.cross(z, [0, 1.0, 0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z])


def _project(X, R, C, f, k1=0.0):
    Xc = (X - C) @ R.T
    x, y = Xc[:, 0] / Xc[:, 2], Xc[:, 1] / Xc[:, 2]
    d = 1 + k1 * (x * x + y * y)
    return np.stack([f * x * d + CX, f * y * d + CY], 1)


def _sub_model_scene(seed=0, n=700):
    """Scene like a sub-model without parallax: points ~345 units from the camera, spread over a 40-degree cone."""
    rng = np.random.default_rng(seed)
    d = rng.normal([0, 0, 1], [0.22, 0.28, 0.0], (n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return 345.0 * d + rng.normal(0, 8.0, (n, 3))


def _fixed_focal_poses(f_true, f0, C_true, X, rng, noise=0.4):
    """What the fixed-focal sub-model localisation returns for a camera that zooms: the true correspondences, a pose
    that REPRODUCES them with the wrong (fixed) focal f0 by sliding along the optical axis."""
    poses = []
    for i, ft in enumerate(f_true):
        R = _look(np.array([0.08 * np.sin(i / 7.0), 0.05 * np.cos(i / 9.0), 1.0]))
        uv = _project(X, R, C_true, ft) + rng.normal(0, noise, (len(X), 2))
        dep = float(np.median((X - C_true) @ R[2]))
        slide = dep * (1 - f0 / ft)               # same apparent size with the smaller focal: move toward the scene
        C_i = C_true + R[2] * slide
        p = FramePose(ok=True, R=R, t=-R @ C_i, f=f0, k1=0.0, n_inliers=len(X), reproj_px=1.5, source="loc",
                      corr=SimpleNamespace(xyz=X.copy()), q_xy=uv.astype(np.float32))
        poses.append(p)
    return poses


def test_resolve_zoom_recovers_per_frame_focal():
    rng = np.random.default_rng(1)
    X = _sub_model_scene()
    n = 40
    # x1.41 zoom-in over frames 10-20 (cam05 at 9.6-10.8 s), constant before / after
    f0 = 690.0
    ratio = np.r_[np.full(10, 1 / 1.41), 1 / 1.41 * np.exp(np.linspace(0, np.log(1.41), 10)), np.ones(20)]
    f_true = f0 * ratio
    C_true = np.array([0.5, 0.4, -0.2])
    poses = _fixed_focal_poses(f_true, f0, C_true, X, rng)
    fixed_err = np.median([np.median(np.linalg.norm(_project(X, p.R, C_true, f0) - p.q_xy, axis=1)) for p in poses[:10]])
    assert fixed_err > 10.0        # fixed focal at the true centre: the zoom is NOT represented (the bug)
    out, rep = resolve_zoom(poses, C_true, W, H)
    f_new = np.array([p.f for p in out])
    assert np.max(np.abs(f_new / f_true - 1)) < 0.012, np.round(f_new / f_true, 3)
    assert rep["frames_resolved"] == n and rep["frames_dropped"] == 0
    assert rep["reproj_px_resolved_median"] < 1.0
    assert rep["f_ratio_min"] < 0.72 and rep["f_ratio_max"] > 0.99      # the zoom is in the track
    for p in out:                   # centre is the held one, translation consistent
        assert np.allclose(-p.R.T @ p.t, C_true, atol=1e-9)


def test_resolve_zoom_reference_centre_error_is_a_one_percent_focal_error():
    rng = np.random.default_rng(2)
    X = _sub_model_scene(3)
    f0 = 690.0
    poses = _fixed_focal_poses(np.full(10, f0 * 1.3), f0, np.zeros(3), X, rng)
    out, _ = resolve_zoom(poses, np.array([2.0, 1.0, -3.0]), W, H)     # C_ref off by ~3.7 units of 345 (1 %)
    err = np.array([p.f for p in out]) / (f0 * 1.3) - 1
    assert np.max(np.abs(err)) < 0.03


def test_resolve_zoom_drops_frames_that_do_not_fit_and_keeps_non_ok():
    rng = np.random.default_rng(3)
    X = _sub_model_scene(4)
    poses = _fixed_focal_poses(np.full(4, 690.0), 690.0, np.zeros(3), X, rng)
    poses[1].ok = False
    poses[2].q_xy = rng.uniform(0, 400, poses[2].q_xy.shape).astype(np.float32)      # garbage correspondences
    poses[3].corr = SimpleNamespace(xyz=X[:5].copy())
    poses[3].q_xy = poses[3].q_xy[:5]
    out, rep = resolve_zoom(poses, np.zeros(3), W, H, ZoomSettings(min_corr=30))
    assert not out[1].ok and not out[2].ok and not out[3].ok and out[0].ok
    assert rep["frames_resolved"] == 1 and rep["frames_dropped"] == 2
    assert poses[2].ok and poses[3].corr is not None      # the input list is not modified


def test_cache_key_survives_second_resolution_mtimes(tmp_path):
    """A work dir extracted from a pod tar keeps mtimes in whole seconds: the same cache must still validate."""
    key_ns = {"sub": [["images.bin", 1791417362861041100, 1962651]], "v": 3, "flag": True}
    key_s = {"sub": [["images.bin", 1791417362000000000, 1962651]], "v": 3, "flag": True}
    assert _norm_key(key_ns) == _norm_key(key_s)
    p = tmp_path / "c.pkl"
    assert _cached(p, key_ns, lambda: 41) == 41
    assert _cached(p, key_s, lambda: 42) == 41
    other = {"sub": [["images.bin", 1791417999000000000, 1962651]], "v": 3, "flag": True}
    assert _cached(p, other, lambda: 43) == 43           # a really different file still invalidates
    assert _norm_key({"n": 5, "t": True}) == {"n": 5, "t": True}


# ------------------------------------------------------------------------------------------------ video zoom

def _scene_points(rng, n=500):
    return rng.uniform([0, 0], [W, H], (n, 2))


def test_similarity_recovers_scale_and_ignores_outliers():
    rng = np.random.default_rng(0)
    a = _scene_points(rng)
    c = np.array([CX, CY])
    th = np.radians(1.5)
    Rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    b = (a - c) @ Rm.T * 1.07 + c + [4.0, -3.0] + rng.normal(0, 0.3, a.shape)
    b[:100] = rng.uniform([0, 0], [W, H], (100, 2))        # dancers / mismatches
    ls, rot, inl, n = vz.similarity(a, b)
    assert abs(np.exp(ls) - 1.07) < 0.003 and abs(rot - 1.5) < 0.2 and inl > 350
    assert np.isnan(vz.similarity(a[:5], b[:5])[0])


def test_predicted_scale_pairs_equals_focal_ratio_for_pure_zoom_and_zero_for_fixed_focal():
    from dancecap.cameras.report import project_cv  # noqa: F401
    rng = np.random.default_rng(5)
    X = _sub_model_scene(6, 400)
    C = np.zeros(3)
    n = 12
    R = np.array([_look(np.array([0.0, 0.0, 1.0]))] * n)
    tvec = np.array([-r @ C for r in R])
    zoom = np.exp(np.linspace(0, np.log(1.3), n)) * 700.0
    K = np.zeros((n, 3, 3))
    K[:, 0, 0] = K[:, 1, 1] = zoom
    K[:, 0, 2], K[:, 1, 2], K[:, 2, 2] = CX, CY, 1.0
    tr = {"K": K, "R": R, "tvec": tvec, "k1": np.zeros(n)}
    corr = {"corr_off": np.arange(n + 1) * len(X), "corr_xyz": np.tile(X, (n, 1))}
    i0, i1 = np.arange(n - 1), np.arange(1, n)
    pred = vz.predicted_scale_pairs(tr, corr, i0, i1)
    assert np.allclose(pred, np.log(zoom[1:] / zoom[:-1]), atol=2e-3)
    K2 = K.copy()
    K2[:, 0, 0] = K2[:, 1, 1] = 700.0
    pred0 = vz.predicted_scale_pairs({**tr, "K": K2}, corr, i0, i1)
    assert np.all(np.abs(pred0) < 2e-3)


def test_detect_video_events_finds_zoom_not_wander():
    step = 1 / 6
    t = np.arange(600) * step
    rng = np.random.default_rng(0)
    d = rng.normal(0, 0.002, len(t)) + 0.0004                       # wander
    d[60:66] += np.log(1.41) / 6                                    # x1.41 over 1 s
    d[300:306] += np.log(0.9) / 6                                   # x0.9 over 1 s
    ev = gc.detect_video_events(t, d, np.ones(len(t), bool), step)
    assert len(ev) == 2, ev
    assert abs(ev[0]["t0"] - 10.0) < 0.5 and 1.35 < ev[0]["factor"] < 1.5
    assert ev[1]["factor"] < 0.95
    quiet = rng.normal(0, 0.002, len(t))
    assert gc.detect_video_events(t, quiet, np.ones(len(t), bool), step) == []


def test_window_sums_and_zoom_metrics_flag_untracked_zoom():
    t = np.arange(300) / 6.0
    meas = np.zeros(300)
    meas[60:66] = np.log(1.4) / 6
    ok = np.ones(300, bool)
    tracked = {"t": t, "measured": meas, "ok": ok, "predicted": meas.copy(), "dlogf": meas.copy()}
    m = gc.zoom_metrics(tracked)
    assert m["unexplained_max_2s_pct"] < 0.5
    untracked = {**tracked, "predicted": np.zeros(300), "dlogf": np.zeros(300)}
    m2 = gc.zoom_metrics(untracked)
    assert 38 < m2["unexplained_max_2s_pct"] < 42
    b = [x for x in m2["blocks"] if x["unexplained_max_2s_pct"] > 5]
    # a window that STARTS in the previous block reaches into the event: both blocks are flagged (conservative)
    assert [x["t0"] for x in b] == [0.0, 10.0] and all(x["focal_zoom_max_2s_pct"] < 1 for x in b)
    ws = gc.window_sums(np.arange(10.0), np.ones(10), 2.0)
    assert ws[0] == 3 and ws[-1] == 1


def test_jitter_and_block_rows_flag_a_bad_block():
    n = 600
    t = np.arange(n) / 30.0
    R = np.repeat(np.eye(3)[None], n, 0)
    C = np.zeros((n, 3))
    C[400:, 0] += 0.0                                               # still camera
    rng = np.random.default_rng(1)
    C[300:420] += rng.normal(0, 0.05, (120, 3))                     # a shaky stretch: 5 cm of jitter
    d = {"t": t, "R": R, "C": C, "ok": np.ones(n, bool), "ok_raw": np.ones(n, bool), "f": np.full(n, 600.0),
         "maxdim": 640, "reproj_px": np.full(n, 1.0), "n_inliers_raw": np.full(n, 500),
         "recentred": np.zeros(n, bool), "attached": False}
    jc, jr = gc.jitter(d, 1.0)
    assert np.median(jc[:200]) < 1e-6 and np.median(jc[300:420]) > 20.0
    rows = gc.rows_for("x", "01", d, np.full(n, 10.0), 1.0, {"dancers_px_med": 12.0, "centre_jitter_p95": 4.0})
    assert [r["t0"] for r in rows] == [0.0, 10.0]
    for r in rows:
        r["violations"] = gc.judge(r, "01", False, {"dancers_px_med": 12.0, "centre_jitter_p95": 4.0})
    assert rows[1]["violations"] == ["centre_jitter"] and rows[0]["violations"] == []
    bad = dict(rows[0], dancers_px_med=80.0, ok_exported=0.5, static_px_med=6.0, inliers_med=20.0)
    assert set(gc.judge(bad, "01", False, {"dancers_px_med": 12.0})) >= {"ok_exported", "static_px", "inliers", "dancers_px"}


def test_weak_frames_get_the_interpolated_focal_not_a_garbage_one():
    """cam05 130.9-131.5 s: a dancer fills the view, 17-68 clustered inliers; the free focal went to x0.81 for 0.6 s (the
    operator moved while a patch of points could not tell a step toward the scene from a zoom)."""
    rng = np.random.default_rng(11)
    X = _sub_model_scene(8, 900)
    f0 = 690.0
    n = 60
    ratio = np.r_[np.ones(30), np.exp(np.linspace(0, np.log(1.25), 15)), np.full(15, 1.25)]       # a real zoom at 30-45
    C = np.array([0.2, 0.1, 0.0])
    poses = _fixed_focal_poses(f0 * ratio, f0, C, X, rng, noise=1.0)
    weak = list(range(10, 16)) + list(range(36, 39))                       # 6 frames before the zoom, 3 inside it
    for i in weak:
        R = poses[i].R
        step = C + R[2] * 40.0                                              # the operator stepped 40 units (12 % of the depth)
        uv_full = _project(X, R, step, f0 * ratio[i]) + rng.normal(0, 1.0, (len(X), 2))
        near = np.argsort(np.linalg.norm(uv_full - [CX + 80.0, CY + 120.0], axis=1))[:40]    # a small patch of the picture
        poses[i].corr = SimpleNamespace(xyz=X[near].copy())
        poses[i].q_xy = uv_full[near].astype(np.float32)
    spike = 50
    poses[spike].q_xy = _project(X, poses[spike].R, C, f0 * 1.25 * 0.88).astype(np.float32)    # a 12 % focal spike
    unsafe, _ = resolve_zoom(poses, C, W, H, ZoomSettings(min_free=0, min_spread=0, hampel_win=10 ** 6), t=np.arange(n) / 30.0)
    fu = np.array([p.f if p.ok else np.nan for p in unsafe])
    f_true = f0 * ratio
    assert np.nanmax(np.abs(fu[weak] / f_true[weak] - 1)) > 0.05, "the scenario must break the unguarded solve"
    out, rep = resolve_zoom(poses, C, W, H, ZoomSettings(min_free=120), t=np.arange(n) / 30.0)
    f = np.array([p.f if p.ok else np.nan for p in out])
    err = np.abs(f / f_true - 1)
    assert np.nanmax(err[weak]) < 0.02, np.round(err[weak], 3)               # interpolated from the neighbours
    assert err[spike] < 0.02 or np.isnan(f[spike])                           # the spike is not exported
    good = [i for i in range(n) if i not in weak and i != spike]
    assert np.nanmax(err[good]) < 0.015
    assert rep["frames_focal_interpolated"] >= len(weak)
    assert f[44] / f[10] > 1.2                                                # the zoom itself is kept
    # a plain monotone ramp is not flagged as spikes
    from dancecap.cameras.attach_zoom import hampel_spikes
    lf = np.log(np.exp(np.linspace(0, np.log(1.4), 40)))
    assert not hampel_spikes(np.arange(40) / 30.0, lf, 10, 3.0, 0.02).any()


# ------------------------------------------------------------------------------------------------ video motion / repair

def _cams_pair(f0=700.0, f1=700.0, rot_deg=1.2, k1=0.0, depth=8.0, shift=(0.0, 0.0, 0.0)):
    from scipy.spatial.transform import Rotation
    R0 = np.eye(3)
    R1 = Rotation.from_rotvec(np.radians(rot_deg) * np.array([0.3, 1.0, 0.1]) / np.linalg.norm([0.3, 1.0, 0.1])).as_matrix()
    K0 = np.array([[f0, 0, CX], [0, f0, CY], [0, 0, 1.0]])
    K1 = np.array([[f1, 0, CX], [0, f1, CY], [0, 0, 1.0]])
    return K0, R0, K1, R1


def test_predict_matches_forward_projection_and_handles_distortion():
    from dancecap.cameras import video_motion as vm
    rng = np.random.default_rng(0)
    K0, R0, K1, R1 = _cams_pair(700, 910, 3.0)
    k1 = -0.08
    X = rng.normal([0, 0, 60], [15, 20, 5], (200, 3))
    X = X[X[:, 2] > 20]

    def proj(R, K, X, C=np.zeros(3)):
        Xc = (X - C) @ R.T
        x = Xc[:, :2] / Xc[:, 2:]
        return (x * (1 + k1 * (x ** 2).sum(1, keepdims=True))) * K[0, 0] + K[:2, 2]
    a, b = proj(R0, K0, X), proj(R1, K1, X)
    pred = vm.predict(a, K0, R0, k1, K1, R1, k1)
    assert np.abs(pred - b).max() < 0.05
    # with a translation and the true depth, predict3d reproduces it; the pure-rotation prediction does not
    C1 = np.array([0.0, 0.0, 1.5])
    b2 = proj(R1, K1, X, C1)
    z0 = (X @ R0.T)[:, 2]
    p3 = vm.predict3d(a, z0, K0, R0, -R0 @ np.zeros(3), k1, K1, R1, -R1 @ C1, k1)
    assert np.abs(p3 - b2).max() < 0.05
    assert np.median(np.linalg.norm(vm.predict(a, K0, R0, k1, K1, R1, k1) - b2, axis=1)) > 2.0


def test_to_frame_orientation_clockwise_for_minus_90():
    from dancecap.cameras import video_motion as vm
    # landscape 640 x 480 (x right, y down) rotated a quarter turn clockwise -> portrait 480 x 640
    P = np.array([[0.0, 0.0], [639.0, 0.0], [0.0, 479.0], [320.0, 240.0]])
    Q = vm.to_frame_orientation(P, -90, 480, 640)
    assert np.allclose(Q, [[479, 0], [479, 639], [0, 0], [239, 320]])       # top-left -> top-right, ...
    assert np.allclose(vm.to_frame_orientation(P, 0, 480, 640), P)


def test_motion_pairs_flags_an_unmodelled_zoom_and_passes_a_correct_track():
    from dancecap.cameras import video_motion as vm
    rng = np.random.default_rng(1)
    n = 12
    f_true = np.r_[np.full(6, 700.0), 700.0 * np.exp(np.linspace(0, np.log(1.4), 6))]
    X = rng.normal([0, 0, 50], [12, 16, 4], (300, 3))
    from scipy.spatial.transform import Rotation
    Rs = np.array([Rotation.from_rotvec([0, np.radians(0.4 * i), 0]).as_matrix() for i in range(n)])

    def proj(i, f):
        Xc = X @ Rs[i].T
        return Xc[:, :2] / Xc[:, 2:] * f + [CX, CY]
    t = np.arange(n) / 6.0
    ma = np.full((n - 1, 150, 2), np.nan, np.float32)
    mb = ma.copy()
    for i in range(n - 1):
        ma[i], mb[i] = proj(i, f_true[i])[:150], proj(i + 1, f_true[i + 1])[:150]
    vid = {"offset": 0.0, "slope": 1.0, "pts0": t[:-1], "pts1": t[1:], "ma": ma, "mb": mb, "dlog": np.diff(np.log(f_true)),
           "ok": np.ones(n - 1, bool), "inl": np.full(n - 1, 150)}

    def track(f):
        K = np.zeros((n, 3, 3))
        K[:, 0, 0] = K[:, 1, 1] = f
        K[:, 0, 2], K[:, 1, 2], K[:, 2, 2] = CX, CY, 1.0
        return {"t": t, "K": K, "R": Rs, "tvec": np.zeros((n, 3)), "k1": np.zeros(n), "ok": np.ones(n, bool), "w": W, "h": H,
                "attached": True, "rotation": 0}
    good = vm.motion_pairs(track(f_true), vid, max_dt=0.01)
    assert np.nanmax(good["motion_deg"]) < 0.01 and np.nanmax(np.abs(good["dlog_res"])) < 0.002
    bad = vm.motion_pairs(track(np.full(n, 700.0)), vid, max_dt=0.01)           # constant focal: the bug
    assert np.nanmax(bad["motion_deg"]) > 0.3
    assert 0.25 < np.exp(np.abs(np.nansum(bad["dlog_res"]))) - 1 < 0.45


def test_video_repair_chain_recovers_rotation_offset_and_finds_the_jump():
    from scipy.spatial.transform import Rotation

    from dancecap.cameras import video_repair as vr
    rng = np.random.default_rng(2)
    n = 60
    X = rng.normal([0, 0, 40], [12, 16, 4], (300, 3))
    f = 700.0
    ang = np.radians(np.linspace(0, 10, n))                       # a steady pan of 10 deg
    R_true = np.array([Rotation.from_rotvec([0, a, 0]).as_matrix() for a in ang])
    t = np.arange(n) / 30.0
    # exported track: the first 30 frames are a wrong mode (rotated by 21 deg, focal 14 % too large)
    R_exp = R_true.copy()
    R_exp[:30] = np.array([Rotation.from_rotvec([0.3, 0.0, 0.0]).as_matrix() @ r for r in R_true[:30]])
    f_exp = np.r_[np.full(30, f * 1.14), np.full(n - 30, f)]
    K = np.zeros((n, 3, 3))
    K[:, 0, 0] = K[:, 1, 1] = f_exp
    K[:, 0, 2], K[:, 1, 2], K[:, 2, 2] = CX, CY, 1.0
    d = {"t": t, "R": R_exp, "tvec": np.zeros((n, 3)), "C": np.zeros((n, 3)), "f": f_exp, "K": K, "w": W, "h": H, "rotation": 0,
         "maxdim": H, "n_inliers_raw": np.full(n, 800), "ok": np.ones(n, bool)}
    st = vr.rotation_steps(d["R"])
    assert (st > 3).sum() == 1 and np.argmax(st) == 29
    # video pairs every 5 frames
    ks = np.arange(0, n - 5, 5)
    ma = np.full((len(ks), 150, 2), np.nan, np.float32)
    mb = ma.copy()
    for q, i in enumerate(ks):
        for arr, j in ((ma, i), (mb, i + 5)):
            Xc = X @ R_true[j].T
            arr[q] = (Xc[:, :2] / Xc[:, 2:] * f + [CX, CY])[:150]
    vid = {"offset": 0.0, "slope": 1.0, "pts0": t[ks], "pts1": t[ks + 5], "ma": ma, "mb": mb,
           "dlog": np.zeros(len(ks)), "ok": np.ones(len(ks), bool), "inl": np.full(len(ks), 150)}
    jumps = vr.find_jumps(d, vid)
    assert len(jumps) == 1 and abs(jumps[0]["video_zoom_pct"]) < 1.0 and jumps[0]["frame"] == 30
    rep = vr.repair_stretch(d, vid, 30, 35)                      # anchor on a video sample (frame 35)
    err = [np.degrees(np.linalg.norm(Rotation.from_matrix(rep["R"][q] @ R_true[i].T).as_rotvec())) for q, i in enumerate(rep["frames"])]
    assert max(err) < 0.2, max(err)
    assert np.allclose(rep["f"], f, rtol=1e-3)


def test_person_box_masking_helpers(tmp_path):
    import json

    from dancecap.cameras import video_zoom as vz
    boxes = np.array([[10, 10, 50, 100], [200, 300, 260, 420]], float)
    xy = np.array([[20, 20], [60, 60], [230, 350], [400, 400], [49, 99]], float)
    assert vz.outside_boxes(xy, boxes).tolist() == [False, True, False, True, False]
    assert vz.outside_boxes(xy, np.zeros((0, 4))).all()
    # PersonBoxes: one take, 3 frames at 30 Hz, box dilated by the margin around its centre
    w = tmp_path / "take"
    (w / "people").mkdir(parents=True)
    (w / "frames" / "01").mkdir(parents=True)
    np.savez(w / "people" / "01.npz", boxes=np.tile(np.array([[[100, 100, 200, 300]]], float), (3, 1, 1)),
             valid=np.ones((3, 1), bool))
    (w / "frames" / "01" / "frames.json").write_text(json.dumps(
        {"width": 480, "height": 848, "frames": [{"t": 10.0 + i / 30} for i in range(3)]}))
    pb = vz.PersonBoxes("01", ("take",), tmp_path, margin=0.10)
    b = pb.at(10.03)
    assert b.shape == (1, 4) and np.allclose(b[0], [95, 90, 205, 310])
    assert pb.at(50.0).shape == (0, 4)                      # outside every take's frames
