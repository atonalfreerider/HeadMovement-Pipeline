"""Unit tests for dancecap.cameras (synthetic geometry, no GPU) + contract checks on real outputs when present.

    .venv-nopo4d\\Scripts\\python.exe -m pytest tests/test_cameras.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dancecap.cameras.features import ImageRec, keep_outside, prefill  # noqa: E402
from dancecap.cameras.localize import reproj, solve_pose, LocSettings  # noqa: E402
from dancecap.cameras.mapping import MapSettings, map_pairs, select_map_images  # noqa: E402
from dancecap.cameras.smooth import robust_smooth, smooth_track  # noqa: E402
from dancecap.cameras.world import (EYE_MINUS_ANKLE, apply, bootstrap, fit_plane, triangulate_robust,  # noqa: E402
                                    undistort_pts)
from dancecap.take import PACKAGE_ROOT  # noqa: E402

WORK = PACKAGE_ROOT / "work" / "larissa_kadu"
CAMS = ["01", "02", "03", "04", "05", "06", "07", "09"]


def _look_at(C, target, up=np.array([0, 1.0, 0])):
    """OpenCV world->cam rotation for a camera at C looking at target (world Y up -> image y down)."""
    z = target - C
    z /= np.linalg.norm(z)
    x = np.cross(z, up)  # right
    x /= np.linalg.norm(x)
    y = np.cross(z, x)  # down
    R = np.stack([x, y, z])
    return R, -R @ C


def test_prefill_smooth_and_untouched():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (200, 120, 3), dtype=np.uint8)
    mask = np.zeros((200, 120), np.uint8)
    mask[60:140, 30:90] = 255
    out = prefill(img, mask)
    assert out.shape == img.shape
    assert np.array_equal(out[:40], img[:40])  # far from the mask: untouched
    inner = out[80:120, 45:75].astype(float)
    assert np.abs(np.diff(inner, axis=1)).mean() < 0.2 * np.abs(np.diff(img[80:120, 45:75].astype(float), axis=1)).mean()
    assert np.array_equal(prefill(img, np.zeros_like(mask)), img)


def test_keep_outside():
    mask = np.zeros((50, 50), np.uint8)
    mask[20:30, 20:30] = 255
    k = np.array([[25, 25], [31, 25], [40, 40], [5, 5]], float)
    assert keep_outside(k, mask, 0).tolist() == [False, True, True, True]
    assert keep_outside(k, mask, 3).tolist() == [False, False, True, True]
    assert keep_outside(np.zeros((0, 2)), mask, 3).shape == (0,)


def test_map_selection_and_pairs():
    recs = {}
    for cam in ("01", "02"):
        fps = 60 if cam == "02" else 30
        win = [ImageRec(cam, "win", i, 56 + i / fps, f"frames/{cam}/{i:06d}.jpg", "") for i in range(6 * fps)]
        kf = [ImageRec(cam, "kf", j, 30 + 4 * j, f"keyframes/{cam}/{j}.jpg", "") for j in range(10)]
        recs[cam] = win + kf
    s = MapSettings()
    imgs, dense = select_map_images(recs, s)
    for cam in ("01", "02"):
        w = [r for r in imgs if r.cam == cam and r.kind == "win"]
        assert 8 <= len(w) <= 10  # ~1.5 Hz regardless of fps
        assert len([r for r in imgs if r.cam == cam and r.kind == "kf"]) == 10
    pairs = map_pairs(imgs, dense, s)
    assert all(i < j for i, j in pairs) and len(set(pairs)) == len(pairs)
    nd = len(dense)
    assert len(pairs) >= nd * (nd - 1) // 2


def test_robust_smooth_line_outliers_gap():
    t = np.arange(100) / 30.0
    y = 2.0 * t + 1.0 + np.random.default_rng(1).normal(0, 0.01, 100)
    y[[10, 50, 51]] += 5.0  # outliers
    w = np.ones(100)
    w[70:80] = 0  # missing
    ys, rw = robust_smooth(t, y, w, sigma=0.05)
    truth = 2.0 * t + 1.0
    assert np.abs(ys - truth).max() < 0.05
    assert rw[10] < 0.05 and rw[50] < 0.05
    # a long gap (no kernel support) is interpolated
    w2 = np.ones(100)
    w2[20:60] = 0
    ys2, _ = robust_smooth(t, truth, w2, sigma=0.03)
    assert np.abs(ys2 - truth).max() < 1e-6


def test_smooth_track_preserves_motion():
    t = np.arange(90) / 30.0
    from scipy.spatial.transform import Rotation
    R = Rotation.from_euler("y", 0.3 * t[:, None]).as_matrix()
    C = np.stack([np.sin(t), 0.1 * t, np.cos(t)], 1)
    tv = -np.einsum("fij,fj->fi", R, C)
    f = 1000 * np.exp(0.2 * t)
    ok = np.ones(90, bool)
    ok[40] = False
    R_bad = R.copy()
    R_bad[40] = np.eye(3)
    sm = smooth_track(t, R_bad, tv, f, ok)
    assert np.abs(sm["C"] - C).max() < 2e-3
    assert np.abs(np.log(sm["f"] / f)).max() < 2e-3
    ang = np.degrees(np.linalg.norm(Rotation.from_matrix(np.einsum("fij,fkj->fik", sm["R"], R)).as_rotvec(), axis=1))
    assert ang.max() < 0.1


def test_solve_pose_recovers_focal():
    rng = np.random.default_rng(3)
    w, h, f = 480, 864, 700.0
    X = rng.uniform([-4, 0, 3], [4, 3, 9], (400, 3))
    R, t = _look_at(np.array([0.3, 1.5, -2.0]), np.array([0, 1.2, 6.0]))
    xy = np.zeros((400, 2))
    Xc = X @ R.T + t
    xy[:, 0] = f * Xc[:, 0] / Xc[:, 2] + w / 2
    xy[:, 1] = f * Xc[:, 1] / Xc[:, 2] + h / 2
    inside = (xy[:, 0] > 0) & (xy[:, 0] < w) & (xy[:, 1] > 0) & (xy[:, 1] < h)
    xy, X = xy[inside], X[inside]
    xy_noisy = xy + rng.normal(0, 0.5, xy.shape)
    xy_noisy[:20] += rng.uniform(-80, 80, (20, 2))  # outliers
    sol = solve_pose(xy_noisy, X, w, h, f0=820.0, k1=0.0, s=LocSettings())
    assert sol is not None
    assert abs(sol["f"] / f - 1) < 0.02
    assert np.abs(sol["R"] - R).max() < 0.01
    e = reproj(sol["R"], sol["t"], sol["f"], 0.0, w, h, X[20:], xy[20:])
    assert np.median(e) < 0.6


def test_triangulate_and_undistort():
    K = np.array([[800, 0, 240], [0, 800, 432], [0, 0, 1.0]])
    X = np.array([0.2, 0.1, 5.0])
    Rs, ts, xys = [], [], []
    for C in ([-2, 0.5, 0], [2, 0.3, 0.5], [0, 1.5, -1]):
        R, t = _look_at(np.array(C, float), np.array([0, 1, 5.0]))
        Xc = R @ X + t
        k1 = -0.05
        x, y = Xc[:2] / Xc[2]
        d = 1 + k1 * (x * x + y * y)
        xys.append([800 * x * d + 240, 800 * y * d + 432])
        Rs.append(R), ts.append(t)
    n = undistort_pts(np.array(xys[0]), K, -0.05)
    Xc0 = Rs[0] @ X + ts[0]
    assert np.allclose(n, Xc0[:2] / Xc0[2], atol=1e-6)
    xy = np.array(xys)
    xy_bad = xy.copy()
    xy_bad[2] += 60  # one wrong view
    Xh, err, nv = triangulate_robust(xy_bad, np.stack([K] * 3), np.full(3, -0.05), np.array(Rs), np.array(ts))
    assert nv == 2 and np.linalg.norm(Xh - X) < 1e-3


def test_fit_plane():
    rng = np.random.default_rng(0)
    P = np.c_[rng.uniform(-3, 3, 500), rng.normal(0, 0.002, 500), rng.uniform(-3, 3, 500)]
    P = np.r_[P, rng.uniform(-3, 3, (200, 3))]
    n, d, inl = fit_plane(P, 0.01, prior_n=np.array([0, 1.0, 0]))
    assert abs(abs(n[1]) - 1) < 1e-3 and abs(d) < 2e-3 and inl[:500].mean() > 0.97


def test_bootstrap_recovers_similarity():
    """World scene -> random SfM frame; the bootstrap must recover gravity, floor and scale."""
    from scipy.spatial.transform import Rotation
    rng = np.random.default_rng(5)
    stat = {"lead": 1.75, "follow": 1.65}
    # world: dancers' ankles near y = ankle height, eyes at 0.936 H (upright 20 % of the time)
    T = 200
    dancers_w = {}
    for role, H in stat.items():
        base = np.c_[rng.normal(0, 0.5, T), np.zeros(T), rng.normal(0, 0.5, T)]
        ank_h = 0.039 * np.mean(list(stat.values())) + 0.025
        ankL = base + [0, ank_h, 0] + np.c_[np.zeros(T), np.where(rng.random(T) < 0.3, rng.uniform(0, 0.3, T), 0),
                                            np.zeros(T)]
        ankR = base + [0.2, ank_h, 0]
        eye_h = np.where(rng.random(T) < 0.25, 0.936 * H + 0.025, rng.uniform(0.8, 0.9, T) * H)
        eye = base + np.c_[np.zeros(T), eye_h, np.zeros(T)]
        dancers_w[role] = {1: eye, 2: eye, 15: ankL, 16: ankR}
    floor_pts = np.c_[rng.uniform(-5, 5, 3000), np.zeros(3000), rng.uniform(-5, 5, 3000)]
    wall_pts = np.c_[rng.uniform(-5, 5, 2000), rng.uniform(0, 4, 2000), np.full(2000, 6.0)]
    refl = wall_pts * [1, -1, 1]
    pts_w = np.r_[floor_pts, wall_pts, refl]
    # random similarity world -> sfm
    s_true = 0.37
    Rq = Rotation.from_rotvec(rng.normal(0, 1, 3)).as_matrix()
    tq = rng.normal(0, 2, 3)
    to_sfm = lambda X: (s_true * (X @ Rq.T) + tq)  # noqa: E731
    tracks = {}
    for k, cam in enumerate(["06", "01", "02"]):
        a = 2 * np.pi * k / 3
        C = np.array([6 * np.cos(a), 1.4, 6 * np.sin(a)])
        Rw, tw = _look_at(C, np.array([0, 1.0, 0]))
        # world->cam R_w; sfm->cam: x = R_w (X_w) + t; X_w = Rq^T (X_s - tq)/s
        R_s = Rw @ Rq.T
        t_s = s_true * tw - R_s @ tq
        tracks[cam] = {"R": np.repeat(R_s[None], 10, 0), "tvec": np.repeat(t_s[None], 10, 0), "ok": np.ones(10, bool)}
    dancers_s = {r: {k: to_sfm(v) for k, v in d.items()} for r, d in dancers_w.items()}
    sim3 = bootstrap(tracks, dancers_s, to_sfm(pts_w), stat, "test")
    s_est = sim3["s"]
    assert abs(s_est * s_true - 1) < 0.05, s_est * s_true  # sfm->world scale = 1/s_true
    fl = apply(sim3, to_sfm(floor_pts))
    assert np.abs(fl[:, 1]).max() < 0.02, np.abs(fl[:, 1]).max()
    up = apply(sim3, to_sfm(np.array([[0, 1.0, 0]]))) - apply(sim3, to_sfm(np.zeros((1, 3))))
    assert up[0, 1] > 0.95 * np.linalg.norm(up)
    assert sim3["diagnostics"]["points_below_floor"] >= 1500
    assert "RANSAC" in sim3["floor_source"]


def test_attach_sim3_from_dancers():
    """A camera known only up to a similarity (own sub-model) is placed in the main frame from dancer keypoints."""
    from scipy.spatial.transform import Rotation
    from dancecap.cameras.attach import compose, solve_sim3
    rng = np.random.default_rng(2)
    Fn = 60
    K = np.repeat(np.array([[[600.0, 0, 239.5], [0, 600, 319.5], [0, 0, 1]]]), Fn, 0)
    K[:, 0, 0] = K[:, 1, 1] = 600 * np.exp(0.2 * np.sin(np.arange(Fn) / 10))  # zoom
    k1 = np.zeros(Fn)
    R_m, t_m = [], []
    for i in range(Fn):  # handheld wobble around (4, 1.4, -3) looking at the dancers
        C = np.array([4.0, 1.4, -3.0]) + 0.3 * np.array([np.sin(i / 9), 0.2 * np.cos(i / 7), np.cos(i / 11)])
        R, t = _look_at(C, np.array([0, 1.0, 0]) + rng.normal(0, 0.05, 3))
        R_m.append(R), t_m.append(t)
    R_m, t_m = np.array(R_m), np.array(t_m)
    Rs = Rotation.from_rotvec([0.3, -1.2, 0.5]).as_matrix()
    s, T = 2.5, np.array([1.0, -2.0, 0.5])
    R_sub = R_m @ Rs  # inverse of compose
    t_sub = (t_m + np.einsum("fij,j->fi", R_m, T)) / s
    fidx, X, uv = [], [], []
    for i in range(Fn):
        P = rng.normal([0, 1.0, 0], [0.3, 0.5, 0.3], (20, 3))
        Xc = P @ R_m[i].T + t_m[i]
        p = Xc[:, :2] / Xc[:, 2:] * K[i, 0, 0] + K[i, :2, 2]
        fidx += [i] * 20
        X.append(P), uv.append(p + rng.normal(0, 2.0, p.shape))
    fidx, X, uv = np.array(fidx), np.concatenate(X), np.concatenate(uv)
    uv[::25] += 80  # outliers
    sol = solve_sim3(R_sub, t_sub, K, k1, fidx, X, uv, np.ones(len(fidx)))
    Rr, tr = compose(R_sub, t_sub, Rotation.from_matrix(sol["Rs"]).as_rotvec(), np.log(sol["s"]), sol["T"])
    C_true = -np.einsum("fji,fj->fi", R_m, t_m)
    C_est = -np.einsum("fji,fj->fi", Rr, tr)
    assert np.abs(C_est - C_true).max() < 0.05, np.abs(C_est - C_true).max()
    assert abs(sol["s"] / s - 1) < 0.1 and sol["median_px"] < 3.0


def test_attach_with_floor_and_far_dancers():
    """Far, small dancers + a nearly static camera: dancers alone cannot fix the scale; the floor can."""
    from scipy.spatial.transform import Rotation
    from dancecap.cameras.attach import compose, floor_in_model, solve_sim3
    rng = np.random.default_rng(4)
    Fn = 60
    K = np.repeat(np.array([[[700.0, 0, 239.5], [0, 700, 319.5], [0, 0, 1]]]), Fn, 0)
    k1 = np.zeros(Fn)
    R_m, t_m = [], []
    for i in range(Fn):  # 6 m from the dancers, only ~3 cm of hand shake
        C = np.array([6.0, 1.5, 0.0]) + 0.03 * rng.normal(size=3)
        R, t = _look_at(C, np.array([0, 1.0, 0]) + rng.normal(0, 0.05, 3))
        R_m.append(R), t_m.append(t)
    R_m, t_m = np.array(R_m), np.array(t_m)
    Rs = Rotation.from_rotvec([0.2, 0.9, -0.4]).as_matrix()
    s, T = 3.0, np.array([0.5, 1.0, -2.0])
    R_sub = R_m @ Rs
    t_sub = (t_m + np.einsum("fij,j->fi", R_m, T)) / s
    to_sub = lambda X: (X - T) @ Rs / s  # noqa: E731  main -> sub
    floor_pts = np.c_[rng.uniform(-4, 8, 2000), np.zeros(2000), rng.uniform(-4, 4, 2000)]
    refl = np.c_[rng.uniform(-4, 8, 800), -rng.uniform(0.5, 3, 800), rng.uniform(-4, 4, 800)]
    C_sub = -np.einsum("fji,fj->fi", R_sub, t_sub)
    up_sub = Rs.T @ np.array([0, 1.0, 0])
    fl = floor_in_model(to_sub(np.r_[floor_pts, refl]), C_sub, up_sub)
    assert fl is not None and abs(np.dot(fl[0], up_sub)) > 0.999
    fidx, X, uv = [], [], []
    for i in range(Fn):
        P = rng.normal([0, 1.0, 0], [0.25, 0.45, 0.25], (20, 3))
        Xc = P @ R_m[i].T + t_m[i]
        fidx += [i] * 20
        X.append(P), uv.append(Xc[:, :2] / Xc[:, 2:] * 700 + [239.5, 319.5] + rng.normal(0, 3.0, (20, 2)))
    fidx, X, uv = np.array(fidx), np.concatenate(X), np.concatenate(uv)
    sol = solve_sim3(R_sub, t_sub, K, k1, fidx, X, uv, np.ones(len(fidx)), floor_sub=(fl[0], fl[1]),
                     floor_main=(np.array([0, 1.0, 0]), 0.0))
    assert abs(sol["s"] / s - 1) < 0.03, sol["s"]
    Rr, tr = compose(R_sub, t_sub, Rotation.from_matrix(sol["Rs"]).as_rotvec(), np.log(sol["s"]), sol["T"])
    C_est = -np.einsum("fji,fj->fi", Rr, tr)
    C_true = -np.einsum("fji,fj->fi", R_m, t_m)
    assert np.abs(C_est - C_true).max() < 0.1
    assert sol["floor_normal_err_deg"] < 0.5


def test_attach_rotation_only():
    """A phone that only rotates (sub-model translations meaningless): R + constant centre from the dancers."""
    from scipy.spatial.transform import Rotation
    from dancecap.cameras.attach import solve_rotonly
    rng = np.random.default_rng(7)
    Fn = 80
    K = np.repeat(np.array([[[640.0, 0, 239.5], [0, 640, 319.5], [0, 0, 1]]]), Fn, 0)
    C = np.array([-3.0, 1.1, 2.5])
    R_m = np.array([_look_at(C, np.array([0.3 * np.sin(i / 15), 1.0, 0.3 * np.cos(i / 20)]))[0] for i in range(Fn)])
    t_m = -np.einsum("fij,j->fi", R_m, C)
    Rs = Rotation.from_rotvec([1.0, -0.3, 2.0]).as_matrix()
    R_sub = R_m @ Rs
    fidx, X, uv = [], [], []
    for i in range(Fn):  # dancers wander over a 2 m area
        P = rng.normal([np.sin(i / 10), 1.0, np.cos(i / 13)], [0.25, 0.45, 0.25], (17, 3))
        Xc = P @ R_m[i].T + t_m[i]
        fidx += [i] * 17
        X.append(P), uv.append(Xc[:, :2] / Xc[:, 2:] * 640 + [239.5, 319.5] + rng.normal(0, 3.0, (17, 2)))
    fidx, X, uv = np.array(fidx), np.concatenate(X), np.concatenate(uv)
    uv[::30] += 60
    sol = solve_rotonly(R_sub, K, np.zeros(Fn), fidx, X, uv, np.ones(len(fidx)))
    assert np.linalg.norm(sol["C"] - C) < 0.1, sol["C"]
    ang = np.degrees(np.linalg.norm(Rotation.from_matrix(sol["Rs"] @ Rs.T).as_rotvec()))
    assert ang < 0.5 and sol["median_px"] < 4.0


def test_focal_stats_zoom_vs_spike():
    from dancecap.cameras.pipeline import focal_stats
    t = np.arange(120) / 30.0
    f = np.full(120, 800.0)
    f[40:60] = 800 * np.exp(np.linspace(0, 0.3, 20))  # a pinch zoom
    f[60:] = f[59]
    tr = {"t": t, "f": f, "f_raw": f, "ok_raw": np.ones(120, bool), "maxdim": 864}
    st = focal_stats(tr)
    assert st["plausible"] and st["zoom_events"] and st["spikes_outside_zoom"] == 0
    f2 = f.copy()
    f2[100] *= 1.05  # isolated jump that comes straight back
    st2 = focal_stats({**tr, "f": f2, "f_raw": f2})
    assert not st2["plausible"] and st2["spikes_outside_zoom"] == 1
    f3 = 800 * np.exp(0.05 * np.sin(np.arange(120) / 3.0))  # +-5 % wobble every ~2 s: not a zoom
    st3 = focal_stats({**tr, "f": f3, "f_raw": f3})
    assert not st3["plausible"] and st3["oscillation"] > 3


def _room_segments(R, t, K, rng, n_vert=25, n_horiz=40, n_floor=30, noise_px=0.5):
    """Pixel segments of a synthetic room (world Y up) seen by camera (R, t, K): vertical wall edges, horizontal
    wall edges, and floor lines (horizontal, below the horizon) - the last are traps for a vertical estimator."""
    segs, lab = [], []

    def add(P0, P1, kind):
        X = np.stack([P0, P1]) @ R.T + t
        if (X[:, 2] < 0.5).any():
            return
        p = X[:, :2] / X[:, 2:] * K[0, 0] + K[:2, 2]
        segs.append((p + rng.normal(0, noise_px, p.shape)).ravel())
        lab.append(kind)
    for _ in range(n_vert):  # pillars / door frames on the walls z = 8 and x = +-6, 0.5-3.5 m high
        x, z = rng.uniform(-6, 6), 8.0
        y0 = rng.uniform(0.5, 2.0)
        add(np.array([x, y0, z]), np.array([x, y0 + rng.uniform(0.8, 1.5), z]), 0)
    for _ in range(n_horiz):
        y, z, x0 = rng.uniform(1.5, 3.5), 8.0, rng.uniform(-6, 4)
        add(np.array([x0, y, z]), np.array([x0 + rng.uniform(0.5, 2), y, z]), 1)
    for _ in range(n_floor):  # tape lines on the floor running away from the camera
        x, z0 = rng.uniform(-2, 2), rng.uniform(2, 4)
        add(np.array([x, 0.0, z0]), np.array([x + rng.uniform(-0.1, 0.1), 0.0, z0 + rng.uniform(1, 3)]), 2)
    return np.array(segs), np.array(lab)


def test_vertical_vanishing_point_and_horizon():
    """Vertical from wall edges, robust to horizontal edges and to floor lines that look vertical in the image."""
    from scipy.spatial.transform import Rotation
    from dancecap.cameras.vertical import above_horizon, angle_deg, frame_vertical, segment_normals, vertical_from_segments
    rng = np.random.default_rng(11)
    K = np.array([[700.0, 0, 239.5], [0, 700, 431.5], [0, 0, 1]])
    for roll, pitch in ((3.0, -12.0), (-6.0, -5.0), (1.0, -20.0)):
        R0, t0 = _look_at(np.array([0.3, 1.5, -2.0]), np.array([0.0, 1.0, 6.0]))
        Rr = Rotation.from_euler("zx", [roll, pitch + 10], degrees=True).as_matrix()
        R = Rr @ R0
        t = -R @ np.array([0.3, 1.5, -2.0])
        seg, lab = _room_segments(R, t, K, rng)
        _, _, P = segment_normals(seg, K)
        v_true = R @ np.array([0, 1.0, 0])
        prior = Rotation.from_rotvec(rng.normal(0, 1, 3) * np.radians(3.5)).as_matrix() @ v_true  # ~3-8 deg off
        res = vertical_from_segments(seg, K, 0.0, prior)
        assert res is not None and angle_deg(res["v"], v_true) < 0.3, (roll, pitch, angle_deg(prior, v_true))
        # floor lines (vertical-looking in the image, horizontal in 3D) are never used
        assert (lab == 2).sum() > 10 and not res["used"][lab == 2].any() and res["used"][lab == 0].sum() >= 6
        assert not above_horizon(P, v_true)[lab == 2].any()
    # image path: draw a synthetic frame with the segments and run LSD end to end
    img = np.full((864, 480, 3), 200, np.uint8)
    R, t = _look_at(np.array([0.0, 1.5, -2.0]), np.array([0.0, 1.0, 6.0]))
    R = Rotation.from_euler("z", 4.0, degrees=True).as_matrix() @ R
    t = -R @ np.array([0.0, 1.5, -2.0])
    for s in _room_segments(R, t, K, np.random.default_rng(3), n_vert=30, noise_px=0.0)[0]:
        q = np.round(s * 16).astype(int)  # sub-pixel endpoints (shift=4)
        cv2.line(img, (q[0], q[1]), (q[2], q[3]), (20, 20, 20), 3, cv2.LINE_AA, shift=4)
    res = frame_vertical(img, None, K, 0.0, R @ np.array([0, 1.0, 0.05]) / np.linalg.norm([0, 1, 0.05]))
    assert res is not None and angle_deg(res["v"], R @ np.array([0, 1.0, 0])) < 0.5


def test_horizontal_directions_detect_floor_tilt():
    """Horizontal building edges are perpendicular to the true vertical (elevation ~ 0)."""
    from dancecap.cameras.vertical import horizontal_directions, segment_normals
    rng = np.random.default_rng(5)
    K = np.array([[700.0, 0, 239.5], [0, 700, 431.5], [0, 0, 1]])
    N, W = [], []
    for k in range(6):
        C = np.array([rng.uniform(-3, 3), 1.5, -2.0 - rng.uniform(0, 2)])
        R, t = _look_at(C, np.array([rng.uniform(-1, 1), 1.0, 6.0]))
        seg, _ = _room_segments(R, t, K, rng, n_vert=0, n_floor=0, n_horiz=60)
        n, L, _ = segment_normals(seg, K)
        N.append(n @ R), W.append(L)
    hd = horizontal_directions(np.concatenate(N), np.concatenate(W), np.array([0, 1.0, 0]), n_dirs=1)
    assert hd and hd[0]["elev_vs_u_deg"] < 0.2 and abs(abs(hd[0]["d"][0]) - 1) < 1e-3


def test_bootstrap_with_architecture_vertical():
    """Ankle plane tilted 5 deg against the building (mis-reconstructed floor patch): gravity = building vertical,
    y = 0 still on the dancers' floor near the origin."""
    from scipy.spatial.transform import Rotation
    from dancecap.cameras.world import mode_1d
    rng = np.random.default_rng(8)
    stat = {"lead": 1.75, "follow": 1.65}
    tilt = Rotation.from_euler("x", 5.0, degrees=True).as_matrix()  # the floor patch is rotated about x
    T = 300
    dancers = {}
    for role, H in stat.items():
        base = np.c_[rng.normal(0, 0.4, T), np.zeros(T), rng.normal(0, 0.4, T)]
        ank_h = 0.039 * np.mean(list(stat.values())) + 0.025
        ank = base + [0, ank_h, 0]
        eye = base + np.c_[np.zeros(T), np.where(rng.random(T) < 0.3, 0.936 * H + 0.025, 0.8 * H), np.zeros(T)]
        dancers[role] = {1: eye @ tilt.T, 2: eye @ tilt.T, 15: ank @ tilt.T, 16: (ank + [0.2, 0, 0]) @ tilt.T}
    floor = np.c_[rng.uniform(-2.5, 2.5, 2000), np.zeros(2000), rng.uniform(-2.5, 2.5, 2000)] @ tilt.T
    tracks = {}
    for k, cam in enumerate(["06", "01", "02"]):
        a = 2 * np.pi * k / 3
        R, tv = _look_at(np.array([6 * np.cos(a), 1.4, 6 * np.sin(a)]), np.array([0, 1.0, 0]))
        tracks[cam] = {"R": np.repeat(R[None], 10, 0), "tvec": np.repeat(tv[None], 10, 0), "ok": np.ones(10, bool)}
    vert = {"u": np.array([0, 1.0, 0]), "n_cams": 3, "max_cam_deg": 0.4}
    sim3 = bootstrap(tracks, dancers, floor, stat, "test", vertical=vert)
    Rw = np.asarray(sim3["R"])
    assert np.degrees(np.arccos(Rw[1, 1])) < 1e-6  # world Y = building vertical exactly
    d = sim3["diagnostics"]
    assert d["vertical"]["adopted"] and abs(d["vertical"]["vs_ankle_plane_deg"] - 5.0) < 1.0
    near = apply(sim3, floor[np.linalg.norm(floor[:, [0, 2]], axis=1) < 0.5])
    assert np.abs(np.median(near[:, 1])) < 0.02  # floor at y = 0 under the dancers
    without = bootstrap(tracks, dancers, floor, stat, "test")
    assert np.degrees(np.arccos(np.asarray(without["R"])[1, 1])) > 4.0  # old behaviour: follows the ankle plane
    m, inl = mode_1d(np.r_[rng.normal(1.0, 0.01, 200), rng.uniform(-5, 5, 200)], 0.05)
    assert abs(m - 1.0) < 0.005 and inl[:200].mean() > 0.9


def test_attach_rotation_only_with_vertical():
    """Weak roll/tilt from the dancers is fixed by the building's vertical seen in the sub-model frame."""
    from scipy.spatial.transform import Rotation
    from dancecap.cameras.attach import holdout, rotonly_errors, solve_rotonly
    rng = np.random.default_rng(9)
    Fn = 80
    K = np.repeat(np.array([[[640.0, 0, 239.5], [0, 640, 319.5], [0, 0, 1]]]), Fn, 0)
    C = np.array([-4.0, 1.0, 3.0])
    R_m = np.array([_look_at(C, np.array([0.2 * np.sin(i / 15), 1.0, 0.2 * np.cos(i / 20)]))[0] for i in range(Fn)])
    Rs = Rotation.from_rotvec([0.4, 1.3, -0.7]).as_matrix()
    R_sub = R_m @ Rs
    u_main = np.array([0.03, 0.999, -0.02])
    u_main /= np.linalg.norm(u_main)
    u_sub = Rs.T @ u_main  # what the vanishing points in the sub-model frame give
    fidx, X, uv = [], [], []
    t_m = -np.einsum("fij,j->fi", R_m, C)
    for i in range(Fn):  # dancers bunched in a small area far away: tilt is weakly observed
        P = rng.normal([0.3 * np.sin(i / 10), 1.0, 0.3 * np.cos(i / 13)], [0.15, 0.45, 0.15], (17, 3))
        Xc = P @ R_m[i].T + t_m[i]
        fidx += [i] * 17
        X.append(P), uv.append(Xc[:, :2] / Xc[:, 2:] * 640 + [239.5, 319.5] + rng.normal(0, 8.0, (17, 2)))
    fidx, X, uv = np.array(fidx), np.concatenate(X), np.concatenate(uv)
    sol = solve_rotonly(R_sub, K, np.zeros(Fn), fidx, X, uv, np.ones(len(fidx)), vertical=(u_sub, u_main))
    assert sol["vertical_err_deg"] < 1e-6
    ang = np.degrees(np.linalg.norm(Rotation.from_matrix(sol["Rs"] @ Rs.T).as_rotvec()))
    assert ang < 1.0 and np.linalg.norm(sol["C"] - C) < 0.3, (ang, sol["C"])  # yaw: from noisy dancers only
    t_fr = np.arange(Fn) / 30.0
    ho = holdout(lambda sel: solve_rotonly(R_sub, K, np.zeros(Fn), fidx[sel], X[sel], uv[sel], np.ones(sel.sum()),
                                           vertical=(u_sub, u_main)),
                 lambda s, sel: rotonly_errors(s, R_sub, K, np.zeros(Fn), fidx[sel], X[sel], uv[sel]), fidx, t_fr)
    assert ho["n"] == len(fidx) and ho["median_px"] < 1.3 * sol["median_px"] + 1


def test_sim3_plausibility_guard():
    from dancecap.cameras.attach import sim3_plausible
    Fn = 20
    R_sub = np.repeat(np.eye(3)[None], Fn, 0)
    C_s = np.c_[np.linspace(0, 0.1, Fn), np.full(Fn, 1.0), np.zeros(Fn)]
    t_sub = -C_s
    floor = (np.array([0, 1.0, 0]), 0.0)
    good = {"Rs": np.eye(3), "s": 1.2, "T": np.zeros(3), "floor_normal_err_deg": 0.5}
    assert sim3_plausible(good, R_sub, t_sub, floor, floor, 1.0)[0]
    collapsed = {**good, "s": 0.005}
    assert not sim3_plausible(collapsed, R_sub, t_sub, floor, floor, 1.0)[0]  # camera on the floor
    tilted = {**good, "floor_normal_err_deg": 6.5}
    assert not sim3_plausible(tilted, R_sub, t_sub, floor, floor, 1.0)[0]
    huge = {**good, "s": 60.0, "T": np.array([0, -58.0, 0])}  # 1.2 m high but the centre wanders 6 m
    assert not sim3_plausible(huge, R_sub, t_sub, floor, floor, 1.0)[0]


def test_json_bool_encoding():
    from dancecap.cameras.pipeline import _np
    s = json.dumps({"a": np.bool_(True), "b": np.float32(1.5), "c": np.int64(3)}, default=_np)
    assert json.loads(s) == {"a": True, "b": 1.5, "c": 3}


def test_cache_key_invalidation():
    import tempfile
    from dancecap.cameras.pipeline import _cached
    tmp_path = Path(tempfile.mkdtemp())
    calls = []
    f = lambda: calls.append(1) or len(calls)  # noqa: E731
    p = tmp_path / "c.pkl"
    assert _cached(p, {"a": 1}, f) == 1 and _cached(p, {"a": 1}, f) == 1
    assert _cached(p, {"a": 2}, f) == 2  # settings changed -> recomputed
    p.write_bytes(b"not a pickle")
    assert _cached(p, {"a": 2}, f) == 3


# ------------------------------------------------------------------ contract checks on real outputs
def test_real_outputs_contract():
    out = WORK / "cameras"
    if not (out / "report.json").exists():
        return
    rep = json.loads((out / "report.json").read_text())
    assert "gates_G2a" in rep
    for cam in CAMS:
        p = out / f"{cam}.npz"
        if not p.exists():
            continue
        d = np.load(p)
        F = len(json.loads((WORK / "frames" / cam / "frames.json").read_text())["frames"])
        assert d["t"].shape == (F,) and d["K"].shape == (F, 3, 3) and d["R"].shape == (F, 3, 3)
        assert d["tvec"].shape == (F, 3) and d["dist"].shape == (F,) and d["ok"].dtype == bool
        assert d["reproj_px"].shape == (F,) and d["n_inliers"].shape == (F,)
        assert np.allclose(np.einsum("fij,fkj->fik", d["R"], d["R"]), np.eye(3), atol=1e-6)
        assert np.all(np.isfinite(d["R"])) and np.all(np.isfinite(d["tvec"])) and np.all(d["K"][:, 0, 0] > 0)
        if "attached" in d.files:  # additive keys (cameras-fix): scalar bool + per-frame dancer residual
            assert d["attached"].shape == () and d["attached"].dtype == bool
        if "dancer_px" in d.files:
            assert d["dancer_px"].shape == (F,)
    if (out / "world_from_sfm.json").exists():
        w = json.loads((out / "world_from_sfm.json").read_text())
        R = np.asarray(w["R"])
        assert w["s"] > 0 and np.allclose(R @ R.T, np.eye(3), atol=1e-6) and np.linalg.det(R) > 0
        assert {"gravity_source", "scale_source"} <= set(w)
        if "vertical" in w["diagnostics"] and w["diagnostics"]["vertical"].get("adopted"):
            assert "architecture vertical" in w["gravity_source"]
    for cam, v in rep.get("cams", {}).items():  # labels and JSON booleans (no "True" strings)
        if v.get("attached"):
            assert v["how"].endswith(f"({v['attach']['method']})")
            assert isinstance(v["attach"]["sub_floor"]["usable"], bool)
    assert EYE_MINUS_ANKLE > 0.8


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
