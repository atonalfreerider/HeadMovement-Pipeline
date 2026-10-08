"""C7 map extension geometry: fixed-pose two-view triangulation under SIMPLE_RADIAL, failed-stretch grouping."""
import numpy as np

from dancecap import map_extend as me


def _rot(yaw_deg):
    a = np.radians(yaw_deg)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def test_triangulate_recovers_points_with_distortion():
    rng = np.random.default_rng(0)
    w, h, f, k1 = 480, 848, 700.0, -0.05
    X = rng.uniform([-2, -1, 5], [2, 1, 9], (200, 3))
    pa = {"R": np.eye(3), "t": np.zeros(3), "f": f, "k1": k1}
    Rb = _rot(3.0)
    Cb = np.array([0.4, 0.05, 0.0])
    pb = {"R": Rb, "t": -Rb @ Cb, "f": f * 1.02, "k1": k1}
    xa, _ = me.project(pa["R"], pa["t"], pa["f"], pa["k1"], w, h, X)
    xb, _ = me.project(pb["R"], pb["t"], pb["f"], pb["k1"], w, h, X)
    Y, good = me.triangulate(pa, pb, xa, xb, w, h)
    assert good.mean() > 0.95
    assert np.abs(Y[good] - X[good]).max() < 1e-6
    # a mismatched pair (wrong correspondence) is rejected by the reprojection test
    Y2, good2 = me.triangulate(pa, pb, xa, xb[::-1], w, h)
    assert good2.mean() < 0.1


def test_tiny_baseline_is_rejected():
    w, h, f = 480, 848, 700.0
    X = np.array([[0.0, 0.0, 8.0], [0.5, 0.2, 7.0]])
    pa = {"R": np.eye(3), "t": np.zeros(3), "f": f, "k1": 0.0}
    pb = {"R": np.eye(3), "t": -np.array([0.01, 0, 0]), "f": f, "k1": 0.0}  # 1 cm at 8 m: < 1 degree
    xa, _ = me.project(pa["R"], pa["t"], f, 0.0, w, h, X)
    xb, _ = me.project(pb["R"], pb["t"], f, 0.0, w, h, X)
    _, good = me.triangulate(pa, pb, xa, xb, w, h)
    assert not good.any()


def test_stretches_merge_short_islands():
    ok = np.ones(100, bool)
    ok[10:20] = False
    ok[25:30] = False   # 5-frame island 20-24 -> merged with 10-19
    ok[70:72] = False
    assert me.stretches(ok, merge=15) == [(10, 29), (70, 71)]
    assert me.stretches(np.ones(5, bool)) == []
