"""dancecap.pose_quality slip metrics and export_unity.resolve_smplx_dir on tiny synthetic inputs."""
import numpy as np

from dancecap import pose_quality as pq
from dancecap.export_unity import resolve_smplx_dir

FPS = 30.0


def _foot(T=31, n=26):
    """A flat 25 cm sole (n points along x, y = 0) per frame -> (T, n, 3)."""
    x = np.linspace(0.0, 0.25, n)
    pts = np.zeros((T, n, 3))
    pts[..., 0] = x
    return pts


def _kin(pts):
    t = np.arange(len(pts)) / FPS
    return {"t": t, "pts": {"left": pts}, "sole": {"left": pts[..., 1].min(1)}, "cen": {"left": pts.mean(1)}}


def test_contact_point_speed_ignores_heel_toe_roll():
    pts = _foot()
    th = np.radians(np.linspace(0, 40, len(pts)))      # heel rises about the toe (x = 0.25) over 1 s
    x0 = pts[0, :, 0]
    for i, a in enumerate(th):
        pts[i, :, 0] = 0.25 - (0.25 - x0) * np.cos(a)
        pts[i, :, 1] = (0.25 - x0) * np.sin(a)
    k = _kin(pts)
    v = pq.contact_point_speed(k, "left")
    assert np.nanmedian(v) < 0.01                       # the toe stays put: no slip
    # the loaded-point centroid (stance_slip.json's first metric) travels from mid-foot to the toe: > 10 cm of "slip"
    low = pts[..., 1].min(1)
    w = pts[..., 1] <= low[:, None] + 0.01
    cen = np.array([pts[i, w[i], 0].mean() for i in range(len(pts))])
    assert cen[-1] - cen[0] > 0.1


def test_contact_point_speed_and_drift_of_a_slide():
    pts = _foot()
    pts[..., 2] += (0.3 * np.arange(len(pts)) / FPS)[:, None]    # the planted foot slides at 0.3 m/s
    k = _kin(pts)
    np.testing.assert_allclose(pq.contact_point_speed(k, "left"), 0.3, rtol=1e-6)
    m = np.ones(len(pts), bool)
    d = pq.run_drift(k["cen"]["left"], m)
    np.testing.assert_allclose(d, [0.3 * (len(pts) - 1) / FPS / 2], rtol=1e-6)   # half the run's travel
    still = _kin(_foot())
    assert pq.run_drift(still["cen"]["left"], m)[0] == 0.0
    assert np.nanmax(pq.contact_point_speed(still, "left")) == 0.0


def test_runs_of_min_len():
    m = np.array([1, 1, 0, 1, 1, 1, 1, 0, 1], bool)
    assert pq.runs_of(m, 3) == [(3, 6)]
    assert pq.runs_of(m, 1) == [(0, 1), (3, 6), (8, 8)]


def test_resolve_smplx_dir():
    assert resolve_smplx_dir("larissa_kadu_38", "smplx") == ("smplx", "argument")
    assert resolve_smplx_dir("no_such_take_xyz") == ("smplx", "default")
    d, src = resolve_smplx_dir("larissa_kadu_38")       # the take adopts a pose_refine output for the viewer
    assert d != "smplx" and "[export]" in src
