"""Pure helpers of dancecap.mono4d.jl_v2_render (no work/ data needed)."""
import numpy as np

import sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # run from the workspace root too
from dancecap.mono4d import jl_v2_render as V


def test_zoom_envelope_identity_when_nothing_needed():
    need = np.full(100, 1.3)
    assert np.array_equal(V.zoom_envelope(need, 10, 3.0), np.ones(100))


def test_zoom_envelope_covers_need_and_is_smooth():
    need = np.ones(200)
    need[95:100] = 0.94
    s = V.zoom_envelope(need, 18, 6.0)
    assert (s[95:100] <= 0.94 + 1e-9).all()          # never less zoom-out than needed
    assert s[0] == 1.0 and s[-1] == 1.0               # untouched far away
    assert np.abs(np.diff(s)).max() < 0.02            # no jump
    assert s.min() >= 0.94 - 1e-9                     # no more zoom-out than the worst need


def test_focal_need_point_on_border_and_inside():
    K = np.array([[[100.0, 0, 49.5], [0, 100.0, 49.5], [0, 0, 1]]])
    R = np.eye(3)[None]
    t = np.zeros((1, 3))
    # a point projecting to u = 49.5 + 100 * 0.6 = 109.5 (outside a 100 px image): need = (99 - 49.5 - m) / 60
    P = np.array([[[0.6, 0.0, 1.0]]])
    n = V.focal_need(K, R, t, P, 100, 100, 1.0)
    assert np.isclose(n[0], (99 - 49.5 - 1.0) / 60.0)
    P2 = np.array([[[0.1, -0.1, 1.0]]])
    assert V.focal_need(K, R, t, P2, 100, 100, 1.0)[0] > 1


def test_once_index_shows_each_source_frame_once():
    idx = V.once_index(2643, 2643)
    assert len(np.unique(idx)) == 2643 and (np.diff(idx) == 1).all()


def test_nearest_index_repeats_a_frame_at_the_rate_mismatch():
    t_src = np.arange(2643) / V.VIDEO_FPS
    idx = V.nearest_index(np.arange(2643) / V.FPS, t_src)
    assert (np.diff(idx) == 0).sum() >= 1             # the old side-by-side's repeated frame


def test_alias_rows_blend_and_clamp():
    tt = np.arange(10) / 30.0
    j, w = V.alias_rows(np.array([0.0, 0.05, 1.0]), tt)
    assert j[0] == 0 and w[0] == 0
    assert j[1] == 1 and np.isclose(w[1], 0.5)
    assert j[2] == 9 and w[2] == 0                    # after the last track key: hold the last row


def test_crop_centres_clamped():
    c = V.crop_centres(np.array([0.0, 1920.0, 960.0]), 1920, 960, 0.0)
    assert list(c) == [480, 1440, 960]


def test_stretch_rate_drift_small():
    s = V.stretch_rate(2643)
    assert s["asetrate"] == 44109
    assert s["drift_ms"] < 0.5
