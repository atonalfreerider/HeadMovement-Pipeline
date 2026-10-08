"""dancecap.shoe_size helpers on synthetic tracks (no take data needed).

    .venv-nopo4d\\Scripts\\python.exe -m pytest dancecap/tests/test_shoe_size.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dancecap import shoe_size as ss  # noqa: E402


def _foot(T=40, step_at=20, jitter=0.0, seed=0):
    """heel / toe / small toe / ankle of a foot planted until step_at, then lifted and moving 3 cm per frame"""
    rng = np.random.default_rng(seed)
    X = np.zeros((T, 4, 3))
    X[:, 0] = [0.0, 0.02, 0.0]
    X[:, 1] = [0.0, 0.01, 0.25]
    X[:, 2] = [0.04, 0.01, 0.20]
    X[:, 3] = [0.0, 0.09, 0.05]
    for k in range(step_at, T):
        X[k] += [0.0, 0.08, 0.03 * (k - step_at + 1)]
    return X + rng.normal(scale=jitter, size=X.shape)


def test_planted_keeps_the_still_low_foot_only():
    p = ss.planted(_foot())
    assert p[2:17].all() and not p[22:].any()


def test_smoothing_rescues_jittery_landmarks():
    X = _foot(jitter=0.004)
    raw_steps = np.linalg.norm(np.diff(X[:, :2], axis=0), axis=2).max(1)
    assert np.median(raw_steps[:18]) > 0.006          # per-frame jitter alone would fail the stillness test
    assert ss.planted(X)[3:16].mean() > 0.6           # the smoothed track passes it


def test_smooth_track_needs_enough_finite_neighbours():
    X = _foot()
    X[10:14] = np.nan
    S = ss.smooth_track(X)
    assert np.isnan(S[10:14]).all() and np.isfinite(S[5]).all()


def test_mask_run_through_the_middle():
    on = np.zeros(100, bool)
    on[30:71] = True
    assert ss._run(on, 50) == (30, 70)
    assert ss._run(on, 26) == (30, 70)               # within the slack of the run
    on[:] = True
    assert ss._run(on, 50) is None                    # touches both ends: not a closed silhouette
    assert ss._run(np.zeros(100, bool), 50) is None


def test_ma2d_landmark_slots_are_distinct_feet():
    assert len(set(ss.MA2D_FOOT["L"]) | set(ss.MA2D_FOOT["R"])) == 8
