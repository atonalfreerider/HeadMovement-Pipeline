"""export_unity.park_absent: a dancer the stitcher flagged `absent` is parked under the other dancer, not shown as motion.

    .venv-nopo4d\\Scripts\\python.exe -m pytest dancecap/tests/test_park_absent.py -q
Pure numpy: no take, model or Unity needed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dancecap import export_unity as eu  # noqa: E402


def _data(T=10, absent=4):
    t = 30.0 + np.arange(T) / 30.0
    follow_j = np.zeros((T, 55, 3))
    follow_j[:, 0] = np.stack([np.linspace(0.0, 0.9, T), np.full(T, 1.0), np.linspace(0.0, -0.9, T)], axis=1)
    lead_j = np.zeros((T, 55, 3))
    lead_j[:, 0] = [2.0, 1.0, 3.0]
    lead_j[:, 5] = [2.0, 0.2, 3.1]
    ab = np.zeros(T, bool)
    ab[:absent] = True
    return {"lead": {"transl": lead_j[:, 0] + [0.0, 0.3, 0.0], "joints": lead_j, "absent": ab, "t": t},
            "follow": {"transl": follow_j[:, 0] + [0.0, 0.3, 0.0], "joints": follow_j, "absent": np.zeros(T, bool), "t": t}}


def test_absent_frames_are_parked_under_the_partner():
    d = _data()
    lead0, follow0 = d["lead"]["joints"].copy(), d["follow"]["joints"].copy()
    rep = eu.park_absent(d, depth_m=30.0)
    j = d["lead"]["joints"]
    assert np.allclose(j[:4, 0, 0], follow0[:4, 0, 0]) and np.allclose(j[:4, 0, 2], follow0[:4, 0, 2])   # XZ = her pelvis
    assert np.allclose(j[:4, 0, 1], 1.0 - 30.0)                                                          # 30 m under the floor
    assert np.allclose(j[:4, 5] - j[:4, 0], lead0[:4, 5] - lead0[:4, 0])                                  # rigid: pose untouched
    assert np.array_equal(j[4:], lead0[4:])                                                               # tracked frames untouched
    assert np.array_equal(d["follow"]["joints"], follow0)
    assert np.allclose(d["lead"]["transl"][:4, 1], 1.3 - 30.0)                                            # transl moved by the same shift
    assert rep["lead"]["count"] == 4 and rep["lead"]["frames"] == [[0, 3]] and "follow" not in rep
    assert rep["lead"]["t"] == [[30.0, round(30.0 + 3 / 30.0, 6)]]


def test_nothing_absent_is_a_no_op():
    d = _data(absent=0)
    before = {r: v["joints"].copy() for r, v in d.items()}
    assert eu.park_absent(d) == {}
    assert all(np.array_equal(before[r], d[r]["joints"]) for r in d)
    d2 = {"lead": {"transl": np.zeros((3, 3)), "joints": np.zeros((3, 55, 3)), "t": np.arange(3.0)},       # npz without the key
          "follow": {"transl": np.zeros((3, 3)), "joints": np.zeros((3, 55, 3)), "t": np.arange(3.0)}}
    assert eu.park_absent(d2) == {}


def test_two_runs_are_reported_separately():
    d = _data(T=12, absent=0)
    d["lead"]["absent"][[1, 2, 7]] = True
    rep = eu.park_absent(d)
    assert rep["lead"]["frames"] == [[1, 2], [7, 7]] and rep["lead"]["count"] == 3
