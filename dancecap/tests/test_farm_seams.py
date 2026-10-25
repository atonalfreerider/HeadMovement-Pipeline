"""farm.chunks.seam_windows: chunk seams placed where both cameras see the dancers; ca_profile.two_view_support per dancer."""
import numpy as np

from dancecap.farm import ca_profile as cap
from dancecap.farm import chunks as C

HZ = 30.0


def _check_tiling(w, N, max_s=60.0):
    o = w["overlap_k"]
    assert w["windows"][0][0] == 0 and w["windows"][-1][1] == N - 1 and len(w["windows"]) == w["n"] == len(w["cores"])
    for (a0, a1), (b0, b1) in zip(w["windows"], w["windows"][1:]):
        assert a1 - b0 + 1 == o                                          # neighbours overlap by exactly overlap_k
    for (c0, c1), (d0, _), (w0, w1) in zip(w["cores"], w["cores"][1:], w["windows"]):
        assert d0 == c1 + 1 and w0 <= c0 <= c1 <= w1
    assert all(w1 - w0 + 1 <= max_s * HZ for w0, w1 in w["windows"])


def test_uniform_evidence_keeps_the_equal_tiling():
    N = int(196 * HZ)
    eq = C.equal_windows(N, HZ, 60.0, 4.0)
    w = C.seam_windows(np.ones(N), HZ, 60.0, 4.0)
    assert w["windows"] == eq["windows"] and w["cores"] == eq["cores"] and w["rule"].startswith("equal windows")
    assert w["seam_scores"] == [1.0] * 3


def test_seams_leave_the_holes_of_the_moving_camera():
    N = int(196 * HZ)
    t = np.arange(N) / HZ
    sup = np.ones(N)
    for a, b in ((43.0, 67.0), (101.0, 124.0), (148.0, 158.0)):          # lucas_luiza-like holes of the Steadicam's verified track
        sup[(t >= a) & (t < b)] = 0.0
    eq = C.equal_windows(N, HZ, 60.0, 4.0)
    eq_seams = [c0 / HZ for c0, _ in eq["cores"][1:]]
    assert any(sup[int(s * HZ)] == 0 for s in eq_seams)                   # the equal tiling cuts inside a hole
    w = C.seam_windows(sup, HZ, 60.0, 4.0)
    _check_tiling(w, N)
    assert w["n"] == eq["n"] and min(w["seam_scores"]) == 1.0 and "by evidence" in w["rule"]
    for s in w["seam_t"]:
        assert sup[int((s - 2.0) * HZ):int((s + 2.0) * HZ)].min() == 1.0


def test_the_weakest_dancer_scores_a_seam_and_one_more_window_only_when_it_helps():
    N = int(110 * HZ)
    t = np.arange(N) / HZ
    sup = np.ones((N, 2))
    sup[(t >= 45.0) & (t < 65.0), 1] = 0.0                               # the follow is seen by one camera only around the middle,
    w = C.seam_windows(sup, HZ, 60.0, 4.0)                              # where two windows must cut (52-58 s): a third window moves both seams out
    _check_tiling(w, N)
    assert w["n"] == 3 and w["seam_scores"] == [1.0, 1.0] and not any(43.0 < s < 67.0 for s in w["seam_t"])
    sup[(t >= 20.0) & (t < 45.0), 1] = 0.0
    sup[(t >= 65.0) & (t < 90.0), 1] = 0.0
    w2 = C.seam_windows(sup, HZ, 60.0, 4.0)                             # the lead alone is no seam: a third window would not help either
    assert w2["n"] == 2 and w2["rule"].startswith("equal windows")      # no seam can be good: the fewest windows, equal
    assert C.seam_windows(np.ones(int(50 * HZ)), HZ, 60.0, 4.0)["n"] == 1


def test_two_view_support_needs_every_camera_and_the_moving_camera_s_track():
    grid = np.arange(0.0, 4.0, 1.0 / HZ)
    n = len(grid)
    conf = np.full((n, 2), 0.9)
    conf[30:60, 1] = 0.1                                                 # camera 02 loses the follow for 1 s
    ok01 = np.ones(n, bool)
    ok01[90:] = False                                                    # camera 01's track is not ok in the last second
    tt = {"01": {"t": grid.copy(), "ok": np.ones(n, bool), "conf": np.full((n, 2), 0.9), "cam_ok": ok01},
          "02": {"t": grid.copy(), "ok": np.ones(n, bool), "conf": conf}}
    s = cap.two_view_support(tt, grid)
    assert s.shape == (n, 2) and s[:30].all() and s[30:60, 0].all() and not s[30:60, 1].any() and not s[90:].any()
    assert not cap.two_view_support({}, grid).any()


def test_no_overlap_and_a_marginal_gain_keep_the_equal_windows():
    """Review 2026-10-25: overlap 0 made every seam score NaN; a third window was added for +0.02 of support."""
    N = int(110 * HZ)
    t = np.arange(N) / HZ
    w = C.seam_windows(np.ones(N), HZ, 60.0, 0.0)
    assert w["rule"].startswith("equal windows (no overlap") and w["windows"] == C.equal_windows(N, HZ, 60.0, 0.0)["windows"]
    sup = np.full(N, 0.5)
    sup[(t >= 30.0) & (t < 35.0)] = 0.53                                 # a third window would lift the weakest seam by 0.03 only
    sup[(t >= 75.0) & (t < 80.0)] = 0.53
    w = C.seam_windows(sup, HZ, 60.0, 4.0)
    assert w["n"] == 2


def test_forbidden_frames_never_carry_a_seam():
    N = int(196 * HZ)
    t = np.arange(N) / HZ
    sup = np.ones(N)
    forbid = (t >= 146.0) & (t < 159.0)
    w = C.seam_windows(sup, HZ, 60.0, 4.0, forbid=forbid)
    assert all(not forbid[int((s - 2.0) * HZ):int((s + 2.0) * HZ)].any() for s in w["seam_t"])
    assert "by evidence" in w["rule"]
    w = C.seam_windows(sup, HZ, 60.0, 4.0, forbid=np.ones(N, bool))       # nowhere to cut: the equal tiling (the plan check refuses it)
    assert w["rule"].startswith("equal windows")


def test_window_masks_drop_slivers():
    import dataclasses
    P = dataclasses.replace(cap.LL, cam_masks=(("01", 163.3666, 176.0, "x"),))
    assert cap.window_masks(P, 100.0, 163.36667) == []
    assert cap.window_masks(P, 100.0, 165.0) == [("01", 163.3666, 165.0)]
