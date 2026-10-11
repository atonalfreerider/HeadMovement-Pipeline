"""john_lula 2D people stage (dancecap.jl_people): the pure parts (bib reading, merging, tracker, anchors, QA, whole-body helpers)."""
from __future__ import annotations

import cv2
import numpy as np

from dancecap import jl_people as J


def kp_person(cx: float, cy: float, h: float, conf: float = 0.9) -> np.ndarray:
    """A standing COCO-17 skeleton of height h centred horizontally at cx with its hips at cy."""
    k = np.zeros((17, 3), np.float32)
    k[:, 2] = conf
    k[0] = (cx, cy - 0.55 * h, conf)
    for j, (dx, dy) in {5: (-0.1, -0.4), 6: (0.1, -0.4), 7: (-0.15, -0.2), 8: (0.15, -0.2), 9: (-0.15, 0.0), 10: (0.15, 0.0), 11: (-0.06, 0.0), 12: (0.06, 0.0),
                        13: (-0.06, 0.22), 14: (0.06, 0.22), 15: (-0.06, 0.45), 16: (0.06, 0.45)}.items():
        k[j] = (cx + dx * h, cy + dy * h, conf)
    k[1:5] = k[0]
    return k


def test_edit_distance_and_bib_score():
    assert J.edit_distance("124", "124") == 0
    assert J.edit_distance("124", "126") == 1
    assert J.edit_distance("24", "124") == 1
    assert J.bib_score("124", "124") == 1.0
    assert J.bib_score("7124", "124") == 0.9
    assert J.bib_score("24", "124") == 0.35
    assert J.bib_score("1", "124") == 0.0                       # a single digit is not evidence
    assert J.bib_score("126", "124", ("126",)) == 0.0           # a known other bib one digit away is NOT John
    assert J.bib_score("126", "124") == 0.35
    assert J.bib_score("", "124") == 0.0


def test_person_bib_votes():
    j, l, o = J.person_bib([("124", 0.9)])
    assert j > 0.9 and l == 0 and o == 0
    j, l, o = J.person_bib([("116", 0.9), ("126", 0.9)])
    assert l > 0.9 and j == 0 and o > 0.9
    j, l, o = J.person_bib([("124", 0.1)])                      # a doubtful read counts for nothing
    assert j == 0


def test_box_helpers():
    a, b = np.array([0, 0, 10, 10.0]), np.array([5, 0, 15, 10.0])
    assert abs(J.box_iou(a, b) - 1 / 3) < 1e-6
    assert J.box_cover(a, np.array([2, 2, 4, 4.0])) == 1.0
    assert J.box_iou(a, np.array([20, 20, 30, 30.0])) == 0.0


def test_merge_passes_removes_duplicates_and_keeps_the_better():
    box = np.array([[0, 0, 100, 300], [2, 1, 101, 302], [200, 0, 300, 300]], np.float32)
    kp = np.stack([kp_person(50, 150, 300, 0.4), kp_person(50, 150, 300, 0.9), kp_person(250, 150, 300, 0.8)])
    conf = np.array([0.5, 0.9, 0.8], np.float32)
    m = J.merge_passes(box, kp, conf, np.array([0, 0, 0]), 1)
    assert len(m["idx"]) == 2 and 1 in m["idx"] and 0 not in m["idx"]
    assert sorted(m["src"][0] + m["src"][1]) == [0, 1, 2]


def test_pair_states_merged_only_where_allowed():
    s = J.pair_states(2)
    assert not any(a == b and a > 0 for a, b in s)
    s = J.pair_states(2, np.array([True, False]))
    assert any(a == b == 1 for a, b in s) and not any(a == b == 2 for a, b in s)


def test_scale_track_follows_the_three_zoom_steps():
    S = J.scale_track(2643, 29.994)
    assert abs(S[0] - 1.0) < 1e-9 and abs(S[int(30 * 29.994)] - 1.97) < 1e-6
    assert abs(S[int(60 * 29.994)] - 1.97 * 0.51) < 1e-6 and abs(S[-1] - 1.97 * 0.51 * 1.89) < 1e-6
    assert 1.0 < S[int(18.2 * 29.994)] < 1.97                    # inside the first ramp


def test_track_pair_keeps_identity_through_a_crossing_with_a_distractor():
    """John (slot A) and Lula (slot B) walk past each other while a bystander stands still: the colour evidence decides the roles, the continuity the crossing."""
    n = 40
    frames = []
    for i in range(n):
        xj, xl = 100 + 8 * i, 420 - 8 * i                       # they cross at i = 20
        c = np.array([[xj, 500.0], [xl, 500.0], [260.0, 520.0]])
        frames.append({"ids": np.arange(3), "c": c, "h": np.array([400.0, 400.0, 380.0]), "sj": np.array([3.0, -3.0, 0.5]), "sl": np.array([-3.0, 3.0, 0.5]),
                       "pc": np.zeros((3, 3)), "cw": (frames[-1]["c"] if frames else np.zeros((0, 2))), "same_ok": np.zeros(3, bool)})
    res = J.track_pair(frames)
    assert (res["cand"][:, 0] == 0).all() and (res["cand"][:, 1] == 1).all()
    assert res["p"].min() > 0.5


def test_track_pair_missing_dancer_is_missing_not_replaced_by_the_distractor():
    n = 20
    frames = []
    for i in range(n):
        c = np.array([[100.0, 500.0], [260.0, 520.0]])
        frames.append({"ids": np.arange(2), "c": c, "h": np.array([400.0, 380.0]), "sj": np.array([3.0, -4.0]), "sl": np.array([-3.0, -4.0]), "pc": np.zeros((2, 2)),
                       "cw": (frames[-1]["c"] if frames else np.zeros((0, 2))), "same_ok": np.zeros(2, bool)})
    res = J.track_pair(frames)
    assert (res["cand"][:, 0] == 0).all() and (res["cand"][:, 1] == -1).all()


def test_pick_anchors_and_fill_gaps():
    q = np.zeros(400)
    q[[10, 15, 100, 250]] = [0.5, 0.9, 0.8, 0.7]
    an = J.pick_anchors(q, min_gap=24)
    assert an == [15, 100, 250]                                   # 10 is within 24 frames of the better 15
    qr = np.zeros(400)
    qr[[170, 330, 390]] = 0.4
    out = J.fill_gaps(an, qr, max_gap=90, min_gap=24)
    assert 170 in out and np.diff(out).max() <= 150 and out == sorted(out)


def test_choose_variant_prefers_masked_unless_plain_is_clearly_better():
    kp = np.zeros((2, 2, 2, 133, 3), np.float32)
    kp[..., 2] = 4.0
    kp[1, 0, 0, :17, 2] = 6.0                                     # frame 1, John: the plain variant is clearly better
    out, pick = J.choose_variant(kp)
    assert pick[0, 0] == 1 and pick[1, 0] == 0 and out.shape == (2, 2, 133, 3)
    assert out[1, 0, 0, 2] == 6.0


def test_anchors_from_skeletons_builds_box_points_and_negatives():
    kp = np.full((1, 2, 133, 3), np.nan, np.float32)
    kp[0, 0, :17] = kp_person(100, 500, 400) * [1, 1, 0] + [0, 0, 6.0]
    kp[0, 1, :17] = kp_person(220, 500, 380) * [1, 1, 0] + [0, 0, 6.0]
    doc = J.anchors_from_skeletons(kp, [0])
    assert len(doc) == 1 and doc[0]["frame"] == 0
    j, l = doc[0]["john"], doc[0]["lula"]
    assert j["box"][0] < 100 < j["box"][2] and len(j["pos"]) == 10 and len(j["neg"]) == 4
    assert j["neg"][0][0] > 150                                    # the negatives are on Lula
    kp[0, 1, :17, 2] = 1.0                                         # a dancer without confident joints: no anchor
    assert J.anchors_from_skeletons(kp, [0]) == []


def test_mask_helpers():
    a = np.zeros((20, 20), bool)
    b = np.zeros((20, 20), bool)
    a[2:10, 2:10] = True
    b[6:14, 2:10] = True
    assert abs(J.mask_iou(a, b) - 32 / 96) < 1e-9
    assert np.isnan(J.mask_iou(a, None))
    assert list(J.mask_box(a)) == [2, 2, 10, 10]
    speck = a.copy()
    speck[15, 15] = True
    assert not J.clean_mask(speck)[15, 15] and J.clean_mask(speck)[5, 5]


def test_role_and_pose_confidence():
    agree = np.array([[0.95, 0.2], [np.nan, 0.9]])
    dist = np.array([[3, 3], [0, 3]])
    qa = {"skel_in_partner": np.array([[0.0, 0.8], [0.0, 0.0]]), "visible": np.ones((2, 2))}
    bd = np.array([[9999, 9999], [9999, 5]])
    rc = J.role_confidence(agree, dist, qa, bd)
    assert abs(rc[0, 0] - 0.95) < 1e-9 and abs(rc[0, 1] - 0.1) < 1e-9          # a skeleton on the partner halves it
    assert rc[1, 0] == 1.0 and rc[1, 1] >= 0.95                                # an anchor frame / a verified bib within 1 s
    kp = np.zeros((2, 2, 17, 3), np.float32)
    kp[..., 2] = 0.8
    pc = J.pose_confidence(kp, rc, qa)
    assert pc[0, 1] < 0.1 and pc[1, 0] > 0.7
    qa["visible"] = np.full((2, 2), 0.2)
    assert J.pose_confidence(kp, rc, qa)[1, 0] < pc[1, 0]


def test_bib_distance_and_coverage_table():
    d = J.bib_distance({0: [10, 50], 1: []}, 100)
    assert d[10, 0] == 0 and d[30, 0] == 20 and d[0, 1] == 9999
    conf = np.zeros((60, 2))
    conf[:30] = 0.9
    qa = {"skel_in_partner": np.zeros((60, 2)), "edge": np.zeros((60, 2), bool)}
    rows = J.coverage_table(conf, np.ones((60, 2)), qa, 30.0)
    assert len(rows) == 2 and rows[0]["both"] == 1.0 and rows[1]["both"] == 0.0


def test_per_frame_qa_flags_a_skeleton_on_the_partner_and_the_edge():
    F, W, H = 3, 100, 200
    fused = {"area": np.full((F, 2), 1000), "box": np.full((F, 2, 4), np.nan, np.float32)}
    for f in range(F):
        a = np.zeros((H, W), bool)
        b = np.zeros((H, W), bool)
        a[20:100, 10:40] = True
        b[20:100, 50:80] = True
        for o, m in enumerate((a, b)):
            fused[f"m_{f}_{o}"] = np.frombuffer(cv2.imencode(".png", m.astype(np.uint8) * 255)[1].tobytes(), np.uint8)
            ys, xs = np.nonzero(m)
            fused["box"][f, o] = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
    fused["box"][2, 1] = [50, 20, 100, 100]                         # frame 2: Lula's box touches the right edge
    kp = np.full((F, 2, 17, 3), np.nan, np.float32)
    for f in range(F):
        for j in (5, 6, 11, 12):
            kp[f, 0, j] = (20, 40, 0.9)                             # John's torso joints in John's mask
            kp[f, 1, j] = (60, 40, 0.9) if f < 2 else (20, 40, 0.9)  # Lula's: in her own mask, then (frame 2) on John's
    qa = J.per_frame_qa(fused, kp, (W, H), 30.0)
    assert qa["skel_in_partner"][0, 1] == 0.0 and qa["skel_in_partner"][2, 1] == 1.0 and qa["skel_in_partner"][2, 0] == 0.0
    assert qa["edge"][2, 1] and not qa["edge"][0].any()
    assert qa["box_iou"][0] == 0.0 and np.allclose(qa["visible"], 1.0, atol=0.01)
