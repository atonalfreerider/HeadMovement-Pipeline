"""dancers: SAM re-anchor prompts - clean frames re-prompt on drift, overlapping frames (reanchor_lost) only when a role is lost."""
from types import SimpleNamespace

import numpy as np

from dancecap import dancers as D


def _probs(boxes, wh=(100, 200)):
    """(2, h, w) uint8 SAM probabilities with one filled box per object (None = an empty mask)."""
    w, h = wh
    p = np.zeros((2, h, w), np.uint8)
    for k, b in enumerate(boxes):
        if b is not None:
            x0, y0, x1, y1 = (int(v) for v in b)
            p[k, y0:y1, x0:x1] = 255
    return p


def _tracks(pair):
    """Teacher-route candidate tracks and evidence for frames of the given (lead box, follow box) pairs."""
    boxes = np.array(pair, np.float32)
    f = len(boxes)
    pair_iou = np.array([D.box_iou(a, b) for a, b in boxes])
    llr = np.where(pair_iou <= D.DancersConfig().evidence_iou, 3.0, 0.0)   # recap_front.dancer_one's HM_TEACHER_EVIDENCE constant
    return {"idx": np.zeros((f, 2), int), "boxes": boxes}, {"llr": llr, "pair_iou": pair_iou, "a_is_follow": np.zeros(f, bool)}


def test_overlapping_frames_are_prompts_only_for_the_lost_path():
    cfg = D.DancersConfig(reanchor_lost=True)
    cam = SimpleNamespace(wh=(100, 200))
    apart = ([5, 10, 40, 150], [55, 10, 95, 150])            # IoU 0: a clean frame
    close = ([10, 10, 60, 150], [40, 10, 90, 150])           # IoU 0.25: above clean_iou, below evidence_iou
    edge = ([10, 10, 60, 199], [40, 10, 90, 199])            # the same, both boxes at the image bottom
    tracks, ev = _tracks([apart, close, edge])
    assert 0.1 < ev["pair_iou"][1] < 0.3
    strict = D.frame_role_boxes(cam, tracks, ev, cfg, False)
    assert sorted(strict) == [0]
    lost = D.frame_role_boxes(cam, tracks, ev, cfg, False, max_iou=cfg.evidence_iou, edge_ok=True)
    assert sorted(lost) == [0, 1, 2]
    assert sorted(D.frame_role_boxes(cam, tracks, ev, cfg, False, max_iou=cfg.evidence_iou)) == [0, 1]


def test_check_reprompts_a_lost_role_but_not_a_loose_embrace_mask():
    cfg = D.DancersConfig(reanchor_lost=True)
    wh = (100, 200)
    det = {"lead": np.array([10, 10, 60, 150.0]), "follow": np.array([40, 10, 90, 150.0])}
    check = D.make_check(wh, wh, {}, cfg, {7: det})
    assert check(3, _probs([det["lead"], None])) is None                      # not a candidate frame
    assert check(7, _probs([det["lead"], det["follow"]])) is None             # both tracked
    loose = [10, 10, 90, 150]                                                  # the follow's mask spills over the pair: IoU 0.63, not lost
    assert check(7, _probs([det["lead"], loose])) is None
    new = check(7, _probs([det["lead"], None]))                               # the follow vanished
    assert new is not None and sorted(new) == [1, 2] and np.allclose(new[2], det["follow"])
    elsewhere = [0, 160, 20, 199]                                              # the follow's mask sits on somebody else
    assert check(7, _probs([det["lead"], elsewhere])) is not None
    assert check.stats["lost_checked"] == 4 and check.stats["lost"] == 2 and check.stats["checked"] == 0


def test_clean_frames_keep_the_drift_rule():
    cfg = D.DancersConfig(reanchor_lost=True)
    wh = (100, 200)
    det = {"lead": np.array([5, 10, 40, 150.0]), "follow": np.array([55, 10, 95, 150.0])}
    check = D.make_check(wh, wh, {4: det}, cfg, {4: det})
    assert check(4, _probs([det["lead"], [55, 10, 95, 60]])) is not None      # follow drifted (IoU 0.36: > lost_iou, < drift_iou)
    assert check.stats["drifted"] == 1 and check.stats["lost_checked"] == 0
    assert check(4, _probs([det["lead"], det["follow"]])) is None


def test_the_sam_cache_key_marks_the_lost_route():
    label = {"frame": 12, "lead_box": [1, 2, 3, 4], "follow_box": [5, 6, 7, 8]}
    assert D.sam_cache_key(label).shape == (9,)                              # caches made before the flag stay valid
    assert D.sam_cache_key(label, True).shape == (10,)
