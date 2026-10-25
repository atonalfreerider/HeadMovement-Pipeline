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
    new = check(7, _probs([det["lead"], None]))                               # the follow vanished: both are re-prompted (the lead's mask
    assert new is not None and sorted(new) == [1, 2] and np.allclose(new[2], det["follow"])   # may hold the couple)
    elsewhere = [0, 160, 20, 199]                                              # the follow's mask sits on somebody else
    assert sorted(check(7, _probs([det["lead"], elsewhere]))) == [1, 2]
    assert sorted(check(7, _probs([None, None]))) == [1, 2]
    assert check.stats["lost_checked"] == 5 and check.stats["lost"] == 3 and check.stats["checked"] == 0 and check.lost_frames == [7, 7, 7]


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


def test_a_one_role_reprompt_survives_the_sam_cache(tmp_path):
    res = {"probs": np.zeros((3, 2, 4, 4), np.uint8), "anchors": {0: {1: np.arange(4.0), 2: np.arange(4.0) + 10}, 2: {2: np.arange(4.0) + 20}},
           "seconds": 1.0, "frame_evals": 3, "work_wh": (4, 4), "check_stats": {"lost": 1}, "lost_frames": [2]}
    key = D.sam_cache_key({"frame": 0, "lead_box": [0, 1, 2, 3], "follow_box": [10, 11, 12, 13]}, True)
    D.save_sam_cache(tmp_path / "c.npz", key, res)
    got = D.load_sam_cache(tmp_path / "c.npz", key)
    assert sorted(got["anchors"]) == [0, 2] and sorted(got["anchors"][2]) == [2] and np.allclose(got["anchors"][2][2], np.arange(4.0) + 20)
    assert got["lost_frames"] == [2] and sorted(got["anchors"][0]) == [1, 2]


def test_propagate_reprompts_only_the_lost_role_and_never_at_reverse_frame_0():
    """A fake predictor in sam2's processing order (review 2026-10-25): the follow is lost on every frame that is not conditioned; a reverse
    restart from frame 0 yields nothing, so frame 0 is never a re-prompt."""
    torch = __import__("pytest").importorskip("torch")
    from dancecap import dancers_sam as S
    W, H, F = 100, 200, 100
    lead, fol = np.array([10, 10, 60, 150.0]), np.array([40, 10, 90, 150.0])

    class Fake:
        device, image_size = torch.device("cpu"), 64

        def __init__(self):
            self.cond, self.prompts = set(), []

        def add_new_points_or_box(self, state, frame_idx, obj_id, box):
            self.cond.add(frame_idx)
            self.prompts.append((frame_idx, obj_id))

        def propagate_in_video(self, state, start_frame_idx, reverse):
            order = (range(start_frame_idx, -1, -1) if start_frame_idx > 0 else []) if reverse else range(start_frame_idx, F)
            for fi in order:
                lg = torch.full((2, 1, H, W), -10.0)
                x0, y0, x1, y1 = lead.astype(int)
                lg[0, 0, y0:y1, x0:x1] = 10
                if fi in self.cond:
                    x0, y0, x1, y1 = fol.astype(int)
                    lg[1, 0, y0:y1, x0:x1] = 10
                yield fi, [1, 2], lg
    cfg = D.DancersConfig(reanchor_lost=True)
    check = D.make_check((W, H), (W, H), {}, cfg, {i: {"lead": lead, "follow": fol} for i in range(F)})
    p = Fake()
    res = S.propagate(p, [f"f{i}" for i in range(F)], (W, H), 48, {1: lead, 2: fol}, check=check, min_gap=cfg.reanchor_gap, log=lambda m: None)
    assert sorted(f for f, o in p.prompts if o == 1) == sorted(res["anchors"])  # both roles at every re-prompt
    assert 0 not in res["anchors"] and len(check.lost_frames) == len(res["anchors"]) - 1
    assert all((res["probs"][a, 1] > 127).any() for a in res["anchors"])      # every re-prompt was written
