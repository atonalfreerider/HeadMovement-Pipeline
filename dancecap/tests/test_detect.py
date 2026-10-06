"""Unit tests for dancecap.detect (pure helpers, no GPU) + contract checks on real outputs when present.

    .venv-nopo4d\\Scripts\\python.exe -m pytest dancecap/tests/test_detect.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dancecap.detect import (FrameResult, attach_keypoints, dancer_pair_present, dancer_region_present,  # noqa: E402
                             dilate, fill_boxes, greedy_match, iou_matrix, merge_scales, pack, uncovered_boxes)
from dancecap.take import PACKAGE_ROOT  # noqa: E402

WORK = PACKAGE_ROOT / "work" / "larissa_kadu"


def test_iou_matrix():
    a = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], float)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [100, 100, 101, 101]], float)
    m = iou_matrix(a, b)
    assert m.shape == (2, 3)
    assert np.isclose(m[0, 0], 1.0)
    assert np.isclose(m[0, 1], 50 / 150)
    assert np.allclose(m[1], 0)
    assert iou_matrix(np.zeros((0, 4)), b).shape == (0, 3)


def test_greedy_match_one_to_one():
    iou = np.array([[0.9, 0.8], [0.85, 0.1]])
    assert sorted(greedy_match(iou, 0.5)) == [(0, 0)]  # row 1 wants col 0 too, col 1 is below 0.5 for it
    assert sorted(greedy_match(iou, 0.05)) == [(0, 0), (1, 1)]
    assert greedy_match(np.zeros((0, 3)), 0.5) == []


def test_attach_keypoints():
    det = np.array([[0, 0, 10, 20], [50, 0, 60, 20]], float)
    pose_boxes = np.array([[51, 1, 60, 20]], float)
    pose_kpts = np.ones((1, 17, 3), np.float32) * 7
    k = attach_keypoints(det, pose_boxes, pose_kpts)
    assert k.shape == (2, 17, 3)
    assert (k[0] == 0).all() and (k[1] == 7).all()


def test_masks_fill_uncovered_and_dilate():
    mask = np.zeros((100, 100), np.uint8)
    mask[10:50, 10:30] = 255  # a segmented person
    boxes = np.array([[10, 10, 30, 50], [60, 20, 80, 90]], float)  # second person was missed by the segmenter
    miss = uncovered_boxes(boxes, mask, max_area=1.0)
    assert miss.shape == (1, 4) and np.allclose(miss[0], boxes[1])
    huge = np.array([[0, 40, 100, 100]], float)  # e.g. the camera operator's hand across the floor
    assert len(uncovered_boxes(huge, mask)) == 0 and len(uncovered_boxes(huge, mask, max_area=1.0)) == 1
    miss = uncovered_boxes(boxes, mask, max_area=1.0)
    fill_boxes(mask, miss)
    assert mask[55, 70] == 255 and mask[20, 61] == 0  # ellipse centre in, box corner out
    d = dilate(mask, 10)
    assert d[55, 85] == 255 and d[55, 95] == 0  # +10 px, not more
    assert d.dtype == np.uint8 and set(np.unique(d)) <= {0, 255}


def test_pack_pads_and_sorts():
    r1 = FrameResult(np.array([[0, 0, 1, 1], [0, 0, 2, 2]], np.float32), np.array([0.3, 0.9], np.float32),
                     np.zeros((2, 17, 3), np.float32), np.zeros((0, 0), np.uint8))
    r2 = FrameResult(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros((0, 17, 3), np.float32),
                     np.zeros((0, 0), np.uint8))
    p = pack([r1, r2])
    assert p["boxes"].shape == (2, 2, 4) and p["kpts"].shape == (2, 2, 17, 3)
    assert p["valid"].tolist() == [[True, True], [False, False]]
    assert np.allclose(p["scores"][0], [0.9, 0.3])
    assert np.allclose(p["boxes"][0, 0], [0, 0, 2, 2])


def test_dancer_pair_heuristic():
    w, h = 480, 848
    dancers = [[200, 250, 260, 460], [230, 255, 290, 455]]
    crowd = [[20, 300, 60, 360], [400, 300, 440, 360]]
    near_spectator = [[150, 500, 470, 848]]  # truncated by the bottom border
    boxes = np.array(dancers + crowd + near_spectator, float)
    scores = np.full(len(boxes), 0.8)
    valid = np.ones(len(boxes), bool)
    assert dancer_pair_present(boxes, scores, valid, w, h)
    only_one = np.array([dancers[0]] + crowd + near_spectator, float)
    assert not dancer_pair_present(only_one, scores[:4], valid[:4], w, h)
    v = valid.copy()
    v[1] = False
    assert not dancer_pair_present(boxes, scores, v, w, h)


def test_dancer_pair_rejects_duplicate_and_region():
    w, h = 480, 848
    dup = np.array([[200, 250, 260, 460], [201, 251, 261, 459]], float)
    assert not dancer_pair_present(dup, np.array([0.9, 0.8]), np.ones(2, bool), w, h)
    assert dancer_region_present(dup[:1], np.array([0.9]), np.ones(1, bool), w, h)
    assert not dancer_region_present(dup[:1], np.array([0.3]), np.ones(1, bool), w, h)


def test_merge_scales():
    H = 848
    def fr(boxes, scores, kval, mask_px):
        m = np.zeros((H, 480), np.uint8)
        m[mask_px] = 255
        b = np.asarray(boxes, np.float32).reshape(-1, 4)
        return FrameResult(b, np.asarray(scores, np.float32), np.full((len(b), 17, 3), kval, np.float32), m)
    # large person: low-res pass sees the whole body, high-res pass cuts the feet; small person only high-res
    lo = fr([[30, 80, 390, 840]], [0.9], 1.0, (500, 100))
    hi = fr([[35, 85, 392, 700], [400, 300, 440, 360]], [0.88, 0.5], 2.0, (300, 410))
    r = merge_scales(lo, hi, H)
    assert len(r.boxes) == 2
    assert np.allclose(r.boxes[0], [30, 80, 390, 840]) and (r.kpts[0] == 1).all() and np.isclose(r.scores[0], 0.9)
    assert np.allclose(r.boxes[1], [400, 300, 440, 360]) and (r.kpts[1] == 2).all()
    assert r.mask[500, 100] == 255 and r.mask[300, 410] == 255
    # small matched person keeps the high-res box
    lo2 = fr([[400, 302, 441, 362]], [0.4], 1.0, (0, 0))
    r2 = merge_scales(lo2, hi, H)
    assert len(r2.boxes) == 2 and any(np.allclose(b, [400, 300, 440, 360]) for b in r2.boxes)
    empty = fr([], [], 0.0, (0, 0))
    assert len(merge_scales(empty, empty, H).boxes) == 0


def test_real_outputs_contract():
    """When the detect stage has run on larissa_kadu, check every camera against CONTRACTS.md."""
    people = WORK / "people"
    if not people.exists():
        return
    for npz in sorted(people.glob("*.npz")):
        cam = npz.stem
        meta = json.loads((WORK / "frames" / cam / "frames.json").read_text())
        d = np.load(npz)
        f = meta["frame_count"]
        assert d["boxes"].shape[0] == f and d["boxes"].shape[2] == 4
        n = d["boxes"].shape[1]
        assert d["scores"].shape == (f, n) and d["valid"].shape == (f, n) and d["valid"].dtype == bool
        assert d["kpts"].shape == (f, n, 17, 3)
        assert np.allclose(d["t"], [r["t"] for r in meta["frames"]])
        b = d["boxes"][d["valid"]]
        assert (b[:, 2] > b[:, 0]).all() and (b[:, 3] > b[:, 1]).all()
        masks = sorted((WORK / "people_masks" / cam).glob("*.png"))
        assert [m.stem for m in masks] == [Path(r["file"]).stem for r in meta["frames"]]
        import cv2
        m = cv2.imread(str(masks[0]), cv2.IMREAD_UNCHANGED)
        assert m.shape == (meta["height"], meta["width"]) and m.dtype == np.uint8
        assert set(np.unique(m)) <= {0, 255}
        kf = json.loads((WORK / "keyframes" / cam / "keyframes.json").read_text())
        kf_masks = {p.stem for p in (WORK / "people_masks_kf" / cam).glob("*.png")}
        assert kf_masks == {Path(r["file"]).stem for r in kf}
        assert (people / "qa" / f"{cam}.jpg").exists()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
