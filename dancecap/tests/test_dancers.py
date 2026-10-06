"""Unit tests for the dancers stage (pure helpers, no GPU) + contract checks on real outputs when present.

    .venv-nopo4d\\Scripts\\python.exe -m pytest dancecap/tests/test_dancers.py -q
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dancecap.dancers import (Cam, DancersConfig, apply_override, head_top, keep_swap_viterbi,  # noqa: E402
                              mask_box, segments, sheet_frames, split_hair, tightness)
from dancecap.dancers_appearance import colour_names, descriptors, lead_llr, red_hair_mask  # noqa: E402
from dancecap.dancers_sam import _fp16_storage, work_size  # noqa: E402
from dancecap.dancers_select import MISS, SelectConfig, candidate_tracks, reflection_mask  # noqa: E402
from dancecap.take import PACKAGE_ROOT  # noqa: E402

WORK = PACKAGE_ROOT / "work" / "larissa_kadu"
CAMS = ("01", "02", "03", "04", "05", "06", "07", "09")


# ---------------------------------------------------------------------------
# candidate selection
# ---------------------------------------------------------------------------

def test_reflection_mask():
    cfg = SelectConfig()
    boxes = np.array([[100, 100, 200, 400],   # dancer
                      [120, 120, 150, 180],   # tiny copy fully inside the dancer (screen/mirror) -> reject
                      [110, 110, 190, 390],   # partner nested in an embrace, similar size -> keep
                      [300, 380, 310, 395]])  # far too small -> reject
    bad = reflection_mask(boxes, 800, cfg)
    assert bad.tolist() == [False, True, False, True]


def _scene(f=60, w=480, h=848):
    """Two moving central dancers, two static seated spectators, one big truncated near-camera spectator."""
    n = 5
    boxes = np.zeros((f, n, 4), np.float32)
    kpts = np.zeros((f, n, 17, 3), np.float32)
    for i in range(f):
        x = 200 + 30 * np.sin(i / 6)
        boxes[i, 0] = [x - 40, 300, x + 20, 560]  # dancer 1 (slot order is shuffled below)
        boxes[i, 1] = [x + 25, 290, x + 90, 565]  # dancer 2
        boxes[i, 2] = [10, 420, 70, 520]  # seated spectator, left edge
        boxes[i, 3] = [400, 430, 470, 525]  # seated spectator, right edge
        boxes[i, 4] = [150, 600, 400, h]  # near-camera spectator cut by the bottom border
        kpts[i, :2, 15:17, 2] = 0.9  # dancers' ankles visible
    perm = np.random.default_rng(0).permuted(np.tile(np.arange(n), (f, 1)), axis=1)
    boxes = np.take_along_axis(boxes, perm[..., None], 1)
    kpts = np.take_along_axis(kpts, perm[..., None, None], 1)
    scores = np.full((f, n), 0.9, np.float32)
    valid = np.ones((f, n), bool)
    return boxes, scores, valid, kpts, perm


def test_candidate_tracks_pick_the_couple_continuously():
    boxes, scores, valid, kpts, perm = _scene()
    r = candidate_tracks(boxes, scores, valid, kpts, 480, 848, 30.0)
    picked = np.take_along_axis(perm, np.where(r["idx"] == MISS, 0, r["idx"]), 1)
    assert (r["idx"] != MISS).all()
    assert set(map(tuple, np.sort(picked, 1))) == {(0, 1)}  # always the two dancers
    assert len(set(picked[:, 0])) == 1  # slot A follows the same person throughout


def test_candidate_tracks_embrace_merged_box():
    boxes, scores, valid, kpts, perm = _scene()
    k = np.argwhere(perm[30] == 1)[0, 0]  # dancer 2 vanishes for one frame (merged detection)
    valid[30, k] = False
    r = candidate_tracks(boxes, scores, valid, kpts, 480, 848, 30.0)
    picked = [perm[i, j] for i in (29, 31) for j in r["idx"][i]]
    assert sorted(picked) == [0, 0, 1, 1]  # the tracks recover after the gap
    assert (r["idx"][30] == MISS).sum() <= 1


# ---------------------------------------------------------------------------
# appearance / role rule
# ---------------------------------------------------------------------------

def _person(img, x0, top_bgr, leg_bgr, red_hair):
    """Paint a crude standing person into img; return its box and COCO keypoints."""
    cv2.rectangle(img, (x0 + 10, 60), (x0 + 50, 150), top_bgr, -1)  # torso
    cv2.rectangle(img, (x0 + 12, 150), (x0 + 48, 280), leg_bgr, -1)  # legs
    cv2.circle(img, (x0 + 30, 40), 14, (140, 170, 210), -1)  # face/skin
    if red_hair:
        cv2.rectangle(img, (x0 + 8, 18), (x0 + 52, 90), (20, 20, 200), -1)  # long red hair
        cv2.circle(img, (x0 + 30, 42), 9, (140, 170, 210), -1)
    k = np.zeros((17, 3), np.float32)
    pts = {0: (30, 40), 1: (26, 36), 2: (34, 36), 3: (20, 40), 4: (40, 40), 5: (14, 62), 6: (46, 62),
           11: (18, 150), 12: (42, 150), 13: (20, 215), 14: (40, 215), 15: (20, 278), 16: (40, 278)}
    for j, (x, y) in pts.items():
        k[j] = (x0 + x, y, 0.9)
    return np.array([x0, 10, x0 + 60, 285], float), k


def test_colour_rule_identifies_lead():
    img = np.full((300, 200, 3), 90, np.uint8)
    bl, kl = _person(img, 10, (25, 25, 25), (90, 60, 40), False)  # black top, dark jeans
    bf, kf = _person(img, 110, (235, 235, 235), (200, 185, 170), True)  # white top, light jeans, red hair
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    dl, df = descriptors(hsv, bl, kl), descriptors(hsv, bf, kf)
    assert dl["torso_v"] < 60 and df["torso_v"] > 150  # hair hanging over her top pulls the median down
    assert df["red"] > 0.2 and dl["red"] < 0.02
    llr, cues = lead_llr(dl, df)
    assert llr > 4 and all(v > 0 for v in cues.values())
    assert lead_llr(df, dl)[0] == -llr
    assert colour_names(dl)["top"] == "black" and colour_names(df)["red_above_shoulders"] == "strong red"


def test_red_hair_mask_excludes_skin():
    hair = cv2.cvtColor(np.uint8([[[25, 30, 190]]]), cv2.COLOR_BGR2HSV)  # saturated red
    skin = cv2.cvtColor(np.uint8([[[120, 150, 205]]]), cv2.COLOR_BGR2HSV)  # warm skin
    assert red_hair_mask(hair)[0, 0] and not red_hair_mask(skin)[0, 0]


def test_override_swaps_labels():
    lab = {"05": {"lead_box": [1, 2, 3, 4], "follow_box": [5, 6, 7, 8], "by": "auto",
                  "descriptors": {"lead": {"a": 1}, "follow": {"a": 2}},
                  "colour_names": {"lead": {"top": "black"}, "follow": {"top": "white"}}}}
    out = apply_override(lab, {"05": {"swap": True}, "99": {"swap": True}})
    assert out["05"]["lead_box"] == [5, 6, 7, 8] and out["05"]["by"] == "override"
    assert out["05"]["colour_names"]["lead"]["top"] == "white"
    assert lab["05"]["lead_box"] == [1, 2, 3, 4]  # input untouched
    assert apply_override(lab, {"05": {"swap": False}})["05"]["by"] == "auto"


# ---------------------------------------------------------------------------
# identity smoothing
# ---------------------------------------------------------------------------

def test_keep_swap_viterbi_needs_consistent_evidence():
    llr = np.full(60, 2.0)
    llr[10:13] = -2.0  # 3 contrary frames: noise, no switch
    llr[30:45] = -2.0  # 15 contrary frames: a real swap
    state, post = keep_swap_viterbi(llr, clip=2.0, switch_frames=5.0)
    assert state[:30].sum() == 0
    assert state[31:44].all() and not state[46:].any()
    assert post[11] < 0.5 < post[37]
    assert segments(state.astype(bool)) == [(int(np.argmax(state)), 44)]
    s0, _ = keep_swap_viterbi(np.zeros(5))
    assert not s0.any()


def test_segments():
    assert segments(np.array([0, 1, 1, 0, 1], bool)) == [(1, 2), (4, 4)]
    assert segments(np.zeros(3, bool)) == []


# ---------------------------------------------------------------------------
# hair / body, QA helpers
# ---------------------------------------------------------------------------

def test_split_hair():
    img = np.full((300, 200, 3), 90, np.uint8)
    box, k = _person(img, 70, (235, 235, 235), (200, 185, 170), True)
    follow = np.zeros((300, 200), bool)
    follow[5:285, 70:131] = True
    hair, body = split_hair(img, follow, k)
    assert hair[30, 80] and not body[30, 80]  # red hair pixel
    assert body[120, 100] and not hair[120, 100]  # torso
    assert not (hair & body).any() and ((hair | body) == follow).all()
    top = head_top(k)
    assert top is not None and top < 36
    assert not body[: int(top)].any()


def test_mask_box_and_tightness():
    m = np.zeros((100, 80), bool)
    m[10:50, 20:40] = True
    assert mask_box(m).tolist() == [20, 10, 40, 50]
    assert mask_box(np.zeros((5, 5), bool)) is None
    cam = types.SimpleNamespace(people={"valid": np.ones((1, 1), bool), "boxes": np.array([[[20, 10, 40, 50]]], np.float32)})
    other = np.zeros_like(m)
    other[10:50, 45:60] = True
    pair = np.array([[20, 10, 40, 50], [45, 10, 60, 50]], np.float32)
    assert tightness(m, other, pair, cam, 0, DancersConfig())["ok"]
    spill = other.copy()
    spill[60:100, 0:80] = True  # a spectator merged into the follow's mask
    assert not tightness(m, spill, pair, cam, 0, DancersConfig())["ok"]


def test_sheet_frames_one_per_second():
    t = 56.016 + np.arange(180) / 30
    idx = sorted(sheet_frames(t))
    assert len(idx) == 7 and idx[0] == 0 and idx[-1] == 179
    assert len(sheet_frames(t, 0.25)) == 25


def test_gpu_policy_helpers():
    assert work_size(1080, 1920) == (720, 1280)
    assert work_size(480, 848) == (480, 848)
    import torch
    mod = types.SimpleNamespace(torch=torch)
    _fp16_storage(mod)
    assert mod.torch.bfloat16 is torch.float16 and mod.torch.cat is torch.cat


# ---------------------------------------------------------------------------
# contract checks on real outputs (skipped until the stage has run)
# ---------------------------------------------------------------------------

def test_real_labels_contract():
    p = WORK / "dancers" / "labels.json"
    if not p.exists():
        return
    labels = json.loads(p.read_text())
    for cam, e in labels.items():
        assert {"frame", "lead_box", "follow_box", "by", "confidence"} <= set(e)
        assert e["by"] in ("auto", "override") and 0 <= e["confidence"] <= 1
        assert len(e["lead_box"]) == 4 and len(e["follow_box"]) == 4


def test_real_masks_contract():
    roles_p = WORK / "dancers" / "roles.json"
    if not roles_p.exists():
        return
    roles = json.loads(roles_p.read_text())
    assert {"cams", "labels_source", "swaps_repaired"} <= set(roles)
    for cam, d in roles["cams"].items():
        meta = json.loads((WORK / "frames" / cam / "frames.json").read_text())
        n = meta["frame_count"] if "frame_count" in meta else len(meta["frames"])
        assert len(d["frames"]) == n
        fr = d["frames"][n // 2]
        assert {"i", "t", "lead", "follow", "swap_prob", "embrace"} <= set(fr)
        i = fr["i"]
        for sub, pids in (("masks", (1, 2)), ("body", (1, 2)), ("hair", (2,))):
            for pid in pids:
                m = cv2.imread(str(WORK / "dancers" / cam / sub / f"mask_{i:04d}_{pid:02d}.png"), cv2.IMREAD_GRAYSCALE)
                assert m is not None and m.shape == (meta["height"], meta["width"])
                assert set(np.unique(m)) <= {0, 255}
        full = cv2.imread(str(WORK / "dancers" / cam / "masks" / f"mask_{i:04d}_02.png"), 0) > 0
        body = cv2.imread(str(WORK / "dancers" / cam / "body" / f"mask_{i:04d}_02.png"), 0) > 0
        hair = cv2.imread(str(WORK / "dancers" / cam / "hair" / f"mask_{i:04d}_02.png"), 0) > 0
        assert not (body & ~full).any() and not (hair & ~full).any() and not (body & hair).any()
        lead = cv2.imread(str(WORK / "dancers" / cam / "masks" / f"mask_{i:04d}_01.png"), 0) > 0
        assert not (lead & full).any()  # one owner per pixel


def test_cam_loader():
    if not (WORK / "people" / "07.npz").exists():
        return
    cam = Cam.load(WORK, "07")
    assert len(cam.files) == len(cam.t) == cam.people["valid"].shape[0]


def test_clean_mask_drops_detached_specks_only():
    from dancecap.dancers import clean_mask
    m = np.zeros((400, 200), bool)
    m[50:350, 60:120] = True  # body
    m[60:70, 125:135] = True  # hand just beside the body (near): keep
    m[20:30, 180:190] = True  # speck far away (a lit sign): drop
    m[200:260, 0:40] = True  # big detached limb (occluded by the partner): keep
    out, n = clean_mask(m)
    assert n == 100 and not out[25, 185] and out[65, 130] and out[230, 20]
    same, n0 = clean_mask(out)
    assert n0 == 0 and (same == out).all()


def test_reanchor_gpu():
    """GPU (opt-in: DANCECAP_GPU_TESTS=1): a re-anchor prompt takes effect immediately, even when it
    deliberately swaps the objects (cam07, 40 frames, anchor 20, re-prompts at 30 (swapped) and 10)."""
    import os
    if os.environ.get("DANCECAP_GPU_TESTS") != "1" or not (WORK / "dancers" / "candidates" / "07.npz").exists():
        return
    from dancecap.dancers import box_iou
    from dancecap.dancers_sam import build_predictor, propagate
    cam = Cam.load(WORK, "07")
    cand = np.load(WORK / "dancers" / "candidates" / "07.npz")

    def roles(i):
        a, b = cand["boxes"][i]
        return (b, a) if cand["a_is_follow"][i] else (a, b)

    def check(frame, probs):
        if frame == 30:
            lead, fol = roles(30)
            return {1: fol, 2: lead}
        if frame == 10:
            lead, fol = roles(10)
            return {1: lead, 2: fol}
        return None

    lead, fol = roles(20)
    res = propagate(build_predictor(), cam.files[:40], cam.wh, 20, {1: lead, 2: fol}, check=check, min_gap=5)
    s = res["work_wh"][0] / cam.wh[0]
    assert sorted(res["anchors"]) == [10, 20, 30] and res["done"].all()
    for i in (15, 25, 29, 30, 31, 35):
        mb = mask_box(res["probs"][i, 0] > 127) / s
        lead, fol = roles(i)
        assert (box_iou(mb, lead) > box_iou(mb, fol)) == (i < 30), i
