"""Unit tests for the dancers stage (pure helpers, no GPU) + contract checks on real outputs when present.

    .venv-nopo4d\\Scripts\\python.exe -m pytest dancecap/tests/test_dancers.py -q
"""
from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dancecap.dancers import (Cam, DancersConfig, apply_override, clean_mask, identity_evidence,  # noqa: E402
                              identity_flags, keep_swap_viterbi, mask_box, match_kpts, segments, sheet_frames,
                              split_hair, tightness)
from dancecap.dancers_hair import HairColour, head_geom  # noqa: E402
from dancecap.dancers_appearance import colour_names, descriptors, lead_llr, red_hair_mask  # noqa: E402
from dancecap.dancers_sam import _fp16_storage, build_with_overrides, work_size  # noqa: E402
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


def test_keep_swap_viterbi_pinned_anchor():
    llr = np.full(60, 2.0)
    llr[:40] = -2.0  # SAM swapped for most of the window, anchor 50 is verified
    state, post = keep_swap_viterbi(llr, pinned=[50])
    assert state[:38].all() and not state[42:].any() and post[50] < 1e-6
    llr2 = np.full(60, -2.0)  # every frame says 'swapped' - the anchor still wins at its own frame
    s2, _ = keep_swap_viterbi(llr2, pinned=[50])
    assert s2[50] == 0 and s2[:44].all()
    assert identity_flags(llr2, [50], 2.0, 5.0)  # and the camera is flagged
    assert not identity_flags(np.full(60, 2.0), [50], 2.0, 5.0)


def _two_dancer_cam(f=40, swapped_until=30):
    """Synthetic camera: lead (black top) left, follow (white top, red hair) right, static. SAM's objects
    are swapped on frames [0, swapped_until) - a majority of the window."""
    img = np.full((300, 240, 3), 90, np.uint8)
    bl, kl = _person(img, 20, (25, 25, 25), (90, 60, 40), False)
    bf, kf = _person(img, 140, (235, 235, 235), (200, 185, 170), True)
    ml, mf = np.zeros((300, 240), bool), np.zeros((300, 240), bool)
    ml[10:285, 20:80] = True
    mf[10:285, 140:200] = True
    masks = np.zeros((f, 2, 300, 240), bool)
    for i in range(f):
        masks[i] = (mf, ml) if i < swapped_until else (ml, mf)
    cam = types.SimpleNamespace(fps=10.0, people={"valid": np.ones((f, 2), bool),
                                                  "boxes": np.tile(np.array([bl, bf], np.float32), (f, 1, 1)),
                                                  "kpts": np.tile(np.stack([kl, kf]), (f, 1, 1, 1))})
    return cam, masks, [img] * f


def test_identity_gallery_anchored_not_majority():
    """Reviewer repro: SAM swapped on 0-29 of 40 frames, verified anchor at 35. The galleries must come
    from the anchor neighbourhood, and the repair must swap 0-29 back - not the correct segment."""
    cam, masks, imgs = _two_dancer_cam()
    cfg = DancersConfig()
    ide = identity_evidence(cam, masks, imgs, cfg, anchors=[35])
    assert ide["gallery"]["lead"]["torso_v"] < 60 < ide["gallery"]["follow"]["torso_v"]
    assert ide["gallery_majority"]["lead"]["torso_v"] > 150  # the old (majority) gallery would be inverted
    state, post = keep_swap_viterbi(ide["llr"], cfg.llr_clip, cfg.switch_frames, pinned=[35])
    assert state[:28].all() and not state[32:].any()
    assert identity_flags(ide["llr"], [35], cfg.llr_clip, cfg.switch_frames)
    final = np.where(state == 1, 1 - post, post)  # roles.json swap_prob: about the written masks
    assert final[:28].max() < 0.5
    # no usable frame near the anchor: the label's own descriptors orient the galleries
    lab = {"lead": {"torso_v": 25.0, "leg_v": 70.0, "red": 0.0},
           "follow": {"torso_v": 230.0, "leg_v": 190.0, "red": 0.5}}
    cfg2 = DancersConfig(gallery_windows=(0.0,), gallery_min=50)
    ide2 = identity_evidence(cam, masks, imgs, cfg2, anchors=[35], label_desc=lab)
    assert ide2["gallery_source"]["lead"]["torso_v"] == "label" and (ide2["llr"][:30] < 0).all()


def test_segments():
    assert segments(np.array([0, 1, 1, 0, 1], bool)) == [(1, 2), (4, 4)]
    assert segments(np.zeros(3, bool)) == []


# ---------------------------------------------------------------------------
# hair / body, QA helpers
# ---------------------------------------------------------------------------

def _follow_scene(raised_arm=False, red_hand=False, warm_face=False):
    """Follow facing the camera: red hair around/behind the head, face, white top; optionally a skin-coloured
    arm raised above her head, a red-lit hand at hip height far from the head, a red-lit (warm) face."""
    img = np.full((300, 200, 3), 90, np.uint8)
    box, k = _person(img, 70, (235, 235, 235), (200, 185, 170), True)
    follow = np.zeros((300, 200), bool)
    follow[5:285, 70:131] = True
    skin = (140, 170, 210)
    if warm_face:
        cv2.circle(img, (100, 42), 9, (30, 40, 200), -1)  # face lit so red it passes the fixed hair rule
    k[8], k[10] = (112, 30, 0.9), (118, 2, 0.9)  # right elbow at head height, right wrist above the head
    if raised_arm:
        cv2.line(img, (116, 62), (112, 30), skin, 8)
        cv2.line(img, (112, 30), (118, 2), skin, 7)
        follow[0:62, 108:124] |= cv2.line(np.zeros((300, 200), np.uint8), (116, 62), (118, 0), 1, 12)[0:62, 108:124] > 0
    else:
        k[8, 2] = k[10, 2] = 0.0
    if red_hand:  # right hand at hip height, warm-lit (red under the fixed rule), far from the head
        cv2.circle(img, (62, 175), 8, (25, 30, 190), -1)
        follow[165:186, 52:72] = True
        k[7], k[9] = (60, 120, 0.9), (62, 168, 0.9)
    return img, follow, k


def test_split_hair_basic():
    img, follow, k = _follow_scene()
    hair, body = split_hair(img, follow, k)
    assert hair[25, 80] and not body[25, 80]  # red hair beside the head
    assert body[120, 100] and not hair[120, 100]  # torso
    assert body[42, 100] and not hair[42, 100]  # face
    assert not (hair & body).any() and ((hair | body) == follow).all()
    g = head_geom(k)
    assert g is not None and 10 < g.size < 60 and abs(g.centre[0] - 100) < 5 and g.face is not None


def test_split_hair_keeps_raised_arm_and_far_red_hand_in_body():
    img, follow, k = _follow_scene(raised_arm=True, red_hand=True)
    hair, body = split_hair(img, follow, k)
    assert follow[3, 118] and body[3, 118] and not hair[3, 118]  # hand above her head
    assert body[20, 114] and not hair[20, 114]  # forearm beside the head
    assert body[175, 62] and not hair[175, 62]  # red-lit hand far from the head (no spatial link to the hair)
    assert hair[80, 118] and hair[40, 80]  # hair hanging down beside the raised arm is still hair
    k2 = k.copy()
    k2[[7, 9], 2] = 0  # even without the hand's keypoints the far red blob is not connected to the head
    hair2, _ = split_hair(img, follow, k2)
    assert not hair2[175, 62]


def test_split_hair_warm_face_and_no_keypoints():
    img, follow, k = _follow_scene(warm_face=True)
    hair, body = split_hair(img, follow, k)
    assert body[42, 100] and not hair[42, 100]  # red-lit face core stays body
    hair0, body0 = split_hair(img, follow, None)
    assert not hair0.any() and (body0 == follow).all()  # no keypoints: nothing is called hair


def test_hair_colour_model_separates_warm_skin():
    """Per-camera colour model: skin that is red enough to pass the fixed hair rule is learned as skin."""
    cm = HairColour()
    frames = []
    for _ in range(4):
        img, follow, k = _follow_scene(raised_arm=True)
        cv2.circle(img, (100, 42), 9, (60, 80, 200), -1)  # warm face
        cv2.line(img, (116, 62), (112, 30), (60, 80, 200), 8)  # warm forearm
        cv2.line(img, (112, 30), (118, 2), (60, 80, 200), 7)
        frames.append((img, follow, k))
        cm.add_frame(img, follow, head_geom(k), k)
    cm.cfg.min_seeds = 50
    assert cm.fit()
    img, follow, k = frames[0]
    like = cm.hairlike(img)
    assert like[25, 80] and not like[42, 100]
    hair, body = split_hair(img, follow, k, cm)
    assert hair[25, 80] and body[42, 100] and body[3, 118]


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


def test_match_kpts_prefers_keypoints_on_the_mask():
    """Embrace: the partner's detection overlaps her mask box better, but its keypoints are on him."""
    h, w = 300, 200
    fol, lead = np.zeros((h, w), bool), np.zeros((h, w), bool)
    fol[50:280, 60:120] = True
    lead[40:290, 120:190] = True
    k_f = np.zeros((17, 3), np.float32)
    k_f[:, 0], k_f[:, 1], k_f[:, 2] = 90, np.linspace(60, 270, 17), 0.9
    k_l = k_f.copy()
    k_l[:, 0] = 150
    boxes = np.array([[60, 50, 120, 280], [62, 52, 118, 278]], np.float32)  # partner's box hugs her mask box
    cam = types.SimpleNamespace(people={"valid": np.ones((1, 2), bool), "boxes": boxes[None],
                                        "kpts": np.stack([k_l, k_f])[None]})
    box = mask_box(fol)
    assert match_kpts(cam, 0, box)[0, 0] == 150  # box-only matching picks the partner
    assert match_kpts(cam, 0, box, fol, lead)[0, 0] == 90  # mask-aware matching picks her


def test_build_with_overrides_keeps_the_other_override():
    calls = []

    def build(hydra_overrides_extra):
        calls.append(list(hydra_overrides_extra))
        if any("max_cond" in o for o in hydra_overrides_extra):
            raise ValueError("unknown key max_cond_frames_in_attn")
        return types.SimpleNamespace()

    pred = build_with_overrides(build, log=lambda m: None)
    assert pred.dancecap_overrides == ["clear_non_cond_mem_around_input"]
    assert calls[-1] == ["++model.clear_non_cond_mem_around_input=true"]

    def build_old(hydra_overrides_extra):
        if hydra_overrides_extra:
            raise ValueError("no such option")
        return types.SimpleNamespace()

    with pytest.raises(RuntimeError, match="clear_non_cond"):
        build_with_overrides(build_old, log=lambda m: None)


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
        pytest.skip("dancers stage has not run on larissa_kadu")
    labels = json.loads(p.read_text())
    for cam, e in labels.items():
        assert {"frame", "lead_box", "follow_box", "by", "confidence"} <= set(e)
        assert e["by"] in ("auto", "override") and 0 <= e["confidence"] <= 1
        assert len(e["lead_box"]) == 4 and len(e["follow_box"]) == 4


def test_real_masks_contract():
    """Every frame of every camera (DANCECAP_CONTRACT_STRIDE to thin out): PNG layout, native size, binary;
    lead/follow disjoint; follow body|hair == full, body and hair disjoint; lead body == lead full;
    swap_prob describes the written (repaired) masks."""
    roles_p = WORK / "dancers" / "roles.json"
    if not roles_p.exists():
        pytest.skip("dancers stage has not run on larissa_kadu")
    roles = json.loads(roles_p.read_text())
    assert {"cams", "labels_source", "swaps_repaired"} <= set(roles)
    stride = int(os.environ.get("DANCECAP_CONTRACT_STRIDE", "1"))
    for cam, d in roles["cams"].items():
        meta = json.loads((WORK / "frames" / cam / "frames.json").read_text())
        n = len(meta["frames"])
        assert len(d["frames"]) == n
        for fr in d["frames"][::stride]:
            assert {"i", "t", "lead", "follow", "swap_prob", "embrace"} <= set(fr)
            if "sam_swap_prob" in fr:
                exp = 1 - fr["sam_swap_prob"] if fr.get("repaired") else fr["sam_swap_prob"]
                assert abs(fr["swap_prob"] - exp) < 2e-4
            i = fr["i"]
            rd = {}
            for sub, pids in (("masks", (1, 2)), ("body", (1, 2)), ("hair", (2,))):
                for pid in pids:
                    m = cv2.imread(str(WORK / "dancers" / cam / sub / f"mask_{i:04d}_{pid:02d}.png"), cv2.IMREAD_GRAYSCALE)
                    assert m is not None and m.shape == (meta["height"], meta["width"]), (cam, sub, i)
                    assert set(np.unique(m)) <= {0, 255}
                    rd[sub, pid] = m > 0
            full, body, hair = rd["masks", 2], rd["body", 2], rd["hair", 2]
            assert not (body & hair).any() and ((body | hair) == full).all(), (cam, i)
            assert (rd["body", 1] == rd["masks", 1]).all(), (cam, i)
            assert not (rd["masks", 1] & full).any(), (cam, i)  # one owner per pixel


def test_cam_loader():
    if not (WORK / "people" / "07.npz").exists():
        pytest.skip("no larissa_kadu detections")
    cam = Cam.load(WORK, "07")
    assert len(cam.files) == len(cam.t) == cam.people["valid"].shape[0]


def test_clean_mask_relative_to_running_median_and_keeps_hands():
    m = np.zeros((400, 200), bool)
    m[200:350, 60:120] = True  # mostly occluded dancer: 9000 px now, 30000 px median
    m[20:33, 180:193] = True  # 169 px lit sign far away: > 1 % of the current area, < 2 % of the median
    out, n = clean_mask(m, ref_area=30000)
    assert n == 169 and not out[25, 185]
    kept, n2 = clean_mask(m, ref_area=30000, keep_pts=np.array([[186.0, 26.0]]))  # a wrist keypoint there
    assert n2 == 0 and kept[25, 185]


def test_clean_mask_partner_occlusion_vs_odd_colour():
    """A detached blob touching the partner is a hand peeking around them (kept) - unless its colour is
    unlike anything in the dancer's own body this frame (a lit sign touching the partner's hair)."""
    img = np.zeros((400, 200, 3), np.uint8)
    img[:, :] = (40, 40, 40)  # dark clothes
    img[200:350, 60:120] = (40, 40, 40)
    img[205:215, 70:110] = (120, 150, 200)  # some skin on the body
    m = np.zeros((400, 200), bool)
    m[200:350, 60:120] = True
    partner = np.zeros_like(m)
    partner[0:30, 150:200] = True
    m[20:33, 135:148] = True  # blob touching the partner
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    img2 = img.copy()
    img2[20:33, 135:148] = (120, 150, 200)  # skin-coloured: a hand
    out, n = clean_mask(m, ref_area=30000, partner=partner, lab=cv2.cvtColor(img2, cv2.COLOR_BGR2LAB))
    assert n == 0 and out[25, 140]
    img3 = img.copy()
    img3[20:33, 135:148] = (170, 250, 170)  # light green lit sign
    out, n = clean_mask(m, ref_area=30000, partner=partner, lab=cv2.cvtColor(img3, cv2.COLOR_BGR2LAB))
    assert n == 169 and not out[25, 140]
    assert clean_mask(m, ref_area=30000, partner=partner, lab=lab)[1] == 0  # dark like his clothes: kept


def test_clean_mask_drops_detached_specks_only():
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
    if os.environ.get("DANCECAP_GPU_TESTS") != "1":
        pytest.skip("opt-in GPU test: set DANCECAP_GPU_TESTS=1")
    if not (WORK / "dancers" / "candidates" / "07.npz").exists():
        pytest.skip("no larissa_kadu candidates")
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
