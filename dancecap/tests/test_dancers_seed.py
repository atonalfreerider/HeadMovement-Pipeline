"""dancecap.dancers_seed (identity seeded from another take's teacher tracks) and the streaming labels of dancecap.dancers, on synthetic takes.

    ../.venv/bin/python -m pytest tests/test_dancers_seed.py -q
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dancecap import dancers as D  # noqa: E402
from dancecap import dancers_seed as S  # noqa: E402
from dancecap.cameras.world import dancer_keypoints  # noqa: E402

W, H, F = 320, 240, 40
SHIFT = 5        # source frame j = i + 5
SPEC = (0.5, 0.08), (0.45, 0.06), (0.55, 0.06), (0.4, 0.08), (0.6, 0.08), (0.3, 0.22), (0.7, 0.22), (0.2, 0.38), (0.8, 0.38), \
    (0.15, 0.52), (0.85, 0.52), (0.38, 0.55), (0.62, 0.55), (0.38, 0.75), (0.62, 0.75), (0.38, 0.95), (0.62, 0.95)
PASSER = np.array([0, 0, 30, 50], np.float32)     # a passer-by in the top-left corner (detection slot 3, frames 26-28 of a handover take)


def det_boxes(i: int) -> np.ndarray:
    """lead, follow, spectator of target frame i: the couple sways sideways (consecutive frames differ), never overlapping or at an edge."""
    s = 20 * np.sin(0.6 * i)
    return np.array([[30 + s, 50, 90 + s, 210], [180 - s, 50, 240 - s, 210], [262, 20, 300, 80]], np.float32)


def kpts_of(b: np.ndarray) -> np.ndarray:
    xy = np.array(SPEC) * [b[2] - b[0], b[3] - b[1]] + b[:2]
    return np.concatenate([xy, np.full((17, 1), 0.9)], 1).astype(np.float32)


def make_take(root: Path, n_det: int = 4, handover: bool = False) -> tuple[Path, Path]:
    """work/tgt (camera 01: frames + people) and work/srcv (teachers_01.npz: source frame j = i + SHIFT). Frame 35's follow detection carries
    the LEAD's skeleton (YOLO gave the partner's joints). handover: the source's FOLLOW slot is doubted at 24-25 and then on a passer-by at 26-28."""
    work, src = root / "work" / "tgt", root / "work" / "srcv"
    (work / "frames" / "01").mkdir(parents=True)
    (work / "people").mkdir(parents=True)
    (src / "people").mkdir(parents=True)
    t = 10.0 + np.arange(F) / 30
    boxes = np.zeros((F, n_det, 4), np.float32)
    valid = np.zeros((F, n_det), bool)
    kpts = np.zeros((F, n_det, 17, 3), np.float32)
    frames = []
    for i in range(F):
        b = det_boxes(i)
        boxes[i, :3], valid[i, :3] = b, True
        kpts[i, :3] = [kpts_of(x) for x in b]
        img = np.full((H, W, 3), 120, np.uint8)
        lb, fb = b[0].astype(int), b[1].astype(int)
        img[lb[1]:lb[3], lb[0]:lb[2]] = 30                                        # lead: dark clothes
        img[fb[1]:fb[3], fb[0]:fb[2]] = 235                                       # follow: white top and trousers
        img[fb[1]:fb[1] + 40, fb[0]:fb[2]] = (20, 20, 200)                        # red hair (BGR)
        cv2.imwrite(str(work / "frames" / "01" / f"{i:06d}.jpg"), img)
        frames.append({"file": f"{i:06d}.jpg", "pts": float(t[i]), "t": float(t[i])})
    kpts[35, 1] = kpts_of(det_boxes(35)[0])
    if handover:
        boxes[26:29, 3], valid[26:29, 3], kpts[26:29, 3] = PASSER, True, kpts_of(PASSER)
    (work / "frames" / "01" / "frames.json").write_text(json.dumps({"camera": "01", "width": W, "height": H, "fps_median": 30.0,
                                                                    "frames": frames}))
    np.savez_compressed(work / "people" / "01.npz", t=t, boxes=boxes, scores=valid * 0.9, valid=valid, kpts=kpts)
    Fs = F + 2 * SHIFT
    rng = np.random.default_rng(0)
    sbox = np.full((Fs, 2, 4), np.nan, np.float32)
    skp = np.full((Fs, 2, 17, 3), np.nan, np.float32)
    for jj in range(Fs):
        b = det_boxes(jj - SHIFT)[:2]
        sbox[jj] = b + rng.uniform(-1, 1, (2, 4))                                 # teacher boxes: the detections, jittered
        skp[jj] = [kpts_of(x) for x in b]                                         # ... and the source's own skeletons of them
    conf = np.full((Fs, 2), 0.9, np.float32)
    merged = np.zeros((Fs, 2), bool)
    conf[7 + SHIFT, 0] = 0.1                                                      # frame 7: the lead is doubted -> MISS
    sbox[12 + SHIFT] = det_boxes(12)[0]                                           # frame 12: one box for the embraced pair
    merged[12 + SHIFT] = True
    sbox[15 + SHIFT, 1] = det_boxes(15)[0] + 4                                    # frame 15: both on the lead (the follow's worse), NOT merged
    sbox[20 + SHIFT, 1], skp[20 + SHIFT, 1] = np.nan, np.nan                      # frame 20: no follow box
    if handover:
        conf[24 + SHIFT:26 + SHIFT, 1] = 0.1
        sbox[26 + SHIFT:29 + SHIFT, 1], skp[26 + SHIFT:29 + SHIFT, 1] = PASSER + 0.5, kpts_of(PASSER)
    np.savez_compressed(src / "people" / "teachers_01.npz", t=np.round(np.arange(Fs) / 30, 6), box=sbox, conf=conf,
                        ok=np.ones(Fs, bool), merged=merged, role_conf=np.full((Fs, 2), 0.97, np.float32), kp=skp,
                        size=np.array([W, H], np.int32), kp_wb=np.zeros((Fs, 2, 133, 3), np.float32))
    (src / "people" / "teachers.json").write_text(json.dumps({"identity": {"lead_slot": 1, "decided_by": "description", "confidence": 0.942}}))
    return work, src


def roles_of(work: Path) -> list[dict]:
    return json.loads((work / "dancers" / "roles.json").read_text())["cams"]["01"]["frames"]


# ------------------------------------------------------------------------------------------------ pure helpers


def test_parse_map_rules():
    m = S.parse_map("01=lucas_luiza_v:01:+81")
    assert (m["cam"], m["take"], m["src_cam"], m["kind"], m["shift"]) == ("01", "lucas_luiza_v", "01", "frames", 81)
    m = S.parse_map("02=lucas_luiza_h:01:clock-14.000")
    assert m["kind"] == "clock" and m["shift"] == -14.0 and m["rule"] == "clock-14.000"
    assert S.parse_map("03=x:02:-4")["shift"] == -4 and S.parse_map("03=x:02:clock+2.5")["shift"] == 2.5
    for bad in ("01=x:01", "01=x:01:81", "01=x:01:clock14", "x:01:+3"):
        with pytest.raises(ValueError):
            S.parse_map(bad)


def test_frame_map_offset_and_clock():
    t = np.arange(10) / 29.0
    j = S.frame_map(S.parse_map("01=a:01:+3"), t, np.arange(8) / 29.0)
    assert j.tolist() == [3, 4, 5, 6, 7, -1, -1, -1, -1, -1]
    assert S.frame_map(S.parse_map("01=a:01:-2"), t, np.arange(20) / 29.0)[:3].tolist() == [-1, -1, 0]
    src_t = np.round(np.arange(6000) / 30, 6)                                    # a 30 fps cut, t = n/30
    tt = 16.800117 + np.arange(5896) / 29.97003                                   # every 2nd frame of a 59.94 fps original
    j = S.frame_map(S.parse_map("02=h:01:clock-14.000"), tt, src_t)
    ref = np.round((tt - 14.0) * 30).astype(int)
    assert (j >= 0).all() and np.mean(j == ref) > 0.999                           # = round((t - 14) * 30) but at exact half-frame ties
    out = S.frame_map(S.parse_map("02=h:01:clock-14.000"), np.array([13.0, 14.0, 213.98, 214.0]), src_t)
    assert out.tolist() == [-1, 0, 5999, -1]                                      # outside the cut: -1 (half a frame of slack)


def test_match_is_one_to_one_and_optimal():
    """The 2 x N assignment maximises the summed IoU (a greedy lead-first match would leave the follow out); the same detection takes
    both roles only when the source flags `merged` and one role is left without a detection of its own; IoU <= 0.5, a doubted teacher
    (conf < min_conf) and j = -1 give MISS."""
    det = np.array([[0, 0, 100, 100], [30, 0, 130, 100], [300, 0, 400, 100]], np.float32)
    lead_t = np.array([16, 0, 116, 100], np.float32)                           # IoU 0.72 with det 0, 0.75 with det 1
    fol_t = np.array([36, 0, 136, 100], np.float32)                            # IoU 0.89 with det 1, 0.47 with det 0
    one = np.array([-20, 0, 80, 100], np.float32)                              # IoU 0.67 with det 0 only
    far = np.array([200, 0, 250, 100], np.float32)
    sbox = np.stack([np.stack(x) for x in ((lead_t, fol_t), (one, one), (one, one), (far, fol_t), (lead_t, fol_t), (lead_t, fol_t))])
    src = {"t": np.arange(6) / 30, "box": sbox, "conf": np.array([[.9, .9], [.9, .9], [.9, .9], [.9, .9], [.2, .9], [.9, .9]]),
           "merged": np.array([[0, 0], [0, 0], [1, 1], [0, 0], [0, 0], [0, 0]], bool)}
    cam = {"boxes": np.repeat(det[None], 6, 0), "valid": np.ones((6, 3), bool)}
    row, iou = S.match(np.array([0, 1, 2, 3, 4, -1]), src, cam, S.SeedConfig())
    assert row.dtype == np.int16 and row[0].tolist() == [0, 1] and sorted(row[1].tolist()) == [-1, 0]
    assert row[2:].tolist() == [[0, 0], [-1, 1], [-1, 1], [-1, -1]]
    assert abs(iou[0, 1] - 0.887) < 1e-3 and iou[3, 0] == 0 and iou[5].tolist() == [0, 0]
    # one detection, not merged: the role that overlaps it more keeps it
    src1 = {**src, "box": np.stack([np.stack([one, det[0]])]), "conf": src["conf"][:1], "merged": src["merged"][:1]}
    row1, _ = S.match(np.array([0]), src1, {"boxes": det[None, :1], "valid": np.ones((1, 1), bool)}, S.SeedConfig())
    assert row1.tolist() == [[-1, 0]]


def test_match_merged_frame_with_two_detections_never_shares():
    """A merged frame where each teacher box has a detection of its own over 0.5: one detection each, although sharing the pair box D
    (IoU 0.81 with both teachers) would sum higher (1.63 vs 1.58) - the old 'sharing as an equal option' gave both roles D."""
    a, b, d = [0, 0, 100, 200], [60, 0, 160, 200], [0, 0, 160, 200]
    ta, tb = np.array([0, 0, 130, 200], np.float32), np.array([30, 0, 160, 200], np.float32)
    src = {"t": np.arange(2) / 30, "box": np.stack([np.stack([ta, tb])] * 2), "conf": np.full((2, 2), 0.9), "merged": np.ones((2, 2), bool)}
    cam = {"boxes": np.array([[a, b, d], [d, [400, 0, 500, 200], [400, 0, 500, 200]]], np.float32), "valid": np.array([[1, 1, 1], [1, 0, 0]], bool)}
    row, iou = S.match(np.array([0, 1]), src, cam, S.SeedConfig())
    assert row[0, 0] != row[0, 1] and (row[0] >= 0).all() and (iou[0] > 0.5).all()
    assert row[1].tolist() == [0, 0]                                             # only the pair box: shared (merged), as before


def test_world_keep_drops_partner_skeletons_and_resolves_shared_detections():
    """roles.json keeps a role's detection unless its COCO skeleton is clearly the partner's (closer to the other role's source skeleton); a
    shared detection goes to the one role whose source skeleton it is, to neither when that cannot be decided: no source skeleton, or both
    source skeletons on one body (frame 4: the doubted source follow sits on the lead), where the check would be a coin flip."""
    la, fo = np.array([0, 0, 60, 160], np.float32), np.array([150, 0, 210, 160], np.float32)
    row = np.array([[0, 1], [0, 1], [0, 0], [0, 0], [0, 1], [0, 0]], np.int16)
    box = np.array([[la, fo], [la, fo], [la, la], [la, la], [la, fo], [la, la]], np.float32)
    kp = np.array([[kpts_of(la), kpts_of(fo)], [kpts_of(la), kpts_of(la)], [kpts_of(la)] * 2, [kpts_of(la)] * 2, [kpts_of(la)] * 2,
                   [kpts_of(la)] * 2])
    src_kp = np.array([[kpts_of(la), kpts_of(fo)]] * 6)
    src_kp[3] = np.nan
    src_kp[4:, 1] = kpts_of(la + [1, 0, 1, 0])
    keep, rep = S.world_keep(row, box, kp, src_kp)
    assert keep.tolist() == [[True, True], [True, False], [True, False], [False, False], [True, True], [False, False]]
    assert rep["partner_skeleton"] == {"lead": 0, "follow": 1} and rep["shared_frames"] == 3 and rep["shared_kept"] == 1
    assert rep["undecided"] == {"lead": 3, "follow": 3}
    keep0, rep0 = S.world_keep(row, box, kp, None)                                # no source skeletons: shared frames go to neither
    assert keep0[:2].all() and not keep0[[2, 3, 5]].any() and rep0["undecided"] == {"lead": 6, "follow": 6}


def test_scan_reports_every_offset_and_finds_the_rule(tmp_path):
    work, src_w = make_take(tmp_path)
    cam, src = S.load_cam(work, "01"), S.load_src(src_w, "01")
    j = S.frame_map(S.parse_map(f"01=srcv:01:+{SHIFT}"), cam["t"], src["t"])
    sc = S.scan_offset(j, src, cam, S.SeedConfig(scan=2, scan_step=1))
    assert [r["offset"] for r in sc["offsets"]] == [-2, -1, 0, 1, 2] and sc["best"] == 0
    assert sc["rule_median_iou"] > 0.9 and sc["neighbour_margin"] > 0.05
    sc1 = S.scan_offset(S.frame_map(S.parse_map("01=srcv:01:+6"), cam["t"], src["t"]), src, cam, S.SeedConfig(scan=2, scan_step=1))
    assert sc1["best"] == -1                                                     # a rule one frame off is reported, not adopted


def test_role_jumps_and_label_frame():
    box = np.repeat(np.array([[[10, 10, 50, 110], [60, 10, 100, 110]]], np.float32), 6, 0)
    box[3] = box[3, ::-1]                                                        # a swap for one frame: two jumps per role
    j = S.role_jumps(box, 0.25)
    assert j["lead"]["jumps"] == 2 and j["lead"]["pairs"] == 5 and j["lead"]["frames"] == [3, 4]
    assert j["lead"]["across_gaps"]["count"] == 0
    tele = np.repeat(box[:1], 12, 0)
    tele[3:8] = np.nan                                                           # no box for 5 frames, then on another person
    tele[8:, 0] += [300, 0, 300, 0]
    g = S.role_jumps(tele, 0.25)
    assert g["lead"]["jumps"] == 0 and g["follow"]["across_gaps"]["count"] == 0
    assert g["lead"]["across_gaps"] == {"count": 1, "frames": [8], "unreviewed": 1, "unreviewed_frames": [8]}
    kept = np.zeros((12, 2), bool)
    kept[8:10, 0] = True                                                         # --keep 01:lead:8-9 (checked by eye)
    assert S.role_jumps(tele, 0.25, kept=kept)["lead"]["across_gaps"]["unreviewed"] == 0
    assert S.role_jumps(tele, 0.25, max_gap=5)["lead"]["across_gaps"]["count"] == 0
    row = np.zeros((6, 2), np.int16)
    kp = np.full((6, 2, 17, 3), 0.9, np.float32)
    conf = np.array([[.5, .9], [.8, .9], [.9, .7], [.95, .95], [.9, .9], [.9, .9]], np.float32)
    assert S.choose_label(row, box, kp, conf, (200, 200)) == 3                   # highest min(conf)
    box[3, 0, 3] = 199                                                           # ... unless a box touches the bottom edge
    assert S.choose_label(row, box, kp, conf, (200, 200)) in (4, 5)
    assert S.choose_label(np.full((6, 2), -1, np.int16), box, kp, conf, (200, 200)) is None


# ------------------------------------------------------------------------------------------------ end to end


def test_run_writes_every_contract(tmp_path):
    work, _ = make_take(tmp_path)
    (work / "dancers").mkdir()
    (work / "dancers" / "roles.json").write_text(json.dumps({                    # another camera, already seeded by SAM: kept
        "cams": {"09": {"frames": [{"i": 0}]}, "01": {"frames": []}}, "labels_source": {"09": "auto"},
        "swaps_repaired": [{"cam": "09", "start": 1, "end": 2}, {"cam": "01", "start": 0, "end": 3}], "identity_flags": {"01": ["x"], "09": ["y"]}}))
    rep = S.run(work, [S.parse_map(f"01=srcv:01:+{SHIFT}")], S.SeedConfig(scan=2))
    # roles.json: ONE entry per frames.json frame, in order (dancer_keypoints reads it by position); dancers_pod.merge's schema
    roles = json.loads((work / "dancers" / "roles.json").read_text())
    fr = roles["cams"]["01"]["frames"]
    assert [x["i"] for x in fr] == list(range(F)) and set(roles["cams"]) == {"01", "09"}
    assert fr[7]["lead"]["box"] is None and fr[7]["follow"]["box"] is not None and fr[20]["follow"]["box"] is None
    assert fr[15]["follow"]["box"] is None and fr[35]["follow"]["box"] is None and fr[35]["follow"]["conf"] == 0  # partner's skeleton
    assert fr[12]["embrace"] and fr[12]["follow"]["box"] is None and fr[12]["lead"]["box"] is not None           # shared: the lead's skeleton
    assert not any(x["lead"]["box"] is not None and x["lead"]["box"] == x["follow"]["box"] for x in fr)
    assert {"i", "t", "lead", "follow", "swap_prob", "sam_swap_prob", "embrace", "repaired", "anchor"} <= set(fr[0])
    assert roles["swaps_repaired"] == [{"cam": "09", "start": 1, "end": 2}] and roles["identity_flags"] == {"09": ["y"]}
    assert roles["labels_source"] == {"01": "seed:srcv"}
    # world.dancer_keypoints gets exactly the kept detections' joints, role by role, never one skeleton for both
    kp = dancer_keypoints(work, "01", F)
    p = np.load(work / "people" / "01.npz")
    for i in (0, 9, 33):
        assert np.array_equal(kp["lead"][i], p["kpts"][i, 0]) and np.array_equal(kp["follow"][i], p["kpts"][i, 1])
    assert not kp["lead"][7].any() and not kp["follow"][12].any() and np.array_equal(kp["lead"][12], p["kpts"][12, 0])
    assert not kp["follow"][35].any()
    # teachers_<cam>.npz: the recap contract on this take's own detections (the shared and the partner-skeleton matches kept, merged flagged)
    tz = np.load(work / "people" / "teachers_01.npz")
    for k, shape in (("t", (F,)), ("frame_index", (F,)), ("ok", (F,)), ("row", (F, 2)), ("box", (F, 2, 4)), ("kp", (F, 2, 17, 3)),
                     ("conf", (F, 2)), ("cost", (F, 2)), ("kind", (F, 2)), ("nj", (F, 2)), ("gap", (F, 2)), ("merged", (F, 2)),
                     ("role_conf", (F, 2)), ("lead_slot", (F,)), ("floor", (F, 2, 2)), ("size", (2,)), ("src_frame", (F,))):
        assert tz[k].shape == shape, k
    assert tz["row"].dtype == np.int16 and (tz["frame_index"] == -1).all() and np.isnan(tz["floor"]).all()
    assert tz["row"][3].tolist() == [0, 1] and tz["row"][7, 0] == -1 and tz["conf"][7, 0] == 0 and np.isnan(tz["cost"][7, 0])
    assert tz["row"][12].tolist() == [0, 0] and tz["merged"][12].all() and tz["row"][35].tolist() == [0, 1]
    assert np.array_equal(tz["box"][3], p["boxes"][3, :2]) and tz["nj"][3].tolist() == [17, 17] and tz["src_frame"][3] == 3 + SHIFT
    assert tz["size"].tolist() == [W, H] and str(tz["src"]) == f"srcv:01:+{SHIFT}" and not tz["attached"]
    tr = np.load(work / "dancers" / "teacher_tracks" / "01.npz")
    assert np.array_equal(tr["idx"], tz["row"]) and np.isnan(tr["boxes"][7, 0]).all() and tr["conf"].dtype == np.float32
    assert not list(work.rglob("*.part*"))
    # labels: every key dancers.review_tile / apply_override / dancers.run_camera read; front_pod labels-review renders it
    lab = json.loads((work / "dancers" / "label_01.json").read_text())
    assert json.loads((work / "dancers" / "labels.json").read_text()) == {"01": lab}
    assert {"frame", "t", "lead_box", "follow_box", "by", "confidence", "frame_llr", "evidence_agreement", "track_switches", "cues",
            "descriptors", "colour_names"} <= set(lab)
    assert lab["by"] == "seed:srcv" and lab["confidence"] == 0.942 and lab["evidence_agreement"] == 0.97
    for r in ("lead", "follow"):
        assert set(lab["colour_names"][r]) == {"top", "trousers", "red_above_shoulders"} and "torso_v" in lab["descriptors"][r]
    assert lab["colour_names"]["follow"]["top"] == "white" and lab["colour_names"]["lead"]["top"] == "black"
    assert lab["seed"]["colour_rule_llr"] > 0 and lab["frame_llr"] == abs(lab["seed"]["colour_rule_llr"])   # the demo rule agrees here
    i = lab["frame"]
    assert lab["lead_box"] == [round(float(v), 1) for v in p["boxes"][i, 0]] and fr[i]["lead"]["box"] == lab["lead_box"]
    tile = D.review_tile(D.Cam.load(work, "01"), D.load_labels(work)["01"])
    assert tile.shape[0] == 720
    # sheet + report
    assert cv2.imread(str(work / "review" / "seed_roles.jpg")) is not None
    sr = json.loads((work / "dancers" / "seed_report.json").read_text())
    c = sr["cams"]["01"]
    assert c["match"]["same_detection_unmerged"] == 0 and c["scan"]["best"] == 0 and c["map"]["size"] == [W, H]
    assert c["match"]["world"]["shared_frames"] == 1 and c["match"]["world"]["shared_kept"] == 1
    assert c["match"]["world"]["partner_skeleton"] == {"lead": 0, "follow": 1} and c["src_identity"]["npz_slots"] == "lead, follow"
    assert c["match"]["both"] == round(1 - 3 / F, 4) and c["match"]["both_at_min_conf_0"] == round(1 - 2 / F, 4)
    assert set(c["g0"]["checks"]) == {"scan_best_is_rule", "scan_median_iou", "scan_neighbour_margin", "both_matched",
                                       "same_detection_unmerged", "role_jumps", "across_gaps_unreviewed"}
    assert c["g0"]["checks"]["both_matched"]["value"] == round(1 - 5 / F, 4) and rep["g0_pass"] is c["g0"]["pass"] is True
    assert c["teachers_carry_over"] == {"frame_index": False, "floor_kept": 0, "floor_lost": 0}
    assert "pose confidence" in sr["settings"]["min_conf_on"]


def test_override_swaps_exactly_once(tmp_path):
    """labels_override.json {cam: {swap: true}}: re-running the seed flips the camera in teachers / roles.json (read directly), while labels.json
    and teacher_tracks keep the source's order, which load_labels / a dancer_one-style consumer flip: every consumer sees the roles swapped ONCE."""
    work, _ = make_take(tmp_path)
    m = [S.parse_map(f"01=srcv:01:+{SHIFT}")]
    S.run(work, m, S.SeedConfig(scan=1))
    t0 = np.load(work / "people" / "teachers_01.npz")["row"]
    (work / "dancers" / "labels_override.json").write_text(json.dumps({"01": {"swap": True}}))
    rep = S.run(work, m, S.SeedConfig(scan=1))
    tz = np.load(work / "people" / "teachers_01.npz")
    assert np.array_equal(tz["row"], t0[:, ::-1])
    roles = json.loads((work / "dancers" / "roles.json").read_text())
    final = D.load_labels(work)["01"]
    i = final["frame"]
    assert roles["cams"]["01"]["frames"][i]["lead"]["box"] == final["lead_box"]           # roles.json and the reviewed label agree
    assert final["by"] == "override" and roles["labels_source"] == {"01": "override"}
    raw = json.loads((work / "dancers" / "labels.json").read_text())["01"]
    assert raw["by"] == "seed:srcv" and raw["follow_box"] == final["lead_box"]           # stored once, in the source's order
    assert np.array_equal(np.load(work / "dancers" / "teacher_tracks" / "01.npz")["idx"], t0)
    c = json.loads((work / "dancers" / "seed_report.json").read_text())["cams"]["01"]
    assert c["override_swap"] is True and rep["cams"]["01"]["override_swap"] is True
    assert c["match"]["world"]["partner_skeleton"] == {"lead": 1, "follow": 0}           # the report speaks the reviewed roles
    assert c["label"]["colour_names"]["lead"]["top"] == "white" and c["label"]["colour_rule_llr"] < 0
    kp = dancer_keypoints(work, "01", F)
    p = np.load(work / "people" / "01.npz")
    assert np.array_equal(kp["lead"][3], p["kpts"][3, 1])                                # the follow's detection is now the lead


def test_drop_blanks_a_reviewed_stretch_in_the_reviewed_roles(tmp_path):
    """--drop CAM:ROLE:I0-I1 leaves that role unmatched over the stretch everywhere (roles.json, teachers, teacher_tracks); the role is the
    reviewed one, so with an override swap it is the source's other slot. --drop / --keep on a camera without a --map is refused."""
    assert S.parse_drop("02:lead:5181-5221") == ("02", "lead", 5181, 5221)
    for bad in ("02:lead:9-3", "02:judge:1-2", "02:lead:5"):
        with pytest.raises(ValueError):
            S.parse_drop(bad)
    work, _ = make_take(tmp_path)
    m = [S.parse_map(f"01=srcv:01:+{SHIFT}")]
    with pytest.raises(ValueError, match="09"):
        S.run(work, m, S.SeedConfig(scan=1), drops=[S.parse_drop("09:lead:0-5")])
    rep = S.run(work, m, S.SeedConfig(scan=1), drops=[S.parse_drop("01:follow:30-33")])
    fr = roles_of(work)
    assert all(fr[i]["follow"]["box"] is None and fr[i]["lead"]["box"] is not None for i in range(30, 34))
    assert fr[29]["follow"]["box"] is not None and fr[34]["follow"]["box"] is not None
    assert rep["cams"]["01"]["dropped"] == [{"role": "follow", "start": 30, "end": 33, "matched_before": 4}]
    assert (np.load(work / "dancers" / "teacher_tracks" / "01.npz")["idx"][30:34, 1] == -1).all()
    (work / "dancers" / "labels_override.json").write_text(json.dumps({"01": {"swap": True}}))
    S.run(work, m, S.SeedConfig(scan=1), drops=[S.parse_drop("01:lead:30-33")])
    fr = roles_of(work)
    assert fr[31]["lead"]["box"] is None and fr[31]["follow"]["box"] is not None
    assert (np.load(work / "dancers" / "teacher_tracks" / "01.npz")["idx"][30:34, 1] == -1).all()   # the source's follow slot


def test_across_gap_handover_fails_g0_until_dropped_or_kept_in_the_report_roles(tmp_path):
    """The source hands its follow to a passer-by after two doubted frames: G0 fails on the unreviewed across-gap flag (and the return jump),
    the sheet gets a flagged row. With an override swap the report names that role 'lead' - the name --drop takes - and dropping it removes
    the passer-by while the reviewed follow keeps its own detection."""
    work, _ = make_take(tmp_path, handover=True)
    m = [S.parse_map(f"01=srcv:01:+{SHIFT}")]
    rep = S.run(work, m, S.SeedConfig(scan=1))
    c = rep["cams"]["01"]
    assert c["jumps"]["follow"]["across_gaps"]["unreviewed_frames"] == [26] and c["jumps"]["lead"]["across_gaps"]["count"] == 0
    assert not c["g0"]["checks"]["across_gaps_unreviewed"]["pass"] and not rep["g0_pass"] and c["flagged_frames"][0] == 26
    assert roles_of(work)[27]["follow"]["box"] == PASSER.tolist()
    kept = S.run(work, m, S.SeedConfig(scan=1), keeps=[S.parse_drop("01:follow:26-28", "--keep")])["cams"]["01"]
    assert kept["g0"]["checks"]["across_gaps_unreviewed"]["pass"] and kept["jumps"]["follow"]["across_gaps"]["count"] == 1
    (work / "dancers" / "labels_override.json").write_text(json.dumps({"01": {"swap": True}}))
    sw = S.run(work, m, S.SeedConfig(scan=1))["cams"]["01"]
    assert sw["jumps"]["lead"]["across_gaps"]["unreviewed_frames"] == [26] and sw["jumps"]["follow"]["across_gaps"]["count"] == 0
    rep = S.run(work, m, S.SeedConfig(scan=1), drops=[S.parse_drop("01:lead:26-28")])
    fr = roles_of(work)
    assert all(fr[i]["lead"]["box"] is None for i in (26, 27, 28))
    assert fr[27]["follow"]["box"] == [round(float(v), 1) for v in det_boxes(27)[0]]   # the source's lead (now the reviewed follow) untouched
    assert rep["cams"]["01"]["g0"]["checks"]["across_gaps_unreviewed"]["pass"] and rep["cams"]["01"]["g0"]["checks"]["role_jumps"]["pass"]


def test_rerun_keeps_step11_floor_and_frame_index(tmp_path):
    """carlos_aline_inputs --teachers-floor fills frame_index and floor into teachers_<cam>.npz; a re-run (here with a --drop) keeps them on
    the role-frames whose detection is unchanged, and an override swap moves each floor with its detection."""
    work, _ = make_take(tmp_path)
    m = [S.parse_map(f"01=srcv:01:+{SHIFT}")]
    S.run(work, m, S.SeedConfig(scan=1))
    tp = work / "people" / "teachers_01.npz"
    tt = dict(np.load(tp))
    tt["frame_index"] = np.arange(F, dtype=np.int32) + 100
    tt["floor"] = np.where((tt["row"] >= 0)[..., None], np.stack([np.arange(F)] * 2, 1)[..., None] + [[0.0, 0.5], [1.0, 1.5]], np.nan)
    np.savez_compressed(tp, **tt)
    rep = S.run(work, m, S.SeedConfig(scan=1), drops=[S.parse_drop("01:follow:30-33")])
    tz = np.load(tp)
    assert (tz["frame_index"] == np.arange(F) + 100).all()
    assert np.isnan(tz["floor"][30:34, 1]).all() and np.array_equal(tz["floor"][29], tt["floor"][29]) and np.array_equal(tz["floor"][31, 0], [31, 31.5])
    assert rep["cams"]["01"]["teachers_carry_over"]["floor_lost"] == 4 and rep["cams"]["01"]["teachers_carry_over"]["frame_index"]
    (work / "dancers" / "labels_override.json").write_text(json.dumps({"01": {"swap": True}}))
    S.run(work, m, S.SeedConfig(scan=1))
    tz = np.load(tp)
    assert np.array_equal(tz["floor"][3], tt["floor"][3][::-1])                          # the floor follows its detection
    assert np.isnan(tz["floor"][31, 0]).all()                                            # dropped before: lost, not invented


def test_scan_action_writes_nothing_and_main_exits_2_on_a_failed_gate(tmp_path, monkeypatch):
    work, _ = make_take(tmp_path)
    rep = S.run(work, [S.parse_map(f"01=srcv:01:+{SHIFT + 1}")], S.SeedConfig(scan=1), write=False)
    assert not (work / "dancers").exists() and not (work / "review").exists()
    assert rep["cams"]["01"]["g0"]["checks"]["scan_best_is_rule"]["pass"] is False and rep["g0_pass"] is False
    monkeypatch.setattr(S, "Take", types.SimpleNamespace(load=lambda name: types.SimpleNamespace(work=work)))
    with pytest.raises(SystemExit) as e:
        S.main(["scan", "tgt", "--map", f"01=srcv:01:+{SHIFT + 1}", "--scan", "1"])
    assert e.value.code == 2
    assert S.main(["scan", "tgt", "--map", f"01=srcv:01:+{SHIFT}", "--scan", "1"]) is None


def test_source_on_another_frame_size_is_refused(tmp_path):
    work, src = make_take(tmp_path)
    z = dict(np.load(src / "people" / "teachers_01.npz"))
    z["size"] = np.array([2 * W, 2 * H], np.int32)
    np.savez_compressed(src / "people" / "teachers_01.npz", **z)
    with pytest.raises(ValueError, match="640x480"):
        S.run(work, [S.parse_map(f"01=srcv:01:+{SHIFT}")], S.SeedConfig(scan=1), write=False)


# ------------------------------------------------------------------------------------------------ dancers.py: streaming labels


def _cam(tmp_path: Path) -> D.Cam:
    work, _ = make_take(tmp_path)
    return D.Cam.load(work, "01")


def test_labels_for_cam_streams_frames_and_matches_the_decoded_run(tmp_path, monkeypatch):
    """Without imgs, labels_for_cam reads each JPEG on demand (dancers.LazyImages, a few decoded ahead) and the evidence is identical to the
    run on fully decoded frames; only frames with both candidate tracks are decoded, each once."""
    cam = _cam(tmp_path)
    idx = np.tile(np.array([0, 1]), (F, 1))
    idx[5:9, 1] = D.MISS
    tracks = {"idx": idx, "boxes": cam.people["boxes"][:, :2]}
    monkeypatch.setattr(D, "candidate_tracks", lambda *a, **k: tracks)
    monkeypatch.setattr(D, "auto_label", lambda *a, **k: {"frame": 0})
    seen = []
    orig = D.track_evidence

    def spy(cam_, tracks_, imgs, cfg):
        seen.append(imgs)
        return orig(cam_, tracks_, imgs, cfg)
    monkeypatch.setattr(D, "track_evidence", spy)
    reads = []
    real_imread = cv2.imread
    monkeypatch.setattr(D.cv2, "imread", lambda p, *a: reads.append(p) or real_imread(p, *a))
    _, ev_lazy, _ = D.labels_for_cam(cam, D.DancersConfig())
    assert isinstance(seen[0], D.LazyImages) and len(seen[0]) == F and seen[0].pool is None   # closed after use
    assert len(reads) == F - 4 and len(set(reads)) == F - 4                      # the 4 frames without both tracks are never decoded
    _, ev_full, _ = D.labels_for_cam(cam, D.DancersConfig(), D.read_frames(cam.files))
    for k in ("llr", "pair_iou", "a_is_follow", "swap_post"):
        assert np.array_equal(ev_lazy[k], ev_full[k]), k
    assert json.dumps([ev_lazy["desc"], ev_lazy["cues"]]) == json.dumps([ev_full["desc"], ev_full["cues"]])   # NaN-safe
    with pytest.raises(FileNotFoundError):
        D.LazyImages([tmp_path / "missing.jpg"])[0]


def test_lazy_images_prefetch_is_bounded_and_order_free(tmp_path):
    cam = _cam(tmp_path)
    order = list(range(0, F, 2))
    li = D.LazyImages(cam.files, order, ahead=3, workers=2)
    ref = cv2.imread(str(cam.files[4]), cv2.IMREAD_COLOR)
    for i in order[:5]:
        img = li[i]
        assert img.shape == (H, W, 3) and len(li.pending) <= 3
    assert np.array_equal(li[4], ref) and np.array_equal(li[3], cv2.imread(str(cam.files[3]), cv2.IMREAD_COLOR))  # re-read, outside order
    li[30]                                                                       # a jump ahead drops the frames it skipped
    assert all(li.pos[f] > li.pos[30] for f in li.pending) and len(li.pending) <= 3
    with pytest.raises(FileNotFoundError):
        D.LazyImages([tmp_path / "missing.jpg"] * 3, [0, 1, 2], ahead=2)[0]
    li.close()
    assert li.pool is None and not li.pending and li[6].shape == (H, W, 3)


def test_run_labels_writes_per_camera_without_a_partial_done_marker(tmp_path, monkeypatch):
    """Per camera: label_<cam>.json always; labels.json (dancers_labels' done marker) only when it existed before the run, so a first run that
    fails on a later camera leaves no partial labels.json, and a re-run that fails keeps the cameras relabelled so far. The final labels.json
    is the same text as before."""
    work, _ = make_take(tmp_path)
    (work / "frames" / "02").mkdir()
    for f in ("frames/01/frames.json", "people/01.npz"):
        (work / f.replace("01", "02")).write_bytes((work / f).read_bytes())
    take = types.SimpleNamespace(work=work, sources=[types.SimpleNamespace(id="01"), types.SimpleNamespace(id="02")])
    tracks = {"idx": np.zeros((F, 2), int), "boxes": np.zeros((F, 2, 4))}
    ev = {"llr": np.zeros(F), "a_is_follow": np.zeros(F, bool), "swap_post": np.zeros(F), "pair_iou": np.zeros(F)}
    lab = {"frame": 3, "confidence": 0.9, "frame_llr": 2.0, "evidence_agreement": 1.0, "track_switches": 0, "cues": {}}
    state = {"fail": True, "frame": 3}

    def fake(cam, cfg, imgs=None):
        if cam.id == "02" and state["fail"]:
            raise RuntimeError("cam 02: no clean frame")
        return tracks, ev, dict(lab, frame=state["frame"] + int(cam.id))
    monkeypatch.setattr(D, "labels_for_cam", fake)
    monkeypatch.setattr(D, "review_tile", lambda cam, lab_: np.zeros((8, 8, 3), np.uint8))
    with pytest.raises(RuntimeError):
        D.run_labels(take)
    out = work / "dancers"
    assert not (out / "labels.json").exists() and json.loads((out / "label_01.json").read_text()) == dict(lab, frame=4)
    state["fail"] = False
    D.run_labels(take)
    want = {"01": dict(lab, frame=4), "02": dict(lab, frame=5)}
    assert (out / "labels.json").read_text() == json.dumps(want, indent=1)
    state.update(fail=True, frame=10)
    with pytest.raises(RuntimeError):
        D.run_labels(take)
    assert json.loads((out / "labels.json").read_text()) == {"01": dict(lab, frame=11), "02": dict(lab, frame=5)}
