"""Generic people stage of one phone video (dancecap.people_mono): couple pick, slot separation, descriptions, role decision, final contract (synthetic)."""
from __future__ import annotations

import json

import cv2
import numpy as np

from dancecap import people_mono as M

W, H, FPS = 540, 960, 30.0
BLACK = [(35, 35, 38), (40, 40, 45), (45, 45, 50), (150, 110, 90)]          # torso, thigh, shin, head
WHITE = [(215, 215, 212), (150, 170, 200), (160, 175, 200), (170, 60, 50)]
GREY = [(128, 128, 128)] * 4


def kp_person(cx: float, cy: float, h: float, wave: float = 0.0, conf: float = 0.9) -> np.ndarray:
    """A standing COCO-17 skeleton of height h, hips at (cx, cy); `wave` (-1..1) swings the arms and one knee (a dancer's limbs move)."""
    k = np.zeros((17, 3), np.float32)
    k[:, 2] = conf
    k[0] = (cx, cy - 0.55 * h, conf)
    off = {5: (-0.1, -0.4), 6: (0.1, -0.4), 7: (-0.15, -0.2), 8: (0.15, -0.2), 9: (-0.15, 0.0), 10: (0.15, 0.0), 11: (-0.06, 0.0), 12: (0.06, 0.0),
           13: (-0.06, 0.22), 14: (0.06, 0.22), 15: (-0.06, 0.45), 16: (0.06, 0.45)}
    for j, (dx, dy) in off.items():
        k[j] = (cx + dx * h, cy + dy * h, conf)
    k[[7, 9], 1] -= wave * np.array([0.12, 0.3]) * h
    k[[8, 10], 1] += wave * np.array([0.05, 0.12]) * h
    k[[9, 10], 0] += wave * 0.1 * h
    k[13, 0] += wave * 0.06 * h
    k[1:5] = k[0]
    return k


def box_of(k: np.ndarray, h: float) -> np.ndarray:
    return np.array([k[:, 0].min() - 0.05 * h, k[0, 1] - 0.12 * h, k[:, 0].max() + 0.05 * h, k[15, 1] + 0.03 * h], np.float32)


def make_P(per_frame: list[list[tuple]], fps: float = FPS) -> dict:
    """P (load_persons layout) from per-frame lists of (kp, h, colours)."""
    fr, box, kp, col, fstart = [], [], [], [], [0]
    for f, persons in enumerate(per_frame):
        for k, h, c in persons:
            fr.append(f)
            kp.append(k)
            box.append(box_of(k, h))
            col.append(np.array(c, np.float32))
        fstart.append(len(fr))
    P = {"frame": np.array(fr, np.int32), "box": np.array(box, np.float32), "kp": np.array(kp, np.float32), "conf": np.full(len(fr), 0.9, np.float32),
         "col": np.array(col, np.float32), "fstart": np.array(fstart), "fps": fps, "size": np.array([W, H]), "n_frames": len(per_frame)}
    P["height"] = P["box"][:, 3] - P["box"][:, 1]
    return P


def couple_scene(F: int = 90) -> list[list[tuple]]:
    """Slot order in every frame: dancer A (black), dancer B (white), a larger static person, a small distant dancing pair."""
    out = []
    for f in range(F):
        w = np.sin(2 * np.pi * f / FPS)
        sway = 12 * np.sin(2 * np.pi * f / (2 * FPS))
        out.append([(kp_person(120 + sway, 600, 400, w), 400, BLACK), (kp_person(185 + sway, 600, 380, -w), 380, WHITE),
                    (kp_person(390, 610, 470, 0.0), 470, GREY), (kp_person(440, 380, 190, w), 190, GREY), (kp_person(478, 380, 180, -w), 180, GREY)])
    return out


def scene_index(P: dict, ids: np.ndarray) -> np.ndarray:
    """Which scene person (position in its frame) each chosen id is, -1 none."""
    return np.where(ids >= 0, ids - P["fstart"][:-1][:, None], -1)


def test_couple_pick_prefers_moving_close_pair():
    P = make_P(couple_scene())
    res = M.solve_identity(P, iters=1)
    who = scene_index(P, res["ids"])
    pair = np.sort(who, axis=1)
    assert (pair == [0, 1]).all(1).mean() >= 0.95                 # not the larger static person, not the small pair
    pair1 = np.sort(scene_index(P, res["ids1"]), axis=1)
    assert (pair1 == [0, 1]).all(1).mean() >= 0.95               # the symmetric pass already finds the couple


def test_crossing_keeps_slots_apart():
    F = 120
    frames = []
    for f in range(F):
        w = np.sin(2 * np.pi * f / FPS)
        xa = 150 + 240 * f / (F - 1)
        xb = 390 - 240 * f / (F - 1)
        frames.append([(kp_person(xa, 600, 400, w), 400, BLACK), (kp_person(xb, 600, 385, -w), 385, WHITE), (kp_person(40, 300, 150, 0.0), 150, GREY)])
    P = make_P(frames)
    res = M.solve_identity(P, iters=1)
    who = scene_index(P, res["ids"])
    s0 = who[who[:, 0] >= 0, 0]
    s1 = who[who[:, 1] >= 0, 1]
    a = 0 if (s0 == 0).mean() > 0.5 else 1
    assert (s0 == a).mean() >= 0.97 and (s1 == 1 - a).mean() >= 0.97   # each slot keeps ITS dancer through the crossing (frames 50-70)
    mid = who[50:70]
    assert ((mid[:, 0] == a) | (mid[:, 0] < 0)).all()


def test_description_parser_regions_and_colours():
    lead = M.parse_description("black t-shirt, jeans, shaved head")
    follow = M.parse_description("long red hair, white crop top, light ripped jeans, white sneakers")
    assert M.colour_name(lead["torso"]["rgb"]) == "black"
    assert M.colour_name(lead["thigh"]["rgb"]) == "denim" and "shin" in lead
    assert any("skin" in w for w in lead["head"]["words"])
    assert M.colour_name(follow["head"]["rgb"]) == "red"
    assert M.colour_name(follow["torso"]["rgb"]) == "white"
    assert follow["thigh"]["lab"][0] > lead["thigh"]["lab"][0] + 10           # light jeans are lighter than plain jeans
    assert any("sneakers" in w for w in follow["shin"]["words"]) and follow["shin"]["w"] == 0.6
    jl = M.parse_description("sheer long-sleeve cropped top with a brown printed pattern over a coral sports top, loose grey trousers, pale pink slip-on sneakers")
    assert {"torso", "thigh", "shin"} <= set(jl) and len(jl["torso"]["words"]) == 2
    assert M.colour_name(jl["thigh"]["rgb"]) == "grey"
    dark = M.parse_description("dark charcoal sneakers")
    assert dark["shin"]["lab"][0] < M.to_lab(M.COLOUR_RGB["charcoal"])[0]
    assert M.parse_description("") == {} and M.parse_description(None) == {}


def slot_rgb_follow_then_lead() -> np.ndarray:
    return np.array([[(220, 220, 218), (150, 170, 200), (200, 200, 200), (170, 60, 50)],
                     [(35, 35, 38), (70, 90, 130), (60, 70, 90), (180, 140, 120)]], float)


def test_role_by_description():
    lead = M.parse_description("black t-shirt, jeans, shaved head")
    follow = M.parse_description("long red hair, white crop top, light ripped jeans, white sneakers")
    ev = M.description_evidence(slot_rgb_follow_then_lead(), lead, follow)
    assert ev["lead_slot"] == 1 and ev["confidence"] >= 0.9
    d = M.decide_roles(ev, None, None)
    assert d["lead_slot"] == 1 and d["decided_by"] == "description"


def test_role_by_override():
    lead = M.parse_description("black t-shirt, jeans")
    follow = M.parse_description("white crop top, light jeans")
    ev = M.description_evidence(slot_rgb_follow_then_lead(), lead, follow)
    d = M.decide_roles(ev, None, {"swap": True})
    assert d["lead_slot"] == 0 and d["decided_by"] == "override" and d["automatic"]["lead_slot"] == 1
    assert M.decide_roles(ev, None, {"lead_slot": 1})["lead_slot"] == 1


def test_role_by_stature_fallback_and_default():
    ev = {"seconds": 12, "frames": 200, "log_ratio_01": -0.06, "z": 4.0}           # slot 1 is taller in the picture
    st = M.stature_decision(ev, 1.80, 1.65)
    d = M.decide_roles(None, st, None)
    assert d["lead_slot"] == 1 and d["decided_by"] == "stature" and d["confidence"] <= 0.85
    assert M.stature_decision(ev, 1.70, 1.69) is None                              # statures within 2 %: no evidence
    d = M.decide_roles(None, None, None)
    assert d["lead_slot"] == 0 and d["decided_by"] == "default"
    # measured on skeletons: two upright persons standing on the same row, slot 0 taller
    frames = [[(kp_person(150, 600, 400, 0.0), 400, BLACK), (kp_person(330, 600 + 0.45 * 30, 370, 0.0), 370, WHITE)] for _ in range(150)]
    P = make_P(frames)
    ids = np.stack([P["fstart"][:-1], P["fstart"][:-1] + 1], 1)
    sev = M.stature_evidence(P, ids, np.ones((150, 2)), FPS)
    assert sev["seconds"] == 5 and sev["log_ratio_01"] > 0.05
    assert M.stature_decision(sev, 1.80, 1.65)["lead_slot"] == 0


def write_pose_npz(path, P: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, frame=P["frame"], npass=np.zeros(len(P["frame"]), np.int8), box=P["box"], conf=P["conf"], kp=P["kp"], col=P["col"],
             n_frames=P["n_frames"], fps=P["fps"], size=P["size"], sizes=np.array([960, 1280]))


def test_anchors_stage_writes_identity_and_lead_first(tmp_path):
    P = make_P(couple_scene())
    ctx = M.context("synthetic_mono_take", work_root=tmp_path)
    ctx.dancers = {"lead": {"name": "L", "description": "black t-shirt, black trousers", "stature_m": 1.8},
                   "follow": {"name": "F", "description": "white top, light blue jeans", "stature_m": 1.65}}
    write_pose_npz(ctx.inn / "pose.npz", P)
    out = M.cmd_anchors(ctx)
    ident = json.loads((ctx.people / "identity.json").read_text())
    anchors = json.loads((ctx.people / "anchors0.json").read_text())
    assert ident["roles"]["decided_by"] == "description" and len(anchors) >= 2
    for a in anchors:
        f = a["frame"]
        assert {"john", "lula", "tier"} <= set(a)
        lead_box = out["P"]["box"][out["P"]["fstart"][f]]           # person 0 of the scene = the dancer in black
        assert np.allclose(a["john"]["box"], lead_box, atol=1.0)
    assert M.cmd_anchors(ctx) is None                                  # idempotent


def test_finalize_writes_contract(tmp_path):
    F = 3
    frames = [[(kp_person(60, 180, 160, 0.0), 160, BLACK), (kp_person(140, 180, 150, 0.0), 150, WHITE)] for _ in range(F)]
    P = make_P(frames)
    P["size"] = np.array([200, 300])
    ctx = M.context("synthetic_mono_take", work_root=tmp_path)
    write_pose_npz(ctx.inn / "pose.npz", P)
    store, box = {}, np.full((F, 2, 4), np.nan, np.float32)
    for f in range(F):
        for o in range(2):
            m = np.zeros((300, 200), np.uint8)
            b = P["box"][P["fstart"][f] + o].astype(int)
            m[max(b[1], 0):b[3], max(b[0], 0):b[2]] = 255
            store[f"m_{f}_{o}"] = np.frombuffer(cv2.imencode(".png", m)[1].tobytes(), np.uint8)
            box[f, o] = b
    np.savez(ctx.people / "fused1.npz", box=box, agree=np.full((F, 2), 0.9, np.float32), dist=np.zeros((F, 2), np.int32),
             area=np.full((F, 2), 5000, np.int32), **store)
    wb = np.full((F, 2, 2, 133, 3), np.nan, np.float32)
    for f in range(F):
        for o in range(2):
            k = P["kp"][P["fstart"][f] + o]
            wb[f, o, :, :17, :2] = k[:, :2]
            wb[f, o, :, :, 2] = 5.0
            wb[f, o, :, 17:, :2] = k[0, :2]
    np.savez(ctx.inn / "pose_wb1.npz", kp=wb)
    (ctx.people / "identity.json").write_text(json.dumps({"method": "test", "roles": {"lead_slot": 0, "decided_by": "description", "confidence": 0.9}}))
    summ = M.finalize(ctx, pts=np.arange(F) / FPS)
    z = np.load(ctx.people / "teachers_01.npz")
    dtypes = {"t": np.float64, "frame_index": np.int32, "ok": bool, "row": np.int16, "box": np.float32, "kp": np.float32, "conf": np.float32, "merged": bool,
              "role_conf": np.float32, "floor": np.float32, "size": np.int32, "attached": bool, "kp_wb": np.float32, "wb_variant": np.int8,
              "mask_area": np.int32, "mask_agree": np.float32, "mask_dist": np.int32, "visible": np.float32, "skel_in_partner": np.float32, "edge": bool,
              "box_iou": np.float32, "bib_dist": np.int32}
    assert set(z.files) == set(dtypes)
    for k, dt in dtypes.items():
        assert z[k].dtype == dt, k
    assert z["kp"].shape == (F, 2, 17, 3) and z["kp_wb"].shape == (F, 2, 133, 3) and (z["bib_dist"] == 9999).all()
    assert (z["row"] >= 0).all() and float(z["role_conf"].min()) > 0.8
    mk = np.load(ctx.people / "masks_01.npz")
    assert {"box", "area", "m_0_0", "m_2_1"} <= set(mk.files)
    assert (ctx.people / "pose_tables" / "01.npz").exists()
    assert json.loads((ctx.inn / "ocr.json").read_text()) == {"reads": []}
    assert json.loads((ctx.people / "teachers.json").read_text())["identity"]["decided_by"] == "description"
    assert summ["frames"] == F and json.loads((ctx.people / "coverage.json").read_text())[0]["s"] == 0
    assert "colour" in json.loads((ctx.people / "verification.json").read_text())


def test_reference_height_survives_a_merged_couple():
    """lk03_cut20 (2026-10-22): the couple embraced into one detection at 17 s; the 2nd tallest moving person was then a spectator a
    third of their size, the reference height followed it and the tracker locked onto two spectators. The tallest is the reference
    when the 2nd is much smaller, and the couple's own reference never falls far below the scene's."""
    import numpy as np
    from dancecap import people_mono as PM
    F = 90
    heights, frames = [], []
    for f in range(F):
        hs = [320.0, 300.0, 110.0, 100.0] if f < 45 else [330.0, 110.0, 100.0]     # from frame 45 the couple is one box
        heights += hs
        frames += [f] * len(hs)
    frames = np.array(frames)
    P = {"n_frames": F, "height": np.array(heights), "fstart": np.searchsorted(frames, np.arange(F + 1)), "frame": frames}
    ref = PM.reference_height(P, np.ones(len(frames), bool), 30.0)
    assert ref[:20].min() > 250 and ref[-20:].min() > 250
    ids = np.full((F, 2), -1)
    small = [i for i, h in enumerate(heights) if h <= 110.0]
    for f in range(F):          # a wrong pair of spectators chosen everywhere: the guard keeps the reference up
        sp = [i for i in small if frames[i] == f]
        ids[f] = sp[:2]
    r = PM.couple_reference(P, ids, 30.0, ref)
    assert (r >= PM.REF_GUARD * ref - 1e-9).all()
