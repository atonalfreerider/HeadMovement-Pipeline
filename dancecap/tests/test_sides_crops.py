"""dancecap.sides.crops (the pure parts): plan expansion and automatic crop times, the crop box and the undistorted crop maps, the R9
sign map ("her left arm on top" -> the sign of G), Jev answers -> class distributions, Platt calibration with leave-one-out, the
calibration report (R9 needs a relevant negative), the crops.json items in sides.decide's contract, the Jev request loop with a fake
client (cache, refusal without the upload go-ahead), the owner pick, and the package rules (no take names, no import of the sibling
modules at import time)."""
import ast
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dancecap.sides import crops as C

SRC = Path(C.__file__)


# ------------------------------------------------------------------------------------------------ package rules

def test_no_take_constants_in_crops():
    txt = SRC.read_text(encoding="utf-8").lower()
    for name in ("larissa", "kadu", "recap", "carlos", "aline", "john_lula", "lula"):
        assert name not in txt, name


def test_top_level_imports_are_stdlib_and_numpy_only():
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    allowed = {"__future__", "argparse", "hashlib", "json", "math", "subprocess", "sys", "dataclasses", "pathlib", "numpy"}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for a in node.names:
                assert a.name.split(".")[0] in allowed, a.name
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0 and (node.module or "").split(".")[0] in allowed, (node.module, node.level)


# ------------------------------------------------------------------------------------------------ crop times

def test_expand_plan_times_and_range():
    plan = {"groups": [{"name": "b", "times": [34.0, 30.5], "cams": ["00"], "questions": ["R4", "R9"]},
                       {"name": "a", "range": [199.0, 199.3, 0.1], "cams": "auto", "questions": ["R3"], "set": "target"}]}
    e = C.expand_plan(plan)
    assert [x["t"] for x in e] == [30.5, 34.0, 199.0, 199.1, 199.2, 199.3]
    assert e[0]["cams"] == ["00"] and e[0]["questions"] == ["R4", "R9"] and e[0]["set"] == "target" and e[0]["kind"] == "couple"
    assert e[-1]["cams"] == "auto"


def test_crop_times_auto_from_relations():
    T = 300
    gate = np.zeros(T, bool)
    gate[60:120] = True                                     # 2 s defined
    post = np.full((T, 4), 0.25)
    post[60:90, 2] = 0.95                                   # decided back for 1 s, undecided after
    post[60:90, :2] = 0.01
    post[60:90, 3] = 0.03
    rel = {"R4__gate": gate, "R4__band": gate, "R4__post": post}
    out = C.crop_times(rel, None, hz=30.0)
    assert len(out) == 1 and out[0]["cams"] == "auto" and out[0]["questions"] == ["R4"]
    assert abs(out[0]["t"] - (90 + 119) / 2 / 30) < 1e-3
    # a planned time within 0.5 s of it suppresses the automatic one
    plan = {"groups": [{"name": "p", "times": [round(out[0]["t"] + 0.3, 2)], "cams": ["01"]}]}
    out2 = C.crop_times(rel, plan, hz=30.0)
    assert [x["group"] for x in out2] == ["p"]


def test_crop_id_format():
    assert C.crop_id(30.5, "00") == "t0030.50_c00"
    assert C.crop_id(199.7, "06", "face") == "t0199.70_c06_face"


# ------------------------------------------------------------------------------------------------ crop geometry

def test_crop_box_square_margin_and_limits():
    uv = np.array([[100.0, 200.0], [300.0, 260.0], [np.nan, 5.0]])
    cx, cy, side = C.crop_box(uv, (1080, 1920), margin=0.25, min_side=50)
    assert (cx, cy) == (200.0, 230.0) and math.isclose(side, 200 * 1.5)
    assert C.crop_box(uv, (1080, 1920), 0.0, min_side=400)[2] == 400
    assert C.crop_box(uv, (300, 1920), 1.0, min_side=10)[2] == 300           # at most the shorter image side
    with pytest.raises(ValueError):
        C.crop_box(np.full((2, 2), np.nan), (10, 10), 0.1, 1)


def test_in_frame_share():
    uv = np.array([[10, 10], [-5, 10], [10, 2000], [50, 50]], float)
    d = np.array([2.0, 2.0, 2.0, 0.1])
    assert C.in_frame_share(uv, d, (100, 100), 0.3) == 0.25


def _cam():
    K = np.array([[1000.0, 0, 540], [0, 1000.0, 960], [0, 0, 1]])
    R = np.eye(3)
    t = np.array([0.0, 0.0, 3.0])
    return K, R, t


@pytest.mark.parametrize("k1", [0.0, 0.03, -0.02])
def test_crop_maps_send_a_projected_point_to_its_raw_pixel(k1):
    K, R, t = _cam()
    X = np.array([[0.31, -0.22, 0.4]])
    uv, d = C.project(K, R, t, X)                           # undistorted pinhole pixel
    cx, cy, side, tile = 600.0, 900.0, 400.0, 256
    Kc = C.crop_K(K, cx, cy, side, tile)
    uc, _ = C.project(Kc, R, t, X)                          # its pixel in the crop
    mx, my = C.crop_maps(K, cx, cy, side, tile, k1)
    # bilinear read of the map at the (fractional) crop pixel
    u, v = uc[0]
    i0, j0 = int(np.floor(v)), int(np.floor(u))
    a, b = v - i0, u - j0

    def bil(M):
        return (1 - a) * (1 - b) * M[i0, j0] + (1 - a) * b * M[i0, j0 + 1] + a * (1 - b) * M[i0 + 1, j0] + a * b * M[i0 + 1, j0 + 1]
    # the raw (distorted) pixel of X: SIMPLE_RADIAL on the normalised undistorted coordinates
    x = (uv[0] - K[[0, 1], [2, 2]]) / K[[0, 1], [0, 1]]
    raw = K[[0, 1], [2, 2]] + K[[0, 1], [0, 1]] * x * (1 + k1 * (x ** 2).sum())
    assert abs(bil(mx) - raw[0]) < 1e-2 and abs(bil(my) - raw[1]) < 1e-2


def test_crop_img_identity_resample_without_distortion():
    cv2 = pytest.importorskip("cv2")
    K, _, _ = _cam()
    img = np.zeros((1920, 1080, 3), np.uint8)
    img[900:910, 600:610] = 255                              # a white square at (600..609, 900..909)
    out = C.crop_img(img, 605.0, 905.0, 100.0, 100, K, 0.0)
    ys, xs = np.nonzero(out[..., 0] > 128)
    assert abs(xs.mean() - 49.5) < 1.0 and abs(ys.mean() - 49.5) < 1.0
    assert out.shape == (100, 100, 3)


# ------------------------------------------------------------------------------------------------ R9 sign map

def _arms(her_dy: float, his_dir=(1.0, 0.0, 0.0)):
    """his right arm: shoulder -> elbow -> wrist with the forearm along his_dir through the origin; her left arm crossing it along z,
    her_dy above (+) / below (-) it"""
    d = np.asarray(his_dir, float)
    A = np.array([[-0.5, 0.3, 0.0], [-0.15, 0.0, 0.0], [0.0, 0.0, 0.0]])
    A[1] = -0.15 * d
    A[2] = 0.15 * d
    A[0] = A[1] + np.array([-0.05, 0.3, 0.0])
    B = np.array([[0.0, her_dy + 0.3, -0.45], [0.0, her_dy, -0.15], [0.0, her_dy, 0.15]])
    return A, B


def test_r9_sign_map_matches_the_gli_sign_of_her_on_top():
    from dancecap.sides import geom
    A, Bt = _arms(+0.03)
    _, Bb = _arms(-0.03)
    Gt = float(geom.gli_chain(A[None], Bt[None])[0])
    Gb = float(geom.gli_chain(A[None], Bb[None])[0])
    assert np.sign(Gt) == -np.sign(Gb) and abs(Gt) > 0.02
    mt = C.r9_sign_map(A, Bt)
    mb = C.r9_sign_map(A, Bb)
    assert mt["plus_if_her_on_top"] == int(np.sign(Gt))       # computed from her-on-top geometry: + side = sign(G) there
    assert mb["plus_if_her_on_top"] == mt["plus_if_her_on_top"]  # the map does not depend on which side the fit has her
    assert mt["her_above"] is True and mb["her_above"] is False and mt["why"] == "ok"


def test_r9_sign_map_undefined_cases():
    A, B = _arms(+0.6)                                        # far apart: the arms do not meet
    m = C.r9_sign_map(A, B)
    assert m["plus_if_her_on_top"] == 0 and "meet" in m["why"]
    A, B = _arms(+0.03)
    B = B[:, [1, 2, 0]] * 0 + np.array([[0.0, -0.45, 0.03], [0.0, -0.15, 0.03], [0.0, 0.15, 0.03]])   # her forearm vertical, beside his
    m = C.r9_sign_map(A, B)
    assert m["plus_if_her_on_top"] == 0                       # the crossing is not seen from above (or unlinked)


# ------------------------------------------------------------------------------------------------ answers -> evidence

def test_r4_class_probs_flatten_and_normalise():
    p, w = C.r4_class_probs({"back": 0.6, "side": 0.2, "front": 0.2, "elsewhere": 0.0, "not_visible": 0.0})
    assert math.isclose(sum(p.values()), 1.0) and w == 0.0 and math.isclose(p["back"], 0.6)
    p, w = C.r4_class_probs({"back": 0.0, "side": 0.0, "front": 0.0, "elsewhere": 0.3, "not_visible": 0.7})
    assert all(math.isclose(v, 1 / 3) for v in p.values()) and math.isclose(w, 1.0)
    p, w = C.r4_class_probs({"back": 0.3, "side": 0.1, "front": 0.1, "elsewhere": 0.0, "not_visible": 0.5})
    assert math.isclose(w, 0.5) and math.isclose(p["back"], 0.5 * 0.6 + 0.5 / 3)
    assert C.jev_r4_option({"back": 0.1, "side": 0.6, "front": 0.3}) == "side"


def test_look_r4_probs():
    cfg = C.CropCfg()
    p = C.look_r4_probs("side", True, cfg)
    assert p["flank"] == cfg.look_p_sure and math.isclose(sum(p.values()), 1.0)
    assert C.look_r4_probs("not_visible", True, cfg) is None and C.look_r4_probs("unclear", False, cfg) is None
    assert C.look_r4_probs("back", False, cfg)["back"] == cfg.look_p_lean


def test_auc():
    assert C.auc(np.array([0.9, 0.8, 0.1]), np.array([1, 1, 0], bool)) == 1.0
    assert C.auc(np.array([0.5, 0.5]), np.array([1, 0], bool)) == 0.5
    assert C.auc(np.array([0.5]), np.array([1], bool)) is None


def test_platt_recovers_a_shift_and_loo():
    rng = np.random.default_rng(0)
    z = rng.normal(0, 2.0, 3000)
    p_true = 1 / (1 + np.exp(-z))
    y = rng.random(3000) < p_true                            # labels drawn from the true probability
    p_jev = 1 / (1 + np.exp(-(z - 2.0) / 2.0))               # squashed and shifted toward "no": true = sigmoid(2 logit + 2)
    a, b = C.platt_fit(p_jev, y, prior_w=0.01)
    assert abs(a - 2.0) < 0.25 and abs(b - 2.0) < 0.3
    pc = C.platt_apply(p_jev, (a, b))
    assert np.abs(pc - np.clip(p_true, 0.02, 0.98)).max() < 0.06
    assert pc.min() >= 0.02 and pc.max() <= 0.98
    raw = np.mean((p_jev >= 0.5) == y)
    loo = C.loo_accuracy(p_jev[:60], y[:60])
    assert loo > raw - 0.05 and loo > 0.75
    assert C.loo_accuracy(p_jev[:2], y[:2]) is None


def _ans(back=0.0, side=0.0, front=0.0, els=0.0, nv=0.0, r9=0.5, r3=0.5):
    return {"R4": {"back": back, "side": side, "front": front, "elsewhere": els, "not_visible": nv}, "R9": r9, "R3": r3}


def test_calibrate_jev_report():
    labels = {"a": {"R4": "back", "R9": "yes", "R3": "yes"}, "b": {"R4": "side", "R9": "apart", "R3": "no"},
              "c": {"R4": "not_visible", "R9": "yes", "R3": "no"}, "d": {"R4": "unclear", "R9": "unclear", "R3": "yes"},
              "e": {"R4": "front", "R9": "yes", "R3": "yes"}}
    answers = {"a": _ans(back=0.7, side=0.1, front=0.1, nv=0.1, r9=0.7, r3=0.8), "b": _ans(back=0.6, side=0.2, front=0.2, r9=0.2, r3=0.1),
               "c": _ans(nv=0.9, back=0.1, r9=0.6, r3=0.3), "d": _ans(back=1.0, r3=0.9), "e": _ans(front=0.5, back=0.4, side=0.1, r3=0.4)}
    cal = C.calibrate_jev(labels, answers)
    assert cal["R4"]["n"] == 4 and math.isclose(cal["R4"]["accuracy_5opt"], 0.75)
    assert cal["R4"]["n_torso"] == 3 and math.isclose(cal["R4"]["accuracy_3class"], 2 / 3)
    assert cal["R4"]["confusion"]["side"] == {"back": 1}
    # R9: only "yes" in the relevant contrast ("apart" is not a negative of over / under) -> not calibrated, score 0
    assert cal["R9"]["n_no"] == 0 and cal["R9"]["n_apart"] == 1 and cal["R9"]["score"] == 0.0 and cal["R9"]["why"]
    assert math.isclose(cal["R9"]["accuracy_incl_apart"], 1.0)
    # R3: both answers present -> Platt + leave-one-out score
    assert cal["R3"]["n"] == 5 and cal["R3"]["platt"] is not None and cal["R3"]["score"] == cal["R3"]["accuracy_loo_platt"]


def test_decide_items_contract_and_mapping():
    index = [{"id": "x", "t": 30.5, "cam": "01", "set": "target", "questions": ["R4", "R9"]},
             {"id": "y", "t": 199.0, "cam": "05", "set": "target", "questions": ["R3"]},
             {"id": "z", "t": 50.0, "cam": "02", "set": "calib", "questions": ["R4", "R9", "R3"]},
             {"id": "w", "t": 34.0, "cam": "05", "set": "target", "questions": ["R4"]}]
    answers = {"x": _ans(back=0.5, side=0.3, front=0.2, r9=0.8), "y": _ans(r3=0.3), "z": _ans(back=1.0), "w": _ans(nv=1.0)}
    looks = {"x": {"R4": "side", "R4_sure": True, "R9": "yes", "R9_sure": False},
             "y": {"R3": "no", "R3_sure": True},
             "w": {"R4": "not_visible", "R4_p": {"front": 0.1, "flank": 0.45, "back": 0.45}}}
    calib = {"R4": {"score": 0.3}, "R9": {"score": 0.85, "platt": [1.0, 0.0]}, "R3": {"score": 0.9, "platt": [2.0, 1.0]}}
    items = C.decide_items(index, answers, looks, calib, {"x": -1})
    assert all({"t", "cam", "relation", "p", "score", "source"} <= set(i) for i in items)
    assert not [i for i in items if i["id"] == "z"]                                 # calibration crops never become evidence
    r4 = [i for i in items if i["relation"] == "R4"]
    assert {i["source"] for i in r4 if i["id"] == "x"} == {"jev", "agent"}
    assert all(set(i["p"]) == set(C.CLASSES) and math.isclose(sum(i["p"].values()), 1.0) for i in r4)
    assert [i for i in r4 if i["source"] == "jev"][0]["score"] == 0.3
    ag = [i for i in r4 if i["source"] == "agent" and i["id"] == "x"][0]
    assert ag["score"] == 1.0 and ag["p"]["flank"] == C.CropCfg().look_p_sure
    w = [i for i in r4 if i["id"] == "w" and i["source"] == "agent"][0]
    assert math.isclose(w["p"]["front"], 0.1)
    r9 = [i for i in items if i["relation"] == "R9_RL"]
    jev9 = [i for i in r9 if i["source"] == "jev"][0]
    assert math.isclose(jev9["p"]["+"], 1 - 0.8, abs_tol=1e-6)                     # map -1: "her on top" = the - side
    ag9 = [i for i in r9 if i["source"] == "agent"][0]
    assert math.isclose(ag9["p"]["+"], 1 - C.CropCfg().look_p_lean)
    r3 = [i for i in items if i["relation"] == "R3_follow" and i["source"] == "jev"][0]
    assert math.isclose(r3["p"]["+"], float(C.platt_apply(np.array([0.3]), (2.0, 1.0))[0]))
    # an undefined R9 map gives no R9 item
    assert not [i for i in C.decide_items(index, answers, looks, calib, {"x": 0}) if i["relation"] == "R9_RL"]


def test_items_through_the_decide_reader(tmp_path):
    """the sibling's reader (imported here at run time only; skipped if that module is not importable yet)"""
    try:
        from dancecap.sides import decide as D
        from dancecap.sides.config import SidesCfg
    except Exception as e:  # unfinished sibling module
        pytest.skip(f"sides.decide not importable: {e}")
    index = [{"id": "x", "t": 1.0, "cam": "01", "set": "target", "questions": ["R4", "R9"]}]
    looks = {"x": {"R4": "side", "R4_sure": True, "R9": "yes", "R9_sure": True}}
    items = C.decide_items(index, {"x": _ans(back=0.5, side=0.3, front=0.2, r9=0.5)}, looks, {"R4": {"score": 0.2}}, {"x": -1})
    p = tmp_path / "crops.json"
    p.write_text(json.dumps({"items": items}), encoding="utf-8")
    t = np.arange(90) / 30.0
    terms = D.crop_terms(D.load_crops(p), t, SidesCfg())
    assert terms["R4"][30].argmax() == list(D.CLASSES).index("flank")              # the agent look; Jev (score 0.2) ignored
    assert terms["R9_RL"][30] < 0                                                  # "her on top" with map -1


# ------------------------------------------------------------------------------------------------ Jev loop with a fake client

class _FakeClient:
    def __init__(self, fail=False):
        self.calls = 0
        self.fail = fail

    def system_one(self, state, questions, model=None, extra_headers=None, extra_body=None):
        self.calls += 1
        if self.fail:
            raise RuntimeError("boom")
        assert extra_headers.get("TypeSafe-Danger-Zone-Images") == "true"
        assert state["photo"] in extra_body["assets"]
        ans = {"R4": SimpleNamespace(type="choice", probabilities={"back": 0.1, "side": 0.6, "front": 0.1, "elsewhere": 0.1,
                                                                   "not_visible": 0.1}),
               "R9": SimpleNamespace(type="noul", noul=0.7), "R3": SimpleNamespace(type="noul", noul=0.2)}
        return SimpleNamespace(answers=ans, model="jev-test", usage=SimpleNamespace(input_tokens=123))


def _jpg(tmp_path, name="a.jpg"):
    cv2 = pytest.importorskip("cv2")
    img = np.full((320, 320, 3), 128, np.uint8)
    cv2.circle(img, (160, 160), 40, (0, 0, 255), -1)
    p = tmp_path / name
    cv2.imwrite(str(p), img)
    return p


def test_judge_caches_and_refuses_without_go_ahead(tmp_path):
    pytest.importorskip("typesafe_sdk")
    pytest.importorskip("PIL")
    _jpg(tmp_path)
    rows = [{"id": "a", "file": "a.jpg"}]
    with pytest.raises(PermissionError):
        C.judge(rows, tmp_path, allow_upload=False)
    cl = _FakeClient()
    r1 = C.judge(rows, tmp_path, allow_upload=True, client=cl)
    assert r1["sent"] == 1 and cl.calls == 1
    a = r1["answers"]["a"]
    assert a["R4"]["side"] == 0.6 and a["R9"] == 0.7 and a["R3"] == 0.2 and a["_tokens"] == 123
    r2 = C.judge(rows, tmp_path, allow_upload=True, client=cl)
    assert r2["cached"] == 1 and r2["sent"] == 0 and cl.calls == 1 and r2["answers"]["a"] == a
    # a failing request is reported, never an answer
    cv2 = pytest.importorskip("cv2")
    cv2.imwrite(str(tmp_path / "b.jpg"), np.full((300, 300, 3), 60, np.uint8))  # other content: no cache hit
    r3 = C.judge([{"id": "b", "file": "b.jpg"}], tmp_path, allow_upload=True, client=_FakeClient(fail=True))
    assert "b" in r3["errors"] and "b" not in r3["answers"]


def test_build_request_shape(tmp_path):
    pytest.importorskip("typesafe_sdk")
    pytest.importorskip("PIL")
    p = _jpg(tmp_path)
    req = C.build_request(p)
    assert set(req["questions"]) == {"R4", "R9", "R3"}
    assert req["state"]["photo"] in req["assets"] and req["assets"][req["state"]["photo"]]["type"] == "image"
    assert set(req["questions"]["R4"].criteria) == set(C.R4_OPTIONS)


# ------------------------------------------------------------------------------------------------ owner pick / sheets

def test_owner_pick_limits():
    index = [{"id": f"t{t}_c{c}", "t": t, "cam": c, "set": "target", "questions": ["R4"], "group": "closed_hold"}
             for t in (30.5, 32.5, 34.0, 44.0) for c in ("00", "01", "02", "05")]
    index.append({"id": "cal", "t": 1.0, "cam": "00", "set": "calib", "questions": ["R4"], "group": "calib_1"})
    looks = {r["id"]: {"R4": "unclear"} for r in index}
    pick = C.owner_pick(index, {}, looks, n=6)
    assert len(pick) == 6 and "cal" not in [r["id"] for r in pick]
    from collections import Counter
    assert max(Counter(r["t"] for r in pick).values()) <= 2 and max(Counter(r["cam"] for r in pick).values()) <= 2
    assert [r["t"] for r in pick] == sorted(r["t"] for r in pick)


def test_grid_and_label_tile():
    pytest.importorskip("cv2")
    tiles = [np.zeros((10, 20, 3), np.uint8) for _ in range(5)]
    g = C.grid(tiles, 2, pad=2)
    assert g.shape == (3 * 12 + 2, 2 * 22 + 2, 3)
    t = C.label_tile(np.zeros((60, 200, 3), np.uint8), ["A  film 0:30"])
    assert t.shape == (60, 200, 3) and t.max() > 200


def test_film_label():
    assert C.film_label(30.5) == "dance 30.50 s"
    assert C.film_label(250.0, lambda t: t + 7.2) == "film 4:17.20"
    assert C.film_label(250.0, lambda t: None) == "dance 250.00 s"
