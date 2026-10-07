"""dancecap.shoes: colour helpers, clustering and palette guards on synthetic samples, the foot frame on a synthetic
skin, export paths; plus a schema check of larissa_kadu's shoes.json when present. No GPU, no take data needed.

    .venv-nopo4d\\Scripts\\python.exe -m pytest tests/test_shoes.py -q
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dancecap import shoes as sh  # noqa: E402
from dancecap.export_unity import Skin  # noqa: E402
from dancecap.take import PACKAGE_ROOT  # noqa: E402

HEX = re.compile(r"^#[0-9a-f]{6}$")


def samples(rgb_srgb, n, seed=0, noise=0.01):
    rng = np.random.default_rng(seed)
    lin = sh.srgb_to_linear(np.asarray(rgb_srgb, float) / 255.0)
    return np.clip(lin + rng.normal(0, noise, (n, 3)) * lin.max(), 0, 1)


def test_hex_roundtrip():
    for h in ("#000000", "#ffffff", "#c8c8c0", "#202020"):
        assert sh.to_hex(sh.from_hex(h)) == h


def test_delta_e_and_luminance():
    assert sh.delta_e(sh.from_hex("#ffffff"), sh.from_hex("#ffffff")) == pytest.approx(0, abs=1e-9)
    assert sh.delta_e(sh.from_hex("#000000"), sh.from_hex("#ffffff")) == pytest.approx(100, abs=0.5)
    assert sh.luminance(sh.from_hex("#ffffff")) == pytest.approx(1.0, abs=1e-6)


def test_skin_like_flags_skin_not_white_or_black():
    assert sh.skin_like(sh.from_hex("#cc9080"))  # a skin / red-hair bleed tone
    assert not sh.skin_like(sh.from_hex("#e8e4dc"))
    assert not sh.skin_like(sh.from_hex("#202020"))


def test_clusters_split_two_colours_by_share():
    X = np.vstack([samples((30, 30, 30), 800), samples((235, 235, 230), 200, seed=1)])
    c = sh.clusters(X, 2)
    assert len(c) == 2 and c[0]["share"] == pytest.approx(0.8, abs=0.02)
    assert sh.luminance(c[0]["rgb"]) < 0.05 < 0.5 < sh.luminance(c[1]["rgb"])
    assert sh.clusters(X[:5]) == []


def test_palette_white_court_shoe_from_the_bake():
    bake = {"upper": samples((220, 215, 205), 3000), "sole": samples((210, 205, 195), 600, 1),
            "heel": np.vstack([samples((225, 220, 210), 700, 2), samples((40, 40, 40), 300, 3)])}
    col, src, conf = sh.palette(bake, None, None)
    # the warm bake white (#dcd7cd) balanced to a neutral white
    L, a, b = sh.lab(col["upper"])
    assert L >= 90.5 and np.hypot(a, b) <= 2.1 and "white-balanced" in src["upper"]
    assert src["sole"].startswith("bake")
    assert sh.luminance(col["heel_tab"]) < 0.1  # the dark heel tab cluster
    assert sh.auto_style(col) == "court_platform"
    assert set(col) == set(sh.FIELDS)


def test_palette_rejects_a_skin_like_bake_sole_and_uses_the_video():
    bake = {"upper": samples((30, 30, 30), 3000), "sole": samples((205, 150, 135), 400, 1)}
    video = {"upper": np.vstack([samples((28, 28, 28), 2700, 2), samples((225, 225, 220), 300, 3)]),
             "sole": np.vstack([samples((35, 35, 35), 600, 4), samples((240, 235, 225), 400, 5)])}
    col, src, conf = sh.palette(bake, video, None)
    assert src["sole"].startswith("video") and "bake rejected" in src["sole"]
    assert sh.luminance(col["sole"]) > 0.6  # the white midsole, not the dark floor / outsole pixels
    assert src["accent"].startswith("video upper 2nd cluster")  # the light side stripe
    assert sh.auto_style(col) == "runner"


def test_video_gain_maps_video_onto_the_bake_grade():
    bake = {"follow": {"upper": samples((220, 215, 205), 500)}, "lead": {"upper": samples((30, 30, 30), 500)}}
    video = {"follow": {"upper": samples((245, 235, 215), 500)}, "lead": {"upper": samples((25, 25, 25), 500)}}
    g, why = sh.video_gain(bake, video)
    assert "follow" in why
    out = sh.from_hex("#f5ebd7") * g
    assert sh.delta_e(out, sh.from_hex("#dcd7cd")) < 3


def synthetic_foot_skin() -> Skin:
    """A left 'foot' box in Unity rest coords: heel at z = 0, toes towards -z (the avatar faces -z), sole y = 0, body
    midline towards -x (medial). Landmark SMPL-X ids mapped through vertex_ids; everything weighted to the left
    ankle (joint 7) except the front third (left foot joint 10)."""
    xs, ys, zs = np.linspace(-0.04, 0.04, 9), np.linspace(0.0, 0.07, 8), np.linspace(0.0, -0.25, 26)
    V = np.array([[x, y, z] for z in zs for y in ys for x in xs])
    n = len(V)
    W = np.zeros((n, 55))
    front = V[:, 2] < -0.17
    W[~front, 7] = 1.0
    W[front, 10] = 1.0
    ids = np.arange(n) + 20000

    def at(p):
        return int(np.argmin(np.linalg.norm(V - np.asarray(p), axis=1)))

    ids[at([0.0, 0.0, 0.0])] = sh.LANDMARKS["L"]["heel"]
    ids[at([-0.03, 0.0, -0.25])] = sh.LANDMARKS["L"]["big"]
    ids[at([0.03, 0.0, -0.24])] = sh.LANDMARKS["L"]["small"]
    J = np.zeros((55, 3))
    J[7] = [0.0, 0.075, -0.06]
    J[10] = [0.0, 0.018, -0.19]
    return Skin(vertices=V, faces=np.zeros((0, 3), int), weights=W, joints=J, parents=np.full(55, -1), model="synthetic",
                betas=np.zeros(10), uv=None, normals=None, vertex_ids=ids)


def test_foot_frame_axes_and_measure():
    skin = synthetic_foot_skin()
    ff = sh.foot_frame(skin, "L")
    assert np.allclose(ff["fwd"], [0, 0, -1], atol=0.05)
    assert ff["side_ax"][0] < 0  # medial = towards the big toe = -x here
    m = sh.measure(skin, "L")
    assert m["length_m"] == pytest.approx(0.25, abs=0.005)
    assert m["ankle_height_m"] == pytest.approx(0.075, abs=1e-6)
    assert m["ball_from_heel_m"] == pytest.approx(0.19, abs=0.005)
    assert m["ball_width_m"] == pytest.approx(0.08, abs=0.005)
    b = sh.bands(skin, ff)
    assert b["upper"].any() and b["heel"].any()
    assert not (b["sole"] & b["upper"]).any()


def test_export_copies_into_the_takes_captures(tmp_path):
    class T:
        name = "tk"
        work = tmp_path / "work"
    (T.work / "avatar").mkdir(parents=True)
    (T.work / "avatar" / "shoes.json").write_text('{"format": "dancecap shoes", "version": 1, "dancers": {}}')
    streaming = tmp_path / "StreamingAssets"
    for name, take in (("01_A", "tk"), ("02_B", "other")):
        (streaming / name).mkdir(parents=True)
        (streaming / name / "capture.json").write_text(json.dumps({"version": 3, "provenance": {"take": take}}))
    out = sh.export(T, streaming=streaming)
    assert [p.parent.name for p in out] == ["01_A"]
    assert (streaming / "01_A" / "shoes.json").exists() and not (streaming / "02_B" / "shoes.json").exists()


def test_styles_never_add_sole_below_the_fit():
    # the outsole bottom is the fitted sole plane: styles only describe the stack above it
    for st in sh.STYLES.values():
        assert 0.015 <= st["sole_fore_m"] <= st["sole_heel_m"] <= 0.05
        assert 0 <= st["toe_spring_m"] <= 0.02


def test_larissa_kadu_shoes_json_schema():
    p = PACKAGE_ROOT / "work" / "larissa_kadu" / "avatar" / "shoes.json"
    if not p.exists():
        pytest.skip("no work/larissa_kadu/avatar/shoes.json")
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["format"] == "dancecap shoes" and d["version"] == 1 and "likeness" in d
    for role, x in d["dancers"].items():
        assert x["style"] in sh.STYLES
        for f in sh.FIELDS:
            assert HEX.match(x[f]), (role, f, x[f])
            assert 0 <= x["confidence"][f] <= 1
        for k in ("sole_heel_m", "sole_fore_m", "toe_spring_m", "heel_bevel_m"):
            assert 0 <= x[k] < 0.08


def test_neutral_whites_balances_warm_whites_only():
    from dancecap.shoes import from_hex, lab, neutral_whites, to_hex

    col = {"upper": from_hex("#dacab8"), "sole": from_hex("#1d1815")}
    src = {"upper": "bake", "sole": "bake"}
    changed = neutral_whites(col, src)
    assert changed == ["upper"]
    L, a, b = lab(col["upper"])
    assert L >= 90.5 and np.hypot(a, b) <= 2.1, to_hex(col["upper"])
    assert "white-balanced" in src["upper"] and src["sole"] == "bake"
    # a saturated light colour (a red sole) is not a white
    col2 = {"sole": from_hex("#e07060")}
    assert neutral_whites(col2, {}, ("sole",)) == []


def test_lab_round_trip():
    from dancecap.shoes import from_hex, lab, lab_to_linear

    for h in ("#dacab8", "#1d1815", "#e07060", "#ecebe6"):
        c = from_hex(h)
        assert np.allclose(lab_to_linear(*lab(c)), c, atol=2e-3)
