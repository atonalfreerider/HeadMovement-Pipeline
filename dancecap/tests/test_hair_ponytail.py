"""Ponytail restyle of a hair groom (dancecap.hair_ponytail): pure parts, the take hook, and the viewer's C# defaults against the groom's."""
import json
import re
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from dancecap import hair_ponytail as hp

ROOT = Path(__file__).resolve().parents[2]
CS = ROOT / "head-movement" / "Assets" / "Hair" / "HairPonytail.cs"


def groom():
    return {"version": 2, "model": {"rest_height_m": 1.70}, "style": {"type": "sleek_straight", "part_x_unity": 0.025},
            "cards": {"layers": [{"name": "inner", "count": 24}], "cap": {"offset_m": 0.0015}},
            "colour": {"mid": {"linear_srgb": [0.17, 0.04, 0.02]}}, "render": {"albedo_gain": 1.85, "tip_emission": {"e": [0, 1, 2, 3, 4, 5]},
                                                                                "kajiya_kay": {}},
            "dynamics": {"substeps": 8}, "guides": [{"len": 0.5}]}


def test_apply_ponytail_style_cards_and_length():
    g0 = groom()
    snap = json.dumps(g0)
    g = hp.apply_ponytail(g0)
    assert json.dumps(g0) == snap                                  # the input is not modified
    assert g["style"]["type"] == "ponytail"
    p = g["style"]["ponytail"]
    assert p["tail_length_m"] == pytest.approx(0.22 * 1.70, abs=1e-4)
    assert "tail_frac_of_height" not in p
    names = {l["name"]: l["count"] for l in g["cards"]["layers"]}
    assert names["part"] == 0 and names["outer"] > 0              # no side-part cards; outer cards cover every guide
    assert g["cards"]["cap"] == {"offset_m": 0.0015}               # the cap block is kept
    assert g["guides"] == g0["guides"] and g["colour"] == g0["colour"]


def test_tail_length_scales_with_rest_height_and_params_merge():
    g = hp.apply_ponytail(groom(), rest_height_m=1.5, params={"flare": 3.0, "tie": {"clip": {"enabled": False}}})
    p = g["style"]["ponytail"]
    assert p["tail_length_m"] == pytest.approx(0.22 * 1.5, abs=1e-4)
    assert p["flare"] == 3.0
    assert p["tie"]["clip"]["enabled"] is False and p["tie"]["clip"]["az_deg"] == hp.PONYTAIL["tie"]["clip"]["az_deg"]
    assert p["tie"]["enabled"] is True


@pytest.mark.parametrize("bad", [{"tie_az_deg": 20}, {"tie_el_deg": 5}, {"tail_frac_of_height": 0.02}, {"flare": 9}, {"bundle_radius_m": [0.0, 0.02]}])
def test_validation_rejects(bad):
    with pytest.raises(ValueError):
        hp.apply_ponytail(groom(), params=bad)


def test_recolour_follows_the_grooms_rule():
    pal = {"mid": [114, 55, 33], "highlight": [203, 137, 103], "albedo_gain": 2.0, "cap_tint": [0.6, 0.7, 0.8], "tip_glow": 0.5,
           "copper_lift": [1.1, 1.2, 1.3]}
    g = hp.recolour(groom(), pal)
    mid = g["colour"]["mid"]["linear_srgb"]
    assert mid[0] == pytest.approx(0.168, abs=0.002) and mid[2] == pytest.approx(0.0152, abs=0.001)
    assert g["render"]["albedo_linear"] == [round(min(0.95, x * 2.0), 4) for x in mid]
    assert g["render"]["cap_tint"] == [0.6, 0.7, 0.8]
    assert g["render"]["tip_emission"]["e"] == [0, 0.5, 1.0, 1.5, 2.0, 2.5]
    assert g["render"]["copper_lift"] == [1.1, 1.2, 1.3]
    assert g["colour"]["mid"]["hex"] == "#723721"


def take_stub(tmp_path, style="ponytail", **own):
    return SimpleNamespace(cfg={"dancers": {"follow": {"hair_style": style, **own}}}, work=tmp_path)


def test_hook_is_a_noop_for_other_styles(tmp_path):
    g = groom()
    for style in ("", "loose"):
        assert hp.take_hooks(g, take_stub(tmp_path, style)) is g
    assert hp.take_hooks(g, SimpleNamespace(cfg={}, work=tmp_path)) is g


def test_hook_applies_ponytail_palette_and_toml_options(tmp_path):
    (tmp_path / "avatar").mkdir()
    (tmp_path / "avatar" / "hair_palette.json").write_text(json.dumps({"mid": [100, 50, 30], "highlight": [200, 130, 100]}))
    g = hp.take_hooks(groom(), take_stub(tmp_path, hair_tip_glow=0.0, hair_albedo_gain=1.0, hair_ponytail={"tail_frac_of_height": 0.3}))
    assert g["style"]["type"] == "ponytail" and g["style"]["ponytail"]["tail_length_m"] == pytest.approx(0.51, abs=1e-3)
    assert g["render"]["tip_emission"]["e"] == [0, 0, 0, 0, 0, 0]
    assert g["render"]["albedo_gain"] == 1.0


def test_shipped_takes_only_the_follow_of_john_lula_has_a_ponytail():
    styles = {}
    for t in (ROOT / "dancecap" / "takes").glob("*.toml"):
        cfg = tomllib.loads(t.read_text(encoding="utf-8"))
        styles[t.stem] = ((cfg.get("dancers") or {}).get("follow") or {}).get("hair_style")
    assert styles["john_lula"] == "ponytail"
    # chunk takes of the same couple copy the take file; every other couple keeps the loose style (the default)
    assert all(v in (None, "") for k, v in styles.items() if not k.startswith("john_lula"))


@pytest.mark.skipif(not CS.exists(), reason="viewer repository not checked out")
def test_viewer_defaults_match_the_groom_defaults():
    src = CS.read_text(encoding="utf-8")
    p = hp.PONYTAIL

    def cs(name):
        m = re.search(rf"\b{name} = (-?[0-9.]+)f", src)
        assert m, name
        return float(m.group(1))

    assert cs("TieAz") == p["tie_az_deg"] and cs("TieEl") == p["tie_el_deg"] and cs("TieLift") == p["tie_lift_m"]
    assert cs("TailJitter") == p["tail_length_jitter"] and cs("Droop0") == p["tail_droop0"] and cs("DownSlerp") == p["tail_down_slerp"]
    assert cs("Flare") == p["flare"] and cs("SkullHold") == p["skull_hold"] and cs("Deviation") == p["deviation"]
    assert cs("TieBandM") == p["tie"]["band_radius_m"] and cs("ClipAz") == p["tie"]["clip"]["az_deg"] and cs("ClipEl") == p["tie"]["clip"]["el_deg"]
    m = re.search(r"Bundle = new\(([0-9.]+)f, ([0-9.]+)f\)", src)
    assert m and [float(m.group(1)), float(m.group(2))] == p["bundle_radius_m"]
