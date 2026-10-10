"""takes/<take>.toml [dancers.<role>.props.<kind>] -> capture.json props (head-worn props: the teacher's headset microphone).

    .venv-nopo4d/Scripts/python.exe -m pytest dancecap/tests/test_export_props.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dancecap import export_unity as eu  # noqa: E402
from dancecap.take import PACKAGE_ROOT  # noqa: E402


def _toml(tmp_path, body: str) -> Path:
    p = tmp_path / "t.toml"
    p.write_text(body, encoding="utf-8")
    return p


def test_no_props_block_means_no_props(tmp_path):
    assert eu.resolve_props("x", _toml(tmp_path, '[dancers.follow]\nname = "A"\n')) is None
    assert eu.resolve_props("x", _toml(tmp_path, 'name = "x"\n')) is None
    assert eu.resolve_props("no_such_take_toml", tmp_path / "missing.toml") is None


def test_an_empty_headset_block_gets_every_default(tmp_path):
    got = eu.resolve_props("x", _toml(tmp_path, "[dancers.follow.props.headset]\n"))
    assert got == [{"role": "follow", "kind": "headset", **eu.HEADSET_DEFAULTS}]
    assert got[0]["side"] == "left" and got[0]["hook"] is True


def test_values_override_defaults_and_are_normalised(tmp_path):
    body = ('[dancers.lead.props.headset]\nside = "right"\ncolour = "#FFAA00"\nboom_length_m = 0.11\n'
            "boom_path = [[0, 0, 0], [0.01, -0.02, 0.05], [0.02, -0.05, 0.09]]\n"
            "[dancers.follow.props.headset]\nstandoff_m = 0.01\nhook = false\n")
    lead, follow = eu.resolve_props("x", _toml(tmp_path, body))
    assert (lead["role"], follow["role"]) == ("lead", "follow")                  # role order, never names
    assert lead["side"] == "right" and lead["colour"] == "#ffaa00" and lead["boom_length_m"] == 0.11
    assert lead["boom_path"][1] == [0.01, -0.02, 0.05]
    assert follow["standoff_m"] == 0.01 and follow["hook"] is False and follow["side"] == "left"
    assert "boom_length_m" not in follow and "boom_path" not in follow            # optional keys only when given


def test_enabled_false_drops_the_prop(tmp_path):
    assert eu.resolve_props("x", _toml(tmp_path, "[dancers.follow.props.headset]\nenabled = false\n")) is None


@pytest.mark.parametrize("bad", [
    'side = "front"', "side = 1", 'colour = "red"', 'boom_colour = "#12345"', "boom_thickness_m = 0", "boom_thickness_m = 0.5",
    'capsule_radius_m = "big"', "hook = 1", "enabled = 1", "boom_length_m = 0.001", "boom_path = [[0, 0, 0]]",
    "boom_path = [[0, 0], [1, 1]]", "boom_path = [[0, 0, 0], [0, 0, 9]]", "speed = 3", "opacity_boost = 2",
])
def test_bad_values_are_refused(tmp_path, bad):
    with pytest.raises(ValueError):
        eu.resolve_props("x", _toml(tmp_path, f"[dancers.follow.props.headset]\n{bad}\n"))


def test_unknown_kind_or_role_is_refused(tmp_path):
    with pytest.raises(ValueError):
        eu.resolve_props("x", _toml(tmp_path, "[dancers.follow.props.hat]\n"))
    with pytest.raises(ValueError):
        eu.resolve_props("x", _toml(tmp_path, "[dancers.follow]\nprops = 3\n"))
    # a role that is not lead | follow is simply not a dancer block: nothing is read for it
    assert eu.resolve_props("x", _toml(tmp_path, "[dancers.coach.props.headset]\n")) is None


def test_the_defaults_are_the_viewers(tmp_path):
    """HeadPropSpec.cs carries the same numbers (VIEWER_SPEC 3.2b): a drift between the two would show as a different headset."""
    spec = (PACKAGE_ROOT.parent / "head-movement" / "Assets" / "Avatar" / "HeadPropSpec.cs")
    if not spec.exists():
        pytest.skip("Unity project not in this checkout")
    text = spec.read_text(encoding="utf-8")
    for field, key in (("BoomThickness", "boom_thickness_m"), ("Standoff", "standoff_m"), ("Bow", "bow_m"),
                       ("CapsuleLength", "capsule_length_m"), ("CapsuleRadius", "capsule_radius_m"),
                       ("HookThickness", "hook_thickness_m"), ("OpacityBoost", "opacity_boost")):
        assert f"public float {field} = {eu.HEADSET_DEFAULTS[key]}f;" in text, (field, eu.HEADSET_DEFAULTS[key])
    for field, key in (("CapsuleColour", "colour"), ("BoomColour", "boom_colour"), ("HookColour", "hook_colour")):
        assert f'ParseColour("{eu.HEADSET_DEFAULTS[key]}"' in text, field


def test_the_recap_take_gives_her_the_headset_on_her_left_and_him_nothing():
    toml = PACKAGE_ROOT / "takes" / "larissa_kadu_recap.toml"
    if not toml.exists():
        pytest.skip("take TOML not in this checkout")
    props = eu.resolve_props("larissa_kadu_recap")
    assert props and [p["role"] for p in props] == ["follow"]                    # owner 2026-10-09: her headset, nothing on him
    assert props[0]["kind"] == "headset" and props[0]["side"] == "left"
    assert props[0]["colour"] == "#0e0d0d"
