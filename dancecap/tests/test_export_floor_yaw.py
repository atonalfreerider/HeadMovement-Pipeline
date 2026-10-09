"""takes/<take>.toml [export] floor_yaw_deg -> capture.json floor_yaw_deg (the viewer's floor yaw, CCW seen from above).

    .venv-nopo4d/Scripts/python.exe -m pytest dancecap/tests/test_export_floor_yaw.py -q
"""
from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dancecap import export_unity as eu  # noqa: E402
from dancecap.take import PACKAGE_ROOT  # noqa: E402


def _toml(tmp_path, body: str) -> Path:
    p = tmp_path / "t.toml"
    p.write_text(body, encoding="utf-8")
    return p


def test_absent_key_means_no_yaw(tmp_path):
    assert eu.resolve_floor_yaw("x", _toml(tmp_path, '[export]\nsmplx_dir = "smplx"\n')) is None
    assert eu.resolve_floor_yaw("x", _toml(tmp_path, 'name = "x"\n')) is None
    assert eu.resolve_floor_yaw("no_such_take_toml", tmp_path / "missing.toml") is None


def test_value_is_read_as_float(tmp_path):
    assert eu.resolve_floor_yaw("x", _toml(tmp_path, "[export]\nfloor_yaw_deg = 30\n")) == 30.0
    assert eu.resolve_floor_yaw("x", _toml(tmp_path, "[export]\nfloor_yaw_deg = -12.5\n")) == -12.5


@pytest.mark.parametrize("bad", ['"30"', "true", "400.0", "[30]"])
def test_bad_values_are_refused(tmp_path, bad):
    with pytest.raises(ValueError):
        eu.resolve_floor_yaw("x", _toml(tmp_path, f"[export]\nfloor_yaw_deg = {bad}\n"))


@pytest.mark.parametrize("take", ["larissa_kadu_38", "larissa_kadu_full"])
def test_the_demo_takes_carry_the_30_deg_ccw(take):
    toml = PACKAGE_ROOT / "takes" / f"{take}.toml"
    if not toml.exists():
        pytest.skip("take TOML not in this checkout")
    cfg = tomllib.loads(toml.read_text(encoding="utf-8"))
    assert cfg["export"]["smplx_dir"]            # the key sits next to the motion choice, not in place of it
    assert eu.resolve_floor_yaw(take) == 30.0
