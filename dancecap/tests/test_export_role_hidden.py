"""takes/<take>.toml [export] role_hidden -> capture.json role_hidden (capture-clock spans a role is hidden in, with a fade-in).

    .venv-nopo4d/Scripts/python.exe -m pytest dancecap/tests/test_export_role_hidden.py -q
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


def test_absent_key_means_nobody_is_hidden(tmp_path):
    assert eu.resolve_role_hidden("x", _toml(tmp_path, '[export]\nsmplx_dir = "smplx"\n')) is None
    assert eu.resolve_role_hidden("x", _toml(tmp_path, 'name = "x"\n')) is None
    assert eu.resolve_role_hidden("no_such_take_toml", tmp_path / "missing.toml") is None


def test_spans_are_read_as_capture_second_tables(tmp_path):
    body = "[export]\nrole_hidden = { lead = [[0, 4]], follow = [[10.5, 12], [20, 21.25, 0.5, 0.25]] }\n"
    got = eu.resolve_role_hidden("x", _toml(tmp_path, body))
    assert got == {"lead": [{"from": 0.0, "to": 4.0, "fade_in": 0.0}],
                   "follow": [{"from": 10.5, "to": 12.0, "fade_in": 0.0},
                              {"from": 20.0, "to": 21.25, "fade_in": 0.5, "fade_out": 0.25}]}


def test_span_tables_with_a_fade_in(tmp_path):
    body = "[export]\nrole_hidden = { lead = [ { from = 0.0, to = 4.0, fade_in = 1.0 } ] }\n"
    assert eu.resolve_role_hidden("x", _toml(tmp_path, body)) == {"lead": [{"from": 0.0, "to": 4.0, "fade_in": 1.0}]}


@pytest.mark.parametrize("bad", [
    '"lead"', "[1, 2]", "{ lead = [] }", "{ lead = [4.0] }", "{ lead = [[4.0, 4.0]] }", "{ lead = [[5, 4]] }",
    "{ lead = [[0, 1, 2, 3, 4]] }", '{ lead = [["a", 1]] }', "{ lead = [[true, 2]] }", "{ lead = 3 }", "{ coach = [[0, 1]] }",
    "{ lead = [[0, 4, -1]] }", "{ lead = [ { from = 0.0 } ] }", "{ lead = [ { from = 0.0, to = 4.0, speed = 2 } ] }",
])
def test_bad_values_are_refused(tmp_path, bad):
    with pytest.raises(ValueError):
        eu.resolve_role_hidden("x", _toml(tmp_path, f"[export]\nrole_hidden = {bad}\n"))


def test_the_full_take_hides_the_lead_for_four_seconds_then_fades_him_in():
    toml = PACKAGE_ROOT / "takes" / "larissa_kadu_full.toml"
    if not toml.exists():
        pytest.skip("take TOML not in this checkout")
    cfg = tomllib.loads(toml.read_text(encoding="utf-8"))
    assert cfg["export"]["smplx_dir"] and cfg["export"]["floor_yaw_deg"] == 30.0   # next to the motion choice and the floor yaw
    spans = eu.resolve_role_hidden("larissa_kadu_full")
    # capture seconds (from the first frame = the viewer HUD clock): hidden [0, 4), fade-in 1.0 s starting at 4.0 -> fully there at 5.0
    assert spans == {"lead": [{"from": 0.0, "to": 4.0, "fade_in": 1.0}]}
