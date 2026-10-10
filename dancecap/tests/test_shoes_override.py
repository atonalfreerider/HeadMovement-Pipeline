"""avatar/shoes_override.json (owner decisions: whose sneakers are which colour) is applied by every shoes export.

    .venv-nopo4d/Scripts/python.exe -m pytest dancecap/tests/test_shoes_override.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dancecap import shoes  # noqa: E402


def _shoes() -> dict:
    return {"dancers": {"lead": {"style": "runner", "upper": "#1d1815", "sole": "#e9e4e3", "source": {"sole": "video"}, "confidence": {"sole": 0.6}},
                        "follow": {"style": "court_platform", "upper": "#e8e5e1", "sole_heel_m": 0.04}}}


def test_colours_are_set_normalised_and_marked_as_decisions():
    s = _shoes()
    changed = shoes.apply_override(s, {"lead": {"sole": "#221E1C", "upper": "#1d1815"}, "follow": {"upper": "#F3F1ED"}})
    assert sorted(changed) == ["follow.upper", "lead.sole"]                       # an unchanged value is not reported
    assert s["dancers"]["lead"]["sole"] == "#221e1c" and s["dancers"]["follow"]["upper"] == "#f3f1ed"
    assert "override" in s["dancers"]["lead"]["source"]["sole"] and s["dancers"]["lead"]["confidence"]["sole"] == 1.0


def test_style_and_style_numbers_and_unknown_roles():
    s = _shoes()
    shoes.apply_override(s, {"follow": {"style": "neutral", "sole_heel_m": 0.03}, "coach": {"upper": "#000000"}, "_note": "text"})
    assert s["dancers"]["follow"]["style"] == "neutral" and s["dancers"]["follow"]["sole_heel_m"] == 0.03
    assert "coach" not in s["dancers"]


@pytest.mark.parametrize("bad", [{"upper": "white"}, {"upper": "#fff"}, {"upper": 3}, {"style": "heels"}, {"colour": "#ffffff"}])
def test_bad_overrides_are_refused(bad):
    with pytest.raises(ValueError):
        shoes.apply_override(_shoes(), {"lead": bad})


def test_the_recap_decisions_are_black_and_white_when_the_work_dir_is_here():
    p = Path(__file__).resolve().parents[1] / "work" / "larissa_kadu_recap" / "avatar" / "shoes_override.json"
    if not p.exists():
        pytest.skip("work dir (git-ignored) not in this checkout")
    import json
    ov = json.loads(p.read_text(encoding="utf-8"))

    def lum(h):
        r, g, b = (int(h[i:i + 2], 16) for i in (1, 3, 5))
        return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255

    assert all(lum(ov["lead"][f]) < 0.2 for f in ("upper", "sole", "sole_heel", "outsole", "lace"))     # Kadu: black sneakers
    assert all(lum(ov["follow"][f]) > 0.85 for f in ("upper", "sole", "sole_heel", "lace"))            # Larissa: white sneakers
