"""R10: the 16:9 render of the recap film: opt-in `horizontal` camera overrides and presets (synthetic data only).

    .venv-nopo4d/Scripts/python.exe -m pytest dancecap/tests/test_film_recap_horizontal.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dancecap import film_recap as fr  # noqa: E402
from dancecap import film_recap_director as fd  # noqa: E402
from dancecap import film_recap_plan as plan  # noqa: E402

CFG = plan.Cfg()


def test_physics_overhead_and_feet_cameras_carry_horizontal_overrides_and_nothing_else_changes():
    for family in ("weight", "axis", "path", "feet", "talk", "body", "timing"):
        base = plan.camera_for(family, variant=0, cfg=CFG)
        out = plan.with_horizontal(base, CFG)
        h = out.pop("horizontal", None)
        assert out == base                        # the 9:16 keys are untouched
        assert (h is not None) == (family in ("weight", "axis", "path", "feet", "talk", "body", "timing")), family
    phys = plan.with_horizontal(fd.physics_camera(0, 0.7), CFG)
    assert phys["horizontal"]["band_lo"] < phys["band_lo"] and phys["horizontal"]["band_hi"] > phys["band_hi"]   # a taller band in the wide frame
    over = plan.with_horizontal(plan.camera_for("axis", variant=0, cfg=CFG), CFG)["horizontal"]
    assert over["parallax"] is True and over["fill"] > 0.5 and over["never"] > 0.9 and over["follow_s"] < 0.8


def test_the_overrides_can_be_switched_off_and_are_never_applied_twice():
    cam = plan.camera_for("weight", variant=1, cfg=CFG)
    off = plan.Cfg(horizontal_overrides=False)
    assert "horizontal" not in plan.with_horizontal(cam, off)
    once = plan.with_horizontal(cam, CFG)
    assert plan.with_horizontal(once, CFG) == once


def test_phone_shots_are_not_given_overrides():
    cam = {"mode": "camera_tour", "cameras": [{"camera": "00", "dance_t": [1.0, 4.0]}], "crop": {"t0": 1.0, "dt": 0.1, "cx": [0.0], "cy": [0.0], "zoom": [1.5]}}
    assert plan.with_horizontal(cam, CFG) == cam


def test_presets_carry_the_wide_frame_switches_only_for_horizontal():
    p = fr.presets_for(CFG)
    assert p["vertical"] == fr.PRESETS["vertical"]                      # the 9:16 preset is byte-identical
    h = p["horizontal"]
    assert 0 < h["phone_side_fill"] <= 1 and h["safe"]["side_frac"] == 0.05 and h["portrait_zoom"] > 1 and h["legend_scale"] > 1
    assert "phone_side_fill" not in fr.PRESETS["horizontal"]              # the module constant is not mutated
    off = fr.presets_for(plan.Cfg(horizontal_overrides=False))
    assert off == {k: v for k, v in fr.PRESETS.items()}
