"""Rest-shape edits (dancecap.shape_edits) on the plain SMPL-X neutral body, the shoes' toml geometry, and the demo
export's hair-collider patch.

    .venv-nopo4d\\Scripts\\python.exe -m pytest dancecap/tests/test_shape_edits.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dancecap import shape_edits as se  # noqa: E402
from dancecap.export_unity import DEFAULT_MODEL, shaped_skin  # noqa: E402

pytestmark = pytest.mark.skipif(not DEFAULT_MODEL.exists(), reason="SMPL-X model files not installed")


@pytest.fixture(scope="module")
def body():
    return shaped_skin(DEFAULT_MODEL, np.zeros(10))


def test_midsection_scales_the_plateau_and_leaves_the_rest(body, tmp_path):
    v, J, W, F = body.vertices, body.joints, body.weights, body.faces
    out, rep = se.apply_vertices({"midsection_scale": 0.75}, v, J, W, F, "follow", tmp_path)
    m = rep["midsection"]
    assert m["ratio_after_vs_own"] == pytest.approx([0.75, 0.75], abs=0.004)
    b = se.band(J)
    outside = (v[:, 1] < b[0] - 1e-6) | (v[:, 1] > b[3] + 1e-6)
    assert np.abs(out[outside] - v[outside]).max() < 1e-9          # hips / chest / legs / head untouched
    assert np.abs(out[:, 1] - v[:, 1]).max() < 1e-12               # heights never change
    arms = se.arm_weight(W) > 0.999
    assert np.abs(out[arms] - v[arms]).max() < 1e-9


def test_midsection_is_smooth_over_height(body, tmp_path):
    v, J, W, F = body.vertices, body.joints, body.weights, body.faces
    out, _ = se.apply_vertices({"midsection_scale": [0.75, 0.8]}, v, J, W, F, "follow", tmp_path)
    torso = se.arm_weight(W) < 0.5
    b = se.band(J)
    ys = np.arange(b[0] - 0.02, b[3] + 0.02, 0.005)
    ratio = np.array([np.ptp(se.section_points(out, F, torso, y)[:, 0]) / np.ptp(se.section_points(v, F, torso, y)[:, 0])
                      for y in ys])
    assert ratio.min() == pytest.approx(0.75, abs=0.005) and ratio.max() == pytest.approx(1.0, abs=0.005)
    assert np.abs(np.diff(ratio)).max() < 0.03                   # 5 mm steps: no kink


def test_depth_pivot_zero_keeps_the_back(body, tmp_path):
    v, J, W, F = body.vertices, body.joints, body.weights, body.faces
    out, _ = se.apply_vertices({"midsection_scale": 0.75, "midsection_depth_pivot": 0.0}, v, J, W, F, "follow",
                               tmp_path)
    torso = se.arm_weight(W) < 0.5
    b = se.band(J)
    y = 0.5 * (b[1] + b[2])
    P0, P1 = se.section_points(v, F, torso, y), se.section_points(out, F, torso, y)
    assert P1[:, 2].min() == pytest.approx(P0[:, 2].min(), abs=0.002)   # back (min z: SMPL-X faces +z) stays
    assert np.ptp(P1[:, 2]) == pytest.approx(0.75 * np.ptp(P0[:, 2]), abs=0.002)


def test_feet_shrink_about_the_ankle_with_the_sole_on_the_floor(body, tmp_path):
    v, J, W, F = body.vertices, body.joints, body.weights, body.faces
    L0 = se.foot_lengths(v, J, W)
    out, rep = se.apply_vertices({"foot_scale": [0.9, 0.9, 0.8]}, v, J, W, F, "lead", tmp_path)
    L1 = se.foot_lengths(out, J, W)
    for s in ("L", "R"):
        assert L1[s] == pytest.approx(0.9 * L0[s], abs=0.0015)
        ff0, ff1 = se._foot_frame(v, J, W, s), se._foot_frame(out, J, W, s)
        assert ff1["y0"] == pytest.approx(ff0["y0"], abs=1e-9)   # sole stays on the floor
    y0 = v[:, 1].min()
    above = v[:, 1] > max(J[7, 1], J[8, 1]) + 0.0351          # the leg above the taper (ankle + 3.5 cm) is untouched
    assert np.abs(out[above] - v[above]).max() < 1e-9
    near = W[:, [4, 5]].sum(1) > 0.5                          # knee-skinned shin vertices move at most a few mm
    assert np.abs(out[near] - v[near]).max() < 0.004 and out[:, 1].min() == pytest.approx(y0, abs=1e-9)
    assert rep["feet"]["length_scale"] == 0.9


def test_shoe_length_target_sets_the_foot_scale(body, tmp_path):
    v, J, W, F = body.vertices, body.joints, body.weights, body.faces
    L0 = se.foot_lengths(v, J, W)
    Lm = 0.5 * (L0["L"] + L0["R"])
    allow = se.SNEAKER_ALLOWANCE_M["neutral"]
    _, rep = se.apply_vertices({"shoe_length_m": 0.95 * Lm + allow}, v, J, W, F, "lead", tmp_path)
    assert rep["feet"]["length_scale"] == pytest.approx(0.95, abs=1e-4)
    _, rep = se.apply_vertices({"shoe_length_m": 1.2 * Lm + allow}, v, J, W, F, "lead", tmp_path)
    assert rep["feet"]["length_scale"] == 1.0                    # never lengthened


def test_bad_configs_are_refused(body, tmp_path):
    v, J, W, F = body.vertices, body.joints, body.weights, body.faces
    with pytest.raises(ValueError):
        se.apply_vertices({"midsection_scale": 0.3}, v, J, W, F, "follow", tmp_path)
    with pytest.raises(ValueError):
        se.apply_vertices({"midsection_scale": 0.75, "midsection_depth_pivot": 1.5}, v, J, W, F, "follow", tmp_path)
    with pytest.raises(ValueError):
        se.apply_vertices({"foot_scale": [0.9, 0.9]}, v, J, W, F, "lead", tmp_path)


def test_config_for_reads_the_role_tables():
    cfg = {"dancers": {"follow": {"name": "x", "shape_edits": {"midsection_scale": 0.75}}, "lead": {"name": "y"}}}
    assert se.config_for("none", cfg) == {"follow": {"midsection_scale": 0.75}}


def test_shoes_toml_geometry_overrides_and_validates():
    from dancecap import shoes
    cfg = {"dancers": {"follow": {"shoe_geometry": {"sole_heel_m": 0.03, "sole_fore_m": 0.025}}}}
    s = {"dancers": {"follow": {"sole_heel_m": 0.04, "sole_fore_m": 0.035, "source": {}}, "lead": {"sole_heel_m": 0.032}}}
    changed = shoes.apply_geometry(s, cfg)
    assert sorted(changed) == ["follow.sole_fore_m", "follow.sole_heel_m"]
    assert s["dancers"]["follow"]["sole_heel_m"] == 0.03 and s["dancers"]["lead"]["sole_heel_m"] == 0.032
    assert "geometry" in s["dancers"]["follow"]["source"]
    with pytest.raises(ValueError):
        shoes.toml_geometry({"dancers": {"follow": {"shoe_geometry": {"sole_heel_m": 0.2}}}}, "follow")
    with pytest.raises(ValueError):
        shoes.toml_geometry({"dancers": {"follow": {"shoe_geometry": {"upper": "#ffffff"}}}}, "follow")


def test_demo_export_patches_hair_colliders_once(tmp_path):
    from dancecap.demo_export import patch_hair_colliders
    p = tmp_path / "hair_groom.json"
    p.write_text(json.dumps({"colliders": [{"name": "midback", "r": 0.0932}, {"name": "head", "r": 0.0766}]}))
    assert patch_hair_colliders([p], {"midback": -0.0233}) == ["midback"]
    g = json.loads(p.read_text())
    assert g["colliders"][0]["r"] == pytest.approx(0.0699) and g["colliders"][1]["r"] == 0.0766
    patch_hair_colliders([p], {"midback": -0.0233})           # idempotent
    assert json.loads(p.read_text())["colliders"][0]["r"] == pytest.approx(0.0699)
