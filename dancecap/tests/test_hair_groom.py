"""Pure-function checks of dancecap.hair_groom (no SMPL-X model, no take data needed)."""
import json

import numpy as np
import pytest

from dancecap import hair_groom as hg


def test_fit_sphere_recovers_centre_and_radius():
    rng = np.random.default_rng(0)
    d = rng.normal(size=(500, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    c0 = np.array([0.1, 1.6, -0.05])
    c, r = hg.fit_sphere(c0 + 0.09 * d)
    assert np.allclose(c, c0, atol=1e-9) and r == pytest.approx(0.09, abs=1e-9)


def test_az_el_conventions():
    az, el = hg.az_el(np.array([[0, 0, 1.0], [1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]]))
    assert az[0] == pytest.approx(0) and az[1] == pytest.approx(90) and abs(az[2]) == pytest.approx(180)
    assert el[3] == pytest.approx(90) and el[0] == pytest.approx(0)


def test_hairline_is_lowest_at_the_nape():
    assert hg.hairline_el(0) > hg.hairline_el(90) > hg.hairline_el(180)
    assert hg.hairline_el(-180) == hg.hairline_el(180)


def test_strand_length_over_the_skull():
    r = 0.09
    # a crown root falls over a quarter circle (drop r, arc r pi/2) then hangs to the tip level
    L = hg.strand_length(1.75, 90.0, 1.20, r)
    assert L == pytest.approx(0.55 + r * (np.pi / 2 - 1))
    # roots below the head's equator hang straight
    assert hg.strand_length(1.55, -20.0, 1.20, r) == pytest.approx(0.35)
    # never shorter than 5 cm
    assert hg.strand_length(1.3, 10.0, 1.4, r) >= 0.05


def test_tip_level_blends_front_to_back():
    assert hg.tip_level(0.0, 1.34, 1.20) == pytest.approx(1.34)
    assert hg.tip_level(180.0, 1.34, 1.20) == pytest.approx(1.20)
    mid = hg.tip_level(95.0, 1.34, 1.20)
    assert 1.20 < mid < 1.34


def test_farthest_points_spread():
    rng = np.random.default_rng(1)
    P = rng.random((2000, 3))
    idx = hg.farthest_points(P, 50)
    assert len(set(idx.tolist())) == 50
    Q = P[idx]
    d = np.linalg.norm(Q[:, None] - Q[None], axis=-1) + np.eye(50)
    # far more even than random picks: minimum spacing well above that of a random subset
    R = P[rng.choice(2000, 50, replace=False)]
    dr = np.linalg.norm(R[:, None] - R[None], axis=-1) + np.eye(50)
    assert d.min() > 2 * dr.min()


def test_kabsch_angle():
    rng = np.random.default_rng(2)
    A = rng.normal(size=(40, 3))
    th = np.radians(30)
    Rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    assert hg.kabsch_angle_deg(A, A @ Rz.T + 0.3) == pytest.approx(30, abs=1e-6)


def test_peaks_are_separated():
    t = np.arange(100) / 30
    w = np.zeros(100)
    w[[10, 11, 50, 80]] = [300, 290, 200, 100]
    p = hg.peaks(t, w, 3, sep_s=0.5)
    assert [x["k"] for x in p] == [10, 50, 80]


def test_to_unity_flips_z():
    assert hg.to_unity([1.0, 2.0, 3.0]).tolist() == [1.0, 2.0, -3.0]


def test_export_copies_into_capture_folders(tmp_path, monkeypatch):
    class FakeTake:
        name = "faketake"
        work = tmp_path / "work"
    (FakeTake.work / "avatar").mkdir(parents=True)
    (FakeTake.work / "avatar" / "hair_groom.json").write_text(json.dumps({"version": 1}))
    streaming = tmp_path / "StreamingAssets"
    for name, take in (("01_A", "faketake"), ("02_B", "other")):
        (streaming / name).mkdir(parents=True)
        (streaming / name / "capture.json").write_text(json.dumps({"provenance": {"take": take}}))
    (FakeTake.work / "export" / "01_A").mkdir(parents=True)
    out = hg.export(FakeTake, streaming=streaming)
    assert sorted(p.parent.name for p in out) == ["01_A", "01_A"]
    assert (streaming / "01_A" / "hair_groom.json").exists() and not (streaming / "02_B" / "hair_groom.json").exists()
    with pytest.raises(FileNotFoundError):
        hg.export(FakeTake, capture="03_missing", streaming=streaming)


def test_fit_head_capsule_recovers_a_front_back_capsule():
    rng = np.random.default_rng(3)
    c0, h0, r0 = np.array([0.0, 1.66, -0.03]), 0.015, 0.077
    # upper half of a z-capsule surface (a scalp): cylinder part + hemispherical end caps
    n = 3000
    ang = rng.uniform(0, np.pi, n)
    cyl = c0 + np.c_[r0 * np.cos(ang), r0 * np.sin(ang), rng.uniform(-h0, h0, n)]
    d = rng.normal(size=(n, 3))
    d[:, 1] = np.abs(d[:, 1])
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    end = np.where(d[:, 2:3] >= 0, h0, -h0)
    caps = c0 + np.c_[np.zeros(n), np.zeros(n), end[:, 0]] + r0 * d
    c, h, r, res = hg.fit_head_capsule(np.r_[cyl, caps], c0 + [0.004, -0.003, 0.01], 0.09)
    assert r == pytest.approx(r0, abs=0.002) and h == pytest.approx(h0, abs=0.004) and res < 0.002
    assert np.linalg.norm(c - c0) < 0.005


def test_apply_style_upgrades_a_v1_groom_without_the_rest_model():
    g1 = {"version": 1, "guides": [{"len": 0.5}], "shape": {"standoff_m": 0.033, "part_x_unity": 0.0},
          "colour": {"mid": {"linear_srgb": [0.2, 0.03, 0.02]}, "highlight": {"linear_srgb": [0.6, 0.1, 0.08]}},
          "render": {"kajiya_kay": {"primary_shift": -0.06}}, "dynamics": {"targets": {"turn_spread_r95_m": 0.25}},
          "strands": {"children": 3}, "colliders": [{"name": "head"}]}
    g2 = hg.apply_style(g1, ref={"colour": {"portrait": {"palette": {"mid": {"linear_srgb": [0.12, 0.006, 0.014]}}}}})
    assert g2["version"] == hg.VERSION == 2 and g1["version"] == 1  # input untouched
    assert g2["guides"] == g1["guides"] and g2["colliders"] == g1["colliders"]
    assert g2["shape"]["standoff_m"] == hg.SLEEK_STANDOFF_M < 0.02
    assert g2["style"]["type"] == "sleek_straight" and g2["style"]["sweep"] == -1
    assert sum(layer["count"] for layer in g2["cards"]["layers"]) == 136
    assert set(g2["face_box"]) >= {"fwd_min_r", "y_lo_r", "y_hi_r", "half_width_r"}
    # albedo = observed mid x gain at the same hue; portrait option; same-hue highlight for the TRT lobe
    a = np.array(g2["render"]["albedo_linear"])
    assert np.allclose(a, np.array([0.2, 0.03, 0.02]) * hg.RENDER_V2["albedo_gain"], atol=1e-3)
    assert "albedo_portrait_linear" in g2["render"]
    assert g2["render"]["kajiya_kay"]["version"] == 2 and g2["render"]["kajiya_kay"]["highlight_linear"][0] > 0.6
    assert g2["render"]["tip_emission"]["e"][-1] == pytest.approx(3.1)
    assert g2["dynamics"]["targets"] == {"turn_spread_r95_m": 0.25}
    assert g2["dynamics"]["max_deviation_front"] < g2["dynamics"]["max_deviation_back"]


def test_albedo_from_is_clipped():
    assert hg.albedo_from([0.6, 0.1, 0.05], 2.0) == [0.95, 0.2, 0.1]


def _sphere_head(n_az=72, n_el=36, r=0.09):
    """UV sphere around the origin in Unity head-bone axes (face toward -z), uv = (az, el) mapped to 0..1."""
    az = np.radians(np.linspace(-180, 180, n_az + 1))
    el = np.radians(np.linspace(-80, 85, n_el + 1))
    A, E = np.meshgrid(az, el)
    V = np.c_[(r * np.cos(E) * np.sin(A)).ravel(), (r * np.sin(E)).ravel(), (-r * np.cos(E) * np.cos(A)).ravel()]
    UV = np.c_[((A + np.pi) / (2 * np.pi)).ravel(), ((E + np.radians(80)) / np.radians(165)).ravel()]
    F = []
    w = n_az + 1
    for i in range(n_el):
        for j in range(n_az):
            a, b, c, d = i * w + j, i * w + j + 1, (i + 1) * w + j, (i + 1) * w + j + 1
            F += [[a, b, d], [a, d, c]]
    W = np.zeros((len(V), 55))
    W[:, 15] = 1.0
    return V, W, np.array(F), UV


def test_painted_hairline_finds_the_texture_edge():
    V, W, F, UV = _sphere_head()
    h, w = 256, 512
    img = np.zeros((h, w, 3), np.float32)
    v = 1 - (np.arange(h) + 0.5) / h  # PNG row 0 = v 1
    el_row = -80 + v * 165
    img[el_row < 30] = [0.80, 0.55, 0.42]  # skin below 30 deg
    img[el_row >= 30] = [0.20, 0.06, 0.04]  # painted hair above
    table = hg.painted_hairline(V, W, F, UV, np.zeros(3), [0.0, 0.0, 0.0], img, margin_deg=2.0)
    front = [e for a, e in table if a <= 65]
    assert len(front) == 7 and all(26.0 <= e <= 30.0 for e in front), table  # the edge (30) minus the margin, 2-deg steps
    # beyond 65 deg the model's root hairline is kept (ears, sides, back: covered by the hanging hair)
    assert [a for a, _ in table if a > 70] == [a for a, _ in hg.HAIRLINE_EL if a > 70]
    # never above the root hairline: a texture with no hair keeps the model's line
    bald = np.full((h, w, 3), [0.80, 0.55, 0.42], np.float32)
    t2 = hg.painted_hairline(V, W, F, UV, np.zeros(3), [0.0, 0.0, 0.0], bald)
    assert all(e <= float(np.interp(a, *np.array(hg.HAIRLINE_EL, float).T)) + 0.05 for a, e in t2)  # rounded to 0.1


def test_apply_style_review_fixes():
    g1 = {"version": 1, "guides": [{"len": 0.5}], "shape": {}, "colour": {"mid": {"linear_srgb": [0.2, 0.03, 0.02]},
                                                                           "highlight": {"linear_srgb": [0.6, 0.1, 0.08]}},
          "render": {}, "dynamics": {}, "strands": {}, "colliders": []}
    ref = {"colour": {"portrait": {"palette": {"mid": {"linear_srgb": [0.12, 0.006, 0.014]},
                                               "highlight": {"linear_srgb": [0.23, 0.022, 0.039]}}}}}
    cap = [[5.0, 28.0], [180.0, -34.0]]
    g2 = hg.apply_style(g1, ref=ref, cap_hairline=cap)
    assert g2["cards"]["cap"]["hairline_el_deg_vs_abs_az"] == cap and "hairline_el_deg_vs_abs_az" not in hg.CARDS["cap"]
    assert g2["cards"]["cap"]["ao"][0] > g2["cards"]["cap"]["ao"][1] >= 0.6  # lighter at the hairline, never near-black
    assert g2["render"]["albedo_default"] == "portrait"
    hp = g2["render"]["kajiya_kay"]["highlight_portrait_linear"]
    assert max(hp) == pytest.approx(0.63, abs=0.01) and hp[0] > hp[2] > hp[1]  # the portrait's crimson hue, video brightness
    tg = g2["render"]["tip_glow"]
    assert tg["anchor_to_strands"] and tg["core_px"] >= 1.5 and 0 < tg["halo"] < 1 / 3.1 + 1e-9  # halo below the bloom threshold
    fr = g2["dynamics"]["front_right"]
    assert fr["shape_falloff"] > g2["dynamics"]["shape_falloff"] and fr["max_deviation"] < g2["dynamics"]["max_deviation_front"]
    assert fr["side"]["abs_az_max"] > fr["abs_az_max"]
    # no portrait measured: the video stays the default
    g3 = hg.apply_style(g1)
    assert g3["render"]["albedo_default"] == "video" and "hairline_el_deg_vs_abs_az" not in g3["cards"]["cap"]
