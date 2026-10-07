"""Hair reference (dancecap/hair_reference.py): the geometry / colour / bloom-recipe helpers on synthetic data, plus
contract checks on the original head-movement tip bloom and on larissa_kadu's hair_reference.json when present.
No GPU needed.

    .venv-nopo4d\\Scripts\\python.exe -m pytest tests/test_hair_reference.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dancecap import hair_reference as hr  # noqa: E402
from dancecap.take import PACKAGE_ROOT, WORKSPACE  # noqa: E402

ORIGINAL_CS = """
    public float skullRadius = 0.07f;
    public float cardioidScale = 0.16f;
    public int topHairCount = 30;
    public float domeMaxAngle = 60f;
    public int sideLayerCount = 5;
    public int sideStrandsPerLayer = 30;
    public float ringAngleSpan = 60f;
    public int segmentsPerStrand = 6;
    public float strandThickness = 0.01f;
    public float strandEndThickness = 0.001f;
        Color darkGrey = new(0.1f, 0.1f, 0.1f);
            hairColorGradient.SetKeys(
                new[]
                {
                    new GradientColorKey(Color.black, 0.0f),
                    new GradientColorKey(darkGrey, 0.5f),
                    new GradientColorKey(Color.white, 0.95f),
                    new GradientColorKey(new Color(4, 3.2f, 1.5f), 1.0f)
                },
                alphas
            );
            float length   = Mathf.Max(0.2f, cardioidScale * (1f - Mathf.Cos(alphaVal)));
"""

VOLUME_YAML = """--- !u!114 &1
MonoBehaviour:
  m_Name: Bloom
  m_EditorClassIdentifier:
  active: 1
  threshold:
    m_OverrideState: 1
    m_Value: 1
  intensity:
    m_OverrideState: 1
    m_Value: 1
  scatter:
    m_OverrideState: 1
    m_Value: 0.7
--- !u!114 &2
MonoBehaviour:
  m_Name: Tonemapping
  active: 1
  mode:
    m_OverrideState: 1
    m_Value: 0
"""


def test_parse_original_hair_script():
    h = hr.parse_hair_cs(ORIGINAL_CS)
    assert h["segmentsPerStrand"] == 6 and h["topHairCount"] == 30 and h["sideLayerCount"] == 5
    assert h["strandThickness"] == pytest.approx(0.01) and h["strandEndThickness"] == pytest.approx(0.001)
    assert h["min_length_m"] == pytest.approx(0.2)
    keys = h["gradient_keys"]
    assert [k["t"] for k in keys] == [0.0, 0.5, 0.95, 1.0]
    assert keys[1]["rgb_gamma"] == pytest.approx([0.1, 0.1, 0.1])
    assert keys[3]["rgb_gamma"] == pytest.approx([4.0, 3.2, 1.5])  # authored HDR key (clamped later by Unity)


def test_baked_vertex_colours_model_matches_editor_measurement():
    keys = hr.parse_hair_cs(ORIGINAL_CS)["gradient_keys"]
    vc = hr.baked_vertex_colours(keys, 6)  # the model alone (gamma -> linear, clamp, UNorm8)
    # the model reproduces the editor measurement within one 8-bit step (18 vs 19, 115 vs 114) and clamps the HDR key
    assert np.abs(np.round(vc[:, 0] * 255) - np.array(hr.MEASURED_BAKED_8BIT)).max() <= 1
    assert vc.max() == 1.0 and vc[-1, 0] == 1.0
    # the measured values are used only for the measured strand
    assert hr.measured_baked(keys, 6)[:, 0] * 255 == pytest.approx(hr.MEASURED_BAKED_8BIT)
    assert hr.measured_baked(keys, 16) is None
    vc16 = hr.baked_vertex_colours(keys, 16)
    assert np.all(np.diff(vc16[:, 0]) >= 0) and vc16.max() == 1.0
    assert np.allclose(vc16 * 255, np.round(vc16 * 255))


def test_bloom_share_soft_knee():
    assert hr.bloom_share(0.5) == pytest.approx(0.0, abs=1e-6)
    assert hr.bloom_share(1.0) == pytest.approx(0.125, abs=1e-3)   # knee 0.5: 12.5 % of a 1.0 pixel blooms
    assert hr.bloom_share(2.0) == pytest.approx(0.5, abs=1e-6)
    assert hr.bloom_share(hr.LIT_HAIR_MAX_LINEAR) < 0.01           # the lit-hair ceiling keeps it under 1 %


def test_emission_profile_threshold_crossing():
    keys = hr.parse_hair_cs(ORIGINAL_CS)["gradient_keys"]
    rec = {"vertex_colours_linear_baked": hr.measured_baked(keys, 6).tolist(), "shader": {"multiplier": 3.1}}
    s, e = hr.emission_profile(rec, 1001)
    assert e[-1] == pytest.approx(3.1)
    first = s[np.argmax(e > 1.0)]
    # 0.6 + 0.2 * (1 - 18/255*3.1) / ((115 - 18)/255*3.1)
    expect = 0.6 + 0.2 * (1 - 18 / 255 * 3.1) / ((115 - 18) / 255 * 3.1)
    assert first == pytest.approx(expect, abs=2e-3)
    assert e[s < 0.6].max() < 0.25  # roots and mid-lengths never bloom


def test_yaml_block_value():
    assert hr._yaml_block_value(VOLUME_YAML, "Bloom", "threshold") == "1"
    assert hr._yaml_block_value(VOLUME_YAML, "Bloom", "scatter") == "0.7"
    assert hr._yaml_block_value(VOLUME_YAML, "Tonemapping", "mode") == "0"
    assert hr._yaml_block_value(VOLUME_YAML, "Vignette", "intensity") is None


def test_tip_bloom_recipe_fallback_without_repo(tmp_path):
    rec = hr.tip_bloom_recipe(tmp_path)  # no repo: documented fallback values
    assert rec["source"]["parsed_from_repo"] is False
    assert rec["strands"]["count"] == 180
    assert rec["emission_along_strand"]["tip_value"] == pytest.approx(3.1)
    assert rec["vertex_colours_8bit"]["max_step_difference"] <= 1
    assert rec["emission_along_strand"]["soft_knee"]["starts_at_linear"] == pytest.approx(0.5)


@pytest.mark.skipif(not (hr.HM_REPO / "Assets" / "bloom-shader.shadergraph").exists(), reason="head-movement repo absent")
def test_tip_bloom_recipe_from_repo():
    rec = hr.tip_bloom_recipe()
    assert rec["shader"]["multiplier"] == pytest.approx(3.1, abs=1e-4)
    assert rec["shader"]["unlit"] and rec["shader"]["vertex_colour_input"]
    assert rec["shader"]["material_shader_guid_matches"]
    b = rec["bloom_volume"]
    assert float(b["threshold"]) == 1 and float(b["intensity"]) == 1 and float(b["scatter"]) == pytest.approx(0.7)
    assert rec["pipeline"]["hdr"] is True and rec["pipeline"]["asset"] == "Bloom.asset"
    if rec["source"]["parsed_from_repo"]:
        assert rec["strands"]["points_per_strand"] == 6


def test_colour_helpers():
    white = np.array([0.6, 0.5, 0.4])
    wb = hr.white_balance(white, white, 0.8)
    assert wb == pytest.approx([0.8, 0.8, 0.8])
    e = hr.colour_entry([1.0, 0.0, 0.0])
    assert e["hex"] == "#FF0000" and e["hsv"][0] == pytest.approx(0.0, abs=1.5)
    assert hr.delta_e76([0.2, 0.2, 0.2], [0.2, 0.2, 0.2]) == 0.0
    x = np.linspace(0, 1, 11)
    assert hr.linear_to_srgb(hr.srgb_to_linear(x)) == pytest.approx(x, abs=1e-9)
    # CIELAB of D65 white is (100, 0, 0)
    assert hr.linear_to_lab(np.ones(3)) == pytest.approx([100, 0, 0], abs=0.2)


def test_best_shift_recovers_translation():
    a = np.zeros((120, 80), bool)
    a[30:90, 20:50] = True
    a[20:30, 28:42] = True  # head, so the shape is not translation-ambiguous
    b = np.roll(np.roll(a, 4, axis=0), -3, axis=1)
    (dx, dy), iou, iou0 = hr.best_shift(a, b, 8)
    assert (dx, dy) == (-3, 4) and iou == pytest.approx(1.0) and iou0 < 1.0


def test_phase_lag_trailing_hair():
    dt = 1 / 30
    t = np.arange(0, 3, dt)
    phi_head = 2 * np.pi * 0.8 * t + 0.4 * np.sin(3 * t)  # a turning head
    lag = 0.1
    phi_hair = np.interp(t - lag, t, phi_head) + np.pi  # hair mass straight behind, 0.1 s late
    tau, rms = hr.phase_lag(phi_head, phi_hair, dt)
    assert tau == pytest.approx(lag, abs=dt / 2 + 1e-9) and rms < 5


def test_back_envelope_length_bridges_hollows():
    # a straight back (z = -0.1) with a hollow (lumbar curve) between y = 1.0 and 1.2: hair bridges it
    ys = np.linspace(0.9, 1.6, 141)
    z = np.where((ys > 1.0) & (ys < 1.2), -0.06, -0.1)
    V = np.stack([np.zeros_like(ys), ys, z], 1)
    L = hr.back_envelope_length(V, 1.6, 0.9, offset_m=0.0)
    assert L == pytest.approx(0.7, abs=0.02)


def test_describe_level():
    lm = {"crown": 1.75, "neck_joint": 1.5, "shoulder_joints": 1.44, "shoulder_blade_lower_tip_approx": 1.30,
          "mid_back_approx": 1.18, "waist_min_perimeter": 1.06, "pelvis_joint": 0.95}
    assert "shoulder blades" in hr.describe_level(1.29, lm)
    assert "waist" in hr.describe_level(1.07, lm)
    assert hr.describe_level(2.0, lm) == "above the crown"


def test_rodrigues_and_chain():
    R = hr.rodrigues(np.array([[0, np.pi / 2, 0]]))[0]
    assert R @ np.array([0, 0, 1.0]) == pytest.approx([1, 0, 0], abs=1e-9)
    T = 3
    rots = hr.chain_rotations(np.zeros((T, 3)), np.zeros((T, 63)))
    assert set(rots) == set(hr.CHAIN_HEAD) and np.allclose(rots[15], np.eye(3))
    assert hr.azimuth(np.array([1.0, 0, 0])) == pytest.approx(np.pi / 2)


def test_mediapipe_square_crop():
    from dancecap.hair_segment_mp import square_crop
    img = np.arange(10 * 20 * 3, dtype=np.uint8).reshape(10, 20, 3)
    sq, box, pad = square_crop(img, (-5, 2, 12, 8))
    assert box == (0, 2, 12, 8) and sq.shape == (12, 12, 3) and pad == (0, 3)
    assert np.array_equal(sq[3:9, 0:12], img[2:8, 0:12])


def _pf(n_calm=60, n_turn_each=12, offset_deg=30.0, trail_deg=0.0, noise_deg=6.0, seed=1):
    """Synthetic per-frame motion data: calm frames + turns both ways at 220 deg/s; the hair mass sits offset_deg off
    straight behind (over a shoulder) and lags each turn by trail_deg."""
    rng = np.random.default_rng(seed)
    n = n_calm + 2 * n_turn_each
    yaw = np.r_[rng.uniform(-40, 40, n_calm), np.full(n_turn_each, 220.0), np.full(n_turn_each, -220.0)]
    phi = rng.uniform(-np.pi, np.pi, n)
    lag = np.where(np.abs(yaw) > 180, -np.sign(yaw) * np.radians(trail_deg), 0.0)
    psi = hr.wrap(phi + np.pi + np.radians(offset_deg) + lag + np.radians(rng.normal(0, noise_deg, n)))
    calm = np.abs(yaw) < 90
    turn = np.abs(yaw) > 180
    return {"k": np.arange(n), "t": np.arange(n) / 30, "n": np.full(n, 4000.0), "r95": np.full(n, 0.2),
            "psi": psi, "phi": phi, "phi_torso": phi, "yaw": yaw, "yaw_torso": yaw, "centroid_r": np.full(n, 0.06),
            "calm": calm, "turn": turn}, calm, turn


def test_trailing_angle_calm_offset_is_not_trailing():
    pf, calm, turn = _pf(offset_deg=30.0, trail_deg=0.0)
    m = hr.motion_metrics(pf, calm, turn, hr.HairRefConfig(bootstrap=500), 1.0)
    tr = m["trailing"]
    assert tr["head"]["calm_offset_deg"] == pytest.approx(30, abs=4)
    assert abs(tr["head"]["corrected_median"]) < 6
    assert tr["status"] == "not_resolved" and m["lag_time_s"]["status"] == "not_resolved"
    # the version-1 keys stay for consumers (hair_groom) but carry no target
    assert m["trailing_angle_deg_turns"]["median"] is None and m["lag_time_s"]["phase_alignment"] is None
    # without the calm-offset correction the turns would read as +-30 deg "trailing", opposite per direction
    d = tr["head"]["by_turn_direction"]
    assert np.sign(d["positive_yaw"]["raw_median"]) != np.sign(d["negative_yaw"]["raw_median"])
    assert "her right" in tr["calm_hair_mass_side"]  # positive offset -> toward -x = her right (left is +x)


def test_trailing_angle_real_trailing_resolves():
    pf, calm, turn = _pf(offset_deg=30.0, trail_deg=35.0)
    tr = hr.motion_metrics(pf, calm, turn, hr.HairRefConfig(bootstrap=500), 1.0)["trailing"]
    assert tr["status"] == "resolved"
    assert tr["head"]["corrected_median"] == pytest.approx(35, abs=6)
    assert all(all(c.values()) for c in tr["checks"].values())


def test_colour_step_auto_and_lut():
    cfg = hr.HairRefConfig()
    assert hr.colour_step_for(cfg, 176) == 6 and hr.colour_step_for(cfg, 1153) == 39
    assert hr.colour_step_for(hr.HairRefConfig(colour_step=2), 1153) == 2
    px = np.array([[0, 128, 255], [10, 20, 30]], np.uint8)
    assert hr.bgr8_to_linear_rgb(px) == pytest.approx(hr.srgb_to_linear(px[:, ::-1] / 255.0), abs=1e-6)


def test_splat_disks_matches_circles():
    import cv2
    q = np.array([[10, 12], [30, 5], [31, 6], [-3, 4], [50, 50]])
    inside = (q[:, 0] >= 0) & (q[:, 0] < 45) & (q[:, 1] >= 0) & (q[:, 1] < 40)
    ref = np.zeros((40, 45), np.uint8)
    for x, y in q[inside]:
        cv2.circle(ref, (int(x), int(y)), 3, 1, -1)
    got = hr.splat_disks(q, (40, 45), 3)
    # exactly the disks cv2.circle draws, for points inside the image (points outside are dropped, not drawn partially)
    assert np.array_equal(got, ref.astype(bool))


def _samples(seed=0):
    rng = np.random.default_rng(seed)
    S = []
    for cam in ("01", "02", "03"):
        for i in range(3):
            n = 400
            bgr = np.clip(rng.normal([40, 50, 140], [10, 10, 30], (n, 3)), 0, 255).astype(np.uint8)
            S.append({"cam": cam, "i": i, "t": i / 30, "white": np.array([0.6, 0.6, 0.6]), "bgr8": bgr,
                      "rgb": hr.bgr8_to_linear_rgb(bgr), "zone": rng.integers(-1, 3, n).astype(np.int8),
                      "d": rng.uniform(0, 0.5, n).astype(np.float32), "L": 0.5, "n_top": 100,
                      "mp": {"iou": 0.6, "hair_mask_in_mp": 0.8} if i else None})
    return S


def test_palette_image_stats_and_spread():
    pal = hr.palette_from_samples(_samples(), 0.8)
    ist = pal["image_stats"]
    assert set(ist) == {"p10", "p50", "p90"}
    Y = [hr.luminance(ist[n]["linear_srgb"]) for n in ("p10", "p50", "p90")]
    assert Y[0] < Y[1] < Y[2]
    assert "img_p10" not in pal["overall"] and "NOT an albedo" in pal["what"]
    sp = pal["camera_spread"]
    assert sp["mid_L_sd"] is not None and set(sp["mid_camera_dE76_to_consensus"]) == {"01", "02", "03"}
    tol = hr.colour_tolerances(pal)
    assert 10 <= tol["lightness_L"] <= 14 and tol["hue_deg"] == 10


def test_colour_samples_cache_roundtrip(tmp_path):
    S = _samples()
    p = tmp_path / "colour_samples.npz"
    hr.save_colour_samples(p, "k1", {"samples": S, "mp": {"status": "ok"}, "n_views": len(S)})
    assert hr.load_colour_samples(p, "other") is None
    got = hr.load_colour_samples(p, "k1")
    assert len(got["samples"]) == len(S) and got["mp"]["cached"]
    for a, b in zip(S, got["samples"]):
        assert a["cam"] == b["cam"] and a["i"] == b["i"] and np.array_equal(a["bgr8"], b["bgr8"])
        assert np.array_equal(a["zone"], b["zone"]) and (a["mp"] is None) == (b["mp"] is None)
        assert np.allclose(a["rgb"], b["rgb"])


def test_exclusion_summary():
    cfg = hr.HairRefConfig()
    hs = [{"k": k, "recall": 0.5, "n_carve": 8, "views": [{"cam": "01", "carve_only": "x", "iou": 0.3},
                                                         {"cam": "02", "iou": 0.7}]} for k in (139, 140, 141)]
    hs.append({"k": 23, "recall": 0.79, "n_carve": 8, "views": [{"cam": "01", "iou": 0.66}]})
    ex = [hr._exclusion_reason(h, h["k"] / 30, cfg) for h in hs]
    s = hr.summarise_exclusions(ex)
    assert "139-141" in s and "camera 01" in s and "23" in s and "recall just below" in s


def test_face_python_override(monkeypatch, tmp_path):
    monkeypatch.setenv("DANCECAP_FACE_PYTHON", sys.executable)
    monkeypatch.setenv("DANCECAP_FACE_PYTHONPATH", str(tmp_path))
    exe, env = hr.face_python()
    assert Path(exe) == Path(sys.executable) and env["PYTHONPATH"].startswith(str(tmp_path))


REF = PACKAGE_ROOT / "work" / "larissa_kadu" / "avatar" / "hair_reference.json"
TAKE_TOML = PACKAGE_ROOT / "takes" / "larissa_kadu.toml"


@pytest.mark.skipif(not (PACKAGE_ROOT / "work" / "larissa_kadu" / "smplx" / "follow.npz").exists()
                    or not TAKE_TOML.exists(), reason="larissa_kadu fit absent")
def test_pod_job_spec_parses_and_uploads_little(tmp_path):
    from dancecap.remote_jobs import parse_job
    from dancecap.take import Take
    take = Take.load("larissa_kadu")
    spec, warn = hr.pod_job_spec(take, hr.HairRefConfig())
    job = parse_job(spec, "test")
    assert job.name == "hair_ref" and "texture" in job.needs
    ins = [x for x, _ in job.inputs]
    assert not any("/cameras/cache" in x or x.endswith("/cameras/") or x.endswith("/frames/") for x in ins)
    frames = [x for x in ins if "/frames/" in x and x.endswith(".jpg")]
    assert 0 < len(frames) <= 8 * 30 + 8  # the colour views only
    assert any(x.endswith("/dancers/01/hair/") for x in ins) and any(x.endswith("smplx/follow.npz") for x in ins)
    assert "/workspace/mp_site" in spec["env"]["DANCECAP_FACE_PYTHONPATH"]
    p = hr.write_pod_job(take, hr.HairRefConfig(), out=tmp_path / "pod_job.json")
    assert json.loads(p.read_text())["schema"] == "dancecap.pod_job/1"




@pytest.mark.skipif(not REF.exists(), reason="larissa_kadu hair_reference.json not built")
def test_larissa_kadu_reference_contract():
    r = json.loads(REF.read_text())
    assert r["version"] == 2 and r["take"] == "larissa_kadu"
    L = r["shape"]["length"]
    lm = r["shape"]["landmarks_rest_m_above_sole_world"]
    tip = L["tip_level_back_rest_m"]["median"]
    assert lm["pelvis_joint"] < tip < lm["shoulder_joints"]  # long hair: tips between the shoulders and the hips
    assert 0.3 < L["strand_length_crown_to_tip_back"]["true_m"] < 0.9
    assert r["hull_consistency"]["recall_median"] > 0.85
    pal = r["colour"]["video"]["overall"]
    for z in ("root", "mid", "tip"):
        h = pal[z]["hsv"][0]
        assert h < 30 or h > 330, f"{z} hue {h} is not red"
    tb = r["tip_bloom"]
    assert tb["emission_along_strand"]["tip_value"] == pytest.approx(3.1, abs=1e-3)
    acc = r["acceptance"]
    assert set(acc) >= {"length", "colour", "tip_bloom"}
    # motion: a trailing angle only becomes a target when it is resolved
    tr = r["shape"]["motion"]["trailing"]
    assert tr["status"] in ("resolved", "not_resolved")
    assert ("trailing_angle_deg" in acc) == (tr["status"] == "resolved")
    if tr["status"] != "resolved":
        assert r["shape"]["motion"]["trailing_angle_deg_turns"]["median"] is None
    # colour: image statistics with tolerances from the measured spread, not an albedo
    assert set(acc["colour"]["image_stats"]) == {"p10", "p50", "p90"}
    assert 10 <= acc["colour"]["tolerance"]["lightness_L"] <= 14
    assert "albedo" in r["colour"]["what"]
    assert acc["tip_bloom"]["lit_hair_max_linear"] <= 0.7
