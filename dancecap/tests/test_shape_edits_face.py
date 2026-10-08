"""Lower-face rest-shape edit (dancecap.shape_edits [dancers.<role>.shape_edits.face]) on the plain SMPL-X neutral body.

    .venv-nopo4d\\Scripts\\python.exe -m pytest dancecap/tests/test_shape_edits_face.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dancecap import shape_edits as se  # noqa: E402
from dancecap.export_unity import DEFAULT_MODEL, shaped_skin, textured_skin  # noqa: E402

pytestmark = pytest.mark.skipif(not DEFAULT_MODEL.exists(), reason="SMPL-X model files not installed")

TARGET = {"lower_face_height_scale": 0.94, "chin_height_scale": 0.90, "chin_width_scale": 0.92,
          "lower_face_depth_scale": 0.97, "submental_height_scale": 0.65}
STRONG = {"lower_face_height_scale": 0.85, "chin_height_scale": 0.80, "chin_width_scale": 0.80, "jaw_width_scale": 0.9,
          "cheek_scale": 0.9, "lower_face_depth_scale": 0.9, "submental_height_scale": 0.4}


@pytest.fixture(scope="module")
def body():
    return shaped_skin(DEFAULT_MODEL, np.zeros(10))


def _edit(body, face, tmp_path, v=None):
    v = body.vertices if v is None else v
    return se.apply_vertices({"face": face}, v, body.joints, body.weights, body.faces, "follow", tmp_path)


def test_absent_or_neutral_face_is_a_no_op(body, tmp_path):
    out, rep = se.apply_vertices({"foot_scale": [1.0, 1.0, 1.0]}, body.vertices, body.joints, body.weights, body.faces,
                                 "follow", tmp_path)
    assert "face" not in rep
    out1, rep1 = _edit(body, {k: se.FACE_DEFAULTS.get(k, 1.0) for k in se.FACE_KEYS}, tmp_path)
    assert np.abs(out1 - body.vertices).max() < 1e-12 and rep1["face"]["moved_vertices"] == 0
    out2, rep2 = _edit(body, {}, tmp_path)                          # empty table = nothing configured
    assert "face" not in rep2 and np.abs(out2 - body.vertices).max() == 0.0


def test_target_shortens_the_lower_face_and_narrows_the_chin(body, tmp_path):
    out, rep = _edit(body, TARGET, tmp_path)
    own, aft = rep["face"]["own"], rep["face"]["after"]
    fr = se.face_frame(body.vertices)
    # the menton rises by the integrated compression (lower face + chin windows), up to the ramp edges
    expect = (1 - TARGET["lower_face_height_scale"]) * (fr["y_sn"] - fr["y_me"]) + \
        (1 - TARGET["chin_height_scale"]) * (fr["y_li"] - fr["y_me"])
    assert own["face_height_n_me_mm"] - aft["face_height_n_me_mm"] == pytest.approx(1000 * expect, rel=0.08)
    assert aft["upper_lip_sn_sto_mm"] / own["upper_lip_sn_sto_mm"] == pytest.approx(0.94, abs=0.015)
    assert aft["chin_c7_9_mm"] < 0.95 * own["chin_c7_9_mm"]
    assert aft["chin_c6_10_mm"] < own["chin_c6_10_mm"]
    assert aft["jaw_angle_c4_12_mm"] == pytest.approx(own["jaw_angle_c4_12_mm"], rel=0.01)   # jaw angles stay
    assert aft["cheek_c1_15_mm"] == pytest.approx(own["cheek_c1_15_mm"], abs=1e-6)
    assert aft["outer_canthal_mm"] == pytest.approx(own["outer_canthal_mm"], abs=1e-6)
    assert aft["chin_projection_vs_jaw_angle_mm"] < own["chin_projection_vs_jaw_angle_mm"]


@pytest.mark.parametrize("face", [TARGET, STRONG])
def test_eyes_nose_cranium_scalp_ears_untouched(body, tmp_path, face):
    out, rep = _edit(body, face, tmp_path)
    mv = np.linalg.norm(out - body.vertices, axis=1)
    prot = se.face_protected(body.vertices, body.weights)
    assert mv[prot["above_cheekbones"]].max() == 0.0            # eyes, brows, nose bridge, forehead, cranium, scalp
    assert mv[prot["behind_front_mask"]].max() == 0.0           # ears, nape, back of the head (hair roots)
    eyes = body.weights[:, [23, 24]].sum(1) > 0.5
    assert mv[eyes].max() == 0.0
    if face.get("cheek_scale", 1.0) == 1.0:
        assert mv[prot["nose"]].max() == 0.0                    # the nose too unless the cheeks are narrowed
    else:
        assert mv[prot["nose"]].max() < 0.002
    assert rep["face"]["protected_max_move_mm"]["above_cheekbones"] == 0.0


def test_far_regions_untouched(body, tmp_path):
    out, _ = _edit(body, STRONG, tmp_path)
    mv = np.linalg.norm(out - body.vertices, axis=1)
    fr = se.face_frame(body.vertices)
    fade = se.face_neck_fade(fr, se.face_factors({"face": STRONG}))
    assert fade > se.FACE_NECK_FADE_M                           # the large lift lengthens the neck fade
    low = body.vertices[:, 1] < fr["y_me"] - se.FACE_THROAT_M - fade        # chest, arms, hands, legs
    assert mv[low].max() == 0.0
    assert mv[se.arm_weight(body.weights) > 0.999].max() == 0.0


def test_uvs_weights_joints_unchanged(body, tmp_path):
    from dataclasses import replace
    edited, rep = se.apply(body, {"face": TARGET}, "follow", tmp_path)
    assert edited.joints is body.joints and edited.weights is body.weights and edited.faces is body.faces
    a, b = textured_skin(body, DEFAULT_MODEL), textured_skin(replace(body, vertices=edited.vertices), DEFAULT_MODEL)
    if a.uv is None:
        pytest.skip("model without UVs")
    assert np.array_equal(a.uv, b.uv) and np.array_equal(a.faces, b.faces) and np.array_equal(a.vertex_ids, b.vertex_ids)
    assert np.array_equal(a.weights, b.weights)
    s = se.summary(rep)
    assert s["face"]["factors"]["chin_width_scale"] == 0.92 and len(s["face"]["face_height_n_me_mm"]) == 2


@pytest.mark.parametrize("face, max_strain, min_dot", [(TARGET, 0.5, 0.95), (STRONG, 1.0, 0.7)])
def test_smooth_no_folds(body, tmp_path, face, max_strain, min_dot):
    out, _ = _edit(body, face, tmp_path)
    v0, F = body.vertices, body.faces

    def nrm(v):
        n = np.cross(v[F[:, 1]] - v[F[:, 0]], v[F[:, 2]] - v[F[:, 0]])
        return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-15)
    moved = (np.linalg.norm(out - v0, axis=1) > 1e-9)[F].any(1)
    dots = (nrm(v0) * nrm(out)).sum(1)[moved]
    assert dots.min() > min_dot                                 # no triangle flips / creases (STRONG is extreme)
    # displacement gradient along edges: bounded strain everywhere
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    d = out - v0
    strain = np.linalg.norm(d[e[:, 0]] - d[e[:, 1]], axis=1) / np.linalg.norm(v0[e[:, 0]] - v0[e[:, 1]], axis=1)
    assert strain.max() < max_strain                            # (TARGET ~0.35: the lifted under-jaw / neck)


def test_lift_profile_is_c1_and_monotonic(body):
    fr = se.face_frame(body.vertices)
    y = np.arange(fr["y_me"] - 0.08, fr["y_sn"] + 0.02, 2e-4)
    D = se.face_lift(y, fr, 0.9, 0.85, 0.6)
    assert D[-1] == pytest.approx(0.0, abs=1e-12)               # nothing above the subnasale
    assert np.all(np.diff(D) <= 1e-12)                          # more lift further down
    slope = np.diff(D) / np.diff(y)                             # = - compression rate: continuous (C1 positions)
    assert np.abs(np.diff(slope)).max() < 0.02
    assert -slope.min() <= (0.1 + 0.15 + 0.4) + 1e-6
    w = se.face_weights(y, fr)
    for k, prof in w.items():
        assert np.abs(np.diff(prof)).max() < 0.03, k            # smooth: every ramp spans > 1 cm (0.2 mm steps)


def test_translation_invariant(body, tmp_path):
    shift = np.array([0.1, 1.2, -0.3])                          # e.g. hair_groom.RestModel puts the sole at y = 0
    a, _ = _edit(body, TARGET, tmp_path)
    b, _ = _edit(body, TARGET, tmp_path, v=body.vertices + shift)
    assert np.abs((b - shift) - a).max() < 1e-6


def test_rejects_bad_config(body, tmp_path):
    with pytest.raises(ValueError, match="unknown keys"):
        _edit(body, {"jaw_scale": 0.9}, tmp_path)
    with pytest.raises(ValueError, match="allowed range"):
        _edit(body, {"chin_width_scale": 0.5}, tmp_path)
    with pytest.raises(ValueError, match="allowed range"):
        _edit(body, {"submental_height_scale": 0.3}, tmp_path)


def test_landmark_table_matches_flame():
    """FACE_LMK = FLAME 2020's static iBUG-68 embedding through SMPL-X__FLAME_vertex_ids (when installed)."""
    try:
        from dancecap.face_smirk import SMIRK_DIR, SX_FLAME_IDS, load_flame
        LM = np.load(SMIRK_DIR / "assets" / "landmark_embedding.npy", allow_pickle=True, encoding="latin1")[()]
        ids = np.load(SX_FLAME_IDS).astype(np.int64)
        f = load_flame()["f"].astype(np.int64)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"FLAME assets not installed ({e})")
    lf = np.asarray(LM["full_lmk_faces_idx"])[0].astype(np.int64)
    lb = np.asarray(LM["full_lmk_bary_coords"])[0]
    for name, (i, tri, bary) in se.FACE_LMK.items():
        assert tuple(int(x) for x in ids[f[lf[i]]]) == tuple(tri), name
        assert np.allclose(lb[i], bary, atol=1e-5), name


# ------------------------------------------------------------------------------------------------ petite taper

TAPER = {"jaw_taper": 0.88, "cheekbone_keep": 0.5, "chin_point": 0.5, "neck_under_jaw": 0.93}


def _moves(body, face, tmp_path):
    out, rep = _edit(body, face, tmp_path)
    return out, np.linalg.norm(out - body.vertices, axis=1), rep


def test_taper_narrows_jaw_and_chin_only_in_width(body, tmp_path):
    out, mv, rep = _moves(body, TAPER, tmp_path)
    own, aft = rep["face"]["own"], rep["face"]["after"]
    assert aft["jaw_angle_c4_12_mm"] < 0.93 * own["jaw_angle_c4_12_mm"]            # the V-taper reaches the jaw angles
    assert aft["chin_c6_10_mm"] / own["chin_c6_10_mm"] < aft["jaw_angle_c4_12_mm"] / own["jaw_angle_c4_12_mm"]   # more
    assert aft["chin_c7_9_mm"] < own["chin_c7_9_mm"]                                # ... the narrower, the lower
    assert aft["cheek_c1_15_mm"] == pytest.approx(own["cheek_c1_15_mm"], abs=1e-6)  # cheekbone line stays
    assert aft["outer_canthal_mm"] == pytest.approx(own["outer_canthal_mm"], abs=1e-6)
    assert np.abs(out[:, 1] - body.vertices[:, 1]).max() == 0.0                    # heights never change
    fr = se.face_frame(body.vertices)
    dz = np.abs(out[:, 2] - body.vertices[:, 2])
    assert dz[body.vertices[:, 1] > fr["y_me"]].max() == 0.0                       # depth only changes on the neck
    assert dz.max() > 1e-4
    dx0, dx1 = np.abs(body.vertices[:, 0] - fr["x_mid"]), np.abs(out[:, 0] - fr["x_mid"])
    assert (dx1 <= dx0 + 1e-12).all()                                              # everything moves toward the midline


def test_taper_height_profiles_are_monotone_and_c1(body):
    fr = se.face_frame(body.vertices)
    y = np.arange(fr["y_me"] - 0.05, fr["y_zy"] + 0.02, 2e-4)
    w0, w1 = se.face_taper_weights(y, fr, 0.0), se.face_taper_weights(y, fr, 1.0)
    assert w0["taper"][y >= fr["y_zy"]].max() == 0.0 and w0["taper"][y <= fr["y_go"]].min() == 1.0
    assert np.all(np.diff(w0["taper"]) <= 1e-12)                                   # fuller further down
    mid = np.searchsorted(y, 0.5 * (fr["y_go"] + fr["y_zy"]))
    assert np.all(w1["taper"] <= w0["taper"] + 1e-12) and w1["taper"][mid] < 0.6 * w0["taper"][mid]   # keep 1: cheeks stay full
    for k, prof in w0.items():
        assert np.abs(np.diff(prof)).max() < 0.03, k                               # no step (every ramp spans > 1 cm)
        assert np.abs(np.diff(prof, 2)).max() < 5e-4, k                            # and no kink in the slope
    assert w0["point"][np.argmin(np.abs(y - fr["y_me"]))] == pytest.approx(1.0, abs=1e-3)   # chin point: full at the menton
    assert w0["point"][y >= fr["y_li"]].max() == 0.0 and w0["point"][y <= fr["y_me"] - 0.025].max() == 0.0
    assert w0["neck"][y >= fr["y_me"]].max() == 0.0 and w0["neck"][y <= fr["y_me"] - se.FACE_NECK_RAMP_M].min() == 1.0


def test_chin_point_acts_on_the_chin_proper(body, tmp_path):
    out, mv, rep = _moves(body, {"chin_point": 1.0}, tmp_path)
    fr = se.face_frame(body.vertices)
    dx = np.abs(body.vertices[:, 0] - fr["x_mid"])
    assert mv[dx > se.FACE_CHIN_POINT_X_M[1] + 1e-9].max() == 0.0                  # the jaw angles / neck sides are left
    moved = mv > 1e-9
    assert body.vertices[moved, 1].max() <= fr["y_li"] and body.vertices[moved, 1].min() >= fr["y_me"] - 0.025 - 1e-9
    own, aft = rep["face"]["own"], rep["face"]["after"]
    assert aft["chin_c7_9_mm"] < 0.85 * own["chin_c7_9_mm"]
    assert aft["jaw_angle_c4_12_mm"] == pytest.approx(own["jaw_angle_c4_12_mm"], rel=0.01)


@pytest.mark.parametrize("face", [TAPER, {"jaw_taper": 0.8, "cheekbone_keep": 1.0, "chin_point": 1.0, "neck_under_jaw": 0.85}])
def test_taper_keeps_eyes_brows_nose_forehead_scalp_ears_and_the_seam(body, tmp_path, face):
    out, mv, rep = _moves(body, face, tmp_path)
    prot = se.face_protected(body.vertices, body.weights)
    fr = se.face_frame(body.vertices)
    assert mv[prot["above_cheekbones"]].max() == 0.0                               # eyes, brows, nose bridge, forehead, scalp
    assert mv[prot["nose"]].max() < 5e-5                                             # < 0.05 mm: the taper's cosine reaches up to the cheekbone line
    eyes = body.weights[:, [23, 24]].sum(1) > 0.5
    assert mv[eyes].max() == 0.0
    y_lobe = fr["y_sto"] + se.FACE_LOBE_FRAC * (fr["y_sn"] - fr["y_sto"])
    ears = (body.vertices[:, 1] > y_lobe - 0.008) & (body.vertices[:, 2] < fr["z_ear"] + se.FACE_FRONT_M[0])
    assert mv[ears].max() == 0.0
    assert mv[body.vertices[:, 2] < fr["z_ear"] + se.FACE_FRONT_NECK_M[0] - 1e-9].max() == 0.0   # ears, nape, back of the head
    head = body.vertices[:, 1] >= fr["y_me"]
    assert mv[head & (body.vertices[:, 2] < fr["z_ear"] + se.FACE_FRONT_M[0] - se.FACE_TAPER_BACK_M - 1e-9)].max() == 0.0
    seam = body.vertices[:, 1] < fr["y_me"] - se.FACE_THROAT_M - se.FACE_TAPER_FADE_M
    assert mv[seam].max() == 0.0                                                   # the neck-shoulder seam and below
    assert mv[se.arm_weight(body.weights) > 0.999].max() == 0.0


def test_taper_neck_falloff_is_c1(body):
    """Displacement down the front of the neck (a dense synthetic column at 2 cm off the midline, 4.5 cm in front of the
    ear plane): continuous, continuous slope, bounded slope, exactly zero from the fade's end."""
    fr = se.face_frame(body.vertices)
    ys = np.arange(fr["y_me"] - 0.12, fr["y_me"] + 0.03, 1e-4)
    P = np.c_[np.full(len(ys), fr["x_mid"] + 0.02), ys, np.full(len(ys), fr["z_ear"] + 0.045)]
    W = np.zeros((len(ys), body.weights.shape[1]))
    f = se.face_factors({"face": TAPER})
    d = se.edit_face(P, W, f, fr) - P
    for c in (0, 2):
        dc = d[:, c]
        assert np.abs(np.diff(dc)).max() < 2e-5                                    # 0.1 mm steps: no jump
        assert np.abs(np.diff(dc, 2)).max() < 6e-7                                 # no kink either (a slope jump gives > 2e-5)
        assert np.abs(np.diff(dc) / 1e-4).max() < 0.35                             # bounded strain
    assert np.abs(d[ys < fr["y_me"] - se.FACE_THROAT_M - se.FACE_TAPER_FADE_M]).max() == 0.0


@pytest.mark.parametrize("face, max_strain, min_dot", [(TAPER, 0.5, 0.85), ({**TAPER, "jaw_taper": 0.8}, 0.7, 0.7)])
def test_taper_smooth_no_folds(body, tmp_path, face, max_strain, min_dot):
    out, _ = _edit(body, face, tmp_path)
    v0, F = body.vertices, body.faces

    def nrm(v):
        n = np.cross(v[F[:, 1]] - v[F[:, 0]], v[F[:, 2]] - v[F[:, 0]])
        return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-15)
    moved = (np.linalg.norm(out - v0, axis=1) > 1e-9)[F].any(1)
    assert (nrm(v0) * nrm(out)).sum(1)[moved].min() > min_dot
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    d = out - v0
    strain = np.linalg.norm(d[e[:, 0]] - d[e[:, 1]], axis=1) / np.linalg.norm(v0[e[:, 0]] - v0[e[:, 1]], axis=1)
    assert strain.max() < max_strain


def test_taper_translation_invariant_and_keys_validated(body, tmp_path):
    shift = np.array([0.1, 1.2, -0.3])
    a, _ = _edit(body, TAPER, tmp_path)
    b, _ = _edit(body, TAPER, tmp_path, v=body.vertices + shift)
    assert np.abs((b - shift) - a).max() < 5e-6                                   # (a 1.2 m shift: float rounding)
    for bad in ({"jaw_taper": 0.6}, {"jaw_taper": 1.1}, {"chin_point": 1.5}, {"cheekbone_keep": -0.1}, {"neck_under_jaw": 0.5},
                {"identity_weight": 2.0}, {"identity_neck": 1.5}, {"identity_mouth": -0.2}):
        with pytest.raises(ValueError, match="allowed range"):
            _edit(body, bad, tmp_path)
    with pytest.raises(ValueError, match="path string"):
        _edit(body, {"identity": 3}, tmp_path)


def test_summary_lists_only_the_non_default_taper_factors(body, tmp_path):
    _, rep = _edit(body, TAPER, tmp_path)
    s = se.summary(rep)["face"]
    assert s["factors"] == TAPER and "identity" not in s


def test_identity_neck_factor_profile_is_c1(body):
    fr = se.face_frame(body.vertices)
    y = np.arange(fr["y_me"] - 0.06, fr["y_me"] + 0.02, 1e-4)
    P = np.c_[np.zeros(len(y)), y, np.zeros(len(y))]
    g = se.identity_neck_factor(np.vstack([body.vertices, P]), 0.6)[len(body.vertices):]   # the frame comes from the body
    assert g[y >= fr["y_me"]].min() == pytest.approx(1.0) and g[y <= fr["y_me"] - se.FACE_IDENTITY_NECK_M].max() == pytest.approx(0.6)
    assert np.all(np.diff(g) >= -1e-12)                                            # monotone toward the face
    assert np.abs(np.diff(g, 2)).max() < 1.5e-4                                    # C1 (a slope jump would give > 1e-3)
    assert np.allclose(se.identity_neck_factor(body.vertices, 1.0), 1.0)           # default: no extra damping


# ------------------------------------------------------------------------------------------------ chin projection

def test_chin_forward_moves_only_the_chin_proper_in_depth(body, tmp_path):
    out, mv, rep = _moves(body, {"chin_forward_mm": 5.0}, tmp_path)
    v0 = body.vertices
    fr = se.face_frame(v0)
    assert np.abs(out[:, 0] - v0[:, 0]).max() == 0.0 and np.abs(out[:, 1] - v0[:, 1]).max() == 0.0   # depth only
    dz = out[:, 2] - v0[:, 2]
    assert dz.min() >= -1e-12 and dz.max() == pytest.approx(0.005, abs=2e-5)                         # forward, up to 5 mm
    assert rep["face"]["after"]["chin_projection_vs_jaw_angle_mm"] - rep["face"]["own"][
        "chin_projection_vs_jaw_angle_mm"] == pytest.approx(5.0, abs=0.5)                             # the menton comes along
    assert mv[v0[:, 1] >= fr["y_li"]].max() == 0.0                                                  # lips, nose, eyes: fixed
    assert mv[v0[:, 1] <= fr["y_me"] - se.FACE_CHIN_FWD_UNDER_M].max() == 0.0                       # throat and neck: fixed
    assert mv[np.abs(v0[:, 0] - fr["x_mid"]) >= se.FACE_CHIN_POINT_X_M[1]].max() == 0.0             # jaw angles / neck sides
    prot = se.face_protected(v0, body.weights)
    assert mv[prot["above_cheekbones"]].max() == 0.0 and mv[prot["behind_front_mask"]].max() == 0.0
    # negative = a chin set back
    back, _, _ = _moves(body, {"chin_forward_mm": -3.0}, tmp_path)
    assert (back[:, 2] - v0[:, 2]).max() <= 1e-12 and (back[:, 2] - v0[:, 2]).min() == pytest.approx(-0.003, abs=2e-5)


def test_chin_forward_weight_is_c1_and_peaks_on_the_chin(body):
    fr = se.face_frame(body.vertices)
    y = np.arange(fr["y_me"] - 0.05, fr["y_li"] + 0.02, 1e-4)
    P = np.c_[np.full(len(y), fr["x_mid"]), y, np.zeros(len(y))]
    w = se.chin_forward_weight(P, fr)
    assert w[y >= fr["y_li"]].max() == 0.0 and w[y <= fr["y_me"] - se.FACE_CHIN_FWD_UNDER_M].max() == 0.0
    assert w[np.argmin(np.abs(y - fr["y_me"]))] == pytest.approx(1.0, abs=1e-9)                        # full at the menton
    assert np.abs(np.diff(w)).max() < 0.03 and np.abs(np.diff(w, 2)).max() < 5e-4                     # smooth: no step, no kink
    x = np.arange(0.0, 0.08, 1e-4)
    Px = np.c_[fr["x_mid"] + x, np.full(len(x), fr["y_me"]), np.zeros(len(x))]
    wx = se.chin_forward_weight(Px, fr)
    assert wx[x <= se.FACE_CHIN_POINT_X_M[0]].min() == pytest.approx(1.0) and wx[x >= se.FACE_CHIN_POINT_X_M[1]].max() == 0.0
    assert np.all(np.diff(wx) <= 1e-12)


def test_chin_forward_composes_with_the_taper_and_is_validated(body, tmp_path):
    both, _, rep = _moves(body, {**TAPER, "chin_forward_mm": 4.0}, tmp_path)
    taper, _, _ = _moves(body, TAPER, tmp_path)
    fwd, _, _ = _moves(body, {"chin_forward_mm": 4.0}, tmp_path)
    assert np.abs((both - body.vertices) - ((taper - body.vertices) + (fwd - body.vertices))).max() < 2e-5   # fields add
    assert se.summary(rep)["face"]["factors"]["chin_forward_mm"] == 4.0
    for bad in ({"chin_forward_mm": 13.0}, {"chin_forward_mm": -7.0}):
        with pytest.raises(ValueError, match="allowed range"):
            _edit(body, bad, tmp_path)
