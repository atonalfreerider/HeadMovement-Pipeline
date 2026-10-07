"""Hair-aware texture checks (avatar_hair.py): per-view hair exclusion maps, hard exclusion + evidence in the bake hook,
learned scalp region, class-aware fill, seam-free two-band fusion, hair colour sample, pod job readiness, and an end-to-end
synthetic bake where red "hair" lies over the follow's back in most cameras.

    .venv-nopo4d\\Scripts\\python.exe -m pytest dancecap/tests/test_avatar_hair.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dancecap import avatar_hair as ah  # noqa: E402
from dancecap import avatar_texture as at  # noqa: E402
from dancecap import avatar_uv as au  # noqa: E402
from dancecap.mamma_dyn import SMPLX_MODELS  # noqa: E402

needs_model = pytest.mark.skipif(not au.model_npz(SMPLX_MODELS).exists(), reason="SMPL-X model not installed")
SIZE = 256


@pytest.fixture(scope="module")
def atlas():
    return au.build_atlas(SMPLX_MODELS, SIZE, cache=False)


@pytest.fixture(scope="module")
def regions(atlas):
    return ah.texel_regions(atlas, SMPLX_MODELS)


def test_hair_maps_zero_on_and_around_hair_and_catch_nearby_strands():
    H, W = 200, 120
    full = np.zeros((H, W), np.uint8)
    full[10:190, 30:90] = 1
    hair = np.zeros((H, W), np.uint8)
    hair[10:60, 35:85] = 1
    p = np.zeros((H, W), np.float32)
    p[60:66, 40:80] = 0.9    # a strand next to the hair mask that the split missed
    p[150:160, 40:80] = 0.9  # red-lit skin far from her hair: not hair, only the soft colour weighting
    keep, hp = ah.hair_maps(full, hair, p)
    assert hp[30, 50] and hp[62, 60] and not hp[155, 60]
    assert keep[30, 50] == 0 and keep[62, 60] == 0
    h = 180
    r0 = max(2.0, 0.006 * h)
    assert keep[66 + int(np.ceil(r0)) - 1, 60] < 0.5  # mixed border just below the strand
    assert keep[120, 60] == 1.0                     # clean white top far from the hair
    assert keep[155, 60] < 0.05                     # soft colour rule still applies (old behaviour kept)
    # no hair mask (keypoints unknown): only hair-coloured pixels in the top 45 % of her mask count
    keep2, hp2 = ah.hair_maps(full, None, p)
    assert hp2[62, 60] and not hp2[155, 60]


def test_hair_maps_face_is_not_cut_by_an_unreliable_hair_mask():
    H, W = 200, 120
    full = np.zeros((H, W), np.uint8)
    full[10:190, 30:90] = 1
    hair = np.zeros((H, W), np.uint8)
    hair[10:80, 30:90] = 1           # the dancers-stage split swallowed her face (warm-lit skin looks hair-like)
    face = np.zeros((H, W), np.uint8)
    face[30:70, 50:75] = 1           # her projected face
    p = np.full((H, W), 0.3, np.float32)
    p[40:45, 55:70] = 0.95           # a real strand across the face (mask AND colour agree)
    keep, hp = ah.hair_maps(full, hair, p, face=face)
    assert keep[60, 60] > 0.4 and not hp[60, 60]   # face skin keeps (relative) weight, no hair evidence
    assert keep[42, 60] == 0 and hp[42, 60]        # the strand over the face is hair
    assert keep[20, 40] == 0 and hp[20, 40]        # hair outside the face: hard exclusion as before
    keep0, _ = ah.hair_maps(full, hair, p)         # without the face: the whole face would be cut
    assert keep0[60, 60] == 0


def test_dark_hair_guard_and_light_quantile_reference():
    import torch
    rgb = at.srgb_to_linear(np.array([[0.40, 0.12, 0.09], [0.80, 0.60, 0.50], [0.85, 0.70, 0.72]], np.float32))
    conf, n = ah.dark_hair_guard(rgb, np.ones(3), np.array([True, True, True]))
    assert n == 1 and conf.tolist() == [0.0, 1.0, 1.0]  # dark red hair dropped; skin and pale pink kept
    x = torch.tensor([[1.0], [2.0], [3.0], [10.0]])      # 3 dark (hair) samples, 1 light (skin) sample
    w = torch.tensor([[1.0], [1.0], [1.0], [1.5]])
    med, _ = at._weighted_median(torch, x, w)
    q75, _ = at._weighted_median(torch, x, w, torch.tensor([0.75]))
    assert med.item() == 3.0 and q75.item() == 10.0


@needs_model
def test_face_projector_covers_the_face_not_the_hair_beside_it(atlas):
    from dancecap.avatar_review import _look_at
    d = {"t": np.array([0.0, 0.1]), "global_orient": np.zeros((2, 3)), "body_pose": np.zeros((2, 63)),
         "left_hand_pose": np.zeros((2, 45)), "right_hand_pose": np.zeros((2, 45)),
         "transl": np.tile((0.0, 1.0, 0.0), (2, 1)), "betas": np.zeros(16), "num_betas": 16, "flat_hand_mean": True}
    body = au.Body(d, SMPLX_MODELS)
    fp = ah.FaceProjector(body, SMPLX_MODELS)
    K = np.array([[900.0, 0, 200], [0, 900, 300], [0, 0, 1]])
    V = body.at(0.0)[0]
    head = V[:, 1].max() - 0.12
    R, t = _look_at(np.array([0.0, head, 2.0]), np.array([0.0, head, 0.0]))  # in front of her face
    m = fp.mask({"t": 0.0, "K": K, "R": R, "t_w": t, "k1": 0.0}, 600, 400)
    assert m is not None and 1500 < m.sum() < 20000
    ys, xs = np.nonzero(m)
    assert xs.max() - xs.min() < 0.8 * (ys.max() - ys.min()) + 40  # a face, not a hull over the shoulders
    R2, t2 = _look_at(np.array([0.0, head, -2.0]), np.array([0.0, head, 0.0]))  # behind her: no face
    assert fp.mask({"t": 0.0, "K": K, "R": R2, "t_w": t2, "k1": 0.0}, 600, 400) is None


def test_exclude_hair_hard_zero_and_evidence():
    import torch
    store = SimpleNamespace(hair_ev=torch.zeros((2, 4)))
    w = torch.tensor([1.0, 2.0, 3.0, 4.0])
    px = torch.tensor([[1.0, 1.0], [5.0, 1.0], [1.0, 1.0], [5.0, 1.0]])
    keep = np.ones((3, 8), np.float32)
    keep[:, :3] = 0.0
    hmap = np.zeros((3, 8), np.uint8)
    hmap[:, :3] = 1
    scalp = torch.tensor([False, False, True, False])
    w2 = ah.exclude_hair(torch, store, w, px, keep, hmap, scalp, at._sample_nearest)
    assert w2.tolist() == [0.0, 2.0, 3.0, 4.0]  # body texel on hair -> 0 (not just relatively lower); scalp keeps it
    assert store.hair_ev[0].tolist() == [1.0, 0.0, 3.0, 0.0] and store.hair_ev[1].tolist() == [1.0, 2.0, 3.0, 4.0]


@needs_model
def test_scalp_is_anatomical_and_hanging_hair_never_becomes_scalp(atlas, regions):
    """Evidence = hair over the top / back of the head AND hanging hair over her right cheek, right ear and jaw (the
    'bob' the first version learned). The region must stay on the anatomical scalp: never an ear, never below the eye
    line in front of the ear, never the face core; the forehead hairline follows the evidence inside the margin."""
    P0, head = regions["P0"], regions["head"]
    m = np.load(au.model_npz(SMPLX_MODELS))
    J = np.asarray(m["J_regressor"]) @ np.asarray(m["v_template"])
    eye = J[23:25, 1].mean()
    dx, dy, dz = P0[:, 0] - J[15, 0], P0[:, 1] - eye, P0[:, 2] - J[15, 2]
    top_back = head & ((dz < 0.0) | (dy > 0.07))
    bob = head & (dx < -0.03) & (dy < 0.03) & (dy > -0.12)  # one side of the face: cheek, ear, jaw
    is_hair = top_back | bob
    ev_all = np.where(head, 1.0, 0.0)
    ev_hair = np.where(is_hair, 0.95, 0.03) * ev_all
    hb, hs, st = ah.hair_region_from_evidence(atlas, ev_hair, ev_all, regions)
    assert not hb[~head].any() and not hb[regions["face_core"]].any() and not hb[regions["ear"]].any()
    low_front = head & (dz > -0.035) & (dy < 0.0)
    assert not hb[low_front].any(), int(hb[low_front].sum())       # no cheek / jaw / temple below the eye line
    assert hb[regions["prior"] & top_back].mean() > 0.95           # the anatomical scalp under her hair
    assert st["ear_hair_fraction"] == 0.0 and st["evidence_above_thresh_outside_allowed_head_fraction"] > 0.05
    # the learned hairline stays near the anatomical scalp (never below the eye line in front of the ear)
    assert not (hb & ~regions["prior"] & (dy < 0.0)).any()
    # forehead band: skin seen where the anatomical scalp reaches down the forehead -> not hair there
    band_front = regions["prior"] & (dz > 0.02) & (dy < 0.07)
    if band_front.any():
        assert hb[band_front].mean() < 0.5
    # no evidence -> the anatomical scalp
    hb0, _, _ = ah.hair_region_from_evidence(atlas, np.zeros(atlas.n), np.zeros(atlas.n), regions)
    assert (hb0 == regions["prior"])[head].mean() > 0.97
    # hair cover keeps what hung over the body (not a root region)
    cov = ah.hair_cover(ev_hair, ev_all)
    assert cov[bob].mean() > 0.9 and cov[~head].max() == 0


def test_scalp_vertices_match_flame_masks_when_available():
    """The geometric scalp / ear rule against the FLAME masks (local, git-ignored licensed assets; skipped without)."""
    import pickle

    from dancecap.take import WORKSPACE
    mp = WORKSPACE / "external" / "smirk" / "assets" / "FLAME_masks" / "FLAME_masks.pkl"
    ids_p = WORKSPACE / "models" / "flame" / "SMPL-X__FLAME_vertex_ids.npy"
    if not (mp.exists() and ids_p.exists() and au.model_npz(SMPLX_MODELS).exists()):
        pytest.skip("FLAME masks / SMPL-X correspondence not installed")
    with open(mp, "rb") as fh:
        fm = pickle.load(fh, encoding="latin1")
    ids = np.load(ids_p)
    sv = ah.scalp_vertices(np.load(au.model_npz(SMPLX_MODELS), allow_pickle=True))
    n = len(sv["scalp"])
    fl = lambda *k: np.isin(np.arange(n), ids[np.concatenate([fm[x] for x in k])])
    ear, scalp = fl("left_ear", "right_ear"), fl("scalp")
    assert (sv["ear"] & ear).sum() / ear.sum() > 0.95 and (sv["ear"] & ear).sum() / sv["ear"].sum() > 0.95
    assert (sv["scalp"] & scalp).sum() / sv["scalp"].sum() > 0.95          # inside the FLAME scalp
    assert (sv["scalp"] & scalp).sum() / (scalp & sv["head"]).sum() > 0.85  # most of it
    assert not (sv["scalp"] & ear).any()


@needs_model
def test_class_aware_fill_gives_clean_top_or_skin_not_a_blend(atlas, regions):
    P0 = regions["P0"]
    white, skin = np.array([0.85, 0.85, 0.85]), np.array([0.55, 0.32, 0.22])
    m = np.load(au.model_npz(SMPLX_MODELS))
    J = np.asarray(m["J_regressor"]) @ np.asarray(m["v_template"])
    top = P0[:, 1] < J[9, 1] + 0.03  # crop top below the upper chest, skin above (neck, shoulders)
    srgb = np.where(top[:, None], white, skin)
    lin = at.srgb_to_linear(srgb)
    conf = np.ones(atlas.n)
    hole = (P0[:, 2] < J[9, 2] - 0.02) & (np.abs(P0[:, 0]) < 0.12) & (np.abs(P0[:, 1] - (J[9, 1] + 0.03)) < 0.12)
    hole &= ~regions["head"]
    conf[hole] = 0
    assert hole.sum() > 100
    n_verts = len(m["v_template"])
    blend = {}
    for name in ("classed", "plain"):
        out, alpha, src = ah.fill_classed(atlas, lin, conf, n_verts, 1.0, classes=2) if name == "classed" else \
            at.fill_holes(atlas, lin, conf, n_verts, 1.0)
        f = at.linear_to_srgb(out[hole])
        d = np.minimum(np.abs(f - white).max(1), np.abs(f - skin).max(1))
        blend[name] = (d > 0.08).mean()  # neither clean white nor clean skin
        assert (src[hole] > 0).all()
    assert blend["classed"] < 0.12 and blend["plain"] > 2 * blend["classed"], blend


@needs_model
def test_fill_never_copies_hair_and_neck_holes_take_skin(atlas, regions):
    """Bad seeds (hair-coloured / mostly hair) are never mirrored or diffused into holes, a red k-means class seeds
    nothing, neck holes take the skin class even when the nearest measured texel is the white top."""
    P0 = regions["P0"]
    m = np.load(au.model_npz(SMPLX_MODELS))
    J = np.asarray(m["J_regressor"]) @ np.asarray(m["v_template"])
    white, skin, red = np.array([0.86, 0.86, 0.85]), np.array([0.80, 0.60, 0.48]), np.array([0.45, 0.12, 0.09])
    top = ~regions["head"] & ~regions["neck"]
    srgb = np.where(top[:, None], white, skin)
    back = regions["upper"] & ~regions["front"] & ~regions["neck"]
    patch = back & (P0[:, 0] > 0.02) & (P0[:, 1] > J[9, 1])
    srgb[patch] = red                                      # hair leaked onto one side of her upper back (measured)
    lin = at.srgb_to_linear(srgb)
    conf = np.ones(atlas.n)
    hole = back & (P0[:, 0] < -0.02) & (P0[:, 1] > J[9, 1])  # its mirror image is a hole: must not copy the red
    neck_hole = regions["neck"] & (P0[:, 2] < J[12, 2])    # back of the neck (nape) unmeasured
    conf[hole | neck_hole] = 0
    hair = regions["prior"]
    conf2, bad, skin_only, st = ah.fill_guards(lin, conf, regions, hair)
    assert st["leak_guard_dropped"]["upper"] > 0 and (conf2[patch] == 0).all()
    n_verts = len(m["v_template"])
    out, alpha, src = ah.fill_region_aware(atlas, lin, conf2, n_verts, 1.0, hair, hair.astype(np.float32),
                                           classes=3, bad=bad, skin_only=skin_only)
    f = at.linear_to_srgb(out)
    rc = ah.red_cast(ah.lab_of(f))
    assert rc[back].mean() < 0.01, rc[back].mean()        # no hair colour copied or diffused onto her back
    dn = np.abs(f[neck_hole] - skin).max(1)
    assert (dn < 0.08).mean() > 0.9, (dn < 0.08).mean()    # nape filled with her skin, not the white top
    # without the guards the mirror copies the red patch onto the other side of her back
    out0, _, _ = ah.fill_region_aware(atlas, lin, conf, n_verts, 1.0, hair, hair.astype(np.float32), classes=3)
    assert ah.red_cast(ah.lab_of(at.linear_to_srgb(out0[hole]))).mean() > 0.5


def test_fill_guards_keep_the_light_pink_print_but_not_as_a_seed():
    regions = {k: np.ones(3, bool) if k == "upper" else np.zeros(3, bool) for k in ("upper", "neck", "head")}
    srgb = np.array([[0.45, 0.12, 0.09],   # dark red: hair leaked through -> dropped
                     [0.92, 0.62, 0.72],   # light pink print on her top -> kept, but never seeds a fill
                     [0.86, 0.86, 0.85]], np.float32)
    conf2, bad, skin_only, st = ah.fill_guards(at.srgb_to_linear(srgb), np.ones(3), regions, np.zeros(3, bool),
                                               ev=np.array([[0.0, 0.0, 0.9], [1.0, 1.0, 1.0]]))
    assert conf2.tolist() == [0.0, 1.0, 1.0] and bad.tolist() == [True, True, True]  # 3rd: seen 90 % through hair
    assert st["leak_guard_dropped"]["upper"] == 1


def test_hair_flow_mask_and_hair_mask_coverage(tmp_path):
    union = np.zeros((100, 60), bool)
    union[10:90, 10:50] = True
    hp = np.zeros((100, 60), np.uint8)
    hp[20:30, 20:30] = 1
    m = at.hair_flow_mask({"hair_px": {"follow": hp}}, union)
    assert m[25, 25] and m[25, 18] and not m[60, 25]       # dilated by >= 3 px, nothing far away
    assert at.hair_flow_mask({}, union) is None and at.hair_flow_mask({"hair_px": {"follow": hp * 0}}, union) is None
    from test_avatar_texture import _mini_work
    work = _mini_work(tmp_path)
    views = [{"cam": c, "i": i} for c in ("01", "02") for i in range(10)]
    (work / "dancers" / "01" / "hair" / "mask_0003_02.png").unlink()
    (work / "dancers" / "01" / "hair" / "mask_0004_02.png").unlink()
    (work / "dancers" / "01" / "body" / "mask_0004_02.png").unlink()
    cov = at.hair_mask_coverage(work, views)["follow"]
    assert (cov["mask"], cov["derived"], cov["none"]) == (18, 1, 1) and cov["real_fraction"] == 0.9


def test_review_gate_counts_and_flow_compare():
    from dancecap import avatar_hair_review as hr
    pv = [{"cam": "09", "px": 100, "upper_px": 50, "face_px": 0,
           "before": {"mae": 30.0, "mae_gain": 25.0, "mae_upper": 32.0, "mae_gain_upper": 28.0, "mae_face": None,
                      "mae_gain_face": None, "false_hair_pct": 1.0},
           "after": {"mae": 29.0, "mae_gain": 24.0, "mae_upper": 32.05, "mae_gain_upper": 29.0, "mae_face": None,
                     "mae_gain_face": None, "false_hair_pct": 0.5}}]
    s = hr._summarise(pv)
    g = hr._gate(s, "09")
    assert g["checks"]["mae_upper"]["ok"] and not g["checks"]["mae_gain_upper"]["ok"] and g["passed"] is False
    w = hr._worse_counts(pv)
    assert w["mae_gain_upper"] == {"worse": 1, "better": 0, "n": 1} and w["mae_face"]["n"] == 0
    rb = {"views": [{"cam": "01", "i": 1, "flow": {"p90_px": 3.0}}, {"cam": "01", "i": 2, "flow": {"p90_px": 9.0}}]}
    ra = {"views": [{"cam": "01", "i": 1, "flow": {"p90_px": 6.0}}, {"cam": "01", "i": 2, "flow": {"p90_px": 4.0}}]}
    fc = hr._flow_compare(rb, ra)
    assert fc["p90_worse_gt_2px"] == 1 and fc["p90_better_gt_2px"] == 1 and fc["views"] == 2


def test_multiband_removes_exposure_seam_and_keeps_detail():
    import torch
    S = 64
    yy, xx = np.mgrid[0:S, 0:S]
    atlas = SimpleNamespace(texel_yx=np.stack([yy.ravel(), xx.ravel()], 1).astype(np.int32), size=S)
    N = S * S
    base = 0.3 + 0.1 * ((xx + yy) % 2).ravel()          # 1-texel checker detail
    T = np.zeros((2, N, 3), np.float32)
    T[0] = base[:, None]
    T[1] = 1.3 * base[:, None]                           # camera 1: 30 % brighter
    Q = np.zeros((2, N), np.float32)
    Q[0, xx.ravel() < 40] = 1.0
    Q[1, xx.ravel() >= 24] = 1.0
    Ws = np.where(xx.ravel() < 32, [[1.0], [1e-4]], [[1e-4], [1.0]]).astype(np.float32)  # sharp: winner switches at 32
    lay = tuple(torch.as_tensor(a) for a in (T, Ws * (Q > 0), Q))
    groups = np.zeros(N, np.int32)
    out = ah.multiband(torch, lay, atlas, groups, torch.full((N,), 4.0), sigma=6.0).numpy()[:, 0].reshape(S, S)
    sharp = ((Ws * (Q > 0))[..., None] * T).sum(0)[:, 0] / (Ws * (Q > 0)).sum(0)
    sharp = sharp.reshape(S, S)
    col_mean = lambda img: np.array([img[:, x].mean() for x in range(S)])
    jump_sharp = np.abs(np.diff(col_mean(sharp))[28:36]).max()
    jump_mb = np.abs(np.diff(col_mean(out))[28:36]).max()
    assert jump_sharp > 0.1 and jump_mb < 0.25 * jump_sharp, (jump_sharp, jump_mb)
    det = lambda img, x0, x1: np.abs(np.diff(img[:, x0:x1], axis=1)).mean()
    assert det(out, 4, 20) > 0.9 * det(sharp, 4, 20) and det(out, 44, 60) > 0.9 * det(sharp, 44, 60)
    # beyond the low-pass reach of camera 1 (x < 24 - 3 sigma) camera 0 is reproduced exactly (low + high = layer)
    assert np.allclose(out[:, 0:5], sharp[:, 0:5], atol=1e-5)
    assert np.abs(out[:, 6:20] - sharp[:, 6:20]).max() < 0.002  # ^4 weights: camera 1 barely reaches in
    # high-band weights smoothed over ~2 texels (the bake's default): still no seam, detail kept
    out2 = ah.multiband(torch, lay, atlas, groups, torch.full((N,), 4.0), sigma=6.0, sigma_high=2.0).numpy()
    out2 = out2[:, 0].reshape(S, S)
    assert np.abs(np.diff(col_mean(out2))[28:36]).max() < 0.25 * jump_sharp
    assert det(out2, 4, 20) > 0.9 * det(sharp, 4, 20) and np.allclose(out2[:, 0:5], sharp[:, 0:5], atol=1e-5)


def test_lowpass_does_not_cross_groups():
    import torch
    S = 32
    yy, xx = np.mgrid[0:S, 0:S]
    yx = np.stack([yy.ravel(), xx.ravel()], 1)
    groups = (xx.ravel() >= 16).astype(np.int32)
    vals = torch.as_tensor(np.where(groups[:, None] == 1, 1.0, 0.0)[None].astype(np.float32).repeat(3, -1))
    lp, _ = ah.lowpass_groups(torch, vals, torch.ones((1, S * S)), yx, groups, sigma=8.0)
    lp = lp.numpy()[0, :, 0]
    assert lp[groups == 0].max() < 1e-5 and lp[groups == 1].min() > 1 - 1e-5


def test_hair_sample_palette_gradient_and_strip(tmp_path):
    rng = np.random.default_rng(0)
    pix = []
    for vi in range(5):
        n = 600
        d = rng.uniform(0, 0.6, n).astype(np.float32)
        root, tip = np.array([90, 25, 20]), np.array([200, 80, 50])  # dark roots, lighter copper tips
        rgb = (root + (tip - root) * (d / 0.6)[:, None] + rng.normal(0, 6, (n, 3))).clip(0, 255).astype(np.uint8)
        pix.append((vi, rgb, d))
    gains = np.ones((5, 3), np.float32)
    rep = ah.hair_sample(pix, gains, tmp_path, "follow")
    assert (tmp_path / "follow_hair.json").exists() and len(rep["palette"]) == 6
    strip = cv2.imread(str(tmp_path / "follow_hair_strip.png"))[:, :, ::-1]
    assert strip.shape == (512, 256, 3)
    assert strip[:40].mean() < strip[-40:].mean()  # root darker than tips
    assert rep["root_srgb8"][0] < rep["tip_srgb8"][0] and rep["n_pixels"] == 3000
    assert abs(sum(p["fraction"] for p in rep["palette"]) - 1) < 1e-3


def test_red_cast_rule():
    lab = ah.lab_of(np.array([[0.55, 0.18, 0.12],   # red hair
                              [0.85, 0.70, 0.72],   # pink hair-over-white smear
                              [0.80, 0.60, 0.48],   # skin
                              [0.86, 0.86, 0.85],   # white top
                              [0.62, 0.70, 0.80]], np.float32))  # light jeans
    assert ah.red_cast(lab).tolist() == [True, True, False, False, False]


def test_pod_job_carries_hair_masks_of_every_view_and_the_new_code_path(tmp_path):
    import json
    import tarfile

    from dancecap import avatar_pod
    from test_avatar_texture import _mini_work
    work = _mini_work(tmp_path)
    job = avatar_pod.write_pod_job(work, "pod", 20, log=lambda *_: None)
    with tarfile.open(job["inputs"]["texture_inputs"]["local"]) as tar:
        names = set(tar.getnames())
    pre = "dancecap/work/mini/"
    for c, i, _ in at._view_list(work, ["01", "02"], -1e9, 1e9, 1 / 30):
        assert f"{pre}dancers/{c}/hair/mask_{i:04d}_02.png" in names
    assert any(n.endswith("dancecap/avatar_hair.py") for n in names)
    run = job["pod_commands"]["run"]
    assert "--hair-mode exclude" in run and "--blend multiband" in run
    assert job["code_path"]["hair_mode"] == "exclude"
    for o in ("avatar/follow_hair.json", "avatar/follow_hair_strip.png", "avatar/follow_hair_region.png"):
        assert o in job["outputs"]
    assert json.loads((work / "avatar" / "pod_job.json").read_text())["code_path"]["blend"] == "multiband"


def test_pod_job_refuses_short_kill_timer_and_run_pod_threads_step_and_cost(tmp_path, monkeypatch):
    from dancecap import avatar_pod, remote
    from test_avatar_texture import _mini_work
    work = _mini_work(tmp_path)
    job = avatar_pod.write_pod_job(work, "pod", None, pack=False, log=lambda *_: None)
    rec = job["estimate"]["max_minutes_recommended"]
    assert job["max_minutes"] == rec and job["estimate"]["s_per_view"] == avatar_pod.POD_S_PER_VIEW
    with pytest.raises(ValueError):
        avatar_pod.write_pod_job(work, "pod", rec - 1, pack=False, log=lambda *_: None)
    j2 = avatar_pod.write_pod_job(work, "pod", None, pack=False, log=lambda *_: None, step_ms=66.7, hair_mode="soft",
                                  blend="mean")
    assert j2["estimate"]["views"] < job["estimate"]["views"] and "--step-ms 66.7" in j2["pod_commands"]["run"]
    assert "--hair-mode soft --blend mean" in j2["pod_commands"]["run"]
    assert j2["estimate"]["s_per_view"] == avatar_pod.POD_S_PER_VIEW_OLD
    seen = {}

    class Stop(Exception):
        pass

    def fake_session(name, max_minutes, **kw):
        seen.update(kw, max_minutes=max_minutes)
        raise Stop
    monkeypatch.setattr(remote, "pod_session", fake_session)
    monkeypatch.setattr("dancecap.avatar_texture.smplx_freshness", lambda w: {"stale": False, "status": "ok"})
    with pytest.raises(Stop):
        avatar_pod.run_pod(work, "pod", None, log=lambda *_: None, step_ms=66.7, max_cost=4.0)
    j3 = avatar_pod.write_pod_job(work, "pod", None, pack=False, log=lambda *_: None, step_ms=66.7)
    assert seen["max_cost"] == 4.0 and seen["max_minutes"] == j3["max_minutes"]
    with pytest.raises(ValueError):  # refused before any pod is created
        avatar_pod.run_pod(work, "pod", 1, log=lambda *_: None)


# ----------------------------------------------------------------------------------------- synthetic end to end


@needs_model
def test_synthetic_bake_red_hair_over_the_back_never_reaches_the_texture(atlas, regions):
    """Follow seen by 6 cameras; in 4 of them red hair hangs over her upper back (hair mask given, plus a pink mixed
    border just outside the mask). Old soft mode bakes red into the back; exclude mode takes the back from the 2 clean
    cameras, and the hair-only patch (covered in every camera) is filled with the surrounding body colours."""
    import torch

    from dancecap.avatar_review import _look_at
    from test_avatar_texture import _gt_colors
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = {"t": np.array([0.0, 0.1]), "global_orient": np.zeros((2, 3)), "body_pose": np.zeros((2, 63)),
         "left_hand_pose": np.zeros((2, 45)), "right_hand_pose": np.zeros((2, 45)),
         "transl": np.tile((0.0, 1.2, 0.0), (2, 1)), "betas": np.zeros(16), "num_betas": 16, "flat_hand_mean": True}
    d["body_pose"][:, 15 * 3 + 2] = -1.0
    d["body_pose"][:, 16 * 3 + 2] = 1.0
    body = au.Body(d, SMPLX_MODELS)
    far = dict(d, transl=np.tile((5.0, 1.2, 5.0), (2, 1)))  # the lead stands far away (out of every view)
    bodies = {"lead": au.Body(far, SMPLX_MODELS), "follow": body}
    gt = {r: np.clip(_gt_colors(atlas, r) * 0.4 + 0.45, 0, 1).astype(np.float32) for r in at.ROLES}  # no red hues
    geo = at._Geo(atlas, dev)
    P0 = regions["P0"]
    m = np.load(au.model_npz(SMPLX_MODELS))
    J = np.asarray(m["J_regressor"]) @ np.asarray(m["v_template"])
    upper_back = regions["upper"] & ~regions["front"] & (P0[:, 1] > J[6, 1])
    always_hair = upper_back & (np.abs(P0[:, 0]) < 0.04) & (P0[:, 1] > J[9, 1])  # hidden in every camera
    W, H, f = 300, 400, 380.0
    imgs, views = {}, []
    hair_rgb = np.array([190, 40, 30], np.uint8)
    for k in range(6):
        a = 2 * np.pi * k / 6 + 0.3
        C = np.array([3.0 * np.sin(a), 1.3, 3.0 * np.cos(a)])
        R, t = _look_at(C, np.array([0.0, 0.9, 0.0]))
        K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
        posed = {"follow": body.at(0.0)[0]}
        ren, rid = at.render(torch, geo, posed, {"follow": torch.as_tensor(gt["follow"])}, K, R, t, 0.0, H, W, dev)
        regc = np.stack([upper_back, always_hair, np.zeros(atlas.n, bool)], 1).astype(np.float32)
        rr, _ = at.render(torch, geo, posed, {"follow": torch.as_tensor(regc)}, K, R, t, 0.0, H, W, dev)
        img = (np.clip(ren, 0, 1) * 255 + 0.5).astype(np.uint8)
        img[rid < 0] = (90, 90, 90)
        fol = (rid == 0).astype(np.uint8)
        hairm = np.zeros((H, W), np.uint8)
        if k < 4:
            hairm = ((rr[..., 0] > 0.5) & (rid == 0)).astype(np.uint8)
        hairm |= ((rr[..., 1] > 0.5) & (rid == 0)).astype(np.uint8)
        img[hairm > 0] = hair_rgb
        ring = (cv2.dilate(hairm, np.ones((5, 5), np.uint8)) > 0) & (hairm == 0) & (fol > 0)
        img[ring] = (img[ring].astype(np.float32) * 0.5 + hair_rgb * 0.5).astype(np.uint8)  # pink mixed border
        full = {"lead": np.zeros((H, W), np.uint8), "follow": fol}
        bodym = {"lead": full["lead"], "follow": fol & (1 - hairm)}
        imgs[k] = (img, full, bodym, {"follow": hairm})
        views.append({"cam": f"{k:02d}", "i": k, "t": 0.0, "K": K, "R": R, "t_w": t, "k1": 0.0, "ok": True,
                      "swap": 0.0})
    cams = [f"{k:02d}" for k in range(6)]
    view_cam = np.arange(6)
    res = {}
    for mode in ("soft", "exclude"):
        cfg = at.TextureConfig(size=SIZE, device=dev, hd_cams=(), cam_weight={}, chunk=100_000, hair_mode=mode,
                               blend="multiband" if mode == "exclude" else "mean")
        # the old path used a colour model; give it a perfect one (P(hair) = 1 on the hair colour)
        lut = np.zeros((32, 32, 32), np.float32)
        lab = cv2.cvtColor(hair_rgb[None, None], cv2.COLOR_RGB2LAB)[0, 0] // at.HAIR_BIN
        lut[lab[0], lab[1], lab[2]] = 1.0
        loader = lambda v, cfg=cfg, lut=lut: at.view_data(imgs[v["i"]][0], imgs[v["i"]][1], imgs[v["i"]][2],
                                                           cfg.margin_frac, lut, hair=imgs[v["i"]][3], cfg=cfg)
        hr = {"follow": regions["prior"]} if mode == "exclude" else None
        stores, stats = at.bake_samples(views, bodies, atlas, cfg, cams, loader, dev, hair_region=hr)
        groups = {"follow": ah.texel_groups(ah.island_ids(atlas), regions["prior"]), "lead": ah.island_ids(atlas)}
        results, _ = at.fuse_stores(stores, stats, view_cam, cams, cfg, dev, atlas=atlas, groups=groups)
        rgb, conf, ncam, g = results["follow"]
        if mode == "exclude":
            ev = stores["follow"].hair_ev.cpu().numpy()
            assert ev[0][always_hair].sum() > 0.5 * ev[1][always_hair].sum()  # evidence: these texels were hair
            filled, alpha, src = ah.fill_region_aware(atlas, rgb, conf, 10475, None, regions["prior"],
                                                      regions["prior"].astype(np.float32), classes=4)
        else:
            filled, alpha, src = at.fill_holes(atlas, rgb, conf, 10475)
        res[mode] = (at.linear_to_srgb(filled), conf)
    hair_f = hair_rgb / 255.0
    near_hair = lambda x: np.abs(x - hair_f).max(1) < 0.2
    soft, exc, conf_e = res["soft"][0], res["exclude"][0], res["exclude"][1]
    side = upper_back & ~always_hair
    err_soft = np.abs(soft[side] - gt["follow"][side]).max(1)
    seen = side & (conf_e > 0)  # exclude mode: measured from the clean cameras only
    err_exc = np.abs(exc[seen] - gt["follow"][seen]).max(1)
    msg = (f"soft: >0.1 err {(err_soft > 0.1).mean():.3f}, hair-covered texels near hair "
           f"{near_hair(soft[always_hair]).mean():.3f}; exclude: {seen.sum()} of {side.sum()} side texels measured, "
           f"median err {np.median(err_exc):.3f}, >0.15 {(err_exc > 0.15).mean():.3f}, any back texel near hair "
           f"{near_hair(exc[upper_back]).mean():.4f}, hair-covered unmeasured {(conf_e[always_hair] == 0).mean():.3f}")
    assert (err_soft > 0.1).mean() > 0.2 and near_hair(soft[always_hair]).mean() > 0.5, msg  # old: red baked in
    assert seen.sum() > 100 and np.median(err_exc) < 0.05 and (err_exc > 0.15).mean() < 0.05, msg  # clean cameras
    assert near_hair(exc[upper_back]).mean() < 0.005 and (conf_e[always_hair] == 0).mean() > 0.9, msg  # no hair at all
