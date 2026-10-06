"""Avatar texture stage checks: UV atlas, seam-split mesh, FK, and an end-to-end synthetic bake with ground truth
(two SMPL-X bodies, one partly occluding the other, 8 cameras, one camera under-exposed).

    .venv-nopo4d\\Scripts\\python.exe -m pytest dancecap/tests/test_avatar_texture.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dancecap import avatar_texture as at  # noqa: E402
from dancecap import avatar_uv as au  # noqa: E402
from dancecap.mamma_dyn import SMPLX_MODELS  # noqa: E402
from dancecap.take import PACKAGE_ROOT  # noqa: E402

REAL = PACKAGE_ROOT / "work" / "larissa_kadu"
SIZE = 512

needs_model = pytest.mark.skipif(not au.model_npz(SMPLX_MODELS).exists(), reason="SMPL-X model not installed")


@pytest.fixture(scope="module")
def atlas():
    return au.build_atlas(SMPLX_MODELS, SIZE, cache=False)


@needs_model
def test_uv_atlas_texels_lie_on_their_uv_triangles(atlas):
    assert 0.65 < atlas.n / SIZE ** 2 < 0.8  # SMPL-X UV layout fills ~73 % of the square
    assert np.allclose(atlas.bary.sum(1), 1, atol=1e-5) and (atlas.bary >= 0).all()
    # reconstruct each texel centre from its face + barycentrics: within half a texel (edge texels are clamped)
    X = au.uv_pixels(atlas.vt, SIZE)
    rec = np.einsum("nkd,nk->nd", X[atlas.ft[atlas.face]], atlas.bary)
    err = np.abs(rec - atlas.texel_yx[:, ::-1]).max(1)
    assert np.median(err) < 1e-3 and np.percentile(err, 99.9) < 1.0 and err.max() < 2.0  # fillPoly border texels
    # uv convention: texel row 0 is v = 1
    i = int(np.argmin(atlas.texel_yx[:, 0]))
    uv = np.einsum("kd,k->d", atlas.vt[atlas.ft[atlas.face[i]]], atlas.bary[i])
    assert uv[1] > 0.9


@needs_model
def test_mirror_map_is_an_involution_and_symmetric(atlas):
    m = atlas.mirror
    ok = m >= 0
    assert ok.mean() > 0.9
    mm = m[m[ok]]
    back = mm >= 0
    m0 = np.load(au.model_npz(SMPLX_MODELS))
    P0 = atlas.surface(np.asarray(m0["v_template"], float))
    # mirror of mirror lands within 1 cm of the start; the mirror point is the reflected point
    assert np.median(np.linalg.norm(P0[mm[back]] - P0[ok][back], axis=1)) < 0.004
    assert np.median(np.abs(P0[m[ok], 0] + P0[ok, 0])) < 0.004


@needs_model
def test_uv_mesh_split_is_consistent(tmp_path):
    info = au.write_uv_mesh(tmp_path / "uv_mesh.npz", SMPLX_MODELS)
    z = np.load(tmp_path / "uv_mesh.npz")
    m = np.load(au.model_npz(SMPLX_MODELS))
    assert info["vertices"] == len(m["vt"]) == 11313 and info["faces"] == 20908
    assert np.array_equal(z["vertex_ids"][z["faces"]], m["f"])
    assert np.allclose(z["uv"], m["vt"])
    assert set(np.unique(z["vertex_ids"])) == set(range(len(m["v_template"])))


@needs_model
def test_fk_reproduces_stored_joints_real_take():
    if not (REAL / "smplx" / "lead.npz").exists():
        pytest.skip("larissa_kadu smplx/ not produced")
    bodies, info = at.load_bodies(REAL)
    for r, d in info.items():
        assert d["fk_max_err_mm"] < 0.1, (r, d)
    v, vel, q = bodies["lead"].at(float(bodies["lead"].t[10]) + 0.01)
    assert v.shape == (10475, 3) and np.isfinite(vel).all() and 0 <= q <= 1


def test_gutter_extends_islands():
    img = np.zeros((32, 32, 3), np.float32)
    cov = np.zeros((32, 32), bool)
    img[10:20, 10:20] = (0.2, 0.4, 0.6)
    cov[10:20, 10:20] = True
    out = at.gutter(img, cov, 4)
    assert np.allclose(out[8, 12], (0.2, 0.4, 0.6)) and np.allclose(out[12, 22], (0.2, 0.4, 0.6))


# ----------------------------------------------------------------------------------------- synthetic end to end


def _gt_colors(atlas, role):
    m0 = np.load(au.model_npz(SMPLX_MODELS))
    P0 = atlas.surface(np.asarray(m0["v_template"], float))
    ph = 0.0 if role == "lead" else 1.7
    return (0.5 + 0.33 * np.sin(np.stack([6 * P0[:, 0] + ph, 5 * P0[:, 1] + 2 * ph, 7 * P0[:, 2] - ph], 1)
                                + P0[:, [1, 2, 0]] * 4)).astype(np.float32)


def _scene():
    T = 2
    d = {}
    for r, tr in (("lead", (-0.15, 1.2, -0.2)), ("follow", (0.2, 1.2, 0.25))):
        d[r] = {"t": np.array([0.0, 0.1]), "global_orient": np.zeros((T, 3)), "body_pose": np.zeros((T, 63)),
                "left_hand_pose": np.zeros((T, 45)), "right_hand_pose": np.zeros((T, 45)),
                "transl": np.tile(tr, (T, 1)), "betas": np.zeros(16), "num_betas": 16, "flat_hand_mean": True}
        d[r]["body_pose"][:, 15 * 3 + 2] = -1.0  # left shoulder down a bit (joint 16 -> body_pose idx 15)
        d[r]["body_pose"][:, 16 * 3 + 2] = 1.0
    bodies = {r: au.Body(d[r], SMPLX_MODELS) for r in d}
    cams = []
    W, H, f = 360, 480, 430.0
    for k in range(8):
        a = 2 * np.pi * k / 8 + 0.2
        C = np.array([3.4 * np.sin(a), 1.3 + 0.2 * (k % 2), 3.4 * np.cos(a)])
        from dancecap.avatar_review import _look_at
        R, t = _look_at(C, np.array([0.0, 0.85, 0.0]))
        cams.append({"K": np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]]), "R": R, "t": t, "W": W, "H": H})
    return bodies, cams


@needs_model
def test_synthetic_bake_recovers_texture_with_occlusion_and_exposure(atlas):
    import torch

    from dancecap.avatar_texture import render
    bodies, cams = _scene()
    gt = {r: _gt_colors(atlas, r) for r in at.ROLES}
    geo = at._Geo(atlas, "cpu")
    gain_true = {3: 0.7}  # camera 3 under-exposed (linear)
    imgs = {}
    views = []
    for ci, c in enumerate(cams):
        for t in (0.0, 0.1):
            posed = {r: b.at(t)[0] for r, b in bodies.items()}
            ren, rid = render(torch, geo, posed, {r: torch.as_tensor(gt[r]) for r in at.ROLES}, c["K"], c["R"],
                              c["t"], 0.0, c["H"], c["W"], "cpu")
            lin = at.srgb_to_linear(np.clip(ren, 0, 1)) * gain_true.get(ci, 1.0)
            img = (at.linear_to_srgb(lin) * 255 + 0.5).astype(np.uint8)
            img[rid < 0] = (90, 90, 90)
            full = {r: (rid == k).astype(np.uint8) for k, r in enumerate(at.ROLES)}
            imgs[len(views)] = (img, full)
            views.append({"cam": f"{ci:02d}", "i": len(views), "t": 0.9 * t + 0.003, "K": c["K"], "R": c["R"], "t_w": c["t"],
                          "k1": 0.0, "ok": True, "swap": 0.0})
    cfg = at.TextureConfig(size=SIZE, device="cpu", hd_cams=(), cam_weight={}, chunk=100_000)
    cam_ids = [f"{ci:02d}" for ci in range(len(cams))]
    loader = lambda v: at.view_data(imgs[v["i"]][0], imgs[v["i"]][1], imgs[v["i"]][1], cfg.margin_frac)
    stores, stats = at.bake_samples(views, bodies, atlas, cfg, cam_ids, loader, "cpu")
    assert all(s["roles"][r]["iou"] > 0.8 for s in stats for r in at.ROLES)  # model silhouettes = masks here
    view_cam = np.array([int(v["cam"]) for v in views])
    results, rep = at.fuse_stores(stores, stats, view_cam, cam_ids, cfg, "cpu")
    for r in at.ROLES:
        rgb, conf, ncam, g = results[r]
        filled, alpha, src = at.fill_holes(atlas, rgb, conf, 10475)
        srgb = at.linear_to_srgb(filled)
        meas = alpha > 0.5
        assert meas.mean() > 0.6, (r, meas.mean())
        err = np.abs(srgb - gt[r]).max(1)
        # measured texels: nearest-splat rendering + bilinear resampling at ~1 cm pixels limits the accuracy
        assert np.median(err[meas]) < 0.03 and np.percentile(err[meas], 90) < 0.08, (r, np.percentile(err[meas], [50, 90]))
        assert np.isfinite(srgb).all() and (src[~meas] > 0).all()
        # exposure: camera 3's views need ~1/0.7 relative to the others
        g_cam = np.array([np.median(g[view_cam == ci, 1]) for ci in range(len(cams))])
        rel = g_cam[3] / np.median(np.delete(g_cam, 3))
        assert abs(rel - 1 / 0.7) < 0.07, (r, g_cam)
    # the partner never leaks in: lead texels hidden behind the follow from the front stay lead-coloured
    rgb, conf, ncam, _ = results["lead"]
    meas = conf > 0
    leak = np.abs(at.linear_to_srgb(rgb[meas]) - gt["lead"][meas]).max(1) > 0.2
    assert leak.mean() < 0.01, leak.mean()


@needs_model
def test_fill_holes_uses_mirror_then_mesh_diffusion(atlas):
    gt = _gt_colors(atlas, "lead")
    lin = at.srgb_to_linear(gt)
    conf = np.ones(atlas.n)
    m0 = np.load(au.model_npz(SMPLX_MODELS))
    P0 = atlas.surface(np.asarray(m0["v_template"], float))
    hole_l = (P0[:, 0] > 0.55)                       # left hand/forearm: right side measured -> mirror
    y0 = (np.asarray(m0["J_regressor"]) @ np.asarray(m0["v_template"]))[0, 1]  # pelvis height
    hole_both = (np.abs(P0[:, 0]) < 0.05) & (P0[:, 1] > y0 + 0.1) & (P0[:, 1] < y0 + 0.2) & (P0[:, 2] > 0)  # belly
    conf[hole_l | hole_both] = 0
    filled, alpha, src = at.fill_holes(atlas, lin, conf, len(m0["v_template"]), conf_ref=1.0)
    mir = atlas.mirror
    sel = hole_l & (mir >= 0) & ~hole_both[np.clip(mir, 0, None)]
    assert (src[sel] == 1).mean() > 0.95
    assert np.allclose(filled[sel], lin[mir[sel]], atol=1e-5)
    # the belly strip (its mirror is also a hole) is diffused from its surroundings on the mesh
    sel2 = hole_both & (src == 2)
    assert sel2.sum() > 50
    err = np.abs(at.linear_to_srgb(filled[sel2]) - gt[sel2]).max(1)
    assert np.median(err) < 0.12, np.median(err)


# ----------------------------------------------------------------------------------------- head orientation fix


@needs_model
def test_head_fix_neck_pose_is_exact_fk_and_fit_recovers_rotation():
    from dancecap import avatar_head as ah
    from dancecap.avatar_review import _look_at
    T = 6
    rng = np.random.default_rng(1)
    d = {"t": np.arange(T) / 30.0, "global_orient": rng.normal(0, 0.2, (T, 3)), "body_pose": rng.normal(0, 0.15, (T, 63)),
         "left_hand_pose": np.zeros((T, 45)), "right_hand_pose": np.zeros((T, 45)),
         "transl": np.tile((0.0, 1.1, 0.0), (T, 1)), "betas": np.zeros(16), "num_betas": 16, "flat_hand_mean": True}
    true = np.stack([np.array([0.05, 0.45, -0.1]) * (1 + 0.1 * k) for k in range(T)])  # ~26 deg yaw-ish
    pts, neck = ah.face_points(d, SMPLX_MODELS)
    d2 = dict(d)
    bp = d["body_pose"].copy()
    bp[:, 33:36] = ah.neck_pose_with(d, true)
    d2["body_pose"] = bp
    pts2, neck2 = ah.face_points(d2, SMPLX_MODELS)
    Rt = ah._rotvec_to_mat(true)
    pred = neck[:, None] + np.einsum("tij,tpj->tpi", Rt, pts - neck[:, None])
    # rigid rotation about the neck up to SMPL-X pose correctives (posedirs change with the neck angle, ~1 mm)
    assert np.abs(neck2 - neck).max() < 1e-5 and np.abs(pts2 - pred).max() < 3e-3
    # multi-view keypoints of the rotated head -> the fit recovers the rotation
    obs = [[] for _ in range(T)]
    K = np.array([[900.0, 0, 540], [0, 900, 960], [0, 0, 1]])
    for a in np.linspace(-1.2, 1.2, 6):
        C = np.array([3 * np.sin(a), 1.6, 3 * np.cos(a)])
        R, t = _look_at(C, np.array([0.0, 1.5, 0.0]))
        for k in range(T):
            X = pts2[k] @ R.T + t
            uv = X[:, :2] / X[:, 2:3] * 900 + K[:2, 2] + rng.normal(0, 0.5, (5, 2))
            obs[k].append({"cam": f"{a:.1f}", "K": K, "R": R, "t": t, "uv": uv, "conf": np.full(5, 0.9)})
    delta, info = ah.fit_head(pts, neck, obs, smooth=0.05, prior=0.005, max_iter=200)
    err_deg = np.degrees(np.linalg.norm(delta - true, axis=1))
    assert np.median(err_deg) < 2.0, err_deg
    assert info["median_err_after_cm"] < 0.3 < info["median_err_before_cm"]


def test_hair_model_separates_red_hair_from_skin(tmp_path):
    import json

    import cv2
    rng = np.random.default_rng(0)
    work, cam = tmp_path, "02"
    (work / "frames" / cam).mkdir(parents=True)
    for sub in ("masks", "hair"):
        (work / "dancers" / cam / sub).mkdir(parents=True)
    H, W = 120, 80
    frames = []
    for i in range(12):
        img = np.full((H, W, 3), (60, 60, 60), np.uint8)  # BGR background
        hair = np.zeros((H, W), np.uint8)
        hair[5:45, 20:60] = 255
        fol = np.zeros((H, W), np.uint8)
        fol[5:115, 15:65] = 255
        img[fol > 0] = (150, 175, 215) + rng.integers(-12, 12, (int((fol > 0).sum()), 3))     # skin-ish (BGR)
        img[hair > 0] = (30, 40, 190) + rng.integers(-15, 15, (int((hair > 0).sum()), 3))     # red hair (BGR)
        lead = np.zeros((H, W), np.uint8)
        lead[60:115, 0:12] = 255
        img[lead > 0] = (20, 20, 20)
        cv2.imwrite(str(work / "frames" / cam / f"{i:06d}.jpg"), img)
        cv2.imwrite(str(work / "dancers" / cam / "hair" / f"mask_{i:04d}_02.png"), hair)
        cv2.imwrite(str(work / "dancers" / cam / "masks" / f"mask_{i:04d}_02.png"), fol)
        cv2.imwrite(str(work / "dancers" / cam / "masks" / f"mask_{i:04d}_01.png"), lead)
        frames.append({"file": f"{i:06d}.jpg", "t": i / 30})
    (work / "frames" / cam / "frames.json").write_text(json.dumps({"frames": frames}))
    lut, rep = at.hair_model(work, cam, frame_step=1)
    assert rep["status"] == "ok" and rep["hair_recall"] > 0.9 and rep["other_false_pos"] < 0.1, rep
    img = cv2.imread(str(work / "frames" / cam / "000003.jpg"))[:, :, ::-1].copy()
    p = at.hair_probability(img, lut)
    assert np.median(p[10:40, 25:55]) > 0.8 and np.median(p[60:110, 20:60]) < 0.2
