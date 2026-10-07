"""Tests for dancecap.room (+ room_mesh): geometry helpers, PLY io, masks, software renderer, and an opt-in GPU test
that fits a small synthetic textured room with the real training loop (floor clamp + appearance + held-out PSNR).

    .venv-nopo4d\\Scripts\\python.exe -m pytest tests/test_room.py -q
    set DANCECAP_GPU_TESTS=1 & prototype\\run_cuda.cmd -m pytest tests/test_room.py -q   (GPU test: needs MSVC env)
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dancecap import room  # noqa: E402
from dancecap.room import (RoomConfig, View, _scale_K, _unscale_K, fit_affine, fit_floor, load_ply,  # noqa: E402
                           masked_metrics, reflect_mask, save_ply)
from dancecap.room_mesh import barycentric, bilinear, depth_ok, raster_uv, render_mesh  # noqa: E402
from dancecap.take import PACKAGE_ROOT  # noqa: E402

WORK = PACKAGE_ROOT / "work" / "larissa_kadu"


def _look_at(C, target, up=np.array([0, 1.0, 0])):
    z = target - C
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return R, -R @ C


def test_fit_floor_tilted_plane_with_reflections():
    rng = np.random.default_rng(0)
    n_true = np.array([0.05, 1.0, -0.03])
    n_true /= np.linalg.norm(n_true)
    xz = rng.uniform(-6, 6, (3000, 2))
    P = np.stack([xz[:, 0], np.zeros(len(xz)), xz[:, 1]], 1)
    P[:, 1] = -(n_true[0] * P[:, 0] + n_true[2] * P[:, 2] - 0.4) / n_true[1] + rng.normal(0, 0.005, len(P))
    walls = np.concatenate([rng.uniform(-6, 6, (2000, 1)), rng.uniform(0, 4, (2000, 1)), np.full((2000, 1), -6.0)], 1)
    mirror = walls.copy()
    mirror[:, 1] = -mirror[:, 1] - 0.5  # reflections under the floor
    allp = np.concatenate([P, walls, mirror[:300]])
    n, d, k = fit_floor(allp, np.array([0, 1.0, 0]), np.zeros(3))
    assert math.degrees(math.acos(abs(n @ n_true))) < 0.5
    assert abs(-d - 0.4) < 0.02 and k > 2500


def test_scale_K_roundtrip_and_pixel_centres():
    K = np.array([[1600.0, 0, 539.5], [0, 1600, 959.5], [0, 0, 1]])
    K2 = _scale_K(K, 2)
    assert np.allclose(K2[:2, 2], [269.5, 479.5]) and np.isclose(K2[0, 0], 800)
    assert np.allclose(_unscale_K(K2, 2), K)
    v = View("x", "01", "kf", 0, K2, np.eye(3), np.zeros(3), 540, 960, 0, 2, "m")
    assert np.allclose(v.K_gs()[:2, 2], [270, 480])  # gsplat / COLMAP: pixel centres at +0.5


def test_reflect_mask_strip_under_standing_person_only():
    m = np.zeros((400, 200), bool)
    m[100:300, 80:120] = True       # standing person (200 px tall)
    m[330:360, 10:60] = True        # seated spectator (short): no strip
    out = reflect_mask(m, 0.4)
    assert out[300:380, 80:120].all() and not out[381:, 80:120].any()
    assert not out[300:330, 10:60].any() and not out[361:, 10:60].any()
    assert (out & ~m).sum() == 80 * 40


def test_ply_roundtrip_standard_layout(tmp_path):
    rng = np.random.default_rng(1)
    N = 50
    d = {"means": rng.normal(size=(N, 3)), "quats": rng.normal(size=(N, 4)), "scales": rng.normal(size=(N, 3)),
         "opacities": rng.normal(size=N), "sh0": rng.normal(size=(N, 1, 3)), "shN": rng.normal(size=(N, 15, 3))}
    p = tmp_path / "s.ply"
    save_ply(p, d["means"], d["quats"], d["scales"], d["opacities"], d["sh0"], d["shN"])
    from plyfile import PlyData
    names = [x.name for x in PlyData.read(str(p))["vertex"].properties]
    assert names[:9] == ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    assert names[-8:] == ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    assert len([n for n in names if n.startswith("f_rest_")]) == 45
    # channel-major f_rest (Kerbl et al.): f_rest_0..14 are the red coefficients
    v = PlyData.read(str(p))["vertex"]
    assert np.allclose(np.asarray(v["f_rest_1"]), d["shN"][:, 1, 0], atol=1e-6)
    assert np.allclose(np.asarray(v["f_rest_15"]), d["shN"][:, 0, 1], atol=1e-6)
    back = load_ply(p)
    for k in d:
        assert np.allclose(back[k].reshape(d[k].shape), d[k], atol=1e-6), k


def test_metrics_and_affine_colour_correction():
    rng = np.random.default_rng(2)
    gt = rng.uniform(0, 1, (64, 64, 3)).astype(np.float32)
    valid = np.ones((64, 64), bool)
    valid[:, :20] = False
    render = gt.copy()
    render[:, :20] = 0  # garbage only where masked
    m = masked_metrics(render, gt, valid)
    assert m["psnr"] > 90 and m["ssim"] > 0.99
    tinted = np.clip(gt * np.array([0.8, 1.0, 1.1]) + 0.02, 0, 1)
    cc = fit_affine(gt, tinted, valid)
    assert masked_metrics(cc, tinted, valid)["psnr"] > 35


def test_depth_ok_drops_edges_and_transparent():
    d = np.full((20, 20), 3.0, np.float32)
    d[:, 10:] = 6.0
    a = np.ones_like(d)
    a[:3] = 0.5
    ok = depth_ok(d, a)
    assert not ok[:3].any() and not ok[:, 9:11].any() and ok[5:, :8].all() and ok[5:, 12:].all()


def test_barycentric_bilinear_and_uv_raster():
    a, b, c = np.array([[0.0, 0]]), np.array([[1.0, 0]]), np.array([[0.0, 1]])
    w = barycentric(np.array([[0.25, 0.25]]), a, b, c)
    assert np.allclose(w, [[0.5, 0.25, 0.25]])
    img = np.arange(12, dtype=np.float32).reshape(3, 4, 1)
    assert np.isclose(bilinear(img, np.array([1.5]), np.array([1.0]))[0, 0], 5.5)
    uv = np.array([[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]])
    ids = raster_uv(uv, np.array([[0, 1, 2], [1, 3, 2]]), 64)
    assert (ids >= 0).mean() > 0.55 and set(np.unique(ids)) == {-1, 0, 1}


def test_render_mesh_occlusion_and_colour():
    # two quads facing the camera: red at z=5 in front of blue at z=8 (both larger than the view)
    def quad(z, s):
        V = np.array([[-s, -s, z], [s, -s, z], [s, s, z], [-s, s, z]], float)
        return V, np.array([[0, 1, 2], [0, 2, 3]])
    V1, F1 = quad(5.0, 1.0)
    V2, F2 = quad(8.0, 10.0)
    V = np.concatenate([V1, V2])
    F = np.concatenate([F1, F2 + 4])
    col = np.concatenate([np.tile([1.0, 0, 0], (4, 1)), np.tile([0, 0, 1.0], (4, 1))]).astype(np.float32)
    K = np.array([[100.0, 0, 49.5], [0, 100, 49.5], [0, 0, 1]])
    img = render_mesh(V, F, np.eye(4), K, 100, 100, vcol=col)
    assert (img[50, 50] == [255, 0, 0]).all()   # red square spans +-20 px around the centre
    assert (img[5, 5] == [0, 0, 255]).all()     # blue behind it at the corners


# ------------------------------------------------------------------------------------------- bundle / presets

def test_view_resized_matches_area_downscale_geometry():
    K = np.array([[600.0, 0, 239.5], [0, 600, 431.5], [0, 0, 1]])
    v = View("x", "01", "win", 0, K, np.eye(3), np.zeros(3), 480, 864, 0.0, 1, "m")
    assert v.resized(1000) is v
    r = v.resized(432)
    assert (r.W, r.H) == (240, 432) and np.isclose(r.scale, 2.0)
    assert np.allclose(r.K, _scale_K(K, 2))  # integer factor: identical to the old integer downscale
    # a 3D point projects to the same normalised image position at both resolutions
    X = np.array([0.3, -0.2, 4.0])
    p, q = K @ X, r.K @ X
    assert np.allclose(((p[:2] / p[2]) + 0.5) / [480, 864], ((q[:2] / q[2]) + 0.5) / [240, 432])
    back = View.from_json(json.loads(json.dumps(r.to_json())))
    assert back.W == r.W and np.allclose(back.K, r.K) and back.label == "x"


def test_bundle_roundtrip_and_skip_when_fresh(tmp_path, monkeypatch):
    work = tmp_path / "take"
    (work / "frames" / "01").mkdir(parents=True)
    (work / "people_masks" / "01").mkdir(parents=True)
    rng = np.random.default_rng(3)
    K = np.array([[300.0, 0, 79.5], [0, 300, 59.5], [0, 0, 1]])
    views = []
    for i in range(3):
        cv2.imwrite(str(work / "frames" / "01" / f"{i:06d}.jpg"), rng.integers(0, 255, (120, 160, 3), np.uint8))
        m = np.zeros((120, 160), np.uint8)
        m[40:80, 60:90] = 255
        cv2.imwrite(str(work / "people_masks" / "01" / f"{i:06d}.png"), m)
        views.append(View(f"frames/01/{i:06d}.jpg", "01", "win", float(i), K, np.eye(3), np.array([0, 0, i * 0.1]),
                          160, 120, 0.02 if i == 1 else 0.0, 1, f"people_masks/01/{i:06d}.png",
                          "test" if i == 2 else "train"))
    geo = {"floor_n": np.array([0, 1.0, 0]), "floor_d": 0.0, "floor_inliers": 10, "tilt_vs_world_deg": 0.5,
           "box_lo": np.array([-1.0, 0, -1]), "box_hi": np.array([1.0, 2, 1]), "origin": np.zeros(3),
           "up": np.array([0, 1.0, 0]), "scene_scale": 2.0,
           "world": {"s": 1.0, "R": np.eye(3), "t": np.zeros(3)}, "mirror": None,
           "reflectors": [{"name": "floor", "n": np.array([0, 1.0, 0]), "d": 0.0, "rect": None, "schlick": True,
                           "beta_init": 0.15, "front_margin": 0.03}]}
    P = rng.normal(size=(50, 3))
    monkeypatch.setattr(room, "collect_views", lambda w, cfg: views)
    monkeypatch.setattr(room, "room_geometry", lambda w, vs, cfg: geo)
    monkeypatch.setattr(room, "sfm_points", lambda w: (P, np.zeros((50, 3), np.uint8), np.zeros(50), np.full(50, 3)))
    bundle = work / "room" / "bundle"
    man = room.write_bundle(work, bundle, RoomConfig(), log=lambda *a: None)
    assert man["n_views"] == 3 and man["n_train"] == 2 and man["bytes"] > 0
    calls = []
    assert room.write_bundle(work, bundle, RoomConfig(), log=calls.append)["created"] == man["created"]
    assert "up to date" in calls[0]
    monkeypatch.undo()  # the bundle must be readable without any of the patched (pycolmap) helpers
    vb, gb, _ = room.load_bundle(bundle)
    assert [v.src for v in vb] == [v.name for v in views] and all(v.k1 == 0 for v in vb)
    assert isinstance(gb["floor_n"], np.ndarray) and np.allclose(gb["world"]["R"], np.eye(3))
    assert gb["reflectors"][0]["rect"] is None and isinstance(gb["reflectors"][0]["n"], np.ndarray)
    Pb = room.sfm_points(bundle)[0]
    assert np.allclose(Pb, P, atol=1e-6)
    img, valid = room.load_view(bundle, vb[0], RoomConfig(reflect_frac=0))
    assert img.shape == (120, 160, 3) and not valid[50:70, 65:85].any() and valid[:30].all()
    small = vb[0].resized(80)
    img, valid = room.load_view(bundle, small, RoomConfig(reflect_frac=0))
    assert img.shape == (60, 80, 3) and valid.shape == (60, 80)


def test_presets_and_pod_plan(tmp_path):
    for name in room.PRESETS:
        cfg = room.preset(name)
        assert not set(room.PRESETS[name]) & set(room.PREP_FIELDS)  # one bundle serves every preset
        assert cfg.refine_start < cfg.refine_stop <= cfg.iters
    assert room.preset("smoke").iters <= 500 and room.preset("pod").max_side >= 1920
    assert room.preset("smoke", iters=None).iters == 300 and room.preset("smoke", iters=50).iters == 50
    with pytest.raises(ValueError):
        room.preset("nope")
    plan = room.pod_plan(tmp_path / "larissa_kadu")
    cmds = " ".join(plan["pod_commands"]["setup"]) + plan["pod_commands"]["train"] + plan["pod_commands"]["post"]
    assert "gsplat==1.5.3" in cmds and "--stages train,quest" in cmds and "--stages mesh,eval" in cmds
    assert "--preset pod" in cmds and "C:" not in cmds and "\\" not in cmds  # Linux paths only
    assert "room/splat_quest.ply" in plan["outputs"] and "review/room_renders.jpg" in plan["outputs"]


# ------------------------------------------------------------------------------------- GPU end-to-end (opt-in)

FLOOR_REFL = 0.35


def _synthetic_room(n_cams: int = 24, W: int = 160, H: int = 120):
    """Box room 8 x 3 x 8 m (world = SfM frame, Y up, floor y = 0) with checker walls/ceiling and a dark GLOSSY floor
    (diffuse + FLOOR_REFL x one mirror bounce), ray-cast from a ring of cameras at 1.6 m; per-image gains."""
    planes = [(np.array([0, 1.0, 0]), 0.0), (np.array([0, -1.0, 0]), 3.0)] + \
        [(s * np.eye(3)[i], 4.0) for i in (0, 2) for s in (1.0, -1.0)]

    def cast(O, D):
        best = np.full(len(D), np.inf)
        nrm = np.zeros_like(D)
        for n, off in planes:
            den = D @ n
            s = -(O @ n + off) / np.where(np.abs(den) < 1e-9, 1e-9, den)
            hit = (s > 1e-4) & (s < best)
            best = np.where(hit, s, best)
            nrm[hit] = n
        return best, O + best[:, None] * D, nrm

    def shade(P, nrm):
        f = np.floor(P[:, 0] * 2) + np.floor(P[:, 2] * 2) + np.floor(P[:, 1] * 2)
        base = np.where((f % 2)[:, None] == 0, [0.85, 0.75, 0.6], [0.2, 0.3, 0.45])
        col = np.clip(base * (0.5 + 0.5 * np.abs(nrm)), 0, 1)
        floor = nrm[:, 1] > 0.5
        col[floor] = np.where((f[floor] % 2)[:, None] == 0, [0.12, 0.12, 0.12], [0.05, 0.05, 0.06])
        return col, floor

    rows, pts = [], []
    for c in range(n_cams):
        a = 2 * math.pi * c / n_cams
        C = np.array([2.5 * math.cos(a), 1.3 + 0.6 * (c % 2), 2.5 * math.sin(a)])
        tgt = np.array([-1.5 * math.cos(a + 0.3), 0.5, -1.5 * math.sin(a + 0.3)])
        R, t = _look_at(C, tgt)
        K = np.array([[110.0, 0, W / 2 - 0.5], [0, 110, H / 2 - 0.5], [0, 0, 1]])
        yy, xx = np.mgrid[0:H, 0:W]
        d = np.stack([(xx - K[0, 2]) / K[0, 0], (yy - K[1, 2]) / K[1, 1], np.ones(xx.shape)], -1).reshape(-1, 3) @ R
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        s, P, nrm = cast(np.tile(C, (len(d), 1)), d)
        col, floor = shade(P, nrm)
        dr = d[floor] * np.array([1, -1.0, 1])
        _, P2, n2 = cast(P[floor] + 1e-3 * dr, dr)
        col[floor] += FLOOR_REFL * shade(P2, n2)[0]
        gain = 1.0 + 0.12 * (c % 3 - 1)
        img = (np.clip(col.reshape(H, W, 3) * gain, 0, 1) * 255).astype(np.uint8)
        rows.append((C, R, t, K, img))
        pts.append(P[::23])
    return rows, np.concatenate(pts)


@pytest.mark.skipif(os.environ.get("DANCECAP_GPU_TESTS") != "1", reason="opt-in GPU test (DANCECAP_GPU_TESTS=1)")
def test_gpu_synthetic_glossy_room(tmp_path):
    """Real training loop on a synthetic glossy-floor room: with the planar-reflector model the floor reflectance is
    recovered, nothing is reconstructed under the floor and the floor depth lies on the floor plane; without it,
    the clamp still keeps the floor clean (baseline: no reflections, no planar depth prior)."""
    import torch
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    rows, P = _synthetic_room()
    views, imgs = [], {}
    for c, (C, R, t, K, img) in enumerate(rows):
        name = f"kf/{c:02d}.png"
        views.append(View(name, f"{c:02d}", "kf", float(c), K, R, t, img.shape[1], img.shape[0], 0.0, 1,
                          f"mask/{c:02d}.png", "test" if c in (3, 10, 17) else "train"))
        imgs[name] = img
    world = {"s": 1.0, "R": np.eye(3), "t": np.zeros(3)}
    orig = room.load_view, room.sfm_points
    rng = np.random.default_rng(0)
    Pm = P.copy()
    Pm[:, 1] = -Pm[:, 1]  # an upside-down copy of the room under the floor in the init (the clamp must remove it)
    allP = np.concatenate([P, Pm])
    room.load_view = lambda work, v, cfg: (imgs[v.name], np.ones(imgs[v.name].shape[:2], bool))
    room.sfm_points = lambda work: (allP, rng.integers(0, 255, allP.shape).astype(np.uint8), np.zeros(len(allP)),
                                    np.full(len(allP), 5))
    res = {}
    from dancecap import gpu
    lock = gpu.exclusive("test_room gpu")  # never share the 11 GB card with another stage's job
    lock.__enter__()
    try:
        for reflections in (True, False):
            cfg = RoomConfig(iters=3000, cap_max=80_000, sh_degree=1, refine_stop=2000, max_side=400, kf_weight=1.0,
                             reflections=reflections, refl_start=300, plane_depth_w=0.1 if reflections else 0.0)
            geo = {"floor_n": np.array([0, 1.0, 0]), "floor_d": 0.0, "scene_scale": 3.0, "world": world,
                   "box_lo": np.array([-4.0, 0, -4]), "box_hi": np.array([4.0, 3, 4]), "mirror": None,
                   "reflectors": [{"name": "floor", "n": np.array([0, 1.0, 0]), "d": 0.0, "rect": None,
                                   "schlick": True, "beta_init": cfg.floor_beta, "front_margin": 0.03}]}
            splat, app, refl, info = room.train(tmp_path, views, geo, cfg, log=lambda *a, **k: None)
            ev = room.evaluate(tmp_path, splat, app, refl, views, cfg)
            print(reflections, ev["summary"]["all"], refl.betas(), room.floor_depth_check(splat, views, geo))
            res[reflections] = (splat, refl, ev, room.floor_depth_check(splat, views, geo), app)
    finally:
        room.load_view, room.sfm_points = orig
        lock.__exit__(None, None, None)
    splat, refl, ev, fd, app = res[True]
    assert (splat.numpy()["means"][:, 1] < -RoomConfig().floor_margin).sum() == 0
    assert ev["summary"]["all"]["psnr_cc"] > 22, ev["summary"]
    b = refl.betas()["floor"]
    # synthetic reflectance is FLOOR_REFL at every angle; the camera rays meet the floor at cos ~0.3-0.7
    assert 0.2 < b["f0"] + b["fresnel_gain"] * 0.3 ** 5 < 0.5, b
    assert fd["within_10cm"] > 0.8, fd
    s0, _, ev0, fd0, _ = res[False]
    assert (s0.numpy()["means"][:, 1] < -RoomConfig().floor_margin).sum() == 0
    assert ev["summary"]["all"]["psnr_cc"] > ev0["summary"]["all"]["psnr_cc"] + 0.5
    assert fd["floor_pixels"] > 0.8 * fd0["floor_pixels"], (fd, fd0)  # opaque floor (no see-through + reflection)
    A = app.A.detach().cpu().numpy()
    assert np.abs(A.mean(0) - np.eye(3)).max() < 0.08


# ----------------------------------------------------------------------------------------- real outputs

def _need(p: Path):
    if not p.exists():
        pytest.skip(f"{p} not produced yet (python -m dancecap room larissa_kadu)")


def test_real_outputs_contract_and_budgets():
    pytest.importorskip("open3d")
    pytest.importorskip("plyfile")
    torch = pytest.importorskip("torch")
    out = WORK / "room"
    for f in ("splat_full.ply", "splat_quest.ply", "room_mesh.ply", "room_mesh.obj", "room_mesh.mtl",
              "room_mesh_tex.png", "room_floor_tex.png", "transform.json", "report.json", "eval.json"):
        _need(out / f)
    rep = json.loads((out / "report.json").read_text())
    if not all(k in rep for k in ("train", "quest", "mesh", "eval")):
        pytest.skip(f"{out / 'report.json'} is from an incomplete run (stages {sorted(rep)})")
    cfg = RoomConfig()
    q = load_ply(out / "splat_quest.ply")
    assert len(q["means"]) <= cfg.quest_max and q["shN"].shape[1] == (cfg.quest_sh + 1) ** 2 - 1
    tr = json.loads((out / "transform.json").read_text())
    assert tr["frame"] == "sfm" and abs(tr["world_from_sfm"]["s"]) > 0
    w = {"s": tr["world_from_sfm"]["s"], "R": np.array(tr["world_from_sfm"]["R"]),
         "t": np.array(tr["world_from_sfm"]["t"])}
    n, d = np.array(tr["floor_plane_sfm"]["n"]), tr["floor_plane_sfm"]["d"]
    assert abs(tr["floor_plane_sfm"]["world_height_at_origin_m"]) < 0.05
    import open3d as o3d
    mesh = o3d.io.read_triangle_mesh(str(out / "room_mesh.ply"))
    assert len(mesh.triangles) <= cfg.mesh_max_tris
    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.triangles)
    assert np.mean(V @ n + d < -0.05) < 0.01  # no mirrored floor under the floor
    Vw = (w["s"] * (w["R"] @ V.T)).T + w["t"]
    # the floor is a coarse 25 cm quad grid (few vertices) next to a dense TSDF: weight the layers by surface AREA
    cen = Vw[F].mean(1)
    area = 0.5 * np.linalg.norm(np.cross(Vw[F[:, 1]] - Vw[F[:, 0]], Vw[F[:, 2]] - Vw[F[:, 0]]), axis=1)
    near = (np.abs(cen[:, 0]) < 3) & (np.abs(cen[:, 2]) < 3)
    hist, edges = np.histogram(cen[near, 1], bins=np.arange(-1.0, 4.0, 0.05), weights=area[near])
    assert abs(edges[np.argmax(hist)] + 0.025) < 0.1  # the largest layer near the dancers is the floor at y ~ 0
    mtl = (out / "room_mesh.mtl").read_text()
    assert "room_mesh_tex.png" in mtl and "room_floor_tex.png" in mtl
    full = load_ply(out / "splat_full.ply")
    assert np.mean(full["means"] @ n + d < -cfg.floor_margin) == 0
    if "mirror" in tr and tr["mirror"].get("is_mirror"):
        m = tr["mirror"]["sfm"]
        geo = {"floor_n": n, "floor_d": d, "world": w,
               "mirror": {"n": np.array(m["n"]), "d": m["d"], "along": np.array(m["along"]),
                          "a_lo": m["along_range"][0], "a_hi": m["along_range"][1], "top_m": tr["mirror"]["top_m"],
                          "c": np.array(json.loads((out / "report.json").read_text())["geometry"]["mirror"]["c"])}}
        assert room.behind_mirror(torch.as_tensor(full["means"]), geo, cfg.floor_margin).float().mean() < 1e-3
    ev = json.loads((out / "eval.json").read_text())
    assert ev["full"]["all"]["n"] >= 30 and ev["quest"]["all"]["psnr"] > 15
