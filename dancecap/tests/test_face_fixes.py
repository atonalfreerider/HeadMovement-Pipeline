"""Face fixes after the 2026-10-07 review: model selection ships exactly the validated candidate (rest / rest_closed,
relative mouth-opening tolerance, co-visible per-frame check), texture fixes (lip colour, video-look targets measured
on the texture, gradient-domain clean-up, seam metrics, mouth cap, island padding, streak removal) and the Unity face
export (DCXF blob, motion jaw / eye patch, skin mouth cap, albedo variant). Tiny synthetic inputs only; the SMPL-X
tests are skipped without the model.

    .venv-nopo4d\\Scripts\\python.exe -m pytest dancecap/tests/test_face_fixes.py -q
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dancecap import avatar_face as af  # noqa: E402
from dancecap import face_common as fc  # noqa: E402
from dancecap import face_expr as fx  # noqa: E402

needs_model = pytest.mark.skipif(not fc.smplx_model_npz().exists(), reason="SMPL-X model not installed")


# ------------------------------------------------------------------------------------------------ face_expr selection


def _pooled(**modes):
    """mode=(mouth NME % IOD, opening error) -> pooled hold-out dict."""
    return {m: {"mouth_nme_pct_iod_median": n, "mouth_open_abs_err_median": o} for m, (n, o) in modes.items()}


def _check(cc, tb, covis=None):
    c = {"pooled": cc, "covisible": covis or {"views": 0, "testable": False, "pooled": {}}}
    return {"cross_camera": c, "time_blocks": {"pooled": tb, "covisible": {"views": 0, "testable": False, "pooled": {}}}}


def test_relative_open_tolerance_rejects_the_lip_parting_rest_and_takes_rest_closed():
    cfg = fx.FaceConfig()
    # the 2026-10-06 numbers: free rest 6.71 % but opening error 0.0589 vs neutral 0.0205 (x2.9); rest_closed 6.86 / 0.0173
    cc = _pooled(neutral=(9.6, 0.0205), rest=(6.71, 0.0589), rest_closed=(6.86, 0.0173), final=(7.94, 0.0388),
                 final_expr=(7.94, 0.0388))
    tb = _pooled(neutral=(9.6, 0.0205), rest=(6.72, 0.0537), rest_closed=(6.8, 0.0172), final=(7.95, 0.0306),
                 final_expr=(7.95, 0.0305))
    a = fx.adoption(_check(cc, tb), cfg)
    assert a["rest"] and a["rest_kind"] == "rest_closed" and a["shipped"] == "rest_closed"
    assert not a["rest_candidates"]["rest"]["eligible"] and a["rest_candidates"]["rest_closed"]["eligible"]
    assert a["rest_open_err_allowed"] == pytest.approx(max(1.5 * 0.0205, 0.0205 + 0.005), abs=1e-4)
    assert not a["expression"] and not a["per_frame_capture_testable"] and "NOT testable" in a["reason"]
    # without the closed candidate the face stays neutral (the absolute 0.05 tolerance used to let the rest through)
    cc.pop("rest_closed"), tb.pop("rest_closed")
    b = fx.adoption(_check(cc, tb), cfg)
    assert not b["rest"] and b["shipped"] == "neutral" and "opening error" in b["reason"]


def test_covisible_views_gate_the_per_frame_layer():
    cfg = fx.FaceConfig()
    cc = _pooled(neutral=(8.0, 0.03), rest=(6.0, 0.03), final=(5.8, 0.03), final_expr=(5.8, 0.03))
    tb = _pooled(neutral=(8.0, 0.03), rest=(6.0, 0.03), final=(5.0, 0.03), final_expr=(5.0, 0.03))
    worse = {"views": 12, "testable": True, "pooled": _pooled(rest=(6.0, 0.03), final_expr=(6.5, 0.03))}
    better = {"views": 12, "testable": True, "pooled": _pooled(rest=(6.0, 0.03), final_expr=(5.5, 0.03))}
    assert not fx.adoption(_check(cc, tb, worse), cfg)["expression"]
    a = fx.adoption(_check(cc, tb, better), cfg)
    assert a["expression"] and a["per_frame_capture_testable"] and a["shipped"] in ("final", "final_expr")


def test_ship_rest_returns_exactly_the_adopted_candidates_fit():
    cfg = fx.FaceConfig()
    fits = {m: {"e0": np.full(10, v), "jaw0": np.radians(d)} for m, v, d in
            (("frame", 0.1, 1.0), ("rest", 0.2, 2.0), ("rest_closed", 0.3, 20.0))}
    e, j = fx.ship_rest(fits, {"rest": True, "rest_kind": "rest_closed", "expression": False}, cfg)
    assert np.allclose(e, 0.3) and j == pytest.approx(np.radians(cfg.jaw_max_deg))    # jaw clipped like the hold-out
    e, j = fx.ship_rest(fits, {"rest": True, "rest_kind": "rest", "expression": False}, cfg)
    assert np.allclose(e, 0.2) and j == pytest.approx(np.radians(2.0))
    e, j = fx.ship_rest(fits, {"rest": True, "rest_kind": "rest", "expression": True}, cfg)
    assert np.allclose(e, 0.1)                                     # the per-frame layer's anchor (as in final_expr)
    e, j = fx.ship_rest(fits, {"rest": False, "expression": False}, cfg)
    assert not e.any() and j == 0.0


def test_holdout_scores_every_shippable_candidate_and_counts_covisible_views(monkeypatch):
    """holdout_check builds a candidate for every name adoption() can ship, from the same fit modes build() uses."""
    T, E = 20, 10
    calls = []

    def fake_fit(prep, cfg, train=None, mode="frame"):
        calls.append(mode)
        v = {"frame": 0.1, "rest": 0.2, "rest_closed": 0.3, "neutral": 0.0}[mode]
        return {"expr": np.full((T, E), v), "jaw": np.zeros(T), "e0": np.full(E, v), "jaw0": 0.0, "_world": None,
                "_proj": None}

    def fake_err(prep, cfg, expr, jaw, idx, world=None, proj=None):
        v = float(expr[0, 0])
        n = len(idx)
        return {"n": n, "nme_mouth_med": v, "_nme": np.full(n, 0.05 + v), "_open": (np.full(n, v), np.zeros(n)),
                "_width": (np.full(n, 0.5), np.full(n, 0.5))}
    monkeypatch.setattr(fx, "fit_aligned", fake_fit)
    monkeypatch.setattr(fx, "aligned_errors", fake_err)
    obs = {"k": np.array([0, 1, 2, 3, 4, 5, 0, 1, 9, 10, 11, 12]), "q": np.ones(12),
           "cam": np.array(["01"] * 6 + ["02"] * 6)}
    prep = {"obs": obs, "grid_t": np.arange(T) / 30, "fps": 30.0}
    cfg = fx.FaceConfig(loo_min_views=3, covis_min_views=2)
    splits = [(f"cam{c}", obs["cam"] != c, obs["cam"] == c) for c in ("01", "02")]
    ch = fx.holdout_check(prep, cfg, splits)
    assert {"neutral", "rest", "rest_closed", "frame", "final", "final_expr"} <= set(ch["pooled"])
    assert {"rest", "rest_closed", "frame"} <= set(calls)
    assert ch["splits"]["cam01"]["covisible_views"] == 2 and ch["splits"]["cam02"]["covisible_views"] == 2
    assert ch["covisible"]["views"] == 4 and ch["covisible"]["testable"]
    assert ch["pooled"]["rest_closed"]["mouth_nme_pct_iod_median"] == pytest.approx(35.0)


def test_gaze_stylisation_follows_a_lowered_head_only():
    from scipy.spatial.transform import Rotation
    cfg = fx.FaceConfig()
    T = 60
    ang = np.r_[np.zeros(20), np.full(20, 30.0), np.full(20, -20.0)]          # level, 30 deg down, 20 deg up
    Rk = Rotation.from_rotvec(np.stack([np.radians(ang), np.zeros(T), np.zeros(T)], 1)).as_matrix()
    g = fx.gaze_stylisation(Rk, 30.0, cfg)
    assert abs(g["head_pitch_down_deg"][30] - 30.0) < 0.5 and abs(g["head_pitch_down_deg"][50] + 20.0) < 0.5
    assert np.degrees(g["eye_pitch"][30]) == pytest.approx(0.4 * (30 - 5), abs=0.3)   # + x rotation = eyes down
    assert g["droop"][30] == pytest.approx((30 - 10) / 40 * 0.35, abs=0.01)
    assert g["eye_pitch"][5] == 0 and g["droop"][5] == 0 and g["eye_pitch"][55] == 0 and g["droop"][55] == 0
    # + x rotation of the rest eye direction (+z) points it down (y < 0): the sign the eye joints need
    assert (Rotation.from_rotvec([0.3, 0, 0]).as_matrix() @ [0, 0, 1.0])[1] < 0


@needs_model
def test_rest_closed_keeps_the_lip_gap_while_fitting_the_smile():
    import torch

    from dancecap.avatar_face import HeadModel
    from scipy.spatial.transform import Rotation
    cfg = fx.FaceConfig(iters=200)
    emb = fc.LandmarkEmbedding()
    head = HeadModel(np.zeros(16, np.float32), cfg.n_expr)
    blfn = fx.batched_landmarks(head, emb)
    e_true = np.zeros(cfg.n_expr)
    e_true[0], e_true[1] = 1.5, 1.0                    # a smile that also parts the lips
    T = 6
    with torch.no_grad():
        L = blfn(torch.tensor(np.tile(e_true, (T, 1)), dtype=torch.float32), torch.zeros((T, 3))).numpy().astype(float)
        gap = lambda e: float(fx.lip_gap_3d(torch, blfn(torch.tensor(e[None], dtype=torch.float32), torch.zeros((1, 3))))[0])
    K = np.array([[2000.0, 0, 500], [0, 2000, 500], [0, 0, 1]])
    obs = {k: [] for k in ("k", "cam", "i", "uv", "iod", "angle", "raw_nme", "K", "R", "tc", "k1")}
    for k in range(T):
        for yaw in (-0.3, 0.2):
            R = Rotation.from_rotvec([0, np.pi - yaw, 0]).as_matrix()
            C = np.array([2 * np.sin(yaw), L[k, :, 1].mean(), 2 * np.cos(yaw)])
            t = -R @ C
            uv, _ = fc.project(L[k], K, R, t)
            for key, val in (("k", k), ("cam", "01" if yaw < 0 else "02"), ("i", k), ("uv", uv),
                             ("iod", float(np.linalg.norm(uv[emb.index([36])[0]] - uv[emb.index([45])[0]]))),
                             ("angle", 10.0), ("raw_nme", 0.05), ("K", K), ("R", R), ("tc", t), ("k1", 0.0)):
                obs[key].append(val)
    obs = {k: np.array(v) for k, v in obs.items()}
    obs["q"] = np.ones(len(obs["k"]))
    prep = {"role": "follow", "emb": emb, "head": head, "grid_t": np.arange(T) / 30, "fps": 30.0,
            "Rk": np.tile(np.eye(3), (T, 1, 1)), "tk": np.zeros((T, 3)), "D": np.zeros((52, 3)), "has_D": False,
            "obs": obs}
    torch.set_num_threads(1)
    rs = fx.fit_aligned(prep, cfg, mode="rest")
    rc = fx.fit_aligned(prep, cfg, mode="rest_closed")
    g0, g_true, g_rs, g_rc = gap(np.zeros(cfg.n_expr)), gap(e_true), gap(rs["e0"]), gap(rc["e0"])
    assert g_true > g0 + 0.002 and g_rs > g0 + 0.001             # the free rest parts the lips like the data
    assert abs(g_rc - g0) < 0.0005                                # the closed rest keeps the neutral gap (< 0.5 mm)
    idx = np.arange(len(obs["k"]))
    neu = fx.aligned_errors(prep, cfg, np.zeros((T, cfg.n_expr)), np.zeros(T), idx, rs["_world"], rs["_proj"])
    clo = fx.aligned_errors(prep, cfg, rc["expr"], rc["jaw"], idx, rs["_world"], rs["_proj"])
    assert clo["nme_mouth_med"] < neu["nme_mouth_med"]            # it still fits part of the smile


# ------------------------------------------------------------------------------------------------ avatar_face fixes


def fake_atlas(S: int = 64, uncovered: bool = False):
    """As test_face_build.fake_atlas: every texel (optionally minus a 6 x 6 hole) is a head texel, face (bit 2) in the
    centre, mirror = left/right flip, surface(V) = (x, y, z) from the texel position (x = column - centre)."""
    yy, xx = np.mgrid[0:S, 0:S]
    keep = np.ones((S, S), bool)
    if uncovered:
        keep[30:36, 30:36] = False
    yx = np.stack([yy[keep], xx[keep]], 1).astype(np.int32)
    n = len(yx)
    lin = -np.ones((S, S), np.int64)
    lin[keep] = np.arange(n)
    face = ((np.abs(xx - S / 2 + 0.5) < S * 0.3) & (yy > S * 0.2) & (yy < S * 0.85))[keep]
    region = (1 + 2 * face).astype(np.uint8)
    mx = (S - 1 - xx)[keep]
    mirror = lin[yy[keep], mx].astype(np.int32)
    P = np.stack([(xx[keep] - S / 2 + 0.5) / S * 0.1, -yy[keep] / S * 0.1, np.ones(n) * 0.1], -1).astype(float)
    a = SimpleNamespace(size=S, texel_yx=yx, region=region, mirror=mirror, n=n)
    a.surface = lambda V: P
    a.index_image = lambda: lin.astype(np.int32)
    return a


def test_gradient_smooth_removes_a_seam_but_keeps_detail_and_anchors():
    S = 64
    a = fake_atlas(S)
    rng = np.random.default_rng(0)
    img = np.full((S, S, 3), 150, np.float32) + rng.normal(0, 1.5, (S, S, 1))     # fine detail (~1 %)
    img[:, 32:] *= 1.3                                                           # a hard seam at the centre
    img = np.clip(img, 0, 255).astype(np.uint8)
    region = np.zeros((S, S), bool)
    region[10:54, 20:44] = True
    reg = region[a.texel_yx[:, 0], a.texel_yx[:, 1]]
    out, info = af.gradient_smooth(a, img, reg, ~reg, clip=0.02)
    step = lambda im: abs(float(im[20:44, 33].astype(float).mean() - im[20:44, 30].astype(float).mean()))
    assert step(out) < 0.3 * step(img) and info["texels"] == reg.sum()
    assert np.array_equal(out[~region], img[~region])                            # anchors / outside untouched
    det = lambda im: float(np.std(im[20:44, 22:29, 0].astype(float) - np.median(im[20:44, 22:29, 0])))
    assert det(out) > 0.5 * det(img)                                             # the fine detail survives


def test_seam_metrics_measure_a_vertical_seam():
    S = 64
    a = fake_atlas(S)
    img = np.full((S, S, 3), 120, np.uint8)
    flat = af.seam_metrics(a, img, None, np.ones(a.n, np.float32), np.zeros(a.n, bool), brow_y=-0.02)
    img[:, 40:] = 170
    seam = af.seam_metrics(a, img, None, np.ones(a.n, np.float32), np.zeros(a.n, bool), brow_y=-0.02)
    k = "vertical_seam_face_max_step_pct_median_p90"
    assert flat[k][0] < 1.0 and seam[k][0] > 20.0


def test_set_region_colour_reaches_the_target_contrast():
    S = 64
    a = fake_atlas(S)
    img = np.full((S, S, 3), (150, 170, 220), np.uint8)                      # BGR skin
    img[28:36, 20:44] = (140, 150, 205)                                      # pale lips
    lips = np.zeros((S, S), bool)
    lips[28:36, 20:44] = True
    cheek = np.zeros((S, S), bool)
    cheek[44:50, 10:16] = True
    L, C = lips[a.texel_yx[:, 0], a.texel_yx[:, 1]], cheek[a.texel_yx[:, 0], a.texel_yx[:, 1]]
    out, info = af.set_region_colour(a, img, L, C, {"dL": -15.0, "da": 12.0, "db": 2.0}, ramp=1.0)
    assert abs(info["after"]["da"] - 12.0) < 2.5 and abs(info["after"]["dL"] + 15.0) < 3.0
    assert info["before"]["da"] < 6.0 and np.array_equal(out[5, 5], img[5, 5])


def test_look_targets_are_relative_and_the_chin_blend_hits_its_target(tmp_path):
    lk = {"roles": {"lead": {"portrait_native": {"chin_dL": -53.0}, "portrait_degraded": {"lip_da": 6.69, "lip_dL": -19.14,
                                                                                         "chin_dL": -50.0},
                             "compare": {"lip_da": {"portrait_degraded": 6.69, "video_median": 0.84},
                                         "lip_dL": {"portrait_degraded": -19.14, "video_median": -24.22},
                                         "chin_dL": {"portrait_degraded": -50.0, "video_median": -27.33, "differs": True}}},
                    "follow": {"portrait_native": {"lip_da": 18.73}, "portrait_degraded": {"lip_da": 11.23, "lip_dL": -20.11},
                               "compare": {"lip_da": {"portrait_degraded": 11.23, "video_median": -1.48},
                                           "lip_dL": {"portrait_degraded": -20.11, "video_median": -10.09}}}}}
    (tmp_path / "face").mkdir()
    (tmp_path / "face" / "look.json").write_text(json.dumps(lk))
    cfg = af.FaceTexConfig()
    f = af.look_targets(tmp_path, "follow", cfg)
    assert f["lip_target_videoscale_da"] == pytest.approx(-1.48 + (6.69 - 0.84), abs=0.01)          # 4.37: makeup-free
    assert f["lip_ratio_a"] == pytest.approx(4.37 / 11.23, abs=0.01)
    assert f["lip_target_videoscale_dL"] == pytest.approx(-20.11 + 0.5 * ((-10.09 + 20.11) - (-24.22 + 19.14)), abs=0.01)
    ld = af.look_targets(tmp_path, "lead", cfg)
    assert ld["chin_dL"] == pytest.approx(-27.33 * 53.0 / 50.0, abs=0.01)
    # the chin blend is found on the texture: a dark beard at -40 relative to the cheeks moves to the target
    S = 64
    a = fake_atlas(S)
    img = np.full((S, S, 3), (120, 150, 190), np.uint8)
    img[40:56, 18:46] = (40, 50, 70)
    labels = np.zeros(a.n, np.uint8)
    beard = np.zeros((S, S), bool)
    beard[40:56, 18:46] = True
    labels[beard[a.texel_yx[:, 0], a.texel_yx[:, 1]]] |= af.LAB_BEARD
    ck = np.zeros((S, S), bool)
    ck[20:26, 10:16] = True
    labels[ck[a.texel_yx[:, 0], a.texel_yx[:, 1]]] |= af.LAB_CHEEK
    before = af.region_stats(a, img, labels)["chin_dL"]
    out, info = af.video_look(a, img, labels, {"chin_dL": before + 12.0})
    assert abs(info["chin"]["after_dL"] - (before + 12.0)) < 1.5 and 0 < info["chin"]["blend_toward_skin"] < 1


def test_pad_uncovered_and_remove_streaks():
    S = 64
    a = fake_atlas(S, uncovered=True)
    img = np.full((S, S, 3), 90, np.uint8)
    img[30:36, 30:36] = 250                                                    # background inside a UV hole
    out = af.pad_uncovered(a, img)
    assert np.abs(out[32, 32].astype(int) - 90).max() <= 2                    # the hole takes its island's colour
    img2 = np.full((S, S, 3), 60, np.uint8)
    img2[10:50, 31] = 160                                                      # a thin light line through a beard
    reg = np.ones(a.n, bool)
    out2, info = af.remove_streaks(a, img2, reg, ksize=7, thr=7.0)
    assert info["texels"] > 0 and int(out2[20, 31, 0]) < 90


@needs_model
def test_mouth_cap_closes_smplx_consistently():
    tris, ft = af.mouth_cap()
    m = np.load(fc.smplx_model_npz(), allow_pickle=True)
    f = np.asarray(m["f"], np.int64)
    F = np.concatenate([f, tris])
    und = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1)
    _, c = np.unique(und, axis=0, return_counts=True)
    assert (c == 2).all()                                                      # closed, manifold
    d = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    assert len(np.unique(d, axis=0)) == len(d)                                 # consistently oriented
    v = np.asarray(m["v_template"], float)
    nrm = np.cross(v[tris[:, 1]] - v[tris[:, 0]], v[tris[:, 2]] - v[tris[:, 0]]).sum(0)
    assert nrm[2] > 0                                                          # faces out of the mouth (+z)
    assert ft.shape == tris.shape


# ------------------------------------------------------------------------------------------------ Unity export


def _face_work(tmp_path, T=5, S_expr=3):
    w = tmp_path / "work" / "tk"
    (w / "smplx_face").mkdir(parents=True)
    t = np.arange(T) / 30.0
    jaw = np.zeros((T, 3))
    jaw[:, 0] = np.radians([0, 1, 2, 3, 4])
    np.savez(w / "smplx_face" / "lead.npz", t=t, jaw_pose=jaw, expression=np.tile(np.arange(S_expr, dtype=np.float32), (T, 1)),
             eyelid=np.stack([np.linspace(0, 1, T)] * 2, 1).astype(np.float32),
             eyelid_names=np.array(["eyelid_left", "eyelid_right"]), leye_pose=np.zeros((T, 3)),
             reye_pose=np.zeros((T, 3)), n_expr=np.array(S_expr), shipped=np.array("rest_closed"))
    names = [f"expr{k:02d}" for k in range(S_expr)] + ["eyelid_left", "eyelid_right"]
    d = np.zeros((len(names), 4, 3), np.float32)
    d[:, :, 2] = 0.001 * (1 + np.arange(len(names)))[:, None]                 # +z in the model frame
    np.savez(w / "smplx_face" / "face_blendshapes.npz", vertex_ids=np.array([5, 6, 7, 8], np.int32),
             names=np.array(names), deltas=d)
    return w, t


def test_face_bin_roundtrip_and_unity_axes(tmp_path):
    from dancecap import face_export as fe
    w, t = _face_work(tmp_path)
    fa = fe.face_arrays(w, "lead")
    h = fe.write_face_bin(tmp_path / "f.bin", "lead", fa)
    h2, b = fe.read_face_bin(tmp_path / "f.bin")
    assert h2["names"] == ["expr00", "expr01", "expr02", "eyelid_left", "eyelid_right"] and h["frames"] == 5
    assert b["deltas"].shape == (5, 4, 3) and (b["deltas"][..., 2] < 0).all()  # z mirrored for Unity
    assert np.allclose(b["weights"][:, :3], [0, 1, 2]) and np.allclose(b["weights"][:, 3], np.linspace(0, 1, 5))
    assert h2["stylised"] == ["eyelid_left", "eyelid_right"] and h2["face"]["shipped"] == "rest_closed"


def test_patch_motion_replaces_only_the_jaw_and_eyes(tmp_path):
    from dancecap import export_unity as eu
    from dancecap import face_export as fe
    w, t = _face_work(tmp_path)
    T = len(t)
    rng = np.random.default_rng(1)
    q = rng.normal(size=(T, eu.N_JOINTS, 4))
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    eu.write_motion(tmp_path / "m.bin", "lead", t, rng.normal(size=(T, 3)), q, rng.normal(size=(T, eu.N_JOINTS, 3)))
    _, before = eu.read_motion(tmp_path / "m.bin")
    rep = fe.patch_motion(tmp_path / "m.bin", fe.face_arrays(w, "lead"))
    _, after = eu.read_motion(tmp_path / "m.bin")
    jc = lambda j: slice(3 + 4 * j, 3 + 4 * j + 4)
    changed = np.flatnonzero(np.abs(after - before).max(0) > 1e-7)
    assert set(changed) <= set(range(3 + 4 * 22, 3 + 4 * 25))
    assert np.allclose(after[:, jc(22)], eu.to_unity_quats(eu.rotvec_to_quat(np.stack([np.radians([0, 1, 2, 3, 4]),
                                                                                         np.zeros(T), np.zeros(T)], 1))),
                       atol=1e-6)
    assert rep["jaw_deg_max_after"] == pytest.approx(4.0, abs=0.01)
    with np.load(w / "smplx_face" / "lead.npz") as z:
        dd = {k: z[k] for k in z.files}
    dd["t"] = t + 1.0
    np.savez(w / "smplx_face" / "lead.npz", **dd)
    with pytest.raises(ValueError):
        fe.patch_motion(tmp_path / "m.bin", fe.face_arrays(w, "lead"))       # another time grid is refused


@needs_model
def test_cap_skin_appends_the_mouth_cap_once(tmp_path):
    from dancecap import export_unity as eu
    from dancecap import face_export as fe
    m = np.load(fc.smplx_model_npz(), allow_pickle=True)
    f = np.asarray(m["f"], np.int64)
    V = int(np.asarray(m["v_template"]).shape[0])
    skin = eu.Skin(vertices=np.zeros((V, 3)), faces=f[:, [0, 2, 1]], weights=np.zeros((V, eu.N_JOINTS)),
                   joints=np.zeros((eu.N_JOINTS, 3)), parents=-np.ones(eu.N_JOINTS, np.int64), model="SMPLX_NEUTRAL.npz",
                   betas=np.zeros(1))
    eu.write_skin(tmp_path / "s.bin", "lead", skin)
    r1 = fe.cap_skin(tmp_path / "s.bin")
    r2 = fe.cap_skin(tmp_path / "s.bin")
    h, s2 = eu.read_skin(tmp_path / "s.bin")
    tris, _ = af.mouth_cap()
    assert r1["mouth_cap_triangles"] == len(tris) and r2.get("already") and h["faces"] == len(f) + len(tris)
    assert np.array_equal(s2.faces[:len(f)], f[:, [0, 2, 1]])                 # existing triangle indices unchanged
    assert np.array_equal(s2.faces[len(f):], tris[:, [0, 2, 1]])


def test_albedo_source_refuses_a_stale_face_texture(tmp_path):
    from dancecap import face_export as fe
    av = tmp_path / "avatar"
    av.mkdir()
    for nm in ("lead_albedo.png", "lead_albedo_face.png"):
        (av / nm).write_bytes(b"x")
    old = time.time() - 100
    os.utime(av / "lead_albedo_face.png", (old, old))
    p, warn = fe.albedo_source(tmp_path, "lead", "face")
    assert p.name == "lead_albedo.png" and "older" in warn
    p, warn = fe.albedo_source(tmp_path, "lead", "face", allow_stale=True)
    assert p.name == "lead_albedo_face.png" and warn is None
    p, warn = fe.albedo_source(tmp_path, "lead", "videolook")
    assert p.name == "lead_albedo.png" and "missing" in warn


def test_hands_pod_bundles_the_adopted_face_block(tmp_path, monkeypatch):
    from dancecap import hands_pod
    work = tmp_path / "dancecap" / "work" / "tk"
    for p in ("hands/grid_obs.npz", "hands/jobs.json", "hands/detections.npz", "smplx/lead.npz", "smplx_face/lead.npz",
              "mamma/ma_2d/tk_c8/cam01.npz"):
        (work / p).parent.mkdir(parents=True, exist_ok=True)
        (work / p).write_bytes(b"x")
    (work / "mamma" / "select.json").write_text(json.dumps({"chosen": "c8", "cams": ["01", "02"]}))
    np.savez(work / "smplx" / "lead.npz", t=np.arange(176) / 30.0)
    monkeypatch.setattr(hands_pod, "PACKAGE_ROOT", tmp_path / "dancecap")
    d = hands_pod.build_spec("tk")
    assert "dancecap/work/tk/smplx_face/lead.npz" in d["inputs"]
