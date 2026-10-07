"""Face build stage (2026-10-06): final per-frame face block (face_expr.build helpers), avatar_face post-processing
(skin repair, mouth interior, video look, head symmetrisation, output naming), face_worker shards, face_pod spec.
Tiny synthetic inputs only (a fake 64 x 64 UV atlas); the SMPL-X fit test is skipped without the model.

    .venv-nopo4d\\Scripts\\python.exe -m pytest dancecap/tests/test_face_build.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dancecap import avatar_face as af  # noqa: E402
from dancecap import face_common as fc  # noqa: E402
from dancecap import face_expr as fx  # noqa: E402

needs_model = pytest.mark.skipif(not fc.smplx_model_npz().exists(), reason="SMPL-X model not installed")


# ------------------------------------------------------------------------------------------------ face_expr helpers


def test_sim_align_recovers_similarity():
    import torch
    rng = np.random.default_rng(0)
    P = rng.normal(size=(3, 52, 2))
    th, s, t = 0.3, 1.7, np.array([4.0, -2.0])
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    U = s * P @ R.T + t
    w = torch.zeros(52)
    w[torch.as_tensor(fx._ibug_pos(fx.RIGID_ANCHORS))] = 1
    Pa = fx.sim_align(torch, torch.tensor(P), torch.tensor(U), w.double()).numpy()
    assert np.abs(Pa - U).max() < 1e-9          # exact on every point, anchors or not


def test_ibug_positions_and_beard_weights():
    assert fx._ibug_pos((8, 17, 67)).tolist() == [0, 1, 51]
    cfg = fx.FaceConfig()
    w_lead, w_follow = fx.point_weights(cfg, "lead"), fx.point_weights(cfg, "follow")
    assert w_lead[0] == 0 and w_follow[0] == cfg.w_mouth            # chin dropped for the bearded dancer
    assert w_lead[fx._ibug_pos((57,))[0]] == 0.5 * cfg.w_mouth
    assert (w_follow[fx._ibug_pos(fx.EYELIDS)] == 0).all()          # blinks are not fitted


def test_view_quality_and_confidence():
    cfg = fx.FaceConfig()
    q = fx.view_quality([10, 16, 25, 25, 25], [10, 10, 10, 50, 10], [0.1, 0.1, 0.1, 0.1, 0.4], cfg)
    assert q[0] == 0 and 0.4 < q[1] < 0.6 and q[2] == 1 and 0 < q[3] < 1 and q[4] == 0
    c = fx.confidence_curve(np.array([50, 51]), np.array([1.0, 0.5]), 120, 30.0, cfg)
    assert c.max() <= 1 and c[50] > 0.7 and c[0] < 1e-6 and c[119] < 1e-6      # an isolated view: ~0.8
    assert c[47] > 0.3                                              # held +-0.1 s, then smoothed


def test_blend_final_gating_and_clamps():
    cfg = fx.FaceConfig()
    T, E = 60, 10
    e0 = np.full(E, 0.1)
    e_cap = np.tile(e0, (T, 1))
    e_cap[20:40, 0] += 5.0                                          # a large captured excursion
    jaw_cap = np.full(T, np.radians(20.0))                          # far beyond the clamp
    conf = np.zeros(T)
    conf[25:35] = 1.0
    expr, jaw, e_sm, _ = fx.blend_final(e_cap, jaw_cap, e0, 0.0, conf, 30.0, cfg)
    assert np.allclose(expr[conf == 0], e0)                         # no confidence -> exactly the rest face
    assert np.linalg.norm(expr[30] - e0) <= cfg.max_dev + 1e-9      # deviation clamped
    assert jaw.max() <= np.radians(cfg.jaw_max_deg) + 1e-12 and jaw[0] == 0
    assert e_sm[20, 0] < e_cap[20, 0]                               # temporally smoothed


def test_blink_schedule_deterministic_and_spaced():
    cfg = fx.FaceConfig()
    t = np.arange(0, 40, 1 / 30)
    w1, s1 = fx.blink_schedule(t, cfg, "take/follow")
    w2, s2 = fx.blink_schedule(t, cfg, "take/follow")
    _, s3 = fx.blink_schedule(t, cfg, "take/lead")
    assert np.array_equal(w1, w2) and np.array_equal(s1, s2) and not np.array_equal(s1, s3)
    gaps = np.diff(s1)
    assert len(s1) >= 8 and gaps.min() >= cfg.blink_every_s[0] - 1e-9 and gaps.max() <= cfg.blink_every_s[1] + 1e-9
    assert np.isclose(w1.max(), 1.0) and w1.min() == 0.0
    closed = (w1 > 0.5).sum() / 30 / len(s1)                        # time with the lids more than half closed
    assert 0.05 < closed < 0.25


def _pooled(neutral, rest, final_expr, final, on, oe):
    """pooled hold-out dict: mouth NME (% IOD) and opening error per mode."""
    mk = lambda n, o: {"mouth_nme_pct_iod_median": n, "mouth_open_abs_err_median": o}
    return {"neutral": mk(neutral, on), "rest": mk(rest, oe), "final_expr": mk(final_expr, oe), "final": mk(final, oe)}


def test_adoption_rule():
    cfg = fx.FaceConfig()
    # rest clearly better on held-out cameras, per-frame layer worse on held-out time blocks -> rest only
    ch = {"cross_camera": {"pooled": _pooled(9.6, 6.7, 7.9, 7.9, 0.02, 0.025)},
          "time_blocks": {"pooled": _pooled(9.6, 6.7, 7.9, 7.9, 0.02, 0.025)}}
    a = fx.adoption(ch, cfg)
    assert a["rest"] and not a["expression"] and not a["jaw"] and a["shipped"] == "rest"
    # 2026-10-07: a rest that raises the opening error 2.5x (0.02 -> 0.05) is no longer adopted (relative tolerance)
    ch_open = {"cross_camera": {"pooled": _pooled(9.6, 6.7, 7.9, 7.9, 0.02, 0.05)},
               "time_blocks": {"pooled": _pooled(9.6, 6.7, 7.9, 7.9, 0.02, 0.05)}}
    assert not fx.adoption(ch_open, cfg)["rest"]
    # rest no better than neutral -> neutral face, nothing else
    ch2 = {"cross_camera": {"pooled": _pooled(4.4, 4.6, 4.3, 4.3, 0.013, 0.027)},
           "time_blocks": {"pooled": _pooled(4.5, 4.2, 4.1, 4.2, 0.013, 0.03)}}
    assert not any(fx.adoption(ch2, cfg)[k] for k in ("rest", "expression", "jaw"))
    # everything supported
    ch3 = {"cross_camera": {"pooled": _pooled(8.0, 6.0, 5.8, 5.7, 0.03, 0.03)},
           "time_blocks": {"pooled": _pooled(8.0, 6.0, 5.0, 4.9, 0.03, 0.03)}}
    a3 = fx.adoption(ch3, cfg)
    assert a3["rest"] and a3["expression"] and a3["jaw"]
    assert not fx.adoption(None, cfg)["rest"]


@needs_model
def test_aligned_fit_recovers_rest_expression():
    """Synthetic views of a head with a known resting expression: the rest fit beats the neutral face."""
    import torch

    from dancecap.avatar_face import HeadModel
    cfg = fx.FaceConfig(iters=150)
    emb = fc.LandmarkEmbedding()
    head = HeadModel(np.zeros(16, np.float32), cfg.n_expr)
    e_true = np.zeros(cfg.n_expr)
    e_true[0], e_true[3] = 1.2, -0.8
    T = 8
    blfn = fx.batched_landmarks(head, emb)
    with torch.no_grad():
        L = blfn(torch.tensor(np.tile(e_true, (T, 1)), dtype=torch.float32), torch.zeros((T, 3))).numpy().astype(float)
    Rk = np.tile(np.eye(3), (T, 1, 1))
    tk = np.zeros((T, 3))
    K = np.array([[2000.0, 0, 500], [0, 2000, 500], [0, 0, 1]])
    obs = {k: [] for k in ("k", "cam", "i", "uv", "iod", "angle", "raw_nme", "K", "R", "tc", "k1")}
    for k in range(T):
        for yaw in (-0.3, 0.2):
            from scipy.spatial.transform import Rotation
            R = Rotation.from_rotvec([0, np.pi - yaw, 0]).as_matrix()          # camera looks at the face (+z)
            C = np.array([np.sin(yaw), 1.6, np.cos(yaw)]) * np.array([2.0, 1.0, 2.0])
            C[1] = L[k, :, 1].mean()
            t = -R @ C
            uv, _ = fc.project(L[k], K, R, t)
            obs["k"].append(k), obs["cam"].append("01" if yaw < 0 else "02"), obs["i"].append(k)
            obs["uv"].append(uv), obs["iod"].append(float(np.linalg.norm(uv[emb.index([36])[0]] - uv[emb.index([45])[0]])))
            obs["angle"].append(10.0), obs["raw_nme"].append(0.05), obs["K"].append(K), obs["R"].append(R)
            obs["tc"].append(t), obs["k1"].append(0.0)
    obs = {k: np.array(v) for k, v in obs.items()}
    obs["q"] = np.ones(len(obs["k"]))
    prep = {"role": "follow", "emb": emb, "head": head, "grid_t": np.arange(T) / 30, "fps": 30.0, "Rk": Rk, "tk": tk,
            "D": np.zeros((52, 3)), "has_D": False, "obs": obs}
    rs = fx.fit_aligned(prep, cfg, mode="rest")
    idx = np.arange(len(obs["k"]))
    e_neu = fx.aligned_errors(prep, cfg, np.zeros((T, cfg.n_expr)), np.zeros(T), idx, rs["_world"], rs["_proj"])
    e_rst = fx.aligned_errors(prep, cfg, rs["expr"], rs["jaw"], idx, rs["_world"], rs["_proj"])
    assert e_rst["nme_mouth_med"] < 0.5 * e_neu["nme_mouth_med"]
    assert rs["e0"][0] > 0.5


# ------------------------------------------------------------------------------------------------ avatar_face post


def fake_atlas(S: int = 64):
    """Every texel of an S x S image is a texel; head everywhere, face (bit 2) in the centre, mirror = left/right
    flip; surface(V) = (x, y, z) from the texel position (x = column - centre)."""
    yy, xx = np.mgrid[0:S, 0:S]
    yx = np.stack([yy.ravel(), xx.ravel()], 1).astype(np.int32)
    n = len(yx)
    face = (np.abs(xx - S / 2 + 0.5) < S * 0.3) & (yy > S * 0.2) & (yy < S * 0.85)
    region = (1 + 2 * face).ravel().astype(np.uint8)
    mirror = (yy * S + (S - 1 - xx)).ravel().astype(np.int32)
    P = np.stack([(xx - S / 2 + 0.5) / S, -yy / S, np.ones_like(xx) * 0.1], -1).reshape(-1, 3).astype(float)
    a = SimpleNamespace(size=S, texel_yx=yx, region=region, mirror=mirror, n=n)
    a.surface = lambda V: P
    a.index_image = lambda: np.arange(n, dtype=np.int32).reshape(S, S)
    return a


def test_face_outputs_next_to_bake(tmp_path):
    b = tmp_path / "follow_albedo.png"
    o = af.face_outputs(b)
    assert o["portrait"] == tmp_path / "follow_albedo_face.png" and o["video"] == tmp_path / "follow_albedo_face_videolook.png"
    with pytest.raises(ValueError):
        af.face_outputs(tmp_path / "follow_albedo_face.png")


def test_repair_face_skin_replaces_hair_patch_by_mirror():
    S = 64
    a = fake_atlas(S)
    skin, hair = np.array([150, 170, 220], np.uint8), np.array([40, 40, 120], np.uint8)   # BGR
    img = np.empty((S, S, 3), np.uint8)
    img[:] = hair                                                  # scalp
    face = ((a.region & 2) > 0).reshape(S, S)
    img[face] = skin
    img[30:45, 14:24] = hair                                       # hair baked onto the left cheek
    alpha = np.zeros(a.n, np.float32)
    alpha[(face & (np.arange(S)[None, :] >= S // 2)).ravel()] = 1.0  # the portrait covers the right half only
    cfg = af.FaceTexConfig(repair_feather=2)
    out, rep, info = af.repair_face_skin(a, img, alpha, cfg)
    assert info["repaired_texels"] > 0.6 * 15 * 10 and info["mirrored_texels"] > 0
    assert np.abs(out[37, 19].astype(int) - skin).max() < 25       # patch centre is skin again
    assert np.array_equal(out[5, 5], img[5, 5])                    # scalp untouched


def test_mouth_interior_paint_is_darker_lip():
    S = 64
    a = fake_atlas(S)
    img = np.full((S, S, 3), 200, np.uint8)
    img[28:36, 20:44] = (60, 60, 180)                             # lips
    interior = np.zeros(a.n, bool)
    m = np.zeros((S, S), bool)
    m[31:33, 24:40] = True
    interior[m.ravel()] = True
    out, info = af.paint_mouth_interior(a, img, interior)
    assert info["mouth_interior_texels"] == m.sum()
    assert out[32, 30].astype(int).sum() < img[30, 30].astype(int).sum()     # darker than the lips
    assert out[32, 30, 2] > out[32, 30, 0]                         # still reddish (BGR)
    assert np.array_equal(out[5, 5], img[5, 5])


def test_video_look_shifts_lip_redness():
    S = 64
    a = fake_atlas(S)
    img = np.full((S, S, 3), (150, 170, 220), np.uint8)
    img[28:36, 20:44] = (60, 60, 200)
    labels = np.zeros(a.n, np.uint8)
    lip = np.zeros((S, S), bool)
    lip[28:36, 20:44] = True
    labels[lip.ravel()] |= af.LAB_LIPS | af.LAB_MESH_LIPS
    ck = np.zeros((S, S), bool)
    ck[40:46, 10:16] = True
    labels[ck.ravel()] |= af.LAB_CHEEK
    # 2026-10-07: the look is a target relative to the cheeks (the portrait look's lip contrast x ratios), measured
    # on the texture - no fixed shift
    d0 = af.region_stats(a, img, labels)["mesh_lips_da"]
    out, info = af.video_look(a, img, labels, {"lip_ratio_a": 0.5, "lip_ratio_L": 1.0},
                              lip_target={"dL": -20.0, "da": d0, "db": 0.0})
    d1 = info["after"]["mesh_lips_da"]
    assert abs(d1 - 0.5 * d0) < 3.0 and info["lips"]["target"]["da"] == round(0.5 * d0, 2)
    assert np.array_equal(out[5, 5], img[5, 5])


def test_symmetrize_head_evens_left_right():
    S = 64
    a = fake_atlas(S)
    img = np.full((S, S, 3), 120, np.uint8)
    img[:, : S // 2] = 170                                        # one side lit brighter (a bake seam at the centre)
    out, info = af.symmetrize_head(a, img, np.zeros((10, 3)), sigma=6.0, ramp=1)
    b, f = info["luma_left_right_before"], info["luma_left_right_after"]
    assert abs(f[0] - f[1]) < 0.3 * abs(b[0] - b[1])
    assert abs(int(out[30, 30, 0]) - int(out[30, 33, 0])) < 12    # no step across the centre line


def test_extend_forehead_moves_only_upper_points():
    mp = np.zeros((478, 2))
    ang = np.linspace(0, 2 * np.pi, len(af.MP_OVAL), endpoint=False)
    mp[af.MP_OVAL] = np.stack([100 + 40 * np.sin(ang), 100 - 50 * np.cos(ang)], 1)    # index 0 (= 10) at the top
    mp[10], mp[152] = [100, 50], [100, 150]
    mp[33], mp[263] = [80, 90], [120, 90]
    ext = af.extend_forehead(mp, 0.3)
    d = ext - mp[af.MP_OVAL]
    assert np.isclose(d[0, 1], -0.3 * 40)                          # oval top moved up by 0.3 IOD
    low = mp[af.MP_OVAL][:, 1] > 90
    assert np.allclose(d[low], 0)                                  # nothing below the eye line moves


# ------------------------------------------------------------------------------------------------ worker / pod


def test_face_worker_merge_shards(tmp_path, monkeypatch):
    from dancecap import face_worker as fw
    monkeypatch.setattr(fw, "PKG", tmp_path)
    fd = tmp_path / "work" / "tk" / "face"
    fd.mkdir(parents=True)
    (fd / "jobs.json").write_text(json.dumps({"jobs": [{"k": i} for i in range(5)]}))
    for i in range(2):
        idx = np.arange(i, 5, 2)
        np.savez(fd / f"detections_shard{i}of2.npz", mp_ok=np.ones(len(idx), bool), mp_dist_px=idx.astype(np.float32),
                 mp_blend_names=np.array(["a", "b"]), job_index=idx)
    fw.merge_shards("tk", 2)
    z = np.load(fd / "detections.npz")
    assert z["mp_dist_px"].tolist() == [0, 1, 2, 3, 4] and z["mp_ok"].all() and list(z["mp_blend_names"]) == ["a", "b"]
    (fd / "detections_shard1of2.npz").unlink()
    np.savez(fd / "detections_shard1of2.npz", mp_ok=np.ones(1, bool), mp_dist_px=np.ones(1, np.float32),
             mp_blend_names=np.array([]), job_index=np.array([1]))
    with pytest.raises(ValueError):
        fw.merge_shards("tk", 2)


def test_face_pod_spec_parses(tmp_path, monkeypatch):
    from dancecap import face_pod
    from dancecap import remote_jobs as rj
    monkeypatch.setattr(face_pod, "PACKAGE_ROOT", tmp_path)
    w = tmp_path / "work" / "tk"
    for p in ("cameras/01.npz", "cameras/02.npz", "smplx/lead.npz", "smplx/follow.npz"):
        (w / p).parent.mkdir(parents=True, exist_ok=True)
        (w / p).write_bytes(b"x")
    (w / "dancers" / "01" / "hair").mkdir(parents=True)
    d = face_pod.spec("tk")
    job = rj.parse_job(d, "test")
    names = [c.name for c in job.commands]
    assert names[:3] == ["setup_mediapipe", "survey", "mediapipe"] and "--mp-only --shard" in job.commands[2].cmd
    assert any(o.endswith("smplx_face/") for o in job.outputs)
    gated = lambda x: x.replace("\\", "/").lower()
    assert not any(("mano" in gated(loc) or "flame" in gated(loc) or gated(pod).startswith("models/"))
                   for loc, pod in job.inputs)                                     # no gated files in the bundle
    assert any(p.endswith("dancers/01/hair/") for _, p in job.inputs)
    (w / "smplx" / "lead.npz").unlink()
    with pytest.raises(SystemExit):
        face_pod.spec("tk")
