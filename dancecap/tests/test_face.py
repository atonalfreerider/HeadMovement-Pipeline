"""Face stages: landmark embedding, portrait fit (synthetic ground truth), compositing helpers, expression-fit
helpers, detector-evaluation helpers, FLAME loading / transfer (skipped when the gated files are absent).

    .venv-nopo4d\\Scripts\\python.exe -m pytest dancecap/tests/test_face.py -q
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dancecap import face_common as fc  # noqa: E402
from dancecap import face_eval as fe  # noqa: E402

needs_model = pytest.mark.skipif(not fc.smplx_model_npz().exists(), reason="SMPL-X model not installed")


def _rot(rv):
    from scipy.spatial.transform import Rotation
    return Rotation.from_rotvec(rv).as_matrix()


def test_mediapipe_map_is_68_unique():
    assert fc.MP_TO_IBUG68.shape == (68,)
    assert len(set(fc.MP_TO_IBUG68.tolist())) == 68
    assert fc.MP_TO_IBUG68.max() < 478
    assert list(fc.IBUG52[:3]) == [8, 17, 18] and fc.IBUG52[-1] == 67


def test_kabsch_and_projection():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(30, 3))
    R = _rot([0.3, -0.2, 0.5])
    t = np.array([0.1, 0.2, 3.0])
    R2, t2 = fc.kabsch(A, A @ R.T + t)
    assert np.allclose(R2, R, atol=1e-9) and np.allclose(t2, t, atol=1e-9)
    K = np.array([[1000.0, 0, 500], [0, 1000, 400], [0, 0, 1]])
    px, z = fc.project(np.array([[0.0, 0, 2]]), K, np.eye(3), np.zeros(3))
    assert np.allclose(px, [[500, 400]]) and np.allclose(z, [2])


def test_crop_pad_outside_image():
    img = np.full((20, 30, 3), 7, np.uint8)
    c = fc.crop_pad(img, (-5, -5, 15, 15))
    assert c.shape == (20, 20, 3) and c[0, 0, 0] == 0 and c[10, 10, 0] == 7


@needs_model
def test_landmark_embedding_frontal_template():
    emb = fc.LandmarkEmbedding()
    m = np.load(fc.smplx_model_npz(), allow_pickle=True)
    L = emb.points(np.asarray(m["v_template"], float))
    i36, i45, i30 = emb.index([36, 45, 30])
    assert 0.07 < np.linalg.norm(L[i36] - L[i45]) < 0.12          # outer eye corners ~9-10 cm apart
    assert L[i30, 2] > L[i36, 2]                                   # nose tip in front of the eyes (+z)
    assert L[0, 1] < L[i30, 1] < L[i36, 1]                         # chin below nose below eyes
    R, _ = fc.head_frame(L, L @ _rot([0, 0.4, 0]).T)
    assert np.allclose(R, _rot([0, 0.4, 0]), atol=1e-6)


@needs_model
def test_portrait_fit_recovers_synthetic_view():
    import torch

    from dancecap import avatar_face as af
    emb = fc.LandmarkEmbedding()
    cfg = af.FaceTexConfig()
    head = af.HeadModel(np.zeros(16, np.float32), cfg.n_expr)
    lfn = head.landmark_fn(emb)
    e_true = torch.zeros(cfg.n_expr)
    e_true[0] = 1.0
    with torch.no_grad():
        P = lfn(e_true, torch.tensor([0.12, 0.0, 0.0])).numpy()
        Pv = head.verts(e_true, torch.tensor([0.12, 0.0, 0.0])).numpy()
    # fast landmark model == full SMPL-X forward on the landmark triangles (pose correctives aside)
    assert np.abs(emb.points(Pv.astype(np.float64)) - P).max() < 1e-3
    R = _rot([0.15, -0.5, 0.1])
    Q = P @ R.T
    lmk2d = np.stack([Q[:, 0], -Q[:, 1]], 1) * 600 + np.array([220, 230])
    fit = af.fit_portrait(head, emb, lmk2d, cfg)
    assert fit["rms_iod"] < 0.02
    assert np.degrees(np.arccos(np.clip((np.trace(fit["R"].T @ R) - 1) / 2, -1, 1))) < 3.0
    assert abs(fit["s"] - 600) / 600 < 0.03


def test_ortho_init_recovers_rotation():
    from dancecap import avatar_face as af

    class E:
        def index(self, ids):
            return np.arange(5, 36)
    rng = np.random.default_rng(1)
    P = rng.normal(scale=0.05, size=(52, 3))
    R = _rot([0.2, 0.6, -0.1])
    Q = P @ R.T
    lmk = np.stack([Q[:, 0], -Q[:, 1]], 1) * 400 + 100
    R0, s, t = af.ortho_init(P, lmk, E())
    assert abs(s - 400) < 1e-6 * 400 + 1e-3
    assert np.allclose(R0, R, atol=1e-6)


def test_bilinear_many_points_and_blend():
    from dancecap import avatar_face as af
    img = np.arange(100, dtype=np.float32).reshape(10, 10)
    xy = np.tile(np.array([[2.5, 3.0]], np.float32), (40000, 1))   # > SHRT_MAX samples
    v = af._bilinear(img, xy)
    assert v.shape == (40000,) and np.allclose(v, 32.5)
    a = np.zeros((64, 64, 3), np.float32)
    b = np.ones((64, 64, 3), np.float32)
    assert np.allclose(af._laplacian_blend(a, b, np.zeros((64, 64), np.float32), 4), 0, atol=1e-5)
    assert np.allclose(af._laplacian_blend(a, b, np.ones((64, 64), np.float32), 4), 1, atol=1e-5)
    m = np.zeros((64, 64), bool)
    m[10:20, 10:20] = True
    f = af._fill_masked(np.where(m[..., None], b, 0), m)
    assert np.allclose(f, 1, atol=1e-5)


def test_srgb_roundtrip():
    from dancecap import avatar_face as af
    x = np.linspace(0, 1, 50)
    assert np.allclose(af._srgb(af._lin(x)), x, atol=1e-6)


def test_rodrigues_batch_matches_scipy():
    import torch

    from dancecap import face_expr as fx
    rv = np.array([[0.0, 0, 0], [0.3, -0.2, 0.1], [1.2, 0.4, -0.7]])
    R = fx.rodrigues(torch, torch.tensor(rv, dtype=torch.float32)).numpy()
    for k in range(3):
        assert np.allclose(R[k], _rot(rv[k]), atol=1e-5)


@needs_model
def test_batched_landmarks_match_single():
    import torch

    from dancecap import avatar_face as af
    from dancecap import face_expr as fx
    emb = fc.LandmarkEmbedding()
    head = af.HeadModel(np.zeros(16, np.float32), 10)
    single, batched = head.landmark_fn(emb), fx.batched_landmarks(head, emb)
    e = torch.randn(3, 10) * 0.5
    j = torch.tensor([[0.1, 0, 0], [0.0, 0, 0], [0.3, 0.05, 0]])
    B = batched(e, j)
    for k in range(3):
        assert torch.allclose(B[k], single(e[k], j[k]), atol=1e-5)


def test_similarity_residual_and_mouth_open():
    rng = np.random.default_rng(2)
    a = rng.normal(size=(52, 2))
    th = 0.4
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    b = 2.5 * a @ R.T + 3
    assert fe._similarity_residual(a, b).max() < 1e-9
    p = np.zeros((52, 2))
    idx = lambda k: 1 + (k - 17)
    p[idx(48)], p[idx(54)] = [0, 0], [10, 0]
    for up, lo in ((61, 67), (62, 66), (63, 65)):
        p[idx(up)] = [5, -1]
        p[idx(lo)] = [5, 1]
    assert np.isclose(fe._mouth_open(p), 0.2)


def test_stub_unpickler_reads_chumpy_like_objects(tmp_path):
    from dancecap import face_smirk as fs
    mod = type(sys)("chumpy")

    class Ch:
        def __init__(self, x):
            self.x = x
    Ch.__module__ = "chumpy"
    Ch.__qualname__ = "Ch"
    mod.Ch = Ch
    sys.modules["chumpy"] = mod
    try:
        p = tmp_path / "m.pkl"
        with open(p, "wb") as f:
            pickle.dump({"shapedirs": Ch(np.ones((2, 3, 4))), "f": np.zeros((1, 3), int)}, f)
    finally:
        del sys.modules["chumpy"]
    d = fs._NoChumpy(open(p, "rb"), encoding="latin1").load()
    assert isinstance(d["shapedirs"], fs._Stub) and d["shapedirs"].x.shape == (2, 3, 4)


needs_flame = pytest.mark.skipif(not (fc.smplx_model_npz().exists() and
                                      Path(__file__).resolve().parents[2].joinpath("models", "flame", "FLAME2020",
                                                                                     "generic_model.pkl").exists()
                                      and Path(__file__).resolve().parents[2].joinpath("external", "smirk", "assets").exists()),
                                 reason="FLAME 2020 / SMIRK assets not installed (gated, local only)")


@needs_flame
def test_flame_geometry_and_transfer():
    from dancecap import face_smirk as fs
    geo = fs.FlameGeo(n_sx=20)
    z = np.zeros((1, 50))
    m0 = geo.metrics(z, np.zeros((1, 3)), np.zeros((1, 2)))
    m1 = geo.metrics(z, np.array([[0.25, 0, 0]]), np.zeros((1, 2)))
    assert m1["mouth_open_mm"][0] > m0["mouth_open_mm"][0] + 5      # jaw opening opens the mouth
    assert abs(m0["smile_mm"][0]) < 1e-6
    e = np.zeros((1, 50))
    e[0, 0] = 2.0
    sx, _ = geo.to_smplx(e, np.zeros((1, 2)))
    assert abs(sx[0, 0]) > 0.5 * np.abs(sx[0, 1:]).max()            # FLAME comp 0 ~ SMPL-X comp 0 (cos 0.93)
