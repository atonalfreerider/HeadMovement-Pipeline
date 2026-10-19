r"""Unit tests for dancecap.mono4d (synthetic numbers and images; no footage, no network, no TypeSafe or vendor calls).

    .venv-nopo4d\Scripts\python.exe -m pytest dancecap/tests/test_mono4d.py -q      (from the workspace root)
"""
from __future__ import annotations

import base64
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dancecap import mamma_dyn as md  # noqa: E402
from dancecap.mono4d import gate, holdout, judge, providers, rig, vendor_watch, virtual_take  # noqa: E402


def rot(axis, deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    x, y, z = np.asarray(axis, float) / np.linalg.norm(axis)
    return np.array([[c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
                     [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
                     [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)]])


# ------------------------------------------------------------------------------------------------- rig


def test_ring_looks_at_target_and_conventions():
    specs = rig.ring(3, n=6, radius=4.0, heights=(1.2, 2.0), centre=(0.5, -0.3), target_height=0.9)
    assert [s.name for s in specs] == [f"v{i:02d}" for i in range(12)]
    target = np.array([[0.5, 0.9, -0.3]])
    for s in specs:
        px, z = md.project(target, s.K[0], s.R[0], s.t[0])
        assert z[0] > 0 and np.allclose(px[0], [s.width / 2, s.height / 2], atol=1e-6)
        assert abs(np.linalg.norm(s.centres[0][[0, 2]] - [0.5, -0.3]) - 4.0) < 1e-9
        cv, gl = s.c2w_opencv()[0], s.c2w_opengl()[0]
        assert np.allclose(gl[:3, 1], -cv[:3, 1]) and np.allclose(gl[:3, 2], -cv[:3, 2])
        assert np.allclose(cv[:3, 3], s.centres[0])
        assert s.c2w_opencv()[0][1, 1] < 0  # image +Y (down) points down in the Y-up world
        assert s.is_static()


def test_trajectory_round_trip_static_and_moving():
    T = 5
    grid = np.arange(T) / 30.0
    ring = rig.ring(T, n=2)
    C = np.stack([np.array([3.0, 1.5, 0.0]) + [0, 0, 0.1 * k] for k in range(T)])
    R = md._look_at(C, np.repeat([[0.0, 0.9, 0.0]], T, 0), np.zeros(T))
    moving = rig.ViewSpec("h01", "holdout", np.repeat(ring[0].K[:1], T, 0), R, -np.einsum("fij,fj->fi", R, C),
                          1080, 1920)
    doc = rig.trajectory(ring + [moving], grid)
    assert doc["cameras"]["v00"]["static"] and len(doc["cameras"]["v00"]["frames"]) == 1
    assert not doc["cameras"]["h01"]["static"] and len(doc["cameras"]["h01"]["frames"]) == T
    back = {s.name: s for s in rig.specs_from_trajectory(json.loads(json.dumps(doc)))}
    for s in ring + [moving]:
        b = back[s.name]
        assert b.T == T and b.kind == s.kind
        assert np.allclose(b.R, s.R, atol=1e-5) and np.allclose(b.t, s.t, atol=1e-5) and np.allclose(b.K, s.K)


def test_auto_radius_keeps_the_couple_in_frame():
    travel = np.array([[0.0, 0.0], [1.2, 0.0], [0.6, 0.8]])
    r0 = rig.auto_radius(travel[:1], 55.0, (1080, 1920))
    r1 = rig.auto_radius(travel, 55.0, (1080, 1920))
    assert r1 > r0 >= 2.5
    assert rig.auto_radius(travel, 55.0, (1080, 1920), min_radius=0.5) < r1 + 1e-6
    centre = travel.mean(0)
    for s in rig.ring(1, n=12, radius=r1, centre=tuple(centre), az0_deg=7.0):  # other azimuths than the search's
        for c in travel:  # the couple's cylinder at each floor position
            px, z = md.project(rig.couple_points(c, 24), s.K[0], s.R[0], s.t[0])
            assert (z > 0).all()
            assert (px[:, 0] >= 0).all() and (px[:, 0] <= s.width).all()
            assert (px[:, 1] >= 0).all() and (px[:, 1] <= s.height).all()


def test_umeyama_and_world_from_scene():
    rng = np.random.default_rng(1)
    Rws, s, t = rot([0.3, 1.0, -0.2], 37.0), 2.5, np.array([0.4, -1.0, 2.0])
    sim3 = {"s": s, "R": Rws.tolist(), "t": t.tolist()}
    X = rng.normal(size=(20, 3))
    est = rig.umeyama(X, rig.apply_sim3(sim3, X))
    assert abs(est["s"] - s) < 1e-9 and np.allclose(est["R"], Rws) and np.allclose(est["t"], t) and est["rms"] < 1e-9
    # a handheld phone walking 1 m; the provider reports its poses in a scene frame related by sim3
    T = 30
    C = np.stack([np.linspace(-0.5, 0.5, T), 1.5 + 0.05 * np.sin(np.arange(T)), np.full(T, 3.5)], 1)
    R = md._look_at(C, np.repeat([[0.0, 0.9, 0.0]], T, 0), np.zeros(T))
    src = rig.ViewSpec("06", "source", np.repeat(np.eye(3)[None], T, 0), R, -np.einsum("fij,fj->fi", R, C), 1080, 1920)
    scene = rig.to_scene(src, sim3)  # the same cameras expressed in the scene frame
    est = rig.world_from_scene(scene.c2w_opencv(), src)
    assert abs(est["s"] - s) < 1e-6 and np.allclose(est["R"], Rws, atol=1e-6)
    assert est["rot_err_deg"]["p95"] < 1e-4 and est["rms"] < 1e-6
    back_R, back_t = md.world_cameras(scene.R, scene.t, est)
    assert np.allclose(back_R, src.R, atol=1e-6) and np.allclose(back_t, src.t, atol=1e-6)
    tripod = rig.ViewSpec("06", "source", src.K, np.repeat(R[:1], T, 0), np.repeat(src.t[:1], T, 0), 1080, 1920)
    with pytest.raises(ValueError, match="moved only"):
        rig.world_from_scene(rig.to_scene(tripod, sim3).c2w_opencv(), tripod)


def test_qvec_round_trip():
    for axis, deg in (([1, 0, 0], 10), ([0, 1, 0], 179), ([0.2, -1, 0.5], 250), ([0, 0, 1], 90)):
        R = rot(axis, deg)
        w, x, y, z = rig.qvec_wxyz(R)
        assert w >= 0
        Rq = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                       [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                       [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
        assert np.allclose(Rq, R, atol=1e-9)


# ------------------------------------------------------------------------------------------------- holdout


def test_mpjpe_pa_mpjpe_and_verdict():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(10, 55, 3))
    moved = 1.1 * ref @ rot([0, 1, 0], 25).T + [0.3, 0.0, -0.2]
    assert holdout.mpjpe(moved, ref).min() > 0.1
    assert holdout.pa_mpjpe(moved, ref).max() < 1e-9
    noisy = ref + rng.normal(scale=0.01, size=ref.shape)
    assert 0.005 < np.median(holdout.pa_mpjpe(noisy, ref)) < 0.02
    good = {r: {"frames": 10, "pa_mpjpe_mm": {"median": 40.0}, "pelvis_mm": {"median": 50.0}} for r in holdout.ROLES}
    base = {r: {"frames": 10, "pa_mpjpe_mm": {"median": 60.0}, "pelvis_mm": {"median": 120.0}} for r in holdout.ROLES}
    assert holdout.verdict(good, base)["route_beats_baseline"]
    v = holdout.verdict(base, good)
    assert not v["route_beats_baseline"] and len(v["why"]) == 4


def test_match_times_and_compare_fits(tmp_path):
    t_ref = 10 + np.arange(60) / 30.0
    t_test = t_ref[10:50] + 0.004
    ir, it = holdout.match_times(t_ref, t_test)
    assert np.array_equal(ir, np.arange(10, 50)) and np.array_equal(it, np.arange(40))
    rng = np.random.default_rng(2)
    for d, shift in (("ref", 0.0), ("test", 0.05)):
        (tmp_path / d).mkdir()
        for role in holdout.ROLES:
            J = np.random.default_rng(3).normal(size=(60, 55, 3)) + [0, 1, 0]
            J[:, :, 0] += shift  # the test fit sits 5 cm off in x, same pose
            np.savez(tmp_path / d / f"{role}.npz", t=t_ref, joints=J, quality=np.ones(60))
    out = holdout.compare_fits(tmp_path / "ref", tmp_path / "test", window=(10.5, 11.0))
    for role in holdout.ROLES:
        assert out[role]["frames"] == 16
        assert abs(out[role]["mpjpe_mm"]["median"] - 50.0) < 1e-6 and out[role]["pa_mpjpe_mm"]["median"] < 1e-6
        assert abs(out[role]["pelvis_horizontal_mm"]["median"] - 50.0) < 1e-6
        assert out[role]["pelvis_vertical_mm"]["median"] < 1e-6
    del rng


def _people(path: Path, boxes: list[list], kpts=None):
    F = len(boxes)
    N = max(len(b) for b in boxes)
    B = np.zeros((F, N, 4))
    valid = np.zeros((F, N), bool)
    for f, bs in enumerate(boxes):
        for n, b in enumerate(bs):
            B[f, n], valid[f, n] = b, True
    K = np.zeros((F, N, 17, 3))
    for f in range(F):
        for n in range(N):
            x0, y0, x1, y1 = B[f, n]
            K[f, n, :, 0] = np.linspace(x0, x1, 17)
            K[f, n, :, 1] = np.linspace(y0, y1, 17)
            K[f, n, :, 2] = 0.9
    if kpts is not None:
        K = kpts(K)
    np.savez(path, boxes=B, scores=np.where(valid, 0.9, 0.0), valid=valid, kpts=K)


def _frames(d: Path, cam: str, t: np.ndarray, size=(320, 480)):
    (d / "frames" / cam).mkdir(parents=True, exist_ok=True)
    from PIL import Image
    files = []
    for i, ti in enumerate(t):
        name = f"{i:06d}.jpg"
        Image.new("RGB", size, (90 + i % 50, 80, 70)).save(d / "frames" / cam / name)
        files.append({"file": name, "pts": float(ti), "t": float(ti)})
    (d / "frames" / cam / "frames.json").write_text(json.dumps(
        {"camera": cam, "source": "test", "width": size[0], "height": size[1], "rotation": 0, "frames": files}))


def test_compare_views_counts_duplicates_and_pck(tmp_path):
    t = 5 + np.arange(4) / 30.0
    real, virt = tmp_path / "real", tmp_path / "virt"
    for w, cam in ((real, "03"), (virt, "h03")):
        (w / "people").mkdir(parents=True)
        _frames(w, cam, t)
    two = [[10, 10, 110, 300], [150, 20, 250, 310]]
    _people(real / "people" / "03.npz", [two] * 4)
    # generated: frames 0-1 exact, frame 2 has a ghost third person, frame 3 shifts the keypoints by 40 px
    shift = lambda K: np.concatenate([K[:3], K[3:4] + [40.0, 0.0, 0.0]])  # noqa: E731
    _people(virt / "people" / "h03.npz", [two, two, two + [[60, 15, 160, 305]], two], kpts=shift)
    r = holdout.compare_views(real, "03", virt, "h03")
    assert r["frames"] == 4 and r["extra_people_frac"] == 0.25 and r["missing_people_frac"] == 0.0
    assert r["pck_0.1"]["n"] == 8 and 0.7 < r["pck_0.1"]["mean"] < 0.8  # 6 exact people + 2 shifted by 40 px
    assert r["mask_iou"]["n"] == 0


# ------------------------------------------------------------------------------------------------- providers + package


def _fake_take(root: Path, take: str = "tk", cam: str = "06", n: int = 7, t0: float = 10.0) -> Path:
    work = root / "work" / take
    t = t0 + np.arange(n) / 30.0
    _frames(work, cam, t, size=(320, 480))
    C = np.stack([np.linspace(-0.4, 0.4, n), np.full(n, 1.5), np.full(n, 3.5)], 1)
    R = md._look_at(C, np.repeat([[0.0, 0.9, 0.0]], n, 0), np.zeros(n))
    K = np.repeat(np.array([[[400.0, 0, 160], [0, 400.0, 240], [0, 0, 1]]]), n, 0)
    (work / "cameras").mkdir(parents=True)
    np.savez(work / "cameras" / f"{cam}.npz", t=t, K=K, R=R, tvec=-np.einsum("fij,fj->fi", R, C), ok=np.ones(n, bool))
    (work / "cameras" / "world_from_sfm.json").write_text(json.dumps(virtual_take.IDENTITY_SIM3))
    return work


def _request(work: Path, run: str, targets: list[rig.ViewSpec], grid: np.ndarray, provider="manual") -> Path:
    rd = providers.run_dir(work, run)
    (rd / "request").mkdir(parents=True)
    tr = md.load_tracks(work)["06"]
    src = rig.ViewSpec("06", "source", tr.K, tr.R, tr.tvec, tr.width, tr.height)
    (rd / "request" / "source_trajectory.json").write_text(json.dumps(rig.trajectory([src], tr.t)))
    (rd / "request" / "target_trajectory.json").write_text(json.dumps(rig.trajectory(targets, grid)))
    req = providers.Mono4DRequest(take=work.name, input_cam="06", provider=provider, t0=float(tr.t[0]),
                                  t1=float(tr.t[-1]), grid=grid.tolist(), video=str(rd / "request" / "in.mp4"),
                                  video_fps=30.0, video_frames=len(tr.t), targets=[s.name for s in targets])
    providers.save_request(rd, req)
    return rd


def test_manual_provider_fetch_validates_and_package_loads(tmp_path, monkeypatch):
    work = _fake_take(tmp_path)
    grid = md.master_grid(10.0, 10.2)
    targets = rig.ring(len(grid), n=4, radius=3.0, size=(320, 480))
    rd = _request(work, "r1", targets, grid)
    prov = providers.ManualProvider(rd)
    req = providers.load_request(rd)
    job = prov.submit(req)
    assert (rd / "request" / "REQUEST.md").exists() and prov.fetch(req, job) is None
    for s in targets:  # frame-folder deliveries; v03 at the wrong size
        size = (320, 480) if s.name != "v03" else (300, 480)
        _frames(rd / "drop" / "views" / s.name, "x", grid, size=size)
        for f in (rd / "drop" / "views" / s.name / "frames" / "x").glob("*.jpg"):
            shutil.move(f, rd / "drop" / "views" / s.name / f.name)
        shutil.rmtree(rd / "drop" / "views" / s.name / "frames")
    with pytest.raises(ValueError, match="v03: 300x480"):
        prov.fetch(req, job)
    shutil.rmtree(rd / "drop" / "views" / "v03")
    res = prov.fetch(req, job)
    assert res.kind == "views" and sorted(res.views) == ["v00", "v01", "v02"] and "missing views" in res.notes[0]
    providers.save_result(rd, res)
    monkeypatch.setattr(virtual_take, "PACKAGE_ROOT", tmp_path)
    out = virtual_take.package("tk", "r1", req, providers.load_result(rd))
    tracks = md.load_tracks(out)  # MAMMA's own loader accepts the virtual take
    assert sorted(tracks) == ["06", "v00", "v01", "v02"]
    assert len(tracks["v00"].t) == len(grid) and len(tracks["06"].t) == 7
    assert np.allclose(tracks["v01"].R[0], targets[1].R[0]) and np.allclose(tracks["06"].t, 10 + np.arange(7) / 30)
    man = virtual_take.is_virtual(out)
    assert man["synthetic_cams"] == ["v00", "v01", "v02"] and man["real_cams"] == ["06"] and man["independent_views"] == 1
    with pytest.raises(FileExistsError):
        virtual_take.package("tk", "r1", req, res)


def test_scene_alignment_from_provider_poses(tmp_path):
    work = _fake_take(tmp_path)
    grid = md.master_grid(10.0, 10.2)
    rd = _request(work, "r2", rig.ring(len(grid), n=2), grid)
    src = rig.specs_from_trajectory(json.loads((rd / "request" / "source_trajectory.json").read_text()))[0]
    sim3 = {"s": 0.5, "R": rot([0, 1, 0], 30).tolist(), "t": [1.0, 0.0, 0.0]}
    (rd / "drop").mkdir()
    (rd / "drop" / "scene_cameras.json").write_text(json.dumps(
        {"input": rig.to_scene(src, sim3).c2w_opencv().tolist()}))
    (rd / "drop" / "asset.mint").write_bytes(b"x")
    res = providers.ManualProvider(rd).fetch(providers.load_request(rd), {})
    assert res.kind == "mint" and abs(res.world_from_scene["s"] - 0.5) < 1e-6
    assert res.world_from_scene["rot_err_deg"]["p95"] < 1e-3


def test_gracia_provider_marks_monocular_unsupported(tmp_path):
    work = _fake_take(tmp_path)
    grid = md.master_grid(10.0, 10.2)
    rd = _request(work, "g", rig.ring(len(grid), n=2), grid, provider="gracia")
    job = providers.make_provider("gracia", rd).submit(providers.load_request(rd))
    assert job["supported"] is False and "genlock" in job["why"]
    assert "NOT a supported input" in (rd / "request" / "REQUEST.md").read_text()


def test_worldlabs_provider_flow_with_mock_http(tmp_path):
    import httpx
    work = _fake_take(tmp_path)
    grid = md.master_grid(10.0, 10.2)
    rd = _request(work, "w", rig.ring(len(grid), n=2), grid, provider="worldlabs")
    Path(providers.load_request(rd).video).write_bytes(b"\x00" * 1000)
    calls = []

    def api(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path, request.headers.get("WLT-Api-Key")))
        p = request.url.path
        if p.endswith("media-assets:prepare_upload"):
            return httpx.Response(200, json={"media_asset": {"media_asset_id": "ma1"},
                                             "upload_info": {"upload_url": "https://store.example/u", "upload_method": "PUT",
                                                             "required_headers": {"Content-Type": "video/mp4"}}})
        if p.endswith("worlds:generate"):
            body = json.loads(request.content)
            assert body["world_prompt"]["video_prompt"] == {"source": "media_asset", "media_asset_id": "ma1"}
            return httpx.Response(200, json={"operation_id": "op1", "done": False})
        if p.endswith("operations/op1"):
            return httpx.Response(200, json={"done": True, "response": {"world_id": "w1", "model": "marble-1.1", "assets": {
                "splats": {"semantics_metadata": {"metric_scale_factor": 1.3, "ground_plane_offset": 0.2}}}}})
        if p.endswith("worlds/w1:export"):
            return httpx.Response(200, json={"operation_id": "op2", "done": True,
                                             "response": {"url": "https://store.example/w1.ply"}})
        return httpx.Response(404)

    def store(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, str(request.url), request.headers.get("WLT-Api-Key")))
        return httpx.Response(200, content=b"ply\n" if request.method == "GET" else b"")

    http = httpx.Client(base_url="https://api.example", headers={"WLT-Api-Key": "k"}, transport=httpx.MockTransport(api))
    raw = httpx.Client(transport=httpx.MockTransport(store))
    prov = providers.WorldLabsProvider(rd, {"poll_s": 0}, http=http, raw=raw)
    req = providers.load_request(rd)
    with pytest.raises(PermissionError, match="go-ahead"):
        prov.submit(req, spend_ok=True)
    req.allow_upload = True
    with pytest.raises(PermissionError, match="credits"):
        prov.submit(req)
    job = prov.submit(req, spend_ok=True)
    assert job["operation_id"] == "op1"
    res = prov.fetch(req, job)
    assert res.kind == "static_splat" and res.dynamic is False and res.scene_meta["metric_scale_factor"] == 1.3
    assert Path(res.asset).read_bytes() == b"ply\n"
    assert all(key is None for m, u, key in calls if u.startswith("https://store")), "the API key leaked to storage"


# ------------------------------------------------------------------------------------------------- judge


class FakeClient:
    """Stands in for TypeSafeClient: records requests, answers from a function of the question ids."""

    def __init__(self, answer):
        self.answer, self.requests = answer, []

    def system_one(self, state, questions, model=None, extra_headers=None, extra_body=None, **kw):
        self.requests.append({"state": state, "questions": questions, "headers": extra_headers, "body": extra_body})
        answers = {k: SimpleNamespace(noul=self.answer(k, state)) for k in questions}
        return SimpleNamespace(answers=answers, model="jev-test", usage=SimpleNamespace(input_tokens=100))


def _img(path: Path, size=(200, 300)):
    from PIL import Image
    Image.new("RGB", size, (120, 100, 90)).save(path)
    return path


def test_judge_request_shape_and_image_limits(tmp_path):
    v, r, s = _img(tmp_path / "v.png", (2000, 3000)), _img(tmp_path / "r.jpg", (100, 150)), _img(tmp_path / "s.jpg")
    req = judge.build_request(v, reference=r, source=s)
    assert set(req["questions"]) == {*judge.QUESTIONS, "same_pose"}
    refs = {req["state"]["view"], req["state"]["reference"], req["state"]["source"]}
    assert refs == set(req["assets"])  # every asset referenced, each exactly once
    from PIL import Image
    import io
    for a in req["assets"].values():
        im = Image.open(io.BytesIO(base64.b64decode(a["data_base64"])))
        assert a["format"] == "jpg" and im.format == "JPEG"
        assert 280 <= min(im.size) and max(im.size) <= judge.MAX_SIDE and im.width * im.height >= 78_400
    assert "`reference`" in req["state"]["about"] and "`source`" in req["state"]["about"]


def test_frame_verdict_decisive_vs_advisory():
    clean = {"duplicate_person": 0.1, "broken_anatomy": 0.3, "render_defects": 0.3, "cut_off": 0.99}
    assert judge.frame_verdict(clean) == (True, [])  # cut_off is advisory
    ok, why = judge.frame_verdict({**clean, "broken_anatomy": 0.7})  # decisive on real frames (2026-10-19)
    assert not ok and why == ["broken_anatomy 0.70"]
    assert judge.frame_verdict({**clean, "broken_anatomy": 0.7}, judge.PRESETS["mannequin"])[0]  # not on mannequins
    ok, why = judge.frame_verdict({**clean, "duplicate_person": 0.8})
    assert not ok and why[0].startswith("duplicate_person")
    ok, why = judge.frame_verdict({**clean, "same_pose": 0.2})
    assert not ok and why == ["same_pose 0.20"]


def test_judge_views_policy_and_consent(tmp_path):
    frames = {v: [{"t": i, "path": str(_img(tmp_path / f"{v}_{i}.jpg"))} for i in range(10)] for v in ("v00", "v01")}
    with pytest.raises(PermissionError):
        judge.judge_views(frames)
    # v01: one bad frame of ten -> kept (>= 2 bad frames AND > 15 % drop a view); v00: three bad -> dropped
    bad = {("v00", 0), ("v00", 4), ("v00", 7), ("v01", 3)}
    lookup = {}
    for v, fs in frames.items():
        for i, f in enumerate(fs):
            lookup[Path(f["path"]).name] = (v, i)

    client = FakeClient(lambda q, st: 0.0)
    orig = judge.judge_frame

    def fake_frame(path, reference=None, source=None, client=None, model=None, about=None):
        out = {q: 0.1 for q in judge.QUESTIONS}
        if lookup[Path(path).name] in bad:
            out["duplicate_person"] = 0.9
        return {**out, "_model": "jev-test", "_tokens": 10}

    judge.judge_frame = fake_frame
    try:
        rep = judge.judge_views(frames, allow_upload=True, client=client, workers=2)
    finally:
        judge.judge_frame = orig
    assert rep["views"]["v00"]["bad"] == 3 and not rep["views"]["v00"]["ok"]
    assert rep["views"]["v01"]["bad"] == 1 and rep["views"]["v01"]["ok"]
    assert rep["input_tokens"] == 200


def test_judge_frame_sends_image_header_and_assets(tmp_path):
    client = FakeClient(lambda q, st: 0.42)
    ans = judge.judge_frame(_img(tmp_path / "v.jpg", (400, 600)), client=client)
    sent = client.requests[0]
    assert sent["headers"] == judge.IMAGE_HEADERS and set(sent["body"]["assets"]) == {sent["state"]["view"]}
    assert ans["duplicate_person"] == 0.42 and ans["_model"] == "jev-test" and ans["_tokens"] == 100


# ------------------------------------------------------------------------------------------------- vendor watch


PAGE = """<html><head><script type="application/ld+json">{"@type": "FAQPage", "mainEntity": [{"@type": "Question",
"name": "What rig do I need?", "acceptedAnswer": {"@type": "Answer", "text": "Genlocked cameras are required."}}]}</script>
<script>var junk = 1;</script></head><body><nav>Menu</nav><main><h1>Reconstruct in 4D</h1><p>We make 4DGS
volumetric video.</p><p>Export .mint or a PLY sequence.</p></main><footer>(c)</footer></body></html>"""


def test_page_text_reads_visible_text_and_json_ld():
    text = vendor_watch.page_text(PAGE, is_html=True)
    assert "Reconstruct in 4D" in text and "Export .mint" in text and "Genlocked cameras are required." in text
    assert "junk" not in text and "Menu" not in text
    ps = vendor_watch.passages(text * 40, size=200)
    assert all(len(p["text"]) <= 200 + 100 for p in ps) and ps[0]["id"] == "p0"
    assert sum(len(p["text"]) for p in ps) <= vendor_watch.MAX_PAGE_CHARS + len(ps)


def test_vendor_decide_and_changes():
    def page(**caps):
        base = {c: {"stated": 0.05, "ruled_out": 0.05 if vendor_watch.CAPABILITIES[c]["against"] else None,
                    "evidence": ""} for c in vendor_watch.CAPABILITIES}
        for c, (s, r) in caps.items():
            base[c].update({"stated": s, "ruled_out": r})
        return {"url": "u", "capabilities": base}
    ready = [page(monocular_input=(0.9, 0.1), dynamic_output=(0.95, 0.0), self_serve_api=(0.8, 0.2)),
             page(export=(0.9, None))]
    d = vendor_watch.decide(ready)
    assert d["_ready"] and d["monocular_input"]["status"] == "yes" and d["camera_control"]["status"] == "unclear"
    d2 = vendor_watch.decide([page(monocular_input=(0.9, 0.8), dynamic_output=(0.95, 0.0), self_serve_api=(0.8, 0.2),
                                   export=(0.9, None))])
    assert d2["monocular_input"]["status"] == "no" and not d2["_ready"]  # a stated limitation wins
    prev = {"vendors": {"x": {"decision": d2}}}
    msgs = vendor_watch.changes(prev, {"vendors": {"x": {"decision": d}}})
    assert "x: monocular_input no -> yes" in msgs and "x: ready False -> True" in msgs


def test_vendor_run_with_fake_page_and_client(tmp_path):
    class Http:
        def get(self, url, **kw):
            return SimpleNamespace(text=PAGE, content=PAGE.encode(), headers={"content-type": "text/html; charset=utf-8"},
                                   raise_for_status=lambda: None)

    def answer(q, state):
        return 0.9 if q in ("dynamic_output_stated", "export_stated", "monocular_input_ruled_out") else 0.05

    class Client(FakeClient):
        def system_one(self, state, questions, model=None, **kw):
            r = super().system_one(state, questions, model=model, **kw)
            for k, q in questions.items():
                if k.endswith("_evidence"):
                    r.answers[k] = SimpleNamespace(choice="p0", confidence=0.8)
            return r

    doc = vendor_watch.run(["gracia"], save=True, client=Client(answer), http=Http(), out=tmp_path)
    d = doc["vendors"]["gracia"]["decision"]
    assert d["monocular_input"]["status"] == "no" and d["dynamic_output"]["status"] == "yes" and not d["_ready"]
    assert len(list(tmp_path.glob("*.json"))) == 1 and doc["changes"] == []


# ------------------------------------------------------------------------------------------------- gate


def _gate_run(tmp_path, kind="views", dynamic=True, judged_ok=None, views2d=None, wfs=None):
    rd = tmp_path / "run"
    (rd / "request").mkdir(parents=True)
    specs = rig.ring(2, n=8, radius=4.0) + [rig.ViewSpec("h03", "holdout", *[x[:2] for x in (
        rig.ring(2, n=1)[0].K, rig.ring(2, n=1)[0].R, rig.ring(2, n=1)[0].t)], 1080, 1920)]
    (rd / "request" / "target_trajectory.json").write_text(json.dumps(rig.trajectory(specs, np.arange(2) / 30)))
    views = {s.name: f"views/{s.name}.mp4" for s in specs}
    (rd / "result.json").write_text(json.dumps({"provider": "manual", "kind": kind, "dynamic": dynamic, "views": views,
                                                "world_from_scene": wfs}))
    judged_ok = judged_ok if judged_ok is not None else {s.name: True for s in specs}
    (rd / "judge.json").write_text(json.dumps({"views": {
        k: {"ok": v, "bad": 0 if v else 5, "judged": 10, "bad_frac": 0.0 if v else 0.5} for k, v in judged_ok.items()}}))
    if views2d is not None:
        (rd / "views2d.json").write_text(json.dumps(views2d))
    return rd


def test_gate_admits_a_clean_run_and_keeps_holdout_views_out(tmp_path):
    out = gate.decide(_gate_run(tmp_path, views2d={"h03": {"cam": "03", "extra_people_frac": 0.0,
                                                           "pck_0.1": {"median": 0.9}, "mask_iou": {"median": 0.8}}}))
    assert out["admit"] and out["views_admitted"] == [f"v{i:02d}" for i in range(8)]
    assert out["checks"]["azimuth_span_deg"] == 315.0


def test_gate_failures(tmp_path):
    assert not gate.decide(_gate_run(tmp_path / "a", kind="static_splat", dynamic=False))["admit"]
    bad2d = {"h03": {"cam": "03", "extra_people_frac": 0.3, "pck_0.1": {"median": 0.9}, "mask_iou": {"median": 0.8}}}
    out = gate.decide(_gate_run(tmp_path / "b", views2d=bad2d))
    assert not out["admit"] and "extra people" in out["why"][0]
    # Jev drops the whole back half of the ring: 4 views left, but they span only 135 degrees
    half = {**{f"v{i:02d}": i < 4 for i in range(8)}, "h03": True}
    out = gate.decide(_gate_run(tmp_path / "c", judged_ok=half))
    assert not out["admit"] and "span 135 deg" in out["why"][0] and len(out["views_dropped"]) == 4
    wfs = {"s": 1.0, "R": np.eye(3).tolist(), "t": [0, 0, 0], "rms": 0.2, "rot_err_deg": {"median": 4.0, "p95": 9.0}}
    out = gate.decide(_gate_run(tmp_path / "d", wfs=wfs))
    assert not out["admit"] and len([w for w in out["why"] if w.startswith("scene frame")]) == 2


def test_azimuth_span():
    ring = rig.ring(1, n=8)
    assert gate.azimuth_span_deg(np.array([s.centres[0] for s in ring]), around=(0, 0)) == pytest.approx(315.0)
    assert gate.azimuth_span_deg(np.array([s.centres[0] for s in ring[:3]]), around=(0, 0)) == pytest.approx(90.0)


# ------------------------------------------------------------------------------------------------- input cut


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
def test_cut_input_video_keeps_every_frame(tmp_path):
    t = 3.0 + np.cumsum(np.r_[0, np.full(44, 1 / 30.0)])  # 45 frames at 30 fps (shorter than ffmpeg's 1/25 default)
    _frames(tmp_path, "06", t, size=(320, 480))
    info = providers.cut_input_video(tmp_path / "frames" / "06", 3.1, 4.2, tmp_path / "in" / "06.mp4")
    sel = t[(t >= 3.1) & (t <= 4.2)]
    assert info["frames"] == len(sel) and abs(info["fps"] - 30.0) < 1e-6
    assert providers.video_info(tmp_path / "in" / "06.mp4")["frames"] == len(sel)
    assert json.loads((tmp_path / "in" / "06.times.json").read_text())["t"] == pytest.approx(sel.tolist())
    assert not any((tmp_path / "in").glob("_*"))  # the temporary sequence folder is gone
