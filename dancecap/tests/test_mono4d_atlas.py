r"""Unit tests for dancecap.mono4d.atlas (World Labs Atlas beta client): cameras, canvases, media, the client against a
mock server, the Sim3 from room points. No network, no key.

    .venv-nopo4d\Scripts\python.exe -m pytest dancecap/tests/test_mono4d_atlas.py -q      (from the workspace root)
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dancecap import mamma_dyn as md  # noqa: E402
from dancecap.mono4d import atlas as A  # noqa: E402


def look(C, target=(0.0, 0.9, 0.0), roll=0.0):
    C = np.asarray(C, float)[None]
    R = md._look_at(C, np.asarray(target, float)[None], np.array([roll]))[0]
    return R, -R @ C[0]


K_PORTRAIT = np.array([[1150.0, 0, 540], [0, 1150.0, 960], [0, 0, 1]])


@pytest.mark.parametrize("conv", ["rub", "rdf"])
def test_camera_round_trip_both_conventions(conv):
    R, t = look([2.0, 1.4, 3.5], roll=0.2)
    cam = A.to_atlas_camera(K_PORTRAIT, R, t, 1080, 1920, conv)
    assert cam["extrinsics"]["coordinateSystem"] == conv and len(cam["extrinsics"]["quaternion"]) == 4
    K2, R2, t2, W, H = A.from_atlas_camera(cam)
    assert np.allclose(R2, R) and np.allclose(t2, t) and np.allclose(K2, K_PORTRAIT) and (W, H) == (1080, 1920)


def test_rub_camera_looks_along_minus_z_and_up_is_up():
    R, t = look([0.0, 1.5, 4.0], target=(0.0, 1.5, 0.0))  # looking along -z, level
    cam = A.to_atlas_camera(K_PORTRAIT, R, t, 1080, 1920, "rub")
    Rc2w = A.R_from_xyzw(cam["extrinsics"]["quaternion"])
    assert np.allclose(Rc2w[:, 1], [0, 1, 0], atol=1e-9)    # camera up = world up
    assert np.allclose(-Rc2w[:, 2], [0, 0, -1], atol=1e-9)  # looks along -Z (back = +z)
    assert np.allclose(Rc2w, np.eye(3), atol=1e-9)          # so the quaternion is the identity, as Atlas returns


def test_generation_camera_keeps_the_portrait_field_of_view():
    R, t = look([2.0, 1.4, 3.5])
    Kg, box = A.generation_camera(K_PORTRAIT, R, t, 1080, 1920)
    assert box == (437, 0, 842, 720)
    X = np.array([[0.3, 1.0, 0.2], [-0.4, 0.2, 0.1]])
    p_phone, _ = md.project(X, K_PORTRAIT, R, t)
    p_grid, _ = md.project(X, Kg, R, t)
    s = 720 / 1920
    # same rays: the phone image is the centred band, starting at 640 - 540 s = 437.5 px (the integer crop box rounds it)
    assert np.allclose(p_grid, p_phone * s + [640 - 540 * s, 0], atol=1e-6)


def test_landscape_canvas_intrinsics_follow_the_pixels():
    rgb = np.zeros((1280, 720, 3), np.uint8)
    rgb[600:610, 300:310] = 255
    keep = np.ones((1280, 720), bool)
    cam = {"extrinsics": {"position": [0, 1, 0], "quaternion": [0, 0, 0, 1], "coordinateSystem": "rub"},
           "intrinsics": {"fx": 760.0, "fy": 760.0, "cx": 360.0, "cy": 640.0, "width": 720, "height": 1280}}
    canvas, k, c2, box = A.landscape_canvas(rgb, keep, cam)
    assert canvas.shape == (720, 1280, 3) and k.shape == (720, 1280)
    x0, _, x1, _ = box
    assert not k[:, :x0].any() and not k[:, x1:].any() and k[:, x0 + 1:x1 - 1].all()
    it = c2["intrinsics"]
    assert (it["width"], it["height"]) == (1280, 720) and abs(it["cx"] - 640.0) < 1.0 and abs(it["fx"] - 427.5) < 1e-6
    ys, xs = np.nonzero(canvas[..., 0] > 128)
    assert abs(xs.mean() - (305 * 0.5625 + x0)) < 2 and abs(ys.mean() - 605 * 0.5625) < 2


def test_media_round_trips():
    z = (np.random.default_rng(0).random((40, 30)) * 9 + 0.5).astype(np.float32)
    exr = A.decode_data_url(A.b64_exr(z))
    assert np.allclose(A.read_exr(exr), z)
    m = np.zeros((20, 10), bool)
    m[5:9] = True
    from PIL import Image
    import io
    png = Image.open(io.BytesIO(A.decode_data_url(A.b64_mask(m))))
    assert png.mode == "L" and np.array_equal(np.asarray(png) > 127, m)
    img = np.full((30, 40, 3), 128, np.uint8)
    assert A.b64_image(img).startswith("data:image/jpeg;base64,")
    assert A.read_image(A.decode_data_url(A.b64_image(img, "PNG"))).shape == (30, 40, 3)


def test_world_from_atlas_recovers_a_known_sim3():
    rng = np.random.default_rng(3)
    R, t = look([-3.0, 0.8, -2.1])
    W, H = 1080, 1920
    room = np.concatenate([rng.uniform([-6, 0, -6], [6, 3, -5.5], (400, 3)), rng.uniform([-6, 0, -6], [6, 0.01, 6], (400, 3)),
                           rng.uniform([5.5, 0, -6], [6, 3, 6], (400, 3))])
    # Atlas's scene: our world rotated by 40 deg yaw, scaled by 1/1.2, shifted; its camera = our camera expressed there
    sim3 = {"s": 1.2, "R": md._look_at(np.zeros((1, 3)), np.array([[np.sin(0.7), 0, np.cos(0.7)]]), np.zeros(1))[0].T.tolist(),
            "t": [0.5, -0.1, 0.3]}
    Ra, ta = md.sfm_cameras(R[None], t[None], sim3)
    Wa, Ha = 720, 1280
    Ka = K_PORTRAIT * np.array([[Wa / W], [Ha / H], [1]])
    cam = A.to_atlas_camera(Ka, Ra[0], ta[0], Wa, Ha, "rub")
    # Atlas depth of the room in scene units on its grid: rasterise the points' depth (nearest point per pixel)
    Xc = room @ R.T + t
    z = np.full((Ha, Wa), np.inf, np.float32)
    for X in Xc[Xc[:, 2] > 0.3]:
        u, v = int(Ka[0, 0] * X[0] / X[2] + Ka[0, 2]), int(Ka[1, 1] * X[1] / X[2] + Ka[1, 2])
        if 0 <= u < Wa and 0 <= v < Ha:
            for du in range(-2, 3):
                for dv in range(-2, 3):
                    uu, vv = min(max(u + du, 0), Wa - 1), min(max(v + dv, 0), Ha - 1)
                    z[vv, uu] = min(z[vv, uu], X[2] / 1.2)
    view = A.PosedView(camera=cam, image={"assetId": "a"}, z=z)
    est = A.world_from_atlas(view, K_PORTRAIT, R, t, W, H, room)
    assert abs(est["s"] - 1.2) < 0.02 and est["n_points"] > 100 and abs(est["focal_ratio"] - 1.0) < 1e-6
    assert np.allclose(est["R"], sim3["R"], atol=1e-6) and np.allclose(est["t"], sim3["t"], atol=0.05)
    # and a held-out camera sent through camera_in_atlas lands where the true Sim3 puts it
    R2, t2 = look([3.0, 1.6, 2.0])
    ca, box = A.camera_in_atlas(K_PORTRAIT, R2, t2, W, H, est)
    _, Rb, tb, _, _ = A.from_atlas_camera(ca)
    Rt, tt = md.sfm_cameras(R2[None], t2[None], sim3)
    assert np.allclose(Rb, Rt[0], atol=1e-6) and np.allclose(-Rb.T @ tb, -Rt[0].T @ tt[0], atol=0.05)


def test_client_submit_wait_and_fetch_with_mock_server():
    import httpx
    calls = []
    png = A.decode_data_url(A.b64_image(np.zeros((8, 8, 3), np.uint8), "PNG"))
    exr = A.decode_data_url(A.b64_exr(np.ones((8, 8), np.float32) * 2.5))

    def api(req: httpx.Request) -> httpx.Response:
        calls.append((req.method, req.url.path, req.url.params.get("wait"), req.headers.get("WLT-Api-Key"),
                      req.headers.get("Idempotency-Key")))
        if req.url.path.endswith("tasks:images2PosedRGBD"):
            body = json.loads(req.content)
            assert body["targetResolution"] == [720, 1280] and body["frames"][0]["imageAsset"]["base64"].startswith("data:")
            return httpx.Response(200, json={"id": "op1", "done": False})
        if req.url.path.endswith("operations/op1:wait"):
            return httpx.Response(200, json={"id": "op1", "done": True, "error": None, "response": {"frames": [{
                "imageAsset": {"assetId": "img1", "url": "https://store.example/img1.png"},
                "camera": {"extrinsics": {"position": [0, 0.6, 0], "quaternion": [0, 0, 0, 1], "coordinateSystem": "rub"},
                           "intrinsics": {"fx": 700, "fy": 700, "cx": 4, "cy": 4, "width": 8, "height": 8}},
                "depth": {"depthAsset": {"assetId": "d1"}}, "centroidWorld": [0, 0, -4]}]}})
        if req.url.path.endswith("assets/d1:createReadUrl"):
            return httpx.Response(200, json={"readUrl": "https://store.example/d1.exr"})
        if req.url.path.endswith("tasks:atlasGenerate"):
            return httpx.Response(422, json={"detail": [{"msg": "bad camera", "input": {"huge": "x" * 5000}}]})
        return httpx.Response(404)

    def store(req: httpx.Request) -> httpx.Response:
        calls.append((req.method, str(req.url), None, req.headers.get("WLT-Api-Key"), None))
        return httpx.Response(200, content=png if str(req.url).endswith(".png") else exr)

    c = A.AtlasClient(key="k", base_url="https://api.example/api/v2",
                      http=httpx.Client(base_url="https://api.example/api/v2", headers={"WLT-Api-Key": "k"},
                                        transport=httpx.MockTransport(api)),
                      raw=httpx.Client(transport=httpx.MockTransport(store)))
    views, meta = A.pose_images(c, [np.zeros((16, 9, 3), np.uint8)], target_resolution=(720, 1280))
    v = views[0]
    assert v.rgb.shape == (8, 8, 3) and float(v.z.mean()) == 2.5 and v.extra["centroidWorld"] == [0, 0, -4]
    ctx = v.as_context()
    assert ctx["imageAsset"] == {"assetId": "img1"} and ctx["depth"]["depthAsset"] == {"assetId": "d1"}
    sub = [x for x in calls if x[1].endswith("images2PosedRGBD")][0]
    assert sub[2] == "true" and sub[3] == "k" and sub[4].startswith("dancecap-images2PosedRGBD-")
    assert all(key is None for m, u, w, key, i in calls if str(u).startswith("https://store")), "key leaked to storage"
    target = A.to_atlas_camera(np.array([[900.0, 0, 640], [0, 900, 360], [0, 0, 1]]), np.eye(3), np.zeros(3), 1280, 720)
    with pytest.raises(A.AtlasError) as e:
        A.generate(c, [v], [target])
    assert "bad camera" in str(e.value) and "xxxx" not in str(e.value)  # the request echo is stripped
    with pytest.raises(ValueError, match="1280x720"):
        A.generate(c, [v], [A.to_atlas_camera(K_PORTRAIT, np.eye(3), np.zeros(3), 1080, 1920)])


def test_read_colmap_points(tmp_path):
    import struct
    p = tmp_path / "points3D.bin"
    pts = [(1.0, 2.0, 3.0), (-1.5, 0.25, 9.0)]
    with open(p, "wb") as f:
        f.write(struct.pack("<Q", len(pts)))
        for i, (x, y, z) in enumerate(pts):
            f.write(struct.pack("<Qddd", i + 1, x, y, z) + bytes([1, 2, 3]) + struct.pack("<d", 0.5))
            f.write(struct.pack("<Q", 2) + struct.pack("<ii", 7, 1) + struct.pack("<ii", 8, 4))
    assert np.allclose(A.read_colmap_points(p), pts)
