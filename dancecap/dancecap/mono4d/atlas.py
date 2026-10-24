"""World Labs Atlas beta (the Marble 2 Developer API) client, and the dancecap uses of it (built 2026-10-19).

API root https://api.atlas-beta.worldlabs.ai/api/v2 (WLT_API_BASE_URL overrides), header WLT-Api-Key; the key comes from
WLT_API_KEY or the file named by WORLDLABS_ATLAS_KEY_FILE (default ~/Desktop/atlas-key.txt) and is never logged.
Docs: http://atlas-beta.worldlabs.ai/docs/<page>.md (quickstart, how-to-use-atlas, cameras-and-posed-images, operations,
best-practices, api-reference/*). What the beta is: IMAGE tasks on posed RGBD views, 1280x720 generation grid, no video
or temporal task; its own best-practices page says the models are not yet strong on people or dynamic objects.

Measured 2026-10-19 on larissa_kadu_38 phone 06 at 59.98 s (docs/MONO4D_NOVEL_VIEWS.md section 10):
    images2PosedRGBD   GOOD: from one portrait frame, focal within 0.6 % of our calibration, gravity within ~1 deg, depth
                       near metric (x1.11 against 3,586 room SfM points, IQR 1.04-1.25); 2.7 s
    atlasGenerate      dancers at four held-out phones' poses: placed right, posed wrong (YOLO PCK@0.1 0.0-0.67, gate 0.8)
    atlasMasked        removing the people refills the mask with melted figures (Jev: person 1.00, plausible room 0.06)

Tasks used (all accept inline base64 assets < 10 MiB decoded, or asset ids; `?wait=true` answers fast tasks at once):
    images2PosedRGBD   images -> camera (c2w, intrinsics), linear-depth EXR, confidence, centroidWorld per image
    atlasGenerate      posed RGBD context frames + target cameras -> RGB (+ depth) at the targets (1280x720)
    atlasMasked        posed RGB + alpha mask (1 keep, 0 fill) per frame -> completed views (people removed)
    splats2Mesh        a .spz splat scene -> GLB mesh (image texture / vertex colours), decimated to targetFaces

Cameras. Atlas: camera-to-world, XYZW quaternion, coordinateSystem "rdf" (OpenCV axes) or "rub" (OpenGL axes), Y-up
world, positions in the scene's distance unit. dancecap: world->camera R, t (OpenCV axes), Y-up metres (CONTRACTS.md).
to_atlas_camera / from_atlas_camera convert; world_from_atlas() solves the Sim3 between Atlas's frame for a posed phone
frame and our world (rotation from the two camera orientations, scale from the take's room SfM points: their metric
depth along our camera vs Atlas's depth at the same pixels, people masked out - the dancers never set the scale).
"""
from __future__ import annotations

import base64
import io
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")  # Atlas depth comes as EXR; cv2 reads it only with this set

BASE_URL = "https://api.atlas-beta.worldlabs.ai/api/v2"
GEN_SIZE = (1280, 720)  # atlasGenerate / atlasMasked target grid (width, height)
TASKS = ("images2PosedRGBD", "atlasGenerate", "atlasMasked", "splats2Mesh", "atlasChisel", "atlasTextToImage")


# ------------------------------------------------------------------------------------------------- cameras


RUB_FROM_RDF = np.diag([1.0, -1.0, -1.0])  # camera-local axes: (right, down, forward) -> (right, up, back)


def quat_xyzw(R: np.ndarray) -> list[float]:
    from .rig import qvec_wxyz
    w, x, y, z = qvec_wxyz(R)
    return [float(x), float(y), float(z), float(w)]


def R_from_xyzw(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def to_atlas_camera(K: np.ndarray, R: np.ndarray, t: np.ndarray, width: int, height: int,
                    convention: str = "rub") -> dict:
    """Our world->camera (OpenCV) camera -> an Atlas PinholeCamera (c2w, XYZW quaternion).

    RUB by default, the convention Atlas returns its own cameras in: a 2026-10-19 live test with RDF target cameras next
    to an RUB context camera came back upside down and facing away (the request mixed conventions), so every camera of
    a request should use the same one."""
    R, t = np.asarray(R, float), np.asarray(t, float)
    Rc2w = R.T if convention == "rdf" else R.T @ RUB_FROM_RDF
    return {"extrinsics": {"position": (-R.T @ t).tolist(), "quaternion": quat_xyzw(Rc2w), "coordinateSystem": convention},
            "intrinsics": {"fx": float(K[0, 0]), "fy": float(K[1, 1]), "cx": float(K[0, 2]), "cy": float(K[1, 2]),
                           "width": int(width), "height": int(height)}}


def from_atlas_camera(cam: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """An Atlas PinholeCamera -> (K, R world->camera OpenCV, t, width, height)."""
    ex, it = cam["extrinsics"], cam["intrinsics"]
    Rc2w = R_from_xyzw(ex["quaternion"])
    if ex.get("coordinateSystem", "rub") == "rub":
        Rc2w = Rc2w @ RUB_FROM_RDF  # columns: camera right/up/back -> right/down/forward
    C = np.asarray(ex["position"], float)
    R = Rc2w.T
    K = np.array([[it["fx"], 0.0, it["cx"]], [0.0, it["fy"], it["cy"]], [0.0, 0.0, 1.0]])
    return K, R, -R @ C, int(it["width"]), int(it["height"])


def generation_camera(K: np.ndarray, R: np.ndarray, t: np.ndarray, width: int, height: int,
                      size: tuple[int, int] = GEN_SIZE) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Intrinsics on Atlas's 1280x720 grid that keep a (portrait) phone's whole field of view: the phone's height maps
    to the grid height, its width to a centred band. -> (K on the grid, crop box (x0, y0, x1, y1) of the phone image)."""
    W, H = size
    s = H / height
    Kg = np.array([[K[0, 0] * s, 0.0, W / 2 + (K[0, 2] - width / 2) * s],
                   [0.0, K[1, 1] * s, H / 2 + (K[1, 2] - height / 2) * s], [0.0, 0.0, 1.0]])
    bw = int(round(width * s))
    x0 = (W - bw) // 2
    return Kg, (x0, 0, x0 + bw, H)


# ------------------------------------------------------------------------------------------------- media


def b64_image(img, fmt: str = "JPEG", quality: int = 92) -> str:
    """A path, PIL image or HxWx3 uint8 array -> a data URL (TaskAsset.base64)."""
    from PIL import Image
    if isinstance(img, (str, Path)):
        img = Image.open(img).convert("RGB")
    elif isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    buf = io.BytesIO()
    img.save(buf, fmt, **({"quality": quality} if fmt == "JPEG" else {}))
    mime = "image/jpeg" if fmt == "JPEG" else "image/png"
    return f"data:{mime};base64," + base64.b64encode(buf.getvalue()).decode()


def b64_mask(mask: np.ndarray) -> str:
    """HxW bool/uint8 (True or 255 = keep) -> an 8-bit grayscale PNG data URL."""
    from PIL import Image
    m = np.asarray(mask)
    m = (m > 0).astype(np.uint8) * 255 if m.dtype == bool or m.max() <= 1 else m.astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(m, "L").save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def decode_data_url(s: str) -> bytes:
    return base64.b64decode(s.split(",", 1)[1] if s.startswith("data:") else s)


def read_exr(data: bytes) -> np.ndarray:
    """EXR bytes -> float32 HxW (first channel), rows top-down as Atlas declares them."""
    import cv2
    try:
        arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    except cv2.error:
        arr = None
    if arr is None:
        # OpenCV wheels without the EXR codec (opencv-python 5.0 on the Ubuntu PC): the OpenEXR bindings read it
        try:
            import OpenEXR
        except ImportError:
            raise ValueError("could not decode the EXR (OpenCV built without OpenEXR; pip install OpenEXR)") from None
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".exr") as fh:
            fh.write(data)
            fh.flush()
            with OpenEXR.File(fh.name) as f:
                ch = f.channels()
                name = next((k for k in ("Z", "Y", "R", "depth") if k in ch), sorted(ch)[0])
                arr = np.asarray(ch[name].pixels, np.float32)
    return arr if arr.ndim == 2 else arr[..., 0]


def b64_exr(z: np.ndarray) -> str:
    """HxW float32 linear depth -> an EXR data URL (inline LinearDepthBufferAsset.depthAsset)."""
    import cv2
    ok, buf = cv2.imencode(".exr", np.ascontiguousarray(z, np.float32))
    if not ok:
        raise ValueError("could not encode the EXR (OpenCV built without OpenEXR?)")
    return "data:image/x-exr;base64," + base64.b64encode(buf.tobytes()).decode()


def read_image(data: bytes) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))


# ------------------------------------------------------------------------------------------------- client


class AtlasError(RuntimeError):
    pass


def _strip_inputs(obj):
    """An error body without the request echo (422 details repeat the whole input, base64 images included)."""
    if isinstance(obj, dict):
        return {k: _strip_inputs(v) for k, v in obj.items() if k != "input"}
    if isinstance(obj, list):
        return [_strip_inputs(v) for v in obj]
    return obj


class AtlasClient:
    """Thin client: submit with an idempotency key, wait / poll, read outputs. http / raw are injectable for tests;
    raw (signed asset URLs) never carries the API key."""

    def __init__(self, key: str | None = None, base_url: str | None = None, http=None, raw=None, log=print):
        self.base_url = (base_url or os.environ.get("WLT_API_BASE_URL") or BASE_URL).rstrip("/")
        self._key, self._http, self._raw, self.log = key, http, raw, log

    def key(self) -> str:
        if self._key:
            return self._key
        k = os.environ.get("WLT_API_KEY")
        if not k:
            p = Path(os.environ.get("WORLDLABS_ATLAS_KEY_FILE", Path.home() / "Desktop" / "atlas-key.txt"))
            if not p.exists():
                raise AtlasError("no Atlas key: set WLT_API_KEY or put it in ~/Desktop/atlas-key.txt")
            k = p.read_text().strip()
        self._key = k
        return k

    def http(self):
        if self._http is None:
            import httpx
            self._http = httpx.Client(base_url=self.base_url, timeout=180.0, headers={"WLT-Api-Key": self.key()})
        return self._http

    def raw(self):
        if self._raw is None:
            import httpx
            self._raw = httpx.Client(timeout=300.0, follow_redirects=True)
        return self._raw

    def _req(self, method: str, path: str, **kw) -> dict:
        r = self.http().request(method, path, **kw)
        if r.status_code >= 400:
            try:
                body = _strip_inputs(r.json())
            except ValueError:
                body = r.text
            raise AtlasError(f"{method} {path}: HTTP {r.status_code}: {str(body)[:800]}")
        return r.json() if r.content else {}

    def health(self) -> dict:
        import httpx
        r = httpx.get(self.base_url + "/health", timeout=30.0)
        r.raise_for_status()
        return r.json() if r.content else {"status": r.status_code}

    def submit(self, task: str, body: dict, idempotency_key: str | None = None, wait: bool = True) -> dict:
        if task not in TASKS:
            raise ValueError(f"unknown Atlas task {task!r}")
        key = idempotency_key or f"dancecap-{task}-{uuid.uuid4()}"
        return self._req("POST", f"/tasks:{task}", params={"wait": "true"} if wait else None, json=body,
                         headers={"Idempotency-Key": key})

    def operation(self, op_id: str) -> dict:
        return self._req("GET", f"/operations/{op_id}")

    def result(self, op: dict, timeout_s: float = 900.0, poll_s: float = 3.0) -> dict:
        """The terminal operation (waits server-side, then polls with capped backoff) -> response; raises on error."""
        t0 = time.time()
        while not op.get("done"):
            if time.time() - t0 > timeout_s:
                raise AtlasError(f"operation {op.get('id')} still running after {timeout_s:.0f} s")
            try:
                op = self._req("POST", f"/operations/{op['id']}:wait", json={"timeoutSecs": 30})
            except AtlasError:
                time.sleep(poll_s)
                op = self.operation(op["id"])
                poll_s = min(poll_s * 1.6, 20.0)
        if op.get("error"):
            raise AtlasError(f"operation {op.get('id')} failed: {op['error']}")
        return op.get("response") or {}

    def upload(self, data: bytes, name: str, content_type: str = "image/jpeg") -> str:
        """An asset from bytes through an upload URL (create -> createUploadUrl -> PUT -> completeUpload) -> its id. Inline
        base64 inputs share one request-size limit (HTTP 413 at 99 phone keyframes): uploaded assets are referenced by id."""
        a = self._req("POST", "/assets", json={"asset": {"displayName": name}})
        aid = a["id"]
        g = self._req("POST", f"/assets/{aid}:createUploadUrl", json={"contentType": content_type, "contentLength": len(data)})
        r = self.raw().request(g.get("method", "PUT"), g["uploadUrl"], content=data, headers=g.get("headers") or {"Content-Type": content_type})
        if r.status_code >= 400:
            raise AtlasError(f"upload of {name}: HTTP {r.status_code}: {r.text[:300]}")
        self._req("POST", f"/assets/{aid}:completeUpload", json={})
        return aid

    def fetch(self, asset: dict) -> bytes:
        """Bytes of a TaskAsset: inline base64, its signed url, or a fresh read URL for its assetId."""
        if asset.get("base64"):
            return decode_data_url(asset["base64"])
        url = asset.get("url")
        if not url and asset.get("assetId"):
            url = self._req("POST", f"/assets/{asset['assetId']}:createReadUrl")["readUrl"]
        if not url:
            raise AtlasError(f"asset has no bytes, url or id: {asset}")
        r = self.raw().get(url)
        if r.status_code == 403 and asset.get("assetId"):  # an expired signed URL: mint a fresh one
            r = self.raw().get(self._req("POST", f"/assets/{asset['assetId']}:createReadUrl")["readUrl"])
        r.raise_for_status()
        return r.content


# ------------------------------------------------------------------------------------------------- task helpers


@dataclass
class PosedView:
    """One posed RGBD view as Atlas returns it, plus our decoded copies."""
    camera: dict                         # Atlas PinholeCamera
    image: dict                          # TaskAsset (assetId / url)
    depth: dict | None = None            # LinearDepthBufferAsset {depthAsset, confidenceAsset}
    rgb: np.ndarray | None = None        # HxWx3 uint8
    z: np.ndarray | None = None          # HxW float32 linear depth (scene units)
    conf: np.ndarray | None = None       # HxW float32 0..1
    extra: dict = field(default_factory=dict)

    def as_context(self) -> dict:
        """PosedRGBDAsset for atlasGenerate (asset ids when Atlas returned them: the cheapest input; else the decoded
        depth inline as EXR)."""
        d = {"imageAsset": _ref(self.image), "camera": self.camera}
        if not (self.depth and self.depth.get("depthAsset", {}).get("assetId")) and self.z is not None:
            d["depth"] = {"depthAsset": {"base64": b64_exr(self.z)}}
        elif self.depth:
            d["depth"] = {"depthAsset": _ref(self.depth["depthAsset"])}
            if self.depth.get("confidenceAsset"):
                d["depth"]["confidenceAsset"] = _ref(self.depth["confidenceAsset"])
        return d


def _ref(a: dict) -> dict:
    return {"assetId": a["assetId"]} if a.get("assetId") else {k: v for k, v in a.items() if k in ("url", "base64")}


def _decode(client: AtlasClient, frame: dict, want_depth: bool = True) -> PosedView:
    v = PosedView(camera=frame["camera"], image=frame["imageAsset"], depth=frame.get("depth"),
                  extra={k: frame[k] for k in frame if k not in ("camera", "imageAsset", "depth")})
    v.rgb = read_image(client.fetch(frame["imageAsset"]))
    if want_depth and v.depth:
        v.z = read_exr(client.fetch(v.depth["depthAsset"])).astype(np.float32)
        if v.depth.get("confidenceAsset"):
            c = read_image(client.fetch(v.depth["confidenceAsset"]))
            v.conf = c[..., 0].astype(np.float32) / 255.0
    return v


def pose_images(client: AtlasClient, images: list, target_resolution: tuple[int, int] | None = None,
                refine: bool = False, canonicalize: bool = False, upload: bool | None = None) -> tuple[list[PosedView], dict]:
    """images2PosedRGBD on one or more images (paths / arrays), posed together in ONE frame. Pass the phone's own aspect as
    target_resolution (e.g. (720, 1280) for a portrait phone): the default 1280x720 crops a portrait frame to its middle band,
    and one request has one target size (a portrait and a landscape camera cannot share a request). upload (default: more than
    12 images) sends asset ids instead of inline base64 (the inline request has a size limit)."""
    upload = len(images) > 12 if upload is None else upload
    if upload:
        frames = [{"imageAsset": {"assetId": client.upload(decode_data_url(b64_image(im)), f"pose-{k:03d}.jpg")}}
                  for k, im in enumerate(images)]
    else:
        frames = [{"imageAsset": {"base64": b64_image(im)}} for im in images]
    body = {"frames": frames,
            "canonicalizeToFirstFrame": canonicalize, "refinePoses": refine}
    if target_resolution:
        body["targetResolution"] = [int(target_resolution[0]), int(target_resolution[1])]
    resp = client.result(client.submit("images2PosedRGBD", body))
    return [_decode(client, f) for f in resp["frames"]], {k: v for k, v in resp.items() if k != "frames"}


def generate(client: AtlasClient, context: list[PosedView], targets: list[dict], prompt: str | None = None,
             enhance: bool = True, return_depth: bool = False, seed: int = 42) -> tuple[list[PosedView], dict]:
    """atlasGenerate: context posed RGBD views + target PinholeCameras (1280x720 grid) -> generated views."""
    for c in targets:
        if (c["intrinsics"]["width"], c["intrinsics"]["height"]) != GEN_SIZE:
            raise ValueError(f"atlasGenerate targets must be {GEN_SIZE[0]}x{GEN_SIZE[1]}")
    body = {"contextFrames": [v.as_context() for v in context], "targetCameras": targets, "enhancePrompt": enhance,
            "returnDepth": return_depth, "modelParameters": {"seed": seed}}
    if prompt:
        body["prompt"] = prompt
    resp = client.result(client.submit("atlasGenerate", body))
    return [_decode(client, f, want_depth=return_depth) for f in resp.get("frames") or []], \
        {k: v for k, v in resp.items() if k != "frames"}


def landscape_canvas(rgb: np.ndarray, keep: np.ndarray, camera: dict, size: tuple[int, int] = GEN_SIZE):
    """A portrait (or any) posed frame -> Atlas's 1280x720 grid for atlasMasked: scaled to the grid height, centred,
    the side bands marked 'fill' (mask 0) so the model outpaints the room beyond the phone's view as well.
    -> (rgb 720x1280, keep, PinholeCamera with the intrinsics of the canvas, crop box of the original)."""
    from PIL import Image
    W, H = size
    h, w = rgb.shape[:2]
    s = H / h
    bw = int(round(w * s))
    x0 = (W - bw) // 2
    canvas = np.zeros((H, W, 3), np.uint8)
    canvas[:, x0:x0 + bw] = np.asarray(Image.fromarray(rgb).resize((bw, H), Image.LANCZOS))
    k = np.zeros((H, W), bool)
    k[:, x0:x0 + bw] = np.asarray(Image.fromarray(keep.astype(np.uint8) * 255).resize((bw, H), Image.NEAREST)) > 127
    it = camera["intrinsics"]
    cam = {"extrinsics": camera["extrinsics"],
           "intrinsics": {"fx": it["fx"] * s, "fy": it["fy"] * s, "cx": it["cx"] * s + x0, "cy": it["cy"] * s,
                          "width": W, "height": H}}
    return canvas, k, cam, (x0, 0, x0 + bw, H)


def masked(client: AtlasClient, frames: list[tuple[np.ndarray, np.ndarray, dict]], prompt: str,
           enhance: bool = True, seed: int = 42) -> tuple[list[PosedView], dict]:
    """atlasMasked: [(rgb HxWx3, keep-mask HxW (True = keep, False = fill), PinholeCamera)] -> completed views at the
    same cameras (each target is the frame's own camera, as the partial checkpoints were trained)."""
    body = {"contextFrames": [{"imageAsset": {"base64": b64_image(rgb)}, "maskAsset": {"base64": b64_mask(m)},
                               "camera": cam} for rgb, m, cam in frames],
            "targetCameras": [cam for _, _, cam in frames], "prompt": prompt, "enhancePrompt": enhance,
            "modelParameters": {"seed": seed}}
    resp = client.result(client.submit("atlasMasked", body))
    return [_decode(client, f, want_depth=False) for f in resp.get("frames") or []], \
        {k: v for k, v in resp.items() if k != "frames"}


def splats_to_mesh(client: AtlasClient, spz: Path, target_faces: int | None = 200_000,
                   texture_mode: str = "image_texture", texture_size: int = 4096) -> tuple[bytes, dict]:
    """splats2Mesh on a .spz scene -> GLB bytes (Quest-friendly with target_faces)."""
    data = Path(spz).read_bytes()
    if len(data) >= 10 * 1024 * 1024:
        raise ValueError("inline assets must be < 10 MiB: upload the .spz as an asset first")
    body = {"splatAsset": {"base64": "data:application/octet-stream;base64," + base64.b64encode(data).decode()},
            "textureMode": texture_mode, "textureSize": texture_size}
    if target_faces:
        body["targetFaces"] = int(target_faces)
    resp = client.result(client.submit("splats2Mesh", body))
    return client.fetch(resp["mesh"]), {k: v for k, v in resp.items() if k != "mesh"}


# ------------------------------------------------------------------------------------------------- frames


def read_colmap_points(path: Path) -> np.ndarray:
    """COLMAP points3D.bin -> (N, 3) xyz (SfM frame)."""
    import struct
    pts = []
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            _, x, y, z = struct.unpack("<Qddd", f.read(32))
            f.read(3 + 8)  # rgb, error
            (tl,) = struct.unpack("<Q", f.read(8))
            f.read(8 * tl)
            pts.append((x, y, z))
    return np.asarray(pts, float)


def world_from_atlas(view: PosedView, K: np.ndarray, R: np.ndarray, t: np.ndarray, width: int, height: int,
                     room_points_world: np.ndarray, keep: np.ndarray | None = None, min_points: int = 30) -> dict:
    """Sim3 world <- Atlas scene for ONE posed phone frame. R, t, K: our camera of that frame (world->camera, OpenCV,
    at the phone's native size width x height). Rotation: from the two orientations. Scale: room points (world) in
    front of our camera, projected onto Atlas's depth grid (keep = HxW bool on that grid, False on people): median of
    our metric depth / Atlas depth. -> {"s", "R", "t", "n_points", "ratio_iqr", "focal_ratio"}."""
    Ka, Ra, ta, Wa, Ha = from_atlas_camera(view.camera)
    Xc = room_points_world @ R.T + t
    front = Xc[:, 2] > 0.3
    Xc = Xc[front]
    sx, sy = Wa / width, Ha / height  # our camera on Atlas's pixel grid (same aspect: no crop)
    u = (K[0, 0] * Xc[:, 0] / Xc[:, 2] + K[0, 2]) * sx
    v = (K[1, 1] * Xc[:, 1] / Xc[:, 2] + K[1, 2]) * sy
    ok = (u >= 0) & (u < Wa - 1) & (v >= 0) & (v < Ha - 1)
    ui, vi = u[ok].astype(int), v[ok].astype(int)
    za = view.z[vi, ui]
    sel = np.isfinite(za) & (za > 0)
    if keep is not None:
        sel &= keep[vi, ui]
    if view.conf is not None:
        sel &= view.conf[vi, ui] > 0.5
    ratio = Xc[ok][sel, 2] / za[sel]
    if len(ratio) < min_points:
        raise ValueError(f"only {len(ratio)} room points usable for the scale (< {min_points})")
    s = float(np.median(ratio))
    q1, q3 = np.percentile(ratio, [25, 75])
    Rc2w_ours, Rc2w_atlas = R.T, Ra.T
    Rws = Rc2w_ours @ Rc2w_atlas.T
    C_ours, C_atlas = -R.T @ t, -Ra.T @ ta
    tws = C_ours - s * Rws @ C_atlas
    return {"s": s, "R": Rws.tolist(), "t": tws.tolist(), "n_points": int(len(ratio)),
            "ratio_iqr": [float(q1), float(q3)], "focal_ratio": float(Ka[0, 0] / (K[0, 0] * sx))}


def camera_in_atlas(K: np.ndarray, R: np.ndarray, t: np.ndarray, width: int, height: int, sim3: dict) -> tuple[dict, tuple]:
    """Our world camera (a held-out phone) -> an Atlas target camera on the 1280x720 grid in Atlas's scene frame.
    -> (PinholeCamera, crop box of the phone's image on the grid)."""
    from .. import mamma_dyn as md
    Rs, ts = md.sfm_cameras(np.asarray(R)[None], np.asarray(t)[None], sim3)
    Kg, box = generation_camera(K, R, t, width, height)
    return to_atlas_camera(Kg, Rs[0], ts[0], *GEN_SIZE), box


def save_view(v: PosedView, d: Path, name: str) -> dict:
    """Write rgb (png), depth (npy) and camera (json) of a view."""
    from PIL import Image
    d.mkdir(parents=True, exist_ok=True)
    out = {"camera": v.camera, "image": _ref(v.image), "depth_asset": v.depth}
    if v.rgb is not None:
        Image.fromarray(v.rgb).save(d / f"{name}.png")
        out["png"] = str(d / f"{name}.png")
    if v.z is not None:
        np.save(d / f"{name}_depth.npy", v.z)
        out["depth"] = str(d / f"{name}_depth.npy")
    if v.conf is not None:
        np.save(d / f"{name}_conf.npy", v.conf)
    (d / f"{name}.json").write_text(json.dumps(out, indent=1))
    return out
