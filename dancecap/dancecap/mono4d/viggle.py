"""Viggle JST-2 Motion 3D (video -> skeletal animation) as a one-camera pose candidate, scored against a multi-phone fit.

    python -m dancecap.mono4d viggle-test larissa_kadu_38 --cam 06 --t0 58 --t1 62 --allow-upload [--fps 30]

API (docs.viggle.ai, V1): POST https://apis.viggle.ai/v1/motions (multipart: motion_video, type=glb, target_fps,
enable_smoothing) -> poll GET /v1/motions/{id} until ready -> GET /v1/motions/{id}/export?download_type=glb -> signed
URLs of three rigs (mixamo_url: a 50-joint Mixamo rig). Price: 5 credits ($0.05) per source second, rounded up.
Key: VIGGLE_API_KEY or ~/Desktop/viggle-key.txt (API key, never the free PINOC / consumer app, whose terms allow
training on uploads). It extracts ONE person, so each dancer goes in as their own clip: a fixed crop around that
dancer over the window, the partner's pixels (SAM mask, dilated) inpainted into the background.

Scoring (pure numpy, glb_joints reads the GLB itself: nodes, skin-free animation channels, linear/step samplers):
the 14 joints both skeletons define (MIXAMO_TO_SMPLX) against the reference fit's SMPL-X joints at the same reference
times: per-frame PA-MPJPE (pose after a similarity alignment), and limb-direction errors (upper arm, forearm, thigh,
shin, both sides) after the per-frame rotation. Viggle's output is the provider's estimate, never a measurement.
"""
from __future__ import annotations

import json
import os
import struct
import subprocess
import time
from pathlib import Path

import numpy as np

BASE_URL = "https://apis.viggle.ai/v1"
MIXAMO_TO_SMPLX = {"Hips": 0, "LeftUpLeg": 1, "RightUpLeg": 2, "LeftLeg": 4, "RightLeg": 5, "LeftFoot": 7,
                   "RightFoot": 8, "Neck": 12, "LeftArm": 16, "RightArm": 17, "LeftForeArm": 18, "RightForeArm": 19,
                   "LeftHand": 20, "RightHand": 21}
LIMBS = {"l_upper_arm": (16, 18), "l_forearm": (18, 20), "r_upper_arm": (17, 19), "r_forearm": (19, 21),
         "l_thigh": (1, 4), "l_shin": (4, 7), "r_thigh": (2, 5), "r_shin": (5, 8)}


# ------------------------------------------------------------------------------------------------- client


class ViggleClient:
    def __init__(self, key: str | None = None, base_url: str = BASE_URL, http=None, raw=None):
        self.base_url, self._key, self._http, self._raw = base_url.rstrip("/"), key, http, raw

    def key(self) -> str:
        if not self._key:
            k = os.environ.get("VIGGLE_API_KEY")
            if not k:
                p = Path(os.environ.get("VIGGLE_API_KEY_FILE", Path.home() / "Desktop" / "viggle-key.txt"))
                if not p.exists():
                    raise RuntimeError("no Viggle key: set VIGGLE_API_KEY or put it in ~/Desktop/viggle-key.txt")
                k = p.read_text().strip()
            self._key = k
        return self._key

    def http(self):
        if self._http is None:
            import httpx
            self._http = httpx.Client(base_url=self.base_url, timeout=300.0,
                                      headers={"Authorization": f"Bearer {self.key()}"})
        return self._http

    def raw(self):
        if self._raw is None:
            import httpx
            self._raw = httpx.Client(timeout=300.0, follow_redirects=True)
        return self._raw

    def _check(self, r) -> dict:
        if r.status_code >= 400:
            raise RuntimeError(f"Viggle {r.request.method} {r.request.url.path}: HTTP {r.status_code}: {r.text[:500]}")
        return r.json()

    def credits(self) -> dict:
        return self._check(self.http().get("/credits"))

    def create_motion(self, video: Path, target_fps: float = 30.0, smoothing: bool = False, name: str | None = None,
                      task_id: str | None = None) -> dict:
        data = {"type": "glb", "target_fps": str(target_fps), "enable_smoothing": str(smoothing).lower()}
        if name:
            data["name"] = name[:100]
        if task_id:
            data["task_id"] = task_id
        with open(video, "rb") as fh:
            return self._check(self.http().post("/motions", data=data,
                                                files={"motion_video": (Path(video).name, fh, "video/mp4")}))

    def motion(self, motion_id: str) -> dict:
        return self._check(self.http().get(f"/motions/{motion_id}"))

    def wait(self, motion_id: str, timeout_s: float = 900.0, poll_s: float = 5.0) -> dict:
        t0 = time.time()
        while True:
            m = self.motion(motion_id)
            if m.get("status") in ("ready", "failed"):
                if m["status"] == "failed":
                    raise RuntimeError(f"Viggle motion {motion_id} failed: {m.get('error') or m.get('glb')}")
                return m
            if time.time() - t0 > timeout_s:
                raise TimeoutError(f"Viggle motion {motion_id} still {m.get('status')} after {timeout_s:.0f} s")
            time.sleep(poll_s)

    def export(self, motion_id: str, fmt: str = "glb") -> dict:
        return self._check(self.http().get(f"/motions/{motion_id}/export", params={"download_type": fmt}))

    def download(self, url: str) -> bytes:
        r = self.raw().get(url)
        r.raise_for_status()
        return r.content


# ------------------------------------------------------------------------------------------------- GLB animation


def _read_glb(data: bytes) -> tuple[dict, bytes]:
    magic, version, length = struct.unpack_from("<III", data, 0)
    if magic != 0x46546C67:
        raise ValueError("not a GLB file")
    off, doc, binc = 12, None, b""
    while off < length:
        clen, ctype = struct.unpack_from("<II", data, off)
        chunk = data[off + 8:off + 8 + clen]
        if ctype == 0x4E4F534A:
            doc = json.loads(chunk.decode("utf-8"))
        elif ctype == 0x004E4942:
            binc = chunk
        off += 8 + clen
    if doc is None:
        raise ValueError("GLB without a JSON chunk")
    return doc, binc


_NC = {"SCALAR": 1, "VEC3": 3, "VEC4": 4, "MAT4": 16}
_DT = {5126: np.float32, 5123: np.uint16, 5125: np.uint32, 5121: np.uint8}


def _accessor(doc: dict, binc: bytes, i: int) -> np.ndarray:
    a = doc["accessors"][i]
    bv = doc["bufferViews"][a["bufferView"]]
    n, dt = _NC[a["type"]], _DT[a["componentType"]]
    start = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
    stride = bv.get("byteStride", 0)
    item = n * np.dtype(dt).itemsize
    if stride and stride != item:
        rows = [np.frombuffer(binc, dt, n, start + k * stride) for k in range(a["count"])]
        arr = np.stack(rows)
    else:
        arr = np.frombuffer(binc, dt, a["count"] * n, start).reshape(a["count"], n)
    return arr.astype(np.float64)


def _quat_mat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _slerp(a: np.ndarray, b: np.ndarray, w: float) -> np.ndarray:
    d = float(np.dot(a, b))
    if d < 0:
        b, d = -b, -d
    if d > 0.9995:
        q = a + w * (b - a)
        return q / np.linalg.norm(q)
    th = np.arccos(d)
    return (np.sin((1 - w) * th) * a + np.sin(w * th) * b) / np.sin(th)


def _sample(times: np.ndarray, vals: np.ndarray, t: float, path: str, interp: str) -> np.ndarray:
    if t <= times[0]:
        return vals[0]
    if t >= times[-1]:
        return vals[-1]
    k = int(np.searchsorted(times, t)) - 1
    if interp == "STEP":
        return vals[k]
    w = (t - times[k]) / (times[k + 1] - times[k])
    return _slerp(vals[k], vals[k + 1], w) if path == "rotation" else (1 - w) * vals[k] + w * vals[k + 1]


def glb_joints(data: bytes, fps: float | None = None, animation: int = 0) -> tuple[list[str], np.ndarray, np.ndarray]:
    """GLB bytes -> (node names, times (F,), world positions (F, N, 3)) of every node, from the first animation.
    Frames are sampled at fps (default: the animation's own keyframe times)."""
    doc, binc = _read_glb(data)
    nodes = doc["nodes"]
    names = [n.get("name", f"node{i}") for i, n in enumerate(nodes)]
    parent = {c: i for i, n in enumerate(nodes) for c in n.get("children", [])}
    rest = []
    for n in nodes:
        rest.append({"translation": np.array(n.get("translation", [0, 0, 0]), float),
                     "rotation": np.array(n.get("rotation", [0, 0, 0, 1]), float),
                     "scale": np.array(n.get("scale", [1, 1, 1]), float),
                     "matrix": np.array(n["matrix"], float).reshape(4, 4).T if "matrix" in n else None})
    anim = doc.get("animations", [])[animation]
    chans = []
    for ch in anim["channels"]:
        s = anim["samplers"][ch["sampler"]]
        tgt = ch["target"]
        if tgt.get("path") not in ("translation", "rotation", "scale") or "node" not in tgt:
            continue
        chans.append((tgt["node"], tgt["path"], _accessor(doc, binc, s["input"])[:, 0], _accessor(doc, binc, s["output"]),
                      s.get("interpolation", "LINEAR")))
    t_all = np.unique(np.concatenate([c[2] for c in chans])) if chans else np.zeros(1)
    times = t_all if fps is None else np.arange(t_all[0], t_all[-1] + 1e-9, 1.0 / fps)
    order = []
    seen = set()

    def visit(i):
        if i in seen:
            return
        if i in parent:
            visit(parent[i])
        seen.add(i)
        order.append(i)
    for i in range(len(nodes)):
        visit(i)
    P = np.zeros((len(times), len(nodes), 3))
    for f, t in enumerate(times):
        local = [dict(r) for r in rest]
        for node, path, ti, vals, interp in chans:
            local[node][path] = _sample(ti, vals, t, path, interp)
            local[node]["matrix"] = None
        world = [None] * len(nodes)
        for i in order:
            L = local[i]
            if L["matrix"] is not None:
                M = L["matrix"]
            else:
                M = np.eye(4)
                M[:3, :3] = _quat_mat(L["rotation"]) * L["scale"][None, :]
                M[:3, 3] = L["translation"]
            world[i] = world[parent[i]] @ M if i in parent else M
            P[f, i] = world[i][:3, 3]
    return names, times, P


def mixamo_index(names: list[str]) -> dict[int, int]:
    """SMPL-X joint -> node index, matching Mixamo names with or without the 'mixamorig:' prefix."""
    short = {n.split(":")[-1].split("|")[-1].lower(): i for i, n in enumerate(names)}
    return {j: short[m.lower()] for m, j in MIXAMO_TO_SMPLX.items() if m.lower() in short}


# ------------------------------------------------------------------------------------------------- scoring


def score(names: list[str], times: np.ndarray, P: np.ndarray, t_first: float, ref_t: np.ndarray,
          ref_joints: np.ndarray, tol: float = 1 / 50) -> dict:
    """Viggle joints (animation time 0 = reference time t_first) vs the reference SMPL-X joints (T, 55, 3)."""
    from .holdout import procrustes, summary
    idx = mixamo_index(names)
    js = sorted(idx)
    if len(js) < 10:
        raise ValueError(f"only {len(js)} Mixamo joints found ({sorted(MIXAMO_TO_SMPLX)})")
    tv = t_first + times
    pa, limb_err, n = [], {k: [] for k in LIMBS}, 0
    for f, t in enumerate(tv):
        k = int(np.argmin(np.abs(ref_t - t)))
        if abs(ref_t[k] - t) > tol:
            continue
        A = P[f, [idx[j] for j in js]]
        B = ref_joints[k, js]
        A_al = procrustes(A, B)
        pa.append(1e3 * np.linalg.norm(A_al - B, axis=-1).mean())
        for name, (a, b) in LIMBS.items():
            if a in idx and b in idx:
                ia, ib = js.index(a), js.index(b)
                u, v = A_al[ib] - A_al[ia], B[ib] - B[ia]
                c = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))
                limb_err[name].append(float(np.degrees(np.arccos(np.clip(c, -1, 1)))))
        n += 1
    return {"frames": n, "joints": len(js), "pa_mpjpe_mm": summary(pa),
            "limb_err_deg": {k: summary(v) for k, v in limb_err.items()},
            "limb_err_deg_all": summary([x for v in limb_err.values() for x in v])}


# ------------------------------------------------------------------------------------------------- dancer clips


def dancer_clip(work: Path, cam: str, t0: float, t1: float, pid: int, out: Path, pad: float = 0.18,
                partner_dilate: int = 9) -> dict:
    """A fixed crop around one dancer over [t0, t1] with the partner's pixels inpainted away -> mp4 (+ .times.json)."""
    import cv2
    from PIL import Image, ImageFilter
    meta = json.loads((work / "frames" / cam / "frames.json").read_text())
    sel = [(i, f) for i, f in enumerate(meta["frames"]) if t0 <= f["t"] <= t1]
    mdir = work / "dancers" / cam / "masks"
    other = 2 if pid == 1 else 1
    boxes = []
    for i, _ in sel:
        m = np.asarray(Image.open(mdir / f"mask_{i:04d}_{pid:02d}.png").convert("L")) > 127
        ys, xs = np.nonzero(m)
        if len(xs):
            boxes.append((xs.min(), ys.min(), xs.max(), ys.max()))
    if not boxes:
        raise ValueError(f"no mask for dancer {pid:02d} in {cam} [{t0}, {t1}]")
    b = np.array(boxes)
    x0, y0, x1, y1 = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
    w, h = x1 - x0, y1 - y0
    x0, x1 = int(max(0, x0 - pad * w)), int(min(meta["width"], x1 + pad * w))
    y0, y1 = int(max(0, y0 - pad * h)), int(min(meta["height"], y1 + pad * h))
    x1 -= (x1 - x0) % 2
    y1 -= (y1 - y0) % 2
    seq = out.parent / f"_{out.stem}_seq"
    seq.mkdir(parents=True, exist_ok=True)
    for k, (i, f) in enumerate(sel):
        img = np.asarray(Image.open(work / "frames" / cam / f["file"]).convert("RGB")).copy()
        mo = Image.open(mdir / f"mask_{i:04d}_{other:02d}.png").convert("L").filter(ImageFilter.MaxFilter(partner_dilate))
        ms = np.asarray(Image.open(mdir / f"mask_{i:04d}_{pid:02d}.png").convert("L")) > 127
        fill = (np.asarray(mo) > 127) & ~ms
        crop = np.ascontiguousarray(img[y0:y1, x0:x1])
        # inpaint, not a flat grey: a grey body-shaped silhouette is still a "person" a tracker may follow
        crop = cv2.inpaint(crop, fill[y0:y1, x0:x1].astype(np.uint8) * 255, 7, cv2.INPAINT_TELEA)
        Image.fromarray(crop).save(seq / f"{k:06d}.jpg", quality=95)
    fps = (len(sel) - 1) / (sel[-1][1]["t"] - sel[0][1]["t"])
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", f"{fps:.6f}", "-i", str(seq / "%06d.jpg"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", str(out)], check=True)
    for p in seq.glob("*.jpg"):
        p.unlink()
    seq.rmdir()
    times = [f["t"] for _, f in sel]
    out.with_suffix(".times.json").write_text(json.dumps({"t": times, "crop": [x0, y0, x1, y1]}))
    return {"frames": len(sel), "fps": fps, "t_first": times[0], "crop": [x0, y0, x1, y1]}


def run(take: str, cam: str, t0: float, t1: float, allow_upload: bool = False, fps: float = 30.0,
        reference: Path | None = None, client: ViggleClient | None = None) -> dict:
    if not allow_upload:
        raise PermissionError("viggle-test uploads the dancers' video to Viggle: pass --allow-upload")
    from ..take import PACKAGE_ROOT
    from .holdout import load_fit
    work = PACKAGE_ROOT / "work" / take
    out = work / "mono4d" / "viggle" / f"{cam}_{int(t0 * 1000)}_{int(t1 * 1000)}"
    out.mkdir(parents=True, exist_ok=True)
    ref = Path(reference) if reference else work / "smplx"
    c = client or ViggleClient()
    rep = {"take": take, "cam": cam, "window": [t0, t1], "reference": str(ref), "dancers": {}}
    for pid, role in ((1, "lead"), (2, "follow")):
        clip = out / f"{role}.mp4"
        if clip.exists() and clip.with_suffix(".times.json").exists():  # the clip is deterministic: reuse it
            tj = json.loads(clip.with_suffix(".times.json").read_text())
            info = {"frames": len(tj["t"]), "fps": (len(tj["t"]) - 1) / (tj["t"][-1] - tj["t"][0]), "t_first": tj["t"][0],
                    "crop": tj["crop"]}
        else:
            info = dancer_clip(work, cam, t0, t1, pid, clip)
        tsk = f"dancecap-{take}-{cam}-{role}-{int(t0 * 1000)}-{int(t1 * 1000)}"
        glb_p = out / f"{role}_mixamo.glb"
        if not glb_p.exists():
            m = c.create_motion(clip, target_fps=fps, name=f"{take} {cam} {role}", task_id=tsk)
            m = c.wait(m["id"])
            ex = c.export(m["id"], "glb")
            glb_p.write_bytes(c.download(ex["mixamo_url"]))
            (out / f"{role}_motion.json").write_text(json.dumps({"motion": m, "export": {k: bool(v) for k, v in ex.items()}},
                                                                indent=1, default=str))
        names, times, P = glb_joints(glb_p.read_bytes(), fps=fps)
        fit = load_fit(ref, role)
        from .candidates import frame_to_ref  # the clip's first frame on the reference fit's (sync-corrected) clock
        t_ref = float(frame_to_ref(take, cam, info["t_first"]))
        rep["dancers"][role] = {"clip": info, "t_first_ref": t_ref, "glb_nodes": len(names),
                                "glb_seconds": float(times[-1] - times[0]),
                                **score(names, times, P, t_ref, fit["t"], fit["joints"])}
    (out / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    return rep
