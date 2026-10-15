"""Provider contract of the mono4d route and the providers.

A provider turns ONE phone video (a window of a take) into something the performance can be seen from at our cameras:

    kind "views"          camera-controlled renders, one video or frame folder per requested camera (Atlas-style)
    kind "mint"           a Gracia 4DGS file -> render_gracia.py renders our cameras (graciasdk, Vulkan)
    kind "ply_sequence"   per-frame 3DGS PLYs (splats/frame_00000.ply ...) -> a gsplat render job (not built yet)
    kind "static_splat"   a static world (the World API today): the room only. Useful as a background or to align the
                          provider's frame; never for the dancers (the docs say to avoid moving people)

Layout, work/<take>/mono4d/<run>/:
    request/request.json             what was asked (Mono4DRequest)
    request/inputs/<cam>.mp4         the input window cut from the canonical frames (upright, one video frame per
                                     canonical frame; <cam>.times.json holds each frame's reference time)
    request/source_trajectory.json   our track of the input phone (world frame), for providers that take poses
    request/target_trajectory.json   the cameras we want, world frame (rig.trajectory)
    request/REQUEST.md               instructions for a manual provider / a vendor contact
    drop/                            what comes back:
        views/<name>.mp4 or views/<name>/000000.jpg ...    kind views (one frame per grid time)
        asset.mint | splats/frame_00000.ply ...             kind mint / ply_sequence
        scene_cameras.json   optional {"input": [T x 4x4 c2w, OpenCV axes, scene frame]}: the provider's poses of the
                             input frames, from which world_from_scene is solved against our own track
        world_from_scene.json optional Sim3 (wins over scene_cameras.json)
    provider_job.json                job record of an API provider (ids only; never a key)
    result.json                      Mono4DResult after a successful fetch

Facts behind CAPABILITIES were read from the vendors' own pages on 2026-10-15 (docs/MONO4D_NOVEL_VIEWS.md has the
sources); `python -m dancecap.mono4d vendors` re-checks them.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import rig

CAPABILITIES = {  # as stated by each vendor, 2026-10-15
    "worldlabs_world_api": {
        "access": "public self-serve API (platform.worldlabs.ai), key header WLT-Api-Key",
        "input": "text, image(s), panorama, video <= 30 s / 100 MB 'of a static space'",
        "monocular_dynamic": False, "dynamic_output": False, "camera_control": False,
        "output": "static world: SPZ splats (100k/500k/full), PLY export, GLB mesh; arbitrary units + metric scale "
                  "metadata", "price": "1,500 credits = $1.20 per marble-1.1 world"},
    "worldlabs_atlas": {
        "access": "early access with select partners (request form); no public API, model id or pricing",
        "input": "text, images, camera poses, depth maps; video as image sequences",
        "monocular_dynamic": None, "dynamic_output": None, "camera_control": True,
        "output": "camera-controlled video (<= 1 min, 1440p), point clouds, static 3DGS; reframing stated 'with "
                  "footage from as few as three cameras'", "price": None},
    "gracia": {
        "access": "processing service via capture partners, 3-5 days; free 1-2 min test for new partners",
        "input": "genlocked, hardware-synchronised multi-camera rigs (about 60 cameras for 360 degrees)",
        "monocular_dynamic": False, "dynamic_output": True, "camera_control": True,
        "output": ".mint (4DGS) or a .PLY sequence; graciasdk renders them to colour + depth at any camera",
        "price": "$800/min streaming/VR tier, $2,500/min VFX tier; capture $15-20k/day via partners"},
}

KINDS = ("views", "mint", "ply_sequence", "static_splat")


@dataclass
class Mono4DRequest:
    take: str
    input_cam: str
    provider: str
    t0: float
    t1: float
    grid: list[float]                 # reference seconds of the frames wanted back (the 30 Hz master grid)
    video: str                        # request/inputs/<cam>.mp4
    video_fps: float
    video_frames: int
    targets: list[str]                # target camera names (target_trajectory.json holds the poses)
    created: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))
    allow_upload: bool = False        # the owner's go-ahead to send this footage to an external service
    notes: list[str] = field(default_factory=list)

    @property
    def T(self) -> int:
        return len(self.grid)


@dataclass
class Mono4DResult:
    provider: str
    kind: str
    dynamic: bool
    views: dict[str, str] = field(default_factory=dict)   # camera -> video file or frame folder
    asset: str | None = None                              # .mint / splats dir / static .ply
    world_from_scene: dict | None = None
    scene_meta: dict = field(default_factory=dict)
    checks: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def run_dir(work: Path, run: str) -> Path:
    return work / "mono4d" / run


def save_request(rd: Path, req: Mono4DRequest) -> Path:
    p = rd / "request" / "request.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(asdict(req), indent=1))
    return p


def load_request(rd: Path) -> Mono4DRequest:
    return Mono4DRequest(**json.loads((rd / "request" / "request.json").read_text()))


def save_result(rd: Path, res: Mono4DResult) -> Path:
    p = rd / "result.json"
    p.write_text(json.dumps(asdict(res), indent=1, default=str))
    return p


def load_result(rd: Path) -> Mono4DResult:
    return Mono4DResult(**json.loads((rd / "result.json").read_text()))


# --------------------------------------------------------------------------------------------- media helpers


def cut_input_video(frames_dir: Path, t0: float, t1: float, out: Path) -> dict:
    """Canonical frames of one camera in [t0, t1] (reference seconds) -> an H.264 mp4, one video frame per canonical
    frame, at the window's mean rate; the exact time of every frame goes to <out>.times.json (frame i = times[i]).
    (ffmpeg's concat demuxer with per-frame durations drops frames shorter than its 25 fps image default.)
    -> {frames, fps, t_first, t_last, width, height}."""
    meta = json.loads((frames_dir / "frames.json").read_text())
    sel = [f for f in meta["frames"] if t0 <= f["t"] <= t1]
    if len(sel) < 2:
        raise ValueError(f"{frames_dir}: fewer than 2 frames in [{t0}, {t1}]")
    out.parent.mkdir(parents=True, exist_ok=True)
    seq = out.parent / f"_{out.stem}_seq"
    shutil.rmtree(seq, ignore_errors=True)
    seq.mkdir()
    for i, f in enumerate(sel):
        src, dst = frames_dir / f["file"], seq / f"{i:06d}.jpg"
        try:
            os.link(src, dst)  # same volume: no copy
        except OSError:
            shutil.copyfile(src, dst)
    fps = (len(sel) - 1) / (sel[-1]["t"] - sel[0]["t"])
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", f"{fps:.6f}", "-i", str(seq / "%06d.jpg"),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-movflags", "+faststart", str(out)]
    try:
        subprocess.run(cmd, check=True)
    finally:
        shutil.rmtree(seq, ignore_errors=True)
    times = [f["t"] for f in sel]
    out.with_suffix(".times.json").write_text(json.dumps({"t": times, "source_files": [f["file"] for f in sel]}))
    return {"frames": len(sel), "fps": fps, "t_first": times[0], "t_last": times[-1],
            "width": int(meta["width"]), "height": int(meta["height"])}


def video_info(path: Path) -> dict:
    """{frames, fps, width, height} of a video (OpenCV), or of a folder of numbered JPEG/PNG frames."""
    path = Path(path)
    if path.is_dir():
        files = sorted([*path.glob("*.jpg"), *path.glob("*.png")])
        if not files:
            return {"frames": 0, "fps": None, "width": None, "height": None}
        from PIL import Image
        with Image.open(files[0]) as im:
            w, h = im.size
        return {"frames": len(files), "fps": None, "width": w, "height": h}
    import cv2
    cap = cv2.VideoCapture(str(path))
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return {"frames": n, "fps": float(cap.get(cv2.CAP_PROP_FPS)) or None,
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    finally:
        cap.release()


# --------------------------------------------------------------------------------------------- providers


class Provider(ABC):
    name = "base"

    def __init__(self, rd: Path, cfg: dict | None = None):
        self.rd = Path(rd)
        self.cfg = cfg or {}

    @abstractmethod
    def submit(self, req: Mono4DRequest) -> dict:
        """Start (or describe) the job -> job record (written to provider_job.json by the caller)."""

    @abstractmethod
    def fetch(self, req: Mono4DRequest, job: dict) -> Mono4DResult | None:
        """The result if it is there (validated), None while pending. Raises on an invalid delivery."""


class ManualProvider(Provider):
    """A file drop: the request package is handed to a person (web UI, vendor contact) and the delivery is put in drop/.
    Works for any vendor and is the only route to Atlas while it is early-access only."""
    name = "manual"
    vendor_note = ("Any provider. Deliver camera-controlled renders (one per target camera, one frame per grid time), "
                   "or a 4D asset (.mint or a PLY sequence) with the provider's poses of the input frames.")

    def submit(self, req: Mono4DRequest) -> dict:
        drop = self.rd / "drop"
        (drop / "views").mkdir(parents=True, exist_ok=True)
        txt = [f"# mono4d request: {req.take} camera {req.input_cam}, {req.t0:.3f}-{req.t1:.3f} s ({self.name})", "",
               self.vendor_note, "",
               f"- Input: `inputs/{Path(req.video).name}` ({req.video_frames} frames, {req.video_fps:.3f} fps, "
               "upright, the phone's own timing).",
               "- Our track of that phone (world frame, metres, per input frame): `source_trajectory.json`.",
               f"- Wanted cameras: `target_trajectory.json` ({len(req.targets)} cameras, {req.T} times on the 30 Hz "
               "grid; c2w in OpenCV and OpenGL axes plus intrinsics per frame).",
               "", "Put the delivery in `../drop/`:",
               "- renders: `drop/views/<camera>.mp4` (or a folder of numbered frames) at exactly the requested "
               "resolution, frame i = time_s[i];",
               "- or a 4D asset: `drop/asset.mint` or `drop/splats/frame_00000.ply ...`, plus `drop/scene_cameras.json` "
               "(the provider's c2w of every input frame) or `drop/world_from_scene.json`.",
               "", "Then: `python -m dancecap.mono4d fetch <take> --run <run>`.", "",
               "Do not treat generated views as measurements: they go through holdout.py and gate.py before MAMMA."]
        p = self.rd / "request" / "REQUEST.md"
        p.write_text("\n".join(txt) + "\n")
        return {"provider": self.name, "status": "waiting for drop/", "instructions": str(p)}

    def fetch(self, req: Mono4DRequest, job: dict) -> Mono4DResult | None:
        drop = self.rd / "drop"
        targets = {s.name: s for s in rig.specs_from_trajectory(
            json.loads((self.rd / "request" / "target_trajectory.json").read_text()))}
        views = {}
        for name in targets:
            for cand in (drop / "views" / f"{name}.mp4", drop / "views" / name):
                if cand.exists() and (cand.is_file() or any(cand.iterdir())):
                    views[name] = cand
                    break
        res = None
        if views:
            missing = sorted(set(targets) - set(views))
            checks, bad = {}, []
            for name, p in views.items():
                info = video_info(p)
                s = targets[name]
                checks[name] = info
                if info["frames"] != req.T:
                    bad.append(f"{name}: {info['frames']} frames, want {req.T} (one per grid time)")
                if (info["width"], info["height"]) != (s.width, s.height):
                    bad.append(f"{name}: {info['width']}x{info['height']}, requested {s.width}x{s.height} "
                               "(the intrinsics assume the requested size)")
            if bad:
                raise ValueError("invalid delivery: " + "; ".join(bad))
            res = Mono4DResult(provider=self.name, kind="views", dynamic=True,
                               views={k: str(v) for k, v in views.items()}, checks=checks,
                               notes=[f"missing views: {missing}"] if missing else [])
        elif (drop / "asset.mint").exists():
            res = Mono4DResult(provider=self.name, kind="mint", dynamic=True, asset=str(drop / "asset.mint"))
        elif (drop / "splats").is_dir() and any((drop / "splats").glob("frame_*.ply")):
            n = len(list((drop / "splats").glob("frame_*.ply")))
            if n != req.T:
                raise ValueError(f"invalid delivery: {n} PLY frames, want {req.T}")
            res = Mono4DResult(provider=self.name, kind="ply_sequence", dynamic=True, asset=str(drop / "splats"))
        if res is None:
            return None
        if res.kind != "views":  # a 4D asset needs its frame: given, or solved from the provider's input poses
            res.world_from_scene = scene_alignment(self.rd, req)
        return res


class GraciaProvider(ManualProvider):
    """Gracia processes multi-camera captures (their FAQ: shutter sync and genlock required, ~60 cameras for 360) and
    delivers .mint or a PLY sequence. A single phone is outside that workflow, so this provider exists for a performance
    captured at a partner studio (or a free 1-2 minute test): drop the .mint and render our cameras with render_gracia."""
    name = "gracia"
    vendor_note = ("Gracia (support@gracia.ai): processing only; capture through a partner studio with genlocked "
                   "cameras. A one-phone video is NOT a supported input (FAQ, 2026-10-15). Ask for `.mint` plus the "
                   "rig calibration (to align their scene frame with ours).")

    def submit(self, req: Mono4DRequest) -> dict:
        job = super().submit(req)
        job["supported"] = False
        job["why"] = CAPABILITIES["gracia"]["input"]
        return job


class WorldLabsProvider(Provider):
    """World Labs World API (public): media upload -> worlds:generate (video prompt) -> operation -> export PLY.

    Today the API generates STATIC worlds (Marble models; the video guide says to keep the space static and avoid
    moving people), so the result is kind "static_splat": a room model in metric-ish units, not the dancers. Atlas
    "will power future versions of Marble"; when World Labs exposes it, set [mono4d.worldlabs] model to its id and the
    dynamic/camera-controlled outputs get parsed here. Every generate costs credits: submit() refuses unless the
    caller passes spend_ok, and uploads refuse unless the request carries allow_upload (the footage shows people).
    """
    name = "worldlabs"
    BASE_URL = "https://api.worldlabs.ai"
    MAX_VIDEO_S = 30.0
    MAX_VIDEO_BYTES = 100 * 1024 * 1024

    def __init__(self, rd: Path, cfg: dict | None = None, http=None, raw=None):
        super().__init__(rd, cfg)
        self._http, self._raw = http, raw  # API client (carries the key) / plain client for signed storage URLs

    def _key(self) -> str:
        key = os.environ.get("WORLDLABS_API_KEY")
        if not key:
            p = Path(os.environ.get("WORLDLABS_API_KEY_FILE", Path.home() / "Desktop" / "worldlabs-key.txt"))
            if not p.exists():
                raise RuntimeError("no World Labs key: set WORLDLABS_API_KEY or WORLDLABS_API_KEY_FILE "
                                   "(platform.worldlabs.ai -> API keys)")
            key = p.read_text().strip()
        return key

    def http(self):
        if self._http is None:
            import httpx
            self._http = httpx.Client(base_url=self.cfg.get("base_url", self.BASE_URL), timeout=120.0,
                                      headers={"WLT-Api-Key": self._key()})
        return self._http

    def raw(self):
        """Client for the signed upload/download URLs: never the API key header (those URLs are third-party storage)."""
        if self._raw is None:
            import httpx
            self._raw = httpx.Client(timeout=600.0)
        return self._raw

    def _json(self, method: str, path: str, **kw) -> dict:
        r = self.http().request(method, path, **kw)
        if r.status_code == 402:
            raise RuntimeError("World API: 402 payment required (buy credits at platform.worldlabs.ai/billing)")
        r.raise_for_status()
        return r.json()

    def submit(self, req: Mono4DRequest, spend_ok: bool = False) -> dict:
        if not req.allow_upload:
            raise PermissionError("the request does not carry the owner's go-ahead to upload this footage "
                                  "(request --allow-upload)")
        if not spend_ok:
            raise PermissionError("World API generations cost credits (marble-1.1: 1,500 = $1.20); pass --yes-spend")
        video = Path(req.video)
        dur = req.video_frames / req.video_fps
        if dur > self.MAX_VIDEO_S or video.stat().st_size > self.MAX_VIDEO_BYTES:
            raise ValueError(f"World API video limits: <= {self.MAX_VIDEO_S:.0f} s and <= 100 MB "
                             f"(this one: {dur:.1f} s, {video.stat().st_size / 1e6:.1f} MB)")
        prep = self._json("POST", "/marble/v1/media-assets:prepare_upload",
                          json={"file_name": video.name[:64], "kind": "video", "extension": video.suffix.lstrip(".")})
        up = prep["upload_info"]
        r = self.raw().request(up.get("upload_method", "PUT"), up["upload_url"], content=video.read_bytes(),
                               headers=up.get("required_headers") or {})
        r.raise_for_status()
        asset_id = prep["media_asset"]["media_asset_id"]
        body = {"display_name": f"{req.take} {req.input_cam} {req.t0:.0f}-{req.t1:.0f}"[:64],
                "model": self.cfg.get("model", "marble-1.1"), "permission": {"public": False},
                "world_prompt": {"type": "video", "video_prompt": {"source": "media_asset", "media_asset_id": asset_id}}}
        op = self._json("POST", "/marble/v1/worlds:generate", json=body)
        return {"provider": self.name, "status": "generating", "media_asset_id": asset_id,
                "operation_id": op["operation_id"], "model": body["model"]}

    def fetch(self, req: Mono4DRequest, job: dict) -> Mono4DResult | None:
        op = self._json("GET", f"/marble/v1/operations/{job['operation_id']}")
        if not op.get("done"):
            return None
        if op.get("error"):
            raise RuntimeError(f"World API generation failed: {op['error']}")
        world = op["response"] or {}
        splats = (world.get("assets") or {}).get("splats") or {}
        meta = splats.get("semantics_metadata") or {}
        ply = self.rd / "drop" / "static_world.ply"
        if not ply.exists():
            ex = self._json("POST", f"/marble/v1/worlds/{world['world_id']}:export",
                            json={"asset_type": "splats", "format": "ply", "resolution": "full_res"})
            while not ex.get("done"):
                time.sleep(self.cfg.get("poll_s", 5.0))
                ex = self._json("GET", f"/marble/v1/operations/{ex['operation_id']}")
            if ex.get("error"):
                raise RuntimeError(f"World API export failed: {ex['error']}")
            ply.parent.mkdir(parents=True, exist_ok=True)
            with self.raw().stream("GET", ex["response"]["url"]) as r, ply.open("wb") as fh:
                r.raise_for_status()
                for chunk in r.iter_bytes():
                    fh.write(chunk)
        return Mono4DResult(provider=self.name, kind="static_splat", dynamic=False, asset=str(ply),
                            scene_meta={"world_id": world.get("world_id"), "model": world.get("model"),
                                        "metric_scale_factor": meta.get("metric_scale_factor"),
                                        "ground_plane_offset": meta.get("ground_plane_offset"),
                                        "cost": op.get("cost")},
                            notes=["static world: the room only, never the dancers (World API, 2026-10-15)"])


PROVIDERS = {"manual": ManualProvider, "gracia": GraciaProvider, "worldlabs": WorldLabsProvider}


def make_provider(name: str, rd: Path, cfg: dict | None = None, **kw) -> Provider:
    if name not in PROVIDERS:
        raise ValueError(f"unknown provider {name!r}; choose from {sorted(PROVIDERS)}")
    return PROVIDERS[name](rd, cfg, **kw)


def scene_alignment(rd: Path, req: Mono4DRequest) -> dict | None:
    """world_from_scene of a delivery: drop/world_from_scene.json, else solved from drop/scene_cameras.json (the
    provider's poses of the input frames) against our source track resampled to the input frame times."""
    drop = rd / "drop"
    if (drop / "world_from_scene.json").exists():
        return json.loads((drop / "world_from_scene.json").read_text())
    if not (drop / "scene_cameras.json").exists():
        return None
    P = np.asarray(json.loads((drop / "scene_cameras.json").read_text())["input"], float)
    src = rig.specs_from_trajectory(json.loads((rd / "request" / "source_trajectory.json").read_text()))[0]
    return rig.world_from_scene(P, src)


def copy_drop(src: Path, rd: Path) -> None:
    """Copy a vendor delivery folder into drop/ (keeps the original untouched)."""
    shutil.copytree(src, rd / "drop", dirs_exist_ok=True)
