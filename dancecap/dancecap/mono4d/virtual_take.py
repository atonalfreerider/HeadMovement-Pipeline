"""Generated views -> a virtual take in the CONTRACTS.md layout, so the standard stages (detect, dancers, mamma) run on it.

work/<vtake>/   (vtake = <take>__m4d_<run> unless named)
    frames/<cam>/000000.jpg ... + frames.json   one frame per grid time for a generated view (t = the grid time); the
                                                REAL input phone is copied with its own frame times
    cameras/<cam>.npz                           {t, K, R, tvec, ok} in the world frame: exact for a generated view
                                                (it was rendered there), the localised track for the real phone
    cameras/world_from_sfm.json                 identity Sim3 (everything is already in the world frame)
    mono4d.json                                 provenance: source take, run, provider, input camera, which cameras are
                                                synthetic, independent_views = 1. Downstream code must never count the
                                                generated views as measurements (gate.py, docs/MONO4D_NOVEL_VIEWS.md).

The real phone stays in the set on purpose: it is the only measurement. Its name is kept (e.g. 06); generated views are
v00, v01, ... (rig.ring) or h<cam> for a view rendered at a held-out phone's pose (holdout.py compares it with the real
phone and keeps it OUT of the fit set).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from ..take import PACKAGE_ROOT
from . import rig
from .providers import Mono4DRequest, Mono4DResult

IDENTITY_SIM3 = {"s": 1.0, "R": np.eye(3).tolist(), "t": [0.0, 0.0, 0.0],
                 "note": "mono4d virtual take: cameras are defined in the world frame"}


def _write_frames_json(d: Path, cam: str, source: str, width: int, height: int, frames: list[dict]) -> None:
    t = np.array([f["t"] for f in frames], float)
    dt = np.diff(t) if len(t) > 1 else np.array([1 / 30.0])
    doc = {"camera": cam, "source": source, "width": int(width), "height": int(height), "rotation": 0, "slope": 1.0,
           "offset": 0.0, "frame_count": len(frames), "fps_median": float(1.0 / np.median(dt)),
           "frame_interval_ms": {"min": float(dt.min() * 1e3), "max": float(dt.max() * 1e3)}, "frames": frames}
    (d / "frames.json").write_text(json.dumps(doc, indent=1))


def _export_view(src: Path, dst: Path, grid: np.ndarray, size: tuple[int, int]) -> list[dict]:
    """A rendered video or numbered-frame folder -> dst/000000.jpg ... (JPEG q95), one per grid time."""
    dst.mkdir(parents=True, exist_ok=True)
    frames = []
    if src.is_dir():
        files = sorted([*src.glob("*.jpg"), *src.glob("*.png")])
        if len(files) != len(grid):
            raise ValueError(f"{src}: {len(files)} frames, want {len(grid)}")
        from PIL import Image
        for i, f in enumerate(files):
            name = f"{i:06d}.jpg"
            if f.suffix.lower() == ".jpg":
                shutil.copyfile(f, dst / name)
            else:
                with Image.open(f) as im:
                    im.convert("RGB").save(dst / name, "JPEG", quality=95)
            frames.append({"file": name, "pts": float(grid[i]), "t": float(grid[i])})
    else:
        import cv2
        cap = cv2.VideoCapture(str(src))
        try:
            i = 0
            while True:
                ok, img = cap.read()
                if not ok:
                    break
                if i >= len(grid):
                    raise ValueError(f"{src}: more frames than the {len(grid)} grid times")
                if (img.shape[1], img.shape[0]) != tuple(size):
                    raise ValueError(f"{src}: {img.shape[1]}x{img.shape[0]}, requested {size[0]}x{size[1]}")
                name = f"{i:06d}.jpg"
                cv2.imwrite(str(dst / name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                frames.append({"file": name, "pts": float(grid[i]), "t": float(grid[i])})
                i += 1
        finally:
            cap.release()
        if len(frames) != len(grid):
            raise ValueError(f"{src}: {len(frames)} frames decoded, want {len(grid)}")
    return frames


def _write_cameras(path: Path, spec: rig.ViewSpec, t: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, t=np.asarray(t, float), K=spec.K, R=spec.R, tvec=spec.t, ok=spec.ok)


def package(take: str, run: str, req: Mono4DRequest, res: Mono4DResult, vtake: str | None = None,
            exclude: tuple[str, ...] = (), force: bool = False) -> Path:
    """Write work/<vtake>/ from a fetched run (kind "views"). -> the virtual take's work dir."""
    if res.kind != "views":
        raise ValueError(f"package needs rendered views; this run delivered {res.kind!r} (render it first)")
    work = PACKAGE_ROOT / "work" / take
    rd = work / "mono4d" / run
    vtake = vtake or f"{take}__m4d_{run}"
    out = PACKAGE_ROOT / "work" / vtake
    if out.exists():
        if not force:
            raise FileExistsError(f"{out} exists (pass force to rebuild it)")
        shutil.rmtree(out)
    grid = np.asarray(req.grid, float)
    targets = {s.name: s for s in rig.specs_from_trajectory(
        json.loads((rd / "request" / "target_trajectory.json").read_text()))}
    synthetic = []
    for name, src in sorted(res.views.items()):
        if name in exclude or name not in targets:
            continue
        s = targets[name]
        frames = _export_view(Path(src), out / "frames" / name, grid, (s.width, s.height))
        _write_frames_json(out / "frames" / name, name, f"mono4d:{res.provider}:{run}", s.width, s.height, frames)
        _write_cameras(out / "cameras" / f"{name}.npz", s, grid)
        synthetic.append(name)
    # the real input phone: its own frames and track over the window (the only measurement in the set)
    real = req.input_cam
    src_dir = work / "frames" / real
    meta = json.loads((src_dir / "frames.json").read_text())
    keep = [i for i, f in enumerate(meta["frames"]) if req.t0 <= f["t"] <= req.t1]
    (out / "frames" / real).mkdir(parents=True, exist_ok=True)
    frames = []
    for j, i in enumerate(keep):
        f = meta["frames"][i]
        name = f"{j:06d}.jpg"
        shutil.copyfile(src_dir / f["file"], out / "frames" / real / name)
        frames.append({"file": name, "pts": f["pts"], "t": f["t"]})
    _write_frames_json(out / "frames" / real, real, meta["source"], meta["width"], meta["height"], frames)
    src_spec = rig.specs_from_trajectory(json.loads((rd / "request" / "source_trajectory.json").read_text()))[0]
    if src_spec.T != len(keep):
        raise ValueError(f"source trajectory has {src_spec.T} frames, the window {len(keep)}")
    _write_cameras(out / "cameras" / f"{real}.npz", src_spec, np.array([f["t"] for f in frames]))
    (out / "cameras" / "world_from_sfm.json").write_text(json.dumps(IDENTITY_SIM3, indent=1))
    doc = {"format": "dancecap.mono4d.virtual_take/1", "source_take": take, "run": run, "provider": res.provider,
           "input_cam": real, "real_cams": [real], "synthetic_cams": synthetic, "independent_views": 1,
           "window": [req.t0, req.t1], "grid": [float(grid[0]), float(grid[-1]), len(grid)],
           "world_from_scene": res.world_from_scene, "excluded": list(exclude),
           "warning": "generated views are the provider's prior, not measurements; never count them as cameras"}
    (out / "mono4d.json").write_text(json.dumps(doc, indent=1))
    return out


def is_virtual(work: Path) -> dict | None:
    """The mono4d manifest of a work dir, or None for a real take."""
    p = Path(work) / "mono4d.json"
    return json.loads(p.read_text()) if p.exists() else None
