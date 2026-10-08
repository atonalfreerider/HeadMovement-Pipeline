"""Shared fixtures of the dancecap.pipeline tests: a synthetic package root in a temp dir (takes/, work/), a
synthetic take, tiny real media made with ffmpeg, a fake Runpod poller. No pods, no Runpod calls, no real takes
touched (paths.set_root points the backend at the temp tree; shared modules are only used on the real root)."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dancecap.pipeline import paths  # noqa: E402

NO_WINDOW = 0x08000000 if os.name == "nt" else 0


def make_root(tmp_path: Path) -> Path:
    root = tmp_path / "dancecap"
    for d in ("takes", "work", "taxonomy"):
        (root / d).mkdir(parents=True, exist_ok=True)
    paths.set_root(root)
    return root


def make_take(root: Path, name: str = "synth", cams=("01", "02", "03"), window=(10.0, 20.0), extra: str = "",
              video_dir: Path | None = None) -> Path:
    """takes/<name>.toml + work/<name>/clock/alignment.json (sources = cams) - an 'existing take' shape."""
    vdir = video_dir or (root.parent / "dances" / name)
    vdir.mkdir(parents=True, exist_ok=True)
    al = root / "work" / name / "clock" / "alignment.json"
    al.parent.mkdir(parents=True, exist_ok=True)
    al.write_text(json.dumps({"reference": f"{cams[0]}-a.mp4", "sources": [
        {"id": c, "name": f"{c}-a.mp4", "slope": 1.0, "offset": float(i)} for i, c in enumerate(cams)]}),
        encoding="utf-8")
    rel_v = os.path.relpath(vdir, root / "takes").replace("\\", "/")
    win = f"window = [{window[0]}, {window[1]}]\n" if window else ""
    text = (f'# synthetic take\nname = "{name}"\nsource_dir = "{rel_v}"\n'
            f'alignment = "../work/{name}/clock/alignment.json"   # clocks\nexclude = []\n{win}'
            '\n[cameras]\n# reserved keys stay untouched\nscale_rule = "chain"\nloc = { kf_by_time = true }\n'
            '\n[dance]\nlevel = "professional"   # level\ncategory = "demo"\nsong = "Song"\n'
            '\n[dancers.lead]\nname = "Lee"\npid = 1\nstature_m = 1.75   # tall\nshoe_sole_m = 0.025\n'
            '\n[dancers.follow]\nname = "Fay"\npid = 2\nstature_m = 1.62\nshoe_sole_m = 0.025\n' + extra)
    p = root / "takes" / f"{name}.toml"
    p.write_text(text, encoding="utf-8")
    return p


def touch(path: Path, t: float | None = None, text: str = "{}") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    if t is not None:
        os.utime(path, (t, t))
    return path


def make_video(path: Path, seconds: float = 6.0, audio: bool = True, size: str = "160x120", rate: int = 30) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc=size={size}:rate={rate}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100"]
    cmd += ["-t", f"{seconds}", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast"]
    cmd += (["-c:a", "aac", "-shortest"] if audio else [])
    cmd.append(str(path))
    subprocess.run(cmd, check=True, capture_output=True, creationflags=NO_WINDOW)
    return path


def make_audio(path: Path, seconds: float = 8.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=330:sample_rate=44100",
                    "-t", f"{seconds}", str(path)], check=True, capture_output=True, creationflags=NO_WINDOW)
    return path


def have_ffmpeg() -> bool:
    return bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


class FakePoller:
    """Stands in for runpod_view.PodPoller: never calls runpodctl."""

    def __init__(self, pods=None):
        self.pods = pods or []
        self.deleted = []

    def refresh(self, force: bool = False):
        pass

    def summary(self, force: bool = False):
        from dancecap.pipeline import estimate
        return {"connection": "ok", "checked": time.time(), "message": None, "pods": self.pods,
                "pods_running": len(self.pods), "sessions_active": [], "progress_level": "L0",
                "budget": estimate.budget()}


def fake_probe(seconds=60.0, audio=True, fps=30.0, codec="h264"):
    def fn(path):
        return {"kind": "video", "duration_s": seconds, "audio": audio, "fps": fps, "vfr": False, "codec": codec,
                "playable": codec == "h264", "width": 1080, "height": 1920, "rotation": 0, "display_w": 1080,
                "display_h": 1920, "size_bytes": 1000, "problems": []}
    return fn


def wait_for(pred, timeout: float = 30.0, step: float = 0.2):
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = pred()
        if v:
            return v
        time.sleep(step)
    return None
