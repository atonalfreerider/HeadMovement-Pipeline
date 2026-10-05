"""Session folder + session.toml."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

TEMPLATE = """\
# atlas_bridge session. Paths are relative to this file.
name = "{name}"

[input]
# handheld phone videos (1..N). The first one is the timing reference unless `reference` is set.
videos = [{videos}]
# optional clean music track - if set, it becomes the timeline and the Unity audio
music = ""
reference = ""

[sync]
# "source" keeps the reference video's rate (no duplicated/dropped frames); or a number
fps = "source"
# manual offsets in seconds (reference timeline) for videos without usable audio
# offsets = {{ cam2 = 1.234 }}

[rig]
width = 1920
height = 1080
hfov_deg = 65.0
look_at_height = 1.0
dance_radius = 1.2          # metres of travel around the stage origin every camera must see
# low ring for limbs, high ring to see between the dancers in close embrace.
# Defaults keep a 1.2 m-radius dance area fully in frame for every camera (see `rig` warnings).
rings = [
  {{ count = 8, radius = 4.0, height = 1.3 }},
  {{ count = 4, radius = 3.5, height = 2.8, phase_deg = 45.0 }},
]

[atlas]
provider = "manual"         # "manual" (file drop) | "http" (once the API is published)
scale = 1.0                 # Atlas units -> metres; check with `check-scale` after MAMMA
# scene_to_stage = "auto"   # or 16 row-major floats
# dancers_scene = [x, y, z] # override the auto dancer position (Atlas scene frame)

[mamma]
root = "{mamma_root}"
conda_env = "mamma"
preset = "configs/examples/presets/full.yaml"
out_tag = "atlas"

[roles]
lead = "auto"               # MAMMA person id, or "auto" (taller = lead)
follow = "auto"
# expected standing heights (m), used by check-scale
# lead_height = 1.80
# follow_height = 1.65

[beats]
beat_this_root = "{beat_this_root}"
python = "{beat_this_python}"
# fallback when beat_this is not installed: steady grid
# bpm = 88.0

[unity]
streaming_assets = "{streaming_assets}"
capture_name = "{capture_name}"
"""


@dataclass
class Session:
    root: Path
    cfg: dict

    @classmethod
    def load(cls, path: Path) -> "Session":
        path = path.resolve()
        if path.is_dir():
            path = path / "session.toml"
        return cls(path.parent, tomllib.loads(path.read_text()))

    def path(self, p: str | Path) -> Path:
        p = Path(p)
        return p if p.is_absolute() else (self.root / p).resolve()

    @property
    def name(self) -> str:
        return self.cfg["name"]

    @property
    def work(self) -> Path:
        return self.root / "work"

    def section(self, key: str) -> dict:
        return self.cfg.get(key, {})

    def videos(self) -> dict[str, Path]:
        vids = [self.path(v) for v in self.section("input").get("videos", [])]
        return {v.stem: v for v in vids}

    def fps(self, source_fps: float | None = None) -> float:
        """Target frame rate. "source" = the reference video's own rate."""
        v = self.section("sync").get("fps", "source")
        if v == "source":
            if source_fps is None:
                raise ValueError("fps = 'source' needs the probed reference frame rate")
            return source_fps
        return float(v)
