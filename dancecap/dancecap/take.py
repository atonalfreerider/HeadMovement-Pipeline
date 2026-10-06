"""A take: one performance recorded by several unsynchronised handheld cameras.

takes/<name>.toml describes the sources, the audio clock fit that puts every source on the
reference timeline (reference_time = slope * source_time + offset, from prototype/sync_larissa_kadu.py),
and the analysis window. All outputs go to work/<name>/ (git-ignored).
"""
from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]  # .../dancecap
WORKSPACE = PACKAGE_ROOT.parent  # .../HeadMovement


@dataclass
class Source:
    id: str
    path: Path
    slope: float
    offset: float

    def to_reference(self, source_time: float) -> float:
        return self.slope * source_time + self.offset

    def to_source(self, reference_time: float) -> float:
        return (reference_time - self.offset) / self.slope


@dataclass
class Take:
    name: str
    sources: list[Source]
    window: tuple[float, float]  # reference seconds analysed in detail (poses, cameras per frame)
    keyframe_span: tuple[float, float]  # reference seconds sampled for the static room
    keyframe_step: float
    cfg: dict = field(repr=False)

    @property
    def work(self) -> Path:
        return PACKAGE_ROOT / "work" / self.name

    def source(self, cam_id: str) -> Source:
        return next(s for s in self.sources if s.id == cam_id)

    @classmethod
    def load(cls, path: str | Path) -> "Take":
        path = Path(path)
        if not path.suffix:
            path = PACKAGE_ROOT / "takes" / f"{path}.toml"
        cfg = tomllib.loads(path.read_text(encoding="utf-8"))
        root = (path.parent / cfg["source_dir"]).resolve() if not Path(cfg["source_dir"]).is_absolute() \
            else Path(cfg["source_dir"])
        alignment = json.loads(((path.parent / cfg["alignment"]).resolve()).read_text())
        exclude = set(cfg.get("exclude", []))
        sources = [Source(r["id"], root / r["name"], float(r["slope"]), float(r["offset"]))
                   for r in alignment["sources"] if r["id"] not in exclude and r.get("accepted", True)]
        return cls(
            name=cfg["name"],
            sources=sources,
            window=tuple(cfg["window"]),
            keyframe_span=tuple(cfg.get("keyframe_span", cfg["window"])),
            keyframe_step=float(cfg.get("keyframe_step", 2.0)),
            cfg=cfg,
        )
