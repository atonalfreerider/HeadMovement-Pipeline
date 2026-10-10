"""The recap sources and their clocks on the master timeline.

Convention (dancecap.take.Source, recap_sync): reference_time = slope * source_time + offset, reference = the master camera's
video time (01-Recap-Tina). Raw per-source outputs of this package stay in each source's OWN time; everything that is fused
or compared across sources goes through `Clock.to_master` / `to_source`.

Where the clocks come from, in order of trust:
  1. work/<take>/sync/sync.json (written by dancecap.recap_sync: GCC-PHAT audio alignment against the master),
     `cameras[]` {id, offset_s, slope, offset_source, confidence}
  2. camera-time-offsets.json of the recap folder (the owner's numbers, matched by FILE NAME because the keys are paths of another
     machine); provisional
  3. nothing: the source has no clock (04-Recap-Tony until recap_sync finds one) and is left out of cross-source steps.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from . import MASTER, SOURCES, SYNC


@dataclass(frozen=True)
class Clock:
    id: str
    slope: float
    offset: float
    origin: str            # sync.json | camera-time-offsets.json | master
    provisional: bool = False

    def to_master(self, t: float) -> float:
        return self.slope * t + self.offset

    def to_source(self, m: float) -> float:
        return (m - self.offset) / self.slope


def _basename(key: str) -> str:
    return re.split(r"[\\/]", key)[-1]


def read_owner_offsets(recap_dir: Path | str | None) -> dict[str, float]:
    """{file name: offset} from the owner's camera-time-offsets.json (keys are Windows paths of another machine)."""
    if not recap_dir:
        return {}
    p = Path(recap_dir) / "camera-time-offsets.json"
    if not p.exists():
        return {}
    raw = json.loads(p.read_text(encoding="utf-8"))
    return {_basename(k): float(v) for k, v in raw.items() if v is not None}


def load_clocks(sync_json: Path | str | None = None, recap_dir: Path | str | None = None) -> dict[str, Clock | None]:
    """{source id: Clock or None}. A sync.json entry with a null offset (Tony before the audio alignment finds one) falls back to
    the owner's number if there is one, else None."""
    sync_path = Path(sync_json) if sync_json else SYNC / "sync.json"
    owner = read_owner_offsets(recap_dir)
    clocks: dict[str, Clock | None] = {sid: None for sid in SOURCES}
    cams: dict[str, dict] = {}
    if sync_path.exists():
        try:
            doc = json.loads(sync_path.read_text(encoding="utf-8"))
            cams = {str(c.get("id")): c for c in doc.get("cameras", [])}
        except (OSError, json.JSONDecodeError):
            cams = {}
    for sid, fname in SOURCES.items():
        c = cams.get(sid)
        if sid == MASTER:
            clocks[sid] = Clock(sid, 1.0, 0.0, "master")
            continue
        if c is not None and c.get("offset_s") is not None:
            clocks[sid] = Clock(sid, float(c.get("slope") or 1.0), float(c["offset_s"]), "sync.json",
                                provisional=float(c.get("confidence") or 0.0) < 0.0)
        elif fname in owner:
            clocks[sid] = Clock(sid, 1.0, owner[fname], "camera-time-offsets.json", provisional=True)
    return clocks


def master_span(clock: Clock, duration: float) -> tuple[float, float]:
    """(master start, master end) of a source of `duration` seconds."""
    return clock.to_master(0.0), clock.to_master(duration)


def discover(source_dir: Path | str) -> dict[str, Path]:
    """{id: video path} of the NN-Recap-*.mp4 files in the recap folder."""
    out: dict[str, Path] = {}
    for p in sorted(Path(source_dir).glob("[0-9][0-9]-Recap-*.mp4")):
        out[p.name[:2]] = p
    return out
