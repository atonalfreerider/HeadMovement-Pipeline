"""Move timelines: work/<take>/moves/labels.json (version 1) - dataclasses, load/save/validate, time bases.

labels.json:
  {version:1, take, time_base:'reference'|'video', video:<path or null; required when time_base = video>,
   segments:[{id, t0, t1, move:<move slug>, node:<graph node id or null>, tier:'couple',
              provenance:'human'|'narration'|'auto'|'placeholder', confidence:0..1|null, annotator, transcript,
              notes, created}]}
  Unknown keys (top level and per segment) are preserved on load/save. Segments are [t0, t1) in seconds;
  couple-tier segments must not overlap. `move` must be a move slug of the graph (scaffold, proposed or accepted;
  aliases are resolved by `normalise_moves`, never stored). `node` pins one instance of a duplicated move.

Time bases:
  reference  the take's reference clock (camera 06 for larissa_kadu) - the pipeline's time everywhere.
  video      seconds of one source video (its PTS clock); reference = Source.to_reference(t) (affine audio fit).
  capture    (Unity copies only - never in work/<take>/moves/labels.json, where validate_labels rejects it) the
             capture's audio clock = reference + capture.json time_to_audio, i.e. the clock of
             CaptureTimeline.AudioTimes in the viewer (see export.py). Converting a capture-time document back
             (to_reference) needs its time_to_audio key.
"""
from __future__ import annotations

import json
import math
import uuid
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import PROVENANCES, TIERS, moves_dir

VERSION = 1
TIME_BASES = ("reference", "video")  # labels.json contract (work/<take>/moves/)
UNITY_TIME_BASE = "capture"  # Unity copies only (export.py)
EPS = 1e-6


class LabelsError(ValueError):
    pass


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def new_id(prefix: str = "seg") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


@dataclass
class Segment:
    t0: float
    t1: float
    move: str
    id: str = field(default_factory=new_id)
    node: str | None = None
    tier: str = "couple"
    provenance: str = "human"
    confidence: float | None = None
    annotator: str | None = None
    transcript: str | None = None
    notes: str | None = None
    created: str | None = field(default_factory=now_iso)
    extra: dict = field(default_factory=dict)  # unknown keys, preserved

    @property
    def duration(self) -> float:
        return self.t1 - self.t0

    def to_dict(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if k != "extra"}
        d = {"id": d.pop("id"), **d}
        d["t0"], d["t1"] = round(float(self.t0), 6), round(float(self.t1), 6)
        d.update(self.extra)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Segment":
        known = {f.name for f in fields(cls)} - {"extra"}
        kw = {k: d[k] for k in known if k in d}
        kw["t0"], kw["t1"] = float(d["t0"]), float(d["t1"])
        if "created" not in d:
            kw["created"] = None
        return cls(**kw, extra={k: v for k, v in d.items() if k not in known})


@dataclass
class Labels:
    take: str
    segments: list[Segment] = field(default_factory=list)
    time_base: str = "reference"
    video: str | None = None
    version: int = VERSION
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {"version": self.version, "take": self.take, "time_base": self.time_base, "video": self.video}
        d.update({k: v for k, v in self.extra.items() if k not in d and k != "segments"})
        d["segments"] = [s.to_dict() for s in sorted(self.segments, key=lambda s: (s.tier, s.t0, s.t1))]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Labels":
        if "segments" not in d or "take" not in d:
            raise LabelsError("labels.json needs 'take' and 'segments'")
        known = {"version", "take", "time_base", "video", "segments"}
        return cls(take=d["take"], segments=[Segment.from_dict(s) for s in d["segments"]],
                   time_base=d.get("time_base", "reference"), video=d.get("video"),
                   version=d.get("version", VERSION), extra={k: v for k, v in d.items() if k not in known})

    def tier(self, tier: str = "couple") -> list[Segment]:
        return sorted((s for s in self.segments if s.tier == tier), key=lambda s: (s.t0, s.t1))

    @property
    def provenance(self) -> str:
        """'placeholder' / 'human' / ... when uniform, else 'mixed:<a>+<b>' (sorted)."""
        p = sorted({s.provenance for s in self.segments})
        return p[0] if len(p) == 1 else ("mixed:" + "+".join(p) if p else "empty")

    @property
    def labelled_seconds(self) -> float:
        return float(sum(s.duration for s in self.tier("couple")))


def labels_path(take: str) -> Path:
    return moves_dir(take) / "labels.json"


def load_labels(src: str | Path) -> Labels:
    """Labels from a take name (work/<take>/moves/labels.json) or a file path."""
    p = Path(src)
    if not p.suffix:
        p = labels_path(str(src))
    return Labels.from_dict(json.loads(p.read_text(encoding="utf-8")))


def save_labels(labels: Labels, path: str | Path | None = None, graph=None, check: bool = True) -> Path:
    """Write labels.json (validated first unless check=False; raises LabelsError on problems)."""
    if check:
        errs = validate_labels(labels, graph)
        if errs:
            raise LabelsError("; ".join(errs[:10]))
    p = Path(path) if path else labels_path(labels.take)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(labels.to_dict(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return p


def validate_labels(labels: Labels | dict, graph=None) -> list[str]:
    """Contract problems (empty list = valid). With a graph (moves.graph.Graph), moves/nodes are checked too."""
    if isinstance(labels, dict):
        try:
            labels = Labels.from_dict(labels)
        except (LabelsError, KeyError, TypeError, ValueError) as e:
            return [f"unreadable labels: {e}"]
    errs = []
    if labels.version != VERSION:
        errs.append(f"version {labels.version!r} != {VERSION}")
    if labels.time_base == UNITY_TIME_BASE:
        errs.append("time_base 'capture' is only used in the Unity copies (StreamingAssets/<capture>/moves/); "
                    "store labels in 'reference' or 'video' time")
    elif labels.time_base not in TIME_BASES:
        errs.append(f"time_base {labels.time_base!r} not in {TIME_BASES}")
    if labels.time_base == "video" and not labels.video:
        errs.append("time_base 'video' needs 'video' (the source video path)")
    ids = [s.id for s in labels.segments]
    if len(set(ids)) != len(ids):
        errs.append("segment ids are not unique")
    for s in labels.segments:
        tag = f"segment {s.id} [{s.t0:.3f}, {s.t1:.3f})"
        if not (math.isfinite(s.t0) and math.isfinite(s.t1)):
            errs.append(f"{tag}: non-finite time")
        elif not s.t0 < s.t1:
            errs.append(f"{tag}: needs t0 < t1")
        if s.provenance not in PROVENANCES:
            errs.append(f"{tag}: provenance {s.provenance!r} not in {PROVENANCES}")
        if s.tier not in TIERS:
            errs.append(f"{tag}: tier {s.tier!r} not in {TIERS}")
        if s.confidence is not None and not (0.0 <= float(s.confidence) <= 1.0):
            errs.append(f"{tag}: confidence {s.confidence!r} outside 0..1")
        if not s.move:
            errs.append(f"{tag}: move is empty")
        elif graph is not None and s.move not in graph.moves:
            hint = graph.resolve_move(s.move)
            errs.append(f"{tag}: unknown move {s.move!r}" + (f" (alias of {hint!r}: store the slug)" if hint else
                                                                  " (add it to taxonomy/additions.toml)"))
        if s.node is not None and graph is not None:
            n = graph.nodes.get(s.node)
            if n is None:
                errs.append(f"{tag}: unknown node {s.node!r}")
            elif n.get("move") != s.move:
                errs.append(f"{tag}: node {s.node} is move {n.get('move')!r}, not {s.move!r}")
    for tier in TIERS:
        seq = labels.tier(tier)
        for a, b in zip(seq, seq[1:]):
            if b.t0 < a.t1 - EPS:
                errs.append(f"tier {tier}: segments {a.id} [{a.t0:.3f}, {a.t1:.3f}) and {b.id} "
                            f"[{b.t0:.3f}, {b.t1:.3f}) overlap")
    return errs


def normalise_moves(labels: Labels, graph) -> list[str]:
    """Replace aliases / authored names by move slugs in place; returns the unresolvable moves."""
    bad = []
    for s in labels.segments:
        m = graph.resolve_move(s.move)
        if m is None:
            bad.append(s.move)
        elif m != s.move:
            s.extra.setdefault("move_as_given", s.move)
            s.move = m
    return bad


# ---------------------------------------------------------------------------------------------------------------
# time bases
# ---------------------------------------------------------------------------------------------------------------

def source_for_video(take: str, video: str | Path):
    """The take Source (camera) whose file is `video` (matched by file name, or by camera id '06')."""
    from ..take import Take
    tk = Take.load(take)
    v = Path(str(video))
    for s in tk.sources:
        if s.path.name == v.name or s.id == str(video) or Path(s.path).resolve() == v.resolve():
            return s
    raise LabelsError(f"{video} is not a source of take {take} ({[s.path.name for s in tk.sources]})")


def video_to_reference(t: float, take: str, video: str | Path) -> float:
    return float(source_for_video(take, video).to_reference(t))


def reference_to_video(t: float, take: str, video: str | Path) -> float:
    return float(source_for_video(take, video).to_source(t))


def _map_unresolved(out: Labels, fn) -> None:
    """The 'unresolved' items (unknown narrated phrases, annotation tool) on the new clock too (copies, not shared)."""
    items = out.extra.get("unresolved")
    if not isinstance(items, list):
        return
    conv = []
    for u in items:
        u = dict(u) if isinstance(u, dict) else u
        if isinstance(u, dict) and u.get("t0") is not None and u.get("t1") is not None:
            u["t0"], u["t1"] = fn(float(u["t0"])), fn(float(u["t1"]))
        conv.append(u)
    out.extra["unresolved"] = conv


def to_reference(labels: Labels) -> Labels:
    """A copy of the labels on the reference clock (video-time labels mapped through the camera's affine clock);
    the 'unresolved' items (unknown narrated phrases) are converted with the segments."""
    d = labels.to_dict()
    out = Labels.from_dict(d)
    if labels.time_base == "reference":
        return out
    if labels.time_base == "video":
        src = source_for_video(labels.take, labels.video)
        for s in out.segments:
            s.t0, s.t1 = float(src.to_reference(s.t0)), float(src.to_reference(s.t1))
        _map_unresolved(out, lambda t: float(src.to_reference(t)))
        out.extra["converted_from"] = {"time_base": "video", "video": labels.video, "camera": src.id,
                                       "slope": src.slope, "offset": src.offset}
        out.time_base, out.video = "reference", None
        return out
    if labels.time_base == UNITY_TIME_BASE:
        if labels.extra.get("time_to_audio") is None:
            raise LabelsError("capture-time labels need 'time_to_audio' (capture time = reference + time_to_audio)")
        tta = float(labels.extra["time_to_audio"])
        for s in out.segments:
            s.t0, s.t1 = s.t0 - tta, s.t1 - tta
        _map_unresolved(out, lambda t: t - tta)
        out.time_base = "reference"
        return out
    raise LabelsError(f"unknown time_base {labels.time_base!r}")


@dataclass
class CaptureClock:
    """The time base of a Unity capture: capture time = reference time + time_to_audio (= the viewer's audio clock,
    CaptureTimeline.AudioTimes). time_to_audio comes from capture.json (v3), else audio_offset - times[0] (v2 rule
    in CaptureManifest.BuildTimeline), else 0."""
    name: str
    time_to_audio: float
    times: np.ndarray  # reference seconds of every frame (times.json), may be empty
    take: str | None = None
    folder: Path | None = None

    @classmethod
    def from_capture(cls, folder: str | Path) -> "CaptureClock":
        folder = Path(folder)
        man = json.loads((folder / "capture.json").read_text(encoding="utf-8"))
        times = np.zeros(0)
        tp = folder / (man.get("times") or "times.json")
        if tp.exists():
            times = np.asarray(json.loads(tp.read_text()), float)
        tta = man.get("time_to_audio")
        if tta is None:
            tta = float(man.get("audio_offset", 0.0)) - float(times[0]) if len(times) else 0.0
        return cls(folder.name, float(tta) + 0.0, times, (man.get("provenance") or {}).get("take"), folder)

    def to_capture(self, t_ref: float) -> float:
        return float(t_ref) + self.time_to_audio

    def to_reference(self, t_cap: float) -> float:
        return float(t_cap) - self.time_to_audio

    @property
    def range_reference(self) -> tuple[float, float] | None:
        return (float(self.times[0]), float(self.times[-1])) if len(self.times) else None

    def frame(self, t_ref: float) -> int | None:
        """Nearest frame index (clamped), as CaptureTimeline.FrameAt."""
        if not len(self.times):
            return None
        i = int(np.searchsorted(self.times, t_ref))
        if i <= 0:
            return 0
        if i >= len(self.times):
            return len(self.times) - 1
        return i - 1 if t_ref - self.times[i - 1] <= self.times[i] - t_ref else i

    def covers(self, t0: float, t1: float) -> bool:
        r = self.range_reference
        return r is None or (t1 > r[0] and t0 < r[1])
