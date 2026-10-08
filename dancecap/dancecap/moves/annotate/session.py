"""One annotation session: a take (reference time, any of its cameras) or a lone video (video time).

  Session.for_take('larissa_kadu')            labels in reference time; the camera picker offers every take source
  Session.for_video('some.mp4')               labels in that video's own time (time_base 'video'); a video that is a
                                              source of a known take opens that take instead (video_time=True keeps
                                              the video clock)
Files (work/ is git-ignored; nothing leaves the machine):
  work/<name>/moves/labels.json               the move timeline (labels.py contract) + 'unresolved' (unknown phrases)
  work/<name>/moves/narration/<id>/           audio.wav (or .webm) + audio16k.wav, log.json, transcript.json,
                                              result.json for every narration pass
  work/<name>/moves/narration/draft.json      autosaved working timeline (restored on the next start if newer)
  work/<name>/moves/narration/backups/        labels.json before every save
  work/<name>/moves/proxy/<cam>.mp4           optional H.264 proxy of a camera the browser cannot decode (HEVC)
For a lone video <name> = its file name as a slug ('WhatsApp Video 2023...' -> whatsapp_video_2023_...).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ...take import PACKAGE_ROOT, WORKSPACE, Take
from .. import ADDITIONS_PATH, GRAPH_PATH, PROVENANCES, VOCABULARY_PATH, ZOUK1_PATH
from ..graph import Graph, load_graph, slugify
from ..labels import Labels, LabelsError, Segment, to_reference, validate_labels
from ..narration import Lexicon
from .pipeline import TimelineContext
from .timemap import Grid

BROWSER_CODECS = {"h264", "vp8", "vp9", "av1"}
NO_WINDOW = 0x08000000 if os.name == "nt" else 0
UI_ONLY = re.compile(r"^_")
REPLACED_TOP_KEYS = ("placeholder", "note", "generator")  # a placeholder file's keys, dropped when it is replaced


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _rel(p: Path) -> str:
    try:
        return Path(p).resolve().relative_to(WORKSPACE.resolve()).as_posix()
    except ValueError:
        return Path(p).resolve().as_posix()


def probe(path: Path) -> dict:
    """codec / size / rotation / duration of a video's first video stream (ffprobe); {} when unavailable."""
    ff = shutil.which("ffprobe")
    if not ff or not Path(path).exists():
        return {}
    try:
        r = subprocess.run([ff, "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=codec_name,width,height,start_time,time_base:stream_side_data=rotation:format=duration",
                            "-of", "json", str(path)], capture_output=True, text=True, timeout=30,
                           creationflags=NO_WINDOW)
        d = json.loads(r.stdout or "{}")
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return {}
    s = (d.get("streams") or [{}])[0]
    rot = next((sd.get("rotation") for sd in s.get("side_data_list") or [] if "rotation" in sd), 0)
    out = {"codec": s.get("codec_name"), "width": s.get("width"), "height": s.get("height"),
           "rotation": int(rot or 0), "start_time": float(s.get("start_time") or 0.0), "time_base": s.get("time_base")}
    try:
        out["duration"] = float((d.get("format") or {}).get("duration"))
    except (TypeError, ValueError):
        pass
    return out


@dataclass
class Camera:
    id: str
    path: Path
    slope: float = 1.0
    offset: float = 0.0
    info: dict = field(default_factory=dict)
    proxy: Path | None = None

    def to_label(self, v: float) -> float:
        return self.slope * float(v) + self.offset

    def to_video(self, t: float) -> float:
        return (float(t) - self.offset) / self.slope

    @property
    def playable(self) -> bool | None:
        c = (self.info or {}).get("codec")
        return None if not c else c in BROWSER_CODECS

    def serve_path(self) -> Path:
        return self.proxy if self.proxy is not None and self.proxy.exists() else self.path

    def as_dict(self) -> dict:
        portrait = None
        w, h = self.info.get("width"), self.info.get("height")
        if w and h:
            portrait = (h > w) != (abs(self.info.get("rotation", 0)) % 180 == 90)
        return {"id": self.id, "name": self.path.name, "slope": self.slope, "offset": self.offset,
                "url": f"/video/{self.id}", "codec": self.info.get("codec"), "width": w, "height": h,
                "rotation": self.info.get("rotation", 0), "portrait": portrait,
                "duration": self.info.get("duration"), "playable": self.playable,
                "start_time": self.info.get("start_time", 0.0),
                "proxy": bool(self.proxy is not None and self.proxy.exists())}


class Session:
    def __init__(self, name: str, mode: str, cameras: list[Camera], video: Path | None = None,
                 work_root: Path | None = None, graph_path: Path = GRAPH_PATH, additions_path: Path = ADDITIONS_PATH,
                 vocabulary_path: Path = VOCABULARY_PATH, zouk1_path: Path = ZOUK1_PATH, beats: Path | None = None,
                 camera: str | None = None, annotator: str | None = None, settings: dict | None = None):
        if mode not in ("take", "video"):
            raise ValueError(mode)
        self.name, self.mode, self.cameras = name, mode, {c.id: c for c in cameras}
        self.video = Path(video) if video else None
        self.time_base = "reference" if mode == "take" else "video"
        self.work = Path(work_root or PACKAGE_ROOT / "work") / name
        self.moves_dir = self.work / "moves"
        self.labels_path = self.moves_dir / "labels.json"
        self.narration_dir = self.moves_dir / "narration"
        self.graph_path, self.additions_path = Path(graph_path), Path(additions_path)
        self.vocabulary_path, self.zouk1_path = Path(vocabulary_path), Path(zouk1_path)
        self.beats_path = Path(beats) if beats else None
        self.annotator = annotator or os.environ.get("DANCECAP_ANNOTATOR") or os.environ.get("USERNAME") or "annotator"
        self.settings = dict(settings or {})
        for c in self.cameras.values():
            p = self.moves_dir / "proxy" / f"{c.id}.mp4"
            c.proxy = p
        self.default_camera = camera if camera in self.cameras else self._pick_default()
        self._graph: Graph | None = None
        self._lex: Lexicon | None = None

    # -- construction -------------------------------------------------------------------------------------------
    @classmethod
    def for_take(cls, take: str, camera: str | None = None, probe_videos: bool = True, **kw) -> "Session":
        tk = Take.load(take)
        cams = [Camera(s.id, Path(s.path), float(s.slope), float(s.offset), probe(s.path) if probe_videos else {})
                for s in tk.sources]
        if camera and camera not in {c.id for c in cams}:
            m = next((c.id for c in cams if c.path.name == Path(camera).name), None)
            camera = m
        return cls(tk.name, "take", cams, camera=camera, **kw)

    @classmethod
    def for_video(cls, video: str | Path, name: str | None = None, video_time: bool = False,
                  probe_videos: bool = True, **kw) -> "Session":
        v = Path(video).expanduser().resolve()
        if not v.exists():
            raise FileNotFoundError(v)
        if not video_time:
            hit = find_take_of(v)
            if hit:
                return cls.for_take(hit[0], camera=hit[1], probe_videos=probe_videos, **kw)
        nm = name or video_session_name(v)
        return cls(nm, "video", [Camera("video", v, 1.0, 0.0, probe(v) if probe_videos else {})], video=v, **kw)

    def _pick_default(self) -> str:
        cams = list(self.cameras.values())
        ok = [c for c in cams if c.playable]
        if ok:  # the sharpest camera the browser can decode
            return max(ok, key=lambda c: (c.info.get("width") or 0) * (c.info.get("height") or 0)).id
        return cams[0].id

    # -- graph / vocabulary -------------------------------------------------------------------------------------
    @property
    def graph(self) -> Graph:
        if self._graph is None:
            self._graph = load_graph(self.graph_path)
        return self._graph

    @property
    def lexicon(self) -> Lexicon:
        if self._lex is None:
            self._lex = Lexicon.from_graph(self.graph, self.vocabulary_path if self.vocabulary_path.exists() else None)
        return self._lex

    def reload_graph(self, data: dict | None = None) -> None:
        self._graph = Graph(data) if data is not None else None
        self._lex = None

    def moves_for_ui(self) -> list[dict]:
        g = self.graph
        out = []
        for slug, m in g.moves.items():
            n0 = g.nodes.get((m.get("nodes") or [None])[0]) or {}
            out.append({"slug": slug, "name": m.get("name") or slug, "name_en": m.get("name_en"),
                        "aliases": [a for a in m.get("aliases") or [] if a], "family": m.get("family"),
                        "tone": n0.get("tone_name"), "status": m.get("status"), "band": m.get("band"),
                        "energy": m.get("energy"), "instances": len(m.get("nodes") or []),
                        "description": (m.get("description") or "")[:300] or None})
        return sorted(out, key=lambda x: x["name"].lower())

    # -- clocks, grid, proposals -------------------------------------------------------------------------------
    def camera(self, cam: str | None) -> Camera:
        return self.cameras.get(cam or "") or self.cameras[self.default_camera]

    def to_label(self, cam: str, v: float) -> float:
        return self.camera(cam).to_label(v)

    def beats(self) -> list | None:
        cands = [self.beats_path] if self.beats_path else []
        if self.mode == "take":
            cands += [self.work / "beats.json", self.work / "beats" / "beats.json", self.work / "timing" / "beats.json"]
        for p in cands:
            if p and p.exists():
                try:
                    d = json.loads(p.read_text(encoding="utf-8"))
                    return d.get("grid") if isinstance(d, dict) else d
                except (OSError, ValueError):
                    return None
        return None

    def grid(self) -> Grid | None:
        b = self.beats()
        return Grid(b) if b else None

    def proposals(self) -> dict | None:
        p = self.moves_dir / "proposals.json"
        if self.mode != "take" or not p.exists():
            return None
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return {"coverage": d.get("coverage"), "note": d.get("note"),
                "boundaries": [{"t": b.get("t"), "strength": b.get("strength"), "kind": b.get("kind"),
                                "top_changes": [c.get("feature") if isinstance(c, dict) else c
                                                for c in (b.get("top_changes") or [])][:4]}
                               for b in d.get("boundaries") or [] if b.get("t") is not None],
                "segments": [{k: s.get(k) for k in ("id", "t0", "t1", "energy_band", "energy_label", "cluster",
                                                     "repeat_of", "confidence")} for s in d.get("segments") or []]}

    def counterbalance(self) -> list | None:
        p = self.work / "counterbalance" / "counterbalance.json"
        if self.mode != "take" or not p.exists():
            return None
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return [{"t0": i.get("t0"), "t1": i.get("t1"), "kind": i.get("kind"), "confidence": i.get("confidence")}
                for i in d.get("intervals") or []]

    def context(self) -> TimelineContext:
        prop = self.proposals() or {}
        return TimelineContext(to_label=self.to_label, slope=lambda cam: self.camera(cam).slope, grid=self.grid(),
                               boundaries=[float(b["t"]) for b in prop.get("boundaries") or []], take=self.name)

    # -- labels ------------------------------------------------------------------------------------------------
    def load_working(self) -> dict:
        """{segments (incl. unresolved, move null), placeholder: {count, note} | None, warnings, saved}."""
        out = {"segments": [], "placeholder": None, "warnings": [], "saved": None}
        if not self.labels_path.exists():
            return out
        try:
            doc = json.loads(self.labels_path.read_text(encoding="utf-8"))
            lab = Labels.from_dict(doc)
        except (OSError, ValueError, LabelsError, KeyError, TypeError) as e:
            out["warnings"].append(f"labels.json unreadable: {e}")
            return out
        if lab.time_base == "video" and self.mode == "take":
            try:
                lab = to_reference(lab)
                out["warnings"].append("labels.json was in video time: converted to reference time")
            except Exception as e:  # noqa: BLE001
                out["warnings"].append(f"labels.json in video time could not be converted: {e}")
                return out
        elif lab.time_base != self.time_base:
            out["warnings"].append(f"labels.json time base {lab.time_base!r} != this session's {self.time_base!r}")
        segs = [s.to_dict() for s in lab.segments]
        ph = [s for s in segs if s.get("provenance") == "placeholder"]
        if ph:
            out["placeholder"] = {"count": len(ph), "note": doc.get("note") or "placeholder - not an analysis"}
        out["segments"] = [s for s in segs if s.get("provenance") != "placeholder"]
        for u in doc.get("unresolved") or []:
            if isinstance(u, dict) and u.get("t0") is not None and u.get("t1") is not None:
                u = dict(u)
                u["move"] = None
                out["segments"].append(u)
        out["segments"].sort(key=lambda s: (float(s["t0"]), float(s["t1"])))
        out["saved"] = datetime.fromtimestamp(self.labels_path.stat().st_mtime, timezone.utc).isoformat()
        return out

    def _clean(self, segs: list[dict]) -> list[dict]:
        out = []
        for s in segs:
            d = {k: v for k, v in s.items() if not UI_ONLY.match(str(k))}
            d["t0"], d["t1"] = float(d["t0"]), float(d["t1"])
            d.setdefault("tier", "couple")
            d.setdefault("provenance", "human")
            if d.get("provenance") not in PROVENANCES:
                d["provenance"] = "human"
            if d.get("confidence") is not None:
                d["confidence"] = min(1.0, max(0.0, float(d["confidence"])))
            if not d.get("annotator"):
                d["annotator"] = self.annotator
            if not d.get("id"):
                d["id"] = Segment(0, 1, "x").id
            out.append(d)
        return out

    def build_labels(self, segments: list[dict]) -> tuple[Labels, list[dict]]:
        segs = self._clean(segments)
        resolved = [s for s in segs if s.get("move")]
        unresolved = [{k: v for k, v in s.items() if k != "move"} for s in segs if not s.get("move")]
        prev = {}
        if self.labels_path.exists():
            try:
                prev = json.loads(self.labels_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                prev = {}
        extra = {k: v for k, v in prev.items()
                 if k not in ("version", "take", "time_base", "video", "segments", "unresolved")
                 and not (prev.get("placeholder") and k in REPLACED_TOP_KEYS)}
        extra["unresolved"] = unresolved
        extra["annotation_tool"] = {"module": "dancecap.moves.annotate", "saved": now_iso(),
                                    "graph": _rel(self.graph_path)}
        lab = Labels(take=self.name, segments=[Segment.from_dict(s) for s in resolved], time_base=self.time_base,
                     video=_rel(self.video) if self.mode == "video" and self.video else None, extra=extra)
        return lab, unresolved

    def validate(self, segments: list[dict]) -> list[str]:
        try:
            lab, unresolved = self.build_labels(segments)
        except (KeyError, TypeError, ValueError) as e:
            return [f"bad segment data: {e}"]
        errs = validate_labels(lab, self.graph)
        allsegs = sorted(self._clean(segments), key=lambda s: (s["t0"], s["t1"]))
        for a, b in zip(allsegs, allsegs[1:]):
            if b["t0"] < a["t1"] - 1e-6 and (not a.get("move") or not b.get("move")):
                errs.append(f"unresolved segment overlaps: {a['id']} / {b['id']}")
        return errs

    def save(self, segments: list[dict]) -> dict:
        errs = self.validate(segments)
        if errs:
            return {"ok": False, "problems": errs}
        lab, unresolved = self.build_labels(segments)
        self.labels_path.parent.mkdir(parents=True, exist_ok=True)
        backup = None
        if self.labels_path.exists():
            bdir = self.narration_dir / "backups"
            bdir.mkdir(parents=True, exist_ok=True)
            backup = bdir / f"labels-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}.json"
            shutil.copyfile(self.labels_path, backup)
        tmp = self.labels_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(lab.to_dict(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(tmp, self.labels_path)
        return {"ok": True, "path": str(self.labels_path), "segments": len(lab.segments),
                "unresolved": len(unresolved), "labelled_seconds": round(lab.labelled_seconds, 3),
                "provenance": lab.provenance, "backup": str(backup) if backup else None}

    # -- drafts / narrations -----------------------------------------------------------------------------------
    def save_draft(self, segments: list[dict]) -> Path:
        self.narration_dir.mkdir(parents=True, exist_ok=True)
        p = self.narration_dir / "draft.json"
        p.write_text(json.dumps({"saved": now_iso(), "segments": segments}, ensure_ascii=False), encoding="utf-8")
        return p

    def load_draft(self) -> dict | None:
        p = self.narration_dir / "draft.json"
        if not p.exists():
            return None
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        newer = not self.labels_path.exists() or p.stat().st_mtime > self.labels_path.stat().st_mtime
        return {"saved": d.get("saved"), "segments": d.get("segments") or [], "newer_than_labels": newer}

    def clear_draft(self) -> None:
        (self.narration_dir / "draft.json").unlink(missing_ok=True)

    def new_narration_dir(self) -> Path:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        d = self.narration_dir / stamp
        k = 1
        while d.exists():
            k += 1
            d = self.narration_dir / f"{stamp}-{k}"
        d.mkdir(parents=True)
        return d

    def narrations(self) -> list[dict]:
        out = []
        if not self.narration_dir.exists():
            return out
        for d in sorted(self.narration_dir.iterdir(), reverse=True):
            if not d.is_dir() or not (d / "log.json").exists():
                continue
            r = {"id": d.name, "has_result": (d / "result.json").exists(),
                 "has_transcript": (d / "transcript.json").exists()}
            if r["has_result"]:
                try:
                    res = json.loads((d / "result.json").read_text(encoding="utf-8"))
                    r.update(summary=res.get("summary"), engine=res.get("engine"), spans=res.get("spans"),
                             retranscribe=(res.get("asr") or {}).get("retranscribe"))
                except (OSError, ValueError):
                    pass
            out.append(r)
        return out

    # -- proxies -----------------------------------------------------------------------------------------------
    def make_proxy(self, cam: str, log=print) -> Path:
        """H.264 proxy (<= 960 px tall) for a camera the browser cannot decode. Every frame keeps its presentation
        timestamp exactly (-fps_mode passthrough, -enc_time_base demux, the source's track timescale; checked on
        cam 04: 1790/1790 frames, max |dPTS| 0), so the camera's slope / offset stay valid."""
        c = self.camera(cam)
        info = c.info or probe(c.path)
        tb = str(info.get("time_base") or "")
        timescale = tb.split("/")[1] if tb.startswith("1/") and tb.split("/")[1].isdigit() else "90000"
        ff = shutil.which("ffmpeg")
        if not ff:
            raise RuntimeError("ffmpeg not found")
        out = self.moves_dir / "proxy" / f"{c.id}.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".part.mp4")
        cmd = [ff, "-v", "error", "-y", "-i", str(c.path), "-map", "0:v:0", "-map", "0:a:0?", "-c:v", "libx264",
               "-preset", "veryfast", "-crf", "24", "-vf", "scale=-2:'min(960,ih)'", "-pix_fmt", "yuv420p",
               "-fps_mode", "passthrough", "-enc_time_base", "demux", "-video_track_timescale", timescale,
               "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(tmp)]
        log(f"[annotate] proxy {c.id}: {' '.join(cmd)}")
        r = subprocess.run(cmd, capture_output=True, text=True, creationflags=NO_WINDOW)
        if r.returncode != 0:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"ffmpeg proxy failed: {r.stderr.strip()[-300:]}")
        os.replace(tmp, out)
        c.proxy = out
        return out

    # -- summary for the UI ------------------------------------------------------------------------------------
    def info(self) -> dict:
        g = self.grid()
        return {"name": self.name, "mode": self.mode, "time_base": self.time_base,
                "video": _rel(self.video) if self.video else None,
                "labels_path": str(self.labels_path), "narration_dir": str(self.narration_dir),
                "cameras": [c.as_dict() for c in self.cameras.values()], "default_camera": self.default_camera,
                "beats": g.to_list() if g is not None else None, "proposals": self.proposals(),
                "counterbalance": self.counterbalance(), "annotator": self.annotator,
                "graph": _rel(self.graph_path), "additions": _rel(self.additions_path),
                "export_enabled": self.mode == "take", "settings": self.settings}


def video_session_name(v: Path) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", slugify(v.stem).replace("-", "_")).strip("_")[:48] or "video"
    if (PACKAGE_ROOT / "takes" / f"{s}.toml").exists():
        s += "_video"
    return s


def find_take_of(video: Path) -> tuple[str, str] | None:
    """(take, camera id) when `video` is a source of a take in takes/*.toml."""
    for toml in sorted((PACKAGE_ROOT / "takes").glob("*.toml")):
        try:
            tk = Take.load(toml)
        except Exception:  # noqa: BLE001
            continue
        for s in tk.sources:
            try:
                if Path(s.path).resolve() == video.resolve():
                    return tk.name, s.id
            except OSError:
                continue
    return None
