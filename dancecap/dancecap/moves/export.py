"""Unity copies of the move files: head-movement/Assets/StreamingAssets/<capture>/moves/{graph,labels,path,fingerprint}.json

    python -m dancecap.moves export synthetic                       every capture whose capture.json provenance.take
                                                                     is the take (01_SyntheticSMPLX for synthetic)
    python -m dancecap.moves export larissa_kadu --capture 02_LarissaKadu

Capture time base: `capture time = reference time + time_to_audio` (capture.json; v2 captures without it:
audio_offset - times[0]) - exactly the clock of the viewer's CaptureTimeline.AudioTimes, so a segment is current
when t0 <= audio time < t1 and its frames are CaptureTimeline.FrameAt(t). Every exported time keeps its reference
value next to it (t0_reference / t1_reference) plus the nearest capture frames f0 / f1 and in_capture (the segment
overlaps the capture's frames); labels.json 'unresolved' items (unknown narrated phrases = unlabelled gaps) are
converted the same way. The labelled range may extend beyond the capture (the fingerprint covers the whole
labelled dance; the capture only the reconstructed window).

Coordinates: graph positions stay as authored in Zouk1 (that app is Unity: left-handed, y-up - already Unity
axes); graph.display.offset is the suggested placement. There are no world-frame positions in these files. Each
file carries time_base:'capture', frame:'unity', capture, time_to_audio.

NOTE: `python -m dancecap export <take> --capture <Name>` (export_unity.py) deletes and rewrites the whole capture
folder, so re-run this export after every capture export. The files are also written to
work/<take>/export/<capture>/moves/ when that folder exists.

Consistency: the export refuses (ExportError) when path.json no longer matches labels.json + the graph, or when
fingerprint.json was computed from other labels / path / counterbalance files (sha256 in fingerprint.sources) - re-run
`python -m dancecap.moves run <take>` (path + fingerprint + export) instead of shipping stale files to the viewer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import PLACEHOLDER_BANNER, UNITY_STREAMING, work_dir
from .fingerprint import fingerprint_file, source_files, stale_sources
from .graph import GRAPH_PATH, load_graph
from .labels import CaptureClock, Labels, labels_path, load_labels, to_reference, validate_labels
from .path import path_file, validate_path

FILES = ("graph.json", "labels.json", "path.json", "fingerprint.json")


class ExportError(ValueError):
    pass


def consistency_problems(take: str, graph, lab_doc: dict | None = None) -> list[str]:
    """Why path.json / fingerprint.json of a take do not match its labels (+ graph, counterbalance); [] when fine."""
    lab_doc = lab_doc if lab_doc is not None else json.loads(labels_path(take).read_text(encoding="utf-8"))
    out = []
    pf, ff = path_file(take), fingerprint_file(take)
    if pf.exists():
        out += [f"path.json: {e}" for e in validate_path(json.loads(pf.read_text(encoding="utf-8")), graph,
                                                         Labels.from_dict(lab_doc))]
    if ff.exists():
        if not pf.exists():
            out.append("fingerprint.json exists without path.json")
        else:
            out += [f"fingerprint.json: {e}" for e in
                    stale_sources(json.loads(ff.read_text(encoding="utf-8")), source_files(take))]
    return out


def captures_for(take: str, streaming: Path = UNITY_STREAMING) -> list[Path]:
    """Capture folders whose capture.json says provenance.take == take."""
    out = []
    for d in sorted(Path(streaming).iterdir()) if Path(streaming).exists() else []:
        cj = d / "capture.json"
        if d.is_dir() and cj.exists():
            try:
                man = json.loads(cj.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if (man.get("provenance") or {}).get("take") == take:
                out.append(d)
    return out


def _stamp(doc: dict, clock: CaptureClock) -> dict:
    r = clock.range_reference
    doc.update({"time_base": "capture", "frame": "unity", "capture": clock.name,
                "time_to_audio": clock.time_to_audio,
                "capture_range": [round(clock.to_capture(r[0]), 6), round(clock.to_capture(r[1]), 6)] if r else None,
                "time_note": "capture time = reference time + time_to_audio (CaptureTimeline audio clock); "
                             "*_reference keep the pipeline's reference seconds"})
    return doc


def _times(item: dict, clock: CaptureClock) -> dict:
    t0, t1 = float(item["t0"]), float(item["t1"])
    item.update({"t0": round(clock.to_capture(t0), 6), "t1": round(clock.to_capture(t1), 6),
                 "t0_reference": round(t0, 6), "t1_reference": round(t1, 6),
                 "f0": clock.frame(t0), "f1": clock.frame(t1), "in_capture": clock.covers(t0, t1)})
    return item


def convert_labels(labels_doc: dict, clock: CaptureClock) -> dict:
    from .labels import Labels
    lab = to_reference(Labels.from_dict(labels_doc))
    d = json.loads(json.dumps(lab.to_dict()))
    d["segments"] = [_times(s, clock) for s in d["segments"]]
    if isinstance(d.get("unresolved"), list):  # unknown narrated phrases (unlabelled gaps): same clock as segments
        d["unresolved"] = [_times(u, clock) if isinstance(u, dict) and u.get("t0") is not None
                           and u.get("t1") is not None else u for u in d["unresolved"]]
    d["video"] = None
    if lab.provenance == "placeholder":
        d["banner"] = PLACEHOLDER_BANNER
    return _stamp(d, clock)


def convert_path(path_doc: dict, clock: CaptureClock) -> dict:
    d = json.loads(json.dumps(path_doc))
    d["steps"] = [_times(s, clock) for s in d["steps"]]
    for s in d["steps"]:  # routed steps (path_overrides): the inferred nodes' crossing times follow the capture clock
        if s.get("via_t"):
            s["via_t_reference"] = list(s["via_t"])
            s["via_t"] = [round(clock.to_capture(x), 6) for x in s["via_t"]]
    if d.get("provenance") == "placeholder":
        d["banner"] = PLACEHOLDER_BANNER
    return _stamp(d, clock)


def convert_fingerprint(fp: dict, clock: CaptureClock) -> dict:
    d = json.loads(json.dumps(fp))
    if d.get("longest_phrase"):
        _times(d["longest_phrase"], clock)
    if d.get("provenance") == "placeholder":
        d["banner"] = PLACEHOLDER_BANNER
    return _stamp(d, clock)


def convert_graph(graph_doc: dict, clock: CaptureClock) -> dict:
    d = json.loads(json.dumps(graph_doc))
    d["positions"] = "as authored in Zouk1.json (a Unity app: left-handed, y-up = Unity axes); not world positions"
    return _stamp(d, clock)


def export_moves(take: str, capture: str | Path, streaming: Path = UNITY_STREAMING,
                 graph_path: Path = GRAPH_PATH, allow_stale: bool = False) -> dict:
    """Write the Unity copies for one capture; returns a report {capture, files, segments, in_capture, ...}.
    Raises ExportError when path / fingerprint are stale (see consistency_problems) unless allow_stale."""
    cdir = Path(capture) if Path(capture).is_absolute() else Path(streaming) / str(capture)
    if not (cdir / "capture.json").exists():
        raise FileNotFoundError(f"{cdir}/capture.json missing (export the capture first)")
    clock = CaptureClock.from_capture(cdir)
    graph = load_graph(graph_path)
    lab_doc = json.loads(labels_path(take).read_text(encoding="utf-8"))
    errs = validate_labels(lab_doc, graph)
    if errs:
        raise ValueError(f"labels invalid: {errs[:5]}")
    stale = consistency_problems(take, graph, lab_doc)
    if stale and not allow_stale:
        raise ExportError(f"{take}: stale moves files, not exported (run `python -m dancecap.moves run {take}`): "
                          + "; ".join(stale[:5]))
    docs = {"graph.json": convert_graph(graph.data, clock), "labels.json": convert_labels(lab_doc, clock)}
    for name, src, conv in (("path.json", path_file(take), convert_path),
                            ("fingerprint.json", fingerprint_file(take), convert_fingerprint)):
        if src.exists():
            docs[name] = conv(json.loads(src.read_text(encoding="utf-8")), clock)
    targets = [cdir / "moves"]
    mirror = work_dir(take) / "export" / cdir.name
    if mirror.exists():
        targets.append(mirror / "moves")
    for t in targets:
        t.mkdir(parents=True, exist_ok=True)
        for name, doc in docs.items():
            (t / name).write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    segs = docs["labels.json"]["segments"]
    return {"capture": cdir.name, "dir": str(cdir / "moves"), "files": sorted(docs), "targets": [str(t) for t in targets],
            "time_to_audio": clock.time_to_audio, "segments": len(segs),
            "in_capture": sum(bool(s["in_capture"]) for s in segs),
            "capture_range": docs["labels.json"]["capture_range"]}


def validate_export(cdir: Path) -> list[str]:
    errs = []
    m = Path(cdir) / "moves"
    for name in FILES:
        p = m / name
        if not p.exists():
            errs.append(f"{p} missing")
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("time_base") != "capture" or d.get("frame") != "unity":
            errs.append(f"{p}: time_base/frame must be capture/unity")
        if d.get("version") != 1:
            errs.append(f"{p}: version != 1")
    return errs


def cmd_export(args: argparse.Namespace) -> int:
    caps = [Path(args.streaming) / c for c in args.capture] if args.capture else captures_for(args.take,
                                                                                             Path(args.streaming))
    if not caps:
        print(f"[moves] no capture in {args.streaming} has provenance.take == {args.take!r} (pass --capture)")
        return 1
    if not labels_path(args.take).exists():
        print(f"[moves] {labels_path(args.take)} missing")
        return 1
    rc = 0
    for c in caps:
        try:
            rep = export_moves(args.take, c, Path(args.streaming), Path(args.graph),
                               allow_stale=getattr(args, "allow_stale", False))
        except ExportError as e:
            print(f"[moves] {e}")
            return 1
        errs = validate_export(c)
        print(f"[moves] {rep['capture']}: {', '.join(rep['files'])} -> {rep['dir']} (time_to_audio "
              f"{rep['time_to_audio']:+.4f}; {rep['in_capture']}/{rep['segments']} segments inside the capture "
              f"{rep['capture_range']})" + (f"  INVALID {errs}" if errs else ""))
        rc |= bool(errs)
    return rc


def register(sub) -> None:
    p = sub.add_parser("export", help="Unity copies in StreamingAssets/<capture>/moves/ (capture time, Unity frame)")
    p.add_argument("take")
    p.add_argument("--capture", action="append", help="capture folder name (repeatable; default: by provenance.take)")
    p.add_argument("--streaming", default=str(UNITY_STREAMING))
    p.add_argument("--graph", default=str(GRAPH_PATH))
    p.add_argument("--allow-stale", action="store_true",
                   help="export even when path/fingerprint no longer match the labels (not recommended)")
    p.set_defaults(func=cmd_export)
