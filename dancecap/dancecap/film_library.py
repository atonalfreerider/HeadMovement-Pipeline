"""The film library of the Unity viewer: builds head-movement/Assets/StreamingAssets/library/library.json from a take's film files.

The viewer's start screen offers exactly the films listed in that file (docs/FILM_LIBRARY.md in the viewer repository): a card per
film with a poster, title, subtitle, length and a Play button, and the playback bar's chapter marks. The file is DATA (git-ignored,
it may hold names); this module is the one tool that writes it, so the class recap is added later with the same command:

    python -m dancecap.film_library build --demo                 # the finished demo film (preset "demo" of takes/library/films.toml)
    python -m dancecap.film_library build --recap                # the class recap; "Processing" in the viewer until its film exists
    python -m dancecap.film_library build --id x --title T --subtitle S --capture C --direction work/<take>/film/direction.json \
                                         [--thumbnail IMG] [--chapters-file PLAN.toml|.json] [--copy-film] [--hold]
    python -m dancecap.film_library chapters --direction work/<take>/film/direction.json [--plan PLAN]   # preview the chapters
    python -m dancecap.film_library list | check

`build` upserts the entry by id (the other entries stay, in their order), copies the thumbnail to library/thumbs/<id>.<ext> and
writes library.json atomically. Run it again whenever the film is rebuilt (the length and the chapter times are read from the
direction); the viewer re-reads the file by itself.

CHAPTERS (the marks on the viewer's scrubber, in FILM seconds), first source that exists wins:
  1. a plan (--chapters-file, or [[<preset>.chapters]] in the preset file): a list of {title, segment} (the chapter starts where the
     named direction segment starts; it survives a rebuild of the film) or {title, startS};
  2. chapters written by the direction itself: segments with a "chapter": "<title>" field, or a top-level "chapters" list of
     {title, segment | film_s};
  3. an automatic split: one chapter per run of segments that share the id prefix (the text before the first "_"), runs shorter than
     MIN_CHAPTER_S merged into the chapter before (the title is the prefix, capitalised; edit it with a plan).
Chapters are contiguous: each ends where the next begins, the last at the film's end.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import tomllib
from pathlib import Path

from .take import PACKAGE_ROOT, WORKSPACE

LIBRARY_VERSION = 1
VIEWER = WORKSPACE / "head-movement"
LIBRARY_DIR = VIEWER / "Assets" / "StreamingAssets" / "library"
PRESETS = PACKAGE_ROOT / "takes" / "library" / "films.toml"
MIN_CHAPTER_S = 8.0
MAX_TITLE = 48


class LibraryError(ValueError):
    """a problem with a film spec, a plan or a library file (the message says what to change)"""


# ----------------------------------------------------------------------------------------------- direction


def read_direction(path: Path) -> dict:
    """the film direction JSON (UTF-8)"""
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        raise LibraryError(f"no film direction at {path}") from None
    except json.JSONDecodeError as e:
        raise LibraryError(f"{path} is not valid JSON ({e})") from None


def segments(direction: dict) -> list[dict]:
    """segments with a film window, in film order: {id, f0, f1, chapter}"""
    out = []
    for i, s in enumerate(direction.get("segments") or []):
        film = s.get("film")
        if not (isinstance(film, list) and len(film) == 2):
            continue
        out.append({"id": str(s.get("id") or f"seg{i}"), "f0": float(film[0]), "f1": float(film[1]),
                    "chapter": s.get("chapter") if isinstance(s.get("chapter"), str) else None})
    out.sort(key=lambda s: s["f0"])
    return out


def film_duration(direction: dict) -> float:
    d = (direction.get("timeline") or {}).get("film_duration_s")
    if isinstance(d, (int, float)) and d > 0:
        return float(d)
    segs = segments(direction)
    return max((s["f1"] for s in segs), default=0.0)


# ----------------------------------------------------------------------------------------------- chapters


def _clean_title(title: str, where: str) -> str:
    t = " ".join(str(title).split())
    if not t:
        raise LibraryError(f"{where}: a chapter needs a title")
    return t[:MAX_TITLE]


def _from_marks(direction: dict, marks: list[dict], where: str) -> list[dict]:
    """[{title, segment | startS | film_s}] -> [{title, startS}] sorted"""
    by_id = {s["id"]: s for s in segments(direction)}
    out = []
    for m in marks:
        title = _clean_title(m.get("title", ""), where)
        if "segment" in m:
            seg = by_id.get(str(m["segment"]))
            if seg is None:
                raise LibraryError(f"{where}: chapter '{title}' names the segment '{m['segment']}', which the direction does not have")
            start = seg["f0"] + float(m.get("offset_s", 0.0) or 0.0)       # offset_s: the chapter starts this long after the segment does (the shot's lead-in)
        elif "startS" in m or "film_s" in m:
            start = float(m.get("startS", m.get("film_s")))
        else:
            raise LibraryError(f"{where}: chapter '{title}' needs a segment or a startS")
        out.append({"title": title, "startS": start})
    out.sort(key=lambda c: c["startS"])
    return out


def _auto_marks(direction: dict) -> list[dict]:
    """one mark per run of segments with the same id prefix; short runs join the chapter before"""
    segs = segments(direction)
    runs: list[dict] = []
    for s in segs:
        prefix = s["id"].split("_", 1)[0]
        if runs and runs[-1]["prefix"] == prefix:
            runs[-1]["end"] = s["f1"]
        else:
            runs.append({"prefix": prefix, "start": s["f0"], "end": s["f1"]})
    merged: list[dict] = []
    for r in runs:
        if merged and (r["end"] - r["start"]) < MIN_CHAPTER_S:
            merged[-1]["end"] = r["end"]
        else:
            merged.append(dict(r))
    # the first run may itself be short: it stays (a chapter starts at 0)
    return [{"title": m["prefix"].replace("-", " ").capitalize(), "startS": m["start"]} for m in merged]


def derive_chapters(direction: dict, plan: list[dict] | None = None, *, where: str = "the direction") -> list[dict]:
    """chapters [{title, startS, endS}] in film seconds (see the module docstring for the order of sources)"""
    dur = film_duration(direction)
    if plan:
        marks = _from_marks(direction, plan, "the chapter plan")
    else:
        own = [{"title": s["chapter"], "segment": s["id"]} for s in segments(direction) if s["chapter"]]
        top = direction.get("chapters")
        if own:
            marks = _from_marks(direction, own, where)
        elif isinstance(top, list) and top:
            marks = _from_marks(direction, top, where)
        else:
            marks = _auto_marks(direction)
    if not marks:
        return []
    chapters = []
    for i, m in enumerate(marks):
        end = marks[i + 1]["startS"] if i + 1 < len(marks) else dur
        if not (0.0 <= m["startS"] < max(dur, 1e-9)):
            raise LibraryError(f"chapter '{m['title']}' starts at {m['startS']:.2f} s, outside the film (0 .. {dur:.2f} s)")
        chapters.append({"title": m["title"], "startS": round(m["startS"], 3), "endS": round(end, 3)})
    return chapters


def read_plan(path: Path) -> list[dict]:
    """a chapter plan file: .toml with [[chapters]] / .json with a list or {"chapters": [...]}"""
    if path.suffix.lower() == ".toml":
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
        plan = data.get("chapters")
    else:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        plan = data if isinstance(data, list) else data.get("chapters")
    if not isinstance(plan, list):
        raise LibraryError(f"{path}: no chapters list")
    return plan


# ----------------------------------------------------------------------------------------------- entries and the library file


def _rel_to_workspace(p: Path) -> str:
    """a path as the library stores it: relative to the workspace when inside it (the viewer finds it from there), else absolute"""
    try:
        return p.resolve().relative_to(WORKSPACE.resolve()).as_posix()
    except ValueError:
        return p.resolve().as_posix()


def _abs(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else WORKSPACE / p


def copy_film(direction_path: Path, dest: Path) -> Path:
    """copy the film's own files (direction, narration timeline(s), the soundtrack mix) into dest/film/ and return the copied direction"""
    film = dest / "film"
    src = direction_path.parent
    film.mkdir(parents=True, exist_ok=True)
    shutil.copy2(direction_path, film / direction_path.name)
    for sub, pattern in (("narration", "timeline*.json"), ("audio", "*_mix.wav")):
        (film / sub).mkdir(exist_ok=True)
        for f in (src / sub).glob(pattern):
            target = film / sub / f.name
            if not target.exists() or target.stat().st_size != f.stat().st_size or target.stat().st_mtime < f.stat().st_mtime:
                shutil.copy2(f, target)
    return film / direction_path.name


def build_entry(spec: dict, *, library_dir: Path, copy: bool = False) -> tuple[dict, list[str]]:
    """one library entry from a spec {id, title, subtitle, capture, direction, thumbnail, chapters, hold}; returns (entry, notes)"""
    notes: list[str] = []
    fid = str(spec.get("id") or "").strip()
    if not fid or any(c in fid for c in '/\\:*?"<>| '):
        raise LibraryError(f"film id '{fid}': use letters, digits, - and _ only")
    if not spec.get("title"):
        raise LibraryError(f"film '{fid}' needs a title")
    direction_path = _abs(spec["direction"]) if spec.get("direction") else None
    if direction_path is None:
        raise LibraryError(f"film '{fid}' needs a direction path (where the film's direction.json is, or will be)")

    entry: dict = {"id": fid, "title": str(spec["title"]), "subtitle": str(spec.get("subtitle") or ""),
                   "capture": spec.get("capture"), "thumbnail": None, "direction": None, "durationS": None, "chapters": [],
                   "available": not bool(spec.get("hold"))}

    if direction_path.exists():
        direction = read_direction(direction_path)
        entry["capture"] = entry["capture"] or direction.get("capture")
        entry["durationS"] = round(film_duration(direction), 3)
        entry["chapters"] = derive_chapters(direction, spec.get("chapters"), where=direction_path.name)
        if not entry["chapters"]:
            notes.append(f"{fid}: no chapters (the plan is empty and the direction has no segments)")
        elif not 8 <= len(entry["chapters"]) <= 14:
            notes.append(f"{fid}: {len(entry['chapters'])} chapters (8-14 reads best on the scrubber)")
        if copy:
            direction_path = copy_film(direction_path, library_dir / "films" / fid)
            entry["direction"] = direction_path.relative_to(library_dir).as_posix()
        else:
            entry["direction"] = _rel_to_workspace(direction_path)
    else:
        # the film does not exist yet: the viewer shows the card as "Processing" and turns it playable when the files appear
        entry["direction"] = _rel_to_workspace(direction_path)
        notes.append(f"{fid}: the film is not built yet ({direction_path.name} missing): entry written as pending")
    if not entry["capture"]:
        raise LibraryError(f"film '{fid}' needs a capture folder name (--capture, or a direction that names one)")

    thumb = spec.get("thumbnail")
    if thumb:
        src = _abs(thumb)
        if src.exists():
            dest = library_dir / "thumbs" / f"{fid}{src.suffix.lower()}"
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists() or dest.stat().st_size != src.stat().st_size or dest.stat().st_mtime < src.stat().st_mtime:
                shutil.copy2(src, dest)
            entry["thumbnail"] = dest.relative_to(library_dir).as_posix()
        else:
            notes.append(f"{fid}: thumbnail {src.name} not found (the card shows a placeholder poster)")
    return entry, notes


def empty_library() -> dict:
    return {"version": LIBRARY_VERSION, "films": []}


def read_library(path: Path) -> dict:
    if not path.exists():
        return empty_library()
    with open(path, encoding="utf-8") as fh:
        lib = json.load(fh)
    if not isinstance(lib, dict) or not isinstance(lib.get("films"), list):
        raise LibraryError(f"{path} is not a film library (no \"films\" list)")
    return lib


def upsert(library: dict, entry: dict) -> dict:
    """replace the entry with the same id, else append (the order of the other films is kept)"""
    films = library.setdefault("films", [])
    for i, f in enumerate(films):
        if f.get("id") == entry["id"]:
            films[i] = entry
            break
    else:
        films.append(entry)
    library["version"] = LIBRARY_VERSION
    return library


def write_library(path: Path, library: dict) -> None:
    """UTF-8 without BOM, via a temporary file in the same folder (the viewer may be reading it)"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".library-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(library, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def check_library(library: dict) -> list[str]:
    """problems in a library dict (empty list = fine)"""
    problems: list[str] = []
    seen: set[str] = set()
    for f in library.get("films", []):
        fid = f.get("id")
        if not fid or fid in seen:
            problems.append(f"film id {fid!r} is empty or repeated")
        seen.add(fid)
        for key in ("title", "capture", "direction"):
            if not f.get(key):
                problems.append(f"{fid}: no {key}")
        ch = f.get("chapters") or []
        starts = [c.get("startS") for c in ch]
        if starts != sorted(starts):
            problems.append(f"{fid}: chapters are not in time order")
        dur = f.get("durationS")
        if dur is not None and any(s is not None and s >= dur for s in starts):
            problems.append(f"{fid}: a chapter starts after the film's end ({dur} s)")
    return problems


# ----------------------------------------------------------------------------------------------- presets


def read_presets(path: Path = PRESETS) -> dict:
    if not path.exists():
        raise LibraryError(f"no preset file {path}")
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def preset_spec(name: str, path: Path = PRESETS) -> dict:
    presets = read_presets(path)
    if name not in presets or not isinstance(presets[name], dict):
        raise LibraryError(f"no preset '{name}' in {path.name} (have: {', '.join(k for k, v in presets.items() if isinstance(v, dict))})")
    return dict(presets[name])


# ----------------------------------------------------------------------------------------------- command line


def _spec_from_args(a: argparse.Namespace) -> dict:
    spec: dict = {}
    preset = "demo" if a.demo else "recap" if a.recap else a.preset
    if preset:
        spec.update(preset_spec(preset, Path(a.presets) if a.presets else PRESETS))
    for key in ("id", "title", "subtitle", "capture", "direction", "thumbnail"):
        v = getattr(a, key)
        if v is not None:
            spec[key] = v
    if a.chapters_file:
        spec["chapters"] = read_plan(Path(a.chapters_file))
    if a.hold:
        spec["hold"] = True
    return spec


def cmd_build(a: argparse.Namespace) -> int:
    lib_dir = Path(a.library_dir) if a.library_dir else LIBRARY_DIR
    spec = _spec_from_args(a)
    entry, notes = build_entry(spec, library_dir=lib_dir, copy=a.copy_film)
    path = lib_dir / "library.json"
    library = upsert(read_library(path), entry)
    problems = check_library(library)
    for n in notes:
        print(f"note: {n}")
    for p in problems:
        print(f"PROBLEM: {p}")
    if problems:
        return 1
    if a.dry_run:
        print(json.dumps(entry, ensure_ascii=False, indent=2))
        return 0
    write_library(path, library)
    ch = entry["chapters"]
    print(f"wrote {path} : {len(library['films'])} film(s); {entry['id']}: {entry['durationS']} s, {len(ch)} chapters, "
          f"capture {entry['capture']}, thumbnail {entry['thumbnail']}")
    return 0


def cmd_chapters(a: argparse.Namespace) -> int:
    direction = read_direction(_abs(a.direction))
    plan = read_plan(Path(a.plan)) if a.plan else None
    dur = film_duration(direction)
    print(f"film {dur:.2f} s")
    for i, c in enumerate(derive_chapters(direction, plan)):
        print(f"{i + 1:2d}  {c['startS']:8.2f} .. {c['endS']:8.2f}  {c['title']}")
    return 0


def cmd_list(a: argparse.Namespace) -> int:
    lib_dir = Path(a.library_dir) if a.library_dir else LIBRARY_DIR
    lib = read_library(lib_dir / "library.json")
    for f in lib["films"]:
        print(f"{f['id']:12s} {str(f.get('durationS')):>9s} s  {len(f.get('chapters') or []):2d} chapters  capture {f.get('capture')}  {f.get('title')}")
    return 0


def cmd_check(a: argparse.Namespace) -> int:
    lib_dir = Path(a.library_dir) if a.library_dir else LIBRARY_DIR
    problems = check_library(read_library(lib_dir / "library.json"))
    for p in problems:
        print(f"PROBLEM: {p}")
    print("library ok" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.film_library", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="add or update one film of the library")
    b.add_argument("--demo", action="store_true", help="the preset 'demo' of takes/library/films.toml")
    b.add_argument("--recap", action="store_true", help="the preset 'recap'")
    b.add_argument("--preset", help="any other preset name of the preset file")
    b.add_argument("--presets", help="a preset file other than takes/library/films.toml")
    for key in ("id", "title", "subtitle", "capture", "direction", "thumbnail"):
        b.add_argument(f"--{key}")
    b.add_argument("--chapters-file", help="a chapter plan (.toml with [[chapters]], or .json)")
    b.add_argument("--copy-film", action="store_true", help="copy direction + narration + mix into library/films/<id>/ instead of referencing the take's folder")
    b.add_argument("--hold", action="store_true", help="available=false: shown as Processing even when the files exist")
    b.add_argument("--library-dir", help="default: head-movement/Assets/StreamingAssets/library")
    b.add_argument("--dry-run", action="store_true")
    b.set_defaults(fn=cmd_build)

    c = sub.add_parser("chapters", help="print the chapters a direction (and a plan) give")
    c.add_argument("--direction", required=True)
    c.add_argument("--plan")
    c.set_defaults(fn=cmd_chapters)

    for name, fn, help_ in (("list", cmd_list, "list the library's films"), ("check", cmd_check, "validate library.json")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("--library-dir")
        p.set_defaults(fn=fn)

    a = ap.parse_args(argv)
    try:
        return a.fn(a)
    except LibraryError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
