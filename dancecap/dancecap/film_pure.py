"""A PURE-DANCE film of a capture: the whole dance once, at 1x, with its own song and nothing else (owner 2026-11-15: "let's not have any
narration ... just the playback of the pure dance itself, with no programmed slowdowns or direction to steps").

The demo lesson's library film used to be the narrated overview (35 segments with slow motion, replays, call-outs, a title and an end card,
the narration baked into its mix). This module writes the smallest direction the viewer's film director plays instead: ONE segment over the
whole danced span at speed 1.0, film 0 = audio 0 (dance time = film time - the capture's `time_to_audio`), the capture's own audio.wav as
the film's mix (`audio/<name>_mix.wav`, the song only), and the chapter marks of the preset. Same JSON shape as `film_heat` writes.

    python -m dancecap.film_pure build --preset demo [--out DIR] [--dry-run]     # the preset's [<name>.pure] block of takes/library/films.toml
    python -m dancecap.film_library build --demo                                  # then the library entry (reads the direction just written)

THE FILM
  * One segment `dance`: dance [-time_to_audio, film_end - time_to_audio], film [0, film_end], speed 1.0, audio locked to the song.
    film_end = the capture's last frame time + time_to_audio (the dance's last pose; the song's tail after it is cut).
  * No annotations, no title or end card, no flow, no narration; the direction's name is new (`[<name>.pure] name`) so no narration
    timeline of the narrated film matches it.
  * Layers: the floor ON (the footprints of the physics view need it), everything that directs the eye OFF (moves, graph inset, floor craft,
    counterbalance, traces, neck, hud, beat counter, rhythm score). No `skeleton_mode`: the viewer's own toggle (VR wrist: physics / rhythm)
    decides it.
  * Camera (the desktop player; VR ignores it): one slow orbit in front of the couple.
  * Chapters: the preset's [[<name>.chapters]] ({title, segment = "dance", offset_s} = film seconds), also written into the direction.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from .take import WORKSPACE

VIEWER = WORKSPACE / "head-movement"
STREAMING = VIEWER / "Assets" / "StreamingAssets"
SEGMENT_ID = "dance"

# the layers of the one segment: the floor (footprints) and nothing that points at a step
LAYERS = {"floor": True, "floorcraft": "off", "counterbalance": False, "traces": False, "neck": False, "moves": False, "graph_inset": False,
          "hud": False, "beat_counter": False, "rhythm_score": False}
# one slow orbit for the desktop player (VR uses the viewer's own head)
CAMERA = {"mode": "orbit", "radius_m": 4.2, "height_m": 2.0, "look_height_m": 1.0, "period_s": -120}


class PureError(ValueError):
    """a problem with the preset or the capture (the message says what to change)"""


def _abs(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else WORKSPACE / p


def read_json(path: Path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        raise PureError(f"no file {path}") from None
    except json.JSONDecodeError as e:
        raise PureError(f"{path} is not valid JSON ({e})") from None


def _wav_seconds(path: Path) -> float:
    import wave
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def chapter_marks(chapters: list[dict], film_end: float) -> list[dict]:
    """[{title, film_s}] sorted, from the preset's chapters ({title, segment = "dance", offset_s} or {title, startS | film_s})"""
    out = []
    for c in chapters or []:
        title = " ".join(str(c.get("title") or "").split())
        if not title:
            raise PureError("a chapter needs a title")
        if "segment" in c:
            if str(c["segment"]) != SEGMENT_ID:
                raise PureError(f"chapter '{title}' names the segment '{c['segment']}': a pure-dance film has one segment, '{SEGMENT_ID}'")
            t = float(c.get("offset_s", 0.0) or 0.0)
        elif "startS" in c or "film_s" in c:
            t = float(c.get("startS", c.get("film_s")))
        else:
            raise PureError(f"chapter '{title}' needs segment + offset_s or startS")
        if not 0.0 <= t < film_end:
            raise PureError(f"chapter '{title}' starts at {t:.3f} s, outside the film (0 .. {film_end:.3f} s)")
        out.append({"title": title, "film_s": round(t, 3)})
    out.sort(key=lambda c: c["film_s"])
    return out


def build_direction(*, name: str, title: str, capture: str, take: str, dance_range: list[float], time_to_audio: float, film_end: float,
                    chapters: list[dict] | None = None) -> dict:
    """the direction dict (JSON-serialisable): one segment, film 0 = audio 0, dance = film - time_to_audio"""
    marks = chapter_marks(chapters or [], film_end)
    seg = {"id": SEGMENT_ID, "dance": [round(-time_to_audio, 7), round(film_end - time_to_audio, 7)], "film": [0.0, round(film_end, 3)],
           "speed": 1.0, "camera": dict(CAMERA), "view_state": "orbit", "layers": dict(LAYERS), "annotations": [], "audio": "locked", "status": "ok"}
    return {
        "version": 1, "kind": "direction", "name": name, "title": title, "capture": capture, "take": take,
        "time_base": f"FILM time = AUDIO time (speed 1.0, no cuts); DANCE time = film time - {time_to_audio} s (the capture's time_to_audio): "
                     "the soundtrack is the capture's own audio.",
        "dance_range": [round(dance_range[0], 3), round(dance_range[1], 3)],
        "created": "generated by dancecap.film_pure (python -m dancecap.film_pure build): edit the rules there, not this file",
        "audio": {"film_mix": f"audio/{name}_mix.wav (a copy of the capture's audio.wav: film time = audio time)"},
        "segments": [seg],
        "chapters": [{"title": m["title"], "segment": SEGMENT_ID, "offset_s": m["film_s"], "film_s": m["film_s"]} for m in marks],
        "timeline": {"film_duration_s": round(film_end, 3), "dance_span_s": round(film_end, 3), "inserts": [], "narration_lines": 0},
    }


def pure_spec(preset: dict) -> dict:
    pure = preset.get("pure")
    if not isinstance(pure, dict):
        raise PureError(f"preset '{preset.get('id')}' has no [pure] block")
    return pure


def build_from_preset(preset: dict, *, out_dir: Path | None = None, dry_run: bool = False) -> tuple[Path, dict, list[str]]:
    """write the pure-dance direction and its mix for a preset of films.toml; returns (direction path, direction, notes)"""
    notes: list[str] = []
    pure = pure_spec(preset)
    capture = str(preset.get("capture") or "")
    if not capture:
        raise PureError("the preset names no capture")
    name = str(pure.get("name") or "")
    if not name:
        raise PureError("the [pure] block needs a name (a new one: no narration timeline may match it)")
    cap_dir = STREAMING / capture
    cap = read_json(cap_dir / "capture.json")
    times = read_json(cap_dir / (cap.get("times") or "times.json"))
    if not isinstance(times, list) or not times:
        raise PureError(f"{cap_dir / 'times.json'} holds no frame times")
    first, last = float(times[0]), float(times[-1])
    tta = cap.get("time_to_audio")
    if not isinstance(tta, (int, float)):
        raise PureError(f"{cap_dir / 'capture.json'} has no time_to_audio (dance time -> audio time)")
    tta = float(tta)
    audio = cap_dir / (cap.get("audio") or "audio.wav")
    if not audio.exists():
        raise PureError(f"the capture has no soundtrack at {audio}")
    audio_s = _wav_seconds(audio)
    film_end = round(float(pure.get("film_end_s") or (last + tta)), 3)
    if film_end > audio_s + 1e-3:
        raise PureError(f"film end {film_end} s is past the soundtrack ({audio_s:.2f} s)")
    direction_path = _abs(pure.get("direction") or preset["direction"])
    if out_dir is not None:
        direction_path = out_dir / direction_path.name
    direction = build_direction(name=name, title=str(preset.get("title") or name), capture=capture, take=str(pure.get("take") or ""),
                                dance_range=[first, last], time_to_audio=tta, film_end=film_end, chapters=preset.get("chapters"))
    notes.append(f"{name}: 1 segment, {len(direction['chapters'])} chapters, film {film_end:.3f} s (dance {first:.3f}..{last:.3f} s, "
                 f"time_to_audio {tta})")
    if dry_run:
        return direction_path, direction, notes
    direction_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = direction_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(direction, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    tmp.replace(direction_path)
    mix = direction_path.parent / "audio" / f"{name}_mix.wav"
    mix.parent.mkdir(parents=True, exist_ok=True)
    if not mix.exists() or mix.stat().st_size != audio.stat().st_size or mix.stat().st_mtime < audio.stat().st_mtime:
        shutil.copy2(audio, mix)
        notes.append(f"copied the capture's soundtrack to {mix}")
    os.utime(mix, None)   # the recorder refuses a mix older than the direction
    return direction_path, direction, notes


def main(argv: list[str] | None = None) -> int:
    from .film_library import PRESETS, read_presets  # local: keep this module importable without the library

    ap = argparse.ArgumentParser(prog="python -m dancecap.film_pure", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="write the pure-dance direction and the mix of a preset")
    b.add_argument("--preset", required=True)
    b.add_argument("--presets", help="a preset file other than takes/library/films.toml")
    b.add_argument("--out", help="write the direction (and audio/) here instead of the preset's direction path")
    b.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    try:
        presets = read_presets(Path(a.presets) if a.presets else PRESETS)
        if a.preset not in presets:
            raise PureError(f"no preset '{a.preset}'")
        path, direction, notes = build_from_preset(presets[a.preset], out_dir=Path(a.out) if a.out else None, dry_run=a.dry_run)
    except PureError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    for n in notes:
        print(n)
    for c in direction["chapters"]:
        print(f"  {c['film_s']:8.3f} s  {c['title']}")
    print(("would write " if a.dry_run else "wrote ") + str(path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
