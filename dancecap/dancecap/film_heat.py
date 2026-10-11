"""An AUTO-DIRECTED film for a captured heat (or any take whose only soundtrack is its own): the viewer's library plays it like a
directed film (playback bar, chapters, exact seeking), without narration, captions or call-outs.

The viewer's film director plays any `direction.json` (docs/FILM_LIBRARY.md); the class films are written by `film_recap`, whose
camera and speech are driven by a transcript. A competition heat has neither: this module writes the smallest direction that is still
a good film of it. Film time = dance time (reference seconds, speed 1.0, no cuts), so the capture's own soundtrack IS the film's mix
(`film/audio/<name>_mix.wav` is a copy of the capture's audio) and every layer the capture carries (beat grid, timing, floor craft)
stays in step with the sound however the viewer seeks.

    python -m dancecap.film_heat build --preset carlos_aline [--out DIR] [--dry-run]     # the preset's [<name>.heat] block of films.toml
    python -m dancecap.film_library build --carlos-aline                                  # the same, then the library entry

THE FILM
  * Chapters = the song structure (`song/segments.json` "timeline"): the entrance, each song, the DJ mix, the bow. One chapter per
    timeline state; titles from the preset ([<name>.heat.titles], by state). Every chapter starts where a segment starts.
  * Camera: a slow 3D orbit that stays on the audience side of the room (the room model has a back wall; the sides are not modelled),
    `orbit` with `azimuth_deg` + `drift_deg_s` (an absolute azimuth that drifts: the camera glides across an arc in front of the wall, a
    new arc every shot, alternating direction), `fit_span` (both dancers always inside the frame however far apart they stand) and a close
    radius (the couple fills more than half of the frame height). Every `cycle_s` the camera goes overhead for `overhead_s` seconds
    (`overhead_floorcraft`, couple framing with `fill`: the owner's "for overhead views get closer with the camera") with the
    current T and the footprints of the floor craft. No hand close-ups, no phone point-of-view shots.
  * Layers: the dancers as avatars (`avatar_opacity`, one value at the top of the direction) over the room (`room`; off = the dark floor grid
    only), the skeleton lines at `skeleton_lines` strength, the zouk beat counter while a song is audible and has a beat grid (never in the
    no-beat intervals), the live rhythm score HUD + final score card when `rhythm_score` is on (docs/RHYTHM.md); nothing else.
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

# camera defaults (all overridable in the preset's [<name>.heat] block)
DEFAULTS = {
    "radius_m": 2.7, "radius_vertical_m": 2.15, "height_m": 1.9, "look_height_m": 0.95,
    "front_azimuth_deg": -90.0,       # Unity azimuth atan2(z, x) of the audience side: the cameras stand at -z, the room's wall at +z
    "arc_deg": 70.0,                  # how far an orbit shot glides across the arc in front of the wall
    "arc_centres_deg": [0.0, 22.0, -22.0, 12.0, -12.0],   # the arc's centre offset from the front, per shot (cycled)
    "cycle_s": 28.0, "overhead_s": 6.0, "first_overhead_s": 22.0, "no_overhead_before_s": 16.0, "no_overhead_after_s": 188.0,
    "overhead_fill": 0.8, "overhead_min_fill": 0.55, "overhead_across_m": 2.2,
    "min_shot_s": 4.0, "blend_s": 1.2,
    # looks (the layers of every segment): avatar_opacity = the dancers' OPACITY (1 = solid; the viewer's own default is 0.35), room = the capture's room mesh,
    # skeleton_lines = the skeleton strokes' width factor (0 = hidden, 1 = normal), rhythm_score = the live rhythm score HUD + the final score card
    # (needs the capture's rhythm layer: python -m dancecap.rhythm run <take>, then the export)
    "avatar_opacity": 0.95, "room": True, "skeleton_lines": 0.0, "rhythm_score": False,
}

def layer_set(cfg: dict, overhead: bool) -> dict:
    """the layers of one segment from the film's looks (see DEFAULTS). The avatars' opacity is written once at the top of the direction
    (`avatar_opacity`), which the viewer uses for every segment that has no "avatars" layer of its own (edit that one number to change the film's look)."""
    lay = {"skeleton_mode": "rhythm", "room": bool(cfg["room"]), "skeleton_lines": float(cfg["skeleton_lines"]), "floorcraft": "current_t" if overhead else "off",
           "counterbalance": False, "moves": False, "graph_inset": False, "hud": False}
    if overhead:
        lay.update({"footprints": "recent", "grid": False, "t_width_m": 0.05, "t_brightness": 1.0})
    else:
        lay.update({"floor": True, "traces": False, "neck": False})
    if overhead and not cfg["room"]:
        lay["grid"] = True                      # no room: the dark floor grid is what the dancers float over, from above too
    if cfg.get("rhythm_score"):
        lay["rhythm_score"] = True              # the live score HUD + the final card
        lay["rhythm_rings"] = True              # the touchdown rings, coloured by the step's class
    return lay


DEFAULT_TITLES = {"entrance": "Entrance and announcement", "none": "Entrance and announcement", "mix": "DJ mix", "end": "Bow and applause"}


class HeatError(ValueError):
    """a problem with the preset, the song structure or the capture (the message says what to change)"""


def _abs(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else WORKSPACE / p


def read_json(path: Path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        raise HeatError(f"no file {path}") from None
    except json.JSONDecodeError as e:
        raise HeatError(f"{path} is not valid JSON ({e})") from None


# ----------------------------------------------------------------------------------------------- structure


def chapters_from_songs(songs: dict, titles: dict[str, str], film_end: float, song_names: dict[str, str] | None = None) -> list[dict]:
    """[{title, start, end}] from song/segments.json: one chapter per timeline state, in order. A state "none" before the first song is the
    entrance, after the last the bow; a song state is titled by the preset (key = the state: "kojo", "guetta", "mix"). Chapters are clipped to
    [0, film_end] and a chapter shorter than 1 s is merged into the one before."""
    timeline = songs.get("timeline")
    if not isinstance(timeline, list) or not timeline:
        raise HeatError("the song structure has no timeline (song/segments.json)")
    rows = sorted(({"from": float(t["from_s"]), "to": float(t["to_s"]), "state": str(t.get("state", ""))} for t in timeline), key=lambda r: r["from"])
    out: list[dict] = []
    for i, r in enumerate(rows):
        state = r["state"]
        if state == "none":
            state = "entrance" if i == 0 else "end"
        title = (titles.get(state) or (song_names or {}).get(state) or DEFAULT_TITLES.get(state) or state.capitalize()).strip()
        start, end = max(0.0, r["from"]), min(film_end, r["to"])
        if i == len(rows) - 1:
            end = film_end          # the last chapter runs to the film's end (the applause outlasts the danced span)
        if end - start < 1.0:
            if out:
                out[-1]["end"] = max(out[-1]["end"], end)
            continue
        out.append({"title": title, "start": round(start, 3), "end": round(end, 3), "state": r["state"]})
    if not out:
        raise HeatError("the song timeline leaves no chapter inside the film")
    out[0]["start"] = 0.0
    for a, b in zip(out, out[1:]):
        a["end"] = b["start"]
    return out


def shot_plan(film_end: float, cfg: dict) -> list[dict]:
    """camera shots [{kind, start, end}] covering [0, film_end]: orbit shots with an overhead of `overhead_s` every `cycle_s`"""
    ov = []
    t = float(cfg["first_overhead_s"])
    while t + cfg["overhead_s"] <= min(film_end, cfg["no_overhead_after_s"]):
        if t >= cfg["no_overhead_before_s"]:
            ov.append((t, t + cfg["overhead_s"]))
        t += cfg["cycle_s"]
    shots: list[dict] = []
    cur = 0.0
    for a, b in ov:
        if a - cur >= cfg["min_shot_s"]:
            shots.append({"kind": "orbit", "start": cur, "end": a})
        shots.append({"kind": "overhead", "start": a, "end": b})
        cur = b
    if film_end - cur > 1e-6:
        if shots and film_end - cur < cfg["min_shot_s"] and shots[-1]["kind"] == "orbit":
            shots[-1]["end"] = film_end
        else:
            shots.append({"kind": "orbit", "start": cur, "end": film_end})
    # a first orbit shorter than min_shot_s cannot occur (first_overhead_s >= no_overhead_before_s)
    return shots


def orbit_camera(index: int, length_s: float, cfg: dict) -> dict:
    """the camera block of the `index`-th orbit shot: an arc of `arc_deg` across the audience side, glided in `length_s`; direction alternates"""
    centres = list(cfg["arc_centres_deg"]) or [0.0]
    centre = cfg["front_azimuth_deg"] + centres[index % len(centres)]
    sign = 1.0 if index % 2 == 0 else -1.0
    arc = float(cfg["arc_deg"])
    # glide speed: the arc over the shot, but never faster than 4 deg/s nor slower than 0.8 deg/s (a long shot drifts, a short one does not rush)
    speed = max(0.8, min(4.0, arc / max(length_s, 1.0)))
    span = speed * length_s
    start = centre - sign * span / 2.0
    return {"mode": "orbit", "radius_m": cfg["radius_m"], "radius_vertical_m": cfg["radius_vertical_m"], "height_m": cfg["height_m"], "look_height_m": cfg["look_height_m"],
            "azimuth_deg": round(start, 2), "drift_deg_s": round(sign * speed, 3), "fit_span": True, "avoid_inline": True,
            "blend_s": cfg["blend_s"]}


def overhead_camera(cfg: dict) -> dict:
    return {"mode": "overhead_floorcraft", "framing": "couple", "across_m": cfg["overhead_across_m"], "fill": cfg["overhead_fill"],
            "min_fill": cfg["overhead_min_fill"], "blend_s": 0.7}


def build_direction(*, name: str, title: str, capture: str, take: str, songs: dict, dance_range: list[float], film_end: float,
                    titles: dict[str, str] | None = None, cfg: dict | None = None, beat_start_s: float | None = None,
                    beat_end_s: float | None = None) -> dict:
    """the direction dict (JSON-serialisable). Segments are cut at every chapter start and every camera change; consecutive segments with
    the same camera block are one shot for the viewer."""
    conf = dict(DEFAULTS)
    conf.update(cfg or {})
    chapters = chapters_from_songs(songs, titles or {}, film_end, None)
    shots = shot_plan(film_end, conf)
    no_beat = [(float(a), float(b)) for a, b in (songs.get("no_beat") or [])]

    def beat_on(t0: float, t1: float) -> bool:
        mid = 0.5 * (t0 + t1)
        if beat_start_s is not None and mid < beat_start_s:
            return False
        if beat_end_s is not None and mid > beat_end_s:
            return False
        return not any(a <= mid <= b for a, b in no_beat)

    cuts = sorted({round(x, 3) for x in [0.0, film_end] + [c["start"] for c in chapters] + [s["start"] for s in shots]
                   + [no_b for a, b in no_beat for no_b in (a, b) if 0.0 < no_b < film_end]})
    camera_of: dict[int, dict] = {}
    orbit_i = 0
    for i, sh in enumerate(shots):
        if sh["kind"] == "orbit":
            camera_of[i] = orbit_camera(orbit_i, sh["end"] - sh["start"], conf)
            orbit_i += 1
        else:
            camera_of[i] = overhead_camera(conf)

    segs = []
    chapter_first: dict[int, str] = {}
    for k, (a, b) in enumerate(zip(cuts, cuts[1:])):
        if b - a < 1e-3:
            continue
        mid = 0.5 * (a + b)
        ci = next(i for i, c in enumerate(chapters) if c["start"] - 1e-6 <= mid < c["end"] + 1e-6)
        si = next(i for i, s in enumerate(shots) if s["start"] - 1e-6 <= mid < s["end"] + 1e-6)
        kind = shots[si]["kind"]
        layers = layer_set(conf, kind == "overhead")
        layers["beat_counter"] = beat_on(a, b)
        seg = {"id": f"c{ci + 1:02d}_{kind}_{k:02d}", "dance": [a, b], "film": [a, b], "speed": 1.0, "camera": dict(camera_of[si]),
               "view_state": "overhead" if kind == "overhead" else "orbit", "layers": layers, "annotations": [], "audio": "locked",
               "status": "ok"}
        if abs(a - chapters[ci]["start"]) < 1e-6:
            seg["chapter"] = chapters[ci]["title"]
            chapter_first[ci] = seg["id"]
        segs.append(seg)
    segs[0]["film"][0] = 0.0
    segs[-1]["film"][1] = film_end
    segs[-1]["dance"][1] = film_end

    return {
        "version": 1, "kind": "direction", "name": name, "title": title, "capture": capture, "take": take,
        "time_base": "FILM time = DANCE time = reference seconds (speed 1.0, no cuts): the soundtrack is the capture's own audio.",
        "dance_range": [round(dance_range[0], 3), round(dance_range[1], 3)],
        "created": "generated by dancecap.film_heat (python -m dancecap.film_heat build): edit the rules there, not this file",
        "audio": {"film_mix": f"audio/{name}_mix.wav (a copy of the capture's audio.wav: film time = audio time)"},
        "avatar_opacity": float(conf["avatar_opacity"]),
        "segments": segs,
        "chapters": [{"title": c["title"], "segment": chapter_first[i], "film_s": c["start"]} for i, c in enumerate(chapters)],
        "timeline": {"film_duration_s": round(film_end, 3), "dance_span_s": round(film_end, 3), "inserts": [], "narration_lines": 0},
        "heat_film": {"camera": {k: conf[k] for k in ("radius_m", "height_m", "arc_deg", "cycle_s", "overhead_s", "overhead_fill")},
                      "shots": [{"kind": s["kind"], "film": [round(s["start"], 3), round(s["end"], 3)]} for s in shots]},
    }


# ----------------------------------------------------------------------------------------------- preset -> files


def heat_spec(preset: dict) -> dict:
    """the preset's [<name>.heat] block, paths resolved"""
    heat = preset.get("heat")
    if not isinstance(heat, dict):
        raise HeatError(f"preset '{preset.get('id')}' has no [heat] block")
    return heat


def build_from_preset(preset: dict, *, out_dir: Path | None = None, dry_run: bool = False) -> tuple[Path, dict, list[str]]:
    """write direction.json and the mix for a preset of films.toml; returns (direction path, direction, notes)"""
    notes: list[str] = []
    heat = heat_spec(preset)
    capture = str(preset.get("capture") or "")
    if not capture:
        raise HeatError("the preset names no capture")
    cap_dir = STREAMING / capture
    cap = read_json(cap_dir / "capture.json")
    times = read_json(cap_dir / (cap.get("times") or "times.json"))
    if not isinstance(times, list) or not times:
        raise HeatError(f"{cap_dir / 'times.json'} holds no frame times")
    first, last = float(times[0]), float(times[-1])
    songs = read_json(_abs(heat["songs"]))
    audio = cap_dir / (cap.get("audio") or "audio.wav")
    if not audio.exists():
        raise HeatError(f"the capture has no soundtrack at {audio}")
    audio_s = _wav_seconds(audio)
    # the film ends with the danced span (the last pose); the soundtrack outlasts it (applause) and is cut there
    film_end = round(float(heat.get("film_end_s") or min(last + 0.1, audio_s)), 3)
    if film_end > audio_s + 1e-3:
        raise HeatError(f"film end {film_end} s is past the soundtrack ({audio_s:.2f} s)")
    beats = songs.get("timeline") or []
    song_rows = [t for t in beats if t.get("songs")]
    beat_lo = float(song_rows[0]["from_s"]) if song_rows else None
    beat_hi = float(song_rows[-1]["to_s"]) if song_rows else None
    direction_path = _abs(preset["direction"])
    if out_dir is not None:
        direction_path = out_dir / "direction.json"
    name = str(heat.get("name") or preset.get("id") or "heat")
    cfg = {k: v for k, v in heat.items() if k in DEFAULTS}
    direction = build_direction(name=name, title=str(preset.get("title") or name), capture=capture, take=str(heat.get("take") or ""), songs=songs,
                                dance_range=[first, last], film_end=film_end, titles=dict(heat.get("titles") or {}), cfg=cfg,
                                beat_start_s=beat_lo, beat_end_s=beat_hi)
    notes.append(f"{name}: {len(direction['segments'])} segments, {len(direction['chapters'])} chapters, film {film_end:.2f} s (danced span {first:.2f}-{last:.2f} s)")
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
    # the recorder refuses a mix older than the direction: the mix is a copy of the capture's audio, so it is as current as the direction just written
    os.utime(mix, None)
    return direction_path, direction, notes


def _wav_seconds(path: Path) -> float:
    import wave
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


# ----------------------------------------------------------------------------------------------- command line


def main(argv: list[str] | None = None) -> int:
    from .film_library import read_presets, PRESETS  # local: film_library imports this module

    ap = argparse.ArgumentParser(prog="python -m dancecap.film_heat", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="write the direction and the mix of a preset")
    b.add_argument("--preset", required=True)
    b.add_argument("--presets", help="a preset file other than takes/library/films.toml")
    b.add_argument("--out", help="write direction.json (and audio/) here instead of the preset's direction path")
    b.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    try:
        presets = read_presets(Path(a.presets) if a.presets else PRESETS)
        if a.preset not in presets:
            raise HeatError(f"no preset '{a.preset}'")
        path, direction, notes = build_from_preset(presets[a.preset], out_dir=Path(a.out) if a.out else None, dry_run=a.dry_run)
    except HeatError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    for n in notes:
        print(n)
    for c in direction["chapters"]:
        print(f"  {c['film_s']:8.2f} s  {c['title']}")
    print(("would write " if a.dry_run else "wrote ") + str(path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
