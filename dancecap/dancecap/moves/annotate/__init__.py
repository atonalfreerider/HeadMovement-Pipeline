"""Narration annotation tool: label the dance moves of a video by watching it and saying the move names.

    python -m dancecap.moves annotate --take larissa_kadu            reference time, any camera of the take
    python -m dancecap.moves annotate --video <any.mp4>               video time (a dance not processed yet)
    python -m dancecap.moves annotate transcribe --take T --audio narration.wav [--log log.json | --start S --rate R]
    python -m dancecap.moves annotate asr-check                       which speech recognisers work offline

Local only: the server binds 127.0.0.1, refuses other Host headers, and never sends video or voice anywhere. The
optional TypeSafe judgment for ambiguous phrases (--typesafe or the UI toggle; off by default) sends only the
transcribed words and candidate move names.

USER GUIDE
  Start      `python -m dancecap.moves annotate --take larissa_kadu` (from dancecap/, with the .venv-nopo4d python);
             the browser opens http://127.0.0.1:8765/. Pick a camera (any of them - times are converted to the take's
             reference clock), a speed (0.25 - 1x; 0.5x is comfortable for narrating), and use headphones.
  Narrate    press R (or the red button): the microphone records while the video plays. Say each move's name as it
             starts ("corredor", "viradinha", "sarrada" ...). Pause (Space), seek, change speed or camera at will -
             the player is logged against the recording clock. Tap M at a transition if you like (taps beat the
             beat grid for the start). Say "scratch that" / "undo" to drop the last move, "again" to repeat it.
             Press R again: the narration is transcribed locally (faster-whisper when installed, else the Windows
             recogniser), phrases are matched to the dance graph's moves (names, Portuguese names, aliases, your
             learned spellings), starts are moved back by the reaction lag (0.4 s default) and snapped to the beat
             grid (half measures by default) or to proposal boundaries. Review the list, then Apply.
  Correct    narrated segments are hatched (provenance 'narration') until you confirm them (Enter -> 'human').
             Drag segment edges (snap to eighths; hold Alt for free), M / S split at the playhead, J merges with the
             next, Delete removes, E / the picker changes the move (search names, aliases, English names). Unknown
             phrases are red ("unknown: ...") - pick a move, teach the phrase as an alias, or create a proposed move
             (written to taxonomy/additions.toml only when you click it).
  Keys       Space play/pause, Left/Right one eighth (Shift: one measure), [ ] previous/next segment, 1-4 speed,
             L loop the segment, Ctrl+Z / Ctrl+Y undo/redo, Ctrl+S save, ? help.
  Save       Ctrl+S writes work/<take>/moves/labels.json (validated against the graph; the previous file is backed
             up under moves/narration/backups/). Unknown phrases are kept in labels.json 'unresolved'.
  Export     'Export to Unity' runs `python -m dancecap.moves run <take>` (graph path + fingerprint + the Unity copies
             in StreamingAssets/<capture>/moves/).
Files: see session.py. Design: docs/MOVES_RESEARCH.md section 7 (narration to timeline).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _session(args):
    from .session import Session
    kw = dict(annotator=args.annotator, beats=Path(args.beats) if getattr(args, "beats", None) else None,
              settings={k: v for k, v in (("lag", args.lag), ("snap", args.snap), ("engine", args.engine),
                                          ("typesafe", True if args.typesafe else None)) if v is not None})
    if args.video:
        if args.take and not args.video_time:
            from .session import find_take_of
            hit = find_take_of(Path(args.video).resolve())
            if hit and hit[0] == args.take:
                return Session.for_take(args.take, camera=hit[1], **kw)
        return Session.for_video(args.video, name=args.name or (args.take if args.take else None),
                                 video_time=args.video_time, **kw)
    if args.take:
        return Session.for_take(args.take, camera=args.camera, **kw)
    raise SystemExit("annotate needs --take <take> or --video <file.mp4>")


def cmd_annotate(args: argparse.Namespace) -> int:
    if args.action == "asr-check":
        from . import asr
        st = asr.engines(refresh=True)
        print(json.dumps(st, indent=1))
        if not st["faster-whisper"].get("available"):
            print(asr.setup_help())
        return 0
    s = _session(args)
    if args.action == "transcribe":
        return cmd_transcribe(s, args)
    from .server import serve
    serve(s, port=args.port, open_browser=not args.no_browser)
    return 0


def cmd_transcribe(s, args) -> int:
    """Offline narration processing (a wav + its player log, or a constant-rate playback from --start)."""
    import shutil

    from .pipeline import NarrationSettings, process
    if not args.audio:
        raise SystemExit("transcribe needs --audio <narration.wav>")
    ndir = s.new_narration_dir()
    src = Path(args.audio)
    shutil.copyfile(src, ndir / f"audio{src.suffix.lower() or '.wav'}")
    if args.log:
        log_doc = json.loads(Path(args.log).read_text(encoding="utf-8"))
    else:
        cam = s.camera(args.camera)
        v0 = cam.to_video(args.start) if args.start is not None else 0.0
        log_doc = {"clock": "constant", "entries": [{"a": 0.0, "v": v0, "cam": cam.id, "rate": args.rate,
                                                     "paused": False, "ev": "start"}]}
    (ndir / "log.json").write_text(json.dumps(log_doc, indent=1), encoding="utf-8")
    st = NarrationSettings.from_dict({**s.settings, "annotator": s.annotator})
    res = process(ndir, s.context(), s.lexicon, st)
    print(f"[annotate] {ndir}")
    print(f"[annotate] engine {res['engine']} ({res['model']}); {res['summary']}")
    for seg in res["segments"]:
        print(f"  [{seg['t0']:8.3f}, {seg['t1']:8.3f})  {seg['move'] or 'unknown: ' + str(seg.get('unresolved'))}"
              f"  heard {seg['anchor_word']!r}  conf {seg['confidence']}  snap {seg['snap']}")
    for w in res["warnings"]:
        print("  warning:", w)
    return 0


def register(sub) -> None:
    p = sub.add_parser("annotate", help="narration annotation tool (local web UI) for move labels",
                       description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", nargs="?", default="serve", choices=["serve", "transcribe", "asr-check"])
    p.add_argument("--take", help="take name (reference time base)")
    p.add_argument("--video", help="any mp4 (video time base unless it is a source of a take)")
    p.add_argument("--video-time", action="store_true", help="keep the video clock even for a take source")
    p.add_argument("--name", help="work/<name> for a lone video (default: its file name as a slug)")
    p.add_argument("--camera", help="initial camera id (take mode)")
    p.add_argument("--beats", help="beat grid JSON [[t, type], ...] in label time (lone videos)")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--annotator", help="name stored in the labels (default: $DANCECAP_ANNOTATOR or the OS user)")
    p.add_argument("--typesafe", action="store_true", help="ask TypeSafe about ambiguous phrases (text only)")
    p.add_argument("--lag", type=float, help="reaction lag in seconds (default 0.4)")
    p.add_argument("--snap", choices=["measure", "half", "beat", "eighth", "none"], help="start snapping (half)")
    p.add_argument("--engine", choices=["auto", "faster-whisper", "sapi"], help="speech recogniser (auto)")
    p.add_argument("--audio", help="transcribe: the narration recording")
    p.add_argument("--log", help="transcribe: the player log JSON {entries:[{a, v, cam, rate, paused, ev}]}")
    p.add_argument("--start", type=float, help="transcribe without a log: label time at recording start")
    p.add_argument("--rate", type=float, default=1.0, help="transcribe without a log: playback rate")
    p.set_defaults(func=cmd_annotate)


if __name__ == "__main__":  # pragma: no cover
    ap = argparse.ArgumentParser()
    register(ap.add_subparsers(dest="command", required=True))
    a = ap.parse_args(["annotate", *sys.argv[1:]])
    sys.exit(a.func(a))
