"""dancecap: multi-phone dance capture -> cameras, room, dancers, poses, physics, timing -> Unity.

    python -m dancecap ingest larissa_kadu            # window frames + room keyframes, native resolution
"""
from __future__ import annotations

import argparse
import json

from .take import Take


def cmd_ingest(a):
    from .ingest import ingest_keyframes, ingest_window
    take = Take.load(a.take)
    cams = a.cams.split(",") if a.cams else None
    known = {s.id for s in take.sources}
    if cams and not set(cams) <= known:
        raise SystemExit(f"unknown camera ids {sorted(set(cams) - known)}; take has {sorted(known)}")
    if not a.keyframes_only:
        metas = ingest_window(take, cams)
        for cam, m in sorted(metas.items()):
            iv = m["frame_interval_ms"]
            print(f"[ingest] {cam}: {m['frame_count']} frames {m['width']}x{m['height']} "
                  f"~{m['fps_median']:.2f} fps (interval {iv.get('min', 0):.1f}-{iv.get('max', 0):.1f} ms) "
                  f"t={m['frames'][0]['t']:.3f}..{m['frames'][-1]['t']:.3f}")
    if not a.window_only:
        kfs = ingest_keyframes(take, cams)
        print(f"[ingest] keyframes: " + ", ".join(f"{c}:{len(v)}" for c, v in sorted(kfs.items())))


UNITY_PROJECT = None  # resolved lazily: WORKSPACE/head-movement


def _unity(command: str, args: dict | None = None) -> str:
    import subprocess

    from .take import WORKSPACE
    cmd = ["unity", "command", command, "--result-only", "--no-banner", "--caller", "plugin", "--skill", "typesafe-ai",
           "--project-path", str(WORKSPACE / "head-movement")]
    for k, v in (args or {}).items():
        cmd += [f"--{k}", str(v).lower() if isinstance(v, bool) else str(v)]
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()


def cmd_ask(a):
    from .judgments import route_lesson_command
    lesson = {}
    if a.run:
        raw = _unity("hm_state")
        try:
            lesson = json.loads(json.loads(raw))
        except (json.JSONDecodeError, TypeError):
            raise SystemExit(f"hm_state unavailable (is the editor in Play mode?): {raw[:200]}")
        lesson = {k: lesson.get(k) for k in ("captures", "capture", "measure", "measureCount", "speed", "playing", "loopMeasure", "layers")}
    routed = route_lesson_command(a.text, lesson)
    print(json.dumps({"commands": routed.commands, "confidence": round(routed.confidence, 3), "clarify": routed.clarify},
                     indent=1))
    if a.debug:
        print(json.dumps(routed.debug, indent=1, default=str))
    if a.run and routed.commands and routed.confidence < 0.4 and not a.force:
        print(f"weakest judgment {routed.confidence:.2f} < 0.4: not executing; rerun with --force to apply this plan")
        return
    if a.run and routed.commands and (routed.clarify is None or a.force):
        for name, args in routed.commands:
            print(f"> {name} {args}: {_unity(name, args)[:160]}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="dancecap", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("ingest")
    sp.add_argument("take")
    sp.add_argument("--cams")
    sp.add_argument("--window-only", action="store_true")
    sp.add_argument("--keyframes-only", action="store_true")
    sp.set_defaults(fn=cmd_ingest)
    sp = sub.add_parser("ask", help="natural-language lesson control -> hm_* Unity commands (TypeSafe)")
    sp.add_argument("text")
    sp.add_argument("--run", action="store_true", help="read hm_state and execute in the open Unity editor")
    sp.add_argument("--force", action="store_true", help="execute even when a clarification is suggested")
    sp.add_argument("--debug", action="store_true")
    sp.set_defaults(fn=cmd_ask)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
