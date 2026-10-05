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
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
