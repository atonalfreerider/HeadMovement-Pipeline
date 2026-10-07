"""python -m dancecap.moves <command> ...   (see dancecap/moves/__init__.py for the command list)

Core commands: graph | placeholder | path | fingerprint | export | validate (+ review, run).
Optional commands come from sibling modules: for each name in OPTIONAL, dancecap.moves.<name> is imported if it
exists and its register(subparsers) adds its own sub-commands (each sub-parser sets `func`). A module that fails to
import is reported on stderr and skipped; the core commands keep working.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

from . import UNITY_STREAMING, moves_dir

OPTIONAL = ["features", "proposals", "dataset", "model", "annotate"]


def cmd_validate(args: argparse.Namespace) -> int:
    from .export import captures_for, validate_export
    from .fingerprint import source_files, stale_sources, validate_fingerprint
    from .graph import GRAPH_PATH, Graph, validate_graph
    from .labels import load_labels, validate_labels
    from .path import validate_path

    problems: list[str] = []
    gp = Path(args.graph or GRAPH_PATH)
    gdoc = json.loads(gp.read_text(encoding="utf-8"))
    problems += [f"graph: {e}" for e in validate_graph(gdoc)]
    graph = Graph(gdoc)
    print(f"[moves] {gp}: {len(gdoc['nodes'])} nodes, {len(gdoc['links'])} links, {len(gdoc['moves'])} moves")
    for take in args.take:
        md = moves_dir(take)
        lab = None
        if (md / "labels.json").exists():
            lab = load_labels(md / "labels.json")
            e = validate_labels(lab, graph)
            problems += [f"{take} labels: {x}" for x in e]
            print(f"[moves] {take} labels: {len(lab.segments)} segments, {lab.labelled_seconds:.2f} s, "
                  f"provenance {lab.provenance}{'' if e else ', valid'}")
        else:
            print(f"[moves] {take}: no labels.json")
        if (md / "path.json").exists():
            e = validate_path(json.loads((md / "path.json").read_text(encoding="utf-8")), graph, lab)
            problems += [f"{take} path: {x}" for x in e]
            print(f"[moves] {take} path: {'valid' if not e else f'{len(e)} problems'}")
        if (md / "fingerprint.json").exists():
            fpd = json.loads((md / "fingerprint.json").read_text(encoding="utf-8"))
            e = validate_fingerprint(fpd) + [f"stale: {x}" for x in stale_sources(fpd, source_files(take))]
            problems += [f"{take} fingerprint: {x}" for x in e]
            print(f"[moves] {take} fingerprint: {'valid, current' if not e else f'{len(e)} problems'}")
        for c in captures_for(take, Path(args.streaming)):
            if (c / "moves").exists():
                e = validate_export(c)
                problems += [f"{c.name}: {x}" for x in e]
                print(f"[moves] {c.name}/moves: {'valid' if not e else f'{len(e)} problems'}")
    for p in problems:
        print("[moves] INVALID", p)
    return 1 if problems else 0


def cmd_run(args: argparse.Namespace) -> int:
    """path + fingerprint (+ review image) + export for a take with labels."""
    ns = argparse.Namespace
    from .export import cmd_export
    from .fingerprint import cmd_fingerprint
    from .graph import GRAPH_PATH
    from .path import PathParams, cmd_path
    d = PathParams()
    g = str(GRAPH_PATH)
    rc = cmd_path(ns(take=args.take, labels=None, graph=g, out=None, no_link_cost=d.no_link_cost,
                     reverse_cost=d.reverse_cost, via_cost=d.via_cost, distance_weight=d.distance_weight,
                     max_gap=d.max_gap_s, strict_direction=False))
    rc = rc or cmd_fingerprint(ns(take=args.take, path=None, counterbalance=None, graph=g, out=None,
                                  no_review=args.no_review))
    if not rc and not args.no_export:
        from .export import captures_for
        if not args.capture and not captures_for(args.take, UNITY_STREAMING):
            print(f"[moves] export skipped: no capture in {UNITY_STREAMING} has provenance.take == {args.take!r} "
                  "yet (re-run `run` after the capture export, or pass --capture)")
            return rc
        rc = cmd_export(ns(take=args.take, capture=args.capture, streaming=str(UNITY_STREAMING), graph=g))
    return rc


def build_parser() -> argparse.ArgumentParser:
    from . import export, fingerprint, graph, path, placeholder, review
    ap = argparse.ArgumentParser(prog="python -m dancecap.moves",
                                 description="zouk dance graph, move labels, graph paths and fingerprints")
    sub = ap.add_subparsers(dest="command", required=True)
    for mod in (graph, placeholder, path, fingerprint, export, review):
        mod.register(sub)
    v = sub.add_parser("validate", help="check graph (+ the takes' labels/path/fingerprint/Unity copies)")
    v.add_argument("take", nargs="*")
    v.add_argument("--graph")
    v.add_argument("--streaming", default=str(UNITY_STREAMING))
    v.set_defaults(func=cmd_validate)
    r = sub.add_parser("run", help="path + fingerprint + review + export for a labelled take")
    r.add_argument("take")
    r.add_argument("--capture", action="append")
    r.add_argument("--no-review", action="store_true")
    r.add_argument("--no-export", action="store_true")
    r.set_defaults(func=cmd_run)
    for name in OPTIONAL:
        try:
            mod = importlib.import_module(f"{__package__}.{name}")
        except ModuleNotFoundError as e:
            if e.name == f"{__package__}.{name}":
                continue
            print(f"[moves] optional module {name!r} not loaded: {e}", file=sys.stderr)
            continue
        except Exception as e:  # noqa: BLE001 - a broken optional module must not take the core CLI down
            print(f"[moves] optional module {name!r} failed to import: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        reg = getattr(mod, "register", None)
        if callable(reg):
            try:
                reg(sub)
            except Exception as e:  # noqa: BLE001
                print(f"[moves] optional module {name!r}: register() failed: {type(e).__name__}: {e}",
                      file=sys.stderr)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
