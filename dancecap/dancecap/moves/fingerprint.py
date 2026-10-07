"""Dance fingerprint: the time-independent summary of a path through the graph - work/<take>/moves/fingerprint.json.

fingerprint.json (version 1):
  {version:1, take, provenance, labelled_seconds,
   nodes:{<node id>:{seconds, share, visits, + move, band}}          visited nodes only (unvisited = absent)
   links:{'<from>|<to>':count}                                       transitions as traversed (from -> to), incl.
                                                                     new links and authored links walked backwards
   energy_bands:[{band, seconds, share, + label}],
   distinct_moves, transitions, counterbalance_share:<float|null>,
   + link_ids:{'<from>|<to>': <graph link id | null>}, via:{'<from>|<to>': [blank junction ids]},
     new_link_keys:[...], reverse_keys:[...], counterbalance_seconds, counterbalance_analysed:[t0, t1] | null,
     counterbalance_analysed_seconds, counterbalance_source, longest_phrase:{t0, t1, seconds, steps, moves},
     sources:{labels|path|counterbalance: {path, sha256_16}}, note}

  visits       = runs of consecutive steps on the node (a stay continues the visit; a path break ends it)
  transitions  = node changes between adjacent steps (not across path breaks)
  counterbalance_share = seconds of labelled time inside work/<take>/counterbalance/counterbalance.json intervals
                 (merged, reference time) / labelled seconds INSIDE THE ANALYSED RANGE of that file
                 (source.t_range, else the track's t range; the detection only ran there - e.g. 56-62 s of a dance
                 labelled 30-65 s). null when the file does not exist or the labels do not overlap its range.
  sources      sha256 (16 hex) of the labels, path and counterbalance files the fingerprint was computed from, so
                 `validate` and `export` can tell a stale fingerprint
  longest_phrase = the longest run of steps joined only by graph links (forward or walked backwards) or stays
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from . import PLACEHOLDER_BANNER, counterbalance_path, moves_dir, work_dir
from .graph import GRAPH_PATH, Graph, _rel, _sha, load_graph

VERSION = 1


def fingerprint_file(take: str) -> Path:
    return moves_dir(take) / "fingerprint.json"


def _merge(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for a, b in sorted(intervals):
        if b <= a:
            continue
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def counterbalance_intervals(doc: dict) -> list[tuple[float, float]]:
    """Merged [t0, t1) reference-time intervals of a counterbalance.json (version 1)."""
    if doc.get("time_base", "reference") != "reference":
        raise ValueError("counterbalance.json must be in reference time for the fingerprint")
    return _merge([(float(i["t0"]), float(i["t1"])) for i in doc.get("intervals", [])])


def counterbalance_range(doc: dict) -> tuple[float, float] | None:
    """[t0, t1] reference seconds the counterbalance detection analysed: source.t_range, else track.t; None when the
    file does not say (then the whole labelled time counts as analysed)."""
    r = (doc.get("source") or {}).get("t_range")
    if isinstance(r, (list, tuple)) and len(r) == 2:
        return float(r[0]), float(r[1])
    t = (doc.get("track") or {}).get("t")
    if isinstance(t, list) and len(t) >= 2:
        return float(t[0]), float(t[-1])
    return None


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def compute_fingerprint(path_doc: dict, graph: Graph, counterbalance: dict | None = None,
                        counterbalance_source: str | None = None, sources: dict | None = None) -> dict:
    steps = path_doc["steps"]
    labelled = float(sum(s["t1"] - s["t0"] for s in steps))
    secs: dict[str, float] = defaultdict(float)
    visits: Counter = Counter()
    links: Counter = Counter()
    link_ids: dict[str, str | None] = {}
    via: dict[str, list[str]] = {}
    new_keys, rev_keys = set(), set()
    transitions = 0
    for k, s in enumerate(steps):
        secs[s["node"]] += s["t1"] - s["t0"]
        prev = steps[k - 1] if k else None
        joined = prev is not None and not s.get("break_before", False)
        if not joined or prev["node"] != s["node"]:
            visits[s["node"]] += 1
        if joined and prev["node"] != s["node"]:
            transitions += 1
            key = f"{prev['node']}|{s['node']}"
            links[key] += 1
            link_ids[key] = s.get("link")
            if s.get("via"):
                via[key] = list(s["via"])
            if s.get("new_link"):
                new_keys.add(key)
            if s.get("reverse"):
                rev_keys.add(key)
    nodes = {}
    for nid in sorted(secs, key=lambda i: -secs[i]):
        nd = graph.nodes.get(nid, {})
        nodes[nid] = {"seconds": round(secs[nid], 6), "share": round(secs[nid] / labelled, 6) if labelled else 0.0,
                      "visits": int(visits[nid]), "move": nd.get("move"), "band": nd.get("band")}
    band_secs: dict[int, float] = defaultdict(float)
    for nid, sec in secs.items():
        band_secs[int(graph.nodes[nid]["band"])] += sec
    bands = [{"band": b["band"], "seconds": round(band_secs[b["band"]], 6),
              "share": round(band_secs[b["band"]] / labelled, 6) if labelled else 0.0, "label": b["label"]}
             for b in graph.data["energy_bands"]]

    # longest phrase: steps joined by links / stays (no new link, no break)
    best, cur = None, None
    for k, s in enumerate(steps):
        ok = k > 0 and not s.get("break_before") and not s.get("new_link")
        if not ok:
            cur = [k, k]
        else:
            cur[1] = k
        span = (steps[cur[0]]["t0"], steps[cur[1]]["t1"])
        if best is None or span[1] - span[0] > best[1][1] - best[1][0] + 1e-9:
            best = (tuple(cur), span)
    phrase = None
    if best is not None:
        (a, b), (t0, t1) = best
        mv = []
        for s in steps[a:b + 1]:
            if not mv or mv[-1] != s["move"]:
                mv.append(s["move"])
        phrase = {"t0": round(t0, 6), "t1": round(t1, 6), "seconds": round(t1 - t0, 6), "steps": b - a + 1,
                  "moves": mv}

    cb_share, cb_secs, cb_range, cb_analysed = None, None, None, None
    if counterbalance is not None:
        iv = counterbalance_intervals(counterbalance)
        cb_range = counterbalance_range(counterbalance)
        r0, r1 = cb_range if cb_range else (-float("inf"), float("inf"))
        cb_secs, cb_analysed = 0.0, 0.0
        for s in steps:
            cb_analysed += _overlap(s["t0"], s["t1"], r0, r1)
            for a, b in iv:
                cb_secs += _overlap(s["t0"], s["t1"], max(a, r0), min(b, r1))
        cb_share = round(cb_secs / cb_analysed, 6) if cb_analysed > 1e-9 else None
        cb_secs, cb_analysed = round(cb_secs, 6), round(cb_analysed, 6)
    out = {
        "version": VERSION, "take": path_doc["take"], "provenance": path_doc.get("provenance"),
        "labelled_seconds": round(labelled, 6),
        "nodes": nodes,
        "links": dict(sorted(links.items(), key=lambda kv: (-kv[1], kv[0]))),
        "energy_bands": bands,
        "distinct_moves": len({s["move"] for s in steps}),
        "transitions": transitions,
        "counterbalance_share": cb_share,
        "counterbalance_seconds": cb_secs,
        "counterbalance_analysed": [round(cb_range[0], 6), round(cb_range[1], 6)] if cb_range else None,
        "counterbalance_analysed_seconds": cb_analysed,
        "counterbalance_source": counterbalance_source,
        "link_ids": link_ids, "via": via, "new_link_keys": sorted(new_keys), "reverse_keys": sorted(rev_keys),
        "longest_phrase": phrase,
        "time_base": "reference",
        "sources": sources or {},
    }
    if path_doc.get("provenance") == "placeholder":
        out["note"] = f"{PLACEHOLDER_BANNER}: fingerprint of a placeholder timeline"
    return out


def validate_fingerprint(fp: dict) -> list[str]:
    errs = []
    if fp.get("version") != VERSION:
        errs.append(f"fingerprint version {fp.get('version')!r} != {VERSION}")
    for k in ("take", "provenance", "labelled_seconds", "nodes", "links", "energy_bands", "distinct_moves",
              "transitions", "counterbalance_share"):
        if k not in fp:
            errs.append(f"fingerprint: missing {k!r}")
    if errs:
        return errs
    tot = sum(v["seconds"] for v in fp["nodes"].values())
    if abs(tot - fp["labelled_seconds"]) > 1e-3:
        errs.append(f"node seconds sum {tot:.4f} != labelled_seconds {fp['labelled_seconds']:.4f}")
    bt = sum(b["seconds"] for b in fp["energy_bands"])
    if abs(bt - fp["labelled_seconds"]) > 1e-3:
        errs.append(f"energy band seconds sum {bt:.4f} != labelled_seconds")
    if fp["nodes"] and abs(sum(v["share"] for v in fp["nodes"].values()) - 1.0) > 1e-3:
        errs.append("node shares do not sum to 1")
    if sum(fp["links"].values()) != fp["transitions"]:
        errs.append("link counts do not sum to transitions")
    cb = fp["counterbalance_share"]
    if cb is not None and not (0.0 <= cb <= 1.0 + 1e-9):
        errs.append(f"counterbalance_share {cb} outside 0..1")
    an = fp.get("counterbalance_analysed_seconds")
    if an is not None and an > fp["labelled_seconds"] + 1e-3:
        errs.append("counterbalance_analysed_seconds exceeds labelled_seconds")
    return errs


def stale_sources(fp: dict, files: dict[str, Path]) -> list[str]:
    """Inputs that changed since the fingerprint was computed: files = {labels|path|counterbalance: Path}. A file
    recorded but now missing, present but not recorded, or with a different sha256 counts as changed. Fingerprints
    without `sources` (older files) report 'sources not recorded'."""
    rec = fp.get("sources")
    if not rec:
        return ["sources not recorded (fingerprint made before sha tracking): re-run fingerprint"]
    out = []
    for key, path in files.items():
        old = (rec.get(key) or {}).get("sha256_16")
        new = _sha(Path(path)) if path is not None else None
        if old != new:
            out.append(f"{key} changed since the fingerprint ({old} -> {new})")
    return out


def source_files(take: str, path_file_: Path | None = None, cb_file: Path | None = None) -> dict[str, Path]:
    md = moves_dir(take)
    return {"labels": md / "labels.json", "path": Path(path_file_) if path_file_ else md / "path.json",
            "counterbalance": Path(cb_file) if cb_file else counterbalance_path(take)}


def source_record(files: dict[str, Path]) -> dict:
    return {k: ({"path": _rel(p), "sha256_16": _sha(Path(p))} if p is not None and Path(p).exists() else None)
            for k, p in files.items()}


def cmd_fingerprint(args: argparse.Namespace) -> int:
    graph = load_graph(Path(args.graph))
    pf = Path(args.path) if args.path else moves_dir(args.take) / "path.json"
    path_doc = json.loads(pf.read_text(encoding="utf-8"))
    cbp = Path(args.counterbalance) if args.counterbalance else counterbalance_path(args.take)
    cb = json.loads(cbp.read_text(encoding="utf-8")) if cbp.exists() else None
    fp = compute_fingerprint(path_doc, graph, cb, _rel(cbp) if cb is not None else None,
                             sources=source_record(source_files(args.take, pf, cbp)))
    errs = validate_fingerprint(fp)
    if errs:
        for e in errs:
            print("[moves] fingerprint INVALID", e)
        return 1
    out = Path(args.out) if args.out else fingerprint_file(args.take)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fp, indent=1) + "\n", encoding="utf-8")
    an = fp["counterbalance_analysed"]
    print(f"[moves] wrote {out}: {fp['labelled_seconds']:.2f} s labelled, {len(fp['nodes'])} nodes visited, "
          f"{fp['distinct_moves']} distinct moves, {fp['transitions']} transitions, counterbalance_share "
          f"{fp['counterbalance_share']}"
          + (f" (of {fp['counterbalance_analysed_seconds']:.2f} labelled s inside the analysed range "
             f"[{an[0]:.2f}, {an[1]:.2f}])" if an else "")
          + f"  [provenance {fp['provenance']}]")
    if not args.no_review:
        from .review import draw_fingerprint
        img = draw_fingerprint(graph, fp, path_doc, work_dir(args.take) / "review" / "fingerprint.png")
        print(f"[moves] review image {img}")
    return 0


def register(sub) -> None:
    p = sub.add_parser("fingerprint", help="path.json -> fingerprint.json (+ work/<take>/review/fingerprint.png)")
    p.add_argument("take")
    p.add_argument("--path", help="path file (default work/<take>/moves/path.json)")
    p.add_argument("--counterbalance", help="default work/<take>/counterbalance/counterbalance.json (if present)")
    p.add_argument("--graph", default=str(GRAPH_PATH))
    p.add_argument("--out")
    p.add_argument("--no-review", action="store_true")
    p.set_defaults(func=cmd_fingerprint)
