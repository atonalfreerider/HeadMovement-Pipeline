"""Labels -> path through the dance graph: work/<take>/moves/path.json (version 1).

Each couple-tier segment names a move; a move can have several node instances in the graph (the same move in
different contexts, e.g. two "Lateral" (Corredor) nodes). The instance sequence is chosen by Viterbi, minimising per
transition
  0                 an authored link prev -> next exists (scaffold or proposed)
  via_cost x k      prev -> k blank junction nodes -> next (the user's unnamed structural nodes, which labels cannot
                    name: e.g. Tilt Right -> (unnamed) -> Howl Up) - a graph connection, not a new link
  reverse_cost      only next -> prev exists (an authored link / blank chain walked backwards, e.g. back into the
                    hug); with strict_direction it counts as a missing link instead
  no_link_cost      no connection in either direction  -> flagged new_link (a candidate to add to the graph)
  + distance_weight x |position(prev) - position(next)| (metres in the graph) as a small tie-breaker.
Staying on the same node (consecutive segments of one move) costs 0. A gap longer than max_gap_s between segments
breaks the path: no transition is scored or reported across it. So does (break_on_unknown, default on) an unknown
narrated move between two segments - a labels.json 'unresolved' item (a phrase the narration could not map) lying in
the gap: the dance went A -> ? -> B, so A -> B is neither scored nor reported as a new link, however short the gap.
A segment with `node` set is pinned to that node.

path.json:
  {version:1, take, provenance (of the labels: human|narration|auto|placeholder|mixed:...), time_base:'reference',
   labels:<file>, params:{...},
   steps:[{t0, t1, node, move, link:<link id or null; the first link in travel order>, new_link:bool,
           + segment, stay, reverse, via:[blank node ids passed through, in travel order],
             links:[every link id in travel order], break_before, gap_before,
             unknown_before:[phrases of the unresolved items in the gap] (only when there are some)}],
   new_links:[{from, to, count, + from_move, to_move}], cost, labels_sha256_16, note}
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import PLACEHOLDER_BANNER, moves_dir
from .graph import GRAPH_PATH, Graph, _rel, _sha, load_graph
from .labels import Labels, load_labels, to_reference, validate_labels

VERSION = 1


class PathError(ValueError):
    pass


@dataclass
class PathParams:
    no_link_cost: float = 10.0
    reverse_cost: float = 0.5
    via_cost: float = 0.1  # per blank junction node passed through
    distance_weight: float = 0.05
    max_gap_s: float = 1.0
    strict_direction: bool = False
    break_on_unknown: bool = True  # an unresolved (unknown) narrated move in the gap breaks the path


def path_file(take: str) -> Path:
    return moves_dir(take) / "path.json"


def unknown_between(a: float, b: float, unresolved: list[dict], eps: float = 1e-6) -> list[str]:
    """Phrases of the unresolved items (reference time) lying in the gap [a, b] between two segments."""
    out = []
    for u in unresolved:
        try:
            u0, u1 = float(u["t0"]), float(u["t1"])
        except (KeyError, TypeError, ValueError):
            continue
        if u1 > u0 and u0 >= a - eps and u1 <= b + eps:
            out.append(str(u.get("unresolved") or u.get("anchor_word") or "unknown"))
    return out


def _transition(graph: Graph, a: str, b: str, p: PathParams) -> tuple[float, dict]:
    """Cost and description of moving from node a to node b (adjacent segments, no break)."""
    if a == b:
        return 0.0, {"link": None, "new_link": False, "stay": True, "reverse": False, "via": [], "links": []}
    dist = p.distance_weight * float(np.linalg.norm(graph.position(a) - graph.position(b)))
    kind, info = graph.connection(a, b, p.strict_direction)
    if kind is None:
        return p.no_link_cost + dist, {"link": None, "new_link": True, "stay": False, "reverse": False, "via": [],
                                       "links": []}
    cost = dist + p.via_cost * len(info["via"]) + (p.reverse_cost if kind.startswith("reverse") else 0.0)
    return cost, {"link": info["link"], "new_link": False, "stay": False, "reverse": kind.startswith("reverse"),
                  "via": info["via"], "links": info["links"]}


def resolve_path(labels: Labels, graph: Graph, params: PathParams | None = None,
                 labels_file: str | None = None, labels_sha: str | None = None) -> dict:
    """Viterbi over node instances of the couple-tier segments -> path.json dict."""
    p = params or PathParams()
    lab = to_reference(labels)
    segs = lab.tier("couple")
    cands: list[list[str]] = []
    for s in segs:
        if s.node is not None:
            if s.node not in graph.nodes:
                raise PathError(f"segment {s.id}: unknown node {s.node}")
            cands.append([s.node])
            continue
        inst = graph.instances(s.move)
        if not inst:
            raise PathError(f"segment {s.id}: move {s.move!r} has no node in the graph")
        cands.append(inst)
    n = len(segs)
    unresolved = [u for u in lab.extra.get("unresolved") or [] if isinstance(u, dict) and u.get("tier", "couple") == "couple"]
    unknown = [unknown_between(segs[k - 1].t1, segs[k].t0, unresolved) if k and p.break_on_unknown else []
               for k in range(n)]
    breaks = [False] + [segs[k].t0 - segs[k - 1].t1 > p.max_gap_s or bool(unknown[k]) for k in range(1, n)]

    # Viterbi
    cost = [np.zeros(len(c)) for c in cands]
    back: list[np.ndarray] = [np.zeros(len(c), int) for c in cands]
    for k in range(1, n):
        for j, b in enumerate(cands[k]):
            if breaks[k]:
                i = int(np.argmin(cost[k - 1]))
                cost[k][j], back[k][j] = cost[k - 1][i], i
                continue
            best, bi = np.inf, 0
            for i, a in enumerate(cands[k - 1]):
                c = cost[k - 1][i] + _transition(graph, a, b, p)[0]
                if c < best - 1e-12:
                    best, bi = c, i
            cost[k][j], back[k][j] = best, bi
    choice = [0] * n
    if n:
        choice[-1] = int(np.argmin(cost[-1]))
        for k in range(n - 1, 0, -1):
            choice[k - 1] = int(back[k][choice[k]])

    steps, new_links = [], Counter()
    for k, s in enumerate(segs):
        node = cands[k][choice[k]]
        st = {"t0": round(s.t0, 6), "t1": round(s.t1, 6), "node": node, "move": s.move, "link": None,
              "new_link": False, "segment": s.id, "stay": False, "reverse": False, "via": [], "links": [],
              "break_before": bool(breaks[k]), "gap_before": round(s.t0 - segs[k - 1].t1, 6) if k else None}
        if unknown[k]:
            st["unknown_before"] = unknown[k]
        if k and not breaks[k]:
            prev = cands[k - 1][choice[k - 1]]
            _, info = _transition(graph, prev, node, p)
            st.update(info)
            if info["new_link"]:
                new_links[(prev, node)] += 1
        steps.append(st)
    total = float(cost[-1][choice[-1]]) if n else 0.0
    out = {
        "version": VERSION, "take": labels.take, "provenance": lab.provenance, "time_base": "reference",
        "labels": labels_file, "graph": str(GRAPH_PATH.name), "params": asdict(p),
        "steps": steps,
        "new_links": [{"from": a, "to": b, "count": c, "from_move": graph.nodes[a]["move"],
                       "to_move": graph.nodes[b]["move"]} for (a, b), c in sorted(new_links.items())],
        "cost": round(total, 6),
        "labels_sha256_16": labels_sha,
    }
    if lab.provenance == "placeholder":
        out["note"] = f"{PLACEHOLDER_BANNER}: path of a placeholder timeline"
    return out


def validate_path(doc: dict, graph: Graph | None = None, labels: Labels | None = None) -> list[str]:
    """Contract checks; with the graph also node/move/link consistency (the link id must belong to the transition,
    new_link must match the graph), with the labels also that every step is the labels' segment (stale paths)."""
    errs = []
    if doc.get("version") != VERSION:
        errs.append(f"path version {doc.get('version')!r} != {VERSION}")
    for k in ("take", "provenance", "steps", "new_links"):
        if k not in doc:
            errs.append(f"path: missing {k!r}")
    if errs:
        return errs
    strict = bool(doc.get("params", {}).get("strict_direction"))
    prev = None
    for i, st in enumerate(doc["steps"]):
        missing = [k for k in ("t0", "t1", "node", "move", "link", "new_link") if k not in st]
        if missing:
            errs.append(f"step {i}: missing {missing}")
            prev = None
            continue
        if not st["t0"] < st["t1"]:
            errs.append(f"step {i}: t0 >= t1")
        if prev is not None and st["t0"] < prev["t1"] - 1e-6:
            errs.append(f"step {i}: overlaps the previous step")
        nd = None
        if graph is not None:
            nd = graph.nodes.get(st["node"])
            if nd is None:
                errs.append(f"step {i}: unknown node {st['node']}")
            elif nd["move"] != st["move"]:
                errs.append(f"step {i}: node {st['node']} is {nd['move']!r}, not {st['move']!r}")
            if st["link"] is not None and st["link"] not in graph.link_by_id:
                errs.append(f"step {i}: unknown link {st['link']}")
            joined = prev is not None and not st.get("break_before")
            if joined and st["node"] != prev["node"]:
                if nd is not None and prev["node"] in graph.nodes:
                    kind, info = graph.connection(prev["node"], st["node"], strict)
                    if bool(st["new_link"]) == (kind is not None):
                        errs.append(f"step {i}: new_link={st['new_link']} but the graph "
                                    f"{'connects' if kind else 'does not connect'} the two nodes")
                    if kind is None and st["link"] is not None:
                        errs.append(f"step {i}: link {st['link']} given for a transition the graph does not have")
                    if kind is not None and st["link"] != info["link"]:
                        errs.append(f"step {i}: link {st['link']} does not belong to {prev['node']} -> "
                                    f"{st['node']} (expected {info['link']})")
                    if kind is not None and list(st.get("via") or []) != info["via"]:
                        errs.append(f"step {i}: via {st.get('via')} != {info['via']}")
            elif st["link"] is not None:
                errs.append(f"step {i}: link {st['link']} on a step without a transition")
        prev = st
    if labels is not None:
        segs = to_reference(labels).tier("couple")
        if len(doc["steps"]) != len(segs):
            errs.append(f"path has {len(doc['steps'])} steps, labels have {len(segs)} segments (stale path: re-run "
                        "`python -m dancecap.moves path`)")
        else:
            for i, (st, sg) in enumerate(zip(doc["steps"], segs)):
                if abs(st["t0"] - sg.t0) > 1e-5 or abs(st["t1"] - sg.t1) > 1e-5 or st["move"] != sg.move:
                    errs.append(f"step {i}: [{st['t0']}, {st['t1']}) {st['move']} != label segment {sg.id} "
                                f"[{sg.t0:.6f}, {sg.t1:.6f}) {sg.move} (stale path: re-run "
                                "`python -m dancecap.moves path`)")
                    break
    return errs


def cmd_path(args: argparse.Namespace) -> int:
    graph = load_graph(Path(args.graph))
    src = Path(args.labels) if args.labels else moves_dir(args.take) / "labels.json"
    labels = load_labels(src)
    errs = validate_labels(labels, graph)
    if errs:
        for e in errs:
            print("[moves] labels INVALID", e)
        return 1
    params = PathParams(no_link_cost=args.no_link_cost, reverse_cost=args.reverse_cost,
                        via_cost=getattr(args, "via_cost", PathParams.via_cost),
                        distance_weight=args.distance_weight, max_gap_s=args.max_gap,
                        strict_direction=args.strict_direction,
                        break_on_unknown=not getattr(args, "no_unknown_breaks", False))
    doc = resolve_path(labels, graph, params, labels_file=_rel(src), labels_sha=_sha(src))
    out = Path(args.out) if args.out else path_file(args.take)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    nl = sum(st["new_link"] for st in doc["steps"])
    print(f"[moves] wrote {out}: {len(doc['steps'])} steps, {nl} new-link transitions "
          f"({len(doc['new_links'])} distinct), {sum(st['reverse'] for st in doc['steps'])} reverse, "
          f"{sum(bool(st['via']) for st in doc['steps'])} through blank junctions, cost {doc['cost']}"
          f"  [provenance {doc['provenance']}]")
    return 0


def register(sub) -> None:
    p = sub.add_parser("path", help="labels.json -> path.json (Viterbi over node instances)")
    p.add_argument("take")
    p.add_argument("--labels", help="labels file (default work/<take>/moves/labels.json)")
    p.add_argument("--graph", default=str(GRAPH_PATH))
    p.add_argument("--out")
    d = PathParams()
    p.add_argument("--no-link-cost", type=float, default=d.no_link_cost)
    p.add_argument("--reverse-cost", type=float, default=d.reverse_cost)
    p.add_argument("--via-cost", type=float, default=d.via_cost, help="per blank junction node passed through")
    p.add_argument("--distance-weight", type=float, default=d.distance_weight)
    p.add_argument("--max-gap", type=float, default=d.max_gap_s, help="seconds; longer gaps break the path")
    p.add_argument("--strict-direction", action="store_true", help="a reverse traversal counts as a new link")
    p.add_argument("--no-unknown-breaks", action="store_true",
                   help="score transitions across unresolved (unknown) narrated moves too (default: they break the path)")
    p.set_defaults(func=cmd_path)
