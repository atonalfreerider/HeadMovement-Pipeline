"""PLACEHOLDER move timelines for the viewer prototype - NOT an analysis of the dance.

    python -m dancecap.moves placeholder synthetic            whole beat range
    python -m dancecap.moves placeholder larissa_kadu         the take's reconstructed window ([56, 62] s = the
                                                              capture / SMPL-X range; --span T0 T1 for another)

A seeded random walk along the graph's connections (links, or chains through blank junction nodes; either
direction, forward preferred) that starts in the grounded band, rises through the energy bands towards the peak and
comes back down, with one deliberate jump between moves that have no connection (so the viewer's dashed "new link"
display has something to show).

Segment boundaries sit on the take's beat grid (beats.json [[t, type]]: 8 eighth slots per measure, zouk type pattern
1 3 3 2 1 3 2 3 from the downbeat). The measure phase is derived from that pattern (measure_phase; a grid that does
not follow it is refused). Unit = 1 measure (1-2 measures per segment) when the span holds at least MIN_UNITS
measures; a short span (e.g. the 6-s larissa_kadu window, < 2 measures) uses half or quarter measures, one unit per
segment, so the walk still climbs the bands and shows a new link inside the capture. With a span, the segments cover
it fully (from the last unit start <= t0 to the first unit boundary >= t1).

Every segment has provenance 'placeholder', node null (the path resolver picks the instances), and the file carries
placeholder:true + a note. The moves have nothing to do with what the dancers actually did.

Refuses to overwrite a labels.json that holds any non-placeholder segment (human / narration / auto).
"""
from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path

import numpy as np

from . import PLACEHOLDER_BANNER, work_dir
from .graph import GRAPH_PATH, Graph, _rel, load_graph
from .labels import Labels, LabelsError, Segment, labels_path, load_labels, save_labels

NOISE = 1.5  # seeded cost noise: variety between seeds
ZOUK_PATTERN = (1, 3, 3, 2, 1, 3, 2, 3)  # beat types of the 8 eighth slots of a measure, from the downbeat
MIN_UNITS = 6  # a span with fewer whole measures uses half / quarter measures

NOTE = (f"{PLACEHOLDER_BANNER.upper()}: a seeded random walk along the dance graph's links on the measure grid, "
        "made only to exercise the viewer prototype (graph path + fingerprint). It says nothing about the moves "
        "actually danced.")


def beats_file(take: str) -> Path:
    w = work_dir(take)
    for p in (w / "beats.json", w / "beats" / "beats.json", w / "timing" / "beats.json"):
        if p.exists():
            return p
    raise FileNotFoundError(f"no beats.json for take {take} (work/{take}/beats.json)")


def measure_phase(grid: list, min_score: float = 0.9) -> int:
    """Index of the first downbeat in the grid: the phase p (0..7) whose shifted ZOUK_PATTERN matches the grid's beat
    types best. A grid without type information (all one type) is assumed to start on a downbeat (p = 0); a grid whose
    best match is below min_score raises ValueError (measure starts would be wrong)."""
    types = [int(r[1]) for r in grid if len(r) > 1]
    if len(types) < 8 or len(set(types)) == 1:
        return 0
    scores = [float(np.mean([types[i] == ZOUK_PATTERN[(i - p) % 8] for i in range(len(types))])) for p in range(8)]
    p = int(np.argmax(scores))
    if scores[p] < min_score:
        raise ValueError(f"beat grid does not follow the zouk type pattern {ZOUK_PATTERN} (best phase {p}: "
                         f"{scores[p]:.2f} of the slots match)")
    return p


def unit_bounds(grid: list, step: int = 8, phase: int | None = None) -> list[float]:
    """Start times of the complete units of `step` eighth slots (8 = measure, 4 = half, 2 = quarter) from the measure
    phase, + the end of the last one (extrapolated when the grid stops inside it)."""
    t = [float(r[0]) for r in grid]
    p = (measure_phase(grid) if phase is None else int(phase)) % step
    n = (len(t) - p) // step
    if n == 0:
        raise ValueError(f"beat grid shorter than one unit of {step} eighths")
    starts = [t[p + step * k] for k in range(n)]
    if p + step * n < len(t):
        end = t[p + step * n]
    else:
        end = starts[-1] + (starts[-1] - starts[-2] if n > 1 else step * float(np.median(np.diff(t))))
    return starts + [end]


def measure_bounds(grid: list, phase: int | None = None) -> list[float]:
    """Measure start times + the end of the last complete measure (grid = [[t, type], ...], 8 slots/measure; the
    first downbeat from measure_phase)."""
    return unit_bounds(grid, 8, phase)


def cover(bounds: list[float], span: tuple[float, float]) -> list[float]:
    """The bounds from the last one <= span[0] to the first one >= span[1] (clipped to the grid)."""
    eps = 1e-6
    i0 = max([i for i, b in enumerate(bounds) if b <= span[0] + eps] or [0])
    i1 = min([i for i, b in enumerate(bounds) if b >= span[1] - eps] or [len(bounds) - 1])
    return bounds[i0:i1 + 1]


def default_span(take: str) -> tuple[float, float] | None:
    """The take's reconstructed window (takes/<take>.toml `window` = the capture / SMPL-X range) when it has a toml,
    else None (= whole beat range)."""
    from ..take import PACKAGE_ROOT, Take
    if not (PACKAGE_ROOT / "takes" / f"{take}.toml").exists():
        return None
    try:
        return tuple(Take.load(take).window)
    except Exception:  # alignment file missing etc.
        import tomllib
        cfg = tomllib.loads((PACKAGE_ROOT / "takes" / f"{take}.toml").read_text(encoding="utf-8"))
        return tuple(cfg["window"])


def partition(bounds: list[float], rng: np.random.Generator, min_m: int = 1, max_m: int = 2) -> list[tuple[float, float]]:
    """Consecutive [t0, t1) segments of min_m..max_m measures covering all measures in bounds."""
    segs, i, n = [], 0, len(bounds) - 1
    while i < n:
        k = int(rng.integers(min_m, max_m + 1))
        k = min(k, n - i)
        segs.append((bounds[i], bounds[i + k]))
        i += k
    return segs


def energy_profile(n: int, peak: float = 4.0) -> np.ndarray:
    """Target band per segment: grounded -> peak -> grounded (half sine)."""
    if n == 1:
        return np.zeros(1)
    return peak * np.sin(np.pi * np.arange(n) / (n - 1))


def walk(graph: Graph, n: int, rng: np.random.Generator, beam: int = 256, peak: float = 4.0,
         jumps: int = 1) -> list[str]:
    """n node ids: a walk along links (either direction) following energy_profile; beam search with seeded noise.

    `jumps` (0 or 1): at one seeded step in the middle the walk moves to a node whose move has no connection at all
    to the previous move (so path.json flags a new link and the viewer's dashed new-link rendering is exercised).
    Chains through blank junction nodes (graph.via) count as connections, like links."""
    named = [i for i, nd in graph.nodes.items() if nd["move"]]
    nbr: dict[str, list[tuple[str, bool]]] = {i: [] for i in named}
    linked_moves: set[tuple[str, str]] = set()
    edges = [(l["from"], l["to"]) for l in graph.links] + sorted(graph.via)
    for a, b in edges:
        if a in nbr and b in nbr:
            nbr[a].append((b, True))
            nbr[b].append((a, False))
            ma, mb = graph.nodes[a]["move"], graph.nodes[b]["move"]
            linked_moves |= {(ma, mb), (mb, ma)}
    jump_at = int(rng.integers(2, n - 1)) if jumps and n >= 5 else -1
    band = {i: graph.nodes[i]["band"] for i in named}
    target = energy_profile(n, peak)
    starts = [i for i in named if band[i] == 0 and nbr[i]]
    w_last = 3.0  # the walk must come back down to the grounded band

    def band_cost(b: str, k: int) -> float:
        return 2.0 * abs(band[b] - target[k]) * (w_last if k == n - 1 else 1.0)

    states = [(band_cost(i, 0) + float(rng.uniform(0, NOISE)), [i]) for i in starts]
    states.sort(key=lambda s: s[0])
    states = states[:beam]
    for k in range(1, n):
        nxt = []
        for c, seq in states:
            last = seq[-1]
            used_moves = {graph.nodes[x]["move"] for x in seq}
            if k == jump_at:
                lm = graph.nodes[last]["move"]
                options = [(b, True) for b in named if graph.nodes[b]["move"] != lm
                           and (lm, graph.nodes[b]["move"]) not in linked_moves and nbr[b]]
            else:
                options = nbr[last]
            for b, fwd in options:
                cc = (c + band_cost(b, k) + 1.0 * max(0, abs(band[b] - band[last]) - 1)  # climb band by band
                      + (0.0 if fwd else 0.6) + float(rng.uniform(0, NOISE)))
                if b in seq:
                    cc += 1.5
                if graph.nodes[b]["move"] in used_moves:
                    cc += 0.7
                nxt.append((cc, seq + [b]))
        if not nxt:  # dead end everywhere: jump (the path will flag a new link)
            for c, seq in states:
                for b in named:
                    if b != seq[-1]:
                        nxt.append((c + 5.0 + band_cost(b, k) + float(rng.uniform(0, NOISE)), seq + [b]))
        nxt.sort(key=lambda s: s[0])
        states = nxt[:beam]
    # sample among the near-best complete walks (variety between seeds, all of them follow the profile)
    c0 = states[0][0]
    good = [s for s in states if s[0] <= c0 + 2.0][:32]
    w = np.exp(-np.array([s[0] - c0 for s in good]))
    return good[int(rng.choice(len(good), p=w / w.sum()))][1]


def default_seed(take: str) -> int:
    return zlib.crc32(take.encode()) % 1000


def make_placeholder(take: str, graph: Graph, seed: int | None = None, span: tuple[float, float] | None = None,
                     beats: Path | None = None) -> Labels:
    seed = default_seed(take) if seed is None else int(seed)
    bp = Path(beats) if beats else beats_file(take)
    grid = json.loads(bp.read_text())
    phase = measure_phase(grid)
    step, max_m = 8, 2
    bounds = unit_bounds(grid, 8, phase)
    if span is not None:
        for step in (8, 4, 2):  # measure, half measure, quarter measure
            bounds = cover(unit_bounds(grid, step, phase), span)
            if len(bounds) - 1 >= MIN_UNITS:
                break
        max_m = 2 if step == 8 else 1
        if len(bounds) < 2:
            raise ValueError(f"no beat-grid unit inside span {span}")
    rng = np.random.default_rng(seed)
    segs = partition(bounds, rng, 1, max_m)
    nodes = walk(graph, len(segs), rng)
    out = Labels(take=take, time_base="reference")
    for k, ((t0, t1), nid) in enumerate(zip(segs, nodes), 1):
        nd = graph.nodes[nid]
        units = sum(t0 - 1e-9 <= b < t1 - 1e-9 for b in bounds)
        measures = units if step == 8 else units * step / 8
        out.segments.append(Segment(
            id=f"ph-{k:04d}", t0=round(t0, 6), t1=round(t1, 6), move=nd["move"], node=None, tier="couple",
            provenance="placeholder", confidence=None, annotator="dancecap.moves.placeholder",
            transcript=None, notes=f"PLACEHOLDER (seed {seed}) - not an analysis",
            extra={"placeholder_node": nid, "measures": measures, "unit_eighths": step}))
    out.extra.update({
        "placeholder": True, "note": NOTE,
        "generator": {"module": "dancecap.moves.placeholder", "seed": seed,
                      "span": list(span) if span else [bounds[0], bounds[-1]], "beats": _rel(bp),
                      "covered": [bounds[0], bounds[-1]], "measure_phase": phase,
                      "unit": {8: "measure", 4: "half measure", 2: "quarter measure"}[step],
                      "units": len(bounds) - 1, "measures": round((len(bounds) - 1) * step / 8, 3),
                      "segments": len(segs),
                      "profile": "grounded -> peak -> grounded (half sine over the bands)",
                      "walk": "links and blank-junction chains in either direction, forward preferred; beam "
                              "search with seeded noise",
                      "jump": "one deliberate transition without a graph link (exercises the new-link display)"},
    })
    return out


def cmd_placeholder(args: argparse.Namespace) -> int:
    graph = load_graph(Path(args.graph))
    dst = Path(args.out) if args.out else labels_path(args.take)
    if dst.exists() and not args.force:
        try:
            old = load_labels(dst)
            real = [s for s in old.segments if s.provenance != "placeholder"]
        except (LabelsError, KeyError, ValueError):
            real = ["unreadable"]
        if real:
            print(f"[moves] {dst} holds {len(real)} non-placeholder segments; refusing to overwrite (--force)")
            return 1
    span = tuple(args.span) if args.span else default_span(args.take)
    lab = make_placeholder(args.take, graph, seed=args.seed, span=span)
    seed = lab.extra["generator"]["seed"]
    p = save_labels(lab, dst, graph)
    print(f"[moves] wrote {p}: PLACEHOLDER, {len(lab.segments)} segments over "
          f"[{lab.segments[0].t0:.2f}, {lab.segments[-1].t1:.2f}] s (seed {seed})")
    for s in lab.segments:
        nd = graph.nodes[s.extra["placeholder_node"]]
        print(f"   {s.t0:8.3f} {s.t1:8.3f}  band {nd['band']} {nd['name']}")
    return 0


def register(sub) -> None:
    p = sub.add_parser("placeholder", help="PLACEHOLDER labels for the prototype (not an analysis)")
    p.add_argument("take")
    p.add_argument("--seed", type=int, default=None, help="default: crc32(take) %% 1000")
    p.add_argument("--span", type=float, nargs=2, metavar=("T0", "T1"),
                   help="reference seconds (default: the take's reconstructed window, else the whole beat grid)")
    p.add_argument("--graph", default=str(GRAPH_PATH))
    p.add_argument("--out")
    p.add_argument("--force", action="store_true", help="overwrite even non-placeholder labels")
    p.set_defaults(func=cmd_placeholder)
