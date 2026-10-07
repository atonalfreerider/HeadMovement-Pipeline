"""The zouk dance graph: import the user's Zouk1.json state machine, merge additions, load it for path/fingerprint.

    python -m dancecap.moves graph            -> taxonomy/zouk_graph.json (+ work/review/moves_graph.png)

Inputs (re-runnable; the output is regenerated from scratch each time):
  <workspace>/Zouk1.json                    the user's 3D node diagram (143 nodes, 181 directed links), authored in a
                                            Unity app: left-handed, y-up. Positions are kept exactly as authored.
  taxonomy/additions.toml                   user-editable: display names, aliases, proposed moves (Counterbalance
                                            Pivot, ...) and proposed links. Created with commented examples when
                                            missing. It wins over the vocabulary research.
  taxonomy/vocabulary_research.json         optional (written by the vocabulary research, format: its merge_rules):
                                            vocabulary[] rows with status 'scaffold' name an existing move by slug and
                                            add aliases + a Brazilian Portuguese display name (only when
                                            display_confidence is 'confirmed' or 'likely'); rows with status
                                            'proposed' become proposed moves (or only add aliases + display name when
                                            the slug already exists). A scaffold row whose slug is not a graph move
                                            becomes a proposed move (with a warning). proposed_links[], hypotheses[],
                                            do_not_add[] and graph_cleanup[] are NOT merged (questions for the user);
                                            unresolved_aliases[] is passed through for the narration step.

Names (VIEWER_SPEC 3.10): a move is labelled with its Brazilian Portuguese term where one is established, the user's
English Zouk1 name kept as an alias. "Lateral" is displayed as "Corredor" (user decision 2026-10-06); its slug stays
'lateral' so labels keep working, and 'corredor' resolves to the two Lateral nodes.

graph.json (version 1):
  {version, source:'Zouk1.json', coordinates:'zouk1 (Unity left-handed, y-up, as authored)',
   energy_bands:[{band, lo, hi, label, y_lo, y_hi, nodes}],
   nodes:[{id (Zouk1 uuid | 'proposed-<slug>'), name (display name = the move's name; '' for blank nodes), slug
           (unique per node, from the authored name), move (canonical move slug | null for the blank structural
           nodes), shape, tone [r,g,b], tone_name, family (from shape), position [x,y,z] (as authored), energy 0..1
           (normalised height), band, status scaffold|proposed|accepted,
           + authored_name (Zouk1 text as authored), display_name, instance, instances, in_degree, out_degree,
             isolated, blank, facing [x,y,z,w], size, aliases}],
   links:[{id, from, to, status scaffold|proposed, + from_move, to_move, accepted (proposed links the user
           accepted in additions.toml), source}],
   moves:[{slug, name (display name), aliases[], nodes[ids], status, + display_name, display_confidence,
           display_source (zouk1 | vocabulary_research | additions), name_en (the user's English name), family,
           energy, band, variants, description, sources[], source}],
   legend:{shapes:{<Shape>:{family, label (user-confirmed meaning), count, named, agreement, exceptions,
           evidence_note}}, tones:{<name>:{rgb, meaning (user-confirmed), count, mean_energy, shapes, examples,
           evidence_note}}, inferred:true, confirmed_by_user:true, confirmed:{date, source}},
   unresolved_aliases:[{alias, candidates[]}], y_range:[ymin, ymax] (energy normalisation, Zouk1 nodes only),
   display:{offset, note}, sources, warnings}

Blank (unnamed) Zouk1 nodes are structural junctions: a chain a -> blank+ -> b counts as a graph connection between
the named moves a and b (Graph.via, used by the path resolver and the placeholder walk).

Energy = (y - ymin) / (ymax - ymin) over the Zouk1 nodes (y = authored height; the user's layout puts grounded,
stationary states at the bottom and high-kinetic-energy moves at the top). Bands are equal-width in energy:
  0 grounded [0.0, 0.2)   1 low [0.2, 0.4)   2 travelling [0.4, 0.6)   3 high [0.6, 0.8)   4 peak [0.8, 1.0]
(for Zouk1: y cut points -0.718, -0.124, 0.471, 1.065; 34 / 43 / 36 / 19 / 11 nodes).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import tomllib
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from . import ADDITIONS_PATH, GRAPH_PATH, VOCABULARY_PATH, WORKSPACE, ZOUK1_PATH

VERSION = 1

ENERGY_BANDS = [  # (band, lo, hi, label) in normalised energy; equal width
    (0, 0.0, 0.2, "grounded"),
    (1, 0.2, 0.4, "low"),
    (2, 0.4, 0.6, "travelling"),
    (3, 0.6, 0.8, "high"),
    (4, 0.8, 1.0, "peak"),
]

TONES = {  # Zouk1 tones (exact authored values, rounded)
    "blue": (0.2471, 0.3176, 0.7098),
    "green": (0.2980, 0.6863, 0.3137),
    "yellow": (1.0, 0.9216, 0.2314),
    "orange": (1.0, 0.6, 0.0),
}

# Shape legend, CONFIRMED by the user on 2026-10-06 (VIEWER_SPEC 3.10): `label` is the user's meaning. `keywords` are
# the node-name words that fit that meaning (used only to measure `agreement` on the data); `evidence` notes what else
# the names show under that shape - evidence for the record, not a correction of the user's legend.
LEGEND_CONFIRMED = {"date": "2026-10-06", "source": "head-movement/docs/VIEWER_SPEC.md section 3.10"}
SHAPES = {
    "Cylinder": dict(
        family="rest", label="standing rest",
        keywords=["stand"],
        evidence="all named cylinders are stands"),
    "Hourglass": dict(
        family="hold", label="holds / embraces / isolations",
        keywords=["hug", "hold", "chair", "tilt", "lean", "body roll", "twist", "wrap", "hip", "shoulders",
                  "spring load", "stop", "raise", "lower", "dip", "project"],
        evidence="the names also include falls and drags, and turns started from the embrace (the yellow "
                 "hourglasses)"),
    "Diamond": dict(
        family="step", label="steps and basics",
        keywords=["basic", "step", "walk", "cross", "lunge", "launch", "back and forth", "180", "crab"],
        evidence="the names also include step-driven figures: small turns (viradinha), lunges, splits, dips"),
    "Plus": dict(
        family="open", label="open / hand-connection moves",
        keywords=["open", "hand", "arm", "switch", "lateral", "wing", "airplane", "cloak", "pin", "soltinho",
                  "wrap", "break", "walk", "bonus"],
        evidence="corredor (lateral), soltinho, hand switches, wings"),
    "Star": dict(
        family="turn", label="turns and spins",
        keywords=["turn", "spin", "pirouette", "unwind", "unwrap", "unwined", "roll", "boomerang", "downswing"],
        evidence="incl. pirouettes, boomerang, downswing"),
    "Tetra": dict(
        family="head", label="head movements (howls) and dips",
        keywords=["howl", "dip", "head"],
        evidence="the names also include behind-the-follower / overhead positions (gostosinho, reposition "
                 "behind, overhead springload)"),
    "Ball": dict(
        family="wave", label="special wave moves",
        keywords=["wave", "yoyo"],
        evidence="yoyo, do the wave, plus overhead/reverse specials"),
}
FAMILY_SHAPE = {v["family"]: k for k, v in SHAPES.items()}

TONE_MEANING = {  # confirmed by the user on 2026-10-06 (VIEWER_SPEC 3.10)
    "blue": "connection / position states",
    "green": "moves",
    "yellow": "advanced / high-commitment moves",
    "orange": "advanced / high-commitment moves",
}
TONE_EVIDENCE = {
    "blue": "embraces, holds, tilts, stands, open break, hand positions",
    "green": "the bulk of the vocabulary",
    "yellow": "mostly turns and pirouettes, plus leg split and drag",
    "orange": "the top of the graph: downswing, fast pirouette",
}

# display names from the vocabulary research are used only at these confidences ('translation' keeps the user's name)
DISPLAY_CONFIDENCES = ("confirmed", "likely")

SCAFFOLD, PROPOSED, ACCEPTED = "scaffold", "proposed", "accepted"
NODE_STATUSES = (SCAFFOLD, PROPOSED, ACCEPTED)
LINK_STATUSES = (SCAFFOLD, PROPOSED)  # contract; a link the user accepted stays 'proposed' with accepted:true


# ---------------------------------------------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------------------------------------------

def slugify(text: str | None) -> str:
    """'Lateral From Behind' -> 'lateral-from-behind'; accents dropped ('Pião' -> 'piao'); '' for blank."""
    if not text:
        return ""
    t = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", t.lower()).strip("-")


def tone_name(rgb) -> str:
    rgb = np.asarray(rgb[:3], float)
    best = min(TONES, key=lambda k: float(np.linalg.norm(rgb - np.asarray(TONES[k]))))
    return best if float(np.linalg.norm(rgb - np.asarray(TONES[best]))) < 0.08 else "other"


def band_of(energy: float) -> int:
    for b, lo, hi, _ in ENERGY_BANDS:
        if energy < hi:
            return b
    return ENERGY_BANDS[-1][0]


def band_label(band: int) -> str:
    return ENERGY_BANDS[band][3]


def _sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.exists() else None


def _rel(path) -> str:
    """Workspace-relative posix path when inside the workspace (the graph file is tracked: no machine paths)."""
    try:
        return Path(path).resolve().relative_to(WORKSPACE.resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()


def _as_list(v) -> list:
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return [x for x in v if x not in (None, "")]
    return [v]


def _canonical_name(variants: list[str]) -> str:
    """Most common authored spelling; ties -> the one with more capitals ('Tight Walk' over 'tight Walk')."""
    c = Counter(variants)
    return max(c, key=lambda s: (c[s], sum(ch.isupper() for ch in s), s))


def _display_name(text: str) -> str:
    """'howl Down' -> 'Howl Down' (canonical move names only; node names stay as authored)."""
    return " ".join(w[:1].upper() + w[1:] for w in text.split())


def proposed_link_id(a: str, b: str) -> str:
    return "proposed-" + hashlib.sha1(f"{a}|{b}".encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------------------------------------------
# Zouk1 import
# ---------------------------------------------------------------------------------------------------------------

def read_zouk1(path: Path = ZOUK1_PATH) -> tuple[list[dict], list[dict]]:
    """Raw Zouk1 nodes [{uuid, name, position, facing, tone, size, shape}] and links [{uuid, from, to}]."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if d.get("format") != "v2":
        raise ValueError(f"{path}: expected Zouk1 format v2, got {d.get('format')!r}")
    nodes = []
    for n in d["nodes"]:
        name = next((p.get("text") or "" for p in (n.get("properties") or []) if p.get("name") == "Name"), "")
        p, f, t = n["position"], n.get("facing") or {}, n["tone"]
        nodes.append({"uuid": n["uuid"], "name": name.strip(), "position": [p["x"], p["y"], p["z"]],
                      "facing": [f.get("x", 0.0), f.get("y", 0.0), f.get("z", 0.0), f.get("w", 1.0)],
                      "tone": [t["r"], t["g"], t["b"]], "size": n.get("size"), "shape": n["shape"]})
    links = [{"uuid": l["uuid"], "from": l["fromNode"]["Uuid"], "to": l["toNode"]["Uuid"]} for l in d["links"]]
    return nodes, links


def infer_legend(nodes: list[dict]) -> dict:
    """The user-confirmed shape/tone legend + evidence from the node names (authored names, Zouk1 nodes)."""
    def nm(n):
        return n.get("authored_name", n["name"])
    shapes = {}
    for shape, spec in SHAPES.items():
        named = [n for n in nodes if n["shape"] == shape and nm(n)]
        hits = [nm(n) for n in named if any(k in nm(n).lower() for k in spec["keywords"])]
        exc = sorted({nm(n) for n in named} - set(hits))
        agreement = round(len(hits) / len(named), 3) if named else None
        shapes[shape] = {"family": spec["family"], "label": spec["label"], "confirmed": True,
                         "count": sum(n["shape"] == shape for n in nodes), "named": len(named),
                         "agreement": agreement, "exceptions": exc, "evidence_note": spec["evidence"]}
    tones = {}
    for name, rgb in TONES.items():
        sel = [n for n in nodes if n["tone_name"] == name]
        tones[name] = {"rgb": list(rgb), "meaning": TONE_MEANING[name], "confirmed": True,
                       "count": len(sel),
                       "mean_energy": round(float(np.mean([n["energy"] for n in sel])), 3) if sel else None,
                       "shapes": dict(Counter(n["shape"] for n in sel).most_common()),
                       "examples": sorted({nm(n) for n in sel if nm(n)})[:12],
                       "evidence_note": TONE_EVIDENCE[name]}
    return {"shapes": shapes, "tones": tones, "inferred": True, "confirmed_by_user": True,
            "confirmed": dict(LEGEND_CONFIRMED),
            "note": "Shape and tone meanings as confirmed by the user on 2026-10-06 (VIEWER_SPEC 3.10). agreement = "
                    "share of named nodes of that shape whose authored name fits the meaning; exceptions and "
                    "evidence_note are evidence only. Corrections go in taxonomy/additions.toml ([legend] table)."}


# ---------------------------------------------------------------------------------------------------------------
# additions + vocabulary research
# ---------------------------------------------------------------------------------------------------------------

ADDITIONS_TEMPLATE = '''# Additions to the zouk dance graph (taxonomy/zouk_graph.json is regenerated from Zouk1.json + this file
# + taxonomy/vocabulary_research.json). This file wins over the vocabulary research.
# Edit freely, then run:   python -m dancecap.moves graph
#
# Moves are referred to by slug (lower-case, hyphens: "Lateral From Behind" -> "lateral-from-behind"), by an alias,
# or by a node id (Zouk1 uuid) when one specific instance of a duplicated move is meant.
# Energy is 0..1 = graph height: 0-0.2 grounded, 0.2-0.4 low, 0.4-0.6 travelling, 0.6-0.8 high, 0.8-1 peak.
# Legend (confirmed 2026-10-06): Cylinder standing rest, Hourglass holds / embraces / isolations, Diamond steps and
# basics, Plus open / hand-connection moves, Star turns and spins, Tetra head movements (howls) and dips, Ball special
# wave moves. Families for proposed moves: rest, hold, step, open, turn, head, wave (= those shapes).
# Tones: blue (connection / position state), green (move), yellow / orange (advanced, high-commitment).

version = 1

# ---- display names: the label shown in the viewer (Brazilian Portuguese where established; the English Zouk1
# name stays an alias). Wins over the vocabulary research. ------------------------------------------------------
[display_names]
lateral = "Corredor"                # user decision 2026-10-06: "Lateral" is renamed Corredor (slug stays 'lateral')

# ---- aliases: other names (spellings, Portuguese terms, what you say while narrating) for existing moves --------
[aliases]
lateral = ["corredor", "corridor"]
viradinha = ["veraginha", "viradinha turn"]
double-unwined-turn = ["double unwind turn"]
howl-down = ["head movement down"]
# soltinho = ["solto"]

# ---- proposed moves: not in Zouk1 yet; they appear as dashed "proposed" nodes until status = "accepted" ---------
[[moves]]
slug = "counterbalance-pivot"
name = "Counterbalance Pivot"       # shown as "Contrapeso" (vocabulary research, likely); set display_name to override
aliases = ["counterbalance", "counter balance pivot", "pivot around"]
family = "turn"                     # the follower rotates in place (Star)
tone = "yellow"                     # high commitment: both dancers lean out and hold each other in tension
energy = 0.62                       # vocabulary research estimate
near = ["walk-around", "hug", "pirouette"]
links_from = ["lateral", "open-break", "hug"]   # proposed links  <move> -> counterbalance-pivot
links_to = ["walk-around", "pirouette"]         # proposed links  counterbalance-pivot -> <move>
status = "proposed"
description = "The leader anchors the follower's support foot at one point and walks a full circle around her at arm's length; both lean away and hold each other in tension (shared centre of mass between them). Researched name: contrapeso (the pivot itself is a planada); the pião is a different move (both rotate in closed frame)."

# [[moves]]
# slug = "my-new-move"
# name = "My New Move"
# display_name = "Nome em Português"
# family = "open"
# energy = 0.55
# near = ["soltinho"]
# status = "proposed"               # "accepted" once you are happy with it

# ---- proposed links between existing moves (dashed until accepted) --------------------------------------------
# Suggestions from the vocabulary research (Yoyo is isolated in Zouk1) - uncomment to add:
# [[links]]
# from = "lateral"
# to = "yoyo"
# [[links]]
# from = "yoyo"
# to = "open-break"
# [[links]]
# from = "yoyo"
# to = "simple-turn"
# status = "proposed"               # "accepted" once you are happy with it

# ---- legend corrections (shape / tone meanings), shown in the viewer -----------------------------------------
# [legend.shapes.Tetra]
# label = "head movements and dips"
# [legend.tones.blue]
# meaning = "connection states"
'''


def ensure_additions(path: Path = ADDITIONS_PATH) -> bool:
    """Create the additions file (Corredor display name, Counterbalance Pivot, commented examples). True if created."""
    path = Path(path)
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(ADDITIONS_TEMPLATE, encoding="utf-8")
    return True


def read_additions(path: Path = ADDITIONS_PATH) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _energy_hint(v) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(min(1.0, max(0.0, v)))
    s = slugify(str(v))
    for b, lo, hi, label in ENERGY_BANDS:
        if s == label or s == f"band-{b}":
            return round((lo + hi) / 2, 3)
    try:
        return float(min(1.0, max(0.0, float(v))))
    except ValueError:
        return None


def _display_of(e: dict, source: str) -> dict | None:
    """Display-name info of an entry: {display_name, display_confidence, display_source} or None.

    Vocabulary research: only display_confidence 'confirmed' / 'likely' (a missing confidence counts as given).
    additions.toml: always (the user's own choice, confidence 'user')."""
    dn = e.get("display_name") or e.get("name_pt")
    if not dn:
        return None
    conf = e.get("display_confidence")
    if source == "additions":
        conf = conf or "user"
    elif conf is not None and conf not in DISPLAY_CONFIDENCES:
        return None
    return {"display_name": str(dn), "display_confidence": conf or "given", "display_source": source}


def _norm_move_entry(e: dict, source: str) -> dict | None:
    """Tolerant normalisation of a proposed-move entry from additions.toml or vocabulary_research.json."""
    name = e.get("name") or e.get("term") or e.get("move")
    slug = slugify(e.get("slug") or name)
    if not slug:
        return None
    family = e.get("family") or e.get("family_hint")
    shape = e.get("shape") or FAMILY_SHAPE.get(family or "", None)
    if family is None and shape in SHAPES:
        family = SHAPES[shape]["family"]
    ph = e.get("position_hint")
    return {
        "slug": slug, "name": name or slug.replace("-", " ").title(),
        "aliases": [str(a) for a in _as_list(e.get("aliases") or e.get("alias") or e.get("spellings"))],
        "family": family if family in FAMILY_SHAPE else None,
        "shape": shape if shape in SHAPES else None,
        "tone": e.get("tone") if e.get("tone") in TONES else None,
        "energy": _energy_hint(e.get("energy", e.get("energy_hint"))),
        "near": [str(x) for x in _as_list(e.get("near") or e.get("neighbours") or e.get("neighbors")
                                         or e.get("near_moves") or e.get("related"))],
        "links_from": [str(x) for x in _as_list(e.get("links_from") or e.get("from"))],
        "links_to": [str(x) for x in _as_list(e.get("links_to") or e.get("to"))],
        "status": e.get("status") if e.get("status") in (PROPOSED, ACCEPTED) else PROPOSED,
        "description": e.get("description") or e.get("definition") or e.get("notes"),
        "position_hint": [float(x) for x in ph] if isinstance(ph, (list, tuple)) and len(ph) == 3 else None,
        "display": _display_of(e, source),
        "sources": [str(x) for x in _as_list(e.get("sources"))],
        "source": source,
    }


def read_vocabulary(path: Path = VOCABULARY_PATH) -> dict:
    """Aliases, display names and proposed moves from the vocabulary research (format tolerant).

    Research format (vocabulary_research.json version 1, see its merge_rules): vocabulary[] rows
    {slug, name, display_name, display_confidence, aliases, zouk1_name, zouk1_matches, status 'scaffold'|'proposed',
    energy_hint, family|family_hint, tone, near, links_from, links_to, position_hint, description, sources}.
    Also accepted: {"aliases": {<move>: [..]} | [{move|slug|canonical, aliases}]},
    {"proposed_moves" | "proposed" | "proposals": [...]}, and rows under "terms" | "moves" | "entries" with
    graph_move | maps_to | canonical_slug | canonical | slug as the target.

    Returns {aliases: {target: [names]}, rows: [scaffold rows {target, names, display, description, sources,
    zouk1_matches, entry}], moves: [proposed entries], unresolved_aliases: [...], present}. Rows are resolved
    against the graph in build_graph (a row whose target is not a graph move becomes a proposed move).
    """
    path = Path(path)
    out: dict = {"aliases": defaultdict(list), "rows": [], "moves": [], "unresolved_aliases": [],
                 "present": path.exists()}
    if not path.exists():
        return out
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return out
    src = "vocabulary_research"
    al = data.get("aliases")
    if isinstance(al, dict):
        for k, v in al.items():
            out["aliases"][str(k)].extend(str(x) for x in _as_list(v))
    elif isinstance(al, list):
        for e in al:
            if isinstance(e, dict):
                key = e.get("move") or e.get("slug") or e.get("canonical") or e.get("graph_move")
                if key:
                    out["aliases"][str(key)].extend(str(x) for x in _as_list(e.get("aliases") or e.get("alias")))
    for key in ("proposed_moves", "proposed", "proposals"):
        for e in data.get(key) or []:
            if isinstance(e, dict) and (m := _norm_move_entry(e, src)):
                out["moves"].append(m)
    for key in ("vocabulary", "terms", "moves", "entries"):
        for e in data.get(key) or []:
            if not isinstance(e, dict):
                continue
            proposed = e.get("status") == PROPOSED or e.get("propose") is True or e.get("proposed") is True
            if proposed:
                if m := _norm_move_entry(e, src):
                    out["moves"].append(m)
                continue
            target = (e.get("graph_move") or e.get("maps_to") or e.get("canonical_slug") or e.get("canonical")
                      or e.get("slug"))
            if not target:
                continue
            names = (_as_list(e.get("name")) + _as_list(e.get("term")) + _as_list(e.get("zouk1_name"))
                     + _as_list(e.get("aliases")))
            out["rows"].append({"target": str(target), "names": [str(x) for x in names],
                                "display": _display_of(e, src),
                                "description": e.get("description") or e.get("definition"),
                                "sources": [str(x) for x in _as_list(e.get("sources"))],
                                "zouk1_matches": [str(x) for x in _as_list(e.get("zouk1_matches"))],
                                "entry": e})
    for u in data.get("unresolved_aliases") or []:
        if isinstance(u, dict) and u.get("alias"):
            cands = [c.get("term") if isinstance(c, dict) else str(c) for c in _as_list(u.get("candidates"))]
            out["unresolved_aliases"].append({"alias": str(u["alias"]), "candidates": [c for c in cands if c],
                                              "source": src})
    return out


# ---------------------------------------------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------------------------------------------

def _place(target_y: float, near_ids: list[str], nodes: list[dict], band: int,
           hint: list[float] | None = None) -> list[float]:
    """Position for a proposed node: at the energy height, beside its neighbours (or at the research's position_hint
    x/z when given), away from existing nodes."""
    P = np.array([n["position"] for n in nodes], float)
    if hint is not None:
        c = np.array([hint[0], target_y, hint[2]], float)
        if not len(P) or float(np.min(np.linalg.norm(P - c, axis=1))) >= 0.2:
            return [round(float(x), 6) for x in c]
        cxz = np.array([hint[0], hint[2]], float)
    elif near_ids:
        anchors = []
        for group in near_ids:
            cand = [n for n in nodes if n["id"] in group]
            if cand:
                anchors.append(min(cand, key=lambda n: abs(n["position"][1] - target_y))["position"])
        cxz = np.mean([[p[0], p[2]] for p in anchors], 0) if anchors else None
    else:
        cxz = None
    if cxz is None:
        same = [n["position"] for n in nodes if n["band"] == band] or [n["position"] for n in nodes]
        cxz = np.mean([[p[0], p[2]] for p in same], 0)
    best, best_d = None, -1.0
    for r in (0.25, 0.35, 0.5, 0.7):
        for k in range(12):
            a = math.radians(30 * k)
            c = np.array([cxz[0] + r * math.cos(a), target_y, cxz[1] + r * math.sin(a)])
            d = float(np.min(np.linalg.norm(P - c, axis=1))) if len(P) else 9.0
            if d > best_d + 1e-9:
                best, best_d = c, d
        if best_d >= 0.2:
            break
    return [round(float(x), 6) for x in best]


def build_graph(zouk1: Path = ZOUK1_PATH, additions: Path | None = ADDITIONS_PATH,
                vocabulary: Path | None = VOCABULARY_PATH) -> dict:
    """Zouk1.json (+ additions.toml + vocabulary_research.json) -> graph dict (graph.json contract, version 1)."""
    raw_nodes, raw_links = read_zouk1(zouk1)
    warnings: list[str] = []
    ys = np.array([n["position"][1] for n in raw_nodes], float)
    ymin, ymax = float(ys.min()), float(ys.max())

    def energy_of_y(y: float) -> float:
        return float(min(1.0, max(0.0, (y - ymin) / (ymax - ymin))))

    def y_of_energy(e: float) -> float:
        return ymin + e * (ymax - ymin)

    indeg = Counter(l["to"] for l in raw_links)
    outdeg = Counter(l["from"] for l in raw_links)
    ids = {n["uuid"] for n in raw_nodes}
    for l in raw_links:
        if l["from"] not in ids or l["to"] not in ids:
            warnings.append(f"Zouk1 link {l['uuid']} refers to a missing node")

    # canonical moves: group nodes by slug(name); blank names stay structural (move null)
    groups: dict[str, list[dict]] = defaultdict(list)
    for n in raw_nodes:
        s = slugify(n["name"])
        if s:
            groups[s].append(n)
    nodes: list[dict] = []
    for n in raw_nodes:
        e = energy_of_y(n["position"][1])
        nodes.append({
            "id": n["uuid"], "name": n["name"], "authored_name": n["name"], "slug": None,
            "move": slugify(n["name"]) or None,
            "shape": n["shape"], "tone": [round(float(x), 4) for x in n["tone"]], "tone_name": tone_name(n["tone"]),
            "family": SHAPES.get(n["shape"], {}).get("family", "other"),
            "position": [round(float(x), 6) for x in n["position"]],
            "energy": round(e, 4), "band": band_of(e), "status": SCAFFOLD,
            "instance": 1, "instances": 1, "in_degree": indeg[n["uuid"]], "out_degree": outdeg[n["uuid"]],
            "isolated": indeg[n["uuid"]] + outdeg[n["uuid"]] == 0, "blank": not n["name"],
            "facing": [round(float(x), 6) for x in n["facing"]], "size": n["size"],
        })
    by_id = {n["id"]: n for n in nodes}
    for s, members in groups.items():
        # instance order: by height, then x, z (deterministic); node slug = move slug (+ -2, -3 for later instances)
        ordered = sorted(members, key=lambda m: (m["position"][1], m["position"][0], m["position"][2]))
        for k, m in enumerate(ordered, 1):
            nd = by_id[m["uuid"]]
            nd["instance"], nd["instances"] = k, len(ordered)
            nd["slug"] = s if k == 1 else f"{s}-{k}"
    for nd in nodes:
        if nd["blank"]:
            nd["slug"] = f"unnamed-{nd['id'][:8]}"
    slugs = Counter(nd["slug"] for nd in nodes)
    for s, c in slugs.items():
        if c > 1:  # e.g. a move named "lateral-2" next to a second "Lateral"
            warnings.append(f"node slug {s!r} is not unique ({c} nodes)")

    moves: dict[str, dict] = {}
    for s, members in sorted(groups.items()):
        ids_ = [m["uuid"] for m in sorted(members, key=lambda m: (m["position"][1], m["position"][0]))]
        fams = {by_id[i]["family"] for i in ids_}
        en = float(np.mean([by_id[i]["energy"] for i in ids_]))
        variants = sorted({m["name"] for m in members})
        name_en = _display_name(_canonical_name([m["name"] for m in members]))
        moves[s] = {"slug": s, "name": name_en, "aliases": [], "nodes": ids_, "status": SCAFFOLD,
                    "family": fams.pop() if len(fams) == 1 else None, "energy": round(en, 4), "band": band_of(en),
                    "variants": variants, "description": None, "source": "zouk1", "sources": [],
                    "name_en": name_en}

    links = [{"id": l["uuid"], "from": l["from"], "to": l["to"], "status": SCAFFOLD,
              "from_move": by_id[l["from"]]["move"] if l["from"] in by_id else None,
              "to_move": by_id[l["to"]]["move"] if l["to"] in by_id else None} for l in raw_links]

    # ---- merge additions + vocabulary research -------------------------------------------------------------
    # additions.toml wins over the research. Order: additions moves, additions aliases, research proposed moves,
    # research scaffold rows (aliases + display names), research legacy aliases, Zouk1 spelling variants.
    # Display names: the user's English Zouk1 name < research (confirmed / likely) < additions (display_name).
    add = read_additions(additions) if additions else {}
    vocab = (read_vocabulary(vocabulary) if vocabulary else
             {"aliases": {}, "rows": [], "moves": [], "unresolved_aliases": [], "present": False})
    alias_index: dict[str, str] = {s: s for s in moves}  # slug(alias) -> move slug
    display: dict[str, dict] = {}  # move slug -> {display_name, display_confidence, display_source}

    def resolve_move(ref: str) -> str | None:
        s = slugify(ref)
        if s in moves:
            return s
        return alias_index.get(s)

    def resolve_nodes(ref: str) -> list[str]:
        if ref in by_id:
            return [ref]
        m = resolve_move(ref)
        return list(moves[m]["nodes"]) if m else []

    def add_alias(move: str, alias: str, source: str) -> None:
        a = slugify(alias)
        if not a or a == move:
            return
        if a in moves and a != move:
            warnings.append(f"alias {alias!r} ({source}) is itself the move {a!r}; not added to {move!r}")
            return
        if alias_index.get(a, move) != move:
            warnings.append(f"alias {alias!r} ({source}) already names {alias_index[a]!r}; not added to {move!r}")
            return
        alias_index[a] = move
        if alias not in moves[move]["aliases"] and slugify(alias) not in {slugify(x) for x in moves[move]["aliases"]}:
            moves[move]["aliases"].append(alias)

    def describe(move: str, text: str | None, sources: list[str]) -> None:
        if text and not moves[move].get("description"):
            moves[move]["description"] = text
        for s in sources or []:
            if s not in moves[move]["sources"]:
                moves[move]["sources"].append(s)

    created: list[dict] = []  # proposed entries that made a node (their links are added below)

    def create_proposed(e: dict) -> None:
        energy = e["energy"] if e["energy"] is not None else 0.4
        band = band_of(energy)
        near_groups = []
        for ref in e["near"] or (e["links_from"] + e["links_to"]):
            ns = resolve_nodes(ref)
            if ns:
                near_groups.append(ns)
            else:
                warnings.append(f"proposed move {e['slug']!r}: neighbour {ref!r} is not in the graph")
        shape = e["shape"] or FAMILY_SHAPE.get(e["family"] or "", "Diamond")
        tone = e["tone"] or "green"
        nid = f"proposed-{e['slug']}"
        node = {"id": nid, "name": e["name"], "slug": e["slug"], "move": e["slug"], "shape": shape,
                "tone": list(TONES[tone]), "tone_name": tone, "family": SHAPES[shape]["family"],
                "position": _place(y_of_energy(energy), near_groups, nodes, band, e.get("position_hint")),
                "energy": round(energy, 4), "band": band, "status": e["status"], "authored_name": None,
                "instance": 1, "instances": 1, "in_degree": 0, "out_degree": 0, "isolated": True, "blank": False,
                "facing": [0.0, 0.0, 0.0, 1.0], "size": 5.0, "source": e["source"]}
        nodes.append(node)
        by_id[nid] = node
        moves[e["slug"]] = {"slug": e["slug"], "name": e["name"], "aliases": [], "nodes": [nid],
                            "status": e["status"], "family": node["family"], "energy": round(energy, 4),
                            "band": band, "variants": [e["name"]], "description": e["description"],
                            "source": e["source"], "near": e["near"], "sources": [],
                            "name_en": e["name"] if e["source"] == "additions" else None}
        alias_index.setdefault(e["slug"], e["slug"])
        for a in e["aliases"]:
            add_alias(e["slug"], a, e["source"])
        describe(e["slug"], None, e["sources"])
        if e["display"]:
            display[e["slug"]] = e["display"]
        e["_node"] = nid
        created.append(e)

    def take_entry(e: dict) -> None:
        """A proposed-move entry: a new proposed node, or only aliases / display name / description when the slug
        (or one of its aliases) already names a move."""
        existing = resolve_move(e["slug"]) or next((resolve_move(a) for a in e["aliases"] if resolve_move(a)), None)
        if existing:
            for a in [e["name"], *e["aliases"]]:
                add_alias(existing, a, e["source"])
            describe(existing, e["description"], e["sources"])
            if e["display"]:
                display[existing] = e["display"]
            return
        create_proposed(e)

    add_entries = [_norm_move_entry(e, "additions") for e in add.get("moves", []) if isinstance(e, dict)]
    add_entries = [e for e in add_entries if e]
    for e in add_entries:                                               # 1 additions moves
        take_entry(e)
    deferred = []
    for key, vals in (add.get("aliases") or {}).items():               # 2 additions aliases
        m = resolve_move(key)
        if m is None:
            deferred.append((key, vals))  # may name a move that only the research proposes
            continue
        for a in _as_list(vals):
            add_alias(m, str(a), "additions")

    def merge_move(src: str, dst: str) -> None:
        """additions [same_move] src = dst: the user says src is the same move as dst, so src's nodes become
        instances of dst (node ids and node slugs stay; the move slug src resolves to dst from now on)."""
        ms, md = moves.pop(src), moves[dst]
        for i in ms["nodes"]:
            by_id[i]["move"] = dst
        md["nodes"] = sorted(md["nodes"] + ms["nodes"],
                             key=lambda i: (by_id[i]["position"][1], by_id[i]["position"][0]))
        for k, i in enumerate(md["nodes"], 1):
            by_id[i]["instance"], by_id[i]["instances"] = k, len(md["nodes"])
        fams = {by_id[i]["family"] for i in md["nodes"]}
        md["family"] = fams.pop() if len(fams) == 1 else None
        md["energy"] = round(float(np.mean([by_id[i]["energy"] for i in md["nodes"]])), 4)
        md["band"] = band_of(md["energy"])
        md["variants"] = sorted(set(md["variants"]) | set(ms["variants"]))
        for k, v in list(alias_index.items()):
            if v == src:
                alias_index[k] = dst
        alias_index[src] = dst
        for a in [ms["name"], ms.get("name_en") or "", *ms["aliases"]]:
            add_alias(dst, a, "additions")
        describe(dst, ms.get("description"), ms.get("sources") or [])
        for l in links:
            for end in ("from_move", "to_move"):
                if l[end] == src:
                    l[end] = dst
        display.pop(src, None)

    for key, target in (add.get("same_move") or {}).items():           # 2b additions: same move
        src, dst = resolve_move(key), resolve_move(str(target))
        if src is None or dst is None or src == dst:
            warnings.append(f"same_move {key!r} = {target!r} (additions): unknown move or same move; skipped")
            continue
        merge_move(src, dst)
    for e in vocab["moves"]:                                            # 3 research proposed moves
        take_entry(e)
    for r in vocab["rows"]:                                             # 4 research scaffold rows
        m = resolve_move(r["target"])
        if m is None:
            e = _norm_move_entry(r["entry"], "vocabulary_research")
            if e:
                e["status"] = PROPOSED
                warnings.append(f"vocabulary row {r['target']!r} names no graph move: added as the proposed move "
                                f"{e['slug']!r} (review it)")
                take_entry(e)
            continue
        miss = [u for u in r["zouk1_matches"] if u not in moves[m]["nodes"]]
        if miss:
            warnings.append(f"vocabulary row {r['target']!r}: zouk1_matches {miss} are not nodes of {m!r}")
        for a in r["names"]:
            add_alias(m, a, "vocabulary_research")
        describe(m, r["description"], r["sources"])
        if r["display"]:
            display[m] = r["display"]
    for key, vals, source in ([(k, v, "additions") for k, v in deferred]   # 5 deferred + research legacy aliases
                              + [(k, v, "vocabulary_research") for k, v in vocab["aliases"].items()]):
        m = resolve_move(key)
        if m is None:
            warnings.append(f"aliases for unknown move {key!r} ({source}) skipped")
            continue
        for a in _as_list(vals):
            add_alias(m, str(a), source)
    for e in add_entries:                                               # 6 additions display names win
        m = resolve_move(e["slug"])
        if m and e["display"]:
            display[m] = e["display"]
    for key, dn in (add.get("display_names") or {}).items():
        m = resolve_move(key)
        if m is None:
            warnings.append(f"display name for unknown move {key!r} (additions) skipped")
            continue
        display[m] = {"display_name": str(dn), "display_confidence": "user", "display_source": "additions"}
    for s, m in moves.items():                                          # 7 authored spelling variants as aliases
        for v in m["variants"]:
            if v != m["name"]:
                add_alias(s, v, "zouk1")
    for s, m in moves.items():                                          # 8 move name = display name
        d = display.get(s)
        if d:
            m["name"] = d["display_name"]
            add_alias(s, d["display_name"], d["display_source"])
        m["display_name"] = m["name"]
        m["display_confidence"] = d["display_confidence"] if d else None
        m["display_source"] = d["display_source"] if d else m["source"]
        en = m.get("name_en")
        if en and en != m["name"] and en not in m["aliases"]:  # the user's English name stays an alias
            m["aliases"].insert(0, en)
            alias_index.setdefault(slugify(en), s)

    link_keys = {(l["from"], l["to"]) for l in links}

    def degree(i: str) -> int:
        return sum(l["from"] == i or l["to"] == i for l in links)

    def add_link(a_ref: str, b_ref: str, status: str, source: str) -> None:
        A, B = resolve_nodes(a_ref), resolve_nodes(b_ref)
        if not A or not B:
            warnings.append(f"proposed link {a_ref!r} -> {b_ref!r} ({source}): unknown move/node")
            return
        # one link between instances: prefer instances that are connected in the graph (an isolated instance
        # would make the proposed node reachable only from a node nothing else reaches), then the closest pair
        a, b = min(((x, y) for x in A for y in B if x != y),
                   key=lambda p: ((degree(p[0]) == 0) + (degree(p[1]) == 0),
                                  float(np.linalg.norm(np.subtract(by_id[p[0]]["position"],
                                                                   by_id[p[1]]["position"])))),
                   default=(None, None))
        if a is None:
            return
        if (a, b) in link_keys:
            return
        link_keys.add((a, b))
        links.append({"id": proposed_link_id(a, b), "from": a, "to": b, "status": PROPOSED,
                      "accepted": status == ACCEPTED,
                      "from_move": by_id[a]["move"], "to_move": by_id[b]["move"], "source": source})

    for e in created:
        for ref in e["links_from"]:
            add_link(ref, e["_node"], PROPOSED, e["source"])
        for ref in e["links_to"]:
            add_link(e["_node"], ref, PROPOSED, e["source"])
    for l in add.get("links", []) or []:
        if isinstance(l, dict) and l.get("from") and l.get("to"):
            add_link(str(l["from"]), str(l["to"]), l.get("status", PROPOSED), "additions")

    # degrees after the merge; node names = the move's display name (authored_name keeps the Zouk1 text)
    indeg = Counter(l["to"] for l in links)
    outdeg = Counter(l["from"] for l in links)
    for n in nodes:
        n["in_degree"], n["out_degree"] = indeg[n["id"]], outdeg[n["id"]]
        n["isolated"] = n["in_degree"] + n["out_degree"] == 0
        n["aliases"] = list(moves[n["move"]]["aliases"]) if n["move"] in moves else []  # copy of the move's
        if n["move"] in moves:
            n["name"] = moves[n["move"]]["name"]
        n["display_name"] = n["name"]

    legend = infer_legend([n for n in nodes if n["status"] == SCAFFOLD])
    for kind in ("shapes", "tones"):  # user corrections from additions [legend.shapes.X] / [legend.tones.Y]
        for k, v in ((add.get("legend") or {}).get(kind) or {}).items():
            if k in legend[kind] and isinstance(v, dict):
                legend[kind][k].update({kk: vv for kk, vv in v.items() if kk in ("label", "meaning", "family")})
                legend[kind][k]["user_corrected"] = True

    scaffold = [n for n in nodes if n["status"] == SCAFFOLD]
    bands = []
    for b, lo, hi, label in ENERGY_BANDS:
        bands.append({"band": b, "lo": lo, "hi": hi, "label": label,
                      "y_lo": round(y_of_energy(lo), 4), "y_hi": round(y_of_energy(hi), 4),
                      "nodes": sum(n["band"] == b for n in scaffold)})
    P = np.array([n["position"] for n in scaffold], float)
    cx, cz = (P[:, 0].min() + P[:, 0].max()) / 2, (P[:, 2].min() + P[:, 2].max()) / 2
    return {
        "version": VERSION,
        "source": "Zouk1.json",
        "coordinates": "zouk1 (Unity left-handed, y-up, as authored)",
        "energy_bands": bands,
        "nodes": nodes,
        "links": links,
        "moves": sorted(moves.values(), key=lambda m: (m["status"] != SCAFFOLD, m["slug"])),
        "legend": legend,
        "unresolved_aliases": [u for u in vocab["unresolved_aliases"] if resolve_move(u["alias"]) is None],
        "y_range": [round(ymin, 6), round(ymax, 6)],
        "energy": "energy = (y - y_range[0]) / (y_range[1] - y_range[0]); bands equal-width in energy",
        "display": {"offset": [round(-cx, 4), round(0.2 - ymin, 4), round(-cz, 4)], "scale": 1.0,
                    "note": "suggested viewer placement (VIEWER_SPEC 3.10): position + offset centres the graph on "
                            "the origin with its lowest node 0.2 m above the floor; positions themselves stay as "
                            "authored"},
        "counts": {"nodes": len(nodes), "links": len(links), "scaffold_nodes": len(scaffold),
                   "scaffold_links": sum(l["status"] == SCAFFOLD for l in links),
                   "proposed_nodes": len(nodes) - len(scaffold),
                   "proposed_links": sum(l["status"] != SCAFFOLD for l in links),
                   "moves": len(moves), "blank_nodes": sum(n["blank"] for n in nodes),
                   "isolated_nodes": sum(n["isolated"] for n in nodes),
                   "duplicated_moves": sum(len(m["nodes"]) > 1 for m in moves.values()),
                   "display_names": sum(m["display_source"] != m["source"] or m["name"] != m.get("name_en")
                                        for m in moves.values() if m.get("name_en")),
                   "via_links": len(_via_index(nodes, links))},
        "sources": {"zouk1": {"path": _rel(zouk1), "sha256_16": _sha(Path(zouk1))},
                    "additions": {"path": _rel(additions), "sha256_16": _sha(Path(additions))} if additions else None,
                    "vocabulary_research": ({"path": _rel(vocabulary), "sha256_16": _sha(Path(vocabulary))}
                                            if vocabulary and Path(vocabulary).exists() else None)},
        "warnings": warnings,
    }


def write_graph(graph: dict, path: Path = GRAPH_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps_graph(graph), encoding="utf-8")
    return path


def dumps_graph(graph: dict) -> str:
    """JSON with one node / link / move / band per line (the file is tracked: readable diffs)."""
    parts = []
    for k, v in graph.items():
        if k in ("nodes", "links", "moves", "energy_bands") and isinstance(v, list):
            inner = ",\n".join("  " + json.dumps(x, ensure_ascii=False) for x in v)
            parts.append(f" {json.dumps(k)}: [\n{inner}\n ]")
        else:
            parts.append(f" {json.dumps(k)}: " + json.dumps(v, ensure_ascii=False, indent=1).replace("\n", "\n "))
    return "{\n" + ",\n".join(parts) + "\n}\n"


def validate_graph(g: dict) -> list[str]:
    """Contract checks for graph.json; returns a list of problems (empty = valid)."""
    errs = []
    if g.get("version") != VERSION:
        errs.append(f"graph version {g.get('version')!r} != {VERSION}")
    for k in ("source", "coordinates", "energy_bands", "nodes", "links", "moves", "legend"):
        if k not in g:
            errs.append(f"graph: missing key {k!r}")
    if errs:
        return errs
    ids = [n["id"] for n in g["nodes"]]
    if len(set(ids)) != len(ids):
        errs.append("graph: duplicate node ids")
    idset = set(ids)
    move_slugs = {m["slug"] for m in g["moves"]}
    for n in g["nodes"]:
        for k in ("id", "name", "slug", "move", "shape", "tone", "tone_name", "family", "position", "energy", "band",
                  "status"):
            if k not in n:
                errs.append(f"node {n.get('id')}: missing {k!r}")
        if n.get("status") not in NODE_STATUSES:
            errs.append(f"node {n['id']}: bad status {n.get('status')!r}")
        if not (0.0 <= float(n.get("energy", -1)) <= 1.0):
            errs.append(f"node {n['id']}: energy outside 0..1")
        if n.get("move") is not None and n["move"] not in move_slugs:
            errs.append(f"node {n['id']}: move {n['move']!r} not in moves")
        if len(n.get("position") or []) != 3:
            errs.append(f"node {n['id']}: position must be [x,y,z]")
    for l in g["links"]:
        if l["from"] not in idset or l["to"] not in idset:
            errs.append(f"link {l['id']}: unknown endpoint")
        if l.get("status") not in LINK_STATUSES:
            errs.append(f"link {l['id']}: bad status {l.get('status')!r}")
    names = {m["slug"]: m["name"] for m in g["moves"]}
    for n in g["nodes"]:
        if n.get("move") in names and n["name"] != names[n["move"]]:
            errs.append(f"node {n['id']}: name {n['name']!r} is not its move's name {names[n['move']]!r}")
    for m in g["moves"]:
        for i in m["nodes"]:
            if i not in idset:
                errs.append(f"move {m['slug']}: unknown node {i}")
    b = g["energy_bands"]
    if [x["band"] for x in b] != list(range(len(b))) or b[0]["lo"] != 0.0 or b[-1]["hi"] != 1.0 or any(
            b[i]["hi"] != b[i + 1]["lo"] for i in range(len(b) - 1)):
        errs.append("energy_bands must be contiguous bands 0..n covering [0, 1]")
    return errs


# ---------------------------------------------------------------------------------------------------------------
# loaded graph with indexes (used by labels / path / fingerprint / placeholder)
# ---------------------------------------------------------------------------------------------------------------

def _via_index(nodes: list[dict], links: list[dict]) -> dict[tuple[str, str], dict]:
    """Connections between named nodes through blank (unnamed, move null) junction nodes: a -> blank+ -> b.

    {(a, b): {"links": [link ids in order], "via": [blank node ids in order]}} - shortest chain (fewest hops; ties by
    link order), only where there is no direct link a -> b. Blank nodes cannot be labelled, so without this a
    transition the user's graph connects through an unnamed junction (Tilt Right -> (unnamed) -> Howl Up) would look
    like a missing link."""
    move = {n["id"]: n.get("move") for n in nodes}
    blank = {i for i, m in move.items() if m is None}
    out: dict[str, list[dict]] = defaultdict(list)
    for l in links:
        out[l["from"]].append(l)
    direct = {(l["from"], l["to"]) for l in links}
    idx: dict[tuple[str, str], dict] = {}
    for a in move:
        if a in blank:
            continue
        frontier = [(l["to"], [l["id"]], []) for l in out[a] if l["to"] in blank]
        seen: set[str] = set()
        while frontier:
            nxt = []
            for node, lids, via in frontier:
                if node in seen:
                    continue
                seen.add(node)
                for l in out[node]:
                    b = l["to"]
                    if b in blank:
                        nxt.append((b, lids + [l["id"]], via + [node]))
                    elif b != a and (a, b) not in direct and (a, b) not in idx:
                        idx[(a, b)] = {"links": lids + [l["id"]], "via": via + [node]}
            frontier = nxt
    return idx


class Graph:
    def __init__(self, data: dict):
        self.data = data
        self.nodes: dict[str, dict] = {n["id"]: n for n in data["nodes"]}
        self.links: list[dict] = data["links"]
        self.moves: dict[str, dict] = {m["slug"]: m for m in data["moves"]}
        self.link_index: dict[tuple[str, str], dict] = {(l["from"], l["to"]): l for l in self.links}
        self.link_by_id: dict[str, dict] = {l["id"]: l for l in self.links}
        self.alias: dict[str, str] = {}
        for m in data["moves"]:
            self.alias[m["slug"]] = m["slug"]
        for m in data["moves"]:
            for a in [m.get("name"), m.get("name_en"), *m.get("aliases", []), *m.get("variants", [])]:
                if a:
                    self.alias.setdefault(slugify(a), m["slug"])
        self.out_links: dict[str, list[dict]] = defaultdict(list)
        self.in_links: dict[str, list[dict]] = defaultdict(list)
        for l in self.links:
            self.out_links[l["from"]].append(l)
            self.in_links[l["to"]].append(l)
        self.via: dict[tuple[str, str], dict] = _via_index(data["nodes"], self.links)

    def connection(self, a: str, b: str, strict_direction: bool = False) -> tuple[str | None, dict | None]:
        """How the graph connects node a to node b, in order of preference:
        'link' (a -> b), 'via' (a -> blank+ -> b), and unless strict_direction 'reverse' (b -> a) and 'reverse-via'
        (b -> blank+ -> a); (None, None) when not connected. info = {link: first link id in travel order,
        links: [ids in travel order], via: [blank node ids in travel order]}."""
        l = self.link_index.get((a, b))
        if l is not None:
            return "link", {"link": l["id"], "links": [l["id"]], "via": []}
        v = self.via.get((a, b))
        if v is not None:
            return "via", {"link": v["links"][0], "links": list(v["links"]), "via": list(v["via"])}
        if strict_direction:
            return None, None
        l = self.link_index.get((b, a))
        if l is not None:
            return "reverse", {"link": l["id"], "links": [l["id"]], "via": []}
        v = self.via.get((b, a))
        if v is not None:
            return "reverse-via", {"link": v["links"][-1], "links": v["links"][::-1], "via": v["via"][::-1]}
        return None, None

    @classmethod
    def load(cls, path: Path = GRAPH_PATH) -> "Graph":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def resolve_move(self, text: str | None) -> str | None:
        """Move slug for a slug, alias or authored name; None when unknown."""
        if not text:
            return None
        return self.alias.get(slugify(text))

    def instances(self, move: str) -> list[str]:
        m = self.moves.get(move)
        return list(m["nodes"]) if m else []

    def link(self, a: str, b: str) -> dict | None:
        return self.link_index.get((a, b))

    def position(self, node: str) -> np.ndarray:
        return np.asarray(self.nodes[node]["position"], float)

    def band_labels(self) -> list[str]:
        return [b["label"] for b in self.data["energy_bands"]]


def load_graph(path: Path = GRAPH_PATH) -> Graph:
    """The graph; built on the fly (not written) when taxonomy/zouk_graph.json does not exist yet."""
    path = Path(path)
    if not path.exists():
        if path != GRAPH_PATH:
            raise FileNotFoundError(path)
        return Graph(build_graph())
    return Graph.load(path)


def summary(g: dict) -> str:
    c = g["counts"]
    lines = [f"graph: {c['nodes']} nodes ({c['scaffold_nodes']} Zouk1 + {c['proposed_nodes']} proposed), "
             f"{c['links']} links ({c['scaffold_links']} Zouk1 + {c['proposed_links']} proposed), {c['moves']} moves "
             f"({c['duplicated_moves']} with several instances), {c['blank_nodes']} blank, "
             f"{c['isolated_nodes']} isolated, {c.get('via_links', 0)} connections through blank junctions"]
    lines.append("energy bands: " + ", ".join(f"{b['band']} {b['label']} [{b['lo']:.1f},{b['hi']:.1f}) "
                                              f"y [{b['y_lo']:.3f},{b['y_hi']:.3f}) {b['nodes']} nodes"
                                              for b in g["energy_bands"]))
    lines.append("legend confirmed by the user " + str((g["legend"].get("confirmed") or {}).get("date")))
    for shape, s in g["legend"]["shapes"].items():
        lines.append(f"  {shape:9s} {s['family']:5s} n={s['count']:2d} agreement={s['agreement']} {s['label']}")
    renamed = [(m["name_en"], m["name"], m["display_confidence"]) for m in g["moves"]
               if m.get("name_en") and m["name"] != m["name_en"]]
    lines.append(f"display names ({len(renamed)} differ from the English name): "
                 + ", ".join(f"{en} -> {pt} ({c})" for en, pt, c in renamed))
    prop = [m["name"] for m in g["moves"] if m["status"] != SCAFFOLD]
    lines.append(f"proposed moves ({len(prop)}): " + ", ".join(prop))
    for w in g["warnings"]:
        lines.append(f"  warning: {w}")
    return "\n".join(lines)


def cmd_graph(args: argparse.Namespace) -> int:
    created = ensure_additions(Path(args.additions))
    if created:
        print(f"[moves] created {args.additions} (Corredor display name, Counterbalance Pivot, commented examples)")
    g = build_graph(Path(args.zouk1), Path(args.additions), Path(args.vocabulary))
    errs = validate_graph(g)
    if errs:
        for e in errs:
            print("[moves] INVALID", e)
        return 1
    out = write_graph(g, Path(args.out))
    print(f"[moves] wrote {out}")
    print(summary(g))
    if not args.no_review:
        from .review import draw_graph
        img = draw_graph(Graph(g))
        print(f"[moves] review image {img}")
    return 0


def register(sub) -> None:
    p = sub.add_parser("graph", help="import Zouk1.json + additions -> taxonomy/zouk_graph.json")
    p.add_argument("--zouk1", default=str(ZOUK1_PATH))
    p.add_argument("--additions", default=str(ADDITIONS_PATH))
    p.add_argument("--vocabulary", default=str(VOCABULARY_PATH))
    p.add_argument("--out", default=str(GRAPH_PATH))
    p.add_argument("--no-review", action="store_true", help="skip work/review/moves_graph.png")
    p.set_defaults(func=cmd_graph)
