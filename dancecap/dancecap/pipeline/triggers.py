"""Stage `triggers`: classifier and analysis intervals -> viewer triggers (PIPELINE.md 8).

    python -m dancecap.pipeline triggers <take>                         work/<take>/triggers/triggers.json
    python -m dancecap.pipeline triggers <take> --export-capture <Name>  + StreamingAssets/<Name>/triggers.json
                                                                          (capture time) for the viewer
Producers (read, never changed): counterbalance/counterbalance.json (tiers counterbalance / pivot), floorcraft/
floorcraft.json (leader plants, rotations, travel), moves/labels.json (human / narration / auto segments; placeholder
timelines never trigger). Each interval keeps its confidence and provenance; `triggers` does not re-threshold (the
producers already carry their hysteresis), it only maps intervals to viewer layers with the rules of
taxonomy/triggers.toml (built-in defaults when the file is absent). Below the threshold a trigger is drawn dashed.
Times are reference seconds; the export converts them to capture time (capture = reference + time_to_audio, the
moves export's clock), so the viewer can switch a visualisation on when, e.g., counterbalance was classified.

Layer contract with the viewer (head-movement Assets/Tour/DanceLayers.cs, `Handles`): `layers` are the detailed
visual elements of a trigger (counterbalance.axis, leader.floor_axis, ...); `viewer_layers` are the DanceLayers ids
that draw them (counterbalance | traces | graph | neck | axis | floorcraft | balance | moves | contacts |
rhythm), through the [viewer.map] table of
taxonomy/triggers.toml. Only ids in [viewer] layers are ever emitted, so a trigger never names a layer the viewer
does not have. The export runs after demo_export (python -m dancecap.pipeline export, and the demo_export stage).
"""
from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path

from . import paths
from .util import read_json, sha256_16, write_json_atomic

DEFAULT_RULES = {
    "defaults": {"threshold": 0.5, "fade_s": 0.2},
    "kinds": {
        "counterbalance": {"layers": ["counterbalance.axis", "counterbalance.pivot_ring", "leader.arc",
                                      "couple.balance_axis"],
                           "director": {"shot": "overhead", "priority": 2}},
        "pivot": {"layers": ["counterbalance.pivot_ring", "leader.arc", "follower.spiral", "couple.balance_axis"],
                  "director": {"shot": "overhead", "priority": 2}},
        "plant": {"layers": ["leader.floor_axis", "leader.plant_ring"], "director": {"shot": "wide", "priority": 0}},
        "rotation": {"layers": ["leader.rotation_arc"], "director": {"shot": "wide", "priority": 1}},
        "travel": {"layers": ["leader.floor_path"], "director": {"shot": "wide", "priority": 0}},
        "move": {"layers": ["caption", "graph_inset"], "director": {"shot": "medium", "priority": 1}},
    },
    "viewer": {
        # DanceLayers.Handles (head-movement Assets/Tour/DanceLayers.cs)
        "layers": ["counterbalance", "traces", "graph", "neck", "axis", "floorcraft", "balance", "moves", "contacts", "rhythm"],
        "map": {"counterbalance.axis": "counterbalance", "counterbalance.pivot_ring": "counterbalance",
                "leader.arc": "counterbalance", "follower.spiral": "traces", "follower.head_axis": "neck",
                "leader.floor_axis": "axis", "leader.plant_ring": "floorcraft", "leader.rotation_arc": "floorcraft",
                "leader.floor_path": "floorcraft", "caption": "graph", "graph_inset": "graph",
                "couple.balance_axis": "balance"},
    },
    "families": {
        "turn": {"layers": ["follower.head_axis", "follower.spiral"]},
        "giro": {"layers": ["follower.head_axis", "follower.spiral"]},
        "head": {"layers": ["follower.head_axis"]},
        "cambre": {"layers": ["follower.head_axis"]},
        "chicote": {"layers": ["follower.head_axis", "follower.spiral"]},
        "travel": {"layers": ["leader.floor_axis"]},
        "corredor": {"layers": ["leader.floor_axis"]},
    },
}
SOURCES = {"counterbalance": "counterbalance/counterbalance.json", "floorcraft": "floorcraft/floorcraft.json",
           "moves": "moves/labels.json"}


def rules_path() -> Path:
    return paths.taxonomy_dir() / "triggers.toml"


def load_rules() -> tuple[dict, str]:
    p = rules_path()
    if p.exists():
        try:
            r = tomllib.loads(p.read_text(encoding="utf-8"))
            merged = json.loads(json.dumps(DEFAULT_RULES))
            for k in ("defaults", "kinds", "families"):
                merged[k].update(r.get(k) or {})
            v = r.get("viewer") or {}
            if isinstance(v.get("layers"), list):
                merged["viewer"]["layers"] = [str(x) for x in v["layers"]]
            if isinstance(v.get("map"), dict):
                merged["viewer"]["map"].update({str(a): str(b) for a, b in v["map"].items()})
            return merged, f"taxonomy/triggers.toml@{sha256_16(p)}"
        except (OSError, tomllib.TOMLDecodeError):
            pass
    return DEFAULT_RULES, "builtin"


def _families() -> dict[str, str]:
    g = read_json(paths.taxonomy_dir() / "zouk_graph.json", {}) or {}
    out = {}
    for n in (g.get("nodes") or []) + (g.get("moves") or []):
        if isinstance(n, dict) and n.get("family"):
            for k in (n.get("move"), n.get("slug"), n.get("name")):
                if k:
                    out[str(k).lower()] = str(n["family"]).lower()
    return out


def viewer_layers(layers: list[str], rules: dict) -> list[str]:
    """DanceLayers ids for a trigger's detailed layers (unknown ids are dropped, never invented)."""
    v = rules.get("viewer") or DEFAULT_RULES["viewer"]
    known, mp = set(v.get("layers") or []), v.get("map") or {}
    out = []
    for name in layers:
        lid = mp.get(name, name)
        if lid in known and lid not in out:
            out.append(lid)
    return out


def _style(conf, threshold: float) -> str:
    return "dashed" if conf is not None and conf < threshold else "solid"


def collect(take: str) -> dict:
    """The triggers document (reference time) from the producers' current files."""
    w = paths.take_work(take)
    rules, rules_id = load_rules()
    thr = float(rules["defaults"].get("threshold", 0.5))
    fade = float(rules["defaults"].get("fade_s", 0.2))
    kinds = rules["kinds"]
    out, sources = [], {}

    def src(key):
        p = w / SOURCES[key]
        if p.exists():
            sources[key] = {"file": SOURCES[key], "sha256_16": sha256_16(p)}
        return read_json(p, None) if p.exists() else None

    def add(kind, i, t0, t1, conf, prov, source, params=None, extra_layers=(), move=None):
        k = kinds.get(kind, {})
        layers = list(dict.fromkeys([*(k.get("layers") or []), *extra_layers]))
        d = {"id": f"{kind[:2]}-{i:03d}", "kind": kind, "t0": round(float(t0), 4), "t1": round(float(t1), 4),
             "confidence": None if conf is None else round(float(conf), 3), "provenance": prov,
             "source": sources.get(source), "layers": layers, "viewer_layers": viewer_layers(layers, rules),
             "style": _style(conf, thr), "fade_s": fade,
             "director": k.get("director"), "params": params or {}}
        if move:
            d["move"] = move
        out.append(d)

    cb = src("counterbalance")
    for i, iv in enumerate((cb or {}).get("intervals") or []):
        kind = iv.get("kind") if iv.get("kind") in ("counterbalance", "pivot") else "counterbalance"
        piv = iv.get("pivot") or []
        add(kind, i, iv["t0"], iv["t1"], iv.get("confidence"), "auto", "counterbalance",
            {"tier": kind, "pivot_xz": [piv[0], piv[2]] if len(piv) == 3 else None,
             "direction": iv.get("direction"), "swept_deg": iv.get("swept_deg"), "pivot_foot": iv.get("pivot_foot")})
    fc = src("floorcraft")
    if fc:
        for i, p in enumerate(fc.get("plants") or []):
            add("plant", i, p["t0"], p["t1"], None, "auto", "floorcraft",
                {"origin_xz": [p["origin"][0], p["origin"][2]] if len(p.get("origin") or []) == 3 else None,
                 "turn_deg": p.get("turn_deg")})
        for i, r in enumerate(fc.get("rotations") or []):
            add("rotation", i, r["t0"], r["t1"], None, "auto", "floorcraft",
                {"direction": r.get("direction"), "signed_deg": r.get("signed_deg"), "context": r.get("context")})
        for i, m in enumerate(fc.get("moves") or []):
            add("travel", i, m["t0"], m["t1"], None, "auto", "floorcraft",
                {"length_m": m.get("length_m"), "heading_deg": m.get("heading_deg")})
    lab = src("moves")
    if lab and not lab.get("placeholder"):
        fam = _families()
        for i, s in enumerate(lab.get("segments") or []):
            prov = str(s.get("provenance") or "human")
            if s.get("placeholder") or prov == "placeholder":
                continue
            move = str(s.get("move") or "")
            f = fam.get(move.lower())
            extra = (rules["families"].get(f) or {}).get("layers", []) if f else []
            conf = s.get("confidence")
            if conf is None and prov in ("human", "narration"):
                conf = 1.0
            add("move", i, s["t0"], s["t1"], conf, prov, "moves", {"family": f, "tier": s.get("tier")}, extra, move=move)
    out.sort(key=lambda d: (d["t0"], d["kind"]))
    return {"version": 1, "take": take, "time_base": "reference", "rules": rules_id, "threshold": thr,
            "viewer": {"layers": list((rules.get("viewer") or {}).get("layers") or []),
                       "contract": "head-movement Assets/Tour/DanceLayers.cs Handles(); viewer_layers per trigger"},
            "sources": sources, "triggers": out}


def lanes(doc: dict) -> list[dict]:
    """Timeline lanes for the monitor's analysis view (one per trigger kind)."""
    by: dict[str, list] = {}
    for t in doc["triggers"]:
        by.setdefault(t["kind"], []).append({"id": t["id"], "t0": t["t0"], "t1": t["t1"], "confidence": t["confidence"],
                                             "style": t["style"], "label": t.get("move") or t["kind"]})
    return [{"kind": k, "items": v} for k, v in sorted(by.items())]


def write(take: str) -> Path:
    doc = collect(take)
    p = paths.take_work(take) / "triggers" / "triggers.json"
    write_json_atomic(p, doc)
    print(f"[triggers] {len(doc['triggers'])} trigger(s) from {sorted(doc['sources'])} -> {paths.rel_display(p)}")
    return p


def export_capture(take: str, capture: str) -> Path | None:
    """StreamingAssets/<capture>/triggers.json in capture time (the moves export's clock)."""
    from .util import CAPTURE_RE
    if not CAPTURE_RE.match(capture or ""):
        raise ValueError("invalid capture name")
    doc = read_json(paths.take_work(take) / "triggers" / "triggers.json", None) or collect(take)
    cdir = paths.unity_project() / "Assets" / "StreamingAssets" / capture
    if not (cdir / "capture.json").exists():
        print(f"[triggers] no capture {capture} under StreamingAssets: nothing exported")
        return None
    from ..moves.labels import CaptureClock  # lazy: the export is a CLI step, never the monitor server
    clock = CaptureClock.from_capture(cdir)
    trig = []
    for t in doc["triggers"]:
        d = dict(t, t0_reference=t["t0"], t1_reference=t["t1"], t0=round(clock.to_capture(t["t0"]), 6),
                 t1=round(clock.to_capture(t["t1"]), 6), in_capture=clock.covers(t["t0"], t["t1"]))
        trig.append(d)
    out = dict(doc, time_base="capture", capture=capture, time_to_audio=clock.time_to_audio, triggers=trig,
               time_note="capture time = reference time + time_to_audio (CaptureTimeline audio clock)")
    p = cdir / "triggers.json"
    write_json_atomic(p, out)
    wexp = paths.take_work(take) / "export" / capture
    if wexp.exists():
        write_json_atomic(wexp / "triggers.json", out)
    print(f"[triggers] {sum(1 for t in trig if t['in_capture'])}/{len(trig)} trigger(s) inside {capture} -> {p.name}")
    return p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.pipeline triggers")
    ap.add_argument("take")
    ap.add_argument("--export-capture")
    a = ap.parse_args(argv)
    write(a.take)
    if a.export_capture:
        export_capture(a.take, a.export_capture)
    return 0


if __name__ == "__main__":
    sys.exit(main())
