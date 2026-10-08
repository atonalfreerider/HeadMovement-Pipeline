"""One narration -> the labels.json of every take on the same reference clock (narration propagation).

A narration is recorded in ONE take's annotation tool (work/<source>/moves/narration/<id>/: audio, player log,
transcript, settings) but may name the moves of the whole dance; takes cut from the same recordings (larissa_kadu
56-62 s, larissa_kadu_38 30.0-68.4 s, larissa_kadu_full the whole dance) all run on camera 06's reference clock.
Propagation maps the narration into each take the way the annotation tool maps it into its own take:

  1. The player log is in the narrated camera's VIDEO time (camera 02 for 20261007-152143), so every take maps it
     through ITS OWN clock for that camera (its toml's alignment file) and snaps the starts to ITS OWN beat grid
     (beats.json; a take without one borrows another take's grid: --beats-from larissa_kadu_full=larissa_kadu_38,
     same clocks and song). Same code as the tool's remap (pipeline.map_and_build, the narration's settings.json).
  2. The narrated sequence must come out the same in every take (only the clocks differ, by a few ms), else nothing
     is written. Segment ids are shared across takes and stable across runs (matched by anchor word + anchor time
     to the previous mapping), so an item keeps its id when it is confirmed, edited or re-mapped.
  3. Each take keeps the items overlapping its toml window, clipped to it ('narrated_span' keeps the narrated extent
     of a clipped item); unknown phrases go to labels.json 'unresolved' (unlabelled gaps, never a guessed move).
  4. Review state survives: items the user confirmed ('human') or edited in the annotation tool, in any of the takes
     (the MASTER first), are carried into every take and the narration is clipped around them; an item the user
     deleted stays deleted. 'Edited' / 'deleted' = differs from / missing in what the last propagation wrote
     (propagate.json 'written'). Adding an alias (taxonomy/additions.toml) and propagating again resolves an
     unknown phrase everywhere.
  5. A labels.json is rewritten only when its content changes (validated against the graph first; the previous
     file is kept in narration/<id>/propagate_backups/); the source take's result.json likewise (previous kept as
     result.<UTC stamp>.json).

MASTER: the take where the narration is reviewed item by item - the one whose window holds the whole narration
(larissa_kadu_full for 20261007-152143). Open the annotation tool there (`python -m dancecap.moves annotate --take
larissa_kadu_full`), confirm / fix items and resolve the unknown phrases, Save, then propagate again: the other takes
follow. Targets, master and borrowed beat grids are recorded in narration/<id>/propagate.json, so a re-run needs only
--take and --id. Nothing here touches anything but labels.json files, result.json and the narration folder.

    python -m dancecap.moves annotate propagate --take larissa_kadu --id 20261007-152143 \\
        --to larissa_kadu_38 larissa_kadu_full --master larissa_kadu_full --beats-from larissa_kadu_full=larissa_kadu_38
    python -m dancecap.moves annotate propagate --take larissa_kadu --id 20261007-152143            dry run (report)
    python -m dancecap.moves annotate propagate --take larissa_kadu --id 20261007-152143 --write
Then `python -m dancecap.moves run <take>` for every take whose labels changed (path, fingerprint, Unity copies).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from ...take import PACKAGE_ROOT, WORKSPACE, Take
from ..labels import validate_labels
from .pipeline import MIN_LEN, NarrationSettings, _subtract, map_and_build, now_iso

CONFIG = "propagate.json"
VOLATILE = ("created", "saved")  # timestamps: ignored when deciding whether a file changed
WINDOW_EPS = 1e-4
ANCHOR_TOL = 0.06  # s: the same narrated word mapped through two takes' clocks (measured <= 23 ms apart)


class PropagationError(RuntimeError):
    pass


def _rel(p: Path) -> str:
    try:
        return Path(p).resolve().relative_to(WORKSPACE.resolve()).as_posix()
    except ValueError:
        return Path(p).resolve().as_posix()


def _sha(p: Path) -> str | None:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.exists() else None


def _strip(x, _top: bool = True):
    """x as JSON would store it, without volatile keys (timestamps) - for 'did anything change' comparisons."""
    if _top:
        x = json.loads(json.dumps(x, ensure_ascii=False))
    if isinstance(x, dict):
        return {k: _strip(v, False) for k, v in x.items() if k not in VOLATILE}
    if isinstance(x, list):
        return [_strip(v, False) for v in x]
    return x


def _what(it: dict) -> str:
    return it.get("move") or f"unknown '{it.get('unresolved')}'"


def _sig(it: dict) -> list:
    return [round(float(it["t0"]), 4), round(float(it["t1"]), 4), it.get("move"), it.get("provenance")]


def snapshot(items: list[dict]) -> dict[str, list]:
    """{id: [t0, t1, move, provenance, anchor_word, anchor_t]} of what was written (propagate.json 'written')."""
    return {it["id"]: _sig(it) + [it.get("anchor_word"), it.get("anchor_t")] for it in items if it.get("id")}


def sequence(segs: list[dict]) -> list[tuple]:
    return [(s.get("move"), None if s.get("move") else s.get("unresolved")) for s in segs]


def compare_sequences(mapped: dict[str, list[dict]], source: str) -> dict[str, dict]:
    """Every take's mapping must name the same items in the same order as the source take's (only the times may
    differ). Returns {take: {max_dt0_ms, max_dt1_ms}}; raises PropagationError naming the first difference."""
    ref = mapped[source]
    a = sequence(ref)
    out = {}
    for take, segs in mapped.items():
        if take == source:
            continue
        b = sequence(segs)
        if a != b:
            k = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            raise PropagationError(
                f"{take}: the narrated sequence differs from {source}'s at item {k} ({a[k] if k < len(a) else None} "
                f"vs {b[k] if k < len(b) else None}; {len(b)} vs {len(a)} items) - its clocks or beat grid map the "
                "narration differently; nothing written")
        out[take] = {"max_dt0_ms": round(1000 * max((abs(x["t0"] - y["t0"]) for x, y in zip(ref, segs)), default=0), 1),
                     "max_dt1_ms": round(1000 * max((abs(x["t1"] - y["t1"]) for x, y in zip(ref, segs)), default=0), 1)}
    return out


def stable_ids(new: list[dict], pool: list[dict], tol: float = ANCHOR_TOL) -> dict[str, str]:
    """Give freshly mapped segments the ids (and 'created') of earlier items for the same narrated word: same anchor
    word, anchor time within tol, nearest first, each earlier id used once (unsplit ids preferred). Earlier items =
    the previous result.json, the takes' labels.json and the last propagation's snapshot. Returns {fresh: kept}."""
    cands = []
    seen = set()
    for o in pool:
        if not o.get("id") or o["id"] in seen or o.get("anchor_t") is None:
            continue
        seen.add(o["id"])
        cands.append(o)
    cands.sort(key=lambda o: (float(o["anchor_t"]), len(o["id"])))
    used: set[int] = set()
    ren = {}
    for s in new:
        if s.get("anchor_t") is None:
            continue
        best = None
        for j, o in enumerate(cands):
            if j in used or o.get("anchor_word") != s.get("anchor_word"):
                continue
            d = abs(float(o["anchor_t"]) - float(s["anchor_t"]))
            if d <= tol and (best is None or (d, len(o["id"])) < best[0]):
                best = ((d, len(o["id"])), j)
        if best is None:
            continue
        o = cands[best[1]]
        used.add(best[1])
        if o["id"] != s["id"]:
            ren[s["id"]] = o["id"]
            s["id"] = o["id"]
        if o.get("created"):
            s["created"] = o["created"]
    return ren


def unclip(item: dict, window: tuple[float, float] | None) -> dict:
    """An item clipped to a narrower take's window gets its narrated extent back on the clipped side(s) before it
    is carried into another take (it is clipped again to that take's window)."""
    d = dict(item)
    ns = d.pop("narrated_span", None)
    if ns and window is not None:
        if abs(float(d["t0"]) - window[0]) < WINDOW_EPS and float(ns[0]) < float(d["t0"]):
            d["t0"] = float(ns[0])
        if abs(float(d["t1"]) - window[1]) < WINDOW_EPS and float(ns[1]) > float(d["t1"]):
            d["t1"] = float(ns[1])
    return d


def clip(items: list[dict], window: tuple[float, float], grid=None, min_len: float = MIN_LEN
         ) -> tuple[list[dict], list[list]]:
    """Items overlapping [w0, w1), clipped to it ('narrated_span' = the extent before clipping; b0 / b1 re-indexed on
    the take's grid). Returns (kept, dropped [[what, t0, t1]] - shorter than min_len after clipping)."""
    w0, w1 = window
    kept, dropped = [], []
    for seg in items:
        t0, t1 = float(seg["t0"]), float(seg["t1"])
        if t1 <= w0 or t0 >= w1:
            continue
        a, b = max(t0, w0), min(t1, w1)
        if b - a < min_len:
            dropped.append([_what(seg), round(a, 6), round(b, 6)])
            continue
        d = json.loads(json.dumps(seg))
        if (a, b) != (t0, t1):
            d["narrated_span"] = [t0, t1]
            d["t0"], d["t1"] = round(a, 6), round(b, 6)
            if grid is not None and len(grid):
                d["b0"], d["b1"] = grid.eighth_index(d["t0"]), grid.eighth_index(d["t1"] - 1e-6)
        kept.append(d)
    return kept, dropped


def review_state(current: dict[str, list[dict] | None], written: dict[str, dict | None], order: list[str],
                 windows: dict[str, tuple[float, float]]) -> tuple[list[dict], list[str], list[str]]:
    """What the user did in the annotation tool since the last propagation, over the takes in priority order
    (master first). current[take] = the take's labels.json items (segments + unresolved with move None) or None;
    written[take] = the last propagation's snapshot of that take or None (then only 'human' items count).
    Returns (carried items, first take wins per id; deleted ids; notes)."""
    carried, seen, deleted, notes = [], set(), set(), []
    for take in order:
        items = current.get(take)
        if items is None:
            continue
        snap = written.get(take)
        for it in items:
            prov = it.get("provenance")
            if prov == "placeholder" or not it.get("id"):
                continue
            if prov == "human":
                why = "confirmed"
            elif snap is not None and (it["id"] not in snap or _sig(it) != list(snap[it["id"]])[:4]):
                why = "edited"
            else:
                continue
            if it["id"] in seen:
                continue
            seen.add(it["id"])
            carried.append(unclip(it, windows.get(take)))
            notes.append(f"{take}: {why} {it['id']} {_what(it)} [{float(it['t0']):.3f}, {float(it['t1']):.3f})")
        if snap is not None:
            present = {it.get("id") for it in items}
            deleted |= {sid for sid in snap if sid not in present}
    deleted -= seen
    notes += [f"deleted in a take since the last propagation: {sid}" for sid in sorted(deleted)]
    return carried, sorted(deleted), notes


def compose(narr: list[dict], carried: list[dict], deleted: list[str] | set[str], window: tuple[float, float],
            grid=None, min_len: float = MIN_LEN) -> tuple[list[dict], list[list]]:
    """One take's items: the carried review items (earlier ones win overlaps), then the narration minus the carried
    and deleted ids, clipped around the carried spans (pipeline merge policy), all clipped to the take's window."""
    fixed: list[dict] = []
    for c in carried:
        cut = [(float(f["t0"]), float(f["t1"])) for f in fixed]
        for piece in _subtract(c, cut, min_len):
            if grid is not None and len(grid):
                piece["b0"], piece["b1"] = grid.eighth_index(piece["t0"]), grid.eighth_index(piece["t1"] - 1e-6)
            fixed.append(piece)
    skip = {c["id"] for c in carried} | set(deleted)
    cuts = sorted((float(f["t0"]), float(f["t1"])) for f in fixed)
    rest = []
    for s in narr:
        if s.get("id") in skip:
            continue
        rest += _subtract(s, cuts, min_len) if cuts else [s]
    return clip(sorted(fixed + rest, key=lambda s: (float(s["t0"]), float(s["t1"]))), window, grid, min_len)


# ---------------------------------------------------------------------------------------------------------------

def beats_file(take: str, beats_from: dict[str, str], work: Path) -> tuple[Path | None, str | None]:
    """The take's own beat grid, else the one it borrows (beats_from[take] = another take)."""
    for p in (work / take / "beats.json", work / take / "beats" / "beats.json", work / take / "timing" / "beats.json"):
        if p.exists():
            return p, None
    other = beats_from.get(take)
    if other:
        p = work / other / "beats.json"
        if not p.exists():
            raise PropagationError(f"{take}: borrows {other}'s beat grid but {_rel(p)} does not exist")
        return p, other
    return None, None


def _items(doc: dict | None) -> list[dict] | None:
    if doc is None:
        return None
    out = [dict(s) for s in doc.get("segments") or []]
    for u in doc.get("unresolved") or []:
        if isinstance(u, dict) and u.get("t0") is not None:
            out.append({**u, "move": None})
    return out


def _read(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def propagate(source: str, nid: str, targets: list[str] | None = None, master: str | None = None,
              beats_from: dict[str, str] | None = None, write: bool = False, work_root: Path | None = None,
              log=print) -> dict:
    """Map narration <nid> of take <source> into every target take (see the module doc). Dry run unless write.
    Returns {source, id, master, targets, carried, deleted, result_changed, written, takes: {take: stats},
    docs: {take: the labels.json document}, result: the source take's mapping (result.json)}."""
    from .session import Session

    work = Path(work_root or PACKAGE_ROOT / "work")
    ndir = work / source / "moves" / "narration" / nid
    if not (ndir / "transcript.json").exists() or not (ndir / "log.json").exists():
        raise PropagationError(f"{_rel(ndir)}: no transcript.json / log.json (transcribe the narration first)")
    cfg_path = ndir / CONFIG
    cfg = _read(cfg_path) or {}
    targets = list(dict.fromkeys([source, *(targets or cfg.get("targets") or [])]))
    master = master or cfg.get("master") or source
    if master not in targets:
        targets.append(master)
    beats_from = {**(cfg.get("beats_from") or {}), **(beats_from or {})}
    written_prev = cfg.get("written") or {}

    transcript = json.loads((ndir / "transcript.json").read_text(encoding="utf-8"))
    log_doc = json.loads((ndir / "log.json").read_text(encoding="utf-8"))
    saved = _read(ndir / "settings.json") or {}
    entries = log_doc.get("entries") or []
    cams = sorted({str(e["cam"]) for e in entries if e.get("cam") is not None})
    rates = sorted({float(e["rate"]) for e in entries if e.get("rate") is not None})

    sessions, results, mapped, info = {}, {}, {}, {}
    for take in targets:
        bp, borrowed = beats_file(take, beats_from, work)
        s = Session.for_take(take, probe_videos=False, beats=bp, work_root=work)
        missing = [c for c in cams if c not in s.cameras]
        if missing:
            raise PropagationError(f"{take}: no camera {missing} (the narration was watched on {cams})")
        st = NarrationSettings.from_dict({**s.settings, **saved, "annotator": saved.get("annotator") or s.annotator})
        res = map_and_build(transcript, log_doc, s.context(), s.lexicon, st, nid)
        sessions[take], results[take], mapped[take] = s, res, res["segments"]
        tk = Take.load(take)
        toml = PACKAGE_ROOT / "takes" / f"{take}.toml"
        info[take] = {"window": (float(tk.cfg["window"][0]), float(tk.cfg["window"][1])), "beats": bp,
                      "borrowed": borrowed, "alignment": (toml.parent / tk.cfg["alignment"]).resolve(),
                      "grid": s.grid()}
        if bp is None:
            log(f"[propagate] {take}: no beat grid - starts are not snapped")

    # review state of the takes' current labels (only files written from this narration count as 'ours')
    docs = {t: _read(sessions[t].labels_path) for t in targets}
    current = {t: _items(d) for t, d in docs.items()}
    ours = {t: d is not None and (d.get("narration") or {}).get("id") == nid for t, d in docs.items()}
    written = {t: ((written_prev.get(t) or {}).get("items") if ours[t] else None) for t in targets}
    order = [master] + [t for t in targets if t != master]

    # stable ids: the source mapping takes the earlier ids of the same narrated words; every take shares them
    prev_result = _read(ndir / "result.json") or {}
    pool = []
    for t in order:
        pool += [{"id": k, "anchor_word": v[4], "anchor_t": v[5]} for k, v in (written_prev.get(t) or {}).get(
            "items", {}).items() if len(v) > 5]
        pool += [it for it in current[t] or [] if it.get("narration") == nid]
    pool += prev_result.get("segments") or []
    ren = stable_ids(mapped[source], pool)
    for e in results[source].get("events") or []:
        if e.get("segment") in ren:
            e["segment"] = ren[e["segment"]]
    diffs = compare_sequences(mapped, source)
    for take in targets:
        if take == source:
            continue
        for a, b in zip(mapped[source], mapped[take]):
            b["id"] = a["id"]
            b["created"] = a.get("created", b.get("created"))

    windows = {t: info[t]["window"] for t in targets}
    carried, deleted, notes = review_state(current, written, order, windows)
    for n in notes:
        log(f"[propagate] review: {n}")

    plan = {}
    for take in targets:
        s, inf = sessions[take], info[take]
        kept, dropped = compose(mapped[take], carried, deleted, inf["window"], inf["grid"])
        lab, unresolved = s.build_labels(kept)
        cam_clock = {c: [s.cameras[c].slope, s.cameras[c].offset] for c in cams}
        cmd = f"python -m dancecap.moves annotate propagate --take {source} --id {nid} --write"
        lab.extra["narration"] = {
            "id": nid, "source": _rel(ndir), "take_narrated": source, "camera": cams[0] if len(cams) == 1 else cams,
            "rate": rates[0] if len(rates) == 1 else rates, "asr": results[take].get("asr"),
            "settings": results[take].get("settings"),
            "mapped_with": {"clocks": _rel(inf["alignment"]), "camera_clock": cam_clock,
                            "beats": _rel(inf["beats"]) if inf["beats"] else None,
                            **({"beats_borrowed_from": inf["borrowed"]} if inf["borrowed"] else {})},
            "window": list(inf["window"]), "clipped_to_window": True, "dropped_short": dropped,
            "master": master, "propagated_by": cmd,
            "review": (f"provenance 'narration' = mapped automatically, not yet reviewed item by item. Review on the "
                       f"master take: `python -m dancecap.moves annotate --take {master}` (confirm -> 'human', fix, "
                       f"resolve the 'unresolved' phrases), Save, then `{cmd}` carries the review into every take; "
                       "unknown phrases stay unlabelled gaps until then"),
            "carried_review_items": sum(1 for x in lab.segments if x.provenance == "human" or x.id in
                                        {c["id"] for c in carried}),
            "created": now_iso()}
        errs = validate_labels(lab, s.graph)
        if errs:
            raise PropagationError(f"{take}: labels invalid: {errs[:5]}")
        doc = lab.to_dict()
        changed = docs[take] is None or _strip(doc) != _strip(docs[take])
        plan[take] = {"session": s, "doc": doc, "changed": changed, "items": _items(doc),
                      "stats": {"segments": len(lab.segments), "unresolved": len(unresolved),
                                "labelled_s": round(lab.labelled_seconds, 3), "provenance": lab.provenance,
                                "window": [round(w, 6) for w in inf["window"]], "dropped_short": dropped,
                                "beats": _rel(inf["beats"]) if inf["beats"] else None,
                                "beats_borrowed_from": inf["borrowed"], "changed": changed,
                                **({"vs_source": diffs[take]} if take in diffs else {})}}
        log(f"[propagate] {take}: window [{inf['window'][0]:.3f}, {inf['window'][1]:.3f}) -> {len(lab.segments)} "
            f"segments ({lab.labelled_seconds:.2f} s, {lab.provenance}), {len(unresolved)} unresolved gaps"
            + (f", vs {source} max |dt0| {diffs[take]['max_dt0_ms']} ms" if take in diffs else "")
            + (f", dropped short {dropped}" if dropped else "")
            + ("" if changed else ", unchanged"))

    res_changed = _strip(results[source]) != _strip(prev_result)
    out = {"source": source, "id": nid, "master": master, "targets": targets, "carried": len(carried),
           "deleted": deleted, "result_changed": res_changed, "written": [],
           "takes": {t: p["stats"] for t, p in plan.items()},
           "docs": {t: p["doc"] for t, p in plan.items()}, "result": results[source]}
    if not write:
        log("[propagate] dry run - nothing written (pass --write)")
        return out

    stamp = _stamp()
    if res_changed:
        if (ndir / "result.json").exists():
            shutil.copyfile(ndir / "result.json", ndir / f"result.{stamp}.json")
        (ndir / "result.json").write_text(json.dumps(results[source], indent=1, ensure_ascii=False), encoding="utf-8")
        out["written"].append(_rel(ndir / "result.json"))
    snap = {}
    for take, p in plan.items():
        path = p["session"].labels_path
        if p["changed"]:
            if path.exists():
                bdir = ndir / "propagate_backups"
                bdir.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, bdir / f"{take}-labels-{stamp}.json")
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(p["doc"], indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
            os.replace(tmp, path)
            out["written"].append(_rel(path))
            log(f"[propagate] wrote {_rel(path)}")
        snap[take] = {"sha": _sha(path), "items": snapshot(p["items"])}
    cfg_out = {"version": 1, "source": source, "id": nid, "targets": targets, "master": master,
               "beats_from": beats_from, "command": f"python -m dancecap.moves annotate propagate --take {source} "
                                                     f"--id {nid} --write",
               "note": "written by dancecap.moves.annotate.propagate; 'written' = what each take's labels.json held "
                       "after the last propagation (review edits are detected against it)",
               "updated": now_iso(), "written": snap}
    cfg_path.write_text(json.dumps(cfg_out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    out["written"].append(_rel(cfg_path))
    changed = [t for t, p in plan.items() if p["changed"]]
    if changed:
        log("[propagate] next: " + " ; ".join(f"python -m dancecap.moves run {t}" for t in changed)
            + "   (path + fingerprint + Unity copies; a take without a capture yet: re-run after its export)")
    return out


def cmd_propagate(args) -> int:
    if not args.take or not args.id:
        raise SystemExit("propagate needs --take <the take the narration was recorded in> and --id <narration id>")
    beats_from = {}
    for kv in args.beats_from or []:
        if "=" not in kv:
            raise SystemExit(f"--beats-from wants TAKE=OTHER_TAKE, got {kv!r}")
        a, b = kv.split("=", 1)
        beats_from[a.strip()] = b.strip()
    try:
        out = propagate(args.take, args.id, targets=args.to, master=args.master, beats_from=beats_from,
                        write=args.write)
    except PropagationError as e:
        print(f"[propagate] ERROR: {e}")
        return 1
    print(f"[propagate] master {out['master']}; {out['carried']} review items carried, {len(out['deleted'])} deleted;"
          f" result.json {'changed' if out['result_changed'] else 'unchanged'}")
    return 0
