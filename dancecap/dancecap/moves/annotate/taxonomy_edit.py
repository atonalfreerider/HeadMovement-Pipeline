"""Additive edits of taxonomy/additions.toml from the annotation tool - only when the user accepts them in the UI.

  append_alias(move, alias)      a narrated / misheard phrase the user confirmed for a move (MOVES_RESEARCH L5)
  append_proposed_move(entry)    a move the graph does not know yet (status 'proposed': a dashed node until the user
                                 sets status = "accepted" in additions.toml)
Both append one `[[moves]]` table at the end of the file (comments and the user's layout are never rewritten; an
existing slug in a [[moves]] entry only adds aliases - graph.build_graph take_entry), check that the result parses
(tomllib) and that the graph still builds and validates with it, then replace the file atomically (re-read right
before writing; a concurrent change is merged by appending to the newest text). rebuild_graph() then regenerates
taxonomy/zouk_graph.json (the same as `python -m dancecap.moves graph --no-review`).
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import tomllib
from datetime import datetime, timezone
from pathlib import Path

from .. import ADDITIONS_PATH, GRAPH_PATH, VOCABULARY_PATH, ZOUK1_PATH
from ..graph import FAMILY_SHAPE, TONES, Graph, build_graph, slugify, validate_graph, write_graph


class TaxonomyEditError(ValueError):
    pass


def _q(s: str) -> str:
    return json.dumps(str(s), ensure_ascii=False)


def _stamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _write_atomic(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _append(additions: Path, block: str, check, zouk1: Path, vocabulary: Path) -> dict:
    """Append `block`; check(parsed toml) -> error or None; the graph must build + validate. Returns the graph."""
    additions = Path(additions)
    for _attempt in range(2):
        old = additions.read_text(encoding="utf-8") if additions.exists() else "version = 1\n"
        mtime = additions.stat().st_mtime_ns if additions.exists() else None
        new = old.rstrip("\n") + "\n\n" + block.strip("\n") + "\n"
        try:
            parsed = tomllib.loads(new)
        except tomllib.TOMLDecodeError as e:
            raise TaxonomyEditError(f"the addition would make additions.toml invalid: {e}") from e
        err = check(parsed)
        if err:
            raise TaxonomyEditError(err)
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d) / "additions.toml"
            tmp.write_text(new, encoding="utf-8")
            g = build_graph(Path(zouk1), tmp, Path(vocabulary))
        errs = validate_graph(g)
        if errs:
            raise TaxonomyEditError("the graph would not validate: " + "; ".join(errs[:5]))
        now = additions.stat().st_mtime_ns if additions.exists() else None
        if now != mtime:  # changed while we checked: start again from the newest text
            continue
        _write_atomic(additions, new)
        return g
    raise TaxonomyEditError("additions.toml keeps changing; try again")


def append_alias(move: str, alias: str, graph: Graph, additions: Path = ADDITIONS_PATH, zouk1: Path = ZOUK1_PATH,
                 vocabulary: Path = VOCABULARY_PATH, note: str | None = None) -> dict:
    """Make `alias` resolve to `move`. Returns {changed, graph (dict or None), message}."""
    alias = " ".join(str(alias).split())
    if not alias or not slugify(alias):
        raise TaxonomyEditError("empty alias")
    if move not in graph.moves:
        raise TaxonomyEditError(f"unknown move {move!r}")
    cur = graph.resolve_move(alias)
    if cur == move:
        return {"changed": False, "graph": None, "message": f"{alias!r} already names {move}"}
    if cur is not None:
        raise TaxonomyEditError(f"{alias!r} already names the move {cur!r}; change that first")
    block = (f"# learned alias (narration tool, {_stamp()}){': ' + note if note else ''}\n"
             f"[[moves]]\nslug = {_q(move)}\naliases = [{_q(alias)}]\n")

    def check(parsed: dict) -> str | None:
        last = (parsed.get("moves") or [{}])[-1]
        return None if last.get("slug") == move and alias in (last.get("aliases") or []) else "alias not appended"

    g = _append(Path(additions), block, check, zouk1, vocabulary)
    if Graph(g).resolve_move(alias) != move:
        raise TaxonomyEditError(f"{alias!r} would not resolve to {move!r} (see the graph warnings)")
    return {"changed": True, "graph": g, "message": f"{alias!r} -> {move}"}


def append_proposed_move(entry: dict, graph: Graph, additions: Path = ADDITIONS_PATH, zouk1: Path = ZOUK1_PATH,
                         vocabulary: Path = VOCABULARY_PATH) -> dict:
    """entry: {name, slug?, aliases?, family?, tone?, energy?, near?, links_from?, links_to?, description?,
    display_name?}. Returns {changed, slug, graph, message}."""
    name = " ".join(str(entry.get("name") or "").split())
    slug = slugify(entry.get("slug") or name)
    if not slug:
        raise TaxonomyEditError("a proposed move needs a name")
    if graph.resolve_move(slug) or graph.resolve_move(name):
        raise TaxonomyEditError(f"{name!r} already names the move {graph.resolve_move(slug) or graph.resolve_move(name)!r}")
    aliases = [a for a in dict.fromkeys(" ".join(str(x).split()) for x in entry.get("aliases") or []) if a and a != name]
    for a in aliases:
        if graph.resolve_move(a):
            raise TaxonomyEditError(f"alias {a!r} already names {graph.resolve_move(a)!r}")
    family = entry.get("family") or "step"
    if family not in FAMILY_SHAPE:
        raise TaxonomyEditError(f"family {family!r} not in {sorted(FAMILY_SHAPE)}")
    tone = entry.get("tone") or "green"
    if tone not in TONES:
        raise TaxonomyEditError(f"tone {tone!r} not in {sorted(TONES)}")
    try:
        energy = float(entry.get("energy", 0.4))
    except (TypeError, ValueError):
        raise TaxonomyEditError("energy must be a number 0..1") from None
    energy = min(1.0, max(0.0, energy))
    lines = [f"# proposed in the narration tool ({_stamp()})", "[[moves]]", f"slug = {_q(slug)}", f"name = {_q(name)}"]
    if entry.get("display_name"):
        lines.append(f"display_name = {_q(entry['display_name'])}")
    if aliases:
        lines.append("aliases = [" + ", ".join(_q(a) for a in aliases) + "]")
    lines += [f"family = {_q(family)}", f"tone = {_q(tone)}", f"energy = {energy:.2f}"]
    for key in ("near", "links_from", "links_to"):
        vals = [graph.resolve_move(x) or x for x in entry.get(key) or [] if x]
        vals = [v for v in dict.fromkeys(vals) if v in graph.moves]
        if vals:
            lines.append(f"{key} = [" + ", ".join(_q(v) for v in vals) + "]")
    lines.append('status = "proposed"')
    if entry.get("description"):
        desc = re.sub(r"\s+", " ", str(entry["description"])).strip()
        lines.append(f"description = {_q(desc)}")
    block = "\n".join(lines) + "\n"

    def check(parsed: dict) -> str | None:
        last = (parsed.get("moves") or [{}])[-1]
        return None if last.get("slug") == slug else "move not appended"

    g = _append(Path(additions), block, check, zouk1, vocabulary)
    if slug not in Graph(g).moves:
        raise TaxonomyEditError(f"{slug!r} did not become a move (see the graph warnings)")
    return {"changed": True, "slug": slug, "graph": g, "message": f"proposed move {name!r} ({slug})"}


def rebuild_graph(graph_path: Path = GRAPH_PATH, additions: Path = ADDITIONS_PATH, zouk1: Path = ZOUK1_PATH,
                  vocabulary: Path = VOCABULARY_PATH, built: dict | None = None) -> dict:
    """Regenerate the graph file (or write the dict already built from the same inputs)."""
    g = built or build_graph(Path(zouk1), Path(additions), Path(vocabulary))
    errs = validate_graph(g)
    if errs:
        raise TaxonomyEditError("graph invalid: " + "; ".join(errs[:5]))
    write_graph(g, Path(graph_path))
    return g
