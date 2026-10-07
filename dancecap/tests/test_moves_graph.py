"""dancecap.moves: graph import, labels contract, Viterbi path, fingerprint, placeholder, Unity export.

    ..\\.venv-nopo4d\\Scripts\\python -m pytest tests/test_moves_graph.py -q
"""
from __future__ import annotations

import json
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dancecap.moves import ZOUK1_PATH  # noqa: E402
from dancecap.moves import graph as G  # noqa: E402
from dancecap.moves.export import (CaptureClock, ExportError, convert_labels, convert_path, export_moves,  # noqa: E402
                                   validate_export)
from dancecap.moves.fingerprint import (compute_fingerprint, source_record, stale_sources,  # noqa: E402
                                        validate_fingerprint)
from dancecap.moves.labels import (Labels, LabelsError, Segment, load_labels, normalise_moves, save_labels,  # noqa: E402
                                   to_reference, validate_labels)
from dancecap.moves.path import PathParams, resolve_path, validate_path  # noqa: E402
from dancecap.moves.placeholder import make_placeholder, measure_bounds, measure_phase, unit_bounds  # noqa: E402
from dancecap.take import PACKAGE_ROOT  # noqa: E402

pytestmark = pytest.mark.skipif(not ZOUK1_PATH.exists(), reason="Zouk1.json not available")
NOFILE = Path("__no_such_file__")
LATERAL = ["8c3386a3-06a5-4328-8409-e2a6a9c62c81", "1c1bdf22-893e-42a5-89d8-56234cbbae94"]
JUNCTION_HOWL = "58751262-d94f-46a5-be32-c7250e0231f4"   # blank: Tilt Left / Tilt Right -> (blank) -> Howl Up
JUNCTION_LUNGE = "31dee6fb-b974-4656-aa16-76a055676a1b"  # blank: Tilt Right -> (blank) -> Side Lunge
WALK_AROUND_CONNECTED = "9b47a87e-d371-4dc7-8edc-35b5971ae82d"  # the other Walk Around is isolated in Zouk1

USER_SHAPES = {"Cylinder": "standing rest", "Hourglass": "holds / embraces / isolations",
               "Diamond": "steps and basics", "Plus": "open / hand-connection moves", "Star": "turns and spins",
               "Tetra": "head movements (howls) and dips", "Ball": "special wave moves"}
USER_TONES = {"blue": "connection / position states", "green": "moves",
              "yellow": "advanced / high-commitment moves", "orange": "advanced / high-commitment moves"}


@pytest.fixture(scope="module")
def raw():
    """The graph from Zouk1.json alone (no additions, no vocabulary)."""
    return G.build_graph(ZOUK1_PATH, NOFILE, NOFILE)


@pytest.fixture(scope="module")
def merged(tmp_path_factory):
    """additions.toml template + a stand-in vocabulary_research.json in the research format (version 1)."""
    d = tmp_path_factory.mktemp("taxonomy")
    add = d / "additions.toml"
    G.ensure_additions(add)
    vocab = d / "vocabulary_research.json"
    vocab.write_text(json.dumps({
        "version": 1, "kind": "vocabulary_research",
        "aliases": {"soltinho": ["solto"], "nonexistent-move": ["whatever"]},  # legacy alias table
        "vocabulary": [
            {"name": "Corredor", "slug": "lateral", "display_name": "Corredor", "display_confidence": "confirmed",
             "zouk1_name": "Lateral", "aliases": ["corredor", "el corredor", "lateral"], "zouk1_matches": LATERAL,
             "status": "scaffold", "sources": ["https://example.org/corredor"]},
            {"name": "Giro simples", "slug": "simple-turn", "display_name": "Giro simples",
             "display_confidence": "translation", "aliases": ["giro"], "status": "scaffold"},
            {"name": "Abraço", "slug": "hug", "display_name": "Abraço", "display_confidence": "likely",
             "aliases": ["abraco", "close embrace"], "status": "scaffold"},
            {"name": "Contrapeso", "slug": "counterbalance-pivot", "display_name": "Contrapeso",
             "display_confidence": "likely", "aliases": ["planada"], "status": "proposed",
             "links_from": ["soltinho"]},  # existing move: aliases + display only, its links are not merged
            {"name": "Chicote", "slug": "chicote", "display_name": "Chicote", "display_confidence": "confirmed",
             "aliases": ["whip"], "energy_hint": 0.7, "near": ["howl-down"], "family": "head", "status": "proposed",
             "sources": ["https://example.org/chicote"]},
            {"name": "Pião", "slug": "piao", "display_name": "Pião", "display_confidence": "confirmed",
             "aliases": ["spinning top"], "family": "turn", "tone": "yellow", "energy_hint": 0.58,
             "position_hint": [0.2, 0.4, -0.3], "links_from": ["hug"], "links_to": ["hug"], "status": "proposed"},
            {"name": "Sacadinha", "slug": "sacadinha", "display_name": "Sacadinha", "display_confidence": "likely",
             "energy_hint": 0.5, "family": "step", "status": "scaffold"},  # names no graph move
        ],
        "proposed_links": [{"from": "lateral", "to": "yoyo"}],  # suggestions only: never merged
        "unresolved_aliases": [{"alias": "cicada", "candidates": [{"term": "sacada"}, {"term": "chicote"}]}],
    }, ensure_ascii=False), encoding="utf-8")
    return G.build_graph(ZOUK1_PATH, add, vocab), add, vocab


def lab(segs, take="t"):
    return Labels(take=take, segments=[Segment(t0=a, t1=b, move=m, id=f"s{i}", **kw)
                                       for i, (a, b, m, *rest) in enumerate(segs)
                                       for kw in [rest[0] if rest else {}]])


# ---------------------------------------------------------------------------------------------------------------
# graph import
# ---------------------------------------------------------------------------------------------------------------

def test_import_counts_and_contract(raw):
    assert len(raw["nodes"]) == 143 and len(raw["links"]) == 181
    assert all(n["status"] == "scaffold" for n in raw["nodes"])
    assert all(l["status"] == "scaffold" for l in raw["links"])
    assert G.validate_graph(raw) == []
    zouk = json.loads(ZOUK1_PATH.read_text())
    assert {n["id"] for n in raw["nodes"]} == {n["uuid"] for n in zouk["nodes"]}
    assert raw["source"] == "Zouk1.json" and raw["coordinates"].startswith("zouk1")
    # positions kept exactly as authored; authored names kept next to the display names
    by = {n["uuid"]: n for n in zouk["nodes"]}
    for n in raw["nodes"]:
        p = by[n["id"]]["position"]
        assert np.allclose(n["position"], [p["x"], p["y"], p["z"]], atol=1e-6)
        authored = next((q.get("text") or "" for q in by[n["id"]].get("properties") or [] if q.get("name") == "Name"),
                        "").strip()
        assert n["authored_name"] == authored
    assert raw["counts"]["isolated_nodes"] == 11


def test_duplicates_grouped_blank_nodes_kept(raw):
    g = G.Graph(raw)
    assert len(g.instances("lateral")) == 2 and len(g.instances("stand")) == 3 and len(g.instances("dip")) == 3
    dup = {m["slug"] for m in raw["moves"] if len(m["nodes"]) > 1}
    assert dup == {"stand", "dip", "lateral", "tight-walk", "snake", "turn", "walk-around", "switch-hands",
                   "side-dip", "lateral-from-behind", "lead-turn", "tilt-left", "tilt-right"}
    blanks = [n for n in raw["nodes"] if n["blank"]]
    assert len(blanks) == 11
    assert all(n["move"] is None and n["slug"] == f"unnamed-{n['id'][:8]}" and n["name"] == "" for n in blanks)
    slugs = [n["slug"] for n in raw["nodes"]]
    assert len(set(slugs)) == len(slugs)
    assert sorted(g.nodes[i]["slug"] for i in g.instances("lateral")) == ["lateral", "lateral-2"]
    assert g.resolve_move("Howl Down") == g.resolve_move("howl Down") == "howl-down"
    assert "viradinha" in g.moves and "corredor" not in g.moves
    # node names are the move's (title-cased) display name; the authored spelling is kept
    hd = [n for n in raw["nodes"] if n["authored_name"] == "howl Down"]
    assert hd and all(n["name"] == "Howl Down" for n in hd)


def test_energy_range_and_bands(raw):
    e = {n["name"]: n["energy"] for n in raw["nodes"]}
    en = np.array([n["energy"] for n in raw["nodes"]])
    assert en.min() == 0.0 and en.max() == 1.0
    assert e["Fast Pirouette"] == 1.0 and e["Dive"] == 0.0
    bands = raw["energy_bands"]
    assert [b["label"] for b in bands] == ["grounded", "low", "travelling", "high", "peak"]
    assert [b["lo"] for b in bands] == [0.0, 0.2, 0.4, 0.6, 0.8] and bands[-1]["hi"] == 1.0
    assert [b["nodes"] for b in bands] == [34, 43, 36, 19, 11]
    for n in raw["nodes"]:
        assert bands[n["band"]]["lo"] <= n["energy"] <= bands[n["band"]]["hi"]


def test_legend_confirmed_by_user(raw):
    """The legend is the one the user confirmed on 2026-10-06 (VIEWER_SPEC 3.10); name agreement is evidence only."""
    lg = raw["legend"]
    assert lg["inferred"] is True and lg["confirmed_by_user"] is True and lg["confirmed"]["date"] == "2026-10-06"
    assert {k: v["label"] for k, v in lg["shapes"].items()} == USER_SHAPES
    assert {k: v["meaning"] for k, v in lg["tones"].items()} == USER_TONES
    for v in lg["shapes"].values():
        assert v["confirmed"] is True and "evidence_note" in v and "adjusted" not in v and "hypothesis" not in v
    assert lg["shapes"]["Cylinder"]["family"] == "rest" and lg["shapes"]["Cylinder"]["agreement"] == 1.0
    assert lg["shapes"]["Star"]["agreement"] == 1.0 and lg["shapes"]["Diamond"]["agreement"] == pytest.approx(0.407)
    assert {k: v["count"] for k, v in lg["tones"].items()} == {"blue": 28, "green": 99, "yellow": 14, "orange": 2}
    assert sum(s["count"] for s in lg["shapes"].values()) == 143


def test_corredor_is_lateral(merged):
    """User decision 2026-10-06: Lateral is renamed Corredor - one move (slug 'lateral'), not a separate node."""
    g_doc, add, vocab = merged
    assert G.validate_graph(g_doc) == []
    g = G.Graph(g_doc)
    assert "corredor" not in g.moves and not any(n["id"] == "proposed-corredor" for n in g_doc["nodes"])
    for text in ("corredor", "Corredor", "el corredor", "corridor", "lateral", "Lateral"):
        assert g.resolve_move(text) == "lateral"
    assert sorted(g.instances(g.resolve_move("corredor"))) == sorted(LATERAL)
    m = g.moves["lateral"]
    assert m["name"] == m["display_name"] == "Corredor" and m["name_en"] == "Lateral"
    assert m["display_source"] == "additions" and "Lateral" in m["aliases"]
    assert "https://example.org/corredor" in m["sources"]
    for i in LATERAL:
        assert g.nodes[i]["name"] == "Corredor" and g.nodes[i]["authored_name"] == "Lateral"
    # no alias collision warnings any more (the research maps Corredor onto Lateral)
    assert not any("corredor" in w.lower() for w in g_doc["warnings"])
    # a narrated "corredor" is the lateral: same path, no new links
    a = lab([(0, 2, "open-break"), (2, 4, "corredor"), (4, 6, "tight-walk")])
    assert normalise_moves(a, g) == [] and a.segments[1].move == "lateral"
    b = lab([(0, 2, "open-break"), (2, 4, "lateral"), (4, 6, "tight-walk")])
    da, db = resolve_path(a, g), resolve_path(b, g)
    assert [s["node"] for s in da["steps"]] == [s["node"] for s in db["steps"]]
    assert not any(s["new_link"] for s in da["steps"])
    # even without the research file the additions keep the decision
    g2 = G.Graph(G.build_graph(ZOUK1_PATH, add, NOFILE))
    assert g2.resolve_move("corredor") == "lateral" and g2.moves["lateral"]["name"] == "Corredor"


def test_vocabulary_merge(merged):
    g_doc, add, vocab = merged
    g = G.Graph(g_doc)
    # display names: confirmed / likely used, translation not
    assert g.moves["hug"]["name"] == "Abraço" and g.moves["hug"]["display_confidence"] == "likely"
    assert g.resolve_move("Hug") == g.resolve_move("abraço") == g.resolve_move("close embrace") == "hug"
    assert "Hug" in g.moves["hug"]["aliases"]
    assert g.moves["simple-turn"]["name"] == "Simple Turn" and g.resolve_move("giro") == "simple-turn"
    # proposed moves from the research: dashed nodes at their energy height
    ymin, ymax = g_doc["y_range"]
    for slug, energy in (("chicote", 0.7), ("piao", 0.58), ("counterbalance-pivot", 0.62)):
        assert g.moves[slug]["status"] == "proposed"
        n = g.nodes[f"proposed-{slug}"]
        assert n["status"] == "proposed" and n["energy"] == pytest.approx(energy)
        assert n["position"][1] == pytest.approx(ymin + energy * (ymax - ymin), abs=1e-5)
    assert g.moves["chicote"]["name"] == "Chicote" and g.resolve_move("whip") == "chicote"
    assert g.moves["chicote"]["sources"] == ["https://example.org/chicote"]
    pp = g.position("proposed-piao")
    assert np.hypot(pp[0] - 0.2, pp[2] + 0.3) < 0.75  # at (or beside) the research position hint
    # an existing proposed move (from additions) only gets aliases + the display name, not the row's links
    cp = g.moves["counterbalance-pivot"]
    assert cp["name"] == "Contrapeso" and cp["name_en"] == "Counterbalance Pivot"
    assert g.resolve_move("planada") == g.resolve_move("Counterbalance Pivot") == "counterbalance-pivot"
    prop = [l for l in g_doc["links"] if l["status"] == "proposed"]
    pairs = {(l["from_move"], l["to_move"]) for l in prop}
    assert pairs == {("lateral", "counterbalance-pivot"), ("open-break", "counterbalance-pivot"),
                     ("hug", "counterbalance-pivot"), ("counterbalance-pivot", "walk-around"),
                     ("counterbalance-pivot", "pirouette"), ("hug", "piao"), ("piao", "hug")}
    assert all(l["id"].startswith("proposed-") and l["accepted"] is False for l in prop)
    # a scaffold row naming no graph move becomes a proposed move, with a warning
    assert g.moves["sacadinha"]["status"] == "proposed"
    assert any("sacadinha" in w and "names no graph move" in w for w in g_doc["warnings"])
    # legacy aliases, unknown targets, unresolved aliases passed through, proposed_links not merged
    assert g.resolve_move("solto") == "soltinho"
    assert any("nonexistent-move" in w for w in g_doc["warnings"])
    assert g.resolve_move("cicada") is None
    assert g_doc["unresolved_aliases"][0]["alias"] == "cicada" and "sacada" in g_doc["unresolved_aliases"][0][
        "candidates"]
    assert ("lateral", "yoyo") not in pairs
    # every node shows its move's display name
    for n in g_doc["nodes"]:
        if n["move"]:
            assert n["name"] == g.moves[n["move"]]["name"] == n["display_name"]
    assert g_doc["counts"]["scaffold_nodes"] == 143 and g_doc["counts"]["scaffold_links"] == 181
    # re-runnable and deterministic; without the vocabulary file the research-only moves disappear
    again = G.build_graph(ZOUK1_PATH, add, vocab)
    assert json.dumps(again, sort_keys=True) == json.dumps(g_doc, sort_keys=True)
    bare = G.build_graph(ZOUK1_PATH, add, NOFILE)
    assert "chicote" not in {m["slug"] for m in bare["moves"]}
    assert next(m for m in bare["moves"] if m["slug"] == "counterbalance-pivot")["name"] == "Counterbalance Pivot"


def test_additions_win_and_link_choices(tmp_path):
    """additions display_name beats the research; accepted links stay 'proposed' (contract) with accepted:true;
    proposed links prefer connected instances over isolated ones."""
    add = tmp_path / "additions.toml"
    add.write_text(G.ADDITIONS_TEMPLATE + '\n[[links]]\nfrom = "lateral"\nto = "yoyo"\nstatus = "accepted"\n'
                   + '\n[[moves]]\nslug = "test-move"\nname = "Test Move"\ndisplay_name = "Movimento"\n'
                   + 'energy = 0.3\nnear = ["hug"]\n', encoding="utf-8")
    vocab = tmp_path / "v.json"
    vocab.write_text(json.dumps({"vocabulary": [
        {"slug": "test-move", "display_name": "Outro Nome", "display_confidence": "confirmed", "status": "scaffold"},
        {"slug": "lateral", "display_name": "Lateral BR", "display_confidence": "confirmed", "status": "scaffold"}]}))
    g_doc = G.build_graph(ZOUK1_PATH, add, vocab)
    assert G.validate_graph(g_doc) == []
    g = G.Graph(g_doc)
    assert g.moves["test-move"]["name"] == "Movimento" and g.moves["lateral"]["name"] == "Corredor"
    acc = next(l for l in g_doc["links"] if l["from_move"] == "lateral" and l["to_move"] == "yoyo")
    assert acc["status"] == "proposed" and acc["accepted"] is True
    to_wa = next(l for l in g_doc["links"] if l["from_move"] == "counterbalance-pivot" and l["to_move"] ==
                 "walk-around")
    assert to_wa["to"] == WALK_AROUND_CONNECTED


def test_same_move_merges_instances(tmp_path):
    """additions [same_move]: Little Turn is the Viradinha (user 2026-10-06) -> one move with two node instances;
    the old slug and the English name resolve to it; links and a later display name follow the merge."""
    add = tmp_path / "additions.toml"
    add.write_text(G.ADDITIONS_TEMPLATE + '\n[same_move]\nlittle-turn = "viradinha"\nno-such-move = "hug"\n',
                   encoding="utf-8")
    g_doc = G.build_graph(ZOUK1_PATH, add, NOFILE)
    assert G.validate_graph(g_doc) == []
    g = G.Graph(g_doc)
    assert "little-turn" not in g.moves
    vir = g.moves["viradinha"]
    assert len(vir["nodes"]) == 2 and "Little Turn" in vir["aliases"]
    insts = sorted(n["instance"] for n in g_doc["nodes"] if n["move"] == "viradinha")
    assert insts == [1, 2]
    assert not any("little-turn" in (l["from_move"], l["to_move"]) for l in g_doc["links"])
    # Zouk1 has Little Turn -> Viradinha: after the merge that link joins two instances of one move
    assert any(l["from_move"] == "viradinha" and l["to_move"] == "viradinha" for l in g_doc["links"])
    assert any("no-such-move" in w for w in g_doc["warnings"])


def test_blank_junctions_connect_named_moves(raw):
    """Unnamed Zouk1 nodes are junctions: Tilt Left/Right -> (blank) -> Howl Up is a graph connection."""
    g = G.Graph(raw)
    assert raw["counts"]["via_links"] == len(g.via) == 3
    for a, b, j in (("tilt-left", "howl-up", JUNCTION_HOWL), ("tilt-right", "howl-up", JUNCTION_HOWL),
                    ("tilt-right", "side-lunge", JUNCTION_LUNGE)):
        doc = resolve_path(lab([(0, 2, a), (2, 4, b)]), g)
        st = doc["steps"][1]
        assert not st["new_link"] and st["via"] == [j] and len(st["links"]) == 2 and st["link"] == st["links"][0]
        assert doc["new_links"] == [] and validate_path(doc, g) == []
        # walked backwards: reverse, still not new (unless strict_direction)
        back = resolve_path(lab([(0, 2, b), (2, 4, a)]), g)
        assert back["steps"][1]["reverse"] and not back["steps"][1]["new_link"] and back["steps"][1]["via"] == [j]
        assert validate_path(back, g) == []
        strict = resolve_path(lab([(0, 2, b), (2, 4, a)]), g, PathParams(strict_direction=True))
        assert strict["steps"][1]["new_link"] and validate_path(strict, g) == []


def test_real_taxonomy_files():
    """The tracked inputs: additions.toml + vocabulary_research.json (when the research has written it)."""
    if not (G.ADDITIONS_PATH.exists() and G.VOCABULARY_PATH.exists()):
        pytest.skip("taxonomy inputs not present")
    g_doc = G.build_graph()
    assert G.validate_graph(g_doc) == [] and g_doc["warnings"] == []
    g = G.Graph(g_doc)
    assert g.resolve_move("corredor") == "lateral" and g.moves["lateral"]["name"] == "Corredor"
    assert g.resolve_move("veraginha") == "viradinha"
    assert g.moves["counterbalance-pivot"]["name"] == "Contrapeso"
    assert g.moves["piao"]["status"] == "proposed" and g.resolve_move("pião") == "piao"
    assert "corredor" not in g.moves
    # user decisions 2026-10-06: "cicada" was sarrada (accepted move); Little Turn is the Viradinha;
    # Brazilian Portuguese names from the terminology research, English names kept as aliases
    assert g.resolve_move("cicada") == "sarrada" and g.moves["sarrada"]["status"] == "accepted"
    assert g.resolve_move("little-turn") == "viradinha" and "little-turn" not in g.moves
    assert g.moves["hug"]["name"] == "Abraço" and g.resolve_move("hug") == "hug"
    assert g.moves["simple-turn"]["name"] == "Giro Simples" and "Simple Turn" in g.moves["simple-turn"]["aliases"]


# ---------------------------------------------------------------------------------------------------------------
# labels
# ---------------------------------------------------------------------------------------------------------------

def test_labels_validation(raw, tmp_path):
    g = G.Graph(raw)
    good = lab([(0.0, 2.0, "hug"), (2.0, 4.5, "lateral"), (5.0, 6.0, "viradinha", {"confidence": 0.8})])
    assert validate_labels(good, g) == []
    assert "overlap" in " ".join(validate_labels(lab([(0, 2, "hug"), (1.5, 3, "lateral")]), g))
    assert "t0 < t1" in " ".join(validate_labels(lab([(2, 2, "hug")]), g))
    assert "unknown move" in " ".join(validate_labels(lab([(0, 1, "corredor-xyz")]), g))
    assert "alias of 'howl-down'" in " ".join(validate_labels(lab([(0, 1, "Howl Down")]), g))
    assert "provenance" in " ".join(validate_labels(lab([(0, 1, "hug", {"provenance": "guess"})]), g))
    assert "confidence" in " ".join(validate_labels(lab([(0, 1, "hug", {"confidence": 1.5})]), g))
    lateral_node = g.instances("lateral")[0]
    assert "is move 'lateral'" in " ".join(validate_labels(lab([(0, 1, "hug", {"node": lateral_node})]), g))
    assert validate_labels(lab([(0, 1, "lateral", {"node": lateral_node})]), g) == []
    v = lab([(0, 1, "hug")])
    v.time_base = "video"
    assert any("needs 'video'" in e for e in validate_labels(v, g))
    # other tiers may not exist yet; couple only
    assert "tier" in " ".join(validate_labels(lab([(0, 1, "hug", {"tier": "lead"})]), g))
    # round trip keeps unknown keys
    good.segments[0].extra["proposal_id"] = "p-1"
    good.extra["annotation_session"] = {"by": "john"}
    p = save_labels(good, tmp_path / "labels.json", g)
    back = load_labels(p)
    assert back.segments[0].extra["proposal_id"] == "p-1" and back.extra["annotation_session"] == {"by": "john"}
    assert back.to_dict() == good.to_dict()
    with pytest.raises(ValueError):
        save_labels(lab([(0, 2, "hug"), (1, 3, "hug")]), tmp_path / "bad.json", g)


def test_capture_time_base_only_in_unity_copies(raw):
    g = G.Graph(raw)
    c = lab([(10.0, 12.0, "hug")])
    c.time_base = "capture"
    assert any("only used in the Unity copies" in e for e in validate_labels(c, g))
    with pytest.raises(LabelsError):  # no silent time_to_audio = 0
        to_reference(c)
    c.extra["time_to_audio"] = -2.0
    assert to_reference(c).segments[0].t0 == pytest.approx(12.0)


def test_video_time_base_to_reference():
    from dancecap.moves.labels import reference_to_video, video_to_reference
    from dancecap.take import Take
    try:
        tk = Take.load("larissa_kadu")
    except FileNotFoundError:
        pytest.skip("larissa_kadu alignment not available")
    src = next(s for s in tk.sources if s.id == "02")
    v = lab([(10.0, 12.0, "hug")], take="larissa_kadu")
    v.time_base, v.video = "video", str(src.path)
    r = to_reference(v)
    assert r.time_base == "reference" and r.video is None and r.extra["converted_from"]["camera"] == "02"
    assert r.segments[0].t0 == pytest.approx(src.to_reference(10.0))
    assert video_to_reference(10.0, "larissa_kadu", src.path.name) == pytest.approx(src.to_reference(10.0))
    assert reference_to_video(video_to_reference(10.0, "larissa_kadu", "02"), "larissa_kadu", "02") == \
        pytest.approx(10.0)


# ---------------------------------------------------------------------------------------------------------------
# path (Viterbi)
# ---------------------------------------------------------------------------------------------------------------

def _mini_graph():
    """A -> X2 is linked; X1 (same move) sits right next to A but has no link; X1 -> B, X2 -> B both linked."""
    nodes = []
    for nid, move, pos in (("A", "a", [0, 0, 0]), ("X1", "x", [0.1, 0, 0]), ("X2", "x", [3, 0, 0]),
                           ("B", "b", [3, 1, 0]), ("C", "c", [9, 0, 0])):
        nodes.append({"id": nid, "name": move, "slug": nid.lower(), "move": move, "shape": "Diamond",
                      "tone": [0, 0, 0], "tone_name": "green", "family": "step", "position": pos, "energy": 0.5,
                      "band": 2, "status": "scaffold"})
    links = [{"id": "l1", "from": "A", "to": "X2", "status": "scaffold"},
             {"id": "l2", "from": "X1", "to": "B", "status": "scaffold"},
             {"id": "l3", "from": "X2", "to": "B", "status": "scaffold"}]
    moves = [{"slug": s, "name": s, "aliases": [], "nodes": [n["id"] for n in nodes if n["move"] == s],
              "status": "scaffold"} for s in "axbc"]
    bands = [{"band": b, "lo": lo, "hi": hi, "label": lbl} for b, lo, hi, lbl in G.ENERGY_BANDS]
    return G.Graph({"version": 1, "nodes": nodes, "links": links, "moves": moves, "energy_bands": bands})


def test_viterbi_chooses_linked_instance():
    g = _mini_graph()
    doc = resolve_path(lab([(0, 1, "a"), (1, 2, "x"), (2, 3, "b")]), g)
    assert [s["node"] for s in doc["steps"]] == ["A", "X2", "B"]  # not the nearer, unlinked X1
    assert [s["link"] for s in doc["steps"]] == [None, "l1", "l3"]
    assert not any(s["new_link"] for s in doc["steps"]) and doc["new_links"] == []
    assert validate_path(doc, g) == []


def test_validate_path_is_strict(raw):
    """A link id that does not belong to the transition, a wrong new_link flag or a path that no longer matches the
    labels are all reported."""
    g = G.Graph(raw)
    labels = lab([(0, 2, "open-break"), (2, 4, "lateral"), (4, 6, "tight-walk")])
    doc = resolve_path(labels, g)
    assert validate_path(doc, g, labels) == []
    bad = json.loads(json.dumps(doc))
    bad["steps"][1]["link"] = raw["links"][0]["id"] if raw["links"][0]["id"] != doc["steps"][1]["link"] else \
        raw["links"][1]["id"]
    assert any("does not belong" in e for e in validate_path(bad, g))
    bad = json.loads(json.dumps(doc))
    bad["steps"][1]["new_link"] = True
    assert any("new_link=True" in e for e in validate_path(bad, g))
    edited = lab([(0, 2, "open-break"), (2, 4.5, "lateral"), (4.5, 6, "tight-walk")])
    assert any("stale path" in e for e in validate_path(doc, g, edited))
    assert any("stale path" in e for e in validate_path(doc, g, lab([(0, 2, "open-break")])))


def test_viterbi_on_zouk1_open_break_to_lateral(raw):
    g = G.Graph(raw)
    ob = g.instances("open-break")[0]
    linked = [i for i in g.instances("lateral") if g.link(ob, i)]
    assert len(linked) == 1
    doc = resolve_path(lab([(0, 2, "open-break"), (2, 4, "lateral"), (4, 6, "tight-walk")]), g)
    assert doc["steps"][1]["node"] == linked[0]
    assert doc["steps"][1]["link"] == g.link(ob, linked[0])["id"] and not doc["steps"][1]["new_link"]
    assert not doc["steps"][2]["new_link"]
    assert validate_path(doc, g) == []


def test_new_link_reverse_and_break(raw):
    g = G.Graph(raw)
    fp, st = g.instances("fast-pirouette"), g.instances("stand")
    assert not any(g.link(a, b) or g.link(b, a) for a in fp for b in st)
    doc = resolve_path(lab([(0, 1, "fast-pirouette"), (1, 2, "stand"), (2, 3, "fast-pirouette"),
                            (3, 4, "stand")]), g)
    assert [s["new_link"] for s in doc["steps"]] == [False, True, True, True]
    nl = {(d["from_move"], d["to_move"]): d["count"] for d in doc["new_links"]}
    assert nl == {("fast-pirouette", "stand"): 2, ("stand", "fast-pirouette"): 1}
    # an authored link walked backwards: not new by default, new with strict_direction
    lk = next(l for l in raw["links"] if l["from_move"] == "hug" and l["to_move"] == "tilt-right")
    back = lab([(0, 1, "tilt-right", {"node": lk["to"]}), (1, 2, "hug", {"node": lk["from"]})])
    d1 = resolve_path(back, g)
    assert d1["steps"][1]["reverse"] and not d1["steps"][1]["new_link"] and d1["steps"][1]["link"] == lk["id"]
    d2 = resolve_path(back, g, PathParams(strict_direction=True))
    assert d2["steps"][1]["new_link"] and d2["steps"][1]["link"] is None
    assert validate_path(d2, g) == []
    # a long gap breaks the path: no transition, no new link
    d3 = resolve_path(lab([(0, 1, "fast-pirouette"), (5, 6, "stand")]), g)
    assert d3["steps"][1]["break_before"] and not d3["steps"][1]["new_link"] and d3["new_links"] == []
    # consecutive segments of one move stay on one node
    d4 = resolve_path(lab([(0, 1, "hug"), (1, 2, "hug")]), g)
    assert d4["steps"][1]["stay"] and d4["steps"][0]["node"] == d4["steps"][1]["node"]


# ---------------------------------------------------------------------------------------------------------------
# fingerprint
# ---------------------------------------------------------------------------------------------------------------

def test_fingerprint_sums_and_counterbalance(raw):
    g = G.Graph(raw)
    labels = lab([(0, 1.5, "hug"), (1.5, 3.0, "tilt-right"), (3.0, 4.0, "hug"), (4.0, 7.5, "lateral"),
                  (7.5, 8.0, "lateral"), (10.0, 11.0, "fast-pirouette")])
    doc = resolve_path(labels, g)
    cb = {"version": 1, "time_base": "reference", "intervals": [{"t0": 1.0, "t1": 2.0}, {"t0": 1.5, "t1": 2.5},
                                                                 {"t0": 10.5, "t1": 20.0}]}
    fp = compute_fingerprint(doc, g, cb, "test")
    assert validate_fingerprint(fp) == []
    assert fp["labelled_seconds"] == pytest.approx(labels.labelled_seconds) == pytest.approx(9.0)
    assert sum(v["seconds"] for v in fp["nodes"].values()) == pytest.approx(fp["labelled_seconds"])
    assert sum(b["seconds"] for b in fp["energy_bands"]) == pytest.approx(fp["labelled_seconds"])
    assert sum(v["share"] for v in fp["nodes"].values()) == pytest.approx(1.0)
    hug = doc["steps"][0]["node"]
    assert fp["nodes"][hug]["seconds"] == pytest.approx(2.5) and fp["nodes"][hug]["visits"] == 2
    lat = doc["steps"][3]["node"]
    assert doc["steps"][4]["node"] == lat and fp["nodes"][lat]["visits"] == 1
    assert fp["distinct_moves"] == 4
    assert fp["transitions"] == 3 == sum(fp["links"].values())  # hug>tilt, tilt>hug, hug>lateral; gap breaks
    # no analysed range in the file: the whole labelled time counts as analysed
    # merged counterbalance [1, 2.5) + [10.5, 20) overlap labelled time: 1.5 + 0.5 = 2.0 s
    assert fp["counterbalance_seconds"] == pytest.approx(2.0) and fp["counterbalance_analysed"] is None
    assert fp["counterbalance_share"] == pytest.approx(2.0 / 9.0)
    assert compute_fingerprint(doc, g)["counterbalance_share"] is None


def test_counterbalance_share_over_the_analysed_range(raw):
    """The detection only ran on source.t_range (larissa_kadu: 56-62 s of a dance labelled 30-65 s): the share is
    counterbalance time / labelled time INSIDE that range, null when the labels do not overlap it."""
    g = G.Graph(raw)
    labels = lab([(0, 4, "hug"), (4, 8, "lateral"), (8, 20, "tight-walk")])
    doc = resolve_path(labels, g)
    cb = {"version": 1, "time_base": "reference", "source": {"t_range": [6.0, 10.0]},
          "intervals": [{"t0": 6.0, "t1": 9.0}]}
    fp = compute_fingerprint(doc, g, cb, "test")
    assert fp["counterbalance_analysed"] == [6.0, 10.0] and fp["counterbalance_analysed_seconds"] == pytest.approx(4)
    assert fp["counterbalance_seconds"] == pytest.approx(3.0) and fp["counterbalance_share"] == pytest.approx(0.75)
    assert validate_fingerprint(fp) == []
    cb["source"]["t_range"] = [30.0, 40.0]
    fp2 = compute_fingerprint(doc, g, cb, "test")
    assert fp2["counterbalance_share"] is None and fp2["counterbalance_analysed_seconds"] == 0.0
    # the track's t range is used when source.t_range is missing
    cb2 = {"version": 1, "time_base": "reference", "track": {"t": [7.0, 8.0, 9.0]}, "intervals": []}
    fp3 = compute_fingerprint(doc, g, cb2, "test")
    assert fp3["counterbalance_analysed"] == [7.0, 9.0] and fp3["counterbalance_share"] == 0.0


def test_fingerprint_records_sources(tmp_path):
    a, b = tmp_path / "labels.json", tmp_path / "path.json"
    a.write_text("{}")
    b.write_text("{}")
    files = {"labels": a, "path": b, "counterbalance": tmp_path / "missing.json"}
    fp = {"sources": source_record(files)}
    assert fp["sources"]["counterbalance"] is None and fp["sources"]["labels"]["sha256_16"]
    assert stale_sources(fp, files) == []
    b.write_text('{"x": 1}')
    assert any("path changed" in e for e in stale_sources(fp, files))
    (tmp_path / "missing.json").write_text("{}")
    assert any("counterbalance changed" in e for e in stale_sources(fp, files))
    assert stale_sources({}, files) and "not recorded" in stale_sources({}, files)[0]


# ---------------------------------------------------------------------------------------------------------------
# placeholder
# ---------------------------------------------------------------------------------------------------------------

def test_measure_bounds():
    grid = [[0.5 * i, 1] for i in range(8 * 3 + 2)]  # 3 full measures + 2 extra slots (no type information)
    assert measure_bounds(grid) == [0.0, 4.0, 8.0, 12.0]
    assert measure_bounds(grid[:16]) == [0.0, 4.0, 8.0]  # end extrapolated


def test_measure_phase_from_beat_types():
    pat = [1, 3, 3, 2, 1, 3, 2, 3]
    grid = [[0.5 * i, pat[(i - 3) % 8]] for i in range(8 * 4 + 3)]  # first downbeat at slot 3
    assert measure_phase(grid) == 3
    assert measure_bounds(grid) == [1.5, 5.5, 9.5, 13.5, 17.5]
    assert unit_bounds(grid, 2)[:3] == [0.5, 1.5, 2.5]  # quarter measures share the phase (mod 2)
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        measure_phase([[0.5 * i, int(rng.integers(1, 4))] for i in range(64)])


@pytest.mark.skipif(not (PACKAGE_ROOT / "work" / "synthetic" / "beats.json").exists(), reason="no synthetic take")
def test_placeholder_synthetic(raw):
    g = G.Graph(raw)
    a = make_placeholder("synthetic", g, seed=7)
    b = make_placeholder("synthetic", g, seed=7)
    assert [s.to_dict() | {"created": None} for s in a.segments] == [s.to_dict() | {"created": None}
                                                                      for s in b.segments]
    assert validate_labels(a, g) == []
    assert a.extra["placeholder"] is True and "NOT AN ANALYSIS" in a.extra["note"]
    assert all(s.provenance == "placeholder" and s.node is None for s in a.segments)
    bounds = measure_bounds(json.loads((PACKAGE_ROOT / "work" / "synthetic" / "beats.json").read_text()))
    for s in a.segments:
        assert any(abs(s.t0 - x) < 1e-5 for x in bounds) and any(abs(s.t1 - x) < 1e-5 for x in bounds)
        assert s.extra["measures"] in (1, 2)
    assert all(abs(x.t1 - y.t0) < 1e-6 for x, y in zip(a.segments, a.segments[1:]))
    assert a.segments[0].t0 == pytest.approx(bounds[0]) and a.segments[-1].t1 == pytest.approx(bounds[-1])
    walk = [g.nodes[s.extra["placeholder_node"]] for s in a.segments]
    assert walk[0]["band"] == 0
    doc = resolve_path(a, g)
    assert sum(s["new_link"] for s in doc["steps"]) == 1  # the one deliberate jump
    assert [s["node"] for s in doc["steps"]] == [n["id"] for n in walk]  # the path recovers the walk
    # the walk climbs grounded -> high/peak -> grounded (seeded noise: almost always, not for every seed)
    climbs = 0
    for seed in range(10):
        w = [g.nodes[s.extra["placeholder_node"]]["band"] for s in make_placeholder("synthetic", g, seed=seed).segments]
        climbs += w[0] == 0 and max(w) >= 3 and w[-1] <= 1
    assert climbs >= 8


@pytest.mark.skipif(not (PACKAGE_ROOT / "work" / "synthetic" / "beats.json").exists(), reason="no synthetic take")
def test_placeholder_short_span_uses_sub_measures(raw):
    """A span shorter than MIN_UNITS measures (larissa_kadu's 6-s window) is cut in half / quarter measures and
    covered fully, so the walk still climbs the bands and shows one new link inside the capture."""
    g = G.Graph(raw)
    a = make_placeholder("synthetic", g, seed=3, span=(5.0, 11.0))
    gen = a.extra["generator"]
    assert gen["unit"] in ("half measure", "quarter measure") and len(a.segments) >= 6
    assert a.segments[0].t0 <= 5.0 + 1e-6 and a.segments[-1].t1 >= 11.0 - 1e-6
    assert all(s.extra["unit_eighths"] in (2, 4) for s in a.segments)
    doc = resolve_path(a, g)
    assert sum(s["new_link"] for s in doc["steps"]) == 1
    assert max(g.nodes[s["node"]]["band"] for s in doc["steps"]) >= 3


# ---------------------------------------------------------------------------------------------------------------
# Unity export
# ---------------------------------------------------------------------------------------------------------------

def _capture(folder: Path, take: str, times: list[float], **man) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "times.json").write_text(json.dumps(times))
    (folder / "capture.json").write_text(json.dumps({"version": 3, "times": "times.json",
                                                     "provenance": {"take": take}, **man}))
    return folder


def test_capture_clock_and_conversion(raw, tmp_path):
    times = [56.0 + i / 30 for i in range(180)]
    c = CaptureClock.from_capture(_capture(tmp_path / "Cap", "tk", times, time_to_audio=-29.9713246))
    assert c.to_capture(56.0) == pytest.approx(26.0286754) and c.to_reference(c.to_capture(60.0)) == 60.0
    assert c.frame(56.0) == 0 and c.frame(57.0) == 30 and c.frame(100.0) == 179 and c.frame(0.0) == 0
    labels = lab([(50.0, 57.0, "hug"), (57.0, 61.0, "lateral"), (70.0, 72.0, "hug")], take="tk").to_dict()
    out = convert_labels(labels, c)
    assert out["time_base"] == "capture" and out["frame"] == "unity" and out["time_to_audio"] == -29.9713246
    s0, s1, s2 = out["segments"]
    assert s1["t0"] == pytest.approx(57.0 - 29.9713246) and s1["t0_reference"] == 57.0
    assert (s1["f0"], s1["f1"]) == (30, 150)
    assert s0["in_capture"] and s1["in_capture"] and not s2["in_capture"]
    assert out["capture_range"] == pytest.approx([26.0286754, times[-1] - 29.9713246], abs=1e-6)
    p = convert_path({"version": 1, "take": "tk", "provenance": "human", "steps": [
        {"t0": 57.0, "t1": 61.0, "node": "n", "move": "m", "link": None, "new_link": False}], "new_links": []}, c)
    assert p["steps"][0]["t1"] == pytest.approx(61.0 - 29.9713246) and p["time_base"] == "capture"
    # v2 rule when time_to_audio is absent: audio_offset - times[0]
    c2 = CaptureClock.from_capture(_capture(tmp_path / "Cap2", "tk", times, audio_offset=1.5))
    assert c2.time_to_audio == pytest.approx(1.5 - 56.0)


@pytest.mark.skipif(not (PACKAGE_ROOT / "work" / "synthetic" / "moves" / "labels.json").exists(),
                    reason="run `python -m dancecap.moves placeholder synthetic` first")
def test_export_end_to_end(tmp_path):
    times = [i / 30 for i in range(720)]
    cdir = _capture(tmp_path / "_pytest_moves_capture", "synthetic", times, time_to_audio=-0.5)
    rep = export_moves("synthetic", cdir, streaming=tmp_path)
    assert validate_export(cdir) == []
    lab_ref = json.loads((PACKAGE_ROOT / "work" / "synthetic" / "moves" / "labels.json").read_text())
    lab_cap = json.loads((cdir / "moves" / "labels.json").read_text())
    for a, b in zip(lab_ref["segments"], lab_cap["segments"]):
        assert b["t0"] == pytest.approx(a["t0"] - 0.5) and b["t0_reference"] == pytest.approx(a["t0"])
    g = json.loads((cdir / "moves" / "graph.json").read_text())
    assert g["frame"] == "unity" and g["time_base"] == "capture" and len(g["nodes"]) >= 143
    assert rep["segments"] == len(lab_ref["segments"])
    assert not (PACKAGE_ROOT / "work" / "synthetic" / "export" / "_pytest_moves_capture").exists()


@pytest.mark.skipif(not (PACKAGE_ROOT / "work" / "synthetic" / "moves" / "labels.json").exists(),
                    reason="run `python -m dancecap.moves placeholder synthetic` first")
def test_export_refuses_stale_path(tmp_path, monkeypatch):
    from dancecap.moves import export as E
    real = json.loads(E.path_file("synthetic").read_text(encoding="utf-8"))
    stale = json.loads(json.dumps(real))
    stale["steps"] = stale["steps"][:-1]  # labels edited after the path was made
    sp = tmp_path / "path.json"
    sp.write_text(json.dumps(stale))
    monkeypatch.setattr(E, "path_file", lambda take: sp)
    monkeypatch.setattr(E, "fingerprint_file", lambda take: tmp_path / "no_fingerprint.json")
    cdir = _capture(tmp_path / "_pytest_stale_capture", "synthetic", [i / 30 for i in range(30)], time_to_audio=0.0)
    with pytest.raises(ExportError, match="stale"):
        export_moves("synthetic", cdir, streaming=tmp_path)
    assert not (cdir / "moves").exists()
    export_moves("synthetic", cdir, streaming=tmp_path, allow_stale=True)
    assert (cdir / "moves" / "labels.json").exists()


# ---------------------------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------------------------

def test_cli_core_commands_and_optional_registration(monkeypatch):
    from dancecap.moves import __main__ as cli

    def register(sub):
        p = sub.add_parser("features-demo")
        p.set_defaults(func=lambda a: 42)

    def broken(sub):
        raise RuntimeError("boom")

    monkeypatch.setitem(sys.modules, "dancecap.moves.features", types.SimpleNamespace(register=register))
    monkeypatch.setitem(sys.modules, "dancecap.moves.model", types.SimpleNamespace(register=broken))
    ap = cli.build_parser()
    sub = next(a for a in ap._actions if a.__class__.__name__ == "_SubParsersAction")
    assert {"graph", "placeholder", "path", "fingerprint", "export", "validate", "review", "run",
            "features-demo"} <= set(sub.choices)
    assert cli.main(["features-demo"]) == 42


def test_written_graph_is_current():
    """taxonomy/zouk_graph.json (tracked) matches a fresh build from the current inputs."""
    if not G.GRAPH_PATH.exists():
        pytest.skip("graph not built yet")
    on_disk = json.loads(G.GRAPH_PATH.read_text(encoding="utf-8"))
    fresh = G.build_graph()
    assert json.dumps(on_disk, sort_keys=True) == json.dumps(json.loads(json.dumps(fresh)), sort_keys=True), \
        "run `python -m dancecap.moves graph`"


def teardown_module(module):
    shutil.rmtree(PACKAGE_ROOT / "work" / "synthetic" / "export" / "_pytest_moves_capture", ignore_errors=True)
    shutil.rmtree(PACKAGE_ROOT / "work" / "synthetic" / "export" / "_pytest_stale_capture", ignore_errors=True)
