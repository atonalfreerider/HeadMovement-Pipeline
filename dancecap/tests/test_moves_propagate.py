"""dancecap.moves.annotate.propagate: one narration -> the labels.json of every take on the same reference clock.

    ..\\.venv-nopo4d\\Scripts\\python -m pytest tests/test_moves_propagate.py -q

Covers: the same-sequence guard, stable ids across runs and takes, clipping to a take's window ('narrated_span', b0/b1
re-indexed), review state (confirmed / edited items carried into every take with the narration clipped around them,
deleted items stay deleted, a window-clipped item gets its narrated extent back in a wider take), and a dry run on the
real Larissa/Kadu narration when its work files exist (nothing written).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dancecap.moves import ZOUK1_PATH  # noqa: E402
from dancecap.moves.annotate.propagate import (PropagationError, clip, compare_sequences, compose,  # noqa: E402
                                               review_state, snapshot, stable_ids, unclip)
from dancecap.moves.annotate.timemap import Grid  # noqa: E402
from dancecap.take import PACKAGE_ROOT  # noqa: E402

EIGHTH = 60.0 / 76.4 / 2


def seg(i, t0, t1, move="corredor", prov="narration", word=None, anchor=None, **kw):
    d = {"id": f"nar-{i}", "t0": t0, "t1": t1, "move": move, "provenance": prov, "tier": "couple",
         "anchor_word": word or (move or "unknown"), "anchor_t": anchor if anchor is not None else t0 + 0.4,
         "narration": "N1", "created": "2026-10-07T00:00:00+00:00", **kw}
    if move is None:
        d["unresolved"] = kw.get("unresolved", "tilted")
    return d


def grid(t0=0.0, n=400):
    return Grid([[t0 + k * EIGHTH, 3 if k % 2 == 0 else 1] for k in range(n)])


def test_sequence_guard():
    a = [seg(1, 0, 2), seg(2, 2, 4, "viradinha"), seg(3, 4, 5, None)]
    b = [dict(s, t0=s["t0"] + 0.02, t1=s["t1"] + 0.02) for s in a]
    d = compare_sequences({"src": a, "other": b}, "src")
    assert d["other"]["max_dt0_ms"] == pytest.approx(20.0, abs=0.1)
    c = [a[0], dict(a[1], move="sarrada"), a[2]]
    with pytest.raises(PropagationError, match="item 1"):
        compare_sequences({"src": a, "other": c}, "src")
    with pytest.raises(PropagationError):
        compare_sequences({"src": a, "other": a[:2]}, "src")


def test_stable_ids_follow_the_narrated_words():
    old = [seg(1, 10, 12, word="corridor", anchor=10.40), seg(2, 12, 14, "viradinha", word="viradinha", anchor=12.4)]
    new = [dict(seg("x1", 10, 12, word="corridor", anchor=10.41), created="NEW"),
           dict(seg("x2", 12, 14, "viradinha", word="viradinha", anchor=12.45), created="NEW"),
           dict(seg("x3", 14, 15, "sarrada", word="sarrada", anchor=14.4), created="NEW")]
    ren = stable_ids(new, old)
    assert [s["id"] for s in new] == ["nar-1", "nar-2", "nar-x3"]
    assert ren == {"nar-x1": "nar-1", "nar-x2": "nar-2"}
    assert new[0]["created"] == old[0]["created"] and new[2]["created"] == "NEW"
    # another word, or too far away: a new id
    far = [seg("y", 10, 12, word="corridor", anchor=10.6)]
    assert stable_ids(far, old) == {} and far[0]["id"] == "nar-y"


def test_clip_to_window():
    g = grid()
    items = [seg(1, 1.0, 3.0), seg(2, 3.0, 6.0, "viradinha"), seg(3, 6.0, 8.0, None), seg(4, 8.0, 9.0, "sarrada")]
    kept, dropped = clip(items, (2.0, 6.1), g)
    assert [s["id"] for s in kept] == ["nar-1", "nar-2"]
    assert kept[0]["t0"] == 2.0 and kept[0]["narrated_span"] == [1.0, 3.0]
    assert kept[0]["b0"] == g.eighth_index(2.0)
    assert "narrated_span" not in kept[1]
    assert dropped == [["unknown 'tilted'", 6.0, 6.1]]  # 0.1 s left of the unknown phrase: too short


def test_review_state_and_compose():
    win_full, win_38 = (0.0, 100.0), (20.0, 40.0)
    narr = [seg(1, 10, 22), seg(2, 22, 26, "viradinha"), seg(3, 26, 30, None), seg(4, 30, 36, "sarrada"),
            seg(5, 36, 45, "contrapeso")]
    full_written = [dict(s) for s in narr]
    k38, _ = clip(narr, win_38)
    written = {"full": snapshot(full_written), "38": snapshot(k38)}
    # the user, in the master (full): confirms Viradinha, resolves the unknown as Giro Inclinado (edited), deletes
    # Sarrada; in larissa_kadu_38: confirms the window-clipped first item
    cur_full = [narr[0], dict(narr[1], provenance="human"), dict(narr[2], move="bent-spin", t1=29.0), narr[4]]
    cur_38 = [dict(k38[0], provenance="human"), *k38[1:]]
    carried, deleted, notes = review_state({"full": cur_full, "38": cur_38}, written, ["full", "38"],
                                           {"full": win_full, "38": win_38})
    assert [c["id"] for c in carried] == ["nar-2", "nar-3", "nar-1"]
    assert deleted == ["nar-4"]
    assert any("edited nar-3" in n for n in notes) and any("deleted" in n for n in notes)
    # the window-clipped item confirmed in _38 gets its narrated extent back for the wider take
    c1 = next(c for c in carried if c["id"] == "nar-1")
    assert (c1["t0"], c1["t1"]) == (10, 22) and "narrated_span" not in c1

    out, _ = compose(narr, carried, deleted, win_full)
    by = {s["id"]: s for s in out}
    assert "nar-4" not in by                                            # deleted stays deleted
    assert by["nar-3"]["move"] == "bent-spin" and by["nar-3"]["t1"] == 29.0
    assert by["nar-1"]["provenance"] == "human" and by["nar-2"]["provenance"] == "human"
    assert by["nar-5"]["provenance"] == "narration"
    o38, _ = compose(narr, carried, deleted, win_38)
    assert [s["id"] for s in o38] == ["nar-1", "nar-2", "nar-3", "nar-5"]
    assert o38[0]["t0"] == 20.0 and o38[0]["narrated_span"] == [10.0, 22.0] and o38[0]["provenance"] == "human"
    assert o38[-1]["t1"] == 40.0
    # nothing done since the last propagation: nothing carried, nothing deleted
    carried, deleted, _ = review_state({"full": full_written}, {"full": snapshot(full_written)}, ["full"],
                                       {"full": win_full})
    assert carried == [] and deleted == []
    # no snapshot (a file not written by propagate): only confirmed items count
    carried, deleted, _ = review_state({"full": cur_full}, {"full": None}, ["full"], {"full": win_full})
    assert [c["id"] for c in carried] == ["nar-2"] and deleted == []


def test_carried_item_cuts_the_narration():
    narr = [seg(1, 0, 10), seg(2, 10, 20, "viradinha")]
    human = dict(seg(9, 8, 12, "sarrada", prov="human"), id="hum-1")
    out, _ = compose(narr, [human], [], (0, 100))
    assert [(s["id"], s["t0"], s["t1"]) for s in out] == [("nar-1", 0, 8), ("hum-1", 8, 12), ("nar-2", 12, 20)]
    assert unclip({"t0": 5, "t1": 9}, (5, 9)) == {"t0": 5, "t1": 9}


NID = "20261007-152143"
NDIR = PACKAGE_ROOT / "work" / "larissa_kadu" / "moves" / "narration" / NID


@pytest.mark.skipif(not (NDIR / "transcript.json").exists() or not ZOUK1_PATH.exists(),
                    reason="the Larissa/Kadu narration work files are not here")
def test_real_narration_dry_run():
    from dancecap.moves.annotate.propagate import propagate
    from dancecap.moves.graph import load_graph
    from dancecap.moves.labels import Labels, validate_labels
    cfg = NDIR / "propagate.json"
    targets = (json.loads(cfg.read_text(encoding="utf-8")).get("targets") if cfg.exists() else
               ["larissa_kadu", "larissa_kadu_38", "larissa_kadu_full"])
    out = propagate("larissa_kadu", NID, targets=targets, beats_from={"larissa_kadu_full": "larissa_kadu_38"},
                    master="larissa_kadu_full", write=False, log=lambda *a: None)
    assert out["written"] == []
    g = load_graph()
    for take, st in out["takes"].items():
        if take != "larissa_kadu":
            assert st["vs_source"]["max_dt0_ms"] < 50, (take, st)  # the same narration through another clock
        doc = out["docs"][take]
        assert validate_labels(Labels.from_dict(doc), g) == []
        assert doc["narration"]["master"] == "larissa_kadu_full"
        ids = [s["id"] for s in doc["segments"] + doc["unresolved"]]
        assert len(ids) == len(set(ids))
    full = out["docs"]["larissa_kadu_full"]
    # every take's items are the master's items (same ids) clipped to its window
    master_ids = {s["id"] for s in full["segments"] + full["unresolved"]}
    for take in out["takes"]:
        d = out["docs"][take]
        assert {s["id"] for s in d["segments"] + d["unresolved"]} <= master_ids
