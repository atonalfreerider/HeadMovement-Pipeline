"""dancecap.moves.annotate + dancecap.moves.narration: the narration annotation tool.

    ..\\.venv-nopo4d\\Scripts\\python -m pytest tests/test_moves_annotate.py -q

Covers: HTTP Range responses, the labels save / validate round trip through the API, recording -> video -> label
time mapping with a pause, a speed change and a seek, reaction-lag compensation and beat snapping, offline phrase ->
move mapping (+ a stubbed TypeSafe judgment), the merge policy, additive additions.toml edits, and an end-to-end
smoke test with speech synthesised offline by Windows SAPI and recognised by the SAPI engine (skipped elsewhere).
Nothing here touches the real taxonomy/, work/ or the network.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import time
import tomllib
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dancecap.moves import ADDITIONS_PATH, GRAPH_PATH, ZOUK1_PATH  # noqa: E402
from dancecap.moves import graph as G  # noqa: E402
from dancecap.moves.annotate import asr  # noqa: E402
from dancecap.moves.annotate.pipeline import (NarrationSettings, TimelineContext, build_segments,  # noqa: E402
                                              merge_segments, overlaps, process)
from dancecap.moves.annotate.server import create_app, parse_range  # noqa: E402
from dancecap.moves.annotate.session import Camera, Session  # noqa: E402
from dancecap.moves.annotate.taxonomy_edit import (TaxonomyEditError, append_alias,  # noqa: E402
                                                   append_proposed_move)
from dancecap.moves.annotate.timemap import Grid, TimeMap, snap_start  # noqa: E402
from dancecap.moves.narration import Lexicon, map_utterances, sound_key  # noqa: E402

pytestmark = pytest.mark.skipif(not ZOUK1_PATH.exists(), reason="Zouk1.json not available")
NOFILE = Path("__no_such_file__")
HERE = Path(__file__).resolve().parent
SAPI_SYNTH = HERE.parent / "dancecap" / "moves" / "annotate" / "sapi_synth.ps1"
BPM = 76.4
EIGHTH = 60.0 / BPM / 2


def _sha(p: Path) -> str | None:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


# ---------------------------------------------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def taxo(tmp_path_factory):
    """A private taxonomy: the additions template (Corredor, veraginha, Counterbalance Pivot) + the graph from it."""
    d = tmp_path_factory.mktemp("taxonomy")
    add = d / "additions.toml"
    G.ensure_additions(add)
    g = G.build_graph(ZOUK1_PATH, add, NOFILE)
    gp = G.write_graph(g, d / "zouk_graph.json")
    return SimpleNamespace(dir=d, additions=add, graph_path=gp, graph=G.Graph(g))


@pytest.fixture(scope="module")
def lex(taxo):
    return Lexicon.from_graph(taxo.graph, vocabulary=None)


def zouk_grid(t0: float = 10.0, n: int = 200) -> list:
    pattern = [1, 3, 3, 2, 1, 3, 2, 3]
    return [[round(t0 + i * EIGHTH, 6), pattern[i % 8]] for i in range(n)]


@pytest.fixture()
def sess(tmp_path, taxo):
    """A lone-video session in tmp (fake mp4 bytes, a beat grid, the private taxonomy copied so edits stay local)."""
    v = tmp_path / "clip.mp4"
    v.write_bytes(np.random.default_rng(0).integers(0, 256, 3_000_000, dtype=np.uint8).tobytes())
    beats = tmp_path / "beats.json"
    beats.write_text(json.dumps(zouk_grid()))
    tdir = tmp_path / "taxonomy"
    shutil.copytree(taxo.dir, tdir)
    s = Session("clip", "video", [Camera("video", v, 1.0, 0.0, {})], video=v, work_root=tmp_path / "work",
                graph_path=tdir / "zouk_graph.json", additions_path=tdir / "additions.toml",
                vocabulary_path=NOFILE, zouk1_path=ZOUK1_PATH, beats=beats, annotator="tester",
                settings={"engine": "sapi"})
    return s


@pytest.fixture()
def client(sess):
    app = create_app(sess, check_asr=False)
    app.testing = True
    return app.test_client()


# ---------------------------------------------------------------------------------------------------------------
# HTTP Range
# ---------------------------------------------------------------------------------------------------------------

def test_parse_range():
    assert parse_range(None, 100) is None
    assert parse_range("bytes=0-9", 100) == (0, 9)
    assert parse_range("bytes=90-", 100) == (90, 99)
    assert parse_range("bytes=-10", 100) == (90, 99)
    assert parse_range("bytes=-500", 100) == (0, 99)
    assert parse_range("bytes=50-5000", 100) == (50, 99)
    assert parse_range("bytes=100-", 100) == "unsatisfiable"
    assert parse_range("bytes=-0", 100) == "unsatisfiable"
    assert parse_range("bytes=0-1,5-6", 100) is None  # multi-range: ignored -> full 200
    assert parse_range("items=0-1", 100) is None
    assert parse_range("bytes=9-3", 100) is None


def test_video_range_requests(client, sess):
    data = sess.cameras["video"].path.read_bytes()
    n = len(data)
    r = client.get("/video/video")
    assert r.status_code == 200 and r.headers["Accept-Ranges"] == "bytes"
    assert int(r.headers["Content-Length"]) == n and r.data == data
    assert r.headers["Content-Type"] == "video/mp4"
    r = client.get("/video/video", headers={"Range": "bytes=0-99"})
    assert r.status_code == 206 and r.headers["Content-Range"] == f"bytes 0-99/{n}" and r.data == data[:100]
    r = client.get("/video/video", headers={"Range": f"bytes={n - 1000}-"})
    assert r.status_code == 206 and r.data == data[-1000:] and r.headers["Content-Length"] == "1000"
    r = client.get("/video/video", headers={"Range": "bytes=-50"})
    assert r.status_code == 206 and r.data == data[-50:]
    r = client.get("/video/video", headers={"Range": "bytes=1048570-2097160"})  # spans the 1 MiB read chunks
    assert r.status_code == 206 and r.data == data[1048570:2097161]
    r = client.get("/video/video", headers={"Range": f"bytes={n}-"})
    assert r.status_code == 416 and r.headers["Content-Range"] == f"bytes */{n}"
    r = client.get("/video/video", headers={"Range": "bytes=0-1,4-5"})
    assert r.status_code == 200 and len(r.data) == n
    r = client.head("/video/video", headers={"Range": "bytes=10-19"})
    assert r.status_code == 206 and r.headers["Content-Length"] == "10" and r.data == b""
    assert client.get("/video/nope").status_code == 404


def test_host_guard(sess):
    app = create_app(sess, allowed_hosts={"127.0.0.1:9"}, check_asr=False)
    c = app.test_client()
    assert c.get("/api/session", headers={"Host": "evil.example:80"}).status_code == 403
    assert c.get("/video/video", headers={"Host": "127.0.0.1:9", "Range": "bytes=0-0"}).status_code == 206


# ---------------------------------------------------------------------------------------------------------------
# labels round trip
# ---------------------------------------------------------------------------------------------------------------

def _seg(i, t0, t1, move, prov="human", **kw):
    return {"id": f"s{i}", "t0": t0, "t1": t1, "move": move, "node": None, "tier": "couple", "provenance": prov,
            "confidence": kw.pop("confidence", None), "annotator": "tester", "transcript": None, "notes": None,
            "created": "2026-10-06T00:00:00+00:00", **kw}


def test_labels_save_validate_roundtrip(client, sess):
    # a placeholder file first: not loaded as work, replaced (and backed up) on save
    sess.labels_path.parent.mkdir(parents=True, exist_ok=True)
    sess.labels_path.write_text(json.dumps({"version": 1, "take": "clip", "time_base": "video", "video": "clip.mp4",
                                            "placeholder": True, "note": "PLACEHOLDER",
                                            "segments": [_seg(0, 1, 2, "dip", "placeholder")]}))
    info = client.get("/api/session").get_json()
    assert info["labels"]["placeholder"]["count"] == 1 and info["labels"]["segments"] == []
    assert info["time_base"] == "video" and info["beats"][0][2] == 0
    segs = [_seg(1, 10.0, 11.5708, "lateral", "narration", confidence=0.8, anchor_t=10.6, anchor_word="corredor",
                 lag_s=0.4, b0=0, b1=3, asr={"engine": "sapi", "model": "x", "word_prob": 0.9}),
            _seg(2, 11.5708, 13.1416, "viradinha"),
            {**_seg(3, 13.1416, 14.0, None, "narration"), "unresolved": "cicada", "candidates": []},
            {**_seg(4, 14.0, 15.0, "soltinho"), "_teach": "ui only"}]
    r = client.post("/api/labels", json={"segments": segs})
    assert r.status_code == 200, r.get_json()
    out = r.get_json()
    assert out["segments"] == 3 and out["unresolved"] == 1 and out["backup"]
    doc = json.loads(sess.labels_path.read_text(encoding="utf-8"))
    assert doc["version"] == 1 and doc["take"] == "clip" and doc["time_base"] == "video"
    assert doc["video"].endswith("clip.mp4") and "placeholder" not in doc and "note" not in doc
    assert [s["move"] for s in doc["segments"]] == ["lateral", "viradinha", "soltinho"]
    first = doc["segments"][0]
    assert first["anchor_word"] == "corredor" and first["asr"]["engine"] == "sapi" and first["b1"] == 3
    assert all("_teach" not in s for s in doc["segments"])
    assert doc["unresolved"][0]["unresolved"] == "cicada" and "move" not in doc["unresolved"][0]
    from dancecap.moves.labels import load_labels, validate_labels
    assert validate_labels(load_labels(sess.labels_path), sess.graph) == []
    back = client.get("/api/session").get_json()["labels"]["segments"]
    assert [s["move"] for s in back] == ["lateral", "viradinha", None, "soltinho"]
    assert back[2]["unresolved"] == "cicada"
    r2 = client.post("/api/labels", json={"segments": back})  # stable second save
    assert r2.status_code == 200
    doc2 = json.loads(sess.labels_path.read_text(encoding="utf-8"))
    assert doc2["segments"] == doc["segments"] and doc2["unresolved"] == doc["unresolved"]
    # invalid: overlap, unknown move -> 400 with problems, file untouched
    bad = [_seg(1, 10, 12, "lateral"), _seg(2, 11, 13, "viradinha")]
    r = client.post("/api/labels", json={"segments": bad})
    assert r.status_code == 400 and any("overlap" in p for p in r.get_json()["problems"])
    r = client.post("/api/labels", json={"segments": [_seg(1, 10, 12, "no-such-move")]})
    assert r.status_code == 400 and any("unknown move" in p for p in r.get_json()["problems"])
    assert json.loads(sess.labels_path.read_text(encoding="utf-8")) == doc2
    assert client.post("/api/validate", json={"segments": back}).get_json()["problems"] == []


# ---------------------------------------------------------------------------------------------------------------
# clocks
# ---------------------------------------------------------------------------------------------------------------

def player_log(jitter: float = 0.0) -> list[dict]:
    """1x from v=10 for 2 s, paused 1.5 s, 0.5x for 4 s, a seek to 50, 1x for 2 s (recording clock a)."""
    rng = np.random.default_rng(1)
    e = [{"a": 0.0, "v": 10.0, "cam": "c", "rate": 1.0, "paused": False, "ev": "start"}]
    for a in np.arange(0.1, 2.0, 0.1):
        e.append({"a": a, "v": 10 + a + jitter * rng.normal(), "cam": "c", "rate": 1.0, "paused": False, "ev": "tick"})
    e.append({"a": 2.0, "v": 12.0, "cam": "c", "rate": 1.0, "paused": True, "ev": "pause"})
    for a in np.arange(2.1, 3.5, 0.1):
        e.append({"a": a, "v": 12.0, "cam": "c", "rate": 1.0, "paused": True, "ev": "tick"})
    e.append({"a": 3.5, "v": 12.0, "cam": "c", "rate": 0.5, "paused": True, "ev": "ratechange"})
    e.append({"a": 3.5, "v": 12.0, "cam": "c", "rate": 0.5, "paused": False, "ev": "play"})
    for a in np.arange(3.6, 7.5, 0.1):
        e.append({"a": a, "v": 12 + 0.5 * (a - 3.5) + jitter * rng.normal(), "cam": "c", "rate": 0.5,
                  "paused": False, "ev": "tick"})
    e.append({"a": 7.5, "v": 50.0, "cam": "c", "rate": 1.0, "paused": False, "ev": "seeking"})
    for a in np.arange(7.6, 9.55, 0.1):
        e.append({"a": a, "v": 50 + (a - 7.5), "cam": "c", "rate": 1.0, "paused": False, "ev": "tick"})
    return e


@pytest.mark.parametrize("jitter", [0.0, 0.01])
def test_timemap_pause_speed_seek(jitter):
    tm = TimeMap(player_log(jitter))
    tol = 1e-6 if jitter == 0 else 0.04
    assert tm.label_time(1.0) == pytest.approx(11.0, abs=tol)
    assert tm.label_time(2.7) == pytest.approx(12.0, abs=tol)          # paused: held
    assert tm.label_time(5.5) == pytest.approx(13.0, abs=tol)          # 0.5x after the pause
    assert tm.label_time(8.5) == pytest.approx(51.0, abs=tol)          # after the seek
    assert tm.label_time(-0.5) == pytest.approx(9.5, abs=tol)          # before the first entry
    assert tm.label_time(10.0) == pytest.approx(52.5, abs=tol)         # after the last entry
    # reaction lag: 0.4 s wall clock = 0.2 s of video at 0.5x, nothing while paused, never across a seek
    assert tm.event_time(5.5, 0.4)[0] == pytest.approx(12.8, abs=tol)
    assert tm.event_time(3.7, 0.4)[0] == pytest.approx(12.0, abs=tol)
    assert tm.event_time(1.0, 0.4)[0] == pytest.approx(10.6, abs=tol)
    t, a_used = tm.event_time(7.7, 0.4)
    assert a_used == pytest.approx(7.5) and t == pytest.approx(50.0, abs=tol)
    sp = tm.spans()
    assert len(sp) == 2 and sp[0][0] == pytest.approx(10.0, abs=tol) and sp[0][1] == pytest.approx(14.0, abs=0.06)
    assert sp[1][0] == pytest.approx(50.0) and sp[1][1] == pytest.approx(52.0, abs=0.06)
    # an affine camera clock (reference = slope * v + offset) is applied per entry
    tm2 = TimeMap(player_log(), to_label=lambda cam, v: 1.0002 * v + 1.58, slope=lambda cam: 1.0002)
    assert tm2.label_time(5.5) == pytest.approx(1.0002 * 13.0 + 1.58, abs=1e-6)


def test_latency_and_snapping():
    grid = Grid(zouk_grid())
    assert grid.phase == 0 and grid.unit("half") == pytest.approx(4 * EIGHTH, abs=1e-5)
    log = [{"a": 0.0, "v": 10.0, "cam": "c", "rate": 1.0, "paused": False, "ev": "start"},
           {"a": 20.0, "v": 30.0, "cam": "c", "rate": 1.0, "paused": False, "ev": "stop"}]
    tm = TimeMap(log)
    t_ev, _ = tm.event_time(3.0, 0.4)
    assert t_ev == pytest.approx(12.6)
    half = [10 + 4 * EIGHTH * k for k in range(6)]
    assert snap_start(t_ev, grid, "half") == (pytest.approx(half[1], abs=1e-5), "grid")      # at or before
    assert snap_start(t_ev, grid, "eighth")[0] == pytest.approx(10 + 6 * EIGHTH, abs=1e-5)
    assert snap_start(t_ev, grid, "measure")[0] == pytest.approx(10.0, abs=1e-5) or \
        snap_start(t_ev, grid, "measure")[1] in ("eighth", "none")
    assert snap_start(t_ev, grid, "none") == (pytest.approx(12.6), "none")
    assert snap_start(13.10, grid, "half")[0] == pytest.approx(half[2], abs=1e-5)  # just before a boundary: after
    assert snap_start(t_ev, grid, "half", boundaries=[12.5]) == (pytest.approx(12.5), "proposal")
    tap = tm.label_time(2.8 - 0.15)  # an 'M' tap, minus its reaction time
    assert snap_start(t_ev, grid, "half", taps=[tap], boundaries=[12.5]) == (pytest.approx(10 + 7 * EIGHTH, abs=1e-5),
                                                                             "tap")
    # segments: lag + snap + ends at the next start / the end of the watched stretch
    ctx = TimelineContext(grid=grid)
    events = [{"kind": "move", "a0": 3.0, "a1": 3.4, "phrase": "corredor", "move": "lateral", "confidence": 0.9,
               "status": "matched", "method": "alias", "score": 100.0},
              {"kind": "move", "a0": 6.2, "a1": 6.7, "phrase": "viradinha", "move": "viradinha", "confidence": 0.95,
               "status": "matched", "method": "exact", "score": 100.0},
              {"kind": "unknown", "a0": 12.0, "a1": 12.3, "phrase": "cicada", "move": None, "confidence": None}]
    segs = build_segments(events, tm, ctx, NarrationSettings(lag=0.4, snap="half"), {"engine": "test"}, "n1")
    assert [s["move"] for s in segs] == ["lateral", "viradinha", None]
    assert segs[0]["t0"] == pytest.approx(half[1], abs=1e-5) and segs[0]["t1"] == segs[1]["t0"]
    assert segs[1]["t0"] == pytest.approx(half[3], abs=1e-5)   # 16.2 - 0.4 = 15.8 -> half[3] = 14.71
    assert segs[2]["t1"] == pytest.approx(30.0) and segs[2]["unresolved"] == "cicada"
    assert segs[0]["confidence"] == pytest.approx(0.9 * 0.85) and segs[2]["confidence"] is None
    assert segs[0]["anchor_t"] == pytest.approx(13.0) and segs[0]["lag_s"] == 0.4 and segs[0]["b0"] == 4
    assert all(s["provenance"] == "narration" for s in segs) and not overlaps(segs)
    # at 0.5x the same 0.4 s reaction is 0.2 s of video
    tm_half = TimeMap([{"a": 0.0, "v": 10.0, "cam": "c", "rate": 0.5, "paused": False, "ev": "start"}])
    assert tm_half.event_time(3.0, 0.4)[0] == pytest.approx(10 + 0.5 * 2.6)


# ---------------------------------------------------------------------------------------------------------------
# phrase -> move
# ---------------------------------------------------------------------------------------------------------------

def utt(text: str, a0: float, prob: float = 0.9, step: float = 0.35) -> dict:
    ws = [{"word": w, "start": round(a0 + i * step, 3), "end": round(a0 + i * step + 0.3, 3), "prob": prob}
          for i, w in enumerate(text.split())]
    return {"start": ws[0]["start"], "end": ws[-1]["end"], "text": text, "words": ws}


class NoClient:
    def system_one(self, **kw):
        raise AssertionError("TypeSafe must not be called on the offline path")


def test_phrase_to_move_offline(lex):
    assert sound_key("veraginha") == sound_key("viradinha") == sound_key("vira dinha")
    seq = ["now the lateral", "veraginha", "vira dinha", "corridor", "um okay so", "the soltinho", "scratch that",
           "turns", "again", "they look really happy together tonight", "cicada", "lateral from behind"]
    events = map_utterances([utt(t, 2.0 * i) for i, t in enumerate(seq)], lex, judge=False, client=NoClient())
    kinds = [(e["kind"], e.get("move"), e.get("method")) for e in events]
    assert kinds[0] == ("move", "lateral", "exact")
    assert kinds[1][:2] == ("move", "viradinha")                               # alias from additions.toml
    assert kinds[2][:2] == ("move", "viradinha") and kinds[2][2] == "phonetic"
    assert kinds[3][:2] == ("move", "lateral")                                  # 'corridor' alias
    assert ("undone", "soltinho") == kinds[4][:2]                               # removed by "scratch that"
    assert kinds[5][0] == "control"
    assert kinds[6][:2] == ("move", "turn") and kinds[6][2] == "stem"
    assert kinds[7][:3] == ("move", "turn", "repeat")
    assert kinds[8][0] == "ignored"
    assert kinds[9][:2] == ("unknown", None) and events[9]["phrase"] == "cicada"
    assert kinds[10][:2] == ("move", "lateral-from-behind")                     # longest exact phrase wins
    assert events[0]["a0"] == pytest.approx(0.35 * 2) and events[0]["phrase"] == "lateral"
    assert 0.8 < events[0]["confidence"] <= 0.9
    # an unresolved research alias surfaces as unknown with its candidates, never as a move
    lex2 = Lexicon(list(lex.moves.values()), unresolved=[{"alias": "cicada", "candidates": ["sacada", "chicote"]}])
    ev = map_utterances([utt("cicada", 0.0)], lex2)[0]
    assert ev["kind"] == "unknown" and ev["unresolved_alias"] == "cicada"


class FakeTypeSafe:
    def __init__(self, noul: float, choice: str, conf: float):
        self.noul, self.choice, self.conf, self.calls = noul, choice, conf, []

    def system_one(self, state, questions):
        self.calls.append((state, questions))
        assert "names_a_move" in questions and "which_move" in questions
        assert "none_of_these" in questions["which_move"].criteria
        return SimpleNamespace(nouls={"names_a_move": SimpleNamespace(noul=self.noul)},
                               choices={"which_move": SimpleNamespace(choice=self.choice, confidence=self.conf)})


def test_phrase_to_move_typesafe_stub(lex):
    pytest.importorskip("typesafe_sdk")
    ok = FakeTypeSafe(0.92, "viradinha", 0.81)
    ev = map_utterances([utt("very genia", 0.0)], lex, judge=True, client=ok)[0]
    assert ok.calls and ok.calls[0][0]["phrase"] == "very genia"
    assert ev["kind"] == "move" and ev["move"] == "viradinha" and ev["method"] == "judged"
    assert ev["confidence"] == pytest.approx(0.81 * 0.9, abs=1e-3)
    refuse = FakeTypeSafe(0.35, "viradinha", 0.9)  # the weakest judgment < 0.4 -> refused
    ev = map_utterances([utt("very genia", 0.0)], lex, judge=True, client=refuse)[0]
    assert ev["kind"] == "unknown" and ev["judgment"]["refused"] is True
    none = FakeTypeSafe(0.9, "none_of_these", 0.9)
    assert map_utterances([utt("very genia", 0.0)], lex, judge=True, client=none)[0]["kind"] == "unknown"
    # without the judgment the ambiguous phrase stays unknown with the candidates for the user
    ev = map_utterances([utt("very genia", 0.0)], lex, judge=False, client=NoClient())[0]
    assert ev["kind"] == "unknown" and ev["candidates"][0]["move"] == "viradinha"


# ---------------------------------------------------------------------------------------------------------------
# merge + additions
# ---------------------------------------------------------------------------------------------------------------

def test_merge_policy():
    existing = [_seg(0, 0, 5, "dip", "placeholder"), _seg(1, 10, 14, "hug", "narration"),
                _seg(2, 20, 22, "dip", "human"), _seg(3, 30, 40, "pop", "auto")]
    new = [_seg(10, 9, 12, "lateral", "narration"), _seg(11, 12, 25, "viradinha", "narration")]
    out = merge_segments(existing, new)
    assert not overlaps(out)
    assert all(s["provenance"] != "placeholder" for s in out)
    spans = [(s["move"], s["t0"], s["t1"], s["provenance"]) for s in out]
    assert ("dip", 20, 22, "human") in spans                             # human kept as is
    assert ("viradinha", 12, 20, "narration") in spans and ("viradinha", 22, 25, "narration") in spans
    assert ("lateral", 9, 12, "narration") in spans
    assert not any(m == "hug" for m, *_ in spans)                       # old narration fully replaced
    assert ("pop", 30, 40, "auto") in spans                             # outside the new narration: kept


def test_additions_are_additive(tmp_path, taxo):
    add = tmp_path / "additions.toml"
    shutil.copyfile(taxo.additions, add)
    before = add.read_text(encoding="utf-8")
    r = append_alias("viradinha", "during hot", taxo.graph, additions=add, zouk1=ZOUK1_PATH, vocabulary=NOFILE)
    assert r["changed"] and G.Graph(r["graph"]).resolve_move("during hot") == "viradinha"
    text = add.read_text(encoding="utf-8")
    assert text.startswith(before.rstrip("\n")) and tomllib.loads(text)["moves"][-1]["aliases"] == ["during hot"]
    g2 = G.Graph(r["graph"])
    assert append_alias("viradinha", "During Hot", g2, additions=add, zouk1=ZOUK1_PATH,
                        vocabulary=NOFILE)["changed"] is False
    with pytest.raises(TaxonomyEditError):
        append_alias("lateral", "veraginha", g2, additions=add, zouk1=ZOUK1_PATH, vocabulary=NOFILE)
    r = append_proposed_move({"name": "Cicada Spin", "family": "turn", "energy": 0.7, "near": ["pirouette"],
                              "aliases": ["cicada spin turn"], "description": "test\nmove"}, g2, additions=add,
                             zouk1=ZOUK1_PATH, vocabulary=NOFILE)
    g3 = G.Graph(r["graph"])
    assert r["slug"] == "cicada-spin" and g3.moves["cicada-spin"]["status"] == "proposed"
    assert g3.resolve_move("cicada spin turn") == "cicada-spin"
    assert g3.nodes["proposed-cicada-spin"]["family"] == "turn"
    with pytest.raises(TaxonomyEditError):
        append_proposed_move({"name": "Lateral"}, g3, additions=add, zouk1=ZOUK1_PATH, vocabulary=NOFILE)
    assert add.read_text(encoding="utf-8").startswith(before.rstrip("\n"))


def test_api_alias_and_propose_stay_local(client, sess):
    real = (_sha(ADDITIONS_PATH), _sha(GRAPH_PATH))
    r = client.post("/api/alias", json={"move": "viradinha", "alias": "during hot"})
    assert r.status_code == 200 and r.get_json()["ok"], r.get_json()
    assert sess.graph.resolve_move("during hot") == "viradinha"
    assert G.Graph.load(sess.graph_path).resolve_move("during hot") == "viradinha"
    r = client.post("/api/propose", json={"name": "Test Whirl", "family": "turn", "tone": "yellow", "energy": 0.8})
    body = r.get_json()
    assert r.status_code == 200 and body["slug"] == "test-whirl"
    assert any(m["slug"] == "test-whirl" and m["status"] == "proposed" for m in body["moves"])
    assert client.post("/api/alias", json={"move": "lateral", "alias": "veraginha"}).status_code == 400
    assert (_sha(ADDITIONS_PATH), _sha(GRAPH_PATH)) == real  # the real taxonomy is untouched


def test_export_endpoint_wiring(sess, tmp_path, monkeypatch):
    from dancecap.moves.annotate import server as srv
    c = create_app(sess, check_asr=False).test_client()
    assert c.post("/api/export").status_code == 400                       # a lone video has no reference time
    sess.mode, sess.time_base = "take", "reference"
    c = create_app(sess, check_asr=False).test_client()
    assert c.post("/api/export").status_code == 409                       # nothing saved yet
    assert c.post("/api/labels", json={"segments": [_seg(1, 10, 12, "lateral")]}).status_code == 200
    calls = []
    monkeypatch.setattr(srv.subprocess, "run", lambda cmd, **kw: calls.append(cmd) or
                        SimpleNamespace(returncode=0, stdout="[moves] exported", stderr=""))
    jid = c.post("/api/export").get_json()["job"]
    for _ in range(100):
        j = c.get(f"/api/jobs/{jid}").get_json()
        if j["state"] != "running":
            break
        time.sleep(0.02)
    assert j["state"] == "done" and j["result"]["returncode"] == 0
    assert calls and calls[0][-3:] == ["dancecap.moves", "run", "clip"]


# ---------------------------------------------------------------------------------------------------------------
# narration through the API without a recogniser: transcript import
# ---------------------------------------------------------------------------------------------------------------

def _wav(path: Path, seconds: float = 1.0, sr: int = 16000) -> Path:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(sr)
        w.writeframes(b"\0\0" * int(seconds * sr))
    return path


def test_narration_api_with_imported_transcript(client, sess, tmp_path):
    log = {"clock": "audioworklet", "entries": [{"a": 0.0, "v": 10.0, "cam": "video", "rate": 0.5, "paused": False,
                                                 "ev": "start"},
                                                {"a": 8.0, "v": 14.0, "cam": "video", "rate": 0.5, "paused": True,
                                                 "ev": "stop"}]}
    sess.settings["engine"] = "no-such-engine"  # the job fails fast: no recogniser involved here
    wav = _wav(tmp_path / "a.wav")
    r = client.post("/api/narration", data={"audio": (io.BytesIO(wav.read_bytes()), "audio.wav", "audio/wav"),
                                            "log": json.dumps(log), "settings": json.dumps({"lag": 0.4})},
                    content_type="multipart/form-data")
    assert r.status_code == 200
    nid, jid = r.get_json()["id"], r.get_json()["job"]
    for _ in range(100):
        j = client.get(f"/api/jobs/{jid}").get_json()
        if j["state"] != "running":
            break
        time.sleep(0.05)
    assert j["state"] == "error" and "no-such-engine" in j["error"]
    nd = sess.narration_dir / nid
    assert (nd / "audio.wav").exists() and json.loads((nd / "log.json").read_text())["entries"][0]["v"] == 10.0
    words = [{"word": "corredor", "start": 1.0, "end": 1.5, "prob": 0.95},
             {"word": "viradinha", "start": 5.0, "end": 5.6, "prob": 0.9}]
    r = client.post(f"/api/narration/{nid}/transcript", json=words)
    res = r.get_json()
    assert r.status_code == 200 and [s["move"] for s in res["segments"]] == ["lateral", "viradinha"]
    assert res["segments"][0]["anchor_t"] == pytest.approx(10.5)      # 0.5x from 10.0: word at 1.0 s -> 10.5
    assert res["segments"][-1]["t1"] == pytest.approx(14.0)           # the watched stretch ends at 14
    assert client.get(f"/api/narration/{nid}").get_json()["id"] == nid
    r = client.post(f"/api/narration/{nid}/remap", json={"settings": {"lag": 0.0, "snap": "none"}})
    assert r.get_json()["segments"][0]["t0"] == pytest.approx(10.5)
    merged = client.post("/api/merge", json={"current": [], "new": res["segments"]}).get_json()["segments"]
    assert client.post("/api/labels", json={"segments": merged}).status_code == 200
    assert client.get(f"/api/narration/{nid}/audio", headers={"Range": "bytes=0-43"}).status_code == 206


# ---------------------------------------------------------------------------------------------------------------
# end to end: Windows SAPI speech -> SAPI recognition -> moves
# ---------------------------------------------------------------------------------------------------------------

def _sapi_ok() -> bool:
    return os.name == "nt" and shutil.which("powershell") is not None and SAPI_SYNTH.exists() and \
        bool(asr.engines().get("sapi", {}).get("available"))


def onsets(wav: Path, thr: float = 0.01) -> list[float]:
    with wave.open(str(wav), "rb") as w:
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(float) / 32768
        sr = w.getframerate()
    env = np.convolve(np.abs(x), np.ones(int(0.05 * sr)) / int(0.05 * sr), "same")
    on = env > thr
    starts = list(np.where(np.diff(on.astype(int)) == 1)[0] / sr)
    merged = []
    for s in starts:  # one onset per word (gaps inside a word are < 0.4 s)
        if not merged or s - merged[-1] > 1.0:
            merged.append(s)
    return merged


@pytest.mark.skipif(not _sapi_ok(), reason="Windows SAPI (System.Speech, en-US) not available")
def test_sapi_speech_to_moves_end_to_end(tmp_path, sess):
    # SAPI's English voice cannot say Portuguese, so viradinha is written as it is pronounced: vee-ra-JEEN-ya
    script = tmp_path / "script.txt"
    script.write_text("pause 1.0\nlateral\npause 3\ncorredor\npause 3\nvira jeenya\npause 1.5\n", encoding="utf-8")
    nd = sess.new_narration_dir()
    wav = nd / "audio.wav"
    r = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(SAPI_SYNTH),
                        "-Out", str(wav), "-Script", str(script)], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and wav.exists(), r.stderr
    ons = onsets(wav)
    assert len(ons) == 3, ons
    # the narrator watched from v = 20 s at half speed
    (nd / "log.json").write_text(json.dumps({"entries": [{"a": 0.0, "v": 20.0, "cam": "video", "rate": 0.5,
                                                         "paused": False, "ev": "start"}]}))
    res = process(nd, sess.context(), sess.lexicon, NarrationSettings(engine="sapi", lag=0.4, snap="none"))
    moves = [e for e in res["events"] if e["kind"] == "move"]
    assert [e["move"] for e in moves] == ["lateral", "lateral", "viradinha"], res["summary"]
    for e, on in zip(moves, ons):
        assert e["a0"] == pytest.approx(on, abs=0.35)                        # word timing from the recogniser
        assert e["anchor_t"] == pytest.approx(20 + 0.5 * on, abs=0.2)       # recording -> video time at 0.5x
        assert e["t0"] == pytest.approx(20 + 0.5 * (e["a0"] - 0.4), abs=1e-3)  # lag 0.4 s wall = 0.2 s video
    segs = res["segments"]
    assert [s["move"] for s in segs] == ["lateral", "lateral", "viradinha"] and not overlaps(segs)
    assert segs[0]["asr"]["engine"] == "sapi" and segs[1]["anchor_word"] in ("corredor", "corridor")  # both alias
    assert (nd / "transcript.json").exists() and (nd / "result.json").exists()
    # with the beat grid the starts land on half-measure points at or just before the event
    res2 = process(nd, sess.context(), sess.lexicon, NarrationSettings(engine="sapi", lag=0.4, snap="half"))
    half = set(round(10 + 4 * EIGHTH * k, 4) for k in range(60))
    for s in res2["segments"]:
        assert round(s["t0"], 4) in half and s["t0"] <= s["anchor_t"] + 0.2
