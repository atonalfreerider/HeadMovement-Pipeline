"""dancecap.farm.state: atomic JSON writes, the lock file, transactions from several threads."""
import json
import os
import threading
import time

import pytest

from dancecap.farm import state as S


def test_atomic_write_leaves_no_temp_files_and_replaces_whole_content(tmp_path):
    p = tmp_path / "d" / "x.json"
    S.atomic_write_json(p, {"a": 1})
    S.atomic_write_json(p, {"b": [1, 2, 3]})
    assert json.loads(p.read_text()) == {"b": [1, 2, 3]}
    assert [f.name for f in p.parent.iterdir()] == ["x.json"]
    import numpy as np
    S.atomic_write_json(p, {"n": np.int64(3), "f": np.float32(1.5), "arr": np.arange(3), "p": tmp_path})   # numpy / Path ok
    assert json.loads(p.read_text())["arr"] == [0, 1, 2]


def test_a_reader_never_sees_a_half_written_file(tmp_path):
    p = tmp_path / "s.json"
    S.atomic_write_json(p, {"n": 0, "pad": "x" * 200_000})
    stop, bad = threading.Event(), []

    def writer():
        i = 0
        while not stop.is_set():
            i += 1
            S.atomic_write_json(p, {"n": i, "pad": "x" * 200_000}, indent=None)

    def reader():
        while not stop.is_set():
            try:
                d = json.loads(p.read_text())
                assert len(d["pad"]) == 200_000
            except FileNotFoundError:
                bad.append("missing")
            except (ValueError, AssertionError) as e:
                bad.append(repr(e))
            except PermissionError:
                pass   # Windows: the rename in flight; read_json retries this, a bare read may not

    ts = [threading.Thread(target=writer)] + [threading.Thread(target=reader) for _ in range(3)]
    [t.start() for t in ts]
    time.sleep(1.5)
    stop.set()
    [t.join() for t in ts]
    assert bad == []


def test_transactions_from_many_threads_lose_nothing(tmp_path):
    st = S.State("d", path=tmp_path / "state.json")
    N, K = 6, 8

    def work(i):
        for _ in range(K):
            st.update(lambda s: s["spend"].__setitem__("n", s["spend"].get("n", 0) + 1))
            st.event("t", f"thread {i}")

    ts = [threading.Thread(target=work, args=(i,)) for i in range(N)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    s = st.load()
    assert s["spend"]["n"] == N * K
    assert len(s["events"]) == N * K
    assert not st.lock_path.exists()


def test_stale_lock_is_broken_but_a_live_one_times_out(tmp_path):
    lock = tmp_path / "x.lock"
    lock.write_text("{}")
    with pytest.raises(S.LockTimeout):
        S.FileLock(lock, timeout=0.3, stale=60).acquire()
    old = time.time() - 600
    os.utime(lock, (old, old))
    with S.FileLock(lock, timeout=2, stale=60):
        assert lock.exists()
    assert not lock.exists()


def test_chunk_lifecycle_events_and_validation(tmp_path):
    st = S.State("d", path=tmp_path / "state.json")
    st.set_chunk("g1", status="planned", core_k0=0, core_k1=10)
    st.set_chunk("g1", status="running", pod="p1", progress=0.3)
    assert st.chunk("g1")["status"] == "running" and st.chunk("g1")["pod"] == "p1" and "status_ts" in st.chunk("g1")
    assert [e["msg"] for e in st.load()["events"]] == ["g1: new -> planned", "g1: planned -> running"]
    with pytest.raises(ValueError):
        st.set_chunk("g1", status="bogus")
    assert st.chunk("g1")["status"] == "running"                 # the failed transaction wrote nothing
    assert st.claimed_ranges() == [(0, 10, "g1")]
    st.set_chunk("g1", status="failed")
    assert st.claimed_ranges() == []


def test_put_chunks_replaces_planned_keeps_the_rest(tmp_path):
    st = S.State("d", path=tmp_path / "state.json")
    st.set_chunk("a", status="running", k0=0, core_k0=0, core_k1=9)
    st.set_chunk("old", status="planned", k0=50, core_k0=50, core_k1=59)
    changed = st.put_chunks([{"id": "a", "k0": 0, "core_k0": 0, "core_k1": 99}, {"id": "n", "k0": 100, "core_k0": 100, "core_k1": 109}])
    s = st.load()["chunks"]
    assert changed == ["n"]                                       # a is running: the watcher's, never replaced
    assert s["a"]["core_k1"] == 9 and s["n"]["status"] == "planned" and "old" not in s
    assert list(s) == ["a", "n"]                                  # ordered by position


def test_pods_spend_world_watcher_and_event_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "EVENT_CAP", 20)
    st = S.State("d", path=tmp_path / "state.json")
    st.upsert_pod("p1", state="up", gpu="A100", price=1.39)
    st.upsert_pod("p1", state="deleted")
    st.set_spend(spent=1.0, cap=80)
    st.set_world(id="w1")
    st.set_watcher(pid=123)
    s = st.load()
    assert s["pods"]["p1"]["price"] == 1.39 and s["pods"]["p1"]["state"] == "deleted"
    assert s["spend"]["cap"] == 80 and "updated" in s["spend"] and s["world"]["id"] == "w1" and "heartbeat" in s["watcher"]
    assert st.summary()["pods_open"] == 0
    for i in range(45):
        st.event("x", str(i))
    ev = st.load()["events"]
    assert len(ev) == 20 and ev[-1]["msg"] == "44"


def test_a_missing_or_foreign_state_file_loads_as_empty(tmp_path):
    st = S.State("d", path=tmp_path / "nope" / "state.json")
    s = st.load()
    assert s["schema"] == S.SCHEMA and s["chunks"] == {} and s["dance"] == "d"
    (tmp_path / "nope").mkdir()
    (tmp_path / "nope" / "state.json").write_text('{"chunks": {"x": {"id": "x"}}}')
    s = st.load()
    assert "x" in s["chunks"] and s["pods"] == {} and s["events"] == []   # missing keys are filled in
