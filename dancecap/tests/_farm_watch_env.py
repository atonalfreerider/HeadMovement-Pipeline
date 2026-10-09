"""Shared fixtures/helpers of the watcher tests (tests/test_farm_watch*.py): a temp workspace with the farm modules re-pointed at it,
a fake volume + FakeBackend, and builders for queued jobs / recorded pods / a Watcher with test-friendly timing."""
from __future__ import annotations

import sys
import time
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import _farm_fixtures as FX  # noqa: E402
import _farm_watch_fixtures as F  # noqa: E402

from dancecap import remote  # noqa: E402
from dancecap.farm import cli, launch, volume, watch  # noqa: E402
from dancecap.farm.state import State, atomic_write_json  # noqa: E402

DANCE = "t"


class Env:
    pass


@pytest.fixture
def env(tmp_path, monkeypatch):
    e = Env()
    e.tmp = tmp_path
    e.ws = tmp_path / "ws"
    e.root = e.ws / "dancecap"
    e.root.mkdir(parents=True)
    FX.patch_root(monkeypatch, e.root)
    e.vol = tmp_path / "vol"
    e.vol.mkdir()
    F.LocalLink.cmds.clear()
    F.LocalLink.fast_snapshot = True
    F.FlakyLink.drops, F.FlakyLink.down, F.FlakyLink.calls = 0, False, 0
    e.be = F.FakeBackend(e.vol, tmp_path)
    e.cov_calls = []
    monkeypatch.setattr(cli, "refresh_coverage", lambda dance, **k: e.cov_calls.append(dance))
    # pull_file's backoff between retries: a shim module for volume only (patching time.sleep itself would hit every thread)
    monkeypatch.setattr(volume, "time", types.SimpleNamespace(time=time.time, sleep=lambda s: None))
    monkeypatch.setattr(volume, "SHA_CACHE", tmp_path / "sha_cache.json")          # not the real work/farm cache
    monkeypatch.setattr(remote, "ledger", lambda: [])                              # not the real Runpod ledger
    monkeypatch.setattr(launch, "_BACKEND", e.be)
    e.state = State(DANCE)
    e.pods = {}
    return e


def add_job(e: Env, jid: str, status: str = "queued", *, after=(), needs_gpu: bool = True, chunk: bool = True, take: str | None = None,
            **spec_kw) -> dict:
    """A planned-and-queued job in all three places: queue/<id>.json on the volume, farm/jobs/<id>.json, state.json."""
    take = take or f"t_{jid}"
    spec = F.job_spec(jid, take, after=list(after), needs_gpu=needs_gpu, kind="fit" if needs_gpu else "analyse", **spec_kw)
    F.queue_job(e.vol, spec)
    atomic_write_json(watch.jobs_dir(DANCE) / f"{jid}.json", spec)
    if chunk:
        e.state.put_chunks([{"id": jid, "k0": 0, "k1": 899, "core_k0": 0, "core_k1": 899, "take": take, "status": status,
                             "kind": "fit", "core_s": 30.0}])
    e.state.set_job(jid, status=status, kind="fit")
    return spec


def add_pod(e: Env, role: str = "gpu", *, minutes: float = 60, hard_minutes: float = 120, record: bool = True) -> str:
    info = e.be._new("gpu" if role == "gpu" else "cpu", 0.69 if role == "gpu" else 0.28, minutes, hard_minutes, f"hm-farm-t-{role}")
    if record:
        watch.record_pod(e.state, info, role, by="test")
    return info["pod_id"]


def make_watcher(e: Env, **kw) -> watch.Watcher:
    pol = dict(poll_s=0.1, runpod_poll_s=0.0, launch_gap_s=0.0, idle_grace_s=3600.0, state_s=0.0, coverage_s=1e9, pull_workers=2, max_requeues=1,
               start_timeout_s=1e9, stale_hb_s=600.0)
    pol.update(kw)
    return watch.Watcher(DANCE, backend=e.be, policy=watch.Policy(**pol), dest_root=e.ws, log=lambda m: None)


def drain(w: watch.Watcher, timeout: float = 90) -> None:
    for f in list(w.pulls.values()) + list(w.actions.values()):
        try:
            f.result(timeout)
        except Exception:  # noqa: BLE001  the failure is handled by collect()
            pass
    w.collect()


def status_of(e: Env, jid: str) -> str:
    return (e.state.load()["chunks"].get(jid) or e.state.load()["jobs"].get(jid))["status"]


def cycles(w: watch.Watcher, n: int = 1) -> None:
    for _ in range(n):
        w.cycle()
        drain(w)


