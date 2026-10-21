"""One entry point from "here are the videos" to a completed analysis (owner, 2026-10-21; docs/PIPELINE_EFFICIENCY.md).

    python -m dancecap.pipeline analyze <folder|video> [--take NAME] --dry-run          # classify + plan, print it
    python -m dancecap.pipeline analyze <folder|video> [--take NAME] --yes [--max-cost USD] [--detach]
    python -m dancecap.pipeline analysis list | show RUN | resume RUN | snap RUN
    python -m dancecap.pipeline analysis step RUN STEP start|done|fail|skip [--note TEXT] [--session SID]

1. Classify the input: one video = the monocular route; a folder of 2+ phone videos = the multi-view route.
2. Probe the videos (ffprobe on this PC: seconds) and create the take (pipeline.intake) or reuse it.
3. Plan the route as a DAG with Runpod for every heavy step (owner: "any job that can be performed on runpod should be
   placed there"; speed over cost). Every step has where it runs, how many pods side by side, p50 / p90 minutes and
   dollars from the measured 2026-10 runs (PIPELINE_EFFICIENCY.md 3-6), and its dependencies. The end-to-end ETA is the
   critical path; it is recomputed live from what has finished.
4. Run it: the steps with a generic executor run unattended (intake, the volume upload of the source videos at t = 0,
   the audio clocks); the steps that still need take-specific code today (PIPELINE_EFFICIENCY.md 5 rank 1) wait for an
   agent, who drives them with DANCECAP_ANALYSIS=<run> DANCECAP_ANALYSIS_STEP=<step> on its `remote run` sessions (the
   monitor groups them under the step) and marks them with `analysis step ... done`.

The run lives in work/pipeline/analyses/<run>/ (analysis.json + events.jsonl + logs/); the Pipeline Monitor shows it
(Farm page: "Analyses"; #/analysis/<run>: the step map, uploads, sessions, cost and ETA) and keeps snapshots under
work/pipeline/snapshots/<run>/. The monitor is started (headless) by every run.

Pure parts (classify, slug, route_plan, schedule, cost) take plain data and are unit-tested.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

from . import paths
from .util import TAKE_RE, append_jsonl, pid_alive, read_json, read_jsonl, write_json_atomic

SCHEMA = "dancecap.analysis/1"
RUN_RE = re.compile(r"^[0-9]{8}-[0-9]{6}-[a-z0-9_]{1,48}$")
STEP_RE = re.compile(r"^[a-z0-9_]{1,40}$")
VIDEO_EXT = (".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm", ".3gp")
AUDIO_EXT = (".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus")
DONE = ("done", "skipped")
OPEN = ("pending", "running", "waiting_agent")
NO_WINDOW = 0x08000000 if os.name == "nt" else 0

# measured rates (docs/PIPELINE_EFFICIENCY.md 2): the home uplink and the pod prices of the 2026-10 runs
UPLOAD_MBPS = {"p50": 2.1, "p90": 0.73}          # successful uploads > 30 MB, 2026-10-19..21: median, p10 of the rate
PRICE_HR = {"gpu": 0.69,        # RTX 4090, community 0.34-0.59 / secure 0.89 (farm and runner, 2026-10-09..21)
            "gpu_l4": 0.43,     # L4 (speech)
            "cpu": 0.28,        # 8 vCPU cpu5c (FARM.md)
            "cpu_gw": 0.07,     # 2 vCPU gateway
            "local": 0.0}
POD_START_MIN = 1.5             # create to ssh / container up (5 % of the round's billed time)


# ------------------------------------------------------------------------------------------------ paths

def analyses_dir() -> Path:
    return paths.work() / "pipeline" / "analyses"


def run_dir(run: str) -> Path:
    if not RUN_RE.match(run or ""):
        raise ValueError(f"bad analysis run id {run!r}")
    return analyses_dir() / run


# ------------------------------------------------------------------------------------------------ classify (pure)

def slug(name: str) -> str:
    """A take name from a folder / file name: lowercase letters, digits and _ (TAKE_RE), at most 40 characters."""
    s = re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")
    s = re.sub(r"_+", "_", s)[:40].strip("_")
    if not s or not s[0].isalnum():
        s = "take_" + s
    return s if TAKE_RE.match(s) else "take"


def classify(path: str | Path) -> dict:
    """{path, kind: single|two|multi, route: mono|multi, videos[{cam, name, path, size_bytes}], soundtrack, take}.
    A folder holds the videos of one dance (any of VIDEO_EXT, hidden files ignored, sorted by name; cameras 01, 02, ...);
    an audio file in it is the soundtrack. Raises ValueError for anything else."""
    p = Path(path).expanduser()
    if p.is_file():
        if p.suffix.lower() not in VIDEO_EXT:
            raise ValueError(f"{p.name}: not a video ({', '.join(VIDEO_EXT)})")
        vids, snd, name = [p], None, p.stem
    elif p.is_dir():
        files = sorted(f for f in p.iterdir() if f.is_file() and not f.name.startswith("."))
        vids = [f for f in files if f.suffix.lower() in VIDEO_EXT]
        auds = [f for f in files if f.suffix.lower() in AUDIO_EXT]
        snd = auds[0] if len(auds) == 1 else None
        name = p.name
        if not vids:
            raise ValueError(f"{p}: no videos in this folder ({', '.join(VIDEO_EXT)})")
    else:
        raise ValueError(f"{p}: no such file or folder")
    n = len(vids)
    kind = "single" if n == 1 else "two" if n == 2 else "multi"
    if n > 99:
        raise ValueError(f"{n} videos: at most 99 cameras")
    return {"path": str(p.resolve()), "kind": kind, "route": "mono" if n == 1 else "multi",
            "videos": [{"cam": f"{i + 1:02d}", "name": f.name, "path": str(f.resolve()), "size_bytes": f.stat().st_size}
                       for i, f in enumerate(vids)],
            "soundtrack": str(snd.resolve()) if snd else None, "take": slug(name)}


def facts_of(cls: dict, probes: dict | None = None) -> dict:
    """{n, duration_s, size_mb, fps, frames}: the take's workload. probes = {cam: probe_file() result}; without them a
    1080p phone at 30 fps of 1 MB/s is assumed (dry plans of folders that ffprobe cannot read)."""
    probes = probes or {}
    vids = cls["videos"]
    durs = [float((probes.get(v["cam"]) or {}).get("duration_s") or 0) for v in vids]
    durs = [d for d in durs if d > 0]
    size_mb = sum(v["size_bytes"] for v in vids) / 1e6
    # several phones: the analysed window is what most of them cover (the median: a phone that filmed only a part is
    # left out of the fits, as the recap's 77-s phone was); one phone: all of it
    dur = (sorted(durs)[len(durs) // 2] if durs else max(5.0, size_mb / max(1, len(vids)) / 1.0))
    fpss = [float((probes.get(v["cam"]) or {}).get("fps") or 0) for v in vids]
    fps = sorted([f for f in fpss if f > 0])[len([f for f in fpss if f > 0]) // 2] if any(f > 0 for f in fpss) else 30.0
    return {"n": len(vids), "duration_s": round(dur, 2), "size_mb": round(size_mb, 1), "fps": round(fps, 3),
            "frames": int(round(dur * fps)), "probed": bool(durs)}


# ------------------------------------------------------------------------------------------------ the routes (pure)

def _step(id, label, where, deps, p50, p90, lanes=1, basis="", executor="agent", optional=False, price="gpu",
          note="", cmd=None) -> dict:
    return {"id": id, "label": label, "where": where, "deps": list(deps), "lanes": max(1, int(lanes)),
            "p50_min": round(float(p50), 1), "p90_min": round(float(max(p90, p50)), 1), "basis": basis,
            "executor": executor, "optional": optional, "price_hr": PRICE_HR[price] if where != "local" else 0.0,
            "note": note, "cmd": cmd}


def route_plan(route: str, f: dict, opts: dict | None = None) -> list[dict]:
    """The steps of a route for a workload `f` (facts_of). opts: {room_known, realism, speech, film, upload}.
    Where: pod-gpu / pod-cpu = Runpod (every heavy step), local = this PC (light only: intake, the audio clocks, the Unity
    export and film). Executors: builtin (analyze runs it), cmd (a generic command, run by analyze), farm / agent (take-
    specific today: an agent drives it, tagged DANCECAP_ANALYSIS; PIPELINE_EFFICIENCY.md 5 rank 1 makes them generic)."""
    o = {"room_known": False, "realism": True, "speech": False, "film": True, "upload": True, **(opts or {})}
    D = float(f["duration_s"])
    Dm = D / 60.0
    S = float(f["size_mb"])
    frames = max(1, int(f["frames"]))
    up50 = S / UPLOAD_MBPS["p50"] / 60 + 0.5
    up90 = S / UPLOAD_MBPS["p90"] / 60 + 1.0
    steps = [_step("intake", "Probe the videos, create the take", "local", [], 0.5, 1.0, basis="ffprobe: seconds",
                   executor="builtin", price="local")]
    if o["upload"]:
        steps.append(_step("upload", "Upload the videos once to the Runpod volume (6 streams, resumable)", "pod-cpu",
                           ["intake"], up50, up90, basis=f"{S:.0f} MB at {UPLOAD_MBPS['p50']} MB/s (p90 at "
                           f"{UPLOAD_MBPS['p90']} MB/s), measured 2026-10-19..21", executor="builtin", price="cpu_gw"))
    up = ["upload"] if o["upload"] else ["intake"]
    if route == "multi":
        C = int(f["n"])
        K = max(1, math.ceil(D / 30.0))
        steps.append(_step("clocks", "Audio clocks of every phone", "local", ["intake"], 1 + 0.15 * C, 3 + 0.4 * C,
                           basis="pipeline audio-sync (audio only, light; a CPU pod once the volume jobs take it)",
                           executor="cmd", price="local", cmd=["-m", "dancecap.pipeline", "audio-sync", "{take}"]))
        room = []
        if not o["room_known"]:
            steps.append(_step("room", "Room map (new room)", "pod-gpu", up, 20, 35, basis="estimate (audit 6)"))
            room = ["room"]
        steps += [
            _step("cameras", "Camera focal + track, one pod per phone", "pod-gpu", up + ["clocks"] + room, 7, 12, lanes=C,
                  basis="measured 5-9 min per camera job (recap R2)"),
            _step("people", "The two dancers in every phone, one pod per phone", "pod-gpu", up + ["clocks"], 7, 12, lanes=C,
                  basis="measured 4-9 min per camera run (recap R3)"),
            _step("fronts", f"Chunk fronts (SAM 2.1, ma_2d, sync), {K} x 30 s", "pod-gpu", ["cameras", "people"], 12, 25,
                  lanes=K, basis="measured 20-38 min with 2-3 SAM per GPU; one GPU per camera group", executor="farm"),
            _step("shape", "Shape pass (free betas, 30 s)", "pod-gpu", ["cameras", "people"], 25, 35,
                  basis="measured 20-25 min", executor="farm"),
            _step("fits", f"3D pose fits, {K} chunks of 30 s, one pod each", "pod-gpu", ["fronts", "shape"], 18, 30,
                  lanes=K, basis="measured 952 s for 900 frames on a 4090 (farm drill)", executor="farm"),
            _step("stitch", "Stitch the chunks", "pod-cpu", ["fits"], 2.5, 5, basis="measured 70 s + pod start",
                  executor="farm", price="cpu"),
            _step("analyses", "Timing, physics (windowed), counterbalance, floorcraft", "pod-cpu", ["stitch"], 12, 25,
                  lanes=4, basis="measured 8 / 23 / 14 min on 64 GB CPU pods", executor="farm", price="cpu"),
            _step("avatars", "LHM++ splat avatars", "pod-gpu", ["people", "shape"], 10, 20,
                  basis="measured 380 s install + 124 s build (prebuilt env: ~3 min)", optional=not o["realism"]),
        ]
        if o["speech"]:
            steps.append(_step("speech", "Speech (ASR, diarization)", "pod-gpu", up + ["clocks"], 25, 74,
                               basis="measured 74 min (clean re-run ~25)", price="gpu_l4", optional=True))
        steps.append(_step("export", "Export the Unity capture", "local", ["analyses", "avatars"], 3, 6,
                           basis="minutes (writes the Unity folder)", price="local"))
    else:
        shards_people = 6
        people50 = frames * (82 * 60 / 2643) / 60 / shards_people + 3          # 82 min for 2643 frames on one 4090
        sam50 = frames * ((1785 + 306) / 2643) / 60 / 4 + 2                     # 3DB infer + convert, 4 shards
        K = max(1, math.ceil(D / 22.0))
        steps += [
            _step("beats", "Beat grid", "pod-cpu", up, 2, 5, basis="beat_this: seconds of GPU, a minute on a CPU pod",
                  price="cpu"),
            _step("camera", "Camera track + focal, 4 shards", "pod-gpu", up, 9, 15, lanes=4,
                  basis="measured ~11 min on one pod (john_lula J2)"),
            _step("people", f"People, bibs, SAM 2.1, whole-body pose, {shards_people} time shards", "pod-gpu", up,
                  people50, people50 * 1.6, lanes=shards_people, basis="measured 82 min for 2643 frames on one 4090"),
            _step("sam3d_body", "SAM 3D Body pose prior, 4 shards", "pod-gpu", ["people"], sam50, sam50 * 1.6, lanes=4,
                  basis="measured 1785 s infer + 306 s convert for 2643 frames", optional=not o["realism"]),
            _step("shape", "Shape pass (statures held)", "pod-gpu", ["people", "camera"], 20, 30,
                  basis="measured 20 min (fit 1202 s)"),
            _step("fits", f"MAMMA_MONO + prior, {K} chunks of ~22 s", "pod-gpu",
                  ["shape"] + (["sam3d_body"] if o["realism"] else []), 23, 40, lanes=K,
                  basis="measured 40 min for 46 s, scaled per chunk"),
            _step("post", "Stitch, facing repair, foot locking", "pod-cpu", ["fits"], 4, 8, basis="seconds to minutes of CPU",
                  price="cpu"),
            _step("analyses", "Timing, physics, counterbalance, floorcraft", "pod-cpu", ["post"], 9, 15, lanes=2,
                  basis="measured 9 min on 2 CPU pods (J4)", price="cpu"),
            _step("avatars", "LHM++ avatars, renders, encode on the pod", "pod-gpu", ["people", "post"], 10, 18,
                  basis="measured 380 s install + 143 s render (prebuilt env: ~3 min); encode on the pod",
                  optional=not o["realism"]),
            _step("export", "Export the Unity capture", "local", ["analyses", "avatars"], 2, 4, basis="minutes",
                  price="local"),
        ]
    if o["film"]:
        steps.append(_step("film", "Film render (Unity, 1080p)", "local", ["export"], Dm * 1.1 + 0.5, Dm * 1.6 + 2,
                           basis="measured about 28 frames/s", price="local", optional=True))
    return steps


def step_cost(st: dict, minutes: float | None = None) -> float:
    """$ of a pod step: price x lanes x (minutes + pod start)."""
    if st["where"] == "local" or not st.get("price_hr"):
        return 0.0
    m = st["p50_min"] if minutes is None else minutes
    return round(st["price_hr"] * st["lanes"] * (m + POD_START_MIN) / 60.0, 3)


def schedule(steps: list[dict], now: float = 0.0, which: str = "p50") -> dict:
    """Earliest finish of every step (minutes from `now` for open ones; real times for finished ones) and the critical
    path. Each step: its own start/end when known (`started`/`ended` epoch seconds, `status`), else its p50/p90.
    Returns {finish: {id: epoch}, total_min, critical: [ids]}."""
    by = {s["id"]: s for s in steps}
    fin: dict[str, float] = {}
    via: dict[str, str | None] = {}

    def f(sid: str) -> float:
        if sid in fin:
            return fin[sid]
        s = by[sid]
        dur = float(s.get(f"{which}_min") or 0) * 60
        if s.get("status") in DONE:
            fin[sid] = float(s.get("ended") or now)
            via[sid] = None
            return fin[sid]
        deps = [d for d in s["deps"] if d in by]
        start, best = now, None
        for d in deps:
            t = f(d)
            if t > start:
                start, best = t, d
        if s.get("status") == "running" and s.get("started"):
            el = max(0.0, now - float(s["started"]))
            left = max(dur - el, 0.1 * dur)
            if s.get("eta_s") is not None:
                left = float(s["eta_s"])
            fin[sid] = now + left
            via[sid] = None
        else:
            fin[sid] = start + dur
            via[sid] = best
        return fin[sid]
    for s in steps:
        f(s["id"])
    end_id = max(fin, key=lambda k: fin[k]) if fin else None
    crit = []
    k = end_id
    while k:
        crit.append(k)
        k = via.get(k)
    return {"finish": fin, "end": fin.get(end_id, now) if end_id else now, "total_min": round(
        (fin.get(end_id, now) - now) / 60.0, 1) if end_id else 0.0, "critical": crit[::-1]}


def totals(steps: list[dict]) -> dict:
    s50 = schedule([{**s, "status": "pending"} for s in steps], 0.0, "p50")
    s90 = schedule([{**s, "status": "pending"} for s in steps], 0.0, "p90")
    c50 = sum(step_cost(s) for s in steps)
    c90 = sum(step_cost(s, s["p90_min"]) for s in steps)
    pod_steps = [s for s in steps if s["where"] != "local"]
    return {"wall_min": {"p50": s50["total_min"], "p90": s90["total_min"]}, "critical": s50["critical"],
            "cost_usd": {"p50": round(c50, 2), "p90": round(c90, 2)},
            "pods_max": max([s["lanes"] for s in pod_steps] or [0]),
            "on_runpod": [s["id"] for s in pod_steps], "local": [s["id"] for s in steps if s["where"] == "local"]}


def live(doc: dict, now: float | None = None) -> dict:
    """The live figures of a run: done / total, end-to-end ETA p50 / p90 (critical path from now), cost so far (ledger of
    its sessions) and projected. Works on analysis.json as stored (pure but for the clock)."""
    now = time.time() if now is None else now
    steps = doc.get("steps") or []
    req = [s for s in steps if not s.get("optional")]
    s50 = schedule(steps, now, "p50")
    s90 = schedule(steps, now, "p90")
    so_far = float(doc.get("cost_so_far_usd") or 0.0)
    rem50 = sum(step_cost(s) for s in steps if s.get("status") not in DONE and s.get("status") != "failed")
    rem90 = sum(step_cost(s, s["p90_min"]) for s in steps if s.get("status") not in DONE and s.get("status") != "failed")
    open_ = [s for s in steps if s.get("status") in OPEN] if doc.get("status") != "cancelled" else []
    return {"done": sum(1 for s in req if s.get("status") in DONE), "total": len(req),
            "running": [s["id"] for s in steps if s.get("status") == "running"],
            "waiting_agent": [s["id"] for s in steps if s.get("status") == "waiting_agent"] if open_ else [],
            "failed": [s["id"] for s in steps if s.get("status") == "failed"],
            "eta_s": {"p50": round(s50["end"] - now) if open_ else 0, "p90": round(s90["end"] - now) if open_ else 0},
            "finish_at": {"p50": round(s50["end"]) if open_ else None, "p90": round(s90["end"]) if open_ else None},
            "critical": s50["critical"], "elapsed_s": round(now - float(doc.get("created") or now)),
            "cost_usd": {"so_far": round(so_far, 2), "p50": round(so_far + rem50, 2), "p90": round(so_far + rem90, 2)}}


# ------------------------------------------------------------------------------------------------ run state

def _lock_path(run: str) -> Path:
    return run_dir(run) / "analysis.lock"


class _RunLock:
    """O_EXCL lock file around analysis.json read-modify-write (the driver and `analysis step` from agents)."""

    def __init__(self, run: str, timeout: float = 20.0):
        self.p, self.timeout = _lock_path(run), timeout

    def __enter__(self):
        t_end = time.time() + self.timeout
        while True:
            try:
                fd = os.open(self.p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                return self
            except FileExistsError:
                try:
                    if time.time() - self.p.stat().st_mtime > 30:
                        self.p.unlink()
                        continue
                except OSError:
                    pass
            except PermissionError:
                pass
            if time.time() > t_end:
                raise TimeoutError(f"{self.p} busy")
            time.sleep(0.05)

    def __exit__(self, *a):
        try:
            self.p.unlink()
        except OSError:
            pass


def load(run: str) -> dict | None:
    d = read_json(run_dir(run) / "analysis.json", None)
    return d if isinstance(d, dict) else None


def save(doc: dict) -> None:
    doc["updated"] = time.time()
    write_json_atomic(run_dir(doc["run"]) / "analysis.json", doc)


def event(run: str, kind: str, step: str | None = None, msg: str = "", **kw) -> None:
    append_jsonl(run_dir(run) / "events.jsonl", {"ts": round(time.time(), 2), "kind": kind, "step": step, "msg": msg[:300],
                                                 **kw})


def update_step(run: str, step: str, **kw) -> dict:
    with _RunLock(run):
        doc = load(run)
        if doc is None:
            raise KeyError(run)
        st = next((s for s in doc["steps"] if s["id"] == step), None)
        if st is None:
            raise KeyError(f"{run}: no step {step}")
        st.update({k: v for k, v in kw.items() if v is not None})
        save(doc)
        return st


def list_runs(limit: int = 30) -> list[dict]:
    d = analyses_dir()
    if not d.is_dir():
        return []
    out = []
    for p in sorted((x for x in d.iterdir() if RUN_RE.match(x.name)), key=lambda x: x.name, reverse=True)[:limit]:
        doc = read_json(p / "analysis.json", None)
        if isinstance(doc, dict):
            out.append(doc)
    return out


def driver_alive(doc: dict) -> bool:
    drv = doc.get("driver") or {}
    if drv.get("ended") and float(drv["ended"]) >= float(drv.get("started") or 0):
        return False
    return bool(drv.get("pid") and pid_alive(drv["pid"], drv.get("create_time")))


def active_runs() -> list[dict]:
    """Runs that are not finished: a live driver, or steps running / waiting for an agent within the last 2 days."""
    out = []
    for doc in list_runs(10):
        if doc.get("status") in ("done", "failed", "cancelled"):
            continue
        if driver_alive(doc) or time.time() - float(doc.get("updated") or 0) < 2 * 86400:
            out.append(doc)
    return out


def sessions_of(run: str, limit_dirs: int = 400) -> list[dict]:
    """Runner sessions tagged with this analysis (state.json "analysis": {run, step}), newest first. Reads only the newest
    `limit_dirs` session dirs (cheap: state.json is small)."""
    d = paths.sessions_dir()
    if not d.is_dir():
        return []
    out = []
    created = run[:15]
    for p in sorted((x for x in d.iterdir() if x.is_dir()), key=lambda x: x.name, reverse=True)[:limit_dirs]:
        if p.name[:15] < created:
            break
        st = read_json(p / "state.json", None)
        tag = (st or {}).get("analysis") or {}
        if isinstance(tag, dict) and tag.get("run") == run:
            out.append({"sid": p.name, "step": tag.get("step")})
    return out


# ------------------------------------------------------------------------------------------------ create

def _probe_all(cls: dict) -> dict:
    from . import probe
    out = {}
    for v in cls["videos"]:
        try:
            out[v["cam"]] = probe.probe_file(v["path"])
        except Exception as e:  # noqa: BLE001
            out[v["cam"]] = {"error": str(e)[:200]}
    return out


def make_run(path: str, take: str | None = None, opts: dict | None = None, probe: bool = True,
             now: float | None = None) -> dict:
    """Classify + probe + plan -> an analysis document (not saved)."""
    cls = classify(path)
    take = take or cls["take"]
    if not TAKE_RE.match(take):
        raise ValueError(f"take name {take!r}: lowercase letters, digits and _ only")
    probes = _probe_all(cls) if probe else {}
    f = facts_of(cls, probes)
    from . import takes as T
    exists = T.exists(take)
    o = dict(opts or {})
    steps = route_plan(cls["route"], f, o)
    for s in steps:
        s["status"] = "pending"
    now = time.time() if now is None else now
    run = f"{time.strftime('%Y%m%d-%H%M%S', time.localtime(now))}-{take}"
    doc = {"schema": SCHEMA, "run": run, "take": take, "created": now, "updated": now, "status": "planned",
           "agent_hint": (f"Steps marked 'needs an agent' have no generic executor yet (take-specific code today). Drive "
                          f"one with DANCECAP_ANALYSIS={run} DANCECAP_ANALYSIS_STEP=<step> set on its `dancecap remote run` "
                          f"sessions (they appear under the step), then: python -m dancecap.pipeline analysis step {run} "
                          "<step> done"),
           "input": {**cls, "probes": {c: {k: p.get(k) for k in ("duration_s", "fps", "display_w", "display_h", "codec",
                                                                    "audio", "size_bytes", "error") if k in p}
                                       for c, p in probes.items()}},
           "facts": f, "options": o, "take_exists": exists, "steps": steps, "plan": totals(steps),
           "cost_so_far_usd": 0.0}
    return doc


# ------------------------------------------------------------------------------------------------ executors

def _intake(doc: dict, spec: dict | None) -> str:
    """Create the take, or reuse it when it exists with the same videos."""
    from . import intake, takes as T
    take = doc["take"]
    want = {Path(v["path"]).resolve() for v in doc["input"]["videos"]}
    if T.exists(take):
        cfg, _ = T.load_cfg(take)
        have = {Path(v["path"]).resolve() for v in T.videos(take, cfg) if v.get("path")}
        if want <= have:
            return f"take {take} exists with these videos: reused"
        raise RuntimeError(f"take {take} exists with other videos ({len(have)}); pass --take <new name>")
    spec = dict(spec or {})
    missing = [k for k in ("dancers", "dance") if not spec.get(k)]
    if missing:
        raise RuntimeError("a new take needs the dancers and the dance facts: pass --lead NAME:STATURE --follow "
                           "NAME:STATURE --level LEVEL --category CATEGORY [--consent-cloud] or --spec FILE.json "
                           f"(missing: {', '.join(missing)})")
    req = {"take": take, "source_mode": "reference",
           "videos": [{"cam": v["cam"], "path": v["path"]} for v in doc["input"]["videos"]], **spec}
    if doc["input"].get("soundtrack") and not req.get("soundtrack"):
        req["soundtrack"] = {"path": doc["input"]["soundtrack"]}
    res = intake.create(req)
    return f"created {res['toml_path']} ({res['variant']}-view)" + (f"; warnings: {'; '.join(res['warnings'])[:200]}"
                                                                   if res.get("warnings") else "")


def _upload(doc: dict, report) -> str:
    """The source videos to the hm-farm volume (src/<take>/), through a CPU gateway pod THIS run creates (never another
    workflow's pod), striped over 6 ssh streams, hash-diffed (a video already there is skipped). The gateway is deleted at
    the end."""
    from ..farm import pods, volume
    items = [(Path(v["path"]), f"src/{doc['take']}/{v['name']}") for v in doc["input"]["videos"]]
    if doc["input"].get("soundtrack"):
        items.append((Path(doc["input"]["soundtrack"]), f"src/{doc['take']}/{Path(doc['input']['soundtrack']).name}"))
    pod, created = pods.gateway(minutes=45, reuse=False, owner=f"analysis:{doc['run']}")
    try:
        total = sum(p.stat().st_size for p, _ in items)
        sent = {"n": 0}
        done_before = {}

        def prog(rel, done, size):
            done_before[rel] = done
            report(sum(done_before.values()), total)
        res = volume.Link(pod).push_files(items, progress=prog)
        return (f"{len(items)} file(s), {total / 1e6:.0f} MB -> volume src/{doc['take']}/: {res.get('sent_files', '?')} sent "
                f"({res.get('sent_bytes', 0) / 1e6:.0f} MB in {res.get('seconds', '?')} s), {res.get('skipped', 0)} already there")
    finally:
        if created:
            try:
                pods.delete(pod.id)
            except Exception as e:  # noqa: BLE001
                print(f"[analyze] gateway {pod.id}: delete failed ({e}); its kill switch removes it", flush=True)


# ------------------------------------------------------------------------------------------------ the driver

class Driver:
    """Runs the runnable steps in DAG order, several at once; marks the agent steps as waiting; recomputes, snapshots and
    exits when nothing more can run (status waiting / done / failed). Re-reads analysis.json every loop, so marks made
    with `analysis step` by an agent are picked up."""

    def __init__(self, run: str, spec: dict | None = None, allow_pods: bool = False, poll_s: float = 2.0):
        self.run, self.spec, self.allow_pods, self.poll_s = run, spec, allow_pods, poll_s
        self.threads: dict[str, threading.Thread] = {}
        self.results: dict[str, tuple[str, str]] = {}
        self.last_snap = 0.0

    def _register(self) -> None:
        ct = None
        try:
            import psutil
            ct = psutil.Process(os.getpid()).create_time()
        except Exception:  # noqa: BLE001
            pass
        with _RunLock(self.run):
            doc = load(self.run)
            doc["driver"] = {"pid": os.getpid(), "create_time": ct, "started": time.time()}
            doc["status"] = "running"
            for s in doc["steps"]:   # a step this driver's executors ran when a previous driver died: run it again
                if s["status"] == "running" and s["executor"] in ("builtin", "cmd"):
                    s.update(status="pending", message="restarted (the previous driver ended while it ran)")
            save(doc)

    def _snap(self, event_: str, label: str, final: bool = False) -> None:
        try:
            from . import snapshots
            snapshots.request(self.run, event_, label, final=final)
            self.last_snap = time.time()
        except Exception:  # noqa: BLE001
            pass

    def _start(self, st: dict) -> None:
        sid = st["id"]
        update_step(self.run, sid, status="running", started=time.time(), message=None)
        event(self.run, "start", sid, st["label"])
        print(f"[analyze] {sid}: start ({st['where']})", flush=True)

        def work():
            try:
                doc = load(self.run)
                if sid == "intake":
                    msg = _intake(doc, self.spec)
                elif sid == "upload":
                    last = [0.0]

                    def report(done, total):
                        if time.time() - last[0] > 2 or done >= total:
                            last[0] = time.time()
                            update_step(self.run, "upload", progress={"done": done, "total": total, "unit": "bytes",
                                                                      "ts": time.time()})
                    msg = _upload(doc, report)
                elif st.get("executor") == "cmd" and st.get("cmd"):
                    argv = [sys.executable] + [a.replace("{take}", doc["take"]) for a in st["cmd"]]
                    logf = run_dir(self.run) / "logs" / f"{sid}.log"
                    logf.parent.mkdir(parents=True, exist_ok=True)
                    with open(logf, "a", encoding="utf-8") as fh:
                        rc = subprocess.run(argv, cwd=str(paths.root()), stdout=fh, stderr=subprocess.STDOUT,
                                            stdin=subprocess.DEVNULL, creationflags=NO_WINDOW).returncode
                    if rc != 0:
                        raise RuntimeError(f"{' '.join(st['cmd'])} rc={rc} (log {paths.rel_display(logf)})")
                    msg = f"rc 0 (log {paths.rel_display(logf)})"
                else:
                    raise RuntimeError("no executor")
                self.results[sid] = ("done", msg)
            except Exception as e:  # noqa: BLE001
                self.results[sid] = ("failed", f"{type(e).__name__}: {e}"[:400])
        t = threading.Thread(target=work, daemon=True, name=f"analyze-{sid}")
        self.threads[sid] = t
        t.start()

    def loop(self) -> int:
        resumed = bool((load(self.run) or {}).get("driver"))
        self._register()
        self._snap("resume" if resumed else "start", "driver resumed" if resumed else "analysis started: the plan")
        while True:
            for sid, (res, msg) in list(self.results.items()):
                self.results.pop(sid)
                self.threads.pop(sid, None)
                update_step(self.run, sid, status=res, ended=time.time(), message=msg)
                event(self.run, res, sid, msg)
                print(f"[analyze] {sid}: {res}: {msg}", flush=True)
                self._snap(f"{sid}_{res}", f"{sid}: {res}")
            doc = load(self.run)
            by = {s["id"]: s for s in doc["steps"]}
            changed = False
            for st in doc["steps"]:
                if st["status"] != "pending":
                    continue
                deps = [by[d] for d in st["deps"] if d in by]
                if any(d["status"] == "failed" for d in deps):
                    update_step(self.run, st["id"], status="blocked", message="a step it needs failed")
                    changed = True
                    continue
                if not all(d["status"] in DONE or (d.get("optional") and d["status"] in ("blocked",)) for d in deps):
                    continue
                if st["id"] in self.threads:
                    continue
                runnable = st["executor"] in ("builtin", "cmd")
                if st["id"] == "upload" and not self.allow_pods:
                    update_step(self.run, "upload", status="waiting_agent", message="needs --yes (it starts a CPU "
                                "gateway pod, about $0.01-0.05)")
                    changed = True
                    continue
                if runnable:
                    self._start(st)
                    changed = True
                else:
                    update_step(self.run, st["id"], status="waiting_agent",
                                message=f"needs an agent: take-specific code today ({st['executor']} step)")
                    event(self.run, "waiting_agent", st["id"], st["label"])
                    changed = True
            if changed:
                continue
            if self.threads:
                if time.time() - self.last_snap > 900:
                    self._snap("heartbeat", "progress (every 15 min)")
                time.sleep(self.poll_s)
                continue
            doc = load(self.run)
            st_all = [s["status"] for s in doc["steps"]]
            req = [s for s in doc["steps"] if not s.get("optional")]
            if all(s["status"] in DONE for s in req):
                status = "done"
            elif any(s in ("failed", "blocked") for s in st_all) and not any(s == "waiting_agent" for s in st_all):
                status = "failed"
            else:
                status = "waiting"
            with _RunLock(self.run):
                d2 = load(self.run)
                d2["status"] = status
                d2["driver"] = {**(d2.get("driver") or {}), "ended": time.time()}
                save(d2)
            event(self.run, "driver_end", None, status)
            self._snap("driver_end", f"driver stops: {status}", final=status != "waiting")
            print(f"[analyze] {self.run}: {status}" + ("; the waiting steps need an agent (the monitor lists them)"
                                                       if status == "waiting" else ""), flush=True)
            return 0 if status in ("done", "waiting") else 1


def refresh_cost(run: str) -> float:
    """cost_so_far_usd = the ledger cost of the run's sessions (+ the gateway pods it created). Cheap."""
    from . import estimate
    sids = {s["sid"] for s in sessions_of(run)}
    rows = estimate.ledger_rows()
    pod_ids = set()
    for sid in sids:
        st = read_json(paths.sessions_dir() / sid / "state.json", {}) or {}
        if (st.get("pod") or {}).get("id"):
            pod_ids.add(st["pod"]["id"])
    total = 0.0
    now = time.time()
    for r in rows:
        if r.get("pod_id") in pod_ids or str(r.get("owner") or "") == f"analysis:{run}":
            if r.get("deleted_ts") and r.get("cost") is not None:
                total += float(r["cost"])
            else:
                total += float(r.get("cost_per_hr") or 0) * max(0.0, (r.get("deleted_ts") or now) - float(r["created_ts"])) / 3600
    return round(total, 4)


# ------------------------------------------------------------------------------------------------ printing

def fmt_min(m: float) -> str:
    return f"{m / 60:.1f} h" if m >= 90 else f"{m:.0f} min"


def print_plan(doc: dict) -> None:
    f, i = doc["facts"], doc["input"]
    print(f"[analyze] {i['path']}")
    print(f"  input: {f['n']} video(s) -> {i['kind']}-view, route {i['route']}; {f['duration_s']:.1f} s at {f['fps']:g} fps "
          f"({f['frames']} frames), {f['size_mb']:.0f} MB{'' if f['probed'] else ' (not probed: estimates)'}")
    print(f"  take: {doc['take']} ({'exists: reused' if doc['take_exists'] else 'new'}); run {doc['run']}")
    print(f"  {'step':12s} {'where':8s} {'pods':>4s} {'p50':>7s} {'p90':>7s} {'$ p50':>6s}  executor  deps")
    for s in doc["steps"]:
        print(f"  {s['id']:12s} {s['where']:8s} {s['lanes'] if s['where'] != 'local' else '-':>4} "
              f"{fmt_min(s['p50_min']):>7s} {fmt_min(s['p90_min']):>7s} {step_cost(s):>6.2f}  {s['executor']:8s}  "
              f"{','.join(s['deps']) or '-'}{'  (optional)' if s.get('optional') else ''}")
    t = doc["plan"]
    print(f"  end to end: p50 {fmt_min(t['wall_min']['p50'])}, p90 {fmt_min(t['wall_min']['p90'])} "
          f"(critical path: {' -> '.join(t['critical'])})")
    print(f"  pods: ${t['cost_usd']['p50']:.2f} expected, ${t['cost_usd']['p90']:.2f} p90; up to {t['pods_max']} pods at "
          f"once; on Runpod: {', '.join(t['on_runpod'])}; on this PC: {', '.join(t['local'])}")
    agent = [s["id"] for s in doc["steps"] if s["executor"] in ("agent", "farm")]
    if agent:
        print(f"  needs an agent today (take-specific code): {', '.join(agent)}")


# ------------------------------------------------------------------------------------------------ CLI

def _spec_from_args(a) -> dict | None:
    spec = {}
    if a.spec:
        spec = json.loads(Path(a.spec).read_text(encoding="utf-8-sig"))
    ds = {}
    for role in ("lead", "follow"):
        v = getattr(a, role)
        if v:
            name, _, st = v.rpartition(":") if ":" in v else (v, "", "")
            ds[role] = {"name": name or v, "stature_m": st or None}
    if ds:
        spec["dancers"] = {**(spec.get("dancers") or {}), **ds}
    if a.level or a.category:
        spec["dance"] = {**(spec.get("dance") or {}), **({"level": a.level} if a.level else {}),
                         **({"category": a.category} if a.category else {})}
    if a.consent_cloud:
        c = spec.get("consent") or {}
        for role in ("lead", "follow"):
            c[role] = {**(c.get(role) or {}), "cloud_processing": True}
        spec["consent"] = c
    return spec or None


def cmd_analyze(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.pipeline analyze",
                                 description="a folder of phone videos or one video -> a completed analysis")
    ap.add_argument("input", help="a folder of videos (multi-view) or one video (monocular)")
    ap.add_argument("--take", help="take name (default: from the folder / file name)")
    ap.add_argument("--dry-run", action="store_true", help="classify, probe and print the plan; start nothing")
    ap.add_argument("--yes", action="store_true", help="allow the run to start pods (the volume upload gateway)")
    ap.add_argument("--detach", action="store_true", help="run the driver in the background and return")
    ap.add_argument("--no-probe", action="store_true", help="plan from file sizes only (no ffprobe)")
    ap.add_argument("--no-upload", action="store_true", help="no volume upload step")
    ap.add_argument("--room-known", action="store_true", help="multi-view: the room is already solved (no room step)")
    ap.add_argument("--no-realism", action="store_true", help="no SAM 3D Body prior / LHM++ avatars")
    ap.add_argument("--speech", action="store_true", help="multi-view: add ASR / diarization")
    ap.add_argument("--no-film", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--spec", help="JSON with the new take's dancers / dance / consent (pipeline intake format)")
    ap.add_argument("--lead", help="NAME:STATURE, e.g. 'Lula:1.62'")
    ap.add_argument("--follow", help="NAME:STATURE")
    ap.add_argument("--level", choices=["professional", "intermediate", "novice"])
    ap.add_argument("--category", choices=["demo", "lesson", "jack_and_jill"])
    ap.add_argument("--consent-cloud", action="store_true", help="both dancers agreed to cloud processing (pod stages)")
    a = ap.parse_args(argv)
    opts = {"room_known": a.room_known, "realism": not a.no_realism, "speech": a.speech, "film": not a.no_film,
            "upload": not a.no_upload}
    try:
        doc = make_run(a.input, a.take, opts, probe=not a.no_probe)
    except ValueError as e:
        print(f"[analyze] {e}", file=sys.stderr)
        return 2
    if a.json:
        print(json.dumps(doc, indent=1, default=str))
    else:
        print_plan(doc)
    if a.dry_run:
        return 0
    rd = run_dir(doc["run"])
    rd.mkdir(parents=True, exist_ok=True)
    doc["status"] = "planned"
    doc["spec"] = _spec_from_args(a)
    doc["allow_pods"] = bool(a.yes)
    save(doc)
    event(doc["run"], "planned", None, f"{doc['input']['route']} route, {len(doc['steps'])} steps")
    from . import monitor
    monitor.ensure_running(f"analyze {doc['run']}")
    print(f"[analyze] run {doc['run']}: the monitor shows it (python -m dancecap.pipeline monitor --launch); "
          f"snapshots in work/pipeline/snapshots/{doc['run']}/", flush=True)
    if a.detach:
        return spawn_driver(doc["run"])
    return Driver(doc["run"], spec=doc["spec"], allow_pods=bool(a.yes)).loop()


def spawn_driver(run: str) -> int:
    flags = (0x08000000 | 0x00000200) if os.name == "nt" else 0
    logf = run_dir(run) / "logs" / "driver.log"
    logf.parent.mkdir(parents=True, exist_ok=True)
    with open(logf, "a", encoding="utf-8") as fh:
        subprocess.Popen([sys.executable, "-m", "dancecap.pipeline", "analysis", "resume", run], cwd=str(paths.root()),
                         stdin=subprocess.DEVNULL, stdout=fh, stderr=subprocess.STDOUT, creationflags=flags,
                         close_fds=True)
    print(f"[analyze] driver started in the background (log {paths.rel_display(logf)})")
    return 0


def cmd_analysis(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.pipeline analysis")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    p = sub.add_parser("show")
    p.add_argument("run")
    p = sub.add_parser("resume", help="drive the runnable steps again (after an agent marked steps)")
    p.add_argument("run")
    p = sub.add_parser("cancel", help="mark a run cancelled (it leaves the dashboard's active list; nothing is deleted)")
    p.add_argument("run")
    p.add_argument("--note", default="")
    p = sub.add_parser("snap", help="take a dashboard snapshot now")
    p.add_argument("run")
    p.add_argument("--event", default="manual")
    p = sub.add_parser("step", help="mark a step (agents driving a take-specific step)")
    p.add_argument("run")
    p.add_argument("step")
    p.add_argument("action", choices=["start", "done", "fail", "skip"])
    p.add_argument("--note", default="")
    p.add_argument("--session", help="a remote-runner session id that belongs to this step")
    a = ap.parse_args(argv)
    if a.cmd == "list":
        for d in list_runs():
            lv = live(d)
            print(f"{d['run']:40s} {d.get('status', '?'):9s} {lv['done']}/{lv['total']} done  ETA "
                  f"{fmt_min(lv['eta_s']['p50'] / 60)}  ${lv['cost_usd']['so_far']:.2f} -> ${lv['cost_usd']['p50']:.2f}")
        return 0
    if not RUN_RE.match(a.run) or load(a.run) is None:
        print(f"[analysis] no run {a.run}", file=sys.stderr)
        return 2
    if a.cmd == "show":
        d = load(a.run)
        print(json.dumps({**d, "live": live(d), "sessions": sessions_of(a.run)}, indent=1, default=str))
        return 0
    if a.cmd == "cancel":
        with _RunLock(a.run):
            d = load(a.run)
            d["status"] = "cancelled"
            d["cancel_note"] = a.note[:300]
            save(d)
        event(a.run, "cancelled", None, a.note)
        print(f"[analysis] {a.run}: cancelled")
        return 0
    if a.cmd == "snap":
        from . import snapshots
        print(json.dumps(snapshots.take(a.run, a.event, "manual snapshot"), indent=1, default=str))
        return 0
    if a.cmd == "resume":
        from . import monitor
        monitor.ensure_running(f"analysis {a.run}")
        d = load(a.run)
        if driver_alive(d) and (d.get("driver") or {}).get("pid") != os.getpid():
            print(f"[analysis] a driver already runs (pid {d['driver']['pid']})")
            return 0
        return Driver(a.run, spec=d.get("spec"), allow_pods=bool(d.get("allow_pods"))).loop()
    if not STEP_RE.match(a.step):
        return 2
    now = time.time()
    st = {"start": dict(status="running", started=now), "done": dict(status="done", ended=now),
          "fail": dict(status="failed", ended=now), "skip": dict(status="skipped", ended=now)}[a.action]
    if a.note:
        st["message"] = a.note[:300]
    try:
        cur = update_step(a.run, a.step, **st)
    except KeyError as e:
        print(f"[analysis] {e}", file=sys.stderr)
        return 2
    if a.session:
        with _RunLock(a.run):
            d = load(a.run)
            s = next(x for x in d["steps"] if x["id"] == a.step)
            s["sessions"] = sorted(set(s.get("sessions") or []) | {a.session})
            save(d)
    with _RunLock(a.run):
        d = load(a.run)
        d["cost_so_far_usd"] = refresh_cost(a.run)
        if d.get("status") in ("waiting", "planned") and a.action in ("done", "skip"):
            d["status"] = "running"
        save(d)
    event(a.run, a.action, a.step, a.note, by="agent")
    from . import snapshots
    snapshots.request(a.run, f"{a.step}_{a.action}", f"{a.step}: {a.action} (agent)")
    print(f"[analysis] {a.run} {a.step}: {cur['status']}")
    if a.action in ("done", "skip") and not driver_alive(load(a.run)):
        spawn_driver(a.run)   # the next steps may be runnable now
    return 0
