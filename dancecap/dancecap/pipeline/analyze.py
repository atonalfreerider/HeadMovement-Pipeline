"""One entry point from "here are the videos" to a completed analysis (owner, 2026-10-21; docs/PIPELINE_EFFICIENCY.md).

    python -m dancecap.pipeline analyze <folder|video> [--take NAME] --dry-run          # classify + plan, print it
    python -m dancecap.pipeline analyze <folder|video> [--take NAME] --yes [--max-cost USD] [--detach]
    python -m dancecap.pipeline analysis list | show RUN | resume RUN | snap RUN | cancel RUN
    python -m dancecap.pipeline analysis step RUN STEP start|done|fail|skip|retry [--reused] [--note TEXT] [--session SID]
    python -m dancecap.pipeline analysis budget RUN          # what is left of the run's pod limit (for --max-cost)
    python -m dancecap.pipeline analysis limit RUN USD|none  # change the run's pod limit

1. Classify the input: one video = the monocular route; a folder of 2+ phone videos = the multi-view route.
2. Probe the videos (ffprobe on this PC: seconds) and create the take (pipeline.intake) or reuse it. A new one-video
   take gets the window (the whole video), the 30 Hz lattice, [stitch] mono and [export] its stages need.
3. Plan the route as a DAG with Runpod for every heavy step (owner: "any job that can be performed on runpod should be
   placed there"; speed over cost), from what is BUILT today: one pod per step unless the step's split exists in code
   (farm chunk fits, one session per phone), and every pod step pays the measured pod-session overhead (bundle, pod
   start, upload, bootstrap, installs, download: SESSION_OVERHEAD_MIN). Every step has where it runs, how many pods side
   by side, p50 / p90 minutes and dollars, and its dependencies. The end-to-end ETA is the critical path, recomputed
   live: a running step that is past its estimate pushes the ETA out (residual_min), it never shrinks it toward 0.
4. Run it: the steps with a generic executor run unattended (intake, the volume upload of the source videos at t = 0,
   the audio clocks); the steps that still need take-specific code today (PIPELINE_EFFICIENCY.md 5 rank 1) wait for an
   agent, who drives them with DANCECAP_ANALYSIS=<run> DANCECAP_ANALYSIS_STEP=<step> on its `remote run` sessions (the
   monitor groups them under the step) and marks them with `analysis step ... done` (`--reused` when it reused
   results instead of computing them: no pod cost, drawn apart on the dashboard).
5. --max-cost USD is the run's pod limit: the driver starts no pod step past it, `analysis budget` prints what is
   left (agents pass it as each session's --max-cost) and the runner caps tagged sessions with budget_left().

The run lives in work/pipeline/analyses/<run>/ (analysis.json + events.jsonl + logs/); the Pipeline Monitor shows it
(Farm page: "Analyses"; #/analysis/<run>: the step map, uploads, sessions, cost and ETA) and keeps snapshots under
work/pipeline/snapshots/<run>/. The monitor is started (headless) by every run. One driver per run (registered under
the run lock); a detached heartbeat process per unfinished run takes a snapshot every 15 min while anything moves.

Pure parts (classify, slug, route_plan, residual_min, schedule, cost, single_view_defaults) are unit-tested.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

from . import paths
from .util import SID_RE, TAKE_RE, append_fd, append_jsonl, mtime, pid_alive, read_json, read_jsonl, write_json_atomic, \
    write_text_atomic

SCHEMA = "dancecap.analysis/1"
RUN_RE = re.compile(r"^[0-9]{8}-[0-9]{6}-[a-z0-9_]{1,48}$")
STEP_RE = re.compile(r"^[a-z0-9_]{1,40}$")
VIDEO_EXT = (".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm", ".3gp")
AUDIO_EXT = (".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus")
DONE = ("done", "skipped")
OPEN = ("pending", "running", "waiting_agent")
FINAL = ("done", "failed", "cancelled")       # run states: nothing more will run (ELAPSED stops, the heartbeat ends)
NO_WINDOW = 0x08000000 if os.name == "nt" else 0
DETACHED = 0x00000200 if os.name == "nt" else 0   # CREATE_NEW_PROCESS_GROUP: outlives the caller's console

# measured rates (docs/PIPELINE_EFFICIENCY.md 2): the home uplink and the pod prices of the 2026-10 runs
UPLOAD_MBPS = {"p50": 2.1, "p90": 0.73}          # successful uploads > 30 MB, 2026-10-19..21: median, p10 of the rate
PRICE_HR = {"gpu": 0.89,        # RTX 4090 secure: the farm's EU-RO-1 pods and every runner pod that worked in the
                                # 2026-10-21 trial (community 0.34-0.59 only when a non-broken host is in stock)
            "gpu_l4": 0.43,     # L4 (speech)
            "cpu": 0.28,        # 8 vCPU cpu5c (FARM.md)
            "cpu_mem": 0.52,    # 64 GB cpu5m: timing / physics memory grows with the take (J4, the recap's OOM kills)
            "cpu_gw": 0.07,     # 2 vCPU gateway
            "local": 0.0}
# One remote-runner pod session costs this much wall before and after its job's compute: bundle, pod create to ssh,
# CUDA check, upload, bootstrap, tool installs, pack + download. The 32 sessions of 2026-10-19..21 whose jobs all worked:
# p50 6.3 min, p90 15.0, max 23.9. Since 2026-10-22 (uv installs, the bootstrap and the setup commands next to the upload,
# the volume route, CPU pods; docs/PIPELINE_EFFICIENCY.md 7d) the 7 live sessions measured 2.3-4.1 min (pod lifetime minus
# compute, median 2.8) with the home uplink at 0.3-1.8 MB/s; p90 keeps room for a slow uplink and stock retries. A
# volume-farm job starts from the prebuilt env: container up 30 s, env 16 s, pull 15-40 s (FARM.md drill).
SESSION_OVERHEAD_MIN = {"p50": 3.0, "p90": 7.0}
FARM_OVERHEAD_MIN = {"p50": 1.0, "p90": 4.0}
SAME_POD_MIN = {"p50": 0.5, "p90": 2.0}          # a step that runs on the previous step's pod (no new session)
# what a fit session sends on top of the usual bundle: the ma_2d checkpoint's state_dict copy (front_pod.MA2D_STRIPPED, 540 MB;
# the trial uploaded the 1.6 GB file in 600 s) + the MAMMA body models ma_3d loads (locked-head SMPLX_NEUTRAL 109 MB, verts_512
# 21.5 MB; smplx/SMPLX_NEUTRAL is the same file). Measured on the merged runner (2026-10-22, the lk03_cut20b fit jobs): a
# 41 MB bundle + these 670 MB, which a pod on the farm volume links from it (the volume route) and any other pod gets in a
# side bundle through the home uplink. The plan keeps the side bundle (EU-RO-1 stock is not guaranteed).
MA2D_CKPT_MB = 670
RUNNER_MAX_PODS = 6       # `remote run --parallel` starts at most this many pods at once (cli --max-pods default)
# A running step past its plan: the lognormal of its p50 / p90 says how much is left given that it has not finished
# (residual_min); past p90, where the plan says nothing more, at least this many x the elapsed time is left. Measured:
# the trial's SAM 3D Body step had 0.36 x its elapsed time left when it reached p90 (a failed A100 session first).
OVERRUN_FLOOR = {"p50": 0.3, "p90": 1.0}
_NORM = statistics.NormalDist()
HEARTBEAT_S = 900         # owner: snapshots at progress points; one every 15 min while anything of the run moves


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


def _pod_step(id, label, where, deps, c50, c90, lanes=1, basis="", executor="agent", optional=False, price="gpu",
              note="", overhead: str = "session", extra_mb: float = 0.0) -> dict:
    """A pod step whose p50 / p90 are its compute PLUS the pod start-up it really pays: `session` = one remote-runner
    session per pod (SESSION_OVERHEAD_MIN, + `extra_mb` that its bundle uploads on top of the usual), `farm` = a
    volume-farm job (prebuilt env), `same_pod` = it runs on the previous step's pod. compute_min / overhead_min keep the
    split for the dashboard.
    Side by side, runner sessions share two things the plan used to treat as each lane's own (2026-10-21 review): the home
    uplink (every lane's session uploads its own bundle, so K fit lanes push K x the 1.6 GB checkpoint through ~2 MB/s:
    63 min for 5, not 13) and `remote run --parallel`'s RUNNER_MAX_PODS (8 phones run in 2 rounds of 6). `rounds` > 1
    multiplies the wall time; step_cost charges each lane once."""
    o = {"session": SESSION_OVERHEAD_MIN, "farm": FARM_OVERHEAD_MIN, "same_pod": SAME_POD_MIN}[overhead]
    lanes = max(1, int(lanes))
    at_once = min(lanes, RUNNER_MAX_PODS) if overhead == "session" else lanes
    rounds = math.ceil(lanes / at_once)
    o50 = o["p50"] + at_once * extra_mb / UPLOAD_MBPS["p50"] / 60
    o90 = o["p90"] + at_once * extra_mb / UPLOAD_MBPS["p90"] / 60
    c90 = max(c90, c50)
    shared = f" for {at_once} lanes on the one uplink" if extra_mb and at_once > 1 else ""
    why = {"session": f"+ {o50:.0f}-{o90:.0f} min of pod session (bundle, start, upload, bootstrap, installs, download)"
                      + (f", {at_once * extra_mb / 1000:.1f} GB of it the ma_2d checkpoint{shared}" if extra_mb else ""),
           "farm": f"+ {o50:.0f}-{o90:.0f} min farm job start (prebuilt env on the volume)",
           "same_pod": "on the previous step's pod: no new session"}[overhead]
    if rounds > 1:
        why += f"; {lanes} lanes in {rounds} rounds of at most {at_once} pods (remote run --parallel)"
    st = _step(id, label, where, deps, rounds * (c50 + o50), rounds * (c90 + o90), lanes=lanes,
               basis=f"{basis}; {why}" if basis else why, executor=executor, optional=optional, price=price, note=note)
    st.update(compute_min={"p50": round(c50, 1), "p90": round(c90, 1)},
              overhead_min={"p50": round(o50, 1), "p90": round(o90, 1), "kind": overhead})
    if rounds > 1:
        st.update(rounds=rounds, pods_at_once=at_once)
    return st


def route_plan(route: str, f: dict, opts: dict | None = None) -> list[dict]:
    """The steps of a route for a workload `f` (facts_of). opts: {room_known, realism, speech, film, upload}.
    Where: pod-gpu / pod-cpu = Runpod (every heavy step), local = this PC (light only: intake, the audio clocks, the Unity
    export and film). Executors: builtin (analyze runs it), cmd (a generic command, run by analyze), farm / agent (take-
    specific today: an agent drives it, tagged DANCECAP_ANALYSIS; PIPELINE_EFFICIENCY.md 5 rank 1 makes them generic).
    Planned from what is built (2026-10-21 review): the mono time shards (camera 4, people 6, SAM 3D Body 4) do not exist
    in code, so those steps are one pod each; splits that exist keep their lanes (farm chunk jobs, one runner session
    per phone with `remote run --parallel`). Every pod step pays its start-up (_pod_step)."""
    o = {"room_known": False, "realism": True, "speech": False, "film": True, "upload": True, **(opts or {})}
    D = float(f["duration_s"])
    Dm = D / 60.0
    S = float(f["size_mb"])
    frames = max(1, int(f["frames"]))
    jl = frames / 2643.0          # john_lula (88.1 s, 2643 frames): the measured mono reference
    # + the gateway pod's start and delete: the step took 52 s for 9 MB, 58 s for 13 MB, 81 s for 76 MB (2026-10-21)
    up50 = S / UPLOAD_MBPS["p50"] / 60 + 0.9
    up90 = S / UPLOAD_MBPS["p90"] / 60 + 2.0
    steps = [_step("intake", "Probe the videos, create the take", "local", [], 0.5, 1.0, basis="ffprobe: seconds",
                   executor="builtin", price="local")]
    if o["upload"]:
        # the volume farm's copy (multi-view: the farm jobs read src/<take>/). The mono route's runner sessions upload their
        # own inputs, so there it is optional and nothing waits for it
        steps.append(_step("upload", "Upload the videos once to the Runpod volume (resumable; a file over 24 MB striped "
                           "over 4 ssh streams, smaller ones in tar batches; 4 files at once)", "pod-cpu", ["intake"], up50, up90,
                           basis=f"{S:.0f} MB at {UPLOAD_MBPS['p50']} MB/s (p90 at {UPLOAD_MBPS['p90']} MB/s), measured "
                           "2026-10-19..21; gateway up in 26-37 s", executor="builtin", price="cpu_gw",
                           optional=route == "mono"))
    up = ["upload"] if o["upload"] else ["intake"]
    if route == "multi":
        C = int(f["n"])
        K = max(1, math.ceil(D / 30.0))
        steps.append(_step("clocks", "Audio clocks of every phone", "local", ["intake"], 1 + 0.15 * C, 3 + 0.4 * C,
                           basis="pipeline audio-sync (audio only, light; a CPU pod once the volume jobs take it)",
                           executor="cmd", price="local", cmd=["-m", "dancecap.pipeline", "audio-sync", "{take}"]))
        room = []
        if not o["room_known"]:
            steps.append(_pod_step("room", "Room map (new room)", "pod-gpu", up, 20, 35, basis="estimate (audit 6)"))
            room = ["room"]
        steps += [
            _pod_step("cameras", "Camera focal + track, one pod per phone", "pod-gpu", up + ["clocks"] + room, 7, 12,
                      lanes=C, basis="measured 5-9 min per camera job (recap R2); one runner session per phone "
                      "(remote run --parallel)"),
            _pod_step("people", "The two dancers in every phone, one pod per phone", "pod-gpu", up + ["clocks"], 7, 12,
                      lanes=C, basis="measured 4-9 min per camera run (recap R3)"),
            _pod_step("fronts", f"Chunk fronts (SAM 2.1, ma_2d, sync), {K} x 30 s", "pod-gpu", ["cameras", "people"], 12,
                      25, lanes=K, basis="measured 20-38 min with 2-3 SAM per GPU; one GPU per camera group",
                      executor="farm", overhead="farm"),
            _pod_step("shape", "Shape pass (free betas, 30 s)", "pod-gpu", ["cameras", "people"], 25, 35,
                      basis="measured 20-25 min", executor="farm", overhead="farm"),
            _pod_step("fits", f"3D pose fits, {K} chunks of 30 s, one pod each", "pod-gpu", ["fronts", "shape"], 18, 30,
                      lanes=K, basis="measured 952 s for 900 frames on a 4090 (farm drill)", executor="farm",
                      overhead="farm"),
            _pod_step("stitch", "Stitch the chunks", "pod-cpu", ["fits"], 1.5, 4, basis="measured 70 s + pull 26 s",
                      executor="farm", price="cpu", overhead="farm"),
            _pod_step("analyses", "Timing, physics (windowed), counterbalance, floorcraft", "pod-cpu", ["stitch"], 12,
                      25, lanes=4, basis="measured 8 / 23 / 14 min on 64 GB CPU pods", executor="farm",
                      price="cpu_mem", overhead="farm"),
            _pod_step("avatars", "LHM++ splat avatars", "pod-gpu", ["people", "shape"], 10, 20,
                      basis="measured 380 s install + 124 s build (prebuilt env: ~3 min)", optional=not o["realism"]),
        ]
        if o["speech"]:
            steps.append(_pod_step("speech", "Speech (ASR, diarization)", "pod-gpu", up + ["clocks"], 25, 74,
                                   basis="measured 74 min (clean re-run ~25)", price="gpu_l4", optional=True))
        steps.append(_step("export", "Export the Unity capture", "local", ["analyses", "avatars"], 3, 6,
                           basis="minutes (writes the Unity folder)", price="local"))
        for s_ in steps:          # what exactly blocks each step that still needs an agent (the run's waiting message)
            if s_["executor"] in ("agent", "farm"):
                s_["blocker"] = MULTI_BLOCKERS.get(s_["id"], "take-specific code today")
    else:
        # every mono step has a generic executor (pipeline/mono_route.py): `python -m dancecap.pipeline.mono_route <step>`,
        # local or one tagged `remote run` per step. The light steps run here (beats 26 s, world ~1 min, post seconds: a pod
        # session costs 3+ min), the analyses on a CPU runner pod (the ready-pods CPU route, PIPELINE_EFFICIENCY.md 7d).
        K = mono_chunks(D, f.get("fps") or 30.0)
        T = min(frames, int(round(CHUNK_MAX_S * 30)))     # frames of one chunk's fit window
        # measured 2026-10-22 on lk03_mono20 (599 frames, 4090): camera installs 0.5 + probe / skeleton / track / MoGe-2 1.7 min;
        # people installs 0.4 + detect 0.5 + SAM 1.4 + RTMW 0.6 + refinement round 2.1 + the rest 0.4 min
        cam50 = 1.0 + 2.0 * frames / 600                # the track scales with the frames (2 shards on one GPU)
        people50 = 1.0 + 4.6 * frames / 600             # detection, SAM 2.1 and RTMW scale with the frames
        sam50 = frames * (1785 + 306) / 2643 / 60 + 3   # 3DB infer + convert per frame on one 4090 (trial: 0.87 s) + install
        fit50 = 2 + 1.8 * T / 60 + 1                    # front ~2 min, MAMMA_MONO + prior 1.8 s per frame (trial), refine
        ana50 = max(2.0, 9 * jl)                        # 9 min for 2643 frames (J4); 66 s for 599 (trial)

        def mono(id, label, where, deps, p50, p90, **kw):
            """A mono step run by mono_route (cmd executor; pod steps pay the runner session's start-up)."""
            cmd = ["-m", "dancecap.pipeline.mono_route", id, "--take", "{take}", "--run", "{run}"]
            if where == "local":
                return _step(id, label, where, deps, p50, p90, executor="cmd", price="local", cmd=cmd, **kw)
            st = _pod_step(id, label, where, deps, p50, p90, executor="cmd", **kw)
            st["cmd"] = cmd
            return st
        realism = ["sam3d_body"] if o["realism"] else []
        steps += [
            mono("beats", "The video's sound + beat grid (beats, heat_beats)", "local", ["intake"], 1, 3,
                 basis="measured 26 s on this PC (2026-10-21 trial); light, a pod session would cost 6+ min"),
            mono("camera", "Camera: PTZ track + MoGe-2, one pod", "pod-gpu", ["intake"], cam50, cam50 * 1.8,
                 basis="measured 2.2 min of pod commands for 599 frames (lk03_mono20, 2026-10-22), scaled by frames; "
                 "cameras.mono_cam"),
            mono("people", "The couple: detect, lead / follow, SAM 2.1, whole-body pose, one pod", "pod-gpu", ["intake"],
                 people50, people50 * 1.8, basis="measured 5.6 min of pod commands for 599 frames (lk03_mono20, "
                 "2026-10-22: two SAM 2.1 rounds), scaled by frames; people_mono (no bibs)"),
            mono("world", "World frame: focal, gravity, camera height from the statures", "local", ["camera", "people"],
                 1.5, 4, basis="CPU: the skeleton re-solved at one focal + the track refine (2 min for 2643 frames, J2)"),
            mono("sam3d_body", "SAM 3D Body pose prior, one pod (frames decoded on the pod)", "pod-gpu", ["world"], sam50,
                 sam50 * 1.6, basis="measured 1785 s infer + 306 s convert for 2643 frames; trial 0.355 s per person-frame "
                 "on a 4090; + import / QA / prior here (43 s)", optional=not o["realism"]),
            mono("fits", f"MAMMA_MONO + prior, {K} chunk(s) of <= {CHUNK_MAX_S:g} s, one pod each", "pod-gpu",
                 ["world"] + realism, fit50, fit50 * 1.6, lanes=K, extra_mb=MA2D_CKPT_MB,
                 basis="trial: front 46 s + fit 1080 s (1.8 s per frame) + refine 52 s for 599 frames; betas pinned from the "
                 "statures (no shape pass)"),
            mono("post", "Stitch the chunks + depth-spike repair", "local", ["fits"], 0.5, 2,
                 basis="stitch 0.4-8 s, jl_repair 0.6-1.8 s (trial)"),
            mono("analyses", "Timing, physics, counterbalance, floorcraft, one CPU pod", "pod-cpu", ["post", "beats"],
                 ana50, ana50 * 2, price="cpu", basis="measured 9 min for 2643 frames (J4), 66 s for 599 (trial); a CPU "
                 "runner pod (the trial's analyses: 50 s on 8 vCPUs against 70 s on a 4090, 2026-10-22)"),
            mono("avatars", "LHM++ splat avatars from the take's own crops, one pod", "pod-gpu", ["post"], 8, 15,
                 optional=True, basis="crops chosen here 20-50 s (lhmpp_generic); pod: LHM++ venv install 2-7.5 min "
                 "(john_lula), both avatars 2.3 min, LBS export ~2 min; nothing in the export reads them"),
            mono("export", "Export the Unity capture (work/<take>/export/, Unity untouched)", "local", ["analyses"], 0.5, 2,
                 basis="measured 4.5-5 s (trial, round 2)"),
        ]
        if o["film"]:
            steps.append(mono("film", "Film render (needs the Unity editor: skipped here)", "local", ["export"], 0.1, 0.3,
                              optional=True, basis="the Unity editor belongs to another workflow"))
        return steps
    if o["film"]:
        steps.append(_step("film", "Film render (Unity, 1080p)", "local", ["export"], Dm * 1.1 + 0.5, Dm * 1.6 + 2,
                           basis="measured about 28 frames/s", price="local", optional=True))
    return steps


# The multi-view route's steps without a generic executor (2026-10-22): the exact reason, shown when the run waits there
MULTI_BLOCKERS = {
    "room": "no generic room-map executor: each room map was built for its take (room.py / room_two_cams / room_single_cam)",
    "cameras": "no generic multi-phone camera executor: a shared world frame needs every phone localised against a solved room "
               "map, and that code is take-specific (cameras/newcam.py WORK paths, recap_cameras for the recap); "
               "cameras.mono_cam solves one phone in its own frame only",
    "people": "no generic multi-view people executor: the dancers are picked by triangulating across phones (recap_people, "
              "carlos_aline_people are take-specific); people_mono handles one phone",
    "fronts": "the volume farm's chunk profiles are per take (farm/recap_profile, ca_profile, jl_profile): no generic profile yet",
    "shape": "the volume farm's chunk profiles are per take: no generic profile yet",
    "fits": "the volume farm's chunk profiles are per take: no generic profile yet",
    "stitch": "the volume farm's chunk profiles are per take: no generic profile yet",
    "analyses": "farm/recap_analysis and ca_analysis are per take: no generic multi-view analysis job yet",
    "avatars": "the LHM++ avatar job is take-specific",
    "speech": "the speech jobs (dancecap/speech) are set up per take",
    "export": "recap_export / carlos_aline_export are per take: no generic multi-view export yet",
}
CHUNK_MAX_S = 26.0       # mono fit windows (mono_route.CHUNK_MAX_S): <= 26 s overlapping by 4 s, cores of ~22 s


def mono_chunks(duration_s: float, fps: float = 30.0) -> int:
    """How many fit chunks mono_route.plan_chunks makes for a video (farm.chunks.equal_windows arithmetic, no imports)."""
    n = max(2, int(round(float(duration_s) * float(fps or 30.0))))
    N = int(math.floor((n - 1) / float(fps or 30.0) * 30.0 + 1e-6)) + 1
    cap, o = CHUNK_MAX_S * 30.0, 4.0 * 30.0
    k = 1
    while N > cap and (N + k * o) / (k + 1) > cap + 1e-9 and k < 200:
        k += 1
    return 1 if N <= cap else k + 1


def step_cost(st: dict, minutes: float | None = None) -> float:
    """$ of a pod step: price x lanes x minutes (its p50 by default, which includes the pod start-up). A step an agent
    marked `reused` computed nothing: $0."""
    if st.get("where") == "local" or not st.get("price_hr") or st.get("reused"):
        return 0.0
    m = st["p50_min"] if minutes is None else minutes
    # a step run in rounds (more lanes than pods at once): its minutes are wall time, each lane pays one round of them
    return round(st["price_hr"] * st.get("lanes", 1) * max(0.0, m) / max(1, int(st.get("rounds") or 1)) / 60.0, 3)


def residual_min(p50, p90, el) -> tuple[float, float]:
    """Minutes (p50, p90) still to go for a step that has run `el` minutes and has not finished: the conditional
    quantiles of a lognormal with the plan's p50 / p90, given that the step outlasted `el` (so a step past its estimate
    pushes the ETA out, by about the plan's spread, instead of collapsing toward 0). Past p90, where the plan says no more,
    at least OVERRUN_FLOOR x the elapsed time is left (ramped in from p50, so the ETA has no jump)."""
    p50 = float(p50 or 0.0)
    p90 = max(float(p90 or 0.0), p50)
    el = max(0.0, float(el or 0.0))
    if p50 <= 0:
        return 0.0, 0.0
    if el <= 0:
        return p50, p90
    sig = max(0.15, math.log(p90 / p50) / 1.2816)
    surv = _NORM.cdf(-math.log(el / p50) / sig)          # P(the step is still running at el)
    out = []
    for q in (0.5, 0.1):
        tail = q * surv
        t = p50 * math.exp(-sig * _NORM.inv_cdf(tail)) if tail > 1e-300 else el
        out.append(max(0.0, t - el))
    ramp = min(1.0, max(0.0, (el - p50) / max(1e-6, p90 - p50)))
    l50 = max(out[0], OVERRUN_FLOOR["p50"] * el * ramp)
    l90 = max(out[1], OVERRUN_FLOOR["p90"] * el * ramp, l50)
    return l50, l90


def session_covers_step(s: dict) -> bool:
    """A running step's live session ETA is the whole step's: measured progress (`eta_basis` "measured": every live
    session's ETA from progress_eta, no generic prior, no timeout, no stall), the step is one session per lane
    (`whole_step`, true unless the plan says otherwise) and every lane has a live session (`live_lanes`)."""
    return (s.get("eta_basis") == "measured" and s.get("whole_step", True) is not False
            and int(s.get("live_lanes") or 0) >= int(s.get("lanes") or 1))


def schedule(steps: list[dict], now: float = 0.0, which: str = "p50") -> dict:
    """Earliest finish of every step (minutes from `now` for open ones; real times for finished ones) and the critical
    path. Each step: its own start/end when known (`started`/`ended` epoch seconds, `status`), else its p50/p90. A running
    step has residual_min() left. Its live pod sessions' ETA (`eta_s`, `eta_s_p90`):
    - covers the whole step (session_covers_step: measured progress of every lane): it IS the time left, in or out. The
      push-out-only rule made the run ETA 3x the step's at p50 and 9x at p90 (2026-10-21 20:59: the analyses session said
      92 s from measured phases, the run said 396 s / 1188 s, 127 s were left; 2026-10-22 review);
    - covers part of the step or is not measured: it can push the time left out, never pull it in (a session covers part
      of a step: the trial's front session said 5 min while the step had 83 to go);
    - made from a job's timeout (`eta_basis` "timeout"): not used.
    Returns {finish: {id: epoch}, end, total_min, critical: [ids], left: {running id: seconds}}."""
    by = {s["id"]: s for s in steps}
    fin: dict[str, float] = {}
    via: dict[str, str | None] = {}
    left: dict[str, float] = {}

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
        if s.get("status") == "running":
            el = max(0.0, now - float(s.get("started") or now)) / 60.0
            l50, l90 = residual_min(s.get("p50_min"), s.get("p90_min"), el)
            lft = (l90 if which == "p90" else l50) * 60
            eta = s.get("eta_s_p90") if which == "p90" and s.get("eta_s_p90") is not None else s.get("eta_s")
            if eta is not None and session_covers_step(s):
                lft = float(eta)
            elif eta is not None and "timeout" not in str(s.get("eta_basis") or ""):
                lft = max(lft, float(eta))
            left[sid] = lft
            fin[sid] = now + lft
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
        (fin.get(end_id, now) - now) / 60.0, 1) if end_id else 0.0, "critical": crit[::-1], "left": left}


def pods_at_once(steps: list[dict], sched: dict | None = None) -> int:
    """The most pods the p50 schedule has up at the same time (steps that overlap add up: the mono route's beats, camera
    and people all start after the upload; the review saw 'up to 1 pods at once' for 3)."""
    sched = sched or schedule([{**s, "status": "pending"} for s in steps], 0.0, "p50")
    iv = [(sched["finish"][s["id"]] - s["p50_min"] * 60, sched["finish"][s["id"]], int(s.get("pods_at_once") or s["lanes"]))
          for s in steps if s["where"] != "local" and s["id"] in sched["finish"]]
    return max([sum(n for a, b, n in iv if a <= t < b) for t, _, _ in iv] or [0])


def totals(steps: list[dict]) -> dict:
    s50 = schedule([{**s, "status": "pending"} for s in steps], 0.0, "p50")
    s90 = schedule([{**s, "status": "pending"} for s in steps], 0.0, "p90")
    c50 = sum(step_cost(s) for s in steps)
    c90 = sum(step_cost(s, s["p90_min"]) for s in steps)
    pod_steps = [s for s in steps if s["where"] != "local"]
    return {"wall_min": {"p50": s50["total_min"], "p90": s90["total_min"]}, "critical": s50["critical"],
            "cost_usd": {"p50": round(c50, 2), "p90": round(c90, 2)},
            "pods_max": pods_at_once(steps, s50),
            "on_runpod": [s["id"] for s in pod_steps], "local": [s["id"] for s in steps if s["where"] == "local"],
            "session_overhead_min": dict(SESSION_OVERHEAD_MIN)}


def run_end(doc: dict) -> float | None:
    """When a finished run ended (None while it is not finished): `ended`, else (runs from before 2026-10-22) its last
    step or driver end, else its last write."""
    if doc.get("status") not in FINAL:
        return None
    if doc.get("ended"):
        return float(doc["ended"])
    ts = [float(s["ended"]) for s in doc.get("steps") or [] if s.get("ended")]
    if (doc.get("driver") or {}).get("ended"):
        ts.append(float(doc["driver"]["ended"]))
    if doc.get("status") == "cancelled" or not ts:
        ts.append(float(doc.get("updated") or doc.get("created") or 0))
    return max(ts)


def live(doc: dict, now: float | None = None) -> dict:
    """The live figures of a run: done / total, end-to-end ETA p50 / p90 (critical path from now), elapsed (stops when
    the run ends), cost so far (ledger of its sessions) and projected: a running step costs its residual minutes, a
    reused step nothing, a finished run nothing more. Works on analysis.json as stored (pure but for the clock)."""
    now = time.time() if now is None else now
    steps = doc.get("steps") or []
    req = [s for s in steps if not s.get("optional")]
    s50 = schedule(steps, now, "p50")
    s90 = schedule(steps, now, "p90")
    so_far = float(doc.get("cost_so_far_usd") or 0.0)
    final = doc.get("status") in FINAL
    rem50 = rem90 = 0.0
    if not final:
        for s in steps:
            if s.get("status") in DONE or s.get("status") == "failed":
                continue
            if s["id"] in s50["left"]:
                rem50 += step_cost(s, s50["left"][s["id"]] / 60.0)
                rem90 += step_cost(s, s90["left"][s["id"]] / 60.0)
            else:
                rem50 += step_cost(s)
                rem90 += step_cost(s, s["p90_min"])
    open_ = [s for s in steps if s.get("status") in OPEN] if not final else []
    end = run_end(doc)
    limit = doc.get("max_cost_usd")
    cost = {"so_far": round(so_far, 2), "p50": round(so_far + rem50, 2), "p90": round(so_far + rem90, 2)}
    if limit is not None:
        cost.update(limit=round(float(limit), 2), left=round(float(limit) - so_far, 2))
    return {"done": sum(1 for s in req if s.get("status") in DONE), "total": len(req),
            "running": [s["id"] for s in steps if s.get("status") == "running"],
            "waiting_agent": [s["id"] for s in steps if s.get("status") == "waiting_agent"] if open_ else [],
            "failed": [s["id"] for s in steps if s.get("status") == "failed"],
            "reused": [s["id"] for s in steps if s.get("reused")],
            "eta_s": {"p50": round(s50["end"] - now) if open_ else 0, "p90": round(s90["end"] - now) if open_ else 0},
            "finish_at": {"p50": round(s50["end"]) if open_ else None, "p90": round(s90["end"]) if open_ else None},
            "critical": s50["critical"], "ended": round(end, 1) if end else None,
            # each running step's time left as the ETA counts it (the step rows show this, not the raw session ETA)
            "left": {k: {"p50": round(v), "p90": round(s90["left"].get(k, v))} for k, v in s50["left"].items()}
            if open_ else {},
            "elapsed_s": round((end if end else now) - float(doc.get("created") or now)),
            "cost_usd": cost}


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


def _sid_epoch(sid: str) -> float:
    try:
        return time.mktime(time.strptime(sid[:15], "%Y%m%d-%H%M%S"))   # session ids are local time
    except ValueError:
        return 0.0


def _session_infos(run: str, extra=(), limit_dirs: int = 400) -> dict[str, dict]:
    """Every runner session created since the run was (plus the `extra` sids an agent named), from its small files:
    {sid: {sid, tag, label, start, end, pod_id, max_minutes, pods}}. Newest `limit_dirs` dirs only (cheap)."""
    d = paths.sessions_dir()
    if not d.is_dir():
        return {}
    names = sorted((x.name for x in d.iterdir() if x.is_dir() and SID_RE.match(x.name)), reverse=True)[:limit_dirs]
    want = [n for n in names if n[:15] >= run[:15]]
    want += [n for n in sorted(set(extra or ())) if n not in want and SID_RE.match(str(n)) and (d / n).is_dir()]
    out = {}
    for n in want:
        st = read_json(d / n / "state.json", None)
        if not isinstance(st, dict):
            continue
        rj = read_json(d / n / "runner.json", None) or {}
        starts = [x for x in (_sid_epoch(n), float(rj.get("started") or 0)) if x > 0]
        tag = st.get("analysis") if isinstance(st.get("analysis"), dict) else None
        pod = st.get("pod") if isinstance(st.get("pod"), dict) else {}
        # the pods it created, when the runner says so: pods_created, or the pod ids of its GPU choice's tries
        # (remote.pod_session choice["tried"][*].pod_id, kept in state pod.choice / gpu_choice)
        listed = [p for p in st.get("pods_created") or [] if isinstance(p, str)]
        for ch in (pod.get("choice"), st.get("gpu_choice")):
            for row in (ch or {}).get("tried") or [] if isinstance(ch, dict) else []:
                if isinstance(row, dict) and isinstance(row.get("pod_id"), str):
                    listed.append(row["pod_id"])
        out[n] = {"sid": n, "tag": tag, "label": str(rj.get("label") or n[16:] or "batch"),
                  "start": min(starts) if starts else 0.0, "end": mtime(d / n / "result.json"),
                  "pod_id": pod.get("id"), "max_minutes": st.get("max_minutes"), "pods": listed}
    return out


def sessions_of(run: str, limit_dirs: int = 400) -> list[dict]:
    """Runner sessions tagged with this analysis (state.json "analysis": {run, step}), newest first."""
    infos = _session_infos(run, limit_dirs=limit_dirs)
    return [{"sid": sid, "step": (i["tag"] or {}).get("step")} for sid, i in sorted(infos.items(), reverse=True)
            if (i["tag"] or {}).get("run") == run]


def _row_cost(r: dict, now: float) -> float:
    """remote.spent()'s rule for one ledger row: a closed row's recorded cost, else price x lifetime up to its kill
    switch (an open row whose close was lost stops accruing there)."""
    if r.get("deleted_ts") and r.get("cost") is not None:
        return float(r["cost"])
    end = r.get("deleted_ts")
    if not end:
        try:
            from .. import remote
            end = remote._row_end(r, now)
        except Exception:  # noqa: BLE001
            end = now
    return float(r.get("cost_per_hr") or 0) * max(0.0, float(end) - float(r.get("created_ts") or end)) / 3600


def run_pods(run: str, now: float | None = None, rows: list[dict] | None = None) -> list[dict]:
    """Every pod (ledger row) this run paid for: the gateway pods it created (owner analysis:<run>), the final pod of
    each session tagged with it (or named by an agent's --session), and the pods those sessions created and deleted
    themselves before their final one (CUDA check, broken-host deny list, unreachable ssh: $0.032 in the 2026-10-21
    trial that the run's cost missed). Such a pod is matched by the session's label (the ledger row's name), its kill
    time (max_minutes) and its creation inside the session's start window; when two sessions fit, the one that started
    last wins, and a pod that is another session's final pod is never taken. A runner that lists `pods_created` in its
    state.json is read exactly. -> [{pod_id, sid, step, gpu, cost, open, how}]."""
    from . import estimate
    now = time.time() if now is None else now
    doc = load(run) or {}
    named = {sid: s["id"] for s in doc.get("steps") or [] for sid in s.get("sessions") or []}
    infos = _session_infos(run, extra=named)
    mine = {sid: (i["tag"] or {}).get("step") for sid, i in infos.items() if (i["tag"] or {}).get("run") == run}
    for sid, step in named.items():
        if sid in infos:
            mine.setdefault(sid, step)
    rows = estimate.ledger_rows() if rows is None else rows
    by_id = {r.get("pod_id"): r for r in rows}
    final = {i["pod_id"]: sid for sid, i in infos.items() if i["pod_id"]}
    listed = {p: sid for sid, i in infos.items() for p in i["pods"]}
    for i in infos.values():   # a pod the session deleted itself was created before its final pod (or before it ended)
        fp = by_id.get(i["pod_id"]) if i["pod_id"] else None
        i["until"] = float(fp["created_ts"]) if fp and fp.get("created_ts") else (i["end"] or now)
    t0 = float(doc.get("created") or _sid_epoch(run)) - 60
    out = []
    for r in rows:
        pid, ct = r.get("pod_id"), float(r.get("created_ts") or 0)
        if ct < t0:
            continue
        if str(r.get("owner") or "") == f"analysis:{run}":
            sid, step, how = None, "upload", "gateway"
        else:
            sid, how = listed.get(pid) or final.get(pid), "session pod"
            if sid is None and not r.get("owner"):
                mm = r.get("max_minutes")
                cands = [i for i in infos.values() if i["label"] == r.get("name") and i["start"] - 5 <= ct <= i["until"] + 5
                         and (i["max_minutes"] is None or mm is None or abs(float(i["max_minutes"]) - float(mm)) < 0.05)]
                if cands:
                    sid, how = max(cands, key=lambda i: i["start"])["sid"], "deleted inside its session"
            if sid not in mine:
                continue
            step = mine[sid]
        out.append({"pod_id": pid, "sid": sid, "step": step, "gpu": r.get("gpu"), "how": how,
                    "cost": round(_row_cost(r, now), 4), "open": not r.get("deleted_ts")})
    return out


def cost_detail(run: str, now: float | None = None) -> dict:
    """{total, by_step: {step: $}, pods: run_pods()}: the run's cost is the sum of its pods' ledger rows. A finished run
    whose pods are all closed keeps the figure in analysis.json (cost_final): run_pods reads only the newest 400 session
    dirs (~20 a day), so an old run's cost would otherwise drop as its sessions fall out of that window."""
    doc = load(run) or {}
    fin = doc.get("cost_final")
    final = doc.get("status") in FINAL
    if final and isinstance(fin, dict) and fin.get("pods") is not None:
        return {k: fin[k] for k in ("total", "by_step", "pods") if k in fin}
    pods = run_pods(run, now)
    by: dict[str, float] = {}
    for p in pods:
        k = p["step"] or "?"
        by[k] = round(by.get(k, 0.0) + p["cost"], 4)
    out = {"total": round(sum(p["cost"] for p in pods), 4), "by_step": by, "pods": pods}
    if final and not any(p["open"] for p in pods):
        try:
            with _RunLock(run, timeout=2.0):
                d = load(run)
                if d is not None and d.get("status") in FINAL:
                    # not save(): `updated` must not move (a cancelled run from before `ended` existed ends at its last
                    # write, and the strip counts a run written in the last hour as active); its end is pinned first
                    end = run_end(d)
                    if end and not d.get("ended"):
                        d["ended"] = end
                    d["cost_final"] = {**out, "ts": round(time.time(), 1)}
                    d["cost_so_far_usd"] = out["total"]
                    write_json_atomic(run_dir(run) / "analysis.json", d)
        except Exception:  # noqa: BLE001 - bookkeeping only
            pass
    return out


def refresh_cost(run: str) -> float:
    """cost_so_far_usd = the ledger cost of every pod the run paid for (run_pods). Cheap: small files only."""
    return cost_detail(run)["total"]


def budget_left(run: str) -> float | None:
    """What is left of the run's pod limit (analyze --max-cost / analysis limit), None when it has none. The runner caps
    a session tagged with this run at it (remote_jobs.run_session), the driver starts no pod step past it."""
    doc = load(run) or {}
    lim = doc.get("max_cost_usd")
    if lim is None:
        return None
    return round(float(lim) - refresh_cost(run), 2)


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


def agent_hint(run: str, limited: bool = False) -> str:
    return (f"Steps marked 'needs an agent' have no generic executor yet (take-specific code today). Drive one with "
            f"DANCECAP_ANALYSIS={run} DANCECAP_ANALYSIS_STEP=<step> set on its `dancecap remote run` sessions (they appear "
            f"under the step), then: python -m dancecap.pipeline analysis step {run} <step> done (add --reused when the "
            "step reused earlier results instead of computing them)."
            + (f" The run has a pod limit: give each session --max-cost at most `python -m dancecap.pipeline analysis "
               f"budget {run}`." if limited else ""))


def make_run(path: str, take: str | None = None, opts: dict | None = None, probe: bool = True,
             now: float | None = None, max_cost: float | None = None) -> dict:
    """Classify + probe + plan -> an analysis document (not saved). max_cost: the run's pod limit ($), or None."""
    cls = classify(path)
    take = take or cls["take"]
    if not TAKE_RE.match(take):
        raise ValueError(f"take name {take!r}: lowercase letters, digits and _ only")
    if max_cost is not None and not (math.isfinite(float(max_cost)) and float(max_cost) > 0):
        raise ValueError("--max-cost must be a positive number of dollars")
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
           "agent_hint": agent_hint(run, max_cost is not None),
           "input": {**cls, "probes": {c: {k: p.get(k) for k in ("duration_s", "fps", "display_w", "display_h", "codec",
                                                                    "audio", "size_bytes", "error") if k in p}
                                       for c, p in probes.items()}},
           "facts": f, "options": o, "take_exists": exists, "steps": steps, "plan": totals(steps),
           "cost_so_far_usd": 0.0, "max_cost_usd": round(float(max_cost), 2) if max_cost is not None else None}
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
            msg = f"take {take} exists with these videos: reused"
            if len(want) == 1 and len(have) == 1:   # one-video take made elsewhere (the monitor's wizard): same defaults
                f = doc.get("facts") or {}
                added = write_single_view_defaults(take, f.get("duration_s"), f.get("fps"))
                if added:
                    msg += f"; added {added}"
            return msg
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
    msg = f"created {res['toml_path']} ({res['variant']}-view)"
    if res.get("variant") == "single":
        f = doc.get("facts") or {}
        added = write_single_view_defaults(take, f.get("duration_s"), f.get("fps"))
        if added:
            msg += f"; added {added}"
    return msg + (f"; warnings: {'; '.join(res['warnings'])[:200]}" if res.get("warnings") else "")


SINGLE_EXPORT_DIR = "smplx_rf2_noslip"   # the mono route's post step (stitch, facing repair, foot locking) writes it


def single_view_defaults(text: str, duration_s: float | None, fps: float | None) -> tuple[str, list[str]]:
    """A new one-video take's TOML with what its stages need and pipeline.intake does not write (the 2026-10-21 trial had
    to add them by hand before beats and the export ran): the whole video as the window (one camera IS the reference
    clock; last frame = (round(duration x fps) - 1) / fps), the 30 Hz lattice ([timeline], [ingest] max_fps), the
    window flagged monocular ([stitch] mono) and the refined motion for the export ([export] smplx_dir, floor_yaw_deg
    0). Keys the file already has are kept. -> (text, what was added)."""
    import tomllib
    cfg = tomllib.loads(text)
    blocks, added = [], []
    end = None
    if duration_s and fps and float(duration_s) > 0 and float(fps) > 0:
        n = max(2, int(round(float(duration_s) * float(fps))))
        end = math.floor((n - 1) / float(fps) * 100) / 100
    win = cfg.get("window") or ([0.0, end] if end else None)
    top = []
    if "window" not in cfg and end:
        top.append(f"window = [0.0, {end}]   # analyze: the whole video (one camera = the reference clock)")
        added.append("window")
    if "timeline" not in cfg:
        blocks += ["[timeline]", "anchor = 0.0", "hz = 30.0"]
        added.append("[timeline]")
    if "ingest" not in cfg:
        blocks += ["[ingest]", "max_fps = 30"]
        added.append("[ingest]")
    if "stitch" not in cfg and win:
        blocks += ["[stitch]", f"mono = [[{float(win[0])}, {float(win[1])}]]   # one camera: the whole window is monocular"]
        added.append("[stitch] mono")
    if "export" not in cfg:
        blocks += ["[export]", f'smplx_dir = "{SINGLE_EXPORT_DIR}"   # the refined, foot-locked fit (as john_lula)',
                   "floor_yaw_deg = 0.0"]
        added.append("[export]")
    if not added:
        return text, []
    lines = text.splitlines()
    # top-level keys must come before the first table: replace intake's "# window = ..." note, else insert before the
    # first table header
    at = next((i for i, x in enumerate(lines) if x.startswith("# window =")), None)
    if at is not None:
        lines[at:at + 1] = top + blocks
    else:
        at = next((i for i, x in enumerate(lines) if x.lstrip().startswith("[")), len(lines))
        lines[at:at] = top + blocks + [""]
    out = "\n".join(lines) + ("\n" if text.endswith("\n") else "")
    tomllib.loads(out)            # never write a file that does not parse
    return out, added


def write_single_view_defaults(take: str, duration_s: float | None, fps: float | None) -> str:
    """single_view_defaults() applied to takes/<take>.toml in place. -> what was added ('' when nothing)."""
    p = paths.take_toml(take)
    text = p.read_text(encoding="utf-8")
    out, added = single_view_defaults(text, duration_s, fps)
    if added:
        write_text_atomic(p, out)
    return ", ".join(added)


def _upload(doc: dict, report) -> str:
    """The source videos to the hm-farm volume (src/<take>/), through a CPU gateway pod THIS run creates (never another
    workflow's pod), 4 files at once, a file over 24 MB striped over 6 ssh streams (farm.volume), hash-diffed (a video
    already there is skipped). The gateway is deleted at the end."""
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

SKIP_RC = 3    # pipeline/mono_route's exit code for a step that cannot run here for a stated reason (avatars, film)


def run_cmd_step(run: str, doc: dict, st: dict) -> tuple[str, str]:
    """A cmd step: `python <cmd>` with {take} / {run} filled in, its output to logs/<step>.log -> (done | skipped | failed,
    message). The message is the command's last "RESULT: " line (mono_route prints what the step produced), "SKIP: " with rc
    SKIP_RC marks an optional step skipped, a failure carries the log's last lines."""
    sid = st["id"]
    argv = [sys.executable] + [a.replace("{take}", doc["take"]).replace("{run}", run) for a in st["cmd"]]
    logf = run_dir(run) / "logs" / f"{sid}.log"
    logf.parent.mkdir(parents=True, exist_ok=True)
    start = logf.stat().st_size if logf.exists() else 0
    with open(logf, "a", encoding="utf-8") as fh:
        fh.write(f"\n[analyze {time.strftime('%Y-%m-%d %H:%M:%S')}] {' '.join(argv[1:])}\n")
        fh.flush()
        rc = subprocess.run(argv, cwd=str(paths.root()), stdout=fh, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, creationflags=NO_WINDOW).returncode
    try:
        with open(logf, encoding="utf-8", errors="replace") as fh:
            fh.seek(start)
            lines = fh.read().splitlines()
    except OSError:
        lines = []
    tagged = lambda tag: next((x[len(tag):].strip() for x in reversed(lines) if x.startswith(tag)), None)  # noqa: E731
    where = f"(log {paths.rel_display(logf)})"
    if rc == SKIP_RC and tagged("SKIP:") is not None and st.get("optional"):
        return "skipped", f"{tagged('SKIP:')} {where}"[:600]
    if rc != 0:
        tail = " | ".join(x.strip() for x in lines[-6:] if x.strip())
        return "failed", f"rc={rc}: {tail} {where}"[:600]
    return "done", f"{tagged('RESULT:') or 'rc 0'} {where}"[:600]


def _ready(doc: dict, st: dict) -> bool | None:
    """A pending step: True = its deps are satisfied (it can start or wait for an agent now), None = a dep failed (it is
    blocked), False = not yet."""
    by = {s["id"]: s for s in doc["steps"]}
    deps = [by[d] for d in st["deps"] if d in by]
    if any(d["status"] == "failed" for d in deps):
        return None
    return all(d["status"] in DONE or (d.get("optional") and d["status"] in ("blocked",)) for d in deps)


def _has_work(doc: dict) -> bool:
    """Something a driver would act on now: a pending step whose deps finished (or failed)."""
    return any(st["status"] == "pending" and _ready(doc, st) is not False for st in doc["steps"])


def _unblock(doc: dict) -> list[str]:
    """Steps the driver blocked because a step they need failed go back to pending once none of their deps failed or is
    blocked any more (an agent redid the failed step). -> the ids set back."""
    by = {s["id"]: s for s in doc["steps"]}
    back, changed = [], True
    while changed:
        changed = False
        for st in doc["steps"]:
            if st["status"] == "blocked" and not any(by[d]["status"] in ("failed", "blocked") for d in st["deps"]
                                                     if d in by):
                st.update(status="pending", message=None)
                back.append(st["id"])
                changed = True
    return back


def _final_status(doc: dict) -> str:
    """done / failed / running / waiting, when a driver ends. A step still running (an agent's) keeps the run open: a
    builtin step failing beside it used to end the run as failed (ELAPSED frozen, ETA 0, the heartbeat gone) while the
    agent worked on, and the run read "waiting" while that step ran (2026-10-22 review): it is "running" then. "waiting"
    = nothing runs and the open steps wait for an agent or a pod limit."""
    st_all = [s["status"] for s in doc["steps"]]
    req = [s for s in doc["steps"] if not s.get("optional")]
    if all(s["status"] in DONE for s in req):
        return "done"
    if "running" in st_all:
        return "running"
    if any(s in ("failed", "blocked") for s in st_all) and "waiting_agent" not in st_all:
        return "failed"
    return "waiting"


class Driver:
    """Runs the runnable steps in DAG order, several at once; marks the agent steps as waiting; recomputes and exits when
    nothing more can run (status waiting / done / failed). Re-reads analysis.json every loop, so marks made with
    `analysis step` by an agent are picked up.

    One driver per run: it registers under the run lock and leaves when another live driver holds the run (each agent
    mark spawns one; the trial had 2-3 at once). It decides to end under the same lock, after a last look for work, so a
    mark made while it stops is never lost (the mark then sees no live driver and spawns the next one). Snapshots: the
    run's start (first driver only), each builtin step's end, and the run's final state; a plain stop / resume is not a
    progress point (23 of the trial's 80 snapshots were). The 15-min heartbeat is the run's heartbeat process."""

    def __init__(self, run: str, spec: dict | None = None, allow_pods: bool = False, poll_s: float = 2.0):
        self.run, self.spec, self.allow_pods, self.poll_s = run, spec, allow_pods, poll_s
        self.threads: dict[str, threading.Thread] = {}
        self.results: dict[str, tuple[str, str]] = {}
        self.last_snap = 0.0

    def _register(self) -> bool | None:
        """True = the run's first driver, False = a later one, None = do not run (another live driver, or cancelled)."""
        ct = None
        try:
            import psutil
            ct = psutil.Process(os.getpid()).create_time()
        except Exception:  # noqa: BLE001
            pass
        with _RunLock(self.run):
            doc = load(self.run)
            if doc is None or doc.get("status") == "cancelled":
                return None
            drv = doc.get("driver") or {}
            if driver_alive(doc) and drv.get("pid") != os.getpid():
                return None
            first = not drv
            doc["driver"] = {"pid": os.getpid(), "create_time": ct, "started": time.time()}
            doc["status"] = "running"
            doc.pop("ended", None)
            for s in doc["steps"]:   # a step this driver's executors ran when a previous driver died: run it again
                if s["status"] == "running" and s["executor"] in ("builtin", "cmd"):
                    s.update(status="pending", message="restarted (the previous driver ended while it ran)")
            save(doc)
            return first

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
                    res, msg = run_cmd_step(self.run, doc, st)
                    self.results[sid] = (res, msg)
                    return
                else:
                    raise RuntimeError("no executor")
                self.results[sid] = ("done", msg)
            except Exception as e:  # noqa: BLE001
                self.results[sid] = ("failed", f"{type(e).__name__}: {e}"[:600])
        t = threading.Thread(target=work, daemon=True, name=f"analyze-{sid}")
        self.threads[sid] = t
        t.start()

    def loop(self) -> int:
        first = self._register()
        if first is None:
            drv = (load(self.run) or {}).get("driver") or {}
            print(f"[analyze] {self.run}: another driver runs (pid {drv.get('pid')}) or the run is cancelled: this one "
                  "exits", flush=True)
            return 0
        ensure_heartbeat(self.run)
        if first:
            self._snap("start", "analysis started: the plan")
        while True:
            for sid, (res, msg) in list(self.results.items()):
                self.results.pop(sid)
                self.threads.pop(sid, None)
                update_step(self.run, sid, status=res, ended=time.time(), message=msg)
                event(self.run, res, sid, msg)
                print(f"[analyze] {sid}: {res}: {msg}", flush=True)
                self._snap(f"{sid}_{res}", f"{sid}: {res}")
            doc = load(self.run)
            if doc.get("status") == "cancelled":
                print(f"[analyze] {self.run}: cancelled", flush=True)
                return 0
            changed = False
            for st in doc["steps"]:   # parked by a driver: back to pending once --yes / the pod limit allows it
                why = st.get("parked") if st["status"] == "waiting_agent" else None
                if (why == "yes" and self.allow_pods) or why == "limit" and (
                        (lim := budget_left(self.run)) is None or lim >= step_cost(st, st["p90_min"])):
                    update_step(self.run, st["id"], status="pending", message=None, parked="")
                    event(self.run, "unparked", st["id"], f"{why} now allows it")
                    changed = True
            if changed:
                continue
            for st in doc["steps"]:
                if st["status"] != "pending":
                    continue
                ok = _ready(doc, st)
                if ok is None:
                    update_step(self.run, st["id"], status="blocked", message="a step it needs failed")
                    changed = True
                    continue
                if not ok or st["id"] in self.threads:
                    continue
                runnable = st["executor"] in ("builtin", "cmd")
                if runnable and st["where"] != "local" and not self.allow_pods:
                    pods = ("a CPU gateway pod, about $0.01-0.05" if st["id"] == "upload" else
                            f"{st.get('lanes', 1)} Runpod pod(s), about ${step_cost(st):.2f} (p50)")
                    update_step(self.run, st["id"], status="waiting_agent", parked="yes",
                                message=f"needs --yes (it starts {pods}): `analysis resume {self.run} --yes`")
                    changed = True
                    continue
                left = budget_left(self.run) if st["where"] != "local" else None
                if runnable and left is not None:
                    need = step_cost(st, st["p90_min"])
                    if left < need:
                        update_step(self.run, st["id"], status="waiting_agent", parked="limit",
                                    message=f"the run's pod limit has ${left:.2f} left, this step may cost ${need:.2f} "
                                            f"(p90): raise it with `analysis limit {self.run} USD`")
                        event(self.run, "waiting_agent", st["id"], "pod limit reached")
                        changed = True
                        continue
                if runnable:
                    self._start(st)
                    changed = True
                else:
                    lim = ("" if left is None else f"; the run's pod limit has ${left:.2f} left (each session's "
                           f"--max-cost is capped at it)" if left > 0 else "; the run's pod limit is used up: "
                           f"`analysis limit {self.run} USD` before its sessions can start")
                    update_step(self.run, st["id"], status="waiting_agent",
                                message=f"needs an agent: {st.get('blocker') or 'take-specific code today'} "
                                        f"({st['executor']} step){lim}")
                    event(self.run, "waiting_agent", st["id"], st["label"])
                    changed = True
            if changed:
                continue
            if self.threads:
                time.sleep(self.poll_s)
                continue
            with _RunLock(self.run):   # the decision to stop and the agents' marks are serialised by the run lock
                d2 = load(self.run)
                if d2.get("status") != "cancelled" and _has_work(d2):
                    continue           # a mark landed since the scan above: go round again
                prev = d2.get("driver_end_status")      # what the previous driver ended with (marks set "running")
                status = "cancelled" if d2.get("status") == "cancelled" else _final_status(d2)
                now = time.time()
                d2["status"] = status
                d2["driver"] = {**(d2.get("driver") or {}), "ended": now}
                d2["driver_end_status"] = status
                if status in FINAL:
                    d2["ended"] = d2.get("ended") or now
                save(d2)
            if status in FINAL:
                cost_detail(self.run)   # kept in analysis.json once every pod is closed (cost_final)
            if status in FINAL or status != prev:      # one event per change, not one per agent mark
                event(self.run, "driver_end", None, status)
            if status in ("done", "failed"):
                self._snap("final" if status == "done" else "failed", f"analysis {status}", final=True)
            print(f"[analyze] {self.run}: {status}" + ("; the waiting steps need an agent (the monitor lists them)"
                                                       if status == "waiting" else ""), flush=True)
            return 0 if status in ("done", "running", "waiting", "cancelled") else 1


# ------------------------------------------------------------------------------------------------ heartbeat

def _hb_path(run: str) -> Path:
    return run_dir(run) / "heartbeat.json"


def heartbeat_alive(run: str) -> bool:
    hb = read_json(_hb_path(run), None) or {}
    return bool(hb.get("pid") and not hb.get("ended") and pid_alive(hb["pid"], hb.get("create_time")))


HB_SPAWN_GRACE_S = 60.0   # a heartbeat spawned this recently that has not registered yet still counts as running


def _hb_spawning(hb: dict, now: float) -> bool:
    sp = hb.get("spawn") or {}
    return bool(sp.get("pid") and now - float(sp.get("ts") or 0) < HB_SPAWN_GRACE_S and pid_alive(sp["pid"]))


def ensure_heartbeat(run: str, spawn=None) -> bool:
    """Start the run's heartbeat process (detached, no window) unless one lives or is starting. Never in tests or on a temp
    root (unless `spawn` is given: the tests' stand-in for Popen), never raises. The driver calls it when it starts, every
    agent mark calls it. Decided under the run lock with a spawn marker in heartbeat.json: two marks 2 s apart both saw no
    heartbeat yet (the first child had not registered) and started two (2026-10-21); the second only printed 'another
    heartbeat runs', into the same log at the same offset. The log is opened with a true append handle (util.append_fd)."""
    try:
        if spawn is None and (not paths.is_real_root() or os.environ.get("PYTEST_CURRENT_TEST")
                              or os.environ.get("DANCECAP_SNAPSHOTS") == "0"):
            return False
        with _RunLock(run):
            hb = read_json(_hb_path(run), None) or {}
            now = time.time()
            if heartbeat_alive(run) or _hb_spawning(hb, now):
                return False
            argv = [sys.executable, "-m", "dancecap.pipeline", "analysis", "heartbeat", run]
            if spawn is not None:
                pid = spawn(argv)
            else:
                fd = append_fd(run_dir(run) / "logs" / "heartbeat.log")
                try:
                    pid = subprocess.Popen(argv, cwd=str(paths.root()), stdin=subprocess.DEVNULL, stdout=fd,
                                           stderr=subprocess.STDOUT, creationflags=NO_WINDOW | DETACHED,
                                           close_fds=True).pid
                finally:
                    os.close(fd)
            write_json_atomic(_hb_path(run), {**({} if hb.get("pid") and not hb.get("ended") else hb),
                                              "spawn": {"pid": pid, "ts": now}})
        return True
    except Exception:  # noqa: BLE001
        return False


def _last_snapshot_ts(run: str) -> float:
    """The newest snapshot asked for or taken in the run's folder (by the driver, a mark, a session tagged with it)."""
    try:
        from . import snapshots
        d = snapshots.snap_dir(run)
    except Exception:  # noqa: BLE001
        return 0.0
    ts = [float((read_json(d / "requested.json", None) or {}).get("ts") or 0)]
    shots = (read_json(d / "index.json", None) or {}).get("shots") or []
    if shots:
        ts.append(float(shots[-1].get("requested") or shots[-1].get("ts") or 0))
    return max(ts)


RUNNING_BUSY_X = 3.0      # a running step keeps the heartbeat busy up to this many x its plan p90 after its start


def _moving(run: str, doc: dict, now: float) -> tuple[bool, float]:
    """(busy, last activity): busy = a step runs (for at most RUNNING_BUSY_X x its p90: an agent that marked `start` and
    went away must not keep an Edge render every 15 min going for ever) or a session tagged with the run is open and
    wrote in the last 15 min; last activity = the newest of analysis.json, events.jsonl and those sessions' state.json."""
    rd = run_dir(run)
    last = max(float(doc.get("updated") or 0), mtime(rd / "events.jsonl") or 0)
    busy = any(s.get("status") == "running"
               and now < float(s.get("started") or now) + RUNNING_BUSY_X * float(s.get("p90_min") or 60) * 60
               for s in doc.get("steps") or [])
    d = paths.sessions_dir()
    for s in sessions_of(run):
        m = mtime(d / s["sid"] / "state.json") or 0
        last = max(last, m)
        if not (d / s["sid"] / "result.json").exists() and now - m < HEARTBEAT_S:
            busy = True
    return busy, last


def heartbeat_loop(run: str, poll_s: float = 60.0, every_s: float = HEARTBEAT_S, idle_exit_s: float = 6 * 3600,
                   max_loops: int | None = None, clock=time.time, sleep=time.sleep, request=None) -> str:
    """The run's heartbeat (one process per unfinished run): a snapshot every `every_s` while anything of the run moves,
    whether or not a driver runs (the driver exits as soon as every remaining step waits for an agent: the trial had
    18-min gaps, the whole fit job among them). It also keeps the monitor leased. Ends when the run is finished or
    cancelled, or after `idle_exit_s` with nothing moving (the next mark or driver starts it again). -> why it ended."""
    if request is None:
        from . import snapshots

        def request(r, ev, label):
            return snapshots.request(r, ev, label)
    ct = None
    try:
        import psutil
        ct = psutil.Process(os.getpid()).create_time()
    except Exception:  # noqa: BLE001
        pass
    with _RunLock(run):
        hb = read_json(_hb_path(run), None) or {}
        if heartbeat_alive(run) and hb.get("pid") != os.getpid():
            return "another heartbeat runs"
        write_json_atomic(_hb_path(run), {"pid": os.getpid(), "create_time": ct, "started": clock()})
    why, n, leased = "stopped", 0, 0.0
    try:
        while max_loops is None or n < max_loops:
            n += 1
            doc = load(run)
            if doc is None:
                why = "the run is gone"
                break
            if doc.get("status") in FINAL:
                why = f"the run is {doc['status']}"
                break
            now = clock()
            busy, last = _moving(run, doc, now)
            if not busy and now - last > idle_exit_s:
                why = f"nothing moved for {idle_exit_s / 3600:.0f} h"
                break
            if now - leased > 300:
                try:
                    from . import monitor
                    monitor.ensure_running(f"analysis heartbeat {run}")
                except Exception:  # noqa: BLE001
                    pass
                leased = now
            if now - _last_snapshot_ts(run) >= every_s and (busy or now - last < every_s):
                request(run, "heartbeat", "progress (every 15 min)")
            sleep(poll_s)
    finally:
        with _RunLock(run):
            hb = read_json(_hb_path(run), None) or {}
            if hb.get("pid") == os.getpid():
                write_json_atomic(_hb_path(run), {**hb, "ended": clock(), "why": why})
    return why


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
    print(f"  every pod step includes its start-up: a runner session {SESSION_OVERHEAD_MIN['p50']:g} min p50 / "
          f"{SESSION_OVERHEAD_MIN['p90']:g} min p90 (bundle, pod start, upload, bootstrap, installs, download; live sessions "
          f"2026-10-22), a farm job {FARM_OVERHEAD_MIN['p50']:g}-{FARM_OVERHEAD_MIN['p90']:g} min")
    if doc.get("max_cost_usd") is not None:
        lim = float(doc["max_cost_usd"])
        print(f"  run pod limit: ${lim:.2f}" + (f" (below the p90 ${t['cost_usd']['p90']:.2f}: the run may stop at it; "
                                                 "raise it with `analysis limit`)" if lim < t["cost_usd"]["p90"] else ""))
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
    ap.add_argument("--max-cost", type=float, metavar="USD",
                    help="the run's pod limit: no pod step starts past it and every session tagged with the run is "
                         "capped at what is left (analysis budget RUN); the global cap still applies")
    a = ap.parse_args(argv)
    opts = {"room_known": a.room_known, "realism": not a.no_realism, "speech": a.speech, "film": not a.no_film,
            "upload": not a.no_upload}
    try:
        doc = make_run(a.input, a.take, opts, probe=not a.no_probe, max_cost=a.max_cost)
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
    monitor.ensure_running(f"analyze {doc['run']}", route=f"#/analysis/{doc['run']}")
    print(f"[analyze] run {doc['run']}: the monitor shows it (python -m dancecap.pipeline monitor --launch); "
          f"snapshots in work/pipeline/snapshots/{doc['run']}/", flush=True)
    if a.detach:
        return spawn_driver(doc["run"])
    return Driver(doc["run"], spec=doc["spec"], allow_pods=bool(a.yes)).loop()


def spawn_driver(run: str) -> int:
    """Start the run's driver in the background unless one runs or is starting. Decided under the run lock: a live
    driver re-reads analysis.json before it ends (it sees the caller's mark); a driver spawned less than a minute ago that
    has not registered yet will read it when it does. Each extra spawn would only exit again."""
    logf = run_dir(run) / "logs" / "driver.log"
    logf.parent.mkdir(parents=True, exist_ok=True)
    with _RunLock(run):
        doc = load(run) or {}
        if driver_alive(doc):
            return 0
        sp = doc.get("driver_spawn") or {}
        if (sp.get("pid") and time.time() - float(sp.get("ts") or 0) < 60 and pid_alive(sp["pid"])
                and float((doc.get("driver") or {}).get("started") or 0) < float(sp.get("ts") or 0)):
            return 0
        fd = append_fd(logf)   # several drivers of one run append to this log over its life (util.append_fd)
        try:
            p = subprocess.Popen([sys.executable, "-m", "dancecap.pipeline", "analysis", "resume", run],
                                 cwd=str(paths.root()), stdin=subprocess.DEVNULL, stdout=fd, stderr=subprocess.STDOUT,
                                 creationflags=NO_WINDOW | DETACHED, close_fds=True)
        finally:
            os.close(fd)
        if doc:
            doc["driver_spawn"] = {"pid": p.pid, "ts": time.time()}
            save(doc)
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
    p.add_argument("--yes", action="store_true", help="allow the run to start pods from now on (a step parked for "
                                                       "'needs --yes' runs)")
    p = sub.add_parser("cancel", help="mark a run cancelled (it leaves the dashboard's active list; nothing is deleted)")
    p.add_argument("run")
    p.add_argument("--note", default="")
    p = sub.add_parser("snap", help="take a dashboard snapshot now")
    p.add_argument("run")
    p.add_argument("--event", default="manual")
    p = sub.add_parser("step", help="mark a step (agents driving a take-specific step)")
    p.add_argument("run")
    p.add_argument("step")
    p.add_argument("action", choices=["start", "done", "fail", "skip", "retry"],
                   help="retry: a failed step back to pending (the driver reruns a builtin one)")
    p.add_argument("--note", default="")
    p.add_argument("--session", help="a remote-runner session id that belongs to this step")
    p.add_argument("--reused", action="store_true", help="with done: the step reused earlier results instead of "
                                                          "computing them (no pod cost; the dashboard draws it apart)")
    p = sub.add_parser("budget", help="print what is left of the run's pod limit ($), or 'none' (for --max-cost)")
    p.add_argument("run")
    p = sub.add_parser("limit", help="set the run's pod limit in $ ('none' removes it)")
    p.add_argument("run")
    p.add_argument("usd")
    p = sub.add_parser("heartbeat", help="(started by the driver) a snapshot every 15 min while the run moves")
    p.add_argument("run")
    a = ap.parse_args(argv)
    if a.cmd == "list":
        for d in list_runs():
            lv = live({**d, "cost_so_far_usd": refresh_cost(d["run"])})   # the ledger now, not the last mark's figure
            print(f"{d['run']:40s} {d.get('status', '?'):9s} {lv['done']}/{lv['total']} done  ETA "
                  f"{fmt_min(lv['eta_s']['p50'] / 60)}  ${lv['cost_usd']['so_far']:.2f} -> ${lv['cost_usd']['p50']:.2f}")
        return 0
    if not RUN_RE.match(a.run) or load(a.run) is None:
        print(f"[analysis] no run {a.run}", file=sys.stderr)
        return 2
    if a.cmd == "show":
        d = load(a.run)
        det = cost_detail(a.run)
        d.update(cost_so_far_usd=det["total"], cost_by_step=det["by_step"])
        print(json.dumps({**d, "live": live(d), "sessions": sessions_of(a.run), "pods": det["pods"]}, indent=1,
                         default=str))
        return 0
    if a.cmd == "budget":
        left = budget_left(a.run)
        print("none" if left is None else f"{max(0.0, left):.2f}")
        return 0
    if a.cmd == "limit":
        try:
            lim = None if a.usd.lower() == "none" else float(a.usd)
            if lim is not None and not (math.isfinite(lim) and lim > 0):
                raise ValueError
        except ValueError:
            print("[analysis] limit: a positive number of dollars or 'none'", file=sys.stderr)
            return 2
        with _RunLock(a.run):
            d = load(a.run)
            d["max_cost_usd"] = round(lim, 2) if lim is not None else None
            d["agent_hint"] = agent_hint(a.run, lim is not None)
            save(d)
            parked = [s["id"] for s in d["steps"] if s["status"] == "waiting_agent" and s.get("parked") == "limit"]
        txt = "none" if lim is None else f"${lim:.2f}"
        event(a.run, "limit", None, f"pod limit {txt}")
        print(f"[analysis] {a.run}: pod limit {txt}")
        if parked and d.get("status") not in FINAL and paths.is_real_root() and not os.environ.get("PYTEST_CURRENT_TEST"):
            spawn_driver(a.run)   # the driver un-parks what the new limit allows (never a real process from a test)
        return 0
    if a.cmd == "heartbeat":
        why = heartbeat_loop(a.run)
        print(f"[analysis] {time.strftime('%H:%M:%S')} {a.run}: heartbeat ends ({why})", flush=True)
        return 0
    if a.cmd == "cancel":
        with _RunLock(a.run):
            d = load(a.run)
            d["status"] = "cancelled"
            d["cancel_note"] = a.note[:300]
            d["ended"] = d.get("ended") or time.time()   # a finished run keeps the end it had (ELAPSED does not move)
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
        monitor.ensure_running(f"analysis {a.run}", route=f"#/analysis/{a.run}")
        d = load(a.run)
        if d.get("status") == "done":   # re-registering would move its end time
            print(f"[analysis] {a.run}: done; nothing to resume")
            return 0
        if a.yes and not d.get("allow_pods"):
            with _RunLock(a.run):
                d = load(a.run)
                d["allow_pods"] = True
                save(d)
        if driver_alive(d) and (d.get("driver") or {}).get("pid") != os.getpid():
            print(f"[analysis] a driver already runs (pid {d['driver']['pid']})")
            return 0
        return Driver(a.run, spec=d.get("spec"), allow_pods=bool(d.get("allow_pods"))).loop()
    if not STEP_RE.match(a.step):
        return 2
    if a.reused and a.action != "done":
        print("[analysis] --reused goes with done", file=sys.stderr)
        return 2
    if a.action == "retry":
        # a failed step back to pending: the driver runs a builtin step again (live check 20261021-202809: the upload's
        # gateway create failed on stock, every step behind it was blocked, the run ended failed, and no mark could
        # restart a builtin step) and parks an agent step for its agent. The blocked steps behind it unblock.
        with _RunLock(a.run):
            d = load(a.run)
            s = next((x for x in d["steps"] if x["id"] == a.step), None)
            if s is None or s["status"] != "failed" or d.get("status") == "cancelled":
                print(f"[analysis] retry: {a.step} is {s and s['status']} (run {d.get('status')}): only a failed step "
                      "of a run that is not cancelled (the blocked steps behind it follow it)", file=sys.stderr)
                return 2
            for k in ("started", "ended", "progress", "parked"):
                s.pop(k, None)
            s.update(status="pending", message=a.note[:300] or None, reused=False)
            _unblock(d)
            if d.get("status") in ("failed", "waiting", "done"):
                d["status"] = "running"
                d.pop("ended", None)
                d.pop("cost_final", None)
            save(d)
        event(a.run, "retry", a.step, a.note, by="agent")
        print(f"[analysis] {a.run} {a.step}: pending again (retry)")
        ensure_heartbeat(a.run)
        spawn_driver(a.run)
        return 0
    now = time.time()
    st = {"start": dict(status="running", started=now, reused=False), "done": dict(status="done", ended=now),
          "fail": dict(status="failed", ended=now, reused=False),
          "skip": dict(status="skipped", ended=now, reused=False)}[a.action]
    if a.action == "done":
        st["reused"] = bool(a.reused)
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
    det = cost_detail(a.run)
    with _RunLock(a.run):
        d = load(a.run)
        d["cost_so_far_usd"] = det["total"]
        d["cost_by_step"] = det["by_step"]
        # an agent is at work: not finished (ELAPSED runs again). A done run reopens only for a step that STARTS (a late
        # done / skip of an optional step must not move its end); a cancelled run stays cancelled
        if (d.get("status") in ("waiting", "planned", "failed") and a.action in ("start", "done", "skip")
                or d.get("status") == "done" and a.action == "start"):
            d["status"] = "running"
            d.pop("ended", None)
            d.pop("cost_final", None)
        if a.action in ("start", "done", "skip"):
            _unblock(d)                  # a step that failed is being redone: what waited on it may run again
        save(d)
    reused = a.action == "done" and a.reused
    event(a.run, a.action, a.step, a.note, by="agent", **({"reused": True} if reused else {}))
    from . import snapshots
    snapshots.request(a.run, f"{a.step}_{'reused' if reused else a.action}",
                      f"{a.step}: {'reused (not computed)' if reused else a.action} (agent)")
    print(f"[analysis] {a.run} {a.step}: {cur['status']}{' (reused)' if reused else ''}")
    ensure_heartbeat(a.run)
    if a.action in ("done", "skip", "fail"):
        spawn_driver(a.run)   # the next steps may be runnable now (a no-op when a driver runs: it sees this mark)
    return 0
