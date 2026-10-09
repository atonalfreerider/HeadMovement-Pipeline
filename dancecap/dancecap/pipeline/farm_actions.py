"""Pipeline Monitor, farm view actions (docs/FARM.md section 7): Fill gaps, Pull now, Restart watcher, Stop pods, coverage
refresh, log tails, map export. The client never sends an argv: it names an action and a dance; every command line is built
here from the farm CLI (`python -m dancecap.farm ...`, `python -m dancecap.farm.pods ...`).

Rules (user, 2026-10-08):
  * a paid pod is only ever started after a PLAN (no side effects: gaps -> 30-s chunks -> job count, pods, price, time
    estimate, worst case, budget gate) that the user confirmed (`plan_hash` + `approve.max_cost_usd`); the plan expires;
  * the segments that a live `dancecap.fit_full` session is fitting RIGHT NOW (a-d) are never queued or launched here: Fill gaps
    excludes them and marks their chunks `running` (owner fit_full) in state.json before it queues anything;
  * Stop pods deletes only pods recorded in this dance's farm state (never a fit_full pod, never another agent's pod);
  * long commands run DETACHED (their own process, a log file, a record under work/monitor/farm_actions/): closing the window
    or the monitor never stops them, and the monitor shows their state.

`python -m dancecap.pipeline.farm_actions _exec <record.json>` is the detached wrapper that runs a record's steps in order.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from . import fit_sessions, paths
from .util import POD_ID_RE, TAKE_RE, mtime, pid_alive, read_json, read_text, redact, write_json_atomic

NO_WINDOW = 0x08000000
NEW_GROUP = 0x00000200
PLAN_TTL_S = 15 * 60
MAX_PODS = 6
DEFAULT_PODS = 4
START_OVERHEAD_MIN = 2.0           # pod up -> first job step measured at 28 s on the volume; keep a margin for pull + claim
FINAL_CPU_MIN = 12.0               # the stitch + analyses CPU job
GPU_SLOWDOWN = 0.45                # same as `farm launch --slowdown`: the RTX 4090 measured by the drill (0.41 of the A100 model), rounded up
FALLBACK_GPU = {"gpu": "NVIDIA GeForce RTX 4090", "price": 0.69, "stock": "unknown"}
MIN_PER_WINDOW_S = 1.35            # front + fit + refine on one A100 (S1: 51-58 min for a 38-43 s window)
LOG_RE = re.compile(r"^[A-Za-z0-9_.-]{1,80}\.log$")
ACTION_ID_RE = re.compile(r"^a-\d{8}-\d{6}-[0-9a-f]{4}$")
COVERAGE_MIN_INTERVAL_S = 120.0
_last_coverage_try: dict[str, float] = {}


class ActionError(Exception):
    def __init__(self, status: int, code: str, message: str, details: dict | None = None):
        super().__init__(message)
        self.status, self.code, self.message, self.details = status, code, message, details or {}


# ------------------------------------------------------------------------------------------------ records

def actions_dir() -> Path:
    return paths.monitor_dir() / "farm_actions"


def farm_dir(dance: str) -> Path:
    return paths.work() / dance / "farm"


def _py() -> str:
    return sys.executable


def _env() -> dict:
    e = dict(os.environ)
    e.update({"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    return e


def _farm(dance: str, *args: str) -> list[str]:
    return [_py(), "-m", "dancecap.farm", "--dance", dance, *args]


def _check_dance(dance: str) -> None:
    if not TAKE_RE.match(dance or "") or not (paths.work() / dance).is_dir():
        raise ActionError(404, "not_found", f"no dance {dance!r}")


def _rec_path(aid: str) -> Path:
    return actions_dir() / f"{aid}.json"


def new_id() -> str:
    return f"a-{time.strftime('%Y%m%d-%H%M%S')}-{os.urandom(2).hex()}"


def _state_of(rec: dict) -> str:
    if rec.get("ended"):
        return "ok" if rec.get("rc") == 0 else "failed"
    if rec.get("pid") and pid_alive(rec["pid"], rec.get("pid_ct")):
        return "running"
    return "lost" if rec.get("started") else "starting"


def describe(rec: dict) -> dict:
    return {"id": rec["id"], "label": rec.get("label"), "dance": rec.get("dance"), "state": _state_of(rec),
            "created": rec.get("created"), "started": rec.get("started"), "ended": rec.get("ended"), "rc": rec.get("rc"),
            "step": rec.get("step_idx"), "steps": [s.get("label") for s in rec.get("steps") or []],
            "log": rec.get("log_name"), "error": rec.get("error"), "meta": rec.get("meta") or {}}


def recent(dance: str | None = None, limit: int = 6) -> list[dict]:
    d = actions_dir()
    out = []
    try:
        files = sorted(d.glob("a-*.json"), key=lambda p: p.name, reverse=True)
    except OSError:
        return out
    for p in files[:60]:
        rec = read_json(p, None)
        if isinstance(rec, dict) and rec.get("id") and (not dance or rec.get("dance") == dance):
            out.append(describe(rec))
        if len(out) >= limit:
            break
    return out


def running(dance: str, label: str | None = None) -> dict | None:
    for r in recent(dance, 12):
        if r["state"] in ("running", "starting") and (label is None or r["label"] == label):
            return r
    return None


def start(label: str, dance: str, steps: list[dict], meta: dict | None = None) -> dict:
    """Write the record and start the detached wrapper. steps: [{"label", "argv": [...], "optional": bool}]."""
    _check_dance(dance)
    aid = new_id()
    log_dir = farm_dir(dance) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    rec = {"id": aid, "label": label, "dance": dance, "steps": steps, "created": time.time(), "started": None, "ended": None,
           "rc": None, "pid": None, "step_idx": None, "log_name": f"{aid}.log", "log": str(log_dir / f"{aid}.log"),
           "meta": meta or {}}
    write_json_atomic(_rec_path(aid), rec)
    flags = (NO_WINDOW | NEW_GROUP) if os.name == "nt" else 0
    proc = subprocess.Popen([_py(), "-m", "dancecap.pipeline.farm_actions", "_exec", str(_rec_path(aid))],
                            cwd=str(paths.root()), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, creationflags=flags, close_fds=True, env=_env())
    rec["pid"], rec["started"] = proc.pid, time.time()
    try:
        import psutil
        rec["pid_ct"] = psutil.Process(proc.pid).create_time()
    except Exception:  # noqa: BLE001
        pass
    # the wrapper may already have rewritten the record (steps done): merge only the fields it does not own
    cur = read_json(_rec_path(aid), rec) or rec
    cur.update({"pid": rec["pid"], "pid_ct": rec.get("pid_ct")})
    cur["started"] = cur.get("started") or rec["started"]
    write_json_atomic(_rec_path(aid), cur)
    return describe(cur)


def _exec(rec_path: Path) -> int:
    """The detached wrapper: run the steps in order, one log, stop at the first failing non-optional step."""
    rec = read_json(rec_path, None)
    if not isinstance(rec, dict):
        return 2
    log = Path(rec["log"])
    log.parent.mkdir(parents=True, exist_ok=True)
    rec["started"] = rec.get("started") or time.time()
    rc_all = 0
    with open(log, "a", encoding="utf-8", errors="replace", buffering=1) as fh:
        fh.write(f"[{time.strftime('%H:%M:%S')}] {rec['label']} ({rec['dance']}) started\n")
        for i, st in enumerate(rec.get("steps") or []):
            rec["step_idx"] = i
            write_json_atomic(rec_path, rec)
            fh.write(f"\n[{time.strftime('%H:%M:%S')}] step {i + 1}/{len(rec['steps'])}: {st.get('label')}\n$ "
                     f"{' '.join(redact(str(a)) for a in st['argv'])}\n")
            try:
                p = subprocess.run(st["argv"], cwd=str(paths.root()), stdin=subprocess.DEVNULL, stdout=fh,
                                   stderr=subprocess.STDOUT, env=_env(), timeout=float(st.get("timeout_s") or 3 * 3600),
                                   creationflags=NO_WINDOW if os.name == "nt" else 0)
                rc = p.returncode
            except subprocess.TimeoutExpired:
                rc = 124
                fh.write("[timeout]\n")
            except OSError as e:
                rc = 127
                fh.write(f"[cannot start: {e}]\n")
            fh.write(f"[{time.strftime('%H:%M:%S')}] step {i + 1} rc={rc}\n")
            if rc != 0 and not st.get("optional"):
                rc_all = rc
                rec["error"] = f"step {i + 1} ({st.get('label')}) failed with rc {rc}"
                break
        rec["rc"], rec["ended"] = rc_all, time.time()
        write_json_atomic(rec_path, rec)
        fh.write(f"[{time.strftime('%H:%M:%S')}] done rc={rc_all}\n")
    return rc_all


# ------------------------------------------------------------------------------------------------ what the farm CLI can do

_cmd_cache: list = [None, {}]      # (signature of the farm modules' mtimes, commands): the sources are parsed again only when one changed


def _parser_names(src: str) -> tuple[list[str], bool]:
    """(names given to .add_parser(...), whether the module defines register()) from the source, without importing it (the
    farm modules are heavy and may be mid-edit by their owners; a docstring example is not a command)."""
    import ast
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return [], False
    names, has_register = [], False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and (node.name == "register" or node.name.startswith("register_")):
            has_register = True        # launch.py's register_commands(sub) is called by watch.register(sub)
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_parser"
              and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
            names.append(node.args[0].value)
    return names, has_register


def available_commands() -> dict:
    """Sub-commands of `python -m dancecap.farm` (core cli.py + plug-in modules that define register(sub)) and of
    `python -m dancecap.farm.pods` (keys 'pods:<name>')."""
    farm = Path(__file__).resolve().parents[1] / "farm"
    files = sorted(farm.glob("*.py"))
    sig = tuple((p.name, mtime(p)) for p in files)
    if _cmd_cache[0] == sig:
        return dict(_cmd_cache[1])
    names: dict[str, str] = {}
    for p in files:
        found, has_register = _parser_names(read_text(p, "") or "")
        for n in found:
            if p.name == "cli.py":
                names.setdefault(n, "farm")
            elif p.name == "pods.py":
                names.setdefault("pods:" + n, "pods")
                if has_register:
                    names.setdefault(n, "farm")
            elif has_register:
                names.setdefault(n, "farm")
    _cmd_cache[0], _cmd_cache[1] = sig, dict(names)
    return names


def have_cmd(name: str) -> bool:
    return name in available_commands()


# ------------------------------------------------------------------------------------------------ coverage refresh

def refresh_coverage(dance: str, force: bool = False) -> dict | None:
    """Recompute coverage.json (core: ~4 s, one thread, normal priority) as a detached action; never two at once and never more
    often than every COVERAGE_MIN_INTERVAL_S unless forced."""
    _check_dance(dance)
    now = time.time()
    if running(dance, "coverage"):
        return None
    if not force and now - _last_coverage_try.get(dance, 0) < COVERAGE_MIN_INTERVAL_S:
        return None
    _last_coverage_try[dance] = now
    return start("coverage", dance, [{"label": "farm coverage", "argv": _farm(dance, "--json", "coverage"), "timeout_s": 600}])


# ------------------------------------------------------------------------------------------------ Fill gaps

def _run_json(argv: list[str], timeout: float) -> tuple[int, dict | list | None, str]:
    try:
        p = subprocess.run(argv, cwd=str(paths.root()), capture_output=True, text=True, encoding="utf-8", errors="replace",
                           timeout=timeout, env=_env(), stdin=subprocess.DEVNULL,
                           creationflags=NO_WINDOW if os.name == "nt" else 0)
    except subprocess.TimeoutExpired:
        return 124, None, "timeout"
    except OSError as e:
        return 127, None, str(e)
    txt = (p.stdout or "").strip()
    out = None
    try:
        out = json.loads(txt)
    except ValueError:
        for line in reversed(txt.splitlines()):
            line = line.strip()
            if line[:1] in "{[":
                try:
                    out = json.loads(line)
                    break
                except ValueError:
                    continue
    return p.returncode, out, redact(((p.stderr or "") + (txt if out is None else ""))[-600:])


def _gpu_offers() -> tuple[list[dict], str | None]:
    """`python -m dancecap.farm.pods offers`: GPUs with stock in the volume's DC now, priority order. Falls back to a stated price."""
    rc, out, err = _run_json([_py(), "-m", "dancecap.farm.pods", "offers"], timeout=90)
    if rc == 0 and isinstance(out, list) and out:
        return out, None
    if rc == 0 and isinstance(out, list):
        return [], "no GPU with stock in the volume's data center right now; the price below is an assumption"
    return [], f"GPU offers unavailable ({err[:120]}); the price below is an assumption"


def lpt_makespan(minutes: list[float], pods: int) -> tuple[float, list[list[float]]]:
    """Longest-processing-time-first assignment of chunk durations to `pods` pods: (makespan, loads per pod)."""
    pods = max(1, pods)
    loads: list[list[float]] = [[] for _ in range(pods)]
    for m in sorted(minutes, reverse=True):
        min(loads, key=lambda l: sum(l)).append(m)
    return (max((sum(l) for l in loads), default=0.0), loads)


def estimate_plan(chunks: list[dict], *, pods: int, price_hr: float, final_cpu: bool, price_note: str | None = None) -> dict:
    """Cost / time estimate of a set of chunk jobs. chunks: [{id, est_min, core_s}]. Rates are the S1 / 38 s reference runs on
    one A100 (front + fit + refine); a 4090 is not measured yet, so the numbers are labelled estimates."""
    mins = [float(c["est_min"]) for c in chunks]
    n = min(max(1, pods), len(mins))                      # 0 GPU pods when only the CPU stitch job is left
    make, loads = lpt_makespan(mins, n) if n else (0.0, [])
    busy = sum(mins)
    exp_cost = busy * price_hr / 60.0 * 1.08 + (0.07 if final_cpu else 0.0)    # 8 % for pod start / idle tail
    minutes = math.ceil(make * 1.25 + 10) if n else 0                           # the pod's kill-switch minutes
    hard = math.ceil(minutes * 1.5)
    worst = (n * price_hr * (hard + 8) / 60.0 if n else 0.0) + (0.5 if final_cpu else 0.0)
    wall = START_OVERHEAD_MIN + make + (FINAL_CPU_MIN if final_cpu else 0.0)
    return {"pods": n, "price_hr": price_hr, "price_note": price_note, "busy_min": round(busy, 1), "makespan_min": round(make, 1),
            "wall_min": round(wall, 1), "expected_cost_usd": round(exp_cost, 2), "worst_cost_usd": round(worst, 2),
            "pod_minutes": minutes, "pod_hard_minutes": hard,
            "assignment": [{"pod": i + 1, "chunks": [], "min": round(sum(l), 1)} for i, l in enumerate(loads)]}


def _demo_core_plan(dance: str) -> dict:
    """Demo dances have no take TOML for core's planner: plan 30-s chunks over the runs of `none` cells, like core's rule."""
    from . import farm_view
    cov = farm_view.read_coverage(dance) or {}
    cells = farm_view.compact_cells(cov)
    runs, cur = [], None
    for c in cells:
        if c["s"] == "none":
            if cur and c["i"] == cur[1] + 1:
                cur[1] = c["i"]
            else:
                cur = [c["i"], c["i"]]
                runs.append(cur)
    chunks = []
    for a, b in runs:
        n = max(1, round((b - a + 1) / 30))
        length = (b - a + 1) / n
        for j in range(n):
            lo = a + round(j * length)
            hi = a + round((j + 1) * length) - 1
            chunks.append({"id": f"g{lo * 30}", "kind": "full", "core_s": hi - lo + 1, "core_k0": lo * 30, "core_k1": hi * 30 + 29,
                           "T": (hi - lo + 1 + 8) * 30, "estimate": {"total_min": round((hi - lo + 9) * MIN_PER_WINDOW_S, 1)}})
    return {"report": {"gaps": [], "gap_s": []}, "chunks": chunks, "s1": [], "jobs": []}


def fill_plan(dance: str, pods: int | None = None, demo: bool = False) -> dict:
    """Step 1 of Fill gaps: NO side effects. Core's planner in --dry-run (gaps -> 30-s chunks), the live fit_full exclusions,
    the GPU price, the estimate and the budget gate. Returns the plan the user confirms (and stores it under its hash)."""
    from . import farm_view
    _check_dance(dance)
    state = read_json(farm_dir(dance) / "state.json", {}) or {}
    is_demo = bool(state.get(farm_view.DEMO_FLAG))
    if is_demo:
        core = _demo_core_plan(dance)
        rc, err = 0, ""
    else:
        rc, core, err = _run_json(_farm(dance, "--json", "plan", "--dry-run"), timeout=300)
        if rc != 0 or not isinstance(core, dict):
            raise ActionError(502, "plan_failed", f"farm plan --dry-run failed (rc {rc}): {err}")
    live = fit_sessions.live_fits(dance) if not is_demo else {"segments": {}}
    live_ids = {s for s, v in live["segments"].items() if v.get("status") in ("running", "queued")}
    st_chunks = state.get("chunks") or {}
    todo, excluded = [], []
    for c in (core.get("chunks") or []):
        todo.append(c)
    for c in (core.get("s1") or []):
        sid = c.get("id")
        stat = (st_chunks.get(sid) or {}).get("status", "planned")
        if sid in live_ids:
            excluded.append({"id": sid, "why": "a fit_full session is fitting it right now", "status": live["segments"][sid]["status"]})
        elif stat in ("planned",):
            todo.append(c)
        else:
            excluded.append({"id": sid, "why": f"already {stat}", "status": stat})
    rows = []
    for c in todo:
        est = (c.get("estimate") or {}).get("total_min")
        if est is not None and float(est) <= 0:       # core's own rule: no step left = its results are already here
            excluded.append({"id": c["id"], "why": "nothing left to run: its results are already here", "status": "done"})
            continue
        if est is None:
            est = round(((c.get("T") or (c.get("core_s") or 30) * 30 + 240) / 30.0) * MIN_PER_WINDOW_S, 1)
        rows.append({"id": c["id"], "kind": c.get("kind") or "full", "core_s": c.get("core_s"), "est_min": float(est),
                     "k0": c.get("core_k0"), "k1": c.get("core_k1")})
    # the dance's stitch + analysis job (one CPU pod): needs every chunk's motion, so not while a fit_full session is still fitting
    final_known = any(j.get("id") == "final" for j in (core.get("jobs") or [])) or (is_demo and bool(rows))
    fstat = ((state.get("jobs") or {}).get("final") or {}).get("status")
    final_open = final_known and fstat not in ("done", "pulled", "stitched", "running", "queued")
    final_blocked = bool(live_ids)
    final_cpu = bool(final_open and not final_blocked)
    offers, offer_note = ([], None) if is_demo else _gpu_offers()
    gpu = offers[0] if offers else dict(FALLBACK_GPU)
    # the minutes in the job specs are A100 minutes: the watch stage's `farm launch` estimate assumes any other GPU is 1.5x slower
    slow = 1.0 if any(k in str(gpu["gpu"]) for k in ("A100", "H100", "H200", "B200")) else GPU_SLOWDOWN
    for r in rows:
        r["est_min_a100"] = r["est_min"]
        r["est_min"] = round(r["est_min"] * slow, 1)
    note = offer_note if not offers else None
    if is_demo:
        note = "demo dance: price assumed"
    n_pods = max(1, min(int(pods or DEFAULT_PODS), MAX_PODS, max(1, len(rows))))
    est = estimate_plan(rows, pods=n_pods, price_hr=float(gpu["price"]), final_cpu=final_cpu, price_note=note)
    # which chunks go to which pod (display only: the workers claim by priority)
    _, loads = lpt_makespan([r["est_min"] for r in rows], est["pods"])
    taken = set()
    for i, l in enumerate(loads):
        for m in l:
            r = next((x for x in rows if x["est_min"] == m and x["id"] not in taken), None)
            if r:
                taken.add(r["id"])
                est["assignment"][i]["chunks"].append(r["id"])
    bud = None
    try:
        bud = farm_view._budget(_rp_cached(), state)
    except Exception:  # noqa: BLE001
        pass
    avail = (bud or {}).get("available_usd")
    gate = "unknown" if avail is None else ("ok" if est["worst_cost_usd"] <= avail else "over")
    only = [r["id"] for r in rows] + (["final"] if final_cpu else [])
    cmds = _fill_commands(dance, est, bool(rows), final_cpu, only, sorted(live_ids))
    if have_cmd("up"):
        need, have = ["up"], {"up": True}
    else:
        need = ["queue"] + (["launch"] if rows else []) + (["launch-cpu"] if final_cpu else [])
        have = {"queue": have_cmd("queue"), "launch": have_cmd("launch") or have_cmd("pods:launch"),
                "launch-cpu": have_cmd("launch-cpu") or have_cmd("pods:launch-cpu")}
    if live_ids and not rows:
        many = len(live_ids) > 1
        msg = (f"Nothing to launch now: chunk{'s' if many else ''} {', '.join(sorted(live_ids))} "
               f"{'are' if many else 'is'} still being fitted by fit_full. The stitch and analysis job is planned once "
               "they have finished and their motion is pulled: use Fill gaps again then.")
    else:
        msg = "Nothing to do: every second is complete, or already belongs to a chunk that is done or running."
    plan = {"schema": "dancecap.monitor.fillplan/1", "dance": dance, "created": time.time(), "demo": is_demo,
            "chunks": rows, "excluded": excluded, "gaps": (core.get("report") or {}).get("gap_s"), "gpu": gpu,
            "estimate": est, "budget": {"available_usd": avail, "balance": (bud or {}).get("balance"),
                                        "remaining_cap": (bud or {}).get("remaining"), "gate": gate},
            "commands": [" ".join(c["argv"][1:]) if c["argv"] and c["argv"][0] == _py() else " ".join(c["argv"]) for c in cmds],
            "missing_commands": [n for n in need if not have[n]] + ([] if have_cmd("watch") else ["watch"]),
            "final_cpu": final_cpu, "final_blocked": bool(final_blocked and final_open), "only": only, "slowdown": slow,
            "live_fit_full": sorted(live_ids), "nothing_to_do": not rows and not final_cpu,
            "message": msg if (not rows and not final_cpu) else None}
    plan["plan_hash"] = hashlib.sha1(json.dumps({k: plan[k] for k in ("dance", "chunks", "estimate", "excluded")},
                                                sort_keys=True, default=str).encode()).hexdigest()[:16]
    write_json_atomic(actions_dir() / f"plan-{plan['plan_hash']}.json", plan)
    return plan


_rp_provider = {"fn": None}


def set_runpod_provider(fn) -> None:
    """The server hands its cached runpod summary function over (budget gate numbers) so this module never calls runpodctl."""
    _rp_provider["fn"] = fn


def _rp_cached() -> dict | None:
    fn = _rp_provider["fn"]
    return fn() if fn else None


def _fill_commands(dance: str, est: dict, has_chunks: bool, final_cpu: bool, only: list[str] | None = None,
                   exclude: list[str] | None = None) -> list[dict]:
    """The steps of Fill gaps. `farm up` (the watch stage) does plan -> queue (code, world, inputs) -> launch -> watcher in one go with
    its own budget gates; the monitor passes exactly what the user confirmed: --only the planned jobs, --exclude the chunks a
    fit_full session is fitting, the pod count, the kill-switch minutes and the worst-case ceiling. Older trees without `up` get
    plan -> queue -> pods launch."""
    if not has_chunks and not final_cpu:
        return []
    steps = [{"label": "claim live fit_full chunks", "argv": [_py(), "-m", "dancecap.pipeline.farm_actions", "_claim_live", dance],
              "timeout_s": 120}]
    if have_cmd("up"):
        argv = _farm(dance, "up", "--pods", str(max(1, est["pods"])), "--budget", str(est["worst_cost_usd"]))
        if has_chunks:
            argv += ["--minutes", str(est["pod_minutes"]), "--hard-minutes", str(est["pod_hard_minutes"])]
        if only:
            argv += ["--only", ",".join(only)]
        if exclude:
            argv += ["--exclude", ",".join(exclude)]
        steps.append({"label": "farm up (plan, queue, launch, watcher)", "argv": argv, "timeout_s": 3 * 3600})
        return steps
    steps += [{"label": "farm plan", "argv": _farm(dance, "plan"), "timeout_s": 900},
              {"label": "farm queue", "argv": _farm(dance, "queue"), "timeout_s": 3600}]
    if has_chunks:
        steps.append({"label": "launch GPU pods", "argv": [_py(), "-m", "dancecap.farm.pods", "launch", "-n", str(est["pods"]),
                                                            "--minutes", str(est["pod_minutes"]), "--hard-minutes",
                                                            str(est["pod_hard_minutes"]), "--max-cost", str(est["worst_cost_usd"])],
                      "timeout_s": 1800})
    if final_cpu:
        steps.append({"label": "launch the stitch + analysis CPU pod",
                      "argv": [_py(), "-m", "dancecap.farm.pods", "launch-cpu", "-n", "1", "--minutes", "60", "--max-cost", "1.0"],
                      "timeout_s": 900, "optional": True})
    if have_cmd("watch"):
        steps.append({"label": "start the watcher", "argv": _farm(dance, "watch"), "timeout_s": 120})
    return steps


def fill_run(dance: str, plan_hash: str, approve: dict) -> dict:
    """Step 2: the user confirmed `plan_hash` with a cost ceiling. Re-checks everything, then runs the steps detached."""
    _check_dance(dance)
    if not re.fullmatch(r"[0-9a-f]{16}", plan_hash or ""):
        raise ActionError(400, "bad_request", "plan_hash required")
    plan = read_json(actions_dir() / f"plan-{plan_hash}.json", None)
    if not isinstance(plan, dict) or plan.get("dance") != dance:
        raise ActionError(404, "not_found", "unknown plan: plan again")
    if time.time() - float(plan.get("created") or 0) > PLAN_TTL_S:
        raise ActionError(409, "plan_expired", "the plan is older than 15 minutes: plan again (stock and prices move)")
    if plan.get("nothing_to_do"):
        raise ActionError(409, "nothing_to_do", plan.get("message") or "nothing to do")
    if plan.get("demo"):
        raise ActionError(409, "demo", "this is a demo dance: nothing is launched")
    est = plan["estimate"]
    try:
        ceiling = float((approve or {}).get("max_cost_usd"))
    except (TypeError, ValueError):
        raise ActionError(400, "bad_request", "approve.max_cost_usd required") from None
    if ceiling + 1e-6 < est["worst_cost_usd"]:
        raise ActionError(409, "approval_too_low", f"the worst case is ${est['worst_cost_usd']:.2f}; you approved ${ceiling:.2f}")
    if plan["budget"]["gate"] == "over":
        raise ActionError(409, "budget_gate", f"the worst case ${est['worst_cost_usd']:.2f} exceeds what the budget allows "
                                              f"(${plan['budget']['available_usd']:.2f}): shrink the plan or free budget")
    missing = []
    if not have_cmd("up"):
        missing = [n for n in ("queue",) if not have_cmd(n)]
        if plan["chunks"] and not (have_cmd("launch") or have_cmd("pods:launch")):
            missing.append("launch")
        if plan.get("final_cpu") and not (have_cmd("launch-cpu") or have_cmd("pods:launch-cpu")):
            missing.append("launch-cpu")
    if missing:
        raise ActionError(409, "not_installed", "these farm commands are not installed yet: " + ", ".join(missing))
    if running(dance, "fill"):
        raise ActionError(409, "busy", "a Fill gaps run is already in progress")
    # live fit_full chunks may have appeared since the plan: recheck, never queue them
    live = fit_sessions.live_fits(dance)
    live_ids = {s for s, v in live["segments"].items() if v.get("status") in ("running", "queued")}
    if live_ids - set(plan.get("live_fit_full") or []):
        raise ActionError(409, "live_changed", "a fit_full session started since the plan: plan again")
    steps = _fill_commands(dance, est, bool(plan["chunks"]), bool(plan.get("final_cpu")), plan.get("only"), plan.get("live_fit_full"))
    return start("fill", dance, steps, meta={"plan_hash": plan_hash, "chunks": [c["id"] for c in plan["chunks"]],
                                              "pods": est["pods"], "approved_usd": ceiling})


def claim_live(dance: str) -> int:
    """Wrapper step: mark the chunks of live fit_full sessions `running` (owner fit_full) in state.json so `farm queue` skips them."""
    from ..farm.state import State
    live = fit_sessions.live_fits(dance)
    st = State(dance)
    n = 0
    for seg, v in live["segments"].items():
        if v.get("status") not in ("running", "queued"):
            continue
        cur = st.chunk(seg)
        if cur is None or cur.get("status") in ("planned", None):
            st.set_chunk(seg, status="running", owner="fit_full", session=v["sid"], pod=(v.get("pod") or {}).get("id"))
            n += 1
    print(f"claimed {n} live fit_full chunk(s): {', '.join(sorted(live['segments']))}")
    return 0


# ------------------------------------------------------------------------------------------------ simple actions

def pull_now(dance: str) -> dict:
    _check_dance(dance)
    if not have_cmd("pull"):
        raise ActionError(409, "not_installed", "`farm pull` is not installed yet (the watcher stage delivers it)")
    return start("pull", dance, [{"label": "farm pull (verified)", "argv": _farm(dance, "pull"), "timeout_s": 3600}])


def restart_watcher(dance: str) -> dict:
    """`farm watch stop --force` (graceful: it finishes its cycle, a stuck one is killed after 60 s), then `farm watch` (detached,
    idempotent). Nothing it has pulled is lost: the next watcher rebuilds everything from the volume and state.json."""
    _check_dance(dance)
    if not have_cmd("watch"):
        raise ActionError(409, "not_installed", "`farm watch` is not installed yet (the watcher stage delivers it)")
    rec = start("watcher", dance, [{"label": "farm watch stop", "argv": _farm(dance, "watch", "stop", "--force"), "timeout_s": 150,
                                    "optional": True},
                                   {"label": "farm watch start", "argv": _farm(dance, "watch", "start"), "timeout_s": 120}])
    return {"action": rec}


def stop_pods(dance: str, pod_ids: list[str], confirm: str) -> dict:
    """Delete the given FARM pods of this dance (state.json). Typed confirmation: the word STOP."""
    _check_dance(dance)
    if confirm != "STOP":
        raise ActionError(400, "confirm", "type STOP to confirm")
    state = read_json(farm_dir(dance) / "state.json", {}) or {}
    farm_ids = {str(k) for k, p in (state.get("pods") or {}).items() if isinstance(p, dict) and p.get("state") not in ("deleted", "gone")}
    live_ids = {p["id"] for p in fit_sessions.live_fits(dance)["pods"]}
    ids = []
    for pid in pod_ids or sorted(farm_ids):
        if not POD_ID_RE.match(str(pid)):
            raise ActionError(400, "bad_request", f"bad pod id {pid!r}")
        if pid in live_ids:
            raise ActionError(409, "fit_full_pod", f"pod {pid} belongs to a fit_full session: stop that session, not the pod here")
        if pid not in farm_ids:
            raise ActionError(409, "not_farm_pod", f"pod {pid} is not one of this farm's pods")
        ids.append(pid)
    if not ids:
        raise ActionError(409, "no_pods", "no farm pod to stop")
    if have_cmd("down"):
        # the watch stage's `down`: a verified pull first, then deletes only pods recorded in this dance's state.json; a pod whose
        # finished chunks are not pulled yet is KEPT (nothing is lost); --abandon also deletes a pod that is still running a job
        steps = [{"label": "farm down (verified pull, then delete)",
                  "argv": _farm(dance, "down", "--pod", ",".join(ids), "--abandon"), "timeout_s": 3600}]
    else:
        steps = [{"label": f"delete pod {i}", "argv": [_py(), "-m", "dancecap.farm.pods", "delete", i], "timeout_s": 300,
                  "optional": True} for i in ids]
    return start("stop_pods", dance, steps, meta={"pods": ids})


def _log_dirs(dance: str) -> list[Path]:
    return [farm_dir(dance), farm_dir(dance) / "logs"]       # watcher.log (watch stage) lives in farm/, action logs in farm/logs/


def log_names(dance: str) -> list[str]:
    found: dict[str, float] = {}
    for d in _log_dirs(dance):
        try:
            for p in d.iterdir():
                if p.is_file() and LOG_RE.match(p.name):
                    found.setdefault(p.name, p.stat().st_mtime)
        except OSError:
            continue
    return [n for n, _ in sorted(found.items(), key=lambda kv: kv[1], reverse=True)][:20]


def tail_log(dance: str, name: str | None, max_bytes: int = 60000) -> dict:
    _check_dance(dance)
    names = log_names(dance)
    if not name:
        name = next((n for n in names if n.startswith("watch")), None) or (names[0] if names else None)
    if not name:
        return {"name": None, "text": "", "names": [], "message": "no log yet: the watcher writes farm/watcher.log once it runs"}
    if not LOG_RE.match(name):
        raise ActionError(400, "bad_request", "bad log name")
    p = next((d / name for d in _log_dirs(dance) if (d / name).is_file()), None)
    if p is None:
        raise ActionError(404, "not_found", "no such log")
    from .estimate import read_log_tail
    return {"name": name, "text": redact(read_log_tail(p, max_bytes)), "names": names, "mtime": mtime(p),
            "path": paths.rel_display(p)}


def save_map_png(dance: str, data: bytes) -> dict:
    _check_dance(dance)
    if not data.startswith(b"\x89PNG\r\n\x1a\n") or len(data) > 6 << 20:
        raise ActionError(400, "bad_request", "not a PNG (or over 6 MB)")
    d = farm_dir(dance) / "review"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"monitor_map_{time.strftime('%Y%m%d-%H%M%S')}.png"
    p.write_bytes(data)
    return {"path": paths.rel_display(p), "bytes": len(data), "abs": str(p)}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["_exec"] and len(argv) == 2:
        return _exec(Path(argv[1]))
    if argv[:1] == ["_claim_live"] and len(argv) == 2:
        return claim_live(argv[1])
    print("usage: python -m dancecap.pipeline.farm_actions _exec <record.json> | _claim_live <dance>", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
