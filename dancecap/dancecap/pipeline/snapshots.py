"""Dashboard snapshots at progress points (owner, 2026-10-21: "intermediate screenshot snapshots of progress points").

    work/pipeline/snapshots/<run>/NN_<event>.png + .html, index.json {run, shots[{i, ts, event, label, png, html, method, ms}]}

<run> is an analysis run id (pipeline/analyze.py) or, for a pod session outside any analysis, its session id.

- request(run, event, label): called by the analysis driver (start, every step's end, waiting for an agent, a 15-min
  heartbeat, the end), by `analysis step` marks, and by the pod runner (upload done, bootstrap, each job, session end).
  It starts `python -m dancecap.pipeline snapshot <run> ...` DETACHED (no window, no console) and returns at once: a
  snapshot never slows or breaks the work it records.
- take(): builds a self-contained static page from the same data as the monitor (activity.py; no JS, no fetch: the 2026-10-21
  review measured 2-3 s for such a page under 100 % CPU, against 47-63 s and two hangs for the live page in headless Edge),
  renders it to PNG with headless Edge (hard 90 s timeout, scratch profile), falls back to a PIL card when Edge fails, and
  appends to index.json. One snapshot at a time per run (lock file); at most one every 15 s unless it is a final one.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from . import paths
from .util import read_json, write_json_atomic, write_text_atomic

RUN_RE = re.compile(r"^[0-9]{8}-[0-9]{6}-[A-Za-z0-9_.-]{1,80}$")   # analysis runs and session ids
EVENT_RE = re.compile(r"[^a-z0-9_]+")
NO_WINDOW = 0x08000000 if os.name == "nt" else 0
MIN_GAP_S = 15.0
EDGE_TIMEOUT_S = 90.0
WIDTH = 1400


def snap_root() -> Path:
    return paths.work() / "pipeline" / "snapshots"


def snap_dir(run: str) -> Path:
    if not RUN_RE.match(run or ""):
        raise ValueError(f"bad run id {run!r}")
    return snap_root() / run


def index(run: str) -> dict:
    d = read_json(snap_dir(run) / "index.json", None)
    return d if isinstance(d, dict) else {"run": run, "shots": []}


def list_runs(limit: int = 20) -> list[dict]:
    r = snap_root()
    if not r.is_dir():
        return []
    out = []
    for p in sorted((x for x in r.iterdir() if x.is_dir() and RUN_RE.match(x.name)),
                    key=lambda x: (x / "index.json").stat().st_mtime if (x / "index.json").exists() else 0, reverse=True)[:limit]:
        ix = read_json(p / "index.json", None) or {}
        shots = ix.get("shots") or []
        if shots:
            out.append({"run": p.name, "n": len(shots), "last": shots[-1], "kind": ix.get("kind")})
    return out


# ------------------------------------------------------------------------------------------------ request (cheap, detached)

def request(run: str, event: str, label: str = "", sid: str | None = None, final: bool = False) -> bool:
    """Start a detached snapshot process; False when skipped (tests, throttled, bad id). Never raises."""
    try:
        if not RUN_RE.match(run or ""):
            return False
        if os.environ.get("DANCECAP_SNAPSHOTS") == "0" or (not paths.is_real_root() and
                                                           os.environ.get("DANCECAP_SNAPSHOTS") != "1"):
            return False
        if os.environ.get("PYTEST_CURRENT_TEST") and os.environ.get("DANCECAP_SNAPSHOTS") != "1":
            return False
        d = snap_dir(run)
        d.mkdir(parents=True, exist_ok=True)
        last = read_json(d / "requested.json", None) or {}
        if not final and time.time() - float(last.get("ts") or 0) < MIN_GAP_S and last.get("event") == event:
            return False
        at = time.time()
        write_json_atomic(d / "requested.json", {"ts": at, "event": event})
        frozen = freeze(run, event, sid, at)
        py = _python()
        argv = [py, "-m", "dancecap.pipeline", "snapshot", run, "--event", event[:60], "--label", (label or "")[:160],
                "--at", f"{at:.1f}"]
        if sid:
            argv += ["--sid", sid]
        if frozen:
            argv += ["--data", str(frozen)]
        flags = (NO_WINDOW | 0x00000200) if os.name == "nt" else 0
        with open(d / "snapshot.log", "a", encoding="utf-8") as fh:
            subprocess.Popen(argv, cwd=str(paths.root()), stdin=subprocess.DEVNULL, stdout=fh, stderr=subprocess.STDOUT,
                             creationflags=flags, close_fds=True, start_new_session=os.name != "nt")
        return True
    except Exception:  # noqa: BLE001
        return False


def freeze(run: str, event: str, sid: str | None = None, at: float | None = None) -> Path | None:
    """The monitor's data AT THE EVENT (gather), written to <snapshots>/<run>/pending/<ms>_<event>.json for the detached
    renderer. Renders queue behind one lock and take 1-30 s each, so a snapshot that gathered at render time showed the
    state of seconds to minutes later (2026-10-22 review: "snapshots render the state at render time, not at their
    event"). None when gathering fails: the renderer then gathers itself, as before. Never raises."""
    try:
        at = time.time() if at is None else float(at)
        data = gather(run, sid, at)
        ev = EVENT_RE.sub("_", (event or "snap").lower()).strip("_")[:40] or "snap"
        pend = snap_dir(run) / "pending"
        p = pend / f"{int(at * 1000)}_{ev}.json"
        write_json_atomic(p, {"v": 1, "run": run, "event": event, "at": at, "sid": sid, "data": data}, indent=None)
        for old in pend.glob("*.json"):   # a renderer that never started leaves its file: gone after an hour
            try:
                if at - old.stat().st_mtime > 3600:
                    old.unlink()
            except OSError:
                pass
        return p
    except Exception:  # noqa: BLE001 - a snapshot never breaks the work it records
        return None


def load_frozen(path: str | Path | None, run: str) -> dict | None:
    """The data freeze() wrote for this run, or None (missing, unreadable, another run's)."""
    if not path:
        return None
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(doc, dict) or doc.get("run") != run or not isinstance(doc.get("data"), dict):
        return None
    return doc["data"]


def _python() -> str:
    try:
        from .monitor import _monitor_python
        return _monitor_python()
    except Exception:  # noqa: BLE001
        return sys.executable


class _Lock:
    def __init__(self, p: Path, wait_s: float = 150.0, stale_s: float = 240.0):
        self.p, self.wait_s, self.stale_s = p, wait_s, stale_s

    def __enter__(self):
        t_end = time.time() + self.wait_s
        while True:
            try:
                fd = os.open(self.p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                return self
            except FileExistsError:
                try:
                    if time.time() - self.p.stat().st_mtime > self.stale_s:
                        self.p.unlink()
                        continue
                except OSError:
                    pass
            except PermissionError:
                pass
            if time.time() > t_end:
                raise TimeoutError("another snapshot of this run is still rendering")
            time.sleep(0.5)

    def __exit__(self, *a):
        try:
            self.p.unlink()
        except OSError:
            pass


# ------------------------------------------------------------------------------------------------ data

FARM_RUN_RE = re.compile(r"^(?P<ts>[0-9]{8}-[0-9]{6})-farm_(?P<dance>[A-Za-z0-9_]{1,64})$")


def farm_run_id(dance: str, t: float) -> str:
    """The snapshot run id of one volume-farm run of a dance: <start, local time>-farm_<dance>."""
    return f"{time.strftime('%Y%m%d-%H%M%S', time.localtime(float(t)))}-farm_{dance}"


def gather(run: str, sid: str | None = None, now: float | None = None) -> dict:
    """{kind: analysis|session|farm, ...} from activity.py / farm_view.py (the monitor's own data)."""
    from . import activity, analyze
    now = time.time() if now is None else now
    fm = FARM_RUN_RE.match(run)
    if fm and (paths.work() / fm.group("dance") / "farm").is_dir():
        from . import estimate, farm_view
        try:
            bud = estimate.budget()
        except Exception:  # noqa: BLE001
            bud = {}
        return {"kind": "farm", "now": now, "f": farm_view.build(fm.group("dance"), rp=None, now=now), "budget": bud,
                "cap_source": activity._cap_source()}
    if analyze.RUN_RE.match(run) and analyze.load(run) is not None:
        det = activity.analysis_detail(run, now) or {}
        try:
            from . import estimate
            bud = estimate.budget()
        except Exception:  # noqa: BLE001
            bud = {}
        return {"kind": "analysis", "now": now, "a": det, "budget": bud, "cap_source": activity._cap_source()}
    card = activity.session_card(sid or run) or {"sid": sid or run, "state": "unknown"}
    try:
        from . import estimate
        bud = estimate.budget()
    except Exception:  # noqa: BLE001
        bud = {}
    return {"kind": "session", "now": now, "s": card, "budget": bud, "cap_source": activity._cap_source()}


# ------------------------------------------------------------------------------------------------ HTML (static, no JS)

CSS = """
*{box-sizing:border-box}body{margin:0;background:#f4f3ef;color:#1f2328;font:14px/1.45 'Segoe UI',system-ui,sans-serif}
.wrap{padding:22px 26px 30px;width:%dpx}
.top{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap}.brand{font-weight:600;color:#57606a;font-size:13px}
h1{font-size:24px;margin:2px 0 0}.ev{margin-left:auto;background:#ddf4ff;color:#0a3069;border:1px solid #54aeff;border-radius:12px;padding:2px 10px;font-size:13px}
.sub{color:#57606a;font-size:13px;margin:2px 0 14px}
.kpis{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:14px}
.kpi{background:#fff;border:1px solid #d0d7de;border-radius:8px;padding:10px 12px}
.kpi .l{font-size:11.5px;text-transform:uppercase;letter-spacing:.04em;color:#57606a}.kpi .v{font-size:22px;font-weight:650;font-variant-numeric:tabular-nums}
.kpi .s{font-size:12px;color:#57606a}
.card{background:#fff;border:1px solid #d0d7de;border-radius:8px;padding:12px 14px;margin-bottom:12px}
.card h2{font-size:14px;margin:0 0 8px;display:flex;gap:8px 12px;align-items:center;flex-wrap:wrap}
.lg{display:flex;gap:4px 14px;font-size:12px;color:#57606a;font-weight:400;margin-left:auto;flex-wrap:wrap}.lg i{display:inline-block;width:12px;height:12px;border-radius:2px;vertical-align:-2px;margin-right:5px}
.row{display:grid;grid-template-columns:270px 1fr 300px;gap:10px;align-items:start;padding:5px 0;border-top:1px solid #eaeef2}
.row:first-of-type{border-top:0}.nm{font-weight:600}.nm small{display:block;font-weight:400;color:#57606a;font-size:11.5px}
.sq{display:flex;flex-wrap:wrap;gap:3px}.b{width:14px;height:14px;border-radius:2px;background:#dedcd6;border:1px solid #c3c1b9}
.b.done{background:#1f9d3f;border-color:#178a35}.b.run{background:#1c6fd1;border-color:#0b55ab}.b.left{background:#fff;border:1.5px solid #1c6fd1}
.b.wait{background:repeating-linear-gradient(135deg,#d48a00 0 2px,#fbefce 2px 5px);border-color:#d48a00}
.b.fail{background:#cf222e;border-color:#a40e26}.b.skip{background:#eef0f2;border-color:#d0d7de}
.b.time{background:repeating-linear-gradient(135deg,#1c6fd1 0 2px,#dbe9fb 2px 5px);border-color:#1c6fd1}
.b.over{background:#fff;border:1.5px dashed #cf222e}.b.stop{background:#eef0f2;border:1.5px dashed #8c959f}
.b.reused{background:repeating-linear-gradient(45deg,#8c959f 0 2px,#eef0f2 2px 5px);border-color:#8c959f}
.b.c{background:#1f9d3f;border-color:#178a35}.b.p{background:repeating-linear-gradient(135deg,#d48a00 0 2px,#fbefce 2px 5px);border-color:#d48a00}.b.n{background:#dedcd6}
.st{font-size:12.5px;text-align:right;color:#57606a;min-width:0;overflow-wrap:anywhere}.st b{color:#1f2328}
.nm,.sub,td,.ev,.kpi .s{overflow-wrap:anywhere;min-width:0}.ev{max-width:620px}
.alert{border-radius:8px;padding:6px 12px;margin-bottom:6px;font-size:13px;overflow-wrap:anywhere}
.alert.red{background:#ffebe9;border:1px solid #ff8182;color:#82071e}.alert.amber{background:#fff8c5;border:1px solid #d4a72c;color:#7d4e00}
.alert.info{background:#ddf4ff;border:1px solid #54aeff;color:#0a3069}
.pill{display:inline-block;border-radius:10px;padding:0 8px;font-size:11.5px;border:1px solid #d0d7de;color:#57606a;background:#f6f8fa}
.pill.good{background:#dafbe1;border-color:#4ac26b;color:#116329}.pill.info{background:#ddf4ff;border-color:#54aeff;color:#0a3069}
.pill.warn{background:#fff8c5;border-color:#d4a72c;color:#7d4e00}.pill.crit{background:#ffebe9;border-color:#ff8182;color:#a40e26}
table{border-collapse:collapse;width:100%%;font-size:12.5px}th{text-align:left;color:#57606a;font-weight:600;font-size:11.5px;text-transform:uppercase;padding:4px 6px;border-bottom:1px solid #d0d7de}
td{padding:5px 6px;border-bottom:1px solid #eaeef2;vertical-align:top}.mono{font-family:Consolas,monospace;font-size:12px}
.bar{height:8px;background:#eaeef2;border-radius:4px;overflow:hidden;margin-top:3px}.bar i{display:block;height:100%%;background:#1c6fd1}
.cov{display:flex;flex-wrap:wrap;gap:2px}.cov .b{width:11px;height:11px}
.foot{color:#57606a;font-size:12px;margin-top:10px}.warn{color:#9a6700}.crit{color:#a40e26}
""" % WIDTH

STATUS_PILL = {"done": "good", "running": "info", "waiting_agent": "warn", "failed": "crit", "blocked": "crit",
               "pending": "", "skipped": "", "ok": "good", "orphaned": "crit", "ended": "", "preparing": "info",
               "stopped": ""}
STATUS_TEXT = {"waiting_agent": "needs an agent", "pending": "pending", "running": "running", "done": "done",
               "failed": "failed", "blocked": "blocked", "skipped": "skipped", "stopped": "stopped (run ended)"}


def clip(s, n: int = 220) -> str:
    """A long message cut at a word boundary with an ellipsis (the 90-character cut left "governing measu")."""
    s = " ".join(str(s or "").split())
    if len(s) <= n:
        return s
    cut = s[:n].rsplit(" ", 1)[0]
    return (cut if len(cut) > n * 0.6 else s[:n]) + " …"


def upload_text(up: dict) -> str:
    """Upload cell text (static): done / stopped when the session ended / no pod / in progress with a bar."""
    if not up:
        return "–"
    if up.get("no_pod"):
        return "not uploaded: no pod started"
    if not up.get("total"):
        return "–"
    mb = up["total"] / 1e6
    if up.get("done"):
        return f"{mb:.0f} MB done" + (f" at {up['mbps']:.2f} MB/s" if up.get("mbps") else "")
    if up.get("stopped") or up.get("failed"):
        return f"stopped at {(up.get('sent') or 0) / 1e6:.0f} of {mb:.0f} MB (session ended)"
    if up.get("sent") is not None:
        pct = 100.0 * up["sent"] / up["total"]
        return (f"{up['sent'] / 1e6:.0f}/{mb:.0f} MB ({pct:.0f} %)" +
                (f" at {up['mbps']:.2f} MB/s, {fmt_dur(up.get('eta_s'))} left" if up.get("mbps") else "") +
                f'<div class="bar"><i style="width:{min(100, pct):.0f}%"></i></div>')
    return f"{mb:.0f} MB to upload"


def frames_text(fr: dict) -> str:
    """One frame counter: "infer 120/599 frames · 0.355 s per person-frame · 5.6x slower than RTX 4090"."""
    tot = fr.get("total")
    s = f"{fr.get('key')}: {fr.get('done')}{('/' + str(tot)) if tot else ''} {fr.get('unit') or ''}"
    if fr.get("stage"):
        s += f" ({fr['stage']}{('/' + str(fr['stages_total'])) if fr.get('stages_total') else ''})"
    if fr.get("rate_s") is not None:
        s += f" · {float(fr['rate_s']):.3g} s per {fr.get('rate_unit') or 'unit'}"
    if fr.get("slow"):
        ref = fr.get("ref") or {}
        s += f" · {fr['slow_x']:.1f}x slower than on {(ref.get('gpu') or 'the reference GPU').replace('NVIDIA ', '')}"
    return s


def _e(x) -> str:
    return html.escape(str(x if x is not None else ""))


def fmt_dur(s) -> str:
    if s is None:
        return "–"
    s = float(s)
    if s < 90:
        return f"{s:.0f} s"
    m = s / 60
    return f"{m:.0f} min" if m < 90 else f"{int(m // 60)} h {int(m % 60):02d}"


def fmt_usd(x, d: int = 2) -> str:
    return "–" if x is None else f"${float(x):,.{d}f}"


def clock(ts) -> str:
    return time.strftime("%H:%M", time.localtime(float(ts))) if ts else "–"


def step_frac(st: dict) -> float | None:
    """The measured progress of a running step: activity.step_progress (byte / frame counters, live pod session phases),
    else the step's own byte counter; None = no progress figure."""
    pf = st.get("progress_frac") or {}
    if isinstance(pf, dict) and pf.get("frac") is not None:
        return max(0.0, min(1.0, float(pf["frac"])))
    pr = st.get("progress") or {}
    if isinstance(pr, dict) and pr.get("total"):
        return max(0.0, min(1.0, float(pr.get("done") or 0) / float(pr["total"])))
    return None


def step_blocks(st: dict, now: float, unit_min: float = 2.0, cap: int = 90) -> list[str]:
    """The defrag squares of one step: one square per pod-lane x `unit_min` of its p50 minutes (at least one per lane),
    coloured by its state. A running step is filled by MEASURED progress (step_frac: "run" squares) when there is a
    figure; without one, its elapsed time against the plan is drawn as striped "time" squares, and the time past the
    plan as extra "over" squares (2026-10-21 review: the fill was elapsed time under a "done so far" legend, so fits
    showed 11 of 12 squares while no fit had run). A reused step (nothing computed) is ONE grey square: no pods, no cost.
    The same rule is in static/components/stepmap.js."""
    if st.get("reused"):
        return ["reused"]
    lanes = int(st.get("lanes") or 1) if st.get("where") != "local" else 1
    per = max(1, int(round(float(st.get("p50_min") or 1) / unit_min)))
    n = min(cap, lanes * per)
    status = st.get("status") or "pending"
    if status in ("done",):
        return ["done"] * n
    if status == "running":
        frac = step_frac(st)
        if frac is not None:
            k = int(round(frac * n))
            return ["run"] * k + ["left"] * (n - k)
        if not st.get("started"):
            return ["left"] * n
        ratio = (now - float(st["started"])) / 60.0 / max(0.1, float(st.get("p50_min") or 1))
        k = min(n, int(round(max(0.0, ratio) * n)))
        over = min(cap - n, int(round((ratio - 1.0) * n))) if ratio > 1.0 else 0
        return ["time"] * k + ["left"] * (n - k) + ["over"] * max(0, over)
    return {"waiting_agent": ["wait"], "failed": ["fail"], "blocked": ["fail"], "skipped": ["skip"],
            "stopped": ["stop"]}.get(status, ["n"]) * n


def _blocks_html(kinds: list[str]) -> str:
    return '<div class="sq">' + "".join(f'<i class="b {k}"></i>' for k in kinds) + "</div>"


LEGEND = (("done", "done"), ("run", "running: measured progress (bytes, frames, pod phases)"),
          ("time", "running, no progress figure: time elapsed vs the plan"), ("over", "past the plan"),
          ("wait", "needs an agent"), ("reused", "reused (not computed, no pods)"), ("n", "pending"), ("fail", "failed"),
          ("stop", "stopped (the run ended)"), ("skip", "skipped"))


def _legend() -> str:
    return ('<span class="lg">' + "".join(f'<span><i class="b {k}"></i>{_e(t)}</span>' for k, t in LEGEND) +
            '<span>one square = one pod x 2 min of the plan</span></span>')


def _session_rows(cards: list[dict], now: float) -> str:
    rows = []
    for c in cards:
        uptxt = upload_text(c.get("upload") or {})
        cost = c.get("cost") or {}
        eta = c.get("eta") or {}
        tag = c.get("analysis") or {}
        st = c.get("state")
        bad = "; ".join(f"{k}: {v}" for k, v in (c.get("jobs_bad") or {}).items())
        # a session that got no pod: its error says why (2026-10-21: "44.2 MB to upload" hid "every option exceeds the
        # session max $1.00")
        prob = clip(c.get("error"), 200) if c.get("error") and (c.get("no_pod") or not bad) else clip(bad, 160)
        frames = "".join(f"<div style='font-size:11.5px' class='{'warn' if f.get('slow') else ''}'>{_e(clip(frames_text(f), 160))}</div>"
                         for f in (c.get("frames") or [])[:2])
        warns = "".join(f"<div style='font-size:11.5px' class='warn'>{_e(clip(w.get('msg') if isinstance(w, dict) else w, 180))}</div>"
                        for w in ((c.get("warnings") or []) + (c.get("state_warnings") or []))[:2])
        rows.append(f"<tr><td class='mono'>{_e(c['sid'])}</td><td>{_e(tag.get('step') or c.get('take') or '')}</td>"
                    f"<td>{_e((c.get('gpu') or '').replace('NVIDIA ', '') or ('no pod' if c.get('no_pod') else ''))}</td>"
                    f"<td><span class='pill {STATUS_PILL.get(st, '')}'>{_e(st)}</span>"
                    + (f"<div class='crit' style='font-size:11.5px'>{_e(prob)}</div>" if prob else "")
                    + f"</td><td>{_e(c.get('phase') or '')}<div style='font-size:11.5px;color:#57606a'>{_e(clip(c.get('phase_detail') or '', 140))}</div>"
                    f"{frames}{warns}</td>"
                    f"<td>{uptxt}</td><td>{fmt_dur(eta.get('p50')) if eta.get('p50') is not None else '–'}"
                    f"{(' (p90 ' + fmt_dur(eta.get('p90')) + ')') if eta.get('p90') is not None else ''}</td>"
                    f"<td>{fmt_usd(cost.get('so_far', c.get('cost_usd')))}"
                    f"{(' → ' + fmt_usd(cost.get('p50')) + ' · worst ' + fmt_usd(cost.get('worst'))) if cost.get('p50') is not None else ''}</td></tr>")
    if not rows:
        return "<p style='color:#57606a'>No pod session belongs to this run yet.</p>"
    return ("<table><thead><tr><th>session</th><th>step</th><th>GPU</th><th>state</th><th>phase</th><th>upload</th>"
            "<th>ETA</th><th>cost so far → expected</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>")


def step_line(st: dict, now: float) -> str:
    """The time text of one step (same as stepmap.js stepLine)."""
    if st.get("reused"):
        return "reused, not computed"
    if st.get("started") and st.get("ended"):
        return f"took {fmt_dur(float(st['ended']) - float(st['started']))}"
    if st.get("started") and st.get("status") == "running":
        eta = ""
        if st.get("eta_s") is not None:
            p90 = st.get("eta_s_p90")
            eta = (f" · ETA {fmt_dur(st['eta_s'])}" + (f" (p90 {fmt_dur(p90)})" if p90 is not None and p90 > st["eta_s"] else "")
                   + (" · measured" if st.get("eta_basis") == "measured" else ""))
        return f"running {fmt_dur(now - float(st['started']))} of ~{float(st.get('p50_min') or 0):.0f} min{eta}"
    if st.get("status") == "stopped":
        return "stopped when the run ended"
    return f"plan {float(st.get('p50_min') or 0):.0f} min (p90 {float(st.get('p90_min') or 0):.0f})"


def _budget_kpi(b: dict, cap_source: str | None) -> str:
    if not b:
        return ""
    return (f'<div class="kpi"><div class="l">Runpod budget</div><div class="v">{fmt_usd(b.get("spent"))}</div>'
            f'<div class="s">of {fmt_usd(b.get("cap"))} cap · {fmt_usd(b.get("remaining"))} left'
            f'{" · " + _e(cap_source) if cap_source else ""}</div></div>')


def render_analysis(data: dict, event: str, label: str) -> str:
    a, now = data["a"], data["now"]
    lv = a.get("live") or {}
    f = a.get("facts") or {}
    plan = a.get("plan") or {}
    eta = lv.get("eta_s") or {}
    fin = lv.get("finish_at") or {}
    cost = lv.get("cost_usd") or {}
    steps_html = []
    for st in a.get("steps") or []:
        sts = st.get("status") or "pending"
        dur = step_line(st, now)
        where = st.get("where")
        pods = f"{st['lanes']} pod{'s' if st['lanes'] > 1 else ''}" if where != "local" else "this PC"
        if st.get("reused"):
            pods = "no pod (reused)"
        msg = st.get("message") or ""
        pf = st.get("progress_frac") or {}
        basis = (f"progress {100 * float(pf['frac']):.0f} % from {pf.get('basis')}" if sts == "running" and
                 pf.get("frac") is not None else "")
        spent = (f"{fmt_usd(st['cost_usd'], 2 if st['cost_usd'] >= 0.01 else 4)} of pods so far"
                 if st.get("cost_usd") else "")
        steps_html.append(
            f'<div class="row"><div class="nm">{_e(st["id"])} <span class="pill {STATUS_PILL.get(sts, "")}">'
            f'{_e(STATUS_TEXT.get(sts, sts))}</span>{" <span class=pill>reused</span>" if st.get("reused") else ""}'
            f'<small>{_e(st.get("label"))}</small>'
            f'<small>{_e(where)} · {_e(pods)}{" · " + _e(st.get("executor")) if st.get("executor") not in ("builtin", "cmd") else ""}</small></div>'
            f'{_blocks_html(step_blocks(st, now))}<div class="st"><b>{_e(dur)}</b>'
            f'{("<div>" + _e(clip(basis, 160)) + "</div>") if basis else ""}{("<div>" + _e(spent) + "</div>") if spent else ""}'
            f'<div>{_e(clip(msg, 200))}</div></div></div>')
    alerts_html = "".join(f'<div class="alert {_e(x.get("level"))}">{_e(clip(x.get("message"), 300))}</div>'
                          for x in (a.get("alerts") or [])[:6])
    cov = a.get("coverage")
    cov_html = ""
    if cov and cov.get("cells"):
        cov_html = ('<div class="card"><h2>Final result of the take, one square per second (the farm\'s defrag map)'
                    f'<span class="lg">computed {_e(cov.get("updated") or "?")}</span></h2><div class="cov">'
                    + "".join(f'<i class="b {k}"></i>' for k in cov["cells"][:1200]) + "</div></div>")
    title = f"{a.get('take')} — {'monocular' if a.get('route') == 'mono' else 'multi-view'} analysis"
    finish = (f"finish ~{clock(fin.get('p50'))} (p90 {clock(fin.get('p90'))})" if fin.get("p50") else "nothing left to run")
    waiting = lv.get("waiting_agent") or []
    body = f"""
<div class="top"><span class="brand">Pipeline Monitor · snapshot</span><h1>{_e(title)}</h1>
<span class="ev">{_e(clock(now))} · {_e(label or event)}</span></div>
<div class="sub">run {_e(a.get('run'))} · {_e(a.get('input'))} · {f.get('n')} video(s), {f.get('duration_s', 0):.0f} s,
{f.get('size_mb', 0):.0f} MB · status <b>{_e(a.get('status'))}</b>{' · driver running' if a.get('driver_alive') else ''}</div>
<div class="kpis">
<div class="kpi"><div class="l">Steps</div><div class="v">{lv.get('done', 0)}/{lv.get('total', 0)}</div>
<div class="s">{len(lv.get('running') or [])} running · {len(waiting)} need an agent{(' · ' + str(len(lv.get('failed') or [])) + ' failed') if lv.get('failed') else ''}</div></div>
<div class="kpi"><div class="l">Elapsed</div><div class="v">{fmt_dur(lv.get('elapsed_s'))}</div>
<div class="s">plan end to end {plan.get('wall_min', {}).get('p50', 0):.0f} min (p90 {plan.get('wall_min', {}).get('p90', 0):.0f})</div></div>
<div class="kpi"><div class="l">Time left (ETA)</div><div class="v">{fmt_dur(eta.get('p50'))}</div>
<div class="s">p90 {fmt_dur(eta.get('p90'))} · {_e(finish)}</div></div>
<div class="kpi"><div class="l">Pod cost</div><div class="v">{fmt_usd(cost.get('so_far'))}</div>
<div class="s">expected {fmt_usd(cost.get('p50'))} · p90 {fmt_usd(cost.get('p90'))}{(' · limit ' + fmt_usd(cost.get('limit')) + ', ' + fmt_usd(cost.get('left')) + ' left') if cost.get('limit') is not None else ''}</div></div>
{_budget_kpi(data.get('budget') or {}, data.get('cap_source'))}
</div>
{alerts_html}
<div class="card"><h2>Analysis map {_legend()}</h2>{''.join(steps_html)}</div>
<div class="card"><h2>Pod sessions of this run</h2>{_session_rows(a.get('sessions') or [], now)}
{('<p style="color:#57606a;font-size:12px;margin:8px 0 0">' + _e(a.get('agent_hint')) + '</p>') if waiting and a.get('agent_hint') else ''}</div>
{cov_html}
<div class="foot">Critical path: {_e(' → '.join(lv.get('critical') or []))}. Times and costs are the plan's p50 / p90 from
the measured 2026-10 runs until a step reports; then live. Snapshot of the monitor's data at {time.strftime('%Y-%m-%d %H:%M:%S',
time.localtime(now))} (work/pipeline/snapshots/{_e(a.get('run'))}/).</div>"""
    return _page(title, body)


def render_session(data: dict, event: str, label: str) -> str:
    c, now = data["s"], data["now"]
    rows = []
    for ph in c.get("phases") or []:
        el = float(ph.get("elapsed_s") or 0)
        n = max(1, min(120, int(round(el / 15.0))))
        k = {"done": "done", "running": "run", "failed": "fail", "pending": "n"}.get(ph.get("status") or "", "n")
        if k == "n":
            n = 1
        rows.append(f'<div class="row"><div class="nm">{_e(ph.get("name"))} <span class="pill {STATUS_PILL.get(ph.get("status"), "")}">'
                    f'{_e(ph.get("status"))}</span></div>{_blocks_html([k] * n)}<div class="st"><b>{fmt_dur(el) if el else "–"}</b></div></div>')
    cost = c.get("cost") or {}
    eta = c.get("eta") or {}
    up = c.get("upload") or {}
    title = f"Pod session {c.get('sid')}"
    body = f"""
<div class="top"><span class="brand">Pipeline Monitor · snapshot</span><h1>{_e(title)}</h1>
<span class="ev">{_e(clock(now))} · {_e(label or event)}</span></div>
<div class="sub">{_e(', '.join(c.get('jobs') or []))} · take {_e(c.get('take'))} · {_e((c.get('gpu') or '').replace('NVIDIA ', ''))}
{('$%.2f/h' % c['price_hr']) if c.get('price_hr') else ''} · bundle {c.get('bundle_mb') or '?'} MB</div>
<div class="kpis">
<div class="kpi"><div class="l">State</div><div class="v">{_e(c.get('state'))}</div><div class="s">{_e(c.get('phase') or '')}</div></div>
<div class="kpi"><div class="l">Upload</div><div class="v">{(f"{up['sent'] / 1e6:.0f}/{up['total'] / 1e6:.0f} MB" if up.get('sent') is not None and up.get('total') else ('no pod' if up.get('no_pod') else '–'))}</div>
<div class="s">{'stopped: the session ended · ' if up.get('stopped') else ''}{(f"{up['mbps']:.2f} MB/s · {up.get('streams') or 1} stream(s)" if up.get('mbps') else _e(up.get('source') or ''))}</div></div>
<div class="kpi"><div class="l">Time left (ETA)</div><div class="v">{fmt_dur(eta.get('p50'))}</div><div class="s">p90 {fmt_dur(eta.get('p90'))}</div></div>
<div class="kpi"><div class="l">Cost</div><div class="v">{fmt_usd(cost.get('so_far', c.get('cost_usd')))}</div>
<div class="s">expected {fmt_usd(cost.get('p50'))} · worst {fmt_usd(cost.get('worst'))}</div></div>
{_budget_kpi(data.get('budget') or {}, data.get('cap_source'))}
</div>
<div class="card"><h2>Phases, one square per 15 s</h2>{''.join(rows) or '<p>No phase yet.</p>'}</div>
{('<div class="card"><h2>Problems</h2>' + ''.join('<p class="crit">' + _e(clip(x, 600)) + '</p>' for x in (c.get('error'), '; '.join(f'{k}: {v}' for k, v in (c.get('jobs_bad') or {}).items())) if x) + '</div>') if c.get('jobs_bad') or c.get('error') else ''}
{''.join('<div class="alert amber">' + _e(clip(w.get('msg') if isinstance(w, dict) else w, 300)) + '</div>' for w in ((c.get('warnings') or []) + (c.get('state_warnings') or []))[:4])}
{''.join('<div class="alert ' + ('amber' if f.get('slow') else 'info') + '">' + _e(clip(frames_text(f), 300)) + '</div>' for f in (c.get('frames') or [])[:3])}
<div class="foot">Snapshot of the monitor's data at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now))}.</div>"""
    return _page(title, body)


FARM_CELL = {"complete": "c", "partial": "p", "none": "n"}
CHUNK_PILL = {"running": "info", "starting": "info", "done": "good", "pulled": "good", "stitched": "good", "failed": "crit",
              "queued": "", "planned": ""}


def render_farm(data: dict, event: str, label: str) -> str:
    """A volume-farm run (the defrag map): coverage per second, chunks, pods, spend, the last farm events. Taken by the
    monitor server's farm watcher at launch, chunk milestones, a 15-min heartbeat and the end (FarmSnapper)."""
    f, now = data["f"], data["now"]
    cov = f.get("coverage") or {}
    pc, sec = cov.get("percent") or {}, cov.get("seconds") or {}
    chunks = f.get("chunks") or []
    pods = [p for p in f.get("pods") or [] if p.get("source") == "farm"]
    spend = ((f.get("budget") or {}).get("farm_spend") or {})
    n_run = sum(1 for c in chunks if c.get("status") in ("running", "starting"))
    n_done = sum(1 for c in chunks if c.get("status") in ("done", "pulled", "stitched"))
    n_fail = sum(1 for c in chunks if c.get("status") == "failed")
    cells = "".join(f'<i class="b {FARM_CELL.get(c.get("s"), "n")}"></i>' for c in (cov.get("cells") or [])[:1500])
    def span(c: dict) -> str:
        return "" if c.get("t0") is None or c.get("t1") is None else f"{float(c['t0']):.1f} → {float(c['t1']):.1f} s"
    crow = "".join(
        f"<tr><td class='mono'>{_e(c['id'])}</td><td>{_e(c.get('kind') or '')}</td><td>{_e(span(c))}</td>"
        f"<td><span class='pill {CHUNK_PILL.get(c.get('status'), '')}'>{_e(c.get('status'))}</span></td>"
        f"<td class='mono'>{_e(c.get('pod') or '')}</td><td>{_e(', '.join(c.get('stages_done') or []))}</td></tr>"
        for c in chunks[:40])
    prow = "".join(f"<tr><td>{_e((p.get('gpu') or '').replace('NVIDIA ', ''))}</td><td class='mono'>{_e(p.get('id'))}</td>"
                   f"<td>{_e(p.get('state'))}</td><td>{fmt_usd(p.get('price_hr'))}/h</td><td>{fmt_usd(p.get('cost_usd'))}</td>"
                   f"<td>{_e(', '.join(str(x) for x in p.get('chunks') or []))}</td></tr>" for p in pods[:16])
    ev = "".join(f"<tr><td class='mono'>{_e((e.get('ts') or '')[11:19])}</td><td>{_e(e.get('kind'))}</td>"
                 f"<td>{_e(clip(e.get('msg'), 220))}</td></tr>" for e in (f.get("events") or [])[:10])
    alerts_html = "".join(f'<div class="alert {"red" if x.get("level") == "red" else "amber" if x.get("level") == "amber" else "info"}">'
                          f'{_e(clip(x.get("message"), 300))}</div>' for x in (f.get("alerts") or [])[:5])
    title = f"{f.get('dance')} — volume farm"
    body = f"""
<div class="top"><span class="brand">Pipeline Monitor · snapshot</span><h1>{_e(title)}</h1>
<span class="ev">{_e(clock(now))} · {_e(label or event)}</span></div>
<div class="sub">world {_e((f.get('world') or {}).get('id') or '?')} · map computed {_e(cov.get('updated') or '–')} ·
watcher {'alive' if (f.get('watcher') or {}).get('running') else _e((f.get('watcher') or {}).get('state') or 'not running')}</div>
<div class="kpis">
<div class="kpi"><div class="l">Complete</div><div class="v">{float(pc.get('complete') or 0):.0f} %</div>
<div class="s">{float(sec.get('complete') or 0):.0f} s · partial {float(pc.get('partial') or 0):.0f} % · not done {float(pc.get('none') or 0):.0f} %</div></div>
<div class="kpi"><div class="l">Chunks</div><div class="v">{n_done}/{len(chunks)}</div>
<div class="s">{n_run} running · {n_fail} failed</div></div>
<div class="kpi"><div class="l">Farm pods</div><div class="v">{sum(1 for p in pods if p.get('state') in ('running', 'starting'))}</div>
<div class="s">{len(pods)} listed in the farm state</div></div>
<div class="kpi"><div class="l">Farm spend</div><div class="v">{fmt_usd(spend.get('farm_usd'))}</div>
<div class="s">budget {fmt_usd(spend.get('budget_usd'))}</div></div>
{_budget_kpi(data.get('budget') or {}, data.get('cap_source'))}
</div>
{alerts_html}
<div class="card"><h2>Final result, one square per second <span class="lg"><span><i class="b c"></i>complete</span>
<span><i class="b p"></i>partial</span><span><i class="b n"></i>not done</span></span></h2><div class="cov">{cells or '<p>No coverage map yet.</p>'}</div></div>
<div class="card"><h2>Chunks</h2>{('<table><thead><tr><th>chunk</th><th>kind</th><th>span</th><th>status</th><th>pod</th><th>stages done</th></tr></thead><tbody>' + crow + '</tbody></table>') if crow else '<p>No chunks planned.</p>'}</div>
<div class="card"><h2>Pods</h2>{('<table><thead><tr><th>GPU</th><th>pod</th><th>state</th><th>price</th><th>spend</th><th>chunks</th></tr></thead><tbody>' + prow + '</tbody></table>') if prow else '<p>No farm pod.</p>'}</div>
<div class="card"><h2>Farm events</h2>{('<table><tbody>' + ev + '</tbody></table>') if ev else '<p>No events.</p>'}</div>
<div class="foot">Snapshot of the monitor's farm data at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now))}.</div>"""
    return _page(title, body)


def _page(title: str, body: str) -> str:
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>{_e(title)}</title>"
            f"<style>{CSS}</style></head><body><div class='wrap'>{body}</div></body></html>")


def page_height(data: dict) -> int:
    if data["kind"] == "analysis":
        a = data["a"]
        n_rows = len(a.get("steps") or [])
        extra_lines = sum(max(0, len(step_blocks(s, data["now"])) // 46) for s in a.get("steps") or [])
        cov = (a.get("coverage") or {}).get("cells") or ""
        return int(400 + n_rows * 92 + extra_lines * 17 + max(1, len(a.get("sessions") or [])) * 84 + 160 +
                   min(6, len(a.get("alerts") or [])) * 40 + (80 + (len(cov) // 100 + 1) * 13 if cov else 0))
    if data["kind"] == "farm":
        f = data["f"]
        n_cells = len((f.get("coverage") or {}).get("cells") or [])
        n_pods = len([p for p in f.get("pods") or [] if p.get("source") == "farm"])
        return int(420 + (n_cells // 95 + 1) * 15 + min(40, len(f.get("chunks") or [])) * 34 + min(16, n_pods) * 34
                   + 10 * 34 + 260 + min(5, len(f.get("alerts") or [])) * 40)
    s = data["s"]
    return int(330 + len(s.get("phases") or []) * 40 + 160)


# ------------------------------------------------------------------------------------------------ PNG

def render_png(html_path: Path, png_path: Path, height: int) -> tuple[str, float, str | None]:
    """(method, ms, error): headless Edge screenshot of the static page; a PIL card when Edge is missing or fails."""
    t0 = time.time()
    err = None
    try:
        from .monitor import edge_path
        edge = edge_path()
    except Exception:  # noqa: BLE001
        edge = None
    if edge:
        prof = paths.monitor_dir() / "snap-edge"
        prof.mkdir(parents=True, exist_ok=True)
        if png_path.exists():
            png_path.unlink()
        argv = [edge, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--no-first-run", "--no-default-browser-check",
                "--disable-background-networking", "--disable-sync", "--disable-component-update", "--no-pings",
                "--disable-extensions", f"--user-data-dir={prof}", f"--window-size={WIDTH + 52},{max(600, height)}",
                f"--screenshot={png_path}", html_path.resolve().as_uri()]
        try:
            # one headless Edge at a time on this PC (all runs share the scratch profile; two Edges on one profile hang)
            with _Lock(paths.monitor_dir() / "snap-edge.lock", wait_s=120.0, stale_s=EDGE_TIMEOUT_S + 30):
                p = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                     creationflags=NO_WINDOW)
                try:
                    p.wait(timeout=EDGE_TIMEOUT_S)
                except subprocess.TimeoutExpired:
                    err = f"Edge did not finish in {EDGE_TIMEOUT_S:.0f} s"
                    try:
                        from .util import kill_tree
                        kill_tree(p.pid)
                    except Exception:  # noqa: BLE001
                        p.kill()
            if png_path.exists() and png_path.stat().st_size > 1000:
                return "edge", round((time.time() - t0) * 1000), None
            err = err or f"Edge wrote no screenshot (rc {p.returncode})"
        except (OSError, TimeoutError) as e:
            err = f"Edge: {e}"
    else:
        err = "Microsoft Edge not found"
    try:
        _pil_card(html_path, png_path)
        return "pil", round((time.time() - t0) * 1000), err
    except Exception as e:  # noqa: BLE001
        return "none", round((time.time() - t0) * 1000), f"{err}; PIL: {e}"


def _pil_card(html_path: Path, png_path: Path) -> None:
    """A plain text card of the page (fallback): its title and visible text lines."""
    from PIL import Image, ImageDraw
    txt = re.sub(r"<style>.*?</style>", "", html_path.read_text(encoding="utf-8"), flags=re.S)
    txt = re.sub(r"<(div|tr|h1|h2|p)[^>]*>", "\n", txt)
    txt = html.unescape(re.sub(r"<[^>]+>", " ", txt))
    lines = [re.sub(r"\s+", " ", x).strip() for x in txt.splitlines()]
    lines = [x for x in lines if x][:70]
    im = Image.new("RGB", (WIDTH, 40 + 18 * len(lines)), (244, 243, 239))
    dr = ImageDraw.Draw(im)
    for i, line in enumerate(lines):
        dr.text((20, 16 + 18 * i), line[:180], fill=(31, 35, 40))
    im.save(png_path)


# ------------------------------------------------------------------------------------------------ take

def take(run: str, event: str, label: str = "", sid: str | None = None, requested: float | None = None,
         data_file: str | Path | None = None) -> dict:
    """Render one snapshot. data_file: the data freeze() captured at the event (request()); without it (or when it is
    unreadable) the data are gathered now. The shot's ts is the time its data are from; `rendered` when it was drawn."""
    d = snap_dir(run)
    d.mkdir(parents=True, exist_ok=True)
    try:
        with _Lock(d / "render.lock"):
            ix = index(run)
            now = time.time()
            data = load_frozen(data_file, run)
            frozen = data is not None
            if data is None:
                data = gather(run, sid, now)
            at = float(data.get("now") or now)
            page = {"analysis": render_analysis, "farm": render_farm}.get(data["kind"], render_session)(data, event, label)
            i = len(ix.get("shots") or []) + 1
            ev = EVENT_RE.sub("_", (event or "snap").lower()).strip("_")[:40] or "snap"
            base = f"{i:02d}_{ev}"
            hp, pp = d / f"{base}.html", d / f"{base}.png"
            write_text_atomic(hp, page)
            method, ms, err = render_png(hp, pp, page_height(data))
            shot = {"i": i, "ts": round(at, 1), "rendered": round(time.time(), 1), "requested": round(requested or at, 1),
                    "frozen": frozen, "event": ev, "label": label, "html": hp.name,
                    "png": pp.name if pp.exists() else None, "method": method, "ms": ms, "error": err,
                    "kind": data["kind"], "sid": sid}
            return _index_shot(run, d, data, shot)
    finally:
        if data_file:
            try:
                Path(data_file).unlink()
            except OSError:
                pass


def _index_shot(run: str, d: Path, data: dict, shot: dict) -> dict:
    """Append a rendered shot to index.json (with the run's live figures of the data it was drawn from)."""
    if data["kind"] == "analysis":
        lv = data["a"].get("live") or {}
        shot.update({"done": lv.get("done"), "total": lv.get("total"), "eta_s": (lv.get("eta_s") or {}).get("p50"),
                     "cost_so_far": (lv.get("cost_usd") or {}).get("so_far")})
    elif data["kind"] == "farm":
        f = data["f"]
        shot.update({"dance": f.get("dance"), "complete_pct": ((f.get("coverage") or {}).get("percent") or {}).get("complete"),
                     "chunks_done": sum(1 for c in f.get("chunks") or [] if c.get("status") in ("done", "pulled", "stitched")),
                     "chunks": len(f.get("chunks") or [])})
    ix = index(run)      # re-read: another process may have appended meanwhile (the lock serialises renders only)
    ix.setdefault("shots", []).append(shot)
    # renders queue behind one lock and finish in any order: the index is kept in the order they were asked for
    ix["shots"].sort(key=lambda x: (x.get("requested") or x.get("ts") or 0, x.get("i") or 0))
    ix.update({"run": run, "kind": data["kind"], "updated": time.time()})
    write_json_atomic(d / "index.json", ix)
    return shot


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.pipeline snapshot")
    ap.add_argument("run")
    ap.add_argument("--event", default="snap")
    ap.add_argument("--label", default="")
    ap.add_argument("--sid")
    ap.add_argument("--at", type=float, help="when the snapshot was asked for (epoch s)")
    ap.add_argument("--data", help="the monitor's data captured at the event (request() -> freeze())")
    a = ap.parse_args(argv)
    try:
        shot = take(a.run, a.event, a.label, a.sid, requested=a.at, data_file=a.data)
    except Exception as e:  # noqa: BLE001
        print(f"[snapshot] {a.run} {a.event}: {type(e).__name__}: {e}", flush=True)
        return 1
    print(f"[snapshot] {time.strftime('%H:%M:%S')} {a.run} {shot['i']:02d} {shot['event']}: {shot['method']} "
          f"{shot['ms']} ms{(' (' + shot['error'] + ')') if shot.get('error') else ''}", flush=True)
    return 0
