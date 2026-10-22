"""Progress puller (L1): the runner, not the monitor, pulls a pod session's progress over the ssh it already has.

One daemon thread per session. Every `interval` s (15; 60 after 3 failures in a row) it runs ONE short ssh command
with the pod's own options (same key, same mapped port 22, LogLevel=ERROR, + ConnectTimeout/BatchMode):

    ssh <base> "python3 - <base64 spec>"   < PROBE        (PROBE is a fixed stdlib script sent on stdin)

The probe is read-only: stat + read of the pod progress file from a byte offset, the upload's .part size, the kill-switch
deadline, nvidia-smi utilisation, /proc/loadavg, the cgroup CPU quota, the new tail bytes of a few third-party logs
(MAMMA ma_3d iterations, ma_2d tqdm, front task logs) and front_pod.json task states. It prints one ASCII JSON line.
No new ports, nothing installed or left on the pod; it never reads process environments (the pod-scoped key lives
there) and never writes.

Locally (work/runpod/sessions/<sid>/progress/):
    pod.jsonl     the pod file mirrored line by line (+ pod, rx, skew) and tail-adapter events (src "tail")
    puller.json   {pod, offset, last_ok, last_err, rtt_s, skew_s, kill_at, gpu[], load1, cpus, upload_part_bytes,
                   upload_bytes, tails{path: offset}, front{...}, interval_s, ticks, fails}   (tmp + os.replace)

Any exception inside the puller is caught and recorded in puller.json.last_err: a session never fails because of
progress. Monitors only read these files.
"""
from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path

from . import progress as P

PROBE = r'''
import base64, glob, json, os, subprocess, sys, time
spec = json.loads(base64.b64decode(sys.argv[1]).decode("utf-8"))
out = {"now": time.time(), "v": 1}

def rd(path, start, limit):
    with open(path, "rb") as fh:
        fh.seek(start)
        return fh.read(limit)

p = spec.get("progress")
if p:
    try:
        size = os.stat(p).st_size
        off = int(spec.get("offset") or 0)
        if off > size:
            off = 0
        data = rd(p, off, min(size - off, int(spec.get("max_bytes") or 262144)))
        cut = data.rfind(b"\n") + 1
        out.update(size=size, offset=off, next=off + cut, data=data[:cut].decode("utf-8", "replace"))
    except OSError:
        out["size"] = None
d = spec.get("deadline")
if d:
    try:
        out["deadline"] = int(open(d).read().split()[0])
    except (OSError, ValueError, IndexError):
        out["deadline"] = None
u = spec.get("upload_part")
if u:
    for cand in (u, u[:-5] if u.endswith(".part") else None):
        try:
            if cand:
                out["upload_part"] = os.stat(cand).st_size
                out["upload_done"] = cand != u
                break
        except OSError:
            continue
try:
    r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
    out["gpu"] = [dict(zip(("util", "mem_used", "mem_total"), [float(x) for x in ln.split(",")]))
                  for ln in r.stdout.strip().splitlines() if ln.count(",") == 2]
except Exception:
    out["gpu"] = None
try:
    out["load1"] = float(open("/proc/loadavg").read().split()[0])
except Exception:
    out["load1"] = None
try:
    q, per = open("/sys/fs/cgroup/cpu.max").read().split()[:2]
    out["cpus"] = None if q == "max" else round(int(q) / int(per), 2)
except Exception:
    out["cpus"] = None
tails = {}
offs = spec.get("tail_offsets") or {}
tmax = int(spec.get("tail_max") or 16384)
n = 0
for t in spec.get("tails") or []:
    for f in sorted(glob.glob(t.get("glob", "")))[: int(t.get("max_files") or 32)]:
        if n >= 32:
            break
        n += 1
        try:
            st = os.stat(f)
            prev = int(offs.get(f, 0))
            if prev > st.st_size:
                prev = 0
            start = max(prev, st.st_size - tmax)
            data = rd(f, start, st.st_size - start)
            cut = max(data.rfind(b"\n"), data.rfind(b"\r")) + 1
            tails[f] = {"size": st.st_size, "mtime": st.st_mtime, "start": start, "next": start + cut,
                        "skipped": start - prev, "data": data[:cut].decode("utf-8", "replace")}
        except OSError:
            continue
out["tails"] = tails
fp = spec.get("front")
if fp:
    try:
        doc = json.loads(open(fp).read())
        keep = ("state", "kind", "seconds", "start_ts", "deps", "after", "optional", "restarts")
        out["front"] = {"cams": doc.get("cams"), "long_pole": doc.get("long_pole"), "updated": doc.get("updated"),
                        "tasks": {k: {x: v.get(x) for x in keep} for k, v in (doc.get("tasks") or {}).items()}}
    except Exception:
        out["front"] = None
print(json.dumps(out, ensure_ascii=True))
'''

POD_WS = "/workspace/HeadMovement"
DEADLINE_FILE = "/workspace/.dancecap_kill_at"


def pod_progress_path(sid: str) -> str:
    """The session's pod progress file (all pod jobs append to it through DANCECAP_PROGRESS)."""
    return f"{POD_WS}/dancecap/work/runpod/sessions/{sid}/progress.jsonl"


# the take whose mamma/logs a MAMMA command writes: `mamma fit <take>` (often a chunk take, not the job's own: the
# 2026-10-21 trial's fit job showed no iterations), `mamma run`, `mamma_chunks fit|gate <take>`
_MAMMA_TAKE = re.compile(r"\bmamma(?:_chunks)?\s+(?:fit|run|gate)\s+([A-Za-z0-9_.-]+)")


def default_tails(jobs) -> list[dict]:
    """Third-party logs worth tailing for these jobs (L1 bridge for tools we do not instrument)."""
    out = []
    for j in jobs or []:
        take = getattr(j, "take", None) or (j.get("take") if isinstance(j, dict) else None)
        cmds = " ".join(getattr(c, "cmd", "") if not isinstance(c, dict) else c.get("cmd", "")
                        for c in (getattr(j, "commands", None) or (j.get("commands") if isinstance(j, dict) else [])
                                  or []))
        if not take:
            continue
        name = getattr(j, "name", None) or (j.get("name") if isinstance(j, dict) else None) or "mamma"
        w = f"{POD_WS}/dancecap/work/{take}"
        if "mamma" in cmds:
            for t in dict.fromkeys(_MAMMA_TAKE.findall(cmds) or [take]):
                out.append({"glob": f"{POD_WS}/dancecap/work/{t}/mamma/logs/ma_3d_*.log", "adapter": "mamma_iter",
                            "lane_from": r"ma_3d_(.+)\.log$", "step": "fit_chunks", "job": name})
        if "front_pod" in cmds:
            out.append({"glob": f"{w}/front/logs/loc*.log", "adapter": "loc", "lane_from": r"/(loc[^/]*)\.log$",
                        "step": "front", "job": "front"})
            out.append({"glob": f"{w}/front/logs/ma2d*.log", "adapter": "tqdm", "lane_from": r"/(ma2d[^/]*)\.log$",
                        "step": "front/ma2d", "job": "front"})
            out.append({"glob": f"{w}/mamma/logs/ma_2d_*.log", "adapter": "tqdm", "lane_from": r"ma_2d_(\w+)\.log$",
                        "step": "front/ma2d", "job": "front"})
    seen, uniq = set(), []
    for t in out:
        if t["glob"] not in seen:
            seen.add(t["glob"])
            uniq.append(t)
    return uniq


def default_front(jobs) -> str | None:
    for j in jobs or []:
        take = getattr(j, "take", None)
        cmds = " ".join(getattr(c, "cmd", "") for c in getattr(j, "commands", []) or [])
        if take and "front_pod" in cmds:
            return f"{POD_WS}/dancecap/work/{take}/front/front_pod.json"
    return None


def ssh_transport(pod, connect_timeout: int = 10, timeout: float = 45.0):
    """transport(b64 spec) -> probe stdout, through the pod's own ssh options (remote.Pod._ssh_base)."""
    base = list(pod._ssh_base())
    argv = base[:-1] + ["-o", f"ConnectTimeout={connect_timeout}", "-o", "BatchMode=yes", base[-1]]

    def run(b64: str) -> str:
        p = subprocess.run(argv + [f"python3 - {b64}"], input=PROBE, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout)
        if p.returncode != 0:
            raise RuntimeError(f"probe rc={p.returncode}: {P.redact((p.stderr or p.stdout)[-300:])}")
        return p.stdout
    return run


def local_transport(python: str | None = None, timeout: float = 30.0):
    """For tests and dry runs: the same PROBE, run by a local python against local paths."""
    import sys
    exe = python or sys.executable

    def run(b64: str) -> str:
        p = subprocess.run([exe, "-", b64], input=PROBE, capture_output=True, text=True, timeout=timeout)
        if p.returncode != 0:
            raise RuntimeError(f"probe rc={p.returncode}: {(p.stderr or p.stdout)[-300:]}")
        return p.stdout
    return run


class Puller:
    """Pulls one pod's progress every `interval` s into <session>/progress/. start() / stop(); never raises."""

    def __init__(self, session_dir: str | os.PathLike, pod_id: str, transport, pod_progress: str,
                 interval: float = 15.0, slow_interval: float = 60.0, upload_part: str | None = None,
                 upload_bytes: int | None = None, tails: list[dict] | None = None, front: str | None = None,
                 deadline_file: str | None = DEADLINE_FILE, max_bytes: int = 262144):
        self.dir = Path(session_dir) / "progress"
        self.pod_id, self.transport, self.pod_progress = pod_id, transport, pod_progress
        self.interval, self.slow_interval = interval, slow_interval
        self.upload_part, self.upload_bytes = upload_part, upload_bytes
        self.tails = tails or []
        self.front, self.deadline_file, self.max_bytes = front, deadline_file, max_bytes
        self.stop_ev = threading.Event()
        self.thread: threading.Thread | None = None
        prev = P.read_json(self.dir / "puller.json") or {}
        # one offset per pod: a resumed session (same sid, new pod) starts the new pod's file from 0
        self.offset = int(prev.get("offset") or 0) if prev.get("pod") == pod_id else 0
        self.tail_offsets: dict[str, int] = dict(prev.get("tails") or {}) if prev.get("pod") == pod_id else {}
        self.parsers: dict[str, P.LineParser] = {}
        self.fails = 0
        self.ticks = 0
        self.status: dict = {"pod": pod_id, "interval_s": interval, "offset": self.offset, "ticks": 0, "fails": 0,
                             "upload_bytes": upload_bytes}
        self._lock = threading.Lock()

    # lifecycle ----------------------------------------------------------------------------------------------
    def start(self) -> "Puller":
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            self.thread = threading.Thread(target=self._loop, name=f"dancecap-puller-{self.pod_id}", daemon=True)
            self.thread.start()
        except Exception as e:
            self.status["last_err"] = f"start: {e}"[:300]
        return self

    def stop(self, timeout: float = 10.0) -> None:
        try:
            self.stop_ev.set()
            if self.thread is not None:
                self.thread.join(timeout)
        except Exception:
            pass

    def _loop(self) -> None:
        while not self.stop_ev.is_set():
            self.tick()
            wait = self.slow_interval if self.fails >= 3 else self.interval
            if self.stop_ev.wait(wait):
                break
        try:
            self.tick()  # last pull after the jobs: the final events
        except Exception:
            pass

    # one probe ----------------------------------------------------------------------------------------------
    def spec(self) -> dict:
        return {"progress": self.pod_progress, "offset": self.offset, "max_bytes": self.max_bytes,
                "deadline": self.deadline_file, "upload_part": self.upload_part,
                "tails": [{"glob": t["glob"], "max_files": t.get("max_files", 32)} for t in self.tails],
                "tail_offsets": self.tail_offsets, "tail_max": 16384, "front": self.front}

    def tick(self) -> bool:
        with self._lock:
            try:
                return self._tick()
            except Exception as e:
                self.fails += 1
                self.status.update(last_err=P.redact(f"{type(e).__name__}: {e}")[:300], last_err_ts=time.time(),
                                   fails=self.fails)
                self._save()
                return False

    def _tick(self) -> bool:
        b64 = base64.b64encode(json.dumps(self.spec(), separators=(",", ":")).encode()).decode()
        t0 = time.time()
        raw = self.transport(b64)
        rx = time.time()
        rtt = rx - t0
        line = next((ln for ln in reversed(raw.strip().splitlines()) if ln.startswith("{")), None)
        if not line:
            raise RuntimeError("probe printed no JSON")
        out = json.loads(line)
        skew = rx - rtt / 2 - float(out.get("now") or rx)
        lines = []
        data = out.get("data") or ""
        if out.get("size") is not None and out.get("next") is not None:
            for ln in data.split("\n"):
                if not ln.strip():
                    continue
                try:
                    ev = json.loads(ln)
                except ValueError:
                    continue
                if not isinstance(ev, dict) or ev.get("v") != P.V:
                    continue
                ev.update(pod=self.pod_id, rx=round(rx, 3), skew=round(skew, 3))
                if "msg" in ev:
                    ev["msg"] = P.redact(str(ev["msg"]))[:P.MSG_MAX]
                lines.append(P._dumps(ev))
            self.offset = int(out["next"])
        for path, t in (out.get("tails") or {}).items():
            spec = next((x for x in self.tails if _glob_match(x["glob"], path)), None)
            if spec is None:
                continue
            parser = self.parsers.get(path)
            if parser is None:
                lane = None
                if spec.get("lane_from"):
                    m = re.search(spec["lane_from"], path)
                    lane = m.group(1) if m else None
                parser = self.parsers[path] = P.LineParser((spec.get("adapter", "tqdm"),), step=spec.get("step"),
                                                           lane=lane, job=spec.get("job"))
            if t.get("skipped"):  # more than the tail window was written since last tick: counters restart there
                pass
            for seg in re.split(r"[\r\n]", t.get("data") or ""):
                if not seg.strip():
                    continue
                for ev in parser.feed(seg):
                    # local receive time (the line was on the pod at most one interval earlier)
                    ev.update(src="tail", pod=self.pod_id, rx=round(rx, 3), ts=round(rx - rtt / 2, 3), v=P.V,
                              file=path.rsplit("/", 1)[-1])
                    ev.pop("line", None)
                    lines.append(P._dumps(ev))
            self.tail_offsets[path] = int(t.get("next") or 0)
        if lines:
            self.dir.mkdir(parents=True, exist_ok=True)
            with open(self.dir / "pod.jsonl", "a", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + "\n")
        self.fails = 0
        self.ticks += 1
        gpus = out.get("gpu") or []
        if gpus and max(g.get("util", 0) for g in gpus) < 5:
            self.status.setdefault("gpu_idle_since", round(rx, 1))
        else:
            self.status.pop("gpu_idle_since", None)
        self.status.update(
            pod=self.pod_id, offset=self.offset, last_ok=round(rx, 3), rtt_s=round(rtt, 3), skew_s=round(skew, 3),
            kill_at=out.get("deadline"), gpu=out.get("gpu"), load1=out.get("load1"), cpus=out.get("cpus"),
            upload_part_bytes=out.get("upload_part"), upload_done=out.get("upload_done"),
            upload_bytes=self.upload_bytes, tails=self.tail_offsets, front=out.get("front"),
            pod_file_size=out.get("size"), ticks=self.ticks, fails=0, interval_s=self.interval,
            events_last_tick=len(lines))
        self.status.pop("last_err", None)
        self._save()
        return True

    def _save(self) -> None:
        P.write_json_atomic(self.dir / "puller.json", self.status)


def _glob_match(pattern: str, path: str) -> bool:
    import fnmatch
    return fnmatch.fnmatch(path, pattern)


def start_for_session(ses, pod, jobs, upload_part: str | None, upload_bytes: int | None,
                      interval: float = 15.0) -> Puller | None:
    """remote_jobs hook: start a puller for this session's pod, or None (any failure -> no puller, session unaffected)."""
    try:
        if os.environ.get("DANCECAP_PROGRESS_PULL", "1") == "0":
            return None
        p = Puller(ses.dir, pod.id, ssh_transport(pod), pod_progress_path(ses.sid), interval=interval,
                   upload_part=upload_part, upload_bytes=upload_bytes, tails=default_tails(jobs),
                   front=default_front(jobs))
        return p.start()
    except Exception:
        return None
