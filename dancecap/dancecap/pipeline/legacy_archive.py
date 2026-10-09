"""Clear the Pipeline Monitor's old jobs without deleting any evidence (user, 2026-10-08: "I want all the old jobs cleared out
of the monitor"; docs/FARM.md section 7).

    python -m dancecap.pipeline archive-legacy [--date YYYY-MM-DD] [--dry-run]

Moves, never deletes:
    work/pipeline/runs/*          the run registry (one dir per old pipeline / stage / probe / tool run)
    work/pipeline/plans/*         the saved plans (plan_hash.json)
    work/pipeline/runs_archive/   the 14-day archive the monitor's housekeeping fills
    work/<take>/pipeline/events.jsonl   per-take stage history (the files the stage logic READS - gates, accepted inputs,
                                  camera-type sidecars - stay where they are)
into work/pipeline/archive/<date>/{runs,plans,runs_archive,takes/<take>/}, with a MANIFEST.json listing what moved. A run that
is still alive (a pid that exists and no rc.json) is skipped and reported. `work/runpod/sessions/*` (the pod session history,
including the live `fit_full` sessions) and `work/runpod/ledger.json` are NEVER touched: the monitor's farm view simply does
not list old sessions, it reads only the newest session of each fit segment.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

from . import paths
from .util import RUN_RE, pid_alive, read_json, write_json_atomic


def _alive_run(d: Path) -> bool:
    if (d / "rc.json").exists():
        return False
    pid = read_json(d / "pid.json", {}) or {}
    return bool(pid.get("pid")) and pid_alive(pid.get("pid"), pid.get("create_time"))


def archive_legacy(date: str | None = None, dry_run: bool = False) -> dict:
    date = date or time.strftime("%Y-%m-%d")
    base = paths.work() / "pipeline"
    dest = base / "archive" / date
    report: dict = {"date": date, "dest": paths.rel_display(dest), "moved": {"runs": [], "plans": [], "runs_archive": [],
                                                                               "takes": []}, "skipped": [], "dry_run": dry_run}

    def move(src: Path, dst: Path, key: str) -> None:
        if dry_run:
            report["moved"][key].append(src.name)
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        target = dst
        n = 1
        while target.exists():                       # a second run on the same day never overwrites
            target = dst.with_name(f"{dst.name}.{n}")
            n += 1
        shutil.move(str(src), str(target))
        report["moved"][key].append(src.name)

    runs = base / "runs"
    if runs.is_dir():
        for e in sorted(runs.iterdir()):
            if not e.is_dir() or not RUN_RE.match(e.name):
                continue
            if _alive_run(e):
                report["skipped"].append({"run": e.name, "why": "still running"})
                continue
            move(e, dest / "runs" / e.name, "runs")
    plans = base / "plans"
    if plans.is_dir():
        for e in sorted(plans.iterdir()):
            if e.is_file() and e.suffix == ".json":
                move(e, dest / "plans" / e.name, "plans")
    ra = base / "runs_archive"
    if ra.is_dir():
        for e in sorted(ra.iterdir()):
            move(e, dest / "runs_archive" / e.name, "runs_archive")
    try:
        takes = [p for p in paths.work().iterdir() if p.is_dir()]
    except OSError:
        takes = []
    for t in takes:
        ev = t / "pipeline" / "events.jsonl"
        if ev.is_file():
            move(ev, dest / "takes" / t.name / "events.jsonl", "takes")
    report["counts"] = {k: len(v) for k, v in report["moved"].items()}
    if not dry_run and any(report["counts"].values()):
        write_json_atomic(dest / "MANIFEST.json", {**report, "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                                   "note": "evidence kept on disk; nothing deleted; see the module docstring"})
    try:
        from . import runs as _runs
        _runs._desc_cache.clear()
        _runs._info_cache.clear()
    except Exception:  # noqa: BLE001
        pass
    return report


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m dancecap.pipeline archive-legacy")
    ap.add_argument("--date")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    rep = archive_legacy(a.date, a.dry_run)
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
