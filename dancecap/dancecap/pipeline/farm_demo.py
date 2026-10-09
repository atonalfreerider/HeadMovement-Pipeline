"""Synthetic farm state for the Pipeline Monitor's demos and screenshots (no real dance, no likeness, no pod).

    python -m dancecap.pipeline farm-demo [--dance farm_demo] [--scenario mostly_partial|midrun|done|empty] [--remove]

Writes work/<dance>/farm/{coverage.json, state.json} in core's schemas (dancecap.farm.coverage/1, dancecap.farm.state/1) with
state.json["demo"] = true, so the monitor labels it "(demo)" and never lists it unless asked (`?demo=1`, or it is opened by its
route #/farm/<dance>). The pods, jobs, spend and budget in it are made up; nothing here talks to Runpod.
"""
from __future__ import annotations

import json
import shutil
import sys
import time

from . import paths
from .util import write_json_atomic

ANCHOR, HZ, N_CELLS, K_START = 30.151176844227862, 30.0, 190, -90
SCENARIOS = ("mostly_partial", "midrun", "done", "empty")


def _cell(i: int, state: str, stage: str, n: int, mx: int, flag: str, reasons: list[str], chunk: str | None) -> dict:
    k0 = K_START + i * 30
    lv = {"front": 30 if stage != "none" else 0, "fit": 30 if stage in ("fit", "refined", "analysed") else 0,
          "refined": 30 if stage in ("refined", "analysed") else 0, "analysed": 30 if stage == "analysed" else 0}
    return {"i": i, "k0": k0, "k1": k0 + 29, "t0": round(ANCHOR + k0 / HZ, 4), "t1": round(ANCHOR + (k0 + 30) / HZ, 4), "frames": 30,
            "state": state, "stage": stage, "n_cams": n, "max_cams": mx, "flag": flag, "reasons": reasons, "levels": lv,
            "chunk": chunk, "src": "demo"}


def _plan(scenario: str) -> tuple[list[tuple[int, int, str, str]], dict]:
    """(runs of cells: first, last, state, stage), and the chunk table {id: (first cell, last cell, status, extra)}."""
    if scenario == "mostly_partial":
        runs = [(0, 18, "complete", "analysed"), (19, 52, "partial", "front"), (53, 85, "partial", "fit"),
                (86, 98, "partial", "refined"), (99, 132, "partial", "front"), (133, 189, "none", "none")]
        chunks = {"g0": (0, 18, "stitched", {}), "g570": (19, 52, "running", {"step": "fit", "progress": 0.46, "pod": "demo4090a", "eta_min": 24}),
                  "g1590": (53, 85, "running", {"step": "refine", "progress": 0.81, "pod": "demo4090b", "eta_min": 7}),
                  "g2580": (86, 98, "done", {}), "g3000": (99, 132, "queued", {}), "g4020": (133, 161, "planned", {}),
                  "g4890": (162, 189, "planned", {})}
    elif scenario == "midrun":
        runs = [(0, 18, "complete", "analysed"), (19, 52, "complete", "refined"), (53, 85, "partial", "fit"),
                (86, 118, "partial", "front"), (119, 150, "partial", "front"), (151, 189, "none", "none")]
        chunks = {"g0": (0, 18, "stitched", {}), "g570": (19, 52, "pulled", {}),
                  "g1590": (53, 85, "running", {"step": "refine", "progress": 0.7, "pod": "demo4090a", "eta_min": 9}),
                  "g2580": (86, 118, "running", {"step": "fit", "progress": 0.35, "pod": "demo4090b", "eta_min": 31}),
                  "g3570": (119, 150, "running", {"step": "front", "progress": 0.12, "pod": "demo4090c", "eta_min": 44}),
                  "g4530": (151, 189, "queued", {})}
    elif scenario == "done":
        runs = [(0, 189, "complete", "analysed")]
        chunks = {"g0": (0, 94, "stitched", {}), "g2850": (95, 189, "stitched", {})}
    else:
        runs = [(0, 189, "none", "none")]
        chunks = {}
    return runs, chunks


def write_demo(dance: str = "farm_demo", scenario: str = "mostly_partial") -> dict:
    if scenario not in SCENARIOS:
        raise ValueError(f"scenario: {SCENARIOS}")
    now = time.time()
    runs, chunks = _plan(scenario)
    stage_of = {}
    for a, b, st, g in runs:
        for i in range(a, b + 1):
            stage_of[i] = (st, g)
    owner = {}
    for cid, (a, b, _s, _x) in chunks.items():
        for i in range(a, b + 1):
            owner[i] = cid
    cells = []
    for i in range(N_CELLS):
        st, g = stage_of[i]
        n, mx, flag, reasons = (8, 8, "multi", []) if st == "complete" else (0, 0, "none", ["nothing yet"]) if st == "none" else (
            (7, 8, "multi", {"front": ["front only (no 3D yet)"], "fit": ["raw fit (not refined)"],
                             "refined": ["refined, awaiting stitch"]}[g]))
        if st == "partial" and 105 <= i <= 112:
            n, mx, flag, reasons = 1, 2, "mono", ["mono (1 of 2 cameras)", *reasons]
        elif st == "partial" and 70 <= i <= 73:
            n, mx, flag, reasons = 2, 6, "two_view", ["two_view", "reprojection over the gate"]
        cells.append(_cell(i, st, g, n, mx, flag, reasons, owner.get(i)))
    counts = {s: sum(1 for c in cells if c["state"] == s) for s in ("complete", "partial", "none")}
    chunk_list = [{"id": cid, "k0": K_START + max(0, a - 4) * 30, "k1": K_START + min(N_CELLS - 1, b + 4) * 30 + 29,
                   "core_k0": K_START + a * 30, "core_k1": K_START + b * 30 + 29, "status": s, "kind": "full", "origin": "farm"}
                  for cid, (a, b, s, _x) in chunks.items()]
    cov = {"schema": "dancecap.farm.coverage/1", "dance": dance, "t0": cells[0]["t0"], "t1": cells[-1]["t1"], "cell_s": 1.0,
           "anchor": ANCHOR, "hz": HZ, "k0": K_START, "k1": K_START + N_CELLS * 30 - 1, "n_cells": N_CELLS, "cells": cells,
           "counts": counts, "seconds": {k: float(v) for k, v in counts.items()},
           "percent": {k: round(100.0 * v / N_CELLS, 1) for k, v in counts.items()},
           "stages": {s: sum(1 for c in cells if c["stage"] == s) for s in ("none", "front", "fit", "refined", "analysed")},
           "world_id": "demo00000000", "world": {"id": "demo00000000", "pose_id": "demopose0000", "sealed": True},
           "chunks": chunk_list, "updated": time.strftime("%Y-%m-%dT%H:%M:%S")}
    pods = {}
    cs = {}
    for cid, (a, b, s, x) in chunks.items():
        cs[cid] = {"id": cid, "k0": K_START + max(0, a - 4) * 30, "k1": K_START + min(N_CELLS - 1, b + 4) * 30 + 29,
                   "core_k0": K_START + a * 30, "core_k1": K_START + b * 30 + 29, "overlap_k": 120, "kind": "full",
                   "origin": "farm", "status": s, "core_s": b - a + 1, "stages_done": ["front"] if s in ("done", "pulled", "stitched") else [],
                   "estimate": {"total_min": round((b - a + 9) * 1.35, 1)}, **x}
        if s == "running":
            cs[cid]["heartbeat"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now - 12))
            cs[cid]["started"] = now - 60 * (10 + 3 * len(pods))
            cs[cid]["stages_done"] = ["front"] if x.get("step") in ("fit", "refine") else []
            if x.get("step") == "refine":
                cs[cid]["stages_done"] = ["front", "fit"]
    for pid, name, cids in (("demo4090a", "hm-farm-demo-1", [c for c, v in cs.items() if v.get("pod") == "demo4090a"]),
                            ("demo4090b", "hm-farm-demo-2", [c for c, v in cs.items() if v.get("pod") == "demo4090b"]),
                            ("demo4090c", "hm-farm-demo-3", [c for c, v in cs.items() if v.get("pod") == "demo4090c"])):
        if cids:
            pods[pid] = {"id": pid, "name": name, "gpu": "NVIDIA GeForce RTX 4090", "price_hr": 0.69, "state": "running",
                         "created_ts": now - 42 * 60, "chunks": cids, "dc": "EU-RO-1", "kind": "gpu", "kill_at": now + 70 * 60}
    jobs = {"final": {"id": "final", "kind": "stitch_analyse", "status": "planned", "after": [c for c in cs]}} if chunks else {}
    for cid, c in cs.items():
        jobs[cid] = {"id": cid, "chunk": cid, "kind": "full", "status": c["status"], "after": []}
    state = {"schema": "dancecap.farm.state/1", "dance": dance, "demo": True, "created": cov["updated"], "updated": cov["updated"],
             "world": {"id": "demo00000000", "pose_id": "demopose0000", "sealed": True}, "pods": pods, "chunks": cs, "jobs": jobs,
             "events": [{"ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now - 60 * k)), "t": now - 60 * k, "kind": kd, "msg": m}
                        for k, kd, m in ((1, "pull", "g2580 pulled and verified"), (3, "chunk", "g1590: queued -> running"),
                                         (6, "pod", "pod demo4090b: starting -> running"), (9, "plan", f"{len(cs)} chunk(s) planned (demo)"))],
             "spend": {"spent": 1.84, "updated": cov["updated"]},
             "watcher": {"heartbeat": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now - 20)), "pid": None},
             "budget": {"cap": 80.0, "spent": 51.3, "remaining": 28.7, "balance": 39.4, "level": "green", "messages": [],
                        "available_usd": 24.1, "open_rate_hr": 2.07, "worst_open_usd": 4.6}}
    fd = paths.work() / dance / "farm"
    write_json_atomic(fd / "coverage.json", cov)
    write_json_atomic(fd / "state.json", state)
    return {"dance": dance, "scenario": scenario, "counts": counts, "dir": paths.rel_display(fd)}


def remove_demo(dance: str = "farm_demo") -> bool:
    d = paths.work() / dance
    if d.is_dir() and (d / "farm" / "state.json").exists():
        st = json.loads((d / "farm" / "state.json").read_text(encoding="utf-8"))
        if st.get("demo"):
            shutil.rmtree(d)
            return True
    return False


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m dancecap.pipeline farm-demo")
    ap.add_argument("--dance", default="farm_demo")
    ap.add_argument("--scenario", default="mostly_partial", choices=SCENARIOS)
    ap.add_argument("--remove", action="store_true")
    a = ap.parse_args(argv)
    if a.remove:
        print("removed" if remove_demo(a.dance) else "nothing to remove (only a dir whose state.json is a demo is removed)")
        return 0
    print(json.dumps(write_demo(a.dance, a.scenario)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
