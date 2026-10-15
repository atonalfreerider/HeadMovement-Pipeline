"""May a run's generated views enter MAMMA? Numeric checks + Jev judgments -> work/<take>/mono4d/<run>/quality_gate.json.

Reads whatever the run has (each missing input is reported, never assumed good):
    result.json     the delivery: kind, dynamic, world_from_scene (rot_err / rms when solved from the input poses)
    judge.json      judge.judge_views: per view bad-frame fraction and ok
    views2d.json    {vcam: holdout.compare_views} for views rendered at held-out real phones (evaluation takes)
    holdout.json    {"verdict": holdout.verdict, ...} (evaluation takes)

Rules (GATE, provisional until the first real delivery is measured):
    - the delivery must be dynamic: a static world (World API today) never enters MAMMA
    - a solved scene frame must agree with our track: rotation p95 <= 3 deg, centre rms <= 0.05 m (scaled to metres)
    - a view Jev flags on more than 10 % of its sampled frames is dropped; a view that was not judged is dropped
    - where we CAN check the provider (generated views at real phones' poses), it must be right there: extra people on
      <= 5 % of frames, PCK@0.1 median >= 0.8, dancer-mask IoU median >= 0.6. Failing that fails the whole run:
      trust nothing it shows where we cannot check
    - at least 4 ring views survive and they span >= 180 degrees of azimuth around the couple
    - on an evaluation take, the hold-out verdict must say the route beats the one-camera baseline
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import rig

GATE = {"rot_err_p95_deg": 3.0, "align_rms_m": 0.05, "extra_people_frac": 0.05, "pck_median": 0.8,
        "mask_iou_median": 0.6, "min_ring_views": 4, "min_azimuth_span_deg": 180.0}


def azimuth_span_deg(centres: np.ndarray, around: np.ndarray | None = None) -> float:
    """Angular span of camera centres (N,3) around a vertical axis through `around` (x, z): 360 minus the largest gap."""
    c = np.asarray(centres, float)
    if len(c) < 2:
        return 0.0
    o = np.asarray(around, float) if around is not None else c[:, [0, 2]].mean(0)
    az = np.sort(np.degrees(np.arctan2(c[:, 0] - o[0], c[:, 2] - o[1])) % 360.0)
    gaps = np.diff(np.concatenate([az, az[:1] + 360.0]))
    return float(360.0 - gaps.max())


def _read(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def decide(rd: Path, th: dict | None = None) -> dict:
    th = {**GATE, **(th or {})}
    why, checks, notes = [], {}, []
    res = _read(rd / "result.json")
    if res is None:
        return {"admit": False, "why": ["no result.json (fetch first)"], "checks": {}, "thresholds": th}
    checks["kind"], checks["dynamic"] = res["kind"], res["dynamic"]
    if not res["dynamic"]:
        why.append(f"delivery is {res['kind']} (static): never MAMMA input")
    wfs = res.get("world_from_scene") or {}
    if "rot_err_deg" in wfs:
        rot, rms = wfs["rot_err_deg"]["p95"], float(wfs.get("rms", 0.0))  # umeyama rms is in world metres
        checks["alignment"] = {"rot_err_p95_deg": rot, "rms_m": rms}
        if rot > th["rot_err_p95_deg"]:
            why.append(f"scene frame: rotation error p95 {rot:.1f} deg > {th['rot_err_p95_deg']}")
        if rms > th["align_rms_m"]:
            why.append(f"scene frame: camera-centre rms {rms:.3f} m > {th['align_rms_m']}")
    targets = {s.name: s for s in rig.specs_from_trajectory(_read(rd / "request" / "target_trajectory.json"))} \
        if (rd / "request" / "target_trajectory.json").exists() else {}
    delivered = set((res.get("views") or {}).keys())
    judged = (_read(rd / "judge.json") or {}).get("views") or {}
    if not judged:
        notes.append("no judge.json: every view counts as unjudged")
    admitted, dropped = [], {}
    for name in sorted(delivered):
        j = judged.get(name)
        if j is None:
            dropped[name] = "not judged"
        elif not j.get("ok"):
            dropped[name] = (f"Jev flagged {j.get('bad')} of {j.get('judged')} frames" if j.get("bad_frac") is not None
                             else "judging failed")
        else:
            admitted.append(name)
    v2d = _read(rd / "views2d.json") or {}
    checks["held_out_views"] = {}
    for vcam, r in v2d.items():
        bad = []
        if (r.get("extra_people_frac") or 0.0) > th["extra_people_frac"]:
            bad.append(f"extra people on {100 * r['extra_people_frac']:.0f} % of frames")
        pk = (r.get("pck_0.1") or {}).get("median")
        if pk is None or pk < th["pck_median"]:
            bad.append(f"PCK@0.1 median {pk}")
        mi = (r.get("mask_iou") or {}).get("median")
        if mi is not None and mi < th["mask_iou_median"]:
            bad.append(f"mask IoU median {mi:.2f}")
        checks["held_out_views"][vcam] = bad or "ok"
        if bad:
            why.append(f"held-out view {vcam} disagrees with the real phone {r.get('cam')}: " + "; ".join(bad))
    ring = [n for n in admitted if targets.get(n) is not None and targets[n].kind == "ring"]
    all_ring = np.array([s.centres[0] for s in targets.values() if s.kind == "ring"])
    around = all_ring[:, [0, 2]].mean(0) if len(all_ring) else None  # the ring's own centre, not the survivors'
    span = azimuth_span_deg(np.array([targets[n].centres[0] for n in ring]), around) if ring else 0.0
    checks["ring_views"], checks["azimuth_span_deg"] = len(ring), round(span, 1)
    if len(ring) < th["min_ring_views"]:
        why.append(f"{len(ring)} ring views pass (< {th['min_ring_views']})")
    elif span < th["min_azimuth_span_deg"]:
        why.append(f"surviving ring views span {span:.0f} deg (< {th['min_azimuth_span_deg']:.0f})")
    ho = _read(rd / "holdout.json")
    if ho is not None:
        v = ho.get("verdict") or {}
        checks["holdout_verdict"] = v.get("route_beats_baseline")
        if not v.get("route_beats_baseline"):
            why.append("hold-out: the route does not beat the one-camera baseline (" + "; ".join(v.get("why", [])) + ")")
    out = {"admit": not why, "views_admitted": [n for n in admitted if targets.get(n) is None or targets[n].kind != "holdout"],
           "views_dropped": dropped, "why": why, "notes": notes, "checks": checks, "thresholds": th,
           "rule": "generated views are the provider's prior, not measurements"}
    (rd / "quality_gate.json").write_text(json.dumps(out, indent=1))
    return out

