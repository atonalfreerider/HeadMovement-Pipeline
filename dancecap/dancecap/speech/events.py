"""Laughter, applause and audience reactions on the MASTER clock, from the per-phone AST tagging (pod stage `events`).

    python -m dancecap.speech events      # writes work/<take>/speech/events_master.json

Each phone's AudioSet tagging windows (3 s, hop 1 s, source time) are moved to the master clock and spread over a 0.5 s grid with a
triangular weight (a window's score counts fully at its centre and half at its edges, so a short laugh is not smeared over the
whole window). Per class group (laughter, applause, cheering, crowd chatter) and per grid cell the phones' scores are combined:

  peak      the highest phone score (the phone nearest to the source hears it best)
  support   the number of phones that tag the cell above `support_p`
  covered   the number of phones that have audio at that master time

A cell is an event cell when peak >= `p_hi` AND support >= min(2, covered): something only ONE of several phones hears (handling noise,
a hand over the microphone) is rejected. Neighbouring event cells (gap <= 1 s) merge; events shorter than `min_s` are dropped. The 3-8 Hz
envelope modulation of the loudest phone (laughter bursts at ~5 Hz) is reported as supporting evidence, it does not decide.
These are CANDIDATES for the transcript's reaction tokens (laughter -> U+1F602 at the owner's two jokes); nothing here is labelled
a joke, that is identity.find_jokes' job (text + speaker + a laugh within a few seconds).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

GROUPS = {
    "laughter": ["Laughter", "Baby laughter", "Giggle", "Snicker", "Belly laugh", "Chuckle, chortle"],
    "applause": ["Applause", "Clapping"],
    "cheering": ["Cheering", "Whoop"],
    "crowd": ["Crowd", "Hubbub, speech noise, speech babble"],
}
GRID = 0.5


def window_group_score(p: dict, labels: list[str]) -> float:
    return max((float(p.get(l, 0.0)) for l in labels), default=0.0)


def spread_windows(rows: list[tuple[float, float, float]], t_lo: float, n: int, grid: float = GRID) -> np.ndarray:
    """rows [(m0, m1, score)] on the master clock -> (n,) grid values: the max over windows of score x triangular weight (1 at the centre,
    0.5 at the edges)."""
    out = np.zeros(n)
    for m0, m1, s in rows:
        if s <= 0:
            continue
        c = 0.5 * (m0 + m1)
        half = max(0.5 * (m1 - m0), 1e-6)
        a, b = max(0, int((m0 - t_lo) / grid)), min(n, int(np.ceil((m1 - t_lo) / grid)))
        for i in range(a, b):
            tc = t_lo + (i + 0.5) * grid
            w = 1.0 - 0.5 * min(1.0, abs(tc - c) / half)
            out[i] = max(out[i], s * w)
    return out


def detect(per_cam: dict[str, np.ndarray], covered: dict[str, np.ndarray], p_hi: float = 0.4, support_p: float = 0.12, min_s: float = 0.6,
           merge_gap: float = 1.0, grid: float = GRID) -> list[dict]:
    """Event cells -> events from the per-phone grids of ONE group. `covered[cam]` is a boolean grid of the phone's coverage."""
    cams = sorted(per_cam)
    if not cams:
        return []
    n = len(next(iter(per_cam.values())))
    stack = np.stack([per_cam[c] for c in cams])
    cov = np.stack([covered[c] for c in cams])
    peak = np.where(cov, stack, 0.0).max(axis=0)
    support = ((stack >= support_p) & cov).sum(axis=0)
    covered_n = cov.sum(axis=0)
    cell = (peak >= p_hi) & (support >= np.minimum(2, np.maximum(covered_n, 1)))
    events, i = [], 0
    while i < n:
        if not cell[i]:
            i += 1
            continue
        j = i
        gap = 0
        k = i
        while k < n and gap * grid <= merge_gap:
            if cell[k]:
                j = k
                gap = 0
            else:
                gap += 1
            k += 1
        events.append((i, j + 1))
        i = j + 1
    out = []
    for a, b in events:
        if (b - a) * grid < min_s:
            continue
        seg = slice(a, b)
        cam_peaks = {c: round(float(np.where(covered[c][seg], per_cam[c][seg], 0.0).max()), 3) for c in cams}
        out.append({"i0": a, "i1": b, "peak": round(float(peak[seg].max()), 3), "mean": round(float(peak[seg].mean()), 3),
                    "support_max": int(support[seg].max()), "covered_min": int(covered_n[seg].min()), "cams": cam_peaks})
    return out


# (p_hi, support_p) per group. AST is trained on loud, clean clips: a chuckle among talking teachers reaches only 0.1-0.15 on the nearest
# phone (measured on this class), so laughter needs a far lower bar than applause; the several-phones rule keeps it honest.
THRESHOLDS = {"laughter": (0.10, 0.04), "applause": (0.30, 0.10), "cheering": (0.30, 0.10), "crowd": (0.40, 0.12)}


def run(raw_dir: Path, clocks: dict, p_hi: float | None = None, groups: dict | None = None, thresholds: dict | None = None) -> dict:
    groups = groups or GROUPS
    thresholds = thresholds or THRESHOLDS
    cams: dict[str, dict] = {}
    for d in sorted(Path(raw_dir).glob("[0-9][0-9]")):
        f = d / "events.json"
        clk = clocks.get(d.name)
        if f.exists() and clk is not None:
            cams[d.name] = {"doc": json.loads(f.read_text(encoding="utf-8")), "clock": clk}
    if not cams:
        return {"events": [], "note": "no events.json with a clock"}
    t_lo = min(c["clock"].to_master(0.0) for c in cams.values())
    t_hi = max(c["clock"].to_master(c["doc"]["duration"]) for c in cams.values())
    t_lo = np.floor(t_lo)
    n = int(np.ceil((t_hi - t_lo) / GRID))
    covered = {}
    for cam, c in cams.items():
        a, b = c["clock"].to_master(0.0), c["clock"].to_master(c["doc"]["duration"])
        centres = t_lo + (np.arange(n) + 0.5) * GRID
        covered[cam] = (centres >= a) & (centres <= b)
    events = []
    for gname, labels in groups.items():
        grids = {}
        for cam, c in cams.items():
            rows = [(c["clock"].to_master(w["t0"]), c["clock"].to_master(w["t1"]), window_group_score(w["p"], labels)) for w in c["doc"]["windows"]]
            grids[cam] = spread_windows(rows, t_lo, n)
        hi, sup = thresholds.get(gname, (p_hi or 0.4, 0.12))
        for e in detect(grids, covered, p_hi=hi, support_p=sup):
            e["kind"] = gname
            e["t0"] = round(t_lo + e["i0"] * GRID, 2)
            e["t1"] = round(t_lo + e["i1"] * GRID, 2)
            loud = max(e["cams"], key=lambda c: e["cams"][c])
            e["best_cam"] = loud
            am = []
            for w in cams[loud]["doc"]["windows"]:
                m0, m1 = cams[loud]["clock"].to_master(w["t0"]), cams[loud]["clock"].to_master(w["t1"])
                if m1 > e["t0"] and m0 < e["t1"]:
                    am.append(w.get("am_3_8", 0.0))
            e["am_3_8_mean"] = round(float(np.mean(am)), 3) if am else None
            events.append(e)
    events.sort(key=lambda e: (e["t0"], e["kind"]))
    for k, e in enumerate(events):
        e["id"] = f"ev{k:03d}"
    return {"grid_s": GRID, "thresholds": {g: list(thresholds.get(g, (p_hi or 0.4, 0.12))) for g in groups}, "cams": sorted(cams), "events": events,
            "counts": {g: sum(1 for e in events if e["kind"] == g) for g in groups},
            "seconds": {g: round(sum(e["t1"] - e["t0"] for e in events if e["kind"] == g), 1) for g in groups}}


def music_presence(raw_dir: Path, clocks: dict, labels: tuple[str, ...] = ("Music", "Musical instrument", "Singing")) -> dict:
    """Share of windows per phone whose music-like tag is >= 0.3 (the class has no soundtrack by the owner's brief; this checks it)."""
    out = {}
    for d in sorted(Path(raw_dir).glob("[0-9][0-9]")):
        f = d / "events.json"
        if not f.exists():
            continue
        wins = json.loads(f.read_text(encoding="utf-8"))["windows"]
        hit = [max((w["p"].get(l, 0.0) for l in labels), default=0.0) >= 0.3 for w in wins]
        out[d.name] = round(float(np.mean(hit)), 3) if hit else None
    return out
