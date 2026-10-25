"""Low-confidence stretches of the single-camera john_lula fit (J4, 2026-10-11) and the flag they put on the analyses.

    python -m dancecap.jl_lowconf annotate [--take john_lula]     # adds "low_confidence" to the timing / counterbalance / floorcraft results (idempotent), writes lowconf.json
    python -m dancecap.jl_lowconf show

Sources (all written by the pose stage, nothing is guessed here): the fit's own `absent` runs (no 2D evidence fed the fit: the temporal prior's interpolation), the weak stretches of
review/weak.json (PROVISIONAL.md: dancer hidden behind the partner, reprojection >= 5 percent of the height, the depth-spike repair), and the zoom steps of the take TOML (the camera
re-focuses: the per-frame focal is least certain, the fit's 2D agreement worst; +-0.5 s). An event (touchdown, accent, plant, rotation, counterbalance interval) gets
`low_confidence = [reasons]` when its time falls inside a stretch of its dancer (or of both). Levels: interpolated | hidden | poor_2d | repaired | zoom_step.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

from .take import PACKAGE_ROOT

TAKE = "john_lula"
ZOOM_MARGIN_S = 0.5
ROLES = ("lead", "follow")


def _runs(mask: np.ndarray, t: np.ndarray):
    idx = np.flatnonzero(mask)
    runs: list[list[int]] = []
    for i in idx:
        if runs and i == runs[-1][1] + 1:
            runs[-1][1] = int(i)
        else:
            runs.append([int(i), int(i)])
    return [(float(t[a]), float(t[b])) for a, b in runs]


def intervals(work: Path | None = None, smplx_dir: str = "smplx_rf2_noslip") -> list[dict]:
    work = Path(work) if work else PACKAGE_ROOT / "work" / TAKE
    out: list[dict] = []
    for r in ROLES:
        p = work / smplx_dir / f"{r}.npz"
        if not p.exists():
            continue
        with np.load(p, allow_pickle=True) as z:
            t = np.asarray(z["t"], float)
            for a, b in _runs(np.asarray(z["absent"], bool), t):
                out.append({"from": round(a, 3), "to": round(b, 3), "role": r, "level": "interpolated", "reason": "no 2D evidence fed the fit: the temporal prior's interpolation"})
            if "depth_spike_repaired" in z.files:
                for a, b in _runs(np.asarray(z["depth_spike_repaired"], bool), t):
                    out.append({"from": round(a, 3), "to": round(b, 3), "role": r, "level": "repaired", "reason": "monocular depth spike replaced by a straight line"})
    wk = work / "review" / "weak.json"
    if wk.exists():
        for w in json.loads(wk.read_text(encoding="utf-8")):
            if w.get("chunk") == "all":
                continue
            a, b = (float(x) for x in w["t_s"])
            what = str(w.get("what", ""))
            role = "lead" if what.startswith("lead") else "follow" if what.startswith("follow") else "both"
            if "DEPTH SPIKE" in what:
                continue                                    # already from the repaired flags
            level = "hidden" if re.search(r"hidden|guess|skeleton on the partner", what) else "interpolated" if "interpolat" in what and "no 2D" in what else "poor_2d"
            if level == "interpolated":
                continue                                    # the fit's own `absent` runs (above) are the same stretches, per role and per frame
            out.append({"from": a, "to": b, "role": role, "level": level, "reason": re.sub(r"\s+", " ", what).strip()[:120]})
    toml = PACKAGE_ROOT / "takes" / f"{work.name}.toml"     # the take of `work` (was john_lula.toml for every take: the
    if toml.exists():                                        # trial capture listed john_lula's three zoom steps)
        import tomllib

        from .camera_types import cam_tables                 # the per-camera tables: not loc / map / x_cam
        cams = cam_tables(tomllib.loads(toml.read_text(encoding="utf-8")))
        for cam in cams.values():
            for a, b, ratio in cam.get("zoom_events", []):
                out.append({"from": round(a - ZOOM_MARGIN_S, 3), "to": round(b + ZOOM_MARGIN_S, 3), "role": "both", "level": "zoom_step",
                            "reason": f"camera zoom step x{ratio:g}: the per-frame focal and the 2D agreement are least certain"})
    out.sort(key=lambda x: (x["from"], x["to"], x["role"]))
    return out


def reasons_at(ivs: list[dict], t0: float, t1: float | None, role: str | None) -> list[str]:
    """reasons of every stretch that overlaps [t0, t1] (an instant when t1 is None) and concerns `role` (None = any role)"""
    t1 = t0 if t1 is None else t1
    got = []
    for iv in ivs:
        if iv["role"] not in ("both", role) and role is not None:
            continue
        if iv["to"] >= t0 and iv["from"] <= t1:
            got.append(f"{iv['level']}: {iv['reason']} ({iv['from']:.2f}-{iv['to']:.2f} s)")
    return got


def _flag_list(items: list, ivs: list[dict], role_key: str | None, t0_key: str, t1_key: str | None = None, role: str | None = None) -> int:
    n = 0
    for it in items:
        if not isinstance(it, dict) or it.get(t0_key) is None:
            continue
        t0 = float(it[t0_key])
        t1 = float(it[t1_key]) if t1_key and it.get(t1_key) is not None else None
        r = it.get(role_key) if role_key else role
        rs = reasons_at(ivs, t0, t1, r if r in ROLES else None)
        if rs:
            it["low_confidence"] = rs
            n += 1
        else:
            it.pop("low_confidence", None)
    return n


def _restamp(doc: dict, **shas) -> None:
    """The flags are added to files whose sha1 the later analyses record as their input (counterbalance.is_current / floorcraft.is_current): the beat grid and the lead's
    motion are unchanged, only `low_confidence` keys were added, so the recorded hashes are brought to the annotated files (and the note says so)."""
    src = doc.setdefault("source", {})
    for k, v in shas.items():
        if src.get(k) != v:
            src.setdefault("restamped", {})[k] = {"was": src.get(k), "now": v, "why": "low_confidence keys added by dancecap.jl_lowconf (grid and motion unchanged)"}
            src[k] = v


def annotate(take: str = TAKE) -> dict:
    from . import counterbalance as cb
    work = PACKAGE_ROOT / "work" / take
    ivs = intervals(work)
    rep: dict = {"intervals": ivs, "flagged": {}}
    p = work / "timing" / "timing.json"
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        n = 0
        for r in ROLES:
            n += _flag_list(d.get("footsteps", {}).get(r, []), ivs, None, "t", role=r)
        acc = d.get("accents", {})
        if isinstance(acc, dict):
            for r in ROLES:
                if isinstance(acc.get(r), list):
                    rep["flagged"][f"timing accents {r}"] = _flag_list(acc[r], ivs, None, "t", role=r)
        d["low_confidence_intervals"] = ivs
        rep["flagged"]["timing footsteps"] = n
        p.write_text(json.dumps(d, indent=1), encoding="utf-8")
    grid_sha = cb.grid_sha1(work)
    p = work / "counterbalance" / "counterbalance.json"
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        rep["flagged"]["counterbalance intervals"] = _flag_list(d.get("intervals", []), ivs, None, "t0", "t1")
        d["low_confidence_intervals"] = ivs
        _restamp(d, beats_sha1=grid_sha)
        p.write_text(json.dumps(d, indent=1), encoding="utf-8")
    p = work / "floorcraft" / "floorcraft.json"
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        for k, v in d.items():
            if isinstance(v, list) and v and isinstance(v[0], dict) and ("t0" in v[0] or "t" in v[0]):
                role_key = "role" if "role" in v[0] else None
                rep["flagged"][f"floorcraft {k}"] = _flag_list(v, ivs, role_key, "t0" if "t0" in v[0] else "t", "t1" if "t1" in v[0] else None)
        d["low_confidence_intervals"] = ivs
        cbp = work / "counterbalance" / "counterbalance.json"
        _restamp(d, beats_sha1=grid_sha, counterbalance_sha1=cb._sha1(cbp) if cbp.exists() else None)
        p.write_text(json.dumps(d, indent=1), encoding="utf-8")
    (work / "lowconf.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.jl_lowconf", description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=("annotate", "show"))
    ap.add_argument("--take", default=TAKE)
    a = ap.parse_args(argv)
    if a.cmd == "show":
        for iv in intervals(PACKAGE_ROOT / "work" / a.take):
            print(f"{iv['from']:7.2f} {iv['to']:7.2f}  {iv['role']:7} {iv['level']:12} {iv['reason']}")
        return 0
    rep = annotate(a.take)
    print(f"{len(rep['intervals'])} low-confidence stretches; flagged: {rep['flagged']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
