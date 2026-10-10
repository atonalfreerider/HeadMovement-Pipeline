"""Record that a stitched take's smplx/ + refined folder are PROVISIONAL (a seam gate failed and the motion was promoted anyway).

    python -m dancecap.stitch_provisional <take> --decision "why it was promoted" [--weak weak.json] [--write]

The stitch job (`dancecap.stitch ... --force` on a CPU pod) writes the promoted arrays and stitch.json with the pod's own verdict
(`gate.passed` false, `outputs.forced` true). This tool (orchestration only, a few seconds, idempotent):
  1. checks the promoted arrays against the sha256 the pod recorded (stitch.json outputs.sha256),
  2. writes `flags.provisional` + `promotion` into stitch.json (the pod's gate verdict stays untouched: the farm coverage keeps the
     failing seams partial),
  3. writes work/<take>/PROVISIONAL.md: every seam with its blend window and what exceeded which gate (nothing hidden), the frames with
     fewer than three cameras, the known weak stretches given with --weak (JSON list of {chunk, t_s, what}), the improvement path
     (the 12-s boundary refits stitch.json names per failing seam) and the checksums.
Without --write it only prints the report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from .take import PACKAGE_ROOT


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def check_arrays(work: Path, st: dict) -> dict:
    """{relative path: {sha256, same_as_pod_record}} for every array the pod recorded."""
    out = {}
    for rel, want in (st.get("outputs", {}).get("sha256") or {}).items():
        p = work / rel
        got = sha256(p) if p.exists() else None
        out[rel] = {"sha256": got, "same_as_pod_record": got == want}
    return out


def seam_rows(st: dict) -> list[dict]:
    rows = []
    for s in st.get("seams", []):
        g = s.get("gate") or {}
        rf = s.get("refit") or {}
        rows.append({"name": s["name"], "window_t": s.get("window_t"), "gate_passed": bool(g.get("passed")), "exceeded": g.get("why", []),
                     "utilisation": s.get("utilisation"), "refit_window_t": rf.get("window_t"), "refit_window_k": rf.get("window_k")})
    return rows


def render(take: str, st: dict, rows: list[dict], arrays: dict, decision: str, weak: list[dict], recorded: str, prev: dict | None = None,
           improvement: str | None = None) -> str:
    n_fail = sum(1 for r in rows if not r["gate_passed"])
    grid = st.get("grid", {})
    cov = (st.get("coverage") or {}).get("frames", {})
    L = [f"# PROVISIONAL: the stitched whole-take motion ({take})", "",
         f"Recorded {recorded}. `smplx/` and `smplx_rf2_noslip/` here are the stitched arrays (T = {grid.get('T')} on the "
         f"{grid.get('hz')} Hz lattice, {grid.get('t', [None, None])[0]} to {grid.get('t', [None, None])[1]} s) and they are PROVISIONAL: the "
         f"seam gate FAILED on {n_fail} of {len(rows)} seams and the motion was promoted anyway ({decision}). The pod's verdict is untouched "
         "(`stitch.json` gate.passed = false, outputs.forced = true; the farm coverage keeps those cells partial).", "",
         "## What is off", "",
         "Disagreements between independently fitted MAMMA chunks inside the 2 s cross-fade windows (wrists, elbows and fingers first: the "
         "hands are the raw MAMMA hands, no hand refinement; some seams also exceed a body-joint or pelvis limit, listed below). Everywhere "
         "outside the blend windows the motion is the chunk fit, bit for bit (`stitch.json` verified_on_disk identical for every chunk, "
         "raw and refined). The shape is identical in every chunk (the shape job's betas are pinned and frozen: max |db| 0.0 at every seam).",
         "", "| seam | blend window (s) | gate | exceeded |", "|---|---|---|---|"]
    for r in rows:
        w = r["window_t"]
        L.append(f"| {r['name']} | {w[0]:.2f} to {w[1]:.2f} | {'pass' if r['gate_passed'] else 'FAIL'} | "
                 f"{'; '.join(r['exceeded']) or '-'} |" if w else f"| {r['name']} | - | {'pass' if r['gate_passed'] else 'FAIL'} | "
                 f"{'; '.join(r['exceeded']) or '-'} |")
    L += ["", f"Gates (mm, p95 unless noted): {json.dumps(st.get('thresholds'))}.", ""]
    ref = (st.get("refined") or {}).get("seams") or []
    if ref:
        L += ["Refined motion (`" + str((st.get("refined") or {}).get("dir")) + "`, informational; the raw gate decides), worst per seam and role:", ""]
        for s in ref:
            for role, v in s["roles"].items():
                L.append(f"- {s['name']} {role}: body p95 {v['body_p95_mm']} mm ({v['body_joint']}), elbow/wrist p95 {v['elbow_wrist_p95_mm']} mm "
                         f"({v['arm_joint']}), fingers rel. wrist p95 {v['fingers_rel_wrist_p95_mm']} mm, pelvis median {v['pelvis_median_mm']} mm")
        L.append("")
    L += ["## Where it shows", "",
          "Only inside the blend windows above (about 2 s each) as a wrist / elbow / finger pose, and for the seams that exceed a body limit as a "
          "few centimetres of shoulder / foot / pelvis disagreement, which the cross-fade spreads over the window. Coverage by frame "
          f"(complete = seams passed and >= 3 cameras for both dancers): complete {cov.get('complete')}, partial {cov.get('partial')}, none {cov.get('none')}.", ""]
    for role in ("lead", "follow"):
        fl = (st.get("flags") or {}).get(role) or {}
        c = fl.get("counts") or {}
        L.append(f"- {role}: absent {c.get('absent')}, mono {c.get('mono')}, two_view {c.get('two_view')}, multi {c.get('multi')} frames"
                 + (f"; two_view runs (s) {fl.get('two_view_runs_t')}" if fl.get("two_view_runs_t") else ""))
    if weak:
        L += ["", "## Known weak stretches (from the fit stage)", ""]
        L += [f"- {w.get('chunk')} {w.get('t_s')} s: {w.get('what')}" for w in weak]
    if prev:       # T1: the stitch this one replaced, seam by seam (the worst body / elbow-wrist / finger / pelvis measure of its two roles)
        L += ["", "## Before this stitch (status per seam)", "", "| seam | blend window (s) | gate | worst body p95 | worst elbow/wrist p95 | worst fingers p95 | worst pelvis median |", "|---|---|---|---|---|---|---|"]
        for s_ in prev.get("seams", []):
            cs = [v["checks"] for v in (s_.get("roles") or {}).values()]
            w = s_.get("window_t") or [0, 0]
            L.append(f"| {s_['name']} | {w[0]:.2f} to {w[1]:.2f} | {'pass' if (s_.get('gate') or {}).get('passed') else 'FAIL'} | "
                     + " | ".join(f"{max(c[k] for c in cs):.0f} mm" for k in ("body_p95_mm", "elbow_wrist_p95_mm", "fingers_rel_wrist_p95_mm", "pelvis_median_mm")) + " |")
        L.append("")
        L.append("Now (the same measures, this stitch):")
        L += ["", "| seam | gate | worst body p95 | worst elbow/wrist p95 | worst fingers p95 | worst pelvis median |", "|---|---|---|---|---|---|"]
        for s_ in st.get("seams", []):
            cs = [v["checks"] for v in (s_.get("roles") or {}).values()]
            L.append(f"| {s_['name']} | {'pass' if (s_.get('gate') or {}).get('passed') else 'FAIL'} | "
                     + " | ".join(f"{max(c[k] for c in cs):.0f} mm" for k in ("body_p95_mm", "elbow_wrist_p95_mm", "fingers_rel_wrist_p95_mm", "pelvis_median_mm")) + " |")
    L += ["", "## Improvement path", ""]
    if improvement:
        L += [improvement]
    else:
        L += ["12-s boundary refits (stitch.json seams[].refit): refit the window centred on each failing blend as an extra chunk (kind refit) "
              "and stitch again:", ""]
        for r in rows:
            if r["refit_window_t"]:
                L.append(f"- {r['name']}: refit {r['refit_window_t'][0]:.2f} to {r['refit_window_t'][1]:.2f} s (grid k {r['refit_window_k'][0]}..{r['refit_window_k'][1]})")
    L += ["", "After a passing re-stitch, delete this file and set `flags.provisional.status` to `superseded` in stitch.json.", "",
          "## Checksums (sha256)", ""]
    L += [f"- {rel}: {a['sha256']}" for rel, a in arrays.items()]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.stitch_provisional", description=__doc__.split("\n\n")[0])
    ap.add_argument("take")
    ap.add_argument("--decision", required=True, help="why the failed stitch was promoted (one sentence, goes into the record)")
    ap.add_argument("--weak", help="JSON file: list of {chunk, t_s, what} known weak stretches")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--prev", help="T1: the previous stitch.json: PROVISIONAL.md gets a before / now table per seam")
    ap.add_argument("--improvement", help="T1: text of the Improvement path section (default: the 12-s boundary refit windows stitch.json names)")
    a = ap.parse_args(argv)
    work = PACKAGE_ROOT / "work" / a.take
    st = json.loads((work / "stitch.json").read_text(encoding="utf-8"))
    if (st.get("gate") or {}).get("passed"):
        print(f"[provisional] {a.take}: the stitch gate PASSED, nothing is provisional")
        return 0
    arrays = check_arrays(work, st)
    bad = [k for k, v in arrays.items() if not v["same_as_pod_record"]]
    if bad:
        print(f"[provisional] REFUSED: arrays differ from the pod's record: {bad}")
        return 1
    weak = json.loads(Path(a.weak).read_text(encoding="utf-8")) if a.weak else []
    rows = seam_rows(st)
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    prev = json.loads(Path(a.prev).read_text(encoding="utf-8")) if a.prev else None
    md = render(a.take, st, rows, arrays, a.decision, weak, now, prev, a.improvement)
    if not a.write:
        print(md)
        return 0
    prov = {"status": "provisional", "recorded": now, "decision": a.decision, "promoted_by": "dancecap.stitch --force on a CPU pod; stitch.json gate.passed stays false",
            "seams": {r["name"]: {k: r[k] for k in ("gate_passed", "exceeded", "window_t", "refit_window_t", "refit_window_k")} for r in rows},
            "gates_mm": st.get("thresholds"), "weak_stretches": weak, "arrays": arrays,
            "improvement_path": "12-s boundary refits per failing seam (stitch.json seams[].refit), then stitch again"}
    st.setdefault("flags", {})["provisional"] = prov
    st["promotion"] = {"promoted": True, "provisional": True, "forced": bool((st.get("outputs") or {}).get("forced")), "note": "see PROVISIONAL.md"}
    tmp = work / "stitch.json.tmp"
    tmp.write_text(json.dumps(st, indent=1), encoding="utf-8")
    tmp.replace(work / "stitch.json")
    (work / "PROVISIONAL.md").write_text(md, encoding="utf-8")
    print(f"[provisional] wrote {work / 'PROVISIONAL.md'} and flags.provisional in stitch.json ({sum(not r['gate_passed'] for r in rows)} of {len(rows)} seams failed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
