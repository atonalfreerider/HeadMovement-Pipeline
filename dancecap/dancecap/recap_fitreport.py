"""One table of the recap chunk fits: per-chunk quality numbers (pose_quality) against gates taken from the demo's fitted segments (agent F1, 2026-10-09).

    python -m dancecap.recap_fitreport [--ids g0000,g1588,...] [--out work/larissa_kadu_recap/chunks/fit_quality.json]

For every chunk take work/larissa_kadu_recap_<id>/ whose quality/quality_rf2_noslip.json exists: reprojection of the 512 MAMMA points per camera and dancer
(median / p90 over frames), the all-cameras per-frame median, the feet, the pelvis jitter, the lowest sole above / below the floor, the crown-to-sole rest
height against the stature in the take's dancer table. A gate FAILS a chunk (reported, never hidden); the values come from the demo's segment a
(work/larissa_kadu_full_a/quality: all-cameras reprojection lead 5.3 / follow 5.8 px, p90 7.6 / 8.4, per camera medians 3-8 px, rest height within 1 %).
The recap's phones are handheld (the demo's were mostly on tripods) and two of them are attached by rotation only, so the per-camera gate is looser for
05 / 06 and the gates are a reading aid, not a cut-off for use.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .take import PACKAGE_ROOT

DANCE = os.environ.get("HM_FRONT_DANCE", "larissa_kadu_recap")        # C3 (additive): the same report for carlos_aline
ATTACHED = ("05", "06") if DANCE == "larissa_kadu_recap" else ()
GATES = {
    "allcam_median_px": 8.0,         # per-frame median over all cameras, median over frames (demo segment a: 5.3 / 5.8)
    "allcam_p90_px": 12.0,           # p90 over frames of the same (demo: 7.6 / 8.4)
    "cam_median_px": 10.0,           # per camera, main phones (demo: 3.1-6.7)
    "cam_median_px_attached": 14.0,  # per camera, attached phones 05 / 06 (rotation-only geometry)
    "rest_height_rel": 0.03,         # |crown-to-sole - stature| / stature (demo: 0.9 % / 0.5 %)
    "frac_below_15mm": 0.02,         # lowest sole more than 15 mm below the floor
    "jitter_rms_mm": 2.0,            # 4.5-12 Hz pelvis band, whole (demo: 0.5 / 0.7)
}


def _num(x, default=None):
    try:
        v = float(x)
        return v if v == v else default
    except (TypeError, ValueError):
        return default


def summarise(q: dict) -> dict:
    """The numbers of one pose_quality json that the gates read: {dancer: {...}} plus the frame count and span."""
    out = {"frames": q.get("frames"), "span_s": q.get("span_s"), "dancers": {}}
    whole = (q.get("reproj_px") or {}).get("whole") or {}
    allc = (q.get("reproj_px") or {}).get("all_cams_per_frame_median") or {}
    feet = (q.get("reproj_feet_px") or {}).get("all_cams_per_frame_median") or {}
    for role in ("lead", "follow"):
        cams = {c: _num((v.get(role) or {}).get("median")) for c, v in whole.items() if isinstance(v, dict) and v.get(role)}
        p90 = {c: _num((v.get(role) or {}).get("p90")) for c, v in whole.items() if isinstance(v, dict) and v.get(role)}
        fl = (q.get("floor") or {}).get(role) or {}
        rest = (q.get("rest_height_m") or {}).get(role) or {}
        jit = (q.get("jitter_4p5_12hz_mm") or {}).get(role) or {}
        out["dancers"][role] = {
            "allcam_median_px": _num((allc.get(role) or {}).get("median")), "allcam_p90_px": _num((allc.get(role) or {}).get("p90")),
            "feet_allcam_median_px": _num((feet.get(role) or {}).get("median")),
            "cam_median_px": cams, "cam_p90_px": p90,
            "rest_height_rel": _num(rest.get("rel_vs_target")), "rest_height_m": _num(rest.get("fit")),
            "frac_below_15mm": _num(fl.get("frac_below_15mm")), "frac_above_30mm": _num(fl.get("frac_above_30mm")),
            "jitter_rms_mm": _num(jit.get("whole_rms")),
        }
    return out


def gate_failures(summary: dict, gates: dict = GATES) -> list[str]:
    """Human-readable failures of one summarise() result, [] when every gate passes."""
    bad = []
    for role, d in summary["dancers"].items():
        for key in ("allcam_median_px", "allcam_p90_px", "frac_below_15mm", "jitter_rms_mm"):
            v = d.get(key)
            if v is None:
                bad.append(f"{role}: {key} missing")
            elif v > gates[key]:
                bad.append(f"{role}: {key} {v:.2f} > {gates[key]}")
        r = d.get("rest_height_rel")
        if r is not None and abs(r) > gates["rest_height_rel"]:
            bad.append(f"{role}: rest height off by {100 * r:+.1f} % (> {100 * gates['rest_height_rel']:.0f} %)")
        for c, v in sorted(d["cam_median_px"].items()):
            lim = gates["cam_median_px_attached"] if c in ATTACHED else gates["cam_median_px"]
            if v is not None and v > lim:
                bad.append(f"{role}: camera {c} median {v:.1f} px > {lim}")
    return bad


def chunk_ids() -> list[str]:
    plan = PACKAGE_ROOT / "work" / DANCE / "farm" / "plan.json"
    if plan.exists():
        return [c["id"] for c in json.loads(plan.read_text(encoding="utf-8")).get("chunks", [])]
    return sorted(p.name.split("_")[-1] for p in (PACKAGE_ROOT / "work").glob(f"{DANCE}_g*"))


def report(ids: list[str] | None = None) -> dict:
    rows = {}
    for cid in ids or chunk_ids():
        p = PACKAGE_ROOT / "work" / f"{DANCE}_{cid}" / "quality" / "quality_rf2_noslip.json"
        if not p.exists():
            rows[cid] = {"state": "no quality json yet"}
            continue
        s = summarise(json.loads(p.read_text(encoding="utf-8")))
        bad = gate_failures(s)
        rows[cid] = {"state": "pass" if not bad else "FAIL", "failures": bad, **s}
    return {"dance": DANCE, "gates": GATES, "chunks": rows}


def table(rep: dict) -> str:
    lines = ["| chunk | state | frames | lead allcam med / p90 px | follow allcam med / p90 px | worst camera (lead, follow) | rest height lead / follow | jitter mm |",
             "|---|---|---|---|---|---|---|---|"]
    for cid, r in rep["chunks"].items():
        if "dancers" not in r:
            lines.append(f"| {cid} | {r['state']} | | | | | | |")
            continue
        L, F = r["dancers"]["lead"], r["dancers"]["follow"]

        def worst(d):
            cm = {c: v for c, v in d["cam_median_px"].items() if v is not None}
            if not cm:
                return "-"
            c = max(cm, key=cm.get)
            return f"{c} {cm[c]:.1f}"

        def f(x, n=1):
            return "-" if x is None else f"{x:.{n}f}"
        lines.append(f"| {cid} | {r['state']} | {r['frames']} | {f(L['allcam_median_px'])} / {f(L['allcam_p90_px'])} | {f(F['allcam_median_px'])} / {f(F['allcam_p90_px'])} | "
                     f"{worst(L)}, {worst(F)} | {f(None if L['rest_height_rel'] is None else 100 * L['rest_height_rel'])} % / "
                     f"{f(None if F['rest_height_rel'] is None else 100 * F['rest_height_rel'])} % | {f(L['jitter_rms_mm'], 2)} / {f(F['jitter_rms_mm'], 2)} |")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.recap_fitreport", description=__doc__.split("\n")[0])
    ap.add_argument("--ids", help="comma list of chunk ids (default: the planned chunks)")
    ap.add_argument("--out", default=str(PACKAGE_ROOT / "work" / DANCE / "chunks" / "fit_quality.json"))
    a = ap.parse_args(argv)
    rep = report(a.ids.split(",") if a.ids else None)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(table(rep))
    for cid, r in rep["chunks"].items():
        for b in r.get("failures", []):
            print(f"  {cid}: {b}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
