"""Verdict and per-camera table of the camera-tracking gate (reads the data written by gate_cameras / video_motion).

    python -m dancecap.cameras.gate_final refresh-loo [--cams 05,06]   # recompute the dancers' LOO of those cameras (cache)
    python -m dancecap.cameras.gate_final run                          # table, video motion, zoom events, images, verdict
    python -m dancecap.cameras.gate_final verdict                      # only the verdict + camera_table.md from the data

Per camera and 10-s block a block FAILS when any of:
  ok-rate < 0.95 (attached 0.90) | static px > 4 (main model) | inliers < 60 (main model) | dancers' residual (label-free,
  sync-corrected) > max(2 x larissa_kadu_38's, 25 px) | centre jitter p95 > max(3 x larissa_kadu_38's, 30 mm) |
  angular error against the video's own motion > 0.2 deg per 1/6 s pair | unexplained zoom > 8 % over 2 s
and WARNS from 0.08 deg / 5 % / dancers > 1.4 x larissa_kadu_38's. Consecutive failing blocks of a camera are one stretch.
The overall verdict is ready = no failing stretch left that is not explained by something outside the camera (labels).
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from . import gate_cameras as gc

MOTION_FAIL, MOTION_WARN = 0.2, 0.08
ZOOM_FAIL, ZOOM_WARN = 8.0, 5.0


def refresh_loo(cams: list[str], segs=gc.SEGS, log=print) -> None:
    """Recompute the dancers' LOO (sync-corrected) of `cams` in every segment and update the segment caches."""
    for seg in segs:
        p = gc.OUT / "data" / f"dancers_loo_sync{gc.CACHE_TAG}_{seg}.npz"
        cur = dict(np.load(p)) if p.exists() else {}
        new = gc.dancer_residuals(gc.seg_work(seg), cams=cams, log=log)
        cur.update(new)
        np.savez(p, **cur)


def _blocks(seg: str, cam: str, tab: dict, mot: dict) -> dict[float, dict]:
    """t0 -> merged row (table block + motion block) of one camera in one segment."""
    out: dict[float, dict] = {}
    for r in tab["segments"].get(seg, {}).get(cam, {}).get("blocks", []):
        if r["frames"] >= 15:
            out[r["t0"]] = dict(r, seg=seg)
    for b in (mot.get(seg, {}).get(cam, {}) or {}).get("blocks", []):
        if b["t0"] in out:
            out[b["t0"]].update({"motion_deg_med": b["motion_deg_med"], "unexplained_zoom_pct": b["unexplained_zoom_max_2s_pct"],
                                 "video_evidence": b.get("evidence")})
    return out


def judge_block(r: dict, cam: str, base: dict) -> tuple[list[str], list[str]]:
    """(failing reasons, warnings) of a merged block row."""
    att = cam in gc.ATTACHED
    fail, warn = [], []
    if r["ok_exported"] < (0.90 if att else 0.95):
        fail.append(f"ok-rate {r['ok_exported']:.2f}")
    if not att and r.get("static_px_med") is not None and r["static_px_med"] > 4.0:
        fail.append(f"static {r['static_px_med']:.1f} px")
    if not att and r.get("inliers_med") is not None and r["inliers_med"] < 60:
        fail.append(f"inliers {r['inliers_med']:.0f}")
    dm = r.get("dancers_best_px_med")
    b0 = base.get(cam, {}).get("dancers_best_px_med")
    b0 = 12.0 if b0 is None or not np.isfinite(b0) else b0
    if dm is not None:
        if dm > max(2.0 * b0, 25.0):
            fail.append(f"dancers {dm:.0f} px (limit {max(2.0 * b0, 25.0):.0f})")
        elif dm > max(1.4 * b0, 18.0):
            warn.append(f"dancers {dm:.0f} px")
    jb = base.get(cam, {}).get("centre_jitter_p95") or 10.0
    if not att and r["centre_jitter_mm_p95"] > max(3.0 * jb, 30.0):
        fail.append(f"centre jitter {r['centre_jitter_mm_p95']:.0f} mm")
    m = r.get("motion_deg_med")
    if m is not None and m > MOTION_WARN:
        (fail if m > MOTION_FAIL else warn).append(f"video-motion error {m:.2f} deg")
    ev = r.get("video_evidence")
    if ev is not None and ev < 0.25 and cam in gc.ZOOM_CAMS:
        warn.append(f"video evidence only {100 * ev:.0f} % of the pairs (people hide the room)")
    z = r.get("unexplained_zoom_pct")
    if z is not None and z > ZOOM_WARN:
        (fail if z > ZOOM_FAIL else warn).append(f"unexplained zoom {z:.1f} %")
    return fail, warn


def stretches(rows: list[tuple[float, list[str]]], gap: float = 10.0) -> list[dict]:
    """Merge consecutive blocks with reasons into stretches [t0, t1]."""
    out: list[dict] = []
    for t0, why in sorted(rows):
        if not why:
            continue
        if out and t0 <= out[-1]["t1"] + 1e-6 + (gap - 10.0):
            out[-1]["t1"] = t0 + 10.0
            out[-1]["reasons"] = sorted(set(out[-1]["reasons"]) | set(why))
        else:
            out.append({"t0": t0, "t1": t0 + 10.0, "reasons": sorted(set(why))})
    return out


def build_verdict() -> dict:
    tab = json.loads((gc.OUT / "data" / f"table{gc.CACHE_TAG}.json").read_text())
    mot = json.loads((gc.OUT / "data" / "motion.json").read_text())
    base = tab["baseline"]
    zev = (gc.OUT / "data" / "zoom_evidence.json")
    zev = json.loads(zev.read_text()) if zev.exists() else {}
    overlaps = json.loads((gc.OUT / "data" / "overlap_consistency.json").read_text()) if (gc.OUT / "data" / "overlap_consistency.json").exists() else {}
    typed = gc.typed_focal_check()
    repairs = {}
    for seg in gc.SEGS:
        rp = gc.seg_work(seg) / "cameras" / "video_repairs.json"
        if rp.exists():
            repairs[seg] = json.loads(rp.read_text())
    cams_out, failing, warning = {}, [], []
    for cam in gc.CAMS:
        per_seg, fail_rows, warn_rows = {}, {}, {}
        for seg in gc.SEGS:
            tc = tab["segments"].get(seg, {}).get(cam)
            if not tc:
                continue
            blocks = _blocks(seg, cam, tab, mot)
            for t0, r in blocks.items():
                f, w = judge_block(r, cam, base)
                fail_rows.setdefault(t0, set()).update(f)
                warn_rows.setdefault(t0, set()).update(w)
            fr = tc["focal"]
            ms = (mot.get(seg, {}).get(cam) or {})
            per_seg[seg] = {"frames": tc["frames"], "t": [round(tc["t0"], 2), round(tc["t1"], 2)], "ok_raw": tc["ok_raw"],
                            "ok_exported": tc["ok_exported"], "recentred": tc["recentred"], "camera_type": tc["camera_type"],
                            "f_over_maxdim": fr["f_over_maxdim"], "zoom_events_exported": len(fr["zoom_events"]),
                            "focal_spikes": fr["spikes_outside_zoom"], "focal_plausible": fr["plausible"],
                            "motion_deg_med": ms.get("motion_deg_med"), "motion_deg_p95": ms.get("motion_deg_p95"),
                            "unexplained_zoom_max_2s_pct": ms.get("unexplained_zoom_max_2s_pct")}
        st_fail = stretches([(t, sorted(v)) for t, v in fail_rows.items()])
        st_warn = stretches([(t, sorted(v)) for t, v in warn_rows.items() if not fail_rows.get(t)])
        for s_ in st_fail:
            s_["segments"] = sorted(seg for seg in gc.SEGS if any(s_["t0"] <= t0 < s_["t1"] for t0 in _blocks(seg, cam, tab, mot)))
        cams_out[cam] = {"attached": cam in gc.ATTACHED, "zoom_declared": cam in gc.ZOOM_CAMS, "segments": per_seg,
                         "failing_stretches": st_fail, "warning_stretches": st_warn,
                         "video_events": (zev.get(cam) or {}).get("video_events"),
                         "pairing": (zev.get(cam) or {}).get("pairing"), "baseline_38": base.get(cam)}
        failing += [dict(cam=cam, **s) for s in st_fail]
        warning += [dict(cam=cam, **s) for s in st_warn]
    excluded = [dict(e, seg=seg) for seg in gc.SEGS for e in gc.excluded_spans(seg)]
    return {"cameras": cams_out, "failing_stretches": failing, "warning_stretches": warning, "typed_focal": typed,
            "excluded_spans": excluded,
            "overlap_consistency": overlaps, "repairs": repairs, "ready": not failing}


def camera_table(v: dict) -> str:
    """Markdown per-camera table of the whole dance."""
    L = ["# Camera tracking gate: per-camera table (larissa_kadu_full, reference clock)", "",
         "Per camera over the whole dance: worst / median over its 10-s blocks and segments. 'LOO' = dancers triangulated by the other main-model "
         "cameras, projected through the exported camera (sync-corrected), limit max(2 x larissa_kadu_38, 25 px). "
         "'video err' = angular error of the exported cameras against the video's own frame-to-frame motion, per 1/6 s pair. "
         "'unexpl. zoom' = zoom the video shows that the exported cameras do not reproduce (max over 2 s).", "",
         "| cam | type | frames | ok raw / exp | f / max dim | zoom events video / exported | video err med (p95) deg | unexpl. zoom max % | static px med | LOO px med (38) | worst block LOO px | centre jitter p95 mm | recentred | verdict |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    tab = json.loads((gc.OUT / "data" / f"table{gc.CACHE_TAG}.json").read_text())
    for cam, c in v["cameras"].items():
        segs = c["segments"]
        fr = sum(s["frames"] for s in segs.values())
        okr = np.average([s["ok_raw"] for s in segs.values()], weights=[s["frames"] for s in segs.values()])
        oke = np.average([s["ok_exported"] for s in segs.values()], weights=[s["frames"] for s in segs.values()])
        f_lo = min(s["f_over_maxdim"][0] for s in segs.values())
        f_hi = max(s["f_over_maxdim"][1] for s in segs.values())
        rows = [r for seg in segs for r in tab["segments"][seg][cam]["blocks"] if r["frames"] >= 15]
        static = [r["static_px_med"] for r in rows if r.get("static_px_med") is not None]
        loo = [r["dancers_best_px_med"] for r in rows if r.get("dancers_best_px_med") is not None]
        jit = [r["centre_jitter_mm_p95"] for r in rows]
        mot = [s["motion_deg_med"] for s in segs.values() if s.get("motion_deg_med") is not None]
        motp = [s["motion_deg_p95"] for s in segs.values() if s.get("motion_deg_p95") is not None]
        unx = [s["unexplained_zoom_max_2s_pct"] for s in segs.values() if s.get("unexplained_zoom_max_2s_pct") is not None]
        ve = len(c.get("video_events") or [])
        ee = sum(s["zoom_events_exported"] for s in segs.values())
        bl = (c["baseline_38"] or {}).get("dancers_best_px_med")
        verdict = "PASS" if not c["failing_stretches"] else "FAIL " + "; ".join(f"{s['t0']:.0f}-{s['t1']:.0f} s" for s in c["failing_stretches"])
        if not c["failing_stretches"] and c["warning_stretches"]:
            verdict = "PASS (warn " + "; ".join(f"{s['t0']:.0f}-{s['t1']:.0f} s" for s in c["warning_stretches"]) + ")"
        L.append(f"| {cam}{' att.' if c['attached'] else ''} | {'zoom' if c['zoom_declared'] else 'no zoom'} | {fr} | {okr:.3f} / {oke:.3f} | "
                 f"{f_lo:.2f}-{f_hi:.2f} | {ve} / {ee} | {np.median(mot):.3f} ({max(motp):.3f}) | {max(unx) if unx else float('nan'):.1f} | "
                 f"{np.median(static) if static else float('nan'):.2f} | {np.median(loo) if loo else float('nan'):.1f} ({bl if bl is None else round(bl, 1)}) | "
                 f"{max(loo) if loo else float('nan'):.0f} | {max(jit):.0f} | {sum(s['recentred'] for s in segs.values())} | {verdict} |")
    L += ["", "## Failing stretches", ""]
    if not v["failing_stretches"]:
        L.append("None.")
    for s in v["failing_stretches"]:
        L.append(f"- cam {s['cam']}, {s['t0']:.0f}-{s['t1']:.0f} s: " + ", ".join(s["reasons"]))
    L += ["", "## Spans excluded from the fit (vetoed, not judged)", ""]
    for e in v.get("excluded_spans", []):
        L.append(f"- segment {e['seg']} cam {e['cam']}, {e['t0']:.1f}-{e['t1']:.1f} s: {e['why']}")
    if not v.get("excluded_spans"):
        L.append("None.")
    L += ["", "## Warning stretches (pass, but close to a limit)", ""]
    for s in v["warning_stretches"]:
        L.append(f"- cam {s['cam']}, {s['t0']:.0f}-{s['t1']:.0f} s: " + ", ".join(s["reasons"]))
    L += ["", "## Typed focal check (no-zoom handheld cameras against larissa_kadu_38's median focal, rule within 1 %)", ""]
    for k, t in v["typed_focal"].items():
        L.append(f"- {k}: {t['diff_pct']:+.2f} % ({'ok' if t['within_1pct'] else 'OUTSIDE'}), k1 {t['k1_median']:+.4f}, model {t['type_model']}")
    return "\n".join(L) + "\n"


def run_all() -> dict:
    from . import gate_cameras_plots as pl
    from . import video_motion as vm
    gc.OUT.mkdir(parents=True, exist_ok=True)
    gc.run_table(("38",) + gc.SEGS)
    vm.run_motion()
    gc.run_zoom(("a", "38", "b", "c", "d"))
    (gc.OUT / "data" / "overlap_consistency.json").write_text(json.dumps(gc.overlap_consistency(), indent=1))
    v = build_verdict()
    (gc.OUT / "data" / "verdict.json").write_text(json.dumps(v, indent=1, default=gc._np))
    (gc.OUT / "camera_table.md").write_text(camera_table(v))
    pl.focal_figures()
    pl.heatmap_figure()
    pl.overlap_figure()
    return v


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m dancecap.cameras.gate_final", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["refresh-loo", "run", "verdict"])
    ap.add_argument("--cams", default="05,06")
    a = ap.parse_args(argv)
    if a.what == "refresh-loo":
        refresh_loo(a.cams.split(","))
    elif a.what == "run":
        v = run_all()
        print(json.dumps({"ready": v["ready"], "failing": v["failing_stretches"]}, indent=1))
    else:
        v = build_verdict()
        (gc.OUT / "data" / "verdict.json").write_text(json.dumps(v, indent=1, default=gc._np))
        (gc.OUT / "camera_table.md").write_text(camera_table(v))
        print(json.dumps({"ready": v["ready"], "failing": v["failing_stretches"]}, indent=1))


if __name__ == "__main__":
    main()
