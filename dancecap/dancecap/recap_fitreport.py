"""One table of the recap chunk fits: per-chunk quality numbers (pose_quality) against gates taken from the demo's fitted segments (agent F1, 2026-10-09).

    python -m dancecap.recap_fitreport [--ids g0000,g1588,...] [--out work/larissa_kadu_recap/chunks/fit_quality.json]

For every chunk take work/larissa_kadu_recap_<id>/ whose quality/quality_rf2_noslip.json exists: reprojection of the 512 MAMMA points per camera and dancer
(median / p90 over frames), the all-cameras per-frame median, the feet, the pelvis jitter, the lowest sole above / below the floor, the crown-to-sole rest
height against the stature in the take's dancer table. A gate FAILS a chunk (reported, never hidden); the values come from the demo's segment a
(work/larissa_kadu_full_a/quality: all-cameras reprojection lead 5.3 / follow 5.8 px, p90 7.6 / 8.4, per camera medians 3-8 px, rest height within 1 %).
The recap's phones are handheld (the demo's were mostly on tripods) and two of them are attached by rotation only, so the per-camera gate is looser for
05 / 06 and the gates are a reading aid, not a cut-off for use.

TWO-VIEW SHARE (LL2V, 2026-10-24; the two-view takes in TWO_VIEW_MIN_SHARE: lucas_luiza): per chunk, camera and dancer, the share of the grid frames whose landmarks
really enter the fit - the fitted set's mamma/ma_2d/<seq>/cam<NN>.npz after mamma prepare zeroed the frames where the camera track is not ok or no mask / landmark
exists and after qa_mask zeroed the stretches that are not on the teacher - and per dancer the share both cameras enter (the triangulated frames). Each camera AND
the both-cameras share must carry >= 0.7 of the frames for both dancers, else the chunk is silently monocular there (two cameras at 0.7 each on different frames
triangulate only 0.4); without the pulled ma_2d the share is unknown and fails. qa_mask may zero at most 10 % of a camera's frames of a dancer (G9: more means
SAM followed someone else), and a camera qa_mask judged suspect (both dancers off the teacher in more than half of its frames: it masks nothing, usually a clock
or camera error) fails too. The same report compares the video offsets the fit used (prepare.json sync.offsets_ms) with the ones the job planned
(farm/jobs/<id>.json chunk.sync_fixed): a job without HM_FRONT_MASTER gauges the wrong camera to 0 and drops the measured offset without an error.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

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
# two-view takes: the least share of a chunk's grid frames in which each camera's landmarks of each dancer enter the fit (None: not measured, the other takes' report
# is unchanged)
TWO_VIEW_MIN_SHARE = {"lucas_luiza": 0.7}.get(DANCE)
# landmarks (of 512) with a visibility > 0 for a dancer's frame to count as entering the fit from a camera: recap_front.QA_MIN_VIS's count, but the > 0 rule of
# mamma_dyn.build_inputs' grid_visible_frac (what mamma prepare and qa_mask zero is 0; recap_front's QA judges only V > 0.5)
TWO_VIEW_MIN_LANDMARKS = 30
QA_MASK_MAX_SHARE = 0.10         # share of a camera's grid frames qa_mask may zero for one dancer (runbook G9)
SYNC_TOL_MS = 0.5                # fitted vs planned video offset (both are rounded to 0.01 ms)
ROLES = ("lead", "follow")       # the ma_2d person slots (mamma_dyn.PIDS order)


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


def two_view_share(work: Path, min_landmarks: int = TWO_VIEW_MIN_LANDMARKS) -> dict | None:
    """The two-view share of one chunk take (see the module doc): {tag, frames, cams: {cam: {share: {role}, grid_cam_ok, qa_masked_share: {role}, qa_suspect, source}},
    both: {role}, fitted_offsets_ms}. Without the pulled ma_2d a camera's share is None (prepare.json's grid_visible_frac is a landmark share, not a frame share) and
    both is {}. None when the take has no prepare.json. Reads one npz member (visibilities) per camera."""
    prep_p = work / "mamma" / "prepare.json"
    if not prep_p.is_file():
        return None
    prep = json.loads(prep_p.read_text(encoding="utf-8"))
    tag = max(prep["sets"], key=lambda t: len(prep["sets"][t]["cams"]))
    s = prep["sets"][tag]
    T = int(prep["grid"]["T"])
    qa_p = work / "mamma" / "qa_mask.json"
    qa = json.loads(qa_p.read_text(encoding="utf-8")) if qa_p.is_file() else {}
    out = {"tag": tag, "frames": T, "min_landmarks": min_landmarks, "cams": {}, "both": {}, "fitted_offsets_ms": (prep.get("sync") or {}).get("offsets_ms") or {}}
    enter = {}
    for cam in s["cams"]:
        inp = (s.get("inputs") or {}).get(cam) or {}
        q = (qa.get("cams") or {}).get(cam) or {}
        masked = q.get("masked_frames") or {}
        row = {"grid_cam_ok": inp.get("grid_cam_ok"), "qa_masked_share": {r: round(int(masked.get(r) or 0) / max(T, 1), 4) for r in ROLES},
               "qa_suspect": q.get("suspect") or []}
        f = work / "mamma" / "ma_2d" / s["seq"] / f"cam{cam}.npz"
        if f.is_file():
            with np.load(f) as z:
                n = (np.asarray(z["visibilities"]) > 0).sum(-1)               # (T, persons): visible landmarks per grid frame and dancer
            ent = n[:, :len(ROLES)] >= min_landmarks
            enter[cam] = ent
            row.update(share={r: round(float(ent[:, i].mean()), 4) for i, r in enumerate(ROLES)}, source="ma_2d")
        else:
            row.update(share={r: None for r in ROLES}, grid_visible_frac=inp.get("grid_visible_frac"), source="no ma_2d pulled")
        out["cams"][cam] = row
    if len(enter) >= 2 and len(enter) == len(s["cams"]):
        out["both"] = {r: round(float(np.logical_and.reduce([e[:, i] for e in enter.values()]).mean()), 4) for i, r in enumerate(ROLES)}
    return out


def planned_offsets(cid: str) -> dict | None:
    """The video offsets (ms) the chunk's job fixed (farm/jobs/<id>.json chunk.sync_fixed.offsets_ms), None when the job measures its own or has no spec here."""
    p = PACKAGE_ROOT / "work" / DANCE / "farm" / "jobs" / f"{cid}.json"
    if not p.is_file():
        return None
    sf = (json.loads(p.read_text(encoding="utf-8")).get("chunk") or {}).get("sync_fixed") or {}
    return dict(sf.get("offsets_ms") or {}) if sf.get("no_measure") else None


def two_view_failures(tw: dict | None, planned: dict | None = None, min_share: float | None = None) -> list[str]:
    """Human-readable failures of one two_view_share() result (and of the planned vs fitted offsets), [] when the chunk is two-view."""
    min_share = TWO_VIEW_MIN_SHARE if min_share is None else min_share
    if tw is None:
        return ["two-view: no mamma/prepare.json (the front's outputs are not pulled)"]
    bad = []
    if len(tw["cams"]) < 2:
        bad.append(f"two-view: the fitted set {tw['tag']} has only camera(s) {', '.join(sorted(tw['cams']))}")
    for cam, row in sorted(tw["cams"].items()):
        for r in ROLES:
            v = row["share"].get(r)
            if v is None:
                bad.append(f"{r}: camera {cam} two-view share unknown ({row.get('source') or 'missing'})")
            elif v < min_share:
                bad.append(f"{r}: camera {cam} two-view share {v:.2f} < {min_share:g} (camera ok {row['grid_cam_ok']}, qa_mask zeroed {row['qa_masked_share'][r]:.2f})")
            if row["qa_masked_share"][r] > QA_MASK_MAX_SHARE + 1e-9:
                bad.append(f"{r}: camera {cam} qa_mask zeroed {row['qa_masked_share'][r]:.2f} of the frames > {QA_MASK_MAX_SHARE:g}")
        if row.get("qa_suspect"):
            bad.append(f"camera {cam}: qa_mask judged both dancers off the teacher (suspect, nothing masked): check the video offset / the camera track")
    if len(tw["cams"]) >= 2:
        both = tw.get("both") or {}
        for r in ROLES:
            v = both.get(r)
            if v is None:
                bad.append(f"{r}: both cameras' share unknown (no ma_2d pulled)")
            elif v < min_share:
                bad.append(f"{r}: both cameras enter only {v:.2f} of the frames < {min_share:g} (the triangulated frames)")
    got = tw.get("fitted_offsets_ms") or {}
    for c, want in sorted((planned or {}).items()):
        g = got.get(c)
        if g is None or abs(float(g) - float(want)) > SYNC_TOL_MS:
            bad.append(f"sync: camera {c} fitted with {g} ms, the job planned {float(want):+.2f} ms (HM_FRONT_MASTER / sync_fixed not applied)")
    return bad


def chunk_ids() -> list[str]:
    plan = PACKAGE_ROOT / "work" / DANCE / "farm" / "plan.json"
    if plan.exists():
        return [c["id"] for c in json.loads(plan.read_text(encoding="utf-8")).get("chunks", [])]
    return sorted(p.name.split("_")[-1] for p in (PACKAGE_ROOT / "work").glob(f"{DANCE}_g*"))


def report(ids: list[str] | None = None) -> dict:
    rows = {}
    for cid in ids or chunk_ids():
        work = PACKAGE_ROOT / "work" / f"{DANCE}_{cid}"
        tw = two_view_share(work) if TWO_VIEW_MIN_SHARE is not None else None
        extra = {} if TWO_VIEW_MIN_SHARE is None else {"two_view": tw}
        tw_bad = [] if TWO_VIEW_MIN_SHARE is None else two_view_failures(tw, planned_offsets(cid))
        p = work / "quality" / "quality_rf2_noslip.json"
        if not p.exists():
            rows[cid] = {"state": "no quality json yet"}
            if extra:          # the front's share is known before the fit is: G9 / G10 read it from the pulled front outputs
                rows[cid].update(state="FAIL" if tw_bad and tw is not None else rows[cid]["state"], failures=tw_bad, **extra)
            continue
        s = summarise(json.loads(p.read_text(encoding="utf-8")))
        bad = gate_failures(s) + tw_bad
        rows[cid] = {"state": "pass" if not bad else "FAIL", "failures": bad, **s, **extra}
    gates = GATES if TWO_VIEW_MIN_SHARE is None else {**GATES, "two_view_share": TWO_VIEW_MIN_SHARE, "two_view_min_landmarks": TWO_VIEW_MIN_LANDMARKS,
                                                      "qa_mask_max_share": QA_MASK_MAX_SHARE}
    return {"dance": DANCE, "gates": gates, "chunks": rows}


def _two_view_cell(r: dict) -> str:
    """'lead / follow' of the least camera share, and both cameras' share in brackets."""
    tw = r.get("two_view")
    if not tw or not tw.get("cams"):
        return "-"

    def least(role):
        v = [row["share"].get(role) for row in tw["cams"].values() if row["share"].get(role) is not None]
        return "-" if not v else f"{min(v):.2f}"
    both = tw.get("both") or {}
    return f"{least('lead')} / {least('follow')}" + (f" (both {both['lead']:.2f} / {both['follow']:.2f})" if both else "")


def table(rep: dict) -> str:
    tv = "two_view_share" in (rep.get("gates") or {})
    lines = ["| chunk | state | frames | lead allcam med / p90 px | follow allcam med / p90 px | worst camera (lead, follow) | rest height lead / follow | jitter mm |"
             + (" two-view share lead / follow |" if tv else ""),
             "|---|---|---|---|---|---|---|---|" + ("---|" if tv else "")]
    for cid, r in rep["chunks"].items():
        if "dancers" not in r:
            lines.append(f"| {cid} | {r['state']} | | | | | | |" + (f" {_two_view_cell(r)} |" if tv else ""))
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
                     f"{f(None if F['rest_height_rel'] is None else 100 * F['rest_height_rel'])} % | {f(L['jitter_rms_mm'], 2)} / {f(F['jitter_rms_mm'], 2)} |"
                     + (f" {_two_view_cell(r)} |" if tv else ""))
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
