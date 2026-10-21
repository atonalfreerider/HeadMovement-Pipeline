r"""john_lula realism v2, CORRECT: old vs new comparison, contact sheets and the final foot lock of the facing refit
(dancecap.mono4d.jl_facing_refit compare | sheets | footlock).

  old   = realism/refit/audit_patch/smplx_rf2_noslip_3db_patched (the jl_refit render driver; raw: smplx_3db/)
  new   = realism/v2/refit/audit_patch/smplx_rf2_noslip_patched (raw: smplx_3db_v2/)
No ground truth on this take. Facing is scored two ways: against the AGENT'S EYE labels (facing/agent_labels.json: 70
frames per dancer + 23 held-out frames of two of Lula's spins; independent of the fits) and against the image posterior
(facing_targets.json; NOT independent for the new fit: its hint came from it). jl_refit.compare (RTMW 2D, 512 landmarks,
jitter, pelvis pops, feet, couple order, SAM 3D Body agreement, seams) is re-used on the four folders by pointing its module
paths at v2 (its own files are untouched).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import jl_facing_refit as FR
from .jl_facing_refit import (FOOT_FINAL, NAME, OLD_BEST, OLD_RAW, OUT_V2, RF2, ROLES, SHEETS2, STITCH2, V2, WORK, log,
                              new_best)

FIT_DIRS = {"old_raw": OLD_RAW, "new_raw": OUT_V2, "old": OLD_BEST, "new": None}
# NOTES_REALISM_NEXT.md known problems + the facing windows (role, t0, t1, what)
WINDOWS = [("lead", 10.6, 11.1, "preview pop John 10.87 s"), ("follow", 12.5, 13.0, "preview pop Lula 12.77 s"),
           ("lead", 13.64, 14.74, "w00 John 3/4 front (fit side)"), ("follow", 15.50, 15.94, "w01 Lula 3/4 front (fit side)"),
           ("follow", 23.6, 24.2, "Lula 23.9 s arms up while hidden"), ("lead", 26.2, 26.43, "John 26.2 s lurch (audit patch)"),
           ("follow", 27.04, 28.37, "Lula camera-not-ok span (interpolated)"), ("follow", 27.97, 28.17, "w03 Lula faces the phone (CONFIRMED error)"),
           ("follow", 28.17, 29.0, "Lula spin after the bad-camera span"), ("lead", 27.3, 28.37, "John camera-not-ok span"),
           ("follow", 30.54, 32.21, "Lula hidden, phone zooms on legs"), ("lead", 37.5, 37.9, "John spins in 2 frames"),
           ("lead", 40.4, 42.37, "seam 1 (gate failed)"), ("follow", 40.4, 42.37, "seam 1 (gate failed)"),
           ("lead", 42.18, 42.51, "w02 John 3/4 (fit side)"), ("follow", 43.1, 43.7, "Lula 43.4 s ducks (hidden)"),
           ("lead", 45.1, 45.7, "45.4 s arm swing lags"), ("follow", 45.1, 45.7, "45.4 s arm swing lags"),
           ("lead", 46.2, 48.17, "seam 2"), ("follow", 46.2, 48.17, "seam 2")]
SPIN_T = (4.5, 29.5)


def fit_dirs() -> dict:
    d = dict(FIT_DIRS)
    d["new"] = new_best()
    return d


def _load(d: Path) -> dict:
    return {r: dict(np.load(Path(d) / f"{r}.npz", allow_pickle=True)) for r in ROLES}


# ================================================================================================ facing
def facing_scores() -> dict:
    """Torso rel yaw of every fit at the video frames vs the agent's eye labels and the image posterior."""
    from . import jl_facing as JF
    T = JF.Take()
    lab = json.loads((V2 / "facing" / "agent_labels.json").read_text())
    tgt = json.loads(FR.TARGETS.read_text())
    ev = JF.load_evidence()
    obs = JF.observations(ev)
    fits = {k: _load(v) for k, v in fit_dirs().items()}
    out = {"what": "torso facing of each fit vs the agent's eye labels (independent) and the image posterior (the new fit's hint "
                   "came from it: not independent for 'new')", "fits": {k: str(v) for k, v in fit_dirs().items()}, "roles": {}}
    rel_all = {}
    for o, r in enumerate(ROLES):
        R = {}
        rel = {}
        for k, F in fits.items():
            J = JF.interp_joints(np.asarray(F[r]["t"], float), np.asarray(F[r]["joints"], float)[:, :22], T.t)
            fwd, up, _ = JF.torso_axes(J)
            rel[k] = JF.rel_yaw_deg(fwd, up, J[:, 0], T.C)
        # SAM 3D Body (raw per frame) for reference
        with np.load(JF.SAM3DB, allow_pickle=True) as z:
            J3 = np.asarray(z[f"{r}_joints_world"], float)[:, :22]
            v3 = np.asarray(z[f"{r}_valid"], bool) & np.asarray(z["cam_ok"], bool)
        fwd, up, _ = JF.torso_axes(J3)
        rel["sam3db"] = np.where(v3, JF.rel_yaw_deg(fwd, up, J3[:, 0], T.C), np.nan)
        rel_all[r] = rel
        cls = {k: JF.facing_class(v) for k, v in rel.items()}
        # (1) agent eye labels: sets A + B (70 frames) and the held-out dense spin set C (Lula)
        eye = {int(f): JF.LABEL_TO_CLASS.get(v[r]) for f, v in lab["labels"].items()}
        eye = {f: JF.CLS_INDEX[c] for f, c in eye.items() if c is not None}
        dense = {int(f): JF.CLS_INDEX[JF.LABEL_TO_CLASS[c]] for f, c in lab.get("dense_eval", {}).get(r, {}).items()
                 if JF.LABEL_TO_CLASS.get(c)}

        def score(lbl: dict, k: str) -> dict:
            fr = np.array(sorted(lbl))
            if not len(fr):
                return {"n": 0}
            y = np.array([lbl[f] for f in fr])
            p = cls[k][fr]
            ok = p >= 0
            return {"n": int(ok.sum()), "class_accuracy": round(float((p[ok] == y[ok]).mean()), 4),
                    "front_back_swaps": int((((p == 0) & (y == 2)) | ((p == 2) & (y == 0)))[ok].sum()),
                    "wrong_t": [round(float(T.t[f]), 2) for f, a, b in zip(fr, p, y) if a >= 0 and a != b][:30]}
        R["vs_agent_eye_labels"] = {k: score(eye, k) for k in rel}
        R["vs_agent_dense_spin_labels_heldout"] = {k: score(dense, k) for k in rel} if dense else None
        # (2) image posterior (targets): usable confident front / back frames
        cols = tgt["dancers"][r]["columns"]
        ic = np.array([JF.CLS_INDEX[c] for c in cols["image_class"]])
        conf = np.array(cols["confidence"], float)
        src = np.array(cols["source"], dtype=object)
        act = np.array(cols["action"], dtype=object)
        use = (((conf >= 0.8) & (src == "image")) | (src == "owner")) & (ic != 1) & (src != "agent_review") & (act != "none")
        wrongset = use & np.isin(act, ["rotate", "flip180"])            # frames the OLD fit got wrong (the targets' windows)
        spin = use & (T.t >= SPIN_T[0]) & (T.t <= SPIN_T[1])

        def vs_post(m: np.ndarray, k: str) -> dict:
            a = np.abs(rel[k])
            good = m & np.isfinite(a)
            band_ok = np.where(ic == 0, a <= 60, a >= 120)
            near = np.where(ic == 0, a <= 80, a >= 100)
            return {"frames": int(good.sum()), "class_agrees": round(float(band_ok[good].mean()), 4) if good.any() else None,
                    "within_20deg_of_band": round(float(near[good].mean()), 4) if good.any() else None,
                    "front_back_swaps": int((good & (((ic == 0) & (a >= 120)) | ((ic == 2) & (a <= 60)))).sum())}
        R["vs_image_posterior_conf_ge_0.8"] = {k: vs_post(use, k) for k in rel}
        R["vs_image_posterior_on_old_fit_wrong_frames"] = {k: vs_post(wrongset, k) for k in rel}
        if r == "follow":
            R["vs_image_posterior_spins_4.5-29.5s"] = {k: vs_post(spin, k) for k in rel}
        # (3) bib / OCR-confirmed BACK frames: share of fits at |rel| >= 100
        hit = (obs["bib"][:, o] == 1) | (obs["ocr"][:, o] == 1)
        R["bib_or_ocr_back_frames"] = {k: {"frames": int((hit & np.isfinite(rel[k])).sum()),
                                           "abs_rel_ge_100_share": round(float((np.abs(rel[k][hit]) >= 100)[np.isfinite(rel[k][hit])].mean()), 4)}
                                       for k in rel}
        # (4) the targets' windows (median |rel|, image class)
        wins = {}
        for w in tgt["windows"]:
            if w["role"] != r:
                continue
            fr = np.arange(int(w["start"]), int(w["end"]) + 1)
            wins[w["id"]] = {"t": [w["t0"], w["t1"]], "image_class": w["image_class"],
                             **{f"{k}_abs_rel_median": round(float(np.nanmedian(np.abs(rel[k][fr]))), 1) for k in rel}}
        R["target_windows"] = wins
        out["roles"][r] = R
    np.savez_compressed(RF2 / "facing_rel_frames.npz", t=T.t, **{f"{r}_{k}": v for r in ROLES for k, v in rel_all[r].items()})
    (RF2 / "compare_facing.json").write_text(json.dumps(out, indent=1, default=str))
    log(f"facing -> {RF2 / 'compare_facing.json'}")
    return out


# ================================================================================================ jl_refit.compare on the four folders
def refit_compare() -> dict:
    from . import jl_refit as RF
    saved = (RF.FITS, RF.REFIT, RF.STITCH_DIR)
    try:
        d = fit_dirs()
        RF.FITS = {"current": d["old_raw"], "new": d["new_raw"], "current_rf2": d["old"], "new_rf2": d["new"]}
        RF.REFIT = RF2
        RF.STITCH_DIR = STITCH2
        rep = RF.compare(do_sanity=True)
        mk = RF.mask_agreement()
    finally:
        RF.FITS, RF.REFIT, RF.STITCH_DIR = saved
    rep["labels"] = {"current": "old raw smplx_3db/", "new": "new raw smplx_3db_v2/", "current_rf2": "old best (refined, patched)",
                     "new_rf2": "new best (refined, patched)"}
    rep["mask_agreement"] = mk
    (RF2 / "compare.json").write_text(json.dumps(rep, indent=1, default=str))
    return rep


# ================================================================================================ problem windows
def window_table() -> dict:
    from . import jl_facing as JF
    T = JF.Take()
    fits = {k: _load(v) for k, v in fit_dirs().items() if k in ("old", "new")}
    tgt = json.loads(FR.TARGETS.read_text())
    cf = np.load(RF2 / "compare_frames.npz")
    mf = np.load(RF2 / "mask_agreement_frames.npz")
    fr = np.load(RF2 / "facing_rel_frames.npz")
    rows = []
    for role, t0, t1, what in WINDOWS:
        row = {"role": role, "t": [t0, t1], "what": what}
        cols = tgt["dancers"][role]["columns"]
        vf = (T.t >= t0) & (T.t <= t1)
        ic = np.array(cols["image_class"], dtype=object)[vf]
        cc = np.array(cols["confidence"], float)[vf]
        row["image_class_conf_ge_0.8"] = {c: int(((ic == c) & (cc >= 0.8)).sum()) for c in ("front", "side", "back")}
        for k, F in fits.items():
            z = F[role]
            t = np.asarray(z["t"], float)
            m = (t >= t0 - 1e-6) & (t <= t1 + 1e-6)
            J = np.asarray(z["joints"], float)[:, :22]
            step = 1e3 * np.linalg.norm(np.diff(J[:, 0], axis=0), axis=1)
            fwd = JF.torso_axes(J)[0]
            yaw = np.unwrap(np.radians(FR.yaw_deg(fwd)))
            yr = np.degrees(np.abs(np.diff(yaw)))
            mm = m[:-1] & m[1:]
            rel = fr[f"{role}_{k}"][vf]
            a = np.abs(rel)
            band = np.where(ic == "front", a <= 60, np.where(ic == "back", a >= 120, np.nan))
            conf_m = (cc >= 0.8) & (ic != "side")
            mk = mf[f"{role}_{'current_rf2' if k == 'old' else 'new_rf2'}"]
            row[k] = {"pelvis_step_max_mm": round(float(step[mm].max()), 1) if mm.any() else None,
                      "torso_yaw_rate_max_deg_per_frame": round(float(yr[mm].max()), 1) if mm.any() else None,
                      "abs_rel_median_deg": round(float(np.nanmedian(a)), 1),
                      "agrees_with_confident_image_class": round(float(np.nanmean(band[conf_m])), 3) if conf_m.any() else None,
                      "joints_in_own_mask": round(float(np.nanmean(mk[m, 0])), 3) if np.isfinite(mk[m, 0]).any() else None,
                      "joints_outside_both_masks": round(float(np.nanmean(mk[m, 2])), 3) if np.isfinite(mk[m, 2]).any() else None}
        # MAMMA 512-landmark residual of the RAW fits (compare_frames: current = old raw, new = new raw)
        tg = cf["t"]
        m = (tg >= t0) & (tg <= t1)
        for kk, lab in (("current", "old_raw"), ("new", "new_raw")):
            v = cf[f"{role}_reproj512_{kk}"][m]
            row[f"reproj512_px_median_{lab}"] = round(float(np.nanmedian(v)), 1) if np.isfinite(v).any() else None
        rows.append(row)
    out = {"what": "NOTES_REALISM_NEXT.md problem windows + the facing windows: old best vs new best (refined, patched)", "rows": rows}
    (RF2 / "compare_windows.json").write_text(json.dumps(out, indent=1, default=str))
    log(f"windows -> {RF2 / 'compare_windows.json'}")
    return out


def compare_all() -> dict:
    t0 = time.time()
    fs = facing_scores()
    rc = refit_compare()
    wt = window_table()
    summ = summary(fs, rc, wt)
    (RF2 / "compare_summary.json").write_text(json.dumps(summ, indent=1, default=str))
    log(f"compare done in {time.time() - t0:.0f} s -> {RF2 / 'compare_summary.json'}")
    print(json.dumps(summ, indent=1, default=str)[:12000])
    return summ


def summary(fs: dict, rc: dict, wt: dict) -> dict:
    out = {"facing": {}, "pose": {}, "couple_depth_order": {}, "windows": {}}
    for r in ROLES:
        f = fs["roles"][r]
        out["facing"][r] = {k: {"eye_acc": f["vs_agent_eye_labels"][k].get("class_accuracy"),
                                "eye_swaps": f["vs_agent_eye_labels"][k].get("front_back_swaps"),
                                "dense_spin_acc": (f["vs_agent_dense_spin_labels_heldout"] or {}).get(k, {}).get("class_accuracy"),
                                "posterior_agree": f["vs_image_posterior_conf_ge_0.8"][k]["class_agrees"],
                                "posterior_swaps": f["vs_image_posterior_conf_ge_0.8"][k]["front_back_swaps"],
                                "old_wrong_frames_agree": f["vs_image_posterior_on_old_fit_wrong_frames"][k]["class_agrees"],
                                "bib_back_ge100": f["bib_or_ocr_back_frames"][k]["abs_rel_ge_100_share"]}
                            for k in ("old", "new", "old_raw", "new_raw", "sam3db")}
        R = rc["roles"][r]
        out["pose"][r] = {"reproj512_px_median": {k: R["reproj512_px"][k].get("median") for k in R["reproj512_px"]},
                          "teacher_reproj_pct_height_median": {k: v.get("median") for k, v in R["teacher_reproj_pct_height"].items()},
                          "teacher_reproj_pct_height_p90": {k: v.get("p90") for k, v in R["teacher_reproj_pct_height"].items()},
                          "jitter_accel_rr_median": {k: v.get("median") for k, v in R["jitter_accel_rr_mm_per_frame2"].items()},
                          "jitter_accel_rr_p90": {k: v.get("p90") for k, v in R["jitter_accel_rr_mm_per_frame2"].items()},
                          "pelvis_accel_over_40_frames": {k: v["frames_pelvis_accel_over_40"] for k, v in R["pelvis_pops"].items()},
                          "facing_lr_vs_rtmw_agree": {k: v["agree_share"] for k, v in R["facing_lr_vs_teacher"].items()},
                          "new_vs_old_raw_facing_over_90deg_frames": R["new_vs_current"]["facing_over_90deg_frames"],
                          "new_vs_old_raw_root_relative_mm_median": R["new_vs_current"]["root_relative_mm"].get("median"),
                          "mask_own_share_all": {k: v["all"]["own"] for k, v in rc["mask_agreement"][r].items()},
                          "mask_outside_both_share_all": {k: v["all"]["outside_both"] for k, v in rc["mask_agreement"][r].items()}}
    out["couple_depth_order"] = {k: v.get("agree_with_occlusion_evidence") for k, v in rc["couple_depth_order"].items()
                                 if isinstance(v, dict) and "agree_with_occlusion_evidence" in v}
    out["stitch_gate_new"] = (rc.get("stitch", {}).get("new") or {}).get("gate")
    out["windows"] = [{"what": w["what"], "role": w["role"], "t": w["t"],
                       "old": {k: w["old"][k] for k in ("abs_rel_median_deg", "agrees_with_confident_image_class", "pelvis_step_max_mm",
                                                        "torso_yaw_rate_max_deg_per_frame", "joints_in_own_mask")},
                       "new": {k: w["new"][k] for k in ("abs_rel_median_deg", "agrees_with_confident_image_class", "pelvis_step_max_mm",
                                                        "torso_yaw_rate_max_deg_per_frame", "joints_in_own_mask")},
                       "reproj512_raw": [w["reproj512_px_median_old_raw"], w["reproj512_px_median_new_raw"]]} for w in wt["rows"]]
    return out


# ================================================================================================ sheets
SHEET_T = {"w03_lula_27.9-28.7": ("follow", [27.3, 27.7, 27.97, 28.07, 28.17, 28.4, 28.6, 28.87]),
           "w00_john_13.6-14.7": ("lead", [13.4, 13.7, 14.0, 14.3, 14.5, 14.7]),
           "w01_lula_15.5-15.9": ("follow", [15.4, 15.6, 15.75, 15.9]),
           "w02_john_42.2-42.5": ("lead", [42.0, 42.2, 42.35, 42.5]),
           "lula_spins_5-26": ("follow", [5.1, 7.0, 8.6, 11.97, 15.97, 18.87, 22.0, 25.0])}


def sheets() -> list[Path]:
    """v2/sheets/facing_<name>.jpg: per instant the phone frame with the OLD best skeleton (left) and the NEW best (right),
    crop on the dancers; John orange, Lula cyan, body-left bones lighter, eyes magenta."""
    import cv2
    from . import jl_refit as RF
    saved = RF.FITS
    out = []
    try:
        RF.FITS = {"current": OLD_BEST, "new": new_best()}
        ov = RF._Overlay(keys=("current", "new"))
        rel = np.load(RF2 / "facing_rel_frames.npz")
        tgt = json.loads(FR.TARGETS.read_text())
        SHEETS2.mkdir(parents=True, exist_ok=True)
        for name, (role, times) in SHEET_T.items():
            cols = tgt["dancers"][role]["columns"]
            pairs = []
            for x in times:
                f = int(np.argmin(np.abs(ov.tc - x)))
                box = ov.crop_box(f, 360 / 480)
                ic, cc = cols["image_class"][f], cols["confidence"][f]
                a = ov.tile(f, "current", label=f"t={ov.tc[f]:.2f}s {NAME[role]}\nOLD rel {rel[f'{role}_old'][f]:.0f}\nimage {ic} {cc:.2f}", box=box)
                b = ov.tile(f, "new", label=f"t={ov.tc[f]:.2f}s {NAME[role]}\nNEW rel {rel[f'{role}_new'][f]:.0f}\nimage {ic} {cc:.2f}", box=box)
                pairs.append(np.concatenate([a, np.full((480, 4, 3), 255, np.uint8), b], 1))
            cols_n = 2
            rows = []
            for k in range(0, len(pairs), cols_n):
                r = pairs[k:k + cols_n] + [np.zeros_like(pairs[0])] * (cols_n - len(pairs[k:k + cols_n]))
                rows.append(np.concatenate([np.concatenate([p, np.zeros((480, 10, 3), np.uint8)], 1) for p in r], 1))
            img = np.concatenate([np.concatenate([r, np.zeros((10, r.shape[1], 3), np.uint8)], 0) for r in rows], 0)
            p = SHEETS2 / f"facing_{name}.jpg"
            cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
            out.append(p)
    finally:
        RF.FITS = saved
    log(f"sheets: {[str(p) for p in out]}")
    return out


# ================================================================================================ foot lock
def footlock_final() -> dict:
    from . import jl_footlock as FL
    src = new_best()
    rep = FL.footlock_fit(src, FOOT_FINAL)
    ck = FL.check(before=src, after=FOOT_FINAL, out=V2 / "footlock" / "check_final.json")
    old_fl = V2 / "footlock" / "smplx_footlock"
    keys = [str(OLD_BEST), str(src), str(FOOT_FINAL)] + ([str(old_fl)] if (old_fl / "lead.npz").exists() else [])
    ms = FL.measure(keys, out=V2 / "footlock" / "measure_final.json", runs_from=str(OLD_BEST))
    sh = FL.sheet(before=src, after=FOOT_FINAL, out=V2 / "footlock" / "sheets" / "final_before_after.jpg")
    summ = {"render_driver": str(FOOT_FINAL), "src": str(src), "slip_same_video_stance_runs (old best's)": {}}
    for k, v in ms["fits"].items():
        summ["slip_same_video_stance_runs (old best's)"][k] = {
            r: {kk: x.get(kk) for kk in ("stance_runs", "slide_mm_per_frame", "per_stance_spread_cm", "stances_spread_over_5cm_share",
                                         "fit_vs_ray_btoe_cm")} for r, x in v["roles"].items()}
    summ["check_final"] = {r: {k: ck["roles"][r][k] for k in ("slip", "accel_rr_mm", "transl_change_mm")} for r in ROLES}
    summ["sheet"] = str(sh)
    (V2 / "footlock" / "final_summary.json").write_text(json.dumps(summ, indent=1, default=float))
    log(f"footlock final -> {FOOT_FINAL}")
    print(json.dumps(summ, indent=1, default=float)[:8000])
    return summ


# ================================================================================================ RTMW 2D per window
def rtmw_per_frame(fit_dir: Path) -> dict:
    """jl_fitcheck.reproj's per-frame measure (median COCO-17 joint error vs the RTMW teacher, % of the box height; frames
    with the camera ok, the teacher confident and not on the partner) for one fit folder: {role: (T,)}."""
    from .. import jl_fitcheck as FC
    S = {r: dict(np.load(Path(fit_dir) / f"{r}.npz", allow_pickle=True)) for r in ROLES}
    cam = np.load(WORK / "cameras" / "01.npz", allow_pickle=True)
    T_ = np.load(FC.PEOPLE / "teachers_01.npz", allow_pickle=True)
    t = np.asarray(S["lead"]["t"], float)
    j, near = FC.frame_map(t, np.asarray(cam["t"], float))
    ok = np.asarray(cam["ok"], bool)[j] & near
    err = {r: np.full(len(t), np.nan) for r in ROLES}
    for ri, role in enumerate(ROLES):
        J = np.asarray(S[role]["joints"], float)
        for i in np.flatnonzero(ok):
            k = int(j[i])
            fr = int(np.asarray(cam["frame_index"])[k])
            if T_["conf"][fr, ri] < FC.CONF or T_["skel_in_partner"][fr, ri] > 0.5:
                continue
            kp = T_["kp"][fr, ri]
            ids = [c for c in FC.COCO_TO_SMPLX if kp[c, 2] >= FC.CONF]
            if len(ids) < 6:
                continue
            uv, z = FC.project(J[i, [FC.COCO_TO_SMPLX[c] for c in ids]], np.asarray(cam["R"][k], float),
                               np.asarray(cam["tvec"][k], float), np.asarray(cam["K"][k], float),
                               float(np.asarray(cam["dist"][k]).reshape(-1)[0]))
            hpx = max(float(T_["box"][fr, ri, 3] - T_["box"][fr, ri, 1]), 50.0)
            e = 100.0 * np.linalg.norm(uv - kp[ids, :2], axis=-1) / hpx
            if (z > 0).sum() >= 6:
                err[role][i] = float(np.median(e[z > 0]))
    return err


def add_rtmw_to_windows() -> dict:
    """compare_windows.json + compare_summary.json: the RTMW 2D median per window for the old and new best fits."""
    d = fit_dirs()
    E = {k: rtmw_per_frame(d[k]) for k in ("old", "new")}
    t = np.arange(len(E["old"]["lead"])) / 30.0
    wt = json.loads((RF2 / "compare_windows.json").read_text())
    for row in wt["rows"]:
        m = (t >= row["t"][0] - 1e-6) & (t <= row["t"][1] + 1e-6)
        for k in ("old", "new"):
            v = E[k][row["role"]][m]
            row[k]["rtmw_pct_height_median"] = round(float(np.nanmedian(v)), 2) if np.isfinite(v).any() else None
            row[k]["rtmw_frames"] = int(np.isfinite(v).sum())
    (RF2 / "compare_windows.json").write_text(json.dumps(wt, indent=1, default=str))
    sm = json.loads((RF2 / "compare_summary.json").read_text())
    for w, row in zip(sm["windows"], wt["rows"]):
        w["rtmw_pct_height_median"] = [row["old"]["rtmw_pct_height_median"], row["new"]["rtmw_pct_height_median"]]
    (RF2 / "compare_summary.json").write_text(json.dumps(sm, indent=1, default=str))
    np.savez_compressed(RF2 / "rtmw_frames.npz", t=t, **{f"{r}_{k}": E[k][r] for k in E for r in ROLES})
    return wt
