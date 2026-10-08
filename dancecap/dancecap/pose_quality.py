"""Quality numbers of a smplx-contract fit: reprojection per camera (all points and feet only), root jitter, turns,
floor contact, foot slip on fixed frames, joint displacement vs a reference fit, rest heights.

    python -m dancecap.pose_quality <take> [--smplx-dir smplx] [--tag final] [--ref-dir smplx]
                                    [--compare-take larissa_kadu --window 56 62] [--physics]
    python -m dancecap.pose_quality <take> --variants final smplx_filtered smplx_refined ... [--ref-tag final]

-> work/<take>/quality/quality_<tag>.json + quality_<tag>_frames.npz (per-frame arrays for the review figures).
  reproj_px        per camera and dancer, median / p90 over frames of the per-frame median pixel error of MAMMA's 512
                   points (smplx FK WITH pose correctives, like mamma_dyn.camera_residuals) vs the grid landmarks of the
                   chosen set (mamma/select.json); 'whole' and the --window span; all_cams_per_frame_median = per frame
                   the median over cameras. With --compare-take: the same for that take's own fit and landmarks (its
                   whole fit should lie inside --window, e.g. the 6 s larissa_kadu c8 fit on 56-62 s).
  reproj_feet_px   the same for the feet only (mamma_dyn.part_index left_foot + right_foot: 88 of the 512 points). The
                   all-points median hides a foot-level change (the first contact refinement cost +0.34 / +0.38 px on
                   all points but +1.5 / +1.0 px on the feet).
  jitter_4p5_12hz_mm  analysis_review.root_jitter (band-passed pelvis displacement): whole, during turns (pelvis yaw
                   rate > 200 deg/s, +-3 frames), outside turns, window, worst 0.5 s windows. NOTE: a low-pass of the
                   root below 4.5 Hz (pose_refine filter) removes this band BY CONSTRUCTION; the evidence that such a
                   filter removed noise rather than motion is an unchanged reprojection, not this number.
  floor            lowest sole point (physics_body LBS, no correctives = the Unity skin): fractions > 15 mm below /
                   above the floor, > 30 mm above; own stance (sole within 3 cm of the lower foot and its centre slower
                   than 0.25 m/s, on THIS motion) heights; stance_slip_mm = max deviation of the sole centre from its
                   run mean over this motion's own stance runs (grows with run length; a motion with more / longer
                   stance runs scores worse) ; lowest foot joint minus its rest height above the sole.
  slip             foot slip on FIXED frames, comparable across variants: stance runs of the reference fit (--ref-dir,
                   default smplx/ = the MAMMA fit; the same 'own stance' rule run on the reference) ->
                   contact_point_speed_mm_s (per frame pair: median horizontal speed of the sole vertices that are
                   within 1 cm of the foot's lowest point in the first frame - the same vertices in both frames, so a
                   heel-toe roll is not counted as slip) on those frames, and ref_run_drift_mm (stance_slip_mm's
                   statistic on the reference's runs). When the folder has pose_refine's 'stance' (the frames its
                   optimiser pinned): the speed on the pinned frames (IN-SAMPLE: the slip term minimised it) and on
                   reference stance frames that were not pinned (held out).
  displacement_vs_ref_mm  per joint (pelvis, ankles, feet), horizontal and vertical distance to the reference fit,
                   for the feet split by the reference's stance / swing frames (median, p95, p99).
  rest_height_m    crown-to-sole rest height vs stature + shoe sole (takes/<take>.toml).
--physics: also physics.run on this folder -> work/<take>/physics_<smplx-dir>/ (beats from timing/timing.json),
           never physics/ (that one is `python -m dancecap physics`, always on smplx/).
--variants: reads quality_<tag>.json / _frames.npz of each tag (run the per-folder command first) -> one comparison
           table quality/variants.json (per variant and dancer: reprojection all / feet incl. per camera, feet error
           change on reference stance frames where the reference sole is > 15 mm above the floor, displacement, floor,
           slip on fixed frames, jitter) and review/pose_refine_variants.png.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import physics_body as pb
from .take import PACKAGE_ROOT, Take

ROLES = ("lead", "follow")
FEET = ("left", "right")
TURN_DEG_S = 200.0
STANCE_V = 0.25   # m/s, sole-centre speed of the 'own stance' rule
JOINTS = {"pelvis": 0, "left_ankle": 7, "right_ankle": 8, "left_foot": 10, "right_foot": 11}


def load(d: Path) -> dict:
    return {r: dict(np.load(d / f"{r}.npz", allow_pickle=True)) for r in ROLES}


def grid_dirs(work: Path) -> tuple[Path, Path]:
    sel = json.loads((work / "mamma" / "select.json").read_text())
    seq = f"{work.name}_{sel['chosen']}"
    return work / "mamma" / "ma_cap" / seq / "gt", work / "mamma" / "ma_2d" / seq


def _med(e: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """(T, N) errors + mask -> (T,) median over the visible points (NaN below 3)."""
    x = np.where(ok, e, np.nan)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            m = np.nanmedian(x, axis=1)
    return np.where(ok.sum(1) >= 3, m, np.nan)


def per_frame_cam_px(sm: dict, gt_dir: Path, lm_dir: Path) -> tuple[dict, dict]:
    """-> (px, feet): {cam: {role: (T,)}} median px of the 512 points visible in that camera (NaN < 3 obs), and
    {cam: {role: {'feet' | 'left' | 'right': (T,)}}} the same over the foot points (both feet / each foot)."""
    from . import mamma_dyn as md
    D = md._load_verts512()
    parts = md.part_index()
    fidx = {f: parts[f"{f}_foot"] for f in FEET if f"{f}_foot" in parts}
    out, feet = {}, {}
    for p, pid in enumerate(md.PIDS):
        role = md.ROLES[pid]
        r = sm[role]
        pose = np.concatenate([r["global_orient"], r["body_pose"], r["jaw_pose"], r["leye_pose"], r["reye_pose"],
                               r["left_hand_pose"], r["right_hand_pose"]], 1)
        verts, _ = md._forward(pose, r["transl"], r["betas"], True)
        pts = np.matmul(D, verts)
        del verts
        sl = slice(0, len(pose))
        for f in sorted(gt_dir.glob("cam*.npz")):
            e, ok = md._camera_errors(pts, f.stem, gt_dir, lm_dir, p, sl)
            cam = f.stem[3:]
            out.setdefault(cam, {})[role] = _med(e, ok)
            if fidx:
                both = np.concatenate([fidx["left"], fidx["right"]])
                d = {"feet": _med(e[:, both], ok[:, both])}
                d.update({k: _med(e[:, ix], ok[:, ix]) for k, ix in fidx.items()})
                feet.setdefault(cam, {})[role] = d
    return out, feet


def stats(x) -> dict:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0}
    return {"median": round(float(np.median(x)), 2), "p90": round(float(np.percentile(x, 90)), 2), "n": int(len(x))}


def mm_stats(x) -> dict:
    x = 1000 * np.asarray(x, float)
    return {"median": round(float(np.median(x)), 1), "p5": round(float(np.percentile(x, 5)), 1),
            "p95": round(float(np.percentile(x, 95)), 1), "min": round(float(x.min()), 1), "max": round(float(x.max()), 1)}


def mm_tail(x) -> dict:
    x = 1000 * np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0}
    return {"median": round(float(np.median(x)), 1), "p95": round(float(np.percentile(x, 95)), 1),
            "p99": round(float(np.percentile(x, 99)), 1), "n": int(len(x))}


def heading_rate(z: dict) -> np.ndarray:
    """Pelvis yaw rate (deg/s) about world Y (SMPL-X body forward = rest +z)."""
    from scipy.signal import butter, filtfilt
    R = pb.axis_angle_to_matrix(np.asarray(z["global_orient"], float))
    yaw = np.unwrap(np.arctan2(R[:, 0, 2], R[:, 2, 2]))
    t = np.asarray(z["t"], float)
    b, a = butter(2, 4.0 / (0.5 / np.median(np.diff(t))))
    return np.degrees(np.gradient(filtfilt(b, a, yaw), t))


def dilate(m: np.ndarray, n: int) -> np.ndarray:
    return np.convolve(m.astype(float), np.ones(2 * n + 1), "same") > 0


def runs_of(m: np.ndarray, min_len: int = 1) -> list[tuple[int, int]]:
    out, i = [], 0
    while i < len(m):
        if m[i]:
            j = i
            while j + 1 < len(m) and m[j + 1]:
                j += 1
            if j - i + 1 >= min_len:
                out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def sole_kin(z: dict) -> dict:
    """Plain-LBS kinematics of the soles: per foot the sole points (T, n, 3), their lowest height and centre, the
    55 FK joints."""
    body = pb.load_body(z)
    pose = pb.full_pose(z)
    T = len(pose)
    R, p = pb.forward_kinematics(body, pose, np.asarray(z["transl"], float).reshape(T, 3))
    k = {"t": np.asarray(z["t"], float), "J": np.einsum("tjab,jb->tja", R, body.j_rest) + p, "body": body,
         "pts": {}, "sole": {}, "cen": {}}
    for f in FEET:
        pts = pb.lbs_points(body, R, p, body.sole_vertices(f, 0.02))
        k["pts"][f], k["sole"][f], k["cen"][f] = pts, pts[..., 1].min(1), pts.mean(1)
    return k


def own_stance(k: dict) -> dict:
    t = k["t"]
    low = np.minimum(k["sole"]["left"], k["sole"]["right"])
    return {f: (np.linalg.norm(np.gradient(k["cen"][f][:, [0, 2]], t, axis=0), axis=1) < STANCE_V)
            & (k["sole"][f] < low + 0.03) for f in FEET}


def run_drift(cen: np.ndarray, m: np.ndarray, min_len: int = 4) -> np.ndarray:
    """Per stance run (>= min_len frames): max horizontal deviation of the sole centre from the run's mean (m)."""
    out = []
    for a, b in runs_of(m, min_len):
        c = cen[a:b + 1, [0, 2]]
        out.append(float(np.max(np.linalg.norm(c - c.mean(0), axis=1))))
    return np.array(out)


def contact_point_speed(k: dict, f: str) -> np.ndarray:
    """(T-1,) per frame pair: median horizontal speed (m/s) of the sole vertices within 1 cm of the foot's lowest point
    in the first frame (the same vertices in both frames)."""
    pts = k["pts"][f]
    S = pts[:-1, :, 1] <= k["sole"][f][:-1, None] + 0.01
    d = np.linalg.norm(np.diff(pts[..., [0, 2]], axis=0), axis=-1) / np.diff(k["t"])[:, None]
    return _med(d, S | (S.sum(1, keepdims=True) < 3))  # a 1-2 vertex contact: all sole vertices of the frame


def floor_metrics(z: dict, k: dict | None = None) -> dict:
    k = k or sole_kin(z)
    low = np.minimum(k["sole"]["left"], k["sole"]["right"])
    body = k["body"]
    rest_fj = float(body.j_rest[[7, 8, 10, 11], 1].min() - body.v_rest[:, 1].min())
    stance = own_stance(k)
    st_h = np.concatenate([k["sole"][f][stance[f]] for f in FEET])
    slide = np.concatenate([run_drift(k["cen"][f], stance[f]) for f in FEET])
    return {"low": low, "stance": stance, "stance_heights": st_h, "foot_joint": k["J"][:, [7, 8, 10, 11], 1].min(1),
            "rest_foot_joint_above_sole": rest_fj, "slide_m": slide, "com_y": None}


def window_mask(t, lo, hi):
    return (t >= lo - 1e-6) & (t <= hi + 1e-6)


def slip_and_displacement(z: dict, k: dict, zref: dict | None, kref: dict, frames: dict, role: str) -> dict:
    """Slip on the reference's stance frames (+ pinned / held-out when z has 'stance'), displacement vs reference."""
    rst = own_stance(kref)
    pin = np.asarray(z["stance"], bool) if "stance" in z else None
    sl = {"reference_stance": "own-stance rule (sole centre < 0.25 m/s, within 3 cm of the lower foot) on the "
                              "reference fit: the same frames for every variant",
          "contact_point_speed_mm_s": {}, "ref_run_drift_mm": {}}
    for i, f in enumerate(FEET):
        v = contact_point_speed(k, f)
        both = rst[f][1:] & rst[f][:-1]
        frames[f"slipv_{f[0]}_{role}"], frames[f"rst_{f[0]}_{role}"] = v, rst[f]
        frames[f"sole_{f[0]}_{role}"] = k["sole"][f]
        frames[f"refsole_{f[0]}_{role}"] = kref["sole"][f]
        row = {"ref_stance": mm_tail(v[both])}
        if pin is not None:
            pb_ = pin[1:, i] & pin[:-1, i]
            frames[f"pin_{f[0]}_{role}"] = pin[:, i]
            row["pinned_in_sample"] = mm_tail(v[pb_])
            row["ref_stance_not_pinned"] = mm_tail(v[both & ~pb_])
        sl["contact_point_speed_mm_s"][f] = row
        d = run_drift(k["cen"][f], rst[f])
        sl["ref_run_drift_mm"][f] = mm_tail(d)
    allv = np.concatenate([frames[f"slipv_{f[0]}_{role}"][rst[f][1:] & rst[f][:-1]] for f in FEET])
    alld = np.concatenate([run_drift(k["cen"][f], rst[f]) for f in FEET])
    sl["both_feet"] = {"contact_point_speed_ref_stance_mm_s": mm_tail(allv), "ref_run_drift_mm": mm_tail(alld)}
    if pin is not None:
        pv = np.concatenate([frames[f"slipv_{f[0]}_{role}"][pin[1:, i] & pin[:-1, i]] for i, f in enumerate(FEET)])
        hv = np.concatenate([frames[f"slipv_{f[0]}_{role}"][(rst[f][1:] & rst[f][:-1]) & ~(pin[1:, i] & pin[:-1, i])]
                             for i, f in enumerate(FEET)])
        sl["both_feet"].update({"contact_point_speed_pinned_in_sample_mm_s": mm_tail(pv),
                                "contact_point_speed_ref_stance_not_pinned_mm_s": mm_tail(hv)})
    disp = None
    if zref is not None:
        dJ = k["J"] - kref["J"]
        h, vt = np.linalg.norm(dJ[..., [0, 2]], axis=-1), np.abs(dJ[..., 1])
        disp = {}
        for name, j in JOINTS.items():
            frames[f"dh_{name}_{role}"], frames[f"dv_{name}_{role}"] = h[:, j], vt[:, j]
            if name == "pelvis":
                disp[name] = {"all": {"horiz": mm_tail(h[:, j]), "vert": mm_tail(vt[:, j])}}
                continue
            m = rst[name.split("_")[0]]
            disp[name] = {"stance": {"horiz": mm_tail(h[m, j]), "vert": mm_tail(vt[m, j])},
                          "swing": {"horiz": mm_tail(h[~m, j]), "vert": mm_tail(vt[~m, j])}}
    return {"slip": sl, "displacement": disp}


def run(take: str, smplx_dir: str = "smplx", tag: str = "final", compare_take: str | None = None,
        window: tuple[float, float] | None = None, physics: bool = False, ref_dir: str = "smplx") -> dict:
    from .analysis_review import root_jitter
    from .cameras import world_scale as ws
    from .cameras.world import sole_from_take
    work = PACKAGE_ROOT / "work" / take
    tk = Take.load(take)
    soles = sole_from_take(tk.cfg)[0]
    statures = {r: float(tk.cfg["dancers"][r]["stature_m"]) for r in ROLES}
    sm = load(work / smplx_dir)
    t = np.asarray(sm["lead"]["t"], float)
    ref = sm if ref_dir == smplx_dir else load(work / ref_dir)
    for r in ROLES:
        if len(ref[r]["t"]) != len(t) or not np.allclose(ref[r]["t"], t, atol=1e-4):
            raise ValueError(f"{ref_dir}/{r}.npz is not on the time grid of {smplx_dir}/")
    out = {"take": take, "smplx": f"work/{take}/{smplx_dir}", "ref": f"work/{take}/{ref_dir}", "frames": int(len(t)),
           "span_s": [round(float(t[0]), 3), round(float(t[-1]), 3)], "window_s": window,
           "notes": {"jitter": "4.5-12 Hz pelvis band: a root low-pass below 4.5 Hz removes it by construction; judge "
                               "a filter by the reprojection change",
                     "slip": "contact_point_speed on the reference's stance frames is comparable across variants; "
                             "'pinned_in_sample' is measured on the frames pose_refine's slip term minimised",
                     "smoothing": json.loads(str(sm["lead"]["smoothing"])) if "smoothing" in sm["lead"] else None}}
    frames = {"t": t}
    win = window_mask(t, *window) if window else np.ones(len(t), bool)
    px, fpx = per_frame_cam_px(sm, *grid_dirs(work))
    allc = {r: np.nanmedian(np.stack([px[c][r] for c in px]), 0) for r in ROLES}
    out["reproj_px"] = {"whole": {c: {r: stats(px[c][r]) for r in ROLES} for c in px},
                        "window": {c: {r: stats(px[c][r][win]) for r in ROLES} for c in px},
                        "all_cams_per_frame_median": {r: stats(allc[r]) for r in ROLES},
                        "all_cams_per_frame_median_window": {r: stats(allc[r][win]) for r in ROLES}}
    frames.update({f"px_{c}_{r}": px[c][r] for c in px for r in ROLES})
    frames.update({f"allpx_{r}": allc[r] for r in ROLES})
    if fpx:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            fall = {r: {k: np.nanmedian(np.stack([fpx[c][r][k] for c in fpx]), 0) for k in ("feet", *FEET)}
                    for r in ROLES}
        out["reproj_feet_px"] = {
            "points": "mamma_dyn.part_index left_foot + right_foot (88 of the 512)",
            "whole": {c: {r: stats(fpx[c][r]["feet"]) for r in ROLES} for c in fpx},
            "window": {c: {r: stats(fpx[c][r]["feet"][win]) for r in ROLES} for c in fpx},
            "all_cams_per_frame_median": {r: stats(fall[r]["feet"]) for r in ROLES},
            "all_cams_per_frame_median_window": {r: stats(fall[r]["feet"][win]) for r in ROLES},
            "per_foot_all_cams": {r: {f: stats(fall[r][f]) for f in FEET} for r in ROLES}}
        frames.update({f"fpx_{c}_{r}": fpx[c][r]["feet"] for c in fpx for r in ROLES})
        frames.update({f"fpx_{f[0]}_{r}": fall[r][f] for r in ROLES for f in FEET})
        frames.update({f"fpx_{r}": fall[r]["feet"] for r in ROLES})
    jit = root_jitter(sm)
    cmp = None
    if compare_take:
        cw = PACKAGE_ROOT / "work" / compare_take
        cmp = load(cw / "smplx")
        px6, fpx6 = per_frame_cam_px(cmp, *grid_dirs(cw))
        out["compare"] = {"take": compare_take, "smplx": f"work/{compare_take}/smplx",
                          "span_s": [round(float(cmp["lead"]["t"][0]), 3), round(float(cmp["lead"]["t"][-1]), 3)],
                          "reproj_px": {c: {r: stats(px6[c][r]) for r in ROLES} for c in px6},
                          "all_cams_per_frame_median": {r: stats(np.nanmedian(np.stack([px6[c][r] for c in px6]), 0))
                                                        for r in ROLES}}
        if fpx6:
            out["compare"]["feet_all_cams_per_frame_median"] = {
                r: stats(np.nanmedian(np.stack([fpx6[c][r]["feet"] for c in fpx6]), 0)) for r in ROLES}
        jit6 = root_jitter(cmp)
        out["compare"]["jitter_4p5_12hz_rms_mm"] = {r: round(float(np.sqrt(np.mean(jit6[r]["mag_mm"] ** 2))), 2)
                                                   for r in ROLES}
        frames.update({f"cmp_px_{c}_{r}": px6[c][r] for c in px6 for r in ROLES})
        frames["cmp_t"] = np.asarray(cmp["lead"]["t"], float)
    out["jitter_4p5_12hz_mm"] = {}
    for r in ROLES:
        mag = jit[r]["mag_mm"]
        turn = dilate(np.abs(heading_rate(sm[r])) > TURN_DEG_S, 3)
        frames[f"jit_{r}"], frames[f"turn_{r}"] = mag, turn
        rms = lambda m: round(float(np.sqrt(np.mean(mag[m] ** 2))), 2) if m.any() else None
        wr, w0 = np.array(jit[r]["win_rms_mm"]), np.array(jit[r]["win_t0"])
        out["jitter_4p5_12hz_mm"][r] = {
            "whole_rms": rms(np.ones_like(turn)), "turn_rms": rms(turn), "no_turn_rms": rms(~turn),
            "turn_fraction": round(float(turn.mean()), 3), "window_rms": rms(win),
            "worst_0p5s_windows": [[round(float(w0[i]), 2), round(float(wr[i]), 1)] for i in np.argsort(-wr)[:6]],
            "p95_0p5s_window": round(float(np.percentile(wr, 95)), 2)}
    out["turn_threshold_deg_s"] = TURN_DEG_S
    out["floor"], out["rest_height_m"], out["slip"], out["displacement_vs_ref_mm"] = {}, {}, {}, {}
    for r in ROLES:
        k = sole_kin(sm[r])
        kref = k if ref is sm else sole_kin(ref[r])
        fm = floor_metrics(sm[r], k)
        low = fm["low"]
        frames[f"low_{r}"] = low
        frames[f"stance_l_{r}"], frames[f"stance_r_{r}"] = fm["stance"]["left"], fm["stance"]["right"]
        out["floor"][r] = {
            "lowest_sole_mm": mm_stats(low), "frac_below_15mm": round(float((low < -0.015).mean()), 3),
            "frac_above_15mm": round(float((low > 0.015).mean()), 3),
            "frac_above_30mm": round(float((low > 0.030).mean()), 3),
            "stance_sole_mm": mm_stats(fm["stance_heights"]),
            "stance_abs_gt_15mm_frac": round(float((np.abs(fm["stance_heights"]) > 0.015).mean()), 3),
            "stance_slip_mm": {"median": round(float(1000 * np.median(fm["slide_m"])), 1),
                               "p90": round(float(1000 * np.percentile(fm["slide_m"], 90)), 1),
                               "runs": int(len(fm["slide_m"])),
                               "note": "this motion's OWN stance runs (not comparable across variants: see slip)"}
            if len(fm["slide_m"]) else None,
            "lowest_foot_joint_minus_rest_mm": mm_stats(fm["foot_joint"] - fm["rest_foot_joint_above_sole"])}
        sd = slip_and_displacement(sm[r], k, None if ref is sm else ref[r], kref, frames, r)
        out["slip"][r] = sd["slip"]
        if sd["displacement"] is not None:
            out["displacement_vs_ref_mm"][r] = sd["displacement"]
        h = ws.rest_height(sm[r]["betas"], ws.body_model(ws.model_path_for(sm[r])))
        tgt = statures[r] + float(soles.get(r, 0.025) if isinstance(soles, dict) else soles)
        row = {"fit": round(h, 4), "target_stature_plus_sole": round(tgt, 4), "rel_vs_target": round(h / tgt - 1, 4)}
        if cmp is not None:
            h6 = ws.rest_height(cmp[r]["betas"], ws.body_model(ws.model_path_for(cmp[r])))
            row.update({"compare_fit": round(h6, 4), "compare_rel_vs_target": round(h6 / tgt - 1, 4),
                        "compare_rel_vs_stature": round(h6 / statures[r] - 1, 4)})
        out["rest_height_m"][r] = row
    if physics:
        from . import physics as ph
        npzs = {r: ph.load_npz(work / smplx_dir / f"{r}.npz") for r in ROLES}
        npzs = {r: {k: v for k, v in d.items() if k not in ("smoothing",)} for r, d in npzs.items()}
        pdir = work / ("physics" if smplx_dir == "smplx" else f"physics_{smplx_dir}")
        res = ph.run(work, ph.PhysicsConfig(), None, statures=statures, out_dir=pdir, npzs=npzs, soles=soles)
        out["physics"] = {"dir": str(pdir.relative_to(PACKAGE_ROOT)),
                          **{r: {k: res[r]["info"].get(k) for k in ("mass_kg", "contact_fraction",
                                                                    "f_net_band95_median_n", "stature_rel_diff")}
                             for r in ROLES}}
    qd = work / "quality"
    qd.mkdir(exist_ok=True)
    (qd / f"quality_{tag}.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    np.savez_compressed(qd / f"quality_{tag}_frames.npz", **frames)
    return out


# ------------------------------------------------------------------------------------------------ variants table

def _g(d, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def variants(take: str, tags: list[str], ref_tag: str = "final", out_name: str = "variants",
             figure: bool = True) -> dict:
    """Compare quality_<tag> files (see the module doc) -> quality/<out_name>.json (+ review figure)."""
    work = PACKAGE_ROOT / "work" / take
    qd = work / "quality"
    J = {t: json.loads((qd / f"quality_{t}.json").read_text()) for t in [ref_tag, *tags]}
    F = {t: dict(np.load(qd / f"quality_{t}_frames.npz")) for t in [ref_tag, *tags]}
    fr = F[ref_tag]
    table = {"take": take, "ref": ref_tag, "variants": {}}
    for t in tags:
        j, f = J[t], F[t]
        sm = _g(j, "notes", "smoothing") or {}
        row = {"smplx": j["smplx"], "mode": sm.get("mode"),
               "params": {**{k: sm.get("filter", {}).get(k) for k in ("root_hz", "rot_hz")},
                          **{k: v for k, v in (sm.get("refine") or {}).items()
                             if k in ("w_floor", "floor_mm", "w_feet", "w_slip", "slip_max_v", "root_xz", "iters")}}}
        for r in ROLES:
            d = {}
            d["reproj_all_px"] = _g(j, "reproj_px", "all_cams_per_frame_median", r)
            d["reproj_feet_px"] = _g(j, "reproj_feet_px", "all_cams_per_frame_median", r)
            d["reproj_all_per_cam_median_px"] = {c: _g(v, r, "median") for c, v in j["reproj_px"]["whole"].items()}
            d["reproj_feet_per_cam_median_px"] = {c: _g(v, r, "median")
                                                  for c, v in (_g(j, "reproj_feet_px", "whole") or {}).items()}
            d["reproj_all_per_cam_delta_px"] = {
                c: (round(v - J[ref_tag]["reproj_px"]["whole"][c][r]["median"], 2) if v is not None else None)
                for c, v in d["reproj_all_per_cam_median_px"].items()}
            d["reproj_feet_per_cam_delta_px"] = {
                c: (round(v - J[ref_tag]["reproj_feet_px"]["whole"][c][r]["median"], 2) if v is not None else None)
                for c, v in d["reproj_feet_per_cam_median_px"].items()}
            # feet error change where the reference had a planted foot > 15 mm above the floor (did the cameras
            # support the raised foot?) and where it was within 15 mm
            raised, level = [], []
            for ff in ("l", "r"):
                key = f"fpx_{ff}_{r}"
                if key not in f or key not in fr:
                    continue
                m = fr[f"rst_{ff}_{r}"].astype(bool)
                hi = m & (fr[f"refsole_{ff}_{r}"] > 0.015)
                lo = m & (np.abs(fr[f"refsole_{ff}_{r}"]) <= 0.015)
                dd = f[key] - fr[key]
                raised.append(dd[hi])
                level.append(dd[lo])
            if raised:
                rs, lv = np.concatenate(raised), np.concatenate(level)
                d["feet_px_delta_ref_stance_raised_gt15mm"] = {"median": round(float(np.nanmedian(rs)), 2),
                                                               "n": int(np.isfinite(rs).sum())}
                d["feet_px_delta_ref_stance_level"] = {"median": round(float(np.nanmedian(lv)), 2),
                                                       "n": int(np.isfinite(lv).sum())}
            d["displacement_mm"] = {
                "pelvis_horiz": _g(j, "displacement_vs_ref_mm", r, "pelvis", "all", "horiz"),
                "ankles_stance_horiz": None, "ankles_stance_vert": None, "ankles_swing_horiz": None}
            if f"dh_left_ankle_{r}" in f:
                st = {ff: fr[f"rst_{ff[0]}_{r}"].astype(bool) for ff in FEET}
                cat = lambda kind, sel: np.concatenate([f[f"{kind}_{ff}_ankle_{r}"][sel(st[ff])] for ff in FEET])
                d["displacement_mm"].update({
                    "ankles_stance_horiz": mm_tail(cat("dh", lambda m: m)),
                    "ankles_stance_vert": mm_tail(cat("dv", lambda m: m)),
                    "ankles_swing_horiz": mm_tail(cat("dh", lambda m: ~m)),
                    "ankles_all_3d_p99": round(float(1000 * np.percentile(np.concatenate(
                        [np.hypot(f[f"dh_{ff}_ankle_{r}"], f[f"dv_{ff}_ankle_{r}"]) for ff in FEET]), 99)), 1)})
            fl = j["floor"][r]
            d["floor"] = {k: fl[k] for k in ("frac_above_15mm", "frac_below_15mm", "frac_above_30mm")}
            d["slip"] = {"contact_point_speed_ref_stance_mm_s":
                         _g(j, "slip", r, "both_feet", "contact_point_speed_ref_stance_mm_s"),
                         "ref_run_drift_mm": _g(j, "slip", r, "both_feet", "ref_run_drift_mm"),
                         "own_stance_run_drift_mm": fl.get("stance_slip_mm")}
            pins = [k for k in f if k.startswith("pin_") and k.endswith(f"_{r}")]
            if pins:  # the reference's slip on THIS variant's pinned frames vs this variant's (in-sample)
                pv, rv = [], []
                for ff in ("l", "r"):
                    pm = f[f"pin_{ff}_{r}"].astype(bool)
                    pp = pm[1:] & pm[:-1]
                    pv.append(f[f"slipv_{ff}_{r}"][pp])
                    rv.append(fr[f"slipv_{ff}_{r}"][pp])
                d["slip"]["pinned_in_sample_mm_s"] = {"variant": mm_tail(np.concatenate(pv)),
                                                      "ref": mm_tail(np.concatenate(rv))}
                d["slip"]["ref_stance_not_pinned_mm_s"] = _g(
                    j, "slip", r, "both_feet", "contact_point_speed_ref_stance_not_pinned_mm_s")
            d["jitter_4p5_12hz_rms_mm"] = {k: j["jitter_4p5_12hz_mm"][r][k] for k in ("whole_rms", "turn_rms")}
            row[r] = d
        table["variants"][t] = row
    out = qd / f"{out_name}.json"
    out.write_text(json.dumps(table, indent=1, default=float), encoding="utf-8")
    table["_json"] = str(out)
    if figure:
        from .export_unity import resolve_smplx_dir
        d, src = resolve_smplx_dir(take)
        table["_png"] = str(plot_variants(table, work / "review" / f"pose_refine_{out_name}.png",
                                          adopted=d if src != "default" else None))
    return table


def plot_variants(table: dict, path: Path, adopted: str | None = None) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
    COL = {"lead": "#2a78d6", "follow": "#eb6834"}
    NAMES = {"lead": "Kadu (lead)", "follow": "Larissa (follow)"}
    tags = list(table["variants"])
    V = table["variants"]
    panels = [
        ("feet reprojection (px, median over frames; in-sample for variants with a feet term)",
         lambda d: _g(d, "reproj_feet_px", "median")),
        ("all-points reprojection (px)", lambda d: _g(d, "reproj_all_px", "median")),
        ("stance ankle shift vs MAMMA fit, horizontal (mm, median)",
         lambda d: _g(d, "displacement_mm", "ankles_stance_horiz", "median")),
        ("frames with the sole > 15 mm above the floor (%)", lambda d: 100 * d["floor"]["frac_above_15mm"]),
        ("frames with the sole > 15 mm below the floor (%)", lambda d: 100 * d["floor"]["frac_below_15mm"]),
        ("contact-point slip on the MAMMA fit's stance frames (mm/s, median)",
         lambda d: _g(d, "slip", "contact_point_speed_ref_stance_mm_s", "median")),
    ]
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.titlecolor": INK, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axs = plt.subplots(3, 2, figsize=(17, 13.5), facecolor="#fcfcfb")
    x = np.arange(len(tags))
    for ax, (title, get) in zip(axs.ravel(), panels):
        for k, r in enumerate(("lead", "follow")):
            vals = [get(V[t][r]) for t in tags]
            vals = [np.nan if v is None else float(v) for v in vals]
            bars = ax.bar(x + (k - 0.5) * 0.38, vals, 0.36, color=COL[r], label=NAMES[r])
            for b, v in zip(bars, vals):
                if np.isfinite(v):
                    ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8,
                            color=INK2)
        names = {t: ("MAMMA fit" if V[t]["smplx"].replace("\\", "/").endswith("/smplx") else t.replace("smplx_", "").replace("refined", "refined (1st run)"))
                 for t in tags}
        ax.set_xticks(x, [names[t] + (" (adopted)" if V[t]["smplx"].replace("\\", "/").endswith("/" + str(adopted))
                                      else "") for t in tags], rotation=20, ha="right")
        for lab, t in zip(ax.get_xticklabels(), tags):
            if V[t]["smplx"].replace("\\", "/").endswith("/" + str(adopted)):
                lab.set_fontweight("bold")
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_title(title, loc="left", fontsize=11)
        ax.margins(y=0.15)
    h, lab = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper right", ncol=2, frameon=False, fontsize=10)
    fig.suptitle(f"{table['take']}: motion clean-up variants vs the MAMMA fit ('{table['ref']}'); feet = 88 of MAMMA's "
                 f"512 points", x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=85, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m dancecap.pose_quality", description=__doc__.split("\n")[0])
    ap.add_argument("take")
    ap.add_argument("--smplx-dir", default="smplx")
    ap.add_argument("--tag", default=None, help="default: the smplx dir name")
    ap.add_argument("--ref-dir", default="smplx", help="reference fit for displacement and the fixed stance frames")
    ap.add_argument("--compare-take", default=None)
    ap.add_argument("--window", type=float, nargs=2, default=None)
    ap.add_argument("--physics", action="store_true")
    ap.add_argument("--variants", nargs="+", default=None, help="tags to compare (quality_<tag>.json must exist)")
    ap.add_argument("--ref-tag", default="final")
    ap.add_argument("--out-name", default="variants")
    a = ap.parse_args(argv)
    if a.variants:
        tb = variants(a.take, a.variants, a.ref_tag, a.out_name)
        brief = {t: {r: {"feet_px": _g(v, r, "reproj_feet_px", "median"), "all_px": _g(v, r, "reproj_all_px", "median"),
                         "ankle_stance_h_mm": _g(v, r, "displacement_mm", "ankles_stance_horiz", "median"),
                         "above15": v[r]["floor"]["frac_above_15mm"], "below15": v[r]["floor"]["frac_below_15mm"],
                         "slip_ref_stance": _g(v, r, "slip", "contact_point_speed_ref_stance_mm_s", "median")}
                     for r in ROLES} for t, v in tb["variants"].items()}
        print(json.dumps(brief, indent=1))
        print(f"[pose_quality] -> {tb['_json']}" + (f", {tb['_png']}" if "_png" in tb else ""))
        return
    o = run(a.take, a.smplx_dir, a.tag or a.smplx_dir, a.compare_take, tuple(a.window) if a.window else None,
            a.physics, a.ref_dir)
    print(json.dumps({"reproj_all_cams": o["reproj_px"]["all_cams_per_frame_median"],
                      "reproj_feet_all_cams": _g(o, "reproj_feet_px", "all_cams_per_frame_median"),
                      "jitter": {r: {k: v for k, v in d.items() if k != "worst_0p5s_windows"}
                                 for r, d in o["jitter_4p5_12hz_mm"].items()},
                      "floor": {r: {k: d[k] for k in ("frac_below_15mm", "frac_above_15mm", "stance_abs_gt_15mm_frac",
                                                      "stance_slip_mm")} for r, d in o["floor"].items()},
                      "slip": {r: d["both_feet"] for r, d in o["slip"].items()},
                      "physics": o.get("physics")}, indent=1, default=float))


if __name__ == "__main__":
    main()
