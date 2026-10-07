"""Metric-scale check and correction of the world frame (cameras/world_from_sfm.json).

  check        after a MAMMA fit: each dancer's SMPL-X REST height (crown to sole, T-pose, from the fitted betas and
               the model the fit used) vs stature + shoe sole (takes/<take>.toml shoe_sole_m, default 2.5 cm: they
               dance in sneakers). k = H_rest / (stature + sole) is how much too large the world is as seen by that
               dancer; f = 1 / geometric mean(k) is the correction. The two dancers are independent estimates: they
               must agree within SCALE_AGREE_REL (1.5 %), else a stature (or one dancer's fit) is wrong and nothing
               should be rescaled automatically. verdict: ok (|f - 1| <= tol AND each dancer |k - 1| <= tol) |
               rescale (dancers agree, mean or one dancer off by > tol) | inconsistent. --strict exits 2 unless ok.
               ABSOLUTE caveat: sole 2.5 cm vs 0 (are the statures barefoot?) moves f by ~1.5 % = tol, so the check
               cannot detect that assumption; only the user can (SCALE_ABS_REL).
  reference    a reference world for other takes of the same recordings: the take's world x its check factor, with a
               "reference" record (statures, soles, factor, mtimes). Refuses an inconsistent check or a fit older
               than its world.
  adopt_reference / [cameras] scale_reference in takes/<take>.toml: the cameras stage bootstraps the world (chain
               rule), then inherits the reference scale through the SfM similarity of the same camera frames when
               reference_gates pass (>= 4 cameras with >= 30 matched frames, centre residual < 1 % of the spread,
               gravity within 1 deg of the inherited one - compared, never changed). The chain is then a cross-check
               (diagnostics.scale_reference.chain_vs_reference; G2a world_scale fails beyond 2 %).
  rescale      world frame x f about the world origin (world.rescale; origin on the floor, so y = 0, axes and image
               projections are unchanged): rewrites cameras/world_from_sfm.json (previous -> world_from_sfm.prev.json),
               syncs room/transform.json (its world_from_sfm copy and its world-metre fields; previous ->
               transform.prev.json). Camera tracks cameras/<cam>.npz are SfM-frame by contract and stay as they are;
               every WORLD-metre product (smplx/, mamma/ inputs, physics/, timing/, avatar/, export/) is listed as stale.
  rescale_fit  offline preview of a fit in the rescaled world: joints x f, betas re-fitted (linear least squares) so
               the rest mesh is f x the old one about the pelvis, transl so that the pelvis lands at f x its old place.
               Not a substitute for a refit (MAMMA's priors and pose correctives are not re-run).
  compare      the similarity SfM_b -> SfM_a from the same physical camera frames (cameras/<cam>.npz camera centres of
               main-model cameras matched by frames.json pts), i.e. the scale (and gravity) a take would have if it
               inherited another take's (corrected) world. Cross-check for a new take of the same recordings.
               NOTE: once a take has ADOPTED a reference (diagnostics.scale_reference.adopted), compare against the
               same reference returns ratio_to_expected = 1 by construction: it checks nothing then.
  chain        re-measure the chain rule on the take's own cameras/<cam>.npz with sync/offsets.json applied to the
               frame times (after 'python -m dancecap sync'): chain_vs_world = s_chain / s_world - 1 and the
               change vs the pre-sync chain the cameras stage recorded. Coarse cross-check only (CHAIN_FRAC is
               calibrated on larissa_kadu's betas); expect |chain_vs_world| <= 2 %.

  check also reports each dancer's posed lowest mesh vertex over ALL frames (floor: median, p5, frames below
  -15 mm; LBS without pose correctives, mm-level) - MAMMA has no floor term, so feet sinking into the floor show here.
  ABSOLUTE scale: every check here is SELF-CONSISTENT only (the world, the reference and the post-fit check all use
  stature + shoe sole); none is an independent metric measurement (absolute.verified false). +-1.5 %.

CLI:  python -m dancecap.cameras.world_scale check     <take> [--sole 0.025] [--strict] [--no-floor]
      python -m dancecap.cameras.world_scale rescale   <take> --factor F --reason TEXT [--no-room] [--dry-run]
      python -m dancecap.cameras.world_scale compare   <take> --ref <take> [--ref-sim3 file.json]
      python -m dancecap.cameras.world_scale reference <ref take> --out file.json [--sole S] [--smplx-dir D]
      python -m dancecap.cameras.world_scale chain     <take> [--no-sync] [--out file.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from .world import CHAIN_FRAC_SOLE_M, SCALE_ABS_REL, SCALE_AGREE_REL, SCALE_MAX_VS_REFERENCE, SOLE_M, rescale

FLOOR_BELOW_M = 0.015  # posed lowest vertex this far below y = 0 counts as "in the floor"
FLOOR_WARN_FRAC = 0.10  # warn when more frames than this are in the floor
STATURE_MATCH_M = 0.0005  # reference statures must equal the take's within this
REF_INPUTS = ("cameras/sfm/cameras.bin", "cameras/sfm/images.bin", "cameras/sfm/points3D.bin",
              "cameras/[0-9][0-9].npz", "frames/[0-9][0-9]/frames.json")  # what the SfM similarity reads
ABS_NOTE = ("SELF-CONSISTENT, not absolutely verified: the world, the reference and this check all use stature + shoe "
            "sole; whether the statures (whole inches) are barefoot heights is unconfirmed (sole 0 vs 2.5 cm = 1.5 %) "
            "and no object of known size has been measured")

ROLES = ("lead", "follow")
STALE_AFTER_RESCALE = ("smplx", "mamma/prepare.json", "mamma/ma_cap", "mamma/ma_2d", "physics", "timing",
                       "avatar", "export", "counterbalance", "moves", "sync/offsets.json")
_MODELS: dict[str, dict] = {}


# ------------------------------------------------------------------------------------------------ body model

def body_model(path: Path) -> dict:
    """SMPL-X npz arrays needed for the rest shape (cached per path)."""
    key = str(Path(path).resolve())
    if key not in _MODELS:
        m = np.load(path, allow_pickle=True)
        _MODELS[key] = {"path": key, "v_template": np.asarray(m["v_template"], float),
                        "shapedirs": np.asarray(m["shapedirs"], float), "J_regressor": np.asarray(m["J_regressor"], float)}
    return _MODELS[key]


def rest_vertices(betas: np.ndarray, model: dict) -> np.ndarray:
    b = np.asarray(betas, float).reshape(-1)
    return model["v_template"] + model["shapedirs"][:, :, :len(b)] @ b


def rest_height(betas: np.ndarray, model: dict) -> float:
    """Crown-to-sole height of the shaped T-pose mesh (= physics_body.BodyModel.stature)."""
    v = rest_vertices(betas, model)
    return float(v[:, 1].max() - v[:, 1].min())


def model_path_for(npz) -> Path:
    from ..physics_body import model_file
    return model_file(npz)


# ------------------------------------------------------------------------------------------------ check

def betas_check(npzs: dict, statures: dict[str, float], sole_m: float | dict = SOLE_M, tol: float = SCALE_AGREE_REL,
                models: dict | None = None) -> dict:
    """npzs[role] = contract smplx npz (betas, gender, model). models[role] overrides the model dict (tests).
    sole_m: one value or {role: value}. verdict "ok" needs EACH dancer's rest height within tol of its target (not
    only their geometric mean); "rescale" when they agree (<= tol) but the mean or one dancer is off by > tol (after
    x factor both are within tol / 2 of their targets); "inconsistent" when they disagree by > tol."""
    soles = sole_m if isinstance(sole_m, dict) else {r: float(sole_m) for r in ROLES}
    per = {}
    for role in ROLES:
        if role not in npzs or role not in statures:
            continue
        model = (models or {}).get(role) or body_model(model_path_for(npzs[role]))
        H = rest_height(npzs[role]["betas"], model)
        target = statures[role] + soles.get(role, SOLE_M)
        per[role] = {"rest_height_m": H, "stature_m": statures[role], "sole_m": soles.get(role, SOLE_M),
                     "target_m": target, "k": H / target, "excess_m": H - target, "ok": bool(abs(H / target - 1) <= tol),
                     "model": model.get("path", "?")}
    if not per:
        raise ValueError("no dancer with both a fit and a stature")
    ks = np.array([v["k"] for v in per.values()])
    f = float(1.0 / np.exp(np.mean(np.log(ks))))
    out = {"per_dancer": per, "sole_m": soles, "factor": f, "tol": tol,
           "disagreement": float(ks.max() / ks.min() - 1) if len(ks) == 2 else None,
           "max_per_dancer_rel": float(np.abs(ks - 1).max())}
    if out["disagreement"] is not None and out["disagreement"] > tol:
        out["verdict"] = "inconsistent"
        out["advice"] = ("the two dancers imply different scales: check the statures in takes/<take>.toml and each "
                         "dancer's fit (identity swaps, hair/head) before rescaling")
    elif abs(1.0 / f - 1.0) <= tol and all(v["ok"] for v in per.values()):
        out["verdict"] = "ok"
        out["advice"] = "each dancer's rest height matches stature + sole within tolerance: no rescale"
    else:
        out["verdict"] = "rescale"
        out["advice"] = (f"world is {100 * (1 / f - 1):+.1f} % off (worst dancer {100 * out['max_per_dancer_rel']:.1f} %): "
                         f"python -m dancecap.cameras.world_scale rescale <take> --factor {f:.5f} --reason ..., then "
                         "re-run mamma prepare/fit/select and everything downstream")
    out["sole_sensitivity"] = {}
    for s in (0.0, 0.02, 0.025, 0.03):
        kk = np.array([v["rest_height_m"] / (v["stature_m"] + s) for v in per.values()])
        out["sole_sensitivity"][f"{s:.3f}"] = float(1.0 / np.exp(np.mean(np.log(kk))))
    out["note"] = ("absolute: the target assumes barefoot statures and the mesh sole on the shoe sole; sole 0 vs 2.5 cm "
                   "moves the factor by ~1.5 % (= tol), so this check cannot tell them apart - the user must")
    return out


def floor_stats(npz: dict, chunk: int = 32) -> dict:
    """Posed lowest mesh vertex (world y, m) of a contract smplx fit over ALL its frames (LBS of the shaped rest mesh,
    no pose correctives: mm-level). MAMMA's optimisation has no floor term, so this is where sunk feet show."""
    from .. import physics_body as pb
    body = pb.load_body(npz)
    pose = pb.full_pose(npz)
    R, p = pb.forward_kinematics(body, pose, np.asarray(npz["transl"], float).reshape(len(pose), 3))
    V = np.arange(len(body.v_rest))
    low = np.concatenate([pb.lbs_points(body, R[s:s + chunk], p[s:s + chunk], V)[..., 1].min(1)
                          for s in range(0, len(R), chunk)])
    n_in = int((low < -FLOOR_BELOW_M).sum())
    return {"n_frames": int(len(low)), "median_mm": float(1000 * np.median(low)),
            "p5_mm": float(1000 * np.percentile(low, 5)), "p95_mm": float(1000 * np.percentile(low, 95)),
            "min_mm": float(1000 * low.min()), f"n_below_{1000 * FLOOR_BELOW_M:.0f}mm": n_in,
            "frac_below": float(n_in / max(1, len(low))), "warning": bool(n_in > FLOOR_WARN_FRAC * len(low)),
            "lowest_vertex_m": low}


def check_take(take_name: str, sole_m: float | None = None, smplx_dir: Path | None = None,
               floor: bool = True) -> dict:
    """sole_m None = the take's shoe_sole_m (world.sole_from_take). floor: add floor_stats per dancer."""
    from ..take import Take
    from .world import sole_from_take, statures_from_take
    take = Take.load(take_name)
    sd = Path(smplx_dir) if smplx_dir else take.work / "smplx"
    npzs = {r: dict(np.load(sd / f"{r}.npz", allow_pickle=True)) for r in ROLES if (sd / f"{r}.npz").exists()}
    st, src = statures_from_take(take.cfg)
    soles, sole_src = sole_from_take(take.cfg)
    if sole_m is not None:
        soles, sole_src = {r: float(sole_m) for r in ROLES}, "--sole"
    res = betas_check(npzs, st, soles)
    res["statures_source"] = src
    res["sole_source"] = sole_src
    res["smplx_dir"] = str(sd)
    res["absolute"] = {"verified": False, "uncertainty_rel": SCALE_ABS_REL, "note": ABS_NOTE,
                       "verdict_means": "ok = the fitted bodies match stature + sole in this world (MAMMA is "
                                        "consistent with the scale it was given), not that the scale is right"}
    if floor:
        res["floor"] = {}
        for r, z in npzs.items():
            fs = floor_stats(z)
            fs.pop("lowest_vertex_m")
            res["floor"][r] = fs
        res["floor_note"] = (f"posed lowest vertex over all frames; warning when > {FLOOR_WARN_FRAC:.0%} of frames are "
                             f"more than {1000 * FLOOR_BELOW_M:.0f} mm below y = 0 (a rescale only multiplies these "
                             "heights by its factor)")
    wp = take.work / "cameras" / "world_from_sfm.json"
    if wp.exists():
        w = json.loads(wp.read_text())
        res["world"] = {"s": w["s"], "scale_rule": w.get("scale_rule", "eye_p90 (pre-2026-10-06 file)"),
                        "s_corrected": w["s"] * res["factor"]}
    return res


# ------------------------------------------------------------------------------------------------ rescale

def rescale_room_transform(tr: dict, f: float, new_sim3: dict) -> dict:
    """room/transform.json in a world scaled by f: world_from_sfm copy + the fields expressed in world metres."""
    out = json.loads(json.dumps(tr))
    out["world_from_sfm"] = {"s": new_sim3["s"], "R": new_sim3["R"], "t": new_sim3["t"]}
    fp = out.get("floor_plane_sfm", {})
    if fp.get("world_height_at_origin_m") is not None:
        fp["world_height_at_origin_m"] = float(fp["world_height_at_origin_m"]) * f
    m = out.get("mirror")
    if m:
        if m.get("top_m") is not None:
            m["top_m"] = float(m["top_m"]) * f
        if m.get("cams_dist_m") is not None:
            m["cams_dist_m"] = [round(float(x) * f, 3) for x in m["cams_dist_m"]]
        mw = m.get("world")
        if mw:
            if mw.get("d") is not None:
                mw["d"] = float(mw["d"]) * f  # n.X + d = 0 -> n.(fX) + f d = 0
            for k in ("along_range", "height_range_m"):
                if mw.get(k) is not None:
                    mw[k] = [float(x) * f for x in mw[k]]
    out.setdefault("rescaled", []).append({"factor": float(f), "note": "world-metre fields scaled by world_scale.rescale_world; "
                                           "SfM-frame fields and the room assets are unchanged"})
    return out


def rescale_world(work: Path, f: float, reason: str, room: bool = True, dry_run: bool = False) -> dict:
    """Scale the take's world frame by f (see module docstring). Returns a report incl. the stale products."""
    work = Path(work)
    p = work / "cameras" / "world_from_sfm.json"
    sim3 = json.loads(p.read_text())
    new = rescale(sim3, f, reason)
    rep = {"factor": float(f), "s_before": float(sim3["s"]), "s_after": new["s"], "reason": reason,
           "camera_tracks": "cameras/<cam>.npz are SfM-frame (contract): unchanged",
           "stale": [x for x in STALE_AFTER_RESCALE if (work / x).exists()], "room_synced": False, "dry_run": dry_run}
    tp = work / "room" / "transform.json"
    if not dry_run:
        shutil.copyfile(p, p.with_name("world_from_sfm.prev.json"))
        p.write_text(json.dumps(new, indent=1))
        if room and tp.exists():
            tr = json.loads(tp.read_text())
            shutil.copyfile(tp, tp.with_name("transform.prev.json"))
            tp.write_text(json.dumps(rescale_room_transform(tr, f, new), indent=1))
            rep["room_synced"] = True
    elif room and tp.exists():
        rep["room_synced"] = "would sync"
    return rep


def rescale_fit(npz: dict, f: float, model: dict, ridge: float = 1e-8) -> tuple[dict, dict]:
    """A contract smplx fit moved into the world scaled by f (offline preview; see module docstring)."""
    b = np.asarray(npz["betas"], float).reshape(-1)
    B = len(b)
    vt, S, Jr = model["v_template"], model["shapedirs"][:, :, :B], model["J_regressor"]
    v = vt + S @ b
    j0 = Jr[0] @ v
    # rest mesh relative to the pelvis, linear in the betas: (vt - Jr0 vt) + (S - Jr0 S) b'
    A = (S - np.einsum("v,vcb->cb", Jr[0], S)[None]).reshape(-1, B)
    c0 = (vt - Jr[0] @ vt).reshape(-1)
    rhs = (f * (v - j0)).reshape(-1) - c0
    b2 = np.linalg.solve(A.T @ A + ridge * np.eye(B), A.T @ rhs)
    res = (A @ b2 - rhs).reshape(-1, 3)
    v2 = vt + S @ b2
    j02 = Jr[0] @ v2
    J, J2 = Jr @ v, Jr @ v2
    jres = (J2 - j02) - f * (J - j0)
    out = {k: (np.array(npz[k]) if isinstance(npz[k], np.ndarray) else npz[k]) for k in npz}
    out["betas"] = b2.astype(np.asarray(npz["betas"]).dtype)
    out["transl"] = f * (j0[None] + np.asarray(npz["transl"], float)) - j02[None]
    out["joints"] = f * np.asarray(npz["joints"], float)
    out["rescaled_factor"] = np.float64(f)
    info = {"factor": float(f), "betas_before": b.tolist(), "betas_after": b2.tolist(),
            "rest_height_before_m": float(np.ptp(v[:, 1])), "rest_height_after_m": float(np.ptp(v2[:, 1])),
            "rest_height_target_m": float(f * np.ptp(v[:, 1])),
            "vertex_fit_rms_mm": float(1000 * np.sqrt((res ** 2).sum(1).mean())),
            "vertex_fit_max_mm": float(1000 * np.linalg.norm(res, axis=1).max()),
            "joint_fit_max_mm": float(1000 * np.linalg.norm(jres, axis=1).max())}
    return out, info


# ------------------------------------------------------------------------------------------------ compare

def camera_centres(work: Path, cams: list[str] | None = None, include_attached: bool = False) -> dict:
    """{cam: {"pts": (F,), "C": (F,3) SfM-frame camera centres, "ok": (F,)}} from cameras/<cam>.npz + frames.json."""
    work = Path(work)
    out = {}
    for p in sorted((work / "cameras").glob("[0-9][0-9].npz")):
        cam = p.stem
        if cams and cam not in cams:
            continue
        d = np.load(p)
        if not include_attached and "attached" in d.files and bool(d["attached"]):
            continue
        meta = json.loads((work / "frames" / cam / "frames.json").read_text())
        pts = np.array([fr["pts"] for fr in meta["frames"]], float)
        C = -np.einsum("fji,fj->fi", np.asarray(d["R"], float), np.asarray(d["tvec"], float))
        out[cam] = {"pts": pts, "C": C, "ok": np.asarray(d["ok"], bool)}
    return out


def umeyama(A: np.ndarray, B: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """s, R, t minimising |A - (s R B + t)|^2 (rows are points)."""
    ma, mb = A.mean(0), B.mean(0)
    a, b = A - ma, B - mb
    U, S, Vt = np.linalg.svd(a.T @ b / len(A))
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = float(np.trace(np.diag(S) @ D) / (b ** 2).sum(1).mean())
    return s, R, ma - s * R @ mb


def sfm_similarity(work_a: Path, work_b: Path, cams: list[str] | None = None, keep: float = 0.8,
                   pts_tol_s: float = 1e-3) -> dict:
    """X_a = s R X_b + t between two takes' SfM frames from the same camera frames (matched by source pts)."""
    ca, cb = camera_centres(work_a, cams), camera_centres(work_b, cams)
    A, Bm, who = [], [], []
    for cam in sorted(set(ca) & set(cb)):
        pa, pb = ca[cam]["pts"], cb[cam]["pts"]
        j = np.clip(np.searchsorted(pb, pa), 1, len(pb) - 1)
        j = np.where(np.abs(pb[j - 1] - pa) < np.abs(pb[j] - pa), j - 1, j)
        m = (np.abs(pb[j] - pa) < pts_tol_s) & ca[cam]["ok"] & cb[cam]["ok"][j]
        A.append(ca[cam]["C"][m])
        Bm.append(cb[cam]["C"][j[m]])
        who += [cam] * int(m.sum())
    if not A or sum(len(x) for x in A) < 10:
        raise ValueError("fewer than 10 common camera frames between the two takes")
    A, Bm, who = np.concatenate(A), np.concatenate(Bm), np.array(who)
    sel = np.ones(len(A), bool)
    for _ in range(4):  # trimmed re-fit (drops badly localised frames)
        s, R, t = umeyama(A[sel], Bm[sel])
        r = np.linalg.norm(A - (s * Bm @ R.T + t), axis=1)
        sel = r <= np.quantile(r, keep)
    s, R, t = umeyama(A[sel], Bm[sel])
    r = np.linalg.norm(A - (s * Bm @ R.T + t), axis=1)
    spread = float(np.sqrt(((A[sel] - A[sel].mean(0)) ** 2).sum(1).mean()))
    return {"s": s, "R": R.tolist(), "t": t.tolist(), "n": int(len(A)), "n_used": int(sel.sum()),
            "per_cam": {c: int((who == c).sum()) for c in np.unique(who)},
            "resid_median_a": float(np.median(r[sel])), "spread_a": spread,
            "resid_rel": float(np.median(r[sel]) / spread)}


def inherit_world(ref_sim3: dict, sim_ab: dict) -> dict:
    """World of take b if it inherits take a's world: X_w = s_a R_a (s R X_b + t) + t_a."""
    sa, Ra, ta = float(ref_sim3["s"]), np.asarray(ref_sim3["R"], float), np.asarray(ref_sim3["t"], float)
    s, R, t = float(sim_ab["s"]), np.asarray(sim_ab["R"], float), np.asarray(sim_ab["t"], float)
    Rb = Ra @ R
    U, _, Vt = np.linalg.svd(Rb)  # re-orthonormalise
    return {"s": sa * s, "R": (U @ Vt).tolist(), "t": (sa * Ra @ t + ta).tolist()}


def compare_reference(work: Path, ref_work: Path, ref_sim3: dict | None = None) -> dict:
    work, ref_work = Path(work), Path(ref_work)
    ref_sim3 = ref_sim3 or json.loads((ref_work / "cameras" / "world_from_sfm.json").read_text())
    sim = sfm_similarity(ref_work, work)
    inh = inherit_world(ref_sim3, sim)
    out = {"sfm_b_to_a": sim, "inherited_world": inh, "s_expected": inh["s"], "ref_s": float(ref_sim3["s"])}
    wp = work / "cameras" / "world_from_sfm.json"
    if wp.exists():
        w = json.loads(wp.read_text())
        up_this, up_inh = np.asarray(w["R"], float)[1], np.asarray(inh["R"], float)[1]
        out["this"] = {"s": float(w["s"]), "scale_rule": w.get("scale_rule"),
                       "ratio_to_expected": float(w["s"] / inh["s"]),
                       "gravity_vs_inherited_deg": float(np.degrees(np.arccos(np.clip(up_this @ up_inh, -1, 1))))}
        out["geometry_gates"] = reference_gates(sim, out["this"]["gravity_vs_inherited_deg"])
    return out


# ------------------------------------------------------------------------------------------------ reference scale

REF_MIN_CAMS = 4  # cameras with >= REF_MIN_FRAMES matched frames
REF_MIN_FRAMES = 30
REF_MAX_RESID_REL = 0.01  # median camera-centre residual / spread of the centres
REF_MAX_GRAVITY_DEG = 1.0  # this take's gravity vs the inherited one


def reference_gates(sim_ab: dict, gravity_deg: float) -> dict:
    """Is the SfM_b -> SfM_a similarity good enough to inherit a scale through it?"""
    n_cams = int(sum(1 for v in sim_ab["per_cam"].values() if v >= REF_MIN_FRAMES))
    why = []
    if n_cams < REF_MIN_CAMS:
        why.append(f"only {n_cams} cameras with >= {REF_MIN_FRAMES} matched frames (need {REF_MIN_CAMS})")
    if not sim_ab["resid_rel"] < REF_MAX_RESID_REL:
        why.append(f"camera-centre residual {sim_ab['resid_rel']:.4f} of the spread (need < {REF_MAX_RESID_REL})")
    if not gravity_deg <= REF_MAX_GRAVITY_DEG:
        why.append(f"gravity differs from the inherited one by {gravity_deg:.2f} deg (> {REF_MAX_GRAVITY_DEG}): "
                   "reported, not changed")
    return {"n_cams": n_cams, "resid_rel": float(sim_ab["resid_rel"]), "gravity_vs_inherited_deg": float(gravity_deg),
            "passed": not why, "why": "; ".join(why) or "geometry gates passed"}


def inputs_fingerprint(work: Path) -> dict:
    """sha1 + size + mtime of the reference take's files the SfM similarity depends on (REF_INPUTS: the SfM model,
    the camera tracks, the frame pts), keyed by path relative to work, plus one combined sha1."""
    work = Path(work)
    files = {}
    for pat in REF_INPUTS:
        for p in sorted(work.glob(pat)):
            h = hashlib.sha1()
            with open(p, "rb") as fh:
                for blk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(blk)
            st = p.stat()
            files[p.relative_to(work).as_posix()] = {"sha1": h.hexdigest(), "size": st.st_size, "mtime": st.st_mtime}
    comb = hashlib.sha1("".join(f"{k}:{v['sha1']};" for k, v in sorted(files.items())).encode()).hexdigest()
    return {"files": files, "sha1": comb, "patterns": list(REF_INPUTS)}


def verify_inputs(rec: dict | None, ref_work: Path) -> tuple[bool | None, str]:
    """Reference record's inputs fingerprint vs the reference take now: (True, ok) | (False, what changed) |
    (None, not recorded)."""
    if not rec or not rec.get("inputs"):
        return None, ("reference record has no inputs fingerprint (built before 2026-10-06 evening): rebuild it with "
                      "'python -m dancecap.cameras.world_scale reference <ref take> --out <file>'")
    now = inputs_fingerprint(ref_work)["files"]
    then = rec["inputs"]["files"]
    changed = sorted(k for k in set(then) | set(now) if (then.get(k) or {}).get("sha1") != (now.get(k) or {}).get("sha1"))
    if changed:
        return False, (f"reference take's camera inputs changed since the reference was built ({len(changed)} files: "
                       f"{', '.join(changed[:6])}{' ...' if len(changed) > 6 else ''}): the reference scale is tied to "
                       "the old SfM frame / tracks - rebuild it (world_scale reference) after re-fitting that take")
    return True, f"reference inputs unchanged ({len(now)} files)"


def make_reference(ref_take: str, sole_m: float | None = None, smplx_dir: Path | None = None) -> dict:
    """The reference take's world x its post-fit betas factor (check_take), plus a "reference" record (take, factor,
    statures, soles, verdict, mtimes, inputs fingerprint) that adopt_reference verifies. Refuses an inconsistent check
    and a fit older than the world it was supposedly fitted in (a world rescaled after the fit would be corrected
    twice)."""
    from ..take import Take
    take = Take.load(ref_take)
    chk = check_take(ref_take, sole_m, smplx_dir, floor=False)
    if chk["verdict"] == "inconsistent":
        raise ValueError(f"{ref_take}: betas check inconsistent ({chk['disagreement']:.3%}): no reference")
    wp = take.work / "cameras" / "world_from_sfm.json"
    sd = Path(chk["smplx_dir"])
    t_fit = min((sd / f"{r}.npz").stat().st_mtime for r in chk["per_dancer"])
    if t_fit < wp.stat().st_mtime:
        raise ValueError(f"{sd} is older than {wp}: the fit is not in this world (re-fit, or pass --smplx-dir)")
    w = json.loads(wp.read_text())
    f = chk["factor"]
    corr = rescale(w, f, f"{ref_take} MAMMA betas rest heights vs stature + sole (world_scale check)")
    corr["reference"] = {"take": ref_take, "factor": f, "s_world": float(w["s"]), "s_reference": corr["s"],
                         "statures": {r: v["stature_m"] for r, v in chk["per_dancer"].items()},
                         "sole_m": chk["sole_m"], "sole_source": chk["sole_source"],
                         "per_dancer_k": {r: v["k"] for r, v in chk["per_dancer"].items()},
                         "disagreement": chk["disagreement"], "verdict": chk["verdict"], "smplx_dir": str(sd),
                         "smplx_mtime": t_fit, "world_mtime": wp.stat().st_mtime,
                         "inputs": inputs_fingerprint(take.work),
                         "chain_frac_calibration_sole_m": CHAIN_FRAC_SOLE_M,
                         "abs_uncertainty_rel": SCALE_ABS_REL, "abs_verified": False,
                         "note": "absolute scale rests on the statures and the sole assumption (+-1.5 %): " + ABS_NOTE}
    return corr


def reference_from_take(cfg: dict) -> dict | None:
    """takes/<take>.toml [cameras] scale_reference = {take = "<ref take>", sim3 = "<file made by make_reference>"}
    (sim3 relative to the dancecap package root) -> {"take", "work", "sim3_path", "sim3"} or None."""
    from ..take import PACKAGE_ROOT, Take
    r = cfg.get("cameras", {}).get("scale_reference")
    if not r:
        return None
    p = Path(r["sim3"])
    p = p if p.is_absolute() else PACKAGE_ROOT / p
    return {"take": r["take"], "work": Take.load(r["take"]).work, "sim3_path": str(p),
            "sim3": json.loads(p.read_text())}


def adopt_reference(work: Path, sim3: dict, ref_work: Path, ref_sim3: dict, soles: dict | None = None,
                    ref_name: str = "?", statures: dict | None = None) -> tuple[dict, dict]:
    """This take's bootstrapped world (sim3, e.g. the chain rule) re-scaled to the scale it inherits from ref_sim3
    through the SfM similarity of the same physical camera frames, when the geometry gates pass (reference_gates), the
    reference was built with this take's soles AND statures (a corrected stature must rebuild the reference, else the
    world would silently keep the old-stature scale), and the reference take's SfM model / camera tracks / frame pts
    are unchanged since it was built (verify_inputs: refused if changed, a warning if the record predates
    fingerprints). R, origin and floor stay this take's own (rescale about the origin); gravity is compared, never
    changed. The chain becomes a cross-check: chain_vs_reference = s_chain / s_inherited - 1. Returns (new sim3,
    info); info is also diagnostics.scale_reference."""
    sim = sfm_similarity(ref_work, work)
    inh = inherit_world(ref_sim3, sim)
    up_this, up_inh = np.asarray(sim3["R"], float)[1], np.asarray(inh["R"], float)[1]
    grav = float(np.degrees(np.arccos(np.clip(up_this @ up_inh, -1, 1))))
    g = reference_gates(sim, grav)
    why = [] if g["passed"] else [g["why"]]
    rs = ref_sim3.get("reference") or {}
    rebuild = "(python -m dancecap.cameras.world_scale reference <ref take> --out <file>)"
    if soles is not None and rs.get("sole_m") is not None:
        bad = {r: (rs["sole_m"].get(r), soles[r]) for r in soles if rs["sole_m"].get(r) is not None
               and abs(float(rs["sole_m"][r]) - float(soles[r])) > 1e-6}
        if bad:
            why.append(f"reference built with shoe soles {rs['sole_m']} but this take uses {soles}: rebuild it {rebuild}")
    if statures is not None:
        rst = rs.get("statures")
        if rst is None:
            why.append(f"reference record has no statures: cannot check them against this take's {statures}: "
                       f"rebuild it {rebuild}")
        else:
            bad = {r: (rst.get(r), statures[r]) for r in statures
                   if rst.get(r) is None or abs(float(rst[r]) - float(statures[r])) > STATURE_MATCH_M}
            if bad:
                why.append(f"reference built with statures {rst} but this take uses {statures}: rebuild it with the "
                           f"corrected statures in the reference take's toml {rebuild}")
    inputs_ok, inputs_msg = verify_inputs(rs, ref_work)
    if inputs_ok is False:
        why.append(inputs_msg)
    ratio = float(sim3["s"]) / inh["s"]
    info = {"ref_take": ref_name, "ref_s": float(ref_sim3["s"]), "ref_record": rs or None,
            "s_bootstrap": float(sim3["s"]), "s_inherited": float(inh["s"]), "factor": 1.0 / ratio,
            "chain_vs_reference": ratio - 1.0, "gravity_vs_inherited_deg": grav, "n_cams": g["n_cams"],
            "resid_rel": g["resid_rel"], "per_cam": sim["per_cam"], "n_frames": sim["n"], "n_used": sim["n_used"],
            "sfm_scale_ref_from_this": sim["s"], "statures_checked": statures is not None,
            "ref_inputs_verified": inputs_ok, "ref_inputs": inputs_msg, "adopted": not why,
            "abs_verified": False, "chain_vs_reference_max": SCALE_MAX_VS_REFERENCE,
            "why": "; ".join(why) or "geometry gates passed: inherited"}
    if inputs_ok is None:
        info["warning"] = inputs_msg
    if info["ref_record"] and "inputs" in info["ref_record"]:  # keep the diagnostics small: the combined hash only
        info["ref_record"] = {**info["ref_record"], "inputs": {"sha1": rs["inputs"].get("sha1"),
                                                               "n_files": len(rs["inputs"].get("files", {}))}}
    if why:
        out = json.loads(json.dumps(sim3, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    else:
        out = rescale(sim3, 1.0 / ratio, f"inherit {ref_name}'s betas-corrected scale via the SfM similarity")
        out["scale_source"] = (f"inherited from {ref_name}'s betas-corrected world through the SfM similarity of "
                               f"{g['n_cams']} cameras' frames (s {inh['s']:.5f}); the bootstrap rule "
                               f"({sim3.get('scale_rule')}) gave {ratio - 1:+.2%} (cross-check)")
    out.setdefault("diagnostics", {})["scale_reference"] = info
    return out, info


# ------------------------------------------------------------------------------------------------ chain after sync

def load_main_tracks(work: Path, offsets: dict | None = None) -> dict:
    """cameras/<cam>.npz of the non-attached cameras as triangulate_dancers track dicts, frame times shifted by the
    sync offsets {cam: {"delta", "rate", "t_ref"}} (mamma_dyn.read_sync_offsets / apply_offset)."""
    from ..mamma_dyn import apply_offset
    out = {}
    for p in sorted((Path(work) / "cameras").glob("[0-9][0-9].npz")):
        d = np.load(p)
        if "attached" in d.files and bool(d["attached"]):
            continue
        t = np.asarray(d["t"], float)
        off = (offsets or {}).get(p.stem)
        dist = np.asarray(d["dist"], float)
        out[p.stem] = {"t": apply_offset(t, off) if off else t, "K": np.asarray(d["K"], float),
                       "k1": dist if dist.ndim == 1 else dist[:, 0], "R": np.asarray(d["R"], float),
                       "tvec": np.asarray(d["tvec"], float), "ok": np.asarray(d["ok"], bool)}
    return out


def chain_check_work(work: Path, sim3: dict, statures: dict, soles: dict, offsets: dict | None) -> dict:
    """Chain rule on the take's main cameras with the given sync offsets vs the world's scale (see module doc)."""
    from . import world
    tracks = load_main_tracks(work, offsets)
    if len(tracks) < 2:
        raise ValueError(f"{work}: fewer than 2 main-camera tracks")
    dancers = world.triangulate_dancers(work, tracks)
    chains, comb = world.chain_scale(dancers, statures, soles)
    if not comb:
        raise ValueError("no dancer has enough triangulated chain keypoints")
    d = sim3.get("diagnostics", {})
    pre = d.get("scale_chain", {})
    s_pre = None
    if pre:
        s_pre = float(np.exp(sum(d.get("scale_weights", {}).get(r, 1 / len(pre)) * np.log(v["s"])
                                 for r, v in pre.items())))
    out = {"s_world": float(sim3["s"]), "s_chain": comb["s"], "chain_vs_world": comb["s"] / float(sim3["s"]) - 1.0,
           "passed": bool(abs(comb["s"] / float(sim3["s"]) - 1.0) <= SCALE_MAX_VS_REFERENCE),
           "threshold": SCALE_MAX_VS_REFERENCE, "sync_applied": bool(offsets),
           "offsets_ms": {c: round(1000 * float(v.get("delta", 0.0)), 2) for c, v in (offsets or {}).items()},
           "cams": sorted(tracks), "combined": comb,
           "per_dancer": {r: {k: v[k] for k in ("s", "chain", "leg", "trunk", "shoulder_eye", "rel_se", "halves_rel_diff",
                                                "chain_stature_m", "n")} for r, v in chains.items()},
           "s_chain_pre_sync": s_pre,
           "sync_effect_rel": (comb["s"] / s_pre - 1.0) if s_pre else None,
           "world_scale_rule": sim3.get("scale_rule"),
           "world_adopted_reference": bool(d.get("scale_reference", {}).get("adopted")),
           "note": "coarse cross-check (CHAIN_FRAC calibrated on larissa_kadu's betas): |chain_vs_world| <= 2 %; never "
                   "rescale from it alone. " + ABS_NOTE}
    return out


def chain_check(take_name: str, use_sync: bool = True) -> dict:
    from ..take import Take
    from ..mamma_dyn import read_sync_offsets
    from .world import sole_from_take, statures_from_take
    take = Take.load(take_name)
    work = take.work
    sim3 = json.loads((work / "cameras" / "world_from_sfm.json").read_text())
    offsets = read_sync_offsets(work) if use_sync else {}
    if use_sync and not offsets:
        print(f"[world_scale] WARNING {work / 'sync' / 'offsets.json'} missing: chain on the audio clocks (no sync)")
    st, _ = statures_from_take(take.cfg)
    soles, _ = sole_from_take(take.cfg)
    return chain_check_work(work, sim3, st, soles, offsets)


# ------------------------------------------------------------------------------------------------ CLI

def main(argv: list[str] | None = None) -> None:
    from ..take import Take
    ap = argparse.ArgumentParser(prog="python -m dancecap.cameras.world_scale")
    ap.add_argument("action", choices=("check", "rescale", "compare", "reference", "chain"))
    ap.add_argument("take")
    ap.add_argument("--sole", type=float, default=None,
                    help=f"check/reference: shoe sole added to the statures (m; default the take's shoe_sole_m, "
                         f"else {SOLE_M})")
    ap.add_argument("--strict", action="store_true", help="check: exit status 2 unless the verdict is ok")
    ap.add_argument("--no-floor", action="store_true", help="check: skip the posed lowest-vertex floor statistics")
    ap.add_argument("--no-sync", action="store_true", help="chain: ignore sync/offsets.json (audio clocks)")
    ap.add_argument("--smplx-dir", help="check/reference: fit folder (default work/<take>/smplx)")
    ap.add_argument("--factor", type=float, help="rescale: world x factor")
    ap.add_argument("--reason", default="", help="rescale: recorded in diagnostics.rescaled")
    ap.add_argument("--no-room", action="store_true", help="rescale: leave room/transform.json alone")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--ref", help="compare: reference take")
    ap.add_argument("--ref-sim3", help="compare: reference world_from_sfm (e.g. a corrected copy)")
    ap.add_argument("--out", help="write the JSON result here")
    a = ap.parse_args(argv)
    if a.action == "check":
        res = check_take(a.take, a.sole, Path(a.smplx_dir) if a.smplx_dir else None, floor=not a.no_floor)
    elif a.action == "chain":
        res = chain_check(a.take, use_sync=not a.no_sync)
    elif a.action == "reference":
        if not a.out:
            raise SystemExit("reference: --out <file.json> is required")
        res = make_reference(a.take, a.sole, Path(a.smplx_dir) if a.smplx_dir else None)
    elif a.action == "rescale":
        if not a.factor or not a.reason:
            raise SystemExit("rescale: --factor and --reason are required")
        res = rescale_world(Take.load(a.take).work, a.factor, a.reason, room=not a.no_room, dry_run=a.dry_run)
    else:
        if not a.ref:
            raise SystemExit("compare: --ref <take> is required")
        ref_sim3 = json.loads(Path(a.ref_sim3).read_text()) if a.ref_sim3 else None
        res = compare_reference(Take.load(a.take).work, Take.load(a.ref).work, ref_sim3)
    txt = json.dumps(res, indent=1, default=float)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(txt)
    print(txt)
    if a.action == "check" and a.strict and res["verdict"] != "ok":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
