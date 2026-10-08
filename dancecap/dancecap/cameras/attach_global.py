"""One attach for the WHOLE dance of a camera that has no place in the main SfM model (cam05, cam06 of larissa_kadu).

stage_attach solves every segment take on its own: a rotation Rs (sub-model frame -> main frame) and ONE camera centre per
take, from the dancers alone (the static background of a sub-model has no parallax: its points are ~345 units away while
the camera moves a few units). The sub-model is the SAME static map in every take, so Rs is one number for the whole dance;
the per-take solves disagree on it by 2-13 degrees (larissa_kadu_full: cam06 a vs d 13.0 deg, b vs c 8.7 deg) because the
dancers fix a yaw only together with the centre (a yaw error is cancelled by a sideways shift of the centre) and the
operator also walks (the centre moved 0.2-0.8 m between takes; a constant centre per take leaves 133 px dancer errors at
cam06 90-100 s). Here:

  1. the sub-model localisations of every take (cached locsub_<cam>.pkl; for a ZOOMING camera with the per-frame focal of
     attach_zoom) and the dancers triangulated by the main-model cameras at the SYNC-CORRECTED instants (sync/offsets.json)
     are pooled over all takes,
  2. ONE yaw (tilt / roll from the building's vertical, as stage_attach's rotation-only+vertical) and ONE centre are fitted
     to all of them (attach.solve_rotonly),
  3. the centre becomes a piecewise-linear path (knots every knot_s seconds, weak smoothness) with the rotation held, and the
     path is kept only when it lowers the temporal HOLD-OUT dancer error (fit on alternating 0.5 s blocks, test on the others)
     by gate_gain over the constant centre.

apply() writes the result into one take: cameras/cache/loc_<cam>.npz (rotation R_sub Rs^T, centre C(t); the correspondences
are bearings from C(t) at 10 units, as stage_attach's rotation-only branch), attach_<cam>.npz, cameras/<cam>.npz (stage_smooth).

    python -m dancecap.cameras.attach_global solve --takes a,38,b,c,d --cams 05,06      # fit + hold-out, writes the solution
    python -m dancecap.cameras.attach_global apply --takes a,b,c,d --cams 05,06         # re-export those takes' cameras
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .attach import _rot_reproj_R, align_vectors, solve_rotonly

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "work"
SOL_DIR = WORK / "larissa_kadu_full" / "cameras_attach"


def take_name(seg: str) -> str:
    return "larissa_kadu_38" if seg == "38" else f"larissa_kadu_full_{seg}"


def _K(f: np.ndarray, w: int, h: int) -> np.ndarray:
    K = np.zeros((len(f), 3, 3))
    K[:, 0, 0] = K[:, 1, 1] = f
    K[:, 0, 2], K[:, 1, 2], K[:, 2, 2] = w / 2 - 0.5, h / 2 - 0.5, 1.0
    return K


def sync_deltas(work: Path) -> dict[str, float]:
    p = Path(work) / "sync" / "offsets.json"
    if not p.exists():
        return {}
    doc = json.loads(p.read_text())
    return {c: float(v["delta"]) for c, v in (doc.get("offsets") or {}).items() if v.get("use", True) and "delta" in v}


# ------------------------------------------------------------------------------------------------ observations

def dancer_observations(work: Path, cam: str, t: np.ndarray, ok: np.ndarray, main_cams: list[str], step: int = 3,
                        max_dt: float = 0.034, min_conf: float = 0.5, kps=tuple(range(5, 17)), sync: bool = True):
    """(fidx, X, uv): the dancers' COCO keypoints triangulated at the sync-corrected instant of every step-th ok frame of
    `cam` by the other main-model cameras (their EXPORTED tracks, cameras/<c>.npz), with this camera's detections."""
    from .world import dancer_keypoints, triangulate_robust
    delta = sync_deltas(work) if sync else {}
    D = {}
    for c in main_cams:
        p = work / "cameras" / f"{c}.npz"
        if p.exists():
            z = np.load(p)
            D[c] = {"t": z["t"] + delta.get(c, 0.0), "K": z["K"], "R": z["R"], "tvec": z["tvec"], "k1": z["dist"],
                    "ok": z["ok"].astype(bool)}
    KP = {c: dancer_keypoints(work, c, len(d["t"])) for c, d in D.items()}
    own = dancer_keypoints(work, cam, len(t))
    tc = t + delta.get(cam, 0.0)
    fi, X, uv = [], [], []
    for i in range(0, len(t), step):
        if not ok[i]:
            continue
        sel = []
        for c, d in D.items():
            j = int(np.clip(np.searchsorted(d["t"], tc[i]), 0, len(d["t"]) - 1))
            if j > 0 and abs(d["t"][j - 1] - tc[i]) < abs(d["t"][j] - tc[i]):
                j -= 1
            if abs(d["t"][j] - tc[i]) <= max_dt and d["ok"][j]:
                sel.append((c, j))
        if len(sel) < 2:
            continue
        for role in ("lead", "follow"):
            for k in kps:
                if own[role][i, k, 2] < min_conf:
                    continue
                v = [(c, j) for c, j in sel if KP[c][role][j, k, 2] >= min_conf]
                if len(v) < 2:
                    continue
                P, _, nv = triangulate_robust(
                    np.array([KP[c][role][j, k, :2] for c, j in v], float), np.array([D[c]["K"][j] for c, j in v]),
                    np.array([D[c]["k1"][j] for c, j in v]), np.array([D[c]["R"][j] for c, j in v]),
                    np.array([D[c]["tvec"][j] for c, j in v]))
                if P is None or nv < 2:
                    continue
                fi.append(i), X.append(P), uv.append(own[role][i, k, :2])
    return np.asarray(fi, int), np.asarray(X, float).reshape(-1, 3), np.asarray(uv, float).reshape(-1, 2)


def load_take(seg: str, cam: str, zoom: bool, step: int = 3, sync: bool = True) -> dict:
    """Everything the global solve needs from one take: sub-model rotations / focals per frame (the cached locsub), the
    frame times, the dancer observations, the building vertical measured in the sub-model frame and in the main frame."""
    import pycolmap

    from . import pipeline as P
    work = (WORK / take_name(seg)).resolve()
    poses = pickle.loads((work / "cameras" / "cache" / f"locsub_{cam}.pkl").read_bytes())["value"]
    meta = json.loads((work / "frames" / cam / "frames.json").read_text())
    w, h = int(meta["width"]), int(meta["height"])
    t = np.array([f["t"] for f in meta["frames"]], float)
    zoom_rep = None
    if zoom:
        from .attach_zoom import reference_centre, resolve_zoom
        sub = pycolmap.Reconstruction(str(work / "cameras" / "sfm_work" / f"sub_{cam}" / "best"))
        poses, zoom_rep = resolve_zoom(poses, reference_centre(sub), w, h, t=t)
    ok = np.array([p.ok for p in poses])
    R = np.array([p.R for p in poses])
    f = np.array([p.f if p.ok else np.nan for p in poses])
    f = np.where(ok, f, np.interp(t, t[ok], f[ok]))
    k1 = np.array([p.k1 for p in poses])
    rep = json.loads((work / "cameras" / "report.json").read_text())
    main = sorted(c for c, v in rep["cams"].items() if v.get("in_main_model"))
    fidx, X, uv = dancer_observations(work, cam, t, ok, main, step=step, sync=sync)
    vs = pickle.loads((work / "cameras" / "cache" / f"vertical_sub_{cam}.pkl").read_bytes())["value"]["pooled"]
    vm = pickle.loads((work / "cameras" / "cache" / "vertical_main.pkl").read_bytes())["value"]["fused"]
    return {"seg": seg, "work": work, "t": t, "R": R, "K": _K(f, w, h), "k1": k1, "ok": ok, "f": f, "w": w, "h": h,
            "fidx": fidx, "X": X, "uv": uv, "u_sub": None if vs is None else np.asarray(vs["u"], float),
            "u_sub_n": 0 if vs is None else int(vs["n_segments"]), "u_main": np.asarray(vm["u"], float),
            "zoom": zoom_rep}


def pool(takes: list[dict]) -> dict:
    """Concatenate the takes' frames and observations (fidx offset per take)."""
    off, R, K, k1, t, seg, fid, X, uv = 0, [], [], [], [], [], [], [], []
    for d in takes:
        R.append(d["R"]), K.append(d["K"]), k1.append(d["k1"]), t.append(d["t"])
        seg.append(np.full(len(d["t"]), d["seg"]))
        fid.append(d["fidx"] + off), X.append(d["X"]), uv.append(d["uv"])
        off += len(d["t"])
    u_sub = np.sum([d["u_sub"] * d["u_sub_n"] for d in takes if d["u_sub"] is not None], 0)
    return {"R": np.concatenate(R), "K": np.concatenate(K), "k1": np.concatenate(k1), "t": np.concatenate(t),
            "seg": np.concatenate(seg), "fidx": np.concatenate(fid), "X": np.concatenate(X), "uv": np.concatenate(uv),
            "u_sub": u_sub / np.linalg.norm(u_sub), "u_main": np.mean([d["u_main"] for d in takes], 0)}


# ------------------------------------------------------------------------------------------------ the solve

def rs_of(yaw: float, u_sub: np.ndarray, u_main: np.ndarray) -> np.ndarray:
    u_main = u_main / np.linalg.norm(u_main)
    return Rotation.from_rotvec(u_main * yaw).as_matrix() @ align_vectors(u_sub, u_main)


def yaw_of(Rs: np.ndarray, u_sub: np.ndarray, u_main: np.ndarray) -> float:
    """Inverse of rs_of for a vertical-constrained Rs."""
    u_main = u_main / np.linalg.norm(u_main)
    B = Rs @ align_vectors(u_sub, u_main).T
    e1 = np.cross(u_main, [1.0, 0, 0] if abs(u_main[0]) < 0.9 else [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    b = B @ e1 - np.dot(B @ e1, u_main) * u_main
    return float(np.arctan2(np.dot(np.cross(e1, b), u_main), np.dot(e1, b)))


def interp_weights(t: np.ndarray, knots: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Linear interpolation of a piecewise-linear path: (index of the left knot, weight of the left, weight of the right)."""
    j = np.clip(np.searchsorted(knots, t, side="right") - 1, 0, len(knots) - 2)
    a = np.clip((t - knots[j]) / (knots[j + 1] - knots[j]), 0.0, 1.0)
    return j, 1 - a, a


def path_at(t: np.ndarray, knots: np.ndarray, C: np.ndarray) -> np.ndarray:
    j, wl, wr = interp_weights(t, knots)
    return C[j] * wl[:, None] + C[j + 1] * wr[:, None]


def _errors(G: dict, Rs: np.ndarray, C_obs: np.ndarray, sel: np.ndarray) -> np.ndarray:
    """Pixel errors of observations `sel` with a centre per observation (rotation-only attach)."""
    fidx = G["fidx"][sel]
    R_m = G["R"] @ Rs.T                                   # per frame
    Xc = np.einsum("nij,nj->ni", R_m[fidx], G["X"][sel] - C_obs[sel])
    z = Xc[:, 2]
    vis = z > 0.05
    zs = np.where(vis, z, 1.0)
    xn, yn = Xc[:, 0] / zs, Xc[:, 1] / zs
    d = 1 + G["k1"][fidx] * (xn * xn + yn * yn)
    K = G["K"][fidx]
    r = np.stack([K[:, 0, 0] * xn * d + K[:, 0, 2], K[:, 1, 1] * yn * d + K[:, 1, 2]], 1) - G["uv"][sel]
    r[~vis] = 200.0
    return np.linalg.norm(r, axis=1)


def fit_path(G: dict, u_sub, u_main, yaw0: float, C0: np.ndarray, knots: np.ndarray, sel: np.ndarray, lam: float,
             free_yaw: bool = True, huber_px: float = 6.0, max_nfev: int = 60, const: bool = False) -> dict:
    """Yaw (optional) + centre knots from the observations `sel`; const: ONE centre for all knots. lam: weight (px per
    SfM unit of second difference of the knots) of the smoothness prior; knots without observations are not pulled."""
    K = len(knots)
    tobs = G["t"][G["fidx"][sel]]
    j, wl, wr = interp_weights(tobs, knots)
    used = np.zeros(K, bool)
    used[j], used[j + 1] = True, True
    fidx = G["fidx"][sel]
    Xs, uvs, Kp, k1p = G["X"][sel], G["uv"][sel], G["K"][fidx], G["k1"][fidx]
    Rsub = G["R"][fidx]
    dtk = float(np.median(np.diff(knots)))

    def unpack(x):
        yaw = x[0] if free_yaw else yaw0
        C = np.tile(x[1:4], (K, 1)) if const else x[1:1 + 3 * K].reshape(K, 3)
        return yaw, C

    def fun(x):
        yaw, C = unpack(x)
        Rs = rs_of(yaw, u_sub, u_main)
        Cn = C[j] * wl[:, None] + C[j + 1] * wr[:, None]
        R_m = Rsub @ Rs.T
        Xc = np.einsum("nij,nj->ni", R_m, Xs - Cn)
        z = Xc[:, 2]
        vis = z > 0.05
        zs = np.where(vis, z, 1.0)
        xn, yn = Xc[:, 0] / zs, Xc[:, 1] / zs
        d = 1 + k1p * (xn * xn + yn * yn)
        r = np.stack([Kp[:, 0, 0] * xn * d + Kp[:, 0, 2], Kp[:, 1, 1] * yn * d + Kp[:, 1, 2]], 1) - uvs
        r[~vis] = 200.0
        parts = [r.ravel()]
        if lam > 0 and K >= 3:
            d2 = C[2:] - 2 * C[1:-1] + C[:-2]
            parts.append(lam * (d2 * (used[2:] & used[1:-1] & used[:-2])[:, None]).ravel())
        return np.concatenate(parts)
    x0 = np.r_[yaw0, C0] if const else np.r_[yaw0, np.tile(C0, (K, 1)).ravel()]
    r = least_squares(fun, x0, loss="huber", f_scale=huber_px, max_nfev=max_nfev,
                      x_scale=np.r_[0.02, np.full(3 if const else 3 * K, 0.1)])
    yaw, C = unpack(r.x)
    return {"yaw": float(yaw), "C": C, "knots": knots, "cost": float(r.cost)}


def pick_setting(paths: list[dict], tie: float = 0.03) -> dict:
    """Among the (knot_s, lam) settings whose hold-out median is within `tie` of the best, the smoothest one."""
    h_best = min(r["holdout_median_px"] for r in paths)
    near = [r for r in paths if r["holdout_median_px"] <= (1 + tie) * h_best]
    return sorted(near, key=lambda r: (-r["lam"], -r["knot_s"]))[0]


def solve(takes: list[dict], knot_s=8.0, lams: tuple = (0.0,), gate_gain: float = 0.10,
          max_obs: int = 60000, tie: float = 0.03, log=print, init: dict | None = None, holdout_eval: bool = True) -> dict:
    """Global yaw + constant centre, then the piecewise centre path; hold-out decides which one is kept.
    knot_s: knot spacing (s) or a tuple of candidates (the best hold-out wins; larissa_kadu_full cam05/06: 8 s vs 4 s)."""
    G = pool(takes)
    n = len(G["fidx"])
    rng = np.random.default_rng(0)
    use = np.ones(n, bool)
    if n > max_obs:
        use[:] = False
        use[rng.choice(n, max_obs, replace=False)] = True
    tobs = G["t"][G["fidx"]]
    blk = np.floor(tobs / 0.5).astype(int) % 2
    u_sub, u_main = G["u_sub"], G["u_main"]
    t_lo, t_hi = float(G["t"].min()), float(G["t"].max())
    knot_list = tuple(knot_s) if isinstance(knot_s, (tuple, list)) else (knot_s,)

    def knots_of(ks):
        return np.linspace(t_lo, t_hi, max(3, int(round((t_hi - t_lo) / ks)) + 1))
    if init is not None:        # a previous solution's constant yaw / centre (saves the multi-start solve)
        yaw0, C0 = float(init["constant"]["yaw"]), np.asarray(init["constant"]["C"], float)
        sol = {"median_px": float(init["constant"].get("fit_px_median", float("nan")))}
    else:
        sol = solve_rotonly(G["R"], G["K"], G["k1"], G["fidx"][use], G["X"][use], G["uv"][use], np.ones(int(use.sum())),
                            vertical=(u_sub, u_main))
        # solve_rotonly's fidx are indices into G["R"]; it was handed the subsampled observations above
        yaw0 = yaw_of(sol["Rs"], u_sub, u_main)
        C0 = np.asarray(sol["C"], float)
    log(f"[attach_global] constant centre over {len(G['t'])} frames / {int(use.sum())} observations: fit {sol['median_px']:.1f} px, "
        f"yaw {np.degrees(yaw0):.2f} deg")

    def holdout(path_fit):
        e = np.zeros(n)
        for k in (0, 1):
            tr, te = use & (blk == k), use & (blk != k)
            pf = path_fit(tr)
            Rs = rs_of(pf["yaw"], u_sub, u_main)
            Cn = path_at(tobs[te], pf["knots"], pf["C"])
            Cfull = np.zeros((n, 3))
            Cfull[te] = Cn
            e[te] = _errors(G, Rs, Cfull, te)
        return e[use]

    out = {"constant": {"yaw": yaw0, "C": C0, "fit_px_median": float(sol["median_px"])}}
    kc = np.array([t_lo, t_hi + 1.0])
    if holdout_eval:
        e_const = holdout(lambda tr: fit_path(G, u_sub, u_main, yaw0, C0, kc, tr, 0.0, const=True))
        out["constant"].update(holdout_median_px=float(np.median(e_const)), holdout_p90_px=float(np.percentile(e_const, 90)))
        log(f"[attach_global] constant centre hold-out: median {np.median(e_const):.1f} px, p90 {np.percentile(e_const, 90):.1f}")
    else:      # the setting is given (init + a single knot spacing / lam): no selection, the path is used
        out["constant"].update(holdout_median_px=float("inf"), holdout_p90_px=float("inf"))
    for ks in knot_list:
        knots = knots_of(ks)
        for lam in lams:
            if holdout_eval:
                e = holdout(lambda tr, lam=lam, knots=knots: fit_path(G, u_sub, u_main, yaw0, C0, knots, tr, lam))
                rec = {"lam": lam, "knot_s": ks, "holdout_median_px": float(np.median(e)), "holdout_p90_px": float(np.percentile(e, 90))}
            else:
                rec = {"lam": lam, "knot_s": ks, "holdout_median_px": 0.0, "holdout_p90_px": 0.0, "note": "given, not evaluated"}
            out.setdefault("paths", []).append(rec)
            log(f"[attach_global] piecewise centre (knots {ks:g} s, smoothness {lam:g}): " + (
                f"hold-out median {rec['holdout_median_px']:.1f} px, p90 {rec['holdout_p90_px']:.1f}" if holdout_eval else "given"))
    # The alternating 0.5 s hold-out cannot tell a centre that is right from one that absorbs errors which last seconds (a
    # kicked leg triangulated 10 cm off, a sync error): larissa_kadu_full cam05 (visibly static) got a 1.5 m excursion at 21-25 s
    # and cam06 a 2.5 m one in its first knot from a free path. So among the settings within `tie` of the best hold-out the
    # SMOOTHEST (largest smoothness weight, then the coarsest knots) wins.
    best = pick_setting(out["paths"], tie)
    use_path = best["holdout_median_px"] < (1 - gate_gain) * out["constant"]["holdout_median_px"]
    if use_path:
        fit = fit_path(G, u_sub, u_main, yaw0, C0, knots_of(best["knot_s"]), use, best["lam"])
        kind = "piecewise"
    else:
        fit = fit_path(G, u_sub, u_main, yaw0, C0, kc, use, 0.0, const=True)
        kind = "constant"
    Rs = rs_of(fit["yaw"], u_sub, u_main)
    Cn = path_at(tobs, fit["knots"], fit["C"])
    e_all = _errors(G, Rs, Cn, np.ones(n, bool))
    per_seg = {}
    for s in sorted(set(G["seg"])):
        m = G["seg"][G["fidx"]] == s
        per_seg[s] = {"fit_px_median": round(float(np.median(e_all[m])), 2), "fit_px_p90": round(float(np.percentile(e_all[m], 90)), 2),
                      "obs": int(m.sum())}
    out.update({"kind": kind, "chosen": best if use_path else out["constant"], "yaw_deg": float(np.degrees(fit["yaw"])),
                "Rs": Rs.tolist(), "knots": fit["knots"].tolist(), "C": np.asarray(fit["C"]).tolist(), "per_segment": per_seg,
                "u_sub": u_sub.tolist(), "u_main": u_main.tolist(), "obs": int(n)})
    log(f"[attach_global] kept: {kind}; per-take fit {per_seg}")
    return out


# ------------------------------------------------------------------------------------------------ apply

def rotate_poses(poses: list, Rs: np.ndarray, C: np.ndarray) -> None:
    """In place: sub-model poses (R_sub, t_sub, correspondences in the sub-model frame) -> main-frame rotation-only poses
    R = R_sub Rs^T, centre C[i] (t = -R C[i]); the correspondences become the same bearings seen from C[i] at 10 units,
    so projecting them through the new pose reproduces the frame's image points."""
    for i, p in enumerate(poses):
        C_sub = -p.R.T @ p.t
        R_m = p.R @ Rs.T
        if p.corr is not None:
            v = p.corr.xyz - C_sub
            p.corr.xyz = C[i] + 10.0 * (v / np.linalg.norm(v, axis=1, keepdims=True)) @ Rs.T
        p.R, p.t = R_m, -R_m @ C[i]
        if p.ok:
            p.source = "attach"


def apply(d: dict, cam: str, sol: dict, backup: str = "s2_gate_backup", log=print) -> dict:
    """Write the global solution into one take (d = load_take(...)): cameras/cache/loc_<cam>.npz (rotation R_sub Rs^T,
    centre C(t), correspondences = the sub-model's points as bearings from C(t) at 10 units, like stage_attach's
    rotation-only branch), cache/attach_<cam>.npz (per-frame dancer residual) and, through stage_smooth, cameras/<cam>.npz.
    The previous files are copied to cameras/<backup>/ once. Returns the take's summary."""
    import shutil

    from . import pipeline as P
    from .features import list_images
    work = d["work"]
    P.LOOSE_FILE_KEYS = True
    bdir = work / "cameras" / backup
    (bdir / "cache").mkdir(parents=True, exist_ok=True)
    for rel in (f"{cam}.npz", f"cache/loc_{cam}.npz", f"cache/attach_{cam}.npz"):
        src = work / "cameras" / rel
        if src.exists() and not (bdir / rel).exists():
            shutil.copy2(src, bdir / rel)
    poses = pickle.loads((work / "cameras" / "cache" / f"locsub_{cam}.pkl").read_bytes())["value"]
    if d.get("zoom") is not None:
        import pycolmap

        from .attach_zoom import reference_centre, resolve_zoom
        sub = pycolmap.Reconstruction(str(work / "cameras" / "sfm_work" / f"sub_{cam}" / "best"))
        poses, _ = resolve_zoom(poses, reference_centre(sub), d["w"], d["h"], t=d["t"])
    frames = [r for r in list_images(work, [cam])[cam] if r.kind == "win"]
    t = np.array([r.t for r in frames])
    assert len(t) == len(poses) == len(d["t"]), (len(t), len(poses), len(d["t"]))
    Rs = np.asarray(sol["Rs"], float)
    C = path_at(t, np.asarray(sol["knots"], float), np.asarray(sol["C"], float))
    rotate_poses(poses, Rs, C)
    P._save_raw(work / "cameras" / "cache" / f"loc_{cam}.npz", frames, poses)
    # per-frame dancer residual of the result (frames that have observations), as stage_attach's attach_<cam>.npz
    G = {"R": np.array([p.R for p in poses]), "K": d["K"], "k1": d["k1"], "fidx": d["fidx"], "X": d["X"], "uv": d["uv"]}
    R_main = G["R"]                                       # already the main-frame rotations: use Rs = I
    e = _errors({**G, "R": R_main}, np.eye(3), C[d["fidx"]], np.ones(len(d["fidx"]), bool))
    per_frame = np.full(len(t), np.nan)
    for i in np.unique(d["fidx"]):
        per_frame[i] = np.median(e[d["fidx"] == i])
    np.savez(work / "cameras" / "cache" / f"attach_{cam}.npz", dancer_px=per_frame, method=f"rotation-only global ({sol['kind']})")
    from .. import camera_types as ct
    types = {c: ty for c, ty in ct.types_for_work(work).items() if not ty.general}
    P.stage_smooth(work, [cam], 0.04, 4.0, types=types)
    rp = work / "cameras" / "report.json"
    if rp.exists():
        rep = json.loads(rp.read_text())
        rep.setdefault("attach_global", {})[cam] = {
            "kind": sol["kind"], "yaw_deg": sol["yaw_deg"], "obs_take": int(len(d["fidx"])),
            "dancer_px_median": round(float(np.median(e)), 2), "dancer_px_p90": round(float(np.percentile(e, 90)), 2),
            "centre_path_range_m": None, "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "note": f"one Rs + {sol['kind']} centre for the whole dance (cameras/attach_global.py); previous exports in cameras/{backup}/"}
        rp.write_text(json.dumps(rep, indent=1, default=P._np))
    log(f"[attach_global] {work.name} cam {cam}: dancers {np.median(e):.1f} px median, {np.percentile(e, 90):.1f} p90 ({sol['kind']})")
    return {"median_px": float(np.median(e)), "p90_px": float(np.percentile(e, 90))}


def _np(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    return str(o)


def load_take_cached(seg: str, cam: str, zoom: bool, refresh: bool = False, log=print) -> dict:
    """load_take, pickled in cameras_attach/cache/ (the dancer triangulation of a take takes minutes)."""
    p = SOL_DIR / "cache" / f"{cam}_{seg}.pkl"
    if p.exists() and not refresh:
        return pickle.loads(p.read_bytes())
    d = load_take(seg, cam, zoom)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(pickle.dumps(d))
    log(f"[attach_global] cached {p.name}: {len(d['fidx'])} observations")
    return d


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m dancecap.cameras.attach_global", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["solve", "apply", "run"], help="run = solve then apply")
    ap.add_argument("--takes", default="a,38,b,c,d", help="takes the solve pools (38 = larissa_kadu_38)")
    ap.add_argument("--apply-takes", default="a,b,c,d", help="takes whose cameras are re-exported (never 38)")
    ap.add_argument("--cams", default="05,06")
    ap.add_argument("--zoom-cams", default="05")
    ap.add_argument("--knot-s", default="8", help="knot spacing in s; comma-separated candidates (hold-out picks)")
    ap.add_argument("--lams", default="0", help="smoothness weights (px per SfM unit of centre second difference), comma-separated")
    ap.add_argument("--init-from", help="a previous solution json: its constant yaw / centre start the fit")
    ap.add_argument("--given", action="store_true", help="with --init-from and ONE knot spacing and lam: use them, no hold-out selection")
    ap.add_argument("--refresh", action="store_true", help="recompute the cached observations of the takes")
    a = ap.parse_args(argv)
    import os

    from .video_zoom import lower_priority
    if not os.environ.get("DANCECAP_NORMAL_PRIORITY"):   # below-normal threads get ~1 % of a saturated 4-CPU workstation
        lower_priority()
    SOL_DIR.mkdir(parents=True, exist_ok=True)
    zc = a.zoom_cams.split(",")
    for cam in a.cams.split(","):
        t0 = time.time()
        sol_p = SOL_DIR / f"{cam}.json"
        loaded = {}
        if a.what in ("solve", "run"):
            takes = []
            for s in a.takes.split(","):
                loaded[s] = load_take_cached(s, cam, cam in zc, a.refresh)
                takes.append(loaded[s])
            print(f"[attach_global] cam {cam}: observations {[len(x['fidx']) for x in takes]} ({time.time() - t0:.0f}s)", flush=True)
            sol = solve(takes, knot_s=tuple(float(x) for x in a.knot_s.split(",")), lams=tuple(float(x) for x in a.lams.split(",")),
                         init=json.loads(Path(a.init_from).read_text()) if a.init_from else None, holdout_eval=not a.given)
            sol_p.write_text(json.dumps(sol, indent=1, default=_np))
        else:
            sol = json.loads(sol_p.read_text())
        if a.what in ("apply", "run"):
            for s in a.apply_takes.split(","):
                if s == "38":
                    raise SystemExit("larissa_kadu_38's cameras are not re-exported (its fit is finished and reviewed)")
                apply(loaded.get(s) or load_take_cached(s, cam, cam in zc, a.refresh), cam, sol)


if __name__ == "__main__":
    main()
