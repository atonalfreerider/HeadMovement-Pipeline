"""Cameras stage orchestration (see package docstring). Every step caches under cameras/cache and cameras/sfm_work."""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np

from .. import gpu
from ..take import Take
from . import report as qa
from .features import Features, FeatureSettings, ImageRec, list_images
from .localize import FramePose, Localizer, LocSettings, localize_camera
from .mapping import (MapSettings, build_database, link_components, map_pairs, match_pairs, model_cams,
                      refine_model, run_mapper, run_sub_mapper, select_map_images, targeted_pairs)
from .smooth import smooth_track
from .world import apply, bootstrap, statures_from_take, triangulate_dancers

STAGES = ("map", "loc", "attach", "world", "qa")


def _K(f: np.ndarray, w: int, h: int) -> np.ndarray:
    """OpenCV intrinsics per frame (pixel-centre convention: COLMAP's w/2 principal point - 0.5)."""
    K = np.zeros((len(f), 3, 3))
    K[:, 0, 0] = K[:, 1, 1] = f
    K[:, 0, 2], K[:, 1, 2], K[:, 2, 2] = w / 2 - 0.5, h / 2 - 0.5, 1.0
    return K


def stage_map(work: Path, recs: dict, F: Features, ms: MapSettings) -> dict:
    out = work / "cameras"
    images, dense = select_map_images(recs, ms)
    pairs = map_pairs(images, dense, ms)
    print(f"[cameras] map: {len(images)} images ({len(dense)} dense), {len(pairs)} pairs", flush=True)
    t0 = time.time()
    with gpu.exclusive("cameras-map"):
        for r in images:
            F.get(r)
        matches = match_pairs(F, images, pairs, out / "cache" / "matches_map.npz")
        extra, weak = targeted_pairs(images, matches, ms)
        if extra:
            print(f"[cameras] weak cameras {weak}: +{len(extra)} targeted pairs", flush=True)
            matches.update(match_pairs(F, images, extra, out / "cache" / "matches_map.npz"))
    t_match = time.time() - t0
    sw = out / "sfm_work"
    st = build_database(work, F, images, matches, sw / "database_all.db", ms)
    groups = link_components(images, st["pair_inliers"], ms)
    print(f"[cameras] verified {st['verified']}/{st['pairs']} pairs; strong-link groups {[sorted(g) for g in groups]}",
          flush=True)
    for cp, e in sorted(st["per_campair"].items()):
        print(f"[cameras]   {cp}: {e['pairs']} pairs, max {e['max_inliers']} inliers", flush=True)
    st.pop("pair_inliers")
    main_cams = groups[0]
    db = sw / "database.db"
    if len(groups) > 1:
        build_database(work, F, images, matches, db, ms, cams=main_cams)
    else:
        (sw / "database_all.db").replace(db)
    t0 = time.time()
    best, models = run_mapper(work, db, sw / "models", ms)
    if best is None:
        raise RuntimeError("incremental mapping produced no model")
    raw_models = [{"images": r.num_reg_images(), "points": r.num_points3D(),
                   "reproj_px": r.compute_mean_reprojection_error(), "cams": model_cams(r)}
                  for r in sorted(models.values(), key=lambda r: -r.num_reg_images())]
    refine = refine_model(best, ms.refine_px)
    sfm = out / "sfm"
    sfm.mkdir(parents=True, exist_ok=True)
    best.write_binary(str(sfm))
    subs = build_sub_models(work, F, images, matches, groups[1:], ms)
    info = {"images": len(images), "dense": len(dense), "pairs": len(matches), "match_s": round(t_match, 1),
            "mapping_s": round(time.time() - t0, 1), "verification": st,
            "strong_link_groups": [sorted(g) for g in groups], "main_cams": sorted(main_cams),
            "models": raw_models, "refine": refine, "sub_models": subs,
            "map_images": [r.key for r in images]}
    (sw / "map.json").write_text(json.dumps(info, indent=1, default=str))
    for m in raw_models:
        print(f"[cameras] model: {m['images']} images, {m['points']} pts, {m['reproj_px']:.3f} px, {m['cams']}")
    print(f"[cameras] refined main model: {refine}")
    return info


def build_sub_models(work: Path, F: Features, images: list[ImageRec], matches: dict, groups: list[set],
                     ms: MapSettings) -> dict:
    """One COLMAP sub-model per weakly linked camera group -> sfm_work/sub_<name>/best (used by attach)."""
    sw = work / "cameras" / "sfm_work"
    subs: dict = {}
    for g in groups:
        name = "_".join(sorted(g))
        sdb = sw / f"database_sub_{name}.db"
        build_database(work, F, images, matches, sdb, ms, cams=g, single_camera=len(g) == 1)
        if (sw / f"sub_{name}").exists():
            shutil.rmtree(sw / f"sub_{name}")
        sb = run_sub_mapper(work, sdb, sw / f"sub_{name}", ms)
        if sb is None:
            subs[name] = None
            print(f"[cameras] sub-model {name}: mapping failed", flush=True)
            continue
        subs[name] = {"images": sb.num_reg_images(), "points": sb.num_points3D(), "cams": model_cams(sb),
                      "refine": refine_model(sb, ms.refine_px)}
        (sw / f"sub_{name}" / "best").mkdir(parents=True, exist_ok=True)
        sb.write_binary(str(sw / f"sub_{name}" / "best"))
        print(f"[cameras] sub-model {name}: {subs[name]}", flush=True)
    return subs


def load_model(work: Path):
    import pycolmap
    return pycolmap.Reconstruction(str(work / "cameras" / "sfm"))


def stage_loc(work: Path, recs: dict, F: Features, ls: LocSettings, ms: MapSettings, cams: list[str]) -> dict:
    model = load_model(work)
    images, _ = select_map_images(recs, ms)
    L = Localizer(F, model, images, ls)
    out = {}
    with gpu.exclusive("cameras-loc"):
        for cam in cams:
            if cam not in L.cams_reg:
                print(f"[cameras] {cam}: not in the main model (see attach)")
                (work / "cameras" / "cache" / f"loc_{cam}.npz").unlink(missing_ok=True)
                continue
            frames = [r for r in recs[cam] if r.kind == "win"]
            meta = json.loads((work / "frames" / cam / "frames.json").read_text())
            w, h = int(meta["width"]), int(meta["height"])
            t0 = time.time()
            poses = localize_camera(L, frames, w, h)
            _save_raw(work / "cameras" / "cache" / f"loc_{cam}.npz", frames, poses)
            n_ok = sum(p.ok for p in poses)
            print(f"[cameras] {cam}: localised {n_ok}/{len(frames)} ({time.time() - t0:.0f}s)", flush=True)
            out[cam] = n_ok
            L.match_cache.clear()
    return out


def stage_attach(work: Path, recs: dict, F: Features, ls: LocSettings, ms: MapSettings, cams: list[str],
                 sigma_s: float, max_reproj_px: float, statures: tuple[dict, str]) -> dict:
    """Cameras missing from the main model: localise against their own sub-model, then place that sub-model in
    the main frame with the dancers' keypoints (attach.py). Writes loc_<cam>.npz in the MAIN SfM frame."""
    import pycolmap

    import pickle

    from .attach import compose, floor_in_model, observations, solve_rotonly, solve_sim3
    from .world import dancer_keypoints, triangulate_dancers
    from scipy.spatial.transform import Rotation
    main = load_model(work)
    reg = model_cams(main)
    missing = [c for c in cams if c not in reg]
    info: dict = {}
    if not missing:
        return info
    images, _ = select_map_images(recs, ms)
    subs = []
    for d in sorted((work / "cameras" / "sfm_work").glob("sub_*/best")):
        if (d / "images.bin").exists():
            m = pycolmap.Reconstruction(str(d))
            subs.append((d.parent.name, m, model_cams(m)))
    main_tracks = stage_smooth(work, [c for c in cams if c in reg], sigma_s, max_reproj_px, write=False)
    dancers3d = grid = floor_main = None
    for cam in missing:
        cand = [(n, m, mc) for n, m, mc in subs if mc.get(cam, 0) >= 3]
        if not cand:
            info[cam] = {"attached": False, "reason": "no sub-model contains this camera"}
            print(f"[cameras] {cam}: no sub-model -> unregistered")
            continue
        name, sub, mc = max(cand, key=lambda x: x[2][cam])
        frames = [r for r in recs[cam] if r.kind == "win"]
        meta = json.loads((work / "frames" / cam / "frames.json").read_text())
        w, h = int(meta["width"]), int(meta["height"])
        cache = work / "cameras" / "cache" / f"locsub_{cam}.pkl"
        sub_bin = work / "cameras" / "sfm_work" / name / "best" / "images.bin"
        if cache.exists() and cache.stat().st_mtime > sub_bin.stat().st_mtime:
            poses = pickle.loads(cache.read_bytes())
        else:
            single = len({im.camera_id for im in sub.images.values()}) == 1
            L = Localizer(F, sub, images, LocSettings(**{**ls.__dict__, "fix_focal": single}))
            with gpu.exclusive(f"cameras-attach-{cam}"):
                poses = localize_camera(L, frames, w, h)
            cache.write_bytes(pickle.dumps(poses))
        ok = np.array([p.ok for p in poses])
        print(f"[cameras] {cam}: {ok.sum()}/{len(frames)} frames localised in sub-model {name} {mc}", flush=True)
        if ok.sum() < 10:
            info[cam] = {"attached": False, "reason": f"only {int(ok.sum())} frames localised in sub-model {name}"}
            continue
        if dancers3d is None:
            dancers3d, grid = triangulate_dancers(work, main_tracks, kps=tuple(range(17)), return_grid=True)
            pts_main = np.array([p.xyz for p in main.points3D.values()]).reshape(-1, 3)
            pre = bootstrap(main_tracks, triangulate_dancers(work, main_tracks), pts_main, *statures)
            Rw = np.asarray(pre["R"])
            floor_main = (Rw[1], -float(pre["t"][1]) / pre["s"])  # world y = 0  <=>  R[1].X = -t_y / s
        sub_R, sub_C = [], []
        for im in sub.images.values():
            if im.has_pose:
                T_ = im.cam_from_world()
                sub_R.append(T_.rotation.matrix())
                sub_C.append(-sub_R[-1].T @ T_.translation)
        up_s = -np.mean([r[1] for r in sub_R], 0)
        up_s /= np.linalg.norm(up_s)
        pts_sub = np.array([p.xyz for p in sub.points3D.values()]).reshape(-1, 3)
        fl = floor_in_model(pts_sub, np.array(sub_C), up_s)
        floor_sub = (fl[0], fl[1]) if fl is not None else None
        t = np.array([r.t for r in frames])
        R_sub = np.array([p.R for p in poses])
        t_sub = np.array([p.t for p in poses])
        f = np.array([p.f if p.ok else np.nan for p in poses])
        f = np.where(ok, f, np.interp(t, t[ok], f[ok]))
        K = _K(f, w, h)
        k1 = np.array([p.k1 for p in poses])
        kp2d = dancer_keypoints(work, cam, len(frames))
        fidx, X, uv, wt = observations(kp2d, t, ok, dancers3d, grid)
        if len(fidx) < 50:
            info[cam] = {"attached": False, "reason": f"only {len(fidx)} dancer keypoint observations"}
            continue
        sols = {"rotation-only": solve_rotonly(R_sub, K, k1, fidx, X, uv, wt, up_main=floor_main[0])}
        floor_ok = fl is not None and np.degrees(np.arccos(min(1.0, abs(float(fl[0] @ up_s))))) < 10
        if floor_ok:
            sols["sim3+floor"] = solve_sim3(R_sub, t_sub, K, k1, fidx, X, uv, wt, floor_sub=floor_sub,
                                            floor_main=floor_main)
        kind = min(sols, key=lambda k: sols[k]["median_px"])
        sol = sols[kind]
        C_sub = -np.einsum("fji,fj->fi", R_sub, t_sub)
        if kind == "sim3+floor":
            R_m, t_m = compose(R_sub, t_sub, Rotation.from_matrix(sol["Rs"]).as_rotvec(), np.log(sol["s"]),
                               sol["T"])
        else:
            R_m = R_sub @ sol["Rs"].T
            t_m = -np.einsum("fij,j->fi", R_m, sol["C"])
        for i, p in enumerate(poses):
            p.R, p.t = R_m[i], t_m[i]
            if p.corr is not None:
                if kind == "sim3+floor":
                    p.corr.xyz = sol["s"] * p.corr.xyz @ sol["Rs"].T + sol["T"]
                else:  # same bearing from the fixed centre (points at 10 units): projection is unchanged
                    v = p.corr.xyz - C_sub[i]
                    p.corr.xyz = sol["C"] + 10.0 * (v / np.linalg.norm(v, axis=1, keepdims=True)) @ sol["Rs"].T
            if p.ok:
                p.source = "attach"
        _save_raw(work / "cameras" / "cache" / f"loc_{cam}.npz", frames, poses)
        info[cam] = {"attached": True, "method": kind, "sub_model": name, "sub_model_cams": mc,
                     "sub_model_reproj_px": float(sub.compute_mean_reprojection_error()),
                     "sub_floor": {"found": fl is not None, "inliers": fl[2] if fl else 0, "usable": floor_ok},
                     "candidates": {k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()
                                        if kk not in ("Rs", "T", "C")} for k, v in sols.items()},
                     "dancer_reproj_median_px": sol["median_px"], "dancer_reproj_p90_px": sol["p90_px"],
                     "dancer_obs": sol["n_obs"], "frames_used": sol["frames_used"],
                     "camera_centre_sfm": (sol["C"].tolist() if kind == "rotation-only" else None)}
        print(f"[cameras] {cam}: attached ({kind}, floor in sub-model: {'usable' if floor_ok else 'no'}): "
              + ", ".join(f"{k} median {v['median_px']:.1f}px p90 {v['p90_px']:.1f}" for k, v in sols.items()),
              flush=True)
    return info


def _save_raw(p: Path, frames: list[ImageRec], poses: list[FramePose]) -> None:
    n = len(poses)
    corr_off = np.zeros(n + 1, np.int64)
    xy, xyz = [], []
    for i, fp in enumerate(poses):
        k = 0 if fp.corr is None or not fp.ok else len(fp.corr.xyz)
        if k:
            xy.append(fp.q_xy)
            xyz.append(fp.corr.xyz)
        corr_off[i + 1] = corr_off[i] + k
    np.savez(p, t=np.array([r.t for r in frames]), ok=np.array([fp.ok for fp in poses]),
             R=np.array([fp.R for fp in poses]), tvec=np.array([fp.t for fp in poses]),
             f=np.array([fp.f for fp in poses]), k1=np.array([fp.k1 for fp in poses]),
             n_inliers=np.array([fp.n_inliers for fp in poses]), reproj_px=np.array([fp.reproj_px for fp in poses]),
             source=np.array([fp.source for fp in poses]), corr_off=corr_off,
             corr_xy=np.concatenate(xy) if xy else np.zeros((0, 2)),
             corr_xyz=np.concatenate(xyz) if xyz else np.zeros((0, 3)))


def reproj_cv(R, t, K, k1, xyz, xy) -> np.ndarray:
    uv, vis = qa.project_cv(xyz, K, R, t, k1)
    return np.where(vis, np.linalg.norm(uv - xy, axis=1), np.inf)


def stage_smooth(work: Path, cams: list[str], sigma_s: float, max_reproj_px: float,
                 write: bool = True) -> dict[str, dict]:
    """Raw localisation -> smoothed tracks -> cameras/<cam>.npz (SfM frame). Returns track dicts."""
    tracks = {}
    for cam in cams:
        p = work / "cameras" / "cache" / f"loc_{cam}.npz"
        if not p.exists():
            if write:  # no track for this camera any more: never leave a stale contract file behind
                (work / "cameras" / f"{cam}.npz").unlink(missing_ok=True)
            continue
        d = dict(np.load(p))
        meta = json.loads((work / "frames" / cam / "frames.json").read_text())
        w, h = int(meta["width"]), int(meta["height"])
        ok_raw = d["ok"].astype(bool)
        wts = np.where(ok_raw, np.clip(d["n_inliers"] / 100.0, 0.2, 1.0), 0.0)
        sm = smooth_track(d["t"], d["R"], d["tvec"], np.where(ok_raw, d["f"], np.nan), ok_raw, wts, sigma_s=sigma_s)
        k1 = np.interp(d["t"], d["t"][ok_raw], d["k1"][ok_raw]) if ok_raw.any() else np.zeros(len(d["t"]))
        K = _K(sm["f"], w, h)
        K_raw = _K(np.where(ok_raw, d["f"], sm["f"]), w, h)
        R_out, t_out, f_out = sm["R"].copy(), sm["tvec"].copy(), sm["f"].copy()
        rp = np.full(len(d["t"]), np.nan)
        raw_used = np.zeros(len(d["t"]), bool)
        for i in range(len(d["t"])):
            a, b = d["corr_off"][i], d["corr_off"][i + 1]
            if b <= a:
                continue
            X, x = d["corr_xyz"][a:b], d["corr_xy"][a:b]
            e_s = float(np.median(reproj_cv(sm["R"][i], sm["tvec"][i], K[i], k1[i], X, x)))
            e_r = float(np.median(reproj_cv(d["R"][i], d["tvec"][i], K_raw[i], k1[i], X, x)))
            # the smoother lags behind fast zooms / sudden moves: keep the raw pose where it is clearly better
            if ok_raw[i] and e_s > max(1.5 * e_r, e_r + 0.5):
                R_out[i], t_out[i], f_out[i], K[i] = d["R"][i], d["tvec"][i], K_raw[i, 0, 0], K_raw[i]
                raw_used[i] = True
                e_s = e_r
            rp[i] = e_s
        ok = ok_raw & (np.nan_to_num(rp, nan=1e9) <= max_reproj_px)
        n_in = np.where(ok, d["n_inliers"], 0).astype(np.int32)
        if write:
            np.savez(work / "cameras" / f"{cam}.npz", t=d["t"], K=K, R=R_out, tvec=t_out, dist=k1, ok=ok,
                     reproj_px=rp, n_inliers=n_in)
        C_out = -np.einsum("fji,fj->fi", R_out, t_out)
        tracks[cam] = {"t": d["t"], "K": K, "k1": k1, "R": R_out, "tvec": t_out, "ok": ok, "C": C_out,
                       "f": f_out, "f_raw": d["f"], "raw_used": int(raw_used.sum()), "ok_raw": ok_raw, "reproj_px": rp, "reproj_raw": d["reproj_px"],
                       "n_inliers": d["n_inliers"], "source": d["source"], "w": w, "h": h, "maxdim": max(w, h),
                       "outliers": int(sm["outlier"].sum())}
    return tracks


def focal_stats(tr: dict, spike_pct: float = 2.0) -> dict:
    """Plausibility of the exported focal track ("smooth, jumps only at zoom").

    A zoom is a monotonic run (>= 3 frames) of > 3 % focal change; implausible are SPIKES outside zooms (a >spike_pct %
    frame-to-frame change immediately undone by the next frame), a large raw-vs-exported scatter, or a focal
    outside 0.4-4 x max dim, or an OSCILLATING focal (total |dlog f| > 3 x its range: the focal is not observable
    and wobbles, not a zoom). Reversals inside a zoom (pinch-zoom jitter) are reported, not failed."""
    f = tr["f"]
    dl = np.diff(np.log(f))
    rev = np.nonzero((np.abs(dl[:-1]) > spike_pct / 100) & (np.abs(dl[1:]) > spike_pct / 100)
                     & (np.sign(dl[:-1]) != np.sign(dl[1:])))[0] + 1  # frame index of the reversal
    raw = tr["f_raw"][tr["ok_raw"]]
    resid = np.abs(np.log(raw) - np.log(f[tr["ok_raw"]])) if len(raw) else np.zeros(1)
    zooms, i = [], 0  # monotonic runs with > 3 % total change
    while i < len(dl):
        j = i
        sgn = np.sign(dl[i])
        while j + 1 < len(dl) and np.sign(dl[j + 1]) == sgn:
            j += 1
        tot = float(np.sum(dl[i:j + 1]))
        if abs(tot) > 0.03 and j - i >= 2 and sgn != 0:  # >= 3 frames: a one-frame jump is not a zoom
            zooms.append({"t0": round(float(tr["t"][i]), 3), "t1": round(float(tr["t"][j + 1]), 3),
                          "change_pct": round(100 * (np.exp(tot) - 1), 1)})
        i = j + 1
    zt = [(z["t0"] - 0.17, z["t1"] + 0.17) for z in zooms]  # +-5 frames around each zoom
    in_zoom = [bool(any(a <= tr["t"][i] <= b for a, b in zt)) for i in rev]
    spikes = int(sum(not z for z in in_zoom))
    rel = f / tr["maxdim"]
    span = float(np.log(f.max() / f.min()))
    osc = float(np.abs(dl).sum() / span) if span > 0.03 else 1.0  # path / range: a real zoom ~1, wobble >> 1
    return {"f_over_maxdim": [round(float(rel.min()), 3), round(float(rel.max()), 3)],
            "zoom_range": round(float(f.max() / f.min()), 3),
            "max_change_pct_per_frame": round(float(np.abs(dl).max()) * 100, 2) if len(dl) else 0.0,
            "spikes_outside_zoom": spikes, "reversals_inside_zoom": int(sum(in_zoom)), "zoom_events": zooms,
            "raw_vs_export_median_pct": round(float(np.median(resid)) * 100, 3), "oscillation": round(osc, 2),
            "plausible": bool(spikes == 0 and osc < 3.0 and np.median(resid) < 0.01 and 0.4 < rel.min() and rel.max() < 4.0)}


def run(take: Take, cams: list[str] | None = None, stages: tuple[str, ...] = STAGES, sigma_s: float = 0.04,
        max_reproj_px: float = 4.0) -> dict:
    work = take.work.resolve()
    cams = cams or [s.id for s in take.sources]
    out = work / "cameras"
    out.mkdir(parents=True, exist_ok=True)
    recs = list_images(work, cams)
    fs, ms, ls = FeatureSettings(), MapSettings(), LocSettings()
    F = Features(work, out / "cache", fs)
    rep_p = out / "report.json"
    rep = json.loads(rep_p.read_text()) if rep_p.exists() else {}
    if "map" in stages:
        rep["map"] = stage_map(work, recs, F, ms)
    if "loc" in stages:
        stage_loc(work, recs, F, ls, ms, cams)
    if "attach" in stages:
        rep["attach"] = stage_attach(work, recs, F, ls, ms, cams, sigma_s, max_reproj_px,
                                     statures_from_take(take.cfg))
    tracks = stage_smooth(work, cams, sigma_s, max_reproj_px)
    model = load_model(work)
    pts = np.array([p.xyz for p in model.points3D.values()]).reshape(-1, 3)
    rgb = np.array([p.color for p in model.points3D.values()]).reshape(-1, 3)
    sim3 = None
    sim3_p = out / "world_from_sfm.json"
    dancers = None
    if "world" in stages and tracks:
        dancers = triangulate_dancers(work, tracks)
        st, src = statures_from_take(take.cfg)
        main_cams = sorted(model_cams(model))  # world axes from the main model only (attached cams are coarser)
        sim3 = bootstrap(tracks, dancers, pts, st, src, x_cam="06" if "06" in main_cams else main_cams[0],
                         up_cams=main_cams)
        sim3_p.write_text(json.dumps(sim3, indent=1))
        np.savez(out / "cache" / "dancers_sfm.npz",
                 **{f"{r}_{k}": v for r, d in dancers.items() for k, v in d.items()})
        print(f"[cameras] world_from_sfm: s={sim3['s']:.4f} ({sim3['scale_source']}); {sim3['gravity_source']}; "
              f"{sim3['floor_source']}")
    elif sim3_p.exists():
        sim3 = json.loads(sim3_p.read_text())
    if "qa" in stages:
        rep.update(build_report(work, cams, recs, model, tracks, pts, rgb, sim3, rep.get("map", {}), sigma_s,
                                max_reproj_px, rep.get("attach")))
        rep_p.write_text(json.dumps(rep, indent=1, default=_np))
    return rep


def _np(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def dancer_check(work: Path, tracks: dict, cams: list[str], qa_dir: Path | None = None) -> dict:
    """Independent check of every camera: dancers' COCO keypoints triangulated from the OTHER cameras
    (leave-one-out) and reprojected into this camera's ok frames. Static-feature errors cannot show a camera
    that registered at a consistent but wrong place; this does (expect ~5-15 px from keypoint noise + sync)."""
    from .attach import observations
    from .world import dancer_keypoints
    out = {}
    for cam in cams:
        if cam not in tracks or len(tracks) < 3:
            continue
        others = {c: v for c, v in tracks.items() if c != cam}
        d3, grid = triangulate_dancers(work, others, kps=tuple(range(5, 17)), return_grid=True)
        tr = tracks[cam]
        kp2d = dancer_keypoints(work, cam, len(tr["t"]))
        fidx, X, uv, _ = observations(kp2d, tr["t"], tr["ok"], d3, grid)
        if len(fidx) < 20:
            out[cam] = {"n": int(len(fidx))}
            continue
        e, proj = [], {}
        for f in np.unique(fidx):
            sel = fidx == f
            p, vis = qa.project_cv(X[sel], tr["K"][f], tr["R"][f], tr["tvec"][f], tr["k1"][f])
            e.append(np.where(vis, np.linalg.norm(p - uv[sel], axis=1), 1e4))
            proj[int(f)] = (p, uv[sel])
        e = np.concatenate(e)
        if qa_dir is not None:  # overlay on the frame with the most keypoints: detected (green) vs LOO (magenta)
            import cv2
            f = max(proj, key=lambda k: len(proj[k][0]))
            meta = json.loads((work / "frames" / cam / "frames.json").read_text())
            img = cv2.imread(str(work / "frames" / cam / meta["frames"][f]["file"]))
            for (px, py), (ux, uy) in zip(*proj[f]):
                if abs(px) < 1e4 and abs(py) < 1e4:
                    cv2.line(img, (int(ux), int(uy)), (int(px), int(py)), (255, 0, 255), 1)
                    cv2.circle(img, (int(px), int(py)), 5, (255, 0, 255), 2)
                cv2.drawMarker(img, (int(ux), int(uy)), (0, 255, 0), cv2.MARKER_CROSS, 10, 2)
            sz = img.shape[1] / 480
            cv2.putText(img, f"cam {cam} f{f}: dancers from other cams (magenta) vs detected (green)", (8, int(24 * sz)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45 * sz, (255, 255, 255), max(1, int(sz)))
            cv2.imwrite(str(qa_dir / f"dancers_loo_{cam}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        out[cam] = {"n": int(len(e)), "median_px": round(float(np.median(e)), 2),
                    "p90_px": round(float(np.percentile(e, 90)), 2),
                    "median_rel_height": round(float(np.median(e)) / tr["maxdim"], 4)}
    return out


def _model_errors(model) -> dict[str, list[float]]:
    """Per camera id: reprojection errors (px) of every observation in the model."""
    out: dict[str, list[float]] = {}
    for im in model.images.values():
        if not im.has_pose:
            continue
        T = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
        errs = []
        for p2 in im.points2D:
            if p2.has_point3D():
                uv = im.camera.img_from_cam(T * model.points3D[p2.point3D_id].xyz)
                errs.append(float(np.linalg.norm(np.asarray(uv) - p2.xy)))
        out.setdefault(im.name.split("/")[1], []).extend(errs)
    return out


def build_report(work: Path, cams, recs, model, tracks, pts, rgb, sim3, map_info, sigma_s, max_reproj_px,
                 attach: dict | None = None) -> dict:
    import cv2
    qa_dir = work / "cameras" / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    import pycolmap
    per_img_err = _model_errors(model)
    reg = model_cams(model)
    sub_of: dict[str, str] = {}
    for d in sorted((work / "cameras" / "sfm_work").glob("sub_*/best")):
        sm = pycolmap.Reconstruction(str(d))
        for cam, errs in _model_errors(sm).items():
            if cam not in per_img_err:
                per_img_err[cam] = errs
                sub_of[cam] = d.parent.name
    attach = attach or {}
    cams_rep = {}
    for cam in cams:
        win = [r for r in recs[cam] if r.kind == "win"]
        meta = json.loads((work / "frames" / cam / "frames.json").read_text())
        hd = max(meta["width"], meta["height"]) >= 1280
        att = attach.get(cam, {})
        e = {"registered": cam in reg or bool(att.get("attached")),
             "how": "main model" if cam in reg else ("own sub-model + dancer Sim3 (attach)" if att.get("attached")
                                                     else "unregistered"),
             "map_images": reg.get(cam, 0), "frames": len(win), "hd": hd,
             "map": "sfm" if cam in reg else sub_of.get(cam, ""),
             "map_reproj_px": float(np.mean(per_img_err[cam])) if cam in per_img_err else float("nan"),
             "map_reproj_median_px": float(np.median(per_img_err[cam])) if cam in per_img_err else float("nan")}
        if att:
            e["attach"] = att
        tr = tracks.get(cam)
        if tr is not None:
            src = tr["source"]
            e.update({"frames_ok": int(tr["ok"].sum()), "frames_ok_raw": int(tr["ok_raw"].sum()),
                      "ok_rate": round(float(tr["ok"].mean()), 4),
                      "sources": {s: int((src == s).sum()) for s in ("map", "loc", "track", "attach")},
                      "raw_pose_kept_frames": tr["raw_used"],
                      "reproj_px_median": float(np.nanmedian(tr["reproj_px"][tr["ok"]])) if tr["ok"].any() else None,
                      "reproj_px_p90": float(np.nanpercentile(tr["reproj_px"][tr["ok"]], 90)) if tr["ok"].any() else None,
                      "reproj_raw_median": float(np.nanmedian(tr["reproj_raw"][tr["ok_raw"]])) if tr["ok_raw"].any() else None,
                      "n_inliers_median": float(np.median(tr["n_inliers"][tr["ok"]])) if tr["ok"].any() else 0,
                      "focal": focal_stats(tr)})
            if sim3 is not None:
                Cw = apply(sim3, tr["C"])
                e["height_m"] = [round(float(np.percentile(Cw[tr["ok"], 1], q)), 3) for q in (5, 50, 95)] \
                    if tr["ok"].any() else None
                e["dist_to_origin_m"] = round(float(np.median(np.linalg.norm(Cw[:, [0, 2]], axis=1))), 2)
                e["path_length_m"] = round(float(np.linalg.norm(np.diff(Cw, axis=0), axis=1).sum()), 2)
        cams_rep[cam] = e
    rep = {"cams": cams_rep, "settings": {"sigma_s": sigma_s, "max_reproj_px": max_reproj_px},
           "points3D": int(len(pts)), "map_mean_reproj_px": float(model.compute_mean_reprojection_error())}
    if tracks:
        rep["dancer_loo_check"] = dancer_check(work, tracks, cams, qa_dir)
    rep["gates_G2a"] = qa.gates(rep, len(cams))
    # plots
    if tracks:
        wt = {}
        for cam, tr in tracks.items():
            if sim3 is not None:
                Rw = np.asarray(sim3["R"])
                wt[cam] = {"C": apply(sim3, tr["C"]), "R": tr["R"] @ Rw.T, "ok": tr["ok"]}
            else:
                wt[cam] = {"C": tr["C"], "R": tr["R"], "ok": tr["ok"]}
        P = apply(sim3, pts) if sim3 is not None else pts
        ank = None
        dz = work / "cameras" / "cache" / "dancers_sfm.npz"
        if dz.exists() and sim3 is not None:
            d = np.load(dz)
            a = np.concatenate([d[k] for k in d.files if k.endswith("_15") or k.endswith("_16")])
            ank = apply(sim3, a[np.isfinite(a).all(1)])
        qa.plot_layout(qa_dir / "layout.png", wt, P, rgb, ank,
                       f"cameras: {len(tracks)} tracks, {len(pts)} points" + ("" if sim3 else " (SfM frame)"))
        qa.plot_focal(qa_dir / "focal.png", tracks)
        rep["qa"] = {"layout": str(qa_dir / "layout.png"), "focal": str(qa_dir / "focal.png"), "overlays": []}
        for cam, tr in tracks.items():
            raw = np.load(work / "cameras" / "cache" / f"loc_{cam}.npz")
            win = [r for r in recs[cam] if r.kind == "win"]
            ok_idx = np.nonzero(tr["ok"])[0]
            if not len(ok_idx):
                continue
            i = int(ok_idx[len(ok_idx) // 2])
            a, b = raw["corr_off"][i], raw["corr_off"][i + 1]
            img = cv2.imread(str(work / win[i].name))
            p = qa_dir / f"reproj_{cam}.jpg"
            qa.overlay(p, img, tr["K"][i], tr["R"][i], tr["tvec"][i], tr["k1"][i], raw["corr_xy"][a:b],
                       raw["corr_xyz"][a:b], sim3, f"cam {cam} frame {i} ({raw['source'][i]})")
            rep["qa"]["overlays"].append(str(p))
    return rep
