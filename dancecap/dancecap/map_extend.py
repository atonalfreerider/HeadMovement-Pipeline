"""C7: conditional map extension for ONE camera of one segment take (larissa_kadu_full PLAN.md C7), run on the pod by
`dancecap.front_full locgate` when that camera fails the P2 localisation gate (>= 90 % of frames ok, no failed
stretch over 8 s).

    python -m dancecap.map_extend <segment take> <cam>

The reused SfM map stays IDENTICAL: no pose or point of it moves, so cameras/world_from_sfm.json (copied from
larissa_kadu_38: gravity, floor, betas-corrected scale) stays valid. New 3D points are triangulated, with the poses
held fixed, from this camera's own frames that DID localise around each failed stretch (pairs 0.2-1.5 s apart: the
handheld operator's motion is the baseline; dancers and spectators are masked out of the features, so the new points
are background). The failed frames are then re-localised against map points + the new points, in passes of
alternating direction; a newly localised frame contributes its own inlier correspondences (map + new points) to its
neighbours, i.e. local mapping in the SfM frame. Acceptance is stricter than the normal localiser (>= 50 inliers,
median reprojection <= 3 px, >= 8 of them map points so a stretch never runs on new points alone); such frames carry
source 'ext'.

Main-model cameras: cameras/cache/loc_<cam>.npz is rewritten (the original kept as loc_<cam>.pre_c7.npz) and
cameras/<cam>.npz re-smoothed (stage_smooth, the cli defaults, as front_pod.loc_join). Attached cameras (05, 06):
cameras/cache/locsub_<cam>.pkl (sub-model frame) is rewritten under the SAME cache key, so the cameras stage's attach
uses it (original kept as locsub_<cam>.pre_c7.pkl). Report: cameras/map_extend_<cam>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import time
from pathlib import Path

import numpy as np

from .take import Take

MIN_INLIERS = 50
MAX_REPROJ_PX = 3.0
MIN_MAP_INLIERS = 8
TRI_MAX_PX = 2.0
TRI_MIN_ANGLE_DEG = 1.0
ANCHOR_SPAN_S = 2.0      # own localised frames this far around a failed stretch (+ islands inside it)
ANCHOR_STEP = 5          # every 5th of them
PAIR_DT_S = (0.2, 1.5)   # own-camera pair separation (baseline from the operator's motion)
MAX_PASSES = 4


# ------------------------------------------------------------------------------------------------ geometry


def undistort(xy: np.ndarray, f: float, k1: float, w: int, h: int, iters: int = 8) -> np.ndarray:
    """COLMAP pixels (SIMPLE_RADIAL, principal point at the image centre) -> normalised undistorted coordinates."""
    d = (np.asarray(xy, float) - [w / 2, h / 2]) / f
    u = d.copy()
    for _ in range(iters):
        r2 = (u ** 2).sum(-1, keepdims=True)
        u = d / (1 + k1 * r2)
    return u


def project(R, t, f, k1, w, h, X) -> tuple[np.ndarray, np.ndarray]:
    """(COLMAP pixels, depth) of world points X under the SIMPLE_RADIAL model."""
    Xc = np.asarray(X, float) @ np.asarray(R).T + np.asarray(t)
    z = Xc[:, 2]
    uv = Xc[:, :2] / np.where(np.abs(z) < 1e-12, 1e-12, z)[:, None]
    r2 = (uv ** 2).sum(-1, keepdims=True)
    return f * uv * (1 + k1 * r2) + [w / 2, h / 2], z


def triangulate(pa: dict, pb: dict, xa: np.ndarray, xb: np.ndarray, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """Two-view DLT with fixed poses (pose dicts R, t, f, k1; xa / xb COLMAP pixels) -> (X (N, 3), good (N,))."""
    if len(xa) == 0:
        return np.zeros((0, 3)), np.zeros(0, bool)
    ua, ub = undistort(xa, pa["f"], pa["k1"], w, h), undistort(xb, pb["f"], pb["k1"], w, h)
    Pa = np.hstack([pa["R"], np.asarray(pa["t"]).reshape(3, 1)])
    Pb = np.hstack([pb["R"], np.asarray(pb["t"]).reshape(3, 1)])
    A = np.stack([ua[:, :1] * Pa[2] - Pa[0], ua[:, 1:] * Pa[2] - Pa[1],
                  ub[:, :1] * Pb[2] - Pb[0], ub[:, 1:] * Pb[2] - Pb[1]], 1)  # (N, 4, 4)
    _, _, vt = np.linalg.svd(A)
    Xh = vt[:, -1]
    X = Xh[:, :3] / np.where(np.abs(Xh[:, 3:]) < 1e-12, 1e-12, Xh[:, 3:])
    ra, za = project(pa["R"], pa["t"], pa["f"], pa["k1"], w, h, X)
    rb, zb = project(pb["R"], pb["t"], pb["f"], pb["k1"], w, h, X)
    ea, eb = np.linalg.norm(ra - xa, axis=1), np.linalg.norm(rb - xb, axis=1)
    Ca, Cb = -np.asarray(pa["R"]).T @ np.asarray(pa["t"]), -np.asarray(pb["R"]).T @ np.asarray(pb["t"])
    da, db = X - Ca, X - Cb
    cosang = (da * db).sum(1) / np.maximum(np.linalg.norm(da, axis=1) * np.linalg.norm(db, axis=1), 1e-12)
    ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
    good = (za > 0) & (zb > 0) & (ea < TRI_MAX_PX) & (eb < TRI_MAX_PX) & (ang > TRI_MIN_ANGLE_DEG) & np.isfinite(X).all(1)
    return X, good


def stretches(ok: np.ndarray, merge: int = 15) -> list[tuple[int, int]]:
    """[i0, i1] runs of failed frames; runs closer than `merge` frames are one stretch (islands inside)."""
    bad = np.flatnonzero(~np.asarray(ok, bool))
    if len(bad) == 0:
        return []
    out = [[int(bad[0]), int(bad[0])]]
    for i in bad[1:]:
        if i - out[-1][1] <= merge:
            out[-1][1] = int(i)
        else:
            out.append([int(i), int(i)])
    return [(a, b) for a, b in out]


# ------------------------------------------------------------------------------------------------ the extension


def _kp_index(kpts: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Keypoint indices of stored correspondence pixels (corr_xy = kpts[idx], exact); -1 when absent."""
    lut = {(round(float(a), 3), round(float(b), 3)): i for i, (a, b) in enumerate(kpts)}
    return np.array([lut.get((round(float(a), 3), round(float(b), 3)), -1) for a, b in xy], np.int64)


def extend(take_name: str, cam: str, dry_run: bool = False) -> dict:
    from .cameras.features import Features, FeatureSettings, list_images
    from .cameras.localize import FramePose, Localizer, Ref
    from .cameras.mapping import MapSettings
    from .cameras.pipeline import load_model, loc_settings, map_images
    take = Take.load(take_name)
    work = take.work.resolve()
    cache = work / "cameras" / "cache"
    recs = list_images(work, [s.id for s in take.sources])
    frames = [r for r in recs[cam] if r.kind == "win"]
    meta = json.loads((work / "frames" / cam / "frames.json").read_text())
    w, h = int(meta["width"]), int(meta["height"])
    F = Features(work, cache, FeatureSettings())
    images, _ = map_images(work, recs, MapSettings())
    ls = loc_settings(take)
    main_p, sub_p = cache / f"loc_{cam}.npz", cache / f"locsub_{cam}.pkl"
    t_start = time.time()
    if main_p.exists():
        kind = "main"
        model = load_model(work)
        raw = dict(np.load(main_p, allow_pickle=True))
        n = len(raw["ok"])
        poses: list = []
        for i in range(n):
            a, b = int(raw["corr_off"][i]), int(raw["corr_off"][i + 1])
            fp = FramePose(bool(raw["ok"][i]), np.asarray(raw["R"][i]), np.asarray(raw["tvec"][i]), float(raw["f"][i]),
                           float(raw["k1"][i]), int(raw["n_inliers"][i]), float(raw["reproj_px"][i]),
                           str(raw["source"][i]))
            if fp.ok and b > a:
                xy = np.asarray(raw["corr_xy"][a:b], float)
                idx = _kp_index(F.get(frames[i])["kpts"], xy)
                keep = idx >= 0
                fp.corr = Ref(frames[i], idx[keep], np.asarray(raw["corr_xyz"][a:b])[keep], np.zeros(int(keep.sum()), np.int64))
                fp.q_xy = xy[keep]
            poses.append(fp)
    elif sub_p.exists():
        import pycolmap

        from .cameras.mapping import model_cams
        from .cameras.localize import LocSettings
        kind = "sub"
        doc = pickle.loads(sub_p.read_bytes())
        poses = list(doc["value"])
        n = len(poses)
        subs = []
        for d in sorted((work / "cameras" / "sfm_work").glob("sub_*/best")):
            if (d / "images.bin").exists():
                m = pycolmap.Reconstruction(str(d))
                subs.append((d.parent.name, m, model_cams(m)))
        name, model, _ = max([x for x in subs if x[2].get(cam, 0) >= 3], key=lambda x: x[2][cam])
        single = len({im.camera_id for im in model.images.values()}) == 1
        ls = LocSettings(**{**ls.__dict__, "fix_focal": single})
    else:
        raise SystemExit(f"[c7] {take_name} {cam}: no localisation output")
    if n != len(frames):
        raise SystemExit(f"[c7] {take_name} {cam}: {n} poses for {len(frames)} frames")
    L = Localizer(F, model, images, ls)
    t = np.array([r.t for r in frames])
    dt = float(np.median(np.diff(t))) if n > 1 else 1 / 30
    ok0 = np.array([p.ok for p in poses])
    runs = stretches(ok0)
    rep = {"take": take_name, "cam": cam, "model": kind, "frames": n, "ok_before": int(ok0.sum()),
           "stretches": [], "settings": {"min_inliers": MIN_INLIERS, "max_reproj_px": MAX_REPROJ_PX,
                                         "min_map_inliers": MIN_MAP_INLIERS, "tri_max_px": TRI_MAX_PX,
                                         "tri_min_angle_deg": TRI_MIN_ANGLE_DEG}}
    ext_refs: dict[int, Ref] = {}

    def pdict(p):
        return {"R": p.R, "t": p.t, "f": p.f, "k1": p.k1}

    span = int(round(ANCHOR_SPAN_S / dt))
    lo_dt, hi_dt = (int(round(x / dt)) for x in PAIR_DT_S)
    for i0, i1 in runs:
        a0, a1 = max(0, i0 - span), min(n - 1, i1 + span)
        anchors = [j for j in range(a0, a1 + 1) if poses[j].ok and poses[j].corr is not None][::ANCHOR_STEP]
        new_pts = 0
        for j in anchors:
            fj = F.get(frames[j])
            kp_j = fj["kpts"] + 0.5
            add_idx, add_xyz = [], []
            have = set(poses[j].corr.kp_idx.tolist())
            for k in anchors:
                if not (lo_dt <= abs(k - j) <= hi_dt):
                    continue
                m = L.match_cache.get((frames[j].key, frames[k].key))
                if m is None:
                    m = F.match(frames[j], frames[k])
                    L.match_cache[(frames[j].key, frames[k].key)] = m
                if len(m) == 0:
                    continue
                m = m[[int(x) not in have for x in m[:, 0]]]
                if len(m) == 0:
                    continue
                kp_k = F.get(frames[k])["kpts"] + 0.5
                X, good = triangulate(pdict(poses[j]), pdict(poses[k]), kp_j[m[:, 0]], kp_k[m[:, 1]], w, h)
                add_idx.append(m[good, 0].astype(np.int64))
                add_xyz.append(X[good])
                have |= set(m[good, 0].tolist())
            if add_idx:
                ai, ax = np.concatenate(add_idx), np.concatenate(add_xyz)
                c = poses[j].corr
                ext_refs[j] = Ref(frames[j], np.r_[c.kp_idx, ai], np.r_[c.xyz, ax],
                                  np.r_[np.where(c.pid >= 0, c.pid, 0), -np.ones(len(ai), np.int64)])
                new_pts += len(ai)
        rep["stretches"].append({"t": [round(float(t[i0]), 2), round(float(t[i1]), 2)], "frames": i1 - i0 + 1,
                                 "failed": int((~ok0[i0:i1 + 1]).sum()), "anchors": len(anchors), "new_points": new_pts})
    print(f"[c7] {take_name} {cam}: {len(runs)} stretches, {len(ext_refs)} anchor frames with "
          f"{sum(s['new_points'] for s in rep['stretches'])} new points ({time.time() - t_start:.0f} s)", flush=True)
    # re-localise the failed frames: passes of alternating direction, each new pose feeding its neighbours
    n_map = {}
    for k_pass in range(MAX_PASSES):
        changed = 0
        order = range(n) if k_pass % 2 == 0 else range(n - 1, -1, -1)
        for i in order:
            if poses[i].ok:
                continue
            near = sorted((j for j in ext_refs if abs(j - i) <= 3 * span), key=lambda j: abs(j - i))[:4]
            nb = [poses[j].corr for j in (i - 1, i + 1, i - 2, i + 2) if 0 <= j < n and poses[j].ok
                  and poses[j].corr is not None and poses[j].source == "ext"]
            refs = [ext_refs[j] for j in near] + nb + L.candidates(frames[i], cross=False)
            if not near and not nb:
                continue
            prev = next((poses[j] for j in (i - 1, i + 1) if 0 <= j < n and poses[j].ok), None)
            ni = L.neighbour_intrinsics(frames[i], prev)
            if ni is None:
                continue
            fp = L.localize(frames[i], refs, w, h, *ni)
            if fp.corr is None or fp.n_inliers < MIN_INLIERS or not (fp.reproj_px <= MAX_REPROJ_PX):
                continue
            nm = int((fp.corr.pid >= 0).sum())
            if nm < MIN_MAP_INLIERS:
                continue
            fp.ok, fp.source = True, "ext"
            poses[i] = fp
            n_map[i] = nm
            ext_refs[i] = fp.corr
            changed += 1
        print(f"[c7] {take_name} {cam}: pass {k_pass + 1}: +{changed} frames", flush=True)
        if not changed:
            break
    ok1 = np.array([p.ok for p in poses])
    rep.update(ok_after=int(ok1.sum()), recovered=int((ok1 & ~ok0).sum()),
               recovered_map_inliers_median=int(np.median(list(n_map.values()))) if n_map else None,
               seconds=round(time.time() - t_start, 1))
    rep["dry_run"] = dry_run
    if rep["recovered"] and not dry_run:
        if kind == "main":
            from .cameras.pipeline import _save_raw, stage_smooth
            from .cameras.typed import LOC_MARKER_KEYS
            bak = main_p.with_name(f"loc_{cam}.pre_c7.npz")
            if not bak.exists():
                shutil.copyfile(main_p, bak)
            tmp = main_p.with_name(f"loc_{cam}.c7.npz")
            marker = {k: raw[k] for k in LOC_MARKER_KEYS if k in raw}
            for p in poses:  # pid is not stored by _save_raw; restore the map ids' sign convention only in memory
                if p.ok and p.corr is not None and getattr(p, "q_xy", None) is None:
                    p.q_xy = F.get(p.corr.rec)["kpts"][p.corr.kp_idx]
            _save_raw(tmp, frames, poses, marker)
            os.replace(tmp, main_p)
            stage_smooth(work, [cam], 0.04, 4.0)  # as front_pod.loc_join
        else:
            bak = sub_p.with_name(f"locsub_{cam}.pre_c7.pkl")
            if not bak.exists():
                shutil.copyfile(sub_p, bak)
            tmp = sub_p.with_name(f"locsub_{cam}.c7.pkl")
            tmp.write_bytes(pickle.dumps({"key": doc["key"], "value": poses}))
            os.replace(tmp, sub_p)
    p = work / "cameras" / f"map_extend_{cam}{'.dry' if dry_run else ''}.json"
    p.write_text(json.dumps(rep, indent=1))
    print(f"[c7] {take_name} {cam}: ok {rep['ok_before']} -> {rep['ok_after']} of {n} (+{rep['recovered']}) -> {p}",
          flush=True)
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m dancecap.map_extend", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("take")
    ap.add_argument("cam")
    ap.add_argument("--dry-run", action="store_true", help="report only: the localisation files are not rewritten")
    a = ap.parse_args(argv)
    extend(a.take, a.cam, a.dry_run)


if __name__ == "__main__":
    main()
