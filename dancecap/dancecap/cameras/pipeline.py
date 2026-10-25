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
from .localize import FramePose, Localizer, LocSettings, localize_camera, settings_key, sub_settings
from .mapping import (MapSettings, build_database, link_components, map_pairs, map_settings, match_pairs, model_cams,
                      refine_model, run_mapper, run_sub_mapper, select_map_images, targeted_pairs)
from .depth_guard import DepthGuardSettings, apply as depth_guard
from .smooth import smooth_track
from .typed import TypedSettings, loc_marker, loc_mode, localize_typed, locked_tripod, refit as type_refit
from .world import (apply, bootstrap, roles_coverage, scale_checks, scale_rule_from_take, sole_from_take,
                    statures_from_take, triangulate_dancers)
from .world_scale import adopt_reference, reference_from_take

STAGES = ("map", "loc", "attach", "world", "qa")


def _K(f: np.ndarray, w: int, h: int) -> np.ndarray:
    """OpenCV intrinsics per frame (pixel-centre convention: COLMAP's w/2 principal point - 0.5)."""
    K = np.zeros((len(f), 3, 3))
    K[:, 0, 0] = K[:, 1, 1] = f
    K[:, 0, 2], K[:, 1, 2], K[:, 2, 2] = w / 2 - 0.5, h / 2 - 0.5, 1.0
    return K


def map_images(work: Path, recs: dict, ms: MapSettings | None) -> tuple[list[ImageRec], set[str]]:
    """The map's images: select_map_images, or - for a map reused from another take of the same recordings
    (cameras/reuse.py) - exactly the images of the reused models (cameras/sfm_work/map_images.json).
    The list stage_map recorded (sfm_work/map.json map_images) wins over the selection when the two differ for the
    cameras of `recs` - [cameras] map edited after the map was built: the Localizer keeps only listed images as
    references, so the new settings would silently drop map images (a loud WARNING says so). Same lists: the selection
    as it was. ms=None: the recorded list (RuntimeError without one)."""
    p = work / "cameras" / "sfm_work" / "map_images.json"
    if p.exists():
        from .reuse import reused_images
        return reused_images(work, json.loads(p.read_text())), set()
    mp = work / "cameras" / "sfm_work" / "map.json"
    built = json.loads(mp.read_text()).get("map_images") if mp.exists() else None
    if built is not None:
        built = {k for k in built if k.split("/")[0] in recs}

    def recorded():  # in select_map_images' order: per camera its window images, then its keyframes
        return [r for rs in recs.values() for kind in ("win", "kf") for r in rs if r.kind == kind and r.key in built]
    if ms is None:
        if built is None:
            raise RuntimeError(f"[cameras] map images: no MapSettings given and {mp} records no map_images")
        return recorded(), set()
    images, dense = select_map_images(recs, ms)
    if built is None or built == {r.key for r in images}:  # no map yet / a map.json from before the list / unchanged
        return images, dense
    have = {r.key for r in images}
    print(f"[cameras] WARNING map images: the [cameras] map settings select {len(have - built)} image(s) the map was "
          f"not built with and miss {len(built - have)} of its {len(built)} (cameras {sorted(recs)}): the localisation "
          f"uses the map's own list (sfm_work/map.json); restore the settings or re-run --stages map", flush=True)
    return recorded(), dense & built


def loc_settings(take: Take) -> LocSettings:
    """LocSettings with takes/<take>.toml [cameras] loc = {...} overrides (e.g. kf_by_time for a reused map)."""
    return LocSettings(**((take.cfg.get("cameras") or {}).get("loc") or {}))


def world_x_cam(take_cfg: dict, main_cams: list[str], types: dict | None = None) -> tuple[str, str]:
    """(camera whose mean right vector is the world X axis, why): takes/<take>.toml [cameras] x_cam, else the first
    main-model camera of a tripod class (one centre: a stable yaw; an orbiting camera's mean right vector nearly
    cancels and leaves the yaw arbitrary), else '06' when it is in the main model, else the first main camera.
    types: {cam: camera_types.StageType} as applied (camera_types_for)."""
    x = ((take_cfg or {}).get("cameras") or {}).get("x_cam")
    if x is not None and str(x) in main_cams:
        return str(x), "[cameras] x_cam"
    late = "" if x is None else f" ([cameras] x_cam {x!r} is not a main-model camera)"
    for c in main_cams:
        st = (types or {}).get(c)
        if st is not None and st.cls.startswith("tripod"):
            return c, f"first main-model tripod ({st.cls}){late}"
    return ("06", f"camera 06 in the main model{late}") if "06" in main_cams else (main_cams[0], f"first main camera{late}")


def few_main_reason(reg_cams: list[str]) -> str:
    """Why attach and world are skipped (report.json) instead of crashing in world.bootstrap: the main MODEL holds
    fewer than 2 cameras. (Fewer than 2 localised cameras of a larger model is an error: require_main_tracks.)"""
    return (f"the main model holds only {len(reg_cams)} camera(s) {list(reg_cams)}: the dancers cannot be "
            "triangulated (attach and world need >= 2; a two-camera take whose second camera is not in the main "
            "model: see cameras/sfm_work/map.json strong_link_groups)")


def require_main_tracks(reg: dict, tracks: dict, what: str) -> None:
    """RuntimeError, before anything is skipped or deleted, when the main model holds >= 2 cameras but fewer than 2 of
    them have a localisation in this run: a --cams subset without them, or a missing cameras/cache/loc_<cam>.npz."""
    have = sorted(c for c in reg if c in tracks)
    if len(have) < 2:
        raise RuntimeError(f"[cameras] {what} needs the main-model cameras' localisations: the main model holds "
                           f"{sorted(reg)}, this run has {have} - pass --cams with all of them, or run --stages loc "
                           "(front_pod loc-part / loc-join) for the missing ones")


TWO_VIEW_SYNC = "two_view_epipolar"  # sync/offsets.json method of cameras.twoview sync


def world_sync_tracks(work: Path, tracks: dict) -> tuple[dict, dict | None]:
    """(main tracks to triangulate the world's dancers from, sync record or None). A two-camera take's audio clocks are
    off by tens of ms, which biases every triangulated joint of a moving dancer and with them the chain scale and the
    ankle floor: when sync/offsets.json is a two-view epipolar sync (cameras.twoview sync; it needs no world frame),
    the frame times are shifted by it (mamma_dyn.apply_offset, as world_scale chain) before the triangulation. Any
    other offsets file (dancecap.sync) is not applied: those takes' worlds stay on the audio clocks, as before.
    Refuses (SystemExit) offsets measured on other audio clocks (mamma_dyn.read_sync_offsets)."""
    p = work / "sync" / "offsets.json"
    try:
        raw = json.loads(p.read_text()) if p.exists() else {}
    except ValueError:
        raw = {}
    if not isinstance(raw, dict) or raw.get("method") != TWO_VIEW_SYNC:
        return tracks, None
    from ..mamma_dyn import apply_offset, read_sync_offsets
    off = read_sync_offsets(work)
    out = {c: {**tr, "t": apply_offset(tr["t"], off[c])} if c in off else tr for c, tr in tracks.items()}
    rec = {"applied": True, "file": "sync/offsets.json", "method": raw["method"], "gauge": raw.get("gauge"),
           "offsets_ms": {c: round(1000 * float(off[c]["delta"]), 2) for c in sorted(tracks) if c in off},
           "rate_ms_per_100s": {c: round(1e5 * float(off[c].get("rate", 0.0)), 3) for c in sorted(tracks) if c in off},
           "not_in_file": sorted(c for c in tracks if c not in off)}
    return out, rec


def stage_map(work: Path, recs: dict, F: Features, ms: MapSettings) -> dict:
    out = work / "cameras"
    (out / "sfm_work" / "map_images.json").unlink(missing_ok=True)  # a fresh map replaces a reused one
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
    if any(e["mode"] != "per_image" for e in st["intrinsics"].values()):
        print("[cameras] intrinsics: " + ", ".join(f"{c} {e['mode']} ({e['cameras']} camera(s), prior {e['focal_prior_px']:.0f} px)"
                                                   for c, e in st["intrinsics"].items()), flush=True)
    st.pop("pair_inliers")  # per_campair keeps the counts: pairs, max / sum inliers, link_pairs (>= link_min_inliers)
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
    dg = None
    if ms.doppelganger_check:
        from .plausible import _summary, clean_model
        dg = clean_model(best)
        if dg["flagged"]:
            print(_summary(dg), flush=True)
            refine = {**refine, "after_doppelgangers": refine_model(best, ms.refine_px)}
    sfm = out / "sfm"
    sfm.mkdir(parents=True, exist_ok=True)
    best.write_binary(str(sfm))
    shutil.rmtree(out / "sfm_orig", ignore_errors=True)  # a plausible-clean backup belongs to the map it was made from
    subs = build_sub_models(work, F, images, matches, groups[1:], ms)
    info = {"images": len(images), "dense": len(dense), "pairs": len(matches), "match_s": round(t_match, 1),
            "mapping_s": round(time.time() - t0, 1), "verification": st,
            "strong_link_groups": [sorted(g) for g in groups], "main_cams": sorted(main_cams),
            "intrinsics_mode": {**{c: e["mode"] for c, e in st["intrinsics"].items()},  # the model each camera is in
                                **{c: m for v in subs.values() if v for c, m in v["intrinsics_mode"].items()}},
            "models": raw_models, "refine": refine, "sub_models": subs, "doppelgangers": dg,
            "map_images": [r.key for r in images]}
    (sw / "map.json").write_text(json.dumps(info, indent=1, default=str))
    for m in raw_models:
        print(f"[cameras] model: {m['images']} images, {m['points']} pts, {m['reproj_px']:.3f} px, {m['cams']}")
    print(f"[cameras] refined main model: {refine}")
    return info


def build_sub_models(work: Path, F: Features, images: list[ImageRec], matches: dict, groups: list[set],
                     ms: MapSettings) -> dict:
    """One COLMAP sub-model per weakly linked camera group -> sfm_work/sub_<name>/best (used by attach). Each entry
    records its cameras' intrinsics_mode ('single' for a one-camera group: map.json's intrinsics_mode takes it)."""
    sw = work / "cameras" / "sfm_work"
    subs: dict = {}
    for g in groups:
        name = "_".join(sorted(g))
        sdb = sw / f"database_sub_{name}.db"
        sst = build_database(work, F, images, matches, sdb, ms, cams=g, single_camera=len(g) == 1)
        if (sw / f"sub_{name}").exists():
            shutil.rmtree(sw / f"sub_{name}")
        sb = run_sub_mapper(work, sdb, sw / f"sub_{name}", ms)
        if sb is None:
            subs[name] = None
            print(f"[cameras] sub-model {name}: mapping failed", flush=True)
            continue
        subs[name] = {"images": sb.num_reg_images(), "points": sb.num_points3D(), "cams": model_cams(sb),
                      "refine": refine_model(sb, ms.refine_px),
                      "intrinsics_mode": {c: e["mode"] for c, e in sst["intrinsics"].items()}}
        (sw / f"sub_{name}" / "best").mkdir(parents=True, exist_ok=True)
        sb.write_binary(str(sw / f"sub_{name}" / "best"))
        print(f"[cameras] sub-model {name}: {subs[name]}", flush=True)
    return subs


def load_model(work: Path):
    import pycolmap
    return pycolmap.Reconstruction(str(work / "cameras" / "sfm"))


def camera_types_for(work: Path, types: dict | None = None, policy: str | None = None) -> dict:
    """{cam: camera_types.StageType} to apply: `types` as given (an explicit {} = none), else from the take TOML of
    work/<take> under the policy (DANCECAP_CAMERA_TYPES / --camera-types, default 'user' = confirmed answers only).
    Takes without camera types (and every unconfirmed camera) get today's general model."""
    if types is not None:
        return types
    from .. import camera_types as ct
    try:
        return {c: t for c, t in ct.types_for_work(work, policy=policy).items() if not t.general}
    except ct.CameraTypeError as e:
        print(f"[cameras] WARNING camera types not read ({e}): today's general model for every camera")
        return {}


def stage_loc(work: Path, recs: dict, F: Features, ls: LocSettings, ms: MapSettings, cams: list[str],
              types: dict | None = None, policy: str | None = None) -> dict:
    model = load_model(work)
    images, _ = map_images(work, recs, ms)
    L = Localizer(F, model, images, ls)
    out = {}
    types = camera_types_for(work, types, policy)
    pol = _policy_name(policy)
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
            st = types.get(cam)
            if locked_tripod(st):  # keyframes + one pose; every frame when the constant pose is refused there
                poses, how = localize_typed(L, frames, w, h, st)
                out[f"{cam}_typed"] = how
            else:
                poses, how = localize_camera(L, frames, w, h), {"mode": "all", "localised": len(frames)}
            _save_raw(work / "cameras" / "cache" / f"loc_{cam}.npz", frames, poses,
                      loc_marker(how["mode"], st.cls if st is not None else "handheld_zoom", pol, how.get("localised")))
            n_ok = sum(p.ok for p in poses)
            print(f"[cameras] {cam}: localised {n_ok}/{len(frames)} ({time.time() - t0:.0f}s)", flush=True)
            out[cam] = n_ok
            L.match_cache.clear()
    return out


VERTICAL_VERSION = 3  # bump to invalidate cameras/cache/vertical_*.pkl after a change in vertical.py


def _file_key(*paths: Path) -> list:
    return [[str(p.name), p.stat().st_mtime_ns, p.stat().st_size] if p.exists() else [str(p.name), None]
            for p in paths]


def _norm_key(o):
    """A cache key with its file mtimes in whole seconds: _file_key records nanoseconds, a work dir that was downloaded
    from a pod (tar) or copied keeps only seconds, which invalidated every cache of larissa_kadu_full_* locally."""
    if isinstance(o, bool) or o is None:
        return o
    if isinstance(o, int) and o >= 10 ** 18:
        return o // 10 ** 9
    if isinstance(o, dict):
        return {k: _norm_key(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_norm_key(v) for v in o]
    return o


LOOSE_FILE_KEYS = False  # True (attach_zoom.reattach): cache keys compare files by name + size only


def _loose_key(o):
    """[name, mtime, size] file records -> [name, size]. A work dir whose files were copied between takes (reused maps)
    or extracted from a pod tar keeps other mtimes than the ones the cache was keyed with, although the file is the
    same: the sub-model localisations of larissa_kadu_full_{b,c,d} would otherwise need the frame images again."""
    if isinstance(o, (list, tuple)):
        if len(o) == 3 and isinstance(o[0], str) and o[0].endswith((".bin", ".npz", ".pkl", ".db")) \
                and isinstance(o[2], int):
            return [o[0], o[2]]
        return [_loose_key(v) for v in o]
    if isinstance(o, dict):
        return {k: _loose_key(v) for k, v in o.items()}
    return o


def _key_form(o):
    o = _norm_key(o)
    return _loose_key(o) if LOOSE_FILE_KEYS else o


def _cached(path: Path, key, compute):
    """pickle cache with an explicit key (inputs' mtimes + settings + code version)."""
    import pickle
    if path.exists():
        try:
            d = pickle.loads(path.read_bytes())
            if _key_form(d.get("key")) == _key_form(key):
                return d["value"]
        except Exception:  # noqa: BLE001 - stale/corrupt cache: recompute
            pass
    v = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps({"key": key, "value": v}))
    return v


def _frame_step(t: np.ndarray, dt_s: float = 0.1) -> int:
    return max(1, int(round(dt_s / max(np.median(np.diff(t)), 1e-3))))


def main_vertical(work: Path, tracks: dict, main_cams: list[str]) -> dict:
    """Architecture vertical of the main-model cameras (SfM frame): per-camera pooled vertical vanishing points
    + robust fusion (vertical.py). Cached in cameras/cache/vertical_main.pkl, keyed by the raw localisation files."""
    from .vertical import camera_verticals, fuse
    cams = [c for c in sorted(main_cams) if c in tracks]
    up = np.concatenate([-tracks[c]["R"][tracks[c]["ok"], 1, :] for c in cams]).mean(0)
    up /= np.linalg.norm(up)
    cache = work / "cameras" / "cache"
    key = {"v": VERTICAL_VERSION, "cams": cams, "files": _file_key(*[cache / f"loc_{c}.npz" for c in cams])}

    def compute():
        per = {}
        for c in cams:
            tr = tracks[c]
            per[c] = camera_verticals(work, c, tr, up, step=_frame_step(tr["t"]))
            p = per[c]["pooled"]
            print(f"[cameras] vertical {c}: " + ("none" if p is None else
                  f"{p['n_segments']} segments in {p['n_frames']} frames, sigma {p['sigma_deg']:.2f} deg"), flush=True)
        return {"per": per, "fused": fuse(per, cams), "up_cameras": up}
    return _cached(cache / "vertical_main.pkl", key, compute)


def stage_attach(work: Path, recs: dict, F: Features, ls: LocSettings, ms: MapSettings, cams: list[str],
                 sigma_s: float, max_reproj_px: float, statures: tuple[dict, str], scale_rule: str = "chain",
                 types: dict | None = None, zoom_cams: set | None = None) -> dict:
    """Cameras missing from the main model: localise against their own sub-model, then place that sub-model in
    the main frame (attach.py). Tilt/roll come from the building's vertical (vertical vanishing points of the
    camera's own frames vs the main cameras'), yaw + centre from the dancers' keypoints triangulated by the MAIN
    cameras. Writes loc_<cam>.npz in the MAIN SfM frame + attach_<cam>.npz (per-frame dancer residual).

    zoom_cams: attached cameras the user declared as ZOOMING (attach_zoom.zoom_cams_of; None = none: today's one focal
    per sub-model). Their per-frame rotation AND focal are re-solved with the camera centre held at the sub-model's
    reference centre (attach_zoom.py), and only the rotation-only attach is used (a zooming phone slid along its
    axis in the fixed-focal localisation, which the Sim3 variant would take for real motion)."""
    import pycolmap

    from .attach import (compose, floor_in_model, holdout, observations, rotonly_errors, sim3_errors,
                         sim3_plausible, solve_rotonly, solve_sim3)
    from .vertical import angle_deg, camera_verticals
    from .world import dancer_keypoints, triangulate_dancers
    from scipy.spatial.transform import Rotation
    main = load_model(work)
    reg = model_cams(main)
    missing = [c for c in cams if c not in reg]
    info: dict = {}
    if not missing:
        return info
    if len(reg) < 2:  # the dancers the attach is fitted to need >= 2 main-model views (bootstrap would raise)
        why = few_main_reason(sorted(reg))
        for cam in missing:  # as every camera that is not attached: no placement in this model, no stale loc cache
            info[cam] = {"attached": False, "reason": why}
            print(f"[cameras] {cam}: NOT attached - {why}", flush=True)
            (work / "cameras" / "cache" / f"loc_{cam}.npz").unlink(missing_ok=True)
        return info
    images, _ = map_images(work, recs, ms)
    subs = []
    for d in sorted((work / "cameras" / "sfm_work").glob("sub_*/best")):
        if (d / "images.bin").exists():
            m = pycolmap.Reconstruction(str(d))
            subs.append((d.parent.name, m, model_cams(m)))
    main_cams = [c for c in cams if c in reg]
    main_tracks = stage_smooth(work, main_cams, sigma_s, max_reproj_px, write=False, types=types)
    require_main_tracks(reg, main_tracks, "attach")  # a subset run (--cams 05) raises here, as before, deleting nothing
    vert = main_vertical(work, main_tracks, main_cams)
    u_main = np.asarray(vert["fused"]["u"])
    dancers3d = grid = floor_main = None
    for cam in missing:
        cand = [(n, m, mc) for n, m, mc in subs if mc.get(cam, 0) >= 3]
        if not cand:
            info[cam] = {"attached": False, "reason": "no sub-model contains this camera"}
            print(f"[cameras] {cam}: no sub-model -> unregistered")
            (work / "cameras" / "cache" / f"loc_{cam}.npz").unlink(missing_ok=True)
            continue
        name, sub, mc = max(cand, key=lambda x: x[2][cam])
        frames = [r for r in recs[cam] if r.kind == "win"]
        meta = json.loads((work / "frames" / cam / "frames.json").read_text())
        w, h = int(meta["width"]), int(meta["height"])
        sub_bin = work / "cameras" / "sfm_work" / name / "best" / "images.bin"
        single = len({im.camera_id for im in sub.images.values()}) == 1
        ls_sub = sub_settings(ls, single)
        loc_key = {"sub": _file_key(sub_bin), "settings": settings_key(ls_sub),
                   "frames": [r.name for r in frames]}

        def localise():
            L = Localizer(F, sub, images, ls_sub)
            with gpu.exclusive(f"cameras-attach-{cam}"):
                return localize_camera(L, frames, w, h)
        poses = _cached(work / "cameras" / "cache" / f"locsub_{cam}.pkl", loc_key, localise)
        ok = np.array([p.ok for p in poses])
        print(f"[cameras] {cam}: {ok.sum()}/{len(frames)} frames localised in sub-model {name} {mc}", flush=True)
        zoom_rep = None
        if cam in (zoom_cams or ()) and ok.sum() >= 10:
            from .attach_zoom import reference_centre, resolve_zoom
            poses, zoom_rep = resolve_zoom(poses, reference_centre(sub), w, h, t=np.array([r.t for r in frames]))
            ok = np.array([p.ok for p in poses])
            print(f"[cameras] {cam}: per-frame focal (zooming phone): {zoom_rep}", flush=True)
        if ok.sum() < 10:
            info[cam] = {"attached": False, "reason": f"only {int(ok.sum())} frames localised in sub-model {name}"}
            (work / "cameras" / "cache" / f"loc_{cam}.npz").unlink(missing_ok=True)
            continue
        if dancers3d is None:  # dancers from the MAIN cameras only (attached cameras never feed each other)
            dancers3d, grid = triangulate_dancers(work, main_tracks, kps=tuple(range(17)), return_grid=True)
            pts_main = np.array([p.xyz for p in main.points3D.values()]).reshape(-1, 3)
            pre = bootstrap(main_tracks, triangulate_dancers(work, main_tracks), pts_main, *statures,
                            up_cams=main_cams, vertical=vert["fused"], scale_rule=scale_rule)
            Rw = np.asarray(pre["R"])
            floor_main = (Rw[1], -float(pre["t"][1]) / pre["s"])  # world y = 0  <=>  R[1].X = -t_y / s
            s_world = float(pre["s"])
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
        # the building's vertical in the SUB-model frame (this camera's own frames)
        vkey = {"v": VERTICAL_VERSION, "loc": loc_key}
        vs = _cached(work / "cameras" / "cache" / f"vertical_sub_{cam}.pkl", vkey,
                     lambda: camera_verticals(work, cam, {"K": K, "k1": k1, "R": R_sub, "ok": ok}, up_s,
                                              step=_frame_step(t)))
        u_sub = vs["pooled"]["u"] if vs["pooled"] is not None and vs["pooled"]["n_segments"] >= 30 else None
        kp2d = dancer_keypoints(work, cam, len(frames))
        fidx, X, uv, wt = observations(kp2d, t, ok, dancers3d, grid)
        if len(fidx) < 50:
            info[cam] = {"attached": False, "reason": f"only {len(fidx)} dancer keypoint observations"}
            (work / "cameras" / "cache" / f"loc_{cam}.npz").unlink(missing_ok=True)
            continue
        vconstr = (u_sub, u_main) if u_sub is not None else None
        rot_err = lambda sol, sel: rotonly_errors(sol, R_sub, K, k1, fidx[sel], X[sel], uv[sel])  # noqa: E731
        sols, ho = {}, {}
        if vconstr is not None:
            sols["rotation-only+vertical"] = solve_rotonly(R_sub, K, k1, fidx, X, uv, wt, vertical=vconstr)
            ho["rotation-only+vertical"] = holdout(
                lambda sel: solve_rotonly(R_sub, K, k1, fidx[sel], X[sel], uv[sel], wt[sel], vertical=vconstr),
                rot_err, fidx, t)
        sols["rotation-only"] = solve_rotonly(R_sub, K, k1, fidx, X, uv, wt, up_main=floor_main[0])
        ho["rotation-only"] = holdout(
            lambda sel: solve_rotonly(R_sub, K, k1, fidx[sel], X[sel], uv[sel], wt[sel], up_main=floor_main[0]),
            rot_err, fidx, t)
        floor_ok = bool(fl is not None and np.degrees(np.arccos(min(1.0, abs(float(fl[0] @ up_s))))) < 10)             and zoom_rep is None
        rejected = {}
        if floor_ok:
            sol = solve_sim3(R_sub, t_sub, K, k1, fidx, X, uv, wt, floor_sub=floor_sub, floor_main=floor_main)
            okp, why = sim3_plausible(sol, R_sub, t_sub, floor_sub, floor_main, s_world)
            if okp:
                sols["sim3+floor"] = sol
                ho["sim3+floor"] = holdout(
                    lambda sel: solve_sim3(R_sub, t_sub, K, k1, fidx[sel], X[sel], uv[sel], wt[sel],
                                           floor_sub=floor_sub, floor_main=floor_main),
                    lambda s_, sel: sim3_errors(s_, R_sub, t_sub, K, k1, fidx[sel], X[sel], uv[sel]), fidx, t)
            else:
                rejected["sim3+floor"] = {"reason": why, "s": sol["s"], "median_px": sol["median_px"],
                                          "floor_normal_err_deg": sol.get("floor_normal_err_deg")}
        for k, sol in sols.items():
            sol["vertical_err_deg"] = (angle_deg(sol["Rs"] @ u_sub, u_main) if u_sub is not None else None)
            sol["holdout"] = ho[k]
        # choice: candidates whose tilt agrees with the building (when measurable), then the lowest HOLD-OUT error
        pool = [k for k in sols if u_sub is None or sols[k]["vertical_err_deg"] <= 1.5] or list(sols)
        kind = min(pool, key=lambda k: sols[k]["holdout"].get("median_px", np.inf))
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
        e = (rotonly_errors(sol, R_sub, K, k1, fidx, X, uv) if kind != "sim3+floor"
             else sim3_errors(sol, R_sub, t_sub, K, k1, fidx, X, uv))
        per_frame = np.full(len(frames), np.nan)
        for i in np.unique(fidx):
            per_frame[i] = np.median(e[fidx == i])
        np.savez(work / "cameras" / "cache" / f"attach_{cam}.npz", dancer_px=per_frame, method=kind)
        u_sub_main = sol["Rs"] @ u_sub if u_sub is not None else None
        info[cam] = {"attached": True, "method": kind, "sub_model": name, "sub_model_cams": mc,
                     "sub_model_reproj_px": float(sub.compute_mean_reprojection_error()),
                     "sub_floor": {"found": fl is not None, "inliers": int(fl[2]) if fl else 0, "usable": floor_ok},
                     "vertical_sub": None if vs["pooled"] is None else
                     {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in vs["pooled"].items()},
                     "vertical_vs_main_deg": (angle_deg(u_sub_main, u_main) if u_sub_main is not None else None),
                     "candidates": {k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()
                                        if kk not in ("Rs", "T", "C")} for k, v in sols.items()},
                     "rejected": rejected,
                     "dancer_fit_median_px": sol["median_px"], "dancer_fit_p90_px": sol["p90_px"],
                     "dancer_holdout": sol["holdout"],
                     "dancer_obs": sol["n_obs"], "frames_used": sol["frames_used"],
                     "camera_centre_sfm": (sol["C"].tolist() if kind != "sim3+floor" else None)}
        if zoom_rep is not None:
            info[cam]["zoom"] = zoom_rep
        print(f"[cameras] {cam}: attached ({kind}); " + "; ".join(
            f"{k}: fit {v['median_px']:.1f}px, hold-out {v['holdout'].get('median_px', float('nan')):.1f}px, "
            f"vertical err {v['vertical_err_deg'] if v['vertical_err_deg'] is None else round(v['vertical_err_deg'], 2)}"
            for k, v in sols.items()) + (f"; rejected {list(rejected)}" if rejected else ""), flush=True)
    return info


def _save_raw(p: Path, frames: list[ImageRec], poses: list[FramePose], meta: dict | None = None) -> None:
    """Raw localisation -> cameras/cache/loc_<cam>.npz; meta: typed.loc_marker (how the camera was localised)."""
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
             corr_xyz=np.concatenate(xyz) if xyz else np.zeros((0, 3)), **(meta or {}))


def _policy_name(policy: str | None) -> str:
    from .. import camera_types as ct
    try:
        return ct.resolve_policy(policy)
    except ct.CameraTypeError:
        return "user"


class KeyframeCacheError(RuntimeError):
    """A locked tripod's keyframes-only localisation cannot carry the model the camera now needs."""

    def __init__(self, cams: dict[str, str], take: str | None = None):
        self.cams = cams
        super().__init__("; ".join(f"{c}: {why}" for c, why in cams.items()) + " - re-localise every frame: "
                         f"python -m dancecap cameras {take or '<take>'} --stages loc --cams {','.join(sorted(cams))} "
                         f"(Runpod: the front session's loc:<cam> jobs)")


def keyframe_mismatch(work: Path, cams: list[str], types: dict) -> dict[str, str]:
    """{cam: why} of the cameras whose loc_<cam>.npz holds a locked tripod's keyframes only although the camera's
    type (under the current policy) is no longer a locked tripod: that cache must be re-localised on every frame."""
    out = {}
    for cam in cams:
        p = work / "cameras" / "cache" / f"loc_{cam}.npz"
        if not p.exists():
            continue
        with np.load(p) as z:
            if loc_mode(z) != "keyframes":
                continue
            was = str(z["loc_class"]) if "loc_class" in z.files else "tripod"
            n = int(z["loc_localised"]) if "loc_localised" in z.files else -1
        st = types.get(cam)
        if not locked_tripod(st):
            out[cam] = (f"cameras/cache/loc_{cam}.npz holds keyframes only ({n} frames, localised as {was}) but the "
                        f"camera is now {st.cls if st is not None else 'handheld_zoom'}")
    return out


def localisation_record(work: Path, cams: list[str]) -> dict:
    """{cam: {mode, class, policy, localised, frames}} from the loc_<cam>.npz markers (report.json camera_types)."""
    out = {}
    for cam in cams:
        p = work / "cameras" / "cache" / f"loc_{cam}.npz"
        if not p.exists():
            continue
        try:
            with np.load(p) as z:
                out[cam] = {"mode": loc_mode(z), "frames": int(len(z["t"])),
                            "class": str(z["loc_class"]) if "loc_class" in z.files else None,
                            "policy": str(z["loc_policy"]) if "loc_policy" in z.files else None,
                            "localised": int(z["loc_localised"]) if "loc_localised" in z.files else None}
        except (OSError, ValueError, KeyError):
            continue
    return out


def reproj_cv(R, t, K, k1, xyz, xy) -> np.ndarray:
    uv, vis = qa.project_cv(xyz, K, R, t, k1)
    return np.where(vis, np.linalg.norm(uv - xy, axis=1), np.inf)


def adopted_track(work: Path, cam: str) -> Path | None:
    """cameras/orig/<cam>_loc.npz when cameras/<cam>.npz is an ADOPTED track (`cameras.twoview ptz --adopt` moves the
    localisation there and installs the PTZ track): stage_smooth then keeps cameras/<cam>.npz instead of rewriting it
    from cache/loc_<cam>.npz or deleting it. Moving cameras/orig/<cam>_loc.npz back undoes the adoption."""
    o = work / "cameras" / "orig" / f"{cam}_loc.npz"
    return o if o.exists() and (work / "cameras" / f"{cam}.npz").exists() else None


def stage_smooth(work: Path, cams: list[str], sigma_s: float, max_reproj_px: float,
                 write: bool = True, attach_bad_px: float = 60.0, types: dict | None = None,
                 typed_settings: TypedSettings | None = None) -> dict[str, dict]:
    """Raw localisation -> smoothed tracks -> cameras/<cam>.npz (SfM frame). Returns track dicts.

    types: {cam: camera_types.StageType}; None = read the take TOML (camera_types_for), {} = none (today's model).
    A camera with a non-general type gets its model applied first (typed.py: one focal per video, one constant pose
    for a tripod, ...) and skips the zoom-depth guard; the general class runs exactly today's code.
    A locked tripod's keyframes-only localisation (loc_mode 'keyframes') is valid only for the constant pose: with any
    other model the camera is not exported (write=True raises KeyframeCacheError after the other cameras; write=False
    returns the track marked keyframes_only). An adopted track (adopted_track) is never overwritten or deleted; the
    returned track is still the localisation's."""
    tracks = {}
    types = camera_types_for(work, types)
    sparse_bad: dict[str, str] = {}
    for cam in cams:
        p = work / "cameras" / "cache" / f"loc_{cam}.npz"
        kept = adopted_track(work, cam) if write else None
        if kept is not None:
            print(f"[cameras] {cam}: cameras/{cam}.npz is an adopted track (cameras/orig/{kept.name} holds the "
                  f"localisation it replaced): kept, not rewritten - move that file back to re-export", flush=True)
        if not p.exists():
            if write and kept is None:  # no track for this camera any more: never leave a stale contract file behind
                (work / "cameras" / f"{cam}.npz").unlink(missing_ok=True)
            continue
        d = dict(np.load(p))
        meta = json.loads((work / "frames" / cam / "frames.json").read_text())
        w, h = int(meta["width"]), int(meta["height"])
        st = types.get(cam)
        ctype = None
        if st is not None and not st.general:  # camera type (dancecap.camera_types): constrained model first
            d, ctype = type_refit(d, w, h, st, typed_settings, log=print if write else None)
        sparse = loc_mode(d) == "keyframes"
        if sparse and (ctype is None or ctype["model"] != "constant"):
            why = (f"cameras/cache/loc_{cam}.npz holds a locked tripod's keyframes only but the camera's model is "
                   + (f"{ctype['model']} ({ctype['class']}: {(ctype.get('check') or {}).get('why') or 'refused'})"
                      if ctype is not None else f"the general per-frame model ({st.cls if st else 'handheld_zoom'})"))
            if write:
                print(f"[cameras] {cam}: NOT exported - {why}", flush=True)
                sparse_bad[cam] = why
                continue
        if ctype is not None and ctype["model"] != "per_frame":
            # a fixed focal or a fixed centre leaves no focal / depth-along-the-axis ambiguity to guard against
            d, dg = depth_guard(d, w, h, DepthGuardSettings(enabled=False))
            dg["skipped"] = f"camera type {ctype['class']}: model {ctype['model']}"
        else:
            # zoom-depth guard: frames whose centre is ill-determined along the optical axis are re-centred on an
            # inverse-variance smooth of the centres, rotation + focal re-solved (depth_guard.py); attached cams skipped
            d, dg = depth_guard(d, w, h)
        held = np.asarray(d.get("type_held", np.zeros(len(d["t"]), bool)), bool)
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
        if held.any():  # a constant (tripod) pose is valid on frames without static points of their own
            ok |= ok_raw & held & ~np.isfinite(rp)
        attached = bool(np.any(d["source"] == "attach"))
        if attached:  # reproj_px of an attached camera is its SUB-MODEL consistency; the main-frame check is the
            # dancers' residual (attach_<cam>.npz): frames far off the dancers are not trusted
            ap = work / "cameras" / "cache" / f"attach_{cam}.npz"
            if ap.exists():
                dpx = np.load(ap)["dancer_px"]
                lim = max(attach_bad_px, 3.0 * float(np.nanmedian(dpx)))
                ok &= ~(np.nan_to_num(dpx, nan=0.0) > lim)
        n_in = np.where(ok, d["n_inliers"], 0).astype(np.int32)
        typed_keys = {} if ctype is None else {  # additive contract keys (CONTRACTS.md cameras/<cam>.npz)
            "camera_class": np.str_(ctype["class"]), "type_model": np.str_(ctype["model"]), "type_held": held}
        if write and kept is None:
            np.savez(work / "cameras" / f"{cam}.npz", t=d["t"], K=K, R=R_out, tvec=t_out, dist=k1, ok=ok,
                     reproj_px=rp, n_inliers=n_in, attached=np.bool_(attached),
                     axis_sd=d["axis_sd"].astype(np.float32), recentred=d["recentred"], **typed_keys)
        C_out = -np.einsum("fji,fj->fi", R_out, t_out)
        tracks[cam] = {"t": d["t"], "K": K, "k1": k1, "R": R_out, "tvec": t_out, "ok": ok, "C": C_out,
                       "f": f_out, "f_raw": d["f"], "raw_used": int(raw_used.sum()), "ok_raw": ok_raw, "reproj_px": rp, "reproj_raw": d["reproj_px"],
                       "n_inliers": d["n_inliers"], "source": d["source"], "w": w, "h": h, "maxdim": max(w, h),
                       "attached": attached,
                       "outliers": int(sm["outlier"].sum()), "depth_guard": dg}
        if ctype is not None:
            tracks[cam]["camera_type"] = ctype
        if sparse:
            tracks[cam]["keyframes_only"] = ctype is None or ctype["model"] != "constant"
    if sparse_bad:
        raise KeyframeCacheError(sparse_bad, Path(work).name)
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
        max_reproj_px: float = 4.0, camera_types: str | None = None) -> dict:
    """camera_types: policy for the per-video camera types (user | auto | off; None = DANCECAP_CAMERA_TYPES or
    'user': confirmed answers only). See dancecap.camera_types and cameras/typed.py.
    The world's dancers are triangulated on a two-view sync's clocks when sync/offsets.json is one (world_sync_tracks;
    run `cameras.twoview sync` before --stages world). A main model of < 2 cameras skips attach / world with the reason
    in report.json (few_main_reason)."""
    work = take.work.resolve()
    cams = cams or [s.id for s in take.sources]
    out = work / "cameras"
    out.mkdir(parents=True, exist_ok=True)
    recs = list_images(work, cams)
    fs, ms, ls = FeatureSettings(), map_settings(take.cfg), loc_settings(take)
    F = Features(work, out / "cache", fs)
    rep_p = out / "report.json"
    rep = json.loads(rep_p.read_text()) if rep_p.exists() else {}
    types = camera_types_for(work, None, camera_types)
    policy = _policy_name(camera_types)
    rep["camera_types"] = {"policy": policy, "applied": {c: t.as_dict() for c, t in sorted(types.items())}}
    if types:
        print("[cameras] camera types: " + ", ".join(f"{c} {t.cls} ({t.source})" for c, t in sorted(types.items()))
              + f"; every other camera: handheld + zoom (general); policy {policy}")
    if "loc" not in stages:  # a locked tripod's keyframes-only cache cannot carry another model: refuse early
        bad = keyframe_mismatch(work, cams, types)
        if bad:
            raise KeyframeCacheError(bad, take.name)
    if "map" in stages:
        rep["map"] = stage_map(work, recs, F, ms)
    if "loc" in stages:
        stage_loc(work, recs, F, ls, ms, cams, types=types, policy=policy)
    if "attach" in stages:
        from .attach_zoom import zoom_cams_of
        rep["attach"] = stage_attach(work, recs, F, ls, ms, cams, sigma_s, max_reproj_px,
                                     statures_from_take(take.cfg), scale_rule_from_take(take.cfg), types=types,
                                     zoom_cams=zoom_cams_of(work, camera_types))
    tracks = stage_smooth(work, cams, sigma_s, max_reproj_px, types=types)
    model = load_model(work)
    pts = np.array([p.xyz for p in model.points3D.values()]).reshape(-1, 3)
    rgb = np.array([p.color for p in model.points3D.values()]).reshape(-1, 3)
    sim3 = None
    sim3_p = out / "world_from_sfm.json"
    dancers = None
    reg_main = model_cams(model)
    main_cams = sorted(c for c in reg_main if c in tracks)  # world axes from the main model only
    skip_rec = (rep.get("world") or {}).get("solved") is False  # report.json holds an earlier run's skip record
    world_skipped = "world" in stages and len(reg_main) < 2  # with or without this run's tracks: no world frame
    if world_skipped:  # no world frame instead of a crash in bootstrap; the reason goes to report.json
        moved = None
        if sim3_p.exists():  # an earlier run's world frame (maybe of another map): out of every reader's way
            sim3_p.replace(out / "world_from_sfm.prev.json")
            moved = "an earlier run's world_from_sfm.json, moved to world_from_sfm.prev.json"
        rep["world"] = {"solved": False, "reason": few_main_reason(sorted(reg_main)), "world_from_sfm": moved}
        print(f"[cameras] WARNING world NOT solved - {rep['world']['reason']}" + (f" ({moved})" if moved else ""),
              flush=True)
    elif "world" in stages and tracks:
        require_main_tracks(reg_main, tracks, "world")
        if skip_rec:
            rep.pop("world")
        main_tr = {c: tracks[c] for c in main_cams}
        tri_tr, world_sync = world_sync_tracks(work, main_tr)  # a two-view sync's clocks, else the audio clocks
        if world_sync is not None:
            print(f"[cameras] world dancers on the two-view sync clocks: {world_sync['offsets_ms']} ms", flush=True)
        dancers = triangulate_dancers(work, tri_tr)  # attached cameras were fitted TO these: never feed them back
        st, src = statures_from_take(take.cfg)
        vert = main_vertical(work, main_tr, main_cams)
        soles, sole_src = sole_from_take(take.cfg)
        x_cam, x_why = world_x_cam(take.cfg, main_cams, types)
        print(f"[cameras] world X axis: camera {x_cam} ({x_why})", flush=True)
        sim3 = bootstrap(main_tr, dancers, pts, st, src, x_cam=x_cam,
                         up_cams=main_cams, vertical=vert["fused"], scale_rule=scale_rule_from_take(take.cfg),
                         sole_m=soles)  # per dancer (chain rule); the floor uses their mean
        sim3["diagnostics"]["sole_source"] = sole_src
        sim3["diagnostics"]["dancer_roles"] = roles_coverage(work, main_cams)
        if world_sync is not None:  # only then: every other take's world_from_sfm.json is unchanged
            sim3["diagnostics"]["world_sync"] = world_sync
        ref = reference_from_take(take.cfg)  # [cameras] scale_reference: inherit a betas-corrected scale
        if ref is not None:
            sim3, _ = adopt_reference(work, sim3, ref["work"], ref["sim3"], soles, ref_name=ref["take"],
                                      statures=st)  # refuses a reference built with other statures / inputs
            sim3["diagnostics"]["scale_reference"]["sim3_path"] = ref["sim3_path"]
            sr = sim3["diagnostics"]["scale_reference"]
            if not sr["adopted"]:
                print(f"[cameras] WARNING scale reference {ref['take']} NOT adopted: {sr['why']}")
            elif sr.get("warning"):
                print(f"[cameras] WARNING scale reference: {sr['warning']}")
        sim3["diagnostics"]["room_metric_check"] = room_check(work, main_tr, pts, vert, sim3)
        for g, v in scale_checks(sim3).items():
            if not v["passed"]:
                bad = {k: c.get("value") for k, c in v["checks"].items() if not c["passed"]}
                print(f"[cameras] WARNING {g} FAILED: {json.dumps(bad, default=_np)[:600]}")
        if sim3_p.exists():  # keep the previous world frame for comparison (world_from_sfm.prev.json)
            shutil.copyfile(sim3_p, out / "world_from_sfm.prev.json")
        sim3_p.write_text(json.dumps(sim3, indent=1, default=_np))
        np.savez(out / "cache" / "dancers_sfm.npz",
                 **{f"{r}_{k}": v for r, d in dancers.items() for k, v in d.items()})
        print(f"[cameras] world_from_sfm: s={sim3['s']:.4f} ({sim3['scale_source']}; absolute +-1.5 % UNVERIFIED: "
              f"statures barefoot?); {sim3['gravity_source']}; {sim3['floor_source']}")
    elif sim3_p.exists():
        sim3 = json.loads(sim3_p.read_text())
        if skip_rec:  # a world file is back (restored by hand) and used: the skip record is stale
            rep.pop("world")
    if "qa" in stages:
        rep.update(build_report(work, cams, recs, model, tracks, pts, rgb, sim3, rep.get("map", {}), sigma_s,
                                max_reproj_px, rep.get("attach"), n_take=len(take.sources)))
        camera_type_probe(work, tracks, sim3, sigma_s, max_reproj_px, rep)
        rep["camera_types"]["localised"] = localisation_record(work, cams)
        rep_p.write_text(json.dumps(rep, indent=1, default=_np))
    elif world_skipped or ("attach" in stages and rep.get("attach") and len(reg_main) < 2):
        rep_p.write_text(json.dumps(rep, indent=1, default=_np))  # without qa the skip reason still reaches report.json
    return rep


def camera_type_probe(work: Path, tracks: dict, sim3: dict | None, sigma_s: float, max_reproj_px: float,
                      rep: dict) -> None:
    """Probe B of the camera types (dancecap.camera_types) on today's GENERAL per-frame tracks -> probe/
    camera_types_probe_b.json (the suggestion after calibration, and the evidence of a conflict with the user's type);
    cams.<cam>.camera_type = the model applied + its residual check. Never fails the stage."""
    try:
        from .. import camera_types as ct
        typed = [c for c, tr in tracks.items() if tr.get("camera_type")]
        general = {c: tr for c, tr in tracks.items() if c not in typed}
        if typed:  # the typed export would be circular evidence: re-derive today's model for these cameras
            general.update(stage_smooth(work, typed, sigma_s, max_reproj_px, write=False, types={}))
        res = ct.probe_b(work.name, sorted(general), write=work.parent.resolve() == (ct.root() / "work").resolve(),
                         tracks=general, sim3=sim3, log=print)
        for c, tr in tracks.items():
            e = rep.setdefault("cams", {}).setdefault(c, {})
            if tr.get("camera_type"):
                e["camera_type"] = tr["camera_type"]
            if c in res:
                e["camera_type_probe_b"] = res[c]
        _progress_camera_motion(res, sim3)
    except Exception as e:  # noqa: BLE001 - a probe never fails the cameras stage
        print(f"[cameras] camera-type probe B skipped: {type(e).__name__}: {e}")


def _progress_camera_motion(res: dict, sim3: dict | None) -> None:
    """dancecap.progress/1 QA metric `camera_motion` per camera (no-op unless DANCECAP_PROGRESS): probe B's measurement
    of today's UNCONSTRAINED per-frame track + its suggestion, so the monitor can compare it with the declared type
    while the job runs (the type-constrained export would read 0 cm / x1.00 by construction). Never raises."""
    try:
        from .. import progress as _progress
        if not _progress.enabled():
            return
        for cam, s in sorted(res.items()):
            ev = s.get("evidence") or {}
            _progress.metric("camera_motion", step="cameras", lane=cam, source="probe_b", constrained=False,
                             metric_scale=sim3 is not None, at=s.get("at"),
                             **{k: ev.get(k) for k in ("centre_p95_m", "axis_p95_deg", "angvel_p50_dps", "hf_rot_deg",
                                                       "hf_centre_mm", "focal_ratio", "focal_step_pct", "frames",
                                                       "mount_judgeable", "zoom_judgeable", "attached")},
                             **{k: s.get(k) for k in ("mount", "zoom", "pans", "confidence", "mount_confidence",
                                                      "zoom_confidence")})
    except Exception:  # noqa: BLE001 - progress never fails a stage
        pass


def _np(o):
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def dancer_check(work: Path, tracks: dict, cams: list[str], qa_dir: Path | None = None,
                 main_cams: list[str] | None = None, attach: dict | None = None) -> dict:
    """Check of every camera against the dancers: COCO keypoints triangulated from the OTHER MAIN-MODEL cameras
    (leave-one-out; attached cameras never vote, because they were fitted to the main cameras' dancers) and
    reprojected into this camera's ok frames. Static-feature errors cannot show a camera that registered at a
    consistent but wrong place; this does (expect ~5-15 px from keypoint noise + sync).

    Main-model cameras: non-circular (their poses come from static features only). Attached cameras: the same
    number is the attach FIT residual (circular); the validation is the attach's temporal hold-out
    (attach.<cam>.dancer_holdout), copied here as `holdout`. Per-frame medians -> `per_frame` (F,), NaN without
    observations."""
    from .attach import observations
    from .world import dancer_keypoints
    main_cams = sorted(tracks) if main_cams is None else main_cams
    attach = attach or {}
    out = {}
    for cam in cams:
        others = {c: v for c, v in tracks.items() if c != cam and c in main_cams}
        if cam not in tracks or len(others) < 2:
            continue
        d3, grid = triangulate_dancers(work, others, kps=tuple(range(5, 17)), return_grid=True)
        tr = tracks[cam]
        kp2d = dancer_keypoints(work, cam, len(tr["t"]))
        fidx, X, uv, _ = observations(kp2d, tr["t"], tr["ok"], d3, grid)
        if len(fidx) < 20:
            out[cam] = {"n": int(len(fidx))}
            continue
        e, proj = [], {}
        pf = np.full(len(tr["t"]), np.nan)
        for f in np.unique(fidx):
            sel = fidx == f
            p, vis = qa.project_cv(X[sel], tr["K"][f], tr["R"][f], tr["tvec"][f], tr["k1"][f])
            e.append(np.where(vis, np.linalg.norm(p - uv[sel], axis=1), 1e4))
            pf[f] = float(np.median(e[-1]))
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
        circ = cam not in main_cams
        out[cam] = {"n": int(len(e)), "median_px": round(float(np.median(e)), 2),
                    "p90_px": round(float(np.percentile(e, 90)), 2),
                    "median_rel_height": round(float(np.median(e)) / tr["maxdim"], 4),
                    "triangulated_from": sorted(others), "circular": circ,
                    "kind": "attach fit residual (circular)" if circ else "leave-one-out (non-circular)",
                    "per_frame": pf, "_proj": proj}
        if circ and attach.get(cam, {}).get("dancer_holdout"):
            out[cam]["holdout"] = attach[cam]["dancer_holdout"]
    return out



def room_check(work: Path, main_tracks: dict, pts: np.ndarray, vert: dict, sim3: dict) -> dict:
    """Is the reconstruction metric around the room, and which plane disagrees with gravity?

    The building's dominant horizontal edge directions (vanishing points of non-vertical segments, pooled over
    the main cameras in the SfM frame) and its largest wall planes (sparse points) are compared with the adopted
    gravity (world Y), the dancers' ankle plane and the sparse floor near the dancers. Under any affine distortion
    horizontal edges stay parallel to the floor; edges perpendicular to world Y but not to the floor normal mean
    the FLOOR PATCH is mis-reconstructed, not the room."""
    from .vertical import angle_deg, horizontal_directions, pooled_normals, wall_planes
    Rw = np.asarray(sim3["R"])
    y = Rw[1]
    cams = sorted(main_tracks)
    cache = work / "cameras" / "cache"
    key = {"v": VERTICAL_VERSION, "files": _file_key(*[cache / f"loc_{c}.npz" for c in cams])}

    def compute():
        N, W = [], []
        for c in cams:
            tr = main_tracks[c]
            n, w = pooled_normals(work, c, tr, list(range(0, len(tr["t"]), 4 * _frame_step(tr["t"]))), y)
            N.append(n), W.append(w)
        return np.concatenate(N), np.concatenate(W)
    N, W = _cached(cache / "horizontal_main.pkl", key, compute)
    hd = horizontal_directions(N, W, y, n_dirs=3)
    diag = sim3["diagnostics"]
    others = {"ankle_plane": diag.get("ankle_plane_normal_sfm"), "sfm_floor_plane": diag.get("sfm_floor_normal_sfm")}
    out = {"horizontal_edges": [], "walls": []}
    for h in hd:
        e = {"segments": h["n_segments"], "elevation_vs_world_y_deg": round(h["elev_vs_u_deg"], 3),
             "dir_world": (Rw @ h["d"]).round(4).tolist()}
        for name, nv in others.items():
            if nv is not None:
                e[f"elevation_vs_{name}_deg"] = round(abs(90 - angle_deg(h["d"], np.asarray(nv))), 3)
        out["horizontal_edges"].append(e)
    if len(hd) >= 2:
        out["angle_between_first_two_deg"] = round(angle_deg(hd[0]["d"], hd[1]["d"]), 2)
    for wpl in wall_planes(pts, y, float(sim3["s"]), n_planes=4):
        out["walls"].append({"inliers": wpl["inliers"], "tilt_vs_world_y_deg": round(wpl["tilt_from_vertical_deg"], 3),
                             "normal_world": (Rw @ wpl["n"]).round(4).tolist(),
                             **{f"tilt_vs_{name}_deg": round(abs(90 - angle_deg(wpl["n"], np.asarray(nv))), 3)
                                for name, nv in others.items() if nv is not None}})
    return out


def vertical_check(work: Path, tracks: dict, sim3: dict | None, main_cams: list[str]) -> dict:
    """Per camera (main AND attached, exported poses): the building's vertical from the camera's own frames vs the
    world Y axis (gate: <= 1 deg for main-model cameras, ~1.5 deg for attached ones)."""
    from .vertical import angle_deg, camera_verticals, fuse
    if sim3 is None:
        return {}
    y = np.asarray(sim3["R"])[1]
    cache = work / "cameras" / "cache"
    out = {}
    for c in sorted(tracks):
        tr = tracks[c]
        key = {"v": VERTICAL_VERSION, "files": _file_key(cache / f"loc_{c}.npz", work / "cameras" / f"{c}.npz")}
        d = None
        if c in main_cams and (cache / "vertical_main.pkl").exists():  # same frames as the fusion: reuse
            import pickle
            vm = pickle.loads((cache / "vertical_main.pkl").read_bytes()).get("value") or {}
            d = vm.get("per", {}).get(c)
        if d is None:
            d = _cached(cache / f"vertical_check_{c}.pkl", key,
                        lambda: camera_verticals(work, c, tr, y, step=_frame_step(tr["t"])))
        if d["pooled"] is None:
            out[c] = {"measured": False}
            continue
        f = fuse({c: d}, [c])
        # split into roll (in the image plane: precise) and pitch (towards the optical axis: weak for a narrow FOV)
        Rm = tr["R"][int(np.nonzero(tr["ok"])[0][tr["ok"].sum() // 2])] if tr["ok"].any() else tr["R"][0]
        v, p = Rm @ d["pooled"]["u"], Rm @ y
        roll = float(np.degrees(np.arctan2(v[0], -v[1]) - np.arctan2(p[0], -p[1])))
        pitch = float(np.degrees(np.arctan2(v[2], -v[1]) - np.arctan2(p[2], -p[1])))
        sg = max(d["pooled"]["sigma_deg"], np.nan_to_num(d["pooled"].get("sigma_frames_deg", np.nan), nan=0.0))
        ang = angle_deg(d["pooled"]["u"], y)
        out[c] = {"measured": True, "vs_world_y_deg": round(ang, 3), "roll_deg": round(roll, 3),
                  "pitch_deg": round(pitch, 3), "sigma_deg": round(float(sg), 3),
                  "within_2sigma_of_1deg": bool(ang <= 1.0 + 2 * sg),
                  "segments": d["pooled"]["n_segments"], "frames": d["pooled"]["n_frames"], "main_model": c in main_cams,
                  "per_frame_spread_deg": f["cams"][c].get("per_frame", {}).get("spread_deg_median"),
                  "f_over_maxdim": round(float(np.median(tr["f"]) / tr["maxdim"]), 3)}
    return out


def vertical_review(work: Path, tracks: dict, sim3: dict, sim3_old: dict | None, path: Path,
                    cams: list[str] | None = None, check: dict | None = None, height: int = 900,
                    per_row: int = 4) -> Path:
    """review/cameras_vertical_check.jpg: per camera one frame (the one with the most usable vertical edges among
    a few) with the wall/door/window/pillar segments used for the vertical (yellow) and plumb lines through the
    longest of them: NEW world up (green) and the previous world up (red, if any)."""
    import cv2

    from .vertical import angle_deg, draw_check, frame_vertical
    y = np.asarray(sim3["R"])[1]
    y_old = np.asarray(sim3_old["R"])[1] if sim3_old is not None else None
    check = check or {}
    tiles = []
    for c in cams or sorted(tracks):
        tr = tracks[c]
        meta = json.loads((work / "frames" / c / "frames.json").read_text())
        okf = np.nonzero(tr["ok"])[0]
        if not len(okf):
            continue
        best = None
        for q in (0.2, 0.4, 0.6, 0.8):
            i = int(okf[int(q * (len(okf) - 1))])
            img = cv2.imread(str(work / "frames" / c / meta["frames"][i]["file"]))
            mp = work / "people_masks" / c / (Path(meta["frames"][i]["file"]).stem + ".png")
            mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) if mp.exists() else None
            res = frame_vertical(img, mask, tr["K"][i], float(tr["k1"][i]), tr["R"][i] @ y, return_segments=True)
            n = 0 if res is None else int(res["seg_used"].sum())
            if best is None or n > best[0]:
                best = (n, i, img, mask, res)
        n, i, img, mask, res = best
        R, K, k1 = tr["R"][i], tr["K"][i], float(tr["k1"][i])
        lines = {"new": (R @ y, (0, 230, 0))}
        if y_old is not None:
            lines["old"] = (R @ y_old, (40, 40, 255))
        seg, used = (res["seg"], res["seg_used"]) if res is not None else (np.zeros((0, 4)), np.zeros(0, bool))
        im = draw_check(img, seg, used, K, k1, lines, mask)
        im = cv2.resize(im, (int(round(im.shape[1] * height / im.shape[0])), height), interpolation=cv2.INTER_AREA)
        how = "attached" if tr.get("attached") else "main model"
        vc = check.get(c, {})
        txt = [(f"cam {c} ({how}) f{i}", (255, 255, 255))]
        if vc.get("vs_world_y_deg") is not None:
            txt.append((f"building vs new up: {vc['vs_world_y_deg']:.2f} deg", (0, 230, 0)))
        if y_old is not None:
            txt.append((f"old up: {angle_deg(y, y_old):.1f} deg off", (80, 80, 255)))
        if res is not None:
            txt.append((f"this frame: {n} edges, {angle_deg(res['v'], R @ y):.2f} deg", (0, 230, 255)))
        for k, (tx, col) in enumerate(txt):
            org = (8, 30 + 28 * k)
            cv2.putText(im, tx, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(im, tx, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)
        tiles.append(im)
    rows = [tiles[k:k + per_row] for k in range(0, len(tiles), per_row)]
    W = max(sum(t.shape[1] for t in r) + 6 * (len(r) - 1) for r in rows)
    out_rows = []
    for r in rows:
        parts = []
        for t in r:
            parts += [t, np.full((height, 6, 3), 255, np.uint8)]
        row = np.hstack(parts[:-1])
        if row.shape[1] < W:
            row = np.hstack([row, np.zeros((height, W - row.shape[1], 3), np.uint8)])
        out_rows += [row, np.full((6, W, 3), 255, np.uint8)]
    sheet = np.vstack(out_rows[:-1])
    banner = np.full((80, W, 3), 30, np.uint8)
    for k, tx in enumerate(["Gravity check - yellow: vertical edges used (walls, doors, windows, pillars; people darkened).",
                            "Green: NEW world up (building vertical). Red: OLD world up (dancers' ankle plane). "
                            "A correct up runs along the yellow edges."]):
        cv2.putText(banner, tx, (10, 30 + 34 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.vstack([banner, sheet]), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return path


def attach_review(work: Path, tracks: dict, sim3: dict, projs: dict, cams: list[str], path: Path, dl: dict,
                  n_frames: int = 3, height: int = 640) -> Path:
    """review/cameras_attach_check.jpg: attached cameras (and one main-model camera for comparison), a few frames
    each: the dancers' keypoints detected in this camera (green crosses) vs the same keypoints triangulated by the
    OTHER main-model cameras and projected with this camera's pose (magenta circles, line = residual); 1 m floor
    grid of the world frame (cyan, orange = x/z axes); world up plumb lines (green)."""
    import cv2

    from .vertical import _plumb
    s, Rw, tw = float(sim3["s"]), np.asarray(sim3["R"]), np.asarray(sim3["t"])
    y = Rw[1]
    rows = []
    for c in cams:
        tr, pr = tracks[c], projs.get(c, {})
        if not pr:
            continue
        meta = json.loads((work / "frames" / c / "frames.json").read_text())
        fr = sorted(pr)
        pick = [fr[int(q * (len(fr) - 1))] for q in np.linspace(0.1, 0.9, n_frames)]
        tiles = []
        for f in pick:
            img = cv2.imread(str(work / "frames" / c / meta["frames"][f]["file"]))
            K, R, t, k1 = tr["K"][f], tr["R"][f], tr["tvec"][f], float(tr["k1"][f])
            th = max(1, round(max(img.shape[:2]) / 700))
            for a in np.arange(-6, 6.01, 1.0):
                for line in (np.stack([np.full(61, a), np.zeros(61), np.linspace(-6, 6, 61)], 1),
                             np.stack([np.linspace(-6, 6, 61), np.zeros(61), np.full(61, a)], 1)):
                    uv, vis = qa.project_cv((line - tw) @ Rw / s, K, R, t, k1)
                    for p, q, v1, v2 in zip(uv[:-1], uv[1:], vis[:-1], vis[1:]):
                        if v1 and v2 and np.all(np.abs(p) < 1e4) and np.all(np.abs(q) < 1e4):
                            cv2.line(img, tuple(int(x) for x in p), tuple(int(x) for x in q),
                                     (255, 255, 0) if a != 0 else (0, 165, 255), th, cv2.LINE_AA)
            h_, w_ = img.shape[:2]
            for ax in np.linspace(0.1, 0.9, 5):
                pl = _plumb(np.array([ax * w_, 0.3 * h_]), R @ y, K, k1, length=2.0, n=30)
                pl = pl[(np.abs(pl) < 1e5).all(1)]
                if len(pl) > 1:
                    cv2.polylines(img, [pl.astype(np.int32)], False, (0, 220, 0), th, cv2.LINE_AA)
            p, uv = pr[f]
            for (px, py), (ux, uy) in zip(p, uv):
                if abs(px) < 1e4 and abs(py) < 1e4:
                    cv2.line(img, (int(ux), int(uy)), (int(px), int(py)), (255, 0, 255), th, cv2.LINE_AA)
                    cv2.circle(img, (int(px), int(py)), 4 * th, (255, 0, 255), th, cv2.LINE_AA)
                cv2.drawMarker(img, (int(ux), int(uy)), (0, 255, 0), cv2.MARKER_CROSS, 8 * th, 2 * th)
            e = np.linalg.norm(p - uv, axis=1)
            im = cv2.resize(img, (int(round(w_ * height / h_)), height), interpolation=cv2.INTER_AREA)
            how = "attached" if tr.get("attached") else "main model"
            for k, (tx, col) in enumerate([(f"cam {c} ({how}) f{f}", (255, 255, 255)),
                                           (f"dancers: median {np.median(e):.0f} px", (255, 0, 255))]):
                cv2.putText(im, tx, (6, 24 + 24 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(im, tx, (6, 24 + 24 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
            tiles.append(im)
        v = dl.get(c, {})
        ho = v.get("holdout", {})
        lab = (f"cam {c}: " + (f"attach hold-out {ho.get('median_px')} px (fit {v.get('median_px')} px)" if v.get("circular")
                               else f"leave-one-out {v.get('median_px')} px (non-circular)"))
        row = np.hstack(tiles)
        bar = np.full((34, row.shape[1], 3), 40, np.uint8)
        cv2.putText(bar, lab, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        rows.append(np.vstack([bar, row]))
    W = max(r.shape[1] for r in rows)
    sheet = np.vstack([np.hstack([r, np.zeros((r.shape[0], W - r.shape[1], 3), np.uint8)]) for r in rows])
    ban = np.full((70, W, 3), 25, np.uint8)
    for k, tx in enumerate(["Attach check - green +: dancers' keypoints detected in this camera; magenta o: the same keypoints",
                            "triangulated by the OTHER main cameras, projected with this camera. Cyan: 1 m floor grid (y=0). Green lines: world up."]):
        cv2.putText(ban, tx, (8, 28 + 30 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    out = np.vstack([ban, sheet])
    if out.shape[1] > 3600:
        out = cv2.resize(out, (3600, int(out.shape[0] * 3600 / out.shape[1])), interpolation=cv2.INTER_AREA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), out, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return path


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


def store_dancer_px(work: Path, cam: str, per_frame: np.ndarray | None) -> bool:
    """The per-frame dancer residual (dancer_check) into cameras/<cam>.npz as the additive key dancer_px. Not into an
    adopted track (adopted_track): the residual was measured on the localisation, not on the installed track."""
    npz = work / "cameras" / f"{cam}.npz"
    if per_frame is None or not npz.exists() or adopted_track(work, cam) is not None:
        return False
    d = dict(np.load(npz))
    d["dancer_px"] = per_frame.astype(np.float32)
    np.savez(npz, **d)
    return True


def build_report(work: Path, cams, recs, model, tracks, pts, rgb, sim3, map_info, sigma_s, max_reproj_px,
                 attach: dict | None = None, n_take: int | None = None) -> dict:
    """cameras/report.json's per-camera record, the dancers' check, the QA plots and gate G2a (n_take: the take's
    camera count, for cameras_registered; None = len(cams)). cams.<cam>.adopted names the localisation an adopted
    cameras/<cam>.npz replaced: that camera's statistics here describe the localisation, not the contract file."""
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
        e = {"registered": cam in reg or bool(att.get("attached")), "in_main_model": cam in reg,
             "attached": bool(att.get("attached")),
             "how": "main model" if cam in reg else (f"own sub-model + attach ({att.get('method')})"
                                                     if att.get("attached") else "unregistered"),
             "map_images": reg.get(cam, 0), "frames": len(win), "hd": hd, "maxdim": max(meta["width"], meta["height"]),
             "map": "sfm" if cam in reg else sub_of.get(cam, ""),
             "map_reproj_px": float(np.mean(per_img_err[cam])) if cam in per_img_err else float("nan"),
             "map_reproj_median_px": float(np.median(per_img_err[cam])) if cam in per_img_err else float("nan")}
        if att:
            e["attach"] = att
        kept = adopted_track(work, cam)
        if kept is not None:  # only then: every other take's report is unchanged
            e["adopted"] = kept.relative_to(work).as_posix()
        tr = tracks.get(cam)
        if tr is not None:
            src = tr["source"]
            e.update({"frames_ok": int(tr["ok"].sum()), "frames_ok_raw": int(tr["ok_raw"].sum()),
                      "ok_rate": round(float(tr["ok"].mean()), 4),
                      "sources": {s: int((src == s).sum()) for s in ("map", "loc", "track", "attach", "anchor", "cont", "prior",
                                                                      "dancers", "unanchored") if s in ("map", "loc", "track", "attach")
                                  or (src == s).any()},  # cameras.anchored's labels only where present
                      "raw_pose_kept_frames": tr["raw_used"],
                      "reproj_px_median": float(np.nanmedian(tr["reproj_px"][tr["ok"]])) if tr["ok"].any() else None,
                      "reproj_px_p90": float(np.nanpercentile(tr["reproj_px"][tr["ok"]], 90)) if tr["ok"].any() else None,
                      "reproj_raw_median": float(np.nanmedian(tr["reproj_raw"][tr["ok_raw"]])) if tr["ok_raw"].any() else None,
                      "n_inliers_median": float(np.median(tr["n_inliers"][tr["ok"]])) if tr["ok"].any() else 0,
                      "focal": focal_stats(tr),
                      "depth_guard": {k: v for k, v in (tr.get("depth_guard") or {}).items() if k != "settings"}})
            if sim3 is not None:
                Cw = apply(sim3, tr["C"])
                e["height_m"] = [round(float(np.percentile(Cw[tr["ok"], 1], q)), 3) for q in (5, 50, 95)] \
                    if tr["ok"].any() else None
                e["dist_to_origin_m"] = round(float(np.median(np.linalg.norm(Cw[:, [0, 2]], axis=1))), 2)
                e["path_length_m"] = round(float(np.linalg.norm(np.diff(Cw, axis=0), axis=1).sum()), 2)
        cams_rep[cam] = e
    rep = {"cams": cams_rep, "settings": {"sigma_s": sigma_s, "max_reproj_px": max_reproj_px},
           "points3D": int(len(pts)), "map_mean_reproj_px": float(model.compute_mean_reprojection_error())}
    main_cams = sorted(c for c in reg if c in tracks)
    if tracks:
        dl = dancer_check(work, tracks, cams, qa_dir, main_cams=main_cams, attach=attach)
        projs: dict = {}
        for cam, v in dl.items():  # per-frame dancer residual into the contract file (additive key)
            pf = v.pop("per_frame", None)
            projs[cam] = v.pop("_proj", {})
            store_dancer_px(work, cam, pf)
        rep["dancer_loo_check"] = dl
        if sim3 is not None:
            rep["vertical_check"] = vertical_check(work, tracks, sim3, main_cams)
            prev = work / "cameras_v1" / "world_from_sfm.json"  # backup of the previous version, if kept
            prev = prev if prev.exists() else work / "cameras" / "world_from_sfm.prev.json"
            old = json.loads(prev.read_text()) if prev.exists() else None
            if old is not None and np.allclose(np.asarray(old["R"]), np.asarray(sim3["R"])):
                old = None
            rv = vertical_review(work, tracks, sim3, old, work / "review" / "cameras_vertical_check.jpg",
                                 check=rep["vertical_check"])
            rep.setdefault("qa_review", {})["vertical_check"] = str(rv)
            att_cams = [c for c in cams if c in tracks and tracks[c].get("attached")]
            if att_cams:
                ra = attach_review(work, tracks, sim3, projs, att_cams + main_cams[:1],
                                   work / "review" / "cameras_attach_check.jpg", dl)
                rep["qa_review"]["attach_check"] = str(ra)
    rep["gates_G2a"] = qa.gates(rep, len(cams), sim3, n_take=n_take)
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
