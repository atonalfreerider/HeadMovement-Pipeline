"""Static-room SfM map: keyframes + a subset of window frames -> COLMAP model (cameras/sfm/).

Images: every keyframe (10/cam across 30-68 s) and every `win_step`-th window frame (~3 Hz). Pairs:
exhaustive over the "dense" set (keyframes + every `dense_step`-th window frame), plus temporal neighbours
inside each camera over the whole map set. ALIKED/LightGlue matches are verified with pycolmap's two-view
geometry (F/E/H, LO-RANSAC) and imported into a COLMAP database; incremental mapping runs with ONE camera
PER IMAGE (phones zoom), SIMPLE_RADIAL, focal prior 1.2 x max image dimension (has_prior_focal_length).
A video listed in MapSettings.shared_intrinsics (no zoom) gets ONE camera for all its images instead.

The settings come from map_settings(take.cfg): the defaults + takes/<take>.toml [cameras] map = {...}. Every builder
of a take's map-image list (stage_map / loc / attach, front_pod loc-part, map_extend, attach_zoom) uses it, so the
localisation references are exactly the map's images.
"""
from __future__ import annotations

import shutil
import time
from dataclasses import dataclass, field, fields, replace
from itertools import combinations
from pathlib import Path

import numpy as np

from .features import Features, ImageRec


@dataclass
class MapSettings:
    win_step_s: float = 2 / 3  # window frames in the map every ~0.67 s (every 20th at 30 fps); the frames in
    #                            between are localised (and tracked) against them
    dense_step_s: float = 2.0  # window frames that join the exhaustive cross-camera matching
    hd_model: str = "SIMPLE_RADIAL"  # long side >= 1280
    sd_model: str = "SIMPLE_PINHOLE"  # 480p phones: k1 is not observable enough per image (BA rank deficiency)
    seq_neighbours: int = 4  # temporal neighbours matched inside each camera (map set)
    focal_prior: float = 1.2  # x max(w, h)
    min_matches: int = 15  # raw LightGlue matches before verification
    min_inliers: int = 15  # verified inliers to keep a pair (COLMAP default 15)
    ransac_px: float = 3.0  # two-view RANSAC max error (native px)
    abs_pose_min_inliers: int = 20
    abs_pose_min_ratio: float = 0.15
    init_min_inliers: int = 80
    filter_reproj_px: float = 3.0
    refine_px: float = 2.0  # final outlier filter + global BA
    weak_min_pairs: int = 20  # cameras with fewer cross-camera pairs of >= weak_match_count raw matches ...
    weak_match_count: int = 60
    weak_partners: int = 4  # ... get ALL map-image pairs with their best partner cameras
    link_min_inliers: int = 150  # a strong static link between two cameras ...
    link_min_pairs: int = 3  # ... needs this many image pairs with link_min_inliers
    ba_global_ratio: float = 1.3  # global BA after the model grew by 30 % (COLMAP default 1.1: slow here)
    shared_intrinsics: tuple = field(default_factory=tuple)  # cameras (no zoom) whose map images share ONE COLMAP
    #   camera: one focal + k1 for the video, refined by every image (a per-image focal is weakly observable)
    focal_prior_px: dict = field(default_factory=dict)  # {cam: prior focal (px)} instead of focal_prior x max dim


def map_settings(take_cfg: dict | None = None) -> MapSettings:
    """MapSettings with takes/<take>.toml [cameras] map = {...} overrides (e.g. win_step_s, dense_step_s,
    shared_intrinsics = ["01"], focal_prior_px = {"01" = 1100.0}); no table = the defaults. A shared_intrinsics camera
    whose [cameras."NN"] block declares zoom = true is refused (ValueError): one focal for a zooming video is wrong."""
    from ..camera_types import cam_tables
    over = dict(((take_cfg or {}).get("cameras") or {}).get("map") or {})
    unknown = sorted(set(over) - {f.name for f in fields(MapSettings)})
    if unknown:
        raise ValueError(f"[cameras] map: unknown setting(s) {unknown} (MapSettings: "
                         f"{', '.join(f.name for f in fields(MapSettings))})")
    if "shared_intrinsics" in over:  # TOML arrays are lists; a lone "01" is one camera, not two characters
        v = over["shared_intrinsics"]
        over["shared_intrinsics"] = (str(v),) if isinstance(v, str) else tuple(str(c) for c in v)
        zoom = [c for c in over["shared_intrinsics"] if cam_tables(take_cfg or {}).get(c, {}).get("zoom") is True]
        if zoom:
            raise ValueError(f"[cameras] map: shared_intrinsics {zoom} declared zoom = true in their [cameras.\"NN\"] "
                             "block: a zooming video needs a focal per image (drop it from shared_intrinsics)")
    if "focal_prior_px" in over:
        over["focal_prior_px"] = {str(c): float(f) for c, f in over["focal_prior_px"].items()}
    return replace(MapSettings(), **over)


def intrinsics_modes(cams, s: MapSettings, single_camera: bool = False) -> dict[str, str]:
    """{cam: how build_database gives its images COLMAP cameras}: 'single' (single_camera: ONE camera for every image
    of the database), 'shared' (one camera per video: s.shared_intrinsics) or 'per_image' (one camera per image: zoom)."""
    return {c: "single" if single_camera else "shared" if c in s.shared_intrinsics else "per_image" for c in sorted(cams)}


def prior_focal_px(cam: str, w: int, h: int, s: MapSettings, ref_dim: int | None = None) -> float:
    """Prior focal (px) of a camera's COLMAP camera of size w x h: s.focal_prior_px[cam] - the focal at the video's
    largest image size in the map, ref_dim = its max dimension (a smaller image gets it scaled by max(w, h) / ref_dim)
    - else s.focal_prior x max(w, h)."""
    if cam in s.focal_prior_px:
        return float(s.focal_prior_px[cam]) * (max(w, h) / ref_dim if ref_dim else 1.0)
    return float(s.focal_prior * max(w, h))


def select_map_images(recs: dict[str, list[ImageRec]], s: MapSettings) -> tuple[list[ImageRec], set[str]]:
    """(map images, keys of the dense subset used for exhaustive matching)."""
    out, dense = [], set()
    for cam, rs in recs.items():
        win = [r for r in rs if r.kind == "win"]
        last_t = last_d = -1e9
        for r in win:
            if r.t - last_t >= s.win_step_s - 1e-3:
                out.append(r)
                last_t = r.t
                if r.t - last_d >= s.dense_step_s - 1e-3:
                    dense.add(r.key)
                    last_d = r.t
        for r in rs:
            if r.kind == "kf":
                out.append(r)
                dense.add(r.key)
    return out, dense


def map_pairs(images: list[ImageRec], dense: set[str], s: MapSettings) -> list[tuple[int, int]]:
    """Index pairs (i < j) into images: exhaustive over the dense set + per-camera temporal neighbours."""
    idx = {r.key: i for i, r in enumerate(images)}
    pairs = set()
    d = [i for i, r in enumerate(images) if r.key in dense]
    pairs.update((min(a, b), max(a, b)) for a, b in combinations(d, 2))
    for cam in sorted({r.cam for r in images}):
        seq = sorted((r for r in images if r.cam == cam), key=lambda r: r.t)
        for k, r in enumerate(seq):
            for q in seq[k + 1:k + 1 + s.seq_neighbours]:
                a, b = idx[r.key], idx[q.key]
                pairs.add((min(a, b), max(a, b)))
    return sorted(pairs)


def targeted_pairs(images: list[ImageRec], matches: dict[tuple[int, int], np.ndarray],
                   s: MapSettings) -> tuple[list[tuple[int, int]], dict]:
    """Extra pairs for weakly linked cameras: every map image x every map image of its best partner cameras."""
    cams = sorted({r.cam for r in images})
    strong = {c: 0 for c in cams}
    best: dict[str, dict[str, int]] = {c: {} for c in cams}
    for (i, j), m in matches.items():
        a, b = images[i].cam, images[j].cam
        if a == b:
            continue
        n = len(m)
        for x, y in ((a, b), (b, a)):
            best[x][y] = max(best[x].get(y, 0), n)
            strong[x] += n >= s.weak_match_count
    weak = [c for c in cams if strong[c] < s.weak_min_pairs]
    extra, info = set(), {}
    for c in weak:
        partners = sorted(best[c], key=lambda y: -best[c][y])[:s.weak_partners]
        info[c] = {"strong_pairs": strong[c], "partners": {p: best[c][p] for p in partners}}
        ci = [k for k, r in enumerate(images) if r.cam == c]
        pi = [k for k, r in enumerate(images) if r.cam in partners]
        extra.update((min(a, b), max(a, b)) for a in ci for b in pi)
    return sorted(extra - set(matches)), info


def _pair_key(a: ImageRec, b: ImageRec) -> str:
    return f"{a.key}|{b.key}".replace("/", "~")


def match_pairs(F: Features, images: list[ImageRec], pairs: list[tuple[int, int]], cache: Path,
                log_every: int = 500) -> dict[tuple[int, int], np.ndarray]:
    """LightGlue matches for every pair, cached in cache (npz keyed by image keys)."""
    old = dict(np.load(cache)) if cache.exists() else {}
    out, new = {}, {}
    t0 = time.time()
    for n, (i, j) in enumerate(pairs):
        k = _pair_key(images[i], images[j])
        if k in old:
            m = old[k]
        else:
            m = F.match(images[i], images[j])
            new[k] = m
        out[(i, j)] = m
        if log_every and (n + 1) % log_every == 0:
            print(f"[cameras] matched {n + 1}/{len(pairs)} pairs ({time.time() - t0:.0f}s)", flush=True)
    if new:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, **{**old, **new})
    return out


def camera_model(w: int, h: int, s: MapSettings) -> str:
    return s.hd_model if max(w, h) >= 1280 else s.sd_model


def prior_camera(w: int, h: int, factor: float, model: str = "SIMPLE_RADIAL", camera_id: int = 0,
                 focal_px: float | None = None):
    import pycolmap
    f = factor * max(w, h) if focal_px is None else float(focal_px)
    cam = pycolmap.Camera.create_from_model_name(camera_id, model, f, w, h)
    cam.has_prior_focal_length = True
    return cam


def verify(cam_a, kp_a: np.ndarray, cam_b, kp_b: np.ndarray, m: np.ndarray, s: MapSettings):
    """pycolmap two-view geometry on matches m (M,2). Returns the TwoViewGeometry (inlier_matches set)."""
    import pycolmap
    opt = pycolmap.TwoViewGeometryOptions()
    opt.min_num_inliers = s.min_inliers
    opt.ransac.max_error = s.ransac_px
    opt.ransac.min_inlier_ratio = 0.1
    opt.ransac.max_num_trials = 5000
    return pycolmap.estimate_two_view_geometry(cam_a, kp_a.astype(np.float64), cam_b, kp_b.astype(np.float64),
                                               m.astype(np.uint32), opt)


def build_database(work: Path, F: Features, images: list[ImageRec], matches: dict, db_path: Path,
                   s: MapSettings, cams: set[str] | None = None, single_camera: bool = False) -> dict:
    """COLMAP database (ALIKED keypoints, verified matches) of the images of `cams`.

    One camera per image (zoom) by default; single_camera=True shares ONE camera (one focal) between all images -
    used for single-phone sub-models of mostly-rotating phones, whose per-image focal is not observable (+-10 %
    scatter seen on cam05). A camera listed in s.shared_intrinsics (a video without zoom) gets ONE camera for all its
    images, frames and keyframes (one per image size, should they differ), SIMPLE_RADIAL; the others stay per image.
    Prior focal: prior_focal_px (s.focal_prior_px[cam] at the video's largest image size, scaled for a smaller one;
    else focal_prior x max dim). stats['intrinsics'] = {cam: mode, prior focal of its first image, COLMAP cameras};
    stats['per_campair'][pair]['link_pairs'] = its image pairs with >= link_min_inliers inliers (link_components needs
    link_min_pairs of them; both thresholds are in the stats)."""
    import pycolmap
    if db_path.exists():
        db_path.unlink()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    pycolmap.Database.open(db_path).close()  # create the schema
    ro = pycolmap.ImageReaderOptions()
    ro.camera_model = "SIMPLE_RADIAL"
    ro.default_focal_length_factor = s.focal_prior
    use = [k for k, r in enumerate(images) if cams is None or r.cam in cams]
    modes = intrinsics_modes({images[k].cam for k in use}, s, single_camera)
    shared = sorted(c for c, m in modes.items() if m == "shared")
    if not shared:
        mode = pycolmap.CameraMode.SINGLE if single_camera else pycolmap.CameraMode.PER_IMAGE
        pycolmap.import_images(db_path, work, mode, [images[k].name for k in use], ro)
    else:
        for c in shared:  # one import = one camera; COLMAP skips an image of another size, which the next round takes
            left = [images[k].name for k in use if images[k].cam == c]
            while left:
                pycolmap.import_images(db_path, work, pycolmap.CameraMode.SINGLE, left, ro)
                db = pycolmap.Database.open(db_path)
                have = {im.name for im in db.read_all_images()}
                db.close()
                if not have & set(left):
                    raise RuntimeError(f"[cameras] map: camera {c}: {len(left)} images could not be imported")
                left = [n for n in left if n not in have]
        rest = [images[k].name for k in use if modes[images[k].cam] != "shared"]
        if rest:
            pycolmap.import_images(db_path, work, pycolmap.CameraMode.PER_IMAGE, rest, ro)
    db = pycolmap.Database.open(db_path)
    by_name = {im.name: im for im in db.read_all_images()}
    ref_dim: dict[str, int] = {}  # max dimension of the largest image of each video with a focal_prior_px
    for k in use:
        if images[k].cam in s.focal_prior_px:
            c0 = db.read_camera(by_name[images[k].name].camera_id)
            ref_dim[images[k].cam] = max(ref_dim.get(images[k].cam, 0), c0.width, c0.height)
    ids, cams_, made, intr = {}, {}, {}, {}
    for k in use:
        r = images[k]
        im = by_name[r.name]
        cam = made.get(im.camera_id)
        if cam is None:  # the prior, refined per image (zoom), per video (shared) or once (single_camera)
            cam = db.read_camera(im.camera_id)
            # a shared camera keeps k1 at any size: one k1 per video is observable, unlike one per 480p image
            model = s.hd_model if modes[r.cam] == "shared" else camera_model(cam.width, cam.height, s)
            fp = prior_focal_px(r.cam, cam.width, cam.height, s, ref_dim.get(r.cam)) if r.cam in ref_dim else None
            cam = prior_camera(cam.width, cam.height, s.focal_prior, model, cam.camera_id, focal_px=fp)
            db.update_camera(cam)
            made[im.camera_id] = cam
        e = intr.setdefault(r.cam, {"mode": modes[r.cam], "cameras": set(),
                                    "focal_prior_px": round(prior_focal_px(r.cam, cam.width, cam.height, s,
                                                                           ref_dim.get(r.cam)), 1)})
        e["cameras"].add(im.camera_id)
        f = F.get(r)
        db.write_keypoints(im.image_id, (f["kpts"] + 0.5).astype(np.float32))  # COLMAP: pixel corner origin
        ids[k] = im.image_id
        cams_[k] = cam
    stats = {"pairs": 0, "raw_ok": 0, "verified": 0, "per_campair": {}, "pair_inliers": {},
             "intrinsics": {c: {**e, "cameras": len(e["cameras"])} for c, e in sorted(intr.items())},
             "link_min_inliers": s.link_min_inliers, "link_min_pairs": s.link_min_pairs}
    for (i, j), m in matches.items():
        if i not in ids or j not in ids:
            continue
        stats["pairs"] += 1
        if len(m) < s.min_matches:
            continue
        stats["raw_ok"] += 1
        a, b = images[i], images[j]
        tvg = verify(cams_[i], F.get(a)["kpts"] + 0.5, cams_[j], F.get(b)["kpts"] + 0.5, m, s)
        n_in = len(tvg.inlier_matches)
        if n_in < s.min_inliers:
            continue
        stats["pair_inliers"][(i, j)] = n_in
        db.write_matches(ids[i], ids[j], m)
        db.write_two_view_geometry(ids[i], ids[j], tvg)
        stats["verified"] += 1
        cp = "-".join(sorted((a.cam, b.cam)))
        e = stats["per_campair"].setdefault(cp, {"pairs": 0, "max_inliers": 0, "sum_inliers": 0, "link_pairs": 0})
        e["pairs"] += 1
        e["max_inliers"] = max(e["max_inliers"], n_in)
        e["sum_inliers"] += n_in
        e["link_pairs"] += int(n_in >= s.link_min_inliers)  # the count link_components tests against link_min_pairs
    db.close()
    return stats


def link_components(images: list[ImageRec], pair_inliers: dict[tuple[int, int], int], s: MapSettings) -> list[set]:
    """Camera groups connected by STRONG static links: >= link_min_pairs image pairs with >= link_min_inliers
    verified inliers. Weak links (a few dozen inliers on blinds, far crowds, a periodic shutter) can register a
    camera at a wrong, degenerate position (seen: cam06 smeared over 8 m), so they never join the main model."""
    cams = sorted({r.cam for r in images})
    count: dict[tuple[str, str], int] = {}
    for (i, j), n in pair_inliers.items():
        a, b = sorted((images[i].cam, images[j].cam))
        if a != b and n >= s.link_min_inliers:
            count[(a, b)] = count.get((a, b), 0) + 1
    parent = {c: c for c in cams}

    def find(c):
        while parent[c] != c:
            c = parent[c]
        return c
    for (a, b), n in count.items():
        if n >= s.link_min_pairs:
            parent[find(a)] = find(b)
    groups: dict[str, set] = {}
    for c in cams:
        groups.setdefault(find(c), set()).add(c)
    return sorted(groups.values(), key=lambda g: -len(g))


def run_mapper(work: Path, db_path: Path, out: Path, s: MapSettings, init_min_tri_angle: float | None = None,
               seed: int = 0):
    """Incremental mapping; returns (best reconstruction = most registered images, then most points; all models).

    init_min_tri_angle: lower it for single-phone sub-models (a handheld phone mostly rotates)."""
    import pycolmap
    opt = pycolmap.IncrementalPipelineOptions()
    opt.min_num_matches = s.min_inliers
    opt.ba_refine_focal_length = True
    opt.ba_refine_extra_params = True
    opt.mapper.abs_pose_min_num_inliers = s.abs_pose_min_inliers
    opt.mapper.abs_pose_min_inlier_ratio = s.abs_pose_min_ratio
    opt.mapper.init_min_num_inliers = s.init_min_inliers
    if init_min_tri_angle is not None:
        opt.mapper.init_min_tri_angle = init_min_tri_angle
    opt.mapper.filter_max_reproj_error = s.filter_reproj_px
    opt.triangulation.ignore_two_view_tracks = False
    opt.extract_colors = True
    opt.random_seed = seed
    opt.mapper.random_seed = seed
    opt.ba_global_frames_ratio = s.ba_global_ratio
    opt.ba_global_points_ratio = s.ba_global_ratio
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    models = pycolmap.incremental_mapping(db_path, work, out, opt)
    if not models:
        return None, {}
    best = max(models.values(), key=lambda r: (r.num_reg_images(), r.num_points3D()))
    return best, models


def run_sub_mapper(work: Path, db_path: Path, out: Path, s: MapSettings, min_points: int = 300, tries: int = 4):
    """Single-phone sub-model. Rotation-dominated sequences make incremental mapping unstable (a run can register
    every image on a few dozen points), so retry with other seeds and keep the model with the most points."""
    best = None
    for seed in range(tries):
        rec, _ = run_mapper(work, db_path, out / f"try{seed}", s, init_min_tri_angle=4.0, seed=seed)
        if rec is not None and (best is None or (rec.num_reg_images(), rec.num_points3D()) >
                                (best.num_reg_images(), best.num_points3D())):
            best = rec
        if best is not None and best.num_points3D() >= min_points:
            break
    return best


def refine_model(rec, max_px: float, rounds: int = 2) -> dict:
    """Drop observations above max_px (outliers: rolling shutter, blur, residual mismatches), global BA, repeat."""
    import pycolmap
    before = rec.compute_mean_reprojection_error()
    removed = 0
    for _ in range(rounds):
        removed += pycolmap.ObservationManager(rec).filter_all_points3D(max_px, 1.0)
        opt = pycolmap.BundleAdjustmentOptions()
        opt.print_summary = False
        opt.refine_focal_length = True
        opt.refine_extra_params = True
        pycolmap.bundle_adjustment(rec, opt)
    return {"mean_reproj_before": before, "mean_reproj_after": rec.compute_mean_reprojection_error(),
            "filter_px": max_px, "observations_removed": int(removed), "points_after": rec.num_points3D()}


def model_cams(rec, ) -> dict[str, int]:
    """registered images per camera id (image names are frames/<cam>/.. or keyframes/<cam>/..)."""
    out: dict[str, int] = {}
    for im in rec.images.values():
        if im.has_pose:
            cam = im.name.split("/")[1]
            out[cam] = out.get(cam, 0) + 1
    return out
