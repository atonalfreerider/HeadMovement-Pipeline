"""Reuse another take's static-room SfM map for a take of the SAME recordings (no re-mapping).

    python -m dancecap.cameras.reuse larissa_kadu_38 --from larissa_kadu

Two takes of one performance (e.g. larissa_kadu = 6 s window, larissa_kadu_38 = the 38.4 s all-camera span on
other audio clocks) see the same static room from the same physical frames. Incremental mapping is the slowest
cameras step (~17 min for 114 images on this workstation, roughly quadratic in images), and its keyframes already
span the whole 30-68 s overlap, so the bundle-adjusted map of the base take is copied instead:

  cameras/sfm/                     the base main model, every image renamed to THIS take's file of the same frame
                                     (window frames matched by native pts in frames.json; keyframes are the copied
                                     files of `clock_studio reuse-keyframes`, same names, checked byte-identical)
  cameras/sfm_work/sub_<x>/best/   the base sub-models (attached cameras), renamed the same way
  cameras/cache/feats/<cam>/...    the base take's cached ALIKED features of every model image, under the new keys:
                                     the model's 2D points index exactly these keypoints (checked)
  cameras/sfm_work/map_images.json {reused_from, images (names), renamed {base: this}, identical_bytes, subs}
                                     -> pipeline.map_images() uses these instead of select_map_images (the base
                                     window images cover only the base window, so a fresh selection would not be
                                     registered in the model)
  cameras/sfm_work/map.json        the base map.json + reused_from (report provenance)

Localisation of frames far (in time) from the base window relies on the keyframes (every 4 s per camera): set
takes/<take>.toml [cameras] loc = {kf_by_time = true} (localize.LocSettings) so the nearest map images in time of
either kind are the references. The SfM frame is then IDENTICAL to the base take's, so the world similarity of the
base take (and its betas-corrected scale reference) carries over exactly.

`presolve-sub <take> --cams 05,06` localises attached cameras in their sub-models ahead of the attach step (it needs
the dancers; this part does not) and stores the result under the same cache key stage_attach uses
(cameras/cache/locsub_<cam>.pkl), so the attach later only solves the placement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from ..take import PACKAGE_ROOT, Take
from .features import ImageRec, list_images

MAP_IMAGES = "map_images.json"


def _sha1(p: Path) -> str:
    return hashlib.sha1(p.read_bytes()).hexdigest()


def frame_index_map(base_work: Path, work: Path, cam: str, tol: float = 1e-6) -> dict[str, str]:
    """base frames/<cam>/<file> -> this take's frames/<cam>/<file> of the same native pts."""
    a = json.loads((base_work / "frames" / cam / "frames.json").read_text())["frames"]
    b = json.loads((work / "frames" / cam / "frames.json").read_text())["frames"]
    pts_b = np.array([r["pts"] for r in b])
    out = {}
    for r in a:
        j = int(np.argmin(np.abs(pts_b - r["pts"])))
        if abs(pts_b[j] - r["pts"]) <= tol:
            out[f"frames/{cam}/{r['file']}"] = f"frames/{cam}/{b[j]['file']}"
    return out


def _kind(name: str) -> str:
    return "kf" if name.startswith("keyframes/") else "win"


def _feat(work: Path, name: str) -> Path:
    cam = name.split("/")[1]
    return work / "cameras" / "cache" / "feats" / cam / f"{_kind(name)}_{Path(name).stem}.npz"


def rename_model(rec, rename: dict[str, str], base_work: Path, work: Path, check_bytes: bool = True) -> dict:
    """Rename every image of a pycolmap Reconstruction in place; copy its cached features to the new key.
    Returns stats {images, renamed, identical_bytes, different_bytes, missing}."""
    st = {"images": 0, "renamed": 0, "identical_bytes": 0, "different_bytes": [], "missing": []}
    for im in rec.images.values():
        old = im.name
        new = rename.get(old, old if old.startswith("keyframes/") else None)
        if new is None:
            st["missing"].append(old)
            continue
        if not (work / new).exists():
            st["missing"].append(new)
            continue
        if check_bytes:
            if _sha1(base_work / old) == _sha1(work / new):
                st["identical_bytes"] += 1
            else:
                st["different_bytes"].append([old, new])
        f_old, f_new = _feat(base_work, old), _feat(work, new)
        d = np.load(f_old)
        xy = np.array([p.xy for p in im.points2D]).reshape(-1, 2)
        if len(xy) != len(d["kpts"]) or (len(xy) and float(np.abs(xy - (d["kpts"] + 0.5)).max()) > 1e-3):
            raise RuntimeError(f"{old}: cached features do not match the model's 2D points ({len(xy)} vs "
                               f"{len(d['kpts'])} keypoints) - cannot reuse this map")
        f_new.parent.mkdir(parents=True, exist_ok=True)
        if not f_new.exists():
            shutil.copyfile(f_old, f_new)
        if new != old:
            im.name = new
            st["renamed"] += 1
        st["images"] += 1
    return st


def reuse_map(take: Take, ref: str, force: bool = False) -> dict:
    import pycolmap
    work = take.work.resolve()
    base_work = (PACKAGE_ROOT / "work" / ref).resolve()
    out = work / "cameras"
    sw = out / "sfm_work"
    if (out / "sfm").exists() and not (sw / MAP_IMAGES).exists() and not force:
        raise SystemExit(f"{out / 'sfm'} exists and was not made by reuse (pass --force to replace it)")
    cams = [s.id for s in take.sources]
    rename: dict[str, str] = {}
    for c in cams:
        if (base_work / "frames" / c / "frames.json").exists():
            rename.update(frame_index_map(base_work, work, c))
    info: dict = {"reused_from": ref, "base_sfm": str(base_work / "cameras" / "sfm"), "subs": {}}
    main = pycolmap.Reconstruction(str(base_work / "cameras" / "sfm"))
    info["main"] = rename_model(main, rename, base_work, work)
    if info["main"]["missing"]:
        raise RuntimeError(f"model images without a frame in {take.name}: {info['main']['missing'][:5]}")
    (out / "sfm").mkdir(parents=True, exist_ok=True)
    main.write_binary(str(out / "sfm"))
    names = sorted(im.name for im in main.images.values())
    for d in sorted((base_work / "cameras" / "sfm_work").glob("sub_*/best")):
        if not (d / "images.bin").exists():
            continue
        sub = pycolmap.Reconstruction(str(d))
        st = rename_model(sub, rename, base_work, work)
        if st["missing"]:
            raise RuntimeError(f"{d.parent.name}: images without a frame in {take.name}: {st['missing'][:5]}")
        dst = sw / d.parent.name / "best"
        if dst.parent.exists():
            shutil.rmtree(dst.parent)
        dst.mkdir(parents=True)
        sub.write_binary(str(dst))
        info["subs"][d.parent.name] = st
        names += sorted(im.name for im in sub.images.values())
    info["images"] = sorted(set(names))
    info["renamed"] = {k: v for k, v in sorted(rename.items()) if v in set(names)}
    bm = base_work / "cameras" / "sfm_work" / "map.json"
    if bm.exists():
        m = json.loads(bm.read_text())
        m.pop("map_images", None)
        m["reused_from"] = ref
        m["note"] = (f"static map copied from {ref} (cameras/reuse.py): images renamed to this take's frames; "
                     f"match_s/mapping_s are the base take's")
        (sw / "map.json").write_text(json.dumps(m, indent=1, default=str))
    sw.mkdir(parents=True, exist_ok=True)
    (sw / MAP_IMAGES).write_text(json.dumps(info, indent=1))
    rep_p = out / "report.json"
    rep = json.loads(rep_p.read_text()) if rep_p.exists() else {}
    if bm.exists():
        rep["map"] = json.loads((sw / "map.json").read_text())
    rep_p.write_text(json.dumps(rep, indent=1, default=str))
    return info


def reused_images(work: Path, info: dict) -> list[ImageRec]:
    """ImageRecs (this take's times and masks) of every image of the reused main model + sub-models."""
    want = set(info["images"])
    cams = sorted({n.split("/")[1] for n in want})
    recs = list_images(work, cams)
    out = [r for c in cams for r in recs[c] if r.name in want]
    missing = want - {r.name for r in out}
    if missing:
        raise RuntimeError(f"reused map images not in the frame store: {sorted(missing)[:5]}")
    return out


def presolve_sub(take: Take, cam: str) -> int:
    """Sub-model localisation of an attached camera, cached exactly as pipeline.stage_attach caches it."""
    import pycolmap

    from .. import gpu
    from .features import Features, FeatureSettings
    from .localize import Localizer, LocSettings, localize_camera
    from .mapping import model_cams
    from .pipeline import _cached, _file_key, loc_settings, map_images
    work = take.work.resolve()
    ls = loc_settings(take)
    recs = list_images(work, [s.id for s in take.sources])
    images, _ = map_images(work, recs, None)
    subs = []
    for d in sorted((work / "cameras" / "sfm_work").glob("sub_*/best")):
        if (d / "images.bin").exists():
            m = pycolmap.Reconstruction(str(d))
            subs.append((d.parent.name, m, model_cams(m)))
    cand = [(n, m, mc) for n, m, mc in subs if mc.get(cam, 0) >= 3]
    if not cand:
        raise SystemExit(f"{cam}: no sub-model")
    name, sub, mc = max(cand, key=lambda x: x[2][cam])
    frames = [r for r in recs[cam] if r.kind == "win"]
    meta = json.loads((work / "frames" / cam / "frames.json").read_text())
    w, h = int(meta["width"]), int(meta["height"])
    sub_bin = work / "cameras" / "sfm_work" / name / "best" / "images.bin"
    single = len({im.camera_id for im in sub.images.values()}) == 1
    ls_sub = LocSettings(**{**ls.__dict__, "fix_focal": single})
    loc_key = {"sub": _file_key(sub_bin), "settings": dict(sorted(ls_sub.__dict__.items())),
               "frames": [r.name for r in frames]}
    F = Features(work, work / "cameras" / "cache", FeatureSettings())

    def localise():
        L = Localizer(F, sub, images, ls_sub)
        with gpu.exclusive(f"cameras-attach-{cam}"):
            return localize_camera(L, frames, w, h)
    poses = _cached(work / "cameras" / "cache" / f"locsub_{cam}.pkl", loc_key, localise)
    n = int(sum(p.ok for p in poses))
    print(f"[cameras] {cam}: {n}/{len(frames)} frames localised in sub-model {name} {mc} (presolved)", flush=True)
    return n


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m dancecap.cameras.reuse", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=["map", "presolve-sub"])
    p.add_argument("take")
    p.add_argument("--from", dest="ref", help="map: the take whose cameras/sfm is reused")
    p.add_argument("--cams", help="presolve-sub: attached cameras")
    p.add_argument("--force", action="store_true")
    a = p.parse_args(argv)
    take = Take.load(a.take)
    if a.action == "map":
        if not a.ref:
            raise SystemExit("map: --from <take>")
        info = reuse_map(take, a.ref, force=a.force)
        print(json.dumps({"main": {k: v for k, v in info["main"].items() if k != "different_bytes"},
                          "different_bytes": len(info["main"]["different_bytes"]),
                          "subs": {k: {kk: vv for kk, vv in v.items() if kk != "different_bytes"}
                                   for k, v in info["subs"].items()},
                          "images": len(info["images"])}, indent=1))
    else:
        for c in (a.cams or "").split(","):
            if c:
                presolve_sub(take, c)


if __name__ == "__main__":
    main()
