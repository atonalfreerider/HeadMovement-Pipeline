"""Physically plausible camera centres in a static-room map: the doppelganger check.

A room with repeated structure (identical double doors and EXIT signs on several walls, rows of audience) can make
COLMAP register a few images into a wrong copy of the room that still fits well: lucas_luiza's Steadicam had 21 map
images 9-10 map units below its own height band (MAD 0.2), tilted ~23 deg, and 6 more 32-36 units from the room's
centre (90 % of the points lie within 12.1), all at 1-2 px. The localiser takes each frame's 3D references from the
nearest map images IN TIME of the same camera and chains the previous frame's correspondences, so every 2-6 s block
around such an image localised into the copy (rp 1.2-1.9 px, 120-800 inliers) and the two-view sync of that stretch
saw 90 px epipolar residuals. The map gate counted the copy as "registered".

Rule, per camera, from its registered map images (up = the median of the images' up vectors, -row 1 of R; room = the
p90 horizontal radius of the map's points about their median; floor = floor_frac x room):
    height       |h - median h| > max(k_h x 1.4826 MAD(h), floor)
    horizontal   |c - median c| > max(k_xz x median |c - median c|, floor)   (c = the centre's horizontal part)
A camera with fewer than min_images images gets no band (nothing is judged); a camera whose flagged images would be
more than max_share of its images keeps them all (a doppelganger is a minority: a camera far from the room's points,
e.g. a tripod facing one wall, is not one) and the case is reported.

    python -m dancecap.cameras.plausible check <take>            # report only
    python -m dancecap.cameras.plausible clean <take>            # deregister the flagged images of cameras/sfm in place

clean keeps the map's frame (no bundle adjustment: the tracks localised on it stay valid), backs the model up once to
cameras/sfm_orig/ and records map.json "doppelgangers". The frames localised against the copy must be localised again
(`front_pod loc-part`, or `cameras --stages loc`): the Localizer then gates every pose with the same bands
(FramePose.source "implausible", ok False) so a leftover one neither counts nor seeds the tracking of its neighbours.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

K_H = 8.0
K_XZ = 6.0
FLOOR_FRAC = 0.1
MIN_IMAGES = 8
MAX_SHARE = 0.4


@dataclass
class CamBand:
    h_med: float
    h_tol: float
    xz_med: np.ndarray  # (3,) horizontal part of the median centre
    xz_tol: float
    n: int

    def to_json(self) -> dict:
        return {"h_med": round(self.h_med, 4), "h_tol": round(self.h_tol, 4), "xz_med": [round(float(v), 4) for v in self.xz_med],
                "xz_tol": round(self.xz_tol, 4), "n": self.n}


@dataclass
class Bands:
    up: np.ndarray
    room: float
    floor: float
    cams: dict[str, CamBand]

    def horiz(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, float)
        return X - np.multiply.outer(X @ self.up, self.up)

    def why(self, cam: str, C: np.ndarray) -> str | None:
        """None when the centre C (map frame) lies in camera cam's band (or cam has none), else the reason."""
        b = self.cams.get(cam)
        if b is None or not np.all(np.isfinite(C)):
            return None
        dh = float(np.asarray(C, float) @ self.up) - b.h_med
        if abs(dh) > b.h_tol:
            return f"height {dh:+.2f} from the camera's median (tol {b.h_tol:.2f})"
        dxz = float(np.linalg.norm(self.horiz(C) - b.xz_med))
        if dxz > b.xz_tol:
            return f"horizontal {dxz:.2f} from the camera's median (tol {b.xz_tol:.2f})"
        return None

    def to_json(self) -> dict:
        return {"up": [round(float(v), 4) for v in self.up], "room": round(self.room, 4), "floor": round(self.floor, 4),
                "cams": {c: b.to_json() for c, b in sorted(self.cams.items())}}


def bands(cams: np.ndarray, C: np.ndarray, U: np.ndarray, P: np.ndarray, k_h: float = K_H, k_xz: float = K_XZ,
          floor_frac: float = FLOOR_FRAC, min_images: int = MIN_IMAGES) -> Bands:
    """Bands from the registered images' camera ids (N,), centres (N, 3), up vectors (N, 3) and the points (M, 3)."""
    up = np.median(np.asarray(U, float), axis=0)
    up = up / (np.linalg.norm(up) or 1.0)
    B = Bands(up, 0.0, 0.0, {})
    P = np.asarray(P, float).reshape(-1, 3)
    if len(P):
        Hp = B.horiz(P)
        B.room = float(np.percentile(np.linalg.norm(Hp - np.median(Hp, axis=0), axis=1), 90))
    B.floor = floor_frac * B.room
    cams, C = np.asarray(cams), np.asarray(C, float).reshape(-1, 3)
    for c in sorted(set(cams.tolist())):
        m = cams == c
        if m.sum() < min_images:
            continue
        h = C[m] @ up
        hm = float(np.median(h))
        Hc = B.horiz(C[m])
        xm = np.median(Hc, axis=0)
        d = np.linalg.norm(Hc - xm, axis=1)
        B.cams[str(c)] = CamBand(hm, max(k_h * 1.4826 * float(np.median(np.abs(h - hm))), B.floor), xm,
                                 max(k_xz * float(np.median(d)), B.floor), int(m.sum()))
    return B


def flag(names: list[str], cams: np.ndarray, C: np.ndarray, B: Bands, max_share: float = MAX_SHARE) -> tuple[dict, dict]:
    """({image name: why} to drop, {cam: note} for cameras whose flags were kept as a majority)."""
    out, kept = {}, {}
    cams = np.asarray(cams)
    for c in sorted(B.cams):
        idx = np.flatnonzero(cams == c)
        f = {names[i]: w for i in idx if (w := B.why(c, C[i])) is not None}
        if len(f) > max_share * len(idx):
            kept[c] = f"{len(f)}/{len(idx)} images outside the band (> {max_share:.0%}): none dropped"
        else:
            out.update(f)
    return out, kept


def _cam_of(name: str) -> str:
    return name.split("/")[1]  # frames/<cam>/.. or keyframes/<cam>/.. (mapping.model_cams)


def _pose(im) -> tuple[np.ndarray, np.ndarray]:
    T = im.cam_from_world() if callable(getattr(im, "cam_from_world", None)) else im.cam_from_world
    R = np.asarray(T.rotation.matrix())
    return R, -R.T @ np.asarray(T.translation)


def model_arrays(rec) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(names, cams, centres, up vectors, points) of a pycolmap Reconstruction's registered images."""
    names, cams, C, U = [], [], [], []
    for im in rec.images.values():
        if not im.has_pose:
            continue
        R, c = _pose(im)
        names.append(im.name), cams.append(_cam_of(im.name)), C.append(c), U.append(-R[1])
    P = np.array([p.xyz for p in rec.points3D.values()]).reshape(-1, 3)
    return names, np.array(cams), np.array(C).reshape(-1, 3), np.array(U).reshape(-1, 3), P


def model_bands(rec, **kw) -> Bands:
    names, cams, C, U, P = model_arrays(rec)
    return bands(cams, C, U, P, **kw)


def check_model(rec, **kw) -> dict:
    """{flagged {name: why}, kept_majority {cam: note}, bands, per_cam {cam: flagged n}} for a Reconstruction."""
    names, cams, C, U, P = model_arrays(rec)
    B = bands(cams, C, U, P, **kw)
    fl, kept = flag(names, cams, C, B)
    per = {}
    for n in fl:
        per[_cam_of(n)] = per.get(_cam_of(n), 0) + 1
    return {"flagged": fl, "kept_majority": kept, "per_cam": per, "bands": B.to_json(),
            "rule": {"k_h": kw.get("k_h", K_H), "k_xz": kw.get("k_xz", K_XZ), "floor_frac": kw.get("floor_frac", FLOOR_FRAC),
                     "min_images": kw.get("min_images", MIN_IMAGES), "max_share": MAX_SHARE}}


def deregister(rec, names) -> dict:
    """Deregister the images (their frames) named; COLMAP drops their observations and every point left with < 2."""
    names = set(names)
    p0 = rec.num_points3D()
    frames = sorted({im.frame_id for im in rec.images.values() if im.name in names and im.has_pose})
    for fid in frames:
        rec.deregister_frame(fid)
    return {"images": len(names), "frames": len(frames), "points_before": p0, "points_after": rec.num_points3D()}


def clean_model(rec, **kw) -> dict:
    """check_model + deregister of the flagged images (no bundle adjustment: the caller decides)."""
    out = check_model(rec, **kw)
    out["removed"] = deregister(rec, out["flagged"]) if out["flagged"] else {"images": 0}
    return out


def _summary(doc: dict) -> str:
    rows = [f"[plausible] {len(doc['flagged'])} map image(s) outside their camera's band: {doc['per_cam'] or '-'}"]
    for n, w in sorted(doc["flagged"].items()):
        rows.append(f"  {n}: {w}")
    for c, w in doc["kept_majority"].items():
        rows.append(f"  camera {c}: {w}")
    return "\n".join(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.cameras.plausible", description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=("check", "clean"))
    ap.add_argument("take")
    a = ap.parse_args(argv)
    import pycolmap
    from ..take import Take
    work = Take.load(a.take).work.resolve()
    sfm, orig, mj = work / "cameras" / "sfm", work / "cameras" / "sfm_orig", work / "cameras" / "sfm_work" / "map.json"
    rec = pycolmap.Reconstruction(str(sfm))
    if a.cmd == "check":
        doc = check_model(rec)
        print(_summary(doc))
        return 0
    doc = clean_model(rec)
    print(_summary(doc))
    if not doc["flagged"]:
        return 0
    if not orig.exists():
        shutil.copytree(sfm, orig)
    tmp = sfm.with_name("sfm.part")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    rec.write_binary(str(tmp))
    shutil.rmtree(sfm)
    tmp.rename(sfm)
    if mj.exists():
        info = json.loads(mj.read_text())
        prev = info.get("doppelgangers") or {}
        doc["at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        doc["how"] = "plausible clean: deregistered in place, no bundle adjustment (the map frame is kept)"
        doc["flagged"] = {**(prev.get("flagged") or {}), **doc["flagged"]}
        info["doppelgangers"] = doc
        mj.write_text(json.dumps(info, indent=1, default=str))
    print(f"[plausible] cameras/sfm rewritten ({doc['removed']}); original in {orig.relative_to(work)}. "
          f"Localise again the cameras with dropped images: {sorted(doc['per_cam'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
