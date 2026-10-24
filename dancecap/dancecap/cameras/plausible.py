"""Physically plausible camera centres in a static-room map: the doppelganger check.

A room with repeated structure (identical double doors and EXIT signs on several walls, rows of audience) can make
COLMAP register a few images into a wrong copy of the room that still fits well: lucas_luiza's Steadicam had 21 map
images 9-10 map units below its own height band (MAD 0.2), tilted ~23 deg, and 6 more 32-36 units from the room's
centre (90 % of the points lie within 12.1), all at 1-2 px. The localiser takes each frame's 3D references from the
nearest map images IN TIME of the same camera and chains the previous frame's correspondences, so every 2-6 s block
around such an image localised into the copy (rp 1.2-1.9 px, 120-800 inliers) and the two-view sync of that stretch
saw 90 px epipolar residuals. The map gate counted the copy as "registered".

Rule, per axis. Up = the median of the images' up vectors (-row 1 of R).
    height      |h - median h| > max(k_h x 1.4826 MAD(h), floor, room_v x the room's height span): cameras are carried
                or stand at one height; a copy of the room puts one under the floor or at the ceiling
    horizontal  |c - median c| > max(k_xz x median |c - median c|, floor) AND |c - room centre| > room_x x the room's
                p99 radius (c = the centre's horizontal part): a camera that really moves across the floor - a tripod
                carried to another spot, a walking phone past its map images' span, a reused map - stays inside the room;
                a copy puts it past the walls
    band from the camera's registered map images; floor = floor_frac x the p90 horizontal radius of all points; the
    room from the points seen by >= 2 cameras (all points when fewer than MIN_MULTI_POINTS: one-camera models), heights
    p1..p99, centre = their horizontal median.
lucas_luiza (cameras/sfm_orig): the room's heights -0.86..4.58 (span 5.44: a 1.37 floor on the height band), p99 radius
12.5; the copies at h -8.5..-9.8 (the Steadicam's band 0.33 +- 1.59) and at r 34-38; the real Steadicam within h
-0.89..1.19 and r 17.7; the tripod at r 21 (behind the room: its own band keeps it). Before the per-axis rule (096e020:
band AND room on both axes) loc_01 poses 3.6-5.2 above the Steadicam's band passed as "inside the room".
A camera with fewer than min_images images gets no band (nothing is judged); a camera whose flagged images would be
more than max_share of its images keeps them all and gets no band in the localiser either (a doppelganger is a
minority) and the case is reported. A clean repeats until nothing more is flagged (the bands of the cleaned model are
tighter: lucas_luiza needed 2 passes before the room test). Single-camera sub-models (attach, reuse, map_extend) are
never gated: their fixed-focal solutions of a zooming phone slide along the axis on purpose (attach_zoom repairs them).

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
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

K_H = 8.0
K_XZ = 6.0
FLOOR_FRAC = 0.1
MIN_IMAGES = 8
MAX_SHARE = 0.4
ROOM_V = 0.25         # x the room's height span (p1..p99 of the points) under the floor / over the ceiling
ROOM_X = 2.0          # x the p99 horizontal radius of the points
MIN_MULTI_POINTS = 100
MAX_PASSES = 4


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
class Room:
    centre: np.ndarray  # (3,) horizontal part of the points' median
    h_lo: float  # p1 of the points' heights
    h_hi: float  # p99
    radius: float  # p99 horizontal radius
    points: str  # "multi-camera" | "all"
    n: int

    def outside(self, r: float, room_x: float = ROOM_X) -> str | None:
        """Past the walls: the horizontal distance r from the room's centre over room_x x its p99 radius."""
        if r > room_x * self.radius:
            return f"{r:.2f} from the room's centre (> {room_x:g} x its p99 radius {self.radius:.2f})"
        return None

    def height_floor(self, room_v: float = ROOM_V) -> float:
        return room_v * (self.h_hi - self.h_lo)

    def to_json(self) -> dict:
        return {"centre": [round(float(v), 4) for v in self.centre], "h_lo": round(self.h_lo, 4), "h_hi": round(self.h_hi, 4),
                "radius": round(self.radius, 4), "points": self.points, "n": self.n}


@dataclass
class Bands:
    up: np.ndarray
    room: float
    floor: float
    cams: dict[str, CamBand]
    box: Room | None = None
    exempt: dict = field(default_factory=dict)  # {cam: why}: the majority guard (no band in the localiser)

    def horiz(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, float)
        return X - np.multiply.outer(X @ self.up, self.up)

    def why(self, cam: str, C: np.ndarray) -> str | None:
        """None when the centre C (map frame) is plausible for camera cam (or cam has no band / the map no points), else
        the reason (module docstring: height off the band, or horizontally off the band AND past the walls)."""
        b = self.cams.get(cam)
        if b is None or self.box is None or not np.all(np.isfinite(C)):
            return None
        C = np.asarray(C, float)
        dh = float(C @ self.up) - b.h_med
        tol = max(b.h_tol, self.box.height_floor())
        if abs(dh) > tol:
            return f"height {dh:+.2f} from the camera's median (tol {tol:.2f})"
        dxz = float(np.linalg.norm(self.horiz(C) - b.xz_med))
        if dxz <= b.xz_tol:
            return None
        room = self.box.outside(float(np.linalg.norm(self.horiz(C) - self.box.centre)))
        return None if room is None else f"horizontal {dxz:.2f} from the camera's median (tol {b.xz_tol:.2f}); {room}"

    def to_json(self) -> dict:
        return {"up": [round(float(v), 4) for v in self.up], "room": round(self.room, 4), "floor": round(self.floor, 4),
                "box": None if self.box is None else self.box.to_json(), "exempt": dict(self.exempt),
                "cams": {c: b.to_json() for c, b in sorted(self.cams.items())}}


def bands(cams: np.ndarray, C: np.ndarray, U: np.ndarray, P: np.ndarray, multi: np.ndarray | None = None, k_h: float = K_H,
          k_xz: float = K_XZ, floor_frac: float = FLOOR_FRAC, min_images: int = MIN_IMAGES) -> Bands:
    """Bands from the registered images' camera ids (N,), centres (N, 3), up vectors (N, 3), the points (M, 3) and which
    points >= 2 cameras see (M,) bool (None: unknown, every point counts for the room)."""
    up = np.median(np.asarray(U, float), axis=0)
    up = up / (np.linalg.norm(up) or 1.0)
    B = Bands(up, 0.0, 0.0, {})
    P = np.asarray(P, float).reshape(-1, 3)
    if len(P):
        Hp = B.horiz(P)
        B.room = float(np.percentile(np.linalg.norm(Hp - np.median(Hp, axis=0), axis=1), 90))
        use = np.asarray(multi, bool) if multi is not None and int(np.sum(multi)) >= MIN_MULTI_POINTS else None
        Q = P[use] if use is not None else P
        hq, Hq = Q @ up, B.horiz(Q)
        cq = np.median(Hq, axis=0)
        B.box = Room(cq, float(np.percentile(hq, 1)), float(np.percentile(hq, 99)),
                     float(np.percentile(np.linalg.norm(Hq - cq, axis=1), 99)), "multi-camera" if use is not None else "all", len(Q))
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
    """({image name: why} to drop, {cam: note} for cameras whose flags were kept as a majority). Sets B.exempt to the
    latter: the localiser then gates none of their frames."""
    out, kept = {}, {}
    cams = np.asarray(cams)
    for c in sorted(B.cams):
        idx = np.flatnonzero(cams == c)
        f = {names[i]: w for i in idx if (w := B.why(c, C[i])) is not None}
        if len(f) > max_share * len(idx):
            kept[c] = f"{len(f)}/{len(idx)} images implausible (> {max_share:.0%}): none dropped, no band"
        else:
            out.update(f)
    B.exempt = dict(kept)
    return out, kept


def _cam_of(name: str) -> str:
    return name.split("/")[1]  # frames/<cam>/.. or keyframes/<cam>/.. (mapping.model_cams)


def _pose(im) -> tuple[np.ndarray, np.ndarray]:
    T = im.cam_from_world() if callable(getattr(im, "cam_from_world", None)) else im.cam_from_world
    R = np.asarray(T.rotation.matrix())
    return R, -R.T @ np.asarray(T.translation)


def model_arrays(rec) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(names, cams, centres, up vectors, points, multi) of a pycolmap Reconstruction's registered images; multi = which
    points >= 2 cameras see (image names carry the camera id)."""
    names, cams, C, U, cam_of = [], [], [], [], {}
    for iid, im in rec.images.items():  # {image_id: Image}
        if not im.has_pose:
            continue
        R, c = _pose(im)
        names.append(im.name), cams.append(_cam_of(im.name)), C.append(c), U.append(-R[1])
        cam_of[iid] = _cam_of(im.name)
    P, multi = [], []
    for p in rec.points3D.values():
        P.append(p.xyz)
        tr = getattr(p, "track", None)
        multi.append(tr is not None and len({cam_of.get(e.image_id) for e in tr.elements} - {None}) >= 2)
    return (names, np.array(cams), np.array(C).reshape(-1, 3), np.array(U).reshape(-1, 3), np.array(P).reshape(-1, 3),
            np.array(multi, bool))


def model_bands(rec, **kw) -> Bands:
    """The localiser's bands: those of check_model, without the cameras the majority guard exempts."""
    names, cams, C, U, P, multi = model_arrays(rec)
    B = bands(cams, C, U, P, multi, **kw)
    flag(names, cams, C, B)
    for c in B.exempt:
        B.cams.pop(c, None)
    return B


def check_model(rec, **kw) -> dict:
    """{flagged {name: why}, kept_majority {cam: note}, bands, per_cam {cam: flagged n}} for a Reconstruction."""
    names, cams, C, U, P, multi = model_arrays(rec)
    B = bands(cams, C, U, P, multi, **kw)
    fl, kept = flag(names, cams, C, B)
    return {"flagged": fl, "kept_majority": kept, "per_cam": _per_cam(fl), "bands": B.to_json(),
            "rule": {"k_h": kw.get("k_h", K_H), "k_xz": kw.get("k_xz", K_XZ), "floor_frac": kw.get("floor_frac", FLOOR_FRAC),
                     "min_images": kw.get("min_images", MIN_IMAGES), "max_share": MAX_SHARE, "room_v": ROOM_V,
                     "room_x": ROOM_X, "min_multi_points": MIN_MULTI_POINTS}}


def _per_cam(flagged) -> dict:
    per: dict = {}
    for n in flagged:
        per[_cam_of(n)] = per.get(_cam_of(n), 0) + 1
    return per


def deregister(rec, names) -> dict:
    """Deregister the images (their frames) named; COLMAP drops their observations and every point left with < 2."""
    names = set(names)
    p0 = rec.num_points3D()
    frames = sorted({im.frame_id for im in rec.images.values() if im.name in names and im.has_pose})
    for fid in frames:
        rec.deregister_frame(fid)
    return {"images": len(names), "frames": len(frames), "points_before": p0, "points_after": rec.num_points3D()}


def clean_model(rec, **kw) -> dict:
    """check_model + deregister of the flagged images, repeated until nothing more is flagged (<= MAX_PASSES; no bundle
    adjustment: the caller decides). The record: every pass's flags, the last check's bands, the summed removal."""
    flagged, p0, passes, out = {}, rec.num_points3D(), 0, None
    for passes in range(1, MAX_PASSES + 1):
        out = check_model(rec, **kw)
        if not out["flagged"]:
            break
        deregister(rec, out["flagged"])
        flagged.update(out["flagged"])
    out["flagged"], out["per_cam"], out["passes"] = flagged, _per_cam(flagged), passes
    out["removed"] = {"images": len(flagged), "points_before": p0, "points_after": rec.num_points3D()}
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
    reg_before = {im.name for im in rec.images.values() if im.has_pose}
    if a.cmd == "check":
        doc = check_model(rec)
        print(_summary(doc))
        return 0
    doc = clean_model(rec)
    print(_summary(doc))
    if not doc["flagged"]:
        return 0
    info = json.loads(mj.read_text()) if mj.exists() else {}
    if not orig.exists() or not info.get("doppelgangers"):  # sfm_orig belongs to this map only (stage_map / reuse remove it)
        shutil.rmtree(orig, ignore_errors=True)
        shutil.copytree(sfm, orig)
    tmp = sfm.with_name("sfm.part")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    rec.write_binary(str(tmp))
    shutil.rmtree(sfm)
    tmp.rename(sfm)
    if mj.exists():
        prev = info.get("doppelgangers") or {}
        this = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "passes": doc["passes"], "flagged": dict(doc["flagged"]),
                "per_cam": dict(doc["per_cam"]), "removed": dict(doc["removed"]),
                "how": "plausible clean: deregistered in place, no bundle adjustment (the map frame is kept)"}
        # the record of earlier cleans holds only while their images are still out of cameras/sfm (a model copied back
        # from sfm_orig has them registered again: a fresh record)
        cleans = list(prev.get("cleans") or ([] if not prev.get("flagged") else
                                             [{k: prev.get(k) for k in ("at", "how", "passes", "flagged", "per_cam", "removed")}]))
        if cleans and any(n in reg_before for c in cleans for n in (c.get("flagged") or {})):
            cleans = []
        cleans.append(this)
        flagged = {n: w for c in cleans for n, w in (c.get("flagged") or {}).items()}
        doc.update(flagged=flagged, per_cam=_per_cam(flagged), cleans=cleans, at=this["at"], how=this["how"],
                   removed={"images": len(flagged), "points_before": (cleans[0].get("removed") or {}).get("points_before"),
                            "points_after": rec.num_points3D()})
        info["doppelgangers"] = doc
        mj.write_text(json.dumps(info, indent=1, default=str))
    print(f"[plausible] cameras/sfm rewritten ({doc['removed']}); original in {orig.relative_to(work)}. "
          f"Localise again the cameras with dropped images: {sorted(doc['per_cam'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
