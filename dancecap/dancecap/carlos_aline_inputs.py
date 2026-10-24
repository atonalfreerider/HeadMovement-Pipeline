"""carlos_aline (or --take <take>): the inputs the class-recap front end (dancecap.recap_front, run with HM_FRONT_DANCE=<take>) expects (agent C3).

C2 left two camera tracks (cameras/01.npz, 02.npz: per-frame rotation + focal of a tripod head that pans / tilts / zooms) and the YOLO11x-pose
detections of every person of both videos (room_work/podout/pose_A.npz, pose_B.npz). The recap front wants, for a take work dir:

    cameras/<cam>.npz       the demo's schema + `pts` (the video's own presentation time of each canonical frame) + `native_size`
    cameras/recap_cams.json {cam: {file, width, height, rotation, avg_fps, slope, offset}} (reference time = slope * source + offset)
    people/pose_tables/<cam>.npz   kp (F, P, 17, 3) [x, y, conf] NaN padded, n (F,), rgb (F, P, 3) torso colour, box (F, P, 4), det_conf (F, P)

    python -m dancecap.carlos_aline_inputs prepare        (local, light: numpy only)

C2's original track files are kept as cameras/c2_orig/<cam>.npz before the two additive keys are written.

Another take (2026-10-24, the two-view lucas_luiza; without the switches carlos_aline's run is unchanged):

    python -m dancecap.carlos_aline_inputs prepare --take lucas_luiza --pts-from frames --pose-from people --teachers-floor

  --take <take>      names, videos and clocks from takes/<take>.toml: the sources with their audio clocks (clock/alignment.json; carlos_aline keeps C2's
                     sync/sync.json, see --clocks), CAPTURE = the source folder's name (the pod reads $HM_ROOT/src/<capture>/<file>: recap_people).
                     The camera stage's own tracks are kept in cameras/orig/ and refreshed whenever cameras/<cam>.npz is not the file this module
                     wrote last (sha1 in cameras/orig/written.json): the camera or qa stage rewrote it, a PTZ adoption installed one or its undo
                     put one back
  --pts-from frames  the track rows are the ingest's canonical frames: pts from frames/<cam>/frames.json, frame_index = the video frame of that pts
                     (|delta| <= 1 ms enforced). The default (index) is C2's direction: its tracks carry frame_index, pts = the video's pts there
                     (refused for a track without frame_index, i.e. any camera stage's but C2's)
  --pose-from people people/pose_tables/<cam>.npz from the take's own detections people/<cam>.npz instead of C2's podout files: row frame_index[i] =
                     canonical frame i, slot j = detection j (NaN where not valid), n = last valid slot + 1, so a teacher `row` indexes it directly
                     (recap_front.people_arrays); box / det_conf are the detector's, rgb NaN (no colour)
  --teachers-floor   people/teachers_<cam>.npz gets frame_index and floor (F, 2, 2): the world (x, z) where each role's ankle-midpoint ray meets the
                     height of that role's ankle joint (cameras/<cam>.npz in the SfM frame + cameras/world_from_sfm.json, whose y = 0 is the floor and
                     puts the ankles at 0.039 x stature + shoe sole: [dancers.<role>]; cast to y = 0 a ray lands h_cam / (h_cam - h_ankle) too far
                     out, 6-12 % for lucas_luiza's cameras, more for the heeled follow);
                     NaN: camera not ok, no ankle seen
Every camera is read and checked before the first file is written: a refusal (a missing input, pts off the video by > 1 ms, rows that are not the
track's frames) leaves the take as it was.
Gates (people/inputs.json <cam>.gates {value, threshold, pass}; exit 1 when one fails, the files are written): max |slope pts + offset - t| <= 1 ms;
frame_index steps = frames.json decimate_every (1 for a 29 / 30 fps video, 2 for the 59.94 fps one); pose-table row frame_index[i] = people/<cam>.npz
row i; floor finite on >= 80 % of the frames with both roles matched; median lead-follow floor distance 0.3-1.5 m; every teacher row a valid
detection, and >= 95 % of them still valid in the chunk's people/<cam>.npz (people_arrays keeps a slot only with >= 3 joints of conf >= 0.3).
Known reader: film_heat.pov_table indexes the teachers by the track's frame_index, which is the video's frame number after this run (not arange).
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .take import PACKAGE_ROOT, WORKSPACE, Take

DANCE = "carlos_aline"
CAPTURE = "Carlos-Aline"
WORK = PACKAGE_ROOT / "work" / DANCE
PEOPLE = WORK / "people"
SOURCE = WORKSPACE / "dances" / CAPTURE                       # git-ignored junction to the owner's folder
CAMS = ("01", "02")
VIDEO = {"01": "Carlos Aline Zouk Artist Jack Jill at Zouk SEA 2019.mp4", "02": "Zouk Sea 2019 Carlos and Aline Borges Jack Jill.mp4"}
POSE_SRC = {"01": WORK / "room_work" / "podout" / "pose_A.npz", "02": WORK / "room_work" / "podout" / "pose_B.npz"}

PTS_TOL_S = 0.001            # frames.json pts (ffmpeg showinfo) vs the video's packet pts (ffprobe): both 6 decimals, frames >= 16.7 ms apart
CLOCK_TOL_S = 0.001          # gate: slope * pts + offset vs the track's t
KP_ANKLES = (15, 16)         # COCO-17
FLOOR_CONF = 0.3             # an ankle counts for the floor ray from this confidence (carlos_aline_people.JOINT_CONF)
FLOOR_MAX_M = 25.0           # a floor hit farther from the camera than this is a grazing ray, not a foot
FLOOR_MIN_SHARE = 0.8        # gate: floor finite on this share of the frames with both roles
PAIR_M = (0.3, 1.5)          # gate: median lead-follow floor distance of a dancing couple
CHUNK_ROW_MIN_SHARE = 0.95   # gate: teacher rows the chunk's people_arrays keeps valid (lucas_luiza's seeded teachers: 0.980 / 0.979)
WRITTEN = "written.json"     # cameras/<orig>/written.json {cam: sha1 of the cameras/<cam>.npz prepare wrote last}


def log(msg: str) -> None:
    print(f"[carlos_aline_inputs {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def probe(path: Path) -> dict:
    import subprocess
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,avg_frame_rate,duration",
                          "-of", "json", str(path)], capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    a, b = s["avg_frame_rate"].split("/")
    return {"width": int(s["width"]), "height": int(s["height"]), "avg_fps": float(a) / float(b), "duration": float(s["duration"])}


def ankle_heights(dancers: dict) -> tuple[float, float]:
    """(lead, follow) height of the ankle joint above the world floor y = 0: cameras.world's rule ANKLE_FRAC x stature + shoe sole, from the TOML's
    [dancers.<role>] stature_m / shoe_sole_m (world's defaults where one is absent)."""
    from .cameras import world as W
    d = {r: dancers.get(r) or {} for r in ("lead", "follow")}
    return tuple(round(W.ANKLE_FRAC * float(d[r].get("stature_m", W.DEFAULT_STATURE[r])) + float(d[r].get("shoe_sole_m", W.SOLE_M)), 4)
                 for r in ("lead", "follow"))


@dataclass(frozen=True)
class Spec:
    """What prepare needs of one take: its names, the source video and the clock {slope, offset, offset_source} of every camera."""
    take: str
    capture: str
    work: Path
    videos: dict
    clocks: dict
    orig: str = "orig"          # cameras/<orig>/: the camera stage's own track files, kept before the additive keys are written
    ankle_y: tuple | None = None   # (lead, follow) ankle height above the floor for --teachers-floor (None: ankle_heights({}))

    @property
    def cams(self) -> tuple:
        return tuple(self.videos)


def spec(take: str = DANCE, clocks: str | None = None) -> Spec:
    """carlos_aline keeps C2 / C3's constants above and the clocks of sync/sync.json (clocks "sync"); any other take is read from its TOML (clocks
    "alignment": the Take's sources, i.e. its clock/alignment.json)."""
    clocks = clocks or ("sync" if take == DANCE else "alignment")
    t = None if take == DANCE and clocks == "sync" else Take.load(take)
    if take == DANCE:
        work, capture, videos, orig = WORK, CAPTURE, {c: SOURCE / VIDEO[c] for c in CAMS}, "c2_orig"
    else:
        work, capture, videos, orig = t.work, Path(t.cfg["source_dir"]).name, {s.id: s.path for s in t.sources}, "orig"
    if clocks == "sync":
        rows = {c["id"]: c for c in json.loads((work / "sync" / "sync.json").read_text())["cameras"]}
        clk = {c: {"slope": float(rows[c]["slope"]), "offset": float(rows[c]["offset"]), "offset_source": rows[c].get("offset_source", "")}
               for c in videos}
    else:
        clk = {s.id: {"slope": float(s.slope), "offset": float(s.offset), "offset_source": f"audio clock ({t.cfg['alignment']})"} for s in t.sources}
    return Spec(take, capture, work, videos, clk, orig, ankle_heights((t.cfg if t is not None else {}).get("dancers") or {}))


def cams_table(sp: Spec | None = None, frames: dict | None = None, probe_fn=probe) -> dict:
    """recap_cams.json's cams; with frames ({cam: frames.json}) size and rotation are the ingest's (display pixels, as the tracks' K and the
    detections)."""
    sp = sp or spec()
    out = {}
    for cam, path in sp.videos.items():
        p = probe_fn(path)
        fm = (frames or {}).get(cam) or {}
        c = sp.clocks[cam]
        out[cam] = {"id": cam, "file": path.name, "width": int(fm.get("width", p["width"])), "height": int(fm.get("height", p["height"])),
                    "rotation": int(fm.get("rotation", 0)), "avg_fps": p["avg_fps"], "duration": p["duration"],
                    "slope": c["slope"], "offset": c["offset"], "accepted": True, "offset_source": c["offset_source"]}
    return out


def pose_table(src: Path, n_frames: int | None = None) -> dict:
    """The flat detection list of people2d (frame, box, conf, kp, rgb) as per-frame tables, persons ordered by confidence within a frame."""
    z = dict(np.load(src))
    fr = z["frame"].astype(int)
    F = int(z["n_frames"]) if n_frames is None else n_frames
    n = np.bincount(fr, minlength=F).astype(np.int16)
    P = int(max(1, n.max()))
    kp = np.full((F, P, 17, 3), np.nan, np.float32)
    box = np.full((F, P, 4), np.nan, np.float32)
    dconf = np.zeros((F, P), np.float32)
    rgb = np.full((F, P, 3), np.nan, np.float32)
    order = np.lexsort((-z["conf"], fr))
    slot = np.zeros(F, int)
    for i in order:
        f = fr[i]
        j = slot[f]
        slot[f] += 1
        kp[f, j], box[f, j], dconf[f, j], rgb[f, j] = z["kp"][i], z["box"][i], z["conf"][i], z["rgb"][i]
    return {"kp": kp, "n": n, "box": box, "det_conf": dconf, "rgb": rgb}


# ------------------------------------------------------------------------------------------------ the two-view take's inputs (pure numpy)


def frame_index_from_pts(pts_all: np.ndarray, pts: np.ndarray, tol_s: float = PTS_TOL_S) -> tuple[np.ndarray, float]:
    """The video frame (index into the video's sorted pts) of every canonical frame's pts: the nearest one, |delta| <= tol_s enforced
    -> (frame_index int32, max |delta| s). ValueError names the frames that have no video frame within tol_s."""
    pts_all, pts = np.asarray(pts_all, float), np.asarray(pts, float)
    j = np.clip(np.searchsorted(pts_all, pts), 0, len(pts_all) - 1)
    lo = np.maximum(j - 1, 0)
    j = np.where(np.abs(pts_all[lo] - pts) < np.abs(pts_all[j] - pts), lo, j)
    d = np.abs(pts_all[j] - pts)
    bad = np.flatnonzero(d > tol_s)
    if len(bad):
        raise ValueError(f"{len(bad)} of {len(pts)} frames have no video frame within {tol_s * 1000:g} ms (first: frame {int(bad[0])}, pts "
                         f"{pts[bad[0]]:.6f} s, nearest {pts_all[j[bad[0]]]:.6f} s): frames.json is not from this video")
    return j.astype(np.int32), float(d.max()) if len(d) else 0.0


def _gate(value, threshold, ok: bool) -> dict:
    return {"value": value, "threshold": threshold, "pass": bool(ok)}


def step_gate(fi: np.ndarray, every: int) -> dict:
    """frame_index strictly increasing with every step = the ingest's decimation (1, or 2 for a 59.94 fps video kept at 30 Hz)."""
    steps = np.unique(np.diff(np.asarray(fi, int)))
    return _gate([int(s) for s in steps[:8]], every, len(fi) < 2 or (len(steps) == 1 and int(steps[0]) == every))


def people_pose_table(people: dict, frame_index: np.ndarray, n_frames: int) -> dict:
    """people/<cam>.npz (kpts, valid, boxes, scores; one row per canonical frame, the detector's slots) as a pose table on the video's own frame
    numbers: row frame_index[i] = canonical frame i with slot j = detection j (NaN where the slot is not valid), n = last valid slot + 1; the other
    video frames stay empty (n = 0)."""
    valid = np.asarray(people["valid"], bool)
    fi = np.asarray(frame_index, int)
    F, N = valid.shape
    if len(fi) != F:
        raise ValueError(f"people has {F} rows, frame_index {len(fi)}")
    if F and (fi.min() < 0 or fi.max() >= n_frames or len(np.unique(fi)) != F):
        raise ValueError(f"frame_index must be distinct video frames in 0..{n_frames - 1}")
    last = np.where(valid.any(1), N - np.argmax(valid[:, ::-1], 1), 0)
    P = int(max(1, last.max() if F else 0))
    v = valid[:, :P]
    kp = np.full((n_frames, P, 17, 3), np.nan, np.float32)
    box = np.full((n_frames, P, 4), np.nan, np.float32)
    dconf = np.zeros((n_frames, P), np.float32)
    n = np.zeros(n_frames, np.int16)
    kp[fi] = np.where(v[..., None, None], people["kpts"][:, :P], np.nan)
    box[fi] = np.where(v[..., None], people["boxes"][:, :P], np.nan)
    dconf[fi] = np.where(v, people["scores"][:, :P], 0.0)
    n[fi] = last
    return {"kp": kp, "n": n, "box": box, "det_conf": dconf, "rgb": np.full((n_frames, P, 3), np.nan, np.float32)}


def pose_rows_match(pt: dict, people: dict, frame_index: np.ndarray) -> bool:
    """Gate: the pose-table row frame_index[i] holds people row i (valid slots equal, the rest NaN, n = last valid slot + 1)."""
    valid = np.asarray(people["valid"], bool)
    fi = np.asarray(frame_index, int)
    P = pt["kp"].shape[1]
    if valid[:, P:].any():
        return False
    v = valid[:, :P]
    kp = pt["kp"][fi]
    want = np.where(v[..., None, None], people["kpts"][:, :P], np.nan)
    last = np.where(valid.any(1), valid.shape[1] - np.argmax(valid[:, ::-1], 1), 0)
    return bool(np.array_equal(kp, want, equal_nan=True) and np.array_equal(pt["n"][fi], last))


def floor_xz(kp: np.ndarray, cam: dict, sim3: dict, plane_y=0.0, min_conf: float = FLOOR_CONF) -> np.ndarray:
    """(F, 2, 2) world (x, z) where each role's ankle-midpoint ray meets the plane y = plane_y (a number, or one per role (2,): prepare passes the
    ankle heights, so the point is the one above the foot's floor contact). kp (F, 2, 17, 3) per canonical frame; cam = the cameras/<cam>.npz arrays
    (K, R world -> camera and tvec in the SfM frame, dist = k1, ok); sim3 = world_from_sfm.json (X_world = s R X_sfm + t). The midpoint pixel of the
    ankles with conf >= min_conf (the one ankle when only one is seen) is undistorted (SIMPLE_RADIAL, as world.undistort_pts) and cast; NaN where the
    camera is not ok, no ankle is seen, or the ray meets the plane behind the camera or farther than FLOOR_MAX_M."""
    plane_y = np.asarray(plane_y, float)
    kp = np.asarray(kp, float)
    K, R, tv = np.asarray(cam["K"], float), np.asarray(cam["R"], float), np.asarray(cam["tvec"], float)
    F = len(K)
    k1 = np.asarray(cam["dist"], float).reshape(F, -1)[:, 0]
    ok = np.asarray(cam.get("ok", np.ones(F, bool)), bool)
    an = kp[:, :, KP_ANKLES, :]
    w = (np.nan_to_num(an[..., 2], nan=0.0) >= min_conf).astype(float)
    nseen = w.sum(-1)
    uv = (np.nan_to_num(an[..., :2]) * w[..., None]).sum(-2) / np.maximum(nseen, 1)[..., None]
    x = (uv[..., 0] - K[:, None, 0, 2]) / K[:, None, 0, 0]
    y = (uv[..., 1] - K[:, None, 1, 2]) / K[:, None, 1, 1]
    xu, yu = x.copy(), y.copy()
    for _ in range(5):
        d = 1 + k1[:, None] * (xu * xu + yu * yu)
        xu, yu = x / d, y / d
    ray = np.stack([xu, yu, np.ones_like(xu)], -1)
    Rw, s, tw = np.asarray(sim3["R"], float), float(sim3["s"]), np.asarray(sim3["t"], float)
    C = s * (-np.einsum("fji,fj->fi", R, tv)) @ Rw.T + tw                       # camera centre, world
    dw = np.einsum("fji,fkj->fki", R, ray) @ Rw.T                                # ray direction R^T ray, world
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        lam = (plane_y - C[:, None, 1]) / dw[..., 1]
        P = C[:, None] + lam[..., None] * dw
        good = ok[:, None] & (nseen > 0) & np.isfinite(lam) & (lam > 0) & (np.linalg.norm(P - C[:, None], axis=-1) <= FLOOR_MAX_M)
    return np.where(good[..., None], P[..., [0, 2]], np.nan).astype(np.float32)


def floor_gates(floor: np.ndarray, row: np.ndarray) -> dict:
    """Gates on the teachers' floor: finite for both roles on >= FLOOR_MIN_SHARE of the frames where both roles are matched, median lead-follow
    distance inside PAIR_M."""
    both = (np.asarray(row)[:, 0] >= 0) & (np.asarray(row)[:, 1] >= 0)
    pair = both & np.isfinite(floor).all(-1).all(-1)
    share = float(pair.sum() / max(1, both.sum()))
    dist = np.linalg.norm(floor[pair, 0] - floor[pair, 1], axis=-1)
    med = float(np.median(dist)) if len(dist) else float("nan")
    return {"floor_finite_share": _gate(round(share, 4), FLOOR_MIN_SHARE, share >= FLOOR_MIN_SHARE),
            "floor_pair_distance_median_m": _gate(round(med, 3) if np.isfinite(med) else None, list(PAIR_M), PAIR_M[0] <= med <= PAIR_M[1])}


def teacher_row_gates(row: np.ndarray, people: dict, pt: dict, frame_index: np.ndarray) -> dict:
    """Gates on the teachers' rows (row >= 0): each one a valid detection of people/<cam>.npz (the slot exists in the pose table), and the share the
    chunk keeps: recap_front.people_arrays rebuilds the chunk's people/<cam>.npz from the pose table and marks a slot valid only when
    recap_people.person_box finds >= 3 joints of conf >= 0.3, so a teacher on a faint detection is a miss there (the slot's box is zero)."""
    from .recap_front import people_arrays
    r = np.asarray(row, int)
    ii, kk = np.nonzero(r >= 0)
    rr = r[ii, kk]

    def share(valid: np.ndarray) -> float:
        N = valid.shape[1]
        return float(((rr < N) & valid[ii, np.minimum(rr, N - 1)]).mean()) if len(rr) else 1.0
    det = share(np.asarray(people["valid"], bool))
    chunk = share(people_arrays(pt["kp"], pt["n"], np.asarray(frame_index, int))["valid"])
    return {"teacher_rows_valid_in_pose_table": _gate(round(det, 4), 1.0, det == 1.0),
            "teacher_rows_valid_in_chunk_people": _gate(round(chunk, 4), CHUNK_ROW_MIN_SHARE, chunk >= CHUNK_ROW_MIN_SHARE)}


def _frames_meta(work: Path, cam: str) -> dict:
    return json.loads((work / "frames" / cam / "frames.json").read_text())


def _sha1(p: Path) -> str:
    import hashlib
    h = hashlib.sha1()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _fresh(src: Path, bak: Path, written: str | None) -> bool:
    """Whether cameras/<cam>.npz (src) is to be kept as the original (bak) before it is rebuilt: there is no backup yet, or src is not the file prepare
    wrote last (its sha1 `written`): the camera stage replaced it, the qa stage rewrote it in place (dancer_px), a PTZ adoption installed another
    track or its undo moved one back. Without a record (C3's runs, which write none) a file without the additive `pts` is the stage's new one."""
    if not bak.exists():
        return True
    if not src.exists():
        return False
    if written:
        return _sha1(src) != written
    with np.load(src, allow_pickle=True) as z:
        return "pts" not in z.files


def _load(p: Path, what: str, allow_pickle: bool = False) -> dict:
    if not p.exists():
        raise FileNotFoundError(f"{p} missing ({what})")
    with np.load(p, allow_pickle=allow_pickle) as z:
        return {k: z[k] for k in z.files}


def _same_frames(t_a: np.ndarray, t_b: np.ndarray) -> bool:
    return len(t_a) == len(t_b) and (len(t_a) == 0 or float(np.max(np.abs(np.asarray(t_a, float) - np.asarray(t_b, float)))) <= CLOCK_TOL_S)


def _check(sp: Spec, cam: str, table: dict, frames: dict | None, sim3: dict | None, pts_from: str, pose_from: str, pts_fn, fresh: bool) -> dict:
    """One camera's track with its additive keys, its frame_index and (with sim3) the teachers' frame_index / floor, after every refusal: an input
    that is missing or not on the track's frames raises here, before prepare writes anything. The pose table, the heavy part, is built at write time."""
    cdir, people = sp.work / "cameras", sp.work / "people"
    z = _load(cdir / f"{cam}.npz" if fresh else cdir / sp.orig / f"{cam}.npz", "the camera stage's track", allow_pickle=True)
    pts_all = pts_fn(sp.videos[cam], sp.work / "frames" / f"{cam}_pts.json")
    gates, extra = {}, {}
    if pts_from == "frames":
        fm = frames[cam]
        pts = np.array([f["pts"] for f in fm["frames"]], np.float64)
        if len(pts) != len(z["t"]):
            raise ValueError(f"{cam}: cameras/{cam}.npz has {len(z['t'])} rows, frames/{cam}/frames.json {len(pts)} frames: not the canonical frames")
        fi, d_s = frame_index_from_pts(pts_all, pts)
        z["pts"], z["frame_index"] = pts, fi
        gates["frame_index_steps"] = step_gate(fi, int(fm.get("decimate_every", 1)))
        extra = {"pts_from": "frames", "video_pts_max_dev_ms": round(d_s * 1000, 4), "frame_index": [int(fi[0]), int(fi[-1])] if len(fi) else []}
    else:
        if "frame_index" not in z:
            raise SystemExit(f"{cam}: cameras/{cam}.npz has no frame_index, so --pts-from index (C2's direction) cannot place it on the video; "
                             f"{sp.take}: python -m dancecap.carlos_aline_inputs prepare --take {sp.take} --pts-from frames --pose-from people "
                             f"--teachers-floor")
        fi = np.asarray(z["frame_index"], int)
        z["pts"] = pts_all[fi].astype(np.float64)
    z["native_size"] = np.array([table[cam]["width"], table[cam]["height"]], np.int32)
    z["work_size"] = z["native_size"]
    dev = float(np.max(np.abs((table[cam]["slope"] * z["pts"] + table[cam]["offset"]) - z["t"])))
    if pose_from == "people":
        pp = people / f"{cam}.npz"
        if not pp.exists():
            raise FileNotFoundError(f"{pp} missing (the detect stage's people)")
        with np.load(pp) as pz:
            t_p = np.asarray(pz["t"], float) if "t" in pz.files else None
            n_p = len(pz["valid"])
        if n_p != len(z["t"]) or (t_p is not None and not _same_frames(t_p, z["t"])):
            raise ValueError(f"{cam}: people/{cam}.npz is not on the frames of cameras/{cam}.npz ({n_p} vs {len(z['t'])} rows, or t off by > 1 ms)")
        if len(fi) > 1 and np.any(np.diff(fi) <= 0):
            raise ValueError(f"{cam}: frame_index is not strictly increasing (steps {gates.get('frame_index_steps', {}).get('value')}): two canonical "
                             f"frames on one video frame, so the pose table has no row of their own")
    elif not POSE_SRC[cam].exists():
        raise FileNotFoundError(f"{POSE_SRC[cam]} missing (C2's podout pose file)")
    tt, floor_g = None, {}
    if sim3 is not None:
        tp = people / f"teachers_{cam}.npz"
        tt = _load(tp, "dancecap.dancers_seed's teachers")
        if not _same_frames(tt["t"], z["t"]):
            raise ValueError(f"{cam}: {tp.name} is not on the frames of cameras/{cam}.npz ({len(tt['t'])} vs {len(z['t'])} rows, or t off by > 1 ms)")
        ay = np.asarray(sp.ankle_y or ankle_heights({}), float)
        tt["frame_index"] = fi.astype(tt["frame_index"].dtype if "frame_index" in tt else np.int32)
        tt["floor"] = floor_xz(tt["kp"], z, sim3, ay)
        floor_g = floor_gates(tt["floor"], tt["row"])
        extra["floor_finite_lead_follow"] = [round(float(np.isfinite(tt["floor"][:, k, 0]).mean()), 3) for k in (0, 1)]
        extra["floor_plane_y_m"] = [float(v) for v in ay]
    return {"z": z, "fi": fi, "n_video": len(pts_all), "dev": dev, "gates": gates, "floor_gates": floor_g, "extra": extra, "teachers": tt}


def prepare(force: bool = False, take: str = DANCE, pts_from: str = "index", pose_from: str = "podout", teachers_floor: bool = False,
            clocks: str | None = None, sp: Spec | None = None, pts_fn=None, probe_fn=probe) -> dict:
    """Write recap_cams.json, the tracks' additive keys, the pose tables (and with teachers_floor the teachers' frame_index / floor) of a take ->
    people/inputs.json {cam: {...}}; the switches add <cam>.gates and cameras/<orig>/written.json. The defaults are carlos_aline's C3 run. Every
    camera is checked (_check) before the first write."""
    if pts_fn is None:
        from .cameras.newcam import video_pts as pts_fn
    sp = sp or spec(take, clocks)
    if pose_from == "podout" and sp.take != DANCE:
        raise SystemExit(f"--pose-from podout reads carlos_aline's room_work/podout pose files; {sp.take}: use --pose-from people")
    legacy = pts_from == "index" and pose_from == "podout" and not teachers_floor
    cdir = sp.work / "cameras"
    people = sp.work / "people"
    bak = cdir / sp.orig
    missing = [c for c in sp.cams if not (cdir / f"{c}.npz").exists() and not (bak / f"{c}.npz").exists()]
    if missing:
        raise FileNotFoundError(f"{sp.take}: no camera track cameras/{missing[0]}.npz (the camera stage first)")
    rec_p = bak / WRITTEN
    written = json.loads(rec_p.read_text()) if rec_p.exists() else {}
    frames = {cam: _frames_meta(sp.work, cam) for cam in sp.cams} if pts_from == "frames" else None
    table = cams_table(sp, frames, probe_fn)
    sim3 = None
    if teachers_floor:
        if not (cdir / "world_from_sfm.json").exists():
            raise FileNotFoundError(f"{cdir / 'world_from_sfm.json'} missing: --teachers-floor needs the camera stage's world")
        sim3 = json.loads((cdir / "world_from_sfm.json").read_text())
    fresh = {cam: _fresh(cdir / f"{cam}.npz", bak / f"{cam}.npz", written.get(cam)) for cam in sp.cams}
    checked = {cam: _check(sp, cam, table, frames, sim3, pts_from, pose_from, pts_fn, fresh[cam]) for cam in sp.cams}
    record = not legacy or rec_p.exists()                          # C3's run keeps its files as they were (no record), unless one exists
    bak.mkdir(exist_ok=True)
    (cdir / "recap_cams.json").write_text(json.dumps({"take": sp.take, "capture": sp.capture, "cams": table}, indent=1))
    res = {}
    for cam, c in checked.items():
        src = cdir / f"{cam}.npz"
        if fresh[cam]:
            shutil.copyfile(src, bak / f"{cam}.npz")
        np.savez_compressed(src, **c["z"])
        if record:                                                  # at once: a run cut short must not mistake its own track for a new one
            written[cam] = _sha1(src)
            rec_p.write_text(json.dumps(written, indent=1))
        fi, gates = c["fi"], dict(c["gates"])
        if pose_from == "people":
            ppl = _load(people / f"{cam}.npz", "the detect stage's people")
            pt = people_pose_table(ppl, fi, c["n_video"])
            same = pose_rows_match(pt, ppl, fi)
            gates["pose_rows_match_people"] = _gate(same, True, same)
        else:
            pt = pose_table(POSE_SRC[cam], c["n_video"])
        (people / "pose_tables").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(people / "pose_tables" / f"{cam}.npz", **pt)
        z = c["z"]
        res[cam] = {"frames": int(len(fi)), "t": [round(float(z["t"][0]), 3), round(float(z["t"][-1]), 3)],
                    "clock_vs_pts_max_dev_s": round(c["dev"], 5), "pose_frames": int(len(pt["n"])), "max_persons": int(pt["kp"].shape[1])}
        if c["teachers"] is not None:
            np.savez_compressed(people / f"teachers_{cam}.npz", **c["teachers"])
            gates.update(c["floor_gates"])
            if pose_from == "people":
                gates.update(teacher_row_gates(c["teachers"]["row"], ppl, pt, fi))
        if not legacy:
            gates = {"clock_max_dev_ms": _gate(round(c["dev"] * 1000, 4), CLOCK_TOL_S * 1000, c["dev"] <= CLOCK_TOL_S), **gates}
            res[cam].update({**c["extra"], "pose_from": pose_from, "gates": gates})
        log(f"{cam}: {res[cam]}")
    (people / "inputs.json").write_text(json.dumps(res, indent=1))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.carlos_aline_inputs", description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("prepare",))
    ap.add_argument("--take", default=DANCE, help="the take (default carlos_aline: C2's tracks and podout pose files, the clocks of sync/sync.json)")
    ap.add_argument("--pts-from", choices=("index", "frames"), default="index",
                    help="index: pts = the video's pts at the track's frame_index (C2); frames: frames/<cam>/frames.json pts, frame_index = that "
                         "video frame")
    ap.add_argument("--pose-from", choices=("podout", "people"), default="podout",
                    help="podout: C2's room_work/podout/pose_<A|B>.npz; people: the take's people/<cam>.npz on the video's frame numbers")
    ap.add_argument("--teachers-floor", action="store_true",
                    help="fill people/teachers_<cam>.npz frame_index and floor (each role's ankle ray at its ankle height above the floor y = 0)")
    ap.add_argument("--clocks", choices=("sync", "alignment"), default=None,
                    help="slope / offset: sync = work/<take>/sync/sync.json (carlos_aline's default), alignment = the take's clock (any other take's)")
    a = ap.parse_args(argv)
    res = prepare(take=a.take, pts_from=a.pts_from, pose_from=a.pose_from, teachers_floor=a.teachers_floor, clocks=a.clocks)
    failed = [(cam, g, v) for cam, r in res.items() for g, v in r.get("gates", {}).items() if not v["pass"]]
    for cam, g, v in failed:
        log(f"GATE FAILED {cam} {g}: {v['value']} (want {v['threshold']})")
    if any(r.get("gates") for r in res.values()):
        log("gates: " + ("all pass" if not failed else f"{len(failed)} failed"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
