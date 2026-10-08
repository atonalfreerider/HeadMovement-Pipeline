"""Source-phone video + per-frame camera tracks for the Unity film (film/INTERFACE.md section 5, VIEWER_SPEC 3.4).

The film director flies into a phone's point of view and shows that phone's ORIGINAL video full-frame behind the 3D.
For that Unity needs, per phone: the video (upright, trimmed to the capture, on the reference clock) and the phone's
pose + zoom for every frame (the tracks the cameras stage fitted). This module writes both, next to the capture.

    python -m dancecap.camera_video larissa_kadu_38 --capture 04_LarissaKaduDemo --cams 06,04 [--span 30.15 68.42]
    python -m dancecap.camera_video larissa_kadu_38 --capture 04_LarissaKaduDemo --direction auto   # cams from the film
    python -m dancecap.camera_video larissa_kadu_38 --capture 04_LarissaKaduDemo --install-only     # copy the master
    python -m dancecap.camera_video larissa_kadu_38 --capture 04_LarissaKaduDemo --tracks-only --no-install   # tracks + checks only

A new file in the Unity project's Assets folder is picked up by the editor's next asset refresh: when other jobs use
the editor, run with --no-install and mirror later (--install-only) while you hold the editor.

Output (master) work/<take>/camera_video/<capture>/ ; --install (default) mirrors it to head-movement/Assets/StreamingAssets/
<capture>/cameras/ (git-ignored; hard links when the volume allows):

  cameras.json         manifest (cameras, sizes, spans, applied sync deltas, container layout)
  <cam>.track.json     per-frame camera track: t (reference s, SYNC-CORRECTED), pos (Unity capture coordinates,
                       x y -z of the take's world; Unity adds DanceOrigin.Offset), fwd, up (unit), vfov (deg; spikes
                       removed), ok (0 = the cameras stage did not localise the frame: the pose is its own fill), plus the
                       pose policy Unity applies: max_unlocalised_s, max_sample_gap_s, fixed_position
  <cam>.full.hmj       the phone's frames, ROTATION BAKED (ffmpeg autorotate, as ingest), lens-undistorted when the
                       fitted radial term matters, long side <= --long-side, as JPEGs in one file (container below);
                       only for the cameras in --cams / the direction / the default set
  <cam>.thumb.hmj      the same at thumbnail size, every camera that has a track (frustum-glyph thumbnails)
  <cam>.mp4            the full frames re-timed to a constant 30 fps (H.264 baseline, yuv420p, faststart, no audio)

Clocks. A phone's frame at source time s has reference time slope*s + offset (take alignment); the per-phone residual
VIDEO offset of sync/offsets.json (tens of ms) is added exactly as the pose stages did ('corrected t = t + delta',
mamma_dyn.apply_offset), so the frame, the phone's pose and the dancers' poses all sit on the dance (reference)
clock the Unity capture plays (capture times.json). Frames keep their own time (the phones record variable frame
rate): the container stores one double per frame and Unity shows the frame nearest to the film's dance time, so a
film frame is never more than half a source frame (16.7 ms at 30 fps) from the image it shows.

Frames are never re-timed except in the .mp4 (constant 30 fps, nearest source frame per tick).

Gaps. The cameras stage localises most frames and FILLS the others itself (constant-velocity interpolation / smoothing,
ok=0 in the track). Every sample is exported; Unity uses the filled ones up to a run length per phone (max_unlocalised_s:
3 s, 8 s for a phone that never moves) and between rows at most max_sample_gap_s apart (1 s: else the phone was not
recording). The first version DROPPED the ok=0 rows and refused to interpolate across more than 0.6 s: one 17-frame run of
phone 06 left a 0.6000023 s hole (greater than 0.6f in float32) in the middle of the only POV shot, and the video blinked
out for 18 film frames. After an export the direction's POV windows are checked at every 1/30 s (a pose during the whole
shot, a video frame within 0.25 s while the video shows): the log and cameras.json (pov_checks) say what would be missing.

.hmj container (little endian): JPEG payloads back to back, then a table of count records
  (uint64 offset, uint32 length, float64 reference time) and a 24-byte trailer
  'HMJ1', uint32 count, uint64 table offset, uint32 width, uint32 height.
Unity reads the trailer, the table, and decodes single frames on demand (ImageConversion.LoadImage): random access,
no codec, exactly reproducible under the Recorder's frame-locked clock.

Several takes can be merged (later takes win where their tracks overlap): the full dance is assembled from segment
takes whose phones' tracks live in their own work folders.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import ingest
from . import mamma_dyn as md
from .take import PACKAGE_ROOT, WORKSPACE, Take

UNITY_STREAMING = WORKSPACE / "head-movement" / "Assets" / "StreamingAssets"
DEFAULT_FULL_CAMS = ("06", "04")    # added to the cameras the film direction uses (06: sharp tripod-like, 04: zoomer)
FPS = 30.0
UNDISTORT_MIN_K = 0.004             # |k1| below this moves the image edge by < 1.5 px at 1080p: left alone
MAGIC = b"HMJ1"
TRAILER = struct.Struct("<4sIQII")  # magic, count, table offset, width, height
RECORD = struct.Struct("<QId")      # offset, length, time
BELOW_NORMAL = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
PRIORITY = {"flags": BELOW_NORMAL, "name": "below_normal"}   # set by --priority (below-normal starves on a saturated PC)

# What Unity may show (the values travel in every <cam>.track.json; CameraVideoTrack.TryAt applies them, and
# PoseCoverage below is its Python mirror for the checks). One rule for the pose, the video layer and the glyphs, so
# TryGetPose and SetVideo cannot disagree (the 0.6 s dropout of the first version came from two different rules).
MAX_SAMPLE_GAP_S = 1.0          # rows further apart than this: the phone was not recording, no pose between them
MAX_UNLOCALISED_S = 3.0         # a run of frames the cameras stage did NOT localise (it fills them itself by constant-velocity
                                # interpolation / smoothing, ok=False) is trusted up to this long, between its localised neighbours
MAX_UNLOCALISED_FIXED_S = 8.0   # same for phones whose position never moves (tripod: only the rotation is filled)
FIXED_POSITION_M = 0.01         # position spread of the localised frames below which a phone counts as fixed
MAX_FRAME_GAP_S = 0.25          # Unity hides the video when the nearest strip frame is further than this from the dance time
VFOV_SPIKE_DEG = 3.0            # a vertical FOV sample further than this from the running median is an estimation spike
VFOV_MEDIAN_S = 0.6             # ... median window (a real zoom is monotonic and passes a median filter; a spike up to ~0.3 s does not)
REPORT_GAP_S = 0.3              # unlocalised runs longer than this are reported when they touch a direction's POV window
OPTIONS = {"allow_unsynced": False}


# ---------------------------------------------------------------------------------------------------- tracks

@dataclass
class CamTrack:
    """One phone's per-frame pose + zoom on the corrected reference clock, Unity capture coordinates."""
    id: str
    raw_t: np.ndarray          # (F,) reference s before the video offset
    t: np.ndarray              # (F,) corrected reference s (raw_t + delta)
    delta: np.ndarray          # (F,) video offset applied (s)
    pos: np.ndarray            # (F, 3) camera centre, Unity capture coordinates
    fwd: np.ndarray            # (F, 3) optical axis, unit
    up: np.ndarray             # (F, 3) image-up, unit
    vfov: np.ndarray           # (F,) vertical field of view, degrees
    f_px: np.ndarray           # (F,) focal length, px of the native frame
    k1: np.ndarray             # (F,) COLMAP SIMPLE_RADIAL term
    ok: np.ndarray             # (F,) the cameras stage localised this frame
    size: tuple[int, int]      # native (upright) frame size from the intrinsics
    takes: list[str] = field(default_factory=list)
    vfov_spikes: int = 0       # vertical-FOV samples replaced by despike_vfov

    def __len__(self) -> int:
        return len(self.t)

    def nearest(self, t: np.ndarray, raw: bool = False) -> np.ndarray:
        """Index of the sample nearest to t (corrected reference s, or raw reference s)."""
        ts = self.raw_t if raw else self.t
        if len(ts) == 1:
            return np.zeros(len(t), int)
        i = np.clip(np.searchsorted(ts, t), 1, len(ts) - 1)
        return np.where(np.abs(ts[i - 1] - t) <= np.abs(ts[i] - t), i - 1, i)


def cam_pose_unity(R_w: np.ndarray, t_w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """World-to-camera (x_cam = R_w X + t_w, OpenCV: x right, y down, z forward; world right-handed Y-up) ->
    camera centre, optical axis and image-up in UNITY capture coordinates (x, y, -z)."""
    C = -np.einsum("fji,fj->fi", R_w, t_w)
    fwd, up = R_w[:, 2, :], -R_w[:, 1, :]
    flip = np.array([1.0, 1.0, -1.0])
    return C * flip, fwd * flip, up * flip


def require_sync(take_name: str) -> dict[str, dict]:
    """The take's per-phone video offsets (sync/offsets.json), or a hard error: exporting without them silently mis-times
    the phones by tens of ms (phone 04: 3 frames). --allow-unsynced exports on the raw clocks, loudly."""
    work = md.take_work(take_name)
    offs = md.read_sync_offsets(work)          # refuses offsets measured on other audio clocks
    if not offs:
        if not OPTIONS["allow_unsynced"]:
            raise SystemExit(
                f"[camera_video] {work / 'sync' / 'offsets.json'} is missing or empty: the phones' video offsets (tens of ms) "
                f"cannot be applied, so every phone would be mis-timed against the dancers. Run 'python -m dancecap sync "
                f"{take_name}' first, or pass --allow-unsynced to export on the raw clocks anyway.")
        print(f"[camera_video] WARNING {take_name}: NO video offsets (--allow-unsynced): every phone is exported on its raw clock "
              "and may be off by up to ~100 ms")
    return offs


def load_track(take_name: str, cam: str, offs: dict | None = None) -> CamTrack | None:
    work = md.take_work(take_name)
    npz = work / "cameras" / f"{cam}.npz"
    sim3_p = work / "cameras" / "world_from_sfm.json"
    if not npz.exists():
        return None
    if not sim3_p.exists():
        raise SystemExit(f"[camera_video] {sim3_p} missing: the cameras are still in the SfM frame (not metric)")
    sim3 = json.loads(sim3_p.read_text())
    if offs is None:
        offs = require_sync(take_name)
    z = np.load(npz)
    raw_t = np.asarray(z["t"], float)
    off = offs.get(cam)
    if off is None and offs:
        print(f"[camera_video] WARNING {take_name}/{cam}: no entry in sync/offsets.json: delta 0 (raw clock)")
    t = md.apply_offset(raw_t, off)
    R_w, t_w = md.world_cameras(np.asarray(z["R"], float), np.asarray(z["tvec"], float), sim3)
    pos, fwd, up = cam_pose_unity(R_w, t_w)
    K = np.asarray(z["K"], float)
    W, H = int(round(2 * K[0, 0, 2] + 1)), int(round(2 * K[0, 1, 2] + 1))
    f = K[:, 1, 1]
    hh = 2 * K[:, 1, 2] + 1                    # per-frame frame height (px) the intrinsics refer to
    vfov = np.degrees(2 * np.arctan(hh / (2 * f)))
    F = len(raw_t)
    k1 = np.asarray(z["dist"], float).reshape(F, -1)[:, 0] if "dist" in z.files else np.zeros(F)
    ok = np.asarray(z["ok"], bool) if "ok" in z.files else np.ones(F, bool)
    return CamTrack(cam, raw_t, t, t - raw_t, pos, fwd, up, vfov, f, k1, ok, (W, H), [take_name])


def merge_tracks(tracks: list[CamTrack]) -> CamTrack:
    """Later tracks replace earlier ones inside their own time range."""
    if len(tracks) == 1:
        return tracks[0]
    keep = []
    for i, tr in enumerate(tracks):
        m = np.ones(len(tr), bool)
        for later in tracks[i + 1:]:
            m &= ~((tr.raw_t >= later.raw_t[0] - 1e-6) & (tr.raw_t <= later.raw_t[-1] + 1e-6))
        keep.append(m)
    cols = {n: np.concatenate([getattr(tr, n)[m] for tr, m in zip(tracks, keep)])
            for n in ("raw_t", "t", "delta", "pos", "fwd", "up", "vfov", "f_px", "k1", "ok")}
    order = np.argsort(cols["raw_t"], kind="stable")
    out = {n: v[order] for n, v in cols.items()}
    return CamTrack(tracks[0].id, size=tracks[0].size, takes=[n for tr in tracks for n in tr.takes], **out)


def overlap_report(tracks: list[CamTrack]) -> list[dict]:
    """Where two takes' tracks of one phone overlap in time they must describe the same camera in the same world (the
    segment takes reuse the 38 s take's map): position (m) and viewing-direction (deg) differences, a guard against
    stitching tracks of different world frames (the later take wins in the overlap)."""
    out = []
    for i, a in enumerate(tracks):
        for b in tracks[i + 1:]:
            lo, hi = max(a.t[0], b.t[0]), min(a.t[-1], b.t[-1])
            if hi <= lo:
                continue
            m = (a.t >= lo) & (a.t <= hi)
            j = b.nearest(a.t[m])
            dp = np.linalg.norm(a.pos[m] - b.pos[j], axis=1)
            ang = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", a.fwd[m], b.fwd[j]), -1, 1)))
            out.append({"takes": [a.takes[0], b.takes[0]], "span": [round(float(lo), 2), round(float(hi), 2)],
                        "position_m": {"median": round(float(np.median(dp)), 3), "max": round(float(dp.max()), 3)},
                        "direction_deg": {"median": round(float(np.median(ang)), 2), "max": round(float(ang.max()), 2)}})
    return out


def despike_vfov(t: np.ndarray, vfov: np.ndarray) -> tuple[np.ndarray, int]:
    """Replace vertical-FOV samples that sit more than VFOV_SPIKE_DEG from the running median (window VFOV_MEDIAN_S) by that
    median. The focal length is estimated per frame and occasionally jumps for a few frames (phone 07: 54 -> 70 -> 54 deg
    within 0.25 s); flown into, that would pop the video quad's scale and the skeleton overlay. A real zoom is a ramp or a
    step that lasts: it passes the median filter unchanged."""
    t = np.asarray(t, float)
    v = np.asarray(vfov, float)
    if len(v) < 5:
        return v.copy(), 0
    lo = np.searchsorted(t, t - VFOV_MEDIAN_S / 2, side="left")
    hi = np.searchsorted(t, t + VFOV_MEDIAN_S / 2, side="right")
    med = np.array([np.median(v[a:b]) for a, b in zip(lo, hi)])
    bad = np.abs(v - med) > VFOV_SPIKE_DEG
    out = np.where(bad, med, v)
    return out, int(bad.sum())


def unlocalised_runs(t: np.ndarray, ok: np.ndarray) -> list[tuple[int, int, float]]:
    """Runs of samples the cameras stage did not localise (ok=False): (first index, last index, seconds between the localised
    samples that bound the run). The poses inside a run are the cameras stage's own fill (constant-velocity interpolation /
    smoothing); the longer the run, the less they can be trusted."""
    bad = ~np.asarray(ok, bool)
    n = len(t)
    runs, i = [], 0
    while i < n:
        if not bad[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and bad[j + 1]:
            j += 1
        t_prev = t[i - 1] if i > 0 else t[i]
        t_next = t[j + 1] if j + 1 < n else t[j]
        runs.append((i, j, float(t_next - t_prev)))
        i = j + 1
    return runs


def is_fixed_position(pos: np.ndarray, ok: np.ndarray) -> bool:
    """The camera centre never moves (a tripod / stand: only the rotation changes)."""
    g = np.asarray(ok, bool)
    if g.sum() < 2:
        g = np.ones(len(pos), bool)
    return bool(np.ptp(pos[g], axis=0).max() < FIXED_POSITION_M)


def track_policy(t: np.ndarray, pos: np.ndarray, ok: np.ndarray) -> dict:
    fixed = is_fixed_position(pos, ok)
    return {"fixed_position": fixed, "max_unlocalised_s": MAX_UNLOCALISED_FIXED_S if fixed else MAX_UNLOCALISED_S,
            "max_sample_gap_s": MAX_SAMPLE_GAP_S}


class PoseCoverage:
    """Python mirror of CameraVideoTrack.TryAt's availability rule: is there a pose at dance time q?

    inside [t0 - tol, t1 + tol] (the end poses are held within tol) and, between two samples, only when they are at most
    max_sample_gap apart and neither belongs to an unlocalised run longer than max_unlocalised."""

    def __init__(self, t, ok, max_unlocalised: float, max_sample_gap: float = MAX_SAMPLE_GAP_S, tol: float = 0.5):
        self.t = np.asarray(t, float)
        ok = np.asarray(ok, bool)
        if not ok.any():                     # nothing localised: Unity keeps every sample rather than lose the phone
            ok = np.ones(len(self.t), bool)
        self.run = np.zeros(len(self.t))
        for i, j, g in unlocalised_runs(self.t, ok):
            self.run[i:j + 1] = g
        self.max_unloc, self.max_gap, self.tol = float(max_unlocalised), float(max_sample_gap), float(tol)

    @classmethod
    def from_json(cls, d: dict) -> "PoseCoverage":
        ok = np.asarray(d["ok"], bool) if "ok" in d else np.ones(len(d["t"]), bool)
        return cls(d["t"], ok, d.get("max_unlocalised_s", MAX_UNLOCALISED_S), d.get("max_sample_gap_s", MAX_SAMPLE_GAP_S))

    def available(self, q) -> np.ndarray:
        q = np.atleast_1d(np.asarray(q, float))
        t, n = self.t, len(self.t)
        out = np.zeros(len(q), bool)
        if n == 0:
            return out
        inside = (q >= t[0] - self.tol) & (q <= t[-1] + self.tol)
        head, tail = q <= t[0], q >= t[-1]
        out[inside & head] = self.run[0] <= self.max_unloc
        out[inside & tail & ~head] = self.run[-1] <= self.max_unloc
        mid = inside & ~head & ~tail
        if mid.any() and n > 1:
            hi = np.clip(np.searchsorted(t, q[mid], side="right"), 1, n - 1)
            lo = hi - 1
            out[mid] = ((t[hi] - t[lo]) <= self.max_gap + 1e-3) & (np.maximum(self.run[lo], self.run[hi]) <= self.max_unloc + 1e-3)
        return out


def intervals(t: np.ndarray, bad: np.ndarray, step: float) -> list[list[float]]:
    """Merge the grid times t[bad] into [start, end] intervals (end = last bad time + step)."""
    out: list[list[float]] = []
    for ti, b in zip(t, bad):
        if not b:
            continue
        if out and ti - out[-1][1] <= step * 1.5:
            out[-1][1] = round(float(ti) + step, 3)
        else:
            out.append([round(float(ti), 3), round(float(ti) + step, 3)])
    return out


def track_json(tr: CamTrack, t0: float, t1: float) -> dict:
    """Columnar JSON of the samples inside [t0, t1] (corrected reference s). Every sample is kept, localised or not
    (ok 0 = the cameras stage filled it); the policy fields tell Unity how far to trust the filled runs."""
    m = (tr.t >= t0) & (tr.t <= t1)
    r = lambda a, n: np.round(a[m], n).tolist()  # noqa: E731
    pol = track_policy(tr.t[m], tr.pos[m], tr.ok[m])
    return {"id": tr.id, "size": list(tr.size), "frame": "unity capture (x, y, -z of the take's world); add DanceOrigin.Offset",
            "time_base": "reference s, sync-corrected (t = raw + video offset)", **pol,
            "n": int(m.sum()), "t": r(tr.t, 4), "pos": r(tr.pos, 4), "fwd": r(tr.fwd, 5), "up": r(tr.up, 5),
            "vfov": r(tr.vfov, 3), "ok": tr.ok[m].astype(int).tolist()}


def pose_coverage_entry(tj: dict, step: float = 1.0 / 30.0) -> dict:
    """Manifest facts about one exported track: the unlocalised runs, what Unity can answer and where it cannot."""
    t = np.asarray(tj["t"], float)
    ok = np.asarray(tj["ok"], bool)
    runs = unlocalised_runs(t, ok)
    cov = PoseCoverage.from_json(tj)
    grid = np.arange(t[0], t[-1], step)
    no_pose = intervals(grid, ~cov.available(grid), step)
    longest = max((g for _, _, g in runs), default=0.0)
    dts = np.diff(t)
    return {
        "fixed_position": tj["fixed_position"], "max_unlocalised_s": tj["max_unlocalised_s"],
        "unlocalised_frames": int((~ok).sum()), "unlocalised_runs": len(runs),
        "longest_unlocalised_s": round(float(longest), 3),
        "unlocalised_runs_over_0.3s": [[round(float(t[i]), 3), round(float(t[j]), 3), round(g, 3)] for i, j, g in runs if g > REPORT_GAP_S],
        "longest_sample_gap_s": round(float(dts.max()) if len(dts) else 0.0, 4),
        "no_pose_intervals": no_pose, "pose_coverage": round(float(cov.available(grid).mean()), 5) if len(grid) else 1.0,
    }


# ---------------------------------------------------------------------------------------------------- decoding

def _ffmpeg_run(args: list[str], **kw) -> subprocess.Popen:
    return subprocess.Popen(["ffmpeg", *args], creationflags=PRIORITY["flags"], **kw)


def probe_pts(src, start: float, end: float) -> list[float]:
    """Source PTS (s) of every frame ffmpeg delivers for the range (same command as the decode)."""
    seek = max(0.0, start - 1.0)
    cmd = ["ffmpeg", "-v", "info", "-nostats", "-ss", f"{seek:.6f}", "-t", f"{end - seek + 0.5:.6f}", "-copyts", "-threads", "2", "-i",
           str(src), "-map", "0:v:0", "-vf", "showinfo", *ingest.passthrough_args(), "-f", "null", "-"]
    r = subprocess.run(cmd, capture_output=True, creationflags=PRIORITY["flags"])
    tb = ingest.probe(Path(src)).get("time_base")
    return [ingest.showinfo_pts(m, tb) for m in ingest.SHOWINFO.finditer(r.stderr.decode("utf-8", "replace"))]


def iter_frames(src, start: float, end: float, w: int, h: int, n_expected: int):
    """Upright BGR frames of the range, in presentation order (the same sequence probe_pts timestamps)."""
    seek = max(0.0, start - 1.0)
    proc = _ffmpeg_run(["-v", "error", "-nostats", "-ss", f"{seek:.6f}", "-t", f"{end - seek + 0.5:.6f}", "-copyts", "-threads", "2", "-i",
                        str(src), "-map", "0:v:0", *ingest.passthrough_args(), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                       stdout=subprocess.PIPE)
    n = 0
    try:
        while n < n_expected:
            buf = proc.stdout.read(w * h * 3)
            if len(buf) < w * h * 3:
                break
            yield n, np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            n += 1
    finally:
        proc.stdout.close()
        if proc.poll() is None:
            proc.kill()
        proc.wait()


class Undistorter:
    """Removes COLMAP SIMPLE_RADIAL distortion (x_d = x_u (1 + k r_u^2), pinhole-normalised) with the intrinsics'
    principal point at the frame centre: output pixel -> source pixel is closed form (no inversion)."""

    def __init__(self, w: int, h: int):
        self.w, self.h = w, h
        self.cache: dict[tuple[float, int], tuple[np.ndarray, np.ndarray]] = {}

    def __call__(self, img: np.ndarray, f: float, k1: float) -> np.ndarray:
        key = (round(float(k1), 3), int(round(f / 4.0)))
        if key not in self.cache:
            k, ff = key[0], key[1] * 4.0
            cx, cy = (self.w - 1) / 2.0, (self.h - 1) / 2.0
            x = (np.arange(self.w, dtype=np.float32) - cx) / ff
            y = (np.arange(self.h, dtype=np.float32) - cy) / ff
            r2 = x[None, :] ** 2 + y[:, None] ** 2
            fac = 1.0 + k * r2
            self.cache[key] = ((cx + ff * x[None, :] * fac).astype(np.float32), (cy + ff * y[:, None] * fac).astype(np.float32))
            if len(self.cache) > 64:
                self.cache.pop(next(iter(self.cache)))
        mx, my = self.cache[key]
        return cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def fit_size(w: int, h: int, long_side: int) -> tuple[int, int]:
    """Even output size with the long side capped (never upscaled)."""
    s = min(1.0, long_side / max(w, h))
    return max(2, int(round(w * s / 2)) * 2), max(2, int(round(h * s / 2)) * 2)


class HmjWriter:
    def __init__(self, path: Path, w: int, h: int):
        self.path, self.w, self.h = path, w, h
        self.f = open(path, "wb")
        self.rec: list[tuple[int, int, float]] = []

    def add(self, jpeg: bytes, t: float) -> None:
        self.rec.append((self.f.tell(), len(jpeg), float(t)))
        self.f.write(jpeg)

    def close(self) -> None:
        table = self.f.tell()
        for r in self.rec:
            self.f.write(RECORD.pack(*r))
        self.f.write(TRAILER.pack(MAGIC, len(self.rec), table, self.w, self.h))
        self.f.close()


def read_hmj(path: Path) -> tuple[list[tuple[int, int, float]], int, int]:
    with open(path, "rb") as f:
        f.seek(-TRAILER.size, os.SEEK_END)
        magic, n, table, w, h = TRAILER.unpack(f.read(TRAILER.size))
        if magic != MAGIC:
            raise ValueError(f"{path}: not an HMJ1 file")
        f.seek(table)
        return [RECORD.unpack(f.read(RECORD.size)) for _ in range(n)], w, h


def hmj_frame(path: Path, rec: tuple[int, int, float]) -> np.ndarray:
    with open(path, "rb") as f:
        f.seek(rec[0])
        return cv2.imdecode(np.frombuffer(f.read(rec[1]), np.uint8), cv2.IMREAD_COLOR)


def export_camera(cam: str, takes: list[str], track: CamTrack, span: tuple[float, float], out: Path, *, full: bool,
                  thumbs: bool, long_side: int, quality: int, thumb_side: int, thumb_quality: int, log=print) -> dict:
    """Decode one phone's source video over the span and write its strips; returns the manifest entry."""
    srcs = [Take.load(t).source(cam) for t in takes if any(s.id == cam for s in Take.load(t).sources)]
    if not srcs:
        raise SystemExit(f"[camera_video] camera {cam}: no source video in {takes}")
    src = srcs[-1]
    info = ingest.probe(src.path)
    w, h = info["width"], info["height"]
    if (w, h) != tuple(track.size):
        log(f"[camera_video] {cam}: WARNING decoded frame {w}x{h} differs from the intrinsics' {track.size[0]}x{track.size[1]}")
    s0, s1 = span
    # the video offset is tens of ms: decode a little wider, filter on the corrected time after
    ref0, ref1 = s0 - 0.5, s1 + 0.5
    start, end = src.to_source(ref0), src.to_source(ref1)
    pts = probe_pts(src.path, start, end)
    if not pts:
        raise SystemExit(f"[camera_video] camera {cam}: no frames in {start:.2f}..{end:.2f} s of the source")
    pts = np.asarray(pts, float)
    ref = src.slope * pts + src.offset
    delta = track.delta[track.nearest(ref, raw=True)]
    tcorr = ref + delta
    inside = (tcorr >= s0) & (tcorr <= s1) & (pts >= start - 1e-6) & (pts <= end + 1e-6)
    every = max(1, int(round(info["avg_fps"] / FPS))) if info["avg_fps"] > 1.5 * FPS else 1
    idx = np.nonzero(inside)[0]
    keep = np.zeros(len(pts), bool)
    keep[idx[::every]] = True
    if keep.sum() < 2:
        raise SystemExit(f"[camera_video] camera {cam}: {keep.sum()} frames inside the span {s0:.2f}..{s1:.2f}")
    fw, fh = fit_size(w, h, long_side)
    tw, th = fit_size(w, h, thumb_side)
    undist = Undistorter(w, h)
    j = track.nearest(tcorr)
    near = {"f": track.f_px[j], "k1": track.k1[j], "delta": delta}
    writers = {}
    if full:
        writers["full"] = HmjWriter(out / f"{cam}.full.hmj", fw, fh)
    if thumbs:
        writers["thumb"] = HmjWriter(out / f"{cam}.thumb.hmj", tw, th)
    undistorted = 0
    t_start = time.time()
    read = 0
    for n, img in iter_frames(src.path, start, end, w, h, len(pts)):
        read = n + 1
        if not keep[n]:
            continue
        k1, f = float(near["k1"][n]), float(near["f"][n])
        if abs(k1) >= UNDISTORT_MIN_K:
            img = undist(img, f, k1)
            undistorted += 1
        t = float(tcorr[n])
        if "full" in writers:
            small = cv2.resize(img, (fw, fh), interpolation=cv2.INTER_AREA) if (fw, fh) != (w, h) else img
            writers["full"].add(cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes(), t)
        if "thumb" in writers:
            small = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
            writers["thumb"].add(cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, thumb_quality])[1].tobytes(), t)
    for wr in writers.values():
        wr.close()
    if read != len(pts):
        raise SystemExit(f"[camera_video] camera {cam}: decoded {read} frames but showinfo timed {len(pts)}")
    nk = int(keep.sum())
    dt = np.diff(tcorr[keep])
    entry = {
        "id": cam, "native_size": [w, h], "rotation_baked": info["rotation"], "source_fps": round(info["avg_fps"], 3),
        "decimate_every": every, "frames": nk, "t0": round(float(tcorr[keep][0]), 4), "t1": round(float(tcorr[keep][-1]), 4),
        "frame_interval_ms": {"median": round(float(np.median(dt)) * 1000, 2), "min": round(float(dt.min()) * 1000, 2),
                              "max": round(float(dt.max()) * 1000, 2)},
        "video_offset_ms": round(float(np.median(near["delta"])) * 1000, 2),
        "undistorted_frames": undistorted, "k1_median": round(float(np.median(near["k1"])), 5),
        "vfov_deg": [round(float(track.vfov.min()), 2), round(float(np.median(track.vfov)), 2), round(float(track.vfov.max()), 2)],
        "aspect": round(w / h, 5),
    }
    if full:
        entry["full"] = {"file": f"{cam}.full.hmj", "size": [fw, fh], "bytes": (out / f"{cam}.full.hmj").stat().st_size}
    if thumbs:
        entry["thumb"] = {"file": f"{cam}.thumb.hmj", "size": [tw, th], "bytes": (out / f"{cam}.thumb.hmj").stat().st_size}
    log(f"[camera_video] {cam}: {nk} frames {w}x{h}{' (rot %d)' % info['rotation'] if info['rotation'] else ''} -> "
        f"{fw}x{fh} / {tw}x{th}, video offset {entry['video_offset_ms']:+.1f} ms, undistorted {undistorted}, "
        f"{time.time() - t_start:.1f} s")
    return entry


def write_mp4(cam: str, out: Path, entry: dict, log=print) -> None:
    """The full strip re-timed to a constant 30 fps (nearest source frame per tick): H.264 baseline / yuv420p / faststart."""
    recs, w, h = read_hmj(out / entry["full"]["file"])
    t = np.array([r[2] for r in recs])
    t0 = float(t[0])
    n = int(np.floor((t[-1] - t0) * FPS)) + 1
    grid = t0 + np.arange(n) / FPS
    j = np.clip(np.searchsorted(t, grid), 1, len(t) - 1)
    j = np.where(np.abs(t[j - 1] - grid) <= np.abs(t[j] - grid), j - 1, j)
    dst = out / f"{cam}.mp4"
    tmp = out / f"{cam}.mp4.part"
    proc = _ffmpeg_run(["-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{FPS:g}",
                        "-i", "-", "-an", "-threads", "2", "-vf", "scale=out_color_matrix=bt709", "-c:v", "libx264", "-preset", "medium", "-crf", "22", "-profile:v", "baseline",
                        "-level", "4.0", "-pix_fmt", "yuv420p", "-x264-params", "colorprim=bt709:transfer=bt709:colormatrix=bt709", "-movflags", "+faststart", "-f", "mp4", str(tmp)], stdin=subprocess.PIPE)
    last, img = -1, None
    with open(out / entry["full"]["file"], "rb") as f:
        for k in range(n):
            if j[k] != last:
                f.seek(recs[j[k]][0])
                img = cv2.imdecode(np.frombuffer(f.read(recs[j[k]][1]), np.uint8), cv2.IMREAD_COLOR)
                last = j[k]
            proc.stdin.write(img.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit(f"[camera_video] ffmpeg failed encoding {dst}")
    os.replace(tmp, dst)                       # a new file: installed hard links keep the old one until re-installed
    entry["mp4"] = {"file": dst.name, "fps": FPS, "t0": round(t0, 4), "frames": n, "bytes": dst.stat().st_size,
                    "profile": "baseline", "note": "constant 30 fps: frame k shows the source frame nearest to t0 + k/30"}
    log(f"[camera_video] {cam}: mp4 {n} frames, {dst.stat().st_size / 1e6:.1f} MB")


# ---------------------------------------------------------------------------------------------------- discovery / install

def capture_span(capture: str, tracks: dict[str, CamTrack]) -> tuple[float, float]:
    p = UNITY_STREAMING / capture / "times.json"
    if p.exists():
        t = json.loads(p.read_text())
        if isinstance(t, list) and t:
            return float(t[0]), float(t[-1])
    t0 = max(float(tr.t[0]) for tr in tracks.values())
    t1 = min(float(tr.t[-1]) for tr in tracks.values())
    return t0, t1


def direction_files(capture: str, paths: list[str] | None) -> list[Path]:
    if paths and paths != ["auto"]:
        return [Path(p) for p in paths]
    files: list[Path] = []
    pats = [UNITY_STREAMING / capture / "film", *sorted((PACKAGE_ROOT / "work").glob("*/film"))]
    for d in pats:
        for p in sorted(d.glob("direction*.json")) if d.is_dir() else []:
            try:
                if json.loads(p.read_text(encoding="utf-8")).get("capture") == capture:
                    files.append(p)
            except (OSError, ValueError):
                pass
    return files


def _dance_at(seg: dict, f: float) -> float:
    """Dance time of a direction segment at film time f (constant speed, linear speed ramp, hold): the director's
    FilmSegment.RawDance without the clock's smoothing."""
    f0, f1 = seg["film"]
    d0, d1 = seg["dance"]
    dur = f1 - f0
    if dur <= 1e-5 or abs(d1 - d0) < 1e-6:
        return float(d0)
    u = min(1.0, max(0.0, (f - f0) / dur))
    sp = seg.get("speed", 1.0)
    if isinstance(sp, (list, tuple)) and len(sp) == 2 and abs(sp[1] - sp[0]) > 1e-4:
        mean = (sp[0] + sp[1]) / 2
        if mean > 1e-5:
            u = (sp[0] * u + (sp[1] - sp[0]) * u * u / 2) / mean
    return float(d0 + (d1 - d0) * u)


def direction_pov_windows(files: list[Path]) -> list[dict]:
    """Every shot that flies into / sits at a phone: {"camera", "segment", "file", "pose": [d0, d1], "video": [a, b]} in dance
    seconds. The director asks for the phone's POSE during the whole shot (the flight in, the hold, the pull back) and shows
    the VIDEO only while PovCamera is set (after fly_s, for fade_in + hold + fade_out); a camera_tour entry is both at once."""
    out: list[dict] = []
    pad = 0.3                                    # the clock smooths the film -> dance map a little
    for p in files:
        d = json.loads(p.read_text(encoding="utf-8"))
        for seg in d.get("segments", []):
            cam = seg.get("camera") or {}
            mode = cam.get("mode")
            if mode == "fly_to_camera" and cam.get("camera") and seg.get("film") and seg.get("dance"):
                f0, f1 = seg["film"]
                fly, fin, hold, fout = (float(cam.get(k, v)) for k, v in
                                        (("fly_s", 1.5), ("fade_in_s", 0.8), ("hold_s", 6.5), ("fade_out_s", 0.8)))
                v0, v1 = f0 + fly, min(f1, f0 + fly + fin + hold + fout)
                out.append({"camera": str(cam["camera"]), "segment": seg.get("id"), "file": p.name,
                            "pose": [round(_dance_at(seg, f0) - pad, 3), round(_dance_at(seg, f1) + pad, 3)],
                            "video": [round(_dance_at(seg, v0) - pad, 3), round(_dance_at(seg, v1) + pad, 3)]})
            elif mode == "camera_tour":
                for c in cam.get("cameras", []):
                    if c.get("camera") and isinstance(c.get("dance_t"), list) and len(c["dance_t"]) == 2:
                        w = [float(c["dance_t"][0]), float(c["dance_t"][1])]
                        out.append({"camera": str(c["camera"]), "segment": seg.get("id"), "file": p.name, "pose": w, "video": w})
    return out


def direction_cameras(capture: str, paths: list[str] | None) -> tuple[list[str], list[str]]:
    """Camera ids a film direction flies into (fly_to_camera, camera_tour) -> (ids, direction files read)."""
    files = direction_files(capture, paths)
    ids = [w["camera"] for w in direction_pov_windows(files)]
    return sorted(set(ids)), [str(p) for p in files]


def coverage_report(master: Path, windows: list[dict], step: float = 1.0 / 30.0) -> list[dict]:
    """For every POV window: at each 1/30 s step the phone must have a POSE (Unity's TryGetPose, via the PoseCoverage mirror
    of TryAt on the exported track) during the whole shot, and a video FRAME within MAX_FRAME_GAP_S of the time while the
    video is shown. Returns one record per window with the missing intervals; ok = nothing missing."""
    out = []
    cache: dict[str, tuple] = {}
    for w in windows:
        cam = w["camera"]
        if cam not in cache:
            tp, hp = master / f"{cam}.track.json", master / f"{cam}.full.hmj"
            tj = json.loads(tp.read_text(encoding="utf-8")) if tp.exists() else None
            ft = np.array([r[2] for r in read_hmj(hp)[0]]) if hp.exists() else None
            cache[cam] = (PoseCoverage.from_json(tj) if tj else None, ft, tj)
        cov, ft, tj = cache[cam]
        rec = {k: w[k] for k in ("camera", "segment", "file", "pose", "video")}
        problems = []
        if cov is None:
            problems.append("no track in the export")
            rec["pose_missing"] = [w["pose"]]
        else:
            g = np.arange(w["pose"][0], w["pose"][1] + 1e-9, step)
            rec["pose_missing"] = intervals(g, ~cov.available(g), step)
            if rec["pose_missing"]:
                problems.append(f"no pose at {rec['pose_missing']}")
            runs = [r for r in unlocalised_runs(cov.t, np.asarray(tj["ok"], bool))
                    if r[2] > REPORT_GAP_S and cov.t[r[1]] >= w["pose"][0] and cov.t[r[0]] <= w["pose"][1]]
            rec["unlocalised_runs_in_window"] = [[round(float(cov.t[i]), 3), round(float(cov.t[j]), 3), round(gs, 3)] for i, j, gs in runs]
        if ft is None:
            problems.append("no full-frame video for this phone in the export (--cams)")
            rec["video_missing"] = [w["video"]]
        else:
            g = np.arange(w["video"][0], w["video"][1] + 1e-9, step)
            near = np.abs(g[:, None] - ft[None, :]).min(axis=1) if len(ft) else np.full(len(g), np.inf)
            rec["video_missing"] = intervals(g, near > MAX_FRAME_GAP_S, step)
            if rec["video_missing"]:
                problems.append(f"no video frame at {rec['video_missing']}")
        rec["ok"] = not problems
        rec["problems"] = problems
        out.append(rec)
    return out


def log_coverage(rep: list[dict], log=print) -> bool:
    ok = True
    for r in rep:
        tag = f"{r['file']} {r['segment']} phone {r['camera']} (pose {r['pose']}, video {r['video']})"
        if r["ok"]:
            note = r.get("unlocalised_runs_in_window")
            if note:
                log(f"[camera_video] POV check WARN {tag}: the phone has frames the cameras stage did not localise inside the window "
                    f"(start, end, seconds between the localised neighbours) {note}; Unity bridges them with the stage's own fill "
                    f"(pose and video stay on), nothing is missing")
            else:
                log(f"[camera_video] POV check OK   {tag}")
        else:
            ok = False
            log(f"[camera_video] POV check FAIL {tag}: {'; '.join(r['problems'])}")
    return ok


def install(master: Path, capture: str, log=print, names: list[str] | None = None) -> Path:
    """Mirror the master into the Unity project's StreamingAssets/<capture>/cameras/ (hard links when the volume allows).
    names = only (re)link these files and leave the rest: a tracks-only update touches nine small json files, not 375 MB."""
    dst = UNITY_STREAMING / capture / "cameras"
    if not (UNITY_STREAMING / capture).is_dir():
        raise SystemExit(f"[camera_video] {UNITY_STREAMING / capture} does not exist (export the capture first)")
    if names is None:
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True)
        todo = [p for p in sorted(master.iterdir()) if p.is_file()]
    else:
        dst.mkdir(parents=True, exist_ok=True)
        todo = [master / n for n in names]
    for p in todo:
        if (dst / p.name).exists():
            (dst / p.name).unlink()
        try:
            os.link(p, dst / p.name)
        except OSError:
            shutil.copy2(p, dst / p.name)
    log(f"[camera_video] installed {len(todo)} file(s) -> {dst}")
    return dst


def write_atomic(path: Path, text: str) -> None:
    """New file, then replace: a hard-linked copy in StreamingAssets keeps the old content until it is re-installed (Unity
    must not see a half-written or surprise-changed file while someone else's capture is running in the editor)."""
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------------------------------- CLI

def build_tracks(takes: list[str], only: list[str] | None, log=print):
    """Every phone's corrected track from the take(s) (later wins on overlap), vertical-FOV spikes removed.
    -> (tracks, take overlaps, {take: {cam: applied video offset ms}})"""
    per_cam: dict[str, list[CamTrack]] = {}
    applied: dict[str, dict[str, float]] = {}
    for tk in takes:
        offs = require_sync(tk)
        applied[tk] = {}
        for npz in sorted((md.take_work(tk) / "cameras").glob("*.npz")):
            tr = load_track(tk, npz.stem, offs)
            if tr is not None:
                per_cam.setdefault(npz.stem, []).append(tr)
                applied[tk][npz.stem] = round(float(np.median(tr.delta)) * 1000.0, 1)
    tracks: dict[str, CamTrack] = {}
    overlaps = {}
    for c, lst in per_cam.items():
        rep = overlap_report(lst) if len(lst) > 1 else []
        if rep:
            overlaps[c] = rep
            bad = [r for r in rep if r["position_m"]["median"] > 0.15 or r["direction_deg"]["median"] > 3.0]
            log(f"[camera_video] {c}: take overlaps {rep}" + ("  <-- WARNING: different world frames?" if bad else ""))
        tr = merge_tracks(lst)
        tr.vfov, tr.vfov_spikes = despike_vfov(tr.t, tr.vfov)
        tracks[c] = tr
    if only:
        tracks = {c: t for c, t in tracks.items() if c in only}
    if not tracks:
        raise SystemExit(f"[camera_video] no cameras/<cam>.npz in {takes} (--only {only})")
    for tk, d in applied.items():
        log(f"[camera_video] video offsets applied, take {tk} (ms): " + "  ".join(f"{c} {v:+.1f}" for c, v in sorted(d.items())))
    spikes = {c: t.vfov_spikes for c, t in sorted(tracks.items()) if t.vfov_spikes}
    if spikes:
        log(f"[camera_video] vertical-FOV spike samples replaced by the running median: {spikes}")
    return tracks, overlaps, applied


def pov_check(master: Path, capture: str, direction: list[str] | None, log=print) -> tuple[list[dict], bool]:
    files = [] if direction == ["none"] else direction_files(capture, direction)
    rep = coverage_report(master, direction_pov_windows(files))
    return rep, log_coverage(rep, log)


def run(takes: list[str], capture: str, cams: list[str] | None, direction: list[str] | None, all_full: bool,
        only: list[str] | None,
        span: tuple[float, float] | None, out: Path | None, do_install: bool, mp4: bool, thumbs: bool,
        long_side: int, quality: int, thumb_side: int, thumb_quality: int, log=print, tracks_only: bool = False) -> dict:
    tracks, overlaps, applied = build_tracks(takes, only, log)
    dir_cams, dir_files = ([], []) if direction == ["none"] else direction_cameras(capture, direction)
    want = set(cams or []) | set(dir_cams) | (set(tracks) if all_full else (set(DEFAULT_FULL_CAMS) if not cams else set()))
    want &= set(tracks)
    out = out or (md.take_work(takes[-1]) / "camera_video" / capture)
    old_manifest: dict = {}
    if (out / "cameras.json").exists():
        try:
            old_manifest = json.loads((out / "cameras.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            old_manifest = {}
    old: dict[str, dict] = {}
    if tracks_only:
        # regenerate the track json + manifest facts from the existing master: no decoding, no strips touched
        if not old_manifest:
            raise SystemExit(f"[camera_video] --tracks-only needs an existing master with cameras.json in {out}")
        s0, s1 = old_manifest["span"]
        old = {e["id"]: e for e in old_manifest.get("cameras", [])}
        missing = sorted(set(tracks) - set(old))
        if missing:
            log(f"[camera_video] WARNING --tracks-only: phones {missing} are not in the existing master (run a full export for them)")
        tracks = {c: t for c, t in tracks.items() if c in old}
    else:
        if span:
            s0, s1 = span
        else:
            s0, s1 = capture_span(capture, tracks)
            s0, s1 = s0 - 0.2, s1 + 0.2        # a little wider than the capture: the clock may sit between two frames
        if out.exists():
            for p in out.glob("*"):
                mine = not only or p.name.split(".")[0] in only
                if p.is_file() and p.suffix in (".hmj", ".mp4", ".json") and mine and p.name != "cameras.json":
                    p.unlink()
        out.mkdir(parents=True, exist_ok=True)
        log(f"[camera_video] takes {takes}, capture {capture}, span {s0:.3f}..{s1:.3f} s, full strips {sorted(want)}, "
            f"thumbs {sorted(tracks) if thumbs else []}")
    entries = {}
    touched = ["cameras.json"]
    for cam in sorted(tracks):
        tr = tracks[cam]
        if tracks_only:
            e = old[cam]
        else:
            e = export_camera(cam, takes, tr, (s0, s1), out, full=cam in want, thumbs=thumbs, long_side=long_side,
                              quality=quality, thumb_side=thumb_side, thumb_quality=thumb_quality, log=log)
        trk = track_json(tr, s0 - 1.0, s1 + 1.0)
        e["track"] = {"file": f"{cam}.track.json", "n": trk["n"]}
        write_atomic(out / e["track"]["file"], json.dumps(trk, separators=(",", ":")))
        touched.append(e["track"]["file"])
        e["ok_fraction"] = round(float(tr.ok.mean()), 4)
        e["vfov_deg"] = [round(float(tr.vfov.min()), 2), round(float(np.median(tr.vfov)), 2), round(float(tr.vfov.max()), 2)]
        e["vfov_spikes_replaced"] = tr.vfov_spikes
        e["pose"] = pose_coverage_entry(trk)
        if e["pose"]["no_pose_intervals"]:
            log(f"[camera_video] {cam}: Unity has NO pose for dance time {e['pose']['no_pose_intervals']} "
                f"(unlocalised run > {e['pose']['max_unlocalised_s']} s or no frames: phone not recording)")
        if e["pose"]["unlocalised_runs_over_0.3s"]:
            log(f"[camera_video] {cam}: unlocalised runs > 0.3 s (start, end, s): {e['pose']['unlocalised_runs_over_0.3s']} "
                f"(bridged by the stage's own fill up to {e['pose']['max_unlocalised_s']} s)")
        if "full" in e and mp4 and not tracks_only:
            write_mp4(cam, out, e, log)
        entries[cam] = e
    merged = dict(entries)
    if (only or tracks_only) and old_manifest:          # a partial run adds to / replaces entries of the manifest
        for e in old_manifest.get("cameras", []):
            merged.setdefault(e["id"], e)
    entries = merged
    manifest = {
        "version": 2, "capture": capture, "takes": takes, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tool": "python -m dancecap.camera_video", "span": [round(s0, 4), round(s1, 4)],
        "time_base": "reference seconds, sync-corrected (frame time + sync/offsets.json delta); the capture's times.json clock",
        "coordinate_system": "tracks in unity capture coordinates (x, y, -z of the take's world); Unity adds DanceOrigin.Offset",
        "container": "HMJ1: JPEG payloads, table (u64 offset, u32 length, f64 time) x count, trailer 'HMJ1' u32 count u64 table u32 w u32 h",
        "pose_policy": {"max_sample_gap_s": MAX_SAMPLE_GAP_S, "max_unlocalised_s": MAX_UNLOCALISED_S,
                        "max_unlocalised_fixed_s": MAX_UNLOCALISED_FIXED_S, "max_frame_gap_s": MAX_FRAME_GAP_S,
                        "note": "unlocalised (ok=0) samples carry the cameras stage's own fill and are used up to the run limit"},
        "video_offsets_ms_by_take": applied, "unsynced": bool(OPTIONS["allow_unsynced"] and not any(applied.values())),
        "take_overlaps": overlaps, "direction_cameras": dir_cams, "direction_files": [Path(p).name for p in dir_files],
        "cameras": [entries[c] for c in sorted(entries)],
    }
    write_atomic(out / "cameras.json", json.dumps(manifest, indent=1))
    rep, pov_ok = pov_check(out, capture, direction, log)
    manifest["pov_checks"] = rep
    manifest["pov_ok"] = pov_ok
    write_atomic(out / "cameras.json", json.dumps(manifest, indent=1))
    log(f"[camera_video] master {out}" + ("" if pov_ok else "   <-- POV CHECK FAILED: the film would lose the video or the pose in a shot"))
    if do_install:
        install(out, capture, log, names=touched if tracks_only else None)
    return manifest


def set_priority(name: str) -> None:
    """The PC has 4 CPUs shared with a Unity editor and other jobs: below-normal (default) is polite but is starved to
    a crawl when other jobs saturate the CPUs; 'normal' shares fairly (the ffmpeg children use 2 threads)."""
    PRIORITY["name"] = name
    PRIORITY["flags"] = BELOW_NORMAL if name == "below_normal" else 0
    if os.name == "nt" and name == "below_normal":
        try:
            import ctypes
            ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
        except Exception:  # noqa: BLE001
            pass


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m dancecap.camera_video", description=__doc__.split("\n\n")[0])
    ap.add_argument("takes", nargs="+", help="take(s) whose work/<take>/cameras tracks to export (later wins on overlap)")
    ap.add_argument("--capture", required=True, help="Unity capture folder name, e.g. 04_LarissaKaduDemo")
    ap.add_argument("--cams", default=None, help="comma-separated phones that get a FULL-size strip (+mp4)")
    ap.add_argument("--direction", nargs="*", default=["auto"], metavar="JSON",
                    help="film direction file(s) to read the fly_to_camera / camera_tour phones from (default 'auto': "
                         "the directions of this capture; 'none' = ignore)")
    ap.add_argument("--all-full", action="store_true", help="full-size strips for every phone")
    ap.add_argument("--only", default=None, help="comma-separated phones to process at all (testing / partial runs)")
    ap.add_argument("--span", nargs=2, type=float, default=None, metavar=("T0", "T1"),
                    help="reference seconds (default: the capture's times.json range)")
    ap.add_argument("--out", type=Path, default=None, help="master folder (default work/<take>/camera_video/<capture>)")
    ap.add_argument("--install", dest="install", action="store_true", default=True)
    ap.add_argument("--no-install", dest="install", action="store_false", help="do not mirror into StreamingAssets")
    ap.add_argument("--install-only", action="store_true", help="only mirror the existing master into StreamingAssets")
    ap.add_argument("--tracks-only", action="store_true",
                    help="rewrite the <cam>.track.json files + cameras.json of an existing master (new pose policy, FOV despike, POV "
                         "check) without decoding any video; --install then relinks just those files")
    ap.add_argument("--allow-unsynced", action="store_true",
                    help="export even when a take has no sync/offsets.json (raw clocks: phones may be ~100 ms off); default: error")
    ap.add_argument("--no-mp4", dest="mp4", action="store_false")
    ap.add_argument("--no-thumbs", dest="thumbs", action="store_false")
    ap.add_argument("--priority", choices=("below_normal", "normal"), default="below_normal")
    ap.add_argument("--long-side", type=int, default=1280)
    ap.add_argument("--quality", type=int, default=85)
    ap.add_argument("--thumb-side", type=int, default=256)
    ap.add_argument("--thumb-quality", type=int, default=78)
    a = ap.parse_args(argv)
    set_priority(a.priority)
    OPTIONS["allow_unsynced"] = bool(a.allow_unsynced)
    if a.install_only:
        master = a.out or (md.take_work(a.takes[-1]) / "camera_video" / a.capture)
        install(master, a.capture)
        return
    run(a.takes, a.capture, [c.strip() for c in a.cams.split(",")] if a.cams else None, a.direction, a.all_full,
        [c.strip() for c in a.only.split(",")] if a.only else None, tuple(a.span) if a.span else None, a.out, a.install, a.mp4, a.thumbs, a.long_side, a.quality, a.thumb_side,
        a.thumb_quality, tracks_only=a.tracks_only)


if __name__ == "__main__":
    main(sys.argv[1:])
