"""I/O glue of the two-camera solve: tracks, detections, synchronised correspondences (dancecap.cameras.twocam).

Reference clock = camera 01's video time; camera 02's video time = reference - SYNC_02 (work/carlos_aline/sync/sync.json,
NOTES_C1.md: "camera 02 = source + 14.8158 s", 0.45 ms residual).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot, Slerp

from . import ptz as P
from . import ptz_run as PR
from . import twocam as T
from . import twocam_people as TP

SYNC_02 = 14.8158           # reference time = video time of camera 02 + SYNC_02 (camera 01's own time = reference)
FPS = {"01": 24000 / 1001, "02": 30.0}
STEP = {"01": 24, "02": 30}                      # skeleton step (frames)


def frame_time(cam: str, frame: np.ndarray) -> np.ndarray:
    """Reference seconds of (fractional) video frames."""
    f = np.asarray(frame, float)
    return f / FPS[cam] + (SYNC_02 if cam == "02" else 0.0)


def time_frame(cam: str, t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, float)
    return (t - (SYNC_02 if cam == "02" else 0.0)) * FPS[cam]


class Track:
    """Per-frame rotation (reference -> camera) and focal of one camera; fractional frames are interpolated (slerp / log-linear)."""

    def __init__(self, frame: np.ndarray, rv: np.ndarray, f: np.ndarray, lens: P.Lens, n_frames: int, ok: np.ndarray | None = None):
        frame = np.asarray(frame, int)
        good = np.ones(len(frame), bool) if ok is None else np.asarray(ok, bool)
        order = np.argsort(frame)
        frame, rv, f, good = frame[order], np.asarray(rv)[order], np.asarray(f)[order], good[order]
        frame, rv, f = frame[good], rv[good], f[good]
        self.frame, self.rv, self.f, self.lens, self.n = frame, rv, f, lens, n_frames
        self._slerp = Slerp(frame.astype(float), Rot.from_rotvec(rv))

    def at(self, frame_f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(R (M,3,3), f (M,)) at fractional frames (clamped to the solved range)."""
        x = np.clip(np.asarray(frame_f, float), self.frame[0], self.frame[-1])
        R = self._slerp(x).as_matrix()
        f = np.exp(np.interp(x, self.frame, np.log(self.f)))
        return R, f

    def gap_to_solved(self, frame_f: np.ndarray) -> np.ndarray:
        """Distance (frames) from each fractional frame to the nearest solved frame (large = interpolated over a gap)."""
        i = np.searchsorted(self.frame, frame_f)
        lo = self.frame[np.clip(i - 1, 0, len(self.frame) - 1)]
        hi = self.frame[np.clip(i, 0, len(self.frame) - 1)]
        return np.minimum(np.abs(frame_f - lo), np.abs(hi - frame_f))


def load_track(cam: str, track_npz: Path, matches_npz: Path, skeleton_npz: Path, fix_f: float | None,
               refine: bool = True, f_scale: float = 1.0) -> Track:
    """Track from the pod's matches against the skeleton, re-solved here with the focal hypothesis `fix_f` (None: free per frame)."""
    sk = np.load(skeleton_npz)
    lens = P.Lens(int(sk["size"][0]), int(sk["size"][1]), float(sk["lens"][0]), float(sk["lens"][1]),
                  float(sk["lens"][2]), float(sk["lens"][3]))
    tr = np.load(track_npz)
    if not refine:
        return Track(tr["frame"], tr["rv"], tr["f"], lens, int(tr["frame"].max()) + 1)
    m = dict(np.load(matches_npz))
    res = PR.refine_tracks(m, sk["rv"], sk["f"], lens, STEP[cam], fix_f=fix_f)
    return Track(res["frame"], res["rv"], res["f"], lens, int(tr["frame"].max()) + 1, ok=res["rms_px"] < 8.0)


def synced_corr(tab_a: dict, tab_b: dict, tr_a: Track, tr_b: Track, t_ref: np.ndarray, min_conf: float = 0.5,
                max_gap: float = 6.0, dt_a: float = 0.0) -> tuple[T.Corr, T.View, T.View, np.ndarray]:
    """Correspondences at the reference times t_ref (camera 02 at t_ref, camera 01 at t_ref + dt_a: the VIDEO clock offset of 01
    against the audio clock, estimated from the geometry). Returns Corr (t = index into t_ref), the two Views and the kept mask."""
    fa = time_frame("01", np.asarray(t_ref, float) + dt_a)
    fb = time_frame("02", t_ref)
    Ra, fa_f = tr_a.at(fa)
    Rb, fb_f = tr_b.at(fb)
    va = T.View(Ra, fa_f, tr_a.lens)
    vb = T.View(Rb, fb_f, tr_b.lens)
    ok_t = (tr_a.gap_to_solved(fa) <= max_gap) & (tr_b.gap_to_solved(fb) <= max_gap)
    rows = {k: [] for k in ("t", "person", "joint", "pa", "pb", "w")}
    for who in (TP.CARLOS, TP.ALINE):
        ka = TP.interp_kp(tab_a, fa, who)
        kb = TP.interp_kp(tab_b, fb, who)
        conf = np.minimum(ka[..., 2], kb[..., 2])
        valid = (conf >= min_conf) & ok_t[:, None] & np.isfinite(ka[..., 0]) & np.isfinite(kb[..., 0])
        ti, ji = np.nonzero(valid)
        rows["t"].append(ti); rows["person"].append(np.full(len(ti), who)); rows["joint"].append(ji)
        rows["pa"].append(ka[ti, ji, :2]); rows["pb"].append(kb[ti, ji, :2])
        rows["w"].append(T.JOINT_W[ji] * np.clip(conf[ti, ji], 0, 1))
    corr = T.Corr(np.concatenate(rows["t"]), np.concatenate(rows["person"]), np.concatenate(rows["joint"]),
                  np.concatenate(rows["pa"]), np.concatenate(rows["pb"]), np.concatenate(rows["w"]))
    return corr, va, vb, ok_t


def load_detections(npz: Path) -> dict:
    d = np.load(npz)
    return {k: d[k] for k in ("frame", "box", "conf", "kp", "rgb")} | {"n_frames": int(d["n_frames"])}
