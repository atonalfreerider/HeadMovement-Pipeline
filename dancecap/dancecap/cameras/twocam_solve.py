"""Two panning cameras + two dancers -> relative pose, world frame, per-frame camera tracks (the carlos_aline new-room solve).

Inputs (all from earlier stages, see work/carlos_aline/NOTES_C2.md):
  ptz skeleton solutions (dancecap.cameras.ptz_run skeleton / skel_f*.npz), per-frame tracks (ptz_run refine), 2D keypoints of
  every person of every frame (people2d), the sync (twocam_run.SYNC_02, plus the geometric video offset `dt_a` estimated here),
  the dancers' statures (takes/<take>.toml) and the architecture verticals (twocam_vertical).

    python -m dancecap.cameras.twocam_solve run --cfg <json>

solve()    rig (reprojection + architecture vertical + floor flatness) -> world_from_prov (cameras.world.bootstrap rules: chain
           scale from the statures, floor = ankle plane - ankle height, origin = dancers' mean floor position)
arrays()   cameras/<cam>.npz: per-frame K, R, tvec in the metric world frame (CONTRACTS.md cameras/<cam>.npz schema)
"""
from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import ptz as P
from . import twocam as T
from . import twocam_people as TP
from . import twocam_run as R
from . import twocam_world as TW
from . import world as WD

CAMS = ("01", "02")


@dataclass
class Setup:
    f01: float = 4800.0            # focal hypothesis of camera 01 (heat) in px of the 1920 wide frame
    f02: float = 3350.0
    dt_a: float = -0.060           # video clock of 01 against its audio clock relative to 02 (geometric estimate), s
    stride: int = 3                # B frames: every 3rd (10 Hz)
    t0: float = 15.0
    t1: float = 196.0
    min_conf: float = 0.6
    sigma_v_deg: float = 0.5
    sigma_h: float = 0.012         # prov units (~9 cm: the ankle height scatter incl. detector noise)
    outlier_px: float = 12.0
    h_min: tuple[float, float] = (560.0, 400.0)      # dancer box height (px) in 01 / 02


def load_track_file(path: Path, n_frames: int, rms_max: float = 6.0) -> R.Track:
    d = np.load(path)
    lens = P.Lens(int(d["size"][0]), int(d["size"][1]), float(d["lens"][0]), float(d["lens"][1]), float(d["lens"][2]), float(d["lens"][3]))
    return R.Track(d["frame"], d["rv"], d["f"], lens, n_frames, ok=d["rms_px"] < rms_max)


def merged_track_01(main: Path, zoom_out: Path | None, n_frames: int) -> tuple[R.Track, np.ndarray]:
    """Camera 01: the constant-focal tracks (heat) + the zoom-out frames (free focal, other lens). Returns (Track, zoom flag per solved frame)."""
    d = np.load(main)
    lens = P.Lens(int(d["size"][0]), int(d["size"][1]), float(d["lens"][0]), float(d["lens"][1]), float(d["lens"][2]), float(d["lens"][3]))
    fr, rv, f, rms = d["frame"], d["rv"], d["f"], d["rms_px"]
    if zoom_out is not None:
        z = np.load(zoom_out)
        m = z["frame"] > fr.max()
        fr, rv, f, rms = np.r_[fr, z["frame"][m]], np.r_[rv, z["rv"][m]], np.r_[f, z["f"][m]], np.r_[rms, z["rms_px"][m]]
    return R.Track(fr, rv, f, lens, n_frames, ok=rms < 6.0), f


def tables(det_a: dict, det_b: dict, su: Setup) -> tuple[dict, dict]:
    return (TP.dancer_table(det_a, det_a["n_frames"], su.h_min[0]), TP.dancer_table(det_b, det_b["n_frames"], su.h_min[1]))


def correspondences(su: Setup, tab_a: dict, tab_b: dict, tr_a: R.Track, tr_b: R.Track, min_conf: float | None = None):
    fbs = np.arange(int(np.ceil((su.t0 - R.SYNC_02) * 30)), int((su.t1 - R.SYNC_02) * 30), su.stride)
    t_ref = fbs / 30 + R.SYNC_02
    corr, va, vb, ok_t = R.synced_corr(tab_a, tab_b, tr_a, tr_b, t_ref, dt_a=su.dt_a, min_conf=su.min_conf if min_conf is None else min_conf)
    return corr, va, vb, ok_t, t_ref


def solve_rig(su: Setup, corr, va, vb, vA: np.ndarray, vB: np.ndarray) -> tuple[T.Rig, dict, T.Corr]:
    """Rig with the architecture-vertical + floor terms; the outliers (> outlier_px in a view) are dropped and the rig re-solved."""
    g = {"vA": vA, "vB": vB, "sigma_v_rad": np.radians(su.sigma_v_deg), "sigma_h": su.sigma_h, "px_per_sigma": 3.0}
    rig0, i0 = T.solve_rig(corr, va, vb)
    rig1, i1 = T.solve_rig(corr, va, vb, theta0=np.array(i0["theta"]), gravity=g)
    X, err, ok = T.triangulate_all(rig1, corr, va, vb)
    keep = ok & (err < su.outlier_px)
    c2 = corr.sel(keep)
    rig, info = T.solve_rig(c2, va, vb, theta0=np.array(i1["theta"]), gravity=g)
    info["kept_fraction"] = float(keep.mean())
    info["reproj_only_median_px"] = float(i0["median_px"])
    return rig, info, c2


def dancers_dict(rig: T.Rig, corr: T.Corr, va, vb, n_t: int, max_err: float = 10.0) -> dict:
    X, err, ok = T.triangulate_all(rig, corr, va, vb)
    good = ok & (err < max_err)
    out = {r: {k: np.full((n_t, 3), np.nan) for k in WD.KP_SCALE} for r in ("lead", "follow")}
    role = {0: "lead", 1: "follow"}
    for i in np.where(good)[0]:
        j = int(corr.joint[i])
        if j in WD.KP_SCALE:
            out[role[int(corr.person[i])]][j][corr.t[i]] = X[i]
    return out


def world_from_rig(rig: T.Rig, info: dict, corr: T.Corr, va: T.View, vb: T.View, n_t: int, cfg: dict) -> dict:
    """cameras.world.bootstrap on the provisional-frame dancers (cameras 01 / 02 as the two main cameras, X axis along 02's right vector)."""
    st, src = WD.statures_from_take(cfg)
    soles, _ = WD.sole_from_take(cfg)
    dancers = dancers_dict(rig, corr, va, vb, n_t)
    tracks = {"01": {"R": va.R, "ok": np.ones(len(va.R), bool)}, "02": {"R": vb.R @ rig.Q.T, "ok": np.ones(len(vb.R), bool)}}
    gi = info["gravity"]
    vert = {"u": gi["u"], "n_cams": 2, "max_cam_deg": float(max(gi["vA_vs_u_deg"], gi["QvB_vs_u_deg"]))}
    return WD.bootstrap(tracks, dancers, None, st, src, x_cam="02", vertical=vert, sole_m=soles)


def camera_world_arrays(cam: str, track: R.Track, frames: np.ndarray, rig: T.Rig, sim3: dict, k1_override: float | None = None,
                        max_gap: float = 6.0) -> dict:
    """Per-frame world arrays of camera `cam` for the video `frames` (interpolating over short gaps; the flag says which)."""
    Rt, f = track.at(frames.astype(float))
    gap = track.gap_to_solved(frames.astype(float))
    Rcp, Cp = (Rt, np.zeros(3)) if cam == "01" else (Rt @ rig.Q.T, rig.b)
    s, Rw, tw = float(sim3["s"]), np.asarray(sim3["R"], float), np.asarray(sim3["t"], float)
    out = TW.camera_arrays(Rcp, f, tuple(track.lens.c), track.lens.k1 if k1_override is None else k1_override, Cp, s, Rw, tw)
    out["gap_frames"] = gap
    out["interpolated"] = gap > 0.5
    out["ok"] = gap <= max_gap
    out["focal"] = f
    return out


def load_cfg(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def take_cfg(workspace_takes: Path, take: str) -> dict:
    return tomllib.loads((workspace_takes / f"{take}.toml").read_text(encoding="utf-8"))
