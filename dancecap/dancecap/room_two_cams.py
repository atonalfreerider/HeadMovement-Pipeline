"""New room analysis from TWO panning tripod cameras (carlos_aline): cameras, world frame, room model, review images.

    python -m dancecap.room_two_cams verticals          architecture verticals of both cameras (needs the tracks)
    python -m dancecap.room_two_cams solve [--yaw D]    rig + world frame + per-frame camera arrays -> work/carlos_aline/cameras/
    python -m dancecap.room_two_cams uncertainty        block bootstrap + focal variants -> cameras/uncertainty.json
    python -m dancecap.room_two_cams walls              back wall / carpet edge / ceiling from the images -> room/room_model.json
    python -m dancecap.room_two_cams mesh               textured floor + wall mesh -> room/room_mesh.obj (+ textures)
    python -m dancecap.room_two_cams review             room projected into frames of both videos -> review/room/

Heavy compute (per-frame feature tracking, 2D keypoints) ran on one Runpod pod per camera (dancecap.cameras.ptz_pod); every stage
here is numpy / scipy / OpenCV on the PC (the 4 CPUs; the GPU only for the person masks of a few dozen frames).
See work/carlos_aline/NOTES_C2.md for the method, the numbers and how to re-run.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import cv2
import numpy as np

from .cameras import ptz as P
from .cameras import ptz_feats as F
from .cameras import room_draw as D
from .cameras import twocam as T
from .cameras import twocam_run as R
from .cameras import twocam_solve as S
from .cameras import twocam_vertical as TV
from .cameras import twocam_world as TW
from .cameras import world as WD
from .take import PACKAGE_ROOT

TAKE = "carlos_aline"
WORK = PACKAGE_ROOT / "work" / TAKE
RW = WORK / "room_work"            # intermediate data of this analysis (git-ignored)
SOURCE = PACKAGE_ROOT.parent / "dances" / "Carlos-Aline"      # git-ignored junction to the owner's folder
VIDEOS = {"01": "Carlos Aline Zouk Artist Jack Jill at Zouk SEA 2019.mp4", "02": "Zouk Sea 2019 Carlos and Aline Borges Jack Jill.mp4"}
SIZE = {"01": (1920, 1080), "02": (1920, 906)}
RECTS = {"01": [(1440, 800, 1920, 1080)], "02": []}       # camera 01's burnt-in watermark corner (screen-fixed: never a scene feature)
FPS = R.FPS
WINDOW = (3.6, 198.0)             # danced span on the reference clock (takes/carlos_aline.toml)
YOLO_SEG = PACKAGE_ROOT / "weights" / "yolo11x-seg.pt"
CAM_DIR = WORK / "cameras"
ROOM_DIR = WORK / "room"


def log(msg: str) -> None:
    print(f"[room_two_cams {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def video(cam: str) -> str:
    return str(SOURCE / VIDEOS[cam])


# ------------------------------------------------------------------------------------------------ inputs


def inputs(su: S.Setup | None = None):
    """Tracks, detections and take config of both cameras (the focal hypothesis in `su`)."""
    su = su or S.Setup()
    tr_a, _ = S.merged_track_01(RW / "final" / "track01_main.npz", RW / "final" / "track01_free2.npz", 4844)
    tr_b = S.load_track_file(RW / "final" / "track02_main.npz", 5585)
    det_a = R.load_detections(RW / "podout" / "pose_A.npz")
    det_b = R.load_detections(RW / "podout" / "pose_B.npz")
    cfg = S.take_cfg(PACKAGE_ROOT / "takes", TAKE)
    return su, tr_a, tr_b, det_a, det_b, cfg


# ------------------------------------------------------------------------------------------------ stages


def stage_verticals(a) -> dict:
    su, tr_a, tr_b, *_ = inputs()
    fa = [int(x) for x in np.linspace(300, 4600, 28)]
    fb = [int(x) for x in np.linspace(60, 5500, 28)]
    va = TV.architecture_vertical(video("01"), *SIZE["01"], tr_a, fa, RECTS["01"], str(YOLO_SEG))
    vb = TV.architecture_vertical(video("02"), *SIZE["02"], tr_b, fb, RECTS["02"], str(YOLO_SEG))
    np.savez(RW / "final" / "verticals.npz", vA=va["v"], vB=vb["v"])
    out = {k: {x: y for x, y in v.items() if x != "v"} | {"v": np.asarray(v["v"]).tolist()} for k, v in (("01", va), ("02", vb))}
    (RW / "final" / "verticals.json").write_text(json.dumps(out, indent=1))
    log(json.dumps({k: {x: (round(y, 3) if isinstance(y, float) else y) for x, y in v.items() if x != "v"} for k, v in out.items()}))
    return out


def solve_all(su: S.Setup, tr_a, tr_b, det_a, det_b, cfg, vA, vB, yaw_deg: float = 0.0, stride: int | None = None):
    """Rig + world frame (+ optional re-yaw about Y around the origin). Returns dict with everything the writers need."""
    if stride:
        su = S.Setup(**{**su.__dict__, "stride": stride})
    tab_a, tab_b = S.tables(det_a, det_b, su)
    corr, va, vb, ok_t, t_ref = S.correspondences(su, tab_a, tab_b, tr_a, tr_b)
    rig, info, c2 = S.solve_rig(su, corr, va, vb, vA, vB)
    corr_all, va, vb, _, _ = S.correspondences(su, tab_a, tab_b, tr_a, tr_b, min_conf=0.5)
    w = S.world_from_rig(rig, info, corr_all, va, vb, len(t_ref), cfg)
    Rw, tw = np.array(w["R"]), np.array(w["t"])
    if yaw_deg:
        Rw, tw = TW.re_yaw(Rw, tw, yaw_deg)
        w = {**w, "R": Rw.tolist(), "t": tw.tolist(), "yaw_applied_deg": yaw_deg}
    return {"su": su, "rig": rig, "info": info, "world": w, "corr": c2, "corr_all": corr_all, "va": va, "vb": vb, "t_ref": t_ref}


def stage_solve(a) -> dict:
    su, tr_a, tr_b, det_a, det_b, cfg = inputs()
    vv = np.load(RW / "final" / "verticals.npz")
    yaw = float(a.yaw) if a is not None and getattr(a, "yaw", None) is not None else 0.0
    res = solve_all(su, tr_a, tr_b, det_a, det_b, cfg, vv["vA"], vv["vB"], yaw)
    rig, w, info = res["rig"], res["world"], res["info"]
    CAM_DIR.mkdir(parents=True, exist_ok=True)
    write_cameras(res, tr_a, tr_b)
    rj = rig_json(res)
    (CAM_DIR / "rig.json").write_text(json.dumps(rj, indent=1))
    log(f"world scale s = {w['s']:.4f} m / baseline unit; camera 01 {np.round(rj['cameras']['01']['centre_world_m'], 2).tolist()} "
        f"camera 02 {np.round(rj['cameras']['02']['centre_world_m'], 2).tolist()}; reprojection median {info['median_px']:.2f} px")
    pickle.dump({k: v for k, v in res.items() if k not in ("va", "vb")}, open(RW / "final" / "solve_state.pkl", "wb"))
    # the dancers' triangulated joints in the world (floor-contact evidence, review images)
    Xp, err, ok = T.triangulate_all(rig, res["corr_all"], res["va"], res["vb"])
    Xw = TW.to_world(Xp, float(w["s"]), np.array(w["R"]), np.array(w["t"]))
    np.savez_compressed(CAM_DIR / "dancers_world.npz", t=res["t_ref"][res["corr_all"].t], person=res["corr_all"].person.astype(np.int8),
                        joint=res["corr_all"].joint.astype(np.int8), X=Xw.astype(np.float32), err_px=err.astype(np.float32), ok=ok)
    return rj


def rig_json(res: dict) -> dict:
    rig, w, info, su = res["rig"], res["world"], res["info"], res["su"]
    s, Rw, tw = float(w["s"]), np.array(w["R"]), np.array(w["t"])
    cams = {}
    for cam, Cp in (("01", np.zeros(3)), ("02", rig.b)):
        C = TW.to_world(Cp[None], s, Rw, tw)[0]
        cams[cam] = {"centre_world_m": C.tolist(), "height_above_floor_m": float(C[1])}
    gi = info["gravity"]
    d = w["diagnostics"]
    return {"schema": "carlos_aline.two_camera_rig/1", "frame": "world: right-handed, Y up, metres, floor y = 0 (CONTRACTS.md)",
            "setup": {k: (list(v) if isinstance(v, tuple) else v) for k, v in su.__dict__.items()},
            "relative": {"Q_ref02_to_ref01": rig.Q.tolist(), "baseline_dir_prov": rig.b.tolist(),
                         "baseline_m": float(np.linalg.norm(np.array(cams["01"]["centre_world_m"]) - np.array(cams["02"]["centre_world_m"])))},
            "world_from_prov": {"s": s, "R": Rw.tolist(), "t": tw.tolist()}, "cameras": cams,
            "fit": {"median_px": info["median_px"], "p90_px": info["p90_px"], "n_correspondences": info["n"],
                    "kept_fraction": info["kept_fraction"], "reproj_only_median_px": info["reproj_only_median_px"],
                    "vertical_01_vs_up_deg": gi["vA_vs_u_deg"], "vertical_02_vs_up_deg": gi["QvB_vs_u_deg"],
                    "ankle_height_mad_prov": gi["ankle_h_mad"], "bone_rel_mad": info.get("bone_rel_mad")},
            "scale": {"rule": w["scale_rule"], "source": w["scale_source"], "per_dancer": {r: v["s"] for r, v in d["scale_per_dancer"].items()},
                      "disagreement": d.get("scale_disagreement"), "agreement_ok": d.get("scale_agreement_ok"),
                      "rel_se_combined": d.get("scale_rel_se_combined"), "abs_uncertainty_rel": d.get("scale_abs_uncertainty_rel")},
            "gravity_source": w["gravity_source"], "floor_source": w["floor_source"], "x_axis": w["x_axis"],
            "yaw_applied_deg": w.get("yaw_applied_deg", 0.0), "diagnostics": _jsonable(d)}


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def window_frames(cam: str, n_frames: int) -> np.ndarray:
    t = R.frame_time(cam, np.arange(n_frames))
    return np.where((t >= WINDOW[0] - 1e-6) & (t <= WINDOW[1] + 1e-6))[0]


def write_cameras(res: dict, tr_a: R.Track, tr_b: R.Track) -> None:
    """cameras/<cam>.npz (the CONTRACTS.md schema, rows = the canonical window frames of the camera) + <cam>_world.npz."""
    for cam, trk, nfr in (("01", tr_a, 4844), ("02", tr_b, 5585)):
        fr = window_frames(cam, nfr)
        arr = S.camera_world_arrays(cam, trk, fr, res["rig"], res["world"])
        t = R.frame_time(cam, fr)
        zoom = arr["focal"] < 0.95 * float(np.median(arr["focal"]))     # the zoom-out of camera 01 at the end
        ok = arr["ok"] & ~zoom
        extra = dict(frame_index=fr.astype(np.int32), interpolated=arr["interpolated"], focal_px=arr["focal"].astype(np.float32),
                     source=np.where(arr["interpolated"], "interp", "track"), zoom_transition=zoom,
                     camera_class="tripod_zoom" if cam == "01" else "tripod", type_model="fixed_centre",
                     centre_world=arr["C"][0], pp_px=np.array(trk.lens.c), k2_ignored=float(trk.lens.k2),
                     focal_hypothesis_px=float(res["su"].f01 if cam == "01" else res["su"].f02),
                     video_offset_hint_s=float(res["su"].dt_a if cam == "01" else 0.0),
                     attached=False, t_clock="reference seconds = camera 01 video time (audio clock of sync.json)")
        np.savez_compressed(CAM_DIR / f"{cam}.npz", t=t, K=arr["K"].astype(np.float64), R=arr["R"], tvec=arr["tvec"], dist=arr["dist"],
                            ok=ok, reproj_px=np.zeros(len(fr), np.float32), n_inliers=np.zeros(len(fr), np.int32), **extra)
        np.savez_compressed(CAM_DIR / f"{cam}_world.npz", t=t, R=arr["R"], tvec=arr["tvec"], C=arr["C"], K=arr["K"], dist=arr["dist"], ok=ok)
    ident = {"s": 1.0, "R": np.eye(3).tolist(), "t": [0.0, 0.0, 0.0],
             "note": "identity: the cameras of this take are stored directly in the metric world frame (there is no SfM frame; "
                     "the room is NOT a COLMAP model). world_from_prov.json is the Sim3 from the solve's provisional frame.",
             "gravity_source": res["world"]["gravity_source"], "scale_source": res["world"]["scale_source"],
             "floor_source": res["world"]["floor_source"], "x_axis": res["world"]["x_axis"]}
    (CAM_DIR / "world_from_sfm.json").write_text(json.dumps(ident, indent=1))
    (CAM_DIR / "world_from_prov.json").write_text(json.dumps({k: res["world"][k] for k in ("s", "R", "t")}, indent=1))


# ------------------------------------------------------------------------------------------------ uncertainty


def summary(res: dict) -> dict:
    """Yaw-invariant numbers of one solve (the world's X axis follows camera 02, so positions are compared by distances)."""
    rig, w = res["rig"], res["world"]
    s, Rw, tw = float(w["s"]), np.array(w["R"]), np.array(w["t"])
    C1 = TW.to_world(np.zeros((1, 3)), s, Rw, tw)[0]
    C2 = TW.to_world(rig.b[None], s, Rw, tw)[0]
    d = w["diagnostics"]
    return {"s": s, "baseline_m": float(np.linalg.norm(C1 - C2)), "h01_m": float(C1[1]), "h02_m": float(C2[1]),
            "d01_m": float(np.hypot(C1[0], C1[2])), "d02_m": float(np.hypot(C2[0], C2[2])),
            "az01_deg": float(np.degrees(np.arctan2(C1[0], C1[2]))), "az02_deg": float(np.degrees(np.arctan2(C2[0], C2[2]))),
            "Q_deg": float(np.degrees(np.arccos(np.clip((np.trace(rig.Q) - 1) / 2, -1, 1)))),
            "s_lead": d["scale_per_dancer"]["lead"]["s"], "s_follow": d["scale_per_dancer"]["follow"]["s"],
            "median_px": res["info"]["median_px"]}


VARIANTS = [(4300, 3100, -0.06), (5100, 3600, -0.06), (4800, 3100, -0.06), (4800, 3600, -0.06), (4525, 3350, -0.06),
            (4800, 3350, -0.045), (4800, 3350, -0.075)]


def stage_uncertainty(a) -> dict:
    """Leave-one-block-out jackknife over 10 s blocks of the heat (--part jackknife) and focal / clock variants (--part variants
    --which 0,1,..: each a full re-solve with the tracks re-solved at that focal from the stored matches). Stride 6 (5 Hz)."""
    part = getattr(a, "part", "jackknife")
    su, tr_a, tr_b, det_a, det_b, cfg = inputs(S.Setup(stride=6))
    vv = np.load(RW / "final" / "verticals.npz")
    out: dict = {}
    if part == "jackknife":
        tab_a, tab_b = S.tables(det_a, det_b, su)
        corr, va, vb, ok_t, t_ref = S.correspondences(su, tab_a, tab_b, tr_a, tr_b)
        rig, info, c2 = S.solve_rig(su, corr, va, vb, vv["vA"], vv["vB"])
        corr_all, va, vb, _, _ = S.correspondences(su, tab_a, tab_b, tr_a, tr_b, min_conf=0.5)
        base = summary({"rig": rig, "world": S.world_from_rig(rig, info, corr_all, va, vb, len(t_ref), cfg), "info": info})
        blocks = np.floor((t_ref - su.t0) / 10.0).astype(int)
        out = {"base": base, "stride": su.stride, "jackknife": []}
        g = {"vA": vv["vA"], "vB": vv["vB"], "sigma_v_rad": np.radians(su.sigma_v_deg), "sigma_h": su.sigma_h, "px_per_sigma": 3.0}
        for b in np.unique(blocks):
            keep_t = blocks != b
            rig_b, info_b = T.solve_rig(c2.sel(keep_t[c2.t]), va, vb, theta0=np.array(info["theta"]), gravity=g)
            w_b = S.world_from_rig(rig_b, info_b, corr_all.sel(keep_t[corr_all.t]), va, vb, len(t_ref), cfg)
            out["jackknife"].append(summary({"rig": rig_b, "world": w_b, "info": info_b}))
        jk = {k: [r[k] for r in out["jackknife"]] for k in base}
        n = len(out["jackknife"])
        out["jackknife_sd"] = {k: float(np.sqrt((n - 1) / n * np.sum((np.array(v) - np.mean(v)) ** 2))) for k, v in jk.items()}
        (CAM_DIR / "uncertainty_jackknife.json").write_text(json.dumps(out, indent=1))
        log("jackknife sd: " + json.dumps({k: round(v, 3) for k, v in out["jackknife_sd"].items()}))
        return out
    which = [int(x) for x in str(getattr(a, "which", "0")).split(",")]
    mats = {c: PR_matches(c) for c in ("A", "B")}
    fbs = np.arange(int(np.ceil((su.t0 - R.SYNC_02) * 30)), int((su.t1 - R.SYNC_02) * 30), su.stride)
    rows = []
    for k in which:
        fA, fB, dt = VARIANTS[k]
        su_v = S.Setup(**{**su.__dict__, "f01": float(fA), "f02": float(fB), "dt_a": dt})
        trv_a = variant_track("A", fA, mats["A"], dt, fbs)
        trv_b = variant_track("B", fB, mats["B"], 0.0, fbs)
        res = solve_all(su_v, trv_a, trv_b, det_a, det_b, cfg, vv["vA"], vv["vB"])
        row = {"f01": fA, "f02": fB, "dt_a": dt, **summary(res)}
        rows.append(row)
        log(f"variant f01 {fA} f02 {fB} dt {dt}: baseline {row['baseline_m']:.2f} m h01 {row['h01_m']:.2f} h02 {row['h02_m']:.2f} s {row['s']:.3f} median {row['median_px']:.2f} px")
        (CAM_DIR / f"uncertainty_variant_{k}.json").write_text(json.dumps(row, indent=1))
    return {"variants": rows}


def PR_matches(c: str) -> dict:
    from .cameras import ptz_run as PR
    fs = sorted((RW / "podout").glob(f"track_{c}_s*_matches.npz"))
    return PR.load_matches(fs)


def variant_track(c: str, f: float, matches: dict, dt: float, fbs: np.ndarray) -> R.Track:
    """Track of camera `c` ('A' = 01, 'B' = 02) at focal hypothesis f for the frames the solve needs."""
    from .cameras import ptz_run as PR
    cam = "01" if c == "A" else "02"
    rv, ff, lens = PR.load_solution(RW / f"ptz_{c}" / f"skel_f{int(f)}.npz")
    t_ref = fbs / 30 + R.SYNC_02
    fr = R.time_frame(cam, t_ref + (dt if cam == "01" else 0.0))
    frames = sorted(set(np.floor(fr).astype(int)) | set(np.ceil(fr).astype(int))) if cam == "01" else sorted(set(fbs.tolist()))
    m = {k: v[matches["ref"] < len(rv)] for k, v in matches.items()}
    res = PR.refine_tracks(m, rv, ff, lens, R.STEP[cam], fix_f=float(f), frames=frames)
    return R.Track(res["frame"], res["rv"], res["f"], lens, int(matches["frame"].max()) + 1, ok=res["rms_px"] < 6.0)


# ------------------------------------------------------------------------------------------------ walls


def load_cam(cam: str) -> dict:
    d = np.load(CAM_DIR / f"{cam}.npz")
    return {k: d[k] for k in ("t", "K", "R", "tvec", "dist", "ok", "frame_index")} | {"C": np.broadcast_to(d["centre_world"], (len(d["t"]), 3))}


def row_of(arr: dict, frame: int) -> int:
    i = int(np.searchsorted(arr["frame_index"], frame))
    return min(i, len(arr["frame_index"]) - 1)


def grab(cam: str, frames: list[int]):
    """Frames (bgr) + ignore masks (people dilated + the watermark) of the given video frames."""
    ims, igs = [], []
    w, h = SIZE[cam]
    got = dict(F.read_video_frames(video(cam), w, h, frames=frames))
    for fi in frames:
        im = got[fi]
        pm = F.person_masks_yolo([im], str(YOLO_SEG))[0]
        ims.append(im)
        igs.append(cv2.dilate(F.static_ignore(im.shape[:2], pm, RECTS[cam]), np.ones((21, 21), np.uint8)))
    return ims, igs


def stage_walls(a) -> dict:
    from . import room_two_cams_walls as WL
    arr = {c: load_cam(c) for c in ("01", "02")}
    res: dict = {"schema": "carlos_aline.room_model/1", "frame": "world: Y up, metres, floor y = 0 (cameras/world_from_sfm.json = identity)"}
    # --- back wall: stereo plane sweep
    times = np.linspace(30, 180, 8)
    fa = [int(round((t + S.Setup().dt_a) * FPS["01"])) for t in times]
    fb = [int(round((t - R.SYNC_02) * 30)) for t in times]
    ia, ga = grab("01", fa)
    ib, gb = grab("02", fb)
    frames = []
    for k in range(len(times)):
        frames.append({"ia": row_of(arr["01"], fa[k]), "ib": row_of(arr["02"], fb[k]),
                       "eA": WL.edge_map(cv2.cvtColor(ia[k], cv2.COLOR_BGR2GRAY), 1.8), "eB": WL.edge_map(cv2.cvtColor(ib[k], cv2.COLOR_BGR2GRAY), 1.2),
                       "mA": ga[k], "mB": gb[k]})
    zs, phis = np.arange(-10.8, -7.6, 0.2), np.arange(-2, 9.1, 1.0)
    sw = WL.wall_sweep(arr["01"], arr["02"], frames, zs, phis)
    pk = WL.peak(sw["score"], zs, phis)
    log(f"back wall: z {pk['z']:.2f} m, yaw {pk['phi_deg']:.1f} deg (score {pk['score']:.3f}, per-frame z {pk['per_frame_z_median']} +- {pk['per_frame_z_mad']})")
    res["back_wall"] = pk | {"method": "stereo plane sweep, ZNCC of edge maps, 8 instants", "z_grid": [float(zs[0]), float(zs[-1]), float(zs[1] - zs[0])],
                             "score_vs_z_at_best_yaw": [float(x) for x in np.nanmean(sw["score"], axis=2)[:, int(np.argmin(np.abs(phis - pk["phi_deg"])))]]}
    np.savez_compressed(RW / "final" / "wall_sweep.npz", zs=zs, phis=phis, score=sw["score"])
    # --- carpet edge and wall base per camera
    res["floor_lines"] = {}
    for cam in ("01", "02"):
        tt = np.linspace(22, 186, 20)
        fr = [int(round((t + (S.Setup().dt_a if cam == "01" else -R.SYNC_02 * 0)) * FPS[cam])) if cam == "01" else int(round((t - R.SYNC_02) * 30)) for t in tt]
        ims, igs = grab(cam, fr)
        idx = [row_of(arr[cam], f) for f in fr]
        rects = RECTS[cam]
        out = {}
        for name, (z0, z1) in {"carpet_edge": (-8.2, -5.2), "wall_base": (-11.6, -8.0)}.items():
            sc = WL.edge_scan(arr[cam], idx, ims, igs, np.arange(z0, z1, 0.1), np.arange(-3, 3.1, 0.5), rects=rects)
            m = sc["score"]
            i, j = np.unravel_index(np.argmax(m), m.shape)
            zc, pc = WL.plateau_peak(m, sc["zs"], sc["phis"])
            pf = sc["per_frame_best"]
            good = pf[:, 2] > 0.7 * np.median(pf[:, 2])
            out[name] = {"z": zc, "phi_deg": pc, "z_argmax": float(sc["zs"][i]), "score": float(m[i, j]),
                         "per_frame_z_median": float(np.median(pf[good, 0])), "per_frame_z_mad": float(1.4826 * np.median(np.abs(pf[good, 0] - np.median(pf[good, 0]))))}
            log(f"camera {cam} {name}: z {out[name]['z']:.2f} phi {out[name]['phi_deg']:.1f} per-frame {out[name]['per_frame_z_median']:.2f} +- {out[name]['per_frame_z_mad']:.2f}")
        # yaw of the carpet edge from its two halves (phi fixed at 0: the z of the left and the right half at their mid x)
        halves = {}
        for side, xr in (("left", (-15.0, -3.0)), ("right", (3.0, 15.0))):
            sc = WL.edge_scan(arr[cam], idx, ims, igs, np.arange(-8.4, -5.0, 0.05), np.array([0.0]), rects=rects, x_range=xr)
            zc, _ = WL.plateau_peak(sc["score"], sc["zs"], sc["phis"])
            pf = sc["per_frame_best"]
            good = pf[:, 2] > 0.7 * np.median(pf[:, 2])
            halves[side] = {"z": zc, "x_mid": 0.5 * (xr[0] + xr[1]), "per_frame_z_median": float(np.median(pf[good, 0]))}
        dz = halves["right"]["z"] - halves["left"]["z"]
        out["carpet_edge_halves"] = {**halves, "yaw_deg": float(np.degrees(np.arctan2(dz, 12.0)))}
        log(f"camera {cam} carpet edge halves: left z {halves['left']['z']:.2f}, right z {halves['right']['z']:.2f} -> yaw {out['carpet_edge_halves']['yaw_deg']:.2f} deg")
        res["floor_lines"][cam] = out
        if cam == "02":
            wl = res["back_wall"]
            hs = np.arange(2.2, 6.5, 0.05)
            cs = WL.cornice_scan(arr[cam], idx, ims, igs, wl["z"], wl["phi_deg"], hs)
            res["cornice_height_02"] = {"h": float(hs[int(np.argmax(cs["score"]))]), "per_frame_median": float(np.median(cs["per_frame_best"][:, 0])),
                                        "score": [float(x) for x in cs["score"]], "heights": [float(x) for x in hs]}
            log(f"cornice height (camera 02): {res['cornice_height_02']['h']:.2f} m")
    ROOM_DIR.mkdir(parents=True, exist_ok=True)
    (ROOM_DIR / "walls_measure.json").write_text(json.dumps(_jsonable(res), indent=1))
    return res


# ------------------------------------------------------------------------------------------------ model, mesh


def stage_model(a) -> dict:
    """room/room_model.json from walls_measure.json. The room frame is the world turned by the wall's yaw about Y: u = (cos phi, 0, sin phi)
    along the wall, n = (-sin phi, 0, cos phi) its normal towards the cameras; the wall plane is n.X = d_w, the floor rectangle is
    s in [-13, 13] along u by t in [d_w, 12] along n."""
    m = json.loads((ROOM_DIR / "walls_measure.json").read_text())
    rj = json.loads((CAM_DIR / "rig.json").read_text())
    fl_ = m["floor_lines"]
    phis = [fl_["01"]["carpet_edge"]["phi_deg"], fl_["02"]["carpet_edge"]["phi_deg"], fl_["02"]["wall_base"]["phi_deg"],
            fl_["01"]["carpet_edge_halves"]["yaw_deg"], fl_["02"]["carpet_edge_halves"]["yaw_deg"]]
    phi = float(np.round(np.mean(phis) * 2) / 2)
    sw = np.load(RW / "final" / "wall_sweep.npz")
    mean = np.nanmean(sw["score"], 2)
    j = int(np.argmin(np.abs(sw["phis"] - phi)))
    zcol = mean[:, j]
    k = int(np.nanargmax(zcol))
    z_w = float(sw["zs"][k])
    if 0 < k < len(zcol) - 1:
        y0, y1, y2 = zcol[k - 1], zcol[k], zcol[k + 1]
        den = y0 - 2 * y1 + y2
        if den < 0:
            z_w += 0.5 * (y0 - y2) / den * float(sw["zs"][1] - sw["zs"][0])
    c = np.cos(np.radians(phi))
    d_w = z_w * c
    z_e = float(np.mean([fl_["01"]["carpet_edge"]["z"], fl_["02"]["carpet_edge"]["z"]]))
    z_e_diff = float(abs(fl_["01"]["carpet_edge"]["z"] - fl_["02"]["carpet_edge"]["z"]))
    H = float(getattr(a, "wall_height", 2.7))
    u = [float(np.cos(np.radians(phi))), 0.0, float(np.sin(np.radians(phi)))]
    n = [float(-np.sin(np.radians(phi))), 0.0, float(np.cos(np.radians(phi)))]
    model = {"schema": "carlos_aline.room_model/1", "frame": rj["frame"], "yaw_applied_deg": rj["yaw_applied_deg"],
             "origin": "the dancers' mean floor position (cameras.world.bootstrap rule); world X = camera 02's mean right vector",
             "room_frame": {"yaw_deg": phi, "yaw_sd_deg": float(max(1.0, np.std(phis, ddof=1))), "u_along_wall": u, "n_wall_normal_to_cameras": n,
                            "yaw_measurements_deg": {"carpet_edge_01": phis[0], "carpet_edge_02": phis[1], "wall_base_02": phis[2], "carpet_halves_01": phis[3], "carpet_halves_02": phis[4]}},
             "floor": {"y": 0.0, "s": [-13.0, 13.0], "t": [d_w, 12.0],
                       "source": "ankle plane (dancers) lowered by the ankle-joint height; flatness of the lower ankles in rig.json fit.ankle_height_mad_prov",
                       "textured_from": "both cameras, people masked, median of ~26 instants"},
             "back_wall": {"plane": {"normal": n, "d": d_w, "z_at_x0": z_w}, "s": [-13.0, 13.0], "y": [0.0, H], "height_m": H,
                           "height_source": "wood panels end under the cream cove (cornice scan on the wall plane, camera 02)",
                           "d_uncertainty_m": 0.3, "measurement": m["back_wall"]},
             "carpet_edge": {"t": z_e * c, "z_at_x0": z_e, "s": [-13.0, 13.0], "agreement_between_cameras_m": z_e_diff,
                             "note": "border between the pale dance floor and the patterned carpet strip (a floor line both cameras see)",
                             "per_camera": {cc: fl_[cc]["carpet_edge"] for cc in ("01", "02")}},
             "wall_base_per_camera": {cc: fl_[cc]["wall_base"] for cc in ("01", "02")},
             "unobserved": ["side walls (the cameras never see them)", "ceiling (cream cove above the wood; lights seen by camera 02 only)",
                            "floor in front of the cameras (beyond the picture) and to the left / right of the pictures"],
             "cameras_world_m": {cc: rj["cameras"][cc]["centre_world_m"] for cc in ("01", "02")}}
    ROOM_DIR.mkdir(parents=True, exist_ok=True)
    (ROOM_DIR / "room_model.json").write_text(json.dumps(_jsonable(model), indent=1))
    return model


def write_obj(path: Path, groups: dict, textures: dict) -> None:
    """groups: {name: (V (n,3), UV (n,2), F (m,3) int)}; one OBJ + MTL with map_Kd per group (CONTRACTS.md room/room_mesh.obj)."""
    lines = [f"mtllib {path.stem}.mtl"]
    base = 0
    for name, (V, UV, Fc) in groups.items():
        lines += [f"o {name}"] + [f"v {x:.5f} {y:.5f} {z:.5f}" for x, y, z in V] + [f"vt {u:.6f} {v:.6f}" for u, v in UV]
        lines.append(f"usemtl {name}")
        lines += [f"f {a + 1 + base}/{a + 1 + base} {b + 1 + base}/{b + 1 + base} {c + 1 + base}/{c + 1 + base}" for a, b, c in Fc]
        base += len(V)
    path.write_text("\n".join(lines) + "\n")
    path.with_suffix(".mtl").write_text("".join(f"newmtl {n}\nKd 1 1 1\nmap_Kd {tex}\n" for n, tex in textures.items()))


def grid_mesh(origin: np.ndarray, du: np.ndarray, dv: np.ndarray, nu: int, nv: int, flip: bool = False):
    """Regular grid P = origin + i du + j dv (i < nu, j < nv cells) with UV = (i/nu, 1 - j/nv) and triangles."""
    ii, jj = np.meshgrid(np.arange(nu + 1), np.arange(nv + 1))
    V = origin[None] + ii.ravel()[:, None] * du[None] + jj.ravel()[:, None] * dv[None]
    UV = np.stack([ii.ravel() / nu, 1.0 - jj.ravel() / nv], 1)
    idx = lambda i, j: j * (nu + 1) + i          # noqa: E731
    Fc = []
    for j in range(nv):
        for i in range(nu):
            a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            Fc += [(a, c, b), (a, d, c)] if flip else [(a, b, c), (a, c, d)]
    return V, UV, np.array(Fc)


def sources_for(cam: str, arr: dict, times: np.ndarray) -> list[dict]:
    fr = [int(round((t + (S.Setup().dt_a if cam == "01" else 0.0)) * FPS[cam])) if cam == "01" else int(round((t - R.SYNC_02) * 30)) for t in times]
    out = []
    for k in range(0, len(fr), 9):
        ims, igs = grab(cam, fr[k:k + 9])
        for f_, im, ig in zip(fr[k:k + 9], ims, igs):
            out.append({"arr": arr, "i": row_of(arr, f_), "img": im, "ignore": ig})
    return out


def stage_mesh(a) -> dict:
    from . import room_two_cams_walls as WL
    model = json.loads((ROOM_DIR / "room_model.json").read_text())
    arr = {c: load_cam(c) for c in ("01", "02")}
    fl, bw = model["floor"], model["back_wall"]
    s0, s1 = fl["s"]
    t0, t1 = fl["t"]
    H = bw["height_m"]
    d_w = bw["plane"]["d"]
    times = np.linspace(20, 190, 26)
    srcs = {c: sources_for(c, arr[c], times) for c in ("01", "02")}
    log("frames decoded and masked: " + ", ".join(f"{c} {len(v)}" for c, v in srcs.items()))
    res = 0.04                                             # floor: 4 cm texels, row 0 = t0 (far, at the wall), column 0 = s0
    ns, nt = int(round((s1 - s0) / res)), int(round((t1 - t0) / res))
    S_, T_ = np.meshgrid(s0 + (np.arange(ns) + 0.5) * res, t0 + (np.arange(nt) + 0.5) * res)
    pts = room_point(model, S_.ravel(), T_.ravel(), 0.0)
    tex = {cam: WL.ortho_texture(pts, (nt, ns), srcs[cam]) for cam in ("02", "01")}
    g = WL.gain_match(tex["01"][0], tex["02"][0])
    floor = np.where(np.isfinite(tex["02"][0][:, :, :1]), tex["02"][0], tex["01"][0] * g[None, None])
    cover = np.isfinite(floor[:, :, 0])
    floor_img = WL.fill_holes(floor, (150, 175, 195))
    resw = 0.02                                            # wall: 2 cm texels, row 0 = top (y = H), column 0 = s0
    nsw, nyw = int(round((s1 - s0) / resw)), int(round(H / resw))
    SW, YW = np.meshgrid(s0 + (np.arange(nsw) + 0.5) * resw, H - (np.arange(nyw) + 0.5) * resw)
    ptsw = room_point(model, SW.ravel(), np.full(SW.size, d_w), YW.ravel())
    tw_ = {cam: WL.ortho_texture(ptsw, (nyw, nsw), srcs[cam], min_obs=2) for cam in ("02", "01")}
    gw = WL.gain_match(tw_["01"][0], tw_["02"][0])
    wall = np.where(np.isfinite(tw_["02"][0][:, :, :1]), tw_["02"][0], tw_["01"][0] * gw[None, None])
    wall_cover = np.isfinite(wall[:, :, 0])
    wall_img = WL.fill_holes(wall, (60, 70, 110))
    ROOM_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(ROOM_DIR / "room_floor_tex.png"), floor_img)
    cv2.imwrite(str(ROOM_DIR / "room_mesh_tex.png"), wall_img)
    u, n = np.array(model["room_frame"]["u_along_wall"]), np.array(model["room_frame"]["n_wall_normal_to_cameras"])
    nsf, ntf = int(round((s1 - s0) / 0.5)), int(round((t1 - t0) / 0.5))
    Vf, UVf, Ff = grid_mesh(s0 * u + t0 * n, u * (s1 - s0) / nsf, n * (t1 - t0) / ntf, nsf, ntf)
    Vw, UVw, Fw = grid_mesh(s0 * u + d_w * n + np.array([0.0, H, 0.0]), u * (s1 - s0) / 26, np.array([0.0, -H / 3, 0.0]), 26, 3)
    write_obj(ROOM_DIR / "room_mesh.obj", {"floor": (Vf, UVf, Ff), "room": (Vw, UVw, Fw)}, {"floor": "room_floor_tex.png", "room": "room_mesh_tex.png"})
    rep = {"floor_texture": {"px": [ns, nt], "m_per_px": res, "coverage": float(cover.mean()),
                             "cameras": {c: float(np.isfinite(tex[c][0][:, :, 0]).mean()) for c in tex}, "gain_01_to_02_bgr": g.tolist()},
           "wall_texture": {"px": [nsw, nyw], "m_per_px": resw, "coverage": float(wall_cover.mean()),
                            "cameras": {c: float(np.isfinite(tw_[c][0][:, :, 0]).mean()) for c in tw_}, "gain_01_to_02_bgr": gw.tolist()},
           "mesh": {"floor_triangles": int(len(Ff)), "wall_triangles": int(len(Fw)),
                    "files": ["room_mesh.obj", "room_mesh.mtl", "room_floor_tex.png", "room_mesh_tex.png"]}}
    (ROOM_DIR / "mesh_report.json").write_text(json.dumps(rep, indent=1))
    np.savez_compressed(RW / "final" / "tex_cover.npz", floor=cover, wall=wall_cover)
    log(json.dumps(rep))
    return rep


# ------------------------------------------------------------------------------------------------ review images


REVIEW_DIR = WORK / "review" / "room"


def room_point(model: dict, s, tt, y=0.0) -> np.ndarray:
    u, n = np.array(model["room_frame"]["u_along_wall"]), np.array(model["room_frame"]["n_wall_normal_to_cameras"])
    s, tt, y = np.broadcast_arrays(np.asarray(s, float), np.asarray(tt, float), np.asarray(y, float))
    return s[..., None] * u + tt[..., None] * n + y[..., None] * np.array([0.0, 1.0, 0.0])


def draw_room(im: np.ndarray, cd: dict, model: dict, pts: np.ndarray | None = None) -> None:
    fl, bw, ce = model["floor"], model["back_wall"], model["carpet_edge"]
    s0, s1 = fl["s"]
    t0, t1 = fl["t"]
    for s in np.arange(s0, s1 + 0.01, 1.0):
        D.draw_segment3d(im, room_point(model, s, t0), room_point(model, s, t1), cd, (0, 220, 255), 2 if round(s) % 5 == 0 else 1, 80)
    for tt in np.arange(np.ceil(t0), t1 + 0.01, 1.0):
        D.draw_segment3d(im, room_point(model, s0, tt), room_point(model, s1, tt), cd, (0, 220, 255), 1, 80)
    d_w, H = bw["plane"]["d"], bw["height_m"]
    D.draw_quad(im, [room_point(model, s0, d_w, 0), room_point(model, s1, d_w, 0), room_point(model, s1, d_w, H), room_point(model, s0, d_w, H)], cd, (255, 160, 0), 2)
    for s in np.arange(s0, s1 + 0.1, 2.0):
        D.draw_segment3d(im, room_point(model, s, d_w, 0), room_point(model, s, d_w, H), cd, (255, 160, 0), 1, 20)
    D.draw_segment3d(im, room_point(model, s0, ce["t"]), room_point(model, s1, ce["t"]), cd, (255, 0, 255), 3, 200)
    D.draw_axes(im, cd, (0, 0, 0), 1.5)
    if pts is not None and len(pts):
        D.draw_points(im, pts, cd, (255, 255, 255), 5)


def stage_review(a) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    model = json.loads((ROOM_DIR / "room_model.json").read_text())
    rj = json.loads((CAM_DIR / "rig.json").read_text())
    arr = {c: load_cam(c) for c in ("01", "02")}
    dw = np.load(CAM_DIR / "dancers_world.npz")
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    times = [22, 48, 74, 100, 126, 152, 176, 190]
    for cam in ("01", "02"):
        tiles, labels = [], []
        fr = [int(round((t if cam == "01" else t - R.SYNC_02) * FPS[cam])) for t in times]
        got = dict(F.read_video_frames(video(cam), *SIZE[cam], frames=fr))
        for t, f_ in zip(times, fr):
            im = got[f_].copy()
            i = row_of(arr[cam], f_)
            from .room_two_cams_walls import cam_dict
            cd = cam_dict(arr[cam], i)
            sel = (np.abs(dw["t"] - t) < 0.2) & dw["ok"] & (dw["err_px"] < 10) & np.isin(dw["joint"], [0, 15, 16])
            draw_room(im, cd, model, dw["X"][sel])
            tiles.append(im)
            labels.append(f"camera {cam} t={t}s f={cd['f']:.0f}px")
        cv2.imwrite(str(REVIEW_DIR / f"{cam}_contact.jpg"), D.contact_sheet(tiles, 2, 1280, labels), [cv2.IMWRITE_JPEG_QUALITY, 90])
        log(f"wrote {cam}_contact.jpg")
    # plan view
    fig, ax = plt.subplots(figsize=(11, 8))
    ok = dw["ok"] & (dw["err_px"] < 10) & np.isin(dw["joint"], [15, 16])
    ax.scatter(dw["X"][ok, 0], dw["X"][ok, 2], s=2, c="tab:green", alpha=0.25, label="dancers' ankles (triangulated)")
    dw_, te_ = model["back_wall"]["plane"]["d"], model["carpet_edge"]["t"]
    pw = np.array([room_point(model, s, dw_) for s in (-13, 13)])
    pe = np.array([room_point(model, s, te_) for s in (-13, 13)])
    ax.plot(pw[:, 0], pw[:, 2], c="tab:orange", lw=3, label=f"back wall: {dw_:.2f} m from the origin, yaw {model['room_frame']['yaw_deg']:.1f} deg")
    ax.plot(pe[:, 0], pe[:, 2], c="m", lw=2, label=f"carpet edge {te_:.2f} m")
    for cam, col in (("01", "tab:red"), ("02", "tab:blue")):
        C = np.array(rj["cameras"][cam]["centre_world_m"])
        yaw = np.degrees(np.arctan2(*(-arr[cam]["R"][len(arr[cam]["R"]) // 2][2, [0, 2]])))      # forward = row 2 of R_cw
        fwd = arr[cam]["R"][:, 2, :]
        ang = np.arctan2(fwd[:, 0], -fwd[:, 2])                                                   # angle from -z towards +x
        f_med = float(np.median(arr[cam]["K"][:, 0, 0]))
        half = np.degrees(np.arctan(960 / f_med))
        for a_ in (np.percentile(ang, 2), np.percentile(ang, 98)):
            for sgn in (-1, 1):
                aa = a_ + sgn * np.radians(half)
                ax.plot([C[0], C[0] + 22 * np.sin(aa)], [C[2], C[2] - 22 * np.cos(aa)], c=col, lw=0.8, alpha=0.7)
        ax.scatter([C[0]], [C[2]], c=col, s=90, marker="^", label=f"camera {cam}: ({C[0]:.2f}, {C[1]:.2f} m high, {C[2]:.2f})")
    ax.scatter([0], [0], c="k", marker="x")
    ax.set_aspect("equal")
    ax.set_xlim(-14, 14)
    ax.set_ylim(14, -11)
    ax.set_xlabel("x (m)  [world X: along the back wall]")
    ax.set_ylabel("z (m)  [towards the cameras]")
    ax.set_title("carlos_aline room, plan view (world frame, Y up). Wedges: the pan range of each camera")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.savefig(REVIEW_DIR / "plan_view.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    # evidence plots
    sw = np.load(RW / "final" / "wall_sweep.npz")
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    m = np.nanmean(sw["score"], 2)
    j = int(np.nanargmax(np.nanmax(m, 0)))
    axs[0].plot(sw["zs"], m[:, j], "o-")
    axs[0].set(title=f"stereo plane sweep: ZNCC vs wall z (yaw {sw['phis'][j]:.1f} deg)", xlabel="z (m)", ylabel="ZNCC of edge maps")
    wm = json.loads((ROOM_DIR / "walls_measure.json").read_text())
    names = ["wall base 01", "wall base 02", "carpet edge 01", "carpet edge 02"]
    vals = [wm["floor_lines"]["01"]["wall_base"]["z"], wm["floor_lines"]["02"]["wall_base"]["z"],
            wm["floor_lines"]["01"]["carpet_edge"]["z"], wm["floor_lines"]["02"]["carpet_edge"]["z"]]
    axs[1].barh(names, vals, color=["tab:red", "tab:blue", "tab:red", "tab:blue"])
    axs[1].axvline(model["back_wall"]["plane"]["z_at_x0"], c="tab:orange", label="sweep wall plane (z at x = 0)")
    axs[1].set(title="each camera alone, back-projected onto the floor", xlabel="z (m)")
    axs[1].legend()
    ch = wm.get("cornice_height_02")
    if ch:
        axs[2].plot(ch["heights"], ch["score"])
        axs[2].set(title=f"cornice scan on the wall plane (camera 02): {ch['h']:.2f} m", xlabel="height (m)", ylabel="contrast above - below")
    fig.savefig(REVIEW_DIR / "evidence.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    # textures
    if (ROOM_DIR / "room_floor_tex.png").exists():
        for nm, src in (("floor_texture", "room_floor_tex.png"), ("wall_texture", "room_mesh_tex.png")):
            im = cv2.imread(str(ROOM_DIR / src))
            cv2.imwrite(str(REVIEW_DIR / f"{nm}.jpg"), cv2.resize(im, (min(1600, im.shape[1]), int(im.shape[0] * min(1600, im.shape[1]) / im.shape[1]))))
    return {"files": sorted(p.name for p in REVIEW_DIR.iterdir())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("verticals")
    s = sub.add_parser("solve")
    s.add_argument("--yaw", type=float)
    u = sub.add_parser("uncertainty")
    u.add_argument("--part", default="jackknife", choices=["jackknife", "variants"])
    u.add_argument("--which", default="0")
    sub.add_parser("walls")
    mo = sub.add_parser("model")
    mo.add_argument("--wall-height", type=float, default=2.7)
    sub.add_parser("mesh")
    sub.add_parser("review")
    a = ap.parse_args(argv)
    if a.cmd == "verticals":
        stage_verticals(a)
    elif a.cmd == "solve":
        stage_solve(a)
    elif a.cmd == "uncertainty":
        stage_uncertainty(a)
    elif a.cmd == "walls":
        stage_walls(a)
    elif a.cmd == "model":
        stage_model(a)
    elif a.cmd == "mesh":
        stage_mesh(a)
    elif a.cmd == "review":
        stage_review(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
