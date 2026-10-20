r"""mono4d track "multihmr2" (2026-10-19): NAVER Multi-HMR 2 on ONE handheld phone (06) of larissa_kadu_38, 58-62 s.

Multi-HMR 2 (github.com/naver/multi-hmr2 @ f066532, NAVER non-commercial licence; checkpoint "Multi-HMR 2.b" =
multihmr2.pt from download.europe.naverlabs.com; DINOv3 ViT-L/16 backbone code, facebookresearch/dinov3 @ 6876159,
DINOv3 licence) detects every person in a frame, recovers an Anny body (163-bone MakeHuman rig) per person and places it
in metric camera space; per-person features drive an online tracker (FeatPelvisTracker) across video frames.

What runs (pod, dancecap/remote/cand_multihmr2/): the 120 canonical frames of the window (1080x1920, unchanged), the
official preprocessing (longest side 768 -> 432x768), one encoder pass per frame and the HPH decoder twice:
    ourK   OUR intrinsics (cameras/06.npz K, f = 1147 px, scaled to 768 px) passed as the decoder's K. The CLI has no K
           option, but the decoder takes K as an argument and K only sets the person translation (z = f / exp(d),
           x, y = (u - c) / exp(d)): pose and shape do not depend on it.     -> candidate "multihmr2" (the one asked for)
    predK  the encoder's own predicted camera (= `multihmr2 --video`)        -> candidate "multihmr2_predK" (diagnostic)
Anny -> SMPL-X (official path): Anny's "smplx" topology (anny.create_fullbody_model(..., topology="smplx"): the same
rig re-meshed onto the SMPL-X mesh with NAVER's non-commercial anny2smplx barycentric map) re-poses each detection's
captured Anny parameters into a 10475-vertex SMPL-X-topology mesh, placed with the decoder's translation; SMPL-X joints
= the neutral SMPL-X J_regressor (smplx/SMPLX_NEUTRAL.npz) applied to that POSED mesh (no SMPL-X parameter fit). Body
joints 0-21 go to the harness. The decoder itself freezes the neck, head, foot and toe bones (identity rotations:
Multi-HMR 2 does not predict them), so SMPL-X neck / head / feet follow the spine and shins.
Roles, two ways (same detections):
    tracks   dancecap.mono4d.candidates.assign_roles on the tracker's ids (projected SMPL-X joints vs dancers/roles.json
             boxes); same-role tracklets that never co-occur with the chosen one (mean IoU >= 0.3) are chained onto it
    maskid   per frame, from OUR SAM 2.1 dancer masks (projected body joints inside the mask + box IoU gate); the
             tracker ids are ignored. 2026-10-19 run: with ourK the tracker swapped the two dancers on 25 of 89 frames
             (none with predK), so "multihmr2" (ourK + tracks, as specified) carries those swaps.
Candidates: multihmr2 (ourK, tracks), multihmr2_maskid (ourK, maskid), multihmr2_predK (predK, maskid).
J_regressor-on-posed-mesh vs SMPL-X FK joints, measured on the reference fit's own meshes (58-62 s): PA-MPJPE-22 ~4 mm
(shoulders worst, ~13 mm): the conversion step itself is not what the scores measure.
Clock (audit fix 2026-10-19): the pod works on frames.json times (= roles.json / masks / people times), but the harness
and the reference fit run on the sync-corrected reference clock: saved t = frames.json t + phone 06's sync/offsets.json
delta (-30.04 ms, md.apply_offset - as cand_fitb / cand_closeapp do). Before the fix every candidate frame was compared
with the reference one frame (33 ms) later (time-shift test: follow PA-MPJPE minimum at -1 frame).

    python -m dancecap.mono4d.cand_multihmr2 prep         # frames + K + boxes + J_regressor -> mono4d/multihmr2/input/
    python -m dancecap.mono4d.cand_multihmr2 job           # dancecap/remote/cand_multihmr2/pod_job.json
    (DANCECAP_RUNPOD_CAP=... python -m dancecap remote run --jobs <abs path of that json> --max-cost 3 --gpus ...)
    python -m dancecap.mono4d.cand_multihmr2 import [--session SID]   # convert, roles, save, depth order, cand-score
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

from ..take import PACKAGE_ROOT, WORKSPACE
from . import candidates as C

TAKE, CAM = "larissa_kadu_38", "06"
WINDOW = (58.0, 62.0)
WORK = PACKAGE_ROOT / "work" / TAKE
OUT = WORK / "mono4d" / "multihmr2"
INPUT = OUT / "input"
POD_OUT = OUT / "pod_out"
JOB_DIR = PACKAGE_ROOT / "remote" / "cand_multihmr2"
MHMR2_COMMIT = "f066532d096034700e5ba8b7b1ca655729d3ea6e"
DINOV3_COMMIT = "6876159a11b4df116f30f667f8c9888617df0751"
LICENCE = ("Multi-HMR 2 code + checkpoint: NAVER non-commercial licence; Anny code Apache-2.0, its smplx topology data "
           "(anny2smplx.pth) NAVER non-commercial; DINOv3 code: DINOv3 licence; SMPL-X J_regressor: SMPL-X licence "
           "(non-commercial). Research use only; outputs stay local.")


def log(msg: str) -> None:
    print(f"[cand_multihmr2] {msg}", flush=True)


def _ws(p: Path) -> str:
    return str(Path(p).resolve().relative_to(WORKSPACE.resolve())).replace("\\", "/")


# ------------------------------------------------------------------------------------------------ prep + job


def prep(t0: float = WINDOW[0], t1: float = WINDOW[1]) -> dict:
    fj = json.loads((WORK / "frames" / CAM / "frames.json").read_text())
    frames = fj["frames"]
    cam = np.load(WORK / "cameras" / f"{CAM}.npz")
    if len(cam["t"]) != len(frames) or np.abs(cam["t"] - np.array([f["t"] for f in frames])).max() > 1e-3:
        raise SystemExit("cameras/06.npz and frames.json are not index-aligned")
    roles = json.loads((WORK / "dancers" / "roles.json").read_text())["cams"][CAM]["frames"]
    sel = [i for i, f in enumerate(frames) if t0 <= f["t"] <= t1]
    fd = INPUT / "frames"
    if fd.exists():
        shutil.rmtree(fd)
    fd.mkdir(parents=True)
    rows, boxes = [], {}
    for i in sel:
        src = WORK / "frames" / CAM / frames[i]["file"]
        shutil.copy2(src, fd / src.name)
        rows.append({"file": src.name, "i": i, "t": frames[i]["t"], "K": cam["K"][i].tolist()})
        r = roles[i]
        if abs(r["t"] - frames[i]["t"]) > 1e-3:
            raise SystemExit(f"roles.json frame {i} t {r['t']} != {frames[i]['t']}")
        boxes[str(i)] = {k: (r.get(k) or {}).get("box") for k in ("lead", "follow")}
    meta = {"take": TAKE, "cam": CAM, "window": [t0, t1], "width": fj["width"], "height": fj["height"], "frames": rows}
    (INPUT / "meta.json").write_text(json.dumps(meta, indent=1))
    (INPUT / "roles.json").write_text(json.dumps(boxes))
    sm = np.load(WORKSPACE / "smplx" / "SMPLX_NEUTRAL.npz", allow_pickle=True)
    np.savez(INPUT / "smplx_jreg.npz", J_regressor=np.asarray(sm["J_regressor"], np.float32))
    info = {"frames": len(rows), "first": rows[0]["file"], "last": rows[-1]["file"], "t": [rows[0]["t"], rows[-1]["t"]],
            "f_px": float(np.median(cam["K"][sel][:, 0, 0])), "size": [fj["width"], fj["height"]],
            "bytes": int(sum(p.stat().st_size for p in INPUT.rglob("*") if p.is_file()))}
    (INPUT / "prep.json").write_text(json.dumps(info, indent=1))
    return info


def job(max_minutes: float = 35, limit: int = 0) -> Path:
    POD_OUT_REL = _ws(POD_OUT)[len("dancecap/"):]
    lim = f" --limit {limit}" if limit else ""
    d = {"schema": "dancecap.pod_job/1", "name": "cand_multihmr2", "take": TAKE, "max_minutes": max_minutes, "needs": [],
         "inputs": [_ws(INPUT) + "/"],
         # no MAMMA code, body models or cached wheels: frames + code only over the shared 2.5 MB/s uplink
         "exclude": ["dancecap/work/runpod/pywheels", "dancecap/work/runpod/wheelhouse", "mamma-turing"],
         "env": {"MHMR2_COMMIT": MHMR2_COMMIT, "DINOV3_COMMIT": DINOV3_COMMIT, "MHMR2_ROOT": "/workspace/mhmr2_cand",
                 "TORCH_HOME": "/workspace/mhmr2_cand/torch_home", "ANNY_CACHE_DIR": "/workspace/mhmr2_cand/anny_cache"},
         "commands": [
             {"name": "install", "timeout_min": 22, "cmd": "bash remote/cand_multihmr2/install.sh"},
             {"name": "infer", "timeout_min": 12,
              "cmd": f"python remote/cand_multihmr2/mhmr2_pod.py --input {_ws(INPUT)[len('dancecap/'):]} "
                     f"--out {POD_OUT_REL}{lim}"}],
         "outputs": [_ws(POD_OUT) + "/"],
         "verify": [_ws(POD_OUT / "mhmr2_pod.json"), _ws(POD_OUT / "multihmr2_ourK.npz"),
                    _ws(POD_OUT / "multihmr2_predK.npz")],
         "after_download": ["python -m dancecap.mono4d.cand_multihmr2 import --session <sid>"],
         "notes": ["Multi-HMR 2.b (NAVER non-commercial) on the 120 canonical frames of phone 06, 58-62 s; decoder run "
                   "with our K (ourK) and with its predicted K (predK); Anny -> SMPL-X topology (anny2smplx, "
                   "non-commercial) + SMPL-X J_regressor on the pod. Checkpoint and Anny data stay on the pod; "
                   "uploads the dancers' frames (owner consent 2026-10-19) to our own pod only"]}
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    p = JOB_DIR / "pod_job.json"
    p.write_text(json.dumps(d, indent=1), newline="\n")
    return p


# ------------------------------------------------------------------------------------------------ import


def _session_meta(sid: str | None) -> dict:
    root = PACKAGE_ROOT / "work" / "runpod" / "sessions"
    if sid is None:
        cands = sorted((p for p in root.glob("*") if (p / "result.json").exists()
                        and '"cand_multihmr2"' in (p / "result.json").read_text(errors="ignore")),
                       key=lambda p: p.stat().st_mtime)
        if not cands:
            return {}
        sid = cands[-1].name
    p = root / sid / "result.json"
    if not p.exists():
        return {"session": sid}
    r = json.loads(p.read_text())
    pod = r.get("pod") or {}
    cloud = None
    try:
        led = json.loads((PACKAGE_ROOT / "work" / "runpod" / "ledger.json").read_text())
        rows = led if isinstance(led, list) else led.get("rows", led.get("pods", []))
        cloud = next((x.get("cloud") for x in rows if x.get("pod_id") == pod.get("id")), None)
    except Exception:  # noqa: BLE001
        pass
    job = (r.get("jobs") or {}).get("cand_multihmr2") or {}
    return {"session": sid, "gpu": pod.get("gpu"), "cloud": cloud, "cost_per_hr": pod.get("cost_per_hr"),
            "cost_usd": pod.get("ledger_cost"), "pod_minutes": round((pod.get("lifetime_s") or 0) / 60, 1),
            "job_status": job.get("status"), "job_seconds": job.get("seconds")}


def _tracks(z, n_frames: int) -> dict:
    """track id -> (F, 55, 3) camera-frame SMPL-X joints on the frame grid (NaN where the track has no detection)."""
    out = {}
    for tid in np.unique(z["track"]):
        J = np.full((n_frames, 55, 3), np.nan)
        m = z["track"] == tid
        J[z["frame"][m]] = z["j55"][m]
        out[int(tid)] = J
    return out


def _chain(tracks: dict, ass: dict, iou: dict, thr: float = 0.3) -> tuple[dict, dict]:
    """Chain same-role tracklets onto the assigned track: a tracklet joins a role when its mean box IoU with that
    role is >= thr, higher than with the other role, and it never co-occurs with what the role already holds."""
    held = {r: [ass[r]] for r in C.ROLES if r in ass}
    cover = {r: np.isfinite(tracks[ass[r]][:, 0, 0]) for r in held}
    used = set(ass.get(r) for r in C.ROLES)
    order = sorted((tid for tid in tracks if tid not in used),
                   key=lambda tid: -max(iou.get(f"{tid}:lead", 0), iou.get(f"{tid}:follow", 0)))
    for tid in order:
        s = {r: iou.get(f"{tid}:{r}", 0.0) for r in C.ROLES}
        r = max(s, key=s.get)
        o = "follow" if r == "lead" else "lead"
        if r not in held or s[r] < thr or s[r] <= s[o]:
            continue
        have = np.isfinite(tracks[tid][:, 0, 0])
        if (have & cover[r]).any():
            continue
        held[r].append(tid)
        cover[r] |= have
    J = {}
    for r, ids in held.items():
        A = np.full_like(tracks[ids[0]], np.nan)
        for tid in ids:
            m = np.isfinite(tracks[tid][:, 0, 0])
            A[m] = tracks[tid][m]
        J[r] = A
    return J, {r: ids for r, ids in held.items()}


def _mask_roles(z, Ks: np.ndarray, min_score: float = 0.4, min_iou: float = 0.3,
                dilate_px: int = 15) -> tuple[dict, np.ndarray, dict]:
    """Per-frame identities from OUR SAM 2.1 dancer masks (dancers/06/masks, id 01 lead / 02 follow), ignoring the
    tracker: every detection's 22 projected SMPL-X body joints are tested against both (dilated) masks; the detection
    -> role assignment maximising the summed inside-fraction wins, a role needs >= min_score of its joints inside AND
    its projected-joint box must overlap the role's roles.json box by >= min_iou (a small spectator standing behind a
    dancer projects inside her mask: 2026-10-19, frame 887, before this gate).
    Returns (J {role: (F, 55, 3)}, det index (F, 2) (-1 = none) for lead / follow, stats)."""
    import cv2
    from scipy.optimize import linear_sum_assignment
    F = len(z["frame_i"])
    J = {r: np.full((F, 55, 3), np.nan) for r in C.ROLES}
    pick = -np.ones((F, 2), int)
    ker = np.ones((2 * dilate_px + 1, 2 * dilate_px + 1), np.uint8)
    md = WORK / "dancers" / CAM / "masks"
    for k, i in enumerate(z["frame_i"]):
        dets = np.nonzero(z["frame"] == k)[0]
        masks = []
        for pid in ("01", "02"):
            m = cv2.imread(str(md / f"mask_{int(i):04d}_{pid}.png"), cv2.IMREAD_GRAYSCALE)
            masks.append(None if m is None else cv2.dilate((m > 127).astype(np.uint8), ker) > 0)
        if not len(dets) or all(m is None for m in masks):
            continue
        S = np.zeros((len(dets), 2))
        for a, d in enumerate(dets):
            P = z["j55"][d, :22]
            uv = (Ks[k] @ (P / P[:, 2:3]).T).T[:, :2]
            for r, m in enumerate(masks):
                if m is None:
                    continue
                u = np.clip(np.round(uv[:, 0]).astype(int), -1, m.shape[1])
                v = np.clip(np.round(uv[:, 1]).astype(int), -1, m.shape[0])
                ok = (u >= 0) & (u < m.shape[1]) & (v >= 0) & (v < m.shape[0])
                S[a, r] = float(m[v[ok], u[ok]].sum()) / 22.0
        rows, cols = linear_sum_assignment(-S)
        for a, r in zip(rows, cols):
            if S[a, r] >= min_score and z["iou_" + C.ROLES[r]][dets[a]] >= min_iou:
                pick[k, r] = dets[a]
                J[C.ROLES[r]][k] = z["j55"][dets[a]]
    return J, pick, {"coverage": {r: float((pick[:, n] >= 0).mean()) for n, r in enumerate(C.ROLES)},
                     "min_score": min_score, "min_iou": min_iou, "dilate_px": dilate_px}


def _ref_clock(frame_t: np.ndarray) -> np.ndarray:
    """frames.json times of phone 06 -> the reference clock of the 8-phone fit and the harness (t + sync delta)."""
    from .. import mamma_dyn as md
    return md.apply_offset(np.asarray(frame_t, float), md.read_sync_offsets(WORK).get(CAM))


def _ref_cam(frame_t: np.ndarray) -> tuple[dict, dict]:
    """Reference (8-phone fit) joints moved into phone 06's camera frame at the candidate's frame times (REFERENCE
    clock, see _ref_clock; the camera track of load_tracks is on that clock too)."""
    from .. import mamma_dyn as md
    from .holdout import load_fit, match_times
    tr = md.load_tracks(WORK, [CAM])[CAM]
    out, idx = {}, {}
    for role in C.ROLES:
        r = load_fit(WORK / "smplx", role)
        ir, it = match_times(r["t"], frame_t)
        k = np.array([int(np.argmin(np.abs(tr.t - x))) for x in frame_t[it]])
        Xw = r["joints"][ir][:, :22]
        out[role] = np.einsum("fij,fnj->fni", tr.R[k], Xw) + tr.tvec[k][:, None]
        idx[role] = it
    return out, idx


def depth_order(J: dict, frame_t: np.ndarray, thr: float = 0.15) -> dict:
    """Lead-vs-follow front/back ordering in phone 06's camera (pelvis z) against the reference, per frame."""
    ref, idx = _ref_cam(frame_t)
    common = np.intersect1d(idx["lead"], idx["follow"])
    zr = {r: dict(zip(idx[r].tolist(), ref[r][:, 0, 2])) for r in C.ROLES}
    pr = {r: dict(zip(idx[r].tolist(), ref[r][:, 0])) for r in C.ROLES}
    rows = []
    for k in common:
        cl, cf = J["lead"][k, 0], J["follow"][k, 0]
        if not (np.isfinite(cl).all() and np.isfinite(cf).all()):
            continue
        dz_ref = zr["lead"][k] - zr["follow"][k]
        dz_c = cl[2] - cf[2]
        rows.append({"frame": int(k), "t": float(frame_t[k]), "dz_ref": float(dz_ref), "dz_cand": float(dz_c),
                     "agree": bool(np.sign(dz_ref) == np.sign(dz_c)),
                     "dist_ref": float(np.linalg.norm(pr["lead"][k] - pr["follow"][k])),
                     "dist_cand": float(np.linalg.norm(cl - cf)),
                     "z_err_lead": float(cl[2] - zr["lead"][k]), "z_err_follow": float(cf[2] - zr["follow"][k]),
                     "pelvis_cam_err_lead": float(np.linalg.norm(cl - pr["lead"][k])),
                     "pelvis_cam_err_follow": float(np.linalg.norm(cf - pr["follow"][k]))})
    if not rows:
        return {"frames": 0}
    a = lambda key: np.array([r[key] for r in rows])  # noqa: E731
    big = np.abs(a("dz_ref")) >= thr
    med = lambda x: round(float(np.median(x)), 3) if len(x) else None  # noqa: E731
    return {"frames": len(rows), "agree_frac": round(float(a("agree").mean()), 3),
            f"agree_frac_absdz_ref_ge_{thr}": round(float(a("agree")[big].mean()), 3) if big.any() else None,
            f"frames_absdz_ref_ge_{thr}": int(big.sum()),
            "ref_lead_in_front_frac": round(float((a("dz_ref") < 0).mean()), 3),
            "cand_lead_in_front_frac": round(float((a("dz_cand") < 0).mean()), 3),
            "dz_err_m_median_abs": med(np.abs(a("dz_cand") - a("dz_ref"))),
            "pelvis_distance_m": {"ref_median": med(a("dist_ref")), "cand_median": med(a("dist_cand")),
                                  "abs_err_median": med(np.abs(a("dist_cand") - a("dist_ref")))},
            "depth_err_m_median": {"lead": med(a("z_err_lead")), "follow": med(a("z_err_follow"))},
            "pelvis_cam_err_m_median": {"lead": med(a("pelvis_cam_err_lead")), "follow": med(a("pelvis_cam_err_follow"))},
            "per_frame": rows}


CANDS = {("ourK", "tracks"): "multihmr2", ("ourK", "maskid"): "multihmr2_maskid", ("predK", "maskid"): "multihmr2_predK"}


def _score_unsaved(J: dict, frame_t: np.ndarray) -> dict:
    """The harness's score_arrays for a camera-frame candidate that is not saved (same path as candidates.score)."""
    from .. import mamma_dyn as md
    from .holdout import load_fit
    tr = md.load_tracks(WORK, [CAM])[CAM]
    out = {}
    for role, A in J.items():
        m = np.isfinite(A[:, 0, 0])
        k = np.array([int(np.argmin(np.abs(tr.t - x))) for x in frame_t[m]])
        Jw = C.to_world(A[m][:, :22], tr.R[k], tr.tvec[k])
        r = load_fit(WORK / "smplx", role)
        s = C.score_arrays(frame_t[m], Jw, "world", r["t"], r["joints"], WINDOW)
        out[role] = {kk: (v.get("median") if isinstance(v, dict) else v) for kk, v in s.items()}
    return out


def import_run(sid: str | None = None, pod_out: Path | None = None) -> dict:
    pod_out = Path(pod_out) if pod_out else POD_OUT
    pod = json.loads((pod_out / "mhmr2_pod.json").read_text())
    ses = _session_meta(sid)
    meta_in = json.loads((INPUT / "meta.json").read_text())
    K_ours = np.median(np.array([f["K"] for f in meta_in["frames"]]), axis=0)
    report = {"session": ses, "pod": {k: pod.get(k) for k in ("versions", "timings", "checks", "f_pred_px",
                                                                "gpu_mem_peak_gb", "errors", "ourK", "predK")},
              "variants": {}}
    for var in ("ourK", "predK"):
        p = pod_out / f"multihmr2_{var}.npz"
        if not p.exists():
            report["variants"][var] = {"missing": str(p)}
            continue
        z = np.load(p)
        frame_t = np.asarray(z["frame_t"], float)   # frames.json clock: roles.json boxes / masks are looked up on it
        t_ref = _ref_clock(frame_t)                  # reference clock: everything compared with the reference fit
        F = len(frame_t)
        if var == "ourK":
            Ks = np.repeat(K_ours[None], F, 0)
        else:  # its own per-frame predicted focal, principal point at the image centre (as the encoder sets it)
            Ks = np.repeat(K_ours[None], F, 0)
            Ks[:, 0, 0] = Ks[:, 1, 1] = z["f_pred_full"]
        K_med = np.median(Ks, axis=0)
        tracks = _tracks(z, F)
        ass = C.assign_roles({tid: J[:, :22] for tid, J in tracks.items()}, TAKE, CAM, K_med, frame_t)
        raw_cov = {r: float(np.isfinite(tracks[ass[r]][:, 0, 0]).mean()) for r in C.ROLES if r in ass}
        J_tr, chains = _chain(tracks, ass, ass["iou"])
        J_mk, pick, mstats = _mask_roles(z, Ks)
        # tracker identities vs mask identities on frames where both give both dancers
        tid_of = lambda d: int(z["track"][d])  # noqa: E731
        agree = swap = other = 0
        for k in range(F):
            if (pick[k] >= 0).all() and all(np.isfinite(J_tr[r][k, 0, 0]) for r in C.ROLES):
                tl, tf = tid_of(pick[k, 0]), tid_of(pick[k, 1])
                held = {r: [t for t in chains[r] if np.isfinite(tracks[t][k, 0, 0])][0] for r in C.ROLES}
                if (tl, tf) == (held["lead"], held["follow"]):
                    agree += 1
                elif (tl, tf) == (held["follow"], held["lead"]):
                    swap += 1
                else:
                    other += 1
        top = sorted(((k, v) for k, v in ass["iou"].items() if v > 0.05), key=lambda kv: -kv[1])[:12]
        info = {"frames": F, "tracks": len(tracks), "persons_per_frame_median": float(np.median(z["n_per_frame"])),
                "f_used_px_median": float(K_med[0, 0]), "f_pred_px_median": float(np.median(z["f_pred_full"])),
                "tracks_method": {"assigned": {r: ass.get(r) for r in C.ROLES}, "assigned_track_coverage": raw_cov,
                                  "chained_tracks": chains, "top_iou": dict(top)},
                "maskid_method": mstats,
                "tracker_vs_masks": {"frames_both": agree + swap + other, "same": agree, "swapped": swap,
                                     "other": other},
                "methods": {}}
        for meth, J in (("tracks", J_tr), ("maskid", J_mk)):
            dep = depth_order(J, t_ref)
            (OUT / f"depth_order_{var}_{meth}.json").write_text(json.dumps(dep, indent=1))
            cov = {r: float(np.isfinite(J[r][:, 0, 0]).mean()) for r in J}
            name = CANDS.get((var, meth))
            row = {"candidate": name, "coverage": cov, "depth_order": {k: v for k, v in dep.items() if k != "per_frame"}}
            if name:
                for role in C.ROLES:
                    m = np.isfinite(J[role][:, 0, 0])
                    how = ("assign_roles on the tracker ids (+ same-role tracklet chaining: mean box IoU >= 0.3, no "
                           "overlap)" if meth == "tracks" else
                           "per-frame identities from our SAM 2.1 dancer masks (22 projected body joints inside the "
                           "15-px-dilated mask, Hungarian, >= 40 % inside, joint box IoU with the roles.json box >= 0.3); the tracker ids are ignored")
                    C.save(TAKE, name, role, t_ref[m], J[role][m][:, :22].copy(), "camera", {
                        "method": "NAVER Multi-HMR 2.b (multihmr2.pt) on phone 06, "
                                  + ("OUR K (f %.0f px) passed to the decoder" % K_med[0, 0] if var == "ourK" else
                                     "its own per-frame predicted K (f %.0f px median)" % K_med[0, 0])
                                  + "; Anny -> SMPL-X topology (anny 0.6.0 anny2smplx) + SMPL-X neutral J_regressor on "
                                    "the posed mesh; roles: " + how,
                        "input_cam": CAM, "version": f"multi-hmr2 {MHMR2_COMMIT[:7]}, dinov3 {DINOV3_COMMIT[:7]}, "
                                                     "anny 0.6.0", "licence": LICENCE, "gpu": ses.get("gpu"),
                        "cloud": ses.get("cloud"), "cost_usd": ses.get("cost_usd"), "wall_min": ses.get("pod_minutes"),
                        "session": ses.get("session"), "variant": var, "roles_method": meth, "source": str(pod_out),
                        "coverage": cov.get(role),
                        "clock": "t = frames.json t + sync/offsets.json delta of phone 06 (%+.2f ms): reference clock"
                                 % (1e3 * float(t_ref[0] - frame_t[0])),
                        "notes": "the Multi-HMR 2 decoder freezes neck/head/foot/toe bones (identity rotations); "
                                 "frames where the dancer is not detected are absent"})
                sc = C.score(TAKE, name, WINDOW)["roles"]
                row["score"] = {r: {k: (v.get("median") if isinstance(v, dict) else v) for k, v in s.items()}
                                for r, s in sc.items()}
            else:
                row["score"] = _score_unsaved(J, t_ref)
            info["methods"][meth] = row
            log(f"{var}/{meth} -> {name}: coverage {cov} depth agree {row['depth_order'].get('agree_frac')} "
                f"score {row['score']}")
        report["variants"][var] = info
    report["leaderboard"] = C.leaderboard(TAKE, WINDOW)
    report["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (OUT / "import.json").write_text(json.dumps(report, indent=1, default=str))
    return report


BONES = [(0, 1), (0, 2), (0, 3), (3, 6), (6, 9), (9, 12), (12, 15), (1, 4), (2, 5), (4, 7), (5, 8), (7, 10), (8, 11),
         (9, 13), (9, 14), (13, 16), (14, 17), (16, 18), (17, 19), (18, 20), (19, 21)]


def render(name: str = "multihmr2", n: int = 6, width: int = 360) -> Path:
    """Contact sheet: n frames of phone 06 with the candidate's 22 body joints (lead orange, follow cyan; projected with
    the K the candidate was placed with) and the 8-phone reference reprojected through phone 06's track (white)."""
    import cv2
    meta = json.loads((INPUT / "meta.json").read_text())
    fr = meta["frames"]
    ft = _ref_clock(np.array([f["t"] for f in fr]))   # candidate t and the reference are on the reference clock
    d = C.cand_dir(TAKE, name)
    m = json.loads((d / "meta.json").read_text())
    K = np.median(np.array([f["K"] for f in fr]), axis=0)
    if m.get("variant") == "predK":
        z = np.load(POD_OUT / "multihmr2_predK.npz")
        K = K.copy()
        K[0, 0] = K[1, 1] = float(np.median(z["f_pred_full"]))
    ref, ridx = _ref_cam(ft)
    picks = np.linspace(0, len(fr) - 1, n).round().astype(int)
    tiles = []
    col = {"lead": (0, 140, 255), "follow": (255, 220, 0)}
    for k in picks:
        img = cv2.imread(str(INPUT / "frames" / fr[k]["file"]))
        for role in C.ROLES:
            for src, J, c, th in (("ref", None, (255, 255, 255), 2), ("cand", None, col[role], 4)):
                if src == "ref":
                    w = np.nonzero(ridx[role] == k)[0]
                    if not len(w):
                        continue
                    P, KK = ref[role][w[0]], np.median(np.array([f["K"] for f in fr]), axis=0)
                else:
                    p = d / f"{role}.npz"
                    if not p.exists():
                        continue
                    zz = np.load(p)
                    w = np.nonzero(np.abs(zz["t"] - ft[k]) < 1e-3)[0]
                    if not len(w):
                        continue
                    P, KK = zz["joints"][w[0]][:22], K
                uv = (KK @ (P / P[:, 2:3]).T).T[:, :2]
                for a, b in BONES:
                    if np.isfinite(uv[[a, b]]).all():
                        cv2.line(img, tuple(int(x) for x in uv[a]), tuple(int(x) for x in uv[b]), c, th, cv2.LINE_AA)
        cv2.putText(img, f"{name} t={ft[k]:.2f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3)
        h = int(img.shape[0] * width / img.shape[1])
        tiles.append(cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA))
    sheet = np.concatenate(tiles, axis=1)
    out = OUT / "review" / f"{name}_sheet.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.mono4d.cand_multihmr2")
    ap.add_argument("action", choices=["prep", "job", "import", "render"])
    ap.add_argument("--name", default="multihmr2")
    ap.add_argument("--session")
    ap.add_argument("--pod-out")
    ap.add_argument("--max-minutes", type=float, default=35)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    if a.action == "prep":
        print(json.dumps(prep(), indent=1))
    elif a.action == "job":
        print(job(a.max_minutes, a.limit))
    elif a.action == "render":
        print(render(a.name))
    else:
        r = import_run(a.session, a.pod_out)
        for v, i in r["variants"].items():
            print(v, json.dumps({k: x for k, x in i.items() if k != "methods"}, default=str))
            for meth, row in (i.get("methods") or {}).items():
                print(f"  {meth}: {json.dumps(row, default=str)}")
        print(r["leaderboard"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
