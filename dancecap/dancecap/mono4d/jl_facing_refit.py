r"""john_lula realism v2, CORRECT (2026-10-20): fix the FACING errors of the pose with the image evidence, refit the chunks
that hold them, re-stitch / refine as jl_refit did, compare, then lock the feet.

Owner, 2026-10-20: "there is too much foot slip and also the follow (lula) spins and shows her back, but the post track is
not capturing that (we should be able to spot check forward-face/back to correct a pose error like that)" and "monocular
needs a lot of improvement to be useful. do what you can".

INPUT  realism/v2/facing/facing_targets.json (jl_facing targets: per video frame and dancer the image class front / side /
       back, its confidence, the target world yaw + band, the fit's action keep / rotate / flip180 / none) and, when it
       exists, realism/v2/facing/owner_labels.json (the owner's spot-check labels: jl_facing targets() is re-run so they win).
       Fit lattice t_k = k / 30 (2643 instants); video frame of k = the nearest pts (|dt| <= 17 ms).
  prior   realism/sam3db/prior_3db_full.npz (SAM 3D Body, world joints + global_orient + body_pose + conf) CORRECTED on the
          frames the fit gets wrong (targets action rotate / flip180) or the owner labelled, where the confident image evidence
          (posterior >= 0.8, or the owner) says the PRIOR's own torso facing is wrong too: class opposite
          (front <-> back) -> the body turned 180 deg about the vertical through its pelvis; front / back <-> side more than
          BOUNDARY_DEG outside the band -> turned into the band + PRIOR_MARGIN_DEG (smoothed in the run, cosine ramps of
          RAMP frames). Turning = R_y(d) applied to the root-relative joints and composed onto global_orient: body_pose (the
          root-relative pose) is unchanged. Where the FIT is wrong (action rotate / flip180) and the (corrected) prior is in the
          band, its conf is raised to at least CONF_FLOOR (Lula 27.97-28.17 s: SAM 3D Body faces the phone like the picture,
          conf 0.25 because she is half hidden; the fit interpolated her round to -110..-126 deg).
          -> realism/v2/refit/prior_3db_face_full.npz
  hint    MAMMA_FACING_HINT (mamma-turing utils/mono.py facing_hint_loss, a NEW env-gated term, default off): per frame the
          target torso heading, a band and a weight. Frames the fit gets wrong (rotate / flip180): the class band itself
          (front / back +-60 deg); frames the fit gets right (keep, image or owner, not the agent's false alarms): the band +
          the 20 deg boundary allowance (+-80 deg, a guard so the refit cannot leave it); side only from the owner (+-30).
          Weight = the image confidence. -> realism/v2/refit/facing_hint_full.npz
  refit   ONLY the chunks that own a corrected window (facing_targets windows not judged false alarms in
          agent_window_verdicts.json): same recipe as jl_refit (J3b inputs copied from realism/refit/chunks/, betas pinned,
          MAMMA_MONO, prior w 0.5 as the start) + facing_hint_loss weight HINT_W in all four stages; one fit per pod.
          realism/v2/refit/chunks/john_lula_<chunk>/ (take name john_lula/realism/v2/refit/chunks/john_lula_<chunk>)
  stitch  dancecap.stitch with jl_refit's arguments (chunks not refit = the jl_refit chunk folders, unchanged), jl_repair on
          both folders -> realism/v2/refit/stitch/{smplx, smplx_rf2_noslip}; smplx_3db_v2/ = the repaired raw stitch (the smplx/
          contract). The jl_refit audit patch (John 26.20-26.43 s blended from the J3b fit smplx/) is re-applied when the
          lurch is still there -> realism/v2/refit/audit_patch/{smplx_3db_v2_patched, smplx_rf2_noslip_patched}.
  compare realism/v2/refit/compare.json: old (realism/refit) vs new on facing vs the images, RTMW 2D, 512 landmarks, jitter,
          couple order, the NOTES_REALISM_NEXT problem windows.
  feet    jl_footlock.footlock_fit(new refined fit) -> realism/v2/footlock/final/ = THE RENDER DRIVER.

    python -m dancecap.mono4d.jl_facing_refit prior | build | job | stitch | compare | sheets | footlock
Never touched: work/john_lula/{smplx, smplx_rf2_noslip, smplx_3db, avatar, export}/, realism/refit/ (read only).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from ..take import PACKAGE_ROOT, WORKSPACE

TAKE, CAM = "john_lula", "01"
ROLES = ("lead", "follow")
NAME = {"lead": "John", "follow": "Lula"}
WORK = PACKAGE_ROOT / "work" / TAKE
REAL = WORK / "realism"
V2 = REAL / "v2"
FACING = V2 / "facing"
TARGETS = FACING / "facing_targets.json"
OWNER_LABELS = FACING / "owner_labels.json"
VERDICTS = FACING / "agent_window_verdicts.json"
RF2 = V2 / "refit"
CHUNK_ROOT2 = RF2 / "chunks"
STITCH2 = RF2 / "stitch"
PATCH2 = RF2 / "audit_patch"
FREP = RF2 / "facing_repair"
OUT_V2 = WORK / "smplx_3db_v2"
SHEETS2 = V2 / "sheets"
FOOT_FINAL = V2 / "footlock" / "final"
PRIOR_FULL = REAL / "sam3db" / "prior_3db_full.npz"
PRIOR_FACE = RF2 / "prior_3db_face_full.npz"
HINT_FULL = RF2 / "facing_hint_full.npz"
CORRECTIONS = RF2 / "corrections.json"
OLD_REFIT = REAL / "refit"
OLD_CHUNKS = OLD_REFIT / "chunks"
OLD_BEST = OLD_REFIT / "audit_patch" / "smplx_rf2_noslip_3db_patched"
OLD_RAW = WORK / "smplx_3db"
OLD_RF2 = OLD_REFIT / "stitch" / "smplx_rf2_noslip"
JOB_DIR = PACKAGE_ROOT / "remote" / "jl_facing_refit"
POD_WS = "/workspace/HeadMovement"
CHUNKS = {"jlg0000": (0, 1381), "jlx1158": (1158, 1518), "jlg1322": (1262, 2642)}   # = jl_refit.CHUNKS
SET = "c1"
CFG_FACE = "ma3d_config_3db_face.yaml"
PRIOR_RUNS = ("first_run", "second_run", "third_run", "fourth_run")
HINT_W = 10.0              # facing_hint_loss weight (per frame ~half the 2D pull at 25 deg in a visible frame; dominant where hidden)
HINT_TOL_FIX_DEG = 60.0    # band half-width on frames the fit gets wrong (= the class band)
HINT_TOL_KEEP_DEG = 80.0   # ... on frames it gets right (band + the 20 deg boundary allowance of jl_facing targets)
HINT_TOL_SIDE_DEG = 30.0   # owner SIDE labels only
PRIOR_MARGIN_DEG = 5.0     # a corrected prior lands this far inside the band
BOUNDARY_DEG = 20.0        # = jl_facing.BOUNDARY_DEG: a prior closer than this to the band is left alone (image calls only)
CONF_MIN = 0.8             # = jl_facing.CONF_MIN
CONF_FLOOR = 0.5           # prior conf floor where the fit is wrong and the prior's facing agrees with the picture
RAMP = 5                   # cosine ramp frames of a prior rotation run
FRONT_MAX, BACK_MIN = 60.0, 120.0
CLS = {"front": 0, "side": 1, "back": 2}
PROTECTED = [WORK / d for d in ("smplx", "smplx_rf2_noslip", "smplx_3db", "avatar", "export")]
STITCH_ARGS = ["--partial", "--force", "--select-weights", "arm=0.3,fingers=0.3"]   # = jl_refit.STITCH_ARGS


def log(msg: str) -> None:
    print(f"[jl_facing_refit {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _ws(p: Path) -> str:
    return Path(p).resolve().relative_to(WORKSPACE.resolve()).as_posix()


def _work_rel(p: Path) -> str:
    return Path(p).resolve().relative_to((PACKAGE_ROOT / "work").resolve()).as_posix()


def chunk_dir(cid: str) -> Path:
    return CHUNK_ROOT2 / f"{TAKE}_{cid}"


def old_chunk_dir(cid: str) -> Path:
    return OLD_CHUNKS / f"{TAKE}_{cid}"


def take_name(cid: str) -> str:
    return _work_rel(chunk_dir(cid))


def sha1(p: Path) -> str:
    return hashlib.sha1(Path(p).read_bytes()).hexdigest()


def tree_digest(paths: list[Path]) -> dict:
    out = {}
    for d in paths:
        for f in sorted(Path(d).rglob("*")):
            if f.is_file():
                out[str(f.relative_to(WORK))] = sha1(f)
    return out


# ================================================================================================ pure helpers (tested)
def wrap_deg(a):
    return (np.asarray(a, float) + 180.0) % 360.0 - 180.0


def rot_y(theta: np.ndarray) -> np.ndarray:
    """(N,) rad -> (N, 3, 3) rotations about world +Y (a heading atan2(x, z) increases by theta)."""
    th = np.asarray(theta, float).reshape(-1)
    c, s = np.cos(th), np.sin(th)
    R = np.zeros((len(th), 3, 3))
    R[:, 0, 0], R[:, 0, 2], R[:, 1, 1], R[:, 2, 0], R[:, 2, 2] = c, s, 1.0, -s, c
    return R


def rotate_about_vertical(J: np.ndarray, go: np.ndarray, delta_rad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Turn bodies about the vertical through their pelvis: J (N, 22, 3) world joints, go (N, 3) world axis-angle, delta (N,) rad.
    Returns (J', go'): J' = R_y(d) (J - J_0) + J_0, go' = R_y(d) R(go); body_pose (root-relative) is unchanged by construction."""
    from scipy.spatial.transform import Rotation as Rot
    J = np.asarray(J, float)
    Ry = rot_y(delta_rad)
    P = J[:, :1]
    Jn = np.einsum("nij,nkj->nki", Ry, J - P) + P
    gon = (Rot.from_matrix(Ry) * Rot.from_rotvec(np.asarray(go, float))).as_rotvec()
    return Jn, gon


def torso_forward_np(J: np.ndarray) -> np.ndarray:
    """(..., >=18, 3) -> (..., 3) torso forward (jl_facing.torso_axes; the same vectors mono.torso_forward uses)."""
    from .jl_facing import torso_axes
    return torso_axes(J)[0]


def yaw_deg(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float)
    return np.degrees(np.arctan2(v[..., 0], v[..., 2]))


def band_target(cls: int, cam_yaw: float, cur_yaw: float) -> tuple[float, float]:
    """World yaw the torso should have for image class cls given the pelvis -> camera yaw; band half-width (deg)."""
    if cls == 0:
        return float(wrap_deg(cam_yaw)), FRONT_MAX
    if cls == 2:
        return float(wrap_deg(cam_yaw + 180.0)), 180.0 - BACK_MIN
    a, b = wrap_deg(cam_yaw + 90.0), wrap_deg(cam_yaw - 90.0)
    t = a if abs(wrap_deg(a - cur_yaw)) <= abs(wrap_deg(b - cur_yaw)) else b
    return float(t), (BACK_MIN - FRONT_MAX) / 2.0


def class_of(cur_yaw: float, cam_yaw: float) -> int:
    rel = abs(float(wrap_deg(cam_yaw - cur_yaw)))
    return 0 if rel <= FRONT_MAX else (2 if rel >= BACK_MIN else 1)


def prior_action(cls_img: int, cam_yaw: float, prior_yaw: float, from_owner: bool = False,
                 margin: float = PRIOR_MARGIN_DEG, boundary: float = BOUNDARY_DEG) -> tuple[str, float]:
    """What the confident image class says about the PRIOR's torso yaw: ('keep' | 'rotate' | 'flip180', delta deg about +Y).
    Opposite class (front <-> back): 180 deg. Otherwise outside the band: the smallest turn into it + margin; an image call
    (not the owner's) whose prior is within `boundary` of the band is left alone (jl_facing targets' boundary rule)."""
    pc = class_of(prior_yaw, cam_yaw)
    if pc == cls_img:
        return "keep", 0.0
    if {pc, cls_img} == {0, 2}:
        return "flip180", 180.0
    ty, half = band_target(cls_img, cam_yaw, prior_yaw)
    d = float(wrap_deg(ty - prior_yaw))
    if abs(d) <= half:
        return "keep", 0.0
    need = d - np.copysign(half, d)
    if not from_owner and abs(need) < boundary:
        return "keep", 0.0
    return "rotate", float(need + np.copysign(margin, need))


def smooth_rotation_runs(delta: np.ndarray, kind: np.ndarray, ramp: int = RAMP, gap: int = 2) -> np.ndarray:
    """Per-frame prior turns (deg) -> the applied turns: 'rotate' runs (gaps <= gap frames closed) get a 3-frame median then a
    5-frame mean inside the run and cosine ramps of `ramp` frames outside it (never over a frame of another run or a flip);
    'flip180' frames keep exactly 180 (no ramp: a half turn has no halfway). kind (T,) str."""
    d = np.where(np.asarray(kind) == "rotate", np.asarray(delta, float), 0.0)
    T = len(d)
    out = np.zeros(T)
    m = d != 0
    idx = np.flatnonzero(m)
    runs, s = [], None
    for i in idx:
        if s is None:
            s = p = i
        elif i - p <= gap + 1:
            p = i
        else:
            runs.append((s, p))
            s = p = i
    if s is not None:
        runs.append((s, p))
    for a, b in runs:
        seg = np.arange(a, b + 1)
        v = d[seg].copy()
        have = v != 0
        v[~have] = np.interp(seg[~have], seg[have], v[have]) if have.any() else 0.0   # closed gaps
        if len(v) >= 3:
            v = np.array([np.median(v[max(0, i - 1):i + 2]) for i in range(len(v))])
        if len(v) >= 5:
            v = np.array([v[max(0, i - 2):i + 3].mean() for i in range(len(v))])
        out[seg] = v
        for j in range(1, ramp + 1):
            w = 0.5 * (1 + np.cos(np.pi * j / (ramp + 1)))
            for i, edge in ((a - j, v[0]), (b + j, v[-1])):
                if 0 <= i < T and not m[i] and abs(out[i]) < abs(w * edge):
                    out[i] = w * edge
    flip = np.asarray(kind) == "flip180"
    out[flip] = 180.0
    return out


def windows_to_refit(windows: list[dict], verdicts: list[dict], owner: np.ndarray, f_to_k) -> dict:
    """{chunk: [window ids]} of the chunks that OWN (stitch segment_owner of the fit lattice) a frame of a window that is not
    a false alarm. windows: facing_targets 'windows' (video frames start..end); verdicts: agent_window_verdicts 'windows'."""
    fa = {(v["role"], int(v["start"]), int(v["end"])) for v in verdicts if v.get("verdict") == "false_alarm"}
    out: dict = {}
    for w in windows:
        if (w["role"], int(w["start"]), int(w["end"])) in fa:
            continue
        ks = np.unique([f_to_k(f) for f in range(int(w["start"]), int(w["end"]) + 1)])
        for c in np.unique(np.asarray(owner)[ks]).astype(str):
            out.setdefault(str(c), []).append(w["id"])
    return out


# ================================================================================================ evidence on the lattice
def _lattice() -> dict:
    """The fit lattice t_k = k / 30 and its nearest video frame; camera centre; the old best fit's pelvis."""
    from .jl_facing import Take
    T = Take()
    tg = np.arange(len(T.t)) / 30.0
    with np.load(OLD_BEST / "lead.npz", allow_pickle=True) as z:
        if len(z["t"]) != len(tg) or np.abs(np.asarray(z["t"], float) - tg).max() > 1e-6:
            raise SystemExit("the best fit's grid is not k / 30 over the video's frame count")
        owner = np.asarray(z["segment_owner"]).astype(str)
    fk = np.clip(np.round(np.interp(tg, T.t, np.arange(T.F))).astype(int), 0, T.F - 1)
    if np.abs(T.t[fk] - tg).max() > 0.02:
        raise SystemExit("lattice -> video frame mapping off by more than 20 ms")
    C = np.median(T.C, axis=0)
    pel = {}
    for r in ROLES:
        with np.load(OLD_BEST / f"{r}.npz", allow_pickle=True) as z:
            pel[r] = np.asarray(z["joints"], float)[:, 0]
    return {"t": tg, "f_of_k": fk, "C": C, "pelvis": pel, "owner": owner, "video_t": T.t,
            "k_of_f": np.clip(np.round(T.t * 30.0).astype(int), 0, len(tg) - 1)}


def load_targets() -> dict:
    """facing_targets.json (re-made with the owner's labels first when owner_labels.json is newer than it)."""
    if OWNER_LABELS.exists() and (not TARGETS.exists() or OWNER_LABELS.stat().st_mtime > TARGETS.stat().st_mtime):
        from . import jl_facing
        log("owner_labels.json is newer than facing_targets.json: re-making the targets (the owner's labels win)")
        jl_facing.targets()
    return json.loads(TARGETS.read_text())


# ================================================================================================ prior + hint
def prior(write: bool = True) -> dict:
    L = _lattice()
    tgt = load_targets()
    z = dict(np.load(PRIOR_FULL, allow_pickle=True))
    T = len(L["t"])
    if z["joints"].shape[:2] != (T, 2) or np.abs(np.asarray(z["t"], float) - L["t"]).max() > 1e-9:
        raise SystemExit("prior_3db_full.npz is not on the fit lattice")
    J = np.asarray(z["joints"], float).copy()
    go = np.asarray(z["global_orient"], float).copy()
    conf = np.asarray(z["conf"], float).copy()
    hint_y = np.full((T, 2), np.nan)
    hint_tol = np.full((T, 2), np.nan)
    hint_c = np.zeros((T, 2))
    rep = {"what": "SAM 3D Body prior corrected by the image facing evidence + the MAMMA_FACING_HINT targets",
           "made": time.strftime("%Y-%m-%dT%H:%M:%S"), "targets": str(TARGETS), "owner_labels": str(OWNER_LABELS) if OWNER_LABELS.exists() else None,
           "params": {"HINT_W": HINT_W, "HINT_TOL_FIX_DEG": HINT_TOL_FIX_DEG, "HINT_TOL_KEEP_DEG": HINT_TOL_KEEP_DEG,
                      "HINT_TOL_SIDE_DEG": HINT_TOL_SIDE_DEG, "PRIOR_MARGIN_DEG": PRIOR_MARGIN_DEG, "BOUNDARY_DEG": BOUNDARY_DEG,
                      "CONF_MIN": CONF_MIN, "CONF_FLOOR": CONF_FLOOR, "RAMP": RAMP}, "roles": {}}
    per_frame = {}
    for o, r in enumerate(ROLES):
        cols = tgt["dancers"][r]["columns"]
        f = L["f_of_k"]
        act = np.array(cols["action"], dtype=object)[f]
        icls = np.array([CLS[c] for c in cols["image_class"]])[f]
        ic = np.array(cols["confidence"], float)[f]
        src = np.array(cols["source"], dtype=object)[f]
        ty = np.array([np.nan if v is None else v for v in cols["target_yaw_world_deg"]], float)[f]
        cam_yaw = yaw_deg(L["C"][None] - L["pelvis"][r])
        pyaw = yaw_deg(torso_forward_np(J[:, o]))
        conf_ok = ((ic >= CONF_MIN) & (src == "image")) | (src == "owner")
        usable = conf_ok & ((icls != 1) | (src == "owner")) & (src != "agent_review") & (act != "none")
        # ---- the prior's own action, on the frames the FIT gets wrong (targets action rotate / flip180) or the owner labelled:
        # elsewhere the fit already agrees with the picture (or the agent judged the image call a false alarm, e.g. a face
        # turning before the torso), and front calls on profile faces are over-confident (jl_facing calibration)
        fit_wrong = usable & np.isin(act, ["rotate", "flip180"])
        pk = np.array(["keep"] * T, dtype=object)
        pd = np.zeros(T)
        for k in np.flatnonzero((fit_wrong | (src == "owner")) & (conf[:, o] > 0) & np.isfinite(pyaw)):
            pk[k], pd[k] = prior_action(int(icls[k]), float(cam_yaw[k]), float(pyaw[k]), from_owner=src[k] == "owner")
        applied = smooth_rotation_runs(pd, pk)
        changed = np.abs(applied) > 1e-9
        if changed.any():
            J[changed, o], go[changed, o] = rotate_about_vertical(J[changed, o], go[changed, o], np.radians(applied[changed]))
        pyaw2 = yaw_deg(torso_forward_np(J[:, o]))
        pcls2 = np.array([class_of(a, b) for a, b in zip(pyaw2, cam_yaw)])
        # ---- conf floor where the FIT is wrong and the (corrected) prior agrees with the picture
        agree = pcls2 == icls
        floor = fit_wrong & agree & (conf[:, o] > 0) & (conf[:, o] < CONF_FLOOR)
        conf_before = conf[:, o].copy()
        conf[floor, o] = CONF_FLOOR
        # ---- the hint
        hk = usable & np.isfinite(ty)
        hint_y[hk, o] = np.radians(ty[hk])
        tol = np.where(fit_wrong, HINT_TOL_FIX_DEG, HINT_TOL_KEEP_DEG)
        tol = np.where((src == "owner") & (icls == 1), HINT_TOL_SIDE_DEG, tol)
        hint_tol[hk, o] = np.radians(tol[hk])
        hint_c[hk, o] = np.where(src[hk] == "owner", 1.0, ic[hk])
        per_frame[f"{r}_prior_action"] = pk.astype(str)
        per_frame[f"{r}_prior_delta_raw_deg"] = pd
        per_frame[f"{r}_prior_delta_applied_deg"] = applied
        per_frame[f"{r}_prior_yaw_before"] = pyaw
        per_frame[f"{r}_prior_yaw_after"] = pyaw2
        per_frame[f"{r}_cam_yaw"] = cam_yaw
        per_frame[f"{r}_conf_before"] = conf_before
        per_frame[f"{r}_fit_wrong"] = fit_wrong

        def runs_t(m):
            idx = np.flatnonzero(m)
            out, s = [], None
            for i in idx:
                if s is None:
                    s = p = i
                elif i == p + 1:
                    p = i
                else:
                    out.append([round(L["t"][s], 2), round(L["t"][p], 2), int(p - s + 1)])
                    s = p = i
            if s is not None:
                out.append([round(L["t"][s], 2), round(L["t"][p], 2), int(p - s + 1)])
            return out
        rep["roles"][r] = {
            "name": NAME[r], "usable_evidence_frames": int(usable.sum()),
            "fit_wrong_frames": int(fit_wrong.sum()), "fit_wrong_runs_t": runs_t(fit_wrong),
            "prior_actions": {a: int((pk == a).sum()) for a in ("keep", "rotate", "flip180")},
            "prior_rotated_frames_incl_ramps": int(changed.sum()), "prior_rotated_runs_t": runs_t(changed),
            "prior_turn_deg": {"median_abs": round(float(np.median(np.abs(applied[changed]))), 1) if changed.any() else 0.0,
                               "max_abs": round(float(np.abs(applied).max()), 1)},
            "prior_class_agrees_with_image_before": round(float((np.array([class_of(a, b) for a, b in zip(pyaw, cam_yaw)])[usable]
                                                                 == icls[usable]).mean()), 4),
            "prior_class_agrees_with_image_after": round(float((pcls2[usable] == icls[usable]).mean()), 4),
            "conf_floor_frames": int(floor.sum()), "conf_floor_runs_t": runs_t(floor),
            "hint_frames": int(hk.sum()), "hint_frames_fix_band": int((hk & fit_wrong).sum()),
            "hint_frames_keep_band": int((hk & ~fit_wrong).sum())}
        log(f"{r}: {json.dumps(rep['roles'][r])[:900]}")
    rep["refit_chunks"] = windows_to_refit(tgt["windows"], json.loads(VERDICTS.read_text()).get("windows", []), L["owner"],
                                           lambda fr: int(L["k_of_f"][fr]))
    log(f"chunks to refit (own a corrected window): {rep['refit_chunks']}")
    if write:
        RF2.mkdir(parents=True, exist_ok=True)
        d = dict(z)
        d.update(joints=J.astype(np.float32), global_orient=go.astype(np.float32), conf=conf.astype(np.float32),
                 source=np.array(f"{PRIOR_FULL} corrected by dancecap.mono4d.jl_facing_refit prior ({time.strftime('%Y-%m-%d')})"))
        np.savez_compressed(PRIOR_FACE, **d)
        np.savez_compressed(HINT_FULL, target_yaw=hint_y.astype(np.float32), tol=hint_tol.astype(np.float32),
                            conf=hint_c.astype(np.float32), t=L["t"], k=np.arange(T), sigma_rad=np.float32(np.radians(30.0)),
                            source=np.array(str(TARGETS)))
        np.savez_compressed(RF2 / "corrections_frames.npz", t=L["t"], **per_frame)
        CORRECTIONS.write_text(json.dumps(rep, indent=1, default=str))
        log(f"-> {PRIOR_FACE}, {HINT_FULL}, {CORRECTIONS}")
    return rep


def _slice(src: Path, k0: int, k1: int, out: Path) -> Path:
    z = np.load(src, allow_pickle=True)
    k = np.asarray(z["k"], int)
    sel = (k >= k0) & (k <= k1)
    if sel.sum() != k1 - k0 + 1:
        raise SystemExit(f"k {k0}..{k1} not inside {src}")
    d = {n: (np.asarray(z[n])[sel] if np.ndim(z[n]) and len(z[n]) == len(k) else z[n]) for n in z.files}
    d["source"] = np.array(f"{src} k {k0}..{k1}")
    np.savez_compressed(out, **d)
    return out


# ================================================================================================ build + jobs
COPY_FILES = ("prepare.json", "ma3d_config.yaml", "ma3d_config_3db.yaml", "mono_init.npz", "mono_init.json", "stature.json", "qa_mask.json")


def build_chunk(cid: str) -> dict:
    import yaml
    k0, k1 = CHUNKS[cid]
    src, dst = old_chunk_dir(cid) / "mamma", chunk_dir(cid) / "mamma"
    prep = json.loads((src / "prepare.json").read_text())
    seq = prep["sets"][SET]["seq"]
    if dst.exists():
        for sub in ("fits", "logs"):
            if (dst / sub).exists():
                raise SystemExit(f"{dst / sub} exists: this chunk was already fitted; refusing to rebuild over it")
    dst.mkdir(parents=True, exist_ok=True)
    copied = {}
    for f in COPY_FILES:
        shutil.copy2(src / f, dst / f)
        copied[f] = sha1(dst / f)
    for rel in (f"ma_2d/{seq}/cam{CAM}.npz", f"ma_cap/{seq}/gt/cam{CAM}.npz", f"ma_cap/{seq}/gt/global.npz"):
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / rel, dst / rel)
        if sha1(dst / rel) != sha1(src / rel):
            raise SystemExit(f"copy of {rel} differs")
        copied[rel] = sha1(dst / rel)
    cfg = yaml.safe_load((src / "ma3d_config_3db.yaml").read_text())
    for run in PRIOR_RUNS:
        if cfg["optim"][run]["losses"].get("pose_prior_loss", {}).get("weight") != 0.5:
            raise SystemExit(f"{cid}: the jl_refit config has no pose_prior_loss 0.5 in {run}")
        cfg["optim"][run]["losses"]["facing_hint_loss"] = {"weight": float(HINT_W)}
    (dst / CFG_FACE).write_text(f"# dancecap mono4d.jl_facing_refit: ma3d_config_3db.yaml (jl_refit: MAMMA_MONO=1 + pose_prior_loss 0.5) + "
                                f"facing_hint_loss weight {HINT_W:g} in {', '.join(PRIOR_RUNS)} (generated, do not edit)\n"
                                + yaml.safe_dump(cfg, sort_keys=False), newline="\n")
    _slice(PRIOR_FACE, k0, k1, dst / "prior_3db_face.npz")
    _slice(HINT_FULL, k0, k1, dst / "facing_hint.npz")
    p2 = copy.deepcopy(prep)
    p2["sets"][SET]["config"] = CFG_FACE
    p2["sets"][SET]["pose_prior"] = {"weight": 0.5, "npz": "prior_3db_face.npz", "source": str(PRIOR_FACE)}
    p2["sets"][SET]["facing_hint"] = {"weight": HINT_W, "npz": "facing_hint.npz", "source": str(HINT_FULL)}
    p2["facing_refit"] = {"by": "dancecap.mono4d.jl_facing_refit", "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                          "from": str(src / "prepare.json"), "note": "jl_refit's chunk inputs unchanged; corrected prior + facing hint"}
    (dst / "prepare.json").write_text(json.dumps(p2, indent=1))
    tg = np.asarray(prep["grid"]["t"], float)
    pz, hz = np.load(dst / "prior_3db_face.npz"), np.load(dst / "facing_hint.npz")
    if np.abs(np.asarray(pz["t"], float) - tg).max() > 1e-9 or np.abs(np.asarray(hz["t"], float) - tg).max() > 1e-9:
        raise SystemExit(f"{cid}: prior / hint slice does not match the grid")
    hk = np.isfinite(hz["target_yaw"]) & (hz["conf"] > 0)
    rep = {"chunk": cid, "k": [k0, k1], "T": int(len(tg)), "seq": seq, "copied_sha1_equal_to_jl_refit": True, "copied": copied,
           "config": CFG_FACE, "hint_frames": [int(hk[:, b].sum()) for b in range(2)],
           "prior_frames_with_conf": [int((pz["conf"][:, b] > 0).sum()) for b in range(2)]}
    (chunk_dir(cid) / "build.json").write_text(json.dumps(rep, indent=1))
    log(f"build {cid}: {json.dumps({k: rep[k] for k in ('T', 'hint_frames', 'prior_frames_with_conf')})}")
    return rep


def finish_fit(cid: str, note: str) -> dict:
    """A pod fit step that was cut off AFTER ma_3d finished and mamma_dyn.export_smplx wrote fits/c1/smplx/ (the log ends with
    ma_3d's '[mem] ma_3d peak' line): the rest of mamma_dyn.fit, run locally - camera residuals + fit.json, the export summary
    rebuilt from the exported npz (the same fields export_smplx returns)."""
    from .. import mamma_dyn as md
    root = chunk_dir(cid) / "mamma"
    out = root / "fits" / SET
    logp = root / "logs" / f"ma_3d_{SET}.log"
    txt = logp.read_text(errors="replace")
    if "[mem] ma_3d peak" not in txt or "RUN:  fourth_run" not in txt or "Traceback" in txt:
        raise SystemExit(f"{cid}: ma_3d did not finish normally ({logp}); refit on a pod")
    for r in ROLES:
        if not (out / "smplx" / f"{r}.npz").exists():
            raise SystemExit(f"{cid}: {out / 'smplx' / r}.npz missing")
    if (out / "fit.json").exists():
        raise SystemExit(f"{out / 'fit.json'} exists")
    prep = json.loads((root / "prepare.json").read_text())
    s = prep["sets"][SET]
    summary = {}
    for r in ROLES:
        z = np.load(out / "smplx" / f"{r}.npz", allow_pickle=True)
        rp = np.asarray(z["reproj_parts_px"], float)
        parts = [str(p) for p in np.asarray(z["parts"])]
        summary[r] = {"frames": int(len(z["t"])), "median_reproj_px": float(np.nanmedian(z["reproj_px"])),
                      "median_reproj_parts_px": {p: float(np.nanmedian(rp[:, i])) if np.isfinite(rp[:, i]).any() else None
                                                 for i, p in enumerate(parts)},
                      "quality_median": float(np.median(z["quality"])),
                      "repaired_frames": np.flatnonzero(np.asarray(z["repaired"], bool)).tolist(),
                      "suspect_not_repaired": np.flatnonzero(np.asarray(z["suspect"], bool)).tolist(),
                      "betas0": float(np.asarray(z["betas"]).reshape(-1)[0])}
    ev = prep["eval"]
    resid = md.camera_residuals(out / "smplx", root / "ma_cap" / ev["seq"] / "gt", root / "ma_2d" / ev["seq"], ev["cams"], frame0=0)
    for cam in resid:
        resid[cam]["in_fit"] = cam in s["cams"]
    rep = {"tag": SET, "run_tag": SET, "cams": s["cams"], "frames": None, "partial": False, "world_frame": prep["world_frame"],
           "ma_3d": {"returncode": 0, "host": "pod", "finished_normally": True, "note": note},
           "smplx": summary, "residuals": resid, "eval_seq": ev["seq"], "host": "pod (ma_3d + export) / local (residuals, fit.json)",
           "config": s.get("config", prep["config"]), "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (out / "fit.json").write_text(json.dumps(rep, indent=1, default=str))
    log(f"finish-fit {cid}: {json.dumps({r: round(summary[r]['median_reproj_px'], 2) for r in ROLES})}")
    return rep


def refit_chunks() -> list[str]:
    rep = json.loads(CORRECTIONS.read_text())
    return [c for c in CHUNKS if c in rep["refit_chunks"]]


def job(cid: str) -> Path:
    from ..farm import recap_profile as RP
    k0, k1 = CHUNKS[cid]
    T = k1 - k0 + 1
    tk = take_name(cid)
    w = _ws(chunk_dir(cid))
    pw = f"{POD_WS}/{w}"
    prep = json.loads((chunk_dir(cid) / "mamma" / "prepare.json").read_text())
    seq = prep["sets"][SET]["seq"]
    pin = PACKAGE_ROOT / "work" / "john_lula_jlshape" / "mamma_pin_betas.json"
    fit_min = float(round(0.04 * T + 10))
    mm = fit_min + 25
    refine = RP.refine_args_recipe()
    d = {"schema": "dancecap.pod_job/1", "name": f"jl_face_{cid}", "take": TAKE, "max_minutes": mm, "needs": ["mamma"],
         "inputs": [f"{w}/mamma/prepare.json", f"{w}/mamma/ma3d_config.yaml", f"{w}/mamma/{CFG_FACE}",
                    f"{w}/mamma/mono_init.npz", f"{w}/mamma/stature.json", f"{w}/mamma/prior_3db_face.npz",
                    f"{w}/mamma/facing_hint.npz", f"{w}/mamma/ma_2d/{seq}/", f"{w}/mamma/ma_cap/{seq}/", _ws(pin)],
         "env": {"DANCECAP_POD": "1", "PYOPENGL_PLATFORM": "egl", "MAMMA_MONO": "1",
                 "MAMMA_MONO_INIT": f"{pw}/mamma/mono_init.npz",
                 "MAMMA_FREEZE_BETAS": "1", "MAMMA_INIT_BETAS": f"{POD_WS}/{_ws(pin)}",
                 "MAMMA_POSE_PRIOR": f"{pw}/mamma/prior_3db_face.npz",
                 "MAMMA_FACING_HINT": f"{pw}/mamma/facing_hint.npz"},
         "cwd": "dancecap",
         "exclude": ["dancecap/work/runpod/pywheels",
                     "mamma/data/body_models/smplx_locked_head/smplx/SMPLX_MALE.npz",
                     "mamma/data/body_models/smplx_locked_head/smplx/SMPLX_FEMALE.npz"],
         "setup": ["for g in MALE FEMALE; do ln -sf SMPLX_NEUTRAL.npz "
                   f"{POD_WS}/mamma/data/body_models/smplx_locked_head/smplx/SMPLX_$g.npz; done; "
                   "nvidia-smi --query-gpu=name,memory.total --format=csv; "
                   f"grep -c facing_hint_loss {POD_WS}/mamma-turing/optimization/utils/optimization.py"],
         "commands": [{"name": "fit", "cmd": f"python -m dancecap mamma fit {tk} --tag {SET}", "timeout_min": fit_min},
                      {"name": "select", "cmd": f"python -m dancecap mamma select {tk} --no-qa", "timeout_min": 5},
                      {"name": "refine", "cmd": f"python -m dancecap.pose_refine {tk} --src smplx --dst smplx_rf2_noslip {refine}",
                       "timeout_min": 12, "optional": True}],
         "outputs": [f"{w}/mamma/fits/", f"{w}/mamma/logs/", f"{w}/mamma/select.json", f"{w}/mamma/report.json",
                     f"{w}/smplx/", f"{w}/smplx_rf2_noslip/"],
         "verify": [f"{w}/mamma/fits/{SET}/fit.json", f"{w}/smplx/lead.npz", f"{w}/smplx/follow.npz"],
         "after_download": ["python -m dancecap.mono4d.jl_facing_refit stitch (after every refit chunk is home)"],
         "notes": [f"john_lula facing refit: chunk {cid} (k {k0}..{k1}, {T} grid frames) = jl_refit's recipe (MAMMA_MONO, J3b "
                   "inputs, betas pinned, SAM 3D Body prior w 0.5 as the start) with the prior CORRECTED by the image facing "
                   f"evidence and the new facing_hint_loss (MAMMA_FACING_HINT) weight {HINT_W:g}; ONE fit on this pod",
                   "then mamma select (fits/c1 -> smplx/) and pose_refine smplx -> smplx_rf2_noslip (J3b chunk recipe)",
                   "likeness-derived inputs only (landmarks, cameras, joints, headings); no frames leave the PC"]}
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    p = JOB_DIR / f"pod_job_{cid}.json"
    p.write_text(json.dumps(d, indent=1), newline="\n")
    log(f"pod job -> {p} (max {mm} min, fit timeout {fit_min} min)")
    return p


# ================================================================================================ stitch + patch
def stitch_list() -> Path:
    base = json.loads((WORK / "farm" / "stitch_list.json").read_text())
    d = copy.deepcopy(base)
    redo = set(refit_chunks())
    for c in d["chunks"]:
        if list(c["k"]) != list(CHUNKS[c["name"]]):
            raise SystemExit(f"{c['name']}: k {c['k']} != {CHUNKS[c['name']]}")
        cd = _work_rel(chunk_dir(c["name"]) if c["name"] in redo else old_chunk_dir(c["name"]))
        c.update(src=f"{cd}/smplx", refined=f"{cd}/smplx_rf2_noslip", prepare=f"{cd}/mamma/prepare.json",
                 ma_2d=f"{cd}/mamma/ma_2d/{TAKE}_{c['name']}_c1")
    d["note"] = (f"jl_facing_refit: the second stitch's list with the facing refit chunks {sorted(redo)}; the others are jl_refit's "
                 "(realism/refit/chunks, unchanged)")
    p = RF2 / "stitch_list.json"
    p.write_text(json.dumps(d, indent=1))
    return p


def do_stitch() -> dict:
    guard = PROTECTED + [OLD_REFIT]
    before = tree_digest(guard)
    for cid in refit_chunks():
        for sub in ("smplx", "smplx_rf2_noslip"):
            for r in ROLES:
                if not (chunk_dir(cid) / sub / f"{r}.npz").exists():
                    raise SystemExit(f"{cid}: {sub}/{r}.npz missing (pod job not home?)")
    p = stitch_list()
    cmd = [sys.executable, "-m", "dancecap.stitch", TAKE, "--chunks", str(p), "--out", _work_rel(STITCH2), *STITCH_ARGS]
    log(" ".join(cmd))
    STITCH2.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(cmd, cwd=PACKAGE_ROOT, capture_output=True, text=True)
    (RF2 / "stitch_stdout.log").write_text(r.stdout + "\n--- stderr\n" + r.stderr)
    print("\n".join(r.stdout.splitlines()[-14:]))
    if r.returncode not in (0, 2):
        raise SystemExit(f"stitch rc {r.returncode}: {r.stderr[-2000:]}")
    rr = subprocess.run([sys.executable, "-m", "dancecap.jl_repair", str(STITCH2 / "smplx"), str(STITCH2 / "smplx_rf2_noslip")],
                        cwd=PACKAGE_ROOT, capture_output=True, text=True)
    (RF2 / "repair_stdout.log").write_text(rr.stdout + "\n--- stderr\n" + rr.stderr)
    print(rr.stdout[-1500:])
    if rr.returncode != 0:
        raise SystemExit(f"jl_repair rc {rr.returncode}: {rr.stderr[-2000:]}")
    fr = {sub: facing_repair(STITCH2 / sub, FREP / sub) for sub in ("smplx", "smplx_rf2_noslip")}
    OUT_V2.mkdir(parents=True, exist_ok=True)
    for f in ("lead.npz", "follow.npz", "facing_repair.json"):
        shutil.copy2(FREP / "smplx" / f, OUT_V2 / f)
    if (STITCH2 / "smplx" / "depth_spike_repair.json").exists():
        shutil.copy2(STITCH2 / "smplx" / "depth_spike_repair.json", OUT_V2 / "depth_spike_repair.json")
    st = json.loads((STITCH2 / "stitch.json").read_text())
    after = tree_digest(guard)
    if after != before:
        changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
        raise SystemExit(f"PROTECTED / jl_refit FILES CHANGED: {changed[:10]}")
    prov = {"what": "john_lula one-camera MAMMA fit + SAM 3D Body prior CORRECTED by the image facing evidence + the facing hint "
                    "(realism v2), raw stitch after jl_repair and the facing-aware repair of the export's interpolated spans "
                    "(facing_repair.json); same contract as smplx/; NOT promoted (the owner promotes)",
            "facing_repair": {k: {r: v["roles"][r]["runs_repaired"] for r in ROLES} for k, v in fr.items()},
            "made_by": "python -m dancecap.mono4d.jl_facing_refit stitch", "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "refit_chunks": {c: str(chunk_dir(c)) for c in refit_chunks()},
            "kept_chunks_from_jl_refit": {c: str(old_chunk_dir(c)) for c in CHUNKS if c not in refit_chunks()},
            "stitch": str(STITCH2 / "stitch.json"), "refined_stitch": str(STITCH2 / "smplx_rf2_noslip"),
            "gate": st.get("gate"), "stitch_args": STITCH_ARGS, "prior": str(PRIOR_FACE), "hint": str(HINT_FULL),
            "corrections": str(CORRECTIONS), "protected_and_jl_refit_unchanged": True,
            "sha1": {f: sha1(OUT_V2 / f) for f in ("lead.npz", "follow.npz")}}
    (OUT_V2 / "provenance.json").write_text(json.dumps(prov, indent=1, default=str))
    log(f"smplx_3db_v2 -> {OUT_V2}; gate {st.get('gate', {}).get('passed')}")
    return prov


def runs_of(m: np.ndarray) -> list[tuple[int, int]]:
    idx = np.flatnonzero(np.asarray(m, bool))
    out, s = [], None
    for i in idx:
        if s is None:
            s = p = i
        elif i == p + 1:
            p = i
        else:
            out.append((int(s), int(p)))
            s = p = i
    if s is not None:
        out.append((int(s), int(p)))
    return out


def heading_path(fit_yaw: np.ndarray, prior_yaw: np.ndarray, a: int, b: int) -> np.ndarray:
    """New torso headings (deg, unwrapped) for frames a..b of an interpolated span: the (corrected) prior's own heading
    path, unwrapped over a-1..b+1, plus an offset blended linearly from (fit - prior) at a-1 to (fit - prior) at b+1, so the
    path meets the fit exactly at both good neighbours. Pure, tested."""
    k = np.arange(a - 1, b + 2)
    pu = np.degrees(np.unwrap(np.radians(np.asarray(prior_yaw, float)[k])))
    o0 = float(wrap_deg(fit_yaw[a - 1] - pu[0]))
    o1 = float(wrap_deg(fit_yaw[b + 1] - pu[-1]))
    w = (k - (a - 1)) / float(b + 1 - (a - 1))
    return (pu + (1 - w) * o0 + w * o1)[1:-1]


def facing_repair(src: Path, dst: Path) -> dict:
    """mamma_dyn.export_smplx re-fills frames without 2D evidence (undefined reprojection: the camera-not-ok span 27.27-28.37 s
    has its landmarks zeroed) by a per-joint SLERP between the good neighbours: the shortest turn, whatever the optimizer, the
    prior or the picture said. In a re-filled run (the npz 'repaired' flag) whose heading violates a confident facing-hint
    band, the root is turned about the vertical through the pelvis (global_orient composed with R_y; body pose, translation
    unchanged; joints by FK) onto heading_path(): the corrected SAM 3D Body prior's own heading path, meeting the fit at both
    ends. Kept only when it leaves fewer hint violations. Other runs and frames are copied unchanged."""
    from scipy.spatial.transform import Rotation as Rot
    from .. import physics_body as pb
    H = np.load(HINT_FULL)
    P = np.load(PRIOR_FACE)
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    rep = {"what": "facing-aware repair of the export's interpolated (repaired) spans", "src": str(src), "dst": str(dst), "roles": {}}
    for o, r in enumerate(ROLES):
        z = dict(np.load(Path(src) / f"{r}.npz", allow_pickle=True))
        T = len(z["t"])
        J = np.asarray(z["joints"], float)
        fy = yaw_deg(torso_forward_np(J[:, :22]))
        py = yaw_deg(torso_forward_np(np.asarray(P["joints"], float)[:, o]))
        hy, ht, hc = np.degrees(H["target_yaw"][:, o]), np.degrees(H["tol"][:, o]), H["conf"][:, o]
        hk = np.isfinite(hy) & (hc > 0)
        flag = np.zeros(T, bool)
        runs_rep = []
        go = np.asarray(z["global_orient"], float).copy()
        for a, b in runs_of(np.asarray(z.get("repaired", np.zeros(T)), bool)):
            if a < 1 or b > T - 2:
                continue
            ks = np.arange(a, b + 1)
            h = ks[hk[ks]]
            if not len(h):
                continue
            exc0 = np.abs(wrap_deg(fy[h] - hy[h])) - ht[h]
            if not (exc0 > 0).any():
                continue
            new = heading_path(fy, py, a, b)
            exc1 = np.abs(wrap_deg(new[h - a] - hy[h])) - ht[h]
            if (exc1 > 0).sum() >= (exc0 > 0).sum():
                runs_rep.append({"t": [round(float(z["t"][a]), 3), round(float(z["t"][b]), 3)], "applied": False,
                                 "violations_before": int((exc0 > 0).sum()), "violations_after": int((exc1 > 0).sum())})
                continue
            d = wrap_deg(new - fy[ks])
            go[ks] = (Rot.from_matrix(rot_y(np.radians(d))) * Rot.from_rotvec(go[ks])).as_rotvec()
            flag[ks] = True
            full = np.concatenate([[fy[a - 1]], new, [new[-1] + float(wrap_deg(fy[b + 1] - new[-1]))]])
            runs_rep.append({"t": [round(float(z["t"][a]), 3), round(float(z["t"][b]), 3)], "frames": int(b - a + 1), "applied": True,
                             "hint_frames": int(len(h)), "violations_before": int((exc0 > 0).sum()),
                             "violations_after": int((exc1 > 0).sum()), "max_excess_deg_before": round(float(exc0.max()), 1),
                             "max_excess_deg_after": round(float(exc1.max()), 1), "turn_deg_max_abs": round(float(np.abs(d).max()), 1),
                             "heading_step_deg_max": round(float(np.abs(np.diff(full)).max()), 1),
                             "old_heading_step_deg_max": round(float(np.degrees(np.abs(np.diff(np.unwrap(np.radians(fy[a - 1:b + 2])))).max())), 1)})
        if flag.any():
            z["global_orient"] = go.astype(np.asarray(z["global_orient"]).dtype)
            body = pb.load_body(z)
            Rw, pw = pb.forward_kinematics(body, pb.full_pose(z), np.asarray(z["transl"], float))
            Pj = np.einsum("tjab,jb->tja", Rw, body.j_rest) + pw
            n = min(J.shape[1], Pj.shape[1])
            J2 = J.copy()
            J2[flag, :n] = Pj[flag, :n]
            z["joints"] = J2.astype(np.asarray(z["joints"]).dtype)
        z["facing_repaired"] = flag
        np.savez_compressed(dst / f"{r}.npz", **z)
        rep["roles"][r] = {"runs_repaired": runs_rep, "frames_changed": int(flag.sum())}
        log(f"facing repair {Path(src).name} {r}: {json.dumps(runs_rep)}")
    (dst / "facing_repair.json").write_text(json.dumps(rep, indent=1))
    return rep


def _pose_keys(z: dict) -> list[str]:
    return [k for k in ("global_orient", "body_pose", "jaw_pose", "leye_pose", "reye_pose", "left_hand_pose", "right_hand_pose")
            if k in z]


def blend_fit(z_new: dict, z_cur: dict, w: np.ndarray) -> dict:
    """Per-frame blend of two smplx-contract dancers (same grid / betas): rotations slerped per joint (weight w of z_cur),
    transl lerped, joints re-made by FK (physics_body) of the blended parameters."""
    from scipy.spatial.transform import Rotation as Rot, Slerp  # noqa: F401
    from .. import physics_body as pb
    out = {k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v) for k, v in z_new.items()}
    idx = np.flatnonzero(w > 0)
    for k in _pose_keys(z_new):
        a = np.asarray(z_new[k], float).reshape(len(z_new["t"]), -1, 3)
        b = np.asarray(z_cur[k], float).reshape(len(z_cur["t"]), -1, 3)
        o = a.copy()
        for i in idx:
            qa, qb = Rot.from_rotvec(a[i]).as_quat(), Rot.from_rotvec(b[i]).as_quat()
            dot = (qa * qb).sum(-1, keepdims=True)
            qb = np.where(dot < 0, -qb, qb)
            dot = np.abs(dot).clip(max=1.0)
            th = np.arccos(dot)
            s = np.sin(th)
            wa = np.where(s > 1e-6, np.sin((1 - w[i]) * th) / np.maximum(s, 1e-12), 1 - w[i])
            wb = np.where(s > 1e-6, np.sin(w[i] * th) / np.maximum(s, 1e-12), w[i])
            q = wa * qa + wb * qb
            o[i] = Rot.from_quat(q / np.linalg.norm(q, axis=-1, keepdims=True)).as_rotvec()
        out[k] = o.reshape(np.asarray(z_new[k]).shape).astype(np.asarray(z_new[k]).dtype)
    tr = np.asarray(z_new["transl"], float).copy()
    tr[idx] = (1 - w[idx, None]) * tr[idx] + w[idx, None] * np.asarray(z_cur["transl"], float)[idx]
    out["transl"] = tr.astype(np.asarray(z_new["transl"]).dtype)
    body = pb.load_body(out)
    pose = pb.full_pose(out)
    Rw, pw = pb.forward_kinematics(body, pose, np.asarray(out["transl"], float).reshape(len(pose), 3))
    P = np.einsum("tjab,jb->tja", Rw, body.j_rest) + pw          # joint positions = R j_rest + p
    J = np.asarray(z_new["joints"], float).copy()
    n = min(J.shape[1], P.shape[1])
    J[idx, :n] = P[idx, :n]
    out["joints"] = J.astype(np.asarray(z_new["joints"]).dtype)
    out["audit_patch_weight"] = w.astype(np.float32)
    out["audit_patched"] = w > 0
    return out


def lurch_check(fit_dir: Path, role: str = "lead", t0: float = 25.9, t1: float = 26.75) -> dict:
    """The jl_refit audit's lurch (John 26.20-26.43 s back along the camera ray while hidden): pelvis offset along the camera
    ray vs the J3b fit smplx/, and the largest pelvis step, in the window."""
    L = _lattice()
    z = np.load(Path(fit_dir) / f"{role}.npz", allow_pickle=True)
    c = np.load(WORK / "smplx" / f"{role}.npz", allow_pickle=True)
    t = np.asarray(z["t"], float)
    m = (t >= t0) & (t <= t1)
    P, Q = np.asarray(z["joints"], float)[:, 0], np.asarray(c["joints"], float)[:, 0]
    ray = Q - L["C"][None]
    ray[:, 1] = 0
    ray /= np.linalg.norm(ray, axis=1, keepdims=True)
    along = ((P - Q) * ray).sum(1)
    step = 1e3 * np.linalg.norm(np.diff(P, axis=0), axis=1)
    return {"fit": str(fit_dir), "window_t": [t0, t1], "along_ray_vs_smplx_m": {"max": round(float(along[m].max()), 3),
            "min": round(float(along[m].min()), 3), "median_outside": round(float(np.median(along[(t > 24) & (t < 29) & ~m])), 3)},
            "max_pelvis_step_mm": round(float(step[m[:-1]].max()), 1)}


def audit_patch(force: bool | None = None) -> dict:
    """Re-apply the jl_refit audit patch (realism/refit/audit_patch/patch.json: John k 786-793 from smplx/, 5-frame cosine
    ramps) to the new raw and refined stitches when the lurch is still there (or force=True)."""
    pj = json.loads((OLD_REFIT / "audit_patch" / "patch.json").read_text())
    role = pj["role"]
    T = len(np.load(OUT_V2 / f"{role}.npz", allow_pickle=True)["t"])
    w = np.zeros(T)
    for k, v in pj["weights_current_fit"].items():
        w[int(k)] = float(v)
    chk = {"raw": lurch_check(OUT_V2, role), "refined": lurch_check(FREP / "smplx_rf2_noslip", role),
           "old_raw_unpatched": lurch_check(OLD_RAW, role), "old_best_patched": lurch_check(OLD_BEST, role)}
    def lurch(c):   # calibrated on jl_refit's case: old raw +0.35 m along the ray / 121 mm step, the J3b fit 0 / 64 mm
        return (c["along_ray_vs_smplx_m"]["max"] - c["along_ray_vs_smplx_m"]["median_outside"] > 0.25) or c["max_pelvis_step_mm"] > 80
    need = force if force is not None else (lurch(chk["raw"]) or lurch(chk["refined"]))
    rep = {"what": "jl_refit's audit patch re-applied to the facing refit", "patch_json": str(OLD_REFIT / "audit_patch" / "patch.json"),
           "check_before": chk, "applied": bool(need), "outputs": {}}
    PATCH2.mkdir(parents=True, exist_ok=True)
    for name, src, cur in (("smplx_3db_v2_patched", OUT_V2, WORK / "smplx"),
                           ("smplx_rf2_noslip_patched", FREP / "smplx_rf2_noslip", WORK / "smplx_rf2_noslip")):
        dst = PATCH2 / name
        dst.mkdir(parents=True, exist_ok=True)
        for r in ROLES:
            zn = dict(np.load(src / f"{r}.npz", allow_pickle=True))
            if r == role and need:
                zc = dict(np.load(cur / f"{r}.npz", allow_pickle=True))
                zn = blend_fit(zn, zc, w)
            np.savez_compressed(dst / f"{r}.npz", **zn)
        rep["outputs"][name] = {"from_new": str(src), "from_current": str(cur) if need else None,
                                "sha1": {r: sha1(dst / f"{r}.npz") for r in ROLES}}
    if need:
        rep["check_after"] = {"raw": lurch_check(PATCH2 / "smplx_3db_v2_patched", role),
                              "refined": lurch_check(PATCH2 / "smplx_rf2_noslip_patched", role)}
    (PATCH2 / "patch.json").write_text(json.dumps(rep, indent=1, default=str))
    log(f"audit patch applied={need}: {json.dumps(chk['raw'])}")
    return rep


def new_best() -> Path:
    return PATCH2 / "smplx_rf2_noslip_patched"


# ================================================================================================ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.mono4d.jl_facing_refit")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prior").add_argument("--dry", action="store_true")
    b = sub.add_parser("build")
    b.add_argument("--chunk", choices=list(CHUNKS))
    j = sub.add_parser("job")
    j.add_argument("--chunk", choices=list(CHUNKS))
    sub.add_parser("stitch")
    pa = sub.add_parser("patch")
    pa.add_argument("--force", choices=("yes", "no"))
    sub.add_parser("compare")
    sub.add_parser("sheets")
    sub.add_parser("footlock")
    a = ap.parse_args(argv)
    if a.cmd == "prior":
        print(json.dumps(prior(write=not a.dry), indent=1, default=str)[:6000])
    elif a.cmd == "build":
        for cid in ([a.chunk] if a.chunk else refit_chunks()):
            build_chunk(cid)
    elif a.cmd == "job":
        for cid in ([a.chunk] if a.chunk else refit_chunks()):
            print(job(cid))
    elif a.cmd == "stitch":
        print(json.dumps(do_stitch(), indent=1, default=str)[:3000])
        print(json.dumps(audit_patch(), indent=1, default=str)[:3000])
    elif a.cmd == "patch":
        print(json.dumps(audit_patch(None if a.force is None else a.force == "yes"), indent=1, default=str)[:3000])
    elif a.cmd == "compare":
        from .jl_facing_refit_compare import compare_all
        compare_all()
    elif a.cmd == "sheets":
        from .jl_facing_refit_compare import sheets
        sheets()
    elif a.cmd == "footlock":
        from .jl_facing_refit_compare import footlock_final
        footlock_final()
    return 0


if __name__ == "__main__":
    sys.exit(main())
