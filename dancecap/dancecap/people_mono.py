"""Generic 2D people stage of ONE handheld phone video of a couple: who the two dancers are in every frame, who leads, masks, whole-body keypoints.

The take-specific original (jl_people, john_lula) identified the couple by competition bibs and knew the phone's zoom steps. This stage needs neither:
other people may stand, sit or dance elsewhere in the picture, and the outputs are the same files, so every downstream stage runs unchanged.

    persons    the pod's yolo11x-pose detections (jl_people_detect, two inference sizes) merged into persons per frame (jl_people.merge_passes); persons
               shorter than 8 % of the image height or with fewer than 6 joints >= 0.3 never become dancers.
    tracklets  greedy frame-to-frame links (box IoU, or hip-centre distance relative to the height, gaps up to 0.3 s). They give each person-frame a
               time context: POSE DYNAMICS (median change of the height-normalised joint layout over 0.3 s, smoothed over 1 s: dancers move their limbs
               all the time, seated / standing people do not; the measure ignores the camera's pan and zoom) and the camera motion between frames
               (median similarity of the linked persons; it warps the tracker's continuity term).
    couple     pass 1 = ONE Viterbi over the ordered pair (jl_people.track_pair) with the SAME dancer-likeness score in both slots: size relative to a
               per-frame reference height (the 2nd tallest moving person, rolling median over +-2 s: the zoom changes it), pose dynamics, centrality,
               not cut by the bottom border; the couple-distance pair cost (boxes touch, similar heights). The posterior is symmetric in the slots,
               so 2 x the slot marginal is the probability that the chosen person is one of the couple.
    slots      the confident pass-1 frames are split into two appearance clusters (constrained 2-means on the Lab region colours: in every frame the two
               dancers go to different clusters; initialised by the principal direction of the per-frame colour difference, so slot swaps of pass 1
               do not matter). Their colour models (jl_people.colour_models) are the slot scores of 2-3 self-training passes: the two dancers stay apart
               through turns and crossings.
    roles      ONE global decision which slot leads: people/roles_override.json ({"swap": true} or {"lead_slot": 0|1}) > the take TOML's descriptions
               ([dancers.lead/follow] description: colour words on garment words -> expected region colours, compared in Lab with each slot's median
               measured colours) > stature (on frames where both stand upright, apart, feet on the same image row: the taller in the image is the
               taller dancer, only when the statures differ by > 2 %) > default (stature / pass-1 order), each with its confidence (identity.json).
    anchors    as jl_people.stage_anchors (strict + relaxed anchor_quality, pick_anchors + fill_gaps, anchor_doc: "john" = lead, "lula" = follow) with
               the per-frame reference height in place of the zoom table and the gaps in seconds.
    SAM / pose jl_people_sam (SAM 2.1 forward + backward from every anchor), fused masks, jl_people_pose (RTMW whole body); a refinement round replaces
               the relaxed anchors inside stretches where the two SAM runs disagree by skeleton anchors.

    python -m dancecap.people_mono run --take TAKE [--video PATH]      detect, anchors, sam, masks, pose, refine, finalize (each idempotent: --force)
    python -m dancecap.people_mono detect|anchors|sam|masks|pose|refine|finalize|review --take TAKE [--video PATH] [--work-root DIR] [--rounds 1]

Outputs in work/<take>/people/: in/pose.npz, identity.json, anchors0/1.json, in/sam0/1.npz, fused0/1.npz, in/pose_wb0/1.npz, teachers_01.npz,
masks_01.npz, pose_tables/01.npz, teachers.json, coverage.json, verification.json, in/ocr.json (empty reads), people_mono.json (seconds per command);
contact sheets in work/<take>/review/people/final_XX.jpg. Slot 0 = lead, 1 = follow everywhere (the keys "john" / "lula" are the contract's names).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import jl_people as J
from .take import PACKAGE_ROOT

HERE = Path(__file__).resolve().parent
JOINT_CONF = J.JOINT_CONF
MIN_JOINTS = J.MIN_JOINTS
MIN_H_FRAC = 0.08          # x image height: shorter persons are never dancers (phone screens, the far crowd)
LINK_GAP_S = 0.3
LINK_IOU = 0.3
LINK_HIP = 0.25            # hip-centre distance / height that still links two person-frames
DYN_LAG_S = 0.3
DYN_WIN_S = 1.0
DYN0, DYN_S = 0.03, 0.03   # pose dynamics (body heights per 0.3 s) at which the dancer score is neutral, and its scale
REF_WIN_S = 2.0
REF_MERGE_RATIO = 0.6       # the 2nd tallest moving person under this x the tallest: the couple is one detection (an embrace), or only one
                            # dancer is seen; the tallest is the dancer height then (the 2nd tallest was a spectator: lk03_cut20, 17 s)
REF_GUARD = 0.75            # the couple's own reference never falls below this x the scene reference (a wrong pair of small spectators
                            # made itself the reference and was kept: lk03_cut20, 17-20 s)
MAX_CAND = 7
BASE = 1.0                 # dancer score of an average moving, centred person of the reference height (a missing slot scores -MISS_COST)
W_SIZE = 4.0
W_DYN = 2.0
W_CEN = 2.0
CEN_FREE = 0.15            # horizontal offset from the image centre (x width) that costs nothing
W_TRUNC = 0.5
RATIO_FREE = math.log(1.55)  # partners' image heights beyond 1.55 : 1 cost W_RATIO_OUT x (log excess)^2 on top of jl_people's mild ratio term
W_RATIO_OUT = 10.0
W_SHARED = 1.0             # per joint beyond 2 that two detections share (within 5 % of the height): one body detected twice, not a couple
COL_SCALE = 0.6            # x jl_people.colour_llr in the slot scores (the colour models attract look-alikes: a static person in black)
MEMBER_MIN = 0.9
L_WEIGHT = 0.6             # Lab lightness weight in colour distances (exposure differs between video and description)
DIST_CAP = 70.0
DESC_SCALE = 10.0          # description margin (weighted Lab distance) per e-fold of the odds
CMDS = ("detect", "anchors", "sam", "masks", "pose", "refine", "finalize", "review", "run")


def log(msg: str) -> None:
    print(f"[people_mono {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------------------------------------ context


@dataclass
class Ctx:
    take: str
    work: Path
    video: Path | None = None
    dancers: dict = field(default_factory=dict)

    @property
    def people(self) -> Path:
        return self.work / "people"

    @property
    def inn(self) -> Path:
        return self.people / "in"

    @property
    def review_dir(self) -> Path:
        return self.work / "review" / "people"

    def need_video(self) -> Path:
        if self.video is None or not Path(self.video).exists():
            raise SystemExit(f"[people_mono] no video for take {self.take}: pass --video PATH (got {self.video})")
        return Path(self.video)


def context(take: str, video: str | Path | None = None, work_root: str | Path | None = None) -> Ctx:
    """The take's work folder, its video (default: the first source of Take.load) and the [dancers] table of takes/<take>.toml (may be empty)."""
    import tomllib
    work = (Path(work_root) if work_root else PACKAGE_ROOT / "work") / take
    cfg: dict = {}
    vid = Path(video) if video else None
    try:
        from .take import Take
        t = Take.load(take)
        cfg = t.cfg
        if vid is None and t.sources:
            vid = t.sources[0].path
    except Exception:  # noqa: BLE001  (a take without a TOML / clock: the video must be given)
        p = PACKAGE_ROOT / "takes" / f"{take}.toml"
        if p.exists():
            cfg = tomllib.loads(p.read_text(encoding="utf-8"))
    return Ctx(take, work, vid, cfg.get("dancers", {}) or {})


def done(path: Path, force: bool) -> bool:
    if path.exists() and not force:
        log(f"{path.name} exists, skipped (--force to redo)")
        return True
    return False


def record(ctx: Ctx, cmd: str, wall: float, cpu: float | None = None) -> None:
    """people/people_mono.json: wall (and CPU) seconds per subcommand, merged on every call."""
    p = ctx.people / "people_mono.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    d = json.loads(p.read_text()) if p.exists() else {}
    d.setdefault("wall_s", {})[cmd] = round(wall, 1)
    if cpu is not None:
        d.setdefault("cpu_s", {})[cmd] = round(cpu, 1)
    d["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    p.write_text(json.dumps(d, indent=1))


def run_script(args: list, env_extra: dict | None = None, cwd: Path | None = None) -> None:
    env = dict(os.environ)
    env.update(env_extra or {})
    cmd = [sys.executable, "-u", *[str(a) for a in args]]
    log("$ " + " ".join(cmd))
    subprocess.run(cmd, check=True, env=env, cwd=str(cwd) if cwd else None)


def scratch(*parts: str) -> Path:
    p = Path(tempfile.gettempdir()).joinpath("people_mono", *parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ------------------------------------------------------------------------------------------------ persons, tracklets, features (pure)


def load_persons(pose_npz: Path) -> dict:
    """The pod's detections merged into persons per frame (jl_people.load without the bib reads)."""
    z = np.load(pose_npz)
    F = int(z["n_frames"])
    m = J.merge_passes(z["box"], z["kp"], z["conf"], z["frame"], F)
    idx = m["idx"]
    P = {"frame": z["frame"][idx], "box": z["box"][idx], "kp": z["kp"][idx], "conf": z["conf"][idx], "col": z["col"][idx], "fstart": m["fstart"],
         "fps": float(z["fps"]), "size": np.asarray(z["size"]), "n_frames": F}
    P["height"] = P["box"][:, 3] - P["box"][:, 1]
    log(f"{len(idx)} persons over {F} frames ({len(idx) / max(F, 1):.1f} per frame)")
    return P


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), axis=2)
    area = lambda x: np.clip(x[:, 2] - x[:, 0], 0, None) * np.clip(x[:, 3] - x[:, 1], 0, None)  # noqa: E731
    return inter / np.maximum(area(a)[:, None] + area(b)[None, :] - inter, 1e-9)


def hip_centres(kp: np.ndarray, box: np.ndarray) -> np.ndarray:
    """(N, 2) jl_people.hip_centre for all persons at once."""
    ok = (kp[:, [11, 12], 2] >= JOINT_CONF).all(1)
    fb = np.stack([(box[:, 0] + box[:, 2]) / 2, box[:, 1] + 0.55 * (box[:, 3] - box[:, 1])], 1)
    return np.where(ok[:, None], kp[:, [11, 12], :2].mean(1), fb)


def valid_persons(P: dict) -> np.ndarray:
    nj = (P["kp"][:, :, 2] >= JOINT_CONF).sum(1)
    return (P["height"] >= MIN_H_FRAC * float(P["size"][1])) & (nj >= MIN_JOINTS)


def link_tracklets(P: dict, valid: np.ndarray, max_gap: int) -> np.ndarray:
    """(N,) tracklet id per person (-1 = not valid): greedy best-first links to the tracks seen within max_gap frames, by box IoU or by a
    hip-centre distance under LINK_HIP heights with heights within 25 %."""
    hip = P.get("hip", hip_centres(P["kp"], P["box"]))
    tid = np.full(len(P["frame"]), -1, int)
    last_p: list[int] = []
    last_f: list[int] = []
    for f in range(P["n_frames"]):
        ids = np.arange(P["fstart"][f], P["fstart"][f + 1])
        ids = ids[valid[ids]]
        keep = [k for k in range(len(last_p)) if f - last_f[k] <= max_gap]
        tr = np.array([last_p[k] for k in keep], int)
        trk = np.array(keep, int)
        used = np.zeros(len(ids), bool)
        if len(tr) and len(ids):
            iou = iou_matrix(P["box"][tr], P["box"][ids])
            hh = np.maximum(P["height"][tr][:, None], P["height"][ids][None])
            hd = np.linalg.norm(hip[tr][:, None] - hip[ids][None], axis=-1) / np.maximum(hh, 1.0)
            hr = P["height"][tr][:, None] / np.maximum(P["height"][ids][None], 1.0)
            ok = (iou > LINK_IOU) | ((hd < LINK_HIP) & (hr > 0.8) & (hr < 1.25))
            s = np.where(ok, iou + (LINK_HIP - hd), -np.inf)
            while np.isfinite(s).any():
                i, j = np.unravel_index(np.argmax(s), s.shape)
                t = int(trk[i])
                tid[ids[j]] = t
                last_p[t], last_f[t] = int(ids[j]), f
                used[j] = True
                s[i, :] = -np.inf
                s[:, j] = -np.inf
        for j in np.flatnonzero(~used):
            tid[ids[j]] = len(last_p)
            last_p.append(int(ids[j]))
            last_f.append(f)
    return tid


def track_groups(tid: np.ndarray) -> list[np.ndarray]:
    """Person indices of every tracklet, in frame order (persons are stored frame by frame)."""
    v = np.flatnonzero(tid >= 0)
    order = v[np.argsort(tid[v], kind="stable")]
    cuts = np.flatnonzero(np.diff(tid[order])) + 1
    return np.split(order, cuts) if len(order) else []


def pose_dynamics(P: dict, tid: np.ndarray, lag: int, win: int) -> np.ndarray:
    """(N,) median change of the height-normalised joint layout (joints relative to the hip centre / height) between a person-frame and the same
    tracklet `lag` frames earlier (later at a tracklet's start), rolling median over +-win person-frames of the tracklet; NaN when unknown.
    Independent of the camera's pan and zoom."""
    from scipy.ndimage import median_filter
    hip = P.get("hip", hip_centres(P["kp"], P["box"]))
    h = np.maximum(P["height"], 1.0)
    kn = (P["kp"][:, :, :2] - hip[:, None]) / h[:, None, None]
    good = P["kp"][:, :, 2] >= JOINT_CONF
    out = np.full(len(tid), np.nan)
    for g in track_groups(tid):
        if len(g) < 2:
            continue
        fr = P["frame"][g]
        j = np.searchsorted(fr, fr - lag)
        back_ok = (fr - fr[np.minimum(j, len(g) - 1)] >= lag // 2) & (j < np.arange(len(g)))
        jf = np.minimum(np.searchsorted(fr, fr + lag), len(g) - 1)
        q = np.where(back_ok, j, jf)
        a, b = g, g[q]
        both = good[a] & good[b] & (a != b)[:, None]
        d = np.linalg.norm(kn[a] - kn[b], axis=-1)
        d = np.where(both, d, np.nan)
        n = both.sum(1)
        with np.errstate(all="ignore"):
            v = np.where(n >= 4, np.nanmedian(np.where(n[:, None] >= 4, d, 0.0), axis=1), np.nan)
        fin = np.isfinite(v)
        if fin.sum() >= 1:
            v = np.where(fin, v, np.nanmedian(v[fin]))
            out[g] = median_filter(v, size=2 * win + 1, mode="nearest")
    return out


def rolling_nanmedian(x: np.ndarray, half: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        seg = x[max(0, i - half):i + half + 1]
        seg = seg[np.isfinite(seg)]
        if len(seg):
            out[i] = np.median(seg)
    return out


def fill_nan(x: np.ndarray, default: float) -> np.ndarray:
    ok = np.isfinite(x)
    if not ok.any():
        return np.full(len(x), default)
    return np.interp(np.arange(len(x)), np.flatnonzero(ok), x[ok])


def reference_height(P: dict, mask: np.ndarray, fps: float) -> np.ndarray:
    """(F,) dancer height in pixels per frame: the 2nd tallest person of `mask` (the moving ones; a tall passer-by in front does not set
    it), or the tallest when the 2nd is under REF_MERGE_RATIO of it (the couple in one box, or one dancer alone), rolling median over +-2 s."""
    F = P["n_frames"]
    per = np.full(F, np.nan)
    for f in range(F):
        ids = np.arange(P["fstart"][f], P["fstart"][f + 1])
        h = np.sort(P["height"][ids[mask[ids]]])[::-1]
        if len(h):
            per[f] = h[1] if len(h) >= 2 and h[1] >= REF_MERGE_RATIO * h[0] else h[0]
    allh = P["height"][mask] if mask.any() else P["height"]
    return fill_nan(rolling_nanmedian(per, int(round(REF_WIN_S * fps))), float(np.median(allh)) if len(allh) else 100.0)


def couple_reference(P: dict, ids: np.ndarray, fps: float, fallback: np.ndarray) -> np.ndarray:
    """(F,) mean height of the chosen couple (frames with both on separate detections), rolling median over +-2 s; the fallback where unknown."""
    F = len(ids)
    per = np.full(F, np.nan)
    ok = (ids >= 0).all(1) & (ids[:, 0] != ids[:, 1])
    per[ok] = P["height"][ids[ok]].mean(1)
    if ok.sum() < 10:
        return fallback
    r = rolling_nanmedian(per, int(round(REF_WIN_S * fps)))
    return np.maximum(np.where(np.isfinite(r), r, fallback), REF_GUARD * fallback)


def dancer_score(P: dict, dyn: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """(N,) dancer-likeness (log-odds-like, higher = more likely one of the couple): size against the frame's reference height (smaller costs
    more than larger), pose dynamics, horizontal centrality, cut by the bottom border."""
    W, H = float(P["size"][0]), float(P["size"][1])
    lr = np.log(np.maximum(P["height"], 1.0) / np.maximum(ref[P["frame"]], 1.0))
    size = W_SIZE * np.where(lr < 0, lr ** 2, 0.4 * lr ** 2)
    dv = W_DYN * np.clip((np.nan_to_num(dyn, nan=DYN0) - DYN0) / DYN_S, -1.0, 1.0)
    cx = (P["box"][:, 0] + P["box"][:, 2]) / 2 / W
    cen = W_CEN * np.maximum(np.abs(cx - 0.5) - CEN_FREE, 0.0)
    trunc = W_TRUNC * (P["box"][:, 3] >= 0.99 * H)
    return BASE - size + dv - cen - trunc


def people_warp(P: dict, tid: np.ndarray) -> np.ndarray:
    """(F, 4) scale, rot (0), tx, ty of the image motion from frame f-1 to f: median over the persons linked in both frames (heights ratio, hip
    centres), identity where fewer than 2 are linked. Stands in for a measured camera track."""
    F = P["n_frames"]
    w = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (F, 1))
    hip = P.get("hip", hip_centres(P["kp"], P["box"]))
    centre = np.asarray(P["size"], float) / 2
    prev = None
    for f in range(F):
        ids = np.arange(P["fstart"][f], P["fstart"][f + 1])
        ids = ids[tid[ids] >= 0]
        if prev is not None and len(ids) and len(prev):
            _, ia, ib = np.intersect1d(tid[prev], tid[ids], return_indices=True)
            if len(ia) >= 2:
                a, b = prev[ia], ids[ib]
                s = float(np.clip(np.median(P["height"][b] / np.maximum(P["height"][a], 1.0)), 0.8, 1.25))
                t = np.median(hip[b] - (s * (hip[a] - centre) + centre), axis=0)
                w[f] = [s, 0.0, t[0], t[1]]
        prev = ids
    return w


def shared_joints(kp: np.ndarray, h: np.ndarray, tol: float = 0.05) -> np.ndarray:
    """(m, m) number of confident joints two detections have at the same place (within tol x the larger height)."""
    ok = kp[:, :, 2] >= JOINT_CONF
    d = np.linalg.norm(kp[:, None, :, :2] - kp[None, :, :, :2], axis=-1) / np.maximum(np.maximum(h[:, None], h[None]), 1.0)[..., None]
    return (ok[:, None] & ok[None] & (d < tol)).sum(-1)


def couple_cost(bx: np.ndarray, h: np.ndarray, kp: np.ndarray | None = None) -> np.ndarray:
    """(m, m) pair cost of jl_people.build_frames: gap between the boxes / height beyond COUPLE_GAP, unequal heights, minus a reward for overlap;
    plus a duplicate cost when the two detections share joints (kp given)."""
    gx = np.maximum(0.0, np.maximum(bx[:, None, 0], bx[None, :, 0]) - np.minimum(bx[:, None, 2], bx[None, :, 2]))
    gy = np.maximum(0.0, np.maximum(bx[:, None, 1], bx[None, :, 1]) - np.minimum(bx[:, None, 3], bx[None, :, 3]))
    gap = np.hypot(gx, gy) / np.maximum(np.maximum(h[:, None], h[None]), 40.0)
    lr = np.abs(np.log(np.maximum(h[:, None], 1.0) / np.maximum(h[None], 1.0)))
    pc = J.W_COUPLE * np.maximum(gap - J.COUPLE_GAP, 0.0) ** 2 + J.W_RATIO * lr ** 2 + W_RATIO_OUT * np.maximum(lr - RATIO_FREE, 0.0) ** 2
    pc = np.minimum(pc, J.PAIR_CAP)
    ix = np.maximum(0.0, np.minimum(bx[:, None, 2], bx[None, :, 2]) - np.maximum(bx[:, None, 0], bx[None, :, 0]))
    iy = np.maximum(0.0, np.minimum(bx[:, None, 3], bx[None, :, 3]) - np.maximum(bx[:, None, 1], bx[None, :, 1]))
    area = (bx[:, 2] - bx[:, 0]) * (bx[:, 3] - bx[:, 1])
    cover = ix * iy / np.maximum(np.minimum(area[:, None], area[None]), 1.0)
    pc = pc - J.W_TOUCH * np.minimum(cover, 1.0)
    if kp is not None:
        pc = pc + W_SHARED * np.maximum(shared_joints(kp, h) - 2, 0)
    return pc


def build_frames(P: dict, cand: np.ndarray, sa: np.ndarray, sb: np.ndarray, ref: np.ndarray, warp: np.ndarray) -> list[dict]:
    """Per frame the tracker inputs of jl_people.track_pair: the MAX_CAND best candidates (by the better slot score), hip centres, heights,
    slot scores, couple pair cost, the previous frame's centres warped into this frame, and which candidates may hold both bodies."""
    hip = P.get("hip", hip_centres(P["kp"], P["box"]))
    centre = np.asarray(P["size"], float) / 2
    frames: list[dict] = []
    prev_c = None
    for f in range(P["n_frames"]):
        ids = np.arange(P["fstart"][f], P["fstart"][f + 1])
        ids = ids[cand[ids]]
        if len(ids) > MAX_CAND:
            ids = ids[np.argsort(-np.maximum(sa[ids], sb[ids]), kind="stable")[:MAX_CAND]]
        h = P["height"][ids].astype(float)
        c = hip[ids].astype(float)
        bx = P["box"][ids].astype(float)
        pc = couple_cost(bx, h, P["kp"][ids]) if len(ids) else np.zeros((0, 0))
        sc, rot, tx, ty = warp[f]
        if prev_c is not None and len(prev_c):
            R = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
            cw = (sc * (R @ (prev_c - centre).T)).T + centre + np.array([tx, ty])
        else:
            cw = np.zeros((0, 2))
        same_ok = (h >= 0.6 * ref[f]) & ((bx[:, 2] - bx[:, 0]) / np.maximum(h, 1.0) >= J.MERGE_ASPECT) if len(ids) else np.zeros(0, bool)
        frames.append({"ids": ids, "c": c, "h": h, "sj": sa[ids], "sl": sb[ids], "pc": pc, "cw": cw, "same_ok": same_ok})
        prev_c = c
    return frames


def chosen_ids(frames: list[dict], cand: np.ndarray) -> np.ndarray:
    ids = np.full((len(frames), 2), -1, int)
    for i, f in enumerate(frames):
        for r in range(2):
            if cand[i, r] >= 0:
                ids[i, r] = f["ids"][cand[i, r]]
    return ids


# ------------------------------------------------------------------------------------------------ colours (pure)


def to_lab(rgb) -> np.ndarray:
    """RGB 0-255 (..., 3) -> CIE Lab (L 0-100); NaN rows stay NaN."""
    import cv2
    x = np.asarray(rgb, np.float32)
    shp = x.shape
    flat = x.reshape(-1, 3)
    fin = np.isfinite(flat).all(1)
    out = np.full(flat.shape, np.nan, np.float32)
    if fin.any():
        out[fin] = cv2.cvtColor(np.clip(flat[fin], 0, 255).reshape(-1, 1, 3) / 255.0, cv2.COLOR_RGB2Lab).reshape(-1, 3)
    return out.reshape(shp)


def lab_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d * np.array([L_WEIGHT, 1.0, 1.0])
    return np.minimum(np.linalg.norm(d, axis=-1), DIST_CAP)


def split_by_colour(ca: np.ndarray, cb: np.ndarray, iters: int = 20) -> tuple[np.ndarray, dict]:
    """Constrained 2-means of paired person-frames: ca, cb (n, 4, 3) region RGB of the two dancers of n frames. Returns swap (n,) bool (True =
    the frame's b belongs to cluster A) and stats. Initialised by the principal direction of the per-frame Lab difference (sign-free), then
    alternating cluster medians / per-frame assignment, so it does not depend on which slot a person had."""
    w = np.array([J.REGION_W[r] for r in J.REGIONS])[:, None]
    la, lb = to_lab(ca), to_lab(cb)
    diff = np.nan_to_num((la - lb) * w).reshape(len(la), -1)
    if len(diff) < 4 or not np.abs(diff).sum():
        return np.zeros(len(ca), bool), {"frames": int(len(ca)), "separation": 0.0}
    _, _, vt = np.linalg.svd(diff, full_matrices=False)
    swap = diff @ vt[0] < 0

    def cost(x, m):
        d = lab_dist(x, m[None])                                     # (n, 4), NaN regions -> nan
        return np.nansum(np.where(np.isfinite(d), d * w[:, 0], 0.0), axis=1)

    for _ in range(iters):
        A = np.where(swap[:, None, None], lb, la)
        B = np.where(swap[:, None, None], la, lb)
        with np.errstate(all="ignore"):
            mA, mB = np.nanmedian(A, 0), np.nanmedian(B, 0)
        mA, mB = np.nan_to_num(mA), np.nan_to_num(mB)
        keep = cost(la, mA) + cost(lb, mB)
        flip = cost(la, mB) + cost(lb, mA)
        new = flip < keep
        if (new == swap).all():
            break
        swap = new
    margin = np.abs(flip - keep)
    spread = float(np.median(np.minimum(keep, flip))) + 1e-6
    return swap, {"frames": int(len(ca)), "margin_median": round(float(np.median(margin)), 2), "separation": round(float(np.median(margin)) / spread, 3)}


# ------------------------------------------------------------------------------------------------ the generic identity


def solve_identity(P: dict, iters: int = 2) -> dict:
    """Who the couple is (symmetric pass) -> two appearance slots (colour split) -> self-trained colour passes. Returns ids (F, 2) person index
    per slot (-1 none), p (F, 2) slot marginals, frames (tracker inputs, for anchor_quality), ref (F,) couple height, models, and diagnostics."""
    F, fps = P["n_frames"], P["fps"]
    t0 = time.time()
    P["hip"] = hip_centres(P["kp"], P["box"])
    valid = valid_persons(P)
    tid = link_tracklets(P, valid, max(1, int(round(LINK_GAP_S * fps))))
    dyn = pose_dynamics(P, tid, max(1, int(round(DYN_LAG_S * fps))), max(1, int(round(DYN_WIN_S * fps / 2))))
    warp = people_warp(P, tid)
    ref = ref_scene = reference_height(P, valid & (np.nan_to_num(dyn) >= DYN0), fps)
    log(f"{valid.sum()} valid persons, {tid.max() + 1} tracklets, reference height {np.median(ref):.0f} px ({time.time() - t0:.1f} s)")
    u = dancer_score(P, dyn, ref)
    frames = build_frames(P, valid, u, u, ref, warp)
    r = J.track_pair(frames)
    ids1 = chosen_ids(frames, r["cand"])
    member = np.minimum(1.0, 2.0 * r["p"])                             # symmetric posterior: P(one of the couple) = 2 x the slot marginal
    log(f"pass 1 (who): present {(ids1 >= 0).mean(0).round(3)}, member >= {MEMBER_MIN} {(member >= MEMBER_MIN).mean(0).round(3)}")
    ref = couple_reference(P, ids1, fps, ref_scene)
    u = dancer_score(P, dyn, ref)
    seeds, split = seed_slots(P, ids1, member, ref)
    allmask = valid & (P["height"] >= 0.5 * ref[P["frame"]])
    if allmask.sum() < 8:
        allmask = valid if valid.sum() >= 8 else np.ones(len(valid), bool)
    res: dict = {}
    for it in range(iters + 1):
        models = J.colour_models(P["col"], seeds, allmask)
        sa = COL_SCALE * J.colour_llr(P["col"], models, "lead") + u
        sb = COL_SCALE * J.colour_llr(P["col"], models, "follow") + u
        frames = build_frames(P, valid, sa, sb, ref, warp)
        r = J.track_pair(frames)
        ids = chosen_ids(frames, r["cand"])
        res = {"ids": ids, "p": r["p"], "frames": frames, "models": models}
        log(f"pass {it + 2} (slots): present {(ids >= 0).mean(0).round(3)}, p >= 0.9 {(r['p'] >= 0.9).mean(0).round(3)}")
        if it == iters:
            break
        both = (ids >= 0).all(1) & (ids[:, 0] != ids[:, 1]) & (r["p"] >= 0.9).all(1)
        if both.sum() >= 16:
            seeds = {"lead": ids[both, 0], "follow": ids[both, 1]}
            ref = couple_reference(P, ids, fps, ref_scene)
            u = dancer_score(P, dyn, ref)
    res.update({"ref": ref, "ids1": ids1, "member1": member, "split": split, "valid": valid, "tid": tid, "dyn": dyn, "warp": warp,
                "seconds": round(time.time() - t0, 1)})
    return res


def seed_slots(P: dict, ids: np.ndarray, member: np.ndarray, ref: np.ndarray) -> tuple[dict, dict]:
    """Colour seeds of the two slots from the confident frames of the symmetric pass (both present on separate, not overlapping detections of
    plausible size): split_by_colour decides which of the two persons of each frame belongs to slot A; a weak split keeps the pass-1 slots."""
    F = len(ids)
    ok = (ids >= 0).all(1) & (ids[:, 0] != ids[:, 1]) & (member >= MEMBER_MIN).all(1)
    fr = np.flatnonzero(ok)
    if len(fr):
        bx = P["box"][ids[fr]]
        iou = np.array([J.box_iou(a, b) for a, b in bx])
        hmin = P["height"][ids[fr]].min(1)
        fr = fr[(iou < 0.5) & (hmin >= 0.6 * ref[fr])]
    a, b = ids[fr, 0], ids[fr, 1]
    swap, info = split_by_colour(P["col"][a], P["col"][b])
    if info["separation"] < 0.15:
        swap = np.zeros(len(fr), bool)
        info["used"] = "pass-1 slots (weak colour split)"
    else:
        info["used"] = "colour split"
    info["swapped_share"] = round(float(swap.mean()), 3) if len(swap) else 0.0
    A = np.where(swap, b, a)
    B = np.where(swap, a, b)
    info["centre_a_rgb"] = np.nanmedian(P["col"][A], 0).round(1).tolist() if len(A) else None
    info["centre_b_rgb"] = np.nanmedian(P["col"][B], 0).round(1).tolist() if len(B) else None
    log(f"slot seeds: {len(fr)} of {F} frames, {info['used']}, separation {info['separation']}")
    return {"lead": A, "follow": B}, info


# ------------------------------------------------------------------------------------------------ descriptions -> expected colours (pure)

COLOUR_RGB = {
    "black": (32, 32, 35), "white": (220, 220, 218), "grey": (128, 128, 128), "gray": (128, 128, 128), "charcoal": (62, 62, 66), "silver": (180, 180, 185),
    "red": (175, 40, 40), "auburn": (125, 55, 38), "ginger": (190, 95, 45), "copper": (165, 85, 50), "blonde": (205, 175, 115), "blond": (205, 175, 115),
    "brown": (105, 70, 48), "beige": (205, 185, 150), "tan": (185, 145, 105), "khaki": (170, 160, 112), "cream": (232, 222, 195), "ivory": (236, 232, 215),
    "blue": (45, 80, 170), "navy": (32, 42, 82), "denim": (78, 102, 140), "turquoise": (60, 190, 190), "teal": (30, 120, 125), "green": (50, 130, 60),
    "olive": (110, 110, 50), "pink": (232, 150, 172), "coral": (240, 125, 100), "salmon": (240, 145, 120), "rose": (210, 120, 140), "yellow": (232, 205, 60),
    "gold": (205, 165, 60), "orange": (232, 130, 40), "purple": (110, 60, 140), "violet": (140, 90, 190), "lilac": (190, 160, 210), "burgundy": (110, 30, 45),
    "maroon": (110, 30, 40), "wine": (115, 30, 50),
}
SKIN_RGB = (190, 145, 120)
LIGHTER = ("light", "pale", "pastel", "faded", "bleached")
DARKER = ("dark", "deep")
BARE = ("shaved", "bald", "buzzed")
_G = {
    "torso": ("t-shirt", "tshirt", "tee", "shirt", "top", "blouse", "jacket", "vest", "tank", "sweater", "jumper", "hoodie", "bodysuit", "cardigan", "coat",
              "polo", "jersey", "leotard", "bra", "camisole", "corset", "sweatshirt", "blazer"),
    "legs": ("jeans", "trousers", "pants", "leggings", "joggers", "slacks", "chinos", "sweatpants", "overalls"),
    "short": ("skirt", "shorts", "miniskirt"),
    "dress": ("dress", "gown"),
    "hair": ("hair", "ponytail", "bun", "braid", "braids", "plait", "curls", "locks", "dreadlocks", "afro"),
    "beard": ("beard", "moustache", "mustache", "stubble"),
    "hat": ("cap", "hat", "beanie"),
    "head": ("head", "scalp"),
    "shoes": ("shoes", "shoe", "sneakers", "sneaker", "trainers", "boots", "heels", "loafers", "sandals", "socks", "soles", "flats", "pumps", "slippers"),
}
GARMENT_KIND = {w: k for k, ws in _G.items() for w in ws}
KIND_REGIONS = {"torso": [("torso", 1.0)], "legs": [("thigh", 1.0), ("shin", 0.6)], "short": [("thigh", 1.0)], "dress": [("torso", 1.0), ("thigh", 0.8)],
                "hair": [("head", 0.5)], "beard": [("head", 0.3)], "hat": [("head", 0.4)], "head": [("head", 0.4)], "shoes": [("shin", 0.25)]}
DEFAULT_COLOUR = {"jeans": "denim"}
REGION_INDEX = {r: i for i, r in enumerate(J.REGIONS)}


def _modify(rgb, mod: str | None) -> np.ndarray:
    x = np.asarray(rgb, float)
    if mod == "light":
        return x + 0.45 * (255.0 - x)
    if mod == "dark":
        return 0.55 * x
    return x


def parse_description(text: str | None) -> dict:
    """Expected region colours of a dancer from a free description: {region: {"rgb", "lab", "w", "words"}} for torso / thigh / shin / head.
    Phrases are split at commas (and / ;), chunks at with / over / under / and; a colour word (with its light / dark modifier) belongs to the
    garments of its chunk, else to those of its phrase; a garment without a colour takes its default (jeans: denim), a shaved / bald head is skin."""
    acc: dict[str, list] = {}
    if not text:
        return {}
    for phrase in re.split(r"[,;/]", text.lower()):
        chunks = [re.findall(r"[a-z]+(?:-[a-z]+)*", c) for c in re.split(r"\b(?:with|over|under|and)\b", phrase)]
        phrase_g = list(dict.fromkeys(t for c in chunks for t in c if t in GARMENT_KIND))
        for toks in chunks:
            cols, garms, mod, dangling = [], [], None, None
            bare = any(t in BARE for t in toks)
            for t in toks:
                if t in LIGHTER or t in DARKER:
                    mod = "light" if t in LIGHTER else "dark"
                elif t in COLOUR_RGB:
                    cols.append((_modify(COLOUR_RGB[t], mod), f"{mod + ' ' if mod else ''}{t}"))
                    mod = None
                elif t in GARMENT_KIND:
                    garms.append(t)
                    if mod:
                        dangling, mod = mod, None
            targets = list(dict.fromkeys(garms)) or (phrase_g if cols else [])
            for g in targets:
                kind = GARMENT_KIND[g]
                if cols:
                    use = cols
                elif kind == "head" and bare:
                    use = [(np.asarray(SKIN_RGB, float), "skin")]
                elif g in DEFAULT_COLOUR:
                    use = [(_modify(COLOUR_RGB[DEFAULT_COLOUR[g]], dangling), f"{dangling + ' ' if dangling else ''}{DEFAULT_COLOUR[g]}")]
                elif dangling:
                    use = [(_modify(COLOUR_RGB["grey"], dangling), f"{dangling} (no colour)")]
                else:
                    continue
                for reg, w in KIND_REGIONS[kind]:
                    for rgb, word in use:
                        acc.setdefault(reg, []).append((rgb, w, f"{word} {g}"))
    out = {}
    for reg, items in acc.items():
        labs = to_lab(np.array([i[0] for i in items]))
        ws = np.array([i[1] for i in items])
        lab = (labs * ws[:, None]).sum(0) / ws.sum()
        out[reg] = {"rgb": np.array([i[0] for i in items]).mean(0).round(1).tolist(), "lab": lab.round(2).tolist(), "w": float(ws.max()),
                    "words": [i[2] for i in items]}
    return out


def colour_name(rgb) -> str:
    """Nearest named colour (COLOUR_RGB, Lab distance) of a measured RGB."""
    if rgb is None or not np.isfinite(np.asarray(rgb, float)).all():
        return "unknown"
    names = [n for n in COLOUR_RGB if n not in ("gray", "blond")]
    d = lab_dist(to_lab(np.array([COLOUR_RGB[n] for n in names])), to_lab(np.asarray(rgb, float))[None])
    return names[int(np.argmin(d))]


def description_evidence(slot_rgb: np.ndarray, exp_lead: dict, exp_follow: dict) -> dict | None:
    """Which slot fits the lead's description: slot_rgb (2, 4, 3) measured median RGB per slot and region (NaN = unknown). Total weighted Lab
    distance of (slot 0 = lead, slot 1 = follow) against the swapped assignment; margin -> confidence (logistic, DESC_SCALE)."""
    if not exp_lead and not exp_follow:
        return None
    lab = to_lab(slot_rgb)
    keep = swap = 0.0
    regions = []
    for role, exp in (("lead", exp_lead), ("follow", exp_follow)):
        for reg, e in exp.items():
            r = REGION_INDEX[reg]
            if not (np.isfinite(lab[0, r]).all() and np.isfinite(lab[1, r]).all()):
                continue
            d0, d1 = float(lab_dist(lab[0, r], e["lab"])), float(lab_dist(lab[1, r], e["lab"]))
            if role == "lead":
                keep, swap = keep + e["w"] * d0, swap + e["w"] * d1
            else:
                keep, swap = keep + e["w"] * d1, swap + e["w"] * d0
            regions.append({"role": role, "region": reg, "w": e["w"], "dist_slot0": round(d0, 1), "dist_slot1": round(d1, 1)})
    if not regions:
        return None
    margin = swap - keep
    return {"lead_slot": 0 if margin > 0 else 1, "confidence": round(1.0 / (1.0 + math.exp(-abs(margin) / DESC_SCALE)), 3), "margin": round(margin, 2),
            "cost_slot0_lead": round(keep, 2), "cost_slot1_lead": round(swap, 2), "regions": regions}


def body_span(k: np.ndarray) -> float:
    """Vertical nose (or 1.15 x shoulder) to ankle span of an upright skeleton, NaN when not upright / joints missing."""
    c = k[:, 2]
    if (c[[5, 6, 11, 12]] < 0.5).any() or (c[[15, 16]] < 0.5).all():
        return float("nan")
    ank = k[[15, 16], 1][c[[15, 16]] >= 0.5].mean()
    sh = k[[5, 6], :2].mean(0)
    hp = k[[11, 12], :2].mean(0)
    ax = np.array([k[[15, 16], 0][c[[15, 16]] >= 0.5].mean(), ank])
    if not (sh[1] < hp[1] < ank) or abs(ax[0] - sh[0]) > 0.3 * (ank - sh[1]):
        return float("nan")
    return float(ank - k[0, 1]) if c[0] >= 0.5 and k[0, 1] < sh[1] else 1.15 * float(ank - sh[1])


def stature_evidence(P: dict, ids: np.ndarray, p: np.ndarray, fps: float) -> dict:
    """Image-height ratio slot 0 / slot 1 on frames where both are confidently chosen, upright, apart (box IoU < 0.1) and stand on the same
    image row (ankles within 6 % of the height): per-second medians of the log ratio, their median and a z score over the seconds."""
    rows = []
    for f in np.flatnonzero((ids >= 0).all(1) & (ids[:, 0] != ids[:, 1]) & (p >= 0.9).all(1)):
        a, b = ids[f]
        if J.box_iou(P["box"][a], P["box"][b]) >= 0.1:
            continue
        sa, sb = body_span(P["kp"][a]), body_span(P["kp"][b])
        if not (np.isfinite(sa) and np.isfinite(sb)) or min(sa, sb) <= 0:
            continue
        ya = P["kp"][a][[15, 16], 1].max()
        yb = P["kp"][b][[15, 16], 1].max()
        if abs(ya - yb) > 0.06 * max(sa, sb):
            continue
        rows.append((int(f / fps), math.log(sa / sb)))
    if not rows:
        return {"seconds": 0, "frames": 0}
    r = np.array(rows)
    per = np.array([np.median(r[r[:, 0] == s, 1]) for s in np.unique(r[:, 0])])
    med = float(np.median(per))
    mad = float(np.median(np.abs(per - med))) * 1.4826 + 1e-3
    return {"seconds": int(len(per)), "frames": int(len(r)), "log_ratio_01": round(med, 4), "z": round(med / (mad / math.sqrt(len(per))), 2)}


def stature_decision(ev: dict | None, s_lead: float | None, s_follow: float | None) -> dict | None:
    """The taller dancer in the image is the one with the larger stature_m: a tie-breaker (confidence <= 0.85), only when the statures differ by
    more than 2 % and at least 3 seconds were measurable."""
    if not ev or ev.get("seconds", 0) < 3 or not s_lead or not s_follow or abs(s_lead / s_follow - 1.0) <= 0.02:
        return None
    lead_taller = s_lead > s_follow
    slot0_taller = ev["log_ratio_01"] > 0
    conf = 0.5 * (1.0 + math.erf(abs(ev["z"]) / math.sqrt(2.0)))
    return {"lead_slot": 0 if slot0_taller == lead_taller else 1, "confidence": round(min(0.85, conf), 3), "expected_ratio": round(s_lead / s_follow, 4),
            "measured_ratio_01": round(math.exp(ev["log_ratio_01"]), 4), **ev}


def decide_roles(desc: dict | None, stat: dict | None, override: dict | None = None) -> dict:
    """ONE global slot order: override > description (confidence >= 0.6) > stature (>= 0.6) > default (stature order, else slot 0 leads)."""
    if desc and desc["confidence"] >= 0.6:
        auto = {"lead_slot": desc["lead_slot"], "decided_by": "description", "confidence": desc["confidence"]}
    elif stat and stat["confidence"] >= 0.6:
        auto = {"lead_slot": stat["lead_slot"], "decided_by": "stature", "confidence": stat["confidence"]}
    else:
        ls = stat["lead_slot"] if stat else 0
        agree = [e["confidence"] for e in (desc, stat) if e and e["lead_slot"] == ls]
        auto = {"lead_slot": ls, "decided_by": "default", "confidence": round(max(agree + [0.5]), 3),
                "note": "no evidence reached 0.6: " + ("stature order kept" if stat else "pass-1 slot order kept (slot 0 leads)")}
    out = {**auto, "automatic": dict(auto), "evidence": {"description": desc, "stature": stat, "override": override}}
    if override:
        if "lead_slot" in override:
            out["lead_slot"] = int(override["lead_slot"])
        elif override.get("swap"):
            out["lead_slot"] = 1 - auto["lead_slot"]
        out.update({"decided_by": "override", "confidence": 1.0})
        out.pop("note", None)
    return out


def slot_colours(P: dict, ids: np.ndarray, p: np.ndarray) -> np.ndarray:
    """(2, 4, 3) median measured region RGB of each slot over its confident frames."""
    out = np.full((2, 4, 3), np.nan)
    for r in range(2):
        sel = ids[(ids[:, r] >= 0) & (p[:, r] >= 0.9), r]
        if len(sel) < 5:
            sel = ids[ids[:, r] >= 0, r]
        if len(sel):
            with np.errstate(all="ignore"):
                out[r] = np.nanmedian(P["col"][sel].astype(float), 0)
    return out


# ------------------------------------------------------------------------------------------------ anchors


def make_anchors(P: dict, res: dict, lead_slot: int) -> tuple[list[dict], dict]:
    """jl_people.stage_anchors(first=True) with the per-frame reference height (ref_h = 1, S = ref) and gaps in seconds; slot order -> lead first."""
    fps = P["fps"]
    ref = res["ref"]
    min_gap, max_gap = max(1, int(round(0.8 * fps))), max(2, int(round(3.0 * fps)))
    q = J.anchor_quality(P, res, 1.0, ref)
    qr = J.anchor_quality(P, res, 1.0, ref, p_min=0.6, joints_min=11, score_min=-1.0)
    tiers = {}
    if not (qr > 0).any():
        qr = J.anchor_quality(P, res, 1.0, ref, p_min=0.3, joints_min=8, score_min=-5.0)
        tiers["loose"] = True
    frames = J.fill_gaps(J.pick_anchors(q, min_gap), qr, max_gap=max_gap, min_gap=min_gap)
    ordered = dict(res)
    ordered["ids"] = res["ids"][:, [lead_slot, 1 - lead_slot]]
    doc = J.anchor_doc(P, ordered, frames)
    for d in doc:
        d["tier"] = "strict" if q[d["frame"]] > 0 else ("loose" if tiers else "relaxed")
    return doc, {"n": len(doc), "strict": int(sum(d["tier"] == "strict" for d in doc)), "relaxed": int(sum(d["tier"] != "strict" for d in doc)),
                 "strict_frames_available": int((q > 0).sum()), "min_gap": min_gap, "max_gap": max_gap}


def refine_anchors(anchors0: list[dict], fused0: Path, wb0: Path, n_frames: int, fps: float) -> list[dict] | None:
    """jl_people.stage_anchors(first=False) generalised: keep the strict anchors, replace the relaxed ones inside stretches where the two SAM runs
    disagree (IoU < 0.75, dilated by 0.67 s) by skeleton anchors (jl_people.anchors_from_skeletons). None when no frame disagrees."""
    from scipy.ndimage import binary_dilation
    Z = np.load(fused0)
    ag = np.nan_to_num(Z["agree"], nan=1.0)
    bad = ag.min(1) < 0.75
    if not bad.any():
        return None
    hard = binary_dilation(bad, iterations=max(1, int(round(0.67 * fps))))
    kp, _ = J.choose_variant(np.load(wb0)["kp"])
    keep = [a for a in anchors0 if not (hard[a["frame"]] and a.get("tier") != "strict")]
    cand = []
    for f in range(0, n_frames, max(1, int(round(fps / 3)))):
        if not hard[f]:
            continue
        k = kp[f]
        if min(int((k[r, :17, 2] >= 4.0).sum()) for r in range(2)) < 11:
            continue
        c = [np.nanmean(k[r, [5, 6, 11, 12], :2], 0) for r in range(2)]
        hh = max(np.nanmax(k[r, :17, 1]) - np.nanmin(k[r, :17, 1]) for r in range(2))
        if np.linalg.norm(c[0] - c[1]) < 0.10 * hh:
            continue
        cand.append(f)
    doc = sorted(keep + J.anchors_from_skeletons(kp, cand), key=lambda a: a["frame"])
    spacing = max(1, int(round(0.2 * fps)))
    final: list[dict] = []
    for a in doc:
        if final and a["frame"] - final[-1]["frame"] < spacing:
            continue
        final.append(a)
    log(f"round 1: {len(keep)} kept + {len(final) - len(keep)} skeleton anchors ({int(bad.sum())} disagreeing frames)")
    return final


# ------------------------------------------------------------------------------------------------ final tables


def per_frame_qa(fused: dict, kp: np.ndarray, size: tuple[int, int], fps: float) -> dict:
    """jl_people.per_frame_qa without a zoom table (scale 1) and with the visibility window in seconds (+-2 s)."""
    import cv2
    F = len(kp)
    W, H = size
    area = fused["area"].astype(float)
    half = max(1, int(round(2.0 * fps)))
    vis = np.ones((F, 2))
    for o in range(2):
        for f in range(F):
            seg = area[max(0, f - half):f + half + 1, o]
            ref = np.percentile(seg, 90) if len(seg) else 0.0
            vis[f, o] = area[f, o] / ref if ref > 0 else 0.0
    sip = np.zeros((F, 2))
    edge = np.zeros((F, 2), bool)
    biou = np.zeros(F)
    box = fused["box"]
    for f in range(F):
        ms = []
        for o in range(2):
            k = f"m_{f}_{o}"
            ms.append(cv2.imdecode(fused[k], cv2.IMREAD_GRAYSCALE) > 127 if k in fused else None)
            b = box[f, o]
            if np.isfinite(b).all():
                edge[f, o] = bool(b[0] <= 1 or b[1] <= 1 or b[2] >= W - 1 or b[3] >= H - 1)
        if np.isfinite(box[f]).all():
            biou[f] = J.box_iou(box[f, 0], box[f, 1])
        for o in range(2):
            own, par = ms[o], ms[1 - o]
            if own is None or par is None:
                continue
            n = k_in = 0
            for j in (5, 6, 11, 12):
                x, y, c = kp[f, o, j]
                if not np.isfinite(c) or c < 0.5 or not (0 <= int(x) < W and 0 <= int(y) < H):
                    continue
                n += 1
                k_in += int(par[int(y), int(x)] and not own[int(y), int(x)])
            sip[f, o] = k_in / n if n else 0.0
    return {"skel_in_partner": sip, "visible": vis, "edge": edge, "box_iou": biou}


def mask_colours(video: Path | None, fused: dict, kpw: np.ndarray, fps: float, exp: dict | None = None, step_s: float = 0.5) -> dict:
    """Colour summary of each role's masks (every step_s): median RGB of the mask pixels above the shoulders (head), shoulders -> hips (torso)
    and below the hips (legs), with the nearest colour name; with the parsed descriptions also the keep / swap distance (an independent check)."""
    import cv2
    if video is None or not Path(video).exists():
        return {"frames_checked": 0, "note": "no video"}
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return {"frames_checked": 0, "note": "video not readable"}
    step = max(1, int(round(step_s * fps)))
    acc = {o: {"head": [], "torso": [], "legs": []} for o in range(2)}
    n = 0
    for f in range(len(kpw)):
        if f % step:
            if not cap.grab():
                break
            continue
        ok, img = cap.read()
        if not ok:
            break
        hit = False
        for o in range(2):
            k = f"m_{f}_{o}"
            if k not in fused:
                continue
            m = cv2.imdecode(fused[k], cv2.IMREAD_GRAYSCALE) > 127
            if m.sum() < 400:
                continue
            kk = kpw[f, o]
            sh = kk[[5, 6], 1][np.nan_to_num(kk[[5, 6], 2]) >= 0.3]
            hp = kk[[11, 12], 1][np.nan_to_num(kk[[11, 12], 2]) >= 0.3]
            if not len(sh) or not len(hp):
                continue
            ys = np.arange(m.shape[0])[:, None]
            for name, band in (("head", ys < sh.mean()), ("torso", (ys >= sh.mean()) & (ys < hp.mean())), ("legs", ys >= hp.mean())):
                px = img[m & band]
                if len(px) >= 30:
                    acc[o][name].append(np.median(px[:, ::-1], 0))
            hit = True
        n += int(hit)
    cap.release()
    out: dict = {"frames_checked": n}
    meas = np.full((2, 4, 3), np.nan)
    for o, role in ((0, "lead"), (1, "follow")):
        d = {}
        for name, vals in acc[o].items():
            if vals:
                rgb = np.median(np.array(vals), 0)
                d[name] = {"rgb": rgb.round(1).tolist(), "name": colour_name(rgb), "brightness": round(float(rgb.mean()), 1), "frames": len(vals)}
                for reg in {"head": ("head",), "torso": ("torso",), "legs": ("thigh", "shin")}[name]:
                    meas[o, REGION_INDEX[reg]] = rgb
        out[role] = d
    if exp:
        ev = description_evidence(meas, exp.get("lead", {}), exp.get("follow", {}))
        if ev:
            out["description_check"] = {"lead_is_lead": ev["lead_slot"] == 0, "confidence": ev["confidence"], "margin": ev["margin"]}
    return out


def video_times(video: Path, cache: Path, F: int, fps: float) -> np.ndarray:
    from .cameras.newcam import video_pts
    pts = np.asarray(video_pts(video, cache), float)[:F]
    if len(pts) < F:
        last = pts[-1] if len(pts) else -1.0 / fps
        pts = np.r_[pts, last + np.arange(1, F - len(pts) + 1) / fps]
    return pts


def finalize(ctx: Ctx, pts: np.ndarray | None = None, pose_npz: Path | None = None) -> dict:
    """jl_people.finalize without bibs (bib_dist 9999, no bib boost), without a zoom table, with a colour summary of the role masks instead of the
    john_lula colour check: people/teachers_01.npz, masks_01.npz, pose_tables/01.npz, teachers.json, coverage.json, verification.json, in/ocr.json."""
    people, inn = ctx.people, ctx.inn
    P = load_persons(pose_npz or inn / "pose.npz")
    F, fps = P["n_frames"], P["fps"]
    size = (int(P["size"][0]), int(P["size"][1]))
    tag = "1" if (people / "fused1.npz").exists() and (inn / "pose_wb1.npz").exists() else "0"
    fused = dict(np.load(people / f"fused{tag}.npz"))
    with warnings.catch_warnings():                                    # all-NaN rows of absent dancers
        warnings.simplefilter("ignore", RuntimeWarning)
        kpw, pick = J.choose_variant(np.load(inn / f"pose_wb{tag}.npz")["kp"])
    kpw = kpw.copy()
    kpw[..., 2] = np.clip(kpw[..., 2] / J.RTMW_SCALE, 0.0, 1.0)
    kpw[~np.isfinite(fused["box"]).all(-1)] = np.nan
    qa = per_frame_qa(fused, kpw, size, fps)
    bd = np.full((F, 2), 9999, np.int32)
    rconf = J.role_confidence(fused["agree"], fused["dist"], qa, bd)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pconf = J.pose_confidence(kpw, rconf, qa)
    (people / "pose_tables").mkdir(parents=True, exist_ok=True)
    pt = J.pose_tables(P, people / "pose_tables" / "01.npz")
    row = J.match_rows(P, pt["slot"], fused["box"])
    if pts is None:
        pts = video_times(ctx.need_video(), ctx.work / "frames" / "video_pts.json", F, fps)
    pts = np.asarray(pts, float)[:F]
    merged = qa["box_iou"] > 0.5
    teachers = {"t": pts.astype(np.float64), "frame_index": np.arange(F, dtype=np.int32), "ok": np.ones(F, bool), "row": row, "box": fused["box"].astype(np.float32),
                "kp": kpw[:, :, :17].astype(np.float32), "conf": pconf.astype(np.float32), "merged": np.repeat(merged[:, None], 2, axis=1),
                "role_conf": rconf.astype(np.float32), "floor": np.full((F, 2, 2), np.nan, np.float32), "size": np.array(size, np.int32), "attached": np.array(False),
                "kp_wb": kpw.astype(np.float32), "wb_variant": pick.astype(np.int8), "mask_area": fused["area"].astype(np.int32),
                "mask_agree": fused["agree"].astype(np.float32), "mask_dist": fused["dist"].astype(np.int32), "visible": qa["visible"].astype(np.float32),
                "skel_in_partner": qa["skel_in_partner"].astype(np.float32), "edge": qa["edge"], "box_iou": qa["box_iou"].astype(np.float32), "bib_dist": bd}
    np.savez_compressed(people / "teachers_01.npz", **teachers)
    masks = {k: v for k, v in fused.items() if k.startswith("m_")}
    np.savez(people / "masks_01.npz", box=fused["box"], area=fused["area"], **masks)
    cov = J.coverage_table(pconf, rconf, qa, fps)
    (people / "coverage.json").write_text(json.dumps(cov, indent=1))
    ident = json.loads((people / "identity.json").read_text()) if (people / "identity.json").exists() else {}
    roles = ident.get("roles", {})
    exp = {r: parse_description((ctx.dancers.get(r) or {}).get("description")) for r in ("lead", "follow")}
    col = mask_colours(ctx.video, fused, kpw, fps, exp)
    ver = {"colour": col, "mask_agree_lt_0.6_frames": [int((np.nan_to_num(fused["agree"][:, o], nan=1.0) < 0.6).sum()) for o in range(2)],
           "identity": {k: roles.get(k) for k in ("lead_slot", "decided_by", "confidence")}, "bib": "none (generic identity, people_mono)"}
    (people / "verification.json").write_text(json.dumps(ver, indent=1))
    if not (inn / "ocr.json").exists():
        inn.mkdir(parents=True, exist_ok=True)
        (inn / "ocr.json").write_text(json.dumps({"reads": []}))
    summ = {"frames": F, "t": [round(float(pts[0]), 3), round(float(pts[-1]), 3)], "mask_share": [round(float((fused["area"][:, o] > 50).mean()), 3) for o in range(2)],
            "conf_ge_0.5": [round(float((pconf[:, o] >= 0.5).mean()), 3) for o in range(2)], "both_ge_0.5": round(float((pconf >= 0.5).all(1).mean()), 3),
            "role_conf_ge_0.9": [round(float((rconf[:, o] >= 0.9).mean()), 3) for o in range(2)],
            "role_conf_lt_0.5": [round(float((rconf[:, o] < 0.5).mean()), 3) for o in range(2)],
            "hidden_share": [round(float((qa["skel_in_partner"][:, o] > 0.5).mean()), 3) for o in range(2)], "edge_share": round(float(qa["edge"].any(1).mean()), 3),
            "merged_share": round(float(merged.mean()), 3), "matched_detection_share": [round(float((row[:, o] >= 0).mean()), 3) for o in range(2)],
            "variant_masked_share": round(float(pick.mean()), 3), "sam_round": tag,
            "identity": {k: roles.get(k) for k in ("lead_slot", "decided_by", "confidence")} | {"method": ident.get("method", "people_mono")}}
    (people / "teachers.json").write_text(json.dumps(summ, indent=1))
    log(json.dumps(summ))
    try:
        review(ctx)
    except (Exception, SystemExit) as e:  # noqa: BLE001  (review sheets are never fatal)
        log(f"review skipped: {e!r}")
    return summ


# ------------------------------------------------------------------------------------------------ review


def contact_sheets(tiles: list, out: Path, name: str, cols: int = 4, per_sheet: int = 8) -> list[Path]:
    import cv2
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for k0 in range(0, len(tiles), per_sheet):
        grp = tiles[k0:k0 + per_sheet]
        while len(grp) % cols:
            grp.append(np.zeros_like(grp[0]))
        sheet = np.vstack([np.hstack(grp[j:j + cols]) for j in range(0, len(grp), cols)])
        p = out / f"{name}_{k0 // per_sheet:02d}.jpg"
        cv2.imwrite(str(p), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
        paths.append(p)
    return paths


def review(ctx: Ctx, n: int = 16) -> list[Path]:
    """Contact sheets of n instants spread over the clip: each dancer's mask contour + skeleton (lead green, follow magenta), confidences."""
    import cv2
    video = ctx.need_video()
    z = np.load(ctx.people / "teachers_01.npz")
    Z = np.load(ctx.people / "masks_01.npz")
    kp, conf = z["kp_wb"], z["conf"]
    F = len(kp)
    names = [(ctx.dancers.get(r) or {}).get("name", r) for r in ("lead", "follow")]
    feet = [(15, 17), (15, 18), (15, 19), (16, 20), (16, 21), (16, 22)]
    cap = cv2.VideoCapture(str(video))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 540
    scale = min(1.0, 420.0 / W)
    tiles = []
    for f in np.linspace(0, F - 1, n).round().astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, img = cap.read()
        if not ok:
            continue
        for o, col in enumerate(((60, 220, 60), (220, 60, 220))):
            k = f"m_{f}_{o}"
            if k in Z.files:
                m = cv2.imdecode(Z[k], cv2.IMREAD_GRAYSCALE) > 127
                img[m] = (0.75 * img[m] + 0.25 * np.array(col)).astype(np.uint8)
                cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, cs, -1, col, 2)
            kk = kp[f, o]
            for a, b in J.SKEL + (feet if kk.shape[0] >= 133 else []):
                if np.nan_to_num(kk[a, 2]) >= 0.3 and np.nan_to_num(kk[b, 2]) >= 0.3:
                    cv2.line(img, (int(kk[a, 0]), int(kk[a, 1])), (int(kk[b, 0]), int(kk[b, 1])), (255, 255, 255), 3)
                    cv2.line(img, (int(kk[a, 0]), int(kk[a, 1])), (int(kk[b, 0]), int(kk[b, 1])), col, 1)
            if np.isfinite(z["box"][f, o]).all():
                b = z["box"][f, o]
                cv2.putText(img, f"{names[o]} {conf[f, o]:.2f}", (int(b[0]), max(16, int(b[1]) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        img = cv2.resize(img, None, fx=scale, fy=scale)
        cv2.putText(img, f"f={f} t={z['t'][f]:.2f}s", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        tiles.append(img)
    cap.release()
    paths = contact_sheets(tiles, ctx.review_dir, "final") if tiles else []
    for p in paths:
        log(f"review {p}")
    return paths


# ------------------------------------------------------------------------------------------------ subcommands


def imgsz_for(video: Path) -> list[int]:
    import cv2
    cap = cv2.VideoCapture(str(video))
    long_side = max(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    return [960, 1280] if long_side <= 1000 else [1280, 1920]


def cmd_detect(ctx: Ctx, weights: str = "yolo11x-pose.pt", imgsz: list[int] | None = None, force: bool = False) -> None:
    out = ctx.inn / "pose.npz"
    if done(out, force):
        return
    video = ctx.need_video().resolve()
    ctx.inn.mkdir(parents=True, exist_ok=True)
    sizes = imgsz or imgsz_for(video)
    wdir = scratch("weights")
    w = Path(weights)
    weights_arg = str(w.resolve()) if w.parent != Path(".") or w.exists() else weights   # a bare name: ultralytics downloads it into wdir (cwd)
    env = {"YOLO_CONFIG_DIR": os.environ.get("YOLO_CONFIG_DIR") or str(scratch("yolo_cfg"))}
    run_script([HERE / "jl_people_detect.py", "pose", video, out.resolve(), "--weights", weights_arg, "--imgsz", *sizes], env, cwd=wdir)


def cmd_anchors(ctx: Ctx, force: bool = False, pose_npz: Path | None = None) -> dict | None:
    out = ctx.people / "anchors0.json"
    if done(out, force) and (ctx.people / "identity.json").exists():
        return None
    ctx.people.mkdir(parents=True, exist_ok=True)
    P = load_persons(pose_npz or ctx.inn / "pose.npz")
    res = solve_identity(P)
    ident = identity_doc(ctx, P, res)
    doc, ainfo = make_anchors(P, res, ident["roles"]["lead_slot"])
    if not doc:
        raise SystemExit("[people_mono] no frame shows both dancers on separate detections: no SAM anchors")
    ident["anchors"] = ainfo
    out.write_text(json.dumps(doc))
    (ctx.people / "identity.json").write_text(json.dumps(ident, indent=1))
    log(f"round 0: {len(doc)} anchors ({ainfo['relaxed']} relaxed) -> {out}; lead = slot {ident['roles']['lead_slot']} by {ident['roles']['decided_by']} "
        f"({ident['roles']['confidence']})")
    return {"P": P, "res": res, "identity": ident, "anchors": doc}


def identity_doc(ctx: Ctx, P: dict, res: dict) -> dict:
    """identity.json: method, the chosen pair, the role decision with its evidence, per-second presence / confidence."""
    ids, p, fps = res["ids"], res["p"], P["fps"]
    lead, follow = ctx.dancers.get("lead") or {}, ctx.dancers.get("follow") or {}
    slot_rgb = slot_colours(P, ids, p)
    exp = {"lead": parse_description(lead.get("description")), "follow": parse_description(follow.get("description"))}
    desc = description_evidence(slot_rgb, exp["lead"], exp["follow"])
    stat = stature_decision(stature_evidence(P, ids, p, fps), lead.get("stature_m"), follow.get("stature_m"))
    ov_path = ctx.people / "roles_override.json"
    override = json.loads(ov_path.read_text()) if ov_path.exists() else None
    roles = decide_roles(desc, stat, override)
    ls = roles["lead_slot"]
    pair = {}
    for name, s in (("lead", ls), ("follow", 1 - ls)):
        sel = ids[:, s] >= 0
        pair[name] = {"slot": s, "present_share": round(float(sel.mean()), 3), "p_ge_0.9_share": round(float((p[:, s] >= 0.9).mean()), 3),
                      "median_height_px": round(float(np.median(P["height"][ids[sel, s]])), 1) if sel.any() else None,
                      "colours": {reg: {"rgb": slot_rgb[s, i].round(1).tolist(), "name": colour_name(slot_rgb[s, i])} for i, reg in enumerate(J.REGIONS)},
                      "expected": exp[name]}
    per = []
    F = len(ids)
    for sec in range(int(math.ceil(F / fps))):
        a, b = int(round(sec * fps)), min(int(round((sec + 1) * fps)), F)
        if a >= b:
            break
        per.append({"s": sec, "lead": round(float((ids[a:b, ls] >= 0).mean()), 2), "follow": round(float((ids[a:b, 1 - ls] >= 0).mean()), 2),
                    "p_lead": round(float(p[a:b, ls].mean()), 2), "p_follow": round(float(p[a:b, 1 - ls].mean()), 2),
                    "member_pass1": round(float(res["member1"][a:b].min(1).mean()), 2)})
    return {"method": "people_mono: dancer-likeness pair Viterbi (size vs per-frame reference, pose dynamics, centrality, couple distance) -> colour "
                      "slot split + self-trained colour passes -> one global role decision (override > description > stature > default)",
            "frames": F, "fps": fps, "size": [int(v) for v in P["size"]], "reference_height_px": round(float(np.median(res["ref"])), 1),
            "seconds": res["seconds"], "slot_split": res["split"], "pair": pair, "roles": roles, "per_second": per}


def cmd_sam(ctx: Ctx, tag: int = 0, ckpt: str | None = None, frames_dir: str | None = None, force: bool = False) -> None:
    out = ctx.inn / f"sam{tag}.npz"
    if done(out, force):
        return
    video = ctx.need_video().resolve()
    if ckpt:
        ck = Path(ckpt)
    else:
        from .dancers_sam import CHECKPOINT
        ck = CHECKPOINT
        if not ck.exists():
            from .front_pod import fetch_sam2
            fetch_sam2()
    fdir = Path(frames_dir) if frames_dir else scratch(ctx.take, "frames")
    run_script([HERE / "jl_people_sam.py", video, (ctx.people / f"anchors{tag}.json").resolve(), out.resolve(), "--ckpt", ck.resolve(), "--frames", fdir])


def n_frames_of(ctx: Ctx) -> tuple[int, float]:
    z = np.load(ctx.inn / "pose.npz")
    return int(z["n_frames"]), float(z["fps"])


def cmd_masks(ctx: Ctx, tag: int = 0, force: bool = False) -> None:
    out = ctx.people / f"fused{tag}.npz"
    if done(out, force):
        return
    M = J.MaskStore(ctx.inn / f"sam{tag}.npz")
    log(json.dumps(J.export_fused(M, n_frames_of(ctx)[0], out)))


def nvidia_lib_path() -> str:
    """site-packages/nvidia/*/lib of the running interpreter (onnxruntime-gpu takes the CUDA libraries of torch's wheels)."""
    import importlib.util
    try:
        spec = importlib.util.find_spec("nvidia")
    except (ImportError, ValueError):
        return ""
    dirs = [str(p) for loc in (spec.submodule_search_locations if spec else []) or [] for p in sorted(Path(loc).glob("*/lib"))]
    return ":".join(dirs)


def cmd_pose(ctx: Ctx, tag: int = 0, force: bool = False) -> None:
    out = ctx.inn / f"pose_wb{tag}.npz"
    if done(out, force):
        return
    env = {}
    if sys.platform.startswith("linux"):
        libs = nvidia_lib_path()
        if libs:
            env["LD_LIBRARY_PATH"] = libs + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    run_script([HERE / "jl_people_pose.py", ctx.need_video().resolve(), (ctx.people / f"fused{tag}.npz").resolve(), out.resolve()], env)


def copy_round0(ctx: Ctx, why: str) -> None:
    for a, b in ((ctx.people / "anchors0.json", ctx.people / "anchors1.json"), (ctx.people / "fused0.npz", ctx.people / "fused1.npz"),
                 (ctx.inn / "pose_wb0.npz", ctx.inn / "pose_wb1.npz")):
        shutil.copyfile(a, b)
    log(f"round 1 = round 0 ({why}): anchors / fused / pose_wb copied to the round-1 names")


def cmd_refine(ctx: Ctx, rounds: int = 2, ckpt: str | None = None, frames_dir: str | None = None, force: bool = False) -> None:
    if not force and (ctx.people / "fused1.npz").exists() and (ctx.inn / "pose_wb1.npz").exists():
        log("round 1 exists, skipped (--force to redo)")
        return
    if rounds <= 1:
        copy_round0(ctx, "--rounds 1")
        return
    F, fps = n_frames_of(ctx)
    anchors0 = json.loads((ctx.people / "anchors0.json").read_text())
    doc = refine_anchors(anchors0, ctx.people / "fused0.npz", ctx.inn / "pose_wb0.npz", F, fps)
    if doc is None:
        copy_round0(ctx, "the two SAM runs agree everywhere")
        return
    (ctx.people / "anchors1.json").write_text(json.dumps(doc))
    cmd_sam(ctx, 1, ckpt, frames_dir, force=True)
    cmd_masks(ctx, 1, force=True)
    cmd_pose(ctx, 1, force=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.people_mono", description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=CMDS)
    ap.add_argument("--take", required=True)
    ap.add_argument("--video", help="the source video (default: the take's first source)")
    ap.add_argument("--work-root", help="folder holding <take>/ (default dancecap/work)")
    ap.add_argument("--pose", help="detections npz to read instead of people/in/pose.npz (anchors, finalize)")
    ap.add_argument("--weights", default="yolo11x-pose.pt")
    ap.add_argument("--imgsz", type=int, nargs="+", help="detector inference sizes (default from the video's long side)")
    ap.add_argument("--ckpt", help="SAM 2.1 checkpoint (default dancers_sam.CHECKPOINT, fetched when missing)")
    ap.add_argument("--frames", help="scratch folder for the JPEG frames SAM reads")
    ap.add_argument("--rounds", type=int, default=2, help="SAM rounds: 1 skips the refinement")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    ctx = context(a.take, a.video, a.work_root)
    pose = Path(a.pose) if a.pose else None
    steps = {"detect": lambda: cmd_detect(ctx, a.weights, a.imgsz, a.force), "anchors": lambda: cmd_anchors(ctx, a.force, pose),
             "sam": lambda: cmd_sam(ctx, 0, a.ckpt, a.frames, a.force), "masks": lambda: cmd_masks(ctx, 0, a.force), "pose": lambda: cmd_pose(ctx, 0, a.force),
             "refine": lambda: cmd_refine(ctx, a.rounds, a.ckpt, a.frames, a.force), "finalize": lambda: finalize(ctx, pose_npz=pose),
             "review": lambda: review(ctx)}
    todo = ["detect", "anchors", "sam", "masks", "pose", "refine", "finalize"] if a.cmd == "run" else [a.cmd]
    t_all = time.time()
    for name in todo:
        t0, c0 = time.time(), time.process_time()
        steps[name]()
        record(ctx, name, time.time() - t0, time.process_time() - c0)
    if a.cmd == "run":
        record(ctx, "run", time.time() - t_all)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
