"""john_lula: who John (lead, bib 124) and Lula (follow, bib 116) are among everybody on the floor, per frame (agent J3a, 2026-10-11).

A floor heat: 3-5 other couples dance at the same time, judges and spectators stand round the edge, ONE handheld phone camera (538x960,
three zooms). Everybody is detected (jl_people_detect on a Runpod pod: yolo11x-pose at two inference sizes + bib digits read by EasyOCR on
every person crop), this stage decides WHO:

    persons    both inference passes merged into one list of persons per frame (greedy NMS on quality); the OCR reads are attributed to the person
               whose body axis is nearest to the read.
    bibs       John 124 on the back of his t-shirt, Lula 116 on her trousers: a read is evidence for a role (exact, contains, one digit off), another
               number is evidence AGAINST (another couple); the bibs are readable only when the back faces the camera, so they SEED the identity.
    colour     per-role Gaussian models of the torso / thigh / shin / head colours learned from the person-frames whose bib is read (John: black t-shirt;
               Lula: patterned crop top, grey trousers, auburn ponytail) against the colours of everybody else.
    track      ONE Viterbi over the ordered pair (candidate for John, candidate for Lula), each possibly missing: unary = bib + colour, couple distance
               (they dance together: a pair further apart than ~1.6 m costs), transition = image motion of each slot's person relative to its size
               (warped through the measured camera motion of J1, so the three zoom jumps do not break the track). Forward-backward marginals give the
               role confidence: crossings with other couples / occlusions give LOW confidence, not a confident wrong answer.
    verify     bib reads that disagree with the final identity (a read of 124 on the person chosen for Lula) are counted per second.

    python -m dancecap.jl_people run         people/teachers_01.npz (+ pose_tables/01.npz, teachers.json, verification.json)
    python -m dancecap.jl_people review      contact sheets in review/people/

Output contract = recap_people's teachers_<cam>.npz (CONTRACTS.md "Recap people"): t, frame_index, ok, row (F, 2) [lead, follow], box, kp, conf, plus
merged, role_conf, floor (NaN: the room calibration is J2's job), size, attached.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .take import PACKAGE_ROOT, WORKSPACE

WORK = PACKAGE_ROOT / "work" / "john_lula"
PEOPLE = WORK / "people"
IN = PEOPLE / "in"
VIDEO = WORKSPACE / "dances" / "John-Lula" / "john_lula_jnj.mp4"
BIB = {"lead": "124", "follow": "116"}
ZOOMS = [(17.84, 18.57), (51.51, 52.24), (78.75, 79.38)]
JOINT_CONF = 0.3
MIN_JOINTS = 6


def log(msg: str) -> None:
    print(f"[jl_people {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------------------------------------ pure helpers


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / u) if u > 0 else 0.0


def box_cover(a: np.ndarray, b: np.ndarray) -> float:
    """Share of the SMALLER box that lies inside the other."""
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    s = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return float(inter / s) if s > 0 else 0.0


def edit_distance(a: str, b: str) -> int:
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
    return d[len(b)]


def bib_score(text: str, target: str, others: tuple[str, ...] = ()) -> float:
    """How much a read supports `target`: 1 exact, 0.9 contains it, 0.35 one edit away (len >= 2) when it is not just as close to a
    different known bib, else 0."""
    if not text:
        return 0.0
    if text == target:
        return 1.0
    if target in text:
        return 0.9
    if len(text) >= 2 and edit_distance(text, target) == 1 and all(edit_distance(text, o) > 1 for o in others):
        return 0.35
    return 0.0


def quality(kp: np.ndarray, conf: float) -> float:
    c = kp[:, 2]
    return float(conf + c[c >= JOINT_CONF].sum() / 17.0)


def merge_passes(box: np.ndarray, kp: np.ndarray, conf: np.ndarray, frames: np.ndarray, n_frames: int, iou_thr: float = 0.5, cover_thr: float = 0.85) -> dict:
    """Both inference passes (and the detector's double boxes) -> one person list per frame. Greedy by quality; a detection that overlaps a kept one
    (IoU > iou_thr, or contained to cover_thr by it AND of similar height) is a duplicate. Returns flat arrays + `src` (the raw detection indices each
    person absorbed) + per-frame start offsets."""
    order = np.argsort(frames, kind="stable")
    starts = np.searchsorted(frames[order], np.arange(n_frames + 1))
    keep_idx, src = [], []
    fstart = np.zeros(n_frames + 1, int)
    for f in range(n_frames):
        ids = order[starts[f]:starts[f + 1]]
        q = np.array([quality(kp[i], conf[i]) for i in ids])
        kept: list[int] = []
        grp: list[list[int]] = []
        for k in np.argsort(-q):
            i = int(ids[k])
            dup = -1
            for a, j in enumerate(kept):
                hi, hj = box[i, 3] - box[i, 1], box[j, 3] - box[j, 1]
                if box_iou(box[i], box[j]) > iou_thr or (box_cover(box[i], box[j]) > cover_thr and 0.7 < hi / max(hj, 1e-6) < 1.4):
                    dup = a
                    break
            if dup >= 0:
                grp[dup].append(i)
            else:
                kept.append(i)
                grp.append([i])
        keep_idx += kept
        src += grp
        fstart[f + 1] = len(keep_idx)
    return {"idx": np.array(keep_idx, int), "src": src, "fstart": fstart}


def axis_distance(p: np.ndarray, kp: np.ndarray, box: np.ndarray) -> float:
    """Distance of point p to the person's body axis (neck -> hip middle -> knee middle), in box widths (large when the axis is unknown)."""
    c = kp[:, 2]
    pts = []
    if (c[[5, 6]] >= JOINT_CONF).all():
        pts.append(kp[[5, 6], :2].mean(0))
    if (c[[11, 12]] >= JOINT_CONF).all():
        pts.append(kp[[11, 12], :2].mean(0))
    if (c[[13, 14]] >= JOINT_CONF).all():
        pts.append(kp[[13, 14], :2].mean(0))
    w = max(box[2] - box[0], 1.0)
    if len(pts) < 2:
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        return float(np.hypot(p[0] - cx, p[1] - cy) / w) + 1.0
    best = 1e9
    for a, b in zip(pts[:-1], pts[1:]):
        ab = b - a
        t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-9), 0, 1)
        best = min(best, float(np.linalg.norm(p - (a + t * ab))))
    return best / w


def pair_states(m: int, same_ok: np.ndarray | None = None) -> np.ndarray:
    """All ordered (john, lula) assignments of m candidates, index 0 = missing; both present on the same candidate only when it is `same_ok` (one
    detection holding both bodies: an embrace)."""
    st = [(a, b) for a in range(m + 1) for b in range(m + 1)
          if a == 0 or b == 0 or a != b or (same_ok is not None and bool(same_ok[a - 1]))]
    return np.array(st, int)


# ------------------------------------------------------------------------------------------------ loading


def load(pose_npz: Path | None = None, ocr_json: Path | None = None) -> dict:
    """The pod's detections merged into persons per frame, with the bib reads attributed to persons.

    Returns persons arrays (P,): frame, box, kp (P, 17, 3), conf, col (P, 4, 3), thumb (P, 32, 16, 3), height; fstart (F+1,); `reads` per person: list of
    (text, conf) attributed to it; fps, size, n_frames."""
    z = np.load(pose_npz or IN / "pose.npz")
    F = int(z["n_frames"])
    m = merge_passes(z["box"], z["kp"], z["conf"], z["frame"], F)
    idx = m["idx"]
    P = {"frame": z["frame"][idx], "box": z["box"][idx], "kp": z["kp"][idx], "conf": z["conf"][idx], "col": z["col"][idx], "thumb": z["thumb"][idx],
         "fstart": m["fstart"], "fps": float(z["fps"]), "size": z["size"], "n_frames": F}
    P["height"] = P["box"][:, 3] - P["box"][:, 1]
    raw_to_person = np.full(len(z["frame"]), -1, int)
    for p, grp in enumerate(m["src"]):
        raw_to_person[grp] = p
    reads: list[list[tuple[str, float]]] = [[] for _ in range(len(idx))]
    n_reads = n_att = 0
    ocr = json.loads((ocr_json or IN / "ocr.json").read_text())
    for r in ocr["reads"]:
        n_reads += 1
        txt = r["text"].strip()
        if not txt:
            continue
        q = np.array(r["quad"], float)
        c = q.mean(0)
        f = int(r["frame"])
        cand = range(P["fstart"][f], P["fstart"][f + 1])
        home = int(raw_to_person[r["det"]])
        best, bd = home, 1e9
        for p in cand:
            b = P["box"][p]
            if not (b[0] <= c[0] <= b[2] and b[1] <= c[1] <= b[3]):
                continue
            d = axis_distance(c, P["kp"][p], b)
            if d < bd:
                best, bd = p, d
        if best >= 0:
            reads[best].append((txt, float(r["conf"])))
            n_att += 1
    P["reads"] = reads
    log(f"{len(idx)} persons over {F} frames ({len(idx) / F:.1f} per frame), {n_att}/{n_reads} bib reads attributed")
    return P


# ------------------------------------------------------------------------------------------------ evidence

KNOWN_OTHER = ("126",)       # a bib that is read again and again on a person who is neither dancer (rank 2 in size, black clothes: another couple)
MAX_CAND = 7
MIN_H = 70.0                 # px
MISS_COST = 2.0
MERGE_COST = 1.0            # one detection for both bodies (embrace)
MERGE_ASPECT = 0.5           # a box this wide / high may hold both bodies
W_RATIO = 1.0               # the two dancers' heights agree (depth differences allowed)
REACQ_COST = 1.0
LOSE_COST = 0.5
MOVE_SCALE = 0.10            # image displacement per frame / person size at which the continuity cost is 1
MOVE_CAP = 30.0
W_BIB = 4.0                  # log-odds of an exactly read bib
W_COL = 0.6                  # per colour region (clipped +-COL_CLIP)
COL_CLIP = 2.5
COUPLE_GAP = 0.12            # gap between the two boxes / height up to which the pair is free (they dance together: the boxes touch or overlap)
W_COUPLE = 40.0
W_TOUCH = 3.0                # reward for a pair whose boxes overlap (share of the smaller box)
W_SIZE = 4.0
PAIR_CAP = 8.0
REGIONS = ("torso", "thigh", "shin", "head")
REGION_W = {"torso": 1.0, "thigh": 1.0, "shin": 0.4, "head": 0.4}   # shoes / socks / a face or hair seen from either side are weak evidence


def scale_track(n_frames: int, fps: float, zooms=ZOOMS, ratios=(1.97, 0.51, 1.89)) -> np.ndarray:
    """Zoom factor of the phone per frame (1 at the start; the three measured 0.7 s steps as linear ramps in log scale)."""
    t = np.arange(n_frames) / fps
    logs = np.zeros(n_frames)
    cum = 0.0
    for (t0, t1), r in zip(zooms, ratios):
        a = np.clip((t - t0) / (t1 - t0), 0, 1)
        logs += a * np.log(r)
    return np.exp(logs)


def person_bib(reads: list[tuple[str, float]]) -> tuple[float, float, float]:
    """(john, lula, other) bib evidence in 0..1 of one person-frame from its OCR reads."""
    j = l = o = 0.0
    for t, c in reads:
        w = float(np.clip((c - 0.2) / 0.6, 0.0, 1.0))
        j = max(j, bib_score(t, BIB["lead"], (BIB["follow"],) + KNOWN_OTHER) * w)
        l = max(l, bib_score(t, BIB["follow"], (BIB["lead"],) + KNOWN_OTHER) * w)
        if len(t) == 3 and t not in BIB.values() and t in KNOWN_OTHER:
            o = max(o, w)
    return j, l, o


def hip_centre(kp: np.ndarray, box: np.ndarray) -> np.ndarray:
    c = kp[:, 2]
    if (c[[11, 12]] >= JOINT_CONF).all():
        return kp[[11, 12], :2].mean(0)
    return np.array([(box[0] + box[2]) / 2, box[1] + 0.55 * (box[3] - box[1])])


def fit_gauss(x: np.ndarray, floor: float = 18.0) -> tuple[np.ndarray, np.ndarray]:
    x = x[np.isfinite(x).all(1)]
    med = np.median(x, 0)
    sd = np.maximum(1.4826 * np.median(np.abs(x - med), 0), floor)
    return med, sd


def gauss_ll(x: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """(N,) log density of rows of x (N, 3) under a diagonal Gaussian; NaN rows -> NaN."""
    return (-0.5 * ((x - mu) / sd) ** 2 - np.log(sd)).sum(-1)


def colour_models(col: np.ndarray, seeds: dict[str, np.ndarray], allmask: np.ndarray) -> dict:
    """{role: {region: (mu, sd)}, 'other': {region: (mu, sd)}} from the seeds' colours; 'other' = the broad distribution of everybody."""
    out: dict = {"lead": {}, "follow": {}, "other": {}}
    for r, name in enumerate(REGIONS):
        for role in ("lead", "follow"):
            idx = seeds[role]
            if len(idx) >= 8:
                out[role][name] = fit_gauss(col[idx, r])
        x = col[allmask, r]
        x = x[np.isfinite(x).all(1)]
        out["other"][name] = (x.mean(0), np.maximum(x.std(0), 25.0))
    return out


def colour_llr(col: np.ndarray, models: dict, role: str) -> np.ndarray:
    """(N,) summed clipped log-likelihood ratio (role vs everybody) of the four region colours; missing regions add 0."""
    tot = np.zeros(len(col))
    for r, name in enumerate(REGIONS):
        if name not in models[role]:
            continue
        x = col[:, r]
        ll = gauss_ll(x, *models[role][name]) - gauss_ll(x, *models["other"][name])
        tot += REGION_W[name] * np.where(np.isfinite(ll), np.clip(ll, -COL_CLIP, COL_CLIP), 0.0)
    return W_COL * tot


# ------------------------------------------------------------------------------------------------ the tracker


def move_cost(prev_c: np.ndarray, prev_h: np.ndarray, cur_c: np.ndarray, cur_h: np.ndarray) -> np.ndarray:
    """(P+1, Q+1) cost of one slot going from candidate j of the previous frame (index 0 = missing) to candidate k of this one (prev_c already warped)."""
    P, Q = len(prev_c), len(cur_c)
    T = np.zeros((P + 1, Q + 1))
    T[0, 1:] = REACQ_COST
    T[1:, 0] = LOSE_COST
    if P and Q:
        d = np.linalg.norm(prev_c[:, None, :] - cur_c[None, :, :], axis=-1) / np.maximum(np.maximum(prev_h[:, None], cur_h[None, :]), 40.0)
        T[1:, 1:] = np.minimum((d / MOVE_SCALE) ** 2, MOVE_CAP)
    return T


def pair_unary(sts: np.ndarray, sj: np.ndarray, sl: np.ndarray, pc: np.ndarray) -> np.ndarray:
    """Cost of every state: missing cost or minus the role score of the candidate, plus the couple-distance cost of a present pair; a merged state
    (both roles on one detection) costs MERGE_COST minus the mean role score."""
    sj0, sl0 = np.r_[-MISS_COST, sj], np.r_[-MISS_COST, sl]          # score of "missing" = -MISS_COST so cost = -score
    cost = -sj0[sts[:, 0]] - sl0[sts[:, 1]]
    both = (sts[:, 0] > 0) & (sts[:, 1] > 0)
    same = both & (sts[:, 0] == sts[:, 1])
    diff = both & ~same
    if pc.size:
        cost = cost + np.where(diff, pc[np.maximum(sts[:, 0] - 1, 0), np.maximum(sts[:, 1] - 1, 0)], 0.0)
        a = np.maximum(sts[:, 0] - 1, 0)
        hi, lo = np.maximum(sj[a], sl[a]), np.minimum(sj[a], sl[a])
        cost = np.where(same, MERGE_COST - hi - np.maximum(lo, -1.0), cost)
    return cost


def lse(x: np.ndarray, axis: int) -> np.ndarray:
    m = x.max(axis, keepdims=True)
    return (m + np.log(np.exp(x - m).sum(axis, keepdims=True))).squeeze(axis)


def track_pair(frames: list[dict]) -> dict:
    """Viterbi + forward-backward over the frames. Each frame: c (m, 2) warped-from-previous is applied by the caller via `cw` (the previous
    frame's centres in this frame's pixels), h (m,), sj, sl (m,), pc (m, m). Returns per frame the chosen candidate (-1 none) of John and Lula and the
    marginal probability of that choice per role."""
    n = len(frames)
    sts = [pair_states(len(f["h"]), f.get("same_ok")) for f in frames]
    U = [pair_unary(s, f["sj"], f["sl"], f["pc"]) for s, f in zip(sts, frames)]
    V = [U[0]]
    back: list = [None]
    trans: list = [None]
    for i in range(1, n):
        a, b = frames[i - 1], frames[i]
        Tc = move_cost(b["cw"], a["h"], b["c"], b["h"])
        T = Tc[np.ix_(sts[i - 1][:, 0], sts[i][:, 0])] + Tc[np.ix_(sts[i - 1][:, 1], sts[i][:, 1])]
        trans.append(T)
        tot = V[-1][:, None] + T
        back.append(np.argmin(tot, 0))
        V.append(U[i] + tot.min(0))
    path = np.zeros(n, int)
    path[-1] = int(np.argmin(V[-1]))
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i][path[i]]
    al = [-U[0]]
    for i in range(1, n):
        al.append(-U[i] + lse(al[-1][:, None] - trans[i], 0))
    be: list = [None] * n
    be[-1] = np.zeros(len(sts[-1]))
    for i in range(n - 2, -1, -1):
        be[i] = lse(-trans[i + 1] - U[i + 1][None, :] + be[i + 1][None, :], 1)
    cand = np.full((n, 2), -1, int)
    pc = np.zeros((n, 2))
    for i in range(n):
        g = al[i] + be[i]
        g = np.exp(g - g.max())
        g /= g.sum()
        s = sts[i][path[i]]
        cand[i] = s - 1
        for r in range(2):
            pc[i, r] = float(g[sts[i][:, r] == s[r]].sum())
    return {"cand": cand, "p": pc}


def camera_warp(n_frames: int) -> np.ndarray:
    """(F, 4) per frame k: scale, rot, tx, ty of the similarity from frame k-1 to k (J1's camera_motion.json), identity where missing."""
    w = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n_frames, 1))
    p = WORK / "camera_motion.json"
    if p.exists():
        for d in json.loads(p.read_text()):
            if 0 < d["frame"] < n_frames:
                w[d["frame"]] = [d["scale"], d["rot"], d["tx"], d["ty"]]
    return w


def candidate_ids(P: dict, f: int) -> np.ndarray:
    ids = np.arange(P["fstart"][f], P["fstart"][f + 1])
    if not len(ids):
        return ids
    nj = (P["kp"][ids, :, 2] >= JOINT_CONF).sum(1)
    ok = (nj >= MIN_JOINTS - 1) & (P["height"][ids] >= MIN_H)
    ids = ids[ok]
    if len(ids) > MAX_CAND:
        ids = ids[np.argsort(-P["height"][ids])[:MAX_CAND]]
    return ids


def build_frames(P: dict, models: dict, ref_h: float, warp: np.ndarray, S: np.ndarray, w_size: float = W_SIZE) -> list[dict]:
    """Per frame: the candidates and their role scores / pair costs / continuity inputs (centres warped from the previous frame)."""
    F = P["n_frames"]
    centre = np.array([P["size"][0] / 2, P["size"][1] / 2], float)
    bibs = np.array([person_bib(r) for r in P["reads"]]).reshape(-1, 3)
    cj = colour_llr(P["col"], models, "lead")
    cl = colour_llr(P["col"], models, "follow")
    hip = np.array([hip_centre(P["kp"][p], P["box"][p]) for p in range(len(P["frame"]))])
    frames: list[dict] = []
    prev_ids = prev_c = None
    for f in range(F):
        ids = candidate_ids(P, f)
        h = P["height"][ids]
        c = hip[ids]
        lr = np.log(np.maximum(h, 1.0) / (ref_h * S[f]))
        size_cost = w_size * np.where(lr < 0, lr ** 2, 0.4 * lr ** 2)
        b = bibs[ids]
        sj = W_BIB * (b[:, 0] - 0.7 * b[:, 1] - 0.7 * b[:, 2]) + cj[ids] - size_cost
        sl = W_BIB * (b[:, 1] - 0.7 * b[:, 0] - 0.7 * b[:, 2]) + cl[ids] - size_cost
        m = len(ids)
        pc = np.zeros((m, m))
        if m:
            bx = P["box"][ids]
            gx = np.maximum(0.0, np.maximum(bx[:, None, 0], bx[None, :, 0]) - np.minimum(bx[:, None, 2], bx[None, :, 2]))
            gy = np.maximum(0.0, np.maximum(bx[:, None, 1], bx[None, :, 1]) - np.minimum(bx[:, None, 3], bx[None, :, 3]))
            gap = np.hypot(gx, gy) / np.maximum(np.maximum(h[:, None], h[None]), 40.0)
            pc = W_COUPLE * np.maximum(gap - COUPLE_GAP, 0.0) ** 2
            pc = pc + W_RATIO * np.log(np.maximum(h[:, None], 1.0) / np.maximum(h[None], 1.0)) ** 2
            pc = np.minimum(pc, PAIR_CAP)
            ix = np.maximum(0.0, np.minimum(bx[:, None, 2], bx[None, :, 2]) - np.maximum(bx[:, None, 0], bx[None, :, 0]))
            iy = np.maximum(0.0, np.minimum(bx[:, None, 3], bx[None, :, 3]) - np.maximum(bx[:, None, 1], bx[None, :, 1]))
            area = (bx[:, 2] - bx[:, 0]) * (bx[:, 3] - bx[:, 1])
            cover = ix * iy / np.maximum(np.minimum(area[:, None], area[None]), 1.0)
            pc = pc - W_TOUCH * np.minimum(cover, 1.0)               # dancing together: the boxes overlap
        sc, rot, tx, ty = warp[f]
        if prev_c is not None and len(prev_c):
            R = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
            cw = (sc * (R @ (prev_c - centre).T)).T + centre + np.array([tx, ty])
        else:
            cw = np.zeros((0, 2))
        same_ok = (h >= 0.6 * ref_h * S[f]) & ((P["box"][ids, 2] - P["box"][ids, 0]) / np.maximum(h, 1.0) >= MERGE_ASPECT) if m else np.zeros(0, bool)
        frames.append({"ids": ids, "c": c, "h": h, "sj": sj, "sl": sl, "pc": pc, "cw": cw, "same_ok": same_ok})
        prev_c = c
    return frames


def seed_sets(P: dict, min_score: float = 0.9) -> dict[str, np.ndarray]:
    bibs = np.array([person_bib(r) for r in P["reads"]]).reshape(-1, 3)
    big = P["height"] >= 120
    return {"lead": np.flatnonzero((bibs[:, 0] >= min_score) & (bibs[:, 1] < 0.3) & big), "follow": np.flatnonzero((bibs[:, 1] >= min_score) & (bibs[:, 0] < 0.3) & big)}


def solve(P: dict, iters: int = 2) -> dict:
    """Seeds from the bibs -> colour models -> tracker -> refit the colour models on the confident frames -> tracker again."""
    F = P["n_frames"]
    S = scale_track(F, P["fps"])
    warp = camera_warp(F)
    seeds = seed_sets(P)
    allmask = P["height"] >= 150
    ref_h = float(np.median(P["height"][np.r_[seeds["lead"], seeds["follow"]]] / S[P["frame"][np.r_[seeds["lead"], seeds["follow"]]]]))
    log(f"seeds: lead {len(seeds['lead'])}, follow {len(seeds['follow'])} person-frames; reference height {ref_h:.0f} px at 1x")
    res = None
    for it in range(iters + 1):
        models = colour_models(P["col"], seeds, allmask)
        frames = build_frames(P, models, ref_h, warp, S)
        res = track_pair(frames)
        res["frames"] = frames
        res["models"] = models
        ids = np.full((F, 2), -1, int)
        for i, f in enumerate(frames):
            for r in range(2):
                if res["cand"][i, r] >= 0:
                    ids[i, r] = f["ids"][res["cand"][i, r]]
        res["ids"] = ids
        conf = res["p"]
        log(f"iteration {it}: present {(ids >= 0).mean(0).round(3)}, p>=0.9 {(conf >= 0.9).mean(0).round(3)}")
        if it == iters:
            break
        # self-training: confident chosen persons become the colour seeds (bib seeds stay in)
        for r, role in enumerate(("lead", "follow")):
            sel = np.flatnonzero((ids[:, r] >= 0) & (conf[:, r] >= 0.9))
            seeds[role] = np.unique(np.r_[seed_sets(P)[role], ids[sel, r]])
        both = np.flatnonzero((ids >= 0).all(1) & (ids[:, 0] != ids[:, 1]) & (conf >= 0.9).all(1))
        if len(both) > 100:
            ref_h = float(np.median(P["height"][ids[both].ravel()] / np.repeat(S[both], 2)))
            log(f"reference height -> {ref_h:.0f} px at 1x ({len(both)} confident pairs)")
    return res


# ------------------------------------------------------------------------------------------------ review

SKEL = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]


def draw_person(img, kp, box, col, s, label):
    import cv2
    b = box * s
    cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), col, 2)
    for a, c in SKEL:
        if kp[a, 2] >= JOINT_CONF and kp[c, 2] >= JOINT_CONF:
            cv2.line(img, (int(kp[a, 0] * s), int(kp[a, 1] * s)), (int(kp[c, 0] * s), int(kp[c, 1] * s)), col, 1)
    cv2.putText(img, label, (int(b[0]), max(14, int(b[1]) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)


def review(P: dict, ids: np.ndarray, pconf: np.ndarray, times: list[float], out: Path, name: str = "people", scale: float = 0.7, cols: int = 4, per_sheet: int = 8) -> list[Path]:
    import cv2
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(VIDEO))
    fps = P["fps"]
    tiles = []
    for t in times:
        f = int(round(t * fps))
        f = min(max(f, 0), P["n_frames"] - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            continue
        img = cv2.resize(img, None, fx=scale, fy=scale)
        for p in range(P["fstart"][f], P["fstart"][f + 1]):
            if P["height"][p] < 60:
                continue
            b = P["box"][p] * scale
            cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (140, 140, 140), 1)
            for txt, c in P["reads"][p]:
                if c > 0.4 and len(txt) >= 2:
                    cv2.putText(img, txt, (int(b[0]), int(b[3]) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
        for r, (col, nm) in enumerate((((60, 220, 60), "JOHN"), ((220, 60, 220), "LULA"))):
            p = int(ids[f, r])
            if p >= 0:
                draw_person(img, P["kp"][p], P["box"][p], col, scale, f"{nm} {pconf[f, r]:.2f}")
        cv2.putText(img, f"t={f / fps:.2f}s f={f}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        tiles.append(img)
    cap.release()
    paths = []
    for k in range(0, len(tiles), per_sheet):
        grp = tiles[k:k + per_sheet]
        while len(grp) % cols:
            grp.append(np.zeros_like(grp[0]))
        sheet = np.vstack([np.hstack(grp[j:j + cols]) for j in range(0, len(grp), cols)])
        p = out / f"{name}_{k // per_sheet:02d}.jpg"
        cv2.imwrite(str(p), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
        paths.append(p)
    return paths


# ------------------------------------------------------------------------------------------------ anchors for the SAM 2 pass


def anchor_quality(P: dict, res: dict, ref_h: float, S: np.ndarray, p_min: float = 0.95, joints_min: int = 13, score_min: float = 0.5) -> np.ndarray:
    """(F,) 0 for a frame that cannot anchor SAM, else a cleanliness score (higher = better): both dancers on DIFFERENT detections, both role marginals
    >= 0.95, good joints on both, similar heights of plausible size, boxes overlapping less than 0.6, role scores positive."""
    ids, p = res["ids"], res["p"]
    F = P["n_frames"]
    q = np.zeros(F)
    for f in range(F):
        a, b = ids[f]
        if a < 0 or b < 0 or a == b or p[f].min() < p_min:
            continue
        ka, kb = P["kp"][a], P["kp"][b]
        na, nb = int((ka[:, 2] >= JOINT_CONF).sum()), int((kb[:, 2] >= JOINT_CONF).sum())
        if min(na, nb) < joints_min:
            continue
        ha, hb = P["height"][a], P["height"][b]
        r = ref_h * S[f]
        if min(ha, hb) < 0.7 * r or max(ha, hb) > 2.2 * r or not (0.65 < ha / hb < 1.55):
            continue
        iou = box_iou(P["box"][a], P["box"][b])
        if iou > 0.6:
            continue
        fr = res["frames"][f]
        ia, ib = int(np.flatnonzero(fr["ids"] == a)[0]), int(np.flatnonzero(fr["ids"] == b)[0])
        sj, sl = fr["sj"][ia], fr["sl"][ib]
        if min(sj, sl) < score_min:
            continue
        q[f] = (min(ka[:, 2].mean(), kb[:, 2].mean())) * (1.0 - 0.5 * iou) * min(p[f]) + 0.05 * min(sj, sl, 6.0)
    return q


def pick_anchors(q: np.ndarray, min_gap: int = 24, max_gap: int = 90) -> list[int]:
    """Greedy anchors: best frames first with at least min_gap between them, then the best remaining frame inside every gap longer than max_gap."""
    order = [int(i) for i in np.argsort(-q) if q[i] > 0]
    chosen: list[int] = []
    for i in order:
        if all(abs(i - j) >= min_gap for j in chosen):
            chosen.append(i)
    chosen.sort()
    return chosen


def fill_gaps(chosen: list[int], q_relaxed: np.ndarray, max_gap: int = 90, min_gap: int = 24) -> list[int]:
    """Add the best relaxed-quality frame inside every gap longer than max_gap (repeatedly), a frame at least min_gap from its neighbours."""
    chosen = sorted(chosen)
    while True:
        pts = [-1] + chosen + [len(q_relaxed)]
        added = False
        for a, b in zip(pts[:-1], pts[1:]):
            if b - a <= max_gap:
                continue
            lo, hi = (a + min_gap if a >= 0 else 0), (b - min_gap if b < len(q_relaxed) else len(q_relaxed) - 1)
            if hi <= lo:
                continue
            seg = q_relaxed[lo:hi + 1]
            if seg.max() <= 0:
                continue
            # prefer the frame nearest the middle of the gap among the top-quality ones
            top = np.flatnonzero(seg >= 0.7 * seg.max()) + lo
            mid = (a + b) / 2 if a >= 0 and b < len(q_relaxed) else (lo + hi) / 2
            chosen.append(int(top[np.argmin(np.abs(top - mid))]))
            chosen.sort()
            added = True
            break
        if not added:
            return sorted(chosen)


def anchor_doc(P: dict, res: dict, frames: list[int]) -> list[dict]:
    """JSON-able prompts of the anchors: per role the box, positive points (the person's own torso / hip joints) and negative points (the partner's)."""
    out = []
    for f in frames:
        a, b = (int(x) for x in res["ids"][f])
        d = {"frame": int(f)}
        for role, me, other in (("john", a, b), ("lula", b, a)):
            def pts(i, idx):
                k = P["kp"][i]
                return [[float(k[j, 0]), float(k[j, 1])] for j in idx if k[j, 2] >= 0.5]
            d[role] = {"box": [float(v) for v in P["box"][me]], "pos": pts(me, (5, 6, 11, 12, 13, 14)), "neg": pts(other, (5, 6, 11, 12))}
        out.append(d)
    return out


# ------------------------------------------------------------------------------------------------ SAM masks: decode, fuse, check


class MaskStore:
    """The pod's SAM runs (jl_people_sam): per frame and dancer the masks carried from the previous anchor (forward run) and from the next one (backward run)."""

    def __init__(self, npz_path: Path):
        self.z = np.load(npz_path, allow_pickle=False)
        self.runs = json.loads(str(self.z["runs"]))
        self.w, self.h = (int(v) for v in self.z["size"])
        self.keys = set(self.z.files)
        self.by_frame: dict[int, list[tuple[int, str]]] = {}
        for r in self.runs:
            for f in range(r["lo"], r["hi"] + 1):
                if f"m_{r['anchor']}_{r['dir']}_{f}_0" in self.keys:
                    self.by_frame.setdefault(f, []).append((r["anchor"], r["dir"]))

    def mask(self, anchor: int, d: str, f: int, obj: int) -> np.ndarray | None:
        import cv2
        k = f"m_{anchor}_{d}_{f}_{obj}"
        if k not in self.keys:
            return None
        return cv2.imdecode(self.z[k], cv2.IMREAD_GRAYSCALE) > 127

    def score(self, anchor: int, d: str, f: int, obj: int) -> float:
        r = next(r for r in self.runs if r["anchor"] == anchor and r["dir"] == d)
        return float(self.z[f"s_{anchor}_{d}"][f - r["lo"], obj])


def mask_iou(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return float("nan")
    u = int((a | b).sum())
    return float((a & b).sum() / u) if u else float("nan")


def mask_box(m: np.ndarray) -> np.ndarray:
    ys, xs = np.flatnonzero(m.any(1)), np.flatnonzero(m.any(0))
    if not len(ys):
        return np.full(4, np.nan, np.float32)
    return np.array([xs[0], ys[0], xs[-1] + 1, ys[-1] + 1], np.float32)


def fused_mask(M: MaskStore, f: int, obj: int) -> tuple[np.ndarray | None, dict]:
    """The mask of one dancer at frame f from the nearest anchor's run, plus {"other": the other run's mask agreement (IoU, nan if none), "dist": frames
    from the anchor}."""
    runs = M.by_frame.get(f, [])
    if not runs:
        return None, {"agree": float("nan"), "dist": -1}
    runs = sorted(runs, key=lambda r: abs(r[0] - f))
    a, d = runs[0]
    m = M.mask(a, d, f, obj)
    agree = float("nan")
    for a2, d2 in runs[1:]:
        if a2 != a:
            agree = mask_iou(m, M.mask(a2, d2, f, obj))
            break
    return m, {"agree": agree, "dist": abs(a - f)}


def review_masks(M: MaskStore, P: dict, times: list[float], out: Path, name: str, scale: float = 0.7, cols: int = 4, per_sheet: int = 8) -> list[Path]:
    import cv2
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(VIDEO))
    tiles = []
    for t in times:
        f = min(max(int(round(t * P["fps"])), 0), P["n_frames"] - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            continue
        over = img.copy()
        txt = []
        for o, col in enumerate(((60, 220, 60), (220, 60, 220))):
            m, info = fused_mask(M, f, o)
            if m is None:
                continue
            over[m] = (0.45 * over[m] + 0.55 * np.array(col)).astype(np.uint8)
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(over, cs, -1, col, 2)
            txt.append(f"{'JL'[o]}:{info['agree']:.2f}")
        img = cv2.resize(over, None, fx=scale, fy=scale)
        cv2.putText(img, f"t={f / P['fps']:.2f}s f={f} agree " + " ".join(txt), (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        tiles.append(img)
    cap.release()
    paths = []
    for k in range(0, len(tiles), per_sheet):
        grp = tiles[k:k + per_sheet]
        while len(grp) % cols:
            grp.append(np.zeros_like(grp[0]))
        sheet = np.vstack([np.hstack(grp[j:j + cols]) for j in range(0, len(grp), cols)])
        p = out / f"{name}_{k // per_sheet:02d}.jpg"
        cv2.imwrite(str(p), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
        paths.append(p)
    return paths


def clean_mask(m: np.ndarray, keep_frac: float = 0.12) -> np.ndarray:
    """Drop speckle: keep connected components of at least keep_frac of the largest one."""
    import cv2
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    if n <= 2:
        return m
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.flatnonzero(areas >= keep_frac * areas.max()) + 1
    return np.isin(lab, keep)


def export_fused(M: MaskStore, n_frames: int, out: Path) -> dict:
    """Fused, cleaned masks per frame -> npz for the pose pod job: box (F, 2, 4), agree (F, 2), dist (F, 2), area (F, 2), m_<f>_<o> PNG bytes."""
    import cv2
    box = np.full((n_frames, 2, 4), np.nan, np.float32)
    agree = np.full((n_frames, 2), np.nan, np.float32)
    dist = np.full((n_frames, 2), -1, np.int32)
    area = np.zeros((n_frames, 2), np.int32)
    store = {}
    for f in range(n_frames):
        for o in range(2):
            m, info = fused_mask(M, f, o)
            if m is None:
                continue
            m = clean_mask(m)
            agree[f, o], dist[f, o] = info["agree"], info["dist"]
            area[f, o] = int(m.sum())
            if area[f, o] > 50:
                box[f, o] = mask_box(m)
                store[f"m_{f}_{o}"] = np.frombuffer(cv2.imencode(".png", (m * 255).astype(np.uint8))[1].tobytes(), np.uint8)
    np.savez(out, box=box, agree=agree, dist=dist, area=area, **store)
    return {"frames": n_frames, "with_mask": (area > 50).mean(0).round(3).tolist()}


def review_kp(kp: np.ndarray, times: list[float], fps: float, out: Path, name: str, scale: float = 0.7, cols: int = 4, per_sheet: int = 8, conf: np.ndarray | None = None) -> list[Path]:
    """Sheets of the whole-body skeletons (kp (F, 2, K, 3), K = 17 body [+ 6 feet + 68 face + 42 hands]): John green, Lula magenta; joints under 0.3 not drawn."""
    import cv2
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(VIDEO))
    feet = [(15, 17), (15, 18), (15, 19), (16, 20), (16, 21), (16, 22)]
    hands = []
    for base in (91, 112):
        for fgr in range(5):
            idx = [base] + [base + 1 + 4 * fgr + j for j in range(4)]
            hands += list(zip(idx[:-1], idx[1:]))
    tiles = []
    for t in times:
        f = min(max(int(round(t * fps)), 0), len(kp) - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            continue
        for o, col in enumerate(((60, 220, 60), (220, 60, 220))):
            k = kp[f, o]
            if not np.isfinite(k[:17, :2]).any():
                continue
            segs = SKEL + (feet + hands if k.shape[0] >= 133 else [])
            for a, b in segs:
                if a < len(k) and b < len(k) and k[a, 2] >= 0.3 and k[b, 2] >= 0.3:
                    cv2.line(img, (int(k[a, 0]), int(k[a, 1])), (int(k[b, 0]), int(k[b, 1])), col, 2)
            for j in range(min(len(k), 23)):
                if k[j, 2] >= 0.3:
                    cv2.circle(img, (int(k[j, 0]), int(k[j, 1])), 3, col, -1)
        img = cv2.resize(img, None, fx=scale, fy=scale)
        cv2.putText(img, f"t={f / fps:.2f}s f={f}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        tiles.append(img)
    cap.release()
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


# ------------------------------------------------------------------------------------------------ whole-body keypoints: variant choice, anchors from skeletons

RTMW_SCALE = 6.0     # RTMW's SimCC peak values (0 .. ~8) -> a rough 0..1 confidence: min(1, v / 6)


def choose_variant(kp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """kp (F, 2, 2, K, 3) [plain, masked] -> (F, 2, K, 3) the variant with the higher mean body-joint score (the masked one on ties), and the choice (F, 2)."""
    s = np.nanmean(kp[:, :, :, :17, 2], axis=-1)                      # (F, 2, 2)
    s = np.where(np.isfinite(s), s, -1.0)
    pick = np.where(s[:, :, 0] > s[:, :, 1] + 0.05, 0, 1)             # 1 = masked unless plain is clearly better
    out = np.take_along_axis(kp, pick[:, :, None, None, None], axis=2)[:, :, 0]
    return out, pick


def anchors_from_skeletons(kp: np.ndarray, frames: list[int], thr: float = 4.0) -> list[dict]:
    """SAM prompts built from whole-body skeletons (kp (F, 2, K, 3), RTMW scores): per role the hull of its confident joints as the box, positive
    points on its own body joints, negative points on the partner's shoulders / hips."""
    out = []
    own = (5, 6, 7, 8, 11, 12, 13, 14, 15, 16)
    for f in frames:
        d: dict = {"frame": int(f), "tier": "skeleton"}
        ok = True
        for r, role in enumerate(("john", "lula")):
            k, p = kp[f, r], kp[f, 1 - r]
            good = np.flatnonzero(k[:17, 2] >= thr)
            if len(good) < 10:
                ok = False
                break
            xy = k[good, :2]
            x0, y0 = xy.min(0)
            x1, y1 = xy.max(0)
            w, h = x1 - x0, y1 - y0
            box = [x0 - 0.10 * w, y0 - 0.14 * h, x1 + 0.10 * w, y1 + 0.06 * h]
            pos = [[float(k[j, 0]), float(k[j, 1])] for j in own if k[j, 2] >= thr]
            neg = [[float(p[j, 0]), float(p[j, 1])] for j in (5, 6, 11, 12) if p[j, 2] >= thr]
            d[role] = {"box": [float(v) for v in box], "pos": pos, "neg": neg}
        if ok:
            out.append(d)
    return out


def review_final(fused_npz: Path, kp: np.ndarray, conf: np.ndarray, times: list[float], fps: float, out: Path, name: str, reads: list | None = None,
                 scale: float = 0.7, cols: int = 4, per_sheet: int = 8) -> list[Path]:
    """Sheets with the final identity: mask contour + skeleton (whole body when K >= 133) per dancer (John green, Lula magenta), conf label, bib reads."""
    import cv2
    out.mkdir(parents=True, exist_ok=True)
    Z = np.load(fused_npz)
    cap = cv2.VideoCapture(str(VIDEO))
    feet = [(15, 17), (15, 18), (15, 19), (16, 20), (16, 21), (16, 22)]
    tiles = []
    for t in times:
        f = min(max(int(round(t * fps)), 0), len(kp) - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
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
        for o, col in enumerate(((60, 220, 60), (220, 60, 220))):
            kk = kp[f, o]
            for a, b in SKEL + (feet if kk.shape[0] >= 133 else []):
                if kk[a, 2] >= 0.3 and kk[b, 2] >= 0.3:
                    cv2.line(img, (int(kk[a, 0]), int(kk[a, 1])), (int(kk[b, 0]), int(kk[b, 1])), (255, 255, 255), 3)
                    cv2.line(img, (int(kk[a, 0]), int(kk[a, 1])), (int(kk[b, 0]), int(kk[b, 1])), col, 1)
        img = cv2.resize(img, None, fx=scale, fy=scale)
        cv2.putText(img, f"t={f / fps:.2f}s f={f}  J {conf[f, 0]:.2f}  L {conf[f, 1]:.2f}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        tiles.append(img)
    cap.release()
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


# ------------------------------------------------------------------------------------------------ final tables


def per_frame_qa(fused: dict, kp: np.ndarray, size: tuple[int, int], fps: float) -> dict:
    """Per frame and dancer quality evidence from the masks and the skeletons. `fused` = the fused npz (box, agree, dist, area, m_<f>_<o>), kp (F, 2, K, 3) with
    scores already 0..1. Returns arrays: skel_in_partner (share of the confident torso joints that lie in the PARTNER's mask only), visible (own mask area /
    the local 90th percentile of it over +-2 s), edge (the mask touches the frame edge), box_iou (the two masks' boxes)."""
    import cv2
    F = len(kp)
    W, H = size
    S = scale_track(F, fps)
    area = fused["area"].astype(float) / (S[:, None] ** 2)           # the phone's zoom changes every apparent area (x4 at a 2x step)
    vis = np.ones((F, 2))
    for o in range(2):
        for f in range(F):
            seg = area[max(0, f - 60):f + 61, o]
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
            biou[f] = box_iou(box[f, 0], box[f, 1])
        for o in range(2):
            own, par = ms[o], ms[1 - o]
            if own is None or par is None:
                continue
            n = k_in = 0
            for j in (5, 6, 11, 12):
                x, y, c = kp[f, o, j]
                if c < 0.5 or not (0 <= int(x) < W and 0 <= int(y) < H):
                    continue
                n += 1
                k_in += int(par[int(y), int(x)] and not own[int(y), int(x)])
            sip[f, o] = k_in / n if n else 0.0
    return {"skel_in_partner": sip, "visible": vis, "edge": edge, "box_iou": biou}


def role_confidence(agree: np.ndarray, dist: np.ndarray, qa: dict, bib_dist: np.ndarray) -> np.ndarray:
    """(F, 2) how sure we are that the person marked John / Lula IS John / Lula: the agreement of the two SAM runs that reach the frame (1 at an anchor), halved
    when the skeleton sits on the partner's body, raised to 0.95 within 1 s (30 frames) of a verified bib read."""
    a = np.where(np.isfinite(agree), np.clip(agree, 0, 1), 0.9)
    a = np.where(dist == 0, 1.0, a)
    c = a * np.where(qa["skel_in_partner"] > 0.5, 0.5, 1.0)
    c = np.where(bib_dist <= 30, np.maximum(c, 0.95), c)
    return c


def pose_confidence(kp: np.ndarray, role_conf: np.ndarray, qa: dict) -> np.ndarray:
    """(F, 2) the downstream confidence of a dancer's skeleton: mean body-joint score x role confidence, x 0.35 when the skeleton lies on the partner (the dancer is
    hidden), x 0.6 when under 40 % of the usual mask is visible."""
    base = np.nanmean(kp[:, :, :17, 2], axis=-1)
    base = np.where(np.isfinite(base), base, 0.0)
    c = base * role_conf
    c = np.where(qa["skel_in_partner"] > 0.5, c * 0.35, c)
    c = np.where(qa["visible"] < 0.4, c * 0.6, c)
    return np.clip(c, 0, 1)


def bib_verification(P_reads: list, fused: dict, size: tuple[int, int]) -> dict:
    """Every exact bib read (conf >= 0.5) checked against the fused masks: which mask contains the read's centre. Returns counts and per-frame distance to the
    nearest read inside the right mask, per role."""
    import cv2
    W, H = size
    cnt: dict = {}
    hits = {0: [], 1: []}
    wrong = []
    for r in P_reads:
        t, c = r["text"], r["conf"]
        if c < 0.5:
            continue
        for tgt, who in (("124", 0), ("116", 1)):
            if bib_score(t, tgt, KNOWN_OTHER + (("116",) if who == 0 else ("124",))) < 0.9:
                continue
            f = int(r["frame"])
            cx, cy = np.array(r["quad"], float).mean(0)
            x, y = int(cx), int(cy)
            inside = []
            for o in range(2):
                k = f"m_{f}_{o}"
                m = cv2.imdecode(fused[k], cv2.IMREAD_GRAYSCALE) > 127 if k in fused else None
                inside.append(bool(m is not None and 0 <= x < W and 0 <= y < H and m[y, x]))
            tag = ("J" if inside[0] else "") + ("L" if inside[1] else "") or "none"
            cnt[f"{tgt}_in_{tag}"] = cnt.get(f"{tgt}_in_{tag}", 0) + 1
            if inside[who] and not inside[1 - who]:
                hits[who].append(f)
            elif inside[1 - who]:
                wrong.append({"frame": f, "bib": tgt, "in": tag})
    return {"counts": cnt, "hits": {k: sorted(set(v)) for k, v in hits.items()}, "wrong_mask": wrong}


def bib_distance(hits: dict, n_frames: int) -> np.ndarray:
    """(F, 2) frames to the nearest verified read of the dancer's own bib (9999 where none)."""
    out = np.full((n_frames, 2), 9999)
    fr = np.arange(n_frames)
    for o in (0, 1):
        h = np.array(hits[o], int)
        if len(h):
            out[:, o] = np.min(np.abs(fr[:, None] - h[None, :]), axis=1)
    return out


def pose_tables(P: dict, out: Path) -> dict:
    """The merged persons as the recap's per-frame tables (people/pose_tables/01.npz): kp (F, Pm, 17, 3), n (F,), box, det_conf, rgb (torso colour), persons ordered by
    detection confidence within a frame. Returns {"slot": (P,) the person's slot in its frame}."""
    F = P["n_frames"]
    n = np.diff(P["fstart"]).astype(np.int16)
    Pm = int(max(1, n.max()))
    kp = np.full((F, Pm, 17, 3), np.nan, np.float32)
    box = np.full((F, Pm, 4), np.nan, np.float32)
    dconf = np.zeros((F, Pm), np.float32)
    rgb = np.full((F, Pm, 3), np.nan, np.float32)
    slot = np.zeros(len(P["frame"]), int)
    for f in range(F):
        ids = np.arange(P["fstart"][f], P["fstart"][f + 1])
        for j, p in enumerate(ids[np.argsort(-P["conf"][ids], kind="stable")]):
            slot[p] = j
            kp[f, j], box[f, j], dconf[f, j], rgb[f, j] = P["kp"][p], P["box"][p], P["conf"][p], P["col"][p, 0]
    np.savez_compressed(out, kp=kp, n=n, box=box, det_conf=dconf, rgb=rgb)
    return {"slot": slot}


def match_rows(P: dict, slot: np.ndarray, box: np.ndarray, min_iou: float = 0.5) -> np.ndarray:
    """(F, 2) slot of the detected person whose box best matches the mask box of each dancer (IoU >= min_iou), else -1."""
    F = P["n_frames"]
    row = np.full((F, 2), -1, np.int16)
    for f in range(F):
        ids = range(P["fstart"][f], P["fstart"][f + 1])
        for o in range(2):
            if not np.isfinite(box[f, o]).all():
                continue
            best, bi = min_iou, -1
            for p in ids:
                v = box_iou(P["box"][p], box[f, o])
                if v > best:
                    best, bi = v, p
            if bi >= 0:
                row[f, o] = slot[bi]
    return row


def colour_check(fused: dict, n_frames: int, fps: float) -> dict:
    """Independent identity check: John wears a black t-shirt and trousers, Lula a pink crop top and grey trousers. Over the frames where both masks are large, the
    mean brightness of John's mask pixels must be below Lula's (and the pink of Lula's above his)."""
    import cv2
    cap = cv2.VideoCapture(str(VIDEO))
    n_ok = n_tot = 0
    dj, dl, red_j, red_l = [], [], [], []
    bad = []
    for f in range(n_frames):
        ok, img = cap.read()
        if not ok:
            break
        if f % 3:
            continue
        ms = []
        for o in range(2):
            k = f"m_{f}_{o}"
            ms.append(cv2.imdecode(fused[k], cv2.IMREAD_GRAYSCALE) > 127 if k in fused else None)
        if ms[0] is None or ms[1] is None or ms[0].sum() < 4000 or ms[1].sum() < 4000:
            continue
        vj, vl = img[ms[0]].mean(), img[ms[1]].mean()
        n_tot += 1
        if vj < vl:
            n_ok += 1
        else:
            bad.append(f)
        dj.append(vj)
        dl.append(vl)
    cap.release()
    return {"frames_checked": n_tot, "john_darker_share": round(n_ok / max(n_tot, 1), 4), "john_mean_brightness": round(float(np.mean(dj)), 1),
            "lula_mean_brightness": round(float(np.mean(dl)), 1), "swap_suspect_frames": bad[:60]}


def coverage_table(conf: np.ndarray, role_conf: np.ndarray, qa: dict, fps: float, thr: float = 0.5) -> list[dict]:
    """Per second: share of frames with a pose confidence >= thr per dancer, both, and the share flagged hidden / edge-cropped."""
    F = len(conf)
    rows = []
    for s in range(int(np.ceil(F / fps))):
        a, b = int(round(s * fps)), min(int(round((s + 1) * fps)), F)
        if a >= b:
            break
        c = conf[a:b] >= thr
        rows.append({"s": s, "john": round(float(c[:, 0].mean()), 2), "lula": round(float(c[:, 1].mean()), 2), "both": round(float(c.all(1).mean()), 2),
                     "role_conf_min": round(float(role_conf[a:b].min(1).mean()), 2), "hidden_john": round(float((qa["skel_in_partner"][a:b, 0] > 0.5).mean()), 2),
                     "hidden_lula": round(float((qa["skel_in_partner"][a:b, 1] > 0.5).mean()), 2),
                     "edge": round(float(qa["edge"][a:b].any(1).mean()), 2)})
    return rows


# ------------------------------------------------------------------------------------------------ stages (CLI)


def stage_anchors(first: bool = True, fused: Path | None = None, wb: Path | None = None, out: Path | None = None) -> Path:
    """Round 0: strict anchors from the detection-level tracker (+ relaxed ones in gaps). Round 1 (fused + wb given): keep the strict anchors, replace the relaxed
    ones inside the stretches where the two SAM runs disagreed by anchors built from the whole-body skeletons (anchors_from_skeletons)."""
    from scipy.ndimage import binary_dilation
    P = load()
    S = scale_track(P["n_frames"], P["fps"])
    if first:
        res = solve(P, iters=3)
        both = (res["ids"] >= 0).all(1)
        ref_h = float(np.median(P["height"][res["ids"][both].ravel()] / np.repeat(S[both], 2)))
        q = anchor_quality(P, res, ref_h, S)
        qr = anchor_quality(P, res, ref_h, S, p_min=0.6, joints_min=11, score_min=-1.0)
        frames = fill_gaps(pick_anchors(q), qr)
        doc = anchor_doc(P, res, frames)
        for d in doc:
            d["tier"] = "strict" if q[d["frame"]] > 0 else "relaxed"
        out = out or PEOPLE / "anchors0.json"
        out.write_text(json.dumps(doc))
        log(f"round 0: {len(doc)} anchors ({sum(d['tier'] == 'relaxed' for d in doc)} relaxed) -> {out}")
        return out
    Z = np.load(fused)
    ag = np.nan_to_num(Z["agree"], nan=1.0)
    hard = binary_dilation(ag.min(1) < 0.75, iterations=20)
    kp, _ = choose_variant(np.load(wb)["kp"])
    old = json.loads((PEOPLE / "anchors0.json").read_text())
    keep = [a for a in old if not (hard[a["frame"]] and a.get("tier") == "relaxed")]
    cand = []
    for f in range(0, P["n_frames"], 10):
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
    doc = sorted(keep + anchors_from_skeletons(kp, cand), key=lambda a: a["frame"])
    final: list[dict] = []
    for a in doc:
        if final and a["frame"] - final[-1]["frame"] < 6:
            continue
        final.append(a)
    out = out or PEOPLE / "anchors1.json"
    out.write_text(json.dumps(final))
    log(f"round 1: {len(keep)} kept + {len(final) - len(keep)} skeleton anchors -> {out}")
    return out


def finalize(sam_tag: str = "1", wb_tag: str = "1") -> dict:
    """Everything the downstream stages read: people/teachers_01.npz (the recap contract + whole-body keypoints, masks reference, QA arrays), people/masks_01.npz,
    people/pose_tables/01.npz, people/teachers.json, verification.json, coverage.json."""
    from .cameras.newcam import video_pts
    P = load()
    F = P["n_frames"]
    size = (int(P["size"][0]), int(P["size"][1]))
    fused = dict(np.load(PEOPLE / f"fused{sam_tag}.npz"))
    kpw, pick = choose_variant(np.load(IN / f"pose_wb{wb_tag}.npz")["kp"])
    kpw = kpw.copy()
    kpw[..., 2] = np.clip(kpw[..., 2] / RTMW_SCALE, 0.0, 1.0)
    kpw[~np.isfinite(fused["box"]).all(-1)] = np.nan
    qa = per_frame_qa(fused, kpw, size, P["fps"])
    ocr = json.loads((IN / "ocr.json").read_text())["reads"]
    bv = bib_verification(ocr, fused, size)
    bd = bib_distance(bv["hits"], F)
    rconf = role_confidence(fused["agree"], fused["dist"], qa, bd)
    pconf = pose_confidence(kpw, rconf, qa)
    (PEOPLE / "pose_tables").mkdir(parents=True, exist_ok=True)
    pt = pose_tables(P, PEOPLE / "pose_tables" / "01.npz")
    row = match_rows(P, pt["slot"], fused["box"])
    pts = video_pts(VIDEO, WORK / "frames" / "video_pts.json")[:F]
    merged = qa["box_iou"] > 0.5
    teachers = {"t": pts.astype(np.float64), "frame_index": np.arange(F, dtype=np.int32), "ok": np.ones(F, bool), "row": row, "box": fused["box"].astype(np.float32),
                "kp": kpw[:, :, :17].astype(np.float32), "conf": pconf.astype(np.float32), "merged": np.repeat(merged[:, None], 2, axis=1),
                "role_conf": rconf.astype(np.float32), "floor": np.full((F, 2, 2), np.nan, np.float32), "size": np.array(size, np.int32), "attached": np.array(False),
                # additions (J3a): whole-body keypoints (COCO-WholeBody 133: 17 body, 6 feet, 68 face, 42 hands; scores 0..1 = min(1, RTMW peak / 6)), which variant
                # (0 plain, 1 partner-masked) each frame uses, the per-frame QA arrays, and the bib evidence
                "kp_wb": kpw.astype(np.float32), "wb_variant": pick.astype(np.int8), "mask_area": fused["area"], "mask_agree": fused["agree"],
                "mask_dist": fused["dist"], "visible": qa["visible"].astype(np.float32), "skel_in_partner": qa["skel_in_partner"].astype(np.float32),
                "edge": qa["edge"], "box_iou": qa["box_iou"].astype(np.float32), "bib_dist": bd.astype(np.int32)}
    np.savez_compressed(PEOPLE / "teachers_01.npz", **teachers)
    masks = {k: v for k, v in fused.items() if k.startswith("m_")}
    np.savez(PEOPLE / "masks_01.npz", box=fused["box"], area=fused["area"], **masks)
    cov = coverage_table(pconf, rconf, qa, P["fps"])
    (PEOPLE / "coverage.json").write_text(json.dumps(cov, indent=1))
    col = colour_check(fused, F, P["fps"])
    ver = {"bib": {"counts": bv["counts"], "reads_in_wrong_mask": bv["wrong_mask"], "verified_frames": {"john": len(bv["hits"][0]), "lula": len(bv["hits"][1])}},
           "colour": col, "mask_agree_lt_0.6_frames": [int((np.nan_to_num(fused["agree"][:, o], nan=1.0) < 0.6).sum()) for o in range(2)],
           "frames_within_30_of_a_verified_bib": [round(float((bd[:, o] <= 30).mean()), 3) for o in range(2)]}
    (PEOPLE / "verification.json").write_text(json.dumps(ver, indent=1))
    summ = {"frames": F, "t": [round(float(pts[0]), 3), round(float(pts[-1]), 3)], "mask_share": [round(float((fused["area"][:, o] > 50).mean()), 3) for o in range(2)],
            "conf_ge_0.5": [round(float((pconf[:, o] >= 0.5).mean()), 3) for o in range(2)], "both_ge_0.5": round(float((pconf >= 0.5).all(1).mean()), 3),
            "role_conf_ge_0.9": [round(float((rconf[:, o] >= 0.9).mean()), 3) for o in range(2)], "role_conf_lt_0.5": [round(float((rconf[:, o] < 0.5).mean()), 3) for o in range(2)],
            "hidden_share": [round(float((qa["skel_in_partner"][:, o] > 0.5).mean()), 3) for o in range(2)], "edge_share": round(float(qa["edge"].any(1).mean()), 3),
            "merged_share": round(float(merged.mean()), 3), "matched_detection_share": [round(float((row[:, o] >= 0).mean()), 3) for o in range(2)],
            "variant_masked_share": round(float(pick.mean()), 3), "sam_round": sam_tag}
    (PEOPLE / "teachers.json").write_text(json.dumps(summ, indent=1))
    log(json.dumps(summ))
    log(json.dumps(ver))
    return summ


def review_times() -> list[float]:
    """The review set: >= 40 instants incl. the zoom steps, the 29-31 s occlusion, the bib-readable frames and the weak stretches."""
    return [0.5, 2, 5, 8, 11, 14, 17, 17.9, 18.2, 18.6, 19, 20, 21, 22, 23, 24, 26, 28, 29, 30, 31, 32, 33, 35, 37, 38, 40, 42, 44, 46, 48, 50, 51.4, 51.7, 52.0, 52.5, 55, 60,
            66, 72, 78, 78.9, 79.2, 79.6, 80, 82, 84, 85, 86, 87, 88]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.jl_people", description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("anchors0", "anchors1", "masks", "finalize", "review"))
    ap.add_argument("--sam", default="sam1.npz", help="the pod's SAM output in people/in/ (masks)")
    ap.add_argument("--out", default="fused1.npz", help="the fused masks file in people/ (masks)")
    ap.add_argument("--tag", default="1", help="SAM / pose round (finalize, review)")
    a = ap.parse_args(argv)
    if a.cmd == "anchors0":
        stage_anchors(True)
    elif a.cmd == "anchors1":
        stage_anchors(False, PEOPLE / "fused0.npz", IN / "pose_wb0.npz")
    elif a.cmd == "masks":
        M = MaskStore(IN / a.sam)
        log(json.dumps(export_fused(M, int(np.load(IN / "pose.npz")["n_frames"]), PEOPLE / a.out)))
    elif a.cmd == "finalize":
        finalize(a.tag, a.tag)
    else:
        z = np.load(PEOPLE / "teachers_01.npz")
        out = WORK / "review" / "people"
        for p in review_final(PEOPLE / f"fused{a.tag}.npz", z["kp_wb"], z["conf"], review_times(), 29.994, out, "final"):
            log(f"review {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
