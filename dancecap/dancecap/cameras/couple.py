"""Pick the dancing couple out of every person detection of a phone video (people2d.py output): the leader in a dark torso, the follower the
person he embraces. A heuristic with a temporal check, meant for calibration (a few thousand correct frames are enough), not for identity.

    pick(frame, box, rgb, kp, n_frames, dark_luma) -> per frame (index of the lead detection, index of the follow detection) or -1
"""
from __future__ import annotations

import numpy as np


def luma(rgb: np.ndarray) -> np.ndarray:
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def overlap(a: np.ndarray, b: np.ndarray) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    ar = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return ix * iy / max(ar, 1.0)


def pick(frame: np.ndarray, box: np.ndarray, rgb: np.ndarray, n_frames: int, dark_luma: float = 60.0, min_overlap: float = 0.08,
         pairs_out: bool = False):
    lead = np.full(n_frames, -1, int)
    foll = np.full(n_frames, -1, int)
    score = np.zeros(n_frames)
    order = np.argsort(frame, kind="stable")
    starts = np.searchsorted(frame[order], np.arange(n_frames + 1))
    lum = luma(np.nan_to_num(rgb, nan=255.0))
    area = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
    for t in range(n_frames):
        ids = order[starts[t]:starts[t + 1]]
        best = (0.0, -1, -1)
        for d in ids:
            if lum[d] > dark_luma:
                continue
            for l in ids:
                if l == d or lum[l] <= dark_luma:
                    continue
                ov = overlap(box[d], box[l])
                if ov < min_overlap:
                    continue
                s = ov * np.sqrt(area[d] * area[l])
                if s > best[0]:
                    best = (s, d, l)
        score[t], lead[t], foll[t] = best
    return lead, foll, score


def candidates(frame: np.ndarray, box: np.ndarray, rgb: np.ndarray, n_frames: int, k: int = 5, dark_luma: float = 60.0, min_overlap: float = 0.08):
    """Per frame the k best (score, dark person, partner) pairs of detections: a dark torso overlapping a lighter one (score = overlap x geometric mean of the areas)."""
    order = np.argsort(frame, kind="stable")
    starts = np.searchsorted(frame[order], np.arange(n_frames + 1))
    lum = luma(np.nan_to_num(rgb, nan=255.0))
    area = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
    out = []
    for t in range(n_frames):
        ids = order[starts[t]:starts[t + 1]]
        c = []
        for d in ids:
            if lum[d] > dark_luma:
                continue
            for l in ids:
                if l == d or lum[l] <= dark_luma:
                    continue
                ov = overlap(box[d], box[l])
                if ov >= min_overlap:
                    c.append((float(ov * np.sqrt(area[d] * area[l])), int(d), int(l)))
        c.sort(reverse=True)
        out.append(c[:k])
    return out


def track_pairs(cands: list, centre: np.ndarray, redness: np.ndarray | None = None, s0: float = 1e4, lam: float = 2.0, cap_m: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    """Dynamic programming over the candidate pairs of all frames: unary = -log(score / s0) (big pairs preferred) minus a bonus for a reddish partner,
    transition = lam x min(distance of the pair's floor position to the previous pair's, cap_m) (the couple moves continuously, the other couples are elsewhere).
    centre: (n_det, 2) floor position (m) of every detection. Returns (lead index, follow index) per frame (-1 where the frame has no candidate)."""
    n = len(cands)
    lead = np.full(n, -1, int)
    foll = np.full(n, -1, int)
    prev_cost = None
    prev_pos = None
    back = []
    frames = []
    for t in range(n):
        if not cands[t]:
            continue
        cur_cost, cur_pos, bp = [], [], []
        for (s, d, l) in cands[t]:
            pos = 0.5 * (centre[d] + centre[l])
            u = -np.log(max(s, 1.0) / s0) - (0.5 if redness is not None and redness[l] else 0.0)
            if prev_cost is None:
                cur_cost.append(u); bp.append(-1)
            else:
                tr = [prev_cost[j] + lam * min(float(np.linalg.norm(pos - prev_pos[j])), cap_m) for j in range(len(prev_cost))]
                j = int(np.argmin(tr))
                cur_cost.append(u + tr[j]); bp.append(j)
            cur_pos.append(pos)
        back.append(bp)
        frames.append(t)
        prev_cost, prev_pos = cur_cost, cur_pos
    if not frames:
        return lead, foll
    j = int(np.argmin(prev_cost))
    for k in range(len(frames) - 1, -1, -1):
        t = frames[k]
        _, d, l = cands[t][j]
        lead[t], foll[t] = d, l
        j = back[k][j]
    return lead, foll
