"""Central-pair candidates: which two of a camera's ~10-25 YOLO person boxes per frame are the dancers.

Every box gets a unary score from cues that separate the couple from ~40 spectators and from reflections:
  size        dancers stand and are the largest *untruncated* people near the middle of the shot
  bottom      near-camera seated spectators are cut by the bottom border (cam07), dancers' feet are not
  centrality  every phone operator keeps the couple roughly centred horizontally
  standing    visible ankles (COCO 15/16) + tall aspect ratio
  motion      dancers move; seated spectators barely do (box displacement over +-0.5 s)
Reflections are rejected outright: tiny boxes (phone screens held by spectators, cam04) and boxes nested
almost completely inside a much larger box (mirror/screen copies, cam09).

The pair is then chosen jointly over time with a Viterbi over ORDERED pair states (slot A, slot B, each a
candidate box or "missing"), so the output is two continuous candidate tracks A/B per camera. Identity
(lead/follow) is not decided here - only "these two tracks are the couple".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .detect import iou_matrix

MISS = -1


@dataclass
class SelectConfig:
    min_height: float = 0.08  # boxes shorter than this x image height are never dancers (screens, far crowd)
    min_score: float = 0.25
    contain_frac: float = 0.92  # a box this much inside another ...
    contain_area: float = 0.25  # ... and this much smaller is a reflection / screen copy
    bottom_margin: float = 0.03  # box bottom within this x H of the border = truncated
    max_candidates: int = 7  # per frame, by unary score
    motion_lag_s: float = 0.5
    w_size: float = 3.0
    w_center: float = 2.5
    w_bottom: float = 2.5
    w_stand: float = 0.8
    w_motion: float = 1.2
    w_score: float = 1.0
    miss_cost: float = 2.5  # unary cost of an empty slot (embrace: one merged box)
    dup_iou: float = 0.75  # two slots on one person
    w_cont: float = 4.0  # continuity: 1 - IoU with the slot's previous box
    w_jump: float = 2.0  # centre jump in box heights
    switch_miss: float = 1.5  # entering / leaving the missing state


def reflection_mask(boxes: np.ndarray, height: int, cfg: SelectConfig) -> np.ndarray:
    """True for boxes that are reflections or in-hand screens (see module doc)."""
    b = np.asarray(boxes, float).reshape(-1, 4)
    h = b[:, 3] - b[:, 1]
    area = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(h, 0, None)
    bad = h < cfg.min_height * height
    if len(b) > 1:
        lt = np.maximum(b[:, None, :2], b[None, :, :2])
        rb = np.minimum(b[:, None, 2:], b[None, :, 2:])
        inter = np.prod(np.clip(rb - lt, 0, None), axis=2)
        inside = inter / np.maximum(area[:, None], 1e-9)  # fraction of box i inside box j
        small = area[:, None] < cfg.contain_area * area[None, :]
        np.fill_diagonal(inside, 0)
        bad |= ((inside >= cfg.contain_frac) & small).any(1)
    return bad


def motion_score(boxes: np.ndarray, valid: np.ndarray, lag: int) -> np.ndarray:
    """(F, N) displacement (in box heights) of each box to its best-IoU match `lag` frames away (max of +-)."""
    f, n = valid.shape
    out = np.zeros((f, n))
    for i in range(f):
        vi = np.flatnonzero(valid[i])
        if not len(vi):
            continue
        b = boxes[i, vi]
        h = np.maximum(b[:, 3] - b[:, 1], 1.0)
        c = (b[:, :2] + b[:, 2:]) / 2
        best = np.zeros(len(vi))
        for j in (i - lag, i + lag):
            if j < 0 or j >= f or not valid[j].any():
                continue
            bj = boxes[j, valid[j]]
            iou = iou_matrix(b, bj)
            k = iou.argmax(1)
            cj = (bj[k, :2] + bj[k, 2:]) / 2
            d = np.linalg.norm(c - cj, axis=1) / h + (1 - iou.max(1))
            best = np.maximum(best, d)
        out[i, vi] = best
    return out


def unary_scores(boxes: np.ndarray, scores: np.ndarray, kpts: np.ndarray, motion: np.ndarray,
                 width: int, height: int, cfg: SelectConfig) -> np.ndarray:
    """Per-box dancer-likeness for one frame (higher = more likely a dancer)."""
    b = np.asarray(boxes, float).reshape(-1, 4)
    h = np.clip((b[:, 3] - b[:, 1]) / height, 1e-3, None)
    w = np.clip((b[:, 2] - b[:, 0]) / width, 1e-3, None)
    cx = (b[:, 0] + b[:, 2]) / 2 / width
    size = np.log(h / 0.3)
    center = -np.abs(cx - 0.5) * 2
    trunc = (b[:, 3] >= (1 - cfg.bottom_margin) * height).astype(float)
    ankles = (kpts[:, 15:17, 2] > 0.3).any(1).astype(float) if len(kpts) else np.zeros(len(b))
    aspect = np.clip((h * height) / np.maximum(w * width, 1) - 1.3, -1, 1)
    stand = ankles + 0.5 * aspect
    mot = np.clip(motion, 0, 0.6) / 0.6
    return (cfg.w_size * size + cfg.w_center * center - cfg.w_bottom * trunc + cfg.w_stand * stand
            + cfg.w_motion * mot + cfg.w_score * scores)


def _pair_cost(ba: np.ndarray | None, bb: np.ndarray | None, cfg: SelectConfig) -> float:
    """Extra cost of putting two boxes in the two slots: duplicates and implausible partners."""
    if ba is None or bb is None:
        return 0.0
    iou = iou_matrix(ba[None], bb[None])[0, 0]
    ha, hb = ba[3] - ba[1], bb[3] - bb[1]
    ratio = max(ha, hb) / max(min(ha, hb), 1.0)
    gap = abs((ba[0] + ba[2]) - (bb[0] + bb[2])) / 2 / max(ha, hb)
    cost = 0.0
    if iou > cfg.dup_iou:
        cost += 6.0
    cost += 2.0 * max(0.0, ratio - 1.4)  # partners are a similar height in the image
    cost += 2.0 * max(0.0, gap - 1.0)  # and within about one body height of each other
    foot = abs(ba[3] - bb[3]) / max(ha, hb)
    cost += 4.0 * max(0.0, foot - 0.2)  # standing on the same floor: feet at a similar image height
    return cost


def _trans_cost(prev: np.ndarray | None, cur: np.ndarray | None, cfg: SelectConfig) -> float:
    if prev is None and cur is None:
        return 0.0
    if prev is None or cur is None:
        return cfg.switch_miss
    iou = iou_matrix(prev[None], cur[None])[0, 0]
    h = max(prev[3] - prev[1], cur[3] - cur[1], 1.0)
    jump = np.linalg.norm((prev[:2] + prev[2:]) / 2 - (cur[:2] + cur[2:]) / 2) / h
    return cfg.w_cont * (1 - iou) + cfg.w_jump * max(0.0, jump - 0.15)


def candidate_tracks(boxes: np.ndarray, scores: np.ndarray, valid: np.ndarray, kpts: np.ndarray,
                     width: int, height: int, fps: float, cfg: SelectConfig | None = None) -> dict:
    """Two continuous candidate tracks from people/<cam>.npz arrays.

    Returns idx (F, 2) detection index per slot (MISS = -1), boxes (F, 2, 4) (NaN when missing),
    unary (F, N) box scores, rejected (F, N) reflection flags.
    """
    cfg = cfg or SelectConfig()
    f, n = valid.shape
    motion = motion_score(boxes, valid, max(1, int(round(cfg.motion_lag_s * fps))))
    unary = np.full((f, n), -np.inf)
    rejected = np.zeros((f, n), bool)
    cands: list[list[int]] = []
    for i in range(f):
        vi = np.flatnonzero(valid[i] & (scores[i] >= cfg.min_score))
        if len(vi):
            rej = reflection_mask(boxes[i, vi], height, cfg)
            rejected[i, vi] = rej
            u = unary_scores(boxes[i, vi], scores[i, vi], kpts[i, vi], motion[i, vi], width, height, cfg)
            unary[i, vi[~rej]] = u[~rej]
        ok = np.flatnonzero(np.isfinite(unary[i]))
        cands.append([int(k) for k in ok[np.argsort(-unary[i, ok])][:cfg.max_candidates]])

    lists = [c + [MISS] for c in cands]  # slot options per frame
    pos = [(np.array([a for a in range(len(c)) for b in range(len(c)) if a != b or c[a] == MISS]),
            np.array([b for a in range(len(c)) for b in range(len(c)) if a != b or c[a] == MISS]))
           for c in lists]

    def box(i, k):
        return None if k == MISS else boxes[i, k]

    def node_costs(i):
        c = lists[i]
        u = np.array([cfg.miss_cost if k == MISS else -unary[i, k] for k in c])
        sa, sb = pos[i]
        pc = np.array([_pair_cost(box(i, c[a]), box(i, c[b]), cfg) for a, b in zip(sa, sb)])
        return u[sa] + u[sb] + pc

    cost = node_costs(0)
    back: list[np.ndarray] = []
    for i in range(1, f):
        cp, cc = lists[i - 1], lists[i]
        slot_t = np.array([[_trans_cost(box(i - 1, a), box(i, b), cfg) for b in cc] for a in cp])
        (pa, pb), (qa, qb) = pos[i - 1], pos[i]
        tot = cost[:, None] + slot_t[pa[:, None], qa[None, :]] + slot_t[pb[:, None], qb[None, :]]
        arg = tot.argmin(0)
        back.append(arg)
        cost = tot[arg, np.arange(len(qa))] + node_costs(i)
    path = [int(cost.argmin())]
    for i in range(f - 1, 0, -1):
        path.append(int(back[i - 1][path[-1]]))
    path.reverse()
    st = [[(lists[i][a], lists[i][b]) for a, b in zip(*pos[i])] for i in range(f)]
    idx = np.array([st[i][p] for i, p in enumerate(path)], int)
    out_boxes = np.full((f, 2, 4), np.nan, np.float32)
    for i in range(f):
        for s in range(2):
            if idx[i, s] != MISS:
                out_boxes[i, s] = boxes[i, idx[i, s]]
    return {"idx": idx, "boxes": out_boxes, "unary": unary, "rejected": rejected, "motion": motion}
