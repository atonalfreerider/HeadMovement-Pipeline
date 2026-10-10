"""The dancers of a two-camera take from the 2D detections of every person (dancecap.cameras.people2d output).

people2d returns everybody in the picture (judges, audience, the hostess). The two dancers are the two LARGEST boxes of a frame
(they stand nearest the cameras; the judges sit at the far wall) and are told apart by the torso brightness: the white shirt
(Carlos) is brighter than the red top (Aline) in both cameras, also in the dark grade of camera 01. Wrong picks are rare and
are removed by the geometry later (robust fit), never by hand.

dancer_table() returns arrays over ALL frames of a video; interp() reads them at fractional frame numbers (the two cameras
have different frame rates: 23.976 and 30 fps), interpolating the keypoints linearly between the two neighbouring frames when
both have the dancer.
"""
from __future__ import annotations

import numpy as np

CARLOS, ALINE = 0, 1


def lum(rgb: np.ndarray) -> np.ndarray:
    return 0.3 * rgb[..., 0] + 0.59 * rgb[..., 1] + 0.11 * rgb[..., 2]


def dancer_table(det: dict, n_frames: int, h_min: float, h_ratio: float = 0.55, min_conf: float = 0.3) -> dict:
    """{'kp': (N,2,17,3) NaN where absent, 'box': (N,2,4), 'ok': (N,2)} for the frames 0..N-1.

    A detection counts as a dancer when its box is >= h_min px high and >= h_ratio x the tallest box of the frame; the two
    tallest such boxes are the dancers; brighter torso = Carlos. A frame with exactly one dancer box keeps it only when its
    torso brightness is decisive (< 0.6 or > 1.6 x the median torso brightness of the two-dancer frames' Aline / Carlos)."""
    fr, box, kp, rgb, cf = det["frame"], det["box"], det["kp"], det["rgb"], det["conf"]
    h = box[:, 3] - box[:, 1]
    L = lum(np.where(np.isnan(rgb), 0, rgb))
    out_kp = np.full((n_frames, 2, 17, 3), np.nan, np.float32)
    out_box = np.full((n_frames, 2, 4), np.nan, np.float32)
    order = np.argsort(fr, kind="stable")
    uf, start = np.unique(fr[order], return_index=True)
    end = np.r_[start[1:], len(order)]
    singles = []
    lum_pair = []
    for f, s0, s1 in zip(uf, start, end):
        idx = order[s0:s1]
        idx = idx[(h[idx] >= h_min) & (cf[idx] >= min_conf)]
        if len(idx) == 0:
            continue
        idx = idx[h[idx] >= h_ratio * h[idx].max()]
        idx = idx[np.argsort(-h[idx])][:2]
        if len(idx) == 2:
            a, b = idx
            c, al = (a, b) if L[a] >= L[b] else (b, a)
            out_kp[f, CARLOS], out_box[f, CARLOS] = kp[c], box[c]
            out_kp[f, ALINE], out_box[f, ALINE] = kp[al], box[al]
            lum_pair.append((L[c], L[al]))
        elif len(idx) == 1:
            singles.append((f, idx[0]))
    if singles and lum_pair:
        lp = np.array(lum_pair)
        mc, ma = np.median(lp[:, 0]), np.median(lp[:, 1])
        mid = 0.5 * (mc + ma)
        for f, k in singles:
            if L[k] > mid + 0.35 * (mc - mid):
                out_kp[f, CARLOS], out_box[f, CARLOS] = kp[k], box[k]
            elif L[k] < mid - 0.35 * (mid - ma):
                out_kp[f, ALINE], out_box[f, ALINE] = kp[k], box[k]
    return {"kp": out_kp, "box": out_box, "ok": ~np.isnan(out_kp[:, :, 0, 0])}


def interp_kp(table: dict, frame_f: np.ndarray, who: int) -> np.ndarray:
    """Keypoints (M,17,3) of dancer `who` at fractional frames; NaN when either neighbouring frame lacks the dancer.
    The score is the minimum of the two neighbours'."""
    kp = table["kp"][:, who]
    n = len(kp)
    f0 = np.clip(np.floor(frame_f).astype(int), 0, n - 1)
    f1 = np.clip(f0 + 1, 0, n - 1)
    u = (frame_f - f0)[:, None, None]
    a, b = kp[f0], kp[f1]
    out = a * (1 - u) + b * u
    out[..., 2] = np.minimum(a[..., 2], b[..., 2])
    # exactly on a frame: the neighbour need not exist
    on = (frame_f - f0) < 1e-6
    out[on] = a[on]
    return out
