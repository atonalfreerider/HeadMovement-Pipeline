"""Measured colour descriptors of a dancer candidate and the lead/follow colour rule.

Only clothing/hair colours the user described are measured (lead: black t-shirt, dark jeans, shaved head;
follow: white crop top, light ripped jeans, long red hair). Nothing about the person beyond those colours is
inferred. Regions come from the COCO-17 keypoints when the pose model saw the person, else from fixed box
fractions; when a pixel mask is known (SAM 2 output) the regions are intersected with it.

    torso   shoulders -> 45 % of the way to the hips (the crop top ends above the waist)
    head    a box around nose/eyes/ears extended up and sideways (where long hair is)
    legs    hips -> ankles
Descriptors (HSV, OpenCV ranges H 0-180, S/V 0-255): torso_v, torso_s, leg_v, leg_s, red (fraction of
saturated red-hue pixels in the head region and above the shoulders), plus the pixel counts.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

KP_CONF = 0.3
NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SH, R_SH, L_HIP, R_HIP, L_ANK, R_ANK = 0, 1, 2, 3, 4, 5, 6, 11, 12, 15, 16


@dataclass
class RuleConfig:
    torso_v_scale: float = 45.0  # brightness gap in V units that counts as one unit of evidence
    leg_v_scale: float = 50.0
    red_scale: float = 0.08  # red-hair fraction gap per unit of evidence
    w_torso: float = 1.0
    w_leg: float = 0.5
    w_red: float = 1.0
    clip: float = 4.0  # per-cue cap, so one cue never dominates completely
    min_px: int = 30  # fewer measured pixels than this -> the cue abstains


def red_hair_mask(hsv: np.ndarray) -> np.ndarray:
    """Dyed red hair: red/orange-red hue, clearly saturated, not too dark. Skin is less saturated."""
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return (((h <= 6) | (h >= 170)) & (s >= 130) & (v >= 45))


def _pt(k, j):
    return k[j, :2] if k[j, 2] > KP_CONF else None


def regions(box: np.ndarray, kpts: np.ndarray | None, shape: tuple[int, int]) -> dict[str, np.ndarray]:
    """Boolean (H, W) masks for torso / head / legs of one candidate (box xyxy, kpts (17, 3) or None)."""
    hh, ww = shape
    x0, y0, x1, y1 = [float(v) for v in box]
    bw, bh = x1 - x0, y1 - y0
    out = {k: np.zeros((hh, ww), np.uint8) for k in ("torso", "head", "legs")}
    k = kpts if kpts is not None and kpts.shape == (17, 3) and (kpts[:, 2] > KP_CONF).sum() >= 5 else None
    sh = [p for p in (_pt(k, L_SH), _pt(k, R_SH))] if k is not None else [None, None]
    hp = [p for p in (_pt(k, L_HIP), _pt(k, R_HIP))] if k is not None else [None, None]
    if k is not None and all(p is not None for p in sh + hp):
        ls, rs, lh, rh = sh[0], sh[1], hp[0], hp[1]
        lm, rm = ls + 0.45 * (lh - ls), rs + 0.45 * (rh - rs)
        cv2.fillConvexPoly(out["torso"], np.round([ls, rs, rm, lm]).astype(np.int32), 1)
        width = max(np.linalg.norm(ls - rs), 0.15 * bw)
        anks = [p for p in (_pt(k, L_ANK), _pt(k, R_ANK)) if p is not None]
        foot_y = max([p[1] for p in anks], default=y1 - 0.03 * bh)
        hc = (lh + rh) / 2
        cv2.rectangle(out["legs"], (int(hc[0] - 0.9 * width), int(hc[1] + 0.05 * bh)),
                      (int(hc[0] + 0.9 * width), int(foot_y)), 1, -1)
        face = [p for p in (_pt(k, j) for j in (NOSE, L_EYE, R_EYE, L_EAR, R_EAR)) if p is not None]
        top_sh = min(ls[1], rs[1])
        if face:
            fc = np.mean(face, 0)
            cv2.rectangle(out["head"], (int(fc[0] - 1.1 * width), int(fc[1] - 0.8 * width)),
                          (int(fc[0] + 1.1 * width), int(top_sh + 0.35 * width)), 1, -1)
        else:
            cx = (ls[0] + rs[0]) / 2
            cv2.rectangle(out["head"], (int(cx - 1.1 * width), int(top_sh - 1.2 * width)),
                          (int(cx + 1.1 * width), int(top_sh + 0.35 * width)), 1, -1)
    else:  # box fractions (standing person)
        cx = (x0 + x1) / 2
        cv2.rectangle(out["torso"], (int(cx - 0.18 * bw), int(y0 + 0.2 * bh)), (int(cx + 0.18 * bw), int(y0 + 0.36 * bh)), 1, -1)
        cv2.rectangle(out["head"], (int(x0 + 0.1 * bw), int(y0)), (int(x1 - 0.1 * bw), int(y0 + 0.22 * bh)), 1, -1)
        cv2.rectangle(out["legs"], (int(cx - 0.25 * bw), int(y0 + 0.58 * bh)), (int(cx + 0.25 * bw), int(y0 + 0.9 * bh)), 1, -1)
    clip = np.zeros((hh, ww), np.uint8)
    cv2.rectangle(clip, (int(x0 - 0.15 * bw), int(y0 - 0.1 * bh)), (int(x1 + 0.15 * bw), int(y1)), 1, -1)
    return {n: (m & clip).astype(bool) for n, m in out.items()}


def descriptors(hsv: np.ndarray, box: np.ndarray, kpts: np.ndarray | None,
                mask: np.ndarray | None = None, exclude: np.ndarray | None = None) -> dict[str, float]:
    """Colour descriptors of one candidate. mask: its own pixels (optional); exclude: the partner's pixels."""
    reg = regions(box, kpts, hsv.shape[:2])
    out: dict[str, float] = {}
    for name, r in reg.items():
        if mask is not None:
            r = r & mask
        if exclude is not None:
            r = r & ~exclude
        px = hsv[r]
        out[f"{name}_n"] = float(len(px))
        if name == "head":
            out["red"] = float(red_hair_mask(px[None])[0].mean()) if len(px) else float("nan")
        else:
            key = "torso" if name == "torso" else "leg"
            out[f"{key}_v"] = float(np.median(px[:, 2])) if len(px) else float("nan")
            out[f"{key}_s"] = float(np.median(px[:, 1])) if len(px) else float("nan")
    return out


def lead_llr(a: dict, b: dict, cfg: RuleConfig | None = None) -> tuple[float, dict]:
    """Log-odds that candidate a is the LEAD and b the FOLLOW (negative: the other way round).

    Cues (each clipped, abstaining when either side has too few pixels): the follow's top is brighter, her
    jeans are lighter, and she has red hair."""
    cfg = cfg or RuleConfig()
    cues = {}
    if min(a["torso_n"], b["torso_n"]) >= cfg.min_px:
        cues["torso"] = cfg.w_torso * np.clip((b["torso_v"] - a["torso_v"]) / cfg.torso_v_scale, -cfg.clip, cfg.clip)
    if min(a["legs_n"], b["legs_n"]) >= cfg.min_px:
        cues["legs"] = cfg.w_leg * np.clip((b["leg_v"] - a["leg_v"]) / cfg.leg_v_scale, -cfg.clip, cfg.clip)
    if min(a["head_n"], b["head_n"]) >= cfg.min_px:
        cues["red"] = cfg.w_red * np.clip((b["red"] - a["red"]) / cfg.red_scale, -cfg.clip, cfg.clip)
    return float(sum(cues.values())), {k: float(v) for k, v in cues.items()}


def colour_names(d: dict) -> dict[str, str]:
    """Plain-language colour names from measured descriptors (for the review sheet / TypeSafe state)."""
    def bright(v):
        if not np.isfinite(v):
            return "unknown"
        return "black" if v < 70 else "dark" if v < 110 else "medium" if v < 150 else "light" if v < 190 else "white"
    red = d.get("red", float("nan"))
    hair = "unknown" if not np.isfinite(red) else "strong red" if red > 0.15 else "some red" if red > 0.05 else "no red"
    return {"top": bright(d.get("torso_v", float("nan"))), "trousers": bright(d.get("leg_v", float("nan"))),
            "red_above_shoulders": hair}
