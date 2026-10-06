"""Visual QA for the dancers stage: tinted overlays (lead red, follow blue, follow hair yellow outline),
overlay videos and contact sheets."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

LEAD = (60, 60, 235)  # BGR red (matches review.py)
FOLLOW = (235, 140, 40)  # BGR blue
HAIR = (40, 220, 240)  # BGR yellow


def label_text(img, text, org, color, scale=0.8):
    th = max(1, int(round(2 * scale)))
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), th + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, th, cv2.LINE_AA)


def overlay(img: np.ndarray, lead: np.ndarray, follow: np.ndarray, hair: np.ndarray | None = None,
            height: int = 960, title: str = "") -> np.ndarray:
    """Tinted masks + role tags, resized to `height`."""
    vis = img.copy()
    for m, col in ((lead, LEAD), (follow, FOLLOW)):
        if m.any():
            vis[m] = (0.5 * vis[m] + 0.5 * np.array(col)).astype(np.uint8)
    s = height / img.shape[0]
    vis = cv2.resize(vis, (int(round(img.shape[1] * s)), height), interpolation=cv2.INTER_AREA)
    for m, col, tag in ((lead, LEAD, "LEAD"), (follow, FOLLOW, "FOLLOW")):
        if not m.any():
            continue
        small = cv2.resize(m.astype(np.uint8), (vis.shape[1], vis.shape[0]), interpolation=cv2.INTER_NEAREST)
        cnts, _ = cv2.findContours(small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, col, 2)
        ys, xs = np.nonzero(small)
        label_text(vis, tag, (max(0, int(xs.mean()) - 30), max(22, int(ys.min()) - 8)), col, 0.7)
    if hair is not None and hair.any():
        small = cv2.resize(hair.astype(np.uint8), (vis.shape[1], vis.shape[0]), interpolation=cv2.INTER_NEAREST)
        cnts, _ = cv2.findContours(small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, HAIR, 1)
    if title:
        label_text(vis, title, (8, 26), (255, 255, 255), 0.6)
    return vis


def grid(tiles: list[np.ndarray], cols: int = 4, height: int | None = None) -> np.ndarray:
    """Tiles of possibly different sizes on a dark grid (each scaled to a common height)."""
    if not tiles:
        return np.zeros((10, 10, 3), np.uint8)
    hgt = height or max(t.shape[0] for t in tiles)
    tiles = [cv2.resize(t, (int(round(t.shape[1] * hgt / t.shape[0])), hgt), interpolation=cv2.INTER_AREA)
             if t.shape[0] != hgt else t for t in tiles]
    cw = max(t.shape[1] for t in tiles)
    rows = []
    for r in range(0, len(tiles), cols):
        row = [np.pad(t, ((0, 0), (0, cw - t.shape[1]), (0, 0))) for t in tiles[r:r + cols]]
        row += [np.zeros((hgt, cw, 3), np.uint8)] * (cols - len(row))
        rows.append(np.concatenate(row, 1))
    return np.concatenate(rows, 0)


def crop_tile(img: np.ndarray, lead: np.ndarray, follow: np.ndarray, hair: np.ndarray | None, title: str,
              height: int = 520, pad: float = 0.25, min_frac: float = 0.35) -> np.ndarray:
    """Overlay cropped to the couple (union of both masks, padded), so small dancers stay inspectable."""
    h, w = lead.shape
    ys, xs = np.nonzero(lead | follow)
    if len(xs):
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        ch = max((y1 - y0) * (1 + 2 * pad), min_frac * h)
        cw = max((x1 - x0) * (1 + 2 * pad), 0.6 * ch)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        x0, x1 = int(max(0, cx - cw / 2)), int(min(w, cx + cw / 2))
        y0, y1 = int(max(0, cy - ch / 2)), int(min(h, cy + ch / 2))
    else:
        x0, y0, x1, y1 = 0, 0, w, h
    sl = (slice(y0, y1), slice(x0, x1))
    return overlay(img[sl], lead[sl], follow[sl], None if hair is None else hair[sl], height, title)


def contact_sheet(tiles: list[np.ndarray], title: str = "", cols: int | None = None,
                  max_width: int = 3600) -> np.ndarray:
    sheet = grid(tiles, cols=cols or len(tiles))
    if sheet.shape[1] > max_width:
        s = max_width / sheet.shape[1]
        sheet = cv2.resize(sheet, (max_width, int(sheet.shape[0] * s)), interpolation=cv2.INTER_AREA)
    if title:
        band = np.full((40, sheet.shape[1], 3), 16, np.uint8)
        label_text(band, title, (10, 28), (255, 255, 255), 0.7)
        sheet = np.concatenate([band, sheet], 0)
    return sheet


def write_video(path: Path, frames: list[np.ndarray], fps: float) -> None:
    """H.264 via ffmpeg when available (plays in browsers), else OpenCV mp4v."""
    import shutil
    import subprocess
    h, w = frames[0].shape[:2]
    w2, h2 = w - w % 2, h - h % 2
    ff = shutil.which("ffmpeg")
    if ff:
        cmd = [ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w2}x{h2}",
               "-r", f"{fps:.3f}", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(path)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for fr in frames:
            proc.stdin.write(np.ascontiguousarray(fr[:h2, :w2]).tobytes())
        proc.stdin.close()
        if proc.wait() == 0:
            return
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w2, h2))
    for fr in frames:
        vw.write(np.ascontiguousarray(fr[:h2, :w2]))
    vw.release()


SKELETON = ((5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (0, 1), (0, 2), (1, 3), (2, 4))


def body_hair_row(img: np.ndarray, lead: np.ndarray, full: np.ndarray, body: np.ndarray, hair: np.ndarray,
                  kpts: np.ndarray | None, title: str, height: int = 520) -> np.ndarray:
    """One review row for the follow: FULL | BODY | HAIR tinted on the image, cropped to her upper body
    (top of her mask down to ~60 % of its height: hair, head, arms, hands)."""
    h, w = full.shape
    ys, xs = np.nonzero(full) if full.any() else (np.array([0, h - 1]), np.array([0, w - 1]))
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    pad = int(0.08 * (y1 - y0)) + 8
    y1 = y0 + int(0.6 * (y1 - y0))
    sel = ys < y1
    if sel.any():
        x0, x1 = xs[sel].min(), xs[sel].max() + 1
    x0, y0, x1, y1 = max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)
    panels = []
    for name, m, col in (("FULL", full, FOLLOW), ("BODY", body, FOLLOW), ("HAIR", hair, HAIR)):
        v = img.copy()
        if lead.any():
            v[lead] = (0.75 * v[lead] + 0.25 * np.array(LEAD)).astype(np.uint8)
        if m.any():
            v[m] = (0.45 * v[m] + 0.55 * np.array(col)).astype(np.uint8)
        cnts, _ = cv2.findContours(full.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(v, cnts, -1, (255, 255, 255), 1)
        if kpts is not None and name == "BODY":
            for a, b in SKELETON:
                if kpts[a, 2] > 0.3 and kpts[b, 2] > 0.3:
                    cv2.line(v, tuple(int(t) for t in kpts[a, :2]), tuple(int(t) for t in kpts[b, :2]), (60, 230, 60), 1,
                             cv2.LINE_AA)
            for j in range(17):
                if kpts[j, 2] > 0.3:
                    cv2.circle(v, (int(kpts[j, 0]), int(kpts[j, 1])), 2, (60, 230, 60), -1)
        v = v[y0:y1, x0:x1]
        s = height / v.shape[0]
        v = cv2.resize(v, (max(1, int(round(v.shape[1] * s))), height), interpolation=cv2.INTER_AREA)
        label_text(v, name, (6, height - 10), (255, 255, 255), 0.6)
        panels.append(v)
    sep = np.full((height, 4, 3), 30, np.uint8)
    row = np.concatenate([panels[0], sep, panels[1], sep, panels[2]], 1)
    band = np.full((30, row.shape[1], 3), 16, np.uint8)
    label_text(band, title, (6, 21), (255, 255, 255), 0.55)
    return np.concatenate([band, row], 0)
