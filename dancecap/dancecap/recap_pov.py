"""The class recap's phone POV data: which phone can be shown when, how its video is framed on the teachers, and the clips the film needs.

    python -m dancecap.recap_pov table              # per-phone usable seconds, teacher coverage, bad stretches  -> film/pov_table.json
    python -m dancecap.recap_pov offsets            # the phones' video offsets per chunk (piecewise)            -> camera_video_offsets.json
    python -m dancecap.recap_pov clips [--no-install] [--direction PATH]
                                                    # decode the phone clips the direction's POV shots need (camera_video --clip ...)

Why this exists (owner 2026-10-10, director mode): the recap film shows a large share of the lesson as the phones' own video, circling through
the phones, framed on the two teachers. The seven phones are handheld: their poses are per frame (cameras/<id>.npz) and some stretches of
some phones are known to be wrong; a phone must also actually have both teachers in frame. This module answers, for any window of the class:
is the phone's pose usable, are both teachers in view (and how large), and where the frame should be cropped.

Data (all under work/larissa_kadu_recap/, git-ignored):
  people/teachers_<cam>.npz   the two teachers' boxes per phone frame (display pixels, `size`), confidence, on the phone's own clock `t`;
                              the dance (reference) time of a frame is t + the video offset
  cameras/<cam>.npz           per-frame poses (R2; 05 / 06 rotation-only attached)
  chunks/bad_camera_stretches.json + the stretches the audit notes list (BAD below): poses that must not be used
  stitch.json + work/larissa_kadu_recap_<chunk>/sync/offsets.json   the video offsets measured per chunk (ms): applied piecewise

Pure functions (window_quality, choose_phones, crop_track, piecewise_offsets_from) take plain arrays and are tested without data.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .take import PACKAGE_ROOT

TAKE = "larissa_kadu_recap"
CAPTURE = "06_LarissaKaduRecap"
WORK = PACKAGE_ROOT / "work" / TAKE
PHONES = ("00", "01", "02", "03", "05", "06")
MAIN_PHONES = ("00", "01", "02", "03")        # localised directly in the room model; 05 / 06 are attached by rotation only (a few degrees, centre +-0.3 m)

# Poses that must NOT be used for a point-of-view / fly-to shot (seconds on the master clock). Sources: NOTES_F1 section 6 (the fit's masks), NOTES_T1
# (01 is wrong in 9-30 s; the "02 wrong in 0-35 s" claim is NOT supported: 02 agrees with 00 and 05 there, so it is not listed), chunks/bad_camera_stretches.json
# (qa masks), the audit of the first render. 06 is the attached (rotation-only) phone: 152-208 s is where the fit found its follow wrong (it was the
# mask, not the track, but the safe side costs one phone for one minute).
BAD: dict[str, list[tuple[float, float, str]]] = {
    "01": [(9.0, 30.0, "R2 pose wrong (NOTES_T1)"), (327.5, 334.5, "garbage 328-333 s (NOTES_F1)")],
    "02": [(242.0, 278.0, "pose ambiguity 242-278 s (NOTES_F1)"), (350.0, 356.0, "350-355 s (NOTES_F1)")],
    "05": [(203.5, 207.5, "204-207 s (NOTES_F1)")],
    "06": [(152.0, 208.0, "attached 06: the follow was wrong 152-208 s (NOTES_F1)")],
}
BAD_PAD_S = 1.0


@dataclass
class PovCfg:
    min_cover: float = 0.8            # share of the window's frames with both teachers detected and inside the frame
    relaxed_cover: float = 0.62       # accepted when nothing better exists (06 sees both in about half its frames)
    pose_cover: float = 0.97          # share of the window with a usable pose
    fill: float = 0.5                 # the teachers' boxes fill this share of the view height after the crop (the key band of 9:16 is 0.14 .. 0.66)
    top: float = 0.40                 # ... centred at this height from the top of the view (captions own the lower third)
    max_fill: float = 0.56            # the teachers' box never fills more than this share of the view height
    max_zoom: float = 2.2
    min_zoom: float = 1.07            # a crop never shows the frame's outer rim: the undistorted video has black wedges in its corners
    inset: float = 0.02               # ... the view stays this far (fraction of the frame) from the frame's edges
    max_upscale: float = 3.4          # source pixel -> output pixel (1920 px high output)
    view_aspect: float = 9.0 / 16.0
    out_h: int = 1920
    smooth_s: float = 0.6
    step_s: float = 0.1


# ------------------------------------------------------------------------------------------------ bad stretches


def bad_stretches(cams=PHONES, extra: dict | None = None, pad: float = BAD_PAD_S, bad_file: Path | None = None) -> dict[str, list[tuple[float, float, str]]]:
    """camera -> [(t0, t1, why)] padded: the table above plus chunks/bad_camera_stretches.json (and `extra`)"""
    out: dict[str, list[tuple[float, float, str]]] = {c: list(BAD.get(c, [])) for c in cams}
    p = bad_file if bad_file is not None else WORK / "chunks" / "bad_camera_stretches.json"
    if p.exists():
        doc = json.loads(p.read_text(encoding="utf-8"))
        for chunk, per_cam in doc.items():
            for cam, spans in per_cam.items():
                for a, b in spans:
                    out.setdefault(cam, []).append((float(a), float(b), f"chunks/bad_camera_stretches.json {chunk}"))
    for cam, spans in (extra or {}).items():
        out.setdefault(cam, []).extend((float(a), float(b), "extra") for a, b in spans)
    return {c: [(a - pad, b + pad, w) for a, b, w in v] for c, v in out.items() if c in cams}


def overlaps_bad(bad: list[tuple[float, float, str]], a: float, b: float) -> str | None:
    for x, y, why in bad:
        if a < y and b > x:
            return why
    return None


# ------------------------------------------------------------------------------------------------ video offsets


def piecewise_offsets_from(chunks: list[dict]) -> dict[str, list[list[float]]]:
    """chunks = [{"centre": t, "offsets": {cam: {"delta": s, "use": bool}}}] -> {cam: [[t, delta], ...]} (each chunk's value sits at the middle of the part
    of the class it owns; phones a chunk does not have, or flags `use: false`, are skipped)"""
    pts: dict[str, list[list[float]]] = {}
    for c in sorted(chunks, key=lambda x: x["centre"]):
        for cam, v in c["offsets"].items():
            if v.get("use", True) is False:
                continue
            pts.setdefault(cam, []).append([round(float(c["centre"]), 3), round(float(v["delta"]), 5)])
    return pts


def chunk_offsets(work: Path = WORK) -> dict[str, list[list[float]]]:
    """the video offsets the stitched fit used, per chunk (work/larissa_kadu_recap_<chunk>/sync/offsets.json), at the centre of each chunk's owned part"""
    st = json.loads((work / "stitch.json").read_text(encoding="utf-8"))
    hz = float(st["grid"]["hz"])
    chunks = []
    for c in st["chunks"]:
        k0, k1 = c["owned_k"]
        p = work.parent / f"{work.name}_{c['name']}" / "sync" / "offsets.json"
        if not p.exists():
            continue
        offs = json.loads(p.read_text(encoding="utf-8")).get("offsets", {})
        chunks.append({"centre": 0.5 * (k0 + k1) / hz, "offsets": offs, "name": c["name"]})
    return piecewise_offsets_from(chunks)


def write_offsets(work: Path = WORK, out: Path | None = None) -> Path:
    out = out or work / "camera_video_offsets.json"
    pw = chunk_offsets(work)
    doc = {"what": "video offsets of the class recap's phones: corrected reference time = t + delta(t); delta measured per chunk by dancecap.sync "
                   "(ms in sync_chunks/table.md), applied piecewise (linear between the chunks' centres, flat outside)",
           "offsets": {cam: {"piecewise": v, "median_ms": round(float(np.median([d for _, d in v])) * 1000, 1)} for cam, v in sorted(pw.items())}}
    out.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return out


def offset_at(piecewise: list[list[float]] | None, t: np.ndarray | float) -> np.ndarray | float:
    if not piecewise:
        return np.zeros_like(t, dtype=float) if isinstance(t, np.ndarray) else 0.0
    pw = np.asarray(piecewise, float)
    return np.interp(t, pw[:, 0], pw[:, 1])


# ------------------------------------------------------------------------------------------------ one phone's data


@dataclass
class PhoneData:
    """per frame of one phone: dance time (offset applied), both teachers visible inside the frame, their union box as FRACTIONS of the frame
    (x0, y0, x1, y1; y down), the frame size (display pixels), and a 10 Hz pose-usable flag"""
    cam: str
    t: np.ndarray
    ok: np.ndarray
    box: np.ndarray
    size: tuple[int, int]
    pose_t: np.ndarray = field(default_factory=lambda: np.zeros(0))
    pose_ok: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))

    def _sl(self, a: float, b: float) -> slice:
        i0 = int(np.searchsorted(self.t, a))
        i1 = int(np.searchsorted(self.t, b))
        return slice(i0, max(i1, i0))

    def cover(self, a: float, b: float) -> float:
        s = self._sl(a, b)
        return float(self.ok[s].mean()) if s.stop > s.start else 0.0

    def height(self, a: float, b: float) -> float:
        """median height of the teachers' union box in the window (fraction of the frame height)"""
        s = self._sl(a, b)
        m = self.ok[s]
        if not m.any():
            return 0.0
        bx = self.box[s][m]
        return float(np.median(bx[:, 3] - bx[:, 1]))

    def width(self, a: float, b: float) -> float:
        """median width of the teachers' union box in the window (fraction of the frame width)"""
        s = self._sl(a, b)
        m = self.ok[s]
        if not m.any():
            return 0.0
        bx = self.box[s][m]
        return float(np.median(bx[:, 2] - bx[:, 0]))

    def pose_cover(self, a: float, b: float) -> float:
        if len(self.pose_t) == 0:
            return 0.0
        i0, i1 = np.searchsorted(self.pose_t, a), np.searchsorted(self.pose_t, b)
        if i1 <= i0:
            return 0.0
        return float(self.pose_ok[i0:i1].mean())


def union_boxes(box: np.ndarray, conf: np.ndarray, size: tuple[int, int], min_conf: float = 0.5, margin: float = 0.01,
                merged: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """teachers_<cam>.npz box (T, 2, 4) display pixels + conf (T, 2) -> (ok (T,), union box fractions (T, 4)): ok = both teachers confident and their
    union inside the frame (a `margin` of the frame from the sides and the top; the bottom may touch). A frame in which one detection holds BOTH
    bodies (`merged`: the embrace; the other slot is empty and the confidence is halved) counts with that box."""
    w, h = float(size[0]), float(size[1])
    b = np.asarray(box, float)
    c = np.asarray(conf, float)
    fin = np.isfinite(b).all(axis=2)                                    # (T, 2)
    both = (c >= min_conf).all(axis=1) & fin.all(axis=1)
    if merged is not None:
        m = np.asarray(merged, bool)
        one = (fin.sum(axis=1) == 1) & ((m & fin) & (c >= 0.3 * 1.0)).any(axis=1)
        both = both | one
    with np.errstate(all="ignore"):
        x0 = np.nanmin(np.where(fin[:, :, None], b, np.nan)[:, :, 0], axis=1) / w
        y0 = np.nanmin(np.where(fin[:, :, None], b, np.nan)[:, :, 1], axis=1) / h
        x1 = np.nanmax(np.where(fin[:, :, None], b, np.nan)[:, :, 2], axis=1) / w
        y1 = np.nanmax(np.where(fin[:, :, None], b, np.nan)[:, :, 3], axis=1) / h
    inside = (x0 >= margin) & (y0 >= margin) & (x1 <= 1 - margin) & (y1 <= 1.0 + 1e-3)
    u = np.stack([x0, y0, x1, y1], axis=1)
    ok = both & inside & np.isfinite(u).all(axis=1) & (x1 > x0) & (y1 > y0)
    u = np.where(ok[:, None], u, np.nan)
    return ok, u


def load_phone(cam: str, work: Path = WORK, offsets: dict[str, list[list[float]]] | None = None, min_conf: float = 0.5) -> PhoneData:
    z = np.load(work / "people" / f"teachers_{cam}.npz", allow_pickle=True)
    size = (int(z["size"][0]), int(z["size"][1]))
    ok, u = union_boxes(z["box"], z["conf"], size, min_conf, merged=z["merged"])
    t = np.asarray(z["t"], float)
    t = t + offset_at((offsets or {}).get(cam), t)
    c = np.load(work / "cameras" / f"{cam}.npz", allow_pickle=True)
    ct = np.asarray(c["t"], float)
    ct = ct + offset_at((offsets or {}).get(cam), ct)
    loc = np.asarray(c["ok"], bool)
    grid = np.arange(0.0, 361.0, 0.1)
    pose_ok = pose_usable(ct, loc, grid)
    return PhoneData(cam, t, ok, u, size, grid, pose_ok)


def pose_usable(ct: np.ndarray, localised: np.ndarray, grid: np.ndarray, max_gap_s: float = 1.0) -> np.ndarray:
    """a grid time has a usable pose when a localised frame lies within max_gap_s / 2 of it and a frame at all within 0.2 s"""
    if len(ct) == 0:
        return np.zeros(len(grid), bool)
    lt = ct[localised]
    out = np.zeros(len(grid), bool)
    if len(lt) == 0:
        return out
    i = np.clip(np.searchsorted(lt, grid), 1, len(lt) - 1) if len(lt) > 1 else np.zeros(len(grid), int)
    near = np.minimum(np.abs(lt[i] - grid), np.abs(lt[np.maximum(i - 1, 0)] - grid)) if len(lt) > 1 else np.abs(lt[0] - grid)
    j = np.clip(np.searchsorted(ct, grid), 1, len(ct) - 1)
    anyf = np.minimum(np.abs(ct[j] - grid), np.abs(ct[j - 1] - grid))
    return (near <= max_gap_s / 2) & (anyf <= 0.2)


# ------------------------------------------------------------------------------------------------ which phone, when


def window_quality(ph: PhoneData, a: float, b: float, bad: list[tuple[float, float, str]], cfg: PovCfg) -> dict | None:
    """None when the phone cannot be shown in [a, b]; else {cover, height, score}"""
    if overlaps_bad(bad, a, b):
        return None
    if ph.pose_cover(a, b) < cfg.pose_cover:
        return None
    cov = ph.cover(a, b)
    if cov < cfg.relaxed_cover:
        return None
    h = ph.height(a, b)
    # the 9:16 view of a wide frame (a landscape phone) is a narrow strip of it: both teachers must fit into it, else one stands at its edge
    view_w = min(1.0, cfg.view_aspect / (ph.size[0] / ph.size[1]))
    if ph.width(a, b) > 0.96 * view_w:
        return None
    res = ph.size[1] / 1920.0
    # a phone with a bigger picture of the teachers wins; resolution breaks ties (the output is 1920 px high)
    score = 0.6 * cov + 0.25 * min(1.0, h / 0.45) + 0.15 * min(1.0, res * 1.5)
    return {"cover": round(cov, 3), "height": round(h, 3), "score": round(score, 4), "strong": cov >= cfg.min_cover}


def choose_phones(windows: list[tuple], phones: dict[str, PhoneData], bad: dict[str, list], cfg: PovCfg | None = None,
                  history: list[str] | None = None) -> list[dict | None]:
    """a phone for each window, circling through them: among the phones that can be shown (and see both teachers well: `strong`), the one used
    longest ago wins unless its score is clearly worse than the best (85 %); never the phone of the previous shot while another one qualifies.
    None for a window no phone can show."""
    cfg = cfg or PovCfg()
    last_used: dict[str, int] = {}
    used_s: dict[str, float] = {}
    prev = (history or [None])[-1]
    out: list[dict | None] = []
    for n, w in enumerate(windows):
        a, b = w[0], w[1]
        allowed = set(w[2]) if len(w) > 2 and w[2] else None          # e.g. the directly localised phones for a window that carries a 3D arrow
        cands = []
        for cam, ph in phones.items():
            q = window_quality(ph, a, b, bad.get(cam, []), cfg)
            if q:
                cands.append((cam, q))
        if allowed and any(c in allowed for c, _ in cands):
            cands = [(c, q) for c, q in cands if c in allowed]
        strong = [c for c in cands if c[1]["strong"]]
        pool = strong or cands
        if not pool:
            out.append(None)
            continue
        best = max(q["score"] for _, q in pool)
        good = [(c, q) for c, q in pool if q["score"] >= 0.85 * best]
        others = [(c, q) for c, q in good if c != prev]
        pick_from = others or good
        # the phone least recently used; ties: the one with the least POV time so far, then the higher score
        cam, q = min(pick_from, key=lambda cq: (last_used.get(cq[0], -1), used_s.get(cq[0], 0.0), -cq[1]["score"]))
        last_used[cam] = n
        used_s[cam] = used_s.get(cam, 0.0) + (b - a)
        prev = cam
        out.append({"camera": cam, **q, "alternatives": sorted(c for c, _ in cands if c != cam)})
    return out


# ------------------------------------------------------------------------------------------------ the crop


def _smooth(x: np.ndarray, sigma_samples: float) -> np.ndarray:
    if sigma_samples <= 0.3 or len(x) < 3:
        return x
    r = int(3 * sigma_samples)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma_samples) ** 2)
    k /= k.sum()
    pad = np.pad(x, (r, r), mode="edge")
    return np.convolve(pad, k, mode="valid")


def crop_track(ph: PhoneData, a: float, b: float, cfg: PovCfg | None = None) -> dict:
    """The crop of a phone's frame for [a, b]: at `step_s` the centre (cx right +, cy up +; -1 .. 1 of the frame) and the zoom (>= 1) that put the two
    teachers' union box at `fill` of the view height, centred `top` from the top of the view, smoothed, the view kept inside the frame, the zoom
    capped by the source resolution (`max_upscale`) and lowered until the teachers stay inside the view."""
    cfg = cfg or PovCfg()
    n = max(2, int(round((b - a) / cfg.step_s)) + 1)
    t = a + np.arange(n) * cfg.step_s
    fw, fh = ph.size
    frame_aspect = fw / fh
    s = ph._sl(a - 1.0, b + 1.0)
    m = ph.ok[s]
    if m.sum() < 3:
        return {"t0": round(a, 3), "dt": cfg.step_s, "cx": [0.0] * n, "cy": [0.0] * n, "zoom": [1.0] * n, "inside": 0.0, "zoom_median": 1.0}
    tt, bx = ph.t[s][m], ph.box[s][m]
    cx_f = np.interp(t, tt, 0.5 * (bx[:, 0] + bx[:, 2]))
    cy_f = np.interp(t, tt, 0.5 * (bx[:, 1] + bx[:, 3]))
    hh = np.interp(t, tt, bx[:, 3] - bx[:, 1])
    ww = np.interp(t, tt, bx[:, 2] - bx[:, 0])
    sg = cfg.smooth_s / cfg.step_s
    cx_f, cy_f, hh, ww = (_smooth(v, sg) for v in (cx_f, cy_f, hh, ww))
    zmax = max(1.0, min(cfg.max_zoom, cfg.max_upscale * fh / cfg.out_h))
    z0 = np.clip(cfg.fill / np.maximum(hh, 0.05), 1.0, zmax)
    # room to put the teachers `top` from the top of the view: the view (1 / zoom of the frame) must fit between the frame's top and bottom with the
    # box centre at that height; when it does not (a phone that already fills with them) the zoom grows, but never so far that the box fills more
    # than `max_fill` of the view (the feet under the captions are better than a cut-off head)
    room = np.minimum(cy_f / cfg.top, (1.0 - cy_f) / (1.0 - cfg.top))
    z_room = np.minimum(1.0 / np.maximum(room, 0.2), cfg.max_fill / np.maximum(hh, 0.05))
    z0 = np.clip(np.maximum(z0, z_room), cfg.min_zoom, max(zmax, cfg.min_zoom))
    best = None
    for shrink in (1.0, 0.92, 0.85, 0.78, 0.7, 0.62, 0.55, 0.0):
        z = np.maximum(cfg.min_zoom, z0 * shrink) if shrink > 0 else np.full(n, cfg.min_zoom)
        z = _smooth(z, sg)
        vh = 1.0 / z                                          # view height as a fraction of the frame height
        vw = vh * cfg.view_aspect / frame_aspect              # ... width
        # the box centre sits `top` from the view's top edge: view centre = box centre + (0.5 - top) * view height (y down)
        cyv = cy_f + (0.5 - cfg.top) * vh
        i = cfg.inset
        cxv = np.where(vw >= 1.0 - 2 * i, 0.5, np.clip(cx_f, vw / 2 + i, 1 - vw / 2 - i))
        cyv = np.where(vh >= 1.0 - 2 * i, 0.5, np.clip(cyv, vh / 2 + i, 1 - vh / 2 - i))
        # the teachers' union box inside the view (with 4 % of the view as margin)
        x_lo, x_hi = cxv - vw / 2, cxv + vw / 2
        y_lo, y_hi = cyv - vh / 2, cyv + vh / 2
        inside = (cx_f - ww / 2 >= x_lo - 0.04 * vw) & (cx_f + ww / 2 <= x_hi + 0.04 * vw) & (cy_f - hh / 2 >= y_lo - 0.04 * vh) & (cy_f + hh / 2 <= y_hi + 0.04 * vh)
        best = (z, cxv, cyv, float(inside.mean()))
        if inside.mean() >= 0.92:
            break
    z, cxv, cyv, ins = best
    return {"t0": round(float(t[0]), 3), "dt": cfg.step_s, "cx": [round(float(v), 4) for v in (cxv - 0.5) * 2.0],
            "cy": [round(float(v), 4) for v in (0.5 - cyv) * 2.0], "zoom": [round(float(v), 3) for v in z], "inside": round(ins, 3),
            "zoom_median": round(float(np.median(z)), 3)}


# ------------------------------------------------------------------------------------------------ whole-class tables


def build_phones(work: Path = WORK, offsets: dict | None = None, cams=PHONES) -> dict[str, PhoneData]:
    offsets = offsets if offsets is not None else chunk_offsets(work)
    return {c: load_phone(c, work, offsets) for c in cams if (work / "people" / f"teachers_{c}.npz").exists()}


def second_table(phones: dict[str, PhoneData], bad: dict[str, list], cfg: PovCfg | None = None, seconds: int = 361) -> dict[str, list]:
    """per phone and per second of the class: [cover, height, usable(0/1)] as lists (the audit table written to film/pov_table.json)"""
    cfg = cfg or PovCfg()
    out: dict[str, list] = {}
    for cam, ph in phones.items():
        rows = []
        for s in range(seconds):
            q = window_quality(ph, float(s), float(s + 1), bad.get(cam, []), cfg)
            rows.append([round(ph.cover(s, s + 1), 2), round(ph.height(s, s + 1), 2), int(q is not None)])
        out[cam] = rows
    return out


def clips_from_direction(direction: dict, pad: float = 0.5) -> dict[str, list[tuple[float, float]]]:
    """the phone clips a direction needs: {cam: [(t0, t1), ...]} from its camera_tour / fly_to_camera shots (dance seconds, padded)"""
    out: dict[str, list[tuple[float, float]]] = {}
    for seg in direction.get("segments", []):
        cam = seg.get("camera") or {}
        d0, d1 = seg["dance"]
        if cam.get("mode") == "camera_tour":
            for c in cam.get("cameras", []):
                a, b = c["dance_t"]
                out.setdefault(str(c["camera"]), []).append((float(a) - pad, float(b) + pad))
        elif cam.get("mode") == "fly_to_camera" and cam.get("camera"):
            out.setdefault(str(cam["camera"]), []).append((float(d0) - pad, float(d1) + pad))
    return out


# ------------------------------------------------------------------------------------------------ CLI


def cmd_table(a) -> int:
    ph = build_phones()
    bad = bad_stretches()
    tab = second_table(ph, bad)
    out = WORK / "film" / "pov_table.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"what": "per phone, per second of the class: [teacher cover, union-box height, usable]", "bad": {c: [[x, y, w] for x, y, w in v] for c, v in bad.items()},
                               "table": tab}, indent=0), encoding="utf-8")
    for cam, rows in tab.items():
        u = sum(r[2] for r in rows)
        print(f"{cam}: usable {u} of {len(rows)} s, median cover {np.median([r[0] for r in rows]):.2f}, median height {np.median([r[1] for r in rows if r[1] > 0]):.2f}")
    print(out)
    return 0


def cmd_offsets(a) -> int:
    p = write_offsets()
    print(p)
    for cam, v in json.loads(p.read_text())["offsets"].items():
        print(cam, "median", v["median_ms"], "ms", len(v["piecewise"]), "points")
    return 0


def cmd_clips(a) -> int:
    from . import camera_video as cv
    direction = Path(a.direction) if a.direction else WORK / "film" / "direction.json"
    doc = json.loads(direction.read_text(encoding="utf-8"))
    clips = clips_from_direction(doc)
    if not clips:
        print("the direction has no POV shots: nothing to export")
        return 0
    off = write_offsets()
    cv.OPTIONS["offsets_file"] = off
    cv.set_priority(a.priority)
    secs = {c: round(sum(y - x for x, y in cv.merge_ranges(v)), 1) for c, v in clips.items()}
    print("clips (s per phone):", secs)
    only = [c.strip() for c in a.phones.split(",")] if a.phones else list(PHONES)
    clips = {c: v for c, v in clips.items() if c in only}
    man = cv.run([TAKE], CAPTURE, None, [str(direction)], False, only, None, None, not a.no_install, False, False, 1280, 85, 256, 78,
                 clips=clips)
    print("pov_ok:", man.get("pov_ok"))
    return 0 if man.get("pov_ok") else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.recap_pov", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("table").set_defaults(fn=cmd_table)
    sub.add_parser("offsets").set_defaults(fn=cmd_offsets)
    c = sub.add_parser("clips")
    c.add_argument("--direction", default=None)
    c.add_argument("--no-install", action="store_true", help="write the master only (work/<take>/camera_video/<capture>); mirror later with camera_video --install-only")
    c.add_argument("--phones", default=None, help="only these phones (e.g. 00): the others' clips in the master stay as they are")
    c.add_argument("--priority", choices=("below_normal", "normal"), default="normal")
    c.set_defaults(fn=cmd_clips)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
