r"""john_lula FACING (2026-10-20): per frame and per dancer, does the dancer's FRONT, BACK or SIDE face the phone - decided
from the IMAGES - and does the fit agree?

Owner, 2026-10-20: "the follow (lula) spins and shows her back, but the post track is not capturing that (we should be able
to spot check forward-face/back to correct a pose error like that)".

    python -m dancecap.mono4d.jl_facing evidence      # per-frame image cues -> realism/v2/facing/evidence.npz
    python -m dancecap.mono4d.jl_facing labelsheets   # crops for the agent's own labels -> facing/label_sheets/
    python -m dancecap.mono4d.jl_facing fit           # fit facing: current fit, raw smplx_3db, SAM 3D Body -> fit_facing.npz
    python -m dancecap.mono4d.jl_facing calibrate     # precision / recall of every cue vs facing/agent_labels.json
    python -m dancecap.mono4d.jl_facing compare       # combined posterior, disagreement windows, contact sheets (facing/sheets/)
    python -m dancecap.mono4d.jl_facing targets       # facing/facing_targets.json for a refit
    python -m dancecap.mono4d.jl_facing review [--port 8765] [--build-only]   # the owner's spot-check page, 127.0.0.1 only

GEOMETRY. Torso frame of a 3D skeleton (SMPL-X joint order): left = (L hip - R hip) + (L shoulder - R shoulder), up = mid
shoulders - mid hips, forward = left x up (SMPL-X canonical: left +X, up +Y, forward +Z). REL YAW = the signed angle about
the torso's up axis from forward to the direction pelvis -> camera centre: 0 = the dancer faces the phone, +-180 = back to
it, +90 = the phone on the dancer's LEFT. Classes: FRONT |rel| <= 60, SIDE 60 < |rel| < 120, BACK |rel| >= 120. World yaw
of a horizontal direction v = atan2(v_x, v_z) (0 = world +Z, 90 = world +X; world Y up).

IMAGE CUES (per frame, per dancer; masks people/masks_01.npz, index 0 = John = lead, 1 = Lula = follow):
  bib     a white rectangle with dark digits inside the dancer's own mask in the bib's body band (John 124 on the back of the
          t-shirt, Lula 116 on the back / hip of the grey trousers) -> BACK. Colour + shape detector (bib_candidates).
  ocr     an exact EasyOCR read of the dancer's own number inside the own mask (people/in/ocr.json, every 2nd frame,
          jl_people_detect) -> BACK (the digits confirm the bib).
  face    YuNet (OpenCV FaceDetectorYN, the local facefusion .assets yunet_2023_mar.onnx; nothing downloaded) on an upscaled
          head crop, the face inside the own mask -> FRONT; RTMW's 68 face key points' confidence as a second face cue.
  lr      the left/right order of RTMW's shoulders and hips in 2D: rho = (x_L - x_R) / torso px; rho > 0 = the dancer's left
          on the image's right = FRONT, rho < 0 = BACK (against the fit's own projected rho).
  skin    MediaPipe SelfieMulticlass (local weights, .venv-face worker jl_facing_mp.py) on the head crop: face-skin share of
          (face-skin + hair) inside the own mask: the ponytail / back of the head vs a face.
Thresholds are calibrated on frames the agent labelled by LOOKING (facing/agent_labels.json); the combination is a
naive-Bayes product of the calibrated cue likelihoods smoothed by a 3-state (front / side / back) forward-backward HMM.
Likeness data (frames, crops, sheets) stays in work/.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from ..take import PACKAGE_ROOT, WORKSPACE

TAKE, CAM = "john_lula", "01"
ROLES = ("lead", "follow")
NAME = {"lead": "John", "follow": "Lula"}
BIB = {"lead": "124", "follow": "116"}
WORK = PACKAGE_ROOT / "work" / TAKE
REAL = WORK / "realism"
V2 = REAL / "v2"
OUT = V2 / "facing"
SHEETS = OUT / "sheets"
LABEL_SHEETS = OUT / "label_sheets"
REVIEW = OUT / "review"
FITS = {"current": REAL / "refit" / "audit_patch" / "smplx_rf2_noslip_3db_patched", "raw": WORK / "smplx_3db"}
SAM3DB = REAL / "sam3db" / "smplx_3db.npz"
MASKS = WORK / "people" / "masks_01.npz"
TEACHERS = WORK / "people" / "teachers_01.npz"
OCR = WORK / "people" / "in" / "ocr.json"
YUNET = Path("C:/Users/johnb/Desktop/facefusion/.assets/models/yunet_2023_mar.onnx")
MP_MULTICLASS = PACKAGE_ROOT / "weights" / "mediapipe" / "selfie_multiclass_256x256.tflite"
VENV_FACE = WORKSPACE / ".venv-face" / "Scripts" / "python.exe"
FPS = 29.994
CLASSES = ("front", "side", "back")
FRONT_MAX, BACK_MIN = 60.0, 120.0
COL = {"lead": (60, 200, 60), "follow": (230, 60, 230)}   # BGR: John green, Lula magenta
# SMPL-X joints
PELVIS, L_HIP, R_HIP, NECK, HEAD, L_SHO, R_SHO = 0, 1, 2, 12, 15, 16, 17
# COCO-WholeBody (RTMW) body points
KP = {"nose": 0, "leye": 1, "reye": 2, "lear": 3, "rear": 4, "ls": 5, "rs": 6, "lh": 11, "rh": 12}
WB_FACE = slice(23, 91)
BONES22 = [(0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8), (6, 9), (7, 10), (8, 11), (9, 12), (12, 15),
           (9, 13), (9, 14), (13, 16), (14, 17), (16, 18), (17, 19), (18, 20), (19, 21)]
LEFT22 = {1, 4, 7, 10, 13, 16, 18, 20}
RIGHT22 = {2, 5, 8, 11, 14, 17, 19, 21}
SPIN_WINDOW_S = (4.5, 29.5)


def log(msg: str) -> None:
    print(f"[jl_facing] {msg}", flush=True)


def _ws(p: Path) -> str:
    try:
        return Path(p).resolve().relative_to(WORKSPACE.resolve()).as_posix()
    except ValueError:
        return Path(p).as_posix()


# ================================================================================================ pure geometry
def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def torso_axes(J: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(..., >=18, 3) SMPL-X joints -> (forward, up, left) unit vectors of the torso (forward = left x up)."""
    J = np.asarray(J, float)
    left = (J[..., L_HIP, :] - J[..., R_HIP, :]) + (J[..., L_SHO, :] - J[..., R_SHO, :])
    up = _unit(0.5 * (J[..., L_SHO, :] + J[..., R_SHO, :]) - 0.5 * (J[..., L_HIP, :] + J[..., R_HIP, :]))
    left = _unit(left - (left * up).sum(-1, keepdims=True) * up)
    fwd = np.cross(left, up)
    return fwd, up, left


def rel_yaw_deg(fwd: np.ndarray, up: np.ndarray, P: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Signed angle (deg) about `up` from `fwd` to the direction P -> C (the camera centre). 0 = facing the camera,
    +-180 = back to it, +90 = the camera on the body's left."""
    d = np.asarray(C, float) - np.asarray(P, float)
    up = np.asarray(up, float)
    d = d - (d * up).sum(-1, keepdims=True) * up
    s = (np.cross(fwd, d) * up).sum(-1)
    c = (np.asarray(fwd, float) * d).sum(-1)
    return np.degrees(np.arctan2(s, c))


def facing_class(rel_deg: np.ndarray, front_max: float = FRONT_MAX, back_min: float = BACK_MIN) -> np.ndarray:
    """0 front, 1 side, 2 back (-1 where rel is NaN)."""
    a = np.abs(np.asarray(rel_deg, float))
    out = np.where(a <= front_max, 0, np.where(a >= back_min, 2, 1))
    return np.where(np.isfinite(a), out, -1).astype(np.int8)


def world_yaw_deg(v: np.ndarray) -> np.ndarray:
    """Heading of a direction about world Y: atan2(x, z) in degrees (0 = +Z, 90 = +X)."""
    v = np.asarray(v, float)
    return np.degrees(np.arctan2(v[..., 0], v[..., 2]))


def wrap_deg(a: np.ndarray) -> np.ndarray:
    return (np.asarray(a, float) + 180.0) % 360.0 - 180.0


def lr_rho(ls: np.ndarray, rs: np.ndarray, lh: np.ndarray, rh: np.ndarray) -> np.ndarray:
    """2D left/right order: ((x_ls - x_rs) + (x_lh - x_rh)) / 2 / torso length (px). > 0 = the body's left on the image's
    right = FRONT to the camera, < 0 = BACK. Inputs (..., >=2) image points."""
    ls, rs, lh, rh = (np.asarray(a, float)[..., :2] for a in (ls, rs, lh, rh))
    torso = np.linalg.norm(0.5 * (ls + rs) - 0.5 * (lh + rh), axis=-1)
    dx = 0.5 * ((ls[..., 0] - rs[..., 0]) + (lh[..., 0] - rh[..., 0]))
    return dx / np.maximum(torso, 1e-6)


def project(P: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World points (N, 3) -> pixels (N, 2), depth (N,) for one OpenCV pinhole camera (x_cam = R X + t)."""
    Xc = np.asarray(P, float) @ np.asarray(R, float).T + np.asarray(t, float)
    z = Xc[:, 2]
    uv = Xc[:, :2] / np.maximum(z, 1e-6)[:, None] * np.diag(K)[:2] + np.asarray(K)[:2, 2]
    return uv, z


# ================================================================================================ pure: bib detector
def bib_candidates(bgr: np.ndarray, own: np.ndarray, partner: np.ndarray | None, rows: tuple[int, int],
                   scale_px: float, params: dict | None = None) -> list[dict]:
    """White rectangles with dark digits inside the own mask within image rows [r0, r1).

    scale_px = the dancer's 2D stature proxy (px): the bib is ~0.06-0.12 of it wide. Returns candidates (best first):
    {box (x0, y0, x1, y1), area, fill, aspect, dark, contrast, score}. Score in 0..1 (shape x digits x contrast)."""
    import cv2
    p = dict(BIB_PARAMS)
    p.update(params or {})
    H, W = own.shape
    r0, r1 = max(0, int(rows[0])), min(H, int(rows[1]))
    if r1 - r0 < 4:
        return []
    region = own.copy()
    region[:r0] = False
    region[r1:] = False
    region = cv2.erode(region.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    if partner is not None:
        region &= ~cv2.dilate(partner.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    if region.sum() < 20:
        return []
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    S = hsv[..., 1].astype(np.int16)
    V = hsv[..., 2].astype(np.int16)
    vref = float(np.median(V[region]))
    white = region & (V >= max(p["v_min"], vref + p["v_above_body"])) & (S <= p["s_max"])
    if white.sum() < 6:
        return []
    closed = cv2.morphologyEx(white.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    out = []
    wmin, wmax = p["w_frac"][0] * scale_px, p["w_frac"][1] * scale_px
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a < p["area_min"] or max(w, h) < max(5, wmin) or max(w, h) > max(wmax, 8):
            continue
        comp = (lab == i).astype(np.uint8)
        cs, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull = cv2.convexHull(max(cs, key=cv2.contourArea))
        (_, _), (rw, rh), _ = cv2.minAreaRect(hull)
        rect_a = max(rw * rh, 1.0)
        fill_mask = np.zeros_like(comp)
        cv2.fillPoly(fill_mask, [hull], 1)
        inside = fill_mask.astype(bool)
        hull_a = float(inside.sum())
        fill = hull_a / rect_a                     # how rectangular the hull is
        aspect = max(rw, rh) / max(min(rw, rh), 1.0)
        if fill < p["fill_min"] or aspect > p["aspect_max"]:
            continue
        wv = V[inside & white]
        if wv.size < 4:
            continue
        wmed = float(np.median(wv))
        dark = inside & ~white & (V <= wmed - p["digit_drop"])
        dark_frac = float(dark.sum()) / max(hull_a, 1.0)
        # contrast: the ring just outside the hull vs the white paper
        ring = cv2.dilate(fill_mask, np.ones((5, 5), np.uint8)).astype(bool) & ~inside & own
        contrast = (wmed - float(np.median(V[ring]))) if ring.sum() >= 4 else 0.0
        sat_frac = float((inside & (S > p["sat_colour"])).sum()) / max(hull_a, 1.0)   # colour print (a logo), not a bib
        s_shape = min(1.0, (fill - p["fill_min"]) / max(1e-6, 0.9 - p["fill_min"]))
        if dark_frac < p["dark_lo"]:            # too few digit pixels: a plain white patch (sole, logo, floor)
            s_dark = dark_frac / p["dark_lo"]
        elif dark_frac > p["dark_hi"]:
            s_dark = max(0.0, 1.0 - (dark_frac - p["dark_hi"]) / 0.15)
        else:
            s_dark = 1.0
        s_con = min(1.0, max(0.0, contrast / p["contrast_full"]))
        s_col = max(0.0, 1.0 - sat_frac / p["sat_frac_max"])
        score = float(s_shape * s_dark * s_col * (0.3 + 0.7 * s_con))
        out.append({"box": (int(x), int(y), int(x + w), int(y + h)), "area": int(a), "fill": round(fill, 3),
                    "aspect": round(float(aspect), 2), "dark": round(dark_frac, 3), "contrast": round(contrast, 1),
                    "sat": round(sat_frac, 3),
                    "score": round(score, 3)})
    out.sort(key=lambda c: -c["score"])
    return out


BIB_PARAMS = {"v_min": 165, "v_above_body": 25, "s_max": 70, "area_min": 12, "w_frac": (0.07, 0.22),
              "fill_min": 0.55, "aspect_max": 2.2, "digit_drop": 40, "dark_lo": 0.04, "dark_hi": 0.45,
              "contrast_full": 60.0, "sat_colour": 90, "sat_frac_max": 0.15}
# the bib's band of the dancer's mask height (fraction from the mask top; John: back of the t-shirt, Lula: back / hip of
# the trousers; measured from 587 exact OCR reads: John 0.11-0.27, Lula 0.37-0.53 of the box) - widened for bends
BIB_BAND = {"lead": (0.06, 0.45), "follow": (0.25, 0.66)}


ARM_D_MIN = 0.12   # bib candidate centre within 0.12 torso of the shoulder-elbow segment = the sleeve patch


def point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance from point p to the segment a-b (2D or 3D)."""
    p, a, b = (np.asarray(x, float) for x in (p, a, b))
    ab = b - a
    t = float(np.clip(((p - a) @ ab) / max(ab @ ab, 1e-12), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def arm_distance(kp: np.ndarray, x: float, y: float, conf_min: float = 0.3) -> float:
    """Distance (px) from (x, y) to the nearer upper arm (shoulder-elbow, COCO 5-7 / 6-8) with both points confident;
    inf when neither arm is confident."""
    d = math.inf
    for s_, e_ in ((5, 7), (6, 8)):
        if kp[s_, 2] >= conf_min and kp[e_, 2] >= conf_min:
            d = min(d, point_segment_distance((x, y), kp[s_, :2], kp[e_, :2]))
    return d


# ================================================================================================ pure: statistics
def precision_recall(pred: np.ndarray, truth: np.ndarray) -> dict:
    """Binary precision / recall / support for boolean arrays (NaN-free)."""
    pred, truth = np.asarray(pred, bool), np.asarray(truth, bool)
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "n": int(len(pred)),
            "precision": round(tp / (tp + fp), 3) if tp + fp else None,
            "recall": round(tp / (tp + fn), 3) if tp + fn else None}


def hmm_forward_backward(lik: np.ndarray, trans: np.ndarray, prior: np.ndarray | None = None) -> np.ndarray:
    """Posterior marginals (T, S) of a discrete HMM from per-frame likelihoods lik (T, S) and transition matrix (S, S)."""
    lik = np.asarray(lik, float)
    T, S = lik.shape
    pi = np.full(S, 1.0 / S) if prior is None else np.asarray(prior, float)
    a = np.zeros((T, S))
    b = np.ones((T, S))
    x = pi * lik[0]
    a[0] = x / max(x.sum(), 1e-300)
    for t in range(1, T):
        x = (a[t - 1] @ trans) * lik[t]
        a[t] = x / max(x.sum(), 1e-300)
    for t in range(T - 2, -1, -1):
        x = trans @ (lik[t + 1] * b[t + 1])
        b[t] = x / max(x.sum(), 1e-300)
    p = a * b
    return p / np.maximum(p.sum(1, keepdims=True), 1e-300)


def runs(mask: np.ndarray, max_gap: int = 0, min_len: int = 1) -> list[tuple[int, int]]:
    """Inclusive (start, end) runs of True, merging runs separated by <= max_gap False frames."""
    idx = np.flatnonzero(np.asarray(mask, bool))
    if not len(idx):
        return []
    out = []
    s = e = int(idx[0])
    for i in idx[1:]:
        i = int(i)
        if i - e - 1 <= max_gap:
            e = i
        else:
            out.append((s, e))
            s = e = i
    out.append((s, e))
    return [(s, e) for s, e in out if e - s + 1 >= min_len]


# ================================================================================================ data
class Take:
    """Lazy access to the take's frames, masks, RTMW key points and the phone track (world frame)."""

    def __init__(self):
        from .. import mamma_dyn as md
        self.tr = md.load_tracks(WORK, [CAM])[CAM]
        self.F = len(self.tr.t)
        self.t = np.asarray(self.tr.t, float)
        self.K = np.asarray(self.tr.K, float)
        self.R = np.asarray(self.tr.R, float)
        self.tvec = np.asarray(self.tr.tvec, float)
        self.C = np.einsum("fji,fj->fi", self.R, -self.tvec)       # camera centres, world
        self.cam_ok = np.asarray(self.tr.ok, bool)
        self.W, self.H = int(self.tr.width), int(self.tr.height)
        self._masks = np.load(MASKS)
        self.mask_keys = set(self._masks.files)
        z = np.load(TEACHERS)
        self.kp_wb = z["kp_wb"]           # (F, 2, 133, 3)
        self.kp = z["kp"]                 # (F, 2, 17, 3)
        self.tbox = z["box"]
        self.visible = z["visible"]
        self.skel_in_partner = z["skel_in_partner"]
        self.mbox = self._masks["box"]    # (F, 2, 4) mask boxes
        self.marea = self._masks["area"]

    def frame(self, f: int) -> np.ndarray:
        import cv2
        return cv2.imread(self.tr.files[f], cv2.IMREAD_COLOR)

    def mask(self, f: int, o: int) -> np.ndarray | None:
        import cv2
        k = f"m_{int(f)}_{int(o)}"
        if k not in self.mask_keys:
            return None
        m = cv2.imdecode(np.frombuffer(self._masks[k].tobytes(), np.uint8), cv2.IMREAD_UNCHANGED)
        return None if m is None else m > 0


def mask_box(m: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(m)
    if not len(ys):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def head_point(kp_wb: np.ndarray, own: np.ndarray | None, conf_min: float = 0.3) -> tuple[float, float] | None:
    """Mean of RTMW nose / eyes / ears with conf >= conf_min that lie inside the own mask (dilated 6 px)."""
    import cv2
    pts = kp_wb[:5]
    good = pts[:, 2] >= conf_min
    if own is not None and good.any():
        md = cv2.dilate(own.astype(np.uint8), np.ones((13, 13), np.uint8)).astype(bool)
        H, W = md.shape
        for i in np.flatnonzero(good):
            x, y = int(round(pts[i, 0])), int(round(pts[i, 1]))
            if not (0 <= x < W and 0 <= y < H and md[y, x]):
                good[i] = False
    if not good.any():
        return None
    return float(pts[good, 0].mean()), float(pts[good, 1].mean())


def head_box(kp: np.ndarray, own: np.ndarray, mb) -> tuple[tuple[int, int, int, int], str] | None:
    """Square head crop (x0, y0, x1, y1) around the head point (RTMW in the own mask, else the mask's top), ~2.2 heads."""
    if mb is None:
        return None
    hp = head_point(kp, own)
    x0, y0, x1, y1 = mb
    hgt = max(y1 - y0, 1)
    ls, rs, lh, rh = (kp[KP[k]] for k in ("ls", "rs", "lh", "rh"))
    torso = float(np.hypot(*(0.5 * (ls[:2] + rs[:2]) - 0.5 * (lh[:2] + rh[:2]))))
    head = 0.42 * torso if min(ls[2], rs[2], lh[2], rh[2]) > 0.3 and torso > 10 else 0.13 * hgt
    head = float(np.clip(head, 0.08 * hgt, 0.2 * hgt))
    src = "rtmw"
    if hp is None:
        band = own[y0:y0 + max(2, int(0.1 * hgt))]
        xs = np.nonzero(band)[1]
        if not len(xs):
            return None
        hp = (float(np.median(xs)), y0 + 0.5 * head)
        src = "mask_top"
    s = 1.1 * head
    cx, cy = hp
    return (int(round(cx - s)), int(round(cy - s)), int(round(cx + s)), int(round(cy + s))), src


def crop_pad(img: np.ndarray, box, fill=0) -> np.ndarray:
    """Crop with zero padding outside the image."""
    x0, y0, x1, y1 = (int(v) for v in box)
    H, W = img.shape[:2]
    out = np.full((y1 - y0, x1 - x0) + img.shape[2:], fill, img.dtype)
    ax0, ay0, ax1, ay1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if ax1 > ax0 and ay1 > ay0:
        out[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0] = img[ay0:ay1, ax0:ax1]
    return out


# ================================================================================================ label sheets (agent)
def label_frames() -> dict:
    """The frames the agent labels: A = 40 seeded-random frames over the whole take, B = every 25th frame of the
    4.5-29.5 s spin window. Both dancers per frame."""
    F = 2643
    rng = np.random.default_rng(20261020)
    A = sorted(int(x) for x in rng.choice(np.arange(F), 40, replace=False))
    B = list(range(int(SPIN_WINDOW_S[0] * FPS), int(SPIN_WINDOW_S[1] * FPS), 25))
    return {"A_random": A, "B_spin": B}


def _tile(T: Take, f: int, o: int, img: np.ndarray, H: int = 380) -> np.ndarray:
    """Body crop (own mask outlined, outside dimmed) | upper-body zoom + frame thumbnail."""
    import cv2
    role = ROLES[o]
    own = T.mask(f, o)
    out = np.full((H, 470, 3), 30, np.uint8)
    if own is None or not own.any():
        cv2.putText(out, f"{f} {NAME[role]} no mask", (10, 40), 0, 0.7, (255, 255, 255), 2)
        return out
    x0, y0, x1, y1 = mask_box(own)
    pad = int(0.08 * (y1 - y0))
    box = (x0 - pad, y0 - pad, x1 + pad, y1 + pad)
    vis = img.copy()
    dim = (vis * 0.45).astype(np.uint8)
    vis = np.where(own[..., None], vis, dim)
    cs, _ = cv2.findContours(own.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(vis, cs, -1, COL[role], 1)
    body = crop_pad(vis, box)
    s = H / body.shape[0]
    bw = min(270, max(40, int(body.shape[1] * s)))
    body = cv2.resize(body, (bw, H), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    out[:, :bw] = body
    uh = int(0.45 * (y1 - y0)) + 2                       # upper-body zoom: top 45 % of the mask box, square
    cxm = 0.5 * (x0 + x1)
    z = crop_pad(img, (int(cxm - uh / 2), y0 - 4, int(cxm + uh / 2), y0 - 4 + uh))
    out[:200, 270:470] = cv2.resize(z, (200, 200), interpolation=cv2.INTER_CUBIC)
    out[210:370, 275:365] = cv2.resize(vis, (90, 160), interpolation=cv2.INTER_AREA)
    lab = f"{f} {T.t[f]:.2f}s {NAME[role]}"
    cv2.putText(out, lab, (4, 18), 0, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, lab, (4, 18), 0, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def labelsheets(frames: list[int] | None = None, tag: str = "set") -> list[str]:
    import cv2
    T = Take()
    LABEL_SHEETS.mkdir(parents=True, exist_ok=True)
    if frames is None:
        lf = label_frames()
        frames = sorted(set(lf["A_random"]) | set(lf["B_spin"]))
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "label_frames.json").write_text(json.dumps(lf, indent=1))
    per = 6
    paths = []
    gap_v = np.full((380, 6, 3), 255, np.uint8)
    for si in range(0, len(frames), per):
        tiles = []
        for f in frames[si:si + per]:
            img = T.frame(f)
            tiles.append(np.concatenate([_tile(T, f, 0, img), gap_v, _tile(T, f, 1, img)], 1))
        while len(tiles) % 2:
            tiles.append(np.zeros_like(tiles[0]))
        rows = []
        for r in range(0, len(tiles), 2):
            rows.append(np.concatenate([tiles[r], np.full((380, 14, 3), 255, np.uint8), tiles[r + 1]], 1))
            rows.append(np.full((8, rows[-1].shape[1], 3), 255, np.uint8))
        p = LABEL_SHEETS / f"{tag}_{si // per:02d}.jpg"
        cv2.imwrite(str(p), np.concatenate(rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 90])
        paths.append(str(p))
    log(f"{len(paths)} label sheets, {len(frames)} frames -> {LABEL_SHEETS}")
    return paths


# ================================================================================================ face detector (YuNet)
class YuNet:
    """YuNet 2023-03 (OpenCV Zoo, MIT) through cv2.dnn: the facefusion copy renames the 12 outputs to '0'..'11'
    (cls 8/16/32, obj 8/16/32, bbox 8/16/32, kps 8/16/32); decoded as OpenCV's FaceDetectorYN does
    (score = sqrt(cls * obj), centre = (col + dx, row + dy) * stride, size = exp(.) * stride)."""

    def __init__(self, path: Path = YUNET):
        import cv2
        if not Path(path).exists():
            raise SystemExit(f"[jl_facing] YuNet weights missing: {path} (local facefusion assets; nothing is downloaded)")
        self.net = cv2.dnn.readNetFromONNX(str(path))
        self.names = [str(i) for i in range(12)]

    def detect(self, img: np.ndarray, thr: float = 0.3, nms: float = 0.3) -> list[dict]:
        import cv2
        h, w = img.shape[:2]
        S = int(math.ceil(max(h, w, 32) / 32) * 32)
        can = np.zeros((S, S, 3), np.uint8)
        can[:h, :w] = img
        self.net.setInput(cv2.dnn.blobFromImage(can))
        o = self.net.forward(self.names)
        return yunet_decode(o, S, thr, nms)


def yunet_decode(o: list, S: int, thr: float = 0.3, nms: float = 0.3) -> list[dict]:
    """Decode YuNet's 12 outputs for a square S x S input -> [{box (x, y, w, h), score, lm (5, 2)}] after NMS."""
    import cv2
    boxes, scores, lms = [], [], []
    for i, st in enumerate((8, 16, 32)):
        n = S // st
        cls = np.clip(np.asarray(o[i]).reshape(-1), 0, 1)
        obj = np.clip(np.asarray(o[i + 3]).reshape(-1), 0, 1)
        sc = np.sqrt(cls * obj)
        keep = np.flatnonzero(sc >= thr)
        if not len(keep):
            continue
        r, c = np.divmod(keep, n)
        bb = np.asarray(o[i + 6]).reshape(-1, 4)[keep]
        lm = np.asarray(o[i + 9]).reshape(-1, 10)[keep]
        cx, cy = (c + bb[:, 0]) * st, (r + bb[:, 1]) * st
        ww, hh = np.exp(bb[:, 2]) * st, np.exp(bb[:, 3]) * st
        for k in range(len(keep)):
            boxes.append([float(cx[k] - ww[k] / 2), float(cy[k] - hh[k] / 2), float(ww[k]), float(hh[k])])
            scores.append(float(sc[keep[k]]))
            lms.append(np.stack([(lm[k, 0::2] + c[k]) * st, (lm[k, 1::2] + r[k]) * st], -1))
    if not boxes:
        return []
    idx = np.array(cv2.dnn.NMSBoxes(boxes, scores, thr, nms)).reshape(-1)
    return [{"box": boxes[i], "score": scores[i], "lm": lms[i]} for i in idx]


# ================================================================================================ evidence
def body_top(kp: np.ndarray, own: np.ndarray, mb) -> tuple[float, float, float]:
    """(top y, height px, head px) of the body proper: the head point minus one head when RTMW's head is in the own
    mask (a raised arm does not move it), else the mask top."""
    x0, y0, x1, y1 = mb
    hgt = max(y1 - y0, 1)
    hp = head_point(kp, own)
    ls, rs, lh, rh = (kp[KP[k]] for k in ("ls", "rs", "lh", "rh"))
    torso = float(np.hypot(*(0.5 * (ls[:2] + rs[:2]) - 0.5 * (lh[:2] + rh[:2]))))
    head = 0.42 * torso if min(ls[2], rs[2], lh[2], rh[2]) > 0.3 and torso > 10 else 0.13 * hgt
    head = float(np.clip(head, 0.06 * hgt, 0.2 * hgt))
    top = y0 if hp is None else max(y0, hp[1] - 0.8 * head)
    return float(top), float(max(y1 - top, 1)), head


def torso_quad_mask(kp: np.ndarray, shape: tuple[int, int], grow: float = 1.15, conf_min: float = 0.3) -> np.ndarray | None:
    """Filled quadrilateral L shoulder - R shoulder - R hip - L hip of RTMW (scaled by `grow` about its centroid)."""
    import cv2
    q = kp[[KP["ls"], KP["rs"], KP["rh"], KP["lh"]]]
    if (q[:, 2] < conf_min).any():
        return None
    c = q[:, :2].mean(0)
    pts = (c + grow * (q[:, :2] - c)).round().astype(np.int32)
    m = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(m, cv2.convexHull(pts), 1)
    return m.astype(bool)


def frame_evidence(img: np.ndarray, masks: list, kps: np.ndarray, yunet: YuNet | None,
                   skel_in_partner: np.ndarray | None = None) -> list[dict]:
    """Image cues of both dancers in one frame (masks [John, Lula], kps (2, 133, 3) RTMW)."""
    import cv2
    out = []
    for o, role in enumerate(ROLES):
        own, partner = masks[o], masks[1 - o]
        ev = {"ok": False}
        if own is None or own.sum() < 200:
            out.append(ev)
            continue
        mb = mask_box(own)
        kp = kps[o]
        top, hgt, head = body_top(kp, own, mb)
        ev.update(ok=True, mbox=mb, top=top, hgt=hgt, head=head)
        # ---- bib (where the person mask misses most of RTMW's torso - a broken mask - the torso quad joins the search)
        a, b = BIB_BAND[role]
        region = own
        ev["bib_region"] = "mask"
        sip = 0.0 if skel_in_partner is None else float(np.nan_to_num(skel_in_partner[o], nan=1.0))
        tq = torso_quad_mask(kp, own.shape) if sip <= 0.5 else None
        if tq is not None and tq.sum() >= 100 and (own & tq).sum() < 0.5 * tq.sum():
            region = own | (tq & ~(partner if partner is not None else np.zeros_like(own)))
            ev["bib_region"] = "mask+torso"
        cands = bib_candidates(img, region, partner, (top + a * hgt, top + b * hgt), hgt)
        # John's sleeve carries a striped flag patch (white + dark, bib-like): candidates on the upper arm are dropped
        for c in cands:
            c["arm_d"] = round(arm_distance(kp, 0.5 * (c["box"][0] + c["box"][2]), 0.5 * (c["box"][1] + c["box"][3]))
                               / max(2.4 * head, 1.0), 3)
        good = [c for c in cands if not (c["arm_d"] < ARM_D_MIN)]
        ev["bib"] = good[0] if good else None
        ev["bib_on_arm"] = max([c["score"] for c in cands if c["arm_d"] < ARM_D_MIN], default=0.0)
        # ---- face (YuNet on the upscaled upper body)
        ev["face"] = None
        if yunet is not None:
            x0, y0, x1, y1 = mb
            box = (x0 - 12, int(top - 0.3 * head), x1 + 12, int(top + 0.45 * hgt))
            crop = crop_pad(img, box)
            s = float(min(2.5, 640.0 / max(crop.shape[:2])))
            cr = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
            best = None
            H, W = own.shape
            for d in yunet.detect(cr, 0.3):
                bx, by, bw, bh = d["box"]
                cx, cy = box[0] + (bx + bw / 2) / s, box[1] + (by + bh / 2) / s
                fw = bw / s
                xi, yi = int(np.clip(cx, 0, W - 1)), int(np.clip(cy, 0, H - 1))
                lm = d["lm"] / s + np.array([box[0], box[1]])
                # the eyes + nose inside the own mask, not inside the partner's
                pts = lm[:3].round().astype(int)
                inside = [bool(own[np.clip(y, 0, H - 1), np.clip(x, 0, W - 1)]) for x, y in pts]
                in_partner = partner is not None and bool(partner[yi, xi])
                size_ratio = fw / max(head, 1.0)
                iod = float(np.linalg.norm(lm[0] - lm[1]))
                rec = {"score": round(d["score"], 3), "cx": round(cx, 1), "cy": round(cy, 1), "w": round(fw, 1),
                       "own_pts": int(sum(inside)), "in_partner": in_partner, "size_ratio": round(size_ratio, 2),
                       "dy_top": round((cy - top) / hgt, 3),
                       # frontality: inter-ocular distance / face width (frontal ~0.4, profile < 0.2) and the nose's
                       # offset from the eyes' midpoint in inter-ocular units (signed, + = towards the image's right)
                       "iod_ratio": round(iod / max(fw, 1e-6), 3),
                       "nose_off": round(float(lm[2, 0] - 0.5 * (lm[0, 0] + lm[1, 0])) / max(iod, 1.0), 3)}
                if rec["own_pts"] >= 2 and not in_partner and (best is None or rec["score"] > best["score"]):
                    best = rec
            ev["face"] = best
        # ---- RTMW face points + 2D left/right order
        ev["rtmw_face_conf"] = float(np.mean(kp[WB_FACE, 2]))
        ev["rtmw_eyes_conf"] = float(np.mean(kp[[0, 1, 2], 2]))
        ev["rtmw_ears_conf"] = float(np.mean(kp[[3, 4], 2]))
        ls, rs, lh, rh = (kp[KP[k]] for k in ("ls", "rs", "lh", "rh"))
        ev["lr_conf"] = float(min(ls[2], rs[2], lh[2], rh[2]))
        ev["lr_rho"] = float(lr_rho(ls, rs, lh, rh))
        hb = head_box(kp, own, mb)
        ev["head_box"] = None if hb is None else list(hb[0])
        ev["head_src"] = None if hb is None else hb[1]
        out.append(ev)
    return out


def ocr_reads(T: Take) -> np.ndarray:
    """(F, 2) best EasyOCR confidence of the dancer's own number read inside the own mask (NaN = frame not read:
    the pod read every 2nd frame; 0 = read, no own number)."""
    from ..jl_people import bib_score
    F = T.F
    out = np.full((F, 2), np.nan)
    out[::2] = 0.0
    reads = json.loads(OCR.read_text())["reads"]
    cache = {}
    for r in reads:
        f = int(r["frame"])
        c = np.array(r["quad"], float).mean(0)
        for o, role in enumerate(ROLES):
            other = BIB[ROLES[1 - o]]
            s = bib_score(r["text"].strip(), BIB[role], (other, "126"))
            if s < 0.9:
                continue
            key = (f, o)
            if key not in cache:
                cache[key] = T.mask(f, o)
            m = cache[key]
            x, y = int(c[0]), int(c[1])
            if m is not None and 0 <= x < T.W and 0 <= y < T.H and m[y, x]:
                out[f, o] = max(out[f, o], float(r["conf"]))
    return out


def evidence(frames: list[int] | None = None, out_path: Path | None = None) -> Path:
    """Every frame's image cues -> facing/evidence.npz (+ evidence_frames.json with the raw per-frame records)."""
    import cv2
    cv2.setNumThreads(2)
    T = Take()
    yn = YuNet()
    frames = list(range(T.F)) if frames is None else list(frames)
    F = T.F
    arr = {k: np.full((F, 2), np.nan) for k in (
        "bib_score", "bib_fill", "bib_dark", "bib_contrast", "bib_w", "bib_arm_d", "bib_on_arm", "face_score", "face_size_ratio", "face_dy_top",
        "face_iod_ratio", "face_nose_off", "bib_region_torso",
        "rtmw_face_conf", "rtmw_eyes_conf", "rtmw_ears_conf", "lr_rho", "lr_conf", "top", "hgt", "head")}
    arr["bib_box"] = np.full((F, 2, 4), -1, np.int32)
    arr["face_xy"] = np.full((F, 2, 2), np.nan)
    arr["head_box"] = np.full((F, 2, 4), -1, np.int32)
    arr["head_src_rtmw"] = np.zeros((F, 2), bool)
    arr["mask_ok"] = np.zeros((F, 2), bool)
    t0 = time.time()
    for n, f in enumerate(frames):
        img = T.frame(f)
        masks = [T.mask(f, 0), T.mask(f, 1)]
        evs = frame_evidence(img, masks, T.kp_wb[f], yn, T.skel_in_partner[f])
        for o, ev in enumerate(evs):
            if not ev.get("ok"):
                continue
            arr["mask_ok"][f, o] = True
            for k in ("top", "hgt", "head", "rtmw_face_conf", "rtmw_eyes_conf", "rtmw_ears_conf", "lr_rho", "lr_conf"):
                arr[k][f, o] = ev[k]
            b = ev["bib"]
            arr["bib_score"][f, o] = 0.0
            arr["bib_on_arm"][f, o] = ev["bib_on_arm"]
            arr["bib_region_torso"][f, o] = float(ev["bib_region"] == "mask+torso")
            if b is not None:
                arr["bib_score"][f, o] = b["score"]
                arr["bib_fill"][f, o], arr["bib_dark"][f, o], arr["bib_contrast"][f, o] = b["fill"], b["dark"], b["contrast"]
                arr["bib_w"][f, o] = (b["box"][2] - b["box"][0]) / max(ev["hgt"], 1)
                arr["bib_arm_d"][f, o] = b["arm_d"]
                arr["bib_box"][f, o] = b["box"]
            fc = ev["face"]
            arr["face_score"][f, o] = 0.0
            if fc is not None:
                arr["face_score"][f, o] = fc["score"]
                arr["face_size_ratio"][f, o] = fc["size_ratio"]
                arr["face_dy_top"][f, o] = fc["dy_top"]
                arr["face_iod_ratio"][f, o] = fc["iod_ratio"]
                arr["face_nose_off"][f, o] = fc["nose_off"]
                arr["face_xy"][f, o] = (fc["cx"], fc["cy"])
            if ev["head_box"] is not None:
                arr["head_box"][f, o] = ev["head_box"]
                arr["head_src_rtmw"][f, o] = ev["head_src"] == "rtmw"
        if n % 200 == 0:
            log(f"evidence {n}/{len(frames)} frames, {time.time() - t0:.0f} s")
    arr["ocr"] = ocr_reads(T)
    arr["t"] = T.t
    arr["cam_ok"] = T.cam_ok
    arr["visible"] = T.visible
    arr["skel_in_partner"] = T.skel_in_partner
    out_path = Path(out_path or OUT / "evidence.npz")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arr, bib_params=json.dumps(BIB_PARAMS), bib_band=json.dumps(BIB_BAND))
    log(f"evidence -> {out_path} ({time.time() - t0:.0f} s)")
    return out_path


def skin(ev_path: Path | None = None) -> Path:
    """MediaPipe SelfieMulticlass face-skin / hair shares on the head crops (subprocess in .venv-face) -> skin.npz."""
    import subprocess
    ev = np.load(ev_path or OUT / "evidence.npz")
    T = Take()
    tmp = OUT / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    np.save(tmp / "head_boxes.npy", ev["head_box"].astype(np.int32))
    spec = {"model": str(MP_MULTICLASS), "frames": list(T.tr.files), "boxes": str(tmp / "head_boxes.npy"),
            "masks": str(MASKS), "out": str(OUT / "skin.npz")}
    (tmp / "mp_spec.json").write_text(json.dumps(spec))
    if not VENV_FACE.exists():
        raise SystemExit(f"[jl_facing] {VENV_FACE} missing (mediapipe env)")
    cmd = [str(VENV_FACE), "-I", str(Path(__file__).with_name("jl_facing_mp.py")), "--spec", str(tmp / "mp_spec.json")]
    log(" ".join(cmd))
    subprocess.run(cmd, check=True)
    return OUT / "skin.npz"


# ================================================================================================ fit facing
def interp_joints(t_src: np.ndarray, J: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Linear interpolation of (T, N, 3) joints from the fit's grid times to times t (clamped at the ends)."""
    t_src = np.asarray(t_src, float)
    k = np.clip(np.searchsorted(t_src, t) - 1, 0, len(t_src) - 2)
    a = np.clip((t - t_src[k]) / (t_src[k + 1] - t_src[k]), 0.0, 1.0)[:, None, None]
    return (1 - a) * J[k] + a * J[k + 1]


def load_fit_joints() -> dict:
    """{'current' | 'raw' | 'sam3db': (F, 2, 22, 3) world joints at the VIDEO frames' pts, valid (F, 2)}."""
    T = Take()
    out = {}
    for name, d in FITS.items():
        Js, V = [], []
        for role in ROLES:
            with np.load(d / f"{role}.npz", allow_pickle=True) as z:
                J = np.asarray(z["joints"], float)[:, :22]
                tg = np.asarray(z["t"], float)
                ab = np.asarray(z["absent"], bool) if "absent" in z.files else np.zeros(len(tg), bool)
            Js.append(interp_joints(tg, J, T.t))
            k = np.clip(np.round(T.t * 30.0).astype(int), 0, len(tg) - 1)
            V.append(ab[k])
        Js = np.stack(Js, 1)
        # 'absent' frames (the stitch's hidden / camera-not-ok spans) still carry joints that are rendered: kept, flagged
        out[name] = {"joints": Js, "valid": np.isfinite(Js).all((2, 3)), "absent": np.stack(V, 1), "src": _ws(d)}
    with np.load(SAM3DB, allow_pickle=True) as z:
        Js = np.stack([np.asarray(z[f"{r}_joints_world"], float)[:, :22] for r in ROLES], 1)
        V = np.stack([np.asarray(z[f"{r}_valid"], bool) & np.asarray(z["cam_ok"], bool) for r in ROLES], 1)
        assert np.allclose(np.asarray(z["t"], float), T.t, atol=1e-4)
    out["sam3db"] = {"joints": Js, "valid": V & np.isfinite(Js).all((2, 3)), "src": _ws(SAM3DB)}
    return out, T


def fit_facing() -> Path:
    """Rel yaw / class / world yaw / projected left-right rho of every fit, per frame and dancer -> fit_facing.npz."""
    fits, T = load_fit_joints()
    res = {"t": T.t}
    for name, d in fits.items():
        J = d["joints"]
        fwd, up, left = torso_axes(J)
        rel = rel_yaw_deg(fwd, up, J[:, :, PELVIS], T.C[:, None, :])
        rel = np.where(d["valid"], rel, np.nan)
        res[f"{name}_rel"] = rel
        res[f"{name}_cls"] = facing_class(rel)
        res[f"{name}_yaw_world"] = np.where(d["valid"], world_yaw_deg(fwd), np.nan)
        res[f"{name}_valid"] = d["valid"]
        if "absent" in d:
            res[f"{name}_absent"] = d["absent"]
        # projected 2D skeleton + rho
        uv = np.full(J.shape[:3] + (2,), np.nan)
        for f in range(T.F):
            for o in range(2):
                p, z = project(J[f, o], T.K[f], T.R[f], T.tvec[f])
                uv[f, o] = np.where(z[:, None] > 0, p, np.nan)
        res[f"{name}_uv"] = uv.astype(np.float32)
        res[f"{name}_rho"] = lr_rho(uv[:, :, L_SHO], uv[:, :, R_SHO], uv[:, :, L_HIP], uv[:, :, R_HIP])
        res[f"{name}_src"] = d["src"]
    # the direction pelvis -> camera in the world (yaw), per fit pelvis (current fit)
    Jc = fits["current"]["joints"]
    res["cam_dir_yaw_world"] = world_yaw_deg(T.C[:, None, :] - Jc[:, :, PELVIS])
    p = OUT / "fit_facing.npz"
    np.savez_compressed(p, **res)
    log(f"fit facing -> {p}")
    return p


# ================================================================================================ cues -> observations
CLS_INDEX = {"front": 0, "side": 1, "back": 2}
LABEL_TO_CLASS = {"F": "front", "FS": "front", "S": "side", "BS": "back", "B": "back"}
# chosen on the agent labels (calibrate() prints the sweeps; see calibration.json)
CUE_THR = {"bib": 0.6, "ocr": 0.3, "face": 0.6, "face_dy": 0.2, "face_size": (0.3, 1.5), "rtmw_face": 0.9,
           "lr": 0.2, "lr_conf": 0.3, "skin_hi": 0.5, "skin_lo": 0.2, "skel_in_partner_max": 0.5, "face_iod": 0.25, "face_nose": 0.6,
           "visible_min": 0.45}
# discrete observation alphabet of every cue (index 0 .. n-1; -1 = missing / not observed)
CUE_STATES = {"bib": ("no", "bib"), "ocr": ("no", "read"), "face": ("no", "profile", "frontal"), "rtmw_face": ("lo", "hi"),
              "lr": ("back", "mid", "front"), "skin": ("hair", "mid", "face")}


def load_evidence() -> dict:
    ev = dict(np.load(OUT / "evidence.npz", allow_pickle=True))
    sk = OUT / "skin.npz"
    if sk.exists():
        z = np.load(sk)
        share = z["share"]                                # (F, 2, 6): bg, hair, body-skin, face-skin, clothes, other
        prob = z["prob"]
        fs, hr = prob[..., 3], prob[..., 1]
        ev["skin_face_share"] = fs / np.maximum(fs + hr, 1e-6)
        ev["skin_face_share"][(fs + hr) < 0.05] = np.nan
        ev["skin_hair"] = hr
        ev["skin_face"] = fs
        ev["skin_argmax_face"] = share[..., 3]
    return ev


def observations(ev: dict, thr: dict | None = None) -> dict:
    """Per cue (F, 2) int observation index (-1 = missing) from the evidence arrays."""
    th = dict(CUE_THR)
    th.update(thr or {})
    ok = ev["mask_ok"]
    # a dancer mostly hidden behind the partner (teachers 'visible' < visible_min): a MISSING bib / face / OCR read
    # says nothing (only positive detections count), and RTMW's key points are guesses
    seen = ok & (np.nan_to_num(ev["visible"], nan=0.0) >= th["visible_min"])

    def pos_only(hit: np.ndarray, observed: np.ndarray) -> np.ndarray:
        return np.where(observed & (hit | seen), hit.astype(int), -1)
    obs = {}
    b = ev["bib_score"]
    obs["bib"] = pos_only(np.nan_to_num(b) >= th["bib"], ok & np.isfinite(b))
    o = ev["ocr"]
    obs["ocr"] = pos_only(np.nan_to_num(o) >= th["ocr"], ok & np.isfinite(o))
    fs, dy, sr = ev["face_score"], ev["face_dy_top"], ev["face_size_ratio"]
    face = (np.nan_to_num(fs) >= th["face"]) & (np.nan_to_num(dy, nan=9) <= th["face_dy"]) & \
        (np.nan_to_num(sr) >= th["face_size"][0]) & (np.nan_to_num(sr, nan=9) <= th["face_size"][1])
    frontal = face & (np.nan_to_num(ev.get("face_iod_ratio", np.full_like(fs, np.nan))) >= th["face_iod"]) &         (np.abs(np.nan_to_num(ev.get("face_nose_off", np.full_like(fs, np.nan)), nan=9.0)) <= th["face_nose"])
    obs["face"] = np.where(ok & (face | seen), np.where(frontal, 2, face.astype(int)), -1)
    # RTMW locks onto the partner during occlusions (teachers skel_in_partner): its key-point cues are then missing
    own_skel = seen & (np.nan_to_num(ev["skel_in_partner"], nan=1.0) <= th["skel_in_partner_max"])
    rf = ev["rtmw_face_conf"]
    obs["rtmw_face"] = np.where(own_skel & np.isfinite(rf), (np.nan_to_num(rf) >= th["rtmw_face"]).astype(int), -1)
    rho, lc = ev["lr_rho"], ev["lr_conf"]
    lr = np.where(np.nan_to_num(rho) >= th["lr"], 2, np.where(np.nan_to_num(rho) <= -th["lr"], 0, 1))
    obs["lr"] = np.where(own_skel & np.isfinite(rho) & (np.nan_to_num(lc) >= th["lr_conf"]), lr, -1)
    if "skin_face_share" in ev:
        s = ev["skin_face_share"]
        sk = np.where(np.nan_to_num(s) >= th["skin_hi"], 2, np.where(np.nan_to_num(s) <= th["skin_lo"], 0, 1))
        obs["skin"] = np.where(seen & np.isfinite(s), sk, -1)
    return obs


def label_arrays(F: int) -> dict:
    """{role: (F,) class index (-1 unlabelled / U)} from facing/agent_labels.json."""
    lab = json.loads((OUT / "agent_labels.json").read_text())
    out = {r: np.full(F, -1, int) for r in ROLES}
    for f, v in lab["labels"].items():
        for r in ROLES:
            c = LABEL_TO_CLASS.get(v[r])
            if c is not None:
                out[r][int(f)] = CLS_INDEX[c]
    return out


def lik_tables(obs: dict, labels: np.ndarray, o: int, exclude: int | None = None, alpha: float = 1.0) -> dict:
    """P(obs state | class) per cue for dancer o from the labelled frames (Laplace alpha), optionally leaving one out."""
    tabs = {}
    idx = np.flatnonzero(labels >= 0)
    if exclude is not None:
        idx = idx[idx != exclude]
    for cue, x in obs.items():
        n = len(CUE_STATES[cue])
        t = np.full((3, n), alpha)
        for f in idx:
            s = x[f, o]
            if s >= 0:
                t[labels[f], s] += 1
        t = t / t.sum(1, keepdims=True)
        if cue in HARD_BACK:
            # physical prior: the bib is on the dancer's BACK, the body hides it from the front (0 false positives in the
            # labelled detections); the Laplace estimate from ~20 front labels (1/22) would let one wrong RTMW left/right
            # call outvote a visible bib
            t[0, 1] = min(t[0, 1], HARD_BACK[cue])
            t[0] = t[0] / t[0].sum()
        tabs[cue] = t
    return tabs


HARD_BACK = {"bib": 0.01, "ocr": 0.01}   # P(cue fires | FRONT) ceiling


def frame_likelihood(obs: dict, tabs: dict, o: int, power: float = 0.7, cues: tuple | None = None) -> np.ndarray:
    """(F, 3) naive-Bayes likelihood of front / side / back from the cue observations (tempered by `power`:
    the cues are not independent)."""
    F = next(iter(obs.values())).shape[0]
    L = np.ones((F, 3))
    for cue, x in obs.items():
        if cues is not None and cue not in cues:
            continue
        s = x[:, o]
        m = s >= 0
        L[m] *= tabs[cue][:, s[m]].T ** power
    return L / L.sum(1, keepdims=True)


TRANS = np.array([[0.90, 0.09, 0.01],
                  [0.09, 0.82, 0.09],
                  [0.01, 0.09, 0.90]])


def posterior(obs: dict, tabs: dict, o: int, cues: tuple | None = None) -> np.ndarray:
    return hmm_forward_backward(frame_likelihood(obs, tabs, o, cues=cues), TRANS)


def calibrate() -> dict:
    """Precision / recall of every cue (threshold sweeps) and of the combination (leave-one-out) vs the agent labels."""
    ev = load_evidence()
    F = len(ev["t"])
    labs = label_arrays(F)
    fz = np.load(OUT / "fit_facing.npz")
    rep = {"labels": {r: {c: int((labs[r] == i).sum()) for c, i in CLS_INDEX.items()} for r in ROLES},
           "thresholds": CUE_THR, "cues": {}, "sweeps": {}, "combined_loo": {}, "fits_vs_labels": {}}
    sweeps = {
        "bib -> back": ("bib_score", [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99], ">=", 2),
        "face -> front": ("face_score", [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], ">=", 0),
        "face iod ratio -> front": ("face_iod_ratio", [0.1, 0.15, 0.2, 0.25, 0.3, 0.35], ">=", 0),
        "rtmw face conf -> front": ("rtmw_face_conf", [0.6, 0.7, 0.8, 0.85, 0.9, 0.95], ">=", 0),
        "lr rho -> front": ("lr_rho", [0.0, 0.1, 0.2, 0.3, 0.4], ">=", 0),
        "lr rho -> back": ("lr_rho", [0.0, -0.1, -0.2, -0.3, -0.4], "<=", 2),
        "skin face share -> front": ("skin_face_share", [0.3, 0.4, 0.5, 0.6, 0.7], ">=", 0),
        "skin face share -> back": ("skin_face_share", [0.05, 0.1, 0.15, 0.2, 0.3], "<=", 2),
    }
    for name, (key, ths, op, cls) in sweeps.items():
        if key not in ev:
            continue
        rep["sweeps"][name] = {}
        for o, r in enumerate(ROLES):
            lab = labs[r]
            m = lab >= 0
            x = ev[key][:, o]
            rows = []
            for th in ths:
                pred = (np.nan_to_num(x, nan=-9) >= th) if op == ">=" else (np.nan_to_num(x, nan=9) <= th)
                pr = precision_recall(pred[m], lab[m] == cls)
                rows.append({"thr": th, "precision": pr["precision"], "recall": pr["recall"], "tp": pr["tp"], "fp": pr["fp"]})
            rep["sweeps"][name][r] = rows
    obs = observations(ev)
    cue_defs = {"bib -> back": ("bib", lambda x: x == 1, 2), "ocr -> back": ("ocr", lambda x: x == 1, 2),
                "face (any) -> front": ("face", lambda x: x >= 1, 0), "face (frontal) -> front": ("face", lambda x: x == 2, 0),
                "rtmw face conf -> front": ("rtmw_face", lambda x: x == 1, 0),
                "lr -> front": ("lr", lambda x: x == 2, 0), "lr -> back": ("lr", lambda x: x == 0, 2),
                "skin face -> front": ("skin", lambda x: x == 2, 0), "skin hair -> back": ("skin", lambda x: x == 0, 2)}
    rep["cue_note"] = ("per cue: precision / recall over the labelled frames where the cue is OBSERVED (gated: visible, "
                       "RTMW not on the partner; positive detections always count); side labels are negatives")
    for cue, (base, pred_fn, cls) in cue_defs.items():
        if base not in obs:
            continue
        rep["cues"][cue] = {}
        for o, r in enumerate(ROLES):
            lab = labs[r]
            m = (lab >= 0) & (obs[base][:, o] >= 0)
            pr = precision_recall(pred_fn(obs[base][m, o]), lab[m] == cls)
            pr["observed_labelled"] = int(m.sum())
            pr["labelled"] = int((lab >= 0).sum())
            rep["cues"][cue][r] = pr
    # the fits as classifiers
    for n in ("current", "raw", "sam3db"):
        rep["fits_vs_labels"][n] = {}
        for o, r in enumerate(ROLES):
            lab = labs[r]
            c = fz[f"{n}_cls"][:, o]
            m = (lab >= 0) & (c >= 0)
            rep["fits_vs_labels"][n][r] = {"n": int(m.sum()), "accuracy": round(float((c[m] == lab[m]).mean()), 3),
                                           "front_back_swaps": int(((c[m] == 0) & (lab[m] == 2) | (c[m] == 2) & (lab[m] == 0)).sum())}
    # combination, leave one out
    for o, r in enumerate(ROLES):
        lab = labs[r]
        idx = np.flatnonzero(lab >= 0)
        pred, conf = [], []
        for f in idx:
            tabs = lik_tables(obs, lab, o, exclude=f)
            p = posterior(obs, tabs, o)[f]
            pred.append(int(p.argmax()))
            conf.append(float(p.max()))
        pred, conf = np.array(pred), np.array(conf)
        truth = lab[idx]
        out = {"n": int(len(idx)), "accuracy": round(float((pred == truth).mean()), 3),
               "front_back_swaps": int(((pred == 0) & (truth == 2) | (pred == 2) & (truth == 0)).sum())}
        for c, i in CLS_INDEX.items():
            out[c] = precision_recall(pred == i, truth == i)
        for th in (0.6, 0.8, 0.9):
            m = conf >= th
            out[f"conf>={th}"] = {"share": round(float(m.mean()), 3),
                                  "accuracy": round(float((pred[m] == truth[m]).mean()), 3) if m.any() else None,
                                  "back_precision": precision_recall((pred == 2)[m], (truth == 2)[m])["precision"],
                                  "front_precision": precision_recall((pred == 0)[m], (truth == 0)[m])["precision"]}
        out["frames"] = [{"f": int(f), "label": CLASSES[t], "pred": CLASSES[p_], "conf": round(c_, 3)}
                         for f, t, p_, c_ in zip(idx, truth, pred, conf)]
        rep["combined_loo"][r] = out
    # held-out dense spin labels (set C): never used for the tables
    dense = json.loads((OUT / "agent_labels.json").read_text()).get("dense_eval", {})
    rep["dense_eval"] = {}
    for o, r in enumerate(ROLES):
        if r not in dense:
            continue
        fr = np.array([int(f) for f, c in dense[r].items() if c in LABEL_TO_CLASS], int)
        truth = np.array([CLS_INDEX[LABEL_TO_CLASS[dense[r][str(f)]]] for f in fr], int)
        P = posterior(obs, lik_tables(obs, labs[r], o), o)[fr]
        pred = P.argmax(1)
        d = {"n": int(len(fr)), "image_accuracy": round(float((pred == truth).mean()), 3),
             "image_front_back_swaps": int(((pred == 0) & (truth == 2) | (pred == 2) & (truth == 0)).sum()),
             "image_back": precision_recall(pred == 2, truth == 2)}
        for n in ("current", "sam3db"):
            c = fz[f"{n}_cls"][fr, o]
            d[f"{n}_accuracy"] = round(float((c == truth).mean()), 3)
            d[f"{n}_back"] = precision_recall(c == 2, truth == 2)
        d["frames"] = [{"f": int(f), "label": CLASSES[t_], "image": CLASSES[p_], "conf": round(float(q.max()), 2),
                        "fit_rel": None if not np.isfinite(fz[f"current_rel"][f, o]) else int(round(fz["current_rel"][f, o]))}
                       for f, t_, p_, q in zip(fr, truth, pred, P)]
        rep["dense_eval"][r] = d
    (OUT / "calibration.json").write_text(json.dumps(rep, indent=1))
    log(f"calibration -> {OUT / 'calibration.json'}")
    return rep


def timeline_sheets() -> list[str]:
    """sheets/timeline_<role>.png: |rel yaw| of the fit and SAM 3D Body vs the image evidence over the whole take,
    the cue hits, the disagreement windows shaded."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fz = np.load(OUT / "fit_facing.npz", allow_pickle=True)
    ev = load_evidence()
    obs = observations(ev)
    P = np.load(OUT / "posterior.npz")["post"]
    rep = json.loads((OUT / "disagreements.json").read_text())
    t = fz["t"]
    SHEETS.mkdir(parents=True, exist_ok=True)
    paths = []
    spans = [(0, 22.5), (22, 44.5), (44, 66.5), (66, 88.1)]
    for o, r in enumerate(ROLES):
        fig, axs = plt.subplots(len(spans), 1, figsize=(20, 3.1 * len(spans)))
        for ax, (a, b) in zip(axs, spans):
            m = (t >= a) & (t <= b)
            for w in rep["windows"]:
                if w["role"] == r and w["t1"] >= a and w["t0"] <= b:
                    ax.axvspan(w["t0"], w["t1"], color="#ffb000" if w["kind"] == "rotate" else "#ff3030", alpha=0.25)
                    ax.text(w["t0"], 186, w["id"], fontsize=7)
            ax.plot(t[m], np.abs(fz["current_rel"][m, o]), "-", color="#1a8a1a", lw=1.6, label="fit |rel| (current)")
            ax.plot(t[m], np.abs(fz["sam3db_rel"][m, o]), "-", color="#888888", lw=0.9, label="SAM 3D Body |rel|")
            ax.plot(t[m], P[m, o] @ np.array([0.0, 90.0, 180.0]), "--", color="k", lw=0.9, label="image E[class] (0/90/180)")
            for cue, st, y, mk, col, lab in (("bib", 1, 176, "s", "#d00000", "bib"), ("ocr", 1, 169, "^", "#a000a0", "ocr"),
                                            ("face", 2, 4, "o", "#0040ff", "frontal face"),
                                            ("face", 1, 20, "o", "#90a8ff", "profile face"), ("lr", 0, 162, ".", "#00a0a0", "RTMW L/R back"),
                                            ("lr", 2, 12, ".", "#00a0a0", "RTMW L/R front")):
                hit = m & (obs[cue][:, o] == st)
                ax.plot(t[hit], np.full(hit.sum(), y), mk, color=col, ms=3, label=lab)
            ax.axhline(FRONT_MAX, color="0.8", lw=0.8)
            ax.axhline(BACK_MIN, color="0.8", lw=0.8)
            ax.set_xlim(a, b)
            ax.set_ylim(-5, 195)
            ax.set_yticks([0, 60, 90, 120, 180])
            ax.grid(alpha=0.25)
        axs[0].legend(ncol=9, fontsize=8, loc="upper right")
        axs[0].set_title(f"{NAME[r]}: |rel yaw| (0 = front to the phone, 180 = back); red = flip180 window, amber = rotate window")
        axs[-1].set_xlabel("video time (s)")
        plt.tight_layout()
        p = SHEETS / f"timeline_{r}.png"
        plt.savefig(p, dpi=72)
        plt.close(fig)
        paths.append(str(p))
    return paths


# ================================================================================================ compare: posterior vs fit
CONF_MIN = 0.8          # image posterior needed to call a frame
BOUNDARY_DEG = 20.0     # front/back <-> side disagreements closer than this to the class boundary are not flagged
FIT_P_MAX = 0.15        # ... while the posterior of the fit's own class is at most this
FIT = "current"


def image_posterior() -> tuple[dict, dict, dict]:
    """Posterior (F, 2, 3) of front / side / back from the image cues (tables from all agent labels) + per-cue-only
    posteriors for the sheets; returns (post, obs, ev)."""
    ev = load_evidence()
    F = len(ev["t"])
    labs = label_arrays(F)
    obs = observations(ev)
    P = np.zeros((F, 2, 3))
    L = np.zeros((F, 2, 3))
    tabs_all = {}
    for o, r in enumerate(ROLES):
        tabs = lik_tables(obs, labs[r], o)
        tabs_all[r] = {k: v.round(4).tolist() for k, v in tabs.items()}
        P[:, o] = posterior(obs, tabs, o)
        L[:, o] = frame_likelihood(obs, tabs, o)
    return {"post": P, "lik": L, "tables": tabs_all}, obs, ev


def target_yaw(cls_img: int, cam_yaw: float, fit_yaw: float) -> tuple[float, float]:
    """World yaw (deg) the torso should have for image class cls_img (0 front, 1 side, 2 back) given the yaw of the
    pelvis -> camera direction, and the band half-width (deg). Side: the profile nearer to the fit's yaw."""
    if cls_img == 0:
        return float(wrap_deg(cam_yaw)), FRONT_MAX
    if cls_img == 2:
        return float(wrap_deg(cam_yaw + 180.0)), 180.0 - BACK_MIN
    a, b = wrap_deg(cam_yaw + 90.0), wrap_deg(cam_yaw - 90.0)
    t = a if abs(wrap_deg(a - fit_yaw)) <= abs(wrap_deg(b - fit_yaw)) else b
    return float(t), (BACK_MIN - FRONT_MAX) / 2.0


def delta_into_band(fit_yaw: float, target: float, half: float) -> float:
    """Smallest signed rotation (deg) that brings fit_yaw inside [target - half, target + half]."""
    d = float(wrap_deg(target - fit_yaw))
    if abs(d) <= half:
        return 0.0
    return d - math.copysign(half, d)


def disagreements(post: np.ndarray, fit_cls: np.ndarray, conf_min: float = CONF_MIN, fit_p_max: float = FIT_P_MAX,
                  max_gap: int = 3, fit_rel: np.ndarray | None = None, margin: float = BOUNDARY_DEG,
                  include_side: bool = False) -> list[dict]:
    """Windows where the image posterior (F, 3) is confident and gives the fit's class (F,) little probability (and,
    with fit_rel, the fit's rel yaw is `margin` deg clear of the class boundary)."""
    F = len(fit_cls)
    cls_img = post.argmax(1)
    conf = post.max(1)
    pf = np.where(fit_cls >= 0, post[np.arange(F), np.clip(fit_cls, 0, 2)], np.nan)
    bad = (conf >= conf_min) & (np.nan_to_num(pf, nan=1.0) <= fit_p_max) & (cls_img != fit_cls)
    if not include_side:
        bad &= cls_img != 1     # the image's SIDE calls are not reliable (precision 0.2-0.4 on the agent labels)
    if fit_rel is not None:
        # not a boundary case: the fit must sit `margin` deg inside its own class, away from the image class
        a = np.abs(np.asarray(fit_rel, float))
        clear = np.where(cls_img == 0, a >= FRONT_MAX + margin, np.where(cls_img == 2, a <= BACK_MIN - margin,
                                                                           (a <= FRONT_MAX - margin) | (a >= BACK_MIN + margin)))
        bad &= np.nan_to_num(clear, nan=0).astype(bool)
    flip = bad & (((cls_img == 2) & (fit_cls == 0)) | ((cls_img == 0) & (fit_cls == 2)))
    out = []
    for s, e in runs(bad, max_gap=max_gap, min_len=2):
        seg = np.arange(s, e + 1)
        b = seg[bad[seg]]
        kind = "flip180" if flip[b].mean() >= 0.5 else "rotate"
        if kind == "rotate" and len(b) < 4:
            continue
        vals, cnt = np.unique(cls_img[b], return_counts=True)
        out.append({"start": int(s), "end": int(e), "n_bad": int(len(b)), "kind": kind,
                    "image_class": CLASSES[int(vals[cnt.argmax()])],
                    "fit_classes": {CLASSES[int(c)]: int((fit_cls[b] == c).sum()) for c in (0, 1, 2)},
                    "conf_median": round(float(np.median(conf[b])), 3)})
    return out


def compare() -> dict:
    """Posterior, disagreement windows (current fit; raw and SAM 3D Body for reference) -> disagreements.json + sheets."""
    pz, obs, ev = image_posterior()
    P = pz["post"]
    fz = dict(np.load(OUT / "fit_facing.npz", allow_pickle=True))
    t = ev["t"]
    np.savez_compressed(OUT / "posterior.npz", post=P.astype(np.float32), lik=pz["lik"].astype(np.float32), t=t,
                        **{f"obs_{k}": v.astype(np.int8) for k, v in obs.items()})
    rep = {"what": "windows where confident image evidence contradicts the fit's facing", "conf_min": CONF_MIN,
           "fit_p_max": FIT_P_MAX, "tables": pz["tables"], "fits": {}, "windows": []}
    for n in ("current", "raw", "sam3db"):
        rep["fits"][n] = {}
        for o, r in enumerate(ROLES):
            c = fz[f"{n}_cls"][:, o].astype(int)
            w = disagreements(P[:, o], c, fit_rel=fz[f"{n}_rel"][:, o])
            conf = P[:, o].max(1) >= CONF_MIN
            agree = conf & (P[:, o].argmax(1) == c)
            rep["fits"][n][r] = {"windows": len(w), "flip180_windows": sum(x["kind"] == "flip180" for x in w),
                                 "frames_bad": int(sum(x["n_bad"] for x in w)),
                                 "confident_frames": int(conf.sum()), "confident_agree_share": round(float(agree.sum() / max(conf.sum(), 1)), 3)}
            if n == FIT:
                for x in w:
                    x["role"] = r
                    x["t0"], x["t1"] = round(float(t[x["start"]]), 2), round(float(t[x["end"]]), 2)
                    seg = np.arange(x["start"], x["end"] + 1)
                    x["fit_rel_deg"] = [None if not np.isfinite(v) else int(round(v)) for v in fz[f"{n}_rel"][seg, o]]
                    x["sam3db_rel_deg"] = [None if not np.isfinite(v) else int(round(v)) for v in fz[f"sam3db_rel"][seg, o]]
                    x["image_class_per_frame"] = [CLASSES[int(k)] for k in P[seg, o].argmax(1)]
                    x["image_conf_per_frame"] = [round(float(v), 2) for v in P[seg, o].max(1)]
                    x["cues"] = {k: int((obs[k][seg, o] == s).sum()) for k, s in
                                 (("bib", 1), ("ocr", 1), ("face", 2), ("rtmw_face", 1), ("skin", 0), ("lr", 0))
                                 if k in obs}
                    x["cues"]["face_profile"] = int((obs["face"][seg, o] == 1).sum())
                    x["cues"]["face_frontal"] = x["cues"].pop("face")
                    x["cues"]["lr_back"] = x["cues"].pop("lr")
                    if "skin" in x["cues"]:
                        x["cues"]["skin_hair"] = x["cues"].pop("skin")
                    x["cues"]["lr_front"] = int((obs["lr"][seg, o] == 2).sum())
                    x["cues"]["skin_face"] = int((obs["skin"][seg, o] == 2).sum()) if "skin" in obs else 0
                    x["in_spin_window"] = bool(SPIN_WINDOW_S[0] <= x["t0"] <= SPIN_WINDOW_S[1])
                    x["sam3db_agrees_with_image"] = round(float(np.mean(
                        fz["sam3db_cls"][seg, o] == P[seg, o].argmax(1))), 2)
                    rep["windows"].append(x)
    rep["windows"].sort(key=lambda x: (x["kind"] != "flip180", -x["n_bad"]))
    for i, x in enumerate(rep["windows"]):
        x["id"] = f"w{i:02d}"
    rep["check_windows"] = check_windows(P, fz, ev, rep["windows"])
    # the high-precision check alone: frames where a bib / an OCR read of the own number shows the BACK
    rep["fits_on_bib_frames"] = {}
    for n in ("current", "raw", "sam3db"):
        rep["fits_on_bib_frames"][n] = {}
        for o, r in enumerate(ROLES):
            hit = (obs["bib"][:, o] == 1) | (obs["ocr"][:, o] == 1)
            a = np.abs(fz[f"{n}_rel"][:, o])
            m = hit & np.isfinite(a)
            low = m & (a < 100.0)
            rep["fits_on_bib_frames"][n][r] = {"frames": int(m.sum()), "fit_abs_rel_ge_100": round(float((m & (a >= 100)).sum() / max(m.sum(), 1)), 4),
                                               "fit_front_lt_60": int((m & (a < 60)).sum()),
                                               "below_100_runs": [[int(x), int(y)] for x, y in runs(low, max_gap=3)],
                                               "abs_rel_p5_p50": [round(float(np.percentile(a[m], q)), 1) for q in (5, 50)] if m.any() else None}
    (OUT / "disagreements.json").write_text(json.dumps(rep, indent=1))
    log(f"{len(rep['windows'])} disagreement windows -> {OUT / 'disagreements.json'}")
    return rep


def check_windows(P: np.ndarray, fz: dict, ev: dict, flagged: list[dict]) -> list[dict]:
    """Spans for the owner's eye that the image evidence cannot settle or that he asked about: 'absent' = the fit is
    interpolated there (the stitch's absent / camera-not-ok spans), 'hidden' = the dancer mostly hidden (teachers
    visible < 0.45 for >= 15 frames), 'spin' = Lula's back phases in 4.5-29.5 s (image back, conf >= 0.8, >= 5 frames)."""
    t = fz["t"]
    out = []
    taken = {(x["role"], f) for x in flagged for f in range(x["start"], x["end"] + 1)}

    def add(kind, o, s, e, note):
        r = ROLES[o]
        if sum((r, f) in taken for f in range(s, e + 1)) > 0.5 * (e - s + 1):
            return
        seg = np.arange(s, e + 1)
        rel = fz[f"{FIT}_rel"][seg, o]
        out.append({"role": r, "start": int(s), "end": int(e), "t0": round(float(t[s]), 2), "t1": round(float(t[e]), 2),
                    "kind": kind, "note": note, "n_bad": 0,
                    "image_class": CLASSES[int(np.bincount(P[seg, o].argmax(1), minlength=3).argmax())],
                    "conf_median": round(float(np.median(P[seg, o].max(1))), 3),
                    "fit_classes": {CLASSES[c]: int((fz[f"{FIT}_cls"][seg, o] == c).sum()) for c in (0, 1, 2)},
                    "fit_turn_deg": None if not np.isfinite(rel).any() else int(round(float(np.nanmax(np.abs(rel)) - np.nanmin(np.abs(rel)))))})
        for f in seg:
            taken.add((r, int(f)))
    for o in range(2):
        if f"{FIT}_absent" in fz:
            for s, e in runs(fz[f"{FIT}_absent"][:, o], max_gap=2, min_len=5):
                add("absent", o, s, e, "the fit is interpolated here (stitch 'absent': hidden or camera track not ok)")
        for s, e in runs(np.nan_to_num(ev["visible"][:, o]) < 0.45, max_gap=3, min_len=15):
            add("hidden", o, s, e, "mostly hidden behind the partner: the image cannot confirm the facing")
    o = ROLES.index("follow")
    inwin = (t >= SPIN_WINDOW_S[0]) & (t <= SPIN_WINDOW_S[1])
    back = inwin & (P[:, o].argmax(1) == 2) & (P[:, o].max(1) >= CONF_MIN)
    for s, e in runs(back, max_gap=2, min_len=5):
        s2, e2 = max(0, s - 8), min(len(t) - 1, e + 8)
        add("spin", o, s2, e2, "Lula shows her back in the video here (bib / RTMW / hair): does the 3D track turn with her?")
    order = {"absent": 0, "hidden": 1, "spin": 2}
    out.sort(key=lambda x: (order[x["kind"]], x["role"], x["start"]))
    for i, x in enumerate(out):
        x["id"] = f"c{i:02d}"
    return out


# ================================================================================================ drawing
def draw_skeleton(img: np.ndarray, uv: np.ndarray, thick: int = 2) -> np.ndarray:
    """Projected 22-joint skeleton: the body's LEFT side blue, RIGHT side red, spine white (BGR)."""
    import cv2
    for a, b in BONES22:
        if not (np.isfinite(uv[a]).all() and np.isfinite(uv[b]).all()):
            continue
        col = (255, 120, 0) if (a in LEFT22 or b in LEFT22) and not (a in RIGHT22 or b in RIGHT22) else \
            (0, 0, 255) if (a in RIGHT22 or b in RIGHT22) else (235, 235, 235)
        cv2.line(img, tuple(int(v) for v in uv[a]), tuple(int(v) for v in uv[b]), col, thick, cv2.LINE_AA)
    for j in (L_SHO, L_HIP):
        if np.isfinite(uv[j]).all():
            cv2.circle(img, tuple(int(v) for v in uv[j]), thick + 2, (255, 120, 0), -1, cv2.LINE_AA)
    for j in (R_SHO, R_HIP):
        if np.isfinite(uv[j]).all():
            cv2.circle(img, tuple(int(v) for v in uv[j]), thick + 2, (0, 0, 255), -1, cv2.LINE_AA)
    return img


def compass(size: int, rel_fit: float | None, cls_img: int | None, conf: float | None,
            rel_sam: float | None = None) -> np.ndarray:
    """Top view: the phone at the bottom, the dancer in the middle; arrows = where the torso faces (fit green, SAM 3D
    Body grey), the shaded sector = the image evidence's class."""
    import cv2
    img = np.full((size, size, 3), 32, np.uint8)
    c = (size // 2, size // 2 - 6)
    R = int(size * 0.36)
    if cls_img is not None and cls_img >= 0:
        # rel 0 = facing the phone (arrow down); sector centred on the class's rel
        centre = {0: 0.0, 1: 90.0, 2: 180.0}[int(cls_img)]
        half = {0: FRONT_MAX, 1: 30.0, 2: 180 - BACK_MIN}[int(cls_img)]
        col = (40, 140, 230) if conf is not None and conf >= CONF_MIN else (90, 90, 90)
        for sgn in ((1,) if cls_img != 1 else (1, -1)):
            a0 = 90 + sgn * centre - half
            a1 = 90 + sgn * centre + half
            cv2.ellipse(img, c, (R, R), 0, a0, a1, col, -1, cv2.LINE_AA)
    cv2.circle(img, c, R, (120, 120, 120), 1, cv2.LINE_AA)
    cv2.rectangle(img, (c[0] - 7, size - 14), (c[0] + 7, size - 4), (220, 220, 220), -1)   # the phone

    def arrow(rel, col, th):
        if rel is None or not np.isfinite(rel):
            return
        # direction in the top view: rel 0 -> towards the phone (down), +90 -> phone on the body's left
        a = math.radians(rel)
        d = (-math.sin(a), math.cos(a))
        tip = (int(c[0] + d[0] * R * 0.95), int(c[1] + d[1] * R * 0.95))
        cv2.arrowedLine(img, c, tip, col, th, cv2.LINE_AA, tipLength=0.3)
    arrow(rel_sam, (170, 170, 170), 1)
    arrow(rel_fit, (60, 220, 60), 2)
    return img


def _crop_box(mb, H: int, W: int, pad: float = 0.12) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = mb
    h = y1 - y0
    w = x1 - x0
    s = max(h, w * 1.6) * (1 + 2 * pad)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return int(cx - s / 3.2), int(cy - s / 2), int(cx + s / 3.2), int(cy + s / 2)


def keyframe_panel(T: Take, f: int, o: int, fz: dict, P: np.ndarray, ev: dict, H: int = 330) -> np.ndarray:
    """frame crop (own mask, bib / face boxes) | the same crop + the fit's projected skeleton | compass + text."""
    import cv2
    img = T.frame(f)
    own = T.mask(f, o)
    role = ROLES[o]
    mb = mask_box(own) if own is not None and own.any() else None
    if mb is None:
        uv = fz[f"{FIT}_uv"][f, o]
        ok = np.isfinite(uv).all(1)
        mb = (int(uv[ok, 0].min()), int(uv[ok, 1].min()), int(uv[ok, 0].max()), int(uv[ok, 1].max())) if ok.any() else (0, 0, T.W, T.H)
    box = _crop_box(mb, T.H, T.W)
    a = img.copy()
    if own is not None:
        cs, _ = cv2.findContours(own.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(a, cs, -1, COL[role], 1, cv2.LINE_AA)
    bb = ev["bib_box"][f, o]
    if bb[0] >= 0 and ev["bib_score"][f, o] >= CUE_THR["bib"]:
        cv2.rectangle(a, (int(bb[0]) - 2, int(bb[1]) - 2), (int(bb[2]) + 2, int(bb[3]) + 2), (0, 230, 255), 2)
    fxy = ev["face_xy"][f, o]
    if np.isfinite(fxy).all() and ev["face_score"][f, o] >= CUE_THR["face"]:
        cv2.circle(a, (int(fxy[0]), int(fxy[1])), 9, (255, 255, 0), 2, cv2.LINE_AA)
    b = draw_skeleton(img.copy(), fz[f"{FIT}_uv"][f, o].astype(float))
    ca, cb = crop_pad(a, box), crop_pad(b, box)
    s = H / ca.shape[0]
    w = int(round(ca.shape[1] * s))
    ca = cv2.resize(ca, (w, H), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    cb = cv2.resize(cb, (w, H), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    rel = fz[f"{FIT}_rel"][f, o]
    rs = fz["sam3db_rel"][f, o]
    p = P[f, o]
    ci = int(p.argmax())
    cmp_ = compass(150, rel, ci, float(p.max()), rs)
    col = np.full((H, 150, 3), 24, np.uint8)
    col[:150] = cmp_
    lines = [f"{f}  {T.t[f]:.2f}s", f"image: {CLASSES[ci]} {p.max():.2f}",
             f"fit: {'nan' if not np.isfinite(rel) else int(round(rel))} deg {CLASSES[int(fz[f'{FIT}_cls'][f, o])] if np.isfinite(rel) else ''}",
             f"sam3db: {'nan' if not np.isfinite(rs) else int(round(rs))} deg",
             f"bib {ev['bib_score'][f, o]:.2f}  ocr {'' if np.isnan(ev['ocr'][f, o]) else round(float(ev['ocr'][f, o]), 2)}",
             f"face {ev['face_score'][f, o]:.2f}  rho {ev['lr_rho'][f, o]:+.2f}",
             f"skin {ev['skin_face_share'][f, o]:.2f}" if "skin_face_share" in ev and np.isfinite(ev['skin_face_share'][f, o]) else "skin -"]
    for i, ln in enumerate(lines):
        cv2.putText(col, ln, (4, 166 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1, cv2.LINE_AA)
    return np.concatenate([ca, np.full((H, 3, 3), 255, np.uint8), cb, np.full((H, 3, 3), 255, np.uint8), col], 1)


def keyframes(start: int, end: int, n: int = 6, context: int = 4) -> list[int]:
    a, b = max(0, start - context), min(2642, end + context)
    return sorted(set(int(round(x)) for x in np.linspace(a, b, n)))


def window_sheets(rep: dict | None = None, max_windows: int = 40) -> list[str]:
    import cv2
    rep = rep or json.loads((OUT / "disagreements.json").read_text())
    pz = np.load(OUT / "posterior.npz")
    P = pz["post"]
    fz = dict(np.load(OUT / "fit_facing.npz", allow_pickle=True))
    ev = load_evidence()
    T = Take()
    SHEETS.mkdir(parents=True, exist_ok=True)
    for p in SHEETS.glob("*.jpg"):
        p.unlink()
    paths = []
    todo = rep["windows"][:max_windows] + [x for x in rep.get("check_windows", []) if x["kind"] == "absent"]
    for x in todo:
        o = ROLES.index(x["role"])
        ks = keyframes(x["start"], x["end"], 6)
        panels = [keyframe_panel(T, f, o, fz, P, ev) for f in ks]
        rows = []
        for i in range(0, len(panels), 2):
            pr = panels[i:i + 2]
            wmax = max(q.shape[1] for q in panels)
            pr = [np.pad(q, ((0, 0), (0, wmax - q.shape[1]), (0, 0))) for q in pr]
            if len(pr) == 1:
                pr.append(np.zeros_like(pr[0]))
            rows.append(np.concatenate([pr[0], np.full((pr[0].shape[0], 10, 3), 255, np.uint8), pr[1]], 1))
        sheet = np.concatenate([np.concatenate([r, np.full((8, r.shape[1], 3), 255, np.uint8)], 0) for r in rows], 0)
        head = np.full((40, sheet.shape[1], 3), 255, np.uint8)
        title = (f"{x['id']} {NAME[x['role']]} {x['t0']:.2f}-{x['t1']:.2f}s  {x['kind']}: image says {x['image_class']} "
                 f"(conf {x['conf_median']}), fit {x['fit_classes']}  " + (f"cues {x['cues']}" if "cues" in x else x.get("note", "")))
        cv2.putText(head, title[:150], (8, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        sheet = np.concatenate([head, sheet], 0)
        if sheet.shape[1] > 1900:
            s = 1900 / sheet.shape[1]
            sheet = cv2.resize(sheet, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        p = SHEETS / f"{x['id']}_{x['role']}_{x['t0']:06.2f}s_{x['kind']}.jpg"
        cv2.imwrite(str(p), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
        paths.append(str(p))
    log(f"{len(paths)} window sheets -> {SHEETS}")
    return paths


# ================================================================================================ targets for a refit
def targets(owner_labels: Path | None = None) -> Path:
    """facing_targets.json: per dancer, per frame, the TARGET torso facing from the image evidence (and the owner's
    spot-check labels, which win where present), as world yaw + band, the fit's yaw, the action and a confidence."""
    pz = np.load(OUT / "posterior.npz")
    P = pz["post"]
    fz = dict(np.load(OUT / "fit_facing.npz", allow_pickle=True))
    rep = json.loads((OUT / "disagreements.json").read_text())
    F = P.shape[0]
    t = fz["t"]
    own = {}
    ol = Path(owner_labels or OUT / "owner_labels.json")
    if ol.exists():
        for k, v in json.loads(ol.read_text()).get("labels", {}).items():
            r, f = k.split(":")
            if v.get("label") in CLS_INDEX:
                own[(r, int(f))] = CLS_INDEX[v["label"]]
    veto = set()
    vp = OUT / "agent_window_verdicts.json"
    if vp.exists():
        for w in json.loads(vp.read_text()).get("windows", []):
            if w.get("verdict") == "false_alarm":
                veto.update((w["role"], f) for f in range(int(w["start"]), int(w["end"]) + 1))
    doc = {"what": "target torso facing per frame from the IMAGE evidence (jl_facing), for a refit of john_lula",
           "fit": fz[f"{FIT}_src"].item() if hasattr(fz[f"{FIT}_src"], "item") else str(fz[f"{FIT}_src"]),
           "conventions": {
               "rel_deg": "signed angle about the torso's up axis from the torso forward to the pelvis->camera direction: "
                          "0 = faces the phone, +-180 = back to it, +90 = the phone on the dancer's left",
               "yaw_world_deg": "heading of the torso forward about world Y, atan2(x, z) (0 = +Z, 90 = +X)",
               "classes": f"front |rel| <= {FRONT_MAX}, side, back |rel| >= {BACK_MIN}",
               "torso": "forward = left x up; left = (L hip - R hip) + (L shoulder - R shoulder); up = mid shoulders - mid hips "
                        "(SMPL-X joints 1, 2, 16, 17)",
               "action": "keep = the fit agrees (or is within 20 deg of the image class's band); flip180 = the fit faces the opposite way "
                         "(front <-> back): add 180 deg of yaw (and re-solve the twist); rotate = front/back <-> side: rotate by "
                         "delta_into_band_deg; none = no confident image evidence (posterior < 0.8, the dancer hidden, or an image SIDE call: "
                         "side precision is only 0.2-0.4 on the agent labels, so side is never a target unless the owner labels it)",
               "confidence": "the image posterior of the target class (HMM over the calibrated cues); owner labels = 1.0",
               "source": "image | owner (facing/owner_labels.json, wins) | agent_review (a window the agent judged a false "
                         "alarm in agent_window_verdicts.json: action keep)",
               "use": "constrain only frames with action flip180 / rotate and confidence >= 0.8 (or owner); "
                      "tol_deg = the band half-width around target_yaw_world_deg"},
           "conf_min": CONF_MIN, "windows": [{k: x[k] for k in ("id", "role", "start", "end", "t0", "t1", "kind",
                                                                   "image_class", "conf_median")} for x in rep["windows"]],
           "dancers": {}}
    for o, r in enumerate(ROLES):
        cam = fz["cam_dir_yaw_world"][:, o]
        fy = fz[f"{FIT}_yaw_world"][:, o]
        fc = fz[f"{FIT}_cls"][:, o].astype(int)
        frel = fz[f"{FIT}_rel"][:, o]
        cols = {k: [] for k in ("frame", "t", "image_class", "confidence", "source", "fit_class", "fit_rel_deg",
                                "fit_yaw_world_deg", "action", "target_yaw_world_deg", "tol_deg", "delta_into_band_deg",
                                "flip180")}
        n_act = {"keep": 0, "flip180": 0, "rotate": 0, "none": 0}
        n_src = {}
        for f in range(F):
            p = P[f, o]
            ci, conf, src = int(p.argmax()), float(p.max()), "image"
            if (r, f) in own:
                ci, conf, src = own[(r, f)], 1.0, "owner"
            valid = np.isfinite(fy[f]) and np.isfinite(cam[f])
            if not valid or conf < CONF_MIN or (src == "image" and ci == 1):
                act = "none"          # no confident evidence (the image's side calls are not reliable)
            elif src == "image" and (r, f) in veto:
                act, src = "keep", "agent_review"   # the agent looked: the image call is a false alarm here
            elif fc[f] == ci or (src == "image" and p[max(fc[f], 0)] > FIT_P_MAX):
                act = "keep"
            elif {fc[f], ci} == {0, 2}:
                act = "flip180"
            else:
                act = "rotate"
            if valid:
                ty, half = target_yaw(ci, float(cam[f]), float(fy[f]))
                dl = delta_into_band(float(fy[f]), ty, half)
                if act == "rotate" and src == "image" and abs(dl) < BOUNDARY_DEG:
                    act = "keep"          # a boundary case (the fit is within BOUNDARY_DEG of the image class's band)
            else:
                ty = half = dl = None
            n_act[act] += 1
            n_src[src] = n_src.get(src, 0) + 1
            cols["frame"].append(f)
            cols["t"].append(round(float(t[f]), 4))
            cols["image_class"].append(CLASSES[ci])
            cols["confidence"].append(round(conf, 3))
            cols["source"].append(src)
            cols["fit_class"].append(CLASSES[fc[f]] if fc[f] >= 0 else None)
            cols["fit_rel_deg"].append(None if not np.isfinite(frel[f]) else round(float(frel[f]), 1))
            cols["fit_yaw_world_deg"].append(None if not np.isfinite(fy[f]) else round(float(fy[f]), 1))
            cols["action"].append(act)
            cols["target_yaw_world_deg"].append(None if ty is None else round(ty, 1))
            cols["tol_deg"].append(None if half is None else round(half, 1))
            cols["delta_into_band_deg"].append(None if dl is None else round(dl, 1))
            cols["flip180"].append(act == "flip180")
        doc["dancers"][r] = {"name": NAME[r], "actions": n_act, "sources": n_src, "columns": cols}
    p = OUT / "facing_targets.json"
    p.write_text(json.dumps(doc))
    log(f"targets -> {p}  " + "  ".join(f"{r}: {doc['dancers'][r]['actions']}" for r in ROLES))
    return p


# ================================================================================================ owner spot-check page
REVIEW_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Facing spot check</title>
<style>
:root { --bg:#f4f4f2; --fg:#1d1d1f; --mut:#6b6b70; --card:#ffffff; --line:#d8d8dc; --acc:#2266cc; --sel:#ffd34d; }
@media (prefers-color-scheme: dark) { :root { --bg:#17181a; --fg:#ececec; --mut:#9a9aa0; --card:#222326; --line:#3a3b3f; --acc:#7fb0ff; --sel:#8a6d00; } }
* { box-sizing:border-box; } body { margin:0; background:var(--bg); color:var(--fg); font:14px/1.4 system-ui, sans-serif; }
header { position:sticky; top:0; z-index:5; background:var(--card); border-bottom:1px solid var(--line); padding:8px 16px; display:flex; gap:16px; align-items:center; flex-wrap:wrap; }
header b { font-size:16px; } .mut { color:var(--mut); } #status { margin-left:auto; }
main { padding:12px 16px 80px; max-width:1500px; margin:0 auto; }
.win { background:var(--card); border:1px solid var(--line); border-radius:8px; margin:0 0 16px; padding:10px 12px; }
.win h2 { font-size:15px; margin:0 0 6px; } .kf { display:flex; gap:10px; overflow-x:auto; padding-bottom:6px; }
.k { flex:0 0 auto; width:236px; border:2px solid transparent; border-radius:6px; padding:4px; }
.k.cur { border-color:var(--acc); } .k img { width:100%; display:block; border-radius:4px; cursor:zoom-in; }
.k .meta { font-size:12px; color:var(--mut); margin:3px 0; } .btns { display:flex; gap:4px; }
.btns button { flex:1; padding:5px 0; border:1px solid var(--line); background:var(--bg); color:var(--fg); border-radius:4px; cursor:pointer; font-size:12px; }
.btns button.on { background:var(--sel); font-weight:600; }
#zoom { position:fixed; inset:0; background:rgba(0,0,0,.85); display:none; align-items:center; justify-content:center; z-index:10; }
#zoom img { max-width:96vw; max-height:94vh; }
kbd { border:1px solid var(--line); border-radius:3px; padding:0 4px; font-size:12px; }
</style></head><body>
<header><b>Facing spot check</b><span class="mut">John + Lula: each keyframe shows the frame (left) and the fit's skeleton
(body LEFT = blue, RIGHT = red). Does the dancer's FRONT, BACK or SIDE face the camera?</span>
<span class="mut"><kbd>F</kbd> front <kbd>B</kbd> back <kbd>S</kbd> side <kbd>U</kbd> unsure <kbd>&larr;</kbd><kbd>&rarr;</kbd> keyframe
<kbd>&uarr;</kbd><kbd>&darr;</kbd> window <kbd>Z</kbd> zoom</span><span id="status" class="mut">loading</span></header>
<main id="list"></main><div id="zoom"><img alt="zoomed keyframe"></div>
<script>
const NAMES = {lead: "John", follow: "Lula"};
let W = [], labels = {}, cur = [0, 0], saving = false;
const $ = (s, e = document) => e.querySelector(s);
function key(r, f) { return r + ":" + f; }
async function load() {
  W = (await (await fetch("windows.json")).json()).windows;
  try { labels = (await (await fetch("/api/labels")).json()).labels || {}; } catch (e) { labels = {}; }
  render(); mark(); status();
}
function render() {
  const L = $("#list"); L.innerHTML = "";
  W.forEach((w, i) => {
    const d = document.createElement("section"); d.className = "win"; d.id = "w" + i;
    const what = {flip180: "FLAGGED: front/back contradiction", rotate: "FLAGGED: front/back vs side", absent: "CHECK: interpolated fit",
                  hidden: "CHECK: hidden", spin: "CHECK: Lula's back in a spin"}[w.kind] || w.kind;
    d.innerHTML = `<h2>${w.id} - ${NAMES[w.role]} ${w.t0.toFixed(2)}-${w.t1.toFixed(2)} s - ${what} <span class="mut">(${w.note};
      image evidence: <b>${w.image_class}</b>, conf ${w.conf_median}; the fit: ${JSON.stringify(w.fit_classes)})</span></h2>
      <div class="kf"></div>`;
    const K = $(".kf", d);
    w.keyframes.forEach((k, j) => {
      const c = document.createElement("div"); c.className = "k"; c.dataset.w = i; c.dataset.k = j;
      c.innerHTML = `<img src="${k.img}" alt="frame ${k.frame}" loading="lazy"><div class="meta">frame ${k.frame} - ${k.t.toFixed(2)} s
        - fit ${k.fit_class} ${k.fit_rel === null ? "" : k.fit_rel + "&deg;"} - image ${k.image_class} ${k.image_conf}</div>
        <div class="btns">${["front", "back", "side", "unsure"].map(v => `<button data-v="${v}">${v.toUpperCase()}</button>`).join("")}</div>`;
      $("img", c).onclick = () => zoom(k.img_big || k.img);
      c.querySelectorAll("button").forEach(b => b.onclick = () => { cur = [i, j]; mark(); setLabel(b.dataset.v); });
      c.onclick = (e) => { if (e.target.tagName !== "BUTTON" && e.target.tagName !== "IMG") { cur = [i, j]; mark(); } };
      K.appendChild(c);
    });
    L.appendChild(d);
  });
  refreshButtons();
}
function refreshButtons() {
  document.querySelectorAll(".k").forEach(c => {
    const w = W[+c.dataset.w], k = w.keyframes[+c.dataset.k], v = (labels[key(w.role, k.frame)] || {}).label;
    c.querySelectorAll("button").forEach(b => b.classList.toggle("on", b.dataset.v === v));
  });
}
function mark() {
  document.querySelectorAll(".k.cur").forEach(c => c.classList.remove("cur"));
  const c = document.querySelector(`.k[data-w="${cur[0]}"][data-k="${cur[1]}"]`);
  if (c) { c.classList.add("cur"); c.scrollIntoView({block: "nearest", inline: "nearest"}); }
}
function status(msg) {
  const n = Object.keys(labels).length, tot = W.reduce((a, w) => a + w.keyframes.length, 0);
  $("#status").textContent = (msg ? msg + " - " : "") + `${n} labelled of ${tot} keyframes (${W.length} windows)`;
}
async function setLabel(v) {
  const w = W[cur[0]], k = w.keyframes[cur[1]];
  labels[key(w.role, k.frame)] = {label: v, window: w.id, t: k.t};
  refreshButtons();
  try {
    const r = await fetch("/api/labels", {method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({labels: {[key(w.role, k.frame)]: labels[key(w.role, k.frame)]}})});
    const j = await r.json(); status(r.ok ? "saved" : ("error: " + j.error));
  } catch (e) { status("NOT saved (server down?)"); }
  move(0, 1);
}
function move(dw, dk) {
  let [i, j] = cur;
  if (dw) { i = Math.max(0, Math.min(W.length - 1, i + dw)); j = 0; }
  if (dk) { j += dk; if (j >= W[i].keyframes.length) { if (i < W.length - 1) { i++; j = 0; } else j--; } if (j < 0) { if (i > 0) { i--; j = W[i].keyframes.length - 1; } else j = 0; } }
  cur = [i, j]; mark();
}
function zoom(src) { const z = $("#zoom"); $("img", z).src = src; z.style.display = "flex"; }
$("#zoom").onclick = () => $("#zoom").style.display = "none";
document.addEventListener("keydown", (e) => {
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  const k = e.key.toLowerCase();
  if ($("#zoom").style.display === "flex" && (k === "escape" || k === "z")) { $("#zoom").style.display = "none"; return; }
  const m = {f: "front", b: "back", s: "side", u: "unsure"};
  if (m[k]) { setLabel(m[k]); e.preventDefault(); }
  else if (k === "arrowright") { move(0, 1); e.preventDefault(); }
  else if (k === "arrowleft") { move(0, -1); e.preventDefault(); }
  else if (k === "arrowdown") { move(1, 0); e.preventDefault(); }
  else if (k === "arrowup") { move(-1, 0); e.preventDefault(); }
  else if (k === "z") { const w = W[cur[0]], kk = w.keyframes[cur[1]]; zoom(kk.img_big || kk.img); }
});
load();
</script></body></html>
"""


def build_review(max_windows: int = 60) -> Path:
    """facing/review/: index.html + windows.json + one thumbnail per keyframe (frame crop | fit skeleton)."""
    import cv2
    rep = json.loads((OUT / "disagreements.json").read_text())
    pz = np.load(OUT / "posterior.npz")
    P = pz["post"]
    fz = dict(np.load(OUT / "fit_facing.npz", allow_pickle=True))
    ev = load_evidence()
    T = Take()
    img_dir = REVIEW / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    for q in img_dir.glob("*.jpg"):
        q.unlink()
    wins = []
    for x in rep["windows"][:max_windows] + rep.get("check_windows", []):
        o = ROLES.index(x["role"])
        kfs = []
        n_kf = 6 if x["end"] - x["start"] < 45 else 8
        for f in keyframes(x["start"], x["end"], n_kf):
            img = T.frame(f)
            own = T.mask(f, o)
            mb = mask_box(own) if own is not None and own.any() else None
            if mb is None:
                uv = fz[f"{FIT}_uv"][f, o]
                ok = np.isfinite(uv).all(1)
                mb = (int(uv[ok, 0].min()), int(uv[ok, 1].min()), int(uv[ok, 0].max()), int(uv[ok, 1].max())) if ok.any() else (0, 0, T.W, T.H)
            box = _crop_box(mb, T.H, T.W)
            a = img.copy()
            if own is not None:
                cs, _ = cv2.findContours(own.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(a, cs, -1, COL[x["role"]], 1, cv2.LINE_AA)
            b = draw_skeleton(img.copy(), fz[f"{FIT}_uv"][f, o].astype(float))
            ca, cb = crop_pad(a, box), crop_pad(b, box)
            Hh = 420
            s = Hh / ca.shape[0]
            w = int(round(ca.shape[1] * s))
            ca = cv2.resize(ca, (w, Hh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
            cb = cv2.resize(cb, (w, Hh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
            pair = np.concatenate([ca, np.full((Hh, 4, 3), 255, np.uint8), cb], 1)
            name = f"{x['role']}_{f:04d}.jpg"
            cv2.imwrite(str(img_dir / name), pair, [cv2.IMWRITE_JPEG_QUALITY, 85])
            rel = fz[f"{FIT}_rel"][f, o]
            kfs.append({"frame": int(f), "t": round(float(T.t[f]), 3), "img": f"img/{name}",
                        "fit_class": CLASSES[int(fz[f'{FIT}_cls'][f, o])] if np.isfinite(rel) else "-",
                        "fit_rel": None if not np.isfinite(rel) else int(round(rel)),
                        "image_class": CLASSES[int(P[f, o].argmax())], "image_conf": round(float(P[f, o].max()), 2)})
        y = {k: x[k] for k in ("id", "role", "start", "end", "t0", "t1", "kind", "image_class", "conf_median", "fit_classes")}
        y["note"] = x.get("note", "confident image evidence contradicts the fit here")
        y["keyframes"] = kfs
        wins.append(y)
    (REVIEW / "windows.json").write_text(json.dumps({"windows": wins}, indent=1))
    (REVIEW / "index.html").write_text(REVIEW_HTML, encoding="utf-8")
    log(f"review page: {len(wins)} windows, {sum(len(w['keyframes']) for w in wins)} keyframes -> {REVIEW}")
    return REVIEW


VALID_LABELS = ("front", "back", "side", "unsure")


def merge_owner_labels(path: Path, incoming: dict) -> dict:
    """Validate {"labels": {"<role>:<frame>": {"label": front|back|side|unsure, ...}}} and merge it into the json at
    `path` (atomic write). Returns the merged document. Raises ValueError on bad input."""
    if not isinstance(incoming, dict) or not isinstance(incoming.get("labels"), dict):
        raise ValueError("body must be {'labels': {...}}")
    doc = json.loads(path.read_text()) if path.exists() else {
        "what": "the owner's facing spot-check labels (jl_facing review)", "labels": {}}
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    for k, v in incoming["labels"].items():
        try:
            r, f = str(k).split(":")
            f = int(f)
        except ValueError:
            raise ValueError(f"bad key {k!r}") from None
        if r not in ROLES or not 0 <= f < 100000:
            raise ValueError(f"bad key {k!r}")
        lab = v.get("label") if isinstance(v, dict) else v
        if lab not in VALID_LABELS:
            raise ValueError(f"bad label {lab!r}")
        rec = {"label": lab, "saved": now}
        if isinstance(v, dict):
            for extra in ("window", "t"):
                if extra in v and isinstance(v[extra], (str, int, float)):
                    rec[extra] = v[extra]
        doc["labels"][f"{r}:{f}"] = rec
    doc["updated"] = now
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=1))
    tmp.replace(path)
    return doc


def make_server(port: int = 8765, root: Path = REVIEW, labels_path: Path | None = None):
    """ThreadingHTTPServer on 127.0.0.1 serving `root`; GET/POST /api/labels read / merge owner_labels.json."""
    import functools
    import http.server
    labels_path = Path(labels_path or OUT / "owner_labels.json")

    class Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _json(self, code: int, obj) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.split("?")[0] == "/api/labels":
                doc = json.loads(labels_path.read_text()) if labels_path.exists() else {"labels": {}}
                return self._json(200, doc)
            return super().do_GET()

        def do_POST(self):
            if self.path.split("?")[0] != "/api/labels":
                return self._json(404, {"error": "not found"})
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0 or n > 1_000_000:
                return self._json(400, {"error": "bad length"})
            try:
                doc = merge_owner_labels(labels_path, json.loads(self.rfile.read(n).decode("utf-8")))
            except (ValueError, json.JSONDecodeError) as e:
                return self._json(400, {"error": str(e)})
            return self._json(200, {"ok": True, "n": len(doc["labels"])})

    handler = functools.partial(Handler, directory=str(root))
    return http.server.ThreadingHTTPServer(("127.0.0.1", int(port)), handler)


def review(port: int = 8765, build_only: bool = False, labels_path: Path | None = None) -> None:
    if build_only or not (REVIEW / "windows.json").exists():
        build_review()
    if build_only:
        return
    labels_path = Path(labels_path or OUT / "owner_labels.json")
    srv = make_server(port, REVIEW, labels_path)
    log(f"spot-check page: http://127.0.0.1:{port}/   (answers -> {labels_path}; Ctrl+C stops)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


# ================================================================================================ CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.mono4d.jl_facing")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("evidence", "skin", "labelsheets", "fit", "calibrate", "compare", "sheets", "targets", "all"):
        sub.add_parser(c)
    rv = sub.add_parser("review", help="the owner's spot-check page on 127.0.0.1")
    rv.add_argument("--port", type=int, default=8765)
    rv.add_argument("--build-only", action="store_true")
    rv.add_argument("--rebuild", action="store_true")
    rv.add_argument("--labels", default=None, help="answers file (default facing/owner_labels.json)")
    a = ap.parse_args(argv)
    if a.cmd in ("evidence", "all"):
        evidence()
    if a.cmd in ("skin", "all"):
        skin()
    if a.cmd == "labelsheets":
        labelsheets()
    if a.cmd in ("fit", "all"):
        fit_facing()
    if a.cmd in ("calibrate", "all"):
        calibrate()
    if a.cmd in ("compare", "all"):
        compare()
    if a.cmd in ("compare", "sheets", "all"):
        window_sheets()
        timeline_sheets()
    if a.cmd in ("targets", "all"):
        targets()
    if a.cmd == "all":
        build_review()
    if a.cmd == "review":
        if a.rebuild:
            build_review()
        review(a.port, a.build_only, a.labels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
