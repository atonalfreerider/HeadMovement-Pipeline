"""Follow hair / body split, anchored on the follow's pose keypoints (people/<cam>.npz, COCO-17).

The full follow mask (SAM) is split into hair/ and body/ (CONTRACTS.md). body/ feeds the 2D keypoint
crops, so it must never lose her face, hands or raised arms; hair/ must contain her hair only.

Geometry from the keypoints matched to her mask (HeadGeom):
    head size   median of shoulder-width / torso-length / eye-ear / eye-eye estimates (px, ~ head height)
    head centre face keypoints (or shoulders - one head up)
    face core   ellipse around nose + eyes (only when the face is confidently visible) -> always body
    arm zones   capsules elbow->wrist, upper arm, and a hand disk beyond each wrist      -> always body
Colour model (HairColour, per camera, pooled over frames): 3 Lab histograms (hair / skin / other) from
seed pixels - hair: loosely red/maroon pixels in the head zone outside face and arms; skin: forearms,
hands and face core; other: low-saturation torso/leg pixels (white top, light jeans). A pixel is hair-like
when its hair density beats the other two by `ratio`. This adapts to each camera's light (warm light turns
skin red under a fixed hue rule, dark maroon hair fails a fixed saturation rule). Too few seeds -> the
fixed rule dancers_appearance.red_hair_mask is used instead.
Spatial rule: hair = hair-like follow pixels outside face core and arm zones, in connected components
that touch the head zone (or chain to such a component through gaps < chain_gap x head size), within
`reach` x head size of the head centre. Everything else of hers is body - in particular arms raised above
her head and red-lit skin far from the head. No keypoints -> no hair (body = full mask).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .dancers_appearance import red_hair_mask, regions

NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP = range(13)


@dataclass
class HairConfig:
    kp_conf: float = 0.3
    face_conf: float = 0.5  # nose + one eye this confident -> face core is protected
    head_zone: float = 0.75  # head-zone disk radius / head size: hair must touch it
    reach: float = 2.8  # hair farther than this x head size from the head centre is not hair
    chain_gap: float = 0.12  # hair fragments this close (x head size) to kept hair join it
    upper_arm: float = 0.2  # capsule radii / head size
    forearm: float = 0.18
    hand: float = 0.4
    face_axes: tuple[float, float] = (0.3, 0.36)
    buried_zone: float = 0.6  # an arm zone whose follow pixels are > this hair-like is not protected
    close: float = 0.06  # closing kernel / head size (fills strand gaps inside the hair)
    bins: tuple[int, int, int] = (8, 32, 32)  # Lab histogram bins
    ratio: float = 1.5
    weak_ratio: float = 0.5  # weaklike() band (diagnostic; not used by split_hair)
    bridge_arms: bool = True  # hair-coloured pixels in arm zones connect hair (then are removed from it)
    min_seeds: int = 400
    seed_stride: int = 3
    min_hair_px: int = 12


@dataclass
class HeadGeom:
    size: float
    centre: np.ndarray
    face: tuple[float, float, float, float] | None  # cx, cy, ax, ay
    arms: list[tuple[np.ndarray, np.ndarray, float]]  # capsules (p0, p1, radius)
    forearms: list[tuple[np.ndarray, np.ndarray, float]]  # subset of arms + hands used as skin seeds
    hands: list[tuple[np.ndarray, float]]


def _pt(k: np.ndarray, j: int, conf: float):
    return k[j, :2].astype(float) if k[j, 2] > conf else None


def head_geom(k: np.ndarray | None, box_h: float | None = None, cfg: HairConfig | None = None) -> HeadGeom | None:
    """Head size / centre, face core and arm zones from COCO-17 keypoints (None when the head cannot be placed)."""
    cfg = cfg or HairConfig()
    if k is None:
        return None
    p = {j: _pt(k, j, cfg.kp_conf) for j in range(17)}
    est = []
    if p[L_SH] is not None and p[R_SH] is not None:
        est.append(0.6 * np.linalg.norm(p[L_SH] - p[R_SH]))
        if p[L_HIP] is not None and p[R_HIP] is not None:
            est.append(0.45 * np.linalg.norm((p[L_SH] + p[R_SH]) / 2 - (p[L_HIP] + p[R_HIP]) / 2))
    for e, r in ((L_EYE, L_EAR), (R_EYE, R_EAR)):
        if p[e] is not None and p[r] is not None:
            est.append(2.3 * np.linalg.norm(p[e] - p[r]))
    if p[L_EYE] is not None and p[R_EYE] is not None:
        est.append(3.5 * np.linalg.norm(p[L_EYE] - p[R_EYE]))
    if not est:
        return None
    size = float(np.median(est))
    if box_h:
        size = float(np.clip(size, 0.07 * box_h, 0.25 * box_h))
    size = max(size, 6.0)
    face_pts = [p[j] for j in (NOSE, L_EYE, R_EYE, L_EAR, R_EAR) if p[j] is not None]
    if face_pts:
        centre = np.mean(face_pts, 0) - (0, 0.05 * size)
    elif p[L_SH] is not None and p[R_SH] is not None:
        centre = (p[L_SH] + p[R_SH]) / 2 - (0, 0.9 * size)
    else:
        return None
    face = None
    eyes = [e for e in (L_EYE, R_EYE) if k[e, 2] > cfg.face_conf]
    if k[NOSE, 2] > cfg.face_conf and eyes:
        nose = k[NOSE, :2].astype(float)
        ey = float(np.mean([k[e, 1] for e in eyes]))
        cx = nose[0] if len(eyes) == 2 else (nose[0] + float(np.mean([k[e, 0] for e in eyes]))) / 2
        cy = (ey - 0.1 * size + nose[1] + 0.35 * size) / 2  # brows .. chin
        face = (float(cx), float(cy), cfg.face_axes[0] * size, cfg.face_axes[1] * size)
    arms, fore, hands = [], [], []
    for sh, el, wr in ((L_SH, L_EL, L_WR), (R_SH, R_EL, R_WR)):
        if p[sh] is not None and p[el] is not None:
            arms.append((p[sh], p[el], cfg.upper_arm * size))
        if p[el] is not None and p[wr] is not None:
            arms.append((p[el], p[wr], cfg.forearm * size))
            fore.append((p[el], p[wr], 0.7 * cfg.forearm * size))
        if p[wr] is not None:
            d = (p[wr] - p[el]) if p[el] is not None else np.zeros(2)
            n = np.linalg.norm(d)
            c = p[wr] + (0.4 * size * d / n if n > 1 else 0)
            hands.append((c, cfg.hand * size))
            arms.append((p[wr], c, 0.6 * cfg.hand * size))
    return HeadGeom(size, np.asarray(centre, float), face, arms, fore, hands)


def _canvas(shape) -> np.ndarray:
    return np.zeros(shape[:2], np.uint8)


def face_mask(g: HeadGeom, shape) -> np.ndarray:
    m = _canvas(shape)
    if g.face is not None:
        cx, cy, ax, ay = g.face
        cv2.ellipse(m, (int(round(cx)), int(round(cy))), (max(1, int(ax)), max(1, int(ay))), 0, 0, 360, 1, -1)
    return m.astype(bool)


def arm_mask(g: HeadGeom, shape, forearms_only: bool = False) -> np.ndarray:
    m = _canvas(shape)
    for p0, p1, r in (g.forearms if forearms_only else g.arms):
        cv2.line(m, tuple(int(round(v)) for v in p0), tuple(int(round(v)) for v in p1), 1, max(1, int(round(2 * r))))
    for c, r in g.hands:
        cv2.circle(m, tuple(int(round(v)) for v in c), max(1, int(round(r * (0.7 if forearms_only else 1.0)))), 1, -1)
    return m.astype(bool)


def arm_zones(g: HeadGeom, shape) -> list[np.ndarray]:
    """Each arm capsule / hand disk as its own mask (see split_hair: zones buried in hair are not protected)."""
    zones = []
    for p0, p1, r in g.arms:
        m = _canvas(shape)
        cv2.line(m, tuple(int(round(v)) for v in p0), tuple(int(round(v)) for v in p1), 1, max(1, int(round(2 * r))))
        zones.append(m.astype(bool))
    for c, r in g.hands:
        zones.append(disk(shape, c, r))
    return zones


def disk(shape, centre, r: float) -> np.ndarray:
    m = _canvas(shape)
    cv2.circle(m, (int(round(centre[0])), int(round(centre[1]))), max(1, int(round(r))), 1, -1)
    return m.astype(bool)


def _roi(follow: np.ndarray, kpts: np.ndarray | None):
    """Bounding box of the follow mask (everything here is intersected with it) and keypoints shifted into it."""
    ys = np.nonzero(follow.any(1))[0]
    xs = np.nonzero(follow.any(0))[0]
    sl = (slice(ys[0], ys[-1] + 1), slice(xs[0], xs[-1] + 1))
    if kpts is not None:
        kpts = kpts.copy()
        kpts[:, 0] -= xs[0]
        kpts[:, 1] -= ys[0]
    return sl, kpts


def _loose_red(hsv: np.ndarray) -> np.ndarray:
    h, s = hsv[..., 0], hsv[..., 1]
    return ((h <= 15) | (h >= 150)) & (s >= 50)


class HairColour:
    """Per-camera hair / skin / other Lab colour model (see module doc)."""

    CLASSES = ("hair", "skin", "other")

    def __init__(self, cfg: HairConfig | None = None):
        self.cfg = cfg or HairConfig()
        self.hist = {c: np.zeros(self.cfg.bins) for c in self.CLASSES}
        self.lut: np.ndarray | None = None
        self.weak_lut: np.ndarray | None = None

    def _bins(self, lab: np.ndarray) -> tuple[np.ndarray, ...]:
        b = self.cfg.bins
        return tuple((lab[..., c].astype(np.int32) * b[c]) >> 8 for c in range(3))

    def _add(self, cls: str, lab_px: np.ndarray) -> None:
        if len(lab_px):
            np.add.at(self.hist[cls], self._bins(lab_px), 1)

    def add_frame(self, img: np.ndarray, follow: np.ndarray, g: HeadGeom | None, kpts: np.ndarray | None) -> None:
        if g is None or not follow.any():
            return
        sl, kpts = _roi(follow, kpts)
        img, follow = img[sl], follow[sl]
        g = head_geom(kpts, float(follow.shape[0]), self.cfg)
        if g is None:
            return
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        face, arms = face_mask(g, img.shape), arm_mask(g, img.shape)
        head = disk(img.shape, g.centre, self.cfg.head_zone * g.size)
        self._add("hair", lab[follow & head & ~face & ~arms & _loose_red(hsv)])
        self._add("skin", lab[follow & ((arm_mask(g, img.shape, True) & ~head) | face)])
        ys, xs = np.nonzero(follow)
        box = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], float)
        reg = regions(box, kpts, follow.shape)
        body = (reg["torso"] | reg["legs"]) & follow & ~head & ~arms & (hsv[..., 1] < 70)
        self._add("other", lab[body])

    def fit(self) -> bool:
        """Builds the hair-like LUT; False (and no LUT) when a class has too few seeds."""
        from scipy.ndimage import gaussian_filter
        n = {c: h.sum() for c, h in self.hist.items()}
        if min(n["hair"], n["skin"]) < self.cfg.min_seeds:
            self.lut = None
            return False
        dens = {c: gaussian_filter(h, 1.0) / max(n[c], 1) + 1e-7 for c, h in self.hist.items()}
        body = np.maximum(dens["skin"], dens["other"])
        self.lut = (dens["hair"] > self.cfg.ratio * body) & (dens["hair"] > 1e-5)
        # hysteresis band: colours not clearly skin/clothes (sunlit or shadowed hair ends) may extend hair
        self.weak_lut = self.lut | (dens["hair"] > self.cfg.weak_ratio * body)
        return True

    @property
    def seeds(self) -> dict[str, int]:
        return {c: int(h.sum()) for c, h in self.hist.items()}

    def hairlike(self, img: np.ndarray) -> np.ndarray:
        if self.lut is None:
            return red_hair_mask(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        return self.lut[self._bins(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))]

    def weaklike(self, img: np.ndarray) -> np.ndarray:
        """Hair-or-ambiguous colours (hysteresis band); the fixed rule's loose red/maroon without a model."""
        if self.weak_lut is None:
            return _loose_red(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        return self.weak_lut[self._bins(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))]


def split_hair(img: np.ndarray, follow: np.ndarray, kpts: np.ndarray | None, colour: HairColour | None = None,
               cfg: HairConfig | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(hair, body) of the follow's full mask; disjoint, hair | body == follow."""
    cfg = cfg or (colour.cfg if colour is not None else HairConfig())
    empty = np.zeros_like(follow)
    if not follow.any():
        return empty, follow.copy()
    sl, kpts = _roi(follow, kpts)
    full_follow, img, follow = follow, img[sl], follow[sl]
    g = head_geom(kpts, float(follow.shape[0]), cfg)
    if g is None:
        return empty, full_follow.copy()
    like = colour.hairlike(img) if colour is not None else red_hair_mask(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    # arm zones protect her arms and hands, except a zone that is mostly hair-coloured: in a back view
    # the pose model guesses hidden wrists onto her back, under her hair
    arms = np.zeros_like(follow)
    for z in arm_zones(g, img.shape):
        zf = z & follow
        if zf.any() and like[zf].mean() <= cfg.buried_zone:
            arms |= z
    allowed = follow & ~face_mask(g, img.shape) & disk(img.shape, g.centre, cfg.reach * g.size)
    if not cfg.bridge_arms:
        allowed &= ~arms
    cand = (like & allowed).astype(np.uint8)
    k = max(3, int(round(cfg.close * g.size)) | 1)
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    cand &= allowed.astype(np.uint8)
    n, lab = cv2.connectedComponents(cand, connectivity=8)
    if n <= 1:
        return empty, full_follow.copy()
    keep = np.zeros(n, bool)
    keep[np.unique(lab[disk(img.shape, g.centre, cfg.head_zone * g.size) & (lab > 0)])] = True
    gap = max(1, int(round(cfg.chain_gap * g.size)))
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * gap + 1, 2 * gap + 1))
    for _ in range(4):
        if not keep[1:].any():
            break
        grown = cv2.dilate(keep[lab].astype(np.uint8), kern).astype(bool)
        new = np.zeros(n, bool)
        new[np.unique(lab[grown & (lab > 0)])] = True
        if not (new & ~keep).any():
            break
        keep |= new
    keep[0] = False
    strong = keep[lab] & ~arms  # arm pixels only bridge hair hanging behind an arm
    if strong.sum() < cfg.min_hair_px:
        return empty, full_follow.copy()
    hair = empty
    hair[sl] = strong
    return hair, full_follow & ~hair
