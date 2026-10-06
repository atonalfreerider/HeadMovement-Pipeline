"""Masked learned features: ALIKED keypoints + LightGlue matching (both BSD-3 / Apache-2 weights).

pip pycolmap 4.2.1 on Windows has no learned features, so extraction and matching run in PyTorch
(https://github.com/cvg/LightGlue). People, phone screens and captions (people_masks*/, static_masks/) are
removed in two steps: masked pixels are PRE-FILLED with a smooth low-resolution inpaint before extraction
(zero-filling would create a ring of strong edge keypoints around every person), then keypoints inside the
mask dilated by `drop_dilate_px` are dropped. Up to 8192 keypoints per image, fp16 on Turing.

Features are cached per image in cameras/cache/feats/<cam>/<kind>_<stem>.npz (keypoints in native pixels).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class ImageRec:
    """One image of the take: a window frame (kind 'win', index = canonical frame index) or a keyframe."""
    cam: str
    kind: str  # 'win' | 'kf'
    index: int  # canonical window frame index, or keyframe ordinal
    t: float  # reference seconds
    name: str  # path relative to work/<take>, forward slashes (COLMAP image name)
    mask: str  # path relative to work/<take> of the people mask ('' if none)

    @property
    def key(self) -> str:
        return f"{self.cam}/{self.kind}_{Path(self.name).stem}"


def list_images(work: Path, cams: list[str]) -> dict[str, list[ImageRec]]:
    """All window frames and keyframes per camera, in time order."""
    out: dict[str, list[ImageRec]] = {}
    for cam in cams:
        recs = []
        meta = json.loads((work / "frames" / cam / "frames.json").read_text())
        for i, f in enumerate(meta["frames"]):
            m = f"people_masks/{cam}/{Path(f['file']).stem}.png"
            recs.append(ImageRec(cam, "win", i, float(f["t"]), f"frames/{cam}/{f['file']}",
                                 m if (work / m).exists() else ""))
        kf_json = work / "keyframes" / cam / "keyframes.json"
        if kf_json.exists():
            for j, f in enumerate(json.loads(kf_json.read_text())):
                m = f"people_masks_kf/{cam}/{Path(f['file']).stem}.png"
                recs.append(ImageRec(cam, "kf", j, float(f["t"]), f"keyframes/{cam}/{f['file']}",
                                     m if (work / m).exists() else ""))
        out[cam] = sorted(recs, key=lambda r: (r.t, r.kind))
    return out


def load_mask(work: Path, rec: ImageRec, shape: tuple[int, int]) -> np.ndarray:
    """uint8 0/255 ignore mask (people + per-camera static mask) at image resolution."""
    h, w = shape
    mask = np.zeros((h, w), np.uint8)
    if rec.mask:
        m = cv2.imread(str(work / rec.mask), cv2.IMREAD_GRAYSCALE)
        if m is not None:
            if m.shape != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            mask = np.maximum(mask, m)
    static = work / "static_masks" / rec.cam / "ignore.png"  # captions etc. (mirror.png is kept: rigid virtual scene)
    if static.exists():
        m = cv2.imread(str(static), cv2.IMREAD_GRAYSCALE)
        mask = np.maximum(mask, cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST))
    return (mask > 127).astype(np.uint8) * 255


def prefill(img: np.ndarray, mask: np.ndarray, work_px: int = 256) -> np.ndarray:
    """Replace masked pixels by a smooth inpaint computed at low resolution (no hard edges, no texture)."""
    if not mask.any():
        return img
    h, w = mask.shape
    s = work_px / max(h, w)
    small = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    msmall = cv2.resize(mask, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
    msmall = cv2.dilate(msmall, np.ones((3, 3), np.uint8))
    fill = cv2.inpaint(small, msmall, 5, cv2.INPAINT_TELEA)
    fill = cv2.GaussianBlur(fill, (0, 0), 2.0)
    fill = cv2.resize(fill, (w, h), interpolation=cv2.INTER_LINEAR)
    # feather the seam so the mask border itself is not a step edge
    alpha = cv2.GaussianBlur((mask > 0).astype(np.float32), (0, 0), 3.0)[..., None]
    alpha = np.maximum(alpha, (mask > 0)[..., None].astype(np.float32))
    return (img * (1 - alpha) + fill * alpha).astype(np.uint8)


def keep_outside(kpts: np.ndarray, mask: np.ndarray, dilate_px: int) -> np.ndarray:
    """Boolean (N,) - keypoint lies outside the mask dilated by dilate_px."""
    if len(kpts) == 0:
        return np.zeros(0, bool)
    m = mask
    if dilate_px > 0:
        m = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1,) * 2))
    h, w = m.shape
    x = np.clip(np.round(kpts[:, 0]).astype(int), 0, w - 1)
    y = np.clip(np.round(kpts[:, 1]).astype(int), 0, h - 1)
    return m[y, x] == 0


@dataclass
class FeatureSettings:
    max_keypoints: int = 8192
    detection_threshold: float = 0.01  # low: the top-k keeps the best ones anyway
    long_side: int = 1296  # 480p frames are UP-sampled to this long side (x1.5: 3-4x more keypoints on the
    #                        feature-poor white walls); HD frames (long side >= this) are extracted natively
    drop_dilate_px: int = 6
    border_px: int = 4


class Features:
    """ALIKED extractor + LightGlue matcher with an on-disk cache (lazy GPU init)."""

    def __init__(self, work: Path, cache: Path, settings: FeatureSettings | None = None, device: str = "cuda"):
        self.work, self.cache, self.s, self.device = work, cache, settings or FeatureSettings(), device
        self._extractor = self._matcher = None
        self._mem: dict[str, dict] = {}

    # -- extraction ---------------------------------------------------------
    def _ext(self):
        if self._extractor is None:
            from lightglue import ALIKED
            self._extractor = ALIKED(max_num_keypoints=self.s.max_keypoints,
                                     detection_threshold=self.s.detection_threshold).eval().to(self.device)
        return self._extractor

    def path(self, rec: ImageRec) -> Path:
        return self.cache / "feats" / f"{rec.key}.npz"

    def get(self, rec: ImageRec) -> dict:
        """{'kpts' (N,2) f32 native px, 'scores' (N,), 'desc' (N,128) f16, 'size' (w,h)}."""
        if rec.key in self._mem:
            return self._mem[rec.key]
        p = self.path(rec)
        if p.exists():
            d = dict(np.load(p))
        else:
            d = self.extract(rec)
            p.parent.mkdir(parents=True, exist_ok=True)
            np.savez(p, **d)
        if len(self._mem) > 600:
            self._mem.pop(next(iter(self._mem)))
        self._mem[rec.key] = d
        return d

    def extract(self, rec: ImageRec) -> dict:
        import torch
        img = cv2.imread(str(self.work / rec.name), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(self.work / rec.name)
        h, w = img.shape[:2]
        mask = load_mask(self.work, rec, (h, w))
        filled = prefill(img, mask)
        rgb = cv2.cvtColor(filled, cv2.COLOR_BGR2RGB)
        ten = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).to(self.device)
        resize = self.s.long_side if max(h, w) < self.s.long_side else None
        from ..gpu import amp_dtype
        with torch.inference_mode(), torch.autocast("cuda", dtype=amp_dtype(), enabled=self.device == "cuda"):
            f = self._ext().extract(ten, resize=resize)
        k = f["keypoints"][0].float().cpu().numpy()
        sc = f["keypoint_scores"][0].float().cpu().numpy()
        de = f["descriptors"][0].float().cpu().numpy()
        b = self.s.border_px
        keep = keep_outside(k, mask, self.s.drop_dilate_px)
        keep &= (k[:, 0] >= b) & (k[:, 1] >= b) & (k[:, 0] <= w - 1 - b) & (k[:, 1] <= h - 1 - b)
        return {"kpts": k[keep].astype(np.float32), "scores": sc[keep].astype(np.float32),
                "desc": de[keep].astype(np.float16), "size": np.array([w, h], np.int32)}

    # -- matching -----------------------------------------------------------
    def _mat(self):
        if self._matcher is None:
            from lightglue import LightGlue
            self._matcher = LightGlue(features="aliked", mp=self.device == "cuda", flash=True,
                                      filter_threshold=0.1).eval().to(self.device)
        return self._matcher

    def match(self, a: ImageRec, b: ImageRec, min_score: float = 0.0) -> np.ndarray:
        """(M, 2) uint32 index pairs into get(a)['kpts'], get(b)['kpts']."""
        import torch
        fa, fb = self.get(a), self.get(b)
        if len(fa["kpts"]) < 8 or len(fb["kpts"]) < 8:
            return np.zeros((0, 2), np.uint32)

        def t(f):
            return {"keypoints": torch.from_numpy(f["kpts"])[None].to(self.device),
                    "descriptors": torch.from_numpy(f["desc"].astype(np.float32))[None].to(self.device),
                    "image_size": torch.from_numpy(f["size"].astype(np.float32))[None].to(self.device)}
        with torch.inference_mode():
            out = self._mat()({"image0": t(fa), "image1": t(fb)})
        m = out["matches"][0].cpu().numpy()
        s = out["scores"][0].float().cpu().numpy()
        return m[s >= min_score].astype(np.uint32).reshape(-1, 2)
