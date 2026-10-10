"""Masked ALIKED + LightGlue features and the match graph for dancecap.cameras.ptz (GPU; local or on a pod).

People are removed with a YOLO11 person segmentation (dilated), the burnt-in watermark / logo corner of a video with
a fixed rectangle (a watermark is fixed in the SCREEN: its keypoints would pin the homography to the identity).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import ptz as P


@dataclass
class FeatSettings:
    max_keypoints: int = 4096
    detection_threshold: float = 0.01
    mask_dilate_px: int = 15
    border_px: int = 4


class Matcher:
    """ALIKED extractor + LightGlue matcher (lazy GPU init)."""

    def __init__(self, device: str = "cuda", settings: FeatSettings | None = None):
        self.dev = device
        self.s = settings or FeatSettings()
        self._ext = self._mat = None

    def _extractor(self):
        if self._ext is None:
            from lightglue import ALIKED
            self._ext = ALIKED(max_num_keypoints=self.s.max_keypoints,
                               detection_threshold=self.s.detection_threshold).eval().to(self.dev)
        return self._ext

    def _matcher(self):
        if self._mat is None:
            from lightglue import LightGlue
            self._mat = LightGlue(features="aliked", mp=self.dev == "cuda", flash=True,
                                  filter_threshold=0.1).eval().to(self.dev)
        return self._mat

    def extract(self, img_bgr: np.ndarray, ignore: np.ndarray | None) -> dict:
        """{'kpts' (N,2) px, 'desc' (N,128) f32, 'size' (2,)}; keypoints inside the ignore mask are dropped."""
        import torch
        h, w = img_bgr.shape[:2]
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        ten = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).to(self.dev)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=self.dev == "cuda"):
            o = self._extractor().extract(ten)
        k = o["keypoints"][0].float().cpu().numpy()
        d = o["descriptors"][0].float().cpu().numpy()
        keep = np.ones(len(k), bool)
        if ignore is not None:
            m = ignore
            if self.s.mask_dilate_px:
                m = cv2.dilate(m, np.ones((self.s.mask_dilate_px,) * 2, np.uint8))
            x = np.clip(k[:, 0].astype(int), 0, w - 1)
            y = np.clip(k[:, 1].astype(int), 0, h - 1)
            keep &= m[y, x] == 0
        b = self.s.border_px
        keep &= (k[:, 0] >= b) & (k[:, 1] >= b) & (k[:, 0] <= w - 1 - b) & (k[:, 1] <= h - 1 - b)
        return {"kpts": k[keep].astype(np.float32), "desc": d[keep].astype(np.float32),
                "size": np.array([w, h], np.float32)}

    def match(self, fa: dict, fb: dict) -> np.ndarray:
        import torch
        if len(fa["kpts"]) < 8 or len(fb["kpts"]) < 8:
            return np.zeros((0, 2), np.int64)

        def t(f):
            return {"keypoints": torch.from_numpy(f["kpts"])[None].to(self.dev),
                    "descriptors": torch.from_numpy(f["desc"])[None].to(self.dev),
                    "image_size": torch.from_numpy(f["size"])[None].to(self.dev)}
        with torch.inference_mode():
            out = self._matcher()({"image0": t(fa), "image1": t(fb)})
        return out["matches"][0].cpu().numpy().astype(np.int64).reshape(-1, 2)


def static_ignore(shape: tuple[int, int], person: np.ndarray | None, rects: list[tuple[int, int, int, int]]) -> np.ndarray:
    """uint8 (h, w): 255 where features must be ignored; `rects` = (x0, y0, x1, y1) screen-fixed overlays."""
    h, w = shape
    m = np.zeros((h, w), np.uint8)
    if person is not None:
        m = np.maximum(m, person if person.shape == (h, w) else cv2.resize(person, (w, h), interpolation=cv2.INTER_NEAREST))
    for x0, y0, x1, y1 in rects:
        m[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = 255
    return m


def pair_inliers(fa: dict, fb: dict, m: np.ndarray, thresh: float = 4.0):
    """Homography-RANSAC inlier correspondences of a matched pair: (p, q, H) or None."""
    if len(m) < 25:
        return None
    p, q = fa["kpts"][m[:, 0]], fb["kpts"][m[:, 1]]
    H, inl = P.homography_edge(p, q, thresh)
    if H is None or inl.sum() < 25:
        return None
    return p[inl], q[inl], H


_YOLO: dict = {}


def person_masks_yolo(frames_bgr: list[np.ndarray], weights: str, imgsz: int = 1280, conf: float = 0.2) -> list[np.ndarray]:
    """YOLO11-seg person masks (uint8 255) at the size of each frame."""
    from ultralytics import YOLO
    model = _YOLO.get(weights)
    if model is None:
        model = _YOLO[weights] = YOLO(weights)
    out = []
    res = model(frames_bgr, classes=[0], imgsz=imgsz, conf=conf, verbose=False)
    for im, r in zip(frames_bgr, res):
        h, w = im.shape[:2]
        mk = np.zeros((h, w), np.uint8)
        if r.masks is not None:
            for pm in r.masks.data.cpu().numpy():
                mk[cv2.resize(pm, (w, h), interpolation=cv2.INTER_LINEAR) > 0.5] = 255
        out.append(mk)
    return out


def read_video_frames(video: str, width: int, height: int, step: int = 1, start: int = 0, count: int | None = None,
                      frames: list[int] | None = None):
    """Yield (frame_index, bgr) for every `step`-th frame from `start`, or for the explicit `frames` (sorted, unique).
    OpenCV decodes sequentially and converts only the frames that are wanted (an ffmpeg select expression with
    hundreds of terms is refused by ffmpeg)."""
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise FileNotFoundError(video)
    want = sorted(set(int(k) for k in frames)) if frames is not None else None
    wanted = set(want) if want is not None else None
    last = want[-1] if want else None
    n = 0
    emitted = 0
    try:
        while True:
            if last is not None and n > last:
                break
            if not cap.grab():
                break
            take = (n in wanted) if wanted is not None else (n >= start and (n - start) % step == 0)
            if take:
                ok, img = cap.retrieve()
                if not ok:
                    break
                yield n, img
                emitted += 1
                if count and emitted >= count:
                    break
            n += 1
    finally:
        cap.release()
