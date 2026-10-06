"""All-person detection, COCO-17 keypoints and spectator ("ignore") masks for every camera.

Three Ultralytics models run on the same batch of canonical frames (fp16, person class only), each at two
scales (640 px and 960/1280 px, merged by merge_scales: up-sampled 480p frames make YOLO cut off the feet
of people who fill the frame, while 640 px misses the seated far crowd):
  detection     yolo12x.pt       -> people/<cam>.npz boxes/scores (the contract's YOLO detections)
  pose          yolo11x-pose.pt  -> COCO-17 keypoints, attached to detection boxes by IoU
  segmentation  yolo11x-seg.pt   -> instance masks; their union (plus boxes no mask covered, phones held
                                    up by spectators) is dilated ~10 px into people_masks/<cam>/NNNNNN.png
                                    (255 = person / ignore for static geometry)
Keyframes get the same mask (no pose) in people_masks_kf/<cam>/<ref_ms>.png.

Low confidence (0.2) on purpose: seated, half-occluded spectators must be masked; a few extra boxes are
harmless here because the dancers stage picks the two dancers. Mirror (cam09) and phone-screen (cam04)
reflections of people are masked too, which is what static geometry wants.

Licence: yolo11x-seg / yolo11x-pose / yolo12x weights and the ultralytics package are AGPL-3.0.
Weights live in dancecap/weights/ (git-ignored); seg and pose are auto-downloaded there by ultralytics.

    python -m dancecap detect larissa_kadu [--cams 01,02] [--no-keyframes]
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .take import PACKAGE_ROOT, WORKSPACE, Take

WEIGHTS = PACKAGE_ROOT / "weights"
DET_WEIGHTS = WORKSPACE / "mamma" / "data" / "weights" / "yolo" / "yolo12x.pt"
SEG_WEIGHTS = WEIGHTS / "yolo11x-seg.pt"
POSE_WEIGHTS = WEIGHTS / "yolo11x-pose.pt"

PERSON, CELL_PHONE = 0, 67  # COCO class ids
N_KPTS = 17


@dataclass
class Settings:
    conf: float = 0.2  # detection / pose confidence (catches seated spectators)
    seg_conf: float = 0.15  # mask union errs towards masking more
    phone_conf: float = 0.25  # cell phones held up by spectators (screens show the dancers)
    iou: float = 0.6  # NMS IoU; dancers in an embrace overlap heavily, so keep this high
    dilate_px: int = 10
    kpt_match_iou: float = 0.5
    hd_imgsz: int = 1280  # long side >= 1280 px (cams 02, 06)
    sd_imgsz: int = 960  # 480p phones
    lo_imgsz: int = 640  # second, low-res pass: people filling the frame (see merge_scales)
    large_frac: float = 0.35
    hd_batch: int = 4
    sd_batch: int = 8

    def imgsz(self, width: int, height: int) -> int:
        return self.hd_imgsz if max(width, height) >= 1280 else self.sd_imgsz

    def scales(self, width: int, height: int) -> list[int]:
        return [self.lo_imgsz, self.imgsz(width, height)]

    def batch(self, width: int, height: int) -> int:
        return self.hd_batch if max(width, height) >= 1280 else self.sd_batch


@dataclass
class FrameResult:
    boxes: np.ndarray  # (N, 4) xyxy px, person detections
    scores: np.ndarray  # (N,)
    kpts: np.ndarray  # (N, 17, 3) x, y, conf (zeros where no pose matched)
    mask: np.ndarray  # (H, W) uint8 0/255 union ignore mask (dilated)


# ---------------------------------------------------------------------------
# pure helpers (unit-tested without a GPU)
# ---------------------------------------------------------------------------

def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU of xyxy boxes a (N,4) and b (M,4) -> (N, M)."""
    a = np.asarray(a, float).reshape(-1, 4)
    b = np.asarray(b, float).reshape(-1, 4)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), axis=2)
    area_a = np.prod(np.clip(a[:, 2:] - a[:, :2], 0, None), axis=1)
    area_b = np.prod(np.clip(b[:, 2:] - b[:, :2], 0, None), axis=1)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def greedy_match(iou: np.ndarray, min_iou: float) -> list[tuple[int, int]]:
    """One-to-one matching by descending IoU; pairs below min_iou are left unmatched."""
    pairs, used_r, used_c = [], set(), set()
    if iou.size == 0:
        return pairs
    for flat in np.argsort(-iou, axis=None):
        r, c = np.unravel_index(flat, iou.shape)
        if iou[r, c] < min_iou:
            break
        if r in used_r or c in used_c:
            continue
        pairs.append((int(r), int(c)))
        used_r.add(r)
        used_c.add(c)
    return pairs


def attach_keypoints(det_boxes: np.ndarray, pose_boxes: np.ndarray, pose_kpts: np.ndarray,
                     min_iou: float = 0.5) -> np.ndarray:
    """COCO-17 keypoints for each detection box from the best-overlapping pose detection (zeros if none)."""
    out = np.zeros((len(det_boxes), N_KPTS, 3), np.float32)
    for i, j in greedy_match(iou_matrix(det_boxes, pose_boxes), min_iou):
        out[i] = pose_kpts[j]
    return out


def uncovered_boxes(boxes: np.ndarray, mask: np.ndarray, min_cover: float = 0.3,
                    max_area: float = 0.06) -> np.ndarray:
    """Boxes whose area is covered less than min_cover by mask (persons the segmenter missed).

    Only small boxes (<= max_area of the image) qualify: the fallback is for distant / seated spectators.
    Big boxes are near-camera people the segmenter does see (or the operator's own hand, whose box spans
    lots of floor), and an ellipse over them would wipe out static floor.
    """
    keep = []
    h, w = mask.shape
    for b in np.asarray(boxes).reshape(-1, 4):
        x0, y0 = int(max(0, np.floor(b[0]))), int(max(0, np.floor(b[1])))
        x1, y1 = int(min(w, np.ceil(b[2]))), int(min(h, np.ceil(b[3])))
        if x1 <= x0 or y1 <= y0 or (x1 - x0) * (y1 - y0) > max_area * h * w:
            continue
        if (mask[y0:y1, x0:x1] > 0).mean() < min_cover:
            keep.append(b)
    return np.asarray(keep, float).reshape(-1, 4)


def fill_boxes(mask: np.ndarray, boxes: np.ndarray, ellipse: bool = True) -> np.ndarray:
    """Paint boxes into mask (in place): an inscribed ellipse for people (less background lost), else a rect."""
    for b in np.asarray(boxes).reshape(-1, 4):
        x0, y0, x1, y1 = (int(round(v)) for v in b)
        if ellipse:
            cv2.ellipse(mask, (((x0 + x1) / 2, (y0 + y1) / 2), (max(1, x1 - x0), max(1, y1 - y0)), 0.0), 255, -1)
        else:
            cv2.rectangle(mask, (x0, y0), (x1, y1), 255, -1)
    return mask


def dilate(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(mask, k)


def pack(results: list[FrameResult]) -> dict[str, np.ndarray]:
    """Pad per-frame detections to (F, Nmax, ...) arrays with a validity mask (contract people/<cam>.npz)."""
    f = len(results)
    nmax = max([len(r.boxes) for r in results] + [1])
    boxes = np.zeros((f, nmax, 4), np.float32)
    scores = np.zeros((f, nmax), np.float32)
    valid = np.zeros((f, nmax), bool)
    kpts = np.zeros((f, nmax, N_KPTS, 3), np.float32)
    for i, r in enumerate(results):
        n = len(r.boxes)
        order = np.argsort(-r.scores) if n else np.zeros(0, int)
        boxes[i, :n] = r.boxes[order]
        scores[i, :n] = r.scores[order]
        valid[i, :n] = True
        kpts[i, :n] = r.kpts[order]
    return {"boxes": boxes, "scores": scores, "valid": valid, "kpts": kpts}


def merge_scales(lo: FrameResult, hi: FrameResult, height: int, large_frac: float = 0.35,
                 min_iou: float = 0.5) -> FrameResult:
    """Combine a low-resolution pass (whole large, near-camera people) with a high-resolution pass (small,
    seated, distant people). Up-sampled 480p frames make YOLO truncate people that fill the frame (feet cut
    off), while 640 px misses the far crowd, so: boxes matched across scales keep the low-res box when the
    person is large (>= large_frac x image height) and the high-res box otherwise; unmatched boxes from either
    scale are kept; masks are OR-ed (undilated)."""
    pairs = greedy_match(iou_matrix(lo.boxes, hi.boxes), min_iou)
    keep_b, keep_s, keep_k = [], [], []
    for i, j in pairs:
        tall = max(lo.boxes[i, 3] - lo.boxes[i, 1], hi.boxes[j, 3] - hi.boxes[j, 1]) >= large_frac * height
        src, k = (lo, i) if tall else (hi, j)
        keep_b.append(src.boxes[k])
        keep_s.append(max(lo.scores[i], hi.scores[j]))
        keep_k.append(src.kpts[k])
    for src, used in ((lo, {i for i, _ in pairs}), (hi, {j for _, j in pairs})):
        for k in range(len(src.boxes)):
            if k not in used:
                keep_b.append(src.boxes[k])
                keep_s.append(src.scores[k])
                keep_k.append(src.kpts[k])
    mask = lo.mask if hi.mask.size == 0 else (hi.mask if lo.mask.size == 0 else np.maximum(lo.mask, hi.mask))
    return FrameResult(np.asarray(keep_b, np.float32).reshape(-1, 4), np.asarray(keep_s, np.float32),
                       np.asarray(keep_k, np.float32).reshape(-1, N_KPTS, 3), mask)


def dancer_pair_present(boxes: np.ndarray, scores: np.ndarray, valid: np.ndarray, width: int, height: int,
                        min_score: float = 0.2, rel_height: float = 0.5, min_height: float = 0.12,
                        bottom_margin: float = 0.02) -> bool:
    """QA heuristic for the gate "both dancers detected": >= 2 large, central, untruncated person boxes.

    central: box centre x within the middle 80% of the image; untruncated: the feet are not cut by the
    bottom border (near-camera seated spectators are); large: height >= rel_height x the tallest such box
    and >= min_height x image height (rejects the seated / distant crowd).
    """
    b = boxes[valid & (scores >= min_score)]
    if len(b) < 2:
        return False
    cx = (b[:, 0] + b[:, 2]) / 2
    h = b[:, 3] - b[:, 1]
    ok = (cx > 0.1 * width) & (cx < 0.9 * width) & (b[:, 3] < (1 - bottom_margin) * height)
    h = h[ok]
    if len(h) < 2:
        return False
    big = (h >= rel_height * h.max()) & (h >= min_height * height)
    if int(big.sum()) < 2:
        return False
    iou = iou_matrix(b[ok][big], b[ok][big])
    return bool((iou[np.triu_indices(len(iou), 1)] < 0.7).any())  # not two copies of one person


def dancer_region_present(boxes: np.ndarray, scores: np.ndarray, valid: np.ndarray, width: int, height: int,
                          min_score: float = 0.5, min_height: float = 0.12, bottom_margin: float = 0.02) -> bool:
    """Weaker QA check: >= 1 confident, central, untruncated, large box (dancers detected, possibly as one
    merged box while one dancer hides the other in an embrace)."""
    b = boxes[valid & (scores >= min_score)]
    cx = (b[:, 0] + b[:, 2]) / 2
    ok = (cx > 0.1 * width) & (cx < 0.9 * width) & (b[:, 3] < (1 - bottom_margin) * height) &         (b[:, 3] - b[:, 1] >= min_height * height)
    return bool(ok.any())


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------

class Detector:
    """The three YOLO models, run batch-wise on BGR uint8 frames."""

    def __init__(self, settings: Settings | None = None, pose: bool = True):
        from ultralytics import YOLO
        self.s = settings or Settings()
        WEIGHTS.mkdir(parents=True, exist_ok=True)
        self.det = YOLO(str(DET_WEIGHTS))
        self.seg = YOLO(str(SEG_WEIGHTS))  # auto-downloads into WEIGHTS on first use
        self.pose = YOLO(str(POSE_WEIGHTS)) if pose else None

    def _kw(self, imgsz: int, batch: int, conf: float, classes: list[int]) -> dict:
        return dict(imgsz=imgsz, conf=conf, iou=self.s.iou, classes=classes, quantize=16, verbose=False,
                    batch=batch, max_det=300)

    def __call__(self, images: list[np.ndarray], with_pose: bool = True) -> list[FrameResult]:
        """Fused multi-scale result per image: boxes/kpts from merge_scales, mask = dilated union of scales."""
        h, w = images[0].shape[:2]
        per_scale = [self._run(images, imgsz, with_pose) for imgsz in self.s.scales(w, h)]
        out = []
        for i in range(len(images)):
            r = per_scale[0][i]
            for other in per_scale[1:]:
                r = merge_scales(r, other[i], h, self.s.large_frac)
            out.append(FrameResult(r.boxes, r.scores, r.kpts, dilate(r.mask, self.s.dilate_px)))
        return out

    def _run(self, images: list[np.ndarray], imgsz: int, with_pose: bool) -> list[FrameResult]:
        bs = len(images)
        det = self.det.predict(images, **self._kw(imgsz, bs, min(self.s.conf, self.s.phone_conf),
                                                  [PERSON, CELL_PHONE]))
        seg = self.seg.predict(images, retina_masks=True, **self._kw(imgsz, bs, self.s.seg_conf, [PERSON]))
        pose = self.pose.predict(images, **self._kw(imgsz, bs, self.s.conf, [PERSON]))             if (with_pose and self.pose is not None) else [None] * len(images)
        return [self._fuse(img.shape[:2], d, sg, p) for img, d, sg, p in zip(images, det, seg, pose)]

    def _fuse(self, shape: tuple[int, int], d, sg, p) -> FrameResult:
        """One scale: person boxes + matched keypoints + undilated union mask (seg, missed boxes, phones)."""
        cls = d.boxes.cls.cpu().numpy().astype(int)
        conf = d.boxes.conf.cpu().numpy()
        xyxy = d.boxes.xyxy.cpu().numpy()
        person = (cls == PERSON) & (conf >= self.s.conf)
        phone = (cls == CELL_PHONE) & (conf >= self.s.phone_conf)
        boxes, scores = xyxy[person], conf[person]

        if p is not None and p.keypoints is not None and len(p.boxes):
            kpts = attach_keypoints(boxes, p.boxes.xyxy.cpu().numpy(), p.keypoints.data.cpu().numpy(),
                                    self.s.kpt_match_iou)
        else:
            kpts = np.zeros((len(boxes), N_KPTS, 3), np.float32)

        if sg.masks is not None and len(sg.masks):
            mask = (sg.masks.data.any(0).to("cpu").numpy().astype(np.uint8) * 255)
            if mask.shape != shape:  # retina masks are native size; guard anyway
                mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            mask = np.zeros(shape, np.uint8)
        # persons the segmenter missed (detector or pose model saw them): fill their box ellipse
        others = [boxes]
        if p is not None and len(p.boxes):
            others.append(p.boxes.xyxy.cpu().numpy())
        fill_boxes(mask, uncovered_boxes(np.concatenate(others), mask))
        fill_boxes(mask, xyxy[phone], ellipse=False)
        return FrameResult(boxes.astype(np.float32), scores.astype(np.float32), kpts, mask)


# ---------------------------------------------------------------------------
# take-level driver
# ---------------------------------------------------------------------------

def _read(paths: list[Path], pool: ThreadPoolExecutor) -> list[np.ndarray]:
    imgs = list(pool.map(lambda p: cv2.imread(str(p), cv2.IMREAD_COLOR), paths))
    for p, im in zip(paths, imgs):
        if im is None:
            raise FileNotFoundError(p)
    return imgs


def _write_masks(items: list[tuple[Path, np.ndarray]], pool: ThreadPoolExecutor) -> None:
    list(pool.map(lambda pm: cv2.imwrite(str(pm[0]), pm[1]), items))


def detect_camera(take: Take, cam: str, model: Detector, keyframes: bool = True) -> dict:
    """people/<cam>.npz + people_masks/<cam>/ (+ people_masks_kf/<cam>/) + people/qa/<cam>.jpg."""
    meta = json.loads((take.work / "frames" / cam / "frames.json").read_text())
    frame_dir = take.work / "frames" / cam
    files = [frame_dir / r["file"] for r in meta["frames"]]
    w, h = meta["width"], meta["height"]
    bs = model.s.batch(w, h)
    mask_dir = take.work / "people_masks" / cam
    mask_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    results: list[FrameResult] = []
    qa_idx = sorted({0, len(files) // 3, 2 * len(files) // 3, len(files) - 1})
    qa: dict[int, np.ndarray] = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for k in range(0, len(files), bs):
            chunk = files[k:k + bs]
            imgs = _read(chunk, pool)
            res = model(imgs)
            _write_masks([(mask_dir / (f.stem + ".png"), r.mask) for f, r in zip(chunk, res)], pool)
            for j, (img, r) in enumerate(zip(imgs, res)):
                if k + j in qa_idx:
                    qa[k + j] = overlay(img, r)
                results.append(FrameResult(r.boxes, r.scores, r.kpts, np.zeros((0, 0), np.uint8)))
        t_window = time.time() - t_start

        packed = pack(results)
        out = take.work / "people"
        (out / "qa").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / f"{cam}.npz", t=np.array([r["t"] for r in meta["frames"]]), **packed)
        cv2.imwrite(str(out / "qa" / f"{cam}.jpg"), contact_sheet([qa[i] for i in qa_idx],
                                                                 [f"{cam} #{i}" for i in qa_idx]),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])

        n_kf, t_kf = 0, 0.0
        if keyframes:
            t1 = time.time()
            kf_dir = take.work / "keyframes" / cam
            kf_files = [kf_dir / r["file"] for r in json.loads((kf_dir / "keyframes.json").read_text())]
            kf_out = take.work / "people_masks_kf" / cam
            kf_out.mkdir(parents=True, exist_ok=True)
            for k in range(0, len(kf_files), bs):
                chunk = kf_files[k:k + bs]
                res = model(_read(chunk, pool), with_pose=False)
                _write_masks([(kf_out / (f.stem + ".png"), r.mask) for f, r in zip(chunk, res)], pool)
            n_kf, t_kf = len(kf_files), time.time() - t1

    pairs = [dancer_pair_present(packed["boxes"][i], packed["scores"][i], packed["valid"][i], w, h)
             for i in range(len(results))]
    region = [dancer_region_present(packed["boxes"][i], packed["scores"][i], packed["valid"][i], w, h)
              for i in range(len(results))]
    n_people = packed["valid"].sum(1)
    return {
        "camera": cam, "frames": len(results), "width": w, "height": h, "imgsz": model.s.scales(w, h),
        "people_per_frame": {"mean": float(n_people.mean()), "min": int(n_people.min()), "max": int(n_people.max())},
        "kpts_attached": float((packed["kpts"][..., 2].max(-1) > 0)[packed["valid"]].mean()) if packed["valid"].any() else 0.0,
        "dancer_pair_rate": float(np.mean(pairs)),
        "frames_without_pair": [i for i, ok in enumerate(pairs) if not ok],
        "dancer_region_rate": float(np.mean(region)),
        "keyframes": n_kf,
        "runtime_s": {"window": round(t_window, 1), "keyframes": round(t_kf, 1),
                      "per_frame_ms": round(1000 * t_window / max(1, len(results)), 1)},
    }


def detect_take(take: Take, cams: list[str] | None = None, keyframes: bool = True,
                settings: Settings | None = None) -> dict:
    from . import gpu
    cams = cams or [s.id for s in take.sources]
    s = settings or Settings()
    report = {"settings": s.__dict__, "models": {"det": DET_WEIGHTS.name, "seg": SEG_WEIGHTS.name,
                                                  "pose": POSE_WEIGHTS.name, "licence": "AGPL-3.0 (Ultralytics)"},
              "cams": {}}
    t0 = time.time()
    with gpu.exclusive(f"detect {take.name}"):
        model = Detector(s)
        for cam in cams:
            r = detect_camera(take, cam, model, keyframes)
            report["cams"][cam] = r
            print(f"[detect] {cam}: {r['frames']} frames @{'+'.join(map(str, r['imgsz']))}, {r['people_per_frame']['mean']:.1f} people/frame, "
                  f"pair {100 * r['dancer_pair_rate']:.1f}%, {r['runtime_s']['per_frame_ms']:.0f} ms/frame, "
                  f"kf {r['keyframes']} in {r['runtime_s']['keyframes']:.0f}s", flush=True)
    report["runtime_s"] = round(time.time() - t0, 1)
    path = take.work / "people" / "report.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    if old.get("cams"):
        report["cams"] = {**old["cams"], **report["cams"]}
    path.write_text(json.dumps(report, indent=1))
    return report


# ---------------------------------------------------------------------------
# visual QA
# ---------------------------------------------------------------------------

def overlay(img: np.ndarray, r: FrameResult) -> np.ndarray:
    """Mask tinted red, boxes green (score), keypoints yellow."""
    vis = img.copy()
    m = r.mask > 0
    vis[m] = (0.45 * vis[m] + 0.55 * np.array([0, 0, 255])).astype(np.uint8)
    contours, _ = cv2.findContours(r.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (0, 0, 255), 1)
    th = max(1, img.shape[0] // 500)
    for b, s, k in zip(r.boxes, r.scores, r.kpts):
        x0, y0, x1, y1 = (int(v) for v in b)
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), th)
        cv2.putText(vis, f"{s:.2f}", (x0, max(10, y0 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35 * th, (0, 255, 0), th)
        for x, y, c in k:
            if c > 0.3:
                cv2.circle(vis, (int(x), int(y)), 2 * th, (0, 255, 255), -1)
    return vis


def contact_sheet(images: list[np.ndarray], labels: list[str], height: int = 960) -> np.ndarray:
    tiles = []
    for im, lab in zip(images, labels):
        s = height / im.shape[0]
        t = cv2.resize(im, (int(round(im.shape[1] * s)), height), interpolation=cv2.INTER_AREA)
        cv2.putText(t, lab, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        tiles.append(t)
    return np.hstack(tiles)
