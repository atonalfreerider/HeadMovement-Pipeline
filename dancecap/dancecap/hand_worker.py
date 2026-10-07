"""Hand keypoint worker: runs in the separate venv HeadMovement/.venv-face (mediapipe, rtmlib, onnxruntime), never in
.venv-nopo4d. Only numpy / cv2 / the detector packages are imported, so it is safe to run as a subprocess:

    .venv-face\\Scripts\\python.exe -m dancecap.hand_worker <take> [--no-rtmw] [--stride N]
        hands/jobs.json (refine_hands survey) -> hands/detections.npz

Detectors (open licences; weights in dancecap/weights/, git-ignored):
  * RTMW-x 384x288 whole-body (Apache-2.0, openmmlab ONNX via rtmlib, 229 MB; weights/face/rtmw_x384): COCO-WholeBody
    133 keypoints top-down on the dancer's body box; hands = 91..111 (left) and 112..132 (right), with body context
    (it knows which hand is which). The hand is only ~5-15 % of the 288 px input width.
  * RTMPose-m hand5 256x256 (Apache-2.0, openmmlab ONNX, 51 MB zip; weights/hands/rtmpose_m_hand5): 21 keypoints
    top-down on the square hand crop (jobs box, ~2.2 x the fitted hand's projected extent), upscaled to 256.
  * MediaPipe Hand Landmarker (Apache-2.0, hand_landmarker.task float16, 7.8 MB, storage.googleapis.com/
    mediapipe-models; weights/hands): palm detector + 21 landmarks on the hand crop upscaled to MP_SIZE px; the hand
    whose landmarks are nearest the fitted hand's projection is kept (the partner's hand is often in the crop).
All outputs are in full-image pixels of the canonical frame (distorted; refine_hands undistorts).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PKG = Path(__file__).resolve().parents[1]
W_FACE = PKG / "weights" / "face"
W_HANDS = PKG / "weights" / "hands"
RTMW_ONNX = W_FACE / "rtmw_x384" / "end2end.onnx"
MP_TASK = W_HANDS / "hand_landmarker.task"
MP_SIZE = 256
RTM_HAND_DIR = W_HANDS / "rtmpose_m_hand5"


def _rtm_hand_onnx() -> Path:
    hits = sorted(RTM_HAND_DIR.rglob("end2end.onnx"))
    if not hits:
        raise FileNotFoundError(f"RTMPose hand ONNX not found under {RTM_HAND_DIR}")
    return hits[0]


def _crop(img, box, size):
    x0, y0, x1, y1 = [float(v) for v in box]
    s = size / max(x1 - x0, 1e-6)
    A = np.array([[s, 0, -x0 * s], [0, s, -y0 * s]], np.float64)
    out = cv2.warpAffine(img, A, (size, size), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    return out, s, (x0, y0)


def _mp_hands():
    import mediapipe as mp
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision
    opts = vision.HandLandmarkerOptions(
        base_options=mpt.BaseOptions(model_asset_path=str(MP_TASK)), running_mode=vision.RunningMode.IMAGE,
        num_hands=2, min_hand_detection_confidence=0.2, min_hand_presence_confidence=0.2, min_tracking_confidence=0.2)
    return mp, vision.HandLandmarker.create_from_options(opts)


def run_mp(mp, lm, img, box, model_px, size=MP_SIZE):
    """-> (ok, (21,3) px [x, y, z rel], handedness label, score, n_hands, mean px distance to the model hand)."""
    crop, s, (x0, y0) = _crop(img, box, size)
    res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(crop[:, :, ::-1])))
    if not res.hand_landmarks:
        return False, None, "", 0.0, 0, np.inf
    best = None
    for k, h in enumerate(res.hand_landmarks):
        p = np.array([[q.x * size, q.y * size, q.z * size] for q in h], np.float64)
        p[:, 0] = p[:, 0] / s + x0
        p[:, 1] = p[:, 1] / s + y0
        p[:, 2] = p[:, 2] / s
        d = float(np.linalg.norm(p[:, :2] - model_px, axis=1).mean())
        hd = res.handedness[k][0] if res.handedness and res.handedness[k] else None
        cand = (d, p, hd.category_name if hd else "", float(hd.score) if hd else 0.0)
        if best is None or d < best[0]:
            best = cand
    return True, best[1], best[2], best[3], len(res.hand_landmarks), best[0]


def _ort_cuda() -> bool:
    """Make onnxruntime-gpu find CUDA/cuDNN: torch's pip wheels ship them (import torch loads them into the process);
    onnxruntime >= 1.21 can also preload them itself. True when the CUDA execution provider is available."""
    try:
        import torch  # noqa: F401
    except Exception:
        pass
    try:
        import onnxruntime as ort
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
        return "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:
        return False


def run(take: str, rtmw: bool = True, stride: int = 1, device: str = "cpu") -> Path:
    from rtmlib import RTMPose
    if device == "cuda" and not _ort_cuda():
        print("[hand_worker] WARNING: onnxruntime CUDA provider unavailable - running rtmlib on the CPU", flush=True)
        device = "cpu"
    work = PKG / "work" / take
    J = json.loads((work / "hands" / "jobs.json").read_text())
    jobs = [j for j in J["jobs"] if j["i"] % stride == 0]
    bjobs = [j for j in J["body_jobs"] if j["i"] % stride == 0] if rtmw else []
    try:
        mp, mpl = _mp_hands()
    except Exception as e:  # mediapipe missing on a pod: the other two detectors still run
        print(f"[hand_worker] WARNING: MediaPipe unavailable ({type(e).__name__}: {e}); skipping it", flush=True)
        mp = mpl = None
    hand_pose = RTMPose(onnx_model=str(_rtm_hand_onnx()), model_input_size=(256, 256), backend="onnxruntime",
                        device=device)
    n, nb = len(jobs), len(bjobs)
    out = {"job_cam": np.array([j["cam"] for j in jobs]), "job_i": np.array([j["i"] for j in jobs], np.int32),
           "job_role": np.array([j["role"] for j in jobs]), "job_side": np.array([j["side"] for j in jobs]),
           "job_box": np.array([j["box"] for j in jobs], np.float32),
           "rtmh_kp": np.full((n, 21, 2), np.nan, np.float32), "rtmh_score": np.full((n, 21), np.nan, np.float32),
           "mp_ok": np.zeros(n, bool), "mp_kp": np.full((n, 21, 3), np.nan, np.float32),
           "mp_label": np.array([""] * n, dtype="<U8"), "mp_score": np.zeros(n, np.float32),
           "mp_nhands": np.zeros(n, np.int8), "mp_dist_px": np.full(n, np.nan, np.float32),
           "body_cam": np.array([j["cam"] for j in bjobs]), "body_i": np.array([j["i"] for j in bjobs], np.int32),
           "body_role": np.array([j["role"] for j in bjobs]),
           "rtmw_kp": np.full((nb, 133, 2), np.nan, np.float32), "rtmw_score": np.full((nb, 133), np.nan, np.float32)}
    cache: dict = {}

    def image(cam, i):
        key = (cam, i)
        if key not in cache:
            cache.clear()
            cache[key] = cv2.imread(str(work / "frames" / cam / f"{i:06d}.jpg"))
        return cache[key]

    t0 = time.time()
    order = sorted(range(n), key=lambda k: (jobs[k]["cam"], jobs[k]["i"]))
    for c, k in enumerate(order):
        j = jobs[k]
        img = image(j["cam"], j["i"])
        model = np.asarray(j["model_px"], float)
        kp, sc = hand_pose(img, bboxes=[j["box"]])
        out["rtmh_kp"][k], out["rtmh_score"][k] = kp[0], sc[0]
        if mpl is None:
            continue
        ok, p, lab, s, nh, d = run_mp(mp, mpl, img, j["box"], model)
        out["mp_nhands"][k] = nh
        if ok:
            out["mp_ok"][k], out["mp_kp"][k], out["mp_label"][k], out["mp_score"][k], out["mp_dist_px"][k] = \
                True, p, lab, s, d
        if c % 500 == 0:
            print(f"[hand_worker] hands {c}/{n} {time.time() - t0:.0f}s", flush=True)
    if bjobs:
        body = RTMPose(onnx_model=str(RTMW_ONNX), model_input_size=(288, 384), backend="onnxruntime", device=device)
        order = sorted(range(nb), key=lambda k: (bjobs[k]["cam"], bjobs[k]["i"]))
        for c, k in enumerate(order):
            j = bjobs[k]
            kp, sc = body(image(j["cam"], j["i"]), bboxes=[j["box"]])
            out["rtmw_kp"][k], out["rtmw_score"][k] = kp[0], sc[0]
            if c % 200 == 0:
                print(f"[hand_worker] rtmw {c}/{nb} {time.time() - t0:.0f}s", flush=True)
    p = work / "hands" / "detections.npz"
    np.savez_compressed(p, **out)
    print(f"[hand_worker] done {n} hand crops + {nb} body boxes in {time.time() - t0:.0f}s on {device} -> {p}", flush=True)
    return p


def main(argv=None):
    a = list(sys.argv[1:] if argv is None else argv)
    if not a or a[0].startswith("-"):
        print(__doc__)
        return 2
    stride = int(a[a.index("--stride") + 1]) if "--stride" in a else 1
    device = a[a.index("--device") + 1] if "--device" in a else "cpu"
    run(a[0], rtmw="--no-rtmw" not in a, stride=stride, device=device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
