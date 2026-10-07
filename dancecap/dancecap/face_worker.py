"""Face detector worker: runs in the separate venv HeadMovement/.venv-face (mediapipe, rtmlib, onnxruntime), never in
.venv-nopo4d. Only numpy / cv2 / the detector packages are imported, so it is safe to run as a subprocess:

    .venv-face\\Scripts\\python.exe -m dancecap.face_worker jobs  <take> [--mp-only] [--shard I/N]   face/jobs.json -> face/detections.npz
                                       (with --shard: face/detections_shard<I>of<N>.npz; then `merge <take> <N>`)
    .venv-face\\Scripts\\python.exe -m dancecap.face_worker image <img> <out.npz> [--upscale 2]   (portraits)

Detectors (open licences, weights in dancecap/weights/face/, git-ignored):
  * MediaPipe Face Landmarker (Apache-2.0; face_landmarker.task float16 v1, 3.6 MB, storage.googleapis.com/
    mediapipe-models): 478 landmarks (x, y in px, z relative depth), 52 ARKit-style blendshape scores and the facial
    transformation matrix. Run on the face crop (jobs box_face, ~2.2 x the landmark box) resized to MP_SIZE px;
    num_faces=2 and the face nearest the expected centre is kept (the partner is often in the crop).
  * RTMW-x 384x288 (Apache-2.0, rtmlib / openmmlab ONNX, 229 MB): COCO-WholeBody 133 keypoints; face = 23..90
    (iBUG-68 order). Top-down on an upper-body box (jobs box_body) - the face is ~25 % of the input width.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PKG = Path(__file__).resolve().parents[1]
WEIGHTS = PKG / "weights" / "face"
MP_TASK = WEIGHTS / "face_landmarker.task"
RTMW_ONNX = WEIGHTS / "rtmw_x384" / "end2end.onnx"
MP_SIZE = 256


def _mp_landmarker(num_faces: int = 2):
    import mediapipe as mp
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision
    opts = vision.FaceLandmarkerOptions(
        base_options=mpt.BaseOptions(model_asset_path=str(MP_TASK)),
        running_mode=vision.RunningMode.IMAGE, num_faces=num_faces,
        min_face_detection_confidence=0.3, min_face_presence_confidence=0.3, min_tracking_confidence=0.3,
        output_face_blendshapes=True, output_facial_transformation_matrixes=True)
    return mp, vision.FaceLandmarker.create_from_options(opts)


def _crop(img, box, size):
    x0, y0, x1, y1 = [float(v) for v in box]
    s = size / max(x1 - x0, 1e-6)
    A = np.array([[s, 0, -x0 * s], [0, s, -y0 * s]], np.float64)
    out = cv2.warpAffine(img, A, (size, size), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    return out, s, (x0, y0)


def run_mediapipe(mp, lm, img_bgr, box, expect_xy=None, size=MP_SIZE):
    """-> dict(ok, lmk (478,3) px in the full image (z scaled like x), blend (52,), names, M (4,4), n_faces, dist)."""
    crop, s, (x0, y0) = _crop(img_bgr, box, size)
    im = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(crop[:, :, ::-1]))
    res = lm.detect(im)
    if not res.face_landmarks:
        return {"ok": False, "n_faces": 0}
    cands = []
    for k, f in enumerate(res.face_landmarks):
        p = np.array([[q.x * size, q.y * size, q.z * size] for q in f], np.float64)
        p[:, 0] = p[:, 0] / s + x0
        p[:, 1] = p[:, 1] / s + y0
        p[:, 2] = p[:, 2] / s
        c = p[:, :2].mean(0)
        d = float(np.linalg.norm(c - np.asarray(expect_xy))) if expect_xy is not None else 0.0
        cands.append((d, k, p))
    cands.sort(key=lambda c: c[0])
    d, k, p = cands[0]
    bs = res.face_blendshapes[k] if res.face_blendshapes else []
    M = np.asarray(res.facial_transformation_matrixes[k], float) if res.facial_transformation_matrixes else np.eye(4)
    return {"ok": True, "lmk": p, "blend": np.array([b.score for b in bs], np.float32),
            "names": [b.category_name for b in bs], "M": M, "n_faces": len(res.face_landmarks), "dist": d}


def _rtmw():
    from rtmlib import RTMPose
    return RTMPose(onnx_model=str(RTMW_ONNX), model_input_size=(288, 384), backend="onnxruntime", device="cpu")


def run_jobs(take: str, rtmw: bool = True, shard: tuple[int, int] | None = None) -> Path:
    """rtmw=False (--mp-only): MediaPipe only, ~10x faster (RTMW-x on CPU dominates; its face points are not used
    downstream and it answers even when hair covers the face). shard (i, n): only jobs i, i+n, i+2n, ... ->
    face/detections_shard<i>of<n>.npz (run n processes on a many-CPU pod, then `merge <take> <n>`)."""
    work = PKG / "work" / take
    jobs_all = json.loads((work / "face" / "jobs.json").read_text())["jobs"]
    idx = list(range(len(jobs_all))) if shard is None else list(range(shard[0], len(jobs_all), shard[1]))
    jobs = [jobs_all[k] for k in idx]
    mp, lm = _mp_landmarker()
    pose = _rtmw() if rtmw else None
    n = len(jobs)
    out = {"mp_ok": np.zeros(n, bool), "mp_lmk": np.full((n, 478, 3), np.nan, np.float32),
           "mp_blend": np.full((n, 52), np.nan, np.float32), "mp_M": np.full((n, 4, 4), np.nan, np.float32),
           "mp_nfaces": np.zeros(n, np.int8), "mp_dist_px": np.full(n, np.nan, np.float32),
           "rtmw_face": np.full((n, 68, 2), np.nan, np.float32), "rtmw_score": np.full((n, 68), np.nan, np.float32),
           "rtmw_body": np.full((n, 23, 3), np.nan, np.float32)}
    names = None
    cache: dict = {}
    t0 = time.time()
    for k, j in enumerate(jobs):
        key = (j["cam"], j["i"])
        if key not in cache:
            cache.clear()
            cache[key] = cv2.imread(str(work / "frames" / j["cam"] / f"{j['i']:06d}.jpg"))
        img = cache[key]
        expect = np.asarray(j["lmk_model"], float).mean(0)
        r = run_mediapipe(mp, lm, img, j["box_face"], expect)
        out["mp_nfaces"][k] = r["n_faces"]
        if r["ok"]:
            out["mp_ok"][k] = True
            out["mp_lmk"][k] = r["lmk"]
            if len(r["blend"]) == 52:
                out["mp_blend"][k] = r["blend"]
                names = r["names"]
            out["mp_M"][k] = r["M"]
            out["mp_dist_px"][k] = r["dist"]
        if k % 100 == 0:
            print(f"[face_worker] {k}/{n} {time.time() - t0:.0f}s", flush=True)
        if pose is None:
            continue
        kp, sc = pose(img, bboxes=[j["box_body"]])
        out["rtmw_face"][k] = kp[0, 23:91]
        out["rtmw_score"][k] = sc[0, 23:91]
        out["rtmw_body"][k, :, :2] = kp[0, :23]
        out["rtmw_body"][k, :, 2] = sc[0, :23]
    out["mp_blend_names"] = np.array(names or [])
    out["job_index"] = np.array(idx, np.int64)
    p = work / "face" / ("detections.npz" if shard is None else f"detections_shard{shard[0]}of{shard[1]}.npz")
    np.savez_compressed(p, **out)
    print(f"[face_worker] done {n} jobs in {time.time() - t0:.0f}s -> {p}", flush=True)
    return p


def merge_shards(take: str, n: int) -> Path:
    """face/detections_shard<i>of<n>.npz (i < n) -> face/detections.npz in jobs.json order."""
    work = PKG / "work" / take
    n_jobs = len(json.loads((work / "face" / "jobs.json").read_text())["jobs"])
    parts = [np.load(work / "face" / f"detections_shard{i}of{n}.npz") for i in range(n)]
    out, names = {}, []
    for z in parts:
        names = names or [str(x) for x in z["mp_blend_names"]]
    for k in parts[0].files:
        if k in ("mp_blend_names", "job_index"):
            continue
        a0 = parts[0][k]
        arr = np.zeros((n_jobs,) + a0.shape[1:], a0.dtype)
        if np.issubdtype(a0.dtype, np.floating):
            arr[:] = np.nan
        for z in parts:
            arr[z["job_index"]] = z[k]
        out[k] = arr
    seen = np.concatenate([z["job_index"] for z in parts])
    if len(np.unique(seen)) != n_jobs:
        raise ValueError(f"shards cover {len(np.unique(seen))} of {n_jobs} jobs")
    out["mp_blend_names"] = np.array(names)
    p = work / "face" / "detections.npz"
    np.savez_compressed(p, **out)
    print(f"[face_worker] merged {n} shards ({n_jobs} jobs) -> {p}", flush=True)
    return p


def run_image(img_path: str, out_path: str, upscale: float = 2.0) -> Path:
    """Portrait: MediaPipe on the whole image (upscaled), RTMW on the whole image as one person box."""
    img = cv2.imread(img_path)
    H, W = img.shape[:2]
    mp, lm = _mp_landmarker(1)
    side = max(H, W)
    box = [(W - side) / 2, (H - side) / 2, (W + side) / 2, (H + side) / 2]
    r = run_mediapipe(mp, lm, img, box, None, size=int(side * upscale))
    pose = _rtmw()
    kp, sc = pose(img, bboxes=[[0, 0, W, H]])
    np.savez(out_path, mp_ok=r["ok"], mp_lmk=r.get("lmk", np.zeros((478, 3))), mp_blend=r.get("blend", np.zeros(52)),
             mp_blend_names=np.array(r.get("names", [])), mp_M=r.get("M", np.eye(4)), rtmw_face=kp[0, 23:91],
             rtmw_score=sc[0, 23:91], rtmw_all=kp[0], rtmw_all_score=sc[0], size=np.array([W, H]))
    print(f"[face_worker] {img_path}: mediapipe {'ok' if r['ok'] else 'NO FACE'}; rtmw face score "
          f"{float(np.mean(sc[0, 23:91])):.2f} -> {out_path}", flush=True)
    return Path(out_path)


def run_crops(in_npz: str, out_npz: str) -> Path:
    """Pre-cut square crops {name: (N,S,S,3) BGR} -> MediaPipe per crop (face nearest the centre):
    {name}_ok (N,), {name}_blend (N,52), {name}_lmk (N,478,2) in crop pixels, blend_names."""
    z = np.load(in_npz)
    mp, lm = _mp_landmarker()
    out, names = {}, None
    for name in z.files:
        crops = z[name]
        N, S = len(crops), crops.shape[1]
        ok = np.zeros(N, bool)
        bl = np.full((N, 52), np.nan, np.float32)
        lk = np.full((N, 478, 2), np.nan, np.float32)
        for k in range(N):
            r = run_mediapipe(mp, lm, crops[k], [0, 0, S, S], expect_xy=(S / 2, S / 2), size=MP_SIZE)
            if r["ok"] and r["dist"] < 0.25 * S and len(r["blend"]) == 52:
                ok[k], bl[k], lk[k] = True, r["blend"], r["lmk"][:, :2]
                names = r["names"]
        out[f"{name}_ok"], out[f"{name}_blend"], out[f"{name}_lmk"] = ok, bl, lk
        print(f"[face_worker] crops {name}: {ok.mean():.2f} detected", flush=True)
    out["blend_names"] = np.array(names or [])
    np.savez_compressed(out_npz, **out)
    return Path(out_npz)


def main(argv=None):
    a = list(sys.argv[1:] if argv is None else argv)
    if a and a[0] == "jobs":
        sh = None
        if "--shard" in a:
            i, n = a[a.index("--shard") + 1].split("/")
            sh = (int(i), int(n))
        run_jobs(a[1], rtmw="--mp-only" not in a, shard=sh)
    elif a and a[0] == "merge":
        merge_shards(a[1], int(a[2]))
    elif a and a[0] == "crops":
        run_crops(a[1], a[2])
    elif a and a[0] == "image":
        up = float(a[a.index("--upscale") + 1]) if "--upscale" in a else 2.0
        run_image(a[1], a[2], up)
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
