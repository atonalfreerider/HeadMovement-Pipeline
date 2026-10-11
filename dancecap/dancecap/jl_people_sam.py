"""john_lula: pod-side SAM 2.1 masks of the two dancers from the identity anchors (agent J3a, 2026-10-11).

    python jl_people_sam.py VIDEO ANCHORS.json OUT.npz [--ckpt sam2.1_hiera_large.pt] [--frames DIR]

ANCHORS.json = [{"frame": f, "john": {"box": [x0,y0,x1,y1], "pos": [[x,y],..], "neg": [[x,y],..]}, "lula": {...}}, ...] sorted by frame: frames
where the 2D identity stage (dancecap.jl_people) is sure who is who and the two are on separate detections. Per anchor TWO short SAM runs are made
(forward to the next anchor, backward to the previous one; the ends run to the first / last frame); every frame between two anchors therefore has two
independent masks per dancer (one carried from each side) and the local stage uses their agreement as the confidence. A prompt = the person's box +
positive points on its own torso / hips + NEGATIVE points on the partner's torso (two bodies in an embrace, one box). Masks are stored as PNG bytes.

Output npz: runs (json: [{"anchor": f, "dir": "f"|"b", "lo": a, "hi": b}]), and for every mask a uint8 array "m_<anchor>_<dir>_<frame>_<obj>" (PNG bytes,
obj 0 = John, 1 = Lula), "s_<anchor>_<dir>" (n_frames_in_run, 2) float32 = max mask logit per frame and object.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"


def log(msg: str) -> None:
    print(f"[jl_sam {time.strftime('%H:%M:%S')}] {msg}", flush=True)


class LazyFrames:
    """SAM-normalised (3, S, S) tensors read from JPEGs on demand (no 1024^2 x F tensor in RAM)."""

    def __init__(self, files, image_size, cache=4):
        import torch
        self.files, self.size, self.cache, self.cap = files, image_size, OrderedDict(), cache
        self.mean = torch.tensor((0.485, 0.456, 0.406))[:, None, None]
        self.std = torch.tensor((0.229, 0.224, 0.225))[:, None, None]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        import cv2
        import torch
        i = int(i)
        if i in self.cache:
            self.cache.move_to_end(i)
            return self.cache[i]
        img = cv2.imread(str(self.files[i]), cv2.IMREAD_COLOR)
        img = cv2.resize(img[..., ::-1], (self.size, self.size), interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
        t = (t - self.mean) / self.std
        self.cache[i] = t
        if len(self.cache) > self.cap:
            self.cache.popitem(last=False)
        return t


def _clear_obj_non_cond_mem_around_input(self, inference_state, frame_idx, obj_idx):
    r = self.memory_temporal_stride_for_eval
    outs = inference_state["output_dict_per_obj"][obj_idx]["non_cond_frame_outputs"]
    for t in range(frame_idx - r * self.num_maskmem, frame_idx + r * self.num_maskmem + 1):
        outs.pop(t, None)


def build_predictor(ckpt: str):
    import sam2.sam2_video_predictor as svp
    from sam2.build_sam import build_sam2_video_predictor
    if not hasattr(svp.SAM2VideoPredictor, "_clear_obj_non_cond_mem_around_input"):
        svp.SAM2VideoPredictor._clear_obj_non_cond_mem_around_input = _clear_obj_non_cond_mem_around_input
    last = None
    for ov in (["++model.clear_non_cond_mem_around_input=true", "++model.max_cond_frames_in_attn=4"], ["++model.clear_non_cond_mem_around_input=true"], []):
        try:
            pred = build_sam2_video_predictor(CONFIG, ckpt, device="cuda", hydra_overrides_extra=ov)
            log(f"SAM 2 predictor built, overrides {ov}")
            return pred
        except Exception as e:  # noqa: BLE001
            last = e
    raise RuntimeError(f"cannot build the SAM 2 predictor: {last!r}")


def init_state(pred, files, wh):
    import torch
    return {"images": LazyFrames(files, pred.image_size), "num_frames": len(files), "offload_video_to_cpu": True, "offload_state_to_cpu": True,
            "video_height": wh[1], "video_width": wh[0], "device": pred.device, "storage_device": torch.device("cpu"), "point_inputs_per_obj": {},
            "mask_inputs_per_obj": {}, "cached_features": {}, "constants": {}, "obj_id_to_idx": OrderedDict(), "obj_idx_to_id": OrderedDict(),
            "obj_ids": [], "output_dict_per_obj": {}, "temp_output_dict_per_obj": {}, "frames_tracked_per_obj": {}}


def extract_frames(video: str, out: Path) -> list[Path]:
    import cv2
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video)
    files = []
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        p = out / f"{i:05d}.jpg"
        if not p.exists():
            cv2.imwrite(str(p), f, [cv2.IMWRITE_JPEG_QUALITY, 96])
        files.append(p)
        i += 1
    cap.release()
    return files


def run(video: str, anchors_json: str, out: str, ckpt: str, frames_dir: str) -> None:
    import cv2
    import torch
    files = extract_frames(video, Path(frames_dir))
    F = len(files)
    h, w = cv2.imread(str(files[0])).shape[:2]
    anchors = sorted(json.loads(Path(anchors_json).read_text()), key=lambda a: a["frame"])
    fr = [a["frame"] for a in anchors]
    pred = build_predictor(ckpt)
    runs, store = [], {}
    t0 = time.time()
    n_done = 0
    for k, a in enumerate(anchors):
        jobs = []
        hi_f = fr[k + 1] if k + 1 < len(fr) else F - 1
        lo_b = fr[k - 1] if k > 0 else 0
        if hi_f > a["frame"]:
            jobs.append(("f", a["frame"], hi_f))
        if lo_b < a["frame"]:
            jobs.append(("b", lo_b, a["frame"]))
        for d, lo, hi in jobs:
            sub = files[lo:hi + 1]
            loc = a["frame"] - lo
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                st = init_state(pred, sub, (w, h))
                for obj, role in ((1, "john"), (2, "lula")):
                    pr = a[role]
                    pts = [[x, y] for x, y in pr["pos"]] + [[x, y] for x, y in pr["neg"]]
                    lab = [1] * len(pr["pos"]) + [0] * len(pr["neg"])
                    kw = {}
                    if pts:
                        kw = {"points": np.array(pts, np.float32), "labels": np.array(lab, np.int32)}
                    pred.add_new_points_or_box(st, frame_idx=loc, obj_id=obj, box=np.array(pr["box"], np.float32), **kw)
                sc = np.full((len(sub), 2), -99.0, np.float32)
                for fi, obj_ids, logits in pred.propagate_in_video(st, start_frame_idx=loc, reverse=(d == "b")):
                    order = [list(obj_ids).index(o) for o in (1, 2)]
                    lg = logits[order, 0].float().cpu().numpy()
                    for o in range(2):
                        m = lg[o] > 0
                        sc[fi, o] = float(lg[o].max())
                        ok, png = cv2.imencode(".png", (m * 255).astype(np.uint8))
                        store[f"m_{a['frame']}_{d}_{lo + fi}_{o}"] = np.frombuffer(png.tobytes(), np.uint8)
                    n_done += 1
            store[f"s_{a['frame']}_{d}"] = sc
            runs.append({"anchor": a["frame"], "dir": d, "lo": lo, "hi": hi})
        if k % 5 == 0:
            log(f"anchor {k + 1}/{len(anchors)} frame {a['frame']}  {n_done} frame-evals  {time.time() - t0:.0f} s")
    np.savez(out, runs=np.array(json.dumps(runs)), size=np.array([w, h]), **store)
    log(f"done: {len(runs)} runs, {n_done} frame evaluations in {time.time() - t0:.0f} s")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video"); ap.add_argument("anchors"); ap.add_argument("out")
    ap.add_argument("--ckpt", default="/workspace/hm/weights/mamma/sam2/sam2.1_hiera_large.pt")
    ap.add_argument("--frames", default="/opt/hm/j3a/frames")
    a = ap.parse_args(argv)
    run(a.video, a.anchors, a.out, a.ckpt, a.frames)
    return 0


if __name__ == "__main__":
    sys.exit(main())
