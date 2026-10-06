"""SAM 2.1 video propagation of the two dancers on the 11 GB Turing card.

Deviations from the stock SAM2VideoPredictor, all to fit the GPU policy (CONTRACTS.md):
  * fp16 autocast via gpu.amp_dtype() (stock demos use bf16, emulated ~18x slower on sm_75); the
    predictor's hard-coded bf16 *storage* cast of memory features is redirected to fp16 as well.
  * ++model.max_cond_frames_in_attn=4: re-anchored prompts add conditioning frames; only the 4 temporally
    closest are attended, so cost stays flat.
  * ++model.clear_non_cond_mem_around_input=true: when propagation reaches a re-anchored frame, the drifted
    non-conditioning memories around it are dropped so the correction is not out-voted by them. REQUIRED:
    build_with_overrides retries the overrides one by one and raises if this one is rejected.
  * frames are read lazily from the canonical JPEGs (no 1024^2 x F tensor in RAM), state offloaded to CPU.
  * the predictor's output ("video") resolution is the working resolution (long edge <= 1280); callers
    upsample probabilities to native resolution.

Prompts are boxes (xyxy, native px; scaled here). Propagation is forward then backward from the first
anchor. A `check(frame, probs)` callback may return new box prompts for that frame (sparse re-anchoring);
the generator is then restarted from that frame so the correction is used immediately.
"""
from __future__ import annotations

import time
import types
from collections import OrderedDict
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .take import WORKSPACE

CHECKPOINT = WORKSPACE / "mamma" / "data" / "weights" / "sam2" / "sam2.1_hiera_large.pt"
CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
MAX_WORK_EDGE = 1280
OBJ_IDS = (1, 2)  # 1 = lead, 2 = follow

Boxes = dict[int, np.ndarray]  # obj id -> xyxy (native px)


def work_size(width: int, height: int, max_edge: int = MAX_WORK_EDGE) -> tuple[int, int]:
    """(w, h) of the working resolution: native when the long edge fits, else scaled down."""
    s = min(1.0, max_edge / max(width, height))
    return int(round(width * s)), int(round(height * s))


class LazyFrames:
    """Sequence of SAM-normalised (3, S, S) float tensors read from JPEGs on demand (small LRU cache)."""

    def __init__(self, files: list[Path], image_size: int, cache: int = 4):
        import torch
        self.files, self.size, self.cache = files, image_size, OrderedDict()
        self.cap = cache
        self.mean = torch.tensor((0.485, 0.456, 0.406))[:, None, None]
        self.std = torch.tensor((0.229, 0.224, 0.225))[:, None, None]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int):
        import torch
        i = int(i)
        if i in self.cache:
            self.cache.move_to_end(i)
            return self.cache[i]
        img = cv2.imread(str(self.files[i]), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(self.files[i])
        img = cv2.resize(img[..., ::-1], (self.size, self.size), interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
        t = (t - self.mean) / self.std
        self.cache[i] = t
        if len(self.cache) > self.cap:
            self.cache.popitem(last=False)
        return t


def _fp16_storage(module) -> None:
    """The predictor casts stored memory features to torch.bfloat16; point that name at float16."""
    import torch
    if getattr(module, "_dancecap_fp16", False):
        return
    proxy = types.SimpleNamespace(**{k: getattr(torch, k) for k in dir(torch) if not k.startswith("__")})
    proxy.bfloat16 = torch.float16
    module.torch = proxy
    module._dancecap_fp16 = True


def _clear_obj_non_cond_mem_around_input(self, inference_state, frame_idx, obj_idx):
    """Per-object version of SAM2VideoPredictor._clear_non_cond_mem_around_input; the installed sam2
    calls this method when clear_non_cond_mem_around_input is on but does not define it."""
    r = self.memory_temporal_stride_for_eval
    outs = inference_state["output_dict_per_obj"][obj_idx]["non_cond_frame_outputs"]
    for t in range(frame_idx - r * self.num_maskmem, frame_idx + r * self.num_maskmem + 1):
        outs.pop(t, None)


OVERRIDES = {  # name -> (hydra override, required)
    "clear_non_cond_mem_around_input": ("++model.clear_non_cond_mem_around_input=true", True),
    "max_cond_frames_in_attn": ("++model.max_cond_frames_in_attn=4", False),
}


def build_with_overrides(build: Callable[..., object], log: Callable[[str], None] = print):
    """build(hydra_overrides_extra=[...]) with all OVERRIDES; when that is rejected, retry with each subset
    (largest first) so one rejected option never silently drops the other. Re-anchoring correctness needs
    clear_non_cond_mem_around_input, so a build without it raises; max_cond_frames_in_attn only bounds cost.
    The active overrides are logged and stored on the predictor as `dancecap_overrides`."""
    from itertools import combinations
    names = list(OVERRIDES)
    errors = {}
    for n in range(len(names), -1, -1):
        for subset in combinations(names, n):
            missing = [k for k in names if k not in subset]
            if any(OVERRIDES[k][1] for k in missing):
                continue
            try:
                pred = build(hydra_overrides_extra=[OVERRIDES[k][0] for k in subset])
            except Exception as e:  # noqa: BLE001 - hydra raises various config errors
                errors[subset] = repr(e)
                continue
            if missing:
                log(f"[dancers] SAM override(s) {missing} rejected ({errors}); active: {list(subset)}")
            else:
                log(f"[dancers] SAM overrides active: {list(subset)}")
            try:
                pred.dancecap_overrides = list(subset)
            except AttributeError:
                pass
            return pred
    req = [k for k in names if OVERRIDES[k][1]]
    raise RuntimeError(f"SAM 2 predictor could not be built with the required override(s) {req}: {errors}")


def build_predictor(device: str = "cuda"):
    import sam2.sam2_video_predictor as svp
    from sam2.build_sam import build_sam2_video_predictor
    _fp16_storage(svp)
    if not hasattr(svp.SAM2VideoPredictor, "_clear_obj_non_cond_mem_around_input"):
        svp.SAM2VideoPredictor._clear_obj_non_cond_mem_around_input = _clear_obj_non_cond_mem_around_input
    return build_with_overrides(lambda **kw: build_sam2_video_predictor(CONFIG, str(CHECKPOINT), device=device, **kw))


def init_state(predictor, files: list[Path], work_wh: tuple[int, int]) -> dict:
    """SAM2VideoPredictor.init_state with lazy frames, offloaded video + state, working output size."""
    import torch
    st: dict = {
        "images": LazyFrames(files, predictor.image_size), "num_frames": len(files),
        "offload_video_to_cpu": True, "offload_state_to_cpu": True,
        "video_height": work_wh[1], "video_width": work_wh[0], "device": predictor.device,
        "storage_device": torch.device("cpu"), "point_inputs_per_obj": {}, "mask_inputs_per_obj": {},
        "cached_features": {}, "constants": {}, "obj_id_to_idx": OrderedDict(), "obj_idx_to_id": OrderedDict(),
        "obj_ids": [], "output_dict_per_obj": {}, "temp_output_dict_per_obj": {}, "frames_tracked_per_obj": {},
    }
    return st


def forget_frame(state: dict, frame: int, obj_id: int) -> None:
    """Drop everything SAM holds for one object at one frame, so a re-anchor box acts like a fresh initial
    prompt. (add_new_points_or_box on an already tracked frame is a *refinement*: it feeds the old mask
    logits back in as a dense prompt and the old - drifted or swapped - mask wins over the new box.)"""
    idx = state["obj_id_to_idx"].get(obj_id)
    if idx is None:
        return
    state["point_inputs_per_obj"][idx].pop(frame, None)
    state["mask_inputs_per_obj"][idx].pop(frame, None)
    for d in (state["output_dict_per_obj"][idx], state["temp_output_dict_per_obj"][idx]):
        d["cond_frame_outputs"].pop(frame, None)
        d["non_cond_frame_outputs"].pop(frame, None)
    state["frames_tracked_per_obj"][idx].pop(frame, None)


def _scale_box(box: np.ndarray, s: float) -> np.ndarray:
    return np.asarray(box, np.float32) * s


def propagate(predictor, files: list[Path], native_wh: tuple[int, int], anchor: int, anchor_boxes: Boxes,
              check: Callable[[int, np.ndarray], Boxes | None] | None = None, min_gap: int = 8,
              log: Callable[[str], None] = print) -> dict:
    """Track OBJ_IDS through all frames. Returns probs (F, 2, h, w) uint8 (sigmoid*255 at working res),
    score (F, 2) object-presence logits, anchors {frame: boxes} actually used, timing."""
    import torch

    from . import gpu
    work_wh = work_size(*native_wh)
    s = work_wh[0] / native_wh[0]
    f = len(files)
    probs = np.zeros((f, 2, work_wh[1], work_wh[0]), np.uint8)
    done = np.zeros(f, bool)
    anchors: dict[int, Boxes] = {}
    t0 = time.time()
    n_inf = 0
    with torch.inference_mode(), torch.autocast("cuda", dtype=gpu.amp_dtype()):
        state = init_state(predictor, files, work_wh)

        def prompt(frame: int, boxes: Boxes):
            anchors[frame] = {k: np.asarray(v, float) for k, v in boxes.items()}
            for oid in boxes:
                forget_frame(state, frame, oid)
            for oid, b in boxes.items():
                predictor.add_new_points_or_box(state, frame_idx=frame, obj_id=oid, box=_scale_box(b, s))

        prompt(anchor, anchor_boxes)
        for reverse in (False, True):
            start = anchor
            while True:
                restarted = False
                for fi, obj_ids, logits in predictor.propagate_in_video(state, start_frame_idx=start, reverse=reverse):
                    n_inf += 1
                    order = [list(obj_ids).index(o) for o in OBJ_IDS]
                    p = torch.sigmoid(logits[order, 0].float()).mul_(255).to(torch.uint8).cpu().numpy()
                    probs[fi] = p
                    done[fi] = True
                    if check is None or fi in anchors or min(abs(fi - a) for a in anchors) < min_gap:
                        continue
                    new = check(fi, p)
                    if new:
                        log(f"[dancers]   re-anchor frame {fi} ({'bwd' if reverse else 'fwd'}) objs {sorted(new)}")
                        prompt(fi, new)
                        start, restarted = fi, True
                        break
                if not restarted:
                    break
    dt = time.time() - t0
    return {"probs": probs, "done": done, "anchors": anchors, "seconds": dt, "frame_evals": n_inf,
            "work_wh": work_wh}
