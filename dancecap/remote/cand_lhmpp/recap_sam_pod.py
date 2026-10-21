"""Pod side of the recap LHM++ crop masks (dancecap.mono4d.recap_splatav frames / select): SAM 2.1 image predictor
(facebookresearch/sam2 @ 2b90b9f from the bootstrap 'dancers' feature, sam2.1_hiera_large.pt downloaded here with its
sha256 checked) on the chosen phone frames, both dancers per frame.

    input/prompts.json   {"frames": [{"id", "file", "W", "H", "prompts": {role: {"box": [x0,y0,x1,y1],
                          "pos": [[u,v],..], "neg": [[u,v],..]}}}]}
    input/frames/<id>.jpg
Outputs: <out>/<id>_<role>_<k>.png (k = 0..2: SAM's three multimask candidates, 0/255, native size), <out>/sam.json
(per frame and role the three scores, timings, versions). The choice among the candidates is made locally against the
rasterised fitted body (select).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SAM2_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
SAM2_SHA256 = "2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318"
CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"


def log(*a):
    print("[recap-sam]", time.strftime("%H:%M:%S"), *a, flush=True)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default="/workspace/weights/sam2/sam2.1_hiera_large.pt")
    a = ap.parse_args()
    inp, out = Path(a.input), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rep = {"errors": [], "frames": {}, "timings": {}}
    t0 = time.time()
    ck = Path(a.ckpt)
    if not ck.exists() or sha256(ck) != SAM2_SHA256:
        ck.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["curl", "-sSL", "--retry", "6", "-o", str(ck), SAM2_URL], check=True)
    got = sha256(ck)
    if got != SAM2_SHA256:
        raise SystemExit(f"checkpoint sha256 {got} != {SAM2_SHA256}")
    rep["timings"]["ckpt_s"] = round(time.time() - t0, 1)
    import cv2
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    import sam2
    rep["versions"] = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "sam2": sam2.__file__,
                       "ckpt_sha256": got}
    t1 = time.time()
    model = build_sam2(CONFIG, str(ck), device="cuda")
    pred = SAM2ImagePredictor(model)
    rep["timings"]["load_s"] = round(time.time() - t1, 1)
    spec = json.loads((inp / "prompts.json").read_text())
    t2 = time.time()
    for fr in spec["frames"]:
        fid = fr["id"]
        try:
            img = cv2.imread(str(inp / "frames" / f"{fid}.jpg"))[:, :, ::-1].copy()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                pred.set_image(img)
                rr = {}
                for role, pr in fr["prompts"].items():
                    pts = [p for p in pr.get("pos", [])] + [p for p in pr.get("neg", [])]
                    lab = [1] * len(pr.get("pos", [])) + [0] * len(pr.get("neg", []))
                    masks, scores, _ = pred.predict(
                        point_coords=np.asarray(pts, np.float32) if pts else None,
                        point_labels=np.asarray(lab, np.int32) if pts else None,
                        box=np.asarray(pr["box"], np.float32), multimask_output=True)
                    order = np.argsort(-scores)
                    for k, j in enumerate(order):
                        cv2.imwrite(str(out / f"{fid}_{role}_{k}.png"), (masks[j] > 0).astype(np.uint8) * 255)
                    rr[role] = {"scores": [round(float(scores[j]), 4) for j in order],
                                "areas": [int((masks[j] > 0).sum()) for j in order]}
            rep["frames"][fid] = rr
        except Exception as ex:  # noqa: BLE001
            rep["errors"].append(f"{fid}: {type(ex).__name__}: {ex}")
    rep["timings"]["predict_s"] = round(time.time() - t2, 1)
    rep["timings"]["total_s"] = round(time.time() - t0, 1)
    (out / "sam.json").write_text(json.dumps(rep, indent=1))
    log(f"{len(rep['frames'])} frames, {len(rep['errors'])} errors, {rep['timings']}")
    return 0 if not rep["errors"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
