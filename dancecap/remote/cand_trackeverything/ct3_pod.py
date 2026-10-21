"""Pod side of the TrackEverything / CoTracker3 foot + torso tracking test on john_lula (dancecap.mono4d.jl_trackeverything).

STAGE 0 (the TrackEverything weights katefgroup/TrackEverything are behind a Hugging Face click-through gate the owner has
not accepted): CoTracker3 offline (facebookresearch/co-tracker @ 82e02e8, scaled_offline.pth from the public
facebook/cotracker3 repo) on OUR native frames, with OUR queries (RTMW foot points re-seeded every 4 frames + torso points
inside the SAM masks). Arms (all on one pod):

    ct3_512      the official predictor's resize to the model resolution 384 x 512 (H x W): our 538 x 960 portrait frames
                 are squashed 2.5x vertically (TrackEverything's default 384 x 512 does the same)
    ct3_native   536 x 960 (W x H; 1 px trimmed off each side by the resize), aspect kept, no down-scaling
    ct3_full     ct3_native without the cut at the zoom switch (one segment over the whole window)

Each arm runs per segment (the 'cut' arms: frames before / from the zoom switch), backward tracking on (each query is
tracked over its whole segment), the predictor's 6 x 6 support grid, iters 6, queries chunked on CUDA OOM.

    input/meta.json       {"frames": [450..674], "segments": {"cut": [[a, b], ...], "full": [[a, b]]}, ...}
    input/frames/NNNNNN.jpg   native frames (global frame numbers)
    input/queries.npz     q_frame (N,) global frame, q_xy (N, 2) native px
Outputs: <out>/ct3_<arm>.npz  tracks (N, F, 2) native px (NaN outside the query's segment), vis (N, F), conf (N, F)
float16; <out>/ct3_pod.json (versions, checkpoint sha256, GPU, per arm / segment seconds, peak VRAM, chunking).
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

CT3_REPO = "https://github.com/facebookresearch/co-tracker.git"
CT3_COMMIT = "82e02e8029753ad4ef13cf06be7f4fc5facdda4d"
CKPT_URL = "https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth"
ARMS = {"ct3_512": {"size": None, "segments": "cut"},
        "ct3_native": {"size": (960, 536), "segments": "cut"},
        "ct3_full": {"size": (960, 536), "segments": "full"}}
SUPPORT = 6


def log(*a):
    print("[ct3]", time.strftime("%H:%M:%S"), *a, flush=True)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def setup(root: Path) -> tuple[Path, Path]:
    code = root / "co-tracker"
    if not (code / "cotracker" / "predictor.py").exists():
        subprocess.run(["git", "clone", "-q", CT3_REPO, str(code)], check=True)
    subprocess.run(["git", "-C", str(code), "checkout", "-q", CT3_COMMIT], check=True)
    ck = root / "ckpt" / "scaled_offline.pth"
    if not ck.exists() or ck.stat().st_size < 50_000_000:
        ck.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["curl", "-sSL", "--retry", "6", "-o", str(ck), CKPT_URL], check=True)
    return code, ck


def grid_points(n: int, H: int, W: int, device):
    """The predictor's support grid (cotracker.models.core.model_utils.get_points_on_a_grid, center = image centre)."""
    import torch
    margin = W / 64
    ys = torch.linspace(margin, H - margin, n, device=device)
    xs = torch.linspace(margin, W - margin, n, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx, gy], -1).reshape(1, -1, 2)


def run_sparse(model, video, q, size, backward=True, chunk=None):
    """video (1, T, 3, H, W) float 0..255 on cuda, q (N, 3) [t, x, y] native px -> tracks (T, N, 2) native px, vis, conf.
    Mirrors CoTrackerPredictor._compute_sparse_tracks (resize with align_corners, support grid, backward pass, query
    frame overwrite) but keeps the raw visibility and confidence."""
    import torch
    import torch.nn.functional as F
    B, T, C, H, W = video.shape
    h, w = size
    v = F.interpolate(video.reshape(T, C, H, W), (h, w), mode="bilinear", align_corners=True).reshape(1, T, C, h, w)
    sx, sy = (w - 1) / (W - 1), (h - 1) / (H - 1)
    qq = torch.as_tensor(q, dtype=torch.float32, device=video.device).clone()
    qq[:, 1] *= sx
    qq[:, 2] *= sy
    N = len(qq)
    chunk = chunk or N
    out_t, out_v, out_c = [], [], []
    g = grid_points(SUPPORT, h, w, video.device)
    for i in range(0, N, chunk):
        qc = qq[i:i + chunk][None]
        qs = torch.cat([qc, torch.cat([torch.zeros_like(g[..., :1]), g], -1)], 1)
        tr, vi, co, _ = model(video=v, queries=qs, iters=6)
        if backward:
            inv = qs.clone()
            inv[..., 0] = T - inv[..., 0] - 1
            tr2, vi2, co2, _ = model(video=v.flip(1), queries=inv, iters=6)
            tr2, vi2, co2 = tr2.flip(1), vi2.flip(1), co2.flip(1)
            ar = torch.arange(T, device=v.device)[None, :, None]
            m = ar < qs[:, None, :, 0]
            tr = torch.where(m[..., None], tr2, tr)
            vi = torch.where(m, vi2, vi)
            co = torch.where(m, co2, co)
        n = qc.shape[1]
        tr, vi, co = tr[0, :, :n], vi[0, :, :n], co[0, :, :n]
        tq = qc[0, :, 0].long()
        ar = torch.arange(n, device=v.device)
        tr[tq, ar] = qc[0, :, 1:]
        out_t.append(tr)
        out_v.append(vi)
        out_c.append(co)
    tr = torch.cat(out_t, 1)
    tr[..., 0] /= sx
    tr[..., 1] /= sy
    return tr.float().cpu().numpy(), torch.cat(out_v, 1).float().cpu().numpy(), torch.cat(out_c, 1).float().cpu().numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default="/workspace/ct3")
    ap.add_argument("--arms", default=",".join(ARMS))
    a = ap.parse_args()
    inp, out, root = Path(a.input), Path(a.out), Path(a.root)
    out.mkdir(parents=True, exist_ok=True)
    rep = {"errors": [], "arms": {}, "timings": {}}
    t0 = time.time()
    code, ck = setup(root)
    rep["timings"]["setup_s"] = round(time.time() - t0, 1)
    sys.path.insert(0, str(code))
    import cv2
    import torch
    from cotracker.models.build_cotracker import build_cotracker
    rep["versions"] = {"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                       "gpu_mem_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1),
                       "cotracker_commit": CT3_COMMIT, "ckpt": CKPT_URL, "ckpt_bytes": ck.stat().st_size,
                       "ckpt_sha256": sha256(ck)}
    log(rep["versions"])
    meta = json.loads((inp / "meta.json").read_text())
    frames = list(meta["frames"])
    F0 = frames[0]
    Q = np.load(inp / "queries.npz")
    qf, qxy = Q["q_frame"].astype(int), Q["q_xy"].astype(np.float32)
    t1 = time.time()
    imgs = np.stack([cv2.imread(str(inp / "frames" / f"{f:06d}.jpg"))[:, :, ::-1] for f in frames])
    rep["timings"]["read_frames_s"] = round(time.time() - t1, 1)
    H, W = imgs.shape[1:3]
    model = build_cotracker(str(ck), offline=True, window_len=60).cuda().eval()
    for name in a.arms.split(","):
        cfg = ARMS[name]
        size = tuple(cfg["size"]) if cfg["size"] else tuple(model.model_resolution)
        segs = meta["segments"][cfg["segments"]]
        Nq = len(qf)
        tracks = np.full((Nq, len(frames), 2), np.nan, np.float32)
        vis = np.zeros((Nq, len(frames)), np.float16)
        conf = np.zeros((Nq, len(frames)), np.float16)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        arm = {"size_hw": list(size), "segments": segs, "seg": []}
        ta = time.time()
        try:
            for s0, s1 in segs:
                sel = np.where((qf >= s0) & (qf <= s1))[0]
                i0, i1 = s0 - F0, s1 - F0 + 1
                v = torch.from_numpy(imgs[i0:i1].copy()).permute(0, 3, 1, 2).float()[None].cuda()
                q = np.c_[qf[sel] - s0, qxy[sel]].astype(np.float32)
                chunk, ts = len(sel), time.time()
                while True:
                    try:
                        with torch.no_grad():
                            tr, vi, co = run_sparse(model, v, q, size, backward=True, chunk=chunk)
                        break
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        if chunk <= 32:
                            raise
                        chunk = max(32, chunk // 2)
                        log(f"{name} {s0}-{s1}: OOM, chunk -> {chunk}")
                torch.cuda.synchronize()
                tracks[sel, i0:i1] = np.transpose(tr, (1, 0, 2))
                vis[sel, i0:i1] = vi.T
                conf[sel, i0:i1] = co.T
                arm["seg"].append({"frames": [s0, s1], "queries": int(len(sel)), "chunk": int(chunk),
                                   "seconds": round(time.time() - ts, 2)})
                log(f"{name} {s0}-{s1}: {len(sel)} queries, chunk {chunk}, {time.time() - ts:.1f} s")
                del v
        except Exception as ex:  # noqa: BLE001
            rep["errors"].append(f"{name}: {type(ex).__name__}: {ex}")
            log("ERROR", rep["errors"][-1])
            continue
        arm["seconds"] = round(time.time() - ta, 2)
        arm["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        arm["frames_per_s"] = round(len(frames) / max(arm["seconds"], 1e-6), 1)
        np.savez_compressed(out / f"{name}.npz", tracks=tracks, vis=vis, conf=conf, frames=np.asarray(frames))
        rep["arms"][name] = arm
        log(name, {k: arm[k] for k in ("seconds", "peak_vram_gb")})
    rep["timings"]["total_s"] = round(time.time() - t0, 1)
    (out / "ct3_pod.json").write_text(json.dumps(rep, indent=1))
    return 0 if not rep["errors"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
