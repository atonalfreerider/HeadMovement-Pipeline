#!/usr/bin/env python3
"""DiffLocks inference on the pod for dancecap.mono4d.cand_difflocks (venv of remote/cand_difflocks/install.sh).

    python difflocks_pod.py --inputs <inputs.json> --out <pod_out> --repo <difflocks clone> --ckpt-dir <checkpoints>

One DiffLocksInference (DINOv2 + diffusion + strand VAE + rgb2material) for every input of inputs.json; per input
<out>/<name>/: strands_f16.npz (positions (N, 256, 3) float16, template frame, metres), scalp_texture.npz, hair.json
(material), rgb.png (DiffLocks' own 770 px MediaPipe crop), result.json. <out>/infer.json: environment + per-input
timings. The float32 npz of the first input is kept in /workspace/difflocks_cand/tmp for the optional Blender export.

NATTEN: when `import natten` fails (no binary wheels exist for torch 2.5; building it takes ~30 min) an exact
pure-PyTorch implementation of na2d_qk / na2d_av (NATTEN's clamped window: start = clamp(i - k//2, 0, L - k)) is
registered as `natten` before k_diffusion is imported, and checked against a brute-force loop first. flash_attn: the
prebuilt wheel when it imports, else the same functions over torch SDPA.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
import traceback
import types

import numpy as np


def natten_shim():
    import torch

    def _win(L, k, dev):
        i = torch.arange(L, device=dev)
        s = (i - k // 2).clamp(0, L - k)
        return s[:, None] + torch.arange(k, device=dev)[None]

    def _nbr(x, k):  # [B, nh, H, W, D] -> [B, nh, H, W, k*k, D]
        B, nh, H, W, D = x.shape
        ih, iw = _win(H, k, x.device), _win(W, k, x.device)
        y = x[:, :, ih]                 # B nh H kh W D
        y = y[:, :, :, :, iw]           # B nh H kh W kw D
        return y.permute(0, 1, 2, 4, 3, 5, 6).reshape(B, nh, H, W, k * k, D)

    def na2d_qk(query, key, kernel_size, dilation=1, is_causal=False, rpb=None, **kw):
        assert dilation in (1, (1, 1)) and not is_causal and rpb is None, "shim: dilation 1, non-causal, no rpb only"
        k = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        return torch.einsum("bnhwd,bnhwkd->bnhwk", query, _nbr(key, k))

    def na2d_av(attn, value, kernel_size, dilation=1, is_causal=False, **kw):
        k = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        return torch.einsum("bnhwk,bnhwkd->bnhwd", attn, _nbr(value, k))

    def na2d(q, k, v, kernel_size, dilation=1, is_causal=False, rpb=None, scale=None, **kw):  # [B, H, W, nh, D]
        q, k, v = (t.permute(0, 3, 1, 2, 4) for t in (q, k, v))
        sc = q.shape[-1] ** -0.5 if scale is None else scale
        a = torch.softmax(na2d_qk(q * sc, k, kernel_size).float(), -1).to(v.dtype)
        return na2d_av(a, v, kernel_size).permute(0, 2, 3, 1, 4)

    m = types.ModuleType("natten")
    f = types.ModuleType("natten.functional")
    f.na2d_qk, f.na2d_av, f.na2d = na2d_qk, na2d_av, na2d
    m.functional = f
    m.has_fused_na = lambda: False
    m.use_kv_parallelism_in_fused_na = lambda *a, **k: None
    m.__version__ = "pure-pytorch-shim"
    sys.modules["natten"] = m
    sys.modules["natten.functional"] = f
    return m


def natten_selftest(nat, device="cuda") -> dict:
    """Shim (or real natten, non-fused path) vs a brute-force neighbourhood loop."""
    import torch
    g = torch.Generator(device="cpu").manual_seed(0)
    B, nh, H, W, D, k = 1, 2, 11, 9, 8, 7
    q, kk, v = (torch.randn(B, nh, H, W, D, generator=g).to(device) for _ in range(3))
    qk = nat.functional.na2d_qk(q, kk, k)
    a = torch.softmax(qk, -1)
    out = nat.functional.na2d_av(a, v, k)
    ref = torch.zeros_like(out)
    for i in range(H):
        si = min(max(i - k // 2, 0), H - k)
        for j in range(W):
            sj = min(max(j - k // 2, 0), W - k)
            K_ = kk[:, :, si:si + k, sj:sj + k].reshape(B, nh, k * k, D)
            V_ = v[:, :, si:si + k, sj:sj + k].reshape(B, nh, k * k, D)
            w = torch.softmax(torch.einsum("bnd,bnkd->bnk", q[:, :, i, j], K_), -1)
            ref[:, :, i, j] = torch.einsum("bnk,bnkd->bnd", w, V_)
    err = float((out - ref).abs().max())
    return {"max_abs_err": err, "ok": err < 1e-4}


def flash_shim():
    import torch
    import torch.nn.functional as F

    def flash_attn_kvpacked_func(q, kv, dropout_p=0.0, softmax_scale=None, causal=False, **kw):
        k, v = kv.unbind(2)
        o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                           dropout_p=dropout_p, is_causal=causal, scale=softmax_scale)
        return o.transpose(1, 2)

    def flash_attn_qkvpacked_func(qkv, dropout_p=0.0, softmax_scale=None, causal=False, **kw):
        q, k, v = qkv.unbind(2)
        return flash_attn_kvpacked_func(q, torch.stack([k, v], 2), dropout_p, softmax_scale, causal)

    m = types.ModuleType("flash_attn")
    m.flash_attn_kvpacked_func, m.flash_attn_qkvpacked_func = flash_attn_kvpacked_func, flash_attn_qkvpacked_func
    m.__version__ = "sdpa-shim"
    sys.modules["flash_attn"] = m
    return m


def face_picker(mp_obj, prefer_xy, rec: dict):
    """Replacement for DiffLocks' Mediapipe.run: same detector and outputs, but when the input lists the expected face
    position (normalised x, y: the projection of her fitted face centre into our crop) the detection NEAREST to it is
    used instead of the largest one. Session 20261019-165713: on phone 02 the largest 'face' (min confidence 0.1) was
    the 'Dream' print on her T-shirt and DiffLocks grew hair for the shirt crop."""
    import mediapipe as mp

    def run(rgb_image_numpy):
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image_numpy)
        faces = mp_obj.detector.detect(image).face_landmarks
        rec["n_faces"] = len(faces)
        if not faces:
            return None, None
        boxes = []
        for lm in faces:
            xs, ys = [q.x for q in lm], [q.y for q in lm]
            boxes.append((min(xs), min(ys), max(xs), max(ys)))
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
        largest = int(np.argmax(areas))
        if prefer_xy is None:
            k = largest
        else:
            dist = [((b[0] + b[2]) / 2 - prefer_xy[0]) ** 2 + ((b[1] + b[3]) / 2 - prefer_xy[1]) ** 2 for b in boxes]
            k = int(np.argmin(dist))
        rec.update(boxes=[[round(float(v), 4) for v in b] for b in boxes], areas=[round(float(a), 5) for a in areas],
                   largest=largest, chosen=k, prefer_xy=prefer_xy,
                   chosen_centre=[round(float((boxes[k][0] + boxes[k][2]) / 2), 4),
                                  round(float((boxes[k][1] + boxes[k][3]) / 2), 4)])
        lm = faces[k]
        px = np.zeros((478, 3))
        for i, q in enumerate(lm):
            px[i] = [q.x * image.width, q.y * image.height, q.z]
        return px, [(q.x, q.y) for q in lm]
    return run


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--nr-iters", type=int, default=100)
    ap.add_argument("--only", default="", help="comma list of input names")
    a = ap.parse_args()
    t_all = time.time()
    a.out, a.inputs = os.path.abspath(a.out), os.path.abspath(a.inputs)  # before the chdir into the repo
    os.makedirs(a.out, exist_ok=True)
    inputs = json.load(open(a.inputs))
    for x in inputs:  # image paths are relative to the job's cwd (dancecap/)
        x["img_abs"] = os.path.abspath(x["img"])
    if a.only:
        keep = set(a.only.split(","))
        inputs = [x for x in inputs if x["name"] in keep]
    import torch
    env = {"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
           "gpu_mem_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "nr_iters": a.nr_iters}
    try:
        import natten
        nat = natten
        env["natten"] = getattr(natten, "__version__", "?")
    except Exception as e:  # noqa: BLE001
        env["natten_import_error"] = f"{type(e).__name__}: {e}"[:200]
        nat = natten_shim()
        env["natten"] = nat.__version__
    env["natten_selftest"] = natten_selftest(nat)
    print("[difflocks] natten", env["natten"], env["natten_selftest"], flush=True)
    if not env["natten_selftest"]["ok"]:
        raise SystemExit("natten self-test failed")
    try:
        import flash_attn
        env["flash_attn"] = flash_attn.__version__
    except Exception as e:  # noqa: BLE001
        env["flash_attn_import_error"] = f"{type(e).__name__}: {e}"[:200]
        env["flash_attn"] = flash_shim().__version__
    print("[difflocks] flash_attn", env["flash_attn"], flush=True)
    sys.path.insert(0, a.repo)
    os.chdir(a.repo)
    import mediapipe
    env["mediapipe"] = mediapipe.__version__
    from inference.img2hair import DiffLocksInference
    ck = a.ckpt_dir
    t0 = time.time()
    dl = DiffLocksInference(f"{ck}/strand_vae/strand_codec.pt", f"{a.repo}/configs/config_scalp_texture_conditional.json",
                            f"{ck}/difflocks_diffusion/scalp_v9_40k_06730000_ema.pth", f"{ck}/rgb2material/rgb2material.pt",
                            nr_iters_denoise=a.nr_iters)
    env["load_s"] = round(time.time() - t0, 1)
    print(f"[difflocks] models loaded in {env['load_s']} s", flush=True)
    res = []
    tmp = "/workspace/difflocks_cand/tmp"
    os.makedirs(tmp, exist_ok=True)
    for n, inp in enumerate(inputs):
        od = os.path.join(a.out, inp["name"])
        os.makedirs(od, exist_ok=True)
        img = inp["img_abs"]
        seed = int(inp.get("seed", 5))
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.cuda.reset_peak_memory_stats()
        t1 = time.time()
        r = {"name": inp["name"], "img": img, "seed": seed, "face": {}}
        dl.mediapipe_img.run = face_picker(dl.mediapipe_img, inp.get("face_xy"), r["face"])
        try:
            out = dl.file2hair(img, od)
            if out is None:
                raise RuntimeError("no result")
            pts, mat = out
            P = pts.float().cpu().numpy()
            np.savez_compressed(os.path.join(od, "strands_f16.npz"), positions=P.astype(np.float16))
            big = os.path.join(od, "difflocks_output_strands.npz")
            if n == 0 and os.path.exists(big):
                shutil.move(big, os.path.join(tmp, "difflocks_output_strands.npz"))
                json.dump({"name": inp["name"]}, open(os.path.join(tmp, "which.json"), "w"))
            elif os.path.exists(big):
                os.remove(big)
            L = np.linalg.norm(np.diff(P, axis=1), axis=2).sum(1)
            r.update(ok=True, n_strands=int(P.shape[0]), pts=int(P.shape[1]), len_p50=float(np.median(L)),
                     len_p95=float(np.percentile(L, 95)),
                     material=({k: float(v.item()) for k, v in mat.items() if hasattr(v, "item") and v.numel() == 1}
                               if isinstance(mat, dict) else None))
        except Exception as e:  # noqa: BLE001  (TypeError when MediaPipe finds no face: rgb2hair returns None)
            r.update(ok=False, error=f"{type(e).__name__}: {e}"[:300], trace=traceback.format_exc()[-1500:])
        r["seconds"] = round(time.time() - t1, 1)
        r["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        json.dump(r, open(os.path.join(od, "result.json"), "w"), indent=1)
        print(f"[difflocks] {inp['name']}: {r.get('n_strands')} strands, ok {r['ok']}, {r['seconds']} s, "
              f"peak {r['peak_gpu_gb']} GB {r.get('error', '')}", flush=True)
        res.append(r)
    env["total_s"] = round(time.time() - t_all, 1)
    json.dump({"env": env, "results": res}, open(os.path.join(a.out, "infer.json"), "w"), indent=1)
    n_ok = sum(r["ok"] for r in res)
    print(f"[difflocks] {n_ok}/{len(res)} inputs ok in {env['total_s']} s", flush=True)
    return 0 if n_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
