"""Export of a trained AGS model (local after the download; no GPU needed for the PLYs).

  canonical_<role>.ply        rest pose (shaped T-pose of that dancer, no pose correction, no residual; the follow's
                              file includes her hair on the groom's rest strands), standard 3DGS PLY
  frames/frame_<n:05d>.ply    every grid frame of the window, both dancers + hair posed with the pose corrections,
                              residuals and hair tracks, world frame (right-handed, Y up, metres, floor y = 0; Unity =
                              (x, y, -z), as the avatars), SH degree --sh (Unity's disc preview reads DC only)
  splat_sequence.json         the CaptureManifest.splat_sequence block (dir, pattern, fps, t0, scene_to_unity)
`render`: any bundle camera (or 'novel') at any time of the window -> JPEG (real | splat when a photo exists).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from . import ROLES
from .binding import body_gaussians, matrix_to_quat, triangle_frames
from .data import load_bundle, read_view
from .model import AGSModel


def _device(pref: str = "cuda") -> str:
    import torch
    return pref if pref == "cpu" or torch.cuda.is_available() else "cpu"


def _ply_arrays(Gs: dict, sh: int, mask=None):
    """means, wxyz quats, log scales, logit opacities, sh0 (N, 3), shN (N, K-1, 3) as numpy."""
    import torch
    with torch.no_grad():
        sel = (lambda x: x) if mask is None else (lambda x: x[mask])  # noqa: E731
        q = matrix_to_quat(sel(Gs["R"]))
        op = sel(Gs["opac"]).clamp(1e-6, 1 - 1e-6)
        shc = sel(Gs["sh"])
        K = (sh + 1) ** 2
        return (sel(Gs["means"]).cpu().numpy(), q.cpu().numpy(), torch.log(sel(Gs["scales"]).clamp_min(1e-8)).cpu().numpy(),
                torch.logit(op).cpu().numpy(), shc[:, 0].cpu().numpy(), shc[:, 1:K].cpu().numpy())


def _cap_mask(Gs: dict, max_n: int | None):
    """Keep the max_n Gaussians of largest opacity x volume (None = all)."""
    import torch
    n = len(Gs["opac"])
    if not max_n or n <= max_n:
        return None
    score = Gs["opac"] * Gs["scales"].prod(1)
    keep = torch.zeros(n, dtype=torch.bool, device=score.device)
    keep[torch.topk(score, max_n).indices] = True
    return keep


def canonical(model: AGSModel, role: str, B: dict) -> dict:
    """Rest-pose Gaussians of one dancer (+ hair when this dancer has it)."""
    import torch
    g = model.groups[role]
    V = model.v_rest[role]
    fr = triangle_frames(V, model.faces)
    lo, hi = model._normal_range_t(g.buf["region"])
    kr = model.k_rest[role][g.buf["face"]]
    off = torch.stack([0.5 * kr * torch.tanh(g.p["off"][:, 0]), 0.5 * kr * torch.tanh(g.p["off"][:, 1]),
                       lo + (hi - lo) * torch.sigmoid(g.p["off"][:, 2])], 1)
    mu, Rw, sc = body_gaussians(V, model.faces, g.buf["face"], g.buf["bary"], off, model.k_rest[role],
                                g.p["log_scale"], g.p["quat"], None, model.cfg.scale_cap, fr)
    out = {"means": [mu], "R": [Rw], "scales": [sc], "opac": [torch.sigmoid(g.p["opac"])],
           "sh": [torch.cat([g.p["sh0"], g.p["shN"]], 1)]}
    if model.hair is not None and model.hair_role == role and "hair" in model.groups:
        from .binding import quat_to_matrix, segment_frames
        h = model.groups["hair"]
        J15 = torch.as_tensor(B["bodies"][f"{role}_J_rest"][15], dtype=torch.float32, device=model.dev)
        X = model.h_rest + J15
        ref = torch.tensor([0.0, 0.0, 1.0], device=model.dev)
        Rs, L = segment_frames(X, ref)
        gi, si, u = h.buf["guide"], h.buf["seg"], h.buf["u"]
        nps = max(1, model.cfg.n_hair_per_seg)
        Lg = L[gi, si]
        loc = torch.stack([0.5 * Lg / nps * torch.tanh(h.p["off"][:, 0]), 0.03 * torch.tanh(h.p["off"][:, 1]),
                           0.03 * torch.tanh(h.p["off"][:, 2])], 1)
        R_ = Rs[gi, si]
        out["means"].append(X[gi, si] + u[:, None] * (X[gi, si + 1] - X[gi, si]) + (R_ @ loc[..., None])[..., 0])
        out["R"].append(R_ @ quat_to_matrix(h.p["quat"]))
        out["scales"].append(torch.exp(h.p["log_scale"].clamp(max=math.log(0.05))))
        out["opac"].append(torch.sigmoid(h.p["opac"]))
        out["sh"].append(torch.cat([h.p["sh0"], h.p["shN"]], 1))
    return {k: torch.cat(v, 0) for k, v in out.items()}


def export(bundle: Path, out: Path, sh: int = 1, max_gaussians: int | None = None, log=print,
           device: str = "cuda") -> dict:
    import torch

    from ..room import save_ply
    B = load_bundle(bundle)
    dev = _device(device)
    model = AGSModel.load(B, out / "model.pt", dev)
    sh = min(sh, model.cfg.sh_degree)
    res = {"canonical": {}, "frames": 0}
    with torch.no_grad():
        for r in ROLES:
            Gs = canonical(model, r, B)
            n = len(Gs["opac"])
            save_ply(out / f"canonical_{r}.ply", *_ply_arrays(Gs, sh))
            res["canonical"][r] = n
        fd = out / "frames"
        fd.mkdir(parents=True, exist_ok=True)
        for old in fd.glob("frame_*.ply"):
            old.unlink()
        sizes = []
        for n, k in enumerate(model.k_window):
            t = float(model.t_grid[k])
            Gs = model.gaussians(t)
            mask = _cap_mask(Gs, max_gaussians)
            above = Gs["means"][:, 1] >= model.cfg.floor_m  # nothing under the floor
            mask = above if mask is None else (mask & above)
            save_ply(fd / f"frame_{n:05d}.ply", *_ply_arrays(Gs, sh, mask))
            sizes.append((fd / f"frame_{n:05d}.ply").stat().st_size)
        res["frames"] = len(model.k_window)
    dt = float(np.median(np.diff(model.t_grid)))
    seq = {"dir": "frames", "pattern": "frame_{0:05d}.ply", "fps": round(1.0 / dt, 4),
           "t0": round(float(model.t_grid[model.k_window[0]]), 5), "frames": res["frames"],
           "scene_to_unity": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
           "frame": "world: right-handed, Y up, metres, floor y = 0 (the avatars' frame); Unity = (x, y, -z)",
           "sh_degree": sh, "max_gaussians": max_gaussians, "layer": "splats (off by default; dancers only)",
           "likeness": "work/ and git-ignored StreamingAssets only - never the head-movement repo"}
    (out / "splat_sequence.json").write_text(json.dumps(seq, indent=1))
    res["mb"] = round(sum(sizes) / 1e6, 1)
    log(f"[dancer_splats] export: canonical {res['canonical']}, {res['frames']} frames ({res['mb']} MB, SH {sh}) -> "
        f"{out}")
    return res


def render_at(bundle: Path, out: Path, cam: str, t: float, path: Path | None = None, log=print) -> Path:
    """Render the splats from a bundle camera (the view of that camera nearest t; 'novel' = the fixed novel camera)
    at reference time t; with a photo of that view: real | splat side by side."""
    import cv2
    import torch

    from .evaluate import render_view
    B = load_bundle(bundle)
    model = AGSModel.load(B, out / "model.pt", _device("cuda"))
    w0, w1 = model.window
    if not (model.t_grid[0] <= t <= model.t_grid[-1]):
        raise SystemExit(f"dancer_splats: t {t} is outside the trained grid {model.t_grid[0]:.2f}-{model.t_grid[-1]:.2f}")
    if cam == "novel":
        nov = B["novel"]
        if not nov:
            raise SystemExit("dancer_splats: the bundle has no novel camera")
        v = {"i": -1, "cam": "novel", "split": "novel", "K": nov["K"], "R": nov["R"], "t_w": nov["t_w"],
             "W": nov["W"], "H": nov["H"]}
        photo = None
    else:
        cand = [v for v in B["views"] if v["cam"] == cam]
        if not cand:
            raise SystemExit(f"dancer_splats: no view of camera {cam} in the bundle")
        v = dict(min(cand, key=lambda x: abs(x["t"] - t)))
        photo = read_view(bundle, v)[0] if abs(v["t"] - t) < 0.02 else None
    v["t"] = float(t)
    r = render_view(model, v, model.cam_app_index(), model.cfg.sh_degree)
    img = (np.clip(r["rgb"], 0, 1) * 255).round().astype(np.uint8)
    if photo is not None:
        img = np.hstack([photo, img])
    path = path or out / "renders" / f"render_{cam}_{t:.3f}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
    torch.cuda.empty_cache()
    log(f"[dancer_splats] render cam {cam} t {t:.3f} -> {path}")
    return path
