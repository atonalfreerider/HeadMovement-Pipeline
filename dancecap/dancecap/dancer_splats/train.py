"""AGS training: S1 canonical (body Gaussians, colour transforms, pose deltas; residuals and hair tracks frozen, SH 0),
S2 dynamic (+ cosine residuals + hair node tracks, SH 1), S3 polish (SH 2, no relocation). Each step renders up to
`views_per_step` training views of one instant (other cameras within a frame), both dancers in one gsplat call.

Progress (docs/PIPELINE_PROGRESS.md): dancecap.progress/1 events splats/S1..S3 (steps, psnr, Gaussians) and the L0 line
    [dancer_splats] step N/M stage S1 psnr X iou Y N_gauss Z Ns
work/<take>/dancer_splats/progress.json carries the latest of both (+ the ETA from the measured rate)."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from .. import progress as _progress
from . import R_FACE, R_HAND, ROLES, STAGES, SplatConfig, config_dict
from .data import decode_labels, load_bundle, read_view
from .model import AGSModel, K_gs, cam_centre, interpenetration, viewmat


class Logger:
    def __init__(self, path: Path | None, log=print):
        self.path, self.log = path, log
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, msg: str):
        self.log(msg)
        if self.path is not None:
            try:
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(msg + "\n")
            except OSError:
                pass


def write_json_atomic(path: Path, obj) -> None:
    import os
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(obj, indent=1, default=float))
    os.replace(tmp, path)


def photo_loss(comp, target, valid, lam: float):
    """Masked (1 - lam) L1 + lam D-SSIM over the valid pixels, per image: comp / target (B, H, W, 3), valid (B, H, W)
    -> (B,) (masked pixels take the render's own value in the SSIM window)."""
    import torch

    from ..room import ssim_map
    vf = valid.float()
    nv = vf.sum((1, 2)).clamp_min(1)
    l1 = ((comp - target).abs().sum(-1) * vf).sum((1, 2)) / (nv * 3)
    tf = torch.where(valid[..., None], target, comp.detach())
    ss = ssim_map(comp.permute(0, 3, 1, 2).clamp(0, 1), tf.permute(0, 3, 1, 2)).mean(1)
    return (1 - lam) * l1 + lam * (1 - (ss * vf).sum((1, 2)) / nv)


class Schedule:
    """Stage lengths that can shrink (time budget) while keeping the stage order."""

    def __init__(self, iters):
        self.len = [int(x) for x in iters]

    @property
    def total(self) -> int:
        return int(sum(self.len))

    def ends(self) -> list[int]:
        return list(np.cumsum(self.len))

    def stage(self, step: int) -> tuple[int, int]:
        """(stage index, step within the stage)."""
        start = 0
        for i, n in enumerate(self.len):
            if step < start + n:
                return i, step - start
            start += n
        return len(self.len) - 1, self.len[-1] - 1

    def truncate(self, step: int) -> None:
        """End the schedule at `step` (steps done): the current stage ends there, later stages are dropped (0)."""
        si, within = self.stage(max(0, step - 1))
        self.len = [n if i < si else (within + 1 if i == si else 0) for i, n in enumerate(self.len)]

    def shrink(self, step: int, factor: float, min_frac: float = 0.15) -> None:
        si, within = self.stage(step)
        for i in range(si, len(self.len)):
            if i == si:
                rem = self.len[i] - within
                self.len[i] = within + max(1, int(rem * factor))
            else:
                self.len[i] = max(int(self.len[i] * min_frac), int(self.len[i] * factor), 1)


def _view_tensors(model: AGSModel, v: dict):
    import torch
    v["_vm"] = torch.as_tensor(viewmat(v), dtype=torch.float32, device=model.dev)
    v["_K"] = torch.as_tensor(K_gs(v), dtype=torch.float32, device=model.dev)
    v["_C"] = torch.as_tensor(cam_centre(v), dtype=torch.float32, device=model.dev)


def batch_losses(model: AGSModel, Gs: dict, views: list[dict], data: dict, bg, cfg: SplatConfig,
                 sh_degree: int) -> dict:
    """Per-view terms (B,) of a batch rendered in ONE gsplat call: photometric (on a random background), silhouette,
    role ownership, hair; crops padded to the largest one with ignore labels."""
    import torch
    import torch.nn.functional as F
    r = model.render_b(Gs, views, sh_degree)
    H, W = r["alpha"].shape[1:]
    imgs, labs, js = [], [], []
    for v in views:
        img, lab = data[v["i"]]
        ph, pw = H - img.shape[0], W - img.shape[1]
        imgs.append(F.pad(img, (0, 0, 0, pw, 0, ph)) if (ph or pw) else img)
        labs.append(F.pad(lab, (0, pw, 0, ph), value=255) if (ph or pw) else lab)
        js.append(model.app_index[v["i"]])
    gt = torch.stack(imgs).float() / 255
    m = decode_labels(torch.stack(labs))
    valid, fg = m["valid"], m["fg"]
    ji = model.h2d(js, torch.long)
    A, b = model.app.A.index_select(0, ji), model.app.b.index_select(0, ji)
    al = r["alpha"][..., None]
    comp = torch.einsum("bhwc,bdc->bhwd", r["rgb"], A) + al * b[:, None, None, :] + (1 - al) * bg
    target = torch.where(fg[..., None], gt, bg.expand_as(gt))
    out = {"photo": photo_loss(comp, target, valid, cfg.ssim_lambda)}
    vf = valid.float()
    nv = vf.sum((1, 2)).clamp_min(1)
    out["sil"] = ((r["alpha"] - fg.float()).abs() * vf).sum((1, 2)) / nv
    ok = model.h2d([1.0 if v.get("swap_prob", 0.0) <= cfg.swap_max else 0.0 for v in views])
    rv = vf * (~m["both"]).float() * ok[:, None, None]
    nr = rv.sum((1, 2)).clamp_min(1)
    ro = r["roles"]
    out["role"] = (((ro[..., 0] - m["lead"].float()).abs() + (ro[..., 1] - m["follow"].float()).abs()) * rv).sum(
        (1, 2)) / nr
    out["hair"] = ((ro[..., 2] - m["hair"].float()).abs() * rv).sum((1, 2)) / nr if model.hair is not None else \
        torch.zeros_like(nr)
    with torch.no_grad():  # tensors: read back only at log time (a .item() per step would synchronise)
        sel = (fg & valid).float()
        mse = (((comp - gt) ** 2).mean(-1) * sel).sum((1, 2)) / sel.sum((1, 2)).clamp_min(1)
        out["psnr"] = (-10 * torch.log10(mse.clamp_min(1e-10))).mean()
        pa = (r["alpha"] > 0.5) & valid
        out["iou"] = ((pa & fg).sum((1, 2)) / ((pa | fg) & valid).sum((1, 2)).clamp_min(1)).mean()
    out["render"] = r
    return out


def regularisers(model: AGSModel, cfg: SplatConfig, dyn: bool, face_hd: bool, P: dict, ks: list[int]) -> dict:
    import torch

    from .binding import capsule_distance
    reg = {}
    dev = model.dev
    z = torch.zeros((), device=dev)
    op = sc = z
    prior = z
    for gname, g in model.groups.items():
        op = op + torch.sigmoid(g.p["opac"]).mean()
        sc = sc + torch.exp(g.p["log_scale"]).mean() / 0.01
        if gname in ROLES:
            regn = g.buf["region"]
            pm = (((regn & R_HAND) > 0) | (((regn & R_FACE) > 0) & (not face_hd))).float()
            rgb = g.p["sh0"][:, 0] * 0.28209479177387814 + 0.5
            prior = prior + ((rgb - g.buf["rgb0"]).abs().mean(-1) * pm).sum() / pm.sum().clamp_min(1)
    reg["opac"] = cfg.opacity_reg * op
    reg["scale"] = cfg.scale_reg * sc
    reg["prior"] = cfg.w_prior * prior
    reg["app"] = cfg.app_reg * model.app.reg()
    # pose deltas: pull to MAMMA + acceleration over the grid (in units of the caps)
    cr, ct = math.radians(cfg.pose_cap_deg), cfg.pose_cap_m
    allk = torch.arange(model.Tg, device=dev)
    pr = []
    for ri in range(len(ROLES)):
        r, dT = model.pose_deltas(ri, allk)
        pr.append(((r / cr) ** 2).mean() + ((dT / ct) ** 2).mean())
        if model.Tg >= 3:
            ar = (r[2:] - 2 * r[1:-1] + r[:-2]) / cr
            at = (dT[2:] - 2 * dT[1:-1] + dT[:-2]) / ct
            pr.append(cfg.w_pose_acc / max(cfg.w_pose, 1e-9) * ((ar ** 2).mean() + (at ** 2).mean()))
    reg["pose"] = cfg.w_pose * sum(pr)
    if dyn:
        rr = z
        for gname in ROLES:
            g = model.groups[gname]
            K = g.p["resid"].shape[1]
            w = (torch.arange(1, K + 1, device=dev, dtype=torch.float32) / K) ** 2
            rr = rr + ((g.p["resid"] / cfg.resid_cap_cloth) ** 2 * w[None, :, None]).mean()
        reg["resid"] = cfg.w_resid * rr
        if model.nodes is not None:
            off = model.node_offsets(allk)
            loc = model.h_rest[None] + off
            Lk = (loc[:, :, 1:] - loc[:, :, :-1]).norm(dim=-1)
            stretch = torch.relu((Lk / model.h_len0[None].clamp_min(1e-6) - 1).abs() - 0.05) ** 2
            reg["hair_len"] = cfg.w_hair_len * stretch.mean()
            if model.Tg >= 3:
                acc = (off[2:] - 2 * off[1:-1] + off[:-2]) / 0.02
                reg["hair_acc"] = cfg.w_hair_acc * (acc ** 2).mean()
            reg["hair_l2"] = cfg.w_hair_l2 * ((off / 0.05) ** 2).mean()
            d = (loc[:, :, 2:] - model.h_centre).norm(dim=-1)
            col = (torch.relu(model.h_radius + 0.005 - d) / 0.01) ** 2
            # capsules (neck, shoulders, arms, back: the groom's colliders) at this step's frames
            hr = model.hair_role
            if len(model.h_colliders) and P is not None:  # all capsules at once: (frames, capsules, nodes)
                X, _, _ = model.hair_nodes(ks, P[hr]["D"])
                ca, cb = model.collider_ends(ks, P[hr])                       # with the groom's offsets
                dist = capsule_distance(X.reshape(len(ks), 1, -1, 3), ca, cb)
                pen = (torch.relu(model.h_col_r[None, :, None] + 0.003 - dist) / 0.01) ** 2 * model.h_col_mask
                col = col.mean() + pen.sum() / (model.h_col_mask.sum() * len(ks)).clamp_min(1)
            else:
                col = col.mean()
            reg["hair_col"] = cfg.w_hair_col * col
    return reg


def _sample_batch(rng, tr: list[dict], neigh: dict, n: int) -> list[dict]:
    v = tr[int(rng.integers(len(tr)))]
    nb = neigh.get(v["i"], [])
    pick = [v]
    if nb:
        sel = rng.choice(len(nb), size=min(n - 1, len(nb)), replace=False)
        pick += [nb[j] for j in sel]
    while len(pick) < n and len(tr) > len(pick):
        w = tr[int(rng.integers(len(tr)))]
        if all(w is not p for p in pick):
            pick.append(w)
    return pick


RUN_ONLY_FIELDS = ("time_budget_min", "kill_reserve_min", "device", "note", "log_every", "ckpt_every")


def train_key(cfg: SplatConfig) -> dict:
    """The config fields that define the trained model (a checkpoint / model.pt is reusable across runs that differ
    only in the time budget, kill-switch reserve, device, logging)."""
    return {k: v for k, v in config_dict(cfg).items() if k not in RUN_ONLY_FIELDS}


def trained_model_for(out: Path, man: dict, cfg: SplatConfig) -> bool:
    """model.pt + pose_delta.npz exist for this bundle and config (a retry session uploads them: train is skipped)."""
    mp = out / "model.pt"
    if not (mp.exists() and (out / "pose_delta.npz").exists()):
        return False
    try:
        import torch
        st = torch.load(mp, map_location="cpu", weights_only=False)
        same_cfg = {k: v for k, v in st["config"].items() if k not in RUN_ONLY_FIELDS} == train_key(cfg)
        return bool(same_cfg and (st.get("extra") or {}).get("bundle_created") == man["created"])
    except Exception:  # noqa: BLE001 - unreadable -> retrain
        return False


class StopFlag:
    """SIGTERM / SIGINT (the pod command's `timeout`, a kill) -> finish the current step, save model.pt and the pose
    deltas, exit normally (the outputs are packed either way)."""

    def __init__(self):
        self.why = None
        self._old = {}

    def __enter__(self):
        import signal
        for s in (getattr(signal, "SIGTERM", None), getattr(signal, "SIGINT", None)):
            if s is None:
                continue
            try:
                self._old[s] = signal.signal(s, self._handler)
            except (ValueError, OSError):  # not the main thread
                pass
        return self

    def _handler(self, signum, frame):  # noqa: ARG002
        self.why = f"signal {signum}"

    def __exit__(self, *exc):
        import signal
        for s, h in self._old.items():
            try:
                signal.signal(s, h)
            except (ValueError, OSError):
                pass
        return False


def train(bundle: Path, out: Path, cfg: SplatConfig, log=print, resume: bool = True) -> dict:
    import platform
    from collections import deque

    import torch

    from .. import gpu
    from ..env import is_pod
    out.mkdir(parents=True, exist_ok=True)
    L = Logger(out / "train.log", log)
    B = load_bundle(bundle)
    man = B["manifest"]
    if resume and trained_model_for(out, man, cfg):
        L(f"[dancer_splats] {out / 'model.pt'} is already trained for this bundle and config: train skipped "
          f"(--no-resume retrains)")
        _progress.note("model.pt present: train skipped", step="splats/train")
        rp = out / "report.json"
        return (json.loads(rp.read_text()).get("train") if rp.exists() else None) or {"skipped": True}
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    t_load = time.time()
    model = AGSModel(B, cfg, cfg.device)
    tr = [v for v in B["views"] if v["split"] == "train"]
    if not tr:
        raise SystemExit("dancer_splats: no training views in the bundle")
    data = {}
    for v in tr:
        img, lab = read_view(bundle, v)
        data[v["i"]] = (torch.as_tensor(img, device=model.dev), torch.as_tensor(lab, device=model.dev))
        _view_tensors(model, v)
    dt = float(np.median(np.diff(model.t_grid)))
    neigh = {v["i"]: [w for w in tr if w["cam"] != v["cam"] and abs(w["t"] - v["t"]) <= 0.75 * dt] for v in tr}
    sched = Schedule(cfg.iters)
    hist, step0, el0 = [], 0, 0.0
    ck = out / "cache" / "ckpt.pt"
    if resume and ck.exists():
        st = torch.load(ck, map_location="cpu", weights_only=False)
        st_key = {k: v for k, v in (st.get("config") or {}).items() if k not in RUN_ONLY_FIELDS}
        if st.get("bundle_created") == man["created"] and st_key == train_key(cfg):
            model.load_state(st["model"])
            step0, hist, sched.len = int(st["step"]), st.get("history", []), list(st["schedule"])
            # the budget covers the whole training on ONE host; a checkpoint from another pod (a retry session) starts
            # this session's budget afresh
            el0 = float(st.get("elapsed_s", 0.0)) if st.get("host") == platform.node() else 0.0
            L(f"[dancer_splats] resumed from {ck.name} at step {step0}/{sched.total} (from "
              f"{'this host' if el0 else st.get('host') or 'another host'})")
        else:
            L("[dancer_splats] checkpoint from another bundle / config: starting over")
    per_cam = ", ".join(f"{c}:{sum(v['cam'] == c for v in tr)}" for c in sorted({v["cam"] for v in tr}))
    gauss = ", ".join(f"{k} {len(g):,}" for k, g in model.groups.items())
    L(f"[dancer_splats] {len(tr)} training views loaded in {time.time() - t_load:.0f}s ({per_cam}); "
      f"Gaussians {gauss}; grid {model.Tg} frames; {cfg.note or 'custom'} schedule {sched.len} x "
      f"{cfg.views_per_step} views")
    _progress.plan([("splats/train", sched.total, "steps")], msg=f"dancer_splats train: stages {sched.len}")
    t0 = time.time() - el0  # elapsed carries over a resume (the time budget covers the whole training)
    budget_s = cfg.time_budget_min * 60 if cfg.time_budget_min > 0 else None
    kill_at = kill_deadline()
    if kill_at is not None:  # the pod's kill switch: leave time for eval, pack and download
        left_kill = kill_at - time.time() - cfg.kill_reserve_min * 60
        budget_s = min(budget_s or left_kill, left_kill + (time.time() - t0))
        L(f"[dancer_splats] pod kill switch in {(kill_at - time.time()) / 60:.1f} min: training budget "
          f"{budget_s / 60:.1f} min (reserve {cfg.kill_reserve_min:.0f} min for eval / pack / download)")
    rate = deque(maxlen=21)  # (time, step) every 25 steps: the step time over the last ~500 steps
    stopped = None
    run_psnr = run_iou = None
    acc_p, acc_i = [], []
    last_stage = None
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    with gpu.exclusive("dancer_splats"), StopFlag() as stop:
        step = step0
        while step < sched.total:
            si, within = sched.stage(step)
            stage = STAGES[si]
            if stage != last_stage:
                _progress.note(f"stage {stage} from step {step + 1} ({sched.len[si]} steps)", step="splats/train")
                L(f"[dancer_splats] stage {stage} from step {step + 1} ({sched.len[si]} steps)")
                last_stage = stage
            dyn = si >= 1
            sh_deg = 0 if si == 0 else (min(1, cfg.sh_degree) if si == 1 else cfg.sh_degree)
            for gname in ROLES:
                model.groups[gname].p["resid"].requires_grad_(dyn)
            if model.nodes is not None:
                model.nodes.requires_grad_(dyn)
            batch = _sample_batch(rng, tr, neigh, cfg.views_per_step)
            face_hd = any(v.get("face_hd") for v in batch)
            frames = {}
            for v in batch:
                k, _ = model.frame_index(v["t"])
                frames.setdefault(k, len(frames))
                frames.setdefault(k + 1, len(frames))
            ks = list(frames)
            P = model.posed(ks)
            bg = torch.rand(3, device=model.dev)
            Gs = model.gaussians_b([v["t"] for v in batch], P, frames, use_resid=dyn, hair_frozen=not dyn)
            terms = batch_losses(model, Gs, batch, data, bg, cfg, sh_deg)
            wv = model.h2d([float(v.get("weight", 1.0)) for v in batch])
            per = terms["photo"] + cfg.w_sil * terms["sil"] + cfg.w_role * terms["role"] + cfg.w_hair * terms["hair"]
            loss = (wv * per).sum() / len(batch)
            if cfg.w_pen > 0:  # the first view's instant: the bodies are shared by the step's views
                k, a = model.frame_index(batch[0]["t"])
                G0 = {"means": Gs["means"][0], "anchor": Gs["anchor"][0], "group": Gs["group"]}
                loss = loss + cfg.w_pen * interpenetration(model, G0, P, (frames[k], frames[k + 1]), a)
            if cfg.w_floor > 0:  # nothing under the floor (world y = 0; 1 cm tolerance)
                loss = loss + cfg.w_floor * ((torch.relu(cfg.floor_m - Gs["means"][0][:, 1]) / 0.01) ** 2).mean()
            psnrs, ious = [terms["psnr"]], [terms["iou"]]
            reg = regularisers(model, cfg, dyn, face_hd, P, ks)
            loss = loss + sum(reg.values())
            for gname, g in model.groups.items():
                if gname in ROLES:
                    regn = g.buf["region"]
                    pm = ((regn & R_HAND) > 0) | (((regn & R_FACE) > 0) & (not face_hd))
                    g.grad_scale = torch.where(pm, 0.1, 1.0).float()
            loss.backward()
            for gname, g in model.groups.items():
                g.scale_colour_grads()
                g.step()
            model.opt_pose.step()
            model.opt_app.step()
            if dyn and model.opt_nodes is not None:
                model.opt_nodes.step()
            for g in model.groups.values():
                g.zero_grad()
            model.opt_pose.zero_grad(set_to_none=True)
            model.opt_app.zero_grad(set_to_none=True)
            if model.opt_nodes is not None:
                model.opt_nodes.zero_grad(set_to_none=True)
            refine_stop = sched.ends()[1]
            if step < refine_stop:
                model.inject_noise(0.05 * cfg.lr_off / 2e-3)
                if step >= cfg.refine_start and step % cfg.refine_every == 0:
                    model.relocate_and_grow(step, grow=True)
            acc_p.append(torch.stack(psnrs).mean().detach())
            acc_i.append(torch.stack(ious).mean().detach())
            step += 1
            if step % 25 == 0 or step == sched.total or step == step0 + 1 or step % cfg.log_every == 0:
                for p_, i_ in zip(torch.stack(acc_p).tolist(), torch.stack(acc_i).tolist()):  # one sync per 25 steps
                    run_psnr = p_ if run_psnr is None else 0.95 * run_psnr + 0.05 * p_
                    run_iou = i_ if run_iou is None else 0.95 * run_iou + 0.05 * i_
                acc_p, acc_i = [], []
                _progress.tick("splats/train", step, sched.total, "steps", stage=stage, psnr=round(run_psnr, 2),
                               iou=round(run_iou, 3), gaussians=model.n_gaussians())
            el = time.time() - t0
            if step % 25 == 0 and step - step0 >= 25:  # (the first 25 steps carry the CUDA warm-up)
                rate.append((time.time(), step))
            s_per = ((rate[-1][0] - rate[0][0]) / (rate[-1][1] - rate[0][1])) if len(rate) >= 3 else None
            # hard stop: the budget is spent (or the command is being stopped) -> save what exists
            if (budget_s is not None and el >= budget_s) or stop.why:
                stopped = stop.why or f"time budget {budget_s / 60:.1f} min spent"
                old = list(sched.len)
                sched.truncate(step)
                _progress.plan([("splats/train", sched.total, "steps")], msg=f"stopped at step {step}: {stopped}")
                L(f"[dancer_splats] stopped at step {step}: {stopped}; schedule {old} -> {sched.len}")
                break
            # time budget: shrink the rest of the schedule so the whole training ends inside it (recent step time)
            if budget_s is not None and s_per is not None and step % 250 == 0:
                left = budget_s - el
                need = (sched.total - step) * s_per
                if need > left > 0:
                    f = 0.97 * left / need
                    old = list(sched.len)
                    sched.shrink(step, f)
                    _progress.plan([("splats/train", sched.total, "steps")],
                                   msg=f"schedule shrunk to the time budget: {sched.len}")
                    L(f"[dancer_splats] time budget {budget_s / 60:.1f} min: {s_per * 1000:.0f} ms/step -> "
                      f"schedule {old} -> {sched.len}")
            if step % cfg.log_every == 0 or step == sched.total or step == step0 + 1:
                n_g = model.n_gaussians()
                hist.append({"step": step, "stage": stage, "psnr": round(run_psnr, 3), "iou": round(run_iou, 4),
                             "n": n_g, "s": round(el, 1), "loss": round(float(loss), 5),
                             **{k: round(float(v), 6) for k, v in reg.items()}})
                L(f"[dancer_splats] step {step}/{sched.total} stage {stage} psnr {run_psnr:.2f} iou {run_iou:.3f} "
                  f"N_gauss {n_g} {el:.0f}s")
                write_json_atomic(out / "progress.json", {
                    "stage": stage, "step": step, "total": sched.total, "schedule": sched.len, "psnr": run_psnr,
                    "iou": run_iou, "gaussians": n_g, "elapsed_s": round(el, 1),
                    "eta_s": round(s_per * (sched.total - step), 1) if s_per else None,
                    "s_per_step": s_per, "updated": time.time()})
            if step % cfg.ckpt_every == 0 and step < sched.total:
                ck.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"model": model.state(), "step": step, "schedule": sched.len, "history": hist,
                            "bundle_created": man["created"], "config": config_dict(cfg), "host": platform.node(),
                            "elapsed_s": time.time() - t0}, ck)
        _progress.end("splats/train", done=sched.total, msg=f"stopped early: {stopped}" if stopped else None)
        train_s = time.time() - t0
        rep_pose = write_pose_delta(model, out / "pose_delta.npz")
        extra = {"bundle_created": man["created"], "take": man["take"], "schedule": sched.len, "train_s": train_s,
                 "history_tail": hist[-5:], "gpu": gpu_name, "stopped": stopped}
        model.save(out / "model.pt", extra)
        torch.cuda.empty_cache()
    rep = {"config": config_dict(cfg), "bundle": {k: man.get(k) for k in ("created", "take", "n_views", "n_train",
                                                                          "n_test_cam", "n_test_time", "window_frames",
                                                                          "bytes", "hair", "fits")},
           "host": platform.node(), "pod": is_pod(), "gpu": gpu_name, "when": time.strftime("%Y-%m-%d %H:%M:%S"),
           "train_s": round(train_s, 1), "schedule": sched.len, "steps": sched.total, "resumed_at": step0,
           "stopped": stopped, "s_per_step_recent": round(s_per, 5) if s_per else None,
           "gaussians": {k: len(g) for k, g in model.groups.items()}, "history": hist, "pose_delta": rep_pose,
           "final": hist[-1] if hist else None}
    old = json.loads((out / "report.json").read_text()) if (out / "report.json").exists() else {}
    old.pop("eval", None)
    old["train"] = rep
    write_json_atomic(out / "report.json", old)
    write_json_atomic(out / "progress.json", {"stage": "done", "step": sched.total, "total": sched.total,
                                              "psnr": run_psnr, "iou": run_iou, "gaussians": model.n_gaussians(),
                                              "elapsed_s": round(train_s, 1), "eta_s": 0, "updated": time.time()})
    L(f"[dancer_splats] trained {sched.total} steps in {train_s / 60:.1f} min: psnr {run_psnr:.2f} iou {run_iou:.3f}, "
      f"{model.n_gaussians():,} Gaussians -> {out / 'model.pt'}")
    if ck.exists():
        ck.unlink()
    return rep


def kill_deadline(path: str = "/workspace/.dancecap_kill_at") -> float | None:
    """Epoch seconds of the pod's kill switch (remote.pod_session writes it), None off a pod."""
    import os
    try:
        if os.environ.get("DANCECAP_POD") or os.environ.get("RUNPOD_POD_ID"):
            return float(Path(path).read_text().strip())
    except (OSError, ValueError):
        pass
    return None


def write_pose_delta(model: AGSModel, path: Path) -> dict:
    """dancer_splats/pose_delta.npz: the per-frame corrections (a photometric QA signal for the fit; smplx/ is never
    touched) and the hair node offsets' magnitude."""
    import torch
    allk = torch.arange(model.Tg, device=model.dev)
    out = {"t_grid": model.t_grid, "k_window": model.k_window,
           "joints": np.arange(22), "note": np.array("r: local axis-angle (rad) applied on the child side of each of "
                                                     "the 22 body joints; dT: root translation (m), world frame")}
    rep = {}
    with torch.no_grad():
        for ri, r in enumerate(ROLES):
            dr, dT = model.pose_deltas(ri, allk)
            dr, dT = dr.cpu().numpy(), dT.cpu().numpy()
            deg = np.degrees(np.linalg.norm(dr, axis=-1))
            out[f"{r}_r"] = dr.astype(np.float32)
            out[f"{r}_dT"] = dT.astype(np.float32)
            out[f"{r}_max_deg"] = deg.max(1).astype(np.float32)
            out[f"{r}_dT_cm"] = (np.linalg.norm(dT, axis=-1) * 100).astype(np.float32)
            kw = model.k_window
            top = kw[np.argsort(-deg[kw].max(1))[:5]]
            rep[r] = {"max_deg_median": round(float(np.median(deg[kw].max(1))), 3),
                      "max_deg_p95": round(float(np.percentile(deg[kw].max(1), 95)), 3),
                      "dT_cm_median": round(float(np.median(out[f'{r}_dT_cm'][kw])), 3),
                      "largest_at_t": [round(float(model.t_grid[k]), 3) for k in top]}
        if model.nodes is not None:
            off = model.node_offsets(allk).norm(dim=-1).cpu().numpy()
            out["hair_node_offset_m_mean"] = off.mean((1, 2)).astype(np.float32)
            out["hair_node_offset_m_max"] = off.max((1, 2)).astype(np.float32)
            rep["hair_node_offset_cm_mean"] = round(float(off[model.k_window].mean() * 100), 2)
    np.savez(path, **out)
    return rep
