"""The AGS model: Gaussian groups bound to the two dancers' SMPL-X triangles and the follow's groom strands, per-frame
pose corrections and hair node tracks, posing at any time, gsplat rendering (RGB + one-hot role channels in one call),
and MCMC relocation / growth restricted to the anchors."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from . import GROUPS, HEAD_JOINT, N_BODY_JOINTS, R_FOOT, R_SKIN, ROLES, SplatConfig, config_dict
from .binding import (body_ancestor, body_gaussians, chain_levels, dct_basis, dense_weights, pose_delta_transforms,
                      quat_to_matrix, rgb_to_sh0, segment_frames, sh_rgb, skin_delta, triangle_frames)

ROLE_CHANNEL = {"lead": 0, "follow": 1, "hair": 2}
MODEL_VERSION = 1


def _t():
    import torch
    return torch


def logit(p: float) -> float:
    return math.log(p / (1 - p))


def vertex_normals(V, faces):
    torch = _t()
    v = V[..., faces, :]
    n = torch.cross(v[..., 1, :] - v[..., 0, :], v[..., 2, :] - v[..., 0, :], dim=-1)
    out = torch.zeros_like(V)
    for k in range(3):
        out.index_add_(-2, faces[:, k], n)
    return out / out.norm(dim=-1, keepdim=True).clamp_min(1e-12)


class Group:
    """Per-Gaussian parameters (Adam each, own learning rate) + binding buffers of one group."""

    def __init__(self, name: str, params: dict, buffers: dict, lrs: dict, device):
        torch = _t()
        self.name = name
        self.p = {k: torch.nn.Parameter(torch.as_tensor(v, dtype=torch.float32, device=device).contiguous())
                  for k, v in params.items()}
        self.buf = {k: torch.as_tensor(v, device=device) for k, v in buffers.items()}
        self.lrs = dict(lrs)
        self.opt = {k: torch.optim.Adam([self.p[k]], lr=lrs[k], eps=1e-15) for k in self.p}
        self.grad_scale = None  # (N,) colour gradient factor (avatar prior regions), applied in scale_colour_grads
        self.cache: dict = {}    # derived per-Gaussian constants (model._derived); cleared when rows change

    def __len__(self):
        return int(self.p["opac"].shape[0])

    def step(self, keys=None):
        for k in (keys or self.p):
            if self.p[k].grad is not None:
                self.opt[k].step()

    def zero_grad(self):
        for o in self.opt.values():
            o.zero_grad(set_to_none=True)

    def scale_colour_grads(self):
        if self.grad_scale is None:
            return
        for k in ("sh0", "shN"):
            g = self.p[k].grad
            if g is not None:
                g.mul_(self.grad_scale.view(-1, *([1] * (g.dim() - 1))))

    def _reset_state(self, k, idx):
        st = self.opt[k].state.get(self.p[k])
        if st:
            for s in ("exp_avg", "exp_avg_sq"):
                if s in st:
                    st[s][idx] = 0

    def replace(self, dst, src, override: dict | None = None, jitter=None):
        """Rows dst <- rows src (params and buffers), then override; Adam state of dst and src reset."""
        torch = _t()
        self.cache = {}
        with torch.no_grad():
            for k, p in self.p.items():
                p.data[dst] = p.data[src]
            for k, b in self.buf.items():
                b[dst] = b[src]
            if jitter is not None:
                jitter(self, dst)
            for k, v in (override or {}).items():
                tgt = self.p[k].data if k in self.p else self.buf[k]
                tgt[dst] = v[0]
                tgt[src] = v[1]
        for k in self.p:
            self._reset_state(k, dst)
            self._reset_state(k, src)

    def append(self, src, override: dict | None = None, jitter=None):
        """New rows copied from src (then override = {key: (new_values, src_values)}); optimizers extended."""
        torch = _t()
        self.cache = {}
        n0 = len(self)
        with torch.no_grad():
            new = {k: p.data[src].clone() for k, p in self.p.items()}
            for k, b in self.buf.items():
                self.buf[k] = torch.cat([b, b[src]])
            for k, (nv, sv) in (override or {}).items():
                if k in new:
                    new[k] = nv.clone()
                    self.p[k].data[src] = sv
                else:
                    self.buf[k][n0:] = nv
        for k, p in list(self.p.items()):
            old = p
            np_ = torch.nn.Parameter(torch.cat([old.data, new[k]]).contiguous())
            o = self.opt[k]
            st = o.state.pop(old, None)
            o.param_groups[0]["params"] = [np_]
            if st:
                for s in ("exp_avg", "exp_avg_sq"):
                    if s in st:
                        st[s] = torch.cat([st[s], torch.zeros_like(new[k])])
                o.state[np_] = st
            self.p[k] = np_
        if self.grad_scale is not None:
            self.grad_scale = torch.cat([self.grad_scale, self.grad_scale[src]])
        if jitter is not None:
            jitter(self, torch.arange(n0, len(self), device=self.p["opac"].device))
        for k in self.p:
            self._reset_state(k, src)

    def keep(self, mask):
        """Drop rows (export only: optimizers are not kept consistent)."""
        torch = _t()
        self.cache = {}
        for k in list(self.p):
            self.p[k] = torch.nn.Parameter(self.p[k].data[mask].contiguous())
        for k in list(self.buf):
            self.buf[k] = self.buf[k][mask]

    def state(self) -> dict:
        return {"p": {k: v.detach().cpu() for k, v in self.p.items()}, "buf": {k: v.cpu() for k, v in self.buf.items()}}


class AGSModel:
    """Everything that is optimised (Gaussian groups, pose deltas, hair node tracks, colour transforms) + the bundle's
    static geometry on the device."""

    def __init__(self, B: dict, cfg: SplatConfig, device: str = "cuda", init: bool = True):
        torch = _t()
        self.cfg, self.dev = cfg, torch.device(device)
        bd = B["bodies"]
        t = lambda a, dt=torch.float32: torch.as_tensor(np.asarray(a), dtype=dt, device=self.dev)  # noqa: E731
        self.t_grid = np.asarray(bd["t_grid"], float)
        self.k_window = np.asarray(bd["k_window"], int)
        self.window = (float(bd["window"][0]), float(bd["window"][1]))
        self.faces = t(bd["faces"], torch.long)
        self.W = dense_weights(t(bd["w_idx"], torch.long), t(bd["w_val"]), int(np.asarray(bd["parents"]).shape[0]))
        self.parents = np.asarray(bd["parents"], int)
        self.levels = chain_levels(self.parents, N_BODY_JOINTS)
        self.anc = t(body_ancestor(self.parents, N_BODY_JOINTS), torch.long)
        self.verts = {r: t(bd[f"{r}_verts"]) for r in ROLES}
        self.Rg = {r: t(bd[f"{r}_Rg"]) for r in ROLES}
        self.Jp = {r: t(bd[f"{r}_Jp"]) for r in ROLES}
        self.v_rest = {r: t(bd[f"{r}_v_rest"]) for r in ROLES}
        self.k_rest = {r: triangle_frames(self.v_rest[r], self.faces)[1] for r in ROLES}
        self.Tg = len(self.t_grid)
        self._ar = torch.arange(self.Tg, device=self.dev)
        self.hair = B.get("hair")
        self.hair_role = str(self.hair["role"]) if self.hair is not None else None
        if self.hair is not None:
            h = self.hair
            self.h_rest = t(h["rest_local"])                       # (G, S+1, 3) head frame
            self.h_len0 = t(h["seg_len"])                          # (G, S)
            self.h_centre = t(h["head_centre_local"])
            self.h_radius = float(h["head_radius"])
            self.h_colliders = np.asarray(h["colliders"], float).reshape(-1, 4)
            self.G, self.S = int(self.h_rest.shape[0]), int(self.h_rest.shape[1] - 1)
            cl = self.h_colliders
            self.h_col_a, self.h_col_b = t(cl[:, 0], torch.long), t(cl[:, 1], torch.long)
            self.h_col_r = t(cl[:, 2])
            # capsule end offsets in the bones' rest axes (hair.npz collider_oa / _ob; zeros = at the joints)
            z3 = np.zeros((len(cl), 3), np.float32)
            self.h_col_oa = t(np.asarray(h["collider_oa"], np.float32).reshape(-1, 3) if "collider_oa" in h else z3)
            self.h_col_ob = t(np.asarray(h["collider_ob"], np.float32).reshape(-1, 3) if "collider_ob" in h else z3)
            node = np.tile(np.arange(self.S + 1), self.G)                      # strand node index of each node
            self.h_col_mask = t((node[None, :] >= cl[:, 3:4]).astype(np.float32))  # (C, G * (S+1))
        self.sh_K = (cfg.sh_degree + 1) ** 2
        self.groups: dict[str, Group] = {}
        self.pose_r = torch.nn.Parameter(torch.zeros(len(ROLES), self.Tg, N_BODY_JOINTS, 3, device=self.dev))
        self.pose_t = torch.nn.Parameter(torch.zeros(len(ROLES), self.Tg, 3, device=self.dev))
        self.nodes = (torch.nn.Parameter(torch.zeros(self.Tg, self.G, self.S + 1, 3, device=self.dev))
                      if self.hair is not None else None)
        self.train_views = [v for v in B["views"] if v["split"] == "train"]
        self.app_index = {v["i"]: j for j, v in enumerate(self.train_views)}
        from ..room import Appearance
        self.app = Appearance(max(1, len(self.train_views)), self.dev)
        self.opt_pose = torch.optim.Adam([self.pose_r, self.pose_t], lr=cfg.lr_pose)
        self.opt_nodes = torch.optim.Adam([self.nodes], lr=cfg.lr_hair_nodes) if self.nodes is not None else None
        self.opt_app = torch.optim.Adam([self.app.A, self.app.b], lr=cfg.lr_app)
        if init:
            for r, n in (("lead", cfg.n_lead), ("follow", cfg.n_follow)):
                self.groups[r] = self._init_body(r, B["init"][r], n)
            if self.hair is not None:
                self.groups["hair"] = self._init_hair()
        self.cap = {g: int(len(self.groups[g]) * cfg.cap_factor) for g in self.groups}
        self.binoms = None

    # ------------------------------------------------------------------------------------------- init
    def _lrs(self, extra=None):
        c = self.cfg
        d = {"off": c.lr_off, "log_scale": c.lr_scales, "quat": c.lr_quats, "opac": c.lr_opac, "sh0": c.lr_sh0,
             "shN": c.lr_shN}
        d.update(extra or {})
        return d

    def _init_body(self, role: str, s: dict, n: int) -> Group:
        c = self.cfg
        n = min(n, len(s["face"]))
        face, bary, rgb, reg = s["face"][:n], s["bary"][:n], s["rgb"][:n], s["region"][:n].astype(np.int64)
        area = float(self._area(role))
        spacing = math.sqrt(area / n)
        lo, hi = self._normal_range(reg)
        frac = np.clip(-lo / np.maximum(hi - lo, 1e-9), 1e-3, 1 - 1e-3)
        off = np.zeros((n, 3), np.float32)
        off[:, 2] = np.log(frac / (1 - frac))
        ls = np.tile(np.log([0.7 * spacing, 0.7 * spacing, 0.25 * spacing]), (n, 1)).astype(np.float32)
        q = np.tile([1.0, 0, 0, 0], (n, 1)).astype(np.float32)
        params = {"off": off, "log_scale": ls, "quat": q, "opac": np.full(n, logit(0.5), np.float32),
                  "sh0": rgb_to_sh0(rgb.astype(np.float32))[:, None], "shN": np.zeros((n, self.sh_K - 1, 3), np.float32),
                  "resid": np.zeros((n, c.resid_k, 3), np.float32)}
        bufs = {"face": face.astype(np.int64), "bary": bary.astype(np.float32), "region": reg,
                "rgb0": rgb.astype(np.float32)}
        g = Group(role, params, bufs, self._lrs({"resid": c.lr_resid}), self.dev)
        g.spacing = spacing
        return g

    def _area(self, role):
        torch = _t()
        _, k = triangle_frames(self.v_rest[role], self.faces)
        return float((k * k).sum() / 2) if torch.is_tensor(k) else 1.8

    def _normal_range(self, reg):
        c = self.cfg
        reg = np.asarray(reg)
        lo = np.full(len(reg), c.normal_range_cloth[0])
        hi = np.full(len(reg), c.normal_range_cloth[1])
        sk = (reg & R_SKIN) > 0
        ft = (reg & R_FOOT) > 0
        lo[sk], hi[sk] = c.normal_range_skin
        lo[ft], hi[ft] = c.normal_range_foot
        return lo.astype(np.float32), hi.astype(np.float32)

    def _init_hair(self) -> Group:
        c, h = self.cfg, self.hair
        G, S = self.G, self.S
        nps = max(1, c.n_hair_per_seg)
        ch = int(h["children"])
        spread = float(h["child_spread"])
        rng = np.random.default_rng(c.seed + 7)
        seg_len = np.asarray(h["seg_len"], float)
        pal = np.asarray(h["palette_srgb"], float)
        rows = []
        for g in range(G):
            for s in range(S):
                for a in range(nps):
                    for k in range(ch + 1):
                        if k == 0:
                            o = (0.0, 0.0)
                        else:
                            ang = 2 * math.pi * (k - 1) / ch + rng.uniform(-0.4, 0.4)
                            rr = spread * rng.uniform(0.5, 1.0)
                            o = (rr * math.cos(ang), rr * math.sin(ang))
                        rows.append((g, s, (a + 0.5) / nps, o[0], o[1]))
        rows = np.array(rows, float)
        gi, si, u = rows[:, 0].astype(np.int64), rows[:, 1].astype(np.int64), rows[:, 2]
        n = len(rows)
        L = seg_len[gi, si]
        pos = (si + u) / S
        width = float(h["width_root"]) + (float(h["width_tip"]) - float(h["width_root"])) * pos
        off = np.zeros((n, 3), np.float32)
        off[:, 1:] = np.arctanh(np.clip(rows[:, 3:5] / 0.03, -0.95, 0.95))
        ls = np.stack([np.log(0.6 * L / nps + 1e-4), np.log(width / 8), np.log(width / 8)], 1).astype(np.float32)
        x = pos * (len(pal) - 1)
        i0 = np.clip(np.floor(x).astype(int), 0, len(pal) - 2)
        f = (x - i0)[:, None]
        rgb = ((1 - f) * pal[i0] + f * pal[i0 + 1]).astype(np.float32)
        params = {"off": off, "log_scale": ls, "quat": np.tile([1.0, 0, 0, 0], (n, 1)).astype(np.float32),
                  "opac": np.full(n, logit(0.35), np.float32), "sh0": rgb_to_sh0(rgb)[:, None],
                  "shN": np.zeros((n, self.sh_K - 1, 3), np.float32)}
        bufs = {"guide": gi, "seg": si, "u": u.astype(np.float32), "rgb0": rgb}
        return Group("hair", params, bufs, self._lrs(), self.dev)

    # ------------------------------------------------------------------------------------------- posing
    def frame_index(self, t: float) -> tuple[int, float]:
        k = int(np.clip(np.searchsorted(self.t_grid, t) - 1, 0, self.Tg - 2))
        a = float(np.clip((t - self.t_grid[k]) / max(self.t_grid[k + 1] - self.t_grid[k], 1e-9), 0.0, 1.0))
        return k, a

    def kidx(self, ks):
        """Grid frames -> a device index tensor without a host->device copy (that would synchronise the stream)."""
        torch = _t()
        ks = [int(k) for k in ks]
        if ks == list(range(ks[0], ks[0] + len(ks))):
            return self._ar[ks[0]:ks[0] + len(ks)]
        return torch.cat([self._ar[k:k + 1] for k in ks])

    def _derived(self, gname: str, g: Group) -> dict:
        """Per-Gaussian constants of a body group (normal-offset range, residual mask / cap, rest scale, triangle)."""
        torch = _t()
        d = g.cache
        if d:
            return d
        c = self.cfg
        reg = g.buf["region"]
        d["lo"], d["hi"] = self._normal_range_t(reg)
        skin = (reg & R_SKIN) > 0
        d["kmask"] = torch.where(skin[:, None] & (torch.arange(c.resid_k, device=self.dev) >= c.resid_k_skin)[None],
                                 0.0, 1.0)
        d["cap"] = torch.where(skin, c.resid_cap_skin, c.resid_cap_cloth)[:, None].float()
        d["kr"] = self.k_rest[gname][g.buf["face"]]
        d["tri"] = self.faces[g.buf["face"]]
        return d

    def pose_deltas(self, ri: int, ks):
        """Capped (r rad, dT m) of dancer ri at grid frames ks."""
        torch = _t()
        cr, ct = math.radians(self.cfg.pose_cap_deg), self.cfg.pose_cap_m
        return cr * torch.tanh(self.pose_r[ri, ks] / cr), ct * torch.tanh(self.pose_t[ri, ks] / ct)

    def posed(self, ks: list[int], use_delta: bool = True) -> dict:
        """Corrected vertices (len(ks), V, 3) per dancer, the joint transforms D55 and the corrected joints."""
        kt = self.kidx(ks)
        out = {}
        for ri, r in enumerate(ROLES):
            V = self.verts[r][kt]
            if not use_delta:
                out[r] = {"V": V, "D": None, "J": self.Jp[r][kt]}
                continue
            dr, dT = self.pose_deltas(ri, kt)
            D = pose_delta_transforms(self.Rg[r][kt][:, :N_BODY_JOINTS], self.Jp[r][kt][:, :N_BODY_JOINTS], dr, dT,
                                      self.parents, self.levels)[:, self.anc]
            Vc = skin_delta(V, D, self.W)
            J = (D[..., :3, :3] @ self.Jp[r][kt][..., None])[..., 0] + D[..., :3, 3]
            out[r] = {"V": Vc, "D": D, "J": J}
        return out

    def node_offsets(self, ks):
        torch = _t()
        cap = self.cfg.hair_off_cap
        off = cap * torch.tanh(self.nodes[ks] / cap)
        return torch.cat([torch.zeros_like(off[:, :, :1]), off[:, :, 1:]], 2)  # the root never moves

    def hair_nodes(self, ks, D=None, frozen: bool = False):
        """World strand nodes (len(ks), G, S+1, 3), head rotation (len(ks), 3, 3) and local nodes of grid frames ks.
        D: the hair dancer's joint corrections (len(ks), 55, 4, 4) of those frames (None = the fit as is)."""
        r = self.hair_role
        kt = self.kidx(ks)
        loc = self.h_rest[None].expand(len(ks), -1, -1, -1) if frozen else self.h_rest[None] + self.node_offsets(kt)
        Rh = self.Rg[r][kt][:, HEAD_JOINT]
        X = (Rh[:, None, None] @ loc[..., None])[..., 0] + self.Jp[r][kt][:, HEAD_JOINT][:, None, None]
        if D is not None:
            Dh = D[:, HEAD_JOINT]
            X = (Dh[:, None, None, :3, :3] @ X[..., None])[..., 0] + Dh[:, None, None, :3, 3]
            Rh = Dh[:, :3, :3] @ Rh
        return X, Rh, loc

    def collider_ends(self, ks, P_role: dict):
        """World capsule ends (len(ks), C, 3) of the groom's colliders at grid frames ks: joint + its corrected global
        rotation x the rest-axes offset (a point attached to bone j moves as Rg_j (x - J_rest_j) + Jp_j, then the
        pose correction D_j). P_role: posed() of these frames for the hair dancer ('D' None = the fit as is)."""
        kt = self.kidx(ks)
        Rg = self.Rg[self.hair_role][kt]                                                   # (nK, 55, 3, 3)
        D, J = P_role.get("D"), P_role["J"]
        ends = []
        for idx, off in ((self.h_col_a, self.h_col_oa), (self.h_col_b, self.h_col_ob)):
            R = Rg[:, idx]
            if D is not None:
                R = D[:, idx, :3, :3] @ R
            ends.append(J[:, idx] + (R @ off[None, :, :, None])[..., 0])
        return ends[0], ends[1]

    def window_x(self, t: float) -> float:
        w0, w1 = self.window
        return float(np.clip((t - w0) / max(w1 - w0, 1e-6), 0.0, 1.0))

    BATCHED = ("means", "R", "scales", "anchor")  # per time; opac / sh / role are shared

    def h2d(self, vals, dtype=None):
        """Python numbers -> device tensor through pinned memory (no stream synchronisation)."""
        torch = _t()
        t = torch.tensor(vals, dtype=dtype or torch.float32)
        if self.dev.type == "cuda":
            return t.pin_memory().to(self.dev, non_blocking=True)
        return t.to(self.dev)

    def gaussians(self, t: float, P: dict | None = None, frames: dict | None = None, groups=GROUPS,
                  use_resid: bool = True, hair_frozen: bool = False, use_delta: bool = True) -> dict:
        """All Gaussians at one time t (gaussians_b without the batch dimension)."""
        Gs = self.gaussians_b([t], P, frames, groups, use_resid, hair_frozen, use_delta)
        return {k: (v[0] if k in self.BATCHED else v) for k, v in Gs.items()}

    def gaussians_b(self, ts: list[float], P: dict | None = None, frames: dict | None = None, groups=GROUPS,
                    use_resid: bool = True, hair_frozen: bool = False, use_delta: bool = True) -> dict:
        """All Gaussians at the times ts in one pass (leading batch dim B = len(ts) on means / R / scales / anchor):
        each time lerps between its bracketing grid frames. P: posed() of grid frames `frames` ({grid k: row in P})
        that contain every k, k + 1 (shared by several views); None = pose them here."""
        torch = _t()
        ka = [self.frame_index(t) for t in ts]
        if P is None:
            frames = {}
            for k, _ in ka:
                frames.setdefault(k, len(frames))
                frames.setdefault(k + 1, len(frames))
            P = self.posed(list(frames), use_delta)
        B = len(ts)
        r0 = self.h2d([frames[k] for k, _ in ka], torch.long)
        r1 = self.h2d([frames[k + 1] for k, _ in ka], torch.long)
        aw = self.h2d([a for _, a in ka])
        xw = self.h2d([self.window_x(t) for t in ts])
        out = {"means": [], "R": [], "scales": [], "opac": [], "sh": [], "role": [], "anchor": [], "group": []}
        c = self.cfg
        for gname in groups:
            if gname not in self.groups:
                continue
            g = self.groups[gname]
            n = len(g)
            if gname in ROLES:
                V0 = P[gname]["V"].index_select(0, r0)
                V = V0 + aw[:, None, None] * (P[gname]["V"].index_select(0, r1) - V0)          # (B, Nv, 3)
                fr = triangle_frames(V, self.faces)
                dv = self._derived(gname, g)
                kr = dv["kr"]
                o0, o1, o2 = g.p["off"].unbind(-1)
                off = torch.stack([0.5 * kr * torch.tanh(o0), 0.5 * kr * torch.tanh(o1),
                                   dv["lo"] + (dv["hi"] - dv["lo"]) * torch.sigmoid(o2)], 1)
                resid = None
                if use_resid:
                    phi = dct_basis(xw, c.resid_k)                                              # (B, K)
                    raw = torch.einsum("nkc,nk,bk->bnc", g.p["resid"], dv["kmask"], phi)
                    resid = dv["cap"] * torch.tanh(raw / dv["cap"])
                anchor = (V[:, dv["tri"]] * g.buf["bary"][..., None]).sum(-2)
                mu, Rw, sc = body_gaussians(V, self.faces, g.buf["face"], g.buf["bary"], off, self.k_rest[gname],
                                            g.p["log_scale"], g.p["quat"], resid, c.scale_cap, fr, anchor)
                role = ROLE_CHANNEL[gname]
            else:
                hr = self.hair_role
                X, Rh, _ = self.hair_nodes(list(frames), P[hr]["D"], hair_frozen)              # (nK, G, S+1, 3)
                X0 = X.index_select(0, r0)
                Xt = X0 + aw[:, None, None, None] * (X.index_select(0, r1) - X0)              # (B, G, S+1, 3)
                f0 = Rh.index_select(0, r0)[..., 2]
                ref = f0 + aw[:, None] * (Rh.index_select(0, r1)[..., 2] - f0)
                Rs, L = segment_frames(Xt, ref[:, None, :])
                gi, si, u = g.buf["guide"], g.buf["seg"], g.buf["u"]
                Rg_ = Rs[:, gi, si]
                h0, h1 = Xt[:, gi, si], Xt[:, gi, si + 1]
                nps = max(1, c.n_hair_per_seg)
                o0, o1, o2 = g.p["off"].unbind(-1)
                loc = torch.stack([0.5 * L[:, gi, si] / nps * torch.tanh(o0), (0.03 * torch.tanh(o1)).expand(B, n),
                                   (0.03 * torch.tanh(o2)).expand(B, n)], -1)
                anchor = h0 + u[:, None] * (h1 - h0)
                mu = anchor + (Rg_ @ loc[..., None])[..., 0]
                Rw = Rg_ @ quat_to_matrix(g.p["quat"])
                sc = torch.exp(g.p["log_scale"].clamp(max=math.log(0.05))).expand(B, n, 3)
                role = ROLE_CHANNEL["hair"]
            out["means"].append(mu)
            out["R"].append(Rw)
            out["scales"].append(sc)
            out["opac"].append(torch.sigmoid(g.p["opac"]))
            out["sh"].append(torch.cat([g.p["sh0"], g.p["shN"]], 1))
            oh = torch.zeros(n, 3, device=self.dev)
            oh[:, role] = 1.0
            out["role"].append(oh)
            out["anchor"].append(anchor)
            out["group"].append((gname, n))
        res = {k2: torch.cat(v, 1 if k2 in self.BATCHED else 0) for k2, v in out.items() if k2 != "group"}
        res["group"] = out["group"]
        res["frames"], res["ka"] = frames, ka
        return res

    def _normal_range_t(self, reg):
        torch = _t()
        c = self.cfg
        lo = torch.full(reg.shape, c.normal_range_cloth[0], device=self.dev)
        hi = torch.full(reg.shape, c.normal_range_cloth[1], device=self.dev)
        sk, ft = (reg & R_SKIN) > 0, (reg & R_FOOT) > 0
        lo = torch.where(sk, c.normal_range_skin[0], torch.where(ft, c.normal_range_foot[0], lo))
        hi = torch.where(sk, c.normal_range_skin[1], torch.where(ft, c.normal_range_foot[1], hi))
        return lo, hi

    # ------------------------------------------------------------------------------------------- render
    def render_b(self, Gs: dict, views: list[dict], sh_degree: int, W: int | None = None, H: int | None = None,
                 rasterize_mode: str = "antialiased"):
        """One gsplat call for B views (Gs from gaussians_b, one time per view), all rendered on a W x H canvas (the
        largest crop; each crop's K keeps its own origin, so view b is the top-left H_b x W_b of its image):
        premultiplied RGB (B, H, W, 3), alpha (B, H, W), role alphas (B, H, W, 3) [lead, follow body, hair]."""
        torch = _t()
        from gsplat import rasterization
        for v in views:
            if "_vm" not in v:
                v["_vm"] = torch.as_tensor(viewmat(v), dtype=torch.float32, device=self.dev)
                v["_K"] = torch.as_tensor(K_gs(v), dtype=torch.float32, device=self.dev)
                v["_C"] = torch.as_tensor(cam_centre(v), dtype=torch.float32, device=self.dev)
        W = int(W or max(v["W"] for v in views))
        H = int(H or max(v["H"] for v in views))
        B = len(views)
        vm = torch.stack([v["_vm"] for v in views])[:, None]
        K = torch.stack([v["_K"] for v in views])[:, None]
        C = torch.stack([v["_C"] for v in views])
        means = Gs["means"]
        N = means.shape[1]
        rgb = sh_rgb(means - C[:, None, :], Gs["sh"], sh_degree)                              # (B, N, 3)
        colors = torch.cat([rgb, Gs["role"].expand(B, N, 3)], -1)
        M = Gs["R"] * Gs["scales"][..., None, :]
        cov = M @ M.transpose(-1, -2)  # covariances directly: no rotation -> quaternion conversion
        out, alpha, info = rasterization(means, None, None, Gs["opac"].expand(B, N), colors, vm, K, W, H,
                                         sh_degree=None, render_mode="RGB", rasterize_mode=rasterize_mode,
                                         packed=False, near_plane=0.05, far_plane=100.0, covars=cov)
        return {"rgb": out[:, 0, ..., :3], "roles": out[:, 0, ..., 3:6], "alpha": alpha[:, 0, ..., 0], "info": info}

    def render(self, Gs: dict, view: dict, sh_degree: int, rasterize_mode: str = "antialiased"):
        """One view (Gs from gaussians(): no batch dim) -> rgb (H, W, 3) premultiplied, alpha (H, W), roles."""
        Gb = {k: (v[None] if k in self.BATCHED else v) for k, v in Gs.items()}
        r = self.render_b(Gb, [view], sh_degree, rasterize_mode=rasterize_mode)
        return {"rgb": r["rgb"][0], "roles": r["roles"][0], "alpha": r["alpha"][0], "info": r["info"]}

    # ------------------------------------------------------------------------------------------- MCMC
    def relocate_and_grow(self, step: int, grow: bool = True) -> dict:
        """gsplat MCMC on the anchors: dead Gaussians (opacity < min) take the binding + attributes of live ones sampled
        by opacity (jittered on their triangle / segment), opacities and scales split by compute_relocation; then
        growth by 5 % up to the cap. Returns counts."""
        torch = _t()
        rep = {}
        for gname, g in self.groups.items():
            with torch.no_grad():
                op = torch.sigmoid(g.p["opac"])
                dead = (op <= self.cfg.min_opacity) | ~torch.isfinite(op)
                nd = int(dead.sum())
                alive = (~dead).nonzero().squeeze(1)
                if nd and len(alive):
                    probs = op[alive]
                    src = alive[torch.multinomial(probs / probs.sum(), nd, replacement=True)]
                    dst = dead.nonzero().squeeze(1)
                    self._split(g, src, dst, append=False)
                ng = 0
                if grow and len(g) < self.cap[gname]:
                    ng = min(self.cap[gname] - len(g), max(0, int(0.05 * len(g))))
                    if ng:
                        op = torch.sigmoid(g.p["opac"])
                        src = torch.multinomial(op / op.sum(), ng, replacement=True)
                        self._split(g, src, None, append=True)
            rep[gname] = {"relocated": nd, "grown": ng, "n": len(g)}
        return rep

    def _split(self, g: Group, src, dst, append: bool):
        torch = _t()
        cnt = torch.bincount(src, minlength=len(g))[src] + 1
        op = torch.sigmoid(g.p["opac"].data[src])
        sc = torch.exp(g.p["log_scale"].data[src])
        new_op, new_sc = self._relocation(op, sc, cnt)
        new_op = new_op.clamp(self.cfg.min_opacity + 1e-4, 1 - 1e-4)
        lo, ls = torch.logit(new_op), torch.log(new_sc.clamp_min(1e-7))
        over = {"opac": (lo, lo), "log_scale": (ls, ls)}
        jit = self._jitter_body if "face" in g.buf else self._jitter_hair
        if append:
            g.append(src, over, jit)
        else:
            g.replace(dst, src, over, jit)

    def _relocation(self, op, sc, cnt):
        torch = _t()
        if op.is_cuda:
            try:
                from gsplat.relocation import compute_relocation
                if self.binoms is None:
                    n_max = 51
                    b = torch.zeros(n_max, n_max, device=op.device)
                    for n in range(n_max):
                        for k in range(n + 1):
                            b[n, k] = math.comb(n, k)
                    self.binoms = b
                return compute_relocation(op, sc, cnt.clamp(max=50).int(), self.binoms)
            except Exception:  # noqa: BLE001 - fall back to the simple split below
                pass
        new_op = 1 - (1 - op) ** (1.0 / cnt)
        return new_op, sc

    @staticmethod
    def _jitter_body(g: Group, idx):
        torch = _t()
        b = g.buf["bary"][idx] + 0.05 * torch.randn_like(g.buf["bary"][idx])
        b = b.clamp_min(0)
        g.buf["bary"][idx] = b / b.sum(1, keepdim=True).clamp_min(1e-6)

    def _jitter_hair(self, g: Group, idx):
        torch = _t()
        nps = max(1, self.cfg.n_hair_per_seg)
        g.buf["u"][idx] = (g.buf["u"][idx] + (torch.rand_like(g.buf["u"][idx]) - 0.5) / nps).clamp(0.0, 1.0)

    def inject_noise(self, scale: float):
        """MCMC exploration: nearly transparent Gaussians wander on their anchors (raw tangent offsets)."""
        torch = _t()
        with torch.no_grad():
            for g in self.groups.values():
                op = torch.sigmoid(g.p["opac"])
                w = torch.sigmoid(100 * (0.005 - op))[:, None]
                g.p["off"].data[:, :2] += scale * w * torch.randn_like(g.p["off"].data[:, :2])

    # ------------------------------------------------------------------------------------------- io
    def n_gaussians(self) -> int:
        return sum(len(g) for g in self.groups.values())

    def state(self) -> dict:
        return {"version": MODEL_VERSION, "config": config_dict(self.cfg),
                "groups": {k: g.state() for k, g in self.groups.items()},
                "pose_r": self.pose_r.detach().cpu(), "pose_t": self.pose_t.detach().cpu(),
                "nodes": self.nodes.detach().cpu() if self.nodes is not None else None,
                "app_A": self.app.A.detach().cpu(), "app_b": self.app.b.detach().cpu(),
                "train_views": [v["i"] for v in self.train_views], "t_grid": self.t_grid.tolist(),
                "window": list(self.window), "hair_role": self.hair_role}

    def load_state(self, st: dict):
        torch = _t()
        if [v["i"] for v in self.train_views] != list(st["train_views"]):
            raise ValueError("dancer_splats: the model was trained on a different bundle (views differ)")
        if len(st["t_grid"]) != self.Tg or not np.allclose(st["t_grid"], self.t_grid, atol=1e-5):
            raise ValueError("dancer_splats: the model's time grid differs from the bundle's")
        for k, gs in st["groups"].items():
            g = Group(k, {kk: v.numpy() for kk, v in gs["p"].items()}, {kk: v.numpy() for kk, v in gs["buf"].items()},
                      self._lrs({"resid": self.cfg.lr_resid}) if k in ROLES else self._lrs(), self.dev)
            self.groups[k] = g
        with torch.no_grad():
            self.pose_r.data[:] = st["pose_r"].to(self.dev)
            self.pose_t.data[:] = st["pose_t"].to(self.dev)
            if self.nodes is not None and st.get("nodes") is not None:
                self.nodes.data[:] = st["nodes"].to(self.dev)
            self.app.A.data[:] = st["app_A"].to(self.dev)
            self.app.b.data[:] = st["app_b"].to(self.dev)
        self.cap = {g: max(self.cap.get(g, 0), len(self.groups[g])) for g in self.groups}

    def save(self, path: Path, extra: dict | None = None):
        torch = _t()
        path.parent.mkdir(parents=True, exist_ok=True)
        st = self.state()
        st["extra"] = extra or {}
        tmp = path.with_name(path.name + ".part")
        torch.save(st, tmp)
        import os
        os.replace(tmp, path)

    @classmethod
    def load(cls, B: dict, path: Path, device: str = "cuda") -> "AGSModel":
        torch = _t()
        st = torch.load(path, map_location="cpu", weights_only=False)
        from . import SplatConfig
        cd = {k: (tuple(v) if isinstance(v, list) else v) for k, v in st["config"].items()}
        cfg = SplatConfig(**{k: v for k, v in cd.items() if k in SplatConfig.__dataclass_fields__})
        m = cls(B, cfg, device, init=False)
        m.load_state(st)
        m.extra = st.get("extra", {})
        return m

    def appearance_for(self, view: dict, by_cam: dict | None = None):
        """(A, b) for a view: its own transform when it trained, else its camera's mean, else the identity."""
        torch = _t()
        j = self.app_index.get(view["i"])
        if j is not None and view.get("split") == "train":
            return self.app.A[j].detach(), self.app.b[j].detach()
        idx = (by_cam or {}).get(view["cam"])
        if idx:
            return self.app.mean_for(idx)
        return torch.eye(3, device=self.dev), torch.zeros(3, device=self.dev)

    def cam_app_index(self) -> dict:
        out: dict[str, list[int]] = {}
        for j, v in enumerate(self.train_views):
            out.setdefault(v["cam"], []).append(j)
        return out


# ------------------------------------------------------------------------------------------------ cameras

def viewmat(v: dict) -> np.ndarray:
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = np.asarray(v["R"], float), np.asarray(v["t_w"], float)
    return M


def K_gs(v: dict) -> np.ndarray:
    """gsplat convention: pixel centres at +0.5 (the bundle's K has them at integers)."""
    K = np.asarray(v["K"], float).copy()
    K[0, 2] += 0.5
    K[1, 2] += 0.5
    return K


def cam_centre(v: dict) -> np.ndarray:
    return -np.asarray(v["R"], float).T @ np.asarray(v["t_w"], float)


def composite(model: AGSModel, r: dict, A, b, bg):
    """Colour transform on the dancers (premultiplied: the bias scales with alpha) + background."""
    return r["rgb"] @ A.T + r["alpha"][..., None] * b + (1 - r["alpha"][..., None]) * bg


def interpenetration(model: AGSModel, Gs: dict, P: dict, k_rows: tuple[int, int], a: float, n_samp: int = 2048,
                     margin: float = 0.01, exempt: float = 0.02):
    """Hinge on body Gaussians of one dancer more than `margin` inside the other's posed body (nearest-vertex normal
    test, sampled), only where their anchor is > `exempt` outside it (contact / the fit's own intersections exempt)."""
    torch = _t()
    tot = torch.zeros((), device=model.dev)
    start = 0
    spans = {}
    for name, n in Gs["group"]:
        spans[name] = (start, start + n)
        start += n
    for me, other in (("lead", "follow"), ("follow", "lead")):
        if me not in spans:
            continue
        s0, s1 = spans[me]
        V = (1 - a) * P[other]["V"][k_rows[0]] + a * P[other]["V"][k_rows[1]]
        Vs = V[::3]
        Ns = vertex_normals(V, model.faces)[::3]
        lo, hi = Vs.min(0).values - 0.05, Vs.max(0).values + 0.05
        idx = torch.randint(s0, s1, (min(n_samp, s1 - s0),), device=model.dev)
        X, An = Gs["means"][idx], Gs["anchor"][idx].detach()
        near = ((X > lo) & (X < hi)).all(1).float()  # masks, no boolean indexing (that would synchronise)
        nn = torch.cdist(An, Vs).argmin(1)
        sd_a = ((An - Vs[nn]) * Ns[nn]).sum(1)
        sd_x = ((X - Vs[nn]) * Ns[nn]).sum(1)
        w = (sd_a > exempt).float() * near
        tot = tot + (torch.relu(-sd_x - margin) * w).sum() / max(1, len(idx))
    return tot
