"""Phase B of the "sides" step (DESIGN 6.1): the CPU rungs before any solve.

1. **Revert**: the base of both dancers is the detected input fit (`cfg.base_fit`); nothing of an earlier repair is read.
2. **Spin splices** (DESIGN 2.4): for every yaw-defect run of decide's yaw track (`yaw.npz` `<role>__defect`), score each overlapping chunk
   solution (stitch.json frame_offset) against the base over the run - the per-view face bits (head heading), the triangulated hip line
   (calm-frame bias removed) and MAMMA's 512-point 2D error - and splice the winner for BOTH dancers when it beats the base by
   `splice_switch_nats`: entry = a >= 7-frame offset ramp 0.5 s before the run (never before the chunk), exit = the first all-segment
   agreement after the run (F1, `splice.choose_exit_all_segments`) + a >= 6-frame slerp.
3. **Unwraps** (R16 closed form, then the L0 spine split; `unwrap.unwrap_fit`) on the spliced fit.

-> work/<take>/sides/cpu/{lead,follow}.npz (the full contract fit; everything not changed above is byte-identical to the input),
   cpu/pod/{lead,follow}.npz (the slim copy the solve pods read: poses, translation, betas, t, stance), cpu/splice_<k0>.npz (each splice
   window, `splice.splice_window_arrays` + metadata), cpu/prepare.json (inputs + sha256, defects, scores, exits, checks, unwraps).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import body as bd
from . import geom, hmm
from . import splice as sp
from . import unwrap as uw
from .adapters import PIDX, adapter_for, chunk_work
from .config import SidesCfg, load_cfg
from .evidence import load_tri, load_views, rest_joints, sha256, write_json

SLIM_KEYS = (*bd.POSE_KEYS, "transl", "betas", "t", "stance", "segment_owner", "gender", "model", "flat_hand_mean", "num_betas")


def log(*a) -> None:
    print("[sides.prepare]", *a, flush=True)


def cpu_dir(work: Path) -> Path:
    return Path(work) / "sides" / "cpu"


def load_fit(p: Path) -> dict:
    with np.load(p, allow_pickle=True) as z:
        return {k: np.array(z[k], copy=True) for k in z.files}


def save_fit(p: Path, fit: dict, compress: bool = True) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    (np.savez_compressed if compress else np.savez)(p, **fit)
    return p


# ------------------------------------------------------------------------------------------------ evidence of a heading track

def head_heading(fit: dict, ks: np.ndarray) -> np.ndarray:
    o = bd.forward(fit, ks, points=False)
    return geom.heading(o["R"][:, geom.HEAD] @ np.array([0.0, 0.0, 1.0]))


def face_labels(views: dict, cams: list, role: str, ks: np.ndarray) -> np.ndarray:
    """(n, C) +1 face seen / -1 hidden / 0 no bit, only where the base predicts the head visible from the view (decide.yaw_track's rule)"""
    p = PIDX[role]
    out = np.zeros((len(ks), len(cams)), int)
    for i, c in enumerate(cams):
        z = views[c]
        ok = z["ok"][ks].astype(bool) & z["vis_pred"][ks, p, 0].astype(bool)
        out[:, i] = np.where(ok, z["face"][ks, p], 0)
    return out


def azimuths(obs: dict, cams: list, J: np.ndarray, ks: np.ndarray) -> np.ndarray:
    """(n, C) heading from the head joint toward each camera centre"""
    return np.stack([geom.heading(obs[c]["C"][ks] - J[:, geom.HEAD]) for c in cams], 1)


def mamma2d(ad, fit: dict, role: str, ks: np.ndarray, fit_rows: np.ndarray | None = None, lm: dict | None = None) -> np.ndarray:
    """(n, C) median px error of MAMMA's 512 points of `fit` (its rows fit_rows, default ks) against the grid observations at ks;
    lm {cam: (n, 2, 512, 2)} the landmarks of frames ks already sliced (read from the obs when None)"""
    rows = ks if fit_rows is None else fit_rows
    pts = bd.forward(fit, rows)["pts"]
    obs = ad.obs()
    p = PIDX[role]
    e = np.full((len(ks), len(ad.cams)), np.nan)
    for i, c in enumerate(ad.cams):
        o = obs[c]
        L = (lm[c] if lm is not None else ad.landmarks(c)[ks])[:, p].astype(float)
        vis = o["vis"][ks, p].astype(float) > 0.5
        Xc = np.einsum("kij,kpj->kpi", o["R"][ks], pts) + o["t"][ks][:, None]
        uv = np.einsum("kij,kpj->kpi", o["K"][ks], Xc)
        uv = uv[..., :2] / np.where(np.abs(uv[..., 2:]) > 1e-9, uv[..., 2:], 1e-9)
        good = vis & (np.abs(L).sum(-1) > 0) & (Xc[..., 2] > 0) & o["ok"][ks][:, None]
        d = np.linalg.norm(uv - L, axis=-1)
        for j in range(len(ks)):
            if good[j].sum() >= 20:
                e[j, i] = float(np.median(d[j][good[j]]))
    return e


# ------------------------------------------------------------------------------------------------ 2. spin splices

def spin_splices(ad, base: dict, cfg: SidesCfg, sd: Path) -> tuple[dict, list, dict]:
    """splice every yaw-defect run an overlapping chunk wins -> (new base {role: fit}, [splice windows], report)"""
    scfg = cfg.splice_cfg()
    hz = cfg.hz
    T = ad.T
    yaw = dict(np.load(sd / "yaw.npz"))
    spins = json.loads((sd / "spins.json").read_text(encoding="utf-8"))
    views = load_views(ad.work, ad.cams)
    obs = ad.obs()
    tri = {r: load_tri(ad.work, r) for r in cfg.roles}
    owner = ad.owner
    rep = {"defects": [], "scored": [], "splices": [], "skipped": []}
    runs = []
    for r in cfg.roles:
        for a, b in hmm.runs(np.asarray(yaw[f"{r}__defect"], bool), merge=scfg.defect_merge_gap, min_len=1):
            runs.append((a, b, r))
            rep["defects"].append({"role": r, "k": [int(a), int(b)], "t": [round(a / hz, 3), round(b / hz, 3)]})
    runs.sort()
    alts = ad.alternatives()
    out = {r: base[r] for r in cfg.roles}
    windows = []
    taken = np.zeros(T, bool)
    for a, b, role in runs:
        if taken[max(0, a - 15):b + 16].any():
            rep["skipped"].append({"role": role, "t": [round(a / hz, 3), round(b / hz, 3)], "why": "inside an earlier splice"})
            continue
        own = set(owner[a:b + 1].tolist())
        need_after = int(round(scfg.exit_hold_s * hz)) + scfg.exit_slerp_frames + 2
        cands = [al for al in alts if al["chunk"] not in own and al["k0"] <= a - 1 and al["k1"] >= b + need_after]
        if not cands:
            rep["skipped"].append({"role": role, "t": [round(a / hz, 3), round(b / hz, 3)], "why": "no overlapping chunk covers it",
                                   "owner": sorted(own)})
            continue
        ks = np.arange(max(a - 3, 0), min(b + 4, T))
        p = PIDX[role]
        lab = face_labels(views, ad.cams, role, ks)
        q = np.array([spins["roles"][role]["q"].get(c, {}).get("face", cfg.splice_face_q) for c in ad.cams])
        X, nv = tri[role]["Xd"].astype(float), tri[role]["nv"]
        okh = (nv[ks, 11] >= cfg.yaw_line_min_views) & (nv[ks, 12] >= cfg.yaw_line_min_views)
        psi_t = np.where(okh, geom.line_heading(X[ks, 11], X[ks, 12]) - spins["roles"][role]["bias_hip_deg"], np.nan)
        kappa = 1.0 / np.radians(cfg.splice_hip_sigma_deg) ** 2
        lm = {c: ad.landmarks(c)[ks] for c in ad.cams}
        scored = {}
        alt_fits = {}
        for al in [{"chunk": "base", "frame_offset": 0}] + cands:
            if al["chunk"] == "base":
                fit, rows = base[role], ks
            else:
                d = Path(al["dir"])
                alt_fits[al["chunk"]] = {r: load_fit(d / f"{r}.npz") for r in cfg.roles}
                fit, rows = alt_fits[al["chunk"]][role], ks - al["frame_offset"]
                if np.abs(np.asarray(fit["betas"], float) - np.asarray(base[role]["betas"], float)).max() > 2e-5:
                    rep["skipped"].append({"role": role, "t": [round(a / hz, 3), round(b / hz, 3)], "why": f"{al['chunk']}: other betas"})
                    continue
            J = bd.forward(fit, rows, points=False)["J"]
            hh = head_heading(fit, rows)
            az = azimuths(obs, ad.cams, J, ks)
            f_ll = sp.face_bits_ll(lab, az, q)(hh)
            pel = geom.line_heading(J[:, 1], J[:, 2])
            v_ll = sp.von_mises_ll(psi_t, kappa)(pel)
            e2 = mamma2d(ad, fit, role, ks, rows, lm)
            c2 = np.nanmean(e2, 1)
            tot = float(f_ll.sum() + v_ll.sum() - cfg.splice_w2d_nats_per_px * np.nansum(c2))
            scored[al["chunk"]] = {"total": round(tot, 2), "face_ll": round(float(f_ll.sum()), 2), "hip_ll": round(float(v_ll.sum()), 2),
                                   "mamma2d_px_mean": round(float(np.nanmean(c2)), 2) if np.isfinite(c2).any() else None,
                                   "frame_offset": int(al["frame_offset"])}
        best = max((k for k in scored if k != "base"), key=lambda k: scored[k]["total"], default=None)
        margin = None if best is None else scored[best]["total"] - scored["base"]["total"]
        rep["scored"].append({"role": role, "t": [round(a / hz, 3), round(b / hz, 3)], "scores": scored, "best": best,
                              "margin_nats": None if margin is None else round(margin, 2)})
        if best is None or margin < cfg.splice_switch_nats:
            continue
        al = next(x for x in cands if x["chunk"] == best)
        off = int(al["frame_offset"])
        alt = alt_fits[best]
        n_alt = len(alt["lead"]["t"])
        altJ = {}
        for r in cfg.roles:
            Ja = np.full((T, 55, 3), np.nan)
            Ja[off:off + n_alt] = np.asarray(alt[r]["joints"], float)[:, :55]
            altJ[r] = Ja
        ex = sp.choose_exit_all_segments({r: np.asarray(out[r]["joints"], float) for r in cfg.roles}, altJ, b + 1, scfg)
        if ex["k_exit"] is None:
            rep["skipped"].append({"role": role, "t": [round(a / hz, 3), round(b / hz, 3)], "why": f"{best}: no all-segment exit"})
            continue
        k_entry = max(off, a - int(round(0.5 * hz)))
        spl, info = sp.splice({r: out[r] for r in cfg.roles}, alt, off, k_entry, ex["k_exit"], scfg, alt_name=best)
        for r in cfg.roles:
            out[r] = spl[r]
        taken[info["k0"]:info["k1"]] = True
        z = sp.splice_window_arrays(spl, info)
        p_npz = cpu_dir(ad.work) / f"splice_{info['k0']:05d}.npz"
        np.savez_compressed(p_npz, **z, alt_name=np.array(best), alt_offset=np.array(off), k_exit=np.array(ex["k_exit"]))
        win = {"k0": int(info["k0"]), "k1": int(info["k1"]), "k_exit": int(ex["k_exit"]), "t0": round(info["k0"] / hz, 3),
               "t1": round((info["k1"] - 1) / hz, 3), "t_exit": round(ex["k_exit"] / hz, 3), "alt": best, "alt_offset": off,
               "defect_role": role, "defect_t": [round(a / hz, 3), round(b / hz, 3)], "margin_nats": round(margin, 2),
               "pelvis_only_exit_t": None if ex["first_pelvis_only"] is None else round(ex["first_pelvis_only"] / hz, 3),
               "offsets": info["roles"], "file": str(p_npz), "alt_sha256": {r: sha256(Path(al["dir"]) / f"{r}.npz") for r in cfg.roles}}
        windows.append(win)
        rep["splices"].append(win)
        log(f"splice {best} for both dancers {win['t0']}-{win['t1']} s (exit {win['t_exit']} s, margin {win['margin_nats']} nats)")
    return out, windows, rep


# ------------------------------------------------------------------------------------------------ the step

def run(take: str, cfg: SidesCfg | None = None, out_dir: Path | None = None) -> dict:
    t0 = time.time()
    cfg = cfg or load_cfg(take)
    ad = adapter_for(take, cfg)
    work = ad.work
    sd = work / "sides"
    od = Path(out_dir or cpu_dir(work))
    od.mkdir(parents=True, exist_ok=True)
    inp = {r: ad.base_dir / f"{r}.npz" for r in cfg.roles}
    base = {r: load_fit(inp[r]) for r in cfg.roles}
    rep = {"schema": "dancecap.sides.prepare/1", "take": take, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "inputs": {r: {"path": str(inp[r]), "sha256": sha256(inp[r])} for r in cfg.roles},
           "revert": f"both dancers start from the input fit {cfg.base_fit} (no earlier repair is read)", "cfg": cfg.to_json()}
    cur, wins, srep = spin_splices(ad, base, cfg, sd)
    rep["spin"] = srep
    rep["splices"] = wins
    # unwraps on the spliced fit
    betas = {r: np.asarray(base[r]["betas"], float) for r in cfg.roles}
    rest = rest_joints(betas, sd / "evidence" / "rest.json")
    ucfg = cfg.unwrap_cfg()
    rep["unwrap"] = {}
    for r in cfg.roles:
        new, urep = uw.unwrap_fit(cur[r], ucfg, rest=rest[r])
        rep["unwrap"][r] = urep
        cur[r] = new
        log(f"{r}: unwrap changed {urep['changed_frames']} frames (max joint move {urep['max_joint_move_m']} m)")
    # what changed against the input, per role
    changed = {}
    for r in cfg.roles:
        P0, P1 = bd.pose55(base[r]), bd.pose55(cur[r])
        tr0, tr1 = np.asarray(base[r]["transl"]), np.asarray(cur[r]["transl"])
        m = (np.abs(P1 - P0).max((1, 2)) > 0) | (np.abs(tr1 - tr0).max(1) > 0)
        changed[r] = m
        rep.setdefault("changed", {})[r] = {"frames": int(m.sum()), "runs_s": [[round(a / cfg.hz, 3), round(b / cfg.hz, 3)]
                                                                              for a, b in hmm.runs(m)]}
    # checks: FK of the written joints, byte identity outside the changed frames
    chk = {}
    for r in cfg.roles:
        ks = np.flatnonzero(changed[r])
        if len(ks):
            J = bd.forward(cur[r], ks, points=False)["J"]
            chk[r] = {"fk_max_err_mm": round(float(np.abs(J - np.asarray(cur[r]["joints"], float)[ks, :55]).max()) * 1000, 6)}
        else:
            chk[r] = {"fk_max_err_mm": 0.0}
        keep = ~changed[r]
        chk[r]["outside_identical"] = bool(all(np.array_equal(np.asarray(cur[r][k])[keep], np.asarray(base[r][k])[keep])
                                               for k in (*bd.POSE_KEYS, "transl", "joints") if k in base[r]))
    rep["checks"] = chk
    for r in cfg.roles:
        save_fit(od / f"{r}.npz", cur[r])
        save_fit(od / "pod" / f"{r}.npz", {k: cur[r][k] for k in SLIM_KEYS if k in cur[r]})
        rep.setdefault("outputs", {})[r] = {"fit": str(od / f"{r}.npz"), "sha256": sha256(od / f"{r}.npz"),
                                            "pod": str(od / "pod" / f"{r}.npz")}
    rep["seconds"] = round(time.time() - t0, 1)
    write_json(od / "prepare.json", rep)
    log("done", rep["seconds"], "s")
    return rep


def load_cpu(work: Path, roles=("lead", "follow"), slim: bool = False) -> dict:
    d = cpu_dir(work) / ("pod" if slim else "")
    return {r: load_fit(d / f"{r}.npz") for r in roles}
