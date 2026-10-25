"""Merge the solved windows of the "sides" step into one fit (DESIGN 4.1 `merge.py`, 4.2 `merge` + `gate`).

    merge(take, n_parts)  -> work/<take>/smplx_sides/{lead,follow}.npz (the smplx contract + sides_source (T,) int8 + sides_source_names +
                             `sides` JSON), work/<take>/sides/gate.json (per window adopt / revert / flag + reasons), sides/flags.json

Per adopted window the change against the base is kept as local-rotation deltas D = log(R R_base^T) (and a root offset); where adopted
windows overlap (cores split with 1-s overlaps) the deltas are averaged with the windows' ramp weights, so the merged motion is continuous.
Joints a window did not move keep the base's own axis-angle values (byte-identical), and every frame outside the adopted windows + ramps
is byte-identical to the base (sides/cpu: the input + the prepare step's splices / unwraps). The merged fit is gated again per window
(gate.window_gates; the exact mesh depth re-measured only where windows overlap) and as a whole (gate.take_gate); a window that fails
there is reverted and the merge repeated.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import body as bd
from . import hmm
from .config import SidesCfg, load_cfg
from .evidence import sha256, write_json

ROLES = ("lead", "follow")
SRC_NAMES = ("input", "prepare", "window", "ramp")


def log(*a) -> None:
    print("[sides.merge]", *a, flush=True)


def vis_r4_brief(g: dict) -> dict | None:
    """the visibility-consistent part of a window gate (gate.r4_visibility, build C), compact for regate.json / gate.json: decided and
    on-class frames, view-frames where the candidate hides his hand, the detections G1 / G10 dropped, the triangulated wrists G2 dropped"""
    v = (g.get("gates", {}).get("G1") or {}).get("vis_r4")
    if not v:
        return None
    dd = v.get("dropped_detections", {})
    return {"decided": v.get("decided_frames"), "cand_on_class": v.get("cand_on_class_frames"), "base_on_class": v.get("base_on_class_frames"),
            "hand_hidden_view_frames": v.get("hand_hidden_view_frames"), "dropped_mamma_view_frames": dd.get("mamma_view_frames"),
            "dropped_coco_wrist_view_frames": dd.get("coco_wrist_view_frames"), "added_coco_wrist_view_frames": dd.get("coco_wrist_added"),
            "per_cam": dd.get("per_cam"), "g2_wrist_dropped": (g.get("gates", {}).get("G2") or {}).get("vis_r4_wrist_dropped")}


def ramp_weights(win: dict) -> np.ndarray:
    W = win["r1"] - win["r0"] + 1
    w = np.ones(W)
    a, b = win["ramp_in"], win["ramp_out"]
    if a:
        w[:a] = 0.5 - 0.5 * np.cos(np.pi * (np.arange(1, a + 1) / (a + 1)))
    if b:
        w[W - b:] = (0.5 - 0.5 * np.cos(np.pi * (np.arange(1, b + 1) / (b + 1))))[::-1]
    return w


def load_results(sd: Path, solve_dir: Path | None = None) -> dict:
    """{window id: {"json": ..., "npz": {...}}} of every solved window under sides/solve/part_*/ (or solve_dir/part_*/)"""
    out = {}
    for p in sorted(Path(solve_dir or sd / "solve").glob("part_*/win_*.json")):
        j = json.loads(p.read_text(encoding="utf-8"))
        z = p.with_suffix(".npz")
        out[int(j["id"])] = {"json": j, "npz": dict(np.load(z)) if z.exists() else {}, "dir": str(p.parent)}
    return out


def window_candidate(res: dict, win: dict, base: dict, attempt: int | None = None) -> dict | None:
    """{role: {"P": (W,55,3), "transl": (W,3)}} of a window's chosen (or given) attempt"""
    z = res["npz"]
    i = int(z["chosen"]) if attempt is None and "chosen" in z else attempt
    if i is None:
        return None
    out = {}
    for r in ROLES:
        k = f"a{i}_{r}_pose"
        if k in z:
            out[r] = {"P": np.asarray(z[k], float), "transl": np.asarray(z[f"a{i}_{r}_transl"], float)}
    return out


def window_exact(res: dict, attempt: int) -> tuple[dict, dict]:
    z = res["npz"]
    keys = ("hand_pen", "arm_pen", "hand_gap", "hand_pen_part")
    return ({k: z[f"base_exact_{k}"] for k in keys if f"base_exact_{k}" in z}, {k: z[f"a{attempt}_exact_{k}"] for k in keys if f"a{attempt}_exact_{k}" in z})


def merge_fits(base: dict, wins: list, cands: dict) -> tuple[dict, dict]:
    """base {role: fit}, wins [window dict], cands {window id: {role: {"P", "transl"}}} -> ({role: merged fit}, {role: source (T,) int8})"""
    from scipy.spatial.transform import Rotation as Rot
    from ..stitch import nearest_rep
    T = len(base["lead"]["t"])
    out, src = {}, {}
    for r in ROLES:
        acc = np.zeros((T, 55, 3))
        wsum = np.zeros(T)
        tacc = np.zeros((T, 3))
        moved = np.zeros((T, 55), bool)
        core = np.zeros(T, bool)
        cover = np.zeros(T, bool)
        P0 = bd.pose55(base[r])
        tr0 = np.asarray(base[r]["transl"], float)
        for w in wins:
            c = cands.get(w["id"], {}).get(r)
            if c is None:
                continue
            ks = np.arange(w["r0"], w["r1"] + 1)
            ww = ramp_weights(w)
            Rb, Rc = bd.rotmat(P0[ks]), bd.rotmat(c["P"])
            D = Rot.from_matrix((Rc @ np.swapaxes(Rb, -1, -2)).reshape(-1, 3, 3)).as_rotvec().reshape(len(ks), 55, 3)
            ch = np.abs(c["P"] - P0[ks]).max(-1) > 0
            D[~ch] = 0.0
            acc[ks] += ww[:, None, None] * D
            tacc[ks] += ww[:, None] * (c["transl"] - tr0[ks])
            wsum[ks] += ww
            moved[ks] |= ch
            core[w["k0"]:w["k1"] + 1] = True
            cover[ks] = True
        f = dict(base[r])
        for k in (*bd.POSE_KEYS, "transl", "joints"):
            f[k] = np.array(base[r][k], copy=True)
        ks = np.flatnonzero(wsum > 0)
        if len(ks):
            Dm = acc[ks] / wsum[ks, None, None]
            Rn = bd.rotmat(Dm) @ bd.rotmat(P0[ks])
            Pn = Rot.from_matrix(Rn.reshape(-1, 3, 3)).as_rotvec().reshape(len(ks), 55, 3)
            P = P0[ks].copy()
            mv = moved[ks]
            for i in range(len(ks)):
                for j in np.flatnonzero(mv[i]):
                    P[i, j] = nearest_rep(Pn[i, j], P0[ks[i], j])
            bd.write_pose55(f, P, ks)
            dtr = tacc[ks] / wsum[ks, None]
            chg = np.abs(dtr).max(1) > 0
            f["transl"][ks[chg]] = tr0[ks[chg]] + dtr[chg]
            fr = ks[mv.any(1) | chg]
            if len(fr):
                f["joints"][fr, :55] = bd.forward(f, fr, points=False)["J"]
        s = np.zeros(T, np.int8)
        s[cover] = 3
        s[core] = 2
        s[~(moved.any(1) | (np.abs(np.asarray(f["transl"]) - tr0).max(1) > 0))] = 0
        out[r], src[r] = f, s
    return out, src


def cluster_drops(spans: dict, n_failed: dict, tier: dict) -> list:
    """windows that failed after the merge, {id: (first, last gated frame)}: group the ones whose gated frames overlap and revert one
    per group - the one failing the most gates, a non-owner (tier 2) window before an owner window, the later one on a tie"""
    clusters = []
    for i in sorted(spans, key=lambda i: spans[i][0]):
        a, b = spans[i]
        if clusters and a <= clusters[-1]["end"]:
            clusters[-1]["ids"].append(i)
            clusters[-1]["end"] = max(clusters[-1]["end"], b)
        else:
            clusters.append({"ids": [i], "end": b})
    return [max(c["ids"], key=lambda i: (n_failed[i], tier.get(i, 2), spans[i][0])) for c in clusters]


def merge(take: str, cfg: SidesCfg | None = None, out_name: str = "smplx_sides", recheck: bool = True, write: bool = True,
          solve_dir: Path | None = None, gate_name: str = "gate.json") -> dict:
    t0 = time.time()
    cfg = cfg or load_cfg(take)
    from .gate import GateCtx, window_gates, take_gate
    from .plan import load_plan
    from .prepare import load_cpu
    ctx = GateCtx(take, cfg)
    work, sd = ctx.work, ctx.sd
    plan = load_plan(work)
    wins = {w["id"]: w for w in plan["windows"]}
    sdir = Path(solve_dir or sd / "solve")
    res = load_results(sd, sdir)
    base = ctx.base
    prep = json.loads((sd / "cpu" / "prepare.json").read_text(encoding="utf-8")) if (sd / "cpu" / "prepare.json").exists() else {}
    status = {}
    cands = {}
    rg_p = sdir / "regate.json"
    rg = json.loads(rg_p.read_text(encoding="utf-8"))["windows"] if rg_p.exists() else {}
    for i, r in res.items():
        g = rg.get(str(i))
        if g is not None:                      # the local re-gate decides (the attempts' order is kept)
            j = r["json"]
            j["chosen"] = g["chosen"]
            j["status"] = "adopt" if g["chosen"] is not None else "revert"
            for k, row in enumerate(g["attempts"]):
                if k < len(j["attempts"]):
                    j["attempts"][k]["pass"], j["attempts"][k]["failed"] = row["pass"], row["failed"]
                    j["attempts"][k]["targets_missed"] = row.get("targets_missed", [])
            if g["chosen"] is not None:
                r["npz"]["chosen"] = np.array(g["chosen"])
    for i, w in wins.items():
        r = res.get(i)
        if r is None:
            status[i] = {"status": "missing", "why": "no solve result"}
            continue
        j = r["json"]
        if j.get("status") != "adopt":
            status[i] = {"status": "revert", "why": "every attempt failed a gate: " + "; ".join(
                f"{a['name']}: {','.join(a['failed'])}" for a in j.get("attempts", []))}
            continue
        c = window_candidate(r, w, base)
        cands[i] = c
        a = j["attempts"][j["chosen"]]
        status[i] = {"status": "adopt", "attempt": a["name"], "attempt_index": int(j["chosen"]), "targets_missed": a.get("targets_missed", [])}
        rgw = (rg.get(str(i)) or {}).get("attempts", [])
        if j["chosen"] < len(rgw) and rgw[j["chosen"]].get("vis_r4"):
            status[i]["vis_r4"] = rgw[j["chosen"]]["vis_r4"]
    # merge, re-gate the merged motion per adopted window, revert failures (one per overlapping cluster), repeat
    from .gate import window_frames
    rounds = []
    checked = {}                              # window id -> the merged frames (km) it was last gated on
    exact_used = {}                           # window id -> (ks, the exact-mesh arrays of the merged motion it was gated with)

    def km_of(w):
        return window_frames(w, ctx.T)[1]

    def frames_of(fits_, w):
        km = km_of(w)
        return {r: (bd.pose55(fits_[r], km), np.asarray(fits_[r]["transl"], float)[km]) for r in ROLES}

    def same_frames(a, b):
        return all(np.array_equal(a[r][0], b[r][0]) and np.array_equal(a[r][1], b[r][1]) for r in ROLES)
    while True:
        adopted = [wins[i] for i in sorted(cands)]
        merged, src = merge_fits(base, adopted, cands)
        if not recheck:
            break
        bad = []
        n_gated = 0
        for w in adopted:
            i = w["id"]
            ks = np.arange(w["r0"], w["r1"] + 1)
            fm = frames_of(merged, w)
            if i in checked and same_frames(checked[i], fm):
                continue                                   # nothing the gates read changed since the last check
            # the candidate as gated (the base outside the window): identical on every frame the gates read -> its re-gate stands
            alone, _ = merge_fits(base, [w], {i: cands[i]})
            if same_frames(frames_of(alone, w), fm):
                a = res[i]["json"]["attempts"][res[i]["json"]["chosen"]]
                status[i]["merged_gate"] = {"pass": True, "failed": [], "targets_missed": a.get("targets_missed", []),
                                            "note": "the merged motion equals the gated candidate on the gated frames"}
                checked[i] = fm
                exact_used[i] = (ks, window_exact(res[i], res[i]["json"]["chosen"])[1])
                continue
            same = all(np.array_equal(bd.pose55(merged[r], ks), cands[i][r]["P"]) for r in cands[i]) and \
                all(np.array_equal(np.asarray(merged[r]["transl"])[ks], cands[i][r]["transl"]) for r in cands[i])
            att = res[i]["json"]["chosen"]
            ex_b, ex_c = window_exact(res[i], att)
            if not same:
                # overlaps: re-measure the exact depth of the merged motion on the frames where it differs from the candidate
                from .. import zouk_contacts as zc
                diff = np.zeros(len(ks), bool)
                for r in cands[i]:
                    diff |= np.abs(bd.pose55(merged[r], ks) - cands[i][r]["P"]).max((1, 2)) > 0
                ii = np.flatnonzero(diff)
                if len(ii) and ex_c:
                    ex_c = {k: np.array(v, copy=True) for k, v in ex_c.items()}
                    e = zc.exact_of({r: merged[r] for r in ROLES}, ks[ii], cfg.gate_exact_device, 48)
                    for k in ex_c:
                        ex_c[k][ii] = e[k]
            g = window_gates(ctx, merged, w, exact={"base": ex_b, "cand": ex_c} if ex_b and ex_c else None,
                             trust={"share": res[i]["json"]["attempts"][att]["trust"].get("share", 0.0)})
            n_gated += 1
            status[i]["merged_gate"] = {"pass": g["pass"], "failed": g["failed"], "targets_missed": g["targets_missed"], "vis_r4": vis_r4_brief(g)}
            checked[i] = fm
            exact_used[i] = (ks, ex_c)
            if not g["pass"]:
                bad.append(i)
        # one revert per cluster of failing windows whose gated frames overlap: the one failing the most gates, a non-owner (tier 2)
        # window before an owner window, the later one on a tie
        drop = cluster_drops({i: (int(km_of(wins[i])[0]), int(km_of(wins[i])[-1])) for i in bad},
                             {i: len(status[i]["merged_gate"]["failed"]) for i in bad}, {i: wins[i].get("tier", 2) for i in bad})
        rounds.append({"adopted": len(adopted), "gated": n_gated, "failed_after_merge": bad, "reverted": drop})
        if not bad:
            break
        for i in drop:
            status[i] = {"status": "revert", "why": "failed after the merge (overlap with a neighbour): " + ",".join(status[i]["merged_gate"]["failed"])}
            cands.pop(i)
            checked.pop(i, None)
    tg = take_gate(ctx, merged, [wins[i] for i in sorted(cands)])
    flags = []
    for i, s in status.items():
        w = wins[i]
        if s["status"] != "adopt" or s.get("targets_missed") or (s.get("merged_gate") or {}).get("targets_missed"):
            flags.append({"id": i, "t0": w["t0"], "t1": w["t1"], "film0": w["film0"], "film1": w["film1"], "reasons": w["reasons"],
                          "status": s["status"], "why": s.get("why"), "targets_missed": sorted(set(s.get("targets_missed", [])) |
                                                                                             set((s.get("merged_gate") or {}).get("targets_missed", [])))})
    T = ctx.T
    rep = {"schema": "dancecap.sides.gate/1", "take": take, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "windows": {str(i): {**status[i], "t0": wins[i]["t0"], "t1": wins[i]["t1"], "film0": wins[i]["film0"], "film1": wins[i]["film1"],
                                "reasons": wins[i]["reasons"], "dancers": {r: d["joints"] for r, d in wins[i]["dancers"].items()},
                                "r0": wins[i]["r0"], "r1": wins[i]["r1"], "k0": wins[i]["k0"], "k1": wins[i]["k1"]} for i in sorted(status)},
           "rounds": rounds, "take_gate": tg, "solve_dir": str(sdir),
           "summary": {"windows": len(wins), "adopted": sum(s["status"] == "adopt" for s in status.values()),
                       "reverted": sum(s["status"] == "revert" for s in status.values()),
                       "missing": sum(s["status"] == "missing" for s in status.values()), "flagged": len(flags),
                       "frames_changed": {r: int((src[r] > 0).sum()) for r in ROLES},
                       "share_changed": {r: round(float((src[r] > 0).mean()), 3) for r in ROLES}}}
    if write:
        od = work / out_name
        od.mkdir(parents=True, exist_ok=True)
        for r in ROLES:
            f = merged[r]
            s = src[r].copy()
            # the prepare step's frames (splice / unwrap) that no window touched
            pchg = (np.abs(bd.pose55(base[r]) - bd.pose55(ctx.input[r])).max((1, 2)) > 0) | \
                (np.abs(np.asarray(base[r]["transl"]) - np.asarray(ctx.input[r]["transl"])).max(1) > 0)
            s[(s == 0) & pchg] = 1
            f["sides_source"] = s
            f["sides_source_names"] = np.array(SRC_NAMES)
            f["sides"] = np.array(json.dumps({"what": "dancecap.sides: the detected input + the prepare step (splices / unwraps) + the "
                                                      "adopted solver windows (gate.json); everything else byte-identical to the input",
                                              "input": str(ctx.ad.base_dir), "input_sha256": {k: sha256(ctx.ad.base_dir / f"{k}.npz") for k in ROLES},
                                              "prepare": prep.get("splices", []), "adopted": [i for i in sorted(cands)],
                                              "quality_fields": "reproj_px / quality* / contact are the base's (not recomputed)"}))
            np.savez(od / f"{r}.npz", **f)
            rep.setdefault("outputs", {})[r] = {"path": str(od / f"{r}.npz"), "sha256": sha256(od / f"{r}.npz")}
        ez = {}
        for i in sorted(cands):
            if i in exact_used and exact_used[i][1]:
                ez[f"w{i}_ks"] = exact_used[i][0]
                for k, v in exact_used[i][1].items():
                    ez[f"w{i}_{k}"] = np.asarray(v)
        np.savez_compressed(sd / gate_name.replace(".json", "_exact.npz"), **ez)
        rep["exact_file"] = str(sd / gate_name.replace(".json", "_exact.npz"))
        write_json(sd / gate_name, rep)
        write_json(sd / gate_name.replace("gate", "flags"), {"take": take, "flags": flags})
    rep["seconds"] = round(time.time() - t0, 1)
    log(f"merged: {rep['summary']}", rep["seconds"], "s")
    return rep


# ------------------------------------------------------------------------------------------------ re-gating the solved attempts locally

def _regate_worker(args):
    take, ids, cfg_json, solve_dir = args
    from .config import SidesCfg, apply_overrides
    from .gate import GateCtx, window_gates
    from .plan import load_plan
    cfg = apply_overrides(SidesCfg(), cfg_json)
    ctx = GateCtx(take, cfg)
    wins = {w["id"]: w for w in load_plan(ctx.work)["windows"]}
    res = load_results(ctx.sd, Path(solve_dir) if solve_dir else None)
    out = {}
    for i in ids:
        r = res.get(i)
        if r is None:
            continue
        w = wins[i]
        rows = []
        chosen = None
        for ai, a in enumerate(r["json"].get("attempts", [])):
            c = window_candidate(r, w, ctx.base, ai)
            if not c:
                rows.append({"name": a["name"], "pass": False, "failed": ["missing"]})
                continue
            from .gate import fit_window
            ks = np.arange(w["r0"], w["r1"] + 1)
            cand = {rr: (fit_window(ctx.base[rr], c[rr]["P"], c[rr]["transl"], ks) if rr in c else ctx.base[rr]) for rr in ROLES}
            ex_b, ex_c = window_exact(r, ai)
            win = w
            if a.get("only"):
                from .solve import _cut_win
                win = _cut_win(w, a["only"])
            g = window_gates(ctx, cand, win, exact={"base": ex_b, "cand": ex_c} if ex_b and ex_c else None,
                             trust={"share": a.get("trust", {}).get("share", 0.0)})
            rows.append({"name": a["name"], "pass": g["pass"], "failed": g["failed"], "targets_missed": g["targets_missed"],
                         "vis_r4": vis_r4_brief(g),
                         "pod_pass": a.get("pass"), "pod_failed": a.get("failed")})
            if g["pass"] and chosen is None:
                chosen = ai
                break
        out[str(i)] = {"chosen": chosen, "attempts": rows}
    return out


def regate(take: str, cfg: SidesCfg | None = None, workers: int = 4, ids: list | None = None, solve_dir: Path | None = None) -> dict:
    """re-run the window gates (this PC's gate code) on every solved attempt the pods saved, in the attempts' order; the first that
    passes is the window's result -> work/<take>/sides/solve/regate.json (merge prefers it over the pods' own choice)"""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    cfg = cfg or load_cfg(take)
    from .adapters import work_dir
    sd = work_dir(take) / "sides"
    sdir = Path(solve_dir or sd / "solve")
    all_ids = sorted(load_results(sd, sdir)) if ids is None else list(ids)
    nw = max(1, min(workers, len(all_ids)))
    subs = [all_ids[k::nw] for k in range(nw)]
    with ProcessPoolExecutor(max_workers=nw, mp_context=mp.get_context("spawn")) as ex:
        parts = list(ex.map(_regate_worker, [(take, s, cfg.to_json(), str(sdir)) for s in subs]))
    out = {"what": "the window gates of this PC re-run on the pods' saved attempts (merge uses the first passing attempt)",
           "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "windows": {}}
    for p in parts:
        out["windows"].update(p)
    out["seconds"] = round(time.time() - t0, 1)
    write_json(sdir / "regate.json", out)
    log(f"regate: {sum(v['chosen'] is not None for v in out['windows'].values())} of {len(out['windows'])} windows pass", out["seconds"], "s")
    return out


def combine_rounds(take: str, solve_dirs: list, out_dir: Path) -> dict:
    """several solve rounds (each re-gated here: <dir>/regate.json), in priority order -> out_dir/part_c/ with, per window, the first
    round whose local re-gate passes (else the first round that has the window), and out_dir/regate.json for merge"""
    import shutil
    from .adapters import work_dir
    sd = work_dir(take) / "sides"
    out_dir = Path(out_dir)
    pc = out_dir / "part_c"
    if pc.exists():
        shutil.rmtree(pc)
    pc.mkdir(parents=True)
    rounds = []
    for d in solve_dirs:
        d = Path(d)
        rg = json.loads((d / "regate.json").read_text(encoding="utf-8"))["windows"]
        files = {int(p.stem.split("_")[1]): p for p in d.glob("part_*/win_*.json")}
        rounds.append((d, rg, files))
    ids = sorted({i for _, _, f in rounds for i in f})
    out = {"what": "per window the first round (in this order) whose local re-gate passes", "rounds": [str(d) for d, _, _ in rounds],
           "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "windows": {}}
    for i in ids:
        pick = None
        for d, rg, files in rounds:
            if i in files and (rg.get(str(i)) or {}).get("chosen") is not None:
                pick = (d, rg, files)
                break
        if pick is None:
            pick = next((x for x in rounds if i in x[2]), None)
        d, rg, files = pick
        src = files[i]
        shutil.copy2(src, pc / src.name)
        if src.with_suffix(".npz").exists():
            shutil.copy2(src.with_suffix(".npz"), pc / src.with_suffix(".npz").name)
        row = dict(rg.get(str(i)) or {"chosen": None, "attempts": []})
        row["source"] = str(d.name)
        out["windows"][str(i)] = row
    write_json(out_dir / "regate.json", out)
    log(f"combine: {sum(v['chosen'] is not None for v in out['windows'].values())} of {len(ids)} windows pass in some round")
    return out
