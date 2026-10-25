"""The report of the "sides" step: scorecard (DESIGN 7.3), the violation list in film time, the relation lanes (a PNG per window).

    python -m dancecap.sides report --take <t> [--compare <fit dir>]   -> work/<t>/sides/report.json, report.md, review/lanes_*.png

The owner windows of a take (which seconds the scorecard measures, which film spans come first in the list, which lane PNGs to draw) are
DATA, not code: work/<t>/sides/report_windows.json (keys in WINDOW_KEYS; missing keys = skipped rows).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import geom, hmm
from .adapters import adapter_for
from .config import SidesCfg, load_cfg
from .evidence import write_json
from .filmtime import FilmMap, mmss

WINDOW_KEYS = ("gli", "crossings", "swings", "rms", "elbow_tri", "inside", "spin_heading", "spin_foot", "replay_foot", "first_film", "lanes")
REASON_ORDER = ("thru", "side", "inside", "real", "swing", "spin", "limit", "wrap")


def log(*a) -> None:
    print("[sides.report]", *a, flush=True)


# ------------------------------------------------------------------------------------------------ metrics (scout definitions)

def zigzag_reversals(x: np.ndarray, thr: float = 0.03) -> np.ndarray:
    """indices of the turning points of x with hysteresis thr (each turn: a swing of >= thr into it and out of it) (S/elib.py)"""
    turns = []
    lo_i = hi_i = 0
    mode = 0
    ext = 0
    for i in range(1, len(x)):
        if mode == 0:
            if x[i] > x[hi_i]:
                hi_i = i
            if x[i] < x[lo_i]:
                lo_i = i
            if x[hi_i] - x[lo_i] >= thr:
                mode = 1 if hi_i > lo_i else -1
                ext = hi_i if mode == 1 else lo_i
        elif mode == 1:
            if x[i] > x[ext]:
                ext = i
            elif x[ext] - x[i] >= thr:
                turns.append(ext)
                mode, ext = -1, i
        else:
            if x[i] < x[ext]:
                ext = i
            elif x[i] - x[ext] >= thr:
                turns.append(ext)
                mode, ext = 1, i
    return np.array(turns, int)


def swings_within(x: np.ndarray, thr: float = 0.03, win: int = 30) -> int:
    """up-and-down swings: consecutive reversals (>= thr) within `win` frames of each other"""
    r = zigzag_reversals(x, thr)
    return int((np.diff(r) <= win).sum()) if len(r) > 1 else 0


def band_rms_windows(x: np.ndarray, hz: float = 30.0, lo: float = 1.0, hi: float = 5.0, win: int = 60, step: int = 15) -> np.ndarray:
    """sliding 2-s Hann windows (detrended): RMS amplitude in lo..hi Hz (S/elib.py band_rms)"""
    w = np.hanning(win)
    f = np.fft.rfftfreq(win, 1 / hz)
    band = (f >= lo) & (f <= hi)
    out = []
    tt = np.arange(win)
    for a in range(0, len(x) - win + 1, step):
        seg = x[a:a + win]
        if not np.isfinite(seg).all():
            out.append(np.nan)
            continue
        p = np.polyfit(tt, seg, 1)
        seg = (seg - np.polyval(p, tt)) * w
        pw = np.abs(np.fft.rfft(seg)) ** 2
        out.append(np.sqrt(2 * pw[band].sum() / (win * (w ** 2).sum())))
    return np.array(out)


def close_crossings(E: np.ndarray, e: np.ndarray, t: np.ndarray, m: np.ndarray) -> list:
    """vertical elbow crossings while the elbows are horizontally close (< 12 cm): Schmitt +-2 cm on dy (S/judge1/j1_elbow.py)"""
    dh = np.linalg.norm((E - e)[:, [0, 2]], axis=1)
    dy = (E - e)[:, 1]
    st = 0
    xs = []
    for k in np.flatnonzero(m):
        if dh[k] > 0.12:
            st = 0
            continue
        sg = 1 if dy[k] > 0.02 else (-1 if dy[k] < -0.02 else 0)
        if sg != 0:
            if st != 0 and sg != st:
                xs.append(round(float(t[k]), 2))
            st = sg
    return xs


def arm_metrics(JL: np.ndarray, JF: np.ndarray, t: np.ndarray, W: dict, triL: dict, inside: np.ndarray | None, hS: np.ndarray) -> dict:
    """his right arm against her left arm and the triangulated elbow, in the owner windows"""
    out = {}
    if "gli" in W:
        a, b = W["gli"]
        G = geom.gli_chain(JL[:, [17, 19, 21]], JF[:, [16, 18, 20]])
        ev = geom.passthrough_events(G, merge=0)      # the S/rel/r_gli.py count (DESIGN 9.2): neighbouring frames are not merged
        ev = ev[(t[ev] >= a) & (t[ev] < b)]
        out["gli_events"] = int(len(ev))
        out["gli_event_times"] = [round(float(t[k]), 2) for k in ev]
    if "crossings" in W:
        a, b = W["crossings"]
        out["close_crossings"] = close_crossings(JL[:, 19], JF[:, 18], t, (t >= a) & (t < b))
    x = JL[:, 19, 1] - hS
    if "swings" in W:
        a, b = W["swings"]
        m = (t >= a) & (t < b)
        out["swings_3cm_1s"] = swings_within(x[m], 0.03, 30)
        out["reversals_3cm"] = int(len(zigzag_reversals(x[m], 0.03)))
    if "rms" in W:
        a, b = W["rms"]
        m = (t >= a) & (t < b)
        r = band_rms_windows(x[m])
        out["rms_1_5hz_p90_cm"] = round(float(np.nanpercentile(r, 90)) * 100, 2) if np.isfinite(r).any() else None
    if "elbow_tri" in W and triL is not None:
        rows = []
        for a, b in W["elbow_tri"]:
            m = (t >= a) & (t < b)
            row = {}
            for mv in (2, 3):
                ok = m & (triL["nv"][:, 8] >= mv)
                d = np.linalg.norm(JL[ok, 19] - triL["Xd"][ok, 8], axis=1)
                row[f"ge{mv}"] = [round(float(np.median(d)) * 100, 1) if len(d) else None, int(len(d))]
            rows.append(row)
        out["elbow_to_tri_cm"] = rows
    if "inside" in W and inside is not None:
        rows = []
        for a, b in W["inside"]:
            m = (t >= a) & (t < b)
            dd = inside[m]
            rows.append({"frames_gt2cm": int((dd > 0.02).sum()), "frames_gt3cm": int((dd > 0.03).sum()), "frames_gt5cm": int((dd > 0.05).sum()),
                         "p99_cm": round(float(np.percentile(dd, 99)) * 100, 1) if len(dd) else None})
        out["inside_capsules"] = rows
    return out


def foot_minus_pelvis(J: np.ndarray) -> np.ndarray:
    """(T,2) yaw of each foot (ankle -> foot joint) minus the hip-line heading (deg, wrapped) (S/final/f1_exit.py)"""
    pel = geom.line_heading(J[:, 1], J[:, 2])
    out = []
    for a, f in ((7, 10), (8, 11)):
        v = (J[:, f] - J[:, a]) * np.array([1, 0, 1.0])
        out.append(geom.wrap_deg(geom.heading(v, min_h=0.02) - pel))
    return np.stack(out, 1)


def face_contra_rows(fit_f: dict, views: dict, obs: dict, cams: list, t: np.ndarray, W: dict, role: str = "follow") -> dict:
    """frames where >= 2 phones' face bits contradict the fit's own head heading (gate G8's rule: seen but turned > 110 deg away, or
    hidden while turned < 70 deg toward the phone; bits only where the head is predicted visible), in W["spin_heading"] and the take"""
    from .prepare import azimuths, face_labels, head_heading
    from .splice import face_contradictions
    ks = np.arange(len(t))
    J = np.asarray(fit_f["joints"], float)
    fc = face_contradictions(face_labels(views, cams, role, ks), azimuths(obs, cams, J, ks), head_heading(fit_f, ks))
    out = {"face_contra_frames_ge2_take": int((fc >= 2).sum())}
    if "spin_heading" in W:
        a, b = W["spin_heading"]
        m = (t >= a) & (t < b)
        out["face_contra_frames_ge2"] = int((fc[m] >= 2).sum())
        out["face_contra_max_phones"] = int(fc[m].max()) if m.any() else 0
    return out


def spin_metrics(JF: np.ndarray, t: np.ndarray, W: dict, triF: dict, yaw: dict | None, stance: np.ndarray | None) -> dict:
    out = {}
    if "spin_heading" in W:
        a, b = W["spin_heading"]
        # as S/design/e3_x5966.py: the raw COCO-3D hip line (all detections, >= 2 views) on the 10-Hz samples (k % 3 == 0)
        X, nvv = (triF["X_all"], triF["nv_all"]) if "X_all" in triF else (triF["X"], triF["nv"])
        m = (t >= a) & (t <= b) & (nvv[:, 11] >= 2) & (nvv[:, 12] >= 2) & (np.arange(len(t)) % 3 == 0)
        hb = geom.line_heading(JF[:, 1], JF[:, 2])
        ht = geom.line_heading(X[:, 11].astype(float), X[:, 12].astype(float))
        e = np.abs(geom.wrap_deg(hb - ht))[m]
        out["heading_vs_tri_p90_max_deg"] = [round(float(np.percentile(e, 90)), 1), round(float(e.max()), 1)] if len(e) else None
        if yaw is not None:
            mm = (t >= a) & (t < b)
            out["psi_star_err_p90_max_deg"] = [round(float(np.percentile(np.abs(yaw["follow__err"][mm]), 90)), 1),
                                               round(float(np.abs(yaw["follow__err"][mm]).max()), 1)]
    fp = foot_minus_pelvis(JF)
    if "spin_foot" in W:
        a, b = W["spin_foot"]
        m = (t >= a) & (t < b)
        out["left_foot_minus_pelvis_max_deg"] = round(float(np.nanmax(np.abs(fp[m, 0]))), 1)
    if "replay_foot" in W and stance is not None:
        a, b = W["replay_foot"]
        m = (t >= a) & (t < b)
        st = np.asarray(stance, bool)[m]
        v = np.where(st, np.abs(fp[m]), np.nan)
        out["stance_foot_minus_pelvis_max_deg"] = round(float(np.nanmax(v)), 1) if np.isfinite(v).any() else None
    return out


# ------------------------------------------------------------------------------------------------ the violation list in film time

def group_items(items: list, gap_s: float = 1.0) -> list:
    """merge items of the same reason / relation / role / joint closer than gap_s"""
    key = lambda it: (it["reason"], it.get("rel"), it.get("role"), it.get("joint"))
    out = []
    last = {}
    for it in sorted(items, key=lambda x: x["t0"]):
        k = key(it)
        g = last.get(k)
        if g is not None and it["t0"] - g["t1"] <= gap_s:
            g["t1"] = max(g["t1"], it["t1"])
            g["film1"] = it.get("film1", g.get("film1"))
            g["n"] += 1
            g["frames"] += it.get("frames", 0)
            if isinstance(it.get("worst"), (int, float)) and isinstance(g.get("worst"), (int, float)):
                g["worst"] = it["worst"] if abs(it["worst"]) > abs(g["worst"]) else g["worst"]
            g["held"] = g.get("held", False) or it.get("held", False)
            if it.get("film_extra"):
                g["film_extra"] = sorted(set(g.get("film_extra", [])) | set(it["film_extra"]))
            g["in_band"] = g.get("in_band", False) or it.get("in_band", False)
            if abs(it.get("turns_off", 0) or 0) > abs(g.get("turns_off", 0) or 0):
                g["turns_off"] = it["turns_off"]
            if it.get("face_contra_max", 0) > g.get("face_contra_max", 0):
                g["face_contra_max"] = it["face_contra_max"]
            continue
        g = dict(it)
        g["n"] = 1
        out.append(g)
        last[k] = g
    return sorted(out, key=lambda x: x["t0"])


def describe(it: dict) -> str:
    r = it["reason"]
    if r == "thru":
        where = "in a decided (held) run" if it.get("held") else ("while entangled, sign undecided" if it.get("in_band") else "outside the band")
        return f"{it['rel']} pass-through {where}, dG {it['worst']:+.2f}, {it['n']} event(s)"
    if r == "side":
        if it.get("rel") == "R4":
            return f"R4 palm decided {it.get('decided')}, base {it.get('base')} (camera-blind share {it.get('blind_share')})"
        return f"{it['rel']} decided {'+' if it.get('decided', 0) > 0 else '-'}, base on the other side (worst s*m {it.get('worst')})"
    if r == "inside":
        return f"his palm / forearm inside her torso, worst {100 * it['worst']:.1f} cm"
    if r == "real":
        return f"{it['role']} {it['joint']} {100 * it.get('median', 0):.1f} cm (max {100 * it['worst']:.1f}) from the triangulated joint"
    if r == "swing":
        tr = it.get("tri_rms")
        return f"{it['role']} {it['joint']} 1-5 Hz height RMS {100 * it['worst']:.1f} cm (triangulated {'-' if tr is None else f'{100 * tr:.1f}'})"
    if r == "spin":
        tb = it.get("turns_off", 0)
        turn = f"; the base ends {abs(tb)} turn{'s' if abs(tb) > 1 else ''} {'behind' if tb < 0 else 'ahead'}" if tb else ""
        return (f"{it['role']} heading {it['worst']:.0f} deg from the camera yaw track, {it.get('face_contra_max', 0)} phones contradict "
                f"its face direction{turn}")
    if r == "limit":
        return f"{it['role']} {it['joint']} twist beyond the gate"
    if r == "wrap":
        return f"{it['role']} {it['joint']} chain wound {it.get('worst', 0):.0f} deg"
    return r


def in_spans(f: float | None, spans: list) -> bool:
    return f is not None and any(a <= f <= b for a, b in spans)


# ------------------------------------------------------------------------------------------------ lanes PNG

LANES = ("R1_lead", "R1_follow", "R2", "R4", "R6", "R7", "R8", "R9_RL", "R9_LR", "R11_lead", "R12_RL_kn", "R13")


def lanes_png(path: Path, rel: dict, yaw: dict | None, viol: list, t0: float, t1: float, fm: FilmMap, title: str) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    t = rel["t"]
    m = (t >= t0) & (t < t1)
    tt = t[m]
    lanes = [x for x in LANES if f"{x}__s" in rel or x == "R4"]
    n = len(lanes) + 3
    fig, ax = plt.subplots(figsize=(18, 0.55 * n + 2.2))
    cols = {-1: (0.95, 0.55, 0.15), 1: (0.2, 0.45, 0.9), 0: (0.85, 0.85, 0.85)}
    r4c = {1: (0.85, 0.3, 0.3), 2: (0.2, 0.7, 0.4), 3: (0.2, 0.45, 0.9), 0: (0.85, 0.85, 0.85)}
    dt = 1 / 30
    for i, rid in enumerate(lanes):
        y = n - 1 - i
        if rid == "R4":
            s = rel["R4__cls"][m]
            g = rel["R4__gate"][m]
            cb = rel["R4__cls_base"][m]
            for k in range(len(tt)):
                if g[k]:
                    ax.add_patch(plt.Rectangle((tt[k], y + 0.35), dt, 0.55, color=r4c[int(s[k])], lw=0))
                    if cb[k]:
                        ax.add_patch(plt.Rectangle((tt[k], y + 0.05), dt, 0.25, color=r4c[int(cb[k])], lw=0))
            bl = rel["R4__blind"][m]
            ax.scatter(tt[bl], np.full(bl.sum(), y + 0.95), s=2, c="k", marker="|")
        else:
            s = rel[f"{rid}__s"][m]
            g = rel[f"{rid}__gate"][m]
            mb = rel[f"{rid}__m_base"][m]
            dz = geom.REL_TABLE[rid][1]
            for k in range(len(tt)):
                if g[k]:
                    ax.add_patch(plt.Rectangle((tt[k], y + 0.35), dt, 0.55, color=cols[int(s[k])], lw=0))
                    if s[k] != 0:
                        okb = s[k] * mb[k] >= -dz / 2
                        ax.add_patch(plt.Rectangle((tt[k], y + 0.05), dt, 0.25, color=(0.3, 0.75, 0.3) if okb else (0.9, 0.1, 0.1), lw=0))
            band = rel.get(f"{rid}__band")
            if band is not None:
                bb = band[m]
                ax.scatter(tt[bb], np.full(bb.sum(), y + 0.95), s=2, c="k", marker="|")
    # yaw lanes
    if yaw is not None:
        for j, r in enumerate(("lead", "follow")):
            y = 2 - j
            e = np.abs(yaw[f"{r}__err"][m])
            ax.scatter(tt, y + 0.1 + 0.8 * np.clip(e / 180, 0, 1), s=1, c=np.where(e > 45, "r", "g"))
            fc = yaw[f"{r}__face_contra"][m]
            ax.scatter(tt[fc >= 2], np.full((fc >= 2).sum(), y + 0.95), s=6, c="m", marker="v")
    # violations
    y = 0
    vc = {"thru": "k", "side": "r", "inside": "m", "real": "c", "swing": "y", "spin": "b", "limit": "g", "wrap": "brown"}
    for it in viol:
        if it["t1"] < t0 or it["t0"] > t1:
            continue
        k = REASON_ORDER.index(it["reason"]) if it["reason"] in REASON_ORDER else 0
        ax.add_patch(plt.Rectangle((it["t0"], y + 0.1 * k), max(it["t1"] - it["t0"], 0.05), 0.09, color=vc.get(it["reason"], "k"), lw=0))
    labels = list(lanes) + ["yaw lead |err|", "yaw follow |err|", "violations"]
    ax.set_yticks([n - 1 - i + 0.5 for i in range(len(lanes))] + [2.5, 1.5, 0.5])
    ax.set_yticklabels(labels)
    ax.set_ylim(0, n)
    ax.set_xlim(t0, t1)
    ticks = np.arange(np.ceil(t0), t1, 2 if t1 - t0 > 30 else 0.5)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{x:.1f}\n{mmss(fm.main(x))}" for x in ticks], fontsize=7)
    ax.set_xlabel("dance s / film m:ss")
    ax.set_title(title, fontsize=10)
    hand = [Patch(color=cols[1], label="decided +"), Patch(color=cols[-1], label="decided -"), Patch(color=cols[0], label="undecided"),
            Patch(color=(0.3, 0.75, 0.3), label="base on decided side"), Patch(color=(0.9, 0.1, 0.1), label="base violates"),
            Patch(color=r4c[1], label="R4 front"), Patch(color=r4c[2], label="R4 flank"), Patch(color=r4c[3], label="R4 back")]
    hand += [Patch(color=c, label=f"viol {k}") for k, c in vc.items()]
    ax.legend(handles=hand, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=8, fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


# ------------------------------------------------------------------------------------------------ the report

def scorecard(take: str, cfg: SidesCfg, W: dict, compare: dict | None = None) -> dict:
    """the measurable 7.3 rows for the base (input) and any comparison fits ({name: fit dir})"""
    from .decide import inside_depth
    from .evidence import capsule_set, posed_capsules
    ad = adapter_for(take, cfg)
    work = ad.work
    base = ad.base()
    t = ad.t
    evd = work / "sides" / "evidence"
    tri = {r: dict(np.load(evd / f"tri_{r}.npz")) for r in cfg.roles}
    yaw = dict(np.load(work / "sides" / "yaw.npz")) if (work / "sides" / "yaw.npz").exists() else None
    caps = capsule_set({r: np.asarray(base[r]["betas"], float) for r in cfg.roles}, evd / "capsules.json")
    JF = np.asarray(base["follow"]["joints"], float)
    RgF = geom.global_rotations(base["follow"]["global_orient"], base["follow"]["body_pose"])
    pf = posed_capsules(caps["follow"], JF, RgF)
    fits = {"input": base}
    for nm, d in (compare or {}).items():
        out = {}
        for r in cfg.roles:
            p = Path(d) / f"{r}.npz"
            if p.exists():
                with np.load(p, allow_pickle=True) as z:
                    out[r] = {k: z[k] for k in z.files}
            else:
                out[r] = base[r]
        fits[nm] = out
    hS = np.asarray(base["lead"]["joints"], float)[:, 17, 1]
    from .evidence import load_views
    views = load_views(work, ad.cams)
    obs = ad.obs()
    sc = {}
    for nm, f in fits.items():
        JL = np.asarray(f["lead"]["joints"], float)
        JFf = np.asarray(f["follow"]["joints"], float)
        RgL = geom.global_rotations(f["lead"]["global_orient"], f["lead"]["body_pose"])
        pl = posed_capsules(caps["lead"], JL, RgL)
        ins = inside_depth(caps["lead"], caps["follow"], pl, pf if JFf is JF else posed_capsules(
            caps["follow"], JFf, geom.global_rotations(f["follow"]["global_orient"], f["follow"]["body_pose"])))
        row = arm_metrics(JL, JFf, t, W, tri["lead"], ins, hS)
        stance = np.asarray(f["follow"]["stance"], bool) if "stance" in f["follow"] else None
        row.update(spin_metrics(JFf, t, W, tri["follow"], yaw if nm == "input" else None, stance))
        row.update(face_contra_rows(f["follow"], views, obs, ad.cams, t, W))
        sc[nm] = row
    return sc


def run(take: str, cfg: SidesCfg | None = None, compare: dict | None = None, first_film: list | None = None) -> dict:
    cfg = cfg or load_cfg(take)
    ad = adapter_for(take, cfg)
    work = ad.work
    sd = work / "sides"
    fm = FilmMap.for_work(work)
    W = json.loads((sd / "report_windows.json").read_text(encoding="utf-8")) if (sd / "report_windows.json").exists() else {}
    first = first_film or W.get("first_film", [])
    vj = json.loads((sd / "violations.json").read_text(encoding="utf-8"))
    rj = json.loads((sd / "relations.json").read_text(encoding="utf-8"))
    ev = json.loads((sd / "evidence" / "evidence.json").read_text(encoding="utf-8"))
    sj = json.loads((sd / "spins.json").read_text(encoding="utf-8"))
    rel = dict(np.load(sd / "relations.npz"))
    yaw = dict(np.load(sd / "yaw.npz"))
    items = vj["items"]
    groups = group_items(items)
    sc = scorecard(take, cfg, W, compare)
    # lanes
    pngs = []
    for a, b in W.get("lanes", []):
        p = work / "review" / "sides" / f"lanes_{int(a):03d}_{int(b):03d}.png"
        lanes_png(p, rel, yaw, items, float(a), float(b), fm, f"{take}: relation lanes {a}-{b} s (dance) / film {mmss(fm.main(a))}-{mmss(fm.main(b))}")
        pngs.append(str(p))
    # R4 decided runs per hold
    r4 = rj["relations"]["R4"]
    rep = {"take": take, "counts": vj["counts"], "groups": len(groups), "scorecard": sc, "lanes": pngs, "r4": r4,
           "spins": sj["roles"], "film_map": fm.source}
    write_json(sd / "report.json", rep)
    # markdown
    L = [f"# sides report: {take} (Phase A, no motion changed)", "",
         f"Base fit `{ad.cfg.base_fit}`; cameras {', '.join(ad.cams)}; film times from `{fm.source or 'identity'}`. "
         f"Evidence: {ev.get('seconds')} s, decide: {rj.get('seconds')} s.", "",
         "## Scorecard (DESIGN 7.3, the rows measurable before the solver)", ""]
    names = list(sc)
    L.append("| Metric | " + " | ".join(names) + " |")
    L.append("|---|" + "---|" * len(names))
    keys = []
    for v in sc.values():
        for k in v:
            if k not in keys:
                keys.append(k)
    for k in keys:
        if k.endswith("_times"):
            continue
        L.append(f"| {k} | " + " | ".join(json.dumps(sc[n].get(k)) for n in names) + " |")
    L += ["", f"Windows (dance s): `{json.dumps({k: W[k] for k in W if k not in ('first_film', 'lanes', 'what')})}`", ""]
    L += ["## Violation counts", "", "| Reason | items | groups (merged within 1 s) | in the first film spans |", "|---|---|---|---|"]
    for r in REASON_ORDER:
        n_i = sum(1 for it in items if it["reason"] == r)
        n_g = sum(1 for g in groups if g["reason"] == r)
        n_f = sum(1 for g in groups if g["reason"] == r and (in_spans(g.get("film0"), first) or in_spans(g.get("film1"), first)))
        L.append(f"| {r} | {n_i} | {n_g} | {n_f} |")
    L += ["", "## R4: his right palm on her torso, decided runs", "",
          f"Defined on {r4['defined_frames']} frames, decided on {r4['decided_frames']}, camera-blind {r4['blind_frames']} "
          f"(decided there by the prior: {r4['decided_by_prior_frames']}); enforced by the solver: {r4['enforce']} (crop check pending).", ""]
    L.append("| dance s | film | class | camera-blind share |")
    L.append("|---|---|---|---|")
    for a, b, c, bl in r4["runs"]:
        if b - a < 0.5:
            continue
        L.append(f"| {a:.2f}-{b:.2f} | {mmss(fm.main(a))}-{mmss(fm.main(b))} | {c} | {bl} |")
    L += ["", "## Spins (yaw track)", ""]
    for r, v in sj["roles"].items():
        L.append(f"- {r}: {len(v['candidates'])} spin candidates; defects: " +
                 (", ".join(f"{d['t0']:.2f}-{d['t1']:.2f} s (film {d['film']}, err {d['err_max_deg']} deg, {d['face_contra_max']} phones)"
                            for d in v["defects"]) or "none"))
    L += ["", "## Violation list in film time", "",
          "Every item is in `violations.json`; here items of one reason / relation / joint closer than 1 s are merged. `real` (a base joint "
          "farther than max(3 sigma, 6-8 cm) from its triangulated position on >= 5 of 10 frames) is listed only when it lasts >= 0.5 s or "
          "is >= 15 cm off; the rest of `real` is counted per joint below.", ""]

    def show(g):
        return g["reason"] != "real" or (g["t1"] - g["t0"] >= 0.5 or abs(g.get("worst", 0)) >= 0.15)

    def row(g):
        fx = f" (+ replay {', '.join(dict.fromkeys(mmss(x) for x in g['film_extra']))})" if g.get("film_extra") else ""
        return f"| {mmss(g.get('film0'))}-{mmss(g.get('film1'))}{fx} | {g['t0']:.2f}-{g['t1']:.2f} | {g['reason']} | {describe(g)} |"
    inf = [g for g in groups if in_spans(g.get("film0"), first) or in_spans(g.get("film1"), first)]
    rest = [g for g in groups if g not in inf]
    for title, sel in (("First: the owner's film spans " + ", ".join(f"{mmss(a)}-{mmss(b)}" for a, b in first), inf),
                       ("The rest of the take", rest)):
        L += [f"### {title}", ""]
        for r in REASON_ORDER:
            gs = [g for g in sel if g["reason"] == r and show(g)]
            if not gs:
                continue
            L += [f"**{r}** ({len(gs)})", "", "| film | dance s | reason | what |", "|---|---|---|---|"]
            L += [row(g) for g in gs]
            L.append("")
        reals = [g for g in sel if g["reason"] == "real"]
        if reals:
            cnt = {}
            for g in reals:
                k = f"{g['role']} {g['joint']}"
                c = cnt.setdefault(k, [0, 0.0])
                c[0] += 1
                c[1] += g["t1"] - g["t0"] + 1 / 30
            L += ["`real` per joint (groups, seconds): " + ", ".join(f"{k} {v[0]} / {v[1]:.1f} s" for k, v in sorted(cnt.items())), ""]
    L += ["## Files", "", f"- `{sd / 'violations.json'}` (every item), `{sd / 'relations.npz'}`, `{sd / 'yaw.npz'}`",
          *[f"- lanes: `{p}`" for p in pngs]]
    (sd / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    log("report", sd / "report.md")
    return rep


# ------------------------------------------------------------------------------------------------ the final report (Phase D): input | v4.3 | sides

def exact_series(work: Path, sd: Path, name: str, T: int, cfg: SidesCfg, fit_dir: Path | None = None) -> np.ndarray | None:
    """(T,) depth (m) of his right hand inside her torso / shoulders (upper arms less the allowance; gate.inside_series' exact parts)
    of a fit: v4/contacts/exact_<input|v43>.npz when present; for the sides fit the input's series with the adopted windows' own
    measurements (the solve pods' exact arrays) written in; an archived sides candidate (fit_dir's parent holds its gate.json and
    gate_exact.npz, e.g. sides/_fr1_backup/smplx_sides = v4.4a) the same way from its own archive"""
    from .. import zouk_contacts as zc
    parts = list(zc.EXACT_PARTS)

    def depth(ex):
        hp = np.asarray(ex["hand_pen_part"], float)
        d1 = hp[:, [parts.index(p) for p in ("torso", "shoulder_L", "shoulder_R")]].max(1)
        d2 = hp[:, [parts.index(p) for p in ("upper_arm_L", "upper_arm_R")]].max(1) - cfg.gate_g9_allow_m
        return np.maximum(d1, np.maximum(d2, 0.0))
    src = {"input": work / "v4" / "contacts" / "exact_input.npz", "v43": work / "v4" / "contacts" / "exact_v43.npz"}
    if name in src and src[name].exists():
        return depth(dict(np.load(src[name])))
    arch = Path(fit_dir).parent if fit_dir is not None else None
    if name != "sides" and arch is not None and (arch / "gate.json").exists() and (arch / "gate_exact.npz").exists():
        gdir, ef_arch = arch, arch / "gate_exact.npz"
    elif name == "sides":
        gdir, ef_arch = sd, None
    else:
        return None
    if not src["input"].exists() or not (gdir / "gate.json").exists():
        return None
    base = depth(dict(np.load(src["input"])))
    gj = json.loads((gdir / "gate.json").read_text(encoding="utf-8"))
    out = base.copy()
    ef = ef_arch or Path(gj.get("exact_file") or sd / "gate_exact.npz")
    if not ef.exists():
        return None
    z = dict(np.load(ef))
    for i, w in gj["windows"].items():
        if w.get("status") != "adopt" or f"w{i}_hand_pen_part" not in z:
            continue
        out[np.asarray(z[f"w{i}_ks"], int)] = depth({"hand_pen_part": z[f"w{i}_hand_pen_part"]})
    return out


def final_rows(take: str, cfg: SidesCfg, W: dict, fits: dict, dirs: dict | None = None) -> dict:
    """the DESIGN 7.3 rows the Phase A scorecard does not have, per fit: exact inside depth, his elbow's COCO error per phone at 83-100 s,
    the palm class on the decided R4 frames, the follower's twist-limit frames in the spin windows"""
    from .adapters import project
    from .decide import r4_class_base
    from .evidence import load_views
    from .gate import GateCtx, limit_masks
    ad = adapter_for(take, cfg)
    work, sd = ad.work, ad.work / "sides"
    t = ad.t
    T = len(t)
    rel = dict(np.load(sd / "relations.npz"))
    views = load_views(work, ad.cams)
    obs = ad.obs()
    hw = json.loads((sd / "relations.json").read_text(encoding="utf-8"))["relations"]["R4"].get("half_width_m", 0.14)
    out = {}
    ctx = None
    for nm, f in fits.items():
        row = {}
        ex = exact_series(work, sd, nm, T, cfg, (dirs or {}).get(nm))
        row["_exact"] = ex
        if ex is not None:
            for a, b in W.get("inside", []):
                m = (t >= a) & (t < b)
                d = ex[m]
                row[f"exact_inside_{a}-{b}"] = {"frames_gt3cm": int((d > 0.03).sum()), "share_gt3cm": round(float((d > 0.03).mean()), 3),
                                                "frames_gt5cm": int((d > 0.05).sum()), "p99_cm": round(float(np.percentile(d, 99)) * 100, 1)}
        m = np.flatnonzero((t >= 83.2) & (t < 100.0))
        J = np.asarray(f["lead"]["joints"], float)
        per = {}
        for c in ad.cams:
            z = views[c]
            o = obs[c]
            ok = (z["conf"][m, 0, 8].astype(float) >= 0.5) & (z["det"][m, 0].astype(float) >= cfg.det_conf) & o["ok"][m]
            uv, dep = project(o["K"][m], o["R"][m], o["t"][m], J[m][:, [19]])
            e = np.linalg.norm(uv[:, 0] - z["uv"][m, 0, 8].astype(float), axis=-1)
            e = e[ok & (dep[:, 0] > 0.1) & np.isfinite(e)]
            per[c] = round(float(np.median(e)), 1) if len(e) >= 10 else None
        row["elbow_coco_px_83_100"] = per
        M = geom.margins(geom.points_from_smplx(J), geom.points_from_smplx(np.asarray(f["follow"]["joints"], float)), half_width_m=hw)
        cb = r4_class_base(np.asarray(M["R4_fb"]["m"], float), np.asarray(M["R4_fl"]["m"], float), cfg)
        dec = rel["R4__cls"] != 0
        row["palm_class_on_decided_frames"] = {"frames": int(dec.sum()),
                                               "share_equal": round(float((cb[dec] == rel["R4__cls"][dec]).mean()), 3) if dec.any() else None}
        if ctx is None:
            ctx = GateCtx(take, cfg)
        for a, b in ((198.6, 201.0), (240.3, 246.6)):
            ks = np.flatnonzero((t >= a) & (t < b))
            lm = limit_masks(ctx, f["follow"], ks, "follow")
            tw = np.zeros(len(ks), bool)
            for k, v in lm.items():
                if k.startswith("twist_") or k in ("spine_total", "neck_head"):
                    tw |= v
            row[f"follow_twist_limit_share_{a}-{b}"] = round(float(tw.mean()), 3)
        out[nm] = row
    return out


R4_NAMES = {1: "front", 2: "flank", 3: "back", 0: "none"}


def r4_holds(rel: dict, t: np.ndarray, gap_s: float = 3.0, min_frames: int = 15) -> list:
    """the decided R4 runs (his palm on her torso) merged per class across gaps <= gap_s -> [{k0, k1, cls, frames (decided frames),
    blind_share (camera-blind share of them)}] with >= min_frames decided frames"""
    cls = np.asarray(rel["R4__cls"], int)
    blind = np.asarray(rel.get("R4__blind", np.zeros(len(cls), bool)), bool)
    hz = 1.0 / float(np.median(np.diff(t))) if len(t) > 1 else 30.0
    runs = []
    k = 0
    T = len(cls)
    while k < T:
        if cls[k] == 0:
            k += 1
            continue
        j = k
        while j + 1 < T and cls[j + 1] == cls[k]:
            j += 1
        if runs and runs[-1]["cls"] == cls[k] and (k - runs[-1]["k1"]) / hz <= gap_s:
            runs[-1]["k1"] = j
        else:
            runs.append({"k0": k, "k1": j, "cls": int(cls[k])})
        k = j + 1
    out = []
    for r in runs:
        sl = slice(r["k0"], r["k1"] + 1)
        dec = cls[sl] == r["cls"]
        if dec.sum() < min_frames:
            continue
        out.append({**r, "frames": int(dec.sum()), "blind_share": round(float(blind[sl][dec].mean()), 2)})
    return out


def changes_vs(take: str, cfg: SidesCfg, fits: dict, sc: dict, gj: dict, ref: str, new: str = "sides", exacts: dict | None = None) -> dict:
    """what a viewer sees change from the `ref` fit (the installed one) to the `new` one: his palm class on each decided R4 hold (the
    share of decided frames each fit matches, the new fit's class counts there, the windows over the hold and their status), the
    pass-throughs left, the hand inside her, the follower's changed frames per adopted window"""
    from .decide import r4_class_base
    from . import body as bd
    ad = adapter_for(take, cfg)
    sd = ad.work / "sides"
    t = ad.t
    fm = FilmMap.for_work(ad.work)
    rel = dict(np.load(sd / "relations.npz"))
    hw = json.loads((sd / "relations.json").read_text(encoding="utf-8"))["relations"]["R4"].get("half_width_m", 0.14)
    cb = {}
    for nm in fits:
        M = geom.margins(geom.points_from_smplx(np.asarray(fits[nm]["lead"]["joints"], float)),
                         geom.points_from_smplx(np.asarray(fits[nm]["follow"]["joints"], float)), half_width_m=hw)
        cb[nm] = r4_class_base(np.asarray(M["R4_fb"]["m"], float), np.asarray(M["R4_fl"]["m"], float), cfg)
    cls = np.asarray(rel["R4__cls"], int)
    wins = gj.get("windows", {})
    gli_ev = {}
    for nm in fits:
        G = geom.gli_chain(np.asarray(fits[nm]["lead"]["joints"], float)[:, [17, 19, 21]],
                           np.asarray(fits[nm]["follow"]["joints"], float)[:, [16, 18, 20]])
        gli_ev[nm] = [int(k) for k in geom.passthrough_events(G, merge=0)]
    holds = []
    for h in r4_holds(rel, t):
        sl = slice(h["k0"], h["k1"] + 1)
        dec = cls[sl] == h["cls"]
        row = {"dance": [round(float(t[h["k0"]]), 2), round(float(t[h["k1"]]), 2)],
               "film": [mmss(fm.main(float(t[h["k0"]]))), mmss(fm.main(float(t[h["k1"]])))], "decided": R4_NAMES[h["cls"]],
               "frames": h["frames"], "blind_share": h["blind_share"], "share_equal": {}, "classes": {}}
        for nm, c in cb.items():
            row["share_equal"][nm] = round(float((c[sl][dec] == h["cls"]).mean()), 2)
            row["classes"][nm] = {R4_NAMES[k]: int((c[sl][dec] == k).sum()) for k in (1, 2, 3)}
        row["windows"] = [f"{i} {w['status']}" for i, w in sorted(wins.items(), key=lambda x: int(x[0]))
                          if w.get("k0") is not None and w["r0"] <= h["k1"] and w["r1"] >= h["k0"]]
        # per fit on the hold's frames: the hand inside her (exact), the pass-throughs of his right arm through her left arm, and his
        # elbow / wrist height swings (>= 3 cm within 1 s) and 1-5 Hz wobble (p90 of the sliding 2-s RMS) - the owner's v4.3 complaint
        row["per_fit"] = {}
        for nm in fits:
            JL = np.asarray(fits[nm]["lead"]["joints"], float)
            ex = (exacts or {}).get(nm)
            hS = JL[sl, 17, 1]
            el, wr = JL[sl, 19, 1] - hS, JL[sl, 21, 1] - hS
            rb = band_rms_windows(el)
            pf = {"inside_gt3": int((ex[sl] > 0.03).sum()) if ex is not None else None,
                  "inside_gt5": int((ex[sl] > 0.05).sum()) if ex is not None else None,
                  "passthroughs": int(sum(h["k0"] <= k <= h["k1"] for k in gli_ev[nm])),
                  "elbow_swings": swings_within(el, 0.03, 30), "wrist_swings": swings_within(wr, 0.03, 30),
                  "elbow_wobble_p90_cm": round(float(np.nanpercentile(rb, 90)) * 100, 2) if np.isfinite(rb).any() else None}
            row["per_fit"][nm] = pf
        holds.append(row)
    out = {"ref": ref, "new": new, "holds": holds, "fits": list(fits)}
    if (sd / "review_note.md").exists():
        out["review_note"] = str(sd / "review_note.md")
    if new in sc:
        ev = sc[new].get("gli_event_times", [])
        out["passthroughs"] = {"times": ev, "film": [mmss(fm.main(x)) for x in ev]}
    inside = {}
    for nm in (ref, new, "input"):
        if nm in sc:
            rows = [(k, v) for k, v in sc[nm].items() if k.startswith("exact_inside_")]
            inside[nm] = {"frames_gt3cm": sum(v["frames_gt3cm"] for _, v in rows), "frames_gt5cm": sum(v["frames_gt5cm"] for _, v in rows),
                          "spans": [k.replace("exact_inside_", "") for k, _ in rows], "p99_max_cm": max((v["p99_cm"] for _, v in rows), default=None)}
    out["inside"] = inside
    # the follower's changed frames (new vs input), per adopted window and the prepare splices
    if new in fits:
        fi, fn = fits["input"]["follow"], fits[new]["follow"]
        ch = (np.abs(bd.pose55(fn) - bd.pose55(fi)).max((1, 2)) > 0) | (np.abs(np.asarray(fn["transl"]) - np.asarray(fi["transl"])).max(1) > 0)
        from .plan import load_plan
        pl = {w["id"]: w for w in load_plan(ad.work)["windows"]}
        rows = []
        cover = np.zeros(len(ch), bool)
        for i, w in sorted(wins.items(), key=lambda x: int(x[0])):
            if w.get("status") != "adopt":
                continue
            sl = slice(w["r0"], w["r1"] + 1)
            n = int(ch[sl].sum())
            cover[sl] = True
            if n:
                fr = (pl.get(int(i), {}).get("dancers", {}).get("follow") or {}).get("reasons", w.get("reasons", []))
                rows.append({"id": int(i), "film": f"{w.get('film0')}-{w.get('film1')}", "frames": n, "reasons": fr})
        out["follow_changed"] = {"frames": int(ch.sum()), "windows": rows, "outside_windows": int((ch & ~cover).sum()),
                                 "outside_windows_film": sorted({mmss(fm.main(float(t[k]))) for k in np.flatnonzero(ch & ~cover)[[0, -1]]})
                                 if (ch & ~cover).any() else []}
    return out


def changes_md(ch: dict) -> list:
    ref, new = ch["ref"], ch["new"]
    L = [f"## What you will see change compared with {ref} (the installed motion)", ""]
    if ch.get("review_note"):
        L += [f"The same in plain words, with the owner's choices: `{ch['review_note']}`.", ""]
    L += [
         f"His right palm on her torso, per decided R4 hold (decided = the cameras' class, or the D1 prior where no camera sees the hand: "
         f"`blind` share). `{ref}` / `{new}` = the share of the decided frames on which that fit's palm is in the decided class; "
         f"`{new} palm` = where the {new} palm is on those frames.", "",
         f"| film | dance s | decided | frames | blind | {ref} | {new} | {new} palm (front / flank / back) | windows |", "|---|---|---|---|---|---|---|---|---|"]
    for h in ch["holds"]:
        c = h["classes"].get(new, {})
        L.append(f"| {h['film'][0]}-{h['film'][1]} | {h['dance'][0]:.2f}-{h['dance'][1]:.2f} | {h['decided']} | {h['frames']} | "
                 f"{h['blind_share']} | {h['share_equal'].get(ref)} | {h['share_equal'].get(new)} | "
                 f"{c.get('front', 0)} / {c.get('flank', 0)} / {c.get('back', 0)} | {', '.join(h['windows']) or '-'} |")
    L.append("")
    names = [n for n in ch.get("fits", []) if any(n in h.get("per_fit", {}) for h in ch["holds"])]
    if names:
        L += ["Per decided hold and fit: palm on the decided class (share of the decided frames) ; hand > 3 / > 5 cm inside her (exact "
              "mesh frames of the hold) ; pass-throughs of his right arm through her left arm ; his elbow / wrist height swings >= 3 cm "
              "within 1 s ; elbow 1-5 Hz wobble p90 (cm).", "",
              "| film | decided | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
        for h in ch["holds"]:
            cells = []
            for n in names:
                pf = h["per_fit"].get(n, {})
                cells.append(f"{h['share_equal'].get(n, 0):.0%} ; {pf.get('inside_gt3')} / {pf.get('inside_gt5')} ; "
                             f"{pf.get('passthroughs')} ; {pf.get('elbow_swings')} / {pf.get('wrist_swings')} ; {pf.get('elbow_wobble_p90_cm')}")
            L.append(f"| {h['film'][0]}-{h['film'][1]} | {h['decided']} | " + " | ".join(cells) + " |")
        L.append("")
    # the same in plain words: every hold where the new palm matches the decided class on clearly fewer frames than the installed one
    blind_kept = []
    for h in ch["holds"]:
        sr, sn = h["share_equal"].get(ref), h["share_equal"].get(new)
        if sr is None or sn is None or sn > sr - 0.2:
            continue
        maj = {nm: max(h["classes"][nm], key=h["classes"][nm].get) for nm in (ref, new) if nm in h["classes"]}
        adopted = [w.split()[0] for w in h["windows"] if w.endswith("adopt")]
        why = (f"window{'s' if len(adopted) > 1 else ''} {', '.join(adopted)} adopted there, the rest reverted" if adopted else
               "every window there was reverted by the gates, so the detected (input) palm is kept")
        blind = h["blind_share"] >= 0.5
        if blind:
            blind_kept.append(f"{h['film'][0]}-{h['film'][1]}")
        move = (f"his palm moves from her {maj.get(ref)} to her {maj.get(new)}" if maj.get(ref) != maj.get(new) else
                f"his palm is mostly on her {maj.get(new)} in both, but on the decided class on fewer frames")
        L.append(f"- film {h['film'][0]}-{h['film'][1]}: decided `{h['decided']}`"
                 f"{' by the D1 prior (no camera sees his hand)' if blind else ' by the cameras'}; {move} ({ref}: {sr:.0%} of the "
                 f"decided frames on the decided class, {new}: {sn:.0%}): {why}.")
    if blind_kept:
        L.append(f"- D1 = (b) decided the camera-blind hold(s) {', '.join(blind_kept)} from the prior, but no window there passed the gates: "
                 f"the decision is not enforced and the palm is the detected one.")
    L.append("")
    if "passthroughs" in ch:
        L += [f"Pass-throughs of his right arm through her left arm left in the {new} fit (owner span): {len(ch['passthroughs']['times'])} "
              f"at film {', '.join(ch['passthroughs']['film']) or '-'}.", ""]
    ins = ch.get("inside", {})
    if ins:
        L.append("His hand more than 3 cm inside her (exact mesh, spans " + ", ".join(next(iter(ins.values()))["spans"]) + " dance s): " +
                 "; ".join(f"{nm} {v['frames_gt3cm']} frames (> 5 cm: {v['frames_gt5cm']}, p99 up to {v['p99_max_cm']} cm)" for nm, v in ins.items()) + ".")
        L.append("")
    fc = ch.get("follow_changed")
    if fc:
        L.append(f"The follower changes on {fc['frames']} frames: " + "; ".join(
            f"window {r['id']} (film {r['film']}, {', '.join(r['reasons'])}): {r['frames']}" for r in fc["windows"]) +
            (f"; outside the adopted windows (the prepare splice): {fc['outside_windows']} frames (film {' - '.join(fc['outside_windows_film'])})"
             if fc["outside_windows"] else "") + ".")
        L.append("")
    return L


def windows_list(sd: Path) -> list:
    gj = json.loads((sd / "gate.json").read_text(encoding="utf-8"))
    rows = []
    for i, w in sorted(gj["windows"].items(), key=lambda x: x[1]["t0"]):
        rows.append({"id": int(i), "t0": w["t0"], "t1": w["t1"], "film0": w.get("film0"), "film1": w.get("film1"), "status": w["status"],
                     "reasons": w.get("reasons", []), "dancers": sorted(w.get("dancers", {})), "why": w.get("why"),
                     "attempt": w.get("attempt"),
                     "targets_missed": sorted(set(w.get("targets_missed", [])) | set((w.get("merged_gate") or {}).get("targets_missed", [])))})
    return rows


def final(take: str, cfg: SidesCfg | None = None, compare: dict | None = None) -> dict:
    """Phase D report: the 7.3 scorecard input | v4.3 | sides and the film-time list of adopted / reverted / flagged windows
    -> report.json["final"] and the top of report.md (the Phase A report below it)"""
    cfg = cfg or load_cfg(take)
    ad = adapter_for(take, cfg)
    work, sd = ad.work, ad.work / "sides"
    W = json.loads((sd / "report_windows.json").read_text(encoding="utf-8")) if (sd / "report_windows.json").exists() else {}
    comp = dict(compare or {})
    if (work / "smplx_sides" / "lead.npz").exists():
        comp.setdefault("sides", str(work / "smplx_sides"))
    sc = scorecard(take, cfg, W, comp)
    fits = {"input": ad.base()}
    for nm, d in comp.items():
        fits[nm] = {}
        for r in cfg.roles:
            with np.load(Path(d) / f"{r}.npz", allow_pickle=True) as z:
                fits[nm][r] = {k: z[k] for k in z.files}
    extra = final_rows(take, cfg, W, fits, {nm: Path(d) for nm, d in comp.items()})
    exacts = {nm: extra[nm].pop("_exact", None) for nm in extra}
    for nm in sc:
        sc[nm].update(extra.get(nm, {}))
    gj = json.loads((sd / "gate.json").read_text(encoding="utf-8")) if (sd / "gate.json").exists() else {}
    wl = windows_list(sd) if gj else []
    ref = "v43" if "v43" in fits else next((n for n in fits if n not in ("input", "sides")), None)
    chg = changes_vs(take, cfg, fits, sc, gj, ref, exacts=exacts) if (ref and "sides" in fits and gj) else None
    rep = {"scorecard": sc, "windows": wl, "summary": gj.get("summary"), "take_gate": gj.get("take_gate"), "changes": chg,
           "calibration": {k: v.get("ok") for k, v in json.loads((sd / "gate_calibration.json").read_text(encoding="utf-8")).get("cases", {}).items()}
           if (sd / "gate_calibration.json").exists() else None}
    old = json.loads((sd / "report.json").read_text(encoding="utf-8")) if (sd / "report.json").exists() else {}
    old["final"] = rep
    write_json(sd / "report.json", old)
    names = list(sc)
    s = gj.get("summary") or {}
    L = [f"# sides report: {take}", "", "## Phase D: the sides fit against the input and v4.3 (DESIGN 7.3)", "",
         f"Windows {s.get('windows')}: adopted {s.get('adopted')}, reverted {s.get('reverted')}, missing {s.get('missing')}, flagged "
         f"{s.get('flagged')}; frames changed {s.get('frames_changed')} (share {s.get('share_changed')}). Whole-take gate: "
         f"{'PASS' if (gj.get('take_gate') or {}).get('ok') else 'FAIL'}. Gate calibration: {rep['calibration']}.",
         "", "| Metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    keys = []
    for v in sc.values():
        for k in v:
            if k not in keys and not k.endswith("_times"):
                keys.append(k)
    for k in keys:
        L.append(f"| {k} | " + " | ".join(json.dumps(sc[n].get(k)) for n in names) + " |")
    if chg:
        L += [""] + changes_md(chg)
    L += ["", "### Windows in film time (flagged = a revert, or an adopted window that misses a DESIGN 7.1 target)", "",
          "| film | dance s | id | status | dancers | reasons | note |", "|---|---|---|---|---|---|---|"]
    for w in wl:
        note = w.get("why") or ""
        if w["targets_missed"]:
            note = (note + "; " if note else "") + "targets missed: " + ",".join(w["targets_missed"])
        if w.get("attempt") and w["attempt"] != "planned":
            note = (note + "; " if note else "") + f"attempt {w['attempt']}"
        L.append(f"| {w['film0']}-{w['film1']} | {w['t0']:.2f}-{w['t1']:.2f} | {w['id']} | {w['status']} | {', '.join(w['dancers'])} | "
                 f"{', '.join(w['reasons'])} | {note[:240]} |")
    L.append("")
    md = sd / "report.md"
    prev = md.read_text(encoding="utf-8") if md.exists() else ""
    marker = "<!-- phase A -->"
    if marker in prev:
        prev = prev[prev.index(marker):]
    else:
        prev = marker + "\n" + prev
    md.write_text("\n".join(L) + "\n---\n\n" + prev, encoding="utf-8")
    return rep
