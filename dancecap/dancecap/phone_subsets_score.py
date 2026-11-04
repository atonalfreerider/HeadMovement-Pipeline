"""Scores tests E1 and E2 of docs/BODY_TRACKING_RESEARCH_2026-11.md (built and launched by dancecap.phone_subsets).

    python -m dancecap.phone_subsets score   ->   work/larissa_kadu_38/e1e2/{score.json, sheet.md}

E1, per arm and dancer: the arm's refined fit (smplx_rf2_noslip) against the eight-phone fit refined the same way.
- PA22 and pelvis error (mono4d.candidates.score_arrays) and couple-vector error (|(follow - lead pelvis) - the same in
  the reference|), over the whole take, 44-48 s and 58-62 s, and the whole take again on the frames where every arm
  sees the dancer in two phones;
- stance slip on the reference's stance frames (pose_quality);
- held-out phone 01 reprojection, all points and feet, of the raw and the refined fit.
Decision (research doc E1): a three-phone arm at least 30 % better than the good pair b on the follow's 44-48 s PA22 or
on the couple vector -> film three phones; under 10 % -> two well-separated phones are enough.

E2, per input fit (eight phones, E1 arm b's two phones, the two one-phone windows) and refine arm: stance slip on the
reference's stance frames, feet reprojection, horizontal stance-ankle shift against the input fit and against arm 0's
output, and held-out phone 01 feet px for the two- and one-phone inputs. Pass (doc E2): a contact arm beats rf2 on slip
for every input, feet px at most +0.3 over arm 0, the term moved the stance ankles 10 mm at most (against arm 0's
output, the same filter and refine without the term: the filter alone moves the hidden one-phone follower's ankles
14 mm), and slip at least 30 % under arm 0 on the two- and one-phone fits. E2b's contact arms (the gate on a 1.5 Hz
low-passed foot speed: the loaded point over the whole take, or the foot lock's path; and stance alone) are scored the
same way, against E2's arm 0 and rf2, plus the gate itself per input (gate_cover: share of the fit's stance pairs kept,
weight on the reference's stance and swing pairs, for every gate).
"""

import json
from pathlib import Path

import numpy as np

from .mono4d.candidates import score_arrays
from .mono4d.holdout import match_times, summary
from .phone_subsets import ARMS, E2, E2B, HELD, MONO, REF, WINDOWS, WORK

ROLES = ("lead", "follow")
OUT = WORK / REF / "e1e2"
CONTACT = ("e2_b05", "e2_b2", *E2B)
GATE = {"e2_b05": "e2", "e2_b2": "e2", "e2b_b05": "loaded", "e2b_b2": "loaded", "e2b_s05": "stance",
        "e2b_l05": "lock", "e2b_l2": "lock"}  # gate_cover's names
GATES = {"e2": "6 Hz loaded point (E2)", "loaded": "1.5 Hz loaded point, whole take", "lock": "1.5 Hz foot-lock path",
         "stance": "stance alone"}
LABEL = {"e2_b05": "6 Hz gate, weight 0.5", "e2_b2": "6 Hz gate, weight 2",
         "e2b_l05": "foot-lock gate, weight 0.5", "e2b_l2": "foot-lock gate, weight 2",
         "e2b_b05": "loaded-point gate, weight 0.5", "e2b_b2": "loaded-point gate, weight 2",
         "e2b_s05": "stance alone, weight 0.5"}
E2B_SHOW = ("e2b_l05", "e2b_l2", "e2b_b05", "e2b_b2", "e2b_s05")  # the doc's E2b (foot lock) first


def _fit(d: Path) -> dict:
    return {r: np.load(d / f"{r}.npz", allow_pickle=True) for r in ROLES}


def _med(x) -> float | None:
    s = summary(np.asarray(x, float))
    return None if s["median"] is None else round(s["median"], 1)


def _windows(t: np.ndarray) -> dict:
    return {k: (w if w else (float(t[0]), float(t[-1]))) for k, w in WINDOWS.items()}


def couple_error(arm: dict, ref: dict, window: tuple[float, float], keep_t: np.ndarray | None = None) -> float | None:
    """Median |(follow - lead pelvis) of the arm - the same of the reference| (mm) inside the window."""
    va = arm["follow"]["joints"][:, 0] - arm["lead"]["joints"][:, 0]
    vr = ref["follow"]["joints"][:, 0] - ref["lead"]["joints"][:, 0]
    ir, ia = match_times(ref["lead"]["t"], arm["lead"]["t"])
    tr = np.asarray(ref["lead"]["t"], float)[ir]
    ok = (tr >= window[0]) & (tr <= window[1])
    if keep_t is not None:
        ok &= np.isin(np.round(tr, 4), np.round(keep_t, 4))
    return _med(1e3 * np.linalg.norm(va[ia[ok]] - vr[ir[ok]], axis=-1))


def two_view_times() -> dict:
    """Grid times where every E1 arm sees the dancer in at least two of its phones (10+ visible landmarks)."""
    m = WORK / REF / "mamma"
    t = np.asarray(json.loads((m / "prepare.json").read_text())["grid"]["t"], float)
    seen = {}
    for c in sorted({c for cams in ARMS.values() for c in cams}):
        v = np.load(m / "ma_2d" / f"{REF}_c8" / f"cam{c}.npz")["visibilities"]
        seen[c] = (v > 0.5).sum(-1) >= 10  # (T, 2)
    keep = {}
    for p, role in enumerate(ROLES):
        ok = np.ones(len(t), bool)
        for cams in ARMS.values():
            ok &= np.sum([seen[c][:, p] for c in cams], 0) >= 2
        keep[role] = t[ok]
    return keep


def held_out(smplx_dir: Path, gt_dir: Path, lm_dir: Path, cam: str = HELD, frame0: int = 0) -> dict:
    """Median px error per role of a fit in a phone it never saw: all 512 points and the 88 foot points."""
    from .mamma_dyn import PIDS, _camera_errors, _forward, _load_verts512, part_index
    from .mamma_dyn import ROLES as MROLES
    D = _load_verts512()
    parts = part_index()
    feet = np.concatenate([parts["left_foot"], parts["right_foot"]])
    out = {}
    for p, pid in enumerate(PIDS):
        r = np.load(smplx_dir / f"{MROLES[pid]}.npz")
        pose = np.concatenate([r["global_orient"], r["body_pose"], r["jaw_pose"], r["leye_pose"], r["reye_pose"],
                               r["left_hand_pose"], r["right_hand_pose"]], 1)
        verts, _ = _forward(pose, r["transl"], r["betas"], True)
        e, ok = _camera_errors(np.matmul(D, verts), f"cam{cam}", gt_dir, lm_dir, p, slice(frame0, frame0 + len(pose)))
        allpx = [np.median(e[f][ok[f]]) for f in range(len(e)) if ok[f].sum() >= 3]
        ftpx = [np.median(e[f][feet][ok[f][feet]]) for f in range(len(e)) if ok[f][feet].sum() >= 3]
        out[MROLES[pid]] = {"all_px": _med(allpx), "feet_px": _med(ftpx)}
    return out


def quality(take: str, tag: str) -> dict:
    q = json.loads((WORK / take / "quality" / f"quality_{tag}.json").read_text())
    return {r: {"slip_mm_s": q["slip"][r]["both_feet"]["contact_point_speed_ref_stance_mm_s"]["median"],
                "feet_px": q["reproj_feet_px"]["all_cams_per_frame_median"][r]["median"]} for r in ROLES}


def gate_kept(out_dir: Path) -> dict:
    """Share of the stance frame pairs (both frames in pose_refine's stance mask, saved in the output) the contact
    term's gate kept (summed pair weight / stance pairs), per role, both feet pooled; None without the contact term."""
    rep = json.loads((out_dir / "pose_refine.json").read_text())
    res = {}
    for r in ROLES:
        q = rep[r]["refine"]
        if not q.get("w_contact"):
            res[r] = None
            continue
        s = np.load(out_dir / f"{r}.npz")["stance"].astype(bool)
        res[r] = round(sum(q["contact_pair_weight"].values()) / max(int((s[1:] & s[:-1]).sum()), 1), 2)
    return res


def ankle_shift(take: str, out_dir: Path, in_dir: Path, tag: str) -> dict:
    """Median horizontal shift (mm) of the ankles against the input fit on the reference's stance frames."""
    fr = np.load(WORK / take / "quality" / f"quality_{tag}_frames.npz")
    a, b = _fit(out_dir), _fit(in_dir)
    res = {}
    for r in ROLES:
        d = []
        for j, side in ((7, "l"), (8, "r")):
            m = fr[f"rst_{side}_{r}"].astype(bool)
            xa, xb = a[r]["joints"][:, j][:, [0, 2]], b[r]["joints"][:, j][:, [0, 2]]
            d.append(1e3 * np.linalg.norm((xa - xb)[m[: len(xa)]], axis=-1))
        res[r] = _med(np.concatenate(d))
    return res


def score_e1() -> dict:
    ref = _fit(WORK / REF / "smplx_rf2_noslip")
    keep = two_view_times()
    res = {"reference": f"{REF}/smplx_rf2_noslip (8 phones, refined like the arms)", "arms": {}}
    for x, cams in ARMS.items():
        w = WORK / f"{REF}_e1{x}"
        if not (w / "smplx_rf2_noslip" / "follow.npz").exists():
            res["arms"][x] = {"cams": list(cams), "missing": True}
            continue
        fit = json.loads((w / "mamma" / "fits" / f"e1{x}" / "fit.json").read_text())
        arm = _fit(w / "smplx_rf2_noslip")
        row = {"cams": list(cams), "ma_3d_s": fit["ma_3d"].get("seconds"), "sdf_active": fit["ma_3d"].get("sdf_active"),
               "held_out_raw": {r: fit["residuals"][HELD][r]["median_px"] for r in ROLES}}
        for r in ROLES:
            rr = {}
            for name, win in _windows(ref[r]["t"]).items():
                s = score_arrays(arm[r]["t"], arm[r]["joints"], "world", ref[r]["t"], ref[r]["joints"], win)
                rr[name] = {"pa22": _med([s["pa22_mm"]["median"]]), "pelvis": _med([s["pelvis_mm"]["median"]]),
                            "frames": s["frames"]}
            t2 = keep[r]
            ok = np.isin(np.round(arm[r]["t"], 4), np.round(t2, 4))
            s = score_arrays(arm[r]["t"][ok], arm[r]["joints"][ok], "world", ref[r]["t"], ref[r]["joints"],
                             (float(t2[0]), float(t2[-1])))
            rr["whole_2view"] = {"pa22": _med([s["pa22_mm"]["median"]]), "pelvis": _med([s["pelvis_mm"]["median"]]),
                                 "frames": s["frames"]}
            row[r] = rr
        row["couple_mm"] = {name: couple_error(arm, ref, win) for name, win in _windows(ref["lead"]["t"]).items()}
        both = np.intersect1d(np.round(keep["lead"], 4), np.round(keep["follow"], 4))
        row["couple_mm"]["whole_2view"] = couple_error(arm, ref, (float(both[0]), float(both[-1])), both)
        row["quality"] = quality(f"{REF}_e1{x}", "smplx_rf2_noslip")
        row["held_out_refined"] = held_out(w / "smplx_rf2_noslip", w / "mamma" / "ma_cap" / f"{REF}_e1{x}_eval" / "gt",
                                           w / "mamma" / "ma_2d" / f"{REF}_e1{x}_eval")
        res["arms"][x] = row
    res["decision"] = decide(res["arms"])
    return res


def _gone(row: dict) -> bool:
    return not row or bool(row.get("missing"))


def decide(arms: dict) -> dict:
    b = arms.get("b", {})
    if _gone(b):
        return {"verdict": "arm b missing"}
    pa_b, cv_b = b["follow"]["44-48"]["pa22"], b["couple_mm"]["whole_2view"]
    gains = {}
    for x in ("c", "d"):
        a = arms.get(x, {})
        if _gone(a):
            continue
        gains[x] = {"follow_pa22_44_48": round((pa_b - a["follow"]["44-48"]["pa22"]) / pa_b, 3),
                    "couple_whole": round((cv_b - a["couple_mm"]["whole_2view"]) / cv_b, 3)}
    best = max((max(g.values()), x) for x, g in gains.items()) if gains else (None, None)
    if best[0] is None:
        verdict = "three-phone arms missing"
    elif best[0] >= 0.30:
        verdict = f"film three phones: arm {best[1]} ({','.join(ARMS[best[1]])}) is {best[0]:.0%} better than pair b"
    elif best[0] < 0.10:
        verdict = f"two well-separated phones are enough: the best third phone gains {best[0]:.0%} (< 10 %)"
    else:
        verdict = f"in between: the best third phone (arm {best[1]}) gains {best[0]:.0%} (10-30 %)"
    a = arms.get("a", {})
    sep = None if _gone(a) else round(a["follow"]["whole_2view"]["pa22"] / b["follow"]["whole_2view"]["pa22"], 2)
    return {"gains_vs_b": gains, "verdict": verdict, "narrow_pair_a_over_b_follow_pa22": sep}


def e2_inputs() -> dict:
    """Per E2 input: its take, the input fit, {arm: (output folder, pose_quality tag)}, the held-out phone 01 grid."""
    arms = {**E2, **E2B}
    inputs = {
        "eight phones": dict(take=REF, input=WORK / REF / "smplx", arms={k: (f"smplx_{k}", k) for k in arms}, held=None),
        "two phones (E1 b)": dict(take=f"{REF}_e1b", input=WORK / f"{REF}_e1b" / "smplx",
                                  arms={"e2_0": ("smplx_rf2_noslip", "smplx_rf2_noslip"),
                                        **{k: (k, k) for k in arms if k != "e2_0"}},
                                  held=(WORK / f"{REF}_e1b" / "mamma" / "ma_cap" / f"{REF}_e1b_eval" / "gt",
                                        WORK / f"{REF}_e1b" / "mamma" / "ma_2d" / f"{REF}_e1b_eval", 0)),
    }
    c8 = WORK / REF / "mamma"
    for mono, (a, _) in MONO.items():
        inputs[f"one phone {'44-48' if a == 416 else '58-62'}"] = dict(
            take=mono, input=WORK / mono / "mamma" / "fits" / "3db" / "smplx", arms={k: (k, k) for k in arms},
            held=(c8 / "ma_cap" / f"{REF}_c8" / "gt", c8 / "ma_2d" / f"{REF}_c8", a))
    return inputs


def gate_cover(spec: dict) -> dict:
    """The contact term's pair weights, recomputed from the input fit with pose_refine's own filter and stance (the
    pod computes the same), for E2's gate (the loaded point's 6 Hz speed), E2b's (a 1.5 Hz low-pass: of the loaded
    point over the whole take, or of the foot lock's path) and none (stance alone), per role, both feet pooled:
    'kept' = gate_kept's share of the fit's stance pairs; 'ref_stance' /
    'ref_swing' = the mean pair weight where the reference (pose_quality's stance rule on the eight-phone fit) has the
    foot in stance / in swing in both frames; on the reference's stance pairs, with arm 0's per-pair slip (pose_quality
    slipv): 'slip_weighted' = arm 0's slip averaged with the gate's weights (mm/s), 'slip_share' = the share of arm 0's
    summed slip those weights cover. 'pod': per returned contact arm, whether these weight sums match its
    pose_refine.json (0.15 per foot). 'held': per returned contact arm, the arm's median slip over arm 0's on the
    reference's stance pairs its gate weights >= 0.5 ('hi'), in (0, 0.5) ('mid') and 0 ('free'); None under 5 pairs."""
    from . import pose_refine as pr
    fr = np.load(WORK / spec["take"] / "quality" / f"quality_{spec['arms']['e2_0'][1]}_frames.npz")
    res, sums, wts, ref = {**{g: {} for g in GATES}, "pod": {}, "held": {}}, {}, {}, {}
    for r in ROLES:
        z = dict(np.load(spec["input"] / f"{r}.npz", allow_pickle=True))
        z.pop("smoothing", None)
        z.pop("stance", None)
        aux: dict = {}
        zf, frep, st = pr.filter_and_contacts(z, "filter", 5.0, 8.0, aux=aux)
        t = np.asarray(zf["t"], float)
        n = len(t) - 1
        rst = {f: fr[f"rst_{f[0]}_{r}"].astype(bool)[: n + 1] for f in st}
        rs = np.concatenate([rst[f][1:] & rst[f][:-1] for f in st])
        sw = np.concatenate([~rst[f][1:] & ~rst[f][:-1] for f in st])
        pairs = sum(int((st[f][1:] & st[f][:-1]).sum()) for f in st)
        v0 = np.concatenate([fr[f"slipv_{f[0]}_{r}"][:n] for f in st])[rs]
        ref[r] = (rs, v0)
        for g, sp in (("e2", aux["speed"]), ("loaded", {f: pr.path_speed(aux["loaded"][f], t, 1.5) for f in st}),
                      ("lock", {f: pr.lock_speed(aux["centre"][f], st[f], t, 1.5) for f in st}),
                      ("stance", {f: np.zeros(n + 1) for f in st})):
            w = {f: pr.contact_weights(st[f], sp[f], (0.05, 0.25)) for f in st}
            sums[g, r] = {f: float(w[f].sum()) for f in st}
            wa = np.concatenate([w[f] for f in st])
            wts[g, r] = wa[rs]
            res[g][r] = {"kept": round(float(wa.sum()) / max(pairs, 1), 2),
                         "ref_stance": round(float(wa[rs].mean()), 2) if rs.any() else None,
                         "ref_swing": round(float(wa[sw].mean()), 2) if sw.any() else None,
                         "slip_weighted": round(1e3 * float((wa[rs] * v0).sum() / max(wa[rs].sum(), 1e-9)), 1),
                         "slip_share": round(float((wa[rs] * v0).sum() / max(v0.sum(), 1e-9)), 2)}
    for arm, (d, tag) in spec["arms"].items():
        rp = WORK / spec["take"] / d / "pose_refine.json"
        if arm in CONTACT and rp.exists():
            q = json.loads(rp.read_text())
            res["pod"][arm] = all(abs(q[r]["refine"]["contact_pair_weight"][f] - sums[GATE[arm], r][f]) <= 0.15
                                  for r in ROLES for f in sums[GATE[arm], r])
            fa = np.load(WORK / spec["take"] / "quality" / f"quality_{tag}_frames.npz")
            res["held"][arm] = {}
            for r in ROLES:
                rs, v0 = ref[r]
                va = np.concatenate([fa[f"slipv_{f[0]}_{r}"][: len(rs) // 2] for f in ("left", "right")])[rs]
                w = wts[GATE[arm], r]
                res["held"][arm][r] = {k: (round(float(np.median(va[m]) / np.median(v0[m])), 2) if m.sum() >= 5 else None)
                                       for k, m in (("hi", w >= 0.5), ("mid", (w > 0) & (w < 0.5)), ("free", w == 0))}
    return res


def score_gates() -> dict:
    return {name: gate_cover(spec) for name, spec in e2_inputs().items()}


def score_e2() -> dict:
    inputs = e2_inputs()
    res = {}
    for name, spec in inputs.items():
        rows = {}
        for arm, (d, tag) in spec["arms"].items():
            out = WORK / spec["take"] / d
            if not (out / "follow.npz").exists():
                rows[arm] = {"missing": True}
                continue
            row = {"quality": quality(spec["take"], tag), "ankle_shift_mm": ankle_shift(spec["take"], out, spec["input"],
                                                                                         tag), "gate_kept": gate_kept(out)}
            base = WORK / spec["take"] / spec["arms"]["e2_0"][0]
            if arm != "e2_0" and (base / "follow.npz").exists():  # what the term moved: against arm 0's output
                row["ankle_vs_arm0_mm"] = ankle_shift(spec["take"], out, base, tag)
            if spec["held"]:
                gt, lm, f0 = spec["held"]
                row["held_out_01"] = held_out(out, gt, lm, frame0=f0)
            rows[arm] = row
        res[name] = rows
    res["pass"] = e2_pass(res)
    res["gates"] = score_gates()
    return res


def e2_pass(res: dict) -> dict:
    out = {}
    for b in CONTACT:
        checks, ok = {}, True
        for name, rows in res.items():
            if name in ("pass", "gates"):
                continue
            if not isinstance(rows, dict) or any(_gone(rows.get(k, {})) for k in ("e2_0", "e2_a", b)):
                ok = False
                checks[name] = "missing"
                continue
            q0, qa, qb = (rows[k]["quality"] for k in ("e2_0", "e2_a", b))
            c = {}
            for r in ROLES:
                c[r] = {"beats_rf2": qb[r]["slip_mm_s"] < qa[r]["slip_mm_s"],
                        "feet_px_ok": qb[r]["feet_px"] - q0[r]["feet_px"] <= 0.3,
                        "ankles_ok": (rows[b]["ankle_vs_arm0_mm"][r] or 0) <= 10.0,
                        "slip_cut": round(1 - qb[r]["slip_mm_s"] / q0[r]["slip_mm_s"], 3),
                        "feet_px_change": round(qb[r]["feet_px"] - q0[r]["feet_px"], 2),
                        "ankle_mm": rows[b]["ankle_vs_arm0_mm"][r]}
                need_cut = name != "eight phones"
                ok &= c[r]["beats_rf2"] and c[r]["feet_px_ok"] and c[r]["ankles_ok"] and (
                    not need_cut or c[r]["slip_cut"] >= 0.30)
            checks[name] = c
        out[b] = {"pass": ok, "checks": checks}
    return out


def _pct(x) -> str:
    return "-" if x is None else f"{x:.0%}"


def e2b_numbers(e2: dict) -> list[str]:
    """Three numbers for E2b: where the term acts (per gate), the slip it removes and what it costs (per arm)."""
    p, g = e2.get("pass", {}), e2.get("gates", {})
    on = lambda gate, k: "; ".join(f"{n} {_pct(x[gate]['lead'][k])}/{_pct(x[gate]['follow'][k])}" for n, x in g.items())
    L = ["## E2b: three numbers (the contact gate on a 1.5 Hz low-passed foot speed; and stance alone)", "",
         "1. Where the term acts: mean pair weight on the reference's stance pairs, then on its swing pairs (lead/follow)."]
    L += [f"   - {GATES[k]}: stance {on(k, 'ref_stance')}. Swing {on(k, 'ref_swing')}." for k in GATES]
    for b in E2B_SHOW:
        ch = {n: c for n, c in p.get(b, {}).get("checks", {}).items() if isinstance(c, dict)}
        cuts = "; ".join(f"{n} {-c['lead']['slip_cut']:+.0%}/{-c['follow']['slip_cut']:+.0%}" for n, c in ch.items())
        L.append(f"2. {LABEL[b]}: slip change vs arm 0 (lead/follow): {cuts or 'missing'}.")
    for b in E2B_SHOW:
        ch = {n: c for n, c in p.get(b, {}).get("checks", {}).items() if isinstance(c, dict)}
        gone = [n for n, c in p.get(b, {}).get("checks", {}).items() if not isinstance(c, dict)]
        if not ch:
            L.append(f"3. {LABEL[b]}: missing -> fail.")
            continue
        px = max(c[r]["feet_px_change"] for c in ch.values() for r in ROLES)
        mm = max((c[r]["ankle_mm"] or 0) for c in ch.values() for r in ROLES)
        bad = sorted({f"{n} {r} {k}" for n, c in ch.items() for r in ROLES for k in ("beats_rf2", "feet_px_ok", "ankles_ok")
                      if not c[r][k]} | {f"{n} {r} cut" for n, c in ch.items() if n != "eight phones" for r in ROLES
                                         if c[r]["slip_cut"] < 0.30} | {f"{n} missing" for n in gone})
        L.append(f"3. {LABEL[b]}: feet px at most {px:+.2f} (bar +0.3), the term moved the stance ankles at most {mm} mm "
                 f"(bar 10) -> {'PASS' if p[b]['pass'] else 'fail: ' + ', '.join(bad)}.")
    return L + [""]


def sheet(e1: dict, e2: dict) -> str:
    L = ["# Tests E1, E2 and E2b (docs/BODY_TRACKING_RESEARCH_2026-11.md)", ""]
    L += e2b_numbers(e2)
    d = e1.get("decision", {})
    L += ["## E1 and E2: three numbers", ""]
    g = d.get("gains_vs_b", {})
    if g:
        best = max(g, key=lambda x: max(g[x].values()))
        L.append(f"1. Best third phone (arm {best}, phones {','.join(ARMS[best])}) vs the good pair b: error cut "
                 f"{g[best]['follow_pa22_44_48']:.0%} on the follow's 44-48 s PA22, {g[best]['couple_whole']:.0%} on the "
                 f"couple vector. {d['verdict']}.")
    if d.get("narrow_pair_a_over_b_follow_pa22") is not None:
        L.append(f"2. Narrow pair a ({','.join(ARMS['a'])}, 24 deg) over good pair b: follow PA22 x"
                 f"{d['narrow_pair_a_over_b_follow_pa22']} (whole take, two-view frames).")
    p = e2.get("pass", {})
    for b in ("e2_b05", "e2_b2"):
        cuts = [f"{n} {-c['lead']['slip_cut']:+.0%} / {-c['follow']['slip_cut']:+.0%}"
                for n, c in p.get(b, {}).get("checks", {}).items() if isinstance(c, dict)]
        L.append(f"3. Contact term {b[3:]}: slip change vs arm 0 (lead / follow) - {'; '.join(cuts)} -> "
                 f"{'PASS' if p.get(b, {}).get('pass') else 'fail'}.")
    L += ["", "## E1: phone subsets against the eight-phone fit (median mm; px)", "",
          "| arm | phones | role | PA22 whole | PA22 2-view | PA22 44-48 | PA22 58-62 | pelvis whole | couple 2-view | "
          "slip mm/s | held-out 01 all/feet px (raw -> refined) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for x, a in e1["arms"].items():
        if a.get("missing"):
            L.append(f"| {x} | {','.join(a['cams'])} | - | missing | | | | | | | |")
            continue
        for r in ROLES:
            ho = a["held_out_refined"][r]
            L.append(f"| {x} | {','.join(a['cams'])} | {r} | {a[r]['whole']['pa22']} | {a[r]['whole_2view']['pa22']} | "
                     f"{a[r]['44-48']['pa22']} | {a[r]['58-62']['pa22']} | {a[r]['whole']['pelvis']} | "
                     f"{a['couple_mm']['whole_2view']} | {a['quality'][r]['slip_mm_s']} | "
                     f"{a['held_out_raw'][r]:.1f} -> {ho['all_px']} / {ho['feet_px']} |")
    L += ["", "## E2: foot terms (slip mm/s on the reference's stance frames; feet px; stance-ankle shift mm against the "
          "input and against arm 0; held-out 01 feet px; share of stance frame pairs the contact term's gate kept)", "",
          "| input | arm | role | slip | feet px | ankle shift | vs arm 0 | held-out 01 feet | gate kept |",
          "|---|---|---|---|---|---|---|---|---|"]
    for name, rows in e2.items():
        if name in ("pass", "gates"):
            continue
        for arm, row in rows.items():
            if row.get("missing"):
                L.append(f"| {name} | {arm} | - | missing | | | | | |")
                continue
            for r in ROLES:
                ho = row.get("held_out_01", {}).get(r, {}).get("feet_px", "-")
                kept = row.get("gate_kept", {}).get(r)
                va = row.get("ankle_vs_arm0_mm", {}).get(r, "-")
                L.append(f"| {name} | {arm} | {r} | {row['quality'][r]['slip_mm_s']} | {row['quality'][r]['feet_px']} | "
                         f"{row['ankle_shift_mm'][r]} | {va} | {ho} | {'-' if kept is None else f'{kept:.0%}'} |")
    L += ["", "## Contact gate per input, recomputed from the input fit: share of the fit's stance pairs kept / mean pair "
          "weight on the reference's stance pairs / on its swing pairs", "",
          "| input | role | " + " | ".join(GATES.values()) + " | matches pod |", "|---|---|" + "---|" * len(GATES) + "---|"]
    for name, x in e2.get("gates", {}).items():
        pod = ", ".join(f"{a} {'yes' if ok else 'NO'}" for a, ok in x["pod"].items()) or "-"
        for r in ROLES:
            cells = " | ".join(f"{_pct(x[g][r]['kept'])} / {_pct(x[g][r]['ref_stance'])} / {_pct(x[g][r]['ref_swing'])}"
                               for g in GATES)
            L.append(f"| {name} | {r} | {cells} | {pod} |")
    return "\n".join(L) + "\n"


def score() -> dict:
    e1, e2 = score_e1(), score_e2()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "score.json").write_text(json.dumps({"e1": e1, "e2": e2}, indent=1, default=str))
    (OUT / "sheet.md").write_text(sheet(e1, e2), encoding="utf-8")
    return {"e1_decision": e1.get("decision"), "e2_pass": {k: v["pass"] for k, v in e2.get("pass", {}).items()},
            "sheet": str(OUT / "sheet.md")}
