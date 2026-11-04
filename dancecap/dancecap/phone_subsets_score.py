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
reference's stance frames, feet reprojection, horizontal stance-ankle shift against the input fit, and held-out phone
01 feet px for the two- and one-phone inputs. Pass (doc E2): a contact arm beats rf2 on slip for every input, feet px at
most +0.3 over arm 0, stance ankles moved 10 mm at most, and slip at least 30 % under arm 0 on the two- and one-phone
fits.
"""

import json
from pathlib import Path

import numpy as np

from .mono4d.candidates import score_arrays
from .mono4d.holdout import match_times, summary
from .phone_subsets import ARMS, E2, HELD, MONO, REF, WINDOWS, WORK

ROLES = ("lead", "follow")
OUT = WORK / REF / "e1e2"


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
    """Share of the stance frame pairs the contact term's speed gate kept (summed pair weight / stance pairs), per
    role, both feet pooled; None without the contact term."""
    rep = json.loads((out_dir / "pose_refine.json").read_text())
    res = {}
    for r in ROLES:
        q, st = rep[r]["refine"], rep[r]["filter"]["stance_frac"]
        if not q.get("w_contact"):
            res[r] = None
            continue
        n = len(np.load(out_dir / f"{r}.npz")["transl"]) - 1
        res[r] = round(sum(q["contact_pair_weight"].values()) / max(sum(st.values()) * n, 1.0), 2)
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


def score_e2() -> dict:
    inputs = {
        "eight phones": dict(take=REF, input=WORK / REF / "smplx", arms={k: (f"smplx_{k}", k) for k in E2}, held=None),
        "two phones (E1 b)": dict(take=f"{REF}_e1b", input=WORK / f"{REF}_e1b" / "smplx",
                                  arms={"e2_0": ("smplx_rf2_noslip", "smplx_rf2_noslip"),
                                        **{k: (k, k) for k in E2 if k != "e2_0"}},
                                  held=(WORK / f"{REF}_e1b" / "mamma" / "ma_cap" / f"{REF}_e1b_eval" / "gt",
                                        WORK / f"{REF}_e1b" / "mamma" / "ma_2d" / f"{REF}_e1b_eval", 0)),
    }
    c8 = WORK / REF / "mamma"
    for mono, (a, _) in MONO.items():
        inputs[f"one phone {'44-48' if a == 416 else '58-62'}"] = dict(
            take=mono, input=WORK / mono / "mamma" / "fits" / "3db" / "smplx", arms={k: (k, k) for k in E2},
            held=(c8 / "ma_cap" / f"{REF}_c8" / "gt", c8 / "ma_2d" / f"{REF}_c8", a))
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
            if spec["held"]:
                gt, lm, f0 = spec["held"]
                row["held_out_01"] = held_out(out, gt, lm, frame0=f0)
            rows[arm] = row
        res[name] = rows
    res["pass"] = e2_pass(res)
    return res


def e2_pass(res: dict) -> dict:
    out = {}
    for b in ("e2_b05", "e2_b2"):
        checks, ok = {}, True
        for name, rows in res.items():
            if not isinstance(rows, dict) or any(_gone(rows.get(k, {})) for k in ("e2_0", "e2_a", b)):
                ok = False
                checks[name] = "missing"
                continue
            q0, qa, qb = (rows[k]["quality"] for k in ("e2_0", "e2_a", b))
            c = {}
            for r in ROLES:
                c[r] = {"beats_rf2": qb[r]["slip_mm_s"] < qa[r]["slip_mm_s"],
                        "feet_px_ok": qb[r]["feet_px"] - q0[r]["feet_px"] <= 0.3,
                        "ankles_ok": (rows[b]["ankle_shift_mm"][r] or 0) <= 10.0,
                        "slip_cut": round(1 - qb[r]["slip_mm_s"] / q0[r]["slip_mm_s"], 3)}
                need_cut = name != "eight phones"
                ok &= c[r]["beats_rf2"] and c[r]["feet_px_ok"] and c[r]["ankles_ok"] and (
                    not need_cut or c[r]["slip_cut"] >= 0.30)
            checks[name] = c
        out[b] = {"pass": ok, "checks": checks}
    return out


def sheet(e1: dict, e2: dict) -> str:
    L = ["# Tests E1 and E2 (docs/BODY_TRACKING_RESEARCH_2026-11.md)", ""]
    d = e1.get("decision", {})
    L += ["## Three numbers", ""]
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
    L += ["", "## E2: foot terms (slip mm/s on the reference's stance frames; feet px; ankle shift mm; held-out 01 feet px; "
          "share of stance frame pairs the contact term's speed gate kept)", "",
          "| input | arm | role | slip | feet px | ankle shift | held-out 01 feet | gate kept |", "|---|---|---|---|---|---|---|---|"]
    for name, rows in e2.items():
        if name == "pass":
            continue
        for arm, row in rows.items():
            if row.get("missing"):
                L.append(f"| {name} | {arm} | - | missing | | | | |")
                continue
            for r in ROLES:
                ho = row.get("held_out_01", {}).get(r, {}).get("feet_px", "-")
                kept = row.get("gate_kept", {}).get(r)
                L.append(f"| {name} | {arm} | {r} | {row['quality'][r]['slip_mm_s']} | {row['quality'][r]['feet_px']} | "
                         f"{row['ankle_shift_mm'][r]} | {ho} | {'-' if kept is None else f'{kept:.0%}'} |")
    return "\n".join(L) + "\n"


def score() -> dict:
    e1, e2 = score_e1(), score_e2()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "score.json").write_text(json.dumps({"e1": e1, "e2": e2}, indent=1, default=str))
    (OUT / "sheet.md").write_text(sheet(e1, e2), encoding="utf-8")
    return {"e1_decision": e1.get("decision"), "e2_pass": {k: v["pass"] for k, v in e2.get("pass", {}).items()},
            "sheet": str(OUT / "sheet.md")}
