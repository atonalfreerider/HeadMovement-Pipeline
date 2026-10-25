"""The solver windows of the "sides" step (DESIGN 3.1): where the base breaks a decided side, passes a limb through the partner, puts a
hand / forearm > 3 cm inside her, sits far from the real (triangulated) joint, swings, winds or wraps.

    plan_windows(take) -> work/<take>/sides/plan.json

Reasons (decide.violations items + the prepare step's splices) -> per dancer and frame the joints that may move:

| reason | fires (decide.violations)                                   | moved                                                   |
|--------|-------------------------------------------------------------|---------------------------------------------------------|
| side   | a decided arm relation violated (R4, R6-R9)                 | his arm chain of the relation                           |
| thru   | a GLI pass-through in a held run (decided sign nearby)      | his arm chain of the pairing                            |
| inside | his palm / forearm > 3 cm in her torso on >= 6 frames       | his right arm chain                                     |
| real   | a limb joint > max(3 sigma, tau) from the triangulated one   | the joint's two parent rotations (that dancer)          |
| swing  | his elbow / wrist 1-5 Hz height energy beyond the real one  | his arm chain                                           |
| spin   | a yaw defect, a splice entry / exit                         | global orient, spine, neck, head, legs, root (that one) |
| limit  | a swing-twist gate exceeded                                 | that chain                                              |

Not solved (listed as `flagged`): side items of the facing / leg / head relations (R1, R2, R5, R11-R13), pass-throughs outside a held
run (no decided sign to hold), the follower's swings (she moves only in spin / limit windows and where her own joint triggers `real`),
`real` items of the torso joints (hips, shoulders: moving them moves the whole body) and short `real` items (< plan_real_min_s).

Windows: segments padded by plan_pad_s, merged across reasons and dancers where they overlap or are < plan_merge_gap_s apart, cores split
to <= plan_core_max_s with plan_overlap_s overlaps, cosine ramps of plan_ramp_s on each side (clipped at the take ends). Arm windows move
his collar (10 deg), shoulder / elbow / wrist (25 deg) and spine3 (5 deg); spin windows the global orient (45 deg), spine (15 deg each),
neck / head and legs (20 deg) and the root (3 cm). Fingers, jaw, eyes, hand poses and betas never move.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import hmm
from .adapters import adapter_for
from .config import SidesCfg, load_cfg
from .evidence import write_json
from .filmtime import FilmMap, mmss

# SMPL-X body joints
SPINE3 = 9
ARM = {"R": (14, 17, 19, 21), "L": (13, 16, 18, 20)}           # collar, shoulder, elbow, wrist
SPIN_JOINTS = (0, 3, 6, 9, 12, 15, 1, 2, 4, 5, 7, 8, 10, 11)    # global orient, spine1-3, neck, head, hips, knees, ankles, feet
LEG = {"L": (1, 4, 7, 10), "R": (2, 5, 8, 11)}
REAL_PARENTS = {"elL": (13, 16), "elR": (14, 17), "wrL": (16, 18), "wrR": (17, 19), "knL": (1,), "knR": (2,), "anL": (1, 4),
                "anR": (2, 5)}
ARM_RELS = {"R4": "R", "R6": "R", "R7": "R", "R8": "R", "R9_RL": "R", "R9_RR": "R", "R9_LR": "L", "R9_LL": "L"}
LIMIT_JOINTS = {"spine": (3, 6, 9), "neck_head": (12, 15), "hip_L": (1,), "hip_R": (2,), "knee_L": (1, 4), "knee_R": (2, 5),
                "ankle_L": (4, 7), "ankle_R": (5, 8)}


def trust_of(j: int, kind: str, cfg: SidesCfg) -> float:
    """trust radius (deg) of joint j in an arm or spin window (DESIGN 1.1 "Trust region")"""
    if j == 0:
        return cfg.trust_global_deg
    if j in (13, 14):
        return cfg.trust_collar_deg
    if j == SPINE3 and kind == "arm":
        return cfg.trust_spine3_arm_deg
    if j in (3, 6, 9):
        return cfg.trust_spine_deg
    if j in (12, 15):
        return cfg.trust_neck_deg
    if j in (1, 2, 4, 5, 7, 8, 10, 11):
        return cfg.trust_leg_deg
    return cfg.trust_arm_deg


def segments(viol: list, splices: list, rel: dict, cfg: SidesCfg) -> tuple[list, list]:
    """violation items -> ([{role, k0, k1, reason, joints, kind, detail}], [flagged items])"""
    segs, flagged = [], []
    for it in viol:
        r = it["reason"]
        k0, k1 = int(it["k0"]), int(it["k1"])
        if r == "side":
            rid = it.get("rel")
            if rid in ARM_RELS:
                segs.append({"role": "lead", "k0": k0, "k1": k1, "reason": "side", "joints": ARM[ARM_RELS[rid]] + (SPINE3,), "kind": "arm",
                             "detail": rid})
            else:
                flagged.append({**it, "why": "facing / leg / head relation: listed, not solved"})
        elif r == "thru":
            rid = it.get("rel")
            if it.get("held") and rid in ARM_RELS:
                segs.append({"role": "lead", "k0": k0, "k1": k1, "reason": "thru", "joints": ARM[ARM_RELS[rid]] + (SPINE3,), "kind": "arm",
                             "detail": rid})
            else:
                flagged.append({**it, "why": "pass-through outside a held run (no decided sign)"})
        elif r == "inside":
            segs.append({"role": "lead", "k0": k0, "k1": k1, "reason": "inside", "joints": ARM["R"] + (SPINE3,), "kind": "arm"})
        elif r == "swing":
            if it.get("role") == "lead":
                side = it["joint"][-1]
                segs.append({"role": "lead", "k0": k0, "k1": k1, "reason": "swing", "joints": ARM[side] + (SPINE3,), "kind": "arm",
                             "detail": it["joint"]})
            else:
                flagged.append({**it, "why": "the follower's swings are not solved (she moves in spin / limit / real windows only)"})
        elif r == "real":
            j = it.get("joint")
            if j in REAL_PARENTS and (it["frames"] / cfg.hz >= cfg.plan_real_min_s or it.get("worst", 0) >= cfg.plan_real_big_m):
                kind = "arm" if j[:2] in ("el", "wr") else "leg"
                joints = REAL_PARENTS[j] + ((SPINE3,) if kind == "arm" else ())
                segs.append({"role": it["role"], "k0": k0, "k1": k1, "reason": "real", "joints": joints, "kind": kind, "detail": j,
                             "median": it.get("median"), "worst": it.get("worst")})
            else:
                flagged.append({**it, "why": "torso joint or a short item: E_tri inside other windows only"})
        elif r == "spin":
            segs.append({"role": it["role"], "k0": k0, "k1": k1, "reason": "spin", "joints": SPIN_JOINTS, "kind": "spin", "root": True})
        elif r == "limit":
            j = it.get("joint")
            joints = LIMIT_JOINTS.get(j)
            if joints is None:
                flagged.append({**it, "why": "unknown limit chain"})
                continue
            segs.append({"role": it["role"], "k0": k0, "k1": k1, "reason": "limit", "joints": joints,
                         "kind": "spin" if j in ("spine", "neck_head") else "leg", "detail": j})
        elif r == "wrap":
            flagged.append({**it, "why": "wraps are removed in closed form by prepare (unwrap)"})
    # splice polish: a spin window over every splice (entry ramp, the alternative's frames, the exit slerp) for the dancer whose yaw
    # defect started it
    for s in splices:
        segs.append({"role": s["defect_role"], "k0": int(s["k0"]), "k1": int(s["k1"]) - 1, "reason": "spin", "joints": SPIN_JOINTS,
                     "kind": "spin", "root": True, "detail": f"splice {s['alt']} polish"})
    return segs, flagged


def _merge(items: list, gap: int) -> list:
    """merge [a, b] spans (inclusive) closer than gap frames -> [(a, b, [indices])]"""
    out = []
    for i in sorted(range(len(items)), key=lambda i: items[i][0]):
        a, b = items[i]
        if out and a - out[-1][1] <= gap:
            out[-1] = (out[-1][0], max(out[-1][1], b), out[-1][2] + [i])
        else:
            out.append((a, b, [i]))
    return out


def split_core(a: int, b: int, max_len: int, overlap: int) -> list:
    """cores <= max_len frames covering [a, b] with `overlap`-frame overlaps, equal lengths"""
    n = b - a + 1
    if n <= max_len:
        return [(a, b)]
    k = int(np.ceil((n - overlap) / (max_len - overlap)))
    L = int(np.ceil((n + (k - 1) * overlap) / k))
    out = []
    s = a
    for i in range(k):
        e = min(b, s + L - 1)
        if i == k - 1:
            e = b
        out.append((s, e))
        s = e - overlap + 1
    return out


def plan_windows(take: str, cfg: SidesCfg | None = None, write: bool = True) -> dict:
    t0 = time.time()
    cfg = cfg or load_cfg(take)
    ad = adapter_for(take, cfg)
    work = ad.work
    sd = work / "sides"
    T, hz = ad.T, cfg.hz
    viol = json.loads((sd / "violations.json").read_text(encoding="utf-8"))["items"]
    prep = json.loads((sd / "cpu" / "prepare.json").read_text(encoding="utf-8")) if (sd / "cpu" / "prepare.json").exists() else {}
    rel = dict(np.load(sd / "relations.npz"))
    fm = FilmMap.for_work(work)
    segs, flagged = segments(viol, prep.get("splices", []), rel, cfg)
    pad = int(round(cfg.plan_pad_s * hz))
    gap = int(round(cfg.plan_merge_gap_s * hz))
    owner = [tuple(w) for w in cfg.plan_owner_windows]

    def in_owner(a, b):
        return any(a / hz < w1 and b / hz > w0 for w0, w1 in owner)
    spans_all = [(max(0, s["k0"] - pad), min(T - 1, s["k1"] + pad)) for s in segs]
    # scope (plan_real_scope "owner"): `real` items open / extend windows only inside the owner spans; elsewhere they only add their
    # joints to a window another reason opened where they overlap it (clipped to it), and are listed as deferred where none does
    opener = [i for i, s in enumerate(segs) if cfg.plan_real_scope != "owner" or s["reason"] != "real" or in_owner(*spans_all[i])]
    rider = [i for i in range(len(segs)) if i not in set(opener)]
    groups = [(a, b, [opener[j] for j in idx]) for a, b, idx in _merge([spans_all[i] for i in opener], gap)]
    deferred = []
    for i in rider:
        a, b = spans_all[i]
        hit = [g for g in groups if g[0] <= b and g[1] >= a]
        for g in hit:
            g[2].append(i)
        if not hit:
            s = segs[i]
            deferred.append({"role": s["role"], "joint": s.get("detail"), "t0": round(s["k0"] / hz, 3), "t1": round(s["k1"] / hz, 3),
                             "film0": mmss(fm.main(s["k0"] / hz)), "median": s.get("median"), "worst": s.get("worst")})
    ramp = int(round(cfg.plan_ramp_s * hz))
    max_core = int(round(cfg.plan_core_max_s * hz))
    ov = int(round(cfg.plan_overlap_s * hz))
    wins = []
    for a, b, idx in groups:
        for c0, c1 in split_core(a, b, max_core, ov):
            inside = [segs[i] for i in idx if segs[i]["k1"] + pad >= c0 and segs[i]["k0"] - pad <= c1]
            dancers = {}
            for s in inside:
                d = dancers.setdefault(s["role"], {"joints": set(), "reasons": set(), "kinds": set(), "root": False, "details": set(),
                                                   "real_items": []})
                if s["reason"] == "real":
                    d["real_items"].append({"joint": s["detail"], "k0": int(max(s["k0"], c0)), "k1": int(min(s["k1"], c1)),
                                            "median": s.get("median"), "worst": s.get("worst")})
                d["joints"].update(s["joints"])
                d["reasons"].add(s["reason"])
                d["kinds"].add(s["kind"])
                d["root"] |= bool(s.get("root"))
                if s.get("detail"):
                    d["details"].add(str(s["detail"]))
            rl, rr = min(ramp, c0), min(ramp, T - 1 - c1)
            t_a, t_b = c0 / hz, c1 / hz
            tier = 1 if any(t_a < w1 and t_b > w0 for w0, w1 in owner) else 2
            win = {"id": len(wins), "k0": int(c0), "k1": int(c1), "r0": int(c0 - rl), "r1": int(c1 + rr), "ramp_in": int(rl), "ramp_out": int(rr),
                   "t0": round(t_a, 3), "t1": round(t_b, 3), "film0": mmss(fm.main(t_a)), "film1": mmss(fm.main(t_b)), "tier": tier,
                   "reasons": sorted({s["reason"] for s in inside}), "dancers": {}}
            for role, d in dancers.items():
                kind = "spin" if "spin" in d["kinds"] else ("arm" if d["kinds"] <= {"arm"} else "mixed")
                js = sorted(d["joints"])
                win["dancers"][role] = {"joints": js, "trust_deg": {str(j): trust_of(j, "arm" if kind == "arm" else "spin", cfg) for j in js},
                                        "root_m": cfg.trust_root_m if d["root"] else 0.0, "reasons": sorted(d["reasons"]), "kind": kind,
                                        "details": sorted(d["details"]), "real_items": [x for x in d["real_items"] if x["k1"] >= x["k0"]]}
            wins.append(win)
    # parts: balance the frame-work (window length x moved dancers) over the solve pods, windows of one merged group stay on one part
    plan = {"schema": "dancecap.sides.plan/1", "take": take, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "T": int(T), "hz": hz,
            "cfg": {k: getattr(cfg, k) for k in ("plan_pad_s", "plan_merge_gap_s", "plan_core_max_s", "plan_overlap_s", "plan_ramp_s",
                                                  "plan_real_min_s", "plan_real_big_m", "plan_real_scope", "plan_owner_windows")},
            "windows": wins, "flagged": flagged, "deferred_real": deferred,
            "summary": {"windows": len(wins), "tier1": sum(w["tier"] == 1 for w in wins),
                        "frames_in_windows": int(np.unique(np.concatenate([np.arange(w["r0"], w["r1"] + 1) for w in wins])).size) if wins else 0,
                        "reasons": {r: sum(r in w["reasons"] for w in wins) for r in ("side", "thru", "inside", "real", "swing", "spin", "limit")},
                        "follower_windows": sum("follow" in w["dancers"] for w in wins), "flagged": len(flagged),
                        "deferred_real_items": len(deferred)}}
    plan["summary"]["share_of_take"] = round(plan["summary"]["frames_in_windows"] / T, 3)
    plan["seconds"] = round(time.time() - t0, 2)
    if write:
        write_json(sd / "plan.json", plan)
    return plan


def load_plan(work: Path) -> dict:
    return json.loads((Path(work) / "sides" / "plan.json").read_text(encoding="utf-8"))


def parts_of(plan: dict, n: int) -> list[list[int]]:
    """window ids split into n parts of similar work (frames x dancers), greedy longest-first"""
    load = [0.0] * n
    out = [[] for _ in range(n)]
    for w in sorted(plan["windows"], key=lambda w: -(w["r1"] - w["r0"] + 1) * max(1, len(w["dancers"]))):
        i = int(np.argmin(load))
        out[i].append(w["id"])
        load[i] += (w["r1"] - w["r0"] + 1) * max(1, len(w["dancers"]))
    return [sorted(x) for x in out]
