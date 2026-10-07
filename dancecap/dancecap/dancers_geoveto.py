"""Geometric identity veto: drop dancer masks that sit on the WRONG PERSON, as seen by the other cameras.

    python -m dancecap.dancers_geoveto larissa_kadu_38            # measure -> dancers/geo_veto.json + review image
    python -m dancecap.dancers_geoveto larissa_kadu_38 --apply    # + blank the vetoed masks / ma_2d landmarks
    python -m dancecap.dancers_geoveto larissa_kadu_38 --check    # re-measure only -> dancers/geo_veto_check.json
                                                                  #   (geo_veto.json, the applied record, is kept)
review/reid_veto.png shades the vetoes APPLIED so far (roles.json geo_veto.history, solid) and outlines the spans the
automatic rule would veto now (candidates, hatched; not applied unless --apply).

The dancers stage checks lead vs follow with colour evidence, so it cannot see a mask that follows a spectator (a
sitting spectator in the foreground of cam07 at 30-38 s, a standing one at the left edge of cam05 near 52 s on
larissa_kadu_38). The cameras do: for every camera c, 30 Hz grid instant and role, the role's COCO keypoints (the YOLO
detection matched to the role's mask box, cameras.world.dancer_keypoints) are compared with the same role's keypoints
triangulated from the OTHER main cameras (robust DLT, sync-corrected times) and reprojected into c. A role whose
keypoints are explained by NEITHER 3D dancer - median residual to its own and to the partner's projection both above
max(VETO_MIN_PX, VETO_FACTOR x the camera's median) - on at least MIN_RUN consecutive instants (gaps of up to GAP
instants bridged, runs dilated by DILATE instants) is vetoed in the native frames nearest those instants.

--apply (backups first, so it can be undone): dancers/<cam>/{masks,body,hair} PNGs of the vetoed role -> empty
(originals in dancers/geo_veto_backup/<cam>/...), roles.json frames[i][role] -> box None + geo_veto true,
mamma/ma_2d_native/<cam>.npz visibilities of that person -> 0 (original kept as <cam>.pre_veto.npz). Re-run
`mamma prepare <take> --skip-2d` afterwards (the vetoed views then carry no landmarks into MAMMA) and
`python -m dancecap review <take>`.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

from . import mamma_dyn as md
from .cameras.report import project_cv
from .cameras.world import dancer_keypoints, triangulate_robust

ROLES = ("lead", "follow")
KPS = (0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)  # nose + limbs
VETO_MIN_PX, VETO_FACTOR = 60.0, 5.0
MIN_RUN, GAP, DILATE = 3, 6, 3


def _tracks(work: Path, cams: list[str]) -> dict:
    offs = md.read_sync_offsets(work)
    tr = {}
    for c in cams:
        z = np.load(work / "cameras" / f"{c}.npz")
        dist = np.asarray(z["dist"], float)
        tr[c] = {"t": md.apply_offset(np.asarray(z["t"], float), offs.get(c)), "K": np.asarray(z["K"], float),
                 "k1": dist if dist.ndim == 1 else dist[:, 0], "R": np.asarray(z["R"], float),
                 "tvec": np.asarray(z["tvec"], float), "ok": np.asarray(z["ok"], bool),
                 "attached": bool(z["attached"]) if "attached" in z.files else False}
    return tr


def role_costs(work: Path, cams: list[str]) -> dict:
    """{cam: {"t": (G,), "native": (G,) frame index or -1, "lead": (G,), "follow": (G,), "lead_other": (G,),
    "follow_other": (G,)}}: median px residual of each role's keypoints vs its own 3D dancer and vs the other one."""
    tr = _tracks(work, cams)
    kp = {c: dancer_keypoints(work, c, len(tr[c]["t"])) for c in cams}
    main = [c for c in cams if not tr[c]["attached"]]
    t0 = max(tr[c]["t"][0] for c in cams)
    t1 = min(tr[c]["t"][-1] for c in cams)
    grid = np.arange(t0, t1, 1 / 30)
    near = {}
    for c in cams:
        j = np.clip(np.searchsorted(tr[c]["t"], grid), 1, len(tr[c]["t"]) - 1)
        j = np.where(np.abs(tr[c]["t"][j - 1] - grid) < np.abs(tr[c]["t"][j] - grid), j - 1, j)
        near[c] = np.where((np.abs(tr[c]["t"][j] - grid) <= 0.02) & tr[c]["ok"][j], j, -1)

    def tri(role, g, excl):
        X = np.full((17, 3), np.nan)
        for k in KPS:
            v = [(c, near[c][g]) for c in main if c != excl and near[c][g] >= 0
                 and kp[c][role][near[c][g], k, 2] >= 0.5]
            if len(v) < 2:
                continue
            xy = np.array([kp[c][role][i, k, :2] for c, i in v], float)
            Xk, _, _ = triangulate_robust(xy, np.array([tr[c]["K"][i] for c, i in v]),
                                          np.array([tr[c]["k1"][i] for c, i in v]),
                                          np.array([tr[c]["R"][i] for c, i in v]),
                                          np.array([tr[c]["tvec"][i] for c, i in v]), max_px=15.0)
            if Xk is not None:
                X[k] = Xk
        return X

    out = {}
    cache: dict = {}
    for c in cams:
        res = {"t": grid, "native": near[c], **{r: np.full(len(grid), np.nan) for r in ROLES},
               **{f"{r}_other": np.full(len(grid), np.nan) for r in ROLES}}
        excl = c if c in main else None
        for g in range(len(grid)):
            i = near[c][g]
            if i < 0:
                continue
            for r in ROLES:
                obs = kp[c][r][i]
                v = obs[:, 2] >= 0.5
                v[[1, 2, 3, 4]] = False
                if v.sum() < 4:
                    continue
                for r3, name in ((r, r), (ROLES[1 - ROLES.index(r)], f"{r}_other")):
                    key = (r3, g, excl)
                    if key not in cache:
                        cache[key] = tri(r3, g, excl)
                    X = cache[key]
                    ok = v & np.isfinite(X).all(1)
                    if ok.sum() < 4:
                        continue
                    uv, vis = project_cv(X[ok], tr[c]["K"][i], tr[c]["R"][i], tr[c]["tvec"][i], tr[c]["k1"][i])
                    e = np.linalg.norm(uv - obs[ok, :2], axis=1)[vis]
                    if len(e) >= 4:
                        res[name][g] = float(np.median(e))
        out[c] = res
    return out


def _runs(flag: np.ndarray) -> list[tuple[int, int]]:
    out, s = [], None
    for k, f in enumerate(np.r_[flag, False]):
        if f and s is None:
            s = k
        elif not f and s is not None:
            out.append((s, k))
            s = None
    return out


def decide(costs: dict) -> dict:
    """Per camera and role: vetoed grid runs (bridged, min length, dilated) + the thresholds used."""
    dec = {}
    for c, r in costs.items():
        dec[c] = {}
        for role in ROLES:
            x = r[role]
            med = float(np.nanmedian(x)) if np.isfinite(x).any() else float("nan")
            thr = max(VETO_MIN_PX, VETO_FACTOR * med) if np.isfinite(med) else VETO_MIN_PX
            # wrong person = explained by NEITHER dancer (in an embrace the role's detection may be the partner's
            # while its mask is right: that is not vetoed)
            xo = r.get(f"{role}_other", np.full(len(x), np.inf))
            bad = (np.nan_to_num(x, nan=0.0) > thr) & (np.nan_to_num(xo, nan=np.inf) > thr)
            # bridge short gaps (unobserved / momentarily plausible instants inside a wrong-person stretch)
            runs = _runs(bad)
            merged: list[list[int]] = []
            for a, b in runs:
                if merged and a - merged[-1][1] <= GAP:
                    merged[-1][1] = b
                else:
                    merged.append([a, b])
            keep = [(a, b) for a, b in merged if int(bad[a:b].sum()) >= MIN_RUN]
            veto = np.zeros(len(x), bool)
            for a, b in keep:
                veto[max(0, a - DILATE):min(len(x), b + DILATE)] = True
            dec[c][role] = {"median_px": round(med, 2), "threshold_px": round(thr, 1),
                            "over_threshold_instants": int(bad.sum()), "isolated_spikes": int(bad.sum()) - int(
                                sum(bad[a:b].sum() for a, b in keep)),
                            "runs": [{"t0": round(float(r["t"][a]), 3), "t1": round(float(r["t"][b - 1]), 3),
                                      "instants": int(b - a), "median_px": round(float(np.nanmedian(x[a:b])), 1)}
                                     for a, b in keep],
                            "veto_grid": veto}
    return dec


def native_veto(work: Path, cam: str, grid_t: np.ndarray, veto: np.ndarray) -> np.ndarray:
    """Native frame indices of a camera whose (sync-corrected) time is nearest to a vetoed grid instant."""
    t = _tracks(work, [cam])[cam]["t"]
    if not veto.any():
        return np.zeros(0, int)
    j = np.clip(np.searchsorted(grid_t, t), 1, len(grid_t) - 1)
    j = np.where(np.abs(grid_t[j - 1] - t) < np.abs(grid_t[j] - t), j - 1, j)
    return np.flatnonzero(veto[j] & (np.abs(grid_t[j] - t) <= 1 / 30))


def apply(work: Path, plan: dict, source: str = "geometric") -> dict:
    import cv2
    roles_p = work / "dancers" / "roles.json"
    roles = json.loads(roles_p.read_text())
    bak = work / "dancers" / "geo_veto_backup"
    pid = {"lead": 1, "follow": 2}
    done = {}
    for c, per in plan.items():
        nat_p = work / "mamma" / "ma_2d_native" / f"{c}.npz"
        nat = dict(np.load(nat_p)) if nat_p.exists() else None
        changed = False
        for role, frames in per.items():
            if not frames:
                continue
            for i in frames:
                subs = ("masks", "body") + (("hair",) if role == "follow" else ())
                for sub in subs:
                    f = work / "dancers" / c / sub / f"mask_{i:04d}_{pid[role]:02d}.png"
                    if not f.exists():
                        continue
                    b = bak / c / sub / f.name
                    if not b.exists():
                        b.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(f, b)
                    m = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
                    cv2.imwrite(str(f), np.zeros_like(m))
                fr = roles["cams"][c]["frames"][i]
                fr[role] = {**(fr.get(role) or {}), "box": None, "conf": 0.0, "geo_veto": True}
                if nat is not None and i < len(nat["visibilities"]):
                    nat["visibilities"][i, pid[role] - 1] = 0.0
                    changed = True
            done.setdefault(c, {})[role] = len(frames)
        if changed:
            pre = nat_p.with_name(f"{c}.pre_veto.npz")
            if not pre.exists():
                shutil.copy2(nat_p, pre)
            np.savez(nat_p, **nat)
    prev = roles.get("geo_veto") or {}
    hist = prev.get("history", []) + [{"applied": time.strftime("%Y-%m-%dT%H:%M:%S"), "frames": done, "source": source}]
    roles["geo_veto"] = {"history": hist,
                         "note": "masks blanked where the role's keypoints contradict the other cameras (wrong person), "
                                 "or by a person's review (source manual)"}
    roles_p.write_text(json.dumps(roles))
    return done


def applied_frames(work: Path) -> dict:
    """{cam: {role: sorted native frames}} of every veto applied so far (roles.json geo_veto.history)."""
    roles = json.loads((work / "dancers" / "roles.json").read_text())
    out: dict = {}
    for c, cr in roles["cams"].items():  # the frames themselves carry the flag (history stores counts only)
        for i, fr in enumerate(cr["frames"]):
            for r in ROLES:
                if isinstance((fr or {}).get(r), dict) and fr[r].get("geo_veto"):
                    out.setdefault(c, {}).setdefault(r, set()).add(i)
    return {c: {r: sorted(v) for r, v in per.items()} for c, per in out.items()}


def review(work: Path, costs: dict, dec: dict, out: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    cams = list(costs)
    done = applied_frames(work)
    fig, axes = plt.subplots(len(cams), 1, figsize=(15, 1.9 * len(cams)), sharex=True)
    for ax, c in zip(np.atleast_1d(axes), cams):
        t = costs[c]["t"]
        tc = _tracks(work, [c])[c]["t"]  # native frame times (sync-corrected), for the applied vetoes
        for role, col, (y0, y1) in (("lead", "#b03030", (300, 3000)), ("follow", "#2f5fb0", (2, 20))):
            ax.plot(t, costs[c][role], color=col, lw=0.7, label=role)
            ax.axhline(dec[c][role]["threshold_px"], color=col, lw=0.5, ls=":")
            fr = np.asarray(done.get(c, {}).get(role, []), int)
            fr = fr[fr < len(tc)]
            if len(fr):  # applied: solid band in the role's half of the panel (lead top, follow bottom)
                app = np.zeros(len(t), bool)
                for i in fr:
                    app |= np.abs(t - tc[i]) <= 0.5 / 30
                ax.fill_between(t, y0, y1, where=app, color=col, alpha=0.35, lw=0, step="mid")
            v = dec[c][role]["veto_grid"]
            if v.any():  # candidates of the automatic rule now: hatched outline only
                ax.fill_between(t, y0, y1, where=v, facecolor="none", edgecolor=col, hatch="////", lw=0.0,
                                alpha=0.6, step="mid")
        ax.set_yscale("log")
        ax.set_ylim(2, 3000)
        ax.set_ylabel(f"cam {c}\npx", fontsize=8)
        ax.grid(alpha=0.3)
        n = {r: sum(x["instants"] for x in dec[c][r]["runs"]) for r in ROLES}
        na = {r: len(done.get(c, {}).get(r, [])) for r in ROLES}
        ax.text(0.005, 0.85, f"cam {c}: median lead {dec[c]['lead']['median_px']} / follow {dec[c]['follow']['median_px']} "
                             f"px; APPLIED native frames lead {na['lead']}, follow {na['follow']}; candidate instants "
                             f"now lead {n['lead']}, follow {n['follow']}", transform=ax.transAxes, fontsize=8)
    handles = [plt.Line2D([], [], color="#b03030", lw=1, label="lead"),
               plt.Line2D([], [], color="#2f5fb0", lw=1, label="follow"),
               Patch(facecolor="#888888", alpha=0.35, label="veto APPLIED (masks blanked; lead top band, follow bottom)"),
               Patch(facecolor="none", edgecolor="#555555", hatch="////", label="candidate of the automatic rule now "
                                                                                 "(not applied)")]
    np.atleast_1d(axes)[0].legend(handles=handles, loc="upper right", fontsize=7, ncol=4)
    np.atleast_1d(axes)[-1].set_xlabel("reference time (s, sync-corrected)")
    fig.suptitle(f"{work.name}: each role's keypoints vs the same role triangulated from the OTHER main cameras; "
                 "solid = veto applied, hatched = candidate only, dotted = threshold", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=105)
    plt.close(fig)
    return out


def run(take: str, do_apply: bool = False, only: list[str] | None = None, check: bool = False) -> dict:
    work = md.take_work(take)
    cams = sorted(p.stem for p in (work / "cameras").glob("[0-9][0-9].npz"))
    t0 = time.time()
    costs = role_costs(work, cams)
    dec = decide(costs)
    plan = {c: {r: native_veto(work, c, costs[c]["t"], dec[c][r]["veto_grid"]).tolist() for r in ROLES} for c in cams}
    rep = {"take": take, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "seconds": round(time.time() - t0, 1),
           "rule": {"min_px": VETO_MIN_PX, "factor_x_median": VETO_FACTOR, "min_run_instants": MIN_RUN,
                    "gap_bridged": GAP, "dilate": DILATE},
           "cams": {c: {r: {k: v for k, v in dec[c][r].items() if k != "veto_grid"} | {"native_frames": len(plan[c][r])}
                        for r in ROLES} for c in cams},
           "applied": False}
    if do_apply:  # only: apply to these cameras (a wrong-person camera also corrupts the others' triangulations:
        # veto it first, re-measure, then the rest)
        rep["applied_frames"] = apply(work, {c: v for c, v in plan.items() if not only or c in only})
        rep["applied_cams"] = only or cams
        rep["applied"] = True
    # --check re-measures without touching the record of what was applied (geo_veto.json)
    (work / "dancers" / ("geo_veto_check.json" if check and not do_apply else "geo_veto.json")).write_text(
        json.dumps(rep | {"plan": plan, "applied_so_far": applied_frames(work)}, indent=1))
    rep["review"] = str(review(work, costs, dec, work / "review" / "reid_veto.png"))
    for c in cams:
        for r in ROLES:
            for x in dec[c][r]["runs"]:
                print(f"[geoveto] cam {c} {r}: {x['t0']:.2f}-{x['t1']:.2f} s ({x['instants']} instants, median "
                      f"{x['median_px']} px vs threshold {dec[c][r]['threshold_px']})", flush=True)
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m dancecap.dancers_geoveto", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("take")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--only", help="--apply: comma-separated cameras")
    ap.add_argument("--check", action="store_true", help="re-measure only: dancers/geo_veto_check.json + review image")
    ap.add_argument("--manual", action="append", help="cam:role:i0-i1 native frames to veto after a visual check "
                                                       "(e.g. a camera whose pose is not ok there, so geometry is blind)")
    a = ap.parse_args(argv)
    if a.manual:
        work = md.take_work(a.take)
        plan: dict = {}
        for m in a.manual:
            c, r, rng = m.split(":")
            i0, i1 = (int(x) for x in rng.split("-"))
            plan.setdefault(c, {}).setdefault(r, []).extend(range(i0, i1 + 1))
        print(json.dumps(apply(work, plan, source="manual " + " ".join(a.manual))))
        return
    rep = run(a.take, a.apply, a.only.split(",") if a.only else None, check=a.check)
    print(json.dumps({c: {r: v[r]["native_frames"] for r in ROLES} for c, v in rep["cams"].items()}))


if __name__ == "__main__":
    main()
