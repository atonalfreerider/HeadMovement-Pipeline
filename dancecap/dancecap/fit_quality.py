"""Per-segment fit quality for the farm's coverage map, computed on the fit pod (all analysis compute runs on Runpod).

    python -m dancecap.fit_quality <take> [--refined smplx_rf2_noslip] [--flag-px 15]

For a finished segment fit (work/<take>/smplx = the raw MAMMA fit, work/<take>/<refined> = pose_refine's
smplx_rf2_noslip) this runs dancecap.pose_quality on both folders (reprojection per camera and dancer incl. feet,
floor contact / foot slip, root jitter, displacement of the refined motion vs the raw one -> quality/quality_<dir>.json
+ quality_<dir>_frames.npz) and condenses them into two small files:

  work/<take>/quality/fit_quality.json   per camera and dancer: whole-segment reprojection (median / p90 / frames
        observed) for raw and refined; per 10 s block of the reference clock the same per camera, with the camera's
        kind (zoom / attached / handheld) so the zoom cameras 04 / 05 / 07 / 09 and the attached 05 / 06 can be read at
        a glance; the foot-floor and jitter summaries of pose_quality; n_cams statistics per block.
  work/<take>/quality/fit_coverage.npz   per grid frame (the 30 Hz lattice of larissa_kadu_38, k = round((t - anchor) *
        30)): n_cams (cameras that observe the dancers, >= MIN_OBS_PTS visible landmarks of the role), n_cams_lead /
        n_cams_follow, quality_flag_<role> (0 absent, 1 mono, 2 two_view, 3 multi: the farm's codes), and per camera
        (C, 2, T) obs / px_raw / px_refined (NaN where the camera does not observe the role) and flag_raw /
        flag_refined (the per-camera median reprojection of that frame is over --flag-px), bad_<label> (T,) = the
        frame's median over cameras is over the flag, and the fit's own per-frame quality.

Everything is numpy on arrays that already exist on the pod; no videos or frames are needed.
"""
from __future__ import annotations

import argparse
import json
import time
import tomllib
from pathlib import Path

import numpy as np

from .take import PACKAGE_ROOT

ROLES = ("lead", "follow")
BLOCK_S = 10.0
FLAG_PX = 15.0       # per-camera, per-frame median reprojection (px) above which a camera is flagged for that frame
MIN_OBS_PTS = 8      # a camera observes a role in a frame when at least this many of its 512 landmarks are visible
ANCHOR = 30.151176844227862
HZ = 30.0


def quality_flag_codes(n_cams: np.ndarray) -> np.ndarray:
    """farm codes: 0 absent (no camera), 1 mono, 2 two_view, 3 multi."""
    n = np.asarray(n_cams)
    return np.where(n <= 0, 0, np.minimum(n, 3)).astype(np.int8)


def camera_kinds(cams: list[str]) -> dict:
    """{cam: {'zoom': bool, 'attached': bool, 'mount': str}} from takes/larissa_kadu_full.toml ([cameras.NN], [reuse])."""
    p = PACKAGE_ROOT / "takes" / "larissa_kadu_full.toml"
    cfg = tomllib.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    attach = set((cfg.get("reuse") or {}).get("attach", []))
    return {c: {"zoom": bool((cfg.get("cameras") or {}).get(c, {}).get("zoom", False)), "attached": c in attach,
                "mount": (cfg.get("cameras") or {}).get(c, {}).get("mount", "handheld")} for c in cams}


def stats(x: np.ndarray) -> dict:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0, "median": None, "p90": None, "max": None}
    return {"n": int(len(x)), "median": round(float(np.median(x)), 2), "p90": round(float(np.percentile(x, 90)), 2),
            "max": round(float(x.max()), 1)}


def block_table(t: np.ndarray, cams: list[str], px: dict[str, np.ndarray], obs: np.ndarray, n_cams: np.ndarray,
                block_s: float = BLOCK_S) -> list[dict]:
    """px[label] (C, 2, T) NaN where unobserved; obs (C, 2, T) bool; -> blocks of block_s seconds of the reference clock
    (edges at multiples of block_s): per camera / role / label the stats of the observed frames."""
    t = np.asarray(t, float)
    bid = np.floor(t / block_s + 1e-9).astype(int)
    out = []
    for b in np.unique(bid):
        m = bid == b
        row = {"t0": round(float(b * block_s), 3), "t1": round(float((b + 1) * block_s), 3), "frames": int(m.sum()),
               "t_first": round(float(t[m][0]), 3), "t_last": round(float(t[m][-1]), 3),
               "n_cams_median": float(np.median(n_cams[m])), "n_cams_min": int(n_cams[m].min()),
               "n_cams_max": int(n_cams[m].max()), "cams": {}}
        for ci, c in enumerate(cams):
            d = {}
            for ri, r in enumerate(ROLES):
                o = obs[ci, ri][m]
                if not o.any():
                    continue
                d[r] = {"frames_observed": int(o.sum()), **{lab: stats(px[lab][ci, ri][m]) for lab in px}}
            if d:
                row["cams"][c] = d
        out.append(row)
    return out


def per_camera_summary(cams: list[str], px: dict[str, np.ndarray], obs: np.ndarray, kinds: dict) -> dict:
    out = {}
    for ci, c in enumerate(cams):
        out[c] = {"kind": kinds.get(c, {}), **{r: {"frames_observed": int(obs[ci, ri].sum()),
                                                   **{lab: stats(px[lab][ci, ri]) for lab in px}}
                                              for ri, r in enumerate(ROLES)}}
    return out


def run(take: str, refined: str = "smplx_rf2_noslip", flag_px: float = FLAG_PX) -> dict:
    from . import pose_quality as pq
    work = PACKAGE_ROOT / "work" / take
    prep = json.loads((work / "mamma" / "prepare.json").read_text())
    tag = next(iter(prep["sets"]))
    s = prep["sets"][tag]
    cams = list(s["cams"])
    t = np.asarray(prep["grid"]["t"], float)
    T = len(t)
    k = np.rint((t - ANCHOR) * HZ).astype(np.int32)
    d2 = work / "mamma" / "ma_2d" / s["seq"]
    # who observes whom
    obs = np.zeros((len(cams), 2, T), bool)
    for ci, c in enumerate(cams):
        v = np.load(d2 / f"cam{c}.npz")["visibilities"]          # (T, 2, 512)
        obs[ci] = ((v > 0).sum(-1) >= MIN_OBS_PTS).T
    n_role = {r: obs[:, ri].sum(0).astype(np.int16) for ri, r in enumerate(ROLES)}
    n_cams = obs.any(1).sum(0).astype(np.int16)
    labels = [("raw", "smplx")] + ([("refined", refined)] if (work / refined / "lead.npz").exists() else [])
    qjson, px, frames_npz = {}, {}, {}
    for label, d in labels:
        qjson[label] = pq.run(take, smplx_dir=d, tag=d, ref_dir="smplx")
        fr = np.load(work / "quality" / f"quality_{d}_frames.npz")
        arr = np.full((len(cams), 2, T), np.nan, np.float32)
        for ci, c in enumerate(cams):
            for ri, r in enumerate(ROLES):
                key = f"px_{c}_{r}"
                if key in fr.files:
                    arr[ci, ri] = np.where(obs[ci, ri], fr[key][:T], np.nan)
        px[label] = arr
        frames_npz[label] = {r: np.asarray(fr[f"allpx_{r}"], np.float32) for r in ROLES if f"allpx_{r}" in fr.files}
    kinds = camera_kinds(cams)
    cov = {"t": t, "k": k, "n_cams": n_cams, "n_cams_lead": n_role["lead"], "n_cams_follow": n_role["follow"],
           "cams": np.array(cams), "roles": np.array(ROLES), "obs": obs,
           "quality_flag_lead": quality_flag_codes(n_role["lead"]),
           "quality_flag_follow": quality_flag_codes(n_role["follow"]),
           "flag_px": np.float32(flag_px), "min_obs_points": np.int16(MIN_OBS_PTS)}
    for label in px:
        with np.errstate(all="ignore"):
            cov[f"px_{label}"] = px[label]
            cov[f"flag_{label}"] = np.nan_to_num(px[label], nan=0.0) > flag_px
            med = np.nanmedian(np.where(obs, px[label], np.nan), axis=0)       # (2, T) over the observing cameras
        cov[f"allpx_{label}"] = med.astype(np.float32)
        cov[f"bad_{label}"] = np.nan_to_num(med, nan=0.0).max(0) > flag_px
    for label, d in labels:
        z = {r: np.load(work / d / f"{r}.npz", allow_pickle=True) for r in ROLES}
        cov[f"fit_quality_{label}"] = np.stack([np.asarray(z[r]["quality"], np.float32) for r in ROLES])
        cov[f"repaired_{label}"] = np.stack([np.asarray(z[r]["repaired"], bool) for r in ROLES])
    out_dir = work / "quality"
    out_dir.mkdir(exist_ok=True)
    np.savez_compressed(out_dir / "fit_coverage.npz", **cov)
    blocks = block_table(t, cams, px, obs, n_cams)
    rep = {
        "schema": "dancecap.fit_quality/1", "take": take, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "grid": {"T": int(T), "k": [int(k[0]), int(k[-1])], "t": [round(float(t[0]), 4), round(float(t[-1]), 4)],
                 "anchor": ANCHOR, "hz": HZ},
        "tag": tag, "cams": cams, "camera_kinds": kinds, "block_s": BLOCK_S, "flag_px": flag_px,
        "min_obs_points": MIN_OBS_PTS,
        "labels": {lab: {"dir": f"work/{take}/{d}", "quality_json": f"quality/quality_{d}.json",
                         "frames_npz": f"quality/quality_{d}_frames.npz"} for lab, d in labels},
        "n_cams": {"hist": {int(v): int((n_cams == v).sum()) for v in np.unique(n_cams)},
                   "mono_frames": int((n_cams == 1).sum()), "no_camera_frames": int((n_cams == 0).sum())},
        "per_camera": per_camera_summary(cams, px, obs, kinds),
        "blocks": blocks,
        "frames_over_flag": {lab: {"bad_frames": int(cov[f"bad_{lab}"].sum()), "fraction": round(float(cov[f"bad_{lab}"].mean()), 4)}
                             for lab in px},
        "pose_quality": {lab: {
            "reproj_px_all_cams_per_frame_median": q.get("reproj_px", {}).get("all_cams_per_frame_median"),
            "reproj_feet_px_all_cams_per_frame_median": q.get("reproj_feet_px", {}).get("all_cams_per_frame_median"),
            "floor": q.get("floor"), "slip": q.get("slip"), "jitter_4p5_12hz_mm": q.get("jitter_4p5_12hz_mm"),
            "displacement_vs_ref_mm": q.get("displacement_vs_ref_mm"), "rest_height_m": q.get("rest_height_m")}
            for lab, q in qjson.items()},
    }
    (out_dir / "fit_quality.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    zoom = [c for c in cams if kinds[c]["zoom"]]
    att = [c for c in cams if kinds[c]["attached"]]
    print(f"[fit_quality] {take}: {len(labels)} motion(s), cams {cams} (zoom {zoom}, attached {att}), {len(blocks)} "
          f"blocks, flagged frames " + ", ".join(f"{lab} {v['bad_frames']}" for lab, v in rep["frames_over_flag"].items()),
          flush=True)
    return rep


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m dancecap.fit_quality", description=__doc__.split("\n")[0])
    ap.add_argument("take")
    ap.add_argument("--refined", default="smplx_rf2_noslip")
    ap.add_argument("--flag-px", type=float, default=FLAG_PX)
    a = ap.parse_args(argv)
    run(a.take, a.refined, a.flag_px)


if __name__ == "__main__":
    main()
