"""Which stretch of the class gives the best avatar textures (agent A1; stage `texture-select`).

A texture needs every side of each teacher seen by a camera that sees it WELL. In a lesson the teachers stand and talk, so a
random window may show Larissa's back to nobody. This stage scores sliding windows of the STITCHED poses:

  * per teacher and 30 Hz frame: the body heading (the pelvis rotation's forward axis in the world, SMPL-X faces +z, y up) and the
    azimuth of every phone that confidently sees the teacher (people/coverage.json, per master second) as seen from the teacher;
  * the angle between the two (0 = the phone is in front of the teacher, 180 = behind) falls into one of N_SECTORS body-relative sectors;
  * a sector counts as covered by `need_s` seconds of sight, each phone weighted by its picture quality (QUALITY: the two 1080 x 1920
    phones 1.0 / 0.7, the 480-pixel phones 0.35);
  * windows are penalised by the share of frames in which the couple is in contact (pelvis-to-pelvis < CONTACT_M): the partner's dark sleeve over her
    white top and her long hair over his shirt smear the bake there (measured: the 154-200 s window, full coverage but ~half of it in embraces, left 20-31 % hair
    / sleeve residue on her top against 9 % in the 100-130 s window);
  * window score = the mean over both teachers of (the mean sector coverage, capped at 1) plus 0.25 x the worst teacher's worst sector (a window
    that shows one teacher's back to nobody loses) plus 0.1 x "depth" (the sight beyond the first `need_s`, up to 4 x: a tie-break).

`choose` takes the best window, then (greedy) the window that adds the most coverage to what the chosen ones left open, up to `max_windows`
(the avatar_texture bake runs on ONE take: more than one window means one pseudo-chunk each and a fusion that does not exist yet, so the
default is one window of `length_s` and the report says what the best second one would add).

    python -m dancecap.recap_avatars texture-select [--smplx-dir DIR] [--length 60] [--step 5]
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

N_SECTORS = 8
CONTACT_M = 0.8     # pelvis-to-pelvis ground distance below which the couple counts as in contact (an embrace)
QUALITY = {"01": 1.0, "06": 0.7, "00": 0.35, "02": 0.35, "03": 0.35, "05": 0.35}


def rotvec_forward(rv: np.ndarray) -> np.ndarray:
    """(T,3) axis-angle -> (T,3) the rotated +z axis (Rodrigues)."""
    rv = np.asarray(rv, float)
    th = np.linalg.norm(rv, axis=1, keepdims=True)
    k = rv / np.maximum(th, 1e-12)
    z = np.array([0.0, 0.0, 1.0])
    c, s = np.cos(th), np.sin(th)
    return z * c + np.cross(k, z) * s + k * (k @ z)[:, None] * (1 - c)


def heading_azimuth(global_orient: np.ndarray) -> np.ndarray:
    """(T,) azimuth atan2(x, z) of the body's forward axis in the world (y up)."""
    f = rotvec_forward(global_orient)
    return np.arctan2(f[:, 0], f[:, 2])


def wrap(a: np.ndarray) -> np.ndarray:
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def sector_of(rel: np.ndarray, n: int = N_SECTORS) -> np.ndarray:
    """Body-relative sector 0..n-1: 0 is centred on the front (the phone straight in front), counting toward her left; sector n/2 the back."""
    return (np.floor((wrap(rel) + np.pi / n) / (2 * np.pi / n)).astype(int)) % n


def sector_seconds(person_xz: np.ndarray, heading: np.ndarray, cams: dict[str, np.ndarray], seen: dict[str, np.ndarray], dt: float,
                   n: int = N_SECTORS) -> np.ndarray:
    """(T, n) seconds of quality-weighted sight per frame (the frame's weight is dt), for ONE teacher.
    person_xz (T,2) ground position, heading (T,), cams {cam: (T,3) camera centres in the world}, seen {cam: (T,) bool the phone sees him}."""
    out = np.zeros((len(heading), n))
    for c, C in cams.items():
        az = np.arctan2(C[:, 0] - person_xz[:, 0], C[:, 2] - person_xz[:, 1])
        sec = sector_of(az - heading, n)
        w = QUALITY.get(c, 0.35) * dt * seen[c]
        np.add.at(out, (np.arange(len(sec)), sec), w)
    return out


def window_score(sec_a: np.ndarray, sec_b: np.ndarray, need_s: float = 2.0) -> dict:
    """sec_* (T, n): the per-frame sector seconds of the two teachers inside the window."""
    cov = [np.minimum(s.sum(0) / need_s, 1.0) for s in (sec_a, sec_b)]
    mean = float(np.mean([c.mean() for c in cov]))
    worst = float(min(c.min() for c in cov))
    deep = float(np.mean([np.minimum(s.sum(0) / (4 * need_s), 1.0).mean() for s in (sec_a, sec_b)]))   # tie-break: 4 x need seconds of sight
    return {"score": round(mean + 0.25 * worst + 0.1 * deep, 4), "mean_coverage": round(mean, 3), "worst_sector": round(worst, 3), "depth": round(deep, 3),
            "per_teacher": [[round(float(x), 2) for x in c] for c in cov]}


def seen_from_coverage(cov_rows: list[list[str]], t: np.ndarray, cams: list[str], t0: float = 0.0) -> dict[str, np.ndarray]:
    """people/coverage.json's per-master-second camera lists -> {cam: (T,) bool} at the frame times t."""
    sec = np.clip(np.floor(np.asarray(t) - t0).astype(int), 0, len(cov_rows) - 1)
    return {c: np.array([c in cov_rows[s] for s in sec]) for c in cams}


def sliding(sec_a: np.ndarray, sec_b: np.ndarray, t: np.ndarray, length_s: float, step_s: float, need_s: float,
            contact: np.ndarray | None = None, contact_penalty: float = 0.3) -> list[dict]:
    out = []
    t_lo, t_hi = float(t[0]), float(t[-1])
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.0       # the last frame lies one step before the end of its interval
    start = t_lo
    while start + length_s <= t_hi + dt + 1e-6 or (not out and t_hi - t_lo < length_s):
        sel = (t >= start) & (t < start + length_s)
        if sel.any():
            r = window_score(sec_a[sel], sec_b[sel], need_s)
            if contact is not None:     # the partner's dark sleeve over her white top (and her hair over his shirt) smear the bake: penalise embrace time
                r["contact"] = round(float(contact[sel].mean()), 3)
                r["score"] = round(r["score"] - contact_penalty * r["contact"], 4)
            r.update({"t0": round(start, 2), "t1": round(min(start + length_s, t_hi), 2)})
            out.append(r)
        start += step_s
        if t_hi - t_lo < length_s:
            break
    return out


def choose(sec_a: np.ndarray, sec_b: np.ndarray, t: np.ndarray, length_s: float = 60.0, step_s: float = 5.0, need_s: float = 2.0,
           max_windows: int = 1, contact: np.ndarray | None = None) -> dict:
    """The best window, then greedily the windows that add the most to what is still open (never overlapping a chosen one).
    -> {windows, ranking (top 8 single windows), combined (the chosen windows together), second_best_adds (what ONE MORE window would add)}."""
    allw = sliding(sec_a, sec_b, t, length_s, step_s, need_s, contact)
    if not allw:
        raise ValueError("no window: the poses cover less time than one window")

    def seconds_of(ws):
        tot = [np.zeros(N_SECTORS), np.zeros(N_SECTORS)]
        for w in ws:
            sel = (t >= w["t0"]) & (t < w["t1"] + 1e-9)
            for i, sec in enumerate((sec_a, sec_b)):
                tot[i] = tot[i] + sec[sel].sum(0)
        return tot

    def score(ws):
        return window_score(*(c[None, :] for c in seconds_of(ws)), need_s)

    chosen = [max(allw, key=lambda r: r["score"])]
    nxt = None
    while True:
        base = score(chosen)["score"]
        cands = [(score(chosen + [w])["score"] - base, w) for w in allw if all(abs(w["t0"] - c["t0"]) >= length_s for c in chosen)]
        if not cands:
            break
        gain, w = max(cands, key=lambda x: x[0])
        if len(chosen) < max_windows:
            chosen.append(w)
        else:
            nxt = {"t0": w["t0"], "t1": w["t1"], "adds_score": round(float(gain), 4)}
            break
    return {"windows": chosen, "ranking": sorted(allw, key=lambda r: -r["score"])[:8], "combined": score(chosen), "second_best_adds": nxt,
            "length_s": length_s, "need_s": need_s}


def run(work: Path, smplx_dir: Path, length_s: float = 60.0, step_s: float = 5.0, need_s: float = 2.0, max_windows: int = 1,
        out: Path | None = None, t_range: tuple[float, float] | None = None) -> dict:
    """The stage: stitched smplx/{lead,follow}.npz + cameras/<cam>_world.npz + people/coverage.json -> avatar/texture/windows.json.
    t_range restricts the frames considered (master seconds): the core of one farm chunk when the poses are a raw chunk fit."""
    cov = json.loads((work / "people" / "coverage.json").read_text())
    cams = list(cov["per_camera"])
    sm = {r: np.load(smplx_dir / f"{r}.npz", allow_pickle=True) for r in ("lead", "follow")}
    t_all = np.asarray(sm["lead"]["t"], float)
    m = np.ones(len(t_all), bool) if t_range is None else (t_all >= t_range[0]) & (t_all <= t_range[1])
    t = t_all[m]
    cam_c = {}
    for c in cams:
        z = np.load(work / "cameras" / f"{c}_world.npz", allow_pickle=True)
        tc = np.asarray(z["t"], float)
        C = np.asarray(z["C"], float)
        cam_c[c] = np.stack([np.interp(t, tc, C[:, k]) for k in range(3)], 1)
    dt = float(np.median(np.diff(t)))
    secs = []
    pel = {r: np.asarray(sm[r]["joints"], float)[m][:, 0] for r in ("lead", "follow")}
    contact = np.linalg.norm(pel["lead"][:, [0, 2]] - pel["follow"][:, [0, 2]], axis=1) < CONTACT_M
    for r in ("lead", "follow"):
        s = sm[r]
        pelvis = pel[r]
        seen = seen_from_coverage(cov[r], t, cams)
        secs.append(sector_seconds(pelvis[:, [0, 2]], heading_azimuth(np.asarray(s["global_orient"])[m]), cam_c, seen, dt))
    res = choose(secs[0], secs[1], t, length_s, step_s, need_s, max_windows, contact)
    res.update({"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "smplx_dir": str(smplx_dir), "t_range": [float(t[0]), float(t[-1])],
                "sectors": f"{N_SECTORS} body-relative sectors, 0 = a phone straight in front, {N_SECTORS // 2} = straight behind; quality weights {QUALITY}"})
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=1))
    return res


def run_chunks(work: Path, length_s: float = 50.0, step_s: float = 5.0, need_s: float = 2.0, out: Path | None = None) -> dict:
    """BEFORE the stitch: the best window inside each farm chunk's core (a window within one chunk's fit needs no seam, so its raw poses are
    exactly what the stitch will contain there). Chunk plan = work/<take>/chunks.json, fits = work/<take>_<id>/smplx/. -> {best, per_chunk}."""
    plan = json.loads((work / "chunks.json").read_text())
    per = []
    for c in plan["chunks"]:
        d = work.parent / f"{work.name}_{c['id']}" / "smplx"
        if not (d / "lead.npz").exists():
            per.append({"chunk": c["id"], "status": "no fit pulled yet"})
            continue
        k0, k1 = c["core_k"] if "core_k" in c else (c["k"][0], c["k"][1])
        r = run(work, d, min(length_s, (k1 - k0 + 1) / 30.0 - 1.0), step_s, need_s, 1, None, ((k0 / 30.0), (k1 + 1) / 30.0))
        w = r["windows"][0]
        z = np.load(d / "follow.npz", allow_pickle=True)
        sel = (np.asarray(z["t"]) >= w["t0"]) & (np.asarray(z["t"]) <= w["t1"])
        rp = float(np.median(np.asarray(z["reproj_px"])[sel])) if "reproj_px" in z.files else float("nan")
        per.append({"chunk": c["id"], "smplx_dir": str(d), "window": [w["t0"], w["t1"]], "score": w["score"], "mean_coverage": w["mean_coverage"],
                    "worst_sector": w["worst_sector"], "depth": w.get("depth"), "contact": w.get("contact"), "reproj_px_median": round(rp, 2), "per_teacher": w["per_teacher"]})
    ok = [r for r in per if "score" in r]
    if not ok:
        raise FileNotFoundError("no chunk fit pulled yet")
    best = max(ok, key=lambda r: (round(r["score"], 2), -r["reproj_px_median"]))     # equal coverage: the better-fitted window
    res = {"windows": [{"t0": best["window"][0], "t1": best["window"][1], **{k: best[k] for k in ("score", "mean_coverage", "worst_sector", "per_teacher")}}],
           "smplx_dir": best["smplx_dir"], "chunk": best["chunk"], "per_chunk": per, "length_s": length_s, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "note": "raw chunk fit (no seam inside the window); after the stitch re-run `texture-select` on smplx/ for windows across seams"}
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=1))
    return res
