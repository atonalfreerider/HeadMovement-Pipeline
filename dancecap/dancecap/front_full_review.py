"""Local quality gates and review images of the larissa_kadu_full front-end (after the pod session; no pod, no GPU).

    python -m dancecap.front_full_review gates            # -> work/larissa_kadu_full/front/gates.json (+ gates.md)
    python -m dancecap.front_full_review images [--instants 16]   # -> work/larissa_kadu_full/review/*.jpg|png
    python -m dancecap.front_full_review apply-vetoes --manual-only   # review gate, 1: MANUAL_VETOES (looked at)
    python -m dancecap.front_full_review audit --refresh  # 2: union geometry on the cleaned data -> front/veto_plan.json
    python -m dancecap.front_full_review apply-vetoes     # 3: + the geometric identity / pose spans
    python -m dancecap.front_full_review colour [--dt 0.5]    # torso-colour lead/follow check -> front/colour_<s>.json

The whole dance is owned, instant by instant, by one source of front-end data: segment a before larissa_kadu_38's first
grid frame (30.151 s), larissa_kadu_38 (reused, never re-run) up to its last (68.385 s), then segments b, c, d split
at the middle of their 4 s overlaps (105.235 s, 144.101 s). Gates (PLAN.md 3.2 step 6, numbers per camera, per
segment and per 10 s block of the reference clock):
  frames        decoded frames, PTS monotonic, the pod's decode vs larissa_kadu_38's local frames (front/decode_check)
  localisation  ok rate >= 95 % and static (map) reprojection < 4 px per camera; per 10 s block ok rate
  attach        05 / 06 dancer residual (hold-out) within 2x larissa_kadu_38's; camera centre shift vs 38
  sync          used, not at the search edge, |delta - delta_38| <= 15 ms (more = flagged drift, kept)
  labels        per-dancer mask coverage >= 90 % of the frames (cam09 before Kadu enters is EXPECTED to miss the lead)
  identity      segment a / b vs larissa_kadu_38 and c vs b, d vs c on their overlaps: the same role boxes on the same
                native frames (IoU >= 0.5 on >= 90 %); no swapped frames; geometric re-ID (dancers/geo_veto_check.json)
  typed focal   01 / 02 / 03 / 06 (handheld, no zoom: ONE focal per video): within 1 % of larissa_kadu_38's median
  world         world_from_sfm.json identical to larissa_kadu_38's (copied); chain cross-check after sync <= 2 %
  colour        Kadu wears a black T-shirt, Larissa a white crop top: the lead mask's torso must be dark, the
                follow's light (front/colour_<seg>.json; sustained flags = a swap the geometry cannot see, e.g. cam06
                1.6-14.7 s where the follow mask sat on Kadu while the lead mask sat on a spectator)
  vetoes        what the review gate blanked (roles.json geo_veto history: masks + ma_2d visibilities), per camera
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from . import front_full as ff
from .take import PACKAGE_ROOT, Take

OWNERS = [("a", -1e9, 30.151176844227862), ("38", 30.151176844227862, 68.3845101775612),
          ("b", 68.3845101775612, 105.2345), ("c", 105.2345, 144.1012), ("d", 144.1012, 1e9)]
DANCED = (-3.1313, 185.0)
PTS_TOL_S = 1.5e-3  # same native frame in two takes (ffmpeg 4.4 pod decode: pts_time to 6 significant digits)
ROLES = ("lead", "follow")


def take_of(key: str) -> str:
    return ff.REF if key == "38" else ff.seg_name(key)


def owner(t: float) -> str:
    return next(k for k, a, b in OWNERS if a <= t < b)


class TD:
    """One take's front-end outputs (lazy)."""

    def __init__(self, key: str):
        self.key = key
        self.take = Take.load(take_of(key))
        self.work = self.take.work
        self.cams = [s.id for s in self.take.sources if (self.work / "frames" / s.id / "frames.json").exists()]
        self._c: dict = {}

    def meta(self, c):
        return self._get(("meta", c), lambda: json.loads((self.work / "frames" / c / "frames.json").read_text()))

    def t(self, c):
        return self._get(("t", c), lambda: np.array([r["t"] for r in self.meta(c)["frames"]]))

    def pts(self, c):
        return self._get(("pts", c), lambda: np.array([r["pts"] for r in self.meta(c)["frames"]]))

    def cam(self, c):
        p = self.work / "cameras" / f"{c}.npz"
        return self._get(("cam", c), lambda: dict(np.load(p)) if p.exists() else None)

    def roles(self):
        p = self.work / "dancers" / "roles.json"
        return self._get("roles", lambda: json.loads(p.read_text()) if p.exists() else {"cams": {}})

    def boxes(self, c, role):
        def f():
            fr = (self.roles()["cams"].get(c) or {}).get("frames") or []
            out = np.full((len(self.t(c)), 4), np.nan)
            for i, r in enumerate(fr[:len(out)]):
                b = (r.get(role) or {}).get("box")
                if b is not None:
                    out[i] = b
            return out
        return self._get(("box", c, role), f)

    def vis(self, c):
        """(F, 2) fraction of the 512 ma_2d landmarks with visibility > 0.5 per dancer."""
        p = self.work / "mamma" / "ma_2d_native" / f"{c}.npz"

        def f():
            if not p.exists():
                return None
            v = np.load(p)["visibilities"][:, :2]
            out = (v > 0.5).mean(-1)
            return np.pad(out, ((0, 0), (0, 2 - out.shape[1]))) if out.shape[1] < 2 else out
        return self._get(("vis", c), f)

    def json(self, rel):
        p = self.work / rel
        return self._get(("json", rel), lambda: json.loads(p.read_text()) if p.exists() else None)

    def _get(self, k, fn):
        if k not in self._c:
            self._c[k] = fn()
        return self._c[k]


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x0, y0 = np.maximum(a[:, 0], b[:, 0]), np.maximum(a[:, 1], b[:, 1])
    x1, y1 = np.minimum(a[:, 2], b[:, 2]), np.minimum(a[:, 3], b[:, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    ar = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])  # noqa: E731
    return inter / np.maximum(ar(a) + ar(b) - inter, 1e-9)


def identity_overlap(A: TD, B: TD, span: tuple[float, float] | None = None) -> dict:
    """Same native frames (PTS) of the same camera in two takes: role boxes must agree; a swap = A's lead on B's follow."""
    out = {}
    for c in sorted(set(A.cams) & set(B.cams)):
        pa, pb = A.pts(c), B.pts(c)
        # nearest native PTS; 1.5 ms tolerance: the segment takes were decoded on the pod by the apt ffmpeg 4.4, whose
        # showinfo prints pts_time with 6 significant digits (<= 0.5 ms off; pixels identical: front/decode_check_tol)
        j = np.clip(np.searchsorted(pb, pa), 1, max(len(pb) - 1, 1))
        j = np.where(np.abs(pb[j - 1] - pa) <= np.abs(pb[np.minimum(j, len(pb) - 1)] - pa), j - 1,
                     np.minimum(j, len(pb) - 1))
        same = np.abs(pb[j] - pa) < PTS_TOL_S
        ia, ib = np.flatnonzero(same), j[same]
        if span is not None:
            keep = (A.t(c)[ia] >= span[0]) & (A.t(c)[ia] <= span[1])
            ia, ib = ia[keep], ib[keep]
        if len(ia) == 0:
            continue
        rows = {"shared_frames": int(len(ia)), "t": [round(float(A.t(c)[ia[0]]), 2), round(float(A.t(c)[ia[-1]]), 2)]}
        swap_any = np.zeros(len(ia), bool)
        for role, other in (("lead", "follow"), ("follow", "lead")):
            a, b, bo = A.boxes(c, role)[ia], B.boxes(c, role)[ib], B.boxes(c, other)[ib]
            both = np.isfinite(a).all(1) & np.isfinite(b).all(1)
            iou = np.where(both, _iou(np.nan_to_num(a), np.nan_to_num(b)), np.nan)
            iou_o = np.where(np.isfinite(a).all(1) & np.isfinite(bo).all(1), _iou(np.nan_to_num(a), np.nan_to_num(bo)), np.nan)
            swap = (np.nan_to_num(iou_o, nan=0) >= 0.5) & (np.nan_to_num(iou, nan=1) < 0.3)
            swap_any |= swap
            rows[role] = {"both_boxed": int(both.sum()),
                          "iou_median": None if not both.any() else round(float(np.nanmedian(iou)), 3),
                          "agree_frac": None if not both.any() else round(float(np.mean(iou[both] >= 0.5)), 3),
                          "swapped_frames": int(swap.sum())}
        ok = all((rows[r]["agree_frac"] is None or rows[r]["agree_frac"] >= 0.9) and rows[r]["swapped_frames"] == 0
                 for r in ROLES)
        rows["passed"] = bool(ok)
        out[c] = rows
    return out


def block_stats(D: TD, c: str, lo: float, hi: float) -> dict | None:
    t = D.t(c)
    m = (t >= lo) & (t < hi)
    if not m.any():
        return None
    cam = D.cam(c)
    d = {"frames": int(m.sum())}
    if cam is not None:
        ok = np.asarray(cam["ok"], bool)[m]
        d["loc_ok"] = round(float(ok.mean()), 3)
        rp = np.asarray(cam["reproj_px"], float)[m][ok]
        d["reproj_px"] = round(float(np.median(rp)), 2) if len(rp) else None
    for role in ROLES:
        d[f"{role}_mask"] = round(float(np.isfinite(D.boxes(c, role)[m]).all(1).mean()), 3)
    v = D.vis(c)
    if v is not None and len(v) == len(t):
        d["ma2d_lead"] = round(float((v[m, 0] > 0.25).mean()), 3)
        d["ma2d_follow"] = round(float((v[m, 1] > 0.25).mean()), 3)
    return d


def _longest_gap(t: np.ndarray, ok: np.ndarray) -> float:
    best, start = 0.0, None
    for i, o in enumerate(np.r_[ok, True]):
        if not o and start is None:
            start = i
        elif o and start is not None:
            best = max(best, float(t[min(i, len(t) - 1)] - t[start]))
            start = None
    return round(best, 2)


def segment_gates(D: TD, R: TD, prev: TD | None) -> dict:
    g: dict = {"take": D.take.name, "window": list(D.take.window), "cams": D.cams}
    rep = D.json("cameras/report.json") or {}
    rep38 = R.json("cameras/report.json") or {}
    sync = (D.json("sync/offsets.json") or {}).get("offsets", {})
    dec = ff.full_work() / "front" / "decode_check.json"
    loc, att, syn, lab, foc = {}, {}, {}, {}, {}
    for c in D.cams:
        cam = D.cam(c)
        t = D.t(c)
        if cam is None:
            loc[c] = {"passed": False, "why": "no cameras/<cam>.npz"}
            continue
        ok = np.asarray(cam["ok"], bool)
        rc = (rep.get("cams") or {}).get(c) or {}
        rp = np.asarray(cam["reproj_px"], float)[ok]
        loc[c] = {"frames": int(len(ok)), "ok_rate": round(float(ok.mean()), 4),
                  "reproj_px_median": round(float(np.median(rp)), 2) if len(rp) else None,
                  "static_map_reproj_px": rc.get("map_reproj_px"), "sources": rc.get("sources"),
                  "longest_failed_s": _longest_gap(t, ok), "attached": bool(cam.get("attached", False))}
        loc[c]["passed"] = bool(ok.mean() >= 0.95 and (loc[c]["reproj_px_median"] or 99) < 4.0)
        if bool(cam.get("attached", False)):
            a = (rep.get("attach") or {}).get(c) or {}
            a38 = (rep38.get("attach") or {}).get(c) or {}
            ho, ho38 = (a.get("dancer_holdout") or {}).get("median_px"), (a38.get("dancer_holdout") or {}).get("median_px")
            C, C38 = a.get("camera_centre_sfm"), a38.get("camera_centre_sfm")
            shift = None
            if C is not None and C38 is not None:
                s = float(json.loads((D.work / "cameras" / "world_from_sfm.json").read_text())["s"])
                shift = round(s * float(np.linalg.norm(np.subtract(C, C38))), 3)
            att[c] = {"method": a.get("method"), "dancer_holdout_px": ho, "dancer_holdout_px_38": ho38,
                      "ratio_vs_38": round(ho / ho38, 2) if ho and ho38 else None, "centre_shift_m_vs_38": shift,
                      "passed": bool(ho and ho38 and ho <= 2 * ho38)}
        o = sync.get(c)
        if o:
            d = o.get("delta_vs_ref_ms")
            syn[c] = {"delta_ms": o["delta_ms"], "delta_vs_38_ms": d, "stable": o.get("stable"),
                      "spread_ms": o.get("spread_ms"), "use": o.get("use"), "at_search_edge": o.get("at_search_edge"),
                      "drift_flag": bool(d is not None and abs(d) > 15.0),
                      "passed": bool(o.get("use") and not o.get("at_search_edge"))}
        lab[c] = {r: round(float(np.isfinite(D.boxes(c, r)).all(1).mean()), 4) for r in ROLES}
        lab[c]["passed"] = bool(min(lab[c][r] for r in ROLES) >= 0.9)
        st = D.take.camera_type(c)
        if not st.zoom and c in R.cams and R.cam(c) is not None:
            mx = max(int(D.meta(c)["width"]), int(D.meta(c)["height"]))
            f = np.asarray(cam["K"])[ok, 0, 0] / mx
            f38 = np.asarray(R.cam(c)["K"])[np.asarray(R.cam(c)["ok"], bool), 0, 0] / mx
            rel = float(np.median(f) / np.median(f38) - 1)
            foc[c] = {"f_over_maxdim": round(float(np.median(f)), 4), "f38_over_maxdim": round(float(np.median(f38)), 4),
                      "rel": round(rel, 4), "spread": round(float(np.ptp(f)), 5), "passed": abs(rel) <= 0.01}
    g["localisation"] = loc
    g["attach"] = att
    g["sync"] = syn
    g["sync_gauge"] = (D.json("sync/offsets.json") or {}).get("gauge_ref")
    rpt = D.json("dancers/report.json") or {"cams": {}}
    g["labels"] = {c: {**lab[c], "label_frame": ((D.json("dancers/labels.json") or {}).get(c) or {}).get("frame"),
                       "label_conf": ((D.json("dancers/labels.json") or {}).get(c) or {}).get("confidence"),
                       "swaps_repaired": (rpt["cams"].get(c) or {}).get("swaps_repaired"),
                       "identity_flags": (rpt["cams"].get(c) or {}).get("identity_flags")} for c in lab}
    g["typed_focal"] = foc
    gv = D.json("dancers/geo_veto_check.json")
    g["geo_reid"] = None if gv is None else {c: {r: {"runs": v[r]["runs"], "over_threshold": v[r]["over_threshold_instants"],
                                                     "median_px": v[r]["median_px"]} for r in ROLES}
                                              for c, v in gv["cams"].items()}
    if D.key in ("a", "b"):
        span = (R.t(R.cams[0])[0], R.t(R.cams[0])[-1])
        g["identity_vs_38"] = identity_overlap(D, R, None)
    if prev is not None:
        g[f"identity_vs_{prev.key}"] = identity_overlap(D, prev, None)
    w38 = (R.work / "cameras" / "world_from_sfm.json").read_bytes()
    g["world"] = {"identical_to_38": (D.work / "cameras" / "world_from_sfm.json").read_bytes() == w38,
                  "chain_after_sync": D.json("scale/chain_after_sync.json")}
    pj = D.json("mamma/prepare.json")
    g["prepare"] = None if pj is None else {"grid": {k: pj["grid"][k] for k in ("t0", "T", "hz")},
                                            "trim": pj["grid"].get("trim"), "sets": {k: v["cams"] for k, v in pj["sets"].items()},
                                            "inputs": {k: v["inputs"] for k, v in pj["sets"].items()},
                                            "sync_offsets_ms": pj["sync"]["offsets_ms"]}
    if dec.exists() and D.key == "a":
        g["decode_check"] = {c: {k: v[k] for k in ("compared", "ok", "decimate_every", "size")}
                             for c, v in json.loads(dec.read_text())["cams"].items()}
    passes = [v["passed"] for v in loc.values()] + [v["passed"] for v in att.values()] + \
             [v["passed"] for v in syn.values()]
    g["summary"] = {"localisation_failed": [c for c, v in loc.items() if not v["passed"]],
                    "attach_failed": [c for c, v in att.items() if not v["passed"]],
                    "sync_failed": [c for c, v in syn.items() if not v["passed"]],
                    "sync_drift_flags": [c for c, v in syn.items() if v["drift_flag"]],
                    "labels_under_90": [c for c, v in lab.items() if not v["passed"]],
                    "typed_focal_failed": [c for c, v in foc.items() if not v["passed"]],
                    "identity_failed": sorted({c for k in g if k.startswith("identity_vs_")
                                               for c, v in g[k].items() if not v["passed"]}),
                    "geo_reid_runs": sorted({f"{c}:{r}" for c, v in (g["geo_reid"] or {}).items() for r in ROLES
                                             if v[r]["runs"]}),
                    "all_core_passed": bool(all(passes))}
    return g


def gates(segs: list[str] | None = None) -> dict:
    segs = segs or list(ff.SEG_IDS)
    R = TD("38")
    tds = {s: TD(s) for s in segs}
    out = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "owners": OWNERS, "danced_span": DANCED, "segments": {}}
    prev = None
    for s in segs:
        out["segments"][s] = segment_gates(tds[s], R, prev if s in ("c", "d") else None)
        prev = tds[s]
    # per 10 s block over the whole dance (owner of the block centre; the 38 s span is larissa_kadu_38's own data)
    allc = sorted({c for d in [R, *tds.values()] for c in d.cams})
    src = {"38": R, **tds}
    blocks = []
    for lo in np.arange(-10.0, 190.0, 10.0):
        hi = lo + 10.0
        a, b = max(lo, DANCED[0]), min(hi, DANCED[1])
        if a >= b:
            continue
        row = {"block": [float(lo), float(hi)], "cams": {}}
        for k, ka, kb in OWNERS:
            if k not in src:
                continue
            sa, sb = max(a, ka), min(b, kb)
            if sa >= sb:
                continue
            D = src[k]
            for c in D.cams:
                st = block_stats(D, c, sa, sb)
                if st is None:
                    continue
                prev_st = row["cams"].get(c)
                if prev_st is None or prev_st["frames"] < st["frames"]:
                    row["cams"][c] = {**st, "from": k}
        good = [c for c, v in row["cams"].items() if (v.get("loc_ok") or 0) >= 0.5
                and min(v.get("lead_mask", 0), v.get("follow_mask", 0)) >= 0.5]
        row["cams_with_pose_and_both_dancers"] = sorted(good)
        row["n_cams_with_pose_and_both_dancers"] = len(good)
        blocks.append(row)
    out["blocks_10s"] = blocks
    out["cams"] = allc
    out["vetoes"] = veto_summary(segs)
    out["colour"] = {s: colour_flags(json.loads(q.read_text())) for s in segs
                     if (q := ff.full_work() / "front" / f"colour_{s}.json").exists()}
    p = ff.full_work() / "front" / "gates.json"
    p.write_text(json.dumps(out, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    (ff.full_work() / "front" / "gates.md").write_text(gates_md(out), encoding="utf-8")
    print(f"[gates] -> {p}")
    return out


def gates_md(g: dict) -> str:
    L = ["# larissa_kadu_full front-end gates", ""]
    for s, v in g["segments"].items():
        L += [f"## segment {s} ({v['take']}, window {v['window']})", ""]
        L.append("| cam | loc ok | reproj px | longest fail s | lead mask | follow mask | sync ms (vs 38) | focal rel 38 |")
        L.append("|---|---|---|---|---|---|---|---|")
        for c in v["cams"]:
            lo = v["localisation"].get(c, {})
            la = v["labels"].get(c, {})
            sy = v["sync"].get(c, {})
            fo = v["typed_focal"].get(c, {})
            L.append(f"| {c} | {lo.get('ok_rate')} | {lo.get('reproj_px_median')} | {lo.get('longest_failed_s')} | "
                     f"{la.get('lead')} | {la.get('follow')} | {sy.get('delta_ms')} ({sy.get('delta_vs_38_ms')}) | "
                     f"{fo.get('rel', '')} |")
        L += ["", f"summary: `{json.dumps(v['summary'])}`", ""]
    L += ["## 10 s blocks: cameras with a pose and both dancers masked (>= 50 % of the block's frames)", "",
          "| block s | n | cameras |", "|---|---|---|"]
    for b in g["blocks_10s"]:
        L.append(f"| {b['block'][0]:.0f}-{b['block'][1]:.0f} | {b['n_cams_with_pose_and_both_dancers']} | "
                 f"{' '.join(b['cams_with_pose_and_both_dancers'])} |")
    if g.get("vetoes"):
        L += ["", "## Vetoes applied at the review gate (masks + ma_2d visibilities blanked)", "",
              "| seg | cam | role | frames | fraction | spans (s) |", "|---|---|---|---|---|---|"]
        for s, per in g["vetoes"].items():
            for c, roles in sorted(per.items()):
                for r, v in sorted(roles.items()):
                    sp = ", ".join(f"{a:.1f}-{b:.1f}" for a, b in v["spans_s"])
                    L.append(f"| {s} | {c} | {r} | {v['frames']} | {v['frac']:.1%} | {sp} |")
    if g.get("colour"):
        L += ["", "## Torso colour check (lead = black top, follow = white top), after the vetoes", "",
              "| seg | cam | samples | lead / follow masked | flagged samples | sustained runs (>= 3 samples) |",
              "|---|---|---|---|---|---|"]
        for s, per in g["colour"].items():
            for c, v in per.items():
                sus = "; ".join(f"{x['t0']:.1f}-{x['t1']:.1f} {'+'.join(x['kinds'])}" for x in v["sustained_runs"])
                L.append(f"| {s} | {c} | {v['samples']} | {v['lead']} / {v['follow']} | "
                         f"{sum(v['flagged'].values())} | {sus or 'none'} |")
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------------------------------ images


def grab(src: Path, pts: float, w: int, h: int) -> np.ndarray | None:
    """The decoded frame of src whose PTS is pts (same autorotation / timestamps as ingest), BGR."""
    seek = max(0.0, pts - 1.0)
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{seek:.6f}", "-copyts", "-i", str(src), "-map", "0:v:0", "-vf",
           f"select=gte(t\\,{pts - 0.0015:.6f})", "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    r = subprocess.run(cmd, capture_output=True)
    if len(r.stdout) < w * h * 3:
        return None
    return np.frombuffer(r.stdout[:w * h * 3], np.uint8).reshape(h, w, 3).copy()


def reid_sheet(instants: int = 16, tile_w: int = 150) -> Path:
    import cv2

    from .review import FOLLOW, LEAD, _label, _role_masks, _tint
    src = {k: TD(k) for k, *_ in OWNERS if k == "38" or (PACKAGE_ROOT / "takes" / f"{ff.seg_name(k)}.toml").exists()}
    cams = sorted({c for d in src.values() for c in d.cams})
    ts = np.linspace(DANCED[0] + 0.6, DANCED[1] - 0.6, instants)
    tile_h = int(tile_w * 16 / 9)
    rows = []
    for t in ts:
        k = owner(float(t))
        D = src.get(k)
        tiles = []
        for c in cams:
            tile = np.full((tile_h, tile_w, 3), 40, np.uint8)
            if D is not None and c in D.cams:
                tt = D.t(c)
                i = int(np.argmin(np.abs(tt - t)))
                if abs(tt[i] - t) <= 0.05:
                    m = D.meta(c)
                    w, h = int(m["width"]), int(m["height"])
                    jpg = D.work / "frames" / c / m["frames"][i]["file"]
                    img = cv2.imread(str(jpg)) if jpg.exists() else grab(D.take.source(c).path, m["frames"][i]["pts"], w, h)
                    if img is not None:
                        masks = _role_masks(D.work, c, i)
                        for role, col in (("lead", LEAD), ("follow", FOLLOW)):
                            if masks.get(role) is not None and masks[role].shape == img.shape[:2]:
                                img = _tint(img, masks[role], col, outline=max(2, w // 160))
                        s = min(tile_w / w, tile_h / h)
                        sm = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
                        y0, x0 = (tile_h - sm.shape[0]) // 2, (tile_w - sm.shape[1]) // 2
                        tile[y0:y0 + sm.shape[0], x0:x0 + sm.shape[1]] = sm
                        cam = D.cam(c)
                        okp = cam is None or bool(np.asarray(cam["ok"], bool)[i])
                        _label(tile, f"{c}{'' if okp else ' NOPOSE'}", (4, 16), (255, 255, 255) if okp else (0, 0, 255), 0.45)
                        lm = "L" if masks.get("lead") is not None and masks["lead"].any() else "-"
                        fm = "F" if masks.get("follow") is not None and masks["follow"].any() else "-"
                        _label(tile, lm + fm, (4, tile_h - 8), (255, 255, 255), 0.45)
                    else:
                        _label(tile, f"{c} decode?", (4, 16), (0, 0, 255), 0.45)
                else:
                    _label(tile, f"{c} -", (4, 16), (160, 160, 160), 0.45)
            else:
                _label(tile, f"{c} -", (4, 16), (160, 160, 160), 0.45)
            tiles.append(tile)
        head = np.full((tile_h, 92, 3), 255, np.uint8)
        cv2.putText(head, f"{t:6.1f}s", (4, tile_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(head, f"[{k}]", (4, tile_h // 2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        rows.append(np.hstack([head, *tiles]))
    sheet = np.vstack(rows)
    title = np.full((40, sheet.shape[1], 3), 255, np.uint8)
    cv2.putText(title, f"larissa_kadu_full re-ID: lead red, follow blue; {instants} instants over the dance; source "
                "segment in [ ]; NOPOSE = camera not localised", (6, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                cv2.LINE_AA)
    out = ff.full_work() / "review" / "reid_full.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack([title, sheet]), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"[review] -> {out}")
    return out


def cameras_plot() -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .mamma_dyn import world_cameras
    src = {k: TD(k) for k, *_ in OWNERS if k == "38" or (PACKAGE_ROOT / "takes" / f"{ff.seg_name(k)}.toml").exists()}
    cams = sorted({c for d in src.values() for c in d.cams})
    col = {k: c for k, c in zip(["a", "38", "b", "c", "d"], ["#1f77b4", "#7f7f7f", "#2ca02c", "#d62728", "#9467bd"])}
    fig = plt.figure(figsize=(16, 13))
    ax = fig.add_axes([0.06, 0.55, 0.9, 0.4])
    for y, c in enumerate(cams):
        for k, D in src.items():
            if c not in D.cams:
                continue
            t = D.t(c)
            cam = D.cam(c)
            ok = np.asarray(cam["ok"], bool) if cam is not None else np.zeros(len(t), bool)
            ax.scatter(t[ok], np.full(ok.sum(), y + 0.12), s=1.5, c=col[k], marker="|")
            ax.scatter(t[~ok], np.full((~ok).sum(), y + 0.12), s=4, c="red", marker="|")
            for r, dy, cc in (("lead", -0.12, "#e34a33"), ("follow", -0.24, "#3182bd")):
                bm = np.isfinite(D.boxes(c, r)).all(1)
                ax.scatter(t[bm], np.full(bm.sum(), y + dy), s=0.6, c=cc, marker="|")
    for k, a, b in OWNERS:
        ax.axvspan(max(a, -5), min(b, 190), color=col[k], alpha=0.06)
        ax.text((max(a, -5) + min(b, 190)) / 2, len(cams) - 0.4, k, ha="center", fontsize=10, color=col[k])
    ax.set_yticks(range(len(cams)), [f"cam {c}" for c in cams])
    ax.set_xlim(-5, 190)
    ax.set_ylim(-0.6, len(cams))
    ax.axvline(DANCED[0], color="k", lw=0.6, ls=":")
    ax.axvline(DANCED[1], color="k", lw=0.6, ls=":")
    ax.set_xlabel("reference time (s)")
    ax.set_title("per frame: camera pose localised (segment colour; RED = not localised); lead / follow mask present "
                 "(red / blue rows below); segments shaded", fontsize=10)
    # focal over time
    ax2 = fig.add_axes([0.06, 0.32, 0.9, 0.17])
    for i, c in enumerate(cams):
        for k, D in src.items():
            if c not in D.cams or D.cam(c) is None:
                continue
            cam = D.cam(c)
            ok = np.asarray(cam["ok"], bool)
            mx = max(int(D.meta(c)["width"]), int(D.meta(c)["height"]))
            ax2.plot(D.t(c)[ok], np.asarray(cam["K"])[ok, 0, 0] / mx, lw=0.8, color=plt.cm.tab10(i % 10),
                     label=f"cam {c}" if k == "38" or (k == "a" and c not in src["38"].cams) else None)
    ax2.set_xlim(-5, 190)
    ax2.set_ylabel("focal / max dim")
    ax2.legend(fontsize=7, ncol=8, loc="upper left")
    ax2.set_title("focal length over time (zoom cameras 04 05 07 09 vary; 01 02 03 06 one focal per video)", fontsize=10)
    # top view of camera centres (world, metres)
    ax3 = fig.add_axes([0.06, 0.03, 0.4, 0.25])
    for i, c in enumerate(cams):
        for k, D in src.items():
            if c not in D.cams or D.cam(c) is None:
                continue
            cam = D.cam(c)
            sim3 = json.loads((D.work / "cameras" / "world_from_sfm.json").read_text())
            R, tv = world_cameras(np.asarray(cam["R"], float), np.asarray(cam["tvec"], float), sim3)
            C = -np.einsum("fji,fj->fi", R, tv)
            ok = np.asarray(cam["ok"], bool)
            ax3.scatter(C[ok, 0], C[ok, 2], s=1, color=plt.cm.tab10(i % 10), alpha=0.5)
            if k == "38":
                ax3.annotate(c, (np.median(C[ok, 0]), np.median(C[ok, 2])), fontsize=8)
    ax3.set_aspect("equal")
    ax3.set_xlabel("world x (m)")
    ax3.set_ylabel("world z (m)")
    ax3.set_title("camera centres, top view (all segments + 38)", fontsize=10)
    # n cameras with pose + both dancers over time (2 s bins)
    ax4 = fig.add_axes([0.53, 0.03, 0.43, 0.25])
    bins = np.arange(-4, 188, 2.0)
    n = np.zeros(len(bins))
    for j, a in enumerate(bins):
        k = owner(a + 1.0)
        D = src.get(k)
        if D is None:
            continue
        for c in D.cams:
            st = block_stats(D, c, a, a + 2.0)
            if st and (st.get("loc_ok") or 0) >= 0.5 and min(st["lead_mask"], st["follow_mask"]) >= 0.5:
                n[j] += 1
    ax4.step(bins, n, where="post")
    ax4.set_xlim(-5, 190)
    ax4.set_ylim(0, 9)
    ax4.set_ylabel("cameras")
    ax4.set_xlabel("reference time (s)")
    ax4.set_title("cameras with a pose and both dancers masked (2 s bins, >= 50 % of frames)", fontsize=10)
    out = ff.full_work() / "review" / "cameras_coverage_full.png"
    fig.savefig(out, dpi=90)
    plt.close(fig)
    print(f"[review] -> {out}")
    return out


def sync_plot() -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    src = {k: TD(k) for k, *_ in OWNERS if k == "38" or (PACKAGE_ROOT / "takes" / f"{ff.seg_name(k)}.toml").exists()}
    cams = sorted({c for d in src.values() for c in d.cams})
    ref = (src["38"].json("sync/offsets.json") or {}).get("offsets", {})
    fig, ax = plt.subplots(figsize=(13, 5.5))
    keys = ["a", "38", "b", "c", "d"]
    for i, c in enumerate(cams):
        r = ref.get(c, {}).get("delta_ms")
        if r is not None:
            ax.add_patch(plt.Rectangle((i - 0.42, r - 15), 0.84, 30, color="#cccccc", alpha=0.5, lw=0))
        for j, k in enumerate(keys):
            D = src.get(k)
            if D is None:
                continue
            o = (D.json("sync/offsets.json") or {}).get("offsets", {}).get(c)
            if not o:
                continue
            x = i - 0.32 + 0.16 * j
            mk = "o" if o.get("stable") else "s"
            ax.plot(x, o["delta_ms"], mk, color=plt.cm.tab10(j), ms=7, mfc="none" if not o.get("use", True) else None,
                    label=k if i == 0 else None)
            for sw in o.get("sub_windows") or []:
                if sw.get("delta_ms") is not None:
                    ax.plot(x, sw["delta_ms"], ".", color=plt.cm.tab10(j), ms=2, alpha=0.5)
    ax.set_xticks(range(len(cams)), [f"cam {c}" for c in cams])
    ax.set_ylabel("video offset delta (ms): corrected t = t + delta")
    ax.set_title("sync per segment (gauge pinned to larissa_kadu_38); grey band = 38's offset +-15 ms; circles stable, "
                 "squares unstable (2 s sub-window spread >= 10 ms), dots = sub-windows", fontsize=9)
    ax.legend(title="segment", fontsize=8)
    ax.grid(alpha=0.3)
    out = ff.full_work() / "review" / "sync_offsets_full.png"
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print(f"[review] -> {out}")
    return out


# ------------------------------------------------------------------------------------------------ identity audit + vetoes

# Spans decided by looking at the frames (review/identity/*.jpg, checked 2026-10-08): masks on the wrong people.
MANUAL_VETOES = [
    # seg, cam, roles, t0, t1 (reference time, frames.json t), why
    ("a", "09", ("lead",), -10.0, -0.35, "lead absent: a standing spectator at the right edge carries the lead mask "
                                         "(-3.1 to -1.9 s; no lead mask after) until Kadu walks in at -0.1 s "
                                         "(identity/a_09_entry.jpg)"),
    ("a", "07", ("lead", "follow"), -10.0, 31.22, "cam07 30.0-31.2 s: masks on seated foreground spectators (as "
                                                  "vetoed in larissa_kadu_38; identity/a_07.jpg)"),
    ("a", "07", ("lead", "follow"), 33.70, 1e9, "cam07 33.7-36.2 s: masks on seated foreground spectators (as vetoed "
                                                "in larissa_kadu_38; identity/a_07.jpg)"),
    ("a", "06", ("lead", "follow"), 1.56, 14.66, "cam06 1.6-14.7 s: the follow mask sits on Kadu (black top; torso "
                                                 "colour check front/colour_a.json) and the lead mask on a standing "
                                                 "spectator; Larissa unmasked (identity/a_06_*.jpg)"),
    ("a", "06", ("lead",), 30.90, 32.82, "cam06 30.9-32.8 s: the lead mask jumps to a standing spectator at the right edge "
                                         "while Kadu dances with Larissa (larissa_kadu_38 has it on Kadu there; the "
                                         "camera is not localised; gate_identity/cases_a06_fine.jpg)"),
    # gate_identity 2026-10-08: gaps of <= 16 native frames left between two vetoed spans of the same camera and role
    # (the geometric spans are bridged only up to 6 grid instants); the mask on both sides is on the wrong person
    ("a", "05", ("lead",), 21.48, 21.70, "gap between two vetoed spans (identity audit)"),
    ("a", "05", ("lead",), 23.01, 23.03, "gap between two vetoed spans (identity audit)"),
    ("b", "05", ("follow",), 106.05, 106.11, "gap between two vetoed spans (identity audit)"),
    ("b", "06", ("lead",), 84.78, 85.30, "gap between two vetoed spans (identity audit)"),
    ("b", "06", ("lead",), 87.96, 88.11, "gap between two vetoed spans (identity audit)"),
    ("c", "07", ("follow",), 117.84, 118.03, "cam07 117.0-119.1 s: the follow mask is on a spectator at the left edge while "
                                             "Larissa is unmasked; the gap between the two geometric spans is filled "
                                             "(gate_identity sheet_all_07)"),
    ("b", "07", ("lead",), 67.05, 72.4, "lead mask on a standing spectator at the left edge (identity/b_07.jpg)"),
    ("b", "02", ("follow",), 90.47, 97.26, "follow mask on Kadu (90.5 s), a standing spectator at the right edge "
                                           "(90.6-92.2 s), then on Kadu again (92.3-97.2 s) while Larissa is unmasked "
                                           "(identity/b_02_*.jpg)"),
    ("b", "02", ("lead",), 90.47, 90.56, "lead mask on a spectator for 3 frames (identity/b_02_start.jpg)"),
    ("b", "03", ("lead",), 95.3, 103.8, "lead mask on a seated spectator (red top) right of centre while Kadu is "
                                        "unmasked (identity/b_03.jpg)"),
    ("b", "06", ("lead",), 90.2, 93.9, "lead mask on a standing spectator at the back (continues the geometric veto "
                                       "83.6-90.2 s; identity/b_06.jpg)"),
    ("c", "03", ("lead",), -10.0, 103.63, "cam03 103.28-103.61 s: the lead mask sits on a crouching spectator (right of "
                                          "centre) at the start of the segment while Kadu is unmasked; it jumps to Kadu "
                                          "at 103.64 s (gate_identity/cases_b_overlap.jpg)"),
    ("c", "05", ("follow",), 103.81, 109.45, "follow mask on a standing spectator at the right edge (a 25 x 50 px sliver "
                                             "from 103.83 s, Larissa unmasked; gate_identity/cases_c05_start.jpg); Larissa hidden "
                                            "behind Kadu (camera partly not localised; identity/c_05.jpg)"),
    ("c", "05", ("follow",), 113.9, 114.2, "follow mask on a standing spectator at the left edge (identity/c_05.jpg)"),
    ("c", "05", ("lead",), 115.9, 118.2, "Kadu unmasked, the lead mask elsewhere (identity/c_05.jpg)"),
    ("c", "07", ("lead", "follow"), 136.45, 1e9, "cam07 zooms out to its ultra-wide lens at 136.5 s: both masks jump "
                                                 "to seated foreground spectators until the camera stops "
                                                 "(identity/c_07_switch.jpg, c_07_zoomout.jpg)"),
    ("d", "07", ("lead", "follow"), -10.0, 1e9, "cam07 142-159 s (ultra-wide): both masks on seated foreground "
                                                "spectators; the dancers are ~70 px tall in the background "
                                                "(identity/d_07.jpg)"),
]
T_ID_PX, T_REF_PX, T_POSE_OTHER_PX, MIN_ID_INSTANTS, MIN_POSE_S = 100.0, 40.0, 60.0, 6, 1.0


def _costs(seg: str, refresh: bool = False) -> dict:
    """role_costs on the union coverage (dancers_geoveto), cached in dancers/geo_costs_union.npz."""
    from .dancers_geoveto import role_costs
    D = TD(seg)
    p = D.work / "dancers" / "geo_costs_union.npz"
    cams = sorted(q.stem for q in (D.work / "cameras").glob("[0-9][0-9].npz"))
    if p.exists() and not refresh:
        z = np.load(p)
        return {c: {k.split("|", 1)[1]: z[k] for k in z.files if k.startswith(c + "|")} for c in cams}
    costs = role_costs(D.work, cams, union=True)
    np.savez(p, **{f"{c}|{k}": np.asarray(v) for c, d in costs.items() for k, v in d.items()})
    return costs


def _spans(flag: np.ndarray, t: np.ndarray, gap: int = 6, min_n: int = 6, dilate: int = 3) -> list:
    runs, s = [], None
    for k, f in enumerate(np.r_[flag, False]):
        if f and s is None:
            s = k
        elif not f and s is not None:
            if runs and s - runs[-1][1] <= gap:
                runs[-1][1] = k
            else:
                runs.append([s, k])
            s = None
    return [(max(0, a - dilate), min(len(t), b + dilate)) for a, b in runs if int(flag[a:b].sum()) >= min_n]


def identity_audit(segs: list[str] | None = None, apply: bool = False, refresh: bool = False,
                   manual_only: bool = False) -> dict:
    """Per segment, camera and role, on the union coverage of the cameras:
      identity  the role's keypoints are explained by NEITHER 3D dancer (> T_ID_PX to both) while the 3D reference is
                trustworthy (>= 2 other main cameras fit that role within T_REF_PX) and the camera's other role fits
                (< T_POSE_OTHER_PX: the camera pose is fine) -> that role is vetoed;
      pose      both roles explained by neither dancer while the references agree, for >= MIN_POSE_S -> the camera
                pose is wrong there (an attached camera whose operator moved) -> both roles vetoed;
      manual    MANUAL_VETOES (frames looked at).
    A swap (role explained by the partner) is reported by the gates, not vetoed (embraces: dancers_geoveto.decide).
    Writes front/veto_plan.json (apply=False) or front/veto_applied.json; apply=True blanks the spans through
    dancers_geoveto.apply (masks + ma_2d_native visibilities, originals backed up; idempotent). Re-run `mamma prepare
    --skip-2d` afterwards. manual_only: MANUAL_VETOES alone, no geometry (apply these FIRST: a wrong-person camera
    corrupts the 3D references every other camera is judged against; then audit --refresh)."""
    from .dancers_geoveto import _tracks
    from .dancers_geoveto import apply as geo_apply
    segs = segs or list(ff.SEG_IDS)
    doc: dict = {"rule": {"identity_px": T_ID_PX, "reference_px": T_REF_PX, "pose_other_role_px": T_POSE_OTHER_PX,
                          "min_identity_instants": MIN_ID_INSTANTS, "min_pose_s": MIN_POSE_S},
                 "manual": [list(m) for m in MANUAL_VETOES], "segments": {}}
    for seg in segs:
        D = TD(seg)
        costs = {} if manual_only else _costs(seg, refresh)
        cams = sorted(costs) if costs else sorted(q.stem for q in (D.work / "cameras").glob("[0-9][0-9].npz"))
        tr = _tracks(D.work, cams)
        main = [c for c in cams if not tr[c]["attached"]]
        spans: list[dict] = []
        swaps: list[dict] = []

        def refs(c, r, n):
            k = np.zeros(n, int)
            for c2 in main:
                if c2 != c:
                    k += (np.nan_to_num(costs[c2][r], nan=np.inf) < T_REF_PX).astype(int)
            return k
        for c in (cams if costs else []):
            # report only (not vetoed): the role sits on the PARTNER (far from its own 3D dancer, on the other one)
            # while the references are trustworthy - a swap, or one mask covering both dancers in an embrace
            t = costs[c]["t"]
            for r in ROLES:
                x = np.nan_to_num(costs[c][r], nan=0.0)
                xo = np.nan_to_num(costs[c][f"{r}_other"], nan=np.inf)
                on_partner = (x > T_ID_PX) & (xo < T_REF_PX) & (refs(c, r, len(t)) >= 2)
                for a, b in _spans(on_partner, t, min_n=MIN_ID_INSTANTS):
                    swaps.append({"cam": c, "role": r, "t0": round(float(t[a]), 3), "t1": round(float(t[b - 1]), 3),
                                  "median_px": round(float(np.median(x[a:b])), 1),
                                  "median_px_to_partner": round(float(np.median(xo[a:b])), 1)})
        for c in (cams if costs else []):
            t = costs[c]["t"]
            wrong = {}
            for r in ROLES:
                x = np.nan_to_num(costs[c][r], nan=0.0)
                xo = np.nan_to_num(costs[c][f"{r}_other"], nan=np.inf)
                wrong[r] = (x > T_ID_PX) & (xo > T_ID_PX) & (refs(c, r, len(t)) >= 2)
            for ri, r in enumerate(ROLES):
                o = ROLES[1 - ri]
                x = np.nan_to_num(costs[c][r], nan=0.0)
                other_fit = np.nan_to_num(costs[c][o], nan=np.inf) < T_POSE_OTHER_PX
                for a, b in _spans(wrong[r] & other_fit, t, min_n=MIN_ID_INSTANTS):
                    spans.append({"cam": c, "roles": [r], "kind": "identity", "t0": round(float(t[a]), 3),
                                  "t1": round(float(t[b - 1]), 3), "median_px": round(float(np.median(x[a:b])), 1)})
            both = wrong["lead"] & wrong["follow"]
            for a, b in _spans(both, t, gap=15, min_n=int(MIN_POSE_S * 30)):
                xl, xf = np.nan_to_num(costs[c]["lead"]), np.nan_to_num(costs[c]["follow"])
                spans.append({"cam": c, "roles": list(ROLES), "kind": "pose", "t0": round(float(t[a]), 3),
                              "t1": round(float(t[b - 1]), 3),
                              "median_px": round(float(np.median(np.r_[xl[a:b], xf[a:b]])), 1)})
        for s, c, roles, t0, t1, why in MANUAL_VETOES:
            if s == seg and c in cams:
                spans.append({"cam": c, "roles": list(roles), "kind": "manual", "t0": t0, "t1": t1, "why": why})
        # native frames: geometric spans are on the sync-corrected grid (_tracks t), manual ones on frames.json t
        plan: dict = {}
        for sp in spans:
            c = sp["cam"]
            tt = tr[c]["t"] if sp["kind"] != "manual" else D.t(c)
            idx = np.flatnonzero((tt >= sp["t0"] - 1 / 60) & (tt <= sp["t1"] + 1 / 60))
            sp["native_frames"] = [int(idx[0]), int(idx[-1])] if len(idx) else None
            sp["n_frames"] = int(len(idx))
            for r in sp["roles"]:
                plan.setdefault(c, {}).setdefault(r, set()).update(int(i) for i in idx)
        plan = {c: {r: sorted(v) for r, v in per.items()} for c, per in plan.items()}
        n_frames = {c: len(D.t(c)) for c in cams}
        doc["segments"][seg] = {
            "take": D.take.name, "spans": spans, "swap_candidates": swaps,
            "vetoed_fraction": {c: {r: round(len(plan.get(c, {}).get(r, [])) / max(1, n_frames[c]), 3) for r in ROLES}
                                for c in cams}}
        if apply:
            geo_apply(D.work, plan, source="front_full_review identity_audit (" + (
                "manual review)" if manual_only else "union geometry + manual review)"))
            doc["segments"][seg]["applied"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        print(f"[audit] {seg}: {len(spans)} spans: " + "; ".join(
            f"{sp['cam']}:{'+'.join(sp['roles'])}:{sp['kind']} {sp['t0']}-{sp['t1']} ({sp['n_frames']} fr)"
            for sp in spans), flush=True)
        if swaps:
            print(f"[audit] {seg}: role on the partner (report only): " + "; ".join(
                f"{s['cam']}:{s['role']} {s['t0']}-{s['t1']} ({s['median_px']} px own, {s['median_px_to_partner']} px "
                f"partner)" for s in swaps), flush=True)
    name = "veto_manual" if manual_only else "veto"
    p = ff.full_work() / "front" / (f"{name}_applied.json" if apply else f"{name}_plan.json")
    p.write_text(json.dumps(doc, indent=1))
    print(f"[audit] -> {p}")
    return doc


# ------------------------------------------------------------------------------------------------ torso colour check

COLOUR_FOLLOW_DARK = (0.40, 90.0)  # follow torso: dark fraction > 0.40 and median V < 90 -> on a dark-top person
COLOUR_LEAD_BRIGHT = (0.25, 140.0)  # lead torso: white fraction > 0.25 or median V > 140 -> on a light-top person
COLOUR_FLIP_V = 30.0  # both present and the lead's torso brighter than the follow's by > 30 (V, 0-255)


def _torso_stats(img: np.ndarray, m: np.ndarray | None) -> dict | None:
    """HSV statistics of the torso band (17-42 % down the mask's height) of one dancer mask."""
    import cv2
    if m is None or m.sum() < 400:
        return None
    ys, xs = np.nonzero(m)
    y0, y1 = ys.min(), ys.max()
    h = y1 - y0
    band = (ys >= y0 + 0.17 * h) & (ys <= y0 + 0.42 * h)
    if band.sum() < 100:
        return None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[ys[band], xs[band], 2].astype(float)
    s = hsv[ys[band], xs[band], 1].astype(float)
    return {"v": float(np.median(v)), "dark": float((v < 70).mean()), "white": float(((v > 165) & (s < 70)).mean()),
            "px": int(m.sum())}


def colour_check(segs: list[str] | None = None, dt: float = 0.5, cams: list[str] | None = None) -> dict:
    """Every dt s of every camera: torso colour of the lead and follow masks (body masks, else full masks) on the
    decoded frame (ONE ffmpeg pass per camera, frames selected by PTS) -> front/colour_<seg>.json."""
    import cv2
    out_all = {}
    for seg in segs or list(ff.SEG_IDS):
        D = TD(seg)
        p_out = ff.full_work() / "front" / f"colour_{seg}.json"
        out = json.loads(p_out.read_text()) if p_out.exists() and cams else {}
        for c in sorted(D.cams):
            if cams and c not in cams:
                continue
            m = D.meta(c)
            w, h = int(m["width"]), int(m["height"])
            tt = D.t(c)
            idx = sorted({int(np.argmin(np.abs(tt - t))) for t in np.arange(tt[0], tt[-1], dt)})
            pts = [float(m["frames"][i]["pts"]) for i in idx]
            expr = "+".join(f"lt(abs(t-{q:.6f})\\,0.0015)" for q in pts)
            cmd = ["ffmpeg", "-v", "error", "-ss", f"{max(0.0, pts[0] - 1.0):.6f}", "-copyts", "-i",
                   str(D.take.source(c).path), "-map", "0:v:0", "-vf", f"select={expr}", "-fps_mode", "passthrough",
                   "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
            r = subprocess.run(cmd, capture_output=True)
            n = len(r.stdout) // (w * h * 3)
            if n != len(idx):
                print(f"[colour] {seg} {c}: {n} frames decoded for {len(idx)} samples", flush=True)
            rows = []
            for k, i in enumerate(idx[:n]):
                img = np.frombuffer(r.stdout[k * w * h * 3:(k + 1) * w * h * 3], np.uint8).reshape(h, w, 3)
                row = {"t": round(float(tt[i]), 3), "i": i}
                for role, pid in (("lead", 1), ("follow", 2)):
                    mk = None
                    for sub in ("body", "masks"):
                        f = D.work / "dancers" / c / sub / f"mask_{i:04d}_{pid:02d}.png"
                        a = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) if f.exists() else None
                        if a is not None and (a > 127).any():
                            mk = a > 127
                            break
                    row[role] = _torso_stats(img, mk) if mk is not None and mk.shape == img.shape[:2] else None
                rows.append(row)
            out[c] = rows
            print(f"[colour] {seg} {c}: {len(rows)} samples", flush=True)
        p_out.write_text(json.dumps(out))
        out_all[seg] = colour_flags(out)
    return out_all


def colour_flags(doc: dict) -> dict:
    """{cam: {samples, lead, follow, flagged {kind: n}, runs [{t0, t1, n, kinds}]}} (runs: flagged samples <= 1 s apart)."""
    res = {}
    for c, rows in sorted(doc.items()):
        runs: list[dict] = []
        kinds: dict = {}
        for r in rows:
            L, F = r["lead"], r["follow"]
            f = []
            if F and F["dark"] > COLOUR_FOLLOW_DARK[0] and F["v"] < COLOUR_FOLLOW_DARK[1]:
                f.append("follow_dark")
            if L and (L["white"] > COLOUR_LEAD_BRIGHT[0] or L["v"] > COLOUR_LEAD_BRIGHT[1]):
                f.append("lead_bright")
            if L and F and L["v"] > F["v"] + COLOUR_FLIP_V:
                f.append("flip")
            for k in f:
                kinds[k] = kinds.get(k, 0) + 1
            if f:
                if runs and r["t"] - runs[-1]["t1"] <= 1.01:
                    runs[-1]["t1"] = r["t"]
                    runs[-1]["n"] += 1
                    runs[-1]["kinds"] = sorted(set(runs[-1]["kinds"]) | set(f))
                else:
                    runs.append({"t0": r["t"], "t1": r["t"], "n": 1, "kinds": f})
        res[c] = {"samples": len(rows), "lead": sum(1 for r in rows if r["lead"]),
                  "follow": sum(1 for r in rows if r["follow"]), "flagged": kinds, "runs": runs,
                  "sustained_runs": [x for x in runs if x["n"] >= 3]}
    return res


def veto_summary(segs: list[str] | None = None) -> dict:
    """{seg: {cam: {role: {"frames": n, "frac": n / camera frames, "spans_s": [[t0, t1], ...]}}}} of every veto applied
    (roles.json frames flagged geo_veto)."""
    from .dancers_geoveto import applied_frames
    out = {}
    for seg in segs or list(ff.SEG_IDS):
        D = TD(seg)
        af = applied_frames(D.work)
        per = {}
        for c, roles in af.items():
            t = D.t(c)
            per[c] = {}
            for r, fr in roles.items():
                fr = np.asarray(fr, int)
                cut = np.flatnonzero(np.diff(fr) > 3) + 1
                spans = [[round(float(t[g[0]]), 2), round(float(t[g[-1]]), 2)] for g in np.split(fr, cut) if len(g)]
                per[c][r] = {"frames": int(len(fr)), "frac": round(len(fr) / max(1, len(t)), 3), "spans_s": spans}
        out[seg] = per
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m dancecap.front_full_review", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["gates", "images", "all", "audit", "apply-vetoes", "colour"])
    ap.add_argument("--refresh", action="store_true", help="audit: recompute the union geometry costs")
    ap.add_argument("--manual-only", action="store_true", help="audit / apply-vetoes: MANUAL_VETOES only (first)")
    ap.add_argument("--instants", type=int, default=16)
    ap.add_argument("--dt", type=float, default=0.5, help="colour: sample spacing (s)")
    ap.add_argument("--cams", help="colour: comma-separated cameras (default all)")
    ap.add_argument("--segs")
    a = ap.parse_args(argv)
    segs = a.segs.split(",") if a.segs else None
    if a.action in ("audit", "apply-vetoes"):
        identity_audit(segs, apply=a.action == "apply-vetoes", refresh=a.refresh, manual_only=a.manual_only)
        return
    if a.action == "colour":
        res = colour_check(segs, a.dt, a.cams.split(",") if a.cams else None)
        for s, per in res.items():
            for c, v in per.items():
                print(f"[colour] {s}/{c}: {v['samples']} samples, flagged {v['flagged']}, sustained "
                      f"{[(x['t0'], x['t1'], x['kinds']) for x in v['sustained_runs']]}")
        return
    if a.action in ("gates", "all"):
        gates(segs)
    if a.action in ("images", "all"):
        cameras_plot()
        sync_plot()
        reid_sheet(a.instants)


if __name__ == "__main__":
    main()
