"""Continuity of a chunked MAMMA fit at its seams: numbers + one review sheet (local, numpy/matplotlib only).

    python -m dancecap.mamma_seams larissa_kadu_38      -> work/<take>/mamma/chunks_seams.json
                                                           work/<take>/review/pose_seams.jpg

`mamma_chunks merge` reports how much two neighbouring chunks DISAGREE inside their overlap (seam_mm). This module
checks what the merged fit DOES there: whether the cosine cross-fade added motion that neither chunk has.

Per role and seam (blend window = the central half of the overlap, where the weight ramps 0 -> 1):
  disagree_mm      the two chunks' joint distance over the overlap (median of 55 joints / pelvis), per frame
  pelvis_acc       max pelvis acceleration (m/s^2) in the blend window: merged vs each chunk over the same frames
  body_acc         the same for the pelvis-relative body joints (pose only, mean over 21 body joints)
  speed_step       max frame-to-frame change of pelvis speed (m/s) in the window: merged vs each chunk
  smooth           all three merged window maxima <= 1.25 x the larger chunk's over the same frames (the cross-fade
                   added no acceleration or speed jump of its own); the merged peak vs the take's p99 outside the
                   windows is reported for context only (a seam that lands on a real acceleration peak is no defect)
Take-wide: the 5 largest merged pelvis accelerations and whether they fall in a blend window (a seam pop) or not.

If the gate's full-fit fallback replaced the merge (no chunk_ranges in the merged npz), the sheet compares each
chunk with the single full fit instead; the numbers keep the same names.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .take import PACKAGE_ROOT

ROLES = ("lead", "follow")
ROLE_COLOR = {"lead": "#e34948", "follow": "#2a78d6"}  # red / blue, as review.py and the QA sheets
CHUNK_GRAY = {"prev": "#52514e", "next": "#8a8984"}
BODY = list(range(1, 22))  # SMPL-X body joints after the pelvis
HZ = 30.0


def _blend_window(a_next: int, b_prev: int) -> tuple[int, int]:
    """Grid frames [lo, hi) where the later chunk's weight ramps (central half of the overlap a_next..b_prev)."""
    ov = b_prev - a_next
    return a_next + ov // 4, a_next + ov - ov // 4


def _acc(x: np.ndarray) -> np.ndarray:
    """|second difference| * HZ^2 per frame (T,), NaN at both ends; x (T, 3) or (T, J, 3) -> mean over J."""
    a = np.full(len(x), np.nan)
    n = np.linalg.norm(x[2:] - 2 * x[1:-1] + x[:-2], axis=-1) * HZ * HZ
    a[1:-1] = n if n.ndim == 1 else n.mean(axis=1)
    return a


def _speed(x: np.ndarray) -> np.ndarray:
    v = np.full(len(x), np.nan)
    v[1:] = np.linalg.norm(np.diff(x, axis=0), axis=-1) * HZ
    return v


def _local(j: np.ndarray) -> np.ndarray:
    """Body joints relative to the pelvis (pose only, root translation removed)."""
    return j[:, BODY] - j[:, :1]


def analyse(take: str) -> dict:
    work = PACKAGE_ROOT / "work" / take
    root = work / "mamma"
    doc = json.loads((root / "chunks.json").read_text())
    tag, T, chunks = doc["tag"], int(doc["T"]), doc["chunks"]
    merged_dir = work / "smplx" if (work / "smplx" / "lead.npz").exists() else root / "fits" / tag / "smplx"
    out = {"take": take, "tag": tag, "T": T, "merged_from": str(merged_dir.relative_to(work)).replace("\\", "/"),
           "seams": [], "roles": {}}
    data = {}
    for role in ROLES:
        m = dict(np.load(merged_dir / f"{role}.npz", allow_pickle=True))
        if len(m["t"]) != T:
            raise SystemExit(f"[seams] {merged_dir}/{role}.npz has {len(m['t'])} frames, plan {T}")
        cj = [np.asarray(np.load(root / "fits" / c["run_tag"] / "smplx" / f"{role}.npz", allow_pickle=True)["joints"],
                         float)[:, :55] for c in chunks]
        data[role] = {"t": np.asarray(m["t"], float), "J": np.asarray(m["joints"], float)[:, :55], "chunks": cj,
                      "merged_is_chunked": "chunk_ranges" in m}
    out["merged_is_chunked"] = bool(data["lead"]["merged_is_chunked"])
    t = data["lead"]["t"]
    windows = [_blend_window(chunks[k]["a"], chunks[k - 1]["b"]) for k in range(1, len(chunks))]
    in_win = np.zeros(T, bool)
    for lo, hi in windows:
        in_win[lo:hi] = True
    for role in ROLES:
        J = data[role]["J"]
        pel, loc = J[:, 0], _local(J)
        acc_p, acc_b, sp = _acc(pel), _acc(loc), _speed(pel)
        dsp = np.full(T, np.nan)
        dsp[1:] = np.abs(np.diff(sp))
        outside = ~in_win
        base = {"pelvis_acc_p95": float(np.nanpercentile(acc_p[outside], 95)),
                "pelvis_acc_p99": float(np.nanpercentile(acc_p[outside], 99)),
                "body_acc_p99": float(np.nanpercentile(acc_b[outside], 99)),
                "speed_step_p99": float(np.nanpercentile(dsp[outside], 99))}
        top = np.argsort(np.nan_to_num(acc_p, nan=-1))[::-1][:5]
        rr = {"take_outside_windows": {k: round(v, 3) for k, v in base.items()},
              "top_pelvis_acc": [{"frame": int(i), "t": round(float(t[i]), 3), "acc": round(float(acc_p[i]), 2),
                                  "in_blend_window": bool(in_win[i])} for i in top],
              "seams": []}
        for k in range(1, len(chunks)):
            a, b = chunks[k]["a"], chunks[k - 1]["b"]
            lo, hi = windows[k - 1]
            jp = data[role]["chunks"][k - 1][a - chunks[k - 1]["a"]:]  # previous chunk over the overlap
            jn = data[role]["chunks"][k][:b - a]                      # next chunk over the overlap
            d = np.linalg.norm(jp - jn, axis=-1) * 1000
            # each chunk's own motion over the blend window (window inside the overlap: both chunks cover it)
            sl_p = slice(lo - chunks[k - 1]["a"], hi - chunks[k - 1]["a"])
            sl_n = slice(lo - chunks[k]["a"], hi - chunks[k]["a"])

            def own(jc, sl):
                ap, ab, sc = _acc(jc[:, 0]), _acc(_local(jc)), _speed(jc[:, 0])
                ds = np.full(len(sc), np.nan)
                ds[1:] = np.abs(np.diff(sc))
                return float(np.nanmax(ap[sl])), float(np.nanmax(ab[sl])), float(np.nanmax(ds[sl]))
            pp, pb, ps = own(data[role]["chunks"][k - 1], sl_p)
            np_, nb, ns = own(data[role]["chunks"][k], sl_n)
            mp, mb = float(np.nanmax(acc_p[lo:hi])), float(np.nanmax(acc_b[lo:hi]))
            ms = float(np.nanmax(dsp[lo:hi]))
            ratio_p, ratio_b = mp / max(pp, np_, 1e-9), mb / max(pb, nb, 1e-9)
            ratio_s = ms / max(ps, ns, 1e-9)
            # smooth = the cross-fade added no acceleration / speed jump beyond 25% of what the chunks themselves have
            # over the same frames (a seam at a natural acceleration peak is not a defect)
            smooth = bool(ratio_p <= 1.25 and ratio_b <= 1.25 and ratio_s <= 1.25)
            rr["seams"].append({
                "between": [chunks[k - 1]["run_tag"], chunks[k]["run_tag"]], "overlap_frames": [a, b],
                "blend_frames": [lo, hi], "t_blend": [round(float(t[lo]), 3), round(float(t[hi - 1]), 3)],
                "disagree_mm": {"median": round(float(np.median(d)), 1), "p95": round(float(np.percentile(d, 95)), 1),
                                "pelvis_median": round(float(np.median(d[:, 0])), 1),
                                "pelvis_max": round(float(d[:, 0].max()), 1)},
                "pelvis_acc": {"merged_max": round(mp, 2), "prev_max": round(pp, 2), "next_max": round(np_, 2),
                               "ratio_vs_chunks": round(ratio_p, 3)},
                "body_acc": {"merged_max": round(mb, 2), "prev_max": round(pb, 2), "next_max": round(nb, 2),
                             "ratio_vs_chunks": round(ratio_b, 3)},
                "speed_step": {"merged_max": round(ms, 3), "prev_max": round(ps, 3), "next_max": round(ns, 3),
                               "ratio_vs_chunks": round(ratio_s, 3)},
                "merged_pelvis_acc_vs_take_p99": round(mp / max(base["pelvis_acc_p99"], 1e-9), 3), "smooth": smooth})
        rr["all_seams_smooth"] = all(s["smooth"] for s in rr["seams"])
        out["roles"][role] = rr
    out["seams"] = [{"between": [chunks[k - 1]["run_tag"], chunks[k]["run_tag"]],
                     "t_blend": out["roles"]["lead"]["seams"][k - 1]["t_blend"]} for k in range(1, len(chunks))]
    out["all_smooth"] = all(out["roles"][r]["all_seams_smooth"] for r in ROLES)
    out["_data"] = data
    out["_windows"] = windows
    out["_chunks"] = chunks
    return out


def sheet(res: dict, path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data, windows, chunks = res["_data"], res["_windows"], res["_chunks"]
    t = data["lead"]["t"]
    nseam = len(windows)
    fig = plt.figure(figsize=(22, 3.2 + 3.3 * nseam), dpi=80, facecolor="#fcfcfb")
    gs = fig.add_gridspec(nseam + 1, 5, height_ratios=[1.0] + [1.0] * nseam, hspace=0.75, wspace=0.38)
    ink, ink2 = "#0b0b0b", "#52514e"

    def style(ax):
        ax.set_facecolor("#fcfcfb")
        ax.grid(True, color="#e6e5e0", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#b5b4ae")
        ax.tick_params(colors=ink2, labelsize=9)
    # overview: pelvis speed over the whole take, blend windows shaded
    ax = fig.add_subplot(gs[0, :])
    style(ax)
    for lo, hi in windows:
        ax.axvspan(t[lo], t[hi - 1], color="#f0efec", zorder=0)
    for role in ROLES:
        ax.plot(t, _speed(data[role]["J"][:, 0]), color=ROLE_COLOR[role], lw=1.6, label=f"{role} pelvis speed")
    ax.set_xlim(t[0], t[-1])
    ax.set_ylabel("m/s", color=ink2)
    ax.set_xlabel("take time (s)", color=ink2)
    mode = "merged fit" if res["merged_is_chunked"] else "FULL-FIT FALLBACK (no seams in the final fit)"
    ax.set_title(f"{res['take']} {res['tag']}: pelvis speed of the {mode}; grey bands = the {nseam} blend windows "
                 f"(cosine cross-fade, central half of each overlap). Seams smooth: "
                 f"{'yes' if res['all_smooth'] else 'NO - see rows'}", loc="left", fontsize=12, color=ink)
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    for k, (lo, hi) in enumerate(windows):
        a, b = chunks[k + 1]["a"], chunks[k]["b"]
        pad = 9
        g0, g1 = max(0, a - pad), min(len(t), b + pad)
        tt = t[g0:g1]
        for c, role in enumerate(ROLES):
            J = data[role]["J"]
            prev = data[role]["chunks"][k]
            nxt = data[role]["chunks"][k + 1]
            pg = np.arange(chunks[k]["a"], chunks[k]["b"])
            ng = np.arange(chunks[k + 1]["a"], chunks[k + 1]["b"])
            sel_p = (pg >= g0) & (pg < g1)
            sel_n = (ng >= g0) & (ng < g1)
            s = res["roles"][role]["seams"][k]
            # top view of the pelvis path (x, -z as seen from above; cm about the merged path's mean)
            ax = fig.add_subplot(gs[k + 1, c])
            style(ax)
            c0 = J[g0:g1, 0].mean(0)
            for jj, sel, key, ls in ((prev, sel_p, "prev", "--"), (nxt, sel_n, "next", ":")):
                q = (jj[sel, 0] - c0) * 100
                ax.plot(q[:, 0], -q[:, 2], ls, color=CHUNK_GRAY[key], lw=1.6, label=f"chunk {key}")
            q = (J[g0:g1, 0] - c0) * 100
            ax.plot(q[:, 0], -q[:, 2], color=ROLE_COLOR[role], lw=2.0, label="merged")
            qb = (J[lo:hi, 0] - c0) * 100
            ax.plot(qb[[0, -1], 0], -qb[[0, -1], 2], "o", ms=6, color=ROLE_COLOR[role], mec="#fcfcfb", mew=1.5)
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_xlabel("x (cm)", color=ink2, fontsize=9)
            ax.set_ylabel("-z (cm)", color=ink2, fontsize=9)
            ax.set_title(f"seam {k + 1} {role}: pelvis from above\nchunks {s['disagree_mm']['pelvis_median']:.0f} mm "
                         "apart (median); dots = blend", loc="left", fontsize=10, color=ink)
            if k == 0:
                ax.legend(fontsize=8, frameon=False, loc="best")
            # pelvis speed: merged vs chunks
            ax = fig.add_subplot(gs[k + 1, 2 + c])
            style(ax)
            ax.axvspan(t[lo], t[hi - 1], color="#f0efec", zorder=0)
            for jj, sel, gg, key, ls in ((prev, sel_p, pg, "prev", "--"), (nxt, sel_n, ng, "next", ":")):
                sp = _speed(jj[:, 0])
                ax.plot(t[gg[sel]], sp[sel], ls, color=CHUNK_GRAY[key], lw=1.6, label=f"chunk {key}")
            ax.plot(tt, _speed(J[:, 0])[g0:g1], color=ROLE_COLOR[role], lw=2.0, label="merged")
            ax.set_xlabel("take time (s)", color=ink2, fontsize=9)
            ax.set_ylabel("pelvis speed (m/s)", color=ink2, fontsize=9)
            pa, ba = s["pelvis_acc"], s["body_acc"]
            ax.set_title(f"seam {k + 1} {role}: {'smooth' if s['smooth'] else 'CHECK'}\npeak accel x"
                         f"{pa['ratio_vs_chunks']:.2f} pelvis, x{ba['ratio_vs_chunks']:.2f} body vs chunks",
                         loc="left", fontsize=10, color=ink if s["smooth"] else "#d03b3b")
            if k == 0:
                ax.legend(fontsize=8, frameon=False, loc="best")
        # chunk disagreement over the overlap, both roles
        ax = fig.add_subplot(gs[k + 1, 4])
        style(ax)
        ta = t[a:b]
        ax.axvspan(t[lo], t[hi - 1], color="#f0efec", zorder=0)
        for role in ROLES:
            prev, nxt = data[role]["chunks"][k], data[role]["chunks"][k + 1]
            d = np.linalg.norm(prev[a - chunks[k]["a"]:] - nxt[:b - a], axis=-1) * 1000
            ax.plot(ta, np.median(d, 1), color=ROLE_COLOR[role], lw=2.0, label=f"{role} all joints (median)")
            ax.plot(ta, d[:, 0], "--", color=ROLE_COLOR[role], lw=1.4, label=f"{role} pelvis")
        ax.set_ylim(bottom=0)
        ax.set_xlabel("take time (s)", color=ink2, fontsize=9)
        ax.set_ylabel("chunk vs chunk (mm)", color=ink2, fontsize=9)
        ax.set_title(f"seam {k + 1}: chunk vs chunk, {b - a}-frame overlap\n{chunks[k]['run_tag']} | "
                     f"{chunks[k + 1]['run_tag']}", loc="left", fontsize=10, color=ink)
        if k == 0:
            ax.legend(fontsize=8, frameon=False, loc="best")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=80, facecolor=fig.get_facecolor(), bbox_inches="tight", pil_kwargs={"quality": 90})
    plt.close(fig)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m dancecap.mamma_seams", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("take")
    ap.add_argument("--out", help="review image (default work/<take>/review/pose_seams.jpg)")
    a = ap.parse_args(argv)
    res = analyse(a.take)
    work = PACKAGE_ROOT / "work" / a.take
    img = sheet(res, Path(a.out) if a.out else work / "review" / "pose_seams.jpg")
    rep = {k: v for k, v in res.items() if not k.startswith("_")}
    (work / "mamma" / "chunks_seams.json").write_text(json.dumps(rep, indent=1))
    for role in ROLES:
        rr = rep["roles"][role]
        print(f"[seams] {role}: " + "; ".join(
            f"s{i + 1} {s['t_blend'][0]:.1f}s dis {s['disagree_mm']['median']:.0f}/{s['disagree_mm']['pelvis_median']:.0f} mm "
            f"acc x{s['pelvis_acc']['ratio_vs_chunks']:.2f}/x{s['body_acc']['ratio_vs_chunks']:.2f} "
            f"{'ok' if s['smooth'] else 'CHECK'}" for i, s in enumerate(rr["seams"])))
    print(f"[seams] all smooth: {rep['all_smooth']} -> {img}")


if __name__ == "__main__":
    main()
