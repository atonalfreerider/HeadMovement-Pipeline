"""How well do the open face detectors work on the dancers' small video faces? (input: face_survey + face_worker)

    python -m dancecap.face_eval <take>     -> work/<take>/face/eval.json, review/face_detectors.png,
                                               review/face_detectors_crops.jpg

There is no ground truth, so the measures are self-consistency checks, all normalised by the inter-ocular distance
(IOD, from the fitted SMPL-X face) and reported per IOD bin, on the frames face_survey calls usable (frontal < 60 deg,
not occluded, not behind hair):
  * detection rate: MediaPipe returned a face near the expected position (< 0.5 IOD); RTMW face score > 3
  * MP-vs-RTMW: NME between the two detectors' 52 shared iBUG points (two independent models agreeing)
  * model-vs-detector: NME between the projected SMPL-X landmarks and each detector (head pose + camera + fit error)
  * jitter: frame-to-frame landmark motion after removing a similarity transform (same camera, consecutive frames),
    i.e. the noise floor an expression signal has to beat
  * mouth opening (inner-lip gap / IOD) from MP and RTMW, and MediaPipe's jawOpen / mouthSmile / eyeBlink scores:
    cross-camera agreement where two cameras see the same dancer at the same instant (+- 20 ms)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from .face_common import (IBUG52, LABEL, MP_TO_IBUG68, ROLES, crop_pad, label)
from .take import Take

BINS = ((5, 10), (10, 15), (15, 20), (20, 30), (30, 60))


def _similarity_residual(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-point distance after the best similarity transform a -> b."""
    ca, cb = a.mean(0), b.mean(0)
    A, B = a - ca, b - cb
    U, S, Vt = np.linalg.svd(A.T @ B)
    D = np.diag([1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    s = (S * np.diag(D)).sum() / max((A ** 2).sum(), 1e-12)
    return np.linalg.norm(s * A @ R - B, axis=1)


def load(work: Path) -> dict:
    jobs = json.loads((work / "face" / "jobs.json").read_text())["jobs"]
    det = np.load(work / "face" / "detections.npz")
    sv = np.load(work / "face" / "survey.npz")
    n = len(jobs)
    base = np.zeros(n, bool)
    for k, j in enumerate(jobs):
        key = f"{j['role']}_{j['cam']}_"
        ii = sv[key + "i"]
        m = np.flatnonzero(ii == j["i"])
        base[k] = bool(sv[key + "base"][m[0]]) if len(m) else False
    iod = np.array([j["iod"] for j in jobs])
    mp68 = det["mp_lmk"][:, MP_TO_IBUG68, :2]
    d = {"jobs": jobs, "n": n, "base": base, "iod": iod, "role": np.array([j["role"] for j in jobs]),
         "cam": np.array([j["cam"] for j in jobs]), "i": np.array([j["i"] for j in jobs]),
         "t": np.array([j["t"] for j in jobs]), "angle": np.array([j["angle"] for j in jobs]),
         "model52": np.array([j["lmk_model"] for j in jobs], float),
         "mp52": mp68[:, IBUG52], "rtmw52": det["rtmw_face"][:, IBUG52].astype(float),
         "rtmw_score": np.nanmean(det["rtmw_score"], 1), "mp_dist": det["mp_dist_px"] / np.maximum(iod, 1),
         "mp_blend": det["mp_blend"], "blend_names": [str(x) for x in det["mp_blend_names"]]}
    d["mp_ok"] = det["mp_ok"] & (d["mp_dist"] < 0.5)
    d["rtmw_ok"] = d["rtmw_score"] > 3.0
    return d


def _nme(a, b, iod):
    return np.nanmean(np.linalg.norm(a - b, axis=-1), -1) / np.maximum(iod, 1)


def _mouth_open(p52):
    """Inner-lip gap (iBUG 62-66, 61-67, 63-65) / mouth width (48-54), from LMK52 arrays (..., 52, 2)."""
    idx = {k: 1 + (k - 17) for k in range(17, 68)}
    gap = (np.linalg.norm(p52[..., idx[62], :] - p52[..., idx[66], :], axis=-1) +
           np.linalg.norm(p52[..., idx[61], :] - p52[..., idx[67], :], axis=-1) +
           np.linalg.norm(p52[..., idx[63], :] - p52[..., idx[65], :], axis=-1)) / 3
    width = np.linalg.norm(p52[..., idx[48], :] - p52[..., idx[54], :], axis=-1)
    return gap / np.maximum(width, 1e-6)


def evaluate(take: Take) -> dict:
    work = take.work
    d = load(work)
    out = {"take": take.name, "n_jobs": d["n"], "n_usable": int(d["base"].sum()), "bins": {}, "per_role": {}}
    nme_mr = _nme(d["mp52"], d["rtmw52"], d["iod"])
    nme_m_mp = _nme(d["model52"], d["mp52"], d["iod"])
    nme_m_rt = _nme(d["model52"], d["rtmw52"], d["iod"])
    d["nme_mr"], d["nme_m_mp"], d["nme_m_rt"] = nme_mr, nme_m_mp, nme_m_rt
    mo_mp, mo_rt = _mouth_open(d["mp52"]), _mouth_open(d["rtmw52"])
    d["mo_mp"], d["mo_rt"] = mo_mp, mo_rt
    for lo, hi in BINS:
        m = d["base"] & (d["iod"] >= lo) & (d["iod"] < hi)
        both = m & d["mp_ok"] & d["rtmw_ok"]
        out["bins"][f"{lo}-{hi}"] = {
            "usable_views": int(m.sum()),
            "mediapipe_detect": round(float(d["mp_ok"][m].mean()), 3) if m.any() else None,
            "rtmw_detect": round(float(d["rtmw_ok"][m].mean()), 3) if m.any() else None,
            "nme_mp_vs_rtmw_median": round(float(np.nanmedian(nme_mr[both])), 3) if both.any() else None,
            "nme_model_vs_mp_median": round(float(np.nanmedian(nme_m_mp[m & d["mp_ok"]])), 3) if (m & d["mp_ok"]).any() else None,
            "nme_model_vs_rtmw_median": round(float(np.nanmedian(nme_m_rt[m & d["rtmw_ok"]])), 3) if (m & d["rtmw_ok"]).any() else None}
    # jitter: consecutive frames, same role+cam, both detected
    jit = {"mp": [], "rtmw": [], "iod": []}
    order = np.lexsort((d["i"], d["cam"], d["role"]))
    for a, b in zip(order[:-1], order[1:]):
        if d["role"][a] != d["role"][b] or d["cam"][a] != d["cam"][b] or d["i"][b] - d["i"][a] != 1:
            continue
        if not (d["base"][a] and d["base"][b]):
            continue
        if d["mp_ok"][a] and d["mp_ok"][b]:
            jit["mp"].append(np.median(_similarity_residual(d["mp52"][a], d["mp52"][b])) / d["iod"][a])
        else:
            jit["mp"].append(np.nan)
        if d["rtmw_ok"][a] and d["rtmw_ok"][b]:
            jit["rtmw"].append(np.median(_similarity_residual(d["rtmw52"][a], d["rtmw52"][b])) / d["iod"][a])
        else:
            jit["rtmw"].append(np.nan)
        jit["iod"].append(d["iod"][a])
    J = {k: np.array(v) for k, v in jit.items()}
    out["jitter_median_over_iod"] = {
        f"{lo}-{hi}": {"mp": _nanmed(J["mp"][(J["iod"] >= lo) & (J["iod"] < hi)]),
                       "rtmw": _nanmed(J["rtmw"][(J["iod"] >= lo) & (J["iod"] < hi)])} for lo, hi in BINS}
    # cross-camera agreement of expression signals at the same instant
    names = d["blend_names"]
    sig = {"mouth_open_mp": mo_mp, "mouth_open_rtmw": mo_rt}
    if names:
        B = d["mp_blend"]
        g = lambda n: B[:, names.index(n)] if n in names else np.full(d["n"], np.nan)
        sig["jawOpen"] = g("jawOpen")
        sig["mouthSmile"] = (g("mouthSmileLeft") + g("mouthSmileRight")) / 2
        sig["eyeBlink"] = (g("eyeBlinkLeft") + g("eyeBlinkRight")) / 2
    d["sig"] = sig
    pairs = {}
    for r in ROLES:
        idx = np.flatnonzero((d["role"] == r) & d["base"] & (d["iod"] >= 10))
        for p, q in [(p, q) for p in idx for q in idx if p < q and d["cam"][p] != d["cam"][q] and abs(d["t"][p] - d["t"][q]) < 0.02]:
            for s_name, s in sig.items():
                okp = d["mp_ok"] if "rtmw" not in s_name else d["rtmw_ok"]
                if okp[p] and okp[q] and np.isfinite(s[p]) and np.isfinite(s[q]):
                    pairs.setdefault((r, s_name), []).append((s[p], s[q]))
    out["cross_camera"] = {}
    for (r, s_name), v in pairs.items():
        v = np.array(v)
        corr = float(np.corrcoef(v[:, 0], v[:, 1])[0, 1]) if len(v) > 3 and v.std(0).min() > 1e-6 else None
        out["cross_camera"].setdefault(r, {})[s_name] = {"pairs": int(len(v)), "corr": None if corr is None else round(corr, 3),
                                                         "mean_abs_diff": round(float(np.abs(v[:, 0] - v[:, 1]).mean()), 3)}
    # per role: how often is there at least one usable detected face per 30 Hz grid frame
    sv = np.load(work / "face" / "survey.npz")
    grid = sv["grid_t"]
    for r in ROLES:
        cov = {}
        for name, okk in (("mediapipe", d["mp_ok"]), ("rtmw", d["rtmw_ok"])):
            sel = (d["role"] == r) & d["base"] & okk
            hit = np.zeros(len(grid), bool)
            if sel.any():
                k = np.abs(grid[None, :] - d["t"][sel][:, None]).argmin(1)
                near = np.abs(grid[k] - d["t"][sel]) < 0.02
                hit[k[near]] = True
            cov[name] = round(float(hit.mean()), 3)
        out["per_role"][r] = {"grid_frames_with_detected_usable_face": cov,
                              "usable_views": int(((d["role"] == r) & d["base"]).sum())}
    (work / "face" / "eval.json").write_text(json.dumps(out, indent=1))
    rev = work / "review"
    out["review"] = [str(_plots(d, out, rev / "face_detectors.png", take.name)),
                     str(_crops(work, d, rev / "face_detectors_crops.jpg", take.name))]
    (work / "face" / "eval.json").write_text(json.dumps(out, indent=1))
    return out


def _nanmed(x):
    x = x[np.isfinite(x)]
    return round(float(np.median(x)), 4) if len(x) else None


def _plots(d, out, path: Path, take: str) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(17, 8.5))
    labels = list(out["bins"].keys())
    x = np.arange(len(labels))
    b = out["bins"]
    ax[0, 0].bar(x - 0.2, [b[k]["mediapipe_detect"] or 0 for k in labels], 0.4, label="MediaPipe Face Landmarker")
    ax[0, 0].bar(x + 0.2, [b[k]["rtmw_detect"] or 0 for k in labels], 0.4, label="RTMW-x whole-body (face 68)")
    for k, lab in enumerate(labels):
        ax[0, 0].text(k, 1.02, f"n={b[lab]['usable_views']}", ha="center", fontsize=8)
    ax[0, 0].set_xticks(x, labels)
    ax[0, 0].set_xlabel("inter-ocular distance (px)")
    ax[0, 0].set_ylabel("detection rate (usable views)")
    ax[0, 0].set_ylim(0, 1.1)
    ax[0, 0].legend(fontsize=8)
    for key, lab in (("nme_mp_vs_rtmw_median", "MP vs RTMW"), ("nme_model_vs_mp_median", "SMPL-X fit vs MP"),
                     ("nme_model_vs_rtmw_median", "SMPL-X fit vs RTMW")):
        ax[0, 1].plot(x, [b[k][key] if b[k][key] is not None else np.nan for k in labels], "o-", label=lab)
    jm = out["jitter_median_over_iod"]
    ax[0, 1].plot(x, [jm[k]["mp"] if jm[k]["mp"] is not None else np.nan for k in labels], "s--", label="MP jitter (frame to frame)")
    ax[0, 1].plot(x, [jm[k]["rtmw"] if jm[k]["rtmw"] is not None else np.nan for k in labels], "s--", label="RTMW jitter")
    ax[0, 1].set_xticks(x, labels)
    ax[0, 1].set_ylabel("median error / IOD")
    ax[0, 1].set_xlabel("inter-ocular distance (px)")
    ax[0, 1].axhline(0.1, color="grey", lw=0.6)
    ax[0, 1].legend(fontsize=7)
    ax[0, 1].set_title("landmark agreement and noise (lower = better)", fontsize=9)
    # cross-camera scatter of mouth opening (MP) for both roles
    cc = out.get("cross_camera", {})
    txt = []
    for r in ROLES:
        for s_name, v in cc.get(r, {}).items():
            txt.append(f"{r:6s} {s_name:16s} pairs {v['pairs']:3d}  corr {v['corr']}  |diff| {v['mean_abs_diff']}")
    ax[0, 2].axis("off")
    ax[0, 2].text(0, 1, "cross-camera agreement (same dancer, same instant):\n" + "\n".join(txt), va="top",
                  family="monospace", fontsize=7.5)
    for col, r in enumerate(ROLES):
        a = ax[1, col]
        for c in sorted(set(d["cam"])):
            m = (d["role"] == r) & (d["cam"] == c) & d["base"] & d["mp_ok"]
            if m.sum() > 2:
                a.plot(d["t"][m], d["sig"]["jawOpen"][m] if "jawOpen" in d["sig"] else d["mo_mp"][m], ".-", ms=3,
                       lw=0.6, label=f"cam {c} jawOpen")
            m2 = (d["role"] == r) & (d["cam"] == c) & d["base"] & d["rtmw_ok"]
            if m2.sum() > 2:
                a.plot(d["t"][m2], d["mo_rt"][m2], "x", ms=3, alpha=0.5, label=f"cam {c} RTMW lip gap")
        a.set_title(f"{LABEL[r]}: MediaPipe jawOpen (lines) / RTMW inner-lip gap ratio (x) per camera", fontsize=9)
        a.set_xlabel("reference time (s)")
        a.legend(fontsize=6, ncol=2)
    ax[1, 2].axis("off")
    fig.suptitle(f"{take}: open face detectors on the video faces (usable views only)")
    fig.tight_layout()
    fig.savefig(path, dpi=105)
    plt.close(fig)
    return path


def _crops(work: Path, d, path: Path, take: str, tile: int = 180) -> Path:
    """Per IOD bin and dancer: 3 usable views, native crop upscaled, MP (green) and RTMW (magenta) landmarks,
    projected SMPL-X landmarks (red)."""
    rows = []
    rng = np.random.default_rng(0)
    for r in ROLES:
        for lo, hi in BINS[:4]:
            idx = np.flatnonzero((d["role"] == r) & d["base"] & (d["iod"] >= lo) & (d["iod"] < hi))
            if not len(idx):
                continue
            pick = rng.choice(idx, size=min(4, len(idx)), replace=False)
            tiles = []
            for k in pick:
                j = d["jobs"][k]
                img = cv2.imread(str(work / "frames" / j["cam"] / f"{j['i']:06d}.jpg"))
                x0, y0, x1, y1 = [int(round(v)) for v in j["box_face"]]
                s = tile / max(x1 - x0, 1)
                raw = crop_pad(img, (x0, y0, x1, y1), tile, cv2.INTER_NEAREST)
                ov = crop_pad(img, (x0, y0, x1, y1), tile, cv2.INTER_CUBIC)
                for pts, colr, ok in ((d["model52"][k], (0, 0, 255), True), (d["rtmw52"][k], (255, 0, 255), d["rtmw_ok"][k]),
                                      (d["mp52"][k], (0, 255, 0), d["mp_ok"][k])):
                    if not ok:
                        continue
                    for p in pts:
                        if np.isfinite(p).all():
                            cv2.circle(ov, tuple(np.round((p - (x0, y0)) * s).astype(int)), 2, colr, -1)
                label(raw, f"cam {j['cam']} IOD {j['iod']:.0f}", (3, 13), scale=0.4)
                label(ov, f"MP {'ok' if d['mp_ok'][k] else '--'} RTMW {'ok' if d['rtmw_ok'][k] else '--'}", (3, 13), scale=0.4)
                tiles += [raw, ov, np.zeros((tile, 3, 3), np.uint8)]
            row = np.concatenate(tiles, 1)
            hs = np.full((18, row.shape[1], 3), 20, np.uint8)
            label(hs, f"{LABEL[r]}  IOD {lo}-{hi}px", (4, 13), scale=0.42)
            rows.append(np.concatenate([hs, row], 0))
    w = max(rw.shape[1] for rw in rows)
    sheet = np.concatenate([np.pad(rw, ((0, 3), (0, w - rw.shape[1]), (0, 0))) for rw in rows], 0)
    top = np.full((26, w, 3), 10, np.uint8)
    label(top, f"{take}: native crop | landmarks: MediaPipe (green), RTMW (magenta), fitted SMPL-X (red)", (5, 18), scale=0.5)
    cv2.imwrite(str(path), np.concatenate([top, sheet], 0), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return path


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("take")
    a = ap.parse_args(argv)
    out = evaluate(Take.load(a.take))
    print(json.dumps({k: v for k, v in out.items() if k != "review"}, indent=1))
    print("review:", out["review"])


if __name__ == "__main__":
    sys.exit(main())
