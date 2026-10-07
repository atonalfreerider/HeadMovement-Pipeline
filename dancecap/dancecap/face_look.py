"""Do the dancers look in the demo video the way they do in their portraits? (makeup, beard) - measured, not guessed.

    python -m dancecap.face_look <take>     -> work/<take>/face/look.json, review/face_look.jpg

Inputs (no new inference): the portrait MediaPipe landmarks face/portrait_<role>.npz (avatar_face.portrait_landmarks)
and the video MediaPipe landmarks face/detections.npz on face/jobs.json (face_survey + face_worker). CPU only, seconds.

Region statistics in CIE Lab (sRGB -> Lab, L 0-100), each relative to the dancer's own cheeks so that lighting and
white balance cancel to first order:
    lip_da            lip a* - cheek a*            (lips redder than the skin: lipstick)
    lip_dL            lip L  - cheek L
    lip_chroma_ratio  lip chroma / cheek chroma
    chin_dL           chin (between lower lip and chin tip) L - cheek L     (beard density / length)
    must_dL           moustache (between upper lip and nose base) L - cheek L
Regions come from MediaPipe indices (outer minus inner lip polygon, discs scaled by the inter-ocular distance IOD).
Small video faces blur lips into skin, so the portrait is measured twice: native and DEGRADED to the video's median
face size (area resize to that IOD, Gaussian blur sigma 0.8 px, JPEG q60). A measure "differs" when the degraded
portrait lies outside the video's interquartile range by more than TOL (Lab units). Video views: face_survey's usable
flag (frontal, not hidden by arms / partner / hair), MediaPipe face within 0.5 IOD of the expected position and within
MAX_NME IOD of the projected fitted SMPL-X face (else it locked onto a hand), IOD >= MIN_IOD px, angle < MAX_ANGLE deg.
The lead's natural lips are the control for what the capture alone does to lip redness (chroma subsampling, blur,
grading): look.json lip_da_follow_minus_lead_control = the follow's lip difference minus the lead's.

larissa_kadu (2026-10-06): Larissa's portrait lips are clearly redder than her cheeks (lip_da +11.2 at video size)
while in the 7 clean video views (cams 02/09, all broad smiles) they are not (median -1.5, IQR -3.4..-1.1); Kadu's
control drops +6.7 -> +0.8, so ~+6.9 a* of the difference is makeup: red lipstick in the portrait, lighter or none at
the demo (moderate confidence: n = 7, open-mouth smiles). Kadu's moustache matches (-32 vs -30, n = 93, cams 04/05/06)
but his chin beard is much less dark in the video (chin_dL -50 vs -27, IQR -30..-25): a shorter / thinner chin beard
at the demo (or top light while looking down). The portrait texture (avatar_face) keeps the portrait look; a
"video-matched" variant would pull lip chroma / chin darkness toward these video targets.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from .face_common import LABEL, ROLES, crop_pad, label
from .take import Take

LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
CHEEKS = (50, 280)
EYE_OUTER = (33, 263)
MEASURES = ("lip_da", "lip_dL", "lip_chroma_ratio", "chin_dL", "must_dL")
MIN_IOD, MAX_ANGLE, MAX_NME = 15.0, 45.0, 0.25
DEGRADE_SIGMA, DEGRADE_JPEG = 0.8, 60
TOL = {"lip_da": 3.0, "lip_dL": 5.0, "lip_chroma_ratio": 0.15, "chin_dL": 6.0, "must_dL": 6.0}
# the measures that mean something per dancer (beard for the lead, lipstick for the follow; lips for both)
RELEVANT = {"lead": ("chin_dL", "must_dL", "lip_da"), "follow": ("lip_da", "lip_chroma_ratio", "lip_dL")}


def iod_of(lmk: np.ndarray) -> float:
    return float(np.linalg.norm(lmk[EYE_OUTER[0], :2] - lmk[EYE_OUTER[1], :2]))


def regions(shape, lmk: np.ndarray) -> dict:
    """Boolean masks lips / cheek / chin / must for one face (MediaPipe 478 landmarks in image px)."""
    H, W = shape[:2]
    iod = iod_of(lmk)

    def poly(ix):
        m = np.zeros((H, W), np.uint8)
        cv2.fillPoly(m, [np.round(lmk[ix, :2]).astype(np.int32)], 1)
        return m.astype(bool)

    def disc(c, r):
        m = np.zeros((H, W), np.uint8)
        cv2.circle(m, (int(round(c[0])), int(round(c[1]))), max(1, int(round(r))), 1, -1)
        return m.astype(bool)
    return {"lips": poly(LIPS_OUTER) & ~poly(LIPS_INNER),
            "cheek": disc(lmk[CHEEKS[0], :2], 0.12 * iod) | disc(lmk[CHEEKS[1], :2], 0.12 * iod),
            "chin": disc(0.5 * (lmk[17, :2] + lmk[152, :2]), 0.10 * iod),
            "must": disc(0.5 * (lmk[0, :2] + lmk[2, :2]), 0.06 * iod)}


def measure(img_bgr: np.ndarray, lmk: np.ndarray) -> dict | None:
    lab = cv2.cvtColor(img_bgr[:, :, ::-1].astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    rg = regions(img_bgr.shape, lmk)
    if rg["lips"].sum() < 3 or rg["cheek"].sum() < 3 or rg["chin"].sum() < 1 or rg["must"].sum() < 1:
        return None
    med = {k: np.median(lab[m], 0) for k, m in rg.items()}
    c, lp = med["cheek"], med["lips"]
    return {"iod": iod_of(lmk), "cheek_L": float(c[0]),
            "lip_da": float(lp[1] - c[1]), "lip_dL": float(lp[0] - c[0]),
            "lip_chroma_ratio": float(np.hypot(lp[1], lp[2]) / max(np.hypot(c[1], c[2]), 1e-3)),
            "chin_dL": float(med["chin"][0] - c[0]), "must_dL": float(med["must"][0] - c[0])}


def degrade(img_bgr: np.ndarray, lmk: np.ndarray, target_iod: float) -> tuple[np.ndarray, np.ndarray]:
    """Portrait shrunk to the video's face size (area resize, blur, JPEG): resolution loss is not a look difference."""
    s = min(1.0, target_iod / max(iod_of(lmk), 1e-6))
    sm = cv2.resize(img_bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    if DEGRADE_SIGMA:
        sm = cv2.GaussianBlur(sm, (0, 0), DEGRADE_SIGMA)
    sm = cv2.imdecode(cv2.imencode(".jpg", sm, [cv2.IMWRITE_JPEG_QUALITY, DEGRADE_JPEG])[1], cv2.IMREAD_COLOR)
    out = lmk.astype(float).copy()
    out[:, :2] *= s
    return sm, out


def usable_flags(work: Path, jobs: list) -> np.ndarray:
    """face_survey's per-view usable flag (frontal, not occluded by bodies/arms/partner/hair) for every job."""
    sv = np.load(work / "face" / "survey.npz")
    cache, out = {}, np.zeros(len(jobs), bool)
    for k, j in enumerate(jobs):
        key = f"{j['role']}_{j['cam']}_"
        if key not in cache:
            cache[key] = (sv[key + "i"], sv[key + "base"]) if key + "i" in sv.files else (np.zeros(0, int), None)
        ii, base = cache[key]
        m = np.flatnonzero(ii == j["i"])
        out[k] = bool(base[m[0]]) if len(m) else False
    return out


def mp_vs_model_nme(j: dict, mp_lmk: np.ndarray) -> float:
    """MediaPipe vs the projected fitted SMPL-X face (52 iBUG points), in IOD: large = MP locked onto a hand / hair."""
    from .face_common import IBUG52, MP_TO_IBUG68
    model = np.asarray(j["lmk_model"], float)[:, :2]
    mp = mp_lmk[MP_TO_IBUG68][IBUG52, :2]
    return float(np.linalg.norm(mp - model, axis=1).mean() / max(j["iod"], 1e-6))


def video_views(work: Path, role: str, jobs: list, det) -> list[dict]:
    views = []
    usable = usable_flags(work, jobs)
    for k, j in enumerate(jobs):
        if (j["role"] != role or not usable[k] or not bool(det["mp_ok"][k]) or j["iod"] < MIN_IOD
                or j["angle"] > MAX_ANGLE or float(det["mp_dist_px"][k]) > 0.5 * j["iod"]
                or mp_vs_model_nme(j, det["mp_lmk"][k]) > MAX_NME):
            continue
        img = cv2.imread(str(work / "frames" / j["cam"] / f"{j['i']:06d}.jpg"))
        if img is None:
            continue
        lmk = det["mp_lmk"][k].astype(float)
        r = measure(img, lmk)
        if r:
            views.append({**r, "k": k, "cam": j["cam"], "i": j["i"]})
    return views


def compare(portrait_deg: dict, views: list[dict]) -> dict:
    out = {}
    for m in MEASURES:
        v = np.array([x[m] for x in views])
        q1, med, q3 = (float(x) for x in np.percentile(v, [25, 50, 75]))
        p = portrait_deg[m]
        gap = max(q1 - p, p - q3, 0.0)
        out[m] = {"portrait_degraded": round(p, 2), "video_median": round(med, 2), "video_iqr": [round(q1, 2), round(q3, 2)],
                  "differs": bool(gap > TOL[m]), "portrait_minus_video": round(p - med, 2)}
    return out


def _outline(img, lmk, scale, color_by=None):
    rg = regions(img.shape, lmk)
    cols = color_by or {"lips": (0, 0, 255), "cheek": (0, 255, 0), "chin": (255, 0, 255), "must": (255, 255, 0)}
    for k, m in rg.items():
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, cs, -1, cols[k], 1)
    return img


def _tile(img_bgr, lmk, tile: int, caption: str, lines: list[str]) -> np.ndarray:
    iod = iod_of(lmk)
    c = 0.5 * (lmk[EYE_OUTER[0], :2] + lmk[EYE_OUTER[1], :2]) + np.array([0.0, 0.55 * iod])
    side = int(round(2.6 * iod))
    box = (int(round(c[0] - side / 2)), int(round(c[1] - side / 2)), int(round(c[0] + side / 2)), int(round(c[1] + side / 2)))
    s = tile / max(side, 1)
    crop = crop_pad(img_bgr, box, tile, interp=cv2.INTER_NEAREST if s > 2 else None)
    lm = lmk.astype(float).copy()
    lm[:, 0] = (lm[:, 0] - box[0]) * s
    lm[:, 1] = (lm[:, 1] - box[1]) * s
    crop = _outline(crop.copy(), lm, s)
    label(crop, caption, (4, 14), scale=0.42)
    for n, t in enumerate(lines):
        label(crop, t, (4, tile - 6 - 15 * (len(lines) - 1 - n)), scale=0.4)
    return crop


def _fmt(r: dict) -> list[str]:
    return [f"lip da {r['lip_da']:+.1f} dL {r['lip_dL']:+.0f}", f"chin dL {r['chin_dL']:+.0f} must {r['must_dL']:+.0f}"]


def review(take: Take, rows: dict, path: Path, tile: int = 200, n_video: int = 6) -> Path:
    sheets = []
    W = tile * (2 + n_video)
    for role, d in rows.items():
        hdr = np.zeros((22, W, 3), np.uint8)
        c = d["compare"]
        verdict = "; ".join(f"{m} {'DIFFERS' if c[m]['differs'] else 'same'} ({c[m]['portrait_degraded']:+.1f} vs "
                            f"{c[m]['video_median']:+.1f})" for m in RELEVANT.get(role, MEASURES))
        label(hdr, f"{LABEL[role]}: portrait | portrait at video size | video  -  {verdict}", (4, 16), scale=0.45)
        tiles = [_tile(d["portrait_img"], d["portrait_lmk"], tile, "portrait", _fmt(d["portrait"])),
                 _tile(d["deg_img"], d["deg_lmk"], tile, f"portrait @ IOD {d['deg']['iod']:.0f}px", _fmt(d["deg"]))]
        for v in sorted(d["views"], key=lambda x: -x["iod"])[:n_video]:
            img = cv2.imread(str(take.work / "frames" / v["cam"] / f"{v['i']:06d}.jpg"))
            tiles.append(_tile(img, d["det_lmk"][v["k"]], tile, f"cam {v['cam']} #{v['i']} IOD {v['iod']:.0f}", _fmt(v)))
        while len(tiles) < 2 + n_video:
            tiles.append(np.zeros((tile, tile, 3), np.uint8))
        sheets += [hdr, np.concatenate(tiles, 1)]
    top = np.zeros((22, W, 3), np.uint8)
    label(top, f"{take.name}: makeup / beard check, Lab relative to the cheeks (red lips, green cheeks, magenta chin, "
               f"cyan moustache) - likeness data, local only", (4, 16), scale=0.45)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.concatenate([top] + sheets, 0), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return path


def run(take: Take) -> dict:
    from .avatar_face import portrait_path
    work = take.work
    jobs = json.loads((work / "face" / "jobs.json").read_text())["jobs"]
    with np.load(work / "face" / "detections.npz") as z:      # read once (NpzFile re-reads on every [])
        det = {k: z[k] for k in ("mp_ok", "mp_lmk", "mp_dist_px")}
    out, rows = {"take": take.name, "min_iod": MIN_IOD, "max_angle": MAX_ANGLE, "tol": TOL,
                 "degrade": {"sigma_px": DEGRADE_SIGMA, "jpeg_q": DEGRADE_JPEG}, "roles": {}}, {}
    for role in ROLES:
        pz = work / "face" / f"portrait_{role}.npz"
        pimg_p = portrait_path(take, role)
        if not pz.exists() or not pimg_p.exists():
            out["roles"][role] = {"skipped": f"missing {pz.name if not pz.exists() else pimg_p}"}
            continue
        plmk = np.load(pz)["mp_lmk"].astype(float)
        pimg = cv2.imread(str(pimg_p))
        views = video_views(work, role, jobs, det)
        if len(views) < 5:
            out["roles"][role] = {"skipped": f"only {len(views)} usable video views (IOD >= {MIN_IOD})"}
            continue
        v_iod = float(np.median([v["iod"] for v in views]))
        dimg, dlmk = degrade(pimg, plmk, v_iod)
        p_nat, p_deg = measure(pimg, plmk), measure(dimg, dlmk)
        cmp_ = compare(p_deg, views)
        out["roles"][role] = {"portrait_native": {k: round(v, 2) for k, v in p_nat.items()},
                              "portrait_degraded": {k: round(v, 2) for k, v in p_deg.items()},
                              "video_views": len(views), "video_iod_median": round(v_iod, 1),
                              "video_cams": sorted({v["cam"] for v in views}), "compare": cmp_}
        rows[role] = {"portrait_img": pimg, "portrait_lmk": plmk, "deg_img": dimg, "deg_lmk": dlmk,
                      "portrait": p_nat, "deg": p_deg, "views": views, "det_lmk": det["mp_lmk"], "compare": cmp_}
    (work / "face").mkdir(parents=True, exist_ok=True)
    r = out["roles"]
    if all("compare" in r.get(x, {}) for x in ("lead", "follow")):
        # the other dancer's natural lips are a control for what the capture alone does to lip redness (chroma
        # subsampling, motion blur, grading): what is left after subtracting it is the makeup difference
        ctl = r["lead"]["compare"]["lip_da"]["portrait_minus_video"]
        out["lip_da_follow_minus_lead_control"] = round(r["follow"]["compare"]["lip_da"]["portrait_minus_video"] - ctl, 2)
    if rows:
        out["review"] = [str(review(take, rows, work / "review" / "face_look.jpg"))]
    (work / "face" / "look.json").write_text(json.dumps(out, indent=1))
    return out


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="python -m dancecap.face_look", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("take")
    a = ap.parse_args(argv)
    print(json.dumps(run(Take.load(a.take)), indent=1))


if __name__ == "__main__":
    sys.exit(main())
