"""carlos_aline: where is a fitted chunk (or the stitched take) wrong, per camera and second (agent C3).

    python -m dancecap.carlos_aline_fitcheck <take dir name, e.g. carlos_aline_cag0000 or carlos_aline> [--src smplx_rf2_noslip] [--out file.json]

The 3D body joints of the fit (SMPL-X joints of smplx/<role>.npz: shoulders, elbows, wrists, hips, knees, ankles) are projected through the per-frame camera
tracks (cameras/<cam>_world.npz, the video offsets of the take's sync/offsets.json applied) and compared with the YOLO joints of the teacher tracks
(people/teachers_<cam>.npz, confidence >= 0.5, joint confidence >= 0.5). The detector's own noise and the COCO-vs-SMPL-X joint definitions cost 2-3 % of the person's height, so the per-second MEDIAN over joints
is the reading: a camera whose second reads 5 % or more while the other camera reads 2-3 % is the one to leave out of that second (farm refit --mask / qa_mask).
Output: {cam: {role: {"sec": [t0 ...], "px": [median per second]}}, "summary": {cam: {role: {median, p90, bad_seconds: [...]}}}}; errors are in PERCENT of the person's height in the picture (01 shows the dancers 1.5 times larger than 02).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .carlos_aline_inputs import CAMS, PEOPLE, WORK
from .take import PACKAGE_ROOT

# COCO-17 index -> SMPL-X joint index (left/right shoulders, elbows, wrists, hips, knees, ankles)
COCO_TO_SMPLX = {5: 16, 6: 17, 7: 18, 8: 19, 9: 20, 10: 21, 11: 1, 12: 2, 13: 4, 14: 5, 15: 7, 16: 8}
BAD_PCT = 5.0                    # per-second median error, percent of the person's height in the picture (the detector + joint-definition floor is 2-3 %)
CONF = 0.5


def project(X: np.ndarray, R: np.ndarray, tvec: np.ndarray, K: np.ndarray, k1: float) -> tuple[np.ndarray, np.ndarray]:
    Xc = X @ R.T + tvec
    z = np.where(Xc[..., 2] > 1e-6, Xc[..., 2], 1e-6)
    x, y = Xc[..., 0] / z, Xc[..., 1] / z
    d = 1 + k1 * (x * x + y * y)
    return np.stack([K[0, 0] * x * d + K[0, 2], K[1, 1] * y * d + K[1, 2]], -1), Xc[..., 2]


def offsets_s(work: Path) -> dict:
    p = work / "sync" / "offsets.json"
    if not p.exists():      # the stitched take has none: the chunks all carry the same fixed offsets (ca_profile.SYNC_FIXED)
        p = PACKAGE_ROOT / "work" / f"{work.name}_cag0000" / "sync" / "offsets.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text())["offsets"]
    return {c: float(v.get("delta", 0.0)) for c, v in d.items()}


def check(take: str, src: str = "smplx_rf2_noslip") -> dict:
    work = PACKAGE_ROOT / "work" / take
    S = {r: np.load(work / src / f"{r}.npz", allow_pickle=True) for r in ("lead", "follow")}
    t = np.asarray(S["lead"]["t"], float)                          # fit grid times (reference clock)
    offs = offsets_s(work)
    out, summ = {}, {}
    for cam in CAMS:
        zc = np.load(WORK / "cameras" / f"{cam}_world.npz", allow_pickle=True)
        T = np.load(PEOPLE / f"teachers_{cam}.npz")
        tc = np.asarray(zc["t"], float)
        label = t - offs.get(cam, 0.0)                              # corrected = label + delta
        j = np.clip(np.searchsorted(tc, label), 1, len(tc) - 1)
        j = np.where(np.abs(tc[j - 1] - label) < np.abs(tc[j] - label), j - 1, j)
        good = (np.abs(tc[j] - label) < 0.04) & np.asarray(zc["ok"], bool)[j]
        out[cam], summ[cam] = {}, {}
        for r, role in enumerate(("lead", "follow")):
            err = np.full(len(t), np.nan)
            J = np.asarray(S[role]["joints"], float)
            for i in np.flatnonzero(good):
                k = int(j[i])
                if T["row"][k, r] < 0 or T["conf"][k, r] < CONF:
                    continue
                kp = T["kp"][k, r]
                ids = [c for c in COCO_TO_SMPLX if kp[c, 2] >= CONF]
                if len(ids) < 6:
                    continue
                uv, z = project(J[i, [COCO_TO_SMPLX[c] for c in ids]], np.asarray(zc["R"][k], float), np.asarray(zc["tvec"][k], float), np.asarray(zc["K"][k], float),
                                float(zc["dist"][k]))
                hpx = max(float(T["box"][k, r, 3] - T["box"][k, r, 1]), 50.0)
                e = 100.0 * np.linalg.norm(uv - kp[ids, :2], axis=-1) / hpx
                e = e[z > 0]
                if len(e) >= 6:
                    err[i] = float(np.median(e))
            sec = np.floor(t - t[0]).astype(int)
            secs, px = [], []
            for s in range(int(sec.max()) + 1):
                m = (sec == s) & np.isfinite(err)
                secs.append(round(float(t[0] + s), 1))
                px.append(round(float(np.median(err[m])), 1) if m.sum() >= 5 else None)
            fin = err[np.isfinite(err)]
            out[cam][role] = {"sec": secs, "px": px}
            summ[cam][role] = {"median": round(float(np.median(fin)), 2) if len(fin) else None, "p90": round(float(np.percentile(fin, 90)), 2) if len(fin) else None,
                               "frames": int(len(fin)), "bad_seconds": [s for s, v in zip(secs, px) if v is not None and v >= BAD_PCT]}
    return {"take": take, "src": src, "offsets_s": offs, "summary": summ, "per_second": out}


# SMPL-X body bones (pelvis 0, hips 1 2, spine 3 6 9, knees 4 5, ankles 7 8, feet 10 11, neck 12, collars 13 14, head 15, shoulders 16 17, elbows 18 19, wrists 20 21)
BONES = [(0, 1), (0, 2), (0, 3), (3, 6), (6, 9), (9, 12), (12, 15), (1, 4), (2, 5), (4, 7), (5, 8), (7, 10), (8, 11), (9, 13), (9, 14), (13, 16), (14, 17), (16, 18), (17, 19), (18, 20), (19, 21)]


def overlay(take: str, times: list[float], src: str = "smplx_rf2_noslip", out: Path | None = None, width: int = 640, cols: int = 2) -> list[Path]:
    """Contact sheets (one per camera) of the fit's skeletons (lead green, follow red) drawn over the source video frames at the given reference times."""
    import cv2

    from .carlos_aline_inputs import SOURCE, VIDEO
    work = PACKAGE_ROOT / "work" / take
    S = {r: np.load(work / src / f"{r}.npz", allow_pickle=True) for r in ("lead", "follow")}
    t = np.asarray(S["lead"]["t"], float)
    offs = offsets_s(work)
    out = out or WORK / "review" / "fits"
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for cam in CAMS:
        zc = np.load(WORK / "cameras" / f"{cam}_world.npz", allow_pickle=True)
        zf = np.load(WORK / "cameras" / f"{cam}.npz", allow_pickle=True)
        tc = np.asarray(zc["t"], float)
        cap = cv2.VideoCapture(str(SOURCE / VIDEO[cam]))
        tiles = []
        for tt in times:
            i = int(np.argmin(np.abs(t - tt)))
            if abs(t[i] - tt) > 0.1:
                continue
            k = int(np.argmin(np.abs(tc - (t[i] - offs.get(cam, 0.0)))))
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(zf["frame_index"][k]))
            ok, img = cap.read()
            if not ok:
                continue
            sc = width / img.shape[1]
            img = cv2.resize(img, (width, int(img.shape[0] * sc)))
            for role, col in (("lead", (60, 220, 60)), ("follow", (60, 60, 240))):
                J = np.asarray(S[role]["joints"], float)[i]
                uv, z = project(J, np.asarray(zc["R"][k], float), np.asarray(zc["tvec"][k], float), np.asarray(zc["K"][k], float), float(zc["dist"][k]))
                uv = uv * sc
                for a, b in BONES:
                    cv2.line(img, (int(uv[a, 0]), int(uv[a, 1])), (int(uv[b, 0]), int(uv[b, 1])), col, 2)
            cv2.putText(img, f"{cam} t={tt:.1f}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            tiles.append(img)
        cap.release()
        if not tiles:
            continue
        h = max(x.shape[0] for x in tiles)
        tiles = [cv2.copyMakeBorder(x, 0, h - x.shape[0], 0, 0, cv2.BORDER_CONSTANT) for x in tiles]
        while len(tiles) % cols:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[q:q + cols]) for q in range(0, len(tiles), cols)])
        pth = out / f"{take}_{cam}_{int(times[0])}_{int(times[-1])}.jpg"
        cv2.imwrite(str(pth), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
        paths.append(pth)
    return paths


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.carlos_aline_fitcheck", description=__doc__.split("\n")[0])
    ap.add_argument("take")
    ap.add_argument("--src", default="smplx_rf2_noslip")
    ap.add_argument("--out")
    ap.add_argument("--overlay", nargs="*", type=float, help="instead of the table: skeleton overlays at these reference times")
    a = ap.parse_args(argv)
    if a.overlay:
        for pth in overlay(a.take, a.overlay, a.src):
            print(pth)
        return 0
    rep = check(a.take, a.src)
    out = Path(a.out) if a.out else WORK / "chunks" / f"fitcheck_{a.take}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1))
    for cam, d in rep["summary"].items():
        for role, v in d.items():
            print(f"{a.take} cam {cam} {role}: median {v['median']} p90 {v['p90']} %, frames {v['frames']}, seconds >= {BAD_PCT:g} %: {v['bad_seconds']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
