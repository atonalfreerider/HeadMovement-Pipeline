"""Review sheets: every camera at the same instants, one row per instant, so cross-view errors stand out.

    python -m dancecap review larissa_kadu            -> work/<take>/review/{reid,pose,people}.jpg (+ smplx when fits exist)

Each tile is the canonical frame nearest the row's reference time (|dt| printed in the corner). Layers use
whatever stages have produced so far:
  reid    dancers/<cam>/masks (lead red, follow blue); falls back to roles.json boxes
  pose    COCO-17 keypoints of the two dancers from people/<cam>.npz, coloured by role when known;
          SMPL-X joints reprojected with per-frame cameras once smplx/ and cameras/ exist
  people  every detection + the spectator ignore-mask (what SfM is told to ignore)
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .take import Take

LEAD = (60, 60, 235)    # BGR red
FOLLOW = (235, 140, 40)  # BGR blue
NEUTRAL = (40, 220, 240)
COCO_EDGES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
              (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4)]
SMPLX_EDGES = [(0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8), (6, 9), (7, 10), (8, 11),
               (9, 12), (12, 15), (9, 13), (9, 14), (13, 16), (14, 17), (16, 18), (17, 19), (18, 20), (19, 21)]
TILE_H = 420


def _frames(work: Path, cam: str) -> dict:
    return json.loads((work / "frames" / cam / "frames.json").read_text())


def _nearest(meta: dict, t: float) -> tuple[int, float]:
    ts = np.array([f["t"] for f in meta["frames"]])
    i = int(np.argmin(np.abs(ts - t)))
    return i, float(ts[i] - t)


def _mask(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return m > 127 if m is not None else None


def _tint(img: np.ndarray, mask: np.ndarray, color, alpha=0.45, outline=2):
    over = img.copy()
    over[mask] = (np.array(color) * alpha + over[mask] * (1 - alpha)).astype(np.uint8)
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(over, cnts, -1, color, outline)
    return over


def _label(img, text, org, color, scale=0.9):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def _role_masks(work: Path, cam: str, i: int) -> dict:
    d = work / "dancers" / cam / "masks"
    return {role: _mask(d / f"mask_{i:04d}_{pid:02d}.png") for role, pid in (("lead", 1), ("follow", 2))}


def _role_scores(box, masks: dict) -> dict:
    """{role: fraction of the BOX covered by the role's mask} for the masks of which the box holds >= 30 %.

    A fraction of the box, not of the mask: in an embrace the partner's box can hold most of a small mask (larissa_kadu_38
    cam05 30.43 s: Kadu's box held most of Larissa's mask, so his skeleton was drawn as the follow)."""
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    area = max(1, (y1 - max(0, y0)) * (x1 - max(0, x0)))
    out = {}
    for role, m in masks.items():
        if m is None or not m.any():
            continue
        inter = float(m[max(0, y0):y1, max(0, x0):x1].sum())
        if inter / m.sum() >= 0.3:
            out[role] = inter / area
    return out


def _role_of_box(box, masks: dict) -> str | None:
    """Which dancer mask a detection box belongs to (largest share of the box covered by that mask)."""
    s = _role_scores(box, masks)
    return max(s, key=s.get) if s else None


def _dancer_people(people, i: int, masks: dict, img_shape) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """[(role or 'cand', box, kpts)] for the dancers in frame i."""
    valid = people["valid"][i]
    boxes, kpts = people["boxes"][i][valid], people["kpts"][i][valid]
    out = []
    if any(m is not None for m in masks.values()):
        # one box per role: best (box share covered by the mask) first
        cand = sorted(((s, role, j) for j, b in enumerate(boxes) for role, s in _role_scores(b, masks).items()),
                      reverse=True)
        used_b, used_r = set(), set()
        for s, role, j in cand:
            if j in used_b or role in used_r:
                continue
            used_b.add(j)
            used_r.add(role)
            out.append((role, boxes[j], kpts[j]))
        return out
    # no masks yet: the two largest, most central boxes
    h, w = img_shape[:2]
    if len(boxes) == 0:
        return out
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) / (w * h)
    cx = ((boxes[:, 0] + boxes[:, 2]) / 2 - w / 2) / w
    score = area - 0.3 * np.abs(cx)
    for j in np.argsort(-score)[:2]:
        out.append(("cand", boxes[j], kpts[j]))
    return out


def _draw_kpts(img, k, color):
    for a, b in COCO_EDGES:
        if k[a, 2] > 0.3 and k[b, 2] > 0.3:
            cv2.line(img, tuple(int(v) for v in k[a, :2]), tuple(int(v) for v in k[b, :2]), color, 3, cv2.LINE_AA)
    for p in k:
        if p[2] > 0.3:
            cv2.circle(img, (int(p[0]), int(p[1])), 4, color, -1, cv2.LINE_AA)


def _project_smplx(work: Path, cam: str, i: int, t: float) -> dict:
    """SMPL-X joints of both dancers projected with the camera of frame i (needs smplx/ + cameras/)."""
    cam_file = work / "cameras" / f"{cam}.npz"
    out = {}
    if not cam_file.exists():
        return out
    c = np.load(cam_file)
    if "ok" in c.files and not bool(c["ok"][i]):
        return out
    K, R, tv = c["K"][i], c["R"][i], c["tvec"][i]
    wfs = work / "cameras" / "world_from_sfm.json"
    if wfs.exists():  # smplx is in WORLD; cameras are in SfM -> bring joints back to SfM
        w = json.loads(wfs.read_text())
        s, Rw, tw = float(w["s"]), np.array(w["R"]), np.array(w["t"])
    else:
        s, Rw, tw = 1.0, np.eye(3), np.zeros(3)
    for role in ("lead", "follow"):
        f = work / "smplx" / f"{role}.npz"
        if not f.exists():
            continue
        d = np.load(f)
        k = int(np.argmin(np.abs(d["t"] - t)))
        Xw = d["joints"][k][:22]
        Xs = ((Xw - tw) @ Rw) / s
        Xc = Xs @ R.T + tv
        uv = Xc @ K.T
        out[role] = uv[:, :2] / uv[:, 2:3]
    return out


def render(take: Take, instants: int = 6, out_dir: Path | None = None, cams: list[str] | None = None) -> list[Path]:
    work = take.work
    out_dir = out_dir or work / "review"
    out_dir.mkdir(parents=True, exist_ok=True)
    cams = cams or [s.id for s in take.sources if (work / "frames" / s.id / "frames.json").exists()]
    t0, t1 = take.window
    times = np.linspace(t0 + 0.4, t1 - 0.4, instants)
    metas = {c: _frames(work, c) for c in cams}
    people = {c: np.load(work / "people" / f"{c}.npz") for c in cams if (work / "people" / f"{c}.npz").exists()}

    sheets = {"reid": [], "pose": [], "people": []}
    have_smplx = (work / "smplx" / "lead.npz").exists()
    # sync-corrected time: the fitted bodies are on t + delta (avatar_texture.view_time_offsets); the rows pick each
    # camera's frame nearest the row time in corrected time and project the bodies at that frame's corrected time
    from .avatar_texture import view_time_offsets
    from .mamma_dyn import apply_offset
    offs = view_time_offsets(work)
    for c in cams:
        if c in offs:
            metas[c] = {**metas[c], "frames": [{**f, "t": float(apply_offset(np.array([f["t"]]), offs[c])[0])}
                                               for f in metas[c]["frames"]]}
    if have_smplx:
        sheets["smplx"] = []
    for t in times:
        rows = {k: [] for k in sheets}
        for c in cams:
            i, dt = _nearest(metas[c], float(t))
            img = cv2.imread(str(work / "frames" / c / metas[c]["frames"][i]["file"]))
            masks = _role_masks(work, c, i)
            notes = {}  # why a panel shows nothing (drawn after the resize, readable at tile size)
            # reid
            reid = img.copy()
            if any(m is not None for m in masks.values()):
                for role, color in (("lead", LEAD), ("follow", FOLLOW)):
                    if masks[role] is not None and masks[role].any():
                        reid = _tint(reid, masks[role], color)
                        ys, xs = np.nonzero(masks[role])
                        _label(reid, "L" if role == "lead" else "F", (int(xs.mean()) - 12, max(40, int(ys.min()) - 10)),
                               color, 1.4)
            elif c in people:
                for role, b, _ in _dancer_people(people[c], i, masks, img.shape):
                    cv2.rectangle(reid, tuple(int(v) for v in b[:2]), tuple(int(v) for v in b[2:]), NEUTRAL, 3)
                notes["reid"] = "no masks this frame"
            if any(m is not None for m in masks.values()):  # one role without a mask: say so (not a re-ID failure)
                miss = [r for r in ("lead", "follow") if masks[r] is None or not masks[r].any()]
                if miss:
                    notes["reid"] = f"no {'+'.join(miss)} mask"
            # pose
            pose = img.copy()
            if c in people:
                for role, b, k in _dancer_people(people[c], i, masks, img.shape):
                    _draw_kpts(pose, k, LEAD if role == "lead" else FOLLOW if role == "follow" else NEUTRAL)
            # people
            ppl = img.copy()
            pm = _mask(work / "people_masks" / c / f"{i:06d}.png")
            if pm is not None:
                ppl = _tint(ppl, pm, (150, 150, 150), 0.55, 1)
            if c in people:
                v = people[c]["valid"][i]
                for b in people[c]["boxes"][i][v]:
                    cv2.rectangle(ppl, tuple(int(x) for x in b[:2]), tuple(int(x) for x in b[2:]), (0, 255, 255), 2)
            tiles = {"reid": reid, "pose": pose, "people": ppl}
            if have_smplx:
                sm = img.copy()
                proj = _project_smplx(work, c, i, float(metas[c]["frames"][i]["t"]))
                for role, uv in proj.items():
                    col = LEAD if role == "lead" else FOLLOW
                    for a, b in SMPLX_EDGES:
                        cv2.line(sm, tuple(int(v) for v in uv[a]), tuple(int(v) for v in uv[b]), col, 3, cv2.LINE_AA)
                if not proj:  # say why the panel is empty (a frame without a camera pose is not a missing fit)
                    cf = work / "cameras" / f"{c}.npz"
                    why = "no camera"
                    if cf.exists():
                        with np.load(cf) as cz:
                            why = ("no camera pose" if "ok" in cz.files and not bool(cz["ok"][i])
                                   else "no fit")
                    notes["smplx"] = why
                tiles["smplx"] = sm
            for k, tile in tiles.items():
                scale = TILE_H / tile.shape[0]
                tile = cv2.resize(tile, (int(tile.shape[1] * scale), TILE_H), interpolation=cv2.INTER_AREA)
                _label(tile, f"cam {c}", (8, 30), (255, 255, 255), 0.8)
                _label(tile, f"{dt * 1000:+.0f} ms", (8, 58), (200, 200, 200), 0.6)
                if k in notes:
                    _label(tile, notes[k], (8, TILE_H - 12), NEUTRAL, 0.55)
                rows[k].append(tile)
        for k in sheets:
            row = np.concatenate(rows[k], axis=1)
            band = np.full((36, row.shape[1], 3), 24, np.uint8)
            _label(band, f"t = {t:.2f} s", (10, 26), (255, 255, 255), 0.8)
            sheets[k].append(np.concatenate([band, row], axis=0))

    written = []
    titles = {"reid": "RE-ID: lead (red) / follow (blue) masks, same instant across all cameras",
              "pose": "POSE 2D: dancer keypoints (red lead / blue follow / yellow = role unknown)",
              "people": "SPECTATOR MASKS: grey = ignored for room/camera geometry, yellow = detections",
              "smplx": "POSE 3D: MAMMA SMPL-X joints reprojected with per-frame cameras"}
    for k, rows_ in sheets.items():
        sheet = np.concatenate(rows_, axis=0)
        head = np.full((50, sheet.shape[1], 3), 12, np.uint8)
        _label(head, f"{take.name}  {titles[k]}", (12, 34), (255, 255, 255), 0.9)
        sheet = np.concatenate([head, sheet], axis=0)
        if sheet.shape[1] > 3600:  # keep the file viewable
            s = 3600 / sheet.shape[1]
            sheet = cv2.resize(sheet, (3600, int(sheet.shape[0] * s)), interpolation=cv2.INTER_AREA)
        p = out_dir / f"{k}.jpg"
        cv2.imwrite(str(p), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
        written.append(p)
    return written
