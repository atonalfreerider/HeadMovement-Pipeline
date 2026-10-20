r"""john_lula LHM++ appearance preview (2026-10-19): LHM++ Gaussian avatars of John (lead) and Lula (follow) from the ONE
handheld phone of the Jack & Jill heat (+ the sharp photo GM6B6189.jpg), posed by the take's MAMMA mono fit
(smplx/{lead,follow}.npz) and rendered beside the current textured avatar, for the owner's side-by-sides.

Owner, 2026-10-19: "Yes, run the LHM++ appearance preview on john_lula now (photo + best video crops, about $0.20) and send
me side-by-sides."

REUSES the larissa_kadu_38 compare code unchanged: cand_lhmpp (lhm_crop / PadRatioWithScale crops, interp_params, the
yaw / visibility / sharpness definitions, _greedy, the white-patch survey helpers _body_zbuffer / _holes / OCC_LIMITS /
hand_face_mask, check_transfer), remote/cand_lhmpp/install.sh + lhmpp_compare_pod.py (the pod side, generic: refs/,
poses.json, cameras.json) and appearance_avatar_ags.AvatarRenderer (the current avatar). Nothing of the larissa_kadu_38
paths is written.

    python -m dancecap.mono4d.jl_lhmpp_preview frames        # re-create work/john_lula/frames/01/ + frames.json
    python -m dancecap.mono4d.jl_lhmpp_preview mask-check    # which people-table index is John / Lula (sheet)
    python -m dancecap.mono4d.jl_lhmpp_preview photo-seg     # SAM 2.1 (CPU) masks of John and Lula in the photo
    python -m dancecap.mono4d.jl_lhmpp_preview prep [--reuse-survey]
    python -m dancecap.mono4d.jl_lhmpp_preview job           # remote/cand_lhmpp/pod_job_jl_preview.json
    (DANCECAP_RUNPOD_CAP=... python -m dancecap remote run --jobs <that json> --max-cost 0.40 --gpus ...)
    python -m dancecap.mono4d.jl_lhmpp_preview score         # pod (LPIPS) or local (--no-lpips)
    python -m dancecap.mono4d.jl_lhmpp_preview import [--session SID]
    python -m dancecap.mono4d.jl_lhmpp_preview avatar [--device cpu] [--ss 2]
    python -m dancecap.mono4d.jl_lhmpp_preview sheets

FRAMES. The take's ingest (dancecap.ingest.decode_range: ffmpeg, passthrough, showinfo PTS, JPEG q95) over the whole
video. The take's window [0, 88.08] would drop the last frame (pts 88.0843 s) while cameras/01.npz, the people tables
and the fit's camera inputs have all 2643 video frames, so the decode end is the last camera pts + half a frame;
canonical index i = video frame index (checked: count and every pts against cameras/01.npz pts and
frames/video_pts.json). t = pts (identity clock: slope 1, offset 0).

MASKS. people/masks_01.npz (dancecap.jl_people.finalize: m_<frame>_<who> = PNG bytes of the full-frame 960 x 538
mask, who 0 = John / lead, 1 = Lula / follow, as jl_avatars.WHO); people/teachers_01.npz role_conf (bib + colour
evidence) gates the identity.

CROPS (LHM++ input format exactly as cand_lhmpp: tight mask bbox, white outside the mask, PadRatioWithScale 504 x 840;
00 = the most frontal crop). Video: 8 per dancer from the whole take (stride 2), w8 rules - vis (projected fit joints
inside the dancer's own dilated mask) >= 0.9, all joints inside the image, mask not cut by the border, quality q = vis^2
x min(1, area / area_p90) x min(1, sharp / sharp_median)^0.5 (area against the 90th percentile, not the median: the
camera zooms, zoomed-in frames are preferred), greedy best frontal first then q x (yaw distance / 180 + 0.15),
>= MIN_DT_S apart - and the white-patch fix: no OCC_LIMITS exceeded (white_frac: projected body without hands that the
own mask does not cover = partner / another person in front, mask holes; bad_hole_frac), plus here: role_conf >= 0.9,
not within 0.4 s of a zoom step, mask leak (own mask outside both projected bodies dilated by LEAK_DILATE_M) <=
LEAK_LIMIT (another person merged into the mask), and no frame of the clip window +- 0.5 s (held out).
Photo: SAM 2.1 hiera-large image predictor on CPU, box + point prompts per person, overlap resolved by logit (masks
photo/<role>_mask.png, overlay.jpg). John's visible mask is two parts: head + shoulders above Lula's arm, and the
t-shirt / trousers below it - her arm across his chest is a white band between them in a full-mask crop (the
white-patch case the video rules drop). So John's photo crop = head + shoulders: his mask above the first row where
Lula's mask enters his columns (the frontal face reference, no white patch); Lula's = her whole visible mask (back
view: ponytail, top, bib, trousers to mid-thigh; John's forearms across her back stay white).
Variants: lhmpp_video8 = the 8 video crops; lhmpp_video8_photo = the same + the photo crop (John: head + shoulders
photo FIRST as the frontal reference; Lula: photo appended); lhmpp_video8_photofull = DIAGNOSTIC, as video8_photo but
John's photo crop is his whole visible mask (head to mid-thigh with the white band): how LHM++ takes a partial /
occluded body image.

VIEWS (none is an input frame): held-out = 8 instants spread over the dance (one per eighth), both dancers visible and
large, >= 1 s from every crop frame: the phone's own camera at native 538 x 960 (role map) + the couple box re-rendered
at 1080 px on the long side (zoom); turntable = 12 views at one instant of the clip, radius 3 m, height 1.6 m, look at
(pelvis x, 1.0 m, pelvis z), 50 deg vertical FOV, 1080 x 1080, tt00 = the phone's azimuth; clip = 120 frames (~4 s)
with the most follow turning while both dancers are in view (>= 90 % of the frames) and no zoom step: (a) clip_phone = the phone's own track
(each frame's K, R, tvec, native 538 x 960), (b) clip_vr_seat = a fixed seat 2.5 m from the clip's mean pelvis at 1.6 m,
azimuth = the phone's + 180 deg (the largest gap of a one-camera rig), 55 deg vertical FOV, 1080 x 1080, look-at
height that keeps the couple in frame (appearance_compare's rule), posed at the fit's 30 Hz grid times.
Cameras: load_tracks (cameras/01.npz + world_from_sfm.json = identity), OpenCV, pinhole (phone k1 -0.012 ignored).
Likeness data (frames, photo, crops, splats, renders) stays in work/ and our own pod; nothing here is committed.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from ..take import PACKAGE_ROOT, WORKSPACE
from . import cand_lhmpp as CL

TAKE, CAM = "john_lula", "01"
ROLES = ("lead", "follow")
WHO = {"lead": 0, "follow": 1}            # people/masks_01.npz index (jl_avatars.WHO)
NAME = {"lead": "John", "follow": "Lula"}
WORK = PACKAGE_ROOT / "work" / TAKE
OUT = WORK / "lhmpp_preview"
INPUT = OUT / "input"                     # the pod's only input
POD_OUT = OUT / "pod_out"
RENDERS = OUT / "renders"
SOURCE = WORKSPACE / "dances" / "John-Lula"   # git-ignored junction to the owner's folder (the take's source_dir)
VIDEO = SOURCE / "john_lula_jnj.mp4"
PHOTO = SOURCE / "GM6B6189.jpg"
VARIANTS = {
    "lhmpp_video8": "LHM++ from 8 crops of the ONE phone video (whole take, w8 rules + white-patch / occlusion drops)",
    "lhmpp_video8_photo": "LHM++ from the same 8 video crops + the photo crop (John: head + shoulders photo crop "
                          "first as the frontal reference; Lula: back-view photo crop appended)",
    "lhmpp_video8_photofull": "DIAGNOSTIC: as lhmpp_video8_photo but John's photo crop is his whole visible mask (head "
                              "to mid-thigh, Lula's arm across his chest white): a partial / occluded body image",
}
METHODS = ("avatar", "lhmpp_video8", "lhmpp_video8_photo", "lhmpp_video8_photofull")
N_CROPS = 8
MIN_DT_S = 2.0
STRIDE = 2
ZOOM_EVENTS = ((17.84, 18.57), (51.51, 52.24), (78.75, 79.38))   # jl_avatars.ZOOM_FALLBACK (camera stage)
ZOOM_MARGIN_S = 0.4
ROLE_CONF_MIN = 0.9
LEAK_DILATE_M = 0.10
LEAK_LIMIT = 0.20
CLIP_FRAMES = 120
HELDOUT_N = 8
HELDOUT_GAP_S = 1.0
ZOOM_LONG = 1080
TURNTABLE = {"n": 12, "radius_m": 3.0, "height_m": 1.6, "look_y_m": 1.0, "size": 1080, "vfov_deg": 50.0}
VR_SEAT = {"dist_m": 2.5, "height_m": 1.6, "size": 1080, "vfov_deg": 55.0, "azimuth_from_phone_deg": 180.0}
AVATAR_CAPTION = "current avatar: offline render of the shipped texture (no hair cards / ponytail; Unity adds them)"


def log(msg: str) -> None:
    print(f"[jl_lhmpp] {msg}", flush=True)


def _rel(p: Path) -> str:
    try:
        return Path(p).resolve().relative_to(PACKAGE_ROOT.resolve()).as_posix()
    except ValueError:
        return str(p)


def _ws(p: Path) -> str:
    return Path(p).resolve().relative_to(WORKSPACE.resolve()).as_posix()


# ================================================================================================ frames
def frames(force: bool = False) -> dict:
    from ..ingest import decode_range, probe
    from ..take import Take
    take = Take.load(TAKE)
    src = next(s for s in take.sources if s.id == CAM)
    out = WORK / "frames" / CAM
    fj = out / "frames.json"
    with np.load(WORK / "cameras" / f"{CAM}.npz") as z:
        pts_cam = np.asarray(z["pts"], float)
        t_cam = np.asarray(z["t"], float)
    vp = np.asarray(json.loads((WORK / "frames" / "video_pts.json").read_text())["pts"], float)
    F = len(pts_cam)
    if fj.exists() and not force:
        meta = json.loads(fj.read_text())
    else:
        info = probe(src.path)
        t0 = time.time()
        if out.exists():
            for p in out.glob("*.jpg"):
                p.unlink()
        end = float(pts_cam[-1]) + 0.5 / max(info["avg_fps"], 1.0)
        recs = decode_range(src.path, src.to_source(0.0), end, out)
        for r in recs:
            r["t"] = src.to_reference(r["pts"])
        dts = np.diff([r["pts"] for r in recs])
        meta = {"camera": CAM, "source": src.path.name, "width": info["width"], "height": info["height"],
                "rotation": info["rotation"], "slope": src.slope, "offset": src.offset, "frame_count": len(recs),
                "fps_median": float(1 / np.median(dts)) if len(dts) else 0.0,
                "frame_interval_ms": {"min": float(dts.min() * 1000), "max": float(dts.max() * 1000)} if len(dts) else {},
                "frames": recs,
                "recreated": {"by": "dancecap.mono4d.jl_lhmpp_preview frames (dancecap.ingest.decode_range)",
                              "when": time.strftime("%Y-%m-%dT%H:%M:%S"), "decode_s": round(time.time() - t0, 1),
                              "end_s": round(end, 6),
                              "why": "frames were not on disk; window end = last camera pts + half a frame so the "
                                     "canonical indices are the 2643 video frames of cameras/01.npz (the take's window "
                                     "0-88.08 s would drop the last frame at 88.0843 s)"}}
        fj.write_text(json.dumps(meta, indent=1))
    pts = np.array([f["pts"] for f in meta["frames"]], float)
    chk = {"frames": len(pts), "camera_frames": F, "video_pts": len(vp),
           "max_abs_pts_vs_camera_s": float(np.abs(pts - pts_cam).max()) if len(pts) == F else None,
           "max_abs_pts_vs_video_pts_s": float(np.abs(pts - vp).max()) if len(pts) == len(vp) else None,
           "max_abs_t_vs_camera_t_s": float(np.abs(np.array([f["t"] for f in meta["frames"]]) - t_cam).max())
           if len(pts) == F else None,
           "files_on_disk": sum(1 for _ in out.glob("*.jpg")), "size": [meta["width"], meta["height"]]}
    ok = (len(pts) == F == len(vp) and chk["max_abs_pts_vs_camera_s"] < 1e-4 and chk["max_abs_pts_vs_video_pts_s"] < 1e-4
          and chk["files_on_disk"] == F)
    chk["ok"] = bool(ok)
    log(f"frames: {json.dumps(chk)}")
    if not ok:
        raise SystemExit("frames do not match cameras/01.npz")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "frames_check.json").write_text(json.dumps(chk, indent=1))
    return chk


def frame_path(i: int) -> Path:
    return WORK / "frames" / CAM / f"{int(i):06d}.jpg"


def read_rgb(i: int) -> np.ndarray:
    return cv2.imread(str(frame_path(i)), cv2.IMREAD_COLOR)[:, :, ::-1]


# ================================================================================================ masks
class Masks:
    """people/masks_01.npz: full-frame PNG masks per frame and dancer (who 0 = John, 1 = Lula)."""

    def __init__(self):
        self.z = np.load(WORK / "people" / f"masks_{CAM}.npz")
        self.keys = set(self.z.files)
        self._c: dict = {}

    def get(self, i: int, role: str) -> np.ndarray | None:
        k = f"m_{int(i)}_{WHO[role]}"
        if k not in self.keys:
            return None
        if k not in self._c:
            if len(self._c) > 64:
                self._c.clear()
            m = cv2.imdecode(np.frombuffer(self.z[k].tobytes(), np.uint8), cv2.IMREAD_UNCHANGED)
            self._c[k] = None if m is None else (m > 0)
        return self._c[k]


def load_teachers() -> dict:
    with np.load(WORK / "people" / f"teachers_{CAM}.npz", allow_pickle=True) as z:
        return {k: z[k] for k in ("role_conf", "skel_in_partner", "edge", "bib_dist", "visible")}


def mask_check(times=(5.0, 20.0, 31.0, 44.0, 60.0, 72.0, 85.0)) -> Path:
    """John (green) / Lula (red) masks over frames of the whole take, with role_conf and the bib distance."""
    M = Masks()
    T = load_teachers()
    meta = json.loads((WORK / "frames" / CAM / "frames.json").read_text())
    pts = np.array([f["pts"] for f in meta["frames"]])
    tiles = []
    for t in times:
        i = int(np.argmin(np.abs(pts - t)))
        img = cv2.imread(str(frame_path(i)))
        ov = img.copy()
        for role, col in (("lead", (0, 220, 0)), ("follow", (0, 0, 240))):
            m = M.get(i, role)
            if m is not None:
                ov[m] = (0.45 * ov[m] + 0.55 * np.array(col)).astype(np.uint8)
        lab = f"{i} t={pts[i]:.1f} rc {T['role_conf'][i, 0]:.2f}/{T['role_conf'][i, 1]:.2f}"
        cv2.putText(ov, lab, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(ov, lab, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(np.concatenate([img, ov], 1))
    rows = [np.concatenate(tiles[k:k + 4], 1) for k in range(0, len(tiles), 4)]
    W = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, W - r.shape[1]), (0, 0)), constant_values=255) for r in rows]
    sheet = np.concatenate(rows, 0)
    s = min(1.0, 2400 / sheet.shape[1])
    sheet = cv2.resize(sheet, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "mask_check.jpg"
    cv2.imwrite(str(p), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    log(f"mask check -> {p} (green = index 0 = John/lead, red = index 1 = Lula/follow)")
    return p


# ================================================================================================ photo segmentation
# prompts in the photo's native pixels (1654 x 2480), read off the image: John faces the camera (head top right of
# centre, black t-shirt, black trousers bottom right), Lula from behind (ponytail top left, DREAM-print top, bib 116,
# grey trousers bottom left); her left arm crosses his chest to his left shoulder, his arms wrap her back.
PHOTO_PROMPTS = {  # lead: head incl. hair, t-shirt, trousers - NOT his bare arms across Lula's back (crop = body)

    "lead": {"box": [600, 60, 1400, 2480],
             "pos": [[880, 440], [900, 650], [1150, 1400], [1100, 1900], [1150, 2350], [1260, 900], [870, 165],
                     [990, 225], [750, 255], [1240, 2200]],
             "neg": [[420, 600], [300, 2150], [700, 1050], [1000, 950], [520, 1700]]},
    "follow": {"box": [0, 250, 1460, 2480],
               "pos": [[420, 600], [520, 1100], [300, 2150], [700, 2300], [1000, 950], [500, 1700]],
               "neg": [[880, 440], [1150, 1400], [1100, 1900], [1150, 2350]]},
}


def photo_seg(device: str = "cpu") -> dict:
    """SAM 2.1 hiera-large image predictor, one box + points prompt per person (multimask, best score), overlap
    resolved by the larger logit; masks + overlays under lhmpp_preview/photo/ (LOOK at them)."""
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from ..dancers_sam import CHECKPOINT, CONFIG
    out = OUT / "photo"
    out.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(PHOTO), cv2.IMREAD_COLOR)[:, :, ::-1].copy()
    H, W = img.shape[:2]
    t0 = time.time()
    model = build_sam2(CONFIG, str(CHECKPOINT), device=device)
    pred = SAM2ImagePredictor(model)
    with torch.inference_mode():
        pred.set_image(img)
        logits, info = {}, {}
        for role, pr in PHOTO_PROMPTS.items():
            pts = np.array(pr["pos"] + pr["neg"], np.float32)
            lab = np.array([1] * len(pr["pos"]) + [0] * len(pr["neg"]), np.int32)
            m, sc, lo = pred.predict(point_coords=pts, point_labels=lab, box=np.array(pr["box"], np.float32),
                                     multimask_output=True, return_logits=True)
            k = int(np.argmax(sc))
            logits[role] = m[k]
            info[role] = {"scores": [round(float(s), 4) for s in sc], "chosen": k}
    a, b = logits["lead"], logits["follow"]
    masks = {"lead": (a > 0) & ~((b > 0) & (b > a)), "follow": (b > 0) & ~((a > 0) & (a >= b))}
    for role, m in masks.items():
        # keep the largest component (+ any component >= 2 % of it, e.g. a hand separated by the partner's arm)
        n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
        if n > 2:
            big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
            keep = [k for k in range(1, n) if st[k, cv2.CC_STAT_AREA] >= 0.02 * st[big, cv2.CC_STAT_AREA]]
            masks[role] = np.isin(lab, keep)
        # fill enclosed holes (speckle in the black t-shirt / trousers) that the partner's mask does not explain
        other = masks["follow" if role == "lead" else "lead"]
        hole = CL._holes(masks[role], 1) & ~other
        n2, lab2, st2, _ = cv2.connectedComponentsWithStats(hole.astype(np.uint8), connectivity=4)
        small = [k for k in range(1, n2) if st2[k, cv2.CC_STAT_AREA] <= 0.01 * masks[role].sum()]
        info[role]["holes_filled_px"] = int(np.isin(lab2, small).sum())
        masks[role] = masks[role] | np.isin(lab2, small)
        info[role].update({"area_px": int(masks[role].sum()), "box": [int(v) for v in CL._crop_box(masks[role], 0.0)],
                           "components_kept": int(len(np.unique(lab[masks[role]])))})
        cv2.imwrite(str(out / f"{role}_mask.png"), masks[role].astype(np.uint8) * 255)
    ov = img[:, :, ::-1].copy()
    for role, col in (("lead", (0, 220, 0)), ("follow", (0, 0, 240))):
        m = masks[role]
        ov[m] = (0.5 * ov[m] + 0.5 * np.array(col)).astype(np.uint8)
        for (x, y) in PHOTO_PROMPTS[role]["pos"]:
            cv2.circle(ov, (x, y), 14, col, -1)
    cv2.imwrite(str(out / "overlay.jpg"), cv2.resize(ov, (W // 3, H // 3), interpolation=cv2.INTER_AREA),
                [cv2.IMWRITE_JPEG_QUALITY, 88])
    info["seconds"] = round(time.time() - t0, 1)
    info["model"] = f"SAM 2.1 hiera-large ({_rel(CHECKPOINT) if str(CHECKPOINT).startswith(str(PACKAGE_ROOT)) else CHECKPOINT.name}), {device}"
    info["prompts"] = PHOTO_PROMPTS
    (out / "seg.json").write_text(json.dumps(info, indent=1, default=int))
    log(f"photo masks -> {out}: {json.dumps({r: info[r] for r in ROLES})}")
    return info


# ================================================================================================ survey (white-patch fix)
PARTNER_IN_LIMIT = 0.03


def zoom_events() -> list[tuple[float, float]]:
    try:
        import tomllib
        cfg = tomllib.loads((PACKAGE_ROOT / "takes" / f"{TAKE}.toml").read_text(encoding="utf-8"))
        return [(float(a), float(b)) for a, b, *_ in cfg["cameras"][CAM]["zoom_events"]]
    except Exception:  # noqa: BLE001
        return list(ZOOM_EVENTS)


def in_zoom(t: float, ev) -> bool:
    return any(a - ZOOM_MARGIN_S <= t <= b + ZOOM_MARGIN_S for a, b in ev)


def extra_occ(torch, i: int, t: float, tr, bodies: dict, faces, masks: dict, Wimg: int, Himg: int, scale: float) -> dict:
    """Mask leak and partner-in-mask at the occlusion survey's raster scale: leak = own mask outside the own projected
    body dilated by LEAK_DILATE_M (another person merged into the mask; Lula's ponytail is a few %), partner_in = own
    mask on the partner's VISIBLE body (partner nearer, eroded 2 cells): the partner's pixels would enter the crop."""
    posed = {r: bodies[r].at(t) for r in ROLES}
    if any(p is None for p in posed.values()):
        return {}
    s = scale
    Ws, Hs = int(math.ceil(Wimg * s)), int(math.ceil(Himg * s))
    zb = {r: CL._body_zbuffer(torch, posed[r][0], faces, tr.K[i], tr.R[i], tr.tvec[i], s, Ws, Hs) for r in ROLES}
    ms = {r: (cv2.resize(masks[r].astype(np.float32), (Ws, Hs), interpolation=cv2.INTER_AREA) > 0.5)
          if masks.get(r) is not None else np.zeros((Hs, Ws), bool) for r in ROLES}
    n_dil = max(1, int(round(LEAK_DILATE_M / CL.OCC_CELL_M)))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * n_dil + 1, 2 * n_dil + 1))
    out = {}
    for r in ROLES:
        o = ROLES[1 - ROLES.index(r)]
        own = np.isfinite(zb[r])
        po = np.isfinite(zb[o])
        dil = cv2.dilate(own.astype(np.uint8), ker) > 0
        na = max(1, int(ms[r].sum()))
        vis_o = po & ~(own & (zb[r] < zb[o]))
        vis_o = cv2.erode(vis_o.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2).astype(bool)
        out[r] = {"leak_frac": round(float((ms[r] & ~dil).sum() / na), 4),
                  "partner_in_frac": round(float((ms[r] & vis_o).sum() / na), 4)}
    return out


def _bodies_faces(fits: dict):
    import smplx
    import torch

    from ..avatar_texture import load_bodies, model_dir_for
    bodies, _ = load_bodies(WORK, head_fix=False)
    d = fits["lead"]
    sm = smplx.create(str(model_dir_for(d)), model_type="smplx", gender="neutral", ext="npz", use_pca=False,
                      flat_hand_mean=True, num_betas=int(d.get("num_betas", 16)))
    faces_np = np.asarray(sm.faces, np.int64)
    return bodies, torch.tensor(faces_np), CL.hand_face_mask(faces_np, sm.lbs_weights.detach().numpy())


def survey(fits: dict, tr, stride: int = STRIDE) -> list[dict]:
    """Every stride-th frame of the take, per dancer: cand_lhmpp's candidate stats (vis, inimg, border, area, box,
    sharpness, yaw) + role_conf, zoom step, and on frames where a dancer passes the base gate the white-patch survey
    (cand_lhmpp.occlusion_stats) + extra_occ."""
    import torch
    t0 = time.time()
    bodies, faces, hand_faces = _bodies_faces(fits)
    M = Masks()
    T = load_teachers()
    ev = zoom_events()
    Wimg, Himg = tr.width, tr.height
    rad = max(3, int(round(0.015 * Himg)))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rad + 1, 2 * rad + 1))
    rows = []
    for i in range(0, len(tr.t), stride):
        t = float(tr.t[i])
        if not bool(tr.ok[i]):
            continue
        masks = {r: M.get(i, r) for r in ROLES}
        img, frame_rows = None, []
        for role in ROLES:
            dd = fits[role]
            m = masks[role]
            if t < dd["t"][0] - 1e-3 or t > dd["t"][-1] + 1e-3 or m is None or m.sum() < 1500:
                continue
            p = CL.interp_params(dd, t)
            J = p["joints"][list(range(22)) + [15]]
            uv, z = CL.project(J, tr.K[i], tr.R[i], tr.tvec[i])
            md_ = cv2.dilate(m.astype(np.uint8), ker) > 0
            u, v = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
            inside = (u >= 0) & (u < Wimg) & (v >= 0) & (v < Himg) & (z > 0)
            hit = np.zeros(len(u), bool)
            hit[inside] = md_[v[inside], u[inside]]
            ys, xs = np.nonzero(m)
            border = xs.min() <= 3 or ys.min() <= 3 or xs.max() >= Wimg - 4 or ys.max() >= Himg - 4
            if img is None:
                img = cv2.imread(str(frame_path(i)), cv2.IMREAD_GRAYSCALE)
            crop = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            sharp = float(cv2.Laplacian(cv2.resize(crop, None, fx=0.5, fy=0.5), cv2.CV_32F).var())
            frame_rows.append({"cam": CAM, "role": role, "i": i, "t": round(t, 5), "vis": float(hit.mean()),
                               "inimg": float(inside.mean()), "area": int(m.sum()),
                               "box": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                               "border": bool(border), "sharp": sharp,
                               "yaw_deg": round(CL.body_yaw_deg(np.asarray(p["global_orient"]), p["joints"][0],
                                                                tr.R[i], tr.tvec[i]), 1),
                               "role_conf": round(float(T["role_conf"][i, WHO[role]]), 3),
                               "zoom": in_zoom(t, ev), "W": Wimg, "H": Himg})
        occ = None
        if any(CL._base_ok(r) for r in frame_rows):
            occ = CL.occlusion_stats(torch, CAM, i, t, tr, bodies, faces, masks, Wimg, Himg, hand_faces)
            if occ:
                ex = extra_occ(torch, i, t, tr, bodies, faces, masks, Wimg, Himg, occ[ROLES[0]]["scale"])
                for r in ROLES:
                    occ[r].update(ex.get(r, {}))
        for r in frame_rows:
            r["occ"] = (occ or {}).get(r["role"])
            rows.append(r)
        if i % 300 == 0:
            log(f"survey frame {i}/{len(tr.t)} ({time.time() - t0:.0f} s)")
    log(f"survey: {len(rows)} dancer-frames in {time.time() - t0:.0f} s")
    return rows


def drop_reasons(r: dict, exclude=()) -> list[str]:
    why = []
    if r["vis"] < 0.9:
        why.append("vis")
    if r["inimg"] < 1.0:
        why.append("joints outside image")
    if r["border"]:
        why.append("mask cut by border")
    if r["role_conf"] < ROLE_CONF_MIN:
        why.append("role_conf")
    if r["zoom"]:
        why.append("zoom step")
    if any(a <= r["t"] <= b for a, b in exclude):
        why.append("held-out window")
    o = r.get("occ")
    if not why:
        if o is None:
            why.append("no occlusion stats")
        else:
            why += CL._occ_drop(o)
            if (o.get("leak_frac") or 0) > LEAK_LIMIT:
                why.append("leak_frac")
            if (o.get("partner_in_frac") or 0) > PARTNER_IN_LIMIT:
                why.append("partner_in_frac")
    return why


# ================================================================================================ selection
def select_clip(fits: dict, rows: list[dict], tr) -> dict:
    """The CLIP_FRAMES-frame window with the most follow turning (cumulative |pelvis yaw change| of the fit) while both
    dancers are in view (mask, all joints in the image, not cut, vis >= 0.6 in >= 90 % of the surveyed frames), not
    small (median smaller-dancer mask area >= 0.8 x the take's median) and no zoom step inside. (95 % / 1.0 x found no
    window: the zoomed-in stretches cut a dancer at the frame border too often.)"""
    from scipy.spatial.transform import Rotation as Rot
    ev = zoom_events()
    by: dict = {}
    for r in rows:
        by.setdefault(r["i"], {})[r["role"]] = r
    F = len(tr.t)
    both = np.zeros(F, bool)
    amin = np.full(F, np.nan)
    for i, rr in by.items():
        if all(k in rr for k in ROLES):
            both[i] = all(rr[k]["inimg"] == 1.0 and not rr[k]["border"] and rr[k]["vis"] >= 0.6 for k in ROLES)
            amin[i] = min(rr[k]["area"] for k in ROLES)
    surveyed = np.zeros(F, bool)
    surveyed[::STRIDE] = True
    a_med = float(np.nanmedian(amin))
    d = fits["follow"]
    fwd = Rot.from_rotvec(np.asarray(d["global_orient"], float)).apply([0.0, 0.0, 1.0])
    yaw = np.unwrap(np.arctan2(fwd[:, 0], fwd[:, 2]))
    tg = np.asarray(d["t"], float)
    best = None
    for i0 in range(0, F - CLIP_FRAMES, STRIDE):
        i1 = i0 + CLIP_FRAMES - 1
        ta, tb = float(tr.t[i0]), float(tr.t[i1])
        if any(not (tb < a - ZOOM_MARGIN_S or ta > b + ZOOM_MARGIN_S) for a, b in ev):
            continue
        sel = surveyed[i0:i1 + 1]
        frac = float(both[i0:i1 + 1][sel].mean())
        aw = amin[i0:i1 + 1][sel]
        am = float(np.nanmedian(aw)) if np.isfinite(aw).any() else float("nan")
        if frac < 0.90 or not np.isfinite(am) or am < 0.8 * a_med:
            continue
        k = (tg >= ta) & (tg <= tb)
        turn = float(np.degrees(np.abs(np.diff(yaw[k])).sum()))
        net = float(np.degrees(yaw[k][-1] - yaw[k][0]))
        if best is None or turn > best["follow_turn_deg"]:
            best = {"i0": i0, "i1": i1, "t0": round(ta, 5), "t1": round(tb, 5), "follow_turn_deg": round(turn, 1),
                    "follow_net_yaw_deg": round(net, 1), "both_in_view_frac": round(frac, 3),
                    "median_min_area_px": round(am, 0), "take_median_min_area_px": round(a_med, 0)}
    if best is None:
        raise SystemExit("no clip window passes the rules")
    return best


def select_crops(rows: list[dict], role: str, exclude) -> dict:
    rr = [dict(r) for r in rows if r["role"] == role]
    for r in rr:
        r["why"] = drop_reasons(r, exclude)
    base = [r for r in rr if CL._base_ok(r) and r["role_conf"] >= ROLE_CONF_MIN and not r["zoom"]]
    ok = [r for r in rr if not r["why"]]
    a90 = float(np.percentile([r["area"] for r in base], 90))
    s_med = float(np.median([r["sharp"] for r in base]))
    for r in rr:
        r["q"] = r["vis"] ** 2 * min(1.0, r["area"] / a90) * min(1.0, r["sharp"] / s_med) ** 0.5
    picks = CL._greedy(ok, N_CROPS, MIN_DT_S, [], None)
    dt_used = MIN_DT_S
    for dt in (MIN_DT_S / 2, MIN_DT_S / 4):
        if len(picks) >= N_CROPS:
            break
        picks = CL._greedy(ok, N_CROPS, dt, picks, None)
        dt_used = dt
    if picks:
        f0 = min(range(len(picks)), key=lambda k: abs(picks[k]["yaw_deg"]))
        picks.insert(0, picks.pop(f0))
    counts: dict = {}
    for r in rr:
        for w in r["why"]:
            counts[w] = counts.get(w, 0) + 1
    return {"picks": picks, "min_dt_used": dt_used, "candidates": len(rr), "base_ok": len(base), "clean": len(ok),
            "area_p90": round(a90, 1), "sharp_median": round(s_med, 2), "drop_counts": counts}


def select_heldout(rows: list[dict], crops: dict, t_span) -> list[dict]:
    """HELDOUT_N instants, one per equal part of the take: both dancers visible (vis >= 0.8), inside the image, not cut,
    role_conf >= 0.9, no zoom step, >= HELDOUT_GAP_S from every crop frame of either dancer; score = min over the
    dancers of vis x min(1, area / area_p90)."""
    by: dict = {}
    for r in rows:
        by.setdefault(r["i"], {})[r["role"]] = r
    a90 = {ro: float(np.percentile([r["area"] for r in rows if r["role"] == ro], 90)) for ro in ROLES}
    crop_t = [p["t"] for ro in ROLES for p in crops[ro]["picks"]]
    edges = np.linspace(t_span[0], t_span[1], HELDOUT_N + 1)
    out = []
    for n in range(HELDOUT_N):
        best = None
        for i, rr in by.items():
            if not all(k in rr for k in ROLES):
                continue
            t = rr["lead"]["t"]
            if not (edges[n] <= t < edges[n + 1]) or any(abs(t - c) < HELDOUT_GAP_S for c in crop_t):
                continue
            if not all(rr[k]["vis"] >= 0.8 and rr[k]["inimg"] == 1.0 and not rr[k]["border"] and
                       rr[k]["role_conf"] >= ROLE_CONF_MIN and not rr[k]["zoom"] for k in ROLES):
                continue
            s = min(rr[k]["vis"] * min(1.0, rr[k]["area"] / a90[k]) for k in ROLES)
            if best is None or s > best[0]:
                best = (s, i, t)
        if best is None:
            log(f"held-out part {n} ({edges[n]:.1f}-{edges[n + 1]:.1f} s): no frame passes")
            continue
        out.append({"n": n, "i": int(best[1]), "t": float(best[2]), "score": round(float(best[0]), 3),
                    "part_s": [round(float(edges[n]), 2), round(float(edges[n + 1]), 2)],
                    "nearest_crop_s": round(min(abs(best[2] - c) for c in crop_t), 2)})
    return out


def turntable_instant(rows: list[dict], clip: dict) -> int:
    by: dict = {}
    for r in rows:
        if clip["i0"] <= r["i"] <= clip["i1"]:
            by.setdefault(r["i"], {})[r["role"]] = r
    best = max((i for i, rr in by.items() if all(k in rr for k in ROLES)),
               key=lambda i: min(by[i][k]["vis"] * by[i][k]["area"] for k in ROLES) *
               (0.5 if any(by[i][k]["border"] for k in ROLES) else 1.0))
    return int(best)


# ================================================================================================ views, poses, crops
def load_fit(role: str) -> dict:
    with np.load(WORK / "smplx" / f"{role}.npz", allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def _pose_rec(d: dict, key: str, t: float) -> dict:
    p = CL.interp_params(d, t)
    return {"key": key, "t": float(t), "root_pose": p["global_orient"].tolist(), "body_pose": p["body_pose"].tolist(),
            "jaw_pose": p["jaw_pose"].tolist(), "leye_pose": p["leye_pose"].tolist(),
            "reye_pose": p["reye_pose"].tolist(), "lhand_pose": p["left_hand_pose"].tolist(),
            "rhand_pose": p["right_hand_pose"].tolist(), "transl": p["transl"].tolist(),
            "pelvis": p["joints"][0].tolist(), "joints22": p["joints"][:22].tolist()}


def build_views(fits: dict, tr, clip: dict, i_tt: int, heldout: list[dict]) -> tuple[list[dict], dict, dict]:
    """cameras.json renders (id, group, out, pose key, K, R, tvec, W, H, ...) + pose keys {key: t} + novel-camera info;
    copies the held-out real frames and masks into INPUT/real, INPUT/masks (for the score)."""
    from . import appearance_compare as AC
    Fk = {r: AC.Fit(WORK / "smplx" / f"{r}.npz") for r in ROLES}
    M = Masks()
    W, H = tr.width, tr.height
    renders, keys = [], {}

    def pose(key: str, t: float) -> str:
        keys.setdefault(key, float(t))
        return key

    def cam(id_, group, out, key, K, R, tvec, Wc, Hc, **kw):
        renders.append({"id": id_, "group": group, "out": out, "pose": key, "K": np.asarray(K, float).tolist(),
                        "R": np.asarray(R, float).tolist(), "tvec": np.asarray(tvec, float).reshape(3).tolist(),
                        "W": int(Wc), "H": int(Hc), **kw})

    for d in ("real", "masks"):
        if (INPUT / d).exists():
            shutil.rmtree(INPUT / d)
        (INPUT / d).mkdir(parents=True)
    for h in heldout:
        i = h["i"]
        vid = f"h{h['n']}_f{i:04d}"
        k = pose(f"f{i}", tr.t[i])
        cam(vid, "heldout", f"heldout/{vid}.png", k, tr.K[i], tr.R[i], tr.tvec[i], W, H, role_map=True, frame=i,
            t=float(tr.t[i]))
        shutil.copy2(frame_path(i), INPUT / "real" / f"{vid}.jpg")
        u = np.zeros((H, W), bool)
        for r in ROLES:
            m = M.get(i, r)
            cv2.imwrite(str(INPUT / "masks" / f"{vid}_{CL.PID[r]}.png"), m.astype(np.uint8) * 255)
            u |= m
        box = AC.pad_box(AC._box(u), W, H, 0.10)
        K2, W2, H2 = CL._zoom_K(tr.K[i], box, ZOOM_LONG)
        cam(f"{vid}_zoom", "heldout_zoom", f"heldout_zoom/{vid}.png", k, K2, tr.R[i], tr.tvec[i], W2, H2, frame=i,
            t=float(tr.t[i]), view_id=vid, box=[int(x) for x in box])
        h["id"], h["box"] = vid, [int(x) for x in box]
    # ---- turntable (tt00 = the phone's azimuth at that instant)
    t_tt = float(tr.t[i_tt])
    pel = np.mean([Fk[r].joints(t_tt)[0] for r in ROLES], 0)
    target = np.array([pel[0], TURNTABLE["look_y_m"], pel[2]])
    a0 = AC.azimuth(AC.centre(tr.R[i_tt], tr.tvec[i_tt]), pel)
    S = TURNTABLE["size"]
    f_tt = AC._fov_f(S, TURNTABLE["vfov_deg"])
    P_tt = AC._couple_points(Fk, t_tt)
    k = pose(f"f{i_tt}", t_tt)
    tt = []
    for n in range(TURNTABLE["n"]):
        a = math.radians(a0 + 360.0 * n / TURNTABLE["n"])
        C = np.array([pel[0] + TURNTABLE["radius_m"] * math.sin(a), TURNTABLE["height_m"],
                      pel[2] + TURNTABLE["radius_m"] * math.cos(a)])
        c = AC.look_at(C, target, f_tt, S, S)
        mg = AC._frame_margin(P_tt, c)
        cam(f"tt{n:02d}", "turntable", f"turntable/tt{n:02d}.png", k, c["K"], c["R"], c["tvec"], S, S, t=t_tt,
            azimuth_deg=round(math.degrees(a) % 360, 2), azimuth_from_phone_deg=round(360.0 * n / TURNTABLE["n"], 1),
            frame_margin=round(mg, 3))
        tt.append(round(mg, 3))
    # ---- clip (a): the phone's own track
    for j, i in enumerate(range(clip["i0"], clip["i1"] + 1)):
        cam(f"clip_phone/{j:04d}", "clip_phone", f"clip_phone/{j:04d}.png", pose(f"f{i}", tr.t[i]), tr.K[i], tr.R[i],
            tr.tvec[i], W, H, clip_index=j, frame=i, t=float(tr.t[i]))
    # ---- clip (b): the VR seat, posed at the fit's grid times
    tg = np.asarray(fits["lead"]["t"], float)
    k0 = int(math.ceil(clip["t0"] * 30.0 - 1e-6))
    ks = list(range(k0, min(len(tg), k0 + CLIP_FRAMES)))
    pel_clip = np.mean([np.mean([Fk[r].joints(tg[kk])[0] for r in ROLES], 0) for kk in ks], 0)
    C_ph = np.mean([AC.centre(tr.R[i], tr.tvec[i]) for i in range(clip["i0"], clip["i1"] + 1)], 0)
    a_ph = AC.azimuth(C_ph, pel_clip)
    a_seat = (a_ph + VR_SEAT["azimuth_from_phone_deg"]) % 360
    Cs = np.array([pel_clip[0] + VR_SEAT["dist_m"] * math.sin(math.radians(a_seat)), VR_SEAT["height_m"],
                   pel_clip[2] + VR_SEAT["dist_m"] * math.cos(math.radians(a_seat))])
    S2 = VR_SEAT["size"]
    f2 = AC._fov_f(S2, VR_SEAT["vfov_deg"])
    P_clip = [AC._couple_points(Fk, tg[kk]) for kk in ks]
    best = None
    for y in np.arange(0.40, 1.401, 0.01):
        c = AC.look_at(Cs, [pel_clip[0], y, pel_clip[2]], f2, S2, S2)
        m = min(AC._frame_margin(P, c) for P in P_clip)
        if best is None or m > best[1] + 1e-9:
            best = (round(float(y), 2), m)
    seat = AC.look_at(Cs, [pel_clip[0], best[0], pel_clip[2]], f2, S2, S2)
    dmin = min(float(np.linalg.norm(P - Cs, axis=1).min()) for P in P_clip)
    for j, kk in enumerate(ks):
        cam(f"clip_vr_seat/{j:04d}", "clip_vr_seat", f"clip_vr_seat/{j:04d}.png", pose(f"g{kk}", tg[kk]), seat["K"],
            seat["R"], seat["tvec"], S2, S2, clip_index=j, grid_k=kk, t=float(tg[kk]))
    novel = {"turntable": {"frame": i_tt, "t": t_tt, "centre_pelvis": np.round(pel, 4).tolist(),
                           "look_at": np.round(target, 4).tolist(), "phone_azimuth_deg": round(a0, 2),
                           "frame_margins": tt, **TURNTABLE},
             "vr_seat": {"C": np.round(Cs, 4).tolist(), "azimuth_deg": round(a_seat, 2),
                         "phone_azimuth_deg": round(a_ph, 2), "look_at": [round(float(pel_clip[0]), 4), best[0],
                                                                          round(float(pel_clip[2]), 4)],
                         "frame_margin_min": round(best[1], 3), "closest_dancer_point_m": round(dmin, 3),
                         "grid_k": [ks[0], ks[-1]], **VR_SEAT}}
    return renders, keys, novel


def photo_head_cut(lead_m: np.ndarray, follow_m: np.ndarray | None = None) -> int:
    """First row below John's face where Lula's photo mask covers > 2 % of the central columns of his mask."""
    if follow_m is None:
        follow_m = cv2.imread(str(OUT / "photo" / "follow_mask.png"), cv2.IMREAD_GRAYSCALE) > 127
    ys, xs = np.nonzero(lead_m)
    x0, x1 = int(np.percentile(xs, 10)), int(np.percentile(xs, 90))
    y_face = int(PHOTO_PROMPTS["lead"]["pos"][0][1])
    for y in range(y_face, lead_m.shape[0]):
        if follow_m[y, x0:x1].mean() > 0.02:
            return y
    return lead_m.shape[0]


def photo_head_mask(lead_m: np.ndarray) -> np.ndarray:
    """John's head + shoulders: his photo mask above photo_head_cut, the component holding the face."""
    m = lead_m.copy()
    m[photo_head_cut(lead_m):] = False
    n, lab, _, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    fx, fy = PHOTO_PROMPTS["lead"]["pos"][0]
    return lab == lab[fy, fx] if lab[fy, fx] > 0 else m


def write_refs(crops: dict) -> dict:
    """INPUT/refs/<role>/<variant>/NN.png in LHM++'s input format + inputs.jpg (every chosen crop, LOOK at it)."""
    M = Masks()
    d = INPUT / "refs"
    if d.exists():
        shutil.rmtree(d)
    photo = cv2.imread(str(PHOTO), cv2.IMREAD_COLOR)[:, :, ::-1]
    u8 = lambda c: (np.clip(c, 0, 1) * 255 + 0.5).astype(np.uint8)  # noqa: E731
    out, rows_img = {}, []
    for role in ROLES:
        picks = crops[role]["picks"]
        vids = [u8(CL.lhm_crop(read_rgb(p["i"]), M.get(p["i"], role))) for p in picks]
        pm = cv2.imread(str(OUT / "photo" / f"{role}_mask.png"), cv2.IMREAD_GRAYSCALE) > 127
        ph_full = u8(CL.lhm_crop(photo, pm))
        ph = ph_full
        if role == "lead":
            ph, cut = u8(CL.lhm_crop(photo, photo_head_mask(pm))), photo_head_cut(pm)
            out.setdefault("_photo", {})["lead_head_cut_row"] = cut
        vk = [("video", k) for k in range(len(vids))]
        sets = {"lhmpp_video8": vk,
                "lhmpp_video8_photo": [("photo", 0)] + vk if role == "lead" else vk + [("photo", 0)],
                "lhmpp_video8_photofull": [("photo_full", 0)] + vk if role == "lead" else vk + [("photo", 0)]}
        out[role] = {}
        for var, seq in sets.items():
            sd = d / role / var
            sd.mkdir(parents=True, exist_ok=True)
            files = []
            for n, (kind, k) in enumerate(seq):
                im = ph if kind == "photo" else ph_full if kind == "photo_full" else vids[k]
                cv2.imwrite(str(sd / f"{n:02d}.png"), im[:, :, ::-1])
                files.append({"slot": n, "kind": kind, **({"frame": int(picks[k]["i"]), "t": picks[k]["t"]}
                                                         if kind == "video" else {"file": PHOTO.name, "crop":
                                                         "head + shoulders" if (role == "lead" and kind == "photo")
                                                         else "whole visible mask"})})
            out[role][var] = files
        tiles = []
        for k, (p, im) in enumerate(zip(picks, vids)):
            o = p.get("occ") or {}
            tiles.append(CL._tile(np.ascontiguousarray(im[:, :, ::-1]), (0, 0, im.shape[1], im.shape[0]), 300,
                                  f"{k:02d} f{p['i']} {p['t']:.1f}s {p['yaw_deg']:.0f}d",
                                  f"w{o.get('white_frac', 0):.3f} q{p['q']:.2f}"))
        tiles.append(CL._tile(np.ascontiguousarray(ph[:, :, ::-1]), (0, 0, ph.shape[1], ph.shape[0]), 300,
                              "PHOTO " + ("head (1st, +photo)" if role == "lead" else "(last in +photo)")))
        if role == "lead":
            tiles.append(CL._tile(np.ascontiguousarray(ph_full[:, :, ::-1]), (0, 0, ph.shape[1], ph.shape[0]), 300,
                                  "PHOTO full (1st, diag.)"))
        row = np.concatenate([np.pad(t, ((0, 0), (0, 4), (0, 0)), constant_values=255) for t in tiles], 1)
        lab = np.full((28, row.shape[1], 3), 255, np.uint8)
        cv2.putText(lab, f"{NAME[role]} ({role}): 8 video crops (phone 01, whole take) + the photo crop", (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        rows_img.append(np.concatenate([lab, row], 0))
    Wm = max(r.shape[1] for r in rows_img)
    rows_img = [np.pad(r, ((0, 6), (0, Wm - r.shape[1]), (0, 0)), constant_values=255) for r in rows_img]
    cv2.imwrite(str(OUT / "inputs.jpg"), np.concatenate(rows_img, 0), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return out


def dropped_sheet(rows: list[dict], role: str, exclude, n: int = 8) -> Path:
    """The best-q base-gate frames the white-patch / occlusion rules dropped, with the reason (check the rules)."""
    M = Masks()
    sel = select_crops(rows, role, exclude)
    q = {(r["i"]): r for r in rows if r["role"] == role}
    occ_keys = {"white_frac", "bad_hole_frac", "leak_frac", "partner_in_frac"}
    rr = [dict(r) for r in rows if r["role"] == role]
    a90, s_med = sel["area_p90"], sel["sharp_median"]
    cand = []
    for r in rr:
        why = drop_reasons(r, exclude)
        if why and set(why) <= occ_keys:
            r["why"] = why
            r["q"] = r["vis"] ** 2 * min(1.0, r["area"] / a90) * min(1.0, r["sharp"] / s_med) ** 0.5
            cand.append(r)
    cand = sorted(cand, key=lambda r: -r["q"])
    chosen = []
    for r in cand:
        if all(abs(r["t"] - c["t"]) >= 1.0 for c in chosen):
            chosen.append(r)
        if len(chosen) >= n:
            break
    tiles = []
    for r in chosen:
        im = (np.clip(CL.lhm_crop(read_rgb(r["i"]), M.get(r["i"], role)), 0, 1) * 255 + 0.5).astype(np.uint8)
        o = r["occ"]
        tiles.append(CL._tile(np.ascontiguousarray(im[:, :, ::-1]), (0, 0, im.shape[1], im.shape[0]), 280,
                              f"DROP f{r['i']} " + ",".join(w.replace("_frac", "") for w in r["why"]),
                              f"w{o.get('white_frac')} h{o.get('bad_hole_frac')} l{o.get('leak_frac')} "
                              f"p{o.get('partner_in_frac')}"))
    p = OUT / f"dropped_{role}.jpg"
    if tiles:
        cv2.imwrite(str(p), np.concatenate(tiles, 1), [cv2.IMWRITE_JPEG_QUALITY, 88])
    del q
    return p


def prep(reuse_survey: bool = False) -> dict:
    from .. import mamma_dyn as md
    t0 = time.time()
    tr = md.load_tracks(WORK, [CAM])[CAM]
    fits = {r: load_fit(r) for r in ROLES}
    INPUT.mkdir(parents=True, exist_ok=True)
    sp = OUT / "survey.json"
    if reuse_survey and sp.exists():
        rows = json.loads(sp.read_text())["rows"]
    else:
        rows = survey(fits, tr)
        sp.write_text(json.dumps({"stride": STRIDE, "limits": {**CL.OCC_LIMITS, "leak_frac": LEAK_LIMIT,
                                                               "partner_in_frac": PARTNER_IN_LIMIT,
                                                               "role_conf_min": ROLE_CONF_MIN},
                                  "rows": rows}, default=float))
    clip = select_clip(fits, rows, tr)
    log(f"clip: {clip}")
    exclude = [(clip["t0"] - 0.5, clip["t1"] + 0.5)]
    crops = {r: select_crops(rows, r, exclude) for r in ROLES}
    for r in ROLES:
        c = crops[r]
        log(f"{NAME[r]}: {len(c['picks'])} crops {[(p['i'], round(p['t'], 1), int(p['yaw_deg'])) for p in c['picks']]}; "
            f"clean {c['clean']} of base {c['base_ok']} / {c['candidates']}; drops {c['drop_counts']}; "
            f"min_dt {c['min_dt_used']}")
    t_span = (float(tr.t[0]), float(tr.t[-1]))
    heldout = select_heldout(rows, crops, t_span)
    i_tt = turntable_instant(rows, clip)
    renders, keys, novel = build_views(fits, tr, clip, i_tt, heldout)
    refs = write_refs(crops)
    for r in ROLES:
        dropped_sheet(rows, r, exclude)
    poses = {}
    for r in ROLES:
        d = fits[r]
        poses[r] = {"betas": np.asarray(d["betas"], float).reshape(-1).tolist(), "num_betas": int(d.get("num_betas", 16)),
                    "flat_hand_mean": bool(d.get("flat_hand_mean", True)), "model": str(d.get("model", "")),
                    "frames": [_pose_rec(d, k, t) for k, t in keys.items()]}
    (INPUT / "poses.json").write_text(json.dumps(poses))
    (INPUT / "cameras.json").write_text(json.dumps({"take": TAKE, "variants": list(VARIANTS), "renders": renders}))
    keep = ("i", "t", "yaw_deg", "vis", "inimg", "area", "sharp", "box", "role_conf", "occ", "q")
    refs_json = {"rules": {"n": N_CROPS, "min_dt_s": MIN_DT_S, "stride": STRIDE, "limits": {
        **CL.OCC_LIMITS, "leak_frac": LEAK_LIMIT, "partner_in_frac": PARTNER_IN_LIMIT}, "role_conf_min": ROLE_CONF_MIN,
        "zoom_margin_s": ZOOM_MARGIN_S, "excluded_windows": exclude},
        "photo": {"file": PHOTO.name, "seg": json.loads((OUT / "photo" / "seg.json").read_text()),
                  "lead_head_cut_row": refs.get("_photo", {}).get("lead_head_cut_row"),
                  "crops": {"lead": "head + shoulders (rows above lead_head_cut_row, the face component); "
                                    "lhmpp_video8_photofull: the whole visible mask", "follow": "whole visible mask"}},
        **{r: {"frames": [{k: p.get(k) for k in keep} for p in crops[r]["picks"]], "files": refs[r],
               **{k: crops[r][k] for k in ("min_dt_used", "candidates", "base_ok", "clean", "area_p90",
                                            "sharp_median", "drop_counts")}} for r in ROLES}}
    (INPUT / "refs.json").write_text(json.dumps(refs_json, indent=1, default=float))
    chk = CL.check_transfer(poses)
    groups: dict = {}
    for c in renders:
        groups[c["group"]] = groups.get(c["group"], 0) + 1
    info = {"take": TAKE, "clip": clip, "heldout": heldout, "novel": novel, "renders": groups, "pose_keys": len(keys),
            "transfer_check": chk, "crops": {r: len(crops[r]["picks"]) for r in ROLES},
            "bytes": int(sum(p.stat().st_size for p in INPUT.rglob("*") if p.is_file())),
            "seconds": round(time.time() - t0, 1)}
    (INPUT / "prep.json").write_text(json.dumps(info, indent=1, default=float))
    log(f"prep: {json.dumps({k: info[k] for k in ('renders', 'pose_keys', 'crops', 'bytes', 'seconds')})}")
    log(f"transfer check: {json.dumps(chk)}")
    return info


# ================================================================================================ pod job
JOB_NAME = "jl_lhmpp_preview"
JOB_PATH = PACKAGE_ROOT / "remote" / "cand_lhmpp" / "pod_job_jl_preview.json"


def job(max_minutes: float = 25) -> Path:
    rel = lambda p: _ws(p)[len("dancecap/"):]  # noqa: E731 - cwd of the commands is dancecap/
    outs = [_ws(POD_OUT) + "/"] + [_ws(RENDERS / v) + "/" for v in VARIANTS]
    d = {"schema": "dancecap.pod_job/1", "name": JOB_NAME, "take": TAKE, "max_minutes": max_minutes, "needs": [],
         "inputs": [_ws(INPUT) + "/"],
         "exclude": ["dancecap/work/runpod/pywheels", "dancecap/work/runpod/wheelhouse", "mamma-turing"],
         "env": {"LHMPP_COMMIT": CL.LHMPP_COMMIT, "LHMPP_ROOT": "/workspace/lhmpp_cand/LHM-plusplus",
                 "LHMPP_VENV": "/workspace/lhmpp_cand/venv", "LHMPP_MODEL": CL.MODEL_NAME,
                 "TORCH_HOME": "/workspace/lhmpp_cand/torch_home", "HF_HOME": "/workspace/lhmpp_cand/hf_home"},
         "commands": [
             {"name": "install", "timeout_min": 35, "cmd": "bash remote/cand_lhmpp/install.sh"},
             {"name": "render", "timeout_min": 20,
              "cmd": f"/workspace/lhmpp_cand/venv/bin/python remote/cand_lhmpp/lhmpp_compare_pod.py --input "
                     f"{rel(INPUT)} --out {rel(POD_OUT)} --renders {rel(RENDERS)} --variants {','.join(VARIANTS)}"},
             {"name": "score", "timeout_min": 6, "optional": True,
              "cmd": "(python -m pip install -q --no-deps lpips==0.1.4 || echo 'lpips install failed: no LPIPS'); "
                     "python -m dancecap.mono4d.jl_lhmpp_preview score"}],
         "outputs": outs,
         "verify": [_ws(POD_OUT / "lhmpp_compare_pod.json")] +
                   [_ws(RENDERS / v / "turntable" / "tt00.png") for v in VARIANTS] +
                   [_ws(RENDERS / v / "clip_vr_seat" / f"{CLIP_FRAMES - 1:04d}.png") for v in VARIANTS],
         "after_download": [f"python -m dancecap.mono4d.jl_lhmpp_preview import --session <sid>"],
         "notes": [f"LHM++ ({CL.MODEL_NAME}, CC BY-NC 4.0 weights) avatars of John and Lula (take john_lula) from 8 crops "
                   "of the one phone video (lhmpp_video8), those + the photo crop (lhmpp_video8_photo; diagnostic "
                   "lhmpp_video8_photofull with John's whole occluded photo mask), posed "
                   "with the john_lula MAMMA fit and rendered into held-out phone frames, a turntable and two 4 s clips "
                   "(both dancers together); weights stay on the pod; crops and frames go to our own pod only (owner "
                   "consent 2026-10-19)"]}
    JOB_PATH.parent.mkdir(parents=True, exist_ok=True)
    JOB_PATH.write_text(json.dumps(d, indent=1), newline="\n")
    return JOB_PATH


# ================================================================================================ score
def _read_rgba(p: Path):
    im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if im is None or im.ndim != 3 or im.shape[2] != 4:
        return None
    return im[:, :, 2::-1].astype(np.float32) / 255.0, im[:, :, 3].astype(np.float32) / 255.0


def _method_dir(m: str) -> Path:
    """Where a method's held-out renders are: renders/<m>/ (the pod also finds the avatar's in INPUT/avatar/)."""
    if m == "avatar" and not (RENDERS / "avatar" / "heldout").exists():
        return INPUT / "avatar"
    return RENDERS / m


def score(lpips_net: bool = True) -> dict:
    """Masked metrics of every method on the held-out phone frames (cand_lhmpp.compare_score's definitions: crop = the
    dancer's real-mask bbox + 10 %, inside the real mask, the render composited over black; *_both = real mask AND
    rendered dancer; iou / coverage of the rendered role vs the real mask)."""
    from ..dancer_splats.evaluate import _crop_scores
    cams = [c for c in json.loads((INPUT / "cameras.json").read_text())["renders"] if c["group"] == "heldout"]
    methods = [m for m in METHODS if (_method_dir(m) / "heldout").exists()]
    device, net = "cpu", None
    if lpips_net:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            from ..dancer_splats.evaluate import _lpips
            net = _lpips(device)
        except Exception as e:  # noqa: BLE001
            log(f"LPIPS unavailable: {e}")
    rows = []
    for c in cams:
        gt = CL._read_rgb(INPUT / "real" / f"{c['id']}.jpg")
        for meth in methods:
            md_ = _method_dir(meth)
            ra = _read_rgba(md_ / c["out"])
            rid = cv2.imread(str(md_ / "role" / f"{c['id']}.png"), cv2.IMREAD_COLOR)
            if ra is None or gt is None:
                continue
            rgb = ra[0] * ra[1][..., None]
            rw = None if rid is None else rid[:, :, ::-1].astype(np.float32) / 255.0
            for role in ROLES:
                m = cv2.imread(str(INPUT / "masks" / f"{c['id']}_{CL.PID[role]}.png"), cv2.IMREAD_GRAYSCALE)
                if m is None or (m > 127).sum() < 50:
                    continue
                m = m > 127
                x0, y0, x1, y1 = CL._crop_box(m)
                s = _crop_scores(rgb[y0:y1, x0:x1], gt[y0:y1, x0:x1], m[y0:y1, x0:x1], net, device)
                row = {"view": c["id"], "frame": c["frame"], "t": round(c["t"], 4), "role": role, "method": meth,
                       "crop": [int(x0), int(y0), int(x1), int(y1)],
                       **{k: (None if s.get(k) is None else float(s[k])) for k in s}}
                if rw is not None:
                    own = rw[:, :, ROLES.index(role)] > 0.5
                    u = (own | m).sum()
                    row["iou"] = float((own & m).sum() / u) if u else None
                    row["coverage"] = float((own & m).sum() / m.sum())
                    sb = (own & m)[y0:y1, x0:x1]
                    s2 = _crop_scores(rgb[y0:y1, x0:x1], gt[y0:y1, x0:x1], sb, net, device)
                    row.update({f"{k}_both": float(s2[k]) for k in ("psnr", "ssim", "lpips", "psnr_cc")
                                if s2.get(k) is not None})
                rows.append(row)
    keys = ("psnr", "ssim", "lpips", "psnr_cc", "iou", "coverage", "psnr_both", "ssim_both", "lpips_both")
    summ = {}
    for meth in methods:
        summ[meth] = {}
        for g, sel in [("all", lambda r: True)] + [(ro, lambda r, ro=ro: r["role"] == ro) for ro in ROLES]:
            rr = [r for r in rows if r["method"] == meth and sel(r)]
            summ[meth][g] = {"n": len(rr), **{k: (round(float(np.median([r[k] for r in rr if r.get(k) is not None])), 4)
                                                  if any(r.get(k) is not None for r in rr) else None) for k in keys}}
    out = {"metrics": "cand_lhmpp.compare_score's: dancer_splats.evaluate._crop_scores on the dancer's real-mask bbox "
                      "+ 10 %, inside the real mask (render over black: pixels it misses count as black); *_cc after a "
                      "least-squares colour fit; *_both = real mask AND rendered dancer; iou / coverage of the rendered "
                      "role (> 0.5) vs the real mask. 8 held-out phone frames (no input crop within 1 s). The phone's "
                      "k1 (-0.012) is ignored (pinhole renders).", "lpips": net is not None, "device": device,
           "methods": methods, "rows": rows, "summary": summ}
    POD_OUT.mkdir(parents=True, exist_ok=True)
    name = "metrics.json" if net is not None else "metrics_local.json"
    (POD_OUT / name).write_text(json.dumps(out, indent=1, default=float))
    log(json.dumps(summ, default=float)[:3000])
    return out


# ================================================================================================ current avatar (local)
def render_avatar(device: str = "cpu", ss: int = 2, groups: tuple[str, ...] | None = None) -> dict:
    """The current avatar (appearance_avatar_ags.AvatarRenderer: the Unity export 08_JohnLula's shaped SMPL-X skin +
    its 'face' albedo, plain LBS, unlit, both dancers, mutual occlusion) at every camera of cameras.json, posed by
    smplx/<role>.npz at the camera's pose time. RGBA straight alpha; role/<id>.png (R lead, G follow, B alpha) for the
    held-out views; the held-out renders are also copied to INPUT/avatar/ (scored on the pod with LPIPS)."""
    from .appearance_avatar_ags import AvatarRenderer, write_rgba
    cams = json.loads((INPUT / "cameras.json").read_text())["renders"]
    keys = {fr["key"]: fr["t"] for fr in json.loads((INPUT / "poses.json").read_text())["lead"]["frames"]}
    ar = AvatarRenderer(TAKE, device=device, ss=ss, capture="08_JohnLula")
    vd = RENDERS / "avatar"
    t0 = time.time()
    n = 0
    for c in cams:
        if groups and c["group"] not in groups:
            continue
        t = keys[c["pose"]]
        job_ = {"cam": {"K": np.array(c["K"]), "R": np.array(c["R"]), "tvec": np.array(c["tvec"]), "W": c["W"],
                        "H": c["H"]}, "t": t}
        rgb, alpha, roles, _ = ar.render(job_)
        write_rgba(vd / c["out"], rgb, alpha)
        if c.get("role_map"):
            rm = np.stack([(roles == 1) * 1.0, (roles == 2) * 1.0, alpha], -1)
            (vd / "role").mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(vd / "role" / f"{c['id']}.png"), (rm * 255 + 0.5).astype(np.uint8)[:, :, ::-1])
            (INPUT / "avatar" / "heldout").mkdir(parents=True, exist_ok=True)
            (INPUT / "avatar" / "role").mkdir(parents=True, exist_ok=True)
            shutil.copy2(vd / c["out"], INPUT / "avatar" / c["out"])
            shutil.copy2(vd / "role" / f"{c['id']}.png", INPUT / "avatar" / "role" / f"{c['id']}.png")
        n += 1
        if n % 25 == 0:
            log(f"avatar: {n} renders ({time.time() - t0:.0f} s)")
    desc = ar.describe()
    desc["pose"] = ("john_lula MAMMA mono fit smplx/<role>.npz at each camera's pose time (vertices lerped between the "
                    "30 Hz grid frames); the same fit poses the LHM++ variants")
    desc["caption"] = AVATAR_CAPTION
    desc["unity_adds"] = ("hair cards (avatar/hair_groom.json: Lula's ponytail, John's short hair), procedural shoes, eye "
                          "blinks; the export's albedo is the 'face' variant (portrait face on the lead)")
    info = {"renders": n, "seconds": round(time.time() - t0, 1), "device": device, "ss": ss, "describe": desc}
    (vd).mkdir(parents=True, exist_ok=True)
    (vd / "_render.json").write_text(json.dumps(info, indent=1, default=str))
    log(f"avatar: {n} renders in {info['seconds']} s")
    return info


# ================================================================================================ import + manifests
def _session_meta(sid: str | None) -> dict:
    root = PACKAGE_ROOT / "work" / "runpod" / "sessions"
    if sid is None:
        cands = sorted((p for p in root.glob("*") if (p / "result.json").exists()
                        and f'"{JOB_NAME}"' in (p / "result.json").read_text(errors="ignore")),
                       key=lambda p: p.stat().st_mtime)
        if not cands:
            return {}
        sid = cands[-1].name
    p = root / sid / "result.json"
    if not p.exists():
        return {"session": sid}
    r = json.loads(p.read_text())
    pod = r.get("pod") or {}
    j = (r.get("jobs") or {}).get(JOB_NAME) or {}
    steps = {x["step"]: x["seconds"] for x in r.get("timings", []) if "step" in x}
    return {"session": sid, "pod_id": pod.get("id"), "gpu": pod.get("gpu"), "cost_per_hr": pod.get("cost_per_hr"),
            "cost_usd": pod.get("ledger_cost"), "pod_minutes": round((pod.get("lifetime_s") or 0) / 60, 1),
            "job_status": j.get("status"), "commands": j.get("commands"), "job_seconds": j.get("seconds"),
            "download_mb": j.get("download_mb"), "steps_s": steps}


def _check_files(vdir: Path, cams: list[dict]) -> tuple[dict, list, list]:
    groups, missing, bad = {}, [], []
    for c in cams:
        p = vdir / c["out"]
        if not p.exists():
            missing.append(c["out"])
            continue
        im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if im is None or im.shape[:2] != (c["H"], c["W"]) or im.ndim != 3 or im.shape[2] != 4:
            bad.append(c["out"])
            continue
        groups[c["group"]] = groups.get(c["group"], 0) + 1
    return groups, missing, bad


def _files_block(cams: list[dict]) -> dict:
    out: dict = {}
    for c in cams:
        rec = {"id": c["id"], "file": c["out"], "t": c.get("t")}
        for k in ("frame", "clip_index", "grid_k", "azimuth_deg", "azimuth_from_phone_deg", "view_id", "box"):
            if k in c:
                rec[k] = c[k]
        if c.get("role_map"):
            rec["role"] = f"role/{c['id']}.png"
        out.setdefault(c["group"], []).append(rec)
    return out


def import_run(sid: str | None = None) -> dict:
    cams = json.loads((INPUT / "cameras.json").read_text())["renders"]
    refs = json.loads((INPUT / "refs.json").read_text())
    prep_ = json.loads((INPUT / "prep.json").read_text())
    podj = POD_OUT / "lhmpp_compare_pod.json"
    pod = json.loads(podj.read_text()) if podj.exists() else {}
    ses = _session_meta(sid)
    metrics = None
    for name in ("metrics.json", "metrics_local.json"):
        if (POD_OUT / name).exists():
            metrics = json.loads((POD_OUT / name).read_text())
            break
    report = {"take": TAKE, "model": CL.MODEL_NAME, "commit": CL.LHMPP_COMMIT, "licence": CL.LICENCE, "session": ses,
              "pod": pod.get("summary"), "transfer": pod.get("transfer"), "pod_errors": pod.get("errors"),
              "methods": {}}
    common = {"take": TAKE, "views": _ws(INPUT / "cameras.json"), "clip": prep_["clip"], "heldout": prep_["heldout"],
              "novel": prep_["novel"],
              "render": {"format": "RGBA PNG, straight alpha, transparent background", "camera_model":
                         "pinhole (the phone's k1 -0.012 ignored)", "sizes": "heldout + clip_phone at the phone's "
                         "native 538 x 960; heldout_zoom = the couple box at 1080 px on the long side; turntable and "
                         "clip_vr_seat 1080 x 1080"}}
    for meth in METHODS:
        vdir = RENDERS / meth
        if not vdir.exists():
            report["methods"][meth] = {"missing": "not rendered"}
            continue
        groups, missing, bad = _check_files(vdir, cams)
        man = {"schema": "dancecap.mono4d.jl_lhmpp_preview.renders/1", "method": meth, **common,
               "counts": groups, "missing": missing, "bad_size_or_not_rgba": bad, "files": _files_block(cams),
               "metrics_summary": None if metrics is None else metrics["summary"].get(meth),
               "metrics_lpips": None if metrics is None else metrics.get("lpips")}
        if meth == "avatar":
            info = json.loads((vdir / "_render.json").read_text()) if (vdir / "_render.json").exists() else {}
            man.update({"renderer": "avatar_texture.render (appearance_avatar_ags.AvatarRenderer, capture 08_JohnLula)",
                        "what": AVATAR_CAPTION, "describe": info.get("describe"), "local_render": {
                            k: info.get(k) for k in ("renders", "seconds", "device", "ss")}})
        else:
            pv = pod.get("variants", {}).get(meth, {})
            man.update({"renderer": "lhmpp", "what": VARIANTS[meth],
                        "model": f"LHM++ {CL.MODEL_NAME} @ {CL.LHMPP_COMMIT[:7]} (weights CC BY-NC 4.0)",
                        "pose": "john_lula MAMMA mono fit smplx/{lead,follow}.npz, per-joint slerp at each camera's "
                                "time; LHM++ shape mode ref (betas[:10]), trans = pelvis_ref - J0_rest(betas[:10]); both "
                                "dancers in one gsplat rasterisation (mutual occlusion)",
                        "render_settings": pod.get("render_settings"),
                        "inputs": {r: {"files": refs[r]["files"][meth], "video_frames": refs[r]["frames"]}
                                   for r in ROLES} | {"rules": refs["rules"]},
                        "avatar": {r: {k: pv.get(r, {}).get(k) for k in (
                            "gaussians", "n_refs", "infer_s", "animate_s", "rest_height_m", "fk_vs_ref_joints_mm",
                            "gauss_to_own_mesh_mm", "gauss_to_ref_mesh_mm", "opacity_median", "scale_median_mm",
                            "peak_gpu_gb", "error")} | {"cano_ply": _ws(POD_OUT / "gs" / f"{r}_{meth}_cano.ply")}
                            for r in ROLES},
                        "scene_ply": _ws(POD_OUT / "gs" / pv["scene_ply"]) if pv.get("scene_ply") else None,
                        "render_s": pv.get("render_s"), "render_failed": pv.get("render_failed"),
                        "pod": {k: ses.get(k) for k in ("session", "gpu", "pod_minutes", "cost_usd", "cost_per_hr")},
                        "caveats": ["one phone: every held-out view is a frame of the input phone (no input within "
                                    "1 s of it); the novel views (turntable, VR seat) are unconstrained by any photo",
                                    "naked-body pose: hair and loose clothes follow LHM++'s SMPL-X skinning field (no "
                                    "dynamics); the fit's hands are rigid (MAMMA)"]})
        (vdir / "manifest.json").write_text(json.dumps(man, indent=1, default=float))
        report["methods"][meth] = {"counts": groups, "missing": len(missing), "bad": len(bad),
                                   "manifest": _ws(vdir / "manifest.json"),
                                   "gaussians": None if meth == "avatar" else
                                   {r: pod.get("variants", {}).get(meth, {}).get(r, {}).get("gaussians") for r in ROLES},
                                   "metrics": man["metrics_summary"]}
        log(f"{meth}: {groups}; missing {len(missing)}, bad {len(bad)}")
    (OUT / "report.json").write_text(json.dumps(report, indent=1, default=float))
    log(f"report -> {OUT / 'report.json'}")
    return report


# ================================================================================================ side-by-sides
LABEL = {"real": "real phone frame (held out)", "avatar": "current avatar (offline; no hair cards/ponytail)",
         "lhmpp_video8": "LHM++ 8 video crops", "lhmpp_video8_photo": "LHM++ 8 video crops + photo (John: head)",
         "lhmpp_video8_photofull": "LHM++ diagnostic: + John's full occluded photo"}
SHORT = {"real": "real (held out)", "avatar": "current avatar*", "lhmpp_video8": "LHM++ video8",
         "lhmpp_video8_photo": "LHM++ video8+photo",
         "lhmpp_video8_photofull": "diag: +photo full mask"}
FOOT = ("* current avatar = offline render of the shipped texture on the SMPL-X body; the Unity build adds hair cards "
        "(Lula's ponytail) and shoes. All three posed by the same john_lula MAMMA fit.")


def _on_grey(p: Path, grey: int = 128) -> np.ndarray | None:
    im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if im is None:
        return None
    a = im[:, :, 3:4].astype(np.float32) / 255.0
    return (im[:, :, :3] * a + grey * (1 - a) + 0.5).astype(np.uint8)


def _put(img, text, org, s=0.55, th=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, s, (0, 0, 0), th + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, s, (255, 255, 255), th, cv2.LINE_AA)


def _fit(img, h=None, w=None):
    s = (h / img.shape[0]) if h else (w / img.shape[1])
    return cv2.resize(img, (max(1, int(round(img.shape[1] * s))), max(1, int(round(img.shape[0] * s)))),
                      interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)


def _hcat(tiles, gap=4):
    h = max(t.shape[0] for t in tiles)
    tiles = [np.pad(t, ((0, h - t.shape[0]), (0, gap), (0, 0)), constant_values=255) for t in tiles]
    return np.concatenate(tiles, 1)


def _vcat(rows, gap=4):
    W = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, gap), (0, W - r.shape[1]), (0, 0)), constant_values=255) for r in rows]
    return np.concatenate(rows, 0)


def _banner(text, W, h=30, s=0.6):
    b = np.full((h, W, 3), 255, np.uint8)
    cv2.putText(b, text, (8, h - 9), cv2.FONT_HERSHEY_SIMPLEX, s, (0, 0, 0), 1, cv2.LINE_AA)
    return b


def _metric_txt(metrics, view, meth) -> str:
    if not metrics:
        return ""
    rr = [r for r in metrics["rows"] if r["view"] == view and r["method"] == meth]
    parts = []
    for r in rr:
        s = f"{NAME[r['role']]} {r['psnr']:.1f} dB"
        if r.get("lpips") is not None:
            s += f", LPIPS {r['lpips']:.2f}"
        parts.append(s)
    return " | ".join(parts)


def _write_mp4(frames_dir: Path, out: Path, fps: float) -> bool:
    import subprocess
    r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", f"{fps:.4f}", "-i", str(frames_dir / "%04d.png"),
                        "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p", str(out)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        log(f"ffmpeg failed: {r.stderr[-500:]}")
    return r.returncode == 0


def clip_box(ph: list[dict]) -> list[int]:
    """Union of both dancers' masks over the phone clip + 8 %, widened / heightened to a 0.6 aspect (w / h)."""
    M = Masks()
    u = None
    for c in ph:
        for r in ROLES:
            m = M.get(c["frame"], r)
            if m is None or not m.any():
                continue
            ys, xs = np.nonzero(m)
            b = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
            u = b if u is None else [min(u[0], b[0]), min(u[1], b[1]), max(u[2], b[2]), max(u[3], b[3])]
    W, H = c["W"], c["H"]
    x0, y0, x1, y1 = u
    w, h = (x1 - x0) * 1.16, (y1 - y0) * 1.16
    if w / h < 0.6:
        w = 0.6 * h
    else:
        h = w / 0.6
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w, h = min(w, W), min(h, H)
    bx0 = int(round(min(max(cx - w / 2, 0), W - w)))
    by0 = int(round(min(max(cy - h / 2, 0), H - h)))
    return [bx0, by0, bx0 + int(w) // 2 * 2, by0 + int(h) // 2 * 2]


def faces_sheet() -> Path | None:
    """Heads of the canonical LHM++ avatars (pod_out/turntable: front / left / back), every variant, 1.5x."""
    rows = []
    for role in ROLES:
        tiles = []
        for v in VARIANTS:
            p = POD_OUT / "turntable" / f"{role}_{v}.png"
            im = cv2.imread(str(p))
            if im is None:
                continue
            for k, nm in enumerate(("front", "left", "back")):
                c = cv2.resize(im[0:260, k * 504 + 120:k * 504 + 384], None, fx=1.5, fy=1.5,
                               interpolation=cv2.INTER_CUBIC)
                c = np.ascontiguousarray(c)
                cv2.putText(c, f"{v.replace('lhmpp_', '')} {nm}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0, 0, 200), 1, cv2.LINE_AA)
                tiles.append(c)
        if tiles:
            row = _hcat(tiles)
            rows += [_banner(f"{NAME[role]}: LHM++ canonical avatar heads (A-pose render on white, 1.5x)", row.shape[1]),
                     row]
    if not rows:
        return None
    p = OUT / "sheets" / "faces_canonical.jpg"
    cv2.imwrite(str(p), _vcat(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return p


def sheets() -> list[Path]:
    cams = json.loads((INPUT / "cameras.json").read_text())["renders"]
    prep_ = json.loads((INPUT / "prep.json").read_text())
    metrics = None
    for name in ("metrics.json", "metrics_local.json"):
        if (POD_OUT / name).exists():
            metrics = json.loads((POD_OUT / name).read_text())
            break
    meths = [m for m in METHODS if (RENDERS / m).exists()]
    sd = OUT / "sheets"
    sd.mkdir(parents=True, exist_ok=True)
    out = []
    byid = {c["id"]: c for c in cams}
    # ---- 1. held-out phone frames at native pixels (crop = both masks + 10 %), and the 1080 px zoom renders
    for kind in ("native", "zoom"):
        rows = []
        for h in prep_["heldout"]:
            vid = h["id"]
            x0, y0, x1, y1 = h["box"]
            real = cv2.imread(str(frame_path(h["i"])))[y0:y1, x0:x1]
            H = 420 if kind == "native" else 540
            tiles = [_fit(real, h=H)]
            _put(tiles[0], SHORT["real"], (6, 20))
            _put(tiles[0], f"f{h['i']} t={h['t']:.1f}s", (6, 42), 0.5)
            for m in meths:
                if kind == "native":
                    im = _on_grey(RENDERS / m / "heldout" / f"{vid}.png")
                    im = None if im is None else im[y0:y1, x0:x1]
                else:
                    im = _on_grey(RENDERS / m / byid[f"{vid}_zoom"]["out"])
                if im is None:
                    continue
                t = _fit(im, h=H)
                _put(t, SHORT[m], (6, 20))
                mt = _metric_txt(metrics, vid, m)
                for n_, line in enumerate(reversed(mt.split(" | ")) if mt else []):
                    _put(t, line, (6, H - 10 - 22 * n_), 0.5)
                tiles.append(t)
            rows.append(_hcat(tiles))
        per = 4
        for k in range(0, len(rows), per):
            body = _vcat(rows[k:k + per])
            title = (f"john_lula held-out phone frames ({'native 538x960 pixels, upscaled for display' if kind == 'native' else 'couple box re-rendered at 1080 px; real = the phone crop upscaled'}); "
                     "no LHM++ input frame within 1 s")
            sheet = _vcat([_banner(title, body.shape[1]), body, _banner(FOOT, body.shape[1], 26, 0.5)])
            p = sd / f"heldout_{kind}_{k // per}.jpg"
            cv2.imwrite(str(p), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
            out.append(p)
    # ---- 2. turntable (12 views, 30 deg apart, tt00 = the phone's direction)
    tts = [c for c in cams if c["group"] == "turntable"]
    rows = []
    for m in meths:
        ims = []
        for c in tts:
            im = _on_grey(RENDERS / m / c["out"])
            if im is None:
                continue
            im = _fit(im[60:1020, 140:940], h=320)
            _put(im, f"{c['azimuth_from_phone_deg']:.0f} deg", (6, 20), 0.5)
            ims.append(im)
        if ims:
            rows.append(_banner(LABEL[m], _hcat(ims[:6]).shape[1]))
            rows.append(_hcat(ims[:6]))
            rows.append(_hcat(ims[6:]))
    if rows:
        body = _vcat(rows)
        tt = prep_["novel"]["turntable"]
        sheet = _vcat([_banner(f"john_lula turntable at t={tt['t']:.2f}s (frame {tt['frame']}): radius 3 m, height 1.6 m; "
                               "0 deg = the phone's side", body.shape[1]), body, _banner(FOOT, body.shape[1], 26, 0.5)])
        p = sd / "turntable.jpg"
        cv2.imwrite(str(p), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
        out.append(p)
    # ---- 3. clips: key frames + side-by-side videos
    clip = prep_["clip"]
    ph = [c for c in cams if c["group"] == "clip_phone"]
    vr = [c for c in cams if c["group"] == "clip_vr_seat"]
    js = [int(round(x)) for x in np.linspace(0, len(ph) - 1, 6)]
    cb = clip_box(ph)
    crop = lambda im: im[cb[1]:cb[3], cb[0]:cb[2]]  # noqa: E731
    rows = []
    real_t = [_fit(crop(cv2.imread(str(frame_path(ph[j]["frame"])))), h=420) for j in js]
    for t, j in zip(real_t, js):
        _put(t, f"t={ph[j]['t']:.2f}", (6, 20), 0.5)
    row = _hcat(real_t)
    rows += [_banner("real phone frames, cropped to the couple (held out: no input crop within 0.5 s of the clip)",
                     row.shape[1]), row]
    for m in meths:
        ims = [_on_grey(RENDERS / m / ph[j]["out"]) for j in js]
        if all(x is not None for x in ims):
            row = _hcat([_fit(crop(x), h=420) for x in ims])
            rows += [_banner(f"{LABEL[m]} - along the phone's own camera path (same crop)", row.shape[1]), row]
    p = sd / "clip_phone_keyframes.jpg"
    body = _vcat(rows)
    cv2.imwrite(str(p), _vcat([body, _banner(FOOT, body.shape[1], 26, 0.5)]), [cv2.IMWRITE_JPEG_QUALITY, 88])
    out.append(p)
    rows = []
    for m in meths:
        ims = [_on_grey(RENDERS / m / vr[j]["out"]) for j in js]
        if all(x is not None for x in ims):
            row = _hcat([_fit(x[150:930, 150:930], h=330) for x in ims])
            rows += [_banner(f"{LABEL[m]} - fixed VR seat 2.5 m away, 1.6 m high, opposite the phone (centre "
                             "780 px of 1080)", row.shape[1]), row]
    if rows:
        p = sd / "clip_vr_seat_keyframes.jpg"
        body = _vcat(rows)
        cv2.imwrite(str(p), _vcat([body, _banner(FOOT, body.shape[1], 26, 0.5)]), [cv2.IMWRITE_JPEG_QUALITY, 88])
        out.append(p)
    tmp = OUT / "scratch" / "sbs"
    for name, group, panels, fps in (("clip_phone_sbs.mp4", ph, ["real"] + meths, 14997 / 500),
                                     ("clip_vr_seat_sbs.mp4", vr, meths, 30.0)):
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        pw = 1920 // len(panels)
        pw -= pw % 2
        okall = True
        for j, c in enumerate(group):
            tiles = []
            for m in panels:
                im = cv2.imread(str(frame_path(c["frame"]))) if m == "real" else _on_grey(RENDERS / m / c["out"])
                if im is None:
                    okall = False
                    break
                if group is ph:
                    im = im[cb[1]:cb[3], cb[0]:cb[2]]
                else:
                    im = im[150:930, 150:930]   # the VR seat's centre 780 px (the couple)
                t = cv2.resize(im, (pw, int(round(im.shape[0] * pw / im.shape[1])) // 2 * 2), interpolation=cv2.INTER_AREA)
                _put(t, SHORT[m], (8, 24), 0.6)
                tiles.append(t)
            if not okall:
                break
            fr = np.concatenate(tiles, 1)
            _put(fr, f"t={c['t']:.2f}s", (8, fr.shape[0] - 12), 0.55)
            cv2.imwrite(str(tmp / f"{j:04d}.png"), fr)
        if okall and _write_mp4(tmp, sd / name, fps):
            out.append(sd / name)
        shutil.rmtree(tmp, ignore_errors=True)
    # ---- 4. inputs + canonical LHM++ avatars (front / left / back / right, A-pose)
    rows = []
    if (OUT / "inputs.jpg").exists():
        rows.append(cv2.imread(str(OUT / "inputs.jpg")))
    for role in ROLES:
        for v in VARIANTS:
            p = POD_OUT / "turntable" / f"{role}_{v}.png"
            if p.exists():
                im = _fit(cv2.imread(str(p)), h=420)
                rows += [_banner(f"{NAME[role]}: LHM++ {v.replace('lhmpp_', '')} canonical avatar (front / left / back / "
                                 "right)", im.shape[1]), im]
    if rows:
        p = sd / "inputs_and_avatars.jpg"
        cv2.imwrite(str(p), _vcat(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
        out.append(p)
    fp = faces_sheet()
    if fp is not None:
        out.append(fp)
    (sd / "index.json").write_text(json.dumps({"sheets": [p.name for p in out], "clip": clip,
                                               "caption": FOOT}, indent=1))
    log(f"sheets: {[p.name for p in out]}")
    return out


# ================================================================================================ cli
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.mono4d.jl_lhmpp_preview")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("frames")
    p.add_argument("--force", action="store_true")
    sub.add_parser("mask-check")
    p = sub.add_parser("photo-seg")
    p.add_argument("--device", default="cpu")
    p = sub.add_parser("prep")
    p.add_argument("--reuse-survey", action="store_true")
    p = sub.add_parser("job")
    p.add_argument("--max-minutes", type=float, default=25)
    p = sub.add_parser("score")
    p.add_argument("--no-lpips", action="store_true")
    p = sub.add_parser("avatar")
    p.add_argument("--device", default="cpu")
    p.add_argument("--ss", type=int, default=2)
    p.add_argument("--groups", default="")
    p = sub.add_parser("import")
    p.add_argument("--session", default=None)
    sub.add_parser("sheets")
    a = ap.parse_args(argv)
    if a.cmd == "frames":
        frames(a.force)
    elif a.cmd == "mask-check":
        mask_check()
    elif a.cmd == "photo-seg":
        photo_seg(a.device)
    elif a.cmd == "prep":
        prep(a.reuse_survey)
    elif a.cmd == "job":
        print(job(a.max_minutes))
    elif a.cmd == "score":
        score(lpips_net=not a.no_lpips)
    elif a.cmd == "avatar":
        render_avatar(a.device, a.ss, tuple(x for x in a.groups.split(",") if x) or None)
    elif a.cmd == "import":
        print(json.dumps(import_run(a.session)["methods"], indent=1, default=float)[:4000])
    elif a.cmd == "sheets":
        sheets()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
