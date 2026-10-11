"""The `john_lula` avatars (agent J5a): the pose-independent avatar stages of the take `john_lula` (ONE handheld camera with three zoom steps, a floor with
several couples) - the stage graph of ca_avatars.py (two tripod cameras) generalised ADDITIVELY: the pure helpers are imported from ca_avatars, the
constants (take, work dir, one camera, the 2D people tables of agent J3a) are this take's. One command per stage; every stage re-runnable.

    python -m dancecap.jl_avatars plan [--json]               the stage table with its status (what exists, what waits)
    python -m dancecap.jl_avatars <stage> [options]           run one stage

POSE-INDEPENDENT stages (the 2D people tables work/john_lula/people/teachers_01.npz + masks_01.npz; they run before any 3D pose exists):
    face-frames      sharp, large, frontal faces of each dancer (whole-body keypoints, role confidence, no zoom step, the partner not in the box and
                     the face inside the dancer's own mask) -> avatar/face_id/frames/*.jpg, candidates.json
    face-landmarks   MediaPipe 478 + RTMW iBUG-68 (.venv-face) on the lead's portrait photo (the follow has none: video faces only) and the frames
                     -> avatar/face_id/landmarks_{lead,follow}.json, review/avatar/face_frames_*.jpg (LOOK at them: --exclude / --relabel)
    face-identity    POD (kind `avatarjl`, one CPU pod, nothing else claims it): SMIRK + landmark refinement + robust combination per dancer
                     -> face_shape/<role>_identity{,_portrait,_video}.npz, avatar/face_id/fit_report.json (jl_avatars_pod)
    shoes-video      shoe colours measured in the video at the 6 foot keypoints -> avatar/shoes_video.json (+ a suggested shoes_override.json)
    texture-preview  PROVISIONAL texture windows from the 2D keypoints / masks of the ONE moving camera: which seconds show which body side of
                     each dancer -> avatar/texture/windows_preview.json
    head-review      renders of the rest heads (template / identity) on provisional betas beside the portrait (the heads are judged, not the body)

POSE-DEPENDENT stages (wait for the fitted smplx/{lead,follow}.npz of agent J3b; `plan` lists them): texture-select / -run / -fuse, face-texture,
hair-groom, shoes, shoe-size, export - they reuse recap_avatars_fuse / avatar_face / hair_groom / shoes unchanged (take-agnostic given the work dir).

Likeness data (the portrait, face frames, identities, colours sampled from the dancers) stays under work/ (git-ignored) and on the owner's private
Runpod volume; no tracked file carries a portrait, a face crop or a sampled colour.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import tomllib
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import ca_avatars as CA
from .take import PACKAGE_ROOT

TAKE = "john_lula"
OWNER = "J5a"
KIND = "avatarjl"
ROLES = CA.ROLES
WHO = {"lead": 0, "follow": 1}                    # index in the people tables (0 = lead, 1 = follow)
CAM = "01"
WORK = PACKAGE_ROOT / "work" / TAKE
AV = WORK / "avatar"
FACE_ID = AV / "face_id"
FACE_SHAPE = WORK / "face_shape"
PEOPLE = WORK / "people"
REVIEW = WORK / "review" / "avatar"
REL = f"dancecap/work/{TAKE}"
FPS = 14997 / 500
# the lead's portrait photo (the take folder holds it next to the video); the follow has none (seen from behind). The TOML key that makes
# CA.portrait_path-style readers find it: [dancers.lead] portrait = "<this file>"
DEFAULT_PORTRAITS = {"lead": "GM6B6189.jpg", "follow": None}
ZOOM_FALLBACK = ((17.84, 18.57), (51.51, 52.24), (78.75, 79.38))   # measured by the camera stage (the TOML's zoom_events)
COCO = {"nose": 0, "leye": 1, "reye": 2, "lear": 3, "rear": 4, "ls": 5, "rs": 6, "lh": 11, "rh": 12}
WB_FEET = {"L": (17, 18, 19), "R": (20, 21, 22)}   # COCO-WholeBody: big toe, small toe, heel
WB_FACE = slice(23, 91)


def log(msg: str) -> None:
    print(f"[jl_avatars {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------------------------------------ the stage table

STAGES: tuple[CA.Stage, ...] = (
    CA.Stage("face-frames", False, "local", "face-frames [--per-role 40]", ("people/teachers_01.npz", "people/masks_01.npz", "source video"),
             ("avatar/face_id/frames/*.jpg", "avatar/face_id/candidates.json"), "0 (cv2, ~1 min)",
             "sharp, large, frontal faces of each dancer (role confidence, no zoom step, partner not in the box, face inside the own mask)"),
    CA.Stage("face-landmarks", False, "local", "face-landmarks [--exclude NAMES] [--relabel name=role] [--reuse]", ("avatar/face_id/candidates.json", "lead portrait"),
             ("avatar/face_id/landmarks_{lead,follow}.json", "review/avatar/face_frames_{lead,follow}.jpg"), "0 (.venv-face, ~2 min)",
             "MediaPipe + RTMW landmarks; the best frames per dancer (LOOK at the contact sheets); the follow has video faces only"),
    CA.Stage("face-identity", False, "pod", "face-identity [--upload-models] [--max-cost 1.0]",
             ("avatar/face_id/landmarks_*.json + frames", "lead portrait", "volume: models/flame, models/smirk"),
             ("face_shape/{lead,follow}_identity.npz", "avatar/face_id/fit_report.json"), "~0.02-0.05 (CPU pod 8 vCPU, ~8 min)",
             "FLAME identity per dancer (lead: portrait + video, follow: video only) + split-half reliability for the trust estimate"),
    CA.Stage("face-trust", False, "local", "face-trust", ("face_shape/*_identity*.npz", "avatar/face_id/fit_report.json"), ("avatar/face_id/trust.json",), "0 (seconds)",
             "how far to trust a video-only identity (compared with the lead's portrait fit, the split halves, other takes' identities) -> identity_weight"),
    CA.Stage("shoes-video", False, "local", "shoes-video", ("people/teachers_01.npz", "people/masks_01.npz", "source video"),
             ("avatar/shoes_video.json", "avatar/shoes_override.json"), "0 (~1 min)", "shoe upper / sole colours at the 6 foot keypoints; a suggested override"),
    CA.Stage("texture-preview", False, "local", "texture-preview [--length 36] [--n-windows 3]", ("people/teachers_01.npz",),
             ("avatar/texture/windows_preview.json",), "0 (seconds)", "provisional texture windows: which seconds show which body side (one moving camera)"),
    CA.Stage("head-review", False, "local", "head-review", ("face_shape/*", "provisional betas", "lead portrait"), ("review/avatar/head_{lead,follow}.jpg",),
             "0 (~3 min)", "front / 3-4 / profile renders of the rest heads (template vs identity) on the recap's provisional betas"),
    CA.Stage("texture-select", True, "local", "(J3b's smplx first) ca_avatars.texture_select is two-camera: a one-camera variant is to be written", ("smplx/{lead,follow}.npz",),
             ("avatar/texture/windows.json",), "0", "windows of the fitted poses (replaces the preview)"),
    CA.Stage("texture-run", True, "pod", "(like ca_avatars_pod.texture_run, one camera)", ("windows.json", "smplx", "volume: source video"),
             ("work/john_lula_<window>/avatar/*",), "~0.3-0.5 per window (4090 pod)", "front (frames, SAM masks) then the hair-aware bake, one pod job per window"),
    CA.Stage("texture-fuse", True, "local", "texture-fuse --windows tx1,tx2,tx3", ("work/john_lula_<window>/avatar/*_texels.npz",),
             ("avatar/{lead,follow}_albedo.png",), "0 (~1 min)", "per-texel confidence-weighted fusion of the window bakes (only the visible side of a window is textured)"),
    CA.Stage("face-texture", True, "local", "face-texture [--composite]", ("avatar/{lead,follow}_albedo.png", "smplx", "lead portrait"),
             ("avatar/{lead,follow}_albedo_face.png",), "0 (~9 min CPU)", "the lead's portrait composited onto the bake; the follow's face from the video"),
    CA.Stage("hair-groom", True, "local", "hair-groom --from-take <donor take>", ("smplx/follow.npz", "[avatar/follow_albedo_face.png]", "avatar/hair_palette.json", "[dancers.follow] hair_style"),
             ("avatar/hair_groom.json",), "0 (~1 min)", "the ponytail groom: a donor take's roots re-based on her rest model, hair_ponytail restyle, her palette"),
    CA.Stage("shoes", True, "local", "shoes", ("capture", "avatar/shoes_override.json", "avatar/shoes_video.json"), ("avatar/shoes.json",), "0 (<1 min)",
             "sample + apply the colours / style override"),
    CA.Stage("export", True, "local", "python -m dancecap.demo_export john_lula ... (export agent)", ("everything above", "[dancers.*] of the take TOML"),
             ("StreamingAssets/<capture>/",), "0", "the Unity capture"),
)


# ------------------------------------------------------------------------------------------------ take config / files

def take_cfg() -> dict:
    return tomllib.loads((PACKAGE_ROOT / "takes" / f"{TAKE}.toml").read_text(encoding="utf-8"))


def source_dir() -> Path:
    return ((PACKAGE_ROOT / "takes" / f"{TAKE}.toml").parent / take_cfg()["source_dir"]).resolve()


def source_video() -> Path:
    return source_dir() / take_cfg()["cameras"][CAM]["file"]


def portrait_path(role: str) -> Path | None:
    """The role's portrait photo: the TOML's [dancers.<role>] portrait when the owner set it, else this module's DEFAULT_PORTRAITS; None = no portrait
    (the follow, seen from behind in the only photo)."""
    name = (take_cfg()["dancers"][role].get("portrait")) or DEFAULT_PORTRAITS.get(role)
    if not name:
        return None
    p = source_dir() / name
    if not p.exists():
        raise FileNotFoundError(f"portrait of the {role}: {p} missing")
    return p


def zoom_events() -> list[tuple[float, float]]:
    try:
        return [(float(a), float(b)) for a, b, *_ in take_cfg()["cameras"][CAM]["zoom_events"]]
    except (KeyError, ValueError, TypeError):
        return list(ZOOM_FALLBACK)


def load_people() -> dict:
    z = np.load(PEOPLE / f"teachers_{CAM}.npz", allow_pickle=True)
    return {k: z[k] for k in z.files}


def _decode_mask(masks, frame: int, who: int) -> np.ndarray | None:
    import cv2
    key = f"m_{frame}_{who}"
    if key not in masks.files:
        return None
    return cv2.imdecode(np.frombuffer(masks[key].tobytes(), np.uint8), cv2.IMREAD_UNCHANGED) > 0


# ------------------------------------------------------------------------------------------------ pure parts: gates and facing

def in_zoom_step(t: np.ndarray, events: list[tuple[float, float]], margin: float = 0.4) -> np.ndarray:
    """True where t lies within `margin` seconds of a zoom step (motion blur, the focal changes within the frame)."""
    out = np.zeros(len(t), bool)
    for a, b in events:
        out |= (t >= a - margin) & (t <= b + margin)
    return out


def shoulder_ratio(kp: np.ndarray, kp_min: float = 0.3) -> dict[str, np.ndarray]:
    """Per frame of one dancer, kp (N,17,3): rho = (left - right shoulder x) / torso length (> 0 = the dancer faces the camera: her left is on the image's
    right), torso (px), ok (both shoulders and both hips above kp_min)."""
    ls, rs, lh, rh = (kp[:, COCO[k]] for k in ("ls", "rs", "lh", "rh"))
    ok = np.minimum.reduce([ls[:, 2], rs[:, 2], lh[:, 2], rh[:, 2]]) > kp_min
    torso = np.hypot(*(0.5 * (ls[:, :2] + rs[:, :2]) - 0.5 * (lh[:, :2] + rh[:, :2])).T)
    ok &= torso > 1.0
    rho = (ls[:, 0] - rs[:, 0]) / np.maximum(torso, 1e-6)
    return {"rho": np.where(ok, rho, np.nan), "torso": torso, "ok": ok}


def rho_reference(rho: np.ndarray, q: float = 95.0) -> float:
    """The dancer's |rho| when square on to the camera (front or back): a high percentile of |rho| (the shoulders of a turned dancer are foreshortened)."""
    r = np.abs(rho[np.isfinite(rho)])
    return float(np.percentile(r, q)) if len(r) else 1.0


def facing_flags(kp: np.ndarray, rho_ref: float | None = None, min_front: float = 0.55, kp_min: float = 0.3) -> dict[str, np.ndarray]:
    """front: the dancer faces the camera within about 55 degrees (cos psi = rho / rho_ref >= min_front), back: turned away by the same margin."""
    sr = shoulder_ratio(kp, kp_min)
    ref = rho_ref or rho_reference(sr["rho"])
    c = np.clip(sr["rho"] / ref, -1.0, 1.0)
    return {"cos": c, "front": np.isfinite(c) & (c >= min_front), "back": np.isfinite(c) & (c <= -min_front), "ref": ref, "torso": sr["torso"]}


def facing_angle(kp: np.ndarray, kp_wb: np.ndarray, rho_ref: float, kp_min: float = 0.3) -> np.ndarray:
    """Body yaw psi (rad, wrapped) per frame: 0 = facing the camera, pi = away, +pi/2 = facing the image's right (the camera then sees the dancer's
    RIGHT side). |psi| from the shoulder foreshortening (arccos of rho / rho_ref); the sign from where the nose points (relative to the shoulder
    midpoint) and where the toes point (feet keypoints of the whole-body set): weighted, the feet more once the dancer turns away. NaN where the shoulders
    are not seen. A monocular cue, enough for a PROVISIONAL choice of windows (the fitted poses replace it)."""
    sr = shoulder_ratio(kp, kp_min)
    c = np.clip(sr["rho"] / rho_ref, -1.0, 1.0)
    absang = np.arccos(c)
    half_sh = np.maximum(0.5 * np.abs(sr["rho"]) * sr["torso"], 0.25 * rho_ref * sr["torso"])
    mid_s = 0.5 * (kp[:, COCO["ls"], 0] + kp[:, COCO["rs"], 0])
    nose_ok = kp[:, COCO["nose"], 2] > kp_min
    nose = np.where(nose_ok, np.clip((kp[:, COCO["nose"], 0] - mid_s) / np.maximum(half_sh, 1.0), -1, 1), 0.0)
    dx, wsum = np.zeros(len(kp)), np.zeros(len(kp))
    for side in "LR":
        toe, small, heel = WB_FEET[side]
        ok = (kp_wb[:, toe, 2] > 0.4) & (kp_wb[:, heel, 2] > 0.4)
        d = kp_wb[:, toe, :2] - kp_wb[:, heel, :2]
        n = np.hypot(d[:, 0], d[:, 1])
        dx += np.where(ok & (n > 2.0), d[:, 0] / np.maximum(n, 1e-6), 0.0)
        wsum += (ok & (n > 2.0)).astype(float)
    foot = np.where(wsum > 0, dx / np.maximum(wsum, 1.0), 0.0)
    w_nose = np.where(c >= 0, 0.7, 0.25)
    cue = w_nose * nose + (1 - w_nose) * foot
    sign = np.where(cue >= 0, 1.0, -1.0)
    return np.where(np.isfinite(absang), sign * absang, np.nan)


def side_group(psi: np.ndarray) -> np.ndarray:
    """0 front (|psi| < 67.5 deg), 1 right side of the dancer seen (facing the image's right: psi in [67.5, 112.5)), 2 back, 3 left side (psi in
    [-112.5, -67.5)); the in-between quarter views count with the nearer of front / back. -1 where psi is NaN."""
    g = np.full(len(psi), -1, int)
    ok = np.isfinite(psi)
    a = np.where(ok, np.degrees(psi), 0.0)
    g[ok & (np.abs(a) < 67.5)] = 0
    g[ok & (a >= 67.5) & (a < 112.5)] = 1
    g[ok & (np.abs(a) >= 112.5)] = 2
    g[ok & (a <= -67.5) & (a > -112.5)] = 3
    return g


SIDE_NAMES = ("front", "right_side", "back", "left_side")


def candidate_gate(people: dict, who: int, events: list[tuple[float, float]], kp_face_min: float = 0.5, min_iod: float = 6.0, max_off: float = 0.45,
                   max_speed: float = 0.30, role_min: float = 0.9, box_iou_max: float = 0.35, zoom_margin: float = 0.4) -> dict:
    """The face-candidate gate of one dancer from the 2D people tables: -> {score (F,), iod, speed, off, front, why: dict of counts}. A frame scores
    iod / (1 + 2 speed) when the dancer is confidently detected and identified, faces the camera (shoulders AND the nose between the eyes), is not in a zoom
    step, does not overlap the partner (mask boxes / skeleton) and the whole-body face points are confident."""
    t = people["t"]
    kp, wb = people["kp"][:, who], people["kp_wb"][:, who]
    m = CA.face_metrics(kp, 0.4)
    sp = CA.head_speed(kp, m["iod"])
    ff = facing_flags(kp)
    face_score = wb[:, WB_FACE, 2].mean(1)
    gates = {
        "pose": people["conf"][:, who] >= 0.5,
        "role": people["role_conf"][:, who] >= role_min,
        "alone": ~people["merged"][:, who].astype(bool) & (people["skel_in_partner"][:, who] < 0.3) & (people["box_iou"] < box_iou_max),
        "zoom": ~in_zoom_step(t, events, zoom_margin),
        "front": ff["front"],
        "face_seen": m["seen"] & (m["off"] <= max_off),
        "face_conf": face_score >= kp_face_min,
        "size": m["iod"] >= min_iod,
        "sharp": np.isfinite(sp) & (sp <= max_speed),
    }
    ok = np.ones(len(t), bool)
    for g in gates.values():
        ok &= g
    sc = np.where(ok, m["iod"] / (1.0 + 2.0 * np.nan_to_num(sp)), 0.0)
    return {"score": sc, "iod": m["iod"], "speed": sp, "off": m["off"], "front": ff["front"], "cos": ff["cos"], "face_score": face_score,
            "passing": {k: int(g.sum()) for k, g in gates.items()} | {"all": int(ok.sum())}}


def mask_face_check(own: np.ndarray | None, partner: np.ndarray | None, face: list[float], scale: float = 2.6 / 5.5) -> tuple[float, float]:
    """(fraction of the face square inside the dancer's own mask, fraction inside the partner's) for face = [cx, cy, side] (CA.face_box): the square is
    shrunk to `scale` x side = the face itself (the SMIRK box is a loose crop with hair and shoulders)."""
    if own is None:
        return 0.0, 0.0
    cx, cy, side = face
    h = 0.5 * side * scale
    x0, x1 = int(max(0, np.floor(cx - h))), int(min(own.shape[1], np.ceil(cx + h)))
    y0, y1 = int(max(0, np.floor(cy - h))), int(min(own.shape[0], np.ceil(cy + h)))
    if x1 <= x0 or y1 <= y0:
        return 0.0, 0.0
    a = float(own[y0:y1, x0:x1].mean())
    b = float(partner[y0:y1, x0:x1].mean()) if partner is not None else 0.0
    return a, b


# ------------------------------------------------------------------------------------------------ stage: face-frames

def stage_face_frames(per_role: int = 40, min_gap_s: float = 0.5, own_min: float = 0.0, partner_max: float = 0.2, max_speed: float = 0.5,
                      max_off: float = 0.6) -> dict:
    """Candidates per dancer: the gate, then greedy by score with at least min_gap_s between frames, then the identity check on the picked ones (the pose
    model can hand the head keypoints of one dancer to the other in embraces): the face square must not lie in the PARTNER's mask (partner_max) and the
    partner's own head must be a face-width away; the dancer's own mask is recorded but not required (the SAM masks often miss the head: own_min = 0).
    A human reviews the contact sheets afterwards (face-landmarks --exclude / --relabel)."""
    people = load_people()
    masks = np.load(PEOPLE / f"masks_{CAM}.npz", allow_pickle=True)
    events = zoom_events()
    t, fi = people["t"], people["frame_index"].astype(int)
    cand: dict[str, dict] = {}
    rep: dict = {}
    for role in ROLES:
        who = WHO[role]
        g = candidate_gate(people, who, events, max_speed=max_speed, max_off=max_off)
        sel_all = CA.pick_frames(g["score"], t, per_role * 3, min_gap_s)
        kp = people["kp"][:, who]
        chosen, dropped = [], []
        for i in sel_all:
            face = CA.face_box(kp[i])
            own = _decode_mask(masks, int(fi[i]), who)
            par = _decode_mask(masks, int(fi[i]), 1 - who)
            a, b = mask_face_check(own, par, face)
            pk = people["kp"][i, 1 - who]
            near = bool(pk[0, 2] > 0.3 and np.hypot(*(pk[0, :2] - np.asarray(face[:2]))) < 0.5 * 2.6 / 5.5 * face[2])
            (chosen if (a >= own_min and b <= partner_max and not near) else dropped).append((i, a, b))
            if len(chosen) >= per_role:
                break
        paths = {int(fi[i]): FACE_ID / "frames" / f"{role}_{CAM}_{int(fi[i]):06d}.jpg" for i, _, _ in chosen}
        CA.extract_frames(source_video(), list(paths.items()))
        for i, a, b in chosen:
            f = int(fi[i])
            name = f"{role}_{CAM}_{f:06d}"
            if not paths[f].exists():
                log(f"  {name}: frame not decoded")
                continue
            cand[name] = {"role": role, "cam": CAM, "frame_index": f, "t": float(t[i]), "iod_px": float(g["iod"][i]), "head_speed": float(g["speed"][i]),
                          "face": CA.face_box(kp[i]), "file": f"frames/{name}.jpg", "own_mask_frac": round(a, 3), "partner_mask_frac": round(b, 3),
                          "role_conf": float(people["role_conf"][i, who]), "cos_psi": round(float(g["cos"][i]), 3)}
        iods = np.array([cand[n]["iod_px"] for n in cand if cand[n]["role"] == role])
        rep[role] = {"gate_counts": g["passing"], "picked": len(chosen), "dropped_by_mask_check": len(dropped), "iod_px_median": float(np.median(iods)) if len(iods) else None,
                     "iod_px_range": [float(iods.min()), float(iods.max())] if len(iods) else None,
                     "iod_ge_8": int((iods >= 8).sum()), "iod_ge_12": int((iods >= 12).sum()), "iod_ge_16": int((iods >= 16).sum())}
        log(f"{role}: gate {g['passing']}; {len(chosen)} picked ({len(dropped)} dropped by the mask check), eye distance median "
            f"{rep[role]['iod_px_median']} px")
    FACE_ID.mkdir(parents=True, exist_ok=True)
    (FACE_ID / "candidates.json").write_text(json.dumps(cand, indent=1))
    (FACE_ID / "candidates_report.json").write_text(json.dumps(rep, indent=1))
    return rep


# ------------------------------------------------------------------------------------------------ stage: face-landmarks

def load_exclude(extra: tuple[str, ...] = ()) -> tuple[str, ...]:
    p = FACE_ID / "exclude.json"
    cur = set(json.loads(p.read_text())) if p.exists() else set()
    if extra:
        cur |= set(extra)
        p.write_text(json.dumps(sorted(cur), indent=1))
    return tuple(sorted(cur))


def load_relabel(extra: dict[str, str] | None = None) -> dict[str, str]:
    p = FACE_ID / "relabel.json"
    cur = json.loads(p.read_text()) if p.exists() else {}
    cur.update(extra or {})
    if extra:
        p.write_text(json.dumps(cur, indent=1))
    return cur


def rel_to_work(file: str, role: str) -> str:
    p = Path(file)
    try:
        return p.resolve().relative_to(WORK).as_posix()
    except ValueError:
        return f"avatar/face_id/portrait_{role}{p.suffix}"


def contact_sheet(role: str, recs: list[dict], size: int = 224, ctx: bool = False) -> Path:
    """The SMIRK crops (ctx=False) or a 2.4x wider context window (ctx=True: hands, the partner, other people show) of the portrait and the chosen frames."""
    import cv2

    from .face_smirk import warp_crop
    allrec = json.loads((FACE_ID / "landmarks_all.json").read_text())
    tiles = []
    por = next((x for x in allrec if x["name"] == f"portrait_{role}"), None)
    items = ([(por, portrait_path(role))] if por and por.get("ok") and portrait_path(role) else []) + [(x, WORK / x["file"]) for x in recs]
    for r, f in items:
        img = cv2.imread(str(f))
        if img is None:
            continue
        t = warp_crop(img, np.asarray(r["smirk_center"]), float(r["smirk_side"]) * (2.4 if ctx else 1.0), size)
        cv2.putText(t, r["name"][-18:], (4, 16), 0, 0.45, (255, 255, 255), 1)
        cv2.putText(t, f"occ {r['occ_px']:.0f}px yaw {r['yaw']:.0f}", (4, size - 6), 0, 0.42, (255, 255, 0), 1)
        tiles.append(t)
    if not tiles:
        raise RuntimeError(f"no tiles for the {role}")
    blank = np.zeros_like(tiles[0])
    rows = [np.concatenate(tiles[i:i + 6] + [blank] * (6 - len(tiles[i:i + 6])), 1) for i in range(0, len(tiles), 6)]
    out = REVIEW / f"face_frames_{role}{'_ctx' if ctx else ''}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.concatenate(rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return out


def stage_face_landmarks(n_frames: int = 12, exclude: tuple[str, ...] = (), reuse: bool = False, yaw_max: float = 45.0, occ_min: float = 8.0,
                         relabel: dict[str, str] | None = None, min_gap_s: float = 1.5) -> dict:
    cand = json.loads((FACE_ID / "candidates.json").read_text())
    cand = CA.apply_relabel(cand, load_relabel(relabel))
    exclude = load_exclude(exclude)
    w = PACKAGE_ROOT / "weights" / "face"
    allp = FACE_ID / "landmarks_all.json"
    done = {r["name"]: r for r in json.loads(allp.read_text())} if allp.exists() else {}
    if not reuse:
        images = []
        for role in ROLES:
            p = portrait_path(role)
            if p is not None and f"portrait_{role}" not in done:
                images.append({"name": f"portrait_{role}", "file": str(p), "face": None})
        images += [{"name": n, "file": str(FACE_ID / v["file"]), "face": v["face"]} for n, v in cand.items() if n not in done]    # incremental: only the new crops
        if images:
            jobs = FACE_ID / "landmark_jobs.json"
            part = FACE_ID / "landmarks_new.json"
            jobs.write_text(json.dumps({"mp_task": str(w / "face_landmarker.task"), "rtmw_onnx": str(w / "rtmw_x384" / "end2end.onnx"), "images": images}))
            r = CA.run_low([CA.face_python(), str(Path(CA.__file__).with_name("recap_face_lm.py")), "--jobs", str(jobs), "--out", str(part)])
            if r.returncode != 0:
                raise RuntimeError("landmarks failed: " + r.stderr[-2000:])
            done.update({x["name"]: x for x in json.loads(part.read_text())})
            allp.write_text(json.dumps(list(done.values())))
    recs = json.loads(allp.read_text())
    rep: dict = {"exclude": list(exclude)}
    for role in ROLES:
        mine = [x for x in recs if x["name"] in cand and cand[x["name"]]["role"] == role]
        best = CA.select_faces(mine, cand, n_frames, yaw_max, min_gap_s, occ_min, (CAM,), exclude)
        por = next((x for x in recs if x["name"] == f"portrait_{role}"), None)
        out = []
        if portrait_path(role) is not None:
            if por is None or not por["ok"]:
                raise RuntimeError(f"no face found in the portrait of the {role}")
            out.append({**por, "name": "portrait", "file": rel_to_work(por["file"], role)})
        out += [{**x, "file": rel_to_work(x["file"], role)} for x in best]
        (FACE_ID / f"landmarks_{role}.json").write_text(json.dumps(out, indent=1))
        rep[role] = {"detected_by_mediapipe": sum(1 for x in mine if x["ok"]), "of": len(mine), "chosen": [x["name"] for x in best],
                     "portrait": None if por is None else {"yaw": round(por["yaw"], 1), "occ_px": round(por["occ_px"], 1)},
                     "video_occ_px": [round(x["occ_px"], 1) for x in best], "video_yaw": [round(x["yaw"], 1) for x in best]}
        if best:
            contact_sheet(role, best)
            contact_sheet(role, best, ctx=True)
    (FACE_ID / "landmarks_report.json").write_text(json.dumps(rep, indent=1))
    return rep


# ------------------------------------------------------------------------------------------------ pure parts: shoes

def foot_box(wb_foot: np.ndarray, pad: float = 0.45, up: float = 0.12) -> tuple[float, float, float, float, float] | None:
    """(x0, y0, x1, y1, L) of one foot from its three whole-body keypoints (big toe, small toe, heel; (3,3) [x, y, score]): the box of the points padded
    by `pad` x the foot length (more above, `up`, for the instep / ankle); None when a point is missing (score < 0.4)."""
    if (wb_foot[:, 2] < 0.4).any():
        return None
    pts = wb_foot[:, :2]
    L = float(max(np.hypot(*(pts[0] - pts[2])), np.hypot(*(pts[1] - pts[2]))))
    if L < 4.0:
        return None
    x0, y0 = pts.min(0) - pad * L
    x1, y1 = pts.max(0) + pad * L
    y0 -= up * L
    return float(x0), float(y0), float(x1), float(y1), L


def split_strata(rows: np.ndarray, frac: float = 0.3) -> np.ndarray:
    """Boolean per pixel row y (N,): the lowest `frac` of the span = the sole stratum."""
    lo, hi = float(rows.min()), float(rows.max())
    return rows >= hi - frac * max(hi - lo, 1.0)


def chromatic(px_rgb: np.ndarray, min_chroma: float = 14.0) -> np.ndarray:
    """Pixels (N,3) with colour in them (max - min channel >= min_chroma): a pink or tan shoe on a grey floor / grey trousers; neutral greys drop out."""
    p = np.asarray(px_rgb, float)
    return (p.max(1) - p.min(1)) >= min_chroma


def not_floor(px_rgb: np.ndarray, floor_rgb: np.ndarray, dist: float = 22.0) -> np.ndarray:
    """Pixels (N,3) that differ from the floor colour by more than `dist` (sRGB 0..255 Euclidean)."""
    return np.linalg.norm(np.asarray(px_rgb, float) - np.asarray(floor_rgb, float), axis=1) > dist


def stage_shoes_video(n: int = 70, min_gap_s: float = 1.0) -> dict:
    """Shoe colours from the video at the 6 foot keypoints: per dancer the frames where both feet are confidently placed (the dancer alone, not in a zoom
    step, not blurred), the pixels in the foot box that belong to the dancer's mask (not the partner's) and are not the floor, clustered
    (CA.dominant_colours); the lowest 30 % of the foot box rows = the sole stratum. The video is graded by the phone: relative values only."""
    import cv2
    people = load_people()
    masks = np.load(PEOPLE / f"masks_{CAM}.npz", allow_pickle=True)
    events = zoom_events()
    t, fi = people["t"], people["frame_index"].astype(int)
    rep: dict = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "camera": CAM, "roles": {}}
    for role in ROLES:
        who = WHO[role]
        wb = people["kp_wb"][:, who]
        feet = np.minimum.reduce([wb[:, i, 2] for s in "LR" for i in WB_FEET[s]])
        ankle_y = people["kp"][:, who, 15:17, 1].mean(1)
        low = people["kp"][:, who, 15:17, 2].min(1) > 0.5
        ok = ((feet > 0.5) & low & (people["conf"][:, who] >= 0.5) & (people["role_conf"][:, who] >= 0.9) & ~people["merged"][:, who].astype(bool)
              & ~in_zoom_step(t, events, 0.4) & (people["skel_in_partner"][:, who] < 0.3))
        # a foot at rest is sharper: the ankle's image speed (px per frame)
        spd = np.full(len(t), np.nan)
        spd[1:-1] = np.hypot(*(0.5 * (wb[2:, 19, :2] + wb[2:, 22, :2]) - 0.5 * (wb[:-2, 19, :2] + wb[:-2, 22, :2])).T) / 2
        score = np.where(ok & np.isfinite(spd), 1.0 / (1.0 + spd) + 0.001 * np.nan_to_num(ankle_y) / 960.0, 0.0)
        sel = CA.pick_frames(score, t, n, min_gap_s)
        paths = {int(fi[i]): AV / "shoes_stills" / f"{role}_{int(fi[i]):06d}.jpg" for i in sel}
        CA.extract_frames(source_video(), list(paths.items()))
        upper_px, sole_px, floors = [], [], []
        used = 0
        for i in sel:
            p = paths[int(fi[i])]
            img = cv2.imread(str(p)) if p.exists() else None
            if img is None:
                continue
            own = _decode_mask(masks, int(fi[i]), who)
            par = _decode_mask(masks, int(fi[i]), 1 - who)
            if own is None:
                continue
            own = cv2.dilate(own.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            both = own | (par if par is not None else False)
            got = False
            for s in "LR":
                fb = foot_box(wb[i, list(WB_FEET[s])])
                if fb is None:
                    continue
                x0, y0, x1, y1, L = fb
                H, W = img.shape[:2]
                x0, y0, x1, y1 = int(max(0, x0)), int(max(0, y0)), int(min(W, x1)), int(min(H, y1))
                if x1 - x0 < 4 or y1 - y0 < 4:
                    continue
                crop = img[y0:y1, x0:x1][:, :, ::-1].astype(float)
                m_own = own[y0:y1, x0:x1]
                m_par = par[y0:y1, x0:x1] if par is not None else np.zeros_like(m_own)
                ring = ~both[max(0, y0 - 6):y1 + 6, max(0, x0 - 6):x1 + 6]
                rimg = img[max(0, y0 - 6):y1 + 6, max(0, x0 - 6):x1 + 6][:, :, ::-1].astype(float)
                if ring.sum() < 30:
                    continue
                floor = np.median(rimg[ring], 0)
                yy = np.broadcast_to(np.arange(y0, y1)[:, None], m_own.shape)
                keep = m_own & ~m_par & not_floor(crop.reshape(-1, 3), floor).reshape(m_own.shape)
                if keep.sum() < 12:
                    continue
                sole = split_strata(yy[keep])
                upper_px.append(crop[keep][~sole])
                sole_px.append(crop[keep][sole])
                floors.append(floor)
                got = True
            used += int(got)
        ux = np.concatenate(upper_px) if upper_px else np.zeros((0, 3))
        sx = np.concatenate(sole_px) if sole_px else np.zeros((0, 3))
        allpx = np.concatenate([ux, sx]) if len(ux) else ux
        rep["roles"][role] = {"frames": used, "frames_considered": len(sel), "pixels": int(len(allpx)),
                              "clusters": CA.dominant_colours(allpx, 3) if len(allpx) > 200 else [],
                              "upper_clusters": CA.dominant_colours(ux, 2) if len(ux) > 200 else [],
                              "sole_stratum_clusters": CA.dominant_colours(sx, 2) if len(sx) > 100 else [],
                              "chromatic_clusters": CA.dominant_colours(allpx[chromatic(allpx)], 2) if int(chromatic(allpx).sum()) > 200 else [],
                              "chromatic_share": round(float(chromatic(allpx).mean()), 3) if len(allpx) else 0.0,
                              "floor_median_hex": CA.srgb_hex(np.median(np.asarray(floors), 0)) if floors else None,
                              "note": "pixels of the foot boxes inside the dancer's own mask, not the floor colour; the lowest 30 % of rows = sole stratum"}
        log(f"{role}: {used} frames, {len(allpx)} px, chromatic {[c['hex'] for c in rep['roles'][role]['chromatic_clusters']]}, upper {[c['hex'] for c in rep['roles'][role]['upper_clusters']]}, sole stratum "
            f"{[c['hex'] for c in rep['roles'][role]['sole_stratum_clusters']]}")
    AV.mkdir(parents=True, exist_ok=True)
    (AV / "shoes_video.json").write_text(json.dumps(rep, indent=1, default=float))
    return rep


# ------------------------------------------------------------------------------------------------ pure parts + stage: texture preview

def choose_windows_overlap(scores: np.ndarray, starts_s: np.ndarray, n: int, min_start_gap_s: float) -> list[int]:
    """Greedy best window starts with at least min_start_gap_s between any two starts (the clip is short: windows may overlap)."""
    out: list[int] = []
    for i in np.argsort(-scores):
        if len(out) >= n:
            break
        if all(abs(float(starts_s[i]) - float(starts_s[j])) >= min_start_gap_s for j in out):
            out.append(int(i))
    return out


def coverage_objective(sight: np.ndarray, frames: np.ndarray, contact: np.ndarray, need: float) -> float:
    """Score of a set of frames (bool (T,)): the mean over dancers and sectors of the sector's quality-weighted seconds capped at `need`, plus 0.25 x the
    worst sector of the worse dancer, minus 0.5 x the share of embrace frames."""
    if not frames.any():
        return 0.0
    cov = np.minimum(sight[frames].sum(0) / need, 1.0)
    return float(cov.mean() + 0.25 * cov.min(axis=1).min() - 0.5 * contact[frames].mean())


def choose_complementary(sight: np.ndarray, contact: np.ndarray, starts: np.ndarray, length: int, n: int, min_start_gap: int, need: float,
                         penalty: np.ndarray | None = None) -> list[int]:
    """Greedy windows (indices into `starts`, frame starts of `length` frames) that together cover the most body sectors: each pick maximises the
    coverage_objective of the UNION of the frames chosen so far (an overlapped frame counts once), so the next window adds the sides the earlier ones lack.
    `penalty` (per start) is subtracted from the union score when that window is added."""
    T = len(sight)
    pen = np.zeros(len(starts)) if penalty is None else penalty
    chosen: list[int] = []
    union = np.zeros(T, bool)
    for _ in range(n):
        best, arg = -1e9, -1
        for k, s in enumerate(starts):
            if k in chosen or any(abs(int(s) - int(starts[j])) < min_start_gap for j in chosen):
                continue
            u = union.copy()
            u[s:s + length] = True
            v = coverage_objective(sight, u, contact, need) - float(pen[k]) - float(sum(pen[j] for j in chosen))
            if v > best:
                best, arg = v, k
        if arg < 0:
            break
        chosen.append(arg)
        union[starts[arg]:starts[arg] + length] = True
    return chosen


def intervals(mask: np.ndarray, t: np.ndarray, gap_s: float = 1.5, min_s: float = 0.5) -> list[list[float]]:
    """[t0, t1] runs of True in `mask` (frame times t), runs closer than gap_s merged, runs shorter than min_s dropped."""
    out: list[list[float]] = []
    idx = np.nonzero(mask)[0]
    if len(idx) == 0:
        return out
    a = b = float(t[idx[0]])
    for k in idx[1:]:
        tk = float(t[k])
        if tk - b <= gap_s:
            b = tk
        else:
            out.append([a, b])
            a = b = tk
    out.append([a, b])
    return [[round(x, 2), round(y, 2)] for x, y in out if y - x >= min_s]


def side_seconds(group: np.ndarray, weight: np.ndarray, dt: float) -> dict[str, dict]:
    """{side: {seconds (frames x dt where seen), weighted (quality-weighted seconds)}} for one dancer."""
    return {nm: {"seconds": round(float(((group == g) & (weight > 0)).sum() * dt), 1), "weighted": round(float(weight[group == g].sum() * dt), 1)}
            for g, nm in enumerate(SIDE_NAMES)}


def stage_texture_preview(length_s: float = 36.0, n_windows: int = 3, need_s: float = 6.0, min_start_gap_s: float = 14.0) -> dict:
    """PROVISIONAL texture windows. Per frame and dancer: the body yaw against the camera (facing_angle: shoulder foreshortening + nose / toe direction),
    8 body-relative sectors (CA.sector_of), a quality weight = resolution (the mask box height in px per metre of stature against 400) x confidently
    detected x not hidden behind the partner x visible mask x outside the zoom steps. Windows = greedy on the UNION coverage (choose_complementary: sectors capped at
    `need_s` quality-weighted seconds + the worst sector - the embrace share), starts >= min_start_gap_s apart; each pick adds the sides the earlier lack.
    ONE moving camera: a window textures only the side it SEES; the fusion of windows is what gives the other sides."""
    people = load_people()
    cfg = take_cfg()
    events = zoom_events()
    t = people["t"]
    T = len(t)
    dt = 1.0 / FPS
    zoom = in_zoom_step(t, events, 0.35)
    sight = np.zeros((T, 2, CA.N_SECTORS))
    groups, weights, psis = [], [], []
    per_dancer: dict = {}
    for role in ROLES:
        who = WHO[role]
        kp, wb = people["kp"][:, who], people["kp_wb"][:, who]
        sr = shoulder_ratio(kp)
        ref = rho_reference(sr["rho"])
        psi = facing_angle(kp, wb, ref)
        box = people["box"][:, who]
        h_px = (box[:, 3] - box[:, 1]).astype(float)
        pxm = h_px / float(cfg["dancers"][role]["stature_m"])
        q = np.clip(pxm / 400.0, 0.0, 1.5)
        ok = ((people["conf"][:, who] >= 0.5) & (people["skel_in_partner"][:, who] < 0.5) & (people["visible"][:, who] >= 0.5) & ~zoom
              & np.isfinite(psi) & (people["role_conf"][:, who] >= 0.5))
        w = np.where(ok, q, 0.0)
        sec = CA.sector_of(np.nan_to_num(psi))
        np.add.at(sight[:, WHO[role]], (np.arange(T), sec), w * dt)
        g = side_group(psi)
        groups.append(g)
        weights.append(w)
        psis.append(psi)
        per_dancer[role] = {"rho_ref": round(ref, 3), "frames_with_yaw": int(np.isfinite(psi).sum()), "frames_used": int((w > 0).sum()),
                            "sides": side_seconds(g, w, dt),
                            "sector_weighted_seconds": [round(float(sight[:, WHO[role], s].sum()), 1) for s in range(CA.N_SECTORS)],
                            "intervals_s": {nm: intervals((g == k) & (w > 0), t) for k, nm in enumerate(SIDE_NAMES)}}
    contact = (people["merged"].any(1) | (people["box_iou"] > 0.3)).astype(float)
    length = int(round(length_s * FPS))
    step = int(round(FPS))
    t0s = np.arange(0, T - length + 1, step)
    sc = CA.window_scores(sight, contact, t0s, length, need_s, dt)
    # a window that crosses a zoom step is allowed (the bake knows the focal of every frame) but each step costs a little (the blurred 0.7 s, two scales)
    cross = np.array([sum(1 for a, b in events if t[s] < b and t[min(s + length, T - 1)] > a) for s in t0s])
    sc = sc - 0.04 * cross
    best = choose_complementary(sight, contact, t0s, length, n_windows, int(round(min_start_gap_s * FPS)), need_s, 0.04 * cross)
    union = np.zeros(T, bool)
    for b in best:
        union[t0s[b]:t0s[b] + length] = True
    wins = []
    for k, b in enumerate(sorted(best, key=lambda i: t[t0s[i]])):
        s0, s1 = int(t0s[b]), int(t0s[b]) + length
        cov = {role: {nm: round(float(sum(sight[s0:s1, WHO[role], s] for s in secs).sum()), 1) for nm, secs in
                      (("front", (7, 0, 1)), ("right_side", (2,)), ("back", (3, 4, 5)), ("left_side", (6,)))} for role in ROLES}
        wins.append({"id": f"tx{k + 1}", "t0": round(float(t[s0]), 2), "t1": round(float(t[min(s1, T - 1)]), 2), "score": round(float(sc[b]), 3),
                     "contact_share": round(float(contact[s0:s1].mean()), 2), "zoom_steps_inside": int(cross[b]),
                     "weighted_seconds_by_side": cov,
                     "sector_cover": {role: [round(float(min(1.0, sight[s0:s1, WHO[role], s].sum() / need_s)), 2) for s in range(CA.N_SECTORS)] for role in ROLES}})
    res = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "provisional": True,
           "source": "2D keypoints / masks of the single handheld camera (people/teachers_01.npz); replaced by texture-select on the fitted SMPL-X poses",
           "method": "yaw from shoulder foreshortening (sign from nose / toe direction); weight = px per metre / 400 x detected x visible x not behind the partner x outside zoom steps",
           "length_s": length_s, "need_weighted_s": need_s, "sectors": "8 body-relative sectors, 0 = the dancer faces the camera, 4 = away; 2 = the dancer's right side seen",
           "windows": wins, "best_score": float(sc.max()), "median_score": float(np.median(sc)), "per_dancer": per_dancer,
           "union_of_windows": {"seconds": round(float(union.sum() * dt), 1), "score": round(coverage_objective(sight, union, contact, need_s), 3),
                                "weighted_seconds_by_side": {role: {nm: round(float(sum(sight[union, WHO[role], s].sum() for s in secs)), 1) for nm, secs in
                                                                    (("front", (7, 0, 1)), ("right_side", (2,)), ("back", (3, 4, 5)), ("left_side", (6,)))} for role in ROLES},
                                "sector_cover": {role: [round(float(min(1.0, sight[union, WHO[role], s].sum() / need_s)), 2) for s in range(CA.N_SECTORS)] for role in ROLES}},
           "zoom_events": events, "honest": ["one monocular handheld camera: only the visible side of a window is textured, windows must be fused",
                                            "the yaw sign (left vs right side) is a nose / toe cue: unreliable for the lateral views",
                                            "the weight is a resolution proxy from the mask box height, not the true px/m of the fitted body"]}
    p = AV / "texture" / "windows_preview.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=1))
    return res


# ------------------------------------------------------------------------------------------------ plan

def exists_any(rel: str) -> bool:
    from glob import glob
    if any(c in rel for c in "{(<") or rel.startswith(("StreamingAssets", "work/")):
        return False
    return bool(glob(str(WORK / rel)))


def status_of(st: CA.Stage) -> str:
    expanded = []
    for o in st.writes:
        o = o.split(" (")[0]
        if not o.startswith(("avatar/", "face_shape/", "review/")):
            continue
        expanded += [o.replace("{lead,follow}", r) for r in ROLES] if "{lead,follow}" in o else [o]
    if not expanded:
        return "-"
    have = sum(1 for o in expanded if exists_any(o))
    return "done" if have == len(expanded) else ("partial" if have else "todo")


def plan(as_json: bool = False) -> dict:
    rows = [{**asdict(s), "status": status_of(s)} for s in STAGES]
    smplx = all((WORK / "smplx" / f"{r}.npz").exists() for r in ROLES)
    doc = {"take": TAKE, "fitted_poses_present": smplx, "stages": rows}
    if as_json:
        print(json.dumps(doc, indent=1))
        return doc
    for s in rows:
        print(f"{s['name']:<16} {'POSES ' if s['poses'] else 'free  '} {s['where']:<5} {s['status']:<8} {s['cost']}")
        print(f"    {s['what']}")
        print(f"    cmd: {'python -m dancecap.jl_avatars ' if not s['cmd'].startswith(('python', '(')) else ''}{s['cmd']}")
    print(f"\nfitted poses (work/{TAKE}/smplx/*.npz): {'present' if smplx else 'NOT there yet: the pose stages wait'}")
    return doc


# ------------------------------------------------------------------------------------------------ head review (provisional betas)

def stage_head_review() -> dict:
    import cv2

    from . import recap_avatars_review as rv
    from . import shape_edits as se
    betas = CA.pinned_betas(CA.PROVISIONAL_BETAS)
    scratch = AV / "review_scratch"
    rep: dict = {"provisional_betas": True, "betas": "the recap's pinned betas (stand-in body; J3b's shape is not there yet): the HEAD is judged"}
    for role in ROLES:
        sk, mp = CA.rest_skin(betas[role])
        alb = rv.load_albedo(CA.flat_albedo(scratch / f"flat_{role}.png", CA.SKIN_STANDIN[role]))
        lm_path = FACE_ID / f"landmarks_{role}.json"
        lm = {r["name"]: r for r in json.loads(lm_path.read_text())} if lm_path.exists() else {}
        yaw = float(lm["portrait"]["yaw"]) if "portrait" in lm else 0.0
        rows_tex, numbers = [], {}
        variants: dict[str, dict | None] = {"template (betas)": None}
        for name, suffix in (("portrait only", "_portrait"), ("video only", "_video"), ("identity (mixed)", "")):
            f = FACE_SHAPE / f"{role}_identity{suffix}.npz"
            if f.exists():
                variants[name] = {"identity": f"work/{TAKE}/face_shape/{f.name}", "identity_weight": 1.0, "identity_mouth": 0.4}
        for name, face in variants.items():
            if face is None:
                v = sk.vertices
                numbers[name] = {"after": {k: x for k, x in se.face_measures(v).items() if k != "ratios"}}
            else:
                v, r = se.apply_vertices({"face": face}, sk.vertices, sk.joints, sk.weights, sk.faces, role, scratch)
                numbers[name] = {"after": {k: x for k, x in r["face"]["after"].items() if k != "ratios"}, "max_move_mm": r["max_move_mm"]}
            rows_tex.append((name, rv.head_tiles(sk, v, mp, alb, yaw)))
            log(f"{role}: {name} rendered")
        pp = portrait_path(role)
        pt = rv.portrait_tile(pp, lm["portrait"]) if (pp is not None and "portrait" in lm) else None
        REVIEW.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(REVIEW / f"head_{role}.jpg"), rv.sheet(rows_tex, pt), [cv2.IMWRITE_JPEG_QUALITY, 90])
        rep[role] = numbers
    rv.write_report(AV / "shape" / "head_review.json", rep)
    return {r: list(v) for r, v in rep.items() if r in ROLES} | {"provisional_betas": True}


# ------------------------------------------------------------------------------------------------ stage: hair-groom (pose-dependent: needs her rest model)

def stage_hair_groom(from_take: str) -> dict:
    """avatar/hair_groom.json of the follow: a donor take's groom (roots = SMPL-X vertex ids + barycentrics, measured data) re-based onto HER rest model
    (smplx/follow.npz), then dancecap.hair_ponytail's restyle ([dancers.follow] hair_style = "ponytail": tie, tail, cards, dynamics) and her palette
    (avatar/hair_palette.json). Not exported here: the demo export copies it (`python -m dancecap.hair_groom john_lula --export-only` after a re-export)."""
    CA.require([WORK / "smplx" / "follow.npz"], "hair-groom")
    from . import hair_groom, hair_ponytail
    from .take import Take
    take = Take.load(TAKE)
    if hair_ponytail.hair_style_of(take) != "ponytail":
        raise SystemExit(f"{TAKE}: [dancers.follow] hair_style is not 'ponytail'")
    g = hair_groom.seed_from(take, Take.load(from_take), do_export=False)
    p = (g.get("style") or {}).get("ponytail") or {}
    return {"groom": str(AV / "hair_groom.json"), "style": g["style"]["type"], "tail_length_m": p.get("tail_length_m"),
            "problems": hair_ponytail.validate(p), "albedo_linear": g["render"].get("albedo_linear"), "from_take": from_take}


# ------------------------------------------------------------------------------------------------ main

def csv(s: str) -> tuple[str, ...]:
    return tuple(x for x in s.split(",") if x)


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(prog="python -m dancecap.jl_avatars", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("face-frames")
    p.add_argument("--per-role", type=int, default=40)
    p.add_argument("--min-gap", type=float, default=0.5)
    p.add_argument("--max-speed", type=float, default=0.5, help="head speed limit, eye distances per frame (blur)")
    p.add_argument("--max-off", type=float, default=0.6, help="nose offset limit, eye distances (0 = frontal)")
    p = sub.add_parser("face-landmarks")
    p.add_argument("--n-frames", type=int, default=12)
    p.add_argument("--exclude", default="")
    p.add_argument("--reuse", action="store_true")
    p.add_argument("--yaw-max", type=float, default=45.0)
    p.add_argument("--occ-min", type=float, default=8.0)
    p.add_argument("--relabel", default="", help="name=role,...: crops that show the OTHER dancer (kept in avatar/face_id/relabel.json)")
    for nm in ("face-identity",):
        p = sub.add_parser(nm)
        p.add_argument("--max-cost", type=float, default=1.0)
        p.add_argument("--vcpu", type=int, default=8)
        p.add_argument("--minutes", type=float, default=30)
        p.add_argument("--hard-minutes", type=float, default=50)
        p.add_argument("--job-id", default=None)
        p.add_argument("--upload-models", action="store_true")
        p.add_argument("--dry-run", action="store_true")
    sub.add_parser("face-trust")
    sub.add_parser("shoes-video")
    p = sub.add_parser("texture-preview")
    p.add_argument("--length", type=float, default=36.0)
    p.add_argument("--n-windows", type=int, default=3)
    p.add_argument("--need", type=float, default=6.0)
    p.add_argument("--min-start-gap", type=float, default=14.0)
    sub.add_parser("head-review")
    p = sub.add_parser("hair-groom")
    p.add_argument("--from-take", required=True, help="donor take with an avatar/hair_groom.json (roots, measured data)")
    a = ap.parse_args(argv)
    if a.cmd == "plan":
        plan(a.json)
    elif a.cmd == "face-frames":
        print(json.dumps(stage_face_frames(a.per_role, a.min_gap, max_speed=a.max_speed, max_off=a.max_off), indent=1))
    elif a.cmd == "face-landmarks":
        print(json.dumps(stage_face_landmarks(a.n_frames, csv(a.exclude), a.reuse, a.yaw_max, a.occ_min,
                                              dict(x.split("=") for x in csv(a.relabel))), indent=1))
    elif a.cmd == "shoes-video":
        print(json.dumps(stage_shoes_video(), indent=1, default=float))
    elif a.cmd == "texture-preview":
        r = stage_texture_preview(a.length, a.n_windows, a.need, a.min_start_gap)
        print(json.dumps({k: r[k] for k in ("windows", "best_score", "median_score")} | {"per_dancer": {k: v["sides"] for k, v in r["per_dancer"].items()}}, indent=1))
    elif a.cmd == "head-review":
        print(json.dumps(stage_head_review(), indent=1))
    elif a.cmd == "hair-groom":
        print(json.dumps(stage_hair_groom(a.from_take), indent=1))
    else:
        from . import jl_avatars_pod as pod
        if a.cmd == "face-trust":
            print(json.dumps(pod.stage_face_trust(), indent=1))
            return 0
        return pod.cli_job(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
