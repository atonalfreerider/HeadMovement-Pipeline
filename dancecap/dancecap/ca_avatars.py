"""Carlos & Aline avatars (agent C5): the avatar job graph of the take `carlos_aline` (two tripod cameras, one heat) - the recap's
avatar stages (recap_avatars.py: READ-ONLY template) re-expressed for a take whose cameras are two panning / zooming tripod heads.
One command per stage; every stage re-runnable on its own.

    python -m dancecap.ca_avatars plan [--json]               the stage table with its status (what exists, what waits)
    python -m dancecap.ca_avatars <stage> [options]           run one stage

POSE-INDEPENDENT stages (portraits, the 2D detections of every person (work/carlos_aline/room_work/podout/pose_{A,B}.npz, C2) and
the camera tracks; they run before the stitched poses exist):
    face-frames      the clearest, least blurred, most frontal faces of each dancer in camera 01 (the tele, 26-39 px between the eyes)
                     and camera 02 (the wide, 17-26 px) -> avatar/face_id/frames/*.jpg, candidates.json (local ffmpeg)
    face-landmarks   MediaPipe 478 + RTMW iBUG-68 on the two portraits and those frames (.venv-face) -> avatar/face_id/landmarks_*.json,
                     review/avatar/face_frames_*.jpg (LOOK at them: exclude hands / the partner / blur with --exclude)
    face-identity    POD (kind `avatarca`, one CPU pod, nothing else claims it): SMIRK + landmark refinement + robust combination per
                     dancer -> face_shape/<role>_identity{,_portrait,_video}.npz; with the pinned betas present also the retarget to the rest
                     head -> face_shape/<role>_identity_disp.npz (the export's cache)
    face-retarget    the same pod job, retarget only (when the betas land after the identity: ~3 min of a CPU pod)
    hair-ref         hair colour / length of both portraits (MediaPipe hair segment) and of the video (cam 01 stills at the 2D head box)
    shoes-video      shoe colours measured in the video at the 2D ankle keypoints (cam 01 stills): the seed of avatar/shoes.json
    texture-preview  PROVISIONAL texture windows from the triangulated COCO joints (cameras/dancers_world.npz) + the fixed camera
                     centres: body-relative sector coverage per window; the stitched-pose `texture-select` replaces it
    body-shape       the pinned betas (C3's shape pass) against the stated statures + the TOML body shape_edits before / after
    head-review      renders of the rest heads (template / identity / TOML block) beside the portraits (provisional betas while
                     C3's pin is absent: the heads are what is judged, not the body)

POSE-DEPENDENT stages (wait for the stitched smplx/{lead,follow}.npz; `plan` shows what they read):
    texture-select, texture-run (pod), texture-fuse, face-texture, texture-paint, hair-groom, shoes, shoe-size, export
    (AVATAR_PLAN.md: the commands, the windows, the fusion). They reuse recap_avatars_fuse / dancecap.avatar_face / dancecap.hair_groom /
    dancecap.shoes unchanged: those are take-agnostic once given the work dir.

Pod jobs follow the farm worker's design (dancecap.farm.worker): ONE job spec of kind `avatarca` (no farm worker's --kinds contains
it), id `ca-<stage>-<time>`, ONE pod started by this module with HM_KINDS=avatarca, results pulled, the pod deleted in a `finally`.
Ledger owner `C7-ca-avatars`. Likeness data (portraits, face frames, identities) stays under work/ (git-ignored) and on the owner's
private volume; no tracked file carries a portrait, a face crop or a colour sampled from a person.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .take import PACKAGE_ROOT

TAKE = "carlos_aline"
OWNER = "C7-ca-avatars"
KIND = "avatarca"
ROLES = ("lead", "follow")
NAMES = {"lead": "Carlos", "follow": "Aline"}
WHO = {"lead": 0, "follow": 1}                    # dancer_table index (CARLOS, ALINE)
CAMS = {"01": {"pose": "A", "fps": 24000 / 1001, "h_min": 150.0, "native": (1920, 1080), "video": "01"},
        "02": {"pose": "B", "fps": 30.0, "h_min": 100.0, "native": (1920, 906), "video": "02"}}
WORK = PACKAGE_ROOT / "work" / TAKE
AV = WORK / "avatar"
FACE_ID = AV / "face_id"
FACE_SHAPE = WORK / "face_shape"
POSE_DIR = WORK / "room_work" / "podout"
SHAPE_WORK = PACKAGE_ROOT / "work" / "carlos_aline_cashape"          # C3's shape pass
PIN = SHAPE_WORK / "mamma_pin_betas.json"
PROVISIONAL_BETAS = PACKAGE_ROOT / "work" / "larissa_kadu_recap_shape" / "mamma_pin_betas.json"   # head review only (same statures +-2 cm)
REL = f"dancecap/work/{TAKE}"
PRICE_CPU_8 = 0.28
WINDOW = (3.6, 198.0)


def log(msg: str) -> None:
    print(f"[ca_avatars {time.strftime('%H:%M:%S')}] {msg}", flush=True)


@dataclass(frozen=True)
class Stage:
    name: str
    poses: bool
    where: str
    cmd: str
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    cost: str
    what: str


STAGES: tuple[Stage, ...] = (
    Stage("face-frames", False, "local", "face-frames [--per-cam 01=24,02=10]", ("room_work/podout/pose_{A,B}.npz", "cameras/{01,02}.npz", "source videos"),
          ("avatar/face_id/frames/*.jpg", "avatar/face_id/candidates.json"), "0 (ffmpeg, ~1 min)",
          "sharp, frontal, large faces of each dancer from the 2D detections of every person (identity = torso brightness, C2)"),
    Stage("face-landmarks", False, "local", "face-landmarks [--exclude NAMES] [--cams 01] [--reuse]", ("avatar/face_id/candidates.json", "portraits"),
          ("avatar/face_id/landmarks_{lead,follow}.json", "review/avatar/face_frames_{lead,follow}.jpg"), "0 (.venv-face, ~3 min)",
          "MediaPipe + RTMW landmarks; the best frames per dancer (LOOK at the contact sheets)"),
    Stage("face-identity", False, "pod", "face-identity [--upload-models] [--max-cost 1.0]",
          ("avatar/face_id/landmarks_*.json + frames", "portraits", "[betas if the pin exists]", "volume: models/flame, models/smirk"),
          ("face_shape/{lead,follow}_identity.npz", "face_shape/{lead,follow}_identity_disp.npz (with betas)", "avatar/face_id/fit_report.json"),
          "~0.05-0.10 (CPU pod 8 vCPU, ~10 min)", "FLAME identity per dancer from portrait + video faces (+ the retarget to the pinned betas)"),
    Stage("face-retarget", False, "local|pod", "face-retarget [--local | --max-cost 0.5]", ("face_shape/{lead,follow}_identity.npz", "mamma_pin_betas.json (C3)"),
          ("face_shape/{lead,follow}_identity_disp.npz",), "0 local (~25 s measured) or ~0.03 (CPU pod)", "the identity displacement cached for THESE betas (the export then needs no FLAME)"),
    Stage("hair-ref", False, "local", "hair-ref", ("portraits", "cam 01 stills", "pose_A.npz"), ("avatar/hair_ref/hair_ref.json",), "0 (~1 min)",
          "hair colour (portrait + video) and visible length of both dancers"),
    Stage("shoes-video", False, "local", "shoes-video", ("pose_A.npz", "cam 01 stills"), ("avatar/shoes_video.json",), "0 (~1 min)",
          "shoe upper / sole colours measured in the video at the ankle keypoints"),
    Stage("texture-preview", False, "local", "texture-preview [--length 45] [--need 6] [--chunks]", ("cameras/dancers_world.npz", "cameras/{01,02}.npz"),
          ("avatar/texture/windows_preview.json",), "0 (seconds)", "provisional texture windows (sector coverage by the two fixed cameras) before the stitched poses"),
    Stage("body-shape", False, "local", "body-shape", ("pinned betas (C3)", "takes/carlos_aline.toml"), ("avatar/shape/body_report.json",), "0 (~10 s)",
          "stature / proportions of the betas, the body shape_edits before / after"),
    Stage("head-review", False, "local", "head-review", ("face_shape/*", "betas (pinned, else provisional)", "portraits"), ("review/avatar/head_{lead,follow}.jpg",),
          "0 (~3 min)", "front / 3-4 / profile renders of the heads beside the portraits"),
    Stage("texture-select", True, "local", "texture-select [--length 45] [--max-windows 3] [--need 6]", ("smplx/{lead,follow}.npz", "cameras/*_world.npz"),
          ("avatar/texture/windows.json",), "0", "windows of the stitched poses where both cameras see the dancer well and every body side is covered"),
    Stage("texture-run", True, "pod", "texture-run --window-id tx1 [--which 0] [--own-kind] [--max-cost 1.6]",
          ("windows.json", "sliced stitched smplx", "volume: source videos"), ("work/carlos_aline_<window>/avatar/*",), "~0.3-0.5 per window (4090 pod)",
          "front (frames, SAM masks of the window) then the hair-aware bake, ONE pod job per window"),
    Stage("texture-install", True, "local", "texture-install --window-id tx1 [--force]", ("work/carlos_aline_<window>/avatar/*",), ("avatar/{lead,follow}_albedo.png",), "0",
          "ONE window's bake into avatar/ (a stopgap; texture-fuse is the real one)"),
    Stage("texture-fuse", True, "local", "texture-fuse --windows tx1,tx2,tx3",
          ("work/carlos_aline_<window>/avatar/*_texels.npz",), ("avatar/{lead,follow}_albedo.png", "avatar/fuse_report.json"), "0 (~1 min)",
          "per-texel confidence-weighted fusion of the window bakes (thin two-camera coverage: several windows)"),
    Stage("face-texture", True, "local", "face-texture [--composite]", ("avatar/{lead,follow}_albedo.png", "smplx/*.npz", "portraits"),
          ("avatar/{lead,follow}_albedo_face.png",), "0 (~9 min CPU)", "the portrait faces composited onto the bake"),
    Stage("hair-groom", True, "local", "hair-groom", ("smplx/follow.npz", "avatar/follow_albedo_face.png"),
          ("avatar/hair_groom.json",), "0 (~3 min)", "Aline's hair seeded from the demo's groom, re-coloured near-black, length per the plan"),
    Stage("shoes", True, "local", "shoes", ("capture", "avatar/shoes_override.json", "avatar/shoes_video.json"), ("avatar/shoes.json",), "0 (<1 min)",
          "sample + apply the colours / style override"),
    Stage("export", True, "local", "python -m dancecap.demo_export carlos_aline --capture 07_CarlosAline (export agent)", ("everything above", "[dancers.*] of the take TOML"),
          ("StreamingAssets/07_CarlosAline/",), "0", "the Unity capture"),
)


# ------------------------------------------------------------------------------------------------ take config

def take_cfg() -> dict:
    return tomllib.loads((PACKAGE_ROOT / "takes" / f"{TAKE}.toml").read_text(encoding="utf-8"))


def source_dir() -> Path:
    p = PACKAGE_ROOT / "takes" / f"{TAKE}.toml"
    return (p.parent / take_cfg()["source_dir"]).resolve()


def portrait_path(role: str) -> Path:
    d = take_cfg()["dancers"][role]
    p = source_dir() / (d.get("portrait") or f"{d['name'].lower()}.jpg")
    if not p.exists():
        raise FileNotFoundError(f"portrait of {d['name']}: {p} missing")
    return p


def source_video(cam: str) -> Path:
    files = {"01": "Carlos Aline Zouk Artist Jack Jill at Zouk SEA 2019.mp4", "02": "Zouk Sea 2019 Carlos and Aline Borges Jack Jill.mp4"}
    return source_dir() / files[cam]


def pinned_betas(path: Path | None = None) -> dict[str, np.ndarray] | None:
    p = path or PIN
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return {"lead": np.asarray(d["0"], float), "follow": np.asarray(d["1"], float)}


def _below_normal() -> int:
    return 0x00004000 if os.name == "nt" else 0


def run_low(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, creationflags=_below_normal(), **kw)


# ------------------------------------------------------------------------------------------------ pure parts: face candidates

def face_metrics(kp: np.ndarray, kp_min: float = 0.5) -> dict[str, np.ndarray]:
    """Per frame of one dancer, kp (N,17,3) [x, y, score] with NaN where absent (cameras.twocam_people.dancer_table): iod (the eye
    distance, px), seen (nose and both eyes above kp_min), off (nose offset from the eye midpoint in eye distances: 0 = frontal)."""
    iod = np.hypot(kp[:, 1, 0] - kp[:, 2, 0], kp[:, 1, 1] - kp[:, 2, 1])
    seen = (kp[:, 0, 2] > kp_min) & (kp[:, 1, 2] > kp_min) & (kp[:, 2, 2] > kp_min) & np.isfinite(iod)
    off = np.abs(kp[:, 0, 0] - 0.5 * (kp[:, 1, 0] + kp[:, 2, 0])) / np.maximum(iod, 1e-6)
    return {"iod": iod, "seen": seen, "off": off}


def head_speed(kp: np.ndarray, iod: np.ndarray) -> np.ndarray:
    """Head speed in eye distances per frame (central difference of the eye midpoint; the nose is not used: it moves with the turn):
    motion blur grows with it. NaN where a neighbour frame lacks the dancer."""
    mid = 0.5 * (kp[:, 1, :2] + kp[:, 2, :2])
    v = np.full(len(kp), np.nan)
    d = mid[2:] - mid[:-2]
    v[1:-1] = np.hypot(d[:, 0], d[:, 1]) / 2.0
    return v / np.maximum(iod, 1e-6)


def pick_frames(score: np.ndarray, t: np.ndarray, n: int, min_gap_s: float) -> list[int]:
    """Indices of the n best scores (score > 0), at least min_gap_s apart in t (greedy by score)."""
    out: list[int] = []
    for i in np.argsort(-score):
        if score[i] <= 0 or len(out) >= n:
            break
        if all(abs(t[i] - t[j]) >= min_gap_s for j in out):
            out.append(int(i))
    return out


def face_scores(m: dict[str, np.ndarray], speed: np.ndarray, max_off: float = 0.40, max_speed: float = 0.30, iod_min: float = 0.0) -> np.ndarray:
    """Candidate score per frame: the eye distance, discounted for head speed (blur), 0 where not seen, too turned or too small."""
    ok = m["seen"] & (m["off"] <= max_off) & (m["iod"] >= iod_min) & np.isfinite(speed) & (speed <= max_speed)
    return np.where(ok, m["iod"] / (1.0 + 2.0 * np.nan_to_num(speed)), 0.0)


def face_box(kp: np.ndarray, scale: float = 5.5) -> list[float]:
    pts = kp[[0, 1, 2], :2]
    iod = float(np.hypot(*(kp[1, :2] - kp[2, :2])))
    c = pts.mean(0)
    return [float(c[0]), float(c[1]), scale * iod]


# ------------------------------------------------------------------------------------------------ stage: face-frames

TRACK_JOINTS = (0, 5, 6, 11, 12)           # nose, shoulders, hips: the joints the identity continuity compares


def appearance_evidence(rgb: np.ndarray) -> np.ndarray:
    """Per detection pair, rgb (2,3) mean torso colours of the two largest boxes -> e: > 0 says detection 0 is Carlos (white top: brighter and
    less red), < 0 detection 1. e = (luma0 - luma1) / 50 + (redness1 - redness0) / 0.3 with redness = (R - (G + B) / 2) / luma."""
    luma = rgb @ np.array([0.3, 0.59, 0.11])
    red = (rgb[:, 0] - 0.5 * (rgb[:, 1] + rgb[:, 2])) / np.maximum(luma, 1.0)
    e = (luma[0] - luma[1]) / 50.0 + (red[1] - red[0]) / 0.3
    return float(np.nan_to_num(e))


def match_cost(kp_a: np.ndarray, kp_b: np.ndarray, scale: float) -> float:
    """Mean distance of the tracking joints between two detections (kp (17,3)), in `scale` units; 1.0 when no joint is confident in both."""
    ok = (kp_a[list(TRACK_JOINTS), 2] > 0.3) & (kp_b[list(TRACK_JOINTS), 2] > 0.3)
    if not ok.any():
        return 1.0
    d = np.hypot(*(kp_a[list(TRACK_JOINTS), :2] - kp_b[list(TRACK_JOINTS), :2]).T)[ok]
    return float(d.mean() / max(scale, 1.0))


def identity_viterbi(evidence: np.ndarray, flip_match: np.ndarray, margin: np.ndarray, contiguous: np.ndarray, penalty: float = 6.0,
                     ev_clip: float = 3.0) -> np.ndarray:
    """Which of the two ordered detections is Carlos, per frame, by dynamic programming. State s_t = 0 (detection 0 is Carlos) / 1.
    evidence[t] > 0 favours s_t = 0 (appearance_evidence); flip_match[i] (0/1) says the min-cost matching of the detections of entry i and i+1 maps
    detection a -> a (0) or a -> 1 - a (1) (the same person), margin[i] its decisiveness in [0, 1]; contiguous[i] False = a gap (no link).
    Changing identity against the matching costs `penalty` x margin. -> s (T,) of 0/1."""
    T = len(evidence)
    unary = np.stack([-np.clip(evidence, -ev_clip, ev_clip) / 2.0, np.clip(evidence, -ev_clip, ev_clip) / 2.0], 1)
    cost = np.zeros((T, 2))
    back = np.zeros((T, 2), int)
    cost[0] = unary[0]
    for i in range(1, T):
        for s in (0, 1):
            best, arg = np.inf, 0
            for q in (0, 1):
                pen = 0.0
                if contiguous[i - 1] and ((q ^ s) != int(flip_match[i - 1])):
                    pen = penalty * float(margin[i - 1])
                c = cost[i - 1, q] + pen
                if c < best:
                    best, arg = c, q
            cost[i, s] = best + unary[i, s]
            back[i, s] = arg
    s = np.zeros(T, int)
    s[-1] = int(np.argmin(cost[-1]))
    for i in range(T - 1, 0, -1):
        s[i - 1] = back[i, s[i]]
    return s


def identity_table(det: dict, n_frames: int, h_min: float, h_ratio: float = 0.55, min_conf: float = 0.3) -> dict:
    """Like cameras.twocam_people.dancer_table (the two largest boxes of a frame are the dancers) but the Carlos / Aline assignment is a
    Viterbi over time (appearance evidence + continuity of the head / shoulder / hip joints between consecutive frames) instead of a per-frame
    torso-brightness vote, which swaps the two whenever his white sleeve covers her torso. Frames with exactly one dancer box are left out
    (the face stages do not need them). -> {'kp': (N,2,17,3), 'box': (N,2,4), 'ok': (N,2), 'flips_vs_brightness': int, 'frames': int}."""
    fr, box, kp, rgb, cf = det["frame"], det["box"], det["kp"], det["rgb"], det["conf"]
    h = box[:, 3] - box[:, 1]
    order = np.argsort(fr, kind="stable")
    uf, start = np.unique(fr[order], return_index=True)
    end = np.r_[start[1:], len(order)]
    pairs: list[tuple[int, int, int]] = []
    for f, s0, s1 in zip(uf, start, end):
        idx = order[s0:s1]
        idx = idx[(h[idx] >= h_min) & (cf[idx] >= min_conf)]
        if len(idx) < 2:
            continue
        idx = idx[h[idx] >= h_ratio * h[idx].max()]
        idx = idx[np.argsort(-h[idx])][:2]
        if len(idx) == 2:
            pairs.append((int(f), int(idx[0]), int(idx[1])))
    out_kp = np.full((n_frames, 2, 17, 3), np.nan, np.float32)
    out_box = np.full((n_frames, 2, 4), np.nan, np.float32)
    if not pairs:
        return {"kp": out_kp, "box": out_box, "ok": np.zeros((n_frames, 2), bool), "flips_vs_brightness": 0, "frames": 0}
    ev = np.array([appearance_evidence(rgb[[a, b]]) for _, a, b in pairs])
    flip = np.zeros(len(pairs) - 1)
    marg = np.zeros(len(pairs) - 1)
    cont = np.zeros(len(pairs) - 1, bool)
    for i in range(len(pairs) - 1):
        (f0, a0, b0), (f1, a1, b1) = pairs[i], pairs[i + 1]
        cont[i] = (f1 - f0) <= 3
        if not cont[i]:
            continue
        sc = float(max(h[a0], h[b0]))
        keep = match_cost(kp[a0], kp[a1], sc) + match_cost(kp[b0], kp[b1], sc)
        swap = match_cost(kp[a0], kp[b1], sc) + match_cost(kp[b0], kp[a1], sc)
        flip[i] = 0 if keep <= swap else 1
        marg[i] = min(1.0, abs(keep - swap) / 0.15)
    s = identity_viterbi(ev, flip, marg, cont)
    brightness = (ev < 0).astype(int)            # the per-frame rule: detection 0 is Carlos iff evidence > 0
    for (f, a, b), si in zip(pairs, s):
        c, al = (a, b) if si == 0 else (b, a)
        out_kp[f, 0], out_box[f, 0] = kp[c], box[c]
        out_kp[f, 1], out_box[f, 1] = kp[al], box[al]
    return {"kp": out_kp, "box": out_box, "ok": ~np.isnan(out_kp[:, :, 0, 0]), "flips_vs_brightness": int((s != brightness).sum()), "frames": len(pairs)}


_TABLES: dict[str, dict] = {}


def dancer_tables() -> dict[str, dict]:
    """{cam: identity_table} from the people2d detections of every person (cameras/people2d: pose_{A,B}.npz of C2's pods); memoised per process."""
    if _TABLES:
        return _TABLES
    out = _TABLES
    for cam, c in CAMS.items():
        z = np.load(POSE_DIR / f"pose_{c['pose']}.npz")
        det = {n: z[n] for n in ("frame", "box", "conf", "kp", "rgb")}
        out[cam] = identity_table(det, int(z["n_frames"]), c["h_min"])
        log(f"cam {cam}: {out[cam]['frames']} two-dancer frames, identity differs from the per-frame brightness vote in {out[cam]['flips_vs_brightness']}")
    return out


def cam_rows(cam: str) -> dict:
    z = np.load(WORK / "cameras" / f"{cam}.npz")
    return {"frame_index": z["frame_index"].astype(int), "t": z["t"].astype(float), "ok": z["ok"].astype(bool),
            "pts": z["pts"].astype(float) if "pts" in z.files else z["frame_index"].astype(float) / CAMS[cam]["fps"]}


def extract_frame(video: Path, pts: float, out: Path) -> bool:
    out.parent.mkdir(parents=True, exist_ok=True)
    r = run_low(["ffmpeg", "-y", "-v", "error", "-ss", f"{max(pts, 0.0):.4f}", "-i", str(video), "-frames:v", "1", "-q:v", "2", str(out)])
    return r.returncode == 0 and out.exists() and out.stat().st_size > 0


def extract_frames(video: Path, items: list[tuple[int, Path]]) -> int:
    """Write the decoded frames `frame_index` -> file in ONE sequential pass (cv2; an ffmpeg seek per still costs minutes on the busy PC). The
    frame index counts decoded frames (both videos are constant frame rate from PTS 0). -> the number written."""
    import cv2
    todo = {int(f): p for f, p in items if not p.exists()}
    if not todo:
        return 0
    cap = cv2.VideoCapture(str(video))
    n, last = 0, max(todo)
    i = 0
    while i <= last:
        if not cap.grab():
            break
        if i in todo:
            ok, img = cap.retrieve()
            if ok:
                todo[i].parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(todo[i]), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                n += 1
        i += 1
    cap.release()
    return n


def stage_face_frames(per_cam: dict[str, int] | None = None, min_gap_s: float = 2.0) -> dict:
    per_cam = per_cam or {"01": 24, "02": 10}
    tabs = dancer_tables()
    cand: dict[str, dict] = {}
    for cam in per_cam:
        rows = cam_rows(cam)
        fi, tt = rows["frame_index"], rows["t"]
        for role in ROLES:
            kp = tabs[cam]["kp"][fi, WHO[role]]
            m = face_metrics(kp)
            sp = head_speed(kp, m["iod"])
            full_iod = m["iod"].copy()
            sc = face_scores(m, sp)
            sel = pick_frames(sc, tt, per_cam[cam], min_gap_s)
            log(f"{role} cam {cam}: {int(m['seen'].sum())} frames with a face, eye distance median {np.nanmedian(full_iod[m['seen']]):.0f} px, "
                f"{int((sc > 0).sum())} usable (sharp, frontal), {len(sel)} chosen")
            extract_frames(source_video(cam), [(int(fi[i]), FACE_ID / "frames" / f"{role}_{cam}_{int(fi[i]):06d}.jpg") for i in sel])
            for i in sel:
                f = int(fi[i])
                name = f"{role}_{cam}_{f:06d}"
                out = FACE_ID / "frames" / f"{name}.jpg"
                if not out.exists():
                    log(f"  {name}: frame not decoded")
                    continue
                cand[name] = {"role": role, "cam": cam, "frame_index": f, "t": float(tt[i]), "pts": float(rows["pts"][i]),
                              "iod_px": float(m["iod"][i]), "head_speed": float(sp[i]), "face": face_box(kp[i]), "file": f"frames/{name}.jpg"}
    FACE_ID.mkdir(parents=True, exist_ok=True)
    (FACE_ID / "candidates.json").write_text(json.dumps(cand, indent=1))
    return {"candidates": len(cand), "per_role_cam": {f"{r}_{c}": sum(1 for v in cand.values() if v["role"] == r and v["cam"] == c) for r in ROLES for c in per_cam}}


# ------------------------------------------------------------------------------------------------ stage: face-landmarks

def face_python() -> str:
    from .hair_reference import face_python as fp
    exe, _ = fp()
    if exe is None:
        raise RuntimeError(".venv-face (mediapipe + rtmlib) is missing")
    return exe


def rel_to_work(file: str, role: str) -> str:
    p = Path(file)
    try:
        return p.resolve().relative_to(WORK).as_posix()
    except ValueError:
        return f"avatar/face_id/portrait_{role}{p.suffix}"


def select_faces(records: list[dict], cand: dict[str, dict], n: int, yaw_max: float, min_gap_s: float, occ_min: float, cams: tuple[str, ...],
                 exclude: tuple[str, ...]) -> list[dict]:
    """The best landmark records of a dancer's frames (same rule as recap_avatars.select_faces: a detected face, the outer-canthal px >=
    occ_min, yaw <= yaw_max, only `cams`, not `exclude`, ranked by sharpness x size, at least min_gap_s apart in a camera)."""
    pool = [r for r in records if r.get("ok") and r["name"] in cand and abs(r["yaw"]) <= yaw_max and r["occ_px"] >= occ_min
            and cand[r["name"]]["cam"] in cams and r["name"] not in exclude]
    pool.sort(key=lambda r: -r["sharpness"] * r["occ_px"])
    out: list[dict] = []
    for r in pool:
        c = cand[r["name"]]
        if all(not (cand[o["name"]]["cam"] == c["cam"] and abs(cand[o["name"]]["t"] - c["t"]) < min_gap_s) for o in out):
            out.append(r)
        if len(out) >= n:
            break
    return out


def contact_sheet(role: str, recs: list[dict], size: int = 224) -> Path:
    import cv2

    from .face_smirk import warp_crop
    allrec = json.loads((FACE_ID / "landmarks_all.json").read_text())
    por = next(x for x in allrec if x["name"] == f"portrait_{role}")
    tiles = []
    for r, f in [(por, str(portrait_path(role)))] + [(x, str(WORK / x["file"])) for x in recs]:
        img = cv2.imread(f if Path(f).is_absolute() else str(WORK / f))
        if img is None:
            continue
        t = warp_crop(img, np.asarray(r["smirk_center"]), float(r["smirk_side"]), size)
        cv2.putText(t, r["name"][-18:], (4, 16), 0, 0.45, (255, 255, 255), 1)
        tiles.append(t)
    rows = [np.concatenate(tiles[i:i + 6] + [np.zeros_like(tiles[0])] * (6 - len(tiles[i:i + 6])), 1) for i in range(0, len(tiles), 6)]
    out = WORK / "review" / "avatar" / f"face_frames_{role}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.concatenate(rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return out


def apply_relabel(cand: dict[str, dict], relabel: dict[str, str]) -> dict[str, dict]:
    """The pose detector hands the head keypoints of one dancer to the other in embraces (the Viterbi identity cannot see that: keypoints of
    the wrong head are consistent in time), so a human reviews the contact sheets and names the crops that show the OTHER dancer:
    relabel = {candidate name: role the crop actually shows}. Returns a copy with role replaced (and `tracker_role` kept)."""
    out = {n: dict(v) for n, v in cand.items()}
    for n, role in relabel.items():
        if n in out and role in ROLES and out[n]["role"] != role:
            out[n]["tracker_role"] = out[n]["role"]
            out[n]["role"] = role
    return out


def load_relabel(extra: dict[str, str] | None = None) -> dict[str, str]:
    p = FACE_ID / "relabel.json"
    cur = json.loads(p.read_text()) if p.exists() else {}
    cur.update(extra or {})
    if extra:
        p.write_text(json.dumps(cur, indent=1))
    return cur


def stage_face_landmarks(n_frames: int = 12, cams: tuple[str, ...] = ("01",), exclude: tuple[str, ...] = (), reuse: bool = False,
                         yaw_max: float = 42.0, occ_min: float = 12.0, relabel: dict[str, str] | None = None) -> dict:
    cand = json.loads((FACE_ID / "candidates.json").read_text())
    rl = load_relabel(relabel)
    cand = apply_relabel(cand, rl)
    w = PACKAGE_ROOT / "weights" / "face"
    allp = FACE_ID / "landmarks_all.json"
    if not (reuse and allp.exists()):
        images = [{"name": f"portrait_{role}", "file": str(portrait_path(role)), "face": None} for role in ROLES]
        images += [{"name": n, "file": str(FACE_ID / v["file"]), "face": v["face"]} for n, v in cand.items()]
        jobs = FACE_ID / "landmark_jobs.json"
        jobs.write_text(json.dumps({"mp_task": str(w / "face_landmarker.task"), "rtmw_onnx": str(w / "rtmw_x384" / "end2end.onnx"), "images": images}))
        r = run_low([face_python(), str(Path(__file__).with_name("recap_face_lm.py")), "--jobs", str(jobs), "--out", str(allp)])
        if r.returncode != 0:
            raise RuntimeError("landmarks failed:\n" + r.stderr[-2000:])
    recs = json.loads(allp.read_text())
    rep: dict = {}
    for role in ROLES:
        mine = [x for x in recs if x["name"] in cand and cand[x["name"]]["role"] == role]
        best = select_faces(mine, cand, n_frames, yaw_max, 2.0, occ_min, cams, exclude)
        por = next(x for x in recs if x["name"] == f"portrait_{role}")
        if not por["ok"]:
            raise RuntimeError(f"no face found in the portrait of the {role}")
        out = [{**por, "name": "portrait", "file": rel_to_work(por["file"], role)}] + [{**x, "file": rel_to_work(x["file"], role)} for x in best]
        (FACE_ID / f"landmarks_{role}.json").write_text(json.dumps(out, indent=1))
        rep[role] = {"detected": sum(1 for x in mine if x["ok"]), "of": len(mine), "relabelled_in": sorted(n for n, v in cand.items() if v["role"] == role and "tracker_role" in v),
                     "chosen": [x["name"] for x in best],
                     "portrait_yaw": round(por["yaw"], 1), "portrait_occ_px": round(por["occ_px"], 1),
                     "video_occ_px": [round(x["occ_px"], 1) for x in best]}
        contact_sheet(role, best)
    (FACE_ID / "landmarks_report.json").write_text(json.dumps(rep, indent=1))
    return rep


# ------------------------------------------------------------------------------------------------ pure parts: hair / shoe colour

def srgb_hex(rgb: np.ndarray) -> str:
    r, g, b = (int(round(float(np.clip(x, 0, 255)))) for x in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def dominant_colours(pix: np.ndarray, k: int = 2, iters: int = 12) -> list[dict]:
    """k-means on sRGB pixels (N,3) 0..255 -> [{rgb, share}] by share, descending. Deterministic (initialised on the luma extremes)."""
    pix = np.asarray(pix, float)
    if len(pix) == 0:
        return []
    luma = pix @ np.array([0.2126, 0.7152, 0.0722])
    q = np.linspace(5, 95, k)
    c = np.stack([pix[np.argmin(np.abs(luma - np.percentile(luma, p)))] for p in q])
    for _ in range(iters):
        lab = np.argmin(((pix[:, None, :] - c[None]) ** 2).sum(-1), 1)
        for j in range(k):
            if (lab == j).any():
                c[j] = pix[lab == j].mean(0)
    lab = np.argmin(((pix[:, None, :] - c[None]) ** 2).sum(-1), 1)
    out = [{"rgb": [float(x) for x in c[j]], "hex": srgb_hex(c[j]), "share": float((lab == j).mean())} for j in range(k) if (lab == j).any()]
    return sorted(out, key=lambda d: -d["share"])


def box_around(p: np.ndarray, half: float, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    x0, y0 = int(max(0, p[0] - half)), int(max(0, p[1] - half))
    return x0, y0, int(min(w, p[0] + half)), int(min(h, p[1] + half))


# ------------------------------------------------------------------------------------------------ stage: hair-ref

def video_stills(cam: str, role: str, n: int = 14, min_gap_s: float = 6.0) -> list[dict]:
    """n well-spaced stills of the dancer's head (sharp frames, any yaw) from `cam`: [{t, frame, kp (17,3), file}] in avatar/hair_ref/stills/."""
    tabs = dancer_tables()
    rows = cam_rows(cam)
    kp = tabs[cam]["kp"][rows["frame_index"], WHO[role]]
    m = face_metrics(kp, 0.4)
    sp = head_speed(kp, m["iod"])
    sc = np.where(m["seen"] & np.isfinite(sp) & (sp < 0.3), m["iod"], 0.0)
    sel = pick_frames(sc, rows["t"], n, min_gap_s)
    paths = {int(rows["frame_index"][i]): AV / "hair_ref" / "stills" / f"{role}_{cam}_{int(rows['frame_index'][i]):06d}.jpg" for i in sel}
    extract_frames(source_video(cam), list(paths.items()))
    return [{"t": float(rows["t"][i]), "frame": int(rows["frame_index"][i]), "kp": kp[i].tolist(), "file": str(paths[int(rows["frame_index"][i])])}
            for i in sel if paths[int(rows["frame_index"][i])].exists()]


def stage_hair_ref() -> dict:
    """Hair colour / length. Portraits: hair_reference.portrait_reference (white-cloth calibrated: Carlos' white tee; Aline's portrait has no white
    cloth, so it falls back to the MediaPipe hair mask's own median). Video: median sRGB of the MediaPipe hair mask in the head crop of cam 01 stills
    (the grade is dark and graded: values are a sanity check of the portrait, not a calibrated colour)."""
    import cv2

    from . import hair_reference as hr
    out = AV / "hair_ref"
    out.mkdir(parents=True, exist_ok=True)
    rep: dict = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "roles": {}}
    for role in ROLES:
        r: dict = {"name": NAMES[role]}
        pr = hr.portrait_reference(portrait_path(role), out / f"portrait_{role}", 0.8)
        pr.pop("_masks", None)
        r["portrait"] = {k: pr.get(k) for k in ("status", "hair_px", "visible_length_head_heights")}
        r["portrait"]["palette"] = {k: v.get("hex") for k, v in (pr.get("palette") or {}).items() if isinstance(v, dict)}
        if pr.get("status") != "ok":                       # no white cloth to calibrate on: the raw median of the segmented hair
            mpo = out / f"portrait_{role}" / "mp_portrait.png"
            img = cv2.imread(str(portrait_path(role)))
            if mpo.exists() and img is not None:
                m = cv2.imread(str(mpo), cv2.IMREAD_UNCHANGED)
                hair = cv2.erode((m[..., 0] > 127).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
                if hair.sum() > 200:
                    med = np.median(img[hair][:, ::-1], 0)
                    r["portrait"]["uncalibrated_hair_median_srgb"] = [round(float(x), 1) for x in med]
                    r["portrait"]["uncalibrated_hair_hex"] = srgb_hex(med)
                    r["portrait"]["uncalibrated_hair_p10_p90_luma"] = [round(float(x), 1) for x in np.percentile(img[hair][:, ::-1] @ [0.2126, 0.7152, 0.0722], [10, 90])]
        # the video: hair mask from MediaPipe on a square crop around the head
        stills = video_stills("01", role, 10)
        jobs, meta = [], []
        for s in stills:
            kp = np.asarray(s["kp"])
            img = cv2.imread(s["file"])
            c = kp[[0, 1, 2], :2].mean(0)
            iod = float(np.hypot(*(kp[1, :2] - kp[2, :2])))
            x0, y0, x1, y1 = box_around(c + [0, -0.4 * iod], 4.2 * iod, img.shape[:2])
            crop = img[y0:y1, x0:x1]
            cp = out / "crops" / f"{role}_{s['frame']:06d}.png"
            cp.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(cp), cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC))
            jobs.append({"image": str(cp), "box": None, "out": str(cp.with_suffix(".mask.png"))})
            meta.append(s["frame"])
        cols = []
        if jobs:
            st = hr.run_mediapipe(jobs, out / f"mp_jobs_{role}.json")
            r["video_segmentation"] = st.get("status")
            for j, fr in zip(jobs, meta):
                mp_ = Path(j["out"])
                if not mp_.exists():
                    continue
                m = cv2.imread(str(mp_), cv2.IMREAD_UNCHANGED)
                img = cv2.imread(j["image"])
                hair = cv2.erode((m[..., 0] > 127).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
                if hair.sum() > 400:
                    cols.append(np.median(img[hair][:, ::-1], 0))
        if cols:
            med = np.median(np.asarray(cols), 0)
            r["video"] = {"frames": len(cols), "median_srgb": [round(float(x), 1) for x in med], "hex": srgb_hex(med)}
        rep["roles"][role] = r
    (out / "hair_ref.json").write_text(json.dumps(rep, indent=1, default=float))
    return rep


# ------------------------------------------------------------------------------------------------ stage: shoes-video

ANKLE = {"L": 15, "R": 16}        # COCO ankle keypoints


def stage_shoes_video(n: int = 40) -> dict:
    """Shoe colours from cam 01: around each ankle keypoint (box of 0.22 x the ankle-to-knee length... the shoe sits below the ankle), the pixels
    that are not the floor (pale, low saturation) and not skin / trouser are clustered (dominant_colours). Reported per dancer: upper (dominant dark
    / coloured cluster) and the light share (sole). The video is graded dark: relative values only."""
    import cv2
    tabs = dancer_tables()
    rows = cam_rows("01")
    rep: dict = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "roles": {}}
    for role in ROLES:
        kp = tabs["01"]["kp"][rows["frame_index"], WHO[role]]
        sc = np.where((kp[:, 15, 2] > 0.5) & (kp[:, 16, 2] > 0.5) & (kp[:, 13, 2] > 0.5) & (kp[:, 14, 2] > 0.5), 1.0, 0.0)
        # shoes seen best when the foot is not moving much and the ankle is low: use the ankle's own speed
        sel = pick_frames(sc * (1 + np.nan_to_num(kp[:, 15, 1]) / 1080.0), rows["t"], n, 4.0)
        paths = {int(rows["frame_index"][i]): AV / "shoes_stills" / f"{role}_{int(rows['frame_index'][i]):06d}.jpg" for i in sel}
        extract_frames(source_video("01"), list(paths.items()))
        pix = []
        for i in sel:
            p = paths[int(rows["frame_index"][i])]
            if not p.exists():
                continue
            img = cv2.imread(str(p))
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            for a, kn in ((15, 13), (16, 14)):
                shin = float(np.hypot(*(kp[i, kn, :2] - kp[i, a, :2])))
                x0, y0, x1, y1 = box_around(kp[i, a, :2] + [0, 0.12 * shin], 0.16 * shin, img.shape[:2])
                crop, h = img[y0:y1, x0:x1], hsv[y0:y1, x0:x1]
                if crop.size == 0:
                    continue
                keep = ~((h[..., 1] < 55) & (h[..., 2] > 130))         # not the pale floor
                pix.append(crop[keep][:, ::-1])
        px = np.concatenate(pix) if pix else np.zeros((0, 3))
        rep["roles"][role] = {"name": NAMES[role], "frames": len(sel), "pixels": int(len(px)),
                              "clusters": dominant_colours(px, 3) if len(px) > 200 else []}
    # the white / light sole share is the floor-free light remainder: report it from the pale-but-not-floor pixels would need the mask: not attempted
    (AV / "shoes_video.json").write_text(json.dumps(rep, indent=1, default=float))
    return rep


# ------------------------------------------------------------------------------------------------ pure parts: provisional texture windows

N_SECTORS = 8
COCO = {"ls": 5, "rs": 6, "lh": 11, "rh": 12}


def wrap(a: np.ndarray) -> np.ndarray:
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def sector_of(rel: np.ndarray, n: int = N_SECTORS) -> np.ndarray:
    """Body-relative sector 0..n-1: 0 = a camera straight in front of the dancer, n/2 = behind."""
    return (np.floor((wrap(rel) + np.pi / n) / (2 * np.pi / n)).astype(int)) % n


def camera_quality(dist_m: np.ndarray, focal_px: np.ndarray, ref_px_per_m: float = 400.0) -> np.ndarray:
    """Pixels per metre of the dancer (focal / distance) against a reference: 1.0 = ref_px_per_m (cam 01: ~475, cam 02: ~330 at 10 m)."""
    return np.clip((focal_px / np.maximum(dist_m, 1e-3)) / ref_px_per_m, 0.0, 1.5)


def window_scores(sector_sight: np.ndarray, contact: np.ndarray, t0s: np.ndarray, length: int, need: float, dt: float) -> np.ndarray:
    """sector_sight (T, roles=2, N_SECTORS) quality-weighted seconds of sight per frame (already x dt); contact (T,) 1 where the couple is in an
    embrace. Window score = mean over roles of the mean sector coverage (capped at 1 per sector after `need` s) + 0.25 x the worst sector of the worse
    role - 0.5 x the contact share (the partner's sleeve / hair smears the bake there)."""
    cs = np.concatenate([np.zeros((1,) + sector_sight.shape[1:]), np.cumsum(sector_sight, 0)])
    cc = np.concatenate([[0.0], np.cumsum(contact)])
    out = np.zeros(len(t0s))
    for k, s in enumerate(t0s):
        tot = cs[s + length] - cs[s]                     # (2, N)
        cov = np.minimum(tot / need, 1.0)
        out[k] = cov.mean() + 0.25 * cov.min(axis=1).min() - 0.5 * (cc[s + length] - cc[s]) / length
    return out


def choose_windows(scores: np.ndarray, t0s: np.ndarray, length: int, n: int, min_gap: int) -> list[int]:
    """Greedy best non-overlapping window starts (indices into t0s)."""
    out: list[int] = []
    for i in np.argsort(-scores):
        if len(out) >= n:
            break
        if all(abs(int(t0s[i]) - int(t0s[j])) >= length + min_gap for j in out):
            out.append(int(i))
    return out


def chunk_cores(t_min: float = 15.0) -> list[tuple[str, float, float]]:
    """[(chunk id, core start s, core end s)] of C3's farm chunk plan (work/carlos_aline/chunks.json), the first core cut at t_min (the mono span)."""
    plan = json.loads((WORK / "chunks.json").read_text())
    return [(c["id"], max(c["core_k"][0] / 30.0, t_min), (c["core_k"][1] + 1) / 30.0) for c in plan["chunks"]]


def best_in_ranges(scores: np.ndarray, t0s: np.ndarray, tt: np.ndarray, length: int, ranges: list[tuple[str, float, float]]) -> list[tuple[str, int]]:
    """For each (name, lo, hi) the index of the best window that lies inside [lo, hi] (None entries are skipped)."""
    out = []
    for name, lo, hi in ranges:
        ok = [i for i, s in enumerate(t0s) if tt[s] >= lo - 1e-6 and tt[s] + length / 10.0 <= hi + 1e-6]
        if ok:
            out.append((name, max(ok, key=lambda i: scores[i])))
    return out


def stage_texture_preview(length_s: float = 45.0, n_windows: int = 3, need_s: float = 6.0, contact_m: float = 0.8, chunks: bool = False) -> dict:
    """Provisional windows from the triangulated COCO joints (cameras/dancers_world.npz: 10 Hz, both cameras' reprojection-checked keypoints)."""
    z = np.load(WORK / "cameras" / "dancers_world.npz")
    t, person, joint, X, ok = z["t"], z["person"], z["joint"], z["X"], z["ok"].astype(bool)
    ts = np.round(t * 10).astype(int)
    k0, k1 = ts.min(), ts.max()
    T = k1 - k0 + 1
    J = np.full((2, 17, T, 3), np.nan)
    for p in (0, 1):
        for j in range(17):
            m = (person == p) & (joint == j) & ok
            J[p, j, ts[m] - k0] = X[m]
    centres = {c: np.asarray(np.load(WORK / "cameras" / f"{c}.npz")["centre_world"], float) for c in CAMS}
    cam_t = {c: np.load(WORK / "cameras" / f"{c}.npz")["t"] for c in CAMS}
    cam_f = {c: np.load(WORK / "cameras" / f"{c}.npz")["focal_px"] for c in CAMS}
    tt = (np.arange(T) + k0) / 10.0
    sight = np.zeros((T, 2, N_SECTORS))
    contact = np.zeros(T)
    pel = 0.5 * (J[:, 11] + J[:, 12])                                        # (2,T,3)
    d = np.hypot(pel[0, :, 0] - pel[1, :, 0], pel[0, :, 2] - pel[1, :, 2])
    contact = np.where(np.isfinite(d) & (d < contact_m), 1.0, 0.0)
    for p in (0, 1):
        head = np.arctan2(*(_fwd(J[p]).T[[0, 2]]))
        c_pel = pel[p]
        for c, C in centres.items():
            vis = np.interp(tt, cam_t[c], np.ones(len(cam_t[c])), left=0, right=0)       # the camera records there
            az = np.arctan2(C[0] - c_pel[:, 0], C[2] - c_pel[:, 2])
            dist = np.hypot(C[0] - c_pel[:, 0], C[2] - c_pel[:, 2])
            f = np.interp(tt, cam_t[c], cam_f[c])
            q = camera_quality(dist, f)
            good = np.isfinite(head) & np.isfinite(az) & (vis > 0.5)
            sec = sector_of(az - head)
            for s in range(N_SECTORS):
                sight[:, p, s] += np.where(good & (sec == s), q * 0.1, 0.0)
    length = int(length_s * 10)
    t0s = np.arange(0, T - length, 10)
    sc = window_scores(sight, contact, t0s, length, need_s, 0.1)
    if chunks:       # one window per farm chunk core (inside a core the stitched poses are the chunk fit bit for bit: no seam)
        best_named = best_in_ranges(sc, t0s, tt, length, chunk_cores())
        best = [b for _, b in best_named]
    else:
        best = choose_windows(sc, t0s, length, n_windows, 100)
    wins = [{"id": f"tx{i + 1}", "t0": float(tt[t0s[b]]), "t1": float(tt[t0s[b]] + length_s), "score": round(float(sc[b]), 3),
             "contact_share": round(float(contact[t0s[b]:t0s[b] + length].mean()), 2),
             "sector_cover": {NAMES[r]: [round(float(min(1.0, sight[t0s[b]:t0s[b] + length, p, s].sum() / need_s)), 2) for s in range(N_SECTORS)]
                              for p, r in enumerate(ROLES)}} for i, b in enumerate(best)]
    res = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "provisional": True,
           "source": "triangulated COCO joints (10 Hz) + fixed camera centres; replaced by texture-select on the stitched SMPL-X poses",
           "length_s": length_s, "windows": wins, "best_score": float(sc.max()), "median_score": float(np.median(sc))}
    p = AV / "texture" / "windows_preview.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=1))
    return res


def _fwd(J: np.ndarray) -> np.ndarray:
    """(17,T,3) -> (T,3) forward vectors (shoulder + hip lines)."""
    X = {k: J[i] for k, i in COCO.items()}
    f = np.zeros_like(X["ls"])
    for l, r in (("ls", "rs"), ("lh", "rh")):
        side = X[l] - X[r]                                # right -> left (a dancer facing +z, y up, has her left at +x)
        f += np.cross(side, np.array([0.0, 1.0, 0.0]))
    return f


# ------------------------------------------------------------------------------------------------ stage: body-shape / head-review (betas)

def rest_skin(betas: np.ndarray):
    from .export_unity import model_for, shaped_skin
    mp = model_for({"model": "smplx_locked_head"})
    return shaped_skin(mp, betas), mp


def stage_body_shape() -> dict:
    from . import recap_avatars as RA
    from . import shape_edits as se
    betas = pinned_betas()
    if betas is None:
        raise FileNotFoundError(f"{PIN} (C3's shape pass) does not exist yet: body-shape waits for it")
    cfg = take_cfg()
    out: dict = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "betas": "work/carlos_aline_cashape/mamma_pin_betas.json",
                 "gauge": "rest height = crown to sole of the shaped T-pose; target = stature_m + shoe_sole_m", "roles": {}}
    for role in ROLES:
        d = cfg["dancers"][role]
        stature, sole = float(d["stature_m"]), float(d.get("shoe_sole_m", 0.025))
        sk, _ = rest_skin(betas[role])
        rec = {"name": d["name"], "rest": RA.body_numbers(sk, stature, sole), "beta_norm": round(float(np.linalg.norm(betas[role])), 3)}
        edits = d.get("shape_edits") or {}
        body_cfg = {k: v for k, v in edits.items() if k != "face"}
        if body_cfg:
            v, rep = se.apply_vertices(body_cfg, sk.vertices, sk.joints, sk.weights, sk.faces, role, WORK)
            rec["edits_body"] = {"config": rep["config"], "moved_vertices": rep["moved_vertices"], "max_move_mm": rep["max_move_mm"],
                                 "after": {"foot_length_m": se.foot_lengths(v, sk.joints, sk.weights),
                                           "midsection": se.midsection_dims(v, sk.joints, sk.weights, sk.faces)}}
        out["roles"][role] = rec
        log(f"{d['name']}: rest height {rec['rest']['rest_height_m']} m (target {rec['rest']['stature_plus_sole_m']}, ratio {rec['rest']['ratio_to_stature']})")
    p = AV / "shape" / "body_report.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1, default=float))
    return out


def head_variants(role: str) -> dict[str, dict | None]:
    v: dict[str, dict | None] = {"template (betas)": None}
    rel = lambda n: f"work/{TAKE}/face_shape/{role}_identity{n}.npz"   # noqa: E731
    for name, suffix in (("portrait only", "_portrait"), ("video only", "_video"), ("identity (mixed)", "")):
        if (PACKAGE_ROOT / rel(suffix)).exists():
            v[name] = {"identity": rel(suffix), "identity_weight": 1.0, "identity_mouth": 0.4}
    own = ((take_cfg()["dancers"].get(role) or {}).get("shape_edits") or {}).get("face")
    if own:
        v["TOML face block"] = own
    return v


def flat_albedo(path: Path, skin_srgb: tuple[int, int, int], size: int = 256) -> Path:
    """A flat skin-tone stand-in albedo (no texture exists before the bake): the renders then judge SHAPE with a skin colour, not the likeness."""
    import cv2
    img = np.zeros((size, size, 3), np.uint8)
    img[:] = skin_srgb[::-1]
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
    return path


SKIN_STANDIN = {"lead": (125, 85, 62), "follow": (214, 168, 140)}      # stand-ins for the shape review only (neutral stand-ins)


def stage_head_review() -> dict:
    import cv2

    from . import recap_avatars_review as rv
    from . import shape_edits as se
    betas = pinned_betas()
    provisional = betas is None
    if provisional:
        betas = pinned_betas(PROVISIONAL_BETAS)
        log(f"{PIN.name} (C3) absent: heads rendered on the recap's pinned betas as a stand-in body (the face displacement is what is judged)")
    scratch = AV / "review_scratch"
    rep: dict = {"provisional_betas": provisional}
    for role in ROLES:
        sk, mp = rest_skin(betas[role])
        lm = {r["name"]: r for r in json.loads((FACE_ID / f"landmarks_{role}.json").read_text())}
        por = lm["portrait"]
        alb = rv.load_albedo(flat_albedo(AV / "review_scratch" / f"flat_{role}.png", SKIN_STANDIN[role]))
        rows_tex, rows_grey, numbers = [], [], {}
        for name, face in head_variants(role).items():
            if face is None:
                v = sk.vertices
                numbers[name] = {"after": {k: x for k, x in se.face_measures(v).items() if k != "ratios"}}
            else:
                v, r = se.apply_vertices({"face": face}, sk.vertices, sk.joints, sk.weights, sk.faces, role, scratch)
                numbers[name] = {"after": {k: x for k, x in r["face"]["after"].items() if k != "ratios"}, "max_move_mm": r["max_move_mm"]}
            rows_tex.append((name, rv.head_tiles(sk, v, mp, alb, float(por["yaw"]))))
            rows_grey.append((name, rv.head_tiles(sk, v, mp, None, float(por["yaw"]))))
            log(f"{role}: {name} rendered")
        pt = rv.portrait_tile(portrait_path(role), por)
        out = WORK / "review" / "avatar"
        out.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / f"head_{role}.jpg"), rv.sheet(rows_tex, pt), [cv2.IMWRITE_JPEG_QUALITY, 90])
        cv2.imwrite(str(out / f"head_{role}_grey.jpg"), rv.sheet(rows_grey, pt), [cv2.IMWRITE_JPEG_QUALITY, 90])
        rep[role] = numbers
    rv.write_report(AV / "shape" / "head_review.json", rep)
    return {r: list(v) for r, v in rep.items() if r in ROLES} | {"provisional_betas": provisional}


# ------------------------------------------------------------------------------------------------ pose-dependent stages (wait for smplx/)

def require(paths: list[Path], what: str) -> None:
    miss = [str(p.relative_to(PACKAGE_ROOT)) for p in paths if not p.exists()]
    if miss:
        raise FileNotFoundError(f"{what} needs {miss} (see `plan`: these come from the stitched poses / the texture jobs)")


def run_module(module: str, *args: str) -> dict:
    """python -m <module> <args> in this interpreter, below-normal priority, from dancecap/."""
    r = run_low([sys.executable, "-m", module, *args], cwd=PACKAGE_ROOT)
    tail = (r.stdout + r.stderr)[-3000:]
    if r.returncode != 0:
        raise RuntimeError(f"{module} {' '.join(args)} failed (rc {r.returncode}): {tail}")
    return {"module": module, "args": list(args), "tail": tail[-800:]}


def sector_seconds_q(person_xz: np.ndarray, heading: np.ndarray, cams: dict[str, np.ndarray], weight: dict[str, np.ndarray], dt: float,
                     n: int = N_SECTORS) -> np.ndarray:
    """(T, n) seconds of quality-weighted sight per frame for ONE dancer: cams {cam: (3,) fixed centre or (T,3)}, weight {cam: (T,)} = the camera's
    per-frame quality (0 where it does not see the dancer well: not recording, not ok, dancer not detected) x resolution (camera_quality)."""
    out = np.zeros((len(heading), n))
    for c, C in cams.items():
        C = np.broadcast_to(np.asarray(C, float), (len(heading), 3))
        az = np.arctan2(C[:, 0] - person_xz[:, 0], C[:, 2] - person_xz[:, 1])
        np.add.at(out, (np.arange(len(heading)), sector_of(az - heading, n)), weight[c] * dt)
    return out


def texture_select(work: Path, smplx_dir: Path, length_s: float = 45.0, step_s: float = 5.0, need_s: float = 6.0, max_windows: int = 3,
                   t_range: tuple[float, float] = (15.0, 196.5), contact_penalty: float = 0.3, out: Path | None = None) -> dict:
    """Windows of the STITCHED poses (smplx/{lead,follow}.npz: t, joints (T,J,3), global_orient (T,3)) where both fixed-centre cameras see the
    dancer well and every body side is seen: per dancer and frame the camera's azimuth against the body heading falls into one of 8 body-relative
    sectors; a camera counts with its resolution (focal / distance, camera_quality) and only while it records, its track is ok and the dancer's
    2D track is confident (people/teachers_<cam>.npz, when present). Window selection and the contact penalty are recap_avatars_select's (choose).
    t_range excludes the mono span (< 14.82 s: one camera, a weaker fit) and camera 01's zoom-out (> 196.9 s). Several windows (max_windows) because
    two cameras on the same side give each window thin coverage of the back sectors: they are baked separately and fused per texel."""
    from . import recap_avatars_select as rs
    sm = {r: np.load(smplx_dir / f"{r}.npz", allow_pickle=True) for r in ROLES}
    t_all = np.asarray(sm["lead"]["t"], float)
    m = (t_all >= t_range[0]) & (t_all <= t_range[1])
    t = t_all[m]
    dt = float(np.median(np.diff(t)))
    pel = {r: np.asarray(sm[r]["joints"], float)[m][:, 0] for r in ROLES}
    cams, weights = {}, {r: {} for r in ROLES}
    for c in CAMS:
        z = np.load(work / "cameras" / f"{c}.npz", allow_pickle=True)
        tc = np.asarray(z["t"], float)
        C = np.asarray(z["centre_world"], float)
        cams[c] = C
        rec = np.interp(t, tc, np.asarray(z["ok"], float), left=0.0, right=0.0) > 0.99      # camera records and its track is ok
        foc = np.interp(t, tc, np.asarray(z["focal_px"], float))
        tp = work / "people" / f"teachers_{c}.npz"
        seen = {r: np.ones(len(t), bool) for r in ROLES}
        if tp.exists():
            tz = np.load(tp)
            for i, r in enumerate(ROLES):
                good = (tz["conf"][:, i] > 0.5) & ~tz["merged"][:, i].astype(bool)
                seen[r] = np.interp(t, np.asarray(tz["t"], float), good.astype(float), left=0.0, right=0.0) > 0.5
        for r in ROLES:
            dist = np.hypot(C[0] - pel[r][:, 0], C[2] - pel[r][:, 2])
            weights[r][c] = camera_quality(dist, foc) * rec * seen[r]
    secs = [sector_seconds_q(pel[r][:, [0, 2]], rs.heading_azimuth(np.asarray(sm[r]["global_orient"])[m]), cams, weights[r], dt) for r in ROLES]
    contact = np.linalg.norm(pel["lead"][:, [0, 2]] - pel["follow"][:, [0, 2]], axis=1) < rs.CONTACT_M
    res = rs.choose(secs[0], secs[1], t, length_s, step_s, need_s, max_windows, contact)
    for i, w in enumerate(res["windows"]):
        w["id"] = f"tx{i + 1}"
    res.update({"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "smplx_dir": str(smplx_dir), "t_range": [float(t[0]), float(t[-1])],
                "sectors": f"{N_SECTORS} body-relative sectors, 0 = a camera straight in front; weights = resolution (focal / distance / 400 px per m) x ok x tracked"})
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=1))
    return res


# ------------------------------------------------------------------------------------------------ pure: recolour the groom (Aline's hair is black)

def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, float) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _colour_entry(srgb8) -> dict:
    s = [int(round(float(x))) for x in srgb8]
    return {"linear_srgb": [round(float(x), 4) for x in srgb_to_linear(np.array(s))], "srgb8": s, "hex": srgb_hex(np.array(s))}


# texture-fuse settings of THIS pair: Carlos' top is white (the fusion's "light" rule = the white top that hair / a dark sleeve only darken: the recap's follow),
# Aline's is red with a print and her hair swings over it (hair-cover weights apply to her already); a sharp winner for her print, a softer one for his smooth top
FUSE_LIGHT_ROLES = ("lead", "follow")
# use_prior=False: the follow window prior is the hair-residue metric (red-cast = hair): on a RED top it measures the top, not hair, and scaled every window to ~0.007
FUSE_POWERS = {"follow": {"low": 6.0, "high": 6.0}, "lead": {"low": 3.0, "high": 4.0}}

BLACK_HAIR = {"root": (14, 13, 16), "mid": (22, 20, 24), "tip": (28, 26, 30), "shadow": (6, 6, 8), "highlight": (64, 66, 84)}
HAIR_ALBEDO_GAIN = 0.9      # the red groom's 1.85 renders near-black hair grey in Unity (looked at, 2026-10-10); flip with [dancers.follow] hair_albedo_gain
HAIR_TIP_GLOW = 0.0         # the owner's white tip glow was stylisation of RED hair: on black hair it reads as white sparks (looked at); [dancers.follow] hair_tip_glow = 1 brings it back


def recolour_groom(g: dict, palette: dict[str, tuple[int, int, int]] = BLACK_HAIR, albedo_gain: float | None = None, tip_glow: float | None = None) -> dict:
    """A copy of a hair groom (the demo's red one, re-based by hair_groom --from-take) with the hair COLOUR replaced: colour.{root,mid,tip,shadow,
    highlight}, render.albedo_linear / albedo_portrait_linear (mid x albedo_gain: the groom's own rule, same hue), the gloss lobe's highlight, the
    red-specific copper lift off; the geometry (roots, lengths, style, colliders, dynamics) is untouched."""
    g = json.loads(json.dumps(g))
    col = g.setdefault("colour", {})
    for k, v in palette.items():
        col[k] = {**(col.get(k) if isinstance(col.get(k), dict) else {}), **_colour_entry(v)}
    col["source"] = "recoloured by dancecap.ca_avatars.recolour_groom (portrait: black with a blue sheen; video grade reads near-black)"
    r = g.setdefault("render", {})
    gain = float(albedo_gain if albedo_gain is not None else r.get("albedo_gain", 1.85))
    alb = [round(float(min(1.0, x * gain)), 4) for x in col["mid"]["linear_srgb"]]
    r["albedo_linear"] = alb
    r["albedo_portrait_linear"] = alb
    r["albedo_source"] = r["albedo_portrait_source"] = "recoloured: palette.mid x albedo_gain"
    kk = r.setdefault("kajiya_kay", {})
    hl = col["highlight"]["linear_srgb"]
    kk["highlight_linear"] = hl
    kk["highlight_portrait_linear"] = hl
    kk["primary_tint"] = [1.0, 1.0, 1.0]
    r["copper_lift"] = [1.0, 1.0, 1.0]
    r["copper_lift_m"] = 0.0
    glow = HAIR_TIP_GLOW if tip_glow is None else float(tip_glow)
    if glow != 1.0 and isinstance(r.get("tip_emission", {}).get("e"), list):       # the emission curve scaled (0 = no tip glow at all)
        r["tip_emission"]["e"] = [round(float(x) * glow, 5) for x in r["tip_emission"]["e"]]
        r["tip_emission"]["scaled_by"] = glow
    g["recoloured"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    return g


def stage_hair_groom(from_take: str = "larissa_kadu_full") -> dict:
    """The demo's groom (long hair, a side part, sleek) re-based onto Aline's rest model and recoloured black. hair_groom needs the stitched
    smplx/follow.npz (the rest model) and, for the cap's hairline, avatar/follow_albedo_face.png."""
    require([WORK / "smplx" / "follow.npz", AV / "follow_albedo_face.png"], "hair-groom")
    res = run_module("dancecap.hair_groom", TAKE, "--from-take", from_take, "--no-export")
    p = AV / "hair_groom.json"
    own = (take_cfg().get("dancers", {}).get("follow") or {})
    g = recolour_groom(json.loads(p.read_text(encoding="utf-8")), albedo_gain=float(own.get("hair_albedo_gain", HAIR_ALBEDO_GAIN)),
                       tip_glow=float(own.get("hair_tip_glow", HAIR_TIP_GLOW)))
    p.write_text(json.dumps(g, indent=1), encoding="utf-8")
    res["recoloured"] = {"albedo_gain": float(own.get("hair_albedo_gain", HAIR_ALBEDO_GAIN)), "tip_glow": float(own.get("hair_tip_glow", HAIR_TIP_GLOW))}
    return res


# ------------------------------------------------------------------------------------------------ plan

def exists_any(rel: str) -> bool:
    from glob import glob
    if any(c in rel for c in "{(") or rel.startswith(("StreamingAssets", "work/")):
        return False
    return bool(glob(str(WORK / rel)))


def status_of(st: Stage) -> str:
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
    doc = {"take": TAKE, "stitched_poses_present": smplx, "pin_betas_present": PIN.exists(), "stages": rows}
    if as_json:
        print(json.dumps(doc, indent=1))
        return doc
    for s in rows:
        print(f"{s['name']:<16} {'POSES ' if s['poses'] else 'free  '} {s['where']:<5} {s['status']:<8} {s['cost']}")
        print(f"    {s['what']}")
        print(f"    cmd: python -m dancecap.ca_avatars {s['cmd']}" if not s['cmd'].startswith("python") else f"    cmd: {s['cmd']}")
    print(f"\nstitched poses (work/{TAKE}/smplx/*.npz): {'present' if smplx else 'NOT there yet: the pose stages wait'}; "
          f"pinned betas (C3): {'present' if PIN.exists() else 'NOT there yet'}")
    return doc


# ------------------------------------------------------------------------------------------------ main

def csv(s: str) -> tuple[str, ...]:
    return tuple(x for x in s.split(",") if x)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.ca_avatars", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("face-frames")
    p.add_argument("--per-cam", default="01=24,02=10")
    p = sub.add_parser("face-landmarks")
    p.add_argument("--n-frames", type=int, default=12)
    p.add_argument("--cams", default="01")
    p.add_argument("--exclude", default="")
    p.add_argument("--reuse", action="store_true")
    p.add_argument("--yaw-max", type=float, default=42.0)
    p.add_argument("--occ-min", type=float, default=12.0)
    p.add_argument("--relabel", default="", help="name=role,...: crops that show the OTHER dancer (kept in avatar/face_id/relabel.json)")
    for nm in ("face-identity", "face-retarget"):
        p = sub.add_parser(nm)
        p.add_argument("--max-cost", type=float, default=1.0)
        p.add_argument("--vcpu", type=int, default=8)
        p.add_argument("--minutes", type=float, default=35)
        p.add_argument("--hard-minutes", type=float, default=60)
        p.add_argument("--job-id", default=None)
        p.add_argument("--upload-models", action="store_true")
        p.add_argument("--betas", type=Path, default=None, help="betas file (default C3's pin; else the identity is fitted without the retarget)")
        p.add_argument("--dry-run", action="store_true")
        p.add_argument("--local", action="store_true", help="face-retarget only: run on this PC (FLAME's retarget takes ~10 s per dancer)")
    p = sub.add_parser("pod-identity")
    p.add_argument("--work", type=Path)
    p.add_argument("--iters", type=int, default=600)
    p.add_argument("--limit", type=int)
    p.add_argument("--retarget-only", action="store_true")
    sub.add_parser("hair-ref")
    sub.add_parser("shoes-video")
    p = sub.add_parser("texture-preview")
    p.add_argument("--length", type=float, default=45.0)
    p.add_argument("--n-windows", type=int, default=3)
    p.add_argument("--need", type=float, default=6.0, help="quality-weighted seconds of sight that fill a body sector")
    p.add_argument("--chunks", action="store_true", help="the best window inside each farm chunk core instead of the best anywhere")
    sub.add_parser("body-shape")
    sub.add_parser("head-review")
    p = sub.add_parser("texture-select", help="windows of the STITCHED poses (needs smplx/)")
    p.add_argument("--smplx-dir", type=Path, default=WORK / "smplx")
    p.add_argument("--length", type=float, default=45.0)
    p.add_argument("--step", type=float, default=5.0)
    p.add_argument("--need", type=float, default=6.0)
    p.add_argument("--max-windows", type=int, default=3)
    p.add_argument("--t-min", type=float, default=15.0, help="skip the mono span (one camera) before this reference second")
    p.add_argument("--t-max", type=float, default=196.5, help="skip camera 01's zoom-out after this reference second")
    p.add_argument("--out", type=Path, default=AV / "texture" / "windows.json")
    p = sub.add_parser("texture-run", help="ONE pod job per window: the window's front + the hair-aware bake (needs smplx/ + windows.json)")
    p.add_argument("--windows", type=Path, default=AV / "texture" / "windows.json")
    p.add_argument("--which", type=int, default=0)
    p.add_argument("--window-id", default="tx1")
    p.add_argument("--smplx-dir", type=Path, default=WORK / "smplx")
    p.add_argument("--sync", type=Path)
    p.add_argument("--top-k", type=int, default=6)
    p.add_argument("--body-ref-q", type=float, default=0.7)
    p.add_argument("--step-ms", type=float, default=100.0)
    p.add_argument("--max-cost", type=float, default=1.6)
    p.add_argument("--minutes", type=float, default=60)
    p.add_argument("--hard-minutes", type=float, default=85)
    p.add_argument("--mask", default=None, help="cam:t0:t1[,..] reference seconds where a camera track is wrong (the bake skips them)")
    p.add_argument("--own-kind", action="store_true", help="the pod takes only this window's job (kind avatarca<id>): several windows in parallel")
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("texture-install")
    p.add_argument("--window-id", default="tx1")
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("texture-fuse")
    p.add_argument("--windows", default="tx1,tx2,tx3")
    p.add_argument("--paint", type=Path, default=AV / "paint_out.json")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--roles", default="lead,follow")
    p.add_argument("--weights", default="", help="window=factor,..: a window found dirtier counts less on every texel (default: all 1)")
    p.add_argument("--tau", type=float, default=0.10)
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("face-texture")
    p.add_argument("--composite", action="store_true")
    sub.add_parser("hair-groom")
    sub.add_parser("shoes")
    a = ap.parse_args(argv)
    if a.cmd == "plan":
        plan(a.json)
    elif a.cmd == "face-frames":
        pc = {k: int(v) for k, v in (x.split("=") for x in csv(a.per_cam))}
        print(json.dumps(stage_face_frames(pc)))
    elif a.cmd == "face-landmarks":
        print(json.dumps(stage_face_landmarks(a.n_frames, csv(a.cams), csv(a.exclude), a.reuse, a.yaw_max, a.occ_min,
                                              dict(x.split("=") for x in csv(a.relabel))), indent=1))
    elif a.cmd == "hair-ref":
        print(json.dumps(stage_hair_ref(), indent=1, default=float))
    elif a.cmd == "shoes-video":
        print(json.dumps(stage_shoes_video(), indent=1, default=float))
    elif a.cmd == "texture-preview":
        print(json.dumps(stage_texture_preview(a.length, a.n_windows, a.need, chunks=a.chunks), indent=1))
    elif a.cmd == "body-shape":
        stage_body_shape()
    elif a.cmd == "head-review":
        print(json.dumps(stage_head_review(), indent=1))
    elif a.cmd == "texture-select":
        r = texture_select(WORK, a.smplx_dir, a.length, a.step, a.need, a.max_windows, (a.t_min, a.t_max), out=a.out)
        print(json.dumps({k: r[k] for k in ("windows", "combined", "second_best_adds", "t_range")}, indent=1))
    elif a.cmd == "texture-fuse":
        from . import recap_avatars_fuse as fz
        print(json.dumps(fz.run(WORK, list(csv(a.windows)), a.paint, a.out, csv(a.roles), dry_run=a.dry_run, tau=a.tau,
                                  light_roles=FUSE_LIGHT_ROLES, powers=FUSE_POWERS, use_prior=False,
                                  window_weights={k: float(v) for k, v in (x.split("=") for x in csv(a.weights))}), indent=1, default=float))
    elif a.cmd == "face-texture":
        require([AV / f"{r}_albedo.png" for r in ROLES] + [WORK / "smplx" / f"{r}.npz" for r in ROLES], "face-texture")
        print(json.dumps(run_module("dancecap.avatar_face", TAKE, *(["--composite"] if a.composite else []))))
    elif a.cmd == "hair-groom":
        print(json.dumps(stage_hair_groom()))
    elif a.cmd == "shoes":
        print(json.dumps(run_module("dancecap.shoes", TAKE)))
    elif a.cmd == "texture-install":
        from . import ca_avatars_pod as pod
        print(json.dumps(pod.texture_install(a.window_id, a.force), indent=1))
    elif a.cmd == "texture-run":
        from . import ca_avatars_pod as pod
        return pod.texture_run(a)
    else:
        from . import ca_avatars_pod as pod
        if a.cmd == "pod-identity":
            pod.pod_identity(work=a.work, iters=a.iters, limit=a.limit, retarget_only=a.retarget_only)
            return 0
        return pod.cli_job(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
