"""Recap avatars (agent A1): the avatar job graph of the class recap `larissa_kadu_recap` - new avatars for Kadu (lead) and
Larissa (follow) from the Runpod pipeline. One command per stage.

    python -m dancecap.recap_avatars plan                       the stage table with its status (what exists, what waits)
    python -m dancecap.recap_avatars <stage> [options]          run one stage (see STAGES; `plan --json` lists the commands)

POSE-INDEPENDENT stages (need only the portraits, the recap's pinned betas and the people tracks; run now):
    face-frames      the clearest, most frontal video faces of each teacher (people/teachers_<cam>.npz: eye / nose / ear
                     keypoints of the owner's 2D poses) -> avatar/face_id/frames/*.jpg (local ffmpeg, a few dozen frames)
    face-landmarks   MediaPipe 478 + RTMW iBUG-68 on the portraits and those frames (.venv-face, seconds; the 229 MB
                     RTMW model would take longer to upload than to run) -> avatar/face_id/landmarks_<role>.json (the best
                     frames per teacher, portrait first)
    face-identity    POD (kind `avatar`, dedicated worker, no shared claim queue): SMIRK + landmark refinement + robust
                     combination (dancecap.face_identity.fit_images) per teacher -> face_shape/<role>_identity.npz, then the
                     displacement retargeted to the PINNED-betas rest head -> face_shape/<role>_identity_disp.npz (the
                     export's cache: no FLAME needed at export time)
    hair-portrait    portrait hair colour / visible length (hair_reference.portrait_reference, MediaPipe) -> avatar/hair_ref/
    body-shape       the pinned betas -> rest height vs the stated statures, waist, feet, shoe room; the shape_edits that
                     the take TOML would apply, with before / after numbers -> avatar/shape/body_report.json
    head-review      renders of the rest heads (template, identity, identity + taper) beside the portraits -> review/avatar/

POSE-DEPENDENT stages (need the STITCHED poses smplx/{lead,follow}.npz of the take; `plan` shows what they read):
    texture-select   the window of the stitched poses that shows every side of both teachers to a good phone -> avatar/texture/windows.json
    texture-run      ONE POD JOB (a 4090): frames + cameras + people + SAM 2.1 masks + hair masks of the window (the farm's chunk jobs do not
                     return frames or masks), then dancecap.avatar_texture preset pod (head fix, hair-aware albedo)
    texture-install  the bake into the take's own avatar/
    face-texture     local: dancecap.avatar_face (portrait faces composited onto the bake)
    hair-groom       local CPU: dancecap.hair_groom <take> --from-take larissa_kadu_full (seeded from the demo's groom, re-based
                     onto the recap's rest model)
    shoes-seed       (pose-independent) the demo's sneaker style / colours, until `dancecap.shoes` re-measures on the capture
    export           dancecap.demo_export larissa_kadu_recap --capture 06_LarissaKaduRecap (the export agent)

    texture-smoke    DRY RUN of the texture-run on the shape pass's own window (real recap data, a real fit): proves the job end to end

The pod job design is the farm's (dancecap.farm.worker): ONE job spec of kind `avatar` (a kind no farm worker claims: the fit
pods run --kinds front,fit,refine,full,asr), id `ra-<stage>-<n>`, queued on the volume, ONE pod started by this module with
HM_KINDS=avatar, results pulled, the pod deleted in a `finally`. The ledger owner is `A1-recap-avatars`.
Likeness data (portraits, face frames, identity files) stays under work/ (git-ignored) and on the owner's private volume.
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

from .take import PACKAGE_ROOT, WORKSPACE

TAKE = "larissa_kadu_recap"
DEMO = "larissa_kadu_full"
OWNER = "A1-recap-avatars"
KIND = "avatar"
ROLES = ("lead", "follow")
CAMS = ("00", "01", "02", "03", "05", "06")
FACE_CAMS = ("01", "06")           # the only phones whose teacher faces are bigger than ~15 px between the eyes
WORK = PACKAGE_ROOT / "work" / TAKE
AV = WORK / "avatar"
FACE_ID = AV / "face_id"
FACE_SHAPE = WORK / "face_shape"   # work/<take>/face_shape/: the place face_identity.cached_displacement reads and writes
SHAPE_WORK = PACKAGE_ROOT / "work" / "larissa_kadu_recap_shape"
PIN = SHAPE_WORK / "mamma_pin_betas.json"
REL = f"dancecap/work/{TAKE}"
PRICE_CPU_8 = 0.28                 # cpu5c x 8 vCPU, USD per hour (farm.pods.CPU_FLAVORS)
PRICE_4090 = 0.89


def log(msg: str) -> None:
    print(f"[recap_avatars {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------------------------------------ the stage table

@dataclass(frozen=True)
class Stage:
    name: str
    poses: bool            # needs the stitched poses (smplx/)
    where: str             # local | pod
    cmd: str
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    cost: str
    what: str


STAGES: tuple[Stage, ...] = (
    Stage("face-frames", False, "local", "face-frames", ("people/teachers_{01,06}.npz", "cameras/{01,06}.npz", "source videos"),
          ("avatar/face_id/frames/*.jpg", "avatar/face_id/candidates.json"), "0 (ffmpeg, ~4 min: the 817 MB HEVC phone seeks slowly)",
          "clearest frontal faces of each teacher in the two 1080 x 1920 phones"),
    Stage("face-landmarks", False, "local", "face-landmarks [--exclude NAMES]", ("avatar/face_id/candidates.json", "portraits"),
          ("avatar/face_id/landmarks_{lead,follow}.json", "review/avatar/face_frames_{lead,follow}.jpg"), "0 (.venv-face, ~5 min on the busy PC)",
          "MediaPipe + RTMW landmarks, the best frames per teacher (LOOK at the contact sheets, exclude hands / headset / wrong person)"),
    Stage("face-identity", False, "pod", "face-identity [--upload-models] [--max-cost 1.0]",
          ("avatar/face_id/landmarks_*.json + frames", "portraits", "larissa_kadu_recap_shape/mamma_pin_betas.json", "volume: models/flame, models/smirk"),
          ("face_shape/{lead,follow}_identity.npz", "face_shape/{lead,follow}_identity_disp.npz", "avatar/face_id/fit_report.json"),
          "~0.05 (CPU pod 8 vCPU, ~6 min) + one-off model upload (volume, free)", "FLAME identity per teacher + displacement retargeted to the pinned betas"),
    Stage("hair-portrait", False, "local", "hair-portrait", ("portrait of the follow",), ("avatar/hair_ref/portrait_reference.json",),
          "0 (~15 s)", "portrait hair colour and visible length (identical to the demo's: the same portrait)"),
    Stage("body-shape", False, "local", "body-shape", ("pinned betas", "takes/larissa_kadu_recap.toml"),
          ("avatar/shape/body_report.json",), "0 (~10 s)", "stature / proportions of the new betas, the body shape_edits before / after"),
    Stage("head-review", False, "local", "head-review", ("face_shape/*", "pinned betas", "portraits"), ("review/avatar/head_{lead,follow}.jpg",),
          "0 (~3 min)", "front / 3-4 / profile renders of the heads beside the portraits"),
    Stage("shoes-seed", False, "local", "shoes-seed", ("demo avatar/shoes.json",), ("avatar/shoes.json",), "0",
          "sneaker style / colours of the demo (same sneakers: see AVATAR_PLAN.md, the owner confirms)"),
    Stage("texture-select", True, "local", "texture-select [--length 60] [--max-windows 1]", ("smplx/{lead,follow}.npz", "people/coverage.json", "cameras/*_world.npz"),
          ("avatar/texture/windows.json",), "0", "the window of the stitched poses that shows every side of both teachers to a good phone"),
    Stage("texture-run", True, "pod", "texture-run [--window-id tx1] [--max-cost 3]",
          ("windows.json", "sliced stitched smplx", "sync offsets of the chunk", "volume: recap_in.tar + the source videos"),
          ("work/larissa_kadu_recap_<window>/avatar/*  (albedo, coverage, head_fix, hair region, report) + review sheets",), "~0.4 (one 4090 pod, ~28 min, a 46 s window; measured)",
          "front (frames, cameras, people, SAM 2.1 + hair masks: the farm's chunk jobs return neither) then the hair-aware bake, ONE pod job"),
    Stage("texture-install", True, "local", "texture-install [--window-id tx1]", ("work/larissa_kadu_recap_tx1/avatar/*",), ("avatar/{lead,follow}_albedo.png",),
          "0", "the bake into the take's own avatar/ (where export, face-texture and the groom read it)"),
    Stage("texture-fuse", True, "local", "texture-fuse --windows tx4,tx3,tx5", ("work/larissa_kadu_recap_<window>/avatar/*_texels.npz of 2-3 windows", "avatar/paint_out.json", "the demo's albedo (donor)"),
          ("avatar/{lead,follow}_albedo.png", "avatar/{lead,follow}_coverage.png", "avatar/fuse_report.json", "avatar/TEXTURE_FROM.json"), "0 (~1 min)",
          "per-texel confidence-weighted, outlier-aware fusion of several window bakes + paint-out of the headset / belt pack (recap_avatars_fuse)"),
    Stage("face-texture", True, "local", "face-texture [--composite]", ("avatar/{lead,follow}_albedo.png", "smplx/*.npz (betas)", "portraits"),
          ("avatar/{lead,follow}_albedo_face.png",), "0 (~9 min CPU on the shared PC)", "dancecap.avatar_face: the portrait faces composited onto the bake"),
    Stage("texture-paint", True, "local", "texture-paint --roles lead,follow", ("avatar/{lead,follow}_albedo_face.png", "avatar/paint_out.json (stage final regions)"),
          ("avatar/{lead,follow}_albedo_face.png",), "0 (seconds)", "the paint-out regions marked stage final (her cloth's dirt lift) on the face-composited albedos"),
    Stage("hair-groom", True, "local", "hair-groom", ("smplx/follow.npz", "avatar/follow_albedo_face.png", "the demo's groom"),
          ("avatar/hair_groom.json",), "0 (~3 min CPU)", "seeded from the demo's groom (same hair), re-based onto the recap's rest model"),
    Stage("export", True, "local", "(export) python -m dancecap.recap_export larissa_kadu_recap --capture 06_LarissaKaduRecap --avatars",
          ("everything above", "[dancers.*.shape_edits] of the take TOML"), ("StreamingAssets/06_LarissaKaduRecap/",), "0", "the Unity capture"),
)


# ------------------------------------------------------------------------------------------------ take config

def take_cfg() -> dict:
    p = PACKAGE_ROOT / "takes" / f"{TAKE}.toml"
    return tomllib.loads(p.read_text(encoding="utf-8"))


def source_dir() -> Path:
    p = PACKAGE_ROOT / "takes" / f"{TAKE}.toml"
    return (p.parent / take_cfg()["source_dir"]).resolve()


def portrait_path(role: str) -> Path:
    """<source_dir>/<name lower>.jpg (the owner's portraits; likeness data, never copied into a repo)."""
    d = take_cfg()["dancers"][role]
    p = source_dir() / (d.get("portrait") or f"{d['name'].lower()}.jpg")
    if not p.exists():
        raise FileNotFoundError(f"portrait of {d['name']}: {p} missing")
    return p


def pinned_betas() -> dict[str, np.ndarray]:
    """{role: betas (16,)} of the shape pass (MAMMA body id 0 = lead, 1 = follow)."""
    d = json.loads(PIN.read_text())
    return {"lead": np.asarray(d["0"], float), "follow": np.asarray(d["1"], float)}


def _below_normal() -> int:
    return 0x00004000 if os.name == "nt" else 0       # BELOW_NORMAL_PRIORITY_CLASS


def run_low(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    """A local child at below-normal priority (the workstation's 4 CPUs are shared)."""
    return subprocess.run(cmd, capture_output=True, text=True, creationflags=_below_normal(), **kw)


# ------------------------------------------------------------------------------------------------ face frames (pure parts)

# COCO keypoints: 0 nose, 1 left eye, 2 right eye, 3 left ear, 4 right ear
def face_scores(kp: np.ndarray, conf: np.ndarray, merged: np.ndarray, kp_min: float = 0.5, conf_min: float = 0.5,
                max_offset: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    """(iod, ok) per frame of one teacher. kp (F, 17, 3) [x, y, score]; iod = distance between the eye keypoints in px (the face
    size); ok = nose and both eyes seen (score >= kp_min), the person's track confident (>= conf_min) and not a merged
    two-body detection, and frontal: the nose at most max_offset x iod from the midpoint of the eyes."""
    iod = np.hypot(kp[:, 1, 0] - kp[:, 2, 0], kp[:, 1, 1] - kp[:, 2, 1])
    seen = (kp[:, 0, 2] > kp_min) & (kp[:, 1, 2] > kp_min) & (kp[:, 2, 2] > kp_min)
    off = np.abs(kp[:, 0, 0] - 0.5 * (kp[:, 1, 0] + kp[:, 2, 0])) / np.maximum(iod, 1e-6)
    ok = seen & (conf >= conf_min) & ~merged.astype(bool) & (off <= max_offset) & np.isfinite(iod)
    return np.where(ok, iod, 0.0), ok


def pick_frames(iod: np.ndarray, t: np.ndarray, n: int, min_gap_s: float, iod_min: float) -> list[int]:
    """Frame indices of the n largest faces, at least min_gap_s apart (greedy by size)."""
    out: list[int] = []
    for i in np.argsort(-iod):
        if iod[i] < iod_min or len(out) >= n:
            break
        if all(abs(t[i] - t[j]) >= min_gap_s for j in out):
            out.append(int(i))
    return out


def face_box(kp: np.ndarray, scale: float = 5.5) -> list[float]:
    """[centre x, centre y, side] (px) of a square around the face from the eye / nose keypoints: scale x the eye distance."""
    pts = kp[[0, 1, 2], :2]
    iod = float(np.hypot(*(kp[1, :2] - kp[2, :2])))
    c = pts.mean(0)
    return [float(c[0]), float(c[1]), scale * iod]


def select_faces(records: list[dict], cand: dict[str, dict], n: int = 10, yaw_max: float = 38.0, min_gap_s: float = 3.0,
                 occ_min: float = 14.0, cams: tuple[str, ...] | None = None, exclude: tuple[str, ...] = ()) -> list[dict]:
    """The best landmark records of a teacher's video frames: ok, a clear face (outer-canthal px >= occ_min), the head turned at
    most yaw_max degrees, only the cameras `cams` (None = all) and not the names in `exclude` (frames a human rejected: a hand
    or the headset over the face, the other teacher), ranked by sharpness x size, at least min_gap_s apart in the same camera.
    (Sharpness is a Laplacian variance of the landmark box: it grows as a face shrinks and noise is upscaled, so mix cameras
    of different resolution only on purpose.)"""
    pool = [r for r in records if r.get("ok") and r["name"] in cand and abs(r["yaw"]) <= yaw_max and r["occ_px"] >= occ_min
            and (cams is None or cand[r["name"]]["cam"] in cams) and r["name"] not in exclude]
    pool.sort(key=lambda r: -r["sharpness"] * r["occ_px"])
    out: list[dict] = []
    for r in pool:
        c = cand[r["name"]]
        if all(not (cand[o["name"]]["cam"] == c["cam"] and abs(cand[o["name"]]["t"] - c["t"]) < min_gap_s) for o in out):
            out.append(r)
        if len(out) >= n:
            break
    return out


# ------------------------------------------------------------------------------------------------ stage: face-frames

def source_video(cam: str) -> Path:
    cams = json.loads((WORK / "cameras" / "recap_cams.json").read_text())["cams"]
    return source_dir() / cams[cam]["file"]


def extract_frame(video: Path, pts: float, out: Path) -> bool:
    out.parent.mkdir(parents=True, exist_ok=True)
    r = run_low(["ffmpeg", "-y", "-v", "error", "-ss", f"{max(pts, 0.0):.4f}", "-i", str(video), "-frames:v", "1", "-q:v", "2", str(out)])
    return r.returncode == 0 and out.exists() and out.stat().st_size > 0


def stage_face_frames(per_cam: int = 14, min_gap_s: float = 3.0) -> dict:
    """Candidate faces per teacher in cameras 01 and 06 (the others show faces of 9-15 px between the eyes)."""
    cand: dict[str, dict] = {}
    for cam in FACE_CAMS:
        z = np.load(WORK / "people" / f"teachers_{cam}.npz")
        c = np.load(WORK / "cameras" / f"{cam}.npz")
        for r, role in enumerate(ROLES):
            iod, ok = face_scores(z["kp"][:, r], z["conf"][:, r], z["merged"][:, r])
            sel = pick_frames(iod, z["t"], per_cam, min_gap_s, iod_min=0.6 * float(np.percentile(iod[ok], 95)) if ok.any() else 1e9)
            log(f"{role} cam {cam}: {int(ok.sum())} usable frames, eye distance up to {iod.max():.0f} px, {len(sel)} candidates")
            for i in sel:
                name = f"{role}_{cam}_{int(z['frame_index'][i]):06d}"
                out = FACE_ID / "frames" / f"{name}.jpg"
                if not out.exists() and not extract_frame(source_video(cam), float(c["pts"][i]), out):
                    log(f"  {name}: ffmpeg failed")
                    continue
                cand[name] = {"role": role, "cam": cam, "canonical": int(i), "frame_index": int(z["frame_index"][i]),
                              "t": float(z["t"][i]), "pts": float(c["pts"][i]), "iod_px": float(iod[i]),
                              "face": face_box(z["kp"][i, r]), "file": f"frames/{name}.jpg"}
    FACE_ID.mkdir(parents=True, exist_ok=True)
    (FACE_ID / "candidates.json").write_text(json.dumps(cand, indent=1))
    return {"candidates": len(cand), "per_role": {r: sum(1 for v in cand.values() if v["role"] == r) for r in ROLES}}


# ------------------------------------------------------------------------------------------------ stage: face-landmarks

def face_python() -> str:
    from .hair_reference import face_python as fp
    exe, _ = fp()
    if exe is None:
        raise RuntimeError(".venv-face (mediapipe + rtmlib) is missing: python -m venv .venv-face; pip install mediapipe==1.1.0 rtmlib onnxruntime")
    return exe


def stage_face_landmarks(n_frames: int = 10, cams: tuple[str, ...] = ("01",), exclude: tuple[str, ...] = (),
                         reuse: bool = False) -> dict:
    """Detect landmarks on the portraits + every candidate frame (skipped with reuse when landmarks_all.json exists), then
    choose the fit frames per teacher (select_faces) and write landmarks_<role>.json + the contact sheets."""
    cand = json.loads((FACE_ID / "candidates.json").read_text())
    w = PACKAGE_ROOT / "weights" / "face"
    allp = FACE_ID / "landmarks_all.json"
    if not (reuse and allp.exists()):
        images = [{"name": f"portrait_{role}", "file": str(portrait_path(role)), "face": None} for role in ROLES]
        images += [{"name": n, "file": str(FACE_ID / v["file"]), "face": v["face"]} for n, v in cand.items()]
        jobs = FACE_ID / "landmark_jobs.json"
        jobs.write_text(json.dumps({"mp_task": str(w / "face_landmarker.task"),
                                    "rtmw_onnx": str(w / "rtmw_x384" / "end2end.onnx"), "images": images}))
        r = run_low([face_python(), str(Path(__file__).with_name("recap_face_lm.py")), "--jobs", str(jobs), "--out", str(allp)])
        if r.returncode != 0:
            raise RuntimeError("landmarks failed:\n" + r.stderr[-2000:])
    recs = json.loads(allp.read_text())
    rep: dict = {}
    for role in ROLES:
        mine = [x for x in recs if x["name"] in cand and cand[x["name"]]["role"] == role]
        best = select_faces(mine, cand, n_frames, cams=cams, exclude=exclude)
        por = next(x for x in recs if x["name"] == f"portrait_{role}")
        if not por["ok"]:
            raise RuntimeError(f"no face found in the portrait of the {role}")
        out = [{**por, "name": "portrait", "file": rel_to_work(por["file"], role)}] + \
              [{**x, "file": rel_to_work(x["file"], role)} for x in best]
        (FACE_ID / f"landmarks_{role}.json").write_text(json.dumps(out, indent=1))
        rep[role] = {"detected": sum(1 for x in mine if x["ok"]), "of": len(mine), "chosen": [x["name"] for x in best],
                     "portrait_yaw": round(por["yaw"], 1), "portrait_occ_px": round(por["occ_px"], 1)}
        contact_sheet(role, best)
    (FACE_ID / "landmarks_report.json").write_text(json.dumps(rep, indent=1))
    return rep


def rel_to_work(file: str, role: str) -> str:
    """File names in landmarks_<role>.json are relative to the take's work dir ('avatar/face_id/frames/x.jpg', portraits as
    'avatar/face_id/portrait_<role>.<ext>': the job tar carries a copy), so the same file works locally and on the pod."""
    p = Path(file)
    try:
        return p.resolve().relative_to(WORK).as_posix()
    except ValueError:
        return f"avatar/face_id/portrait_{role}{p.suffix}"


def contact_sheet(role: str, recs: list[dict], size: int = 224) -> Path:
    """review/avatar/face_frames_<role>.jpg: the portrait and the chosen video faces as the SMIRK crops (224 px)."""
    import cv2
    from .face_smirk import warp_crop
    tiles = []
    for r in [{"name": "portrait", "file": str(portrait_path(role)), **_rec_of(role, "portrait")}] + recs:
        img = cv2.imread(r["file"] if Path(r["file"]).is_absolute() else str(WORK / r["file"]))
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


def _rec_of(role: str, name: str) -> dict:
    recs = json.loads((FACE_ID / "landmarks_all.json").read_text())
    r = next(x for x in recs if x["name"] == f"{name}_{role}")
    return {"smirk_center": r["smirk_center"], "smirk_side": r["smirk_side"]}


# ------------------------------------------------------------------------------------------------ stage: hair-portrait

def stage_hair_portrait(albedo: float = 0.8) -> dict:
    from . import hair_reference as hr
    out = AV / "hair_ref"
    out.mkdir(parents=True, exist_ok=True)
    res = hr.portrait_reference(portrait_path("follow"), out, albedo)
    res.pop("_masks", None)
    (out / "portrait_reference.json").write_text(json.dumps(res, indent=1, default=float))
    return {k: res.get(k) for k in ("status", "hair_px", "visible_length_head_heights")} | \
        {"palette": {k: v.get("hex") for k, v in (res.get("palette") or {}).items() if isinstance(v, dict)}}


# ------------------------------------------------------------------------------------------------ stage: body-shape

def demo_betas() -> dict[str, np.ndarray]:
    d = json.loads((PACKAGE_ROOT / "work" / DEMO / "mamma_pin_betas.json").read_text())
    return {"lead": np.asarray(d["0"], float), "follow": np.asarray(d["1"], float)}


def rest_skin(betas: np.ndarray):
    """The unedited rest skin of a pinned-betas body (the exporter's shaped_skin on MAMMA's locked-head model)."""
    from .export_unity import model_for, shaped_skin
    mp = model_for({"model": "smplx_locked_head"})
    return shaped_skin(mp, betas), mp


def body_numbers(sk, stature: float, sole: float) -> dict:
    from . import shape_edits as se
    v = sk.vertices
    h = float(v[:, 1].max() - v[:, 1].min())
    return {"rest_height_m": round(h, 4), "stature_plus_sole_m": round(stature + sole, 4), "ratio_to_stature": round(h / (stature + sole), 4),
            "shoulder_joint_width_m": round(float(np.linalg.norm(sk.joints[16] - sk.joints[17])), 4),
            "foot_length_m": se.foot_lengths(v, sk.joints, sk.weights), "midsection": se.midsection_dims(v, sk.joints, sk.weights, sk.faces),
            "face": {k: x for k, x in se.face_measures(v).items() if k != "ratios"}}


def proposed_edits(role: str) -> tuple[dict, str]:
    """(shape_edits table, source) the avatar of this role gets: the recap TOML's own table when it has one, else the demo's table
    (what the owner approved for the same people: midsection 12.5 % thinner, narrower / lower sneakers, the face block)."""
    own = ((take_cfg()["dancers"].get(role) or {}).get("shape_edits"))
    if own:
        return json.loads(json.dumps(own)), f"takes/{TAKE}.toml"
    demo = tomllib.loads((PACKAGE_ROOT / "takes" / f"{DEMO}.toml").read_text(encoding="utf-8"))["dancers"][role].get("shape_edits") or {}
    return json.loads(json.dumps(demo)), f"takes/{DEMO}.toml (carry-over, not in the recap TOML)"


def stage_body_shape() -> dict:
    """The new betas against the stated statures and the demo's betas, and the body shape_edits the avatars get on them (before / after)."""
    from . import shape_edits as se
    cfg = take_cfg()
    bn, dn = pinned_betas(), demo_betas()
    out: dict = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "betas": "work/larissa_kadu_recap_shape/mamma_pin_betas.json",
                 "gauge": "rest height = crown to sole of the shaped T-pose; target = stature_m + shoe_sole_m (the demo's world-scale check, tolerance 1.5 %)",
                 "roles": {}}
    for role in ROLES:
        d = cfg["dancers"][role]
        stature, sole = float(d["stature_m"]), float(d.get("shoe_sole_m", 0.025))
        sk, _ = rest_skin(bn[role])
        sk_d, _ = rest_skin(dn[role])
        rec = {"name": d["name"], "recap": body_numbers(sk, stature, sole), "demo": body_numbers(sk_d, stature, sole),
               "beta_distance_to_demo": round(float(np.linalg.norm(bn[role] - dn[role])), 3),
               "beta_norm": {"recap": round(float(np.linalg.norm(bn[role])), 3), "demo": round(float(np.linalg.norm(dn[role])), 3)}}
        edits, src = proposed_edits(role)
        rec["edits_source"] = src
        body_cfg = {k: v for k, v in edits.items() if k != "face"}
        if body_cfg:
            v, rep = se.apply_vertices(body_cfg, sk.vertices, sk.joints, sk.weights, sk.faces, role, WORK)
            rec["edits_body"] = {"config": rep["config"], "moved_vertices": rep["moved_vertices"], "max_move_mm": rep["max_move_mm"],
                                 **{k: rep[k] for k in ("midsection", "feet") if k in rep}}
            rec["edits_body"]["after"] = {"foot_length_m": se.foot_lengths(v, sk.joints, sk.weights),
                                          "midsection": se.midsection_dims(v, sk.joints, sk.weights, sk.faces)}
        out["roles"][role] = rec
        log(f"{d['name']}: rest height {rec['recap']['rest_height_m']} m (target {rec['recap']['stature_plus_sole_m']}, ratio "
            f"{rec['recap']['ratio_to_stature']}; the demo's betas: {rec['demo']['ratio_to_stature']})")
    p = AV / "shape" / "body_report.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1, default=float))
    return out


# ------------------------------------------------------------------------------------------------ stage: head-review

def head_variants(role: str) -> dict[str, dict | None]:
    """name -> face table of shape_edits (None = the plain template head). The candidates the take TOML can carry: the identity fits of the
    pod (mixed = the shipped one; portrait only / video only = the two sources apart) and, for the follow, the demo's petite taper on the
    recap's identity next to the demo's own block."""
    v: dict[str, dict | None] = {"template (new betas)": None}
    rel = lambda n: f"work/{TAKE}/face_shape/{role}_identity{n}.npz"   # noqa: E731
    for name, suffix in (("portrait only", "_portrait"), ("video only", "_video"), ("identity (mixed)", "")):
        if (PACKAGE_ROOT / rel(suffix)).exists():
            v[name] = {"identity": rel(suffix), "identity_weight": 1.0, "identity_mouth": 0.4}
    if role == "follow":
        demo = tomllib.loads((PACKAGE_ROOT / "takes" / f"{DEMO}.toml").read_text(encoding="utf-8"))["dancers"]["follow"]["shape_edits"]["face"]
        if (PACKAGE_ROOT / rel("")).exists():
            v["recap identity + demo taper"] = {**demo, "identity": rel("")}
        v["demo block (demo identity)"] = demo
    own = ((take_cfg()["dancers"].get(role) or {}).get("shape_edits") or {}).get("face")
    if own:
        v["TOML face block"] = own
    return v


def stage_head_review() -> dict:
    import cv2

    from . import recap_avatars_review as rv
    from . import shape_edits as se
    bn = pinned_betas()
    rep: dict = {}
    scratch = AV / "review_scratch"            # a separate work dir: face_identity's per-role displacement cache is single-file
    for role in ROLES:
        sk, mp = rest_skin(bn[role])
        lm = {r["name"]: r for r in json.loads((FACE_ID / f"landmarks_{role}.json").read_text())}
        por = lm["portrait"]
        alb = rv.load_albedo(PACKAGE_ROOT / "work" / DEMO / "avatar" / f"{role}_albedo_face.png")
        yaw = float(por["yaw"])
        rows_tex, rows_grey, numbers = [], [], {}
        for name, face in head_variants(role).items():
            if face is None:
                v = sk.vertices
                numbers[name] = {"after": {k: x for k, x in se.face_measures(v).items() if k != "ratios"}}
            else:
                same = str(face.get("identity", "")).endswith(f"{role}_identity.npz") and TAKE in str(face.get("identity", ""))
                work = WORK if same else scratch
                v, r = se.apply_vertices({"face": face}, sk.vertices, sk.joints, sk.weights, sk.faces, role, work)
                numbers[name] = {"after": {k: x for k, x in r["face"]["after"].items() if k != "ratios"}, "max_move_mm": r["max_move_mm"],
                                 "protected_flame_max_move_mm": r["face"].get("protected_flame_max_move_mm")}
            rows_tex.append((name, rv.head_tiles(sk, v, mp, alb, yaw)))
            rows_grey.append((name, rv.head_tiles(sk, v, mp, None, yaw)))
            log(f"{role}: {name} rendered")
        pt = rv.portrait_tile(portrait_path(role), por)
        out = WORK / "review" / "avatar"
        out.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / f"head_{role}.jpg"), rv.sheet(rows_tex, pt), [cv2.IMWRITE_JPEG_QUALITY, 90])
        cv2.imwrite(str(out / f"head_{role}_grey.jpg"), rv.sheet(rows_grey, pt), [cv2.IMWRITE_JPEG_QUALITY, 90])
        rep[role] = numbers
    rv.write_report(AV / "shape" / "head_review.json", rep)
    return {r: list(v) for r, v in rep.items()}


# ------------------------------------------------------------------------------------------------ stages that wrap the demo's modules

def require(paths: list[Path], what: str) -> None:
    miss = [str(p.relative_to(PACKAGE_ROOT)) for p in paths if not p.exists()]
    if miss:
        raise FileNotFoundError(f"{what} needs {miss} (see `plan`: these come from the stitched poses / the texture job)")


def run_module(module: str, *args: str) -> dict:
    """python -m <module> <args> in this interpreter, below-normal priority, from dancecap/ (the demo's stages take the take name)."""
    r = run_low([sys.executable, "-m", module, *args], cwd=PACKAGE_ROOT)
    tail = (r.stdout + r.stderr)[-3000:]
    if r.returncode != 0:
        raise RuntimeError(f"{module} {' '.join(args)} failed (rc {r.returncode}): {tail}")
    return {"module": module, "args": list(args), "tail": tail[-800:]}


def stage_shoes_seed(from_take: str = "larissa_kadu_38") -> dict:
    """The demo's sneaker style / colours as this take's avatar/shoes.json (same sneakers: the outfits match in the recap's frames, see
    AVATAR_PLAN.md; `dancecap.shoes` re-measures once the capture exists). demo_export seeds from a sibling take of the SAME source folder
    only, and the recap has its own folder, so this is the explicit seed. The sole stack (heel / fore) comes from the take TOML's
    shoe_geometry at export, never from this file."""
    src = PACKAGE_ROOT / "work" / from_take / "avatar" / "shoes.json"
    dst = AV / "shoes.json"
    if dst.exists():
        return {"seeded": False, "why": f"{dst} exists"}
    s = json.loads(src.read_text(encoding="utf-8"))
    s["take"] = TAKE
    s["seeded_from"] = {"take": from_take, "note": "same sneakers (outfits match); sole stack from this take's toml shoe_geometry at export; "
                                                   "re-measure with dancecap.shoes once the capture exists"}
    ovp = AV / "shoes_override.json"      # the owner's decisions (2026-10-09: Larissa white, Kadu black) apply to the seed too
    if ovp.exists():
        from . import shoes as _shoes
        _shoes.apply_override(s, json.loads(ovp.read_text(encoding="utf-8")))
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(s, indent=1), encoding="utf-8")
    return {"seeded": True, "from": from_take, "path": str(dst)}


def stand_in_take(name: str):
    """A Take-like object for a work dir that has no take TOML here (a texture window's pseudo-take lives on the pod): what
    dancecap.avatar_face reads of a take = name, work, cfg (dancers -> portrait), sources[0].path.parent (the source folder)."""
    from types import SimpleNamespace
    return SimpleNamespace(name=name, work=PACKAGE_ROOT / "work" / name, cfg=take_cfg(), sources=[SimpleNamespace(path=source_dir() / "video.mp4")])


def stage_face_texture(work_take: str = TAKE, composite_only: bool = False) -> dict:
    """dancecap.avatar_face on `work_take` (default the take; a texture window's pseudo-take for a dry run): the portrait fit + UV projection +
    composite onto the bake -> avatar/<role>_albedo_face.png (+ avatar/face/*, review/avatar_face_*.jpg). Runs as a below-normal child."""
    wk = PACKAGE_ROOT / "work" / work_take
    require([wk / "avatar" / f"{r}_albedo.png" for r in ROLES] + [wk / "smplx" / f"{r}.npz" for r in ROLES], "face-texture")
    return run_module("dancecap.recap_avatars", "_face-texture-inproc", "--work-take", work_take, *(["--composite"] if composite_only else []))


def face_texture_inproc(work_take: str, composite_only: bool = False) -> None:
    from . import avatar_face
    take = stand_in_take(work_take)
    rep = avatar_face.run(take, ROLES, composite_only=composite_only, review=True)
    print(json.dumps({r: v.get("outputs") for r, v in rep["roles"].items()}, indent=1, default=str))


def stage_hair_groom(from_take: str = DEMO) -> dict:
    """dancecap.hair_groom <take> --from-take <demo>: the demo's groom (same hair) re-based onto the recap's rest model (roots, head,
    colliders, lengths x rest-height ratio), restyled; the cap's hairline is measured on avatar/follow_albedo_face.png."""
    require([WORK / "smplx" / "follow.npz", AV / "follow_albedo_face.png"], "hair-groom")
    res = run_module("dancecap.hair_groom", TAKE, "--from-take", from_take, "--no-export")
    res["portrait_highlight"] = carry_portrait_highlight(from_take)
    return res


def carry_portrait_highlight(from_take: str = DEMO) -> str:
    """hair_groom.apply_style rewrites render.kajiya_kay from its defaults and re-adds highlight_portrait_linear (the portrait's gloss hue, which
    Unity reads when the hair colour source is `portrait`) only from a hair_reference.json: the demo take has none, so a groom seeded from it loses
    the key and the hair renders with the default gloss (a brighter, flatter red than the demo). Same portrait file, same palette (A1: hair-portrait
    gives the demo's colours), so the demo's value is carried over. -> what was done."""
    dst = AV / "hair_groom.json"
    src = PACKAGE_ROOT / "work" / from_take / "avatar" / "hair_groom.json"
    if not (dst.exists() and src.exists()):
        return "skipped: a groom is missing"
    g = json.loads(dst.read_text(encoding="utf-8"))
    d = json.loads(src.read_text(encoding="utf-8"))
    hl = ((d.get("render") or {}).get("kajiya_kay") or {}).get("highlight_portrait_linear")
    kk = (g.setdefault("render", {})).setdefault("kajiya_kay", {})
    if hl is None or "highlight_portrait_linear" in kk:
        return "nothing to carry"
    kk["highlight_portrait_linear"] = hl
    kk["highlight_portrait_source"] = f"carried over from the {from_take} groom (same portrait, same palette)"
    dst.write_text(json.dumps(g, indent=1), encoding="utf-8")
    return "carried highlight_portrait_linear"


# ------------------------------------------------------------------------------------------------ plan

def exists_any(rel: str) -> bool:
    from glob import glob
    if any(c in rel for c in "{(") or rel.startswith("StreamingAssets"):
        return False
    return bool(glob(str(WORK / rel)))


def status_of(st: Stage) -> str:
    expanded = []
    for o in st.writes:
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
    doc = {"take": TAKE, "stitched_poses_present": smplx, "stages": rows}
    if as_json:
        print(json.dumps(doc, indent=1))
        return doc
    for s in rows:
        print(f"{s['name']:<16} {'POSES ' if s['poses'] else 'free  '} {s['where']:<5} {s['status']:<8} {s['cost']}")
        print(f"    {s['what']}")
        print(f"    cmd: python -m dancecap.recap_avatars {s['cmd']}")
    print(f"\nstitched poses (work/{TAKE}/smplx/*.npz): {'present' if smplx else 'NOT there yet: the pose stages wait'}")
    return doc


# ------------------------------------------------------------------------------------------------ main (CLI)

def csv(s: str) -> tuple[str, ...]:
    return tuple(x for x in s.split(",") if x)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.recap_avatars", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("face-frames")
    p.add_argument("--per-cam", type=int, default=14)
    p = sub.add_parser("face-landmarks")
    p.add_argument("--n-frames", type=int, default=10)
    p.add_argument("--cams", default="01", help="cameras the fit frames come from (comma list)")
    p.add_argument("--exclude", default="", help="comma list of candidate names a human rejected (hand / headset over the face, wrong person)")
    p.add_argument("--reuse", action="store_true", help="keep landmarks_all.json (only re-select)")
    p = sub.add_parser("face-identity")
    p.add_argument("--max-cost", type=float, default=1.0)
    p.add_argument("--vcpu", type=int, default=8)
    p.add_argument("--minutes", type=float, default=35)
    p.add_argument("--hard-minutes", type=float, default=60)
    p.add_argument("--job-id", default=None)
    p.add_argument("--upload-models", action="store_true", help="put FLAME + SMIRK on the volume first (once)")
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("texture-select", help="best window(s) of the stitched poses for the texture bake")
    p.add_argument("--smplx-dir", type=Path, default=WORK / "smplx")
    p.add_argument("--length", type=float, default=60.0)
    p.add_argument("--step", type=float, default=5.0)
    p.add_argument("--need", type=float, default=2.0, help="quality-weighted seconds of sight that fill a body sector")
    p.add_argument("--max-windows", type=int, default=1)
    p.add_argument("--chunks", action="store_true", help="BEFORE the stitch: the best window inside each pulled farm chunk (raw fits, no seam)")
    p.add_argument("--out", type=Path, default=AV / "texture" / "windows.json")
    p = sub.add_parser("texture-run", help="the REAL pose-dependent texture job on a window of the STITCHED poses (needs texture-select's windows.json)")
    p.add_argument("--windows", type=Path, default=AV / "texture" / "windows.json")
    p.add_argument("--which", type=int, default=0, help="index of the window in windows.json")
    p.add_argument("--window-id", default="tx1")
    p.add_argument("--smplx-dir", type=Path, default=WORK / "smplx")
    p.add_argument("--sync", type=Path, help="sync offsets file (default: the farm chunk whose core holds the window's middle)")
    p.add_argument("--top-k", type=int, default=6, help="samples per texel per camera (the demo's final bake: 6; the dry run used 4, peak GPU 3.8 GB)")
    p.add_argument("--body-ref-q", type=float, default=0.7, help="lighter luminance reference over her hair (the demo's final bake: 0.7; 0.5 = off)")
    p.add_argument("--with-hair-ref", action="store_true", help="also run hair_reference (3D hair hull) in the same pod session (not exercised on the recap yet)")
    p.add_argument("--step-ms", type=float, default=100.0)
    p.add_argument("--max-cost", type=float, default=3.0)
    p.add_argument("--minutes", type=float, default=90)
    p.add_argument("--hard-minutes", type=float, default=140)
    p.add_argument("--mask", default=None, help="cam:t0:t1[,cam:t0:t1] master seconds: frames of a camera whose track is wrong (the bake skips them); "
                                                "default the window's own mask list in windows.json")
    p.add_argument("--own-kind", action="store_true", help="the pod of this job takes only this job (kind avatar<window id>): several windows in parallel")
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("texture-fuse", help="fuse the bakes of several windows into one albedo per teacher (confidence-weighted, outlier-aware) and paint out the headset / belt pack")
    p.add_argument("--windows", default="tx3,tx4,tx5", help="comma list of window ids (work/larissa_kadu_recap_<id>/avatar/)")
    p.add_argument("--paint", type=Path, default=AV / "paint_out.json", help="regions to paint out (see recap_avatars_fuse)")
    p.add_argument("--out", type=Path, default=None, help="default the take's own avatar/ (use a scratch dir to compare)")
    p.add_argument("--roles", default="lead,follow")
    p.add_argument("--tau", type=float, default=0.10)
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("texture-paint", help="the paint-out regions marked stage final (paint_out.json) on the face-composited albedos, in place")
    p.add_argument("--roles", default="follow")
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("texture-install", help="copy a texture job's avatar/ outputs into the take's own avatar/ (what the export reads)")
    p.add_argument("--window-id", default="tx1")
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("texture-smoke", help="DRY RUN of the pose-dependent texture stages on the shape pass's window (real recap data, one 4090 pod)")
    p.add_argument("--chunk", default="shape")
    p.add_argument("--step-ms", type=float, default=100.0)
    p.add_argument("--max-cost", type=float, default=2.0)
    p.add_argument("--minutes", type=float, default=70)
    p.add_argument("--hard-minutes", type=float, default=100)
    p.add_argument("--job-id", default=None)
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("pod-texture-front", help="the pod-side front of the texture job (frames, cameras, people, SAM masks, merge)")
    p.add_argument("chunk", nargs="?", default="shape")
    p.add_argument("--resume", action="store_true")
    p = sub.add_parser("pod-mask-cams", help="the pod-side step: switch a camera off for a stretch of the window (cameras/<cam>.npz ok = False)")
    p.add_argument("chunk")
    p.add_argument("--mask", required=True)
    p = sub.add_parser("pod-identity", help="the pod-side step (also a local smoke test: --work <scratch dir> --iters 40 --limit 2)")
    p.add_argument("--work", type=Path)
    p.add_argument("--iters", type=int, default=600)
    p.add_argument("--limit", type=int)
    sub.add_parser("hair-portrait")
    sub.add_parser("shoes-seed")
    p = sub.add_parser("face-texture")
    p.add_argument("--work-take", default=TAKE, help="a work dir with avatar/ + smplx/ (default the take; larissa_kadu_recap_shape = the dry run)")
    p.add_argument("--composite", action="store_true", help="reuse the saved portrait fit / projection (avatar/face/): only composite onto a new albedo (seconds)")
    p = sub.add_parser("_face-texture-inproc")
    p.add_argument("--work-take", default=TAKE)
    p.add_argument("--composite", action="store_true")
    sub.add_parser("hair-groom")
    sub.add_parser("body-shape")
    sub.add_parser("head-review")
    a = ap.parse_args(argv)
    if a.cmd == "plan":
        plan(a.json)
    elif a.cmd == "face-frames":
        print(json.dumps(stage_face_frames(a.per_cam)))
    elif a.cmd == "face-landmarks":
        print(json.dumps(stage_face_landmarks(a.n_frames, csv(a.cams), csv(a.exclude), a.reuse), indent=1))
    elif a.cmd == "texture-select":
        from . import recap_avatars_select as sel
        if a.chunks:
            r = sel.run_chunks(WORK, min(a.length, 50.0), a.step, a.need, a.out)
            print(json.dumps({"best_chunk": r["chunk"], "window": r["windows"][0], "per_chunk": [{k: v for k, v in c.items() if k != "per_teacher"} for c in r["per_chunk"]]}, indent=1))
        else:
            r = sel.run(WORK, a.smplx_dir, a.length, a.step, a.need, a.max_windows, a.out)
            print(json.dumps({k: r[k] for k in ("windows", "combined", "second_best_adds", "t_range")}, indent=1))
    elif a.cmd == "texture-fuse":
        from . import recap_avatars_fuse as fz
        print(json.dumps(fz.run(WORK, list(csv(a.windows)), a.paint, a.out, csv(a.roles), dry_run=a.dry_run, tau=a.tau), indent=1, default=float))
    elif a.cmd == "texture-paint":
        from . import recap_avatars_fuse as fz
        print(json.dumps(fz.paint_final(WORK, roles=csv(a.roles), dry_run=a.dry_run), indent=1, default=float))
    elif a.cmd == "texture-install":
        from . import recap_avatars_pod as pod
        print(json.dumps(pod.texture_install(a.window_id, a.force), indent=1))
    elif a.cmd == "shoes-seed":
        print(json.dumps(stage_shoes_seed()))
    elif a.cmd == "face-texture":
        print(json.dumps(stage_face_texture(a.work_take, a.composite)))
    elif a.cmd == "_face-texture-inproc":
        face_texture_inproc(a.work_take, a.composite)
    elif a.cmd == "hair-groom":
        print(json.dumps(stage_hair_groom()))
    elif a.cmd == "hair-portrait":
        print(json.dumps(stage_hair_portrait(), indent=1))
    elif a.cmd == "body-shape":
        stage_body_shape()
    elif a.cmd == "head-review":
        print(json.dumps(stage_head_review(), indent=1))
    else:
        from . import recap_avatars_pod as pod
        if a.cmd == "pod-identity":
            pod.pod_identity(work=a.work, iters=a.iters, limit=a.limit)
            return 0
        if a.cmd == "pod-texture-front":
            return pod.pod_texture_front(a.chunk, a.resume)
        if a.cmd == "pod-mask-cams":
            return pod.pod_mask_cams(a.chunk, a.mask)
        if a.cmd == "texture-run":
            return pod.texture_run(a)
        if a.cmd == "texture-smoke":
            jid = a.job_id or f"ra-texture-{time.strftime('%m%d%H%M')}"
            spec = pod.texture_spec(jid, a.chunk, a.step_ms)
            log(f"texture dry run on chunk {a.chunk}: {len(spec['steps'])} steps, inputs {[i['path'] for i in spec['inputs']]}")
            if a.dry_run:
                print(json.dumps({"job": {k: spec[k] for k in ("id", "kind", "needs_gpu", "max_minutes")}, "steps": [s["cmd"] for s in spec["steps"]],
                                  "outputs": len(spec["outputs"])}, indent=1))
                return 0
            res = pod.run_job(spec, cpu=False, minutes=a.minutes, hard_minutes=a.hard_minutes, max_cost=a.max_cost)
            print(json.dumps(res, default=str))
            return 0 if res.get("status") == "done" else 1
        jid = a.job_id or f"ra-identity-{time.strftime('%m%d%H%M')}"
        spec = pod.identity_spec(jid)
        est = pod.estimate()
        log(f"estimate: {est['total_min']} min = {est['usd']} USD at {est['price_hr']}/h (ceiling {a.max_cost}); inputs {spec['inputs'][0]['bytes'] / 1e6:.1f} MB")
        if a.dry_run:
            print(json.dumps({"job": {k: spec[k] for k in ("id", "kind", "needs_gpu", "max_minutes")}, "steps": [s["cmd"][:200] for s in spec["steps"]],
                              "outputs": [o["path"] for o in spec["outputs"]], "estimate": est}, indent=1))
            return 0
        res = pod.run_job(spec, cpu=True, vcpu=a.vcpu, minutes=a.minutes, hard_minutes=a.hard_minutes, max_cost=a.max_cost,
                          upload_models=a.upload_models)
        print(json.dumps(res, default=str))
        return 0 if res.get("status") == "done" else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
