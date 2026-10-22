"""Per-dancer face swap of a recorded class-recap film with FaceFusion v4 (inswapper_128), driven by a face sidecar.

Owner, 2026-10-21 (v3 round, request 2): "if we keep track of Larissa and Kadu's face poses (literally the face points detected in
the unified 3d body poses) we could feed those to facefusion and use the inswapper_128 model individually on each person with their
included profile headshots. this would be for the video output. we could also track face pose to warp the splats".

How it works (docs/RECAP_FILM.md, section "Face swap of the recorded film"):
  * pass 1, analysis: every frame of the range is decoded once (ffmpeg, yuv420p); FaceFusion finds the faces (yolo_face on the whole
    frame and on two overlapping square tiles of a 9:16 frame, 2dfan4 68 landmarks, arcface embedding, fairface gender).
      - with a FACE SIDECAR (written by the Unity recorder, format below / in the doc) each dancer's projected 5 face points say WHO
        and WHERE; a detection is only used to refine the landmarks when it lies on the projected face (else the projected points,
        corrected by the running detector - projection offset, are used); yaw / pitch / visibility / UI rects come from the sidecar;
      - without one (the prototype) the identity comes from appearance: arcface similarity to the two headshots, gender, red hair
        around the face (Larissa), voted over a whole track; the shot kind (3D splats or a phone's video) from direction.json.
  * plan: per dancer track the 5 landmarks are gap-filled (<= 4 frames) and smoothed (Savitzky-Golay 7 / 2, zero phase); a weight per
    frame = gates (face size, yaw, pitch, visibility, shot kind; 0 behind a phone's frustum video surface, sidecar "surfaces") with
    hysteresis; dips under 0.4 s where the face is still a face are bridged, and a swap shows for at least 1 s per dancer or not at all
    (no flicker in fast turns); eroded and box-blurred so the swap fades in and out over 1/3 s INSIDE the valid span (it is gone on the
    frame the face stops qualifying);
  * pass 2, swap: inswapper_128 with the dancer's own headshot identity, per-track pixel boost (128 / 256 / 512 by face size),
    FaceFusion's box x occlusion (xseg) mask times the weight, pasted and blended in YUV only where the mask is > 1/255: every other
    pixel of the decoded frame goes to the encoder unchanged (bit-exact; text glyphs of UI overlays over a face are restored too,
    and so are the grey / blue skeleton strokes of the 3D overlays that cross a face: overlay_mask);
    the audio is copied (an excerpt: the same instant re-cut as AAC 192k); a whole film is remuxed afterwards so its video starts
    where the recording's does (0.033 s; restore_start).

Runs in the 'facefusion' conda env (Python 3.13, onnxruntime-gpu 1.30, CUDA 13), not in dancecap's env, from the dancecap root:
    conda run -n facefusion --no-capture-output python -m dancecap.film_faceswap swap --film F.mp4 --out O.mp4 \
        [--start S --end S] [--sidecar F.faces.json] [--direction work/larissa_kadu_recap/film/direction.json] [--side-by-side SBS.mp4]
    ... -m dancecap.film_faceswap canon-points      # face_points_canonical.json (5 / 52 face points per dancer, splat canonical pose)
    ... -m dancecap.film_faceswap mock-sidecar ...  # a sidecar made from a heuristic analysis (tests the sidecar code path)
    ... -m dancecap.film_faceswap verify --report R.json   # outside the swapped faces the output equals the source (lossless) / PSNR
    ... -m dancecap.film_faceswap sheet --report R.json    # before / after face crops of sampled frames, for looking
    ... -m dancecap.film_faceswap restore-start --report R.json   # a whole-film output whose video starts at 0: video start as recorded
The GPU passes hold dancecap's work/gpu.lock. Likeness data (headshots, films, sheets) stays in work/ and Recordings/ (git-ignored).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]                     # the dancecap repository
WORK = ROOT / "work" / "larissa_kadu_recap"
OUT_DIR = WORK / "v3" / "faceswap"
FACE_POINTS = OUT_DIR / "face_points_canonical.json"
DIRECTION = WORK / "film" / "direction.json"
FACEFUSION = Path(os.environ.get("FACEFUSION_ROOT", "C:/Users/johnb/Desktop/facefusion"))
HEADSHOTS = {"lead": Path("C:/Users/johnb/Videos/Zouk/larissa-kadu-counter-recap/kadu.jpg"),
             "follow": Path("C:/Users/johnb/Videos/Zouk/larissa-kadu-counter-recap/larissa.jpg")}
NAMES = {"lead": "Kadu", "follow": "Larissa"}
ROLES = ("lead", "follow")
SPLATS = Path("C:/Users/johnb/Desktop/HeadMovement/head-movement/Assets/StreamingAssets/06_LarissaKaduRecap/splats")
# subject's right eye first = the InsightFace / FaceFusion 5-point order of a face seen from the front
LANDMARKS5 = ("eye_r", "eye_l", "nose_tip", "mouth_r", "mouth_l")
SIDECAR_FORMAT = "hm film faces"
PHONE_MODES = ("camera_tour", "fly_to_camera")


@dataclass
class Cfg:
    facefusion: Path = FACEFUSION
    swapper_model: str = "inswapper_128"
    detector: str = "yolo_face"
    det_score: float = 0.5
    occluder: str = "xseg_1"
    mask_blur: float = 0.3
    providers: tuple = ("cuda",)
    # gates (smooth ramps: full weight inside the first value, zero beyond the second)
    iod_px: tuple = (10.0, 16.0)        # inter-ocular distance in pixels: below 10 no swap, full from 16
    yaw_deg: tuple = (45.0, 65.0)       # |yaw| vs the line of sight (inswapper holds to ~60 deg; gate on at ~55, off at ~59)
    pitch_deg: tuple = (30.0, 45.0)
    vis: tuple = (0.55, 0.8)            # sidecar visibility (share of the face points not hidden)
    aligner_min: float = 0.3            # 2dfan4 score below which a detection does not count (heuristic mode)
    fade_frames: int = 10               # 1/3 s at 30 fps
    # no flicker (v3.1, 2026-10-21: in fast turns the swap came and went in 0.4-0.8 s spans every ~2 s): a swap shows for at least
    # min_on_frames or not at all - counted per DANCER over all of its tracks - and a dip of the gate shorter than bridge_frames
    # where the face is still a face (gate >= gate_bridge) does not switch it off
    min_on_frames: int = 30             # 1 s at 30 fps (v3.0: min_span_frames 10)
    bridge_frames: int = 12             # 0.4 s
    gate_bridge: float = 0.05           # ... e.g. |yaw| up to ~64 deg, a detector-score dip, a hand passing; never the back of the head
    gate_on: float = 0.5                # hysteresis on the face gate: a face starts to qualify at 0.5 ...
    gate_off: float = 0.25              # ... and stops below 0.25 (so the weight is 0 / 1 with fades, never a long half blend)
    # the phones' frustum video surfaces in the 3D world (sidecar "surfaces", v3.1): a face behind one is never swapped
    surface_alpha: float = 0.05         # a surface more opaque than this hides what is behind it
    surface_box_iod: float = 1.0        # ... tested on the 5 points and the corners of the face box (centre +- this x inter-ocular)
    gap_fill: int = 4                   # frames a track may miss and be interpolated
    smooth_window: int = 7
    identity_min: float = 0.35          # |mean identity score| of a track below this: unknown, not swapped
    phone_shots: str = "off"            # off | swap (sidecar only): swap the two teachers on a phone's own video too
    mix_lead_s: float = 0.8             # heuristic: the 3D -> phone flight (fly_to_camera) and ...
    mix_after_phone_s: float = 0.7      # ... the phone -> 3D blend are left alone (the phone's video shows through)
    color_match: float = 0.0            # 0..1: move the swapped face's LAB mean / std toward the original's (0 = FaceFusion's look)
    crf: int = 14
    preset: str = "slow"
    lossless: bool = False


# ----------------------------------------------------------------------------------------------------------------------------- io
def ffprobe(path: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                          "stream=index,codec_type,width,height,r_frame_rate,nb_frames,start_time:format=start_time,duration",
                          "-of", "json", str(path)], capture_output=True, text=True, check=True).stdout
    j = json.loads(out)
    v = next(s for s in j["streams"] if s["codec_type"] == "video")
    num, den = (int(x) for x in v["r_frame_rate"].split("/"))
    return {"width": int(v["width"]), "height": int(v["height"]), "fps": num / den, "frames": int(v.get("nb_frames", 0)),
            "v_start": float(v.get("start_time", 0.0)), "fmt_start": float(j["format"].get("start_time", 0.0)),
            "duration": float(j["format"]["duration"]), "has_audio": any(s["codec_type"] == "audio" for s in j["streams"])}


def frame_seek(info: dict, first: int) -> float:
    """-ss value that makes frame `first` the first decoded frame (ffmpeg adds the file's start time to -ss)."""
    pts = info["v_start"] + first / info["fps"]
    return max(0.0, pts - 0.5 / info["fps"] - info["fmt_start"])


class YuvReader:
    """Frames [first, first + count) of a film as writable yuv420p buffers (one contiguous array per frame).

    `-fps_mode passthrough` (2026-10-21, first full-film run): without it ffmpeg's default constant-rate output for rawvideo fills
    the half-frame gap between the seek point and the first frame by DUPLICATING that frame, so buffer k held frame first + k - 1
    (the sidecar entry of frame i was applied to the picture of frame i - 1, and the output repeated its first frame and lost the
    last one). Passthrough hands over exactly the coded frames, in order."""

    def __init__(self, film: Path, info: dict, first: int, count: int):
        self.w, self.h, self.count = info["width"], info["height"], count
        self.size = self.w * self.h * 3 // 2
        cmd = ["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{frame_seek(info, first):.6f}", "-i", str(film), "-map", "0:v:0",
               "-fps_mode", "passthrough", "-frames:v", str(count), "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"]
        self.p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=self.size * 2)

    def __iter__(self):
        for _ in range(self.count):
            data = self.p.stdout.read(self.size)
            if len(data) < self.size:
                break
            yield np.frombuffer(data, np.uint8).copy()
        self.p.stdout.close()
        self.p.wait()


def planes(buf: np.ndarray, w: int, h: int):
    n = w * h
    return buf[:n].reshape(h, w), buf[n:n + n // 4].reshape(h // 2, w // 2), buf[n + n // 4:].reshape(h // 2, w // 2)


class Writer:
    """yuv420p frames -> H.264 (x264 CRF, or lossless qp 0) + the film's audio for the same instant."""

    def __init__(self, out: Path, film: Path, info: dict, first: int, count: int, w: int, h: int, cfg: Cfg, crf: int | None = None):
        out.parent.mkdir(parents=True, exist_ok=True)
        fps = info["fps"]
        full = first == 0 and count >= info["frames"] > 0
        cmd = ["ffmpeg", "-v", "error", "-y", "-nostdin"]
        if full:
            cmd += ["-itsoffset", f"{info['v_start'] - info['fmt_start']:.6f}"]   # keep the original video start (A/V as recorded)
        cmd += ["-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{w}x{h}", "-r", f"{fps:.6f}", "-i", "-"]
        if info["has_audio"]:
            if full:
                cmd += ["-i", str(film)]
            else:
                t0 = info["v_start"] + first / fps - info["fmt_start"]
                cmd += ["-ss", f"{t0:.6f}", "-t", f"{count / fps:.6f}", "-i", str(film)]
            cmd += ["-map", "0:v:0", "-map", "1:a:0?"]
            cmd += ["-c:a", "copy"] if full else ["-c:a", "aac", "-b:a", "192k"]
        if cfg.lossless and crf is None:
            cmd += ["-c:v", "libx264", "-preset", "medium", "-qp", "0", "-pix_fmt", "yuv420p"]
        else:
            cmd += ["-c:v", "libx264", "-preset", cfg.preset, "-crf", str(cfg.crf if crf is None else crf), "-pix_fmt", "yuv420p"]
        cmd += ["-movflags", "+faststart", "-shortest", str(out)]
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, buf: np.ndarray):
        self.p.stdin.write(buf.tobytes())

    def close(self):
        self.p.stdin.close()
        rc = self.p.wait()
        if rc:
            raise RuntimeError(f"ffmpeg encoder exited {rc}")


def stream_md5(path: Path, stream: str) -> str:
    """md5 of a stream's packets (stream copy): equal = the same coded bits."""
    return subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-map", f"0:{stream}:0", "-c", "copy", "-f", "md5", "-"],
                          capture_output=True, text=True, check=True).stdout.strip()


def restore_start(out: Path, film: Path) -> dict:
    """A whole film: put the recording's video start offset back (2026-10-21, first full-film run).

    The recorder's MP4 starts its video at 0.033 s and its audio at 0 (the A/V relation as recorded). The Writer passes that offset as
    -itsoffset on the raw pipe, but ffmpeg 8 drops it for an encoded rawvideo pipe (the encoded video starts at 0 even with -copyts),
    so the swapped film's picture would run one frame EARLY against its copied audio (before the YuvReader fix the reader's duplicated
    first frame happened to hide this: every picture one frame late in the file, at its recorded time). An -itsoffset on a demuxed MP4
    with stream copy does hold: remux the swapped video (bits unchanged) with the film's own audio (bits unchanged) and the film's
    video timescale."""
    src, got = ffprobe(film), ffprobe(out)
    want = src["v_start"] - src["fmt_start"]
    res = {"film_video_start_s": src["v_start"], "out_video_start_s_before": got["v_start"]}
    if abs((got["v_start"] - got["fmt_start"]) - want) < 0.0005:
        return res | {"remuxed": False}
    tb = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=time_base", "-of", "csv=p=0",
                         str(film)], capture_output=True, text=True, check=True).stdout.strip()
    scale = int(tb.split("/")[1]) if "/" in tb else 15360
    tmp = out.with_name(out.stem + "_restart.mp4")
    cmd = ["ffmpeg", "-v", "error", "-y", "-nostdin", "-itsoffset", f"{want:.6f}", "-i", str(out), "-i", str(film),
           "-map", "0:v:0", "-map", "1:a:0?", "-c", "copy", "-video_track_timescale", str(scale), "-movflags", "+faststart", str(tmp)]
    subprocess.run(cmd, check=True)
    v_before, v_after = stream_md5(out, "v"), stream_md5(tmp, "v")
    a_film, a_after = (stream_md5(film, "a"), stream_md5(tmp, "a")) if src["has_audio"] else ("", "")
    now = ffprobe(tmp)
    ok = v_before == v_after and a_film == a_after and now["frames"] == got["frames"] and \
        abs((now["v_start"] - now["fmt_start"]) - want) < 0.0005
    res |= {"remuxed": True, "out_video_start_s_after": now["v_start"], "video_md5_unchanged": v_before == v_after,
            "audio_md5_equals_film": a_film == a_after, "frames": now["frames"], "ok": ok}
    if not ok:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"[faceswap] restoring the video start failed: {res}")
    os.replace(tmp, out)
    print(f"[faceswap] video start restored to {now['v_start']:.6f} s (as the recording); video and audio bits unchanged", flush=True)
    return res


# ---------------------------------------------------------------------------------------------------------------------- gpu lock
class GpuLock:
    """dancecap's work/gpu.lock (same protocol as dancecap.gpu.exclusive, without importing torch)."""

    def __init__(self, label: str):
        self.label = label
        self.path = ROOT / "work" / "gpu.lock"

    def __enter__(self):
        from .gpu import _alive  # noqa: PLC0415  (stdlib + ctypes only)
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, f"{os.getpid()} {self.label}".encode())
                os.close(fd)
                return self
            except FileExistsError:
                try:
                    pid = int(self.path.read_text().split()[0])
                    if not _alive(pid):
                        self.path.unlink(missing_ok=True)
                        continue
                except (ValueError, IndexError, FileNotFoundError):
                    self.path.unlink(missing_ok=True)
                    continue
                print(f"[faceswap] waiting for gpu.lock ({self.path.read_text()})", flush=True)
                time.sleep(10)

    def __exit__(self, *exc):
        self.path.unlink(missing_ok=True)


# ------------------------------------------------------------------------------------------------------------------- FaceFusion
class FF:
    """FaceFusion v4 used as a library: state initialised from a `run` command line, then its modules called directly."""

    def __init__(self, cfg: Cfg, film: Path):
        root = cfg.facefusion
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.chdir(root)          # FaceFusion finds its processors (and its .caches) relative to the working directory
        argv = ["facefusion.py", "run", "--config-path", str(root / "facefusion.ini"),
                "-s", str(HEADSHOTS["lead"]), str(HEADSHOTS["follow"]), "-t", str(film), "-o", str(OUT_DIR),
                "--processors", "face_swapper", "--face-swapper-model", cfg.swapper_model,
                "--face-detector-model", cfg.detector, "--face-detector-score", str(cfg.det_score),
                "--face-aligner-model", "2dfan4",
                "--face-occluder-model", cfg.occluder, "--face-mask-types", "box", "occlusion",
                "--face-mask-blur", str(cfg.mask_blur), "--execution-providers", *cfg.providers, "--log-level", "warn"]
        old = sys.argv
        sys.argv = argv
        try:
            from facefusion import (content_analyser, content_store, face_aligner, face_classifier, face_creator, face_detector,  # noqa
                                    face_helper, face_masker, face_recognizer, face_store, inference_manager, logger,
                                    process_manager, session_context, session_manager, state_manager, video_manager)
            from facefusion.args_helper import apply_args
            from facefusion.processors import pixel_boost
            from facefusion.processors.modules.face_swapper import core as face_swapper
            from facefusion.program import create_program
            state_manager.init()
            program = create_program()
            args = vars(program.parse_args(argv[1:]))
            apply_args(args, state_manager.init_item)
            logger.init(state_manager.get_item("log_level"))
            session_manager.set_cli_session(session_context.get_session_id(), session_manager.create_cli_session())
            for m in (content_store, face_store, inference_manager, process_manager, video_manager):
                m.init()
        finally:
            sys.argv = old
        self.state = state_manager
        self.detector, self.creator, self.helper, self.masker = face_detector, face_creator, face_helper, face_masker
        self.analyser, self.swapper, self.boost = content_analyser, face_swapper, pixel_boost
        for m in (face_detector, face_aligner, face_recognizer, face_classifier, face_masker, content_analyser):
            if not m.pre_check():
                raise SystemExit(f"[faceswap] FaceFusion model check failed: {m.__name__}")
        if not face_swapper.pre_check():
            raise SystemExit("[faceswap] FaceFusion face swapper model check failed")
        import onnxruntime
        print(f"[faceswap] FaceFusion v4 at {root}, onnxruntime {onnxruntime.__version__} {onnxruntime.get_available_providers()}",
              flush=True)

    # -- detection ----------------------------------------------------------------------------------------------------------
    def detect(self, bgr: np.ndarray) -> list:
        """FaceFusion faces of a frame: the whole frame plus square tiles (a 9:16 frame is 1/3 scale at the 640 detector)."""
        h, w = bgr.shape[:2]
        s = min(w, h)
        boxes, scores, lms = [], [], []
        windows = [(0, 0, w, h)]
        if max(w, h) > 1.3 * s:
            span = max(w, h) - s
            for off in (0, span):
                windows.append((off, 0, s, s) if w > h else (0, off, s, s))
        for x0, y0, ww, hh in windows:
            b, sc, lm = self.detector.detect_faces(np.ascontiguousarray(bgr[y0:y0 + hh, x0:x0 + ww]))
            for bb, ss, ll in zip(b, sc, lm):
                boxes.append(np.asarray(bb, float) + [x0, y0, x0, y0])
                scores.append(float(ss))
                lms.append(np.asarray(ll, float) + [x0, y0])
        if not boxes:
            return []
        return self.creator.create_faces(bgr, boxes, scores, lms)

    # -- identity -----------------------------------------------------------------------------------------------------------
    def source(self, path: Path):
        import cv2
        img = cv2.imread(str(path))
        faces = self.creator.get_many_faces([img])
        if not faces:
            raise SystemExit(f"[faceswap] no face in the headshot {path}")
        face = max(faces, key=lambda f: (f.bounding_box[2] - f.bounding_box[0]) * (f.bounding_box[3] - f.bounding_box[1]))
        emb = self.swapper.prepare_source_embedding(face).astype(np.float32)   # inswapper: arcface x emap / |arcface|
        return face, emb

    # -- swap ---------------------------------------------------------------------------------------------------------------
    def swap_crop(self, bgr: np.ndarray, lm5: np.ndarray, emb: np.ndarray, boost: int):
        """(swapped crop, mask in crop space, affine frame -> crop) for one face; nothing pasted yet."""
        model_size = (128, 128)
        crop, M = self.helper.warp_face_by_face_landmark_5(bgr, lm5.astype(np.float32), "arcface_128", (boost, boost))
        box = self.masker.create_box_mask(crop, self.state.get_item("face_mask_blur"), self.state.get_item("face_mask_padding"))
        occ = self.masker.create_occlusion_mask(crop)
        total = boost // 128
        session = self.swapper.get_inference_pool().get("face_swapper")
        outs = []
        for tile in self.boost.implode_pixel_boost(crop, total, model_size):
            t = self.swapper.prepare_crop_frame(tile)
            r = session.run(None, {"source": emb, "target": t})[0][0]
            outs.append(self.swapper.normalize_crop_frame(r))
        swapped = self.boost.explode_pixel_boost(outs, total, model_size, (boost, boost))
        return swapped.astype(np.float32), np.minimum(box, occ).clip(0, 1).astype(np.float32), M, crop


# ------------------------------------------------------------------------------------------------------------- face geometry
def iod(lm5: np.ndarray) -> float:
    return float(np.linalg.norm(lm5[1] - lm5[0]))


def yaw5(lm5: np.ndarray) -> float:
    """Yaw (deg) from the nose's place between the eyes: SMPL-X head, nose tip 41 mm in front of the eyes, eyes 32 mm off centre."""
    e = lm5[1] - lm5[0]
    n = float(np.dot(lm5[2] - lm5[0], e) / max(1e-6, np.dot(e, e)))
    return float(np.degrees(np.arctan((n - 0.5) / 0.646)))


def pitch5(lm5: np.ndarray) -> float:
    """Pitch proxy (deg, + = face up) from the nose's height between the eye line and the mouth line (SMPL-X head: frontal 0.39)."""
    eye = (lm5[0] + lm5[1]) / 2
    mouth = (lm5[3] + lm5[4]) / 2
    d = mouth - eye
    r = float(np.dot(lm5[2] - eye, d) / max(1e-6, np.dot(d, d)))
    return float(np.degrees(np.arctan((0.39 - r) / 0.54)))


_PNP = {}


def pnp_pose(lm68: np.ndarray, w: int, h: int) -> tuple[float, float, float, float]:
    """(yaw, pitch, roll, rms px) of a 2dfan4 68-point face against the generic SMPL-X head (face_points_canonical.json 'generic')."""
    import cv2
    if "model" not in _PNP:
        fp = json.loads(FACE_POINTS.read_text())
        _PNP["model"] = np.asarray(fp["generic"]["points52_smplx"], np.float64)
        _PNP["ibug"] = np.asarray(fp["ibug52"], int)
    model, ib = _PNP["model"], _PNP["ibug"]
    img = np.asarray(lm68, np.float64)[ib]
    f = float(max(w, h))
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float64)
    ok, rvec, tvec = cv2.solvePnP(model, img, K, None, flags=cv2.SOLVEPNP_EPNP)
    if not ok:
        return float("nan"), float("nan"), float("nan"), float("nan")
    ok, rvec, tvec = cv2.solvePnP(model, img, K, None, rvec, tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
    R = cv2.Rodrigues(rvec)[0]
    proj = cv2.projectPoints(model, rvec, tvec, K, None)[0].reshape(-1, 2)
    rms = float(np.sqrt(np.mean(np.sum((proj - img) ** 2, 1))))
    fwd = R[:, 2]                                  # SMPL-X +z = out of the face
    ray = tvec.ravel() / max(1e-9, np.linalg.norm(tvec))
    # frame of the line of sight: z' = -ray (toward the camera), x' = image right made perpendicular, y' = up
    zc = -ray
    xc = np.array([1.0, 0, 0]) - zc * zc[0]
    xc /= max(1e-9, np.linalg.norm(xc))
    yc = np.cross(zc, xc)                          # camera y is down: y' = z' x x' points up in the image
    yaw = math.degrees(math.atan2(float(fwd @ xc), float(fwd @ zc)))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, float(fwd @ yc)))))   # + = face turned up
    up = R[:, 1]                                   # SMPL-X +y = head up (camera y is down)
    roll = math.degrees(math.atan2(float(up[0]), float(-up[1])))
    return yaw, pitch, roll, rms


def red_hair(bgr: np.ndarray, box: np.ndarray) -> float:
    """Share of crimson pixels in a ring around the face box (Larissa's dyed hair; Kadu's head is shaved)."""
    import cv2
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    X1, Y1 = int(max(0, x1 - 0.6 * bw)), int(max(0, y1 - 0.5 * bh))
    X2, Y2 = int(min(w, x2 + 0.6 * bw)), int(min(h, y2 + 0.8 * bh))
    if X2 - X1 < 4 or Y2 - Y1 < 4:
        return 0.0
    roi = bgr[Y1:Y2, X1:X2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    red = ((H >= 168) | (H <= 3)) & (S >= 120) & (V >= 45)
    ring = np.ones(red.shape, bool)
    ring[int(y1) - Y1:int(y2) - Y1, int(x1) - X1:int(x2) - X1] = False
    return float(red[ring].mean()) if ring.any() else 0.0


# ------------------------------------------------------------------------------------------------------------------ shot kinds
def shot_weights(direction: Path | None, n_first: int, count: int, fps: float, cfg: Cfg) -> tuple[np.ndarray, list]:
    """Heuristic (no sidecar): 1 on 3D frames, 0 on a phone's video and in the flights / blends between them; chapter-card times."""
    w = np.ones(count, np.float32)
    cards = []
    if direction is None or not Path(direction).exists():
        return w, cards
    d = json.loads(Path(direction).read_text())
    segs = d["segments"]
    t = (n_first + np.arange(count)) / fps
    for i, s in enumerate(segs):
        f0, f1 = s["film"]
        mode = s["camera"].get("mode")
        inside = (t >= f0) & (t < f1)
        if mode in PHONE_MODES:
            w[inside] = 0.0
        elif i > 0 and segs[i - 1]["camera"].get("mode") in PHONE_MODES:
            w[inside & (t < f0 + cfg.mix_after_phone_s)] = 0.0
        cc = s.get("chapter_card")
        if cc:
            cards.append((f0 + cc.get("film_in", 0.0) - 0.4, f0 + cc.get("film_out", 2.6) + 0.6, cc.get("kind", "chapter")))
    # a v3 direction (film_recap_flow): a phone's video stands on its frustum surface in the 3D world through every way in and out;
    # without a sidecar nothing says where on the screen it is, so nothing is swapped while one shows (+- 2 frames)
    for fz in (d.get("flow") or {}).get("frustums", []):
        a, _, _, b = fz["film"]
        w[(t >= a - 2.0 / fps) & (t <= b + 2.0 / fps)] = 0.0
    return w, cards


def heuristic_ui(w: int, h: int, t: float, cards: list) -> list:
    """UI bands of the 9:16 film where text may lie over a face: the caption block, the chapter / title card while it shows."""
    if h <= w:
        return []          # 16:9: only the sidecar knows its rects
    rects = [(0, int(0.655 * h), w, int(0.15 * h))]
    for a, b, _ in cards:
        if a <= t <= b:
            rects.append((int(0.12 * w), int(0.12 * h), int(0.76 * w), int(0.13 * h)))
    return rects


def glyph_mask(Y: np.ndarray, U: np.ndarray, V: np.ndarray, rect) -> tuple[slice, slice, np.ndarray] | None:
    """Text pixels (white words, the yellow current-word box and its dark letters) inside a UI rect, dilated 4 px."""
    import cv2
    H, W = Y.shape
    x, y, w, h = rect
    x1, y1 = max(0, int(x) & ~1), max(0, int(y) & ~1)
    x2, y2 = min(W, (int(x + w) + 1) & ~1), min(H, (int(y + h) + 1) & ~1)
    if x2 <= x1 or y2 <= y1:
        return None
    yy = Y[y1:y2, x1:x2].astype(np.int16)
    uu = np.repeat(np.repeat(U[y1 // 2:y2 // 2, x1 // 2:x2 // 2], 2, 0), 2, 1).astype(np.int16) - 128
    vv = np.repeat(np.repeat(V[y1 // 2:y2 // 2, x1 // 2:x2 // 2], 2, 0), 2, 1).astype(np.int16) - 128
    white = (yy > 196) & (np.abs(uu) < 14) & (np.abs(vv) < 14)
    yellow = (yy > 150) & (uu < -40) & (vv > 5)
    m = (white | yellow).astype(np.uint8)
    if not m.any():
        return None
    m = cv2.dilate(m, np.ones((9, 9), np.uint8))
    return slice(y1, y2), slice(x1, x2), m.astype(bool)


# --------------------------------------------------------------------------------------------------------------------- sidecar
def load_sidecar(path: Path, first: int, count: int, info: dict) -> dict:
    sc = json.loads(Path(path).read_text())
    problems = validate_sidecar(sc, info)
    if problems:
        raise SystemExit("[faceswap] sidecar invalid:\n  " + "\n  ".join(problems[:20]))
    return sc


def validate_sidecar(sc: dict, info: dict | None = None) -> list:
    p = []
    if sc.get("format") != SIDECAR_FORMAT or sc.get("version") != 1:
        p.append(f"format/version: {sc.get('format')!r} {sc.get('version')!r} (want {SIDECAR_FORMAT!r} 1)")
    if list(sc.get("landmarks", [])) != list(LANDMARKS5):
        p.append(f"landmarks order {sc.get('landmarks')} != {LANDMARKS5}")
    keys = {d.get("key") for d in sc.get("dancers", [])}
    if not set(ROLES) <= keys:
        p.append(f"dancers {keys} must name lead and follow")
    fr = sc.get("f", [])
    if info is not None:
        if (sc.get("width"), sc.get("height")) != (info["width"], info["height"]):
            p.append(f"size {sc.get('width')}x{sc.get('height')} != film {info['width']}x{info['height']}")
        if info["frames"] and len(fr) != info["frames"]:
            p.append(f"{len(fr)} sidecar frames != film {info['frames']} frames")
    for i, e in enumerate(fr[:: max(1, len(fr) // 400)]):
        for d in e.get("d", []):
            if d.get("in") and (len(d.get("lm", [])) != 5 or any(len(q) != 2 for q in d["lm"])):
                p.append(f"frame entry {i}: dancer {d.get('k')} 'lm' is not 5 x [x, y]")
                break
        for sf in e.get("surfaces", []) or []:
            q = sf.get("quad", [])
            if not (0.0 <= float(sf.get("alpha", -1)) <= 1.0) or len(q) < 3 or any(len(x) != 2 for x in q):
                p.append(f"frame entry {i}: surface {sf.get('phone')} needs alpha 0..1 and a 'quad' of >= 3 [x, y] (4 unless clipped)")
                break
            if sf.get("z") is not None and len(sf["z"]) != len(q):
                p.append(f"frame entry {i}: surface {sf.get('phone')} 'z' must have one depth per 'quad' point")
                break
    return p


def point_in_poly(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """(n, 2) points inside the polygon (m, 2), even-odd rule (any winding, convex or not)."""
    pts, poly = np.asarray(pts, float).reshape(-1, 2), np.asarray(poly, float).reshape(-1, 2)
    x, y = pts[:, 0:1], pts[:, 1:2]
    x1, y1 = poly[:, 0][None], poly[:, 1][None]
    x2, y2 = np.roll(poly[:, 0], -1)[None], np.roll(poly[:, 1], -1)[None]
    cross = (y1 > y) != (y2 > y)
    xi = x1 + (y - y1) * (x2 - x1) / np.where(y2 == y1, 1e-12, y2 - y1)
    return (np.count_nonzero(cross & (x < xi), axis=1) % 2) == 1


def surface_depth_at(pts: np.ndarray, quad: np.ndarray, z: np.ndarray) -> np.ndarray:
    """camera depth of a planar surface at pixel points: 1 / z is affine in the screen coordinates of a plane (fit to its corners)."""
    q = np.asarray(quad, float).reshape(-1, 2)
    P = np.asarray(pts, float).reshape(-1, 2)
    coef, *_ = np.linalg.lstsq(np.c_[q, np.ones(len(q))], 1.0 / np.maximum(1e-6, np.asarray(z, float)), rcond=None)
    return 1.0 / np.maximum(1e-6, np.c_[P, np.ones(len(P))] @ coef)


def surface_hit(lm: np.ndarray, zf, surfaces: list, cfg: Cfg) -> str | None:
    """the phone whose frustum video surface hides this face (None = none does): a surface with alpha > cfg.surface_alpha that covers
    one of the 5 points or a corner of the face box (their centre +- cfg.surface_box_iod x inter-ocular) and lies in front of the face
    there (its plane depth at those pixels < the face's depth; without depths, or `full`: always in front)."""
    if not surfaces:
        return None
    lm = np.asarray(lm, float)
    c = lm.mean(0)
    r = cfg.surface_box_iod * max(4.0, iod(lm))
    pts = np.vstack([lm, c + np.array([[-r, -r], [r, -r], [r, r], [-r, r]])])
    zface = float(np.max(zf)) if zf is not None and len(zf) else None
    for sf in surfaces:
        if float(sf.get("alpha", 0.0)) <= cfg.surface_alpha:
            continue
        if bool(sf.get("full")):
            return str(sf.get("phone"))
        inside = point_in_poly(pts, sf["quad"])
        if not inside.any():
            continue
        if sf.get("z") is None or zface is None:
            return str(sf.get("phone"))
        if np.any(surface_depth_at(pts[inside], sf["quad"], sf["z"]) < zface):
            return str(sf.get("phone"))
    return None


# ------------------------------------------------------------------------------------------------------------------- analysis
def analyse(ff: FF, film: Path, info: dict, first: int, count: int, shot_w: np.ndarray, sources: dict, cfg: Cfg,
            sidecar: dict | None) -> list:
    """Pass 1: per frame, the FaceFusion detections (only where a swap could happen)."""
    import cv2
    w, h = info["width"], info["height"]
    frames = []
    nsfw_hits = 0
    t0 = time.time()
    for k, buf in enumerate(YuvReader(film, info, first, count)):
        i = first + k
        need = shot_w[k] > 0
        if sidecar is not None:
            e = sidecar["f"][i]
            need = cfg.phone_shots == "swap" or e.get("video_alpha", 0.0) < 0.999
            need = need and any(d.get("in") for d in e.get("d", []))
        rec = {"i": i, "dets": []}
        if need:
            bgr = cv2.cvtColor(buf.reshape(h * 3 // 2, w), cv2.COLOR_YUV2BGR_I420)
            if k % 15 == 0 and ff.analyser.analyse_frame(bgr):       # FaceFusion's own content check, kept in the loop
                nsfw_hits += 1
                if nsfw_hits >= 2:
                    raise SystemExit("[faceswap] FaceFusion's content analyser flagged this film; stopping")
            for f in ff.detect(bgr):
                lm5 = np.asarray(f.landmark_set.get("5/68"), float)
                lm68 = np.asarray(f.landmark_set.get("68"), float)
                yaw, pitch, roll, rms = pnp_pose(lm68, w, h)
                en = np.asarray(f.embedding_norm, float)
                rec["dets"].append({
                    "box": [round(float(v), 1) for v in f.bounding_box],
                    "det": round(float(f.score_set["detector"]), 3), "aln": round(float(f.score_set["aligner"]), 3),
                    "lm5": np.round(lm5, 2).tolist(), "lm68": np.round(lm68, 1).tolist(),
                    "iod": round(iod(lm5), 2), "yaw5": round(yaw5(lm5), 1), "pitch5": round(pitch5(lm5), 1),
                    "yaw": round(yaw, 1), "pitch": round(pitch, 1), "roll": round(roll, 1), "pnp_rms": round(rms, 2),
                    "sim": {r: round(float(en @ sources[r]["norm"]), 4) for r in ROLES},
                    "male": f.gender == "male", "age": [f.age.start, f.age.stop],
                    "red": round(red_hair(bgr, np.asarray(f.bounding_box, float)), 4)})
        frames.append(rec)
        if (k + 1) % 60 == 0:
            print(f"[faceswap] analysis {k + 1}/{count} frames, {(k + 1) / (time.time() - t0):.1f} fps", flush=True)
    return frames


# ------------------------------------------------------------------------------------------------------------------------ plan
def ramp(x: float, lo: float, hi: float) -> float:
    """0 at lo, 1 at hi (smoothstep); lo > hi gives the falling version."""
    if hi == lo:
        return float(x >= hi)
    u = min(1.0, max(0.0, (x - lo) / (hi - lo)))
    return u * u * (3 - 2 * u)


def gate(iod_px: float, yaw: float, pitch: float, cfg: Cfg, vis: float = 1.0) -> float:
    g = ramp(iod_px, *cfg.iod_px)
    g *= ramp(abs(yaw), cfg.yaw_deg[1], cfg.yaw_deg[0])
    g *= ramp(abs(pitch), cfg.pitch_deg[1], cfg.pitch_deg[0])
    g *= ramp(vis, *cfg.vis)
    return g


def hysteresis(g: np.ndarray, cfg: Cfg) -> np.ndarray:
    out = np.zeros(len(g))
    on = False
    for k, v in enumerate(g):
        on = v >= cfg.gate_on if not on else v >= cfg.gate_off
        out[k] = 1.0 if on else 0.0
    return out


def spans_of(on: np.ndarray) -> list:
    """[(k, j)): the runs of True"""
    on = np.asarray(on, bool)
    if not on.any():
        return []
    d = np.diff(np.r_[0, on.astype(np.int8), 0])
    return list(zip(np.where(d == 1)[0].tolist(), np.where(d == -1)[0].tolist()))


def bridge_gaps(on: np.ndarray, g: np.ndarray, cfg: Cfg) -> np.ndarray:
    """an off gap of at most bridge_frames between two on runs is filled where the face is still a face (gate >= gate_bridge)"""
    out = np.asarray(on, bool).copy()
    sp = spans_of(out)
    for (_, j), (k2, _) in zip(sp, sp[1:]):
        if k2 - j <= cfg.bridge_frames and np.min(g[j:k2]) >= cfg.gate_bridge:
            out[j:k2] = True
    return out


def drop_short(on: np.ndarray, n: int, open_start: bool = False, open_end: bool = False) -> np.ndarray:
    """runs shorter than n frames go (a run touching an open end of the range stays: it continues outside)"""
    out = np.asarray(on, bool).copy()
    for k, j in spans_of(out):
        if j - k < n and not (open_start and k == 0) and not (open_end and j == len(out)):
            out[k:j] = False
    return out


def track_on(g: np.ndarray, shot: np.ndarray, cfg: Cfg) -> tuple[np.ndarray, np.ndarray]:
    """(on, raw): raw = the face gate with hysteresis x (shot == pure 3D, or a phone shot when allowed); on = raw with the short
    dips bridged (never into a frame where the shot does not allow a swap)"""
    g = np.nan_to_num(np.asarray(g, float))
    pure = np.asarray(shot) >= 0.999
    raw = hysteresis(g, cfg) * pure
    return bridge_gaps(raw > 0.5, g, cfg) & pure, raw


def temporal_weight(g: np.ndarray, shot: np.ndarray, cfg: Cfg, open_start: bool = False,
                    open_end: bool = False, keep: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(weight, raw): `track_on` (hysteresis, shot, bridged dips) & keep; runs shorter than min_on_frames dropped; then erosion +
    box blur: fades of fade_frames that reach 0 on the last qualifying frame. An end of the processed RANGE (not of the track) is
    open: the weight there continues as it is (chunks of a film join without a fade)."""
    on, w_raw = track_on(g, shot, cfg)
    if keep is not None:
        on &= np.asarray(keep, bool)
    on = drop_short(on, cfg.min_on_frames, open_start, open_end)
    return fade_weight(on, cfg, open_start, open_end), w_raw


def fade_weight(on: np.ndarray, cfg: Cfg, open_start: bool = False, open_end: bool = False) -> np.ndarray:
    """erosion + box blur of the on mask: fades of fade_frames that reach 0 on the last on frame (an open range end stays as it is)"""
    from scipy.ndimage import minimum_filter1d, uniform_filter1d
    w = np.asarray(on, bool).astype(np.float64)
    F = cfg.fade_frames
    half = F // 2
    pad = 2 * F + 4                 # longer than the erosion + blur reach, so an open end is not pulled down
    w = np.concatenate([np.full(pad, w[0] if open_start else 0.0), w, np.full(pad, w[-1] if open_end else 0.0)])
    w = minimum_filter1d(w, size=2 * half + 3, mode="constant", cval=0.0)
    w = uniform_filter1d(w, size=2 * half + 1, mode="constant", cval=0.0)
    return w[pad:-pad].clip(0, 1)


def swap_spans(plan: list, fps: float, first: int) -> dict:
    """per dancer, the film seconds its swap shows (weight > 0.004, over all its tracks): spans, the shortest span and gap - the
    flicker check of the report (no span under min_on_frames, no toggling)"""
    out = {}
    for role in ROLES:
        ps = [p for p in plan if p["role"] == role]
        if not ps:
            continue
        n = max(p["k1"] for p in ps) + 1
        on = np.zeros(n, bool)
        for p in ps:
            on[p["k0"]:p["k1"] + 1] |= np.asarray(p["w"]) > 0.004
        sp = spans_of(on)
        out[role] = {"spans_s": [[round((first + k) / fps, 2), round((first + j) / fps, 2)] for k, j in sp], "n": len(sp),
                     "shortest_s": round(min((j - k for k, j in sp), default=0) / fps, 2),
                     "shortest_gap_s": round(min(b - a for (_, a), (b, _) in zip(sp, sp[1:])) / fps, 2) if len(sp) > 1 else None}
    return out


def smooth_track(lms: np.ndarray, cfg: Cfg) -> np.ndarray:
    """(n, 5, 2) -> zero-phase Savitzky-Golay along time (short tracks: a 3-tap mean)."""
    from scipy.signal import savgol_filter
    n = len(lms)
    if n >= cfg.smooth_window:
        return savgol_filter(lms, cfg.smooth_window, 2, axis=0, mode="interp")
    if n >= 3:
        out = lms.copy()
        out[1:-1] = (lms[:-2] + lms[1:-1] + lms[2:]) / 3
        return out
    return lms


def boost_for(iod_px: float) -> int:
    need = iod_px / (0.63696719 - 0.36167656)     # crop side that keeps the face at arcface_128's scale
    return 128 if need <= 160 else 256 if need <= 320 else 512


def build_tracks(frames: list, cfg: Cfg) -> list:
    """Greedy box tracker (IoU / centre distance), gaps up to cfg.gap_fill frames."""
    tracks = []
    for k, rec in enumerate(frames):
        dets = [d for d in rec["dets"] if d["aln"] >= cfg.aligner_min]
        used = set()
        for tr in sorted(tracks, key=lambda t: -len(t["k"])):
            if k - tr["k"][-1] > cfg.gap_fill + 1:
                continue
            last = tr["d"][-1]
            best, bj = None, None
            for j, d in enumerate(dets):
                if j in used:
                    continue
                c0 = np.mean(np.asarray(last["lm5"]), 0)
                c1 = np.mean(np.asarray(d["lm5"]), 0)
                dist = np.linalg.norm(c1 - c0) / max(8.0, last["iod"])
                size = d["iod"] / max(1.0, last["iod"])
                if dist < 1.6 and 0.55 < size < 1.8 and (best is None or dist < best):
                    best, bj = dist, j
            if bj is not None:
                used.add(bj)
                tr["k"].append(k)
                tr["d"].append(dets[bj])
        for j, d in enumerate(dets):
            if j not in used:
                tracks.append({"k": [k], "d": [d]})
    return tracks


def identity_score(d: dict) -> float:
    """> 0 = lead (Kadu), < 0 = follow (Larissa)."""
    s = 6.0 * (d["sim"]["lead"] - d["sim"]["follow"])
    s += 0.5 if d["male"] else -0.5
    s -= 3.0 * min(0.3, max(0.0, d["red"] - 0.08))     # capped: in the embrace her hair lies next to his face
    return float(s)


def plan_heuristic(frames: list, shot_w: np.ndarray, cfg: Cfg) -> tuple[list, list]:
    tracks = build_tracks(frames, cfg)
    out = []
    for n, tr in enumerate(tracks):
        scores = [identity_score(d) for d in tr["d"]]
        mean = float(np.mean(scores))
        tr["score"] = mean
        tr["role"] = None if abs(mean) < cfg.identity_min else ("lead" if mean > 0 else "follow")
    # two tracks of one dancer at the same time: the weaker one gives way (or becomes the other dancer when it leans that way)
    for a in tracks:
        for b in tracks:
            if a is b or a["role"] is None or a["role"] != b["role"]:
                continue
            if set(a["k"]) & set(b["k"]):
                weak = a if abs(a["score"]) < abs(b["score"]) else b
                weak["role"] = None
    for n, tr in enumerate(tracks):
        k0, k1 = tr["k"][0], tr["k"][-1]
        span = np.arange(k0, k1 + 1)
        lm = np.zeros((len(span), 5, 2))
        g = np.zeros(len(span))
        yaw = np.full(len(span), np.nan)
        have = np.zeros(len(span), bool)
        for k, d in zip(tr["k"], tr["d"]):
            j = k - k0
            lm[j] = np.asarray(d["lm5"])
            have[j] = True
            yw = d["yaw"] if np.isfinite(d["yaw"]) and d["pnp_rms"] < 0.25 * d["iod"] else d["yaw5"]
            ya = max(abs(yw), abs(d["yaw5"]))
            pa = d["pitch"] if np.isfinite(d["pitch"]) and d["pnp_rms"] < 0.25 * d["iod"] else d["pitch5"]
            yaw[j] = ya
            g[j] = gate(d["iod"], ya, pa, cfg) * min(1.0, d["det"] / 0.6)
        for c in range(5):                       # interpolate the gaps
            for xy in range(2):
                lm[~have, c, xy] = np.interp(span[~have], span[have], lm[have, c, xy])
        gi = np.interp(span, span[have], g[have])
        gaps = ~have
        if gaps.any():                           # an interpolated frame is at most as good as its weaker neighbour
            prev = np.maximum.accumulate(np.where(have, np.arange(len(span)), 0))
            nxt = np.minimum.accumulate(np.where(have, np.arange(len(span)), len(span) - 1)[::-1])[::-1]
            gi[gaps] = np.minimum(g[prev[gaps]], g[nxt[gaps]])
        lm = smooth_track(lm, cfg)
        if tr["role"] is None:
            gi[:] = 0
        on, w_raw = track_on(gi, shot_w[span], cfg)
        med_iod = float(np.median([d["iod"] for d in tr["d"]]))
        out.append({"track": n, "role": tr["role"], "score": round(tr["score"], 3), "k0": int(k0), "k1": int(k1),
                    "n_det": len(tr["k"]), "boost": boost_for(med_iod), "median_iod": round(med_iod, 1),
                    "lm": lm, "on": on, "w_raw": w_raw, "yaw": yaw,
                    "sim_lead": round(float(np.mean([d["sim"]["lead"] for d in tr["d"]])), 3),
                    "sim_follow": round(float(np.mean([d["sim"]["follow"] for d in tr["d"]])), 3),
                    "male_share": round(float(np.mean([d["male"] for d in tr["d"]])), 2),
                    "red_mean": round(float(np.mean([d["red"] for d in tr["d"]])), 3)})
    min_on_per_dancer(out, len(frames), cfg)
    return out, tracks


def min_on_per_dancer(plan: list, n: int, cfg: Cfg) -> None:
    """the minimum-on rule per DANCER: the on masks of all tracks of one role are joined over the range (a face the tracker split into
    two tracks counts as one run), runs shorter than min_on_frames go, the fades are made on that joined timeline (no dip where one
    track hands over to the next), and each track takes its frames of it (p['w'])"""
    for role in (*ROLES, None):
        ps = [p for p in plan if p["role"] == role]
        if not ps:
            continue
        if role is None:
            for p in ps:
                p["w"] = np.zeros(p["k1"] - p["k0"] + 1)
                p.pop("on", None)
            continue
        union = np.zeros(n, bool)
        for p in ps:
            union[p["k0"]:p["k1"] + 1] |= p["on"]
        kept = drop_short(union, cfg.min_on_frames, open_start=True, open_end=True)
        w_role = fade_weight(kept, cfg, open_start=True, open_end=True)
        for p in ps:
            on = p.pop("on") & kept[p["k0"]:p["k1"] + 1]
            p["w"] = w_role[p["k0"]:p["k1"] + 1] * on


def plan_sidecar(frames: list, sidecar: dict, first: int, count: int, cfg: Cfg) -> list:
    """Per dancer one track over the range: the sidecar says who / where, a detection on the projected face refines the points."""
    out = []
    for role in ROLES:
        lm = np.zeros((count, 5, 2))
        g = np.zeros(count)
        shot = np.zeros(count)
        yaw = np.full(count, np.nan)
        src = np.zeros(count, np.int8)          # 0 none, 1 sidecar only, 2 detection
        offs = np.full((count, 2), np.nan)
        ious = []
        behind = {}                              # frames where a phone's frustum video surface hides the face: phone -> n
        for k in range(count):
            e = sidecar["f"][first + k]
            d = next((q for q in e.get("d", []) if q.get("k") == role), None)
            if not d or not d.get("in"):
                continue
            p = np.asarray(d["lm"], float)
            lm[k] = p
            src[k] = 1
            di = iod(p)
            ious.append(di)
            hid = surface_hit(p, d.get("z"), e.get("surfaces") or [], cfg)
            if hid is not None:
                # the picture there is the phone's video (real faces, maybe the teachers themselves): never swapped, and no detection
                # on it may refine the projected points
                behind[hid] = behind.get(hid, 0) + 1
                yaw[k] = float(d.get("yaw", np.nan))
                continue
            best = None
            for det in frames[k]["dets"]:
                q = np.asarray(det["lm5"])
                err = float(np.mean(np.linalg.norm(q - p, axis=1))) / max(6.0, di)
                if err < 0.45 and (best is None or err < best[0]):
                    best = (err, q)
            if best is not None:
                offs[k] = np.mean(best[1] - p, 0)
                lm[k] = best[1]
                src[k] = 2
            va = float(e.get("video_alpha", 0.0))
            shot[k] = 1.0 if cfg.phone_shots == "swap" else 1.0 - va
            g[k] = gate(di, float(d.get("yaw", 0.0)), float(d.get("pitch", 0.0)), cfg, float(d.get("vis", 1.0)))
            yaw[k] = float(d.get("yaw", np.nan))
        # sidecar-only frames get the running registration offset of the detector (median over +-15 frames)
        have = np.isfinite(offs[:, 0])
        if have.any():
            idx = np.arange(count)
            for k in np.where(src == 1)[0]:
                near = have & (np.abs(idx - k) <= 15)
                if near.any():
                    lm[k] += np.median(offs[near], 0)
        k = 0
        while k < count:                             # smooth each run of frames where the dancer is in view on its own
            if src[k]:
                j = k
                while j < count and src[j]:
                    j += 1
                lm[k:j] = smooth_track(lm[k:j], cfg)
                k = j
            else:
                k += 1
        w, w_raw = temporal_weight(g, shot, cfg, open_start=True, open_end=True)
        med_iod = float(np.median(ious)) if ious else 0.0
        out.append({"track": len(out), "role": role, "score": None, "k0": 0, "k1": count - 1, "n_det": int((src == 2).sum()),
                    "n_sidecar_only": int((src == 1).sum()), "boost": boost_for(med_iod), "median_iod": round(med_iod, 1),
                    "behind_surface": behind, "lm": lm, "w": w, "w_raw": w_raw, "yaw": yaw})
    return out


# ------------------------------------------------------------------------------------------------------------------------ swap
def color_match(swapped: np.ndarray, crop: np.ndarray, mask: np.ndarray, amount: float) -> np.ndarray:
    import cv2
    if amount <= 0:
        return swapped
    m = mask > 0.5
    if m.sum() < 50:
        return swapped
    a = cv2.cvtColor(np.clip(swapped, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    b = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    for c in range(3):
        ma, sa = a[..., c][m].mean(), a[..., c][m].std() + 1e-3
        mb, sb = b[..., c][m].mean(), b[..., c][m].std() + 1e-3
        a[..., c] = (1 - amount) * a[..., c] + amount * ((a[..., c] - ma) * (sb / sa) + mb)
    return cv2.cvtColor(np.clip(a, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)


def paste_yuv(Y, U, V, bgr, swapped, mask, M, weight):
    """Blend one swapped face into the frame planes (and the working BGR) where mask x weight > 1/255; returns the support box."""
    import cv2
    H, W = Y.shape
    crop_h, crop_w = swapped.shape[:2]
    Minv = cv2.invertAffineTransform(M)
    pts = np.array([[0, 0], [crop_w, 0], [crop_w, crop_h], [0, crop_h]], float)
    pr = pts @ Minv[:, :2].T + Minv[:, 2]
    x1, y1 = np.floor(pr.min(0)).astype(int)
    x2, y2 = np.ceil(pr.max(0)).astype(int)
    x1, y1 = max(0, x1) & ~1, max(0, y1) & ~1
    x2, y2 = min(W, (x2 + 1) & ~1), min(H, (y2 + 1) & ~1)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    P = Minv.copy()
    P[0, 2] -= x1
    P[1, 2] -= y1
    size = (x2 - x1, y2 - y1)
    m = cv2.warpAffine(mask * weight, P, size).clip(0, 1)
    face = cv2.warpAffine(swapped, P, size, borderMode=cv2.BORDER_REPLICATE)
    sup = m > (1.0 / 255.0)
    if not sup.any():
        return None
    region = bgr[y1:y2, x1:x2].astype(np.float32)
    region = region * (1 - m[..., None]) + face * m[..., None]
    region_u8 = np.clip(np.rint(region), 0, 255).astype(np.uint8)
    rh, rw = region_u8.shape[:2]
    yuv = cv2.cvtColor(region_u8, cv2.COLOR_BGR2YUV_I420).reshape(-1)
    Yr = yuv[:rh * rw].reshape(rh, rw)
    Ur = yuv[rh * rw:rh * rw * 5 // 4].reshape(rh // 2, rw // 2)
    Vr = yuv[rh * rw * 5 // 4:].reshape(rh // 2, rw // 2)
    sup2 = sup.reshape(rh // 2, 2, rw // 2, 2).any(axis=(1, 3))
    ys = Y[y1:y2, x1:x2]
    ys[sup] = Yr[sup]
    us = U[y1 // 2:y2 // 2, x1 // 2:x2 // 2]
    vs = V[y1 // 2:y2 // 2, x1 // 2:x2 // 2]
    us[sup2] = Ur[sup2]
    vs[sup2] = Vr[sup2]
    bgr[y1:y2, x1:x2][sup] = region_u8[sup]
    return [int(x1), int(y1), int(x2), int(y2)]


def overlay_mask(bgr: np.ndarray, min_share: float = 0.004) -> np.ndarray:
    """The film's own 3D overlay strokes inside a face's paste box (2026-10-21, first full-film run: where the physics view's skeleton
    / load lines cross a swapped face, xseg did not always see them as occluders and the swap partly painted the line away, e.g.
    Larissa at 316 s). Grey-lavender and blue strokes are bluer than skin, lips, brows and the red hair (blue - red > -12 with the
    brightest channel > 90; splat skin has red - blue >= 30, the dark eyes stay below 90); only connected strokes of some size count
    (>= 0.4 % of the box, >= 12 px: no eye whites), grown by 1 px for their anti-aliased rims. The swap is undone there, so the line
    stays in front of the face as recorded. Red / orange tension strokes are not separable from skin by colour and are left to xseg."""
    import cv2
    b, r = bgr[..., 0].astype(np.int16), bgr[..., 2].astype(np.int16)
    m = ((b - r) > -12) & (bgr.max(axis=2) > 90)
    if not m.any():
        return m
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= max(12, int(min_share * m.size))
    if not keep.any():
        return np.zeros_like(m)
    return cv2.dilate(keep[lab].astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)


def restore_overlay(Y, U, V, Yo, Uo, Vo, src_bgr: np.ndarray, box) -> int:
    """Put the recorded pixels back where overlay_mask finds a stroke in the source inside an (even-aligned) paste box."""
    x1, y1, x2, y2 = box
    m = overlay_mask(src_bgr[y1:y2, x1:x2])
    if not m.any():
        return 0
    ys, yo = Y[y1:y2, x1:x2], Yo[y1:y2, x1:x2]
    ys[m] = yo[m]
    m2 = m.reshape(m.shape[0] // 2, 2, m.shape[1] // 2, 2).any(axis=(1, 3))
    cs = (slice(y1 // 2, y2 // 2), slice(x1 // 2, x2 // 2))
    U[cs][m2] = Uo[cs][m2]
    V[cs][m2] = Vo[cs][m2]
    return int(m.sum())


def run_swap(ff: FF, film: Path, info: dict, first: int, count: int, plan: list, sources: dict, cfg: Cfg, out: Path,
             sbs: Path | None, ui_of_frame) -> dict:
    import cv2
    w, h = info["width"], info["height"]
    writer = Writer(out, film, info, first, count, w, h, cfg)
    sbs_writer = Writer(sbs, film, info, first, count, 2 * w, h, cfg, crf=17) if sbs else None
    stats = {"frames": 0, "faces_swapped": 0, "changed_outside_support": 0, "glyph_restored_px": 0, "overlay_restored_px": 0,
             "overlay_restored_frames": 0, "per_frame": []}
    t0 = time.time()
    t_model = 0.0
    for k, buf in enumerate(YuvReader(film, info, first, count)):
        orig = buf.copy()
        Y, U, V = planes(buf, w, h)
        todo = [(p, float(p["w"][k - p["k0"]])) for p in plan if p["role"] and p["k0"] <= k <= p["k1"] and p["w"][k - p["k0"]] > 0.004]
        boxes = []
        if todo:
            bgr = cv2.cvtColor(buf.reshape(h * 3 // 2, w), cv2.COLOR_YUV2BGR_I420)
            src_bgr = bgr.copy()
            for p, wt in todo:
                lm5 = p["lm"][k - p["k0"]]
                tm = time.time()
                swapped, mask, M, crop = ff.swap_crop(bgr, lm5, sources[p["role"]]["emb"], p["boost"])
                swapped = color_match(swapped, crop, mask, cfg.color_match)
                t_model += time.time() - tm
                b = paste_yuv(Y, U, V, bgr, swapped, mask, M, wt)
                if b:
                    boxes.append({"role": p["role"], "w": round(wt, 3), "box": b, "boost": p["boost"],
                                  "lm5": np.round(lm5, 1).tolist()})
                    stats["faces_swapped"] += 1
        # the film's own 3D overlay strokes over a swapped face stay as recorded (in front of the face)
        if boxes:
            Yo, Uo, Vo = planes(orig, w, h)
            n_ov = sum(restore_overlay(Y, U, V, Yo, Uo, Vo, src_bgr, b["box"]) for b in boxes)
            stats["overlay_restored_px"] += n_ov
            stats["overlay_restored_frames"] += int(n_ov > 0)
        # UI text over a swapped face stays as recorded
        if boxes:
            Yo, Uo, Vo = planes(orig, w, h)
            for rect in ui_of_frame(first + k):
                g = glyph_mask(Yo, Uo, Vo, rect)
                if g is None:
                    continue
                sy, sx, m = g
                if not np.any(Y[sy, sx][m] != Yo[sy, sx][m]):
                    continue
                Y[sy, sx][m] = Yo[sy, sx][m]
                m2 = m.reshape(m.shape[0] // 2, 2, m.shape[1] // 2, 2).any(axis=(1, 3))
                csy, csx = slice(sy.start // 2, sy.stop // 2), slice(sx.start // 2, sx.stop // 2)
                U[csy, csx][m2] = Uo[csy, csx][m2]
                V[csy, csx][m2] = Vo[csy, csx][m2]
                stats["glyph_restored_px"] += int(m.sum())
        # check: nothing changed outside the support boxes
        if boxes:
            Yo, _, _ = planes(orig, w, h)
            keep = np.ones((h, w), bool)
            for b in boxes:
                keep[b["box"][1]:b["box"][3], b["box"][0]:b["box"][2]] = False
            stats["changed_outside_support"] += int(np.count_nonzero((Y != Yo) & keep))
        else:
            stats["changed_outside_support"] += int(np.count_nonzero(buf != orig))
        writer.write(buf)
        if sbs_writer:
            Yo, Uo, Vo = planes(orig, w, h)
            sb = np.concatenate([np.hstack([Yo, Y]).ravel(), np.hstack([Uo, U]).ravel(), np.hstack([Vo, V]).ravel()])
            sbs_writer.write(sb)
        stats["per_frame"].append({"i": first + k, "faces": boxes})
        stats["frames"] += 1
        if (k + 1) % 60 == 0:
            print(f"[faceswap] swap {k + 1}/{count} frames, {(k + 1) / (time.time() - t0):.1f} fps", flush=True)
    writer.close()
    if sbs_writer:
        sbs_writer.close()
    stats["swap_seconds"] = round(time.time() - t0, 2)
    stats["swap_fps"] = round(stats["frames"] / max(1e-6, stats["swap_seconds"]), 2)
    stats["model_seconds"] = round(t_model, 2)
    stats["model_ms_per_face"] = round(1000 * t_model / max(1, stats["faces_swapped"]), 1)
    return stats


# ------------------------------------------------------------------------------------------------------------------------- cli
def cmd_swap(a):
    cfg = Cfg(color_match=a.color_match, phone_shots=a.phone_shots, lossless=a.lossless, crf=a.crf, detector=a.detector)
    for k in ("film", "out", "side_by_side", "sidecar", "direction"):
        if getattr(a, k):
            setattr(a, k, str(Path(getattr(a, k)).resolve()))
    film = Path(a.film)
    info = ffprobe(film)
    fps = info["fps"]
    first = int(round((a.start or 0.0) * fps))
    last = info["frames"] if a.end is None else int(round(a.end * fps))
    count = max(1, min(last, info["frames"] or last) - first)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sidecar = load_sidecar(Path(a.sidecar), first, count, info) if a.sidecar else None
    shot_w, cards = shot_weights(None if sidecar else Path(a.direction) if a.direction else None, first, count, fps, cfg)
    report = {"film": str(film), "out": str(out), "side_by_side": a.side_by_side, "first_frame": first, "frames": count,
              "film_s": [first / fps, (first + count) / fps], "mode": "sidecar" if sidecar else "heuristic",
              "sidecar": a.sidecar, "direction": None if sidecar else a.direction, "cfg": {k: (str(v) if isinstance(v, Path) else v)
                                                                                         for k, v in asdict(cfg).items()},
              "sources": {r: str(HEADSHOTS[r]) for r in ROLES}}
    with GpuLock(f"film_faceswap {out.name}"):
        ff = FF(cfg, film)
        sources = {}
        for r in ROLES:
            face, emb = ff.source(HEADSHOTS[r])
            sources[r] = {"emb": emb, "norm": np.asarray(face.embedding_norm, float), "gender": face.gender,
                          "iod": iod(np.asarray(face.landmark_set.get("5/68")))}
        report["source_faces"] = {r: {"gender": sources[r]["gender"], "iod_px": round(sources[r]["iod"], 1),
                                      "sim_to_other": round(float(sources[r]["norm"] @ sources[[x for x in ROLES if x != r][0]]["norm"]), 3)}
                                  for r in ROLES}
        t0 = time.time()
        cache = out.with_suffix(".analysis.json")
        if a.reuse_analysis and cache.exists():
            frames = json.loads(cache.read_text())["frames"]
        else:
            frames = analyse(ff, film, info, first, count, shot_w, sources, cfg, sidecar)
            cache.write_text(json.dumps({"film": str(film), "first": first, "count": count, "frames": frames}))
        report["analysis_seconds"] = round(time.time() - t0, 2)
        report["analysis_fps"] = round(count / max(1e-6, report["analysis_seconds"]), 2)
        report["detections"] = int(sum(len(f["dets"]) for f in frames))
        if sidecar:
            plan = plan_sidecar(frames, sidecar, first, count, cfg)

            def ui_of_frame(i):
                return sidecar["f"][i].get("ui", [])
        else:
            plan, _ = plan_heuristic(frames, shot_w, cfg)

            def ui_of_frame(i):
                return heuristic_ui(info["width"], info["height"], i / fps, cards)
        report["tracks"] = [{k: v for k, v in p.items() if k not in ("lm", "w", "w_raw", "yaw")} |
                            {"frames_swapped": int((p["w"] > 0.004).sum()), "frames_full": int((p["w"] > 0.99).sum())}
                            for p in plan]
        stats = run_swap(ff, film, info, first, count, plan, sources, cfg, out, Path(a.side_by_side) if a.side_by_side else None,
                         ui_of_frame)
    if first == 0 and count >= info["frames"] > 0:          # a whole film: keep the recording's A/V relation
        report["video_start"] = restore_start(out, film)
    report.update({k: v for k, v in stats.items() if k != "per_frame"})
    report["per_frame"] = stats["per_frame"]
    report["swap_spans"] = swap_spans(plan, fps, first)
    report["weights"] = {f"{p['track']}:{p['role']}": {"k0": p["k0"], "w": np.round(p["w"], 3).tolist(),
                                                       "w_raw": np.round(p["w_raw"], 3).tolist()} for p in plan}
    rp = out.with_suffix(".report.json")
    rp.write_text(json.dumps(report, indent=1))
    print(f"[faceswap] {out} ({count} frames): analysis {report['analysis_fps']} fps, swap {report['swap_fps']} fps, "
          f"{report['faces_swapped']} faces, {report['model_ms_per_face']} ms per face; changed outside the faces: "
          f"{report['changed_outside_support']} px; overlay strokes kept over {report['overlay_restored_frames']} swapped frames "
          f"({report['overlay_restored_px']} px); report {rp}", flush=True)
    for t in report["tracks"]:
        print("   ", json.dumps(t), flush=True)
    for r, v in report["swap_spans"].items():
        print(f"    {NAMES[r]}: {v['n']} swap spans, shortest {v['shortest_s']} s, shortest gap {v['shortest_gap_s']} s: {v['spans_s']}", flush=True)


def cmd_canon_points(a):
    """face_points_canonical.json: the 5 swap points and the 52 iBUG points of each dancer's head in the splat avatar's canonical
    pose (Unity coordinates; the shaped SMPL-X zero pose of SPLAT_AVATAR_FORMAT.md section 4) + the generic head for the PnP."""
    model = Path(a.smplx)
    m = np.load(model, allow_pickle=True)
    f = np.asarray(m["f"], np.int64)
    st_f, st_b = np.asarray(m["lmk_faces_idx"], np.int64), np.asarray(m["lmk_bary_coords"], float)
    dyn_f, dyn_b = np.asarray(m["dynamic_lmk_faces_idx"], np.int64), np.asarray(m["dynamic_lmk_bary_coords"], float)
    mid = dyn_f.shape[0] // 2
    faces = np.concatenate([f[dyn_f[mid, 8]][None], f[st_f]], 0)
    bary = np.concatenate([dyn_b[mid, 8][None], st_b], 0)
    ibug = [8] + list(range(17, 68))

    def lmk(v):
        return np.einsum("lkd,lk->ld", v[faces], bary)

    def five(L):
        P = {i: L[ibug.index(i)] for i in ibug}
        return np.array([np.mean([P[i] for i in range(36, 42)], 0), np.mean([P[i] for i in range(42, 48)], 0), P[30], P[48], P[54]])

    out = {"format": "hm face points", "version": 1, "created": time.strftime("%Y-%m-%d %H:%M"),
           "landmarks5": list(LANDMARKS5), "ibug52": ibug,
           "coords": "per dancer: Unity (x, y, -z) of the splat avatar's canonical pose = the shaped SMPL-X zero pose (SPLAT_AVATAR_FORMAT.md 4); "
                     "generic: SMPL-X model coordinates (right-handed, y up, +z out of the face), betas 0",
           "skin": "rigid with the splat skeleton's head joint 15: posed = S_15(t) * p, S_j = G_j(t) T(-J_j) (the matrices the splat skinning uses)",
           "smplx_model": str(model), "roles": {}}
    v0 = m["v_template"]
    L0 = lmk(v0)
    out["generic"] = {"points52_smplx": np.round(L0, 6).tolist(), "points5_smplx": np.round(five(L0), 6).tolist()}
    flip = np.array([1.0, 1.0, -1.0])
    for role in ROLES:
        hdr = json.loads((Path(a.splats) / f"{role}.json").read_text())
        b = np.asarray(hdr["betas"], float)
        v = v0 + np.einsum("vdk,k->vd", m["shapedirs"][:, :, :len(b)], b)
        J = m["J_regressor"] @ v
        err = float(np.abs(J * flip - np.asarray(hdr["rest_joints"])).max())
        L = lmk(v) * flip
        P5 = five(L)
        fwd = np.array([0.0, 0.0, -1.0])            # SMPL-X +z -> Unity -z
        eye_mid = (P5[0] + P5[1]) / 2
        out["roles"][role] = {"name": hdr.get("name", NAMES[role]), "betas": b.tolist(), "head_joint": 15,
                              "J15": np.round(J[15] * flip, 6).tolist(), "rest_joint_check_m": err,
                              "points5": np.round(P5, 6).tolist(), "points52": np.round(L, 6).tolist(),
                              "forward": fwd.tolist(), "up": [0.0, 1.0, 0.0],
                              "head_centre": np.round(J[15] * flip + np.array([0.0, 0.07, -0.02]), 6).tolist(),
                              "samples9": np.round(np.concatenate([P5, [L[ibug.index(8)], eye_mid + [0, 0.05, 0.01],
                                                                        L[ibug.index(36)] + [-0.02, 0, 0.02],
                                                                        L[ibug.index(45)] + [0.02, 0, 0.02]]]), 6).tolist(),
                              "iod_m": round(float(np.linalg.norm(P5[1] - P5[0])), 5)}
        print(f"[faceswap] {role}: rest joints vs the splat file max {err * 1000:.3f} mm, iod {out['roles'][role]['iod_m'] * 1000:.1f} mm")
    out["samples9_names"] = list(LANDMARKS5) + ["chin", "forehead", "temple_r", "temple_l"]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"[faceswap] wrote {a.out}")


def cmd_mock_sidecar(a):
    """A sidecar for the frames of a heuristic run (its smoothed tracks + 1 px noise; yaw from the face; view from direction.json).
    Only to exercise the sidecar code path before the recorder writes real ones."""
    rep = json.loads(Path(a.report).read_text())
    info = ffprobe(Path(rep["film"]))
    an = json.loads(Path(rep["out"]).with_suffix(".analysis.json").read_text())
    cfg = Cfg()
    shot_w, cards = shot_weights(Path(a.direction), an["first"], an["count"], info["fps"], cfg)
    plan, _ = plan_heuristic(an["frames"], shot_w, cfg)
    rng = np.random.default_rng(7)
    n = info["frames"]
    fr = [{"t": round(i / info["fps"], 4), "view": "3d", "video_alpha": 0.0, "d": [], "ui": []} for i in range(n)]
    for k in range(an["count"]):
        i = an["first"] + k
        fr[i]["video_alpha"] = float(1.0 - shot_w[k])
        fr[i]["view"] = "3d" if shot_w[k] > 0.5 else "video"
        fr[i]["ui"] = heuristic_ui(info["width"], info["height"], i / info["fps"], cards)
        for p in plan:
            if p["role"] and p["k0"] <= k <= p["k1"]:
                lm = p["lm"][k - p["k0"]] + rng.normal(0, 1.0, (5, 2))
                yv = p["yaw"][k - p["k0"]]
                fr[i]["d"].append({"k": p["role"], "in": 1, "lm": np.round(lm, 1).tolist(), "yaw": float(0 if np.isnan(yv) else yv),
                                   "pitch": 0.0, "roll": 0.0, "iod": round(iod(lm), 1), "vis": 1.0, "by": ""})
    sc = {"format": SIDECAR_FORMAT, "version": 1, "recording": Path(rep["film"]).name, "mock": True, "width": info["width"],
          "height": info["height"], "fps": info["fps"], "frames": n, "landmarks": list(LANDMARKS5),
          "dancers": [{"key": r, "name": NAMES[r]} for r in ROLES], "f": fr}
    Path(a.out).write_text(json.dumps(sc))
    print(f"[faceswap] mock sidecar {a.out}: frames {an['first']}..{an['first'] + an['count'] - 1} filled")


def cmd_restore_start(a):
    """A whole-film output written before the fix: put the recording's video start back (report gets "video_start")."""
    rep = json.loads(Path(a.report).read_text())
    res = restore_start(Path(rep["out"]), Path(rep["film"]))
    rep["video_start"] = res
    Path(a.report).write_text(json.dumps(rep, indent=1))
    print(json.dumps(res))


def cmd_verify(a):
    """Decode source and output over the report's range: outside the swapped faces' boxes they must be equal (lossless output),
    else the PSNR of the encoding is reported."""
    rep = json.loads(Path(a.report).read_text())
    film, out = Path(rep["film"]), Path(rep["out"])
    info, oinfo = ffprobe(film), ffprobe(out)
    w, h = info["width"], info["height"]
    first, count = rep["first_frame"], rep["frames"]
    src = YuvReader(film, info, first, count)
    dst = YuvReader(out, oinfo, 0, count)
    diff_px, n_px, se, n_face_frames = 0, 0, 0.0, 0
    prev = None
    jit_src, jit_out, jit_n = 0.0, 0.0, 0
    for k, (a_buf, b_buf) in enumerate(zip(src, dst)):
        Ya, _, _ = planes(a_buf, w, h)
        Yb, _, _ = planes(b_buf, w, h)
        # flicker: frame-to-frame change inside a face box that is swapped in both frames, output vs source
        if prev is not None:
            pa, pb, pfaces = prev
            for fb in rep["per_frame"][k]["faces"]:
                if fb["w"] < 0.99 or not any(q["role"] == fb["role"] and q["w"] >= 0.99 for q in pfaces):
                    continue
                x1, y1, x2, y2 = fb["box"]
                jit_src += float(np.mean(np.abs(Ya[y1:y2, x1:x2].astype(np.int16) - pa[y1:y2, x1:x2])))
                jit_out += float(np.mean(np.abs(Yb[y1:y2, x1:x2].astype(np.int16) - pb[y1:y2, x1:x2])))
                jit_n += 1
        prev = (Ya.astype(np.int16), Yb.astype(np.int16), rep["per_frame"][k]["faces"])
        keep = np.ones((h, w), bool)
        faces = rep["per_frame"][k]["faces"]
        n_face_frames += bool(faces)
        for fb in faces:
            x1, y1, x2, y2 = fb["box"]
            keep[y1:y2, x1:x2] = False
        d = Ya[keep].astype(np.int16) - Yb[keep].astype(np.int16)
        diff_px += int(np.count_nonzero(d))
        se += float(np.sum(d.astype(np.float64) ** 2))
        n_px += int(keep.sum())
    mse = se / max(1, n_px)
    psnr = float("inf") if mse == 0 else 10 * math.log10(255 ** 2 / mse)
    res = {"frames": count, "frames_with_faces": n_face_frames, "luma_px_outside_faces": n_px, "differing_px": diff_px,
           "psnr_outside_faces_db": psnr, "face_pairs_full_weight": jit_n,
           "face_frame_change_src": round(jit_src / max(1, jit_n), 3), "face_frame_change_out": round(jit_out / max(1, jit_n), 3),
           "flicker_ratio_out_vs_src": round(jit_out / max(1e-6, jit_src), 3) if jit_n else None}
    print(json.dumps(res))
    rep["verify"] = res
    Path(a.report).write_text(json.dumps(rep, indent=1))


def cmd_sheet(a):
    """Before / after crops of the swapped faces at evenly spaced frames (and frames given with --frames)."""
    import cv2
    rep = json.loads(Path(a.report).read_text())
    film, out = Path(rep["film"]), Path(rep["out"])
    info, oinfo = ffprobe(film), ffprobe(out)
    w, h = info["width"], info["height"]
    first, count = rep["first_frame"], rep["frames"]
    pick = sorted(set(int(x) for x in np.linspace(0, count - 1, a.n)) | set(int(f) - first for f in (a.frames or [])))
    src = list(YuvReader(film, info, first, count))
    dst = list(YuvReader(out, oinfo, 0, count))
    dets_all = json.loads(out.with_suffix(".analysis.json").read_text())["frames"] if a.dets else None
    cells = []
    S = a.cell
    for k in pick:
        bs = cv2.cvtColor(src[k].reshape(h * 3 // 2, w), cv2.COLOR_YUV2BGR_I420)
        bd = cv2.cvtColor(dst[k].reshape(h * 3 // 2, w), cv2.COLOR_YUV2BGR_I420)
        faces = rep["per_frame"][k]["faces"]
        an = dets_all[k]["dets"] if dets_all else []
        rows = []
        targets = [(fb["box"], f"{NAMES[fb['role']]} w{fb['w']:.2f} b{fb['boost']}") for fb in faces]
        if not targets:
            for d in an:
                x1, y1, x2, y2 = d["box"]
                targets.append(([int(x1 - 0.3 * (x2 - x1)), int(y1 - 0.3 * (y2 - y1)), int(x2 + 0.3 * (x2 - x1)), int(y2 + 0.3 * (y2 - y1))],
                                f"not swapped yaw{d['yaw5']:.0f}"))
        thumb_s = cv2.resize(bs, (S * w // h, S))
        thumb_d = cv2.resize(bd, (S * w // h, S))
        row = [thumb_s, thumb_d]
        for box, label in targets[:2]:
            x1, y1, x2, y2 = box
            cx, cy, side = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1) * 1.1
            X1, Y1 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
            X2, Y2 = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
            for img in (bs, bd):
                c = cv2.resize(img[Y1:Y2, X1:X2], (S, S), interpolation=cv2.INTER_CUBIC)
                row.append(c)
            cv2.putText(row[-1], label, (4, S - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        while len(row) < 6:
            row.append(np.zeros((S, S, 3), np.uint8))
        line = np.hstack([cv2.copyMakeBorder(c, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(40, 40, 40)) for c in row])
        cv2.putText(line, f"film {(first + k) / info['fps']:.2f}s f{first + k}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        rows.append(line)
        cells.extend(rows)
    width = max(c.shape[1] for c in cells)
    sheet = np.vstack([cv2.copyMakeBorder(c, 0, 0, 0, width - c.shape[1], cv2.BORDER_CONSTANT) for c in cells])
    cv2.imwrite(a.out, sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"[faceswap] sheet {a.out}: {len(cells)} rows")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m dancecap.film_faceswap", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("swap")
    s.add_argument("--film", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--start", type=float)
    s.add_argument("--end", type=float)
    s.add_argument("--sidecar")
    s.add_argument("--direction", default=str(DIRECTION))
    s.add_argument("--side-by-side")
    s.add_argument("--phone-shots", choices=("off", "swap"), default="off")
    s.add_argument("--color-match", type=float, default=0.0)
    s.add_argument("--detector", default="yolo_face", choices=("yolo_face", "retinaface", "scrfd", "many"))
    s.add_argument("--crf", type=int, default=14)
    s.add_argument("--lossless", action="store_true")
    s.add_argument("--reuse-analysis", action="store_true")
    s.set_defaults(fn=cmd_swap)
    c = sub.add_parser("canon-points")
    c.add_argument("--smplx", default=str(Path("C:/Users/johnb/Desktop/HeadMovement/mamma/data/body_models/smplx_locked_head/smplx/SMPLX_NEUTRAL.npz")))
    c.add_argument("--splats", default=str(SPLATS))
    c.add_argument("--out", default=str(FACE_POINTS))
    c.set_defaults(fn=cmd_canon_points)
    m = sub.add_parser("mock-sidecar")
    m.add_argument("--report", required=True)
    m.add_argument("--direction", default=str(DIRECTION))
    m.add_argument("--out", required=True)
    m.set_defaults(fn=cmd_mock_sidecar)
    v = sub.add_parser("verify")
    v.add_argument("--report", required=True)
    v.set_defaults(fn=cmd_verify)
    rs = sub.add_parser("restore-start")
    rs.add_argument("--report", required=True)
    rs.set_defaults(fn=cmd_restore_start)
    sh = sub.add_parser("sheet")
    sh.add_argument("--report", required=True)
    sh.add_argument("--out", required=True)
    sh.add_argument("--n", type=int, default=8)
    sh.add_argument("--frames", type=int, nargs="*")
    sh.add_argument("--cell", type=int, default=256)
    sh.add_argument("--dets", action="store_true")
    sh.set_defaults(fn=cmd_sheet)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
