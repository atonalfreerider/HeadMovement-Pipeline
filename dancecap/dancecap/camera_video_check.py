"""Measure how well the exported phone tracks (camera_video.py) line the avatars up with the dancers in the phones' video.

    python -m dancecap.camera_video_check larissa_kadu_38 --capture 04_LarissaKaduDemo --cams 06,04,09 \
        [--out <review folder>] [--step 0.5] [--overlays 3]

For every sample time T (reference s) and phone: the avatars' body joints (the capture's figure1/figure2.json, Unity
capture coordinates) are projected through the exported track (the same pose/FOV Unity gets) into the phone's frame and
compared with the people detector's COCO-17 keypoints (work/<take>/people/<cam>.npz, sync-corrected) of the detection
that sits on the dancer's pelvis. The joint definitions differ (SMPL-X joint centres are inside the body, COCO points
on it: a few px of constant bias), so the absolute number has a floor; what the numbers show is
  - the error with the sync-corrected clocks vs. raw clocks vs. a +-N ms scan of the avatar time (best shift ~ 0 = in sync),
  - the error per 5 s bin (no drift over the span),
  - the error per phone (rotation / handedness / zoom: a mirrored or rotated frame would give hundreds of px).
Writes <out>/campov_check.json and overlay stills campov_py_<cam>_<t>.png (avatar skeleton on the phone's own frame).

    python -m dancecap.camera_video_check larissa_kadu_38 --capture 04_LarissaKaduDemo --coverage
        regression sweep (no Unity needed): for every shot of the capture's direction(s) that flies into a phone, a pose at
        every 1/30 s of the whole shot and a video frame within 0.25 s at every step of the video window; exit 1 on any miss.
    python -m dancecap.camera_video_check <take> --capture <c> --cams 06 --recording <mp4> --film-start .. --map .. --window ..
        frame-lock check of a recorded film; a recorded frame that correlates < 0.5 with EVERY source frame of the phone is a
        FAIL (the video blinked out, a black frame, the wrong phone), never "overlays covering the picture".
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from . import camera_video as cv
from . import mamma_dyn as md

# COCO-17 index -> SMPL-X joint (figure json order): shoulders, elbows, wrists, hips, knees, ankles
COCO_TO_SMPL = {5: 16, 6: 17, 7: 18, 8: 19, 9: 20, 10: 21, 11: 1, 12: 2, 13: 4, 14: 5, 15: 7, 16: 8}
MIN_CORR = 0.5     # recorded picture vs. the best-matching source frame: below this the video is missing in that frame
SKELETON = [(1, 4), (4, 7), (2, 5), (5, 8), (0, 1), (0, 2), (0, 3), (3, 6), (6, 9), (9, 12), (12, 15), (9, 13), (9, 14),
            (13, 16), (14, 17), (16, 18), (17, 19), (18, 20), (19, 21)]


def load_track(path: Path) -> dict:
    d = json.loads(path.read_text())
    return {"id": d["id"], "size": d["size"], "t": np.array(d["t"]), "pos": np.array(d["pos"]), "fwd": np.array(d["fwd"]),
            "up": np.array(d["up"]), "vfov": np.array(d["vfov"])}


def pose_at(tr: dict, t: float):
    i = int(np.clip(np.searchsorted(tr["t"], t), 1, len(tr["t"]) - 1))
    a, b = tr["t"][i - 1], tr["t"][i]
    k = float(np.clip((t - a) / max(b - a, 1e-9), 0, 1))
    lerp = lambda x: x[i - 1] * (1 - k) + x[i] * k  # noqa: E731
    fwd, up = lerp(tr["fwd"]), lerp(tr["up"])
    fwd /= np.linalg.norm(fwd)
    up = up - fwd * np.dot(up, fwd)
    up /= np.linalg.norm(up)
    return lerp(tr["pos"]), fwd, up, float(lerp(tr["vfov"]))


def project(p: np.ndarray, pos, fwd, up, vfov: float, size) -> np.ndarray:
    """Unity (left-handed) camera: right = up x forward. Pixel coordinates, y down, principal point at the centre."""
    W, H = size
    right = np.cross(up, fwd)
    v = p - pos
    zc = v @ fwd
    f = (H / 2) / np.tan(np.radians(vfov) / 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = (W - 1) / 2 + f * (v @ right) / zc
        w = (H - 1) / 2 - f * (v @ up) / zc
    out = np.stack([u, w], -1)
    out[zc <= 0.05] = np.nan
    return out


def assign(kp: np.ndarray, pelvis: dict, H: int) -> dict:
    """role -> detection index: nearest mid-hip to the projected pelvis, one detection per dancer (greedy by distance;
    a dancer without its own detection within 5 % of the frame height is dropped: embraces merge detections)."""
    mid = (kp[:, 11, :2] + kp[:, 12, :2]) / 2
    cand = []
    for role, p in pelvis.items():
        if not np.isfinite(p).all():
            continue
        d = np.linalg.norm(mid - p, axis=1)
        cand += [(float(d[i]), role, i) for i in range(len(d)) if d[i] < 0.05 * H]
    out, used = {}, set()
    for d, role, i in sorted(cand):
        if role not in out and i not in used:
            out[role] = i
            used.add(i)
    return out


class Figures:
    def __init__(self, capture: str):
        d = cv.UNITY_STREAMING / capture
        self.t = np.array(json.loads((d / "times.json").read_text()))
        self.pose = {}
        for role, fn in (("lead", "figure1.json"), ("follow", "figure2.json")):
            fr = json.loads((d / fn).read_text())
            self.pose[role] = np.array([[[j["x"], j["y"], j["z"]] for j in f] for f in fr])

    def at(self, role: str, t: float) -> np.ndarray:
        i = int(np.clip(np.searchsorted(self.t, t), 1, len(self.t) - 1))
        k = float(np.clip((t - self.t[i - 1]) / (self.t[i] - self.t[i - 1]), 0, 1))
        return self.pose[role][i - 1] * (1 - k) + self.pose[role][i] * k


def measure(take: str, capture: str, cams: list[str], step: float, out: Path, overlays: int, master: Path | None,
            log=print) -> dict:
    master = master or (md.take_work(take) / "camera_video" / capture)
    fig = Figures(capture)
    t_lo, t_hi = float(fig.t[0]) + 0.5, float(fig.t[-1]) - 0.5
    work = md.take_work(take)
    offs = md.read_sync_offsets(work)
    times = np.arange(t_lo, t_hi, step)
    shifts = np.arange(-120, 121, 10) / 1000.0
    result = {"take": take, "capture": capture, "step_s": step, "cameras": {}}
    for cam in cams:
        tr = load_track(master / f"{cam}.track.json")
        pz = np.load(work / "people" / f"{cam}.npz")
        det_t = md.apply_offset(pz["t"], offs.get(cam))
        raw_t = pz["t"]
        errs = {"corrected": [], "raw_clock": []}
        scan = {round(float(s) * 1000): [] for s in shifts}
        rows = []
        for T in times:
            j = int(np.argmin(np.abs(det_t - T)))
            Td = float(det_t[j])
            if abs(Td - T) > 0.04:
                continue
            valid = pz["valid"][j]
            kp = pz["kpts"][j][valid]
            if not len(kp):
                continue
            for label, t_av in (("corrected", Td), ("raw_clock", float(raw_t[j]))):
                # the camera pose and the detections belong to this frame either way; raw_clock poses the avatars at
                # the frame's UNcorrected clock time (what ignoring the per-phone video offset would give)
                if label == "raw_clock" and not (offs.get(cam) and abs(offs[cam]["delta"]) > 0.002):
                    continue
                pos, fwd, up, vfov = pose_at(tr, Td)
                pxs = {role: project(fig.at(role, t_av), pos, fwd, up, vfov, tr["size"]) for role in ("lead", "follow")}
                got = assign(kp, {r: p[0] for r, p in pxs.items()}, tr["size"][1])
                for role, c in got.items():
                    px = pxs[role]
                    e = []
                    for ci, si in COCO_TO_SMPL.items():
                        if kp[c, ci, 2] >= 0.5 and np.isfinite(px[si]).all():
                            e.append(float(np.linalg.norm(kp[c, ci, :2] - px[si])))
                    if len(e) < 6:
                        continue
                    errs[label].extend(e)
                    if label == "corrected":
                        rows.append({"t": round(Td, 3), "role": role, "median_px": round(float(np.median(e)), 1), "n": len(e)})
                        for s_ in shifts:
                            pxs2 = project(fig.at(role, Td + float(s_)), pos, fwd, up, vfov, tr["size"])
                            es = [float(np.linalg.norm(kp[c, ci, :2] - pxs2[si])) for ci, si in COCO_TO_SMPL.items()
                                  if kp[c, ci, 2] >= 0.5 and np.isfinite(pxs2[si]).all()]
                            scan[round(float(s_) * 1000)].extend(es)
        st = lambda a: {"n": len(a), "median": round(float(np.median(a)), 1), "mean": round(float(np.mean(a)), 1),  # noqa: E731
                        "p90": round(float(np.percentile(a, 90)), 1)} if a else None
        bins = {}
        for r in rows:
            bins.setdefault(int((r["t"] - fig.t[0]) // 5) * 5, []).append(r["median_px"])
        best = min(scan, key=lambda s: np.median(scan[s]) if scan[s] else 1e9)
        xs = np.array([s_ for s_ in scan if scan[s_] and abs(s_) <= 60], float)
        ys = np.array([np.mean(scan[int(s_)]) for s_ in xs])          # mean error: smooth in the shift
        fit = float(-np.polyfit(xs, ys, 2)[1] / (2 * np.polyfit(xs, ys, 2)[0])) if len(xs) >= 5 else float("nan")
        entry = {"size": tr["size"], "corrected": st(errs["corrected"]), "raw_clock": st(errs["raw_clock"]),
                 "video_offset_ms": round(1000 * (offs.get(cam) or {}).get("delta", 0.0), 1),
                 "shift_scan_median_px": {str(s): round(float(np.median(v)), 1) for s, v in scan.items() if v},
                 "best_shift_ms": best, "best_shift_parabola_ms": round(fit, 1), "by_5s_bin_median_px": {str(k): round(float(np.median(v)), 1) for k, v in sorted(bins.items())}}
        result["cameras"][cam] = entry
        log(f"[check] {cam}: corrected median {entry['corrected']['median']} px (p90 {entry['corrected']['p90']}, n {entry['corrected']['n']})"
            + (f", raw clock {entry['raw_clock']['median']} px" if entry["raw_clock"] else "")
            + f", best avatar shift {best} ms (parabola {fit:+.1f} ms); per 5 s {entry['by_5s_bin_median_px']}")
        # overlay stills on the phone's own frames
        full = master / f"{cam}.full.hmj"
        if overlays and full.exists():
            recs, fw, fh = cv.read_hmj(full)
            rt = np.array([r[2] for r in recs])
            for T in np.linspace(t_lo + 2, t_hi - 2, overlays):
                k = int(np.argmin(np.abs(rt - T)))
                img = cv.hmj_frame(full, recs[k])
                s = fw / tr["size"][0]
                pos, fwd, up, vfov = pose_at(tr, float(rt[k]))
                for role, col in (("lead", (255, 200, 0)), ("follow", (200, 0, 255))):
                    px = project(fig.at(role, float(rt[k])), pos, fwd, up, vfov, tr["size"]) * s
                    for a, b in SKELETON:
                        if np.isfinite(px[a]).all() and np.isfinite(px[b]).all():
                            cv2.line(img, tuple(int(v) for v in px[a]), tuple(int(v) for v in px[b]), col, max(2, fw // 240))
                    for p in px:
                        if np.isfinite(p).all():
                            cv2.circle(img, (int(p[0]), int(p[1])), max(3, fw // 200), (255, 255, 255), -1)
                cv2.putText(img, f"{cam} t={rt[k]:.2f}s", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                out.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out / f"campov_py_{cam}_{rt[k]:06.2f}.png"), img)
    out.mkdir(parents=True, exist_ok=True)
    (out / "campov_check.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result


def compare_unity(take: str, capture: str, sweep: Path, master: Path | None, log=print) -> dict:
    """Unity's own projections (hm_camvideo --action sweep: the exported track loaded by FilmCameraVideo, the avatars'
    joints as the viewer plays them with DanceOrigin applied, a Camera component) vs. this module's closed form on the
    JSON figures, and vs. the people detector."""
    master = master or (md.take_work(take) / "camera_video" / capture)
    d = json.loads(sweep.read_text())
    cam = d["cam"]
    tr = load_track(master / f"{cam}.track.json")
    fig = Figures(capture)
    work = md.take_work(take)
    offs = md.read_sync_offsets(work)
    pz = np.load(work / "people" / f"{cam}.npz")
    det_t = md.apply_offset(pz["t"], offs.get(cam))
    diffs, errs, pending = [], [], {}
    for row in d["rows"]:
        T, role = float(row["t"]), row["role"]
        pos, fwd, up, vfov = pose_at(tr, T)
        py = project(fig.at(role, T), pos, fwd, up, vfov, tr["size"])
        for k, v in row["joints"].items():
            diffs.append(float(np.linalg.norm(py[int(k)] - np.array(v[:2]))))
        pending.setdefault(T, {})[role] = row["joints"]
    for T, roles in pending.items():
        j = int(np.argmin(np.abs(det_t - T)))
        if abs(det_t[j] - T) > 0.04:
            continue
        kp = pz["kpts"][j][pz["valid"][j]]
        if not len(kp):
            continue
        got = assign(kp, {r: np.array(jt["0"][:2]) for r, jt in roles.items() if "0" in jt}, tr["size"][1])
        for role, c in got.items():
            for ci, si in COCO_TO_SMPL.items():
                v = roles[role].get(str(si))
                if v is not None and kp[c, ci, 2] >= 0.5:
                    errs.append(float(np.linalg.norm(kp[c, ci, :2] - np.array(v[:2]))))
    out = {"cam": cam, "n_joints": len(diffs), "unity_vs_python_px": {"max": round(max(diffs), 4), "mean": round(float(np.mean(diffs)), 4)},
           "pinhole_vs_camera_component_max_px": d.get("pinhole_vs_camera_component_max_px"),
           "unity_vs_detector_px": {"n": len(errs), "median": round(float(np.median(errs)), 1), "mean": round(float(np.mean(errs)), 1),
                                    "p90": round(float(np.percentile(errs, 90)), 1)} if errs else None}
    log(f"[check] unity sweep {cam}: {out}")
    return out


def compare_view(take: str, capture: str, view: Path, master: Path | None, log=print) -> dict:
    """hm_camvideo --action viewproject: the avatars' joints through the film's RENDER camera (as the director left it on
    a POV shot) -> viewport coordinates -> the phone's frame (the video is drawn at full height, centred): compared with
    the closed form through the phone's own track and with the people detector."""
    master = master or (md.take_work(take) / "camera_video" / capture)
    d = json.loads(view.read_text())
    cam = d["phone"]
    tr = load_track(master / f"{cam}.track.json")
    W, H = tr["size"]
    A = float(d["aspect"])
    T = float(d["dance"])
    fig = Figures(capture)
    pos, fwd, up, vfov = pose_at(tr, T)
    work = md.take_work(take)
    offs = md.read_sync_offsets(work)
    pz = np.load(work / "people" / f"{cam}.npz")
    det_t = md.apply_offset(pz["t"], offs.get(cam))
    j = int(np.argmin(np.abs(det_t - T)))
    kp = pz["kpts"][j][pz["valid"][j]]
    own, errs, pend = [], [], {}
    for role in ("lead", "follow"):
        joints = d["roles"][role]
        px = {int(k): np.array([(W - 1) / 2 + H * A * (v[0] - 0.5), (H - 1) / 2 + H * (0.5 - v[1])]) for k, v in joints.items() if v[2] > 0.05}
        py = project(fig.at(role, T), pos, fwd, up, vfov, tr["size"])
        for k, v in px.items():
            own.append(float(np.linalg.norm(v - py[k])))
        pend[role] = px
    got = assign(kp, {r: p[0] for r, p in pend.items() if 0 in p}, H) if len(kp) else {}
    for role, c in got.items():
        for ci, si in COCO_TO_SMPL.items():
            if si in pend[role] and kp[c, ci, 2] >= 0.5:
                errs.append(float(np.linalg.norm(kp[c, ci, :2] - pend[role][si])))
    out = {"phone": cam, "dance_s": T, "film_s": d.get("film"), "segment": d.get("segment"), "render_camera": {
        "fov": d["fov"], "aspect": A, "pixels": [d["pixelWidth"], d["pixelHeight"]], "lensShift": d["lensShift"],
        "camera_minus_phone_m": d.get("cameraMinusPhone_m"), "angle_to_phone_deg": d.get("cameraAngleToPhone_deg"),
        "fov_minus_phone_deg": d.get("fovMinusPhone_deg")},
        "render_vs_phone_track_px": {"n": len(own), "max": round(max(own), 3), "mean": round(float(np.mean(own)), 3)} if own else None,
        "render_vs_detector_px": {"n": len(errs), "median": round(float(np.median(errs)), 1), "mean": round(float(np.mean(errs)), 1),
                                  "p90": round(float(np.percentile(errs, 90)), 1)} if errs else None}
    log(f"[check] unity view {cam} @ {T:.2f}s: {out}")
    return out


def check_recording(mp4: Path, capture: str, cam: str, film_start: float, mapping: tuple[float, float, float],
                    film_window: tuple[float, float], master: Path | None, step: int = 5, band: tuple[float, float] = (0.30, 0.62),
                    log=print) -> dict:
    """End-to-end frame-lock test on a RECORDED film (Unity Recorder, frame-locked 30 fps): for sampled video frames inside
    the film window where the phone's video is at full opacity, find which of the phone's source frames (the exported
    strip) the recorded picture matches best (normalised correlation over the middle rows, where the dancers are) and
    compare with the frame the dance clock predicts. mapping = (film s, dance s, speed): dance = d0 + (film - f0) * speed."""
    master = master or (md.take_work(take_of(capture)) / "camera_video" / capture)
    recs, fw, fh = cv.read_hmj(master / f"{cam}.full.hmj")
    rt = np.array([r[2] for r in recs])
    f0, d0, spd = mapping
    cap = cv2.VideoCapture(str(mp4))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out_rows = []
    n = -1
    n0, n1 = int(round((film_window[0] - film_start) * fps)), int(round((film_window[1] - film_start) * fps))
    cache: dict[int, np.ndarray] = {}

    def strip_gray(k: int, h: int, w: int, y0: int, y1: int) -> np.ndarray:
        if k not in cache:
            img = cv.hmj_frame(master / f"{cam}.full.hmj", recs[k])
            g = cv2.cvtColor(cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)[y0:y1].astype(np.float32)
            cache[k] = (g - g.mean()) / (g.std() + 1e-6)
        return cache[k]

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        if n < n0 or n > n1 or (n - n0) % step:
            continue
        film = film_start + n / fps
        dance = d0 + (film - f0) * spd
        h, w = frame.shape[:2]
        y0, y1 = int(band[0] * h), int(band[1] * h)
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[y0:y1].astype(np.float32)
        g = (g - g.mean()) / (g.std() + 1e-6)
        k0 = int(np.argmin(np.abs(rt - dance)))
        best, score = None, -2.0
        for k in range(max(0, k0 - 8), min(len(rt), k0 + 9)):
            sc = float((g * strip_gray(k, h, w, y0, y1)).mean())
            if sc > score:
                best, score = k, sc
        out_rows.append({"film": round(film, 3), "dance": round(dance, 3), "expected": k0, "best": best, "offset_frames": best - k0,
                         "corr": round(score, 3), "time_error_ms": round(float(rt[best] - dance) * 1000, 1)})
    cap.release()
    off = np.array([r["offset_frames"] for r in out_rows])
    # a sample whose picture matches NO source frame of the phone is a failure (the video blinked out, a black frame, a
    # wrong phone), never "overlays covering the picture": the first version explained three such samples away
    low = [r for r in out_rows if r["corr"] < MIN_CORR]
    res = {"mp4": str(mp4), "frames_in_file": n_frames, "samples": len(out_rows),
           "min_corr_required": MIN_CORR, "low_corr_samples": [{"film": r["film"], "dance": r["dance"], "corr": r["corr"]} for r in low],
           "verdict": "PASS" if out_rows and not low else "FAIL",
           "exact_fraction": round(float((off == 0).mean()), 3) if len(off) else None,
           "within_1_frame": round(float((np.abs(off) <= 1).mean()), 3) if len(off) else None,
           "mean_offset_frames": round(float(off.mean()), 3) if len(off) else None,
           "median_corr": round(float(np.median([r["corr"] for r in out_rows])), 3) if out_rows else None,
           "mean_abs_time_error_ms": round(float(np.mean([abs(r["time_error_ms"]) for r in out_rows])), 1) if out_rows else None,
           "rows": out_rows}
    log(f"[check] recording {mp4.name}: {len(out_rows)} samples, exact {res['exact_fraction']}, within 1 frame {res['within_1_frame']}, "
        f"mean offset {res['mean_offset_frames']} frames, median corr {res['median_corr']}, verdict {res['verdict']}")
    for r in low:
        log(f"[check]   FAIL film {r['film']} s (dance {r['dance']} s): corr {r['corr']} < {MIN_CORR}: the picture matches no frame of phone {cam}")
    return res


def coverage(capture: str, master: Path, direction: list[str] | None, out: Path | None, step: float = 1.0 / 30.0, log=print) -> dict:
    """Regression sweep without Unity: for every shot of the direction(s) that flies into a phone, a pose must exist at every
    1/30 s step of the whole shot and a video frame within 0.25 s at every step of the video window (the Python mirror of
    CameraVideoTrack.TryAt + CameraVideoRig.SetVideo; the Unity side is `hm_camvideo --action coverage`)."""
    files = cv.direction_files(capture, direction)
    windows = cv.direction_pov_windows(files)
    rep = cv.coverage_report(master, windows, step)
    ok = cv.log_coverage(rep, log)
    res = {"capture": capture, "master": str(master), "step_s": step, "files": [p.name for p in files], "windows": len(rep), "ok": ok,
           "checks": rep}
    if out:
        out.mkdir(parents=True, exist_ok=True)
        (out / "campov_coverage.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def take_of(capture: str) -> str:
    return "larissa_kadu_38" if capture == "04_LarissaKaduDemo" else capture


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m dancecap.camera_video_check", description=__doc__.split("\n\n")[0])
    ap.add_argument("take")
    ap.add_argument("--capture", required=True)
    ap.add_argument("--cams", default=None, help="comma-separated phones (required for the measurement)")
    ap.add_argument("--out", type=Path, default=cv.PACKAGE_ROOT / "work" / "larissa_kadu_full" / "film" / "review")
    ap.add_argument("--master", type=Path, default=None)
    ap.add_argument("--step", type=float, default=0.5)
    ap.add_argument("--overlays", type=int, default=3)
    ap.add_argument("--unity-sweep", type=Path, default=None, help="compare a Unity hm_camvideo sweep json instead")
    ap.add_argument("--unity-view", type=Path, default=None, help="compare a Unity hm_camvideo viewproject json instead")
    ap.add_argument("--coverage", action="store_true", help="pose + video-frame coverage of every POV shot of the direction(s), 1/30 s steps")
    ap.add_argument("--direction", nargs="*", default=["auto"], metavar="JSON", help="--coverage: direction files (default: the capture's)")
    ap.add_argument("--recording", type=Path, default=None, help="frame-lock check of a recorded mp4 (needs --film-start, --map, --window)")
    ap.add_argument("--film-start", type=float, default=0.0, help="film second of the recording's first frame")
    ap.add_argument("--map", default=None, help="film_s,dance_s,speed: dance = dance_s + (film - film_s) * speed")
    ap.add_argument("--window", default=None, help="film_s_a,film_s_b: where the phone's video is at full opacity")
    a = ap.parse_args(argv)
    if a.coverage:
        master = a.master or (md.take_work(a.take) / "camera_video" / a.capture)
        res = coverage(a.capture, master, a.direction, a.out)
        sys.exit(0 if res["ok"] and res["windows"] else 1)
    if a.recording:
        cam = a.cams.split(",")[0] if a.cams else "06"
        m = tuple(float(x) for x in a.map.split(","))
        wdw = tuple(float(x) for x in a.window.split(","))
        res = check_recording(a.recording, a.capture, cam, a.film_start, m, wdw, a.master)
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / f"campov_recording_{cam}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        return
    if a.unity_view:
        compare_view(a.take, a.capture, a.unity_view, a.master)
        return
    if a.unity_sweep:
        compare_unity(a.take, a.capture, a.unity_sweep, a.master)
        return
    if not a.cams:
        ap.error("--cams is required")
    measure(a.take, a.capture, [c.strip() for c in a.cams.split(",")], a.step, a.out, a.overlays, a.master)


if __name__ == "__main__":
    main(sys.argv[1:])
