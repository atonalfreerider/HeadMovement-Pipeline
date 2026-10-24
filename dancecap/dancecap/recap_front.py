"""Recap front end of ONE chunk (agent R3): everything between the source videos and `mamma prepare`, for the class recap.

The demo's front end (front_full / front_pod) localises its cameras against the room map, finds the dancers with a colour rule and SAM. For the
recap those two questions are ALREADY answered by files this job brings along:

    cameras      R2's per-frame camera tracks of the whole class (cameras/<cam>.npz, room frame, 4 localised + 2 attached phones)
    teachers     R3's per-camera per-frame teacher tracks (people/teachers_<cam>.npz: the JSON person of Kadu / Larissa, box, joints, confidence)
    poses        the owner's 2D pose JSON as pose tables (people/pose_tables/<cam>.npz)

so this module only slices them to the chunk and feeds the unchanged demo stages with them:

    frames    the canonical frames of R2's tracks inside the chunk window (+ 2 s) decoded from the source videos at native resolution
              (frames/<cam>/NNNNNN.jpg + frames.json with the audio clocks; the SAME frames the camera tracks and the teacher tracks are indexed by)
    cameras   cameras/<cam>.npz sliced from R2's tracks (+ world_from_sfm.json)
    people    people/<cam>.npz (boxes, scores, valid, kpts of every JSON person of the chunk's frames) and dancers/teacher_tracks/<cam>.npz
    dancers   SAM 2.1 masks per role: the demo's front_pod.dancer_one with `candidate_tracks` replaced by the teacher tracks and the colour
              rule's role decision switched off (the identity is the teachers' 3D tracks + colour Viterbi of recap_people)
    ma2d      MAMMA ma_2d on the body masks (front_pod ma2d)
    merge     dancers_pod merge -> dancers/roles.json
    sync      dancecap sync on the chunk, re-gauged so that the MASTER camera (01) has offset 0
    prepare   mamma prepare --skip-2d --coverage union --grid-anchor 0 --grid-k=k0:k1 --set cN=<cameras of the chunk>

    python -m dancecap.recap_front run <chunk id>          the whole DAG (front_pod.Runner: memory budget, per-kind caps, resumable)
    python -m dancecap.recap_front frames|cameras|people|dancer|sync <chunk id> [cam]   one step

Nothing here is a neural network except what the demo's stages already run (SAM 2.1, MAMMA ma_2d).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from .take import PACKAGE_ROOT

DANCE = os.environ.get("HM_FRONT_DANCE", "larissa_kadu_recap")     # C3 (additive): carlos_aline sets it in its job env
GRID_ANCHOR = 0.0
FRONT_MARGIN_S = 2.0           # frames are decoded this much beyond the fit window (the same margin as farm.chunks.FRONT_MARGIN_S)
MASTER = os.environ.get("HM_FRONT_MASTER", "01")   # LL2V (additive): the camera gauged to offset 0; lucas_luiza's jobs set 02 (the reference clock)


def log(msg: str) -> None:
    print(f"[recap_front {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def dance_work() -> Path:
    return PACKAGE_ROOT / "work" / DANCE


def take_name(chunk_id: str) -> str:
    return f"{DANCE}_{chunk_id}"


def take_work(chunk_id: str) -> Path:
    return PACKAGE_ROOT / "work" / take_name(chunk_id)


def load_chunk(chunk_id: str, job_path: str | None = None) -> dict:
    """The farm job spec of the chunk ($FARM_JOB, farm/jobs/<id>.json, volume queue/<id>.json) -> its chunk dict + cams + window seconds."""
    from .farm.chunks import load_job
    job = load_job(chunk_id, DANCE, job_path)
    c = dict(job["chunk"])
    c["cams"] = list(job["cams"])
    c["t0"] = float(c["k0"]) / 30.0 - FRONT_MARGIN_S
    c["t1"] = float(c["k1"] + 1) / 30.0 + FRONT_MARGIN_S
    return c


# ------------------------------------------------------------------------------------------------ slicing helpers (pure numpy)


def frame_range(t: np.ndarray, t0: float, t1: float) -> np.ndarray:
    """Indices of the canonical frames whose master time lies in [t0, t1] (t is increasing)."""
    return np.flatnonzero((t >= t0) & (t <= t1))


def slice_npz(path: Path, idx: np.ndarray) -> dict:
    """R2's cameras/<cam>.npz restricted to the frames idx: every array with the frame axis first is cut, the rest (scalars, strings,
    sizes) is kept as it is."""
    z = np.load(path, allow_pickle=True)
    F = len(z["t"])
    out = {}
    for k in z.files:
        v = z[k]
        out[k] = v[idx] if (v.ndim >= 1 and v.shape[0] == F) else v
    return out


def people_arrays(kp: np.ndarray, n: np.ndarray, rows: np.ndarray) -> dict:
    """The JSON pose table rows `rows` (video frame index of each canonical frame) as the contract's people/<cam>.npz: boxes (F,N,4), scores
    (F,N) (mean confidence of the joints above 0.3), valid (F,N), kpts (F,N,17,3). Person order = the JSON's, so a row of a teacher track
    indexes it directly."""
    from .recap_people import person_box
    F = len(rows)
    N = int(max(1, n[rows].max())) if F else 1
    boxes = np.zeros((F, N, 4), np.float32)
    scores = np.zeros((F, N), np.float32)
    valid = np.zeros((F, N), bool)
    kpts = np.zeros((F, N, 17, 3), np.float32)
    for i, r in enumerate(rows):
        for j in range(int(n[r])):
            p = kp[r, j]
            kpts[i, j] = np.nan_to_num(p, nan=0.0)
            b = person_box(p)
            if np.isfinite(b).all():
                boxes[i, j] = b
                valid[i, j] = True
                c = p[:, 2]
                scores[i, j] = float(np.mean(c[c >= 0.3])) if (c >= 0.3).any() else 0.0
    return {"boxes": boxes, "scores": scores, "valid": valid, "kpts": kpts}


def teacher_slots(tt: dict, idx: np.ndarray, min_conf: float = 0.3) -> dict:
    """Teacher tracks of the chunk's frames in the demo's candidate-track layout: slot A = the LEAD (Kadu), slot B = the FOLLOW (Larissa),
    idx (F,2) the person row (-1 = MISS), boxes (F,2,4) (NaN where missing). Associations below min_conf are dropped (a person the role
    decision or the geometry doubts is better absent than wrong)."""
    row = tt["row"][idx].astype(int).copy()
    conf = tt["conf"][idx]
    row[conf < min_conf] = -1
    box = tt["box"][idx].astype(np.float32).copy()
    box[row < 0] = np.nan
    return {"idx": row, "boxes": box, "conf": np.where(row >= 0, conf, 0.0).astype(np.float32)}


# ------------------------------------------------------------------------------------------------ steps


def _cams_table() -> dict:
    return json.loads((dance_work() / "cameras" / "recap_cams.json").read_text())["cams"]


def _root() -> Path | None:
    r = os.environ.get("HM_ROOT")
    return Path(r) if r else None


def step_frames(chunk: dict, workers: int = 3) -> dict:
    """Decode the canonical frames of the chunk window of every camera to frames/<cam>/NNNNNN.jpg (native resolution, JPEG q95)."""
    import cv2

    from concurrent.futures import ThreadPoolExecutor

    from .cameras.newcam import FrameStream, video_pts
    from .recap_people import video_path
    work = take_work(chunk["id"])
    table = _cams_table()

    def one(cam: str) -> tuple[str, int]:
        z = np.load(dance_work() / "cameras" / f"{cam}.npz", allow_pickle=True)
        idx = frame_range(np.asarray(z["t"], float), chunk["t0"], chunk["t1"])
        out = work / "frames" / cam
        out.mkdir(parents=True, exist_ok=True)
        if not len(idx):
            (out / "frames.json").write_text(json.dumps({"camera": cam, "frame_count": 0, "frames": []}))
            return cam, 0
        row = table[cam]
        fi, pts, tt = np.asarray(z["frame_index"], int)[idx], np.asarray(z["pts"], float)[idx], np.asarray(z["t"], float)[idx]
        path = video_path(cam)
        allpts = video_pts(path, work / "frames" / f"{cam}_pts.json")
        w, h = int(row["width"]), int(row["height"])
        want = {int(n): k for k, n in enumerate(fi)}
        done = np.zeros(len(idx), bool)
        fs = FrameStream(path, allpts, float(pts[0]) - 0.02, float(pts[-1]) + 0.02, (w, h), want=set(want), every=1)
        for n, p, img, _ in fs:
            k = want[n]
            cv2.imwrite(str(out / f"{k:06d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            done[k] = True
        if not done.all():
            raise RuntimeError(f"{cam}: {int((~done).sum())} of {len(idx)} frames could not be decoded")
        dts = np.diff(pts)
        meta = {"camera": cam, "source": row["file"], "width": w, "height": h, "rotation": int(row.get("rotation", 0)),
                "slope": float(row["slope"]), "offset": float(row["offset"]), "frame_count": int(len(idx)),
                "fps_median": float(1 / np.median(dts)) if len(dts) else 0.0,
                "frame_interval_ms": {"min": float(dts.min() * 1000), "max": float(dts.max() * 1000)} if len(dts) else {},
                "frames": [{"file": f"{k:06d}.jpg", "pts": float(pts[k]), "t": float(tt[k]), "n": int(fi[k])} for k in range(len(idx))]}
        (out / "frames.json").write_text(json.dumps(meta, indent=1))
        return cam, len(idx)

    with ThreadPoolExecutor(workers) as ex:
        res = dict(ex.map(one, chunk["cams"]))
    log(f"frames: {res}")
    return res


def step_cameras(chunk: dict) -> dict:
    """cameras/<cam>.npz of the chunk = R2's tracks cut to the chunk's frames (+ world_from_sfm.json)."""
    work = take_work(chunk["id"])
    out = work / "cameras"
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    for cam in chunk["cams"]:
        src = dance_work() / "cameras" / f"{cam}.npz"
        z = np.load(src, allow_pickle=True)
        idx = frame_range(np.asarray(z["t"], float), chunk["t0"], chunk["t1"])
        np.savez_compressed(out / f"{cam}.npz", **slice_npz(src, idx))
        res[cam] = int(len(idx))
    shutil.copyfile(dance_work() / "cameras" / "world_from_sfm.json", out / "world_from_sfm.json")
    log(f"cameras: {res}")
    return res


def step_people(chunk: dict) -> dict:
    """people/<cam>.npz and dancers/teacher_tracks/<cam>.npz of the chunk from the pose tables and the teacher tracks."""
    from .recap_people import PEOPLE
    work = take_work(chunk["id"])
    (work / "people").mkdir(parents=True, exist_ok=True)
    (work / "dancers" / "teacher_tracks").mkdir(parents=True, exist_ok=True)
    res = {}
    for cam in chunk["cams"]:
        z = np.load(dance_work() / "cameras" / f"{cam}.npz", allow_pickle=True)
        idx = frame_range(np.asarray(z["t"], float), chunk["t0"], chunk["t1"])
        tb = np.load(PEOPLE / "pose_tables" / f"{cam}.npz")
        rows = np.asarray(z["frame_index"], int)[idx]
        np.savez_compressed(work / "people" / f"{cam}.npz", **people_arrays(tb["kp"], tb["n"], rows))
        tt = dict(np.load(PEOPLE / f"teachers_{cam}.npz"))
        ts = teacher_slots(tt, idx)
        np.savez_compressed(work / "dancers" / "teacher_tracks" / f"{cam}.npz", **ts)
        res[cam] = {"frames": int(len(idx)), "lead": round(float((ts["idx"][:, 0] >= 0).mean()), 3), "follow": round(float((ts["idx"][:, 1] >= 0).mean()), 3)}
    log(f"people: {res}")
    return res


def dancer_one(chunk_id: str, cam: str) -> None:
    """front_pod.dancer_one (label + SAM 2.1 + identity repair + body / hair + QA) with the teacher tracks as the candidate tracks. The two
    slots are lead (A) and follow (B) already, so the colour rule's keep / swap decision is forced to 'keep'."""
    from . import dancers as D
    from . import front_pod as fp
    take = take_name(chunk_id)
    tt = np.load(take_work(chunk_id) / "dancers" / "teacher_tracks" / f"{cam}.npz")
    tracks = {"idx": tt["idx"].astype(int), "boxes": tt["boxes"].astype(np.float32)}
    orig_evidence = D.track_evidence

    def evidence(cam_, tracks_, imgs, cfg):
        ev = orig_evidence(cam_, tracks_, imgs, cfg)
        ev["a_is_follow"] = np.zeros(len(ev["llr"]), bool)
        ev["swap_post"] = np.zeros(len(ev["llr"]))
        if os.environ.get("HM_TEACHER_EVIDENCE") == "1":
            # C3 (additive, carlos_aline): the demo's colour rule is calibrated for another pair (black top / white top + red hair); the roles are the teacher tracks'
            # (slot A = lead). Where both are present and apart the evidence for "A is the lead" is therefore the teachers' own, a constant, so that the label-frame choice
            # (dancers.choose_label_frame wants signed evidence > 1) only asks for a clean, separated frame. The recap (env unset) keeps the demo's rule.
            ev["llr"] = np.where((np.asarray(tracks_["idx"]) >= 0).all(1) & (np.asarray(ev["pair_iou"]) <= cfg.evidence_iou), 3.0, 0.0)
        return ev

    D.candidate_tracks = lambda *a, **k: tracks
    D.track_evidence = evidence
    fp.dancer_one(take, cam)


def regauge_master(chunk_id: str, master: str = MASTER) -> dict:
    """dancecap.sync fixes only RELATIVE offsets (median over cameras = 0). Shift them so that the master camera's video offset is 0: the
    reference clock of the whole class is the master's own video time, the same gauge in every chunk."""
    p = take_work(chunk_id) / "sync" / "offsets.json"
    d = json.loads(p.read_text())
    off = d["offsets"]
    if master not in off:
        return {"regauged": False, "why": f"camera {master} has no offset"}
    shift = float(off[master]["delta"])
    for c, v in off.items():
        v["delta"] = float(v["delta"]) - shift
        v["delta_ms"] = round(1000 * v["delta"], 2)
    d["gauge"] = f"camera {master} (the master) = 0; shifted every camera by {-1000 * shift:.2f} ms after the median gauge"
    d["gauge_master"] = {"camera": master, "shift_ms": round(-1000 * shift, 2)}
    p.write_text(json.dumps(d, indent=1))
    return {"regauged": True, "shift_ms": round(-1000 * shift, 2)}


# F1 2026-10-09: the per-phone VIDEO offsets of the six windows from 100 s on (shape, g1588 ... g7699) agree to +-25 ms with residuals of 2.4-8.8 px; the
# first window (0-55 s) did not (16-71 px, estimators 299 ms apart, one phone at the search edge). A phone whose window estimate fails the test below gets
# these medians (ms, master 01 = 0) instead of a number that would shift it by a quarter of a second. Medians of the six reliable windows.
REF_OFFSETS_MS = {"00": 106.5, "01": 0.0, "02": 17.5, "03": 20.0, "05": -0.3, "06": 11.5}
SYNC_MAX_RESID_PX = 10.0       # sync's own along-epipolar residual of the triangulated landmarks at the estimated offset (reliable windows: 2.4-8.8 px)
SYNC_MAX_GAP_MS = 60.0         # the two estimators of dancecap.sync may disagree by this much (reliable windows: <= 27 ms)


def sync_policy(doc: dict, master: str = MASTER, ref: dict | None = None) -> dict:
    """Pure part of the policy above: replaces the unreliable per-camera offsets of a dancecap.sync document (in place) and returns what it did,
    {cam: {"own_ms", "used_ms", "why"}}. The master is never changed (its offset is 0 by the gauge)."""
    ref = REF_OFFSETS_MS if ref is None else ref
    done = {}
    for c, v in doc.get("offsets", {}).items():
        if c == master or c not in ref:
            continue
        res = (v.get("residual_px") or {}).get("at_delta")
        why = []
        if v.get("at_search_edge"):
            why.append("at the search edge")
        if v.get("use") is False:
            why.append("sync marked it unusable")
        if res is None or res > SYNC_MAX_RESID_PX:
            why.append(f"residual {'n/a' if res is None else f'{res:.1f}'} px > {SYNC_MAX_RESID_PX:g}")
        gap = v.get("estimator_gap_ms")
        if gap is not None and gap > SYNC_MAX_GAP_MS:
            why.append(f"estimators {gap:.0f} ms apart > {SYNC_MAX_GAP_MS:g}")
        if why:
            done[c] = {"own_ms": v.get("delta_ms"), "used_ms": ref[c], "why": "; ".join(why)}
            v.update({"delta": ref[c] / 1000.0, "delta_ms": float(ref[c]), "use": True, "use_rate": False, "rate": 0.0, "f1_policy": done[c]})
    if done:
        doc["f1_policy"] = {"what": "unreliable window estimates replaced by the median of the reliable windows", "cams": done}
    return done


def apply_fixed_offsets(doc: dict, fixed: dict | None, master: str = MASTER) -> dict:
    """T1 2026-10-09: replace the video offsets of a dancecap.sync document (in place) by values the JOB fixes (ms, master camera = 0): a short
    boundary window cannot measure its own offsets, so it takes the ones its neighbours measured (chunk["sync_fixed"]["offsets_ms"]).
    Returns {cam: {"own_ms", "used_ms", "why"}} for the cameras changed (the master is never changed)."""
    done = {}
    for c, ms in ((fixed or {}).get("offsets_ms") or {}).items():
        v = doc.get("offsets", {}).get(c)
        if v is None or c == master:
            continue
        done[c] = {"own_ms": v.get("delta_ms"), "used_ms": float(ms), "why": "fixed by the job spec (the neighbouring chunks' measured offsets)"}
        v.update({"delta": round(float(ms) / 1000.0, 5), "delta_ms": float(ms), "use": True, "use_rate": False, "rate": 0.0, "fixed": done[c]})
    if done:
        doc["fixed_offsets"] = {"what": "offsets set by the job spec instead of this window's own estimate", "cams": done}
    return done


def fixed_sync_doc(work: Path, offsets_ms: dict, why: str, cams: list[str], master: str = MASTER) -> dict:
    """C3 (additive): the document dancecap.sync would write, with the video offsets the JOB fixes (ms, master camera = 0). dancecap.sync triangulates each
    camera's landmarks from at least TWO OTHER cameras (residual(): V.sum(0) >= 2), so a two-camera take (carlos_aline) cannot be measured by it (its first run
    died in `arange: cannot compute length`: no valid residual at all). The numbers come from the YOLO joints of both cameras: two-view reprojection of
    both dancers against the shift of camera 02's frames (a parabola over 40 ms steps, work/carlos_aline/people/video_offset.json)."""
    from . import sync as S
    offs = {}
    for c in cams:
        ms = 0.0 if c == master else float(offsets_ms.get(c, 0.0))
        offs[c] = {"delta": round(ms / 1000.0, 5), "delta_ms": ms, "stable": True, "use": True, "use_rate": False, "rate": 0.0, "t_ref": 0.0, "attached": False,
                   "fixed": {"why": why}}
    return {"version": 1, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "what": "per-camera residual video offset: corrected reference time = t + delta. FIXED by the job spec (a two-camera take: dancecap.sync cannot measure it)",
            "gauge": f"camera {master} (the master) = 0", "offsets": offs, "fixed_offsets": {"what": why, "cams": {c: offsets_ms.get(c) for c in cams if c != master}},
            "clocks": {"alignment": (S.take_clocks(work) or {}).get("alignment"), "sources": S.frame_clocks(work)},
            "host": "pod" if os.environ.get("DANCECAP_POD") == "1" else "local"}


def step_sync(chunk: dict, workers: int | None = None) -> dict:
    from . import sync as S
    fx = chunk.get("sync_fixed") or {}
    if fx.get("no_measure"):
        work = take_work(chunk["id"])
        doc = fixed_sync_doc(work, fx.get("offsets_ms") or {}, fx.get("why") or "fixed by the job spec", list(chunk["cams"]))
        p = work / "sync" / "offsets.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(doc, indent=1))
        log(f"sync fixed by the job spec (no measurement): { {c: v['delta_ms'] for c, v in doc['offsets'].items()} } ms")
        return {"fixed": True, "offsets_ms": {c: v["delta_ms"] for c, v in doc["offsets"].items()}}
    cams = [c for c in chunk["cams"] if (take_work(chunk["id"]) / "mamma" / "ma_2d_native" / f"{c}.npz").exists()]
    if len(cams) < 2:
        raise RuntimeError(f"only {cams} have ma_2d landmarks")
    S.run(take_name(chunk["id"]), cams, workers=workers)
    res = regauge_master(chunk["id"])
    p = take_work(chunk["id"]) / "sync" / "offsets.json"
    doc = json.loads(p.read_text())
    done = sync_policy(doc)
    fixed = apply_fixed_offsets(doc, chunk.get("sync_fixed"))
    if done or fixed:
        (p.with_name("offsets.window_estimate.json")).write_text(p.read_text())
        p.write_text(json.dumps(doc, indent=1))
        log(f"sync policy: replaced {done}; fixed by the job: {fixed}")
    res["policy"] = done
    res["fixed"] = fixed
    return res


SHAPE_JOB = os.environ.get("HM_SHAPE_JOB", "shape")        # C3 (additive): the shape job id of the take (carlos_aline: "cashape", its ids must not meet the recap's on the volume)
SHAPE_TAKE = f"{DANCE}_{SHAPE_JOB}"


def pin_path() -> Path:
    return PACKAGE_ROOT / "work" / SHAPE_TAKE / "mamma_pin_betas.json"


def step_pin(chunk_id: str) -> Path:
    """The shape job's betas -> work/<take>/mamma_pin_betas.json, per MAMMA body id ("0" lead, "1" follow: the 0-based person index of the fit,
    NOT the dancers' pids; front_full.pin_betas explains the bug the pid keying had). Every other chunk pins them (MAMMA_INIT_BETAS + FREEZE)."""
    import hashlib

    from .mamma_dyn import PIDS, ROLES
    work = take_work(chunk_id)
    doc, src = {}, {}
    for body_id, pid in enumerate(PIDS):
        role = ROLES[pid]
        f = work / "smplx" / f"{role}.npz"
        b = np.asarray(np.load(f)["betas"], float).reshape(-1)
        doc[str(body_id)] = [float(x) for x in b]
        src[str(body_id)] = {"role": role, "pid": pid, "file": f"{take_name(chunk_id)}/smplx/{role}.npz", "sha1": hashlib.sha1(f.read_bytes()).hexdigest(), "n": int(len(b))}
    out = work.parent / take_name(chunk_id) / "mamma_pin_betas.json"
    out.write_text(json.dumps(doc, indent=1))
    out.with_name(out.stem + ".source.json").write_text(json.dumps(src, indent=1))
    log(f"pinned betas -> {out} ({ {k: v['role'] for k, v in src.items()} })")
    return out


def step_await_pin(chunk_id: str, timeout_min: float = 120.0, poll_s: float = 20.0) -> Path:
    """Wait for the shape job's betas on the VOLUME and put them where `fit --pin` reads them (F1, 2026-10-09: the chunk jobs no longer run
    `after` the shape job, so every chunk's front (SAM / ma_2d / sync / prepare, about as long as the shape pass) runs in PARALLEL with it and
    only the fit waits for the pin). Source: $HM_ROOT/chunks/shape/out/result.tar once chunks/shape/done.json exists. Idempotent."""
    import tarfile
    out = pin_path()
    if out.exists():
        log(f"pin already in place: {out}")
        return out
    root = _root()
    if root is None:
        raise FileNotFoundError(f"{out} is missing and HM_ROOT is not set (no volume to wait on)")
    sdir = root / "chunks" / SHAPE_JOB
    t_end = time.time() + 60.0 * timeout_min
    last = 0.0
    while time.time() < t_end:
        if (sdir / "done.json").exists() and (sdir / "out" / "result.tar").exists():
            with tarfile.open(sdir / "out" / "result.tar") as tar:
                want = {m.name.rsplit("/", 1)[-1]: m for m in tar.getmembers() if m.isfile() and m.name.rsplit("/", 1)[-1].startswith("mamma_pin_betas")}
                if "mamma_pin_betas.json" not in want:
                    raise FileNotFoundError(f"{sdir / 'out' / 'result.tar'} holds no mamma_pin_betas.json (members: {sorted(want)})")
                out.parent.mkdir(parents=True, exist_ok=True)
                for name, m in want.items():
                    tmp = out.parent / (name + ".part")
                    tmp.write_bytes(tar.extractfile(m).read())
                    os.replace(tmp, out.parent / name)
            log(f"pin from the shape job -> {out}")
            return out
        if time.time() - last > 120:
            last = time.time()
            state = "FAILED (waiting for a re-queue)" if (sdir / "failed.json").exists() else ("running" if (sdir / "claim.lock").exists() else "not claimed yet")
            log(f"waiting for the shape job's betas: shape job {state}")
        time.sleep(poll_s)
    raise TimeoutError(f"the shape job's betas did not appear within {timeout_min:g} min")


def parse_masks(spec: str) -> list[tuple[str, float, float]]:
    """'06:152:208,02:242.5:258' -> [('06', 152.0, 208.0), ('02', 242.5, 258.0)]: camera, first and last master second to leave out of the fit."""
    out = []
    for part in [x for x in (spec or "").split(",") if x.strip()]:
        c, a, b = part.strip().split(":")
        if float(b) <= float(a):
            raise ValueError(f"mask {part!r}: the end is not after the start")
        out.append((c.strip(), float(a), float(b)))
    return out


def mask_frames(grid_t0: float, hz: float, T: int, t0: float, t1: float) -> tuple[int, int] | None:
    """Grid frames [k0, k1) of a chunk whose time lies in [t0, t1) (None when the range misses the chunk)."""
    k0 = max(0, int(np.ceil((t0 - grid_t0) * hz - 1e-9)))
    k1 = min(int(T), int(np.ceil((t1 - grid_t0) * hz - 1e-9)))
    return (k0, k1) if k1 > k0 else None


def step_mask(chunk_id: str, spec: str) -> dict:
    """REFIT helper (F1, 2026-10-09): leave a camera out of the fit for a stretch of the chunk, exactly as `mamma prepare` does for a frame on which a
    camera is not registered (landmarks, visibilities, contacts and floor contacts of that camera zeroed in mamma/ma_2d/<seq>/cam<NN>.npz). Run on a
    pod after the front checkpoint was restored and before `fit`. Idempotent; the record is mamma/mask.json."""
    take = take_name(chunk_id)
    work = take_work(chunk_id)
    prep = json.loads((work / "mamma" / "prepare.json").read_text(encoding="utf-8"))
    tag = max(prep["sets"], key=lambda t: len(prep["sets"][t]["cams"]))
    g = prep["grid"]
    t0g, hz, T = float(g.get("t0", 0.0)), float(g.get("hz", 30.0)), int(g["T"])
    d = work / "mamma" / "ma_2d" / f"{take}_{tag}"
    rec = []
    for cam, a, b in parse_masks(spec):
        f = d / f"cam{cam}.npz"
        kk = mask_frames(t0g, hz, T, a, b)
        if not f.exists() or kk is None:
            rec.append({"cam": cam, "t": [a, b], "frames": None, "why": "no such camera file" if not f.exists() else "outside the chunk"})
            continue
        z = dict(np.load(f))
        for k in ("landmarks", "visibilities", "contacts", "floor_contacts"):
            if k in z:
                z[k][kk[0]:kk[1]] = 0.0
        tmp = f.with_name(f.stem + ".part.npz")
        np.savez_compressed(tmp, **z)
        os.replace(tmp, f)
        rec.append({"cam": cam, "t": [a, b], "frames": [kk[0], kk[1]]})
    (work / "mamma" / "mask.json").write_text(json.dumps({"tag": tag, "grid": {"t0": t0g, "hz": hz, "T": T}, "masked": rec}, indent=1), encoding="utf-8")
    log(f"masked: {rec}")
    return {"masked": rec}


# T1 2026-10-09: a SAM mask that follows the wrong person (F1: phone 06's follow in 150-208 s, phone 05's follow in 200-210 s, phone 02's follow
# in 260-270 s) drags the whole fit, and the blunt cure (leave the camera out for BOTH dancers) also throws away the other dancer's good
# landmarks. This check compares, per camera, dancer and frame, the extent of the ma_2d landmarks with the extent of the teacher's 2D joints
# (the same person by construction of the teacher tracks): the healthy chunks agree to 1-7 % of the person's height, a wrong person is 15-100 %.
QA_THR = 0.15              # rolling-median extent-centre distance / person height above which the landmarks of that dancer are not his / hers
QA_WIN = 31                # frames of the rolling median (1 s)
QA_PAD = 15                # frames added on both sides of a wrong stretch
QA_MIN_VIS = 30            # visible landmarks needed for a frame to be judged
QA_MAX_SHARE = 0.5         # BOTH dancers judged wrong in more than this share of a camera's frames = the check itself is suspect (a clock or coordinate
                           # problem, not two SAM failures at once): nothing is masked for that camera. ONE dancer wrong all along (SAM on the
                           # wrong person for the whole chunk, F1: phone 06's follow) is what the step is for and is masked.


def extent_error(L, V, kp, conf: float = 0.3, min_joints: int = 8, min_vis: int = QA_MIN_VIS) -> float:
    """One frame, one dancer: ma_2d landmarks L (512, 3) with visibility V (512,) against the teacher's COCO joints kp (17, 3). The distance
    between the centre of the landmarks' extent (5-95 percentile of x and y) and the centre of the joints' extent, divided by the joints'
    height (at least 40 px). NaN when either side is missing."""
    L = np.asarray(L, float)
    m = np.asarray(V, float) > 0.5
    kp = np.asarray(kp, float)
    ok = np.isfinite(kp[:, 2]) & (kp[:, 2] >= conf)
    if int(m.sum()) < min_vis or int(ok.sum()) < min_joints:
        return float("nan")
    p = L[m][:, :2]
    cm = 0.5 * (np.percentile(p, 5, axis=0) + np.percentile(p, 95, axis=0))
    tp = kp[ok, :2]
    ct = 0.5 * (tp.min(0) + tp.max(0))
    h = max(float(tp[:, 1].max() - tp[:, 1].min()), 40.0)
    return float(np.hypot(*(cm - ct)) / h)


def qa_ranges(d, thr: float = QA_THR, win: int = QA_WIN, pad: int = QA_PAD) -> list[tuple[int, int]]:
    """Frame ranges [a, b) where the rolling median of d (NaN = unknown, ignored) exceeds thr, widened by pad frames on both sides."""
    d = np.asarray(d, float)
    T = len(d)
    half = win // 2
    bad = np.zeros(T, bool)
    need = max(5, win // 4)
    for t in range(T):
        seg = d[max(0, t - half):t + half + 1]
        seg = seg[np.isfinite(seg)]
        if len(seg) >= need and float(np.median(seg)) > thr:
            bad[t] = True
    if pad and bad.any():
        bad = np.convolve(bad.astype(float), np.ones(2 * pad + 1), mode="same") > 0
    out, a = [], None
    for t in range(T + 1):
        on = bool(bad[t]) if t < T else False
        if on and a is None:
            a = t
        elif not on and a is not None:
            out.append((a, t))
            a = None
    return out


def qa_mask_camera(z: dict, kp, j, good, thr: float = QA_THR) -> dict:
    """Judge and mask ONE camera's ma_2d arrays z (landmarks / visibilities / contacts / floor_contacts, frame axis first, slot axis 1 = lead,
    follow; modified in place) against the teacher joints kp (F, 2, 17, 3) at the camera frames j (one per grid frame) where good is True.
    Returns {"lead": [(a, b), ...], "follow": [...], "suspect": [roles judged wrong in the camera, left alone because BOTH were (QA_MAX_SHARE)],
    "d": {role: per-frame error array}}."""
    T = int(np.asarray(z["landmarks"]).shape[0])
    res = {"lead": [], "follow": [], "suspect": [], "d": {}}
    ranges = {}
    for slot, role in ((0, "lead"), (1, "follow")):
        d = np.full(T, np.nan)
        for i in np.flatnonzero(np.asarray(good, bool)[:T]):
            d[i] = extent_error(z["landmarks"][i, slot], z["visibilities"][i, slot], kp[int(j[i]), slot])
        res["d"][role] = d
        ranges[role] = qa_ranges(d, thr)
    if all(sum(b - a for a, b in ranges[r]) > QA_MAX_SHARE * T for r in ("lead", "follow")):
        res["suspect"] = ["lead", "follow"]
        return res
    for slot, role in ((0, "lead"), (1, "follow")):
        for a, b in ranges[role]:
            for k in ("landmarks", "visibilities", "contacts", "floor_contacts"):
                if k in z:
                    z[k][a:b, slot] = 0.0
        res[role] = ranges[role]
    return res


def step_qa_mask(chunk_id: str, thr: float = QA_THR) -> dict:
    """Pod step after the front (T1, 2026-10-09): zero, per camera and per dancer, the stretches in which the ma_2d landmarks are not on the
    teacher (qa_mask_camera). Reads mamma/prepare.json (grid, offsets), mamma/ma_2d/<seq>/cam<NN>.npz, cameras/<cam>.npz and the teacher tracks
    people/teachers_<cam>.npz of the dance. Idempotent; the record is mamma/qa_mask.json."""
    from .recap_people import PEOPLE
    take = take_name(chunk_id)
    work = take_work(chunk_id)
    prep = json.loads((work / "mamma" / "prepare.json").read_text(encoding="utf-8"))
    tag = max(prep["sets"], key=lambda t: len(prep["sets"][t]["cams"]))
    g = prep["grid"]
    T = int(g["T"])
    tg = np.asarray(g["t"], float)[:T] if g.get("t") else float(g.get("t0", 0.0)) + np.arange(T) / float(g.get("hz", 30.0))
    offs = (prep.get("sync") or {}).get("offsets_ms") or {}
    d = work / "mamma" / "ma_2d" / f"{take}_{tag}"
    rec = {"thr": thr, "grid_T": T, "cams": {}}
    for cam in prep["sets"][tag]["cams"]:
        f = d / f"cam{cam}.npz"
        if not f.exists():
            continue
        zc = np.load(dance_work() / "cameras" / f"{cam}.npz", allow_pickle=True)
        tcam, okc = np.asarray(zc["t"], float), np.asarray(zc["ok"], bool)
        tt = np.load(PEOPLE / f"teachers_{cam}.npz", allow_pickle=True)
        tau = tg - float(offs.get(cam, 0.0)) / 1000.0
        j = np.clip(np.searchsorted(tcam, tau), 1, len(tcam) - 1)
        j = np.where(np.abs(tcam[j - 1] - tau) < np.abs(tcam[j] - tau), j - 1, j)
        good = (np.abs(tcam[j] - tau) <= 0.04) & okc[j] & np.asarray(tt["ok"], bool)[j]
        z = dict(np.load(f))
        r = qa_mask_camera(z, np.asarray(tt["kp"], float), j, good, thr)
        n = {role: int(sum(b - a for a, b in r[role])) for role in ("lead", "follow")}
        rec["cams"][cam] = {"masked_frames": n, "ranges": {role: [[int(a), int(b)] for a, b in r[role]] for role in ("lead", "follow")},
                            "suspect": r["suspect"],
                            "median_error": {role: (None if not np.isfinite(r["d"][role]).any() else round(float(np.nanmedian(r["d"][role])), 4))
                                             for role in ("lead", "follow")}}
        if any(n.values()):
            tmp = f.with_name(f.stem + ".part.npz")
            np.savez_compressed(tmp, **z)
            os.replace(tmp, f)
    (work / "mamma" / "qa_mask.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    log("qa-mask: " + json.dumps({c: v["masked_frames"] for c, v in rec["cams"].items()}) + " suspect: "
        + json.dumps({c: v["suspect"] for c, v in rec["cams"].items() if v["suspect"]}))
    return rec


def step_fit(chunk_id: str, pin: bool = False) -> dict:
    """The MAMMA fit of the chunk's prepared camera set, then `select` (promotes the fit to smplx/). pin=True: the shape job's betas are
    pinned and frozen (the demo's mechanism); the shape job itself (pin=False) estimates them."""
    from . import mamma_dyn as md
    take = take_name(chunk_id)
    if pin:
        p = pin_path()
        if not p.exists():
            raise FileNotFoundError(f"{p}: the shape job's betas are missing (this job runs after {SHAPE_TAKE})")
        os.environ["MAMMA_INIT_BETAS"] = str(p)
        os.environ["MAMMA_FREEZE_BETAS"] = "1"
    else:
        os.environ.pop("MAMMA_INIT_BETAS", None)
        os.environ.pop("MAMMA_FREEZE_BETAS", None)
    prep = json.loads((md.take_work(take) / "mamma" / "prepare.json").read_text(encoding="utf-8"))
    tag = max(prep["sets"], key=lambda t: len(prep["sets"][t]["cams"]))
    md.fit(take, tag)
    md.select(take, qa=False)
    return {"take": take, "tag": tag, "pinned": bool(pin)}


# ------------------------------------------------------------------------------------------------ the DAG


def sam_mem_gb(chunk: dict, cam: str) -> float:
    """Host RAM of one dancers worker (front_pod._sam_mem_gb's formula: all frames BGR + 2 bool masks + SAM probabilities + QA frames, + 3 GB) for
    the frames of `cam` inside the chunk window: about 30 GB for the two 1080x1920 phones, 7-8 GB for the others.
    QA frames as front_pod._qa_frame_px: 1707 x 960 for a landscape 1280x720 video."""
    from .front_pod import _qa_frame_px
    z = np.load(dance_work() / "cameras" / f"{cam}.npz", allow_pickle=True)
    f = len(frame_range(np.asarray(z["t"], float), chunk["t0"], chunk["t1"]))
    w, h = (int(x) for x in z["native_size"])
    px = w * h
    work_px = min(1.0, 1280 / max(w, h)) ** 2 * px
    return round((f * (3 * px + 2 * px + 2 * work_px + 3 * _qa_frame_px(w, h))) / 2**30 * 1.1 + 3.0, 1)


def build_tasks(R, chunk: dict) -> None:
    """The chunk's front DAG on a front_pod.Runner (see the module doc). Cameras with no teacher coverage in the chunk are not in chunk['cams']."""
    from .front_pod import Task, merge_label
    py = sys.executable
    work = take_work(chunk["id"])
    take = take_name(chunk["id"])
    cams = list(chunk["cams"])
    me = "dancecap.recap_front"
    done_dir = work / "front" / "done"
    done_dir.mkdir(parents=True, exist_ok=True)
    flag = lambda n: (lambda: (done_dir / f"{n}.ok").exists())            # noqa: E731
    mark = lambda n: (lambda: (done_dir / f"{n}.ok").write_text(time.strftime("%FT%T")))   # noqa: E731
    R.add(Task("frames", [py, "-m", me, "frames", chunk["id"]], kind="cpu", mem_gb=6.0, timeout_min=60, done_if=flag("frames"), on_ok=mark("frames")))
    R.add(Task("cameras", [py, "-m", me, "cameras", chunk["id"]], kind="cpu", mem_gb=2.0, timeout_min=10, done_if=flag("cameras"), on_ok=mark("cameras")))
    R.add(Task("people", [py, "-m", me, "people", chunk["id"]], kind="cpu", mem_gb=4.0, timeout_min=15, deps=["frames"],
               done_if=flag("people"), on_ok=mark("people")))
    for c in cams:
        # F1 2026-10-09: on_ok=merge_label as in the demo's front_pod DAG (dancers_pod.merge reads dancers/labels.json, which only merge_label writes;
        # without it every recap job died at `merge` after SAM / ma_2d had run).
        R.add(Task(f"sam:{c}", [py, "-m", me, "dancer", chunk["id"], c], deps=["people", "cameras"], kind="sam", mem_gb=sam_mem_gb(chunk, c), gpu=True,
                   timeout_min=120, on_ok=lambda c=c: merge_label(work, c), done_if=lambda c=c: (work / "dancers" / f"pod_cam_{c}.json").exists()))
        R.add(Task(f"ma2d:{c}", [py, "-m", "dancecap.front_pod", "ma2d", take, c], deps=[f"sam:{c}"], kind="ma2d", mem_gb=8.0, gpu=True,
                   timeout_min=60, done_if=lambda c=c: (work / "mamma" / "ma_2d_native" / f"{c}.manifest.json").exists()))
    R.add(Task("merge", [py, "-m", "dancecap.dancers_pod", "merge", take, "--cams", ",".join(cams)], after=[f"sam:{c}" for c in cams], kind="cpu",
               mem_gb=8.0, timeout_min=20, retries=0))
    R.add(Task("sync", [py, "-m", me, "sync", chunk["id"]], deps=["merge", "cameras"], after=[f"ma2d:{c}" for c in cams], kind="cpu", mem_gb=12.0,
               timeout_min=90, retries=0))
    k0, k1 = int(chunk["k0"]), int(chunk["k1"])
    R.add(Task("prepare", [py, "-m", "dancecap", "mamma", "prepare", take, "--skip-2d", "--set", f"c{len(cams)}={','.join(cams)}",
                           "--coverage", "union", "--grid-anchor", repr(GRID_ANCHOR), f"--grid-k={k0}:{k1}"], deps=["sync"], kind="cpu",
               mem_gb=8.0, timeout_min=30, retries=0))


def run(chunk_id: str, resume: bool = False) -> int:
    """The whole front of one chunk (front_pod.run's driver around recap_front.build_tasks)."""
    from . import front_pod as fp
    chunk = load_chunk(chunk_id)
    work = take_work(chunk_id)
    cpus, ram = fp.cgroup_cpus(), fp.mem_gb()
    ws0, limit = fp.mem_state()
    threads = str(max(2, min(4, cpus // 8)))
    env = dict(os.environ, DANCECAP_GPU_SHARED="1", OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads, OPENBLAS_NUM_THREADS=threads,
               OPENCV_FOR_THREADS_NUM=threads, PYTHONUNBUFFERED="1", DANCECAP_POD=os.environ.get("DANCECAP_POD", "1"), YOLO_CONFIG_DIR="/tmp/Ultralytics")
    cap_gb = limit if limit else ram
    n_sam = max(2, min(4, cpus // 4))
    caps = {"sam": n_sam, "ma2d": 2, "cpu": 3, "loc": 1}
    R = fp.Runner(work, env, max(16.0, 0.62 * cap_gb), caps, work / "front" / "logs", limit)
    build_tasks(R, chunk)
    if resume:
        R.resume()
    else:
        R.memlog.unlink(missing_ok=True)
        for f in R.rc_dir.glob("*.rc"):
            f.unlink()
    R.meta = {"take": take_name(chunk_id), "chunk": chunk_id, "cams": chunk["cams"], "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "host": {"cpus": cpus, "ram_gb": round(ram, 1), "mem_limit_gb": limit, "caps": caps}}
    R.save()
    log(f"{take_name(chunk_id)}: {len(R.tasks)} tasks, cameras {chunk['cams']}, caps {caps}")
    R.loop()
    R.meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    R.save()
    bad = [k for k, t in R.tasks.items() if t.state != "ok" and not t.optional]
    ok = R.tasks["prepare"].state == "ok"
    (work / "front").mkdir(parents=True, exist_ok=True)
    (work / "front" / "front_done.json").write_text(json.dumps({"rc": 0 if ok else 1, "not_ok": bad, "finished": R.meta["finished"]}))
    log(f"done; not ok: {bad or 'none'}")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.recap_front", description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("run", "frames", "cameras", "people", "dancer", "sync", "fit", "pin", "await-pin", "mask", "qa-mask"))
    ap.add_argument("chunk")
    ap.add_argument("cam", nargs="?")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--pin", action="store_true", help="fit: pin + freeze the shape job's betas")
    ap.add_argument("--mask", default="", help="mask: 'cam:t0:t1,cam:t0:t1' master seconds to leave out of the fit (refit)")
    ap.add_argument("--thr", type=float, default=QA_THR, help="qa-mask: extent-centre distance / person height above which the landmarks are not the teacher's")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        return run(a.chunk, a.resume)
    if a.cmd == "fit":
        step_fit(a.chunk, a.pin)
        return 0
    if a.cmd == "pin":
        step_pin(a.chunk)
        return 0
    if a.cmd == "await-pin":
        step_await_pin(a.chunk)
        return 0
    if a.cmd == "mask":
        step_mask(a.chunk, a.mask)
        return 0
    if a.cmd == "qa-mask":
        step_qa_mask(a.chunk, a.thr)
        return 0
    if a.cmd == "dancer":
        dancer_one(a.chunk, a.cam)
        return 0
    chunk = load_chunk(a.chunk)
    {"frames": step_frames, "cameras": step_cameras, "people": step_people, "sync": step_sync}[a.cmd](chunk)
    return 0


if __name__ == "__main__":
    sys.exit(main())
