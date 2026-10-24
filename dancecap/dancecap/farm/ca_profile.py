"""Two-camera profiles of the farm planner: carlos_aline (agent C3, 2026-10-10), the Jack & Jill heat of Carlos and Aline, and lucas_luiza (LL2V, 2026-10-24),
the triangulated two-view take of Lucas and Luiza, as equal-window chunk job graphs.

A take opts in with `[chunks] profile = "carlos_aline"` (takes/carlos_aline.toml) or `"lucas_luiza"`; `farm plan --dance <take>` then calls cmd_plan() below with
the take's TwoView (PROFILES). It is the class recap's profile (farm/recap_profile.py, which stays untouched) made take-agnostic where the heat differs:

  cameras    TWO cameras: per-frame tracks (cameras/<cam>.npz) are inputs. carlos_aline: two tripod heads that pan / tilt (01 also zooms); camera 02 starts 14.8 s
             after 01, so the first 11.2 s of the danced span are monocular ([stitch] mono in the take TOML flags them). lucas_luiza: a Steadicam (01) and a
             tripod (02) that both record the whole window; every window is FORCED onto both cameras (the owner wants every chunk triangulated; the recap's
             camera rule is still evaluated and reported).
  people     NOT the owner's 2D JSON: YOLO11x-pose of every person (C2's pods) as pose tables, and the teacher tracks people/teachers_<cam>.npz in the recap's
             contract (carlos_aline_people; lucas_luiza: dancers_seed + carlos_aline_inputs), so dancecap.recap_front runs unchanged on them (job env
             HM_FRONT_DANCE / HM_FRONT_CAPTURE / HM_SHAPE_JOB select the take; defaults = the recap).
  ids        every job id starts with the profile's prefix (queue/ and chunks/ of the volume are shared with the recap's g0000 ... and shape): cag0000 ...,
             cashape, cafinal; llg0000 ..., llshape, llfinal. lucas_luiza's specs also carry "owner": "LL2V" (worker.claimable: other sessions' pods leave them).
  clocks     dancecap.sync cannot measure two cameras. carlos_aline: C3's fixed offsets (SYNC_FIXED). lucas_luiza: the MEASURED two-view epipolar offsets of
             work/lucas_luiza/sync/offsets.json (cameras.twoview sync, gauge 02 = the reference clock), evaluated at each chunk's centre (TwoView.sync_for), and
             HM_FRONT_MASTER=02 in every job so that recap_front gauges on camera 02.
  steps      the recap's chunk steps + the T1 `qa_mask` step (zero the landmarks that are not on the teacher) before the fit waits for the pinned betas;
             lucas_luiza adds the foot-slip refine variant (smplx_rf2_slip + its pose_quality) and a stature check of the shape pass.
  shape      a 30 s window with BOTH cameras on both dancers, the couple apart / not merged (so the betas see the whole body), fitted with free betas;
             every chunk freezes them (await_pin), the stitcher verifies they are identical; no pin to any other take.
  final      one CPU job: dancecap.stitch of the windows (the demo's seam gates), no analysis, no world check. `plan --final-only` rebuilds only this job from
             the current TOML (e.g. after [export] smplx_dir picked the refine variant).
  money      lucas_luiza carries its own farm budget: `plan` writes it into the farm policy (state.json: budget, pod hard ceiling, RTX 4090), which `farm
             launch` and the watcher honour (the analysis run's budget does not count farm pods).
  gates      lucas_luiza's `plan` writes nothing when a step 13 gate fails (validate_plan -> strict_checks) or while its chunks are claimed (busy_jobs).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import analysis as A
from . import chunks as C
from . import farm_dir, jobs_dir
from . import recap_profile as RP
from .state import State, atomic_write_json, now_iso
from .timeline import HZ, DanceSpec
from ..take import PACKAGE_ROOT, WORKSPACE

# ------------------------------------------------------------------------------------------------ carlos_aline's values (the CA profile below; kept as names)

PROFILE = "carlos_aline"
DANCE = "carlos_aline"
CAPTURE = "Carlos-Aline"
ID_PREFIX = "ca"
SHAPE_ID = "cashape"
FINAL_ID = "cafinal"
SHAPE_TAKE = f"{DANCE}_{SHAPE_ID}"
IN_TAR_REL = f"src/{CAPTURE}/ca_in.tar"
PIN_REL = f"dancecap/work/{SHAPE_TAKE}/mamma_pin_betas.json"
WORK_ROOT = PACKAGE_ROOT / "work"
WORK = WORK_ROOT / DANCE
PEOPLE = WORK / "people"
CAMS = ("01", "02")
ALIGNMENT = WORKSPACE / "prototype" / "larissa_kadu" / "alignment_carlos_aline.json"
JOB_ENV = {"DANCECAP_FARM_DANCE": DANCE, "HM_FRONT_DANCE": DANCE, "HM_FRONT_CAPTURE": CAPTURE, "HM_SHAPE_JOB": SHAPE_ID}
CONF = 0.5

# the fit of two cameras is cheaper per frame than the recap's six (1.62 s/frame measured with 6): scaled by this share of the 6-camera rate, never below the drill's model
FIT_S_PER_FRAME_2CAM = 0.9

# dancecap.sync needs >= 2 OTHER cameras per camera: a two-camera take cannot be measured by it (the first pods died at the sync step). The video offsets of the pair are
# the two-view reprojection minimum of the YOLO joints of both dancers against the shift of camera 02's frames (work/carlos_aline/people/video_offset.json: minimum at
# +62 ms, i.e. camera 02's frame labelled t + 0.062 s shows what camera 01's frame labelled t shows, so camera 02's delta is -62 ms; C2 found 60 ms the same way).
SYNC_FIXED = {"no_measure": True, "offsets_ms": {"01": 0.0, "02": -62.0},
              "why": "two cameras: dancecap.sync cannot measure; two-view joint reprojection minimum (C3, people/video_offset.json), C2 -60 ms"}

# step 13 gates of a strict (two-view) profile: the shape window must really carry both cameras and the couple apart
SHAPE_MIN_BOTH, SHAPE_MIN_APART = 0.6, 0.5
SYNC_TOL_MS = 0.01             # a job's fixed offset equals the measured delta(t) at its centre (both are rounded to 0.01 ms)
FIT_S_PER_FRAME_MEASURED = 1.9  # carlos's measured two-camera fit of a 1548-frame chunk (FARM.md): the fit timeouts are judged on it, not on the profile's own rate
BOOT_SLACK_MIN = 15.0          # pod creation -> the job's first step (boot, bootstrap, input download): the pods' hard ceiling = the longest GPU job + this


class SyncError(RuntimeError):
    """The take's measured video offsets are missing or not usable: no chunk is planned on an offset that was not measured."""


def _num(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _use_rate(o: dict) -> bool:
    """Whether a camera's drift rate is applied: mamma_dyn.read_sync_offsets' rule (dancecap.sync always reports a rate, use_rate says when it counts)."""
    return bool(o.get("use_rate", "rate" in o and "use_rate" not in o and "stable" not in o))


@dataclass(frozen=True)
class TwoView:
    """One two-camera take of this planner. sync_fixed: the chunks' fixed video offsets {no_measure, offsets_ms, why}, or "file" = the measured two-view offsets of
    work/<dance>/sync/offsets.json at each chunk's centre (sync_for). shape_t: (t_min, t_max) of the shape window search, None = the take's window minus 1 s at each
    end. apart_cam: the camera whose floor positions say whether the couple is apart. force_cams: every window gets every camera (the rule's verdict is reported).
    refine_variants: ((dst dir, pose_refine args appended to the recipe's), ...) refined and measured next to the recipe's dir. strict: validate_plan also checks
    the two-view rules (owner, env, measured offsets, both cameras at the window ends, shape stats, timeouts, costs, budget). budget_usd: the farm's own ceiling
    (watch.Policy.budget_usd) that `plan` writes into the farm policy; max_plan_usd: the expected cost the plan may not exceed."""
    profile: str
    dance: str
    capture: str
    id_prefix: str
    shape_id: str
    final_id: str
    owner: str | None
    cams: tuple
    alignment: Path
    alignment_arc: str
    sync_fixed: dict | str
    fit_s_per_frame: float
    extra_env: dict
    shape_t: tuple | None
    apart_cam: str
    force_cams: bool
    monocular_until_s: float | None
    refine_variants: tuple = ()
    scale_check: bool = False
    strict: bool = False
    budget_usd: float | None = None
    max_plan_usd: float | None = None
    budget_why: str = ""

    @property
    def shape_take(self) -> str:
        return f"{self.dance}_{self.shape_id}"

    @property
    def in_tar_rel(self) -> str:
        return f"src/{self.capture}/{self.id_prefix}_in.tar"

    @property
    def pin_rel(self) -> str:
        return f"dancecap/work/{self.shape_take}/mamma_pin_betas.json"

    @property
    def work(self) -> Path:
        return WORK_ROOT / self.dance

    @property
    def people(self) -> Path:
        return self.work / "people"

    @property
    def job_env(self) -> dict:
        return {"DANCECAP_FARM_DANCE": self.dance, "HM_FRONT_DANCE": self.dance, "HM_FRONT_CAPTURE": self.capture, "HM_SHAPE_JOB": self.shape_id, **self.extra_env}

    @property
    def master(self) -> str:
        """The camera whose video offset is 0: recap_front.MASTER, read from the job env HM_FRONT_MASTER (recap_front's default is 01)."""
        return str(self.extra_env.get("HM_FRONT_MASTER", "01"))

    # ---- the measured video offsets (sync_fixed == "file")

    def offsets_doc(self) -> dict:
        """work/<dance>/sync/offsets.json, checked: written by the two-view epipolar scan (cameras.twoview sync) on this take's current clocks (sync.offsets_check,
        the rule mamma_dyn.read_sync_offsets applies: the chunks re-stamp their own clocks, so a stale file would pass on the pods unnoticed), gauged on the master
        camera (its delta 0), every other camera present with finite numbers, stable, used and not at the search edge. Raises SyncError otherwise."""
        from .. import sync as S
        p = self.work / "sync" / "offsets.json"
        if not p.is_file():
            raise SyncError(f"{p} missing: the two-view sync (python -m dancecap.cameras.twoview sync {self.dance}, Job A) has not been pulled")
        doc = json.loads(p.read_text(encoding="utf-8"))
        if doc.get("method") != "two_view_epipolar":
            raise SyncError(f"{p}: method {doc.get('method')!r}, not the two-view epipolar scan (cameras.twoview sync)")
        chk = S.offsets_check(self.work, doc)
        if chk["status"] == "stale" or (chk["status"] == "unrecorded" and chk["required"]):
            raise SyncError(f"{p} is {chk['status']} ({'; '.join(chk['detail'])}): not measured on this take's clocks; run the two-view sync again")
        offs = doc.get("offsets") or {}
        m = offs.get(self.master)
        if not isinstance(m, dict) or not math.isfinite(_num(m.get("delta") or 0.0)) or abs(_num(m.get("delta") or 0.0)) > 1e-9:
            raise SyncError(f"{p}: not gauged on camera {self.master} (HM_FRONT_MASTER): its offset is {None if not isinstance(m, dict) else m.get('delta')}")
        for c in self.cams:
            if c == self.master:
                continue
            o = offs.get(c)
            if not isinstance(o, dict) or o.get("delta") is None:
                raise SyncError(f"{p}: no measured offset for camera {c}")
            nums = [o["delta"], o.get("t_ref") or 0.0] + ([o.get("rate") or 0.0] if _use_rate(o) else [])
            if not all(math.isfinite(_num(x)) for x in nums):
                raise SyncError(f"{p}: camera {c}'s offset is not a number (delta {o['delta']}, rate {o.get('rate')}, t_ref {o.get('t_ref')})")
            if o.get("stable") is not True or o.get("use") is False or o.get("at_search_edge"):
                raise SyncError(f"{p}: camera {c}'s offset is not usable (stable {o.get('stable')}, use {o.get('use')}, at the search edge {o.get('at_search_edge')})")
        return doc

    def offset_ms(self, doc: dict, cam: str, t: float) -> float:
        """delta_cam(t) in ms on the take's clock: delta + rate x (t - t_ref). The rate counts as mamma_dyn.read_sync_offsets counts it (use_rate), so the chunks
        and every whole-take reader of the same file (camera_video, world_scale chain) apply the same offset."""
        if cam == self.master:
            return 0.0
        o = doc["offsets"][cam]
        if o.get("piecewise"):  # cameras.twoview sync: a delta that varies along the take (mamma_dyn.apply_offset reads it)
            pw = np.asarray(o["piecewise"], float).reshape(-1, 2)
            return 1000.0 * float(np.interp(float(t), pw[:, 0], pw[:, 1]))
        rate = float(o.get("rate") or 0.0) if _use_rate(o) else 0.0
        return 1000.0 * (float(o["delta"]) + rate * (float(t) - float(o.get("t_ref") or 0.0)))

    def sync_for(self, chunk: C.Chunk, grid, doc: dict | None = None) -> dict:
        """chunk["sync_fixed"] of a job: the profile's fixed offsets, or (sync_fixed == "file") the measured offsets at the chunk's centre, master = 0."""
        if self.sync_fixed != "file":
            return self.sync_fixed
        doc = doc if doc is not None else self.offsets_doc()
        t_mid = 0.5 * (float(grid.t(chunk.k0)) + float(grid.t(chunk.k1 + 1)))
        return {"no_measure": True, "offsets_ms": {c: round(self.offset_ms(doc, c, t_mid), 2) for c in self.cams},
                "why": f"two-view epipolar scan (cameras.twoview sync), gauge {self.master}", "t_mid_s": round(t_mid, 3)}


CA = TwoView(profile=PROFILE, dance=DANCE, capture=CAPTURE, id_prefix=ID_PREFIX, shape_id=SHAPE_ID, final_id=FINAL_ID, owner=None, cams=CAMS, alignment=ALIGNMENT,
             alignment_arc="prototype/larissa_kadu/alignment_carlos_aline.json", sync_fixed=SYNC_FIXED, fit_s_per_frame=FIT_S_PER_FRAME_2CAM, extra_env={},
             shape_t=(14.9, 197.0), apart_cam="01", force_cams=False, monocular_until_s=14.82)

# lucas_luiza (LL2V, runbook step 12): 01 = the Steadicam (Video-44260.mp4), 02 = the tripod original = the reference clock. The fit rate is carlos's measured two-camera
# rate (1.65 s/frame for the 900-frame shape, ~1.9 s/frame for a 1548-frame chunk; FARM.md); HM_TEACHER_EVIDENCE=1 because the demo's colour rule is another couple's.
LL = TwoView(profile="lucas_luiza", dance="lucas_luiza", capture="lucas-luisa-jnj", id_prefix="ll", shape_id="llshape", final_id="llfinal", owner="LL2V",
             cams=("01", "02"), alignment=WORK_ROOT / "lucas_luiza" / "clock" / "alignment.json", alignment_arc="dancecap/work/lucas_luiza/clock/alignment.json",
             sync_fixed="file", fit_s_per_frame=1.9, extra_env={"HM_TEACHER_EVIDENCE": "1", "HM_FRONT_MASTER": "02"}, shape_t=None, apart_cam="02", force_cams=True,
             monocular_until_s=None, refine_variants=(("smplx_rf2_slip", "--w-slip 0.5 --slip-max-v 0.15"),), scale_check=True, strict=True,
             budget_usd=14.0, max_plan_usd=6.0,
             budget_why="the farm's share of analysis run 20261023-100155-lucas_luiza's $25 limit: ~$23.30 left - Job A's --max-cost 7.5 - analyses 1.0 - ~$0.8 reserve "
                        "(farm pods are not counted by the run: analyze.run_pods counts its gateway and tagged sessions only)")

PROFILES = {"carlos_aline": CA, "lucas_luiza": LL}


def load_teachers(P: TwoView = CA) -> dict:
    """{cam: {t, ok, conf (F, 2)}} and the per-frame 'merged' flags from people/teachers_<cam>.npz."""
    out = {}
    for c in P.cams:
        p = P.people / f"teachers_{c}.npz"
        if p.exists():
            z = np.load(p)
            out[c] = {"t": np.asarray(z["t"], float), "ok": np.asarray(z["ok"], bool), "conf": np.asarray(z["conf"], float), "merged": np.asarray(z["merged"], bool),
                      "floor": np.asarray(z["floor"], float)}
    return out


def shape_window(tt: dict, P: TwoView = CA, frames: int = 900, step_s: float = 2.5, t_min: float | None = None, t_max: float | None = None) -> dict:
    """The 30 s window (900 grid frames) where BOTH cameras see BOTH dancers best and the couple is not in a clinch: per 30 Hz grid instant, the share of the two cameras
    that have both dancers with confidence >= CONF (weight 0.55), the share of instants whose two detections are apart (floor distance in camera P.apart_cam >= 0.9 m,
    weight 0.3) and not merged (weight 0.15). Searched over [t_min, t_max] (default P.shape_t: carlos_aline only after camera 02 starts, monocular frames carry no
    second view)."""
    t_min = P.shape_t[0] if t_min is None else t_min
    t_max = P.shape_t[1] if t_max is None else t_max
    L = frames / HZ
    grid = np.arange(0.0, t_max + 30.0, 1.0 / HZ)
    both = np.zeros((len(grid), len(tt)), bool)
    apart = np.zeros(len(grid), bool)
    clear = np.ones(len(grid), bool)
    for n, (c, o) in enumerate(tt.items()):
        j = np.clip(np.searchsorted(o["t"], grid), 1, len(o["t"]) - 1)
        j = np.where(np.abs(o["t"][j - 1] - grid) < np.abs(o["t"][j] - grid), j - 1, j)
        good = (np.abs(o["t"][j] - grid) <= 0.05) & o["ok"][j]
        both[:, n] = good & (o["conf"][j] >= CONF).all(1)
        clear &= ~(good & o["merged"][j].any(1)) | ~good
        if c == P.apart_cam:
            d = np.linalg.norm(o["floor"][j, 0] - o["floor"][j, 1], axis=-1)
            apart = good & (d >= 0.9)
    score = 0.55 * both.mean(1) + 0.30 * apart + 0.15 * clear
    best, rows = None, []
    for t0 in np.arange(t_min, t_max - L, step_s):
        m = (grid >= t0) & (grid < t0 + L)
        s = float(score[m].mean() * (both[m].all(1).mean() ** 0.5))          # a window where both cameras really carry both dancers
        rows.append((round(float(t0), 2), round(s, 3)))
        if best is None or s > best[1]:
            best = (float(t0), s, {"both_cameras_both_dancers": round(float(both[m].all(1).mean()), 3), "apart": round(float(apart[m].mean()), 3), "not_merged": round(float(clear[m].mean()), 3)})
    k0 = int(round(best[0] * HZ))
    return {"t0": best[0], "t1": best[0] + L, "k0": k0, "k1": k0 + frames - 1, "score": round(best[1], 3), "stats": best[2], "scores": rows}


def window_cams(tt: dict, t0: float, t1: float, P: TwoView = CA) -> tuple[list[str], dict]:
    """Cameras of a window: those that record >= 35 % of it with at least one dancer in 30 % of their frames (the recap's rule, RP.choose_cams, on the teacher tracks);
    with P.force_cams every camera of the profile (the stats are still returned, make_plan reports the rule's verdict)."""
    stats = RP.window_stats({c: {"t": o["t"], "ok": o["ok"], "conf": o["conf"]} for c, o in tt.items()}, t0, t1)
    if P.force_cams:
        return list(P.cams), stats
    return RP.choose_cams(stats)[0], stats


# ------------------------------------------------------------------------------------------------ the shared input tar


def build_in(out: Path | None = None, P: TwoView = CA) -> dict:
    """ONE deterministic tar of what a chunk job needs and the volume does not hold: the camera tracks (with pts / native_size), recap_cams.json, world_from_sfm.json, the
    teacher tracks, the pose tables (kp, n only) and the alignment file (at the path the chunk take's TOML names). Workspace-relative members; uploaded once to
    src/<capture>/ (shared by every job)."""
    out = out or P.people / "in" / f"{P.id_prefix}_in.tar"
    rel = f"dancecap/work/{P.dance}"
    slim = P.people / "in" / "pose_tables"
    slim.mkdir(parents=True, exist_ok=True)
    items = []
    for c in P.cams:
        z = np.load(P.people / "pose_tables" / f"{c}.npz")
        np.savez_compressed(slim / f"{c}.npz", kp=z["kp"], n=z["n"])
        items += [(P.work / "cameras" / f"{c}.npz", f"{rel}/cameras/{c}.npz"), (P.people / f"teachers_{c}.npz", f"{rel}/people/teachers_{c}.npz"),
                  (slim / f"{c}.npz", f"{rel}/people/pose_tables/{c}.npz")]
    items += [(P.work / "cameras" / "recap_cams.json", f"{rel}/cameras/recap_cams.json"), (P.work / "cameras" / "world_from_sfm.json", f"{rel}/cameras/world_from_sfm.json"),
              (P.alignment, P.alignment_arc)]
    missing = [str(p) for p, _ in items if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(f"{P.id_prefix}_in.tar: missing {missing[:3]}")
    C.pack_tar(out, items)
    h = hashlib.sha256()
    with open(out, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    cams = {c: hashlib.sha256((P.work / "cameras" / f"{c}.npz").read_bytes()).hexdigest()[:16] for c in P.cams}
    return {"path": P.in_tar_rel, "sha256": h.hexdigest(), "bytes": out.stat().st_size, "local": C._ws(out), "archive": True, "n_files": len(items), "cameras": cams}


# ------------------------------------------------------------------------------------------------ estimates and job specs


def variant_tag(dst: str) -> str:
    """pose_quality tag of a refine variant dir (smplx_rf2_slip -> rf2_slip)."""
    return dst[len("smplx_"):] if dst.startswith("smplx_") else dst


def estimate_chunk(chunk: C.Chunk, cam_frames: int | None = None, shape: bool = False, P: TwoView = CA) -> dict:
    e = RP.estimate_chunk(chunk, cam_frames=cam_frames, shape=shape)
    fit = (RP.FIT_A + P.fit_s_per_frame * chunk.T) / 60.0
    tot = e["front_min"] + fit + e["refine_min"] + e["quality_min"] + RP.BOOT_MIN
    e.update({"fit_min": round(fit, 1), "total_min_4090": round(tot, 1), "usd_4090": round(tot / 60 * RP.PRICE_4090, 2), "total_min": round(tot / RP.GPU_SLOWDOWN, 1)})
    if P.refine_variants and not shape:          # each variant: one more refine + pose_quality on the same pod
        var = len(P.refine_variants) * (e["refine_min"] + e["quality_min"])
        tot += var
        e.update({"variants_min": round(var, 1), "total_min_4090": round(tot, 1), "usd_4090": round(tot / 60 * RP.PRICE_4090, 2), "total_min": round(tot / RP.GPU_SLOWDOWN, 1)})
    return e


def chunk_steps(chunk: C.Chunk, est: dict, refine_args: str, *, shape: bool, refined_dir: str, P: TwoView = CA) -> list[dict]:
    """chunk_take -> front -> qa_mask -> [await_pin] -> fit -> [pin -> scale_check | refine -> quality -> (refine_<v> -> quality_<v>)...] -> evidence (RP.chunk_steps
    with this take's pin and the qa_mask step; the refine variants re-refine the raw fit with the recipe's arguments and the variant's appended: argparse keeps the last)."""
    take = chunk.take
    ctx = {"chunk": chunk.id, "take": take}
    w = f"dancecap/work/{take}"
    steps = [C._step("chunk_take", C.fmt("python -m dancecap.farm chunk-take {chunk}", **ctx), 3),
             C._step("front", C.fmt("python -m dancecap.recap_front run {chunk}", **ctx), max(45, math.ceil(2.0 * est["front_min"] + 15)),
                     save=[f"{w}/mamma/prepare.json", f"{w}/mamma/ma3d_config.yaml", f"{w}/mamma/ma_2d/", f"{w}/mamma/ma_cap/", f"{w}/sync/offsets.json",
                           f"{w}/cameras/world_from_sfm.json", f"{w}/dancers/roles.json"]),
             C._step("qa_mask", C.fmt("python -m dancecap.recap_front qa-mask {chunk}", **ctx), 10, save=[f"{w}/mamma/ma_2d/", f"{w}/mamma/qa_mask.json"])]
    if not shape:
        steps.append(C._step("await_pin", C.fmt("python -m dancecap.recap_front await-pin {chunk}", **ctx), 130))
    steps.append(C._step("fit", C.fmt("python -m dancecap.recap_front fit {chunk}" + ("" if shape else " --pin"), **ctx), max(30, math.ceil(1.6 * est["fit_min"] + 15)),
                         env=({} if shape else {"MAMMA_FREEZE_BETAS": "1", "MAMMA_INIT_BETAS": "{WS}/" + P.pin_rel}), live=[f"{w}/mamma/ma_3d/"],
                         save=[f"{w}/smplx/", f"{w}/mamma/fits/", f"{w}/mamma/select.json", f"{w}/mamma/report.json"]))
    stage = {"chunk_take": "front", "front": "front", "qa_mask": "front", "await_pin": "front", "fit": "fit", "pin": "fit", "refine": "refined", "quality": "refined"}
    if shape:
        steps.append(C._step("pin", C.fmt("python -m dancecap.recap_front pin {chunk}", **ctx), 3, save=[f"dancecap/work/{take}/mamma_pin_betas.json"]))
        if P.scale_check:      # the free-beta rest heights against stature + sole in this world (G9; exit 2 unless ok: optional, the json says why)
            steps.append(C._step("scale_check", C.fmt("python -m dancecap.cameras.world_scale check {dance} --smplx-dir work/{take}/smplx --strict --out "
                                                      "work/{take}/scale_check.json > work/{take}/scale_check.log 2>&1", dance=P.dance, **ctx), 10, optional=True,
                                 save=[f"{w}/scale_check.json", f"{w}/scale_check.log"]))
            stage["scale_check"] = "fit"
    else:
        steps += [C._step("refine", C.fmt("python -m dancecap.pose_refine {take} --src smplx --dst {d} {a}", d=refined_dir, a=refine_args, **ctx),
                          max(10, math.ceil(2 * est["refine_min"] + 5)), save=[f"{w}/{refined_dir}/"]),
                  C._step("quality", C.fmt("python -m dancecap.pose_quality {take} --smplx-dir {d} --tag rf2_noslip", d=refined_dir, **ctx), 20, optional=True)]
        for dst, extra in P.refine_variants:
            tag = variant_tag(dst)
            v = tag.rsplit("_", 1)[-1]
            steps += [C._step(f"refine_{v}", C.fmt("python -m dancecap.pose_refine {take} --src smplx --dst {d} {a} {x}", d=dst, a=refine_args, x=extra, **ctx),
                              max(10, math.ceil(2 * est["refine_min"] + 5)), save=[f"{w}/{dst}/"]),
                      C._step(f"quality_{v}", C.fmt("python -m dancecap.pose_quality {take} --smplx-dir {d} --tag {t}", d=dst, t=tag, **ctx), 20, optional=True)]
            stage.update({f"refine_{v}": "refined", f"quality_{v}": "refined"})
    last = "fit" if shape else "refined"
    steps.append({"name": "evidence", "cmd": f"python -m dancecap.farm chunk-evidence {chunk.id} --stage {last}", "timeout_min": 10.0, "optional": False, "stage": last})
    names = [s["name"] for s in steps]
    if len(set(names)) != len(names):            # the variant's step is named by the tag's last word (refine_slip): two variants may not share it
        raise ValueError(f"{P.profile}: the refine variants {[d for d, _ in P.refine_variants]} give duplicate step names {names}")
    for s in steps[:-1]:
        s["stage"] = stage[s["name"]]
    return steps


def chunk_outputs(chunk: C.Chunk, *, shape: bool, refined_dir: str, P: TwoView = CA) -> list[dict]:
    rows = RP.chunk_outputs(chunk, shape=shape, refined_dir=refined_dir)
    rows.append({"path": f"dancecap/work/{chunk.take}/mamma/qa_mask.json", "optional": True})
    rows.append({"path": f"dancecap/work/{chunk.take}/front/front_done.json", "optional": True})
    if shape and P.scale_check:
        rows += [{"path": f"dancecap/work/{chunk.take}/scale_check.json", "optional": True}, {"path": f"dancecap/work/{chunk.take}/scale_check.log", "optional": True}]
    if not shape:
        rows += [{"path": f"dancecap/work/{chunk.take}/{dst}/"} for dst, _ in P.refine_variants]
    return rows


def build_job(chunk: C.Chunk, spec: DanceSpec, *, hash_inputs: bool = True, code: dict | None = None, shape: bool = False, in_tar: dict | None = None,
              P: TwoView = CA, sync_doc: dict | None = None) -> dict:
    est = chunk.estimate if chunk.estimate.get("front_min") is not None and "cam_frames" in chunk.estimate else estimate_chunk(chunk, shape=shape, P=P)
    steps = chunk_steps(chunk, est, RP.refine_args_recipe(), shape=shape, refined_dir=spec.refined_dir, P=P)
    if hash_inputs:
        inputs, sources = C.source_inputs(spec, chunk.cams)
        inputs.append(in_tar or build_in(P=P))
    else:
        inputs, sources = [{"path": P.in_tar_rel, "archive": True}], {}
    total = est["total_min_4090"]
    return {"schema": C.JOB_SCHEMA, "id": chunk.id, "dance": spec.name, "kind": "full", "what": "shape" if shape else "chunk", "stages": ["front", "fit"] + ([] if shape else ["refined"]),
            "take": chunk.take, "chunk": {**chunk.to_dict(), "sync_fixed": P.sync_for(chunk, spec.grid, sync_doc)}, "cams": chunk.cams, "after": [], "world_id": None,
            "world_sha256": None, "pose_id": None, "code": code if code is not None else C.code_info(), "sources": sources, "inputs": inputs, "steps": steps,
            "outputs": chunk_outputs(chunk, shape=shape, refined_dir=spec.refined_dir, P=P), "env": dict(P.job_env), "cwd": "dancecap", **({"owner": P.owner} if P.owner else {}),
            "max_minutes": round(max(60.0, 1.6 * total + 20.0), 1), "priority": 100.0 if shape else float(total), "min_vram_gb": 24, "needs_gpu": True, "estimate": est,
            "created": now_iso(), "profile": P.profile, "retries": 1}


def final_from_volume(job: dict) -> dict:
    """The final job takes EVERY chunk's result.tar from the volume (its inputs): it runs after all of them, no chunk counts as local motion and none carries a world.
    analysis.build_final_job would list chunks already pulled here as local motion (after = [], local_chunks = them), which stitch-prep then refuses."""
    ids = [c["id"] for c in job.get("stitch_chunks") or []]
    job["after"] = ids
    job["local_chunks"] = []
    job["chunk_worlds"] = {c: {"world_id": None, "pose_id": None, "how": None} for c in ids}
    return job


def tiles(chunks: list[dict], grid) -> bool:
    """Do the chunks' cores tile the grid k0..k1 exactly (no gap, no overlap)?"""
    cl = sorted(chunks, key=lambda c: c["core_k0"])
    return bool(cl) and cl[0]["core_k0"] == grid.k0 and cl[-1]["core_k1"] == grid.k1 and all(b["core_k0"] == x["core_k1"] + 1 for x, b in zip(cl, cl[1:]))


def build_final(chunks: list[C.Chunk], spec: DanceSpec, *, code: dict | None = None, P: TwoView = CA) -> dict:
    cl = [c.to_dict() for c in chunks]
    job = A.build_final_job(spec, cl, None, analyses=(), jid=P.final_id, hash_inputs=False, code=code)
    for s in job["steps"]:
        if s["name"] == "stitch":
            s["cmd"] = s["cmd"].replace(" --require-world", "")
    job["inputs"] = [{"path": f"chunks/{c.id}/out/result.tar", "archive": True} for c in chunks]
    job["profile"] = P.profile
    if P.owner:
        job["owner"] = P.owner
    job["env"] = {**(job.get("env") or {}), **P.job_env}
    if P.strict:
        final_from_volume(job)
    return job


# ------------------------------------------------------------------------------------------------ the plan


def budget_doc(jobs: list[dict], costs: dict, P: TwoView) -> dict | None:
    """The farm budget the plan carries: P.budget_usd, the pods' hard ceiling (the longest GPU job's max_minutes + BOOT_SLACK_MIN), the GPU type it is priced on,
    and the worst cases pods.launch / launch_cpu reserve against it (price x (hard ceiling + start timeout)). fleet_worst_usd is the SUM over the route's pods (one
    RTX 4090 pod per GPU job + the CPU stitch pod), not pods at once: watch.farm_cost counts a finished pod's spend and a live pod's remaining worst case, so with
    the budget >= that sum no pod of the route is refused even when every earlier one ran to its ceiling. The pod ceiling, not the worker's own job limit
    (max_minutes x 1.25 + 5), is then what cuts the longest job: at about its max_minutes (1.6x its estimate + 20 min), the price of fitting every worst case
    into the budget (a ceiling sized for the worker's limit makes a 4090 pod's worst case ~$3.1 and lucas_luiza's route ~$16.4, over its $14)."""
    if P.budget_usd is None:
        return None
    from . import pods
    from . import worker
    from .watch import GPU_ALIASES, Policy
    gpu = [j for j in jobs if j.get("needs_gpu")]
    gmax = max([float(j["max_minutes"]) for j in gpu] or [60.0])
    hard = int(math.ceil(gmax + BOOT_SLACK_MIN))
    per_pod = RP.PRICE_4090 * (hard + pods.START_TIMEOUT_MIN) / 60.0
    pol = Policy()
    cpu = pods.cpu_price(pods.CPU_PRIORITY[0], pol.cpu_vcpu) * (pol.cpu_hard_minutes + pods.START_TIMEOUT_MIN) / 60.0
    price_max = (P.budget_usd - cpu) / max(1, len(gpu)) * 60.0 / (hard + pods.START_TIMEOUT_MIN)
    return {"usd": P.budget_usd, "hard_minutes": hard, "gpus": [GPU_ALIASES["4090"]], "gpu_pod_worst_usd": round(per_pod, 2), "cpu_pod_worst_usd": round(cpu, 2),
            "fleet_worst_usd": round(len(gpu) * per_pod + cpu, 2), "gpu_jobs": len(gpu), "price_4090_hr": RP.PRICE_4090, "fits_up_to_4090_hr": round(price_max, 3),
            "worker_job_limit_min": round(gmax * worker.JOB_LIMIT_FACTOR + worker.JOB_LIMIT_SLACK_MIN, 1), "expected_usd": costs.get("usd"), "why": P.budget_why,
            "note": "written into the farm policy (state.json) by `farm plan` (budget_usd, hard_minutes, gpus): `farm launch` and the watcher cap spent + committed "
                    "worst cases at it; `farm up` re-saves its own earlier policy (use queue + launch)"}


def make_plan(spec: DanceSpec, *, hash_inputs: bool = True, code: dict | None = None, claimed_ids: set | None = None, P: TwoView = CA) -> dict:
    tt = load_teachers(P)
    if len(tt) < 2:
        raise FileNotFoundError("people/teachers_<cam>.npz missing: run " + ("dancecap.carlos_aline_people run" if P.profile == "carlos_aline" else
                                f"dancecap.dancers_seed run {P.dance} + dancecap.carlos_aline_inputs prepare --take {P.dance}") + " first")
    sync_doc = P.offsets_doc() if P.sync_fixed == "file" else None
    g = spec.grid
    chunks, rep = C.plan_windows(spec, kind="full", claimed_ids=claimed_ids, cams_fn=lambda k0, k1: window_cams(tt, g.t(k0), g.t(k1 + 1), P)[0])
    rows = []
    for ch in chunks:
        ch.id = P.id_prefix + ch.id                     # the volume's chunks/ and queue/ are shared with the class recap (g0000 ...)
        ch.take = f"{spec.name}_{ch.id}"
        cams, st = window_cams(tt, g.t(ch.k0), g.t(ch.k1 + 1), P)
        ch.cams = cams
        ch.estimate = estimate_chunk(ch, cam_frames=sum(st[c]["frames"] for c in cams), P=P)
        ch.priority = ch.estimate["total_min_4090"]
        rows.append({"id": ch.id, "k": [ch.k0, ch.k1], "core_k": [ch.core_k0, ch.core_k1], "window_s": round(ch.T / HZ, 2), "core_s": round(ch.core_T / HZ, 2),
                     "t": [round(float(g.t(ch.k0)), 2), round(float(g.t(ch.k1 + 1)), 2)], "cams": cams, "cams_stats": st, "estimate": ch.estimate,
                     "monocular_until_s": P.monocular_until_s if P.monocular_until_s is not None and g.t(ch.k0) < P.monocular_until_s else None})
        if P.force_cams:                                # what the recap's rule would have picked (the owner wants both views in every window)
            rule, why = RP.choose_cams(st)
            rows[-1]["cams_rule"] = {"would_pick": rule, "why": why}
    if P.shape_t is None:
        sw = shape_window(tt, P, t_min=float(g.t_start) + 1.0, t_max=float(g.t_end) - 1.0)
    else:
        sw = shape_window(tt, P)
    scams, _ = window_cams(tt, sw["t0"], sw["t1"], P)
    shape = C.Chunk(id=P.shape_id, k0=sw["k0"], k1=sw["k1"], core_k0=sw["k0"], core_k1=sw["k1"], overlap_k=0, kind="full", cams=scams, world_id=None, origin="shape",
                    take=P.shape_take)
    shape.estimate = estimate_chunk(shape, cam_frames=sum(int(HZ * 30) * 1 for _ in scams), shape=True, P=P)
    jobs = [build_job(shape, spec, hash_inputs=hash_inputs, code=code, shape=True, P=P, sync_doc=sync_doc)]
    in_tar = jobs[0]["inputs"][-1] if hash_inputs else None
    jobs += [build_job(ch, spec, hash_inputs=hash_inputs, code=code, in_tar=in_tar, P=P, sync_doc=sync_doc) for ch in chunks]
    final = build_final(chunks, spec, code=code, P=P)
    costs = RP.cost_table(jobs)
    extra = {}
    if P.strict:
        # with use_rate the chunks apply delta + rate (t - t_ref) as mamma_dyn's whole-take reader does; the scan's own drift line starts at fit.a_ms instead
        drift = {c: {"delta_ms": round(1000.0 * float(o["delta"]), 2), "fit_a_ms": (o.get("fit") or {}).get("a_ms"), "t_ref": o.get("t_ref")}
                 for c, o in ((sync_doc or {}).get("offsets") or {}).items() if c != P.master and _use_rate(o)}
        extra = {"owner": P.owner, "master": P.master, "budget": budget_doc(jobs + [final], costs, P),
                 "sync": {"source": "fixed" if sync_doc is None else f"work/{P.dance}/sync/offsets.json", "gauge": P.master,
                          "offsets_ms": {j["id"]: j["chunk"]["sync_fixed"]["offsets_ms"] for j in jobs}, **({"drift": drift} if drift else {})}}
    return {"created": now_iso(), "dance": spec.name, "profile": P.profile, "params": C.chunk_params(spec), "grid": g.to_dict(), "arithmetic": rep["arithmetic"], "chunks": rows,
            "shape": {"id": P.shape_id, "window": [round(sw["t0"], 2), round(sw["t1"], 2)], "k": [sw["k0"], sw["k1"]], "cams": scams, "score": sw["score"], "stats": sw["stats"],
                      "scores": sw["scores"], "estimate": shape.estimate},
            "costs": costs, "seams": [{"between": [chunks[i].id, chunks[i + 1].id], "overlap_s": round((chunks[i].k1 - chunks[i + 1].k0 + 1) / HZ, 3),
                                       "cut_t": round(float(g.t(chunks[i + 1].core_k0)), 2)} for i in range(len(chunks) - 1)],
            **extra, "_chunks": chunks, "_shape": shape, "_jobs": jobs + [final], "_teachers": tt}


def coverage_times(tt: dict, P: TwoView) -> dict:
    """{cam: (frame times, source)} of the coverage mamma prepare sees: the camera track's registered frames (cameras/<cam>.npz t[ok], read without its other
    members) when it is here, else the teacher track's frames."""
    out = {}
    for c in P.cams:
        p = P.work / "cameras" / f"{c}.npz"
        if p.is_file():
            with np.load(p) as z:
                t = np.asarray(z["t"], float)
                ok = np.asarray(z["ok"], bool) if "ok" in z.files else np.ones(len(t), bool)
            out[c] = (t[ok], f"cameras/{c}.npz t[ok]")
        elif c in tt:
            out[c] = (np.asarray(tt[c]["t"], float), f"people/teachers_{c}.npz t")
    return out


def window_ends(times: dict, grid, P: TwoView, doc: dict | None) -> dict:
    """Do both cameras cover the first and the last grid instant after their video offset (corrected time = t + delta(t)), within one of their frames? The chunk
    front prepares with `--coverage union` (recap_front.build_tasks), which keeps every grid frame that ANY camera covers: an end that one camera does not cover
    is fitted from the other camera alone (monocular first / last frames), not cut. times: coverage_times()."""
    a, b = float(grid.t(grid.k0)), float(grid.t(grid.k1))
    fixed = ((P.sync_fixed if isinstance(P.sync_fixed, dict) else {}).get("offsets_ms") or {})

    def delta_s(c, t):
        return P.offset_ms(doc, c, t) / 1000.0 if doc is not None else float(fixed.get(c, 0.0)) / 1000.0
    out, lo, hi = {}, -np.inf, np.inf
    for c in P.cams:
        t, src = times.get(c, (np.zeros(0), None))
        if len(t) < 2:
            out[c] = {"ok": False, "why": "no camera or teacher track"}
            continue
        dt = float(np.median(np.diff(t)))
        first, last = float(t[0]) + delta_s(c, t[0]), float(t[-1]) + delta_s(c, t[-1])
        lo, hi = max(lo, first), min(hi, last)
        out[c] = {"ok": bool(first <= a + dt + 1e-9 and last >= b - dt - 1e-9), "first_s": round(first, 4), "last_s": round(last, 4), "frame_s": round(dt, 4),
                  "from": src}
    ok = all(v["ok"] for v in out.values())
    res = {"ok": ok, "grid_ends_s": [round(a, 4), round(b, 4)], "cams": out}
    if not ok and np.isfinite(lo) and np.isfinite(hi):
        res["both_cover_s"] = [round(lo, 4), round(hi, 4)]
        res["advice"] = (f"narrow the take's window to [{math.ceil(lo * 100 - 1e-6) / 100:.2f}, {math.floor(hi * 100 + 1e-6) / 100:.2f}] (runbook: [16.85, 213.45]) "
                         "and plan again")
    return res


def strict_checks(plan: dict, spec: DanceSpec, P: TwoView) -> dict:
    """Step 13's gates of a two-view profile: every job carries the owner and the env of the two-view fit (teacher evidence, the master camera, no MAMMA_MONO), the
    launching shell claims the owner, every fit job has both cameras and the MEASURED offsets at its centre, both cameras cover the window ends, the shape window
    carries both views, the fit timeouts cover the measured fit rate, the plan fits its costs and its farm budget, and the final job stitches every chunk."""
    jobs = plan["_jobs"]
    fits = [j for j in jobs if j.get("what") in ("chunk", "shape")]
    res = {}

    def check(name, ok, **info):
        res[name] = {"ok": bool(ok), **info}
    check("owner_set", bool(P.owner) and all(j.get("owner") == P.owner for j in jobs), owner=P.owner, missing=[j["id"] for j in jobs if j.get("owner") != P.owner])
    env = {k: os.environ.get(k) for k in ("HM_QUEUE_OWNER", "HM_POD_OWNER")}
    # pods.create gives its pods HM_OWNER = HM_QUEUE_OWNER (worker.claimable: they claim only these specs) and labels them HM_POD_OWNER; the watcher launches
    # for any queued job, so without the export its pods never claim these specs and may take other dances' unowned ones on this budget
    check("launch_env_claims_the_owner", bool(P.owner) and all(v == P.owner for v in env.values()), env=env,
          fix=f"export HM_OWNER={P.owner} HM_QUEUE_OWNER={P.owner} HM_POD_OWNER={P.owner} in the shell that plans, queues, launches and starts the watcher")
    mono = [j["id"] for j in jobs if any(k.startswith("MAMMA_MONO") for e in [j.get("env") or {}] + [s.get("env") or {} for s in j.get("steps", [])] for k in e)]
    check("no_MAMMA_MONO", not mono, jobs=mono)
    check("env_has_teacher_evidence", all((j.get("env") or {}).get("HM_TEACHER_EVIDENCE") == "1" for j in jobs),
          missing=[j["id"] for j in jobs if (j.get("env") or {}).get("HM_TEACHER_EVIDENCE") != "1"])
    check("env_has_the_master_camera", P.master in P.cams and all((j.get("env") or {}).get("HM_FRONT_MASTER") == P.master for j in jobs), master=P.master,
          missing=[j["id"] for j in jobs if (j.get("env") or {}).get("HM_FRONT_MASTER") != P.master])
    check("every_fit_job_has_both_cameras", all(list(j.get("cams") or []) == list(P.cams) for j in fits), cams={j["id"]: j.get("cams") for j in fits})
    bad, doc, err = [], None, None
    try:
        doc = P.offsets_doc() if P.sync_fixed == "file" else None
    except (SyncError, OSError, ValueError) as e:
        err = str(e)
    for j in fits:
        sf = (j.get("chunk") or {}).get("sync_fixed") or {}
        got = sf.get("offsets_ms") or {}
        if not sf.get("no_measure") or not set(j.get("cams") or []) <= set(got) or abs(float(got.get(P.master, 0.0))) > SYNC_TOL_MS:
            bad.append(j["id"])
            continue
        if err is None:
            want = P.sync_for(C.Chunk.from_dict(j["chunk"]), spec.grid, doc)["offsets_ms"]
            if any(c not in got or abs(float(got[c]) - float(want[c])) > SYNC_TOL_MS for c in want):
                bad.append(j["id"])
    check("sync_fixed_is_the_measured_offset", err is None and not bad, bad=bad, error=err,
          offsets_ms={j["id"]: ((j.get("chunk") or {}).get("sync_fixed") or {}).get("offsets_ms") for j in fits})
    ends = window_ends(coverage_times(plan.get("_teachers") or load_teachers(P), P), spec.grid, P, doc) if err is None else {"ok": False, "why": err}
    check("both_cams_cover_window_ends", ends["ok"], **{k: v for k, v in ends.items() if k != "ok"})
    st = (plan.get("shape") or {}).get("stats") or {}
    check("shape_window_sees_both", float(st.get("both_cameras_both_dancers") or 0.0) >= SHAPE_MIN_BOTH and float(st.get("apart") or 0.0) >= SHAPE_MIN_APART,
          stats=st, need={"both_cameras_both_dancers": SHAPE_MIN_BOTH, "apart": SHAPE_MIN_APART})
    # judged on the MEASURED two-camera rate, not on P.fit_s_per_frame (the timeout is 1.6x the profile's own estimate + 15 by construction)
    tmo = {j["id"]: {"timeout": next(s["timeout_min"] for s in j["steps"] if s["name"] == "fit"), "estimate": j["estimate"]["fit_min"],
                     "at_measured_rate": round((RP.FIT_A + FIT_S_PER_FRAME_MEASURED * int(j["chunk"]["T"])) / 60.0, 1)} for j in fits}
    check("fit_timeouts_cover_the_estimate", all(v["timeout"] >= 1.6 * max(v["estimate"], v["at_measured_rate"]) for v in tmo.values()), fit_min=tmo,
          measured_s_per_frame=FIT_S_PER_FRAME_MEASURED)
    usd = float((plan.get("costs") or {}).get("usd") or 0.0)
    check("plan_cost_within", P.max_plan_usd is None or usd <= P.max_plan_usd, usd=usd, max_usd=P.max_plan_usd)
    b = plan.get("budget")
    if P.budget_usd is not None:
        gmax = max([float(j["max_minutes"]) for j in jobs if j.get("needs_gpu")] or [0.0])
        check("budget_covers_the_fleet", bool(b) and b["fleet_worst_usd"] <= b["usd"] + 1e-9 and b["hard_minutes"] >= gmax, budget=b, longest_gpu_job_min=gmax,
              note="what can fail is the route's worst case against the budget (more or longer GPU jobs); hard_minutes >= the longest job is how budget_doc sizes it")
    final = next((j for j in jobs if j["id"] == P.final_id), None)
    ids = [j["id"] for j in jobs if j.get("what") == "chunk"]
    sc = (final or {}).get("stitch_chunks") or []
    check("final_stitches_every_chunk", bool(final) and tiles(sc, spec.grid) and sorted(c["id"] for c in sc) == sorted(ids)
          and sorted(final.get("after") or []) == sorted(ids), stitch=[c["id"] for c in sc], after=(final or {}).get("after"), chunks=ids)
    check("capture_is_the_upload_folder", P.capture == Path(str(spec.cfg.get("source_dir", ""))).name, capture=P.capture, source_dir=spec.cfg.get("source_dir"))
    return res


def validate_plan(plan: dict, spec: DanceSpec, P: TwoView = CA) -> dict:
    """RP.validate_plan's offline checks with this take's ids (the recap's checks hard-code its shape id and require three cameras), the claim rules as the profile's
    own pods see them (worker.claimable compares the spec's owner with HM_OWNER), plus the two-view checks of a strict profile (strict_checks)."""
    saved, saved_owner = RP.SHAPE_ID, os.environ.get("HM_OWNER")
    RP.SHAPE_ID = P.shape_id
    if P.owner:
        os.environ["HM_OWNER"] = P.owner
    p2 = dict(plan)
    p2["_jobs"] = [dict(j, id="final") if j["id"] == P.final_id else j for j in plan["_jobs"]]      # the recap's check names the final job "final"
    try:
        v = RP.validate_plan(p2, spec)
    finally:
        RP.SHAPE_ID = saved
        if P.owner:
            if saved_owner is None:
                os.environ.pop("HM_OWNER", None)
            else:
                os.environ["HM_OWNER"] = saved_owner
    v["checks"].pop("every_chunk_has_at_least_3_cameras", None)
    v["checks"]["every_chunk_has_both_cameras"] = {"ok": all(len(c.cams) == 2 for c in plan["_chunks"]), "cams": {c.id: c.cams for c in plan["_chunks"]}}
    envs_ok = all(all((j.get("env") or {}).get(k) == val for k, val in P.job_env.items()) for j in plan["_jobs"])
    v["checks"]["job_env_selects_the_take"] = {"ok": envs_ok, "env": P.job_env}
    ids_ok = all(j["id"].startswith(P.id_prefix) for j in plan["_jobs"])
    v["checks"]["ids_do_not_meet_the_recap"] = {"ok": ids_ok, "ids": [j["id"] for j in plan["_jobs"]]}
    if P.strict:
        v["checks"].update(strict_checks(plan, spec, P))
    v["ok"] = all(c["ok"] for c in v["checks"].values())
    return v


def print_plan(plan: dict, P: TwoView = CA) -> None:
    ar = plan["arithmetic"]
    print(f"{P.profile} plan: {len(plan['chunks'])} chunks of {min(ar['window_s'])}-{max(ar['window_s'])} s, overlap {ar['overlap_k'] / HZ:g} s "
          f"(n = {ar['n']}: " + ", ".join(f"{t['n']} x {t['window_s']} s" for t in ar["tried"][-3:]) + ")")
    print(f"{'id':<9}{'window (s)':<18}{'core (s)':<9}{'cams':<8}{'front':>6}{'fit':>6}{'ref+q':>6}{'4090 min':>9}{'USD':>6}")
    for r in plan["chunks"]:
        e = r["estimate"]
        rq = e["refine_min"] + e["quality_min"] + e.get("variants_min", 0)          # lucas_luiza: + the slip variant's refine and quality
        print(f"{r['id']:<9}{r['t'][0]:>7.1f}-{r['t'][1]:<9.1f}{r['core_s']:<9}{','.join(r['cams']):<8}{e['front_min']:>6}{e['fit_min']:>6}{rq:>6.1f}"
              f"{e['total_min_4090']:>9}{e['usd_4090']:>6}")
    s = plan["shape"]
    e = s["estimate"]
    print(f"{s['id']:<9}{s['window'][0]:>7.1f}-{s['window'][1]:<9.1f}{30.0:<9}{','.join(s['cams']):<8}{e['front_min']:>6}{e['fit_min']:>6}{'':>6}{e['total_min_4090']:>9}{e['usd_4090']:>6}  (shape: free betas -> pin; {s['stats']})")
    c = plan["costs"]
    print(f"total: {c['gpu_hours']} GPU h = {c['usd']} USD on RTX 4090 pods ({c['usd_with_30pct_margin']} with a 30 % margin); wall about {c['wall_min_with_one_pod_per_job']} min with one pod per job")
    if P.strict:
        sy = plan["sync"]
        print(f"owner {plan['owner']}; video offsets (ms, gauge camera {sy['gauge']}, {sy['source']}): "
              + ", ".join(f"{jid} " + "/".join(f"{v:+.2f}" for c, v in o.items() if c != sy["gauge"]) for jid, o in sy["offsets_ms"].items()))
        b = plan.get("budget")
        if b:
            print(f"farm budget ${b['usd']:.2f} (worst case of one pod per job ${b['fleet_worst_usd']:.2f} at ${b['price_4090_hr']}/h, fits up to "
                  f"${b['fits_up_to_4090_hr']}/h; pod hard ceiling {b['hard_minutes']} min, RTX 4090 only): written into the farm policy by `plan` "
                  "(launch / watch honour it; `launch --budget X` overrides)")


def save_budget(dance: str, b: dict) -> dict:
    """The plan's budget -> the farm policy in state.json (watch.Policy: budget_usd, hard_minutes, gpus), so `farm launch` and the watcher cap the farm's spent +
    committed worst cases at it and launch the GPU it is priced on. A lower budget already stored is kept, and so is a GPU list already stored. The pods' hard ceiling
    never drops below what a GPU job the watcher still works on needs (its max_minutes + BOOT_SLACK_MIN); the initial kill-switch lead never exceeds it."""
    from . import watch as Wt
    base = Wt.Policy.load(dance)
    cur = State(dance).load()
    live = []
    for jid, rec in cur["jobs"].items():
        p = jobs_dir(dance) / f"{jid}.json"
        if ((cur["chunks"].get(jid) or rec).get("status") in Wt.ACTIVE) and p.is_file():
            j = json.loads(p.read_text(encoding="utf-8"))
            if j.get("needs_gpu"):
                live.append(math.ceil(float(j.get("max_minutes") or 0.0) + BOOT_SLACK_MIN))
    usd = float(b["usd"]) if base.budget_usd is None else min(float(base.budget_usd), float(b["usd"]))
    hard = float(max([float(b["hard_minutes"])] + live))
    pol = Wt.Policy.from_dict({"budget_usd": usd, "hard_minutes": hard, "minutes": min(float(base.minutes), hard), "gpus": list(base.gpus or b.get("gpus") or [])},
                              base)
    if pol != base:
        pol.save(dance)
    return {"budget_usd": pol.budget_usd, "hard_minutes": pol.hard_minutes, "minutes": pol.minutes, "gpus": pol.gpus}


def cmd_final_only(a, spec: DanceSpec, P: TwoView) -> int:
    """`plan --final-only`: rebuild and write ONLY the final (stitch) job from the current TOML (e.g. [export] smplx_dir = the chosen refine variant) over the profile's
    planned chunks in state.json. Refused while the final job is queued / running / done-not-pulled (the watcher is working on it), when the chunks do not tile the
    grid, and when a chunk's job does not produce the refined dir the TOML names (a typo would only show on the CPU pod). Nothing else changes."""
    from .watch import ACTIVE
    st = State(a.dance)
    cur = st.load()
    rec = cur["jobs"].get(P.final_id) or {}
    if rec.get("status") in ACTIVE:
        print(f"plan --final-only: {P.final_id} is {rec['status']} (queued on the volume / the watcher works on it): refused; plan it again once it is pulled",
              file=sys.stderr)
        return 2
    cl = sorted((c for cid, c in cur["chunks"].items() if cid.startswith(P.id_prefix + "g") and c.get("origin") == "planned" and "core_k0" in c
                 and c.get("status") not in ("failed", "cancelled")), key=lambda c: c["k0"])
    g = spec.grid
    if not tiles(cl, g):
        print(f"plan --final-only: the {P.id_prefix}g* chunks in state.json do not tile k {g.k0}..{g.k1} ({[(c['id'], c['core_k0'], c['core_k1']) for c in cl]}): "
              "run the full `plan` first", file=sys.stderr)
        return 2
    lacking = []
    for c in cl:
        p = jobs_dir(a.dance) / f"{c['id']}.json"
        outs = {o["path"] for o in json.loads(p.read_text(encoding="utf-8")).get("outputs") or []} if p.is_file() else set()
        if f"dancecap/work/{c['take']}/{spec.refined_dir}/" not in outs:
            lacking.append(c["id"])
    if lacking:
        print(f"plan --final-only: [export] smplx_dir = {spec.refined_dir!r} is not a refined dir the chunk jobs {lacking} write (their outputs): refused",
              file=sys.stderr)
        return 2
    job = final_from_volume(build_final([C.Chunk.from_dict(c) for c in cl], spec, code=C.code_info(), P=P))
    print(f"{P.final_id}: stitch of {', '.join(c['id'] for c in cl)} with the refined dir {spec.refined_dir} (from the take's [export] smplx_dir); owner {job.get('owner')}")
    if a.out:
        atomic_write_json(Path(a.out), job)
    if a.dry_run:
        return 0
    A.write_job(job)
    st.set_job(P.final_id, chunk=None, kind="stitch_analyse", status="planned", after=job.get("after") or [], written=now_iso(), spec=str(jobs_dir(a.dance) / f"{P.final_id}.json"))
    print(f"written {jobs_dir(a.dance) / (P.final_id + '.json')}   next: python -m dancecap.farm --dance {a.dance} queue --only {P.final_id}")
    if P.owner and os.environ.get("HM_QUEUE_OWNER") != P.owner:      # the watcher's CPU pod claims the owned final only with the export (launch_env_claims_the_owner)
        print(f"note: HM_QUEUE_OWNER is {os.environ.get('HM_QUEUE_OWNER')!r}: export HM_OWNER={P.owner} HM_QUEUE_OWNER={P.owner} HM_POD_OWNER={P.owner} before queue",
              file=sys.stderr)
    return 0


def busy_jobs(cur: dict, P: TwoView) -> dict:
    """{id: status} of the profile's chunks that are claimed (queued .. stitched) and of its shape / final job while the watcher works on it (queued / running /
    done-not-pulled): a full re-plan would reset the shape whose pinned betas the chunks freeze and rebuild the final without the claimed chunks."""
    from .state import CLAIMING
    from .watch import ACTIVE
    out = {cid: c.get("status") for cid, c in cur["chunks"].items() if cid.startswith(P.id_prefix + "g") and c.get("status") in CLAIMING}
    out.update({jid: cur["jobs"][jid]["status"] for jid in (P.shape_id, P.final_id) if (cur["jobs"].get(jid) or {}).get("status") in ACTIVE})
    return out


def cmd_plan(a, spec: DanceSpec, P: TwoView = CA) -> int:
    if getattr(a, "final_only", False):
        return cmd_final_only(a, spec, P)
    st = State(a.dance)
    cur = st.load()
    busy = busy_jobs(cur, P) if P.strict else {}
    if busy:
        msg = (f"plan: {', '.join(f'{k} {v}' for k, v in sorted(busy.items()))}: a full re-plan would reset {P.shape_id} (the chunks freeze its betas), "
               f"rebuild {P.final_id} without the claimed chunks and size the pods' ceiling without them. `plan --final-only` rebuilds the final, "
               "`chunks retry <id>` + `queue` re-runs a failed chunk")
        if not a.dry_run:
            print(msg + ": refused", file=sys.stderr)
            return 2
        print(msg + " (dry run: shown only)", file=sys.stderr)
    claimed = {cid for cid, c in cur["chunks"].items() if c.get("status") in ("queued", "running", "done", "pulled", "stitched")}
    claimed |= {cid[len(P.id_prefix):] for cid in claimed if cid.startswith(P.id_prefix)}
    try:
        plan = make_plan(spec, hash_inputs=not a.dry_run, code=C.code_info(), claimed_ids=claimed, P=P)
    except SyncError as e:          # no chunk is planned on an offset that was not measured
        print(f"plan: {e}", file=sys.stderr)
        return 2
    public = {k: v for k, v in plan.items() if not k.startswith("_")}
    public["validation"] = validate_plan(plan, spec, P)
    if a.json:
        print(json.dumps(public, default=str))
    else:
        print_plan(plan, P)
        v = public["validation"]
        print("validation (offline): " + ("OK" if v["ok"] else "FAILED") + " - " + ", ".join(f"{k}{'' if c['ok'] else ' FAILED'}" for k, c in v["checks"].items()))
    if a.out:
        atomic_write_json(Path(a.out), public)
    if a.dry_run:
        return 0
    if P.strict and not public["validation"]["ok"]:      # step 13's gates: no spec that failed them can be queued
        bad = [k for k, c in public["validation"]["checks"].items() if not c["ok"]]
        print(f"plan: validation FAILED ({', '.join(bad)}): nothing written (`plan --dry-run --out FILE` has the details)", file=sys.stderr)
        return 2
    shape_rec = cur["jobs"].get(P.shape_id) or {}
    if P.strict and shape_rec.get("status") not in (None, "planned"):
        print(f"note: {P.shape_id} was {shape_rec['status']}: this plan sets it back to planned; `queue` runs the shape pass again (new betas) before any chunk")
    jobs = plan["_jobs"]
    for j in jobs:
        (A.write_job if j.get("what") == "stitch_analyse" else C.write_job)(j)
    st.put_chunks([c.to_dict() for c in plan["_chunks"]], replace_planned=True)
    for j in jobs:
        st.set_job(j["id"], chunk=j["id"] if j.get("what") != "stitch_analyse" else None, kind=j.get("what") or j["kind"], status="planned", after=j.get("after") or [],
                   written=now_iso(), spec=str(jobs_dir(a.dance) / f"{j['id']}.json"))
    atomic_write_json(farm_dir(a.dance) / "plan.json", public)
    atomic_write_json(P.work / "chunks.json", public)
    if public.get("budget"):
        pol = save_budget(a.dance, public["budget"])
        print(f"farm policy: budget ${pol['budget_usd']:.2f}, pod hard ceiling {pol['hard_minutes']:g} min, kill-switch lead {pol['minutes']:g} min, "
              f"GPUs {pol['gpus'] or 'pods.GPU_PRIORITY'} (state.json)")
    if P.strict:        # not `farm up`: it loads the policy before its own plan and re-saves that one, dropping the budget written above
        nxt = (f"python -m dancecap.farm --dance {a.dance} queue --only {P.shape_id}; python -m dancecap.farm --dance {a.dance} launch --pods 1 --gpus 4090 "
               f"(the same shell: HM_QUEUE_OWNER={P.owner}; not `up`, which re-saves its earlier policy without this budget)")
    else:
        nxt = f"python -m dancecap.farm --dance {a.dance} up --dry-run"
    print(f"{len(jobs)} job spec(s) -> {jobs_dir(a.dance)}   chunk plan -> {P.work / 'chunks.json'}   next: {nxt}")
    return 0
