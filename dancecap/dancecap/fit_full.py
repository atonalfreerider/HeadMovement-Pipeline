"""larissa_kadu_full: the four MAMMA fits (segments a-d), one Runpod pod each, independent and resumable.

    python -m dancecap.fit_full check a,b,c,d         # local: verify mamma/prepare.json + ma_2d/ma_cap of every segment
    python -m dancecap.fit_full write-jobs a,b,c,d    # local: work/<seg>/fit/pod_job.json + work/<seg>/refine/pod_job.json
    python -m dancecap.fit_full plan a,b,c,d          # local: build the bundles, upload size / time, budget, FITS_READY.md
    python -m dancecap.fit_full preflight <take> [--local]   # pod (and local smoke): GPU, imports, inputs, pinned-beta fit
    python -m dancecap.fit_full verify <take>         # pod / local: the finished fit (T, betas == pin, finite, jumps)
    python -m dancecap.fit_full launch a,b,c,d        # local: DETACHED runners (one process and one pod per segment)
    python -m dancecap.fit_full session --seg a       # local (foreground): one segment's session
    python -m dancecap.fit_full resume --seg a        # local: re-attach to the segment's session / live pod
    python -m dancecap.fit_full extend --seg a --add-min 10   # move the on-pod kill switch (ceiling 150 min / $5)
    python -m dancecap.fit_full status

Per segment ONE session (remote_jobs machinery: bundle, upload, bootstrap, job commands, packed + sha256-checked
download, pod deleted at the end) with two jobs on the same pod:

  fit      preflight (GPU, imports, pytorch_sdf, inputs, a 20-frame smoke fit with the betas PINNED and checked) ->
           mamma fit --tag c<n> (full, betas pinned to larissa_kadu_38's per dancer) -> mamma select (promotes smplx/)
           -> verify.  Downloaded and sha256-checked the moment it finishes.
  refine   pose_refine with the larissa_kadu_38 recipe (smplx_rf2_noslip = the defaults: 5 Hz root, 8 Hz rotation,
           image-aware root height + legs with the floor term) -> smplx_rf2_noslip/.  Downloaded the same way.
  quality  (user directive 2026-10-08: all analysis on Runpod) dancecap.fit_quality: pose_quality on the raw and the refined
           motion + the 10 s block table and the per-frame coverage arrays (n_cams, per-camera residual flags) ->
           work/<take>/quality/.  Needs only the fit, so a failed refine still gets the raw numbers.

Robustness (the home link may drop; the fit must not): every long pod command runs through dancecap.pod_detach (own
session on the pod, the ssh command only tails its log); the local pod object is swapped for one that re-attaches while
ssh is down; a lost download is retried before the pod is deleted; the on-pod kill switch is only a safety net and is
extended while the fit is healthy (ceiling 150 min / $5 per pod); `resume` re-attaches to a live pod after a crash of
the local runner. Ledger updates of the four concurrent sessions are serialised with a lock; the sum of the sessions'
worst cases is checked against the cap minus a reserve; uploads are sequential (one at a time over the 1-3 MB/s link).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import re
import shlex
import subprocess
import sys
import threading
import time
import tomllib
from pathlib import Path

import numpy as np

from . import pod_detach
from .take import PACKAGE_ROOT, WORKSPACE

FULL = "larissa_kadu_full"
REF = "larissa_kadu_38"
SEGS = ("a", "b", "c", "d")
GRID_ANCHOR = 30.151176844227862
GRID_HZ = 30.0
WORK = PACKAGE_ROOT / "work"
FIT_DIR = WORK / FULL / "fit"
PIN_FILE = WORK / FULL / "mamma_pin_betas.json"
POD_WS = "/workspace/HeadMovement"
JOBS = ("fit", "refine", "quality")   # jobs of one segment session, in this order, on the same pod
REFINE_DST = "smplx_rf2_noslip"   # the adopted larissa_kadu_38 recipe = pose_refine's defaults
PIDS = (1, 2)                     # mamma_dyn.PIDS: body_id-00 = lead, body_id-01 = follow
ROLE_OF_BODY = {0: "lead", 1: "follow"}

# per-pod ceilings (user: --max-minutes 110 --max-cost 3.6 per fit pod, hard ceiling 150 min / $5)
HARD_MIN, HARD_COST = 150.0, 5.0
START_TIMEOUT_MIN = 5.0
BAD_HOSTS = {"60.249.37.148"}    # CUDA init failed on this L40S host (2026-10-07)
RESERVE = 6.0                     # $ kept free under the cap
TIERS = [
    # H100 first (speed) when listed at <= $3.0/h: worst case = price x (max_minutes + start timeout) <= max_cost
    # (110 min at $2.69/h would be $5.16, above the $5 hard ceiling, hence 95 min for this tier)
    {"name": "h100", "gpus": ["NVIDIA H100 80GB HBM3", "NVIDIA H100 NVL", "NVIDIA H100 PCIe"],
     "max_minutes": 95.0, "max_cost": 4.6, "start_attempts": 1},   # one 5 min wait, then the next tier
    {"name": "a100", "gpus": ["NVIDIA A100-SXM4-80GB", "NVIDIA A100 80GB PCIe"],
     "max_minutes": 110.0, "max_cost": 3.6, "start_attempts": 2},
]


def seg_name(sid: str) -> str:
    return f"{FULL}_{sid}"


def seg_work(sid: str) -> Path:
    return WORK / seg_name(sid)


def seg_cfg(sid: str) -> dict:
    return tomllib.loads((PACKAGE_ROOT / "takes" / f"{seg_name(sid)}.toml").read_text(encoding="utf-8"))


def _rel(p: Path) -> str:
    return Path(p).resolve().relative_to(WORKSPACE).as_posix()


def _segs(arg: str | None) -> list[str]:
    s = [x for x in (arg or ",".join(SEGS)).replace(" ", "").split(",") if x]
    bad = [x for x in s if x not in SEGS]
    if bad:
        raise SystemExit(f"unknown segment(s) {bad}; known {list(SEGS)}")
    return s


def _prepare(sid: str) -> dict:
    return json.loads((seg_work(sid) / "mamma" / "prepare.json").read_text())


def _tag(prep: dict) -> str:
    tags = list(prep["sets"])
    if len(tags) != 1:
        raise SystemExit(f"{prep['take']}: expected exactly one prepared camera set, found {tags}")
    return tags[0]


# ------------------------------------------------------------------------------------------------ local checks


def _native_index_vetoed(work: Path, cam: str) -> dict[int, np.ndarray]:
    """{role index: bool (n_native,)} frames whose role was blanked by the identity gate (visible in *.pre_veto.npz,
    not any more in the current ma_2d_native)."""
    pre = work / "mamma" / "ma_2d_native" / f"{cam}.pre_veto.npz"
    cur = work / "mamma" / "ma_2d_native" / f"{cam}.npz"
    if not (pre.exists() and cur.exists()):
        return {}
    a = np.load(pre)["visibilities"].sum(-1)   # (n, 2)
    b = np.load(cur)["visibilities"].sum(-1)
    if a.shape != b.shape:
        return {}
    return {p: (a[:, p] > 0) & (b[:, p] <= 0) for p in (0, 1)}


def check_segment(sid: str) -> dict:
    """Verify one segment's prepare output (PLAN.md C6 and the review-gate fixes). Returns the report (also written to
    work/larissa_kadu_full/fit/prepare_check_<sid>.json); rep['ok'] is False when a hard check fails."""
    work = seg_work(sid)
    prep = _prepare(sid)
    cfg = seg_cfg(sid)
    seg = cfg["segment"]
    tag = _tag(prep)
    s = prep["sets"][tag]
    rep: dict = {"seg": sid, "take": prep["take"], "created": prep["created"], "tag": tag, "hard": {}, "soft": {},
                 "info": {}}
    hard, soft, info = rep["hard"], rep["soft"], rep["info"]
    # the prepare must be newer than every gate fix (cameras, dancers, ma_2d_native)
    prep_t = (work / "mamma" / "prepare.json").stat().st_mtime
    newest = 0.0
    for pat in ("cameras/*.npz", "mamma/ma_2d_native/*.npz", "sync/offsets.json"):
        for f in work.glob(pat):
            newest = max(newest, f.stat().st_mtime)
    hard["prepare_newer_than_inputs"] = prep_t >= newest
    info["prepare_age_vs_newest_input_s"] = round(prep_t - newest, 1)
    # grid on 38's lattice
    t = np.asarray(prep["grid"]["t"], float)
    p0, p1 = (int(x) for x in seg["fit_k"])   # planned fit_k; prepare trims it to every camera's sync-corrected coverage
    tr = prep["grid"]["trim"]
    k0, k1 = (int(x) for x in (tr.get("grid_k") or [p0, p1]))
    ks = np.arange(k0, k1 + 1)
    on_lattice = len(t) == len(ks) and float(np.abs(t - (GRID_ANCHOR + ks / GRID_HZ)).max()) < 1e-6
    hard["grid_on_38_lattice"] = bool(on_lattice and abs(tr.get("grid_anchor", 0) - GRID_ANCHOR) < 1e-12)
    # a 1-3 frame trim at an end (the cameras' sync-corrected coverage: c 2194..3477, d 3360..4645) is normal
    hard["grid_k_is_fit_k"] = prep["grid"]["T"] == len(ks) and 0 <= k0 - p0 <= 3 and 0 <= p1 - k1 <= 3
    hard["coverage_union"] = tr.get("coverage") == "union"
    info["grid"] = {"T": int(len(t)), "k": [k0, k1], "planned_fit_k": [p0, p1], "trimmed_frames": [k0 - p0, p1 - k1],
                    "t": [round(float(t[0]), 4), round(float(t[-1]), 4)]}
    hard["single_set"] = len(prep["sets"]) == 1
    hard["world_frame"] = bool(prep.get("world_frame"))
    hard["sync_required_and_used"] = bool(prep["sync"]["required"] and prep["sync"]["file"])
    info["sync_offsets_ms"] = prep["sync"]["offsets_ms"]
    # camera sets
    cams = list(s["cams"])
    exp = list(seg.get("cams", cams))
    info["cams"] = cams
    soft["cams_equal_toml"] = sorted(cams) == sorted(exp)
    if not soft["cams_equal_toml"]:
        info["cams_toml"] = exp
    # inputs
    seq = s["seq"]
    d2 = work / "mamma" / "ma_2d" / seq
    dg = work / "mamma" / "ma_cap" / seq / "gt"
    hard["inputs_exist"] = all((d2 / f"cam{c}.npz").exists() and (dg / f"cam{c}.npz").exists() for c in cams) and \
        (work / "mamma" / "ma3d_config.yaml").exists()
    if not hard["inputs_exist"]:
        rep["ok"] = False
        _write_check(sid, rep)
        return rep
    vis, focal, ncams_t = {}, {}, np.zeros(len(t), int)
    per_cam: dict = {}
    for c in cams:
        z = np.load(d2 / f"cam{c}.npz")
        v = z["visibilities"].sum(-1) > 0              # (T, 2) any landmark visible per role
        vis[c] = v
        ncams_t += v.any(1)
        g = np.load(dg / f"cam{c}.npz")
        K = g["cam_int"]
        ok = v.any(1)
        f = K[:, 0, 0].astype(float)
        focal[c] = f
        fv = f[ok] if ok.any() else f
        per_cam[c] = {"frames_any": int(ok.sum()), "lead_frames": int(v[:, 0].sum()), "follow_frames": int(v[:, 1].sum()),
                      "focal_px": [round(float(fv.min()), 1), round(float(np.median(fv)), 1), round(float(fv.max()), 1)],
                      "focal_rel_range": round(float((fv.max() - fv.min()) / max(np.median(fv), 1e-9)), 4),
                      "n_distinct_focal": int(len(np.unique(np.round(fv, 3))))}
    info["per_cam"] = per_cam
    empty = [c for c in cams if per_cam[c]["frames_any"] == 0]
    soft["no_camera_without_observations"] = not empty
    if empty:
        info["cameras_with_no_observation"] = empty
    # camera types
    toml_cams = (tomllib.loads((PACKAGE_ROOT / "takes" / f"{FULL}.toml").read_text(encoding="utf-8"))).get("cameras", {})
    types_ok, ct = True, {}
    for c in cams:
        zoom = bool(toml_cams.get(c, {}).get("zoom"))
        pt = prep["camera_types"].get(c, {})
        want_cls = "handheld_zoom" if zoom else "handheld"
        ok_cls = pt.get("class") == want_cls
        pc = per_cam[c]
        zoom_pf = (pt.get("model") == "per_frame") and pc["n_distinct_focal"] > 1 if zoom else True
        ct[c] = {"declared_zoom": zoom, "prepared": pt, "per_frame_focal_varies": pc["n_distinct_focal"] > 1,
                 "ok": bool(ok_cls and zoom_pf)}
        types_ok &= ct[c]["ok"]
    hard["camera_types_match_user_declaration"] = types_ok
    info["camera_types"] = ct
    # n cams per frame, mono span
    info["n_cams_hist"] = {int(k): int((ncams_t == k).sum()) for k in np.unique(ncams_t)}
    mono = np.flatnonzero(ncams_t == 1)
    zero = np.flatnonzero(ncams_t == 0)
    info["mono_frames"] = int(len(mono))
    if len(mono):
        info["mono_span_s"] = [round(float(t[mono[0]]), 3), round(float(t[mono[-1]]), 3)]
        info["mono_cameras"] = sorted({c for c in cams if vis[c][mono].any()})
    info["frames_with_no_camera"] = int(len(zero))
    soft["no_frame_without_camera"] = len(zero) == 0
    if sid == "a":
        mono_exp = (t >= -3.14) & (t < 0.0)
        only09 = [c for c in cams if vis[c][mono_exp].any()]
        hard["a_mono_span_is_cam09_only"] = only09 == ["09"] or set(only09) <= {"09", "06"} and \
            bool(not vis.get("06", np.zeros((len(t), 2), bool))[(t >= -3.14) & (t < -0.1)].any())
        info["a_cameras_seeing_anyone_before_0s"] = only09
        if "09" in vis:
            lead_early = vis["09"][t < -0.6, 0]
            hard["a_cam09_lead_absent_before_entry"] = int(lead_early.sum()) == 0
            info["a_cam09_lead_frames_before_-0.6s"] = int(lead_early.sum())
    # vetoed frames excluded (identity gate): grid frames whose nearest native frame was blanked must be empty
    ve = {}
    leaks = 0       # grid frames whose nearest native frame was vetoed but still carry landmarks
    leaks_bad = 0   # camera/role pairs with more than max(3, 2 %) of them (1-2 frames next to a run end are the
    #                 bracket interpolation between a vetoed and a kept native frame)
    for c in cams:
        vet = _native_index_vetoed(work, c)
        if not vet:
            continue
        ni = np.load(dg / f"cam{c}.npz")["native_index"]
        ok_i = (ni >= 0) & (ni < len(vet[0]))
        for p in (0, 1):
            vm = np.zeros(len(t), bool)
            vm[ok_i] = vet[p][ni[ok_i]]
            if vm.any():
                leak = int((vis[c][vm, p]).sum())
                ve[f"{c}:{ROLE_OF_BODY[p]}"] = {"vetoed_grid_frames": int(vm.sum()), "still_visible": leak}
                leaks += leak
                leaks_bad += int(leak > max(3, 0.02 * int(vm.sum())))
    info["vetoed_frames"] = ve
    info["vetoed_frames_leaked_total"] = leaks
    hard["vetoed_frames_excluded"] = leaks_bad == 0
    # betas pin file
    pin = json.loads(PIN_FILE.read_text()) if PIN_FILE.exists() else {}
    src = json.loads(PIN_FILE.with_name(PIN_FILE.stem + ".source.json").read_text()) if PIN_FILE.exists() else {}
    hard["pin_file_keys_are_body_ids"] = sorted(pin) == ["0", "1"] and \
        [src.get(k, {}).get("role") for k in ("0", "1")] == ["lead", "follow"]
    ok_all = all(hard.values())
    rep["ok"] = bool(ok_all)
    rep["soft_warnings"] = [k for k, v in soft.items() if not v]
    _write_check(sid, rep)
    return rep


def _write_check(sid: str, rep: dict) -> None:
    FIT_DIR.mkdir(parents=True, exist_ok=True)
    (FIT_DIR / f"prepare_check_{sid}.json").write_text(json.dumps(rep, indent=1, default=str))


# ------------------------------------------------------------------------------------------------ job specs


def _detached(job: str, name: str, cmd: str, timeout_min: float) -> dict:
    return {"name": name, "cmd": pod_detach.detached(name, cmd, job, timeout_min), "timeout_min": timeout_min + 6}


# Slim bundle (the uplink is the slowest shared resource: 0.9-2.8 MB/s for four uploads). A MAMMA fit pod needs neither the
# front-end wheels (opencv-python non-headless, polars, ultralytics: 120 MB; opencv-python-headless stays) nor the male /
# female SMPL-X models (2 x 109 MB): get_smplx_models builds all three but the fit only ever uses 'neutral'
# (run_ma_3d.py gender = "neutral"), so the pod symlinks them to the neutral file. fit_full preflight proves it on
# the pod (a real short fit) before the long one. `write-jobs --full-bundle` turns this off.
_PYW = "dancecap/work/runpod/pywheels/cp311-linux"
_LH = "mamma/data/body_models/smplx_locked_head/smplx"
SLIM_EXCLUDE = [f"{_PYW}/opencv_python-*", f"{_PYW}/polars*", f"{_PYW}/ultralytics-*", f"{_LH}/SMPLX_MALE.npz",
                f"{_LH}/SMPLX_FEMALE.npz"]
SLIM_SETUP = [f"for g in MALE FEMALE; do ln -sf SMPLX_NEUTRAL.npz {POD_WS}/{_LH}/SMPLX_$g.npz; done; "
              f"ls -l {POD_WS}/{_LH}/"]


def job_specs(sid: str, tier: dict | None = None, slim: bool = True) -> dict:
    """{'fit': pod_job/1 dict, 'refine': pod_job/1 dict} for one segment."""
    prep = _prepare(sid)
    take = prep["take"]
    tag = _tag(prep)
    seq = prep["sets"][tag]["seq"]
    w = f"dancecap/work/{take}"
    ws = f"cd {POD_WS}/dancecap &&"
    pin = f"{POD_WS}/dancecap/work/{FULL}/mamma_pin_betas.json"
    env = {"DANCECAP_POD": "1", "PYOPENGL_PLATFORM": "egl", "MAMMA_INIT_BETAS": pin, "MAMMA_FREEZE_BETAS": "1"}
    inputs = [f"{w}/mamma/prepare.json", f"{w}/mamma/ma3d_config.yaml", f"{w}/mamma/ma_2d/{seq}/",
              f"{w}/mamma/ma_cap/{seq}/", f"dancecap/work/{FULL}/mamma_pin_betas.json",
              f"dancecap/work/{FULL}/mamma_pin_betas.source.json"]
    fit_outputs = [f"{w}/mamma/fits/", f"{w}/mamma/ma_3d/", f"{w}/mamma/logs/", f"{w}/mamma/select.json",
                   f"{w}/mamma/report.json", f"{w}/mamma/preflight.json", f"{w}/mamma/fit_check.json", f"{w}/smplx/"]
    fit = {
        "schema": "dancecap.pod_job/1", "name": "fit", "take": take, "max_minutes": 80, "needs": ["mamma"],
        "inputs": inputs, "env": env, "cwd": "dancecap",
        **({"exclude": SLIM_EXCLUDE, "setup": SLIM_SETUP} if slim else {}),
        "commands": [
            _detached("fit", "preflight", f"{ws} python -m dancecap.fit_full preflight {take}", 12),
            # 130 min: the pod's kill switch (and its auto-extension, ceiling 150 min) is the bound, not this timeout
            _detached("fit", f"fit_{tag}", f"{ws} python -m dancecap mamma fit {take} --tag {tag}", 130),
            _detached("fit", "select", f"{ws} python -m dancecap mamma select {take} --no-qa", 8),
            _detached("fit", "verify", f"{ws} python -m dancecap.fit_full verify {take}", 6) | {"optional": True},
        ],
        "outputs": fit_outputs,
        "verify": [f"{w}/mamma/fits/{tag}/fit.json", f"{w}/smplx/lead.npz", f"{w}/smplx/follow.npz",
                   f"{w}/mamma/preflight.json"],
        "notes": [f"larissa_kadu_full fit {sid}: full MAMMA ma_3d fit of {take} ({tag}), betas pinned to "
                  f"larissa_kadu_38's (MAMMA_INIT_BETAS + MAMMA_FREEZE_BETAS), promoted to smplx/ by select",
                  "downloaded and sha256-checked right after select/verify, before the refinement job"],
    }
    refine = {
        "schema": "dancecap.pod_job/1", "name": "refine", "take": take, "max_minutes": 12, "needs": ["mamma"],
        "after": ["fit"], "inputs": [], "env": {"DANCECAP_POD": "1"}, "cwd": "dancecap",
        "commands": [_detached("refine", "refine", f"{ws} python -m dancecap.pose_refine {take} --src smplx "
                                                   f"--dst {REFINE_DST} --device cuda", 20)],
        "outputs": [f"{w}/{REFINE_DST}/"],
        "verify": [f"{w}/{REFINE_DST}/lead.npz", f"{w}/{REFINE_DST}/follow.npz"],
        "notes": [f"pose_refine with larissa_kadu_38's adopted recipe ({REFINE_DST} = the defaults: root 5 Hz, "
                  "rotation 8 Hz, image-aware root height + legs with the floor term, 400 iterations)",
                  "the raw fit was already downloaded by the 'fit' job; a refine failure loses nothing else"],
    }
    quality = {
        # user directive 2026-10-08: all analysis compute runs on Runpod. pose_quality (reprojection per camera / dancer /
        # 10 s block incl. the zoom cameras 04 / 05 / 07 / 09 and the attached 05 / 06, feet and floor, slip, jitter) on the
        # raw and the refined motion, plus the farm's per-frame coverage arrays (fit_quality.py). Needs the fit only (a
        # failed refine still gets the raw numbers); Take.load reads the alignment json (no videos).
        "schema": "dancecap.pod_job/1", "name": "quality", "take": take, "max_minutes": 15, "needs": ["mamma"],
        "after": ["fit"], "inputs": ["prototype/larissa_kadu/alignment_studio.json"],
        "env": {"DANCECAP_POD": "1"}, "cwd": "dancecap",
        "commands": [_detached("quality", "fit_quality", f"{ws} python -m dancecap.fit_quality {take} "
                                                         f"--refined {REFINE_DST}", 25)],
        "outputs": [f"{w}/quality/"],
        "verify": [f"{w}/quality/fit_quality.json", f"{w}/quality/fit_coverage.npz",
                   f"{w}/quality/quality_smplx.json"],
        "notes": ["pose_quality on smplx/ and smplx_rf2_noslip/ + fit_quality's 10 s block table and per-frame arrays "
                  "(n_cams, per-camera residual flags) for the coverage map"],
    }
    return {"fit": fit, "refine": refine, "quality": quality}


def write_jobs(sids: list[str], slim: bool = True) -> list[Path]:
    out = []
    for sid in sids:
        for name, spec in job_specs(sid, slim=slim).items():
            p = seg_work(sid) / name / "pod_job.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(spec, indent=1))
            out.append(p)
            print(f"[fit_full] {p.relative_to(PACKAGE_ROOT)}")
    return out


# ------------------------------------------------------------------------------------------------ pod: preflight / verify


def _pin() -> dict[int, np.ndarray]:
    d = json.loads(PIN_FILE.read_text())
    return {int(k): np.asarray(v, float) for k, v in d.items()}


def _smoke_config(src: Path, dst: Path, div: int = 10) -> Path:
    import yaml
    cfg = yaml.safe_load(src.read_text())
    for run in (cfg.get("optim") or {}).values():
        if isinstance(run, dict) and "num_iters" in run:
            run["num_iters"] = max(2, int(run["num_iters"]) // div)
    dst.write_text("# dancecap fit_full preflight: ma3d_config.yaml with every stage's num_iters / %d\n%s"
                   % (div, yaml.safe_dump(cfg, sort_keys=False)))
    return dst


def preflight(take: str, frames: int = 20, local: bool = False, smoke_div: int = 10) -> dict:
    """GPU, imports, inputs, and a short REAL ma_3d run (same code path as the fit: pinned betas, export) on the
    middle `frames` grid frames with every stage's iterations divided by `smoke_div`. Raises SystemExit(1) when a
    check fails (the session then stops early and cheaply)."""
    from . import mamma_dyn as md
    t0 = time.time()
    work = md.take_work(take)
    root = work / "mamma"
    rep: dict = {"take": take, "host": "local" if local else "pod", "checks": {}, "t0": time.strftime("%FT%T")}
    ck = rep["checks"]

    def check(name: str, ok: bool, **kw):
        ck[name] = {"ok": bool(ok), **kw}
        print(f"[preflight] {'ok  ' if ok else 'FAIL'} {name} {json.dumps(kw, default=str)[:300]}", flush=True)

    try:
        import torch
        cuda = torch.cuda.is_available()
        info = {"torch": torch.__version__, "cuda": torch.version.cuda}
        if cuda:
            props = torch.cuda.get_device_properties(0)
            a = torch.randn(4096, 4096, device="cuda")
            torch.cuda.synchronize()
            t = time.time()
            for _ in range(20):
                a = a @ a / 64.0
            torch.cuda.synchronize()
            info.update(gpu=props.name, mem_gb=round(props.total_memory / 2**30, 1), capability=list(
                torch.cuda.get_device_capability(0)), matmul_tflops=round(20 * 2 * 4096 ** 3 / (time.time() - t) / 1e12, 1))
        check("gpu", cuda and (local or info["mem_gb"] >= 40), **info)
    except Exception as e:  # noqa: BLE001
        check("gpu", False, error=f"{type(e).__name__}: {e}")
    for mod in ("smplx", "pytorch_sdf", "dancecap.mamma_dyn", "dancecap.pose_refine", "dancecap.pod_detach",
                "dancecap.fit_full", "scipy", "yaml", "cv2"):
        try:
            __import__(mod)
            check(f"import {mod}", True)
        except Exception as e:  # noqa: BLE001
            check(f"import {mod}", False, error=f"{type(e).__name__}: {str(e)[:160]}")
    try:
        import pytorch_sdf  # noqa: F401
        from pytorch_sdf import sdf  # noqa: F401
        check("pytorch_sdf.sdf", True)
    except Exception as e:  # noqa: BLE001
        check("pytorch_sdf.sdf", False, error=f"{type(e).__name__}: {str(e)[:160]}")
    if not local:
        try:
            import pyrender  # noqa: F401
            check("import pyrender", True)
        except Exception as e:  # noqa: BLE001
            check("import pyrender", False, error=str(e)[:160])
    # inputs
    try:
        prep = json.loads((root / "prepare.json").read_text())
        tag = _tag(prep)
        s = prep["sets"][tag]
        seq, cams = s["seq"], s["cams"]
        T = int(prep["grid"]["T"])
        d2, dg = root / "ma_2d" / seq, root / "ma_cap" / seq / "gt"
        missing = [c for c in cams if not ((d2 / f"cam{c}.npz").exists() and (dg / f"cam{c}.npz").exists())]
        bad_T = []
        for c in cams:
            if c in missing:
                continue
            z = np.load(d2 / f"cam{c}.npz")
            g = np.load(dg / f"cam{c}.npz")
            if z["landmarks"].shape[0] != T or g["cam_ext"].shape[0] != T or not np.isfinite(g["cam_ext"]).all():
                bad_T.append(c)
        check("inputs", not missing and not bad_T, T=T, cams=cams, missing=missing, bad_T_or_nan=bad_T)
    except Exception as e:  # noqa: BLE001
        check("inputs", False, error=f"{type(e).__name__}: {e}")
        return _finish_preflight(rep, root, t0)
    # pin file + env
    try:
        pin = _pin()
        env_ok = os.environ.get("MAMMA_FREEZE_BETAS") == "1" and bool(os.environ.get("MAMMA_INIT_BETAS")) and \
            Path(os.environ["MAMMA_INIT_BETAS"]).resolve() == PIN_FILE.resolve()
        check("pin env", env_ok or local, init=os.environ.get("MAMMA_INIT_BETAS"),
              freeze=os.environ.get("MAMMA_FREEZE_BETAS"))
        check("pin file", sorted(pin) == [0, 1] and all(len(v) == 16 for v in pin.values()),
              n={k: len(v) for k, v in pin.items()})
    except Exception as e:  # noqa: BLE001
        check("pin file", False, error=str(e)[:200])
        return _finish_preflight(rep, root, t0)
    if any(not v["ok"] for v in ck.values()):
        return _finish_preflight(rep, root, t0)
    # smoke fit
    k = frames
    f0 = max(0, T // 2 - k // 2)
    fr = (f0, f0 + k)
    smoke = _smoke_config(root / prep["config"], root / "ma3d_config_smoke.yaml", smoke_div)
    run_tag = f"preflight_f{fr[0]}-{fr[1]}"
    t1 = time.time()
    env_keep = {x: os.environ.get(x) for x in ("MAMMA_INIT_BETAS", "MAMMA_FREEZE_BETAS")}
    if local:  # the local run must pin too
        os.environ["MAMMA_INIT_BETAS"], os.environ["MAMMA_FREEZE_BETAS"] = str(PIN_FILE), "1"
    try:
        run_dir, stats = md.run_ma3d(work, seq, [f"cam{c}" for c in cams], smoke, tag=run_tag, frames=fr)
        log = (root / "logs" / f"ma_3d_{run_tag}.log").read_text(encoding="utf-8", errors="replace")
        pinned = len(re.findall(r"betas PINNED from", log))
        out = root / "fits" / run_tag
        grid = np.asarray(prep["grid"]["t"], float)
        md.export_smplx(run_dir, grid, dg, d2, out / "smplx", frame0=fr[0])
        dev, finite_all = {}, True
        for body_id, role in ROLE_OF_BODY.items():
            z = np.load(out / "smplx" / f"{role}.npz")
            dev[role] = float(np.abs(np.asarray(z["betas"], float).reshape(-1) - pin[body_id]).max())
            finite_all &= bool(np.isfinite(z["transl"]).all() and np.isfinite(z["body_pose"]).all())
        check("smoke fit ran", stats.get("returncode") == 0, seconds=round(time.time() - t1, 1), frames=list(fr),
              iters_div=smoke_div, vram_mib=stats.get("gpu_used_peak_mib"))
        check("sdf active", bool(stats.get("sdf_active")))
        check("betas pinned in log", pinned == 2, lines=pinned)
        check("betas == larissa_kadu_38's", max(dev.values()) < 1e-4, max_abs_dev=dev)
        check("smoke output finite", finite_all)
    except Exception as e:  # noqa: BLE001
        check("smoke fit ran", False, error=f"{type(e).__name__}: {str(e)[:300]}")
    finally:
        for x, v in env_keep.items():
            if v is None:
                os.environ.pop(x, None)
            else:
                os.environ[x] = v
    return _finish_preflight(rep, root, t0)


def _finish_preflight(rep: dict, root: Path, t0: float) -> dict:
    rep["ok"] = all(v["ok"] for v in rep["checks"].values())
    rep["seconds"] = round(time.time() - t0, 1)
    root.mkdir(parents=True, exist_ok=True)
    (root / "preflight.json").write_text(json.dumps(rep, indent=1, default=str))
    print(f"[preflight] {'PASS' if rep['ok'] else 'FAIL'} in {rep['seconds']:.0f} s", flush=True)
    if not rep["ok"]:
        raise SystemExit(1)
    return rep


def verify(take: str) -> dict:
    """The finished full fit: T == grid, t == grid, betas == pin, finite, pelvis jump statistics, reprojection."""
    from . import mamma_dyn as md
    work = md.take_work(take)
    root = work / "mamma"
    prep = json.loads((root / "prepare.json").read_text())
    tag = _tag(prep)
    grid = np.asarray(prep["grid"]["t"], float)
    pin = _pin()
    rep: dict = {"take": take, "tag": tag, "T": int(len(grid)), "roles": {}, "ok": True}
    fit = json.loads((root / "fits" / tag / "fit.json").read_text())
    for body_id, role in ROLE_OF_BODY.items():
        z = np.load(root / "fits" / tag / "smplx" / f"{role}.npz")
        tr = np.asarray(z["transl"], float)
        jump = np.linalg.norm(np.diff(tr, axis=0), axis=1)
        r = {"frames": int(len(z["t"])), "t_equals_grid": bool(len(z["t"]) == len(grid) and
                                                              np.abs(np.asarray(z["t"], float) - grid).max() < 1e-6),
             "betas_max_dev_from_pin": float(np.abs(np.asarray(z["betas"], float).reshape(-1) - pin[body_id]).max()),
             "finite": bool(np.isfinite(tr).all() and np.isfinite(np.asarray(z["body_pose"], float)).all()),
             "pelvis_step_max_m": round(float(jump.max()), 4), "pelvis_steps_over_0.15m": int((jump > 0.15).sum()),
             "repaired_frames": int(np.asarray(z["repaired"]).sum()),
             "median_reproj_px": round(float(np.nanmedian(z["reproj_px"])), 2),
             "quality_median": round(float(np.median(z["quality"])), 3)}
        r["ok"] = r["t_equals_grid"] and r["betas_max_dev_from_pin"] < 1e-4 and r["finite"]
        rep["ok"] &= r["ok"]
        rep["roles"][role] = r
    rep["ma_3d"] = {k: fit["ma_3d"].get(k) for k in ("seconds", "gpu_used_peak_mib", "sdf_active", "torch_peak")}
    ref = WORK / REF / "mamma" / "fit_summary_38.json"
    if ref.exists():  # local only: the 38 s fit's medians for comparison
        rep["ref_38_median_px"] = {c: {r: v[r]["median_px"] for r in ("lead", "follow")}
                                   for c, v in json.loads(ref.read_text()).get("per_cam", {}).items()}
    (root / "fit_check.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps({k: rep[k] for k in ("take", "tag", "T", "ok")}), flush=True)
    for role, r in rep["roles"].items():
        print(f"[verify] {role}: {r}", flush=True)
    if not rep["ok"]:
        raise SystemExit(1)
    return rep


# ------------------------------------------------------------------------------------------------ local: locks, budget


@contextlib.contextmanager
def _dirlock(path: Path, stale_s: float = 120.0, wait_s: float = 120.0):
    """Cross-process mutex by atomic mkdir (works on Windows); a lock older than stale_s is taken over."""
    t_end = time.time() + wait_s
    path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            path.mkdir()
            break
        except FileExistsError:
            try:
                if time.time() - path.stat().st_mtime > stale_s:
                    path.rmdir()
                    continue
            except OSError:
                pass
            if time.time() > t_end:
                raise TimeoutError(f"lock {path} busy")
            time.sleep(0.05 + random.random() * 0.15)
    try:
        yield
    finally:
        with contextlib.suppress(OSError):
            path.rmdir()


_PATCHED = False
_KEEP: set[str] = set()   # pods whose session ended with a command still running: pod_session's exit keeps them


def patch_ledger_lock() -> None:
    """The four concurrent sessions read-modify-write work/runpod/ledger.json: serialise remote._record/_delete
    (this process only; remote.py itself is untouched)."""
    global _PATCHED
    if _PATCHED:
        return
    from . import remote
    lock = remote.LEDGER.parent / "ledger.lock"

    def wrap(fn):
        def inner(*a, **k):
            with _dirlock(lock, stale_s=30, wait_s=60):
                return fn(*a, **k)
        inner.__name__ = fn.__name__
        return inner

    _orig_delete = wrap(remote._delete)

    def _delete_unless_kept(pod_id, reason):
        if pod_id in _KEEP and reason == "session end":
            print(f"[runpod] pod {pod_id} kept running (a detached command is still going; kill switch armed)", flush=True)
            return None
        return _orig_delete(pod_id, reason)

    remote._record = wrap(remote._record)
    remote._delete = _delete_unless_kept
    _PATCHED = True


def _pid_alive(pid: int | None) -> bool:
    return bool(pid) and pod_detach._alive(int(pid))


def commitments() -> dict:
    p = FIT_DIR / "commitments.json"
    try:
        d = json.loads(p.read_text())
    except (OSError, ValueError):
        d = {}
    return {k: v for k, v in d.items() if _pid_alive(v.get("pid"))}


def commit(seg: str, worst: float) -> None:
    with _dirlock(FIT_DIR / "commitments.lock"):
        d = commitments()
        d[seg] = {"worst": round(worst, 2), "pid": os.getpid(), "at": time.strftime("%FT%T")}
        (FIT_DIR / "commitments.json").write_text(json.dumps(d, indent=1))


def release(seg: str) -> None:
    with _dirlock(FIT_DIR / "commitments.lock"):
        d = commitments()
        d.pop(seg, None)
        (FIT_DIR / "commitments.json").write_text(json.dumps(d, indent=1))


def budget_gate(seg: str, worst: float) -> dict:
    """Each session's worst case <= balance - $2, and this + the other live sessions' worst cases <= remaining cap -
    reserve."""
    from . import remote
    st = remote.status()
    bal = float((st.get("account") or {}).get("clientBalance") or 0.0)
    others = sum(v["worst"] for k, v in commitments().items() if k != seg)
    ok = worst <= bal - 2.0 and worst + others <= st["remaining"] - RESERVE
    return {"seg": seg, "worst": round(worst, 2), "others_worst": round(others, 2), "balance": round(bal, 2),
            "cap": st["cap"], "spent": st["spent"], "remaining": st["remaining"], "reserve": RESERVE, "ok": bool(ok)}


class UploadSlot:
    """One bundle upload at a time over the home uplink (1-3 MB/s), in launch `order`. A session registers a ticket,
    may create its pod while at most one session is ahead of it (`wait_turn(1)`: the pod boots while the previous
    segment uploads, so the uplink never idles), and takes the upload lock right before its upload (`acquire`). Dead
    waiters are ignored; `release` frees the lock and the ticket."""

    def __init__(self, seg: str, order: int, log=print, poll_s: float = 8.0):
        self.seg, self.order, self.log, self.poll_s = seg, order, log, poll_s
        self.qdir = FIT_DIR / "upload_queue"
        self.ticket = self.qdir / f"{order:03d}_{seg}.json"
        self.lock = FIT_DIR / "upload.lock"
        self.held = False

    def register(self) -> None:
        self.qdir.mkdir(parents=True, exist_ok=True)
        tmp = self.ticket.with_suffix(".tmp")      # atomic: a waiter must never read a half-written ticket
        tmp.write_text(json.dumps({"seg": self.seg, "pid": os.getpid(), "order": self.order, "at": time.time()}))
        os.replace(tmp, self.ticket)

    def _ahead(self) -> list[str]:
        ahead = []
        for f in sorted(self.qdir.glob("*.json")):
            try:
                d = json.loads(f.read_text())
            except (OSError, ValueError):
                d = None                      # unreadable: count it (order and segment are in the file name)
            if d is not None and not _pid_alive(d.get("pid")):
                with contextlib.suppress(OSError):
                    f.unlink()
                continue
            try:
                order = int(f.name.split("_", 1)[0])
            except ValueError:
                continue
            if f != self.ticket and (order, f.name) < (self.order, self.ticket.name):
                ahead.append(f.stem.split("_", 1)[-1])
        return ahead

    def wait_turn(self, max_ahead: int = 0) -> None:
        """Block until at most `max_ahead` live sessions are ahead of this one in the queue."""
        if not self.ticket.exists():
            self.register()
        said = False
        while True:
            ahead = self._ahead()
            if len(ahead) <= max_ahead:
                return
            if not said:
                self.log(f"waiting for my turn (ahead: {ahead}; may start when <= {max_ahead} are ahead)")
                said = True
            time.sleep(self.poll_s)

    def acquire(self) -> None:
        """Take the upload lock (only when nobody is ahead in the queue)."""
        self.wait_turn(0)
        said = False
        while True:
            try:
                self.lock.mkdir()
                break
            except FileExistsError:
                try:
                    if time.time() - self.lock.stat().st_mtime > 3 * 3600:
                        self.lock.rmdir()
                        continue
                except OSError:
                    pass
            if not said:
                self.log("waiting for the running upload to finish")
                said = True
            time.sleep(self.poll_s)
        self.held = True
        self.log("upload slot acquired")

    def release(self) -> None:
        if self.held:
            with contextlib.suppress(OSError):
                self.lock.rmdir()
            self.held = False
        with contextlib.suppress(OSError):
            self.ticket.unlink()


# ------------------------------------------------------------------------------------------------ local: session


def host_facts(pod, seg: str, probe_mb: int = 8) -> dict:
    out = pod.output(
        "q=$(cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo \"$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us) "
        "$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)\"); echo CPU $q; "
        "m=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes); echo MEM $m; "
        "echo AVAIL $(awk '/MemAvailable/ {print $2}' /proc/meminfo); echo DISK $(df -B1 --output=avail /workspace | tail -1); "
        "echo NPROC $(nproc)", timeout=60)
    f: dict = {}
    for line in out.splitlines():
        k, _, v = line.partition(" ")
        f[k] = v.strip()
    try:
        q, per = f.get("CPU", "max 100000").split()[:2]
        cpus = (int(q) / int(per)) if q not in ("max", "-1") else float(f.get("NPROC", 0))
    except (ValueError, ZeroDivisionError):
        cpus = float(f.get("NPROC", 0) or 0)
    try:
        mem = int(f.get("MEM", "max")) / 2**30
        if mem > 4096:
            raise ValueError
    except ValueError:
        mem = int(f.get("AVAIL", 0) or 0) / 2**20
    FIT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = FIT_DIR / f"_upload_probe_{seg}.bin"
    tmp.write_bytes(os.urandom(probe_mb * 2**20))
    t0 = time.time()
    try:
        pod.put(tmp, "/workspace/_upload_probe.bin", timeout=120)
        up = probe_mb * 2**20 / 1e6 / max(time.time() - t0, 1e-3)
    except Exception:  # noqa: BLE001
        up = 0.0
    tmp.unlink(missing_ok=True)
    return {"cpus": round(cpus, 1), "mem_gb": round(mem, 1), "disk_gb": round(int(f.get("DISK", 0) or 0) / 1e9, 1),
            "upload_mb_s": round(up, 2), "host": pod.host, "gpu": pod.gpu, "cost_per_hr": pod.cost_per_hr}


def _upload_bundle(ses, pod, bundle: Path, man: dict) -> None:
    from . import remote_jobs as rj
    remote_b = f"{rj.POD_UPLOAD}/{bundle.name}"
    have = pod.output(f"mkdir -p {rj.POD_UPLOAD}; [ -f {remote_b} ] && sha256sum {remote_b} | cut -d' ' -f1 || true")
    if have.strip() == man["sha256"]:
        ses.log("upload: pod already has this bundle")
        return
    pod.put(bundle, remote_b + ".part", timeout=max(900, man["bytes"] / 1e6 / 0.3))
    got = pod.output(f"sha256sum {remote_b}.part | cut -d' ' -f1", timeout=600).strip()
    if got != man["sha256"]:
        raise RuntimeError(f"upload sha256 mismatch {got} != {man['sha256']}")
    pod.output(f"mv {remote_b}.part {remote_b}")


def killswitch_status(pod) -> dict:
    out = pod.output("echo NOW $(date +%s); echo KILL $(cat /workspace/.dancecap_kill_at 2>/dev/null || echo 0)", timeout=40)
    d = dict(x.split() for x in out.split("\n") if x.strip())
    return {"now": int(d["NOW"]), "kill_at": int(d["KILL"])}


def extend_killswitch(pod, created_ts: float, cost_per_hr: float, add_min: float, log=print) -> dict:
    """Move the on-pod kill switch `add_min` later, refusing to go past 150 min of lifetime or $5 per pod."""
    from . import remote
    ks = killswitch_status(pod)
    left = (ks["kill_at"] - ks["now"]) / 60
    life_now = (time.time() - created_ts) / 60
    new_left = left + add_min
    total = life_now + new_left
    cost = cost_per_hr * total / 60
    if total > HARD_MIN or cost > HARD_COST:
        room = min(HARD_MIN - life_now - left, HARD_COST / cost_per_hr * 60 - life_now - left)
        if room < 1:
            return {"extended": False, "why": f"ceiling reached (life {life_now:.0f} min, {left:.0f} min left)"}
        add_min = room
        new_left = left + add_min
    pod.run("pkill -f '[d]ancecap_kill_at' || true", check=False, timeout=30, echo=False)
    info = remote.arm_killswitch(pod, new_left)
    log(f"kill switch moved: {left:.1f} -> {new_left:.1f} min left (lifetime then {life_now + new_left:.0f} min, "
        f"${cost_per_hr * (life_now + new_left) / 60:.2f}); armed={info.get('armed')}")
    return {"extended": True, "left_min": round(new_left, 1), "armed": info.get("armed")}


def _watcher(ses, pod, created_ts: float, cost_per_hr: float, stop: threading.Event) -> None:
    """While the session is healthy and the kill switch is < 14 min away, move it 10 min later (ceiling 150 min / $5).
    Healthy = no command is running (upload, bootstrap, pack, download) or the running command's child log changed in
    the last 4 min. Never raises."""
    glob = f"{POD_WS}/dancecap/work/runpod/sessions/{ses.sid}/pod_logs/*/*.child.log"
    while not stop.wait(60):
        try:
            ks = killswitch_status(pod)
            left = (ks["kill_at"] - ks["now"]) / 60
            if left > 14:
                continue
            running_cmd = any(v.get("status") == "running" and v.get("current") for v in ses.state["jobs"].values())
            healthy, age = True, "n/a"
            if running_cmd:
                age = pod.output(f"f=$(ls -t {glob} 2>/dev/null | head -1); [ -n \"$f\" ] && echo $(( $(date +%s) - "
                                 f"$(stat -c %Y \"$f\") )) || echo 99999", timeout=40).strip().split()[-1]
                healthy = int(age) < 240
            if healthy:
                r = extend_killswitch(pod, created_ts, cost_per_hr, 10.0, log=ses.log)
                ses.log(f"auto-extend: {r}")
            else:
                ses.log(f"auto-extend: NOT extended, the running command's log is {age} s old")
        except Exception:  # noqa: BLE001
            pass


def session(seg: str, sid: str | None = None, order: int = 0, tiers: list[dict] | None = None, min_cpus: float = 8.0,
            min_mem_gb: float = 32.0, min_upload: float = 0.6, host_attempts: int = 3, dry_run: bool = False,
            wait_for_slot: bool = True) -> dict:
    """One segment: bundle -> (upload slot) -> pod (H100 tier, then A100 tier) -> host check -> upload -> jobs fit +
    refine -> download -> pod deleted. See the module doc."""
    from dataclasses import asdict

    from . import remote
    from . import remote_jobs as rj
    patch_ledger_lock()
    tiers = tiers or TIERS
    take = seg_name(seg)
    _smoke_import()
    jobs = rj.resolve_jobs(take, list(JOBS))
    label = f"fit-{seg}"
    ses = rj.Session(sid, label)
    ses.state.pop("error", None)
    ses.state.pop("bootstrap_rc", None)
    ses.write_runner_info(label, [j.name for j in jobs])
    todo = []
    for j in jobs:
        fpr = j.fingerprint()
        prev = ses.state["jobs"].get(j.name, {})
        if prev.get("status") == "ok" and prev.get("fingerprint") == fpr:
            ses.log(f"{j.name}: already done in this session; skipped")
            continue
        ses.state["jobs"][j.name] = {"status": "pending", "fingerprint": fpr, "spec": asdict(j)}
        todo.append(j)
    ses.save()
    if not todo:
        return rj._finish(ses, None)
    files = ses.timed("bundle_scan", lambda: rj.bundle_files(todo))
    bundle, man = ses.timed("bundle_build", lambda: rj.build_bundle(files))
    ses.state["bundle"] = {k: man[k] for k in ("key", "path", "bytes", "raw_bytes", "n_files", "sha256",
                                                "groups_raw_bytes", "links", "build_s")}
    feats = rj.needed_features(todo)
    ses.state["estimate_min"] = rj.estimate_minutes(todo, man["bytes"])
    ses.log(f"bundle {bundle.name}: {man['bytes'] / 1e6:.1f} MB ({man['n_files']} files); features {feats}; "
            f"upload ~{man['bytes'] / 1e6 / 2.0 / 60:.1f} min at 2 MB/s")
    ses.save()
    if dry_run:
        ses.log("dry run: no pod")
        return rj._finish(ses, None)

    pod_id, spent0, ok_run = None, remote.spent(), False
    slot = UploadSlot(seg, order, log=ses.log) if wait_for_slot else None
    try:
        if slot:
            slot.register()
            slot.wait_turn(1)   # the pod boots while the segment ahead uploads
        for tier in tiers:
            gate = budget_gate(seg, tier["max_cost"])
            ses.log(f"budget gate {tier['name']}: {gate}")
            ses.state.setdefault("budget_gates", []).append(gate)
            if not gate["ok"]:
                continue
            commit(seg, tier["max_cost"])
            try:
                pod_id = _run_on_pod(ses, jobs=todo, bundle=bundle, man=man, feats=feats, tier=tier,
                                     min_cpus=min_cpus, min_mem_gb=min_mem_gb, min_upload=min_upload,
                                     host_attempts=host_attempts, label=label, slot=slot)
                if pod_id:
                    ok_run = True
                    break
            except RuntimeError as e:
                ses.log(f"tier {tier['name']}: {e}")
                if "could be started" not in str(e):
                    raise
            except remote.BudgetError as e:
                ses.log(f"tier {tier['name']}: budget: {e}")
        if not ok_run and not ses.state.get("error"):
            ses.state["error"] = "no pod could be started in any GPU tier / the budget gate failed"
    except Exception as e:  # noqa: BLE001
        ses.state["error"] = f"{type(e).__name__}: {e}"[:2000]
        ses.log(f"session error: {ses.state['error']}")
    finally:
        if slot:
            slot.release()
        release(seg)
        ses.state["session_cost"] = round(remote.spent() - spent0, 4)
        ses.save()
    res = rj._finish(ses, ses.state.get("pod", {}).get("id") or pod_id)
    res["session_cost"] = ses.state["session_cost"]
    (ses.dir / "result.json").write_text(json.dumps(res, indent=1, default=str))
    return res


def _smoke_import() -> None:
    r = subprocess.run([sys.executable, "-c", "import dancecap.cli, dancecap.mamma_dyn, dancecap.pose_refine, "
                        "dancecap.fit_full, dancecap.pod_detach, dancecap.remote_jobs"], capture_output=True, text=True,
                       cwd=str(PACKAGE_ROOT), timeout=180)
    if r.returncode != 0:
        raise SystemExit(f"[fit_full] the dancecap package does not import (someone is editing it?):\n{r.stderr[-800:]}")


def _run_on_pod(ses, jobs, bundle: Path, man: dict, feats: list[str], tier: dict, min_cpus: float, min_mem_gb: float,
                min_upload: float, host_attempts: int, label: str, slot: "UploadSlot | None" = None) -> str | None:
    from . import remote
    from . import remote_jobs as rj
    pod_id = None
    for k in range(host_attempts):
        with remote.pod_session(label, max_minutes=tier["max_minutes"], gpus=tier["gpus"], max_cost=tier["max_cost"],
                                disk_gb=60, start_timeout_min=START_TIMEOUT_MIN,
                                max_start_attempts=int(tier.get("start_attempts", 2))) as pod:
            pod_id = pod.id
            facts = host_facts(pod, label)
            ses.state.setdefault("host_checks", []).append({"attempt": k + 1, "pod": pod.id, "tier": tier["name"], **facts})
            bad = [x for x, v, lim in (("cpus", facts["cpus"], min_cpus), ("mem_gb", facts["mem_gb"], min_mem_gb),
                                       ("upload_mb_s", facts["upload_mb_s"], min_upload)) if v < lim]
            if str(facts.get("host")) in BAD_HOSTS:   # CUDA init failed there on 2026-10-07: never keep it
                ses.log(f"host {facts.get('host')} is blocklisted; pod {pod.id} is deleted")
                ses.save()
                if k >= host_attempts - 1:
                    raise RuntimeError(f"no usable pod could be started: blocklisted host {facts.get('host')}")
                continue
            ses.log(f"host check {k + 1}/{host_attempts}: {facts}" + (f" -> REJECTED ({bad})" if bad else " -> ok"))
            ses.save()
            if bad and k < host_attempts - 1:
                continue  # leaving the with-block deletes this pod
            pod_detach.make_resilient(pod)
            row = next((r for r in remote.ledger() if r["pod_id"] == pod.id), {})
            created_ts = float(row.get("created_ts") or time.time())
            pod_info = {"id": pod.id, "name": pod.name, "gpu": pod.gpu, "cost_per_hr": pod.cost_per_hr,
                        "host": pod.host, "port": pod.port, "killswitch": pod.killswitch, "started": time.time(),
                        "created_ts": created_ts, "tier": tier["name"], "host_facts": facts,
                        "kill_at_est": round(time.time() + tier["max_minutes"] * 60, 1)}
            ses.state["pod"] = pod_info
            ses.state["max_minutes"] = tier["max_minutes"]
            ses.save()
            ses.event("note", "pod", msg=f"pod {pod.id} {pod.gpu} ${pod.cost_per_hr:.2f}/h ready")
            try:
                _drive(ses, pod, jobs, bundle, man, feats, created_ts, slot)
            except BaseException as e:  # noqa: BLE001
                # a detached command may still be running (connection down for too long, a bug here): keep the pod
                # (its kill switch is the bound) so `resume` can re-attach; delete it when nothing was running
                if any(v.get("status") == "running" for v in ses.state["jobs"].values()):
                    _KEEP.add(pod.id)
                    ses.state["pod_kept"] = pod.id
                    ses.log(f"pod {pod.id} KEPT after {type(e).__name__}: {e}; resume with: python -m dancecap.fit_full "
                            f"resume --seg {label.split('-')[-1]}")
                    ses.save()
                raise
            return pod_id
    return pod_id


def _drive(ses, pod, jobs, bundle, man, feats, created_ts: float, slot: "UploadSlot | None" = None) -> None:
    """Upload (inside the caller's slot, released right after), then the standard remote_jobs steps with the
    auto-extending watcher and a download-retry before the pod is allowed to go."""
    from . import remote_jobs as rj
    try:
        if slot:
            ses.timed("upload_slot", lambda: slot.acquire())
        ses.timed("upload", lambda: rj._retry(lambda: _upload_bundle(ses, pod, bundle, man), what="upload"),
                  mb=round(man["bytes"] / 1e6, 1))
    finally:
        if slot:
            slot.release()
    stop = threading.Event()
    th = threading.Thread(target=_watcher, args=(ses, pod, created_ts, pod.cost_per_hr, stop), daemon=True)
    th.start()
    puller = rj._start_puller(ses, pod, jobs, bundle, man)
    try:
        rj._pod_steps(ses, pod, jobs, bundle, man, feats)
        _ensure_downloads(ses, pod, jobs)
    finally:
        stop.set()
        rj._stop_puller(puller)


def _ensure_downloads(ses, pod, jobs) -> None:
    """A job whose pack / download failed (connection) is retried while the pod still holds the archive; the pod is
    only deleted after this returns."""
    from . import remote_jobs as rj
    for job in jobs:
        for attempt in range(6):
            st = ses.state["jobs"][job.name]
            if not st.get("download_error") and not str(st.get("status", "")).startswith("failed: pack/download"):
                break
            ses.log(f"{job.name}: download not complete ({st.get('download_error') or st.get('status')}); retry "
                    f"{attempt + 1}/6 in 60 s (the pod is kept)")
            time.sleep(60)
            st.pop("download_error", None)
            rj._run_job(ses, pod, job, st)


# ------------------------------------------------------------------------------------------------ local: resume / extend / launch


def _pod_from_state(ses):
    from . import remote
    info = ses.state.get("pod") or {}
    if not info.get("id"):
        raise SystemExit(f"session {ses.sid} has no pod recorded")
    pod = remote.Pod(info["id"], info.get("name", "fit"), info.get("gpu", "?"), float(info.get("cost_per_hr") or 0),
                     info["host"], int(info["port"]), killswitch=info.get("killswitch"))
    return pod_detach.make_resilient(pod), info


def _latest_session(seg: str) -> str | None:
    from . import remote_jobs as rj
    c = sorted(p.name for p in rj.SESSIONS.glob(f"*-fit-{seg}") if (p / "state.json").exists())
    return c[-1] if c else None


def resume(seg: str, sid: str | None = None) -> dict:
    """Re-attach to the segment's session after the local runner died: the pod (if it is alive) keeps the detached
    fit running; upload/extract/bootstrap are idempotent, finished commands are skipped (.done) and the running one is
    attached to. The pod is deleted only after both jobs are downloaded."""
    from . import remote
    from . import remote_jobs as rj
    patch_ledger_lock()
    sid = sid or _latest_session(seg)
    if not sid:
        raise SystemExit(f"no session for segment {seg}")
    ses = rj.Session(sid, f"fit-{seg}")
    pod, info = _pod_from_state(ses)
    ses.log(f"resume: re-attaching to pod {pod.id} ({pod.gpu}) at {pod.host}:{pod.port}")
    try:
        if pod.run("true", check=False, timeout=60, echo=False) != 0:
            raise RuntimeError("pod not reachable over ssh")
    except Exception as e:  # noqa: BLE001
        ses.log(f"resume: pod {pod.id} is not reachable ({e}); start a fresh session instead (fit_full session --seg {seg})")
        return {"sid": sid, "resumed": False, "error": str(e)}
    jobs = rj.resolve_jobs(seg_name(seg), list(JOBS))
    todo = [j for j in jobs if ses.state["jobs"].get(j.name, {}).get("status") != "ok"]
    bundle = Path(ses.state["bundle"]["path"])
    man = json.loads(bundle.with_suffix("").with_suffix(".json").read_text())
    feats = rj.needed_features(todo or jobs)
    commit(seg, float(info.get("cost_per_hr") or 0) * 2.5)
    spent0 = remote.spent()
    try:
        if todo:
            for j in todo:
                ses.state["jobs"][j.name].update(status="pending")
            ses.save()
            _drive(ses, pod, todo, bundle, man, feats, float(info.get("created_ts") or time.time()))
    except Exception as e:  # noqa: BLE001
        ses.state["error"] = f"{type(e).__name__}: {e}"[:2000]
        ses.log(f"resume error: {ses.state['error']}")
    finally:
        all_ok = all(ses.state["jobs"].get(j.name, {}).get("status") == "ok" for j in jobs)
        still_running = any(ses.state["jobs"].get(j.name, {}).get("status") == "running" for j in jobs)
        if all_ok or (ses.state.get("error") and not still_running):
            remote._delete(pod.id, reason="session end (resumed)")
        elif still_running:  # a detached command may still be going (ssh down again): the kill switch bounds the cost
            ses.log(f"pod {pod.id} KEPT (a command was still running when the resume ended); resume again with: "
                    f"python -m dancecap.fit_full resume --seg {seg}")
        release(seg)
        ses.state["session_cost"] = round(ses.state.get("session_cost", 0) + remote.spent() - spent0, 4)
        ses.save()
    res = rj._finish(ses, pod.id)
    (ses.dir / "result.json").write_text(json.dumps(res, indent=1, default=str))
    return res


def extend(seg: str, add_min: float, sid: str | None = None) -> dict:
    from . import remote_jobs as rj
    sid = sid or _latest_session(seg)
    ses = rj.Session(sid, f"fit-{seg}")
    pod, info = _pod_from_state(ses)
    return extend_killswitch(pod, float(info.get("created_ts") or time.time()), pod.cost_per_hr, add_min)


def launch(sids: list[str], gap_s: float = 20.0) -> list[dict]:
    """Detached runners (Windows: DETACHED_PROCESS, no console, survive the terminal): one `session` per segment, in
    the given order; the upload slot serialises their uploads."""
    FIT_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for i, sid in enumerate(sids):
        log = FIT_DIR / f"runner_{sid}.log"
        cmd = [sys.executable, "-m", "dancecap.fit_full", "session", "--seg", sid, "--order", str(i)]
        kw: dict = {}
        if os.name == "nt":
            kw["creationflags"] = 0x00000008 | 0x00000200  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
        else:
            kw["start_new_session"] = True
        fh = open(log, "ab")
        p = subprocess.Popen(cmd, cwd=str(PACKAGE_ROOT), stdin=subprocess.DEVNULL, stdout=fh, stderr=subprocess.STDOUT,
                             env=dict(os.environ, PYTHONUNBUFFERED="1", PYTHONUTF8="1"), **kw)
        info = {"seg": sid, "pid": p.pid, "log": str(log), "started": time.strftime("%FT%T"), "order": i}
        (FIT_DIR / f"runner_{sid}.json").write_text(json.dumps(info, indent=1))
        print(f"[fit_full] launched {sid}: pid {p.pid}, log {log}")
        out.append(info)
        time.sleep(gap_s)
    return out


def status() -> dict:
    from . import remote
    from . import remote_jobs as rj
    out = {"ledger": remote.status(), "commitments": commitments(), "segments": {}}
    for sid in SEGS:
        d: dict = {}
        r = FIT_DIR / f"runner_{sid}.json"
        if r.exists():
            ri = json.loads(r.read_text())
            d["runner_pid"], d["runner_alive"] = ri["pid"], _pid_alive(ri["pid"])
        s = _latest_session(sid)
        if s:
            st = json.loads((rj.SESSIONS / s / "state.json").read_text())
            d.update(session=s, jobs={k: {"status": v.get("status"), "current": v.get("current")}
                                      for k, v in st.get("jobs", {}).items()},
                     pod=(st.get("pod") or {}).get("id"), last_step=(st.get("timings") or [{}])[-1].get("step"),
                     error=st.get("error"))
        out["segments"][sid] = d
    return out


# ------------------------------------------------------------------------------------------------ plan / FITS_READY.md


def plan(sids: list[str]) -> dict:
    """Bundles (offline), upload sizes and times, budget, GPU table -> FIT_DIR/plan.json."""
    from . import remote
    from . import remote_jobs as rj
    res: dict = {"at": time.strftime("%FT%T"), "segments": {}}
    for sid in sids:
        jobs = rj.resolve_jobs(seg_name(sid), list(JOBS))
        files = rj.bundle_files(jobs)
        bundle, man = rj.build_bundle(files)
        mb = man["bytes"] / 1e6
        res["segments"][sid] = {
            "bundle": bundle.name, "mb": round(mb, 1), "raw_mb": round(man["raw_bytes"] / 1e6, 1),
            "n_files": man["n_files"], "upload_min": {"2.8 MB/s": round(mb / 2.8 / 60, 1), "2.0 MB/s": round(mb / 2.0 / 60, 1),
                                                       "0.9 MB/s": round(mb / 0.9 / 60, 1)},
            "groups_mb": {k: round(v / 1e6, 1) for k, v in sorted(man["groups_raw_bytes"].items(), key=lambda kv: -kv[1])[:8]}}
    st = remote.status()
    res["ledger"] = st
    try:
        res["gpus"] = remote.gpu_table()
    except Exception as e:  # noqa: BLE001
        res["gpus"] = str(e)
    FIT_DIR.mkdir(parents=True, exist_ok=True)
    (FIT_DIR / "plan.json").write_text(json.dumps(res, indent=1, default=str))
    return res


# ------------------------------------------------------------------------------------------------ CLI


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m dancecap.fit_full", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["check", "write-jobs", "plan", "preflight", "verify", "launch", "session",
                                       "resume", "extend", "status"])
    ap.add_argument("arg", nargs="?", help="check/write-jobs/plan/launch: segment list (a,b,c,d); preflight/verify: take")
    ap.add_argument("--seg")
    ap.add_argument("--sid")
    ap.add_argument("--order", type=int, default=0)
    ap.add_argument("--local", action="store_true")
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--smoke-div", type=int, default=10)
    ap.add_argument("--add-min", type=float, default=10.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--gap", type=float, default=20.0)
    ap.add_argument("--no-slot", action="store_true")
    ap.add_argument("--tiers", help="comma list of tier names (h100,a100)")
    ap.add_argument("--full-bundle", action="store_true",
                    help="write-jobs: keep the front-end wheels and the male/female SMPL-X models in the bundle")
    a = ap.parse_args(argv)
    if a.action == "check":
        bad = 0
        for sid in _segs(a.arg):
            r = check_segment(sid)
            print(f"[fit_full] {sid}: {'OK' if r['ok'] else 'FAILED'}  hard: {[k for k, v in r['hard'].items() if not v] or 'all pass'}"
                  f"  soft: {r.get('soft_warnings') or 'none'}  cams {r['info'].get('cams')}  "
                  f"n_cams {r['info'].get('n_cams_hist')}")
            bad += 0 if r["ok"] else 1
        raise SystemExit(1 if bad else 0)
    elif a.action == "write-jobs":
        write_jobs(_segs(a.arg), slim=not a.full_bundle)
    elif a.action == "plan":
        print(json.dumps(plan(_segs(a.arg)), indent=1, default=str))
    elif a.action == "preflight":
        preflight(a.arg, frames=a.frames, local=a.local, smoke_div=a.smoke_div)
    elif a.action == "verify":
        verify(a.arg)
    elif a.action == "launch":
        launch(_segs(a.arg), a.gap)
    elif a.action == "session":
        tiers = [t for t in TIERS if not a.tiers or t["name"] in a.tiers.split(",")]
        res = session(a.seg or a.arg, a.sid, a.order, tiers, dry_run=a.dry_run, wait_for_slot=not a.no_slot)
        print(json.dumps(res, indent=1, default=str))
    elif a.action == "resume":
        print(json.dumps(resume(a.seg or a.arg, a.sid), indent=1, default=str))
    elif a.action == "extend":
        print(json.dumps(extend(a.seg or a.arg, a.add_min, a.sid), indent=1))
    elif a.action == "status":
        print(json.dumps(status(), indent=1, default=str))


if __name__ == "__main__":
    main()
