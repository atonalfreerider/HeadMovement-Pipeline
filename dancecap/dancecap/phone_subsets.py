"""Tests E1 and E2 of docs/BODY_TRACKING_RESEARCH_2026-11.md on larissa_kadu_38, the eight-phone take.

E1: do three phones beat two? MAMMA fits of phone subsets on the reference's own prepared inputs (same landmarks,
masks, cameras, sync; betas pinned to the reference's), refined with the shipped recipe (smplx_rf2_noslip) and scored
against the eight-phone fit. Phone 01 is in no arm, so it is a held-out view for every arm.
E2: does the contact-patch term (pose_refine --w-contact) cut stance slip where rf2 did not? Four refine arms (0 = the
shipped recipe, a = rf2, b05 / b2 = contact term at weight 0.5 / 2) on the eight-phone fit, on E1 arm b's two-phone fit
and on the two one-phone windows (phone 06 with the SAM 3D Body prior at weight 2, 44-48 s and 58-62 s).
E2b: the two contact arms again, one change: the gate reads a 1.5 Hz low-passed foot speed (--contact-hz 1.5), not the
loaded point's 6 Hz one, which the noise of the two- and one-phone fits pushed over the gate on most planted frames (E2):
first run (job e2b) the loaded point over the whole take, plus the doc's other option, stance alone (weight 0.5, no
speed gate); second run (job e2b2) the foot lock's own path (--contact-path lock), which the doc named. Same four inputs.

    python -m dancecap.phone_subsets make     # shadow takes, glue and pod jobs (no pod, no upload)
    python -m dancecap.phone_subsets make-e2b --run e2b2  # an E2b pod job (after make: it reuses E1 arm b's fit)
    python -m dancecap remote run --jobs <the printed list> --parallel --max-cost 2.5
    python -m dancecap.phone_subsets score    # -> work/larissa_kadu_38/e1e2/{score.json, sheet.md}

Each E1 arm runs in its own shadow take, work/larissa_kadu_38_e1<x>/, whose mamma/ holds the reference's prepared
grid files (hard links) for the arm's phones only, plus an eval set with phone 01. Never inside larissa_kadu_38: a
tag-less `mamma select` there could promote a two-phone fit over the reference smplx/, and pose_refine / pose_quality
read every camera of the chosen set, which would leak the held-out phone into the arm.
"""

import argparse
import copy
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from .take import PACKAGE_ROOT

WORK = PACKAGE_ROOT / "work"
TAKES = PACKAGE_ROOT / "takes"
REF = "larissa_kadu_38"
ARMS = {"a": ("02", "09"), "b": ("06", "07"), "c": ("04", "06", "07"), "d": ("05", "06", "07")}
HELD = "01"
PIN = "dancecap/work/larissa_kadu_full/mamma_pin_betas.json"
# smplx_rf2_noslip = pose_refine's defaults, spelled out (world 612dfea3d607 shared/refine_recipe.json)
RECIPE = ("--mode refine --root-hz 5.0 --rot-hz 8.0 --w-floor 1.0 --floor-mm 5.0 --w-feet 1.0 --w-slip 0.0 "
          "--slip-max-v 0.15 --w-anchor 1.0 --anchor-mm 10.0 --iters 400")
E2 = {"e2_0": "", "e2_a": " --w-slip 0.5 --slip-max-v 0.15", "e2_b05": " --w-contact 0.5 --contact-v 0.05,0.25",
      "e2_b2": " --w-contact 2.0 --contact-v 0.05,0.25"}
E2B = {"e2b_b05": E2["e2_b05"] + " --contact-hz 1.5", "e2b_b2": E2["e2_b2"] + " --contact-hz 1.5",
       "e2b_s05": " --w-contact 0.5 --contact-v 99,100",  # the doc's other E2b option, stance alone (no speed gate)
       "e2b_l05": E2["e2_b05"] + " --contact-hz 1.5 --contact-path lock",
       "e2b_l2": E2["e2_b2"] + " --contact-hz 1.5 --contact-path lock"}
# b05 / b2 (first run): the loaded point low-passed over the whole take, which smears each swing into the planted frames
# beside it; l05 / l2 (second run): the foot lock's own path, the sole centre inside each stance run, as the doc named it.
E2B_RUNS = {"e2b": ("e2b_b05", "e2b_b2", "e2b_s05"), "e2b2": ("e2b_l05", "e2b_l2")}
MONO = {"larissa_kadu_38_mono06": (836, 956), "larissa_kadu_38_mono06_w44": (416, 536)}  # reference grid rows
MONO_BRANCH = "claude/vigilant-moser-a1afa5"
WINDOWS = {"whole": None, "44-48": (44.0, 48.0), "58-62": (58.0, 62.0)}
EXCLUDE = ["dancecap/work/runpod/pywheels", "dancecap/work/runpod/wheelhouse",
           "mamma/data/body_models/smplx_locked_head/smplx/SMPLX_MALE.npz",
           "mamma/data/body_models/smplx_locked_head/smplx/SMPLX_FEMALE.npz"]
SETUP = ["for g in MALE FEMALE; do ln -sf SMPLX_NEUTRAL.npz "
         "/workspace/HeadMovement/mamma/data/body_models/smplx_locked_head/smplx/SMPLX_$g.npz; done",
         "nvidia-smi --query-gpu=name,memory.total --format=csv"]
ALIGNMENT = ["prototype/larissa_kadu/alignment_studio.json", "prototype/larissa_kadu/alignment.json"]


def _link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _grid(ref_mamma: Path, mamma: Path, seq: str, cams: list[str]) -> None:
    """The reference's c8 grid files for these cameras, under a new sequence name (rebuilt from scratch)."""
    src_lm, src_gt = ref_mamma / "ma_2d" / f"{REF}_c8", ref_mamma / "ma_cap" / f"{REF}_c8" / "gt"
    lm, gt = mamma / "ma_2d" / seq, mamma / "ma_cap" / seq
    for d in (lm, gt):
        if d.exists():
            shutil.rmtree(d)
    for c in cams:
        _link(src_lm / f"cam{c}.npz", lm / f"cam{c}.npz")
        _link(src_gt / f"cam{c}.npz", gt / "gt" / f"cam{c}.npz")
    _link(src_gt / "global.npz", gt / "gt" / "global.npz")


def _toml(name: str, base: str = REF, text: str | None = None) -> Path:
    """takes/<name>.toml: the base take's metadata under another name (pose_quality's Take.load needs it). Not
    committed: .gitignore lists the E1 shadow takes."""
    out = TAKES / f"{name}.toml"
    if text is None:
        src = (TAKES / f"{base}.toml").read_text(encoding="utf-8")
        text = (f"# E1 shadow take of {base} (dancecap.phone_subsets): its phones' grid files only.\n"
                + src.replace(f'name = "{base}"', f'name = "{name}"', 1))
    out.write_text(text, encoding="utf-8")
    return out


def _job(path: Path, take: str, name: str, commands: list[tuple[str, int, str]], inputs: list[str],
         outputs: list[str], verify: list[str], max_minutes: int, vram: int, env: dict | None = None,
         notes: str = "") -> Path:
    job = {"schema": "dancecap.pod_job/1", "take": take, "cwd": "dancecap", "name": name, "max_minutes": max_minutes,
           "needs": ["mamma"], "min_vram_gb": vram, "inputs": inputs, "exclude": EXCLUDE,
           "env": {"DANCECAP_POD": "1", "PYOPENGL_PLATFORM": "egl", "PYTHONUNBUFFERED": "1", **(env or {})},
           "setup": SETUP, "commands": [{"name": n, "timeout_min": m, "cmd": c} for n, m, c in commands],
           "outputs": outputs, "verify": verify, "notes": [notes] if notes else []}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(job, indent=1), encoding="utf-8")
    return path


def make_e1(arms: str = "abcd", max_minutes: int = 100) -> list[Path]:
    ref_mamma = WORK / REF / "mamma"
    prep0 = json.loads((ref_mamma / "prepare.json").read_text())
    c8 = prep0["sets"]["c8"]
    jobs = []
    for x in arms:
        cams, shadow, tag = list(ARMS[x]), f"{REF}_e1{x}", f"e1{x}"
        mamma = WORK / shadow / "mamma"
        evcams = sorted({*cams, HELD})
        prep = copy.deepcopy(prep0)
        prep.update(take=shadow, created=time.strftime("%Y-%m-%dT%H:%M:%S"),
                    sets={tag: {"seq": f"{shadow}_{tag}", "cams": cams, "inputs": {c: c8["inputs"][c] for c in cams}}},
                    eval={"seq": f"{shadow}_eval", "cams": evcams},
                    e1={"reference": f"{REF} c8", "arm": x, "held_out": HELD})
        _grid(ref_mamma, mamma, f"{shadow}_{tag}", cams)
        _grid(ref_mamma, mamma, f"{shadow}_eval", evcams)
        shutil.copy2(ref_mamma / "ma3d_config.yaml", mamma / "ma3d_config.yaml")
        (mamma / "prepare.json").write_text(json.dumps(prep, indent=1))
        _toml(shadow)
        refines = {"smplx_rf2_noslip": ""} | ({k: v for k, v in E2.items() if k != "e2_0"} if x == "b" else {})
        w = f"dancecap/work/{shadow}"
        commands = [("fit", 60, f"python -m dancecap mamma fit {shadow} --tag {tag}"),
                    ("select", 5, f"python -m dancecap mamma select {shadow} --tags {tag} --no-qa")]
        commands += [(f"refine_{d}", 15, f"python -m dancecap.pose_refine {shadow} --src smplx --dst {d} {RECIPE}{extra} "
                                         f"--device cuda") for d, extra in refines.items()]
        commands += [(f"quality_{d}", 15, f"python -m dancecap.pose_quality {shadow} --smplx-dir {d} --tag {d} "
                                          f"--ref-dir ../{REF}/smplx") for d in refines]
        jobs.append(_job(
            WORK / shadow / "e1" / "pod_job.json", shadow, f"e1{x}", commands,
            inputs=[f"{w}/mamma/prepare.json", f"{w}/mamma/ma3d_config.yaml", f"{w}/mamma/ma_2d/", f"{w}/mamma/ma_cap/",
                    PIN, f"dancecap/work/{REF}/smplx/", *ALIGNMENT],
            outputs=[f"{w}/mamma/fits/", f"{w}/mamma/logs/", f"{w}/mamma/select.json", f"{w}/mamma/report.json",
                     f"{w}/smplx/", *[f"{w}/{d}/" for d in refines], f"{w}/quality/"],
            verify=[f"{w}/mamma/fits/{tag}/fit.json", f"{w}/smplx_rf2_noslip/follow.npz"],
            max_minutes=max_minutes, vram=24,
            env={"MAMMA_FREEZE_BETAS": "1", "MAMMA_INIT_BETAS": f"/workspace/HeadMovement/{PIN}"},
            notes=f"E1 arm {x}: phones {','.join(cams)} of {REF} on the reference's c8 grid files, betas pinned, "
                  f"phone {HELD} held out (eval set); refine = smplx_rf2_noslip"
                  + ("; plus the E2 refine arms on this two-phone fit" if x == "b" else "")))
    return jobs


def _slice_npz(src: Path, dst: Path, a: int, b: int) -> None:
    z = np.load(src, allow_pickle=True)
    n = len(z["t"])
    out = {k: (z[k][a:b] if z[k].ndim and len(z[k]) == n else z[k]) for k in z.files}
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, **out)


def _e2_glue() -> None:
    """Glue for the one-phone windows: select.json naming their grid set, the eight-phone fit sliced onto their grid
    as pose_quality's reference, the w44 take file from the unmerged branch."""
    for mono, (a, b) in MONO.items():
        m = WORK / mono / "mamma"
        sel = m / "select.json"
        if not sel.exists():  # the 3db fits share the m06 sequence; pose_refine / pose_quality read it from here
            sel.write_text(json.dumps({"chosen": "m06", "note": "E2 (phone_subsets): grid set of the 3db fits"}))
        for role in ("lead", "follow"):
            _slice_npz(WORK / REF / "smplx" / f"{role}.npz", WORK / mono / "ref8" / f"{role}.npz", a, b)
        if not (TAKES / f"{mono}.toml").exists():
            text = subprocess.run(["git", "show", f"{MONO_BRANCH}:dancecap/takes/{mono}.toml"], capture_output=True,
                                  text=True, check=True, cwd=PACKAGE_ROOT).stdout
            _toml(mono, text=text)


def _e2_inputs() -> list[str]:
    """The eight-phone fit and grid, and each one-phone window's fit, grid and sliced reference."""
    w = f"dancecap/work/{REF}"
    inputs = [f"{w}/smplx/", f"{w}/mamma/select.json", f"{w}/mamma/prepare.json", f"{w}/mamma/ma_cap/{REF}_c8/",
              f"{w}/mamma/ma_2d/{REF}_c8/", *ALIGNMENT]
    for mono in MONO:
        mw = f"dancecap/work/{mono}"
        inputs += [f"{mw}/mamma/select.json", f"{mw}/mamma/prepare.json", f"{mw}/mamma/ma_cap/{mono}_m06/",
                   f"{mw}/mamma/ma_2d/{mono}_m06/", f"{mw}/mamma/fits/3db/smplx/", f"{mw}/ref8/"]
    return inputs


def make_e2(max_minutes: int = 45) -> Path:
    """The one-phone glue and one pod job for every E2 refine."""
    _e2_glue()
    commands, outputs = [], []
    w = f"dancecap/work/{REF}"
    for d, extra in E2.items():
        commands.append((f"refine8_{d}", 15, f"python -m dancecap.pose_refine {REF} --src smplx --dst smplx_{d} {RECIPE}"
                                             f"{extra} --device cuda"))
        commands.append((f"quality8_{d}", 15, f"python -m dancecap.pose_quality {REF} --smplx-dir smplx_{d} --tag {d} "
                                              f"--ref-dir smplx"))
        outputs.append(f"{w}/smplx_{d}/")
    for mono in MONO:
        for d, extra in E2.items():
            commands.append((f"refine_{mono[-3:]}_{d}", 10, f"python -m dancecap.pose_refine {mono} --src "
                                                            f"mamma/fits/3db/smplx --dst {d} {RECIPE}{extra} --device cuda"))
            commands.append((f"quality_{mono[-3:]}_{d}", 10, f"python -m dancecap.pose_quality {mono} --smplx-dir {d} "
                                                             f"--tag {d} --ref-dir ref8"))
            outputs.append(f"dancecap/work/{mono}/{d}/")
        outputs.append(f"dancecap/work/{mono}/quality/")
    outputs += [f"{w}/quality/quality_{d}.json" for d in E2] + [f"{w}/quality/quality_{d}_frames.npz" for d in E2]
    return _job(WORK / REF / "e1e2" / "pod_job_e2.json", REF, "e2", commands, _e2_inputs(), outputs,
                verify=[f"{w}/smplx_e2_b05/follow.npz", f"dancecap/work/{list(MONO)[1]}/e2_b05/follow.npz"],
                max_minutes=max_minutes, vram=16,
                notes="E2 refine arms (0 / rf2 / contact 0.5 / contact 2) on the eight-phone fit and the two one-phone "
                      "windows; pose_quality against the eight-phone fit's stance frames")


def make_e2b(run: str = "e2b", max_minutes: int = 45) -> Path:
    """One pod job for one E2b run's refines (E2B_RUNS[run]) on the four E2 inputs: the eight-phone fit, E1 arm b's
    two-phone fit (refined in its shadow take, as in E1: its grid set only), and the two one-phone windows. Arm 0 and
    rf2 are E2's."""
    _e2_glue()
    w, b = f"dancecap/work/{REF}", f"dancecap/work/{REF}_e1b"
    arms = {d: E2B[d] for d in E2B_RUNS[run]}
    commands, outputs = [], []
    for d, extra in arms.items():
        commands += [(f"refine8_{d}", 15, f"python -m dancecap.pose_refine {REF} --src smplx --dst smplx_{d} {RECIPE}"
                                          f"{extra} --device cuda"),
                     (f"quality8_{d}", 15, f"python -m dancecap.pose_quality {REF} --smplx-dir smplx_{d} --tag {d} "
                                           f"--ref-dir smplx"),
                     (f"refine2_{d}", 15, f"python -m dancecap.pose_refine {REF}_e1b --src smplx --dst {d} {RECIPE}"
                                          f"{extra} --device cuda"),
                     (f"quality2_{d}", 15, f"python -m dancecap.pose_quality {REF}_e1b --smplx-dir {d} --tag {d} "
                                           f"--ref-dir ../{REF}/smplx")]
        outputs += [f"{w}/smplx_{d}/", f"{b}/{d}/"]
        for mono in MONO:
            commands += [(f"refine_{mono[-3:]}_{d}", 10, f"python -m dancecap.pose_refine {mono} --src "
                                                         f"mamma/fits/3db/smplx --dst {d} {RECIPE}{extra} --device cuda"),
                         (f"quality_{mono[-3:]}_{d}", 10, f"python -m dancecap.pose_quality {mono} --smplx-dir {d} "
                                                          f"--tag {d} --ref-dir ref8")]
            outputs.append(f"dancecap/work/{mono}/{d}/")
    for take in (REF, f"{REF}_e1b", *MONO):
        outputs += [f"dancecap/work/{take}/quality/quality_{d}{s}" for d in arms for s in (".json", "_frames.npz")]
    inputs = _e2_inputs() + [f"{b}/smplx/", f"{b}/mamma/select.json", f"{b}/mamma/prepare.json",
                             f"{b}/mamma/ma_cap/{REF}_e1b_e1b/", f"{b}/mamma/ma_2d/{REF}_e1b_e1b/"]
    first, last = list(arms)[0], list(arms)[-1]
    return _job(WORK / REF / "e1e2" / f"pod_job_{run}.json", REF, run, commands, inputs, outputs,
                verify=[f"{w}/smplx_{first}/follow.npz", f"{b}/{first}/follow.npz",
                        f"dancecap/work/{list(MONO)[1]}/{last}/follow.npz"],
                max_minutes=max_minutes, vram=16,
                notes=f"E2b run {run}: contact arms {', '.join(arms)} (the gate on a 1.5 Hz low-passed foot speed, or "
                      "stance alone) on the eight-phone, two-phone (E1 b) and one-phone fits; pose_quality against the "
                      "eight-phone fit's stance frames")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m dancecap.phone_subsets", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("make", help="shadow takes, glue and pod jobs (no pod)")
    p.add_argument("--arms", default="abcd")
    p = sub.add_parser("make-e2b", help="one E2b run's pod job (no pod)")
    p.add_argument("--run", default="e2b", choices=tuple(E2B_RUNS))
    sub.add_parser("score", help="score the returned fits -> work/larissa_kadu_38/e1e2/")
    a = ap.parse_args(argv)
    if a.cmd in ("make", "make-e2b"):
        jobs = [*make_e1(a.arms), make_e2()] if a.cmd == "make" else [make_e2b(a.run)]
        print("jobs: " + ",".join(str(j.relative_to(PACKAGE_ROOT.parent)).replace("\\", "/") for j in jobs))
    else:
        from .phone_subsets_score import score
        print(json.dumps(score(), indent=1)[:4000])


if __name__ == "__main__":
    main()
