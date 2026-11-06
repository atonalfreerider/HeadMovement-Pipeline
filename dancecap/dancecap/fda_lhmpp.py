"""The appearance baseline for 4DAnyone round 2 B: our existing LHM++ splat avatars of lucas_luiza (built 2026-10-25 from
lucas_luiza_h crops, betas from the two-view fit), posed by the two-view fit, rendered on a pod (remote/cand_lhmpp/
lhmpp_render_pod.py, unchanged) at
  <run>_v0     the SAME camera as 4DAnyone B's view 0 (its cameras.json, canonical -> our world through view.json's
               world_to_canon), only the run's dancer (the partner is moved 1 km away for these keys: 4DAnyone B shows
               the dancer alone), 704 x 1280, at the input clip's instants;
  <run>_real   the REAL held-out camera (its K, R, t), both dancers (the partner occludes as in the real frame; the score
               counts the real dancer mask only), at the held-out frames' own instants.
Both are scored against the held-out camera's real crops of the dancer exactly as fda_pod.py appearance scores 4DAnyone B
(remote/cand_fda/lhmpp_score.py): letterboxed canvas, LPIPS (alex) + colour-matched PSNR inside the real mask, the same
detection frames and the same < 20 deg angle filter for v0.

    python -m dancecap.fda_lhmpp prep        # local, after round 2's results are pulled: input/{refs,poses.npz,views.*}
    python -m dancecap.fda_lhmpp job         # local -> work/lucas_luiza/jobs/pod/lucas_luiza_fdam_lhmpp.json
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from .fda_masked import CLIPS_M, MASK_TAKE, OTHER, rname
from .fda_test import TAKE, WORK, log
from .hmr_test import ROLES

FT = WORK / TAKE / "fda_test"
INP = FT / "lhmpp" / "input"
FAR = 1000.0


def cmd_prep(args) -> None:
    from . import mamma_dyn as md
    from .mono4d.jl_realism_render import ROT_KEYS, interp_many
    ll = WORK / TAKE
    offs = md.read_sync_offsets(ll)
    sim3 = json.loads((ll / "cameras" / "world_from_sfm.json").read_text())
    cams = {k: dict(np.load(ll / "cameras" / f"{k}.npz")) for k in ("01", "02")}
    Rw = {k: md.world_cameras(cams[k]["R"], cams[k]["tvec"], sim3) for k in cams}
    fits = {r: dict(np.load(ll / "smplx_rf2_noslip" / f"{r}.npz", allow_pickle=True)) for r in ROLES}
    if INP.exists():
        shutil.rmtree(INP)
    shutil.copytree(ll / "realism" / "lhmpp_generic" / "input" / "refs", INP / "refs")
    times, owner, groups = [], [], {}
    for cam in ("02", "01"):
        for clip in CLIPS_M:
            for role in ROLES:
                name = rname(cam, clip, role)
                b = FT / "runs_m" / (name + "_ours")
                if not (b / "fda.npz").exists():
                    log(f"{name}: no B run, skipped")
                    continue
                meta = json.loads((FT / "inputs_m" / name / "meta.json").read_text())
                view = json.loads((FT / "inputs_m" / (name + "_ours") / "view.json").read_text())
                fda = np.load(b / "fda.npz")
                det_frames = np.load(b / "det.npz")["frames"].astype(int)
                rows = np.array(meta["rows"])
                tau = md.apply_offset(cams[cam]["t"][rows], offs.get(cam))
                # view 0 in our world: canonical = Wc . world
                Wc = np.array(view["world_to_canon"])
                c2w = np.linalg.inv(Wc) @ fda["cam_c2w"][0]
                R0 = c2w[:3, :3].T
                t0 = -R0 @ c2w[:3, 3]
                k0 = len(times)
                times += list(tau[det_frames])
                owner += [role] * len(det_frames)
                n = len(det_frames)
                groups[f"{name}_v0"] = {"W": int(fda["cam_wh"][0, 0]), "H": int(fda["cam_wh"][0, 1]),
                                        "K": np.repeat(fda["cam_K"][0][None], n, 0), "R": np.repeat(R0[None], n, 0),
                                        "tvec": np.repeat(t0[None], n, 0), "key": k0 + np.arange(n), "index": det_frames,
                                        "t": tau[det_frames]}
                # the real held-out camera at the held-out rows matched to those frames
                oc = OTHER[cam]
                hz = np.load(FT / "heldout" / f"{name}.npz") if (FT / "heldout" / f"{name}.npz").exists() else None
                orows = np.array(meta["held_rows"])
                otau = md.apply_offset(cams[oc]["t"][orows], offs.get(oc))
                from .jl_fitcheck import frame_map
                j, _ = frame_map(tau, otau, max_dt=0.05)
                hi = np.unique(j[det_frames])   # two clip frames can share a held-out frame: one render each
                n = len(hi)
                k1 = len(times)
                times += list(otau[hi])
                owner += ["both"] * n
                W_, H_ = (720, 1280) if oc == "01" else (1280, 720)
                groups[f"{name}_real"] = {"W": W_, "H": H_, "K": cams[oc]["K"][orows[hi]], "R": Rw[oc][0][orows[hi]],
                                          "tvec": Rw[oc][1][orows[hi]], "key": k1 + np.arange(n), "index": hi,
                                          "t": otau[hi]}
                del hz
    ts = np.asarray(times, float)
    poses = {"t": ts}
    for r in ROLES:
        p = interp_many(fits[r], ts)
        for nm, _n, short in ROT_KEYS:
            poses[f"{r}_{short}"] = p[nm].astype(np.float32)
        far = np.array([o not in (r, "both") for o in owner])
        J = p["joints"][:, :22].copy()
        tr = p["transl"].copy()
        J[far] += FAR
        tr[far] += FAR
        poses[f"{r}_pelvis"] = J[:, 0].astype(np.float32)
        poses[f"{r}_joints22"] = J.astype(np.float32)
        poses[f"{r}_transl"] = tr.astype(np.float32)
        poses[f"{r}_betas"] = np.asarray(fits[r]["betas"], np.float32).reshape(-1)
    (INP).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(INP / "poses.npz", **poses)
    arrs = {}
    for g, v in groups.items():
        for k in ("K", "R", "tvec"):
            arrs[f"{g}_{k}"] = np.asarray(v[k], np.float32)
        arrs[f"{g}_key"] = np.asarray(v["key"], np.int32)
        arrs[f"{g}_index"] = np.asarray(v["index"], np.int32)
        arrs[f"{g}_t"] = np.asarray(v["t"], np.float64)
    np.savez_compressed(INP / "views.npz", **arrs)
    vj = {"take": TAKE, "passes": [[g for g in groups if g.endswith("_v0")], [g for g in groups if g.endswith("_real")]],
          "groups": {g: {"W": v["W"], "H": v["H"], "n": int(len(v["key"]))} for g, v in groups.items()}}
    (INP / "views.json").write_text(json.dumps(vj, indent=1))
    log(f"LHM++ input: {len(ts)} pose keys, {len(groups)} groups -> {INP}")


def cmd_job(args) -> None:
    base = json.loads((WORK / TAKE / "jobs" / "pod" / "lucas_luiza_fdam_c02.json").read_text())
    frames = [x for x in base["inputs"] if isinstance(x, dict) and "/fda_test/frames/" in x["pod"]]
    ref = [x for x in base["inputs"] if isinstance(x, str) and "/fda_test/" not in x and "hmr_test" not in x]
    masks = [f"dancecap/work/{MASK_TAKE[k]}/people/masks_01.npz" for k in ("02", "01")]
    b_out = [f"dancecap/work/{TAKE}/fda_test/runs_m/{rname(c, cl, r)}_ours/{f}" for c in ("02", "01") for cl in CLIPS_M
             for r in ROLES for f in ("appearance.json", "det.npz")
             if (FT / "runs_m" / f"{rname(c, cl, r)}_ours" / f).exists()]
    inp = f"work/{TAKE}/fda_test/lhmpp/input"
    job = {"schema": "dancecap.pod_job/1", "name": "lucas_luiza_fdam_lhmpp", "take": TAKE, "cwd": "dancecap",
           "max_minutes": 60, "needs": ["ffmpeg"], "base": True, "min_vram_gb": 24,
           "inputs": ref + frames + masks + b_out + [f"dancecap/{inp}/"],
           "exclude": base["exclude"],
           "env": {"LHMPP_COMMIT": "906b5d9fb967ab42efb92f6fa55bf22cac86b653",
                   "LHMPP_ROOT": "/workspace/lhmpp_cand/LHM-plusplus", "LHMPP_VENV": "/workspace/lhmpp_cand/venv",
                   "LHMPP_MODEL": "LHMPP-700M-PixelShuffle", "TORCH_HOME": "/workspace/lhmpp_cand/torch_home",
                   "HF_HOME": "/workspace/lhmpp_cand/hf_home", "PYTHONUNBUFFERED": "1"},
           "commands": [
               {"name": "install", "timeout_min": 35, "cmd": "bash remote/cand_lhmpp/install.sh"},
               {"name": "held02", "timeout_min": 10, "cmd": "python -m dancecap.fda_masked inputs --cam 02 --held-only"},
               {"name": "held01", "timeout_min": 10, "cmd": "python -m dancecap.fda_masked inputs --cam 01 --held-only"},
               {"name": "render", "timeout_min": 20,
                "cmd": f"/workspace/lhmpp_cand/venv/bin/python remote/cand_lhmpp/lhmpp_render_pod.py --input {inp} "
                       f"--out work/{TAKE}/fda_test/lhmpp/pod_out --renders work/{TAKE}/fda_test/lhmpp/renders"},
               {"name": "lpips", "timeout_min": 5, "cmd": "/workspace/lhmpp_cand/venv/bin/pip install -q lpips==0.1.4"},
               {"name": "score", "timeout_min": 10,
                "cmd": f"/workspace/lhmpp_cand/venv/bin/python remote/cand_fda/lhmpp_score.py --ft work/{TAKE}/fda_test"}],
           "outputs": [f"dancecap/work/{TAKE}/fda_test/lhmpp/pod_out/", f"dancecap/work/{TAKE}/fda_test/lhmpp/renders/",
                       f"dancecap/work/{TAKE}/fda_test/lhmpp/score.json", f"dancecap/work/{TAKE}/fda_test/lhmpp/pairs/"],
           "verify": [f"dancecap/work/{TAKE}/fda_test/lhmpp/score.json"],
           "notes": ["LHM++ baseline for 4DAnyone round 2 B (dancecap.fda_lhmpp): the take's LHM++ avatars (CC BY-NC 4.0 "
                     "weights, research use) posed by the two-view fit at 4DAnyone's view-0 cameras and at the real "
                     "held-out cameras; same score as fda_pod.py appearance."]}
    p = WORK / TAKE / "jobs" / "pod" / "lucas_luiza_fdam_lhmpp.json"
    p.write_text(json.dumps(job, indent=1))
    log(f"job {p}: {len(b_out)} B files, {len(frames)} frames")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prep")
    sub.add_parser("job")
    a = ap.parse_args(argv)
    {"prep": cmd_prep, "job": cmd_job}[a.cmd](a)


if __name__ == "__main__":
    main()
