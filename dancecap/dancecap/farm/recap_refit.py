"""Refit of ONE recap chunk with a camera left out of a stretch (agent F1, 2026-10-09).

    python -m dancecap.farm --dance larissa_kadu_recap refit g4644 --mask 06:152:208 [--tag r1] [--dry-run]

WHY. The first fits of the class showed single-camera failures that drag the whole chunk: in g4644 phone 06 reprojects the follow at 40-210 px (the other
five phones 5-35 px, the lead 5 px: the follow in 06 is a wrong person or a wrong geometry) and the follow's fit is bad in all cameras; in g6171 phone 02 sits
100-180 px off during 243-257 s (its camera track moves 1 m along the optical axis there: R2's pose is ambiguous with a zoom). The fix is the one `mamma
prepare` already applies to a frame on which a camera is not registered: its landmarks are zeroed for that stretch.

HOW. The chunk's FRONT is not repeated: the first job's checkpoint (chunks/<id>/ckpt/front.tar on the volume: ma_2d, ma_cap, prepare.json, ma3d_config, sync
offsets) is copied to src/<capture>/refit/<id>.<tag>.front.tar and becomes an archive input of a NEW job with the same id (the recap_front commands derive the
take from the id) whose steps are chunk_take -> mask (`recap_front mask`) -> await_pin -> fit --pin -> refine -> quality -> evidence. The first job's results are
kept: the volume dir is moved aside by the uploader (chunks/<id>.old.<ts>), the local take dir is copied to work/larissa_kadu_recap/chunks/<tag0>/<id>/, the
old spec stays as farm/jobs/<id>.<tag0>.json. The watcher then launches a pod for it as for any queued job and pulls it as usual.
"""
from __future__ import annotations

import copy
import json
import shutil
import time
from pathlib import Path

from . import farm_dir, jobs_dir
from .state import State, now_iso

DANCE = "larissa_kadu_recap"
CAPTURE = "Larissa-Kadu-recap"
REFIT_STEPS = ("chunk_take", "mask", "await_pin", "fit", "refine", "quality", "evidence")
DROP_INPUT_PREFIX = f"src/{CAPTURE}/"        # the source videos: a refit has no front, the volume holds them anyway


def names(dance: str) -> tuple[str, str]:
    """(take, capture folder) of a dance: a two-view profile's (farm/ca_profile.PROFILES: lucas_luiza, carlos_aline), else the recap's."""
    from . import ca_profile as CAPf
    P = CAPf.PROFILES.get(dance)
    return (P.dance, P.capture) if P is not None else (DANCE, CAPTURE)


def mask_arg(masks: list[tuple[str, float, float]]) -> str:
    return ",".join(f"{c}:{a:g}:{b:g}" for c, a, b in masks)


def refit_spec(base: dict, masks: list[tuple[str, float, float]], ckpt_rel: str, tag: str, capture: str = CAPTURE) -> dict:
    """The refit job spec made from the first job's spec: no `front` step, a `mask` step after chunk_take (then the base's qa_mask
    again when it had one), the front checkpoint as an archive input; every step the base runs after `fit` (refine variants,
    their quality passes, evidence) in its own order."""
    spec = copy.deepcopy(base)
    cid = spec["id"]
    steps = {s["name"]: copy.deepcopy(s) for s in base["steps"]}
    missing = [n for n in REFIT_STEPS if n != "mask" and n not in steps]
    if missing:
        raise ValueError(f"{cid}: the base job has no step(s) {missing}")
    mask = {"name": "mask", "cmd": f"python -m dancecap.recap_front mask {cid} --mask {mask_arg(masks)}", "timeout_min": 10, "optional": False, "stage": "front"}
    steps["mask"] = mask
    order = [s["name"] for s in base["steps"]]
    after_fit = order[order.index("fit") + 1:]
    spec["steps"] = [steps[n] for n in ("chunk_take", "mask", *(("qa_mask",) if "qa_mask" in steps else ()), "await_pin", "fit", *after_fit)]
    drop = f"src/{capture}/"
    keep = [i for i in base.get("inputs", []) if not str(i.get("path", "")).startswith(drop) or str(i["path"]).endswith("_in.tar")]
    spec["inputs"] = keep + [{"path": ckpt_rel, "archive": True}]
    spec["refit"] = {"tag": tag, "masks": [{"cam": c, "t0": a, "t1": b} for c, a, b in masks], "front_checkpoint": ckpt_rel, "created": now_iso(),
                     "of_code": (base.get("code") or {}).get("key")}
    est = spec.get("estimate") or {}
    spec["max_minutes"] = round(1.6 * (float(est.get("fit_min") or 30.0) + float(est.get("refine_min") or 3.0) + float(est.get("quality_min") or 3.0)) + 25.0, 1)
    spec["stages"] = ["fit", "refined"]
    spec["created"] = now_iso()
    return spec


def parse_masks(s: str) -> list[tuple[str, float, float]]:
    from ..recap_front import parse_masks as pm
    return pm(s)


def cmd_refit(a) -> int:
    from . import cli, launch
    dance = a.dance
    cid = a.chunk
    masks = parse_masks(a.mask)
    if not masks:
        print("refit: --mask cam:t0:t1[,cam:t0:t1] is required")
        return 2
    tag = a.tag
    take, capture = names(dance)
    st = State(dance)
    cur = st.load()
    status = (cur["chunks"].get(cid) or {}).get("status")
    spec_path = jobs_dir(dance) / f"{cid}.json"
    if status not in ("pulled", "done", "failed", "stitched") and not a.force:
        print(f"refit: chunk {cid} is {status!r}; refit only a finished chunk (or --force)")
        return 2
    base = json.loads(spec_path.read_text(encoding="utf-8"))
    if base.get("refit"):
        print(f"refit: {cid}'s spec is already a refit ({base['refit']['tag']}); refit from the original spec {cid}.<tag>.json")
        return 2
    ckpt_rel = f"src/{capture}/refit/{cid}.{tag}.front.tar"
    new = refit_spec(base, masks, ckpt_rel, tag, capture)
    if a.dry_run:
        print(json.dumps({"id": cid, "steps": [s["name"] for s in new["steps"]], "mask": new["steps"][1]["cmd"], "inputs": [i["path"] for i in new["inputs"]],
                          "max_minutes": new["max_minutes"], "status_now": status}, indent=1))
        return 0
    with launch.volume_link(dance, keep=True) as link:
        out = link.sh(f"cd /workspace/hm && test -f chunks/{cid}/ckpt/front.tar && mkdir -p src/{capture}/refit && "
                      f"{{ if [ ! -f {ckpt_rel} ]; then cp chunks/{cid}/ckpt/front.tar {ckpt_rel}.part && mv -f {ckpt_rel}.part {ckpt_rel}; fi; }} ; ls -l {ckpt_rel}", 300)
        if cid not in out:
            print(f"refit: the front checkpoint of {cid} is not on the volume: {out[-200:]}")
            return 3
    # keep the first run: local take dir, farm pull evidence, old spec
    root = Path(__file__).resolve().parents[2] / "work"
    keep = root / take / "chunks" / "v1" / cid
    take_dir = root / f"{take}_{cid}"
    if take_dir.exists() and not keep.exists():
        keep.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(take_dir, keep / "work")
        pulled = farm_dir(dance) / "chunks" / cid
        if pulled.exists():
            shutil.copytree(pulled, keep / "farm_chunk")
            shutil.rmtree(pulled, ignore_errors=True)
    shutil.copyfile(spec_path, spec_path.with_name(f"{cid}.v1.json"))
    spec_path.write_text(json.dumps(new, indent=1), encoding="utf-8")
    st.set_chunk(cid, status="planned", pod=None, progress=0.0, step=None, eta_min=None, pull=None, done=None, last_error=None, requeues=0, refit=tag,
                 note=f"refit {tag}: {mask_arg(masks)}")
    st.set_job(cid, status="planned", refit=tag)
    ns = type("NS", (), {"only": cid, "exclude": None, "force_foreign": False, "keep_gateway": False, "hold": False})()
    res = cli.queue_jobs(dance, args=ns)
    print(json.dumps({k: res.get(k) for k in ("queued", "errors", "missing")}, default=str))
    return 0 if cid in (res.get("queued") or []) else 1


def register(sub) -> None:
    p = sub.add_parser("refit", help="refit one finished recap chunk with a camera left out of a stretch (no front: from its checkpoint)")
    p.add_argument("chunk")
    p.add_argument("--mask", required=True, help="cam:t0:t1[,cam:t0:t1] master seconds to leave out, e.g. 06:152:208")
    p.add_argument("--tag", default="r1")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true", help="also when the chunk is not pulled")
    p.set_defaults(func=cmd_refit)
