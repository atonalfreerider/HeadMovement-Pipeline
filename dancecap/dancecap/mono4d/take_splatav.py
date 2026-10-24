"""LHM++ splat avatars of ANY take -> the viewer's splat-avatar files (docs/SPLAT_AVATAR_FORMAT.md, `SPAV` v1) in a capture.

The generic avatars step (mono4d.lhmpp_generic, pipeline/mono_route "avatars") leaves work/<take>/avatar_lhmpp/ with LHM++'s own export
per dancer (export/<role>_export.npz: canonical zero-pose means, rotations, scales, opacities, colours, LHM++'s LBS weights, rest joints;
written by remote/cand_lhmpp/recap_lhmpp_pod.py) and nothing read it ("Nothing in the Unity export reads them yet",
PIPELINE_EFFICIENCY.md 7e). This writes them the way the class recap's exact path does (mono4d.recap_splatav.write_splatav: the same
record, coordinate mirror and JSON), then checks the written file by posing it with the format's rule against LHM++'s world-frame
pose at the job's pose key (recap_splatav.file_selfcheck).

    python -m dancecap.mono4d.take_splatav --take lucas_luiza_h --capture 09_LucasLuiza [--dest DIR]

Hair: no strand binding (strand = -1, hair = 0 for every Gaussian): the recap's hair selection is a colour rule for one dancer's red
hair, not a general one. The splats still carry the hair LHM++ reconstructed, skinned to the head and neck.
Likeness data: the output goes to the git-ignored StreamingAssets capture folder (and work/), never into a repository.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from ..take import PACKAGE_ROOT, WORKSPACE, Take
from . import recap_splatav as RS

ROLES = ("lead", "follow")


def convert(take: str, capture: str, dest: Path | None = None, key: int | None = None) -> dict:
    av = PACKAGE_ROOT / "work" / take / "avatar_lhmpp"
    summ = json.loads((av / "avatar_lhmpp.json").read_text(encoding="utf-8"))
    gen = PACKAGE_ROOT / "work" / take / "realism" / "lhmpp_generic"
    pod_out = gen / "pod_out"
    pod = json.loads((pod_out / "lhmpp_render_pod.json").read_text(encoding="utf-8"))
    rec_pod = pod_out / "recap_lhmpp_pod.json"
    dest = Path(dest) if dest else WORKSPACE / "head-movement" / "Assets" / "StreamingAssets" / capture / "splats"
    poses = gen / "input" / "poses.npz"
    Z = None
    if poses.exists():
        with np.load(poses) as z:
            Z = {k: z[k] for k in z.files}
    dancers = Take.load(take).cfg.get("dancers", {})
    names = {r: str(dancers.get(r, {}).get("name") or r) for r in ROLES}
    rep = {"take": take, "capture": capture, "dest": str(dest), "dancers": {}}
    for role in ROLES:
        ex_p = av / "export" / f"{role}_export.npz"
        if not ex_p.exists():
            raise SystemExit(f"[take_splatav] {ex_p} is missing: the avatars step ran without its LBS export")
        with np.load(ex_p) as z:
            ex = {k: z[k] for k in z.files}
        hair = np.zeros(len(ex["mean0"]), bool)
        picks = (summ.get("dancers") or {}).get(role, {}).get("picks") or []
        js = RS.write_splatav(role, ex, hair, {"hair": 0}, pod, {}, {}, dest, name=names.get(role) or role, capture=capture, take=take,
                              inputs=picks, pod_report=rec_pod if rec_pod.exists() else pod_out / "lhmpp_render_pod.json",
                              hair_note="none bound (dancecap.mono4d.take_splatav): LHM++'s hair Gaussians are skinned to the head and neck",
                              likeness="photographic likeness of a real person (owner consent recorded in the take TOML); never commit, never ship")
        chk = None
        if Z is not None:
            k = key if key is not None else 0
            try:
                chk = RS.file_selfcheck(role, dest, Z, k, ex)
            except (KeyError, IndexError, ImportError) as e:
                chk = {"error": f"{type(e).__name__}: {e}"}
            js["checks"]["file_selfcheck"] = chk
            (dest / f"{role}.json").write_text(json.dumps(js, indent=1))
        mirror = av / "splatav"
        mirror.mkdir(parents=True, exist_ok=True)
        for f in (f"{role}.splatav", f"{role}.json"):
            shutil.copy2(dest / f, mirror / f)
        rep["dancers"][role] = {"count": js["count"], "bytes": js["data_bytes"], "truncation": js["skinning"]["truncation"],
                                "selfcheck": chk, "rest_joints_vs_avatar_skin_mm": (js["checks"]["rest_joints_vs_avatar_skin_mm"] or {}).get("max_mm")}
        print(f"[take_splatav] {role}: {js['count']} Gaussians -> {dest / (role + '.splatav')}; self-check {chk}", flush=True)
    (av / "splatav" / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.mono4d.take_splatav", description=__doc__.split("\n\n")[0])
    ap.add_argument("--take", required=True)
    ap.add_argument("--capture", required=True, help="the Unity capture folder name (StreamingAssets/<capture>/splats/)")
    ap.add_argument("--dest", type=Path)
    ap.add_argument("--key", type=int, help="pose key of the job's poses.npz for the self-check (default 0)")
    a = ap.parse_args(argv)
    convert(a.take, a.capture, a.dest, a.key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
