"""One Runpod job for a SINGLE handheld phone video: per-frame (rotation, zoom) track + 2D body keypoints + monocular geometry priors.

    python -m dancecap.cameras.single_cam_pod run --video <mp4> --skeleton <dir> --out <dir> --geom-frames 15,90,... [--owner J2]

A DIRECT pod job (never the shared farm queue: autostart False, no worker runs on the pod): a GPU pod mounting the hm-farm
volume (its environment tar gives torch / ultralytics / lightglue), the four small modules of dancecap/cameras are copied
over, then on the one GPU in parallel (nohup, detached from ssh):
  track   `ptz_run track` on every frame, `shards` processes on contiguous blocks (matches stored for the refine step),
  pose    `people2d` on every frame,
  geom    `geom_priors` (MoGe-2, Depth-Anything-V2 metric) on the requested frames.
Pulls every npz, then deletes the pod in `finally`. Cost guard: pods.gate(worst case) first, an on-pod kill switch, delete in
`finally`. Written for agent J2 (john_lula, 538x960 phone video, 2643 frames); nothing here is specific to that video.
"""
from __future__ import annotations

import argparse
import json
import shlex
import time
from pathlib import Path

from .. import remote
from ..farm import pods

HERE = Path(__file__).resolve().parent
MODULES = ["ptz.py", "ptz_feats.py", "ptz_run.py", "people2d.py", "geom_priors.py"]
GPU = "NVIDIA GeForce RTX 4090"
FALLBACK_GPUS = ["NVIDIA L4"]
MOGE_COMMIT = "07444410f1e33f402353b99d6ccd26bd31e469e8"       # MoGe-2 code, before the V3 rewrite
UTILS3D_COMMIT = "3fab839f0be9931dac7c8488eb0e1600c236e183"    # the utils3d that commit pins
POD_PY = "/opt/hm/env/bin/python"
WDIR = "/opt/hm/j2"
HM = "/workspace/hm"


def log(msg: str) -> None:
    print(f"[single_cam_pod {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def env_prefix() -> str:
    return (f"cd {WDIR} && export PATH=/opt/hm/env/bin:$PATH PYTHONPATH={WDIR} PYTHONUNBUFFERED=1 "
            f"TORCH_HOME={HM}/weights/torch_home HF_HOME={WDIR}/hf YOLO_CONFIG_DIR={WDIR}/yolo_cfg PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")


def detach(pod, name: str, cmd: str) -> None:
    inner = f"{env_prefix()} && {cmd} > {name}.log 2>&1; echo $? > {name}.rc"
    pod.run(f"nohup setsid bash -c {shlex.quote(inner)} > /dev/null 2>&1 < /dev/null &")


def start_jobs(pod, shards: int, geom_frames: str, models: str, weights: str, rects: str = "", only: tuple[str, ...] = ("track", "pose", "geom"),
               vggt_groups: str = "") -> list[str]:
    names = []
    shards = shards if "track" in only else 0
    if shards:
        pod.run(f"cd {WDIR} && {POD_PY} -c \"import json;a=json.load(open('frames.json'));n={shards};"
                f"[json.dump(a[i*len(a)//n:(i+1)*len(a)//n],open(f'frames_{{i}}.json','w')) for i in range(n)]\"")
    for k in range(shards):
        detach(pod, f"track{k}", f"{POD_PY} -u -m j2cam.ptz_run track in/video.mp4 --skeleton skel --out out/track_s{k}.npz "
                                 f"--frames {WDIR}/frames_{k}.json {rects} --yolo {weights}/yolo11x-seg.pt")
        names.append(f"track{k}")
    if "pose" in only:
        detach(pod, "pose", f"{POD_PY} -u -m j2cam.people2d in/video.mp4 --out out/pose.npz --weights {weights}/yolo11x-pose.pt")
        names.append("pose")
    if geom_frames and "geom" in only:
        vg = f" --vggt-groups {shlex.quote(vggt_groups)}" if vggt_groups else ""
        # MoGe-2 (the MoGe commit before V3) and its pinned utils3d are installed WITHOUT dependencies (the environment's torch / numpy stay as
        # they are: a newer torch needs a newer driver than some pods have), then the few small packages they import; own shell step first.
        pre = (f"{POD_PY} -m pip install -q --no-deps git+https://github.com/microsoft/MoGe.git@{MOGE_COMMIT} "
               f"git+https://github.com/EasternJournalist/utils3d.git@{UTILS3D_COMMIT} && {POD_PY} -m pip install -q click trimesh matplotlib scipy && "
               if "moge" in models.split(",") else "")
        detach(pod, "geom", f"{pre}{POD_PY} -u -m j2cam.geom_priors in/video.mp4 --frames {geom_frames} --out out/geom --models {models}{vg}")
        names.append("geom")
    return names


def wait_and_pull(pod, names: list[str], out: Path, minutes: float, poll: float = 45) -> None:
    deadline = time.time() + minutes * 60
    while time.time() < deadline:
        time.sleep(poll)
        try:
            st = pod.output(f"cd {WDIR}; tail -qn1 {' '.join(n + '.log' for n in names)} 2>/dev/null | cut -c1-50 | tr '\\n' '|'; echo; ls *.rc 2>/dev/null | wc -l",
                            timeout=60).strip().splitlines()
        except Exception as e:      # ssh hiccup: keep polling
            log(f"poll failed ({str(e)[:80]})")
            continue
        log(" || ".join(st))
        if st and st[-1].strip() == str(len(names)):
            break
    rcs = pod.output(f"cd {WDIR}; for f in *.rc; do echo $f $(cat $f); done").strip().replace(chr(10), "; ")
    log(f"exit codes {rcs}")
    pod.run(f"cd {WDIR}; tar -czf logs.tgz *.log && echo ok", check=False)
    pod.get(f"{WDIR}/logs.tgz", out / "logs.tgz", timeout=300)
    pod.run(f"cd {WDIR}/out && tar -cf ../out.tar . && echo ok")
    pod.get(f"{WDIR}/out.tar", out / "out.tar", timeout=1800)


def run(video: Path, skeleton: Path, out: Path, geom_frames: str, models: str = "moge,dav2", shards: int = 4, minutes: float = 40,
        max_cost: float = 1.5, gpu: str = GPU, owner: str = "J2", name: str = "hm-j2-cam", rects: str = "",
        only: tuple[str, ...] = ("track", "pose", "geom"), vggt_groups: str = "") -> dict:
    out.mkdir(parents=True, exist_ok=True)
    info, price = None, 0.9
    for rnd in range(4):                # a listed GPU can be gone by the time of the create: retry, then the next GPU of the list
        for g in [gpu, *FALLBACK_GPUS]:
            offers = pods.gpu_offers(priority=[g])
            price = offers[0]["price"] if offers else 0.9
            slow = 1.0 if g == gpu else 2.0
            worst = price * (minutes * slow + 15 + pods.START_TIMEOUT_MIN) / 60
            if worst > max_cost:
                log(f"{g}: worst case ${worst:.2f} > max_cost {max_cost}, skipped")
                continue
            pods.gate(worst)
            try:
                info = pods.create(name, gpu=g, minutes=minutes * slow, hard_minutes=minutes * slow + 15, entry=True, autostart=False,
                                   disk_gb=40, price=price, owner=owner)
                log(f"{g} ${price:.2f}/h, worst case ${worst:.2f}")
                break
            except (RuntimeError, pods.PodError) as e:
                log(f"{g}: create failed ({str(e)[:120]})")
        if info:
            break
        time.sleep(30)
    if not info:
        raise pods.PodError("no GPU pod could be created")
    pod_id = info["pod_id"]
    t0 = time.time()
    try:
        pod = pods.pod_handle(info)
        pod.run(f"bash {HM}/bin/pod_prepare.sh env libs 2>&1 | tail -3", timeout=900)
        pod.run(f"mkdir -p {WDIR}/j2cam {WDIR}/in {WDIR}/out {WDIR}/skel")
        for m in MODULES:
            pod.put(HERE / m, f"{WDIR}/j2cam/{m}")
        pod.run(f"touch {WDIR}/j2cam/__init__.py")
        for f in ("skeleton.npz", "skeleton.json"):
            pod.put(skeleton / f, f"{WDIR}/skel/{f}")
        pod.put(video, f"{WDIR}/in/video.mp4", timeout=1800)
        weights = f"{HM}/weights/dancecap"
        for w in ("yolo11x-seg.pt", "yolo11x-pose.pt"):
            if pod.run(f"test -s {weights}/{w}", check=False) != 0:
                raise RuntimeError(f"{w} is not on the volume")
        n = int(pod.output(f"{POD_PY} -c \"import cv2;print(int(cv2.VideoCapture('{WDIR}/in/video.mp4').get(7)))\"").strip())
        pod.run(f"{POD_PY} -c \"import json;json.dump(list(range({n})),open('{WDIR}/frames.json','w'))\"")
        names = start_jobs(pod, shards, geom_frames, models, weights, f"--rect {rects}" if rects else "", only, vggt_groups)
        log(f"jobs {names} started ({time.time() - t0:.0f} s since pod creation)")
        wait_and_pull(pod, names, out, info["minutes"])
        secs = time.time() - t0
        res = {"seconds": secs, "cost_usd": price * secs / 3600, "price_per_h": price, "pod_name": name}
        log(f"done in {secs:.0f} s, about ${res['cost_usd']:.2f}")
        return res
    finally:
        try:
            pods.delete(pod_id, reason="J2 camera job done")
            log("pod deleted")
        except Exception as e:
            log(f"DELETE FAILED for {pod_id}: {e}")
            raise


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--video", required=True)
    r.add_argument("--skeleton", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--geom-frames", default="")
    r.add_argument("--models", default="moge,dav2")
    r.add_argument("--shards", type=int, default=4)
    r.add_argument("--minutes", type=float, default=40)
    r.add_argument("--max-cost", type=float, default=1.5)
    r.add_argument("--owner", default="J2")
    r.add_argument("--only", default="track,pose,geom")
    r.add_argument("--vggt-groups", default="")
    a = ap.parse_args(argv)
    print(json.dumps(run(Path(a.video), Path(a.skeleton), Path(a.out), a.geom_frames, a.models, a.shards, a.minutes, a.max_cost, owner=a.owner,
                         only=tuple(a.only.split(",")), vggt_groups=a.vggt_groups)))


if __name__ == "__main__":
    main()
