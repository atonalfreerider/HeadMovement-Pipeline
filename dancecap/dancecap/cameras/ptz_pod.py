"""One-camera Runpod job of the pan / tilt / zoom calibration: per-frame tracks + 2D body keypoints of every frame.

    python -m dancecap.cameras.ptz_pod run <cam> --video <file> --skeleton <dir> --out <dir> [--rect x0,y0,x1,y1] [--fix-f F]

A DIRECT pod job (never the shared farm queue: no worker runs on the pod, nobody else can claim anything): create a GPU
pod that mounts the hm-farm volume (the farm environment tar gives torch / ultralytics / lightglue), unpack the env, copy
the four small modules (cameras/ptz*.py, people2d.py), the video and the skeleton, run `ptz_run track` (every frame,
matches stored) and `people2d` (every frame) in parallel on the one GPU, pull the npz files and DELETE the pod (finally).

Cost guard: pods.gate(max_cost) before creation, a kill switch on the pod (minutes / hard_minutes), delete in `finally`.
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
MODULES = ["ptz.py", "ptz_feats.py", "ptz_run.py", "people2d.py"]
OWNER = "C2-room"
GPU = "NVIDIA GeForce RTX 4090"
POD_PY = "/opt/hm/env/bin/python"
WDIR = "/opt/hm/c2"
HM = "/workspace/hm"


def log(msg: str) -> None:
    print(f"[ptz_pod {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def start_jobs(pod, cam: str, env: str, fr: str, rect: str, ff: str, seg: str, pose: str,
               only: tuple[str, ...] = ("track", "pose"), shards: int = 1) -> None:
    """Start the detached jobs (each one a single backgrounded nohup, so the ssh call returns at once). The track job
    runs as `shards` processes on contiguous frame blocks (the pod has many CPUs and the GPU idles on one process)."""
    jobs = {"pose": f"{POD_PY} -u -m c2cam.people2d in/video.mp4 --out out/pose_{cam}.npz {fr} --weights {pose} > pose.log 2>&1; echo $? > pose.rc"}
    if shards <= 1:
        jobs["track"] = (f"{POD_PY} -u -m c2cam.ptz_run track in/video.mp4 --skeleton skel --out out/track_{cam}.npz {fr} {rect} {ff} "
                         f"--yolo {seg} > track.log 2>&1; echo $? > track.rc")
    else:
        pod.run(f"cd {WDIR} && {POD_PY} -c \"import json;a=json.load(open('frames.json'));n={shards};"
                f"[json.dump(a[i*len(a)//n:(i+1)*len(a)//n],open(f'frames_{{i}}.json','w')) for i in range(n)]\"")
        for k in range(shards):
            jobs[f"track{k}"] = (f"{POD_PY} -u -m c2cam.ptz_run track in/video.mp4 --skeleton skel --out out/track_{cam}_s{k}.npz "
                                 f"--frames {WDIR}/frames_{k}.json {rect} {ff} --yolo {seg} > track{k}.log 2>&1; echo $? > track{k}.rc")
    for name in only:
        for key in [k for k in jobs if k.startswith(name)]:
            inner = f"{env} && {jobs[key]}"
            pod.run(f"nohup setsid bash -c {shlex.quote(inner)} > /dev/null 2>&1 < /dev/null &")


def reshard(cam: str, pod_id: str, shards: int, rect: str, fix_f: float | None) -> None:
    """Replace a running single-process track job by `shards` parallel ones (and start the pose job)."""
    pod = pods.attach(pod_id)
    weights = f"{HM}/weights/dancecap"
    pod.run("pkill -f c2cam; sleep 2; pkill -9 -f c2cam; true", check=False)
    pod.run(f"cd {WDIR} && rm -f track.rc track*.rc pose.rc out/* track*.log")
    env = (f"cd {WDIR} && export PATH=/opt/hm/env/bin:$PATH PYTHONPATH={WDIR} PYTHONUNBUFFERED=1 "
           f"TORCH_HOME={HM}/weights/torch_home YOLO_CONFIG_DIR={WDIR}/yolo_cfg PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
    start_jobs(pod, cam, env, f"--frames {WDIR}/frames.json", f"--rect {rect}" if rect else "", f"--fix-f {fix_f}" if fix_f else "",
               f"{weights}/yolo11x-seg.pt", f"{weights}/yolo11x-pose.pt", shards=shards)
    log(f"{cam}: {shards} track shards + pose started")


def finish(cam: str, pod_id: str, out: Path, minutes: float = 30, shards: int = 1, price: float = 0.89) -> dict:
    """Attach to a pod of this job, (re)start the pose job when it is not running, wait for both jobs, pull, DELETE the pod."""
    out.mkdir(parents=True, exist_ok=True)
    pod = pods.attach(pod_id)
    t0 = time.time()
    try:
        wait_and_pull(pod, cam, out, minutes, shards)
        secs = time.time() - t0
        return {"cam": cam, "seconds": secs}
    finally:
        pods.delete(pod_id, reason="C2 room job done")
        log(f"{cam}: pod deleted")


def wait_and_pull(pod, cam: str, out: Path, minutes: float, shards: int = 1) -> None:
    need = shards + 1
    deadline = time.time() + minutes * 60
    while time.time() < deadline:
        time.sleep(45)
        try:
            st = pod.output(f"cd {WDIR}; tail -qn1 track*.log pose.log 2>/dev/null | cut -c1-60 | tr '\\n' '|'; echo; ls *.rc 2>/dev/null | wc -l",
                            timeout=60).strip().splitlines()
        except Exception as e:      # ssh hiccup: keep polling
            log(f"{cam}: poll failed ({str(e)[:80]})")
            continue
        log(f"{cam}: " + " || ".join(st))
        if st and st[-1].strip() == str(need):
            break
    rcs = pod.output(f"cd {WDIR}; for f in *.rc; do echo $f $(cat $f); done").strip().replace(chr(10), "; ")
    log(f"{cam}: exit codes {rcs}")
    files = pod.output(f"cd {WDIR}/out; ls").split()
    for f in files:
        pod.get(f"{WDIR}/out/{f}", out / f, timeout=1800)
    pod.run(f"cd {WDIR}; tar -czf logs.tgz *.log && echo ok", check=False)
    pod.get(f"{WDIR}/logs.tgz", out / f"{cam}_logs.tgz", timeout=300)


def run_cam(cam: str, video: Path, skeleton: Path, out: Path, rects: list[str], fix_f: float | None,
            frames: str | None = None, minutes: float = 40, max_cost: float = 1.2, gpu: str = GPU,
            yolo_seg: str = "yolo11x-seg.pt", yolo_pose: str = "yolo11x-pose.pt") -> dict:
    out.mkdir(parents=True, exist_ok=True)
    price = next((o["price"] for o in pods.gpu_offers(priority=[gpu])), 0.9)
    worst = price * (minutes + 15 + pods.START_TIMEOUT_MIN) / 60
    log(f"{cam}: {gpu} ${price:.2f}/h, worst case ${worst:.2f}")
    if worst > max_cost:
        raise remote.BudgetError(f"worst case {worst:.2f} > max_cost {max_cost}")
    pods.gate(worst)
    info = pods.create(f"hm-c2-room-{cam}", gpu=gpu, minutes=minutes, hard_minutes=minutes + 15, entry=True, autostart=False,
                       disk_gb=40, price=price, owner=OWNER)
    pod_id = info["pod_id"]
    t0 = time.time()
    try:
        pod = pods.pod_handle(info)
        pod.run(f"bash {HM}/bin/pod_prepare.sh env libs 2>&1 | tail -3", timeout=900)
        pod.run(f"mkdir -p {WDIR}/c2cam {WDIR}/in {WDIR}/out {WDIR}/skel")
        for m in MODULES:
            pod.put(HERE / m, f"{WDIR}/c2cam/{m}")
        pod.run(f"touch {WDIR}/c2cam/__init__.py")
        for f in ("skeleton.npz", "skeleton.json"):
            if (skeleton / f).exists():
                pod.put(skeleton / f, f"{WDIR}/skel/{f}")
        t1 = time.time()
        pod.put(video, f"{WDIR}/in/video.mp4", timeout=1800)
        log(f"{cam}: video uploaded in {time.time() - t1:.0f} s")
        weights = f"{HM}/weights/dancecap"
        for w in (yolo_seg, yolo_pose):
            if pod.run(f"test -s {weights}/{w}", check=False) != 0:
                local = HERE.parent.parent / "weights" / w
                log(f"{cam}: {w} not on the volume, uploading {local.stat().st_size >> 20} MB")
                pod.run(f"mkdir -p {WDIR}/w")
                pod.put(local, f"{WDIR}/w/{w}", timeout=1800)
        seg_path = lambda w: f"{weights}/{w}" if pod.run(f"test -s {weights}/{w}", check=False) == 0 else f"{WDIR}/w/{w}"
        env = (f"cd {WDIR} && export PATH=/opt/hm/env/bin:$PATH PYTHONPATH={WDIR} PYTHONUNBUFFERED=1 "
               f"TORCH_HOME={HM}/weights/torch_home YOLO_CONFIG_DIR={WDIR}/yolo_cfg PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
        rect = " ".join(f"--rect {r}" for r in rects)
        ff = f"--fix-f {fix_f}" if fix_f else ""
        fr = f"--frames {WDIR}/frames.json" if frames else ""
        if frames:
            pod.put(Path(frames), f"{WDIR}/frames.json")
        else:
            n = int(pod.output(f"{POD_PY} -c \"import cv2;print(int(cv2.VideoCapture('{WDIR}/in/video.mp4').get(7)))\"").strip())
            pod.run(f"{POD_PY} -c \"import json;json.dump(list(range({n})),open('{WDIR}/frames.json','w'))\"")
            fr = f"--frames {WDIR}/frames.json"
        seg, pose = seg_path(yolo_seg), seg_path(yolo_pose)
        start_jobs(pod, cam, env, fr, rect, ff, seg, pose)
        log(f"{cam}: jobs started ({time.time() - t0:.0f} s since pod creation)")
        wait_and_pull(pod, cam, out, minutes)
        secs = time.time() - t0
        cost = price * secs / 3600
        log(f"{cam}: done in {secs:.0f} s, about ${cost:.2f}")
        return {"cam": cam, "seconds": secs, "cost_usd": cost, "price_per_h": price}
    finally:
        try:
            pods.delete(pod_id, reason="C2 room job done")
            log(f"{cam}: pod {pod_id[:4]}... deleted")
        except Exception as e:
            log(f"{cam}: DELETE FAILED for {pod_id}: {e}")
            raise


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("cam")
    r.add_argument("--video", required=True)
    r.add_argument("--skeleton", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--rect", action="append", default=[])
    r.add_argument("--fix-f", type=float)
    r.add_argument("--frames")
    r.add_argument("--minutes", type=float, default=40)
    r.add_argument("--max-cost", type=float, default=1.2)
    f = sub.add_parser("finish")
    f.add_argument("cam")
    f.add_argument("pod_id")
    f.add_argument("--out", required=True)
    f.add_argument("--minutes", type=float, default=30)
    f.add_argument("--shards", type=int, default=1)
    g = sub.add_parser("reshard")
    g.add_argument("cam")
    g.add_argument("pod_id")
    g.add_argument("--shards", type=int, default=5)
    g.add_argument("--rect", default="")
    g.add_argument("--fix-f", type=float)
    a = ap.parse_args(argv)
    if a.cmd == "finish":
        print(json.dumps(finish(a.cam, a.pod_id, Path(a.out), a.minutes, a.shards)))
        return
    if a.cmd == "reshard":
        reshard(a.cam, a.pod_id, a.shards, a.rect, a.fix_f)
        return
    res = run_cam(a.cam, Path(a.video), Path(a.skeleton), Path(a.out), a.rect, a.fix_f, a.frames, a.minutes, a.max_cost)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
