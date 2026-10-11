"""john_lula: the DIRECT Runpod job of the 2D people stage (agent J3a): one GPU pod mounting the hm-farm volume, never the shared farm queue.

    python -m dancecap.jl_people_pod up                 create the pod (owner J3a), unpack the farm env, pip install easyocr, upload video + code
    python -m dancecap.jl_people_pod start pose|ocr|all   start the detached pod job(s) (jl_people_detect.py)
    python -m dancecap.jl_people_pod status             tails of the logs + exit codes
    python -m dancecap.jl_people_pod pull               download out/* to work/john_lula/people/in/
    python -m dancecap.jl_people_pod put FILE [DEST]    copy a file to the pod's work directory
    python -m dancecap.jl_people_pod run "CMD"          run a shell command in the pod's work directory
    python -m dancecap.jl_people_pod bg NAME "CMD"      the same, detached (NAME.log, NAME.rc)
    python -m dancecap.jl_people_pod down               delete the pod (verified)

The video is likeness data: it goes to the owner's Runpod pod / volume only and is never copied into a repository. State: work/john_lula/people/pod.json.
"""
from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from pathlib import Path

from . import remote
from .farm import pods
from .take import PACKAGE_ROOT, WORKSPACE

HERE = Path(__file__).resolve().parent
WORK = PACKAGE_ROOT / "work" / "john_lula"
PEOPLE = WORK / "people"
STATE = PEOPLE / "pod.json"
IN = PEOPLE / "in"
VIDEO = WORKSPACE / "dances" / "John-Lula" / "john_lula_jnj.mp4"
OWNER = "J3a"
GPU = "NVIDIA GeForce RTX 4090"
HM = "/workspace/hm"
WDIR = "/opt/hm/j3a"
POD_PY = "/opt/hm/env/bin/python"
ENV = (f"cd {WDIR} && export PATH=/opt/hm/env/bin:$PATH PYTHONUNBUFFERED=1 TORCH_HOME={HM}/weights/torch_home "
       f"EASYOCR_MODULE_PATH={WDIR}/easyocr YOLO_CONFIG_DIR={WDIR}/yolo_cfg PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME={WDIR}/hf "
       f"&& export LD_LIBRARY_PATH=$(ls -d /opt/hm/env/lib/python3.11/site-packages/nvidia/*/lib | tr '\n' ':')$LD_LIBRARY_PATH")
WEIGHTS = f"{HM}/weights/dancecap/yolo11x-pose.pt"


def log(msg: str) -> None:
    print(f"[jl_people_pod {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def pod() -> remote.Pod:
    return pods.attach(state()["pod_id"])


def up(minutes: float = 90, max_cost: float = 2.0) -> None:
    PEOPLE.mkdir(parents=True, exist_ok=True)
    price = next((o["price"] for o in pods.gpu_offers(priority=[GPU])), 0.9)
    worst = price * (minutes + 15 + pods.START_TIMEOUT_MIN) / 60
    log(f"{GPU} ${price:.2f}/h, worst case ${worst:.2f}")
    if worst > max_cost:
        raise remote.BudgetError(f"worst case {worst:.2f} > max_cost {max_cost}")
    pods.gate(worst)
    info = pods.create("hm-j3a-people", gpu=GPU, minutes=minutes, hard_minutes=minutes + 15, entry=True, autostart=False, disk_gb=40, price=price,
                       owner=OWNER)
    STATE.write_text(json.dumps({"pod_id": info["pod_id"], "created": time.time(), "price": price, "minutes": minutes}))
    p = pods.pod_handle(info)
    try:
        p.run(f"bash {HM}/bin/pod_prepare.sh env libs 2>&1 | tail -3", timeout=900)
        p.run(f"mkdir -p {WDIR}/in {WDIR}/out {WDIR}/yolo_cfg {WDIR}/easyocr")
        p.run(f"printf '%s\\n' numpy==1.26.4 torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 > {WDIR}/constraints.txt; "
              f"{POD_PY} -m pip install -q -c {WDIR}/constraints.txt easyocr 2>&1 | tail -3; "
              f"{POD_PY} -c 'import easyocr, cv2, ultralytics, torch; print(easyocr.__version__, cv2.__version__, ultralytics.__version__, torch.__version__)'", timeout=900)
        p.put(HERE / "jl_people_detect.py", f"{WDIR}/jl_people_detect.py")
        t1 = time.time()
        p.put(VIDEO, f"{WDIR}/in/video.mp4", timeout=900)
        log(f"video uploaded in {time.time() - t1:.0f} s")
        p.run(f"test -s {WEIGHTS} && ls -la {WEIGHTS}")
    except BaseException:
        pods.delete(info["pod_id"], reason="J3a up failed")
        STATE.unlink(missing_ok=True)
        raise
    log(f"pod {info['pod_id']} ready")


def start(what: str, extra: str = "") -> None:
    p = pod()
    jobs = {"pose": f"{POD_PY} -u jl_people_detect.py pose in/video.mp4 out/pose.npz --weights {WEIGHTS} {extra} > pose.log 2>&1; echo $? > pose.rc",
            "ocr": f"{POD_PY} -u jl_people_detect.py ocr in/video.mp4 out/pose.npz out/ocr.json {extra} > ocr.log 2>&1; echo $? > ocr.rc"}
    for name in (["pose", "ocr"] if what == "all" else [what]):
        inner = f"{ENV} && rm -f {name}.rc && {jobs[name]}"
        p.run(f"nohup setsid bash -c {shlex.quote(inner)} > /dev/null 2>&1 < /dev/null &")
        log(f"started {name}")


def bg(name: str, cmd: str) -> None:
    """Run `cmd` detached in the pod's work directory: log <name>.log, exit code <name>.rc."""
    p = pod()
    inner = f"{ENV} && rm -f {name}.rc && {cmd} > {name}.log 2>&1; echo $? > {name}.rc"
    p.run(f"nohup setsid bash -c {shlex.quote(inner)} > /dev/null 2>&1 < /dev/null &")
    log(f"started {name}")


def status() -> None:
    p = pod()
    print(p.output(f"cd {WDIR}; for f in *.log; do echo == $f; tail -n3 $f | cut -c1-200; done; for f in *.rc; do echo $f $(cat $f); done; ls -la out; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader"))


def pull() -> None:
    p = pod()
    IN.mkdir(parents=True, exist_ok=True)
    for line in p.output(f"cd {WDIR}/out; stat -c '%n %s' *").strip().splitlines():
        f, size = line.split()
        if (IN / f).exists() and (IN / f).stat().st_size == int(size):
            continue
        p.get(f"{WDIR}/out/{f}", IN / f, timeout=1800)
        log(f"pulled {f} {(IN / f).stat().st_size >> 10} KB")


def down() -> None:
    s = state()
    if not s:
        log("no pod in the state file")
        return
    pods.delete(s["pod_id"], reason="J3a done")
    log(f"pod {s['pod_id']} deleted")
    STATE.write_text(json.dumps({**s, "deleted": time.time()}))


def main(argv=None) -> int:
    sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(prog="python -m dancecap.jl_people_pod")
    ap.add_argument("cmd", choices=("up", "start", "status", "pull", "put", "run", "bg", "down"))
    ap.add_argument("args", nargs="*")
    ap.add_argument("--extra", default="")
    ns = ap.parse_args(argv)
    if ns.cmd == "up":
        up()
    elif ns.cmd == "start":
        start(ns.args[0], ns.extra)
    elif ns.cmd == "status":
        status()
    elif ns.cmd == "pull":
        pull()
    elif ns.cmd == "put":
        src = Path(ns.args[0])
        pod().put(src, f"{WDIR}/{ns.args[1] if len(ns.args) > 1 else src.name}", timeout=900)
    elif ns.cmd == "bg":
        bg(ns.args[0], ns.args[1])
    elif ns.cmd == "run":
        pod().run(f"{ENV} && {ns.args[0]}")
    else:
        down()
    return 0


if __name__ == "__main__":
    sys.exit(main())
