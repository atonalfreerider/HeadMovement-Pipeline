"""Farm pods: start N Runpod pods that mount the hm-farm network volume, with budget gates and kill switches.

A farm pod runs the image's python 3.11 for nothing but its own start: everything it needs (the venv, weights, gated
body models, code at a hash, the world package, queued job specs) is on the volume mounted at /workspace, so the pod's
container command (`bootstrap/pod_entry.sh` on the volume) arms the kill switch, installs the cached system libs and
starts the worker (dancecap.farm.worker) without any upload or pip install.

    from dancecap.farm import pods
    ps = pods.launch(n=3, gpus=None, minutes=90, max_cost=9.0, kinds="fit,refine")   # N pods, GPU priority list
    pod = pods.attach(ps[0]["pod_id"])                                              # ssh handle (remote.Pod)
    pods.extend_killswitch(pod, add_minutes=30)                                     # never past the hard ceiling
    pods.delete(pod.id, reason="chunks pulled")                                     # ledger closed

Budget (user rule, FARM.md section 6): a launch passes the gate `max_cost <= balance - 2` and `<= remaining cap`, counting
the worst case of every pod already open in the ledger (other workflows' pods included). Every pod has TWO guards: the
on-pod deadline file (extendable over ssh, bounded by the hard ceiling given at creation) and the local delete.

CPU pods (gateway for uploads / pulls / `ls` while no GPU pod is up) cost about $0.035 per vCPU-hour.
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from pathlib import Path

from .. import remote
from ..take import PACKAGE_ROOT

VOLUME_JSON = PACKAGE_ROOT / "farm_volume.json"
POD_PREFIX = "hm-farm"
IMAGE = remote.DEFAULT_IMAGE
ENTRY_CMD = "bash /workspace/hm/bin/pod_entry.sh"
# GPUs MAMMA's CUDA wheels (sm80/sm89/sm90) run on, best value first; the ones the volume's DC has no stock of are skipped
GPU_PRIORITY = ["NVIDIA GeForce RTX 4090", "NVIDIA L4", "NVIDIA RTX A4500", "NVIDIA RTX A5000", "NVIDIA A40", "NVIDIA RTX A6000",
                "NVIDIA RTX 4000 Ada Generation", "NVIDIA GeForce RTX 3090", "NVIDIA L40S", "NVIDIA L40",
                "NVIDIA RTX 6000 Ada Generation", "NVIDIA A100 80GB PCIe", "NVIDIA A100-SXM4-80GB", "NVIDIA H100 PCIe",
                "NVIDIA H100 80GB HBM3", "NVIDIA H100 NVL", "NVIDIA H200", "NVIDIA RTX A4000"]
# sm80+ only (A100 / A40 / A5000 / 4090 / L4 / H100 ...): torch 2.5.1+cu124 and the compiled wheels have no Blackwell
# (sm120: RTX PRO 4500 / 6000, 5090, B200) kernels. >= 16 GB (ma_3d needs ~11 GB).
MIN_BALANCE_RESERVE = 2.0   # user rule: never let the sum of worst cases exceed (balance - $2)
REST = remote.REST
START_TIMEOUT_MIN = 8.0


class PodError(RuntimeError):
    pass


def volume_info() -> dict:
    """farm_volume.json: {id, dc, size_gb, tier, root, mount, ...} (tracked, no secrets)."""
    if not VOLUME_JSON.exists():
        raise PodError(f"{VOLUME_JSON} missing: create the volume first (python -m dancecap.farm volume create)")
    return json.loads(VOLUME_JSON.read_text())


# ------------------------------------------------------------------------------------------------ budget


def balance() -> float:
    me = remote._ctl("user", check=False)
    if not isinstance(me, dict) or "clientBalance" not in me:
        raise PodError(f"cannot read the account balance: {str(me)[:200]}")
    return float(me["clientBalance"])


def committed(now: float | None = None) -> float:
    """Worst case still to be paid for pods that are open in the ledger: price x (max_minutes - age), never negative."""
    now = now or time.time()
    total = 0.0
    for r in remote.ledger():
        if r.get("deleted_ts"):
            continue
        left = max(0.0, float(r.get("max_minutes") or 0) * 60 - (now - r["created_ts"]))
        total += r["cost_per_hr"] * left / 3600
    return total


def gate(max_cost: float, *, balance_usd: float | None = None) -> dict:
    """Raise BudgetError unless `max_cost` fits both the remaining cap and (balance - $2), counting open pods' worst case."""
    bal = balance() if balance_usd is None else balance_usd
    rem = remote.remaining()
    com = committed()
    limit = min(rem, bal - MIN_BALANCE_RESERVE) - com
    info = {"max_cost": round(max_cost, 3), "balance": round(bal, 2), "cap_remaining": round(rem, 2),
            "committed_open_pods": round(com, 2), "available": round(limit, 2)}
    if max_cost > limit + 1e-9:
        raise remote.BudgetError(f"worst case ${max_cost:.2f} > available ${limit:.2f} "
                                 f"(min(cap remaining ${rem:.2f}, balance ${bal:.2f} - ${MIN_BALANCE_RESERVE:.0f}) "
                                 f"- open pods' worst case ${com:.2f})")
    return info


# ------------------------------------------------------------------------------------------------ catalog


def gpu_offers(dc: str | None = None, priority: list[str] | None = None) -> list[dict]:
    """GPUs from the priority list that have stock in the volume's data center right now: [{gpu, price, stock}]."""
    dc = dc or volume_info()["dc"]
    by_id = {g["gpuId"]: g for g in remote.gpu_list()}
    out = []
    for gid in priority or GPU_PRIORITY:
        g = by_id.get(gid)
        if not g or not g.get("secureCloud") or not g.get("securePricePerHr"):
            continue
        st = next((x.get("stockStatus") for x in g.get("dataCenterAvailability", []) if x.get("dataCenterId") == dc), None)
        if st and str(st).lower() not in ("none", ""):
            out.append({"gpu": gid, "price": float(g["securePricePerHr"]), "stock": st, "memory_gb": g.get("memoryInGb")})
    return out


# ------------------------------------------------------------------------------------------------ REST (CPU pods)


def _rest(method: str, path: str, body: dict | None = None, timeout: float = 60):
    try:
        return remote.rest(method, path, body, timeout=timeout)   # the runner's CPU pods use the same call
    except remote.RestError as e:
        raise PodError(str(e)) from None


# ------------------------------------------------------------------------------------------------ create / attach / delete


def _env_for(name: str, minutes: float, hard_minutes: float, autostart: bool, kinds: str | None, max_jobs: int | None,
             idle_min: float, post_grace_min: float, extra: dict | None, price: float | None = None) -> dict:
    now = int(time.time())
    env = {"HM_ROOT": "/workspace/hm", "HM_POD_NAME": name, "HM_AUTOSTART": "1" if autostart else "0",
           "HM_KILL_AT": str(now + int(minutes * 60) + int(START_TIMEOUT_MIN * 60)),
           "HM_HARD_CEIL": str(now + int(hard_minutes * 60) + int(START_TIMEOUT_MIN * 60)),
           "HM_IDLE_MIN": str(idle_min), "HM_POST_GRACE_MIN": str(post_grace_min),
           "HM_IMAGE": IMAGE, "HM_PRICE_HR": str(price or 0)}
    if kinds:
        env["HM_KINDS"] = kinds
    if max_jobs:
        env["HM_MAX_JOBS"] = str(max_jobs)
    env.update({k: str(v) for k, v in (extra or {}).items()})
    return env


def create(name: str, *, gpu: str | None = None, cpu: str | None = None, vcpu: int = 4, minutes: float = 60,
           hard_minutes: float | None = None, entry: bool = True, autostart: bool = True, kinds: str | None = None,
           max_jobs: int | None = None, idle_min: float = 5, post_grace_min: float = 15, disk_gb: int = 40,
           image: str = IMAGE, volume: bool = True, extra_env: dict | None = None, price: float | None = None,
           wait: bool = True, start_timeout_min: float = START_TIMEOUT_MIN, owner: str | None = None) -> dict:
    """Create ONE pod (GPU if `gpu`, else a CPU pod of flavor `cpu` with `vcpu` vCPUs), record it in the ledger the moment
    it exists, and (wait=True) return once its public ssh port answers.

    entry=True runs bootstrap/pod_entry.sh from the volume as the container command (kill switch + worker); entry=False
    keeps the image's own start (used for the very first bootstrap, before the volume has any script).
    Returns {pod_id, name, gpu, cost_per_hr, host, port, t_create, t_ssh, hard_minutes, ...}; raises PodError.
    """
    vol = volume_info()
    hard_minutes = float(hard_minutes or minutes)
    owner = owner or os.environ.get("HM_POD_OWNER") or None      # C6 2026-10-10: a session labels its pods in the ledger without touching every caller
    remote.ensure_ssh_key()
    if os.environ.get("HM_QUEUE_OWNER"):         # J3b (additive): this session's pods claim only the jobs whose spec names the same owner (worker.claimable)
        extra_env = {**(extra_env or {}), "HM_OWNER": os.environ["HM_QUEUE_OWNER"]}
    env = _env_for(name, minutes, hard_minutes, autostart, kinds, max_jobs, idle_min, post_grace_min, extra_env, price)
    t_create = time.time()
    if gpu:
        args = ["pod", "create", "--name", name, "--image", image, "--gpu-id", gpu, "--cloud-type", "SECURE",
                "--data-center-ids", vol["dc"], "--container-disk-in-gb", str(disk_gb), "--ports", "22/tcp",
                "--env", json.dumps(env)]
        if volume:
            args += ["--network-volume-id", vol["id"], "--volume-mount-path", vol.get("mount", "/workspace")]
        if entry:
            args += ["--docker-args", ENTRY_CMD]
        created = remote._ctl(*args, timeout=240)
        if not isinstance(created, dict) or "id" not in created:
            raise PodError(f"pod create returned {str(created)[:300]}")
        pod_id = created["id"]
        cost = float(created.get("costPerHr") or price or 0)
        gname = gpu
    else:
        body = {"name": name, "imageName": image, "computeType": "CPU", "cloudType": "SECURE",
                "cpuFlavorIds": [cpu or "cpu5c"], "vcpuCount": vcpu, "dataCenterIds": [vol["dc"]],
                "containerDiskInGb": disk_gb, "ports": ["22/tcp"], "env": env}
        if volume:
            body.update({"networkVolumeId": vol["id"], "volumeMountPath": vol.get("mount", "/workspace")})
        if entry:
            body["dockerStartCmd"] = shlex.split(ENTRY_CMD)
        created = _create_cpu(name, body)
        pod_id = created["id"]
        cost = float(created.get("costPerHr") or price or 0)
        gname = f"cpu:{cpu or 'cpu5c'}x{vcpu}"
    remote._record(pod_id, name, gname, "SECURE", cost, hard_minutes + start_timeout_min, **({"owner": owner} if owner else {}))
    out = {"pod_id": pod_id, "name": name, "gpu": gname, "cost_per_hr": cost, "t_create": t_create,
           "minutes": minutes, "hard_minutes": hard_minutes, "kill_at": int(env["HM_KILL_AT"]),
           "hard_ceil": int(env["HM_HARD_CEIL"]), "dc": vol["dc"], "volume": vol["id"]}
    print(f"[pods] created {pod_id} {gname} ${cost:.2f}/h (worst case ${cost * (hard_minutes + start_timeout_min) / 60:.2f}); "
          f"waiting for ssh", flush=True)
    if wait:
        try:
            p = remote._wait_running(pod_id, start_timeout_min * 60)
        except BaseException:
            delete(pod_id, reason="interrupted while starting")
            raise
        if not p:
            delete(pod_id, reason="not reachable over direct ssh")
            raise PodError(f"{gname} {pod_id}: no ssh in {start_timeout_min:.0f} min (deleted)")
        host, port = remote._pod_ssh(p)
        out.update({"host": host, "port": port, "t_ssh": time.time()})
        print(f"[pods] {pod_id} ssh {host}:{port} after {out['t_ssh'] - t_create:.0f} s", flush=True)
    return out


def _create_cpu(name: str, body: dict, tries: int = 4) -> dict:
    """POST /pods with retries on 5xx. A 5xx may still have created the pod (it would bill unrecorded), so before every
    retry the account's pods are searched by name and an existing one is adopted instead of creating a second."""
    last: Exception | None = None
    for k in range(tries):
        try:
            return _rest("POST", "/pods", body)
        except PodError as e:
            last = e
            if "HTTP 5" not in str(e):
                raise
            time.sleep(4 + 4 * k)
            for q in list_farm_pods(include_all=True):
                if q["name"] == name:
                    print(f"[pods] adopting {q['id']} created despite the error", flush=True)
                    return {"id": q["id"], "costPerHr": q.get("cost_per_hr")}
    raise PodError(f"create {name}: {last}")


def pod_handle(info: dict) -> remote.Pod:
    return remote.Pod(info["pod_id"], info.get("name", ""), info.get("gpu", ""), float(info.get("cost_per_hr") or 0),
                      info["host"], int(info["port"]))


def attach(pod_id: str) -> remote.Pod:
    """ssh handle for an existing pod (reattach after a dropped session); raises PodError if it has no public ssh."""
    j = remote._ctl("pod", "get", pod_id, check=False)
    if not isinstance(j, dict) or not j.get("id"):
        raise PodError(f"pod {pod_id} not found: {str(j)[:200]}")
    found = remote._direct_ssh(j)
    if not found:
        raise PodError(f"pod {pod_id} has no public ssh yet (status {j.get('desiredStatus') or j.get('status')})")
    gpu = (j.get("machine") or {}).get("gpuTypeId") or j.get("gpuTypeId") or (j.get("gpu") or {}).get("id") or ""
    cost = float(j.get("costPerHr") or 0)
    return remote.Pod(pod_id, j.get("name", ""), gpu, cost, found[0], int(found[1]))


def list_farm_pods(include_all: bool = False) -> list[dict]:
    """Pods of this account whose name starts with hm-farm (all pods when include_all): [{id, name, status, cost}]."""
    data = remote._ctl("pod", "list", check=False)
    pods = data if isinstance(data, list) else (data.get("pods", []) if isinstance(data, dict) else [])
    return [{"id": p.get("id"), "name": p.get("name"), "status": p.get("desiredStatus") or p.get("status"),
             "cost_per_hr": p.get("costPerHr"), "gpu": (p.get("machine") or {}).get("gpuTypeId") or p.get("gpuTypeId"),
             "created": p.get("createdAt") or p.get("lastStartedAt")}
            for p in pods if include_all or str(p.get("name", "")).startswith(POD_PREFIX)]


def delete(pod_id: str, reason: str = "done") -> None:
    """Delete a pod and close its ledger row. Only call for pods this farm created (the name starts with hm-farm)."""
    remote._delete(pod_id, reason=reason)


# ------------------------------------------------------------------------------------------------ kill switch


def arm_killswitch(pod: remote.Pod, minutes: float, hard_epoch: int | None = None) -> dict:
    """Start (or re-arm) the on-pod dead-man switch on a pod that did not start through pod_entry.sh (the bootstrap pod,
    a pod from another image): writes the deadline file and runs hm_killswitch.sh from the volume (or inline)."""
    deadline = int(time.time() + minutes * 60)
    hard = int(hard_epoch or deadline)
    pd = f"/workspace/hm/pods/{pod.id}"
    script = (Path(__file__).parent / "bootstrap" / "hm_killswitch.sh").read_text(encoding="utf-8").replace("\r\n", "\n")
    pod.run(f"mkdir -p {pd} /workspace/hm/bin && cat > /workspace/hm/bin/hm_killswitch.sh <<'HM_EOF'\n{script}\nHM_EOF\n"
            f"chmod +x /workspace/hm/bin/hm_killswitch.sh && echo {deadline} > {pd}/kill_at", check=True, timeout=60, echo=False)
    pod.run(f"pgrep -f 'hm_killswitch.sh {pd}' > /dev/null || (HM_KILL_AT={deadline} nohup /workspace/hm/bin/hm_killswitch.sh "
            f"{pd}/kill_at {hard} >> {pd}/killswitch.log 2>&1 < /dev/null &)", check=True, timeout=60, echo=False)
    try:
        out = pod.output(f"sleep 2; pgrep -f 'hm_killswitch.sh {pd}' > /dev/null && echo armed=1 || echo armed=0; "
                         "command -v runpodctl > /dev/null && echo ctl=1 || echo ctl=0; "
                         "tr '\\0' '\\n' < /proc/1/environ | grep -c -E '^RUNPOD_(API_KEY|POD_ID)=' || true", timeout=60)
        w = out.split()
        info = {"armed": "armed=1" in w and "ctl=1" in w and w[-1] == "2", "deadline": deadline, "hard": hard}
    except (RuntimeError, subprocess.SubprocessError) as e:
        info = {"armed": False, "error": str(e)[:200], "deadline": deadline, "hard": hard}
    if not info["armed"]:
        print(f"[pods] WARNING: on-pod kill switch of {pod.id} not verified ({info}); the local delete is the only guard")
    pod.killswitch = info
    return info


def read_killswitch(pod: remote.Pod) -> dict:
    out = pod.output(f"cat /workspace/hm/pods/{pod.id}/kill_at 2>/dev/null; echo; date +%s", timeout=60).split()
    return {"kill_at": int(out[0]) if out and out[0].isdigit() else None, "pod_now": int(out[-1]) if out else None}


def extend_killswitch(pod: remote.Pod, add_minutes: float = 30, hard_epoch: int | None = None,
                      via: remote.Pod | None = None) -> int:
    """Push the pod's on-pod deadline later by `add_minutes` from the LATER of now and the current deadline, never past
    `hard_epoch` (the ceiling given at creation; the on-pod script enforces it too). `via` = any other attached pod
    (the deadline file is on the shared volume), for a pod whose own ssh is down. Returns the new deadline epoch."""
    p = via or pod
    pd = f"/workspace/hm/pods/{pod.id}"
    cur = p.output(f"cat {pd}/kill_at 2>/dev/null | tr -dc 0-9; echo; date +%s", timeout=60).split()
    now = int(cur[-1])
    old = int(cur[0]) if len(cur) > 1 and cur[0].isdigit() else now
    new = max(now, old) + int(add_minutes * 60)
    if hard_epoch:
        new = min(new, int(hard_epoch))
    if new <= old:
        return old
    p.output(f"echo {new} > {pd}/kill_at.tmp && mv {pd}/kill_at.tmp {pd}/kill_at", timeout=60)
    return new


# ------------------------------------------------------------------------------------------------ launch N


def launch(n: int = 1, *, gpus: list[str] | None = None, minutes: float = 90, hard_minutes: float | None = None,
           max_cost: float | None = None, kinds: str | None = None, max_jobs: int | None = None, idle_min: float = 5,
           post_grace_min: float = 15, prefix: str = POD_PREFIX, dry_run: bool = False, extra_env: dict | None = None,
           owner: str | None = None) -> list[dict]:
    """Start `n` GPU pods from the volume. Each pod takes the first GPU of the priority list that has stock in the
    volume's DC and whose worst case (price x hard minutes) fits the budget gate together with the pods already started
    in this call. `max_cost` = ceiling for the SUM of the launched pods' worst cases (default: what the gate allows).
    Pods that cannot be placed are reported as {"error": ...} entries instead of aborting the others."""
    hard = float(hard_minutes or minutes * 1.5)
    offers = gpu_offers(priority=gpus)
    if not offers:
        raise PodError(f"no stock for {gpus or GPU_PRIORITY} in {volume_info()['dc']} right now")
    bal = balance()
    spent_cost = 0.0
    out: list[dict] = []
    stamp = time.strftime("%m%d-%H%M")
    for i in range(n):
        placed = None
        for o in offers:
            worst = o["price"] * (hard + START_TIMEOUT_MIN) / 60
            if max_cost is not None and spent_cost + worst > max_cost + 1e-9:
                continue
            try:
                gate(spent_cost + worst, balance_usd=bal)
            except remote.BudgetError as e:
                out.append({"error": f"budget: {e}", "gpu": o["gpu"]})
                continue
            if dry_run:
                placed = {"dry_run": True, "gpu": o["gpu"], "price": o["price"], "worst": round(worst, 3)}
                spent_cost += worst
                break
            try:
                placed = create(f"{prefix}-{stamp}-{i + 1}", gpu=o["gpu"], minutes=minutes, hard_minutes=hard, kinds=kinds,
                                max_jobs=max_jobs, idle_min=idle_min, post_grace_min=post_grace_min, price=o["price"],
                                extra_env=extra_env, wait=True, owner=owner)
                spent_cost += worst
                break
            except (PodError, RuntimeError) as e:
                out.append({"error": str(e)[:300], "gpu": o["gpu"]})
        if placed:
            out.append(placed)
        else:
            out.append({"error": "no GPU could be placed within the budget"})
            break
    return out


# ------------------------------------------------------------------------------------------------ CPU pods (analysis)

# Secure-cloud CPU flavors: $ per vCPU-hour (list-cpu-types 2026-10-08) and GB RAM per vCPU. All six flavors had HIGH stock in
# EU-RO-1 (list-data-centers include=CPU_AVAILABILITY, 2026-10-08) and mount the volume (verified: CPU pods created there with
# the volume attached). Compute-optimized (2 GB/vCPU) is the default; memory-heavy stitches use cpu5g / cpu3g.
CPU_FLAVORS = {"cpu5c": (0.035, 2), "cpu3c": (0.03, 2), "cpu5g": (0.046, 4), "cpu3g": (0.04, 4), "cpu5m": (0.065, 8),
               "cpu3m": (0.055, 8)}
CPU_PRIORITY = ["cpu5c", "cpu3c", "cpu5g", "cpu3g"]


def cpu_price(flavor: str, vcpu: int) -> float:
    return round(CPU_FLAVORS[flavor][0] * vcpu, 4)


def launch_cpu(n: int = 1, *, vcpu: int = 8, flavors: list[str] | None = None, minutes: float = 60,
               hard_minutes: float | None = None, max_cost: float | None = None, kinds: str = "analyse",
               max_jobs: int | None = None, idle_min: float = 3, post_grace_min: float = 10, prefix: str = POD_PREFIX + "-cpu",
               dry_run: bool = False, extra_env: dict | None = None, owner: str | None = None) -> list[dict]:
    """Start `n` CPU pods from the volume for the analysis jobs (kind analyse: prepare, stitch, pose_quality, timing, physics,
    counterbalance, floorcraft, analysis_review, camera-gate evidence). Same budget gate and kill switches as `launch`; the
    flavors are tried in order. A CPU pod's worker never claims a GPU job (it has no GPU)."""
    hard = float(hard_minutes or minutes * 1.5)
    bal = balance()
    spent_cost = 0.0
    out: list[dict] = []
    stamp = time.strftime("%m%d-%H%M")
    for i in range(n):
        placed = None
        for fl in flavors or CPU_PRIORITY:
            price = cpu_price(fl, vcpu)
            worst = price * (hard + START_TIMEOUT_MIN) / 60
            if max_cost is not None and spent_cost + worst > max_cost + 1e-9:
                out.append({"error": f"over max_cost: {fl} x{vcpu} worst case ${worst:.2f}", "flavor": fl})
                continue
            try:
                gate(spent_cost + worst, balance_usd=bal)
            except remote.BudgetError as e:
                out.append({"error": f"budget: {e}", "flavor": fl})
                break
            if dry_run:
                placed = {"dry_run": True, "flavor": fl, "vcpu": vcpu, "price": price, "worst": round(worst, 3)}
                spent_cost += worst
                break
            try:
                placed = create(f"{prefix}-{stamp}-{i + 1}", cpu=fl, vcpu=vcpu, minutes=minutes, hard_minutes=hard, kinds=kinds,
                                max_jobs=max_jobs, idle_min=idle_min, post_grace_min=post_grace_min, price=price,
                                extra_env=extra_env, wait=True, owner=owner)
                spent_cost += worst
                break
            except (PodError, RuntimeError) as e:
                out.append({"error": str(e)[:300], "flavor": fl})
        if placed:
            out.append(placed)
        else:
            out.append({"error": "no CPU pod could be placed within the budget"})
            break
    return out


# ------------------------------------------------------------------------------------------------ gateway


def gateway(minutes: float = 30, vcpu: int = 2, reuse: bool = True, owner: str | None = None) -> tuple[remote.Pod, bool]:
    """(ssh handle, created): a pod that mounts the volume, for uploads / pulls / `ls` while no worker pod is the right
    place. Reuses any running farm pod with public ssh (reuse=True), else starts a small CPU pod in gateway mode (kill
    switch armed, no worker, nothing prepared: up in about 40 s, $0.035 per vCPU-hour). The caller deletes it when
    `created` is True (`delete(pod.id)`)."""
    if reuse:
        for q in list_farm_pods():
            if str(q.get("status")).upper() == "RUNNING":
                try:
                    return attach(q["id"]), False
                except PodError:
                    continue
    # the flavors in CPU_PRIORITY order: cpu5c alone left an analysis without its upload step when EU-RO-1 had no cpu5c
    # (HTTP 500 "no longer any instances available" through every _create_cpu retry, 2026-10-21 20:29). Only a stock
    # refusal falls through; any other error (no ssh, budget) is the caller's.
    err: PodError | None = None
    for fl in CPU_PRIORITY:
        try:
            info = create(f"{POD_PREFIX}-gateway", cpu=fl, vcpu=vcpu, minutes=minutes, hard_minutes=minutes * 1.5, entry=True,
                          autostart=False, disk_gb=20, price=cpu_price(fl, vcpu), wait=True, owner=owner)
            return pod_handle(info), True
        except PodError as e:
            if "no longer any instances" not in str(e):
                raise
            print(f"[pods] gateway: no {fl} in stock; trying the next CPU flavor", flush=True)
            err = e
    raise err


# ------------------------------------------------------------------------------------------------ CLI


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m dancecap.farm.pods", description="farm pods on the hm-farm volume")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("offers", help="GPUs with stock in the volume's DC now")
    g = sub.add_parser("gate", help="budget gate for a worst-case cost")
    g.add_argument("cost", type=float)
    sub.add_parser("list", help="pods of this farm")
    l = sub.add_parser("launch", help="start N GPU pods from the volume (workers start by themselves)")
    l.add_argument("-n", type=int, default=1)
    l.add_argument("--gpus", help="comma list (default: the priority list)")
    l.add_argument("--minutes", type=float, default=90)
    l.add_argument("--hard-minutes", type=float)
    l.add_argument("--max-cost", type=float)
    l.add_argument("--kinds")
    l.add_argument("--max-jobs", type=int)
    l.add_argument("--idle-min", type=float, default=5)
    l.add_argument("--dry-run", action="store_true")
    c = sub.add_parser("launch-cpu", help="start N CPU pods from the volume for the analysis jobs (kind analyse)")
    c.add_argument("-n", type=int, default=1)
    c.add_argument("--vcpu", type=int, default=8)
    c.add_argument("--flavors", help="comma list (default cpu5c,cpu3c,cpu5g,cpu3g)")
    c.add_argument("--minutes", type=float, default=60)
    c.add_argument("--hard-minutes", type=float)
    c.add_argument("--max-cost", type=float)
    c.add_argument("--kinds", default="analyse")
    c.add_argument("--max-jobs", type=int)
    c.add_argument("--idle-min", type=float, default=3)
    c.add_argument("--dry-run", action="store_true")
    e = sub.add_parser("extend", help="push a pod's kill switch later")
    e.add_argument("pod_id")
    e.add_argument("--minutes", type=float, default=30)
    d = sub.add_parser("delete", help="delete a pod (ledger closed)")
    d.add_argument("pod_id")
    dn = sub.add_parser("down", help="delete the hm-farm pods listed (dry run unless --yes; --pod limits it; another agent's pods match the prefix too)")
    dn.add_argument("--yes", action="store_true", help="really delete")
    dn.add_argument("--pod", help="comma list of pod ids (default: every hm-farm* pod)")
    sub.add_parser("gateway", help="print an ssh handle of a pod that mounts the volume (starts a CPU gateway if none runs)")
    a = ap.parse_args(argv)
    if a.cmd == "offers":
        print(json.dumps(gpu_offers(), indent=1))
    elif a.cmd == "gate":
        print(json.dumps(gate(a.cost), indent=1))
    elif a.cmd == "list":
        print(json.dumps(list_farm_pods(), indent=1))
    elif a.cmd == "launch":
        print(json.dumps(launch(a.n, gpus=a.gpus.split(",") if a.gpus else None, minutes=a.minutes, hard_minutes=a.hard_minutes,
                                max_cost=a.max_cost, kinds=a.kinds, max_jobs=a.max_jobs, idle_min=a.idle_min, dry_run=a.dry_run), indent=1))
    elif a.cmd == "launch-cpu":
        print(json.dumps(launch_cpu(a.n, vcpu=a.vcpu, flavors=a.flavors.split(",") if a.flavors else None, minutes=a.minutes,
                                    hard_minutes=a.hard_minutes, max_cost=a.max_cost, kinds=a.kinds, max_jobs=a.max_jobs,
                                    idle_min=a.idle_min, dry_run=a.dry_run), indent=1))
    elif a.cmd == "extend":
        pod = attach(a.pod_id)
        print(extend_killswitch(pod, a.minutes))
    elif a.cmd == "delete":
        delete(a.pod_id, reason="cli")
    elif a.cmd == "gateway":
        pod, created = gateway()
        print(json.dumps({"pod_id": pod.id, "host": pod.host, "port": pod.port, "created": created}))
    elif a.cmd == "down":
        only = {x for x in (a.pod or "").split(",") if x}
        found = [q for q in list_farm_pods() if not only or q["id"] in only]
        for q in found:
            print(f"{'deleting' if a.yes else 'would delete'} {q['id']} {q['name']} ({q['status']})")
            if a.yes:
                delete(q["id"], reason="down")
        if found and not a.yes:
            print("dry run: nothing deleted. These may include pods of another agent or session; repeat with --yes (and --pod ids).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
