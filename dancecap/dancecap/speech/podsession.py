"""One pod for the speech jobs, controlled across CLI calls (state in work/<take>/speech/pod_session.json).

The farm's normal way to run work is a queued job that a worker on a farm pod claims (farm/worker.py). This module only starts and
stops THE speech pod (a GPU pod of the hm-farm volume in gateway mode: kill switch armed, no worker yet), runs commands on it, and
starts the farm worker on it for the speech job (jobs.start_worker). Budget rules of the farm apply unchanged: `pods.create` records
the pod in the ledger the moment it exists, the pod has an on-pod kill switch (HM_KILL_AT / HM_HARD_CEIL) and the budget gate
(`pods.gate`) runs BEFORE anything is created. Only the pod recorded in this session file is ever deleted by this module.

    python -m dancecap.speech pod up [--gpu "NVIDIA L4"] [--minutes 100] [--hard-minutes 130]
    python -m dancecap.speech pod sh "nvidia-smi"        # run a command (streams the output)
    python -m dancecap.speech pod status | extend --minutes 30 | down
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .. import remote
from ..farm import pods
from . import SPEECH

STATE = SPEECH / "pod_session.json"
DEFAULT_GPUS = ["NVIDIA L4", "NVIDIA GeForce RTX 4090", "NVIDIA RTX A5000", "NVIDIA RTX A4500"]


def load() -> dict:
    try:
        return json.loads(STATE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def save(doc: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_name(STATE.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    tmp.replace(STATE)


def offers(prefer: list[str] | None = None) -> list[dict]:
    """GPUs with stock in the volume's data center, cheapest of the preferred list first."""
    want = prefer or DEFAULT_GPUS
    have = {o["gpu"]: o for o in pods.gpu_offers()}
    return [have[g] for g in want if g in have]


def estimate(gpu_price: float, minutes: float, hard_minutes: float) -> dict:
    worst = gpu_price * (hard_minutes + pods.START_TIMEOUT_MIN) / 60
    return {"price_hr": gpu_price, "expected_usd": round(gpu_price * minutes / 60, 2), "worst_case_usd": round(worst, 2)}


def up(gpu: str | None = None, minutes: float = 100, hard_minutes: float = 130, max_cost: float = 2.0, disk_gb: int = 60) -> dict:
    """Create the speech pod (budget gate first). Returns the session record; raises when a pod of this session is still recorded."""
    cur = load()
    if cur.get("pod_id") and not cur.get("deleted"):
        raise RuntimeError(f"speech pod {cur['pod_id']} is still recorded in {STATE}; use `pod down` first")
    offs = offers([gpu] if gpu else None)
    if not offs:
        raise RuntimeError(f"no stock for {gpu or DEFAULT_GPUS} in the volume's data center right now")
    errors = []
    for o in offs:
        est = estimate(o["price"], minutes, hard_minutes)
        if est["worst_case_usd"] > max_cost:
            errors.append(f"{o['gpu']}: worst case ${est['worst_case_usd']} > max ${max_cost}")
            continue
        try:
            gate = pods.gate(est["worst_case_usd"])
        except remote.BudgetError as e:
            errors.append(f"{o['gpu']}: {e}")
            continue
        name = f"hm-farm-speech-{time.strftime('%m%d-%H%M')}"
        try:
            info = pods.create(name, gpu=o["gpu"], minutes=minutes, hard_minutes=hard_minutes, entry=True, autostart=False,
                               disk_gb=disk_gb, price=o["price"], wait=True, owner="speech")
        except (pods.PodError, RuntimeError) as e:
            errors.append(f"{o['gpu']}: {str(e)[:200]}")
            continue
        doc = {"pod_id": info["pod_id"], "name": name, "gpu": o["gpu"], "price_hr": info.get("cost_per_hr") or o["price"],
               "created": time.time(), "minutes": minutes, "hard_minutes": hard_minutes, "estimate": est, "gate": gate,
               "host": info.get("host"), "port": info.get("port"), "kill_at": info.get("kill_at"), "hard_ceil": info.get("hard_ceil")}
        save(doc)
        return doc
    raise RuntimeError("no speech pod could be placed: " + "; ".join(errors))


def handle():
    cur = load()
    if not cur.get("pod_id") or cur.get("deleted"):
        raise RuntimeError("no speech pod in the session file; run `pod up`")
    return pods.attach(cur["pod_id"])


def sh(cmd: str, timeout: float | None = 900, log: Path | None = None) -> int:
    return handle().run(cmd, check=False, timeout=timeout, log=log)


def out(cmd: str, timeout: float = 300) -> str:
    return handle().output(cmd, timeout=timeout)


def status() -> dict:
    cur = load()
    if not cur.get("pod_id"):
        return {"pod": None}
    live = [p for p in pods.list_farm_pods(include_all=True) if p["id"] == cur["pod_id"]]
    age = (time.time() - cur["created"]) / 60 if cur.get("created") else None
    return {"pod": cur["pod_id"], "gpu": cur.get("gpu"), "price_hr": cur.get("price_hr"), "age_min": round(age, 1) if age else None,
            "cost_so_far_usd": round(cur["price_hr"] * age / 60, 3) if age and not cur.get("deleted") else cur.get("cost_usd"),
            "deleted": bool(cur.get("deleted")), "runpod": live[0] if live else None}


def extend(minutes: float = 30) -> int:
    cur = load()
    pod = handle()
    return pods.extend_killswitch(pod, minutes, hard_epoch=cur.get("hard_ceil"))


def down(reason: str = "speech job finished and pulled") -> dict:
    cur = load()
    if not cur.get("pod_id") or cur.get("deleted"):
        return {"deleted": False, "note": "no live speech pod recorded"}
    pods.delete(cur["pod_id"], reason=reason)
    age = (time.time() - cur["created"]) / 3600
    cur.update(deleted=True, deleted_ts=time.time(), cost_usd=round(cur["price_hr"] * age, 3))
    save(cur)
    return {"deleted": True, "pod": cur["pod_id"], "cost_usd": cur["cost_usd"]}
