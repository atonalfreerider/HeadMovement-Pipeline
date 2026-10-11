"""One GPU pod for the V2 re-recording work (agent V2, owner label "V2"), controlled across CLI calls with its own state file
(work/<take>/speech/v2v2/pod.json), independent of the speech pod session of other agents.

    python -m dancecap.speech.v2v_pod up [--gpu NAME] [--minutes 150] [--hard-minutes 180]
    python -m dancecap.speech.v2v_pod sh "nvidia-smi"
    python -m dancecap.speech.v2v_pod put LOCAL REMOTE | get REMOTE LOCAL
    python -m dancecap.speech.v2v_pod status | down
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from .. import remote
from ..farm import pods
from . import SPEECH

STATE = SPEECH / "v2v2" / "pod.json"
OWNER = "V2"
GPUS = ["NVIDIA RTX A4500", "NVIDIA L4", "NVIDIA RTX A5000", "NVIDIA GeForce RTX 4090"]


def use_label(owner: str, state_name: str) -> None:
    """work under another owner label and state file (agent V3: "V3", pod_v3.json) without touching V2's record"""
    global OWNER, STATE
    OWNER = owner
    STATE = SPEECH / "v2v2" / state_name


def load() -> dict:
    try:
        return json.loads(STATE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def save(doc: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(doc, indent=1), encoding="utf-8")


def up(gpu: str | None = None, minutes: float = 150, hard_minutes: float = 180, disk_gb: int = 60) -> dict:
    cur = load()
    if cur.get("pod_id") and not cur.get("deleted"):
        raise RuntimeError(f"pod {cur['pod_id']} still recorded; `down` first")
    have = {o["gpu"]: o for o in pods.gpu_offers()}
    errs = []
    for g in ([gpu] if gpu else GPUS):
        o = have.get(g)
        if not o:
            errs.append(f"{g}: no stock")
            continue
        worst = o["price"] * (hard_minutes + pods.START_TIMEOUT_MIN) / 60
        if worst > 2.0:
            errs.append(f"{g}: worst case {worst:.2f} > 2.0")
            continue
        try:
            gate = pods.gate(worst)
            info = pods.create(f"hm-farm-{OWNER.lower()}-{time.strftime('%m%d-%H%M')}", gpu=g, minutes=minutes, hard_minutes=hard_minutes, entry=True,
                               autostart=False, disk_gb=disk_gb, price=o["price"], wait=True, owner=OWNER)
        except Exception as e:  # noqa: BLE001
            errs.append(f"{g}: {str(e)[:200]}")
            continue
        doc = {"pod_id": info["pod_id"], "gpu": g, "price_hr": info.get("cost_per_hr") or o["price"], "created": time.time(), "gate": gate,
               "host": info.get("host"), "port": info.get("port"), "hard_ceil": info.get("hard_ceil")}
        save(doc)
        return doc
    raise RuntimeError("no pod: " + "; ".join(errs))


def handle():
    cur = load()
    if not cur.get("pod_id") or cur.get("deleted"):
        raise RuntimeError("no pod")
    return pods.attach(cur["pod_id"])


def down() -> dict:
    cur = load()
    if not cur.get("pod_id") or cur.get("deleted"):
        return {"deleted": False}
    pods.delete(cur["pod_id"], reason=f"{OWNER} work finished")
    cur.update(deleted=True, cost_usd=round(cur["price_hr"] * (time.time() - cur["created"]) / 3600, 3))
    save(cur)
    return {"deleted": True, "cost_usd": cur["cost_usd"]}


def main(argv=None) -> int:
    a = list(sys.argv[1:] if argv is None else argv)
    cmd = a.pop(0)
    if cmd == "up":
        kw = {}
        while a:
            k = a.pop(0)
            v = a.pop(0)
            kw[k.lstrip("-").replace("-", "_")] = v if k == "--gpu" else float(v)
        print(json.dumps(up(**kw)))
    elif cmd == "sh":
        return handle().run(a[0], check=False, timeout=float(a[1]) if len(a) > 1 else 1800)
    elif cmd == "put":
        handle().put(Path(a[0]), a[1], timeout=1800)
    elif cmd == "get":
        handle().get(a[0], Path(a[1]), timeout=1800)
    elif cmd == "status":
        cur = load()
        print(json.dumps({**cur, "age_min": round((time.time() - cur.get("created", time.time())) / 60, 1)}))
    elif cmd == "down":
        print(json.dumps(down()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
