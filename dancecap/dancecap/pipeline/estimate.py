"""Workload, stage estimates (p50 / p90 minutes and pod dollars) and the Runpod budget.

Model (stages_draft conventions.est): wall_min = fixed + per_CM*CM + per_V*V + per_cam*C + per_view*views +
per_image*images. Per-camera pod work (stages with `concurrency`, per_CM) is summed task time: wall = work / lanes.
- p50 uses the declared camera types; p90 prices every unconfirmed camera as handheld + zoom (PIPELINE_PROGRESS 9.1)
  and adds the known failure risk (`risk_min`, e.g. the MAMMA seam-gate fallback).
- The band widens with the basis: measured x1.3, derived x1.6, estimate x2.0 at p90.
- Live: once a progress fraction exists, the remaining time blends the live rate with the model in log space,
  w = x / (x + 0.05) (PIPELINE_PROGRESS 6.2, simplified).
Budget: spent = remote.spent() (the ledger: one source of truth); running pods add price x ETA (p50/p90) and the hard
worst case up to their kill switch; the launch gate is S + sum(worst of running pods) + p90(new) <= cap.
"""
from __future__ import annotations

import math
import os
import re
import time
from pathlib import Path

from . import camtypes, paths, takes
from .util import read_json, read_text

UNITS_DEFAULT = {"V": 0.64, "C": 8, "CM": 5.76}
P90_FACTOR = {"measured": 1.3, "derived": 1.6, "estimate": 2.0}
UPLOAD_MBPS = {"p50": 2.5, "p90": 0.7}
BOOTSTRAP_MIN = {"p50": 7.0, "p90": 20.0}
POD_START_MIN = 0.6
DOWNLOAD_MIN = 3.0
DEFAULT_PRICE = {"COMMUNITY": 1.19, "SECURE": 1.59}
A40_SECURE = 0.49
RESERVE_DEFAULT = 2.0
_FRAME_COUNT = re.compile(r'"frame_count"\s*:\s*(\d+)')
_fc_cache: dict[str, tuple[float, int]] = {}


def frame_count(path: Path) -> int | None:
    """frames/<cam>/frames.json frame_count, read from the file head only (the file holds every frame)."""
    try:
        st = path.stat()
    except OSError:
        return None
    key = str(path)
    c = _fc_cache.get(key)
    if c and c[0] == st.st_mtime:
        return c[1]
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            head = fh.read(2048)
    except OSError:
        return None
    m = _FRAME_COUNT.search(head)
    if not m:
        return None
    n = int(m.group(1))
    _fc_cache[key] = (st.st_mtime, n)
    return n


def workload(take: str, cfg: dict) -> dict:
    vids = [v for v in takes.videos(take, cfg) if not v["excluded"]]
    probe = (read_json(paths.take_work(take) / "probe" / "probe.json", {}) or {}).get("cameras", {})
    win = takes.window(cfg)
    durs = [probe.get(v["cam"], {}).get("duration_s") for v in vids]
    durs = [d for d in durs if d]
    if win:
        V, basis = (win[1] - win[0]) / 60.0, "window"
    elif durs:
        V, basis = min(durs) / 60.0, "shortest video (no window yet)"
    else:
        V, basis = UNITS_DEFAULT["V"], "default (unknown durations)"
    C = len(vids) or 1
    CM, per_cam = 0.0, {}
    for v in vids:
        n = frame_count(paths.take_work(take) / "frames" / v["cam"] / "frames.json")
        if n:
            cm = n / 1800.0
        else:
            fps = probe.get(v["cam"], {}).get("fps") or 30.0
            cm = V * fps / 30.0
        per_cam[v["cam"]] = cm
        CM += cm
    span = cfg.get("keyframe_span") or (list(win) if win else None)
    step = float(cfg.get("keyframe_step", 2.0) or 2.0)
    images = C * max(3, int((span[1] - span[0]) / step) if span else 10)
    views = C * (V * 60.0) / 0.1333
    return {"V": round(V, 4), "C": C, "CM": round(CM, 3), "per_cam_CM": per_cam, "views": round(views),
            "images": images, "basis": basis}


def _price(stage: dict, prices: dict | None = None) -> float:
    prices = prices or {}
    if stage.get("runs_on") == "pod_secure":
        return float(prices.get("pod_secure", A40_SECURE))
    if "pod" not in prices and stage.get("session"):  # a session priced at the GPU its runner picks (S_splats)
        from . import registry
        sp = (registry.POD_SESSIONS.get(str(stage["session"])) or {}).get("price_per_hr")
        if sp:
            return float(sp)
    return float(prices.get("pod", DEFAULT_PRICE["COMMUNITY"]))


def _lanes(est: dict) -> int:
    c = est.get("concurrency")
    if c is None:
        return 1
    m = re.match(r"\s*(\d+)", str(c))
    return max(1, int(m.group(1))) if m else 1


def stage_estimate(stage: dict, wl: dict, types: list[dict] | None = None, prices: dict | None = None) -> dict:
    """{p50_s, p90_s, pod_min_p50, pod_min_p90, cost_p50, cost_p90, billed, confidence, basis, branches?}."""
    est = stage.get("est") or {}
    wm = est.get("wall_min") or {}
    V, C, CM = wl["V"], wl["C"], wl["CM"]
    lanes = _lanes(est)
    cm_part = float(wm.get("per_CM", 0)) * CM
    cm_part90 = cm_part
    if stage["id"] == "cam_loc" and types:
        per = wl.get("per_cam_CM") or {}
        tot = sum(per.values()) or 1.0
        f50 = sum(per.get(t["cam"], tot / len(types)) * camtypes.loc_factor(t) for t in types) / tot
        f90 = sum(per.get(t["cam"], tot / len(types)) * camtypes.loc_factor(t, conservative=True) for t in types) / tot
        cm_part, cm_part90 = cm_part * f50, cm_part * f90
    if lanes > 1:
        cm_part, cm_part90 = cm_part / lanes, cm_part90 / lanes
    base = (float(wm.get("fixed", 0)) + float(wm.get("per_V", 0)) * V + float(wm.get("per_cam", 0)) * C
            + float(wm.get("per_view", 0)) * wl["views"] + float(wm.get("per_image", 0)) * wl["images"])
    p50 = base + cm_part
    conf = est.get("confidence", "estimate")
    risk = est.get("risk_min") or {}
    risk_min = float(risk.get("fixed", 0)) + float(risk.get("per_V", 0)) * V
    p90 = (base + cm_part90) * P90_FACTOR.get(conf, 2.0) + risk_min
    billed = bool(est.get("billed")) or stage.get("runs_on") in ("pod", "pod_secure")
    price = _price(stage, prices)
    out = {"p50_s": round(p50 * 60), "p90_s": round(p90 * 60), "confidence": conf,
           "basis": est.get("basis") or f"model ({conf})", "billed": billed, "host": est.get("host", stage.get("runs_on")),
           "price_per_hr": price if billed else 0.0,
           "cost_p50_usd": round(price * p50 / 60, 2) if billed else 0.0,
           "cost_p90_usd": round(price * p90 / 60, 2) if billed else 0.0}
    if risk_min:
        out["branches"] = [{"name": "pass", "eta_p50_s": round(p50 * 60)},
                           {"name": risk.get("when", "fallback"), "eta_p50_s": round((p50 + risk_min) * 60)}]
    return out


def session_overhead(upload_mb: float) -> dict:
    up50 = upload_mb / UPLOAD_MBPS["p50"] / 60
    up90 = upload_mb / UPLOAD_MBPS["p90"] / 60
    return {"upload_mb": round(upload_mb), "p50_min": round(POD_START_MIN + up50 + BOOTSTRAP_MIN["p50"] + DOWNLOAD_MIN, 1),
            "p90_min": round(POD_START_MIN + up90 + BOOTSTRAP_MIN["p90"] + DOWNLOAD_MIN * 2, 1),
            "upload_p50_min": round(up50, 1), "upload_p90_min": round(up90, 1)}


def live_blend(p50_s: float, p90_s: float, frac: float | None, elapsed_s: float | None) -> dict:
    """Remaining time of a running step: model x (1 - x) blended with the live rate (log space)."""
    if frac is None or elapsed_s is None or frac <= 0.0:
        rem50 = max(0.0, p50_s - (elapsed_s or 0))
        rem90 = max(rem50, p90_s - (elapsed_s or 0))
        return {"p50_s": round(rem50), "p90_s": round(rem90), "p10_s": round(rem50 * 0.6), "basis": "model",
                "confidence": "estimate"}
    frac = min(max(frac, 1e-3), 0.999)
    live = elapsed_s * (1 - frac) / frac
    model = p50_s * (1 - frac)
    w = frac / (frac + 0.05)
    r50 = math.exp(w * math.log(max(live, 1.0)) + (1 - w) * math.log(max(model, 1.0)))
    sigma = (1 - w) * 0.4 + w * 0.15
    return {"p10_s": round(r50 * math.exp(-1.28 * sigma)), "p50_s": round(r50), "p90_s": round(r50 * math.exp(1.28 * sigma)),
            "basis": f"live {frac:.0%} + model", "confidence": "live"}


# ------------------------------------------------------------------------------------------------ budget

def ledger_rows() -> list[dict]:
    rows = read_json(paths.ledger_path(), []) or []
    return rows if isinstance(rows, list) else []


def cap() -> float:
    """The Runpod spend cap: remote.cap() on the real tree (DANCECAP_RUNPOD_CAP or remote.DEFAULT_CAP, $80 since
    2026-10-07), so the plan shows the cap the budget gate and the pod runner use; a test tree keeps $30."""
    if paths.is_real_root():
        try:
            from .. import remote
            return float(remote.cap())
        except Exception:  # noqa: BLE001
            pass
    try:
        return float(os.environ.get("DANCECAP_RUNPOD_CAP", 30.0))
    except ValueError:
        return 30.0


def reserve() -> float:
    try:
        return float(os.environ.get("DANCECAP_RUNPOD_RESERVE", RESERVE_DEFAULT))
    except ValueError:
        return RESERVE_DEFAULT


def spent(rows: list[dict] | None = None, now: float | None = None) -> float:
    """Same rule as remote.spent(); when the backend runs on the real tree, remote.spent() itself is used."""
    if rows is None and paths.is_real_root():
        try:
            from .. import remote
            return float(remote.spent())
        except Exception:
            pass
    now = now or time.time()
    total = 0.0
    for r in rows if rows is not None else ledger_rows():
        try:
            if r.get("deleted_ts") and r.get("cost") is not None:
                total += float(r["cost"])
                continue
            end = r.get("deleted_ts") or now
            total += float(r["cost_per_hr"]) * max(0.0, end - float(r["created_ts"])) / 3600
        except (KeyError, TypeError, ValueError):
            continue
    return total


def open_rows(rows: list[dict] | None = None) -> list[dict]:
    return [r for r in (rows if rows is not None else ledger_rows()) if not r.get("deleted_ts")]


def _progress_eta():
    if not paths.is_real_root():
        return None
    try:
        from .. import progress_eta  # type: ignore[attr-defined]
        return progress_eta
    except Exception:
        return None


def _budget_from_progress(pe, queued: list[dict] | None, untracked_per_hr: float, now: float | None) -> dict:
    """progress_eta.budget() (PIPELINE_PROGRESS 8) mapped onto the monitor's Budget shape (PIPELINE_MONITOR 7.3)."""
    q = [{"p50_cost": float(x.get("cost_usd", {}).get("p50", 0)), "p90_cost": float(x.get("cost_usd", {}).get("p90", 0)),
          "label": x.get("id")} for x in queued or []]
    b = pe.budget(now=now, queued=q)
    pods = b.get("open_pods") or []
    msgs = [w.get("msg") if isinstance(w, dict) else str(w) for w in b.get("warnings") or []]
    level = b.get("level", "green")
    if untracked_per_hr > 0.05:
        level = "red"
        msgs.append(f"untracked spend ${untracked_per_hr:.2f}/h (a pod the ledger does not know)")
    phantom = [p for p in pods if p.get("phantom_cost") is not None]
    corrected = None
    if phantom:
        corrected = round(b["spent"] - sum(p["so_far"] - p["phantom_cost"] for p in phantom), 4)
    return {"cap": b["cap"], "reserve": b["reserve"], "spent": b["spent"], "remaining": b["remaining"],
            "open_accrual_usd": round(sum(p.get("so_far", 0) for p in pods), 4),
            "running": {"p50_usd": round(sum(p.get("remaining_p50", 0) for p in pods), 2),
                        "p90_usd": round(sum(p.get("remaining_p90", 0) for p in pods), 2),
                        "worst_usd": round(sum(p.get("worst_remaining", 0) for p in pods), 2)},
            "queued": {"p50_usd": round(sum(x["p50_cost"] for x in q), 2), "p90_usd": round(sum(x["p90_cost"] for x in q), 2)},
            "projected": {"t50": round(b["projected"]["p50"], 2), "t90": round(b["projected"]["p90"], 2),
                          "worst": round(b["worst"], 2)},
            "level": level, "messages": msgs, "untracked_spend_per_hr_usd": round(untracked_per_hr, 2),
            "corrected_spent": corrected, "open_pods": len(pods), "pods": pods,
            "phase_costs": b.get("phase_costs"), "source": "progress_eta"}


def budget(running_eta: dict[str, dict] | None = None, queued: list[dict] | None = None,
           untracked_per_hr: float = 0.0, now: float | None = None) -> dict:
    """Budget (PIPELINE_MONITOR 7.3 / PIPELINE_PROGRESS 8.1-8.2). running_eta: {pod_id: {p50_s, p90_s}}.
    Delegates to dancecap.progress_eta.budget() when available (same ledger, live session ETAs)."""
    pe = _progress_eta()
    if pe is not None and running_eta is None:
        try:
            return _budget_from_progress(pe, queued, untracked_per_hr, now)
        except Exception:
            pass
    now = now or time.time()
    rows = ledger_rows()
    S = spent()
    c, res = cap(), reserve()
    run50 = run90 = worst = accr = 0.0
    msgs = []
    for r in open_rows(rows):
        p = float(r.get("cost_per_hr") or 0)
        accr += p * max(0.0, now - float(r.get("created_ts") or now)) / 3600
        kill_at = float(r.get("created_ts") or now) + float(r.get("max_minutes") or 0) * 60
        eta = (running_eta or {}).get(r.get("pod_id"), {})
        left = max(0.0, kill_at - now)
        run50 += p * min(left, float(eta.get("p50_s", left))) / 3600
        run90 += p * min(left, float(eta.get("p90_s", left))) / 3600
        worst += p * left / 3600
    q50 = sum(float(q.get("cost_usd", {}).get("p50", 0)) for q in queued or [])
    q90 = sum(float(q.get("cost_usd", {}).get("p90", 0)) for q in queued or [])
    T50, T90, Wc = S + run50 + q50, S + run90 + q90, S + worst + q90
    level = "green"
    if T90 > c - res:
        level = "amber"
        msgs.append(f"p90 ${T90:.2f} is over cap - reserve (${c - res:.2f})")
    if c - S < DEFAULT_PRICE["COMMUNITY"]:
        level = "amber"
        msgs.append(f"less than one A100-hour left (${c - S:.2f})")
    if T50 > c or Wc > c:
        level = "red"
        msgs.append(f"the worst case ${Wc:.2f} is over the cap ${c:.2f}" if Wc > c else f"p50 ${T50:.2f} is over the cap")
    if untracked_per_hr > 0.05:
        level = "red"
        msgs.append(f"untracked spend ${untracked_per_hr:.2f}/h (a pod the ledger does not know)")
    return {"cap": c, "reserve": res, "spent": round(S, 4), "remaining": round(c - S, 4),
            "open_accrual_usd": round(accr, 4),
            "running": {"p50_usd": round(run50, 2), "p90_usd": round(run90, 2), "worst_usd": round(worst, 2)},
            "queued": {"p50_usd": round(q50, 2), "p90_usd": round(q90, 2)},
            "projected": {"t50": round(T50, 2), "t90": round(T90, 2), "worst": round(Wc, 2)},
            "level": level, "messages": msgs, "untracked_spend_per_hr_usd": round(untracked_per_hr, 2),
            "corrected_spent": None, "open_pods": len(open_rows(rows)), "source": "ledger"}


def launch_gate(new_p90_usd: float, now: float | None = None) -> dict:
    """PIPELINE_PROGRESS 8.3: S + sum(worst of running pods) + p90(new) <= cap; over cap - reserve needs a flag."""
    b = budget(now=now)
    total = b["spent"] + b["running"]["worst_usd"] + new_p90_usd
    gate = "ok" if total <= b["cap"] - b["reserve"] else "over_reserve" if total <= b["cap"] else "over_cap"
    return {"spent": b["spent"], "running_worst": b["running"]["worst_usd"], "after_p50": None,
            "after_p90": round(total, 2), "cap": b["cap"], "reserve": b["reserve"], "gate": gate}


def read_log_tail(path: Path, max_bytes: int = 16384) -> str:
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - max_bytes))
            return fh.read().decode("utf-8", errors="replace")
    except OSError:
        return read_text(path, "") or ""
