"""Human QA gates the orchestrator halts at (PIPELINE.md: fit_qa, geoveto): work/<take>/pipeline/gates.json.

When the orchestrator itself runs a gated stage, the gate opens ({opened_at, run, message}); the stage then shows
`blocked` (its outputs are there, your review is not) and nothing downstream of it runs, in that run or in a later
plan, until you confirm it (POST /api/takes/<take>/gates/<stage>; geoveto: decision "apply" runs
`dancers_geoveto --apply` first, "keep" keeps the measured veto file as it is). Stages run by hand outside the
orchestrator open no gate (you looked at them yourself), and takes processed before the gates existed are unaffected.
labels_review keeps its own record (intake.json labels_reviewed_at).
"""
from __future__ import annotations

import time

from . import paths
from .util import read_json, write_json_atomic

DECISIONS = {"fit_qa": ("ok",), "geoveto": ("apply", "keep")}


def path(take: str):
    return paths.pipeline_dir(take) / "gates.json"


def load(take: str) -> dict:
    d = read_json(path(take), {}) or {}
    return d if isinstance(d, dict) else {}


def is_open(take: str, stage: str, doc: dict | None = None) -> dict | None:
    g = (doc if doc is not None else load(take)).get(stage)
    return g if isinstance(g, dict) and g.get("opened_at") and not g.get("confirmed_at") else None


def open_gate(take: str, stage: str, message: str, run: str | None = None) -> dict:
    d = load(take)
    d[stage] = {"opened_at": time.time(), "run": run, "message": message, "confirmed_at": None, "decision": None}
    write_json_atomic(path(take), d)
    return d[stage]


def confirm(take: str, stage: str, decision: str | None = None, run: str | None = None) -> dict:
    d = load(take)
    g = dict(d.get(stage) or {"opened_at": None, "message": None})
    g.update(confirmed_at=time.time(), decision=decision, apply_run=run)
    d[stage] = g
    write_json_atomic(path(take), d)
    return g
