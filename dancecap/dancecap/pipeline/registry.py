"""The stage registry: the DAG of the capture pipeline per capture type (multi / two / single view).

Data comes from `stages_draft.json` (research stage 'dag', mirrored in docs/PIPELINE.md: ids, deps, where each stage
runs, outputs, done markers, estimate models, gates, camera-type behaviour). This module owns the final registry: it
adds what the backend needs to *run* and *judge* a stage, in one Python overlay:

- `commands`: the allowlisted argv templates of a local stage (the client never sends an argv; values are validated
  before substitution). Pod stages run through `pod` session specs (remote_jobs batched sessions, created only by an
  approved plan). Human stages are gates.
- `implemented`: False for stages that are proposed but not built yet. They are registered and shown with status
  `planned` (never faked); when their outputs exist anyway the status says so.
- `status_rule`: the few stages whose 'done' is not "every done_when file exists" (the take TOML, alignment, labels).
- `heavy`: local stages that must run alone on the 4-CPU workstation.
- `keep_existing`: stages whose existing outputs a plan never rebuilds without an explicit opt-in (cam_map: the static
  SfM map defines the world frame; a take may reuse another take's map on purpose).
- `gate_human`: stages after which the orchestrator halts for the user's review (gates.py).
- `helps`: why the order is what it is (edge tooltips in the monitor).

Paths in `done_when` are relative to work/<take>/ unless they start with takes/, taxonomy/, work/, prototype/ or
head-movement/ (stages_draft conventions.paths).
"""
from __future__ import annotations

import copy
import glob
import json
import os
import re
from functools import lru_cache
from pathlib import Path

from . import paths

DRAFT = Path(__file__).resolve().parent / "stages_draft.json"
SCHEMA = "dancecap.pipeline.stages/1"
VARIANTS = ("multi", "two", "single")
COLUMNS = ["0 intake+clocks", "1 frames/audio/2D", "2 cameras", "3 body", "4 detail", "5 avatar", "6 analysis",
           "7 output"]
STATUSES = ("done", "stale", "ready", "waiting", "blocked", "queued", "running", "failed", "stopped", "skipped",
            "planned", "n/a")

M = ["-m"]
DC = ["-m", "dancecap"]

# --------------------------------------------------------------------------------------------- overlay

OVERLAY: dict[str, dict] = {
    "intake": {"implemented": True, "status_rule": "intake", "ui": "#/new"},
    "probe": {"implemented": True, "commands": [[*M, "dancecap.pipeline", "probe", "{take}"]], "ui_helper": True},
    "camera_types": {"implemented": True, "status_rule": "camera_types", "ui": "#/take/{take}/videos"},
    "audio_sync": {"implemented": True, "status_rule": "alignment",
                   "commands": [[*M, "dancecap.pipeline", "audio-sync", "{take}"]],
                   "notes_extra": "port of prototype/sync_larissa_kadu.py (same matcher and thresholds); writes the "
                                  "file named by the take's `alignment` (never the prototype / studio clocks)"},
    "song_align": {"implemented": True, "optional_rule": "soundtrack",
                   "commands": [[*M, "dancecap.song", "align", "--take", "{take}"],
                                [*M, "dancecap.song", "soundtrack", "--take", "{take}"],
                                [*M, "dancecap.song", "compare", "--take", "{take}"]]},
    "clock_studio": {"implemented": True, "optional_rule": "soundtrack",
                     "commands": [[*M, "dancecap.clock_studio", "build", "--base", "{take}", "--out",
                                   "work/{take}/clock/alignment_studio.json"],
                                  [*M, "dancecap.clock_studio", "validate", "--base", "{take}", "--alignment",
                                   "work/{take}/clock/alignment_studio.json"]],
                     "notes_extra": "switching the take's alignment to the studio clocks is a TOML edit (by hand)"},
    "window": {"implemented": True, "status_rule": "window", "manual": True,
               "manual_hint": "python -m dancecap.clock_studio window --alignment <alignment> --exclude <cams> "
                              "--within <t0> <t1>, then confirm `window` in the take TOML"},
    "ingest": {"implemented": True, "heavy": True, "commands": [[*DC, "ingest", "{take}"]]},
    "beats": {"implemented": True, "commands": [[*DC, "beats", "{take}"]]},
    "song_beats": {"implemented": True, "optional_rule": "soundtrack",
                   "commands": [[*M, "dancecap.song", "beats", "--take", "{take}"]]},
    "detect": {"implemented": True, "commands": [[*DC, "detect", "{take}"]], "alt_runs_on": "pod"},
    "dancers_labels": {"implemented": True, "commands": [[*DC, "dancers-labels", "{take}"]]},
    "labels_review": {"implemented": True, "status_rule": "labels_review", "ui": "#/take/{take}/stage/labels_review"},
    "dancers_sam": {"implemented": True, "pod": "S1_front"},
    "ma2d": {"implemented": True, "pod": "S1_front"},
    "cam_map": {"implemented": True, "heavy": True, "commands": "cam_map", "alt_runs_on": "pod",
                "keep_existing": "the static SfM map defines the world frame (and this take may reuse another take's "
                                 "map on purpose): an existing map is kept unless the plan opts in (rebuild_map)"},
    "cam_loc": {"implemented": True, "pod": "S1_front", "camera_stage": True},
    "cam_world": {"implemented": True, "pod": "S1_front", "camera_stage": True, "alt_runs_on": "local"},
    "sync": {"implemented": True, "pod": "S1_front", "alt_runs_on": "local"},
    "scale_chain": {"implemented": True, "pod": "S1_front"},
    "geoveto": {"implemented": True, "commands": [[*M, "dancecap.dancers_geoveto", "{take}"]],
                "gate_human": "look at review/reid_veto.png, then apply the veto (or keep the file as measured)",
                "gate_apply": [[*M, "dancecap.dancers_geoveto", "{take}", "--apply"]]},
    "mamma_prepare": {"implemented": True,
                      "commands": [[*DC, "mamma", "prepare", "{take}", "--skip-2d", "--set", "{tag}=all"],
                                   [*M, "dancecap.mamma_chunks", "plan", "{take}", "--tag", "{tag}", "--chunks", "6"]]},
    "scale_probe": {"implemented": False, "planned_reason": "the 3 short spread fits + stature rescale are not built "
                                                            "yet (PIPELINE.md 6.1); mamma_fit runs without them"},
    "mamma_fit": {"implemented": True, "pod": "S2_fit"},
    "fit_qa": {"implemented": True,
               "commands": [[*M, "dancecap.mamma_seams", "{take}"],
                            [*M, "dancecap.cameras.world_scale", "check", "{take}", "--strict"],
                            [*DC, "review", "{take}"]],
               "gate_human": "look at the pose / seam sheets under review/ before the detail stages"},
    "cam_refine_body": {"implemented": False, "planned_reason": "human-keypoint camera refinement is not built yet "
                                                                "(PIPELINE.md 6.2)"},
    "mamma_loo": {"implemented": True, "manual": True, "manual_hint": "remote run --jobs mamma_loo (pod_loo.py)"},
    # hands_survey / hands_pod / hands_build (S3_hands: HaMeR, WiLoR, fingers, contact) were retired by the user on
    # 2026-10-07 (stages_draft `retired`): no stage, no pod session, no remote-job mapping. Nothing reads their outputs.
    "face_pod": {"implemented": True, "pod": "S2_fit"},
    "texture": {"implemented": True, "pod": "S2_fit"},
    "avatar_face": {"implemented": True, "optional_rule": "portraits", "commands": [[*M, "dancecap.avatar_face", "{take}"]]},
    "hair_reference": {"implemented": True, "pod": "S2_fit"},
    "hair_groom": {"implemented": True, "commands": [[*M, "dancecap.hair_groom", "{take}"]]},
    "shoe_size": {"implemented": True, "commands": [[*M, "dancecap.shoe_size", "{take}"]]},
    "avatar_edits": {"implemented": True, "status_rule": "optional_human"},
    "timing": {"implemented": True, "commands": "timing"},
    "physics": {"implemented": True, "commands": [[*DC, "physics", "{take}"], [*M, "dancecap.analysis_review", "{take}"]]},
    "counterbalance": {"implemented": True, "commands": [[*M, "dancecap.counterbalance", "detect", "--take", "{take}"]]},
    "floorcraft": {"implemented": True, "commands": [[*M, "dancecap.floorcraft", "{take}"]]},
    "moves_features": {"implemented": True, "commands": [[*M, "dancecap.moves", "features", "{take}"]]},
    "moves_proposals": {"implemented": True, "commands": [[*M, "dancecap.moves", "proposals", "{take}"]]},
    "narrate": {"implemented": True, "status_rule": "narrate", "tool": "narrate", "no_implied": True},
    "moves_run": {"implemented": True, "commands": [[*M, "dancecap.moves", "run", "{take}"]]},
    "classifier": {"implemented": False, "status_rule": "classifier",
                   "planned_reason": "the move classifier waits on approvals G0/G5/G9 (docs/TMR_SOMA_PLAN.md)"},
    "triggers": {"implemented": True, "commands": [[*M, "dancecap.pipeline", "triggers", "{take}"]]},
    "demo_export": {"implemented": True, "status_rule": "demo_export", "needs_capture": True,
                    "commands": [[*M, "dancecap.demo_export", "{take}", "--capture", "{capture}"],
                                 [*M, "dancecap.pipeline", "triggers", "{take}", "--export-capture", "{capture}"]],
                    "unity_lock": True},
    "unity_check": {"implemented": True, "manual": True, "status_rule": "manual",
                    "manual_hint": "head-movement playtests (Unity)"},
    "room": {"implemented": True, "pod": "S_room"},
    # avatar-guided dancer splats (dancecap.dancer_splats, approved 2026-10-08 as a prototype; docs/DANCER_SPLATS_
    # RESEARCH.md): its own pod session, a review gate on the held-out sheets before any Unity export
    "dancer_splats": {"implemented": True, "pod": "S_splats",
                      "gate_human": "look at review/splats/dancer_splats_*.jpg (held-out real | splat | avatar) and the "
                                    "gates in dancer_splats/metrics.json before exporting to Unity"},
    "pair_calib": {"implemented": False, "camera_stage": True,
                   "planned_reason": "two-view calibration from the dancers (python -m dancecap.pipeline pair-calib) "
                                     "is not built yet"},
    "mono_init": {"implemented": False, "planned_reason": "PromptHMR-Video seeding (Linux pod) is not built yet; "
                                                         "licence to check before any download"},
    "mono_camera": {"implemented": False, "camera_stage": True,
                    "planned_reason": "single-view camera (static / MegaSaM / VGGT-Omega / MapAnything by camera "
                                      "type) is not built yet"},
    "mono_hmr": {"implemented": False, "planned_reason": "monocular couple HMR (PromptHMR-Video / GVHMR / GEM-X) is "
                                                         "not built yet; licences to check"},
    "couple_refine": {"implemented": False, "planned_reason": "single-view couple refinement (statures, floor, "
                                                              "contact, non-penetration) is not built yet"},
}

# Pod sessions: what to prepare locally (write the pod job), which remote_jobs job each stage maps to, and how the
# session is launched. A session is only ever started by an approved plan (never by the monitor on its own).
POD_SESSIONS: dict[str, dict] = {
    "S1_front": {"cloud": "COMMUNITY", "gpu_hint": "NVIDIA A100 80GB PCIe", "max_cost": 5.0,
                 "prep": [[*M, "dancecap.front_pod", "write-job", "{take}"]],
                 "jobs": {"*": "work/{take}/front/pod_job.json"}, "upload_mb_per_cm": 730,
                 "remote_names": ["front"]},
    "S2_fit": {"cloud": "COMMUNITY", "gpu_hint": "NVIDIA A100 80GB PCIe", "max_cost": 5.0,
               "prep": {"texture": [[*DC, "texture", "{take}", "--write-pod-job"]],
                        "hair_reference": [[*M, "dancecap.hair_reference", "{take}", "--write-pod-job"]],
                        "face_pod": [[*M, "dancecap.face_pod", "spec", "{take}", "--write"]]},
               "jobs": {"mamma_fit": "work/{take}/mamma_chunks/pod_job.json", "texture": "avatar",
                        "hair_reference": "hair_ref", "face_pod": "face"}, "upload_mb_per_cm": 260,
               "remote_names": ["mamma", "mamma_chunks", "avatar", "avatar_fix", "texture", "hair_ref", "face"]},
    "S_room": {"cloud": "COMMUNITY", "gpu_hint": "NVIDIA A100 80GB PCIe", "max_cost": 3.0, "optional": True,
               "prep": [[*DC, "room", "{take}", "--write-pod-job"]], "jobs": {"room": "room"},
               "upload_mb_per_cm": 150, "remote_names": ["room"]},
    # dancer splats: the job bounds itself (training inside its time budget and 15 min before the kill switch), so the
    # session runs at the user-approved ceiling (fixed_budget: max_cost $ and max_minutes, 2026-10-08) and uploads a
    # fixed-size bundle (one short window, whatever the take's length: upload_mb, not per camera-minute)
    # (priced at the A100 SXM community rate the runner picks first: $1.39/h x (55 + 6) min = $1.41 worst case)
    "S_splats": {"cloud": "COMMUNITY", "gpu_hint": "NVIDIA A100-SXM4-80GB", "price_per_hr": 1.39, "max_cost": 1.5,
                 "max_minutes": 55,
                 "fixed_budget": True, "optional": True,
                 "prep": [[*M, "dancecap.dancer_splats", "{take}", "prepare"],
                          [*M, "dancecap.dancer_splats", "{take}", "write-pod-job"]],
                 "jobs": {"dancer_splats": "dancer_splats"}, "upload_mb": 260, "remote_names": ["dancer_splats"]},
}

# remote_jobs job name (or prefix) -> stages it advances (for sessions the monitor did not start)
REMOTE_JOB_STAGES = [
    ("front", ["dancers_sam", "ma2d", "cam_loc", "cam_world", "sync", "scale_chain"]),
    ("mamma_loo", ["mamma_loo"]), ("mamma", ["mamma_fit"]), ("avatar", ["texture"]), ("texture", ["texture"]),
    ("hair_ref", ["hair_reference"]), ("face", ["face_pod"]), ("room", ["room"]),
    ("dancer_splats", ["dancer_splats"]),
]

# why the order (monitor edge tooltips; PIPELINE.md 1 and 6)
HELPS = {
    ("intake", "probe"): "the probe reads the registered videos and pre-fills the camera types",
    ("probe", "camera_types"): "probe A suggests mount + zoom per video; you confirm",
    ("audio_sync", "window"): "every frame carries reference time: the clocks come first, a later clock change "
                              "invalidates everything",
    ("song_align", "clock_studio"): "the studio track gives +-1-2 ms phone clocks",
    ("window", "ingest"): "ingest decodes only the frames inside the window",
    ("detect", "cam_map"): "people masks before any static-room geometry (moving people break SfM)",
    ("camera_types", "cam_map"): "types are settled before any camera work starts (the static map itself does not "
                                 "read them today: a type change never rebuilds it)",
    ("camera_types", "cam_loc"): "a locked tripod = one pose per video (keyframes only, ~10 % of the work); no zoom = "
                                 "one focal + k1 per video; a type the data refutes is refused (safer model kept)",
    ("cam_map", "cam_loc"): "cameras from the static room before any body: many precise, time-independent points",
    ("labels_review", "dancers_sam"): "a swapped lead/follow label wastes the whole front session",
    ("cam_world", "sync"): "sync triangulates the dancers from calibrated cameras",
    ("ma2d", "sync"): "sync uses the MAMMA 2D landmarks of the other cameras",
    ("cam_world", "mamma_prepare"): "cameras are fit inputs (a tripod's constant pose adds no per-frame camera "
                                    "noise; tripods do not anchor the world frame yet: PIPELINE.md 4.4)",
    ("sync", "mamma_prepare"): "bodies need video sync: at 2 m/s, 50 ms is 10 cm",
    ("mamma_prepare", "scale_probe"): "short spread fits check the scale against the statures before the full fit",
    ("mamma_fit", "fit_qa"): "the seam gate and stature check decide the detail stages and the camera refinement",
    ("fit_qa", "cam_refine_body"): "bodies refine only the weak cameras (attached, zoomed, two-view)",
    ("fit_qa", "texture"): "detail stages read the final smplx/: they run after the last refit",
    ("counterbalance", "triggers"): "classifier intervals switch viewer layers on during playback",
    ("triggers", "demo_export"): "the export copies triggers.json into the capture (capture time)",
    ("window", "narrate"): "narration needs only the videos, the clock and the beat grid: start it early",
    ("fit_qa", "dancer_splats"): "every Gaussian is bound to the fitted SMPL-X; the splats only add small per-frame "
                                 "corrections (pose_delta.npz)",
    ("texture", "dancer_splats"): "the splats start from the avatar's albedo and are judged against the textured "
                                  "avatar on the held-out camera",
    ("hair_groom", "dancer_splats"): "hair Gaussians start on the groom's 160 guide strands",
}


# --------------------------------------------------------------------------------------------- loading

@lru_cache(maxsize=1)
def _draft(mtime_ns: int) -> dict:
    return json.loads(DRAFT.read_text(encoding="utf-8"))


def draft() -> dict:
    return _draft(DRAFT.stat().st_mtime_ns)


def column_of(phase: str) -> int:
    m = re.match(r"(\d)", phase or "")
    return int(m.group(1)) if m else 0


@lru_cache(maxsize=4)
def _registry(mtime_ns: int) -> dict:
    d = copy.deepcopy(_draft(mtime_ns))
    stages = []
    for s in d["stages"]:
        ov = OVERLAY.get(s["id"], {})
        st = dict(s)
        for k, v in ov.items():
            if k in ("commands",):
                continue
            st[k] = v
        st.setdefault("implemented", s.get("exists") in ("yes", "partial", "in_progress"))
        st["column"] = column_of(s.get("phase", ""))
        st["soft_deps"] = list(s.get("soft_deps") or [])
        st["helps"] = [{"stage": b, "why": w} for (a, b), w in HELPS.items() if a == s["id"]]
        cmds = ov.get("commands")
        st["runnable"] = ("local" if cmds else "session" if ov.get("pod") else "human"
                          if s.get("runs_on") == "human" else "manual" if ov.get("manual") else None)
        if not st["implemented"]:
            st["runnable"] = None
        if cmds == "timing":
            st["commands_display"] = [["python", "-m", "dancecap", "timing", "{take}", "[--beats work/{take}/song/"
                                       "beats_studio.json --no-refine]"]]
        elif cmds == "cam_map":
            st["commands_display"] = [["python", "-m", "dancecap", "cameras", "{take}", "--stages", "map"],
                                      ["python", "-m", "dancecap.cameras.reuse", "map", "{take}", "--from", "<base>",
                                       "(when this take reuses another take's map)"]]
        elif cmds:
            st["commands_display"] = [["python", *c] for c in cmds]
        else:
            st["commands_display"] = []
        st["heavy"] = bool(ov.get("heavy"))
        st["stop_modes"] = ["now"] if st["runnable"] == "local" else []
        stages.append(st)
    d["stages"] = stages
    d["schema"] = SCHEMA
    d["pod_sessions"] = {k: {kk: vv for kk, vv in v.items() if kk not in ("prep", "runner")}
                         for k, v in POD_SESSIONS.items()}
    return d


def registry() -> dict:
    return _registry(DRAFT.stat().st_mtime_ns)


def stage_map() -> dict[str, dict]:
    return {s["id"]: s for s in registry()["stages"]}


def get(stage_id: str) -> dict | None:
    return stage_map().get(stage_id)


# --------------------------------------------------------------------------------------------- variants / DAG

def _replacements(variant: str) -> dict[str, list[str]]:
    rep = (registry().get("variants", {}).get(variant, {}) or {}).get("replaces", {}) or {}
    ids = set(stage_map())
    out = {}
    for k, v in rep.items():
        if "." in k:
            continue
        if v is None:
            out[k] = []
            continue
        toks = [t for t in re.split(r"[\s+,()]+", str(v)) if t in ids]
        out[k] = toks
    return out


def in_variant(stage: dict, variant: str) -> bool:
    return variant in (stage.get("variants") or VARIANTS)


def variant_stages(variant: str) -> list[dict]:
    """Stages of a variant in topological order, with deps resolved for that variant (replaced deps substituted,
    deps outside the variant dropped)."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    rep = _replacements(variant)
    sm = stage_map()
    keep = {sid: s for sid, s in sm.items() if in_variant(s, variant)}

    def resolve(deps):
        out = []
        for d in deps or []:
            if d in keep:
                cand = [d]
            else:
                cand = [r for r in rep.get(d, []) if r in keep]
            for c in cand:
                if c not in out:
                    out.append(c)
        return out

    staged = {}
    for sid, s in keep.items():
        st = dict(s)
        st["deps"] = resolve(s.get("deps"))
        st["soft_deps"] = [d for d in resolve(s.get("soft_deps")) if d not in st["deps"]]
        staged[sid] = st
    return topo_order(list(staged.values()))


def topo_order(stages: list[dict]) -> list[dict]:
    """Kahn's algorithm; among ready stages the lowest (column, draft index) goes first."""
    idx = {s["id"]: i for i, s in enumerate(registry()["stages"])}
    by = {s["id"]: s for s in stages}
    indeg = {sid: sum(1 for d in s["deps"] if d in by) for sid, s in by.items()}
    out, ready = [], sorted([sid for sid, n in indeg.items() if n == 0], key=lambda x: (by[x]["column"], idx.get(x, 999)))
    children: dict[str, list[str]] = {sid: [] for sid in by}
    for sid, s in by.items():
        for d in s["deps"]:
            if d in by:
                children[d].append(sid)
    while ready:
        sid = ready.pop(0)
        out.append(by[sid])
        for c in children[sid]:
            indeg[c] -= 1
            if indeg[c] == 0:
                ready.append(c)
        ready.sort(key=lambda x: (by[x]["column"], idx.get(x, 999)))
    if len(out) != len(by):
        raise ValueError(f"cycle in the stage graph: {sorted(set(by) - {s['id'] for s in out})}")
    return out


def edges(stages: list[dict]) -> list[list[str]]:
    ids = {s["id"] for s in stages}
    out = []
    for s in stages:
        for d in s["deps"]:
            out.append([d, s["id"], "dep"])
        for d in s.get("soft_deps", []):
            if d in ids:
                out.append([d, s["id"], "soft"])
    for lp in registry().get("loops", []):
        if lp.get("from") in ids and lp.get("to") in ids:
            out.append([lp["from"], lp["to"], lp["id"]])
    return out


def descendants(stages: list[dict], roots: list[str]) -> list[str]:
    children: dict[str, list[str]] = {}
    for s in stages:
        for d in s["deps"]:
            children.setdefault(d, []).append(s["id"])
    seen, stack = [], list(roots)
    while stack:
        for c in children.get(stack.pop(), []):
            if c not in seen and c not in roots:
                seen.append(c)
                stack.append(c)
    order = [s["id"] for s in stages]
    return sorted(seen, key=order.index)


def ancestors(stages: list[dict], target: str) -> list[str]:
    by = {s["id"]: s for s in stages}
    seen, stack = [], [target]
    while stack:
        for d in by.get(stack.pop(), {}).get("deps", []):
            if d not in seen:
                seen.append(d)
                stack.append(d)
    order = [s["id"] for s in stages]
    return sorted(seen, key=order.index)


def variant_for(n_used: int) -> str:
    return "single" if n_used <= 1 else "two" if n_used == 2 else "multi"


# --------------------------------------------------------------------------------------------- paths

_ROOT_PREFIXES = ("takes/", "taxonomy/", "work/")
_WS_PREFIXES = ("prototype/", "head-movement/")


def resolve_path(take: str, rel: str) -> Path:
    rel = rel.replace("\\", "/").strip()
    rel = rel.replace("{take}", take)
    if rel.startswith(_ROOT_PREFIXES):
        return paths.root() / rel
    if rel.startswith(_WS_PREFIXES):
        return paths.workspace() / rel
    return paths.take_work(take) / rel


def done_patterns(stage: dict, take: str) -> list[str]:
    out = []
    for p in stage.get("done_when") or []:
        p = re.sub(r"\{(capture|cam|tag|sid)\}", "*", p)
        out.append(str(resolve_path(take, p)))
    return out


def pattern_files(pattern: str) -> list[Path]:
    if any(ch in pattern for ch in "*?["):
        return [Path(p) for p in glob.glob(pattern)]
    p = Path(pattern)
    return [p] if p.exists() else []


# --------------------------------------------------------------------------------------------- commands

class NotRunnable(RuntimeError):
    pass


def _subst(argv: list[str], ctx: dict) -> list[str]:
    out = []
    for a in argv:
        for k in ("take", "capture", "tag", "max_cost"):
            if "{" + k + "}" in a:
                v = ctx.get(k)
                if v is None or v == "":
                    raise NotRunnable(f"needs a value for {k}")
                a = a.replace("{" + k + "}", str(v))
        if re.search(r"\{[a-z_]+\}", a):
            raise NotRunnable(f"unresolved placeholder in {a!r}")
        out.append(a)
    return out


def reused_map_base(take: str) -> str | None:
    """The take whose static SfM map this take reuses (cameras/sfm_work/map_images.json `reused_from`), or None."""
    from .util import TAKE_RE, read_json
    d = read_json(paths.take_work(take) / "cameras" / "sfm_work" / "map_images.json", None)
    base = (d or {}).get("reused_from") if isinstance(d, dict) else None
    return base if isinstance(base, str) and TAKE_RE.match(base) and base != take else None


def commands(stage_id: str, ctx: dict) -> list[list[str]]:
    """argv lists (after the interpreter) of a local stage, from the allowlisted templates only."""
    ov = OVERLAY.get(stage_id, {})
    cmds = ov.get("commands")
    if not cmds:
        raise NotRunnable(f"{stage_id} has no local command")
    if cmds == "cam_map":
        base = reused_map_base(ctx["take"])
        cmds = [[*M, "dancecap.cameras.reuse", "map", "{take}", "--from", base]] if base else \
            [[*DC, "cameras", "{take}", "--stages", "map"]]
    if cmds == "timing":
        take = ctx["take"]
        studio = paths.take_work(take) / "song" / "beats_studio.json"
        argv = [*DC, "timing", "{take}"]
        if studio.exists():
            argv += ["--beats", f"work/{take}/song/beats_studio.json", "--no-refine"]
        cmds = [argv]
    return [_subst(c, ctx) for c in cmds]


def session_prep(session: str, stages: list[str], ctx: dict) -> list[list[str]]:
    spec = POD_SESSIONS[session]
    prep = spec.get("prep") or []
    if isinstance(prep, dict):
        out = []
        for st in stages:
            out += [_subst(c, ctx) for c in prep.get(st, [])]
        return out
    return [_subst(c, ctx) for c in prep]


def session_jobs(session: str, stages: list[str], take: str) -> list[str]:
    spec = POD_SESSIONS[session]
    jobs = spec.get("jobs") or {}
    out = []
    for st in stages:
        j = jobs.get(st, jobs.get("*"))
        if j:
            j = j.replace("{take}", take)
            if j not in out:
                out.append(j)
    return out


def session_command(session: str, stages: list[str], ctx: dict, dry_run: bool = False) -> list[list[str]]:
    """The remote_jobs CLI that runs a batched session."""
    spec = POD_SESSIONS[session]
    jobs = session_jobs(session, stages, ctx["take"])
    if not jobs:
        raise NotRunnable(f"no pod jobs for {stages} in {session}")
    argv = [*DC, "remote", "run", "--take", ctx["take"], "--jobs", ",".join(jobs),
            "--max-cost", f"{float(ctx.get('max_cost') or spec['max_cost']):.2f}"]
    if ctx.get("max_minutes"):
        argv += ["--max-minutes", f"{float(ctx['max_minutes']):.0f}"]
    if ctx.get("gpus"):
        argv += ["--gpus", ",".join(ctx["gpus"])]
    if dry_run:
        argv.append("--dry-run")
    return [argv]


def gate_apply_commands(stage_id: str, ctx: dict) -> list[list[str]]:
    """The command a review gate's "apply" decision runs (geoveto --apply), from the templates only."""
    cmds = OVERLAY.get(stage_id, {}).get("gate_apply")
    if not cmds:
        raise NotRunnable(f"{stage_id} has no apply step")
    return [_subst(c, ctx) for c in cmds]


def _frame_count(take: str, cam: str) -> int | None:
    from .estimate import frame_count
    return frame_count(paths.take_work(take) / "frames" / cam / "frames.json")


def _count_masks(d: Path) -> int:
    try:
        return sum(1 for e in os.scandir(d) if e.name.startswith("mask_") and e.name.endswith(".png"))
    except OSError:
        return 0


def session_job_reruns(session: str, take: str, cams: list[str]) -> dict[str, float]:
    """{stage: fraction of its work} that the session's pod JOB will really run, whatever the plan selected.

    S1_front (front_pod.build_tasks) has no per-stage selection: per camera, dancers (label + SAM), localisation and
    ma_2d are skipped only when their resume markers exist (the same checks as front_pod: pod_cam/label json + every
    mask, loc_<cam>.npz + cameras/<cam>.npz, ma_2d_native/<cam>.manifest.json); merge, cameras (attach/world/qa),
    sync, chain and prepare always run. The plan prices, kill-times and re-runs (honestly) exactly this set.
    Other sessions run what they are given: {}."""
    if session != "S1_front" or not cams:
        return {}
    w = paths.take_work(take)

    def dancers_done(c: str) -> bool:
        if not ((w / "dancers" / f"pod_cam_{c}.json").exists() and (w / "dancers" / f"label_{c}.json").exists()):
            return False
        n = _frame_count(take, c)
        return bool(n) and all(_count_masks(w / "dancers" / c / sub) >= 2 * n for sub in ("masks", "body"))

    def loc_done(c: str) -> bool:
        return (w / "cameras" / "cache" / f"loc_{c}.npz").exists() and (w / "cameras" / f"{c}.npz").exists()

    def ma2d_done(c: str) -> bool:
        return (w / "mamma" / "ma_2d_native" / f"{c}.manifest.json").exists()

    n = float(len(cams))
    out = {"dancers_sam": sum(not dancers_done(c) for c in cams) / n,
           "cam_loc": sum(not loc_done(c) for c in cams) / n,
           "ma2d": sum(not ma2d_done(c) for c in cams) / n,
           "cam_world": 1.0, "sync": 1.0, "scale_chain": 1.0}
    return {k: round(v, 3) for k, v in out.items() if v > 0}


def stages_for_remote_job(job_name: str) -> list[str]:
    for prefix, stages in REMOTE_JOB_STAGES:
        if job_name == prefix or job_name.startswith(prefix):
            return stages
    return []


def public_registry() -> dict:
    """GET /api/stages: the registry, passed through (minus internal templates)."""
    d = registry()
    return {k: v for k, v in d.items() if k not in ("status",)} | {"columns": COLUMNS, "statuses": list(STATUSES)}
