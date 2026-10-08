"""Camera type per source video: mount (handheld | tripod) and zoom (yes | no), entered by the user, pre-filled by an
automatic suggestion from the data, and used by the camera stages.

    python -m dancecap.camera_types show <take> [--json]
    python -m dancecap.camera_types suggest <take> [--cams 01,02] [--probe a|b|both] [--write-toml] [--json]
    python -m dancecap.camera_types set <take> <cam> --mount handheld|tripod --zoom yes|no
    python -m dancecap.camera_types accept <take> [--cams 01,02] [--min-confidence 0.7]
    python -m dancecap.camera_types clear <take> <cam>          the user's answer goes, the suggestion stays

STORAGE (docs/PIPELINE.md 4.1, docs/PIPELINE_MONITOR.md 7.3-7.4): takes/<take>.toml, one table per source id, beside
the reserved [cameras] keys scale_rule / scale_reference / loc (only two-digit keys are camera tables):

    [cameras."05"]              # camera type (Pipeline Monitor / narration tool; block written by dancecap.camera_types)
    mount = "tripod"            # handheld | tripod                 (user)
    zoom = false                # true | false, independent of mount (user)
    source = "user"             # user = confirmed | auto = suggestion, not confirmed yet
    set_at = "2026-10-07T18:02:11Z"
    # written by the probe, kept for the record; never overrides the user:
    suggested = { mount = "tripod", zoom = false, pans = false, confidence = 0.64, by = "probe_a", at = "...",
                  evidence = { parallax_px = 0.3, jitter_hf = 0.05, zoom_scale = 1.004 } }   (one line in the file)
    pans = false                # tripod only, automatic
    zoom_events = []            # automatic: [t0, t1, focal ratio] in reference seconds

The [cameras."NN"] blocks are OWNED by this module: a block runs from its header to the next header and is rewritten
as a whole (a comment typed inside it is not kept); every byte outside the owned blocks is preserved. Keys of a block
that this module does not own (e.g. the monitor wizard's `file`) are re-emitted unchanged. Each write checks the
etag ("sha1:" + hex of the file bytes, the monitor's scheme), re-parses the result with tomllib and compares
everything outside the camera tables with the old file, backs the old file up under
work/<take>/monitor/toml_backups/ (millisecond names, never overwritten), and replaces it atomically. No tomlkit
dependency. Only `[cameras."NN"]` table headers are supported: a hand-written inline (`cameras = { "01" = {...} }`)
or dotted (`cameras."01".mount = ...`) form makes every save refuse with a parse/consistency error (nothing written).

RULES
  * Missing type = handheld + zoom: the most general model, exactly today's solver. Old takes are unchanged.
  * The user always wins. A measurement (probe B, newer than the user's answer, confidence >= 0.7) that contradicts
    the user sets state "conflict" (the camera_types gate opens with the evidence); it never changes the value.
    Answering again (set_type, even with the same values) acknowledges the evidence.
  * EFFECTIVE type (what the camera stages apply), policy `user` (default everywhere: CLI, pod jobs, orchestrator):
    confirmed answers only, anything else is handheld + zoom. Policy `auto`: plus suggestions with confidence >= 0.7.
    Policy `off`: handheld + zoom everywhere. Set it with `python -m dancecap cameras --camera-types ...` or
    DANCECAP_CAMERA_TYPES. CameraType.effective is computed with the `user` policy; every cameras run records the
    policy it used (report.json camera_types.policy, loc_<cam>.npz loc_policy) and the monitor's stale check compares
    with the class under THAT policy (stage_type(ct, policy)).
  * What the stage DID with the type (report.json cams.<cam>.camera_type) comes back as CameraType.model_check: a
    model the stage refused (the camera moves / zooms / the track would get shakier; it kept a safer model) is shown
    next to the answer and as a monitor warning (summary 'refused').

AUTOMATIC SUGGESTION (pre-fills the review UI; the user confirms)
  probe A, before calibration (suggest_from_video, local CPU, ~5-15 s per video at below-normal priority): background
    motion of a few 4 s windows of the video at 15 Hz, 640 px grey, central (dancer) region masked, LK tracks:
    parallax_px = homography transfer error where the epipolar geometry holds (1 s spans; ~0 for a camera that only
    rotates/zooms), jitter_hf = high-pass (> 2 Hz) frame-to-frame background motion in px (hand tremor), zoom_scale =
    cumulative similarity scale range inside the windows. Confidence capped at 0.85.
  probe B, after calibration (suggest_from_track / probe_b): today's per-frame track (raw localisation -> zoom-depth
    guard -> light smoothing, never the type-constrained export, which would be circular) in world metres:
    centre_p95_m (deviation from the median centre), axis_p95_deg, angvel_p50_dps, hf_rot_deg (rotation jitter after
    a 0.25 s smooth), focal_ratio (p95 / p5), focal_step_pct (max change within 0.5 s) and zoom_events.
    Thresholds (registry camera_types.suggest; uncalibrated for tripods - no tripod has been measured yet):
      tripod   centre p95 < 2 cm and rotation jitter < 0.05 deg     handheld  centre p95 > 5 cm (or jitter + motion)
      pans     tripod and axis p95 > 0.3 deg                        zoom      focal p95/p5 > 1.05 or a step > 3 %
      no zoom  focal p95/p5 < 1.02 and no step > 2 %                between   confidence < 0.7: the user decides
    Between the mount bounds the pre-fill is HANDHELD (the prior: a tripod must be evidenced; the lean is kept as
    evidence mount_lean); the same holds for any merged 'tripod' below 0.7 (prior_mount).
    Attached cameras: the attach model fixes the centre (rotation-only attach) and the focal (the sub-model's), so
    probe B cannot judge their mount (centre fixed) or zoom (always): None, probe A decides, never a conflict.
  Files: work/<take>/probe/camera_types_suggested.json (probe A), probe/camera_types_probe_b.json (probe B, also
  written by the cameras stage's qa step). `suggest --write-toml` also records them in the TOML (`suggested`, and
  mount/zoom with source = "auto" unless the user already answered).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
import tomllib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .take import PACKAGE_ROOT

MOUNTS = ("handheld", "tripod")
CLASSES = ("tripod", "tripod_zoom", "handheld", "handheld_zoom")
DEFAULT = {"mount": "handheld", "zoom": True}
MIN_CONFIDENCE = 0.7
POLICIES = ("user", "auto", "off")
POLICY_ENV = "DANCECAP_CAMERA_TYPES"
CAM_RE = re.compile(r"^[0-9]{2}$")
OWNED_KEYS = ("mount", "zoom", "source", "set_at", "suggested", "pans", "zoom_events", "conflict")
PROBE_A_MAX_CONF = 0.85

CLASS_EFFECTS = {
    "tripod": ["one rigid pose for the whole video, solved once robustly (keyframe localisation, ~10 % of the "
               "per-frame work); per-frame only if the residual demands it", "one intrinsic set per video",
               "a constant camera track for sync and the dancers (no per-frame pose noise)"],
    "tripod_zoom": ["fixed camera centre; per-frame rotation + focal", "no zoom-depth ambiguity (centre known)"],
    "handheld": ["per-frame 6-dof pose", "ONE focal + k1 per video (no per-frame focal race, no zoom-depth guard)"],
    "handheld_zoom": ["per-frame pose + focal (P4Pf / P3P race)", "log-focal smoothing, zoom-depth guard (today's "
                      "general solver)"],
}

_root: list[Path] = [PACKAGE_ROOT]


class CameraTypeError(ValueError):
    """Bad value or an unwritable take TOML."""


class StaleEtag(RuntimeError):
    """The take TOML changed since it was read (HTTP 412)."""

    def __init__(self, etag: str):
        super().__init__("the take TOML changed since it was read: reload")
        self.etag = etag


# ------------------------------------------------------------------------------------------------ paths

def set_root(p: str | Path | None) -> None:
    """Package root holding takes/ and work/ (tests point it at a temp tree). None restores the default."""
    _root[0] = Path(p).resolve() if p is not None else PACKAGE_ROOT


def root() -> Path:
    return _root[0]


def toml_path(take: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", take or "") or take.startswith("."):
        raise CameraTypeError(f"invalid take name {take!r}")
    return root() / "takes" / f"{take}.toml"


def work_dir(take: str) -> Path:
    return root() / "work" / take


def probe_dir(take: str) -> Path:
    return work_dir(take) / "probe"


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def to_epoch(v) -> float | None:
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, datetime):
        return (v if v.tzinfo else v.replace(tzinfo=timezone.utc)).timestamp()
    if isinstance(v, str):
        try:
            d = datetime.fromisoformat(v.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
        return (d if d.tzinfo else d.replace(tzinfo=timezone.utc)).timestamp()
    return None


def etag_of(b: bytes) -> str:
    return "sha1:" + hashlib.sha1(b).hexdigest()


def class_of(mount: str | None, zoom: bool | None) -> str:
    m = mount if mount in MOUNTS else DEFAULT["mount"]
    z = DEFAULT["zoom"] if zoom is None else bool(zoom)
    return f"{m}_zoom" if z else m


def resolve_policy(policy: str | None = None) -> str:
    p = (policy or os.environ.get(POLICY_ENV) or "user").strip().lower()
    if p not in POLICIES:
        raise CameraTypeError(f"camera-types policy must be one of {POLICIES}, not {p!r}")
    return p


# ------------------------------------------------------------------------------------------------ reading

def read_take(take: str) -> tuple[dict, bytes]:
    """(parsed TOML, raw bytes); ({}, b"") when the file does not exist."""
    p = toml_path(take)
    try:
        b = p.read_bytes()
    except FileNotFoundError:
        return {}, b""
    try:
        return tomllib.loads(b.decode("utf-8-sig")), b
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as e:
        raise CameraTypeError(f"{p.name}: {e}") from e


def cam_tables(cfg: dict) -> dict[str, dict]:
    """{cam: table} of the per-camera tables (two-digit keys only; scale_rule / scale_reference / loc are reserved)."""
    cams = cfg.get("cameras") or {}
    return {str(k): v for k, v in cams.items() if CAM_RE.match(str(k)) and isinstance(v, dict)}


def _alignment(cfg: dict) -> dict:
    a = cfg.get("alignment")
    if not isinstance(a, str) or not a:
        return {}
    p = Path(a)
    p = p if p.is_absolute() else (root() / "takes" / p).resolve()
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def known_cams(take: str, cfg: dict | None = None) -> list[str]:
    """Used source ids (audio-clock sources minus `exclude`) plus every camera that has a table or a probe result."""
    cfg = read_take(take)[0] if cfg is None else cfg
    excl = {str(x) for x in cfg.get("exclude", []) or []}
    cams = {str(r.get("id")) for r in _alignment(cfg).get("sources", []) or []
            if CAM_RE.match(str(r.get("id", ""))) and str(r.get("id")) not in excl and r.get("accepted", True)}
    cams |= set(cam_tables(cfg))
    for which in ("a", "b"):
        cams |= {c for c in read_probe(take, which) if c not in excl}
    return sorted(cams)


def read_probe(take: str, which: str) -> dict[str, dict]:
    """{cam: suggestion} from probe/camera_types_suggested.json (a) or probe/camera_types_probe_b.json (b)."""
    name = {"a": "camera_types_suggested.json", "b": "camera_types_probe_b.json"}[which]
    try:
        d = json.loads((probe_dir(take) / name).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    cams = d.get("cameras", d) if isinstance(d, dict) else {}
    return {str(k): v for k, v in cams.items() if CAM_RE.match(str(k)) and isinstance(v, dict)}


def _sugg(s) -> dict | None:
    """Normalised suggestion {mount, zoom, pans, confidence, mount_confidence, zoom_confidence, by, at, evidence,
    zoom_events} or None."""
    if not isinstance(s, dict):
        return None
    out = {"mount": s.get("mount") if s.get("mount") in MOUNTS else None,
           "zoom": s.get("zoom") if isinstance(s.get("zoom"), bool) else None,
           "pans": s.get("pans") if isinstance(s.get("pans"), bool) else None,
           "confidence": _num(s.get("confidence")), "mount_confidence": _num(s.get("mount_confidence")),
           "zoom_confidence": _num(s.get("zoom_confidence")), "by": s.get("by") or None,
           "at": to_epoch(s.get("at")), "evidence": dict(s.get("evidence") or {}),
           "zoom_events": list(s.get("zoom_events") or (s.get("evidence") or {}).get("zoom_events") or [])}
    if out["mount"] is None and out["zoom"] is None:
        return None
    return out


def _num(v) -> float | None:
    if isinstance(v, bool) or v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def prior_mount(s: dict | None) -> dict | None:
    """A 'tripod' below the confidence bar pre-fills HANDHELD (the prior: a tripod must be evidenced; larissa_kadu 07,
    a hand-held phone held very still, measured 'tripod' at 0.55); the lean stays in the evidence (mount_lean)."""
    if not s or s.get("mount") != "tripod":
        return s
    c = s.get("mount_confidence") if s.get("mount_confidence") is not None else s.get("confidence")
    if c is not None and c >= MIN_CONFIDENCE:
        return s
    return dict(s, mount="handheld", pans=None, evidence={**(s.get("evidence") or {}), "mount_lean": "tripod"})


def merge_suggestions(a: dict | None, b: dict | None) -> dict | None:
    """Probe B (calibrated camera track) over probe A (video before calibration), field by field: mount from B when
    B could judge it (not for cameras whose centre the attach model fixes), zoom from B when B could judge it (not
    for cameras whose focal the attach model fixes). A weak 'tripod' pre-fills handheld (prior_mount)."""
    return prior_mount(_merge(a, b))


def _merge(a: dict | None, b: dict | None) -> dict | None:
    a, b = _sugg(a), _sugg(b)
    if a is None or b is None:
        return a or b
    out = dict(b)
    used = []
    if b["mount"] is None and a["mount"] is not None:
        out["mount"], out["mount_confidence"], out["pans"] = a["mount"], a["mount_confidence"] or a["confidence"], a["pans"]
        used.append("mount")
    if b["zoom"] is None and a["zoom"] is not None:
        out["zoom"], out["zoom_confidence"] = a["zoom"], a["zoom_confidence"] or a["confidence"]
        used.append("zoom")
    if used:
        confs = [c for c in (out["mount_confidence"], out["zoom_confidence"]) if c is not None]
        out["confidence"] = round(min(confs), 3) if confs else min(x for x in (a["confidence"], b["confidence"])
                                                                    if x is not None)
        out["by"] = f"{b['by'] or 'probe_b'}+{a['by'] or 'probe_a'} ({', '.join(used)})"
        out["evidence"] = {**{k: v for k, v in a["evidence"].items()}, **b["evidence"]}
    return out


def _conflict(table: dict, mount: str, zoom: bool, measured: dict | None) -> dict | None:
    """A probe-B measurement newer than the user's answer, confident enough, that contradicts it."""
    m = _sugg(measured)
    if m is None or not (m.get("by") or "").startswith("probe_b"):
        return None
    set_at = to_epoch(table.get("set_at")) or 0.0
    if (m.get("at") or 0.0) <= set_at:
        return None  # the user answered after this measurement: the evidence was acknowledged
    for field, user, meas, conf in (("mount", mount, m["mount"], m["mount_confidence"]),
                                    ("zoom", zoom, m["zoom"], m["zoom_confidence"])):
        c = conf if conf is not None else m["confidence"]
        if meas is None or meas == user or c is None or c < MIN_CONFIDENCE:
            continue
        ev = m["evidence"]
        why = (f"centre moves {100 * ev['centre_p95_m']:.0f} cm (p95)" if field == "mount" and ev.get("centre_p95_m")
               is not None else f"focal range x{ev['focal_ratio']:.2f}" if field == "zoom" and ev.get("focal_ratio")
               is not None else "")
        return {"field": field, "user": user, "measured": meas, "confidence": round(c, 3),
                "message": f"you said: {_word(field, user)}; the camera track says: {_word(field, meas)}"
                           + (f" ({why}, confidence {c:.2f})" if why else f" (confidence {c:.2f})"),
                "evidence": ev}
    return None


def _word(field: str, v) -> str:
    if field == "mount":
        return "on a tripod" if v == "tripod" else "hand-held"
    return "zoomed during the video" if v else "no zoom during the video"


def describe(cam: str, table: dict | None, sug_a: dict | None = None, sug_b: dict | None = None,
             applied: dict | None = None) -> dict:
    """CameraType JSON (PIPELINE_MONITOR.md 7.3) from the TOML table, the probe files' suggestions and what the last
    cameras run applied (report.json cams.<cam>.camera_type -> model_check)."""
    t = dict(table or {})
    sug = merge_suggestions(sug_a, sug_b)
    tab_sug = prior_mount(_sugg(t.get("suggested")))
    if tab_sug is not None and (sug is None or (tab_sug.get("at") or 0) > (sug.get("at") or 0)):
        sug = tab_sug
    mount = t.get("mount") if t.get("mount") in MOUNTS else None
    zoom = t.get("zoom") if isinstance(t.get("zoom"), bool) else None
    source = t.get("source")
    conflict = None
    if source == "user" and mount is not None and zoom is not None:
        state = "confirmed"
        conflict = _conflict(t, mount, zoom, sug_b)
        if conflict:
            state = "conflict"
    elif sug is not None or source == "auto":
        state, source = "suggested", "auto"
        if sug is not None:
            mount = sug["mount"] if sug["mount"] is not None else mount
            zoom = sug["zoom"] if sug["zoom"] is not None else zoom
    else:
        state, source = "unset", "default"
    user = state in ("confirmed", "conflict")
    eff = {"mount": mount if user else DEFAULT["mount"], "zoom": zoom if user else DEFAULT["zoom"]}
    eff["class"] = class_of(eff["mount"], eff["zoom"])
    pans = t.get("pans") if isinstance(t.get("pans"), bool) else (sug or {}).get("pans")
    effects = list(CLASS_EFFECTS[eff["class"]])
    if eff["class"] == "tripod" and pans:
        effects[0] = "fixed camera centre, per-frame rotation only (the head pans)"
    s_out = None
    if sug is not None:
        s_out = {k: sug.get(k) for k in ("mount", "zoom", "pans", "confidence", "mount_confidence",
                                          "zoom_confidence", "by", "at", "evidence", "zoom_events")}
    return {"cam": cam, "mount": mount, "zoom": zoom, "source": source, "state": state,
            "set_at": to_epoch(t.get("set_at")), "pans": pans,
            "zoom_events": t.get("zoom_events") or (sug or {}).get("zoom_events") or [],
            "effective": eff, "suggestion": s_out, "conflict": conflict, "effects": effects,
            "model_check": model_check(applied, eff["class"])}


def model_check(applied: dict | None, current_class: str) -> dict | None:
    """What the last cameras run did with this camera's type (report.json cams.<cam>.camera_type): {declared, model,
    passed, why, at, current, refused}. refused = the stage could not use the CURRENT declared type and kept another
    model (the reason says why: the camera moves / zooms / the track got shakier) - shown next to the answer and as a
    warning in the monitor; change the answer or keep it (the safe fallback model stays)."""
    if not isinstance(applied, dict) or not applied.get("class"):
        return None
    chk = applied.get("check") if isinstance(applied.get("check"), dict) else {}
    declared = str(applied["class"])
    passed = chk.get("passed")
    out = {"declared": declared, "model": applied.get("model"), "passed": passed,
           "why": chk.get("why") or applied.get("why"), "at": to_epoch(applied.get("at")),
           "current": declared == current_class}
    out["refused"] = bool(out["current"] and passed is False and declared != "handheld_zoom")
    return out


_REPORT_CACHE: dict = {}


def applied_types(take: str) -> dict[str, dict]:
    """{cam: report.json cams.<cam>.camera_type} of the last cameras run (cached by file size + mtime)."""
    p = work_dir(take) / "cameras" / "report.json"
    try:
        st = p.stat()
    except OSError:
        return {}
    key = (st.st_size, st.st_mtime_ns)
    hit = _REPORT_CACHE.get(str(p))
    if hit and hit[0] == key:
        return hit[1]
    try:
        rep_ = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out = {str(c): v["camera_type"] for c, v in ((rep_ or {}).get("cams") or {}).items()
           if isinstance(v, dict) and isinstance(v.get("camera_type"), dict)}
    _REPORT_CACHE[str(p)] = (key, out)
    return out


def load(take: str, cams: list[str] | None = None) -> tuple[list[dict], str]:
    """(CameraType per camera, TOML etag). cams=None: every known camera of the take."""
    cfg, b = read_take(take)
    tabs = cam_tables(cfg)
    pa, pb = read_probe(take, "a"), read_probe(take, "b")
    ap = applied_types(take)
    cams = known_cams(take, cfg) if cams is None else list(cams)
    return [describe(c, tabs.get(c), pa.get(c), pb.get(c), ap.get(c)) for c in cams], etag_of(b)


def get(take: str, cam: str) -> dict:
    return load(take, [cam])[0][0]


def needs_user(ct: dict) -> bool:
    """True when this camera should block its camera stages: a conflict, or no answer and no confident suggestion."""
    if ct["state"] == "conflict":
        return True
    if ct["state"] == "confirmed":
        return False
    return float((ct.get("suggestion") or {}).get("confidence") or 0.0) < MIN_CONFIDENCE


def summary(types: list[dict]) -> dict:
    out = {"confirmed": 0, "suggested": 0, "unset": 0, "conflict": 0, "refused": 0}
    for t in types:
        out[t["state"]] += 1
        out["refused"] += bool((t.get("model_check") or {}).get("refused"))
    return out


# ------------------------------------------------------------------------------------------------ stage view

@dataclass(frozen=True)
class StageType:
    """What a camera stage applies to one camera (see the module docstring, EFFECTIVE type)."""
    cam: str
    mount: str = "handheld"
    zoom: bool = True
    pans: bool | None = None
    source: str = "default"  # user | auto | default
    confidence: float | None = None

    @property
    def cls(self) -> str:
        return class_of(self.mount, self.zoom)

    @property
    def tripod(self) -> bool:
        return self.mount == "tripod"

    @property
    def fixed_focal(self) -> bool:
        return not self.zoom

    @property
    def general(self) -> bool:
        return self.cls == "handheld_zoom"

    def as_dict(self) -> dict:
        return {**asdict(self), "class": self.cls}


def stage_type(ct: dict, policy: str = "user") -> StageType:
    """CameraType -> StageType under a policy (user: confirmed only; auto: + confident suggestions; off: none)."""
    cam = ct["cam"]
    if policy == "off":
        return StageType(cam)
    if ct["state"] in ("confirmed", "conflict"):
        return StageType(cam, ct["mount"], bool(ct["zoom"]), ct.get("pans"), "user")
    s = ct.get("suggestion") or {}
    if policy == "auto" and ct["state"] == "suggested" and float(s.get("confidence") or 0.0) >= MIN_CONFIDENCE \
            and s.get("mount") in MOUNTS and isinstance(s.get("zoom"), bool):
        return StageType(cam, s["mount"], s["zoom"], s.get("pans"), "auto", s.get("confidence"))
    return StageType(cam)


def stage_types(take: str, cams: list[str] | None = None, policy: str | None = None) -> dict[str, StageType]:
    """{cam: StageType} of a take under the policy (default: DANCECAP_CAMERA_TYPES or 'user'). Missing TOML: {}."""
    policy = resolve_policy(policy)
    try:
        types, _ = load(take, cams)
    except CameraTypeError:
        return {}
    return {t["cam"]: stage_type(t, policy) for t in types}


def types_for_work(work: Path, cams: list[str] | None = None, policy: str | None = None) -> dict[str, StageType]:
    """stage_types for a work directory: only for work/<take>/ of this package root (a scratch copy gets none)."""
    work = Path(work).resolve()
    if work.parent != (root() / "work").resolve() or not toml_path(work.name).exists():
        return {}
    return stage_types(work.name, cams, policy)


def effective(take: str, cam: str, policy: str | None = None) -> str:
    """Class the camera stages apply to this camera (tripod | tripod_zoom | handheld | handheld_zoom)."""
    return stage_types(take, [cam], policy).get(cam, StageType(cam)).cls


# ------------------------------------------------------------------------------------------------ TOML writing

_KEY = r'(?:[A-Za-z0-9_-]+|"[^"\n]*"|\'[^\'\n]*\')'
_HDR = re.compile(r"^[ \t]*\[\[?[ \t]*(" + _KEY + r"(?:[ \t]*\.[ \t]*" + _KEY + r")*)[ \t]*\]\]?[ \t]*(?:#.*)?$")
_PART = re.compile(_KEY)
_INTRO = "# Camera type per source video (dancecap.camera_types): set in the narration tool or the Pipeline Monitor"


def _header_parts(line: str) -> list[str] | None:
    m = _HDR.match(line.rstrip("\r\n"))
    if not m:
        return None
    return [p.strip("\"'") for p in _PART.findall(m.group(1))]


def _subtable_keys(text: str, cam: str) -> set[str]:
    """Keys of [cameras."NN".<key>...] headers in the file (sub-tables outside the owned block)."""
    out = set()
    for line in text.splitlines():
        parts = _header_parts(line)
        if parts and len(parts) >= 3 and parts[0] == "cameras" and parts[1] == cam:
            out.add(parts[2])
    return out


def _block_cam(line: str) -> str | None:
    parts = _header_parts(line)
    if parts and len(parts) == 2 and parts[0] == "cameras" and CAM_RE.match(parts[1]) \
            and not line.lstrip().startswith("[["):
        return parts[1]
    return None


def _toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if not math.isfinite(v):
            raise CameraTypeError("non-finite number")
        return repr(round(v, 6))
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    if isinstance(v, dict):
        items = [f"{_toml_key(k)} = {_toml_value(x)}" for k, x in v.items() if _keep(x)]
        return "{ " + ", ".join(items) + " }" if items else "{}"
    raise CameraTypeError(f"cannot write {type(v).__name__} to the take TOML")


def _keep(x) -> bool:
    return x is not None and not (isinstance(x, float) and not math.isfinite(x))


def _toml_key(k: str) -> str:
    return k if re.fullmatch(r"[A-Za-z0-9_-]+", str(k)) else json.dumps(str(k))


def render_block(cam: str, t: dict, nl: str = "\n", subtables: set[str] | frozenset = frozenset()) -> str:
    """One owned [cameras."NN"] block (fixed comments; keys this module does not own are re-emitted first, except
    sub-tables that have their own [cameras."NN".<key>] header elsewhere in the file)."""
    lines = [f'[cameras."{cam}"]              # camera type (Pipeline Monitor / narration tool; block written by '
             f'dancecap.camera_types)']
    for k, v in t.items():
        if k not in OWNED_KEYS and _keep(v) and k not in subtables:
            lines.append(f"{_toml_key(k)} = {_toml_value(v)}")
    if t.get("mount") in MOUNTS:
        lines.append(f"{'mount = ' + _toml_value(t['mount']):<28}# handheld | tripod                 (user)")
    if isinstance(t.get("zoom"), bool):
        lines.append(f"{'zoom = ' + _toml_value(t['zoom']):<28}# true | false, independent of mount (user)")
    if t.get("source") in ("user", "auto"):
        lines.append(f"{'source = ' + _toml_value(t['source']):<28}# user = confirmed | auto = suggestion, not "
                     f"confirmed yet")
    if t.get("set_at"):
        lines.append(f"set_at = {_toml_value(str(t['set_at']))}")
    auto = [k for k in ("suggested", "pans", "zoom_events") if _keep(t.get(k))]
    if auto:
        lines.append("# written by the probe, kept for the record; never overrides the user:")
        for k in auto:
            v = t[k]
            line = f"{k} = {_toml_value(v)}"
            if k == "pans":
                line = f"{line:<28}# tripod only, automatic"
            elif k == "zoom_events":
                line = f"{line:<28}# automatic: [t0, t1, focal ratio] in reference seconds"
            lines.append(line)
    return nl.join(lines) + nl


def replace_blocks(text: str, blocks: dict[str, str]) -> str:
    """Rewrite each owned [cameras."NN"] block in place (header to the next header; blank and comment lines just before
    the next header stay with it), append the missing ones at the end of the file in camera order."""
    nl = "\r\n" if "\r\n" in text else "\n"
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    done: set[str] = set()
    had_block = False
    i = 0
    while i < len(lines):
        cam = _block_cam(lines[i])
        if cam is not None:
            had_block = True
        if cam is not None and cam in blocks:
            j = i + 1
            while j < len(lines) and _header_parts(lines[j]) is None:
                j += 1
            k = j
            while k > i + 1 and (not lines[k - 1].strip() or lines[k - 1].lstrip().startswith("#")):
                k -= 1
            out.append(blocks[cam])
            out.extend(lines[k:j])
            done.add(cam)
            i = j
            continue
        out.append(lines[i])
        i += 1
    rest = [c for c in sorted(blocks) if c not in done]
    if rest:
        if out and not out[-1].endswith(("\n", "\r")):
            out.append(nl)
        if not had_block:
            out.append(nl + _INTRO + nl)
        for c in rest:
            if out and out[-1].strip():
                out.append(nl)
            out.append(blocks[c])
    return "".join(out)


def _strip_cams(cfg: dict) -> dict:
    d = dict(cfg)
    cams = {k: v for k, v in (d.get("cameras") or {}).items() if not (CAM_RE.match(str(k)) and isinstance(v, dict))}
    if cams or any(not CAM_RE.match(str(k)) for k in (cfg.get("cameras") or {})):
        d["cameras"] = cams
    else:
        d.pop("cameras", None)
    return d


def _norm(v):
    if isinstance(v, dict):
        return {k: _norm(x) for k, x in v.items() if _keep(x)}
    if isinstance(v, (list, tuple)):
        return [_norm(x) for x in v]
    if isinstance(v, float):
        return round(v, 6)
    return v


def _replace_file(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    for k in range(8):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:  # a reader holding the file on Windows: retry briefly
            if k == 7:
                tmp.unlink(missing_ok=True)
                raise
            time.sleep(0.05 * (k + 1))


def write_tables(take: str, updates: dict[str, dict], if_match: str | None = None) -> str:
    """Apply {cam: {key: value | None (remove)}} to the owned blocks; returns the new etag. Raises StaleEtag when
    if_match does not match the file, CameraTypeError when the result would not be the intended file."""
    path = toml_path(take)
    for attempt in range(3):
        try:
            b = path.read_bytes()
        except FileNotFoundError as e:
            raise CameraTypeError(f"no take TOML for {take}") from e
        cur = etag_of(b)
        if if_match is not None and if_match.strip().strip('"') != cur:
            raise StaleEtag(cur)
        text = b.decode("utf-8-sig")
        bom = b.startswith(b"\xef\xbb\xbf")
        cfg = tomllib.loads(text)
        tabs = cam_tables(cfg)
        nl = "\r\n" if "\r\n" in text else "\n"
        want: dict[str, dict] = {}
        blocks = {}
        for cam, upd in updates.items():
            if not CAM_RE.match(cam):
                raise CameraTypeError("camera ids are two digits")
            t = dict(tabs.get(cam, {}))
            for k, v in upd.items():
                if v is None:
                    t.pop(k, None)
                else:
                    t[k] = v
            t.pop("conflict", None)  # conflicts are computed from the evidence, never stored
            want[cam] = t
            blocks[cam] = render_block(cam, t, nl, _subtable_keys(text, cam))
        new = replace_blocks(text, blocks)
        try:
            parsed = tomllib.loads(new)
        except tomllib.TOMLDecodeError as e:
            raise CameraTypeError(f"refused: the edited file would not parse ({e})") from e
        if _norm(_strip_cams(parsed)) != _norm(_strip_cams(cfg)):
            raise CameraTypeError("refused: the edit would change keys outside the camera-type tables")
        ptabs = cam_tables(parsed)
        for cam, t in want.items():
            exp = {k: v for k, v in t.items() if _keep(v)}
            if _norm(ptabs.get(cam, {})) != _norm(exp):
                raise CameraTypeError(f"refused: the written table for {cam} would differ from the intended one")
        for cam in set(tabs) - set(want):
            if _norm(ptabs.get(cam)) != _norm(tabs[cam]):
                raise CameraTypeError(f"refused: the edit would change the table of camera {cam}")
        data = (b"\xef\xbb\xbf" if bom else b"") + new.encode("utf-8")
        if path.read_bytes() != b:  # changed under us between read and write: start again (or 412)
            if if_match is not None:
                raise StaleEtag(etag_of(path.read_bytes()))
            continue
        backup(take, b)
        _replace_file(path, data)
        return etag_of(data)
    raise StaleEtag(etag_of(path.read_bytes()))


def backup(take: str, b: bytes) -> Path:
    """work/<take>/monitor/toml_backups/<YYYYmmdd-HHMMSS-mmm>-<take>.toml; never overwrites an earlier backup (two
    saves in one second keep both previous versions)."""
    bdir = work_dir(take) / "monitor" / "toml_backups"
    bdir.mkdir(parents=True, exist_ok=True)
    now = time.time()
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(now)) + f"-{int(1000 * (now % 1)):03d}"
    for k in range(1000):
        p = bdir / (f"{stamp}-{take}.toml" if k == 0 else f"{stamp}-{k}-{take}.toml")
        try:
            with open(p, "xb") as fh:
                fh.write(b)
            return p
        except FileExistsError:
            continue
    raise CameraTypeError("could not create a TOML backup")


def _check_values(cam: str, mount, zoom) -> None:
    if not CAM_RE.match(cam or ""):
        raise CameraTypeError("cam must be two digits")
    if mount not in MOUNTS:
        raise CameraTypeError("mount must be handheld or tripod")
    if not isinstance(zoom, bool):
        raise CameraTypeError("zoom must be true or false")


def set_type(take: str, cam: str, mount: str, zoom: bool, if_match: str | None = None) -> tuple[dict, str]:
    """The user's answer (source = user). Answering again also acknowledges a conflict's evidence."""
    _check_values(cam, mount, zoom)
    etag = write_tables(take, {cam: {"mount": mount, "zoom": zoom, "source": "user", "set_at": now_iso()}}, if_match)
    return get(take, cam), etag


def clear(take: str, cam: str, if_match: str | None = None) -> tuple[dict, str]:
    """Remove the user's answer (the suggestion stays): the camera reads as suggested / unset again."""
    if not CAM_RE.match(cam or ""):
        raise CameraTypeError("cam must be two digits")
    etag = write_tables(take, {cam: {"mount": None, "zoom": None, "source": None, "set_at": None}}, if_match)
    return get(take, cam), etag


def accept(take: str, cams: list[str] | None = None, min_confidence: float = MIN_CONFIDENCE,
           if_match: str | None = None) -> tuple[list[dict], str]:
    """Suggestions with confidence >= min_confidence become the user's answer (source = user)."""
    types, etag = load(take, cams)
    upd = {}
    for t in types:
        s = t.get("suggestion") or {}
        if t["state"] != "suggested" or s.get("mount") not in MOUNTS or not isinstance(s.get("zoom"), bool):
            continue
        if float(s.get("confidence") or 0.0) < min_confidence:
            continue
        upd[t["cam"]] = {"mount": s["mount"], "zoom": s["zoom"], "source": "user", "set_at": now_iso()}
    if upd:
        etag = write_tables(take, upd, if_match)
    elif if_match is not None and if_match.strip().strip('"') != etag:
        raise StaleEtag(etag)
    return load(take, [t["cam"] for t in types])[0], etag


def _toml_suggestion(s: dict) -> dict:
    """Compact suggestion for the TOML (one inline table): rounded evidence, no nulls, no long lists."""
    ev = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in (s.get("evidence") or {}).items()
          if isinstance(v, (int, float, bool)) and not isinstance(v, bool) and _keep(v)}
    out = {k: s.get(k) for k in ("mount", "zoom", "pans", "confidence", "by") if s.get(k) is not None}
    if isinstance(out.get("confidence"), float):
        out["confidence"] = round(out["confidence"], 3)
    out["at"] = s["at"] if isinstance(s.get("at"), str) else (
        datetime.fromtimestamp(s["at"], timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if s.get("at") else now_iso())
    if ev:
        out["evidence"] = dict(list(ev.items())[:8])
    return out


def write_suggestion(take: str, cam: str, suggestion: dict, if_match: str | None = None) -> str:
    """Record an automatic suggestion in the TOML; pre-fills mount/zoom (source = auto) unless the user answered."""
    if not CAM_RE.match(cam or ""):
        raise CameraTypeError("cam must be two digits")
    s = _sugg(suggestion)
    if s is None:
        raise CameraTypeError("empty suggestion")
    cfg, _ = read_take(take)
    t = cam_tables(cfg).get(cam, {})
    upd: dict = {"suggested": _toml_suggestion({**suggestion, **{k: v for k, v in s.items() if v is not None}})}
    if s.get("pans") is not None:
        upd["pans"] = bool(s["pans"])
    if s.get("zoom_events"):
        upd["zoom_events"] = [[round(float(x), 3) for x in e[:3]] for e in s["zoom_events"] if len(e) >= 3]
    if t.get("source") != "user":
        upd.update({"mount": s["mount"], "zoom": s["zoom"], "source": "auto"})
    return write_tables(take, {cam: upd}, if_match)


# ------------------------------------------------------------------------------------------------ probe B (tracks)

TRIPOD_CENTRE_M, HANDHELD_CENTRE_M = 0.02, 0.05
TRIPOD_HF_ROT_DEG = 0.05
PANS_AXIS_DEG = 0.3
ZOOM_RATIO, NOZOOM_RATIO = 1.05, 1.02
ZOOM_STEP_PCT, NOZOOM_STEP_PCT = 3.0, 2.0


def track_metrics(t, R, C, f, ok, scale: float | None = None) -> dict:
    """Camera-motion metrics of one per-frame track (world->camera R, centres C in SfM units x scale = metres)."""
    import numpy as np

    from .cameras.smooth import robust_smooth
    t, R, C, f = (np.asarray(x, float) for x in (t, R, C, f))
    ok = np.asarray(ok, bool) & np.isfinite(C).all(1) & np.isfinite(f) & (f > 0)
    out: dict = {"frames": int(ok.sum()), "metric": scale is not None}
    if ok.sum() < 5:
        return out
    s = float(scale) if scale is not None else 1.0
    t, R, C, f = t[ok], R[ok], C[ok] * s, f[ok]
    med = np.median(C, 0)
    out["centre_p95_m"] = round(float(np.percentile(np.linalg.norm(C - med, axis=1), 95)), 4)
    ax = R[:, 2, :]
    m = ax.mean(0)
    m /= np.linalg.norm(m)
    out["axis_p95_deg"] = round(float(np.percentile(np.degrees(np.arccos(np.clip(ax @ m, -1, 1))), 95)), 3)
    dt = np.diff(t)
    rel = np.einsum("fij,fkj->fik", R[1:], R[:-1])
    ang = np.degrees(np.arccos(np.clip((np.trace(rel, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    good = dt > 1e-4
    out["angvel_p50_dps"] = round(float(np.median(ang[good] / dt[good])), 3) if good.any() else None
    # high-frequency rotation / centre jitter: residual after a 0.25 s robust local-linear smooth
    q = _quat(R)
    qs, _ = robust_smooth(t, q, np.ones(len(t)), 0.25)
    qs /= np.linalg.norm(qs, axis=1, keepdims=True)
    out["hf_rot_deg"] = round(float(np.sqrt(np.mean(np.degrees(2 * np.arccos(np.clip(np.abs((q * qs).sum(1)), -1, 1)))
                                                   ** 2))), 4)
    Cs, _ = robust_smooth(t, C, np.ones(len(t)), 0.25)
    out["hf_centre_mm"] = round(float(1000 * np.sqrt(np.mean(np.sum((C - Cs) ** 2, 1)))), 3)
    lf = np.log(f)
    out["focal_ratio"] = round(float(np.exp(np.percentile(lf, 95) - np.percentile(lf, 5))), 4)
    lfs, _ = robust_smooth(t, lf, np.ones(len(t)), 0.25)
    out["hf_focal_pct"] = round(float(100 * np.sqrt(np.mean((lf - lfs) ** 2))), 4)
    lfm, _ = robust_smooth(t, lf, np.ones(len(t)), 0.1)  # per-frame P4Pf noise off, real zooms kept
    step = 0.0
    j = 0
    for i in range(len(t)):
        while t[j] < t[i] - 0.5:
            j += 1
        step = max(step, float(np.abs(lfm[i] - lfm[j:i + 1]).max()))
    out["focal_step_pct"] = round(100 * (math.exp(step) - 1), 2)
    return out


def _quat(R):
    from scipy.spatial.transform import Rotation

    from .cameras.smooth import quat_from_R
    try:
        return quat_from_R(R)
    except Exception:  # noqa: BLE001
        return Rotation.from_matrix(R).as_quat()


def suggest_from_metrics(m: dict, mount_judgeable: bool = True, zoom_events: list | None = None,
                         by: str = "probe_b", zoom_judgeable: bool = True) -> dict | None:
    """Rules of the module docstring -> suggestion {mount, zoom, pans, confidence, mount_confidence, zoom_confidence,
    by, at, evidence}. mount_judgeable / zoom_judgeable False: the attach model fixes the centre / the focal, so the
    track says nothing about it (None; probe A decides) - never evidence for a conflict."""
    if not m or m.get("frames", 0) < 5:
        return None
    c95, hf, ax = m.get("centre_p95_m"), m.get("hf_rot_deg") or 0.0, m.get("axis_p95_deg") or 0.0
    av = m.get("angvel_p50_dps") or 0.0
    mount = mconf = pans = None
    if mount_judgeable and c95 is not None and m.get("metric", True):
        if c95 < TRIPOD_CENTRE_M and hf < TRIPOD_HF_ROT_DEG:
            mount, mconf = "tripod", 0.75 + 0.2 * min(1.0, (TRIPOD_CENTRE_M - c95) / TRIPOD_CENTRE_M)
            pans = ax > PANS_AXIS_DEG
        elif c95 > HANDHELD_CENTRE_M:
            mount, mconf = "handheld", min(0.98, 0.7 + 0.28 * (c95 - HANDHELD_CENTRE_M) / HANDHELD_CENTRE_M)
        elif av > 0.5 and hf >= TRIPOD_HF_ROT_DEG:
            mount, mconf = "handheld", 0.72
        else:  # between the bounds: the user decides; pre-fill handheld (the prior), keep the lean as evidence
            lean = "tripod" if (c95 < 0.5 * (TRIPOD_CENTRE_M + HANDHELD_CENTRE_M) and hf < 2 * TRIPOD_HF_ROT_DEG) \
                else "handheld"
            mount, mconf = "handheld", 0.55
            m = dict(m, mount_lean=lean)
    fr, st = m.get("focal_ratio"), m.get("focal_step_pct") or 0.0
    zoom = zconf = None
    if fr is not None and zoom_judgeable:
        if fr > ZOOM_RATIO or st > ZOOM_STEP_PCT:
            zoom, zconf = True, min(0.98, 0.75 + 2.0 * max(fr - ZOOM_RATIO, (st - ZOOM_STEP_PCT) / 100))
        elif fr < NOZOOM_RATIO and st < NOZOOM_STEP_PCT:
            zoom, zconf = False, max(0.75, min(0.98, 0.98 - 10.0 * (fr - 1.0)))
        else:
            zoom, zconf = fr > 0.5 * (ZOOM_RATIO + NOZOOM_RATIO), 0.55
    confs = [c for c in (mconf, zconf) if c is not None]
    if not confs:
        return None
    ev = {k: m.get(k) for k in ("centre_p95_m", "axis_p95_deg", "angvel_p50_dps", "hf_rot_deg", "hf_centre_mm",
                                "focal_ratio", "focal_step_pct", "frames")}
    ev["mount_judgeable"] = mount_judgeable
    ev["zoom_judgeable"] = zoom_judgeable
    if m.get("mount_lean"):
        ev["mount_lean"] = m["mount_lean"]
    return {"mount": mount, "zoom": zoom, "pans": pans if mount == "tripod" else None,
            "confidence": round(min(confs), 3), "mount_confidence": round(mconf, 3) if mconf is not None else None,
            "zoom_confidence": round(zconf, 3) if zconf is not None else None, "by": by, "at": now_iso(),
            "evidence": ev, "zoom_events": zoom_events or []}


def suggest_from_track(tr: dict, scale: float | None) -> dict | None:
    """Probe B for one camera from a cameras.pipeline.stage_smooth track (the general per-frame model). An attached
    camera's centre (rotation-only attach) and focal (its sub-model's) are fixed BY THE MODEL: the track cannot judge
    the mount / the zoom then (None, probe A decides; never a conflict)."""
    import numpy as np
    m = track_metrics(tr["t"], tr["R"], tr["C"], tr["f"], tr["ok"], scale)
    fixed_by_model = bool(tr.get("attached")) and m.get("centre_p95_m") is not None and \
        m["centre_p95_m"] < 1e-4 * (1.0 if scale else 1e-3)
    f_raw = np.asarray(tr.get("f_raw", tr["f"]), float)
    ok_raw = np.asarray(tr.get("ok_raw", tr["ok"]), bool) & np.isfinite(f_raw) & (f_raw > 0)
    focal_fixed = bool(tr.get("attached")) or (ok_raw.sum() >= 5 and float(np.ptp(np.log(f_raw[ok_raw]))) < 1e-9)
    zev = []
    try:
        from .cameras.pipeline import focal_stats
        fs = focal_stats(tr)
        f = np.asarray(tr["f"], float)
        for z in fs.get("zoom_events", []):
            i0 = int(np.argmin(np.abs(tr["t"] - z["t0"])))
            i1 = int(np.argmin(np.abs(tr["t"] - z["t1"])))
            zev.append([z["t0"], z["t1"], round(float(f[i1] / f[i0]), 3)])
    except Exception:  # noqa: BLE001 - the events are evidence only
        pass
    s = suggest_from_metrics(m, mount_judgeable=not fixed_by_model, zoom_events=[] if focal_fixed else zev,
                             zoom_judgeable=not focal_fixed)
    if s is not None:
        s["evidence"]["attached"] = bool(tr.get("attached"))
        if tr.get("keyframes_only"):
            s["evidence"]["keyframes_only"] = True
    return s


def probe_b(take: str, cams: list[str] | None = None, write: bool = True, tracks: dict | None = None,
            sim3: dict | None = None, log=print) -> dict[str, dict]:
    """Probe B on a calibrated take: today's per-frame tracks (never the type-constrained export) in world metres ->
    {cam: suggestion}; writes work/<take>/probe/camera_types_probe_b.json."""
    work = work_dir(take)
    sim3_p = work / "cameras" / "world_from_sfm.json"
    if sim3 is None and sim3_p.exists():
        sim3 = json.loads(sim3_p.read_text())
    if tracks is None:
        cams = cams or sorted(p.stem for p in (work / "cameras").glob("[0-9][0-9].npz"))
        tracks, redo = {}, []
        for c in cams:  # an export without a type model IS today's general per-frame track
            tr = _track_from_npz(work, c)
            if tr is None:
                redo.append(c)
            else:
                tracks[c] = tr
        redo = [c for c in redo if (work / "cameras" / "cache" / f"loc_{c}.npz").exists()]
        if redo:
            from .cameras.pipeline import stage_smooth
            tracks.update(stage_smooth(work.resolve(), redo, 0.04, 4.0, write=False, types={}))
    res, nothing = {}, []
    for cam in sorted(tracks):
        if cams and cam not in cams:
            continue
        s = suggest_from_track(tracks[cam], float(sim3["s"]) if sim3 else None)
        if s is None:  # nothing the track can judge (attached: centre and focal fixed by the model) - probe A decides
            nothing.append(cam)
            log(f"[camera_types] probe B {cam}: no verdict (" + ("centre and focal fixed by the attach model"
                if tracks[cam].get("attached") else "too few frames") + "): probe A decides")
            continue
        res[cam] = s
        ev = s["evidence"]
        log(f"[camera_types] probe B {cam}: {s['mount'] or '?'} ({_c(s['mount_confidence'])}), "
            f"{'zoom' if s['zoom'] else 'no zoom' if s['zoom'] is False else '?'} ({_c(s['zoom_confidence'])}); "
            f"centre p95 {_f(ev.get('centre_p95_m'), 100, 'cm')}, axis p95 {_f(ev.get('axis_p95_deg'), 1, 'deg')}, "
            f"rot jitter {_f(ev.get('hf_rot_deg'), 1, 'deg', 3)}, focal x{ev.get('focal_ratio')}, "
            f"step {ev.get('focal_step_pct')} %" + ("" if ev.get("mount_judgeable", True) else
                                                   "; mount: centre fixed by the attach model")
            + ("" if ev.get("zoom_judgeable", True) else "; zoom: focal fixed by the attach model"))
    if write and (res or nothing):
        p = probe_dir(take) / "camera_types_probe_b.json"
        old = read_probe(take, "b")
        old.update(res)
        for c in nothing:  # an older verdict for a camera the track cannot judge must not linger (circular evidence)
            old.pop(c, None)
        _write_json(p, {"version": 1, "take": take, "by": "probe_b", "at": now_iso(), "metric": sim3 is not None,
                        "cameras": old})
    return res


def _track_from_npz(work: Path, cam: str) -> dict | None:
    """cameras/<cam>.npz as a stage_smooth-like track when it holds today's general model (no type_model, or
    'per_frame'); None otherwise (a typed export would be circular evidence)."""
    import numpy as np
    p = work / "cameras" / f"{cam}.npz"
    if not p.exists():
        return None
    z = np.load(p)
    if "type_model" in z.files and str(z["type_model"]) != "per_frame":
        return None
    try:
        meta = json.loads((work / "frames" / cam / "frames.json").read_text())
        w, h = int(meta["width"]), int(meta["height"])
    except (OSError, ValueError, KeyError):
        K = np.asarray(z["K"])
        w, h = int(round(2 * (K[0, 0, 2] + 0.5))), int(round(2 * (K[0, 1, 2] + 0.5)))
    R, tv = np.asarray(z["R"], float), np.asarray(z["tvec"], float)
    f = np.asarray(z["K"], float)[:, 0, 0]
    ok = np.asarray(z["ok"], bool)
    return {"t": np.asarray(z["t"], float), "R": R, "tvec": tv, "C": -np.einsum("fji,fj->fi", R, tv), "f": f,
            "f_raw": f, "ok": ok, "ok_raw": ok, "maxdim": max(w, h), "w": w, "h": h,
            "attached": bool(z["attached"]) if "attached" in z.files else False}


def _c(x) -> str:
    return "-" if x is None else f"{x:.2f}"


def _f(x, k: float, unit: str, nd: int = 1) -> str:
    return "-" if x is None else f"{x * k:.{nd}f} {unit}"


def _write_json(p: Path, d: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    _replace_file(p, json.dumps(d, indent=1, default=_json_default).encode("utf-8"))


def _json_default(o):
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    return str(o)


# ------------------------------------------------------------------------------------------------ probe A (video)

NO_WINDOW = 0x08000000 if os.name == "nt" else 0
BELOW_NORMAL = 0x00004000 if os.name == "nt" else 0
A_TRIPOD_PARALLAX, A_HANDHELD_PARALLAX = 0.6, 1.5   # px at 640 px long side, 1-2 s spans
A_TRIPOD_JITTER, A_HANDHELD_JITTER = 0.12, 0.35     # px per frame at 15 Hz, > 2 Hz band
A_ZOOM_SCALE, A_NOZOOM_SCALE = 1.08, 1.03


def _ffprobe_dims(video: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height:stream_side_data=rotation:stream_tags=rotate:format=duration", "-of",
                          "json", str(video)], capture_output=True, text=True, timeout=60, creationflags=NO_WINDOW,
                         stdin=subprocess.DEVNULL, encoding="utf-8", errors="replace")
    d = json.loads(out.stdout or "{}")
    s = (d.get("streams") or [{}])[0]
    rot = 0
    for sd in s.get("side_data_list") or []:
        if "rotation" in sd:
            rot = int(round(float(sd["rotation"])))
    rot = rot or int((s.get("tags") or {}).get("rotate") or 0)
    w, h = int(s.get("width") or 0), int(s.get("height") or 0)
    dw, dh = (h, w) if rot % 180 else (w, h)
    try:
        dur = float((d.get("format") or {}).get("duration"))
    except (TypeError, ValueError):
        dur = None
    return {"display_w": dw, "display_h": dh, "duration_s": dur}


def _decode_window(video: Path, t0: float, dur: float, w: int, h: int, hz: float) -> "list":
    import numpy as np
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{t0:.3f}", "-t", f"{dur:.3f}", "-i", str(video),
           "-vf", f"fps={hz:g},scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    p = subprocess.run(cmd, capture_output=True, timeout=180, creationflags=NO_WINDOW | BELOW_NORMAL,
                       stdin=subprocess.DEVNULL)
    n = len(p.stdout) // (w * h)
    return [np.frombuffer(p.stdout, np.uint8, w * h, k * w * h).reshape(h, w) for k in range(n)]


PROBE_A_VERSION = 2  # bump when probe A measures differently (older results are not reused)
A_ZOOM_STEP = 0.08  # |log scale| between consecutive keyframes (~1-2 s): phone zoom steps are 10-100 %


def _sample_frames(video: Path, w: int, h: int, keyframes: bool, max_frames: int) -> tuple[list, list]:
    import numpy as np
    pre = ["-skip_frame", "nokey"] if keyframes else []
    vf = f"scale={w}:{h},showinfo" if keyframes else f"fps=1,scale={w}:{h},showinfo"
    cmd = ["ffmpeg", "-v", "info", "-nostdin", *pre, "-i", str(video), "-vf", vf, "-fps_mode", "passthrough",
           "-frames:v", str(max_frames), "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    p = subprocess.run(cmd, capture_output=True, timeout=900, creationflags=NO_WINDOW | BELOW_NORMAL,
                       stdin=subprocess.DEVNULL)
    ts = [float(m) for m in re.findall(rb"pts_time:\s*([0-9.]+)", p.stderr)]
    n = min(len(p.stdout) // (w * h), len(ts))
    return [np.frombuffer(p.stdout, np.uint8, w * h, k * w * h).reshape(h, w) for k in range(n)], ts[:n]


def keyframe_zoom_scan(video: Path, w: int, h: int, max_frames: int = 600, duration: float | None = None) -> dict:
    """Whole-video zoom scan, cheap: only the coded keyframes are decoded (-skip_frame nokey, a few % of a full
    decode); when they are sparser than one per 4 s (long GOP), one frame per second is sampled instead (full decode).
    Consecutive samples are matched (ORB, background prior mask) and their similarity scale chained.
    Returns {keyframes, sampling, steps: [[t0, t1, ratio]] with |log ratio| > A_ZOOM_STEP, max_step, scale_range}."""
    import cv2
    import numpy as np
    frames, ts = _sample_frames(video, w, h, True, max_frames)
    sampling = "keyframes"
    dur = duration or (ts[-1] if ts else 0.0)
    if dur and len(frames) < dur / 4.0:
        frames, ts = _sample_frames(video, w, h, False, max_frames)
        sampling = "1 Hz"
    n = len(frames)
    out: dict = {"keyframes": n, "sampling": sampling, "steps": [], "max_step": 0.0, "scale_range": 1.0, "breaks": 0}
    if n < 2:
        return out
    orb = cv2.ORB_create(1500)
    mask = _bg_mask(h, w)
    feats = [orb.detectAndCompute(f, mask) for f in frames]
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    def scale(a: int, b: int) -> float | None:
        (ka, da), (kb, db) = feats[a], feats[b]
        if da is None or db is None or len(ka) < 30 or len(kb) < 30:
            return None
        pairs = [x[0] for x in bf.knnMatch(da, db, k=2) if len(x) == 2 and x[0].distance < 0.75 * x[1].distance]
        if len(pairs) < 30:
            return None
        A = np.float32([ka[m.queryIdx].pt for m in pairs])
        B = np.float32([kb[m.trainIdx].pt for m in pairs])
        M, inl = cv2.estimateAffinePartial2D(A, B, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if M is None or inl is None or inl.sum() < 20:
            return None
        return math.log(max(1e-6, float(np.hypot(M[0, 0], M[1, 0]))))
    # chain: each sample against the nearest earlier sample already on the chain (up to 3 back), so a zoom across a
    # blurred / unmatched sample is still measured (larissa_kadu cam04: the 26 s sample matched nothing)
    cum: list[float | None] = [0.0] + [None] * (n - 1)
    for k in range(1, n):
        for j in range(k - 1, max(-1, k - 4), -1):
            if cum[j] is None:
                continue
            ls = scale(j, k)
            if ls is None:
                continue
            cum[k] = cum[j] + ls
            out["max_step"] = max(out["max_step"], abs(ls))
            if abs(ls) > A_ZOOM_STEP:
                out["steps"].append([round(ts[j], 2), round(ts[k], 2), round(math.exp(ls), 3)])
            break
        if cum[k] is None and all(c is None for c in cum[max(0, k - 3):k]):
            cum[k] = 0.0  # chain lost for good: restart (scale differences across the gap are not measured)
            out["breaks"] = out.get("breaks", 0) + 1
    vals = [c for c in cum if c is not None]
    out["linked"] = round(len(vals) / n, 3)  # share of samples on the chain (followed)
    out["max_step"] = round(math.exp(out["max_step"]), 4)
    out["scale_range"] = round(math.exp(max(vals) - min(vals)), 4) if vals else 1.0
    return out


def _bg_mask(h: int, w: int):
    """Background prior: the central region where dancers usually are is masked (RANSAC handles the rest)."""
    import numpy as np
    m = np.full((h, w), 255, np.uint8)
    yy, xx = np.ogrid[:h, :w]
    m[((xx - w / 2) / (0.27 * w)) ** 2 + ((yy - 0.55 * h) / (0.36 * h)) ** 2 <= 1.0] = 0
    return m


def _track(frames: list, mask) -> list:
    """Chained LK tracks with forward-backward check: list per frame of (ids, pts)."""
    import cv2
    import numpy as np
    lk = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    out = []
    pts = cv2.goodFeaturesToTrack(frames[0], 500, 0.01, 8, mask=mask)
    pts = np.zeros((0, 2), np.float32) if pts is None else pts.reshape(-1, 2)
    ids = np.arange(len(pts))
    nxt = len(pts)
    out.append((ids, pts))
    for k in range(1, len(frames)):
        if len(pts):
            p1, st, _ = cv2.calcOpticalFlowPyrLK(frames[k - 1], frames[k], pts.reshape(-1, 1, 2), None, **lk)
            p0, st2, _ = cv2.calcOpticalFlowPyrLK(frames[k], frames[k - 1], p1, None, **lk)
            good = (st.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm(p0.reshape(-1, 2) - pts, axis=1) < 0.5)
            ids, pts = ids[good], p1.reshape(-1, 2)[good]
        if len(pts) < 250:  # top up with new features away from the existing ones
            m2 = mask.copy()
            for x, y in pts.astype(int):
                cv2.circle(m2, (int(x), int(y)), 8, 0, -1)
            new = cv2.goodFeaturesToTrack(frames[k], 500 - len(pts), 0.01, 8, mask=m2)
            if new is not None:
                new = new.reshape(-1, 2)
                ids = np.concatenate([ids, np.arange(nxt, nxt + len(new))])
                pts = np.concatenate([pts, new]).astype(np.float32)
                nxt += len(new)
        out.append((ids, pts))
    return out


def _pairs(tracks: list, a: int, b: int):
    import numpy as np
    ia, pa = tracks[a]
    ib, pb = tracks[b]
    common, xa, xb = np.intersect1d(ia, ib, return_indices=True)
    return pa[xa], pb[xb]


def window_motion(frames: list, hz: float) -> dict:
    """Background motion statistics of one window of grey frames (see probe A in the module docstring)."""
    import cv2
    import numpy as np
    h, w = frames[0].shape
    mask = _bg_mask(h, w)
    tr = _track(frames, mask)
    steps, scales = [], []
    for k in range(1, len(frames)):
        a, b = _pairs(tr, k - 1, k)
        if len(a) < 20:
            steps.append([np.nan, np.nan, np.nan])
            scales.append(np.nan)
            continue
        M, inl = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=1.0, maxIters=2000)
        if M is None:
            steps.append([np.nan, np.nan, np.nan])
            scales.append(np.nan)
            continue
        s = float(np.hypot(M[0, 0], M[1, 0]))
        rot = float(np.arctan2(M[1, 0], M[0, 0]))
        c = np.array([w / 2, h / 2])
        d = M[:, :2] @ c + M[:, 2] - c  # motion of the image centre (px)
        steps.append([d[0], d[1], rot * w / 2])
        scales.append(math.log(s))
    S = np.asarray(steps, float)
    good = np.isfinite(S).all(1)
    out: dict = {"pairs": int(good.sum())}
    if good.sum() >= 8:
        k = max(3, int(round(0.5 * hz)) | 1)  # 0.5 s moving average = low-pass (< ~2 Hz)
        Sg = np.where(good[:, None], S, 0.0)
        ker = np.ones(k) / k
        wsum = np.convolve(good.astype(float), ker, "same")
        lp = np.stack([np.convolve(Sg[:, j], ker, "same") for j in range(3)], 1) / np.maximum(wsum, 1e-9)[:, None]
        hp = (S - lp)[good & (wsum > 0.6)]
        out["jitter_hf"] = float(np.sqrt(np.mean(np.sum(hp[:, :2] ** 2, 1)))) if len(hp) else None
        out["motion_px_s"] = float(np.median(np.linalg.norm(S[good, :2], axis=1)) * hz)
        out["pan_px"] = float(np.linalg.norm(np.nansum(S[good, :2], 0)))
    ls = np.asarray(scales, float)
    cum = np.nancumsum(np.where(np.isfinite(ls), ls, 0.0))
    out["zoom_log_range"] = float(cum.max() - cum.min()) if len(cum) else 0.0
    par = []
    for span in (int(hz), int(2 * hz)):
        for a0 in range(0, len(frames) - span, max(1, int(hz // 2))):
            pa, pb = _pairs(tr, a0, a0 + span)
            if len(pa) < 40:
                continue
            Fm, inl = cv2.findFundamentalMat(pa, pb, cv2.FM_RANSAC, 1.0, 0.999)
            if Fm is None or inl is None or inl.sum() < 30:
                continue
            sel = inl.ravel() == 1
            H, _ = cv2.findHomography(pa[sel], pb[sel], cv2.LMEDS)
            if H is None:
                continue
            pr = cv2.perspectiveTransform(pa[sel].reshape(-1, 1, 2), H).reshape(-1, 2)
            par.append(float(np.median(np.linalg.norm(pr - pb[sel], axis=1))))
    out["parallax_px"] = float(np.percentile(par, 75)) if par else None
    out["parallax_n"] = len(par)
    return out


class _BelowNormalThread:
    """Run the calling thread at below-normal priority (Windows; no-op elsewhere): probe A is called from the monitor
    server's worker threads and the narration tool, which must stay responsive on a 4-CPU machine."""

    def __enter__(self):
        self.h = self.old = None
        if os.name == "nt":
            try:
                import ctypes
                k32 = ctypes.windll.kernel32
                self.h = k32.GetCurrentThread()
                self.old = k32.GetThreadPriority(self.h)
                k32.SetThreadPriority(self.h, -1)  # THREAD_PRIORITY_BELOW_NORMAL
            except (OSError, AttributeError):
                self.h = None
        return self

    def __exit__(self, *exc):
        if self.h is not None and self.old is not None:
            try:
                import ctypes
                ctypes.windll.kernel32.SetThreadPriority(self.h, self.old)
            except (OSError, AttributeError):
                pass
        return False


def suggest_from_video(video: str | Path, probe: dict | None = None, **kw) -> dict | None:
    """Probe A (before calibration): mount / zoom suggestion from the background motion of a few windows of the video
    + a whole-video zoom scan. `probe` may carry ffprobe facts (display_w / display_h / duration_s, as
    dancecap.pipeline.probe writes them). Runs at below-normal priority."""
    p = probe or {}
    tiny = (float(p.get("duration_s") or 1e9) < 20.0
            and max(int(p.get("display_w") or 1e9), int(p.get("display_h") or 1e9)) <= 640)
    if tiny:  # a few frames of a small clip: no reason to yield (keeps the monitor's staging responsive)
        return _suggest_from_video(video, probe, **kw)
    with _BelowNormalThread():
        return _suggest_from_video(video, probe, **kw)


def _suggest_from_video(video: str | Path, probe: dict | None = None, windows: int = 5, win_s: float = 4.0,
                        hz: float = 15.0, long_side: int = 640, t_range: tuple[float, float] | None = None) -> dict | None:
    import numpy as np
    video = Path(video)
    if not video.exists():
        return None
    pr = dict(probe or {})
    if not (pr.get("display_w") and pr.get("display_h") and pr.get("duration_s")):
        try:
            pr.update(_ffprobe_dims(video))
        except (OSError, ValueError, subprocess.TimeoutExpired):
            return None
    dw, dh, dur = int(pr.get("display_w") or 0), int(pr.get("display_h") or 0), float(pr.get("duration_s") or 0)
    if not (dw and dh and dur > win_s):
        return None
    sc = min(1.0, long_side / max(dw, dh))  # never upscale a small video
    w, h = max(64, int(round(dw * sc / 2)) * 2), max(64, int(round(dh * sc / 2)) * 2)
    lo, hi = t_range or (0.0, dur)
    lo, hi = max(0.0, lo), min(dur, hi)
    if hi - lo < win_s:
        return None
    windows = max(1, min(windows, int((hi - lo) // (1.5 * win_s))))  # windows do not overlap on a short video
    starts = np.linspace(lo + 0.05 * (hi - lo), hi - win_s - 0.05 * (hi - lo), windows) if windows > 1 else [lo]
    per = []
    for t0 in starts:
        fr = _decode_window(video, float(max(lo, t0)), win_s, w, h, hz)
        if len(fr) < hz:
            continue
        m = window_motion(fr, hz)
        m["t0"] = round(float(t0), 2)
        per.append(m)
    if not per:
        return None
    try:
        zscan = keyframe_zoom_scan(video, w, h, duration=dur)
    except (OSError, subprocess.TimeoutExpired, ValueError):
        zscan = None
    return decide_a(per, w, h, hz, zscan=zscan)


def decide_a(per: list[dict], w: int, h: int, hz: float, at: str | None = None, zscan: dict | None = None) -> dict | None:
    """Probe-A suggestion from the per-window motion statistics and the keyframe zoom scan (re-scorable from the
    stored values)."""
    import numpy as np

    def q(k, p=50):
        v = [x[k] for x in per if x.get(k) is not None]
        return float(np.percentile(v, p)) if v else None
    # upper quartile over the windows: a hand-held phone shakes / moves in at least some windows (larissa_kadu cam03
    # looked still in 2 of 5), a tripod is still in all of them
    par, jit = q("parallax_px", 75), q("jitter_hf", 75)
    zr = float(np.exp(max(x.get("zoom_log_range", 0.0) for x in per)))
    pan = q("pan_px")
    ev = {"parallax_px": _r(par, 3), "jitter_hf": _r(jit, 4), "zoom_scale": _r(zr, 4), "pan_px": _r(pan, 1),
          "windows": len(per), "frame_px": [w, h], "hz": hz}
    mount = mconf = pans = None
    if par is not None and jit is not None:
        if par < A_TRIPOD_PARALLAX and jit < A_TRIPOD_JITTER:
            mount = "tripod"
            mconf = 0.7 + 0.15 * min(1.0, (A_TRIPOD_PARALLAX - par) / A_TRIPOD_PARALLAX)
            pans = bool(pan is not None and pan > 0.05 * max(w, h))
        elif par > A_HANDHELD_PARALLAX or jit > A_HANDHELD_JITTER:
            mount = "handheld"
            mconf = 0.7 + 0.15 * min(1.0, max((par - A_HANDHELD_PARALLAX) / A_HANDHELD_PARALLAX,
                                               (jit - A_HANDHELD_JITTER) / A_HANDHELD_JITTER, 0.0))
        else:  # between the bounds: the user decides. Pre-fill handheld (the prior: a tripod must be evidenced; phone
            # stabilisation hides hand shake, larissa_kadu 01 / 04 sit here; the parallax at room depth is weak)
            mount, mconf = "handheld", 0.5
    steps = (zscan or {}).get("steps") or []
    covered = bool(zscan and zscan.get("keyframes", 0) >= 10 and zscan.get("linked", 1.0) >= 0.8
                   and not zscan.get("breaks"))  # a restart of the chain can hide a zoom
    if zscan:
        ev.update(kf_keyframes=zscan.get("keyframes"), kf_max_step=zscan.get("max_step"), kf_linked=zscan.get("linked"), kf_breaks=zscan.get("breaks"),
                  kf_zoom_steps=len(steps), zoom_events_video_s=steps[:20])
    if zr > A_ZOOM_SCALE or steps:
        zoom = True
        zconf = 0.8 if steps else min(PROBE_A_MAX_CONF, 0.7 + 0.5 * (zr - A_ZOOM_SCALE))
    elif zr < A_NOZOOM_SCALE and (not zscan or (zscan.get("max_step") or 1.0) < 1.04):
        zoom, zconf = False, 0.8 if covered else 0.6  # the scan could not follow the whole video: user decides
    else:
        zoom, zconf = False, 0.5
    confs = [c for c in (mconf, zconf) if c is not None]
    if mount is None and zoom is None:
        return None
    return {"mount": mount, "zoom": zoom, "pans": pans if mount == "tripod" else None,
            "confidence": round(min(min(confs), PROBE_A_MAX_CONF), 3),
            "mount_confidence": _r(min(mconf, PROBE_A_MAX_CONF) if mconf is not None else None, 3),
            "zoom_confidence": _r(min(zconf, PROBE_A_MAX_CONF), 3), "by": "probe_a", "at": at or now_iso(),
            "evidence": ev, "windows": per, "keyframe_scan": zscan, "version": PROBE_A_VERSION}


def rescore_a(take: str, log=print) -> dict[str, dict]:
    """Re-derive the stored probe-A suggestions of a take from their stored windows with the current thresholds
    (no video decoding); keeps file signatures and times."""
    p = probe_dir(take) / "camera_types_suggested.json"
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out = {}
    for cam, s in (doc.get("cameras") or {}).items():
        ev = s.get("evidence") or {}
        if not s.get("windows") or not ev.get("frame_px"):
            out[cam] = s
            continue
        n = decide_a(s["windows"], *ev["frame_px"], ev.get("hz", 15.0), at=s.get("at"), zscan=s.get("keyframe_scan"))
        if n is None:
            continue
        for k in ("file", "reused_from"):
            if k in s:
                n[k] = s[k]
        n["version"] = s.get("version", 1)  # the measurement is as old as before: rescoring does not upgrade it
        out[cam] = n
        log(f"[camera_types] probe A {cam} (rescored): {n['mount']} ({_c(n['mount_confidence'])}), "
            f"{'zoom' if n['zoom'] else 'no zoom'} ({_c(n['zoom_confidence'])})")
    doc["cameras"] = out
    _write_json(p, doc)
    return out


# aliases the monitor's adapter looks for
probe_a = suggest_from_video
suggest = suggest_from_video


def _r(x, nd):
    return None if x is None else round(float(x), nd)


def _file_sig(p: Path) -> dict | None:
    try:
        st = p.stat()
    except OSError:
        return None
    return {"name": p.name, "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def _probe_a_reuse(sig: dict) -> dict | None:
    """A probe-A result of the same video file (name, size, mtime) in any take's work/<take>/probe/."""
    for p in sorted((root() / "work").glob("*/probe/camera_types_suggested.json")):
        try:
            cams = json.loads(p.read_text(encoding="utf-8")).get("cameras", {})
        except (OSError, ValueError):
            continue
        for s in cams.values():
            if isinstance(s, dict) and s.get("file") == sig and s.get("version") == PROBE_A_VERSION:
                return dict(s, reused_from=p.parent.parent.name)
    return None


def probe_a_take(take: str, cams: list[str] | None = None, write: bool = True, log=print, **kw) -> dict[str, dict]:
    """Probe A for every used source video of a take -> work/<take>/probe/camera_types_suggested.json."""
    cfg, _ = read_take(take)
    sdir = cfg.get("source_dir")
    sroot = (root() / "takes" / sdir).resolve() if sdir and not Path(sdir).is_absolute() else Path(sdir or ".")
    files = {}
    for r in _alignment(cfg).get("sources", []) or []:
        if CAM_RE.match(str(r.get("id", ""))) and r.get("name"):
            files[str(r["id"])] = sroot / r["name"]
    for cam, t in cam_tables(cfg).items():
        if isinstance(t.get("file"), str):
            files.setdefault(cam, sroot / t["file"])
    use = [c for c in known_cams(take, cfg) if c in files and (not cams or c in cams)]
    res = {}
    for cam in use:
        t0 = time.time()
        sig = _file_sig(files[cam])
        s = _probe_a_reuse(sig) if sig and not kw else None  # the same video probed for another take
        if s is None:
            s = suggest_from_video(files[cam], **kw)
            if s is not None and sig:
                s["file"] = sig
        if s is None:
            log(f"[camera_types] probe A {cam}: no result ({files[cam].name})")
            continue
        res[cam] = s
        ev = s["evidence"]
        log(f"[camera_types] probe A {cam}: {s['mount']} ({_c(s['mount_confidence'])}), "
            f"{'zoom' if s['zoom'] else 'no zoom'} ({_c(s['zoom_confidence'])}); parallax {ev['parallax_px']} px, "
            f"jitter {ev['jitter_hf']} px, zoom scale x{ev['zoom_scale']} ({time.time() - t0:.0f} s)")
    if write and res:
        old = read_probe(take, "a")
        old.update(res)
        _write_json(probe_dir(take) / "camera_types_suggested.json",
                    {"version": 1, "take": take, "by": "probe_a", "at": now_iso(), "cameras": old})
    return res


# ------------------------------------------------------------------------------------------------ CLI

def _print_types(take: str, as_json: bool) -> None:
    types, etag = load(take)
    if as_json:
        print(json.dumps({"take": take, "toml_etag": etag, "types": types}, indent=1, default=_json_default))
        return
    print(f"{take}: " + ", ".join(f"{v} {k}" for k, v in summary(types).items() if v))
    for t in types:
        s = t.get("suggestion") or {}
        sug = (f"{s.get('mount') or '?'}/{'zoom' if s.get('zoom') else 'no zoom' if s.get('zoom') is False else '?'} "
               f"{_c(s.get('confidence'))} by {s.get('by')}") if s else "-"
        print(f"  {t['cam']}  {t['state']:<9}  answer {t['mount'] or '-'}/"
              f"{'-' if t['zoom'] is None else 'zoom' if t['zoom'] else 'no zoom'}  effective "
              f"{t['effective']['class']:<13}  suggestion {sug}"
              + (f"\n      CONFLICT: {t['conflict']['message']}" if t.get("conflict") else ""))


def _bool(s: str) -> bool:
    v = s.strip().lower()
    if v in ("yes", "true", "1", "zoom", "y"):
        return True
    if v in ("no", "false", "0", "n", "none"):
        return False
    raise argparse.ArgumentTypeError("yes | no")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.camera_types", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("show")
    p.add_argument("take")
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("suggest")
    p.add_argument("take")
    p.add_argument("--cams")
    p.add_argument("--probe", choices=["a", "b", "both"], default="both")
    p.add_argument("--write-toml", action="store_true", help="also record the suggestions in the take TOML (source = "
                                                             "auto; never over a user answer)")
    p.add_argument("--rescore", action="store_true", help="probe A: re-derive the stored suggestions from their "
                                                          "stored windows with the current thresholds (no decoding)")
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("set")
    p.add_argument("take")
    p.add_argument("cam")
    p.add_argument("--mount", required=True, choices=MOUNTS)
    p.add_argument("--zoom", required=True, type=_bool)
    p = sub.add_parser("accept")
    p.add_argument("take")
    p.add_argument("--cams")
    p.add_argument("--min-confidence", type=float, default=MIN_CONFIDENCE)
    p = sub.add_parser("clear")
    p.add_argument("take")
    p.add_argument("cam")
    a = ap.parse_args(argv)
    cams = a.cams.split(",") if getattr(a, "cams", None) else None
    if a.cmd == "show":
        _print_types(a.take, a.json)
    elif a.cmd == "suggest":
        res = {}
        if a.probe in ("a", "both"):
            res["a"] = rescore_a(a.take) if a.rescore else probe_a_take(a.take, cams)
        if a.probe in ("b", "both"):
            if (work_dir(a.take) / "cameras" / "cache").exists():
                res["b"] = probe_b(a.take, cams)
            else:
                print("[camera_types] probe B: no camera tracks yet (cameras stage)")
        if a.write_toml:
            types, _ = load(a.take, cams)
            for t in types:
                s = t.get("suggestion")
                if s and (s.get("mount") in MOUNTS and isinstance(s.get("zoom"), bool)):
                    write_suggestion(a.take, t["cam"], s)
        if a.json:
            print(json.dumps(res, indent=1, default=_json_default))
        _print_types(a.take, False)
    elif a.cmd == "set":
        t, _ = set_type(a.take, a.cam, a.mount, a.zoom)
        print(f"{a.take} {a.cam}: {t['mount']} / {'zoom' if t['zoom'] else 'no zoom'} (confirmed)")
    elif a.cmd == "accept":
        types, _ = accept(a.take, cams, a.min_confidence)
        _print_types(a.take, False)
    elif a.cmd == "clear":
        clear(a.take, a.cam)
        _print_types(a.take, False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
