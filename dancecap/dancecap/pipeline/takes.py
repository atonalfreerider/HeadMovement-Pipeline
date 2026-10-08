"""Tolerant take reader for the pipeline backend and the monitor.

Never `Take.load` (it requires `alignment` and `window`, which a new take only gets from its clock stages): this
reads `takes/<take>.toml` with tomllib and pulls what the monitor shows. The video list is the union of
- `[cameras."NN"] file = "..."` (written by the new-capture wizard),
- the alignment's `sources` (existing takes: id + name, relative to `source_dir`),
- `work/<take>/intake.json` `videos` (written at creation; survives any rewrite of the camera blocks),
with `exclude` applied.
"""
from __future__ import annotations

import math
import tomllib
from pathlib import Path

from . import camtypes, paths
from .util import CAM_RE, TAKE_RE, mtime, read_json, sha1_bytes

LEVELS = ("professional", "intermediate", "novice")
CATEGORIES = ("demo", "lesson", "jack_and_jill")
ROLES = ("lead", "follow")
CONSENT_SCOPES = ("capture", "fit_3d", "ml_analysis", "cloud_processing", "name_shown", "level_shown", "publication",
                  "public_repo")
VIDEO_EXT = (".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm", ".3gp")
AUDIO_EXT = (".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus")
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".webp")


class TakeError(ValueError):
    pass


def list_takes() -> list[str]:
    d = paths.takes_dir()
    if not d.exists():
        return []
    return sorted(p.stem for p in d.glob("*.toml") if TAKE_RE.match(p.stem))


def exists(take: str) -> bool:
    return bool(TAKE_RE.match(take or "")) and paths.take_toml(take).exists()


def toml_bytes(take: str) -> bytes:
    try:
        return paths.take_toml(take).read_bytes()
    except OSError:
        return b""


def load_cfg(take: str) -> tuple[dict, str]:
    """(parsed TOML or {} plus a '_error' key, etag)."""
    if not TAKE_RE.match(take or ""):
        raise TakeError("invalid take name")
    b = toml_bytes(take)
    if not b:
        raise TakeError(f"no take {take}")
    try:
        cfg = tomllib.loads(b.decode("utf-8-sig"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as e:
        cfg = {"_error": f"TOML parse error: {e}"}
    return cfg, sha1_bytes(b)


def _rel_to_takes(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else (paths.takes_dir() / q).resolve()


def source_dir(cfg: dict) -> Path | None:
    s = cfg.get("source_dir")
    return _rel_to_takes(s) if s else None


def alignment_path(cfg: dict) -> Path | None:
    a = cfg.get("alignment")
    return _rel_to_takes(a) if a else None


def alignment(cfg: dict) -> dict | None:
    p = alignment_path(cfg)
    return read_json(p) if p and p.exists() else None


def intake_json(take: str) -> dict:
    return read_json(paths.take_work(take) / "intake.json", {}) or {}


def videos(take: str, cfg: dict) -> list[dict]:
    """[{cam, file, path (Path|None), excluded, exclude_reason, clock}] sorted by cam."""
    out: dict[str, dict] = {}
    sdir = source_dir(cfg)
    for cam, t in camtypes.cam_tables(cfg).items():
        f = t.get("file")
        if isinstance(f, str) and f:
            out.setdefault(cam, {"cam": cam, "file": f})
    al = alignment(cfg) or {}
    ref = al.get("reference")
    for r in al.get("sources", []) or []:
        cam = str(r.get("id", ""))
        if not CAM_RE.match(cam):
            continue
        v = out.setdefault(cam, {"cam": cam, "file": r.get("name")})
        v.setdefault("file", r.get("name"))
        v["clock"] = {"slope": r.get("slope"), "offset": r.get("offset"), "reference": r.get("name") == ref}
        if r.get("accepted") is False:
            v["exclude_reason"] = "audio clock not accepted"
    for r in intake_json(take).get("videos", []) or []:
        cam = str(r.get("cam", ""))
        if CAM_RE.match(cam):
            v = out.setdefault(cam, {"cam": cam, "file": r.get("file")})
            v.setdefault("file", r.get("file"))
            if r.get("path") and not v.get("abs"):
                v["abs"] = r.get("path")
    excl = {str(x) for x in cfg.get("exclude", []) or []}
    res = []
    for cam in sorted(out):
        v = out[cam]
        f = v.get("file")
        p = None
        if v.get("abs"):
            p = Path(v.pop("abs"))
        elif f and sdir:
            p = sdir / f
        v["path"] = p
        v["excluded"] = cam in excl or bool(v.get("exclude_reason"))
        if cam in excl:
            v["exclude_reason"] = v.get("exclude_reason") or "excluded in the take TOML"
        v.setdefault("clock", None)
        res.append(v)
    return res


def used_cams(take: str, cfg: dict) -> list[str]:
    return [v["cam"] for v in videos(take, cfg) if not v["excluded"]]


def soundtrack(cfg: dict) -> dict | None:
    d = cfg.get("dance") or {}
    f = d.get("song_file")
    if not f:
        return None
    p = _rel_to_takes(f)
    return {"name": p.name, "present": p.exists(), "path": p}


def dancers(cfg: dict) -> dict:
    out = {}
    for role in ROLES:
        d = (cfg.get("dancers") or {}).get(role) or {}
        if not d:
            continue
        port = d.get("portrait")
        out[role] = {"name": d.get("name"), "stature_m": d.get("stature_m"), "shoe_sole_m": d.get("shoe_sole_m"),
                     "description": d.get("description"), "portrait": bool(port), "pid": d.get("pid")}
    return out


def consent(cfg: dict) -> dict:
    c = cfg.get("consent") or {}
    out = {}
    for role in ROLES:
        r = c.get(role) or {}
        out[role] = {k: (r[k] if isinstance(r.get(k), bool) else "?") for k in CONSENT_SCOPES}
    out["source_rights"] = c.get("source_rights", "?")
    out["bystanders"] = c.get("bystanders", "?")
    return out


def cloud_consent(cfg: dict) -> dict:
    c = consent(cfg)
    roles = [r for r in ROLES if r in (cfg.get("dancers") or {})] or list(ROLES)
    per = {r: c[r]["cloud_processing"] is True for r in roles}
    return {"cloud_processing": per, "ok": all(per.values())}


def title(cfg: dict) -> str:
    ds = dancers(cfg)
    lead = (ds.get("lead") or {}).get("name") or "Leader"
    follow = (ds.get("follow") or {}).get("name") or "Follower"
    song = (cfg.get("dance") or {}).get("song_display") or (cfg.get("dance") or {}).get("song") or "?"
    return f"{lead} + {follow} x {song}"


def window(cfg: dict) -> tuple[float, float] | None:
    """[t0, t1] reference seconds; None unless both are finite numbers (never a bool, never inf/nan) with t0 < t1."""
    w = cfg.get("window")
    if isinstance(w, list) and len(w) == 2 and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                                                   and math.isfinite(x) for x in w) and w[0] < w[1]:
        return float(w[0]), float(w[1])
    return None


def variant(take: str, cfg: dict) -> str:
    forced = ((cfg.get("pipeline") or {}).get("variant") or "").strip()
    n = len(used_cams(take, cfg))
    auto = "single" if n <= 1 else "two" if n == 2 else "multi"
    order = {"single": 1, "two": 2, "multi": 3}
    if forced in order and order[forced] <= order[auto]:  # only downwards (PIPELINE_MONITOR 6.3)
        return forced
    return auto


def exports(take: str) -> list[dict]:
    d = paths.take_work(take) / "export"
    out = []
    for p in sorted(d.glob("*_demo_export.json")) if d.exists() else []:
        rep = read_json(p, {}) or {}
        ok = rep.get("ok")
        if ok is None and isinstance(rep.get("validation"), dict):
            ok = rep["validation"].get("ok")
        out.append({"capture": p.name[: -len("_demo_export.json")], "ok": ok, "at": mtime(p)})
    return sorted(out, key=lambda e: e["at"] or 0)


def last_capture(take: str) -> str | None:
    ex = exports(take)
    return ex[-1]["capture"] if ex else None
