"""Camera type per source video (mount = handheld | tripod, zoom = yes | no): the pipeline backend's adapter.

The contract (PIPELINE.md 4.1, PIPELINE_MONITOR.md 7.3-7.4) is owned by the shared module `dancecap.camera_types`
(camera-types builder): the TOML keys, the `CameraType` JSON shape, the owned-block writer and the automatic
suggestion (probe A / probe B). This adapter ALWAYS delegates to that module when it is importable - on the real
package root and on a test tree alike (the module's root is pointed at paths.root() for the call), so the tests
exercise exactly what production runs. Only when the module cannot be imported does the small fallback below run;
it follows the same contract and the same semantics (EFFECTIVE type = the user's answer only: a suggestion pre-fills
the review screen but the camera stages apply handheld + zoom until the user confirms):

    [cameras."05"]            # one table per source id, beside [cameras] scale_rule / scale_reference / loc
    mount = "tripod"          # handheld | tripod                  (user)
    zoom = false              # true | false, independent of mount (user)
    source = "user"           # user = confirmed | auto = suggestion, not confirmed yet
    set_at = "2026-10-07T18:02:11Z"
    suggested = { mount = "tripod", zoom = false, pans = false, confidence = 0.64, by = "probe_a", at = "..." }
    pans = false              # tripod only, automatic
    zoom_events = []          # automatic: [t0, t1, focal ratio] in reference seconds

Rules: a missing type means handheld + zoom (today's solver, so old takes are unchanged); the user always wins; a
data conflict never overrides the user, it sets state "conflict" and opens the camera_types gate.
Only the `[cameras."NN"]` blocks are rewritten; every byte outside them is kept. Keys of the block that this
adapter does not own (e.g. the wizard's `file`) are re-emitted unchanged.
"""
from __future__ import annotations

import json
import re
import threading
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path

from . import paths
from .util import CAM_RE, sha1_bytes, write_text_atomic

MOUNTS = ("handheld", "tripod")
DEFAULT = {"mount": "handheld", "zoom": True}
MIN_CONFIDENCE = 0.7
OWNED_KEYS = ("mount", "zoom", "source", "set_at", "suggested", "pans", "zoom_events", "conflict")

CLASS_EFFECTS = {
    "tripod": ["one pose for the whole video (keyframe localisation, ~1 frame pair per 2 s)", "one intrinsic set",
               "a constant camera track for sync and the dancers"],
    "tripod_zoom": ["fixed camera centre; per-frame rotation + focal",
                    "zoom-depth guard off (no axis ambiguity with a known centre)"],
    "handheld": ["per-frame 6-dof pose", "one intrinsic set per video (one focal + k1)"],
    "handheld_zoom": ["per-frame pose + focal (P4Pf/P3P race)", "log-focal smoothing, zoom-depth guard (today's solver)"],
}
# cam_loc work factor per class, as IMPLEMENTED (cameras/typed.py): only a LOCKED tripod (no zoom, no pans) is
# localised on keyframes (one frame pair per 2 s + the map's frames, ~10 %); every other class localises every frame
# (the type constrains the solve after localisation, not the matching). p90: a locked tripod whose keyframes refuse the
# constant pose is localised twice (keyframes, then every frame: ~1.05-1.1).
LOC_FACTOR = {"tripod": 0.1, "tripod_pans": 1.0, "tripod_zoom": 1.0, "handheld": 1.0, "handheld_zoom": 1.0}
LOC_FACTOR_P90 = {"tripod": 1.1, "tripod_pans": 1.0, "tripod_zoom": 1.0, "handheld": 1.0, "handheld_zoom": 1.0}


class CameraTypeError(ValueError):
    pass


class StaleEtag(RuntimeError):
    def __init__(self, etag: str):
        super().__init__("the take TOML changed since it was read")
        self.etag = etag


# ------------------------------------------------------------------------------------------------ shared module

_SHARED_LOCK = threading.RLock()


def _shared_module():
    try:
        from .. import camera_types as ct  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        return None
    return ct if all(hasattr(ct, n) for n in ("load", "set_type", "accept", "set_root", "root")) else None


class _shared:
    """with _shared() as ct: the shared module with its root pointed at paths.root() (None: not importable).
    The root is restored afterwards; the lock serialises calls that need another root (tests)."""

    def __enter__(self):
        self.ct = _shared_module()
        self.prev = None
        if self.ct is None:
            return None
        _SHARED_LOCK.acquire()
        try:
            cur = Path(self.ct.root()).resolve()
            if cur != paths.root():
                self.prev = cur
                self.ct.set_root(paths.root())
        except Exception:  # noqa: BLE001
            _SHARED_LOCK.release()
            self.ct = None
            return None
        return self.ct

    def __exit__(self, et, e, tb):
        if self.ct is not None:
            try:
                if self.prev is not None:
                    self.ct.set_root(self.prev)
            finally:
                _SHARED_LOCK.release()
        return False


class _mapped_errors:
    """The shared module's errors as this adapter's: a stale etag -> StaleEtag, bad values -> CameraTypeError."""

    def __enter__(self):
        return self

    def __exit__(self, et, e, tb):
        if e is None or isinstance(e, (StaleEtag, CameraTypeError)):
            return False
        name = type(e).__name__.lower()
        if "stale" in name or "etag" in name or "precondition" in name:
            raise StaleEtag(str(getattr(e, "etag", "") or etag_of(b""))) from e
        if isinstance(e, (ValueError, KeyError, TypeError)):
            raise CameraTypeError(str(e)) from e
        return False


def class_of(mount: str | None, zoom: bool | None) -> str:
    m = mount if mount in MOUNTS else DEFAULT["mount"]
    z = DEFAULT["zoom"] if zoom is None else bool(zoom)
    return f"{m}_zoom" if z else m


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _iso_to_epoch(v) -> float | None:
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, datetime):
        return v.replace(tzinfo=v.tzinfo or timezone.utc).timestamp()
    if isinstance(v, str):
        try:
            return datetime.fromisoformat(v.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None


def cam_tables(cfg: dict) -> dict[str, dict]:
    cams = cfg.get("cameras") or {}
    return {k: v for k, v in cams.items() if CAM_RE.match(str(k)) and isinstance(v, dict)}


def probe_b(take: str) -> dict:
    """probe/camera_types_probe_b.json (written after cam_world by the camera-types builder), {cam: suggestion}."""
    from .util import read_json
    d = read_json(paths.take_work(take) / "probe" / "camera_types_probe_b.json", {}) or {}
    cams = d.get("cameras", d) if isinstance(d, dict) else {}
    return {k: v for k, v in cams.items() if CAM_RE.match(str(k)) and isinstance(v, dict)}


def probe_a(take: str) -> dict:
    from .util import read_json
    d = read_json(paths.take_work(take) / "probe" / "camera_types_suggested.json", {}) or {}
    cams = d.get("cameras", d) if isinstance(d, dict) else {}
    return {k: v for k, v in cams.items() if CAM_RE.match(str(k)) and isinstance(v, dict)}


def describe(cam: str, table: dict | None, suggestion: dict | None = None, measured: dict | None = None) -> dict:
    """CameraType JSON (PIPELINE_MONITOR 7.3) from the TOML table, an optional probe-file suggestion and probe B."""
    t = dict(table or {})
    sug = t.get("suggested") if isinstance(t.get("suggested"), dict) else None
    if suggestion and (sug is None or (_iso_to_epoch(suggestion.get("at")) or 0) > (_iso_to_epoch(sug.get("at")) or 0)):
        sug = suggestion
    source = t.get("source")
    mount = t.get("mount") if t.get("mount") in MOUNTS else None
    zoom = t.get("zoom") if isinstance(t.get("zoom"), bool) else None
    if source == "user" and mount is not None and zoom is not None:
        state = "confirmed"
    elif sug is not None or source == "auto":
        state = "suggested"
        if mount is None and sug:
            mount = sug.get("mount") if sug.get("mount") in MOUNTS else None
        if zoom is None and sug and isinstance(sug.get("zoom"), bool):
            zoom = sug.get("zoom")
        source = "auto"
    else:
        state, source = "unset", "default"
    conflict = t.get("conflict") if isinstance(t.get("conflict"), dict) else None
    m = measured or {}
    if state == "confirmed" and conflict is None and m and float(m.get("confidence", 0) or 0) >= MIN_CONFIDENCE:
        for field in ("mount", "zoom"):
            if field in m and m[field] is not None and m[field] != (mount if field == "mount" else zoom):
                conflict = {"field": field, "user": mount if field == "mount" else zoom, "measured": m[field],
                            "message": f"you said {field} = {mount if field == 'mount' else zoom}; the camera track "
                                       f"says {m[field]} (probe_b, confidence {float(m.get('confidence', 0)):.2f})",
                            "evidence": m.get("evidence")}
                break
    if conflict:
        state = "conflict"
    # EFFECTIVE = what the camera stages apply: the user's answer only (shared module, policy `user`)
    eff_m = mount if state in ("confirmed", "conflict") and mount else DEFAULT["mount"]
    eff_z = zoom if state in ("confirmed", "conflict") and zoom is not None else DEFAULT["zoom"]
    cls = class_of(eff_m, eff_z)
    effects = list(CLASS_EFFECTS[cls])
    if cls == "tripod" and t.get("pans"):
        effects[0] = "fixed centre, per-frame rotation only (pans)"
    return {"cam": cam, "mount": mount, "zoom": zoom, "source": source, "state": state,
            "set_at": _iso_to_epoch(t.get("set_at")), "pans": t.get("pans"), "zoom_events": t.get("zoom_events") or [],
            "effective": {"mount": eff_m, "zoom": eff_z, "class": cls},
            "suggestion": _suggestion_out(sug), "conflict": conflict, "effects": effects, "model_check": None}


def _suggestion_out(s: dict | None) -> dict | None:
    if not s:
        return None
    out = {k: s.get(k) for k in ("mount", "zoom", "pans", "confidence", "by")}
    out["at"] = _iso_to_epoch(s.get("at"))
    out["evidence"] = s.get("evidence") or {}
    return out


def needs_user(ct: dict) -> bool:
    """True when this camera blocks its camera stages (PIPELINE.md 4.2 'uncertain', or a conflict)."""
    if ct["state"] == "conflict":
        return True
    if ct["state"] == "confirmed":
        return False
    s = ct.get("suggestion") or {}
    return float(s.get("confidence") or 0) < MIN_CONFIDENCE


def effective_class(ct: dict, policy: str | None = None) -> str:
    """Class a camera stage applies to this camera under a policy (user | auto | off; None = DANCECAP_CAMERA_TYPES or
    'user'): exactly dancecap.camera_types.stage_type, so the monitor compares like with like."""
    shared = _shared_module()
    if shared is not None and hasattr(shared, "stage_type"):
        try:
            return shared.stage_type(ct, shared.resolve_policy(policy)).cls
        except Exception:  # noqa: BLE001 - a bad policy string: the default
            return shared.stage_type(ct, "user").cls
    pol = (policy or "user").lower()
    if pol == "off":
        return class_of(None, None)
    if ct.get("state") in ("confirmed", "conflict"):
        return class_of(ct.get("mount"), ct.get("zoom"))
    s = ct.get("suggestion") or {}
    if pol == "auto" and ct.get("state") == "suggested" and float(s.get("confidence") or 0) >= MIN_CONFIDENCE \
            and s.get("mount") in MOUNTS and isinstance(s.get("zoom"), bool):
        return class_of(s["mount"], s["zoom"])
    return class_of(None, None)


def loc_factor(ct: dict, conservative: bool = False, policy: str | None = None) -> float:
    """cam_loc work factor of the class the stage applies (policy as effective_class); conservative=True is the p90
    (an unconfirmed camera priced as handheld + zoom; a tripod whose keyframes refuse the constant pose localised
    twice)."""
    if conservative and ct["state"] != "confirmed":
        return LOC_FACTOR_P90["handheld_zoom"]
    cls = effective_class(ct, policy) if policy is not None else ct["effective"]["class"]
    if cls == "tripod" and ct.get("pans"):
        cls = "tripod_pans"
    return (LOC_FACTOR_P90 if conservative else LOC_FACTOR)[cls]


# ------------------------------------------------------------------------------------------------ reading

def toml_bytes(take: str) -> bytes:
    try:
        return paths.take_toml(take).read_bytes()
    except OSError:
        return b""


def etag_of(b: bytes) -> str:
    return sha1_bytes(b)


def load(take: str, cams: list[str]) -> tuple[list[dict], str]:
    """CameraType per camera id (in the given order) and the TOML etag."""
    with _shared() as ct:
        if ct is not None:
            try:
                types, etag = ct.load(take, list(cams))
                if all(isinstance(t, dict) and {"cam", "state", "effective", "source"} <= set(t) and
                       "class" in (t.get("effective") or {}) for t in types):
                    return types, etag
            except Exception:  # noqa: BLE001 - a broken TOML reads as "no types" (the take shows its error)
                pass
    return _load_local(take, cams)


def _load_local(take: str, cams: list[str]) -> tuple[list[dict], str]:
    b = toml_bytes(take)
    try:
        cfg = tomllib.loads(b.decode("utf-8-sig")) if b else {}
    except (tomllib.TOMLDecodeError, UnicodeDecodeError):
        cfg = {}
    tabs = cam_tables(cfg)
    pa, pb = probe_a(take), probe_b(take)
    return [describe(c, tabs.get(c), pa.get(c), pb.get(c)) for c in cams], etag_of(b)


# ------------------------------------------------------------------------------------------------ writing

_HEADER = re.compile(r'^\s*\[\s*cameras\s*\.\s*(?:"(\d{2})"|\'(\d{2})\'|(\d{2}))\s*\]\s*(?:#.*)?$')
_ANY_HEADER = re.compile(r"^\s*\[")


def _toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return repr(round(v, 6))
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, datetime):
        return json.dumps(v.isoformat())
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{ " + ", ".join(f"{_key(k)} = {_toml_value(x)}" for k, x in v.items() if x is not None) + " }"
    raise CameraTypeError(f"cannot write {type(v).__name__} to TOML")


def _key(k: str) -> str:
    return k if re.fullmatch(r"[A-Za-z0-9_-]+", k) else json.dumps(k)


def render_block(cam: str, t: dict) -> str:
    lines = [f'[cameras."{cam}"]   # camera type (Pipeline Monitor / narration tool); this block is rewritten as a '
             f'whole, comments typed inside it are not kept']
    for k, v in t.items():  # keys we do not own first (e.g. the wizard's file = "...")
        if k not in OWNED_KEYS and v is not None and not isinstance(v, dict):
            lines.append(f"{_key(k)} = {_toml_value(v)}")
    if t.get("mount") is not None:
        lines.append(f'mount = {_toml_value(t["mount"])}            # handheld | tripod')
    if t.get("zoom") is not None:
        lines.append(f'zoom = {_toml_value(t["zoom"])}              # true | false, independent of mount')
    if t.get("source"):
        lines.append(f'source = {_toml_value(t["source"])}           # user = confirmed | auto = suggestion only')
    if t.get("set_at"):
        lines.append(f"set_at = {_toml_value(t['set_at'])}")
    if isinstance(t.get("conflict"), dict):
        lines.append(f"conflict = {_toml_value(t['conflict'])}")
    auto = [k for k in ("suggested", "pans", "zoom_events") if t.get(k) is not None]
    if auto:
        lines.append("# written by the probe, kept for the record; never overrides the user:")
        for k in auto:
            lines.append(f"{k} = {_toml_value(t[k])}")
    return "\n".join(lines) + "\n"


def replace_blocks(text: str, blocks: dict[str, str]) -> str:
    """Rewrite each [cameras."NN"] block (header to the next header or EOF) or append it at the end."""
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    done: set[str] = set()
    i = 0
    while i < len(lines):
        m = _HEADER.match(lines[i].rstrip("\r\n"))
        cam = next((g for g in m.groups() if g), None) if m else None
        if cam is not None and cam in blocks:
            j = i + 1
            while j < len(lines) and not _ANY_HEADER.match(lines[j]):
                j += 1
            # keep trailing blank/comment lines that belong to the next section's preamble
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
    rest = [c for c in blocks if c not in done]
    if rest:
        if out and not out[-1].endswith("\n"):
            out.append("\n")
        for c in sorted(rest):
            if out and out[-1].strip():
                out.append("\n")
            out.append(blocks[c])
    return "".join(out)


def _write(take: str, updates: dict[str, dict], if_match: str | None) -> str:
    """Owned-block write with the etag check, a parse check, a backup and an atomic replace. Returns the new etag."""
    path = paths.take_toml(take)
    b = toml_bytes(take)
    if not b:
        raise CameraTypeError(f"no take TOML for {take}")
    cur = etag_of(b)
    if if_match is not None and if_match.strip('"') != cur:
        raise StaleEtag(cur)
    text = b.decode("utf-8-sig")
    cfg = tomllib.loads(text)
    tabs = cam_tables(cfg)
    blocks = {}
    for cam, upd in updates.items():
        t = dict(tabs.get(cam, {}))
        for k, v in upd.items():
            if v is None:
                t.pop(k, None)
            else:
                t[k] = v
        blocks[cam] = render_block(cam, t)
    new = replace_blocks(text, blocks)
    parsed = tomllib.loads(new)  # raises on a broken file: never written
    ptabs = cam_tables(parsed)
    for cam, upd in updates.items():
        for k, v in upd.items():
            if v is not None and k in ("mount", "zoom", "source") and ptabs.get(cam, {}).get(k) != v:
                raise CameraTypeError(f"write check failed for {cam}.{k}")
    for k in ("name", "source_dir", "alignment", "window", "dance", "dancers", "sync"):
        if cfg.get(k) != parsed.get(k):
            raise CameraTypeError(f"write check failed: {k} changed")
    base = {k: v for k, v in (cfg.get("cameras") or {}).items() if not CAM_RE.match(str(k))}
    if base != {k: v for k, v in (parsed.get("cameras") or {}).items() if not CAM_RE.match(str(k))}:
        raise CameraTypeError("write check failed: reserved [cameras] keys changed")
    bdir = paths.take_work(take) / "monitor" / "toml_backups"
    bdir.mkdir(parents=True, exist_ok=True)
    now = time.time()
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(now)) + f"-{int(1000 * (now % 1)):03d}"
    for k in range(1000):  # never overwrite an earlier backup
        try:
            with open(bdir / (f"{stamp}-{take}.toml" if k == 0 else f"{stamp}-{k}-{take}.toml"), "xb") as fh:
                fh.write(b)
            break
        except FileExistsError:
            continue
    write_text_atomic(path, new)
    return etag_of(new.encode("utf-8"))


def set_type(take: str, cam: str, mount: str, zoom: bool, if_match: str | None) -> tuple[dict, str]:
    if not CAM_RE.match(cam or ""):
        raise CameraTypeError("cam must be two digits")
    if mount not in MOUNTS:
        raise CameraTypeError("mount must be handheld or tripod")
    if not isinstance(zoom, bool):
        raise CameraTypeError("zoom must be true or false")
    with _shared() as ct:
        if ct is not None:
            with _mapped_errors():
                ct.set_type(take, cam, mount, zoom, if_match)
    if ct is not None:
        types, etag = load(take, [cam])
        return types[0], etag
    etag = _write(take, {cam: {"mount": mount, "zoom": zoom, "source": "user", "set_at": now_iso(),
                               "conflict": None}}, if_match)
    types, etag2 = load(take, [cam])
    return types[0], etag2


def accept(take: str, cams: list[str] | None, min_confidence: float, if_match: str | None,
           all_cams: list[str] | None = None) -> tuple[list[dict], str]:
    """Suggestions with confidence >= min_confidence become user values (by = 'accepted')."""
    with _shared() as ct:
        if ct is not None:
            with _mapped_errors():
                ct.accept(take, list(cams or all_cams or []) or None, min_confidence, if_match)
    if ct is not None:
        return load(take, list(cams or all_cams or []))
    cams = list(cams or all_cams or [])
    types, _ = load(take, cams)
    upd = {}
    for t in types:
        s = t.get("suggestion") or {}
        if t["state"] == "confirmed" or s.get("mount") not in MOUNTS or not isinstance(s.get("zoom"), bool):
            continue
        if float(s.get("confidence") or 0) < min_confidence:
            continue
        upd[t["cam"]] = {"mount": s["mount"], "zoom": s["zoom"], "source": "user", "set_at": now_iso()}
    if not upd:
        return load(take, cams)
    _write(take, upd, if_match)
    return load(take, cams)


def write_suggestion(take: str, cam: str, suggestion: dict, if_match: str | None = None) -> str:
    """Record an automatic suggestion; pre-fills mount/zoom (source = auto) unless the user already decided."""
    with _shared() as ct:
        if ct is not None and hasattr(ct, "write_suggestion"):
            with _mapped_errors():
                ct.write_suggestion(take, cam, suggestion, if_match)
            return etag_of(toml_bytes(take))
    b = toml_bytes(take)
    cfg = tomllib.loads(b.decode("utf-8-sig")) if b else {}
    t = cam_tables(cfg).get(cam, {})
    sug = {k: suggestion.get(k) for k in ("mount", "zoom", "pans", "confidence", "by") if suggestion.get(k) is not None}
    sug["at"] = suggestion.get("at") or now_iso()
    upd: dict = {"suggested": sug}
    if suggestion.get("pans") is not None:
        upd["pans"] = bool(suggestion["pans"])
    if suggestion.get("zoom_events") is not None:
        upd["zoom_events"] = suggestion["zoom_events"]
    if t.get("source") != "user":
        upd.update({"mount": sug.get("mount"), "zoom": sug.get("zoom"), "source": "auto"})
    return _write(take, {cam: upd}, if_match)


TYPE_ROOTS = ("cam_loc", "mono_camera", "pair_calib")  # first stages that READ the types (cam_map does not)


def invalidated_by_change(take: str, cam: str, mount: str, zoom: bool, statuses: dict[str, str],
                          descendants) -> list[str]:
    """Done stages a type change would make stale: cam_loc / mono_camera / pair_calib and their descendants (cam_world
    ...), only when the effective class actually changes. Never the static SfM map (cam_map): stage_map does not
    read camera types, and rebuilding it would move the world frame."""
    types, _ = load(take, [cam])
    if types[0]["effective"]["class"] == class_of(mount, zoom):
        return []
    roots = [s for s in TYPE_ROOTS if s in statuses]
    out = []
    for s in [*roots, *descendants(roots)]:
        if statuses.get(s) in ("done", "stale") and s not in out:
            out.append(s)
    return out


def suggest_for_video(video: Path, probe: dict | None = None) -> dict | None:
    """Probe A suggestion through the shared module (None when it is not available).

    The result carries `file` = the video's signature (name, size, mtime). A result already measured for the same
    file - by the new-capture wizard while staging (intake records it in work/<take>/probe/camera_types_suggested
    .json) or for another take of the same recordings - is reused instead of decoding the video again (the shared
    module's `_probe_a_reuse`, the same rule `python -m dancecap.camera_types suggest` applies)."""
    try:
        from .. import camera_types as ct  # type: ignore[attr-defined]
    except Exception:
        return None
    sig = None
    try:
        sig = ct._file_sig(Path(video)) if callable(getattr(ct, "_file_sig", None)) else None
    except Exception:  # noqa: BLE001
        sig = None
    if sig and callable(getattr(ct, "_probe_a_reuse", None)):
        # only the lookup holds the shared lock (a fresh measurement takes seconds and must not block a save)
        with _shared() as m:
            try:
                hit = m._probe_a_reuse(sig) if m is not None else None
            except Exception:  # noqa: BLE001
                hit = None
        if hit:
            return hit
    for name in ("suggest_from_video", "probe_a", "suggest"):
        fn = getattr(ct, name, None)
        if callable(fn):
            try:
                s = fn(video) if probe is None else fn(video, probe=probe)
            except TypeError:
                try:
                    s = fn(video)
                except Exception:
                    return None
            except Exception:
                return None
            if isinstance(s, dict) and sig and not s.get("file"):
                s["file"] = sig
            return s
    return None
