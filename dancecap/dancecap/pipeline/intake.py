"""Stage `intake`: register a new capture from its inputs (videos + soundtrack + metadata) as takes/<take>.toml.

Copy-free: the videos and the soundtrack are referenced in place (`source_dir` + per-camera `file`), never copied
or uploaded. What is written:
- takes/<take>.toml: name, source_dir, alignment (written later by audio_sync), exclude, [sync], [dance],
  [dancers.lead|follow] (name, pid, stature_m, shoe_sole_m, description, portrait), [consent] (TMR_SOMA_PLAN G0)
  and one [cameras."NN"] block per video (file + the camera type: mount handheld|tripod, zoom true|false,
  source user when given, else no type yet: the probe suggestion and the review screen fill it in).
- work/<take>/intake.json: the registered videos (path, size, probe facts, exclusions), problems and warnings.
- work/<take>/probe/probe.json: the ffprobe facts measured during validation (the `probe` stage then only adds the
  camera-type suggestion).
- work/<take>/clock/alignment.json for a SINGLE-view take: the identity clock (its only video is the reference), so
  every stage can load the take (multi- and two-view takes get theirs from the audio_sync stage).
- the automatic camera-type suggestion the wizard measured while staging (probe A), when there is one: recorded as
  `suggested` (and mount/zoom with source = "auto" for a camera the user did not answer), so the probe stage does
  not have to measure it again.

Validation (422 with per-field problems): take name, unique two-digit camera ids, files exist and decode as video,
duration >= 5 s, plausible fps (VFR is reported, never assumed constant), rotation (reported; applied at ingest),
audio (a camera without audio cannot join the audio clocks of a multi- or two-view take: it is excluded), the
soundtrack decodes as audio, both dancers with a name and a plausible stature, level and category, consent values.
"""
from __future__ import annotations

import json
import os
import re
import time
from datetime import date
from pathlib import Path

from . import camtypes, paths, probe, takes
from .util import CAM_RE, TAKE_RE, write_json_atomic, write_text_atomic

CONSENT_VALUES = (True, False, "?")


class IntakeError(ValueError):
    def __init__(self, problems: list[dict]):
        super().__init__("; ".join(f"{p['field']}: {p['message']}" for p in problems))
        self.problems = problems


def parse_stature(v) -> float | None:
    """5'8" | 5 ft 8 in | 173 cm | 173 | 1.727 | 1.727 m -> metres (3 decimals); None when unparsable."""
    if v is None or v == "":
        return None
    if isinstance(v, (int, float)):
        x = float(v)
        return round(x / 100 if x > 3 else x, 3)
    s = str(v).strip().lower().replace("’", "'").replace("”", '"').replace("″", '"').replace("′", "'")
    m = re.fullmatch(r"(\d+)\s*(?:'|ft|feet)\s*(?:(\d+(?:\.\d+)?)\s*(?:\"|in|inch|inches|'')?)?", s)
    if m:
        return round(int(m.group(1)) * 0.3048 + float(m.group(2) or 0) * 0.0254, 3)
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(cm|m)?", s)
    if m:
        x = float(m.group(1))
        unit = m.group(2) or ("cm" if x > 3 else "m")
        return round(x / 100 if unit == "cm" else x, 3)
    return None


def _toml_str(s: str) -> str:
    return json.dumps(str(s), ensure_ascii=False)


def _abs(p: Path) -> Path:
    """Absolute without following links: in a worktree whose work/ is a junction, resolve() would climb out of the checkout."""
    return Path(os.path.abspath(p))


def _rel(p: Path, base: Path) -> str:
    try:
        return Path(os.path.relpath(p, base)).as_posix()
    except ValueError:  # another drive
        return p.as_posix()


def _consent_value(v):
    if v in CONSENT_VALUES:
        return v
    if isinstance(v, str) and v.lower() in ("true", "yes"):
        return True
    if isinstance(v, str) and v.lower() in ("false", "no"):
        return False
    return "?"


def validate(req: dict, probe_fn=probe.probe_file) -> tuple[dict, list[dict], list[str]]:
    """Normalise a TakeCreate request (videos/soundtrack/portraits as resolved absolute `path`s).
    Returns (normalised, problems, warnings)."""
    P: list[dict] = []
    W: list[str] = []

    def bad(field, msg):
        P.append({"field": field, "message": msg})

    take = str(req.get("take") or "").strip()
    if not TAKE_RE.match(take):
        bad("take", "lowercase letters, digits and _ only (1-48 characters, starting with a letter or digit)")
    elif paths.take_toml(take).exists():
        bad("take", f"takes/{take}.toml already exists")
    if (req.get("source_mode") or "reference") != "reference":
        bad("source_mode", "only 'reference' (register the files in place) is supported")
    vids_in = req.get("videos") or []
    vids, seen = [], set()
    for i, v in enumerate(vids_in):
        f = f"videos[{i}]"
        cam = str(v.get("cam") or "").strip()
        path = Path(str(v.get("path") or ""))
        if not CAM_RE.match(cam):
            bad(f"{f}.cam", "camera id must be two digits")
        elif cam in seen:
            bad(f"{f}.cam", f"camera id {cam} used twice")
        seen.add(cam)
        if not v.get("path") or not path.is_file():
            bad(f"{f}.path", "file not found")
            continue
        try:
            pr = v.get("probe") or probe_fn(path)
        except Exception as e:  # noqa: BLE001
            bad(f"{f}.path", f"cannot read the video: {str(e)[:160]}")
            continue
        for h in probe.check_video(pr):
            bad(f"{f}.path", h)
        t = v.get("type") or {}
        mount, zoom = t.get("mount"), t.get("zoom")
        if mount is not None and mount not in camtypes.MOUNTS:
            bad(f"{f}.type.mount", "handheld or tripod")
        if zoom is not None and not isinstance(zoom, bool):
            bad(f"{f}.type.zoom", "true or false")
        sug = v.get("suggestion") if isinstance(v.get("suggestion"), dict) else None
        if sug is not None and (sug.get("mount") not in camtypes.MOUNTS or not isinstance(sug.get("zoom"), bool)):
            sug = None
        vids.append({"cam": cam, "path": _abs(path), "probe": pr, "excluded": bool(v.get("excluded")),
                     "reason": "excluded by you" if v.get("excluded") else None,
                     "type": {"mount": mount, "zoom": zoom} if mount and zoom is not None else None,
                     "suggestion": sug})
    used = [v for v in vids if not v["excluded"]]
    if not vids_in:
        bad("videos", "add at least one video")
    elif not used and not P:
        bad("videos", "every video is excluded")
    if len(used) >= 2:
        for v in used:
            if not v["probe"].get("audio"):
                v["excluded"], v["reason"] = True, "no audio: cannot join the audio clocks"
                W.append(f"camera {v['cam']}: no audio track, excluded from the clocks (and the take)")
        used = [v for v in vids if not v["excluded"]]
    for v in used:
        pr = v["probe"]
        if pr.get("vfr"):
            W.append(f"camera {v['cam']}: variable frame rate (median {pr.get('fps')} fps): timestamps are used")
        if pr.get("rotation"):
            W.append(f"camera {v['cam']}: rotated {pr['rotation']} deg (applied at ingest)")
        if not pr.get("playable", True):
            W.append(f"camera {v['cam']}: {pr.get('codec')} is not playable in the review window (make a proxy)")
    if used:
        durs = [v["probe"].get("duration_s") or 0 for v in used]
        if max(durs) > 0 and min(durs) < 0.5 * max(durs):
            W.append(f"durations differ a lot ({min(durs):.0f}-{max(durs):.0f} s): the window will be the overlap")
    snd = None
    s_in = req.get("soundtrack")
    if s_in:
        sp = Path(str(s_in.get("path") or ""))
        if not sp.is_file():
            bad("soundtrack", "file not found")
        elif sp.suffix.lower() not in takes.AUDIO_EXT:
            bad("soundtrack", f"an audio file ({', '.join(takes.AUDIO_EXT)}), not {sp.suffix or 'this file'}")
        else:
            try:
                spr = s_in.get("probe") or probe_fn(sp)
                if not spr.get("audio"):
                    bad("soundtrack", "no audio stream")
                snd = {"path": _abs(sp), "probe": spr}
            except Exception as e:  # noqa: BLE001
                bad("soundtrack", f"cannot read: {str(e)[:160]}")
    ct = req.get("capture_type") or "auto"
    auto = "single" if len(used) <= 1 else "two" if len(used) == 2 else "multi"
    order = {"single": 1, "two": 2, "multi": 3}
    if ct != "auto" and ct not in order:
        bad("capture_type", "auto, multi, two or single")
    elif ct != "auto" and order[ct] > order[auto]:
        bad("capture_type", f"{ct}-view needs more cameras than the {len(used)} used (a variant can only be forced down)")
    variant = auto if ct == "auto" or ct not in order or order[ct] > order[auto] else ct
    dancers = {}
    din = req.get("dancers") or {}
    for pid, role in enumerate(takes.ROLES, start=1):
        d = din.get(role) or {}
        name = str(d.get("name") or "").strip()
        if not name:
            bad(f"dancers.{role}.name", "required")
        st = parse_stature(d.get("stature_m", d.get("stature")))
        if st is None:
            bad(f"dancers.{role}.stature_m", "a stature like 5'8\", 173 cm or 1.73 m")
        elif not 1.2 <= st <= 2.2:
            bad(f"dancers.{role}.stature_m", f"{st:.3f} m is not plausible")
        sole = d.get("shoe_sole_m", 0.025)
        try:
            sole = float(sole)
            if not 0 <= sole <= 0.1:
                raise ValueError
        except (TypeError, ValueError):
            bad(f"dancers.{role}.shoe_sole_m", "0 (barefoot) .. 0.1 m")
            sole = 0.025
        port = None
        if d.get("portrait"):
            pp = Path(str((d["portrait"] or {}).get("path") if isinstance(d["portrait"], dict) else d["portrait"]))
            if not pp.is_file() or pp.suffix.lower() not in takes.IMAGE_EXT:
                bad(f"dancers.{role}.portrait", "an existing .jpg/.png image")
            else:
                port = _abs(pp)
        dancers[role] = {"name": name, "pid": pid, "stature_m": st, "stature_in": str(d.get("stature_m", d.get("stature")) or ""),
                         "shoe_sole_m": sole, "description": str(d.get("description") or "").strip(), "portrait": port}
    dance = dict(req.get("dance") or {})
    if dance.get("level") not in takes.LEVELS:
        bad("dance.level", " | ".join(takes.LEVELS))
    if dance.get("category") not in takes.CATEGORIES:
        bad("dance.category", " | ".join(takes.CATEGORIES))
    if dance.get("date") and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(dance["date"])):
        bad("dance.date", "YYYY-MM-DD")
    cin = req.get("consent") or {}
    consent = {}
    for role in takes.ROLES:
        r = cin.get(role) or {}
        consent[role] = {k: _consent_value(r.get(k, "?")) for k in takes.CONSENT_SCOPES}
    consent["source_rights"] = str(cin.get("source_rights") or "?")
    consent["bystanders"] = str(cin.get("bystanders") or "?")
    if any(consent[r]["cloud_processing"] is not True for r in takes.ROLES):
        W.append("cloud_processing consent is not recorded for every dancer: pod stages stay blocked until it is")
    norm = {"take": take, "videos": vids, "soundtrack": snd, "variant": variant, "capture_type": ct,
            "dancers": dancers, "dance": dance, "consent": consent}
    return norm, P, W


def _suggestion_toml(s: dict) -> dict:
    """The staged probe-A suggestion as one inline table (numbers rounded, no nested lists)."""
    out = {k: s.get(k) for k in ("mount", "zoom", "pans", "by") if s.get(k) is not None}
    c = s.get("confidence")
    if isinstance(c, (int, float)) and not isinstance(c, bool):
        out["confidence"] = round(float(c), 3)
    at = s.get("at")
    out["at"] = at if isinstance(at, str) else camtypes.now_iso()
    ev = {k: round(float(v), 4) for k, v in (s.get("evidence") or {}).items()
          if isinstance(v, (int, float)) and not isinstance(v, bool)}
    if ev:
        out["evidence"] = dict(list(ev.items())[:8])
    return out


def render_toml(n: dict) -> str:
    take = n["take"]
    tdir = paths.takes_dir()
    vids = n["videos"]
    srcs = [v["path"] for v in vids]
    common = Path(os.path.commonpath([str(p.parent) for p in srcs])) if srcs else tdir
    title = f"{n['dancers']['lead']['name'] or 'Leader'} + {n['dancers']['follow']['name'] or 'Follower'} x " \
            f"{n['dance'].get('song') or '?'}"
    L = [f"# {title} - registered by the Pipeline Monitor new-capture wizard on {date.today().isoformat()}.",
         "# The videos and the soundtrack are referenced in place (never copied); outputs go to work/<take>/ (git-ignored).",
         f"# Capture type: {n['variant']}-view ({sum(1 for v in vids if not v['excluded'])} camera(s) used).",
         f"name = {_toml_str(take)}",
         f"source_dir = {_toml_str(_rel(common, tdir))}",
         "# affine audio clocks (reference_time = slope * source_time + offset), written by the audio_sync stage",
         f"alignment = {_toml_str(f'../work/{take}/clock/alignment.json')}"]
    excl = [v["cam"] for v in vids if v["excluded"]]
    L.append(f"exclude = [{', '.join(_toml_str(c) for c in excl)}]" + (
        "   # " + "; ".join(f"{v['cam']}: {v['reason']}" for v in vids if v["excluded"]) if excl else ""))
    L.append("# window = [t0, t1]   # reference seconds analysed in detail: written by the window stage")
    if n["variant"] != {1: "single", 2: "two"}.get(len([v for v in vids if not v["excluded"]]), "multi"):
        L += ["", "[pipeline]", f"variant = {_toml_str(n['variant'])}   # forced down from the camera count"]
    if n["variant"] == "multi":
        L += ["", "# Video sync (dancecap.sync): mamma prepare refuses to build fit inputs until this take's",
              "# sync/offsets.json exists and records these clocks.", "[sync]", "required = true"]
    d = n["dance"]
    L += ["", "# Dance metadata: data is siloed by level; the title is \"Leader + Follower x Song\".", "[dance]",
          f"level = {_toml_str(d['level'])}            # professional | intermediate | novice",
          f"category = {_toml_str(d['category'])}      # demo | lesson | jack_and_jill"]
    for k in ("song", "artist", "date", "venue"):
        if d.get(k):
            L.append(f"{k} = {_toml_str(d[k])}")
    if n["soundtrack"]:
        L.append(f"song_file = {_toml_str(_rel(n['soundtrack']['path'], tdir))}   # studio track (stays local, never "
                 "in a repo)")
    for role in takes.ROLES:
        x = n["dancers"][role]
        L += ["", f"[dancers.{role}]", f"name = {_toml_str(x['name'])}", f"pid = {x['pid']}",
              f"stature_m = {x['stature_m']}" + (f"   # entered as {x['stature_in']}" if x["stature_in"] and
                                                 x["stature_in"] != str(x["stature_m"]) else ""),
              f"shoe_sole_m = {x['shoe_sole_m']}"]
        if x["description"]:
            L.append(f"description = {_toml_str(x['description'])}")
        if x["portrait"]:
            L.append(f"portrait = {_toml_str(_rel(x['portrait'], tdir))}   # stays local (likeness data)")
    c = n["consent"]

    def inline(r):
        vals = ", ".join(f"{k} = {('true' if v is True else 'false' if v is False else _toml_str(v))}"
                         for k, v in c[r].items())
        return "{ " + vals + f", recorded = {_toml_str(date.today().isoformat())}" + " }"

    L += ["", "# Consent per dancer as reported by the user (TMR_SOMA_PLAN G0): \"?\" = unknown. cloud_processing is",
          "# required before any Runpod stage; public_repo before anything leaves the PC.", "[consent]",
          f"lead = {inline('lead')}", f"follow = {inline('follow')}",
          f"source_rights = {_toml_str(c['source_rights'])}", f"bystanders = {_toml_str(c['bystanders'])}"]
    text = "\n".join(L) + "\n"
    blocks = []
    for v in vids:
        t = {"file": _rel(v["path"], common)}
        sug = v.get("suggestion")
        if sug:
            t["suggested"] = _suggestion_toml(sug)
            if sug.get("pans") is not None:
                t["pans"] = bool(sug["pans"])
        if v["type"]:
            t.update({"mount": v["type"]["mount"], "zoom": v["type"]["zoom"], "source": "user",
                      "set_at": camtypes.now_iso()})
        elif sug:
            t.update({"mount": sug["mount"], "zoom": sug["zoom"], "source": "auto"})  # pre-filled, not confirmed
        blocks.append(camtypes.render_block(v["cam"], t))
    return text + "\n" + "\n".join(blocks)


def create(req: dict, dry_run: bool = False, probe_fn=probe.probe_file) -> dict:
    norm, problems, warnings = validate(req, probe_fn=probe_fn)
    if problems:
        raise IntakeError(problems)
    text = render_toml(norm)
    import tomllib
    parsed = tomllib.loads(text)  # never write a file that does not parse
    take = norm["take"]
    out = {"take": take, "variant": norm["variant"], "toml_preview": text, "problems": [], "warnings": warnings,
           "toml_path": f"dancecap/takes/{take}.toml"}
    if dry_run:
        return out
    p = paths.take_toml(take)
    if p.exists():
        raise IntakeError([{"field": "take", "message": f"takes/{take}.toml already exists"}])
    write_text_atomic(p, text)
    w = paths.take_work(take)
    vids = [{"cam": v["cam"], "file": parsed["cameras"][v["cam"]]["file"], "path": str(v["path"]),
             "size_bytes": v["probe"].get("size_bytes"), "excluded": v["excluded"], "reason": v["reason"],
             "probe": v["probe"]} for v in norm["videos"]]
    write_json_atomic(w / "intake.json", {"version": 1, "take": take, "created": time.time(), "variant": norm["variant"],
                                          "capture_type": norm["capture_type"], "videos": vids,
                                          "soundtrack": {"path": str(norm["soundtrack"]["path"]),
                                                         "probe": norm["soundtrack"]["probe"]} if norm["soundtrack"] else None,
                                          "warnings": warnings, "source_mode": "reference"})
    write_json_atomic(w / "probe" / "probe.json", {"version": 1, "take": take, "at": time.time(), "by": "intake",
                                                   "cameras": {v["cam"]: dict(v["probe"], file=v["file"]) for v in vids}})
    staged = {v["cam"]: v["suggestion"] for v in norm["videos"]
              if isinstance(v.get("suggestion"), dict) and isinstance(v["suggestion"].get("file"), dict)}
    if staged:  # the probe stage reuses these (same file signature) instead of decoding the videos again
        write_json_atomic(w / "probe" / "camera_types_suggested.json",
                          {"version": 1, "take": take, "by": "probe_a", "at": camtypes.now_iso(),
                           "source": "new-capture wizard (staged)", "cameras": staged})
    used = [v for v in vids if not v["excluded"]]
    if norm["variant"] == "single" and len(used) == 1:
        from .audio_sync import identity
        u = used[0]
        write_json_atomic(w / "clock" / "alignment.json",
                          identity(take, u["cam"], u["file"], (u["probe"] or {}).get("duration_s")))
        out["alignment"] = f"work/{take}/clock/alignment.json (identity: single view)"
    from .util import sha1_bytes
    out["toml_etag"] = sha1_bytes(p.read_bytes())
    return out
