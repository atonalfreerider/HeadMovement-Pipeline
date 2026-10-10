"""The Unity capture 07_CarlosAline: the stitched, refined motion of Carlos (lead) and Aline (follow) in their own room, with the heat's soundtrack and beat grid (C6, 2026-10-10).

    python -m dancecap.carlos_aline_export [--capture 07_CarlosAline] [--avatars] [--no-install] [--validate-only] [--audio WAV] [--no-audio]

ONE command, re-runnable (the capture is rewritten). What recap_export does for the class recap, with what this take has and the recap has not:
  * motion + shaped skins from the stitched refined fit (takes/carlos_aline.toml [export] smplx_dir); PLAIN default body materials unless --avatars;
  * the ROOM: room/room_mesh.obj of the world package (work/carlos_aline/room, dancecap/worlds/carlos_aline) as room_mesh.bin (export_unity.export_room: floor + wood back
    wall, side walls / ceiling are not observed) and the viewer floor turned by the take TOML's [export] floor_yaw_deg (the room's wall alignment, not the demo's 30);
  * the AUDIO = work/carlos_aline/audio/heat_studio.wav (dancecap.carlos_aline_audio: both studio songs on the reference clock, the heat's own sound where no song plays),
    sample 0 = reference 0 s. PRIVATE builds only (the songs are copyrighted studio recordings);
  * the 'beats' layer = zouk-time-analysis.json: timing/timing.json's grid when the timing analysis ran (it is song/beats_studio.json, --no-refine), else song/beats_studio.json;
  * song metadata (both songs, tempo, hand-over, no-beat intervals) in capture.json provenance.song;
  * timing / physics / counterbalance / floorcraft when the analyses exist (a missing layer is reported, not fatal), NO moves layer (the demo's labels came from the owner's
    narration; this heat has none and no motion classifier exists), NO camera clips.
--avatars: the owner's avatars on top (recap_export.avatar_hooks: shape edits and the identity faces of the TOML, albedo textures, hair groom, shoes) read from work/carlos_aline/avatar/
when present; the avatar agent re-runs this command with the switch.
Validation = demo_export.validate; the checks a plain capture fails by design are reported as expected. Report: work/carlos_aline/export/<capture>_ca_export.json. Exit 1 on a real failure.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import demo_export as de
from . import export_unity as eu
from . import recap_export as re_
from .take import PACKAGE_ROOT, Take

TAKE = "carlos_aline"
CAPTURE = "07_CarlosAline"
ROOM_BY_DESIGN = ("no room / splat",)       # demo_export.validate expects the dancers only; this capture carries the room mesh on purpose
EXPECTED_PLAIN = tuple(x for x in re_.EXPECTED_PLAIN if x != "audio") + ROOM_BY_DESIGN     # the heat soundtrack is in
EXPECTED_AVATARS: tuple[str, ...] = ROOM_BY_DESIGN        # avatars + the heat soundtrack: nothing else may fail


def song_meta(work: Path) -> dict:
    """provenance.song: the songs, the clocks, the hand-over and the no-beat intervals, from song/ and audio/ (no audio data, numbers and names only)."""
    song = work / "song"
    out: dict = {"note": "two songs mixed by the DJ; the studio mp3s are time-stretched (pitch kept) onto the reference clock for the soundtrack (private builds only)"}
    try:
        sj = json.loads((song / "song.json").read_text(encoding="utf-8"))
        out["songs"] = sj.get("songs") or sj
    except (OSError, ValueError):
        pass
    try:
        seg = json.loads((song / "segments.json").read_text(encoding="utf-8"))
        out["timeline"] = seg["timeline"]
        out["no_beat"] = seg["no_beat"]
        out["governing"] = seg["governing"]
    except (OSError, ValueError, KeyError):
        pass
    try:
        tj = json.loads((song / "tempo.json").read_text(encoding="utf-8"))
        out["tempo"] = tj.get("songs") or {k: tj[k] for k in list(tj)[:4]}
    except (OSError, ValueError):
        pass
    try:
        a = json.loads((work / "audio" / "heat_studio.json").read_text(encoding="utf-8"))
        out["soundtrack"] = {k: a.get(k) for k in ("file", "sample_rate", "reference_start", "reference_end", "plan", "gain")}
        out["soundtrack"]["verify"] = a.get("verify")
    except (OSError, ValueError):
        pass
    return out


def patch_capture_json(dirs: list[Path], extra_provenance: dict) -> list[str]:
    done = []
    for d in dirs:
        p = d / "capture.json"
        if p.exists():
            m = json.loads(p.read_text(encoding="utf-8"))
            m.setdefault("provenance", {}).update(extra_provenance)
            p.write_text(json.dumps(m, indent=2), encoding="utf-8")
            done.append(str(p))
    return done


def export(capture: str = CAPTURE, install: bool = True, avatars: bool = False, audio: Path | None = None, no_audio: bool = False) -> dict:
    capture = eu.validate_capture_name(capture)
    take = Take.load(TAKE)
    work = take.work
    if audio is None and not no_audio:
        audio = work / "audio" / "heat_studio.wav"
        if not audio.exists():
            raise FileNotFoundError(f"{audio} missing (python -m dancecap.carlos_aline_audio build)")
    t0 = 0.0
    if audio is not None and audio.with_suffix(".json").exists():
        t0 = float(json.loads(audio.with_suffix(".json").read_text(encoding="utf-8"))["reference_start"])
    beats = work / "song" / "beats_studio.json"
    rep = eu.export(TAKE, capture, install=install, room=True, textures=avatars, faces=avatars, shape_edits=avatars,
                    allow_no_audio=audio is None, audio=audio, audio_t0=t0 if audio is not None else None, beats=beats if beats.exists() else None)
    stream = eu.UNITY_STREAMING if install else work / "export"
    res: dict = {"take": TAKE, "capture": capture, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "steps": {}, "warnings": list(rep["warnings"]), "avatars": avatars}
    res["steps"]["export"] = {"frames": rep["frames"], "fps": rep["fps"], "bytes": rep["bytes"], "audio": rep["audio"], "room": rep.get("room"),
                              "dancers": {r: {k: v for k, v in d.items() if k != "shape_edits"} for r, d in rep["dancers"].items()}, "absent": rep.get("absent")}
    dirs = [d for d in (stream / capture, work / "export" / capture) if d.exists()]
    if audio is not None:
        res["warnings"].append("studio soundtrack in the capture: PRIVATE builds only (the songs are copyrighted studio recordings; audio.wav is git-ignored)")
    res["steps"]["song_metadata"] = patch_capture_json(dirs, {"song": song_meta(work)})
    if avatars:
        res["steps"].update(re_.avatar_hooks(take, capture, stream, install))
        for k, v in res["steps"].items():
            if isinstance(v, dict) and "error" in v and k in ("hair_groom", "hair_colliders", "shoes_seed", "shoes", "face_export"):
                res["warnings"].append(f"{k}: {v['error']}")
    smplx_dir, _ = eu.resolve_smplx_dir(TAKE)
    from . import counterbalance as cb
    from . import floorcraft as fc
    for kind, mod in (("counterbalance", cb), ("floorcraft", fc)):
        if not (work / kind / f"{kind}.json").exists():
            res["steps"][kind] = f"skipped: no work/{TAKE}/{kind}/{kind}.json yet"
        elif not mod.is_current(TAKE, smplx_dir):
            res["steps"][kind] = f"skipped: {kind}.json was computed on other fits than {smplx_dir}/ (re-run it on a pod: farm ca-analysis)"
            res["warnings"].append(res["steps"][kind])
        else:
            try:
                res["steps"][kind] = de.floor_layer(kind, TAKE, capture, stream, install, smplx_dir)
            except Exception as e:  # noqa: BLE001 - a layer is optional; report and go on
                res["steps"][kind] = {"error": f"{type(e).__name__}: {e}"}
                res["warnings"].append(f"{kind}: {type(e).__name__}: {e}")
    res["steps"]["moves"] = "skipped: no move timeline (the demo's labels came from the owner's narration; this heat has none and no motion-only classifier exists)"
    res["steps"]["beats"] = "zouk-time-analysis.json from " + ("timing/timing.json (the studio grid)" if (work / "timing" / "timing.json").exists() else "song/beats_studio.json")
    return res


def validate(capture: str, install: bool, rep: dict | None, avatars: bool) -> dict:
    take = Take.load(TAKE)
    cap = (eu.UNITY_STREAMING if install else take.work / "export") / capture
    v = de.validate(cap, take, rep)
    exp = EXPECTED_AVATARS if avatars else EXPECTED_PLAIN
    c = re_.classify(v["checks"], exp)
    return {**c, "checks": v["checks"], "ok": not c["failed"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.carlos_aline_export", description=__doc__.split("\n\n")[0])
    ap.add_argument("--capture", default=CAPTURE)
    ap.add_argument("--no-install", action="store_true", help="only work/carlos_aline/export/<capture>/")
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--avatars", action="store_true", help="the avatars of work/carlos_aline/avatar/ (shape edits, textures, groom, shoes, faces)")
    ap.add_argument("--audio", type=Path, help="another soundtrack wav (default work/carlos_aline/audio/heat_studio.wav, sample 0 = reference 0 s)")
    ap.add_argument("--no-audio", action="store_true")
    a = ap.parse_args(argv)
    install = not a.no_install
    t0 = time.time()
    res: dict = {"take": TAKE, "capture": a.capture, "steps": {}, "warnings": []}
    rep_v = None
    if not a.validate_only:
        res = export(a.capture, install, a.avatars, a.audio, a.no_audio)
        rep_v = {"dancers": {r: {"fk_max_err_mm": d["fk_max_err_mm"]} for r, d in res["steps"]["export"]["dancers"].items()}}
    res["validation"] = validate(a.capture, install, rep_v, a.avatars)
    res["seconds"] = round(time.time() - t0, 1)
    out = PACKAGE_ROOT / "work" / TAKE / "export" / f"{a.capture}_ca_export.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    for name, st in res["steps"].items():
        print(f"[ca_export] {name}: " + (st if isinstance(st, str) else ("error: " + st["error"] if isinstance(st, dict) and "error" in st else "ok")))
    v = res["validation"]
    for n in v["passed"]:
        print(f"[ca_export] PASS {n}")
    for e in v["expected"]:
        print(f"[ca_export] EXPECTED (plain capture) {e['name']}: {e['detail']}")
    for f in v["failed"]:
        print(f"[ca_export] FAIL {f['name']}: {f['detail']}")
    for w in res["warnings"]:
        print(f"[ca_export] WARNING {w}")
    print(f"[ca_export] {'OK' if v['ok'] else 'FAILED'} in {res['seconds']} s -> {out}")
    return 0 if v["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
