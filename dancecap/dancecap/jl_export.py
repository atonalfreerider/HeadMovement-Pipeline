"""The Unity capture 08_JohnLula: the stitched, refined, single-camera motion of the lead and the follow of the john_lula take, with the heat's own sound and beat grid (J4, 2026-10-11).

    python -m dancecap.jl_export [--capture 08_JohnLula] [--avatars] [--no-install] [--validate-only] [--audio WAV] [--no-audio]

carlos_aline_export for a take with ONE phone: ONE command, re-runnable (the capture is rewritten). Content:
  * motion + skins from the stitched refined fit (takes/john_lula.toml [export] smplx_dir = smplx_rf2_noslip); PLAIN default body materials unless --avatars;
  * the ROOM of the world package (work/john_lula/room: floor + far and left wall, J2; weak: wall offsets +-2.5 m, 15 percent of the floor observed) as room_mesh.bin, the viewer floor
    turned by the take TOML's [export] floor_yaw_deg (world X is already the hall's axis: 0). The library film does not show it;
  * the AUDIO = the video's own sound (work/john_lula/audio/audio_overlap.wav: the phone's mono mix on the reference clock, sample 0 = 0 s, music + crowd). PRIVATE (the owner's footage);
  * the 'beats' layer = zouk-time-analysis.json (timing/timing.json's grid = song/beats_heat.json: constant 75 BPM, music 3.40-80.20 s, no beat in the intro and the applause);
  * timing / physics / counterbalance / floorcraft / rhythm when the analyses exist (a missing layer is reported, not fatal); results inside the interpolated stretches and the zoom steps carry a
    low-confidence flag where the format allows (dancecap.jl_lowconf); NO moves layer, NO camera clips, the 2D video is NOT embedded;
  * provenance.song / provenance.reliability in capture.json (tempo, no-beat intervals, the monocular caveat, the weak stretches), numbers and names only.
--avatars: the owner's avatars on top (recap_export.avatar_hooks: shape edits and the identity faces of the TOML, albedo, hair groom (ponytail), shoes, face texture) read from
work/john_lula/avatar/ (agent J5b): re-run this command with the switch when it has landed.
Validation = demo_export.validate; the checks a plain capture fails by design are reported as expected. Report: work/john_lula/export/<capture>_jl_export.json. Exit 1 on a real failure.
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

TAKE = "john_lula"
CAPTURE = "08_JohnLula"


def configure(take: str, capture: str | None = None) -> None:
    """Export another one-camera take (pipeline/mono_route; the 2026-10-21 trial patched TAKE / CAPTURE from a glue script)."""
    global TAKE, CAPTURE
    TAKE = take
    CAPTURE = capture or default_capture(take)


def default_capture(take: str) -> str:
    """A capture folder name for a take: "Mono_" + its CamelCase name (letters, digits, _ . - only: export_unity's rule)."""
    import re
    return "Mono_" + "".join(w[:1].upper() + w[1:] for w in re.split(r"[^A-Za-z0-9]+", take) if w)[:48]
AUDIO = "audio/audio_overlap.wav"
ROOM_BY_DESIGN = ("no room / splat",)       # demo_export.validate expects the dancers only; this capture carries the room mesh on purpose
EXPECTED_PLAIN = tuple(x for x in re_.EXPECTED_PLAIN if x != "audio") + ROOM_BY_DESIGN
EXPECTED_AVATARS: tuple[str, ...] = ROOM_BY_DESIGN


def _json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def song_meta(work: Path) -> dict:
    """provenance.song: the song structure, tempo and no-beat intervals of song/ (numbers only; the song is unknown)."""
    song = work / "song"
    out: dict = {"note": "the song is unknown: the beat grid is measured on the heat's own audio (dancecap.heat_beats); no studio recording"}
    seg = _json(song / "segments.json")
    if seg:
        out["timeline"], out["no_beat"] = seg.get("timeline"), seg.get("no_beat")
        out["music_span"], out["danced_span"] = seg.get("music_span"), seg.get("danced_span")
    tj = _json(song / "tempo.json")
    if tj:
        out["tempo"] = {k: tj[k] for k in list(tj)[:6]}
    return out


def camera_summary(work: Path) -> str:
    """"one handheld phone (WxH, N zoom steps)" from the take's cameras/01.npz (was john_lula's text for every take)."""
    import numpy as np
    try:
        z = np.load(work / "cameras" / "01.npz", allow_pickle=True)
        w, h = (int(v) for v in z["native_size"]) if "native_size" in z.files else (int(z["K"][0, 0, 2] * 2), int(z["K"][0, 1, 2] * 2))
        zoom = np.asarray(z["zoom_2x"], bool) if "zoom_2x" in z.files else np.zeros(1, bool)
        steps = int(np.count_nonzero(np.diff(zoom.astype(int))))
    except (OSError, KeyError, ValueError):
        return "one handheld phone"
    words = {0: "no zoom step", 1: "one zoom step", 2: "two zoom steps", 3: "three zoom steps"}
    return f"one handheld phone ({w}x{h}, {words.get(steps, f'{steps} zoom steps')})"


def reliability_meta(work: Path, take: Take | None = None) -> dict:
    """provenance.reliability: what the single-camera fit does and does not support (PROVISIONAL.md / NOTES_J3b.md), and the low-confidence intervals the layers carry.
    A take fitted from two or more views (2026-10-24, the two-view lucas_luiza) says so instead: "monocular": False, `views`, the camera text "two views
    triangulated (<name> 01 + <name> 02)" with the names of [export] camera_names {cam: name} else the [cameras."NN"] mount, and the low-confidence
    intervals of the exported fit ([export] smplx_dir). The views are the fitted camera set when the TOML names one ([segment] cams of a chunk take,
    [derived] cams, a top-level cams), else the sources. Monocular: fewer than two views, or a [stitch] mono over the whole window (the single-video
    route: lucas_luiza_h carries the Steadicam cut as a second source but is fitted from camera 01 alone)."""
    from . import jl_lowconf as lc
    take = take or Take.load(TAKE)
    cfg = take.cfg
    fit = next((c["cams"] for c in (cfg.get("segment"), cfg.get("derived"), cfg) if isinstance(c, dict) and c.get("cams")), None)
    views = [str(c) for c in fit] if fit else [s.id for s in take.sources]
    w0, w1 = take.window
    mono = [[float(a), float(b)] for a, b in (cfg.get("stitch") or {}).get("mono", [])]
    if len(views) < 2 or any(a <= w0 + 0.05 and b >= w1 - 0.05 for a, b in mono):
        return {"camera": f"{camera_summary(work)}: the distance of the couple from the camera is +-10 percent (focal length), the pose relative to the pelvis, "
                          "the left-right position and the height above the floor are good; hands are the raw fit's",
                "provisional": True, "low_confidence": lc.intervals(work), "monocular": True}
    from . import camera_types as ct
    names, tables = (cfg.get("export") or {}).get("camera_names") or {}, ct.cam_tables(cfg)
    what = " + ".join(f"{names.get(c) or tables.get(c, {}).get('mount') or 'camera'} {c}" for c in views)
    n = {2: "two", 3: "three", 4: "four"}.get(len(views), str(len(views)))
    return {"camera": f"{n} views triangulated ({what})", "provisional": True,
            "low_confidence": lc.intervals(work, eu.resolve_smplx_dir(take.name)[0]), "monocular": False, "views": views,
            **({"single_view_s": mono} if mono else {})}


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
        audio = work / AUDIO
        if not audio.exists():
            raise FileNotFoundError(f"{audio} missing (the pipeline's audio stage: audio_sync writes the video's own sound)")
    t0 = 0.0
    if audio is not None and audio.with_suffix(".json").exists():
        t0 = float(json.loads(audio.with_suffix(".json").read_text(encoding="utf-8")).get("reference_start", 0.0))
    beats = work / "song" / "beats_heat.json"
    rep = eu.export(TAKE, capture, install=install, room=True, textures=avatars, faces=avatars, shape_edits=avatars,
                    allow_no_audio=audio is None, audio=audio, audio_t0=t0 if audio is not None else None, beats=beats if beats.exists() else None,
                    park=False)    # the interpolated stretches are SHOWN (flagged low-confidence in the layers), not parked
    stream = eu.UNITY_STREAMING if install else work / "export"
    res: dict = {"take": TAKE, "capture": capture, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "steps": {}, "warnings": list(rep["warnings"]), "avatars": avatars}
    res["steps"]["export"] = {"frames": rep["frames"], "fps": rep["fps"], "bytes": rep["bytes"], "audio": rep["audio"], "room": rep.get("room"),
                              "dancers": {r: {k: v for k, v in d.items() if k != "shape_edits"} for r, d in rep["dancers"].items()}, "absent": rep.get("absent")}
    dirs = [d for d in (stream / capture, work / "export" / capture) if d.exists()]
    if audio is not None:
        res["warnings"].append("the video's own sound in the capture: PRIVATE (the owner's footage; audio.wav is git-ignored)")
    res["steps"]["song_metadata"] = patch_capture_json(dirs, {"song": song_meta(work), "reliability": reliability_meta(work)})
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
            res["steps"][kind] = f"skipped: {kind}.json was computed on other fits than {smplx_dir}/ (re-run it on a pod: farm jl-analysis)"
            res["warnings"].append(res["steps"][kind])
        else:
            try:
                res["steps"][kind] = de.floor_layer(kind, TAKE, capture, stream, install, smplx_dir)
            except Exception as e:  # noqa: BLE001 - a layer is optional; report and go on
                res["steps"][kind] = {"error": f"{type(e).__name__}: {e}"}
                res["warnings"].append(f"{kind}: {type(e).__name__}: {e}")
    res["steps"]["moves"] = "skipped: no move labels (no narration, no trained classifier)"
    res["steps"]["beats"] = "zouk-time-analysis.json from " + ("timing/timing.json (the heat's grid)" if (work / "timing" / "timing.json").exists() else "song/beats_heat.json")
    res["steps"]["rhythm"] = "rhythm.json (live rhythm score)" if (work / "rhythm" / "rhythm.json").exists() else f"skipped: no work/{TAKE}/rhythm/rhythm.json yet (python -m dancecap.rhythm run {TAKE})"
    return res


def validate(capture: str, install: bool, rep: dict | None, avatars: bool) -> dict:
    take = Take.load(TAKE)
    cap = (eu.UNITY_STREAMING if install else take.work / "export") / capture
    v = de.validate(cap, take, rep)
    exp = EXPECTED_AVATARS if avatars else EXPECTED_PLAIN
    c = re_.classify(v["checks"], exp)
    return {**c, "checks": v["checks"], "ok": not c["failed"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.jl_export", description=__doc__.split("\n\n")[0])
    ap.add_argument("--take", default=TAKE, help="the one-camera take (default john_lula)")
    ap.add_argument("--capture", default=None, help="capture folder (default 08_JohnLula for john_lula, else Mono_<Take>)")
    ap.add_argument("--no-install", action="store_true", help="only work/john_lula/export/<capture>/")
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--avatars", action="store_true", help="the avatars of work/john_lula/avatar/ (shape edits, textures, groom, shoes, faces)")
    ap.add_argument("--audio", type=Path, help="another soundtrack wav (default work/john_lula/audio/audio_overlap.wav, sample 0 = reference 0 s)")
    ap.add_argument("--no-audio", action="store_true")
    a = ap.parse_args(argv)
    if a.take != TAKE:
        configure(a.take, a.capture)
    elif a.capture:
        configure(TAKE, a.capture)
    a.capture = CAPTURE
    install = not a.no_install
    t0 = time.time()
    res: dict = {"take": TAKE, "capture": a.capture, "steps": {}, "warnings": []}
    rep_v = None
    if not a.validate_only:
        res = export(a.capture, install, a.avatars, a.audio, a.no_audio)
        rep_v = {"dancers": {r: {"fk_max_err_mm": d["fk_max_err_mm"]} for r, d in res["steps"]["export"]["dancers"].items()}}
    res["validation"] = validate(a.capture, install, rep_v, a.avatars)
    res["seconds"] = round(time.time() - t0, 1)
    out = PACKAGE_ROOT / "work" / TAKE / "export" / f"{a.capture}_jl_export.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    for name, st in res["steps"].items():
        print(f"[jl_export] {name}: " + (st if isinstance(st, str) else ("error: " + st["error"] if isinstance(st, dict) and "error" in st else "ok")))
    v = res["validation"]
    for n in v["passed"]:
        print(f"[jl_export] PASS {n}")
    for e in v["expected"]:
        print(f"[jl_export] EXPECTED (plain capture) {e['name']}: {e['detail']}")
    for f in v["failed"]:
        print(f"[jl_export] FAIL {f['name']}: {f['detail']}")
    for w in res["warnings"]:
        print(f"[jl_export] WARNING {w}")
    print(f"[jl_export] {'OK' if v['ok'] else 'FAILED'} in {res['seconds']} s -> {out}")
    return 0 if v["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
