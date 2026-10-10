"""First playable Unity capture of the class recap: plain SMPL-X bodies, no audio, no camera clips (F2, 2026-10-09).

    python -m dancecap.recap_export larissa_kadu_recap --capture 06_LarissaKaduRecap [--no-install] [--validate-only] [--avatars] [--audio [WAV]]

What `dancecap.demo_export` does for the demo, minus everything that belongs to the owner's avatars (a separate avatar run, later):
  * motion + shaped skins from the stitched, refined fit (takes/<take>.toml [export] smplx_dir), the PLAIN fitted shape (the take's
    [dancers.<role>.shape_edits] are NOT applied), no albedo texture, no face / hair / shoes layers: the viewer draws its default body
    material, colour-coded lead / follow;
  * no audio unless --audio (the exporter's silent clock) and no camera clips (the phone videos are source footage the film flies to later);
  * timing.json (beat-free: dancecap.timing_free), physics.json, counterbalance.json and floorcraft.json when the analyses exist
    (a layer that is missing is reported, not fatal); NO beat / zouk-time-analysis layer (there is no beat grid) and no moves layer
    (there is no move timeline: dancecap.moves labels come from narration, not from motion).
--avatars (T2, 2026-10-09): the owner's avatars on top of the same motion: the take TOML's [dancers.<role>.shape_edits] (thinner midsection,
sneaker feet, the identity-fitted faces), the albedo textures (the portrait-look face composite when avatar/<role>_albedo_face.png is current), the
hair groom (avatar/hair_groom.json + its collider patch), the shoes (avatar/shoes.json + the TOML shoe_geometry) and the mouth cap: the hooks
dancecap.demo_export.run applies after the exporter, minus the beat / moves layers. --audio [WAV] (T3): the class mix film/audio/class_mix.wav as audio.wav on the capture clock (t0 = 0: sample 0 = master 0 s). Re-runnable: the capture is rewritten.
Validation = demo_export.validate (capture folder: required files, frame counts, skins, FK vs the fit, timing / physics grids,
counterbalance / floorcraft on the capture clock); the checks that a plain capture fails BY DESIGN (albedo, shape edits, audio, hair,
shoes) are reported as expected, not as failures. Report: work/<take>/export/<capture>_recap_export.json. Exit 1 on a real failure.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import demo_export as de
from . import export_unity as eu
from .take import PACKAGE_ROOT, Take

# demo_export.validate checks a plain capture fails by design (a name is matched as a prefix, "<role> <check>")
EXPECTED_PLAIN = ("lead shape edits as configured", "follow shape edits as configured", "lead albedo", "follow albedo", "audio",
                  "hair groom", "shoes")


EXPECTED_AVATARS = ("audio",)      # an avatar capture WITHOUT --audio has no soundtrack (silent clock)
EXPECTED_FULL: tuple[str, ...] = ()   # avatars + the class audio: nothing may fail


def class_audio(take) -> Path | None:
    """The class mix of the recap film (film_recap: 48 kHz stereo, sample 0 = master 0 s = the capture's time base), when it exists."""
    p = take.work / "film" / "audio" / "class_mix.wav"
    return p if p.exists() else None


def classify(checks: list[dict], expected: tuple[str, ...] = EXPECTED_PLAIN) -> dict:
    """Split demo_export.validate checks into real failures and the plain capture's expected ones."""
    out = {"failed": [], "expected": [], "passed": []}
    for c in checks:
        if c["ok"]:
            out["passed"].append(c["name"])
        elif c["name"] in expected:
            out["expected"].append({"name": c["name"], "detail": c["detail"]})
        else:
            out["failed"].append({"name": c["name"], "detail": c["detail"]})
    return out


def avatar_hooks(take, capture: str, stream: Path, install: bool) -> dict:
    """What demo_export.run does after the exporter for the avatars: hair groom (+ its collider patch for the edited rest shape), shoes, the face
    texture + mouth cap. Each hook is optional: a failure is reported, not fatal. -> {hook: result}."""
    from . import face_export, hair_groom, shoes
    work = take.work
    cap = stream / capture
    dirs = [d for d in (cap, work / "export" / capture) if d.exists()]
    steps: dict = {}

    def hook(name, fn):
        try:
            steps[name] = fn()
        except Exception as e:  # noqa: BLE001 - every hook is optional; report and go on
            steps[name] = {"error": f"{type(e).__name__}: {e}"}

    hook("hair_groom", lambda: sorted({str(p) for p in hair_groom.export(take, capture, stream)}))
    if isinstance(steps["hair_groom"], list) and steps["hair_groom"]:
        hook("hair_colliders", lambda: (lambda dl: {"delta_r_m": dl, "patched": de.patch_hair_colliders(
            [d / "hair_groom.json" for d in dirs if (d / "hair_groom.json").exists()], dl)})(de.hair_collider_delta(take)))
    hook("shoes_seed", lambda: de.seed_shoes(take))
    if (work / "avatar" / "shoes.json").exists():
        hook("shoes", lambda: sorted({str(p) for p in shoes.export(take, capture, stream)}))
    else:
        steps["shoes"] = "skipped: no avatar/shoes.json (neutral sneakers)"
    hook("face_export", lambda: {k: {"dancers": {r: {kk: vv for kk, vv in d.items() if kk != "motion"} for r, d in v.get("dancers", {}).items()},
                                     "warnings": v.get("warnings")} for k, v in face_export.export(take.name, capture, streaming=stream).items()})
    return steps


def export_plain(take_name: str, capture: str, install: bool = True, avatars: bool = False, audio: Path | None = None) -> dict:
    """The motion + skins (export_unity.export, plain; avatars=True: shape edits + albedo textures + faces), then (avatars) the groom / shoes /
    face hooks, then the analysis layers that exist."""
    capture = eu.validate_capture_name(capture)
    take = Take.load(take_name)
    work = take.work
    rep = eu.export(take_name, capture, install=install, room=False, textures=avatars, faces=avatars, shape_edits=avatars,
                    allow_no_audio=audio is None, audio=audio, audio_t0=0.0 if audio is not None else None)
    stream = eu.UNITY_STREAMING if install else work / "export"
    res: dict = {"take": take_name, "capture": capture, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "steps": {}, "warnings": list(rep["warnings"])}
    res["steps"]["export"] = {"frames": rep["frames"], "fps": rep["fps"], "bytes": rep["bytes"], "audio": rep["audio"],
                              "dancers": {r: {k: v for k, v in d.items() if k != "shape_edits"} for r, d in rep["dancers"].items()},
                              "absent": rep.get("absent")}
    if avatars:
        res["steps"].update(avatar_hooks(take, capture, stream, install))
        for k, v in res["steps"].items():
            if isinstance(v, dict) and "error" in v and k in ("hair_groom", "hair_colliders", "shoes_seed", "shoes", "face_export"):
                res["warnings"].append(f"{k}: {v['error']}")
        res["steps"]["export"]["shape_edits"] = {r: d["shape_edits"] for r, d in rep["dancers"].items() if d.get("shape_edits")}
    smplx_dir, _ = eu.resolve_smplx_dir(take_name)
    from . import counterbalance as cb
    from . import floorcraft as fc
    for kind, mod in (("counterbalance", cb), ("floorcraft", fc)):
        if not (work / kind / f"{kind}.json").exists():
            res["steps"][kind] = f"skipped: no work/{take_name}/{kind}/{kind}.json yet"
        elif not mod.is_current(take_name, smplx_dir):
            # demo_export.floor_layer would RE-ANALYSE here (minutes of CPU on the workstation): analysis runs on pods, never here
            res["steps"][kind] = f"skipped: {kind}.json was computed on other fits than {smplx_dir}/ (re-run it on a pod: farm recap-analysis)"
            res["warnings"].append(res["steps"][kind])
        else:
            try:
                res["steps"][kind] = de.floor_layer(kind, take_name, capture, stream, install, smplx_dir)
            except Exception as e:  # noqa: BLE001 - a layer is optional; report and go on
                res["steps"][kind] = {"error": f"{type(e).__name__}: {e}"}
                res["warnings"].append(f"{kind}: {type(e).__name__}: {e}")
    res["steps"]["moves"] = "skipped: no move timeline (labels come from narration, not from motion)"
    res["steps"]["beats"] = "skipped: no beat grid (the recap has no soundtrack)"
    return res


def validate_plain(take_name: str, capture: str, install: bool, rep: dict | None = None, avatars: bool = False, audio: bool = False) -> dict:
    take = Take.load(take_name)
    cap = (eu.UNITY_STREAMING if install else take.work / "export") / capture
    v = de.validate(cap, take, rep)
    exp = EXPECTED_FULL if (avatars and audio) else EXPECTED_AVATARS if avatars else EXPECTED_PLAIN
    return {**classify(v["checks"], exp), "checks": v["checks"], "ok": not classify(v["checks"], exp)["failed"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.recap_export", description=__doc__.split("\n\n")[0])
    ap.add_argument("take")
    ap.add_argument("--capture", required=True)
    ap.add_argument("--no-install", action="store_true", help="only work/<take>/export/<capture>/")
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--audio", nargs="?", const="auto", default=None, metavar="WAV",
                    help="the capture's soundtrack on the capture clock (sample 0 = 0 s): 'auto'/no value = film/audio/class_mix.wav; default none")
    ap.add_argument("--avatars", action="store_true", help="the owner's avatars: shape edits, textures, groom, shoes, face texture (see the module doc)")
    a = ap.parse_args(argv)
    install = not a.no_install
    t0 = time.time()
    res: dict = {"take": a.take, "capture": a.capture, "steps": {}, "warnings": []}
    rep_for_validate = None
    audio = None
    if a.audio is not None:
        audio = class_audio(Take.load(a.take)) if a.audio == "auto" else Path(a.audio)
        if audio is None or not audio.exists():
            ap.error("--audio: the class mix film/audio/class_mix.wav does not exist (python -m dancecap.film_recap mix class)")
    if not a.validate_only:
        res = export_plain(a.take, a.capture, install, a.avatars, audio)
        rep_for_validate = {"dancers": {r: {"fk_max_err_mm": d["fk_max_err_mm"]} for r, d in res["steps"]["export"]["dancers"].items()}}
    res["validation"] = validate_plain(a.take, a.capture, install, rep_for_validate, a.avatars, a.audio is not None)
    res["seconds"] = round(time.time() - t0, 1)
    out = PACKAGE_ROOT / "work" / a.take / "export" / f"{a.capture}_recap_export.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    for name, st in res["steps"].items():
        print(f"[recap_export] {name}: " + (st if isinstance(st, str) else ("error: " + st["error"] if "error" in st else "ok")))
    v = res["validation"]
    for n in v["passed"]:
        print(f"[recap_export] PASS {n}")
    for e in v["expected"]:
        print(f"[recap_export] EXPECTED (plain capture) {e['name']}: {e['detail']}")
    for f in v["failed"]:
        print(f"[recap_export] FAIL {f['name']}: {f['detail']}")
    for w in res["warnings"]:
        print(f"[recap_export] WARNING {w}")
    print(f"[recap_export] {'OK' if v['ok'] else 'FAILED'} in {res['seconds']} s -> {out}")
    return 0 if v["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
