"""One-command demo export: a take -> a complete head-movement capture (avatars + every add-on layer), validated.

    python -m dancecap.demo_export <take> --capture <Name> [--smplx-dir smplx] [--audio auto|studio|phone]
                                   [--room] [--no-install] [--hooks-only] [--validate-only]

    e.g.  python -m dancecap.demo_export larissa_kadu --capture 02_LarissaKadu
          python -m dancecap.demo_export larissa_kadu_38 --capture 04_LarissaKaduDemo

Steps (each layer is optional: a missing input is reported, not fatal; a failing hook is a warning):
  1. export_unity.export: motion + shaped skins with the take TOML's [dancers.<role>.shape_edits] (dancecap.shape_edits:
     thinner midsection, feet sized for the measured sneakers), textures, timing / physics, faces when smplx_face/ exists
     (portrait-look albedo + blendshapes), NO room / splat (the viewer shows the dancers only; --room keeps them).
     Audio: auto = the take's studio soundtrack (work/<take>/song/studio_reference.wav, placed on the reference clock by
     dancecap.song; PRIVATE builds only - see its json) when present, else the phone audio (audio/audio_overlap.wav).
  2. hooks that must follow every export (it rewrites the capture folder):
       hair_groom export (avatar/hair_groom.json; a take without one is seeded by hair_groom.export from a sibling
         take of the same recordings, e.g. larissa_kadu for larissa_kadu_38), then its body colliders re-sized for the
         edited rest shape (the
         groom's capsules are measured on the plain fitted body: a 25 % thinner waist would leave the hair resting on
         an invisible mid-back capsule ~2 cm off her back) - the delta of hair_groom.colliders on the edited vs the
         plain rest mesh is added to each capsule radius;
       shoes export (avatar/shoes.json + the toml's shoe_geometry; a take without one gets a copy of a sibling take's:
         same sneakers, colours sampled from the same recordings - the sole stack comes from its own toml);
       face_export when the export did not already add the faces (installs the portrait-look albedo if present);
       counterbalance export (work/<take>/counterbalance/counterbalance.json) and floorcraft export
         (work/<take>/floorcraft/floorcraft.json: the leader-axis plants / moves / rotations + the pivots): capture-
         time Unity copies; each is (re-)analysed first only when missing or computed on other fits (lead npz sha1;
         floorcraft also when counterbalance.json changed) - a few seconds of light Python;
       moves export (work/<take>/moves/: capture-time copies, they depend on time_to_audio) - kept, not re-analysed.
  3. validation (capture folder): required files, frame counts (times.json, motion / face bins, figure json, timing,
     physics), skins (vertex / face counts, shape edits recorded as configured, UV split), textures decode with sane
     sizes, groom (scalp vertices inside the skin), shoes (both roles, sole numbers in range), audio covers the capture
     on the capture clock, FK error, counterbalance.json / floorcraft.json on the capture clock (time_to_audio) with
     no motion mismatch. -> work/<take>/export/<Name>_demo_export.json; exit code 1 on any failure.

--hooks-only re-runs steps 2-3 on the existing capture (e.g. after `python -m dancecap.hair_groom <take> --export-only`
overwrote the capture's groom with the unpatched colliders).
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import time
import wave
from pathlib import Path

import numpy as np

from .export_unity import ROLES, UNITY_STREAMING, read_blob, read_motion, read_skin, validate_capture_name
from .take import PACKAGE_ROOT, Take


# ------------------------------------------------------------------------------------------------ audio

def studio_audio(work: Path) -> tuple[Path, float] | None:
    """(wav, reference time of its first sample) of the take's studio soundtrack, when dancecap.song made one."""
    js, wav = work / "song" / "studio_reference.json", work / "song" / "studio_reference.wav"
    if js.exists() and wav.exists():
        return wav, float(json.loads(js.read_text(encoding="utf-8"))["reference_start"])
    return None


def sibling_with(take: Take, rel: str) -> Take | None:
    """Another take of the SAME recordings (same source folder) that has work/<other>/<rel>, else None."""
    mine = Path(take.sources[0].path).parent.resolve() if take.sources else None
    for t in sorted((PACKAGE_ROOT / "takes").glob("*.toml")):
        if t.stem == take.name:
            continue
        try:
            other = Take.load(t)
        except Exception:  # noqa: BLE001 - an unrelated, half-configured take
            continue
        theirs = Path(other.sources[0].path).parent.resolve() if other.sources else None
        if mine is not None and theirs == mine and (other.work / rel).exists():
            return other
    return None


def seed_shoes(take: Take) -> dict:
    """work/<take>/avatar/shoes.json from a sibling take when the take has none (noted in the file)."""
    dst = take.work / "avatar" / "shoes.json"
    if dst.exists():
        return {"seeded": False}
    sib = sibling_with(take, "avatar/shoes.json")
    if sib is None:
        return {"seeded": False, "note": "no shoes.json here or in a sibling take: neutral sneakers"}
    s = json.loads((sib.work / "avatar" / "shoes.json").read_text(encoding="utf-8"))
    s["take"] = take.name
    s["seeded_from"] = {"take": sib.name, "note": "same recordings / same sneakers; sole stack from this take's toml "
                                                  "shoe_geometry at export (dancecap.demo_export seed_shoes)"}
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(s, indent=1), encoding="utf-8")
    return {"seeded": True, "from": sib.name, "path": str(dst)}


# ------------------------------------------------------------------------------------------------ hair colliders

def hair_collider_delta(take: Take) -> dict:
    """{collider name: radius change (m)} that the follow's shape edits make to hair_groom.colliders (0 for none)."""
    from . import hair_groom as hg
    from . import shape_edits as se
    from .avatar_uv import vertex_normals
    cfg = se.config_for(take.name, take.cfg).get(hg.ROLE)
    if not cfg:
        return {}
    rm = hg.RestModel(take.work)
    sp = hg.scalp(rm)
    before = {c["name"]: float(c["r"]) for c in hg.colliders(rm, sp)}
    v, _ = se.apply_vertices(cfg, rm.v, rm.J, rm.w, rm.f, hg.ROLE, take.work)
    rm.v, rm.n = v, vertex_normals(v, rm.f)
    after = {c["name"]: float(c["r"]) for c in hg.colliders(rm, sp)}
    return {k: round(after[k] - before[k], 4) for k in before if abs(after[k] - before[k]) >= 5e-4}


def patch_hair_colliders(paths: list[Path], delta: dict) -> list[str]:
    """Add the radius deltas to the capsules of exported hair_groom.json copies (in place). -> changed names."""
    changed = set()
    for p in paths:
        g = json.loads(p.read_text(encoding="utf-8"))
        if g.get("colliders_shape_edits"):
            continue  # already patched
        for c in g.get("colliders", []):
            if c.get("name") in delta:
                c["r"] = round(max(0.02, float(c["r"]) + delta[c["name"]]), 4)
                changed.add(c["name"])
        g["colliders_shape_edits"] = {"delta_r_m": delta,
                                      "note": "capsule radii re-sized for the follow's rest-shape edits "
                                              "(dancecap.demo_export: hair_groom.colliders on the edited minus the "
                                              "plain rest mesh)"}
        p.write_text(json.dumps(g, indent=1), encoding="utf-8")
    return sorted(changed)


# ------------------------------------------------------------------------------------------------ floor layers

def floor_layer(kind: str, take_name: str, capture: str, stream: Path, mirror: bool,
                smplx_dir: str | None = None) -> dict:
    """counterbalance / floorcraft into the capture: (re-)analyse when missing or computed on other fits
    (counterbalance.is_current / floorcraft.is_current), then the capture-time Unity copy (+ the work mirror)."""
    from unittest import mock

    from . import counterbalance as cb
    from . import floorcraft as fc
    mod = cb if kind == "counterbalance" else fc
    out: dict = {"analysed": False}
    if not mod.is_current(take_name, smplx_dir):
        r = mod.run(take_name, smplx_dir, review=False)
        out.update({"analysed": True, "path": str(r["_path"])})
    # a capture whose dancer is PARKED while absent (export_unity.park_absent: stitched takes) differs from the fits there on
    # purpose: the motion check skips the probe times inside those windows (the other probes still compare)
    orig = cb.check_capture_motion

    def check(cap_dir, doc, tol_m=0.01):
        man = json.loads((Path(cap_dir) / "capture.json").read_text(encoding="utf-8"))
        wins = [w for v in (man.get("absent") or {}).values() for w in v.get("t", [])]
        probe = (doc.get("source") or {}).get("lead_probe")
        if not wins or not probe:
            return orig(cap_dir, doc, tol_m)
        keep = [i for i, t in enumerate(probe["t"]) if not any(a - 1e-6 <= t <= b + 1e-6 for a, b in wins)]
        doc2 = json.loads(json.dumps(doc))
        doc2["source"]["lead_probe"] = {"t": [probe["t"][i] for i in keep], "transl": [probe["transl"][i] for i in keep]}
        doc2["source"].pop("lead_transl0", None)
        res = orig(cap_dir, doc2, tol_m)
        return f"{res}; {len(probe['t']) - len(keep)} probe(s) inside the parked-absent window skipped"
    with mock.patch.object(cb, "check_capture_motion", check):
        out["export"] = mod.export(take_name, capture, streaming=stream, mirror=mirror)
    doc = json.loads((cb.work_dir(take_name) / kind / f"{kind}.json").read_text())
    out["summary"] = ({"intervals": [[iv["kind"], iv["t0"], iv["t1"]] for iv in doc["intervals"]]} if kind ==
                      "counterbalance" else doc["params"]["counts"])
    return out


# ------------------------------------------------------------------------------------------------ validation

def _png_size(p: Path) -> tuple[int, int] | None:
    raw = p.read_bytes()[:32]
    if raw[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    return struct.unpack(">II", raw[16:24])


def validate(cap: Path, take: Take, report: dict | None = None) -> dict:
    """Checks of an exported capture folder -> {"ok": bool, "checks": [{name, ok, detail}]}."""
    from . import shape_edits as se
    checks: list[dict] = []

    def check(name: str, ok: bool, detail=None):
        checks.append({"name": name, "ok": bool(ok), "detail": detail})
        return ok

    man_p = cap / "capture.json"
    if not check("capture.json", man_p.exists(), str(man_p)):
        return {"ok": False, "checks": checks}
    man = json.loads(man_p.read_text(encoding="utf-8"))
    T = int(man["frame_count"])
    times = json.loads((cap / "times.json").read_text()) if (cap / "times.json").exists() else []
    check("times.json frames", len(times) == T, f"{len(times)} vs frame_count {T}")
    check("times strictly increasing", bool(len(times) > 1 and np.all(np.diff(times) > 0)))
    for r, fig in zip(ROLES, ("figure1.json", "figure2.json")):
        ok = (cap / fig).exists()
        n = len(json.loads((cap / fig).read_text())) if ok else 0
        check(f"{fig} frames", ok and n == T, f"{n}")
    edit_cfg = se.config_for(take.name, take.cfg)
    from .export_unity import archived_from
    # owner rule 2026-10-23: the mesh-only files of a dancer with splats (albedo, face, the shared shoes) are archived, not missing
    # (export_unity.archive_mesh_avatars, work/<take>/archive/mesh_avatars_2026-10-23/): checked there by sha256
    archived = archived_from(cap, take.name)
    for r in ROLES:
        mp, sp = cap / f"smplx_{r}.bin", cap / f"smplx_{r}_skin.bin"
        if not check(f"{r} motion + skin present", mp.exists() and sp.exists()):
            continue
        mh, fr = read_motion(mp)
        check(f"{r} motion frames", mh["frames"] == T and fr.shape[0] == T and np.isfinite(fr).all(),
              f"{mh['frames']} frames, finite {bool(np.isfinite(fr).all())}")
        h, sk = read_skin(sp)
        V = len(sk.vertices)
        check(f"{r} skin", V > 10000 and len(sk.faces) > 20000 and np.isfinite(sk.vertices).all() and
              int(sk.faces.max()) < V, f"{V} vertices, {len(sk.faces)} faces, version {h.get('version')}")
        want = r in edit_cfg
        has = bool(h.get("shape_edits"))
        check(f"{r} shape edits as configured", want == has,
              {"configured": edit_cfg.get(r), "recorded": h.get("shape_edits")})
        alb = (man.get("smplx_albedo") or {}).get(r)
        if alb and not (cap / alb).exists() and Path(alb).name in archived:
            check(f"{r} albedo", True, f"{alb} archived (splat dancers): {archived[Path(alb).name]['to']}")
        elif alb:
            sz = _png_size(cap / alb) if (cap / alb).exists() else None
            check(f"{r} albedo", sz is not None and min(sz) >= 512 and h.get("version", 1) >= 2,
                  f"{alb} {sz} variant {(man.get('smplx_albedo_variant') or {}).get(r, 'bake')}")
        else:
            check(f"{r} albedo", False, "no smplx_albedo entry (untextured avatar)")
        face = (man.get("smplx_face") or {}).get(r)
        if face and not (cap / face).exists() and Path(face).name in archived:
            check(f"{r} face frames", True, f"{face} archived (splat dancers)")
        elif face:
            _, fh, _ = read_blob(cap / face)
            check(f"{r} face frames", fh.get("frames") == T, f"{fh.get('frames')}")
    for name in ("timing.json", "physics.json"):
        if (cap / name).exists():
            d = json.loads((cap / name).read_text())
            n = len(d.get("t", [])) if name == "physics.json" else None
            check(f"{name}", n is None or n == T, f"t frames {n}" if n is not None else "present")
    if (cap / "hair_groom.json").exists():
        g = json.loads((cap / "hair_groom.json").read_text(encoding="utf-8"))
        _, skf = read_skin(cap / "smplx_follow_skin.bin")
        vids = [int(x) for gd in g.get("guides", []) for x in np.atleast_1d(gd.get("vid", []))]
        nsm = int(skf.vertex_ids.max()) + 1 if skf.vertex_ids is not None else len(skf.vertices)
        check("hair groom", bool(g.get("colliders")) and bool(vids) and max(vids) < nsm,
              {"version": g.get("version"), "guides": len(g.get("guides", [])), "colliders": len(g.get("colliders", [])),
               "colliders_shape_edits": g.get("colliders_shape_edits", {}).get("delta_r_m")})
    else:
        check("hair groom", not (take.work / "avatar" / "hair_groom.json").exists(), "hair_groom.json missing")
    if (cap / "shoes.json").exists():
        s = json.loads((cap / "shoes.json").read_text(encoding="utf-8"))
        ok = s.get("version") == 1 and all(r in s.get("dancers", {}) for r in ROLES)
        ok = ok and all(0.01 <= s["dancers"][r]["sole_heel_m"] <= 0.07 and 0.008 <= s["dancers"][r]["sole_fore_m"] <= 0.06
                        for r in ROLES)
        check("shoes", ok, {r: {k: s["dancers"][r].get(k) for k in ("style", "sole_heel_m", "sole_fore_m")}
                            for r in ROLES if r in s.get("dancers", {})})
    elif "shoes.json" in archived:
        check("shoes", True, f"shoes.json archived (splat dancers): {archived['shoes.json']['to']}")
    else:
        check("shoes", not (take.work / "avatar" / "shoes.json").exists(), "shoes.json missing")
    if man.get("audio"):
        with wave.open(str(cap / man["audio"]), "rb") as w:
            dur = w.getnframes() / float(w.getframerate())
        t_a0, t_a1 = times[0] + man["time_to_audio"], times[-1] + man["time_to_audio"]
        check("audio covers the capture", t_a0 >= -1e-3 and t_a1 <= dur + 1e-3,
              f"capture audio time {t_a0:.3f}..{t_a1:.3f} s in a {dur:.2f} s wav ({(man.get('provenance') or {}).get('audio')})")
    else:
        check("audio", False, "no audio (silent clock)")
    for kind in ("counterbalance", "floorcraft"):
        p = cap / f"{kind}.json"
        if p.exists():
            from .moves.labels import CaptureClock
            d = json.loads(p.read_text())
            tta = CaptureClock.from_capture(cap).time_to_audio
            mc = str(d.get("motion_check", ""))
            ok = (d.get("time_base") == "capture" and abs(float(d.get("time_to_audio", 1e9)) - tta) < 1e-6
                  and not mc.startswith("MISMATCH") and man.get(kind) == f"{kind}.json")
            detail = {"motion_check": mc, "time_to_audio": d.get("time_to_audio")}
            detail.update({"intervals": [[iv["kind"], iv["t0"], iv["t1"]] for iv in d.get("intervals", [])]}
                          if kind == "counterbalance" else (d.get("params") or {}).get("counts", {}))
            check(f"{kind}.json", ok, detail)
        else:
            check(f"{kind}.json", False, f"{kind}.json missing from the capture (python -m dancecap.{kind})")
    check("no room / splat", "room" not in man and "environment_splat" not in man,
          "room present" if "room" in man else "dancers only")
    from .export_unity import resolve_role_hidden
    want_hidden = resolve_role_hidden(take.name)   # takes/<take>.toml [export] role_hidden (muted-role spans + fade-in, capture seconds)
    check("role_hidden as configured", (man.get("role_hidden") or None) == want_hidden,
          {"configured": want_hidden, "recorded": man.get("role_hidden")})
    from .export_unity import resolve_props
    want_props = resolve_props(take.name)   # takes/<take>.toml [dancers.<role>.props.<kind>] (head-worn props: the headset), written by the exporter
    check("props as configured", (man.get("props") or None) == want_props, {"configured": want_props, "recorded": man.get("props")})
    if report:
        for r, d in (report.get("dancers") or {}).items():
            check(f"{r} FK vs npz joints", d["fk_max_err_mm"] < 10.0, f"{d['fk_max_err_mm']} mm")
    return {"ok": all(c["ok"] for c in checks), "checks": checks}


# ------------------------------------------------------------------------------------------------ run

def run(take_name: str, capture: str, smplx_dir: str | None = None, audio: str = "auto", room: bool = False,
        install: bool = True, validate_only: bool = False, hooks_only: bool = False) -> dict:
    from . import export_unity
    capture = validate_capture_name(capture)
    take = Take.load(take_name)
    work = take.work
    t0 = time.time()
    # the motion folder: --smplx-dir, else takes/<take>.toml [export] smplx_dir (e.g. a dancecap.pose_refine output),
    # else smplx/ (so a plain re-export keeps the take's chosen motion)
    smplx_dir = smplx_dir or (take.cfg.get("export") or {}).get("smplx_dir")
    res: dict = {"take": take_name, "capture": capture, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                 "steps": {}, "warnings": [], "smplx_dir": smplx_dir or "smplx"}
    stream = UNITY_STREAMING if install else work / "export"  # the hooks write into stream / capture (+ the mirror)
    cap = stream / capture
    rep = None
    if not validate_only and not hooks_only:
        kw: dict = {}
        st = studio_audio(work) if audio in ("auto", "studio") else None
        if audio == "studio" and st is None:
            raise FileNotFoundError(f"{work}/song/studio_reference.wav(.json) missing (python -m dancecap.song)")
        if st is not None:
            kw = {"audio": st[0], "audio_t0": st[1]}
            res["warnings"].append("studio soundtrack in the capture: PRIVATE builds only (song/studio_reference.json "
                                   "distribution)")
        rep = export_unity.export(take_name, capture, smplx_dir=smplx_dir, room=room, install=install, **kw)
        res["steps"]["export"] = {"audio": rep["audio"], "frames": rep["frames"], "bytes": rep["bytes"],
                                  "dancers": {r: {k: v for k, v in d.items() if k != "shape_edits"}
                                              for r, d in rep["dancers"].items()},
                                  "shape_edits": {r: d["shape_edits"] for r, d in rep["dancers"].items()
                                                  if d.get("shape_edits")},
                                  "faces": bool(rep.get("faces"))}
        res["warnings"] += rep["warnings"]
    if not validate_only:
        if hooks_only and not (cap / "capture.json").exists():
            raise FileNotFoundError(f"{cap}/capture.json missing: export it first (drop --hooks-only)")
        mirror = work / "export" / capture
        dirs = [d for d in (cap, mirror) if d.exists()]

        def hook(name, fn):
            try:
                res["steps"][name] = fn()
            except Exception as e:  # noqa: BLE001 - every hook is optional; report and go on
                res["steps"][name] = {"error": f"{type(e).__name__}: {e}"}
                res["warnings"].append(f"{name}: {type(e).__name__}: {e}")

        from . import hair_groom
        hook("hair_groom", lambda: sorted({str(p) for p in hair_groom.export(take, capture, stream)}))
        if isinstance(res["steps"]["hair_groom"], list) and res["steps"]["hair_groom"]:
            hook("hair_colliders", lambda: (lambda dl: {"delta_r_m": dl, "patched": patch_hair_colliders(
                [d / "hair_groom.json" for d in dirs if (d / "hair_groom.json").exists()], dl)})(
                hair_collider_delta(take)))
        hook("shoes_seed", lambda: seed_shoes(take))
        if (work / "avatar" / "shoes.json").exists():
            from . import shoes
            hook("shoes", lambda: sorted({str(p) for p in shoes.export(take, capture, stream)}))
        else:
            res["steps"]["shoes"] = "skipped: no avatar/shoes.json (neutral sneakers)"
        if hooks_only or not rep.get("faces"):
            from . import face_export
            hook("face_export", lambda: {k: {"dancers": {r: {kk: vv for kk, vv in d.items() if kk != "motion"}
                                                          for r, d in v.get("dancers", {}).items()},
                                              "warnings": v.get("warnings")}
                                         for k, v in face_export.export(take_name, capture, streaming=stream).items()})
        else:
            res["steps"]["face_export"] = "done by the export (smplx_face/)"
        hook("counterbalance", lambda: floor_layer("counterbalance", take_name, capture, stream, install, smplx_dir))
        hook("floorcraft", lambda: floor_layer("floorcraft", take_name, capture, stream, install, smplx_dir))
        if (work / "moves" / "labels.json").exists():
            from .moves.export import export_moves
            hook("moves", lambda: export_moves(take_name, cap))
        else:
            res["steps"]["moves"] = "skipped: no moves/labels.json"
    res["validation"] = validate(cap, take, rep)
    if install and (work / "export" / capture).exists():
        res["validation_mirror"] = {"ok": validate(work / "export" / capture, take, rep)["ok"]}
    res["seconds"] = round(time.time() - t0, 1)
    out = work / "export" / f"{capture}_demo_export.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    res["report"] = str(out)
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.demo_export", description=__doc__.split("\n")[0])
    ap.add_argument("take")
    ap.add_argument("--capture", required=True, help="capture folder name, e.g. 02_LarissaKadu")
    ap.add_argument("--smplx-dir", default=None, help="contract folder under work/<take> (default: the take toml's "
                                                       "[export] smplx_dir, else smplx/)")
    ap.add_argument("--audio", choices=("auto", "studio", "phone"), default="auto",
                    help="auto: the studio soundtrack when dancecap.song made one, else the phone audio")
    ap.add_argument("--room", action="store_true", help="also export the room mesh / splat (off: dancers only)")
    ap.add_argument("--no-install", action="store_true", help="only work/<take>/export/<capture>/")
    ap.add_argument("--hooks-only", action="store_true", help="re-run the hooks + validation on the existing capture")
    ap.add_argument("--validate-only", action="store_true", help="only validate the existing capture folder")
    a = ap.parse_args(argv)
    res = run(a.take, a.capture, a.smplx_dir, a.audio, a.room, not a.no_install, a.validate_only, a.hooks_only)
    for name, st in res["steps"].items():
        brief = st if isinstance(st, str) else ("error: " + st["error"] if isinstance(st, dict) and "error" in st
                                               else "ok")
        print(f"[demo_export] {name}: {brief}")
    for r, s in (res["steps"].get("export", {}).get("shape_edits") or {}).items():
        from .shape_edits import summary
        print(f"[demo_export] {r} shape edits: {json.dumps(summary(s))}")
    for c in res["validation"]["checks"]:
        print(f"[demo_export] {'PASS' if c['ok'] else 'FAIL'} {c['name']}: {c['detail']}")
    for w in res["warnings"]:
        print(f"[demo_export] WARNING {w}")
    print(f"[demo_export] {'OK' if res['validation']['ok'] else 'FAILED'} in {res['seconds']} s -> {res['report']}")
    return 0 if res["validation"]["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
