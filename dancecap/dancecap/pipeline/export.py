"""`python -m dancecap.pipeline export <take> --capture <Name> [demo_export options] [--dry-run]`: the monitor's
Export button.

Runs dancecap.demo_export with the same arguments, then - when the export succeeded - the triggers stage with its
capture export (work/<take>/triggers/triggers.json + StreamingAssets/<Name>/triggers.json in capture time), so the
viewer gets the classifier-triggered layers of the capture it just received. demo_export itself is not changed (it
belongs to the demo workflow); the pipeline's demo_export stage runs the same two commands.

--dry-run writes nothing: it prints the two commands, the motion folder and the inputs demo_export would read (which
layers are present / missing), the folders it would replace (work/<take>/export/<Name>/ and the Unity
StreamingAssets/<Name>/), the Unity lock, and the triggers the export would carry (counts per kind and per viewer
layer). Exit 0 when the required inputs exist, 1 otherwise.
"""
from __future__ import annotations

import sys
from pathlib import Path


def _dry_run(take: str, capture: str | None, argv: list[str]) -> int:
    from . import paths, takes, triggers

    def opt(name: str, default=None):
        return argv[argv.index(name) + 1] if name in argv and argv.index(name) + 1 < len(argv) else default

    print(f"[export] DRY RUN for {take} -> capture {capture}: nothing is written", flush=True)
    if not paths.take_toml(take).exists():
        print(f"[export] takes/{take}.toml not found", flush=True)
        return 1
    cfg, _ = takes.load_cfg(take)
    w = paths.take_work(take)
    sd = opt("--smplx-dir") or ((cfg.get("export") or {}).get("smplx_dir")) or "smplx"
    audio = opt("--audio", "auto")
    rest = [a for a in argv[1:] if a != "--dry-run"]
    print(f"[export] would run: python -m dancecap.demo_export {take} {' '.join(rest)}", flush=True)
    print(f"[export] then:      python -m dancecap.pipeline triggers {take} --export-capture {capture}", flush=True)

    def rel(p: Path) -> str:
        return paths.rel_display(p)

    required = [(f"motion ({sd}/)", [w / sd / "lead.npz", w / sd / "follow.npz"]),
                ("world frame", [w / "cameras" / "world_from_sfm.json"])]
    studio = [w / "song" / "studio_reference.wav", w / "song" / "studio_reference.json"]
    phone = [w / "audio" / "audio_overlap.wav", w / "audio" / "audio_overlap.json"]
    if audio == "studio":
        required.append(("studio soundtrack", studio))
    elif audio == "phone":
        required.append(("phone audio", phone))
    else:
        required.append(("audio (studio soundtrack, else phone)",
                         studio if all(p.exists() for p in studio) else phone))
    optional = [("textures", [w / "avatar" / "lead_albedo.png", w / "avatar" / "follow_albedo.png"]),
                ("timing", [w / "timing" / "timing.json"]), ("physics", [w / "physics" / "physics.json"]),
                ("faces", [w / "smplx_face"]), ("hair groom", [w / "avatar" / "hair_groom.json"]),
                ("shoes", [w / "avatar" / "shoes.json"]),
                ("counterbalance", [w / "counterbalance" / "counterbalance.json"]),
                ("floorcraft", [w / "floorcraft" / "floorcraft.json"]), ("move labels", [w / "moves" / "labels.json"])]
    if "--room" in argv:
        optional.append(("room (--room)", [w / "room"]))
    ok = True
    for label, files in required:
        miss = [f for f in files if not f.exists()]
        ok &= not miss
        print(f"[export]   {'ok     ' if not miss else 'MISSING'} {label}: "
              + ", ".join(rel(f) for f in (miss or files)), flush=True)
    for label, files in optional:
        miss = [f for f in files if not f.exists()]
        print(f"[export]   {'ok     ' if not miss else 'absent '} {label}"
              + ("" if not miss else " (layer left out; demo_export reports it, not fatal)"), flush=True)
    try:
        from ..export_unity import UNITY_STREAMING
    except Exception:  # noqa: BLE001
        UNITY_STREAMING = paths.unity_project() / "Assets" / "StreamingAssets"
    install = "--no-install" not in argv
    for d in [w / "export" / str(capture)] + ([Path(UNITY_STREAMING) / str(capture)] if install else []):
        print(f"[export]   would {'replace' if d.exists() else 'create'} {rel(d)}/", flush=True)
    lock = paths.work() / "unity.lock"
    if install and lock.exists():
        owner = ""
        try:
            owner = (lock / "owner.txt").read_text(encoding="utf-8", errors="replace").strip()[:120] if lock.is_dir() \
                else lock.read_text(encoding="utf-8", errors="replace").strip()[:120]
        except OSError:
            pass
        print(f"[export]   work/unity.lock is held ({owner or 'Unity work in progress'}): the real export waits for it "
              "or needs 'export anyway'", flush=True)
    try:
        doc = triggers.collect(take)
        trig = doc.get("triggers") or []
        kinds: dict[str, int] = {}
        layers: dict[str, int] = {}
        for t in trig:
            kinds[t.get("kind", "?")] = kinds.get(t.get("kind", "?"), 0) + 1
            for lid in t.get("viewer_layers") or []:
                layers[lid] = layers.get(lid, 0) + 1
        print(f"[export]   triggers: {len(trig)} ({', '.join(f'{k} {n}' for k, n in sorted(kinds.items())) or 'none'}); "
              f"viewer layers: {', '.join(f'{k} {n}' for k, n in sorted(layers.items())) or 'none'}", flush=True)
    except Exception as e:  # noqa: BLE001 - informational
        print(f"[export]   triggers: not computable now ({type(e).__name__}: {e})", flush=True)
    print(f"[export] DRY RUN {'OK: the export can run' if ok else 'FAILED: required inputs are missing'}", flush=True)
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0].startswith("-") or "--capture" not in argv:
        print("usage: python -m dancecap.pipeline export <take> --capture <Name> [demo_export options] [--dry-run]",
              file=sys.stderr)
        return 2
    take = argv[0]
    capture = argv[argv.index("--capture") + 1] if argv.index("--capture") + 1 < len(argv) else None
    if "--dry-run" in argv:
        return _dry_run(take, capture, argv)
    from .. import demo_export
    try:
        rc = demo_export.main(argv)
    except SystemExit as e:
        rc = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    rc = int(rc or 0)
    if rc != 0:
        print(f"[export] demo_export failed (rc {rc}): triggers not exported", flush=True)
        return rc
    from . import triggers
    try:
        triggers.write(take)
        if capture:
            triggers.export_capture(take, capture)
    except Exception as e:  # noqa: BLE001 - the capture itself is exported; the triggers are an extra file
        print(f"[export] triggers export failed: {type(e).__name__}: {e}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
