"""World package of a NEW room measured from ONE handheld phone video (john_lula): dancecap/worlds/john_lula/<id>/manifest.json.

Same layout, ids and verification as dancecap.farm.world (FARM.md section 3) and as dancecap.room_world_two_cams: a tracked manifest (hashes +
relative paths, no media, no likeness) and the data tree under work/worlds/john_lula/<id>/ (git-ignored). The members are what ONE camera and the
dancers' statures determine: the room model + mesh, the world frame, the per-frame camera track (K, R, tvec of every frame: handheld, fixed centre,
two zoom states), the focal / distance / height ambiguity table, the scale reference. The members that need the MAMMA shape pass (betas), the
skeleton configuration, the ma_3d config, the refine recipe and the avatar are NOT in it yet: `partial` = true and `missing_members` list them (a
later world with them is a NEW id; this one stays readable). There is no COLMAP room model: the room is the planes of room/room_model.json, and
cameras/world_from_sfm.json is the identity (the camera is stored in the world frame).

    python -m dancecap.room_world_single_cam build [--no-current]
    python -m dancecap.room_world_single_cam verify
"""
from __future__ import annotations

import argparse
import json
import time
import tomllib
from pathlib import Path

from .camera_types import cam_tables
from .farm import world as W
from .farm.state import atomic_write_json, now_iso
from .farm import worlds_data, worlds_tracked
from .farm.timeline import DanceSpec
from .take import PACKAGE_ROOT

DANCE = "john_lula"
WORK = PACKAGE_ROOT / "work" / DANCE
MISSING = ["room_sfm (no COLMAP model: the room is room_model.json)", "betas (MAMMA shape pass)", "skeleton", "ma3d_config",
           "refine_recipe", "avatar"]


def member_list() -> list[W.Member]:
    cfg = tomllib.loads((PACKAGE_ROOT / "takes" / f"{DANCE}.toml").read_text(encoding="utf-8"))
    cams = {k: {x: y for x, y in v.items() if x not in ("source", "set_at")} for k, v in cam_tables(cfg).items()}  # not loc / map / x_cam
    cam, room = WORK / "cameras", WORK / "room"
    out: list[W.Member] = []

    def add(name, scope, note, entries, source=""):
        out.append(W.Member(name, scope, note, entries, source))

    add("room_model", W.POSE_SCOPE, "the hall as measured planes (floor, far wall, left wall, ceiling) with the evidence numbers",
        W._file(room / "room_model.json", "ref/room/room_model.json"), "work/john_lula/room")
    add("room_mesh", W.POSE_SCOPE, "textured floor + far / left wall for the viewer (room/room_mesh.obj + 2 textures; capture data: git-ignored)",
        [W.Entry(f"ref/room/{n}", src=room / n) for n in ("room_mesh.obj", "room_mesh.mtl", "room_floor_tex.png", "room_mesh_tex.png")]
        + W._file(room / "mesh_report.json", "ref/room/mesh_report.json"), "work/john_lula/room")
    add("world_frame", W.POSE_SCOPE, "world_from_sfm.json (identity: the camera is stored in the world frame)",
        W._file(cam / "world_from_sfm.json", "ref/cameras/world_from_sfm.json"), "work/john_lula/cameras")
    add("camera_rig", W.POSE_SCOPE, "intrinsics, centre, orientation, fit numbers, scale evidence, the focal / distance / height table, motion statistics",
        W._file(cam / "rig.json", "ref/cameras/rig.json") + W._file(cam / "uncertainty_variants.json", "ref/cameras/uncertainty_variants.json")
        + W._file(cam / "motion_stats.json", "ref/cameras/motion_stats.json"), "work/john_lula/cameras")
    add("camera_tracks", W.POSE_SCOPE, "per-frame K, R, tvec of the camera in the world frame (CONTRACTS.md cameras/<cam>.npz schema) + the dancers' floor positions",
        [W.Entry(f"ref/cameras/{n}", src=cam / n) for n in ("01.npz", "01_world.npz", "dancers_world.npz")], "work/john_lula/cameras")
    add("camera_types", W.POSE_SCOPE, "per-camera mount / zoom (handheld, three 1x <-> 2x steps), without timestamps",
        [W.Entry("shared/camera_types.json", data=W._json_bytes(cams))], "takes/john_lula.toml [cameras]")
    d = json.loads((cam / "rig.json").read_text())
    add("scale_reference", W.POSE_SCOPE, "the statures, the camera height they give and the focal hypothesis it rests on (statures from the owner)",
        [W.Entry("shared/scale_reference.json", data=W._json_bytes({"statures_m": {k: v.get("stature_m") for k, v in (cfg.get("dancers") or {}).items()},
                                                                     "soles_m": {k: v.get("shoe_sole_m") for k, v in (cfg.get("dancers") or {}).items()},
                                                                     "camera_height_m": d["camera_height_m"], "focal_1x_px": d["focal_1x_px"],
                                                                     "dancers_mean_distance_m": d["dancers_mean_distance_m"],
                                                                     "height": d["height"]}))],
        "takes/john_lula.toml + cameras/rig.json")
    return out


def build(make_current: bool = True) -> dict:
    t0 = time.time()
    spec = DanceSpec.load(DANCE)
    members = member_list()
    hm, files = W.hash_members(members)
    W.cache().save()
    wid = W.ids(hm)
    world_id = wid["world_id"]
    manifest = {"schema": W.SCHEMA, "dance": DANCE, **wid, "created": now_iso(), "reference_take": DANCE, "grid": spec.grid.to_dict(),
                "partial": True, "missing_members": MISSING,
                "kind": "new room, ONE handheld zooming phone camera, no room scan (planes + per-frame camera track)",
                "members": hm, "n_files": len(files), "bytes": sum(v[1] for v in files.values()),
                "builder": "dancecap.room_world_single_cam.build",
                "files": [{"path": k, "sha256": v[0], "bytes": v[1], "member": v[2]} for k, v in sorted(files.items())]}
    td, dd = W.tracked_dir(DANCE, world_id), W.data_dir(DANCE, world_id)
    existing = W.read_json(td / "manifest.json", None)
    if existing:
        manifest["created"] = existing.get("created", manifest["created"])
    created = False
    if not (existing and dd.is_dir() and W.World(DANCE, manifest, dd).verify("quick").ok):
        W._materialise(members, dd, link=False)
        created = True
    atomic_write_json(dd / "manifest.json", manifest)
    rep = W.World(DANCE, manifest, dd).verify("full")
    if not rep.ok:
        raise W.WorldError(f"world {world_id} failed verification: {rep.errors[:5]}")
    td.mkdir(parents=True, exist_ok=True)
    atomic_write_json(td / "manifest.json", manifest)
    if make_current:
        atomic_write_json(worlds_tracked(DANCE) / "CURRENT.json",
                          {"world_id": world_id, "world_sha256": wid["world_sha256"], "pose_id": wid["pose_id"], "avatar_id": wid["avatar_id"],
                           "partial": True, "updated": now_iso()})
    return {"world_id": world_id, "pose_id": wid["pose_id"], "created": created, "tracked": str(td), "data": str(dd),
            "bytes": manifest["bytes"], "n_files": manifest["n_files"], "seconds": round(time.time() - t0, 1), "verify": rep.to_dict()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--no-current", action="store_true")
    sub.add_parser("verify")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        print(json.dumps(build(not a.no_current), indent=1, default=str))
    else:
        cur = W.read_json(worlds_tracked(DANCE) / "CURRENT.json", None)
        w = W.load(DANCE, cur["world_id"])
        print(json.dumps(w.verify("full").to_dict(), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
