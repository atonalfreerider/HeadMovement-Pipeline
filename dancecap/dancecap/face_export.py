"""Faces -> the Unity capture (additive, like the shoes / hair exports; also called by export_unity.export).

    python -m dancecap.face_export <take> [--capture NAME] [--albedo face|videolook|bake] [--dry-run]

Per dancer, into every StreamingAssets capture of the take (capture.json provenance.take) and its work/<take>/export
mirror - nothing else in the folder is touched:
  smplx_<role>_face.bin  "DCXF" (same blob layout as the other .bin files: magic, u32 header length, JSON header padded
                         to 4, blocks listed in header["blocks"], each padded to 4):
                           vertex_ids (Vh,) i4      SMPL-X vertices the shapes move (map onto the seam-split skin
                                                    through its vertex_ids block)
                           deltas     (S, Vh, 3) f4 rest-pose offsets in metres, Unity axes (x, y, -z); applied before
                                                    skinning (Mesh.AddBlendShapeFrame at weight 100)
                           weights    (T, S) f4     per capture frame; SetBlendShapeWeight(s, 100 w) - signed, may
                                                    exceed 1 (keep legacyClampBlendShapeWeights off)
                         shapes = the SMPL-X expression dirs expr00..expr{n-1} (+ eyelid_left / eyelid_right: FLAME
                         eyelid shapes moved onto SMPL-X; their weights are PROCEDURAL blinks = stylisation)
  smplx_<role>.bin       the jaw (joint 22) and eye (23, 24) local rotations are replaced by smplx_face's (same time
                         grid; jaw = x-rotation only; eyes = the body fit's); joint positions are unchanged (the jaw
                         has no child joints and a joint's own rotation does not move it)
  smplx_<role>_skin.bin  + the mouth cap: 30 triangles appended to the faces block (header "mouth_cap") closing the
                         mouth bag, so the closed mouth's lip gap shows the dark mouth interior (existing triangle
                         indices are unchanged)
  <role>_albedo.png      the chosen texture: face = the portrait look (avatar_face, default), videolook = the
                         video-matched look, bake = the plain video bake; a face texture older than the bake is
                         refused (re-run `avatar_face <take> --composite`) unless allow_stale
  capture.json           + "smplx_face": {role: file}, "smplx_albedo_variant": {role: variant},
                         "provenance": {..., "faces": {...}}
Inputs: work/<take>/smplx_face/{<role>.npz, face_blendshapes.npz} (face_expr), avatar/<role>_albedo*.png.
Likeness / licences: the textures carry the dancers' likeness and the blendshapes are SMPL-X / FLAME-derived: they go
to work/ and the git-ignored StreamingAssets only, never to a tracked path.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

from .export_unity import (FRAME_FLOATS, N_JOINTS, ROLES, UNITY_STREAMING, _read_blocks, _write_blob, read_blob,
                           read_motion, rotvec_to_quat, to_unity_points, to_unity_quats)
from .take import PACKAGE_ROOT

FACE_MAGIC = b"DCXF"
ALBEDO_VARIANTS = {"face": "{r}_albedo_face.png", "videolook": "{r}_albedo_face_videolook.png", "bake": "{r}_albedo.png"}
JAW, LEYE, REYE = 22, 23, 24


def face_arrays(work: Path, role: str) -> dict:
    """smplx_face/<role>.npz + face_blendshapes.npz -> Unity-ready arrays (names, vertex_ids, deltas (S,Vh,3) Unity
    axes, weights (T,S), t, jaw/eye rotvecs, meta)."""
    fd = work / "smplx_face"
    with np.load(fd / f"{role}.npz", allow_pickle=True) as z:
        f = {k: z[k] for k in z.files}
    with np.load(fd / "face_blendshapes.npz", allow_pickle=True) as z:
        bs = {k: z[k] for k in z.files}
    names = [str(x) for x in bs["names"]]
    n_expr = int(f.get("n_expr", f["expression"].shape[1]))
    T = len(f["t"])
    cols, keep = [], []
    for s, nm in enumerate(names):
        if nm.startswith("expr"):
            k = int(nm[4:])
            if k < n_expr and k < f["expression"].shape[1]:
                cols.append(np.asarray(f["expression"][:, k], np.float32))
                keep.append(s)
        elif nm.startswith("eyelid_") and "eyelid" in f:
            en = [str(x) for x in f.get("eyelid_names", np.array(["eyelid_left", "eyelid_right"]))]
            if nm in en:
                cols.append(np.asarray(f["eyelid"][:, en.index(nm)], np.float32))
                keep.append(s)
    weights = np.stack(cols, 1) if cols else np.zeros((T, 0), np.float32)
    deltas = to_unity_points(np.asarray(bs["deltas"], np.float32)[keep]).astype(np.float32)
    jaw = np.asarray(f["jaw_pose"], float).reshape(T, 3)
    meta = {k: (str(f[k]) if np.asarray(f[k]).dtype.kind in "US" else np.asarray(f[k]).tolist())
            for k in ("shipped", "adopted_rest", "adopted_expression", "adopted_jaw", "jaw_rest_deg") if k in f}
    return {"names": [names[s] for s in keep], "vertex_ids": np.asarray(bs["vertex_ids"], np.int32), "deltas": deltas,
            "weights": weights.astype(np.float32), "t": np.asarray(f["t"], float), "jaw": jaw,
            "leye": np.asarray(f.get("leye_pose", np.zeros((T, 3))), float).reshape(T, 3),
            "reye": np.asarray(f.get("reye_pose", np.zeros((T, 3))), float).reshape(T, 3), "meta": meta}


def write_face_bin(path: Path, role: str, fa: dict) -> dict:
    S, Vh, T = len(fa["names"]), len(fa["vertex_ids"]), len(fa["weights"])
    header = {"format": "dancecap smplx face", "version": 1, "role": role, "frames": T, "shapes": S, "vertices": Vh,
              "names": fa["names"],
              "blocks": [["vertex_ids", "i4", [Vh]], ["deltas", "f4", [S, Vh, 3]], ["weights", "f4", [T, S]]],
              "coords": "unity (x, y, -z), metres, rest pose; add before skinning (blendshape weight 100 = 1.0)",
              "weights_note": "SetBlendShapeWeight(s, 100 * w); expression weights are signed and may exceed 1",
              "stylised": [n for n in fa["names"] if n.startswith("eyelid")],
              "t0": float(fa["t"][0]), "t1": float(fa["t"][-1]), "face": fa["meta"],
              "licence": "SMPL-X expression dirs (+ FLAME eyelid shapes): model-licensed, non-commercial; never commit"}
    _write_blob(path, FACE_MAGIC, header, [fa["vertex_ids"], fa["deltas"], fa["weights"]])
    return header


def read_face_bin(path: Path) -> tuple[dict, dict]:
    magic, h, body = read_blob(path)
    assert magic == FACE_MAGIC, magic
    return h, _read_blocks(h, body)


def patch_motion(path: Path, fa: dict) -> dict:
    """Replace joints 22-24's local rotations in a DCXM motion file by the face block's (grids must match)."""
    magic, h, body = read_blob(path)
    _, frames = read_motion(path)
    T = h["frames"]
    if T != len(fa["t"]) or abs(h["t0"] - fa["t"][0]) > 1e-4 or abs(h["t1"] - fa["t"][-1]) > 1e-4:
        raise ValueError(f"{path.name}: motion grid ({T} frames {h['t0']:.3f}..{h['t1']:.3f}) differs from smplx_face's "
                         f"({len(fa['t'])} frames {fa['t'][0]:.3f}..{fa['t'][-1]:.3f})")
    fr = frames.astype(np.float32).copy()
    before = {}
    for j, rv in ((JAW, fa["jaw"]), (LEYE, fa["leye"]), (REYE, fa["reye"])):
        c = 3 + 4 * j
        q = to_unity_quats(rotvec_to_quat(rv)).astype(np.float32)
        before[j] = float(np.degrees(2 * np.arccos(np.clip(np.abs(fr[:, c + 3]), 0, 1))).max())
        fr[:, c:c + 4] = q
    assert fr.shape[1] == FRAME_FLOATS and N_JOINTS == 55
    _write_blob(path, magic, h, [fr])
    return {"jaw_deg_max_before": round(before[JAW], 2),
            "jaw_deg_max_after": round(float(np.degrees(np.linalg.norm(fa["jaw"], axis=1)).max()), 2)}


def cap_skin(path: Path) -> dict:
    """Close SMPL-X's one open boundary - the mouth bag ~16 mm behind the lips - in a written DCXS skin (appends the
    avatar_face.mouth_cap triangles in Unity winding, on the split vertices whose UV is the mouth interior's, so they
    show its dark paint). Without it the closed mouth's 1.2 mm lip gap shows the inside of the back of the head (a pale
    line in the reviews; the background behind the avatar in Unity, which culls back faces). Idempotent."""
    from .avatar_face import mouth_cap
    from .face_common import smplx_model_npz
    magic, h, body = read_blob(path)
    if h.get("mouth_cap"):
        return {"mouth_cap_triangles": int(h["mouth_cap"]), "already": True}
    b = {k: np.array(v) for k, v in _read_blocks(h, body).items()}
    tris, tft = mouth_cap()
    if not len(tris):
        return {"mouth_cap_triangles": 0}
    vid, uv = b.get("vertex_ids"), b.get("uv")
    if vid is None or uv is None:
        cap = tris
    else:
        vt = np.asarray(np.load(smplx_model_npz(), allow_pickle=True)["vt"], float)
        cap = np.zeros_like(tris)
        for n_, (tri, tt) in enumerate(zip(tris, tft)):
            for c in range(3):
                cand = np.flatnonzero(vid == tri[c])
                if not len(cand):
                    raise ValueError(f"{path.name}: SMPL-X vertex {tri[c]} not in the split skin")
                cap[n_, c] = cand[np.argmin(np.abs(uv[cand] - vt[tt[c]]).sum(1))]
    cap_u = cap[:, [0, 2, 1]].astype(np.int32)              # Unity winding (export_unity.unity_skin reverses)
    b["faces"] = np.concatenate([b["faces"].astype(np.int32), cap_u], 0)
    for blk in h["blocks"]:
        if blk[0] == "faces":
            blk[2] = [int(len(b["faces"])), 3]
    h["faces"] = int(len(b["faces"]))
    h["mouth_cap"] = int(len(cap_u))
    _write_blob(path, magic, h, [np.ascontiguousarray(b[blk[0]]).astype(blk[1]) for blk in h["blocks"]])
    return {"mouth_cap_triangles": int(len(cap_u))}


def albedo_source(work: Path, role: str, variant: str, allow_stale: bool = False) -> tuple[Path | None, str | None]:
    """-> (png, warning). Falls back to the bake (with a warning) when the face texture is missing or stale."""
    av = work / "avatar"
    bake = av / f"{role}_albedo.png"
    p = av / ALBEDO_VARIANTS[variant].format(r=role)
    if variant == "bake":
        return (bake if bake.exists() else None), None
    if not p.exists():
        return (bake if bake.exists() else None), f"{role}: {p.name} missing (run avatar_face) - the bake is used"
    if bake.exists() and p.stat().st_mtime < bake.stat().st_mtime and not allow_stale:
        return bake, (f"{role}: {p.name} is older than {bake.name} (re-run `python -m dancecap.avatar_face "
                      f"{work.name} --composite`) - the bake is used")
    return p, None


def add_faces(work: Path, out: Path, manifest: dict, albedo: str = "face", warnings: list | None = None,
              allow_stale: bool = False, patch: bool = True) -> dict:
    """Write the face files into a capture folder `out` whose motion / skin bins exist and update `manifest` (the
    capture.json dict) in place. -> report."""
    warnings = warnings if warnings is not None else []
    rep = {"albedo_variant": albedo, "dancers": {}}
    if albedo not in ALBEDO_VARIANTS:
        raise ValueError(f"albedo must be one of {sorted(ALBEDO_VARIANTS)}")
    for r in ROLES:
        rr = {}
        if (work / "smplx_face" / f"{r}.npz").exists() and (work / "smplx_face" / "face_blendshapes.npz").exists():
            fa = face_arrays(work, r)
            mot = out / f"smplx_{r}.bin"
            try:
                if patch and mot.exists():
                    rr["motion"] = patch_motion(mot, fa)
                h = write_face_bin(out / f"smplx_{r}_face.bin", r, fa)
                manifest.setdefault("smplx_face", {})[r] = f"smplx_{r}_face.bin"
                rr["face_bin"] = {"shapes": h["shapes"], "vertices": h["vertices"], "frames": h["frames"],
                                  "names": h["names"], "face": h["face"]}
            except ValueError as e:
                warnings.append(str(e))
                rr["error"] = str(e)
        skin = out / f"smplx_{r}_skin.bin"
        textured = skin.exists() and read_blob(skin)[1].get("version", 1) >= 2
        if skin.exists():
            try:
                rr["skin"] = cap_skin(skin)
            except ValueError as e:
                warnings.append(str(e))
        src, warn = albedo_source(work, r, albedo, allow_stale)
        if warn:
            warnings.append(warn)
        if src is not None and textured:
            shutil.copy2(src, out / f"{r}_albedo.png")
            manifest.setdefault("smplx_albedo", {})[r] = f"{r}_albedo.png"
            manifest.setdefault("smplx_albedo_variant", {})[r] = albedo if src.name != f"{r}_albedo.png" else "bake"
            rr["albedo"] = src.name
        rep["dancers"][r] = rr
    manifest.setdefault("provenance", {})["faces"] = {
        "source": "dancecap.face_export", "albedo_variant": manifest.get("smplx_albedo_variant"),
        "blinks": "procedural (stylisation)", "expression": {r: v.get("face_bin", {}).get("face", {}).get("shipped")
                                                             for r, v in rep["dancers"].items()}}
    return rep


def captures_for(take: str, streaming: Path = UNITY_STREAMING) -> list[Path]:
    out = []
    for d in sorted(Path(streaming).iterdir()) if Path(streaming).exists() else []:
        cj = d / "capture.json"
        if d.is_dir() and cj.exists():
            try:
                man = json.loads(cj.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if (man.get("provenance") or {}).get("take") == take:
                out.append(d)
    return out


def export(take: str, capture: str | None = None, albedo: str = "face", streaming: Path = UNITY_STREAMING,
           dry_run: bool = False, allow_stale: bool = False) -> dict:
    """Install the faces into the take's existing captures (StreamingAssets + the work/<take>/export mirror)."""
    work = PACKAGE_ROOT / "work" / take
    dirs = [Path(streaming) / capture] if capture else captures_for(take, streaming)
    if not dirs:
        raise FileNotFoundError(f"no StreamingAssets capture with provenance.take = {take} (run `python -m dancecap "
                                f"export {take}` first)")
    res = {}
    for d in dirs:
        for dst in (d, work / "export" / d.name):
            if not (dst / "capture.json").exists():
                continue
            if dry_run:
                res[str(dst)] = {"dry_run": True, "would_write": [f"smplx_{r}_face.bin" for r in ROLES] +
                                 [f"{r}_albedo.png" for r in ROLES] + ["capture.json (smplx_face, smplx_albedo_variant)",
                                                                      "smplx_<role>.bin (joints 22-24)"]}
                continue
            man = json.loads((dst / "capture.json").read_text(encoding="utf-8"))
            warnings: list = []
            rep = add_faces(work, dst, man, albedo, warnings, allow_stale)
            (dst / "capture.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
            rep["warnings"] = warnings
            res[str(dst)] = rep
    return res


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="faces (expression / blink blendshapes, jaw, face texture) -> Unity capture")
    ap.add_argument("take")
    ap.add_argument("--capture", help="StreamingAssets capture folder (default: every capture of the take)")
    ap.add_argument("--albedo", choices=sorted(ALBEDO_VARIANTS), default="face",
                    help="face = portrait look (default), videolook = video-matched look, bake = plain video bake")
    ap.add_argument("--allow-stale", action="store_true", help="use a face texture older than the bake")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    res = export(a.take, a.capture, a.albedo, dry_run=a.dry_run, allow_stale=a.allow_stale)
    print(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
