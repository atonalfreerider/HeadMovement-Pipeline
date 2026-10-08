"""The WORLD PACKAGE: what every analysis of a dance shares (FARM.md section 3).

A world is built ONCE from the solved reference take (larissa_kadu_38 for larissa_kadu_full) and then used, unchanged and
hash-verified, by every chunk job on every pod: the room model (SfM map + cached features + keyframes + the map's window
images), the world frame (gravity, floor, scale), the camera clocks and types, the per-dancer betas, the skeleton
configuration (SMPL-X joint set, the MAMMA ma_3d config, the refine recipe) and the avatar configuration (shape edits,
texture / groom / shoe file HASHES: references only).

    dancecap/worlds/<dance>/<world_id>/manifest.json   tracked: member hashes + the sha256 of every data file + relative
                                                       paths, NO likeness data
    dancecap/worlds/<dance>/CURRENT.json               tracked: the world new jobs use
    dancecap/worlds/<dance>/seals.json                 tracked: {pose_id: {...}} sealed room models
    work/worlds/<dance>/<world_id>/{ref/, shared/, manifest.json}   git-ignored data; the same tree sits on
                                                       the Runpod volume at /workspace/hm/worlds/<world_id>/

IDs. Every member has a sha256 (a file's, or the tree hash of its files: sha256 over sorted "path\\0sha\\0bytes\\n" lines).
`world_sha256` = sha256 over the sorted "member\\0sha\\n" lines of ALL members and `world_id` its first 12 hex digits, as the
contract says. Two more ids split the members by what they influence: `pose_id` covers the members that determine the 3D
poses (room, world frame, scale, clocks, camera types, betas, skeleton) and `avatar_id` the avatar configuration.
Rebaking a face or a hair groom changes `world_id` (a NEW world; old ids stay readable) but NOT `pose_id`: poses computed
under one pose_id are interchangeable (the stitcher and exporters compare pose_id; see check_same_world).

`ref/` mirrors the work directory layout of the reference take (cameras/sfm, cameras/sfm_work, cameras/cache/feats,
keyframes, people_masks_kf, frames/<cam>/frames.json + the map's window images, sync/offsets.json, ...), so the
existing front-end code that reads "the reference take" (front_full.prep_segment) can run against `ref/` unchanged.

    python -m dancecap.farm world publish          build + verify + install the world from the reference take
    python -m dancecap.farm world verify [--full]  re-hash the data tree against the manifest
    python -m dancecap.farm world seal             the room model is solved: freeze this pose_id
    python -m dancecap.farm world compare <take>   does a take really share the room / world frame / betas?
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from . import DEFAULT_DANCE, VOLUME_ROOT, worlds_data, worlds_tracked
from .state import atomic_write_json, now_iso, read_json
from ..take import PACKAGE_ROOT, WORKSPACE

SCHEMA = "dancecap.farm.world/1"
ID_LEN = 12
POSE_SCOPE, AVATAR_SCOPE = "pose", "avatar"
HASH_CACHE = PACKAGE_ROOT / "work" / "worlds" / ".hash_cache.json"
AVATAR_FILES = ("avatar/lead_albedo.png", "avatar/follow_albedo.png", "avatar/lead_albedo_face.png",
                "avatar/follow_albedo_face.png", "avatar/hair_groom.json", "avatar/follow_hair.json", "avatar/shoes.json",
                "avatar/shoe_size.json", "avatar/head_fix.npz", "avatar/uv_mesh.npz", "avatar/lead_texels.npz",
                "avatar/follow_texels.npz")
BODY_JOINTS = ["pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2", "left_ankle", "right_ankle",
               "spine3", "left_foot", "right_foot", "neck", "left_collar", "right_collar", "head", "left_shoulder",
               "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"]
SMPLX_JOINTS = (BODY_JOINTS + ["jaw", "left_eye_smplhf", "right_eye_smplhf"]
                + [f"{side}_{f}{j}" for side in ("left", "right") for f in ("index", "middle", "pinky", "ring", "thumb")
                   for j in (1, 2, 3)])


class WorldError(RuntimeError):
    pass


class WorldMismatch(WorldError):
    """A job / chunk / output belongs to another world than the one it is being used with."""


# ----------------------------------------------------------------------------------------------------- hashing


class HashCache:
    """sha256 of files keyed by (path, size, mtime_ns): a re-hash of 330 MB of unchanged files costs milliseconds.
    `full` verification bypasses it."""

    def __init__(self, path: Path = HASH_CACHE):
        self.path = path
        self.d: dict = {}
        self.dirty = False
        try:
            self.d = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.d = {}

    def sha(self, p: Path, use_cache: bool = True) -> tuple[str, int]:
        st = p.stat()
        key = os.path.abspath(p).replace("\\", "/")
        hit = self.d.get(key) if use_cache else None
        if hit and hit[0] == st.st_size and hit[1] == st.st_mtime_ns:
            return hit[2], st.st_size
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for blk in iter(lambda: fh.read(1 << 20), b""):
                h.update(blk)
        self.d[key] = [st.st_size, st.st_mtime_ns, h.hexdigest()]
        self.dirty = True
        return h.hexdigest(), st.st_size

    def save(self) -> None:
        if self.dirty:
            try:
                atomic_write_json(self.path, self.d, indent=None)
                self.dirty = False
            except OSError:
                pass


_CACHE: HashCache | None = None


def cache() -> HashCache:
    global _CACHE
    if _CACHE is None:
        _CACHE = HashCache()
    return _CACHE


def sha256_file(path: Path, use_cache: bool = True) -> str:
    return cache().sha(Path(path), use_cache)[0]


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def tree_sha(entries: Iterable[tuple[str, str, int]]) -> str:
    """Hash of a set of files: sha256 over the sorted 'relpath NUL sha256 NUL bytes LF' lines."""
    h = hashlib.sha256()
    for rel, sha, n in sorted(entries):
        h.update(f"{rel}\0{sha}\0{n}\n".encode())
    return h.hexdigest()


def hash_tree(root: Path, use_cache: bool = True, rel_to: Path | None = None) -> tuple[str, list[tuple[str, str, int]]]:
    """(tree sha256, [(relpath, sha256, bytes)]) of every file under root (relative to rel_to, default root)."""
    root = Path(root)
    base = Path(rel_to) if rel_to else root
    ent = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            sha, n = cache().sha(p, use_cache)
            ent.append((p.relative_to(base).as_posix(), sha, n))
    return tree_sha(ent), ent


def member_sha(entries: list[tuple[str, str, int]]) -> str:
    """A member's sha256: the FILE's own sha256 when it is one file (the worker re-hashes member files against it), else the
    tree hash of its files."""
    return entries[0][1] if len(entries) == 1 else tree_sha(entries)


def combine(members: dict[str, dict], scope: str | None = None) -> str:
    """sha256 over the sorted 'member NUL sha' lines (optionally only the members of one scope)."""
    h = hashlib.sha256()
    for name in sorted(members):
        m = members[name]
        if scope is None or m["scope"] == scope:
            h.update(f"{name}\0{m['sha256']}\n".encode())
    return h.hexdigest()


# --------------------------------------------------------------------------------------------------- members


@dataclass
class Entry:
    dst: str                       # path inside the world directory (posix)
    src: Path | None = None        # a file to copy ...
    data: bytes | None = None      # ... or generated content


@dataclass
class Member:
    name: str
    scope: str
    note: str
    entries: list[Entry] = field(default_factory=list)
    source: str = ""               # workspace-relative description of where it came from


def _rel_ws(p: Path) -> str:
    try:
        return Path(p).resolve().relative_to(WORKSPACE.resolve()).as_posix()
    except ValueError:
        return str(p).replace("\\", "/")


def _tree(src: Path, dst: str) -> list[Entry]:
    return [Entry(f"{dst}/{p.relative_to(src).as_posix()}", src=p) for p in sorted(src.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts]


def _file(src: Path, dst: str) -> list[Entry]:
    return [Entry(dst, src=src)]


def _json_bytes(obj) -> bytes:
    return (json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def member_specs(spec) -> list[Member]:
    """The members of the dance's world, from its reference take's solved outputs and the dance TOML (spec: timeline.DanceSpec)."""
    import numpy as np
    if not spec.reference:
        raise WorldError(f"{spec.name}: takes/{spec.name}.toml has no [reuse] take: nothing to build a world from")
    ref = PACKAGE_ROOT / "work" / spec.reference
    cfg = spec.cfg
    need = [ref / "cameras" / "sfm" / "images.bin", ref / "cameras" / "sfm_work" / "map_images.json",
            ref / "cameras" / "world_from_sfm.json", ref / "cameras" / "cache" / "feats", ref / "keyframes",
            ref / "sync" / "offsets.json", ref / "mamma" / "ma3d_config.yaml", ref / "smplx_rf2_noslip" / "pose_refine.json"]
    miss = [str(p) for p in need if not p.exists()]
    if miss:
        raise WorldError(f"reference take {spec.reference} is missing: {miss}")
    M: list[Member] = []

    def add(name, scope, note, entries, source=""):
        M.append(Member(name, scope, note, entries, source))

    # ---- room model ----
    add("room_sfm", POSE_SCOPE, "bundle-adjusted SfM sparse model of the room (main model)",
        _tree(ref / "cameras" / "sfm", "ref/cameras/sfm"), _rel_ws(ref / "cameras" / "sfm"))
    add("room_sfm_work", POSE_SCOPE, "map.json, map_images.json and the attached cameras' sub-models",
        _tree(ref / "cameras" / "sfm_work", "ref/cameras/sfm_work"), _rel_ws(ref / "cameras" / "sfm_work"))
    add("room_features", POSE_SCOPE, "cached ALIKED features of every map image (index the model's 2D points)",
        _tree(ref / "cameras" / "cache" / "feats", "ref/cameras/cache/feats"), _rel_ws(ref / "cameras" / "cache" / "feats"))
    add("room_keyframes", POSE_SCOPE, "map keyframes (jpg + keyframes.json) and their people masks",
        _tree(ref / "keyframes", "ref/keyframes") + _tree(ref / "people_masks_kf", "ref/people_masks_kf"),
        _rel_ws(ref / "keyframes"))
    imgs = json.loads((ref / "cameras" / "sfm_work" / "map_images.json").read_text(encoding="utf-8"))["images"]
    win = sorted(n for n in imgs if n.startswith("frames/"))
    fe = []
    for c in sorted({n.split("/")[1] for n in win}):
        fe.append(Entry(f"ref/frames/{c}/frames.json", src=ref / "frames" / c / "frames.json"))
    for n in win:
        fe.append(Entry(f"ref/{n}", src=ref / n))
        mask = ref / "people_masks" / n.split("/")[1] / (Path(n).stem + ".png")
        if mask.exists():
            fe.append(Entry(f"ref/people_masks/{n.split('/')[1]}/{mask.name}", src=mask))
    add("room_map_frames", POSE_SCOPE, "the map's window images (not keyframes) with their frames.json and masks",
        fe, _rel_ws(ref / "frames"))
    # ---- world frame, scale, clocks ----
    add("world_frame", POSE_SCOPE, "world_from_sfm.json: gravity, floor and the betas-corrected scale (Sim3)",
        _file(ref / "cameras" / "world_from_sfm.json", "ref/cameras/world_from_sfm.json"),
        _rel_ws(ref / "cameras" / "world_from_sfm.json"))
    sc = sorted((ref / "scale").glob("reference_*world_corrected.json"))
    if sc:
        add("scale_reference", POSE_SCOPE, "the scale reference the world frame inherits (betas rest heights vs statures)",
            _file(sc[0], f"ref/scale/{sc[0].name}"), _rel_ws(sc[0]))
    alg = (PACKAGE_ROOT / "takes" / cfg["alignment"]).resolve()
    add("camera_clocks", POSE_SCOPE, "alignment_studio.json: affine audio clock of every phone on the reference clock",
        _file(alg, "shared/alignment_studio.json"), _rel_ws(alg))
    add("clock_gauge", POSE_SCOPE, "the reference take's measured per-camera video offsets (segment sync is gauge-pinned to them)",
        _file(ref / "sync" / "offsets.json", "ref/sync/offsets.json"), _rel_ws(ref / "sync" / "offsets.json"))
    cams = {c: {k: v[k] for k in ("mount", "zoom", "source") if k in v}
            for c, v in sorted((cfg.get("cameras") or {}).items()) if isinstance(v, dict)}
    add("camera_types", POSE_SCOPE, "per-camera mount / zoom (user-confirmed), without timestamps",
        [Entry("shared/camera_types.json", data=_json_bytes({"schema": SCHEMA + "/camera_types", "cameras": cams}))],
        f"takes/{spec.name}.toml [cameras]")
    # ---- betas ----
    pin = PACKAGE_ROOT / "work" / spec.name / "mamma_pin_betas.json"
    pin_src = pin.with_name("mamma_pin_betas.source.json")
    if not pin.exists():
        raise WorldError(f"{pin} missing (python -m dancecap.front_full pin-betas)")
    doc = json.loads(pin.read_text())
    srcdoc = json.loads(pin_src.read_text()) if pin_src.exists() else {}
    for key, m in srcdoc.items():   # the pinned betas ARE the reference take's: check, never assume
        z = np.load(ref / "smplx" / f"{m['role']}.npz")
        err = float(np.abs(np.asarray(doc[key], float) - np.asarray(z["betas"], float).reshape(-1)).max())
        if err > 1e-5:
            raise WorldError(f"{pin.name}: body {key} ({m['role']}) differs from {spec.reference}/smplx by {err:.2e}")
    add("betas", POSE_SCOPE, "per-dancer body shape pinned in every fit (MAMMA_INIT_BETAS / MAMMA_FREEZE_BETAS format)",
        _file(pin, "shared/betas.json") + ([Entry("shared/betas.source.json", src=pin_src)] if pin_src.exists() else []),
        _rel_ws(pin))
    # ---- skeleton configuration ----
    from .. import mamma_dyn as md
    rr = json.loads((ref / "smplx_rf2_noslip" / "pose_refine.json").read_text())
    r0 = rr["lead"]
    refine = {"schema": SCHEMA + "/refine_recipe", "name": "smplx_rf2_noslip", "tool": "dancecap.pose_refine",
              "src": "smplx", "dst": "smplx_rf2_noslip", "mode": r0["mode"],
              "filter": {k: r0["filter"][k] for k in ("mode", "root_hz", "rot_hz")},
              "refine": {k: r0["refine"][k] for k in ("iters", "w_floor", "floor_mm", "w_feet", "w_slip", "slip_max_v",
                                                      "root_xz", "w_anchor", "anchor_mm")},
              "margin_s": 2.0, "device": "cuda", "reference": f"{spec.reference}/smplx_rf2_noslip/pose_refine.json"}
    add("refine_recipe", POSE_SCOPE, "pose_refine parameters of the accepted recipe (smplx_rf2_noslip)",
        [Entry("shared/refine_recipe.json", data=_json_bytes(refine))], _rel_ws(ref / "smplx_rf2_noslip" / "pose_refine.json"))
    z = np.load(ref / "smplx_rf2_noslip" / "lead.npz", allow_pickle=False)
    skel = {"schema": SCHEMA + "/skeleton", "model": str(z["model"]), "gender": str(z["gender"]),
            "num_betas": int(z["num_betas"]), "flat_hand_mean": bool(z["flat_hand_mean"]),
            "joints": {"n": len(SMPLX_JOINTS), "names": SMPLX_JOINTS},
            "pose_layout": {"global_orient": 3, "body_pose": 63, "jaw_pose": 3, "leye_pose": 3, "reye_pose": 3,
                            "left_hand_pose": 45, "right_hand_pose": 45, "transl": 3},
            "npz_keys": sorted(z.files),
            "roles": {str(k): v for k, v in sorted(md.ROLES.items())}, "grid": {"hz": spec.grid.hz, "anchor": spec.grid.anchor},
            "ma_3d": {"base_config": md.MA3D_CONFIG, "overrides": md.MA3D_OVERRIDES,
                      "config_file": "ref/mamma/ma3d_config.yaml", "n_landmarks": md.N_LMK},
            "betas_pin": {"file": "shared/betas.json", "env": {"MAMMA_INIT_BETAS": "{world}/shared/betas.json",
                                                              "MAMMA_FREEZE_BETAS": "1"}},
            "refine_recipe": "shared/refine_recipe.json"}
    add("skeleton", POSE_SCOPE, "SMPL-X joint set + parameter layout + ma_3d settings + pinning rule",
        [Entry("shared/skeleton.json", data=_json_bytes(skel))], "mamma_dyn constants + reference smplx contract")
    add("ma3d_config", POSE_SCOPE, "the MAMMA ma_3d config template the reference fit used (contact config + overrides)",
        _file(ref / "mamma" / "ma3d_config.yaml", "ref/mamma/ma3d_config.yaml"), _rel_ws(ref / "mamma" / "ma3d_config.yaml"))
    # ---- avatar configuration (references only) ----
    files = {}
    for f in AVATAR_FILES:
        p = ref / f
        if p.exists():
            sha, n = cache().sha(p)
            files[f] = {"path": _rel_ws(p), "sha256": sha, "bytes": n}
    dancers = {}
    for role in ("lead", "follow"):
        d = (cfg.get("dancers") or {}).get(role) or {}
        dancers[role] = {k: d[k] for k in ("stature_m", "shoe_sole_m", "shoe_geometry", "shape_edits") if k in d}
    av = {"schema": SCHEMA + "/avatar", "reference_take": spec.reference, "dancers": dancers, "files": files,
          "note": "references only: shape edits and file hashes; textures, grooms and shoes are never copied here"}
    add("avatar", AVATAR_SCOPE, "avatar configuration: shape edits and texture / groom / shoe file hashes (references)",
        [Entry("shared/avatar.json", data=_json_bytes(av))], f"takes/{spec.name}.toml + {_rel_ws(ref / 'avatar')}")
    return M


# ------------------------------------------------------------------------------------------------- the world


def _entry_hash(e: Entry, use_cache: bool = True) -> tuple[str, int]:
    if e.data is not None:
        return sha256_bytes(e.data), len(e.data)
    return cache().sha(e.src, use_cache)


def hash_members(members: list[Member]) -> tuple[dict, dict]:
    """-> ({name: {scope, note, source, sha256, bytes, n_files, path}}, {dst: [sha256, bytes, member]})."""
    out, files = {}, {}
    for m in members:
        ent = []
        for e in m.entries:
            sha, n = _entry_hash(e)
            ent.append((e.dst, sha, n))
            files[e.dst] = [sha, n, m.name]
        dsts = sorted(e.dst for e in m.entries)
        out[m.name] = {"scope": m.scope, "note": m.note, "source": m.source, "sha256": member_sha(ent),
                       "bytes": sum(x[2] for x in ent), "n_files": len(ent),
                       "root": dsts[0] if len(dsts) == 1 else os.path.commonpath(dsts).replace("\\", "/") + "/"}
    return out, files


def ids(members: dict) -> dict:
    ws, ps, av = combine(members), combine(members, POSE_SCOPE), combine(members, AVATAR_SCOPE)
    return {"world_sha256": ws, "world_id": ws[:ID_LEN], "pose_sha256": ps, "pose_id": ps[:ID_LEN],
            "avatar_sha256": av, "avatar_id": av[:ID_LEN]}


def tracked_dir(dance: str, world_id: str) -> Path:
    return worlds_tracked(dance) / world_id


def data_dir(dance: str, world_id: str) -> Path:
    return worlds_data(dance) / world_id


def volume_path(world_id: str, root: str = VOLUME_ROOT) -> str:
    """Where the world's data tree sits on the Runpod volume (the manifest names the dance)."""
    return f"{root}/worlds/{world_id}"


class World:
    def __init__(self, dance: str, manifest: dict, root: Path):
        self.dance, self.manifest, self.root = dance, manifest, Path(root)

    # identity
    @property
    def id(self) -> str:
        return self.manifest["world_id"]

    @property
    def sha256(self) -> str:
        return self.manifest["world_sha256"]

    @property
    def pose_id(self) -> str:
        return self.manifest["pose_id"]

    @property
    def avatar_id(self) -> str:
        return self.manifest["avatar_id"]

    @property
    def members(self) -> dict:
        return self.manifest["members"]

    @property
    def sealed(self) -> bool:
        return self.pose_id in seals(self.dance)

    def header(self) -> dict:
        """What every chunk output records about its world (see check_same_world)."""
        return {"dance": self.dance, "world_id": self.id, "world_sha256": self.sha256, "pose_id": self.pose_id,
                "avatar_id": self.avatar_id}

    # access
    def path(self, rel: str) -> Path:
        return self.root / rel

    def ref(self, rel: str = "") -> Path:
        """The reference-take-shaped tree (cameras/, keyframes/, sync/, ...)."""
        return self.root / "ref" / rel

    def load_json(self, rel: str):
        return json.loads(self.path(rel).read_text(encoding="utf-8"))

    def betas_file(self) -> Path:
        return self.path("shared/betas.json")

    def files(self) -> dict[str, tuple[str, int, str]]:
        """{relative path: (sha256, bytes, member)} from the manifest's `files` list."""
        return {f["path"]: (f["sha256"], int(f["bytes"]), f.get("member", "")) for f in self.manifest.get("files", [])}

    def volume_path(self, root: str = VOLUME_ROOT) -> str:
        return volume_path(self.id, root)

    def verify(self, level: str = "full") -> "VerifyReport":
        """level 'full': re-hash every file (cache bypassed) and recompute every member hash, the world_sha256 and
        the ids from the manifest; 'quick': the file list and sizes only (what a pod checks at job start)."""
        t0 = time.time()
        errors: list[str] = []
        files = self.files()
        if not files:
            errors.append("manifest has no `files` list")
        n = 0
        for rel, (sha, size, _m) in files.items():
            p = self.root / rel
            if not p.is_file():
                errors.append(f"missing {rel}")
                continue
            if p.stat().st_size != size:
                errors.append(f"size differs {rel}: {p.stat().st_size} != {size}")
                continue
            if level == "full":
                got = cache().sha(p, use_cache=False)[0]
                if got != sha:
                    errors.append(f"sha256 differs {rel}")
            n += 1
        extra = []
        if self.root.is_dir():
            on_disk = {p.relative_to(self.root).as_posix() for p in self.root.rglob("*") if p.is_file()}
            extra = sorted(on_disk - set(files) - {"manifest.json"})
            if extra:
                errors.append(f"{len(extra)} file(s) not in the manifest, e.g. {extra[:3]}")
        # the manifest must be self-consistent: files -> member hashes -> world sha -> id
        by_member: dict[str, list] = {}
        for rel, (sha, size, mem) in files.items():
            by_member.setdefault(mem, []).append((rel, sha, size))
        for name, m in self.members.items():
            if member_sha(by_member.get(name, [])) != m["sha256"]:
                errors.append(f"member {name}: hash does not match its files")
        want = ids(self.members)
        for k in ("world_sha256", "world_id", "pose_sha256", "pose_id", "avatar_sha256", "avatar_id"):
            if self.manifest.get(k) != want[k]:
                errors.append(f"manifest {k} {self.manifest.get(k)} != {want[k]}")
        cache().save()
        return VerifyReport(ok=not errors, errors=errors, files=n, level=level, seconds=round(time.time() - t0, 2))


@dataclass
class VerifyReport:
    ok: bool
    errors: list[str]
    files: int
    level: str
    seconds: float

    def to_dict(self) -> dict:
        return {"ok": self.ok, "errors": self.errors[:20], "n_errors": len(self.errors), "files": self.files,
                "level": self.level, "seconds": self.seconds}


# ------------------------------------------------------------------------------------- pointers, seals, loading


def current(dance: str = DEFAULT_DANCE) -> dict | None:
    return read_json(worlds_tracked(dance) / "CURRENT.json", None)


def seals(dance: str = DEFAULT_DANCE) -> dict:
    return read_json(worlds_tracked(dance) / "seals.json", {}) or {}


def list_worlds(dance: str = DEFAULT_DANCE) -> list[dict]:
    out = []
    d = worlds_tracked(dance)
    if d.is_dir():
        for p in sorted(d.iterdir()):
            m = read_json(p / "manifest.json", None) if p.is_dir() else None
            if m:
                out.append({"world_id": m["world_id"], "pose_id": m["pose_id"], "avatar_id": m["avatar_id"],
                            "created": m.get("created"), "sealed": m["pose_id"] in seals(dance),
                            "data": data_dir(dance, m["world_id"]).is_dir()})
    return out


def manifest_of(dance: str, world_id: str) -> dict:
    """The tracked manifest of a world id (a unique prefix is accepted)."""
    d = worlds_tracked(dance)
    cands = [p for p in d.iterdir() if p.is_dir() and p.name.startswith(world_id)] if d.is_dir() else []
    if not cands:   # a pod's code bundle may not carry dancecap/worlds: the data tree has its own copy of the manifest
        dd = worlds_data(dance)
        cands = [p for p in dd.iterdir() if p.is_dir() and p.name.startswith(world_id) and (p / "manifest.json").exists()] \
            if dd.is_dir() else []
    if len(cands) != 1:
        raise WorldError(f"world {world_id!r} of {dance}: {'unknown' if not cands else 'ambiguous'}")
    return json.loads((cands[0] / "manifest.json").read_text(encoding="utf-8"))


def load(dance: str = DEFAULT_DANCE, world_id: str | None = None, root: Path | None = None) -> World:
    """The world `world_id` (default: CURRENT) with its data under `root` (default work/worlds/<dance>/<world_id>)."""
    if world_id is None:
        cur = current(dance)
        if not cur:
            raise WorldError(f"{dance}: no world published yet (python -m dancecap.farm world publish)")
        world_id = cur["world_id"]
    if root is not None and (Path(root) / "manifest.json").exists():   # an explicit data tree (a volume copy) names itself
        man = json.loads((Path(root) / "manifest.json").read_text(encoding="utf-8"))
        if not man["world_id"].startswith(world_id):
            raise WorldMismatch(f"{root} holds world {man['world_id']}, not {world_id}")
    else:
        man = manifest_of(dance, world_id)
    return World(dance, man, Path(root) if root else data_dir(dance, man["world_id"]))


def require(dance: str, world_id: str, world_sha256: str | None = None, root: Path | None = None,
            level: str = "quick") -> World:
    """What a job / worker / stitcher calls before using a world: the world must exist, match the given sha256 and its data
    tree must verify (quick = file list + sizes, full = hashes). Raises WorldMismatch / WorldError."""
    w = load(dance, world_id, root)
    if world_sha256 and w.sha256 != world_sha256:
        raise WorldMismatch(f"world {world_id}: sha256 {w.sha256} != the job's {world_sha256}")
    rep = w.verify(level)
    if not rep.ok:
        raise WorldMismatch(f"world {world_id} at {w.root} does not verify ({level}): {rep.errors[:3]}")
    return w


def check_same_world(items: Iterable[dict], dance: str = DEFAULT_DANCE, allow_mixed: bool = False) -> dict:
    """The stitcher / exporter guard: items are chunk or output headers with a `world_id` (and optionally `pose_id`).
    Different pose_ids (different room / frame / scale / clocks / betas / skeleton) are refused unless allow_mixed;
    different world_ids with one pose_id (an avatar-only difference) pass with a warning."""
    items = [dict(i) for i in items]
    for i in items:
        if not i.get("world_id"):
            raise WorldMismatch(f"output {i.get('id') or i.get('chunk') or '?'} has no world_id")
        if not i.get("pose_id"):
            try:
                i["pose_id"] = manifest_of(dance, i["world_id"])["pose_id"]
            except WorldError:
                i["pose_id"] = None
    wids, pids = sorted({i["world_id"] for i in items}), sorted({i["pose_id"] or i["world_id"] for i in items})
    rep = {"ok": True, "world_ids": wids, "pose_ids": pids, "warnings": []}
    if len(pids) > 1 and not allow_mixed:
        raise WorldMismatch(f"chunks come from different worlds (pose ids {pids}); refusing without --allow-mixed-world")
    if len(pids) > 1:
        rep["warnings"].append(f"MIXED worlds allowed by flag: pose ids {pids}")
    elif len(wids) > 1:
        rep["warnings"].append(f"world ids {wids} differ only in the avatar configuration (same pose id {pids[0]})")
    return rep


# ----------------------------------------------------------------------------------------- build / publish / seal


@dataclass
class BuildResult:
    world_id: str
    pose_id: str
    avatar_id: str
    created: bool
    root: Path
    manifest: dict
    seconds: float
    verify: dict | None = None

    def to_dict(self) -> dict:
        return {"world_id": self.world_id, "pose_id": self.pose_id, "avatar_id": self.avatar_id, "created": self.created,
                "root": str(self.root), "seconds": self.seconds, "verify": self.verify,
                "members": {k: {"sha256": v["sha256"][:16], "scope": v["scope"], "files": v["n_files"], "bytes": v["bytes"]}
                            for k, v in self.manifest["members"].items()}}


def plan(dance: str = DEFAULT_DANCE):
    """Hash every member from its source: (spec, members, member hashes, files {dst: [sha, bytes, member]}, ids).
    Nothing is written."""
    from .timeline import DanceSpec
    spec = DanceSpec.load(dance)
    members = member_specs(spec)
    hm, files = hash_members(members)
    cache().save()
    return spec, members, hm, files, ids(hm)


def build(dance: str = DEFAULT_DANCE, *, link: bool = False, make_current: bool = True, allow_new_pose: bool = False,
          verify_level: str = "full") -> BuildResult:
    """Build (or find) the world: hash the sources, materialise work/worlds/<dance>/<id>/ if it is not there, verify
    it and write the tracked manifest (members + the sha256 of every file) (+ CURRENT.json). Idempotent: unchanged sources -> same id, no work.
    A changed room model (new pose_id) while the current world is sealed needs allow_new_pose."""
    t0 = time.time()
    spec, members, hm, files, wid = plan(dance)
    world_id = wid["world_id"]
    cur = current(dance)
    if make_current and cur and cur.get("pose_id") != wid["pose_id"] and cur.get("pose_id") in seals(dance) \
            and not allow_new_pose:
        raise WorldError(f"the current world {cur['world_id']} is sealed (pose_id {cur['pose_id']}) and these sources would "
                         f"give pose_id {wid['pose_id']}: refusing (a solved room model is not re-solved); "
                         f"pass --new-pose-world to publish it as a new world anyway")
    manifest = {"schema": SCHEMA, "dance": dance, **wid, "created": now_iso(), "reference_take": spec.reference,
                "grid": spec.grid.to_dict(),
                "members": hm, "n_files": len(files), "bytes": sum(v[1] for v in files.values()),
                "sealed_pose": wid["pose_id"] in seals(dance), "builder": "dancecap.farm.world.build",
                "files": [{"path": k, "sha256": v[0], "bytes": v[1], "member": v[2]} for k, v in sorted(files.items())]}
    td, dd = tracked_dir(dance, world_id), data_dir(dance, world_id)
    existing = read_json(td / "manifest.json", None)
    created = False
    if existing:
        manifest["created"] = existing.get("created", manifest["created"])
    if not (existing and dd.is_dir() and World(dance, manifest, dd).verify("quick").ok):
        _materialise(members, dd, link)
        created = True
    (dd / "files.json").unlink(missing_ok=True)   # legacy: the file list lives in the manifest now
    atomic_write_json(dd / "manifest.json", manifest)
    rep = World(dance, manifest, dd).verify(verify_level)
    if not rep.ok:
        raise WorldError(f"world {world_id} failed verification after the build: {rep.errors[:5]}")
    td.mkdir(parents=True, exist_ok=True)
    (td / "files.json").unlink(missing_ok=True)
    atomic_write_json(td / "manifest.json", manifest)
    if make_current:
        atomic_write_json(worlds_tracked(dance) / "CURRENT.json",
                          {"world_id": world_id, "world_sha256": wid["world_sha256"], "pose_id": wid["pose_id"],
                           "avatar_id": wid["avatar_id"], "updated": now_iso()})
    return BuildResult(world_id, wid["pose_id"], wid["avatar_id"], created, dd, manifest, round(time.time() - t0, 1),
                       rep.to_dict())


def _materialise(members: list[Member], dest: Path, link: bool) -> None:
    tmp = dest.with_name(f".{dest.name}.partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    for m in members:
        for e in m.entries:
            out = tmp / e.dst
            out.parent.mkdir(parents=True, exist_ok=True)
            if e.data is not None:
                out.write_bytes(e.data)
            elif link:
                try:
                    os.link(e.src, out)
                except OSError:
                    shutil.copy2(e.src, out)
            else:
                shutil.copy2(e.src, out)
    if dest.exists():
        shutil.rmtree(dest)
    os.replace(tmp, dest)


def check_current(dance: str = DEFAULT_DANCE) -> dict:
    """Do CURRENT's members still equal what the sources say today? (the avatar configuration changes whenever a face / groom
    is re-baked: a NEW world id with the same pose id; a changed pose member means the room / clocks / betas moved)."""
    cur = load(dance)
    spec, members, hm, files, wid = plan(dance)
    changed = sorted(k for k in hm if cur.members.get(k, {}).get("sha256") != hm[k]["sha256"])
    pose_changed = [k for k in changed if hm[k]["scope"] == POSE_SCOPE]
    return {"current": cur.id, "sources_would_give": wid["world_id"], "pose_same": wid["pose_id"] == cur.pose_id,
            "avatar_same": wid["avatar_id"] == cur.avatar_id, "changed_members": changed, "changed_pose_members": pose_changed,
            "sealed": cur.sealed,
            "advice": "up to date" if not changed else
            ("only the avatar configuration changed: `world publish` makes a new world id with the same pose id"
             if not pose_changed else "a POSE member changed: poses made under the current pose id are not from these sources")}


def seal(dance: str = DEFAULT_DANCE, world_id: str | None = None, note: str = "") -> dict:
    """Freeze the world's pose_id: its room model is solved. Later builds that would change the pose_id are refused
    unless --new-pose-world. Idempotent."""
    w = load(dance, world_id)
    rep = w.verify("full")
    if not rep.ok:
        raise WorldError(f"refusing to seal a world that does not verify: {rep.errors[:3]}")
    s = seals(dance)
    s.setdefault(w.pose_id, {"sealed": now_iso(), "world_id": w.id, "note": note or "room model solved",
                             "by": "dancecap.farm world seal"})
    atomic_write_json(worlds_tracked(dance) / "seals.json", s)
    return s[w.pose_id]


# ------------------------------------------------------------------------------------------- takes vs the world


def _sha1_or_none(p: Path) -> str | None:
    return hashlib.sha1(p.read_bytes()).hexdigest() if p.exists() else None


def _rot_angle_deg(Ra, Rb) -> float:
    import numpy as np
    R = np.asarray(Ra, float) @ np.asarray(Rb, float).T
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def _model_equal(a: Path, b: Path) -> dict:
    """Do two COLMAP models describe the same room? Same cameras / points3D bytes, and the same registered poses image by
    image (names may differ: front_full renames the map's window images)."""
    out = {"cameras_bin": None, "points3D_bin": None}
    for f in ("cameras.bin", "points3D.bin"):
        pa, pb = a / f, b / f
        out[f.replace(".", "_")] = (pa.exists() and pb.exists() and sha256_file(pa) == sha256_file(pb))
    try:
        import numpy as np
        import pycolmap
    except ImportError:
        out["images"] = "pycolmap unavailable"
        return out
    ra, rb = pycolmap.Reconstruction(str(a)), pycolmap.Reconstruction(str(b))
    out["n_images"] = [len(ra.images), len(rb.images)]
    out["n_points3D"] = [len(ra.points3D), len(rb.points3D)]
    worst = 0.0
    same_ids = set(ra.images) == set(rb.images)
    out["same_image_ids"] = same_ids
    if same_ids:
        for iid, ia in ra.images.items():
            ib = rb.images[iid]
            pa_, pb_ = ia.cam_from_world(), ib.cam_from_world()
            qa = np.asarray(pa_.rotation.quat), np.asarray(pb_.rotation.quat)
            worst = max(worst, float(np.abs(np.asarray(pa_.translation) - np.asarray(pb_.translation)).max()),
                        float(min(np.abs(qa[0] - qa[1]).max(), np.abs(qa[0] + qa[1]).max())))
        out["max_pose_diff"] = worst
    pa_xyz = np.array(sorted(tuple(p.xyz) for p in ra.points3D.values()))
    pb_xyz = np.array(sorted(tuple(p.xyz) for p in rb.points3D.values()))
    out["points_identical"] = bool(pa_xyz.shape == pb_xyz.shape and (pa_xyz.size == 0 or float(np.abs(pa_xyz - pb_xyz).max()) == 0.0))
    out["same_room"] = bool(out["points_identical"] and same_ids and worst < 1e-9)
    return out


def compare_take(take: str, dance: str = DEFAULT_DANCE, world_id: str | None = None, deep: bool = True) -> dict:
    """Does a take really share the world? Hash / numeric checks of its room model, world frame, scale, clocks, camera
    types, betas and shape edits against the world package. Returns {take, world_id, ok, checks: [{name, ok, detail}]}.
    An inconsistency is reported, never fixed."""
    import numpy as np
    import tomllib
    w = load(dance, world_id)
    tw = PACKAGE_ROOT / "work" / take
    cfg = tomllib.loads((PACKAGE_ROOT / "takes" / f"{take}.toml").read_text(encoding="utf-8"))
    checks: list[dict] = []

    def chk(name, ok, detail="", scope=POSE_SCOPE):
        checks.append({"name": name, "ok": bool(ok) if ok is not None else None, "detail": detail, "scope": scope})

    # world frame
    mine, theirs = tw / "cameras" / "world_from_sfm.json", w.ref("cameras/world_from_sfm.json")
    if not mine.exists():
        chk("world_frame", None, f"{take} has no cameras/world_from_sfm.json yet")
    elif sha256_file(mine) == sha256_file(theirs):
        chk("world_frame", True, "world_from_sfm.json is byte-identical to the world's")
    else:
        a, b = json.loads(mine.read_text()), json.loads(theirs.read_text())
        chk("world_frame", False, f"differs: gravity/rotation {_rot_angle_deg(a['R'], b['R']):.2f} deg, scale "
                                  f"{a['s']:.5f} vs {b['s']:.5f}, origin shift {float(np.linalg.norm(np.array(a['t']) - np.array(b['t']))):.3f} m")
    # room model
    mm = tw / "cameras" / "sfm"
    if not mm.exists():
        chk("room_model", None, f"{take} has no cameras/sfm yet")
    elif deep:
        eq = _model_equal(mm, w.ref("cameras/sfm"))
        chk("room_model", eq.get("same_room", eq.get("points3D_bin") and eq.get("cameras_bin")), json.dumps(eq))
    for sub in ("sub_05", "sub_06"):
        a_, b_ = tw / "cameras" / "sfm_work" / sub / "best", w.ref(f"cameras/sfm_work/{sub}/best")
        if a_.exists() and b_.exists() and deep:
            eq = _model_equal(a_, b_)
            chk(f"room_{sub}", eq.get("same_room", eq.get("points3D_bin")), json.dumps(eq))
    # scale reference / clocks
    sref = sorted(w.ref("scale").glob("reference_*.json"))
    mine_s = sorted((tw / "scale").glob("reference_*world_corrected.json"))
    if sref and mine_s:
        chk("scale_reference", sha256_file(sref[0]) == sha256_file(mine_s[0]), "reference_*world_corrected.json hash")
    alg = (PACKAGE_ROOT / "takes" / cfg["alignment"]).resolve()
    chk("camera_clocks", alg.exists() and sha256_file(alg) == sha256_file(w.path("shared/alignment_studio.json")),
        f"{cfg['alignment']} vs the world's alignment_studio.json")
    # camera types
    wt = w.load_json("shared/camera_types.json")["cameras"]
    mt = {c: {k: v[k] for k in ("mount", "zoom", "source") if k in v} for c, v in (cfg.get("cameras") or {}).items()
          if isinstance(v, dict) and "mount" in v}
    diff = {c: (mt[c], wt.get(c)) for c in mt if wt.get(c) != mt[c]}
    if not mt:
        chk("camera_types", None, "no [cameras] blocks: this take was solved with the general handheld + zoom model")
    else:
        chk("camera_types", not diff, f"{len(mt)} camera(s) typed" + (f"; differ: {diff}" if diff else ""))
    # betas: the take's pin file, and any fit it already has
    wb = w.load_json("shared/betas.json")
    pinf = tw.parent / ((cfg.get("segment") or {}).get("parent") or take) / "mamma_pin_betas.json"
    if pinf.exists():
        pb = json.loads(pinf.read_text())
        err = max((float(np.abs(np.asarray(pb[k], float) - np.asarray(wb[k], float)).max()) for k in wb if k in pb), default=None)
        chk("betas_pin_file", err is not None and err < 1e-9, f"{pinf.relative_to(PACKAGE_ROOT).as_posix()} max |diff| {err}")
    src = json.loads(w.path("shared/betas.source.json").read_text()) if w.path("shared/betas.source.json").exists() else {}
    sel = read_json(tw / "mamma" / "select.json", {}) or {}
    for key, m in src.items():   # the promoted + refined motion and the selected fit (not the historical chunk fits)
        cands = [tw / "smplx" / f"{m['role']}.npz", tw / "smplx_rf2_noslip" / f"{m['role']}.npz"]
        if sel.get("chosen"):
            cands.append(tw / "mamma" / "fits" / sel["chosen"] / "smplx" / f"{m['role']}.npz")
        else:
            cands += sorted((tw / "mamma" / "fits").glob(f"*/smplx/{m['role']}.npz"))
        for cand in cands:
            if cand.exists():
                z = np.load(cand, allow_pickle=False)
                err = float(np.abs(np.asarray(z["betas"], float).reshape(-1) - np.asarray(wb[key], float)).max())
                chk(f"betas_{m['role']}:{cand.relative_to(tw).as_posix()}", err < 1e-5, f"max |diff| {err:.2e}")
    # avatar configuration
    av = w.load_json("shared/avatar.json")["dancers"]
    for role, d in av.items():
        mine_d = (cfg.get("dancers") or {}).get(role) or {}
        for k, v in d.items():
            if mine_d.get(k) != v:
                chk(f"avatar_{role}_{k}", False, f"{mine_d.get(k)!r} != the world's {v!r}", AVATAR_SCOPE)
    if not any(c["name"].startswith("avatar_") for c in checks):
        chk("avatar_config", True, "statures, soles, shoe geometry and shape edits equal the world's", AVATAR_SCOPE)
    # pin in the TOML
    pin = cfg.get("world") or {}
    member = pin.get("member", True)
    if not pin:
        chk("toml_world_pin", None, "no [world] table in the take TOML")
    else:
        ok = pin.get("pose_id") in (None, w.pose_id) and pin.get("dance") in (None, dance)
        chk("toml_world_pin", ok, f"[world] id {pin.get('id')} pose_id {pin.get('pose_id')} vs current {w.id} / {w.pose_id}")
    cache().save()
    return {"take": take, "dance": dance, "world_id": w.id, "pose_id": w.pose_id, "member": bool(member),
            "ok": all(c["ok"] is not False for c in checks if c["scope"] == POSE_SCOPE),
            "avatar_ok": all(c["ok"] is not False for c in checks if c["scope"] == AVATAR_SCOPE), "checks": checks}


# ------------------------------------------------------------------------------------------------- TOML pins


def pin_tomls(dance: str = DEFAULT_DANCE, takes: list[str] | None = None, world_id: str | None = None) -> list[dict]:
    """Write/refresh the [world] table of take TOMLs (targeted, comment-preserving, backed up, atomic): dance, id, pose_id
    and the take's role. Each file is re-read right before its edit."""
    from ..pipeline import tomledit
    w = load(dance, world_id)
    from .timeline import DanceSpec
    spec = DanceSpec.load(dance)
    roles = {spec.name: "assembled"}
    extra = {}
    if spec.reference:
        roles[spec.reference] = "reference"
        pre = spec.reference.rsplit("_", 1)[0]   # larissa_kadu_38 -> larissa_kadu: the capture the map came from
        if pre != spec.reference and (PACKAGE_ROOT / "takes" / f"{pre}.toml").exists():
            roles[pre] = "predecessor"
            extra[pre] = {"member": False, "note": "shares the room model but has its own world frame, clocks and betas "
                                                   "(farm world compare): never mix its poses into this dance"}
    for sid in spec.segments:
        roles[spec.segment_take(sid)] = "segment"
    if takes is None:
        takes = list(roles)
    out = []
    for t in takes:
        if not (PACKAGE_ROOT / "takes" / f"{t}.toml").exists():
            out.append({"take": t, "skipped": "no TOML"})
            continue
        kv = {"dance": dance, "id": w.id, "pose_id": w.pose_id, "role": roles.get(t, "other"), **extra.get(t, {})}
        tomledit.write(t, [("world", kv)], if_match=None)
        out.append({"take": t, **kv})
    return out
