"""Stitch the fits of a long take into ONE motion on the fused 30 Hz grid (numpy + scipy only, headless: pod-ready).

    python -m dancecap.stitch larissa_kadu_full                    # stitch + gates + outputs + review sheets
    python -m dancecap.stitch larissa_kadu_full --check            # same analysis, writes nothing (exit 2 = a gate failed)
    python -m dancecap.stitch larissa_kadu_full --seam b|c         # analyse one seam (implies --check)
    python -m dancecap.stitch <dance> --chunks chunks.json --out DIR [--work-root ROOT]      # any chunk list (the pod job)
        [--partial] [--no-refined] [--no-sheets] [--tag c8] [--force] [--gate-exit report] [--allow-mixed-world]
    python -m dancecap.stitch <dance> --chunks chunks.json --out DIR --check    # verification job: stitch_check.json + sheets
    python -m dancecap.stitch <take> --write-chunks chunks.json    # the take's chunk list as JSON (relative paths)

Analysis runs on Runpod (FARM.md section 0.10): this module needs only numpy + scipy (the SMPL-X joints are a numpy
forward kinematics of the model's J_regressor / shapedirs; matplotlib only for the two review sheets), so a CPU pod runs it
from the volume's environment: `dancecap.stitch_job` builds that job (farm job spec `kind analyse`, the stitch step
template, the optional chunk-staging step). Every relative path (the chunk list, --out) resolves against the WORK ROOT:
--work-root / $DANCECAP_WORK_ROOT, default dancecap/work (on a pod: the workspace's dancecap/work or /workspace/hm/...).

A CHUNK is one fit of a window of the fused grid: a folder with the smplx contract ({lead,follow}.npz, CONTRACTS.md
`smplx/`) and an optional refined folder (pose_refine, same grid). The chunk list comes from the take TOML
([segments.*] fit windows, [reuse] = an already analysed take, [stitch]) or from a JSON file (below), so the 4 segment
fits of larissa_kadu_full, later 30-second gap-fill chunks and future dances go through the same code. The fused grid is
t_k = anchor + k / hz (larissa_kadu_38's lattice: anchor 30.151176844227862 s, 30 Hz); a chunk's window is read from its
own `t` (k = round((t - anchor) * hz), checked to be on the lattice and contiguous; larissa_kadu_38's frame i is k = i,
so its frame i is full-grid frame i + 998 for the full dance, k from -998 to 4645).

What it does (PLAN.md 3.5), per role (lead, follow):
  betas     the reference chunk's (larissa_kadu_38, `reference`) are the output's. A chunk fitted with them pinned (max |db|
            <= 2e-5, float32 round-off) is only verified; any other chunk is re-expressed with them keeping its PELVIS
            trajectory (transl += J0(b_chunk) - J0(b_ref), as mamma_chunks.merge) and the joint displacement is gated
            (betas swap p95 <= 40 mm).
  seams     chunks are ordered by their first frame; neighbours must overlap (~4 s). Inside each overlap ONE blend window
            of `blend_s` (2 s) is placed where the two chunks disagree least: every candidate position is scored by the
            worst gate utilisation (per-joint p95 over the window / its limit, max over both roles, so the dancers switch
            chunk at the same instant and their relative geometry survives); candidates within 5 % of the best are TIES,
            broken toward low-motion instants (the beat grid gives the motion-vs-beat-phase profile of the dance; a
            tied position closest to its lowest-motion phase wins, else the lowest local motion). The window stays inside
            both chunks minus their edge margins (`margin_s`: 0.5 s for the reused larissa_kadu_38, whose first and last
            frames are never used, so a|38 blends inside 30.65-34.15 s and 38|b inside 64.38-67.88 s; 0.2 s for fits)
            and after the previous seam's window.
  blend     2 s cosine cross-fade: per-joint quaternion slerp (the axis-angle branch nearest the linear blend is kept so
            the output stays continuous), linear translation and contacts; joints re-computed by FK with the common betas.
            Before the window the earlier chunk, after it the later chunk, frame for frame (bit-identical: the check
            compares every owned frame with its source and reports max abs diff 0.0).
  gate      per role and seam over the blend window (mamma_chunks.GATE, [stitch] gate_mm): every body joint 0-17 p95 <=
            60 mm, elbows / wrists p95 <= 75 mm, fingers relative to their wrist p95 <= 80 mm, pelvis median <= 40 mm,
            betas swap p95 <= 40 mm. A failing seam names the failed gate and the centred 12 s boundary refit window.
  flags     n_cams (T,) per role = cameras that see the dancer (>= 50 of 512 landmarks visible in the chunk's ma_2d, else
            the prepare.json camera spans); quality_flag (T,) int8 = 0 absent / 1 mono / 2 two_view / 3 multi (>= 3).
            [stitch] mono / low_view / <role>_absent spans force a flag at most that good; a zero-camera run of >= 1 s is
            `absent` too (quality 0, never tracked motion: the lead before he enters, -3.13 .. about -0.6 s).

Outputs (--out, default <work root>/<take>): smplx/{lead,follow}.npz (smplx contract + segment_owner (T,) str = the owning
chunk's name ("" = none), segment_index (T,) i2 index into chunk_names, chunk_names, blend_seam (T,) i2, blend_weight (T,)
weight of the later chunk, covered (T,) bool, n_cams (T,) i2, quality_flag (T,) i1, quality_flag_names, absent (T,) bool;
quality = 0 where absent), smplx_rf2_noslip/ (the refined
chunks stitched with the SAME windows; only when every chunk has one), stitch.json, review/pose_seams_full.png,
review/stitch_overview.png. A failed gate writes to work/<take>/stitch_rejected/ instead (smplx/ untouched) unless
--force. Files are written byte-deterministically (fixed zip timestamps): running twice gives identical files.

Chunk folders of larissa_kadu_full: a segment's fit is work/<take>_<id>/mamma/fits/<tag>/smplx (tag: mamma/select.json's
choice, else its biggest prepared set: c8 for a and b, c6 for c and d) and its refined motion work/<take>_<id>/
smplx_rf2_noslip (python -m dancecap.pose_refine <segment take> --src smplx --dst smplx_rf2_noslip: the 38 s recipe, 5 Hz
root + 8 Hz rotation low-pass + image-aware root height and legs with the floor term); the reused chunk is
work/larissa_kadu_38/smplx and smplx_rf2_noslip.

JSON chunk list (--chunks): {"take": "name", "out_raw": "smplx", "out_refined": "smplx_rf2_noslip", "anchor": 30.15..,
"hz": 30, "blend_s": 2.0, "edge_margin_s": 0.2, "gate_mm": {...}, "beats": "path", "mono": [[t0, t1]],
"low_view": [[t0, t1]], "lead_absent": [[t0, t1]], "chunks": [{"name": "a", "src": "dir", "refined": "dir",
"k": [k0, k1], "margin_s": 0.0, "reference": false, "prepare": "prepare.json", "ma_2d": "dir", "world_id": "...",
"pose_id": "...", "header": "out/chunk.json"}, ...], "world_id": "default for chunks without one"}; relative paths resolve
against the work root. Chunks fitted in different pose worlds are refused (exit 3) unless --allow-mixed-world.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tomllib
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from .take import PACKAGE_ROOT

SCHEMA = "dancecap.stitch/1"
ROLES = ("lead", "follow")
ANCHOR = 30.151176844227862
HZ = 30.0
POSE_KEYS = ("global_orient", "body_pose", "jaw_pose", "leye_pose", "reye_pose", "left_hand_pose", "right_hand_pose")
REQUIRED = ("t", *POSE_KEYS, "transl", "betas", "joints")
STATIC = ("betas", "parts")                    # arrays that are not per-frame even when their length could match
LINEAR_KEYS = ("transl", "contact", "floor_contact")
WORST_HI = ("reproj_px", "reproj_px_raw", "reproj_parts_px")   # conservative inside a blend window
WORST_LO = ("quality", "quality_parts")
DOMINANT_KEYS = ("stance",)                    # per-foot flags: the chunk with the larger weight
ZEROED_WHEN_ABSENT = ("quality", "quality_parts")   # set to 0 on `absent` frames (compared on the other frames)
BETAS_TOL = 2e-5                               # |betas - reference| below this = pinned (float32 round-off of the pin)
FLAG_NAMES = ("absent", "mono", "two_view", "multi")
GATE_DEFAULT = {"body_p95": 60.0, "elbow_wrist_p95": 75.0, "fingers_rel_wrist_p95": 80.0, "pelvis_median": 40.0,
                "betas_swap_p95": 40.0}
ARM = (18, 19, 20, 21)                         # elbows + wrists (own limit)
BODY_IDX = [j for j in range(22) if j not in ARM]   # joints 0-17
FINGERS = ((20, tuple(range(25, 40))), (21, tuple(range(40, 55))))   # (wrist, its 15 finger joints)
JOINT_NAMES = ("pelvis", "l_hip", "r_hip", "spine1", "l_knee", "r_knee", "spine2", "l_ankle", "r_ankle", "spine3",
               "l_foot", "r_foot", "neck", "l_collar", "r_collar", "head", "l_shoulder", "r_shoulder", "l_elbow",
               "r_elbow", "l_wrist", "r_wrist")
ROLE_COLOR = {"lead": "#e34948", "follow": "#2a78d6"}
CHUNK_COLORS = ("#2a78d6", "#1f9d74", "#c98a1b", "#8a5cd0", "#d0683b", "#3a9ab5", "#8a8984")


# ------------------------------------------------------------------------------------------------ roots


def work_root() -> Path:
    """The folder relative chunk / output paths resolve against: $DANCECAP_WORK_ROOT (--work-root), else dancecap/work.
    On a pod: the workspace's dancecap/work (the worker's scratch) or any folder of the volume (/workspace/hm/...)."""
    o = os.environ.get("DANCECAP_WORK_ROOT")
    return Path(o).resolve() if o else PACKAGE_ROOT / "work"


def smplx_model_file() -> Path:
    """MAMMA's locked-head SMPL-X neutral model (the one every fit used): $DANCECAP_SMPLX_MODEL (file or folder), else
    <mamma data>/body_models/smplx_locked_head/smplx/SMPLX_NEUTRAL.npz ($DANCECAP_MAMMA_DATA, as env.mamma_data)."""
    o = os.environ.get("DANCECAP_SMPLX_MODEL")
    base = Path(o) if o else Path(os.environ.get("DANCECAP_MAMMA_DATA") or PACKAGE_ROOT.parent / "mamma" / "data")         / "body_models" / "smplx_locked_head"
    if base.is_file():
        return base
    for c in (base / "smplx" / "SMPLX_NEUTRAL.npz", base / "SMPLX_NEUTRAL.npz"):
        if c.is_file():
            return c
    raise FileNotFoundError(f"SMPL-X neutral model not found under {base} (set DANCECAP_SMPLX_MODEL)")


# ------------------------------------------------------------------------------------------------ spec


@dataclass
class Chunk:
    name: str
    src: Path                       # folder with {lead,follow}.npz (smplx contract)
    refined: Path | None = None     # folder with the refined {lead,follow}.npz on the same window
    k0: int | None = None           # first / last lattice index (inclusive); None = read from the files' t
    k1: int | None = None
    margin_s: float = 0.0           # first / last frames that are never used (edge frames of a reused result)
    reference: bool = False         # betas of the output + bit-identical reuse
    kind: str = "segment"           # segment | reuse | refit (informational)
    prepare: Path | None = None     # mamma/prepare.json (camera spans, grid)
    ma_2d: Path | None = None       # mamma/ma_2d/<seq>/ (visibilities per camera -> n_cams)
    k_hint: tuple | None = None     # window announced by the take (prepare.json / TOML): used while the fit is missing;
    #                                 the fit's own `t` is authoritative (k0 / k1 set explicitly must match it)
    world_id: str | None = None     # the world package the chunk was fitted in (FARM.md section 3)
    pose_id: str | None = None      # its pose id (room + frame + scale + clocks + betas + skeleton)

    @property
    def k_lo(self) -> int:
        return self.k0 if self.k0 is not None else (self.k_hint[0] if self.k_hint else 10 ** 9)

    @property
    def k_hi(self) -> int:
        return self.k1 if self.k1 is not None else (self.k_hint[1] if self.k_hint else 10 ** 9)


@dataclass
class Spec:
    take: str
    chunks: list
    work: Path = None
    anchor: float = ANCHOR
    hz: float = HZ
    blend_s: float = 2.0
    edge_margin_s: float = 0.2      # default margin of a chunk without its own
    gate: dict = field(default_factory=dict)
    absent: dict = field(default_factory=dict)      # role -> [[t0, t1], ...] reference seconds
    mono: list = field(default_factory=list)
    low_view: list = field(default_factory=list)
    out_raw: str = "smplx"
    out_refined: str | None = "smplx_rf2_noslip"
    beats: Path | None = None
    tie_rel: float = 0.05
    tie_abs: float = 0.02           # in gate utilisation (1.0 = at the limit)
    select_weights: dict = field(default_factory=dict)   # T1: {"arm": w, "fingers": w}: the window CHOICE counts those gates w times (w < 1: body / pelvis first); the gates themselves are unchanged
    min_points: int = 50            # visible landmarks (of 512) for a camera to count in n_cams
    absent_min_s: float = 1.0
    refit_s: float = 12.0
    min_blend_frames: int = 8
    source_json: Path | None = None
    work_set: bool = False          # the output folder was named explicitly (JSON "work" / --out)
    world_id: str | None = None     # default world of the chunks that carry none
    pose_id: str | None = None
    allow_mixed_world: bool = False
    require_world: bool = False

    def thresholds(self) -> dict:
        return {**GATE_DEFAULT, **self.gate}


def _wp(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else work_root() / p


def _fit_tag(w: Path) -> str:
    """The fit tag of a segment take: mamma/select.json's choice, else its biggest prepared set, else c8."""
    try:
        return json.loads((w / "mamma" / "select.json").read_text())["chosen"]
    except (OSError, KeyError, ValueError):
        pass
    try:
        sets = json.loads((w / "mamma" / "prepare.json").read_text())["sets"]
        return max(sets, key=lambda k: len(sets[k]["cams"]))
    except (OSError, KeyError, ValueError):
        return "c8"


def _k_from_prepare(w: Path, anchor: float, hz: float):
    try:
        t = np.asarray(json.loads((w / "mamma" / "prepare.json").read_text())["grid"]["t"], float)
    except (OSError, KeyError, ValueError):
        return None
    k = np.rint((t - anchor) * hz).astype(int)
    return int(k[0]), int(k[-1])


def _ma2d_dir(w: Path, tag: str) -> Path | None:
    try:
        seq = json.loads((w / "mamma" / "prepare.json").read_text())["sets"][tag]["seq"]
    except (OSError, KeyError, ValueError):
        return None
    return w / "mamma" / "ma_2d" / seq


def spec_from_take(take: str, tag: str | None = None) -> Spec:
    """Chunk list of a take from takes/<take>.toml: [segments.*] (fit_k windows; work/<take>_<id>/mamma/fits/<tag>/smplx),
    [reuse] (an analysed take: motion_span, motion_raw, motion_refined), [stitch] (blend, gates, spans)."""
    cfg = tomllib.loads((PACKAGE_ROOT / "takes" / f"{take}.toml").read_text(encoding="utf-8"))
    reuse, segs, st = cfg.get("reuse", {}), cfg.get("segments", {}), cfg.get("stitch", {})
    out_refined = (cfg.get("export", {}) or {}).get("smplx_dir") or reuse.get("motion_refined") or "smplx_rf2_noslip"
    out_refined = None if out_refined == "smplx" else out_refined
    anchor, hz = float(st.get("anchor", ANCHOR)), float(st.get("hz", HZ))
    edge = float(st.get("edge_margin_s", 0.2))
    chunks = []
    for sid, sc in segs.items():
        name = f"{take}_{sid}"
        w = work_root() / name
        tg = tag or _fit_tag(w)
        src = w / "mamma" / "fits" / tg / "smplx"
        if not (src / "lead.npz").exists() and (w / "smplx" / "lead.npz").exists():
            src = w / "smplx"
        k = _k_from_prepare(w, anchor, hz) or tuple(int(x) for x in sc["fit_k"])
        chunks.append(Chunk(sid, src, w / out_refined if out_refined else None, None, None, edge, False, "segment",
                            w / "mamma" / "prepare.json", _ma2d_dir(w, tg), tuple(k)))
    if reuse:
        rt = reuse["take"]
        w = work_root() / rt
        span = reuse.get("motion_span")
        k = (int(round((span[0] - anchor) * hz)), int(round((span[1] - anchor) * hz))) if span else \
            _k_from_prepare(w, anchor, hz)
        short = rt.rsplit("_", 1)[-1] if "_" in rt else rt
        chunks.append(Chunk(short, w / reuse.get("motion_raw", "smplx"),
                            w / reuse["motion_refined"] if reuse.get("motion_refined") else None, None, None,
                            float(st.get("reuse_margin_s", 0.5)), True, "reuse", w / "mamma" / "prepare.json",
                            _ma2d_dir(w, _fit_tag(w)), tuple(k)))
    beats = next((p for p in (work_root() / take / "beats.json", work_root() / reuse.get("take", "") / "beats.json")
                  if p.is_file()), None)
    return Spec(take=take, chunks=sorted(chunks, key=lambda c: (c.k_lo, c.k_hi)), work=work_root() / take,
                anchor=anchor, hz=hz, blend_s=float(st.get("blend_s", 2.0)), edge_margin_s=edge,
                gate=dict(st.get("gate_mm", {})),
                absent={r: st[f"{r}_absent"] for r in ROLES if f"{r}_absent" in st},
                mono=list(st.get("mono", [])), low_view=list(st.get("low_view", [])),
                out_raw=str((cfg.get("stitch", {}) or {}).get("out_raw", "smplx")), out_refined=out_refined,
                beats=beats, tie_rel=float(st.get("tie_rel", 0.05)), tie_abs=float(st.get("tie_abs", 0.02)),
                min_points=int(st.get("n_cams_min_points", 50)), absent_min_s=float(st.get("absent_min_s", 1.0)),
                refit_s=float(st.get("refit_s", 12.0)))


def _margin(v):
    """A chunk's edge margin from the JSON list: seconds (both ends) or a pair [first, last] (T1, 2026-10-09: the ends that face a refit window get a bigger one)."""
    return [float(v[0]), float(v[1])] if isinstance(v, (list, tuple)) else float(v)


def spec_from_json(path: str | Path) -> Spec:
    path = Path(path)
    d = json.loads(path.read_text(encoding="utf-8"))
    chunks = []
    for c in d["chunks"]:
        k = c.get("k")
        wid, pid = c.get("world_id"), c.get("pose_id")
        if c.get("header") and _wp(c["header"]).is_file():     # an out/chunk.json-style header naming the world
            try:
                h = json.loads(_wp(c["header"]).read_text(encoding="utf-8"))
                hw = h.get("world") if isinstance(h.get("world"), dict) else {}
                wid = wid or h.get("world_id") or hw.get("world_id") or hw.get("id")
                pid = pid or h.get("pose_id") or hw.get("pose_id")
            except (OSError, ValueError):
                pass
        chunks.append(Chunk(c["name"], _wp(c["src"]), _wp(c["refined"]) if c.get("refined") else None,
                            int(k[0]) if k else None, int(k[1]) if k else None,
                            _margin(c.get("margin_s", d.get("edge_margin_s", 0.2))), bool(c.get("reference", False)),
                            c.get("kind", "segment"), _wp(c["prepare"]) if c.get("prepare") else None,
                            _wp(c["ma_2d"]) if c.get("ma_2d") else None, world_id=wid, pose_id=pid))
    take = d["take"]
    return Spec(take=take, chunks=sorted(chunks, key=lambda c: (c.k_lo, c.k_hi)),
                work=_wp(d["work"]) if d.get("work") else work_root() / take,
                anchor=float(d.get("anchor", ANCHOR)), hz=float(d.get("hz", HZ)), blend_s=float(d.get("blend_s", 2.0)),
                edge_margin_s=float(d.get("edge_margin_s", 0.2)), gate=dict(d.get("gate_mm", {})),
                work_set="work" in d, world_id=d.get("world_id"), pose_id=d.get("pose_id"),
                absent={r: d[f"{r}_absent"] for r in ROLES if f"{r}_absent" in d}, mono=list(d.get("mono", [])),
                low_view=list(d.get("low_view", [])), out_raw=d.get("out_raw", "smplx"),
                out_refined=d.get("out_refined", "smplx_rf2_noslip"),
                beats=_wp(d["beats"]) if d.get("beats") else None, tie_rel=float(d.get("tie_rel", 0.05)),
                tie_abs=float(d.get("tie_abs", 0.02)), select_weights=dict(d.get("select_weights", {})), min_points=int(d.get("n_cams_min_points", 50)),
                absent_min_s=float(d.get("absent_min_s", 1.0)), refit_s=float(d.get("refit_s", 12.0)),
                source_json=path)


def load_spec(arg: str, tag: str | None = None) -> Spec:
    return spec_from_json(arg) if arg.endswith(".json") and Path(arg).is_file() else spec_from_take(arg, tag)


def _rel(p) -> str | None:
    """A path as the chunk list stores it: relative to the work root when it lies under it, else absolute (posix)."""
    if p is None:
        return None
    p = Path(p)
    try:
        return p.resolve().relative_to(work_root().resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def chunk_list(spec: Spec) -> dict:
    """The spec as the --chunks JSON (the pod job's input): paths relative to the work root where they lie under it."""
    return {
        "take": spec.take, "anchor": spec.anchor, "hz": spec.hz, "blend_s": spec.blend_s, "edge_margin_s": spec.edge_margin_s,
        "gate_mm": spec.gate, "out_raw": spec.out_raw, "out_refined": spec.out_refined, "beats": _rel(spec.beats),
        "mono": spec.mono, "low_view": spec.low_view, **{f"{r}_absent": v for r, v in spec.absent.items()},
        "tie_rel": spec.tie_rel, "tie_abs": spec.tie_abs, "select_weights": spec.select_weights, "n_cams_min_points": spec.min_points,
        "absent_min_s": spec.absent_min_s, "refit_s": spec.refit_s,
        **({"world_id": spec.world_id} if spec.world_id else {}), **({"pose_id": spec.pose_id} if spec.pose_id else {}),
        "chunks": [{"name": c.name, "src": _rel(c.src), "refined": _rel(c.refined),
                    "k": [c.k_lo, c.k_hi] if c.k_lo < 10 ** 9 else None, "margin_s": c.margin_s, "reference": c.reference,
                    "kind": c.kind, "prepare": _rel(c.prepare), "ma_2d": _rel(c.ma_2d),
                    **({"world_id": c.world_id} if c.world_id else {}), **({"pose_id": c.pose_id} if c.pose_id else {})}
                   for c in spec.chunks]}


class WorldMismatch(ValueError):
    pass


def check_worlds(L: list, spec: Spec) -> dict:
    """The world guard (FARM.md section 3): chunks fitted in different POSE worlds (room / frame / scale / clocks / betas /
    skeleton) are refused unless allow_mixed_world; different world ids with one pose id (avatar-only) only warn; a chunk
    without a world id is reported (refused with require_world). Ids come from the chunk entry, its header, the list."""
    items = [(c.name, c.ch.world_id or spec.world_id, c.ch.pose_id or spec.pose_id) for c in L]
    known = [(n, w, p or w) for n, w, p in items if w or p]
    unknown = [n for n, w, p in items if not (w or p)]
    wids = sorted({w for _, w, _ in known if w})
    pids = sorted({p for _, _, p in known if p})
    rep = {"world_ids": wids, "pose_ids": pids, "unknown": unknown, "warnings": [], "mixed_allowed": bool(spec.allow_mixed_world)}
    if unknown and spec.require_world:
        raise WorldMismatch(f"chunk(s) {unknown} carry no world_id (--require-world)")
    if len(pids) > 1:
        by = {n: p for n, _, p in known}
        if not spec.allow_mixed_world:
            raise WorldMismatch(f"chunks come from different worlds (pose ids {pids}: {by}); refusing without --allow-mixed-world")
        rep["warnings"].append(f"MIXED worlds allowed by flag: pose ids {pids}")
    elif len(wids) > 1:
        rep["warnings"].append(f"world ids {wids} differ only in the avatar configuration (same pose id {pids[0] if pids else '?'})")
    if unknown and known:
        rep["warnings"].append(f"chunk(s) {unknown} carry no world id (not checked)")
    return rep


# ------------------------------------------------------------------------------------------------ loading


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_dir(d: Path) -> dict:
    out = {}
    for r in ROLES:
        with np.load(Path(d) / f"{r}.npz", allow_pickle=True) as z:
            out[r] = {k: z[k] for k in z.files}
    return out


def lattice_range(t, anchor: float, hz: float, label: str) -> tuple[int, int]:
    """First / last lattice index of a time axis; it must lie on anchor + k / hz (1 microsecond) and be contiguous."""
    t = np.asarray(t, float)
    k = np.rint((t - anchor) * hz).astype(np.int64)
    err = float(np.abs(t - (anchor + k / hz)).max())
    if err > 1e-6:
        raise ValueError(f"{label}: t is {err * 1000:.4f} ms off the fused lattice (anchor {anchor}, {hz} Hz)")
    if len(k) > 1 and not np.all(np.diff(k) == 1):
        raise ValueError(f"{label}: t is not contiguous on the lattice")
    return int(k[0]), int(k[-1])


@dataclass
class Loaded:
    ch: Chunk
    k0: int
    k1: int
    m0: int
    m1: int
    raw: dict
    refined: dict | None = None
    sha: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    @property
    def n(self) -> int:
        return self.k1 - self.k0 + 1

    @property
    def name(self) -> str:
        return self.ch.name


def load_chunk(ch: Chunk, spec: Spec, want_refined: bool) -> Loaded:
    raw = read_dir(ch.src)
    for r in ROLES:
        miss = [k for k in REQUIRED if k not in raw[r]]
        if miss:
            raise ValueError(f"{ch.name} {r}: {ch.src} lacks {miss}")
    k0, k1 = lattice_range(raw["lead"]["t"], spec.anchor, spec.hz, f"{ch.name} lead")
    f0, f1 = lattice_range(raw["follow"]["t"], spec.anchor, spec.hz, f"{ch.name} follow")
    if (k0, k1) != (f0, f1):
        raise ValueError(f"{ch.name}: lead covers k {k0}..{k1}, follow {f0}..{f1}")
    if (ch.k0 is not None and ch.k0 != k0) or (ch.k1 is not None and ch.k1 != k1):
        raise ValueError(f"{ch.name}: the spec says k {ch.k0}..{ch.k1}, {ch.src} holds k {k0}..{k1}")
    warn = []
    if ch.k_hint is not None and tuple(ch.k_hint) != (k0, k1):
        warn.append(f"{ch.name}: the take announced k {ch.k_hint[0]}..{ch.k_hint[1]}, the fit holds k {k0}..{k1} (the fit wins)")
    n = k1 - k0 + 1
    for r in ROLES:
        for key in ("transl", "joints", *POSE_KEYS):
            if len(raw[r][key]) != n:
                raise ValueError(f"{ch.name} {r}: {key} has {len(raw[r][key])} frames, t has {n}")
    ref = None
    if want_refined and ch.refined is not None and (Path(ch.refined) / "lead.npz").exists():
        ref = read_dir(ch.refined)
        for r in ROLES:
            if len(ref[r]["t"]) != n or lattice_range(ref[r]["t"], spec.anchor, spec.hz, f"{ch.name} refined {r}") \
                    != (k0, k1):
                raise ValueError(f"{ch.name}: refined {r} is not on the raw window k {k0}..{k1}")
    ms = ch.margin_s if ch.margin_s is not None else spec.edge_margin_s
    ms0, ms1 = (ms[0], ms[1]) if isinstance(ms, (list, tuple)) else (ms, ms)         # T1: a pair = (first, last) margin, a number = both
    m0, m1 = int(round(ms0 * spec.hz)), int(round(ms1 * spec.hz))
    sha = {"raw": {r: sha256_file(Path(ch.src) / f"{r}.npz") for r in ROLES},
           "refined": {r: sha256_file(Path(ch.refined) / f"{r}.npz") for r in ROLES} if ref is not None else None}
    return Loaded(ch, k0, k1, m0, m1, raw, ref, sha, warn)


# ------------------------------------------------------------------------------------------------ rotations


def _canon(v: np.ndarray) -> np.ndarray:
    return Rot.from_rotvec(v.reshape(-1, 3)).as_rotvec().reshape(v.shape)


def nearest_rep(v: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """The axis-angle vector of v's rotation (any branch: angle + 2 pi m) nearest to `ref` (same shape (..., 3))."""
    v0 = _canon(np.asarray(v, float))
    ang = np.linalg.norm(v0, axis=-1, keepdims=True)
    ax = v0 / np.maximum(ang, 1e-12)
    best, bd = v0, np.linalg.norm(v0 - ref, axis=-1)
    for m in (-2, -1, 1, 2):
        c = ax * (ang + 2 * np.pi * m)
        d = np.linalg.norm(c - ref, axis=-1)
        better = d < bd - 1e-12
        best = np.where(better[..., None], c, best)
        bd = np.where(better, d, bd)
    return best


def blend_rotvec(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Per-joint slerp of axis-angle arrays a, b (n, J, 3) from a (w=0) to b (w=1), w (n,); the shortest arc whatever
    the quaternion signs / axis-angle branches of the inputs, and the output vector on the branch nearest the linear
    blend (so a frame-to-frame comparison of the axis-angle parameters shows no jump)."""
    n, J, _ = a.shape
    qa = Rot.from_rotvec(a.reshape(-1, 3)).as_quat()
    qb = Rot.from_rotvec(b.reshape(-1, 3)).as_quat()
    dot = (qa * qb).sum(-1, keepdims=True)
    qb = np.where(dot < 0, -qb, qb)
    th = np.arccos(np.clip(np.abs(dot), -1.0, 1.0))
    s = np.sin(th)
    ww = np.repeat(np.asarray(w, float), J)[:, None]
    small = s < 1e-6
    k0 = np.where(small, 1 - ww, np.sin((1 - ww) * th) / np.where(small, 1, s))
    k1 = np.where(small, ww, np.sin(ww * th) / np.where(small, 1, s))
    q = k0 * qa + k1 * qb
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    v = Rot.from_quat(q).as_rotvec()
    a_f, b_f = a.reshape(-1, 3), b.reshape(-1, 3)
    b_al = nearest_rep(b_f, a_f)
    ref = (1 - ww) * a_f + ww * b_al
    return nearest_rep(v, ref).reshape(n, J, 3)


def cosine_weights(n: int) -> np.ndarray:
    """Weight of the LATER chunk over an n-frame window: strictly inside (0, 1), cosine ramp (zero slope at both ends)."""
    x = (np.arange(n) + 0.5) / n
    return 0.5 - 0.5 * np.cos(np.pi * x)


# ------------------------------------------------------------------------------------------------ forward kinematics


def pose165(z: dict) -> np.ndarray:
    n = len(z["transl"])
    return np.concatenate([np.asarray(z[k], float).reshape(n, -1) for k in POSE_KEYS], 1)


_FK_CACHE: dict = {}


def _fk_model() -> dict:
    """What joint FK needs of the SMPL-X model, loaded once: J_template (55, 3) = J_regressor @ v_template, J_dirs
    (55, 3, 400) = J_regressor @ shapedirs, the kinematic tree. Joints are rigid transforms of the shaped rest joints
    (pose correctives move vertices, never joints), so no mesh and no torch / smplx is needed."""
    f = smplx_model_file()
    m = _FK_CACHE.get(str(f))
    if m is None:
        with np.load(f, allow_pickle=True) as z:
            Jr = np.asarray(z["J_regressor"], np.float64)
            m = {"J_template": Jr @ np.asarray(z["v_template"], np.float64),
                 "J_dirs": np.einsum("jv,vcb->jcb", Jr, np.asarray(z["shapedirs"][:, :, :64], np.float64)),
                 "parents": np.asarray(z["kintree_table"][0], np.int64)}
        m["parents"][0] = -1
        _FK_CACHE.clear()
        _FK_CACHE[str(f)] = m
    return m


def rest_joints(betas: np.ndarray) -> np.ndarray:
    """Rest-pose joints (55, 3) of the model for these betas (no translation)."""
    m = _fk_model()
    b = np.asarray(betas, np.float64).reshape(-1)
    if len(b) > m["J_dirs"].shape[-1]:
        raise ValueError(f"{len(b)} betas, the FK table holds {m['J_dirs'].shape[-1]}")
    return m["J_template"] + m["J_dirs"][:, :, :len(b)] @ b


def _rodrigues(aa: np.ndarray) -> np.ndarray:
    return Rot.from_rotvec(np.asarray(aa, np.float64).reshape(-1, 3)).as_matrix().reshape(aa.shape[:-1] + (3, 3))


def fk_joints(pose: np.ndarray, transl: np.ndarray, betas: np.ndarray, block: int = 600) -> np.ndarray:
    """Posed joints (n, 55, 3) float64 of MAMMA's locked-head SMPL-X neutral model (flat hand mean: hand poses are
    absolute), pose in the 165-D ma_3d layout. Equals mamma_dyn._forward's joints to float32 round-off (tested)."""
    m = _fk_model()
    J = rest_joints(betas)
    par = m["parents"]
    out = []
    for i in range(0, len(pose), block):
        P = np.asarray(pose[i:i + block], np.float64)
        n = len(P)
        R = _rodrigues(P.reshape(n, -1, 3))
        G = np.empty_like(R)
        pj = np.empty((n, 55, 3))
        tr = np.asarray(transl[i:i + block], np.float64)
        for j in range(55):
            if par[j] < 0:
                G[:, j] = R[:, j]
                pj[:, j] = J[j] + tr
            else:
                G[:, j] = G[:, par[j]] @ R[:, j]
                pj[:, j] = pj[:, par[j]] + np.einsum("tab,b->ta", G[:, par[j]], J[j] - J[par[j]])
        out.append(pj)
    return np.concatenate(out) if out else np.zeros((0, 55, 3))


def reexpress(z: dict, betas_ref: np.ndarray) -> dict:
    """z with betas_ref instead of its own betas, the pelvis world trajectory kept (mamma_chunks.merge): transl +=
    J0(b) - J0(b_ref); joints re-computed. Returns {median, p95} of the joint displacement in mm."""
    b = np.asarray(z["betas"], float)
    dJ = rest_joints(b)[0] - rest_joints(betas_ref)[0]
    tr = np.asarray(z["transl"], float) + dJ
    j = fk_joints(pose165(z), tr, betas_ref)
    d = np.linalg.norm(j - np.asarray(z["joints"], float)[:, :55], axis=-1) * 1000
    z["transl"], z["joints"], z["betas"] = tr, j, np.asarray(betas_ref, float).copy()
    return {"median": round(float(np.median(d)), 3), "p95": round(float(np.percentile(d, 95)), 3)}


def unify_betas(L: list, ref: int) -> dict:
    """Make every chunk carry the reference chunk's betas (verify when pinned, else re-express). -> report."""
    rep = {}
    for r in ROLES:
        b_ref = np.asarray(L[ref].raw[r]["betas"], float)
        rr = {}
        for i, c in enumerate(L):
            b = np.asarray(c.raw[r]["betas"], float)
            if b.shape != b_ref.shape:
                raise ValueError(f"{c.name} {r}: {b.shape[0]} betas, the reference {L[ref].name} has {b_ref.shape[0]}")
            diff = float(np.abs(b - b_ref).max())
            ent = {"max_abs_diff": diff, "pinned": diff <= BETAS_TOL, "reference": i == ref}
            if i != ref:
                if diff <= BETAS_TOL:
                    for z in [c.raw[r]] + ([c.refined[r]] if c.refined else []):
                        z["betas"] = b_ref.copy()
                    ent["swap_mm"] = {"median": 0.0, "p95": 0.0}
                else:
                    ent["swap_mm"] = reexpress(c.raw[r], b_ref)
                    if c.refined:
                        reexpress(c.refined[r], b_ref)
                    ent["re_expressed"] = True
            else:
                ent["swap_mm"] = {"median": 0.0, "p95": 0.0}
            rr[c.name] = ent
        rep[r] = rr
    return rep


# ------------------------------------------------------------------------------------------------ disagreement + gate


def disagreement(jp: np.ndarray, jn: np.ndarray):
    """jp, jn (n, 55, 3) m -> d (n, 55) mm joint distance, rel (n, 30) mm finger joints relative to their wrist."""
    d = np.linalg.norm(jp - jn, axis=-1) * 1000.0
    rel = np.concatenate([np.linalg.norm((jp[:, list(f)] - jp[:, [w]]) - (jn[:, list(f)] - jn[:, [w]]), axis=-1)
                          for w, f in FINGERS], 1) * 1000.0
    return d, rel


def window_stats(d: np.ndarray, rel: np.ndarray, sl: slice) -> dict:
    """The gate's measures over the frames d[sl]: per-joint p95 of joints 0-21, worst body (0-17), worst elbow / wrist,
    worst finger joint relative to its wrist, pelvis median."""
    dw, rw = d[sl], rel[sl]
    pj = np.percentile(dw[:, :22], 95, axis=0)
    jb = BODY_IDX[int(np.argmax(pj[BODY_IDX]))]
    ja = ARM[int(np.argmax(pj[list(ARM)]))]
    return {"joint_p95_max": float(pj[jb]), "joint": JOINT_NAMES[jb], "arm_p95_max": float(pj[ja]),
            "arm_joint": JOINT_NAMES[ja], "fingers_rel_wrist_p95_max": float(np.percentile(rw, 95, axis=0).max()),
            "pelvis_median": float(np.median(dw[:, 0])), "per_joint_p95": pj, "frames": int(len(dw))}


def utilisation(s: dict, th: dict) -> float:
    """Worst gate measure as a fraction of its limit (<= 1 passes)."""
    return float(max(s["joint_p95_max"] / th["body_p95"], s["arm_p95_max"] / th["elbow_wrist_p95"],
                     s["fingers_rel_wrist_p95_max"] / th["fingers_rel_wrist_p95"],
                     s["pelvis_median"] / th["pelvis_median"]))


def gate_checks(role: str, seam: str, s: dict, swap_p95: float, th: dict) -> tuple[dict, list]:
    why = []
    checks = {"body_p95_mm": round(s["joint_p95_max"], 2), "body_joint": s["joint"],
              "elbow_wrist_p95_mm": round(s["arm_p95_max"], 2), "arm_joint": s["arm_joint"],
              "fingers_rel_wrist_p95_mm": round(s["fingers_rel_wrist_p95_max"], 2),
              "pelvis_median_mm": round(s["pelvis_median"], 2), "betas_swap_p95_mm": round(swap_p95, 2)}
    if not s["joint_p95_max"] <= th["body_p95"]:
        why.append(f"{role} {seam}: {s['joint']} p95 {s['joint_p95_max']:.0f} mm > {th['body_p95']:.0f} (body joints 0-17)")
    if not s["arm_p95_max"] <= th["elbow_wrist_p95"]:
        why.append(f"{role} {seam}: {s['arm_joint']} p95 {s['arm_p95_max']:.0f} mm > {th['elbow_wrist_p95']:.0f} "
                   "(elbows / wrists)")
    if not s["fingers_rel_wrist_p95_max"] <= th["fingers_rel_wrist_p95"]:
        why.append(f"{role} {seam}: fingers relative to the wrist p95 {s['fingers_rel_wrist_p95_max']:.0f} mm > "
                   f"{th['fingers_rel_wrist_p95']:.0f}")
    if not s["pelvis_median"] <= th["pelvis_median"]:
        why.append(f"{role} {seam}: pelvis median {s['pelvis_median']:.0f} mm > {th['pelvis_median']:.0f}")
    if not swap_p95 <= th["betas_swap_p95"]:
        why.append(f"{role} {seam}: betas swap p95 {swap_p95:.0f} mm > {th['betas_swap_p95']:.0f}")
    return checks, why


# ------------------------------------------------------------------------------------------------ motion + beats


def motion_series(J: np.ndarray, hz: float) -> np.ndarray:
    """Mean speed (m/s) of the pelvis-relative body joints per frame (frame 0 repeats frame 1)."""
    loc = J[:, 1:22] - J[:, :1]
    v = np.zeros(len(J))
    if len(J) > 1:
        v[1:] = np.linalg.norm(np.diff(loc, axis=0), axis=-1).mean(1) * hz
        v[0] = v[1]
    return v


def load_beats(path: Path | None) -> np.ndarray | None:
    if path is None or not Path(path).is_file():
        return None
    try:
        b = np.asarray(json.loads(Path(path).read_text()), float)
        return np.sort(b[:, 0]) if b.ndim == 2 and len(b) > 3 else None
    except (ValueError, IndexError):
        return None


def beat_phase(beats: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Position of t inside its beat-grid interval (0..1); NaN outside the grid."""
    t = np.atleast_1d(np.asarray(t, float))
    i = np.searchsorted(beats, t, "right") - 1
    ok = (i >= 0) & (i < len(beats) - 1)
    ph = np.full(len(t), np.nan)
    ph[ok] = (t[ok] - beats[i[ok]]) / (beats[i[ok] + 1] - beats[i[ok]])
    return ph


def low_motion_phase(beats: np.ndarray, t: np.ndarray, motion: np.ndarray, bins: int = 8) -> dict | None:
    ph = beat_phase(beats, t)
    ok = np.isfinite(ph) & np.isfinite(motion)
    if ok.sum() < 4 * bins:
        return None
    b = np.minimum((ph[ok] * bins).astype(int), bins - 1)
    prof = np.array([motion[ok][b == i].mean() if (b == i).any() else np.inf for i in range(bins)])
    return {"phase": float((np.argmin(prof) + 0.5) / bins), "profile": [round(float(x), 4) for x in prof], "bins": bins}


# ------------------------------------------------------------------------------------------------ seam planning


def plan_seams(spec: Spec, L: list) -> list:
    """One record per neighbouring pair: overlap, allowed range, the chosen blend window and its scan."""
    S = len(L) - 1
    n_blend = int(round(spec.blend_s * spec.hz))
    for i in range(S):
        if L[i + 1].k1 <= L[i].k1:
            raise ValueError(f"chunk {L[i + 1].name} (k {L[i + 1].k0}..{L[i + 1].k1}) ends inside {L[i].name} "
                             f"(k {L[i].k0}..{L[i].k1}): drop it or merge it into its neighbour")
    seams = []
    for i in range(S):
        P, N = L[i], L[i + 1]
        ov0, ov1 = max(P.k0, N.k0), min(P.k1, N.k1) + 1
        A0 = max(ov0, P.k0 + P.m0, N.k0 + N.m0)
        A1 = min(ov1, P.k1 + 1 - P.m1, N.k1 + 1 - N.m1)
        n = min(n_blend, A1 - A0)
        sm = {"name": f"{P.name}|{N.name}", "i": i, "P": i, "N": i + 1, "overlap": [ov0, ov1], "allowed": [A0, A1],
              "n": int(n), "n_wanted": n_blend}
        if ov1 - ov0 <= 0:
            sm.update(status="gap" if N.k0 > P.k1 + 1 else "cut")
            sm["why"] = (f"no overlap: {P.name} ends at k {P.k1}, {N.name} starts at k {N.k0}"
                         + (f" ({N.k0 - P.k1 - 1} frames uncovered)" if N.k0 > P.k1 + 1 else " (hard cut, nothing to blend)"))
        elif n < spec.min_blend_frames:
            sm.update(status="no_room", why=f"usable overlap {max(A1 - A0, 0)} frames < {spec.min_blend_frames} "
                      f"(overlap {ov1 - ov0}, margins {P.m1}+{N.m0})")
        else:
            sm["status"] = "ok"
        sm["short"] = bool(sm["status"] == "ok" and n < n_blend)
        seams.append(sm)
    # feasibility: seam i's window must end before seam i+1's can start (backward pass of the latest end)
    latest_hi = [None] * S
    for i in reversed(range(S)):
        sm = seams[i]
        if sm["status"] != "ok":
            continue
        hi = sm["allowed"][1]
        j = i + 1
        while j < S and seams[j]["status"] != "ok":
            j += 1
        if j < S:
            hi = min(hi, latest_hi[j] - seams[j]["n"])
        latest_hi[i] = hi
    for i, sm in enumerate(seams):
        if sm["status"] != "ok":
            continue
        lo_min, lo_max = sm["allowed"][0], latest_hi[i] - sm["n"]
        if lo_max < lo_min:
            sm.update(status="no_room", why=f"no room for a {sm['n']}-frame window before the next seam's")
            continue
        sm["lo_range"] = [lo_min, lo_max]    # _choose_windows adds the previous seam's window end
    return seams


def _choose_windows(spec: Spec, L: list, seams: list, J: dict, th: dict, beat_info: dict | None,
                    beats: np.ndarray | None) -> None:
    """Fill each ok seam with its window: scan, ties, choice (modifies `seams`)."""
    prev_hi = None
    for sm in seams:
        if sm["status"] != "ok":
            continue
        P, N = L[sm["P"]], L[sm["N"]]
        ov0, ov1 = sm["overlap"]
        lo_min, lo_max = sm["lo_range"]
        if prev_hi is not None:
            lo_min = max(lo_min, prev_hi)
        if lo_max < lo_min:
            sm.update(status="no_room", why="no room left after the previous seam's window")
            continue
        n = sm["n"]
        th_sel = dict(th)                                           # the thresholds the CHOICE sees (select_weights), the gate keeps `th`
        for key, name in (("arm", "elbow_wrist_p95"), ("fingers", "fingers_rel_wrist_p95")):
            w = float(spec.select_weights.get(key, 1.0))
            if w > 0:
                th_sel[name] = th[name] / w
        dist = {}
        for r in ROLES:
            jp = J[r][sm["P"]][ov0 - P.k0:ov1 - P.k0]
            jn = J[r][sm["N"]][ov0 - N.k0:ov1 - N.k0]
            dist[r] = disagreement(jp, jn)
        los = np.arange(lo_min, lo_max + 1)
        util = {r: np.array([utilisation(window_stats(*dist[r], slice(lo - ov0, lo - ov0 + n)), th_sel) for lo in los])
                for r in ROLES}
        worst = np.max([util[r] for r in ROLES], axis=0)
        worst = np.where(np.isfinite(worst), worst, np.inf)
        if not np.isfinite(worst).any():
            sm.update(status="no_room", why="the chunks' joints are not finite over the overlap")
            continue
        umin = float(worst.min())
        tied = np.flatnonzero(worst <= umin * (1 + spec.tie_rel) + spec.tie_abs)
        centre_k = los + (n - 1) / 2.0
        centre_t = spec.anchor + centre_k / spec.hz
        # low-motion tie-break: local motion (mean of both chunks and roles, +-0.25 s) and the beat phase profile
        half = int(round(0.25 * spec.hz))
        mot = np.zeros(len(los))
        for r in ROLES:
            for ci in (sm["P"], sm["N"]):
                c = L[ci]
                m = motion_series(J[r][ci], spec.hz)
                for q, lo in enumerate(los):
                    c0 = int(round(lo + (n - 1) / 2.0)) - c.k0
                    seg = m[max(c0 - half, 0):c0 + half + 1]
                    seg = seg[np.isfinite(seg)]
                    mot[q] += float(seg.mean()) / 4 if len(seg) else 0.0
        method = "motion"
        phase_dist = np.zeros(len(los))
        chosen_phase = None
        if beats is not None and beat_info is not None:
            ph = beat_phase(beats, centre_t)
            if np.isfinite(ph[tied]).any():
                d = np.abs(ph - beat_info["phase"])
                phase_dist = np.where(np.isfinite(ph), np.minimum(d, 1 - d), 1.0)
                method = "beat_phase"
        order = tied[np.lexsort((los[tied], mot[tied], phase_dist[tied]))]
        pick = int(order[0])
        lo = int(los[pick])
        if method == "beat_phase":
            chosen_phase = float(beat_phase(beats, centre_t[pick:pick + 1])[0])
        sm.update(lo=lo, hi=lo + n, centre_k=float(centre_k[pick]), centre_t=float(centre_t[pick]),
                  window_t=[float(spec.anchor + lo / spec.hz), float(spec.anchor + (lo + n - 1) / spec.hz)],
                  utilisation=float(max(utilisation(window_stats(*dist[r], slice(lo - ov0, lo - ov0 + n)), th) for r in ROLES)),
                  selection_utilisation=float(worst[pick]), select_weights=dict(spec.select_weights), best_utilisation=umin,
                  tie_break={"method": method, "n_tied": int(len(tied)), "tie_rel": spec.tie_rel, "tie_abs": spec.tie_abs,
                             "low_motion_phase": None if beat_info is None else beat_info["phase"],
                             "chosen_phase": chosen_phase, "local_motion_mps": round(float(mot[pick]), 4)},
                  scan={"centre_t": [round(float(x), 4) for x in centre_t], "utilisation": {
                      r: [round(float(x), 4) for x in util[r]] for r in ROLES},
                      "worst": [round(float(x), 4) for x in worst]})
        prev_hi = lo + n


# ------------------------------------------------------------------------------------------------ assembly


def _frame_keys(datas: list, ns: list) -> tuple[list, list]:
    """Per-frame arrays every chunk has with the same trailing shape (-> keys), and those only some chunks have."""
    def is_frame(d, k, n):
        v = d.get(k)
        return v is not None and hasattr(v, "ndim") and v.ndim >= 1 and v.shape[0] == n

    cand = sorted({k for d, n in zip(datas, ns) for k in d if k not in STATIC and is_frame(d, k, n)})
    keys, dropped = [], []
    for k in cand:
        shapes = {d[k].shape[1:] for d, n in zip(datas, ns) if is_frame(d, k, n)}
        if all(is_frame(d, k, n) for d, n in zip(datas, ns)) and len(shapes) == 1:
            keys.append(k)
        else:
            dropped.append(k)
    return keys, dropped


def ownership(spec: Spec, L: list, seams: list, kmin: int, kmax: int) -> dict:
    """Per grid frame: owning chunk (-1 = uncovered), seam index inside a blend window (-1 outside), later chunk's
    weight; plus each chunk's fully owned range [s, e)."""
    T = kmax - kmin + 1
    owner = np.full(T, -1, np.int16)
    seam_of = np.full(T, -1, np.int16)
    wnext = np.zeros(T)
    owned = []
    for c, ch in enumerate(L):
        s, e = ch.k0, ch.k1 + 1
        if c > 0 and seams[c - 1].get("lo") is not None:     # an edge is only dropped where a neighbour replaces it
            s = max(ch.k0 + ch.m0, seams[c - 1]["hi"])
        if c < len(L) - 1 and seams[c].get("lo") is not None:
            e = min(ch.k1 + 1 - ch.m1, seams[c]["lo"])
        if e < s:
            raise ValueError(f"chunk {ch.name} owns no frames (window ordering: {s}..{e})")
        owned.append((s, e))
        owner[s - kmin:e - kmin] = c
    for sm in seams:
        if sm.get("lo") is None:
            continue
        lo, hi = sm["lo"], sm["hi"]
        seam_of[lo - kmin:hi - kmin] = sm["i"]
        w = cosine_weights(hi - lo)
        wnext[lo - kmin:hi - kmin] = w
        owner[lo - kmin:hi - kmin] = np.where(w >= 0.5, sm["N"], sm["P"])
    return {"owner": owner, "seam": seam_of, "wnext": wnext, "owned": owned, "T": T, "kmin": kmin, "kmax": kmax}


def _hold_uncovered(out: dict, keys: list, covered: np.ndarray) -> None:
    idx = np.flatnonzero(covered)
    if not len(idx) or covered.all():
        return
    miss = np.flatnonzero(~covered)
    pos = np.searchsorted(idx, miss)
    left = idx[np.clip(pos - 1, 0, len(idx) - 1)]
    right = idx[np.clip(pos, 0, len(idx) - 1)]
    near = np.where(np.abs(miss - left) <= np.abs(right - miss), left, right)
    for k in keys:
        out[k][miss] = out[k][near]


def _blend_key(key: str, a: np.ndarray, b: np.ndarray, w: np.ndarray, dtype) -> np.ndarray:
    if key in POSE_KEYS:
        n = len(a)
        return blend_rotvec(np.asarray(a, float).reshape(n, -1, 3), np.asarray(b, float).reshape(n, -1, 3),
                            w).reshape(a.shape).astype(dtype)
    wb = w.reshape((-1,) + (1,) * (a.ndim - 1))
    if key in LINEAR_KEYS:
        return ((1 - wb) * a + wb * b).astype(dtype)
    if key in WORST_HI:
        return np.fmax(a, b).astype(dtype)
    if key in WORST_LO:
        return np.fmin(a, b).astype(dtype)
    if np.asarray(a).dtype == bool and key not in DOMINANT_KEYS:
        return a | b
    return np.where(wb >= 0.5, b, a).astype(dtype)


def assemble(L: list, own: dict, seams: list, role: str, which: str, ref: int = 0) -> tuple[dict, dict]:
    """The stitched arrays of one role from the chunks' `raw` or `refined` data. Returns (arrays, info)."""
    datas = [getattr(c, which)[role] for c in L]
    ns = [c.n for c in L]
    keys, dropped = _frame_keys(datas, ns)
    T, kmin, owner = own["T"], own["kmin"], own["owner"]
    out = {}
    for k in keys:
        dt = np.result_type(*[d[k].dtype for d in datas])
        out[k] = np.zeros((T,) + datas[0][k].shape[1:], dt)
    covered = np.zeros(T, bool)
    for c, (s, e) in enumerate(own["owned"]):
        if e > s:
            for k in keys:
                out[k][s - kmin:e - kmin] = datas[c][k][s - L[c].k0:e - L[c].k0]
            covered[s - kmin:e - kmin] = True
    blended = []
    for sm in seams:
        if sm.get("lo") is None:
            continue
        lo, hi = sm["lo"], sm["hi"]
        P, N = L[sm["P"]], L[sm["N"]]
        w = cosine_weights(hi - lo)
        sl = slice(lo - kmin, hi - kmin)
        for k in keys:
            a = datas[sm["P"]][k][lo - P.k0:hi - P.k0]
            b = datas[sm["N"]][k][lo - N.k0:hi - N.k0]
            out[k][sl] = _blend_key(k, a, b, w, out[k].dtype)
        covered[sl] = True
        blended.append(np.arange(lo - kmin, hi - kmin))
    _hold_uncovered(out, keys, covered)
    if blended:   # joints of the blended frames: FK with the common betas (batched over every window)
        idx = np.concatenate(blended)
        z = {k: out[k][idx] for k in (*POSE_KEYS, "transl")}
        out["joints"][idx, :55] = fk_joints(pose165(z), z["transl"], datas[ref]["betas"]).astype(out["joints"].dtype)
    for k in STATIC:
        if k in datas[ref]:
            out[k] = np.array(datas[ref][k])
    for k, v in datas[ref].items():     # 0-d metadata of the reference
        if k not in out and hasattr(v, "ndim") and v.ndim == 0 and k not in ("smoothing", "source"):
            out[k] = np.array(v)
    return out, {"keys": keys, "dropped": dropped, "covered": covered}


def pelvis_speed(J: np.ndarray, hz: float) -> np.ndarray:
    v = np.full(len(J), np.nan)
    v[1:] = np.linalg.norm(np.diff(J[:, 0], axis=0), axis=-1) * hz
    return v


def continuity(J_out: np.ndarray, JP: np.ndarray, JN: np.ndarray, sm: dict, kmin: int, P, N, hz: float) -> dict:
    """What the cross-fade did inside the window vs what each chunk does by itself (mamma_seams' definitions): peak
    pelvis / body acceleration and speed step; max frame-to-frame joint step."""
    lo, hi = sm["lo"], sm["hi"]

    def acc(x):            # |second difference| * hz^2 per frame (NaN at both ends); (T, 3) or (T, J, 3) -> mean over J
        a = np.full(len(x), np.nan)
        n = np.linalg.norm(x[2:] - 2 * x[1:-1] + x[:-2], axis=-1) * hz * hz
        a[1:-1] = n if n.ndim == 1 else n.mean(axis=1)
        return a

    def speed(x):
        v = np.full(len(x), np.nan)
        v[1:] = np.linalg.norm(np.diff(x, axis=0), axis=-1) * hz
        return v

    def peaks(J, a, b):
        ap, ab, sp = acc(J[:, 0]), acc(J[:, 1:22] - J[:, :1]), speed(J[:, 0])
        ds = np.full(len(sp), np.nan)
        ds[1:] = np.abs(np.diff(sp))
        return float(np.nanmax(ap[a:b])), float(np.nanmax(ab[a:b])), float(np.nanmax(ds[a:b]))
    mp, mb, ms_ = peaks(J_out, lo - kmin, hi - kmin)
    pp, pb, ps = peaks(JP, lo - P.k0, hi - P.k0)
    np_, nb, ns = peaks(JN, lo - N.k0, hi - N.k0)
    ratio = [mp / max(pp, np_, 1e-9), mb / max(pb, nb, 1e-9), ms_ / max(ps, ns, 1e-9)]
    step = np.linalg.norm(np.diff(J_out[max(lo - kmin - 1, 0):hi - kmin + 1, :55], axis=0), axis=-1) * 1000
    src = max(float(np.linalg.norm(np.diff(JP[lo - P.k0:hi - P.k0], axis=0), axis=-1).max()),
              float(np.linalg.norm(np.diff(JN[lo - N.k0:hi - N.k0], axis=0), axis=-1).max())) * 1000
    return {"pelvis_acc_ratio": round(ratio[0], 3), "body_acc_ratio": round(ratio[1], 3),
            "speed_step_ratio": round(ratio[2], 3), "smooth": bool(max(ratio) <= 1.25),
            "max_joint_step_mm": round(float(step.max()), 2), "max_joint_step_chunks_mm": round(src, 2)}


# ------------------------------------------------------------------------------------------------ flags


def _spans_mask(t: np.ndarray, spans: list) -> np.ndarray:
    m = np.zeros(len(t), bool)
    for a, b in spans or []:
        m |= (t >= a) & (t <= b)
    return m


def chunk_ncams(c: Loaded, spec: Spec, role: str) -> tuple[np.ndarray, str]:
    """Cameras that see `role` on each frame of the chunk."""
    p = ROLES.index(role)
    d = c.ch.ma_2d
    files = sorted(Path(d).glob("cam*.npz")) if d is not None and Path(d).is_dir() else []
    if files:
        tot = np.zeros(c.n, np.int16)
        for f in files:
            with np.load(f) as z:
                v = z["visibilities"]
            if len(v) != c.n:    # re-prepared on another grid after the fit: not this fit's observations
                c.warnings.append(f"{c.name}: {f.name} has {len(v)} frames, the fit {c.n}: n_cams from the camera spans")
                tot = None
                break
            tot += ((v[:, p] > 0.5).sum(-1) >= spec.min_points).astype(np.int16)
        if tot is not None:
            return tot, "ma_2d visibilities"
    t = spec.anchor + np.arange(c.k0, c.k1 + 1) / spec.hz
    prep = c.ch.prepare
    if prep is not None and Path(prep).is_file():
        try:
            doc = json.loads(Path(prep).read_text())
            spans = doc["grid"]["trim"]["cams_per_frame"]
            return sum(((t >= a) & (t <= b)).astype(np.int16) for a, b in spans.values()), "prepare.json camera spans"
        except (KeyError, TypeError, ValueError):
            try:
                tag = max(doc["sets"], key=lambda k: len(doc["sets"][k]["cams"]))
                return np.full(c.n, len(doc["sets"][tag]["cams"]), np.int16), "prepare.json camera set"
            except (KeyError, ValueError):
                pass
    return np.full(c.n, -1, np.int16), "unknown"


def quality_flags(spec: Spec, role: str, n_cams: np.ndarray, t: np.ndarray, covered: np.ndarray) -> tuple[np.ndarray, dict]:
    n = n_cams.astype(int)
    flag = np.select([n == 0, n == 1, n == 2], [0, 1, 2], 3).astype(np.int8)
    flag[n < 0] = 3                      # unknown camera count: not claimed, reported in stitch.json
    # a dropout (0 cameras) shorter than absent_min_s is a weak frame, not an absence
    zero = n == 0
    run_ok = np.zeros(len(n), bool)
    i = 0
    while i < len(n):
        if zero[i]:
            j = i
            while j + 1 < len(n) and zero[j + 1]:
                j += 1
            run_ok[i:j + 1] = (j - i + 1) >= int(round(spec.absent_min_s * spec.hz))
            i = j + 1
        else:
            i += 1
    flag[zero & ~run_ok] = 1
    flag = np.minimum(flag, np.where(_spans_mask(t, spec.mono), 1, 3)).astype(np.int8)
    flag = np.minimum(flag, np.where(_spans_mask(t, spec.low_view), 2, 3)).astype(np.int8)
    flag[_spans_mask(t, spec.absent.get(role, []))] = 0
    flag[~covered] = 0
    return flag, {"declared_absent": spec.absent.get(role, []), "zero_camera_frames": int(zero.sum()),
                  "absent_frames": int((flag == 0).sum())}


def _runs(mask: np.ndarray, t: np.ndarray) -> list:
    out, i = [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j + 1 < len(mask) and mask[j + 1]:
                j += 1
            out.append([round(float(t[i]), 3), round(float(t[j]), 3)])
            i = j + 1
        else:
            i += 1
    return out


# ------------------------------------------------------------------------------------------------ deterministic npz


def save_npz(path: Path, arrays: dict) -> None:
    """np.savez_compressed with fixed zip timestamps: the same arrays give the same bytes (running twice is a no-op)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for k, v in arrays.items():
            zi = zipfile.ZipInfo(k + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            with zf.open(zi, "w", force_zip64=True) as f:
                np.lib.format.write_array(f, np.asanyarray(v), allow_pickle=False)
    os.replace(tmp, path)


def _js(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return o.as_posix()
    return str(o)


# ------------------------------------------------------------------------------------------------ bit identity


def bit_identity(out: dict, L: list, own: dict, which: str, keys: list, reload_reference: int | None = None) -> dict:
    """Max abs difference between the stitched arrays and the chunk arrays on every frame the chunk owns outright
    (outside every blend window and edge margin). reload_reference: that chunk is re-read from disk (verifying the
    written files against the source, not against memory)."""
    rep = {}
    for c, (s, e) in enumerate(own["owned"]):
        ch = L[c]
        per = {}
        for r in ROLES:
            src = getattr(ch, which)[r]
            if reload_reference == c:
                d = Path(ch.ch.src if which == "raw" else ch.ch.refined)
                with np.load(d / f"{r}.npz", allow_pickle=True) as z:
                    src = {k: z[k] for k in keys if k in z.files}
            mx = {}
            for k in keys:
                if k == "t" or k not in out[r] or k not in src or e <= s:
                    continue
                a = out[r][k][s - own["kmin"]:e - own["kmin"]]
                b = src[k][s - ch.k0:e - ch.k0]
                if k in ZEROED_WHEN_ABSENT and "absent" in out[r]:     # zeroed on absent frames by design
                    keep = ~np.asarray(out[r]["absent"][s - own["kmin"]:e - own["kmin"]], bool)
                    a, b = a[keep], b[keep]
                if a.dtype == bool or b.dtype == bool:
                    mx[k] = float((a != b).sum())
                else:
                    diff = np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))
                    mx[k] = float(np.nanmax(diff)) if diff.size else 0.0
            pose = max([mx.get(k, 0.0) for k in POSE_KEYS] or [0.0])
            tdiff = 0.0
            if e > s and "t" in src:
                tdiff = float(np.abs(out[r]["t"][s - own["kmin"]:e - own["kmin"]] - np.asarray(src["t"][s - ch.k0:e - ch.k0], float)).max())
            per[r] = {"t_max_abs_diff_s": tdiff, "max_abs_diff_pose": pose, "max_abs_diff_transl": mx.get("transl", 0.0),
                      "max_abs_diff_joints": mx.get("joints", 0.0),
                      "max_abs_diff_all": max(mx.values()) if mx else 0.0, "worst_key": max(mx, key=mx.get) if mx else None}
        rep[ch.name] = {"owned_k": [s, e], "frames": max(e - s, 0), "roles": per,
                        "identical": bool(all(v["max_abs_diff_all"] == 0.0 for v in per.values()))}
    return rep


# ------------------------------------------------------------------------------------------------ the run


def _refit_recommendation(spec: Spec, L: list, sm: dict) -> dict:
    half = int(round(spec.refit_s * spec.hz / 2))
    c = int(round(sm["centre_k"]))
    k0, k1 = c - half, c + half
    rec = {"window_k": [k0, k1], "frames": k1 - k0,
           "window_t": [round(spec.anchor + k0 / spec.hz, 3), round(spec.anchor + (k1 - 1) / spec.hz, 3)],
           "centred_on_t": round(sm["centre_t"], 3), "inside": {}, "outside_k": []}
    for ci in (sm["P"], sm["N"]):
        ch = L[ci]
        inside = k0 >= ch.k0 and k1 <= ch.k1 + 1
        rec["inside"][ch.name] = {"grid_k": [ch.k0, ch.k1], "contains_window": inside,
                                  "prepared_index": [k0 - ch.k0, k1 - ch.k0] if inside else None}
    lo_all = min(L[sm["P"]].k0, L[sm["N"]].k0)
    hi_all = max(L[sm["P"]].k1, L[sm["N"]].k1) + 1
    if k0 < lo_all:
        rec["outside_k"].append([k0, lo_all])
    if k1 > hi_all:
        rec["outside_k"].append([hi_all, k1])
    rec["how"] = (f"refit the {spec.refit_s:.0f} s window k {k0}..{k1 - 1} ({rec['window_t'][0]:.2f}-{rec['window_t'][1]:.2f} s) "
                  "centred on the blend: `mamma fit <segment take> --tag <tag> --fit-frames <a>:<b>` with a:b the "
                  "prepared index above (re-run `mamma prepare` with a wider fit_k when no chunk contains it), add "
                  "the result to the chunk list as an extra chunk (kind refit) and stitch again")
    return rec


def stitch(spec: Spec, *, check: bool = False, only_seam: str | None = None, partial: bool = False,
           want_refined: bool = True, sheets: bool = True, force: bool = False, out_dir: Path | None = None,
           verbose: bool = True) -> dict:
    """Run the whole stitch; returns the stitch.json document (+ arrays under '_' keys when check/in-memory)."""
    log = (lambda *a: print("[stitch]", *a, flush=True)) if verbose else (lambda *a: None)
    th = spec.thresholds()
    work = Path(out_dir) if out_dir else spec.work
    if only_seam:
        check = True
    chunks, missing = [], []
    for ch in spec.chunks:
        if (Path(ch.src) / "lead.npz").exists() and (Path(ch.src) / "follow.npz").exists():
            chunks.append(ch)
        else:
            missing.append(ch.name)
    if missing and not partial:
        raise FileNotFoundError(f"no fit yet for chunk(s) {missing} (expected {{lead,follow}}.npz in "
                                f"{[str(c.src) for c in spec.chunks if c.name in missing]}); --partial stitches the rest")
    if len(chunks) < 1:
        raise FileNotFoundError("no chunk has a fit")
    t0 = time.time()
    L = [load_chunk(ch, spec, want_refined) for ch in chunks]
    L.sort(key=lambda c: (c.k0, c.k1))
    ref = next((i for i, c in enumerate(L) if c.ch.reference), 0)
    names = [c.name for c in L]
    world = check_worlds(L, spec)
    for w in world["warnings"]:
        log("WARNING " + w)
    log(f"{len(L)} chunks: " + ", ".join(f"{c.name} k {c.k0}..{c.k1} ({c.n} fr)" for c in L) + f"; reference {names[ref]}")
    # grid span: all chunks (a partial stitch keeps the declared full span so the gaps show)
    kmin = min(c.k0 for c in L)
    kmax = max(c.k1 for c in L)
    if partial:    # keep the announced span of the chunks that are still missing, so the gaps show
        ann = [c for c in spec.chunks if c.k_lo < 10 ** 9]
        kmin, kmax = min(kmin, min(c.k_lo for c in ann)), max(kmax, max(c.k_hi for c in ann))
    for c in L:
        for w in c.warnings:
            log("WARNING " + w)
    T = kmax - kmin + 1
    # betas
    betas_rep = unify_betas(L, ref)
    for r in ROLES:
        bad = [n for n, e in betas_rep[r].items() if not e["pinned"]]
        log(f"betas {r}: " + ("all chunks carry the reference's (pinned)" if not bad else f"re-expressed: {bad}"))
    J = {r: [np.asarray(c.raw[r]["joints"], float)[:, :55] for c in L] for r in ROLES}
    # windows
    seams = plan_seams(spec, L)
    beats = load_beats(spec.beats)
    beat_info = None
    if beats is not None:
        tt, mm = [], []
        for ci, c in enumerate(L):
            tt.append(spec.anchor + np.arange(c.k0, c.k1 + 1) / spec.hz)
            mm.append(np.mean([motion_series(J[r][ci], spec.hz) for r in ROLES], axis=0))
        beat_info = low_motion_phase(beats, np.concatenate(tt), np.concatenate(mm))
    _choose_windows(spec, L, seams, J, th, beat_info, beats)
    own = ownership(spec, L, seams, kmin, kmax)
    grid_t = spec.anchor + np.arange(kmin, kmax + 1) / spec.hz
    # raw assembly
    out, info = {}, {}
    for r in ROLES:
        out[r], info[r] = assemble(L, own, seams, r, "raw", ref)
    covered = info["lead"]["covered"]
    # per-seam statistics + gate
    doc_seams, why_all = [], []
    for sm in seams:
        rec = {k: v for k, v in sm.items() if k != "lo_range"}
        rec["between"] = [L[sm["P"]].name, L[sm["N"]].name]
        if sm["status"] != "ok" or sm.get("lo") is None:
            rec["gate"] = {"passed": False, "why": [f"{sm['name']}: {sm.get('why', sm['status'])}"]}
            rec["ok"] = False
            if sm["status"] != "gap" and sm["overlap"][1] > sm["overlap"][0]:
                rec["window_k"] = [sm["overlap"][0], sm["overlap"][1] - 1]
            why_all += rec["gate"]["why"]
            doc_seams.append(rec)
            continue
        P, N = L[sm["P"]], L[sm["N"]]
        ov0, ov1 = sm["overlap"]
        lo, hi = sm["lo"], sm["hi"]
        rec["roles"], why = {}, []
        for r in ROLES:
            jp = J[r][sm["P"]][ov0 - P.k0:ov1 - P.k0]
            jn = J[r][sm["N"]][ov0 - N.k0:ov1 - N.k0]
            d, relf = disagreement(jp, jn)
            st = window_stats(d, relf, slice(lo - ov0, hi - ov0))      # the gate: over the blend window
            ov = window_stats(d, relf, slice(None))                    # informational: the whole overlap
            swap = max(betas_rep[r][P.name]["swap_mm"]["p95"], betas_rep[r][N.name]["swap_mm"]["p95"])
            checks, w_r = gate_checks(r, sm["name"], st, swap, th)
            why += w_r
            cont = continuity(out[r]["joints"], J[r][sm["P"]], J[r][sm["N"]], sm, kmin, P, N, spec.hz)

            def brief(x):
                return {"joint_p95_max": round(x["joint_p95_max"], 2), "joint": x["joint"],
                        "arm_p95_max": round(x["arm_p95_max"], 2), "arm_joint": x["arm_joint"],
                        "fingers_rel_wrist_p95_max": round(x["fingers_rel_wrist_p95_max"], 2), "frames": x["frames"]}
            rec["roles"][r] = {"blend": brief(st),
                               "per_joint_p95": {JOINT_NAMES[j]: round(float(st["per_joint_p95"][j]), 1) for j in range(22)},
                               "pelvis_median_blend_mm": round(st["pelvis_median"], 2),
                               "overlap": brief(ov),
                               "pooled": {"median": round(float(np.median(d)), 2), "p95": round(float(np.percentile(d, 95)), 2),
                                          "p95_body": round(float(np.percentile(d[:, :22], 95)), 2),
                                          "p95_fingers": round(float(np.percentile(d[:, 25:], 95)), 2),
                                          "pelvis_median": round(float(np.median(d[:, 0])), 2)},
                               "betas_swap_p95_mm": swap, "checks": checks, "continuity": cont,
                               "utilisation": round(utilisation(st, th), 3)}
        rec["gate"] = {"passed": not why, "why": why}
        rec["ok"] = not why
        rec["window_k"] = [lo, hi - 1]            # inclusive lattice indices (what farm.timeline.parse_stitch reads)
        if why:
            rec["refit"] = _refit_recommendation(spec, L, sm)
        why_all += why
        doc_seams.append(rec)
        log(f"seam {sm['name']}: window t {sm['window_t'][0]:.2f}-{sm['window_t'][1]:.2f} s (centre {sm['centre_t']:.2f} s, "
            f"{sm['tie_break']['n_tied']} tied, {sm['tie_break']['method']}), utilisation {sm['utilisation']:.2f}: "
            + ("PASS" if not why else "FAIL " + "; ".join(why)))
    # flags
    ncams = {r: np.full(T, -1, np.int16) for r in ROLES}
    nsrc = {}
    for r in ROLES:
        per_chunk = []
        for c in L:
            n, src = chunk_ncams(c, spec, r)
            per_chunk.append(n)
            nsrc.setdefault(r, {})[c.name] = src
        for ci, (s, e) in enumerate(own["owned"]):
            ncams[r][s - kmin:e - kmin] = per_chunk[ci][s - L[ci].k0:e - L[ci].k0]
        for sm in seams:
            if sm.get("lo") is None:
                continue
            sl = slice(sm["lo"] - kmin, sm["hi"] - kmin)
            P, N = L[sm["P"]], L[sm["N"]]
            w = cosine_weights(sm["hi"] - sm["lo"])
            a = per_chunk[sm["P"]][sm["lo"] - P.k0:sm["hi"] - P.k0]
            b = per_chunk[sm["N"]][sm["lo"] - N.k0:sm["hi"] - N.k0]
            ncams[r][sl] = np.where(w >= 0.5, b, a)
        miss = ~covered
        if miss.any():
            ncams[r][miss] = 0
    flags, fl_info = {}, {}
    for r in ROLES:
        flags[r], fl_info[r] = quality_flags(spec, r, ncams[r], grid_t, covered)
    # provenance arrays + quality override, per role
    seam_idx = own["seam"]
    for r in ROLES:
        o = out[r]
        o["t"] = grid_t.copy()
        o["segment_owner"] = np.array([names[i] if i >= 0 else "" for i in own["owner"]])   # the chunk's name ("" = none)
        o["segment_index"] = own["owner"].copy()
        o["chunk_names"] = np.array(names)
        o["blend_seam"] = seam_idx.copy()
        o["blend_weight"] = own["wnext"].astype(np.float32)
        o["covered"] = covered.copy()
        o["n_cams"] = ncams[r].copy()
        o["quality_flag"] = flags[r].copy()
        o["quality_flag_names"] = np.array(FLAG_NAMES)
        o["absent"] = flags[r] == 0
        ab = o["absent"]
        if "quality" in o:
            o["quality"] = np.where(ab, 0.0, o["quality"]).astype(o["quality"].dtype)
        if "quality_parts" in o:
            o["quality_parts"] = np.where(ab[:, None], 0.0, o["quality_parts"]).astype(o["quality_parts"].dtype)
        o["source"] = np.array("dancecap.stitch of " + ", ".join(
            f"{c.name} (k {c.k0}..{c.k1})" for c in L) + "; betas " + names[ref])
    # bit identity (raw)
    keys_raw = info["lead"]["keys"]
    bit = {"raw": bit_identity(out, L, own, "raw", keys_raw)}
    rc = L[ref]
    s_ref, e_ref = own["owned"][ref]

    def _who(a: int, b: int) -> list:
        """Run-length list of what produces the frames k in [a, b): a blend window or the chunk that owns them."""
        runs = []
        for k in range(a, b):
            g = k - kmin
            lab = (f"blend {seams[int(own['seam'][g])]['name']}" if own["seam"][g] >= 0 else
                   (f"owned by {names[int(own['owner'][g])]}" if own["owner"][g] >= 0 else "uncovered"))
            if runs and runs[-1]["what"] == lab:
                runs[-1]["k"][1] = k + 1
                runs[-1]["frames"] += 1
            else:
                runs.append({"k": [k, k + 1], "frames": 1, "what": lab})
        return runs
    bit["reference"] = {
        "chunk": rc.name, "k": [rc.k0, rc.k1], "owned_k": [s_ref, e_ref],
        "windows": [[sm["lo"], sm["hi"]] for sm in seams if sm.get("lo") is not None and ref in (sm["P"], sm["N"])],
        "not_owned": [r for a, b in ((rc.k0, s_ref), (e_ref, rc.k1 + 1)) if b > a for r in _who(a, b)],
        "note": f"{rc.name}'s frames outside the blend windows are copied bit for bit wherever {rc.name} owns the output "
                f"(owned_k); the rest of its k range is a blend window or belongs to the neighbouring chunk: its first / "
                f"last {rc.m0} frames (margin) are never used, so the windows stay away from its edge frames"}
    # refined
    refined_doc = None
    out_ref = None
    have_ref = all(c.refined is not None for c in L)
    if want_refined and spec.out_refined and have_ref:
        out_ref, info_ref = {}, {}
        for r in ROLES:
            out_ref[r], info_ref[r] = assemble(L, own, seams, r, "refined", ref)
            o = out_ref[r]
            for k in ("t", "segment_owner", "segment_index", "chunk_names", "blend_seam", "blend_weight", "covered", "n_cams",
                      "quality_flag", "quality_flag_names", "absent"):
                o[k] = out[r][k]
            if "quality" in o:
                o["quality"] = np.where(o["absent"], 0.0, o["quality"]).astype(o["quality"].dtype)
            if "quality_parts" in o:
                o["quality_parts"] = np.where(o["absent"][:, None], 0.0, o["quality_parts"]).astype(o["quality_parts"].dtype)
            o["source"] = out[r]["source"]
            sm_chunks = {}
            for c in L:
                try:
                    sm_chunks[c.name] = json.loads(str(c.refined[r]["smoothing"]))
                except (KeyError, ValueError):
                    sm_chunks[c.name] = None
            o["smoothing"] = np.array(json.dumps({"tool": "dancecap.stitch", "mode": "stitched refined chunks",
                                                  "windows": [[s["lo"], s["hi"]] for s in seams if s.get("lo") is not None],
                                                  "chunks": sm_chunks}, default=_js))
        bit["refined"] = bit_identity(out_ref, L, own, "refined", info_ref["lead"]["keys"])
        # informational refined seam statistics over the same windows
        rs = []
        for sm in seams:
            if sm.get("lo") is None:
                continue
            P, N = L[sm["P"]], L[sm["N"]]
            ov0, ov1 = sm["overlap"]
            ent = {"name": sm["name"], "roles": {}}
            for r in ROLES:
                jp = np.asarray(P.refined[r]["joints"], float)[ov0 - P.k0:ov1 - P.k0, :55]
                jn = np.asarray(N.refined[r]["joints"], float)[ov0 - N.k0:ov1 - N.k0, :55]
                d, relf = disagreement(jp, jn)
                st = window_stats(d, relf, slice(sm["lo"] - ov0, sm["hi"] - ov0))
                ent["roles"][r] = {"body_p95_mm": round(st["joint_p95_max"], 1), "body_joint": st["joint"],
                                   "elbow_wrist_p95_mm": round(st["arm_p95_max"], 1), "arm_joint": st["arm_joint"],
                                   "fingers_rel_wrist_p95_mm": round(st["fingers_rel_wrist_p95_max"], 1),
                                   "pelvis_median_mm": round(st["pelvis_median"], 1), "utilisation": round(utilisation(st, th), 3)}
            ent["within_gate"] = all(v["utilisation"] <= 1.0 for v in ent["roles"].values())
            rs.append(ent)
        refined_doc = {"dir": spec.out_refined, "seams": rs, "informational": True,
                       "warnings": [f"refined {e['name']} is over a gate limit (raw gate decides)" for e in rs if not e["within_gate"]]}
    elif want_refined and spec.out_refined:
        refined_doc = {"dir": spec.out_refined, "written": False,
                       "why": "chunks without a refined folder: " + ", ".join(c.name for c in L if c.refined is None)}
    # coverage (1 s tiles: C complete = covered, >= 3 cameras for both dancers, gates passed; P partial; N none)
    fail_seam = np.zeros(T, bool)
    for rec in doc_seams:
        if not rec["gate"]["passed"] and rec.get("lo") is not None:
            fail_seam[rec["lo"] - kmin:rec["hi"] - kmin] = True
    full = covered & (flags["lead"] == 3) & (flags["follow"] == 3) & ~fail_seam
    per_frame = np.where(~covered, "N", np.where(full, "C", "P"))
    hz = int(round(spec.hz))
    tiles = "".join("N" if (g == "N").all() else ("C" if (g == "C").all() else "P")
                    for g in (per_frame[i:i + hz] for i in range(0, T, hz)))
    coverage = {"frames": {"complete": int((per_frame == "C").sum()), "partial": int((per_frame == "P").sum()),
                           "none": int((per_frame == "N").sum())},
                "seconds_per_tile": 1.0, "tiles": tiles, "t0": float(grid_t[0]),
                "legend": "C complete (covered, >= 3 cameras for both dancers, seams passed), P partial (mono / two_view / "
                          "absent dancer or a failing seam), N not covered"}
    # first tracked frame per role
    first = {r: (int(np.flatnonzero(flags[r] > 0)[0]) if (flags[r] > 0).any() else None) for r in ROLES}
    sel_names = [s["name"] for s in doc_seams]
    if only_seam and only_seam not in sel_names:
        raise SystemExit(f"[stitch] no seam {only_seam!r}; seams: {sel_names}")
    if missing:
        why_all.append(f"partial stitch: no fit yet for chunk(s) {missing}")
    gate_passed = not why_all
    doc = {
        "schema": SCHEMA, "take": spec.take, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "grid": {"anchor": spec.anchor, "hz": spec.hz, "k": [kmin, kmax], "T": T, "t": [float(grid_t[0]), float(grid_t[-1])],
                 "frame_offset": {c.name: c.k0 - kmin for c in L},
                 "note": "full-grid frame = k - k_min; a chunk's local frame i is full-grid frame i + frame_offset"},
        "reference": names[ref], "chunk_names": names, "missing_chunks": missing, "world": world,
        "warnings": [w for c in L for w in c.warnings],
        "chunks": [{"name": c.name, "kind": c.ch.kind, "k": [c.k0, c.k1], "frames": c.n,
                    "t": [round(spec.anchor + c.k0 / spec.hz, 4), round(spec.anchor + c.k1 / spec.hz, 4)],
                    "margin_frames": c.m0, "owned_k": list(own["owned"][i]), "src": c.ch.src.as_posix(),
                    "refined": c.ch.refined.as_posix() if c.ch.refined is not None else None,
                    "sha256": c.sha, "betas": {r: betas_rep[r][c.name] for r in ROLES},
                    "n_cams_source": {r: nsrc[r][c.name] for r in ROLES}} for i, c in enumerate(L)],
        "thresholds": th, "blend": {"seconds": spec.blend_s, "frames": int(round(spec.blend_s * spec.hz)),
                                    "shape": "cosine; per-joint slerp, linear translation and contacts",
                                    "edge_margin_s_default": spec.edge_margin_s},
        "beat_phase": beat_info, "seams": doc_seams,
        "gate": {"passed": gate_passed, "why": why_all},
        "bit_identity": bit, "flags": {r: {**fl_info[r], "first_tracked_frame": first[r],
                                           "first_tracked_t": None if first[r] is None else float(grid_t[first[r]]),
                                           "counts": {FLAG_NAMES[i]: int((flags[r] == i).sum()) for i in range(4)},
                                           "absent_runs_t": _runs(flags[r] == 0, grid_t),
                                           "mono_runs_t": _runs(flags[r] == 1, grid_t),
                                           "two_view_runs_t": _runs(flags[r] == 2, grid_t)} for r in ROLES},
        "flag_names": list(FLAG_NAMES), "dropped_keys": {r: info[r]["dropped"] for r in ROLES},
        "refined": refined_doc, "coverage": coverage, "check": bool(check),
        "seconds": round(time.time() - t0, 1)}
    ref_bit = bit["raw"][names[ref]]
    log(f"bit identity: {names[ref]} owns {ref_bit['frames']} frames outright, max abs diff "
        f"{max(v['max_abs_diff_all'] for v in ref_bit['roles'].values()):.1f} (pose {max(v['max_abs_diff_pose'] for v in ref_bit['roles'].values()):.1f}, "
        f"transl {max(v['max_abs_diff_transl'] for v in ref_bit['roles'].values()):.1f})")
    result = dict(doc)
    result["_"] = {"out": out, "out_refined": out_ref, "L": L, "seams": seams, "own": own, "J": J, "grid_t": grid_t,
                   "ncams": ncams, "flags": flags, "ref": ref}
    if only_seam:
        doc["seams"] = [s for s in doc["seams"] if s["name"] == only_seam]
        doc["gate"] = {"passed": all(s["gate"]["passed"] for s in doc["seams"]),
                       "why": [w for s in doc["seams"] for w in s["gate"]["why"]]}
        result.update(seams=doc["seams"], gate=doc["gate"])
        gate_passed = doc["gate"]["passed"]
    if check:
        if out_dir is None:
            log("check only: nothing written; gate " + ("PASSED" if gate_passed else "FAILED"))
            return result
        # a verification job (pod): the full-length gates + seam plots into the output folder, never the motion
        if sheets:
            try:
                doc["review"] = {"pose_seams_full": pose_seams_sheet(result, work / "review" / "pose_seams_full.png").as_posix(),
                                 "stitch_overview": overview_sheet(result, work / "review" / "stitch_overview.png").as_posix()}
            except Exception as e:
                doc["review"] = {"error": f"{type(e).__name__}: {e}"}
                log(f"review sheets failed: {e}")
        work.mkdir(parents=True, exist_ok=True)
        (work / "stitch_check.json").write_text(json.dumps(doc, indent=1, default=_js), encoding="utf-8")
        log(f"check: gate {'PASSED' if gate_passed else 'FAILED'}; wrote {work / 'stitch_check.json'}"
            + (" + review sheets" if sheets else "") + " (no motion written)")
        return result
    # ---------------- write
    promoted = gate_passed or force
    rej = "stitch_partial" if missing else "stitch_rejected"
    raw_dir = work / spec.out_raw if promoted else work / rej / spec.out_raw
    ref_dir = (work / spec.out_refined if promoted else work / rej / spec.out_refined) \
        if (out_ref is not None) else None
    for r in ROLES:
        save_npz(raw_dir / f"{r}.npz", {k: out[r][k] for k in out[r]})
        if ref_dir is not None:
            save_npz(ref_dir / f"{r}.npz", {k: out_ref[r][k] for k in out_ref[r]})
    doc["outputs"] = {"promoted": promoted, "forced": bool(force and not gate_passed), "raw": raw_dir.as_posix(),
                      "refined": None if ref_dir is None else ref_dir.as_posix(),
                      "sha256": {f"{raw_dir.name}/{r}.npz": sha256_file(raw_dir / f"{r}.npz") for r in ROLES}}
    if ref_dir is not None:
        doc["outputs"]["sha256"].update({f"{ref_dir.name}/{r}.npz": sha256_file(ref_dir / f"{r}.npz") for r in ROLES})
    if not promoted:
        log(f"gate FAILED: outputs written to {raw_dir.parent} (smplx/ untouched); --force promotes them")
    # verify what was written against the sources
    ver = {}
    wr = read_dir(raw_dir)
    ver["raw"] = bit_identity(wr, L, own, "raw", keys_raw, reload_reference=ref)
    if ref_dir is not None:
        wf = read_dir(ref_dir)
        ver["refined"] = bit_identity(wf, L, own, "refined", info_ref["lead"]["keys"], reload_reference=ref)
    doc["verified_on_disk"] = {k: {n: {"frames": v["frames"], "identical": v["identical"]} for n, v in rep.items()}
                               for k, rep in ver.items()}
    review = work / "review"
    if sheets:
        try:
            doc["review"] = {"pose_seams_full": pose_seams_sheet(result, review / "pose_seams_full.png").as_posix(),
                             "stitch_overview": overview_sheet(result, review / "stitch_overview.png").as_posix()}
        except Exception as e:    # a missing display / matplotlib problem must not lose the stitch
            doc["review"] = {"error": f"{type(e).__name__}: {e}"}
            log(f"review sheets failed: {e}")
    (work / "stitch.json").write_text(json.dumps(doc, indent=1, default=_js), encoding="utf-8")
    log(f"gate {'PASSED' if gate_passed else 'FAILED'}; wrote {raw_dir}" + (f" + {ref_dir}" if ref_dir else "")
        + f", stitch.json ({doc['seconds']} s)")
    result.update(outputs=doc["outputs"], verified_on_disk=doc["verified_on_disk"])
    return result


# ------------------------------------------------------------------------------------------------ review sheets


def _style(ax, ink2="#52514e"):
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, color="#e6e5e0", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#b5b4ae")
    ax.tick_params(colors=ink2, labelsize=8)


def pose_seams_sheet(res: dict, path: Path) -> Path:
    """Per seam (row): pelvis path of both chunks and the stitch, chunk-vs-chunk joint distance over the overlap with the
    blend window, per-joint p95 in the window vs the gate limits, and the window scan (utilisation vs centre)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = res["_"]
    L, seams, J, out, kmin, hz = d["L"], d["seams"], d["J"], d["out"], d["own"]["kmin"], res["grid"]["hz"]
    rows = [s for s in seams if s.get("lo") is not None]
    th = res["thresholds"]
    ink, ink2 = "#0b0b0b", "#52514e"
    fig = plt.figure(figsize=(23, 1.6 + 3.4 * max(len(rows), 1)), dpi=70, facecolor="#fcfcfb")
    if not rows:
        fig.text(0.5, 0.5, "no seam with a blend window", ha="center")
    gs = fig.add_gridspec(max(len(rows), 1), 5, hspace=0.62, wspace=0.34, width_ratios=[1, 1, 1.25, 1.25, 1.15],
                          top=0.955, bottom=0.045, left=0.045, right=0.99)
    rec = {s["name"]: s for s in res["seams"]}
    for q, sm in enumerate(rows):
        P, N = L[sm["P"]], L[sm["N"]]
        ov0, ov1 = sm["overlap"]
        lo, hi = sm["lo"], sm["hi"]
        pad = 12
        g0, g1 = max(ov0 - pad, kmin), min(ov1 + pad, d["own"]["kmax"] + 1)
        verdict = rec.get(sm["name"], {}).get("gate", {}).get("passed", True)
        for c, r in enumerate(ROLES):
            ax = fig.add_subplot(gs[q, c])
            _style(ax)
            ks = np.arange(g0, g1)
            ctr = out[r]["joints"][g0 - kmin:g1 - kmin, 0].mean(0)
            for ci, ls, col, lab in ((sm["P"], "--", "#52514e", P.name), (sm["N"], ":", "#8a8984", N.name)):
                cc = L[ci]
                sel = (ks >= cc.k0) & (ks <= cc.k1)
                p = (J[r][ci][ks[sel] - cc.k0, 0] - ctr) * 100
                ax.plot(p[:, 0], -p[:, 2], ls, color=col, lw=1.6, label=f"chunk {lab}")
            p = (out[r]["joints"][g0 - kmin:g1 - kmin, 0] - ctr) * 100
            ax.plot(p[:, 0], -p[:, 2], color=ROLE_COLOR[r], lw=2.0, label="stitched")
            pw = (out[r]["joints"][lo - kmin:hi - kmin, 0] - ctr) * 100
            ax.plot(pw[[0, -1], 0], -pw[[0, -1], 2], "o", ms=6, color=ROLE_COLOR[r], mec="#fcfcfb", mew=1.5)
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_xlabel("x (cm)", color=ink2, fontsize=8)
            ax.set_ylabel("-z (cm)", color=ink2, fontsize=8)
            ax.set_title(f"{sm['name']} {r}: pelvis from above (dots = window)", loc="left", fontsize=9, color=ink)
            if q == 0 and c == 0:
                ax.legend(fontsize=7, frameon=False, loc="best")
        ax = fig.add_subplot(gs[q, 2])
        _style(ax)
        ta = (d["grid_t"][ov0 - kmin:ov1 - kmin])
        ax.axvspan(d["grid_t"][lo - kmin], d["grid_t"][hi - 1 - kmin], color="#efeee9", zorder=0)
        for r in ROLES:
            jp = J[r][sm["P"]][ov0 - P.k0:ov1 - P.k0]
            jn = J[r][sm["N"]][ov0 - N.k0:ov1 - N.k0]
            dd, _ = disagreement(jp, jn)
            ax.plot(ta, dd[:, :22].max(1), color=ROLE_COLOR[r], lw=1.8, label=f"{r} worst body joint")
            ax.plot(ta, np.median(dd, 1), "--", color=ROLE_COLOR[r], lw=1.2, label=f"{r} median joint")
        ax.axhline(th["body_p95"], color="#b5b4ae", lw=1, ls=":")
        ax.axvline(sm["centre_t"], color=ink, lw=1)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("take time (s)", color=ink2, fontsize=8)
        ax.set_ylabel("chunk vs chunk (mm)", color=ink2, fontsize=8)
        ax.set_title(f"{sm['name']}: overlap {ov1 - ov0} fr, window {sm['window_t'][0]:.2f}-{sm['window_t'][1]:.2f} s "
                     f"(centre line)", loc="left", fontsize=9, color=ink if verdict else "#d03b3b")
        if q == 0:
            ax.legend(fontsize=7, frameon=False, loc="best")
        ax = fig.add_subplot(gs[q, 3])
        _style(ax)
        x = np.arange(22)
        for c, r in enumerate(ROLES):
            pj = rec.get(sm["name"], {}).get("roles", {}).get(r, {}).get("per_joint_p95", {})
            vals = [pj.get(n, 0.0) for n in JOINT_NAMES]
            ax.bar(x + (c - 0.5) * 0.38, vals, width=0.38, color=ROLE_COLOR[r], label=r)
        ax.hlines(th["body_p95"], -0.5, 17.5, color="#d03b3b", lw=1.3)
        ax.hlines(th["elbow_wrist_p95"], 17.5, 21.5, color="#d03b3b", lw=1.3)
        ax.set_xticks(x)
        ax.set_xticklabels(JOINT_NAMES, rotation=90, fontsize=6)
        ax.set_ylabel("p95 in window (mm)", color=ink2, fontsize=8)
        ax.set_title(f"{sm['name']}: per-joint p95 vs gate (red lines)", loc="left", fontsize=9, color=ink)
        ax = fig.add_subplot(gs[q, 4])
        _style(ax)
        scan = sm["scan"]
        for r in ROLES:
            ax.plot(scan["centre_t"], scan["utilisation"][r], color=ROLE_COLOR[r], lw=1.4, label=r)
        ax.plot(scan["centre_t"], scan["worst"], color=ink, lw=2, label="worst of both")
        ax.axhline(1.0, color="#d03b3b", lw=1.2)
        ax.axvline(sm["centre_t"], color=ink, lw=1, ls="--")
        ax.set_xlabel("window centre (s)", color=ink2, fontsize=8)
        ax.set_ylabel("gate utilisation (1 = limit)", color=ink2, fontsize=8)
        tb = sm["tie_break"]
        ax.set_title(f"{sm['name']}: window scan, {tb['n_tied']} tied ({tb['method']})", loc="left", fontsize=9, color=ink)
        if q == 0:
            ax.legend(fontsize=7, frameon=False, loc="best")
    ok = res["gate"]["passed"]
    fig.suptitle(f"{res['take']}: seams of {', '.join(res['chunk_names'])} - gate " + ("PASSED" if ok else "FAILED: " + "; ".join(
        res["gate"]["why"][:3])), x=0.01, y=0.995, ha="left", fontsize=12, color=ink if ok else "#d03b3b")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=70, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return path


def overview_sheet(res: dict, path: Path) -> Path:
    """The whole timeline: who owns each frame (blend windows marked), coverage tiles, cameras per dancer, reprojection,
    pelvis speed, quality flags."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    d = res["_"]
    L, out, seams, t, own = d["L"], d["out"], d["seams"], d["grid_t"], d["own"]
    ink, ink2 = "#0b0b0b", "#52514e"
    fig, axs = plt.subplots(6, 1, figsize=(22, 14), dpi=70, sharex=True, facecolor="#fcfcfb",
                            gridspec_kw={"height_ratios": [1.5, 0.45, 1.0, 1.0, 1.0, 0.7], "hspace": 0.35})
    for a in axs:
        _style(a)
    ax = axs[0]
    for i, c in enumerate(L):
        col = CHUNK_COLORS[i % len(CHUNK_COLORS)]
        t0, t1 = res["grid"]["anchor"] + c.k0 / res["grid"]["hz"], res["grid"]["anchor"] + c.k1 / res["grid"]["hz"]
        ax.plot([t0, t1], [i, i], color=col, lw=2, alpha=0.45, solid_capstyle="butt")
        s, e = own["owned"][i]
        ax.plot([t[s - own["kmin"]], t[max(e - 1, s) - own["kmin"]]], [i, i], color=col, lw=9, solid_capstyle="butt")
        ax.text(t0, i + 0.18, f" {c.name}", color=col, fontsize=9, va="bottom")
    for sm in seams:
        if sm.get("lo") is not None:
            ok = next((q["gate"]["passed"] for q in res["seams"] if q["name"] == sm["name"]), True)
            for a in axs:
                a.axvspan(t[sm["lo"] - own["kmin"]], t[sm["hi"] - 1 - own["kmin"]], color="#d8d7d1" if ok else "#f3c9c9", zorder=0)
    ax.set_yticks(range(len(L)))
    ax.set_yticklabels([c.name for c in L], fontsize=8)
    ax.set_ylim(-0.6, len(L) - 0.1)
    ax.set_title(f"thick = frames each chunk owns outright (bit-identical copy), thin = its whole fit, "
                 f"grey bands = {sum(1 for s in seams if s.get('lo') is not None)} blend windows", loc="left", fontsize=10, color=ink)
    ax = axs[1]
    code = {"C": 2, "P": 1, "N": 0}
    tiles = res["coverage"]["tiles"]
    arr = np.array([[code[c] for c in tiles]])
    ax.imshow(arr, aspect="auto", cmap=ListedColormap(["#d9d8d3", "#e8b04a", "#2f9e6e"]), vmin=0, vmax=2,
              extent=[res["coverage"]["t0"], res["coverage"]["t0"] + len(tiles), 0, 1], interpolation="nearest")
    ax.set_yticks([])
    cv = res["coverage"]["frames"]
    ax.set_title(f"coverage by second: complete {cv['complete'] / res['grid']['hz']:.0f} s (green), partial "
                 f"{cv['partial'] / res['grid']['hz']:.0f} s (amber), not covered {cv['none'] / res['grid']['hz']:.0f} s (grey)",
                 loc="left", fontsize=9, color=ink)
    axs[2].set_title("cameras seeing each dancer", loc="left", fontsize=9, color=ink)
    for r in ROLES:
        axs[2].step(t, np.where(out[r]["n_cams"] < 0, np.nan, out[r]["n_cams"]), where="mid", color=ROLE_COLOR[r],
                    lw=1.6, label=r, alpha=0.85)
    axs[2].legend(fontsize=8, frameon=False, loc="upper right")
    axs[3].set_title("median reprojection of the fit (px; blend windows: the worse chunk)", loc="left", fontsize=9, color=ink)
    for r in ROLES:
        if "reproj_px" in out[r]:
            axs[3].plot(t, np.clip(out[r]["reproj_px"], 0, 40), color=ROLE_COLOR[r], lw=1.0)
    axs[4].set_title("pelvis speed (m/s)", loc="left", fontsize=9, color=ink)
    for r in ROLES:
        axs[4].plot(t, pelvis_speed(out[r]["joints"], res["grid"]["hz"]), color=ROLE_COLOR[r], lw=1.0)
    ax = axs[5]
    cmap = ListedColormap(["#0b0b0b", "#e34948", "#e8b04a", "#2f9e6e"])
    for q, r in enumerate(ROLES):
        ax.imshow(out[r]["quality_flag"][None], aspect="auto", cmap=cmap, vmin=0, vmax=3, interpolation="nearest",
                  extent=[t[0], t[-1], q, q + 1])
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(list(ROLES), fontsize=8)
    ax.set_ylim(0, 2)
    ax.set_title("quality flag: black absent, red mono, amber two_view, green multi", loc="left", fontsize=9, color=ink)
    ax.set_xlabel("take time (s)", color=ink2, fontsize=9)
    ax.set_xlim(t[0], t[-1])
    ok = res["gate"]["passed"]
    fig.suptitle(f"{res['take']} stitch overview - seam gates " + ("PASSED" if ok else "FAILED"), x=0.01, y=0.99, ha="left",
                 fontsize=12, color=ink if ok else "#d03b3b")
    fig.subplots_adjust(top=0.94, left=0.05, right=0.99)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=70, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return path


# ------------------------------------------------------------------------------------------------ cli


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.stitch", description=__doc__.split("\n\n")[0])
    ap.add_argument("take", help="the dance / take name (takes/<take>.toml unless --chunks is given; names the output "
                                 "folder <work root>/<take> unless --out is given)")
    ap.add_argument("--chunks", help="chunk list JSON (see the module docstring)")
    ap.add_argument("--work-root", help="folder that relative paths (chunk list, --out) resolve against (default "
                                        "$DANCECAP_WORK_ROOT, else dancecap/work); on a pod e.g. the workspace's "
                                        "dancecap/work or /workspace/hm/...")
    ap.add_argument("--out", "--out-dir", dest="out", help="output folder holding smplx/, <refined>/, stitch.json, review/ "
                                                           "(default <work root>/<take>); relative = under the work root")
    ap.add_argument("--check", action="store_true",
                    help="analyse + gate; writes no motion (with --out: stitch_check.json + review sheets there)")
    ap.add_argument("--seam", help="analyse one seam, e.g. 'b|c' (implies --check)")
    ap.add_argument("--partial", action="store_true", help="stitch the chunks that exist; gaps are flagged absent")
    ap.add_argument("--no-refined", action="store_true")
    ap.add_argument("--no-sheets", action="store_true")
    ap.add_argument("--tag", help="fit tag of the segment takes (default: mamma/select.json, else the biggest set)")
    ap.add_argument("--force", action="store_true", help="promote a stitch whose gate failed into smplx/")
    ap.add_argument("--gate-exit", choices=("strict", "report"), default="strict",
                    help="strict: exit 2 when a gate fails; report: exit 0 (stitch.json carries the verdict, the refit "
                         "windows and the rejected outputs; a pod job uses this so the results are pulled)")
    ap.add_argument("--allow-mixed-world", action="store_true", help="stitch chunks of different pose worlds (flagged)")
    ap.add_argument("--select-weights", metavar="arm=W,fingers=W", help="T1: weigh the elbow / wrist and finger gates by W (< 1) when the blend window is CHOSEN, so "
                                                                         "the body and pelvis gates decide first; the gates themselves are unchanged")
    ap.add_argument("--require-world", action="store_true", help="refuse chunks that carry no world_id")
    ap.add_argument("--write-chunks", metavar="FILE", help="write the chunk list JSON of this take and exit")
    ap.add_argument("--json", help="also write the report here")
    a = ap.parse_args(argv)
    if a.work_root:
        os.environ["DANCECAP_WORK_ROOT"] = str(Path(a.work_root).resolve())
    spec = spec_from_json(a.chunks) if a.chunks else load_spec(a.take, a.tag)
    if a.chunks and a.take and a.take != spec.take:
        spec.take = a.take
        if not spec.work_set:
            spec.work = work_root() / a.take
    if a.out:
        spec.work = _wp(a.out)
        spec.work_set = True
    if a.select_weights:
        spec.select_weights = {k.strip(): float(v) for k, v in (kv.split("=") for kv in a.select_weights.split(",") if kv.strip())}
    spec.allow_mixed_world = spec.allow_mixed_world or a.allow_mixed_world
    spec.require_world = spec.require_world or a.require_world
    if a.write_chunks:
        Path(a.write_chunks).parent.mkdir(parents=True, exist_ok=True)
        Path(a.write_chunks).write_text(json.dumps(chunk_list(spec), indent=1), encoding="utf-8")
        print(f"[stitch] chunk list of {spec.take} ({len(spec.chunks)} chunks) -> {a.write_chunks}")
        return 0
    try:
        res = stitch(spec, check=a.check, only_seam=a.seam, partial=a.partial, want_refined=not a.no_refined,
                     sheets=not a.no_sheets, force=a.force,
                     out_dir=spec.work if (a.out or spec.work_set) and (a.check or a.seam) else None)
    except FileNotFoundError as e:
        print(f"[stitch] {e}")
        return 1
    except WorldMismatch as e:
        print(f"[stitch] REFUSED: {e}")
        return 3
    if a.json:
        Path(a.json).write_text(json.dumps({k: v for k, v in res.items() if k != "_"}, indent=1, default=_js), encoding="utf-8")
    for s in res["seams"]:
        for r, rr in s.get("roles", {}).items():
            c = rr["checks"]
            print(f"[stitch] {s['name']:>6} {r:6} body {c['body_p95_mm']:5.1f} ({c['body_joint']}) arm {c['elbow_wrist_p95_mm']:5.1f} "
                  f"({c['arm_joint']}) fingers {c['fingers_rel_wrist_p95_mm']:5.1f} pelvis {c['pelvis_median_mm']:5.1f} mm; "
                  f"smooth {rr['continuity']['smooth']}")
        if not s["gate"]["passed"] and "refit" in s:
            print(f"[stitch] {s['name']}: FAILED -> refit window k {s['refit']['window_k']} ({s['refit']['window_t'][0]:.2f}-"
                  f"{s['refit']['window_t'][1]:.2f} s)")
    ok = res["gate"]["passed"]
    print(f"[stitch] gate {'PASSED' if ok else 'FAILED: ' + '; '.join(res['gate']['why'])}")
    return 0 if ok or a.gate_exit == "report" else 2


if __name__ == "__main__":
    raise SystemExit(main())
