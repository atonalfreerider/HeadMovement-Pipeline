"""Timeline, cells and COVERAGE of a dance's final result (FARM.md sections 1 and 7).

The fused grid of a dance is 30 Hz on the lattice t_k = anchor + k/30 (larissa_kadu_full: anchor 30.151176844227862,
k = -998..4645, T = 5644, 188.1 s). A CELL is 1 s = 30 consecutive frames (cell i = frames k0+30i .. k0+30i+29, the last
cell may be shorter). Coverage is COMPUTED from the real outputs, never from job state:

    level of a frame   0 none | 1 front (cameras + 2D + sync + prepared inputs) | 2 fit (raw 3D) | 3 refined | 4 analysed
    + per frame:       n_cams, quality flag (absent | mono | two_view | multi), reprojection above the gate, a failed
                       seam, "final" (the frame is part of the stitched result or of the reused reference span)
    complete frame     final, level >= refined, flag multi, reprojection ok, no failed seam
    cell state         complete  >= 99 % of its frames are refined and >= 90 % are complete frames
                       none      no frame has any level
                       partial   everything between (front only, raw fit, awaiting stitch, mono / two_view, absent,
                                 a failed seam, reprojection over the gate, a cell cut by a span boundary)

Evidence read (every source optional): the reused reference take's smplx / refined dirs (a complete span), the
stitched result work/<dance>/{smplx, <refined>}/*.npz (+ quality_flag / n_cams / segment_owner) and work/<dance>/stitch.json
(per-seam gates), the segment takes' and pulled chunk takes' (work/<dance>_<id>/) mamma/prepare.json, fits and refined dirs,
the analysis outputs (physics/timing) and, optionally, another workflow's
coverage_cells.json ({cells: [{t0, t1, state, reasons}]}), which can only raise a cell.

    python -m dancecap.farm coverage [--png] [--ascii]
"""
from __future__ import annotations

import json
import math
import os
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import DEFAULT_DANCE, coverage_path, farm_dir, work_dir
from .state import atomic_write_json, now_iso, read_json
from ..take import PACKAGE_ROOT

HZ = 30.0
CELL_S = 1.0
ROLES = ("lead", "follow")
LEVELS = ("none", "front", "fit", "refined", "analysed")
L_NONE, L_FRONT, L_FIT, L_REFINED, L_ANALYSED = range(5)
FLAG_UNKNOWN, FLAG_ABSENT, FLAG_MONO, FLAG_TWO, FLAG_MULTI = -1, 0, 1, 2, 3
FLAG_NAMES = {FLAG_UNKNOWN: "unknown", FLAG_ABSENT: "absent", FLAG_MONO: "mono", FLAG_TWO: "two_view",
              FLAG_MULTI: "multi"}
_FLAG_BY_NAME = {"absent": 0, "none": 0, "mono": 1, "monocular": 1, "two_view": 2, "twoview": 2, "two-view": 2,
                 "low_view": 2, "multi": 3, "multiview": 3, "multi_view": 3}
STATES = ("none", "partial", "complete")
REPROJ_FACTOR = 3.0      # a frame's reprojection is "over the gate" above this x the span's median ...
REPROJ_MIN_PX = 25.0     # ... and above this absolute value (38 s take: median 5.5-7.3 px, p99 9.5-21 px)
COMPLETE_HAVE = 0.99     # cell: fraction of frames at level >= refined
COMPLETE_GOOD = 0.90     # cell: fraction of complete frames
ON_LATTICE_TOL_S = 2e-3  # a motion whose t is further than this from the lattice is not on the fused grid
REFINED_DIR = "smplx_rf2_noslip"


# ------------------------------------------------------------------------------------------------------------ grid


@dataclass(frozen=True)
class Grid:
    anchor: float
    k0: int
    k1: int
    hz: float = HZ

    @property
    def T(self) -> int:
        return self.k1 - self.k0 + 1

    @property
    def cell_frames(self) -> int:
        return int(round(self.hz * CELL_S))

    @property
    def n_cells(self) -> int:
        return -(-self.T // self.cell_frames)

    @property
    def t_start(self) -> float:
        return self.t(self.k0)

    @property
    def t_end(self) -> float:
        """End of the last frame."""
        return self.t(self.k1 + 1)

    @property
    def seconds(self) -> float:
        return self.T / self.hz

    def t(self, k):
        return self.anchor + k / self.hz

    def k_of(self, t: float) -> int:
        return int(round((t - self.anchor) * self.hz))

    def k_floor(self, t: float, eps: float = 1e-6) -> int:
        return int(math.floor((t - self.anchor) * self.hz + eps))

    def k_ceil(self, t: float, eps: float = 1e-6) -> int:
        return int(math.ceil((t - self.anchor) * self.hz - eps))

    def idx(self, k):
        return k - self.k0

    def cell_of_k(self, k):
        return (k - self.k0) // self.cell_frames

    def cell_k_range(self, i: int) -> tuple[int, int]:
        a = self.k0 + i * self.cell_frames
        return a, min(self.k1, a + self.cell_frames - 1)

    def cell_t_range(self, i: int) -> tuple[float, float]:
        a, b = self.cell_k_range(i)
        return float(self.t(a)), float(self.t(b + 1))

    def to_dict(self) -> dict:
        return {"anchor": self.anchor, "k0": self.k0, "k1": self.k1, "hz": self.hz}

    @classmethod
    def from_dict(cls, d: dict) -> "Grid":
        return cls(float(d["anchor"]), int(d["k0"]), int(d["k1"]), float(d.get("hz", HZ)))


@dataclass
class DanceSpec:
    """What the farm needs to know about a dance, read from takes/<dance>.toml (see [reuse], [segments], [timeline])."""
    name: str
    grid: Grid
    reference: str | None = None            # reused take (larissa_kadu_38): its span is complete
    ref_k: tuple[int, int] | None = None    # that span on the grid
    cams: list[str] = field(default_factory=list)
    segments: dict[str, dict] = field(default_factory=dict)   # id -> {fit_k, front, cams}
    refined_dir: str = REFINED_DIR
    cfg: dict = field(default_factory=dict, repr=False)

    @property
    def work(self) -> Path:
        return work_dir(self.name)

    def segment_take(self, sid: str) -> str:
        return f"{self.name}_{sid}"

    def segment_work(self, sid: str) -> Path:
        return work_dir(self.segment_take(sid))

    @classmethod
    def load(cls, name: str = DEFAULT_DANCE) -> "DanceSpec":
        p = PACKAGE_ROOT / "takes" / f"{name}.toml"
        cfg = tomllib.loads(p.read_text(encoding="utf-8"))
        tl, reuse = cfg.get("timeline") or {}, cfg.get("reuse") or {}
        hz = float(tl.get("hz", HZ))
        ms = reuse.get("motion_span")
        anchor = float(tl["anchor"]) if "anchor" in tl else float(ms[0] if ms else cfg["window"][0])
        w0, w1 = cfg["window"]
        if "k" in tl:
            k0, k1 = (int(x) for x in tl["k"])
        else:
            k0, k1 = int(math.ceil((w0 - anchor) * hz - 1e-6)), int(math.floor((w1 - anchor) * hz + 1e-6))
        ref_k = (int(round((ms[0] - anchor) * hz)), int(round((ms[1] - anchor) * hz))) if ms else None
        segs = {sid: {"fit_k": [int(x) for x in s["fit_k"]], "front": [float(x) for x in s.get("front", [])],
                      "cams": list(s.get("cams", []))} for sid, s in (cfg.get("segments") or {}).items()}
        cams = _dance_cams(cfg, segs)
        return cls(name=name, grid=Grid(anchor, k0, k1, hz), reference=reuse.get("take"), ref_k=ref_k, cams=cams,
                   segments=segs, refined_dir=(cfg.get("export") or {}).get("smplx_dir") or REFINED_DIR, cfg=cfg)


def _dance_cams(cfg: dict, segs: dict) -> list[str]:
    try:
        from ..take import Take
        return [s.id for s in Take.load(cfg["name"]).sources]
    except Exception:  # noqa: BLE001  (no alignment file on this machine)
        return sorted({c for s in segs.values() for c in s.get("cams", [])})


# ------------------------------------------------------------------------------------------------ camera spans


def camera_spans(spec: DanceSpec, refresh: bool = False) -> dict[str, tuple[float, float]]:
    """{cam: (t_first, t_last)} on the reference clock: the span in which each phone records. Cached in
    farm/cameras.json; read from front/plan.json when present, else ffprobe via front_full.source_facts."""
    cache = farm_dir(spec.name) / "cameras.json"
    if not refresh:
        d = read_json(cache, None)
        if isinstance(d, dict) and d.get("spans"):
            return {c: (float(a), float(b)) for c, (a, b) in d["spans"].items()}
        plan = read_json(spec.work / "front" / "plan.json", None)
        if isinstance(plan, dict) and plan.get("sources"):
            sp = {c: (float(v["t_first"]), float(v["t_last"])) for c, v in plan["sources"].items()}
            atomic_write_json(cache, {"spans": sp, "source": "front/plan.json", "created": now_iso()})
            return sp
    from ..front_full import source_facts   # ffprobe on the source MP4s (slow: cached afterwards)
    from ..take import Take
    facts = source_facts(Take.load(spec.name))
    sp = {c: (float(v["t_first"]), float(v["t_last"])) for c, v in facts.items()}
    atomic_write_json(cache, {"spans": sp, "source": "ffprobe", "created": now_iso()})
    return sp


def cams_recording(grid: Grid, spans: dict[str, tuple[float, float]]) -> np.ndarray:
    """(T,) number of phones recording at each grid frame."""
    t = grid.t(np.arange(grid.k0, grid.k1 + 1))
    n = np.zeros(grid.T, np.int16)
    for a, b in spans.values():
        n += ((t >= a) & (t <= b)).astype(np.int16)
    return n


# --------------------------------------------------------------------------------------------- frame evidence


def decode_flag(a) -> np.ndarray:
    """quality_flag array (strings or small ints) -> codes (0 absent, 1 mono, 2 two_view, 3 multi, -1 unknown)."""
    a = np.asarray(a)
    if a.dtype.kind in "US" or a.dtype == object:
        return np.array([_FLAG_BY_NAME.get(str(x).strip().lower(), FLAG_UNKNOWN) for x in a], np.int8)
    return np.where((a >= 0) & (a <= 3), a, FLAG_UNKNOWN).astype(np.int8)


def flag_from_ncams(n) -> np.ndarray:
    n = np.asarray(n)
    return np.where(n <= 0, FLAG_UNKNOWN, np.where(n == 1, FLAG_MONO, np.where(n == 2, FLAG_TWO, FLAG_MULTI))).astype(np.int8)


class Frames:
    """Per-frame arrays over the grid, filled by the evidence collectors (later calls win where their level is >=)."""

    def __init__(self, grid: Grid):
        T = grid.T
        self.grid = grid
        self.level = np.zeros(T, np.int8)         # best level seen from any source (final or not)
        self.final_level = np.zeros(T, np.int8)   # best level that is part of the stitched / reused result
        self.n_cams = np.zeros(T, np.int16)
        self.flag = np.full(T, FLAG_UNKNOWN, np.int8)
        self.bad = np.zeros(T, bool)              # reprojection over the gate
        self.seam_fail = np.zeros(T, bool)
        self.max_cams = np.zeros(T, np.int16)     # phones recording
        self.owner = np.full(T, -1, np.int16)
        self.owners: list[str] = []
        self.seam_names: dict[int, str] = {}
        self.sources: list[dict] = []

    def owner_id(self, name: str) -> int:
        if name not in self.owners:
            self.owners.append(name)
        return self.owners.index(name)

    @property
    def final(self) -> np.ndarray:
        return self.final_level > 0

    def apply(self, k, level: int, *, n_cams=None, flag=None, bad=None, final: bool = False,
              owner: str | None = None) -> int:
        """Raise the level of the grid frames `k` (int array) to `level`; the metadata (n_cams, flag, bad, owner) of this
        source replaces what is there wherever the source is at least as good. n_cams / flag / bad are scalars or arrays
        aligned with k. final=True: these frames are part of the stitched result. Returns the number of frames touched."""
        g = self.grid
        k = np.asarray(k, int)
        inside = (k >= g.k0) & (k <= g.k1)
        if not inside.any():
            return 0

        def sel(a):
            if a is None:
                return None
            a = np.asarray(a)
            return a if a.ndim == 0 else a[inside]
        n_cams, flag, bad = sel(n_cams), sel(flag), sel(bad)
        i = k[inside] - g.k0
        win = level >= self.level[i]

        def w(a):
            return None if a is None else (a if np.ndim(a) == 0 else a[win])
        iw = i[win]
        self.level[i] = np.maximum(self.level[i], level)
        if final:
            self.final_level[i] = np.maximum(self.final_level[i], level)
        if n_cams is not None:
            self.n_cams[iw] = w(n_cams)
        if flag is not None:
            f = w(flag)
            self.flag[iw] = np.where(f == FLAG_UNKNOWN, self.flag[iw], f)
        if bad is not None:
            self.bad[iw] = w(bad)
        if owner is not None:
            self.owner[iw] = self.owner_id(owner)
        return int(len(i))

    def absorb(self, other: "Frames", final: bool = False) -> int:
        """Merge another Frames over the SAME grid (a chunk's evidence file): levels take the maximum, the metadata of the
        better source wins, seam failures / bad frames are OR-ed. final=False keeps `other`'s final_level out (a chunk's own
        evidence is not part of the stitched result; only the stitched dance's evidence is)."""
        if other.grid != self.grid:
            raise ValueError(f"evidence grid {other.grid} != {self.grid}")
        n = 0
        for lv in np.unique(other.level[other.level > 0]):
            m = other.level == lv
            k = self.grid.k0 + np.flatnonzero(m)
            n += self.apply(k, int(lv), n_cams=other.n_cams[m], flag=other.flag[m], bad=other.bad[m])
        self.seam_fail |= other.seam_fail
        self.max_cams = np.maximum(self.max_cams, other.max_cams)
        if final:
            self.final_level = np.maximum(self.final_level, other.final_level)
            self.level = np.maximum(self.level, other.final_level)
        for name in other.owners:
            self.owner_id(name)
        self.seam_names.update(other.seam_names)
        return n

    def to_npz(self, path: Path) -> Path:
        """The per-frame evidence as a small compressed file (a few KB per 190 s): what a pod writes next to a chunk's or the
        stitched result's outputs so that coverage can be computed locally without pulling the motion files."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
        np.savez_compressed(tmp, level=self.level, final_level=self.final_level, flag=self.flag, n_cams=self.n_cams,
                            max_cams=self.max_cams, bad=self.bad, seam_fail=self.seam_fail, owner=self.owner,
                            owners=np.array(self.owners, dtype=object).astype(str),
                            seam_names=np.array(json.dumps({str(k): v for k, v in self.seam_names.items()})),
                            grid=np.array(json.dumps(self.grid.to_dict())), written=np.array(now_iso()))
        _replace(tmp, path)
        return path

    @classmethod
    def from_npz(cls, path: Path, grid: Grid | None = None) -> "Frames":
        z = np.load(path, allow_pickle=False)
        g = Grid.from_dict(json.loads(str(z["grid"])))
        if grid is not None and g != grid:
            raise ValueError(f"{path}: evidence grid {g} != {grid}")
        fr = cls(g)
        for k in ("level", "final_level", "flag", "n_cams", "max_cams", "bad", "seam_fail", "owner"):
            setattr(fr, k, z[k].copy())
        fr.owners = [str(x) for x in z["owners"]]
        fr.seam_names = {int(k): v for k, v in json.loads(str(z["seam_names"])).items()}
        return fr

    # derived -----------------------------------------------------------------------------------------------------
    def ok_flag(self) -> np.ndarray:
        """flag is acceptable (multi, or unknown = nothing says otherwise)."""
        return (self.flag == FLAG_MULTI) | (self.flag == FLAG_UNKNOWN)

    def full(self) -> np.ndarray:
        """Complete frames: part of the result, refined, multi-view, reprojection and seams fine."""
        return (self.final_level >= L_REFINED) & self.ok_flag() & ~self.bad & ~self.seam_fail

    def unfixable(self) -> np.ndarray:
        """Refined frames that are not complete only because the cameras cannot do better (absent / mono / two_view
        with every recording phone already used): re-running them changes nothing."""
        limited = (self.flag >= FLAG_ABSENT) & (self.flag <= FLAG_TWO)
        cam_bound = (self.flag == FLAG_ABSENT) | ((self.max_cams > 0) & (self.n_cams >= self.max_cams))
        return (self.level >= L_REFINED) & limited & cam_bound & ~self.seam_fail


def _lattice_ks(t: np.ndarray, grid: Grid) -> tuple[np.ndarray, bool]:
    ks = np.rint((t - grid.anchor) * grid.hz).astype(int)
    err = float(np.abs(t - grid.t(ks)).max()) if len(t) else 0.0
    return ks, err <= ON_LATTICE_TOL_S


def load_motion(d: Path, grid: Grid) -> dict | None:
    """The two dancers' npz in a smplx-contract directory -> {k, reproj, flag, n_cams, owner} on the common frames,
    or None (missing / not on the fused lattice). Only the small per-frame arrays are read."""
    recs = {}
    for role in ROLES:
        f = Path(d) / f"{role}.npz"
        if not f.exists():
            return None
        try:
            z = np.load(f, allow_pickle=False)
            rec = {"k": None}
            ks, ok = _lattice_ks(np.asarray(z["t"], float), grid)
            if not ok:
                return None
            rec["k"] = ks
            for key in ("reproj_px", "quality_flag", "n_cams", "segment_owner", "repaired"):
                if key in z.files:
                    rec[key] = z[key]
            recs[role] = rec
        except (OSError, ValueError, KeyError):
            return None
    kl, kf = recs["lead"]["k"], recs["follow"]["k"]
    common = np.intersect1d(kl, kf)
    if len(common) == 0:
        return None
    ol, of = np.argsort(kl), np.argsort(kf)
    il, jf = ol[np.searchsorted(kl[ol], common)], of[np.searchsorted(kf[of], common)]
    out = {"k": common, "dir": str(d)}
    rp = []
    for role, idx in (("lead", il), ("follow", jf)):
        r = recs[role]
        if "reproj_px" in r:
            v = np.asarray(r["reproj_px"], float)[idx]
            fin = v[np.isfinite(v)]
            med = float(np.median(fin)) if len(fin) else 0.0
            b = ~np.isfinite(v) | (v > max(REPROJ_FACTOR * med, REPROJ_MIN_PX))
            if "repaired" in r:   # frames the fit's own repair step already interpolated are handled, not failures
                b &= ~np.asarray(r["repaired"], bool)[idx]
            rp.append(b)
    out["bad"] = np.any(rp, axis=0) if rp else None
    fl = [decode_flag(recs[r]["quality_flag"])[idx] for r, idx in (("lead", il), ("follow", jf)) if "quality_flag" in recs[r]]
    if fl:   # the worse known flag of the two dancers (absent < mono < two_view < multi); unknown only if both are
        st = np.where(np.asarray(fl) == FLAG_UNKNOWN, 99, np.asarray(fl)).min(axis=0)
        out["flag"] = np.where(st == 99, FLAG_UNKNOWN, st).astype(np.int8)
    nc = [np.asarray(recs[r]["n_cams"])[idx] for r, idx in (("lead", il), ("follow", jf)) if "n_cams" in recs[r]]
    if nc:
        out["n_cams"] = np.min(nc, axis=0).astype(np.int16)
    for r, idx in (("follow", jf), ("lead", il)):
        if "segment_owner" in recs[r]:
            so = np.asarray(recs[r]["segment_owner"])[idx]
            out["owner"] = np.array([str(x) for x in so]) if so.dtype.kind in "USO" else so
            break
    return out


def prepared_ncams(prep: dict, grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    """(k array, n_cams array) of a mamma/prepare.json: the prepared grid frames and how many of the fit's cameras
    have frames at each (cams_per_frame = each camera's first/last sync-corrected time)."""
    g = prep["grid"]
    k0 = grid.k_of(float(g["t0"]))
    ks = np.arange(k0, k0 + int(g["T"]))
    t = grid.t(ks)
    cams = set()
    for s in (prep.get("sets") or {}).values():
        cams |= set(s.get("cams", []))
    cpf = (g.get("trim") or {}).get("cams_per_frame")
    if not cpf:
        return ks, np.full(len(ks), len(cams), np.int16)
    n = np.zeros(len(ks), np.int16)
    for c, (a, b) in cpf.items():
        if not cams or c in cams:
            n += ((t >= a - 1e-9) & (t <= b + 1e-9)).astype(np.int16)
    return ks, n


def parse_stitch(doc: dict, grid: Grid) -> list[dict]:
    """stitch.json -> [{name, k0, k1, ok, why}] seam windows. Tolerant of the exact layout (the stitcher is another
    workflow's): `seams` as a list or a dict; the window as blend_s / blend / window_s / overlap_s ([t0, t1] seconds) or
    blend_k / window_k / overlap_k ([k0, k1]); the verdict as ok / passed / pass / gate_ok / status, or a `gates`
    dict whose values are all passed; `repaired` true turns a failed seam into an ok one."""
    seams = doc.get("seams") if isinstance(doc, dict) else None
    if isinstance(seams, dict):
        seams = [{"name": k, **(v if isinstance(v, dict) else {"ok": bool(v)})} for k, v in seams.items()]
    out = []
    for i, s in enumerate(seams or []):
        if not isinstance(s, dict):
            continue
        name = str(s.get("name") or s.get("seam") or s.get("id") or s.get("between") or f"seam{i}")
        win = None
        for key in ("blend_k", "window_k", "overlap_k", "k"):
            v = s.get(key)
            if isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
                win = (int(v[0]), int(v[1]) - (1 if key == "window_k" else 0))   # dancecap.stitch: window_k = [k0, k1)
                break
        if win is None:
            for key in ("blend_s", "blend", "window_s", "overlap_s", "blend_t", "window"):
                v = s.get(key)
                if isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
                    win = (grid.k_floor(float(v[0])), grid.k_ceil(float(v[1])))
                    break
        if win is None and isinstance(s.get("centre_s", s.get("center_s")), (int, float)):
            c = grid.k_of(float(s.get("centre_s", s.get("center_s"))))
            win = (c - int(grid.hz), c + int(grid.hz))
        verdict = None
        g = s.get("gate")
        if isinstance(g, dict) and isinstance(g.get("passed"), bool):   # dancecap.stitch: {"gate": {"passed", "why"}}
            verdict = g["passed"]
        for key in (() if verdict is not None else ("ok", "passed", "pass", "gate_ok", "gates_passed")):
            if isinstance(s.get(key), bool):
                verdict = s[key]
                break
        if verdict is None and isinstance(s.get("status"), str) and s["status"].lower() != "ok":
            verdict = s["status"].lower() in ("pass", "passed", "repaired")   # 'ok' = the seam was computed, not the verdict
        if verdict is None and isinstance(s.get("gates"), dict) and s["gates"]:
            vals = [v if isinstance(v, bool) else (v.get("ok", v.get("passed")) if isinstance(v, dict) else None)
                    for v in s["gates"].values()]
            vals = [v for v in vals if isinstance(v, bool)]
            verdict = all(vals) if vals else None
        if s.get("repaired") is True and verdict is False:
            verdict = True
        if win is None:
            continue
        out.append({"name": name, "k0": win[0], "k1": win[1], "ok": True if verdict is None else bool(verdict),
                    "why": "" if verdict is not False else "seam gate failed"})
    return out


# -------------------------------------------------------------------------------------- the evidence collectors


def _note(fr: Frames, kind: str, path, n: int, **extra) -> None:
    fr.sources.append({"kind": kind, "path": str(path), "frames": int(n), **extra})


def collect_cameras(spec: DanceSpec, fr: Frames) -> None:
    try:
        sp = camera_spans(spec)
    except Exception as e:  # noqa: BLE001  (no ffprobe / sources on this machine)
        _note(fr, "cameras", "-", 0, error=f"{type(e).__name__}: {e}"[:200])
        return
    fr.max_cams = cams_recording(fr.grid, sp)
    _note(fr, "cameras", farm_dir(spec.name) / "cameras.json", fr.grid.T, cams=len(sp))


def _collect_take(spec: DanceSpec, fr: Frames, take_work: Path, owner: str, *, final: bool, refined_dir: str,
                  prepared_from: dict | None = None) -> None:
    """One take's (or segment's, or chunk's) dirs: prepared inputs (front), raw fits, refined fit."""
    g = fr.grid
    nc_k = nc = None
    prep = read_json(take_work / "mamma" / "prepare.json", None)
    if isinstance(prep, dict) and prep.get("grid"):
        nc_k, nc = prepared_ncams(prep, g)
        fr.apply(nc_k, L_FRONT, n_cams=nc, flag=flag_from_ncams(nc), owner=owner)
        _note(fr, "prepared", take_work / "mamma" / "prepare.json", len(nc_k), take=owner)
    # raw fits: fits/<tag>/smplx (full fits only) and the promoted smplx/
    cands = []
    for fj in sorted((take_work / "mamma" / "fits").glob("*/fit.json")):
        rep = read_json(fj, {}) or {}
        if not rep.get("partial"):
            cands.append(fj.parent / "smplx")
    cands.append(take_work / "smplx")
    for d in cands:
        m = load_motion(d, g)
        if m is None:
            continue
        ncm = _ncams_for(m, nc_k, nc)
        fl = m.get("flag") if m.get("flag") is not None else flag_from_ncams(ncm) if ncm is not None else None
        fr.apply(m["k"], L_FIT, n_cams=ncm, flag=fl, bad=m.get("bad"), final=final, owner=owner)
        _note(fr, "fit", d, len(m["k"]), take=owner)
    m = load_motion(take_work / refined_dir, g)
    if m is not None:
        ncm = _ncams_for(m, nc_k, nc)
        fl = m.get("flag") if m.get("flag") is not None else flag_from_ncams(ncm) if ncm is not None else None
        fr.apply(m["k"], L_REFINED, n_cams=ncm, flag=fl, bad=m.get("bad"), final=final, owner=owner)
        _note(fr, "refined", take_work / refined_dir, len(m["k"]), take=owner)


def _ncams_for(m: dict, nc_k, nc):
    if m.get("n_cams") is not None:
        return m["n_cams"]
    if nc_k is None:
        return None
    pos = np.searchsorted(nc_k, m["k"])
    pos = np.clip(pos, 0, len(nc_k) - 1)
    return np.where(nc_k[pos] == m["k"], nc[pos], 0).astype(np.int16)


def collect_segments(spec: DanceSpec, fr: Frames) -> None:
    for sid in spec.segments:
        w = spec.segment_work(sid)
        if w.exists():
            _collect_take(spec, fr, w, sid, final=False, refined_dir=spec.refined_dir)


def collect_chunks(spec: DanceSpec, fr: Frames) -> None:
    """Pulled chunk outputs. A chunk's take is a work dir like the S1 segments' (work/<dance>_<chunk id>/, the worker's
    declared outputs are extracted at the same workspace paths): prepared inputs, fits, refined motion."""
    base = work_dir(spec.name).parent
    seg_takes = {spec.segment_take(s) for s in spec.segments}
    for d in sorted(base.glob(f"{spec.name}_*")):
        if d.is_dir() and d.name not in seg_takes:
            _collect_take(spec, fr, d, d.name[len(spec.name) + 1:], final=False, refined_dir=spec.refined_dir)


def collect_reference(spec: DanceSpec, fr: Frames) -> None:
    """The reused reference take is a complete span of the final result (its motion is copied verbatim by the stitch)."""
    if not spec.reference:
        return
    w = work_dir(spec.reference)
    _collect_take(spec, fr, w, spec.reference, final=True, refined_dir=spec.refined_dir)


def collect_stitched(spec: DanceSpec, fr: Frames) -> None:
    w = spec.work
    for sub, lvl in (("smplx", L_FIT), (spec.refined_dir, L_REFINED)):
        m = load_motion(w / sub, fr.grid)
        if m is None:
            continue
        ncm = m.get("n_cams")
        fl = m.get("flag") if m.get("flag") is not None else flag_from_ncams(ncm) if ncm is not None else None
        own = m.get("owner")
        fr.apply(m["k"], lvl, n_cams=ncm, flag=fl, bad=m.get("bad"), final=True)
        if own is not None:
            i = m["k"] - fr.grid.k0
            ok = (i >= 0) & (i < fr.grid.T)
            fr.owner[i[ok]] = [fr.owner_id(str(x)) for x in own[ok]]
        _note(fr, "stitched", w / sub, len(m["k"]))
    sj = read_json(w / "stitch.json", None)
    if isinstance(sj, dict):
        seams = parse_stitch(sj, fr.grid)
        for s in seams:
            if not s["ok"]:
                a, b = max(s["k0"], fr.grid.k0), min(s["k1"], fr.grid.k1)
                if a <= b:
                    fr.seam_fail[a - fr.grid.k0:b - fr.grid.k0 + 1] = True
                    for k in range(a, b + 1):
                        fr.seam_names[k] = s["name"]
        _note(fr, "stitch.json", w / "stitch.json", sum(s["k1"] - s["k0"] + 1 for s in seams), seams=len(seams),
              failed=[s["name"] for s in seams if not s["ok"]])


def collect_analysis(spec: DanceSpec, fr: Frames) -> None:
    """physics + timing present for a span => analysed (needs the frame to be refined already)."""
    takes = [spec.name] + ([spec.reference] if spec.reference else [])
    for tk in takes:
        w = work_dir(tk)
        pz, tj = w / "physics" / "physics.npz", w / "timing" / "timing.json"
        if not (pz.exists() and tj.exists()):
            continue
        try:
            t = np.asarray(np.load(pz, allow_pickle=False)["t"], float)
        except (OSError, ValueError, KeyError):
            continue
        ks, ok = _lattice_ks(t, fr.grid)
        if not ok:
            continue
        ks = ks[(ks >= fr.grid.k0) & (ks <= fr.grid.k1)]
        i = ks - fr.grid.k0
        sel = fr.level[i] >= L_REFINED
        fr.level[i[sel]] = L_ANALYSED
        fsel = fr.final_level[i] >= L_REFINED
        fr.final_level[i[fsel]] = L_ANALYSED
        _note(fr, "analysis", pz, int(sel.sum()), take=tk)


EVIDENCE = "farm/evidence.npz"   # work/<take>/farm/evidence.npz: written next to a chunk's / the result's outputs on the pod


def evidence_files(spec: DanceSpec) -> list[tuple[str, Path, bool]]:
    """(owner, path, final) of every evidence file on this machine: pulled chunk takes (work/<dance>_<id>/farm/evidence.npz,
    not final), the stitched dance (work/<dance>/farm/evidence.npz, final: written by the stitch job on a pod)."""
    out = []
    for p in sorted((PACKAGE_ROOT / "work").glob(f"{spec.name}_*/{EVIDENCE}")):
        out.append((p.parents[1].name[len(spec.name) + 1:], p, False))
    q = spec.work / EVIDENCE
    if q.exists():
        out.append((spec.name, q, True))
    return out


def collect_evidence(spec: DanceSpec, fr: Frames) -> None:
    """Absorb the small per-frame evidence files that pods wrote (FARM.md 0.10: analysis runs on pods, the PC only reads the
    small results). A corrupt / foreign-grid file is skipped and noted."""
    for owner, p, final in evidence_files(spec):
        try:
            ev = Frames.from_npz(p, fr.grid)
            fr.absorb(ev, final=final)
            _note(fr, "evidence", p, int((ev.level > 0).sum()), owner=owner, final=final)
        except (OSError, ValueError, KeyError) as e:
            _note(fr, "evidence", p, 0, error=f"{type(e).__name__}: {e}"[:200])


def build_evidence(spec: DanceSpec, take_work: Path, owner: str, *, stitched: bool = False) -> Frames:
    """The evidence of ONE take dir (pod side, `chunk-evidence`): a chunk take (prepared / fit / refined) or, with
    stitched=True, the assembled dance (stitched motion + stitch.json seams + analysis + the reference span + every chunk
    evidence file already extracted next to it)."""
    fr = Frames(spec.grid)
    if stitched:
        collect_cameras(spec, fr)
        collect_evidence(spec, fr)
        collect_reference(spec, fr)
        collect_stitched(spec, fr)
        collect_analysis(spec, fr)
    else:
        _collect_take(spec, fr, take_work, owner, final=False, refined_dir=spec.refined_dir)
        try:
            fr.max_cams = cams_recording(fr.grid, camera_spans(spec))
        except Exception:  # noqa: BLE001
            pass
    return fr


def collect_all(spec: DanceSpec) -> Frames:
    fr = Frames(spec.grid)
    collect_cameras(spec, fr)
    collect_evidence(spec, fr)
    collect_segments(spec, fr)
    collect_chunks(spec, fr)
    collect_reference(spec, fr)
    collect_stitched(spec, fr)
    collect_analysis(spec, fr)
    return fr


# ------------------------------------------------------------------------------------------------ cells / folding

_ORDER = {"none": 0, "partial": 1, "complete": 2}


def _fold(a: np.ndarray, grid: Grid, fill=0) -> np.ndarray:
    """(T,) -> (n_cells, cell_frames), padding the last cell."""
    cf, n = grid.cell_frames, grid.n_cells
    pad = n * cf - len(a)
    if pad:
        a = np.concatenate([a, np.full(pad, fill, a.dtype)])
    return a.reshape(n, cf)


def fold_cells(fr: Frames, chunks: list[dict] | None = None) -> list[dict]:
    """Frames -> 1-s cells (state, stage, reasons, per-level frame counts)."""
    g = fr.grid
    cf = g.cell_frames
    valid = _fold(np.ones(g.T, bool), g, False)
    lvl = _fold(fr.level, g)
    flv = _fold(fr.final_level, g)
    full = _fold(fr.full(), g, False)
    bad = _fold(fr.bad, g, False)
    seam = _fold(fr.seam_fail, g, False)
    flag = _fold(fr.flag, g, FLAG_UNKNOWN)
    ncam = _fold(fr.n_cams, g, 0)
    maxc = _fold(fr.max_cams, g, 0)
    cells = []
    for i in range(g.n_cells):
        v = valid[i]
        L, FL, ni = lvl[i][v], flv[i][v], int(v.sum())
        a0 = g.k0 + i * cf
        n_ref = int((L >= L_REFINED).sum())
        n_fin = int((FL >= L_REFINED).sum())
        good = int(full[i][v].sum())
        counts = {"front": int((L >= L_FRONT).sum()), "fit": int((L >= L_FIT).sum()), "refined": n_ref,
                  "analysed": int((L >= L_ANALYSED).sum()), "final": n_fin, "complete": good}
        if n_ref >= COMPLETE_HAVE * ni and good >= COMPLETE_GOOD * ni:
            state = "complete"
        elif counts["front"] == 0:
            state = "none"
        else:
            state = "partial"
        reasons: list[str] = []
        if state == "none":
            reasons.append("no data")
        elif state == "partial":
            if int(L.max()) <= L_FRONT:
                reasons.append("front only (no 3D yet)")
            elif n_ref == 0:
                reasons.append("raw fit, not refined")
            elif n_ref < COMPLETE_HAVE * ni:
                reasons.append(f"{n_ref}/{ni} frames refined")
            if n_ref and n_fin < n_ref:
                reasons.append("refined, awaiting stitch")
            fl_ref = flag[i][v][L >= L_REFINED]
            for code, name in ((FLAG_ABSENT, "absent"), (FLAG_MONO, "mono"), (FLAG_TWO, "two_view")):
                if (fl_ref == code).any():
                    reasons.append(name)
            if seam[i][v].any():
                names = sorted({fr.seam_names.get(a0 + j, "seam") for j in range(ni) if seam[i][j]})
                reasons.append("seam gate failed: " + ", ".join(names))
            if n_ref and bad[i][v][L >= L_REFINED].any():
                reasons.append(f"reprojection over the gate on {int(bad[i][v][L >= L_REFINED].sum())} frames")
        has = L >= L_FRONT
        fl_has = flag[i][v][has]
        known = fl_has[fl_has != FLAG_UNKNOWN]
        cell = {"i": i, "k0": int(a0), "k1": int(a0 + ni - 1),
                "t0": round(float(g.t(a0)), 4), "t1": round(float(g.t(a0 + ni)), 4),
                "frames": ni, "state": state, "stage": LEVELS[int(L.min())] if ni else "none",
                "n_cams": int(ncam[i][v][has].min()) if has.any() else 0,
                "max_cams": int(maxc[i][v].max()) if ni else 0,
                "flag": FLAG_NAMES[int(known.min())] if len(known) else "unknown",
                "reasons": reasons, "levels": counts, "chunk": None, "src": "computed"}
        cells.append(cell)
    if chunks:
        assign_chunks(cells, chunks)
    return cells


def assign_chunks(cells: list[dict], chunks: list[dict]) -> None:
    """Tag each cell with the chunk that owns most of its frames (core_k0..core_k1)."""
    for c in cells:
        best, bn = None, 0
        for ch in chunks:
            a, b = max(c["k0"], int(ch["core_k0"])), min(c["k1"], int(ch["core_k1"]))
            if b - a + 1 > bn:
                best, bn = ch["id"], b - a + 1
        c["chunk"] = best


# ------------------------------------------------------------------------------------------- external cells


def load_cells(path: Path) -> list[dict]:
    """Cells of a coverage file in either format ({cells: [{t0, t1, state, reasons}]} or farm's coverage.json)."""
    d = read_json(Path(path), None)
    if isinstance(d, dict):
        d = d.get("cells")
    out = []
    for c in d or []:
        if not isinstance(c, dict) or "t0" not in c or "state" not in c:
            continue
        st = str(c["state"]).lower()
        st = {"done": "complete", "full": "complete", "complete": "complete", "partial": "partial", "none": "none",
              "missing": "none", "empty": "none"}.get(st, st)
        if st in _ORDER:
            out.append({"t0": float(c["t0"]), "t1": float(c.get("t1", c["t0"] + 1.0)), "state": st,
                        "reasons": list(c.get("reasons") or []), "stage": c.get("stage")})
    return out


def merge_external(cells: list[dict], ext: list[dict], label: str) -> int:
    """Raise our cells to another file's state (never lowers). A cell takes the external cell that contains its centre."""
    if not ext:
        return 0
    ext = sorted(ext, key=lambda c: c["t0"])
    starts = np.array([c["t0"] for c in ext])
    changed = 0
    for c in cells:
        mid = 0.5 * (c["t0"] + c["t1"])
        j = int(np.searchsorted(starts, mid, side="right")) - 1
        if j < 0 or not (ext[j]["t0"] - 1e-6 <= mid <= ext[j]["t1"] + 1e-6):
            continue
        e = ext[j]
        if _ORDER[e["state"]] > _ORDER[c["state"]]:
            c["state"] = e["state"]
            c["src"] = f"external:{label}"
            c["reasons"] = [f"{label}: {r}" for r in e["reasons"]] or [f"{label}: {e['state']}"]
            changed += 1
    return changed


# ------------------------------------------------------------------------------------------------ coverage


@dataclass
class Coverage:
    dance: str
    grid: Grid
    cells: list[dict]
    frames: Frames | None = None
    world: dict = field(default_factory=dict)
    sources: list[dict] = field(default_factory=list)
    chunks: list[dict] = field(default_factory=list)
    updated: str = ""

    # numbers ----------------------------------------------------------------------------------------------------
    def counts(self) -> dict:
        c = {s: 0 for s in STATES}
        for x in self.cells:
            c[x["state"]] += 1
        return c

    def seconds(self) -> dict:
        s = {k: 0.0 for k in STATES}
        for x in self.cells:
            s[x["state"]] += x["frames"] / self.grid.hz
        return {k: round(v, 3) for k, v in s.items()}

    def percent(self) -> dict:
        tot = self.grid.T / self.grid.hz
        return {k: round(100.0 * v / tot, 1) for k, v in self.seconds().items()}

    def stages(self) -> dict:
        st = {k: 0 for k in LEVELS}
        for x in self.cells:
            st[x["stage"]] += 1
        return st

    # gaps -------------------------------------------------------------------------------------------------------
    def cell_state_by_frame(self) -> np.ndarray:
        """(T,) 0/1/2 state of the cell each frame belongs to."""
        out = np.zeros(self.grid.T, np.int8)
        for c in self.cells:
            out[c["k0"] - self.grid.k0:c["k1"] - self.grid.k0 + 1] = _ORDER[c["state"]]
        return out

    def gap_mask(self, mode: str = "missing") -> np.ndarray:
        """(T,) frames the planner should fill. mode 'missing': frames of non-complete cells that have no refined 3D
        yet or sit in a failed seam, minus what the cameras cannot improve (mono / two_view / absent);
        'all': every frame of a non-complete cell that is not already complete (and fixable). With only cell data
        (no frame arrays) whole non-complete cells are gaps."""
        st = self.cell_state_by_frame()
        noncomplete = st < 2
        fr = self.frames
        if fr is None:
            return noncomplete
        if mode == "missing":
            need = (fr.level < L_REFINED) | fr.seam_fail
        elif mode == "all":
            need = ~fr.full()
        else:
            raise ValueError(f"gap mode must be 'missing' or 'all', not {mode!r}")
        return noncomplete & need & ~fr.unfixable()

    def gaps(self, mode: str = "missing") -> list[tuple[int, int]]:
        """Runs of True in gap_mask() as inclusive (k0, k1) grid indices."""
        return runs(self.gap_mask(mode), self.grid.k0)

    # json -------------------------------------------------------------------------------------------------------
    def to_json(self) -> dict:
        g = self.grid
        return {"schema": "dancecap.farm.coverage/1", "dance": self.dance, "t0": round(float(g.t_start), 4),
                "t1": round(float(g.t_end), 4), "cell_s": CELL_S, "anchor": g.anchor, "hz": g.hz, "k0": g.k0, "k1": g.k1,
                "n_cells": len(self.cells), "cells": self.cells, "counts": self.counts(), "seconds": self.seconds(),
                "percent": self.percent(), "stages": self.stages(), "world_id": self.world.get("id"),
                "world": self.world, "chunks": self.chunks, "sources": self.sources, "updated": self.updated}

    def save(self, path: Path | None = None) -> Path:
        path = Path(path) if path else coverage_path(self.dance)
        atomic_write_json(path, self.to_json(), indent=None)
        fr = self.frames
        if fr is not None:
            side = path.with_name("coverage_frames.npz")
            tmp = side.with_name(f".{side.stem}.{id(self) % 10**6}.tmp.npz")
            np.savez_compressed(tmp, level=fr.level, flag=fr.flag, n_cams=fr.n_cams, max_cams=fr.max_cams,
                                bad=fr.bad, seam_fail=fr.seam_fail, final_level=fr.final_level, owner=fr.owner,
                                owners=np.array(fr.owners, dtype=object).astype(str),
                                grid=np.array(json.dumps(self.grid.to_dict())))
            _replace(tmp, side)
        return path

    @classmethod
    def load(cls, dance: str = DEFAULT_DANCE, path: Path | None = None, with_frames: bool = True) -> "Coverage | None":
        path = Path(path) if path else coverage_path(dance)
        d = read_json(path, None)
        if not isinstance(d, dict) or "cells" not in d:
            return None
        g = Grid(float(d["anchor"]), int(d["k0"]), int(d["k1"]), float(d.get("hz", HZ))) if "anchor" in d else None
        if g is None:   # a foreign coverage file: grid from the spec
            g = DanceSpec.load(dance).grid
        cells = d["cells"]
        for i, c in enumerate(cells):
            c.setdefault("i", i)
            if "k0" not in c:
                c["k0"], c["k1"] = g.cell_k_range(int(c["i"]))
                c["frames"] = c["k1"] - c["k0"] + 1
            c.setdefault("stage", "none")
            c.setdefault("reasons", [])
        fr = None
        side = path.with_name("coverage_frames.npz")
        if with_frames and side.exists():
            try:
                z = np.load(side, allow_pickle=False)
                if json.loads(str(z["grid"])) == g.to_dict():
                    fr = Frames(g)
                    fr.level, fr.flag, fr.n_cams = z["level"], z["flag"], z["n_cams"]
                    fr.max_cams, fr.bad, fr.seam_fail = z["max_cams"], z["bad"], z["seam_fail"]
                    fr.final_level, fr.owner, fr.owners = z["final_level"], z["owner"], [str(x) for x in z["owners"]]
            except (OSError, ValueError, KeyError):
                fr = None
        return cls(dance=d.get("dance", dance), grid=g, cells=cells, frames=fr, world=d.get("world") or {},
                   sources=d.get("sources") or [], chunks=d.get("chunks") or [], updated=d.get("updated", ""))


def _replace(src: Path, dst: Path, retries: int = 40) -> None:
    for i in range(retries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if i == retries - 1:
                raise
            time.sleep(0.05)


def runs(mask: np.ndarray, k0: int) -> list[tuple[int, int]]:
    """Runs of True as inclusive (k_start, k_end) grid indices."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(np.int8))
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    return [(int(k0 + a), int(k0 + b - 1)) for a, b in zip(starts, ends)]


def compute_coverage(dance: str = DEFAULT_DANCE, *, chunks: list[dict] | None = None, external: Path | str | bool | None = True,
                     world: dict | None = None, write: bool = True, spec: DanceSpec | None = None) -> Coverage:
    """Coverage of the dance's final result from the real outputs (see the module docstring).
    external: True = also read work/<dance>/coverage_cells.json when it exists, a path = that file, False/None = never."""
    spec = spec or DanceSpec.load(dance)
    fr = collect_all(spec)
    cells = fold_cells(fr, chunks)
    sources = list(fr.sources)
    ext_path = None
    if external is True:
        p = spec.work / "coverage_cells.json"
        ext_path = p if p.exists() else None
    elif external:
        ext_path = Path(external)
    if ext_path is not None and ext_path.exists():
        n = merge_external(cells, load_cells(ext_path), ext_path.name)
        sources.append({"kind": "external_cells", "path": str(ext_path), "raised_cells": n})
    cov = Coverage(dance=spec.name, grid=spec.grid, cells=cells, frames=fr, world=world or {}, sources=sources,
                   chunks=[{k: c[k] for k in ("id", "k0", "k1", "core_k0", "core_k1", "status", "kind", "origin") if k in c}
                           for c in (chunks or [])], updated=now_iso())
    if write:
        cov.save()
    return cov


# ------------------------------------------------------------------------------------------------- text + PNG

_ASCII = {"complete": "#", "partial": "+", "none": "."}


def ascii_map(cov: Coverage, per_row: int = 10) -> str:
    """Terminal defrag map: # complete, + partial, . none; 10 cells (10 s) per row with the row's start time."""
    lines = []
    for r in range(0, len(cov.cells), per_row):
        row = cov.cells[r:r + per_row]
        lines.append(f"{row[0]['t0']:8.1f}s  " + "".join(_ASCII[c["state"]] for c in row))
    cn, pc, sc = cov.counts(), cov.percent(), cov.seconds()
    lines.append(f"complete {cn['complete']} cells ({pc['complete']}%, {sc['complete']:.1f} s)  partial {cn['partial']} "
                 f"({pc['partial']}%, {sc['partial']:.1f} s)  none {cn['none']} ({pc['none']}%, {sc['none']:.1f} s)")
    return "\n".join(lines)


def _font(size: int, bold: bool = False):
    from PIL import ImageFont
    names = (["segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"] if bold else
             ["segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"])
    for n in names:
        try:
            return ImageFont.truetype(n, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size)
    except TypeError:   # Pillow < 10.1
        return ImageFont.load_default()


PALETTE = {
    "bg": (18, 21, 27), "text": (236, 239, 244), "dim": (150, 158, 172),
    "none": (47, 52, 63), "none_edge": (74, 80, 95),
    "complete": (46, 160, 84), "complete2": (30, 124, 64),
    "front": (226, 205, 150), "fit": (240, 168, 40), "ref": (228, 118, 28), "hatch": (30, 22, 8),
}
CHUNK_COLORS = [(86, 156, 255), (255, 107, 190), (120, 224, 224), (190, 140, 255), (255, 214, 90), (140, 230, 120)]


def _hatch_mask(size: int, radius: int, step: int = 15, width: int = 3):
    """L-mode mask: diagonal lines clipped to a rounded square (used to hatch the partial cells)."""
    from PIL import Image, ImageChops, ImageDraw
    m = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(m)
    for s in range(-size, size, step):
        d.line([(s, size), (s + size, 0)], fill=255, width=width)
    rr = Image.new("L", (size, size), 0)
    ImageDraw.Draw(rr).rounded_rectangle([0, 0, size - 1, size - 1], radius, fill=255)
    return ImageChops.multiply(m, rr)


def render_png(cov: Coverage, out: Path, chunks: list[dict] | None = None, cols: int = 10, width: int = 1080,
               title: str | None = None) -> Path:
    """Headless defrag-style map, portrait, readable on a phone. dark = not done, hatched amber = partial (paler =
    less done), green = complete; the digit in a cell is the number of cameras; a coloured bar under a cell marks the
    chunk that owns it (exact frame positions), a thin bar above it that chunk's overlap into its neighbours; a white
    outline marks a chunk that is running now."""
    from PIL import Image, ImageDraw
    chunks = list(chunks if chunks is not None else cov.chunks)
    g = cov.grid
    margin, gutter = 36, 112
    cs = (width - 2 * margin - gutter) // cols
    rows = -(-len(cov.cells) // cols)
    f_big, f_mid, f_sm, f_xs = _font(104, True), _font(34, True), _font(26), _font(21)
    H = 420 + rows * cs + 140 + 34 * min(len(chunks), 14) + 120
    im = Image.new("RGB", (width, H), PALETTE["bg"])
    d = ImageDraw.Draw(im)
    d.text((margin, 26), title or f"{cov.dance}  3D poses", font=_font(44, True), fill=PALETTE["text"])
    wid = (cov.world or {}).get("id") or "-"
    d.text((margin, 80), f"world {wid}    {g.t_start:.1f} to {g.t_end:.1f} s    {g.seconds:.1f} s    "
                         f"{(cov.updated or '')[5:16].replace('T', ' ')}", font=f_xs, fill=PALETTE["dim"])
    pc, sc = cov.percent(), cov.seconds()
    colw = (width - 2 * margin) // 3
    for j, (st, lab, col) in enumerate((("complete", "complete", PALETTE["complete"]), ("partial", "partial", PALETTE["fit"]),
                                         ("none", "not done", (176, 182, 196)))):
        x = margin + j * colw
        d.text((x, 112), f"{pc[st]:.0f}%", font=f_big, fill=col)
        d.text((x + 4, 228), f"{lab}   {sc[st]:.0f} s", font=f_mid, fill=PALETTE["text"])
    bar_y, x = 286, float(margin)
    tot = max(1e-9, sum(sc.values()))
    for st, col in (("complete", PALETTE["complete"]), ("partial", PALETTE["fit"]), ("none", PALETTE["none_edge"])):
        w = (width - 2 * margin) * sc[st] / tot
        if w > 0:
            d.rectangle([x, bar_y, x + w, bar_y + 22], fill=col)
        x += w
    top = 336
    inner = cs - 6
    tile = _hatch_mask(inner, 8, step=17, width=2)
    ck_idx = {c["id"]: i for i, c in enumerate(chunks)}
    pos = lambda n: (margin + gutter + (n % cols) * cs, top + (n // cols) * cs)   # noqa: E731
    stage_col = {"none": PALETTE["front"], "front": PALETTE["front"], "fit": PALETTE["fit"]}
    for n, c in enumerate(cov.cells):
        x0, y0 = pos(n)
        if n % cols == 0:
            d.text((margin, y0 + cs // 2 - 15), f"{c['t0']:.0f}s", font=f_sm, fill=PALETTE["dim"])
        box = [x0 + 3, y0 + 3, x0 + cs - 4, y0 + cs - 4]
        if c["state"] == "complete":
            d.rounded_rectangle(box, 8, fill=PALETTE["complete2"] if c["stage"] == "analysed" else PALETTE["complete"])
        elif c["state"] == "none":
            d.rounded_rectangle(box, 8, fill=PALETTE["none"], outline=PALETTE["none_edge"], width=2)
        else:
            col = stage_col.get(c["stage"], PALETTE["ref"])
            d.rounded_rectangle(box, 8, fill=col)
            im.paste(tuple(int(v * 0.66) for v in col), (x0 + 3, y0 + 3), tile)
        if c.get("n_cams"):   # the number of cameras, on a small dark badge so it reads on every fill
            d.rounded_rectangle([x0 + 7, y0 + 6, x0 + 31, y0 + 30], 5, fill=PALETTE["bg"])
            d.text((x0 + 13, y0 + 6), str(c["n_cams"]), font=f_xs, fill=(255, 255, 255))
    for ch in chunks:
        col = CHUNK_COLORS[ck_idx[ch["id"]] % len(CHUNK_COLORS)]
        dim = tuple(int(v * 0.7) for v in col)
        k0, k1, c0, c1 = int(ch["k0"]), int(ch["k1"]), int(ch["core_k0"]), int(ch["core_k1"])
        for a, b, off, th, cc in ((c0, c1, 8, 8, col), (k0, c0 - 1, 20, 4, dim), (c1 + 1, k1, 20, 4, dim)):
            if b < a:
                continue
            for i in range(g.cell_of_k(max(a, g.k0)), g.cell_of_k(min(b, g.k1)) + 1):
                ca, cb = g.cell_k_range(i)
                lo, hi = max(ca, a), min(cb, b)
                if hi < lo:
                    continue
                x0, y0 = pos(i)
                fx0 = x0 + 5 + (cs - 10) * (lo - ca) / g.cell_frames
                fx1 = x0 + 5 + (cs - 10) * (hi - ca + 1) / g.cell_frames
                yb = y0 + cs - 6 - off
                d.rectangle([fx0, yb - th, fx1, yb], fill=cc)
        if ch.get("status") == "running":
            for i in range(g.cell_of_k(max(c0, g.k0)), g.cell_of_k(min(c1, g.k1)) + 1):
                x0, y0 = pos(i)
                d.rounded_rectangle([x0 + 2, y0 + 2, x0 + cs - 3, y0 + cs - 3], 8, outline=(255, 255, 255), width=3)
    ly = top + rows * cs + 24
    sw = 30
    items = [("complete", PALETTE["complete"], False), ("raw fit / flagged", PALETTE["ref"], True),
             ("fit, no refine", PALETTE["fit"], True), ("front only", PALETTE["front"], True),
             ("not done", PALETTE["none"], False)]
    x = margin
    small = _hatch_mask(sw, 5, step=9, width=2)
    for lab, col, hatch in items:
        if x + sw + 14 + d.textlength(lab, font=f_xs) > width - margin:
            x, ly = margin, ly + 42
        d.rounded_rectangle([x, ly, x + sw, ly + sw], 5, fill=col,
                            outline=PALETTE["none_edge"] if lab == "not done" else None, width=2)
        if hatch:
            im.paste(PALETTE["hatch"], (x, ly), small)
        d.text((x + sw + 10, ly + 3), lab, font=f_xs, fill=PALETTE["text"])
        x += sw + 10 + int(d.textlength(lab, font=f_xs)) + 22
    ly += 50
    d.text((margin, ly), "digit = cameras    bar under a cell = chunk core    thin bar = overlap    white outline = running",
           font=f_xs, fill=PALETTE["dim"])
    ly += 38
    for ch in chunks[:14]:
        col = CHUNK_COLORS[ck_idx[ch["id"]] % len(CHUNK_COLORS)]
        d.rectangle([margin, ly + 6, margin + 26, ly + 20], fill=col)
        secs = (int(ch["core_k1"]) - int(ch["core_k0"]) + 1) / g.hz
        d.text((margin + 38, ly), f"{ch['id']}   {g.t(int(ch['core_k0'])):.1f} to {g.t(int(ch['core_k1']) + 1):.1f} s   "
                                  f"{secs:.0f} s   {ch.get('status', '')}   {ch.get('kind', '')}", font=f_xs, fill=PALETTE["text"])
        ly += 34
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    im.crop((0, 0, width, min(H, ly + 24))).save(out)
    return out
