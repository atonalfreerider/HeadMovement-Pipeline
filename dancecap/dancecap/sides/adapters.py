"""Evidence adapters of the "sides" step (DESIGN 2.2): one schema for every take, picked from the data (not from names).

    union of the chunks' select.json cameras >= 3 -> multi; = 2 -> two; = 1 -> mono; no stitch.json (a single mamma/select.json) -> single_set

Only `MultiView` is implemented now (the recap: 6 handheld phones); TwoCam / Mono / SingleSet come in the all-dances phase.

Sync rule (the trap designed out): every COCO detection is placed on the grid by the OWNING chunk's video offset (the fit's
segment_owner + mamma_dyn.apply_offset of that chunk's sync/offsets.json), exactly as hand_contact.CocoJudge does. Nothing here reads
people/track3d.npz or the world-people file (both on the audio clock: phone 00 is +113 to +125 ms off).

Ported with a take parameter (hand_contact keeps its module constants): _offsets, _camera_file, CocoJudge (all 17 joints, vectorised),
build_obs (cameras from the chunks' select.json union, compressed output).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..take import PACKAGE_ROOT
from . import geom
from .config import SidesCfg, load_cfg

PIDX = {"lead": 0, "follow": 1}     # person index in ma_2d / teachers rows (lead_slot-resolved rows: 0 lead, 1 follow)


def work_dir(take: str) -> Path:
    return PACKAGE_ROOT / "work" / take


def read_json(p: Path) -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------------------ chunks, offsets, cameras (ported)

def chunk_work(work: Path, name: str) -> Path:
    return work.parent / f"{work.name}_{name}"


def offsets(cw: Path) -> dict:
    """a chunk's sync/offsets.json as mamma_dyn.read_sync_offsets reads it (hand_contact._offsets, ported)"""
    p = cw / "sync" / "offsets.json"
    if not p.exists():
        return {}
    raw = read_json(p)
    raw = raw.get("offsets", raw)
    out = {}
    for cam, v in raw.items():
        if isinstance(v, dict):
            if v.get("use", True) is False:
                out[str(cam)] = {"delta": 0.0, "rate": 0.0, "t_ref": 0.0}
                continue
            d = v.get("delta", v.get("delta_c", v.get("offset", 0.0)))
            use_rate = v.get("use_rate", "rate" in v and "use_rate" not in v and "stable" not in v)
            rate = (v.get("rate") or 0.0) if use_rate else 0.0
            out[str(cam)] = {"delta": float(d), "rate": float(rate), "t_ref": float(v.get("t_ref", 0.0) or 0.0)}
        else:
            out[str(cam)] = {"delta": float(v), "rate": 0.0, "t_ref": 0.0}
    return out


def camera_file(work: Path, cam: str) -> Path:
    """the teacher-refined attach when adopted, else the camera track (hand_contact._camera_file, ported)"""
    j, f = work / "people" / "attach" / f"{cam}.json", work / "people" / "attach" / f"{cam}.npz"
    if j.is_file() and f.is_file():
        try:
            if read_json(j).get("adopted") is True:
                return f
        except (OSError, ValueError):
            pass
    return work / "cameras" / f"{cam}.npz"


def chunk_select_cams(work: Path) -> dict:
    """{chunk: [cams]} from stitch.json + each chunk's mamma/select.json ({} without stitch.json)"""
    p = work / "stitch.json"
    if not p.exists():
        return {}
    st = read_json(p)
    out = {}
    for ch in st.get("chunks", []):
        sp = chunk_work(work, ch["name"]) / "mamma" / "select.json"
        out[ch["name"]] = [str(c) for c in read_json(sp).get("cams", [])] if sp.exists() else []
    return out


def take_kind(work: Path) -> str:
    """multi / two / mono / single_set from the data (DESIGN 2.2)"""
    if not (work / "stitch.json").exists():
        if (work / "mamma" / "select.json").exists():
            return "single_set"
        raise FileNotFoundError(f"{work}: neither stitch.json nor mamma/select.json")
    cams = sorted({c for v in chunk_select_cams(work).values() for c in v})
    n = len(cams)
    if n >= 3:
        return "multi"
    if n == 2:
        return "two"
    if n == 1:
        return "mono"
    raise ValueError(f"{work}: no chunk select.json lists a camera")


def adapter_for(take: str, cfg: SidesCfg | None = None, work: Path | None = None):
    work = Path(work) if work else work_dir(take)
    kind = take_kind(work)
    if kind == "multi":
        return MultiView(take, cfg, work)
    raise NotImplementedError(f"{take}: the '{kind}' adapter is part of the all-dances phase (DESIGN 6.3); only 'multi' exists now")


# ------------------------------------------------------------------------------------------------ helpers

def undistort_radial(px: np.ndarray, K: np.ndarray, k1: np.ndarray, iters: int = 10) -> np.ndarray:
    """SIMPLE_RADIAL inverse (mamma_dyn.undistort_simple_radial, vectorised over frames): px (N,P,2), K (N,3,3), k1 (N,)"""
    f, cx, cy = K[:, 0, 0][:, None], K[:, 0, 2][:, None], K[:, 1, 2][:, None]
    k = np.asarray(k1, float)[:, None]
    xd, yd = (px[..., 0] - cx) / f, (px[..., 1] - cy) / f
    x, y = xd.copy(), yd.copy()
    for _ in range(iters):
        r2 = x * x + y * y
        x, y = xd / (1 + k * r2), yd / (1 + k * r2)
    return np.stack([cx + f * x, cy + f * y], -1)


def nearest_index(tc: np.ndarray, tq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """index of the nearest tc to each tq (tc sorted) and |dt|"""
    i = np.clip(np.searchsorted(tc, tq), 1, len(tc) - 1)
    left = np.abs(tc[i - 1] - tq) <= np.abs(tc[i] - tq)
    i = np.where(left, i - 1, i)
    return i, np.abs(tc[i] - tq)


def project(K, R, t, X):
    """X (n,P,3) world points at the n frames of per-frame cameras (n,3,3), (n,3,3), (n,3) -> uv (n,P,2), depth (n,P) (pinhole)"""
    Xc = np.einsum("kij,kpj->kpi", R, X) + t[:, None]
    d = Xc[..., 2]
    uv = np.einsum("kij,kpj->kpi", K, Xc)
    return uv[..., :2] / np.where(np.abs(uv[..., 2:]) > 1e-9, uv[..., 2:], 1e-9), d


# ------------------------------------------------------------------------------------------------ the multi-view adapter

class MultiView:
    """Evidence of a take filmed by >= 3 cameras, stitched from chunks (the recap). Lazy: nothing heavy is read until asked."""

    kind = "multi"

    def __init__(self, take: str, cfg: SidesCfg | None = None, work: Path | None = None, frames: slice | None = None):
        self.take = take
        self.cfg = cfg or load_cfg(take)
        self.work = Path(work) if work else work_dir(take)
        self.stitch = read_json(self.work / "stitch.json")
        self.chunk_names = [c["name"] for c in self.stitch["chunks"]]
        self.select = chunk_select_cams(self.work)
        self.cams = sorted({c for v in self.select.values() for c in v})
        self.offs = {n: offsets(chunk_work(self.work, n)) for n in self.chunk_names}
        self._base = None
        self._obs = None
        self._coco = None
        self.frames = frames

    # ---------------------------------------------------------------- base fit
    @property
    def base_dir(self) -> Path:
        return self.work / self.cfg.base_fit

    def base(self) -> dict:
        if self._base is None:
            out = {}
            for r in self.cfg.roles:
                with np.load(self.base_dir / f"{r}.npz", allow_pickle=True) as z:
                    out[r] = {k: z[k] for k in z.files}
            self._base = out
        return self._base

    @property
    def t(self) -> np.ndarray:
        return np.asarray(self.base()["lead"]["t"], float)

    @property
    def T(self) -> int:
        return len(self.t)

    @property
    def owner(self) -> np.ndarray:
        return np.asarray(self.base()["lead"]["segment_owner"]).astype(str)

    # ---------------------------------------------------------------- consolidated grid observations (MAMMA 512 + grid cameras)
    def obs_dir(self) -> Path:
        for d in self.cfg.obs_dirs:
            p = self.work / d
            if all((p / f"cam{c}.npz").exists() for c in self.cams):
                return p
        raise FileNotFoundError(f"{self.take}: no consolidated obs in {list(self.cfg.obs_dirs)} for cams {self.cams} "
                                f"(python -m dancecap.sides obs --take {self.take} builds them from the chunks' ma_2d)")

    def obs(self) -> dict:
        """{cam: {"K","R","t" (T,...), "ok" (T,), "C" (T,3) camera centre, "f" (T,) focal / 1000, "vis" (T,2,512) float16, "size"}}"""
        if self._obs is None:
            d = self.obs_dir()
            out = {}
            for c in self.cams:
                with np.load(d / f"cam{c}.npz") as z:
                    o = {k: np.asarray(z[k]) for k in ("K", "R", "t", "ok", "vis")}
                    o["size"] = np.asarray(z["size"]) if "size" in z.files else np.array([1920, 1080])
                o["ok"] = o["ok"].astype(bool)
                o["C"] = -np.einsum("kji,kj->ki", o["R"], o["t"])
                o["f"] = o["K"][:, 0, 0] / 1000.0
                out[c] = o
            self._obs = out
        return self._obs

    def landmarks(self, cam: str) -> np.ndarray:
        """MAMMA's 512 2D landmarks (T,2,512,2) of one camera (read on demand: large)"""
        with np.load(self.obs_dir() / f"cam{cam}.npz") as z:
            if "lm" in z.files:
                return np.asarray(z["lm"])
            return np.asarray(z["lm_q8"], np.float32) / 8.0          # the pods' quantized copy (1/8 px)

    def proj_mats(self, cam: str) -> np.ndarray:
        o = self.obs()[cam]
        return np.einsum("kij,kjl->kil", o["K"], np.concatenate([o["R"], o["t"][:, :, None]], 2))

    # ---------------------------------------------------------------- COCO-17 on the grid (the owner's teachers, synced per chunk)
    def coco_tracks(self) -> dict:
        """{cam: {"uv" (T,2,17,2) undistorted px (NaN = none), "conf" (T,2,17), "det" (T,2) person confidence, "dt" (T,) |time error|}}
        Each grid frame takes the phone frame nearest to its time on the OWNING chunk's video clock (within cfg.max_dt_s)."""
        if self._coco is not None:
            return self._coco
        from .. import mamma_dyn as md
        cfg = self.cfg
        T, tg, own = self.T, self.t, self.owner
        out = {}
        for c in self.cams:
            tz = np.load(self.work / "people" / f"teachers_{c}.npz", allow_pickle=True)
            cz = np.load(camera_file(self.work, c), allow_pickle=True)
            tt = np.asarray(tz["t"], float)
            kp = np.asarray(tz["kp"], float)
            pc = np.asarray(tz["conf"], float)
            Kc = np.asarray(cz["K"], float)
            k1 = np.asarray(cz["dist"], float).ravel()
            uv = np.full((T, 2, 17, 2), np.nan, np.float32)
            cf = np.zeros((T, 2, 17), np.float32)
            det = np.zeros((T, 2), np.float32)
            dt = np.full(T, np.inf)
            for name in np.unique(own):
                ks = np.flatnonzero(own == name)
                tc = md.apply_offset(tt, self.offs.get(name, {}).get(c))
                i, e = nearest_index(tc, tg[ks])
                good = e <= cfg.max_dt_s
                ks, i = ks[good], i[good]
                dt[ks] = e[good]
                if not len(ks):
                    continue
                k = kp[i]                                       # (n,2,17,3)
                fin = np.isfinite(k).all(-1)
                u = undistort_radial(np.nan_to_num(k[..., :2]).reshape(len(i), 34, 2), Kc[i], k1[i]).reshape(len(i), 2, 17, 2)
                u[~fin] = np.nan
                uv[ks] = u
                cf[ks] = np.where(fin, np.nan_to_num(k[..., 2]), 0.0)
                det[ks] = np.nan_to_num(pc[i])
            out[c] = {"uv": uv, "conf": cf, "det": det, "dt": dt}
        self._coco = out
        return out

    def coco_valid(self, cam: str, role: str, kp_conf: float | None = None) -> np.ndarray:
        """(T,17) detections usable for triangulation: person conf >= det_conf, key point conf >= kp_conf, finite, camera ok"""
        cfg = self.cfg
        c = self.coco_tracks()[cam]
        p = PIDX[role]
        kc = cfg.kp_conf if kp_conf is None else kp_conf
        ok = self.obs()[cam]["ok"]
        return (c["det"][:, p, None] >= cfg.det_conf) & (c["conf"][:, p] >= kc) & np.isfinite(c["uv"][:, p, :, 0]) & ok[:, None]

    # ---------------------------------------------------------------- base joints in COCO order
    def base_coco(self, role: str) -> np.ndarray:
        J = np.asarray(self.base()[role]["joints"], float)
        return J[:, list(geom.C2S)]

    # ---------------------------------------------------------------- role-swap self-check (DESIGN 2.2)
    def role_swap_check(self, max_frames: int = 3000, seed: int = 0) -> dict:
        """per camera: on frames where both dancers are detected, the COCO error with roles as stored must be smaller than with roles
        swapped on >= role_check_min of frames -> {cam: {"n", "stored_better", "ok"}}; raises when cfg.role_check_strict and any fails"""
        cfg = self.cfg
        rng = np.random.default_rng(seed)
        out = {}
        XL, XF = self.base_coco("lead"), self.base_coco("follow")
        body = np.arange(5, 17)
        for c in self.cams:
            o = self.obs()[c]
            co = self.coco_tracks()[c]
            vl, vf = self.coco_valid(c, "lead")[:, body], self.coco_valid(c, "follow")[:, body]
            both = (vl.sum(1) >= 6) & (vf.sum(1) >= 6)
            ks = np.flatnonzero(both)
            if len(ks) > max_frames:
                ks = np.sort(rng.choice(ks, max_frames, replace=False))
            if not len(ks):
                out[c] = {"n": 0, "stored_better": None, "ok": True}
                continue
            uL, _ = project(o["K"][ks], o["R"][ks], o["t"][ks], XL[ks][:, body])
            uF, _ = project(o["K"][ks], o["R"][ks], o["t"][ks], XF[ks][:, body])
            dL, dF = co["uv"][ks, 0][:, body], co["uv"][ks, 1][:, body]

            def med(a, b, m):
                e = np.linalg.norm(a - b, axis=-1)
                e = np.where(m, e, np.nan)
                return np.nanmedian(e, 1)
            with np.errstate(invalid="ignore"):
                st = med(dL, uL, vl[ks]) + med(dF, uF, vf[ks])
                sw = med(dL, uF, vl[ks]) + med(dF, uL, vf[ks])
            good = np.isfinite(st) & np.isfinite(sw)
            frac = float((st[good] < sw[good]).mean()) if good.any() else None
            out[c] = {"n": int(good.sum()), "stored_better": frac, "ok": frac is None or frac >= cfg.role_check_min,
                      "median_px_stored": float(np.median(st[good])) if good.any() else None,
                      "median_px_swapped": float(np.median(sw[good])) if good.any() else None}
        bad = [c for c, v in out.items() if not v["ok"]]
        if bad and cfg.role_check_strict:
            raise RuntimeError(f"{self.take}: role-swap check failed for cams {bad}: {out}")
        return out

    # ---------------------------------------------------------------- overlapping chunk solutions (second opinions for spins)
    def alternatives(self) -> list[dict]:
        """[{"chunk", "frame_offset", "k0", "k1", "dir"}] of every chunk fit on disk (placed by stitch.json grid.frame_offset)"""
        fo = self.stitch.get("grid", {}).get("frame_offset", {})
        out = []
        for ch in self.stitch["chunks"]:
            d = chunk_work(self.work, ch["name"]) / self.cfg.base_fit
            if not (d / "lead.npz").exists():
                continue
            with np.load(d / "lead.npz", allow_pickle=True) as z:
                n = len(z["t"])
            off = int(fo.get(ch["name"], ch.get("k", [0])[0]))
            out.append({"chunk": ch["name"], "frame_offset": off, "k0": off, "k1": off + n - 1, "dir": str(d)})
        return out

    def summary(self) -> dict:
        return {"take": self.take, "kind": self.kind, "cams": self.cams, "chunks": self.chunk_names, "T": self.T,
                "base": str(self.base_dir), "select": self.select}


def load_obs(take: str, cfg: SidesCfg | None = None) -> dict:
    return MultiView(take, cfg).obs()


def coco_tracks(take: str, cfg: SidesCfg | None = None) -> dict:
    return MultiView(take, cfg).coco_tracks()


def alternatives(take: str, cfg: SidesCfg | None = None) -> list[dict]:
    return MultiView(take, cfg).alternatives()


def role_swap_check(take: str, cfg: SidesCfg | None = None) -> dict:
    return MultiView(take, cfg).role_swap_check()


# ------------------------------------------------------------------------------------------------ building the obs (ported hc.build_obs)

def build_obs(take: str, cfg: SidesCfg | None = None, out_dir: Path | None = None, force: bool = False) -> dict:
    """Per camera on the full grid: the owning chunk's ma_2d landmarks + its grid cameras (hand_contact.build_obs with the take as a
    parameter and the cameras from the chunks' select.json union) -> work/<take>/sides/obs/cam<c>.npz (compressed)"""
    from .. import mamma_dyn as md
    cfg = cfg or load_cfg(take)
    work = work_dir(take)
    out_dir = Path(out_dir or work / "sides" / "obs")
    out_dir.mkdir(parents=True, exist_ok=True)
    st = read_json(work / "stitch.json")
    with np.load(work / cfg.base_fit / "lead.npz", allow_pickle=True) as z:
        t = np.asarray(z["t"], float)
        owner = np.asarray(z["segment_owner"]).astype(str)
    T = len(t)
    sim3 = read_json(work / "cameras" / "world_from_sfm.json")
    fo = st["grid"]["frame_offset"]
    cams = sorted({c for v in chunk_select_cams(work).values() for c in v})
    rep = {"chunks": {}, "cams": {}}
    cams_np = {}
    for cam in cams:
        if (out_dir / f"cam{cam}.npz").exists() and not force:
            rep["cams"][cam] = "cached"
            continue
        c = np.load(camera_file(work, cam), allow_pickle=True)
        cams_np[cam] = {"t": np.asarray(c["t"], float), "K": np.asarray(c["K"], float), "ok": np.asarray(c["ok"], bool),
                        "Rw_tw": md.world_cameras(np.asarray(c["R"], float), np.asarray(c["tvec"], float), sim3),
                        "size": tuple(int(x) for x in c["native_size"])}
    buf = {cam: {"lm": np.zeros((T, 2, 512, 2), np.float32), "vis": np.zeros((T, 2, 512), np.float16),
                 "con": np.zeros((T, 2, 512), np.float16), "K": np.zeros((T, 3, 3)), "R": np.tile(np.eye(3), (T, 1, 1)),
                 "t": np.zeros((T, 3)), "ok": np.zeros(T, bool), "chunk": np.full(T, "", "<U8")} for cam in cams_np}
    for ch in st["chunks"]:
        name = ch["name"]
        ks = np.flatnonzero(owner == name)
        if not len(ks) or not cams_np:
            continue
        cw = chunk_work(work, name)
        sel = read_json(cw / "mamma" / "select.json")
        lm_dir = cw / "mamma" / "ma_2d" / f"{cw.name}_{sel['chosen']}"
        offs = offsets(cw)
        loc = ks - int(fo[name])
        rep["chunks"][name] = {"frames_owned": int(len(ks)), "cams": sel["cams"]}
        for cam in sel["cams"]:
            if cam not in cams_np:
                continue
            z = np.load(lm_dir / f"cam{cam}.npz")
            b = buf[cam]
            b["lm"][ks] = z["landmarks"][loc, :2, :, :2]
            b["vis"][ks] = z["visibilities"][loc, :2]
            b["con"][ks] = z["contacts"][loc, :2]
            cn = cams_np[cam]
            tc = md.apply_offset(cn["t"], offs.get(cam))
            R, tv = cn["Rw_tw"]
            cg = md.resample_cameras(tc, cn["K"], R, tv, cn["ok"], t[ks])
            b["K"][ks], b["R"][ks], b["t"][ks], b["ok"][ks] = cg["K"], cg["R"], cg["t"], cg["ok"]
            b["chunk"][ks] = name
    for cam, b in buf.items():
        b["vis"][~b["ok"]] = 0
        np.savez_compressed(out_dir / f"cam{cam}.npz", **b, size=np.array(cams_np[cam]["size"]))
        rep["cams"][cam] = {"frames_with_camera": int(b["ok"].sum())}
    return rep
