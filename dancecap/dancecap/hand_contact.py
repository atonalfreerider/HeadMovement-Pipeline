"""Hand-to-partner contacts of the class recap (larissa_kadu_recap): find them, then make the hands land on the partner (v3, 2026-10-21).

    python -m dancecap.hand_contact obs [--check]   # the 6 phones' 2D on the 30 Hz grid             -> v3/hand_contact/obs/cam<c>.npz, obs.json
    python -m dancecap.hand_contact detect          # contact events (mesh + 2D + hysteresis)         -> v3/hand_contact/events.json, series.npz
    python -m dancecap.hand_contact sheet [--n 20]  # phone crops + the fit's mesh per event          -> v3/hand_contact/review/events_sheet_*.jpg
    python -m dancecap.hand_contact refine          # the contact refinement (GPU, gpu.lock)          -> v3/smplx_contact/{lead,follow}.npz,
                                                    #                                                    v3/hand_contact/refine_all.json
    python -m dancecap.hand_contact folds           # held-out check of the COCO-driven corrections   -> v3/hand_contact/refine_fold{A,B}.json
    python -m dancecap.hand_contact pass2           # hand-only pass on the events pass 1 left           -> refine_pass2.json (merged in)
    python -m dancecap.hand_contact pass3           # looser fingertip-jitter guard on the rest          -> refine_pass3.json (merged in)
    python -m dancecap.hand_contact pass4           # post-check (after validate): events whose hand now sinks deeper than in the input
                                                    #   against the FINAL fit, solved again on top of it      -> refine_pass4.json (merged in)
    python -m dancecap.hand_contact validate        # before / after numbers + the safety checks      -> v3/hand_contact/validate.json
    python -m dancecap.hand_contact speed-audit     # whips / added jumps over the whole class (every hand, against the input)
    python -m dancecap.hand_contact before-after    # before / after sheets (12 events + --extra 118,47) -> v3/hand_contact/review/before_after_*.jpg
    python -m dancecap.hand_contact unity-motion    # the refined motion through export_unity into a SIDE capture (never installed)
                                                    #   -> work/<take>/export/06_LarissaKaduRecap_v3hands/, v3/unity_motion/ (+ the comparison
                                                    #      with the installed capture's motion, sha256 of the files for install_motion.sh)

Safety pass (2026-10-21, after an independent review of the first result: a 43 cm whip at 223.4 s and Kadu's hand put on Larissa's front waist
in #118 while the video shows it on her back, 2.1 m/s at 139.3 s in #47, 16 holds 3-20 px further from MAMMA's hand points, new 3-5 mm
penetration in four hand-hand holds, fanned-out open hands, palms 2 cm off while the fingertips touched): the refinement now has a loss on the
motion it ADDS (free up to 0.3 m/s), blend ramps as long as the move needs (up to 1.5 s), the flat 'open' hand with the fingers closed
(finger_template), the palm itself on the surface for flat holds (FLAT_HOLDS), exact inside tests for hand-hand and a push-out polish, and
guards on every event: MAMMA's hand points over all core frames (<= +3 px), the independent COCO wrist / elbow (<= +4 px), no new penetration
(exact), no added jump (> 0.55 m/s) or whip (> the input's p99 + 0.25 m/s where the input was slower), besides the earlier ones.
Decisions taken by looking at the phone crops are in v3/hand_contact/curation.json (load_curation: veto, keep, overrides), each with its reason.

Owner (2026-10-21): "is there anything we can do about hand to hand contact and hand to forearm and hand to shoulder contact? there is a
tendency for arms and hands to land in awkward places and shapes in dance positions".

The input is the stitched, refined recap fit work/larissa_kadu_recap/smplx_rf2_noslip/ (MAMMA locked-head SMPL-X, 30 Hz grid, T = 10815, hands
absolute with flat_hand_mean = True: hand pose 0 = a flat hand). Its hands are MAMMA's raw hands (no hand refinement: the demo's 18-hour hand
pipeline, refine_hands / hands_detail, was retired by the owner on 2026-10-07; MAMMA's fingers here sit at MANO's mean relaxed pose almost all the
time). This module is the light, targeted replacement for the moments that matter: the partner holds. The refinement is described at its section
("the contact refinement"): arm IK to the multi-view COCO wrists / elbows where MAMMA's own hand landmarks do not support the fitted hand, then the
hand onto the partner's surface (soft-min gap, palm within reach and facing the surface, a second start with the forearm twisted when the palm
faces away), the finger template of the hold, no penetration, raised-cosine blends; guards keep an event's input when the 2D gets worse.

2D evidence (`obs`): the 512 MAMMA landmarks (ma_2d) of every phone, resampled on each chunk's grid by the chunk's own front step (the fit's input),
taken from the chunk that OWNS each grid frame (the fit's segment_owner), with the chunk's grid cameras rebuilt from the camera tracks
(cameras/<cam>.npz, R3's adopted attach for 05) + the chunk's own video offsets (sync/offsets.json) exactly as mamma_dyn.build_inputs did
(pinhole, undistorted). Checked by reprojecting the fit: the per-camera medians must match the chunks' own pose_quality numbers.

Region of a partner surface point: the dominant skinning joint of the closest vertex, the torso split front / back by the surface normal against
the partner's facing (refine_hands.region_name), refined into hold targets: hand, forearm, upper arm, shoulder (top / front of the shoulder
girdle), shoulder blade (back of the shoulder girdle and the upper back above the scapula's lower tip), back (mid / lower back), waist / side,
chest, neck / head.

Everything is written under work/larissa_kadu_recap/v3/ (git-ignored likeness data); the input fit is never touched.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .take import PACKAGE_ROOT

TAKE = "larissa_kadu_recap"
WORK = PACKAGE_ROOT / "work" / TAKE
SRC_FIT = WORK / "smplx_rf2_noslip"
V3 = WORK / "v3"
OUT = V3 / "hand_contact"
FIT_OUT = V3 / "smplx_contact"
REVIEW = OUT / "review"
ROLES = ("lead", "follow")
OTHER = {"lead": "follow", "follow": "lead"}
SIDES = ("left", "right")
CAMS = ("00", "01", "02", "03", "05", "06")
PIDX = {"lead": 0, "follow": 1}           # person index in ma_2d (mamma_dyn.PIDS = (1, 2): 01 lead, 02 follow)
HZ = 30.0
DEBUG = bool(__import__("os").environ.get("HC_DEBUG"))


def log(*a) -> None:
    print("[hand_contact]", *a, flush=True)


def py_json(x):
    if isinstance(x, dict):
        return {str(k): py_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [py_json(v) for v in x]
    if isinstance(x, np.ndarray):
        return py_json(x.tolist())
    if isinstance(x, (np.floating,)):
        v = float(x)
        return None if v != v else round(v, 5)
    if isinstance(x, float):
        return None if x != x else round(x, 5)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def write_json(p: Path, obj) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(py_json(obj), indent=1), encoding="utf-8")
    return p


# ------------------------------------------------------------------------------------------------ fit

def load_fit(d: Path = SRC_FIT) -> dict:
    out = {}
    for r in ROLES:
        with np.load(d / f"{r}.npz", allow_pickle=True) as z:
            out[r] = {k: z[k] for k in z.files}
    return out


def model_dir() -> Path:
    from . import mamma_dyn as md
    return md.SMPLX_MODELS


def full_pose(d: dict) -> np.ndarray:
    from .hand_kin import full_pose as fp
    return fp(d)


_KIN_CACHE: dict = {}


def kin_for(betas: np.ndarray, device: str, subset: np.ndarray | None = None, dtype: str = "float64"):
    """a SmplxKin of these betas on `device` (the model file is read once per process), with its own vertex subset; dtype float32 casts
    every tensor (the optimiser's FK: the 2080 Ti runs float64 at 1/32 rate)"""
    import copy
    from .hand_kin import SmplxKin
    key = (tuple(np.round(np.asarray(betas, float).ravel(), 8)), str(device))
    if key not in _KIN_CACHE:
        _KIN_CACHE[key] = SmplxKin(model_dir(), np.asarray(betas, float).reshape(-1), device)
    k = copy.copy(_KIN_CACHE[key])
    k.set_subset(subset)
    if dtype == "float32":
        import torch
        k.J, k.rel = k.J.float(), k.rel.float()
        k._tip = {kk: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v) for kk, v in k._tip.items()}
        if k.subset is not None:
            k.subset = {kk: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v) for kk, v in k.subset.items()}
    return k


_FULL_CACHE: dict = {}


def full_kin(betas: np.ndarray, device: str):
    """the full-surface SmplxKin of these betas (the posedirs of all 10475 vertices are built once per process)"""
    key = (tuple(np.round(np.asarray(betas, float).ravel(), 8)), str(device))
    if key not in _FULL_CACHE:
        k = kin_for(betas, device)
        k.set_subset(np.arange(len(k.v_shaped_np)))
        _FULL_CACHE[key] = k
    return _FULL_CACHE[key]


class Body:
    """One dancer's SMPL-X kinematics (MAMMA locked head, hand_kin.SmplxKin) with the full surface on demand."""

    def __init__(self, d: dict, device: str = "cpu"):
        self.d = d
        self.device = device
        self.kin = kin_for(d["betas"], device)
        self.faces = self.kin.faces
        self.T = len(d["t"])

    def forward_np(self, pose: np.ndarray, transl: np.ndarray, verts: bool = True, batch: int = 512,
                   ids: np.ndarray | None = None) -> dict:
        """pose (T,55,3), transl (T,3) -> joints (T,55,3) float64 and verts (T,V,3) float32 (all vertices, or ids)."""
        import torch
        if ids is None:
            kin = full_kin(self.d["betas"], self.device)
        else:
            kin = self.kin
            kin.set_subset(np.asarray(ids))
        T = len(pose)
        J = np.zeros((T, 55, 3))
        V = np.zeros((T, len(kin.subset["ids"]), 3), np.float32) if verts else None
        with torch.no_grad():
            for a in range(0, T, batch):
                p = torch.tensor(pose[a:a + batch], dtype=torch.float64, device=self.device)
                tr = torch.tensor(transl[a:a + batch], dtype=torch.float64, device=self.device)
                o = kin.forward(p, tr, want_verts=verts)
                J[a:a + batch] = o["joints"].cpu().numpy()
                if verts:
                    V[a:a + batch] = o["verts"].cpu().numpy().astype(np.float32)
        if ids is not None:
            kin.set_subset(None)
        return {"joints": J, "verts": V}


def vertex_joint() -> np.ndarray:
    from .hand_kin import model_npz
    return np.asarray(np.load(model_npz(model_dir()), allow_pickle=True)["weights"], float).argmax(1)


def skin_weights() -> np.ndarray:
    from .hand_kin import model_npz
    return np.asarray(np.load(model_npz(model_dir()), allow_pickle=True)["weights"], float)


# ------------------------------------------------------------------------------------------------ grid observations (obs)

def _chunk_work(name: str) -> Path:
    return WORK.parent / f"{TAKE}_{name}"


def _offsets(chunk_work: Path) -> dict:
    """the chunk's sync/offsets.json as mamma_dyn.read_sync_offsets reads it (without the clock-fingerprint check)"""
    p = chunk_work / "sync" / "offsets.json"
    if not p.exists():
        return {}
    raw = json.loads(p.read_text(encoding="utf-8"))
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


def _camera_file(cam: str) -> Path:
    """farm/recap_profile.camera_source: R3's teacher-refined attach when adopted, else R2's track"""
    j, f = WORK / "people" / "attach" / f"{cam}.json", WORK / "people" / "attach" / f"{cam}.npz"
    if j.is_file() and f.is_file():
        try:
            if json.loads(j.read_text(encoding="utf-8")).get("adopted") is True:
                return f
        except (OSError, ValueError):
            pass
    return WORK / "cameras" / f"{cam}.npz"


def build_obs(force: bool = False) -> dict:
    """Per camera on the full 30 Hz grid: ma_2d landmarks of the owning chunk + its grid cameras -> OUT/obs/cam<c>.npz"""
    from . import mamma_dyn as md
    out_dir = OUT / "obs"
    out_dir.mkdir(parents=True, exist_ok=True)
    st = json.loads((WORK / "stitch.json").read_text(encoding="utf-8"))
    fit = load_fit()
    t = np.asarray(fit["lead"]["t"], float)
    T = len(t)
    owner = np.asarray(fit["lead"]["segment_owner"]).astype(str)
    sim3 = json.loads((WORK / "cameras" / "world_from_sfm.json").read_text(encoding="utf-8"))
    fo = st["grid"]["frame_offset"]
    rep = {"chunks": {}, "cams": {}}
    cams_np = {}
    for cam in CAMS:
        f = out_dir / f"cam{cam}.npz"
        if f.exists() and not force:
            rep["cams"][cam] = "cached"
            continue
        c = np.load(_camera_file(cam), allow_pickle=True)
        cams_np[cam] = {"t": np.asarray(c["t"], float), "K": np.asarray(c["K"], float), "ok": np.asarray(c["ok"], bool),
                        "Rw_tw": md.world_cameras(np.asarray(c["R"], float), np.asarray(c["tvec"], float), sim3),
                        "size": tuple(int(x) for x in c["native_size"]), "file": str(_camera_file(cam))}
    if not cams_np:
        return json.loads((OUT / "obs.json").read_text(encoding="utf-8"))
    buf = {cam: {"lm": np.zeros((T, 2, 512, 2), np.float32), "vis": np.zeros((T, 2, 512), np.float16),
                 "con": np.zeros((T, 2, 512), np.float16), "K": np.zeros((T, 3, 3)), "R": np.tile(np.eye(3), (T, 1, 1)),
                 "t": np.zeros((T, 3)), "ok": np.zeros(T, bool), "chunk": np.full(T, "", "<U5")} for cam in cams_np}
    for ch in st["chunks"]:
        name = ch["name"]
        ks = np.flatnonzero(owner == name)
        if not len(ks):
            continue
        cw = _chunk_work(name)
        sel = json.loads((cw / "mamma" / "select.json").read_text(encoding="utf-8"))
        lm_dir = cw / "mamma" / "ma_2d" / f"{cw.name}_{sel['chosen']}"
        offs = _offsets(cw)
        loc = ks - int(fo[name])
        rep["chunks"][name] = {"frames_owned": int(len(ks)), "cams": sel["cams"], "ma_2d": str(lm_dir),
                               "offsets_ms": {k: round(v["delta"] * 1000, 2) for k, v in offs.items()}}
        for cam in sel["cams"]:
            if cam not in cams_np:
                continue
            z = np.load(lm_dir / f"cam{cam}.npz")
            n = z["landmarks"].shape[0]
            if loc.max() >= n:
                raise ValueError(f"{name}/{cam}: ma_2d has {n} frames, owned local frames reach {loc.max()}")
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
        np.savez(out_dir / f"cam{cam}.npz", **b, size=np.array(cams_np[cam]["size"]))
        rep["cams"][cam] = {"frames_with_camera": int(b["ok"].sum()), "file": cams_np[cam]["file"],
                            "visible_frac": {r: float((b["vis"][:, PIDX[r]] > 0.5).mean()) for r in ROLES}}
        log(f"obs cam{cam}: {int(b['ok'].sum())} grid frames with a camera")
    write_json(OUT / "obs.json", rep)
    return rep


class Obs:
    """The consolidated grid observations (lazy per camera)."""

    def __init__(self, cams=CAMS):
        self.cams = list(cams)
        self.z = {}
        for c in self.cams:
            with np.load(OUT / "obs" / f"cam{c}.npz") as z:
                self.z[c] = {k: z[k] for k in z.files}

    def project(self, cam: str, X: np.ndarray, ks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """X (n_k, P, 3) world points at grid frames ks -> pixels (n_k, P, 2), depth (n_k, P) (pinhole, undistorted: MAMMA's frame)"""
        z = self.z[cam]
        R, t, K = z["R"][ks], z["t"][ks], z["K"][ks]
        Xc = np.einsum("kij,kpj->kpi", R, X) + t[:, None]
        d = Xc[..., 2]
        uv = np.einsum("kij,kpj->kpi", K, Xc)
        return uv[..., :2] / np.where(np.abs(uv[..., 2:]) > 1e-9, uv[..., 2:], 1e-9), d


def landmark_matrix() -> np.ndarray:
    from . import mamma_dyn as md
    return md._load_verts512()          # (512, 10475) barycentric-like weights of MAMMA's points on the SMPL-X surface


def landmark_parts() -> dict:
    from . import mamma_dyn as md
    return md.part_index()


def check_obs(n_frames: int = 1500, seed: int = 0) -> dict:
    """Reprojection of the fit's 512 points against the consolidated landmarks, per camera and dancer (median over frames of the per-frame median):
    must match the chunks' own pose_quality numbers (a few px), else the grid cameras are wrong."""
    fit = load_fit()
    obs = Obs()
    D = landmark_matrix().astype(np.float32)
    rng = np.random.default_rng(seed)
    T = len(fit["lead"]["t"])
    ks = np.sort(rng.choice(T, min(n_frames, T), replace=False))
    res = {}
    for r in ROLES:
        b = Body(fit[r])
        fw = b.forward_np(full_pose(fit[r])[ks], fit[r]["transl"][ks])
        fk_err = float(np.abs(fw["joints"] - fit[r]["joints"][ks]).max())
        pts = np.einsum("pv,kvd->kpd", D, fw["verts"])
        res[r] = {"fk_max_err_mm": round(fk_err * 1000, 3)}
        for c in obs.cams:
            z = obs.z[c]
            uv, dep = obs.project(c, pts, ks)
            o = z["lm"][ks, PIDX[r]]
            ok = (z["vis"][ks, PIDX[r]] > 0.5) & (np.abs(o).sum(-1) > 0) & (dep > 0)
            e = np.linalg.norm(uv - o, axis=-1)
            per = np.array([np.median(e[i][ok[i]]) if ok[i].sum() >= 3 else np.nan for i in range(len(ks))])
            res[r][c] = {"median_px": float(np.nanmedian(per)) if np.isfinite(per).any() else None,
                         "p90_px": float(np.nanpercentile(per, 90)) if np.isfinite(per).any() else None,
                         "frames": int(np.isfinite(per).sum())}
    write_json(OUT / "obs_check.json", res)
    return res


# ------------------------------------------------------------------------------------------------ regions + hand geometry

# dominant skinning joint -> base region of the partner's surface (torso parts are split front / back / side by the normal)
REGION_BASE = {20: "hand", 21: "hand", 18: "forearm", 19: "forearm", 16: "upper_arm", 17: "upper_arm",
               13: "girdle", 14: "girdle", 9: "spine3", 6: "spine2", 3: "spine1", 0: "pelvis",
               12: "neck_head", 15: "neck_head", 22: "neck_head", 23: "neck_head", 24: "neck_head",
               1: "hip", 2: "hip", 4: "leg", 5: "leg", 7: "leg", 8: "leg", 10: "leg", 11: "leg"}
for _j in range(25, 55):
    REGION_BASE[_j] = "hand"
REGIONS = ("none", "hand", "forearm", "upper_arm", "shoulder", "shoulder_blade", "back", "chest", "waist", "neck_head", "hip", "leg")
REGION_ID = {n: i for i, n in enumerate(REGIONS)}
# what the hand does on each region (the finger template and the palm's job)
HOLD_OF_REGION = {"hand": "hand_hold", "forearm": "hand_on_forearm", "upper_arm": "hand_on_upper_arm", "shoulder": "hand_on_shoulder",
                  "shoulder_blade": "hand_on_shoulder_blade", "back": "hand_on_back", "chest": "hand_on_chest", "waist": "hand_on_waist",
                  "neck_head": "hand_on_neck_head", "hip": "hand_on_waist", "leg": "hand_on_leg"}


def region_of(joint: np.ndarray, normal_dot_fwd: np.ndarray) -> np.ndarray:
    """(n,) dominant joint + the surface normal against the partner's facing (+1 front, -1 back) -> region ids (REGIONS).
    girdle (collar bones): the back of it is the shoulder blade, the rest the shoulder; spine3 (upper torso): back = shoulder blade
    (upper back), front = chest, side = the side under the arm (waist); spine2: back = back, front = chest; spine1 / pelvis: back = back
    (lower back), else waist."""
    out = np.zeros(len(joint), np.int8)
    for i, (j, d) in enumerate(zip(np.asarray(joint), np.asarray(normal_dot_fwd))):
        b = REGION_BASE.get(int(j), "none")
        if b == "girdle":
            r = "shoulder_blade" if d < -0.3 else "shoulder"
        elif b == "spine3":
            r = "shoulder_blade" if d < -0.35 else ("chest" if d > 0.35 else "waist")
        elif b == "spine2":
            r = "back" if d < -0.35 else ("chest" if d > 0.35 else "waist")
        elif b in ("spine1", "pelvis"):
            r = "back" if d < -0.35 else "waist"
        else:
            r = b
        out[i] = REGION_ID[r]
    return out


def torso_forward(joints: np.ndarray) -> np.ndarray:
    """(T,3) horizontal facing (left hip - right hip) x up"""
    fwd = -np.cross(joints[:, 2] - joints[:, 1], np.array([0.0, 1.0, 0.0]))
    return fwd / np.maximum(np.linalg.norm(fwd, axis=1, keepdims=True), 1e-9)


@dataclass
class HandIds:
    """Vertex sets of one hand of the shaped SMPL-X: the hand surface (sampled), the palm patch (the volar side of the palm), the forearm"""
    hand: np.ndarray
    palm: np.ndarray
    arm: np.ndarray
    mcp: tuple          # SMPL-X joints: index1, middle1, ring1, pinky1 (the knuckles)


def mcp_joints(side: str) -> tuple:
    """SMPL-X knuckle joints (index1, middle1, ring1, pinky1); SMPL-X finger order is index, middle, pinky, ring, thumb"""
    from .hand_kin import FINGER_JOINTS
    f0 = FINGER_JOINTS[side][0]
    return f0, f0 + 3, f0 + 9, f0 + 6


def palm_centre(J: np.ndarray, side: str) -> np.ndarray:
    """(T,55,3) -> (T,3): the middle of the palm (wrist and the four knuckles)"""
    from .hand_kin import WRIST
    i, m, r, p = mcp_joints(side)
    return 0.4 * J[:, WRIST[side]] + 0.15 * (J[:, i] + J[:, m] + J[:, r] + J[:, p])


def palm_normal(J: np.ndarray, side: str) -> np.ndarray:
    """(T,55,3) -> (T,3) unit normal out of the palm (volar side): in the SMPL-X rest pose (palms down) it is -Y for both hands"""
    from .hand_kin import WRIST
    i, _, _, p = mcp_joints(side)
    a = J[:, i] - J[:, WRIST[side]]
    b = J[:, p] - J[:, WRIST[side]]
    n = np.cross(a, b) * (-1.0 if side == "left" else 1.0)
    return n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)


def finger_dir(J: np.ndarray, side: str) -> np.ndarray:
    """(T,3) unit direction wrist -> middle knuckle"""
    from .hand_kin import WRIST
    v = J[:, mcp_joints(side)[1]] - J[:, WRIST[side]]
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def hand_ids(rest_verts: np.ndarray, rest_joints: np.ndarray, faces: np.ndarray, step: int = 2) -> dict:
    """Per side: hand surface ids (every `step`-th hand vertex), the palm patch (wrist / knuckle-dominated vertices whose normal points to the
    palm side within 45 deg, within 4.5 cm of the palm centre), the forearm ids (elbow-dominated, distal part)."""
    from .hand_kin import WRIST, ELBOW, hand_vertex_ids
    from .avatar_uv import vertex_normals
    W = skin_weights()
    dom = W.argmax(1)
    N = vertex_normals(rest_verts, faces)
    out = {}
    for s in SIDES:
        mcp = mcp_joints(s)
        hv = hand_vertex_ids(model_dir(), s)
        n = palm_normal(rest_joints[None], s)[0]
        c = palm_centre(rest_joints[None], s)[0]
        cand = np.flatnonzero((dom == WRIST[s]) | np.isin(dom, list(mcp)))
        cand = cand[(N[cand] @ n > 0.7) & (np.linalg.norm(rest_verts[cand] - c, axis=1) < 0.045)]
        el = np.flatnonzero(dom == ELBOW[s])
        wr, eb = rest_joints[WRIST[s]], rest_joints[ELBOW[s]]
        u = ((rest_verts[el] - eb) @ (wr - eb)) / np.dot(wr - eb, wr - eb)
        out[s] = HandIds(hand=hv[::step], palm=cand, arm=el[u > 0.35][::2], mcp=mcp)
    return out


class GpuSurface:
    """Exact closest points and an inside test of query points against a batch of posed partner meshes, on the GPU.
    Candidates = the faces around the k nearest vertices; the sign = generalized winding number (mesh_contact) for points within
    `wn_within` of the surface or on the inner side of the interpolated vertex normal (a cheap first sign), else outside."""

    def __init__(self, faces: np.ndarray, n_verts: int, device: str, k: int = 4, max_deg: int = 10):
        import torch
        from . import mesh_contact as mc
        self.torch = torch
        self.dev = device
        topo = mc.Topology(faces, n_verts, max_deg)
        self.F = torch.as_tensor(topo.F, device=device)
        self.adj = torch.as_tensor(topo.adj, device=device)
        self.k = k

    def vertex_normals(self, V):
        torch = self.torch
        A, B, C = V[:, self.F[:, 0]], V[:, self.F[:, 1]], V[:, self.F[:, 2]]
        fn = torch.cross(B - A, C - A, dim=-1)
        N = torch.zeros_like(V)
        for c in range(3):
            N.index_add_(1, self.F[:, c], fn)
        return N / N.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    def query(self, P, V, wn_within: float = 0.03, far: float = 0.15, N=None, inside_test: bool = True) -> dict:
        """P (B,n,3), V (B,nv,3) float32 tensors -> dict of tensors: dist (B,n), sd (B,n) signed (< 0 inside), vert (B,n) closest vertex of the
        closest face, q (B,n,3) closest point, nq (B,n,3) interpolated normal at q."""
        torch = self.torch
        from .mesh_contact import _pt_tri_sq_torch
        Bn, n = P.shape[:2]
        d = torch.cdist(P, V)                                                          # (B,n,nv)
        _, iv = d.topk(self.k, dim=-1, largest=False)                                   # (B,n,k)
        del d
        cand = self.adj[iv].reshape(Bn, n, -1)                                         # (B,n,k*deg)
        valid = cand >= 0
        cf = torch.where(valid, cand, torch.zeros_like(cand))
        tri = self.F[cf]                                                              # (B,n,m,3)
        bidx = torch.arange(Bn, device=self.dev)[:, None, None, None]
        Vt = V[bidx, tri]                                                             # (B,n,m,3,3)
        p = P[:, :, None, :]
        d2, u, v, w = _pt_tri_sq_torch(torch, p, Vt[..., 0, :], Vt[..., 1, :], Vt[..., 2, :])
        d2 = torch.where(valid, d2, torch.full_like(d2, float("inf")))
        j = d2.argmin(-1, keepdim=True)
        dist = d2.gather(-1, j)[..., 0].clamp_min(0).sqrt()
        bary = torch.stack([u.gather(-1, j)[..., 0], v.gather(-1, j)[..., 0], w.gather(-1, j)[..., 0]], -1)   # (B,n,3)
        face = cf.gather(-1, j)[..., 0]                                                 # (B,n)
        tv = self.F[face]                                                              # (B,n,3)
        bi = torch.arange(Bn, device=self.dev)[:, None, None]
        q = (V[bi, tv] * bary[..., None]).sum(-2)
        if N is None:
            N = self.vertex_normals(V)
        nq = (N[bi, tv] * bary[..., None]).sum(-2)
        nq = nq / nq.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        vert = tv.gather(-1, bary.argmax(-1, keepdim=True))[..., 0]
        side = ((P - q) * nq).sum(-1)
        sd = dist.clone()
        if not inside_test:                                                # the interpolated normal's side only (fast; the refinement's planes)
            return {"dist": dist, "sd": torch.where(side < 0, -dist, dist), "vert": vert, "q": q, "nq": nq}
        need = (dist < far) & ((dist < wn_within) | (side < 0))
        if need.any():
            bb, ii = torch.nonzero(need, as_tuple=True)
            wn = self.winding(P[bb, ii], V, bb)
            sd[bb, ii] = torch.where(wn > 0.5, -dist[bb, ii], dist[bb, ii])
        return {"dist": dist, "sd": sd, "vert": vert, "q": q, "nq": nq}

    def winding(self, Pq, V, bb, chunk: int = 48):
        """generalized winding numbers of points Pq (m,3), point i against mesh bb[i] of the batch (vectorised over points)"""
        torch = self.torch
        out = torch.zeros(len(Pq), device=self.dev, dtype=V.dtype)
        F = self.F
        for i in range(0, len(Pq), chunk):
            b = bb[i:i + chunk]
            p = Pq[i:i + chunk][:, None]
            Vb = V[b]                                                       # (c,nv,3)
            a, bv, c = Vb[:, F[:, 0]] - p, Vb[:, F[:, 1]] - p, Vb[:, F[:, 2]] - p
            la, lb, lc = a.norm(dim=-1), bv.norm(dim=-1), c.norm(dim=-1)
            det = (a * torch.cross(bv, c, dim=-1)).sum(-1)
            den = la * lb * lc + (a * bv).sum(-1) * lc + (bv * c).sum(-1) * la + (c * a).sum(-1) * lb
            out[i:i + chunk] = (2 * torch.atan2(det, den)).sum(1) / (4 * np.pi)
        return out


def pose_meshes(fit: dict, device: str, ks: np.ndarray | None = None, batch: int = 256) -> dict:
    """{role: (verts (n,V,3) float32, joints (n,55,3))} of the fit (all frames, or ks)"""
    out = {}
    for r in ROLES:
        b = Body(fit[r], device)
        P = full_pose(fit[r])
        tr = np.asarray(fit[r]["transl"], float)
        if ks is not None:
            P, tr = P[ks], tr[ks]
        fw = b.forward_np(P, tr, verts=True, batch=batch)
        out[r] = (fw["verts"], fw["joints"])
    return out


_IDS_CACHE: dict = {}


def rest_hand_ids(betas: np.ndarray) -> dict:
    key = tuple(np.round(np.asarray(betas, float).ravel(), 6))
    if key not in _IDS_CACHE:
        b0 = Body({"betas": betas, "t": np.zeros(1)}, "cpu")
        rest = b0.forward_np(np.zeros((1, 55, 3)), np.zeros((1, 3)))
        _IDS_CACHE[key] = hand_ids(rest["verts"][0].astype(float), rest["joints"][0], b0.faces)
    return _IDS_CACHE[key]


def hand_geometry(fit: dict, device: str = "cuda", batch: int = 24, meshes: dict | None = None, ks: np.ndarray | None = None,
                  roles=ROLES, sides=SIDES) -> dict:
    """Per (role, side) and grid frame: the hand's signed gap to the partner's surface and what it touches.
    -> {(r, s): {gap (n,) signed min over the hand surface (< 0 = sunk), pen (n,) deepest hand point, arm_pen (n,) deepest forearm point,
       n_close (n,) hand points within 1.5 cm, palm_gap (n,) signed min over the palm patch, region (n,) REGION id of the partner point closest
       to the hand, q (n,3) that point, nq (n,3) the partner's surface normal there, pvert (n,), partner_side (n,) 0/1 left/right when the region
       is a hand, else -1, hp (n,) index (into ids.hand) of the closest hand point}"""
    import torch
    from .hand_kin import WRIST, FINGER_JOINTS
    T = len(fit["lead"]["t"]) if ks is None else len(ks)
    if meshes is None:
        meshes = pose_meshes(fit, device, ks)
    faces = Body(fit["lead"], "cpu").faces
    vj = vertex_joint()
    hand_side_of_vert = np.full(len(vj), -1, np.int8)
    for si, s in enumerate(SIDES):
        hand_side_of_vert[np.isin(vj, [WRIST[s], *FINGER_JOINTS[s]])] = si
    surf = GpuSurface(faces, len(vj), device)
    res = {}
    for r in roles:
        o = OTHER[r]
        ids = rest_hand_ids(fit[r]["betas"])
        Vr, Jr = meshes[r]
        Vo, Jo = meshes[o]
        fwd_o = torso_forward(Jo)
        for s in sides:
            e = {"gap": np.full(T, np.nan), "pen": np.zeros(T), "arm_pen": np.zeros(T), "n_close": np.zeros(T, np.int16),
                 "palm_gap": np.full(T, np.nan), "region": np.zeros(T, np.int8), "q": np.zeros((T, 3)), "nq": np.zeros((T, 3)),
                 "pvert": np.zeros(T, np.int32), "partner_side": np.full(T, -1, np.int8), "hp": np.zeros(T, np.int16)}
            idx = ids[s]
            allp = np.concatenate([idx.hand, idx.palm, idx.arm])
            nh, npm = len(idx.hand), len(idx.palm)
            with torch.no_grad():
                for a in range(0, T, batch):
                    sl = slice(a, min(T, a + batch))
                    P = torch.as_tensor(Vr[sl][:, allp], device=device)
                    V = torch.as_tensor(Vo[sl], device=device)
                    qr = surf.query(P, V)
                    sd = qr["sd"].cpu().numpy()
                    h = sd[:, :nh]
                    m = h.argmin(1)
                    rows = np.arange(len(m))
                    e["gap"][sl] = h[rows, m]
                    e["hp"][sl] = m
                    e["pen"][sl] = np.maximum(0.0, -h.min(1))
                    e["n_close"][sl] = (h < 0.015).sum(1)
                    e["palm_gap"][sl] = sd[:, nh:nh + npm].min(1)
                    e["arm_pen"][sl] = np.maximum(0.0, -sd[:, nh + npm:].min(1))
                    q = qr["q"].cpu().numpy()[rows, m]
                    nq = qr["nq"].cpu().numpy()[rows, m]
                    pv = qr["vert"].cpu().numpy()[rows, m]
                    e["q"][sl], e["nq"][sl], e["pvert"][sl] = q, nq, pv
                    e["region"][sl] = region_of(vj[pv], (nq * fwd_o[sl]).sum(1))
                    e["partner_side"][sl] = np.where(e["region"][sl] == REGION_ID["hand"], hand_side_of_vert[pv], -1)
            res[(r, s)] = e
    return res


# ------------------------------------------------------------------------------------------------ 2D evidence

BASE2D = ("hand", "forearm", "upper_arm", "girdle", "torso", "neck_head", "low")      # coarse partner regions for the 2D vote
_B2 = {"hand": 0, "forearm": 1, "upper_arm": 2, "girdle": 3, "spine3": 4, "spine2": 4, "spine1": 4, "pelvis": 6, "neck_head": 5, "hip": 6,
       "leg": 6, "none": 6}


def landmark_regions() -> np.ndarray:
    """(512,) coarse region (BASE2D index) of each MAMMA landmark"""
    vj = vertex_joint()
    lv = landmark_matrix().argmax(1)
    return np.array([_B2[REGION_BASE.get(int(j), "none")] for j in vj[lv]], np.int8)


@dataclass
class EvidenceCfg:
    touch_len: float = 0.25      # a view supports contact: some visible hand landmark of A within this x A's hand length of a visible landmark of B
    touch_min_px: float = 3.0
    gap_len: float = 1.0         # a view contradicts contact: every visible hand landmark further than this x hand length from every B landmark
    min_hand_pts: int = 5
    min_partner_pts: int = 20
    vis: float = 0.5


def view_evidence(fit: dict, obs: Obs, cfg: EvidenceCfg | None = None, device: str = "cuda", batch: int = 512) -> dict:
    """Per (role, side): verdict (T, n_cams) int8 (+1 the 2D hand touches the partner's 2D body, -1 a clear 2D gap, 0 no verdict), gap_len
    (T, n_cams) the 2D distance / hand length, region2d (T, n_cams) coarse BASE2D region of the nearest partner landmark, mamma_contact (T, n_cams)
    MAMMA's own contact channel (mean of the 8 highest hand landmarks)."""
    import torch
    from .hand_kin import WRIST
    cfg = cfg or EvidenceCfg()
    parts = landmark_parts()
    lreg = torch.as_tensor(landmark_regions().astype(np.int64), device=device)
    upper = np.flatnonzero(landmark_regions() < 6)
    T = len(fit["lead"]["t"])
    C = len(obs.cams)
    out = {}
    for r in ROLES:
        o = OTHER[r]
        for s in SIDES:
            hid = parts[f"{s}_hand"]
            ver = np.zeros((T, C), np.int8)
            gl = np.full((T, C), np.nan, np.float32)
            rg = np.full((T, C), -1, np.int8)
            mc = np.full((T, C), np.nan, np.float32)
            J = fit[r]["joints"]
            tip = J[:, mcp_joints(s)[1] + 2]                  # middle3 (distal joint)
            for ci, c in enumerate(obs.cams):
                z = obs.z[c]
                ks = np.arange(T)
                uv, dep = obs.project(c, np.stack([J[:, WRIST[s]], tip], 1), ks)
                L = np.maximum(np.linalg.norm(uv[:, 0] - uv[:, 1], axis=-1) / 0.85, 6.0)
                camok = z["ok"] & (dep > 0.1).all(1)
                for a in range(0, T, batch):
                    sl = slice(a, min(T, a + batch))
                    A = torch.as_tensor(z["lm"][sl, PIDX[r]][:, hid].astype(np.float32), device=device)          # (b,63,2)
                    Av = torch.as_tensor(z["vis"][sl, PIDX[r]][:, hid].astype(np.float32), device=device) > cfg.vis
                    Bp = torch.as_tensor(z["lm"][sl, PIDX[o]][:, upper].astype(np.float32), device=device)       # (b,nu,2)
                    Bv = torch.as_tensor(z["vis"][sl, PIDX[o]][:, upper].astype(np.float32), device=device) > cfg.vis
                    d = torch.cdist(A, Bp)                                                                       # (b,63,nu)
                    d = torch.where(Av[:, :, None] & Bv[:, None, :], d, torch.full_like(d, float("inf")))
                    dmin, j = d.reshape(len(d), -1).min(1)
                    jb = j % Bp.shape[1]
                    ok = (Av.sum(1) >= cfg.min_hand_pts) & (Bv.sum(1) >= cfg.min_partner_pts) & torch.as_tensor(camok[sl], device=device)
                    Lt = torch.as_tensor(L[sl].astype(np.float32), device=device)
                    g = (dmin / Lt).cpu().numpy()
                    okn = ok.cpu().numpy()
                    v = np.where(okn & ((dmin <= torch.maximum(Lt * cfg.touch_len, torch.full_like(Lt, cfg.touch_min_px))).cpu().numpy()), 1,
                                 np.where(okn & (g >= cfg.gap_len), -1, 0))
                    ver[sl, ci] = v
                    gl[sl, ci] = np.where(okn, g, np.nan)
                    rg[sl, ci] = np.where(okn, lreg[torch.as_tensor(upper, device=device)[jb]].cpu().numpy(), -1)
                    con = torch.as_tensor(z["con"][sl, PIDX[r]][:, hid].astype(np.float32), device=device)
                    top = con.topk(8, dim=1).values.mean(1).cpu().numpy()
                    mc[sl, ci] = np.where(z["ok"][sl], top, np.nan)
            out[(r, s)] = {"verdict": ver, "gap_len": gl, "region2d": rg, "mamma_contact": mc}
            log(f"2D evidence {r} {s}: views for {np.mean((ver > 0).sum(1)):.2f}, against {np.mean((ver < 0).sum(1)):.2f} per frame")
    return out


# ------------------------------------------------------------------------------------------------ series + events

@dataclass
class EventCfg:
    near_full_m: float = 0.015     # 3D: score 1 at / below this signed gap (sunk counts as 1)
    near_zero_m: float = 0.045     # ... 0 at / above
    far_full_m: float = 0.05       # a larger gap counts only with strong 2D support (>= 2 views for, evidence >= 0.75):
    far_zero_m: float = 0.11       #   the fit's hand can be a few cm off a real touch
    on: float = 0.55
    off: float = 0.30
    min_frames: int = 6            # 0.2 s
    merge_gap: int = 4
    hold_min_s: float = 0.4        # a hold lasts at least this long (else a touch)
    smooth: int = 2                # median filter half-width on the score


def _medfilt(x: np.ndarray, h: int) -> np.ndarray:
    if h <= 0:
        return x
    pad = np.pad(x, h, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(pad, 2 * h + 1), axis=-1)


def score_series(geo: dict, ev: dict, cfg: EventCfg | None = None) -> dict:
    cfg = cfg or EventCfg()
    out = {}
    for key, g in geo.items():
        v = ev[key]["verdict"]
        pos, neg = (v > 0).sum(1), (v < 0).sum(1)
        evid = np.where(pos + neg > 0, pos / np.maximum(pos + neg, 1), 0.5)
        gap = g["gap"]
        near = np.clip((cfg.near_zero_m - gap) / (cfg.near_zero_m - cfg.near_full_m), 0, 1) * np.clip(evid / 0.35, 0, 1)
        far = np.clip((cfg.far_zero_m - gap) / (cfg.far_zero_m - cfg.far_full_m), 0, 1) * np.clip((evid - 0.6) / 0.3, 0, 1) * (pos >= 2)
        sc = _medfilt(np.maximum(near, far), cfg.smooth)
        mcon = np.nanmedian(np.where(np.isfinite(ev[key]["mamma_contact"]), ev[key]["mamma_contact"], np.nan), axis=1) \
            if np.isfinite(ev[key]["mamma_contact"]).any() else np.full(len(gap), np.nan)
        out[key] = {"score": sc, "evidence": evid, "views_for": pos, "views_against": neg, "mamma_contact": mcon}
    return out


def hold_subtype(Ja: np.ndarray, sa: str, Jb: np.ndarray, sb: str) -> str:
    """hand-hand hold: palms facing (normals opposed) and fingers the same way = palm-to-palm; palms facing and fingers opposed or crossed =
    handshake / grip; otherwise one hand over the back of the other (hand_on_hand)"""
    na, nb = palm_normal(Ja, sa), palm_normal(Jb, sb)
    fa, fb = finger_dir(Ja, sa), finger_dir(Jb, sb)
    nn = float(np.median((na * nb).sum(1)))
    ff = float(np.median((fa * fb).sum(1)))
    if nn < -0.35:
        return "palm_to_palm" if ff > 0.5 else "handshake"
    return "hand_on_hand"


def detect_events(fit: dict, geo: dict, ev: dict, cfg: EventCfg | None = None) -> tuple[list, dict]:
    from .counterbalance import hysteresis
    cfg = cfg or EventCfg()
    series = score_series(geo, ev, cfg)
    t = np.asarray(fit["lead"]["t"], float)
    events = []
    for (r, s), g in geo.items():
        se = series[(r, s)]
        for a, b in hysteresis(se["score"], cfg.on, cfg.off, cfg.min_frames, cfg.merge_gap):
            sl = slice(a, b + 1)
            reg = g["region"][sl]
            close = g["gap"][sl] < 0.06
            cnt = np.bincount(reg[close] if close.any() else reg, minlength=len(REGIONS))
            cnt[0] = 0
            region = REGIONS[int(cnt.argmax())] if cnt.sum() else "none"
            hold = HOLD_OF_REGION.get(region, "none")
            ps = None
            if region == "hand":
                pss = g["partner_side"][sl][reg == REGION_ID["hand"]]
                pss = pss[pss >= 0]
                ps = SIDES[int(np.bincount(pss, minlength=2).argmax())] if len(pss) else None
                if ps is not None:
                    hold = hold_subtype(fit[r]["joints"][sl], s, fit[OTHER[r]]["joints"][sl], ps)
            v2 = ev[(r, s)]["region2d"][sl]
            v2 = v2[(ev[(r, s)]["verdict"][sl] > 0) & (v2 >= 0)]
            reg2d = BASE2D[int(np.bincount(v2, minlength=len(BASE2D)).argmax())] if len(v2) else None
            dur = (b - a + 1) / HZ
            events.append({"role": r, "side": s, "partner": OTHER[r], "region": region, "hold": hold, "partner_side": ps,
                           "region_2d_vote": reg2d, "kind": "hold" if dur >= cfg.hold_min_s else "touch",
                           "i0": int(a), "i1": int(b), "t0": round(float(t[a]), 3), "t1": round(float(t[b]), 3), "duration_s": round(dur, 3),
                           "gap_cm_median": round(float(np.median(g["gap"][sl])) * 100, 2),
                           "gap_cm_min": round(float(g["gap"][sl].min()) * 100, 2),
                           "palm_gap_cm_median": round(float(np.median(g["palm_gap"][sl])) * 100, 2),
                           "pen_cm_max": round(float(g["pen"][sl].max()) * 100, 2),
                           "pen_cm_median": round(float(np.median(g["pen"][sl])) * 100, 2),
                           "arm_pen_cm_max": round(float(g["arm_pen"][sl].max()) * 100, 2),
                           "evidence_median": round(float(np.median(se["evidence"][sl])), 3),
                           "views_for_median": float(np.median(se["views_for"][sl])),
                           "views_against_median": float(np.median(se["views_against"][sl])),
                           "mamma_contact_median": round(float(np.nanmedian(se["mamma_contact"][sl])), 3)
                           if np.isfinite(se["mamma_contact"][sl]).any() else None,
                           "score_mean": round(float(se["score"][sl].mean()), 3)})
    events.sort(key=lambda e: (e["t0"], e["role"], e["side"]))
    # pair the two records of one hand-hand hold
    for i, e in enumerate(events):
        e["id"] = i
    for e in events:
        if e["region"] != "hand" or e["partner_side"] is None:
            continue
        for f in events:
            if f is e or f["role"] != e["partner"] or f["side"] != e["partner_side"] or f["region"] != "hand":
                continue
            if f["i0"] <= e["i1"] and e["i0"] <= f["i1"]:
                e["twin"] = f["id"]
    return events, series


def run_detect(device: str = "cuda") -> dict:
    from . import gpu
    fit = load_fit()
    obs = Obs()
    t0 = time.time()
    with gpu.exclusive("hand_contact detect"):
        meshes = pose_meshes(fit, device)
        log(f"meshes posed in {time.time() - t0:.0f} s")
        geo = hand_geometry(fit, device, meshes=meshes)
        log(f"geometry in {time.time() - t0:.0f} s")
        ev = view_evidence(fit, obs, device=device)
        log(f"2D evidence in {time.time() - t0:.0f} s")
    events, series = detect_events(fit, geo, ev)
    arr = {}
    for (r, s), g in geo.items():
        for k, v in g.items():
            arr[f"{r}_{s}_{k}"] = v
        for k, v in ev[(r, s)].items():
            arr[f"{r}_{s}_{k}"] = v
        for k, v in series[(r, s)].items():
            arr[f"{r}_{s}_s_{k}"] = v
    np.savez_compressed(OUT / "series.npz", **arr, cams=np.array(obs.cams), regions=np.array(REGIONS))
    summ = summarise_events(events)
    write_json(OUT / "events.json", {"what": "hand-partner contact events of the recap fit (smplx_rf2_noslip): mesh gap + 6-phone 2D evidence + hysteresis",
                                     "config": {"event": asdict(EventCfg()), "evidence": asdict(EvidenceCfg())},
                                     "summary": summ, "events": events, "seconds": round(time.time() - t0, 1)})
    log(f"{len(events)} events ({summ['holds']} holds) in {time.time() - t0:.0f} s -> {OUT / 'events.json'}")
    return {"events": events, "summary": summ}


def summarise_events(events: list) -> dict:
    from collections import Counter
    holds = [e for e in events if e["kind"] == "hold"]
    return {"events": len(events), "holds": len(holds),
            "by_hold": dict(Counter(e["hold"] for e in events)),
            "by_hand": dict(Counter(f"{e['role']}_{e['side']}" for e in events)),
            "seconds_in_contact": round(sum(e["duration_s"] for e in events), 1),
            "sunk_gt_1cm": sum(e["pen_cm_max"] > 1.0 for e in events),
            "floating_gt_1cm": sum(e["gap_cm_median"] > 1.0 for e in events)}


# ------------------------------------------------------------------------------------------------ phone frames + mesh overlays

class Phones:
    """Decoded phone frames at grid frames: source video + the owning chunk's video offset (the same clock as the fit's 2D input)."""

    def __init__(self):
        from .take import Take
        self.take = Take.load(TAKE)
        self.cam = {}
        self.offs = {}
        st = json.loads((WORK / "stitch.json").read_text(encoding="utf-8"))
        for ch in st["chunks"]:
            self.offs[ch["name"]] = _offsets(_chunk_work(ch["name"]))
        self.owner = np.asarray(load_fit()["lead"]["segment_owner"]).astype(str)

    def _c(self, cam: str) -> dict:
        if cam not in self.cam:
            z = np.load(_camera_file(cam), allow_pickle=True)
            self.cam[cam] = {"t": np.asarray(z["t"], float), "pts": np.asarray(z["pts"], float), "k1": np.asarray(z["dist"], float).ravel(),
                             "size": tuple(int(x) for x in z["native_size"]), "src": self.take.source(cam)}
        return self.cam[cam]

    def frame(self, cam: str, k: int) -> tuple[np.ndarray | None, float, float]:
        """(BGR upright frame, k1 of that frame, |time error| s) of grid frame k"""
        from . import mamma_dyn as md
        from .front_full_review import grab
        c = self._c(cam)
        tk = k / HZ
        off = self.offs.get(self.owner[k], {}).get(cam)
        tc = md.apply_offset(c["t"], off)
        i = int(np.argmin(np.abs(tc - tk)))
        W, H = c["size"]
        img = grab(c["src"].path, float(c["pts"][i]), W, H)
        return img, float(c["k1"][i]), float(abs(tc[i] - tk))


def render_faces(V: np.ndarray, F: np.ndarray, fcol: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray, H: int, W: int):
    """V (n,3) world, F (m,3), per-face rgb (m,3) 0..1 -> (H,W,3) float image (Lambert, headlight), coverage mask, depth"""
    import torch
    from .avatar_raster import rasterize
    Xc = V @ R.T + t
    z = Xc[:, 2]
    px = np.stack([K[0, 0] * Xc[:, 0] / np.maximum(z, 1e-4) + K[0, 2], K[1, 1] * Xc[:, 1] / np.maximum(z, 1e-4) + K[1, 2]], 1)
    zb, fid = rasterize(torch, torch.tensor(px), torch.tensor(z), torch.tensor(F), H, W)
    fid = fid.numpy()
    tri = Xc[F]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    view = -tri.mean(1)
    view /= np.maximum(np.linalg.norm(view, axis=1, keepdims=True), 1e-12)
    shade = 0.3 + 0.7 * np.abs((n * view).sum(1))
    img = np.zeros((H, W, 3))
    cov = fid >= 0
    img[cov] = fcol[fid[cov]] * shade[fid[cov], None]
    return img, cov


def crop_K(K: np.ndarray, cx: float, cy: float, side: float, tile: int) -> np.ndarray:
    """intrinsics of a square crop of `side` px centred at (cx, cy) resampled to tile x tile"""
    s = tile / side
    Kc = K.copy().astype(float)
    Kc[0, 0] *= s
    Kc[1, 1] *= s
    Kc[0, 2] = (K[0, 2] - (cx - side / 2)) * s
    Kc[1, 2] = (K[1, 2] - (cy - side / 2)) * s
    return Kc


def crop_img(img: np.ndarray, cx: float, cy: float, side: float, tile: int, K: np.ndarray, k1: float) -> np.ndarray:
    """the undistorted (pinhole, MAMMA's frame) square crop around (cx, cy) of the raw phone frame"""
    import cv2
    Kc = crop_K(K, cx, cy, side, tile)
    u, v = np.meshgrid(np.arange(tile, dtype=np.float32), np.arange(tile, dtype=np.float32))
    x = (u - Kc[0, 2]) / Kc[0, 0]
    y = (v - Kc[1, 2]) / Kc[1, 1]
    r2 = x * x + y * y
    fac = 1 + k1 * r2
    mx = (K[0, 0] * x * fac + K[0, 2]).astype(np.float32)
    my = (K[1, 1] * y * fac + K[1, 2]).astype(np.float32)
    return cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(30, 30, 30))


ROLE_RGB = {"lead": (0.25, 0.45, 0.95), "follow": (0.95, 0.55, 0.2)}
HAND_RGB = (1.0, 0.92, 0.15)


def face_colours(faces: np.ndarray, role: str, side: str | None) -> np.ndarray:
    """per-face colour of a dancer: role colour, the acting hand (+ wrist) in yellow"""
    from .hand_kin import WRIST, FINGER_JOINTS
    c = np.tile(np.asarray(ROLE_RGB[role]), (len(faces), 1))
    if side is not None:
        vj = vertex_joint()
        hv = np.isin(vj, [WRIST[side], *FINGER_JOINTS[side]])
        c[hv[faces].any(1)] = HAND_RGB
    return c


def overlay_tile(img_crop: np.ndarray, rend: np.ndarray, cov: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    out = img_crop.astype(float).copy()
    rb = (rend[..., ::-1] * 255.0)
    out[cov] = (1 - alpha) * out[cov] + alpha * rb[cov]
    return np.clip(out, 0, 255).astype(np.uint8)


def pick_cams(obs: Obs, ev_series: dict, key: tuple, k: int, point: np.ndarray, n: int = 2, exclude_bad: bool = True) -> list:
    """cameras in which the contact point is inside the frame, the camera is ok and the hand is visible (2D verdict != 0 preferred),
    preferring large hand size and different viewing directions"""
    from .recap_pov import bad_stretches, overlaps_bad
    bad = bad_stretches() if exclude_bad else {}
    cands = []
    for ci, c in enumerate(obs.cams):
        z = obs.z[c]
        if not z["ok"][k]:
            continue
        if exclude_bad and overlaps_bad(bad.get(c, []), k / HZ, k / HZ + 0.01):
            continue
        uv, d = obs.project(c, point[None, None], np.array([k]))
        Wd, Hd = [int(x) for x in z["size"]]
        u, v = uv[0, 0]
        if d[0, 0] <= 0.2 or not (0.08 * Wd < u < 0.92 * Wd and 0.05 * Hd < v < 0.95 * Hd):
            continue
        f = z["K"][k][0, 0]
        size = f / d[0, 0]
        vis = 1.0 if ev_series is None else (1.0 if ev_series[key]["verdict"][k, ci] != 0 else 0.5)
        C = -z["R"][k].T @ z["t"][k]
        dirn = (point - C) / np.linalg.norm(point - C)
        cands.append((size * vis, c, dirn))
    cands.sort(key=lambda x: -x[0])
    out = []
    for sc, c, d in cands:
        if all(float(d @ d2) < 0.8 for _, d2 in out):
            out.append((c, d))
        if len(out) >= n:
            break
    for sc, c, d in cands:
        if len(out) >= n:
            break
        if c not in [x[0] for x in out]:
            out.append((c, d))
    return [c for c, _ in out]


def label_tile(img: np.ndarray, lines: list[str], scale: float = 0.42) -> np.ndarray:
    import cv2
    out = img.copy()
    y = 14
    for ln in lines:
        cv2.putText(out, ln, (4, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, ln, (4, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
        y += int(32 * scale) + 2
    return out


def event_tiles(ev: dict, k: int, fits: list[tuple[str, dict]], obs: Obs, phones: Phones, cams: list[str], tile: int = 300,
                device: str = "cpu", point: np.ndarray | None = None) -> list[np.ndarray]:
    """For each camera: [raw crop, overlay fit 1, overlay fit 2 ...] at grid frame k around the contact point."""
    from .hand_kin import WRIST
    r, s = ev["role"], ev["side"]
    tiles = []
    posed = []
    for name, fit in fits:
        b = {rr: Body(fit[rr], device) for rr in ROLES}
        fw = {rr: b[rr].forward_np(full_pose(fit[rr])[k:k + 1], fit[rr]["transl"][k:k + 1]) for rr in ROLES}
        posed.append((name, fw, b["lead"].faces))
    J0 = posed[0][1][r]["joints"][0]
    if point is None:
        point = palm_centre(J0[None], s)[0]
    for c in cams:
        z = obs.z[c]
        img, k1, terr = phones.frame(c, k)
        K, R, t = z["K"][k], z["R"][k], z["t"][k]
        uv, d = obs.project(c, point[None, None], np.array([k]))
        cx, cy = uv[0, 0]
        hl = np.linalg.norm(J0[WRIST[s]] - J0[mcp_joints(s)[1] + 2]) / 0.85
        side_px = max(110.0, 5.5 * hl * K[0, 0] / d[0, 0])
        if img is None:
            raw = np.full((tile, tile, 3), 40, np.uint8)
        else:
            raw = crop_img(img, cx, cy, side_px, tile, K, k1)
        Kc = crop_K(K, cx, cy, side_px, tile)
        row = [label_tile(raw, [f"cam {c}  t {k / HZ:.2f} s", f"video {terr * 1000:.0f} ms"])]
        for name, fw, faces in posed:
            Vs = np.concatenate([fw["lead"]["verts"][0], fw["follow"]["verts"][0]]).astype(float)
            nv = len(fw["lead"]["verts"][0])
            Fs = np.concatenate([faces, faces + nv])
            col = np.concatenate([face_colours(faces, "lead", s if r == "lead" else None),
                                  face_colours(faces, "follow", s if r == "follow" else None)])
            rend, cov = render_faces(Vs, Fs, col, Kc, R, t, tile, tile)
            row.append(label_tile(overlay_tile(raw, rend, cov), [name]))
        tiles.append(row)
    return tiles


def contact_sheet(events: list, fits: list[tuple[str, dict]], path: Path, title: str, tile: int = 260, n_cams: int = 2,
                  frame_of=None) -> Path:
    """One row per event: per camera the raw phone crop + the overlays of each fit."""
    import cv2
    obs = Obs()
    phones = Phones()
    ser = np.load(OUT / "series.npz")
    ev2d = {(r, s): {"verdict": ser[f"{r}_{s}_verdict"]} for r in ROLES for s in SIDES}
    rows = []
    for ev in events:
        k = frame_of(ev) if frame_of else (ev["i0"] + ev["i1"]) // 2
        fit0 = fits[0][1]
        J = fit0[ev["role"]]["joints"][k:k + 1]
        pt = palm_centre(J, ev["side"])[0]
        cams = pick_cams(obs, ev2d, (ev["role"], ev["side"]), k, pt, n_cams)
        if not cams:
            continue
        tl = event_tiles(ev, k, fits, obs, phones, cams, tile, point=pt)
        cells = [x for row in tl for x in row]
        while len(cells) < n_cams * (1 + len(fits)):
            cells.append(np.full((tile, tile, 3), 30, np.uint8))
        head = np.full((tile, 230, 3), 245, np.uint8)
        lines = [f"#{ev['id']}  {ev['role']} {ev['side']} hand", f"-> {ev['partner']} {ev['region']}", f"{ev['hold']}",
                 f"{ev['t0']:.2f}-{ev['t1']:.2f} s ({ev['kind']})", f"frame {k}",
                 f"gap med {ev['gap_cm_median']:.1f} cm", f"pen max {ev['pen_cm_max']:.1f} cm",
                 f"2D ev {ev['evidence_median']:.2f} ({ev['views_for_median']:.0f}+/{ev['views_against_median']:.0f}-)",
                 f"2D region {ev.get('region_2d_vote')}"]
        for extra in ev.get("_sheet_lines", []):
            lines.append(extra)
        y = 20
        for ln in lines:
            cv2.putText(head, ln, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 20, 20), 1, cv2.LINE_AA)
            y += 17
        rows.append(np.concatenate([head] + cells, 1))
    if not rows:
        return path
    W = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 4), (0, W - r.shape[1]), (0, 0)), constant_values=255) for r in rows]
    top = np.full((34, W, 3), 255, np.uint8)
    cv2.putText(top, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA)
    sheet = np.concatenate([top] + rows, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return path


# ------------------------------------------------------------------------------------------------ the contact refinement
#
# Why it is more than "snap the hand onto the partner" (contact sheet of 20 events, 2026-10-21): in the torso holds the fit's hand is not a few
# millimetres off, it is 10-20 cm off and INSIDE the partner (Kadu's right hand on Larissa's back is fitted in the front of her chest; his hand
# on her sternum is fitted 3 cm inside her chest with the palm turned away). MAMMA's own hand landmarks are poor there (the hand is hidden by
# the partner's body / mask: 23-72 px off in the views that still report it), but the owner's 17-joint COCO detections see the wrists and the
# elbows in 2-4 phones and triangulate to within a few cm of the video. So each event is solved in two stages:
#   stage 1  arm IK (shoulder, elbow, wrist) to the multi-view COCO wrist + elbow (bias-corrected, robust, smoothed) with MAMMA's hand landmarks
#            as a weak 2D term, keeping the arm out of the partner (hand, forearm and upper arm against the partner's surface);
#   stage 2  the contact: the palm onto the partner's surface point nearest the stage-1 palm (smoothed in the partner's frame), the palm facing
#            the surface (a second start with the forearm twisted when the palm faces away by more than 100 deg), the finger template of the hold
#            with per-finger curl, fingertips drawn onto the surface, no penetration; hand-hand holds move both hands symmetrically.
# The deltas are blended in / out with raised-cosine ramps; everything else of the pose is untouched; the partner is never moved for a palm hold.

# finger templates: per-finger curl c (finger pose = c x flex + fixed, finger_template; SMPL-X order index, middle, pinky, ring, thumb), its start
# value and the range the optimiser may use; tips = fingertips are drawn onto the partner's surface (wrap / rest) where they are near it
TEMPLATES = {
    "relaxed": {"c0": 1.0, "lo": 0.6, "hi": 1.5, "tips": True},     # resting on a shoulder / neck / upper arm (flex = MANO's mean)
    "wrap": {"c0": 1.45, "lo": 1.0, "hi": 2.0, "tips": True},       # around a forearm / the partner's hand (flex = MANO's mean)
    # flat on a back / shoulder blade / waist / chest, palm to palm: a flat hand with the fingers closed together. 2026-10-21 review: the
    # first version (0.45 x MANO's mean) kept the flat SMPL-X hand's splay (index 13 deg, pinky 31 deg off the middle finger): fingers fanned
    # out on the partner's back. c scales the flexion part only (c = 1: OPEN_FLEX x the mean's flexion); the closing is fixed.
    "open": {"c0": 1.0, "lo": 0.3, "hi": 2.2, "tips": True},
}
OPEN_FLEX = 0.35          # c = 1 of the open template = this fraction of MANO's mean flexion (knuckles ~10 deg, middle joints ~15 deg)
OPEN_CLOSE = 1.2          # index / ring / pinky knuckles turned toward the middle finger by this x their rest-pose spread (fingers touching)
OPEN_THUMB_DEG = 25.0     # the thumb turned this far toward the index about the palm normal (alongside the index, not sticking out)
# flat-hand holds: the palm patch itself lies on the partner (palm contact, not fingertips only; 2026-10-21 review: palms hovered 2 cm)
FLAT_HOLDS = {"hand_on_back", "hand_on_shoulder_blade", "hand_on_chest", "hand_on_waist"}
_TMPL_CACHE: dict = {}


def finger_template(betas: np.ndarray, side: str, template: str) -> tuple[np.ndarray, np.ndarray]:
    """(flex (15,3), fixed (15,3)) local axis-angle of a hold's finger template; the finger pose is c x flex + fixed (c per finger).
    relaxed / wrap: flex = MANO's mean relaxed pose, fixed = 0 (unchanged). open: flex = OPEN_FLEX x the flexion part of the mean (each joint's
    rotation about its finger's flexion axis = finger direction x palm normal at rest; the mean's abduction and twist are dropped), fixed =
    the index / ring / pinky knuckles turned about the palm normal toward the middle finger by OPEN_CLOSE x their rest-pose spread, and the
    thumb turned OPEN_THUMB_DEG toward the index. Rest-pose axes are the local axes (every global rotation is the identity at rest)."""
    from .hand_kin import FINGER_JOINTS, model_npz
    key = (tuple(np.round(np.asarray(betas, float).ravel(), 6)), side, template)
    if key in _TMPL_CACHE:
        return _TMPL_CACHE[key]
    m = np.load(model_npz(model_dir()), allow_pickle=True)
    mean = np.asarray(m["hands_meanl" if side == "left" else "hands_meanr"], float).reshape(15, 3)
    if template != "open":
        out = (mean, np.zeros((15, 3)))
        _TMPL_CACHE[key] = out
        return out
    b0 = Body({"betas": betas, "t": np.zeros(1)}, "cpu")
    J = b0.forward_np(np.zeros((1, 55, 3)), np.zeros((1, 3)), verts=False)["joints"][0]
    n = palm_normal(J[None], side)[0]
    fj = FINGER_JOINTS[side]

    def in_plane(f):
        d = J[fj[3 * f + 1]] - J[fj[3 * f]]
        d = d - (d @ n) * n
        return d / np.linalg.norm(d)
    dm = in_plane(1)
    flex, fixed = np.zeros((15, 3)), np.zeros((15, 3))
    for f in range(5):
        d = J[fj[3 * f + 1]] - J[fj[3 * f]]
        d /= np.linalg.norm(d)
        a = np.cross(d, n)
        a /= np.linalg.norm(a)                                          # rotating d about +a moves it toward the palm side (flexion)
        for q in range(3):
            flex[3 * f + q] = OPEN_FLEX * (mean[3 * f + q] @ a) * a
        if f in (0, 2, 3):                                              # index, pinky, ring: close toward the middle finger
            dfp = in_plane(f)
            spread = np.arctan2(np.cross(dfp, dm) @ n, dfp @ dm)
            fixed[3 * f] = OPEN_CLOSE * spread * n
        elif f == 4:                                                    # the thumb toward the index (the sense mirrors with the side)
            dt, di = in_plane(4), in_plane(0)
            fixed[3 * f] = np.sign(np.cross(dt, di) @ n) * np.radians(OPEN_THUMB_DEG) * n
    _TMPL_CACHE[key] = (flex, fixed)
    return flex, fixed
TEMPLATE_OF_HOLD = {"hand_on_shoulder": "relaxed", "hand_on_neck_head": "relaxed", "hand_on_upper_arm": "relaxed", "hand_on_hand": "relaxed",
                    "hand_on_forearm": "wrap", "handshake": "wrap", "hand_hold": "wrap",
                    "palm_to_palm": "open", "hand_on_back": "open", "hand_on_shoulder_blade": "open", "hand_on_waist": "open",
                    "hand_on_chest": "open", "hand_on_leg": "open"}
PALM_HOLDS = {"hand_on_shoulder", "hand_on_neck_head", "hand_on_upper_arm", "hand_on_forearm", "hand_on_back", "hand_on_shoulder_blade",
              "hand_on_waist", "hand_on_chest"}
HAND_HOLDS = {"handshake", "palm_to_palm", "hand_on_hand", "hand_hold"}
COCO_JOINT = {"wrist": {"left": 9, "right": 10}, "elbow": {"left": 7, "right": 8}}


@dataclass
class RefineCfg:
    ramp: int = 12                 # blend-in / blend-out frames (raised cosine) outside the event
    iters1: int = 60              # stage 1: arm to the multi-view wrist / elbow
    iters2: int = 90              # stage 2: contact + fingers
    iters_polish: int = 10
    lr: float = 0.02
    lr2: float = 0.03
    refresh: int = 15              # partner surface planes recomputed every this many iterations
    palm_offset_m: float = 0.004   # the palm patch centroid sits this far off the partner's surface (skin + cloth)
    hand_gap_m: float = 0.003      # hand-hand: the soft-min gap between the two hands' surfaces
    margin_m: float = 0.0015       # hand points keep at least this far outside the partner
    w_wrist: float = 3.0           # robust (Geman-McClure-like, scale sigma_wrist) distance to the triangulated COCO wrist
    sigma_wrist_m: float = 0.035
    w_elbow: float = 1.5
    sigma_elbow_m: float = 0.045
    w_contact: float = 1.2e4       # per m^2: the palm patch within palm_reach of the surface (palm holds)
    palm_reach_m: float = 0.012
    w_hand_gap: float = 1.2e4      # per m^2, soft-min gap -> hand_gap (hand-hand)
    w_normal: float = 10.0         # (1 + n_palm . n_surface): the palm faces the surface
    w_pen: float = 2e5             # per m^2 (mean over the points of a group)
    w_tip: float = 1.5e3           # per m^2: fingertips within 5 cm drawn onto the surface
    w_2d: float = 1.0              # robust 2D reprojection of MAMMA's hand / arm landmarks (the fold's phones)
    sigma_px: float = 8.0
    w_rot: tuple = (10.0, 10.0, 5.0)  # per rad^2: shoulder, elbow, wrist delta (minimal arm change)
    w_elbow_pos: float = 1.0e3     # per m^2: the elbow keeps its place where no elbow detection says otherwise
    w_smooth: float = 4000.0       # per rad^2: second difference of the deltas over time
    w_smooth_pos: float = 4.0e4    # per m^2: second difference of the wrist and palm centre (5 mm per frame^2 costs 1)
    w_curl: float = 0.6            # (c - c0)^2
    w_curl_smooth: float = 400.0
    wrist_cap_rad: float = 1.35    # soft cap of the wrist's local rotation (about 77 deg)
    w_wrist_cap: float = 30.0
    coco_min_run: int = 8          # a COCO target counts only inside a run of at least this many frames
    coco_far_m: float = 0.06       # a target further than this from the fitted wrist needs >= 3 agreeing phones
    coco_near_partner_m: float = 0.10  # ... and every target must lie within this of the partner's surface (the event is a contact)
    max_wrist_move_m: float = 0.25  # guard: no frame's wrist moves further (a palm turning over moves the palm centre a lot, the wrist little)
    float_slack_cm: float = 2.0    # guard: the hand may not end further from the partner than before + this (median over the core)
    w_stage2_move: float = 2.0e3   # per m^2 beyond stage2_free_m: stage 2 stays local to the stage-1 palm (minimal change)
    stage2_free_m: float = 0.04
    hand_only: bool = False         # pass 2: shoulder + elbow frozen, the wrist turns a little, the fingers take the hold, no stage 1
    jitter_rel: float = 1.5        # guard: the wrist's jitter (RMS second difference, window) may not exceed max(rel x before, before + abs)
    jitter_abs_mm: float = 3.0
    tips_jitter_rel: float = 1.5   # ... the same for the mean fingertip
    tips_jitter_abs_mm: float = 3.0
    guard_min_frames: int = 10     # a 2D guard is applied only when it has at least this many frames
    agree_px: float = 12.0         # MAMMA's hand landmarks support the fitted hand when they reproject within this (median, >= 2 phones)
    flip_deg: float = 100.0        # palm facing away by more than this after stage 1: also try a forearm-twisted start
    max_turn_deg: float = 95.0     # guard: shoulder / elbow (a 20 cm correction of an arm around the partner needs large turns)
    pen_slack_cm: float = 0.5      # guard: the deepest hand point may not sink further than before + this
    ma2d_slack_px: float = 0.5    # guard: MAMMA's hand landmarks, on the frames where they supported the fitted hand, may not get worse by more
    target_slack_cm: float = 0.5   # guard: the wrist may not end further from the triangulated COCO wrist (fit phones) than this beyond before
    # --- 2026-10-21 safety pass (independent review of the first v3 result: whips of 1.5-2.9 m/s, hands 3-20 px further from MAMMA's own
    # hand points, new penetration in hand-hand holds, splayed open hands, palms hovering 2 cm while the fingertips touched)
    ma2d_all_slack_px: float = 3.0  # guard: MAMMA's hand landmarks over ALL core frames (median, 6 phones) may not get worse by more
    coco_slack_px: float = 4.0     # guard: the independent COCO wrist / elbow (median px, test phones) may not get worse by more
    v_free_mps: float = 0.30       # loss: the motion the refinement ADDS to the wrist / palm centre is free up to this speed ...
    w_speed: float = 600.0         # ... and costs this per (m/s)^2 x s above it (a 0.5 m/s whip over 5 frames costs ~20)
    add_speed_max_mps: float = 0.55  # guard: no frame where the added motion (refined - input, wrist or palm centre) moves faster (a new jump)
    speed_margin_mps: float = 0.25  # guard: no wrist / palm speed above max(input p99 of that hand, the input's own speed there) + this
    v_ramp_mps: float = 0.35       # the blend ramps are lengthened so that the blend itself adds at most ~this speed (raised cosine)
    ramp_max: int = 45             # ... up to this many frames (1.5 s)
    pen_ok_cm: float = 0.2         # guard (exact mesh distances, winding numbers): the deepest hand point's p90 over the core may not exceed
    pen_slack_new_cm: float = 0.05  #   max(before, pen_ok) + this; its max may not exceed max(before, 2 x pen_ok) + 2 x this (no new penetration)
    hh_margin_m: float = 0.003     # hand-hand: the hand points keep at least this far outside the partner's hand (exact sign in the planes)
    w_flat: float = 3.0e4          # flat-hand holds: per m^2, the palm patch's soft-min gap -> palm_offset (both ways: on the surface) ...
    w_flat_patch: float = 2.0e4    # ... and every palm point within flat_patch_m of the surface (the whole palm on it, not a fingertip touch)
    flat_patch_m: float = 0.008
    w_normal_flat: float = 30.0    # flat holds: the palm faces the surface more firmly
    w_anchor: float = 2.0e3        # per m^2: an anchored hold (curated override): the palm centre onto the anchor point ...
    w_anchor_elbow: float = 4.0e2  # ... the elbow to the multi-view COCO elbow where it is triangulated


KNOT_STEP = 5


def temporal_basis(W: int, step: int = KNOT_STEP) -> np.ndarray:
    """(W, n) smooth partition of unity: cubic B-spline-like hat functions (Gaussian, sigma = 0.6 step) on knots every `step` frames"""
    n = max(2, int(np.ceil((W - 1) / step)) + 1)
    knots = np.linspace(0, W - 1, n)
    sig = max(1.0, 0.6 * (knots[1] - knots[0]))
    B = np.exp(-0.5 * ((np.arange(W)[:, None] - knots[None]) / sig) ** 2)
    return B / B.sum(1, keepdims=True)


def ramp_weights(n_core: int, ramp: int) -> np.ndarray:
    r = 0.5 - 0.5 * np.cos(np.pi * (np.arange(1, ramp + 1) / (ramp + 1)))
    return np.concatenate([r, np.ones(n_core), r[::-1]])


class Actor:
    """One dancer inside an event window: torch FK of its hand / arm subset, the variables that move it."""

    def __init__(self, fit_r: dict, role: str, side: str, ks: np.ndarray, pose: np.ndarray, device: str, template: str | None):
        import torch
        from .hand_kin import SHOULDER, ELBOW, WRIST, FINGER_JOINTS, model_npz
        self.torch, self.dev = torch, device
        self.role, self.side, self.ks = role, side, ks
        self.ids = rest_hand_ids(fit_r["betas"])[side]
        parts = landmark_parts()
        lv = landmark_matrix().argmax(1)
        vj = vertex_joint()
        arm_lm = [i for i in parts["body"] if vj[lv[i]] in (SHOULDER[side], ELBOW[side])]
        self.lm_idx = np.concatenate([parts[f"{side}_hand"], np.array(arm_lm, int)])
        self.n_hand_lm = len(parts[f"{side}_hand"])
        upper = np.flatnonzero(vj == SHOULDER[side])[::3]
        sub = np.concatenate([self.ids.hand, self.ids.palm, self.ids.arm, upper, lv[self.lm_idx]])
        self.nh, self.npm, self.na, self.nu = len(self.ids.hand), len(self.ids.palm), len(self.ids.arm), len(upper)
        self.kin = kin_for(fit_r["betas"], device, sub, dtype="float32")
        f64 = dict(dtype=torch.float32, device=device)                 # float32 throughout the optimiser
        self.base = torch.tensor(pose[ks], **f64)                       # (W,55,3)
        self._base_np = np.asarray(pose[ks], np.float64).copy()
        self.transl = torch.tensor(np.asarray(fit_r["transl"], float)[ks], **f64)
        self.chain = [SHOULDER[side], ELBOW[side], WRIST[side]]
        self.fj = list(FINGER_JOINTS[side])
        m = np.load(model_npz(model_dir()), allow_pickle=True)
        mean = np.asarray(m["hands_meanl" if side == "left" else "hands_meanr"], float).reshape(15, 3)
        self.mean = torch.tensor(mean, **f64)
        flex, fixed = finger_template(fit_r["betas"], side, template or "relaxed")
        self.flex, self.fixed = torch.tensor(flex, **f64), torch.tensor(fixed, **f64)
        W = len(ks)
        self.B = torch.tensor(temporal_basis(W, KNOT_STEP), **f64)      # (W, n_knots)
        nk = self.B.shape[1]
        self.delta_k = torch.zeros((nk, 3, 3), **f64, requires_grad=True)
        self.template = template
        self.palm1 = None                                               # the stage-1 palm (stage 2 stays near it)
        self.core = None                                                # (first, last) window index of the event's core (refine_event)
        self.fingers_on = False                                         # stage 1 keeps the input fingers
        if template is not None:
            tp = TEMPLATES[template]
            self.lo, self.hi, self.c0 = tp["lo"], tp["hi"], tp["c0"]
            x = (tp["c0"] - self.lo) / (self.hi - self.lo)
            self.u_k = torch.full((nk, 5), float(np.log(x / (1 - x))), **f64, requires_grad=True)
        else:
            self.u_k = None
        with torch.no_grad():
            o = self.kin.forward(self.base, self.transl, want_verts=True)
            self.elbow0 = o["joints"][:, ELBOW[side]].clone()
            self.wrist0 = o["joints"][:, WRIST[side]].clone()
            self.palm_c0 = _palm_centre_t(o["joints"], side).clone()     # the base's palm centre (the speed term's reference)
            # the fixed rest of the body for the fast forward: only the arm chain (shoulder, elbow, wrist) and the fingers change
            self._Rb = o["R"].clone()
            self._Pb = (o["joints"] - self.transl[:, None]).clone()
            self._pfb = o["pose_feature"].clone()
        self._chg = self.chain + self.fj                               # topological order (parents first)
        self._eye = torch.eye(3, dtype=torch.float32, device=device)
        from .hand_kin import rodrigues
        self._Rb_chain = rodrigues(torch, self.base[:, self.chain]).detach()

    def fast_forward(self, P, Rchain=None):
        """kin.forward for a pose that differs from the base only on the arm chain + fingers (18 of 55 joints recomposed): same outputs.
        Rchain (W,3,3,3): the chain's local rotations (else from P's axis-angle)"""
        from .hand_kin import rodrigues
        torch = self.torch
        kin = self.kin
        ch = self._chg
        Rl = rodrigues(torch, P[:, ch])                                 # (W,18,3,3)
        if Rchain is not None:
            Rl = torch.cat([Rchain, Rl[:, 3:]], 1)
        newR, newP = {}, {}
        for n, j in enumerate(ch):
            par = int(kin.parents[j])
            Rp = newR[par] if par in newR else self._Rb[:, par]
            Pp = newP[par] if par in newP else self._Pb[:, par]
            newR[j] = Rp @ Rl[:, n]
            newP[j] = Pp + (Rp @ kin.rel[j][:, None])[..., 0]
        R = self._Rb.clone()
        Jp = self._Pb.clone()
        idx = torch.as_tensor(ch, device=self.dev)
        R[:, idx] = torch.stack([newR[j] for j in ch], 1)
        Jp[:, idx] = torch.stack([newP[j] for j in ch], 1)
        tl = Jp - (R @ kin.J[None, :, :, None])[..., 0]
        pf = self._pfb.clone().reshape(len(P), 54, 9)
        pf[:, idx - 1] = (Rl - self._eye).reshape(len(P), len(ch), 9)
        out = {"R": R, "joints": Jp + self.transl[:, None], "tl": tl, "pose_feature": pf.reshape(len(P), -1)}
        out["tips"] = kin._lbs(out, kin._tip, self.transl)
        out["verts"] = kin._lbs(out, kin.subset, self.transl)
        return out

    def params(self, stage: int):
        return [self.delta_k] + ([self.u_k] if (self.u_k is not None and stage == 2) else [])

    def _edge(self, x):
        """outside the core (the blend ramps) a variable keeps its value at the core's edge: the ramp is a pure blend of the core's
        correction (2026-10-21: a free ramp let the hand overshoot 10 cm and come back inside 0.4 s, a 0.67 m/s jump)"""
        if self.core is None:
            return x
        c0, c1 = self.core
        idx = self.torch.clamp(self.torch.arange(x.shape[0], device=x.device), c0, c1)
        return x[idx]

    @property
    def delta(self):
        return self._edge(self.torch.einsum("wn,nij->wij", self.B, self.delta_k))

    @property
    def u(self):
        return None if self.u_k is None else self._edge(self.B @ self.u_k)

    def freeze_arm(self):
        """hand-only refinement: zero the shoulder / elbow deltas' gradient (the delta stays 0 there)"""
        self.delta_k.register_hook(lambda g: g * self._arm_mask)
        self._arm_mask = self.torch.zeros_like(self.delta_k)
        self._arm_mask[:, 2] = 1.0

    def curl(self):
        return self.lo + (self.hi - self.lo) * self.torch.sigmoid(self.u)

    def chain_R(self, w):
        """(W,3,3,3) local rotations of shoulder, elbow, wrist: R_base @ exp(w * delta) (a local-frame delta: smooth whatever the base's
        axis-angle representation does)"""
        from .hand_kin import rodrigues
        return self._Rb_chain @ rodrigues(self.torch, w[:, None, None] * self.delta)

    def pose(self, w):
        """current (W,55,3) pose as axis-angle for everything but the arm chain (whose rotations are chain_R; the chain entries here are the
        base's), fingers blended to the template by w (stage 2)"""
        P = self.base.clone()
        wk = w[:, None, None]
        if self.u is not None and self.fingers_on:
            cj = self.curl().repeat_interleave(3, dim=1)                # (W,15): the curl of each joint's finger
            tmpl = cj[..., None] * self.flex[None] + self.fixed[None]
            P[:, self.fj] = (1 - wk) * self.base[:, self.fj] + wk * tmpl
        return P

    def forward(self, w):
        o = self.fast_forward(self.pose(w), self.chain_R(w))
        V = o["verts"]
        a = 0
        for key, n in (("hand", self.nh), ("palm", self.npm), ("arm", self.na), ("upper", self.nu)):
            o[key] = V[:, a:a + n]
            a += n
        o["lm"] = V[:, a:]
        o["tips5"] = o["tips"][:, :5] if self.side == "left" else o["tips"][:, 5:]
        return o

    def pose_np(self, w_np):
        """the pose as axis-angle (T,55,3): the chain from chain_R (log map), the rest as pose()"""
        from scipy.spatial.transform import Rotation as Ro
        torch = self.torch
        with torch.no_grad():
            wt = torch.tensor(w_np, dtype=torch.float32, device=self.dev)
            P = self.pose(wt).cpu().numpy().astype(np.float64)
            Rc = self.chain_R(wt).cpu().numpy().astype(np.float64)
        Wn = len(P)
        P[:, self.chain] = Ro.from_matrix(Rc.reshape(-1, 3, 3)).as_rotvec().reshape(Wn, 3, 3)
        # keep the input's own axis-angle where the joint was not moved (w = 0: bit-identical input outside the blend)
        same = np.asarray(w_np) <= 0
        P[same] = self._base_np[same]
        return P

    def wrist_angle(self, w):
        """(W,) local rotation angle of the wrist (rad)"""
        R = self.chain_R(w)[:, 2]
        tr = (R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]).clamp(-1.0, 3.0)
        return torch_acos_safe(self.torch, (tr - 1) / 2)

    def snapshot(self):
        return self.delta_k.detach().clone(), (self.u_k.detach().clone() if self.u_k is not None else None)

    def restore(self, snap):
        with self.torch.no_grad():
            self.delta_k.copy_(snap[0])
            if snap[1] is not None:
                self.u_k.copy_(snap[1])


def torch_acos_safe(torch, x):
    return torch.acos(x.clamp(-1 + 1e-6, 1 - 1e-6))


def _palm_centre_t(J, side):
    """torch (W,55,3) -> (W,3): palm_centre"""
    from .hand_kin import WRIST
    i, m, r, p = mcp_joints(side)
    return 0.4 * J[:, WRIST[side]] + 0.15 * (J[:, i] + J[:, m] + J[:, r] + J[:, p])


def _palm_normal_t(torch, J, side):
    from .hand_kin import WRIST
    i, _, _, p = mcp_joints(side)
    a = J[:, i] - J[:, WRIST[side]]
    b = J[:, p] - J[:, WRIST[side]]
    n = torch.cross(a, b, dim=-1) * (-1.0 if side == "left" else 1.0)
    return n / n.norm(dim=-1, keepdim=True).clamp_min(1e-9)


def query_batched(surf: GpuSurface, P, V, bs: int = 12, wn_within: float = 0.01, inside_test: bool = True) -> dict:
    """GpuSurface.query over frame batches (the cdist of a long window does not fit the GPU at once)"""
    torch = surf.torch
    out = {}
    for a in range(0, P.shape[0], bs):
        q = surf.query(P[a:a + bs].float(), V[a:a + bs].float(), wn_within=wn_within, inside_test=inside_test)
        for k, v in q.items():
            out.setdefault(k, []).append(v)
    return {k: torch.cat(v, 0) for k, v in out.items()}


class Planes:
    """Closest-point planes of query points on a (fixed, per refresh) partner surface: sd ~ (p - q) . n near the surface."""

    def __init__(self, torch, surf: GpuSurface, reach: float = 0.12):
        self.torch, self.surf, self.reach = torch, surf, reach
        self.q = None

    def refresh(self, P, V, inside_test: bool = False):
        torch = self.torch
        with torch.no_grad():
            qr = query_batched(self.surf, P, V, inside_test=inside_test)
            self.q, self.n = qr["q"].float(), qr["nq"].float()
            inside = qr["sd"] < 0
            side = ((P - self.q) * self.n).sum(-1)
            flip = (inside & (side > 0)) | (~inside & (side < 0))          # the winding test decides the side, the plane follows
            self.n = torch.where(flip[..., None], -self.n, self.n)
            self.near = (qr["dist"] < self.reach).float()

    def sd(self, P):
        return ((P - self.q) * self.n).sum(-1) * self.near + (1 - self.near) * 1.0


def _robust2d(torch, e2, sigma):
    return sigma ** 2 * torch.log1p(e2 / sigma ** 2)


class TwoD:
    """The actor's MAMMA hand / arm landmarks (ma_2d, visible) in the given phones over the window"""

    def __init__(self, torch, obs: Obs, actor: Actor, device, cams):
        self.torch = torch
        ks = actor.ks
        self.items = []
        p = PIDX[actor.role]
        for c in cams:
            z = obs.z[c]
            lm = z["lm"][ks, p][:, actor.lm_idx]
            vis = (z["vis"][ks, p][:, actor.lm_idx] > 0.5) & z["ok"][ks][:, None] & (np.abs(lm).sum(-1) > 0)
            if vis.sum() < 10:
                continue
            f64 = dict(dtype=torch.float32, device=device)
            self.items.append({"cam": c, "lm": torch.tensor(lm, **f64), "vis": torch.tensor(vis, device=device),
                               "K": torch.tensor(z["K"][ks], **f64), "R": torch.tensor(z["R"][ks], **f64), "t": torch.tensor(z["t"][ks], **f64)})

    def errors(self, X):
        torch = self.torch
        out = []
        for it in self.items:
            Xc = torch.einsum("kij,knj->kni", it["R"], X) + it["t"][:, None]
            uv = torch.einsum("kij,knj->kni", it["K"], Xc)
            uv = uv[..., :2] / uv[..., 2:].clamp_min(1e-6)
            out.append((it["cam"], (uv - it["lm"]).norm(dim=-1), it["vis"] & (Xc[..., 2] > 0.1)))
        return out

    def loss(self, X, sigma, w):
        torch = self.torch
        tot = 0.0
        for _, e, v in self.errors(X):
            r = _robust2d(torch, e * e, sigma)
            n = v.sum(1).clamp_min(1)
            tot = tot + ((r * v).sum(1) / n * w).sum() / sigma ** 2
        return tot

    def per_frame(self, X, hand_only: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(W,) median over phones of the per-phone median px error (NaN = unseen), (W,) number of phones that see the hand"""
        torch = self.torch
        meds = []
        with torch.no_grad():
            for _, e, v in self.errors(X):
                if hand_only is not None:
                    e, v = e[:, :hand_only], v[:, :hand_only]
                ev = torch.where(v, e, torch.full_like(e, float("nan")))
                m = torch.nanmedian(ev, dim=1).values.cpu().numpy()
                meds.append(np.where(v.sum(1).cpu().numpy() >= 5, m, np.nan))
        if not meds:
            return np.full(X.shape[0], np.nan), np.zeros(X.shape[0], int)
        M = np.stack(meds, 1)
        n = np.isfinite(M).sum(1)
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                return np.nanmedian(M, 1), n

    def median_px(self, X) -> float:
        torch = self.torch
        vals = []
        with torch.no_grad():
            for _, e, v in self.errors(X):
                vals.append(e[v].cpu().numpy())
        return float(np.median(np.concatenate(vals))) if vals and sum(len(x) for x in vals) else float("nan")

    def core_median(self, X, core: np.ndarray, n_first: int | None = None) -> float:
        """median px over every visible landmark (the first n_first: the hand) of the frames `core` in all phones (validate's hand_lm_px)"""
        torch = self.torch
        vals = []
        with torch.no_grad():
            for _, e, v in self.errors(X):
                if n_first is not None:
                    e, v = e[:, :n_first], v[:, :n_first]
                m = v & torch.as_tensor(np.asarray(core, bool)[:, None], device=v.device)
                vals.append(e[m].cpu().numpy())
        n = sum(len(x) for x in vals)
        return float(np.median(np.concatenate(vals))) if n >= 20 else float("nan")


def _forearm_frame(J: np.ndarray, side: str) -> np.ndarray:
    """(W,3,3) columns = forearm axis (elbow -> wrist), palm normal (orthogonalised), their cross"""
    from .hand_kin import WRIST, ELBOW
    ax = J[:, WRIST[side]] - J[:, ELBOW[side]]
    ax /= np.maximum(np.linalg.norm(ax, axis=1, keepdims=True), 1e-9)
    n = palm_normal(J, side)
    n = n - (n * ax).sum(1, keepdims=True) * ax
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    return np.stack([ax, n, np.cross(ax, n)], 2)


def calibrate_coco_bias(fit: dict, obs: Obs, judge, n: int = 700, seed: int = 0) -> dict:
    """median offset (forearm frame) of the triangulated COCO wrist / elbow from the SMPL-X wrist / elbow joint on free hands
    (> 15 cm from the partner, multi-view frames, >= 3 agreeing views) -> {"<role>_<side>_<joint>": [axis, palm, side] m}"""
    from .hand_kin import WRIST, ELBOW
    p = OUT / "coco_bias.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))["bias"]
    S = np.load(OUT / "series.npz")
    rng = np.random.default_rng(seed)
    out, stats = {}, {}
    for r in ROLES:
        for s in SIDES:
            free = np.flatnonzero((S[f"{r}_{s}_gap"] > 0.15) & (np.asarray(fit[r]["quality_flag"]) == 3))
            ks = np.sort(rng.choice(free, min(n, len(free)), replace=False))
            Fr = _forearm_frame(fit[r]["joints"][ks], s)
            for which, j in (("wrist", WRIST[s]), ("elbow", ELBOW[s])):
                tr = triangulate_joint(fit[r], r, s, ks, judge, obs, which=which)
                ok = np.isfinite(tr["X"]).all(1) & (tr["n_views"] >= 3)
                d = tr["X"][ok] - fit[r]["joints"][ks[ok], j]
                loc = np.einsum("kji,kj->ki", Fr[ok], d)
                out[f"{r}_{s}_{which}"] = np.median(loc, 0).tolist()
                stats[f"{r}_{s}_{which}"] = {"n": int(ok.sum()), "dist_cm_median": float(np.median(np.linalg.norm(d, axis=1)) * 100),
                                             "local_iqr_cm": ((np.percentile(loc, 75, 0) - np.percentile(loc, 25, 0)) * 100).tolist()}
    write_json(p, {"what": "COCO (owner's 2D JSON, triangulated) minus SMPL-X joint on free hands, forearm frame [axis elbow->wrist, palm normal, "
                           "cross] in m; subtracted from the triangulated targets", "bias": out, "stats": stats})
    return out


def coco_targets(fit_r: dict, role: str, side: str, ks: np.ndarray, judge, obs: Obs, cams, bias: dict) -> dict:
    """bias-corrected, cleaned multi-view COCO wrist / elbow tracks over ks -> {which: (X (W,3), valid (W,), n_views (W,))}"""
    Fr = _forearm_frame(fit_r["joints"][ks], side)
    out = {}
    for which in ("wrist", "elbow"):
        tr = triangulate_joint(fit_r, role, side, ks, judge, obs, cams, which=which)
        b = np.asarray(bias.get(f"{role}_{side}_{which}", [0, 0, 0]), float)
        X = tr["X"] - np.einsum("kij,j->ki", Fr, b)
        Y, valid = smooth_track(X)
        out[which] = (Y, valid & np.isfinite(Y).all(1), tr["n_views"])
    return out


def anchored_targets(fitB: dict, poseB: np.ndarray, ksB: np.ndarray, palm0: np.ndarray, VB: np.ndarray, JB: np.ndarray, surf: GpuSurface,
                     device: str, sigma: float = 4.0) -> tuple[np.ndarray, np.ndarray, int]:
    """Per window frame: the partner surface point nearest to the actor's palm, smoothed over time in the partner's local frame of the
    dominant joint (the hold point moves with the partner and does not jitter), projected back onto the surface -> (q (W,3), n (W,3), joint)"""
    import torch
    from scipy.ndimage import gaussian_filter1d
    vj = vertex_joint()
    with torch.no_grad():
        qr = query_batched(surf, torch.as_tensor(palm0[:, None].astype(np.float32), device=device), torch.as_tensor(VB, device=device), bs=256)
    q = qr["q"][:, 0].cpu().numpy().astype(float)
    pv = qr["vert"][:, 0].cpu().numpy()
    j = int(np.bincount(vj[pv], minlength=55).argmax())
    Rb = _joint_rot(fitB, poseB, ksB)[:, j]
    loc = np.einsum("kji,kj->ki", Rb, q - JB[:, j])
    if len(loc) > 3:
        loc = gaussian_filter1d(loc, sigma, axis=0, mode="nearest")
    qs = np.einsum("kij,kj->ki", Rb, loc) + JB[:, j]
    with torch.no_grad():
        qr = query_batched(surf, torch.as_tensor(qs[:, None].astype(np.float32), device=device), torch.as_tensor(VB, device=device), bs=256)
    return qr["q"][:, 0].cpu().numpy().astype(float), qr["nq"][:, 0].cpu().numpy().astype(float), j


def _joint_rot(fit_r: dict, pose: np.ndarray, ks: np.ndarray) -> np.ndarray:
    """(W,55,3,3) global joint rotations of a pose array (T,55,3) of the dancer at ks"""
    import torch
    b = Body(fit_r, "cpu")
    with torch.no_grad():
        o = b.kin.forward(torch.tensor(pose[ks], dtype=torch.float64), torch.tensor(np.asarray(fit_r["transl"], float)[ks], dtype=torch.float64))
    return o["R"].numpy()


def twist_start(actor: Actor, w, ntg_np: np.ndarray) -> None:
    """set the actor's wrist delta so that, at the current arm, the palm normal is turned about the forearm axis onto -n_surface (per frame,
    smoothed): the second start for a palm that faces away from what it touches"""
    import torch
    from scipy.ndimage import gaussian_filter1d
    from scipy.spatial.transform import Rotation as Ro
    from .hand_kin import WRIST, ELBOW
    with torch.no_grad():
        w1 = torch.ones_like(w)
        o = actor.forward(w1)
        J = o["joints"].cpu().numpy()
        Rg = o["R"].cpu().numpy()
        Rc = actor.chain_R(w1).cpu().numpy().astype(np.float64)
        Rb = actor._Rb_chain.cpu().numpy().astype(np.float64)
    s = actor.side
    ax = J[:, WRIST[s]] - J[:, ELBOW[s]]
    ax /= np.linalg.norm(ax, axis=1, keepdims=True)
    n = palm_normal(J, s)
    want = -ntg_np
    pn = n - (n * ax).sum(1, keepdims=True) * ax
    pw = want - (want * ax).sum(1, keepdims=True) * ax
    ang = np.arctan2((np.cross(pn, pw) * ax).sum(1), (pn * pw).sum(1))
    ang = np.unwrap(ang)
    if len(ang) > 3:
        ang = gaussian_filter1d(ang, 3.0, mode="nearest")
    dl = actor.delta.detach().cpu().numpy().copy()
    for i in range(len(J)):
        Rp = Rg[i, ELBOW[s]]
        Rt = Ro.from_rotvec(ang[i] * ax[i]).as_matrix()
        Rnew = Rp.T @ Rt @ Rp @ Rc[i, 2]                                # the wrist's new local rotation
        dl[i, 2] = Ro.from_matrix(Rb[i, 2].T @ Rnew).as_rotvec()        # local-frame delta from the base
    with torch.no_grad():
        B = actor.B.cpu().numpy().astype(np.float64)
        knots = np.linalg.lstsq(B, dl.reshape(len(dl), -1), rcond=None)[0].reshape(-1, 3, 3)   # the twist on the smooth basis
        actor.delta_k.copy_(torch.tensor(knots, dtype=torch.float32, device=actor.dev))


def ray_first_hit(O: np.ndarray, D: np.ndarray, V: np.ndarray, faces: np.ndarray, device: str = "cuda", chunk: int = 48) -> tuple[np.ndarray, np.ndarray]:
    """Moeller-Trumbore: ray i (origin O[i], unit direction D[i]) against mesh V[i] (n,nv,3) -> (t (n,) first hit, inf = none; face (n,))"""
    import torch
    F = torch.as_tensor(np.asarray(faces, np.int64), device=device)
    Ot = torch.as_tensor(O, dtype=torch.float32, device=device)
    Dt = torch.as_tensor(D, dtype=torch.float32, device=device)
    Vt = torch.as_tensor(V, dtype=torch.float32, device=device)
    tt = torch.full((len(O),), float("inf"), device=device)
    ff = torch.full((len(O),), -1, dtype=torch.long, device=device)
    with torch.no_grad():
        for a in range(0, len(O), chunk):
            sl = slice(a, a + chunk)
            A, B, C = Vt[sl][:, F[:, 0]], Vt[sl][:, F[:, 1]], Vt[sl][:, F[:, 2]]
            e1, e2 = B - A, C - A
            d = Dt[sl][:, None].expand_as(e2)
            p = torch.cross(d, e2, dim=-1)
            det = (e1 * p).sum(-1)
            ok = det.abs() > 1e-12
            inv = 1.0 / torch.where(ok, det, torch.ones_like(det))
            tv = Ot[sl][:, None] - A
            u = (tv * p).sum(-1) * inv
            q = torch.cross(tv, e1, dim=-1)
            v = (d * q).sum(-1) * inv
            t = (e2 * q).sum(-1) * inv
            hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0)
            t = torch.where(hit, t, torch.full_like(t, float("inf")))
            tm, fm = t.min(1)
            tt[sl] = tm
            ff[sl] = torch.where(torch.isfinite(tm), fm, torch.full_like(fm, -1))
    return tt.cpu().numpy(), ff.cpu().numpy()


def anchor_targets(fitB: dict, poseB: np.ndarray, ks: np.ndarray, VB: np.ndarray, JB: np.ndarray, anchor: dict, faces: np.ndarray,
                   device: str = "cuda") -> tuple[np.ndarray, np.ndarray]:
    """A curated hold point on the partner: local (x, y) in the frame of the partner's joint anchor["joint"] (SMPL-X body frame: x = her left,
    y = up, z = her front), on the surface side anchor["side"] ("back": the first surface hit of a ray from behind along +z; "front": from the
    front along -z) -> per window frame the surface point (W,3) and its outward normal (W,3). The hold point moves with the partner."""
    j = int(anchor.get("joint", 9))
    x, y = (float(v) for v in anchor["local_xy"])
    back = anchor.get("side", "back") == "back"
    Rb = _joint_rot(fitB, poseB, ks)[:, j]
    o_loc = np.array([x, y, -0.6 if back else 0.6])
    d_loc = np.array([0.0, 0.0, 1.0 if back else -1.0])
    O = JB[:, j] + np.einsum("kij,j->ki", Rb, o_loc)
    D = np.einsum("kij,j->ki", Rb, d_loc)
    t, f = ray_first_hit(O, D, VB, faces, device)
    if not np.isfinite(t).all():
        raise ValueError(f"anchor {anchor}: the ray misses the partner on {int((~np.isfinite(t)).sum())} frames")
    P = O + t[:, None] * D
    tri = VB[np.arange(len(ks))[:, None], faces[f]].astype(float)          # (W,3,3)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    n *= np.sign((n * -D).sum(1, keepdims=True))                         # outward = toward the ray's origin
    if len(n) > 3:
        from scipy.ndimage import gaussian_filter1d
        n = gaussian_filter1d(n, 3.0, axis=0, mode="nearest")                # face normals step from face to face
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    return P, n


def _keep_runs(m: np.ndarray, n: int) -> np.ndarray:
    """m with the True runs shorter than n frames removed"""
    out = np.zeros_like(m)
    i = 0
    while i < len(m):
        if m[i]:
            j = i
            while j < len(m) and m[j]:
                j += 1
            if j - i >= n:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


def _det(x):
    """a loss term for the record without a GPU sync (converted to float once, after the loop)"""
    return x.detach() if hasattr(x, "detach") else x


def _floats(terms: dict) -> dict:
    return {k: float(v) for k, v in terms.items()}


def _rho(torch, d2, sigma):
    return torch.log1p(d2 / sigma ** 2)  # Cauchy: ~ (d / sigma)^2 near, still pulls (slowly) when far


# ------------------------------------------------------------------------------------------------ speed: no whips, no new jumps
#
# 2026-10-21 review of the first v3 result: Kadu's right hand moved 43 cm in 0.27 s at 223.4 s (#118), 2.1 m/s at 139.3 s (#47), and the
# refinement's added motion exceeded 0.6 m/s in ~150 places. The contact terms pull frame by frame and the smoothness terms are means over the
# window, so a local whip was cheap. Now: a loss on the ADDED velocity (above v_free), blend ramps long enough for the move, and guards on the
# result against the input (no speed above the input's p99 + margin where the input was slower, no added frame-to-frame jump).

def hand_tracks(J: np.ndarray, side: str) -> dict:
    """(T,55,3) joints -> {"wrist": (T,3), "palm": (T,3) palm centre}"""
    from .hand_kin import WRIST
    return {"wrist": J[:, WRIST[side]], "palm": palm_centre(J, side)}


def speeds(X: np.ndarray) -> np.ndarray:
    """(T,3) -> (T-1,) m/s between consecutive grid frames"""
    return np.linalg.norm(np.diff(X, axis=0), axis=1) * HZ


def input_speed_ref(fit: dict) -> dict:
    """p99 of the input's wrist / palm-centre speed per hand over the whole class: {"<role>_<side>": {"wrist": m/s, "palm": m/s}}"""
    out = {}
    for r in ROLES:
        for s in SIDES:
            tr = hand_tracks(np.asarray(fit[r]["joints"], float), s)
            out[f"{r}_{s}"] = {k: float(np.percentile(speeds(v), 99)) for k, v in tr.items()}
    return out


def speed_violations(X_in: np.ndarray, X_new: np.ndarray, p99: float, add_max: float, margin: float) -> dict:
    """Against the input track X_in of the same frames: added (T-1,) speed of the refinement's own motion |d(X_new - X_in)/dt|, and the frames
    where (a) the added motion is faster than add_max (a new jump) or (b) the refined hand is faster than max(p99, input speed) + margin."""
    add = speeds(X_new - X_in)
    v0, v1 = speeds(X_in), speeds(X_new)
    jump = np.flatnonzero(add > add_max)
    fast = np.flatnonzero(v1 > np.maximum(p99, v0) + margin)
    return {"add": add, "v_in": v0, "v_new": v1, "jump": jump, "fast": fast,
            "max_add": float(add.max()) if len(add) else 0.0, "max_excess": float((v1 - np.maximum(p99, v0)).max()) if len(v1) else 0.0}


def ramp_for(move_m: float, v_ramp: float, lo: int, hi: int) -> int:
    """frames of a raised-cosine blend that moves move_m with a peak speed of v_ramp (peak = pi / 2 x move / duration)"""
    need = int(np.ceil(np.pi * move_m * HZ / (2.0 * max(v_ramp, 1e-3))))
    return int(np.clip(need, lo, hi))


def speed_audit(fit_in: dict, poses_or_fit, add_max: float = 0.55, margin: float = 0.25) -> dict:
    """The whole class, every hand: incidents (runs of frames within 3 of each other) where the refined fit adds a jump or a whip against the
    input fit (speed_violations with the hand's own input p99). poses_or_fit: {role: (T,55,3) pose} or a fit dict (joints used)."""
    ref = input_speed_ref(fit_in)
    out = {}
    for r in ROLES:
        if isinstance(poses_or_fit[r], dict):
            J1 = np.asarray(poses_or_fit[r]["joints"], float)
        else:
            b = Body(fit_in[r], "cpu")
            J1 = b.forward_np(np.asarray(poses_or_fit[r], float), np.asarray(fit_in[r]["transl"], float), verts=False)["joints"]
        J0 = np.asarray(fit_in[r]["joints"], float)
        for s in SIDES:
            t0, t1 = hand_tracks(J0, s), hand_tracks(J1, s)
            inc = []
            for k in ("wrist", "palm"):
                sv = speed_violations(t0[k], t1[k], ref[f"{r}_{s}"][k], add_max, margin)
                bad = np.union1d(sv["jump"], sv["fast"])
                groups = []
                for i in bad:
                    if groups and i - groups[-1][-1] <= 3:
                        groups[-1].append(int(i))
                    else:
                        groups.append([int(i)])
                for g in groups:
                    j = g[int(np.argmax(sv["add"][g]))]
                    inc.append({"track": k, "t": round(j / HZ, 2), "frame": j, "frames": [g[0], g[-1] + 1], "added_mps": round(float(sv["add"][j]), 2),
                                "speed_mps": round(float(sv["v_new"][j]), 2), "input_mps": round(float(sv["v_in"][j]), 2)})
            out[f"{r}_{s}"] = {"p99_input_mps": ref[f"{r}_{s}"], "incidents": sorted(inc, key=lambda x: x["frame"])}
    out["n_incidents"] = sum(len(v["incidents"]) for k, v in out.items() if isinstance(v, dict) and "incidents" in v)
    return out


def refine_event(fit: dict, poses: dict, ev: dict, twin: dict | None, obs: Obs, judge, bias: dict, surf: GpuSurface, cfg: RefineCfg,
                 cams: list[str], device: str = "cuda", speed_ref: dict | None = None, keep: bool = False) -> dict:
    """Refine one event (and its twin for a hand-hand hold) using the 2D of `cams` only. poses = {role: (T,55,3)} current poses; fit = the
    input fit (smplx_rf2_noslip: the speed guards compare with it). speed_ref = input_speed_ref(fit); keep = a curated event (curation.json
    "keep"): the 2D guards (MAMMA all frames, COCO) are waived after a visual check, every geometric / speed guard still holds."""
    import torch
    from .hand_kin import ELBOW, WRIST
    T = len(fit["lead"]["t"])
    r, s, o = ev["role"], ev["side"], ev["partner"]
    i0, i1 = ev["i0"], ev["i1"]
    if twin is not None:
        i0, i1 = min(i0, twin["i0"]), max(i1, twin["i1"])
    a0, a1 = max(0, i0 - cfg.ramp), min(T - 1, i1 + cfg.ramp)
    ks = np.arange(a0, a1 + 1)
    full = ramp_weights(i1 - i0 + 1, cfg.ramp)
    w_np = full[(a0 - (i0 - cfg.ramp)):(a0 - (i0 - cfg.ramp)) + len(ks)]
    w = torch.tensor(w_np, dtype=torch.float32, device=device)
    wsum = w.sum()
    hold = ev["hold"]
    hand_hand = twin is not None and hold in HAND_HOLDS
    anchor = ev.get("anchor")                    # curated hold point on the partner (curation.json overrides; e.g. #118 her left upper back)
    flat = (not hand_hand) and hold in FLAT_HOLDS
    margin = cfg.hh_margin_m if hand_hand else cfg.margin_m
    tmpl = TEMPLATE_OF_HOLD.get(hold, "relaxed")
    actors = [Actor(fit[r], r, s, ks, poses[r], device, tmpl)]
    if hand_hand:
        actors.append(Actor(fit[o], o, twin["side"], ks, poses[o], device, TEMPLATE_OF_HOLD.get(twin["hold"], tmpl)))
    twod = [TwoD(torch, obs, a, device, cams) for a in actors]
    if cfg.hand_only:
        for a in actors:
            a.freeze_arm()
    f64 = dict(dtype=torch.float32, device=device)
    _ce = np.flatnonzero(w_np > 0.99)
    for a in actors:
        a.core = (int(_ce[0]), int(_ce[-1])) if len(_ce) else None
    # the loss's speed cap: no faster than max(the input's p99 of that hand, the input's own speed there) + 0.1 m/s (the guard allows + margin)
    speed_cap = []
    for a in actors:
        J_in = np.asarray(fit[a.role]["joints"], float)[ks]
        ref = (speed_ref or {}).get(f"{a.role}_{a.side}", {"wrist": 2.0, "palm": 2.2})
        tr = hand_tracks(J_in, a.side)
        speed_cap.append([torch.tensor(np.maximum(ref[k], speeds(tr[k])) + 0.1, **f64) if len(ks) > 1 else None for k in ("wrist", "palm")])
    bodies = {}

    def mesh_of(role, pose_np):
        if role not in bodies:
            bodies[role] = Body(fit[role], device)
        fw = bodies[role].forward_np(pose_np, np.asarray(fit[role]["transl"], float)[ks], verts=True, batch=256)
        return fw["verts"], fw["joints"]

    VO, JO = mesh_of(o, poses[o][ks])
    tg = []
    with torch.no_grad():
        outs_b = [a.forward(w) for a in actors]
    gate_info = []
    gate_masks = []
    anc, anc_elbow = None, None
    for a, t2, ob in zip(actors, twod, outs_b):
        ct = coco_targets(fit[a.role], a.role, a.side, ks, judge, obs, cams, bias)
        if anchor is not None and a is actors[0]:
            # curated hold point: the palm centre onto the anchor on the partner's surface (palm facing it); the multi-view COCO elbow
            # (triangulated, bias-corrected, smoothed) keeps the elbow where the phones see it; the COCO wrist is not a target (in the
            # case this was built for, #118, the hidden wrist is detected on the wrong side in most phones: see curation.json)
            P_t, N_t = anchor_targets(fit[o], poses[o], ks, VO, JO, anchor, Body(fit[o], "cpu").faces, device)
            anc = {"P": torch.tensor(P_t + N_t * cfg.palm_offset_m, **f64), "n": torch.tensor(N_t, **f64), "P_np": P_t, "n_np": N_t}
            Xe_raw, ve_raw, _ = ct["elbow"]
            if ve_raw.sum() >= cfg.coco_min_run:
                anc_elbow = (torch.tensor(np.nan_to_num(Xe_raw), **f64), torch.tensor(ve_raw.astype(float), **f64))
            Xw, _, nvw = ct["wrist"]
            ct["wrist"] = (Xw, np.zeros(len(ks), bool), nvw)
            Xe, _, nve = ct["elbow"]
            ct["elbow"] = (Xe, np.zeros(len(ks), bool), nve)
            med, nview = t2.per_frame(ob["lm"], a.n_hand_lm)
            gate_info.append({"fit_supported_by_ma2d": float(((nview >= 2) & (med <= cfg.agree_px)).mean()), "coco_wrist_used": 0.0,
                              "anchor": anchor, "coco_elbow_used": float(ve_raw.mean())})
            gate_masks.append(((nview >= 2) & (med <= cfg.agree_px), med))
            tg.append({k: (torch.tensor(np.nan_to_num(X), **f64), torch.tensor(v.astype(float), **f64), nv) for k, (X, v, nv) in ct.items()})
            continue
        # the multi-view COCO wrist / elbow is a target only where MAMMA's own hand landmarks do NOT support the fitted hand
        # (>= 2 phones see the hand and the fit reprojects within agree_px there = the fit is right, leave it)
        med, nview = t2.per_frame(ob["lm"], a.n_hand_lm)
        supported = (nview >= 2) & (med <= cfg.agree_px)
        Xw, vw, nvw = ct["wrist"]
        # a far correction needs >= 3 agreeing phones, and in a contact event the wrist target must lie near the partner's surface
        # (2026-10-21 review: a 2-view target 36 cm off pulled a hand that was right on her back into the air)
        w0 = ob["joints"][:, WRIST[a.side]].cpu().numpy()
        far = np.linalg.norm(np.nan_to_num(Xw) - w0, axis=1) > cfg.coco_far_m
        Vp = VO if not hand_hand else mesh_of(OTHER[a.role], poses[OTHER[a.role]][ks])[0]
        with torch.no_grad():
            dq = query_batched(surf, torch.as_tensor(np.nan_to_num(Xw)[:, None].astype(np.float32), device=device),
                               torch.as_tensor(Vp, device=device), bs=128, inside_test=False)["dist"][:, 0].cpu().numpy()
        near_partner = dq < cfg.coco_near_partner_m
        vw = vw & ~supported & near_partner & (~far | (nvw >= 3))
        vw = _keep_runs(vw, cfg.coco_min_run)
        ct["wrist"] = (Xw, vw, nvw)
        Xe, ve, nve = ct["elbow"]
        e0 = ob["joints"][:, ELBOW[a.side]].cpu().numpy()
        ct["elbow"] = (Xe, ve & vw & (np.linalg.norm(np.nan_to_num(Xe) - e0, axis=1) < 0.15), nve)   # the elbow only with its wrist
        gate_info.append({"fit_supported_by_ma2d": float(supported.mean()), "coco_wrist_used": float(ct["wrist"][1].mean())})
        gate_masks.append((supported, med))
        tg.append({k: (torch.tensor(np.nan_to_num(X), **f64), torch.tensor(v.astype(float), **f64), nv) for k, (X, v, nv) in ct.items()})
    planes = [Planes(torch, surf) for _ in actors]
    wr = torch.tensor(cfg.w_rot, **f64)

    def coco_px(a: Actor, which: str, test_cams, core_only: bool = False) -> float:
        """median px of the actor's SMPL-X wrist / elbow (+ the bias) against the COCO detections in test_cams (window or core frames)"""
        with torch.no_grad():
            J = a.forward(w)["joints"].cpu().numpy()
        jj = WRIST[a.side] if which == "wrist" else ELBOW[a.side]
        b = np.asarray(bias.get(f"{a.role}_{a.side}_{which}", [0, 0, 0]), float)
        X = J[:, jj] + np.einsum("kij,j->ki", _forearm_frame(J, a.side), b)
        es = []
        for c in test_cams:
            z = obs.z[c]
            uv, d = obs.project(c, X[:, None], ks)
            for i, k in enumerate(ks):
                if core_only and w_np[i] < 0.99:
                    continue
                u, ok = judge.joint(c, a.role, a.side, int(k), which)
                if ok and z["ok"][k] and d[i, 0] > 0.1:
                    es.append(float(np.linalg.norm(uv[i, 0] - u)))
        return float(np.median(es)) if es else float("nan")

    def target_dist(a: Actor, g: dict, which: str) -> float:
        """median 3D distance (cm) of the SMPL-X wrist / elbow to the triangulated COCO target over the valid core frames"""
        with torch.no_grad():
            J = a.forward(w)["joints"]
        jj = WRIST[a.side] if which == "wrist" else ELBOW[a.side]
        X, v, _ = g[which]
        m = (v > 0.5) & (w > 0.99)
        return float((J[:, jj] - X).norm(dim=-1)[m].median()) * 100 if m.any() else float("nan")

    def penetration(a: Actor, oo, pl: Planes, stage: int = 2):
        """signed distances of hand / forearm / upper arm / tips + the penetration penalty (mean of squared depths per group; stage 1 lets
        the hand pass: only the arm is kept out, the hand is placed in stage 2)"""
        P = torch.cat([oo["hand"], oo["arm"], oo["upper"], oo["tips5"], oo["palm"]], 1)
        sd = pl.sd(P)
        nh, na, nu = a.nh, a.na, a.nu
        hand = torch.cat([sd[:, :nh], sd[:, nh + na + nu:]], 1)
        arm = sd[:, nh:nh + na + nu]
        dh = torch.relu(margin - hand) ** 2
        da = torch.relu(-arm) ** 2
        top = dh.topk(min(12, dh.shape[1]), dim=1).values.mean(1)          # the deepest points count, not only the average
        # an anchored hold's stage 1 is pure IK (the arm may have to swing through the partner to reach the other side: no arm term there)
        arm_w = 0.0 if (anchor is not None and stage < 2) else 0.5
        pen = (dh.mean(1) + top) * (1.0 if stage == 2 else 0.0) + arm_w * (da.mean(1) + da.topk(min(8, da.shape[1]), dim=1).values.mean(1))
        return sd, pen

    push_exact = [False]                     # the push-out polish: exact signs for palm holds too

    def refresh(outs):
        if hand_hand:
            VA, _ = mesh_of(actors[0].role, actors[0].pose_np(w_np))
            VB, _ = mesh_of(actors[1].role, actors[1].pose_np(w_np))
            others = [VB, VA]
        else:
            others = [VO]
        for a, pl, oo, Vm in zip(actors, planes, outs, others):
            P = torch.cat([oo["hand"], oo["arm"], oo["upper"], oo["tips5"], oo["palm"]], 1).detach()
            # the push-out polish: the exact inside test (winding numbers) decides each point's side - a closest-point plane of a thin
            # finger is often on the wrong side, which left hands 3-5 mm inside each other (2026-10-21 review: #9, 60, 64, 104). Only there:
            # in every refresh it made the hand-hand solves 2.5x slower.
            pl.refresh(P, torch.as_tensor(Vm, device=device), inside_test=push_exact[0])

    contact = {}

    def total_loss(outs, stage: int, terms: dict):
        loss = 0.0
        for ai, (a, pl, oo, t2, g) in enumerate(zip(actors, planes, outs, twod, tg)):
            J = oo["joints"]
            sd, pen = penetration(a, oo, pl, stage)
            loss = loss + cfg.w_pen * (pen * w).sum() / wsum
            terms[f"pen{ai}"] = _det((pen * w).sum() / wsum)
            for which, jj, wt, sg in (("wrist", WRIST[a.side], cfg.w_wrist, cfg.sigma_wrist_m), ("elbow", ELBOW[a.side], cfg.w_elbow, cfg.sigma_elbow_m)):
                X, v, _ = g[which]
                d2 = ((J[:, jj] - X) ** 2).sum(-1)
                lw = wt * (_rho(torch, d2, sg) * v * w).sum() / wsum
                loss = loss + lw
                terms[f"{which}{ai}"] = _det(lw)
            anchored = anc is not None and ai == 0
            ev_free = (1 - g["elbow"][1]) * (0.0 if anchored else 1.0)
            loss = loss + cfg.w_elbow_pos * (((J[:, ELBOW[a.side]] - a.elbow0) ** 2).sum(-1) * ev_free * w).sum() / wsum
            if t2.items and not anchored:
                l2 = cfg.w_2d * t2.loss(oo["lm"], cfg.sigma_px, w) / wsum
                loss = loss + l2
                terms[f"2d{ai}"] = _det(l2)
            if anchored:
                pc = oo["palm"].mean(1)
                la = cfg.w_anchor * ((((pc - anc["P"]) ** 2).sum(-1)) * w).sum() / wsum
                la = la + cfg.w_normal_flat * ((1 + (_palm_normal_t(torch, J, a.side) * anc["n"]).sum(-1)) * w).sum() / wsum
                if anc_elbow is not None:
                    la = la + cfg.w_anchor_elbow * ((((J[:, ELBOW[a.side]] - anc_elbow[0]) ** 2).sum(-1)) * anc_elbow[1] * w).sum() / wsum
                la = la * (1.0 if stage < 2 else 0.5)
                loss = loss + la
                terms["anchor"] = _det(la)
            if len(ks) > 2:
                # the motion the refinement adds to the wrist and the palm centre: free up to v_free, then w_speed per (m/s)^2 x s (no whips)
                lsp = 0.0
                for (X, X0), cap in zip(((J[:, WRIST[a.side]], a.wrist0), (_palm_centre_t(J, a.side), a.palm_c0)), speed_cap[ai]):
                    dX = X - X0
                    dv = (dX[1:] - dX[:-1]) * HZ
                    ex = torch.relu(torch.sqrt((dv ** 2).sum(-1) + 1e-10) - cfg.v_free_mps)
                    lsp = lsp + cfg.w_speed * (ex ** 2).sum() / HZ
                    v = (X[1:] - X[:-1]) * HZ                          # and the hand's own speed under the cap (no whip)
                    ex2 = torch.relu(torch.sqrt((v ** 2).sum(-1) + 1e-10) - cap)
                    lsp = lsp + cfg.w_speed * (ex2 ** 2).sum() / HZ
                loss = loss + lsp
                terms[f"speed{ai}"] = _det(lsp)
            lr_ = ((a.delta ** 2).sum(-1) * wr).sum(-1).mean()
            loss = loss + lr_
            terms[f"rot{ai}"] = _det(lr_)
            terms["_reg"] = terms.get("_reg", 0.0) + lr_.detach()
            if len(ks) > 2:
                acc = a.delta[2:] - 2 * a.delta[1:-1] + a.delta[:-2]
                lsm = cfg.w_smooth * (acc ** 2).sum((1, 2)).mean()
                # and the hand itself: second differences of the wrist and the palm centre (m) - the deltas' smoothness alone let the hand
                # wobble by a few mm per frame where the contact terms pulled frame by frame (2026-10-21 validation: jitter x4)
                for X in (J[:, WRIST[a.side]], oo["palm"].mean(1)):
                    ax_ = X[2:] - 2 * X[1:-1] + X[:-2]
                    lsm = lsm + cfg.w_smooth_pos * (ax_ ** 2).sum(-1).mean()
                loss = loss + lsm
                terms[f"smooth{ai}"] = _det(lsm)
            lwc = cfg.w_wrist_cap * (torch.relu(a.wrist_angle(w) - cfg.wrist_cap_rad) ** 2).mean()
            loss = loss + lwc
            terms[f"wcap{ai}"] = _det(lwc)
            le = cfg.w_elbow_pos * (((J[:, ELBOW[a.side]] - a.elbow0) ** 2).sum(-1) * (1 - g["elbow"][1]) * w).sum() / wsum
            terms[f"elbowpos{ai}"] = _det(le)
            if stage < 2:
                continue
            npalm = _palm_normal_t(torch, J, a.side)
            nb = a.nh + a.na + a.nu + 5
            if a.palm1 is not None:
                mv = (oo["palm"].mean(1) - a.palm1).norm(dim=-1)
                lm_ = cfg.w_stage2_move * ((torch.relu(mv - cfg.stage2_free_m) ** 2) * w).sum() / wsum
                loss = loss + lm_
                terms[f"move{ai}"] = _det(lm_)
            if not hand_hand:
                # minimal change: the hand surface just touches (soft-min gap -> hand_gap), the palm comes within palm_reach, the palm faces
                # the surface under it (the planes' normals at the palm patch)
                hd = sd[:, :a.nh]
                pd = sd[:, nb:]
                soft = -0.004 * torch.logsumexp(-hd / 0.004, dim=1)
                softp = -0.004 * torch.logsumexp(-pd / 0.004, dim=1)
                reach = (pl.near[:, :a.nh].sum(1) > 0).float()               # a frame with no surface within reach has no contact term
                lg = (soft - cfg.hand_gap_m) ** 2 * reach
                lp = torch.relu(softp - cfg.palm_reach_m) ** 2 * reach
                nsurf = (pl.n[:, nb:] * pl.near[:, nb:, None]).sum(1)
                nsurf = nsurf / nsurf.norm(dim=-1, keepdim=True).clamp_min(1e-9)
                has = (pl.near[:, nb:].sum(1) > 0).float()
                ln = (1 + (npalm * nsurf).sum(-1)) * has
                loss = loss + cfg.w_hand_gap * (lg * w).sum() / wsum + cfg.w_contact * (lp * w).sum() / wsum
                loss = loss + (cfg.w_normal_flat if flat else cfg.w_normal) * (ln * w).sum() / wsum
                if flat:
                    # a flat hand: the palm itself on the surface (the mean of its 8 closest points -> palm_offset, both ways: a soft-min
                    # target left the real minimum ~1 cm off) and the whole palm patch close
                    p8 = pd.topk(min(8, pd.shape[1]), dim=1, largest=False).values.mean(1)
                    lf = (p8 - cfg.palm_offset_m) ** 2 * reach
                    lfp = (torch.relu(pd - cfg.flat_patch_m) ** 2).mean(1) * reach
                    loss = loss + cfg.w_flat * (lf * w).sum() / wsum + cfg.w_flat_patch * (lfp * w).sum() / wsum
                    terms["flat"] = _det(cfg.w_flat * (lf * w).sum() / wsum + cfg.w_flat_patch * (lfp * w).sum() / wsum)
                terms["gap_cm"] = _det((soft * w).sum() / wsum * 100)
                terms["palm_gap_cm"] = _det((softp * w).sum() / wsum * 100)
                terms["normal"] = _det((ln * w).sum() / wsum)
            else:
                hd = sd[:, :a.nh]
                soft = -0.004 * torch.logsumexp(-hd / 0.004, dim=1)
                reach = (pl.near[:, :a.nh].sum(1) > 0).float()
                loss = loss + cfg.w_hand_gap * (((soft - cfg.hand_gap_m) ** 2) * reach * w).sum() / wsum
                terms[f"gap{ai}_cm"] = _det((soft * w).sum() / wsum * 100)
            if a.u is not None:
                st = sd[:, a.nh + a.na + a.nu:a.nh + a.na + a.nu + 5]
                pull = torch.relu(st - 0.004) * (st < 0.05).float()
                lt = cfg.w_tip * ((pull ** 2).sum(1) * w).sum() / wsum
                c = a.curl()
                lcu = cfg.w_curl * ((c - a.c0) ** 2).sum(-1).mean()
                if len(ks) > 2:
                    lcu = lcu + cfg.w_curl_smooth * ((c[2:] - 2 * c[1:-1] + c[:-2]) ** 2).sum(-1).mean()
                loss = loss + lt + lcu
                terms[f"tips{ai}"] = _det(lt)
                terms[f"curl{ai}"] = _det(lcu)
        if hand_hand and stage == 2:
            n0 = _palm_normal_t(torch, outs[0]["joints"], actors[0].side)
            n1 = _palm_normal_t(torch, outs[1]["joints"], actors[1].side)
            if hold in ("palm_to_palm", "handshake"):
                loss = loss + 0.5 * cfg.w_normal * ((1 + (n0 * n1).sum(-1)) * w).sum() / wsum
        terms["_total"] = _det(loss)
        return loss

    def optimise(stage: int, iters: int) -> tuple[float, dict]:
        params = [p for a in actors for p in a.params(stage)]
        opt = torch.optim.Adam(params, lr=cfg.lr if stage == 1 else cfg.lr2)
        terms = {}
        for it in range(iters):
            outs = [a.forward(w) for a in actors]
            if it % cfg.refresh == 0:
                refresh(outs)
            terms = {}
            loss = total_loss(outs, stage, terms)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if DEBUG and (it % 25 == 0 or it == iters - 1):
                log(f"  stage {stage} it {it}: loss {float(loss):.3f} " + " ".join(f"{k} {v:.3f}" for k, v in _floats(terms).items()))
        with torch.no_grad():
            outs = [a.forward(w) for a in actors]
            refresh(outs)
            terms = {}
            loss = float(total_loss(outs, stage, terms))
        return loss, _floats(terms)

    with torch.no_grad():
        outs0 = [a.forward(w) for a in actors]
    core_np = w_np > 0.99
    test_cams = [c for c in obs.cams if c not in cams] or list(obs.cams)
    res = {"event": ev["id"], "twin": twin["id"] if twin else None, "hold": hold, "template": tmpl, "frames": [int(a0), int(a1)],
           "core": [int(i0), int(i1)], "hand_hand": hand_hand, "flat": flat, "fit_cams": list(cams), "test_cams": test_cams, "gate": gate_info,
           "ramp": int(cfg.ramp), "curated_keep": bool(keep)}
    if anchor is not None:
        res["anchor"] = anchor
    if ev.get("merged"):
        res["merged"] = list(ev["merged"])

    def partner_meshes(after: bool) -> list:
        """per actor: the partner surface it is judged against (hand-hand: the other actor's dancer, refined or not)"""
        if not hand_hand:
            return [VO]
        if after:
            VA = mesh_of(actors[0].role, actors[0].pose_np(w_np))[0]
            VB = mesh_of(actors[1].role, actors[1].pose_np(w_np))[0]
        else:
            VA = mesh_of(actors[0].role, poses[actors[0].role][ks])[0]
            VB = mesh_of(actors[1].role, poses[actors[1].role][ks])[0]
        return [VB, VA]

    def exact_geo(outs_, Vps) -> list:
        """per actor, per window frame, exact (winding numbers, as validate): the deepest hand point (m, >= 0), the palm patch's signed min gap"""
        out = []
        with torch.no_grad():
            for a, oo, Vp in zip(actors, outs_, Vps):
                P = torch.cat([oo["hand"], oo["palm"]], 1).detach().float()
                q = query_batched(surf, P, torch.as_tensor(Vp, device=device), bs=8, wn_within=0.03, inside_test=True)
                sd = q["sd"]
                out.append((torch.relu(-sd[:, :a.nh].min(1).values).cpu().numpy().astype(float),
                            sd[:, a.nh:].min(1).values.cpu().numpy().astype(float)))
        return out

    def pen_stats(g) -> tuple[float, float]:
        p = g[0][core_np] if core_np.any() else g[0]
        return float(np.percentile(p, 90)) * 100, float(p.max()) * 100

    def is_new_pen(g0, g1) -> bool:
        b90, bmx = pen_stats(g0)
        a90, amx = pen_stats(g1)
        return a90 > max(b90, cfg.pen_ok_cm) + cfg.pen_slack_new_cm or amx > max(bmx, 2 * cfg.pen_ok_cm) + 2 * cfg.pen_slack_new_cm

    def hand_gap_cm(outs_) -> list:
        """median over the core frames of the closest hand point's distance to the partner (planes; > 12 cm counts as 12)"""
        refresh(outs_)
        core_t = torch.as_tensor(core_np, device=device)
        g = []
        for a, pl, oo in zip(actors, planes, outs_):
            sd, _ = penetration(a, oo, pl, 2)
            hmin = sd[:, :a.nh].min(1).values.clamp(max=0.12)
            g.append(float(hmin[core_t].median()) * 100 if core_t.any() else float("nan"))
        return g
    with torch.no_grad():
        gap_before = hand_gap_cm(outs0)
    geo0 = exact_geo(outs0, partner_meshes(False))
    # the penetration guard compares with the INPUT fit (hand and partner as fitted), not with the current base: a pass-2 / pass-4 solve or
    # a partner moved by another event must not accumulate new penetration (2026-10-21 post-check: #17, 62, 89, 104)
    with torch.no_grad():
        gin = hand_geometry(fit, device, ks=ks, roles=tuple(dict.fromkeys(a.role for a in actors)), sides=tuple(dict.fromkeys(a.side for a in actors)))
    geo_in = [(np.asarray(gin[(a.role, a.side)]["pen"], float), np.asarray(gin[(a.role, a.side)]["palm_gap"], float)) for a in actors]
    res["before"] = {"hand_gap_cm": gap_before, "ma2d_px": [t2.median_px(oo["lm"]) if t2.items else float("nan") for oo, t2 in zip(outs0, twod)],
                     "ma2d_px_core_all": [t2.core_median(oo["lm"], core_np, a.n_hand_lm) if t2.items else float("nan")
                                          for a, oo, t2 in zip(actors, outs0, twod)],
                     "coco_wrist_px_fit": [coco_px(a, "wrist", cams) for a in actors],
                     "coco_wrist_px_test": [coco_px(a, "wrist", test_cams) for a in actors],
                     "coco_elbow_px_test": [coco_px(a, "elbow", test_cams) for a in actors],
                     "coco_wrist_px_core": [coco_px(a, "wrist", test_cams, True) for a in actors],
                     "coco_elbow_px_core": [coco_px(a, "elbow", test_cams, True) for a in actors],
                     "coco_views_wrist": [float(np.mean(g["wrist"][1].cpu().numpy())) for g in tg],
                     "target_wrist_cm": [target_dist(a, g, "wrist") for a, g in zip(actors, tg)],
                     "target_elbow_cm": [target_dist(a, g, "elbow") for a, g in zip(actors, tg)],
                     "pen_exact_cm": [list(pen_stats(g)) for g in geo0],
                     "pen_exact_cm_input": [list(pen_stats(g)) for g in geo_in],
                     "palm_gap_exact_cm": [float(np.median(g[1][core_np])) * 100 if core_np.any() else float("nan") for g in geo0]}
    # stage 1: the arm to the multi-view wrist / elbow (skipped when no phone pair gives a usable target: nothing to move it to), or for an
    # anchored hold the arm to the anchor (pure IK, twice the iterations: the hand may have to go round the partner)
    if not cfg.hand_only and (anc is not None or any(float(g["wrist"][1].sum()) >= cfg.coco_min_run for g in tg)):
        loss1, terms1 = optimise(1, cfg.iters1 * (2 if anc is not None else 1))
        res["stage1"] = {"loss": loss1, **terms1}
    else:
        res["stage1"] = "skipped: no multi-view wrist target (the fitted hand is supported by MAMMA's landmarks or no phone pair sees it)"
    # stage 2: contact targets from the stage-1 palm, fingers on
    for a in actors:
        a.fingers_on = True
    with torch.no_grad():
        outs1 = [a.forward(w) for a in actors]
        for a, o1, o0 in zip(actors, outs1, outs0):
            a.palm1 = o1["palm"].mean(1).detach().clone()
        res["palm_move_stage1_max_m"] = [float((o1["palm"].mean(1) - o0["palm"].mean(1)).norm(dim=-1).max()) for o1, o0 in zip(outs1, outs0)]
        if DEBUG:
            mv1 = (outs1[0]["palm"].mean(1) - outs0[0]["palm"].mean(1)).norm(dim=-1).cpu().numpy()
            top = np.argsort(-mv1)[:6]
            log("  stage 1 largest palm moves (frame, w, cm):", [(int(ks[i]), round(float(w_np[i]), 2), round(float(mv1[i]) * 100, 1)) for i in top])
    starts = [[a.snapshot() for a in actors]]
    if not hand_hand:
        if anc is not None:
            q_t, n_t = anc["P_np"], anc["n_np"]
            res["anchor_joint"] = int(anchor.get("joint", 9))
        else:
            palm1 = outs1[0]["palm"].mean(1).cpu().numpy()
            q_t, n_t, anchor_j = anchored_targets(fit[o], poses[o], ks, palm1, VO, JO, surf, device)
            res["anchor_joint"] = anchor_j
        contact["tgt"] = torch.tensor(q_t + n_t * cfg.palm_offset_m, **f64)
        contact["n"] = torch.tensor(n_t, **f64)
        n1 = palm_normal(outs1[0]["joints"].cpu().numpy(), s)
        ang = np.degrees(np.arccos(np.clip((n1 * -n_t).sum(1), -1, 1)))
        res["palm_angle_deg_stage1"] = float(np.median(ang[core_np])) if core_np.any() else float(np.median(ang))
        if res["palm_angle_deg_stage1"] > cfg.flip_deg and anc is None:
            twist_start(actors[0], w, n_t)
            starts.append([a.snapshot() for a in actors])
            actors[0].restore(starts[0][0])
    tried = []
    for si, st in enumerate(starts):
        for a, sn in zip(actors, st):
            a.restore(sn)
        loss2, terms2 = optimise(2, cfg.iters2)
        tried.append((loss2, terms2, [a.snapshot() for a in actors]))
    best = min(range(len(tried)), key=lambda i: tried[i][0] - tried[i][1].get("_reg", 0.0))
    for a, sn in zip(actors, tried[best][2]):
        a.restore(sn)
    # polish: short, with the surface planes refreshed every few steps (no stale plane lets a point slip into the partner)
    base_cfg = cfg
    cfg = RefineCfg(**{**asdict(base_cfg), "refresh": 4, "lr2": base_cfg.lr2 / 3})
    lossp, termsp = optimise(2, cfg.iters_polish)
    tried[best] = (lossp, termsp, [a.snapshot() for a in actors])
    cfg = base_cfg
    # push-out: where the exact test (winding numbers) finds a hand deeper in the partner than before, polish again with the exact sign in
    # every plane, refreshed every 2 steps, 4x the penetration weight
    with torch.no_grad():
        geo1 = exact_geo([a.forward(w) for a in actors], partner_meshes(True))
    res["push_out"] = False
    if any(is_new_pen(g0, g1) for g0, g1 in zip(geo_in, geo1)):
        cfg = RefineCfg(**{**asdict(base_cfg), "refresh": 2, "w_pen": base_cfg.w_pen * 4, "lr2": base_cfg.lr2 / 3})
        push_exact[0] = True
        lossp, termsp = optimise(2, 30)
        push_exact[0] = False
        cfg = base_cfg
        tried[best] = (lossp, termsp, [a.snapshot() for a in actors])
        res["push_out"] = True
        with torch.no_grad():
            geo1 = exact_geo([a.forward(w) for a in actors], partner_meshes(True))
    res["stage2"] = {"loss": tried[best][0], **tried[best][1], "starts": len(starts), "chosen_start": int(best),
                     "losses": [t[0] for t in tried]}
    with torch.no_grad():
        outs = [a.forward(w) for a in actors]
        refresh(outs)
        res["after"] = {"ma2d_px": [t2.median_px(oo["lm"]) if t2.items else float("nan") for oo, t2 in zip(outs, twod)],
                        "ma2d_px_core_all": [t2.core_median(oo["lm"], core_np, a.n_hand_lm) if t2.items else float("nan")
                                             for a, oo, t2 in zip(actors, outs, twod)],
                        "coco_wrist_px_fit": [coco_px(a, "wrist", cams) for a in actors],
                        "coco_wrist_px_test": [coco_px(a, "wrist", test_cams) for a in actors],
                        "coco_elbow_px_test": [coco_px(a, "elbow", test_cams) for a in actors],
                        "coco_wrist_px_core": [coco_px(a, "wrist", test_cams, True) for a in actors],
                        "coco_elbow_px_core": [coco_px(a, "elbow", test_cams, True) for a in actors],
                        "target_wrist_cm": [target_dist(a, g, "wrist") for a, g in zip(actors, tg)],
                        "target_elbow_cm": [target_dist(a, g, "elbow") for a, g in zip(actors, tg)],
                        "pen_exact_cm": [list(pen_stats(g)) for g in geo1],
                        "palm_gap_exact_cm": [float(np.median(g[1][core_np])) * 100 if core_np.any() else float("nan") for g in geo1]}
        pen_a, moves, turns, wmoves, jit = [], [], [], [], []
        for a, pl, oo, o0 in zip(actors, planes, outs, outs0):
            sd, _ = penetration(a, oo, pl)
            core = torch.as_tensor(core_np, device=device)
            pen_a.append(float(torch.relu(-sd[:, :a.nh])[core].max()) * 100 if core.any() else 0.0)
            moves.append(float((oo["palm"].mean(1) - o0["palm"].mean(1)).norm(dim=-1).max()))
            wmoves.append(float((oo["joints"][:, WRIST[a.side]] - o0["joints"][:, WRIST[a.side]]).norm(dim=-1).max()))
            jit.append((_jitter_mm(o0["joints"][:, WRIST[a.side]].cpu().numpy().astype(float)),
                        _jitter_mm(oo["joints"][:, WRIST[a.side]].cpu().numpy().astype(float)),
                        float(np.mean([_jitter_mm(o0["tips5"][:, t].cpu().numpy().astype(float)) for t in range(5)])),
                        float(np.mean([_jitter_mm(oo["tips5"][:, t].cpu().numpy().astype(float)) for t in range(5)]))))
            turns.append(float(np.degrees((a.delta * w[:, None, None]).norm(dim=-1)[:, :2].max().cpu().numpy())))
        res["after"]["pen_cm_core_planes"] = pen_a
        res["after"]["hand_gap_cm"] = hand_gap_cm(outs)
        res["palm_move_max_m"] = moves
        res["wrist_move_max_m"] = wmoves
        res["jitter_mm"] = [{"wrist_before": a_, "wrist_after": b_, "tips_before": c_, "tips_after": d_} for a_, b_, c_, d_ in jit]
        res["turn_max_deg"] = turns
        if not hand_hand:
            pc = outs[0]["palm"].mean(1)
            core = torch.as_tensor(core_np, device=device)
            res["contact_err_cm_core"] = float((pc - contact["tgt"]).norm(dim=-1)[core].median()) * 100 if core.any() else None
            n2 = palm_normal(outs[0]["joints"].cpu().numpy(), s)
            res["palm_angle_deg_after"] = float(np.median(np.degrees(np.arccos(np.clip((n2 * -contact["n"].cpu().numpy()).sum(1), -1, 1)))[core_np]))
        for a in actors:
            if a.u is not None:
                res.setdefault("curl_median", []).append([float(x) for x in a.curl().median(0).values.cpu().numpy()])
    why, notes = [], []
    with torch.no_grad():
        outs = [a.forward(w) for a in actors]
    res["after"]["ma2d_px_supported"] = []
    res["before"]["ma2d_px_supported"] = []
    for i, (a, t2) in enumerate(zip(actors, twod)):
        sup, med0 = gate_masks[i]
        med1, _ = t2.per_frame(outs[i]["lm"], a.n_hand_lm)
        m = sup & core_np & np.isfinite(med0) & np.isfinite(med1)
        enough = m.sum() >= cfg.guard_min_frames
        b2 = float(np.median(med0[m])) if enough else float("nan")
        a2 = float(np.median(med1[m])) if enough else float("nan")
        res["before"]["ma2d_px_supported"].append(b2)
        res["after"]["ma2d_px_supported"].append(a2)
        if np.isfinite(b2) and a2 > b2 + cfg.ma2d_slack_px:
            (notes if keep else why).append(f"MAMMA hand landmarks (where they support the fit) {b2:.1f} -> {a2:.1f} px")
        # every core frame (the review's measure): MAMMA's own hand points, then the independent COCO wrist / elbow
        b3, a3 = res["before"]["ma2d_px_core_all"][i], res["after"]["ma2d_px_core_all"][i]
        if np.isfinite(b3) and np.isfinite(a3) and a3 > b3 + cfg.ma2d_all_slack_px:
            (notes if keep else why).append(f"MAMMA hand landmarks (all core frames) {b3:.1f} -> {a3:.1f} px")
        for which in ("wrist", "elbow"):
            b4, a4 = res["before"][f"coco_{which}_px_core"][i], res["after"][f"coco_{which}_px_core"][i]
            if np.isfinite(b4) and np.isfinite(a4) and a4 > b4 + cfg.coco_slack_px:
                (notes if keep else why).append(f"COCO {which} (independent, core) {b4:.1f} -> {a4:.1f} px")
    for i in range(len(actors)):
        gb, ga = res["before"].get("hand_gap_cm", [np.nan] * 2)[i], res["after"].get("hand_gap_cm", [np.nan] * 2)[i]
        if np.isfinite(gb) and np.isfinite(ga) and ga > max(gb, 0.0) + cfg.float_slack_cm:
            why.append(f"hand left the partner: gap {gb:.1f} -> {ga:.1f} cm")
        jb = res["jitter_mm"][i]
        for nm in ("wrist", "tips"):
            b_, a_ = jb[f"{nm}_before"], jb[f"{nm}_after"]
            rel, ab = (cfg.jitter_rel, cfg.jitter_abs_mm) if nm == "wrist" else (cfg.tips_jitter_rel, cfg.tips_jitter_abs_mm)
            if np.isfinite(b_) and np.isfinite(a_) and a_ > max(rel * b_, b_ + ab):
                why.append(f"{nm} jitter {b_:.1f} -> {a_:.1f} mm")
        if res["wrist_move_max_m"][i] > cfg.max_wrist_move_m:
            why.append(f"wrist moved {res['wrist_move_max_m'][i] * 100:.0f} cm")
        if is_new_pen(geo_in[i], geo1[i]):
            (b90, bmx), (a90, amx) = pen_stats(geo_in[i]), pen_stats(geo1[i])
            why.append(f"new penetration (exact): p90 {b90:.2f} -> {a90:.2f} cm, max {bmx:.2f} -> {amx:.2f} cm")
    for i in range(len(actors)):
        b2, a2 = res["before"]["target_wrist_cm"][i], res["after"]["target_wrist_cm"][i]
        if np.isfinite(b2) and np.isfinite(a2) and a2 > b2 + cfg.target_slack_cm:
            why.append(f"wrist further from the multi-view COCO wrist {b2:.1f} -> {a2:.1f} cm")
        if res["turn_max_deg"][i] > cfg.max_turn_deg:
            why.append(f"arm turn {res['turn_max_deg'][i]:.0f} deg")
    # speed against the INPUT fit (everything adopted so far included): no new jump, no whip; and the ramp this event's own move needs
    res["speed"] = []
    ramp_need = int(cfg.ramp)
    ce = np.flatnonzero(core_np)
    for i, a in enumerate(actors):
        J_in = np.asarray(fit[a.role]["joints"], float)[ks]
        J_b = outs0[i]["joints"].cpu().numpy().astype(float)
        J_new = outs[i]["joints"].cpu().numpy().astype(float)
        ref = (speed_ref or {}).get(f"{a.role}_{a.side}", {"wrist": 2.0, "palm": 2.2})
        rec = {}
        for k in ("wrist", "palm"):
            ti, tb, tn = hand_tracks(J_in, a.side)[k], hand_tracks(J_b, a.side)[k], hand_tracks(J_new, a.side)[k]
            sv = speed_violations(ti, tn, ref[k], cfg.add_speed_max_mps, cfg.speed_margin_mps)
            rec[k] = {"max_added_mps": sv["max_add"], "max_excess_mps": sv["max_excess"], "jumps": int(len(sv["jump"])), "fast": int(len(sv["fast"]))}
            bad = np.union1d(sv["jump"], sv["fast"])
            if len(bad):
                j = int(bad[np.argmax(sv["add"][bad])])
                where = "ramp" if (w_np[j] < 0.99 or w_np[j + 1] < 0.99) else "core"
                why.append(f"{'jump' if len(sv['jump']) else 'whip'} ({where}): {a.role} {a.side} {k} at {ks[j] / HZ:.2f} s added "
                           f"{sv['add'][j]:.2f} m/s, speed {sv['v_new'][j]:.2f} (input {sv['v_in'][j]:.2f}, p99 {ref[k]:.2f})")
            if len(ce):
                ends = np.concatenate([ce[:3], ce[-3:]])
                D = float(np.linalg.norm(tn[ends] - tb[ends], axis=1).max())
                ramp_need = max(ramp_need, ramp_for(D, cfg.v_ramp_mps, cfg.ramp, cfg.ramp_max))
        res["speed"].append(rec)
    res["ramp_needed"] = int(ramp_need)
    res["adopted"] = not why
    res["guard"] = why
    res["notes"] = notes
    res["new_pose"] = {a.role: (ks, a.pose_np(w_np), list(a.chain) + list(a.fj)) for a in actors}
    return res


FOLDS = {"foldA": ("00", "02", "05"), "foldB": ("01", "03", "06")}


PASS2 = {"hand_only": True, "w_2d": 4.0, "w_hand_gap": 2.4e3, "w_rot": (1.0, 1.0, 4.0), "iters2": 90}


def run_pass2(n: int = 3) -> dict:
    """the events the production run left as they were (a 2D or target guard): hand-only (shoulder and elbow kept, the wrist turns a little,
    the fingers take the hold, no penetration), from the production result; merged into v3/smplx_contact/ (refine_pass2.json)"""
    rep = json.loads((OUT / "refine_all.json").read_text(encoding="utf-8"))
    ids = sorted(r["event"] for r in rep["results"] if not r.get("adopted"))
    log(f"pass 2: {len(ids)} events left by the production run")
    return run_parallel("pass2", n, only=ids, write=True, cfg_over=PASS2, start_from=FIT_OUT)


PASS3 = {"tips_jitter_rel": 1.7, "tips_jitter_abs_mm": 5.0}


def run_pass3(n: int = 3) -> dict:
    """the events whose ONLY veto in the production run was the fingertip-jitter guard (the gross 'hand through the partner' corrections
    among them, 2026-10-21): again with the full refinement and a looser fingertip guard (1.7x / +5 mm; the wrist guard unchanged), from
    the pass-2 result; merged into v3/smplx_contact/ (refine_pass3.json)"""
    rep = json.loads((OUT / "refine_all.json").read_text(encoding="utf-8"))
    p2 = OUT / "refine_pass2.json"
    done2 = {r["event"] for r in json.loads(p2.read_text(encoding="utf-8"))["results"] if r.get("adopted")} if p2.exists() else set()
    ids = sorted(r["event"] for r in rep["results"] if not r.get("adopted") and r.get("guard") and r["event"] not in done2 and
                 all(g.startswith("tips jitter") for g in r["guard"]))
    log(f"pass 3: {len(ids)} events vetoed only by the fingertip jitter")
    return run_parallel("pass3", n, only=ids, write=True, cfg_over=PASS3, start_from=FIT_OUT)


def run_pass4() -> dict:
    """post-check: every adopted event measured against the INPUT with the FINAL fit (validate's rule); an event whose hand now sinks deeper
    than in the input (the partner was moved by another event, or a later pass changed the same hand) is solved again on top of the final
    fit, with the input-relative penetration guard (refine_pass4.json, merged into v3/smplx_contact/). Run validate first."""
    v = json.loads((OUT / "validate.json").read_text(encoding="utf-8"))
    ids = sorted({x["event"] for x in v["summary"]["safety"]["new_penetration"]})
    log(f"pass 4: {len(ids)} events with new penetration against the final fit: {ids}")
    if not ids:
        rep = {"what": "pass 4: nothing to do", "summary": {"refined": 0, "adopted": 0}, "results": [], "skipped": []}
        write_json(OUT / "refine_pass4.json", rep)
        return rep
    return run_refine(only=ids, out_dir=FIT_OUT, tag="pass4", start_from=FIT_OUT)


def adopted_results(tag: str = "all") -> list:
    """every adopted result of the passes (refine_<tag>, pass2, pass3, pass4), the latest pass per event, minus curated vetoes"""
    cur = load_curation()
    out = {}
    for t in (tag, "pass2", "pass3", "pass4"):
        pth = OUT / f"refine_{t}.json"
        if not pth.exists() or (t == "pass4" and pth.stat().st_mtime < (OUT / f"refine_{tag}.json").stat().st_mtime):
            continue
        for r in json.loads(pth.read_text(encoding="utf-8"))["results"]:
            if r.get("adopted") and str(r["event"]) not in cur["veto"]:
                out[r["event"]] = dict(r, _pass={"all": 1, "pass2": 2, "pass3": 3, "pass4": 4}.get(t, 1))
    return sorted(out.values(), key=lambda r: r["frames"][0])


def revert_events(ids: list[int], why: str) -> dict:
    """post-check: put the hands of these adopted events back to the input fit over their windows - together with every other adopted event
    that moves the same hand in an overlapping window (the closure, reported: a later solve was made on top of it) - add them all to
    curation.json's veto with the reason, and rewrite v3/smplx_contact/. Exact: outside the windows nothing changes."""
    from .hand_kin import SHOULDER, ELBOW, WRIST, FINGER_JOINTS
    events = {e["id"]: e for e in curated_events(load_events())}
    res = adopted_results()

    def hands(r):
        h = {(events[r["event"]]["role"], events[r["event"]]["side"])}
        if r.get("twin") is not None:
            h.add((events[r["twin"]]["role"], events[r["twin"]]["side"]))
        return h
    todo = [r for r in res if r["event"] in set(ids)]
    closure = {r["event"]: r for r in todo}
    grew = True
    while grew:
        grew = False
        for r in res:
            if r["event"] in closure:
                continue
            for c in list(closure.values()):
                if hands(r) & hands(c) and r["frames"][0] <= c["frames"][1] and c["frames"][0] <= r["frames"][1]:
                    closure[r["event"]] = r
                    grew = True
                    break
    fa, fb = load_fit(SRC_FIT), load_fit(FIT_OUT)
    poses = {r: full_pose(fb[r]).copy() for r in ROLES}
    pin = {r: full_pose(fa[r]) for r in ROLES}
    for r in closure.values():
        a0, a1 = r["frames"]
        for role, side in hands(r):
            js = [SHOULDER[side], ELBOW[side], WRIST[side], *FINGER_JOINTS[side]]
            poses[role][a0:a1 + 1][:, js] = pin[role][a0:a1 + 1][:, js]
    c = json.loads(CURATION.read_text(encoding="utf-8")) if CURATION.exists() else {"veto": {}, "keep": {}, "overrides": {}}
    for i in closure:
        c.setdefault("veto", {})[str(i)] = (why if i in ids else f"post-check: reverted with #{', #'.join(str(x) for x in ids)} "
                                            f"(the same hand in an overlapping window): {why}")
    write_json(CURATION, c)
    info = write_fit(fa, poses, FIT_OUT)
    log(f"reverted {sorted(closure)} (asked {sorted(ids)}) -> {FIT_OUT}")
    return {"reverted": sorted(closure), "asked": sorted(ids), "written": info}


def run_folds() -> dict:
    """Held-out check of the COCO-driven corrections: the events where the multi-view COCO wrist was a target in the production run are
    refined again with the 2D of half of the phones (foldA / foldB) and judged on the other half's COCO wrists / elbows (refine_<fold>.json)."""
    rep = json.loads((OUT / "refine_all.json").read_text(encoding="utf-8"))
    ids = sorted({r["event"] for r in rep["results"] if r.get("adopted") and any((g.get("coco_wrist_used") or 0) > 0.05 for g in r.get("gate", []))})
    log(f"folds: {len(ids)} adopted COCO-driven events")
    out = {}
    for tag, cams in FOLDS.items():
        out[tag] = run_parallel(tag, 3, cams=list(cams), only=ids, write=False)["summary"]
    return out


def load_events() -> list:
    return json.loads((OUT / "events.json").read_text(encoding="utf-8"))["events"]


CURATION = OUT / "curation.json"


def load_curation() -> dict:
    """v3/hand_contact/curation.json: the decisions taken by LOOKING at the phone crops (before / after sheets), each with its reason:
    veto {id: why} (left as the input fit), keep {id: why} (the 2D guards are waived after a visual check; geometry and speed guards still
    hold), overrides {id: {merge: [ids], hold, region, anchor: {joint, local_xy, side}, cfg: {RefineCfg fields}, why}}"""
    if CURATION.exists():
        c = json.loads(CURATION.read_text(encoding="utf-8"))
    else:
        c = {}
    return {"veto": {str(k): v for k, v in (c.get("veto") or {}).items()}, "keep": {str(k): v for k, v in (c.get("keep") or {}).items()},
            "overrides": {str(k): v for k, v in (c.get("overrides") or {}).items()}}


def curated_events(events: list, cur: dict | None = None) -> list:
    """events.json with the curation's overrides applied: an override's merge ids are folded into its event (one hold over the union of
    their frames: the detector split one hold at short gaps), its hold / region / anchor / cfg replace the detected ones"""
    cur = cur if cur is not None else load_curation()
    byid = {e["id"]: dict(e) for e in events}
    drop = set()
    for k, ov in cur.get("overrides", {}).items():
        i = int(k)
        e = byid[i]
        ids = [i] + [int(x) for x in ov.get("merge", [])]
        e["i0"], e["i1"] = min(byid[j]["i0"] for j in ids), max(byid[j]["i1"] for j in ids)
        e["t0"], e["t1"] = min(byid[j]["t0"] for j in ids), max(byid[j]["t1"] for j in ids)
        if "core_s" in ov:                                     # a curated core (s on the grid): e.g. only the part the override is right for
            a_, b_ = (float(x) for x in ov["core_s"])
            e["i0"], e["i1"] = int(round(a_ * HZ)), int(round(b_ * HZ))
            e["t0"], e["t1"] = round(e["i0"] / HZ, 3), round(e["i1"] / HZ, 3)
        e["duration_s"] = round((e["i1"] - e["i0"] + 1) / HZ, 3)
        for key in ("hold", "region", "anchor", "cfg"):
            if key in ov:
                e[key] = ov[key]
        if ov.get("merge"):
            e["merged"] = [int(x) for x in ov["merge"]]
            drop |= set(e["merged"])
        e["override"] = ov.get("why", True)
    return [e for i, e in sorted(byid.items()) if i not in drop]


def refinable(ev: dict, fit: dict, veto: dict | None = None) -> tuple[bool, str]:
    """which events the refinement touches: holds and touches the 2D does not contradict, on frames both dancers are fitted multi-view,
    not vetoed in curation.json"""
    if veto and str(ev["id"]) in veto:
        return False, f"curated veto: {veto[str(ev['id'])]}"
    if ev["hold"] in ("none", "hand_on_leg"):
        return False, f"hold {ev['hold']}"
    if ev["evidence_median"] < 0.35:
        return False, f"2D contradicts (evidence {ev['evidence_median']:.2f})"
    sl = slice(ev["i0"], ev["i1"] + 1)
    for r in ROLES:
        qf = np.asarray(fit[r]["quality_flag"])[sl]
        if (qf < 2).mean() > 0.5:
            return False, f"{r} fitted from fewer than 2 views"
    if ev["duration_s"] < 0.2:
        return False, "shorter than 0.2 s"
    return True, ""


def split_parts(todo: list, n: int, ramp: int, T: int) -> list[list]:
    """todo [(event, twin)] -> n parts: two events that move the same hand in overlapping windows (event +- ramp) are joined (connected
    components); a component stays in one part, in time order, so no two parts ever move the same arm in the same frames. Components are dealt
    out by frame work, largest first. (A palm hold whose target is the partner's arm can see that arm before another part moved it: tiny.)"""
    def win(x):
        e, tw = x
        i0, i1 = (min(e["i0"], tw["i0"]), max(e["i1"], tw["i1"])) if tw else (e["i0"], e["i1"])
        return max(0, i0 - ramp - 2), min(T - 1, i1 + ramp + 2)

    def hands(x):
        e, tw = x
        return {(e["role"], e["side"])} | ({(tw["role"], tw["side"])} if tw else set())
    m = len(todo)
    parent = list(range(m))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    W = [win(x) for x in todo]
    H = [hands(x) for x in todo]
    for i in range(m):
        for j in range(i + 1, m):
            if H[i] & H[j] and W[i][0] <= W[j][1] and W[j][0] <= W[i][1]:
                parent[find(i)] = find(j)
    comps: dict = {}
    for i in range(m):
        comps.setdefault(find(i), []).append(todo[i])

    def work(xs):
        return sum(win(x)[1] - win(x)[0] + 60 for x in xs)
    parts: list[list] = [[] for _ in range(n)]
    for g in sorted(comps.values(), key=work, reverse=True):
        i = int(np.argmin([work(p_) for p_ in parts]))
        parts[i] += g
    return parts


def run_refine(device: str = "cuda", only: list[int] | None = None, cfg: RefineCfg | None = None, out_dir: Path | None = FIT_OUT,
               cams: list[str] | None = None, tag: str = "all", part: tuple[int, int] | None = None, start_from: Path | None = None) -> dict:
    """Refine every refinable event with the 2D of `cams` (default all six phones); writes OUT/refine_<tag>.json and (out_dir) the fit."""
    import torch
    from . import gpu
    cfg = cfg or RefineCfg()
    cams = list(cams or CAMS)
    fit = load_fit()
    cur = load_curation()
    events = curated_events(load_events(), cur)
    byid = {e["id"]: e for e in events}
    speed_ref = input_speed_ref(fit)
    poses = {r: full_pose(fit[r]).copy() for r in ROLES}
    if start_from is not None:
        st = load_fit(Path(start_from))
        poses = {r: full_pose(st[r]).copy() for r in ROLES}
    obs = Obs()
    judge = CocoJudge()
    bias = calibrate_coco_bias(fit, obs, judge)
    todo, skipped = [], []
    done_twins = set()
    for e in events:
        if only is not None and e["id"] not in only:
            continue
        ok, why = refinable(e, fit, cur["veto"])
        if not ok:
            skipped.append({"event": e["id"], "why": why})
            continue
        if e["id"] in done_twins:
            continue
        tw = byid.get(e.get("twin")) if e.get("twin") is not None else None
        if tw is not None and tw["hold"] in HAND_HOLDS and e["hold"] in HAND_HOLDS and refinable(tw, fit, cur["veto"])[0]:
            done_twins.add(tw["id"])
        else:
            tw = None
        todo.append((e, tw))
    if part is not None:
        todo = split_parts(todo, part[1], cfg.ramp_max, len(fit["lead"]["t"]))[part[0]]     # ramps can grow to ramp_max
    # hand-hand pairs first (they move both dancers), then the palm holds, each in time order
    todo.sort(key=lambda x: (x[1] is None, x[0]["t0"]))
    results = []
    tried = {r: poses[r].copy() for r in ROLES} if (out_dir is None and __import__("os").environ.get("HC_KEEP_TRIED")) else None
    t0 = time.time()
    with gpu.exclusive(f"hand_contact refine {tag}"):
        faces = Body(fit["lead"], "cpu").faces
        surf = GpuSurface(faces, len(vertex_joint()), device)
        for n, (e, tw) in enumerate(todo):
            te = time.time()
            attempts = []
            c = RefineCfg(**{**asdict(cfg), **(e.get("cfg") or {})})          # a curated override may set its own RefineCfg fields
            keep = str(e["id"]) in cur["keep"] or bool(e.get("override"))
            for attempt in range(3):
                try:
                    res = refine_event(fit, poses, e, tw, obs, judge, bias, surf, c, cams, device, speed_ref, keep)
                except Exception as ex:  # noqa: BLE001 - one event must not stop the run; reported
                    import traceback
                    res = {"event": e["id"], "adopted": False, "guard": [f"{type(ex).__name__}: {ex}"], "new_pose": {},
                           "traceback": traceback.format_exc()[-1500:]}
                attempts.append({"w_2d": c.w_2d, "ramp": c.ramp, "adopted": bool(res.get("adopted")), "guard": res.get("guard")})
                if res.get("adopted"):
                    break
                g = res.get("guard") or []
                # a jump / whip with a move the blend ramp is too short for: again with the ramp it needs (up to ramp_max)
                if res.get("ramp_needed", c.ramp) > c.ramp and any(x.startswith(("jump", "whip")) for x in g):
                    c = RefineCfg(**{**asdict(c), "ramp": int(res["ramp_needed"])})
                    continue
                # a retry with a 4x stronger 2D term only when a 2D guard failed (not for a crash or an arm-turn veto)
                if not cfg.hand_only and c.w_2d == cfg.w_2d and any("px" in x for x in g):
                    c = RefineCfg(**{**asdict(c), "w_2d": cfg.w_2d * 4, "iters2": max(60, cfg.iters2 * 2 // 3)})
                    continue
                break
            res["attempts"] = attempts
            if res.get("adopted"):
                for role, (ks, P, joints) in res["new_pose"].items():
                    poses[role][np.ix_(ks, joints)] = P[:, joints]
            if tried is not None:                                  # review only: every solve applied, adopted or not
                for role, (ks, P, joints) in (res.get("new_pose") or {}).items():
                    tried[role][np.ix_(ks, joints)] = P[:, joints]
            rec = {k: v for k, v in res.items() if k != "new_pose"}
            rec["seconds"] = round(time.time() - te, 1)
            results.append(rec)
            log(f"refine [{tag}] {n + 1}/{len(todo)}: #{e['id']} {e['role']} {e['side']} {e['hold']} {e['duration_s']:.1f} s "
                f"adopted={res.get('adopted')} {res.get('guard') or ''} ({rec['seconds']} s, total {time.time() - t0:.0f} s)")
            torch.cuda.empty_cache()
    summ = {"refined": len(todo), "adopted": sum(bool(r.get("adopted")) for r in results), "skipped": len(skipped),
            "seconds": round(time.time() - t0, 1), "cams": cams}
    rep = {"what": "hand contact refinement of smplx_rf2_noslip (hand_contact.refine_event)", "config": asdict(cfg), "summary": summ,
           "results": results, "skipped": skipped}
    if out_dir is not None:
        rep["written"] = write_fit(fit, poses, out_dir)
    else:
        np.savez_compressed(OUT / f"poses_{tag}.npz", **{r: poses[r] for r in ROLES})
        if tried is not None:
            np.savez_compressed(OUT / f"poses_{tag}_tried.npz", **{r: tried[r] for r in ROLES})
    write_json(OUT / f"refine_{tag}.json", rep)
    log(f"[{tag}] refined {summ['adopted']}/{summ['refined']} events in {summ['seconds']} s")
    return rep


def merge_parts(tag: str, n: int, write: bool = True, start_from: Path | None = None, cfg_over: dict | None = None,
                cams: list[str] | None = None, seconds: float | None = None) -> dict:
    """merge the parts' refine_<tag>_p<i>.json / poses_<tag>_p<i>.npz: per frame AND joint (the parts move different arms)"""
    fit = load_fit()
    base = load_fit(Path(start_from)) if start_from else fit
    poses = {r: full_pose(base[r]).copy() for r in ROLES}
    owner = {r: np.full(poses[r].shape[:2], -1) for r in ROLES}
    results, skipped, conflicts = [], [], 0
    for i in range(n):
        rp = json.loads((OUT / f"refine_{tag}_p{i}.json").read_text(encoding="utf-8"))
        results += rp["results"]
        skipped += rp["skipped"]
        with np.load(OUT / f"poses_{tag}_p{i}.npz") as z:
            for r in ROLES:
                ch = np.abs(z[r] - full_pose(base[r])).max(2) > 1e-6            # (T,55); float32 round trips are not changes
                conflicts += int((owner[r][ch] >= 0).sum())
                owner[r][ch] = i
                poses[r][ch] = z[r][ch]
    results.sort(key=lambda r: (r.get("frames") or [0])[0])
    summ = {"refined": len(results), "adopted": sum(bool(r.get("adopted")) for r in results), "skipped": len({x["event"] for x in skipped}),
            "seconds": seconds, "cams": list(cams or CAMS), "parts": n, "joint_frame_conflicts_between_parts": conflicts}
    rep = {"what": "hand contact refinement of smplx_rf2_noslip (hand_contact.refine_event)",
           "config": asdict(RefineCfg(**(cfg_over or {}))), "start_from": str(start_from) if start_from else None, "summary": summ,
           "results": results, "skipped": skipped}
    if write:
        rep["written"] = write_fit(fit, poses, FIT_OUT)
    else:
        np.savez_compressed(OUT / f"poses_{tag}.npz", **{r: poses[r] for r in ROLES})
    write_json(OUT / f"refine_{tag}.json", rep)
    log(f"[{tag}] {summ['adopted']}/{summ['refined']} adopted ({n} parts, {conflicts} conflicting joint-frames)")
    return rep


def run_parallel(tag: str = "all", n: int = 3, cams: list[str] | None = None, only: list[int] | None = None, write: bool = True,
                 cfg_over: dict | None = None, start_from: Path | None = None) -> dict:
    """run_refine in n processes on disjoint time parts of the events (the GPU is launch-bound: parallel processes use it better); this
    process holds work/gpu.lock, the children share it (DANCECAP_GPU_SHARED). Then the parts are merged: refine_<tag>.json, and (write)
    the fit v3/smplx_contact/, else poses_<tag>.npz."""
    import os
    import subprocess
    from . import gpu
    args = {"cams": list(cams) if cams else None, "only": only, "cfg": cfg_over or {}, "start_from": str(start_from) if start_from else None}
    (OUT / f"_parts_{tag}.json").write_text(json.dumps(args), encoding="utf-8")
    t0 = time.time()
    with gpu.exclusive(f"hand_contact refine {tag} ({n} parts)"):
        env = {**os.environ, "DANCECAP_GPU_SHARED": "1"}
        procs = []
        for i in range(n):
            lf = open(OUT / f"refine_{tag}_p{i}.log", "w", encoding="utf-8")
            procs.append((subprocess.Popen([sys.executable, "-m", "dancecap.hand_contact", "refine-part", tag, str(i), str(n)],
                                           cwd=str(PACKAGE_ROOT), env=env, stdout=lf, stderr=subprocess.STDOUT), lf))
        for pr, lf in procs:
            pr.wait()
            lf.close()
    return merge_parts(tag, n, write, start_from, cfg_over, cams, round(time.time() - t0, 1))


def write_fit(fit: dict, poses: dict, out_dir: Path) -> dict:
    """the contract npz with the refined body / hand poses and the FK joints recomputed (every other key copied unchanged)"""
    out_dir.mkdir(parents=True, exist_ok=True)
    info = {}
    for r in ROLES:
        d = dict(fit[r])
        P = poses[r]
        T = len(P)
        changed = np.flatnonzero(np.abs(P - full_pose(fit[r])).max((1, 2)) > 0)
        d["global_orient"] = P[:, 0].copy()
        d["body_pose"] = P[:, 1:22].reshape(T, 63).copy()
        d["left_hand_pose"] = P[:, 25:40].reshape(T, 45).copy()
        d["right_hand_pose"] = P[:, 40:55].reshape(T, 45).copy()
        J = np.asarray(fit[r]["joints"], float).copy()
        if len(changed):
            b = Body(fit[r], "cpu")
            J[changed] = b.forward_np(P[changed], np.asarray(fit[r]["transl"], float)[changed], verts=False)["joints"]
        d["joints"] = J
        d["source"] = np.array(str(fit[r].get("source", "")) + " | v3 hand_contact refinement (dancecap.hand_contact, 2026-10-21) of smplx_rf2_noslip")
        np.savez(out_dir / f"{r}.npz", **d)
        info[r] = {"frames_changed": int(len(changed)), "file": str(out_dir / f"{r}.npz")}
    # FK check against the smplx package (the exporter's contract check)
    info["fk_check_mm"] = fk_check(out_dir)
    write_json(out_dir / "hand_contact.json", {"what": "smplx_rf2_noslip + the v3 hand contact refinement (dancecap.hand_contact)",
                                               "source": str(SRC_FIT), "info": info})
    return info


def fk_check(d: Path, n: int = 400) -> dict:
    """max |smplx FK joints - stored joints| (mm) on the changed frames (and a random sample), with the smplx package itself"""
    from . import mamma_dyn as md
    out = {}
    for r in ROLES:
        z = dict(np.load(d / f"{r}.npz", allow_pickle=True))
        src = dict(np.load(SRC_FIT / f"{r}.npz", allow_pickle=True))
        P, P0 = full_pose(z), full_pose(src)
        ch = np.flatnonzero(np.abs(P - P0).max((1, 2)) > 0)
        rng = np.random.default_rng(1)
        ks = np.unique(np.concatenate([ch[np.linspace(0, len(ch) - 1, min(n, len(ch))).astype(int)] if len(ch) else np.zeros(0, int),
                                       rng.choice(len(P), 100, replace=False)]))
        pose = np.concatenate([z["global_orient"], z["body_pose"], z["jaw_pose"], z["leye_pose"], z["reye_pose"], z["left_hand_pose"],
                               z["right_hand_pose"]], 1)[ks]
        _, J = md._forward(pose, z["transl"][ks], z["betas"], bool(z["flat_hand_mean"]))
        out[r] = round(float(np.abs(J - z["joints"][ks]).max()) * 1000, 3)
    return out


# ------------------------------------------------------------------------------------------------ validation

COCO_WRIST = {"left": 9, "right": 10}


class CocoJudge:
    """The owner's 17-joint COCO detections (people/teachers_<cam>.npz: a detector the fit never saw) on the grid: the wrist keypoints,
    undistorted into MAMMA's pinhole frame, at the phone frame nearest to each grid time (within 1/50 s)."""

    def __init__(self, cams=CAMS):
        from . import mamma_dyn as md
        self.cams = list(cams)
        self.z = {}
        ph = Phones()
        for c in self.cams:
            tz = np.load(WORK / "people" / f"teachers_{c}.npz", allow_pickle=True)
            cz = np.load(_camera_file(c), allow_pickle=True)
            self.z[c] = {"t": np.asarray(tz["t"], float), "kp": np.asarray(tz["kp"], float), "conf": np.asarray(tz["conf"], float),
                         "K": np.asarray(cz["K"], float), "k1": np.asarray(cz["dist"], float).ravel()}
        self.ph = ph
        self.md = md
        self._tc = {}

    def wrist(self, cam: str, role: str, side: str, k: int):
        return self.joint(cam, role, side, k, "wrist")

    def joint(self, cam: str, role: str, side: str, k: int, which: str = "wrist"):
        """(undistorted px (2,), ok) of the detected wrist / elbow at grid frame k"""
        z = self.z[cam]
        key = (cam, self.ph.owner[k])
        if key not in self._tc:
            self._tc[key] = self.md.apply_offset(z["t"], self.ph.offs.get(self.ph.owner[k], {}).get(cam))
        tc = self._tc[key]
        i = int(np.clip(np.searchsorted(tc, k / HZ), 1, len(tc) - 1))
        i = i - 1 if abs(tc[i - 1] - k / HZ) <= abs(tc[i] - k / HZ) else i
        if abs(tc[i] - k / HZ) > 1 / 50:
            return None, False
        rp = PIDX[role]
        kp = z["kp"][i, rp, COCO_JOINT[which][side]]
        if not (z["conf"][i, rp] >= 0.5 and np.isfinite(kp).all() and kp[2] >= 0.5):
            return None, False
        u = self.md.undistort_simple_radial(kp[None, :2], z["K"][i], float(z["k1"][i]))[0]
        return u, True


def hand_points_np(fit_r: dict, pose: np.ndarray, ks: np.ndarray, side: str) -> dict:
    """joints, palm centre, fingertips, the MAMMA landmark points of the hand at ks"""
    import torch
    lv = landmark_matrix().argmax(1)
    parts = landmark_parts()
    hid = lv[parts[f"{side}_hand"]]
    kin = kin_for(fit_r["betas"], "cpu", hid)
    with torch.no_grad():
        o = kin.forward(torch.tensor(pose[ks], dtype=torch.float64), torch.tensor(np.asarray(fit_r["transl"], float)[ks]), want_verts=True)
    tips = o["tips"].numpy()
    return {"joints": o["joints"].numpy(), "lm": o["verts"].numpy(), "tips": tips[:, :5] if side == "left" else tips[:, 5:]}


def _jitter_mm(X: np.ndarray) -> float:
    """RMS of the second difference (mm per frame^2) of a (W,3) track: the high-frequency shake; 0 for < 3 frames"""
    if len(X) < 3:
        return float("nan")
    a = X[2:] - 2 * X[1:-1] + X[:-2]
    return float(np.sqrt((np.linalg.norm(a, axis=-1) ** 2).mean()) * 1000)


def _coco_errors(judge, obs: Obs, X: np.ndarray, ks: np.ndarray, role: str, side: str, which: str, cams) -> np.ndarray:
    """px errors of the points X (W,3) against the COCO detections of `which` in `cams` at ks (one value per detection)"""
    es = []
    for c in cams:
        z = obs.z[c]
        uv, d = obs.project(c, X[:, None], ks)
        for i, k in enumerate(ks):
            u, ok = judge.joint(c, role, side, int(k), which)
            if ok and z["ok"][k] and d[i, 0] > 0.1:
                es.append(float(np.linalg.norm(uv[i, 0] - u)))
    return np.array(es)


def validate(device: str = "cuda", tag: str = "all") -> dict:
    """before (smplx_rf2_noslip) vs after (v3/smplx_contact) on every adopted event, on the core frames of the event:
      2D in the 6 phones: MAMMA's hand landmarks (the fit's own input: all frames, and the frames where they supported the fitted hand),
      the owner's COCO wrists / elbows (bias-corrected joints; an independent detector wherever the refinement did not use it as a target;
      where it did, see the held-out folds), PCK@20 px of the COCO wrist;
      penetration / gap (exact mesh distances + winding numbers, hand_geometry), the palm patch's gap; wrist / palm / fingertip jitter;
      the safety checks of the 2026-10-21 review: events with MAMMA's hand points > 3 px worse or the COCO wrist / elbow > 4 px worse, new
      penetration, the flat holds' palm gap, and speed_audit over the whole class (no added jump, no whip, every hand)."""
    from . import gpu
    from .hand_kin import WRIST, ELBOW
    fa, fb = load_fit(SRC_FIT), load_fit(FIT_OUT)
    cur = load_curation()
    rep = json.loads((OUT / f"refine_{tag}.json").read_text(encoding="utf-8"))
    p2 = OUT / "refine_pass2.json"
    rep2 = json.loads(p2.read_text(encoding="utf-8")) if p2.exists() else {"results": []}
    p3 = OUT / "refine_pass3.json"
    rep3 = json.loads(p3.read_text(encoding="utf-8")) if p3.exists() else {"results": []}
    p4 = OUT / "refine_pass4.json"
    rep4 = json.loads(p4.read_text(encoding="utf-8")) if p4.exists() and p4.stat().st_mtime > (OUT / f"refine_{tag}.json").stat().st_mtime         else {"results": []}
    events = {e["id"]: e for e in curated_events(load_events(), cur)}
    obs = Obs()
    judge = CocoJudge()
    bias = calibrate_coco_bias(fa, obs, judge)
    parts = landmark_parts()
    adopted = adopted_results(tag)                             # the latest pass per event, minus curated (post-check) vetoes
    allk = np.unique(np.concatenate([np.arange(r["frames"][0], r["frames"][1] + 1) for r in adopted])) if adopted else np.zeros(0, int)
    with gpu.exclusive("hand_contact validate"):
        geo = {name: hand_geometry(f, device, ks=allk) for name, f in (("before", fa), ("after", fb))}
    kpos = {k: i for i, k in enumerate(allk)}
    rows = []
    for res in adopted:
        evs = [events[res["event"]]] + ([events[res["twin"]]] if res.get("twin") is not None else [])
        coco_used = [g.get("coco_wrist_used", 0.0) for g in res.get("gate", [])]
        for j, ev in enumerate(evs):
            r, s = ev["role"], ev["side"]
            core = np.arange(ev["i0"], ev["i1"] + 1)
            row = {"event": ev["id"], "role": r, "side": s, "hold": ev["hold"], "region": ev["region"], "t0": ev["t0"], "t1": ev["t1"],
                   "frames": int(len(core)), "pass": res["_pass"], "coco_target_share": coco_used[j] if j < len(coco_used) else None,
                   "flat": ev["hold"] in FLAT_HOLDS and res.get("twin") is None, "merged": ev.get("merged"), "override": bool(ev.get("override")),
                   "curated_keep": str(ev["id"]) in cur["keep"], "notes": res.get("notes"), "ramp": res.get("ramp"),
                   "window": res.get("frames"),
                   "palm_angle_deg_before_after": [res.get("palm_angle_deg_stage1"), res.get("palm_angle_deg_after")] if j == 0 else None,
                   "curl": (res.get("curl_median") or [None, None])[j] if res.get("curl_median") and j < len(res["curl_median"]) else None}
            hid = parts[f"{s}_hand"]
            for name, f in (("before", fa), ("after", fb)):
                hp = hand_points_np(f[r], full_pose(f[r]), core, s)
                errs = []
                for c in obs.cams:
                    z = obs.z[c]
                    uv, dep = obs.project(c, hp["lm"], core)
                    o = z["lm"][core, PIDX[r]][:, hid]
                    ok = (z["vis"][core, PIDX[r]][:, hid] > 0.5) & (np.abs(o).sum(-1) > 0) & (dep > 0.1) & z["ok"][core][:, None]
                    errs.append(np.linalg.norm(uv - o, axis=-1)[ok])
                e = np.concatenate(errs) if errs else np.zeros(0)
                row[f"hand_lm_px_{name}"] = float(np.median(e)) if len(e) else None
                J = hp["joints"]
                Fr = _forearm_frame(J, s)
                for which, jj in (("wrist", WRIST[s]), ("elbow", ELBOW[s])):
                    b = np.asarray(bias.get(f"{r}_{s}_{which}", [0, 0, 0]), float)
                    X = J[:, jj] + np.einsum("kij,j->ki", Fr, b)
                    ce = _coco_errors(judge, obs, X, core, r, s, which, obs.cams)
                    row[f"coco_{which}_px_{name}"] = float(np.median(ce)) if len(ce) else None
                    if which == "wrist":
                        row[f"coco_wrist_pck20_{name}"] = float((ce < 20).mean()) if len(ce) else None
                gi = np.array([kpos[k] for k in core])
                g = geo[name][(r, s)]
                row[f"gap_cm_{name}"] = float(np.median(g["gap"][gi])) * 100
                row[f"pen_cm_max_{name}"] = float(g["pen"][gi].max()) * 100
                row[f"pen_cm_p90_{name}"] = float(np.percentile(g["pen"][gi], 90)) * 100
                row[f"pen_cm_med_{name}"] = float(np.median(g["pen"][gi])) * 100
                row[f"arm_pen_cm_p90_{name}"] = float(np.percentile(g["arm_pen"][gi], 90)) * 100
                row[f"palm_gap_cm_{name}"] = float(np.median(g["palm_gap"][gi])) * 100
                # jitter over the whole refined window (the blend ramps included: a blend must not shake)
                win = np.arange(res["frames"][0], res["frames"][1] + 1)
                hw = hand_points_np(f[r], full_pose(f[r]), win, s)
                row[f"wrist_jitter_mm_{name}"] = _jitter_mm(hw["joints"][:, WRIST[s]])
                row[f"palm_jitter_mm_{name}"] = _jitter_mm(palm_centre(hw["joints"], s))
                row[f"tip_jitter_mm_{name}"] = float(np.mean([_jitter_mm(hw["tips"][:, i]) for i in range(5)]))
            def _at(lst, i):
                return lst[i] if isinstance(lst, list) and i < len(lst) else None
            row["ma2d_px_supported_before_after"] = [_at(res["before"].get("ma2d_px_supported"), j), _at(res["after"].get("ma2d_px_supported"), j)]
            row["coco_wrist_px_fit_before_after"] = [_at(res["before"].get("coco_wrist_px_fit"), j), _at(res["after"].get("coco_wrist_px_fit"), j)]
            rows.append(row)

    def agg(key, sub=None):
        xs = [x for x in rows if (sub is None or sub(x)) and x.get(f"{key}_before") is not None and x.get(f"{key}_after") is not None]
        if not xs:
            return None
        b = np.array([x[f"{key}_before"] for x in xs], float)
        a = np.array([x[f"{key}_after"] for x in xs], float)
        wts = np.array([x["frames"] for x in xs], float)
        return {"n": int(len(b)), "median_before": float(np.median(b)), "median_after": float(np.median(a)),
                "frame_weighted_mean_before": float(np.average(b, weights=wts)), "frame_weighted_mean_after": float(np.average(a, weights=wts)),
                "share_events_better_or_equal": float(np.mean(a <= b + 1e-9))}

    keys = ("hand_lm_px", "coco_wrist_px", "coco_wrist_pck20", "coco_elbow_px", "gap_cm", "pen_cm_max", "pen_cm_p90", "pen_cm_med",
            "arm_pen_cm_p90", "palm_gap_cm", "wrist_jitter_mm", "palm_jitter_mm", "tip_jitter_mm")
    used = lambda x: (x.get("coco_target_share") or 0) > 0.05          # noqa: E731
    summ = {"all": {k: agg(k) for k in keys},
            "contact_only (no COCO target: COCO is an independent judge)": {k: agg(k, lambda x: not used(x)) for k in keys},
            "coco_driven (COCO was a target: see the held-out folds)": {k: agg(k, used) for k in keys}}
    summ["events"] = len(rows)
    summ["refine"] = rep["summary"]
    summ["pass2"] = rep2.get("summary")
    summ["pass3"] = rep3.get("summary")
    summ["pass4"] = rep4.get("summary")
    left = {r["event"] for r in rep["results"] if not r.get("adopted")} - {r["event"] for r in rep2["results"] + rep3["results"] if r.get("adopted")}
    seen = set()
    summ["not_adopted"] = []
    for r in rep3["results"] + rep2["results"] + rep["results"]:
        if r["event"] in left and not r.get("adopted") and r["event"] not in seen:
            seen.add(r["event"])
            summ["not_adopted"].append({"event": r["event"], "hold": r.get("hold"), "guard": r.get("guard")})
    summ["by_pass"] = {"pass1 (arm + hand)": sum(x["pass"] == 1 for x in rows), "pass2 (hand only)": sum(x["pass"] == 2 for x in rows),
                       "pass3 (looser fingertip guard)": sum(x["pass"] == 3 for x in rows),
                       "pass4 (post-check: solved again on the final fit)": sum(x["pass"] == 4 for x in rows)}
    for nm, f in (("sunk_gt_1cm", lambda x, n: x[f"pen_cm_p90_{n}"] > 1), ("floating_gt_1cm", lambda x, n: x[f"gap_cm_{n}"] > 1)):
        summ[nm] = {n: int(sum(f(x, n) for x in rows)) for n in ("before", "after")}
    by = {}
    for x in rows:
        by.setdefault(x["hold"], []).append(x)
    summ["by_hold"] = {h: {"n": len(v), **{f"{k}_{n}": float(np.nanmedian([y[f"{k}_{n}"] if y[f"{k}_{n}"] is not None else np.nan for y in v]))
                                           for k in ("gap_cm", "pen_cm_p90", "hand_lm_px", "coco_wrist_px") for n in ("before", "after")}}
                       for h, v in by.items()}
    # the 2026-10-21 safety checks
    def f_(x):
        return x if x is not None else float("nan")
    summ["safety"] = {
        "mamma_hand_px_worse_gt_3": [{"event": x["event"], "before": x["hand_lm_px_before"], "after": x["hand_lm_px_after"], "curated_keep": x["curated_keep"] or x["override"]}
                                     for x in rows if f_(x["hand_lm_px_after"]) > f_(x["hand_lm_px_before"]) + 3.0],
        "coco_worse_gt_4": [{"event": x["event"], "wrist": [x["coco_wrist_px_before"], x["coco_wrist_px_after"]],
                             "elbow": [x["coco_elbow_px_before"], x["coco_elbow_px_after"]], "curated_keep": x["curated_keep"] or x["override"]}
                            for x in rows if f_(x["coco_wrist_px_after"]) > f_(x["coco_wrist_px_before"]) + 4.0
                            or f_(x["coco_elbow_px_after"]) > f_(x["coco_elbow_px_before"]) + 4.0],
        "new_penetration": [{"event": x["event"], "p90": [x["pen_cm_p90_before"], x["pen_cm_p90_after"]], "max": [x["pen_cm_max_before"], x["pen_cm_max_after"]]}
                            for x in rows if x["pen_cm_p90_after"] > max(x["pen_cm_p90_before"], 0.2) + 0.05
                            or x["pen_cm_max_after"] > max(x["pen_cm_max_before"], 0.4) + 0.1],
        "flat_holds_palm_gap_cm": agg("palm_gap_cm", lambda x: x["flat"]),
        "flat_holds_palm_gap_gt_1cm": {n: int(sum(x[f"palm_gap_cm_{n}"] > 1.0 for x in rows if x["flat"])) for n in ("before", "after")},
        "speed_audit": speed_audit(fa, fb),
        "curation": {"veto": cur["veto"], "keep": cur["keep"], "overrides": cur["overrides"]}}
    folds = {}
    for ft in ("foldA", "foldB"):
        p = OUT / f"refine_{ft}.json"
        # the held-out folds count only when they were run on the current production result (folds re-run after the last refine)
        if p.exists() and p.stat().st_mtime > (OUT / f"refine_{tag}.json").stat().st_mtime:
            fr = json.loads(p.read_text(encoding="utf-8"))
            b, a, be, ae = [], [], [], []
            for r in fr["results"]:
                if not r.get("before"):
                    continue
                for i in range(len(r["before"].get("coco_wrist_px_test", []))):
                    x0, x1 = r["before"]["coco_wrist_px_test"][i], r["after"]["coco_wrist_px_test"][i]
                    if x0 is not None and x1 is not None and np.isfinite(x0) and np.isfinite(x1):
                        b.append(x0)
                        a.append(x1)
                    y0, y1 = r["before"]["coco_elbow_px_test"][i], r["after"]["coco_elbow_px_test"][i]
                    if y0 is not None and y1 is not None and np.isfinite(y0) and np.isfinite(y1):
                        be.append(y0)
                        ae.append(y1)
            folds[ft] = {"fit_cams": fr["summary"]["cams"], "events": len(b),
                         "heldout_coco_wrist_px_median_before": float(np.median(b)) if b else None,
                         "heldout_coco_wrist_px_median_after": float(np.median(a)) if a else None,
                         "heldout_share_better_or_equal": float(np.mean(np.array(a) <= np.array(b) + 1e-9)) if b else None,
                         "heldout_coco_elbow_px_median_before": float(np.median(be)) if be else None,
                         "heldout_coco_elbow_px_median_after": float(np.median(ae)) if ae else None}
    summ["heldout_folds"] = folds
    out = {"what": "hand_contact validation: before = smplx_rf2_noslip, after = v3/smplx_contact, adopted events, core frames", "summary": summ,
           "events": rows}
    write_json(OUT / "validate.json", out)
    return out


# ------------------------------------------------------------------------------------------------ where the wrist really is: multi-view COCO wrists

def _dlt(Ps: np.ndarray, x: np.ndarray) -> np.ndarray:
    A = []
    for P, (u, v) in zip(Ps, x):
        A.append(u * P[2] - P[0])
        A.append(v * P[2] - P[1])
    _, _, Vt = np.linalg.svd(np.asarray(A))
    X = Vt[-1]
    return X[:3] / X[3] if abs(X[3]) > 1e-12 else np.full(3, np.nan)


def _reproj(P: np.ndarray, X: np.ndarray) -> np.ndarray:
    h = P @ np.append(X, 1.0)
    return h[:2] / h[2] if h[2] > 1e-6 else np.array([np.inf, np.inf])


@dataclass
class TriCfg:
    inlier_px: float = 14.0        # a view agrees when the triangulated wrist reprojects within this (at 1080p; scaled by focal / 1000)
    min_views: int = 2
    min_angle_deg: float = 8.0     # the two rays of the seed pair must open at least this much
    max_reach_m: float = 0.80      # wrist within this of the actor's shoulder
    max_from_fit_m: float = 0.30   # and within this of the fit's wrist (a detection of the wrong arm / person is not a target)


def triangulate_joint(fit_r: dict, role: str, side: str, ks: np.ndarray, judge: "CocoJudge", obs: Obs, cams: list[str] | None = None,
                      cfg: TriCfg | None = None, which: str = "wrist") -> dict:
    """Per grid frame of ks: the actor's wrist (or elbow) triangulated from the owner's COCO detections in `cams` (RANSAC over view pairs)
    -> X (n,3) NaN where < 2 agreeing views, n_views (n,), err_px (n,) median reprojection of the inliers."""
    from .hand_kin import WRIST, SHOULDER, ELBOW
    cfg = cfg or TriCfg()
    reach = cfg.max_reach_m if which == "wrist" else 0.45
    cams = list(cams or obs.cams)
    n = len(ks)
    X = np.full((n, 3), np.nan)
    nv = np.zeros(n, np.int8)
    er = np.full(n, np.nan)
    Jw = fit_r["joints"][ks, WRIST[side] if which == "wrist" else ELBOW[side]]
    Js = fit_r["joints"][ks, SHOULDER[side]]
    for i, k in enumerate(ks):
        k = int(k)
        Ps, xs, fs, Cs = [], [], [], []
        for c in cams:
            z = obs.z[c]
            if not z["ok"][k]:
                continue
            u, ok = judge.joint(c, role, side, k, which)
            if not ok:
                continue
            P = z["K"][k] @ np.hstack([z["R"][k], z["t"][k][:, None]])
            Ps.append(P)
            xs.append(u)
            fs.append(z["K"][k][0, 0] / 1000.0)
            Cs.append(-z["R"][k].T @ z["t"][k])
        if len(Ps) < cfg.min_views:
            continue
        best = None
        for a in range(len(Ps)):
            for b in range(a + 1, len(Ps)):
                Xa = _dlt(np.array([Ps[a], Ps[b]]), np.array([xs[a], xs[b]]))
                if not np.isfinite(Xa).all():
                    continue
                ra, rb = Xa - Cs[a], Xa - Cs[b]
                ang = np.degrees(np.arccos(np.clip(ra @ rb / (np.linalg.norm(ra) * np.linalg.norm(rb)), -1, 1)))
                if ang < cfg.min_angle_deg:
                    continue
                e = np.array([np.linalg.norm(_reproj(P, Xa) - x) / f for P, x, f in zip(Ps, xs, fs)])
                inl = np.flatnonzero(e < cfg.inlier_px)
                if len(inl) < cfg.min_views:
                    continue
                sc = (len(inl), -float(np.median(e[inl])))
                if best is None or sc > best[0]:
                    best = (sc, inl)
        if best is None:
            continue
        inl = best[1]
        Xr = _dlt(np.array(Ps)[inl], np.array(xs)[inl])
        if not np.isfinite(Xr).all():
            continue
        if np.linalg.norm(Xr - Js[i]) > reach or np.linalg.norm(Xr - Jw[i]) > cfg.max_from_fit_m:
            continue
        e = [np.linalg.norm(_reproj(Ps[j], Xr) - xs[j]) for j in inl]
        X[i], nv[i], er[i] = Xr, len(inl), float(np.median(e))
    return {"X": X, "n_views": nv, "err_px": er}


def triangulate_wrists(fit_r, role, side, ks, judge, obs, cams=None, cfg=None) -> dict:
    return triangulate_joint(fit_r, role, side, ks, judge, obs, cams, cfg, "wrist")


def smooth_track(X: np.ndarray, max_gap: int = 8, sigma: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """robust temporal clean-up of a (n,3) track with NaN holes: 5-frame median, gaps <= max_gap interpolated, Gaussian smoothing;
    -> (track, valid mask)"""
    from scipy.ndimage import gaussian_filter1d
    n = len(X)
    ok = np.isfinite(X).all(1)
    if ok.sum() < 3:
        return X, ok
    Y = X.copy()
    idx = np.flatnonzero(ok)
    for d in range(3):
        Y[:, d] = np.interp(np.arange(n), idx, X[idx, d])
    med = np.stack([_medfilt(Y[:, d], 2) for d in range(3)], 1)
    # drop outliers > 6 cm from the running median
    bad = ok & (np.linalg.norm(X - med, axis=1) > 0.06)
    ok2 = ok & ~bad
    if ok2.sum() < 3:
        return X, ok
    idx = np.flatnonzero(ok2)
    for d in range(3):
        Y[:, d] = np.interp(np.arange(n), idx, X[idx, d])
    valid = np.zeros(n, bool)
    for a, b in zip(idx[:-1], idx[1:]):
        if b - a <= max_gap + 1:
            valid[a:b + 1] = True
    valid[idx] = True
    Y = gaussian_filter1d(Y, sigma, axis=0, mode="nearest")
    return Y, valid


# ------------------------------------------------------------------------------------------------ the Unity stage's motion files

UNITY_CAPTURE = "06_LarissaKaduRecap"
SIDE_CAPTURE = "06_LarissaKaduRecap_v3hands"     # a side export (never installed): only its motion files are used
MOTION_FILES = ("smplx_lead.bin", "smplx_follow.bin", "figure1.json", "figure2.json")


def export_motion() -> dict:
    """Export the refined fit with export_unity into the SIDE capture work/<take>/export/06_LarissaKaduRecap_v3hands/ (install=False: the
    Unity project is not touched), copy its motion files to v3/unity_motion/, and compare them with the installed capture (read-only): same
    frame count / header, identical outside the refined windows, changed only on arm / hand joints inside."""
    import shutil
    from . import export_unity as eu
    rep = eu.export(TAKE, SIDE_CAPTURE, install=False, room=False, textures=False, faces=False, shape_edits=True, allow_no_audio=True,
                    smplx_dir="v3/smplx_contact")
    side = WORK / "export" / SIDE_CAPTURE
    dst = V3 / "unity_motion"
    dst.mkdir(parents=True, exist_ok=True)
    for f in MOTION_FILES:
        shutil.copy2(side / f, dst / f)
    import hashlib
    out = {"side_export": str(side), "files": [str(dst / f) for f in MOTION_FILES], "fk_max_err_mm": {r: d["fk_max_err_mm"] for r, d in rep["dancers"].items()},
           "warnings": rep["warnings"], "compare_installed": {},
           # install_motion.sh checks these before it copies anything (a half-written or stale motion file is never installed)
           "sha256": {f: hashlib.sha256((dst / f).read_bytes()).hexdigest() for f in MOTION_FILES},
           "fit": str(FIT_OUT), "fit_info": json.loads((FIT_OUT / "hand_contact.json").read_text(encoding="utf-8")).get("info"),
           "exported": time.strftime("%Y-%m-%d %H:%M:%S")}
    vp = OUT / "validate.json"
    if vp.exists():
        sf = json.loads(vp.read_text(encoding="utf-8"))["summary"].get("safety", {})
        out["validate_safety"] = {"speed_incidents": (sf.get("speed_audit") or {}).get("n_incidents"),
                                  "new_penetration": len(sf.get("new_penetration") or []),
                                  "mamma_hand_px_worse_gt_3": [x["event"] for x in sf.get("mamma_hand_px_worse_gt_3") or []],
                                  "coco_worse_gt_4": [x["event"] for x in sf.get("coco_worse_gt_4") or []]}
    inst = eu.UNITY_STREAMING / UNITY_CAPTURE
    if not (inst / "smplx_lead.bin").exists():                       # the installed capture can be away (a Quest build moves it): the work master
        inst = WORK / "export" / UNITY_CAPTURE
    out["compared_with"] = str(inst)
    fit0, fit1 = load_fit(SRC_FIT), load_fit(FIT_OUT)
    for r, f in (("lead", "smplx_lead.bin"), ("follow", "smplx_follow.bin")):
        if not (inst / f).exists():
            out["compare_installed"][r] = "installed capture has no motion file"
            continue
        h0, m0 = eu.read_motion(inst / f)
        h1, m1 = eu.read_motion(dst / f)
        if m0.shape != m1.shape:
            out["compare_installed"][r] = f"shape differs {m0.shape} vs {m1.shape}"
            continue
        changed = np.flatnonzero(np.abs(full_pose(fit1[r]) - full_pose(fit0[r])).max((1, 2)) > 0)
        same = np.setdiff1d(np.arange(len(m0)), changed)
        pos0 = m0[:, 3 + 55 * 4:].reshape(len(m0), 55, 3)
        pos1 = m1[:, 3 + 55 * 4:].reshape(len(m1), 55, 3)
        d = np.linalg.norm(pos1 - pos0, axis=-1)
        from .hand_kin import SHOULDER, ELBOW, WRIST, FINGER_JOINTS
        arm = sorted({SHOULDER[s] for s in SIDES} | {ELBOW[s] for s in SIDES} | {WRIST[s] for s in SIDES} | set(FINGER_JOINTS["left"]) |
                     set(FINGER_JOINTS["right"]))
        other = [j for j in range(55) if j not in arm]
        out["compare_installed"][r] = {
            "frames": int(len(m0)), "frames_changed_by_refinement": int(len(changed)),
            "max_abs_diff_unchanged_frames": float(np.abs(m1[same] - m0[same]).max()) if len(same) else None,
            "max_joint_move_cm_non_arm_joints": float(d[:, other].max() * 100),
            "max_joint_move_cm_arm_hand_joints": float(d[:, arm].max() * 100),
            "median_wrist_move_cm_changed_frames": float(np.median(d[changed][:, [WRIST["left"], WRIST["right"]]]) * 100) if len(changed) else None}
    write_json(dst / "unity_motion.json", out)
    return out


# ------------------------------------------------------------------------------------------------ CLI

def pick_sheet_events(events: list, n: int = 20, seed: int = 0) -> list:
    """n events for a contact sheet: spread over hold types (largest first) and over the class, holds before touches"""
    holds = [e for e in events if e["kind"] == "hold" and e["hold"] not in ("none",)]
    pool = holds if len(holds) >= n else events
    by = {}
    for e in pool:
        by.setdefault(e["hold"], []).append(e)
    for v in by.values():
        v.sort(key=lambda e: e["t0"])
    out = []
    while len(out) < n and any(by.values()):
        for h in sorted(by, key=lambda h: -len(by[h])):
            if by[h] and len(out) < n:
                v = by[h]
                out.append(v.pop(len(v) // 2 if len(out) % 2 else 0))
    return sorted(out, key=lambda e: e["t0"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.hand_contact", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("obs")
    p.add_argument("--force", action="store_true")
    p.add_argument("--check", action="store_true")
    sub.add_parser("detect")
    p = sub.add_parser("sheet")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--ids", default=None)
    p = sub.add_parser("refine")
    p.add_argument("--ids", default=None)
    p.add_argument("--parts", type=int, default=3)
    p = sub.add_parser("refine-part")
    p.add_argument("tag")
    p.add_argument("i", type=int)
    p.add_argument("n", type=int)
    sub.add_parser("validate")
    sub.add_parser("folds")
    sub.add_parser("pass2")
    sub.add_parser("pass3")
    sub.add_parser("pass4")
    p = sub.add_parser("revert")
    p.add_argument("ids")
    p.add_argument("--why", required=True)
    p = sub.add_parser("merge")
    p.add_argument("tag")
    p.add_argument("n", type=int)
    sub.add_parser("unity-motion")
    p = sub.add_parser("before-after")
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--ids", default=None)
    p.add_argument("--extra", default="118,47", help="events always added to the picked ones")
    sub.add_parser("speed-audit")
    a = ap.parse_args(argv)
    if a.cmd == "obs":
        print(json.dumps(py_json(build_obs(a.force)), indent=1)[:3000])
        if a.check:
            print(json.dumps(py_json(check_obs()), indent=1))
    elif a.cmd == "detect":
        run_detect()
    elif a.cmd == "sheet":
        evs = load_events()
        sel = [e for e in evs if e["id"] in [int(x) for x in a.ids.split(",")]] if a.ids else pick_sheet_events(evs, a.n)
        fit = load_fit()
        for i in range(0, len(sel), 10):
            pth = contact_sheet(sel[i:i + 10], [("fit smplx_rf2_noslip", fit)], REVIEW / f"events_sheet_{i // 10 + 1}.jpg",
                                f"recap hand contacts {i // 10 + 1}: phone crop (undistorted) | fit mesh (lead blue, follow orange, acting hand yellow)")
            log(f"sheet -> {pth}")
    elif a.cmd == "refine":
        only = [int(x) for x in a.ids.split(",")] if a.ids else None
        if a.parts > 1:
            run_parallel("all", a.parts, only=only)
        else:
            run_refine(only=only)
    elif a.cmd == "refine-part":
        args = json.loads((OUT / f"_parts_{a.tag}.json").read_text(encoding="utf-8"))
        run_refine(only=args.get("only"), cams=args.get("cams"), out_dir=None, tag=f"{a.tag}_p{a.i}", part=(a.i, a.n),
                   cfg=RefineCfg(**args.get("cfg", {})), start_from=Path(args["start_from"]) if args.get("start_from") else None)
    elif a.cmd == "folds":
        run_folds()
    elif a.cmd == "pass2":
        run_pass2()
    elif a.cmd == "pass3":
        run_pass3()
    elif a.cmd == "pass4":
        run_pass4()
    elif a.cmd == "revert":
        print(json.dumps(py_json(revert_events([int(x) for x in a.ids.split(",")], a.why)), indent=1))
    elif a.cmd == "merge":
        merge_parts(a.tag, a.n, write=(a.tag == "all"))
    elif a.cmd == "unity-motion":
        print(json.dumps(py_json(export_motion()), indent=1))
    elif a.cmd == "validate":
        v = validate()
        print(json.dumps(py_json({k: x for k, x in v["summary"].items() if k != "safety"}), indent=1))
        sf = v["summary"]["safety"]
        print(json.dumps(py_json({k: (x if k != "speed_audit" else {"n_incidents": x["n_incidents"]}) for k, x in sf.items() if k != "curation"}), indent=1))
    elif a.cmd == "speed-audit":
        au = speed_audit(load_fit(SRC_FIT), load_fit(FIT_OUT))
        print(json.dumps(py_json(au), indent=1)[:6000])
    elif a.cmd == "before-after":
        evs = {e["id"]: e for e in curated_events(load_events())}
        adopted = {r["event"] for r in adopted_results()}                    # all passes, minus curated / post-check vetoes
        extra = [int(x) for x in a.extra.split(",")] if a.extra else []
        ad = [evs[i] for i in sorted(adopted) if i in evs and i not in extra]
        sel = [evs[int(x)] for x in a.ids.split(",")] if a.ids else pick_sheet_events(ad, a.n)
        sel += [evs[i] for i in extra if i in evs and evs[i] not in sel]
        vp = OUT / "validate.json"
        vrows = {x["event"]: x for x in json.loads(vp.read_text(encoding="utf-8"))["events"]} if vp.exists() else {}

        def f1(x):
            return "-" if x is None else f"{x:.1f}"
        for e in sel:
            v = vrows.get(e["id"])
            if e["id"] not in adopted:
                e["_sheet_lines"] = ["NOT refined: after = input"]
            if v:
                e["_sheet_lines"] = ["before -> after:", f"gap {f1(v['gap_cm_before'])} -> {f1(v['gap_cm_after'])} cm",
                                     f"sunk p90 {f1(v['pen_cm_p90_before'])} -> {f1(v['pen_cm_p90_after'])} cm",
                                     f"hand lm {f1(v['hand_lm_px_before'])} -> {f1(v['hand_lm_px_after'])} px",
                                     f"COCO wrist {f1(v['coco_wrist_px_before'])} -> {f1(v['coco_wrist_px_after'])} px"]
        fits = [("before", load_fit(SRC_FIT)), ("after", load_fit(FIT_OUT))]
        for i in range(0, len(sel), 6):
            pth = contact_sheet(sel[i:i + 6], fits, REVIEW / f"before_after_{i // 6 + 1}.jpg",
                                f"recap hand contacts before / after {i // 6 + 1}: phone crop | before (smplx_rf2_noslip) | after (v3/smplx_contact)")
            log(f"sheet -> {pth}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
