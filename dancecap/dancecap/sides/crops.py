"""The crop check of the "sides" step (DESIGN 2.3 l_crop, 4.1 row crops.py, 6.1 Phase A step 3).

Raw phone crops (no overlay) at chosen dance times, Jev questions on them, a calibration of every question on frames with known answers,
the agent's own look next to Jev's, and the <= 6-crop owner sheet. Everything runs on the PC: the phone videos and the Jev key never leave
it (the key is read by dancecap.judgments.client and never printed). Nothing here imports the unfinished sibling modules at import time
(adapters / config / geom are imported inside the functions that need them).

    python -m dancecap.sides.crops make      --take <t> [--plan plan.json]        # crops/*.jpg + crops/index.json
    python -m dancecap.sides.crops ask       --take <t> --allow-upload [--set calib|target|all]   # Jev -> crops/jev_rows.json (cached)
    python -m dancecap.sides.crops looksheet --take <t>                            # 2x2 sheets for the agent's look -> crops/look/
    python -m dancecap.sides.crops calibrate --take <t>                            # accuracy per question -> crops/calibration.json
    python -m dancecap.sides.crops write     --take <t>                            # -> sides/crops.json (the decide contract)
    python -m dancecap.sides.crops sheet     --take <t>                            # -> sides/crops/owner_sheet.jpg (<= 6 crops)

Data (all under work/<take>/sides/, git-ignored):
    crops/plan.json         the take's crop times (data, not code: no take constant lives in this package)
    crops/index.json        one row per crop: id, t, k, cam, kind, box, file, set (target / calib), questions asked for
    crops/<id>.jpg          the raw crop (undistorted pinhole frame, MAMMA's camera model)
    crops/looks.json        the agent's look per crop id (R4 class, R9 yes / no, R3 yes / no, note); written by the agent
    crops/jev_rows.json     Jev's answers per crop id (+ model, tokens); crops/jev_cache.jsonl caches by image sha256
    crops/calibration.json  per question: n, accuracy, AUC, confusion, the score used by decide
    crops.json              {"items": [...]} in the contract of sides.decide.load_crops, plus per time / camera: question, P(yes),
                            the agent's look, the calibration score

crops.json item (sides.decide.load_crops reads t, cam, relation, p, score, source; the rest is for people):
    {"t": dance s, "cam": "00", "relation": "R4" | "R9_RL" | "R3_follow", "p": {"front", "flank", "back"} | {"+": p},
     "score": calibration score (agent looks: 1.0), "source": "jev" | "agent", "id": crop id, "question": text,
     "p_yes": Noul P(yes) | "probabilities": Choice distribution, "agent_look": the look's answer, "calibration_score": score}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------------------------------------ configuration (no take constants)

R4_OPTIONS = ("back", "side", "front", "elsewhere", "not_visible")
R4_TO_CLASS = {"back": "back", "side": "flank", "front": "front"}      # Jev / look option -> sides.hmm.R4_CLASSES
CLASSES = ("front", "flank", "back")                                    # = sides.hmm.R4_CLASSES (kept literal: no import at load time)
QUESTIONS = ("R4", "R9", "R3")
LOOK_R4 = R4_OPTIONS + ("unclear",)
LOOK_YN = ("yes", "no", "unclear")


@dataclass
class CropCfg:
    hz: float = 30.0
    tile: int = 640                 # crop side in px (Jev accepts 280-1568; 640 keeps a hand > 25 px)
    margin: float = 0.22            # around the union box of the projected joints
    min_side_px: float = 260.0      # native px
    max_side_frac: float = 1.0      # of the image's shorter side
    min_in_frame: float = 0.6       # share of the crop joints inside the image
    min_depth_m: float = 0.3
    max_dt_s: float = 1.0 / 50      # decoded frame within this of the grid time
    jpeg_quality: int = 92
    # joints (SMPL-X) whose projections define the box: both dancers' upper bodies (head, neck, spine3, shoulders, elbows, wrists, hips)
    couple_joints: tuple = (15, 12, 9, 16, 17, 18, 19, 20, 21, 1, 2)
    face_joints: tuple = (15, 12, 9, 16, 17, 1, 2)          # R3: her upper body
    face_margin: float = 0.45
    # Jev
    model: str = "jev-latest"
    workers: int = 4
    yes: float = 0.5                # Noul answer threshold for accuracy
    crop_min_score: float = 0.8     # = SidesCfg.crop_min_score (decide ignores items below it)
    # R4: Choice probabilities -> class distribution; the not-visible / elsewhere mass flattens it toward uniform
    r4_flatten: bool = True
    # the agent's look -> evidence (DESIGN 2.3: "the agent's own look at a crop counts the same (+-3)")
    look_p_sure: float = 0.9
    look_p_lean: float = 0.7
    # R9 sign map: the crossing must be seen from above or below (|n . up| of the crossing pair's normal) to define "on top"
    r9_min_vertical: float = 0.5
    r9_min_abs_g: float = 0.02
    r9_max_dist_m: float = 0.15     # the crossing bones closer than this (the arms meet)

    def to_json(self) -> dict:
        return json.loads(json.dumps(asdict(self), default=list))


def log(*a) -> None:
    print("[sides.crops]", *a, flush=True)


def py_json(x):
    if isinstance(x, dict):
        return {str(k): py_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [py_json(v) for v in x]
    if isinstance(x, np.ndarray):
        return py_json(x.tolist())
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return None if v != v else round(v, 5)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def write_json(p: Path, obj) -> Path:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(py_json(obj), indent=1), encoding="utf-8")
    return p


def read_json(p: Path, default=None):
    p = Path(p)
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding="utf-8"))


def work_dir(take: str) -> Path:
    from ..take import PACKAGE_ROOT
    return PACKAGE_ROOT / "work" / take


def crops_dir(take: str, work: Path | None = None) -> Path:
    return (Path(work) if work else work_dir(take)) / "sides" / "crops"


# ------------------------------------------------------------------------------------------------ crop times (pure)

def _frange(a: float, b: float, step: float) -> list:
    n = int(math.floor((b - a) / step + 1e-6)) + 1
    return [round(a + i * step, 4) for i in range(max(n, 0))]


def expand_plan(plan: dict) -> list:
    """plan {"groups": [{"name", "times": [s, ...] | "range": [a, b, step], "cams": [..] | "auto", "questions": ["R4", ...],
    "set": "target" | "calib", "kind": "couple" | "face", "why": text}]} -> one entry per (t, group), sorted by t"""
    out = []
    for g in plan.get("groups", []):
        ts = list(g.get("times", []))
        if "range" in g:
            a, b, st = g["range"]
            ts += _frange(float(a), float(b), float(st))
        for t in ts:
            out.append({"t": float(t), "cams": g.get("cams", "auto"), "questions": list(g.get("questions", ["R4"])),
                        "group": g.get("name", ""), "set": g.get("set", "target"), "kind": g.get("kind", "couple"),
                        "why": g.get("why", "")})
    out.sort(key=lambda e: (e["t"], e["group"]))
    return out


def _runs(mask: np.ndarray) -> list:
    idx = np.flatnonzero(mask)
    if not len(idx):
        return []
    cut = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[idx[0], idx[cut + 1]]
    ends = np.r_[idx[cut], idx[-1]]
    return list(zip(starts.tolist(), ends.tolist()))


def crop_times(rel: dict | None = None, plan: dict | None = None, hz: float = 30.0, post_decide: float = 0.9,
               min_run_s: float = 0.5, max_auto: int = 12) -> list:
    """the crop times: the plan's explicit times, plus (from decide's relations.npz arrays) the middle of each run where a relation
    the crops can answer is defined but undecided (R4: gate and in contact band, the class posterior < post_decide; R9_RL: in the
    entangled band, the sign posterior < post_decide), longest runs first, at most max_auto; times within 0.5 s of a planned one are
    dropped. Cams of the automatic entries: "auto" (every camera that sees the couple)."""
    out = expand_plan(plan) if plan else []
    if rel is None:
        return out
    auto = []
    for rid, q in (("R4", "R4"), ("R9_RL", "R9")):
        gk = f"{rid}__gate"
        pk = f"{rid}__post"
        if gk not in rel or pk not in rel:
            continue
        gate = np.asarray(rel[gk], bool)
        band = np.asarray(rel.get(f"{rid}__band", gate), bool)
        post = np.asarray(rel[pk], float)
        pmax = post.max(-1) if post.ndim == 2 else post
        if rid == "R4" and post.ndim == 2 and post.shape[1] == 4:      # front / flank / back / undecided
            pmax = post[:, :3].max(-1)
        und = gate & band & (pmax < post_decide)
        for a, b in _runs(und):
            if (b - a + 1) / hz >= min_run_s:
                auto.append((b - a + 1, round(0.5 * (a + b) / hz, 3), q, rid))
    auto.sort(key=lambda x: -x[0])
    planned = [e["t"] for e in out]
    for n, t, q, rid in auto[:max_auto]:
        if any(abs(t - p) < 0.5 for p in planned):
            continue
        out.append({"t": t, "cams": "auto", "questions": [q], "group": f"auto_{rid}", "set": "target", "kind": "couple",
                    "why": f"{rid} defined but undecided for {n / hz:.1f} s"})
        planned.append(t)
    out.sort(key=lambda e: (e["t"], e["group"]))
    return out


def crop_id(t: float, cam: str, kind: str = "couple") -> str:
    return f"t{t:07.2f}_c{cam}" + ("" if kind == "couple" else f"_{kind}")


# ------------------------------------------------------------------------------------------------ crop geometry (pure)

def crop_box(uv: np.ndarray, size: tuple, margin: float, min_side: float, max_side_frac: float = 1.0) -> tuple:
    """a square box (cx, cy, side) around the finite pixels uv (n,2): the union box grown by `margin` on each side, at least min_side,
    at most max_side_frac x the shorter image side; the centre stays where the points are (the crop may run over the image edge, which
    crop_img fills grey)"""
    uv = np.asarray(uv, float)
    uv = uv[np.isfinite(uv).all(1)]
    if not len(uv):
        raise ValueError("no finite points for the crop box")
    lo, hi = uv.min(0), uv.max(0)
    c = 0.5 * (lo + hi)
    side = float(max(hi - lo)) * (1.0 + 2.0 * margin)
    side = max(side, float(min_side))
    side = min(side, float(max_side_frac) * float(min(size)))
    return float(c[0]), float(c[1]), side


def in_frame_share(uv: np.ndarray, depth: np.ndarray, size: tuple, min_depth: float) -> float:
    W, H = size
    uv = np.asarray(uv, float)
    ok = np.isfinite(uv).all(1) & (np.asarray(depth) > min_depth) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    return float(ok.mean()) if len(ok) else 0.0


def crop_K(K: np.ndarray, cx: float, cy: float, side: float, tile: int) -> np.ndarray:
    """intrinsics of a square crop of `side` px centred at (cx, cy) resampled to tile x tile (hand_contact.crop_K, ported)"""
    s = tile / side
    Kc = np.asarray(K, float).copy()
    Kc[0, 0] *= s
    Kc[1, 1] *= s
    Kc[0, 2] = (K[0, 2] - (cx - side / 2)) * s
    Kc[1, 2] = (K[1, 2] - (cy - side / 2)) * s
    return Kc


def crop_maps(K: np.ndarray, cx: float, cy: float, side: float, tile: int, k1: float) -> tuple[np.ndarray, np.ndarray]:
    """the remap tables of the undistorted (pinhole) square crop: crop pixel -> raw (SIMPLE_RADIAL k1) pixel (hc.crop_img's maps)"""
    Kc = crop_K(K, cx, cy, side, tile)
    u, v = np.meshgrid(np.arange(tile, dtype=np.float64), np.arange(tile, dtype=np.float64))
    x = (u - Kc[0, 2]) / Kc[0, 0]
    y = (v - Kc[1, 2]) / Kc[1, 1]
    fac = 1 + float(k1) * (x * x + y * y)
    mx = (K[0, 0] * x * fac + K[0, 2]).astype(np.float32)
    my = (K[1, 1] * y * fac + K[1, 2]).astype(np.float32)
    return mx, my


def crop_img(img: np.ndarray, cx: float, cy: float, side: float, tile: int, K: np.ndarray, k1: float) -> np.ndarray:
    import cv2
    mx, my = crop_maps(K, cx, cy, side, tile, k1)
    return cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(30, 30, 30))


def project(K, R, t, X):
    """X (P,3) world -> uv (P,2), depth (P,) for one pinhole camera (R, t world->camera)"""
    Xc = np.asarray(X, float) @ np.asarray(R, float).T + np.asarray(t, float)
    d = Xc[:, 2]
    uv = Xc @ np.asarray(K, float).T
    return uv[:, :2] / np.where(np.abs(uv[:, 2:]) > 1e-9, uv[:, 2:], 1e-9), d


# ------------------------------------------------------------------------------------------------ phone frames (ported with a take parameter)

def grab(src: Path, pts: float, w: int, h: int) -> np.ndarray | None:
    """the decoded frame of src whose PTS is pts, BGR (front_full_review.grab, ported: same autorotation / timestamps as ingest)"""
    seek = max(0.0, pts - 1.0)
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{seek:.6f}", "-copyts", "-i", str(src), "-map", "0:v:0", "-vf",
           f"select=gte(t\\,{pts - 0.0015:.6f})", "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    r = subprocess.run(cmd, capture_output=True)
    if len(r.stdout) < w * h * 3:
        return None
    return np.frombuffer(r.stdout[:w * h * 3], np.uint8).reshape(h, w, 3).copy()


class Phones:
    """Decoded phone frames at grid frames: the source video + the OWNING chunk's video offset (the clock of the fit's 2D input);
    hand_contact.Phones with the take as a parameter. Local only: the videos are never uploaded."""

    def __init__(self, take: str, work: Path | None = None, owner: np.ndarray | None = None):
        from ..take import Take
        from .adapters import chunk_work, offsets
        self.take = Take.load(take)
        self.work = Path(work) if work else work_dir(take)
        st = read_json(self.work / "stitch.json")
        self.offs = {ch["name"]: offsets(chunk_work(self.work, ch["name"])) for ch in st["chunks"]}
        self.owner = owner
        self.cam = {}

    def _c(self, cam: str) -> dict:
        if cam not in self.cam:
            from .adapters import camera_file
            z = np.load(camera_file(self.work, cam), allow_pickle=True)
            self.cam[cam] = {"t": np.asarray(z["t"], float), "pts": np.asarray(z["pts"], float), "k1": np.asarray(z["dist"], float).ravel(),
                             "size": tuple(int(x) for x in z["native_size"]), "src": self.take.source(cam)}
        return self.cam[cam]

    def frame(self, cam: str, k: int, hz: float = 30.0) -> tuple[np.ndarray | None, float, float]:
        """(BGR upright frame, k1 of that frame, |time error| s) of grid frame k"""
        from .. import mamma_dyn as md
        c = self._c(cam)
        tk = k / hz
        off = self.offs.get(str(self.owner[k]), {}).get(cam) if self.owner is not None else None
        tc = md.apply_offset(c["t"], off)
        i = int(np.argmin(np.abs(tc - tk)))
        W, H = c["size"]
        img = grab(c["src"].path, float(c["pts"][i]), W, H)
        return img, float(c["k1"][i]), float(abs(tc[i] - tk))


class Scene:
    """what the crops need of a take: the base fit's joints (both dancers), the grid cameras, the phones (lazy)"""

    def __init__(self, take: str, work: Path | None = None, base_fit: str = "smplx_rf2_noslip", cams: list | None = None):
        from .adapters import MultiView
        self.take = take
        self.work = Path(work) if work else work_dir(take)
        self.ad = MultiView(take, None, self.work)
        if base_fit != self.ad.cfg.base_fit:
            from dataclasses import replace
            self.ad.cfg = replace(self.ad.cfg, base_fit=base_fit)
        self.cams = list(cams or self.ad.cams)
        b = self.ad.base()
        self.J = {r: np.asarray(b[r]["joints"], float) for r in ("lead", "follow")}
        self.T = len(self.J["lead"])
        self._phones = None

    @property
    def phones(self) -> Phones:
        if self._phones is None:
            self._phones = Phones(self.take, self.work, self.ad.owner)
        return self._phones

    def camera(self, cam: str, k: int) -> dict:
        o = self.ad.obs()[cam]
        return {"K": o["K"][k], "R": o["R"][k], "t": o["t"][k], "ok": bool(o["ok"][k]), "size": tuple(int(x) for x in o["size"])}

    def crop_points(self, k: int, kind: str, cfg: CropCfg) -> np.ndarray:
        if kind == "face":
            return self.J["follow"][k, list(cfg.face_joints)]
        js = list(cfg.couple_joints)
        return np.concatenate([self.J["lead"][k, js], self.J["follow"][k, js]], 0)

    def sees(self, cam: str, k: int, kind: str, cfg: CropCfg) -> tuple[bool, float]:
        c = self.camera(cam, k)
        if not c["ok"]:
            return False, 0.0
        uv, d = project(c["K"], c["R"], c["t"], self.crop_points(k, kind, cfg))
        share = in_frame_share(uv, d, c["size"], cfg.min_depth_m)
        return share >= cfg.min_in_frame, share


def crop(scene: Scene, cam: str, t: float, kind: str = "couple", cfg: CropCfg | None = None) -> tuple[np.ndarray | None, dict]:
    """the raw (no overlay) undistorted square crop of phone `cam` at dance time t around the couple's upper bodies (kind "couple") or her
    upper body ("face") -> (BGR tile or None, meta: k, box, in-frame share, |dt|, why when None)"""
    cfg = cfg or CropCfg()
    k = int(round(t * cfg.hz))
    meta = {"t": float(t), "k": k, "cam": cam, "kind": kind}
    if not (0 <= k < scene.T):
        return None, {**meta, "why": "outside the fit"}
    c = scene.camera(cam, k)
    if not c["ok"]:
        return None, {**meta, "why": "camera not ok"}
    uv, d = project(c["K"], c["R"], c["t"], scene.crop_points(k, kind, cfg))
    share = in_frame_share(uv, d, c["size"], cfg.min_depth_m)
    meta["in_frame"] = share
    if share < cfg.min_in_frame:
        return None, {**meta, "why": f"couple in frame {share:.2f} < {cfg.min_in_frame}"}
    W, H = c["size"]
    vis = np.isfinite(uv).all(1) & (d > cfg.min_depth_m)
    uvc = np.clip(uv[vis], [-0.1 * W, -0.1 * H], [1.1 * W, 1.1 * H])
    cx, cy, side = crop_box(uvc, (W, H), cfg.face_margin if kind == "face" else cfg.margin, cfg.min_side_px, cfg.max_side_frac)
    img, k1, dt = scene.phones.frame(cam, k, cfg.hz)
    meta.update({"box": [cx, cy, side], "dt_s": dt, "k1": k1})
    if img is None:
        return None, {**meta, "why": "no decoded frame"}
    if dt > cfg.max_dt_s:
        return None, {**meta, "why": f"frame time error {dt * 1000:.0f} ms"}
    return crop_img(img, cx, cy, side, cfg.tile, c["K"], k1), meta


# ------------------------------------------------------------------------------------------------ the R9 sign map (pure)

def r9_sign_map(A: np.ndarray, B: np.ndarray, min_vertical: float = 0.5, min_abs_g: float = 0.02, max_dist_m: float = 0.15) -> dict:
    """his right arm chain A (3,3) (shoulder, elbow, wrist) and her left arm chain B (3,3) at one frame -> the sign of the R9_RL margin
    (the Gauss linking integral G of A with B, sides.geom.gli_chain) that "her left arm lies on top of his right arm" means.

    The crossing pair = the bone pair with the largest |GLI|; its sign = sign((c_A - c_B) . unit(u x v)) at interior closest points
    (sides.geom.signed_sep, DESIGN F2). "Her arm on top" puts c_B above c_A, so (c_A - c_B) ~ -h up and the + side is
    plus_if_her_on_top = -sign(up . unit(u x v)). Undefined (0) when the crossing bones are more than max_dist_m apart (the arms do not
    meet), the chains are not linked (|G| < min_abs_g) or the crossing is not seen from above (|up . n| < min_vertical: "on top" says
    nothing about the linking then).
    -> {"plus_if_her_on_top": +1 | -1 | 0, "G": G, "pair": (i, j), "n_up": up . n, "her_above": bool, "why": text}"""
    from . import geom
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    G, M = geom.gli_chain(A[None], B[None], pairs=True)
    G, M = float(G[0]), np.asarray(M[0], float)
    i, j = np.unravel_index(int(np.argmax(np.abs(M))), M.shape)
    p1, q1, p2, q2 = A[i], A[i + 1], B[j], B[j + 1]
    n = np.cross(q1 - p1, q2 - p2)
    nn = float(np.linalg.norm(n))
    out = {"G": G, "pair": (int(i), int(j)), "n_up": 0.0, "her_above": None, "plus_if_her_on_top": 0, "why": ""}
    if nn < 1e-9:
        out["why"] = "parallel bones"
        return out
    n = n / nn
    ds, s, t, dist, interior = geom.signed_sep(p1[None], q1[None], p2[None], q2[None])
    cA = p1 + (q1 - p1) * float(s[0])
    cB = p2 + (q2 - p2) * float(t[0])
    out["n_up"] = float(n[1])
    out["her_above"] = bool(cB[1] > cA[1])
    out["dist_m"] = float(dist[0])
    out["interior"] = bool(interior[0])
    if out["dist_m"] > max_dist_m:
        out["why"] = f"the arms do not meet ({out['dist_m'] * 100:.0f} cm > {max_dist_m * 100:.0f} cm)"
        return out
    if abs(G) < min_abs_g:
        out["why"] = f"not linked (|G| {abs(G):.3f} < {min_abs_g})"
        return out
    if abs(n[1]) < min_vertical:
        out["why"] = f"crossing not seen from above (|n.up| {abs(n[1]):.2f} < {min_vertical})"
        return out
    out["plus_if_her_on_top"] = int(-np.sign(n[1]))
    out["why"] = "ok"
    return out


# ------------------------------------------------------------------------------------------------ Jev questions and requests

ABOUT = ("`photo` is a crop of one frame of a phone video of a man and a woman partner dancing (Brazilian zouk) in a dance studio; "
         "the man leads, the woman follows, and they often hold each other closely. Left and right always mean the dancer's own left "
         "and right (when a dancer faces the camera, their right side is on the left of the picture).")


def jev_questions(which: tuple = QUESTIONS) -> dict:
    """{question id: Noul / Choice}: R4 his right hand on her torso (Choice of 5), R9 her left arm on top of his right arm (Noul), R3 her
    face visible (Noul)"""
    from typesafe_sdk import Choice, Noul
    q = {}
    if "R4" in which:
        q["R4"] = Choice(
            instructions="Where is the man's right hand in `photo`?",
            criteria={
                "back": "on the woman's back: behind her body (shoulder blade, upper back or lower back)",
                "side": "at the woman's side, under her arm: on her ribs or waist at the side of her body, between her front and her back",
                "front": "on the front of the woman's body: her chest, stomach or the front of her shoulder",
                "elsewhere": "visible but not on the woman's torso: e.g. holding her hand, on her arm, in the air, or on his own body",
                "not_visible": "the man's right hand cannot be seen in `photo`: hidden behind a body, or outside the picture"})
    if "R9" in which:
        q["R9"] = Noul(
            instructions="Does the woman's left arm lie on top of the man's right arm, where their arms meet?",
            criteria={"true": "Her left arm rests over his right arm: her left forearm, elbow or hand is above his right arm and lies on it "
                              "(for example her left hand on his right shoulder with her forearm over his upper arm).",
                      "false": "Her left arm is under his right arm, or their arms do not meet or cross."})
    if "R3" in which:
        q["R3"] = Noul(
            instructions="Can the woman's face be seen in `photo`?",
            criteria={"true": "Her face is at least partly turned toward the camera: her eyes, nose or mouth can be seen.",
                      "false": "Only the back or the side of her head and her hair can be seen, or her face is hidden by the man."})
    return q


def build_request(jpg: Path | str, which: tuple = QUESTIONS) -> dict:
    """state, questions and assets of one crop's request (mono4d.judge helpers: upright baseline JPEG, the image asset)"""
    from ..mono4d.judge import _asset, prepare_image
    key, asset = _asset(prepare_image(jpg))
    return {"state": {"about": ABOUT, "photo": key}, "questions": jev_questions(which), "assets": {key: asset}}


def parse_answers(r) -> dict:
    """a SystemOneResponse -> {"R4": {option: p}, "R9": p, "R3": p, "_model", "_tokens"}"""
    out = {}
    for k, v in r.answers.items():
        if getattr(v, "type", "") == "choice":
            out[k] = {str(o): float(p) for o, p in dict(v.probabilities).items()}
        else:
            out[k] = float(v.noul)
    out["_model"] = getattr(r, "model", None)
    out["_tokens"] = int(getattr(getattr(r, "usage", None), "input_tokens", 0) or 0)
    return out


def ask(jpg: Path | str, which: tuple = QUESTIONS, client=None, model: str = "jev-latest") -> dict:
    """one Jev request for one crop (header TypeSafe-Danger-Zone-Images: true)"""
    from ..mono4d.judge import IMAGE_HEADERS
    if client is None:
        from ..judgments import client as _client
        client = _client()
    req = build_request(jpg, which)
    r = client.system_one(state=req["state"], questions=req["questions"], model=model, extra_headers=IMAGE_HEADERS,
                          extra_body={"assets": req["assets"]})
    return parse_answers(r)


def file_sha256(p: Path | str) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cache_key(sha: str, which: tuple, model: str) -> str:
    return f"{sha}|{','.join(sorted(which))}|{model}|{hashlib.sha256(ABOUT.encode()).hexdigest()[:8]}"


def load_cache(p: Path) -> dict:
    out = {}
    if Path(p).exists():
        for ln in Path(p).read_text(encoding="utf-8").splitlines():
            if ln.strip():
                d = json.loads(ln)
                out[d["key"]] = d["answers"]
    return out


def judge(rows: list, folder: Path, allow_upload: bool = False, which: tuple = QUESTIONS, client=None, cfg: CropCfg | None = None,
          cache: Path | None = None) -> dict:
    """Jev on every crop row ({"id", "file"}) -> {id: answers}; answers cached by image sha256 (+ questions, model, ABOUT) so a re-run
    sends nothing. Real dancers: refuses unless allow_upload (the owner's go-ahead, 2026-10-19). Failed requests are reported."""
    from concurrent.futures import ThreadPoolExecutor
    cfg = cfg or CropCfg()
    if not allow_upload:
        raise PermissionError("the crops show real people: pass allow_upload (the owner's go-ahead to send them to TypeSafe)")
    cache = Path(cache) if cache else Path(folder) / "jev_cache.jsonl"
    got = load_cache(cache)
    jobs, out, keys = [], {}, {}
    for r in rows:
        f = Path(folder) / r["file"]
        key = cache_key(file_sha256(f), which, cfg.model)
        keys[r["id"]] = key
        if key in got:
            out[r["id"]] = got[key]
        else:
            jobs.append((r["id"], f))

    def one(job):
        cid, f = job
        try:
            return cid, ask(f, which, client, cfg.model), None
        except Exception as e:  # reported, never counted as an answer
            return cid, None, f"{type(e).__name__}: {e}"

    errors = {}
    if jobs:
        with ThreadPoolExecutor(max_workers=max(1, cfg.workers)) as ex:
            done = list(ex.map(one, jobs))
        with open(cache, "a", encoding="utf-8") as fh:
            for cid, ans, err in done:
                if ans is None:
                    errors[cid] = err
                    continue
                out[cid] = ans
                fh.write(json.dumps({"key": keys[cid], "id": cid, "answers": ans}) + "\n")
    return {"answers": out, "errors": errors, "sent": len(jobs) - len(errors), "cached": len(rows) - len(jobs),
            "input_tokens_sent": int(sum(out[c].get("_tokens", 0) for c, _ in jobs if c in out))}


# ------------------------------------------------------------------------------------------------ answers -> evidence (pure)

def r4_class_probs(probs: dict, flatten: bool = True) -> tuple[dict, float]:
    """the Choice distribution over R4_OPTIONS -> ({front, flank, back} summing to 1, w_off): the torso classes renormalised, flattened
    toward uniform by the mass w_off on "elsewhere" + "not_visible" (an answer that does not see the hand on her torso is no evidence)"""
    p = {o: float(probs.get(o, 0.0)) for o in R4_OPTIONS}
    tors = np.array([p["front"], p["side"], p["back"]], float)
    w_off = float(min(1.0, max(0.0, p["elsewhere"] + p["not_visible"])))
    tot = float(tors.sum())
    norm = tors / tot if tot > 1e-9 else np.full(3, 1.0 / 3)
    if flatten:
        norm = (1.0 - w_off) * norm + w_off / 3.0
    return {"front": float(norm[0]), "flank": float(norm[1]), "back": float(norm[2])}, w_off


def look_r4_probs(answer: str, sure: bool, cfg: CropCfg) -> dict | None:
    """the agent's look -> a class distribution (None when the look gives no torso class)"""
    cls = R4_TO_CLASS.get(answer)
    if cls is None:
        return None
    hi = cfg.look_p_sure if sure else cfg.look_p_lean
    return {c: (hi if c == cls else (1.0 - hi) / 2.0) for c in CLASSES}


def jev_r4_option(probs: dict) -> str:
    return max(R4_OPTIONS, key=lambda o: float(probs.get(o, 0.0)))


def auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """ROC AUC of scores for binary labels (ties count 1/2); None without both classes"""
    s = np.asarray(scores, float)
    y = np.asarray(labels, bool)
    pos, neg = s[y], s[~y]
    if not len(pos) or not len(neg):
        return None
    gt = (pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()
    return float(gt / (len(pos) * len(neg)))


def _logit(p: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    p = np.clip(np.asarray(p, float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def platt_fit(p, y, prior_w: float = 1.0, iters: int = 60) -> tuple[float, float]:
    """Platt scaling of a Noul: P(yes) = sigmoid(a logit(p) + b), fitted by Newton on the logistic loss with an L2 prior pulling (a, b)
    toward the identity (1, 0) with weight prior_w (few labels)"""
    x = _logit(p)
    t = np.asarray(y, float)
    w = np.array([1.0, 0.0])
    for _ in range(iters):
        z = w[0] * x + w[1]
        q = 1.0 / (1.0 + np.exp(-z))
        g = np.array([((q - t) * x).sum(), (q - t).sum()]) + prior_w * (w - np.array([1.0, 0.0]))
        s = q * (1 - q)
        H = np.array([[(s * x * x).sum(), (s * x).sum()], [(s * x).sum(), s.sum()]]) + prior_w * np.eye(2)
        step = np.linalg.solve(H, g)
        w = w - step
        if np.abs(step).max() < 1e-10:
            break
    return float(w[0]), float(w[1])


def platt_apply(p, ab: tuple | list, lo: float = 0.02, hi: float = 0.98):
    a, b = ab
    z = a * _logit(p) + b
    return np.clip(1.0 / (1.0 + np.exp(-z)), lo, hi)


def loo_accuracy(p, y, prior_w: float = 1.0) -> float | None:
    """leave-one-out accuracy (at 0.5) of the Platt-calibrated Noul"""
    p = np.asarray(p, float)
    y = np.asarray(y, bool)
    if len(p) < 3:
        return None
    ok = []
    for i in range(len(p)):
        m = np.ones(len(p), bool)
        m[i] = False
        ab = platt_fit(p[m], y[m], prior_w)
        ok.append(bool((platt_apply(p[i:i + 1], ab)[0] >= 0.5) == y[i]))
    return float(np.mean(ok))


def calibrate_jev(labels: dict, answers: dict, cfg: CropCfg | None = None, prior_w: float = 1.0) -> dict:
    """labels {crop id: {"R4": option | "unclear", "R9": yes / no / unclear, "R3": ...}} (the known answers), answers {crop id: Jev}
    -> per question: n, accuracy (R4: arg-max option; also the 3-class accuracy on torso-labelled frames, which is the score decide
    uses, since only the torso classes become evidence), AUC (Nouls), the confusion, and the per-frame rows.
    Nouls are also Platt-calibrated (platt = (a, b)); their score is the leave-one-out accuracy of the calibrated answer, which is
    what the crops.json items carry (p = the calibrated probability)."""
    cfg = cfg or CropCfg()
    out = {}
    # R4
    rows, conf = [], {}
    for cid, lab in labels.items():
        a = answers.get(cid, {})
        L = lab.get("R4")
        if L not in R4_OPTIONS or "R4" not in a:
            continue
        opt = jev_r4_option(a["R4"])
        cp, w_off = r4_class_probs(a["R4"], cfg.r4_flatten)
        cls3 = max(CLASSES, key=lambda c: cp[c])
        rows.append({"id": cid, "label": L, "jev": opt, "p": a["R4"], "cls3": cls3, "w_off": w_off,
                     "ok5": opt == L, "ok3": (cls3 == R4_TO_CLASS[L]) if L in R4_TO_CLASS else None})
        conf.setdefault(L, {}).setdefault(opt, 0)
        conf[L][opt] += 1
    n5 = len(rows)
    t3 = [r for r in rows if r["ok3"] is not None]
    acc5 = float(np.mean([r["ok5"] for r in rows])) if rows else None
    acc3 = float(np.mean([r["ok3"] for r in t3])) if t3 else None
    out["R4"] = {"n": n5, "accuracy_5opt": acc5, "n_torso": len(t3), "accuracy_3class": acc3, "confusion": conf,
                 "score": acc3 if acc3 is not None else 0.0, "rows": rows}
    # Nouls. R9 labels: "yes" (her L arm over his R arm where they meet), "no" (they meet and her arm is UNDER his: the relevant
    # negative), "apart" (the arms do not meet: a trivial negative, reported in accuracy_incl_apart but kept out of the score, since
    # the decision the crops feed is over vs under in an entangled hold)
    for q in ("R9", "R3"):
        rows, apart = [], []
        for cid, lab in labels.items():
            a = answers.get(cid, {})
            L = lab.get(q)
            if q not in a:
                continue
            p = float(a[q])
            if L == "apart":
                apart.append({"id": cid, "label": L, "p": p, "ok": p < cfg.yes})
                continue
            if L not in ("yes", "no"):
                continue
            rows.append({"id": cid, "label": L, "p": p, "ok": (p >= cfg.yes) == (L == "yes")})
        acc_all = float(np.mean([r["ok"] for r in rows + apart])) if rows + apart else None
        acc = float(np.mean([r["ok"] for r in rows])) if rows else None
        pp = np.array([r["p"] for r in rows])
        yy = np.array([r["label"] == "yes" for r in rows])
        A = auc(pp, yy) if rows else None
        both = len(rows) >= 3 and yy.any() and (~yy).any()
        ab = platt_fit(pp, yy, prior_w) if both else None
        loo = loo_accuracy(pp, yy, prior_w) if both else None
        if ab is not None:
            pc = platt_apply(pp, ab)
            for r, c in zip(rows, pc):
                r["p_cal"] = float(c)
        why = "" if both else "no calibration frame of one of the two answers: the question is not calibrated (check only)"
        out[q] = {"n": len(rows), "n_yes": int(yy.sum()), "n_no": int((~yy).sum()) if rows else 0, "n_apart": len(apart),
                  "accuracy": acc, "accuracy_incl_apart": acc_all, "auc": A, "platt": list(ab) if ab else None,
                  "accuracy_loo_platt": loo, "score": loo if loo is not None else 0.0, "why": why, "rows": rows + apart}
    return out


def noul_p(p: float, cal_q: dict) -> float:
    """a Noul's P(yes) through its Platt calibration when the question has one (else the raw answer)"""
    ab = (cal_q or {}).get("platt")
    return float(platt_apply(np.array([p]), ab)[0]) if ab else float(p)


def decide_items(index: list, answers: dict, looks: dict, calib: dict, r9_map: dict | None = None, cfg: CropCfg | None = None) -> list:
    """crops.json items (the sides.decide.load_crops contract) for the TARGET crops and their planned questions: one Jev item per
    (crop, question) with the question's calibration score, and one agent item per crop where the look gives an answer (score 1.0).
    r9_map {crop id: plus_if_her_on_top (+1 / -1 / 0)} turns "her arm on top" into P(the + side of R9_RL); 0 = no R9 item."""
    cfg = cfg or CropCfg()
    qtext = {k: (v.instructions if hasattr(v, "instructions") else str(v)) for k, v in _question_texts().items()}
    items = []
    for row in index:
        if row.get("set") != "target":
            continue
        cid, t, cam = row["id"], float(row["t"]), row["cam"]
        a = answers.get(cid, {})
        lk = looks.get(cid, {})
        for q in row.get("questions", []):
            score = float(calib.get(q, {}).get("score", 0.0) or 0.0)
            look = lk.get(q)
            base = {"t": t, "cam": cam, "id": cid, "question": qtext.get(q, q), "calibration_score": score, "agent_look": look,
                    "agent_sure": bool(lk.get(f"{q}_sure", False)), "agent_note": lk.get("note", "")}
            if q == "R4":
                if "R4" in a:
                    cp, w_off = r4_class_probs(a["R4"], cfg.r4_flatten)
                    items.append({**base, "relation": "R4", "p": cp, "score": score, "source": "jev", "probabilities": a["R4"],
                                  "w_off": w_off})
                lp = look_r4_probs(look, bool(lk.get("R4_sure", False)), cfg) if look else None
                if lp is None and isinstance(lk.get("R4_p"), dict):     # a look that only rules a class out (e.g. "not on her front")
                    v = np.array([float(lk["R4_p"].get(c, 0.0)) for c in CLASSES])
                    lp = {c: float(x) for c, x in zip(CLASSES, v / v.sum())} if v.sum() > 0 else None
                if lp is not None:
                    items.append({**base, "relation": "R4", "p": lp, "score": 1.0, "source": "agent"})
            elif q == "R9":
                sgn = int((r9_map or {}).get(cid, 0))
                if sgn == 0:
                    continue
                if "R9" in a:
                    p = noul_p(float(a["R9"]), calib.get("R9"))
                    items.append({**base, "relation": "R9_RL", "p": {"+": p if sgn > 0 else 1.0 - p}, "score": score, "source": "jev",
                                  "p_yes": float(a["R9"]), "p_yes_cal": p, "plus_if_her_on_top": sgn})
                if look in ("yes", "no"):
                    hi = cfg.look_p_sure if lk.get("R9_sure") else cfg.look_p_lean
                    p = hi if look == "yes" else 1.0 - hi
                    items.append({**base, "relation": "R9_RL", "p": {"+": p if sgn > 0 else 1.0 - p}, "score": 1.0, "source": "agent",
                                  "plus_if_her_on_top": sgn})
            elif q == "R3":
                if "R3" in a:
                    p = noul_p(float(a["R3"]), calib.get("R3"))
                    items.append({**base, "relation": "R3_follow", "role": "follow", "p": {"+": p}, "score": score, "source": "jev",
                                  "p_yes": float(a["R3"]), "p_yes_cal": p})
                if look in ("yes", "no"):
                    hi = cfg.look_p_sure if lk.get("R3_sure") else cfg.look_p_lean
                    items.append({**base, "relation": "R3_follow", "role": "follow", "p": {"+": hi if look == "yes" else 1.0 - hi},
                                  "score": 1.0, "source": "agent"})
    return items


def _question_texts() -> dict:
    try:
        return jev_questions()
    except Exception:  # typesafe_sdk missing: plain texts
        return {"R4": "Where is the man's right hand?", "R9": "Does the woman's left arm lie on top of the man's right arm?",
                "R3": "Can the woman's face be seen?"}


# ------------------------------------------------------------------------------------------------ sheets

def label_tile(img: np.ndarray, lines: list, scale: float = 0.55) -> np.ndarray:
    import cv2
    out = img.copy()
    y = int(22 * scale / 0.55)
    for ln in lines:
        cv2.putText(out, ln, (6, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, ln, (6, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
        y += int(30 * scale) + 4
    return out


def grid(tiles: list, cols: int, pad: int = 4, bg: int = 20) -> np.ndarray:
    if not tiles:
        raise ValueError("no tiles")
    h, w = tiles[0].shape[:2]
    rows = int(math.ceil(len(tiles) / cols))
    out = np.full((rows * (h + pad) + pad, cols * (w + pad) + pad, 3), bg, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        out[pad + r * (h + pad): pad + r * (h + pad) + h, pad + c * (w + pad): pad + c * (w + pad) + w] = t
    return out


def owner_pick(index: list, answers: dict, looks: dict, n: int = 6, group_prefix: str | None = None) -> list:
    """<= n target R4 crops for the owner's one look (DESIGN D3): the ones where the evidence is least settled (agent look unclear or
    leaning, or Jev's torso class disagreeing with the look, or Jev spread), one per (time, camera), at most 2 per time, spread over
    the cameras; ties by time"""
    cand = []
    for r in index:
        if r.get("set") != "target" or "R4" not in r.get("questions", []):
            continue
        if group_prefix and not str(r.get("group", "")).startswith(group_prefix):
            continue
        a = answers.get(r["id"], {}).get("R4")
        lk = looks.get(r["id"], {})
        look = lk.get("R4")
        u = 0.0
        if look in (None, "unclear"):
            u += 2.0
        elif not lk.get("R4_sure"):
            u += 1.0
        if look in ("not_visible",):
            u -= 1.5           # nothing to show the owner
        if a is not None:
            cp, w_off = r4_class_probs(a)
            ent = -sum(p * math.log(max(p, 1e-9)) for p in cp.values()) / math.log(3)
            u += ent
            if look in R4_TO_CLASS and max(cp, key=cp.get) != R4_TO_CLASS[look]:
                u += 1.0
        cand.append((u, r))
    cand.sort(key=lambda x: (-x[0], x[1]["t"]))
    out, per_t, per_cam = [], {}, {}
    for u, r in cand:
        if len(out) >= n:
            break
        if per_t.get(r["t"], 0) >= 2 or per_cam.get(r["cam"], 0) >= 2:
            continue
        out.append(r)
        per_t[r["t"]] = per_t.get(r["t"], 0) + 1
        per_cam[r["cam"]] = per_cam.get(r["cam"], 0) + 1
    return sorted(out, key=lambda r: (r["t"], r["cam"]))[:n]


def film_label(t: float, film_map=None) -> str:
    ft = film_map(t) if film_map is not None else None
    if ft is None:
        return f"dance {t:.2f} s"
    ft = float(ft)
    return f"film {int(ft // 60)}:{ft % 60:05.2f}"


OWNER_QUESTION = "The leader's right hand: on her back / at her side under her arm / on her front?  (answer per letter)"


def owner_sheet(take: str, n: int = 6, work: Path | None = None, film_map=None, group_prefix: str | None = None,
                ids: list | None = None, question: str | None = None) -> Path:
    """the <= n-crop owner sheet (DESIGN D3): raw crops, letters A.., time, phone, and the one question (ids: the agent's explicit
    pick, else owner_pick; question: the plan's "owner_question", else OWNER_QUESTION)"""
    import cv2
    folder = crops_dir(take, work)
    question = question or (read_json(folder / "plan.json", {}) or {}).get("owner_question") or OWNER_QUESTION
    index = read_json(folder / "index.json", {"rows": []})["rows"]
    answers = read_json(folder / "jev_rows.json", {"answers": {}})["answers"]
    looks = read_json(folder / "looks.json", {})
    if ids:
        by = {r["id"]: r for r in index}
        pick = [by[i] for i in ids if i in by][:6]
    else:
        pick = owner_pick(index, answers, looks, min(int(n), 6), group_prefix)
    tiles = []
    for i, r in enumerate(pick):
        img = cv2.imread(str(folder / r["file"]))
        tiles.append(label_tile(img, [f"{chr(65 + i)}   {film_label(r['t'], film_map)}   phone {r['cam']}"], 0.7))
    sheet = grid(tiles, 3 if len(tiles) > 4 else 2)
    head = np.full((70, sheet.shape[1], 3), 20, np.uint8)
    cv2.putText(head, question, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(head, "raw phone crops, no overlay; left / right = the dancer's own", (8, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (200, 200, 200), 1, cv2.LINE_AA)
    out = folder / "owner_sheet.jpg"
    cv2.imwrite(str(out), np.concatenate([head, sheet], 0), [cv2.IMWRITE_JPEG_QUALITY, 90])
    write_json(folder / "owner_sheet.json", {"question": question,
                                             "letters": {chr(65 + i): {"id": r["id"], "t": r["t"], "cam": r["cam"],
                                                                      "film": film_label(r["t"], film_map)} for i, r in enumerate(pick)}})
    return out


# markers of the agent's look sheets (BGR): which arm is which, from the base fit (never shown to Jev)
MARKS = (("lead", 21, (0, 255, 255), "R"), ("lead", 19, (0, 200, 255), "e"), ("lead", 20, (255, 255, 0), "L"),
         ("follow", 20, (255, 0, 255), "l"), ("follow", 18, (200, 0, 255), "e"), ("follow", 21, (0, 255, 0), "r"))


def annotate(scene: "Scene", row: dict, img: np.ndarray, cfg: CropCfg | None = None) -> np.ndarray:
    """small hollow markers of the base fit's wrists / elbows on a crop (agent's look only): his right wrist yellow "R", his right
    elbow orange "e", his left wrist cyan "L"; her left wrist magenta "l", her left elbow purple "e", her right wrist green "r"""
    import cv2
    cfg = cfg or CropCfg()
    k = int(row["k"])
    c = scene.camera(row["cam"], k)
    cx, cy, side = row["box"]
    Kc = crop_K(c["K"], cx, cy, side, cfg.tile)
    out = img.copy()
    for role, j, col, txt in MARKS:
        uv, d = project(Kc, c["R"], c["t"], scene.J[role][k, j][None])
        if d[0] <= 0 or not np.isfinite(uv).all():
            continue
        p = (int(round(uv[0, 0])), int(round(uv[0, 1])))
        cv2.circle(out, p, 8, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.circle(out, p, 8, col, 1, cv2.LINE_AA)
        cv2.putText(out, txt, (p[0] + 9, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, txt, (p[0] + 9, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    return out


def look_sheets(take: str, work: Path | None = None, per: int = 4, ids: list | None = None, which_set: str | None = None,
                marks: bool = False, name: str | None = None) -> list:
    """2x2 sheets of the crops (with their id) for the agent's own look -> crops/look/<name>_<nn>.jpg (marks: the base fit's wrists)"""
    import cv2
    folder = crops_dir(take, work)
    rows = read_json(folder / "index.json", {"rows": []})["rows"]
    if ids is not None:
        order = {i: n for n, i in enumerate(ids)}
        rows = sorted([r for r in rows if r["id"] in order], key=lambda r: order[r["id"]])
    if which_set:
        rows = [r for r in rows if r["set"] == which_set]
    scene = Scene(take, work) if marks else None
    name = name or ((which_set or "all") + ("_m" if marks else ""))
    out = []
    (folder / "look").mkdir(parents=True, exist_ok=True)
    for s in range(0, len(rows), per):
        tiles = []
        for r in rows[s:s + per]:
            img = cv2.imread(str(folder / r["file"]))
            if marks:
                img = annotate(scene, r, img)
            tiles.append(label_tile(img, [f"{r['id']}  {r['set']}  {'/'.join(r.get('questions', []))}"], 0.6))
        while len(tiles) < per:
            tiles.append(np.full_like(tiles[0], 20))
        p = folder / "look" / f"{name}_{s // per:02d}.jpg"
        cv2.imwrite(str(p), grid(tiles, 2), [cv2.IMWRITE_JPEG_QUALITY, 88])
        out.append({"sheet": str(p), "ids": [r["id"] for r in rows[s:s + per]]})
    write_json(folder / "look" / f"sheets_{name}.json", out)
    return out


# ------------------------------------------------------------------------------------------------ steps

def make(take: str, plan_path: Path | None = None, work: Path | None = None, cfg: CropCfg | None = None, rel_path: Path | None = None,
         force: bool = False) -> dict:
    """crop every planned (time, camera) -> crops/<id>.jpg + crops/index.json (cameras "auto" = every camera that sees the couple)"""
    import cv2
    cfg = cfg or CropCfg()
    folder = crops_dir(take, work)
    folder.mkdir(parents=True, exist_ok=True)
    plan = read_json(plan_path or folder / "plan.json")
    if plan is None:
        raise FileNotFoundError(f"no crop plan at {plan_path or folder / 'plan.json'}")
    rel = None
    rp = rel_path or (folder.parent / "relations.npz")
    if Path(rp).exists() and plan.get("auto_from_relations", False):
        with np.load(rp, allow_pickle=True) as z:
            rel = {k: z[k] for k in z.files}
    entries = crop_times(rel, plan, cfg.hz)
    scene = Scene(take, work, plan.get("base_fit", "smplx_rf2_noslip"))
    old = {r["id"]: r for r in read_json(folder / "index.json", {"rows": []})["rows"]}
    rows, skipped = {}, []
    for e in entries:
        cams = scene.cams if e["cams"] == "auto" else [c for c in e["cams"] if c in scene.cams]
        k = int(round(e["t"] * cfg.hz))
        for cam in cams:
            cid = crop_id(e["t"], cam, e["kind"])
            if cid in rows:                             # the same crop asked by two groups: merge the questions
                r = rows[cid]
                r["questions"] = sorted(set(r["questions"]) | set(e["questions"]))
                if e["set"] == "target":
                    r["set"] = "target"
                continue
            if e["cams"] == "auto":
                ok, share = scene.sees(cam, k, e["kind"], cfg)
                if not ok:
                    skipped.append({"id": cid, "why": f"auto: couple in frame {share:.2f}"})
                    continue
            f = folder / f"{cid}.jpg"
            row = {"id": cid, "t": e["t"], "k": k, "cam": cam, "kind": e["kind"], "file": f.name, "set": e["set"],
                   "group": e["group"], "questions": list(e["questions"]), "why": e["why"]}
            if f.exists() and not force and cid in old and "box" in old[cid]:
                row.update({kk: old[cid][kk] for kk in ("box", "dt_s", "in_frame") if kk in old[cid]})
                rows[cid] = row
                continue
            img, meta = crop(scene, cam, e["t"], e["kind"], cfg)
            if img is None:
                skipped.append({"id": cid, "why": meta.get("why")})
                continue
            cv2.imwrite(str(f), img, [cv2.IMWRITE_JPEG_QUALITY, cfg.jpeg_quality])
            row.update({"box": meta["box"], "dt_s": meta["dt_s"], "in_frame": meta["in_frame"]})
            rows[cid] = row
            log(f"{cid}  ({e['group']})")
    rows = sorted(rows.values(), key=lambda r: (r["t"], r["cam"], r["kind"]))
    rep = {"what": "sides crop check: raw phone crops (local only, never uploaded)", "take": take, "cfg": cfg.to_json(),
           "plan": str(plan_path or folder / "plan.json"), "rows": rows, "skipped": skipped}
    write_json(folder / "index.json", rep)
    return {"crops": len(rows), "skipped": len(skipped)}


def ask_step(take: str, allow_upload: bool, which_set: str = "all", work: Path | None = None, cfg: CropCfg | None = None) -> dict:
    cfg = cfg or CropCfg()
    folder = crops_dir(take, work)
    rows = read_json(folder / "index.json")["rows"]
    if which_set != "all":
        rows = [r for r in rows if r["set"] == which_set]
    res = judge(rows, folder, allow_upload, QUESTIONS, None, cfg)
    prev = read_json(folder / "jev_rows.json", {"answers": {}})
    prev["answers"].update(res["answers"])
    prev.update({"model": cfg.model, "questions": {k: str(getattr(v, "instructions", v)) for k, v in _question_texts().items()},
                 "about": ABOUT})
    prev.setdefault("runs", []).append({k: res[k] for k in ("sent", "cached", "input_tokens_sent")} | {"errors": res["errors"]})
    write_json(folder / "jev_rows.json", prev)
    return {k: res[k] for k in ("sent", "cached", "input_tokens_sent")} | {"errors": len(res["errors"])}


def calibrate_step(take: str, work: Path | None = None, cfg: CropCfg | None = None) -> dict:
    """the calibration crops (set "calib") against their known answers: crops/calib_labels.json {id: {"R4", "R9", "R3", "basis"}}"""
    cfg = cfg or CropCfg()
    folder = crops_dir(take, work)
    labels = read_json(folder / "calib_labels.json")
    if labels is None:
        raise FileNotFoundError(f"{folder / 'calib_labels.json'}: the known answers of the calibration crops")
    labels = {k: v for k, v in labels.items() if not k.startswith("_")}
    answers = read_json(folder / "jev_rows.json", {"answers": {}})["answers"]
    cal = calibrate_jev(labels, answers, cfg)
    looks = {k: v for k, v in read_json(folder / "looks.json", {}).items() if not k.startswith("_") and k not in labels}
    hold = holdout_vs_looks(looks, answers, cal, cfg)
    for q in QUESTIONS:
        cal[q]["holdout_looks"] = hold.get(q)
        h = hold.get(q) or {}
        if h.get("n", 0) >= min_holdout and h.get("accuracy") is not None:
            cal[q]["score"] = float(min(cal[q]["score"], h["accuracy"]))
    rep = {"what": "Jev calibration on crops with known answers (+ a holdout: the agent's looks at every other crop)", "n_frames": len(labels),
           "scores": {q: cal[q]["score"] for q in QUESTIONS}, "min_score_for_evidence": cfg.crop_min_score, **cal}
    write_json(folder / "calibration.json", rep)
    return {q: {kk: cal[q][kk] for kk in cal[q] if kk not in ("rows", "confusion")} for q in QUESTIONS}


min_holdout = 20


def holdout_vs_looks(looks: dict, answers: dict, cal: dict, cfg: CropCfg | None = None) -> dict:
    """Jev against the agent's looks on the crops outside the calibration set: R4 (arg-max option, and the torso 3-class arg-max on
    torso-labelled crops), Nouls through their Platt calibration (yes / no looks only)"""
    cfg = cfg or CropCfg()
    out = {}
    r4 = [(k, v["R4"]) for k, v in looks.items() if v.get("R4") in R4_OPTIONS and "R4" in answers.get(k, {})]
    if r4:
        tors = [(k, L) for k, L in r4 if L in R4_TO_CLASS]
        a3 = [max(r4_class_probs(answers[k]["R4"], cfg.r4_flatten)[0].items(), key=lambda x: x[1])[0] == R4_TO_CLASS[L] for k, L in tors]
        out["R4"] = {"n": len(r4), "accuracy_5opt": float(np.mean([jev_r4_option(answers[k]["R4"]) == L for k, L in r4])),
                     "n_torso": len(tors), "accuracy": float(np.mean(a3)) if a3 else None}
    for q in ("R9", "R3"):
        rows = [(k, v[q]) for k, v in looks.items() if v.get(q) in ("yes", "no") and q in answers.get(k, {})]
        if rows:
            out[q] = {"n": len(rows), "n_yes": sum(L == "yes" for _, L in rows),
                      "accuracy_raw": float(np.mean([(float(answers[k][q]) >= cfg.yes) == (L == "yes") for k, L in rows])),
                      "accuracy": float(np.mean([(noul_p(float(answers[k][q]), cal.get(q)) >= 0.5) == (L == "yes") for k, L in rows]))}
    return out


def r9_map_for(take: str, index: list, work: Path | None = None, base_fit: str = "smplx_rf2_noslip", cfg: CropCfg | None = None) -> dict:
    """{crop id: the R9 sign map row} for the target crops that ask R9, from the base fit's arm chains at the crop's frame"""
    cfg = cfg or CropCfg()
    work = Path(work) if work else work_dir(take)
    J = {}
    for r in ("lead", "follow"):
        with np.load(work / base_fit / f"{r}.npz", allow_pickle=True) as z:
            J[r] = np.asarray(z["joints"], float)
    out = {}
    for row in index:
        if "R9" not in row.get("questions", []):
            continue
        k = int(row["k"])
        A = J["lead"][k, [17, 19, 21]]
        B = J["follow"][k, [16, 18, 20]]
        out[row["id"]] = r9_sign_map(A, B, cfg.r9_min_vertical, cfg.r9_min_abs_g, cfg.r9_max_dist_m)
    return out


def write_step(take: str, work: Path | None = None, cfg: CropCfg | None = None) -> dict:
    """-> work/<take>/sides/crops.json (decide's contract + per crop: question, P(yes), agent look, calibration score)"""
    cfg = cfg or CropCfg()
    folder = crops_dir(take, work)
    idx = read_json(folder / "index.json")
    index = idx["rows"]
    answers = read_json(folder / "jev_rows.json", {"answers": {}})["answers"]
    looks = {k: v for k, v in read_json(folder / "looks.json", {}).items() if not k.startswith("_")}
    cal = read_json(folder / "calibration.json", {})
    r9 = r9_map_for(take, index, work, cfg=cfg)
    items = decide_items(index, answers, looks, cal, {k: v["plus_if_her_on_top"] for k, v in r9.items()}, cfg)
    per = []
    for row in index:
        a = answers.get(row["id"], {})
        per.append({"id": row["id"], "t": row["t"], "cam": row["cam"], "set": row["set"], "group": row.get("group"),
                    "questions": row.get("questions", []), "file": f"crops/{row['file']}",
                    "jev": {q: a.get(q) for q in QUESTIONS if q in a}, "look": looks.get(row["id"]),
                    "r9_map": r9.get(row["id"])})
    out = {"what": "sides crop check (DESIGN 2.3 l_crop): Jev + the agent's look on raw phone crops",
           "take": take, "contract": "items: t, cam, relation, p, score, source (sides.decide.load_crops); score < crop_min_score is ignored",
           "crop_min_score": cfg.crop_min_score,
           "calibration": {q: {kk: cal.get(q, {}).get(kk) for kk in ("n", "accuracy", "accuracy_5opt", "accuracy_3class", "n_torso", "auc",
                                                                       "score")} for q in QUESTIONS},
           "questions": {k: str(getattr(v, "instructions", v)) for k, v in _question_texts().items()},
           "items": items, "crops": per}
    p = folder.parent / "crops.json"
    write_json(p, out)
    used = [it for it in items if float(it["score"]) >= cfg.crop_min_score]
    return {"items": len(items), "used_by_decide": len(used), "file": str(p)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.sides.crops")
    ap.add_argument("verb", choices=["make", "ask", "looksheet", "calibrate", "write", "sheet"])
    ap.add_argument("--take", required=True)
    ap.add_argument("--plan", type=Path, default=None)
    ap.add_argument("--set", default="all", choices=["all", "calib", "target"])
    ap.add_argument("--allow-upload", action="store_true", help="ask: send the crops (real dancers) to TypeSafe (owner's go-ahead)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--group", default=None, help="sheet: only crops of plan groups starting with this")
    ap.add_argument("--ids", default=None, help="sheet: comma-separated crop ids (the agent's pick, <= 6)")
    a = ap.parse_args(argv)
    if a.verb == "make":
        print(json.dumps(make(a.take, a.plan, force=a.force)))
    elif a.verb == "ask":
        print(json.dumps(ask_step(a.take, a.allow_upload, a.set)))
    elif a.verb == "looksheet":
        print(json.dumps([s["sheet"] for s in look_sheets(a.take, which_set=None if a.set == "all" else a.set)]))
    elif a.verb == "calibrate":
        print(json.dumps(calibrate_step(a.take), indent=1))
    elif a.verb == "write":
        print(json.dumps(write_step(a.take)))
    elif a.verb == "sheet":
        fm = None
        try:
            from .filmtime import FilmMap
            fm = FilmMap.for_work(work_dir(a.take)).main
        except Exception:
            fm = None
        print(owner_sheet(a.take, a.n, film_map=fm, group_prefix=a.group, ids=a.ids.split(",") if a.ids else None))
    return 0


if __name__ == "__main__":
    sys.exit(main())
