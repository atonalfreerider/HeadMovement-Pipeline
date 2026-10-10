"""Recap people (agent R3): who are the two TEACHERS in every camera, at every frame of the class (2026-10-09).

Input: the owner's per-video 2D pose JSONs (17 COCO joints per person per frame, display pixels, ALL people in view: teachers,
about 80-100 spectators, mirror ghosts) and the camera tracks of R2 (cameras/<cam>.npz + <cam>_world.npz, room frame). Output:
per camera and canonical frame, which JSON person is Kadu (lead) and which is Larissa (follow), with a confidence, and a coverage
map per teacher per second. No student, bystander or reflection gets an identity.

How (every step is numpy on the pose tables, plus ffmpeg + OpenCV for the colour evidence; nothing here needs a neural network):
  tables      JSON(.gz) -> work/larissa_kadu_recap/people/pose_tables/<cam>.npz  kp (F, P, 17, 3), n (F,)
  world       every 30 Hz instant of the master clock: the people of the four main cameras are matched across cameras by the
              epipolar cost of their joints, clustered (one person per camera and cluster) and triangulated into 3D people in the
              metric room frame (Y up, floor y = 0). A mirror ghost is seen by one camera only (no cluster) or triangulates
              behind the wall (outside the dance zone); seated spectators are low (shoulders < 1 m); standing spectators are
              far from the dance floor. -> people/world_people.npz
  teachers    a Viterbi over ORDERED PAIRS of 3D candidates (slot A, slot B, or "missing") picks the two continuous 3D tracks of
              the people who stand in the dance zone (continuity, standing height, zone, number of views). Anonymous so far.
  assoc       the two 3D tracks are projected into EVERY camera (also the two attached phones 05 / 06) and matched to the JSON
              persons of the frame (Hungarian, joint distances relative to the person's size). -> people/assoc_<cam>.npz
  appearance  colour evidence on the pixels of the associated persons (dancers_appearance: the lead's black top / dark jeans /
              no red hair against the follow's white top / light jeans / red hair), one 2-state keep / swap Viterbi over ALL
              cameras of both 3D tracks (switching only allowed where the two are close). The reference portraits are used
              only to look at the colours; no face matching, no copy of them anywhere.
  outputs     people/teachers_<cam>.npz (per canonical frame: person index, box, 17 joints, confidence per role), people/
              teachers.json (summary), people/coverage.json (per second and role: cameras that see the teacher).

    python -m dancecap.recap_people tables|world|teachers|assoc|appearance|finalize|review|run [--cams ...]
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .take import PACKAGE_ROOT

# HM_FRONT_DANCE / HM_FRONT_CAPTURE (C3, additive): the same stages serve another take that supplies the same inputs (carlos_aline); the defaults are the recap
TAKE = os.environ.get("HM_FRONT_DANCE", "larissa_kadu_recap")
CAPTURE = os.environ.get("HM_FRONT_CAPTURE", "Larissa-Kadu-recap")
WORK = PACKAGE_ROOT / "work" / TAKE
PEOPLE = WORK / "people"
SOURCE = PACKAGE_ROOT.parent / "dances" / CAPTURE      # git-ignored junction to the owner's folder (never a repo path)
WINDOW = (0.0, 360.48)
HZ = 30.0
K_N = 10815                                            # grid instants 0 .. 360.47 s
MAIN = ("00", "01", "02", "03")                        # cameras localised directly in the room model (R2)
ATTACHED = ("05", "06")                                # rotation-only attach: low weight
ALL_CAMS = MAIN + ATTACHED
ROLES = ("lead", "follow")                              # Kadu = lead (pid 1), Larissa = follow (pid 2)
CONF = 0.5                                              # a joint counts as seen above this confidence (attach_people.CONF)
MIN_JOINTS = 6


def log(msg: str) -> None:
    print(f"[recap_people {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _root() -> Path | None:
    """The volume root on a pod ($HM_ROOT) else None."""
    r = os.environ.get("HM_ROOT")
    return Path(r) if r else None


# ------------------------------------------------------------------------------------------------ inputs


def cams_table() -> dict:
    """{cam: {file, width, height, rotation, avg_fps, offset, slope, ...}} (R2's recap_cams.json, the pod gets it as an input)."""
    return json.loads((WORK / "cameras" / "recap_cams.json").read_text())["cams"]


def video_path(cam: str) -> Path:
    row = cams_table()[cam]
    r = _root()
    p = (r / "src" / CAPTURE / row["file"]) if r else SOURCE / row["file"]
    return p


def pose_json_path(cam: str) -> Path:
    row = cams_table()[cam]
    r = _root()
    if r:
        p = r / "src" / CAPTURE / "poses" / f"{row['file']}.json.gz"
        if p.exists():
            return p
        return r / "src" / CAPTURE / f"{row['file']}.json"
    return SOURCE / f"{row['file']}.json"


def load_pose_json(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """The owner's 2D pose JSON (.json / .json.gz) -> (kp (F, P, 17, 3) float32 [X, Y, conf] with NaN padding, n (F,) persons per frame).
    Coordinates are DISPLAY pixels (R1/R2 checked the two rotated videos)."""
    path = Path(path)
    with (gzip.open if path.suffix == ".gz" else open)(path, "rb") as f:
        raw = json.load(f)
    F = len(raw)
    P = max([len(fr) for fr in raw] + [1])
    kp = np.full((F, P, 17, 3), np.nan, np.float32)
    n = np.zeros(F, np.int16)
    for i, fr in enumerate(raw):
        n[i] = len(fr)
        for j, person in enumerate(fr):
            a = np.array([[k["X"], k["Y"], k["Z"]] for k in person], np.float32)
            kp[i, j, :len(a)] = a
    return kp, n


def pose_table(cam: str, force: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Cached pose table of one camera (people/pose_tables/<cam>.npz)."""
    p = PEOPLE / "pose_tables" / f"{cam}.npz"
    if p.exists() and not force:
        z = np.load(p)
        return z["kp"], z["n"]
    t0 = time.time()
    kp, n = load_pose_json(pose_json_path(cam))
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, kp=kp, n=n)
    log(f"pose table {cam}: {kp.shape[0]} frames, up to {kp.shape[1]} persons ({time.time() - t0:.0f} s)")
    return kp, n


@dataclass
class Cam:
    """One camera: R2's track (canonical frames) + the pose table. R, tvec are WORLD -> camera (the _world.npz of R2)."""
    id: str
    t: np.ndarray                # (F,) master clock (audio clock of R1)
    ok: np.ndarray               # (F,)
    fi: np.ndarray               # (F,) video frame index = row of the pose table
    K: np.ndarray                # (F,3,3)
    R: np.ndarray                # (F,3,3)
    tvec: np.ndarray             # (F,3)
    C: np.ndarray                # (F,3) centre in the world frame
    k1: np.ndarray               # (F,)
    size: tuple                  # display (w, h)
    attached: bool
    pts: np.ndarray = field(repr=False, default=None)   # (F,) the video's own presentation time of the frame
    kp: np.ndarray = field(repr=False, default=None)
    n: np.ndarray = field(repr=False, default=None)

    def persons(self, i: int) -> np.ndarray:
        """(P, 17, 3) people of canonical frame i."""
        r = int(self.fi[i])
        if r >= len(self.n):
            return np.zeros((0, 17, 3), np.float32)
        return self.kp[r, :int(self.n[r])]

    def near(self, grid_t: np.ndarray, max_gap: float = 0.04) -> np.ndarray:
        """For every instant of grid_t the canonical frame index nearest in time (and ok), else -1."""
        j = np.clip(np.searchsorted(self.t, grid_t), 1, len(self.t) - 1)
        j = np.where(np.abs(self.t[j - 1] - grid_t) < np.abs(self.t[j] - grid_t), j - 1, j)
        good = (np.abs(self.t[j] - grid_t) <= max_gap) & self.ok[j]
        return np.where(good, j, -1)


def load_cam(cam: str, with_poses: bool = True) -> Cam:
    cdir = WORK / "cameras"
    z = np.load(cdir / f"{cam}_world.npz", allow_pickle=True)
    z0 = np.load(cdir / f"{cam}.npz", allow_pickle=True)
    row = cams_table()[cam]
    w, h = (int(x) for x in z0["native_size"])
    c = Cam(cam, np.asarray(z["t"], float), np.asarray(z["ok"], bool), np.asarray(z["frame_index"], int), np.asarray(z["K"], float),
            np.asarray(z["R"], float), np.asarray(z["tvec"], float), np.asarray(z["C"], float), np.asarray(z["dist"], float),
            (w, h), bool(z0["attached"]) if "attached" in z0.files else False, np.asarray(z["pts"], float))
    if with_poses:
        c.kp, c.n = pose_table(cam)
    return c


# ------------------------------------------------------------------------------------------------ geometry helpers


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def undistort_norm(xy: np.ndarray, K: np.ndarray, k1: float) -> np.ndarray:
    """Pixel -> normalised undistorted coordinates (SIMPLE_RADIAL k1, as R2's tracks)."""
    d = (np.asarray(xy, float) - [K[0, 2], K[1, 2]]) / K[0, 0]
    u = d.copy()
    for _ in range(6):
        u = d / (1 + k1 * (u ** 2).sum(-1, keepdims=True))
    return u


def project(X: np.ndarray, R: np.ndarray, tvec: np.ndarray, K: np.ndarray, k1: float) -> tuple[np.ndarray, np.ndarray]:
    """World points (...,3) -> (pixels (...,2), depth (...)) through a camera (x_cam = R X + t, SIMPLE_RADIAL k1 on the normalised plane)."""
    Xc = X @ R.T + tvec
    z = Xc[..., 2]
    zz = np.where(z > 1e-6, z, 1e-6)
    x, y = Xc[..., 0] / zz, Xc[..., 1] / zz
    d = 1 + k1 * (x * x + y * y)
    return np.stack([K[0, 0] * x * d + K[0, 2], K[1, 1] * y * d + K[1, 2]], -1), z


def person_height_px(p: np.ndarray, conf: float = 0.3) -> float:
    """Pixel extent of a person's confident joints (max of the two sides): its scale."""
    s = p[:, 2] >= conf
    if s.sum() < 3:
        return float("nan")
    xy = p[s, :2]
    return float(max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1])))


def person_box(p: np.ndarray, conf: float = 0.3, margin: float = 0.08) -> np.ndarray:
    """xyxy box of the confident joints grown by `margin` of the size (and 12 % at the top for the head): the JSON has joints only."""
    s = p[:, 2] >= conf
    if s.sum() < 3:
        return np.full(4, np.nan, np.float32)
    xy = p[s, :2]
    x0, y0 = xy.min(0)
    x1, y1 = xy.max(0)
    w, h = x1 - x0, y1 - y0
    m = margin * max(w, h)
    return np.array([x0 - m, y0 - m - 0.12 * h, x1 + m, y1 + m], np.float32)


# ------------------------------------------------------------------------------------------------ 3D people of one instant


def dlt_joints(U: np.ndarray, ok: np.ndarray, Pm: np.ndarray) -> np.ndarray:
    """Batched multi-view DLT. U (V,17,2) normalised coordinates, ok (V,17) usable, Pm (V,3,4) -> X (17,3) (NaN: fewer than 2 views)."""
    V = len(U)
    rows = np.zeros((17, 2 * V, 4))
    for v in range(V):
        w = ok[v].astype(float)[:, None]
        rows[:, 2 * v] = w * (U[v, :, :1] * Pm[v, 2] - Pm[v, 0])
        rows[:, 2 * v + 1] = w * (U[v, :, 1:] * Pm[v, 2] - Pm[v, 1])
    _, _, vt = np.linalg.svd(rows)
    Xh = vt[:, -1, :]
    X = Xh[:, :3] / np.where(np.abs(Xh[:, 3:]) < 1e-12, 1e-12, Xh[:, 3:])
    X[ok.sum(0) < 2] = np.nan
    return X


def reproj_px(X: np.ndarray, Pm: np.ndarray, U: np.ndarray, f: np.ndarray) -> np.ndarray:
    """(V,17) reprojection distance in pixels of the 3D joints X against the normalised observations U (NaN: no 3D / behind a camera)."""
    xc = np.einsum("vij,kj->vki", Pm[:, :, :3], np.nan_to_num(X)) + Pm[:, None, :, 3]
    z = xc[..., 2]
    uh = xc[..., :2] / np.where(np.abs(z) < 1e-9, 1e-9, z)[..., None]
    res = np.linalg.norm(uh - U, axis=-1) * f[:, None]
    return np.where(np.isfinite(X[:, 0])[None] & (z > 0), res, np.nan)


def triangulate_views(U: np.ndarray, ok: np.ndarray, Pm: np.ndarray, f: np.ndarray, max_px: float = 10.0) -> tuple[np.ndarray, np.ndarray, float]:
    """3D joints of ONE person seen by V >= 2 views (U (V,17,2) normalised joints, ok (V,17) seen, Pm (V,3,4) world -> normalised camera,
    f (V,) focal px). Per joint the view with the worst reprojection (> max_px) is dropped while more than two views remain.
    Returns (X (17,3), views used per joint (17,), median reprojection px of the kept observations)."""
    ok = ok.copy()
    for _ in range(len(U)):
        X = dlt_joints(U, ok, Pm)
        res = reproj_px(X, Pm, U, f)
        r = np.where(ok, np.where(np.isfinite(res), res, np.inf), 0.0)
        worst, v = r.max(0), r.argmax(0)
        drop = (worst > max_px) & (ok.sum(0) > 2)
        if not drop.any():
            break
        for j in np.flatnonzero(drop):
            ok[v[j], j] = False
    X = dlt_joints(U, ok, Pm)
    res = reproj_px(X, Pm, U, f)
    good = ok & np.isfinite(res)
    nv = good.sum(0)
    X[(nv < 2) | (np.where(good, res, 0.0).max(0) > 2.5 * max_px)] = np.nan
    return X, nv, (float(np.median(res[good])) if good.any() else float("nan"))


@dataclass
class View:
    """The people of one camera at one instant that can take part in a match."""
    cam: str
    i: int                       # canonical frame
    idx: np.ndarray              # row of each kept person in the frame's person list
    P: np.ndarray                # (n,17,3) kept persons
    U: np.ndarray                # (n,17,2) normalised joints
    ok: np.ndarray               # (n,17)
    Pm: np.ndarray               # (3,4)
    f: float
    xh: np.ndarray               # (n,17,3) homogeneous normalised joints
    size: np.ndarray             # (n,) pixel extent


def make_view(cam: Cam, i: int) -> View | None:
    pp = cam.persons(i)
    if not len(pp):
        return None
    keep = (pp[:, :, 2] >= CONF).sum(1) >= MIN_JOINTS
    if not keep.any():
        return None
    idx = np.flatnonzero(keep)
    P = pp[keep]
    U = np.stack([undistort_norm(p[:, :2].astype(float), cam.K[i], float(cam.k1[i])) for p in P])
    ok = P[:, :, 2] >= CONF
    Pm = np.hstack([cam.R[i], cam.tvec[i][:, None]])
    return View(cam.id, i, idx, P, U, ok, Pm, float(cam.K[i][0, 0]),
                np.concatenate([U, np.ones((len(P), 17, 1))], -1), np.array([person_height_px(p) for p in P]))


def pair_cost_matrix(va: View, vb: View, tau: float = 25.0) -> np.ndarray:
    """(Pa,Pb) mean truncated epipolar distance (px) of all shared joints of every person pair (+inf below MIN_JOINTS shared joints)."""
    Ra, ta = va.Pm[:, :3], va.Pm[:, 3]
    Rb, tb = vb.Pm[:, :3], vb.Pm[:, 3]
    Rab = Rb @ Ra.T
    E = skew(tb - Rab @ ta) @ Rab
    la = va.xh @ E.T                                  # epipolar lines of A's joints in B
    lb = vb.xh @ E                                    # epipolar lines of B's joints in A
    na = np.maximum(np.hypot(la[..., 0], la[..., 1]), 1e-12)
    nb = np.maximum(np.hypot(lb[..., 0], lb[..., 1]), 1e-12)
    num = np.abs(np.einsum("qkc,pkc->pqk", vb.xh, la))
    d = np.minimum(0.5 * (num / na[:, None, :] * vb.f + num / nb[None, :, :] * va.f), tau)
    v = va.ok[:, None, :] & vb.ok[None, :, :]
    cnt = v.sum(-1)
    c = (d * v).sum(-1) / np.maximum(cnt, 1)
    return np.where(cnt >= MIN_JOINTS, c, np.inf)


def cluster_views(views: dict[str, View], pair_px: float = 6.0, rel: float = 0.012) -> list[dict[str, int]]:
    """People of different cameras that are the same person: pairwise epipolar costs -> accepted edges (cost <= max(pair_px, rel x the
    pair's pixel size)) -> greedy merge by ascending cost, never two persons of one camera in one cluster. Returns [{cam: row in View.P}]."""
    edges = []
    names = sorted(views)
    for ai in range(len(names)):
        for bi in range(ai + 1, len(names)):
            va, vb = views[names[ai]], views[names[bi]]
            C = pair_cost_matrix(va, vb)
            sz = np.minimum(va.size[:, None], vb.size[None, :])
            thr = np.maximum(pair_px, rel * sz)
            for p, q in zip(*np.nonzero(C <= thr)):
                edges.append((float(C[p, q] / thr[p, q]), names[ai], int(p), names[bi], int(q)))
    edges.sort()
    cl: dict[tuple, set] = {}                              # (cam, row) -> the cluster (a set of (cam, row))
    for _, a, p, b, q in edges:
        ca, cb = cl.get((a, p), {(a, p)}), cl.get((b, q), {(b, q)})
        if ca is cb:
            continue
        if {c for c, _ in ca} & {c for c, _ in cb}:
            continue                                        # one person per camera
        m = ca | cb
        for x in m:
            cl[x] = m
    out, seen = [], set()
    for m in cl.values():
        if id(m) in seen:
            continue
        seen.add(id(m))
        out.append({c: r for c, r in sorted(m)})
    return out


@dataclass
class Cand:
    """One 3D person of one instant."""
    X: np.ndarray                # (17,3) world metres, NaN where not triangulated
    nv: np.ndarray               # (17,) views per joint
    res: float                   # median reprojection px
    views: dict                  # {cam: (canonical frame, row in the frame's person list)}


def centre(X: np.ndarray) -> np.ndarray:
    """Body centre (x, y, z) of a 3D person: mean of shoulders and hips (fallback: every joint)."""
    t = X[[5, 6, 11, 12]]
    if np.isfinite(t[:, 0]).sum() >= 2:
        return np.nanmean(t, axis=0)
    return np.nanmean(X, axis=0) if np.isfinite(X[:, 0]).any() else np.full(3, np.nan)


def instant_candidates(cams: dict[str, Cam], near: dict[str, np.ndarray], k: int, pair_px: float = 6.0, rel: float = 0.012,
                       max_px: float = 10.0, dedupe_m: float = 0.3) -> list[Cand]:
    """Every person that two or more cameras of `cams` see at grid instant k, matched and triangulated (see the module doc)."""
    views = {}
    for c, cam in cams.items():
        i = int(near[c][k])
        if i >= 0:
            v = make_view(cam, i)
            if v is not None:
                views[c] = v
    if len(views) < 2:
        return []
    cands: list[Cand] = []
    for cl in cluster_views(views, pair_px, rel):
        if len(cl) < 2:
            continue
        vs = [views[c] for c in cl]
        U = np.stack([v.U[cl[v.cam]] for v in vs])
        ok = np.stack([v.ok[cl[v.cam]] for v in vs])
        Pm = np.stack([v.Pm for v in vs])
        X, nv, res = triangulate_views(U, ok, Pm, np.array([v.f for v in vs]), max_px)
        if np.isfinite(X[:, 0]).sum() < 6:
            continue
        y = X[:, 1][np.isfinite(X[:, 1])]
        if y.min() < -0.6 or y.max() > 2.8:
            continue                                        # not a person standing in the room
        cands.append(Cand(X, nv, res, {v.cam: (v.i, int(v.idx[cl[v.cam]])) for v in vs}))
    # duplicates: one person that no cross-link joined (cameras of the two clusters do not overlap, centres within dedupe_m)
    merged = True
    while merged and len(cands) > 1:
        merged = False
        cs = [centre(c.X) for c in cands]
        for a in range(len(cands)):
            for b in range(a + 1, len(cands)):
                if np.linalg.norm(cs[a] - cs[b]) <= dedupe_m and not (set(cands[a].views) & set(cands[b].views)):
                    cl = {**cands[a].views, **cands[b].views}
                    vs = [views[c] for c in cl]
                    rows = {c: int(np.flatnonzero(views[c].idx == cl[c][1])[0]) for c in cl}
                    U = np.stack([v.U[rows[v.cam]] for v in vs])
                    ok = np.stack([v.ok[rows[v.cam]] for v in vs])
                    X, nv, res = triangulate_views(U, ok, np.stack([v.Pm for v in vs]), np.array([v.f for v in vs]), max_px)
                    if np.isfinite(X[:, 0]).sum() >= 6:
                        cands[a] = Cand(X, nv, res, cl)
                        del cands[b]
                        merged = True
                        break
            if merged:
                break
    return cands


# ------------------------------------------------------------------------------------------------ the whole timeline


def grid_times() -> np.ndarray:
    return np.arange(K_N) / HZ


@dataclass
class World:
    """3D people of many instants, flattened: instant j has the candidates off[j]..off[j+1]."""
    k: np.ndarray                # (J,) grid indices processed
    off: np.ndarray              # (J+1,)
    X: np.ndarray                # (N,17,3) float32
    nv: np.ndarray               # (N,17) int8
    res: np.ndarray              # (N,) float32
    fi: np.ndarray               # (N, n_cam) int32 canonical frame per camera (-1 not in the cluster)
    row: np.ndarray              # (N, n_cam) int16 person row in the pose table frame
    cams: list

    def cands(self, j: int) -> slice:
        return slice(int(self.off[j]), int(self.off[j + 1]))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, k=self.k, off=self.off, X=self.X, nv=self.nv, res=self.res, fi=self.fi, row=self.row, cams=np.array(self.cams))

    @classmethod
    def load(cls, path: Path) -> "World":
        z = np.load(path)
        return cls(z["k"], z["off"], z["X"], z["nv"], z["res"], z["fi"], z["row"], [str(c) for c in z["cams"]])


def world_people(cams: dict[str, Cam], ks: np.ndarray, log_every: int = 1000, **kw) -> World:
    """instant_candidates for every grid index in ks (the four main cameras)."""
    names = sorted(cams)
    grid = grid_times()
    near = {c: cams[c].near(grid) for c in names}
    X, nv, res, fi, row, off = [], [], [], [], [], [0]
    t0 = time.time()
    for n, k in enumerate(ks):
        for c in instant_candidates(cams, near, int(k), **kw):
            X.append(c.X.astype(np.float32))
            nv.append(c.nv.astype(np.int8))
            res.append(c.res)
            fi.append([c.views[x][0] if x in c.views else -1 for x in names])
            row.append([c.views[x][1] if x in c.views else -1 for x in names])
        off.append(len(X))
        if log_every and (n + 1) % log_every == 0:
            log(f"world people: {n + 1}/{len(ks)} instants, {len(X)} candidates ({time.time() - t0:.0f} s)")
    z = lambda a, shape, dt: np.array(a, dt).reshape(shape)
    return World(np.asarray(ks, int), np.array(off, int), z(X, (-1, 17, 3), np.float32), z(nv, (-1, 17), np.int8),
                 z(res, (-1,), np.float32), z(fi, (-1, len(names)), np.int32), z(row, (-1, len(names)), np.int16), names)


# ------------------------------------------------------------------------------------------------ the two teachers: Viterbi over ordered pairs

MISS = -1


@dataclass
class TeacherCfg:
    """Costs of the pair Viterbi (all in 'nats'-like units; MISS = the slot has no candidate at that instant)."""
    centre0: tuple = (0.25, -0.55)       # first guess of the dance-floor centre (x, z) in metres (the median of the first pass replaces it)
    max_cands: int = 6
    zone_r: float = 4.5                  # candidates further from the centre are never considered
    miss_cost: float = 3.0
    w_views: tuple = (0.0, 0.0, 0.8, 0.3, 0.0)   # by the maximum number of views of the candidate (index = views)
    w_res: float = 0.15                  # per pixel of median reprojection
    w_stand: float = 4.0                 # shoulders below 1.05 m (seated / kneeling): up to this
    w_ankle: float = 1.5                 # lowest ankle above 0.25 m (floating lower body: a mismatch)
    w_joints: float = 0.05               # per missing joint of 17
    w_bone: float = 6.0                  # per missing share of humanly possible bone lengths
    w_zone: float = 1.5                  # pass 1: (distance to the centre / 2.5 m) squared
    w_density: float = 1.2               # pass 2: penalty for a position where pass 1 put no teacher ...
    density_sat: float = 0.15            # ... none from this share of the peak density upwards
    pair_free: float = 3.0               # the partners may stand this far apart for free ...
    pair_w: float = 1.5                  # ... and pay pair_w x excess^2 beyond
    dup_m: float = 0.15                  # two slots on one person
    dup_cost: float = 6.0
    speed_scale: float = 2.5             # m/s: a slot moving this fast between two instants costs 1
    speed_cap: float = 12.0
    switch_miss: float = 1.0             # entering / leaving MISS


BONES = (((5, 7), 0.30), ((6, 8), 0.30), ((7, 9), 0.26), ((8, 10), 0.26), ((11, 13), 0.42), ((12, 14), 0.42), ((13, 15), 0.42),
         ((14, 16), 0.42), ((5, 6), 0.38), ((11, 12), 0.28))      # (joint pair, nominal adult length in m): the same for every adult within +-45 %


def bone_score(X: np.ndarray) -> float:
    """Share of the measurable bones of a triangulated person whose length is humanly possible (0.55 .. 1.45 x nominal), plus the torso.
    A person assembled from two different people (a wrong match) has bones of any length."""
    ok = tot = 0
    for (a, b), L in BONES:
        d = np.linalg.norm(X[a] - X[b])
        if np.isfinite(d):
            tot += 1
            ok += int(0.55 * L <= d <= 1.45 * L)
    sh, hp = X[[5, 6]], X[[11, 12]]
    if np.isfinite(sh).all() and np.isfinite(hp).all():
        tot += 1
        ok += int(0.30 <= np.linalg.norm(sh.mean(0) - hp.mean(0)) <= 0.80)
    return ok / tot if tot >= 4 else 0.5


def cand_features(W: World) -> dict:
    """Per candidate: centre xz, shoulder / hip / ankle heights, max views, joints, residual, bone plausibility."""
    cen = np.array([centre(x) for x in W.X])
    with np.errstate(all="ignore"):
        sh = np.nanmean(W.X[:, [5, 6], 1], axis=1)
        hip = np.nanmean(W.X[:, [11, 12], 1], axis=1)
        ank = np.nanmin(np.where(np.isfinite(W.X[:, [15, 16], 1]), W.X[:, [15, 16], 1], np.inf), axis=1)
    ank = np.where(np.isfinite(ank), ank, np.nan)
    return {"cen": cen, "sh": sh, "hip": hip, "ank": ank, "nvmax": W.nv.max(1), "nj": np.isfinite(W.X[:, :, 0]).sum(1), "res": W.res,
            "bone": np.array([bone_score(x) for x in W.X])}


def unary_cost(f: dict, sl: slice, cfg: TeacherCfg, centre0, density=None) -> np.ndarray:
    """Cost of each candidate of an instant being a teacher (lower = more teacher-like)."""
    cen = f["cen"][sl]
    d = np.hypot(cen[:, 0] - centre0[0], cen[:, 2] - centre0[1])
    sh = np.nan_to_num(f["sh"][sl], nan=0.7)
    hip = np.nan_to_num(f["hip"][sl], nan=0.2)
    stand = np.clip((1.05 - np.maximum(sh, hip + 0.45)) / 0.2, 0, 1)
    ank = np.nan_to_num(f["ank"][sl], nan=0.1)
    u = (np.array([cfg.w_views[min(int(v), 4)] for v in f["nvmax"][sl]]) + cfg.w_res * np.nan_to_num(f["res"][sl], nan=5.0)
         + cfg.w_bone * (1.0 - f["bone"][sl])
         + cfg.w_stand * stand + cfg.w_ankle * np.clip((ank - 0.25) / 0.3, 0, 1) + cfg.w_joints * (17 - f["nj"][sl]))
    if density is None:
        u = u + cfg.w_zone * (d / 2.5) ** 2
    else:
        u = u + cfg.w_density * (1.0 - np.clip(density(cen[:, 0], cen[:, 2]) / cfg.density_sat, 0.0, 1.0))
    return np.where(d > cfg.zone_r, np.inf, u)


def pair_viterbi(W: World, cfg: TeacherCfg | None = None, centre0=None, density=None, dt: float | None = None) -> dict:
    """The two teachers as two continuous tracks of 3D candidates. Returns {"idx": (J,2) candidate row in W (MISS = -1), "cost"}."""
    cfg = cfg or TeacherCfg()
    centre0 = centre0 if centre0 is not None else cfg.centre0
    f = cand_features(W)
    J = len(W.k)
    step_dt = dt if dt is not None else float(np.median(np.diff(W.k))) / HZ if J > 1 else 1 / HZ
    # candidate lists per instant (best max_cands by unary)
    lists, unary = [], []
    for j in range(J):
        sl = W.cands(j)
        u = unary_cost(f, sl, cfg, centre0, density)
        order = np.argsort(u)[:cfg.max_cands]
        order = order[np.isfinite(u[order])]
        lists.append(np.arange(sl.start, sl.stop)[order])
        unary.append(np.concatenate([u[order], [cfg.miss_cost]]))
    xz = f["cen"][:, [0, 2]]

    def node(j):
        c = lists[j]
        n = len(c)
        u = unary[j]
        M = u[:, None] + u[None, :]
        if n:
            d = np.linalg.norm(xz[c][:, None, :] - xz[c][None, :, :], axis=-1)
            pc = cfg.pair_w * np.clip(d - cfg.pair_free, 0, None) ** 2 + np.where(d < cfg.dup_m, cfg.dup_cost, 0.0)
            M[:n, :n] += pc
            M[np.arange(n), np.arange(n)] = np.inf          # one candidate cannot fill both slots
        return M

    def trans(j0, j1):
        a, b = lists[j0], lists[j1]
        T = np.zeros((len(a) + 1, len(b) + 1))
        if len(a) and len(b):
            d = np.linalg.norm(xz[a][:, None, :] - xz[b][None, :, :], axis=-1)
            T[:len(a), :len(b)] = np.minimum((d / step_dt / cfg.speed_scale) ** 2, cfg.speed_cap)
        T[:len(a), len(b)] = cfg.switch_miss
        T[len(a), :len(b)] = cfg.switch_miss
        return T

    V = node(0)
    back = []
    for j in range(1, J):
        T = trans(j - 1, j)
        # slot A: M[a', b] = min_a V[a, b] + T[a, a'] ; slot B: N[a', b'] = min_b M[a', b] + T[b, b']
        S1 = V[:, :, None] + T[:, None, :]                 # (a, b, a')
        argA = S1.argmin(0)                                # (b, a')
        M = np.take_along_axis(S1, argA[None], 0)[0].T     # (a', b)
        S2 = M[:, :, None] + T[None, :, :]                 # (a', b, b')
        argB = S2.argmin(1)                                # (a', b')
        N = np.take_along_axis(S2, argB[:, None, :], 1)[:, 0, :]
        V = N + node(j)
        back.append((argA, argB))
    a, b = np.unravel_index(int(np.argmin(V)), V.shape)
    total = float(V[a, b])
    path = [(int(a), int(b))]
    for j in range(J - 1, 0, -1):
        argA, argB = back[j - 1]
        bp = int(argB[a, b])                                # slot B's previous state
        ap = int(argA[bp, a])                               # slot A's previous state
        a, b = ap, bp
        path.append((a, b))
    path.reverse()
    idx = np.full((J, 2), MISS, int)
    for j, (a, b) in enumerate(path):
        n = len(lists[j])
        idx[j, 0] = lists[j][a] if a < n else MISS
        idx[j, 1] = lists[j][b] if b < n else MISS
    return {"idx": idx, "cost": total, "lists": [len(x) for x in lists]}


def kde_density(xz: np.ndarray, cell: float = 0.1, sigma: float = 0.4, extent: float = 8.0):
    """Smoothed floor-plane density of points xz (N,2), normalised to a peak of 1: returns f(x, z) (bilinear lookup, 0 outside)."""
    from scipy.ndimage import gaussian_filter
    n = int(2 * extent / cell)
    H, _, _ = np.histogram2d(xz[:, 0], xz[:, 1], bins=n, range=[[-extent, extent], [-extent, extent]])
    H = gaussian_filter(H, sigma / cell)
    H = H / max(H.max(), 1e-9)

    def f(x, z):
        gx = (np.asarray(x) + extent) / cell - 0.5
        gz = (np.asarray(z) + extent) / cell - 0.5
        x0, z0 = np.floor(gx).astype(int), np.floor(gz).astype(int)
        wx, wz = gx - x0, gz - z0
        def g(a, b):
            ok = (a >= 0) & (a < n) & (b >= 0) & (b < n)
            return np.where(ok, H[np.clip(a, 0, n - 1), np.clip(b, 0, n - 1)], 0.0)
        return (g(x0, z0) * (1 - wx) * (1 - wz) + g(x0 + 1, z0) * wx * (1 - wz) + g(x0, z0 + 1) * (1 - wx) * wz + g(x0 + 1, z0 + 1) * wx * wz)
    f.H, f.cell, f.extent = H, cell, extent
    return f


@dataclass
class Track3D:
    """The two anonymous teacher tracks (slots A, B) on the processed instants: X (J,2,17,3) NaN where the slot has no candidate, `obs` (J,2)
    candidate row (-1 missing), `n_views` (J,2)."""
    k: np.ndarray
    obs: np.ndarray
    X: np.ndarray
    nv: np.ndarray
    res: np.ndarray
    centre0: tuple

    def save(self, path: Path) -> None:
        np.savez_compressed(path, k=self.k, obs=self.obs, X=self.X, nv=self.nv, res=self.res, centre0=np.array(self.centre0))

    @classmethod
    def load(cls, path: Path) -> "Track3D":
        z = np.load(path)
        return cls(z["k"], z["obs"], z["X"], z["nv"], z["res"], tuple(z["centre0"]))

    def at(self, s: int, t: float, max_gap_s: float = 1.0, snap_s: float = 0.05):
        """3D joints (17,3) of slot s at time t (s): the nearest observed instant when within snap_s, else a linear interpolation
        between the observed instants around t when both lie within max_gap_s. Returns (X or None, kind) kind observed | interp | none."""
        tk = self.k / HZ
        ok = np.isfinite(self.X[:, s, 0, 0]) | np.isfinite(self.X[:, s, 5, 0])
        if not ok.any():
            return None, "none"
        idx = np.flatnonzero(ok)
        pos = np.searchsorted(tk[idx], t)
        lo = idx[pos - 1] if pos > 0 else None
        hi = idx[pos] if pos < len(idx) else None
        for j in (lo, hi):
            if j is not None and abs(tk[j] - t) <= snap_s:
                return self.X[j, s], "observed"
        if lo is None or hi is None or tk[hi] - tk[lo] > 2 * max_gap_s or t - tk[lo] > max_gap_s or tk[hi] - t > max_gap_s:
            return None, "none"
        w = (t - tk[lo]) / (tk[hi] - tk[lo])
        a, b = self.X[lo, s], self.X[hi, s]
        X = np.where(np.isfinite(a) & np.isfinite(b), (1 - w) * a + w * b, np.where(np.isfinite(a), a, b))
        return X, "interp"


def teacher_tracks(W: World, cfg: TeacherCfg | None = None, dt: float | None = None, log_fn=log) -> Track3D:
    """Two-pass pair Viterbi (see TeacherCfg). Pass 1 uses a zone around the first-guess centre; pass 2 a floor-plane density of the
    pass-1 teachers, so standing spectators at the edge of the floor do not pass for a teacher when one teacher is hidden."""
    cfg = cfg or TeacherCfg()
    f = cand_features(W)
    r1 = pair_viterbi(W, cfg, cfg.centre0, None, dt)
    sel = r1["idx"][r1["idx"] >= 0]
    xz = f["cen"][sel][:, [0, 2]]
    c1 = tuple(np.median(xz, axis=0))
    dens = kde_density(xz)
    log_fn(f"teachers pass 1: {len(sel)} instants-slots, centre {np.round(c1, 2).tolist()} (first guess {cfg.centre0})")
    r2 = pair_viterbi(W, cfg, c1, dens, dt)
    idx = r2["idx"]
    J = len(W.k)
    X = np.full((J, 2, 17, 3), np.nan, np.float32)
    nv = np.zeros((J, 2), np.int8)
    res = np.full((J, 2), np.nan, np.float32)
    for s in range(2):
        ok = idx[:, s] >= 0
        X[ok, s] = W.X[idx[ok, s]]
        nv[ok, s] = W.nv[idx[ok, s]].max(1)
        res[ok, s] = W.res[idx[ok, s]]
    both = (idx >= 0).all(1).mean()
    log_fn(f"teachers pass 2: both present at {100 * both:.1f} % of {J} instants, one at {100 * ((idx >= 0).any(1).mean() - both):.1f} %")
    t = Track3D(W.k.copy(), idx.astype(np.int32), X, nv, res, c1)
    t.density = dens
    return t


# ------------------------------------------------------------------------------------------------ the teachers in every camera


@dataclass
class Assoc:
    """Which JSON person of each canonical frame of one camera is teacher slot A / B (anonymous slots; roles come with the appearance)."""
    cam: str
    row: np.ndarray              # (F,2) int16 row in the frame's person list, -1 none
    cost: np.ndarray             # (F,2) float32 median joint distance / person size
    kind: np.ndarray             # (F,2) int8 0 none, 1 member of the 3D candidate of that instant, 2 projected observed 3D, 3 projected interpolated 3D,
                                 # 4 followed in 2D from the last 3D-supported frame (fill_by_2d_continuity)
    nj: np.ndarray               # (F,2) int8 joints compared
    gap: np.ndarray = None       # (F,2) int16 kind 4 only: frames since the last association that the 3D track supported
    inside: np.ndarray = None    # (F,2) float16 share of the OTHER teacher's projected joints that lie inside this person's box (1 = the partner is hidden in it)

    def __post_init__(self):
        if self.gap is None:
            self.gap = np.zeros(self.row.shape, np.int16)
        if self.inside is None:
            self.inside = np.zeros(self.row.shape, np.float16)

    def merged(self) -> np.ndarray:
        """(F,2) True where this slot has a person, the OTHER slot has none, and that person's box holds the partner: one detection for two bodies,
        so its colours are a mixture and its role is a guess."""
        other_missing = self.row[:, ::-1] < 0
        return (self.row >= 0) & other_missing & (self.inside.astype(float) > 0.5)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, cam=self.cam, row=self.row, cost=self.cost, kind=self.kind, nj=self.nj, gap=self.gap, inside=self.inside)

    @classmethod
    def load(cls, path: Path) -> "Assoc":
        z = np.load(path)
        return cls(str(z["cam"]), z["row"], z["cost"], z["kind"], z["nj"], z["gap"] if "gap" in z.files else None, z["inside"] if "inside" in z.files else None)


def members_of(W: World, tr: Track3D, cam: str) -> dict:
    """{(canonical frame, row): slot} for the persons of `cam` that belong to the selected 3D candidate of a slot."""
    if cam not in W.cams:
        return {}
    c = W.cams.index(cam)
    out = {}
    for s in range(2):
        for j in np.flatnonzero(tr.obs[:, s] >= 0):
            r = int(tr.obs[j, s])
            if W.fi[r, c] >= 0:
                out[(int(W.fi[r, c]), int(W.row[r, c]))] = s
    return out


def associate(cam: Cam, tr: Track3D, members: dict | None = None, gate: float | None = None, min_joints: int = 5,
              conf: float = 0.3) -> Assoc:
    """Project both 3D teacher tracks into every canonical frame of `cam` and match them to the frame's JSON persons (Hungarian on the
    median joint distance relative to the person's size; a person may serve one slot only). The attached phones 05 / 06 have a pose
    good to a few degrees and 0.3 m: a looser gate."""
    from scipy.optimize import linear_sum_assignment
    gate = gate if gate is not None else (0.45 if cam.attached else 0.22)
    F = len(cam.t)
    row = np.full((F, 2), -1, np.int16)
    cost = np.full((F, 2), np.nan, np.float32)
    kind = np.zeros((F, 2), np.int8)
    nj = np.zeros((F, 2), np.int8)
    inside = np.zeros((F, 2), np.float16)
    w, h = cam.size
    for i in range(F):
        if not cam.ok[i]:
            continue
        P = cam.persons(i)
        if not len(P):
            continue
        size = np.array([person_height_px(p) for p in P])
        okp = np.isfinite(size) & (size > 20)
        if not okp.any():
            continue
        C = np.full((2, len(P)), np.inf)
        kinds = [None, None]
        pred = [None, None]
        for s in range(2):
            X, kd = tr.at(s, float(cam.t[i]))
            if X is None:
                continue
            uv, z = project(X, cam.R[i], cam.tvec[i], cam.K[i], float(cam.k1[i]))
            uv = np.where((z > 0.2)[:, None] & np.isfinite(uv), uv, np.nan)
            pred[s], kinds[s] = uv, kd
            for p in np.flatnonzero(okp):
                v = (P[p, :, 2] >= conf) & np.isfinite(uv[:, 0])
                if v.sum() < min_joints:
                    continue
                d = np.linalg.norm(uv[v] - P[p, v, :2], axis=1)
                C[s, p] = np.median(d) / max(size[p], 40.0)
        if not np.isfinite(C).any():
            continue
        Cg = np.where(C <= gate, C, 1e3)
        r, c = linear_sum_assignment(Cg)
        for s, p in zip(r, c):
            if Cg[s, p] >= 1e3:
                continue
            row[i, s], cost[i, s] = p, C[s, p]
            v = (P[p, :, 2] >= conf) & np.isfinite(pred[s][:, 0])
            nj[i, s] = int(v.sum())
            kind[i, s] = 1 if (members and members.get((i, int(p))) == s) else (2 if kinds[s] == "observed" else 3)
        for s in range(2):                                     # is the partner's projected body inside this person's box?
            o = pred[1 - s]
            if row[i, s] >= 0 and o is not None:
                b = person_box(P[row[i, s]])
                ok_o = np.isfinite(o[:, 0]) & np.isfinite(b).all()
                if ok_o.sum() >= 5:
                    inside[i, s] = float(((o[ok_o, 0] >= b[0]) & (o[ok_o, 0] <= b[2]) & (o[ok_o, 1] >= b[1]) & (o[ok_o, 1] <= b[3])).mean())
    return Assoc(cam.id, row, cost, kind, nj, None, inside)


# ------------------------------------------------------------------------------------------------ pictures


def decode_frame(cam: str, pts: float, width: int = 640) -> np.ndarray | None:
    """One frame of a source video at presentation time `pts` (s), display orientation (ffmpeg autorotate), scaled to `width` px wide, BGR."""
    import subprocess
    row = cams_table()[cam]
    w0, h0 = int(row["width"]), int(row["height"])
    h = int(round(width * h0 / w0 / 2) * 2)
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{max(pts, 0):.4f}", "-i", str(video_path(cam)), "-frames:v", "1",
           "-vf", f"scale={width}:{h}:flags=area", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    out = subprocess.run(cmd, capture_output=True, timeout=120).stdout
    if len(out) != width * h * 3:
        return None
    return np.frombuffer(out, np.uint8).reshape(h, width, 3).copy()


def draw_person(img: np.ndarray, p: np.ndarray, scale: float, color, thick: int = 2, conf: float = 0.3, box: bool = True, label: str | None = None) -> None:
    """COCO-17 skeleton + box of one JSON person on an image scaled by `scale` from the display pixels."""
    import cv2
    LIMBS = ((5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (11, 12), (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6))
    q = p[:, :2] * scale
    ok = p[:, 2] >= conf
    for a, b in LIMBS:
        if ok[a] and ok[b]:
            cv2.line(img, tuple(np.round(q[a]).astype(int)), tuple(np.round(q[b]).astype(int)), color, thick, cv2.LINE_AA)
    for j in np.flatnonzero(ok):
        cv2.circle(img, tuple(np.round(q[j]).astype(int)), max(2, thick), color, -1, cv2.LINE_AA)
    b = person_box(p, conf) * scale
    if box and np.isfinite(b).all():
        cv2.rectangle(img, tuple(np.round(b[:2]).astype(int)), tuple(np.round(b[2:]).astype(int)), color, 1, cv2.LINE_AA)
        if label:
            cv2.putText(img, label, (int(b[0]), max(12, int(b[1]) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


SLOT_COLOURS = {0: (0, 140, 255), 1: (255, 200, 0)}          # BGR: A orange, B cyan (anonymous slots)
ROLE_COLOURS = {"lead": (60, 60, 255), "follow": (60, 220, 60)}   # BGR: Kadu red, Larissa green


def review_tile(cam: Cam, i: int, rows: dict, labels: dict, colours: dict, width: int = 420, others: bool = True) -> np.ndarray | None:
    """One picture: frame i of `cam` with the selected teachers drawn ({key: row in the person list or -1}, labels / colours per key) and,
    thin and grey, every other person of the JSON."""
    import cv2
    img = decode_frame(cam.id, float(cam.pts[i]), width)
    if img is None:
        return None
    sc = width / cam.size[0]
    P = cam.persons(i)
    chosen = {r for r in rows.values() if r is not None and r >= 0}
    if others:
        for j, p in enumerate(P):
            if j not in chosen and (p[:, 2] >= 0.3).sum() >= 4:
                draw_person(img, p, sc, (150, 150, 150), 1, box=False)
    for key, r in rows.items():
        if r is not None and r >= 0 and r < len(P):
            draw_person(img, P[r], sc, colours[key], 2, label=labels.get(key))
    cv2.putText(img, f"{cam.id} t={cam.t[i]:.1f}s", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    return img


def sheet(tiles: list, cols: int, pad: int = 4) -> np.ndarray:
    tiles = [t for t in tiles if t is not None]
    h = max(t.shape[0] for t in tiles)
    w = max(t.shape[1] for t in tiles)
    rows = -(-len(tiles) // cols)
    out = np.full((rows * (h + pad), cols * (w + pad), 3), 30, np.uint8)
    for n, t in enumerate(tiles):
        r, c = divmod(n, cols)
        out[r * (h + pad):r * (h + pad) + t.shape[0], c * (w + pad):c * (w + pad) + t.shape[1]] = t
    return out


def fill_by_2d_continuity(cam: Cam, A: Assoc, max_chain_s: float = 2.0, gate: float = 0.12, conf: float = 0.3, min_joints: int = 5) -> Assoc:
    """Frames where the 3D track gave no association (an embrace the pose model could not separate in two cameras, a gap longer than the track
    interpolates) are bridged in 2D: the person of the previous (and, in a second sweep, the next) 3D-supported frame is followed to the person of the
    frame whose joints agree best (median joint distance / size <= gate), both slots at once (Hungarian), for at most max_chain_s from the last
    3D-supported frame. kind 4, `gap` = frames since that anchor (its confidence decays with it)."""
    from scipy.optimize import linear_sum_assignment
    F = len(cam.t)
    row, cost, kind, nj, gap = A.row.copy(), A.cost.copy(), A.kind.copy(), A.nj.copy(), A.gap.copy()
    max_chain = int(round(max_chain_s * 30))
    for order in (range(F), range(F - 1, -1, -1)):
        last = [None, None]                                   # (joints (17,3), frame of the last 3D-supported association)
        for i in order:
            P = cam.persons(i) if cam.ok[i] else np.zeros((0, 17, 3), np.float32)
            for s in range(2):
                if 0 <= row[i, s] < len(P) and kind[i, s] in (1, 2, 3):
                    last[s] = [P[row[i, s]], i, i]
            need = [s for s in range(2) if row[i, s] < 0 and last[s] is not None and abs(i - last[s][2]) <= max_chain]
            if not need or not len(P):
                continue
            used = {int(row[i, s]) for s in range(2) if row[i, s] >= 0}
            cand = [p for p in range(len(P)) if p not in used]
            if not cand:
                continue
            C = np.full((len(need), len(cand)), 1e3)
            for a, s in enumerate(need):
                for b, p in enumerate(cand):
                    c, _, n = person_cost(last[s][0][:, :2], P[p], conf, min_joints)
                    if c <= gate:
                        C[a, b] = c
            r, c = linear_sum_assignment(C)
            for a, b in zip(r, c):
                if C[a, b] >= 1e3:
                    continue
                s, p = need[a], cand[b]
                row[i, s], cost[i, s], kind[i, s] = p, C[a, b], 4
                nj[i, s] = int(((P[p, :, 2] >= conf) & np.isfinite(last[s][0][:, 0])).sum())
                gap[i, s] = abs(i - last[s][2])
                last[s][0] = P[p]                              # follow the person from frame to frame; the anchor frame stays the 3D-supported one
    return Assoc(A.cam, row, cost, kind, nj, gap, A.inside.copy())


# ------------------------------------------------------------------------------------------------ appearance (colour evidence)


def stream_frames(cam: str, width: int, last: int | None = None):
    """Yield (video frame index n, BGR frame scaled to `width` px wide) of a whole video in presentation order, display orientation, no
    frame duplicated or dropped (n counts like the rows of the pose JSON). Stops after frame `last`."""
    import subprocess
    from .ingest import passthrough_args
    row = cams_table()[cam]
    w0, h0 = int(row["width"]), int(row["height"])
    h = int(round(width * h0 / w0 / 2) * 2)
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-threads", "4", "-i", str(video_path(cam)), "-map", "0:v:0", "-vf", f"scale={width}:{h}:flags=area",
           *passthrough_args(), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=1 << 24)
    fb = width * h * 3
    n = 0
    try:
        while last is None or n <= last:
            buf = proc.stdout.read(fb)
            if len(buf) < fb:
                break
            yield n, np.frombuffer(buf, np.uint8).reshape(h, width, 3)
            n += 1
    finally:
        proc.kill()
        proc.wait()


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return 0.0
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / u) if u > 0 else 0.0


DESC_KEYS = ("torso_v", "torso_s", "leg_v", "leg_s", "red", "torso_n", "legs_n", "head_n")


def appearance_camera(cam: Cam, A: Assoc, step: int = 3, width: int | None = None, t_range: tuple | None = None, log_fn=log) -> dict:
    """Colour descriptors (dancers_appearance: torso / legs brightness, red-hair share) of the associated teacher persons of every
    `step`-th canonical frame, measured on the decoded pixels. Returns {i (N,), desc (N,2,len(DESC_KEYS)), iou (N,)}."""
    import cv2
    from .dancers_appearance import descriptors
    w0 = cam.size[0]
    width = width or min(w0, 720)
    sc = width / w0
    want = {}
    merged = A.merged()
    for i in range(0, len(cam.t), step):
        if not cam.ok[i] or not (A.row[i] >= 0).any():
            continue
        if t_range and not (t_range[0] <= cam.t[i] <= t_range[1]):
            continue
        want[int(cam.fi[i])] = i
    if not want:
        return {"i": np.zeros(0, int), "desc": np.zeros((0, 2, len(DESC_KEYS)), np.float32), "iou": np.zeros(0, np.float32), "merged": np.zeros((0, 2), bool)}
    out_i, out_d, out_iou, out_in = [], [], [], []
    t0 = time.time()
    first = min(want)
    for n, img in stream_frames(cam.id, width, last=max(want)):
        if n not in want:
            continue
        i = want[n]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        P = cam.persons(i)
        d = np.full((2, len(DESC_KEYS)), np.nan, np.float32)
        boxes = []
        for s in range(2):
            r = int(A.row[i, s])
            if r < 0 or r >= len(P):
                boxes.append(np.full(4, np.nan))
                continue
            p = P[r].copy()
            p[:, :2] *= sc
            b = person_box(p)
            boxes.append(b)
            if not np.isfinite(b).all():
                continue
            dd = descriptors(hsv, b, p.astype(np.float32))
            d[s] = [dd[k] if k in dd else np.nan for k in DESC_KEYS]
        out_i.append(i)
        out_d.append(d)
        out_iou.append(box_iou(boxes[0], boxes[1]))
        out_in.append(merged[i])
    log_fn(f"appearance {cam.id}: {len(out_i)} frames ({time.time() - t0:.0f} s)")
    return {"i": np.array(out_i, int), "desc": np.array(out_d, np.float32).reshape(-1, 2, len(DESC_KEYS)), "iou": np.array(out_iou, np.float32),
            "merged": np.array(out_in, bool).reshape(-1, 2)}


# ------------------------------------------------------------------------------------------------ roles: who is Kadu, who is Larissa

FEAT = (("torso_v", "torso_n", 150), ("leg_v", "legs_n", 150), ("red", "head_n", 300))     # (feature, pixel-count key, minimum pixels)
FEAT_W = {"torso_v": 1.0, "leg_v": 0.5, "red": 1.0}
CLASS0 = {"lead": {"torso_v": (40.0, 45.0), "leg_v": (90.0, 35.0), "red": (0.0, 0.15)},        # (mean, sd): Kadu = black top, blue jeans, no red hair
          "follow": {"torso_v": (170.0, 45.0), "leg_v": (150.0, 35.0), "red": (0.45, 0.15)}}   # Larissa = white top, light jeans, red hair


def _feat_matrix(desc: np.ndarray) -> dict:
    """{feature: values (N,2) NaN where the cue abstains (too few pixels)} of the observations desc (N,2,len(DESC_KEYS))."""
    out = {}
    for f, nk, nmin in FEAT:
        v = desc[:, :, DESC_KEYS.index(f)].astype(float).copy()
        n = desc[:, :, DESC_KEYS.index(nk)]
        v[~(n >= nmin)] = np.nan
        out[f] = v
    return out


def llr_obs(feats: dict, model: dict, clip: float = 2.0) -> np.ndarray:
    """(N,2) log-odds that the person of each slot is the LEAD (Gaussian class models per feature, each cue clipped, weights FEAT_W)."""
    out = np.zeros(next(iter(feats.values())).shape)
    for f, x in feats.items():
        (ml, sl), (mf, sf) = model["lead"][f], model["follow"][f]
        z = (-0.5 * ((x - ml) / sl) ** 2 - np.log(sl)) - (-0.5 * ((x - mf) / sf) ** 2 - np.log(sf))
        out += FEAT_W[f] * np.clip(np.nan_to_num(z, nan=0.0), -clip, clip)
    return out


def fit_class_model(feats: dict, llr0: np.ndarray, conf: float = 2.0) -> dict:
    """Per-camera Gaussian class models: start from CLASS0, refit on the observations the current model calls with |llr| > conf."""
    model = {r: {f: tuple(v) for f, v in d.items()} for r, d in CLASS0.items()}
    for _ in range(3):
        llr = llr_obs(feats, model)
        for role, sel in (("lead", llr > conf), ("follow", llr < -conf)):
            for f, x in feats.items():
                v = x[sel]
                v = v[np.isfinite(v)]
                if len(v) >= 30:
                    med = float(np.median(v))
                    mad = max(1.4826 * float(np.median(np.abs(v - med))), 0.5 * CLASS0[role][f][1])
                    model[role][f] = (med, min(mad * 1.2, 2.0 * CLASS0[role][f][1]))
    return model


def role_hmm(e: np.ndarray, switch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two states (0: slot A is the lead, 1: slot B is the lead) over time bins. e (B,) summed log-odds that A is the lead; switch (B,) cost
    of changing state between bin b-1 and b. Returns (viterbi state (B,), posterior P(A is the lead) (B,))."""
    B = len(e)
    nll = np.stack([np.logaddexp(0, -e), np.logaddexp(0, e)], 1)
    V = nll[0].copy()
    back = np.zeros((B, 2), int)
    fwd = np.zeros((B, 2))
    fwd[0] = -nll[0]
    for b in range(1, B):
        s = switch[b]
        stay, sw = V, V[::-1] + s
        back[b] = np.where(sw < stay, 1 - np.arange(2), np.arange(2))
        V = np.minimum(stay, sw) + nll[b]
        lf = fwd[b - 1]
        fwd[b] = -nll[b] + np.logaddexp(lf, lf[::-1] - s)
    st = np.zeros(B, int)
    st[-1] = int(np.argmin(V))
    for b in range(B - 1, 0, -1):
        st[b - 1] = back[b, st[b]]
    bwd = np.zeros((B, 2))
    for b in range(B - 2, -1, -1):
        s = switch[b + 1]
        nxt = bwd[b + 1] - nll[b + 1]
        bwd[b] = np.logaddexp(nxt, nxt[::-1] - s)
    lp = fwd + bwd
    p0 = np.exp(lp[:, 0] - np.logaddexp(lp[:, 0], lp[:, 1]))
    return st, p0


BIN_S = 1.0 / 30.0                    # the role decision has the resolution of one 30 Hz frame (a 0.1 s bin mislabelled the 2-3 frames around a swap)


@dataclass
class Roles:
    """Which slot is the lead (Kadu): state (B,) 0 = A, 1 = B per BIN_S bin of the master clock, posterior p_a_lead (B,), evidence e (B,) and
    the per-camera class models."""
    state: np.ndarray
    p_a_lead: np.ndarray
    e: np.ndarray
    models: dict
    n_obs: np.ndarray            # (B,) observations that fed the bin
    switches: list

    def lead_slot(self, t: float) -> int:
        b = int(np.clip(round(t / BIN_S), 0, len(self.state) - 1))
        return int(self.state[b])

    def conf(self, t: float) -> float:
        """Posterior probability that the lead / follow decision at t is right."""
        b = int(np.clip(round(t / BIN_S), 0, len(self.state) - 1))
        p = float(self.p_a_lead[b])
        return p if self.state[b] == 0 else 1.0 - p


def separation_per_bin(tr: Track3D, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """(centre distance of the two slots per bin (NaN if one is missing), both-observed flag)."""
    sep = np.full(n_bins, np.nan)
    c = np.full((len(tr.k), 2, 3), np.nan)
    for s in range(2):
        for j in np.flatnonzero(np.isfinite(tr.X[:, s, 5, 0])):
            c[j, s] = centre(tr.X[j, s])
    d = np.linalg.norm(c[:, 0] - c[:, 1], axis=-1)
    b = np.clip(np.round(tr.k / HZ / BIN_S).astype(int), 0, n_bins - 1)
    for j in np.flatnonzero(np.isfinite(d)):
        sep[b[j]] = d[j] if not np.isfinite(sep[b[j]]) else min(sep[b[j]], d[j])
    return sep, np.isfinite(sep)


def roles_from_appearance(tr: Track3D, apps: dict, cams: dict, switch_near: float = 12.0, switch_far: float = 40.0, far_m: float = 1.0) -> Roles:
    """Evidence of every camera at every sampled frame -> one two-state keep / swap model over the whole class (see the module doc)."""
    n_bins = int(np.ceil(WINDOW[1] / BIN_S)) + 1
    e = np.zeros(n_bins)
    nobs = np.zeros(n_bins, int)
    models = {}
    for c, a in apps.items():
        if not len(a["i"]):
            continue
        feats = _feat_matrix(a["desc"])
        models[c] = fit_class_model(feats, None)
        llr = llr_obs(feats, models[c])                         # (N,2): slot is the lead
        has = np.isfinite(a["desc"][:, :, 0])
        mg = a["merged"] if "merged" in a else np.zeros(has.shape, bool)           # one detection for two bodies: its colours are a mixture
        wm = np.where(mg, 0.25, 1.0)
        ev = np.where(has[:, 0], llr[:, 0] * wm[:, 0], 0.0) - np.where(has[:, 1], llr[:, 1] * wm[:, 1], 0.0)
        w = np.clip(1.0 - np.clip(a["iou"] - 0.3, 0, None) / 0.5, 0.3, 1.0)                # overlapping boxes: the pixels are mixed
        ev = np.clip(ev * w, -4.0, 4.0)
        b = np.clip(np.round(cams[c].t[a["i"]] / BIN_S).astype(int), 0, n_bins - 1)
        np.add.at(e, b, ev)
        np.add.at(nobs, b, (has[:, 0] | has[:, 1]).astype(int))
    sep, both = separation_per_bin(tr, n_bins)
    switch = np.where(both & (np.nan_to_num(sep, nan=0.0) > far_m), switch_far, switch_near)
    st, p = role_hmm(e, switch)
    sw = [(round(float(b * BIN_S), 1), int(st[b])) for b in np.flatnonzero(np.diff(st) != 0) + 1]
    return Roles(st, p, e, models, nobs, sw)


# ------------------------------------------------------------------------------------------------ outputs: teacher tracks per camera


KIND_FACTOR = {0: 0.0, 1: 1.0, 2: 0.9, 3: 0.7, 4: 0.6}


def teacher_conf(cost: np.ndarray, kind: np.ndarray, nj: np.ndarray, role_conf: np.ndarray, attached: bool, gap: np.ndarray | None = None,
                 merged: np.ndarray | None = None) -> np.ndarray:
    """Confidence (0..1) of an association: how it was found, how well the joints agree with the 3D track, how many joints were compared,
    and how sure the role (Kadu / Larissa) decision is. An attached phone (rotation-only pose) counts 0.9 with a looser cost scale."""
    kf = np.vectorize(lambda k: KIND_FACTOR[int(k)])(kind)
    scale = 0.2 if attached else 0.12                     # an attached phone's pose is good to a few degrees: its joints sit further from the 3D track
    c = kf * np.exp(-(np.nan_to_num(cost, nan=9.0) / scale) ** 2) * np.clip(nj / 8.0, 0, 1) * role_conf
    if gap is not None:                                    # followed in 2D: the longer since the last 3D-supported frame, the less sure
        c = c * np.where(kind == 4, np.clip(1.0 - gap / 90.0, 0.4, 1.0), 1.0)
    if merged is not None:                                 # one detection for two bodies: which of the two it follows is a guess
        c = c * np.where(merged, 0.5, 1.0)
    return np.where(kind > 0, c, 0.0) * (0.9 if attached else 1.0)


def slot_of_role(lead_slot: np.ndarray, role: int) -> np.ndarray:
    """Slot (0 = A, 1 = B) of role 0 (lead) / 1 (follow) per frame, given which slot is the lead."""
    return np.where(lead_slot == 0, role, 1 - role)


def teachers_for_camera(cam: Cam, A: Assoc, roles: Roles) -> dict:
    """Per canonical frame of one camera: the JSON person of Kadu (lead, index 0) and Larissa (follow, index 1) with box, joints, confidence."""
    F = len(cam.t)
    row = np.full((F, 2), -1, np.int16)
    box = np.full((F, 2, 4), np.nan, np.float32)
    kp = np.full((F, 2, 17, 3), np.nan, np.float32)
    cost = np.full((F, 2), np.nan, np.float32)
    kind = np.zeros((F, 2), np.int8)
    nj = np.zeros((F, 2), np.int8)
    gp = np.zeros((F, 2), np.int16)
    mg = np.zeros((F, 2), bool)
    rc = np.ones((F, 2))
    merged_slots = A.merged()
    lead_slot = np.array([roles.lead_slot(float(t)) for t in cam.t])
    rconf = np.array([roles.conf(float(t)) for t in cam.t])
    for r in range(2):                                           # role 0 lead, 1 follow
        slot = slot_of_role(lead_slot, r)
        for i in np.flatnonzero(A.row[np.arange(F), slot] >= 0):
            s = int(slot[i])
            pr = int(A.row[i, s])
            P = cam.persons(i)
            if pr >= len(P):
                continue
            row[i, r] = pr
            box[i, r] = person_box(P[pr])
            kp[i, r] = P[pr]
            cost[i, r], kind[i, r], nj[i, r], gp[i, r], mg[i, r] = A.cost[i, s], A.kind[i, s], A.nj[i, s], A.gap[i, s], merged_slots[i, s]
            rc[i, r] = rconf[i]
    conf = teacher_conf(cost, kind, nj, rc, cam.attached, gp, mg).astype(np.float32)
    return {"t": cam.t.astype(np.float64), "frame_index": cam.fi.astype(np.int32), "ok": cam.ok, "row": row, "box": box, "kp": kp, "conf": conf,
            "cost": cost, "kind": kind, "nj": nj, "gap": gp, "merged": mg, "role_conf": rconf.astype(np.float32), "lead_slot": lead_slot.astype(np.int8),
            "size": np.array(cam.size, np.int32), "attached": np.array(cam.attached)}


def coverage_from(out: dict, cams: dict, min_conf: float = 0.5, seconds: int = 361) -> dict:
    """Per second of the master clock and role: which cameras see the teacher (>= half of the camera's frames of that second have an
    association with confidence >= min_conf), plus per-camera shares."""
    rep = {"seconds": seconds, "min_conf": min_conf, "per_camera": {}, "lead": [], "follow": [], "both": []}
    seen = {r: np.zeros((seconds, len(out)), bool) for r in range(2)}
    names = sorted(out)
    for ci, c in enumerate(names):
        o = out[c]
        t, ok = o["t"], o["ok"]
        for r in range(2):
            good = (o["conf"][:, r] >= min_conf) & ok
            for s in range(seconds):
                m = ok & (t >= s) & (t < s + 1)
                if m.sum() >= 3:
                    seen[r][s, ci] = good[m].mean() >= 0.5
        rep["per_camera"][c] = {"frames": int(ok.sum()), **{ROLES[r]: round(float((o["conf"][ok, r] >= min_conf).mean()), 3) for r in range(2)},
                                "both": round(float(((o["conf"][ok, 0] >= min_conf) & (o["conf"][ok, 1] >= min_conf)).mean()), 3),
                                "attached": bool(o["attached"])}
    for s in range(seconds):
        rep["lead"].append([names[j] for j in np.flatnonzero(seen[0][s])])
        rep["follow"].append([names[j] for j in np.flatnonzero(seen[1][s])])
        rep["both"].append([names[j] for j in np.flatnonzero(seen[0][s] & seen[1][s])])
    for r in ("lead", "follow", "both"):
        n = np.array([len(x) for x in rep[r]])
        rep[f"{r}_cameras_per_second"] = {str(k): int((n == k).sum()) for k in range(0, len(names) + 1)}
        rep[f"{r}_seconds_with_3_or_more"] = int((n >= 3).sum())
        rep[f"{r}_seconds_with_less_than_2"] = int((n < 2).sum())
    return rep


# ------------------------------------------------------------------------------------------------ review pictures


def role_tile(cam: Cam, o: dict, i: int, width: int = 420) -> np.ndarray | None:
    rows = {"lead": int(o["row"][i, 0]), "follow": int(o["row"][i, 1])}
    labels = {"lead": f"Kadu {o['conf'][i, 0]:.2f}", "follow": f"Larissa {o['conf'][i, 1]:.2f}"}
    return review_tile(cam, i, rows, labels, ROLE_COLOURS, width)


def review_sheets(tag: str = "", cams=ALL_CAMS, times=None, out_dir: Path | None = None) -> list[Path]:
    """One contact sheet per camera: the teachers of ten frames spread over the class (Kadu red, Larissa green, everybody else grey)."""
    import cv2
    d = pdir(tag)
    out_dir = out_dir or WORK / "review" / "people"
    out_dir.mkdir(parents=True, exist_ok=True)
    times = times or [12, 48, 84, 120, 156, 192, 228, 264, 300, 336]
    paths = []
    for c in cams:
        cam = load_cam(c)
        o = dict(np.load(d / f"teachers_{c}.npz"))
        tiles = []
        for t in times:
            i = int(np.argmin(np.abs(cam.t - t)))
            if abs(cam.t[i] - t) < 2.0 and cam.ok[i]:
                tiles.append(role_tile(cam, o, i, 360 if cam.size[0] < cam.size[1] else 520))
        p = out_dir / f"teachers_{c}{('_' + tag) if tag else ''}.jpg"
        cv2.imwrite(str(p), sheet(tiles, 5), [cv2.IMWRITE_JPEG_QUALITY, 85])
        paths.append(p)
    return paths


def review_switches(tag: str = "", cams=("02", "01"), first: int = 0, n: int = 6, dt: float = 1.0) -> list[Path]:
    """For the role switches of the 3D slots (and so the places where an identity swap could hide): the frames 1 s before, at and 1 s after."""
    import cv2
    d = pdir(tag)
    summ = json.loads((d / "teachers.json").read_text())
    sw = summ["role_switches_of_the_3d_slots"][first:first + n]
    out = []
    for c in cams:
        cam = load_cam(c)
        o = dict(np.load(d / f"teachers_{c}.npz"))
        tiles = []
        for t, _ in sw:
            for tt in (t - dt, t, t + dt):
                i = int(np.argmin(np.abs(cam.t - tt)))
                if cam.ok[i]:
                    tiles.append(role_tile(cam, o, i, 330 if cam.size[0] < cam.size[1] else 480))
        p = WORK / "review" / "people" / f"switches_{c}_{first}{('_' + tag) if tag else ''}.jpg"
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), sheet(tiles, 3), [cv2.IMWRITE_JPEG_QUALITY, 85])
        out.append(p)
    return out


# ------------------------------------------------------------------------------------------------ the owner's JSON versus the pipeline's own detector


def loo_reference(W: World, tr: Track3D, cams: dict, j: int, s: int, cam_id: str):
    """3D joints (17,3) of teacher slot s at instant j for judging camera `cam_id`: the selected candidate re-triangulated WITHOUT that
    camera's view (needs two other views), or the track's own joints when the camera is not part of the candidate (an attached phone, or
    a main camera that did not see the person). None when the camera is part of the candidate and fewer than two other views remain."""
    r = int(tr.obs[j, s])
    if r < 0:
        return None
    views = {c: (int(W.fi[r, n]), int(W.row[r, n])) for n, c in enumerate(W.cams) if W.fi[r, n] >= 0}
    if cam_id not in views:
        return tr.X[j, s]
    others = {c: v for c, v in views.items() if c != cam_id}
    if len(others) < 2:
        return None
    P, U, ok, Pm, f = [], [], [], [], []
    for c, (i, row) in others.items():
        cm = cams[c]
        p = cm.persons(i)[row]
        P.append(p)
        U.append(undistort_norm(p[:, :2].astype(float), cm.K[i], float(cm.k1[i])))
        ok.append(p[:, 2] >= CONF)
        Pm.append(np.hstack([cm.R[i], cm.tvec[i][:, None]]))
        f.append(float(cm.K[i][0, 0]))
    X, nv, res = triangulate_views(np.stack(U), np.stack(ok), np.stack(Pm), np.array(f), 14.0)
    return X if np.isfinite(X[:, 0]).sum() >= 8 else None


def person_cost(uv: np.ndarray, P: np.ndarray, conf: float = 0.3, min_joints: int = 5) -> tuple[float, float, int]:
    """(median joint distance / person size, median joint distance in px, joints compared) of a person P (17,3) against projected joints uv (17,2)."""
    v = (P[:, 2] >= conf) & np.isfinite(uv[:, 0])
    if v.sum() < min_joints:
        return float("inf"), float("inf"), int(v.sum())
    d = np.linalg.norm(uv[v] - P[v, :2], axis=1)
    return float(np.median(d) / max(person_height_px(P), 40.0)), float(np.median(d)), int(v.sum())


def best_person(uv: np.ndarray, persons: np.ndarray, conf: float = 0.3) -> tuple[int, float, float, int]:
    """(row, relative cost, px cost, joints) of the person of a detection list that best explains the projected joints uv."""
    best = (-1, float("inf"), float("inf"), 0)
    for r, P in enumerate(persons):
        c, px, n = person_cost(uv, P, conf)
        if c < best[1]:
            best = (r, c, px, n)
    return best


def yolo_people(models: dict, imgs: list, conf: float = 0.2) -> list:
    """The pipeline's own detector (dancecap.detect: person boxes of yolo12x + the keypoints of yolo11x-pose, a low-resolution and a
    high-resolution pass merged by detect.merge_scales) WITHOUT the segmentation model: per image (P,17,3) joints [x, y, conf] of its people."""
    from . import detect as D
    h, w = imgs[0].shape[:2]
    s = D.Settings()
    per_scale = []
    for imgsz in s.scales(w, h):
        det = models["det"].predict(imgs, imgsz=imgsz, conf=conf, iou=s.iou, classes=[D.PERSON], quantize=16, verbose=False, batch=len(imgs), max_det=300)
        pose = models["pose"].predict(imgs, imgsz=imgsz, conf=conf, iou=s.iou, classes=[D.PERSON], quantize=16, verbose=False, batch=len(imgs), max_det=300)
        res = []
        for d_, p_ in zip(det, pose):
            boxes = d_.boxes.xyxy.cpu().numpy() if len(d_.boxes) else np.zeros((0, 4), np.float32)
            scores = d_.boxes.conf.cpu().numpy() if len(d_.boxes) else np.zeros(0, np.float32)
            if p_.keypoints is not None and len(p_.boxes):
                kp = D.attach_keypoints(boxes, p_.boxes.xyxy.cpu().numpy(), p_.keypoints.data.cpu().numpy(), s.kpt_match_iou)
            else:
                kp = np.zeros((len(boxes), 17, 3), np.float32)
            res.append(D.FrameResult(boxes.astype(np.float32), scores.astype(np.float32), kp.astype(np.float32), np.zeros((0, 0), np.uint8)))
        per_scale.append(res)
    out = []
    for i in range(len(imgs)):
        r = per_scale[0][i]
        for other in per_scale[1:]:
            r = D.merge_scales(r, other[i], h, s.large_frac)
        out.append(r.kpts[r.scores >= conf] if len(r.kpts) else np.zeros((0, 17, 3), np.float32))
    return out


def load_yolo() -> dict:
    from ultralytics import YOLO
    from . import detect as D
    return {"det": YOLO(str(D.DET_WEIGHTS)), "pose": YOLO(str(D.POSE_WEIGHTS))}


def detcompare_samples(W: World, tr: Track3D, cams: dict, cam_id: str, n: int, miss_of: "Assoc | None" = None) -> list[dict]:
    """About n instants spread over the class at which both teachers are observed (>= 2 views) and the camera has a frame: the reference
    (leave-one-camera-out 3D joints) of each slot and the canonical frame. miss_of: an association of the camera: only instants where the JSON
    gave NO person for at least one teacher in that frame (the hard frames: embraces, occlusions) are sampled."""
    cam = cams[cam_id]
    ok_j = np.flatnonzero((tr.obs >= 0).all(1) & (tr.nv.min(1) >= 2))
    near = cam.near(tr.k[ok_j] / HZ)
    ok_j = ok_j[near >= 0]
    near = near[near >= 0]
    if miss_of is not None and len(ok_j):
        hard = (miss_of.row[near] < 0).any(1)
        ok_j, near = ok_j[hard], near[hard]
    if not len(ok_j):
        return []
    pick = np.unique(np.linspace(0, len(ok_j) - 1, n * 2).astype(int))
    out = []
    for q in pick:
        j, i = int(ok_j[q]), int(near[q])
        refs = [loo_reference(W, tr, cams, j, s, cam_id) for s in range(2)]
        if any(r is None for r in refs):
            continue
        out.append({"j": j, "i": i, "t": float(tr.k[j] / HZ), "ref": refs})
        if len(out) >= n:
            break
    return out


def detcompare_camera(cam_id: str, n: int, models: dict, cams: dict, W: World, tr: Track3D, batch: int = 4, log_fn=log, miss_of=None) -> dict:
    """The owner's JSON and the pipeline's detector against the same independent reference on ~n frames of one camera."""
    cam = cams[cam_id]
    samples = detcompare_samples(W, tr, cams, cam_id, n, miss_of)
    gate = 0.45 if cam.attached else 0.22
    want = {int(cam.fi[s["i"]]): s for s in samples}
    rows, buf, t0 = [], [], time.time()

    def flush():
        if not buf:
            return
        dets = yolo_people(models, [b[1] for b in buf])
        for (smp, _), yp in zip(buf, dets):
            i = smp["i"]
            Pj = cam.persons(i)
            r = {"t": smp["t"], "n_json": int(len(Pj)), "n_yolo": int(len(yp)), "slots": []}
            for s in range(2):
                uv, z = project(smp["ref"][s], cam.R[i], cam.tvec[i], cam.K[i], float(cam.k1[i]))
                uv = np.where((z > 0.2)[:, None], uv, np.nan)
                jr = best_person(uv, Pj)
                yr = best_person(uv, yp)
                sl = {"json": {"found": bool(jr[1] <= gate), "cost": jr[1], "px": jr[2], "nj": jr[3]},
                      "yolo": {"found": bool(yr[1] <= gate), "cost": yr[1], "px": yr[2], "nj": yr[3]}}
                if jr[0] >= 0 and yr[0] >= 0 and sl["json"]["found"] and sl["yolo"]["found"]:       # both found: how far apart are their joints
                    v = (Pj[jr[0], :, 2] >= 0.3) & (yp[yr[0], :, 2] >= 0.3)
                    if v.sum() >= 5:
                        d = np.linalg.norm(Pj[jr[0], v, :2] - yp[yr[0], v, :2], axis=1)
                        sl["json_vs_yolo_px"] = float(np.median(d))
                        sl["json_vs_yolo_rel"] = float(np.median(d) / max(person_height_px(Pj[jr[0]]), 40.0))
                r["slots"].append(sl)
            rows.append(r)
        buf.clear()

    for n_, img in stream_frames(cam_id, cam.size[0], last=max(want) if want else -1):
        if n_ in want:
            buf.append((want[n_], img.copy()))
            if len(buf) >= batch:
                flush()
    flush()
    log_fn(f"detcompare {cam_id}: {len(rows)} frames ({time.time() - t0:.0f} s)")
    return {"cam": cam_id, "frames": len(rows), "gate": gate, "rows": rows}


def summarise_detcompare(res: dict) -> dict:
    """Per camera and overall: recall of a teacher (a person whose joints agree with the independent reference within the gate), median joint
    error against the reference, agreement of the two sources, mean number of people per frame."""
    out = {"cams": {}}
    allr = {"json": [], "yolo": []}
    for c, r in res.items():
        rows = r["rows"]
        d = {"frames": len(rows), "people_per_frame": {"json": round(float(np.mean([x["n_json"] for x in rows])), 2) if rows else None,
                                                       "yolo": round(float(np.mean([x["n_yolo"] for x in rows])), 2) if rows else None}}
        for src in ("json", "yolo"):
            sl = [s[src] for x in rows for s in x["slots"]]
            found = np.array([s["found"] for s in sl], bool)
            cost = np.array([s["cost"] for s in sl])
            px = np.array([s["px"] for s in sl])
            d[src] = {"teacher_instances": len(sl), "recall": round(float(found.mean()), 3) if len(sl) else None,
                      "median_error_rel_found": round(float(np.median(cost[found])), 4) if found.any() else None,
                      "median_error_px_found": round(float(np.median(px[found])), 2) if found.any() else None,
                      "p90_error_px_found": round(float(np.percentile(px[found], 90)), 2) if found.any() else None}
            allr[src] += [(f, co, p) for f, co, p in zip(found, cost, px)]
        agree = [s.get("json_vs_yolo_rel") for x in rows for s in x["slots"] if s.get("json_vs_yolo_rel") is not None]
        d["json_vs_yolo_median_rel_distance"] = round(float(np.median(agree)), 4) if agree else None
        d["both_found"] = len(agree)
        out["cams"][c] = d
    for src in ("json", "yolo"):
        a = allr[src]
        f = np.array([x[0] for x in a], bool)
        out[src] = {"teacher_instances": len(a), "recall": round(float(f.mean()), 3) if len(a) else None,
                    "median_error_px_found": round(float(np.median([x[2] for x in a if x[0]])), 2) if f.any() else None,
                    "median_error_rel_found": round(float(np.median([x[1] for x in a if x[0]])), 4) if f.any() else None}
    return out


def step_detcompare(tag: str = "", cams=ALL_CAMS, n: int = 300, mode: str = "all") -> dict:
    """mode "all": frames where the cameras triangulated both teachers (the easy frames); "miss": only frames where the JSON has no person for one of the
    teachers although the other cameras place him / her there (does the pipeline's own detector find the people the JSON misses?)."""
    d = pdir(tag)
    W = World.load(d / "world_people.npz")
    tr = Track3D.load(d / "track3d.npz")
    cs = {c: load_cam(c) for c in ALL_CAMS}
    models = load_yolo()
    res = {}
    stem = "detcompare" if mode == "all" else "detcompare_miss"
    for c in cams:
        miss_of = Assoc.load(d / f"assoc_{c}.npz") if mode == "miss" else None
        res[c] = detcompare_camera(c, n, models, cs, W, tr, miss_of=miss_of)
        (d / f"{stem}_{c}.json").write_text(json.dumps(res[c], default=float))
    summ = summarise_detcompare(res)
    (d / f"{stem}.json").write_text(json.dumps(summ, indent=1, default=float))
    return summ


def coverage_figure(tag: str = "", chunks: list | None = None, out: Path | None = None) -> Path:
    """The coverage map: per second of the class and per teacher, which phones see him / her (green = confident association in >= half of the frames of
    that second); below, the number of phones for each teacher and both; the chunk windows (and their overlaps) of the plan on top when given."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cov = json.loads((pdir(tag) / "coverage.json").read_text())
    names = sorted(cov["per_camera"])
    n = cov["seconds"]
    fig, ax = plt.subplots(4, 1, figsize=(16, 9.5), sharex=True, gridspec_kw={"height_ratios": [1.1, 3, 3, 2]})
    if chunks:
        for j, c in enumerate(chunks):
            ax[0].barh(0, c["t"][1] - c["t"][0], left=c["t"][0], height=0.8, color="#4c78a8" if j % 2 == 0 else "#9ecae9", edgecolor="k", lw=0.5)
            ax[0].text((c["t"][0] + c["t"][1]) / 2, 0, f"{c['id']}  {len(c['cams'])} cams", ha="center", va="center", fontsize=7)
        ax[0].set_yticks([])
    ax[0].set_title("teacher coverage per second: which phones see Kadu / Larissa (confident association in >= half of the frames of the second)", fontsize=10)
    for a, role in ((ax[1], "lead"), (ax[2], "follow")):
        img = np.zeros((len(names), n))
        for s in range(n):
            for c in cov[role][s]:
                img[names.index(c), s] = 1
        a.imshow(img, aspect="auto", interpolation="nearest", cmap="Greens", vmin=0, vmax=1.3, extent=[0, n, len(names), 0])
        a.set_yticks(np.arange(len(names)) + 0.5)
        a.set_yticklabels([f"cam {c}{' (attached)' if cov['per_camera'][c]['attached'] else ''}" for c in names], fontsize=8)
        a.set_ylabel("Kadu (lead)" if role == "lead" else "Larissa (follow)", fontsize=9)
    for role, col in (("lead", "#b03030"), ("follow", "#2f9f2f"), ("both", "k")):
        ax[3].step(np.arange(n), [len(x) for x in cov[role]], where="post", color=col, lw=1.0, label={"lead": "Kadu", "follow": "Larissa", "both": "both"}[role])
    ax[3].axhline(3, color="gray", lw=0.5, ls=":")
    ax[3].set_ylabel("phones")
    ax[3].legend(ncol=3, fontsize=8, loc="lower right")
    ax[3].set_xlabel("master clock (s)")
    fig.tight_layout()
    out = out or WORK / "review" / "people" / "coverage_map.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=90)
    plt.close(fig)
    return out


def portrait_colours() -> dict:
    """The colour rule's cues measured on the two reference portraits (read through the junction, never copied; only these numbers are kept): the share
    of red-hair pixels in the upper half of the picture and the median brightness (HSV V) of the lower middle (the top)."""
    import cv2
    from .dancers_appearance import red_hair_mask
    out = {}
    for who in ("kadu", "larissa"):
        p = SOURCE / f"{who}.jpg"
        if not p.exists():
            continue
        img = cv2.imread(str(p))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, w = hsv.shape[:2]
        upper = hsv[: h // 2]
        lower = hsv[int(0.6 * h):, int(0.25 * w):int(0.75 * w)]
        out[who] = {"size": [w, h], "red_hair_share_upper_half": round(float(red_hair_mask(upper).mean()), 3),
                    "top_median_V_lower_middle": float(np.median(lower[..., 2])), "top_median_S_lower_middle": float(np.median(lower[..., 1]))}
    return out


def verify_report(tag: str = "", cams=ALL_CAMS) -> dict:
    """Numbers that do not depend on the identity model itself: (1) every selected teacher stands inside the dance zone and is standing; (2) the lead is TALLER
    than the follow in the 3D tracks (Kadu 1.727 m, Larissa 1.651 m: an independent check of the colour decision); (3) how well the final roles agree with the
    colour of each single observation; (4) where the role switches of the 3D slots happen (the two close together = an embrace); (5) the portraits' colours."""
    d = pdir(tag)
    tr = Track3D.load(d / "track3d.npz")
    roles_z = np.load(d / "roles.npz")
    state = roles_z["state"]
    rep = {}
    cen = np.full((len(tr.k), 2, 3), np.nan)
    hgt = np.full((len(tr.k), 2), np.nan)
    for s in range(2):
        for j in np.flatnonzero(np.isfinite(tr.X[:, s, 5, 0])):
            X = tr.X[j, s]
            cen[j, s] = centre(X)
            ank = np.nanmin(X[[15, 16], 1]) if np.isfinite(X[[15, 16], 1]).any() else np.nan
            if np.isfinite(X[0, 1]) and np.isfinite(ank):
                hgt[j, s] = X[0, 1] - ank
    c0 = np.array(tr.centre0)
    dist = np.linalg.norm(cen[..., [0, 2]] - c0, axis=-1)
    rep["zone"] = {"centre_xz_m": np.round(c0, 2).tolist(), "selected_instants": int(np.isfinite(dist).sum()), "distance_to_centre_m_p50_p95_max":
                   [round(float(np.nanpercentile(dist, q)), 2) for q in (50, 95, 100)], "share_within_2_5_m": round(float(np.nanmean(dist[np.isfinite(dist)] < 2.5)), 4),
                   "shoulder_height_m_p5": round(float(np.nanpercentile(np.nanmean(tr.X[:, :, [5, 6], 1], axis=2), 5)), 2)}
    b = np.clip(np.round(tr.k / HZ / BIN_S).astype(int), 0, len(state) - 1)
    lead_slot = state[b]                                           # 0 = slot A is the lead
    lead_h = np.where(lead_slot == 0, hgt[:, 0], hgt[:, 1])
    fol_h = np.where(lead_slot == 0, hgt[:, 1], hgt[:, 0])
    both = np.isfinite(lead_h) & np.isfinite(fol_h) & (np.abs(lead_h - 1.5) < 0.4) & (np.abs(fol_h - 1.5) < 0.4)
    rep["height_nose_to_lowest_ankle_m"] = {"instants": int(both.sum()), "lead_median": round(float(np.median(lead_h[both])), 3), "follow_median": round(float(np.median(fol_h[both])), 3),
                                            "lead_minus_follow_cm": round(100 * float(np.median(lead_h[both] - fol_h[both])), 1),
                                            "share_of_instants_lead_taller": round(float((lead_h[both] > fol_h[both]).mean()), 3),
                                            "stature_difference_cm_from_the_take": round(100 * (1.727 - 1.651), 1),
                                            "swapped_roles_would_give_cm": round(-100 * float(np.median(lead_h[both] - fol_h[both])), 1)}
    agree, total = 0, 0
    per = {}
    for c in cams:
        a = dict(np.load(d / f"appearance_{c}.npz"))
        if not len(a["i"]):
            continue
        cam = load_cam(c, with_poses=False)
        feats = _feat_matrix(a["desc"])
        model = fit_class_model(feats, None)
        llr = llr_obs(feats, model)
        has = np.isfinite(a["desc"][:, :, 0])
        bi = np.clip(np.round(cam.t[a["i"]] / BIN_S).astype(int), 0, len(state) - 1)
        lead_is_a = state[bi] == 0
        ok = has[:, 0] & has[:, 1] & (np.abs(llr[:, 0] - llr[:, 1]) > 1.0)
        a_says_lead = (llr[:, 0] - llr[:, 1]) > 0
        ag = (a_says_lead == lead_is_a)[ok]
        per[c] = {"observations_with_both_persons_and_clear_colour": int(ok.sum()), "final_role_agrees_with_the_colour_of_the_observation": round(float(ag.mean()), 4)}
        agree += int(ag.sum())
        total += int(ok.sum())
    rep["colour_agreement"] = {"overall": round(agree / max(total, 1), 4), "observations": total, "per_camera": per}
    summ = json.loads((d / "teachers.json").read_text())
    sw = summ["role_switches_of_the_3d_slots"]
    sep, both_obs = separation_per_bin(tr, len(state))
    near = [bool(np.isfinite(sep[min(len(sep) - 1, int(round(t / BIN_S)))]) and sep[min(len(sep) - 1, int(round(t / BIN_S)))] < 1.0) or not np.isfinite(sep[min(len(sep) - 1, int(round(t / BIN_S)))])
            for t, _ in sw]
    rep["role_switches"] = {"count": len(sw), "at_a_close_pair_or_with_a_missing_partner": int(sum(near)), "at_a_separated_pair": int(len(near) - sum(near)),
                            "times_s": [t for t, _ in sw]}
    rep["portraits"] = portrait_colours()
    (d / "verification.json").write_text(json.dumps(rep, indent=1, default=float))
    return rep


# ------------------------------------------------------------------------------------------------ steps (cached on disk, resumable)


def pdir(tag: str = "") -> Path:
    d = PEOPLE / tag if tag else PEOPLE
    d.mkdir(parents=True, exist_ok=True)
    return d


def step_tables(cams=ALL_CAMS, force: bool = False) -> None:
    for c in cams:
        pose_table(c, force=force)


def step_world(step: int = 1, tag: str = "", force: bool = False, **kw) -> World:
    p = pdir(tag) / "world_people.npz"
    if p.exists() and not force:
        return World.load(p)
    mains = {c: load_cam(c) for c in MAIN}
    ks = np.arange(0, K_N, step)
    W = world_people(mains, ks, pair_px=kw.get("pair_px", 12.0), rel=kw.get("rel", 0.025), max_px=kw.get("max_px", 14.0))
    W.save(p)
    log(f"world people: {len(W.X)} candidates over {len(ks)} instants -> {p.name}")
    return W


def step_teachers(tag: str = "", force: bool = False, step: int = 1) -> Track3D:
    p = pdir(tag) / "track3d.npz"
    if p.exists() and not force:
        return Track3D.load(p)
    W = World.load(pdir(tag) / "world_people.npz")
    tr = teacher_tracks(W, dt=step / HZ)
    tr.save(p)
    return tr


def _assoc_one(args) -> str:
    cam_id, tag = args
    d = pdir(tag)
    W = World.load(d / "world_people.npz")
    tr = Track3D.load(d / "track3d.npz")
    cam = load_cam(cam_id)
    t0 = time.time()
    A = fill_by_2d_continuity(cam, associate(cam, tr, members_of(W, tr, cam_id)))
    A.save(d / f"assoc_{cam_id}.npz")
    ok = cam.ok
    a, b = A.row[ok, 0] >= 0, A.row[ok, 1] >= 0
    return f"assoc {cam_id}: slot A {a.mean():.3f} slot B {b.mean():.3f} both {(a & b).mean():.3f} ({time.time() - t0:.0f} s)"


def _app_one(args) -> str:
    cam_id, tag, step = args
    d = pdir(tag)
    cam = load_cam(cam_id)
    A = Assoc.load(d / f"assoc_{cam_id}.npz")
    r = appearance_camera(cam, A, step=step)
    np.savez_compressed(d / f"appearance_{cam_id}.npz", **r)
    return f"appearance {cam_id}: {len(r['i'])} frames"


def _pool(fn, jobs: list, procs: int) -> None:
    if procs <= 1 or len(jobs) == 1:
        for j in jobs:
            log(fn(j))
        return
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(min(procs, len(jobs))) as ex:
        for msg in ex.map(fn, jobs):
            log(msg)


def step_assoc(cams=ALL_CAMS, tag: str = "", procs: int = 1) -> None:
    _pool(_assoc_one, [(c, tag) for c in cams], procs)


def step_appearance(cams=ALL_CAMS, tag: str = "", step: int = 1, procs: int = 1) -> None:
    _pool(_app_one, [(c, tag, step) for c in cams], procs)


def step_finalize(cams=ALL_CAMS, tag: str = "") -> dict:
    d = pdir(tag)
    tr = Track3D.load(d / "track3d.npz")
    cs = {c: load_cam(c, with_poses=True) for c in cams}
    apps = {c: dict(np.load(d / f"appearance_{c}.npz")) for c in cams}
    roles = roles_from_appearance(tr, apps, cs)
    np.savez_compressed(d / "roles.npz", state=roles.state, p_a_lead=roles.p_a_lead, e=roles.e, n_obs=roles.n_obs, bin_s=BIN_S)
    out = {}
    for c in cams:
        A = Assoc.load(d / f"assoc_{c}.npz")
        o = teachers_for_camera(cs[c], A, roles)
        np.savez_compressed(d / f"teachers_{c}.npz", **o)
        out[c] = o
    cov = coverage_from(out, cs)
    (d / "coverage.json").write_text(json.dumps(cov))
    both = (np.array([len(x) for x in cov["both"]]))
    summ = {"take": TAKE, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "tag": tag, "cams": list(cams),
            "role_switches_of_the_3d_slots": roles.switches, "n_switches": len(roles.switches),
            "role_decisive_share": round(float(((roles.p_a_lead < 0.05) | (roles.p_a_lead > 0.95)).mean()), 4),
            "bins_with_colour_evidence": round(float((roles.n_obs > 0).mean()), 3),
            "class_models": roles.models,
            "per_camera": cov["per_camera"],
            "seconds_both_teachers_seen_by": {str(k): int((both == k).sum()) for k in range(0, len(cams) + 1)},
            "lead_cameras_per_second": cov["lead_cameras_per_second"], "follow_cameras_per_second": cov["follow_cameras_per_second"]}
    (d / "teachers.json").write_text(json.dumps(summ, indent=1, default=float))
    log(f"finalize: {len(roles.switches)} slot flips corrected by colour; both teachers seen by >= 3 cameras in {int((both >= 3).sum())} of {len(both)} s")
    return summ


def run_all(tag: str = "", step: int = 1, cams=ALL_CAMS, procs: int = 1) -> dict:
    step_tables(cams)
    step_world(step, tag)
    step_teachers(tag, step=step)
    step_assoc(cams, tag, procs)
    step_appearance(cams, tag, procs=procs)
    return step_finalize(cams, tag)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.recap_people", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("tables", "world", "teachers", "assoc", "appearance", "finalize", "run", "detcompare"):
        p = sub.add_parser(name)
        p.add_argument("--cams", default=",".join(ALL_CAMS))
        p.add_argument("--tag", default="", help="work in people/<tag>/ (a development run on a subset of instants)")
        p.add_argument("--step", type=int, default=1, help="world / teachers: grid stride (1 = every 30 Hz instant)")
        p.add_argument("--procs", type=int, default=1)
        p.add_argument("--force", action="store_true")
        p.add_argument("--n", type=int, default=300, help="detcompare: frames per camera")
        p.add_argument("--mode", default="all", choices=("all", "miss"), help="detcompare: easy frames (all) or the frames the JSON misses (miss)")
    a = ap.parse_args(argv)
    cams = tuple(c for c in a.cams.split(",") if c)
    if a.cmd == "tables":
        step_tables(cams, a.force)
    elif a.cmd == "world":
        step_world(a.step, a.tag, a.force)
    elif a.cmd == "teachers":
        step_teachers(a.tag, a.force, a.step)
    elif a.cmd == "assoc":
        step_assoc(cams, a.tag, a.procs)
    elif a.cmd == "appearance":
        step_appearance(cams, a.tag, procs=a.procs)
    elif a.cmd == "finalize":
        step_finalize(cams, a.tag)
    elif a.cmd == "detcompare":
        step_detcompare(a.tag, cams, a.n, a.mode)
    else:
        run_all(a.tag, a.step, cams, a.procs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
