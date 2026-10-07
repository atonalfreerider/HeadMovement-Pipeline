"""Exact contact geometry between posed SMPL-X meshes, for the hand stages (refine_hands, hands_detail).

Why (adversarial review of the hands build, 2026-10-07): the signed gap of a hand to the partner was the distance to
the partner's NEAREST VERTEX, signed by that vertex's normal. Near concave regions (armpit, between the partner's arm
and torso, between fingers) the sign is often wrong: a hand hovering 2 cm above a shoulder was counted as 2 cm sunk
(Larissa's shoulder hand: 24 % of frames 'sunk > 1 cm' reported, 0 % really). A ray-parity inside test is no better:
the SMPL-X body is not watertight (a 32-edge mouth hole) and gave false sinks of 7-34 cm. Here:

  closest_points    exact point-to-triangle distance (Ericson's closest point; candidates = the faces around the k
                    nearest vertices), with the face and barycentric coordinates of the closest point
  winding_numbers   generalized winding number (Jacobson et al. 2013; solid angles, van Oosterom & Strackee): ~1 inside
                    a closed surface, ~0 outside, robust to small holes - inside = w > 0.5. Brute force over all faces
                    in torch: seconds on a pod GPU, slow on the 4-CPU workstation (default_device: CUDA only on a pod,
                    DANCECAP_CONTACT_DEVICE overrides)
  signed_distance   both together: negative = inside the partner (depth = exact distance to its surface)
  HandFaces / self_intersections
                    exact triangle-triangle intersections between DIFFERENT fingers of one hand (edge-triangle tests
                    after an AABB prefilter; same finger and palm-vs-proximal-phalanx skin folds ignored), as the
                    reviewer's Open3D check
  finger capsules   21-point hand skeleton -> sample points along each finger's phalanges with per-segment radii
                    from the shaped rest mesh: a cheap differentiable spacing proxy for finger-finger collisions
"""
from __future__ import annotations

import os

import numpy as np

SMPLX_FINGERS = ("index", "middle", "pinky", "ring", "thumb")      # SMPL-X order of the 15 finger joints per hand
FINGER_BASE = {"left": 25, "right": 40}
WRIST_J = {"left": 20, "right": 21}
# 21-point (COCO-WholeBody) chains used for the spacing proxy: MCP -> PIP -> DIP -> tip; thumb MCP -> IP -> tip
CHAINS21 = {"thumb": (2, 3, 4), "index": (5, 6, 7, 8), "middle": (9, 10, 11, 12), "ring": (13, 14, 15, 16),
            "pinky": (17, 18, 19, 20)}


def default_device() -> str:
    """'cuda' on a Runpod pod with a GPU, else 'cpu' (the workstation's GPU is shared: heavy jobs go to pods);
    DANCECAP_CONTACT_DEVICE=cpu|cuda overrides."""
    d = os.environ.get("DANCECAP_CONTACT_DEVICE")
    if d:
        return d
    try:
        from . import env
        import torch
        return "cuda" if env.is_pod() and torch.cuda.is_available() else "cpu"
    except Exception:   # noqa: BLE001
        return "cpu"


# ------------------------------------------------------------------------------------------------ closest points

class Topology:
    """Faces (a subset of a mesh's faces is fine: closest points are then restricted to that subset) + vertex -> face
    adjacency for candidate lookup."""

    def __init__(self, faces: np.ndarray, n_verts: int, max_deg: int = 10):
        self.F = np.asarray(faces, np.int64)
        self.nV = int(n_verts)
        cnt = np.bincount(self.F.ravel(), minlength=self.nV)
        # SMPL-X: degree <= 8 for 99.9 % of the vertices, a few eye / mouth vertices up to 32: capping the table at
        # max_deg keeps the candidate arrays 3x smaller (their other faces are reached from neighbouring vertices)
        self.maxdeg = int(min(max(cnt.max(), 1), max_deg))
        adj = -np.ones((self.nV, self.maxdeg), np.int64)
        fill = np.zeros(self.nV, np.int64)
        for f, tri in enumerate(self.F):
            for v in tri:
                if fill[v] < self.maxdeg:
                    adj[v, fill[v]] = f
                    fill[v] += 1
        self.adj = adj
        self.used = np.flatnonzero(cnt > 0)          # vertices that belong to at least one face


def point_triangle(P: np.ndarray, A: np.ndarray, B: np.ndarray, C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Closest points of P (..., 3) on triangles (A, B, C) (..., 3) -> (q (..., 3), barycentric (..., 3)).
    Ericson, Real-Time Collision Detection 5.1.5, vectorised (degenerate triangles fall back to a vertex/edge)."""
    ab, ac, ap = B - A, C - A, P - A
    d1, d2 = (ab * ap).sum(-1), (ac * ap).sum(-1)
    bp = P - B
    d3, d4 = (ab * bp).sum(-1), (ac * bp).sum(-1)
    cp = P - C
    d5, d6 = (ab * cp).sum(-1), (ac * cp).sum(-1)
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4
    eps = 1e-30
    bary = np.zeros(P.shape[:-1] + (3,))
    done = np.zeros(P.shape[:-1], bool)

    def put(mask, u, v, w):
        m = mask & ~done
        bary[m] = np.stack(np.broadcast_arrays(u, v, w), -1)[m]
        done[m] = True

    one, zero = np.ones(P.shape[:-1]), np.zeros(P.shape[:-1])
    put((d1 <= 0) & (d2 <= 0), one, zero, zero)                                    # vertex A
    put((d3 >= 0) & (d4 <= d3), zero, one, zero)                                   # vertex B
    v = d1 / np.where(np.abs(d1 - d3) > eps, d1 - d3, eps)
    put((vc <= 0) & (d1 >= 0) & (d3 <= 0), 1 - v, v, zero)                         # edge AB
    put((d6 >= 0) & (d5 <= d6), zero, zero, one)                                   # vertex C
    w = d2 / np.where(np.abs(d2 - d6) > eps, d2 - d6, eps)
    put((vb <= 0) & (d2 >= 0) & (d6 <= 0), 1 - w, zero, w)                         # edge AC
    den = (d4 - d3) + (d5 - d6)
    w = (d4 - d3) / np.where(np.abs(den) > eps, den, eps)
    put((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0), zero, 1 - w, w)           # edge BC
    s = va + vb + vc
    s = np.where(np.abs(s) > eps, s, eps)
    put(np.ones_like(done), va / s, vb / s, vc / s)                                 # inside the face
    q = bary[..., :1] * A + bary[..., 1:2] * B + bary[..., 2:] * C
    return q, bary


def closest_points(P: np.ndarray, V: np.ndarray, topo: Topology, k: int = 6, tree=None) -> dict:
    """Exact closest points of P (n,3) on the surface (V, topo.F): dist (n,), face (n,) index into topo.F, bary (n,3),
    q (n,3). Candidates are the faces around the k nearest (face-using) vertices."""
    from scipy.spatial import cKDTree
    P = np.asarray(P, float)
    used = topo.used
    if tree is None:
        tree = cKDTree(V[used])
    kk = int(min(k, len(used)))
    _, vi = tree.query(P, kk)
    vi = used[np.asarray(vi).reshape(len(P), kk)]
    cand = topo.adj[vi].reshape(len(P), -1)                                        # (n, k*maxdeg), -1 = none
    valid = cand >= 0
    cf = np.where(valid, cand, 0)
    tri = V[topo.F[cf]]                                                            # (n, m, 3, 3)
    q, bary = point_triangle(np.broadcast_to(P[:, None], tri.shape[:2] + (3,)), tri[..., 0, :], tri[..., 1, :],
                             tri[..., 2, :])
    d = np.linalg.norm(q - P[:, None], axis=-1)
    d = np.where(valid, d, np.inf)
    j = np.argmin(d, 1)
    r = np.arange(len(P))
    return {"dist": d[r, j], "face": cf[r, j], "bary": bary[r, j], "q": q[r, j]}


# ------------------------------------------------------------------------------------------------ winding numbers

def winding_numbers(P: np.ndarray, V: np.ndarray, F: np.ndarray, device: str | None = None,
                    chunk: int | None = None) -> np.ndarray:
    """Generalized winding numbers (n,) of points P (n,3) w.r.t. the triangle soup (V, F) (outward CCW faces)."""
    import torch
    P = np.asarray(P, float)
    if not len(P):
        return np.zeros(0)
    device = device or default_device()
    dt = torch.float32 if device.startswith("cuda") else torch.float64
    Vt = torch.as_tensor(np.asarray(V), dtype=dt, device=device)
    Ft = torch.as_tensor(np.asarray(F, np.int64), device=device)
    A, B, C = Vt[Ft[:, 0]], Vt[Ft[:, 1]], Vt[Ft[:, 2]]
    Pt = torch.as_tensor(P, dtype=dt, device=device)
    if chunk is None:
        chunk = max(16, int((2.5e8 if device.startswith("cuda") else 2.5e7) // max(len(F) * 12, 1)))
    out = []
    with torch.no_grad():
        for i in range(0, len(Pt), chunk):
            p = Pt[i:i + chunk, None]
            a, b, c = A[None] - p, B[None] - p, C[None] - p
            la, lb, lc = a.norm(dim=-1), b.norm(dim=-1), c.norm(dim=-1)
            det = (a * torch.cross(b, c, dim=-1)).sum(-1)
            den = la * lb * lc + (a * b).sum(-1) * lc + (b * c).sum(-1) * la + (c * a).sum(-1) * lb
            out.append((2 * torch.atan2(det, den)).sum(1) / (4 * np.pi))
            del a, b, c, la, lb, lc, det, den
    w = torch.cat(out).double().cpu().numpy()
    return w


def _pt_tri_sq_torch(torch, p, A, B, C):
    """Squared distance + barycentrics of points p (m,1,3) to triangles A, B, C (1,F,3) -> (m,F) each; Ericson's
    regions as a torch.where cascade (later regions override earlier ones in reverse precedence)."""
    ab, ac = B - A, C - A
    ap, bp, cp = p - A, p - B, p - C
    d1, d2 = (ab * ap).sum(-1), (ac * ap).sum(-1)
    d3, d4 = (ab * bp).sum(-1), (ac * bp).sum(-1)
    d5, d6 = (ab * cp).sum(-1), (ac * cp).sum(-1)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    tiny = torch.finfo(p.dtype).tiny * 1e6

    def sdiv(x, y):
        return x / torch.where(y.abs() > tiny, y, torch.full_like(y, tiny))
    den = va + vb + vc
    v, w = sdiv(vb, den), sdiv(vc, den)                                          # interior
    u = 1 - v - w
    t_bc = sdiv(d4 - d3, (d4 - d3) + (d5 - d6))
    m = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    u, v, w = torch.where(m, 0.0, u), torch.where(m, 1 - t_bc, v), torch.where(m, t_bc, w)
    t_ac = sdiv(d2, d2 - d6)
    m = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    u, v, w = torch.where(m, 1 - t_ac, u), torch.where(m, 0.0, v), torch.where(m, t_ac, w)
    m = (d6 >= 0) & (d5 <= d6)
    u, v, w = torch.where(m, 0.0, u), torch.where(m, 0.0, v), torch.where(m, 1.0, w)
    t_ab = sdiv(d1, d1 - d3)
    m = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    u, v, w = torch.where(m, 1 - t_ab, u), torch.where(m, t_ab, v), torch.where(m, 0.0, w)
    m = (d3 >= 0) & (d4 <= d3)
    u, v, w = torch.where(m, 0.0, u), torch.where(m, 1.0, v), torch.where(m, 0.0, w)
    m = (d1 <= 0) & (d2 <= 0)
    u, v, w = torch.where(m, 1.0, u), torch.where(m, 0.0, v), torch.where(m, 0.0, w)
    dq = ap - v[..., None] * ab - w[..., None] * ac                              # p - q
    return (dq * dq).sum(-1), u, v, w


def surface_query_torch(P: np.ndarray, V: np.ndarray, F_closest: np.ndarray, F_all: np.ndarray | None,
                        device: str, max_depth: float = 0.16, chunk: int | None = None) -> dict:
    """Brute force on a GPU: exact closest point of P (n,3) on the faces F_closest of V and (when F_all is given) the
    generalized winding number over F_all, sharing the per-face vectors. Points further than max_depth (+ 4 cm) from
    every vertex are skipped (dist = their vertex distance, outside)."""
    import torch
    dt = torch.float32
    Vt = torch.as_tensor(np.asarray(V), dtype=dt, device=device)
    Pt = torch.as_tensor(np.asarray(P), dtype=dt, device=device)
    n = len(Pt)
    Fc_np = np.asarray(F_closest, np.int64)
    # far points (skipped): closest = their nearest vertex of the queried faces, expressed as (a face of it, one-hot)
    vc = np.unique(Fc_np)
    first = np.zeros((len(V), 2), np.int64)
    for c_ in (2, 1, 0):                         # (a face containing the vertex, the vertex's corner in it)
        first[Fc_np[:, c_], 0] = np.arange(len(Fc_np))
        first[Fc_np[:, c_], 1] = c_
    dmin = torch.cdist(Pt, Vt[torch.as_tensor(vc, device=device)]).min(1)
    dist = dmin.values                                                           # vertex distance (upper bound)
    jv = vc[dmin.indices.cpu().numpy()]
    near = torch.nonzero(dist < max_depth + 0.04).flatten()
    face = torch.as_tensor(first[jv, 0], device=device)
    bary = torch.zeros((n, 3), dtype=dt, device=device)
    bary[torch.arange(n, device=device), torch.as_tensor(first[jv, 1], device=device)] = 1.0
    w_out = torch.full((n,), float("nan"), dtype=dt, device=device)
    Fc = torch.as_tensor(np.asarray(F_closest, np.int64), device=device)
    Ac, Bc, Cc = Vt[Fc[:, 0]][None], Vt[Fc[:, 1]][None], Vt[Fc[:, 2]][None]
    if F_all is not None:
        Fa = torch.as_tensor(np.asarray(F_all, np.int64), device=device)
        Aa, Ba, Ca = Vt[Fa[:, 0]][None], Vt[Fa[:, 1]][None], Vt[Fa[:, 2]][None]
    nf = max(len(Fc), len(F_all) if F_all is not None else 0)
    chunk = chunk or max(8, int(2e7 // max(nf, 1)))
    with torch.no_grad():
        for i in range(0, len(near), chunk):
            idx = near[i:i + chunk]
            p = Pt[idx][:, None]
            d2, u, v, w = _pt_tri_sq_torch(torch, p, Ac, Bc, Cc)
            j = d2.argmin(1)
            r = torch.arange(len(idx), device=device)
            dist[idx] = d2[r, j].clamp(min=0).sqrt()
            face[idx] = j
            bary[idx] = torch.stack([u[r, j], v[r, j], w[r, j]], 1)
            del d2, u, v, w
            if F_all is not None:
                a, b, c = Aa - p, Ba - p, Ca - p
                la, lb, lc = a.norm(dim=-1), b.norm(dim=-1), c.norm(dim=-1)
                det = (a * torch.cross(b, c, dim=-1)).sum(-1)
                den = la * lb * lc + (a * b).sum(-1) * lc + (b * c).sum(-1) * la + (c * a).sum(-1) * lb
                ww = (2 * torch.atan2(det, den)).sum(1) / (4 * np.pi)
                ok = dist[idx] < max_depth
                w_out[idx] = torch.where(ok, ww, torch.full_like(ww, float("nan")))
                del a, b, c, la, lb, lc, det, den
    out = {"dist": dist.double().cpu().numpy(), "face": face.cpu().numpy(), "bary": bary.double().cpu().numpy(),
           "w": w_out.double().cpu().numpy()}
    out["q"] = np.einsum("nk,nkd->nd", out["bary"], np.asarray(V, float)[Fc_np[out["face"]]])
    return out


def signed_distance(P: np.ndarray, V: np.ndarray, topo: Topology, faces_all: np.ndarray | None = None,
                    device: str | None = None, max_depth: float = 0.16, k: int = 6, tree=None,
                    inside_test: bool = True) -> dict:
    """Signed distance of P (n,3) to a body surface: exact distance to (V, topo.F), negative where the generalized
    winding number of the CLOSED body (V, faces_all; default topo.F) exceeds 0.5. Points further than max_depth from
    the surface are outside without a winding-number test (nothing sinks 16 cm into a body). Returns closest_points'
    dict + 'sd' (n,), 'inside' (n,) bool, 'w' (n,) (NaN where untested). On a CUDA device everything is brute force
    on the GPU (surface_query_torch); on the CPU closest points come from KD-tree candidates and only the winding
    numbers are brute force (slow: tests and small inputs)."""
    device = device or default_device()
    Fall = (topo.F if faces_all is None else faces_all) if inside_test else None
    if str(device).startswith("cuda"):
        cp = surface_query_torch(P, V, topo.F, Fall, device, max_depth)
        w = cp["w"]
    else:
        cp = closest_points(P, V, topo, k, tree)
        w = np.full(len(P), np.nan)
        cand = cp["dist"] < max_depth
        if inside_test and cand.any():
            w[cand] = winding_numbers(np.asarray(P)[cand], V, Fall, device)
        cp["w"] = w
    inside = np.nan_to_num(w, nan=0.0) > 0.5
    cp.update({"sd": np.where(inside, -cp["dist"], cp["dist"]), "inside": inside})
    return cp


def signed_distance_frames(P: np.ndarray, V: np.ndarray, topo: Topology, faces_all: np.ndarray | None = None,
                           device: str | None = None, max_depth: float = 0.16, k: int = 6) -> dict:
    """signed_distance per frame: P (K,n,3), V (K,nv,3) -> arrays (K,n[,3])."""
    from scipy.spatial import cKDTree
    device = device or default_device()
    gpu = str(device).startswith("cuda")
    out = {}
    for f in range(len(P)):
        r = signed_distance(P[f], V[f], topo, faces_all, device, max_depth, k,
                            None if gpu else cKDTree(V[f][topo.used]))
        for key, v in r.items():
            out.setdefault(key, []).append(v)
    return {key: np.stack(v) for key, v in out.items()}


# ------------------------------------------------------------------------------------------------ hand self-intersections

class HandFaces:
    """The faces of one hand (vertices dominated by the wrist or a finger joint of `side`), labelled by segment:
    palm (the wrist joint) or <finger><1..3>. Pairs eligible for a self-intersection: different fingers, palm vs a
    middle / distal phalanx; never same-finger phalanges (knuckle creases) or palm vs proximal phalanx (skin folding
    at the knuckles), never faces sharing a vertex."""

    def __init__(self, weights: np.ndarray, faces: np.ndarray, side: str):
        dom = np.asarray(weights).argmax(1)
        base, wr = FINGER_BASE[side], WRIST_J[side]
        seg = {wr: (0, 0)}                                   # joint -> (finger id 0 palm / 1..5, phalanx 0..3)
        for i in range(15):
            seg[base + i] = (1 + i // 3, 1 + i % 3)
        vids = np.flatnonzero(np.isin(dom, list(seg)))
        F = np.asarray(faces, np.int64)
        keep = np.isin(F, vids).all(1)
        self.face_ids = np.flatnonzero(keep)
        self.F = F[keep]                                     # indices into the full mesh
        lab = np.array([seg[int(np.bincount(dom[f]).argmax())] for f in self.F])
        self.finger, self.phal = lab[:, 0], lab[:, 1]
        self.seg_name = np.array(["palm" if f == 0 else f"{SMPLX_FINGERS[f - 1]}{p}"
                                  for f, p in zip(self.finger, self.phal)])
        fa, fb = self.finger[:, None], self.finger[None]
        pa, pb = self.phal[:, None], self.phal[None]
        ok = (fa != fb) & ~(((fa == 0) & (pb == 1)) | ((fb == 0) & (pa == 1)))
        share = (self.F[:, None, :, None] == self.F[None, :, None, :]).any((2, 3))
        self.eligible = ok & ~share & np.triu(np.ones_like(ok), 1)


def _seg_tri_hits(p0, p1, a, b, c, eps: float = 1e-12) -> np.ndarray:
    """Segments p0->p1 vs triangles (a,b,c), all (m,3): proper crossings (Moller-Trumbore, open intervals)."""
    e1, e2, d = b - a, c - a, p1 - p0
    h = np.cross(d, e2)
    det = (e1 * h).sum(-1)
    ok = np.abs(det) > eps
    f = 1.0 / np.where(ok, det, 1.0)
    s = p0 - a
    u = f * (s * h).sum(-1)
    q = np.cross(s, e1)
    v = f * (d * q).sum(-1)
    t = f * (e2 * q).sum(-1)
    tol = 1e-9
    return ok & (u > tol) & (v > tol) & (u + v < 1 - tol) & (t > tol) & (t < 1 - tol)


def self_intersections(V: np.ndarray, hf: HandFaces, return_faces: bool = False) -> dict:
    """Intersecting eligible face pairs of one hand in one posed mesh V (full mesh vertices, (nv,3)):
    {'n': count, 'pairs': {'<seg a>-<seg b>': count}} (+ 'faces': indices into hf.F of the faces involved)."""
    T = V[hf.F]                                              # (m,3,3)
    lo, hi = T.min(1), T.max(1)
    # sweep and prune along x: candidates j of face i have lo_x[j] in [lo_x[i], hi_x[i]] (sorted order)
    o = np.argsort(lo[:, 0], kind="stable")
    lx = lo[o, 0]
    end = np.searchsorted(lx, hi[o, 0], side="right")
    cnt = np.maximum(end - np.arange(len(o)) - 1, 0)
    ii = np.repeat(np.arange(len(o)), cnt)
    jj = (np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)) + ii + 1
    i, j = o[ii], o[jj]
    keep = (lo[i, 1:] <= hi[j, 1:]).all(1) & (lo[j, 1:] <= hi[i, 1:]).all(1)
    i, j = i[keep], j[keep]
    a_, b_ = np.minimum(i, j), np.maximum(i, j)
    keep = hf.eligible[a_, b_]
    i, j = a_[keep], b_[keep]
    if not len(i):
        return {"n": 0, "pairs": {}, "faces": np.zeros(0, int)} if return_faces else {"n": 0, "pairs": {}}
    A, B = T[i], T[j]
    hit = np.zeros(len(i), bool)
    for X, Y in ((A, B), (B, A)):
        for e0, e1 in ((0, 1), (1, 2), (2, 0)):
            hit |= _seg_tri_hits(X[:, e0], X[:, e1], Y[:, 0], Y[:, 1], Y[:, 2])
    pairs = {}
    for a, b in zip(i[hit], j[hit]):
        key = "-".join(sorted((hf.seg_name[a], hf.seg_name[b])))
        pairs[key] = pairs.get(key, 0) + 1
    out = {"n": int(hit.sum()), "pairs": pairs}
    if return_faces:
        out["faces"] = np.unique(np.concatenate([i[hit], j[hit]]))
    return out


def crossing_series(Vseq: np.ndarray, hf: HandFaces) -> tuple[np.ndarray, dict]:
    """Per frame number of intersecting finger-finger face pairs over Vseq (T,nv,3) + summed pair counts."""
    n = np.zeros(len(Vseq), int)
    tot: dict = {}
    for k in range(len(Vseq)):
        r = self_intersections(Vseq[k], hf)
        n[k] = r["n"]
        for key, v in r["pairs"].items():
            tot[key] = tot.get(key, 0) + v
    return n, tot


# ------------------------------------------------------------------------------------------------ finger capsules

def finger_radii(v_shaped: np.ndarray, J_rest: np.ndarray, weights: np.ndarray, tip_ids: list, side: str) -> dict:
    """{finger: (n_seg,) capsule radii (m)} from the shaped rest mesh: median distance of each phalanx's dominant
    vertices to its bone segment (joint -> child joint, the distal phalanx -> the fingertip vertex)."""
    dom = np.asarray(weights).argmax(1)
    base = FINGER_BASE[side]
    tips = dict(zip(("thumb", "index", "middle", "ring", "pinky"), tip_ids))
    out = {}
    for name in CHAINS21:
        # the SMPL-X joints of this finger, proximal -> distal (thumb: its 3 joints; the spacing chain uses the last 2)
        fj = list(range(base + 3 * SMPLX_FINGERS.index(name), base + 3 * SMPLX_FINGERS.index(name) + 3))
        ends = [J_rest[j] for j in fj] + [v_shaped[tips[name]]]
        rad = []
        for n_, j in enumerate(fj):
            a, b = ends[n_], ends[n_ + 1]
            X = v_shaped[dom == j]
            if not len(X):
                rad.append(np.nan)
                continue
            ab = b - a
            u = np.clip(((X - a) @ ab) / max(ab @ ab, 1e-12), 0, 1)
            rad.append(float(np.median(np.linalg.norm(X - (a + u[:, None] * ab), axis=1))))
        r = np.asarray(rad)
        r = np.where(np.isfinite(r), r, np.nanmedian(r))
        out[name] = r[1:] if name == "thumb" else r              # thumb chain MCP -> IP -> tip = joints 2, 3
    return out


def capsule_layout(radii: dict, samples: int = 4) -> dict:
    """Sample layout along the 21-point chains: per sample (chain point a, chain point b, fraction u, finger id,
    radius) + the eligible pair mask (different fingers)."""
    rows = []
    for fi, (name, ch) in enumerate(CHAINS21.items()):
        r = radii[name]
        for s_ in range(len(ch) - 1):
            for u in np.linspace(0, 1, samples):
                if s_ > 0 and u == 0:
                    continue                                       # shared with the previous segment's end
                rows.append((ch[s_], ch[s_ + 1], float(u), fi, float(r[min(s_, len(r) - 1)])))
    a = np.array([x[0] for x in rows])
    b = np.array([x[1] for x in rows])
    u = np.array([x[2] for x in rows])
    fid = np.array([x[3] for x in rows])
    rad = np.array([x[4] for x in rows])
    return {"a": a, "b": b, "u": u, "finger": fid, "radius": rad, "pair": fid[:, None] != fid[None]}


def capsule_points(J21, lay: dict):
    """(K,21,3) hand points (numpy or torch) -> (K,n,3) capsule sample points."""
    u = lay["u"]
    if hasattr(J21, "detach"):
        import torch
        uu = torch.as_tensor(u, dtype=J21.dtype, device=J21.device)[None, :, None]
    else:
        uu = u[None, :, None]
    return J21[:, lay["a"]] * (1 - uu) + J21[:, lay["b"]] * uu


def spacing_ratio(J21: np.ndarray, lay: dict) -> np.ndarray:
    """(K,) per frame min over different-finger sample pairs of distance / (r_a + r_b) (< 1: capsules overlap)."""
    X = capsule_points(np.asarray(J21, float), lay)
    D = np.linalg.norm(X[:, :, None] - X[:, None], axis=-1)
    S = lay["radius"][:, None] + lay["radius"][None]
    R = np.where(lay["pair"][None], D / S[None], np.inf)
    return R.reshape(len(X), -1).min(1)
