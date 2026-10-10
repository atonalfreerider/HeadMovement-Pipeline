"""Self-calibration of a camera that PANS, TILTS and ZOOMS about a fixed centre (a tripod head with a zoom lens).

The two cameras of the carlos_aline take stand on tripods but the operators pan, tilt and zoom all the time: the
scene is NOT seen from a constant orientation or focal. For a camera rotating about its centre the static background
is explained by one rotation R_i and one focal f_i per frame (plus a lens that does not change): two frames i, j see a
background point at pixels p, q with  ray(q) ~ R_j R_i^T ray(p).  No depth enters, so no parallax is needed and the
solution is well conditioned as soon as the head moves.

Model (all pixels are image pixels, y down)
  K_i = [[f_i, 0, cx], [0, f_i, cy], [0, 0, 1]],  (cx, cy) = image centre + one shared offset (ox, oy),
  distortion in normalised coordinates (x_d = x_u (1 + k1 r_u^2 + k2 r_u^4)), shared by all frames,
  R_i = rotation reference frame -> camera i (frame 0 is the reference: R_0 = I).

  solve_ptz()     robust sparse Levenberg-Marquardt over the pairwise point transfer errors of every matched pair
  init_chain()    spanning-tree initialisation from the pairwise homographies (focal ratio from the local scale)
  refine_frame()  one more frame (a video frame between two solved ones) with everything else fixed

Pure numpy / scipy: no GPU, no files. The matching lives in dancecap.cameras.ptz_feats.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot


# ------------------------------------------------------------------------------------------------ camera model


@dataclass
class Lens:
    """The part of the camera model shared by all frames."""
    width: int
    height: int
    k1: float = 0.0
    k2: float = 0.0
    ox: float = 0.0   # principal point offset from the image centre, px
    oy: float = 0.0

    @property
    def c(self) -> np.ndarray:
        return np.array([self.width / 2.0 + self.ox, self.height / 2.0 + self.oy])


def distort(xu: np.ndarray, k1: float, k2: float) -> np.ndarray:
    r2 = (xu ** 2).sum(-1, keepdims=True)
    return xu * (1.0 + k1 * r2 + k2 * r2 * r2)


def undistort(xd: np.ndarray, k1: float, k2: float, iters: int = 8) -> np.ndarray:
    """Inverse of distort() by fixed-point iteration (exact for the mild distortion of a camera lens)."""
    xu = xd.copy()
    for _ in range(iters):
        r2 = (xu ** 2).sum(-1, keepdims=True)
        xu = xd / (1.0 + k1 * r2 + k2 * r2 * r2)
    return xu


def pix_to_ray(p: np.ndarray, f: np.ndarray, lens: Lens) -> np.ndarray:
    """(M,2) pixels, (M,) focal -> (M,3) unit rays in the camera frame (x right, y down, z forward)."""
    xd = (p - lens.c) / np.asarray(f, float)[..., None]
    xu = undistort(xd, lens.k1, lens.k2)
    d = np.concatenate([xu, np.ones(xu.shape[:-1] + (1,))], -1)
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def ray_to_pix(d: np.ndarray, f: np.ndarray, lens: Lens) -> np.ndarray:
    xu = d[..., :2] / d[..., 2:3]
    xd = distort(xu, lens.k1, lens.k2)
    return xd * np.asarray(f, float)[..., None] + lens.c


def rotations(rv: np.ndarray) -> np.ndarray:
    return Rot.from_rotvec(np.asarray(rv, float).reshape(-1, 3)).as_matrix()


# ------------------------------------------------------------------------------------------------ problem


@dataclass
class Pairs:
    """Point correspondences between frames: pixels p in frame i, q in frame j (all arrays of length M)."""
    i: np.ndarray
    j: np.ndarray
    p: np.ndarray
    q: np.ndarray

    def __len__(self) -> int:
        return len(self.i)


@dataclass
class PtzSolution:
    rv: np.ndarray            # (N,3) rotation vectors, reference -> camera i
    f: np.ndarray             # (N,) focal px
    lens: Lens
    rms_px: float = 0.0
    median_px: float = 0.0
    p90_px: float = 0.0
    n_resid: int = 0
    info: dict = field(default_factory=dict)

    @property
    def R(self) -> np.ndarray:
        return rotations(self.rv)


def transfer(sol_rv, sol_f, lens: Lens, pairs: Pairs) -> np.ndarray:
    """Pixel position in frame j predicted for every p of frame i (M,2)."""
    R = rotations(sol_rv)
    d = pix_to_ray(pairs.p, sol_f[pairs.i], lens)
    dj = np.einsum("mab,mb->ma", R[pairs.j], np.einsum("mba,mb->ma", R[pairs.i], d))
    ok = dj[:, 2] > 1e-6
    out = np.full((len(pairs), 2), 1e6)
    out[ok] = ray_to_pix(dj[ok], sol_f[pairs.j][ok], lens)
    return out


def _pack(rv, f, lens, free, tie=False):
    x = [rv[1:].ravel(), np.log(f[:1] if tie else f)]
    if "k1" in free:
        x.append([lens.k1])
    if "k2" in free:
        x.append([lens.k2])
    if "pp" in free:
        x.append([lens.ox, lens.oy])
    return np.concatenate([np.asarray(a, float).ravel() for a in x])


def _unpack(x, n, lens, free, tie=False):
    rv = np.zeros((n, 3))
    rv[1:] = x[:3 * (n - 1)].reshape(-1, 3)
    nf = 1 if tie else n
    f = np.exp(x[3 * (n - 1):3 * (n - 1) + nf])
    if tie:
        f = np.full(n, f[0])
    k = 3 * (n - 1) + nf
    lens = Lens(lens.width, lens.height, lens.k1, lens.k2, lens.ox, lens.oy)
    if "k1" in free:
        lens.k1 = float(x[k]); k += 1
    if "k2" in free:
        lens.k2 = float(x[k]); k += 1
    if "pp" in free:
        lens.ox, lens.oy = float(x[k]), float(x[k + 1]); k += 2
    return rv, f, lens


def _local_residual_fn():
    """torch function theta(12) -> pixel residual (2): theta = rv_i(3) lf_i rv_j(3) lf_j k1 k2 ox oy (p, q, size consts)."""
    import torch

    def rod(v):
        th = torch.sqrt((v * v).sum() + 1e-18)
        k = v / th
        K = torch.zeros(3, 3, dtype=v.dtype)
        K = torch.stack([torch.stack([torch.zeros_like(k[0]), -k[2], k[1]]),
                         torch.stack([k[2], torch.zeros_like(k[0]), -k[0]]),
                         torch.stack([-k[1], k[0], torch.zeros_like(k[0])])])
        return torch.eye(3, dtype=v.dtype) + torch.sin(th) * K + (1 - torch.cos(th)) * (K @ K)

    def fn(theta, p, q, wh):
        rvi, fi, rvj, fj = theta[0:3], torch.exp(theta[3]), theta[4:7], torch.exp(theta[7])
        k1, k2, ox, oy = theta[8], theta[9], theta[10], theta[11]
        c = torch.stack([wh[0] / 2 + ox, wh[1] / 2 + oy])
        xd = (p - c) / fi
        xu = xd
        for _ in range(8):
            r2 = (xu * xu).sum()
            xu = xd / (1 + k1 * r2 + k2 * r2 * r2)
        d = torch.cat([xu, torch.ones(1, dtype=xu.dtype)])
        dj = rod(rvj) @ (rod(rvi).T @ d)
        xj = dj[:2] / dj[2]
        r2 = (xj * xj).sum()
        pj = xj * (1 + k1 * r2 + k2 * r2 * r2) * fj + c
        return pj - q
    return fn


def _sparse_jac(x, n, lens0, free, pairs, w, fixed, x0, tie=False):
    """Analytic (autodiff) sparse Jacobian of the weighted point-transfer residuals."""
    import torch
    from scipy.sparse import coo_matrix
    xf = x0.copy()
    xf[~fixed] = x
    rv, f, lens = _unpack(xf, n, lens0, free, tie)
    M = len(pairs)
    th = np.concatenate([rv[pairs.i], np.log(f[pairs.i])[:, None], rv[pairs.j], np.log(f[pairs.j])[:, None],
                         np.tile([lens.k1, lens.k2, lens.ox, lens.oy], (M, 1))], 1)
    fn = _local_residual_fn()
    wh = torch.tensor([lens.width, lens.height], dtype=torch.float64)
    T = torch.tensor(th, dtype=torch.float64)
    Pp = torch.tensor(pairs.p, dtype=torch.float64)
    Qq = torch.tensor(pairs.q, dtype=torch.float64)
    J = torch.func.vmap(torch.func.jacrev(fn, argnums=0), in_dims=(0, 0, 0, None))(T, Pp, Qq, wh).numpy()  # (M,2,12)
    J = J * w[:, None, None]
    n_rv = 3 * (n - 1)
    nf = 1 if tie else n
    cols_i = np.concatenate([np.where(pairs.i[:, None] > 0, 3 * (pairs.i[:, None] - 1) + np.arange(3), -1),
                             (n_rv + (0 * pairs.i if tie else pairs.i))[:, None]], 1)
    cols_j = np.concatenate([np.where(pairs.j[:, None] > 0, 3 * (pairs.j[:, None] - 1) + np.arange(3), -1),
                             (n_rv + (0 * pairs.j if tie else pairs.j))[:, None]], 1)
    shared_names = [("k1", 8), ("k2", 9), ("pp", 10), ("pp", 11)]
    sh_cols, sh_src = [], []
    k = n_rv + nf
    for name in ("k1", "k2", "pp"):
        if name in free:
            if name == "pp":
                sh_cols += [k, k + 1]; sh_src += [10, 11]; k += 2
            else:
                sh_cols += [k]; sh_src += [8 if name == "k1" else 9]; k += 1
    rows, cols, vals = [], [], []
    mr = np.arange(M)
    for r in range(2):
        for src_cols, off in ((cols_i, 0), (cols_j, 4)):
            for a in range(4):
                cc = src_cols[:, a]
                ok = cc >= 0
                rows.append(2 * mr[ok] + r); cols.append(cc[ok]); vals.append(J[ok, r, off + a])
        for cc, s in zip(sh_cols, sh_src):
            rows.append(2 * mr + r); cols.append(np.full(M, cc)); vals.append(J[:, r, s])
    Jm = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(2 * M, len(x0))).tocsr()
    return Jm[:, np.where(~fixed)[0]]


def solve_ptz(n: int, pairs: Pairs, rv0: np.ndarray, f0: np.ndarray, lens0: Lens,
              free: tuple[str, ...] = ("k1",), loss: str = "soft_l1", f_scale: float = 1.0,
              max_nfev: int = 60, weights: np.ndarray | None = None,
              fixed_frames: tuple[int, ...] = (), tie_f: bool = False, fix_f: bool = False) -> PtzSolution:
    """Robust LM over all pairs (autodiff sparse Jacobian). Frame 0 is the rotation gauge. `free` picks the shared lens
    terms k1 / k2 / pp. `weights` (M,) multiplies the residuals of a correspondence (1 = normal)."""
    M = len(pairs)
    x0 = _pack(rv0, f0, lens0, free, tie_f)
    nx = len(x0)
    nf = 1 if tie_f else n
    w = np.ones(M) if weights is None else np.asarray(weights, float)
    fixed = np.zeros(nx, bool)
    for fr in fixed_frames:
        if fr > 0:
            fixed[3 * (fr - 1):3 * fr] = True
        fixed[3 * (n - 1) + (0 if tie_f else fr)] = True
    if fix_f:
        fixed[3 * (n - 1):3 * (n - 1) + nf] = True      # the focal(s) stay at f0: rotations (and lens) only

    def fun(xv):
        x = x0.copy()
        x[~fixed] = xv
        rv, f, lens = _unpack(x, n, lens0, free, tie_f)
        return ((transfer(rv, f, lens, pairs) - pairs.q) * w[:, None]).ravel()

    def jac(xv):
        return _sparse_jac(xv, n, lens0, free, pairs, w, fixed, x0, tie_f)

    r = least_squares(fun, x0[~fixed], jac=jac, loss=loss, f_scale=f_scale, x_scale="jac", method="trf",
                      max_nfev=max_nfev, tr_solver="lsmr")
    x = x0.copy()
    x[~fixed] = r.x
    rv, f, lens = _unpack(x, n, lens0, free, tie_f)
    e = np.linalg.norm(transfer(rv, f, lens, pairs) - pairs.q, axis=1)
    return PtzSolution(rv, f, lens, rms_px=float(np.sqrt((e ** 2).mean())), median_px=float(np.median(e)),
                       p90_px=float(np.percentile(e, 90)), n_resid=M,
                       info={"cost": float(r.cost), "nfev": int(r.nfev), "status": int(r.status), "free": list(free)})


def reject_outliers(sol: PtzSolution, pairs: Pairs, thresh_px: float) -> np.ndarray:
    e = np.linalg.norm(transfer(sol.rv, sol.f, sol.lens, pairs) - pairs.q, axis=1)
    return e < thresh_px


# ------------------------------------------------------------------------------------------------ initialisation


def local_scale(H: np.ndarray, at: np.ndarray) -> float:
    """sqrt(det of the Jacobian of the homography) at pixel `at`: the focal ratio f_j / f_i for a rotation + zoom."""
    x, y = at
    w = H[2, 0] * x + H[2, 1] * y + H[2, 2]
    u = (H[0, 0] * x + H[0, 1] * y + H[0, 2]) / w
    v = (H[1, 0] * x + H[1, 1] * y + H[1, 2]) / w
    J = np.array([[(H[0, 0] - u * H[2, 0]) / w, (H[0, 1] - u * H[2, 1]) / w],
                  [(H[1, 0] - v * H[2, 0]) / w, (H[1, 1] - v * H[2, 1]) / w]])
    return float(np.sqrt(abs(np.linalg.det(J))))


def homography_rotation(H: np.ndarray, f_i: float, f_j: float, c: np.ndarray) -> np.ndarray:
    """Nearest rotation of K_j^-1 H K_i (pure-rotation homography i -> j)."""
    Ki = np.array([[f_i, 0, c[0]], [0, f_i, c[1]], [0, 0, 1.0]])
    Kj = np.array([[f_j, 0, c[0]], [0, f_j, c[1]], [0, 0, 1.0]])
    M = np.linalg.inv(Kj) @ H @ Ki
    M = M / np.cbrt(np.linalg.det(M))
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = -R
    return R


def init_chain(n: int, edges: dict[tuple[int, int], np.ndarray], lens: Lens, f_start: float,
               root: int = 0) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Initial (rv, f) of every frame reachable from `root` through the homography edges {(i, j): H_ij}.
    Breadth first over the strongest edges; returns rv (n,3), f (n,), the frames that were reached."""
    adj: dict[int, list[tuple[int, np.ndarray, bool]]] = {k: [] for k in range(n)}
    for (i, j), H in edges.items():
        adj[i].append((j, H, False))
        adj[j].append((i, np.linalg.inv(H), True))
    R = [None] * n
    f = np.full(n, np.nan)
    R[root], f[root] = np.eye(3), f_start
    order, seen = [root], {root}
    k = 0
    c = lens.c
    while k < len(order):
        i = order[k]
        k += 1
        for j, H, _ in sorted(adj[i], key=lambda e: abs(e[0] - i)):
            if j in seen:
                continue
            s = local_scale(H, c)
            fj = f[i] * s
            Rij = homography_rotation(H, f[i], fj, c)
            R[j], f[j] = Rij @ R[i], fj
            seen.add(j)
            order.append(j)
    rv = np.zeros((n, 3))
    for j in order:
        rv[j] = Rot.from_matrix(R[j]).as_rotvec()
    return rv, np.where(np.isnan(f), f_start, f), order


def refine_frame(sol: PtzSolution, ref_frames: list[int], p_new: list[np.ndarray], q_ref: list[np.ndarray],
                 rv_guess: np.ndarray, f_guess: float, f_scale: float = 1.0,
                 fix_f: float | None = None) -> tuple[np.ndarray, float, float]:
    """Solve (rotation, focal) of one new frame from its matches p_new (pixels in the NEW frame) to the pixels q_ref in
    the already solved frames `ref_frames`; everything else fixed. Returns (rotvec, focal, rms px)."""
    lens = sol.lens
    R_ref = rotations(sol.rv[ref_frames])
    f_ref = sol.f[ref_frames]

    def fun(x):
        Rn = Rot.from_rotvec(x[:3]).as_matrix()
        fn = fix_f if fix_f else np.exp(x[3])
        out = []
        for k in range(len(ref_frames)):
            d = pix_to_ray(p_new[k], np.full(len(p_new[k]), fn), lens)
            dj = (R_ref[k] @ Rn.T @ d.T).T
            ok = dj[:, 2] > 1e-6
            pr = np.full((len(d), 2), 1e6)
            pr[ok] = ray_to_pix(dj[ok], np.full(ok.sum(), f_ref[k]), lens)
            out.append((pr - q_ref[k]).ravel())
        return np.concatenate(out)

    r = least_squares(fun, np.concatenate([rv_guess, [np.log(f_guess)]]), loss="soft_l1", f_scale=f_scale,
                      x_scale=np.array([0.01, 0.01, 0.01, 0.01 if not fix_f else 1e6]), method="trf", max_nfev=40)
    e = fun(r.x).reshape(-1, 2)
    rms = float(np.sqrt((e ** 2).sum(1).mean()))
    return r.x[:3], float(fix_f if fix_f else np.exp(r.x[3])), rms


def ref_errors(sol: PtzSolution, ref_frames: list[int], p_new: list[np.ndarray], q_ref: list[np.ndarray], rv: np.ndarray, f: float) -> np.ndarray:
    """Median reprojection error (px) of the new frame's matches against each reference frame, for the solved (rv, f)."""
    lens = sol.lens
    Rn = Rot.from_rotvec(rv).as_matrix()
    R_ref = rotations(sol.rv[ref_frames])
    out = []
    for k in range(len(ref_frames)):
        d = pix_to_ray(p_new[k], np.full(len(p_new[k]), f), lens)
        dj = (R_ref[k] @ Rn.T @ d.T).T
        ok = dj[:, 2] > 1e-6
        pr = np.full((len(d), 2), 1e6)
        pr[ok] = ray_to_pix(dj[ok], np.full(ok.sum(), sol.f[ref_frames[k]]), lens)
        out.append(float(np.median(np.linalg.norm(pr - q_ref[k], axis=1))))
    return np.array(out)


def refine_frame_robust(sol: PtzSolution, ref_frames: list[int], p_new: list[np.ndarray], q_ref: list[np.ndarray],
                        rv_guess: np.ndarray, f_guess: float, fix_f: float | None = None, bad_px: float = 4.0,
                        bad_ratio: float = 2.5) -> tuple[np.ndarray, float, float, list[int]]:
    """refine_frame(), then drop the reference frames whose own pose disagrees with the others (a reference frame of the
    skeleton that is a little off pulls the frame by its error): a reference with median error > bad_px and > bad_ratio x the best
    one is dropped and the frame re-solved with the rest. Returns (rv, f, rms, used reference frames)."""
    rv, f, rms = refine_frame(sol, ref_frames, p_new, q_ref, rv_guess, f_guess, fix_f=fix_f)
    refs = list(ref_frames)
    pn, qr = list(p_new), list(q_ref)
    for _ in range(2):
        if len(refs) < 2:
            break
        e = ref_errors(sol, refs, pn, qr, rv, f)
        worst = int(np.argmax(e))
        if e[worst] > bad_px and e[worst] > bad_ratio * max(np.min(e), 0.5):
            refs.pop(worst); pn.pop(worst); qr.pop(worst)
            rv, f, rms = refine_frame(sol, refs, pn, qr, rv, f, fix_f=fix_f)
        else:
            break
    return rv, f, rms, refs


# ------------------------------------------------------------------------------------------------ helpers


def fov_deg(f: np.ndarray, width: int) -> np.ndarray:
    return np.degrees(2 * np.arctan(width / 2.0 / np.asarray(f, float)))


def pan_tilt_roll(rv: np.ndarray) -> np.ndarray:
    """Euler angles (deg) yaw (about y), pitch (about x), roll (about z) of the camera-from-reference rotations: only
    for human-readable summaries."""
    return Rot.from_rotvec(np.asarray(rv, float).reshape(-1, 3)).as_euler("YXZ", degrees=True)


def balance_pairs(i: np.ndarray, j: np.ndarray, p: np.ndarray, q: np.ndarray, size: tuple[int, int],
                  per_pair: int = 120, grid: int = 8, seed: int = 0) -> Pairs:
    """Keep at most `per_pair` correspondences per frame pair, spread over a grid x grid raster of the first image."""
    rng = np.random.default_rng(seed)
    keep = []
    w, h = size
    key = np.stack([i, j], 1)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    for g in np.unique(inv):
        idx = np.where(inv == g)[0]
        if len(idx) <= per_pair:
            keep.extend(idx.tolist())
            continue
        cell = (np.clip((p[idx, 0] / w * grid).astype(int), 0, grid - 1) * grid
                + np.clip((p[idx, 1] / h * grid).astype(int), 0, grid - 1))
        order = rng.permutation(len(idx))
        counts: dict[int, int] = {}
        cap = max(1, per_pair // (grid * grid) + 1)
        chosen = []
        for o in order:
            c = int(cell[o])
            if counts.get(c, 0) < cap:
                counts[c] = counts.get(c, 0) + 1
                chosen.append(idx[o])
        if len(chosen) < per_pair:
            rest = [idx[o] for o in order if idx[o] not in set(chosen)]
            chosen += rest[:per_pair - len(chosen)]
        keep.extend(chosen[:per_pair])
    keep = np.array(sorted(keep), int)
    return Pairs(i[keep], j[keep], p[keep], q[keep])


def homography_edge(p: np.ndarray, q: np.ndarray, thresh: float = 4.0) -> tuple[np.ndarray | None, np.ndarray]:
    """RANSAC homography p -> q and the inlier mask."""
    if len(p) < 12:
        return None, np.zeros(len(p), bool)
    H, m = cv2.findHomography(p.astype(np.float64), q.astype(np.float64), cv2.USAC_MAGSAC, thresh)
    if H is None:
        return None, np.zeros(len(p), bool)
    return H, m.ravel().astype(bool)
