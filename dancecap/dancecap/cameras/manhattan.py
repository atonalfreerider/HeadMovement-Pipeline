"""Manhattan-frame (three orthogonal families of parallel lines) fit on line-plane normals carried into one reference frame.

The camera pans / tilts / zooms about a fixed centre (dancecap.cameras.ptz), so every frame's rotation R_i (reference -> camera i) and
focal f_i carry its line-plane normals into the reference frame: n_ref = R_i^T n_i. In an architectural room (a hall: columns, walls,
trusses, floor boards) the normals fall in three orthogonal families; the best orthogonal frame, and how tightly the lines obey it, depend on
the focal that was used to build the rays: a wrong focal (a wrong scale s of every f_i) smears the families. Scanning s gives the focal
EVIDENCE the rotation-only track cannot (the track itself is flat in f), and the fitted frame gives gravity ("vertical") and the room axes.

    fit_frame(normals, weights)  -> Frame(M (3,3 columns = vertical, horizontal a, horizontal b in the reference frame), cost, inlier shares)

Pure numpy.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Frame:
    M: np.ndarray            # (3,3) columns: vertical (v), horizontal a, horizontal b (reference frame)
    cost: float              # robust cost (weighted, per unit weight)
    share_v: float           # weight share of the lines consistent with the vertical
    share_a: float
    share_b: float
    tau_deg: float


def _rho(x: np.ndarray, tau: float) -> np.ndarray:
    return np.minimum((x / tau) ** 2, 1.0)


def rodrigues(axis: np.ndarray, ang: float) -> np.ndarray:
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def fit_vertical(n: np.ndarray, w: np.ndarray, prior: np.ndarray, tau: float, cone_deg: float = 20.0, step_deg: float = 0.5) -> tuple[np.ndarray, float]:
    """Direction v (unit, near `prior`) minimising sum w * rho(n . v): grid in a cone, then local refinement."""
    prior = prior / np.linalg.norm(prior)
    e1 = np.cross(prior, [1.0, 0, 0]) if abs(prior[0]) < 0.9 else np.cross(prior, [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(prior, e1)
    best = (None, np.inf)
    g = np.radians(np.arange(-cone_deg, cone_deg + 1e-9, step_deg))
    for a in g:
        for b in g:
            v = prior + np.tan(a) * e1 + np.tan(b) * e2
            v /= np.linalg.norm(v)
            c = (w * _rho(np.abs(n @ v), tau)).sum()
            if c < best[1]:
                best = (v, c)
    v, c = best
    for st in (0.25, 0.1, 0.05):
        improved = True
        while improved:
            improved = False
            for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                u = v + np.tan(np.radians(st)) * (da * e1 + db * e2)
                u /= np.linalg.norm(u)
                cu = (w * _rho(np.abs(n @ u), tau)).sum()
                if cu < c - 1e-12:
                    v, c, improved = u, cu, True
    return v, c


def fit_azimuth(n: np.ndarray, w: np.ndarray, v: np.ndarray, tau_deg: float = 1.2, refine: bool = True) -> Frame:
    """Orthogonal frame (v, a, b) for a GIVEN vertical v: the horizontal azimuth about v by a scan of the normals that are not vertical-line
    normals: every horizontal line of direction a_phi or b_phi = v x a_phi has n . a = 0 or n . b = 0."""
    tau = np.sin(np.radians(tau_deg))
    v = v / np.linalg.norm(v)
    e1 = np.cross(v, [1.0, 0, 0]) if abs(v[0]) < 0.9 else np.cross(v, [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(v, e1)
    nv = np.abs(n @ v)
    vert = nv < tau * 1.5
    nh = n[~vert]
    wh = w[~vert]
    phis = np.radians(np.arange(0, 180, 0.25))
    A = np.cos(phis)[:, None] * e1 + np.sin(phis)[:, None] * e2     # candidate direction a (P,3)
    B = np.cross(v, A)                                              # b = v x a
    ca = np.abs(nh @ A.T)                                           # (N,P)
    cb = np.abs(nh @ B.T)
    cost = (wh[:, None] * _rho(np.minimum(ca, cb), tau)).sum(0)
    k = int(np.argmin(cost))
    a = A[k]
    if refine:
        for st in (0.1, 0.05, 0.02):
            for d in (-st, st):
                for _ in range(40):
                    a2 = rodrigues(v, np.radians(d)) @ a
                    b2 = np.cross(v, a2)
                    c2 = (wh * _rho(np.minimum(np.abs(nh @ a2), np.abs(nh @ b2)), tau)).sum()
                    c1 = (wh * _rho(np.minimum(np.abs(nh @ a), np.abs(nh @ np.cross(v, a))), tau)).sum()
                    if c2 < c1 - 1e-12:
                        a = a2
                    else:
                        break
    b = np.cross(v, a)
    M = np.stack([v, a, b], 1)
    # a normal of a horizontal line of direction a is perpendicular to a: it contributes to family a
    in_a = np.abs(n @ a) < tau * 1.5
    in_b = np.abs(n @ b) < tau * 1.5
    in_v = np.abs(n @ v) < tau * 1.5
    tot = w.sum()
    total_cost = float((w * _rho(np.minimum(np.minimum(np.abs(n @ v), np.abs(n @ a)), np.abs(n @ b)), tau)).sum() / tot)
    return Frame(M, total_cost, float(w[in_v].sum() / tot), float(w[in_a & ~in_v].sum() / tot), float(w[in_b & ~in_v & ~in_a].sum() / tot), tau_deg)


def fit_frame(n: np.ndarray, w: np.ndarray, prior_up: np.ndarray, tau_deg: float = 1.2, refine: bool = True) -> Frame:
    """Orthogonal frame (v, a, b) of the line-plane normals n (N,3 unit, reference frame) with weights w.
    v from a near-`prior_up` search over all normals (verticals are by far the strongest family of a hall), then the horizontal
    azimuth about v (fit_azimuth)."""
    tau = np.sin(np.radians(tau_deg))
    v, _ = fit_vertical(n, w, prior_up, tau)
    return fit_azimuth(n, w, v, tau_deg, refine)
