"""HMMs of the "sides" decisions (DESIGN 2.3, 2.4): per relation a sign HMM {-1, 0, +1} (R4: undecided / front / flank / back) whose
direct flips are (almost) forbidden inside the entangled band ("decide on approach, hold while entangled"), and the per-dancer yaw HMM.

Ported (pure, no take state): zouk_contacts.viterbi / _min_runs (constant transitions; here also per-frame), jl_facing.hmm_forward_backward
(here with per-frame transition matrices). The decision rule: forward-backward posterior >= post_decide to enter a side, Schmitt exit
below post_exit, the Viterbi path's side, runs shorter than the minimum run -> undecided.
"""
from __future__ import annotations

import numpy as np


# ------------------------------------------------------------------------------------------------ generic

def viterbi(score: np.ndarray, trans: np.ndarray) -> np.ndarray:
    """score (n,S) log-likelihoods, trans (S,S) costs (-log p) or (n,S,S) per frame (row a -> col b, frame i uses trans[i]) -> path (n,)"""
    score = np.asarray(score, float)
    n, S = score.shape
    tv = np.asarray(trans, float)
    per = tv.ndim == 3
    dp = score[0].copy()
    bp = np.zeros((n, S), np.int32)
    for i in range(1, n):
        cand = dp[:, None] - (tv[i] if per else tv)
        bp[i] = cand.argmax(0)
        dp = cand.max(0) + score[i]
    path = np.zeros(n, np.int32)
    path[-1] = int(dp.argmax())
    for i in range(n - 1, 0, -1):
        path[i - 1] = bp[i, path[i]]
    return path


def forward_backward(lik: np.ndarray, trans: np.ndarray, prior: np.ndarray | None = None, log: bool = False) -> np.ndarray:
    """posterior marginals (n,S) from per-frame likelihoods (n,S) (or log-likelihoods when log=True) and transition probabilities
    (S,S) or (n,S,S) (frame i's matrix moves i-1 -> i); scaled, as jl_facing.hmm_forward_backward"""
    L = np.asarray(lik, float)
    if log:
        L = np.exp(L - L.max(1, keepdims=True))
    n, S = L.shape
    tv = np.asarray(trans, float)
    per = tv.ndim == 3
    pi = np.full(S, 1.0 / S) if prior is None else np.asarray(prior, float)
    a = np.zeros((n, S))
    b = np.ones((n, S))
    x = pi * L[0]
    a[0] = x / max(x.sum(), 1e-300)
    for t in range(1, n):
        x = (a[t - 1] @ (tv[t] if per else tv)) * L[t]
        a[t] = x / max(x.sum(), 1e-300)
    for t in range(n - 2, -1, -1):
        x = (tv[t + 1] if per else tv) @ (L[t + 1] * b[t + 1])
        b[t] = x / max(x.sum(), 1e-300)
    p = a * b
    return p / np.maximum(p.sum(1, keepdims=True), 1e-300)


def runs(mask: np.ndarray, merge: int = 0, min_len: int = 1) -> list[tuple[int, int]]:
    """inclusive (start, end) runs of True, merging runs separated by <= merge False frames"""
    idx = np.flatnonzero(np.asarray(mask, bool))
    if not len(idx):
        return []
    out = []
    a = b = int(idx[0])
    for k in idx[1:]:
        k = int(k)
        if k - b - 1 <= merge:
            b = k
        else:
            out.append((a, b))
            a = b = k
    out.append((a, b))
    return [(a, b) for a, b in out if b - a + 1 >= min_len]


def label_runs(lab: np.ndarray) -> list[tuple[int, int, int]]:
    """[(start, end inclusive, label)] of a label sequence"""
    lab = np.asarray(lab)
    out = []
    i = 0
    n = len(lab)
    while i < n:
        j = i
        while j + 1 < n and lab[j + 1] == lab[i]:
            j += 1
        out.append((i, j, int(lab[i])))
        i = j + 1
    return out


def min_runs(lab: np.ndarray, n: int, fill: int = 0) -> np.ndarray:
    """runs of a non-`fill` label shorter than n frames -> fill (a decided side must last; DESIGN minimum run)"""
    lab = np.asarray(lab).copy()
    for a, b, v in label_runs(lab):
        if v != fill and b - a + 1 < n:
            lab[a:b + 1] = fill
    return lab


def min_runs_merge(lab: np.ndarray, n: int, keep=()) -> np.ndarray:
    """zouk_contacts._min_runs: runs shorter than n take the label of the longer neighbouring run (labels in keep never change)"""
    from .. import zouk_contacts as zc
    return zc._min_runs(np.asarray(lab), n, keep)


# ------------------------------------------------------------------------------------------------ the sign HMMs

def sign_transitions(cfg, n_states: int = 3, through_pairs=None) -> tuple[np.ndarray, np.ndarray]:
    """(outside the band, inside the band) transition probability matrices. State 0 = undecided; states 1.. = the sides / classes.
    through_pairs: pairs of side states whose direct switch inside the band is a pass-through (default: all pairs)."""
    S = n_states
    out = np.zeros((S, S))
    k = S - 1
    out[0, 0] = 1.0 - cfg.p_enter * k
    out[0, 1:] = cfg.p_enter
    for i in range(1, S):
        out[i, 0] = cfg.p_undecided
        for j in range(1, S):
            out[i, j] = cfg.p_flip / max(1, k - 1) if i != j else 0.0
        out[i, i] = 1.0 - out[i].sum()
    band = out.copy()
    pairs = through_pairs if through_pairs is not None else [(i, j) for i in range(1, S) for j in range(1, S) if i != j]
    for i, j in pairs:
        band[i, j] = cfg.p_flip_band
    for i in range(1, S):
        band[i, i] = 0.0
        band[i, i] = 1.0 - band[i].sum()
    return out, band


def per_frame(band: np.ndarray, out_m: np.ndarray, band_m: np.ndarray) -> np.ndarray:
    """(n,S,S) per-frame matrices: frame i (the move i-1 -> i) uses the band matrix where band[i] or band[i-1]"""
    b = np.asarray(band, bool)
    bb = b.copy()
    bb[1:] |= b[:-1]
    return np.where(bb[:, None, None], band_m[None], out_m[None])


def sign_emissions(L: np.ndarray, gate: np.ndarray, side_cost: float) -> np.ndarray:
    """log-likelihoods (n,3) of states (undecided, -1, +1) from log-odds L (n,) (+ favours +1): a side costs side_cost against undecided
    and gains half its log-odds; outside the gate the relation is undefined (sides impossible)"""
    L = np.nan_to_num(np.asarray(L, float))
    e = np.zeros((len(L), 3))
    e[:, 1] = -0.5 * L - side_cost
    e[:, 2] = 0.5 * L - side_cost
    g = np.asarray(gate, bool)
    e[~g, 1:] = -50.0
    return e


def class_emissions(Lc: np.ndarray, gate: np.ndarray, side_cost: float) -> np.ndarray:
    """log-likelihoods (n,1+K) of (undecided, class 1..K) from per-class log-likelihoods Lc (n,K) (centred over the classes)"""
    Lc = np.nan_to_num(np.asarray(Lc, float))
    Lc = Lc - Lc.mean(1, keepdims=True)
    e = np.zeros((len(Lc), Lc.shape[1] + 1))
    e[:, 1:] = Lc - side_cost
    e[~np.asarray(gate, bool), 1:] = -50.0
    return e


def decide_states(loglik: np.ndarray, trans: np.ndarray, cfg, hz: float = 30.0) -> dict:
    """the decision rule on one relation: Viterbi path + forward-backward posterior; enter a side at post >= post_decide, Schmitt exit
    below post_exit, minimum run -> {"state" (n,) int (0 = undecided), "post" (n,S), "path" (n,)}"""
    loglik = np.asarray(loglik, float)
    tv = np.asarray(trans, float)
    cost = -np.log(np.maximum(tv, 1e-300))
    path = viterbi(loglik, cost)
    post = forward_backward(loglik, tv, log=True)
    st = schmitt(path, post, cfg.post_decide, cfg.post_exit)
    st = min_runs(st, max(1, int(round(cfg.min_run_s * hz))), 0)
    return {"state": st, "post": post, "path": path}


def schmitt(path: np.ndarray, post: np.ndarray, enter: float, exit_: float) -> np.ndarray:
    """decided state per frame: the Viterbi state when its posterior reaches `enter`; kept while the path stays on it and its posterior
    stays >= `exit_` (Schmitt hysteresis); 0 otherwise"""
    n = len(path)
    st = np.zeros(n, np.int32)
    cur = 0
    for i in range(n):
        p = int(path[i])
        if cur != 0 and p == cur and post[i, cur] >= exit_:
            st[i] = cur
            continue
        cur = 0
        if p != 0 and post[i, p] >= enter:
            cur = p
        st[i] = cur
    return st


def decide_sign(L: np.ndarray, gate: np.ndarray, band: np.ndarray, cfg, hz: float = 30.0) -> dict:
    """3-state sign HMM -> {"s" (n,) int8 in {-1, 0, +1}, "post" (n,3) [undecided, -, +], "path"}"""
    out_m, band_m = sign_transitions(cfg, 3)
    tr = per_frame(band, out_m, band_m)
    r = decide_states(sign_emissions(L, gate, cfg.side_cost), tr, cfg, hz)
    s = np.zeros(len(L), np.int8)
    s[r["state"] == 1] = -1
    s[r["state"] == 2] = 1
    return {"s": s, "post": r["post"], "path": r["path"]}


R4_CLASSES = ("front", "flank", "back")


def four_state_r4(Lc: np.ndarray, gate: np.ndarray, band: np.ndarray, cfg, hz: float = 30.0) -> dict:
    """R4: states undecided / front / flank / back; a direct front <-> back switch while the palm is on her (band) goes through her
    (1e-6); front <-> flank <-> back are moves along her surface -> {"cls" (n,) int8 (0 undecided, 1 front, 2 flank, 3 back), "post"}"""
    out_m, band_m = sign_transitions(cfg, 4, through_pairs=[(1, 3), (3, 1)])
    tr = per_frame(band, out_m, band_m)
    r = decide_states(class_emissions(Lc, gate, cfg.side_cost), tr, cfg, hz)
    return {"cls": r["state"].astype(np.int8), "post": r["post"], "path": r["path"]}


def three_state(L, gate, band, cfg, hz: float = 30.0) -> dict:
    return decide_sign(L, gate, band, cfg, hz)


# ------------------------------------------------------------------------------------------------ the yaw HMM (DESIGN 2.4)

def _vm_log(d_deg: np.ndarray, sigma_deg: float) -> np.ndarray:
    """log of a von Mises density (kappa = 1 / sigma^2) of an angle difference (deg), up to a constant"""
    k = 1.0 / np.radians(sigma_deg) ** 2
    return k * (np.cos(np.radians(d_deg)) - 1.0)


class YawTransitions:
    """per-frame transition probabilities between heading bins (built on the fly: (n,B,B) would not fit in memory): a mixture of a von
    Mises around the base's own angular velocity and a wide one around 0; moves beyond yaw_max_dps (+ one bin) are impossible"""

    def __init__(self, omega_dps: np.ndarray, cfg, hz: float = 30.0, B: int | None = None):
        self.B = int(B or cfg.yaw_bins)
        step = 360.0 / self.B
        c = np.arange(self.B) * step
        self.D = (c[None, :] - c[:, None] + 180.0) % 360.0 - 180.0          # move from bin a (rows) to b (cols)
        self.lim = cfg.yaw_max_dps / hz + step
        self.om = np.nan_to_num(np.asarray(omega_dps, float)) / hz
        self.w = cfg.yaw_trans_wide_w
        self.sig = cfg.yaw_trans_sigma_deg
        self.wide = np.exp(_vm_log(self.D, cfg.yaw_trans_wide_deg))
        self.wide[np.abs(self.D) > self.lim] = 0.0

    def p(self, i: int) -> np.ndarray:
        nar = np.exp(_vm_log(self.D - self.om[i], self.sig))
        nar[np.abs(self.D) > self.lim] = 0.0
        p = (1 - self.w) * nar + self.w * self.wide
        return p / np.maximum(p.sum(1, keepdims=True), 1e-300)


def yaw_viterbi(loglik: np.ndarray, omega_dps: np.ndarray, cfg, hz: float = 30.0, psi0: float | None = None) -> dict:
    """loglik (n, B) emission log-likelihoods over heading bins (bin b = b x 360/B deg), omega the base's angular velocity (deg/s)
    -> {"psi" (n,) unwrapped deg (Viterbi bins, refined by the local posterior mean), "bins", "conf" (posterior mass within +-15 deg)}"""
    loglik = np.asarray(loglik, float)
    n, B = loglik.shape
    step = 360.0 / B
    tr = YawTransitions(omega_dps, cfg, hz, B)
    # Viterbi
    dp = loglik[0].copy()
    bp = np.zeros((n, B), np.int16)
    for i in range(1, n):
        cand = dp[:, None] + np.log(np.maximum(tr.p(i), 1e-300))
        bp[i] = cand.argmax(0)
        dp = cand.max(0) + loglik[i]
    path = np.zeros(n, np.int32)
    path[-1] = int(dp.argmax())
    for i in range(n - 1, 0, -1):
        path[i - 1] = bp[i, path[i]]
    # forward-backward
    L = np.exp(loglik - loglik.max(1, keepdims=True))
    a = np.zeros((n, B))
    b = np.ones((n, B))
    x = L[0] / B
    a[0] = x / max(x.sum(), 1e-300)
    for t in range(1, n):
        x = (a[t - 1] @ tr.p(t)) * L[t]
        a[t] = x / max(x.sum(), 1e-300)
    for t in range(n - 2, -1, -1):
        x = tr.p(t + 1) @ (L[t + 1] * b[t + 1])
        b[t] = x / max(x.sum(), 1e-300)
    post = a * b
    post /= np.maximum(post.sum(1, keepdims=True), 1e-300)
    c = np.arange(B) * step
    n = len(path)
    # local refinement: circular mean of the posterior within +-15 deg of the Viterbi bin
    D = (c[None, :] - c[path][:, None] + 180.0) % 360.0 - 180.0
    m = np.abs(D) <= 15.0 + 1e-9
    w = post * m
    conf = w.sum(1)
    off = (w * D).sum(1) / np.maximum(conf, 1e-12)
    ang = c[path] + off
    d = (np.diff(ang) + 180.0) % 360.0 - 180.0
    start = ang[0] if psi0 is None else psi0 + ((ang[0] - psi0 + 180.0) % 360.0 - 180.0)
    psi = np.concatenate([[start], start + np.cumsum(d)])
    return {"psi": psi, "bins": path, "conf": conf, "post_max": post.max(1)}
