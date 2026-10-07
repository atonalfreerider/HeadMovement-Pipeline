"""Unlabelled candidate move boundaries -> work/<take>/moves/proposals.json.

    python -m dancecap.moves proposals <take> [--threshold 0.25] [--half-width 8] [--snap half|measure|eighth]

PROPOSALS, NOT CLASSIFICATIONS. A proposal says "the couple's motion changes character here" - never which move is
danced (move: null). They are shown to the narrator first (docs/MOVES_RESEARCH.md L2, F3-F6) so the human labels
start from sensible boundaries; they are never written to labels.json.

Method (Stage A: numpy only, CPU, no training)
  0. Masking: an eighth is analysed when >= min_valid_frac (50 %) of its frames were pooled (features.py: finite fits
     inside a run, fit quality > min_quality). Masked eighths (tracking dropouts, bad fits) and missing grid eighths
     split the take into BLOCKS of consecutive analysed eighths; every step below stays inside a block and the block
     edges become 'gap' boundaries (with the hole [t0, t1] and the reason). A dropout therefore costs the proposals
     only its own neighbourhood - it never empties them.
  1. features.npz per-eighth pooled means, robust-z-scored per take (features.py; the z floors keep a homogeneous take
     from being blown up). 'novelty' features are used (beat-phase and raw contact columns are excluded); a feature
     is dropped only when NaN on more than half of the analysed eighths (cb_* without counterbalance.json), other
     NaN cells are skipped by the NaN-aware similarity. z clipped to +-6.
  2. Self-similarity of the eighths: S_ij = exp(-d_ij^2 / (2 h^2)), d_ij = RMS z difference over the features finite
     in both eighths (h = 1 robust sd).
  3. Foote novelty at every eighth boundary b: mean similarity WITHIN the w eighths before and the w after (diagonal
     excluded, same block only) minus the mean similarity ACROSS b, Gaussian-tapered (sd w/2). 0 = no change, 1 = two
     perfectly distinct homogeneous blocks. w = 8 eighths (one measure: the beat-periodic within-measure motion
     averages out) when >= 32 eighths are analysed, else max(2, N // 4). Computed for all features together AND per
     feature group (couple relation, lead, follow, counterbalance; groups with >= 3 usable features): boundary
     strength = the maximum, so a change confined to one dancer (e.g. the follower's head and spins) is not diluted
     by the unchanged rest. On the homogeneous synthetic take every curve stays <= 0.14; at synthetic_cb's
     transitions every curve reaches >= 0.62. Known limit: z is relative to the take's own spread (median / MAD), so
     a take split ~50/50 between two very different motions gives a weaker boundary (~0.2 when a third of a group's
     features change) - lower --threshold there; a first-difference noise scale was tried and rejected (it lifts
     the homogeneous synthetic to 0.30).
  4. Peaks >= threshold are snapped to the nearest type-1 eighth (slots 0 and 4 = half-measures, F4) of their block,
     suppressed to >= min_eighths (4) apart and >= min_eighths from the block's edges (no stub segments). Peaks that
     are not proposed are listed in rejected_peaks with the reason (edge | no snap target | merged | spacing).
     Calibration: boundaries below `strong` (0.4) are flagged weak - ~1 cm of slow per-joint wobble on the
     homogeneous synthetic produces false boundaries of 0.28-0.37 (tests: the noise calibration case), the true
     synthetic_cb transitions are ~0.73. novelty.noise_floor = the median strength; margin = strength - noise_floor.
  5. Segments between the boundaries (per block): energy (couple kinetic energy, J/kg) -> energy_rank = its
     percentile among the take's analysed eighths (F5: ranked within the take until labels calibrate it) and
     energy_abs = log scale between energy_abs_range; energy = rank (default) or abs, mapped to the graph's
     energy_bands (taxonomy/zouk_graph.json) - with rank, 'grounded' means the calmest part of THIS take.
     cluster = leader clustering of segment mean z-vectors (RMS distance < repeat_rms joins the nearest cluster);
     repeat_of = the nearest earlier segment within repeat_rms. top_changes = the features that change most at the
     boundary (z units) - a hint for the narrator, not an explanation.
  6. run() adds `stale` (features.staleness: the fits / counterbalance / timing / beat grid changed since
     features.npz was computed) and the beat-grid span the proposals do not cover (no SMPL-X fits there) as warnings.

Truth check (synthetic takes): work/<take>/truth.json -> transitions (synthetic_cb: ordinary -> counterbalance pivot
-> ordinary; synthetic: none, a homogeneous dance). `evaluation` reports hits / misses / false positives with a
half-measure tolerance. Defaults were set so synthetic gives no boundary and synthetic_cb finds both transitions.

proposals.json (version 1): {version, kind:'proposals', take, time_base:'reference', provenance:'auto', note,
  method:{...}, source:{features, features_sha256_16, graph}, coverage:[t0, t1], grid_span:[t0, t1]|null,
  uncovered:[[t0, t1], ...], stale:[...], energy_bands:[...], energy_basis,
  analysis:{eighths, analysed_eighths, blocks:[[t0, t1]], masked:[{t0, t1, eighths, reason, valid_frac}]},
  novelty:{t:[boundary time per eighth boundary], strength:[...], groups:{name: [...]}, half_width_eighths,
           noise_floor},
  boundaries:[{t, eighth, slot, kind:'novelty'|'gap', strength, weak, margin, group, strength_by_group, peak_t,
               strength_at_snap, top_changes, (gap: gap:[t0, t1], reason)}],
  segments:[{id, t0, t1, move:null, node:null, tier:'couple', provenance:'auto', confidence, boundary_strength,
             end_strength, start/end:'edge'|'novelty'|'gap', n_eighths, partial, valid_frac, energy, energy_rank,
             energy_abs, energy_band, energy_label, cluster, repeat_of, summary:{...}, created}],
  rejected_peaks:[{t, eighth, strength, reason}], clusters:{...}, evaluation:{...}|null, warnings:[...]}
"""
from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import moves_dir, work_dir
from .features import FEATURES, INDEX, STATS, FeatureSet, features_paths

VERSION = 1
NOTE = ("PROPOSALS, NOT CLASSIFICATIONS: automatic candidate move boundaries from changes in the motion features. "
        "move is null on purpose - which move is danced comes from the user's narration / labels. Energy bands are "
        "estimates from kinetic energy ranked within this take.")
SUMMARY = ("couple_ke", "couple_speed", "com_distance", "couple_yaw_rate", "follow_yaw_rel", "follow_head_off_axis",
           "lead_com_height", "follow_com_height", "n_hand_holds", "hand_hand_min", "cb_score")


@dataclass
class ProposalConfig:
    half_width: int | None = None       # novelty kernel half width in eighths (None: 8, or max(2, N // 4) if N < 32)
    bandwidth: float = 1.0              # RBF similarity bandwidth, RMS robust-z units
    z_clip: float = 6.0
    min_group_features: int = 3         # per-group novelty only for groups with at least this many usable features
    threshold: float = 0.25             # boundary strength (0..1) to propose a boundary; synthetic (no move change)
                                        # peaks at ~0.14, synthetic_cb's two transitions at ~0.73
    strong: float = 0.4                 # boundaries below this are flagged weak: ~1 cm of slow joint wobble on the
                                        # homogeneous synthetic reaches 0.28-0.37 (the noise calibration case)
    min_eighths: int = 4                # minimum segment length (F4) and distance from the analysed range's edges
                                        # and from gaps
    snap: str = "half"                  # half (type-1 eighths: slots 0, 4) | measure (slot 0) | eighth
    stat: str = "mean"                  # pooled statistic used
    min_valid_frac: float = 0.5         # eighths with fewer valid (pooled) frames are masked -> a gap
    min_feature_coverage: float = 0.5   # a feature is dropped only when NaN on more than this share of the
                                        # analysed eighths (cb_* without counterbalance.json); else NaN-aware
    repeat_rms: float = 0.75            # segment mean z-vectors closer than this (RMS) = same cluster / a repeat
    energy: str = "rank"                # rank (F5) | abs
    energy_abs_range: tuple = (0.02, 2.0)   # J/kg mapped log-linearly to energy 0..1 (provisional)
    top_changes: int = 4


# ----------------------------------------------------------------------------------------------- masking

def _valid_frac(fs: FeatureSet) -> np.ndarray:
    from .features import eighth_valid_fraction
    return eighth_valid_fraction(fs)


def eighth_mask(fs: FeatureSet, cfg: ProposalConfig) -> tuple[np.ndarray, list[str | None]]:
    """(ok (N,) bool, reason per eighth or None): an eighth is analysed when >= min_valid_frac of its frames were
    pooled (features.py: finite fits inside a run, fit quality above min_quality) and some novelty feature is finite
    on it. Masked eighths are not dropped silently: they split the analysis into blocks and become 'gap' boundaries."""
    from .features import eighth_low_quality_fraction
    N = len(fs.eighths["t0"])
    vf = _valid_frac(fs)
    lowq = eighth_low_quality_fraction(fs)
    si = STATS.index(cfg.stat)
    nov = [INDEX[f.name] for f in FEATURES if f.novelty]
    finite = np.isfinite(fs.pooled_z[:, nov, si]).any(1) if N else np.zeros(0, bool)
    ok = (vf >= cfg.min_valid_frac) & finite
    reasons: list[str | None] = []
    for i in range(N):
        if ok[i]:
            reasons.append(None)
        elif vf[i] >= cfg.min_valid_frac:
            reasons.append("no finite feature")
        elif lowq[i] >= 0.5:
            reasons.append("low fit quality")
        else:
            reasons.append("tracking dropout")
    return ok, reasons


def analysis_blocks(eighths: dict, ok: np.ndarray) -> np.ndarray:
    """Block id per eighth (-1 = masked): runs of analysed eighths with consecutive grid indices. Novelty windows,
    boundary snapping and segments never cross a block edge."""
    idx = np.asarray(eighths["index"], int)
    out = np.full(len(idx), -1, int)
    k, prev = -1, False
    for i in range(len(idx)):
        if not ok[i]:
            prev = False
            continue
        if not prev or idx[i] != idx[i - 1] + 1:
            k += 1
        out[i] = k
        prev = True
    return out


def block_spans(blocks: np.ndarray) -> list[tuple[int, int]]:
    """[(first eighth, last eighth + 1)] per block, in order."""
    blocks = np.asarray(blocks, int)
    if not len(blocks) or blocks.max() < 0:
        return []
    out = []
    for k in range(int(blocks.max()) + 1):
        ii = np.flatnonzero(blocks == k)
        out.append((int(ii[0]), int(ii[-1]) + 1))
    return out


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[(a, b)) runs of True."""
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            out.append((i, j + 1))
            i = j + 1
        else:
            i += 1
    return out


# ----------------------------------------------------------------------------------------------- novelty

def novelty_matrix(fs: FeatureSet, cfg: ProposalConfig, ok: np.ndarray | None = None
                   ) -> tuple[np.ndarray, list[str], list[str]]:
    """(N, D) clipped robust-z matrix of the usable novelty features (rows of masked eighths NaN), their names, and
    the dropped ones. A feature is dropped only when it is NaN on more than 1 - min_feature_coverage of the analysed
    eighths (e.g. cb_* without counterbalance.json); remaining NaN cells are skipped by the NaN-aware similarity."""
    si = STATS.index(cfg.stat)
    N = len(fs.pooled_z)
    ok = np.ones(N, bool) if ok is None else np.asarray(ok, bool)
    n_ok = int(ok.sum())
    use, dropped = [], []
    for f in FEATURES:
        if not f.novelty:
            continue
        col = fs.pooled_z[ok, INDEX[f.name], si]
        cov = float(np.isfinite(col).mean()) if n_ok else 0.0
        (use if n_ok and cov >= cfg.min_feature_coverage else dropped).append(f.name)
    Z = np.stack([fs.pooled_z[:, INDEX[n], si] for n in use], 1).astype(float) if use else np.zeros((N, 0))
    Z[~ok] = np.nan
    return np.clip(Z, -cfg.z_clip, cfg.z_clip), use, dropped


def feature_groups(names: list[str]) -> dict[str, list[int]]:
    """Column indices (into the novelty matrix) per group: couple relation, lead, follow, counterbalance."""
    out: dict[str, list[int]] = {"couple": [], "lead": [], "follow": [], "counterbalance": []}
    for i, n in enumerate(names):
        f = FEATURES[INDEX[n]]
        g = "counterbalance" if f.group == "counterbalance" else (f.role or "couple")
        if g in out:
            out[g].append(i)
    return out


def similarity(Z: np.ndarray, bandwidth: float = 1.0) -> np.ndarray:
    """RBF similarity of the rows of Z on the RMS difference over the features finite in BOTH rows (NaN-aware);
    NaN where two rows share no finite feature. Identical to the plain RBF when Z has no NaN."""
    N = len(Z)
    if Z.shape[1] == 0:
        return np.ones((N, N))
    M = np.isfinite(Z)
    Mf = M.astype(float)
    Z0 = np.where(M, Z, 0.0)
    Q = Z0 ** 2
    cnt = Mf @ Mf.T
    d2 = np.maximum(Q @ Mf.T + Mf @ Q.T - 2.0 * (Z0 @ Z0.T), 0.0) / np.maximum(cnt, 1.0)
    S = np.exp(-d2 / (2.0 * bandwidth ** 2))
    S[cnt == 0] = np.nan
    return S


def foote_novelty(S: np.ndarray, w: int, blocks: np.ndarray | None = None) -> np.ndarray:
    """novelty[b] for the boundary before eighth b (b = 0..N; 0 and N are the edges, NaN): Gaussian-tapered mean
    within-block similarity (diagonal excluded) minus mean cross-block similarity over w eighths each side. With
    `blocks` (analysis_blocks) the windows stay inside the boundary's block; boundaries at a block edge or next to a
    masked eighth are NaN. NaN similarities carry no weight."""
    N = len(S)
    blocks = np.zeros(N, int) if blocks is None else np.asarray(blocks, int)
    out = np.full(N + 1, np.nan)
    sd = max(w / 2.0, 1.0)
    for b in range(1, N):
        k = blocks[b]
        if k < 0 or blocks[b - 1] != k:
            continue
        L = np.arange(max(0, b - w), b)
        L = L[blocks[L] == k]
        R = np.arange(b, min(N, b + w))
        R = R[blocks[R] == k]
        gL = np.exp(-0.5 * ((b - 0.5 - L) / sd) ** 2)
        gR = np.exp(-0.5 * ((R - (b - 0.5)) / sd) ** 2)
        within_n = within_d = 0.0
        for idx, g in ((L, gL), (R, gR)):
            if len(idx) < 2:
                continue
            W = np.outer(g, g)
            np.fill_diagonal(W, 0.0)
            Sb = S[np.ix_(idx, idx)]
            m = np.isfinite(Sb)
            within_n += float((W * np.where(m, Sb, 0.0)).sum())
            within_d += float((W * m).sum())
        within = within_n / within_d if within_d > 0 else 1.0
        Wc = np.outer(gL, gR)
        Sc = S[np.ix_(L, R)]
        m = np.isfinite(Sc)
        wsum = float((Wc * m).sum())
        if wsum <= 0:
            continue
        out[b] = within - float((Wc * np.where(m, Sc, 0.0)).sum()) / wsum
    return out


def kernel_half_width(n: int, cfg: ProposalConfig) -> int:
    if cfg.half_width:
        return int(cfg.half_width)
    return 8 if n >= 32 else max(2, n // 4)


def snap_targets(slots: np.ndarray, mode: str) -> np.ndarray:
    """Eighth positions b (0..N) that are allowed boundaries: the start of an eighth with the right slot."""
    slots = np.asarray(slots, int)
    if mode == "eighth":
        ok = np.ones(len(slots), bool)
    elif mode == "half":
        ok = (slots % 4) == 0
    elif mode == "measure":
        ok = slots == 0
    else:
        raise ValueError(f"snap {mode!r} not in half | measure | eighth")
    return np.flatnonzero(ok)


def pick_boundaries(strength: np.ndarray, slots: np.ndarray, cfg: ProposalConfig, blocks: np.ndarray | None = None,
                    rejected: list | None = None) -> list[dict]:
    """Peaks of strength (len N+1) >= threshold, snapped to the snap targets of their block, >= min_eighths apart
    and >= min_eighths from the block's edges (the analysed range's ends and gaps); strongest first wins. Returns
    [{eighth: b, strength, peak_eighth, strength_at_snap}] sorted by b. Peaks >= threshold that are not proposed are
    appended to `rejected` as {peak_eighth, strength, reason: 'edge' (too close to an end of the analysed range or
    to a gap) | 'no snap target' | 'merged' (a stronger peak snapped to the same eighth) | 'spacing' (a stronger
    boundary within min_eighths)}."""
    N = len(strength) - 1
    s = np.nan_to_num(np.asarray(strength, float), nan=-1.0)
    blocks = np.zeros(N, int) if blocks is None else np.asarray(blocks, int)
    spans = block_spans(blocks)
    targets = snap_targets(slots, cfg.snap)
    reach = {"eighth": 0, "half": 2, "measure": 4}[cfg.snap]       # half the snap unit, in eighths

    def rej(b: int, why: str) -> None:
        if rejected is not None:
            rejected.append({"peak_eighth": int(b), "strength": float(s[b]), "reason": why})

    peaks = [b for b in range(1, N) if s[b] >= cfg.threshold and s[b] >= s[b - 1] and s[b] >= s[b + 1]]
    cands: dict[int, dict] = {}
    for b in peaks:
        lo, hi = spans[blocks[b]]
        within = [int(x) for x in targets if abs(int(x) - b) <= reach]
        near = [x for x in within if lo + cfg.min_eighths <= x <= hi - cfg.min_eighths]
        if not near:
            rej(b, "edge" if within else "no snap target")
            continue
        tb = min(near, key=lambda x: (abs(x - b), -s[x]))      # nearest; a tie goes to the stronger side
        if tb in cands and s[b] <= cands[tb]["strength"]:
            rej(b, "merged")
            continue
        if tb in cands:
            rej(cands[tb]["peak_eighth"], "merged")
        cands[tb] = {"eighth": tb, "strength": float(s[b]), "peak_eighth": int(b), "strength_at_snap": float(s[tb])}
    chosen: list[dict] = []
    for c in sorted(cands.values(), key=lambda c: -c["strength"]):
        if all(abs(c["eighth"] - o["eighth"]) >= cfg.min_eighths for o in chosen):
            chosen.append(c)
        else:
            rej(c["peak_eighth"], "spacing")
    return sorted(chosen, key=lambda c: c["eighth"])


# ----------------------------------------------------------------------------------------------- segments

def graph_energy_bands(graph_path: Path | None = None) -> list[dict]:
    from .graph import ENERGY_BANDS, GRAPH_PATH
    p = Path(graph_path) if graph_path else GRAPH_PATH
    if p.exists():
        bands = json.loads(p.read_text(encoding="utf-8")).get("energy_bands")
        if bands:
            return [{"band": int(b["band"]), "lo": float(b["lo"]), "hi": float(b["hi"]), "label": b["label"]}
                    for b in bands]
    return [{"band": b, "lo": lo, "hi": hi, "label": lab} for b, lo, hi, lab in ENERGY_BANDS]


def band_for(energy: float, bands: list[dict]) -> dict:
    for b in bands:
        if energy < b["hi"]:
            return b
    return bands[-1]


def energy_abs(e: float, rng: tuple) -> float:
    lo, hi = math.log10(rng[0]), math.log10(rng[1])
    return float(np.clip((math.log10(max(e, 1e-6)) - lo) / (hi - lo), 0.0, 1.0))


def _nanmean0(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, float)
    if not len(a):
        return np.full(a.shape[1:], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, 0)


def _top_changes(fs: FeatureSet, Z: np.ndarray, names: list[str], b: int, w: int, k: int,
                 lo: int = 0, hi: int | None = None) -> list[dict]:
    """The k features whose mean z changes most across boundary b (windows of w eighths, inside [lo, hi))."""
    hi = len(Z) if hi is None else hi
    a0, b1 = max(lo, b - w), min(hi, b + w)
    if b - a0 <= 0 or b1 - b <= 0:
        return []
    dz = _nanmean0(Z[b:b1]) - _nanmean0(Z[a0:b])                # after - before
    si = STATS.index("mean")
    out = []
    for j in np.argsort(-np.abs(np.nan_to_num(dz, nan=0.0)))[:k]:
        if not np.isfinite(dz[j]):
            continue
        n = names[j]
        raw = fs.pooled[:, INDEX[n], si].astype(float)
        out.append({"feature": n, "unit": FEATURES[INDEX[n]].unit, "dz": round(float(dz[j]), 3),
                    "before": round(float(_nanmean0(raw[a0:b])), 4),
                    "after": round(float(_nanmean0(raw[b:b1])), 4)})
    return out


def _rms(a: np.ndarray, b: np.ndarray) -> float:
    d = (np.asarray(a, float) - np.asarray(b, float)) ** 2
    m = np.isfinite(d)
    return float(np.sqrt(d[m].mean())) if m.any() else math.inf


def _clusters(V: np.ndarray, repeat_rms: float) -> tuple[list[int], list[int | None]]:
    """Leader clustering (in time order) of segment mean z-vectors + repeat_of (nearest earlier within repeat_rms).
    Distances are RMS over the features finite in both vectors."""
    cent: list[np.ndarray] = []
    members: list[list[int]] = []
    lab, rep = [], []
    for i, v in enumerate(V):
        if V.shape[1] == 0:
            lab.append(0)
            rep.append(None)
            continue
        d_prev = [_rms(v, V[j]) for j in range(i)]
        j = int(np.argmin(d_prev)) if d_prev else -1
        rep.append(j if j >= 0 and d_prev[j] < repeat_rms else None)
        dc = [_rms(v, c) for c in cent]
        k = int(np.argmin(dc)) if dc else -1
        if k >= 0 and dc[k] < repeat_rms:
            members[k].append(i)
            cent[k] = _nanmean0(V[members[k]])
            lab.append(k)
        else:
            cent.append(v.copy())
            members.append([i])
            lab.append(len(cent) - 1)
    return lab, rep


def propose(fs: FeatureSet, cfg: ProposalConfig | None = None, bands: list[dict] | None = None) -> dict:
    """The proposals document (without source/evaluation/stale) for a FeatureSet."""
    from .labels import new_id, now_iso
    cfg = cfg or ProposalConfig()
    bands = bands or graph_energy_bands()
    e = fs.eighths
    N = len(e["t0"])
    warn = list(fs.meta.get("warnings", []))
    if N < 2:
        raise ValueError(f"only {N} pooled eighth(s): nothing to segment")
    vf = _valid_frac(fs)
    ok, reasons = eighth_mask(fs, cfg)
    blocks = analysis_blocks(e, ok)
    spans = block_spans(blocks)
    n_ok = int(ok.sum())
    Z, names, dropped = novelty_matrix(fs, cfg, ok)
    w = kernel_half_width(n_ok, cfg)
    curves = {"all": np.clip(foote_novelty(similarity(Z, cfg.bandwidth), w, blocks), 0.0, 1.0)}
    for g, cols in feature_groups(names).items():
        if len(cols) >= cfg.min_group_features:
            curves[g] = np.clip(foote_novelty(similarity(Z[:, cols], cfg.bandwidth), w, blocks), 0.0, 1.0)
    stack = np.stack(list(curves.values()))
    fin = np.isfinite(stack)
    strength = np.where(fin.any(0), np.max(np.where(fin, stack, -1.0), 0), np.nan)
    gnames = list(curves)
    rejected: list[dict] = []
    picked = pick_boundaries(strength, e["slot"], cfg, blocks, rejected)
    finite_s = strength[np.isfinite(strength)]
    noise_floor = float(np.median(finite_s)) if len(finite_s) else None
    t_first, t_last = float(fs.t[0]), float(fs.t[-1])
    dt = float(np.median(np.diff(fs.t))) if len(fs.t) > 1 else 0.0
    cov_end = t_last + dt

    def btime(b: int) -> float:
        return float(e["t0"][b]) if b < N else float(e["t1"][N - 1])

    def by_group(b: int) -> dict:
        return {g: (None if not np.isfinite(c[b]) else round(float(c[b]), 4)) for g, c in curves.items()}

    # masked eighths and gaps between blocks (tracking dropouts, low fit quality, missing grid eighths)
    masked = [{"t0": round(float(e["t0"][a]), 4), "t1": round(float(e["t1"][b - 1]), 4), "eighths": [int(a), int(b)],
               "reason": " + ".join(sorted({reasons[i] for i in range(a, b)})),
               "valid_frac": [round(float(x), 3) for x in vf[a:b]]} for a, b in _runs(~ok)]
    gap_info: dict[int, dict] = {}
    for (_, a1), (b0, _) in zip(spans, spans[1:]):
        why = sorted({reasons[i] for i in range(a1, b0)})
        if int(e["index"][b0]) - int(e["index"][a1 - 1]) - 1 > b0 - a1:
            why.append("missing eighths (frame gap or < min_eighth_coverage)")
        gap_info[b0] = {"gap": [round(float(e["t1"][a1 - 1]), 4), round(float(e["t0"][b0]), 4)],
                        "reason": " + ".join(why)}
    if masked:
        n_m = sum(m["eighths"][1] - m["eighths"][0] for m in masked)
        warn.append(f"{n_m} of {N} eighth(s) masked (< {cfg.min_valid_frac:.0%} valid frames: "
                    + "; ".join(f"{m['t0']:.2f}-{m['t1']:.2f} s {m['reason']}" for m in masked[:4])
                    + ("; ..." if len(masked) > 4 else "") + "): no proposals there, segments split (kind 'gap')")
    elif gap_info:
        warn.append(f"{len(gap_info)} gap(s) in the analysed eighths: segments split there (kind 'gap')")
    non_cb = [n for n in dropped if not n.startswith("cb_")]
    if non_cb:
        warn.append(f"{len(non_cb)} motion feature(s) dropped (NaN on > {1 - cfg.min_feature_coverage:.0%} of the "
                    f"analysed eighths): {', '.join(non_cb[:8])}" + (" ..." if len(non_cb) > 8 else ""))
    if not spans:
        warn.append("every eighth is masked: no proposals")

    by_b = {b["eighth"]: b for b in picked}
    boundaries = []
    for b in sorted(set(by_b) | set(gap_info)):
        base = {"t": round(btime(b), 4), "eighth": int(b), "grid_index": int(e["index"][b]),
                "slot": int(e["slot"][b])}
        info = by_b.get(b)
        if info is None:
            boundaries.append({**base, "kind": "gap", "strength": None, "weak": None, "margin": None, "group": None,
                               "strength_by_group": {}, "strength_at_snap": None, "peak_t": None, "top_changes": [],
                               **gap_info[b]})
            continue
        lo, hi = spans[int(blocks[b])]
        sg = by_group(info["peak_eighth"])
        st = info["strength"]
        boundaries.append({
            **base, "kind": "novelty", "strength": round(st, 4), "weak": bool(st < cfg.strong),
            "margin": None if noise_floor is None else round(st - noise_floor, 4),
            "group": max((g for g in gnames if sg.get(g) is not None), key=lambda g: sg[g]),
            "strength_by_group": sg, "strength_at_snap": round(info["strength_at_snap"], 4),
            "peak_t": round(btime(info["peak_eighth"]), 4),
            "top_changes": _top_changes(fs, Z, names, b, w, cfg.top_changes, lo, hi)})
    ke = fs.pooled[:, INDEX["couple_ke"], STATS.index("mean")].astype(float)
    ke_all = ke[np.isfinite(ke) & ok]
    cover = e["coverage"].astype(float)
    segs, V = [], []
    for k, (lo, hi) in enumerate(spans):
        edges = [lo] + [b for b in sorted(by_b) if lo < b < hi] + [hi]
        for a, b in zip(edges[:-1], edges[1:]):
            sl = slice(a, b)
            wts = cover[sl]
            kk = ke[sl]
            okk = np.isfinite(kk)
            e_mean = float(np.average(kk[okk], weights=wts[okk])) if okk.any() else float("nan")
            rank = float((ke_all <= e_mean).mean()) if len(ke_all) and np.isfinite(e_mean) else 0.5
            eabs = energy_abs(e_mean, cfg.energy_abs_range) if np.isfinite(e_mean) else 0.0
            energy = rank if cfg.energy == "rank" else eabs
            band = band_for(energy, bands)
            V.append(_nanmean0(Z[sl]) if Z.shape[1] else np.zeros(0))
            summary = {}
            for n in SUMMARY:
                col = fs.pooled[sl, INDEX[n], STATS.index("min" if n.endswith("com_height") else "mean")].astype(float)
                m = np.isfinite(col)
                summary[n] = round(float(np.average(col[m], weights=wts[m])), 4) if m.any() else None
            # the edge segments span the whole analysed range (frames outside the pooled eighths belong to them)
            t0 = t_first if a == 0 else float(e["t0"][a])
            t1 = cov_end if b == N else float(e["t1"][b - 1])
            sb = by_b.get(a, {}).get("strength")
            se = by_b.get(b, {}).get("strength")
            conf = [x for x in (sb, se) if x is not None]
            segs.append({
                "id": new_id("prop"), "t0": round(t0, 4) + 0.0, "t1": round(t1, 4) + 0.0, "move": None, "node": None,
                "tier": "couple", "provenance": "auto",
                "confidence": round(float(np.mean(conf)), 4) if conf else None,
                "boundary_strength": None if sb is None else round(sb, 4),
                "end_strength": None if se is None else round(se, 4),
                "start": "novelty" if a in by_b else ("gap" if k > 0 else "edge"),
                "end": "novelty" if b in by_b else ("gap" if k < len(spans) - 1 else "edge"),
                "eighths": [int(a), int(b)], "n_eighths": int(b - a),
                "partial": bool(t0 > float(e["t0"][a]) + 1e-6 or t1 < float(e["t1"][b - 1]) - 1e-6),
                "valid_frac": round(float(np.mean(vf[sl])), 3),
                "energy": round(energy, 4), "energy_rank": round(rank, 4), "energy_abs": round(eabs, 4),
                "couple_ke_j_per_kg": None if not np.isfinite(e_mean) else round(e_mean, 4),
                "energy_band": int(band["band"]), "energy_label": band["label"],
                "summary": summary, "created": now_iso()})
    V = np.stack(V) if V and len(V[0]) else np.zeros((len(segs), 0))
    lab, rep = _clusters(V, cfg.repeat_rms)
    for s, c, r in zip(segs, lab, rep):
        s["cluster"] = int(c)
        s["repeat_of"] = segs[r]["id"] if r is not None else None
    weak = [b for b in boundaries if b["kind"] == "novelty" and b["weak"]]
    if weak:
        warn.append(f"{len(weak)} weak boundar{'y' if len(weak) == 1 else 'ies'} (strength < {cfg.strong:g}, the range "
                    "~1 cm of slow fit wobble reaches on the homogeneous synthetic): low confidence")
    return {
        "version": VERSION, "kind": "proposals", "take": fs.take, "time_base": "reference", "provenance": "auto",
        "note": NOTE,
        "method": {"name": "foote-novelty", **asdict(cfg), "half_width_used": w, "n_features": len(names),
                   "features": names, "features_dropped": dropped,
                   "description": "robust-z per-eighth pooled feature means (eighths with < min_valid_frac valid "
                                  "frames masked) -> NaN-aware RBF self-similarity -> Gaussian checkerboard novelty "
                                  "inside each block of analysed eighths (all features and per group: couple, lead, "
                                  "follow, counterbalance; strength = the maximum) -> peaks >= threshold snapped to "
                                  "type-1 eighths (half-measures); see proposals.py"},
        "coverage": [round(t_first, 4), round(cov_end, 4)],
        "energy_bands": bands,
        "energy_basis": ("rank within this take (F5): the band says where the segment's couple kinetic energy ranks "
                         "among this take's eighths, not an absolute level" if cfg.energy == "rank" else
                         f"absolute: couple kinetic energy log-mapped over {list(cfg.energy_abs_range)} J/kg "
                         "(provisional)"),
        "analysis": {"eighths": N, "analysed_eighths": n_ok,
                     "blocks": [[round(float(e["t0"][a]), 4), round(float(e["t1"][b - 1]), 4)] for a, b in spans],
                     "masked": masked},
        "novelty": {"t": [round(btime(b), 4) for b in range(N + 1)],
                    "strength": [None if not np.isfinite(x) else round(float(x), 4) for x in strength],
                    "groups": {g: [None if not np.isfinite(x) else round(float(x), 4) for x in c]
                               for g, c in curves.items()},
                    "group_features": {g: [names[i] for i in cols] for g, cols in feature_groups(names).items()
                                       if len(cols) >= cfg.min_group_features},
                    "half_width_eighths": w,
                    "noise_floor": None if noise_floor is None else round(noise_floor, 4),
                    "noise_floor_note": "median boundary strength over the analysed eighths; boundary margin = "
                                        "strength - noise_floor"},
        "boundaries": boundaries, "segments": segs,
        "rejected_peaks": [{"t": round(btime(r["peak_eighth"]), 4), "eighth": r["peak_eighth"],
                            "strength": round(r["strength"], 4), "reason": r["reason"]}
                           for r in sorted(rejected, key=lambda r: r["peak_eighth"])],
        "clusters": {"method": f"leader clustering of segment mean z-vectors, RMS < {cfg.repeat_rms}",
                     "n": int(max(lab) + 1) if lab else 0},
        "warnings": warn,
    }


# ----------------------------------------------------------------------------------------------- truth

def truth_transitions(work: Path) -> list[dict] | None:
    """Move transitions of a synthetic take from truth.json: [{t, lo, hi}] (a proposal anywhere in [lo, hi] +-
    tolerance hits it); [] for a homogeneous synthetic take; None without truth."""
    p = Path(work) / "truth.json"
    if not p.exists():
        return None
    tr = json.loads(p.read_text())
    out = []
    for c in tr.get("counterbalance") or []:
        a0 = (c.get("setup") or [c["t0"]])[0]
        a1 = c.get("pivot_t0", c["t0"])
        b0 = c.get("pivot_t1", c["t1"])
        b1 = (c.get("release") or [None, c["t1"]])[1]
        out.append({"t": float(c["t0"]), "lo": float(min(a0, a1, c["t0"])), "hi": float(max(a0, a1, c["t0"])),
                    "what": "ordinary -> counterbalance"})
        out.append({"t": float(c["t1"]), "lo": float(min(b0, b1, c["t1"])), "hi": float(max(b0, b1, c["t1"])),
                    "what": "counterbalance -> ordinary"})
    return out


def evaluate(doc: dict, truth: list[dict], tol_s: float) -> dict:
    """Hits / misses / false positives of the proposed boundaries against truth transitions (one-to-one)."""
    props = [b["t"] for b in doc["boundaries"] if b["kind"] == "novelty"]
    used, hits = set(), []
    for tr in truth:
        best, bd = None, None
        for i, t in enumerate(props):
            if i in used:
                continue
            d = 0.0 if tr["lo"] <= t <= tr["hi"] else min(abs(t - tr["lo"]), abs(t - tr["hi"]))
            if d <= tol_s and (bd is None or d < bd):
                best, bd = i, d
        if best is not None:
            used.add(best)
            hits.append({"truth_t": tr["t"], "proposal_t": props[best], "distance_s": round(bd, 4),
                         "what": tr.get("what")})
    fp = [t for i, t in enumerate(props) if i not in used]
    n_t, n_p = len(truth), len(props)
    return {"tolerance_s": round(tol_s, 4), "truth": truth, "hits": hits,
            "misses": [tr for tr in truth if tr["t"] not in {h["truth_t"] for h in hits}],
            "false_positives": fp, "precision": (len(hits) / n_p) if n_p else (1.0 if not n_t else 0.0),
            "recall": (len(hits) / n_t) if n_t else 1.0}


# ----------------------------------------------------------------------------------------------- I/O + CLI

def proposals_path(take: str) -> Path:
    return moves_dir(take) / "proposals.json"


def grid_coverage(work: Path, coverage: list[float]) -> tuple[list[float] | None, list[list[float]]]:
    """(the take's beat-grid span, the parts of it outside `coverage` = no fits there, no motion proposals)."""
    from .features import load_grid
    g = load_grid(Path(work))
    if g is None or not len(g[0]):
        return None, []
    g0, g1 = float(g[0][0]), float(g[0][-1])
    out = []
    if coverage[0] > g0 + 1e-6:
        out.append([round(g0, 4), round(min(coverage[0], g1), 4)])
    if coverage[1] < g1 - 1e-6:
        out.append([round(max(coverage[1], g0), 4), round(g1, 4)])
    return [round(g0, 4), round(g1, 4)], [u for u in out if u[1] > u[0]]


def run(take: str, cfg: ProposalConfig | None = None, features_npz: Path | None = None,
        out: Path | None = None, work: Path | None = None) -> tuple[dict, Path]:
    from . import features as F
    from .graph import GRAPH_PATH
    cfg = cfg or ProposalConfig()
    work = Path(work) if work else work_dir(take)
    npz_p = Path(features_npz) if features_npz else features_paths(take)[0]
    fs = F.load(npz_p)
    doc = propose(fs, cfg)
    doc["source"] = {"features": F._rel(npz_p), "features_sha256_16": F._sha16(npz_p),
                     "graph": F._rel(GRAPH_PATH) if GRAPH_PATH.exists() else "graph.ENERGY_BANDS (no graph file)"}
    stale = F.staleness(take, work, npz_p)
    doc["stale"] = stale
    for m in stale:
        doc["warnings"].insert(0, f"STALE: {m} - re-run `python -m dancecap.moves features {take}` and proposals")
    span, unc = grid_coverage(work, doc["coverage"])
    doc["grid_span"], doc["uncovered"] = span, unc
    if span and unc:
        u = sum(b - a for a, b in unc)
        doc["warnings"].append(f"proposals cover {doc['coverage'][1] - doc['coverage'][0]:.1f} s of the "
                               f"{span[1] - span[0]:.1f} s beat grid: {u:.1f} s have no SMPL-X fits and no motion "
                               "proposals (uncovered)")
    truth = truth_transitions(work)
    doc["evaluation"] = None
    if truth is not None:
        tol = 4 * float(np.median(np.diff(fs.grid_t)))   # half a measure
        doc["evaluation"] = evaluate(doc, truth, tol)
    out = Path(out) if out else proposals_path(take)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    return doc, out


def validate_proposals(doc: dict) -> list[str]:
    errs = []
    if doc.get("version") != VERSION:
        errs.append(f"version {doc.get('version')!r} != {VERSION}")
    if doc.get("time_base") != "reference":
        errs.append("time_base must be 'reference'")
    segs = sorted(doc.get("segments", []), key=lambda s: s["t0"])
    for s in segs:
        if s.get("move") is not None:
            errs.append(f"segment {s['id']}: proposals never name a move (move must be null)")
        if s.get("provenance") != "auto":
            errs.append(f"segment {s['id']}: provenance must be 'auto'")
        if not s["t0"] < s["t1"]:
            errs.append(f"segment {s['id']}: t0 >= t1")
    for a, b in zip(segs, segs[1:]):
        if b["t0"] < a["t1"] - 1e-6:
            errs.append(f"segments {a['id']} and {b['id']} overlap")
    for m in (doc.get("analysis") or {}).get("masked", []):
        for s in segs:
            if s["t0"] < m["t1"] - 1e-6 and s["t1"] > m["t0"] + 1e-6:
                errs.append(f"segment {s['id']} covers masked eighths [{m['t0']}, {m['t1']}]")
    return errs


def cmd_proposals(args: argparse.Namespace) -> int:
    from . import features as F
    npz_p = features_paths(args.take)[0]
    if not npz_p.exists() or args.recompute:
        print(f"[moves] computing features first ({'--recompute' if args.recompute else 'no features.npz'})")
        F.run(args.take, None, F.FeatureConfig(com=args.com))
    cfg = ProposalConfig(half_width=args.half_width, threshold=args.threshold, min_eighths=args.min_eighths,
                         snap=args.snap, energy=args.energy, strong=args.strong)
    doc, out = run(args.take, cfg)
    errs = validate_proposals(doc)
    for e in errs:
        print("[moves] proposals INVALID", e)
    nb = sum(b["kind"] == "novelty" for b in doc["boundaries"])
    print(f"[moves] wrote {out}: {nb} proposed boundar{'y' if nb == 1 else 'ies'}, {len(doc['segments'])} "
          f"segment(s) over [{doc['coverage'][0]:.2f}, {doc['coverage'][1]:.2f}] s (w = "
          f"{doc['novelty']['half_width_eighths']} eighths, threshold {cfg.threshold}, noise floor "
          f"{doc['novelty']['noise_floor']}) - proposals, not classifications")
    an = doc["analysis"]
    if an["masked"]:
        print(f"[moves]   {an['eighths'] - an['analysed_eighths']} of {an['eighths']} eighths masked: "
              + "; ".join(f"{m['t0']:.2f}-{m['t1']:.2f} s {m['reason']}" for m in an["masked"]))
    for b in doc["boundaries"]:
        if b["kind"] == "gap":
            print(f"[moves]   boundary {b['t']:.2f} s (gap {b['gap'][0]:.2f}-{b['gap'][1]:.2f} s: {b['reason']})")
            continue
        ch = ", ".join(f"{c['feature']} {c['dz']:+.1f}z" for c in b["top_changes"][:3])
        print(f"[moves]   boundary {b['t']:.2f} s (peak {b['peak_t']:.2f} s, strength {b['strength']}"
              f"{' WEAK' if b['weak'] else ''}, margin {b['margin']}, strongest in {b.get('group')}): {ch}")
    for r in doc["rejected_peaks"]:
        print(f"[moves]   not proposed: peak {r['t']:.2f} s strength {r['strength']} ({r['reason']})")
    for s in doc["segments"]:
        print(f"[moves]   [{s['t0']:.2f}, {s['t1']:.2f}) {s['n_eighths']} eighths  energy band {s['energy_band']} "
              f"{s['energy_label']} (rank {s['energy_rank']:.2f} within the take, {s['couple_ke_j_per_kg']} J/kg)  "
              f"cluster {s['cluster']}" + (f"  repeat of {s['repeat_of']}" if s["repeat_of"] else ""))
    for m in doc.get("stale") or []:
        print(f"[moves] STALE {m}")
    ev = doc.get("evaluation")
    if ev is not None:
        print(f"[moves] truth check: {len(ev['hits'])}/{len(ev['truth'])} transitions found, "
              f"{len(ev['false_positives'])} false positive(s) (tolerance {ev['tolerance_s']:.2f} s)")
    if not args.no_review:
        from .features_review import draw_features
        print(f"[moves] review image {draw_features(args.take)}")
    return 1 if errs else 0


def register(sub) -> None:
    d = ProposalConfig()
    p = sub.add_parser("proposals", help="unlabelled candidate move boundaries (novelty on the features) -> "
                                         "work/<take>/moves/proposals.json - proposals, not classifications")
    p.add_argument("take")
    p.add_argument("--threshold", type=float, default=d.threshold)
    p.add_argument("--strong", type=float, default=d.strong, help="boundaries below this are flagged weak")
    p.add_argument("--half-width", type=int, default=None, help="novelty kernel half width in eighths")
    p.add_argument("--min-eighths", type=int, default=d.min_eighths)
    p.add_argument("--snap", choices=("half", "measure", "eighth"), default=d.snap)
    p.add_argument("--energy", choices=("rank", "abs"), default=d.energy)
    p.add_argument("--recompute", action="store_true", help="recompute features.npz first")
    p.add_argument("--com", choices=("physics", "joints"), default="physics", help="for a (re)computation")
    p.add_argument("--no-review", action="store_true")
    p.set_defaults(func=cmd_proposals)
