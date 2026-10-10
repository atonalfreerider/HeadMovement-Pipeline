"""Joint clustering of the speaker embeddings of ALL phones on the master clock into global voices (small numpy / sklearn work, local).

Why joint: the seven phones hear the same room, so one voice is on every phone at the same master time, only through a different
microphone, distance and codec. Clustering each phone alone and matching clusters afterwards would have to guess the matching;
clustering the pooled windows directly would split by phone (the channel is a stronger signal than the speaker for short windows).
Three measures keep the channel out:

  1. per-phone mean normalisation: the phone's mean embedding (over its speech windows) is subtracted and the result L2-normalised,
     the standard cepstral-mean-style fix for a constant channel;
  2. "moments": the windows of all phones whose centres fall in the same 0.75 s master-clock bin are fused into ONE embedding (a
     level-weighted mean of the normalised ones). A moment is the same utterance heard by up to seven microphones, so the channel
     errors of the phones average out and a window that only one phone heard does not become its own cluster;
  3. validation by something the embeddings never saw: the median F0 of every cluster (a woman and a man speak around 200 and 120
     Hz) and the loudest phone per cluster must be consistent (reported, see `describe`).

Then: average-linkage agglomerative clustering with the distance threshold chosen by silhouette among the thresholds that leave the
two biggest clusters most of the speech; small clusters become "other" (students, ambient); Viterbi smoothing over time with a switch
penalty; turns with a per-turn confidence from the margin between the best and the second-best centroid.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

BIN_S = 0.75


# ------------------------------------------------------------------------------------------------ table of windows


@dataclass
class Table:
    cam: np.ndarray                      # (N,) '<id>' strings
    t0: np.ndarray                       # source seconds
    t1: np.ndarray
    m0: np.ndarray                       # master seconds
    m1: np.ndarray
    f0: np.ndarray
    f0v: np.ndarray
    rms: np.ndarray
    vad: np.ndarray
    emb: np.ndarray                      # (N, D) raw concatenated embedding
    models: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.cam)

    @property
    def mc(self) -> np.ndarray:
        return 0.5 * (self.m0 + self.m1)


def l2n(x: np.ndarray, axis: int = -1, eps: float = 1e-9) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + eps)


def load_table(raw_dir: Path, clocks: dict, models: tuple[str, ...] = ("ecapa", "titanet"), cams: list[str] | None = None) -> Table:
    """Pool raw/<cam>/embed.npz of every phone that has a clock. Each model's embedding is L2-normalised per phone mean-subtracted
    FIRST (so a phone's channel offset is removed per model), then the models are concatenated with equal weight."""
    parts = []
    used: list[str] = []
    for cam_dir in sorted(Path(raw_dir).glob("[0-9][0-9]")):
        cam = cam_dir.name
        if cams and cam not in cams:
            continue
        clk = clocks.get(cam)
        f = cam_dir / "embed.npz"
        if clk is None or not f.exists():
            continue
        z = np.load(f)
        if len(z["t0"]) == 0:
            continue
        have = [m for m in models if m in z.files]
        if not have:
            continue
        blocks = []
        for m in have:
            e = l2n(z[m].astype(np.float64))
            e = l2n(e - e.mean(axis=0, keepdims=True))
            blocks.append(e)
        used = have if not used else [m for m in used if m in have]
        parts.append((cam, clk, z, np.concatenate(blocks, axis=1) / math.sqrt(len(blocks)), have))
    if not parts:
        raise RuntimeError("no embedding files with a clock found")
    dims = {p[3].shape[1] // len(p[4]) for p in parts}
    n_models = {len(p[4]) for p in parts}
    if len(n_models) > 1:      # some phone lacks a model: use only the common ones
        common = [m for m in models if all(m in p[4] for p in parts)]
        return load_table(raw_dir, clocks, tuple(common), cams)
    rows = {k: [] for k in ("cam", "t0", "t1", "m0", "m1", "f0", "f0v", "rms", "vad", "emb")}
    for cam, clk, z, emb, have in parts:
        rows["cam"] += [cam] * len(z["t0"])
        rows["t0"].append(z["t0"]); rows["t1"].append(z["t1"])
        rows["m0"].append(clk.slope * z["t0"] + clk.offset); rows["m1"].append(clk.slope * z["t1"] + clk.offset)
        rows["f0"].append(z["f0_med"]); rows["f0v"].append(z["f0_voiced"]); rows["rms"].append(z["rms_db"]); rows["vad"].append(z["vadfrac"])
        rows["emb"].append(emb)
    return Table(np.array(rows["cam"]), *(np.concatenate(rows[k]).astype(np.float64) for k in ("t0", "t1", "m0", "m1", "f0", "f0v", "rms", "vad")),
                 np.concatenate(rows["emb"]), list(parts[0][4]))


# ------------------------------------------------------------------------------------------------ moments


@dataclass
class Moments:
    bin: np.ndarray                      # (M,) bin index
    t0: np.ndarray                       # master seconds
    t1: np.ndarray
    emb: np.ndarray                      # (M, D) fused, L2-normalised
    members: list[list[int]]             # table row indices
    cams: list[list[str]]
    f0: np.ndarray                       # median F0 over the voiced member windows (0 if none)
    loud_cam: list[str]                  # the member phone with the highest level
    loud_db: np.ndarray
    spread: np.ndarray                   # mean cosine distance of the members to the fused embedding (disagreement between phones)


def build_moments(t: Table, bin_s: float = BIN_S, min_vad: float = 0.0) -> Moments:
    keys = np.floor(t.mc / bin_s).astype(int)
    order = np.argsort(keys, kind="stable")
    out_bin, out_t0, out_t1, embs, members, cams, f0s, loud, loud_db, spread = [], [], [], [], [], [], [], [], [], []
    start = 0
    n = len(order)
    while start < n:
        end = start
        while end < n and keys[order[end]] == keys[order[start]]:
            end += 1
        idx = order[start:end]
        idx = idx[t.vad[idx] >= min_vad]
        if len(idx):
            amp = np.power(10.0, t.rms[idx] / 20.0)
            w = np.minimum(amp, 3.0 * np.median(amp)) + 1e-9
            e = (t.emb[idx] * w[:, None]).sum(axis=0) / w.sum()
            e = l2n(e)
            voiced = idx[t.f0[idx] > 0]
            out_bin.append(int(keys[idx[0]]))
            out_t0.append(float(keys[idx[0]] * bin_s)); out_t1.append(float((keys[idx[0]] + 1) * bin_s))
            embs.append(e)
            members.append([int(i) for i in idx])
            cams.append([str(c) for c in t.cam[idx]])
            f0s.append(float(np.median(t.f0[voiced])) if len(voiced) else 0.0)
            k = int(idx[np.argmax(t.rms[idx])])
            loud.append(str(t.cam[k])); loud_db.append(float(t.rms[k]))
            spread.append(float(np.mean(1.0 - l2n(t.emb[idx]) @ e)))
        start = end
    return Moments(np.array(out_bin), np.array(out_t0), np.array(out_t1), np.array(embs), members, cams, np.array(f0s), loud,
                   np.array(loud_db), np.array(spread))


# ------------------------------------------------------------------------------------------------ clustering


def cosine_distance_matrix(e: np.ndarray) -> np.ndarray:
    s = np.clip(e @ e.T, -1, 1)
    d = 1.0 - s
    np.fill_diagonal(d, 0.0)
    return d


def agglomerate(e: np.ndarray, threshold: float) -> np.ndarray:
    from sklearn.cluster import AgglomerativeClustering
    if len(e) < 2:
        return np.zeros(len(e), int)
    d = cosine_distance_matrix(e)
    return AgglomerativeClustering(n_clusters=None, metric="precomputed", linkage="average", distance_threshold=threshold).fit_predict(d)


def pick_threshold(e: np.ndarray, grid: np.ndarray | None = None, min_top2: float = 0.6) -> dict:
    """Try thresholds; keep those where the two biggest clusters hold at least `min_top2` of the moments and there are 2-12 clusters;
    among them take the best silhouette. Returns {threshold, labels, table: [{threshold, k, top2, silhouette}]}."""
    from sklearn.metrics import silhouette_score
    grid = np.linspace(0.25, 1.0, 31) if grid is None else grid
    d = cosine_distance_matrix(e)
    rows, best = [], None
    for thr in grid:
        lab = agglomerate(e, float(thr))
        k = len(set(lab.tolist()))
        counts = np.bincount(lab)
        top2 = float(np.sort(counts)[::-1][:2].sum() / len(lab))
        sil = float(silhouette_score(d, lab, metric="precomputed")) if 2 <= k < len(lab) else -1.0
        rows.append({"threshold": round(float(thr), 3), "k": k, "top2": round(top2, 3), "silhouette": round(sil, 4)})
        if 2 <= k <= 12 and top2 >= min_top2 and (best is None or sil > best[0]):
            best = (sil, float(thr), lab)
    if best is None:                       # nothing qualified: two clusters by k-means style cut
        from sklearn.cluster import AgglomerativeClustering
        lab = AgglomerativeClustering(n_clusters=2, metric="precomputed", linkage="average").fit_predict(d)
        return {"threshold": None, "labels": lab, "table": rows, "fallback": "two-cluster cut"}
    return {"threshold": round(best[1], 3), "labels": best[2], "table": rows, "silhouette": round(best[0], 4)}


def relabel_by_size(labels: np.ndarray, min_size: int, other: int = -1) -> tuple[np.ndarray, list[int]]:
    """Clusters ordered by size (0 = biggest); clusters smaller than `min_size` moments become `other` (-1). Returns (labels, sizes)."""
    counts = np.bincount(labels)
    order = [int(c) for c in np.argsort(-counts, kind="stable")]
    new = np.full(labels.shape, other, int)
    sizes = []
    k = 0
    for c in order:
        if counts[c] < min_size:
            continue
        new[labels == c] = k
        sizes.append(int(counts[c]))
        k += 1
    return new, sizes


def centroids(e: np.ndarray, labels: np.ndarray) -> np.ndarray:
    ks = sorted(k for k in set(labels.tolist()) if k >= 0)
    return l2n(np.stack([e[labels == k].mean(axis=0) for k in ks])) if ks else np.zeros((0, e.shape[1]))


def refine_labels(e: np.ndarray, labels: np.ndarray, iters: int = 3, min_sim: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """A few k-means-like passes: every moment goes to its most similar centroid (the small clusters' moments too, unless the best
    similarity is below `min_sim`, then -1). Returns (labels, similarity matrix to the final centroids)."""
    lab = labels.copy()
    for _ in range(iters):
        c = centroids(e, lab)
        if not len(c):
            break
        sim = e @ c.T
        best = sim.argmax(axis=1)
        new = np.where(sim.max(axis=1) >= min_sim, best, -1)
        if np.array_equal(new, lab):
            break
        lab = new
    c = centroids(e, lab)
    return lab, (e @ c.T if len(c) else np.zeros((len(e), 0)))


# ------------------------------------------------------------------------------------------------ smoothing, turns


def viterbi_smooth(sim: np.ndarray, gaps_after: np.ndarray, penalty: float = 0.15, temp: float = 0.08) -> np.ndarray:
    """Best label path over moments with a cost of `penalty` (in log-likelihood units x temp) per switch. `sim` (M, K) are cosine
    similarities to the K centroids, used as log-likelihoods sim / temp; `gaps_after[i]` is True when there is a break between moment
    i and i+1 (a pause of more than ~1.5 s: the chain restarts, switching is free)."""
    m, k = sim.shape
    if m == 0 or k == 0:
        return np.zeros(m, int)
    ll = sim / temp
    pen = penalty / temp
    score = np.zeros((m, k))
    back = np.zeros((m, k), int)
    score[0] = ll[0]
    for i in range(1, m):
        if gaps_after[i - 1]:
            best = int(score[i - 1].argmax())
            score[i] = ll[i] + score[i - 1][best]
            back[i] = best
            continue
        stay = score[i - 1]
        switch = score[i - 1].max() - pen
        arg = int(score[i - 1].argmax())
        for j in range(k):
            if stay[j] >= switch:
                score[i, j] = stay[j] + ll[i, j]
                back[i, j] = j
            else:
                score[i, j] = switch + ll[i, j]
                back[i, j] = arg
    path = np.zeros(m, int)
    path[-1] = int(score[-1].argmax())
    for i in range(m - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path


def margins(sim: np.ndarray, path: np.ndarray) -> np.ndarray:
    """Similarity of each moment to its assigned centroid minus its best other centroid (0 with a single centroid)."""
    if sim.shape[1] < 2:
        return np.ones(len(path))
    own = sim[np.arange(len(path)), path]
    other = np.where(np.eye(sim.shape[1], dtype=bool)[path], -np.inf, sim).max(axis=1)
    return own - other


def to_turns(m: Moments, path: np.ndarray, marg: np.ndarray, max_gap: float = 0.8, min_len: float = 0.0) -> list[dict]:
    """Consecutive moments with the same label (gap <= max_gap) -> turns {start, end, voice, margin, n, f0, loud_cam}."""
    turns: list[dict] = []
    cur = None
    for i in range(len(path)):
        lab = int(path[i])
        if cur is not None and cur["voice"] == lab and m.t0[i] - cur["end"] <= max_gap:
            cur["end"] = float(m.t1[i]); cur["_i"].append(i)
            continue
        if cur is not None:
            turns.append(cur)
        cur = {"start": float(m.t0[i]), "end": float(m.t1[i]), "voice": lab, "_i": [i]}
    if cur is not None:
        turns.append(cur)
    out = []
    for t in turns:
        idx = t.pop("_i")
        t["n_moments"] = len(idx)
        t["margin"] = round(float(np.mean(marg[idx])), 4)
        f = m.f0[idx]
        f = f[f > 0]
        t["f0_median"] = round(float(np.median(f)), 1) if len(f) else None
        cams = [m.loud_cam[i] for i in idx]
        t["loudest_cam"] = max(set(cams), key=cams.count)
        t["n_cams_mean"] = round(float(np.mean([len(set(m.cams[i])) for i in idx])), 2)
        if t["end"] - t["start"] >= min_len:
            out.append(t)
    return out


def margin_to_confidence(margin: float, n_moments: int, scale: float = 0.12) -> float:
    """0..1: a logistic of the margin, shrunk for turns of one or two moments. Not a calibrated probability."""
    c = 1.0 / (1.0 + math.exp(-(margin - 0.05) / (scale / 4)))
    return round(c * min(1.0, 0.5 + 0.25 * n_moments), 3)


# ------------------------------------------------------------------------------------------------ description


def describe(m: Moments, labels: np.ndarray, t: Table) -> list[dict]:
    """Per cluster: moments, seconds, F0 median / quartiles, loudest phone shares, mean number of phones."""
    out = []
    for k in sorted(set(labels.tolist())):
        idx = np.where(labels == k)[0]
        f = m.f0[idx]
        f = f[f > 0]
        loud = [m.loud_cam[i] for i in idx]
        shares = {c: round(loud.count(c) / len(loud), 3) for c in sorted(set(loud))}
        out.append({"voice": int(k), "moments": int(len(idx)), "seconds": round(len(idx) * BIN_S, 1),
                    "f0_median": round(float(np.median(f)), 1) if len(f) else None,
                    "f0_q25_q75": [round(float(np.percentile(f, 25)), 1), round(float(np.percentile(f, 75)), 1)] if len(f) else None,
                    "f0_voiced_share": round(float(len(f) / len(idx)), 3),
                    "f0_voiced_frames": round(float(np.mean([t.f0v[j] for i in idx for j in m.members[i]])), 3), "loudest_cam_share": shares,
                    "mean_phones": round(float(np.mean([len(set(m.cams[i])) for i in idx])), 2),
                    "mean_phone_disagreement": round(float(np.mean(m.spread[idx])), 4)})
    return out


def gaps_after(m: Moments, max_gap: float = 1.5) -> np.ndarray:
    g = np.zeros(len(m.t0), bool)
    if len(m.t0) > 1:
        g[:-1] = (m.t0[1:] - m.t1[:-1]) > max_gap
    return g


def run(raw_dir: Path, clocks: dict, models: tuple[str, ...] = ("ecapa", "titanet"), min_cluster_s: float = 6.0, penalty: float = 0.15) -> dict:
    """Whole joint clustering: table -> moments -> clusters -> smoothed path -> turns. Returns a dict with arrays (for the identity
    stage) and JSON-able summaries."""
    t = load_table(raw_dir, clocks, models)
    m = build_moments(t)
    pick = pick_threshold(m.emb)
    lab0 = pick["labels"]
    lab, sizes = relabel_by_size(lab0, max(2, int(min_cluster_s / BIN_S)))
    lab, sim = refine_labels(m.emb, lab)
    path = viterbi_smooth(sim, gaps_after(m), penalty=penalty) if sim.shape[1] else np.zeros(len(lab), int)
    marg = margins(sim, path)
    turns = to_turns(m, path, marg)
    for tr in turns:
        tr["confidence"] = margin_to_confidence(tr["margin"], tr["n_moments"])
    return {"table": t, "moments": m, "labels": path, "sim": sim, "turns": turns, "threshold": pick.get("threshold"),
            "threshold_table": pick["table"], "silhouette": pick.get("silhouette"), "sizes": sizes, "models": t.models,
            "voices": describe(m, path, t)}
