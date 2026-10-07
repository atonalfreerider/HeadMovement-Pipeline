"""Training windows from features + move labels -> work/moves_dataset/ (the Stage B input; NOTHING is trained here).

    python -m dancecap.moves dataset [--takes larissa_kadu ...] [--window 16] [--hop 4] [--out DIR]

For every take with work/<take>/moves/features.npz (features.py) and work/<take>/moves/labels.json:
  - every labels.json is validated against the contract AND the graph (labels.validate_labels with
    taxonomy/zouk_graph.json: unknown moves, aliases instead of slugs, overlapping couple segments, bad times). Any
    invalid file REFUSES the dataset (DatasetError) - fix the labels first, nothing is skipped silently.
  - training labels = couple-tier segments with an EXPLICIT provenance 'human' or 'narration' in the file (a segment
    without a 'provenance' key is never training data - labels.Segment would default it to 'human'). 'placeholder'
    segments (a random walk for the viewer prototype) and 'auto' segments (proposals / classifier output) are NEVER
    training targets. A whole file marked as a placeholder (top-level placeholder:true, or generator.module
    dancecap.moves.placeholder) is excluded; if such a file contains human / narration segments the dataset is
    refused (ambiguous - remove the placeholder marker when the labels are real). Exclusions are counted in the
    manifest under `excluded`. Labels in 'video' time are converted to reference time.
  - each pooled eighth gets the move of the training segment covering the largest part of it (>= 50 %).
  - beat-aligned windows: `window` eighths (default 16 = two measures) starting on half-measures (type-1 eighths,
    slots 0 and 4). Windows never contain a grid gap or an eighth with < 50 % valid frames (features.py); the start
    phase is reset after every gap, then every `hop` eighths (hop must be a multiple of 4 = half a measure). A window
    is kept when >= `min_labelled` of its eighths are labelled. Its label = the move covering most of its labelled
    eighths; `purity` = that share; `center` = the move at the window's centre eighth.
  - the split is BY DANCE (never by window: windows of one dance overlap): `fold` = take index (leave-one-take-out),
    and `split` = train / val / test per take once there are >= 3 labelled takes (deterministic by take-name hash;
    'loto' before that).
Refuses (DatasetError, exit code 2) when no take has a human or narration label.

Inputs to a classifier (X_FEATURES): every motion feature of features.py - the beat-bookkeeping columns (group
'beat': eighth_pos = absolute grid position, beat_phase, measure_phase) are NOT in X (eighth_pos would let a model
learn "where in the dance" instead of "what motion"); they are kept apart in X_beat. X is the per-take robust z
(features.py); X_raw keeps the same pooled values in feature units so Stage B can normalise with statistics of its
training folds (absolute levels, e.g. a deep vs a shallow dip, survive).

Outputs (work/moves_dataset/, git-ignored):
  windows.npz     X (n, window, F, 4) float32 per-take robust z | X_raw (n, window, F, 4) float32 feature units |
                  X_beat (n, window, B, 4) float32 beat bookkeeping (raw) | y (n,) class index | classes (C,) move
                  slugs | take (n,) | takes (K,) | t0, t1 (n,) reference s | start_eighth (n,) | purity, labelled (n,) |
                  center (n,) class index or -1 | fold (n,) | split (n,) | feature_names (F,) | beat_feature_names
                  (B,) | stats (4,)
  eighths.parquet one row per pooled eighth of every take: take, eighth, grid_index, t0, t1, slot, valid_frac, move,
                  provenance, segment, overlap, then z__<feature>__<stat> (robust z), raw__<feature>__<stat> (feature
                  units) and beat__<feature>__<stat> (bookkeeping) columns (eighths.csv when pyarrow is missing)
  manifest.json   config, inputs (what X / X_raw / X_beat hold), per-take sources + sha, counts, exclusions,
                  classes (windows / segments / seconds / takes), split, Stage B readiness (model.readiness)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .features import FEATURES, INDEX, STATS, FeatureSet

VERSION = 1
TRAINING_PROVENANCES = ("human", "narration")
PLACEHOLDER_MODULE = "dancecap.moves.placeholder"
X_FEATURES = [f.name for f in FEATURES if f.group != "beat"]       # classifier inputs (motion only)
BEAT_FEATURES = [f.name for f in FEATURES if f.group == "beat"]    # bookkeeping, kept apart (X_beat)
X_IDX = [INDEX[n] for n in X_FEATURES]
BEAT_IDX = [INDEX[n] for n in BEAT_FEATURES]
MISSING_PROVENANCE = "unspecified"                                  # counted, never trained on


class DatasetError(ValueError):
    pass


@dataclass
class DatasetConfig:
    window: int = 16                  # eighths per window (two measures)
    hop: int = 4                      # eighths between window starts (a multiple of 4 = half a measure)
    min_labelled: float = 0.5         # fraction of a window's eighths that must carry a training label
    min_eighth_overlap: float = 0.5   # an eighth takes a segment's move when the segment covers >= this of it
    min_valid_frac: float = 0.5       # windows only contain eighths with >= this fraction of valid frames
    provenances: tuple = TRAINING_PROVENANCES


def default_root() -> Path:
    from ..take import PACKAGE_ROOT
    return PACKAGE_ROOT / "work"


def default_out() -> Path:
    return default_root() / "moves_dataset"


def _sha16(p: Path) -> str | None:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16] if Path(p).exists() else None


def check_hop(hop: int) -> int:
    hop = int(hop)
    if hop <= 0 or hop % 4:
        raise DatasetError(f"--hop {hop}: windows start on half-measures, so the hop must be a positive multiple of 4 "
                           "eighths (4 = half a measure, 8 = a measure)")
    return hop


def is_placeholder_file(doc: dict) -> bool:
    """A labels.json written as a placeholder: top-level placeholder:true or a placeholder generator."""
    if doc.get("placeholder"):
        return True
    g = doc.get("generator")
    if isinstance(g, dict):
        return str(g.get("module", "")).startswith(PLACEHOLDER_MODULE)
    return isinstance(g, str) and "placeholder" in g.lower()


def read_labels(path: Path, graph=None) -> tuple:
    """(Labels with explicit provenance only, raw doc, validation errors). Segments without a 'provenance' key get
    provenance 'unspecified' (labels.Segment would default them to 'human'); segments of a placeholder generator
    (placeholder_node) are forced to 'placeholder'."""
    from .labels import Labels, validate_labels
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    errs = validate_labels(doc, graph)
    try:
        lab = Labels.from_dict(doc)
    except (KeyError, TypeError, ValueError) as e:
        return None, doc, errs or [f"unreadable labels: {e}"]
    for s, d in zip(lab.segments, doc.get("segments") or []):
        if not isinstance(d, dict) or "provenance" not in d:
            s.provenance = MISSING_PROVENANCE
        elif "placeholder_node" in d:
            s.provenance = "placeholder"
    return lab, doc, errs


def training_segments(labels, provenances=TRAINING_PROVENANCES) -> tuple[list, dict]:
    """(couple-tier training segments in reference time, {provenance: count} of every segment)."""
    from .labels import to_reference
    ref = to_reference(labels)
    counts: dict[str, int] = {}
    for s in ref.segments:
        counts[s.provenance] = counts.get(s.provenance, 0) + 1
    keep = [s for s in ref.tier("couple") if s.provenance in provenances]
    return keep, counts


def eighth_labels(fs: FeatureSet, segments: list, min_overlap: float = 0.5) -> list[dict | None]:
    """Per pooled eighth: {move, provenance, segment, overlap, confidence} of the segment covering most of it."""
    e = fs.eighths
    out: list[dict | None] = []
    for a, b in zip(e["t0"], e["t1"]):
        best, bo = None, 0.0
        for s in segments:
            ov = max(0.0, min(b, s.t1) - max(a, s.t0)) / max(b - a, 1e-9)
            if ov > bo:
                best, bo = s, ov
        out.append(None if best is None or bo < min_overlap else
                   {"move": best.move, "provenance": best.provenance, "segment": best.id, "overlap": round(bo, 4),
                    "confidence": best.confidence})
    return out


def window_starts(fs: FeatureSet, window: int, hop: int, min_valid_frac: float = 0.5) -> list[int]:
    """Eighth indices where a window can start: a half-measure (slot 0 / 4) whose `window` eighths are consecutive on
    the grid and each have >= min_valid_frac valid frames. Inside each such block the starts are thinned to every
    `hop` eighths (by grid index), the phase restarting at the block's first candidate - a gap never shifts the later
    windows off the half-measures."""
    from .features import eighth_valid_fraction
    check_hop(hop)
    e = fs.eighths
    N = len(e["t0"])
    if N < window:
        return []
    idx = np.asarray(e["index"], int)
    slots = np.asarray(e["slot"], int)
    good = eighth_valid_fraction(fs) >= min_valid_frac
    # block id: runs of good eighths with consecutive grid indices
    block = np.full(N, -1, int)
    k = -1
    for i in range(N):
        if not good[i]:
            continue
        if i == 0 or not good[i - 1] or idx[i] != idx[i - 1] + 1:
            k += 1
        block[i] = k
    starts, phase = [], {}
    for i in range(N - window + 1):
        if slots[i] % 4 or block[i] < 0 or block[i + window - 1] != block[i]:
            continue
        p0 = phase.setdefault(int(block[i]), int(idx[i]))
        if (idx[i] - p0) % hop == 0:
            starts.append(i)
    return starts


def windows_for(fs: FeatureSet, window: int = 16, hop: int = 4, kind: str = "z") -> tuple[np.ndarray, list[int]]:
    """Unlabelled windows (n, window, F, 4) of a take's pooled X_FEATURES - robust z (kind 'z', as dataset X) or
    feature units (kind 'raw', as X_raw) - plus their start eighths (for prediction with
    model.MoveClassifier.propose_labels). NaN (e.g. no counterbalance.json) stays NaN."""
    src = {"z": fs.pooled_z, "raw": fs.pooled}[kind][:, X_IDX]
    starts = window_starts(fs, window, hop)
    X = np.stack([src[i:i + window] for i in starts]) if starts else \
        np.zeros((0, window, len(X_IDX), len(STATS)), np.float32)
    return X.astype(np.float32), starts


def _majority(labs: list[dict | None]) -> tuple[str | None, float, float]:
    """(move, purity among labelled, labelled fraction) weighted by overlap."""
    w: dict[str, float] = {}
    for x in labs:
        if x:
            w[x["move"]] = w.get(x["move"], 0.0) + x["overlap"]
    n_lab = sum(1 for x in labs if x)
    if not w:
        return None, 0.0, 0.0
    m = max(sorted(w), key=lambda k: w[k])
    return m, w[m] / sum(w.values()), n_lab / len(labs)


def _split_for(takes: list[str]) -> dict[str, str]:
    """train / val / test by dance (deterministic hash order) once >= 3 takes are labelled; 'loto' before."""
    if len(takes) < 3:
        return {t: "loto" for t in takes}
    order = sorted(takes, key=lambda t: hashlib.sha1(t.encode()).hexdigest())
    out = {t: "train" for t in takes}
    out[order[0]] = "test"
    if len(takes) >= 4:
        out[order[1]] = "val"
    return out


def discover_takes(root: Path) -> list[str]:
    return sorted(p.parent.parent.name for p in Path(root).glob("*/moves/features.npz")
                  if (p.parent / "labels.json").exists())


def build_dataset(takes: list[str] | None = None, cfg: DatasetConfig | None = None, out_dir: Path | None = None,
                  root: Path | None = None, graph=None) -> dict:
    """Build windows.npz + eighths table + manifest.json; returns the manifest. Raises DatasetError when a labels file
    is invalid (contract or graph), when a placeholder-marked file holds human / narration segments, or when there
    is no human / narration label in any of the takes. graph: moves.graph.Graph (default taxonomy/zouk_graph.json)."""
    from . import features as F
    from .graph import load_graph
    cfg = cfg or DatasetConfig()
    check_hop(cfg.hop)
    graph = graph or load_graph()
    root = Path(root) if root else default_root()
    out_dir = Path(out_dir) if out_dir else default_out()
    takes = list(takes) if takes else discover_takes(root)
    if not takes:
        raise DatasetError(f"no take under {root} has both moves/features.npz and moves/labels.json")
    per_take, rows, Xs, Xr, Xb = [], [], [], [], []
    missing, invalid = [], []
    for take in takes:
        npz_p = root / take / "moves" / "features.npz"
        lab_p = root / take / "moves" / "labels.json"
        if not npz_p.exists() or not lab_p.exists():
            missing.append(f"{take}: " + ", ".join(str(p.relative_to(root)) for p in (npz_p, lab_p)
                                                   if not p.exists()) + " missing")
            continue
        labels, doc, errs = read_labels(lab_p, graph)
        if errs:
            invalid.append(f"{take} ({lab_p}): " + "; ".join(errs[:5]) + (" ..." if len(errs) > 5 else ""))
            continue
        fs = F.load(npz_p)
        segs, counts = training_segments(labels, cfg.provenances)
        excluded_file = None
        if is_placeholder_file(doc):
            if segs:
                invalid.append(f"{take} ({lab_p}): marked as a placeholder file (placeholder:true / placeholder "
                               f"generator) but holds {len(segs)} human / narration segment(s) - remove the "
                               "placeholder marker if these labels are real, or delete them")
                continue
            excluded_file = "placeholder file (placeholder:true / generator dancecap.moves.placeholder)"
        labs = eighth_labels(fs, segs, cfg.min_eighth_overlap)
        info = {"take": take, "features": str(npz_p), "features_sha256_16": _sha16(npz_p), "labels": str(lab_p),
                "labels_sha256_16": _sha16(lab_p), "segments_by_provenance": counts,
                "training_segments": len(segs),
                "excluded": {p: n for p, n in counts.items() if p not in cfg.provenances},
                "excluded_file": excluded_file,
                "n_eighths": len(labs), "labelled_eighths": sum(1 for x in labs if x),
                "labelled_seconds": round(sum(s.duration for s in segs), 3), "windows": 0,
                "segment_moves": {}}
        for s in segs:
            info["segment_moves"][s.move] = info["segment_moves"].get(s.move, 0) + 1
        per_take.append((take, fs, labs, segs, info))
    if invalid:
        raise DatasetError("invalid labels (fix them before building a dataset; nothing was written): "
                           + " | ".join(invalid))
    labelled = [p for p in per_take if p[4]["training_segments"] > 0]
    if not labelled:
        detail = "; ".join(f"{i['take']}: {i['segments_by_provenance'] or 'no segments'}"
                           + (" (placeholder file)" if i["excluded_file"] else "") for *_, i in per_take)
        raise DatasetError("no human or narration labels to train from (placeholder, auto and unspecified-provenance "
                           f"segments are excluded) - {detail or 'no usable take'}"
                           + (f"; {'; '.join(missing)}" if missing else "")
                           + ". Narrate / label the dances first (labels.json segments with provenance 'human' or "
                             "'narration').")
    split = _split_for([p[0] for p in labelled])
    classes = sorted({s.move for p in labelled for s in p[3]})
    cidx = {c: i for i, c in enumerate(classes)}
    win = {k: [] for k in ("y", "take", "t0", "t1", "start", "purity", "labelled", "center", "fold", "split")}
    for k, (take, fs, labs, segs, info) in enumerate(labelled):
        e = fs.eighths
        vf = F.eighth_valid_fraction(fs)
        for i in window_starts(fs, cfg.window, cfg.hop, cfg.min_valid_frac):
            sl = labs[i:i + cfg.window]
            m, purity, frac = _majority(sl)
            if m is None or frac < cfg.min_labelled:
                continue
            c = sl[cfg.window // 2]
            Xs.append(fs.pooled_z[i:i + cfg.window][:, X_IDX])
            Xr.append(fs.pooled[i:i + cfg.window][:, X_IDX])
            Xb.append(fs.pooled[i:i + cfg.window][:, BEAT_IDX])
            for key, v in (("y", cidx[m]), ("take", k), ("t0", float(e["t0"][i])),
                           ("t1", float(e["t1"][i + cfg.window - 1])), ("start", i), ("purity", purity),
                           ("labelled", frac), ("center", cidx[c["move"]] if c else -1), ("fold", k),
                           ("split", split[take])):
                win[key].append(v)
            info["windows"] += 1
        for j, x in enumerate(labs):
            row = {"take": take, "eighth": j, "grid_index": int(e["index"][j]), "t0": float(e["t0"][j]),
                   "t1": float(e["t1"][j]), "slot": int(e["slot"][j]), "valid_frac": float(vf[j]),
                   "move": x["move"] if x else None, "provenance": x["provenance"] if x else None,
                   "segment": x["segment"] if x else None, "overlap": x["overlap"] if x else 0.0}
            for n in X_FEATURES:
                for si, st in enumerate(STATS):
                    row[f"z__{n}__{st}"] = float(fs.pooled_z[j, INDEX[n], si])
                    row[f"raw__{n}__{st}"] = float(fs.pooled[j, INDEX[n], si])
            for n in BEAT_FEATURES:
                for si, st in enumerate(STATS):
                    row[f"beat__{n}__{st}"] = float(fs.pooled[j, INDEX[n], si])
            rows.append(row)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(win["y"])

    def stack(lst, nf):
        return np.stack(lst).astype(np.float32) if lst else np.zeros((0, cfg.window, nf, len(STATS)), np.float32)

    X, X_raw, X_beat = stack(Xs, len(X_IDX)), stack(Xr, len(X_IDX)), stack(Xb, len(BEAT_IDX))
    np.savez_compressed(out_dir / "windows.npz", version=np.int64(VERSION), X=X, X_raw=X_raw, X_beat=X_beat,
                        y=np.asarray(win["y"], int), classes=np.array(classes), take=np.asarray(win["take"], int),
                        takes=np.array([p[0] for p in labelled]), t0=np.asarray(win["t0"]), t1=np.asarray(win["t1"]),
                        start_eighth=np.asarray(win["start"], int), purity=np.asarray(win["purity"]),
                        labelled=np.asarray(win["labelled"]), center=np.asarray(win["center"], int),
                        fold=np.asarray(win["fold"], int), split=np.array(win["split"], dtype="<U8"),
                        feature_names=np.array(X_FEATURES), beat_feature_names=np.array(BEAT_FEATURES),
                        stats=np.array(STATS))
    table = _write_table(rows, out_dir)
    cls = []
    for c in classes:
        segs_c = [s for p in labelled for s in p[3] if s.move == c]
        cls.append({"slug": c, "index": cidx[c], "windows": int(sum(1 for y in win["y"] if y == cidx[c])),
                    "segments": len(segs_c), "seconds": round(sum(s.duration for s in segs_c), 3),
                    "takes": sorted({p[0] for p in labelled if any(s.move == c for s in p[3])})})
    from .model import readiness
    manifest = {
        "version": VERSION, "kind": "moves_dataset", "created": _now(), "config": asdict(cfg),
        "note": "Training windows for the Stage B classifier (docs/MOVES_RESEARCH.md section 8). Nothing has been "
                "trained. Split by dance only; placeholder, auto and unspecified-provenance labels are excluded; "
                "every labels.json was validated against the graph.",
        "inputs": {"X": "per-take robust z of the pooled motion features (features.py robust_z)",
                   "X_raw": "the same pooled values in feature units: normalise in Stage B with statistics of the "
                            "training folds only (keeps absolute levels comparable across takes)",
                   "X_beat": "beat bookkeeping (raw pooled), NOT classifier input by default",
                   "excluded_from_X": BEAT_FEATURES,
                   "why": "eighth_pos is the absolute grid position (a ramp through the dance after per-take z): a "
                          "shortcut on 'where in the dance', not motion; beat/measure phase are fixed by the "
                          "half-measure-aligned window starts",
                   "feature_names": X_FEATURES, "beat_feature_names": BEAT_FEATURES},
        "takes": [p[4] for p in per_take], "missing": missing, "classes": cls, "n_windows": n,
        "window_shape": list(X.shape[1:]), "stats": list(STATS), "feature_names": X_FEATURES,
        "split": {"method": "by dance: fold = take index (leave-one-take-out); split = train/val/test per take "
                            "(deterministic hash order) from 3 labelled takes", "takes": split},
        "files": {"windows": "windows.npz", "eighths": table.name},
        "readiness": readiness(per_take=[p[4] for p in labelled]),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest


def load_windows(path: Path | None = None) -> dict:
    """work/moves_dataset/windows.npz (or `path`) as a dict of arrays - the input of model.MoveClassifier.fit."""
    p = Path(path) if path else default_out() / "windows.npz"
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _write_table(rows: list[dict], out_dir: Path) -> Path:
    import pandas as pd
    df = pd.DataFrame(rows)
    try:
        p = out_dir / "eighths.parquet"
        df.to_parquet(p, index=False)
    except (ImportError, ValueError):
        p = out_dir / "eighths.csv"
        df.to_csv(p, index=False)
    return p


def _now() -> str:
    from .labels import now_iso
    return now_iso()


def cmd_dataset(args: argparse.Namespace) -> int:
    cfg = DatasetConfig(window=args.window, hop=args.hop, min_labelled=args.min_labelled)
    try:
        man = build_dataset(args.takes or None, cfg, Path(args.out) if args.out else None)
    except DatasetError as e:
        print(f"[moves] dataset REFUSED: {e}")
        return 2
    out = Path(args.out) if args.out else default_out()
    print(f"[moves] wrote {out}: {man['n_windows']} windows {man['window_shape']} from "
          f"{sum(1 for t in man['takes'] if t['training_segments'])} labelled take(s), {len(man['classes'])} moves")
    for c in man["classes"]:
        print(f"[moves]   {c['slug']}: {c['windows']} windows, {c['segments']} segments, {c['seconds']} s, "
              f"takes {c['takes']}")
    for t in man["takes"]:
        if t["excluded"] or t["excluded_file"]:
            print(f"[moves]   {t['take']}: excluded (not training labels) {t['excluded']}"
                  + (f" - {t['excluded_file']}" if t["excluded_file"] else ""))
    r = man["readiness"]
    print(f"[moves] Stage B readiness: {r['summary']}")
    return 0


def register(sub) -> None:
    d = DatasetConfig()
    p = sub.add_parser("dataset", help="training windows from features + human/narration labels -> "
                                       "work/moves_dataset/ (refuses placeholder-only or invalid labels; trains "
                                       "nothing)")
    p.add_argument("--takes", nargs="*", help="takes (default: every work/<take> with features.npz + labels.json)")
    p.add_argument("--window", type=int, default=d.window, help="eighths per window (16 = two measures)")
    p.add_argument("--hop", type=int, default=d.hop, help="eighths between window starts (a multiple of 4)")
    p.add_argument("--min-labelled", type=float, default=d.min_labelled)
    p.add_argument("--out", help="output directory (default work/moves_dataset)")
    p.set_defaults(func=cmd_dataset)
