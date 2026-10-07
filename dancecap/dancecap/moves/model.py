"""The move-classifier interface and the staged plan (docs/MOVES_RESEARCH.md section 8). NOTHING IS TRAINED OR RUN.

    python -m dancecap.moves model plan [--json]     the staged plan (what, when, where it runs)
    python -m dancecap.moves model status            Stage B readiness from work/moves_dataset/manifest.json
    python -m dancecap.moves model fit|predict       refused in Stage A (the user's decision, 2026-10-06)

The user decided on 2026-10-06: build the infrastructure now - features, proposals, narration labelling, the
dataset - and train our own classifier LATER, once enough dances are narrated. This module fixes the contract a
future classifier must meet so the rest of the pipeline (labels.json, path.json, the fingerprint, the viewer) does
not change when it arrives:

  MoveClassifier.fit(windows)          windows = dataset.load_windows() (work/moves_dataset/windows.npz; X robust
                                       z or X_raw feature units - `input` says which; beat bookkeeping is not in X)
  MoveClassifier.predict_proba(X)      X (n, window, F, 4) pooled dataset.X_FEATURES -> (n, C) over .classes;
                                       CALIBRATED probabilities, or a calibrator attached (`calibration`)
  MoveClassifier.predict(X)            argmax slugs
  MoveClassifier.propose_labels(fs)    a take's FeatureSet -> labels.Labels of provenance 'auto' segments
                                       (annotator 'model:<name>@<version>'); per segment (VIEWER_SPEC 10 / 3.12):
                                       confidence = the segment's mean probability of its move, alternatives =
                                       [{move, p}] top 3 (the chosen move first), calibrated: bool, calibration:
                                       <record id> | null. Uncalibrated output must not be shown as a confidence
                                       (require_calibrated=True refuses it).
  merge_into_gaps(existing, auto)      auto segments only where no human / narration / auto label exists -
                                       classifier output never overwrites the user's labels

The plumbing in propose_labels (window probabilities -> per-eighth votes -> runs -> segments snapped to
half-measures) is tested with a fixed-probability stand-in, not a trained model.
"""
from __future__ import annotations

import abc
import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

VERSION = 1

PLAN = {
    "version": 1,
    "source": "dancecap/docs/MOVES_RESEARCH.md section 8 (staged plan) and section 2 (decisions)",
    "decided": "2026-10-06: infrastructure now, no classifier training or inference until dances are narrated",
    "stages": [
        {"id": "A", "name": "infrastructure (now)", "status": "built",
         "what": ["features.py: interpretable per-frame couple features pooled per eighth (F1, F2)",
                  "proposals.py: Foote-novelty boundary proposals snapped to half-measures, energy bands (F3-F6)",
                  "annotate: narration -> labels.json (human / narration provenance)",
                  "dataset.py: beat-aligned windows + labels, split by dance"],
         "runs": "local CPU; no pods; no new packages"},
        {"id": "B1", "name": "first classifier (pre-fill + rank for the narrator)",
         "gate": {"labelled_dances": 10, "min_instances_per_move": 5, "moves_meeting_min": 5},
         "classes": "only moves with >= min_instances_per_move labelled instances become classes; rarer moves "
                    "stay unlabelled ('other') until they have enough",
         "model": "nearest-prototype / kNN baseline + MiniRocket (aeon in its own venv, or a numpy port: aeon's "
                  "pins conflict with .venv-nopo4d) on 16-eighth windows",
         "inputs": "the interpretable features (fixed role order: leader, follower) + optional frozen TMR/TMR++ "
                   "per-dancer embeddings (SMPL-X body joints 0-21 -> HumanML3D 263-d) + relational features",
         "decode": "HMM / Viterbi over Zouk1 links + proposed links with beat-duration priors (path.py already "
                   "does Viterbi over node instances)",
         "validation": "leave-one-take-out (dataset fold)",
         "use": "pre-fill and rank uncertain segments for the narrator (active learning, boundary-centric); output "
                "provenance 'auto', merged only into gaps",
         "runs": "local CPU (MiniRocket trains in seconds); TMR on the RTX 2080 Ti (fp16, gpu.exclusive)"},
        {"id": "B2", "name": "reported classifier",
         "gate": {"labelled_dances": 25, "min_instances_per_move": 20, "moves_meeting_min": 10},
         "model": "MS-TCN++ / ASFormer head over per-eighth features with a reimplemented timestamp-supervision "
                  "loss (Li, Abu Farha and Gall 2021; trains from the narration anchors)",
         "metrics": "per-beat accuracy, Edit score, F1@{10,25,50}, boundary F at +-0.5 beat, macro-F1, graph-edge "
                    "precision, fingerprint JSD; whole takes held out, later whole couples",
         "runs": "local RTX 2080 Ti, fp16 (Turing: no bf16, no flash-attn), minutes per run"},
        {"id": "C", "name": "pretrained encoders + fine-tuning",
         "gate": {"labelled_dances": 60},
         "model": "self-supervised masked-motion pretraining (MAMP recipe reimplemented, or USDRL/FoundSkelModel "
                  "Apache-2.0) on our unlabelled captures + licence-permitting partner corpora; two-person backbone "
                  "with fixed roles (SkateFormer M=2, a 2P-GCN-style graph with hand/hip edges, or MotionBERT per "
                  "dancer - never its person-averaging action head); MS-TCN++ head, then FACT or DiffAct",
         "runs": {"pretraining": "Runpod via dancecap.remote for corpora > a few hours (within the $30 cap)",
                  "fine-tuning": "local RTX 2080 Ti fp16"},
         "optional_nvidia": ["TMR-SOMA embeddings after SMPL-X -> SOMA (SOMA-X); local if fp32 works on Turing, "
                             "else Runpod", "ProtoMotions follower-alone physics check (Runpod, Linux)",
                             "Cosmos Reason 2 video proposals (Runpod only; BF16, >= 24 GB)"]},
    ],
    "rules": [
        "classifier output is provenance 'auto' and never overwrites human / narration labels (merge_into_gaps)",
        "placeholder labels are never training data; auto labels are never training targets",
        "split by dance (and later by couple), never by window",
        "fixed role order (leader, follower) for every per-dancer input",
        "NC / AGPL / gated weights, datasets and anything derived from them (embeddings, checkpoints) stay in "
        "dancecap/work/ or weights/ (git-ignored) - never in the public MIT Unity repo",
        "local GPU work inside dancecap.gpu.exclusive(label); heavy jobs on Runpod via dancecap.remote only",
    ],
}


@dataclass
class ModelConfig:
    """What a Stage B1 classifier would be configured with (documentation + the propose_labels plumbing)."""
    stage: str = "B1"
    window: int = 16                   # eighths (dataset.DatasetConfig.window)
    hop: int = 4                       # eighths between windows at prediction time
    role_order: tuple = ("lead", "follow")
    features: str = "features.py FEATURES (robust-z, pooled mean/min/max/delta per eighth)"
    decode: str = "argmax"             # argmax now; 'viterbi' (graph links + beat-duration priors) in B1
    snap: str = "half"                 # segment edges on half-measures (type-1 eighths)
    min_confidence: float = 0.0        # segments below this become gaps
    validation: str = "leave-one-take-out"
    device: str = "cpu"
    notes: list = field(default_factory=lambda: ["no training is executed in Stage A"])


class MoveClassifier(abc.ABC):
    """Interface of a move classifier over dataset windows. Subclasses implement fit / predict_proba.

    predict_proba must return CALIBRATED class probabilities before propose_labels output is shown to anyone
    (VIEWER_SPEC 10: auto segments carry "the classifier's calibrated probability ... and the calibration record it
    came from"). Either the model is calibrated by construction, or a calibrator (e.g. temperature scaling fitted on
    held-out folds) is applied inside predict_proba and described by `calibration` = {id, method, fitted_on, ...}.
    Until then calibration is None and every segment says calibrated: false.

    input: 'z' (windows.npz X, per-take robust z) or 'raw' (X_raw, feature units; normalise with training-fold
    statistics inside the model)."""
    name = "abstract"
    version = "0"
    input = "z"

    def __init__(self, classes: list[str] | None = None, config: ModelConfig | None = None,
                 calibration: dict | None = None):
        self.classes: list[str] = list(classes or [])
        self.config = config or ModelConfig()
        self.calibration: dict | None = calibration

    @property
    def calibrated(self) -> bool:
        return bool(self.calibration)

    @abc.abstractmethod
    def fit(self, windows: dict) -> "MoveClassifier":
        """windows: dataset.load_windows() dict (X, y, classes, fold, ...). Sets self.classes."""

    @abc.abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """(n, window, F, 4) -> (n, len(self.classes)) probabilities (rows sum to 1)."""

    def predict(self, X: np.ndarray) -> list[str]:
        p = self.predict_proba(X)
        return [self.classes[i] for i in np.argmax(p, 1)]

    def propose_labels(self, fs, take: str | None = None, graph=None, min_confidence: float | None = None,
                       require_calibrated: bool = False):
        """labels.Labels of provenance 'auto' for a take's FeatureSet (reference time, couple tier). With
        require_calibrated (anything shown in the viewer) an uncalibrated classifier is refused."""
        from .dataset import windows_for
        if require_calibrated and not self.calibrated:
            raise ValueError(f"{self._annotator()} has no calibration record: its probabilities cannot be shown as "
                             "a confidence (VIEWER_SPEC 10) - attach a calibrator first")
        cfg = self.config
        X, starts = windows_for(fs, cfg.window, cfg.hop, kind=self.input)
        if not len(starts):
            return segments_from_votes(fs, np.zeros((len(fs.eighths["t0"]), len(self.classes))), self.classes,
                                       take or fs.take, self._annotator(), cfg, min_confidence, self.calibration)
        P = self.predict_proba(X)
        votes = window_votes(len(fs.eighths["t0"]), starts, P, cfg.window)
        lab = segments_from_votes(fs, votes, self.classes, take or fs.take, self._annotator(), cfg, min_confidence,
                                  self.calibration)
        if graph is not None:
            from .labels import validate_labels
            errs = validate_labels(lab, graph)
            if errs:
                raise ValueError("; ".join(errs[:5]))
        return lab

    def _annotator(self) -> str:
        return f"model:{self.name}@{self.version}"


def window_votes(n_eighths: int, starts: list[int], P: np.ndarray, window: int) -> np.ndarray:
    """Per-eighth class votes (n_eighths, C): the mean probability of every window covering the eighth, each window
    weighted by a Gaussian centred on its middle (sd = window / 4), so the centre of a window counts most."""
    C = P.shape[1]
    V = np.zeros((n_eighths, C))
    W = np.zeros(n_eighths)
    off = np.arange(window)
    g = np.exp(-0.5 * ((off - (window - 1) / 2) / (window / 4)) ** 2)
    for s, p in zip(starts, P):
        idx = s + off
        ok = idx < n_eighths
        V[idx[ok]] += g[ok, None] * p[None, :]
        W[idx[ok]] += g[ok]
    return np.where(W[:, None] > 0, V / np.maximum(W[:, None], 1e-12), 0.0)


def segments_from_votes(fs, votes: np.ndarray, classes: list[str], take: str, annotator: str,
                        cfg: ModelConfig | None = None, min_confidence: float | None = None,
                        calibration: dict | None = None, n_alternatives: int = 3):
    """Per-eighth votes -> runs of the argmax class -> segments whose inner edges are snapped to half-measures
    (type-1 eighths); runs below min_confidence (or with no vote) become gaps. Provenance 'auto'. Each segment:
    confidence = mean vote probability of its move over the segment; alternatives = the top n_alternatives classes
    by that mean ([{move, p}], the chosen move first); calibrated / calibration = whether the probabilities are
    calibrated and the record id they came from (VIEWER_SPEC 10)."""
    from .labels import Labels, Segment
    cfg = cfg or ModelConfig()
    mc = cfg.min_confidence if min_confidence is None else float(min_confidence)
    e = fs.eighths
    N = len(e["t0"])
    cal_id = calibration.get("id") if calibration else None
    lab = Labels(take=take, segments=[], time_base="reference")
    lab.extra["provenance_note"] = "automatic classifier output - unreviewed (provenance 'auto')"
    lab.extra["classifier"] = {"annotator": annotator, "calibrated": bool(calibration), "calibration": calibration}
    if N == 0 or votes.shape[1] == 0:
        return lab
    has = votes.sum(1) > 0
    arg = np.where(has, votes.argmax(1), -1)
    runs, i = [], 0
    while i < N:
        j = i
        while j + 1 < N and arg[j + 1] == arg[i]:
            j += 1
        runs.append([i, j + 1, int(arg[i])])
        i = j + 1
    # snap inner edges to the nearest half-measure start
    half = np.flatnonzero(np.asarray(e["slot"], int) % 4 == 0)
    if len(half) and cfg.snap == "half":
        for k in range(1, len(runs)):
            b = runs[k][0]
            nb = int(half[np.argmin(np.abs(half - b))])
            nb = min(max(nb, runs[k - 1][0] + 1), runs[k][1] - 1) if runs[k][1] - runs[k - 1][0] > 1 else b
            runs[k - 1][1] = runs[k][0] = nb
    for a, b, c in runs:
        if c < 0 or b <= a:
            continue
        mean_p = votes[a:b].mean(0)
        cm = float(mean_p[c])
        if cm < mc:
            continue
        order = [c] + [int(j) for j in np.argsort(-mean_p) if int(j) != c]
        alts = [{"move": classes[j], "p": round(float(mean_p[j]), 4)} for j in order[:n_alternatives]]
        lab.segments.append(Segment(t0=float(e["t0"][a]), t1=float(e["t1"][b - 1]), move=classes[c],
                                    provenance="auto", confidence=round(min(max(cm, 0.0), 1.0), 4),
                                    annotator=annotator, notes="classifier output - review before use",
                                    extra={"alternatives": alts, "calibrated": bool(calibration),
                                           "calibration": cal_id}))
    return lab


def merge_into_gaps(existing, auto, min_s: float = 0.2):
    """A copy of `existing` (labels.Labels) plus the parts of `auto`'s segments that fall in its couple-tier gaps
    (pieces shorter than min_s dropped). Existing segments - human, narration or auto - are never changed."""
    from .labels import Labels, Segment, new_id
    out = Labels.from_dict(existing.to_dict())
    occupied = sorted((s.t0, s.t1) for s in out.tier("couple"))
    for s in auto.tier("couple"):
        pieces = [(s.t0, s.t1)]
        for a, b in occupied:
            nxt = []
            for p0, p1 in pieces:
                if b <= p0 or a >= p1:
                    nxt.append((p0, p1))
                    continue
                if a > p0:
                    nxt.append((p0, a))
                if b < p1:
                    nxt.append((b, p1))
            pieces = nxt
        for p0, p1 in pieces:
            if p1 - p0 >= min_s:
                d = s.to_dict()
                d.update({"id": new_id(), "t0": p0, "t1": p1})
                out.segments.append(Segment.from_dict(d))
    return out


# ----------------------------------------------------------------------------------------------- readiness

def readiness(per_take: list[dict] | None = None, manifest: dict | None = None) -> dict:
    """Stage B gates vs the labelled data: per_take = dataset manifest 'takes' entries (training_segments,
    segment_moves). Returns {stages: {id: {met, have, need}}, summary}."""
    takes = per_take if per_take is not None else (manifest or {}).get("takes", [])
    labelled = [t for t in takes if t.get("training_segments", 0) > 0]
    inst: dict[str, int] = {}
    for t in labelled:
        for m, n in (t.get("segment_moves") or {}).items():
            inst[m] = inst.get(m, 0) + int(n)
    out = {}
    for st in PLAN["stages"]:
        g = st.get("gate")
        if not g:
            continue
        k = g.get("min_instances_per_move", 1)
        have = {"labelled_dances": len(labelled),
                "moves_with_min_instances": sum(1 for n in inst.values() if n >= k),
                "moves_below_min": sorted(m for m, n in inst.items() if n < k)}
        met = have["labelled_dances"] >= g["labelled_dances"]
        if "min_instances_per_move" in g:
            met = met and have["moves_with_min_instances"] >= g.get("moves_meeting_min", 1)
        out[st["id"]] = {"met": bool(met), "have": have, "need": g}
    b1 = out.get("B1", {})
    g1 = PLAN["stages"][1]["gate"]
    summary = (f"{len(labelled)} labelled dance(s), {len(inst)} move(s) "
               f"({sum(1 for n in inst.values() if n >= g1['min_instances_per_move'])} with >= "
               f"{g1['min_instances_per_move']} instances); B1 gate {'MET' if b1.get('met') else 'not met'} "
               f"(needs {g1['labelled_dances']} dances and {g1['moves_meeting_min']} moves with >= "
               f"{g1['min_instances_per_move']} instances; rarer moves are not classes)")
    return {"stages": out, "instances": inst, "summary": summary}


# ----------------------------------------------------------------------------------------------- CLI

def cmd_model(args: argparse.Namespace) -> int:
    if args.action == "plan":
        if args.json:
            print(json.dumps({"plan": PLAN, "config": asdict(ModelConfig())}, indent=1))
            return 0
        print(f"[moves] staged plan ({PLAN['source']}); decided {PLAN['decided']}")
        for st in PLAN["stages"]:
            print(f"[moves] Stage {st['id']} - {st['name']}" + (f"  gate {st['gate']}" if st.get("gate") else ""))
            for k in ("model", "decode", "runs"):
                if st.get(k):
                    print(f"[moves]     {k}: {st[k]}")
        for r in PLAN["rules"]:
            print(f"[moves] rule: {r}")
        return 0
    if args.action == "status":
        from .dataset import default_out
        mp = Path(args.manifest) if args.manifest else default_out() / "manifest.json"
        if not mp.exists():
            print(f"[moves] no dataset manifest at {mp}: run `python -m dancecap.moves dataset` once dances are "
                  "narrated (it refuses while only placeholder labels exist)")
            return 0
        r = readiness(manifest=json.loads(mp.read_text(encoding="utf-8")))
        print(f"[moves] {r['summary']}")
        for k, v in r["stages"].items():
            print(f"[moves]   {k}: {'MET' if v['met'] else 'not met'}  have {v['have']}  need {v['need']}")
        return 0
    print(f"[moves] model {args.action}: REFUSED - Stage A builds the infrastructure only; no classifier is trained "
          "or run (the user's decision, 2026-10-06). See `python -m dancecap.moves model plan` and "
          "`model status` for the gates.")
    return 2


def register(sub) -> None:
    p = sub.add_parser("model", help="classifier interface + staged plan: plan | status (fit / predict are refused "
                                     "in Stage A)")
    p.add_argument("action", choices=("plan", "status", "fit", "predict"))
    p.add_argument("--json", action="store_true")
    p.add_argument("--manifest", help="dataset manifest (default work/moves_dataset/manifest.json)")
    p.set_defaults(func=cmd_model)
