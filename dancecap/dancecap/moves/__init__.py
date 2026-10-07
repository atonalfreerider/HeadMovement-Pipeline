"""Dance moves: the zouk dance graph, move timelines (labels), graph paths and dance fingerprints.

    python -m dancecap.moves graph                      Zouk1.json + taxonomy/additions.toml -> taxonomy/zouk_graph.json
    python -m dancecap.moves placeholder <take>         PLACEHOLDER timeline for the prototype (not an analysis)
    python -m dancecap.moves path <take>                labels.json -> path.json (Viterbi over node instances)
    python -m dancecap.moves fingerprint <take>         path.json -> fingerprint.json (+ review/fingerprint.png)
    python -m dancecap.moves export <take> [--capture]  Unity copies in StreamingAssets/<capture>/moves/
    python -m dancecap.moves validate [<take>]          check every file against the version-1 contracts
    python -m dancecap.moves review [<take>]            work/review/moves_graph.png (+ the take's fingerprint.png)
    python -m dancecap.moves run <take>                 path + fingerprint + review + export in one go

Files (all JSON, version 1; contracts in graph.py / labels.py / path.py / fingerprint.py / export.py):
  taxonomy/zouk_graph.json             the graph (tracked; regenerated from Zouk1.json + taxonomy/additions.toml
                                       + taxonomy/vocabulary_research.json when present)
  work/<take>/moves/labels.json        move timeline (couple tier, non-overlapping segments)
  work/<take>/moves/path.json          resolved node instances + links, new links flagged
  work/<take>/moves/fingerprint.json   dwell time per node, link counts, energy bands, counterbalance share
  head-movement/Assets/StreamingAssets/<capture>/moves/{graph,labels,path,fingerprint}.json  (capture time, Unity)

Optional sub-commands are added by sibling modules (features, proposals, dataset, model, annotate) that define
register(subparsers); see __main__.py.
"""
from __future__ import annotations

from pathlib import Path

from ..take import PACKAGE_ROOT, WORKSPACE

TAXONOMY = PACKAGE_ROOT / "taxonomy"
GRAPH_PATH = TAXONOMY / "zouk_graph.json"
ADDITIONS_PATH = TAXONOMY / "additions.toml"
VOCABULARY_PATH = TAXONOMY / "vocabulary_research.json"
ZOUK1_PATH = WORKSPACE / "Zouk1.json"
REVIEW_DIR = PACKAGE_ROOT / "work" / "review"
UNITY_STREAMING = WORKSPACE / "head-movement" / "Assets" / "StreamingAssets"

PROVENANCES = ("human", "narration", "auto", "placeholder")
TIERS = ("couple",)
PLACEHOLDER_BANNER = "placeholder - not an analysis"


def work_dir(take: str) -> Path:
    return PACKAGE_ROOT / "work" / take


def moves_dir(take: str) -> Path:
    return work_dir(take) / "moves"


def counterbalance_path(take: str) -> Path:
    return work_dir(take) / "counterbalance" / "counterbalance.json"
