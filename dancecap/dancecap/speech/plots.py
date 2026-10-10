"""Two review images (matplotlib, local, seconds): the voice timeline on the master clock, and the embedding space coloured by voice and by phone.

    python -m dancecap.speech plots     # work/<take>/speech/review/voice_timeline.png, embedding_space.png

voice_timeline.png  one row per voice (turns), the interjections / extra turns, the detected laughter, the Sortformer overlaps and the
                    span each phone covers: the first thing to look at to see whether the diarization is plausible;
embedding_space.png a 2-D PCA of the fused moments: left coloured by voice (two blobs expected), right by the phone that heard the
                    moment loudest (no blob may be a phone: that would mean the channel, not the speaker, is clustered).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import SPEECH, SOURCES
from .review import REVIEW


def voice_timeline(doc: dict, out: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = {v["voice"]: (v.get("name") or f"voice {v['voice']}") for v in doc["voices"]}
    colors = {"Larissa": "#d6336c", "Kadu": "#1c7ed6"}
    rows = sorted(names, key=lambda k: names[k])
    fig, ax = plt.subplots(figsize=(15, 4.6), constrained_layout=True)
    y = 0
    labels = []
    for vid in rows:
        for t in doc["turns"]:
            if t["voice"] == vid:
                ax.barh(y, t["end"] - t["start"], left=t["start"], height=0.7, color=colors.get(names[vid], "#868e96"), alpha=0.35 + 0.65 * t["confidence"])
        labels.append(names[vid])
        y += 1
    for it in doc["interjections"]:
        if it.get("gap_fill"):
            continue
        yy = rows.index(it["voice"]) if it["voice"] in rows else 0
        ax.barh(yy, it["end"] - it["start"], left=it["start"], height=0.35, color="black")
    ax.barh(y, 0, left=0)
    for o in doc["overlaps_sortformer"]:
        ax.barh(y, o["end"] - o["start"], left=o["start"], height=0.7, color="#f08c00")
    labels.append("overlap (Sortformer)")
    y += 1
    for e in doc["reactions"]:
        ax.barh(y, max(e["t1"] - e["t0"], 0.6), left=e["t0"], height=0.7, color="#2f9e44")
    labels.append("laughter / applause")
    y += 1
    for cam, c in doc["sources"]["clocks"].items():
        vj = json.loads((SPEECH / "raw" / cam / "vad.json").read_text(encoding="utf-8"))
        ax.barh(y, vj["duration"] * c["slope"], left=c["offset"], height=0.5, color="#adb5bd")
        labels.append(f"phone {cam} ({SOURCES[cam].split('-')[-1][:-4]})")
        y += 1
    for t in doc.get("emoji_tokens", []):
        ax.axvline(t["t"], color="#2f9e44", lw=1, ls="--")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(-10, 362)
    ax.set_xlabel("master clock (01-Recap-Tina video time), s")
    ax.set_title("Global voices: turns (opacity = confidence), black = interjection / extra turn, green dashes = joke tokens")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def embedding_space(res: dict, doc: dict, out: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mom, path = res["moments"], res["path"]
    x = mom.emb - mom.emb.mean(axis=0)
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    p = x @ vt[:2].T
    names = {v["voice"]: (v.get("name") or f"voice {v['voice']}") for v in doc["voices"]}
    colors = {"Larissa": "#d6336c", "Kadu": "#1c7ed6"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True)
    for vid in sorted(set(path.tolist())):
        m = path == vid
        axes[0].scatter(p[m, 0], p[m, 1], s=12, color=colors.get(names.get(vid), "#868e96"), label=names.get(vid, str(vid)), alpha=0.7)
    axes[0].set_title("fused moments by global voice")
    axes[0].legend()
    cams = sorted(set(mom.loud_cam))
    cmap = plt.get_cmap("tab10")
    for i, c in enumerate(cams):
        m = np.array([lc == c for lc in mom.loud_cam])
        axes[1].scatter(p[m, 0], p[m, 1], s=12, color=cmap(i), label=f"loudest phone {c}", alpha=0.7)
    axes[1].set_title("the same moments by loudest phone (must not form phone blobs)")
    axes[1].legend(fontsize=8)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def run() -> list[Path]:
    from . import cluster
    from .sources import load_clocks
    REVIEW.mkdir(parents=True, exist_ok=True)
    doc = json.loads((SPEECH / "diarization_global.json").read_text(encoding="utf-8"))
    cl = cluster.run(SPEECH / "raw", load_clocks())
    res = {"moments": cl["moments"], "path": cl["labels"]}
    return [voice_timeline(doc, REVIEW / "voice_timeline.png"), embedding_space(res, doc, REVIEW / "embedding_space.png")]
