"""S4 end to end: the final transcript and everything derived from it (local, seconds; no pod).

    python -m dancecap.speech final            # transcript_final.json, .srt, .vtt, .txt, captions.json, chapters.json, teaching_events.json, jokes.json, review.html, evaluation.json
    python -m dancecap.speech final --no-clips # skip the ffmpeg audio clips of the review page

Inputs (all under work/<take>/speech/): consensus_v0.json, disagreements/batch_NN.json + batch_NN.decisions.json, final_overrides.json (this recording's own
corrections, curated chapters and teaching events), diarization / events files, review/review_list.json (S1's 30 clips), fusion_metrics.json, the raw per-phone transcripts.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from . import SPEECH, TAKE, fuse
from . import final_direct as FD, final_eval as FEV, final_exports as FX, final_merge as FM, final_review as FR
from .fuse_post import json_safe
from .sources import load_clocks

MASTER_DURATION_FALLBACK = 360.477


def _dump(path: Path, doc) -> Path:
    Path(path).write_text(json.dumps(json_safe(doc), indent=1, ensure_ascii=False, allow_nan=False), encoding="utf-8", newline="\n")
    return Path(path)


def master_duration() -> float:
    try:
        sync = json.loads((SPEECH.parent / "sync" / "sync.json").read_text(encoding="utf-8"))
        return float(sync["master"]["duration_s"])
    except (OSError, ValueError, KeyError):
        return MASTER_DURATION_FALLBACK


def run_final(clips: bool = True, log=print) -> dict:
    t0 = time.time()
    cfg = SPEECH / "config.json"
    recap = json.loads(cfg.read_text(encoding="utf-8")).get("source_dir") if cfg.exists() else None
    clocks = load_clocks(recap_dir=recap)
    consensus = json.loads((SPEECH / "consensus_v0.json").read_text(encoding="utf-8"))
    overrides = FM.load_overrides()
    final = FM.run_final_merge(log=log)
    chapters = FD.build_chapters(final, overrides, master_duration())
    _dump(SPEECH / "chapters.json", chapters)
    teaching = FD.build_teaching_events(final, overrides, chapters["chapters"])
    _dump(SPEECH / "teaching_events.json", teaching)
    peaks = FD.ast_peaks(SPEECH / "raw", clocks)
    jokes = FD.build_jokes(final, consensus, overrides, peaks)
    _dump(SPEECH / "jokes.json", jokes)
    captions = FX.write_all(final, SPEECH, jokes)
    log(f"captions.json: {captions['stats']}")
    hyps = fuse.load_hyps(SPEECH / "raw", clocks)
    metrics_p = SPEECH / "fusion_metrics.json"
    metrics = json.loads(metrics_p.read_text(encoding="utf-8")) if metrics_p.exists() else None
    evaluation = FEV.evaluate(final, consensus, hyps, metrics)
    _dump(SPEECH / "evaluation.json", evaluation)
    clip_files: dict[str, str] = {}
    for s in final["adjudication"]["spans"]:
        if not s.get("owner_resolved") and (s["needs_owner"] or "unclear" in s["flags"] or s["decision"] == "uncertain"):
            p = FR.cut_clip(s, clocks, make=clips)          # --no-clips: no ffmpeg, but clips that exist are linked
            if p is not None:
                clip_files[s["id"]] = str(Path("review") / "final_clips" / p.name).replace("\\", "/")
    owner_clips: dict[str, str] = {}
    for oc in overrides.get("owner_clips", []):             # final_overrides.json["owner_clips"]: [{id, start, end, listen_to?}] = the clips the owner's questions point at
        p = FR.cut_clip({"id": oc["id"], "start": float(oc["start"]), "end": float(oc["end"]), "listen_to": oc.get("listen_to")}, clocks, pad=float(oc.get("pad", 0.0)), make=clips)
        if p is not None:
            owner_clips[oc["id"]] = str(Path("review") / "final_clips" / p.name).replace("\\", "/")
    qp = SPEECH / "OWNER_QUESTIONS.md"
    questions = qp.read_text(encoding="utf-8") if qp.exists() else None
    rl = SPEECH / "review" / "review_list.json"
    who = json.loads(rl.read_text(encoding="utf-8")).get("clips", []) if rl.exists() else []
    page = FR.build_review(final, chapters, jokes, who, evaluation, clip_files, owner_questions=questions)
    (SPEECH / "review.html").write_text(page, encoding="utf-8", newline="\n")
    summary = {"transcript_final": str(FM.OUT_FINAL), "stats": final["stats"], "checks": final["checks"], "chapters": chapters["n_chapters"], "chapter_checks": chapters["checks"],
               "teaching_events": teaching["n_events"], "teaching_by_tag": teaching["by_tag"], "captions": captions["stats"], "laugh_candidates": len(jokes["laugh_candidates"]),
               "clips": len(clip_files), "owner_clips": owner_clips, "evaluation_mean": evaluation["mean"], "expected_accuracy": evaluation["expected_accuracy"]["statement"], "seconds": round(time.time() - t0, 1)}
    return summary
