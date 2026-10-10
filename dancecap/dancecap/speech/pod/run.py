"""Entry point of the pod-side speech stages (one farm job step per stage).

    python -m dancecap.speech.pod.run <stage> [--cams 00,01,...] [--force] [stage options]

stages:  check     versions, GPU, which engines import
         vad       Silero VAD regions + frame probabilities
         whisper   faster-whisper (--model large-v3 | large-v3-turbo, --passes auto,en,pt)
         parakeet  NeMo parakeet-tdt-0.6b-v3
         canary    NeMo canary-1b-v2 (third family; language from the whisper region ids)
         embed     ECAPA + TitaNet speaker embeddings, F0, level on a sliding window
         sortformer overlap-aware diarization windows
         events    AST laughter / applause / audience tagging + modulation heuristic
         quality   per-source speech-quality statistics
         segembed  embeddings of the Sortformer segments (needs sortformer + embed)
         regions   targeted multi-setting decoding of hard passages listed in regions.json (raw_regions/)

Every stage is idempotent per source (an existing output is skipped unless --force), so a restarted job resumes. Inputs and outputs:
see common.py (SPEECH_AUDIO, SPEECH_RAW). Exit code 0 when the stage finished, 3 when it ran but an engine was unavailable.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

from . import common as C


def stage_check(a, cams) -> int:
    info = {"python": sys.version.split()[0], "gpu": C.gpu_name(), "cams": sorted(cams), "audio_dir": str(C.audio_dir()),
            "raw_dir": str(C.raw_dir()), "versions": C.versions("numpy", "scipy", "soundfile", "torch", "torchaudio", "faster_whisper",
                                                                  "ctranslate2", "onnxruntime", "nemo", "speechbrain", "transformers",
                                                                  "sklearn", "librosa")}
    ok = {}
    for name, stmt in [("silero_vad", "from faster_whisper.vad import get_vad_model; get_vad_model()"),
                       ("nemo_asr", "import nemo.collections.asr as a"),
                       ("sortformer", "from nemo.collections.asr.models import SortformerEncLabelModel"),
                       ("speechbrain_ecapa", "from speechbrain.inference.speaker import EncoderClassifier"),
                       ("ast", "from transformers import ASTForAudioClassification, ASTFeatureExtractor")]:
        try:
            exec(stmt, {})
            ok[name] = "ok"
        except Exception as e:  # noqa: BLE001
            ok[name] = f"{type(e).__name__}: {str(e)[:200]}"
    info["imports"] = ok
    C.write_json(C.raw_dir() / "env_check.json", info)
    print(info)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.speech.pod.run", description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["check", "vad", "whisper", "parakeet", "canary", "embed", "sortformer", "events", "quality", "regions", "segembed"])
    ap.add_argument("--cams", help="comma list of source ids (default: all found)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--model", default="large-v3", help="whisper: large-v3 | large-v3-turbo")
    ap.add_argument("--passes", default="auto,en,pt", help="whisper passes")
    ap.add_argument("--no-lid", action="store_true", help="whisper: skip the per-region language id")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--regions", help="regions: path of regions.json (default: speech/regions.json)")
    ap.add_argument("--win", type=float)
    ap.add_argument("--hop", type=float)
    ap.add_argument("--step", type=float)
    ap.add_argument("--models", default="ecapa,titanet", help="embed: which embedding models")
    a = ap.parse_args(argv)
    cams = C.cams(C.parse_cams_arg(a.cams))
    if not cams:
        print(f"no audio found in {C.audio_dir()}", file=sys.stderr)
        return 2
    t0 = time.time()
    C.log(f"stage {a.stage} on {len(cams)} sources: {', '.join(cams)} (audio {C.audio_dir()}, raw {C.raw_dir()})")
    try:
        if a.stage == "check":
            return stage_check(a, cams)
        if a.stage == "vad":
            from . import stage_vad
            stage_vad.run(cams, a.force)
        elif a.stage == "whisper":
            from . import stage_whisper
            stage_whisper.run(cams, a.model, a.passes, a.force, a.device, lid=not a.no_lid)
        elif a.stage == "parakeet":
            from . import stage_parakeet
            stage_parakeet.run(cams, a.force, a.device)
        elif a.stage == "canary":
            from . import stage_canary
            stage_canary.run(cams, a.force, a.device)
        elif a.stage == "embed":
            from . import stage_embed
            stage_embed.run(cams, a.force, a.win or 1.5, a.hop or 0.75, a.device, a.models)
        elif a.stage == "sortformer":
            from . import stage_sortformer
            stage_sortformer.run(cams, a.force, a.win or 60.0, a.step or 50.0, a.device)
        elif a.stage == "events":
            from . import stage_events
            stage_events.run(cams, a.force, a.win or 3.0, a.hop or 1.0, a.device)
        elif a.stage == "quality":
            from . import stage_quality
            stage_quality.run(cams, a.force)
        elif a.stage == "segembed":
            from . import stage_embed
            stage_embed.run_segments(cams, a.force, a.device, a.models)
        elif a.stage == "regions":
            from . import stage_regions
            stage_regions.run(cams, a.regions, a.force, a.device)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        C.log(f"stage {a.stage} FAILED: {type(e).__name__}: {str(e)[:300]}")
        return 1
    C.log(f"stage {a.stage} done in {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
