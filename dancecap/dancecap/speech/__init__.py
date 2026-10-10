"""Class speech: ASR + speaker diarization + audio events of the recap class, run on the Runpod farm.

The recap take (larissa_kadu_recap) is a 6-minute zouk class taught by two teachers (Larissa and Kadu) and filmed by seven
phones. Every phone hears the same room, so the audio of all seven is one multi-channel recording on the master clock
(`reference_time = slope * source_time + offset`, dancecap.recap_sync / work/<take>/sync/sync.json). This package

  1. builds a domain glossary (English + pt-BR zouk terms) for ASR prompting and for the later transcript fusion (glossary.py),
  2. extracts the audio of every source (audio.py) and queues pod jobs on the hm-farm volume (jobs.py) that run, per source,
     two ASR families (faster-whisper large-v3 and large-v3-turbo, NVIDIA parakeet-tdt-0.6b-v3), Silero VAD, speaker
     embeddings + F0, an overlap detector (Sortformer, when it installs), an AudioSet laughter / applause detector and
     per-source speech-quality statistics (pod/*.py: standalone scripts that only need numpy, torch and the engine
     packages),
  3. clusters the speaker embeddings of ALL sources jointly on the master clock into global voices (cluster.py) and names
     them with independent evidence (identity.py): how the two teachers address each other over the whole class, a pitch
     prior, which phone is closest. A single self-introduction or remark about appearance is NEVER proof: the class has
     two jokes of exactly that kind (see identity.JOKE_NOTE),
  4. writes work/<take>/speech/diarization_global.json, events_master.json and the 30-clip review list (report.py,
     review.py),
  5. fuses the 24 transcripts (6 phones x 4 engines) into ONE consensus transcript by ROVER-style word voting, scores it against held-out
     transcripts and writes the places where the vote did not settle as batches for a language-model editor (fuse.py, fuse_post.py,
     fuse_batches.py, fuse_eval.py, fuse_levels.py, fuse_run.py; `python -m dancecap.speech fuse | fuse-eval`; docs/SPEECH.md).
  6. merges the editors' decisions on the disputed spans into the FINAL transcript and derives the exports (srt / vtt / txt / captions.json), chapters, the teaching
     outline for the film, the jokes list, a review page and an evaluation (final_merge.py, final_exports.py, final_direct.py, final_review.py, final_eval.py,
     final_run.py; `python -m dancecap.speech final`; docs/SPEECH.md).

CLI: `python -m dancecap.speech --help`. Everything heavy runs on a pod; locally only ffmpeg audio extraction, small numpy
work and orchestration.
"""
from __future__ import annotations

from pathlib import Path

from ..take import PACKAGE_ROOT, WORKSPACE

TAKE = "larissa_kadu_recap"
WORK = PACKAGE_ROOT / "work" / TAKE
SPEECH = WORK / "speech"
SYNC = WORK / "sync"
SCHEMA = "dancecap.speech/1"

# The recap sources by id; the file names are stable (the source folder lives outside the repo, its path is passed with
# --source or kept in work/<take>/speech/config.json).
SOURCES = {
    "00": "00-Recap-Ali.mp4",
    "01": "01-Recap-Tina.mp4",        # THE MASTER clock
    "02": "02-Recap-Alyssa.mp4",
    "03": "03-Recap-Redshirt.mp4",
    "04": "04-Recap-Tony.mp4",
    "05": "05-Recap-Liudmyla.mp4",
    "06": "06-Recap-John.mp4",
}
MASTER = "01"
SR = 16000

__all__ = ["TAKE", "WORK", "SPEECH", "SYNC", "SCHEMA", "SOURCES", "MASTER", "SR", "PACKAGE_ROOT", "WORKSPACE", "Path"]
