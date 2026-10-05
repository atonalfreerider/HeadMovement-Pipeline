"""beat_this (zouk branch) wrapper + a tempo-grid fallback in the same format.

beat_this_analyzer.py writes `<stem>_zouk-time-analysis.json` next to the audio;
head-movement reads `zouk-time-analysis.json`. Entries are [seconds, type] with
type 1 = downbeat-ish (eighths 0,4), 2 = accent (eighths 3,6), 3 = other.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


def run_beat_this(audio: Path, beat_this_root: Path, python: str, out_json: Path) -> Path:
    work = out_json.parent / "_beat_this"
    work.mkdir(parents=True, exist_ok=True)
    local = work / audio.name
    shutil.copy2(audio, local)
    subprocess.run([python, str(beat_this_root / "beat_this_analyzer.py"), str(local)], cwd=beat_this_root, check=True)
    produced = local.with_name(f"{local.stem}_zouk-time-analysis.json")
    if not produced.exists():
        raise FileNotFoundError(f"beat_this did not produce {produced}")
    shutil.move(produced, out_json)
    return out_json


def zouk_grid(bpm: float, duration: float, first_downbeat: float = 0.0) -> list[list[float]]:
    """Same 8-per-measure pattern calculate_zouk_time() emits, for a steady tempo."""
    measure = 4 * 60.0 / bpm
    out, t = [], first_downbeat
    while t < duration:
        for j in range(8):
            bt = 1 if j in (0, 4) else (2 if j in (3, 6) else 3)
            out.append([round(t + measure * j / 8, 6), bt])
        t += measure
    return [b for b in out if b[0] < duration]


def write_beats(path: Path, beats: list[list[float]]) -> None:
    path.write_text(json.dumps(beats, indent=1))
