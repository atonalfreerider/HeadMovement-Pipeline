"""ASR (faster-whisper large-v3, no prompt, English forced: the same call as the V1 gate) of whole folders of 48 kHz wavs, on the pod (agent V2).

    python -m dancecap.speech.pod.asr_v2v --root /root/data --dirs in/r1,out/dfn3__r1 [--embed]

For every folder `<root>/<d>` the wavs are resampled to 16 kHz (<root>/_16k/<d>/) and transcribed; writes `<root>/asr/<d with / -> __>.json`
{id: {"text", "avg_logprob"}} (ids = file stems). With --embed also an ECAPA embedding per file (<root>/asr/<name>.npz). A folder whose json exists is skipped
(--force to redo)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from . import stage_v2v as S


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--dirs", required=True)
    ap.add_argument("--embed", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--embed-only", action="store_true", help="only the ECAPA embeddings (<root>/asr/<name>.npz) of the folders, no transcription")
    ap.add_argument("--model", default="large-v3")
    a = ap.parse_args(argv)
    root = Path(a.root)
    (root / "asr").mkdir(parents=True, exist_ok=True)
    todo = []
    for d in a.dirs.split(","):
        name = d.replace("/", "__")
        if (root / "asr" / f"{name}.json").exists() and not a.force and not a.embed_only:
            continue
        files = sorted((root / d).glob("*.wav"))
        for f in files:
            x, sr = sf.read(str(f), dtype="float32")
            if x.ndim > 1:
                x = x.mean(axis=1)
            if sr != 16000:
                g = gcd(16000, sr)
                x = signal.resample_poly(x, 16000 // g, sr // g).astype(np.float32)
            o = root / "_16k" / d / f.name
            o.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(o), x, 16000, subtype="PCM_16")
        todo.append((name, d, files))
    if not todo:
        return 0
    from . import common as C
    items = [{"key": f"{n}::{f.stem}", "file": f"_16k/{d}/{f.name}"} for n, d, files in todo for f in files]
    t0 = time.time()
    res = {} if a.embed_only else S.transcribe_all(items, root, a.model, "cuda")
    for n, d, files in ([] if a.embed_only else todo):
        out = {f.stem: {"text": res[f"{n}::{f.stem}"]["text"], "avg_logprob": res[f"{n}::{f.stem}"]["avg_logprob"],
                        "words": [[w["word"], w["start"], w["end"]] for w in res[f"{n}::{f.stem}"]["words"]]} for f in files}
        (root / "asr" / f"{n}.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    if a.embed or a.embed_only:
        keys, e = S.embed_all(items, root, "cuda")
        for n, d, files in todo:
            sel = [i for i, k in enumerate(keys) if k.startswith(n + "::")]
            np.savez(root / "asr" / f"{n}.npz", keys=np.array([keys[i].split("::")[1] for i in sel]), ecapa=e[sel])
    C.log(f"asr_v2v: {len(items)} files in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
