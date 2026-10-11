"""Local ASR for short clips (agent O1): faster-whisper large-v3 on the local GPU, the same call as the pod's `pod/asr_v2v.py` (English forced, no prompt, word times).

Run it with the interpreter that has faster-whisper (`.venv-asr\\Scripts\\python.exe`); it needs nothing else (decoding is faster-whisper's own, PyAV):

    .venv-asr\\Scripts\\python.exe dancecap\\speech\\asr_local.py items.json out.json [--model large-v3] [--clip-start 0 --clip-end 0]

`items.json` = [{"key": str, "file": path, "t0": optional seconds, "t1": optional seconds}] (a part of the file; word times in `out.json` are file seconds);
`out.json` = {key: {"text", "avg_logprob", "words": [[word, start, end], ...]}}. For a whole 6 min mix it takes about 1-2 minutes on a 2080 Ti.
"""
from __future__ import annotations

import json
import sys


def main(argv: list[str] | None = None) -> int:
    a = list(sys.argv[1:] if argv is None else argv)
    model_name = "large-v3"
    if "--model" in a:
        i = a.index("--model")
        model_name = a[i + 1]
        del a[i:i + 2]
    items = json.load(open(a[0], encoding="utf-8"))
    from faster_whisper import WhisperModel
    from faster_whisper.audio import decode_audio
    m = WhisperModel(model_name, device="cuda", compute_type="float16")
    out = {}
    for it in items:
        x = decode_audio(it["file"], sampling_rate=16000)
        off = float(it.get("t0") or 0.0)
        if it.get("t0") is not None or it.get("t1") is not None:
            x = x[int(off * 16000):int(float(it["t1"]) * 16000) if it.get("t1") is not None else None]
        segs, _ = m.transcribe(x, language="en", word_timestamps=True, condition_on_previous_text=False, beam_size=5, temperature=0)
        segs = list(segs)
        words = [[w.word.strip(), round(off + w.start, 3), round(off + w.end, 3)] for s in segs for w in (s.words or [])]
        lp = [s.avg_logprob for s in segs]
        out[it["key"]] = {"text": " ".join(s.text.strip() for s in segs).strip(), "avg_logprob": round(sum(lp) / len(lp), 4) if lp else None, "words": words}
    json.dump(out, open(a[1], "w", encoding="utf-8"), ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
