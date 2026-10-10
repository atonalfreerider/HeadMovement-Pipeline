"""Stage `events`: laughter, applause, cheering and other audience reactions, per source, with timestamps.

    python -m dancecap.speech.pod.run events [--cams 00,01] [--win 3.0] [--hop 1.0] [--force]

ASR engines drop laughter (or hallucinate "Thanks for watching"), but the owner wants the jokes marked. Two independent detectors:
  * AudioSet tagging with the Audio Spectrogram Transformer (MIT/ast-finetuned-audioset-10-10-0.4593, mAP 0.485, ungated) on
    `win`-second windows every `hop` s. AST takes 10.24 s clips; a short window is TILED (repeated) to that length instead of
    zero-padded, which keeps the padding from looking like silence. Stored: the probability of every laughter / applause / crowd /
    speech-like class and the top-5 labels of each window.
  * a signal heuristic: envelope modulation in the 3-8 Hz band (the "ha-ha-ha" burst rhythm of laughter, ~5 Hz), voiced fraction, level.
The PC combines them with the agreement of several phones (an event heard on 3 phones at the same master time is not handling
noise), dancecap.speech.events.

Output raw/<cam>/events.json: {stage, model, cam, win, hop, classes: {name: index}, labels_top: ..., windows: [{t0, t1, p: {class: prob},
  top: [[label, prob] x5], am_3_8: ..., rms_db, f0_voiced}]}
"""
from __future__ import annotations

import time

import numpy as np

from . import common as C

MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
# substrings of the AudioSet labels worth keeping (case-insensitive); the result maps label -> index
KEEP = ("laugh", "giggle", "snicker", "chuckle", "chortle", "applause", "clapping", "cheering", "crowd", "hubbub", "speech", "conversation",
        "narration", "shout", "yell", "scream", "whoop", "music", "singing", "whistl", "cough", "sneeze", "breath", "gasp", "silence",
        "finger snapping", "footsteps", "door", "wind", "noise", "static")


def keep_labels(id2label: dict) -> dict[str, int]:
    out = {}
    for i, name in id2label.items():
        low = str(name).lower()
        if any(k in low for k in KEEP):
            out[str(name)] = int(i)
    return out


def am_band_ratio(x: np.ndarray, lo: float = 3.0, hi: float = 8.0, frame: int = 320, hop: int = 160) -> float:
    """Share of the envelope's modulation power in [lo, hi] Hz (laughter bursts at ~5 Hz); 0 for silence."""
    env = C.frame_rms_db(x, frame, hop)
    env = np.power(10.0, env / 20.0)
    env = env - env.mean()
    if env.size < 32 or float(np.sum(env * env)) < 1e-12:
        return 0.0
    spec = np.abs(np.fft.rfft(env * np.hanning(env.size))) ** 2
    freqs = np.fft.rfftfreq(env.size, d=hop / C.SR)
    tot = float(spec[(freqs >= 0.5) & (freqs <= 20)].sum()) + 1e-12
    return float(spec[(freqs >= lo) & (freqs <= hi)].sum() / tot)


def window_starts(duration: float, win: float, hop: float) -> list[float]:
    n = max(0, int((duration - win) / hop) + 1)
    return [round(k * hop, 3) for k in range(n)]


def run(cams: dict, force: bool = False, win: float = 3.0, hop: float = 1.0, device: str = "cuda", batch: int = 32) -> dict:
    todo = {c: w for c, w in cams.items() if force or not (C.cam_dir(c) / "events.json").exists()}
    if not todo:
        C.log("events: nothing to do")
        return {}
    import torch
    from transformers import ASTFeatureExtractor, ASTForAudioClassification
    t0 = time.time()
    fe = ASTFeatureExtractor.from_pretrained(MODEL)
    model = ASTForAudioClassification.from_pretrained(MODEL).to(device).eval()
    if device == "cuda":
        model = model.half()
    classes = keep_labels(model.config.id2label)
    C.log(f"AST loaded in {time.time() - t0:.1f} s, {len(classes)} of {len(model.config.id2label)} labels kept")
    target = int(10.24 * C.SR)
    done = {}
    for i, (cam, wav) in enumerate(todo.items()):
        t1 = time.time()
        audio = C.load_audio(wav)
        dur = len(audio) / C.SR
        starts = window_starts(dur, win, hop)
        rows = []
        n = int(win * C.SR)
        for k in range(0, len(starts), batch):
            chunk = starts[k:k + batch]
            clips = []
            for s in chunk:
                x = audio[int(s * C.SR):int(s * C.SR) + n]
                reps = int(np.ceil(target / max(len(x), 1)))
                clips.append(np.tile(x, reps)[:target])
            feats = fe(clips, sampling_rate=C.SR, return_tensors="pt")["input_values"].to(device)
            if device == "cuda":
                feats = feats.half()
            with torch.no_grad():
                p = torch.sigmoid(model(feats).logits.float()).cpu().numpy()
            for j, s in enumerate(chunk):
                x = audio[int(s * C.SR):int(s * C.SR) + n]
                top = np.argsort(-p[j])[:5]
                f0, _ = C.f0_track(x)
                rows.append({"t0": s, "t1": round(s + win, 3),
                             "p": {name: round(float(p[j, idx]), 4) for name, idx in classes.items() if p[j, idx] >= 0.02},
                             "top": [[model.config.id2label[int(t)], round(float(p[j, t]), 3)] for t in top],
                             "am_3_8": round(am_band_ratio(x), 4), "rms_db": round(C.rms_db(x), 2),
                             "f0_voiced": round(float(np.mean(f0 > 0)) if f0.size else 0.0, 3)})
        C.write_json(C.cam_dir(cam) / "events.json",
                     {"stage": "events", "version": C.STAGE_VERSION, "model": MODEL, "versions": C.versions("transformers", "torch"),
                      "cam": cam, "duration": round(dur, 3), "win": win, "hop": hop, "classes": classes, "windows": rows,
                      "seconds": round(time.time() - t1, 2)})
        C.log(f"events {cam}: {len(rows)} windows ({time.time() - t1:.1f} s)")
        C.progress_tick("speech/events", i + 1, len(todo))
        done[cam] = len(rows)
    return done
