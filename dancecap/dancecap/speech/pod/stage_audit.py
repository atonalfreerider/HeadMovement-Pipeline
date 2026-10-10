"""Pod-side steps of the S5 audit of the final class transcript (independent engines the vote and the editors never saw).

    python -m dancecap.speech.pod.stage_audit <step> --inputs audit_inputs.json --out <dir> [--cams 00,01,...]

steps (each writes <out>/<name>.json (+ .npz) and skips work whose output exists unless --force):

    wv2_full     faster-whisper large-v2, NO prompt, forced English, its own Silero VAD, word timestamps, on the full 16 kHz audio of every phone
    wv2_clips    the same model on the audit clips (the sampled utterances +-1 s, best and second phone), no VAD
    pk11_full    NVIDIA parakeet-tdt-1.1b (a different, larger model than the 0.6b-v3 of the pipeline) on the full audio of every phone
    pk11_clips   the same on the clips
    w2v          torchaudio wav2vec2 large lv60k 960h CTC log-probabilities for the windows of inputs["w2v"] (forced alignment is done on the PC)
    score        teacher-forced log-probability of the candidate texts of inputs["scores"] under whisper large-v2 and large-v3 (transformers)
    speaker_open WavLM x-vectors and F0 of 1 s windows over the opening 10 master seconds on phones 00, 01, 02
    speaker      WavLM x-vectors (microsoft/wavlm-base-plus-sv) and pyin F0 of every utterance of inputs["speaker"] on its best phone and the next one
    laughter     CLAP zero-shot scores (laughing / speech / silence / applause) in 1 s windows over the first 12 master seconds of the nearest phones

The audio is read from the volume's 16 kHz wavs (common.audio_dir()). All weights go to HF_HOME (set it to the pod's local disk).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from . import common as C

SR = C.SR


def allow_trusted_bin() -> None:
    """transformers 4.53 refuses torch.load of .bin checkpoints below torch 2.6; the two checkpoints that need it (Microsoft's WavLM x-vector, LAION's CLAP) are
    the publishers' official files and the pod is a throw-away machine, so the check is switched off for them."""
    import transformers.modeling_utils as mu
    import transformers.utils.import_utils as iu
    mu.check_torch_load_is_safe = lambda: None
    iu.check_torch_load_is_safe = lambda: None


def read_inputs(p: str) -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def cut(audio: np.ndarray, t0: float, t1: float) -> np.ndarray:
    return audio[max(0, int(t0 * SR)):int(t1 * SR)]


def audio_cache():
    cache: dict[str, np.ndarray] = {}
    paths = C.cams()

    def get(cam: str) -> np.ndarray:
        if cam not in cache:
            cache[cam] = C.load_audio(paths[cam])
        return cache[cam]
    return get


def word_rows(words) -> list[dict]:
    return [{"word": w.word.strip(), "start": round(float(w.start), 3), "end": round(float(w.end), 3), "prob": round(float(w.probability), 4)} for w in words]


# ---------------------------------------------------------------------------------------------- whisper large-v2


def load_wv2(device: str = "cuda"):
    from faster_whisper import WhisperModel
    from faster_whisper.utils import download_model
    path = download_model("large-v2")
    return WhisperModel(path, device=device, compute_type="float16" if device == "cuda" else "int8")


def step_wv2_full(inp, out: Path, cams: list[str], force: bool, device: str) -> None:
    model = None
    for cam in cams:
        dst = out / f"wv2_full_{cam}.json"
        if dst.exists() and not force:
            continue
        model = model or load_wv2(device)
        audio = C.load_audio(C.cams()[cam])
        t0 = time.time()
        segs, info = model.transcribe(audio, language="en", beam_size=5, word_timestamps=True, vad_filter=True,
                                      vad_parameters={"min_silence_duration_ms": 300, "speech_pad_ms": 200},
                                      condition_on_previous_text=False, temperature=[0.0, 0.2, 0.4])
        rows, seg_rows = [], []
        for s in segs:
            seg_rows.append({"start": round(float(s.start), 3), "end": round(float(s.end), 3), "text": s.text.strip(),
                             "avg_logprob": round(float(s.avg_logprob), 4), "no_speech_prob": round(float(s.no_speech_prob), 4)})
            for w in word_rows(s.words or []):
                w["seg"] = len(seg_rows) - 1
                rows.append(w)
        C.write_json(dst, {"model": "faster-whisper large-v2 (no prompt, en, vad_filter)", "cam": cam, "seconds": round(time.time() - t0, 1),
                           "language_probability": getattr(info, "language_probability", None), "segments": seg_rows, "words": rows})
        C.log(f"wv2_full {cam}: {len(rows)} words in {time.time() - t0:.1f} s")


def step_wv2_clips(inp, out: Path, force: bool, device: str) -> None:
    dst = out / "wv2_clips.json"
    if dst.exists() and not force:
        return
    model = load_wv2(device)
    get = audio_cache()
    res = []
    for c in inp["clips"]:
        x = cut(get(c["cam"]), c["t0"], c["t1"])
        segs, _ = model.transcribe(x, language="en", beam_size=5, word_timestamps=True, vad_filter=False, condition_on_previous_text=False,
                                   temperature=[0.0, 0.2, 0.4])
        rows = []
        for s in segs:
            rows += word_rows(s.words or [])
        res.append({**{k: c[k] for k in ("id", "utt", "cam", "rank", "t0", "t1")}, "text": " ".join(w["word"] for w in rows), "words": rows})
    C.write_json(dst, {"model": "faster-whisper large-v2 (no prompt, en)", "clips": res})
    C.log(f"wv2_clips: {len(res)} clips")


# ---------------------------------------------------------------------------------------------- parakeet 1.1b


def load_pk11(device: str = "cuda"):
    import nemo.collections.asr as nemo_asr
    m = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-1.1b")
    m.eval()
    return m.to(device)


def step_pk11_full(inp, out: Path, cams: list[str], force: bool, device: str) -> None:
    from . import stage_parakeet as P
    model = None
    for cam in cams:
        dst = out / f"pk11_full_{cam}.json"
        if dst.exists() and not force:
            continue
        model = model or load_pk11(device)
        audio = C.load_audio(C.cams()[cam])
        t0 = time.time()
        regions = [(s / SR, e / SR) for s, e in C.vad_regions(audio)]
        segs = C.split_for_asr(regions, 10.0)          # the 1.1b model drops words in long (20-28 s) segments: 10 s
        words, seg_rows = P.transcribe_segments(model, audio, segs)
        C.write_json(dst, {"model": "nvidia/parakeet-tdt-1.1b", "cam": cam, "seconds": round(time.time() - t0, 1), "segments": seg_rows, "words": words})
        C.log(f"pk11_full {cam}: {len(words)} words in {time.time() - t0:.1f} s")


def step_pk11_clips(inp, out: Path, force: bool, device: str) -> None:
    from . import stage_parakeet as P
    dst = out / "pk11_clips.json"
    if dst.exists() and not force:
        return
    model = load_pk11(device)
    get = audio_cache()
    res = []
    for c in inp["clips"]:
        x = cut(get(c["cam"]), c["t0"], c["t1"])
        words, _ = P.transcribe_segments(model, x, [(0.0, len(x) / SR)])
        res.append({**{k: c[k] for k in ("id", "utt", "cam", "rank", "t0", "t1")}, "text": " ".join(w["word"] for w in words), "words": words})
    C.write_json(dst, {"model": "nvidia/parakeet-tdt-1.1b", "clips": res})
    C.log(f"pk11_clips: {len(res)} clips")


# ---------------------------------------------------------------------------------------------- wav2vec2 CTC emissions


def step_w2v(inp, out: Path, force: bool, device: str) -> None:
    dst = out / "w2v.npz"
    if dst.exists() and not force:
        return
    import torch
    import torchaudio
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_LARGE_LV60K_960H       # character CTC; (the HF checkpoint needs torch >= 2.6 to load: .bin only)
    model = bundle.get_model().to(device).eval()
    get = audio_cache()
    arrays, meta = {}, []
    for w in inp["w2v"]:
        x = cut(get(w["cam"]), w["t0"], w["t1"]).astype(np.float32)
        x = (x - x.mean()) / (x.std() + 1e-7)
        with torch.inference_mode():
            em, _ = model(torch.from_numpy(x)[None].to(device))
            lp = torch.log_softmax(em.float(), dim=-1)[0].cpu().numpy()
        arrays[w["id"]] = lp.astype(np.float32)
        meta.append({"id": w["id"], "cam": w["cam"], "t0": w["t0"], "t1": w["t1"], "frames": int(lp.shape[0]), "samples": int(len(x))})
    np.savez(dst, **arrays)
    C.write_json(out / "w2v_meta.json", {"model": "torchaudio WAV2VEC2_ASR_LARGE_LV60K_960H", "labels": list(bundle.get_labels()), "frame_s": 0.02, "items": meta})
    C.log(f"w2v: {len(arrays)} windows")


# ---------------------------------------------------------------------------------------------- teacher-forced scores


def step_score(inp, out: Path, force: bool, device: str) -> None:
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    dst = out / "score.json"
    if dst.exists() and not force:
        return
    get = audio_cache()
    res = []
    for mname in ("openai/whisper-large-v2", "openai/whisper-large-v3"):
        proc = WhisperProcessor.from_pretrained(mname)
        model = WhisperForConditionalGeneration.from_pretrained(mname, torch_dtype=torch.float16).to(device).eval()
        tok = proc.tokenizer
        tok.set_prefix_tokens(language="en", task="transcribe", predict_timestamps=False)
        n_prefix = len(tok.prefix_tokens)
        for ent in inp["scores"]:
            for cam, (t0, t1) in ent["windows"].items():
                x = cut(get(cam), t0, t1)
                feats = proc.feature_extractor(x, sampling_rate=SR, return_tensors="pt").input_features.to(device, torch.float16)
                with torch.no_grad():
                    enc = model.model.encoder(feats).last_hidden_state
                    for key, text in ent["candidates"].items():
                        ids = tok(" " + text).input_ids
                        dec_in = torch.tensor([ids[:-1]], device=device)
                        logits = model(encoder_outputs=(enc,), decoder_input_ids=dec_in).logits.float()
                        lp = torch.log_softmax(logits, dim=-1)[0]
                        tgt = torch.tensor(ids[1:], device=device)
                        tok_lp = lp.gather(1, tgt[:, None])[:, 0].cpu().numpy()
                        body = tok_lp[n_prefix - 1:]            # text tokens + end-of-text; the forced prefix tokens are skipped
                        res.append({"model": mname, "entry": ent["id"], "cam": cam, "cand": key, "logprob": round(float(body.sum()), 3),
                                    "n_tokens": int(len(body)), "mean": round(float(body.mean()), 4)})
        del model
        torch.cuda.empty_cache()
        C.log(f"score {mname}: done")
    C.write_json(dst, {"what": "teacher-forced log-probability of each candidate text given the window audio", "rows": res})


# ---------------------------------------------------------------------------------------------- speakers


def step_speaker(inp, out: Path, force: bool, device: str) -> None:
    dst = out / "speaker.npz"
    if dst.exists() and not force:
        return
    import librosa
    import torch
    from transformers import AutoFeatureExtractor, WavLMForXVector
    allow_trusted_bin()
    name = "microsoft/wavlm-base-plus-sv"
    fe = AutoFeatureExtractor.from_pretrained(name)
    model = WavLMForXVector.from_pretrained(name).to(device).eval()
    get = audio_cache()
    arrays, meta = {}, []
    for it in inp["speaker"]:
        for rank in ("best", "second"):
            cam = it.get(rank)
            if not cam:
                continue
            t0, t1 = it["win"][rank]
            x = cut(get(cam), t0, t1)
            dur = len(x) / SR
            pad = max(0.0, 1.0 - dur) / 2          # a very short utterance is padded with its own surroundings to 1 s for the embedding (flagged)
            xe = cut(get(cam), t0 - pad, t1 + pad) if pad > 0 else x
            f = fe(xe, sampling_rate=SR, return_tensors="pt", padding=True)
            with torch.no_grad():
                emb = model(input_values=f.input_values.to(device)).embeddings[0].float().cpu().numpy()
            emb = emb / (np.linalg.norm(emb) + 1e-9)
            f0, vf, _ = librosa.pyin(x, fmin=65, fmax=450, sr=SR, frame_length=1024, hop_length=160) if len(x) > 1100 else (np.array([]), np.array([]), None)
            voiced = f0[~np.isnan(f0)] if f0.size else np.array([])
            key = f"{it['id']}_{rank}"
            arrays[key] = emb.astype(np.float32)
            meta.append({"key": key, "id": it["id"], "rank": rank, "cam": cam, "seconds": round(dur, 3), "padded": pad > 0,
                         "f0_median": round(float(np.median(voiced)), 1) if voiced.size else None,
                         "f0_p10": round(float(np.percentile(voiced, 10)), 1) if voiced.size else None,
                         "f0_p90": round(float(np.percentile(voiced, 90)), 1) if voiced.size else None, "voiced_frames": int(voiced.size),
                         "voiced_frac": round(float(voiced.size / max(1, f0.size)), 3) if f0.size else None})
    np.savez(dst, **arrays)
    C.write_json(out / "speaker_meta.json", {"model": name, "items": meta})
    C.log(f"speaker: {len(meta)} embeddings")


# ---------------------------------------------------------------------------------------------- laughter (CLAP zero-shot)


def step_laughter(inp, out: Path, force: bool, device: str) -> None:
    dst = out / "laughter.json"
    if dst.exists() and not force:
        return
    import librosa
    import torch
    from transformers import ClapModel, ClapProcessor
    allow_trusted_bin()
    name = "laion/clap-htsat-unfused"
    proc = ClapProcessor.from_pretrained(name)
    model = ClapModel.from_pretrained(name).to(device).eval()
    labels = ["a person laughing", "people laughing and giggling", "a person talking", "silence", "applause", "background noise of a room"]
    get = audio_cache()
    res = {}
    for cam, off in inp["laughter_cams"].items():
        x = get(cam)
        rows = []
        t = 0.0
        while t < 12.0:
            m0, m1 = t, t + 1.0
            p0, p1 = (m0 - off) / 1.0, (m1 - off) / 1.0
            if p0 < 0:
                t += 0.25
                continue
            seg = librosa.resample(cut(x, p0, p1), orig_sr=SR, target_sr=48000)     # CLAP wants 48 kHz (the volume holds 16 kHz audio: band-limited to 8 kHz)
            feats = proc(text=labels, audios=[seg], sampling_rate=48000, return_tensors="pt", padding=True)
            feats = {k: v.to(device) for k, v in feats.items()}
            with torch.no_grad():
                o = model(**feats)
            pr = torch.softmax(o.logits_per_audio[0].float(), dim=-1).cpu().numpy()
            rows.append({"m0": round(m0, 2), "m1": round(m1, 2), **{l: round(float(p), 3) for l, p in zip(labels, pr)}})
            t += 0.25
        res[cam] = rows
    C.write_json(dst, {"model": name, "labels": labels, "cams": res})
    C.log("laughter: done")


def step_speaker_open(inp, out: Path, force: bool, device: str) -> None:
    """WavLM x-vector + pyin F0 of 1.0 s windows (hop 0.25 s) over the first 10 master seconds on the phones that heard the opening."""
    dst = out / "speaker_open.npz"
    if dst.exists() and not force:
        return
    import librosa
    import torch
    from transformers import AutoFeatureExtractor, WavLMForXVector
    allow_trusted_bin()
    name = "microsoft/wavlm-base-plus-sv"
    fe = AutoFeatureExtractor.from_pretrained(name)
    model = WavLMForXVector.from_pretrained(name).to(device).eval()
    get = audio_cache()
    arrays, meta = {}, []
    for cam, off in inp["laughter_cams"].items():
        x = get(cam)
        m = -0.5
        while m + 1.0 <= 10.0:
            p0, p1 = m - off, m + 1.0 - off
            if p0 >= 0:
                seg = cut(x, p0, p1)
                f = fe(seg, sampling_rate=SR, return_tensors="pt", padding=True)
                with torch.no_grad():
                    emb = model(input_values=f.input_values.to(device)).embeddings[0].float().cpu().numpy()
                emb = emb / (np.linalg.norm(emb) + 1e-9)
                f0, _, _ = librosa.pyin(seg, fmin=65, fmax=450, sr=SR, frame_length=1024, hop_length=160)
                v = f0[~np.isnan(f0)]
                key = f"{cam}_{m:.2f}"
                arrays[key] = emb.astype(np.float32)
                meta.append({"key": key, "cam": cam, "m0": round(m, 2), "m1": round(m + 1.0, 2), "f0_median": round(float(np.median(v)), 1) if v.size >= 8 else None,
                             "voiced_frames": int(v.size), "rms_db": round(C.rms_db(seg), 1)})
            m += 0.25
    np.savez(dst, **arrays)
    C.write_json(out / "speaker_open_meta.json", {"model": name, "items": meta})
    C.log(f"speaker_open: {len(meta)} windows")


def step_laughter_all(inp, out: Path, force: bool, device: str) -> None:
    """CLAP zero-shot scores in 1 s windows (hop 0.5 s) over the WHOLE class on phones 01, 02 and 05 (candidates for other laughs / audience reactions)."""
    dst = out / "laughter_all.json"
    if dst.exists() and not force:
        return
    import librosa
    import torch
    from transformers import ClapModel, ClapProcessor
    allow_trusted_bin()
    name = "laion/clap-htsat-unfused"
    proc = ClapProcessor.from_pretrained(name)
    model = ClapModel.from_pretrained(name).to(device).eval()
    labels = ["a person laughing", "people laughing and giggling", "a person talking", "silence", "applause", "background noise of a room"]
    get = audio_cache()
    res = {}
    for cam, off in inp["all_cams_offsets"].items():
        x = get(cam)
        rows = []
        m = max(0.0, off)
        dur = len(x) / SR
        while (m - off) + 1.0 <= dur and m + 1.0 <= 361.0:
            seg = librosa.resample(cut(x, m - off, m - off + 1.0), orig_sr=SR, target_sr=48000)
            feats = proc(text=labels, audios=[seg], sampling_rate=48000, return_tensors="pt", padding=True)
            feats = {k: v.to(device) for k, v in feats.items()}
            with torch.no_grad():
                pr = torch.softmax(model(**feats).logits_per_audio[0].float(), dim=-1).cpu().numpy()
            rows.append([round(m, 2)] + [round(float(p), 3) for p in pr])
            m += 0.5
        res[cam] = rows
    C.write_json(dst, {"model": name, "labels": labels, "columns": ["m0"] + labels, "cams": res})
    C.log("laughter_all: done")


STEPS = ["wv2_full", "wv2_clips", "pk11_full", "pk11_clips", "w2v", "score", "speaker", "speaker_open", "laughter", "laughter_all"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.speech.pod.stage_audit")
    ap.add_argument("step", choices=STEPS + ["all"])
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cams", default="00,01,02,03,05,06")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    inp = read_inputs(a.inputs)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cams = [c for c in a.cams.split(",") if c]
    todo = STEPS if a.step == "all" else [a.step]
    rc = 0
    for st in todo:
        t0 = time.time()
        try:
            if st == "wv2_full":
                step_wv2_full(inp, out, cams, a.force, a.device)
            elif st == "wv2_clips":
                step_wv2_clips(inp, out, a.force, a.device)
            elif st == "pk11_full":
                step_pk11_full(inp, out, cams, a.force, a.device)
            elif st == "pk11_clips":
                step_pk11_clips(inp, out, a.force, a.device)
            elif st == "w2v":
                step_w2v(inp, out, a.force, a.device)
            elif st == "score":
                step_score(inp, out, a.force, a.device)
            elif st == "speaker":
                step_speaker(inp, out, a.force, a.device)
            elif st == "speaker_open":
                step_speaker_open(inp, out, a.force, a.device)
            elif st == "laughter":
                step_laughter(inp, out, a.force, a.device)
            elif st == "laughter_all":
                step_laughter_all(inp, out, a.force, a.device)
            C.log(f"audit step {st} done in {time.time() - t0:.1f} s")
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            C.log(f"audit step {st} FAILED: {type(e).__name__}: {str(e)[:300]}")
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
