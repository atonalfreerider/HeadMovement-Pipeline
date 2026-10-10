"""Stage `embed`: speaker embeddings + pitch + level on a sliding window over the speech of every source.

    python -m dancecap.speech.pod.run embed [--cams 00,01] [--win 1.5] [--hop 0.75] [--force]

Windows of `win` s every `hop` s are placed where the Silero VAD (stage vad) says at least 60 % of the window is speech. For each
window: a speechbrain ECAPA-TDNN embedding (speechbrain/spkrec-ecapa-voxceleb, 192-d, ungated) and a NeMo TitaNet-Large embedding
(nvidia/speakerverification_en_titanet_large, 192-d, ungated); either may be missing if its package does not install (recorded in
embed.json). Also the window's median F0 (common.f0_track), its voiced fraction and its RMS level: the inputs of the pitch prior and
of the "which phone is closest" evidence. Embeddings are what the PC clusters JOINTLY across the seven phones
(dancecap.speech.cluster); nothing is clustered on the pod.

Output raw/<cam>/embed.npz   t0, t1, vadfrac, rms_db, f0_med, f0_voiced, [ecapa], [titanet]   (N rows)
       raw/<cam>/embed.json  {models, versions, n, win, hop, seconds, ...}
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from . import common as C
from . import stage_vad


def make_windows(duration: float, probs: np.ndarray | None, step: float, win: float = 1.5, hop: float = 0.75,
                 min_speech: float = 0.6, regions: list[dict] | None = None) -> list[tuple[float, float, float]]:
    """[(t0, t1, speech fraction)] windows of `win` s every `hop` s with at least `min_speech` of speech (frame probabilities >= 0.5;
    without probabilities the VAD regions decide)."""
    out = []
    n = int((duration - win) / hop) + 1
    for k in range(max(n, 0)):
        t0 = k * hop
        t1 = t0 + win
        if probs is not None and len(probs):
            a, b = int(t0 / step), max(int(t1 / step), int(t0 / step) + 1)
            fr = float(np.mean(probs[a:b] >= 0.5)) if b > a else 0.0
        else:
            cover = 0.0
            for r in regions or []:
                cover += max(0.0, min(t1, r["end"]) - max(t0, r["start"]))
            fr = cover / win
        if fr >= min_speech:
            out.append((round(t0, 3), round(t1, 3), round(fr, 3)))
    return out


def window_stats(audio: np.ndarray, wins: list[tuple[float, float, float]]) -> dict:
    f0, _ = C.f0_track(audio)
    f0_med = np.zeros(len(wins), np.float32)
    f0_vf = np.zeros(len(wins), np.float32)
    rms = np.zeros(len(wins), np.float32)
    for i, (t0, t1, _) in enumerate(wins):
        a, b = int(t0 * 100), int(t1 * 100)
        seg = f0[a:b]
        v = seg[seg > 0]
        f0_vf[i] = len(v) / max(len(seg), 1)
        f0_med[i] = float(np.median(v)) if len(v) >= 5 else 0.0
        rms[i] = C.rms_db(audio[int(t0 * C.SR):int(t1 * C.SR)])
    return {"f0_med": f0_med, "f0_voiced": f0_vf, "rms_db": rms}


def _batches(audio: np.ndarray, wins, size: int):
    n = int(round((wins[0][1] - wins[0][0]) * C.SR)) if wins else 0
    for k in range(0, len(wins), size):
        chunk = wins[k:k + size]
        batch = np.zeros((len(chunk), n), np.float32)
        for j, (t0, t1, _) in enumerate(chunk):
            seg = audio[int(round(t0 * C.SR)):int(round(t0 * C.SR)) + n]
            batch[j, :len(seg)] = seg
        yield batch


def load_ecapa(device: str):
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError:
        from speechbrain.pretrained import EncoderClassifier   # speechbrain < 1.0
    root = Path(os.environ.get("HM_ROOT", "/workspace/hm")) / "weights" / "speech" / "ecapa"
    clf = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(root), run_opts={"device": device})
    return clf


def ecapa_embed(clf, audio: np.ndarray, wins, device: str, size: int = 96) -> np.ndarray:
    import torch
    out = []
    with torch.no_grad():
        for batch in _batches(audio, wins, size):
            e = clf.encode_batch(torch.from_numpy(batch).to(device)).squeeze(1)
            out.append(e.float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 192), np.float32)


def load_titanet(device: str):
    import nemo.collections.asr as nemo_asr
    m = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large")
    return m.eval().to(device)


def titanet_embed(m, audio: np.ndarray, wins, device: str, size: int = 96) -> np.ndarray:
    import torch
    out = []
    with torch.no_grad():
        for batch in _batches(audio, wins, size):
            x = torch.from_numpy(batch).to(device)
            ln = torch.full((x.shape[0],), x.shape[1], dtype=torch.long, device=device)
            _, emb = m.forward(input_signal=x, input_signal_length=ln)
            out.append(emb.float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 192), np.float32)


def run(cams: dict, force: bool = False, win: float = 1.5, hop: float = 0.75, device: str = "cuda", models: str = "ecapa,titanet") -> dict:
    todo = {c: w for c, w in cams.items() if force or not (C.cam_dir(c) / "embed.npz").exists()}
    if not todo:
        C.log("embed: nothing to do")
        return {}
    want = {m.strip() for m in models.split(",")}
    ecapa = titanet = None
    status: dict = {}
    if "ecapa" in want:
        try:
            ecapa = load_ecapa(device)
            status["ecapa"] = "ok"
        except Exception as e:  # noqa: BLE001
            status["ecapa"] = f"unavailable: {type(e).__name__}: {str(e)[:160]}"
            C.log(f"embed: ECAPA {status['ecapa']}")
    if "titanet" in want:
        try:
            titanet = load_titanet(device)
            status["titanet"] = "ok"
        except Exception as e:  # noqa: BLE001
            status["titanet"] = f"unavailable: {type(e).__name__}: {str(e)[:160]}"
            C.log(f"embed: TitaNet {status['titanet']}")
    if ecapa is None and titanet is None:
        raise RuntimeError(f"no embedding model loaded: {status}")
    done = {}
    for i, (cam, wav) in enumerate(todo.items()):
        t0 = time.time()
        audio = C.load_audio(wav)
        vad = stage_vad.load(cam)
        if vad is not None:
            wins = make_windows(len(audio) / C.SR, np.asarray(vad["probs"], np.float32), vad["frame_step"], win, hop)
        else:
            wins = make_windows(len(audio) / C.SR, None, 0.032, win, hop, regions=[{"start": s / C.SR, "end": e / C.SR} for s, e in C.vad_regions(audio)])
        arrays = {"t0": np.array([w[0] for w in wins], np.float32), "t1": np.array([w[1] for w in wins], np.float32),
                  "vadfrac": np.array([w[2] for w in wins], np.float32)}
        arrays.update(window_stats(audio, wins))
        if ecapa is not None:
            arrays["ecapa"] = ecapa_embed(ecapa, audio, wins, device)
        if titanet is not None:
            arrays["titanet"] = titanet_embed(titanet, audio, wins, device)
        out_npz = C.cam_dir(cam) / "embed.npz"
        tmp = out_npz.with_name("embed.tmp.npz")
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, out_npz)
        C.write_json(C.cam_dir(cam) / "embed.json", {"stage": "embed", "version": C.STAGE_VERSION, "cam": cam, "n": len(wins), "win": win,
                                                      "hop": hop, "min_speech_frac": 0.6, "models": status,
                                                      "versions": C.versions("speechbrain", "nemo", "torch"),
                                                      "seconds": round(time.time() - t0, 2)})
        C.log(f"embed {cam}: {len(wins)} windows ({time.time() - t0:.1f} s)")
        C.progress_tick("speech/embed", i + 1, len(todo))
        done[cam] = len(wins)
    return done


# ------------------------------------------------------------------------------------------------ Sortformer segments


def split_long(t0: float, t1: float, max_len: float = 3.0) -> list[tuple[float, float]]:
    """A long segment -> equal pieces of at most max_len seconds (a 38-s stretch of one local speaker is embedded piece by piece, so a change
    of voice that Sortformer missed still shows)."""
    n = max(1, int(np.ceil((t1 - t0) / max_len - 1e-9)))
    step = (t1 - t0) / n
    return [(t0 + i * step, t0 + (i + 1) * step) for i in range(n)]


def core_segments(doc: dict, min_len: float = 0.3, max_len: float = 3.0) -> list[dict]:
    """Sortformer segments of a source without the duplicates of the 10-s window overlaps: a segment is kept from the window whose core
    (the window minus half the overlap at each inner edge) contains the segment's middle. {t0, t1, spk, win}; spk is local to the window."""
    wins = doc["windows"]
    overlap = 0.0
    if len(wins) > 1:
        overlap = max(0.0, wins[0]["t1"] - wins[1]["t0"])
    out = []
    for k, w in enumerate(wins):
        lo = w["t0"] + (overlap / 2 if k > 0 else 0.0)
        hi = w["t1"] - (overlap / 2 if k < len(wins) - 1 else 0.0)
        for s in w["segments"]:
            mid = 0.5 * (s["start"] + s["end"])
            if lo <= mid < hi and s["end"] - s["start"] >= min_len:
                for a, b in split_long(s["start"], s["end"], max_len):
                    out.append({"t0": round(a, 3), "t1": round(b, 3), "spk": s["spk"], "win": k, "seg_t0": s["start"], "seg_t1": s["end"]})
    out.sort(key=lambda r: (r["t0"], r["spk"]))
    return out


def widen(t0: float, t1: float, dur: float, min_len: float = 0.8) -> tuple[float, float]:
    """Segment -> embedding window: at least `min_len` s (ECAPA is unreliable below), centred on the segment, inside the recording."""
    if t1 - t0 >= min_len:
        return t0, t1
    c = 0.5 * (t0 + t1)
    a, b = c - min_len / 2, c + min_len / 2
    if a < 0:
        a, b = 0.0, min_len
    if b > dur:
        a, b = max(0.0, dur - min_len), dur
    return a, b


def run_segments(cams: dict, force: bool = False, device: str = "cuda", models: str = "ecapa,titanet") -> dict:
    """Embed every Sortformer segment (raw/<cam>/sortformer.json) with the same models as `embed`: raw/<cam>/segembed.npz
    {t0, t1 (the segment), e0, e1 (the embedded window), spk, win, rms_db, f0_med, f0_voiced, [ecapa], [titanet]}."""
    todo = {c: w for c, w in cams.items() if (force or not (C.cam_dir(c) / "segembed.npz").exists()) and (C.cam_dir(c) / "sortformer.json").exists()}
    if not todo:
        C.log("segembed: nothing to do")
        return {}
    want = {m.strip() for m in models.split(",")}
    ecapa = load_ecapa(device) if "ecapa" in want else None
    titanet = load_titanet(device) if "titanet" in want else None
    done = {}
    for cam, wav in todo.items():
        t0 = time.time()
        audio = C.load_audio(wav)
        dur = len(audio) / C.SR
        segs = core_segments(C.read_json(C.cam_dir(cam) / "sortformer.json"))
        if not segs:
            continue
        spans = [widen(s["t0"], s["t1"], dur) for s in segs]
        arrays = {"t0": np.array([s["t0"] for s in segs], np.float32), "t1": np.array([s["t1"] for s in segs], np.float32),
                  "e0": np.array([a for a, _ in spans], np.float32), "e1": np.array([b for _, b in spans], np.float32),
                  "spk": np.array([s["spk"] for s in segs], np.int16), "win": np.array([s["win"] for s in segs], np.int16)}
        # variable-length windows: embed one by one in groups of equal length (pad to the longest of the batch with the signal's own context)
        order = np.argsort([b - a for a, b in spans])
        e_ecapa = np.zeros((len(segs), 192), np.float32)
        e_tita = np.zeros((len(segs), 192), np.float32)
        rms = np.zeros(len(segs), np.float32)
        f0m = np.zeros(len(segs), np.float32)
        f0v = np.zeros(len(segs), np.float32)
        import torch
        for k in order:
            a, b = spans[k]
            x = audio[int(a * C.SR):int(b * C.SR)]
            rms[k] = C.rms_db(x)
            f0, _ = C.f0_track(x)
            v = f0[f0 > 0]
            f0v[k] = len(v) / max(len(f0), 1)
            f0m[k] = float(np.median(v)) if len(v) >= 5 else 0.0
            with torch.no_grad():
                xt = torch.from_numpy(x[None, :]).to(device)
                if ecapa is not None:
                    e_ecapa[k] = ecapa.encode_batch(xt).squeeze().float().cpu().numpy()
                if titanet is not None:
                    ln = torch.tensor([xt.shape[1]], dtype=torch.long, device=device)
                    _, emb = titanet.forward(input_signal=xt, input_signal_length=ln)
                    e_tita[k] = emb.squeeze().float().cpu().numpy()
        arrays.update(rms_db=rms, f0_med=f0m, f0_voiced=f0v)
        if ecapa is not None:
            arrays["ecapa"] = e_ecapa
        if titanet is not None:
            arrays["titanet"] = e_tita
        tmp = C.cam_dir(cam) / "segembed.tmp.npz"
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, C.cam_dir(cam) / "segembed.npz")
        C.log(f"segembed {cam}: {len(segs)} segments ({time.time() - t0:.1f} s)")
        done[cam] = len(segs)
    return done
