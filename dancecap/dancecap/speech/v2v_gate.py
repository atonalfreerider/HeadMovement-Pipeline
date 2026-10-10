"""The quality gate of the voice-to-voice pieces (agent V1): an independent ASR and a voice check on a Runpod pod, and the per-piece decision.

For every piece the gate compares THREE audios against transcript_final.json (word error rate of the piece's own words):

    mix     the piece cut from film/audio/class_mix.wav (what the film plays today)
    pre     the audio that was SENT to the voice changer (best phone, light denoise)
    <var>   the converted piece of a conversion variant, aligned to the master clock

ASR = faster-whisper large-v3 without any prompt (pod, `pod/stage_v2v.py`), a different pipeline from the one that made the transcript, so a converted
piece is judged by whether a machine that was NOT helped understands it. A converted piece is ACCEPTED when

    WER(conv) <= WER(mix) - CLEAR_GAIN                     (clearly better than the film's current audio), or
    WER(conv) <= OK_WER and WER(mix) >= BAD_WER           (not worse than 0.1 while the original is bad too)  -- the owner's rule
and the voice check passes (the output is nearer to the REFERENCE of its own teacher than to the other teacher's) and there is no hallucination
(inserted words: the ASR text must not add more than `MAX_INS` words that are not in the transcript) and no clipping.

Pod: one job of the dedicated kind `v2vgate` on ONE GPU pod (ledger owner V1-v2v), pod deleted in a `finally`, on-pod kill switch as the backstop.
Nothing here uploads audio anywhere except to the owner's own Runpod volume (the same as the speech ASR of the class): the files are the 16 kHz
pieces of the TEACHERS (no student speech exists in the class).

    python -m dancecap.speech v2v gate-build --run g1 --variants A,B
    python -m dancecap.speech v2v gate-run --run g1 [--dry-run]
    python -m dancecap.speech v2v gate-score --run g1
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from . import SPEECH, TAKE
from . import v2v as V

GATE = V.V2V / "gate"
KIND = "v2vgate"
OWNER = "V1-v2v"
CLEAR_GAIN = 0.10         # conv must beat the film's current audio by this much WER (absolute) ...
OK_WER = 0.10             # ... or be this good while the original (mix) is at least BAD_WER
BAD_WER = 0.20
MAX_INS = 2               # extra words in the ASR text of a converted piece that the transcript does not have (hallucination)
PRE_TOLERANCE = 0.0       # a converted piece must not be worse than the cleaned original it would replace
MAX_ACCEPT_WER = 0.6      # and never garbled
VOICE_MARGIN = 0.10       # cosine to the own teacher's reference minus to the other's


# ------------------------------------------------------------------------------------------------ inputs of a run


def to16k(x: np.ndarray, sr: int = V.SR) -> np.ndarray:
    return signal.resample_poly(x, 1, sr // 16000).astype(np.float32)


def read_mix_cut(t0: float, t1: float, mix_path: Path | None = None) -> np.ndarray:
    from .. import film_recap_audio as fa
    p = mix_path or (fa.AUDIO / "class_mix.wav")
    info = sf.info(str(p))
    x, _ = sf.read(str(p), start=int(t0 * info.samplerate), stop=int(t1 * info.samplerate), dtype="float32", always_2d=True)
    return x.mean(axis=1)


def build_items(pieces: list[V.Piece], variants: list[str], *, with_mix: bool = True, with_pre: bool = True, only: set[str] | None = None,
                inputs: list[str] | None = None) -> list[dict]:
    """[{key, piece, kind, audio16k}] for the ASR run (keys `<piece>__mix`, `<piece>__pre`, `<piece>__<variant>`; `inputs` = more prepared-audio folders
    under v2v/ (e.g. `iso`, `pre16`) whose ASR is wanted too, key `<piece>__in_<folder>`)"""
    items = []
    for p in pieces:
        if only and p.id not in only:
            continue
        if with_mix:
            items.append({"key": f"{p.id}__mix", "piece": p.id, "kind": "mix", "audio": to16k(read_mix_cut(p.t0, p.t1))})
        if with_pre and p.cam:
            f = V.V2V / "pre" / f"{p.id}.{p.cam}.wav"
            if f.exists():
                items.append({"key": f"{p.id}__pre", "piece": p.id, "kind": "pre", "audio": to16k(sf.read(str(f), dtype="float32")[0])})
        for d in inputs or []:
            f = V.V2V / d / f"{p.id}.{p.cam}.wav"
            if f.exists():
                items.append({"key": f"{p.id}__in_{d}", "piece": p.id, "kind": f"in_{d}", "audio": to16k(sf.read(str(f), dtype="float32")[0])})
        for v in variants:
            f = V.V2V / "aligned" / v / f"{p.id}.wav"
            if f.exists():
                items.append({"key": f"{p.id}__{v}", "piece": p.id, "kind": v, "audio": to16k(sf.read(str(f), dtype="float32")[0])})
    return items


def build_run(run: str, items: list[dict], extra: list[dict] | None = None) -> dict:
    """writes gate/<run>/{manifest.json, wav/*.wav} and the deterministic input tar; returns {manifest, tar, sha256, bytes, files}"""
    root = GATE / run
    (root / "wav").mkdir(parents=True, exist_ok=True)
    man = []
    for it in items + (extra or []):
        f = root / "wav" / f"{it['key']}.wav"
        sf.write(str(f), np.clip(it["audio"], -1, 1), 16000, subtype="PCM_16")
        man.append({"key": it["key"], "file": f"wav/{f.name}", "piece": it.get("piece"), "kind": it.get("kind"), **({"enhance": True} if it.get("enhance") else {})})
    (root / "manifest.json").write_text(json.dumps({"schema": "dancecap.speech/1/v2v_gate", "run": run, "items": man}, indent=1), encoding="utf-8")
    from ..farm import chunks as C
    rel = f"dancecap/work/{TAKE}/speech/v2v/gate/{run}"
    files = [(root / "manifest.json", f"{rel}/manifest.json")] + [(root / "wav" / f"{m['key']}.wav", f"{rel}/{m['file']}") for m in man]
    tar = root / "in.tar"
    C.pack_tar(tar, files)
    h = hashlib.sha256(tar.read_bytes()).hexdigest()
    return {"manifest": str(root / "manifest.json"), "tar": str(tar), "sha256": h, "bytes": tar.stat().st_size, "files": len(files), "items": len(man)}


def job_spec(run: str, inp: dict, max_minutes: float = 40, enhance: bool = False) -> dict:
    from ..farm import worker
    from .jobs import code_archive
    rel = f"work/{TAKE}/speech/v2v/gate/{run}"          # steps run in <ws>/dancecap
    steps = [{"name": "env", "cmd": "bash dancecap/speech/bootstrap/speech_env.sh unpack", "timeout_min": 15},
             {"name": "asr", "cmd": f"/opt/hm/env_speech/bin/python -m dancecap.speech.pod.stage_v2v asr --manifest {rel}/manifest.json --out {rel}/out" + (" --enhance" if enhance else ""),
              "timeout_min": 30, "optional": False, "save": [f"dancecap/{rel}/out/"]}]
    inputs = [{"path": f"dancecap/work/{TAKE}/speech/v2v/gate/{run}/in.tar", "archive": True, "sha256": inp["sha256"], "bytes": inp["bytes"], "local": inp["tar"]},
              {"path": "speech_code.tar", "archive": True, "local": str(code_archive())}]
    outputs = [{"path": f"dancecap/{rel}/out/"}]
    spec = worker.analyse_spec(f"v2vgate-{run}", TAKE, steps, outputs, inputs=inputs, max_minutes=max_minutes, priority=10, gpu=True, kind=KIND)
    spec.update({"retries": 1, "env": {"PYTHONPATH": "{WS}/dancecap"}})
    return spec


# ------------------------------------------------------------------------------------------------ the pod


def log(msg: str) -> None:
    print(f"[v2v_gate {time.strftime('%H:%M:%S')}] {msg}", flush=True)


STATE = GATE / "pod_state.json"


def _save_state(st: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st, indent=1, default=str), encoding="utf-8")


GPUS = ["NVIDIA L4", "NVIDIA RTX A5000", "NVIDIA RTX A4500", "NVIDIA GeForce RTX 4090"]


def run_pod(run: str, spec: dict, *, minutes: float = 30, hard_minutes: float = 45, max_cost: float = 1.0, poll_s: float = 20.0, gpus: list[str] | None = None) -> dict:
    """Queue the job, start ONE GPU pod of kind v2vgate (plus the 2-vCPU gateway the volume uploads go through), wait, pull the result, delete every pod
    this call created (also on an error). The same flow as the avatar jobs (recap_avatars_pod.run_job), with its own owner and kind."""
    from .. import recap_cameras as rc
    from ..farm import pods, volume
    from ..take import PACKAGE_ROOT
    st = {"pods": {}, "job": spec["id"], "started": time.time()}
    mine: list[str] = []
    gw = None
    try:
        gw, _ = pods.gateway(minutes=45, vcpu=2, reuse=False, owner=OWNER)
        mine.append(gw.id)
        st["pods"][gw.id] = {"role": "gateway", "started": time.time()}
        _save_state(st)
        link = volume.Link(gw)
        info = volume.sync_code(link, make_current=False)
        spec["code"] = {"git_sha": info.get("git_sha"), "dirty": info.get("dirty"), "bundle_sha256": info.get("bundle_sha256")}
        volume.submit_job(link, spec)
        log(f"job {spec['id']} queued; starting one GPU pod of kind {KIND}")
        launched = pods.launch(1, gpus=gpus or GPUS, minutes=minutes, hard_minutes=hard_minutes, max_cost=max_cost, kinds=KIND, idle_min=4,
                               prefix="hm-farm-v2v", owner=OWNER)
        for l in launched:
            if l.get("pod_id"):
                mine.append(l["pod_id"])
                st["pods"][l["pod_id"]] = {"role": "gpu", "gpu": l.get("gpu"), "price_hr": l.get("cost_per_hr"), "started": time.time()}
            else:
                log(f"launch problem: {l}")
        _save_state(st)
        if not any(l.get("pod_id") for l in launched):
            raise RuntimeError(f"no pod could be started: {launched}")
        t_last = 0.0
        while True:
            status = rc.poll(link, [spec["id"]]).get(spec["id"], "queued")
            if time.time() - t_last > 90:
                t_last = time.time()
                log(f"job {spec['id']}: {status}")
            if status == "done":
                r = volume.pull_result(link, spec["id"], PACKAGE_ROOT.parent)
                log(f"pulled {spec['id']}: {r['files']} files {r['bytes'] / 1e6:.1f} MB")
                try:                                        # keep the step logs next to the result (why an optional stage did not run)
                    (GATE / spec["id"].replace("v2vgate-", "") / "job.log").write_text(
                        link.sh(f"tail -n 400 /workspace/hm/chunks/{spec['id']}/log/*.log 2>/dev/null", 60), encoding="utf-8")
                except Exception as e:  # noqa: BLE001
                    log(f"no job log: {e}")
                return {**r, "status": "done"}
            if status == "failed":
                try:
                    f = link.read_json(f"chunks/{spec['id']}/failed.json")
                    log("failed.json: " + json.dumps(f)[:2000])
                    log("log tail:\n" + link.sh(f"tail -n 40 /workspace/hm/chunks/{spec['id']}/log/*.log 2>/dev/null | tail -n 60", 60)[-2500:])
                except Exception as e:  # noqa: BLE001
                    log(f"cannot read failed.json: {e}")
                return {"status": "failed"}
            live = [p for p in mine if p != gw.id]
            if live and not any(rc._pod_alive(p) for p in live):
                raise RuntimeError("the worker pod is gone while the job is still open")
            time.sleep(poll_s)
    finally:
        for pid in reversed(mine):
            try:
                pods.delete(pid, reason="v2v gate finished")
                log(f"pod {pid} deleted")
                st["pods"].get(pid, {})["deleted"] = time.time()
            except Exception as e:  # noqa: BLE001
                log(f"WARNING pod {pid} NOT confirmed deleted: {e}")
        _save_state(st)


def owner_spend() -> float:
    """USD booked in the farm ledger under this module's owner"""
    from .. import remote
    return round(sum(float(r.get("cost") or 0.0) for r in remote.ledger() if r.get("owner") == OWNER), 3)


# ------------------------------------------------------------------------------------------------ scoring


def read_asr(run: str) -> dict:
    d = json.loads((GATE / run / "out" / "asr.json").read_text(encoding="utf-8"))
    return d["items"]


def read_embeddings(run: str) -> dict[str, np.ndarray]:
    p = GATE / run / "out" / "embeddings.npz"
    if not p.exists():
        return {}
    z = np.load(p)
    return {str(k): v for k, v in zip(z["keys"], z["ecapa"])}


def piece_wer(piece: V.Piece, asr_item: dict | None) -> dict | None:
    """WER of the ASR text of one audio against the piece's final words (the pad context of the cut may add a word at an edge: counted as insertions)"""
    if asr_item is None:
        return None
    c = V.edit_counts(V.norm_words(piece.text), V.norm_words(asr_item["text"]))
    return {**c, "text": asr_item["text"]}


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def voice_check(speaker: str, emb: np.ndarray, refs: dict[str, np.ndarray]) -> dict:
    """cosine of a piece's embedding to the references of both teachers; ok when its own teacher's is higher by VOICE_MARGIN"""
    other = "Kadu" if speaker == "Larissa" else "Larissa"
    own, oth = cosine(emb, refs[speaker]), cosine(emb, refs[other])
    return {"own": round(own, 3), "other": round(oth, 3), "ok": own - oth >= VOICE_MARGIN}


def decide(piece: V.Piece, wers: dict[str, dict | None], voice: dict | None, clipped: int = 0, *, kind: str) -> dict:
    """accept / reject one converted candidate of a piece. wers needs 'mix' and `kind`, and 'pre' (the cleaned best-phone original, the fallback) when
    available: a converted piece that is WORSE than the cleaned original it would replace is never used (it would say wrong words in a clean voice)."""
    mix, conv, pre = wers.get("mix"), wers.get(kind), wers.get("pre")
    if conv is None or mix is None:
        return {"accept": False, "why": "no ASR result"}
    w_mix, w_conv = mix["wer"], conv["wer"]
    reasons = []
    gain = w_mix - w_conv
    better = gain >= CLEAR_GAIN
    both_bad = w_conv <= OK_WER and w_mix >= BAD_WER
    if not (better or both_bad):
        reasons.append(f"WER {w_conv:.2f} not clearly better than the mix's {w_mix:.2f} (needs gain >= {CLEAR_GAIN} or <= {OK_WER} with a bad original)")
    if pre is not None and w_conv > pre["wer"] + PRE_TOLERANCE:
        reasons.append(f"WER {w_conv:.2f} is worse than the cleaned original's {pre['wer']:.2f}")
    if w_conv > MAX_ACCEPT_WER:
        reasons.append(f"WER {w_conv:.2f} > {MAX_ACCEPT_WER}: garbled")
    if conv["ins"] > MAX_INS:
        reasons.append(f"{conv['ins']} inserted words (hallucination?): '{conv['text'][:80]}'")
    if voice is not None and not voice["ok"]:
        reasons.append(f"voice check failed (own {voice['own']}, other {voice['other']})")
    if clipped > 5:
        reasons.append(f"{clipped} clipped samples")
    return {"accept": not reasons, "wer_mix": round(w_mix, 3), "wer_conv": round(w_conv, 3), "wer_pre": None if pre is None else round(pre["wer"], 3), "gain": round(gain, 3),
            "why": "; ".join(reasons) or ("clearly better than the film's current audio" if better else "as good as 0.1 while the original is bad too")}


def decide_all(pieces: list[V.Piece], asr: dict, emb: dict[str, np.ndarray], variants: list[str], cams: dict[str, dict[str, str]] | None = None) -> tuple[dict, list[dict]]:
    """decisions {piece id: {"variant": name | None, "cam": phone, "why": ...}} and the table rows. For each piece the variants are tried in order (the
    first accepted wins); a piece with no accepted variant keeps the cleaned original (`variant` None). `cams` = {variant: {piece id: phone}} where a
    variant used another phone than the plan's."""
    refs = {n: np.mean([emb[k] for k in emb if k.startswith(f"ref_{n}_")], axis=0) for n in V.VOICE_NAMES if any(k.startswith(f"ref_{n}_") for k in emb)}
    dec, rows = {}, []
    for p in pieces:
        w = {"mix": piece_wer(p, asr.get(f"{p.id}__mix")), "pre": piece_wer(p, asr.get(f"{p.id}__pre"))}
        row = {"id": p.id, "speaker": p.speaker, "dur": round(p.dur, 1), "words": p.n_words, "wer_mix": None if w["mix"] is None else round(w["mix"]["wer"], 3),
               "wer_cleaned_original": None if w["pre"] is None else round(w["pre"]["wer"], 3), "variants": {}}
        chosen = None
        for v in variants:
            it = asr.get(f"{p.id}__{v}")
            if it is None:
                continue
            w[v] = piece_wer(p, it)
            vc = voice_check(p.speaker, emb[f"{p.id}__{v}"], refs) if f"{p.id}__{v}" in emb and refs else None
            al = V.V2V / "aligned" / v / f"{p.id}.json"
            clipped = json.loads(al.read_text(encoding="utf-8")).get("clipped_samples", 0) if al.exists() else 0
            d = decide(p, w, vc, clipped, kind=v)
            row["variants"][v] = {**d, "voice": vc, "ins": w[v]["ins"], "asr": w[v]["text"][:200]}
            if d["accept"] and chosen is None:
                chosen = v
        cam = (cams or {}).get(chosen, {}).get(p.id) if chosen else None
        dec[p.id] = {"variant": chosen, "cam": cam or p.cam,
                     "why": (row["variants"][chosen]["why"] if chosen else "no converted variant passed the gate: the cleaned original (best phone, light denoise) is used")}
        row["chosen"] = chosen
        rows.append(row)
    return dec, rows


# ------------------------------------------------------------------------------------------------ the voices' own reference clips


def reference_items(n_clips: int = 3, clip_s: float = 10.0) -> list[dict]:
    """16 kHz clips of the two voices' OWN training samples (downloaded from the owner's account once into v2v/ref/, the clean recordings the clones were
    made from): the reference of the voice check. `n_clips` slices of `clip_s` seconds spread over the sample."""
    items = []
    for name in V.VOICE_NAMES:
        f = V.V2V / "ref" / f"{name}_sample.bin"
        if not f.exists():
            raise V.V2VError(f"{f.name} missing: run `python -m dancecap.speech v2v refs`")
        x, sr = _decode_any(f)
        n = len(x) / 16000
        for k in range(n_clips):
            t0 = (n - clip_s) * (k / max(n_clips - 1, 1)) if n > clip_s else 0.0
            items.append({"key": f"ref_{name}_{k}", "piece": None, "kind": "ref", "speaker": name, "audio": x[int(t0 * 16000):int((t0 + clip_s) * 16000)]})
    return items


def _decode_any(path: Path) -> tuple[np.ndarray, int]:
    """mp3 (or anything ffmpeg reads) -> 16 kHz mono float32"""
    from .. import film_recap_audio as fa
    tmp = path.with_suffix(".16k.wav")
    if not tmp.exists():
        fa.ffmpeg("-y", "-i", str(path), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(tmp))
    x, sr = sf.read(str(tmp), dtype="float32")
    return x, sr


def fetch_references(session=None) -> dict:
    """download the single training sample of each of the two voices (owner's own account) into v2v/ref/"""
    session = session or V.http_session()
    ids = V.voice_ids()
    out = {}
    (V.V2V / "ref").mkdir(parents=True, exist_ok=True)
    for name, vid in ids.items():
        r = session.get(f"{V.API}/v1/voices/{vid}", timeout=60)
        if r.status_code != 200:
            raise V.V2VError(f"GET voice {name} -> {r.status_code}")
        samples = r.json().get("samples") or []
        if not samples:
            raise V.V2VError(f"voice {name} has no samples")
        a = session.get(f"{V.API}/v1/voices/{vid}/samples/{samples[0]['sample_id']}/audio", timeout=180)
        if a.status_code != 200:
            raise V.V2VError(f"GET sample of {name} -> {a.status_code}")
        (V.V2V / "ref" / f"{name}_sample.bin").write_bytes(a.content)
        out[name] = {"bytes": len(a.content), "seconds": samples[0].get("duration_secs")}
    return out
