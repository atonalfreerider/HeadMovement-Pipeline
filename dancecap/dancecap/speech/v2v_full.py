"""The full cloned-voice re-recording of the class recap (agent V2), building on V1's voice-to-voice machinery (`v2v`, `v2v_gate`, `v2v_mix`).

V1 converted whole turns (1-35 s) from one lightly cleaned phone and the speech-to-speech model garbled most of them (the noisy, reverberant phone
audio is the input the model cannot read). V2 changes the INPUT and the GRANULARITY, keeps the independent ASR check, and makes the owner's policy
explicit:

  * sentence-level pieces (about 2-8 s, cut at pauses and utterance changes) so a failure is local;
  * several candidate inputs per piece (best phone, second phone, a time-aligned weighted sum of the three best phones, each optionally passed through a
    neural enhancer on the pod); the enhancer is picked by the independent ASR WER against transcript_final;
  * every piece is converted, checked with the ASR, and re-tried (next candidate input / settings, up to `MAX_TRIES`) before the enhanced original is used;
  * the decision (`decide_policy`): the converted piece is used unless it is clearly garbled, i.e. its WER is worse than the enhanced original's by more than
    0.10 AND above 0.25 (absolute); then the enhanced original is used and the piece is flagged.

Work files live in work/<take>/speech/v2v2/. The voices of real people never leave the machine except to the owner's ElevenLabs account (the owner's
explicit permission, 2026-10-10) and his own Runpod volume/pod.
"""
from __future__ import annotations

import math

import numpy as np

from . import v2v as V

SR = V.SR
MIN_S, MAX_S, PAUSE_S, HARD_GAP_S, MIN_TAIL_S = 2.5, 8.0, 0.30, 1.2, 1.5
UTT_BONUS_S = 0.3             # a change of utterance counts like this much extra pause when choosing a cut
MAX_TRIES = 3
GARBLE_MARGIN = 0.10          # converted WER worse than the enhanced original by more than this ...
GARBLE_ABS = 0.25             # ... and above this absolute WER = clearly garbled


# ------------------------------------------------------------------------------------------------ sentence-level pieces


def _gap_eff(a: dict, b: dict) -> float:
    return (b["t0"] - a["t1"]) + (UTT_BONUS_S if a.get("utt") != b.get("utt") else 0.0)


def sentence_split(run: list[dict], *, min_s: float = MIN_S, max_s: float = MAX_S, pause: float = PAUSE_S, hard_gap: float = HARD_GAP_S,
                   min_tail: float = MIN_TAIL_S) -> list[list[dict]]:
    """Cut one run of one teacher's words into pieces of about min_s..max_s seconds at pauses (hard cut at gaps >= hard_gap; soft cut at a gap >= pause
    once the piece is min_s long; when a piece would pass max_s it is cut at the longest gap that leaves both sides >= 1.5 s, falling back to the longest
    gap); a short tail is merged into the piece before it when the result stays <= max_s + 3 s."""
    out: list[list[dict]] = []
    cur: list[dict] = []

    def dur(ws):
        return ws[-1]["t1"] - ws[0]["t0"]

    def cut_long(ws):
        while dur(ws) > max_s and len(ws) > 1:
            cands = [(_gap_eff(ws[i], ws[i + 1]), i) for i in range(len(ws) - 1)
                     if dur(ws[:i + 1]) >= 1.5 and ws[-1]["t1"] - ws[i + 1]["t0"] >= 1.5 and dur(ws[:i + 1]) <= max_s + 1.0]
            if not cands:
                cands = [(_gap_eff(ws[i], ws[i + 1]), i) for i in range(len(ws) - 1)]
            _, i = max(cands)
            out.append(ws[:i + 1])
            ws = ws[i + 1:]
        return ws

    for k, w in enumerate(run):
        cur.append(w)
        nxt = run[k + 1] if k + 1 < len(run) else None
        if nxt is None:
            break
        gap = nxt["t0"] - w["t1"]
        if gap >= hard_gap:
            cur = cut_long(cur)
            out.append(cur)
            cur = []
        elif dur(cur) >= min_s and _gap_eff(w, nxt) >= pause:
            cur = cut_long(cur)
            out.append(cur)
            cur = []
        elif dur(cur) > max_s + 1.0:
            cur = cut_long(cur)
    if cur:
        cur = cut_long(cur)
        out.append(cur)
    # a short tail joins the piece before it (not across a hard gap)
    merged: list[list[dict]] = []
    for p in out:
        if merged and dur(p) < min_tail and p[0]["t0"] - merged[-1][-1]["t1"] < hard_gap and dur(merged[-1] + p) <= max_s + 3.0:
            merged[-1] = merged[-1] + p
        else:
            merged.append(p)
    return merged


# ------------------------------------------------------------------------------------------------ policy


def decide_policy(wer_conv: float | None, wer_orig: float | None, *, margin: float = GARBLE_MARGIN, abs_wer: float = GARBLE_ABS) -> dict:
    """The owner wants the cloned voice: use the converted piece unless its WER is worse than the enhanced original's by more than `margin` AND above
    `abs_wer` (clearly garbled). No converted candidate -> the enhanced original (flagged); no original ASR -> the converted piece."""
    if wer_conv is None:
        return {"use": "original", "flag": "no converted candidate"}
    if wer_orig is None:
        return {"use": "converted", "flag": None}
    if wer_conv - wer_orig > margin and wer_conv > abs_wer:
        return {"use": "original", "flag": f"garbled: converted WER {wer_conv:.2f} vs enhanced original {wer_orig:.2f}"}
    return {"use": "converted", "flag": None}


def onset_bad(t: dict, limit_ms: float = 80.0) -> bool:
    """a try whose measured word onsets (see v2v_check.piece_onset_errors, stored by the timing pass) are off by more than limit_ms"""
    errs = (t.get("onset") or {}).get("errs") or []
    return any(abs(e) > limit_ms for e in errs)


def pick_best_try(tries: list[dict], timing_penalty: float = 0.0) -> dict | None:
    """of the converted tries of a piece [{name, wer, ...}] the one with the lowest WER (ties: the earlier try); with timing_penalty a try with a measured
    word onset off by more than 80 ms counts that much worse"""
    ok = [t for t in tries if t.get("wer") is not None]
    return min(ok, key=lambda t: t["wer"] + (timing_penalty if onset_bad(t) else 0.0)) if ok else None


# ------------------------------------------------------------------------------------------------ several phones -> one input


def fractional_shift(x: np.ndarray, delay_samples: float) -> np.ndarray:
    """delay `x` by delay_samples (may be fractional / negative) with a phase shift in the frequency domain (zero padded, same length)"""
    n = len(x)
    nf = int(2 ** math.ceil(math.log2(n + 2048)))
    X = np.fft.rfft(x.astype(np.float64), nf)
    X *= np.exp(-2j * np.pi * np.fft.rfftfreq(nf) * delay_samples)
    return np.fft.irfft(X, nf)[:n].astype(np.float32)


def gcc_phat_delay(ref: np.ndarray, sig: np.ndarray, sr: int = SR, max_s: float = 0.02, band: tuple[float, float] = (200.0, 4000.0)) -> tuple[float, float]:
    """delay (samples, fractional) by which `sig` lags `ref` (positive = sig later), GCC-PHAT in the speech band, searched within +-max_s; also the
    peak-to-median ratio of the correlation as a confidence"""
    n = min(len(ref), len(sig))
    nf = int(2 ** math.ceil(math.log2(2 * n)))
    A, B = np.fft.rfft(ref[:n].astype(np.float64), nf), np.fft.rfft(sig[:n].astype(np.float64), nf)
    f = np.fft.rfftfreq(nf, 1 / sr)
    R = B * np.conj(A)
    R = R / (np.abs(R) + 1e-12) * ((f >= band[0]) & (f <= band[1]))
    cc = np.fft.irfft(R, nf)
    k = int(max_s * sr)
    seg = np.concatenate([cc[-k:], cc[:k + 1]])
    j = int(np.argmax(seg))
    frac = 0.0
    if 0 < j < len(seg) - 1:
        y0, y1, y2 = seg[j - 1], seg[j], seg[j + 1]
        d = y0 - 2 * y1 + y2
        frac = 0.5 * (y0 - y2) / d if abs(d) > 1e-12 else 0.0
    conf = float(seg[j] / (np.median(np.abs(seg)) + 1e-12))
    return float(j - k + frac), conf


def beamform(chans: list[np.ndarray], snr_db: list[float], sr: int = SR, *, min_conf: float = 6.0, max_shift_s: float = 0.02) -> tuple[np.ndarray, dict]:
    """Weighted sum of phones cut from the master clock (first = the best phone = the reference): every other phone is aligned to the reference by GCC-PHAT
    (kept only with a clear peak), its gain brought to the reference's by an RMS ratio, and weights follow the linear speech-to-noise ratio. Returns
    (mono, info). With a single phone it is that phone."""
    ref = chans[0].astype(np.float32)
    acc = ref * 1.0
    wsum = 1.0
    used = [{"idx": 0, "delay_ms": 0.0, "w": 1.0}]
    r0 = float(np.sqrt((ref.astype(np.float64) ** 2).mean()) + 1e-9)
    for i, (c, s) in enumerate(zip(chans[1:], snr_db[1:]), start=1):
        d, conf = gcc_phat_delay(ref, c, sr, max_shift_s)
        if conf < min_conf:
            used.append({"idx": i, "skipped": f"no clear correlation peak ({conf:.1f})"})
            continue
        a = fractional_shift(c, -d)
        g = r0 / (float(np.sqrt((a.astype(np.float64) ** 2).mean())) + 1e-9)
        w = float(10 ** ((s - snr_db[0]) / 20.0))
        w = min(max(w, 0.25), 1.0)
        acc = acc + w * g * a
        wsum += w
        used.append({"idx": i, "delay_ms": round(d / sr * 1000, 2), "conf": round(conf, 1), "gain": round(g, 2), "w": round(w, 2)})
    return (acc / wsum).astype(np.float32), {"phones": used}


# ------------------------------------------------------------------------------------------------ plan and candidate inputs (files under work/<take>/speech/v2v2/)

import json
import sys
from dataclasses import asdict
from pathlib import Path

import soundfile as sf

from . import SPEECH

V2 = SPEECH / "v2v2"
DURATION = 360.476783
PHONES = ("00", "01", "02", "03", "05", "06")
IN_DIR = V2 / "in"


def plan_segments() -> tuple[list[V.Piece], list[dict], dict]:
    """sentence-level pieces of the whole class (ids s001..), the kept-original stretches, and the ranked phones per piece; writes v2v2/plan.json and
    v2v2/sources.json. The opening is original until speech/owner_opening.json exists (then it is converted too)."""
    tr = V.read_transcript()
    ws = V.transcript_words(tr)
    open_end = V.opening_end(ws)
    pieces, kept = V.plan_pieces(ws, open_end=open_end, duration=DURATION, split_fn=sentence_split)
    for i, p in enumerate(pieces, 1):
        p.id = f"s{i:03d}"
    table = V.choose_sources(pieces, ws, DURATION)
    V2.mkdir(parents=True, exist_ok=True)
    V.write_plan(pieces, kept, V2 / "plan.json")
    (V2 / "sources.json").write_text(json.dumps(table, indent=1), encoding="utf-8")
    return pieces, kept, table


def read_segments() -> tuple[list[V.Piece], list[dict], dict]:
    pieces, kept = V.read_plan(V2 / "plan.json")
    return pieces, kept, json.loads((V2 / "sources.json").read_text(encoding="utf-8"))


def build_inputs(pieces: list[V.Piece], table: dict) -> dict:
    """v2v2/in/<cand>/<id>.wav (48 kHz mono, cut [t0, t1] of the master clock): r1 / r2 = the best / second-best phone as recorded (no processing),
    bf = delay-aligned weighted sum of the best three, pre1 = V1's light clean-up of r1 (for reference). Returns {id: info}."""
    cache: dict[str, np.ndarray] = {}
    info = {}
    for p in pieces:
        rows = table[p.id]
        cams = [r["cam"] for r in rows[:3]]
        for c in cams:
            if c not in cache:
                cache[c] = V.phone_on_master(c, DURATION)
        cuts = [V.cut_master(cache[c], p.t0, p.t1) for c in cams]
        bf, binfo = beamform(cuts, [r["snr_db"] for r in rows[:3]], max_shift_s=0.006, min_conf=4.5)
        outs = {"r1": cuts[0], "r2": cuts[1] if len(cuts) > 1 else cuts[0], "bf": bf, "pre1": V.prepare_piece(cuts[0])[0]}
        for k, a in outs.items():
            f = IN_DIR / k / f"{p.id}.wav"
            f.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(f), np.clip(a, -1, 1), SR, subtype="PCM_16")
        info[p.id] = {"cams": cams, "beamform": binfo}
    (V2 / "inputs.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    return info


def main(argv=None) -> int:
    a = list(sys.argv[1:] if argv is None else argv)
    cmd = a.pop(0)
    if cmd == "plan":
        pieces, kept, table = plan_segments()
        d = [p.dur for p in pieces]
        print(f"{len(pieces)} pieces, {len(kept)} kept; dur mean {np.mean(d):.1f} median {np.median(d):.1f} max {max(d):.1f} total {sum(d):.0f} s", V.cost_estimate(pieces))
    elif cmd == "inputs":
        pieces, kept, table = read_segments()
        build_inputs(pieces, table)
        print("inputs built for", len(pieces))
    return 0


if __name__ == "__main__":
    sys.exit(main())


def build_mix_inputs(pieces: list[V.Piece]) -> None:
    """in/mix/<id>.wav = the piece cut from the old class mix (the baseline the film plays today)"""
    from . import v2v_gate as G
    for p in pieces:
        f = IN_DIR / "mix" / f"{p.id}.wav"
        f.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(f), np.clip(G.read_mix_cut(p.t0, p.t1), -1, 1), SR, subtype="PCM_16")


def pack_dirs(dirs: list[str], tar_path: Path, base: Path | None = None) -> Path:
    """tar of the folders (paths relative to `base`, default v2v2/) for the pod"""
    import tarfile
    base = base or V2
    with tarfile.open(tar_path, "w") as tar:
        for d in dirs:
            for f in sorted((base / d).rglob("*")):
                if f.is_file():
                    tar.add(f, arcname=f.relative_to(base).as_posix())
    return tar_path


# ------------------------------------------------------------------------------------------------ conversion of a file (cached), alignment, credits of THIS session

LEDGER = V2 / "spend_ledger.jsonl"
SESSION_CAP_CREDITS = 60000
SETTINGS = {
    "default": None,
    "B": {"stability": 0.8, "similarity_boost": 0.9, "style": 0.0, "use_speaker_boost": True},
    "S": {"stability": 0.35, "similarity_boost": 0.6, "style": 0.0, "use_speaker_boost": True},
    "T": {"stability": 1.0, "similarity_boost": 1.0, "style": 0.0, "use_speaker_boost": True},
    "L0": {"stability": 0.0, "similarity_boost": 0.6, "style": 0.0, "use_speaker_boost": True},
    "L2": {"stability": 0.2, "similarity_boost": 0.6, "style": 0.0, "use_speaker_boost": True},
    "L2s3": {"stability": 0.2, "similarity_boost": 0.3, "style": 0.0, "use_speaker_boost": True},
    "M5s3": {"stability": 0.5, "similarity_boost": 0.3, "style": 0.0, "use_speaker_boost": True},
    "L2nb": {"stability": 0.2, "similarity_boost": 0.6, "style": 0.0, "use_speaker_boost": False},
}


def use_v2_ledger() -> None:
    """the ElevenLabs request ledger of this session (credits + request ids, never the key)"""
    V.LEDGER = LEDGER


def session_credits() -> int:
    return V.ledger_credits(LEDGER)


def prepare_for_sts(x: np.ndarray, *, hp: bool = True) -> np.ndarray:
    """the file sent to the voice changer from an (enhanced) 48 kHz input: optional 80 Hz high-pass, level -20 dBFS (95th pct of 50 ms RMS), peak <= -1.5 dBFS"""
    if hp:
        x = V.highpass(x, SR)
    y, _ = V.level_to(x.astype(np.float32))
    return y


def convert_file(session, voice_id: str, src_wav: Path, out_dir: Path, name: str, *, model: str = V.MODEL_DEFAULT, settings_key: str = "B",
                 rbn: bool = True, seed: int | None = None, purpose: str = "", request=None, budget: int = SESSION_CAP_CREDITS) -> dict:
    """send src_wav (48 kHz mono) to speech-to-speech with the teacher's voice; writes out_dir/<name>.wav (24 kHz as received) + <name>.json (cache key hash,
    credits, request id). Cached by a hash of the audio + voice + model + settings + seed. Raises QuotaError / V2VError; refuses beyond the session budget."""
    request = request or V.sts_request
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_f, meta_f = out_dir / f"{name}.wav", out_dir / f"{name}.json"
    audio = src_wav.read_bytes()
    st = SETTINGS[settings_key]
    h = V.request_hash(audio, voice_id, model, {"s": st, "rbn": rbn, "seed": seed, "f": "pcm_24000"})
    if wav_f.exists() and meta_f.exists():
        meta = json.loads(meta_f.read_text(encoding="utf-8"))
        if meta.get("hash") == h:
            return {**meta, "cached": True}
    sec = len(audio) / 2 / SR
    if session_credits() + math.ceil(sec * 10.3) > budget:
        raise V.V2VError(f"session credit budget {budget} would be exceeded (used {session_credits()})")
    raw, info = request(session, voice_id, audio, model=model, settings=st, remove_background_noise=rbn, output_format="pcm_24000", seed=seed, purpose=purpose or name)
    x, sr = V.pcm_to_float(raw, "pcm_24000")
    sf.write(str(wav_f), x, sr, subtype="PCM_16")
    meta = {"hash": h, "model": model, "settings": settings_key, "rbn": rbn, "seed": seed, "sent_s": round(sec, 3), "rate": sr, "seconds_audio": round(len(x) / sr, 3), **info}
    meta_f.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return {**meta, "cached": False}


def align_to_source(conv_wav: Path, src_wav: Path, out_wav: Path) -> dict:
    """the converted piece (24 kHz) put back on its source's time base (48 kHz, same length): envelope lag, no stretch unless the drift exceeds 80 ms"""
    from scipy import signal as sg
    conv, sr = sf.read(str(conv_wav), dtype="float32")
    if sr != SR:
        g = math.gcd(SR, sr)
        conv = sg.resample_poly(conv, SR // g, sr // g).astype(np.float32)
    src, _ = sf.read(str(src_wav), dtype="float32")
    y, rep = V.align_converted(src, conv)
    rep["clipped_samples"] = int((np.abs(y) >= 0.999).sum())
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_wav), np.clip(y, -1, 1), SR, subtype="PCM_16")
    return rep


# ------------------------------------------------------------------------------------------------ ASR scores (the ASR runs on the pod: pod/asr_v2v.py)

ASR_DIR = V2 / "asr"


def pull_asr(names: list[str] | None = None, npz: bool = False) -> list[str]:
    """copy the asr json (and npz with npz=True) files of the pod (/root/data/asr) to v2v2/asr/ in ONE transfer (all, or those starting with one of `names`)"""
    import tarfile
    from . import v2v_pod as P
    h = P.handle()
    pats = " ".join(f"{n}*.json" + (f" {n}*.npz" if npz else "") for n in names) if names else ("*.json *.npz" if npz else "*.json")
    h.run(f"cd /root/data/asr && tar -cf /root/asr_pull.tar {pats} 2>/dev/null", check=False, timeout=300, echo=False)
    loc = V2 / "asr_pull.tar"
    h.get("/root/asr_pull.tar", loc, timeout=900)
    ASR_DIR.mkdir(parents=True, exist_ok=True)
    with tarfile.open(loc) as t:
        got = t.getnames()
        t.extractall(ASR_DIR)
    loc.unlink()
    return got


def read_asr_set(name: str) -> dict[str, dict]:
    return json.loads((ASR_DIR / f"{name}.json").read_text(encoding="utf-8"))


def score_set(pieces: list[V.Piece], asr: dict[str, dict]) -> dict:
    """{wer (errors / reference words over all pieces present), n_words, per_piece {id: {wer, sub, ins, del, n_ref}}}"""
    per, err, n = {}, 0, 0
    for p in pieces:
        it = asr.get(p.id)
        if it is None:
            continue
        c = V.edit_counts(V.norm_words(p.text), V.norm_words(it["text"]))
        per[p.id] = c
        err += c["sub"] + c["ins"] + c["del"]
        n += c["n_ref"]
    return {"wer": round(err / max(n, 1), 4), "n_words": n, "pieces": len(per), "per_piece": per}


def sts_input_path(cand: str, pid: str) -> Path:
    return V2 / "sts_in" / cand / f"{pid}.wav"


def make_sts_inputs(pieces: list[V.Piece], cand: str, src_dir: Path | None = None, hp: bool = True) -> None:
    """v2v2/sts_in/<cand>/<id>.wav = the candidate input (v2v2/in/<cand> or src_dir) high-passed and levelled for the voice changer"""
    src_dir = src_dir or IN_DIR / cand
    for p in pieces:
        f = src_dir / f"{p.id}.wav"
        if not f.exists():
            continue
        x, sr = sf.read(str(f), dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        assert sr == SR, (f, sr)
        o = sts_input_path(cand, p.id)
        o.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(o), np.clip(prepare_for_sts(x, hp=hp), -1, 1), SR, subtype="PCM_16")


def convert_set(pieces: list[V.Piece], cand: str, tag: str, ids: set[str] | None = None, *, model: str = V.MODEL_DEFAULT, settings_key: str = "B",
                rbn: bool = True, seed: int | None = None, session=None, request=None) -> dict:
    """convert the sts inputs of candidate `cand` for the pieces `ids` into v2v2/conv/<tag>/ and align them to their inputs (v2v2/aligned/<tag>/<id>.wav).
    Stops at a quota error. Returns {done, cached, credits, failed, stopped}."""
    use_v2_ledger()
    session = session or V.http_session()
    vids = V.voice_ids()
    res: dict = {"done": [], "cached": [], "credits": 0, "failed": {}, "stopped": None}
    for p in pieces:
        if ids and p.id not in ids:
            continue
        src = sts_input_path(cand, p.id)
        if not src.exists():
            continue
        try:
            m = convert_file(session, vids[p.speaker], src, V2 / "conv" / tag, p.id, model=model, settings_key=settings_key, rbn=rbn, seed=seed,
                             purpose=f"{tag} {p.id}", request=request)
        except V.QuotaError as e:
            res["stopped"] = f"quota: {e}"
            break
        except V.V2VError as e:
            res["failed"][p.id] = str(e)[:200]
            if "budget" in str(e):
                res["stopped"] = str(e)
                break
            continue
        (res["cached"] if m["cached"] else res["done"]).append(p.id)
        res["credits"] += 0 if m["cached"] else int(m.get("credits") or 0)
        rep = align_to_source(V2 / "conv" / tag / f"{p.id}.wav", src, V2 / "aligned" / tag / f"{p.id}.wav")
        (V2 / "aligned" / tag / f"{p.id}.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    return res


def push_aligned(tags: list[str], root: str = "aligned") -> None:
    """send v2v2/<root>/<tag>/*.wav (as 16 kHz wavs) to the pod's /root/data/<root>/<tag>/"""
    import tarfile
    import tempfile
    from scipy import signal as sg
    from . import v2v_pod as P
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        for tag in tags:
            for f in sorted((V2 / root / tag).glob("*.wav")):
                x, sr = sf.read(str(f), dtype="float32")
                if sr != 16000:
                    g = math.gcd(16000, sr)
                    x = sg.resample_poly(x, 16000 // g, sr // g).astype(np.float32)
                o = tdp / root / tag / f.name
                o.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(o), x, 16000, subtype="PCM_16")
        tar = tdp / "push.tar"
        with tarfile.open(tar, "w") as t:
            t.add(tdp / root, arcname=root)
        h = P.handle()
        h.put(tar, "/root/push.tar", timeout=1800)
        h.run("cd /root/data && tar -xf /root/push.tar", timeout=600, echo=False)


def pull_out(name: str, ids: list[str] | None = None) -> int:
    """enhanced candidate `name` (pod /root/data/out/<name>, 48 kHz wavs) -> v2v2/in/<name>/ (only `ids` when given); returns the number of files"""
    import tarfile
    from . import v2v_pod as P
    h = P.handle()
    sel = " ".join(f"{i}.wav" for i in ids) if ids else "*.wav"
    h.run(f"cd /root/data/out/{name} && tar -cf /root/pull_{name}.tar {sel}", timeout=300, echo=False)
    loc = V2 / f"pull_{name}.tar"
    h.get(f"/root/pull_{name}.tar", loc, timeout=1800)
    dst = IN_DIR / name
    dst.mkdir(parents=True, exist_ok=True)
    with tarfile.open(loc) as t:
        t.extractall(dst)
    loc.unlink()
    return len(list(dst.glob("*.wav")))


# ------------------------------------------------------------------------------------------------ the campaign: convert, check with the ASR, retry (best of N tries per piece)

TRIES_FILE = V2 / "tries.json"
TARGET_WER = 0.10             # a piece is "done" when a converted try reaches this word error rate vs the transcript
# try k uses RECIPES[k-1]: (candidate input, settings key, model, seed); settings with similarity 0.6 or more only (the voice must stay itself)
RECIPES = [
    ("mossformer48__bf", "S", V.MODEL_DEFAULT, 1),
    ("mossformer48__r1", "M5s3", V.MODEL_DEFAULT, 2),
    ("bf", "L0", V.MODEL_DEFAULT, 3),
    ("drrof_dn__bf", "B", V.MODEL_DEFAULT, 4),
    ("mossformer48__bf", "L2", V.MODEL_DEFAULT, 5),
    ("bf", "S", V.MODEL_DEFAULT, 6),
    ("mossformer48__r1", "L0", V.MODEL_DEFAULT, 7),
    ("drrof_dn__bf", "S", V.MODEL_DEFAULT, 8),
    # a second lap with new seeds (the same piece converted again is a fresh draw): only for pieces still above the target
    ("mossformer48__bf", "L0", V.MODEL_DEFAULT, 9),
    ("bf", "L2", V.MODEL_DEFAULT, 10),
    ("mossformer48__r1", "S", V.MODEL_DEFAULT, 11),
    ("mossformer48__bf", "B", V.MODEL_DEFAULT, 12),
    ("drrof_dn__bf", "L0", V.MODEL_DEFAULT, 13),
    ("bf", "M5s3", V.MODEL_DEFAULT, 14),
    ("mossformer48__r1", "L2", V.MODEL_DEFAULT, 15),
    ("mossformer48__bf", "S", V.MODEL_DEFAULT, 16),
]


def load_tries() -> dict:
    return json.loads(TRIES_FILE.read_text(encoding="utf-8")) if TRIES_FILE.exists() else {}


def save_tries(t: dict) -> None:
    TRIES_FILE.write_text(json.dumps(t, indent=1), encoding="utf-8")


def need_try(tries: list[dict], target: float = TARGET_WER, max_tries: int = len(RECIPES)) -> bool:
    """a piece needs another try while no try reached the target and fewer than max_tries were made"""
    best = pick_best_try(tries)
    return len(tries) < max_tries and (best is None or best["wer"] > target)


def pod_asr(dirs: list[str], embed: bool = False, force: bool = False) -> None:
    from . import v2v_pod as P
    P.handle().run("cd /root/data && PYTHONPATH=/root/ws/dancecap HM_ROOT=/workspace/hm /opt/hm/env_speech/bin/python -m dancecap.speech.pod.asr_v2v --root /root/data --dirs "
                   + ",".join(dirs) + (" --embed" if embed else "") + (" --force" if force else "") + " 2>&1 | grep -v 'asr [0-9]*/' | tail -1", timeout=1800)


def campaign_round(k: int, pieces: list[V.Piece], *, target: float = TARGET_WER, max_credits: int | None = None) -> dict:
    """try number k (1-based) for every piece that still needs one: convert with RECIPES[k-1], align, ASR on the pod, record the WER in tries.json"""
    cand, skey, model, seed = RECIPES[k - 1]
    tries = load_tries()
    todo = {p.id for p in pieces if len(tries.get(p.id, [])) == k - 1 and need_try(tries.get(p.id, []), target)}
    if not todo:
        return {"round": k, "pieces": 0}
    if not (IN_DIR / cand).exists() or len(list((IN_DIR / cand).glob("*.wav"))) < len(pieces):
        pull_out(cand)
    make_sts_inputs(pieces, cand)
    tag = f"t{k}"
    before = session_credits()
    r = convert_set(pieces, cand, tag, todo, model=model, settings_key=skey, rbn=True, seed=seed)
    done_ids = sorted(set(r["done"]) | set(r["cached"]))
    if done_ids:
        push_aligned([tag])
        pod_asr([f"aligned/{tag}"])
        pull_asr([f"aligned__{tag}"])
        sc = score_set([p for p in pieces if p.id in set(done_ids)], read_asr_set(f"aligned__{tag}"))
        for pid in done_ids:
            c = sc["per_piece"].get(pid)
            tries.setdefault(pid, []).append({"k": k, "tag": tag, "cand": cand, "settings": skey, "model": model, "seed": seed,
                                              "wer": None if c is None else round(c["wer"], 4), "ins": None if c is None else c["ins"]})
        save_tries(tries)
    else:
        sc = {"wer": None}
    return {"round": k, "cand": cand, "settings": skey, "pieces": len(todo), "converted": len(done_ids), "credits": session_credits() - before, "failed": r["failed"],
            "stopped": r["stopped"], "wer_round": sc["wer"]}


# ------------------------------------------------------------------------------------------------ decisions and the assembly of class_mix_v2v_full.wav

ORIG_CAND = "mossformer48__bf"      # the enhanced original: 3-phone delay-and-sum + MossFormer2_SE_48K (lowest ASR WER of the candidates, see NOTES_V2.md)


def build_decisions(pieces: list[V.Piece], tries: dict, orig_wer: dict[str, float | None], *, bad_tries: dict[str, set] | None = None,
                    timing_penalty: float = 0.0) -> dict:
    """{piece id: {"variant": tag | None, "use", "flag", "wer_conv", "wer_orig", "tries", "cam"}} by the owner's policy (`decide_policy`) on the best try of the
    piece (lowest WER, tries listed in `bad_tries[id]` -- e.g. failing the voice check -- are ignored)."""
    dec = {}
    for p in pieces:
        ts = [t for t in tries.get(p.id, []) if t["tag"] not in (bad_tries or {}).get(p.id, set())]
        best = pick_best_try(ts, timing_penalty)
        d = decide_policy(None if best is None else best["wer"], orig_wer.get(p.id))
        dec[p.id] = {"variant": best["tag"] if best and d["use"] == "converted" else None, "use": d["use"], "flag": d["flag"],
                     "wer_conv": None if best is None else best["wer"], "wer_orig": orig_wer.get(p.id), "n_tries": len(ts),
                     "why": d["flag"] or "converted (cloned voice)"}
    return dec


def make_loader(dec_map: dict):
    """loader for v2v_mix.assemble: converted -> v2v2/aligned/<tag>/<id>.wav, original -> the enhanced original cut v2v2/in/<ORIG_CAND>/<id>.wav"""
    def load(p: V.Piece, d: dict):
        v = dec_map[p.id]["variant"]
        if v:
            x, _ = sf.read(str(V2 / "aligned" / v / f"{p.id}.wav"), dtype="float32")
            return x, f"converted ({v}, {tries_settings(p.id, v)})", True
        f = IN_DIR / ORIG_CAND / f"{p.id}.wav"
        x, _ = sf.read(str(f), dtype="float32")
        x = V.highpass(x, SR)
        return x, "original (enhanced: beamformed 3 phones + MossFormer2_SE_48K)", False
    return load


_TRY_CACHE: dict = {}


def tries_settings(pid: str, tag: str) -> str:
    if not _TRY_CACHE:
        _TRY_CACHE.update(load_tries())
    t = next((x for x in _TRY_CACHE.get(pid, []) if x["tag"] == tag), None)
    return "" if t is None else f"{t['cand']}, settings {t['settings']}, seed {t['seed']}"


BED_BELOW_SPEECH_DB = 38.0      # the room-tone bed under the cloned speech (V1's mix: 26): measured on the whole-class ASR, a louder bed costs about 0.03 WER on the cloned voice


def assemble_full(pieces: list[V.Piece], kept: list[dict], dec: dict, out_wav: Path, stems_dir: Path, report_path: Path, bed_below_db: float = BED_BELOW_SPEECH_DB) -> dict:
    from .. import film_recap_audio as fa
    from . import v2v_mix as M
    return M.assemble(pieces, kept, dec, orig_path=fa.AUDIO / "class_mix.wav", out_wav=out_wav, stems_dir=stems_dir, report_path=report_path, loader=make_loader(dec),
                      bed_below_db=bed_below_db)


# ------------------------------------------------------------------------------------------------ splicing the right words of several tries (v2v_splice)


def splice_piece(p: V.Piece, tries: list[dict], asr_by_tag: dict[str, dict], max_tries: int = 10, switch_cost: float | None = None) -> dict | None:
    """a spliced audio for piece `p` from its tries (best `max_tries` by WER that have word-timed ASR); returns {audio, path, changes, wer_expected, tags} or None
    when there is nothing to gain (a single try, or the best path is a single try)"""
    from . import v2v_splice as SP
    from .v2v_check import speech_env_db
    cand = [t for t in sorted([t for t in tries if t.get("wer") is not None], key=lambda t: t["wer"])
            if p.id in asr_by_tag.get(t["tag"], {}) and asr_by_tag[t["tag"]][p.id].get("words")][:max_tries]
    if len(cand) < 2:
        return None
    auds, envs, lv, oks, spans = [], [], [], [], []
    ref = None
    for t in cand:
        x, _ = sf.read(str(V2 / "aligned" / t["tag"] / f"{p.id}.wav"), dtype="float32")
        e = speech_env_db(x, SR)
        auds.append(x)
        envs.append(e)
        lv.append(float(np.percentile(e, 90)))
        ref_t, ok, sp = SP.word_status(p.text, asr_by_tag[t["tag"]][p.id]["words"])
        oks.append(ok)
        spans.append(sp)
        ref = ref_t
    n = len(ref)
    if n < 2:
        return None
    for i in range(1, len(auds)):                                  # level-match every try to the best one (speech level P90 of the envelope)
        auds[i] = auds[i] * 10 ** ((lv[0] - lv[i]) / 20.0)
        envs[i] = envs[i] + (lv[0] - lv[i])
        lv[i] = lv[0]
    L = min(len(a) for a in auds)
    auds = [a[:L] for a in auds]
    cuts = {}
    for a in range(len(cand)):
        for b in range(len(cand)):
            if a != b:
                for w in range(n - 1):
                    c = SP.cut_between(spans[a], spans[b], w, envs[a], envs[b], lv[a], lv[b])
                    if c is not None:
                        cuts[(a, b, w)] = c
    path, cost = SP.choose_path(oks, cuts, n, **({} if switch_cost is None else {"switch_cost": switch_cost}))
    if len(set(path)) == 1:
        return None
    y, changes = SP.render(auds, path, cuts, SR)
    return {"audio": y, "path": path, "changes": changes, "tags": [t["tag"] for t in cand], "words_wrong_expected": int(round(cost - SP.SWITCH_COST * len(changes))),
            "n_words": n}


def splice_all(pieces: list[V.Piece], tries: dict, tag: str = "sp", target: float = TARGET_WER, load_asr=None) -> dict:
    """splice every piece whose best try is above `target`; writes v2v2/aligned/<tag>/<id>.wav and returns {id: info}"""
    tags = sorted({t["tag"] for v in tries.values() for t in v})
    asr_by_tag = {tg: read_asr_set(f"aligned__{tg}") for tg in tags if (ASR_DIR / f"aligned__{tg}.json").exists()}
    out = {}
    for p in pieces:
        ts = tries.get(p.id, [])
        best = pick_best_try(ts)
        if best is None or best["wer"] <= target:
            continue
        r = splice_piece(p, ts, asr_by_tag)
        if r is None:
            continue
        f = V2 / "aligned" / tag / f"{p.id}.wav"
        f.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(f), np.clip(r.pop("audio"), -1, 1), SR, subtype="PCM_16")
        out[p.id] = r
    (V2 / f"splice_{tag}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


# ------------------------------------------------------------------------------------------------ voice identity (ECAPA vs the voices' own training samples)


def reference_embeddings() -> dict[str, np.ndarray]:
    """mean ECAPA embedding of each teacher's own training sample (from the V1 gate runs that embedded the reference clips)"""
    from . import v2v_gate as G
    for run in ("g0", "g1", "g2", "g3", "g4"):
        try:
            emb = G.read_embeddings(run)
        except OSError:
            continue
        refs = {n: np.mean([emb[k] for k in emb if k.startswith(f"ref_{n}_")], axis=0) for n in V.VOICE_NAMES if any(k.startswith(f"ref_{n}_") for k in emb)}
        if len(refs) == len(V.VOICE_NAMES):
            return refs
    raise V.V2VError("no reference embeddings of the two voices in the V1 gate runs")


def read_embeddings_set(name: str) -> dict[str, np.ndarray]:
    z = np.load(ASR_DIR / f"{name}.npz")
    return {str(k): v for k, v in zip(z["keys"], z["ecapa"])}


def identity_failures(pieces: list[V.Piece], tries: dict, tags: list[str]) -> dict[str, set]:
    """{piece id: {tags whose embedding is not nearer to its own teacher's reference by VOICE_MARGIN}} for the tags that have an embedding set"""
    from . import v2v_gate as G
    refs = reference_embeddings()
    bad: dict[str, set] = {}
    P = {p.id: p for p in pieces}
    for tg in tags:
        f = ASR_DIR / f"aligned__{tg}.npz"
        if not f.exists():
            continue
        emb = read_embeddings_set(f"aligned__{tg}")
        for pid, e in emb.items():
            if pid in P:
                vc = G.voice_check(P[pid].speaker, e, refs)
                if not vc["ok"]:
                    bad.setdefault(pid, set()).add(tg)
    return bad


# ------------------------------------------------------------------------------------------------ whole-class ASR of the finished mixes


def full_mix_wer(names_paths: dict[str, Path], open_end: float = 9.67) -> dict:
    """whole-class ASR (faster-whisper large-v3, no prompt, on the pod) of each mix (16 kHz mono mean of the channels); WER vs the transcript's teacher words
    (t1 > 0), overall and after `open_end`. Returns {name: {wer, wer_after_opening, sub, ins, del, n_ref, n_hyp, text}}"""
    from scipy import signal as sg
    tdir = V2 / "full"
    for nm, pth in names_paths.items():
        x, sr = sf.read(str(pth), dtype="float32", always_2d=True)
        x = x.mean(axis=1)
        g = math.gcd(16000, sr)
        y = sg.resample_poly(x, 16000 // g, sr // g).astype(np.float32)
        (tdir).mkdir(exist_ok=True)
        sf.write(str(tdir / f"{nm}.wav"), y, 16000, subtype="PCM_16")
    import tarfile
    from . import v2v_pod as P
    h = P.handle()
    tar = V2 / "full_push.tar"
    with tarfile.open(tar, "w") as t:
        for nm in names_paths:
            t.add(tdir / f"{nm}.wav", arcname=f"full/{nm}.wav")
    h.put(tar, "/root/full_push.tar", timeout=1800)
    h.run("cd /root/data && tar -xf /root/full_push.tar", timeout=600, echo=False)
    pod_asr(["full"], force=True)
    pull_asr(["full"])
    asr = read_asr_set("full")
    ws = [w for w in V.transcript_words(V.read_transcript()) if w["t1"] > 0]
    out = {}
    for nm in names_paths:
        it = asr[nm]
        hyp = " ".join(it["text"].split())
        ref_all = V.norm_words(" ".join(w["text"] for w in ws))
        c = V.edit_counts(ref_all, V.norm_words(hyp))
        ref_b = V.norm_words(" ".join(w["text"] for w in ws if w["t0"] >= open_end))
        hyp_b = V.norm_words(" ".join(w for w, s, e in it["words"] if s >= open_end))
        cb = V.edit_counts(ref_b, hyp_b)
        out[nm] = {"wer": round(c["wer"], 4), "sub": c["sub"], "ins": c["ins"], "del": c["del"], "n_ref": c["n_ref"], "n_hyp": len(V.norm_words(hyp)),
                   "wer_after_opening": round(cb["wer"], 4), "n_ref_after_opening": cb["n_ref"]}
    return out


# ------------------------------------------------------------------------------------------------ word timing of a try against the transcript (ASR word times)


def word_timing_errors(p: V.Piece, asr_words: list, words: list[dict]) -> list[dict]:
    """for each correctly recognised word of the piece: [{text, ref_t, asr_t, err_ms}] with ref_t = the transcript start (piece time), asr_t = the ASR word start
    (faster-whisper word timestamps, accurate to about +-50 ms). `words` = the transcript words of the piece (kind word, with t0)"""
    from . import v2v_splice as SP
    ref, ok, spans = SP.word_status(p.text, asr_words)
    ref_t = []
    for w in words:
        n_tok = len(V.norm_words(w["text"]))
        ref_t += [w["t0"] - p.t0] * n_tok
    out = []
    if len(ref_t) != len(ref):
        return out
    for i, (okk, sp) in enumerate(zip(ok, spans)):
        if okk and sp is not None:
            out.append({"text": ref[i], "ref_t": round(ref_t[i], 3), "asr_t": sp[0], "err_ms": round((sp[0] - ref_t[i]) * 1000, 1)})
    return out


def piece_words(p: V.Piece, all_words: list[dict]) -> list[dict]:
    return [w for w in all_words if w["speaker"] == p.speaker and w["t0"] >= p.w0 - 1e-3 and w["t1"] <= p.w1 + 1e-3]


# ------------------------------------------------------------------------------------------------ per-piece WER measured IN the whole-class ASR of a mix


def in_context_piece_wer(pieces: list[V.Piece], asr_item: dict, delta: float = 0.0, pad: float = 0.2) -> dict[str, dict]:
    """per piece the error counts of the words the whole-file ASR placed inside [w0 - pad + delta, w1 + pad + delta] (hyp word midpoints; delta absorbs the ASR's
    own timestamp bias) against the piece's words"""
    ws = asr_item["words"]
    out = {}
    for p in pieces:
        sel = [w for w, s, e in ws if p.w0 - pad + delta <= (s + e) / 2 <= p.w1 + pad + delta]
        out[p.id] = V.edit_counts(V.norm_words(p.text), V.norm_words(" ".join(sel)))
    return out


def calibrate_delta(pieces: list[V.Piece], asr_item: dict, grid=None) -> tuple[float, float]:
    """the delta (s) that minimises the total per-piece error of one mix (use the enhanced-original mix); returns (delta, total error rate)"""
    best = (0.0, 9.0)
    n = sum(len(V.norm_words(p.text)) for p in pieces)
    for d in (grid if grid is not None else np.arange(-0.1, 0.45, 0.05)):
        r = in_context_piece_wer(pieces, asr_item, float(d))
        e = sum(c["sub"] + c["ins"] + c["del"] for c in r.values()) / n
        if e < best[1]:
            best = (float(d), e)
    return best
