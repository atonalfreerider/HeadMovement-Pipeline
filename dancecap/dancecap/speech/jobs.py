"""The speech job on the hm-farm volume: spec, upload, worker start, status, pull.

A farm job (schema dancecap.farm.job/1, kind `asr` = GPU) whose steps run the pod-side stages of dancecap.speech.pod.run for ALL
sources, in order, each step idempotent per source and snapshotted to the volume (`save`), so a restarted worker (any pod) resumes after
the last finished step:

    env        speech_env.sh unpack         the farm env + NeMo / speechbrain / transformers (built once, speech/CURRENT on the volume)
    check      versions, GPU, which engines import
    vad        Silero regions + frame probabilities
    whisper_large-v3        passes auto, en, pt + language id per region
    parakeet                parakeet-tdt-0.6b-v3
    canary                  canary-1b-v2 (language per segment from the whisper region ids)
    embed                   ECAPA + TitaNet embeddings, F0, level
    whisper_large-v3-turbo  passes auto, en
    events                  AST laughter / applause
    sortformer              overlap-aware diarization windows
    quality                 per-source statistics
    segembed                ECAPA + TitaNet embedding of every Sortformer segment (sharper turns, short interjections)
    regions                 (only on request) targeted multi-setting decoding of hard passages

Inputs: the seven 16 kHz wavs at src/larissa_kadu_recap/speech_audio/ (uploaded once, hash-diffed) and `speech_code.tar` (this package
and the glossary, extracted into the job workspace so the job does not depend on the volume's code bundle). Outputs: the raw directory
dancecap/work/larissa_kadu_recap/speech/raw/ (a few MB), pulled with volume.pull_result into the local work tree.
"""
from __future__ import annotations

import hashlib
import json
import tarfile
import time
from pathlib import Path

from ..farm import volume as V
from ..take import PACKAGE_ROOT, WORKSPACE
from . import SPEECH, TAKE

SCHEMA = "dancecap.farm.job/1"
VOL_AUDIO = "src/larissa_kadu_recap/speech_audio"
SPEECH_PY = "/opt/hm/env_speech/bin/python"
RAW_WS = f"dancecap/work/{TAKE}/speech/raw"
JOB_ID = "speech-recap-1"
PKG_DIR = Path(__file__).resolve().parent


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def code_archive(out: Path | None = None) -> Path:
    """speech_code.tar: this package (sources and bootstrap scripts only, LF line endings) + the glossary (+ regions.json when it exists), at workspace-relative paths."""
    out = out or SPEECH / "pod_inputs" / "speech_code.tar"
    out.parent.mkdir(parents=True, exist_ok=True)
    gl = SPEECH / "glossary.json"
    if not gl.exists():
        from . import glossary
        glossary.write()
    tmp = out.with_name(out.name + ".part")
    with tarfile.open(tmp, "w") as tar:
        for f in sorted(PKG_DIR.rglob("*")):
            if not f.is_file() or "__pycache__" in f.parts or f.suffix in (".pyc", ".pyo"):
                continue
            rel = "dancecap/dancecap/speech/" + f.relative_to(PKG_DIR).as_posix()
            data = f.read_bytes()
            if f.suffix in (".py", ".sh", ".md", ".json"):
                data = data.replace(b"\r\n", b"\n")
            info = tarfile.TarInfo(rel)
            info.size = len(data)
            info.mtime = int(f.stat().st_mtime)
            info.mode = 0o755 if f.suffix == ".sh" else 0o644
            import io
            tar.addfile(info, io.BytesIO(data))
        tar.add(gl, arcname=f"dancecap/work/{TAKE}/speech/glossary.json")
        rg = SPEECH / "regions.json"
        if rg.exists():
            tar.add(rg, arcname=f"dancecap/work/{TAKE}/speech/regions.json")
    tmp.replace(out)
    return out


def audio_files() -> list[Path]:
    d = SPEECH / "audio"
    files = sorted(d.glob("[0-9][0-9]-Recap-*.16k.wav"))
    if len(files) != 7:
        raise RuntimeError(f"expected 7 16 kHz wavs in {d}, found {len(files)}: run `python -m dancecap.speech audio`")
    return files


def steps(skip: set[str] | None = None, whisper_passes: str = "auto,en,pt", only: set[str] | None = None) -> list[dict]:
    py = SPEECH_PY
    run = f"{py} -m dancecap.speech.pod.run"
    save = [RAW_WS + "/"]

    def st(name, args, minutes, optional=True, env=None):
        d = {"name": name, "cmd": f"{run} {args}", "timeout_min": minutes, "optional": optional, "save": save}
        if env:
            d["env"] = env
        return d

    all_steps = [
        {"name": "env", "cmd": "bash dancecap/speech/bootstrap/speech_env.sh unpack", "timeout_min": 15},   # cwd = <ws>/dancecap
        st("check", "check", 10),
        st("vad", "vad", 15, optional=False),
        st("whisper_large-v3", f"whisper --model large-v3 --passes {whisper_passes}", 40, optional=False),
        st("parakeet", "parakeet", 30),
        st("canary", "canary", 30),
        st("embed", "embed", 20),
        st("whisper_large-v3-turbo", "whisper --model large-v3-turbo --passes auto,en", 25),
        st("events", "events", 20),
        st("sortformer", "sortformer", 30),
        st("quality", "quality", 10),
        st("segembed", "segembed", 20),
        {"name": "regions", "cmd": f"{run} regions", "timeout_min": 40, "optional": True,
         "save": save + [RAW_WS.replace("/raw", "/raw_regions") + "/"]},
    ]
    skip = skip or set()
    if only:
        return [s for s in all_steps if s["name"] in only or s["name"] == "env"]
    return [s for s in all_steps if s["name"] not in skip and (s["name"] != "regions" or "regions" in (only or set()))]


def job_spec(job_id: str = JOB_ID, skip: set[str] | None = None, max_minutes: float = 90, priority: float = 500,
             only: set[str] | None = None) -> dict:
    """The job spec; inputs carry `local` paths that volume.submit_job uploads (hash-diffed) and strips."""
    inputs = [{"path": f"{VOL_AUDIO}/{f.name}", "local": str(f)} for f in audio_files()]
    inputs.append({"path": "speech_code.tar", "archive": True, "local": str(code_archive())})
    return {"schema": SCHEMA, "id": job_id, "dance": TAKE, "kind": "asr", "chunk": {"id": job_id, "kind": "speech"},
            "world_id": None, "world_sha256": None, "code": {}, "after": [], "inputs": inputs, "steps": steps(skip, only=only),
            "outputs": [{"path": RAW_WS + "/"}, {"path": RAW_WS.replace("/raw", "/raw_regions") + "/", "optional": True}], "max_minutes": max_minutes, "priority": priority, "needs_gpu": True,
            "retries": 2, "env": {"PYTHONPATH": "{WS}/dancecap"}}


def link(pod=None) -> V.Link:
    from . import podsession
    return V.Link(pod or podsession.handle())


def submit(lnk: V.Link, spec: dict) -> dict:
    """Upload the inputs the volume does not hold yet and queue the job (queue/<id>.json is written last)."""
    spec = json.loads(json.dumps(spec))
    def progress(rel, done, size):
        if size and (done == size or done % (8 << 20) < (1 << 20)):
            print(f"[upload] {rel}: {done / 1e6:.1f} / {size / 1e6:.1f} MB", flush=True)

    return V.submit_job(lnk, spec, progress=progress)


def start_worker(pod, job_kinds: str = "asr", price_hr: float = 0.0, idle_exit_min: float = 3) -> int:
    """Run the farm worker on a pod in gateway mode (pod_entry.sh with HM_AUTOSTART=0 prepares nothing): prepare env / code / libs /
    links exactly as pod_entry would, then start the worker detached with `--once`."""
    cmd = f"""
set -u
HM=/workspace/hm; WS=/opt/hm/ws; PID={pod.id}
mkdir -p $HM/pods/$PID
bash $HM/bin/pod_prepare.sh env >> $HM/pods/$PID/prepare.log 2>&1 &
P1=$!
bash $HM/bin/pod_prepare.sh libs >> $HM/pods/$PID/prepare.log 2>&1 &
P2=$!
bash $HM/bin/pod_prepare.sh code links >> $HM/pods/$PID/prepare.log 2>&1
wait $P1; wait $P2
export HM_ROOT=$HM HM_WS=$WS PATH=/opt/hm/env/bin:$PATH PYTHONPATH=$WS/dancecap DANCECAP_POD=1 PYTHONUNBUFFERED=1 \\
  TORCH_HOME=$HM/weights/torch_home HF_HOME=$HM/weights/hf MPLBACKEND=Agg HM_PRICE_HR={price_hr}
cd $WS/dancecap
nohup python -u -m dancecap.farm.worker --volume $HM --kinds {job_kinds} --once --idle-exit-min {idle_exit_min} --pod-id $PID \\
  >> $HM/pods/$PID/worker.log 2>&1 < /dev/null &
sleep 3; pgrep -f dancecap.farm.worker > /dev/null && echo WORKER_STARTED || echo WORKER_NOT_RUNNING
"""
    return pod.run(cmd, check=False, timeout=300)


def status(lnk: V.Link, jid: str = JOB_ID) -> dict:
    st = V.job_status(lnk, jid)
    hb = st.get("heartbeat") or {}
    done, failed = st.get("done"), st.get("failed")
    return {"id": jid, "state": "done" if done else "failed" if failed else "running" if hb else "queued",
            "step": hb.get("step"), "step_i": hb.get("step_i"), "steps_n": hb.get("steps_n"),
            "heartbeat_age_s": round(time.time() - hb["t"], 1) if hb.get("t") else None,
            "progress": hb.get("progress"), "gpu": (hb.get("gpu") or {}).get("name"),
            "done": {k: done.get(k) for k in ("status", "seconds", "cost_usd", "out_files", "out_bytes")} if done else None,
            "failed": {k: failed.get(k) for k in ("status", "error", "seconds")} if failed else None,
            "steps": (done or failed or {}).get("steps")}


def tail_log(lnk: V.Link, step: str, lines: int = 30, jid: str = JOB_ID) -> str:
    return lnk.sh(f"tail -n {lines} {lnk.abs(f'chunks/{jid}/log/{step}.log')} 2>/dev/null || true", 60)


def pull(lnk: V.Link, jid: str = JOB_ID) -> dict:
    return V.pull_result(lnk, jid, WORKSPACE)
