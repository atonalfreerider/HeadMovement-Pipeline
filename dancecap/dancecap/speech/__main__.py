"""python -m dancecap.speech <command>   (see the module docstrings; all heavy work runs on a farm pod)

  glossary                 write work/<take>/speech/glossary.json
  audio [--source DIR]     extract 16 kHz / 48 kHz wav + m4a of all sources (ffmpeg, local, seconds)
  pod up|sh|put|get|status|extend|down     the one GPU pod of the speech job (podsession.py)
  push-audio               upload the 16 kHz wavs and the code archive to the hm-farm volume (via the pod)
  build-env                build the speech environment on the pod (speech_env.sh build), stored on the volume
  submit [--skip a,b]      queue the speech job; start-worker starts the farm worker on the pod
  job status|log STEP|pull
  clock                    print the clock table (sync.json / owner offsets)
  cluster | identity | events | report | review     local post-processing of the pulled raw files
  fuse | fuse-eval         S2: the fused transcript (consensus_v0.json), disagreement batches, objective agreement metrics (local, numpy)
  v2v <command>            V1: voice-to-voice class audio (plan | sources | prepare | convert | align | gate-* | assemble ...; see speech/v2v.py)
  final [--no-clips]       S4: the FINAL transcript (adjudicated spans merged), srt / vtt / txt / captions.json, chapters, teaching events, jokes, review.html (local)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import SPEECH, SOURCES


def _cfg_path() -> Path:
    return SPEECH / "config.json"


def source_dir(arg: str | None) -> Path:
    cfg = {}
    if _cfg_path().exists():
        cfg = json.loads(_cfg_path().read_text(encoding="utf-8"))
    d = arg or cfg.get("source_dir")
    if not d:
        raise SystemExit("pass --source <recap folder> once (it is remembered in work/<take>/speech/config.json)")
    SPEECH.mkdir(parents=True, exist_ok=True)
    cfg["source_dir"] = str(d)
    _cfg_path().write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    return Path(d)


def progress_print(rel, done, size) -> None:
    if size and (done == size or done % (8 << 20) < (1 << 20)):
        print(f"[upload] {rel}: {done / 1e6:.1f} / {size / 1e6:.1f} MB", flush=True)


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "v2v":                       # V1: voice-to-voice class audio (v2v_cli.py)
        from . import v2v_cli
        return v2v_cli.main(args[1:])
    ap = argparse.ArgumentParser(prog="python -m dancecap.speech", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("glossary")
    a_ = sub.add_parser("audio")
    a_.add_argument("--source")
    a_.add_argument("--force", action="store_true")
    p = sub.add_parser("pod")
    p.add_argument("action", choices=["up", "sh", "put", "get", "sync", "status", "extend", "down"])
    p.add_argument("args", nargs="*")
    p.add_argument("--gpu")
    p.add_argument("--minutes", type=float, default=100)
    p.add_argument("--hard-minutes", type=float, default=130)
    p.add_argument("--max-cost", type=float, default=2.0)
    p.add_argument("--timeout", type=float, default=900)
    sub.add_parser("push-audio")
    sub.add_parser("build-env")
    s = sub.add_parser("submit")
    s.add_argument("--skip", default="")
    s.add_argument("--id", default=None)
    s.add_argument("--only", default="", help="comma list of steps to run (env is always added)")
    sub.add_parser("start-worker")
    j = sub.add_parser("job")
    j.add_argument("action", choices=["status", "log", "pull"])
    j.add_argument("args", nargs="*")
    j.add_argument("--id", default=None, help="job id (default speech-recap-1)")
    sub.add_parser("clock")
    sub.add_parser("events", help="laughter / applause events on the master clock -> events_master.json")
    sub.add_parser("report", help="engine statistics, agreement, language, quality -> report.json")
    sub.add_parser("diarize", help="global voices, names, interjections, utterances, jokes -> diarization_global.json")
    sub.add_parser("review", help="the 30-clip who-is-who list (clips, json, md, html)")
    sub.add_parser("plots", help="voice timeline and embedding-space images")
    sub.add_parser("fuse", help="S2: fused transcript (consensus_v0.json) + disagreement batches (disagreements/batch_NN.json); local, no pod")
    fe = sub.add_parser("fuse-eval", help="S2: leave-one-source-out agreement, ablations, reliability curve -> fusion_metrics.json (about 20 min, local)")
    fe.add_argument("--only", default="", help="comma list of ablations to run")
    fin = sub.add_parser("final", help="S4: transcript_final.json + exports + chapters + teaching events + jokes + review page + evaluation (local, seconds)")
    fin.add_argument("--no-clips", action="store_true", help="do not cut the review audio clips (ffmpeg)")
    a = ap.parse_args(argv)

    if a.cmd == "glossary":
        from . import glossary
        path = glossary.write()
        print(path)
    elif a.cmd == "audio":
        from . import audio
        audio.extract_all(source_dir(a.source), force=a.force)
    elif a.cmd in ("events", "report", "diarize", "review"):
        from .sources import load_clocks
        cfg = json.loads(_cfg_path().read_text(encoding="utf-8")) if _cfg_path().exists() else {}
        clocks = load_clocks(recap_dir=cfg.get("source_dir"))
        if a.cmd == "events":
            from . import events
            doc = events.run(SPEECH / "raw", clocks)
            doc["music_presence"] = events.music_presence(SPEECH / "raw", clocks)
            (SPEECH / "events_master.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")
            print(json.dumps({k: doc[k] for k in ("counts", "seconds", "music_presence")}))
        elif a.cmd == "report":
            from . import report
            doc = report.build(SPEECH / "raw", clocks)
            (SPEECH / "report.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")
            print("report.json:", len(doc["engines"]), "phones")
        elif a.cmd == "diarize":
            from . import diarize
            doc = diarize.run(recap_dir=cfg.get("source_dir"))
            print(json.dumps({"voices": [{k: v.get(k) for k in ("voice", "name", "seconds")} for v in doc["voices"]], "naming": {k: doc["naming"].get(k) for k in ("assignment", "naming_confidence", "evidence")},
                              "turns": len(doc["turns"]), "interjections": len(doc["interjections"]), "jokes": len(doc["jokes"]), "emoji_tokens": doc["emoji_tokens"]}, indent=1, default=str))
        else:
            from . import review
            clips = review.run()
            print(len(clips), "clips in", review.REVIEW)
    elif a.cmd == "fuse":
        from . import fuse_run
        print(json.dumps(fuse_run.run_fuse(), indent=1, default=str))
    elif a.cmd == "final":
        from . import final_run
        print(json.dumps(final_run.run_final(clips=not a.no_clips), indent=1, default=str, ensure_ascii=False))
    elif a.cmd == "fuse-eval":
        from . import fuse_run
        res = fuse_run.run_eval([x for x in a.only.split(",") if x] or None, log=lambda m: print(m, flush=True))
        print(json.dumps({"loso": {k: res["loso"][k] for k in ("fused_pwer_mean", "best_single_engine", "fused_beats_best_engine_mean")}, "seconds": res["seconds"]}, indent=1))
    elif a.cmd == "plots":
        from . import plots
        for p in plots.run():
            print(p)
    elif a.cmd == "clock":
        from .sources import load_clocks
        cfg = json.loads(_cfg_path().read_text(encoding="utf-8")) if _cfg_path().exists() else {}
        for sid, c in load_clocks(recap_dir=cfg.get("source_dir")).items():
            print(sid, SOURCES[sid], "-" if c is None else f"slope {c.slope:.8f} offset {c.offset:+.4f} s ({c.origin}{', provisional' if c.provisional else ''})")
    elif a.cmd == "pod":
        from . import podsession as ps
        if a.action == "up":
            print(json.dumps(ps.up(a.gpu, a.minutes, a.hard_minutes, a.max_cost), indent=1))
        elif a.action == "sh":
            return ps.sh(" ".join(a.args), timeout=a.timeout)
        elif a.action == "status":
            print(json.dumps(ps.status(), indent=1))
        elif a.action == "extend":
            print(ps.extend(a.minutes))
        elif a.action == "down":
            print(json.dumps(ps.down(), indent=1))
        elif a.action == "sync":      # dev loop: push this package + glossary and unpack it in the pod's workspace
            from . import jobs
            pod = ps.handle()
            pod.put(jobs.code_archive(), "/tmp/speech_code.tar")
            return pod.run("tar -xf /tmp/speech_code.tar -C /opt/hm/ws && echo synced", check=False, timeout=120)
        elif a.action in ("put", "get"):
            pod = ps.handle()
            if a.action == "put":
                pod.put(Path(a.args[0]), a.args[1])
            else:
                pod.get(a.args[0], Path(a.args[1]))
    elif a.cmd == "push-audio":
        from . import jobs
        lnk = jobs.link()
        items = [(f, f"{jobs.VOL_AUDIO}/{f.name}") for f in jobs.audio_files()]
        print(json.dumps(lnk.push_files(items, progress=progress_print), indent=1, default=str))
    elif a.cmd == "build-env":
        from . import jobs, podsession as ps
        lnk = jobs.link()
        script = (Path(__file__).parent / "bootstrap" / "speech_env.sh").read_text(encoding="utf-8").replace("\r\n", "\n")
        lnk.write_text("speech/bin/speech_env.sh", script)
        pod = ps.handle()
        return pod.run("bash /workspace/hm/speech/bin/speech_env.sh build", check=False, timeout=3600)
    elif a.cmd == "submit":
        from . import jobs
        lnk = jobs.link()
        spec = jobs.job_spec(a.id or jobs.JOB_ID, skip={x for x in a.skip.split(",") if x}, only={x for x in a.only.split(",") if x} or None)
        print(json.dumps(jobs.submit(lnk, spec), indent=1, default=str))
    elif a.cmd == "start-worker":
        from . import jobs, podsession as ps
        pod = ps.handle()
        return jobs.start_worker(pod, price_hr=ps.load().get("price_hr", 0))
    elif a.cmd == "job":
        from . import jobs
        lnk = jobs.link()
        if a.action == "status":
            print(json.dumps(jobs.status(lnk, a.id or jobs.JOB_ID), indent=1, default=str))
        elif a.action == "log":
            print(jobs.tail_log(lnk, a.args[0] if a.args else "vad", int(a.args[1]) if len(a.args) > 1 else 40, a.id or jobs.JOB_ID))
        elif a.action == "pull":
            print(json.dumps(jobs.pull(lnk, a.id or jobs.JOB_ID), indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
