"""CLI of the voice-to-voice class audio (agent V1): `python -m dancecap.speech v2v <command>` (see v2v.py for the stages).

    plan [--no-sources]       transcript_final.json -> v2v/plan.json (pieces, kept-original stretches, cost estimate) + the phone per piece + the prepared audio
    refs                      download the two voices' own training samples (voice-check references) into v2v/ref/
    status                    pieces, credits used (ledger), variants converted, gate runs
    convert --variant NAME [--only p001,p002] [--budget CREDITS]
    align --variant NAME
    gate-build / gate-run / gate-score --run RUN
    assemble [--decisions FILE]  film/audio/class_mix_v2v.wav + stems/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import v2v as V

VARIANTS = {
    "A": V.Variant("A"),
    "B": V.Variant("B", settings={"stability": 0.8, "similarity_boost": 0.9, "style": 0.0, "use_speaker_boost": True}),
    "C": V.Variant("C", remove_background_noise=True),
    "D": V.Variant("D", settings={"stability": 0.8, "similarity_boost": 0.9, "style": 0.0, "use_speaker_boost": True}, remove_background_noise=True),
    "E": V.Variant("E", model="eleven_english_sts_v2"),
    "F": V.Variant("F", pre="pre16", remove_background_noise=True),
    "G": V.Variant("G", pre="iso"),
    "H": V.Variant("H", pre="iso", settings={"stability": 0.8, "similarity_boost": 0.9, "style": 0.0, "use_speaker_boost": True}),
    # second chances for pieces whose first conversion failed the gate: another stability / similarity, the SECOND-best phone
    "S": V.Variant("S", settings={"stability": 0.35, "similarity_boost": 0.6, "style": 0.0, "use_speaker_boost": True}, remove_background_noise=True),
    "T": V.Variant("T", settings={"stability": 1.0, "similarity_boost": 1.0, "style": 0.0, "use_speaker_boost": True}, remove_background_noise=True),
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.speech v2v", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--no-sources", action="store_true")
    sub.add_parser("refs")
    sub.add_parser("status")
    c = sub.add_parser("convert")
    c.add_argument("--variant", required=True)
    c.add_argument("--only")
    c.add_argument("--budget", type=int)
    a_ = sub.add_parser("align")
    a_.add_argument("--variant", required=True)
    d_ = sub.add_parser("decide", help="gate results (pod ASR runs) -> decisions.json + gate_table.json")
    d_.add_argument("--runs", required=True, help="comma list of gate runs, e.g. g2,g3")
    d_.add_argument("--variants", required=True, help="comma list in order of preference, e.g. D,D2")
    d_.add_argument("--cams", help="JSON {variant: {piece: phone}} for variants that used another phone")
    m_ = sub.add_parser("assemble", help="film/audio/class_mix_v2v.wav + stems/ from decisions.json")
    m_.add_argument("--decisions")
    m_.add_argument("--out")
    a = ap.parse_args(argv)
    duration = 360.476783
    if a.cmd == "plan":
        tr = V.read_transcript()
        ws = V.transcript_words(tr)
        pieces, kept = V.plan_pieces(ws, open_end=V.opening_end(ws), duration=duration)
        if (V.V2V / "plan.json").exists():
            pieces = V.stabilise_ids(pieces, V.read_plan()[0])      # unchanged stretches keep their id (and so their conversions)
        if not a.no_sources:
            table = V.choose_sources(pieces, ws, duration)
            (V.V2V / "sources.json").write_text(json.dumps(table, indent=1), encoding="utf-8")
        V.write_plan(pieces, kept)
        if not a.no_sources:
            info = V.prepare_all(pieces, duration)
            (V.V2V / "prepare.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
        print(json.dumps(V.cost_estimate(pieces)), f"{len(pieces)} pieces, {len(kept)} kept-original stretches")
    elif a.cmd == "refs":
        from . import v2v_gate as G
        print(G.fetch_references())
    elif a.cmd == "status":
        pieces, kept = V.read_plan()
        print(f"{len(pieces)} pieces, credits used so far (ledger): {V.ledger_credits()}")
    elif a.cmd == "convert":
        pieces, _ = V.read_plan()
        only = set(a.only.split(",")) if a.only else None
        print(V.convert_all(pieces, VARIANTS[a.variant], only=only, budget_credits=a.budget))
    elif a.cmd == "align":
        pieces, _ = V.read_plan()
        print(len(V.align_all(pieces, VARIANTS[a.variant])), "aligned")
    elif a.cmd == "decide":
        import numpy as np
        from . import v2v_gate as G
        pieces, _ = V.read_plan()
        asr, emb = {}, {}
        for r in a.runs.split(","):
            asr.update(G.read_asr(r))
            emb.update(G.read_embeddings(r))
        dec, rows = G.decide_all(pieces, asr, emb, a.variants.split(","), cams=json.loads(a.cams) if a.cams else None)
        (V.V2V / "decisions.json").write_text(json.dumps(dec, indent=1), encoding="utf-8")
        (V.V2V / "gate_table.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"{sum(1 for d in dec.values() if d['variant'])} of {len(dec)} pieces use a converted variant; the others keep the cleaned original")
    elif a.cmd == "assemble":
        from .. import film_recap_audio as fa
        from . import v2v_mix as M
        pieces, kept = V.read_plan()
        dec = M.load_decisions(a.decisions and Path(a.decisions))
        out = Path(a.out) if a.out else fa.AUDIO / fa.V2V_MIX
        M.assemble(pieces, kept, dec, orig_path=fa.AUDIO / "class_mix.wav", out_wav=out, stems_dir=out.parent / "stems",
                   report_path=out.with_suffix(".json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
