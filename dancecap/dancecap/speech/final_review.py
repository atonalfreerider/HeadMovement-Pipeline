"""S4: the human review page of the final transcript (one self-contained HTML file, local only) and the evaluation tables.

    speech/review.html            transcript by speaker with timestamps, low-confidence words highlighted, the spans to proofread first with a 6-12 s audio clip each,
                                  the jokes, and S1's 30 'who is who' clips; click a time to play the master audio from there
    speech/review/final_clips/    the clips of the spans (16 kHz mono wav cut from the phone each span points at; voices of real people: never commit or publish)
    speech/evaluation.json        final vs every source transcript (pseudo-WER), counts of changes, a model-based estimate of the expected accuracy

The page has no external files except the audio it plays (relative paths), no network and no libraries.
"""
from __future__ import annotations

import html
import json
import re
import subprocess
from pathlib import Path

from . import SOURCES, SPEECH
from .final_merge import SENTENCE_END, mmss

CLIPS = SPEECH / "review" / "final_clips"
LOW = 0.6
MID = 0.8


def _e(s) -> str:
    return html.escape("" if s is None else str(s), quote=True)


# ------------------------------------------------------------------------------------------------ clips


PHONE_ORDER = ["05", "01", "02", "00", "06", "03"]      # the phones nearest the teachers first (S1 / S2 loudness statistics), the late starter last


def wav_seconds(path: Path) -> float | None:
    import wave
    try:
        with wave.open(str(path), "rb") as w:
            return w.getnframes() / float(w.getframerate())
    except (OSError, EOFError, wave.Error):
        return None


def choose_phone(span: dict, clocks: dict, pad: float = 1.0) -> str | None:
    """The phone to cut a span from: the one `listen_to` names when it was recording the whole window, else the first phone of PHONE_ORDER that was (the listen_to choice of the
    fusion step ignores when a phone started: phone 03 starts at master 61 s). If none covers it all, the phone that covers most of it."""
    m0, m1 = span["start"] - pad, span["end"] + pad
    best, best_cov = None, -1.0
    lt = (span.get("listen_to") or {}).get("phone")
    for cam in ([lt] if lt else []) + [c for c in PHONE_ORDER if c != lt]:
        clk = clocks.get(cam)
        stem = Path(SOURCES.get(cam, "")).stem
        dur = wav_seconds(SPEECH / "audio" / f"{stem}.16k.wav") if cam in SOURCES else None
        if clk is None or dur is None:
            continue
        a, b = max(m0, clk.to_master(0.0)), min(m1, clk.to_master(dur))
        cov = max(0.0, b - a) / max(1e-6, m1 - m0)
        if cov >= 0.98:
            return cam
        if cov > best_cov:
            best, best_cov = cam, cov
    return best if best_cov > 0.3 else None


def cut_clip(span: dict, clocks: dict, out_dir: Path = CLIPS, pad: float = 1.0, ffmpeg: str = "ffmpeg", make: bool = True) -> Path | None:
    """The audio of one span (16 kHz wav, padded by `pad` s) from a phone that was recording it (choose_phone); an existing clip is reused; None when no phone / file / ffmpeg is
    available (or make is False and there is no clip)."""
    cam = choose_phone(span, clocks, pad)
    clk = clocks.get(cam) if cam else None
    if cam is None or clk is None:
        return None
    src = SPEECH / "audio" / f"{Path(SOURCES[cam]).stem}.16k.wav"
    m0, m1 = span["start"] - pad, span["end"] + pad
    t0 = max(0.0, (m0 - clk.offset) / clk.slope)
    dur = max(1.0, (m1 - max(m0, clk.to_master(0.0))) / clk.slope)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{span['id']}_{cam}_{max(m0, clk.to_master(0.0)):.1f}s.wav"
    if out.exists():
        return out
    if not make:
        return None
    try:
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-ss", f"{t0:.3f}", "-t", f"{dur:.3f}", "-i", str(src), "-ac", "1", "-ar", "16000", str(out)], check=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return out


# ------------------------------------------------------------------------------------------------ the page

CSS = """
:root{--bg:#fbfaf7;--fg:#1d1d1f;--mut:#6b6b70;--card:#fff;--line:#d9d6cf;--l:#c2410c;--k:#1d6fb8;--o:#6b7280;--low:#fde68a;--mid:#fef3c7;--own:#dc2626;--joke:#db2777}
@media (prefers-color-scheme:dark){:root{--bg:#16161a;--fg:#ececf0;--mut:#9a9aa3;--card:#1f1f25;--line:#34343c;--l:#fb923c;--k:#60a5fa;--o:#9ca3af;--low:#78580a;--mid:#4a3a0c;--own:#f87171;--joke:#f472b6}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,Segoe UI,Roboto,sans-serif}
header,main{max-width:1000px;margin:0 auto;padding:12px 16px}h1{font-size:22px;margin:8px 0}h2{font-size:18px;margin:28px 0 8px;border-bottom:1px solid var(--line);padding-bottom:4px}
small,.mut{color:var(--mut)}.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px;margin:8px 0}
nav a{margin-right:10px;white-space:nowrap}audio{width:100%;max-width:420px;height:34px}
.row{display:grid;grid-template-columns:64px 78px 1fr;gap:8px;padding:6px 0;border-bottom:1px solid var(--line)}
.t{cursor:pointer;color:var(--mut);font-variant-numeric:tabular-nums}.t:hover{text-decoration:underline}
.spk{font-weight:600}.spk.Larissa{color:var(--l)}.spk.Kadu{color:var(--k)}.spk.other{color:var(--o)}
.w{padding:0 1px;border-radius:3px}.q0{background:var(--low);outline:1px solid #b45309}.q1{background:var(--low)}.q2{background:var(--mid)}
.own{box-shadow:inset 0 -2px 0 var(--own)}.joke{color:var(--joke);font-weight:600}.emo{font-size:1.25em}
.chap{background:var(--card);border-left:4px solid var(--k);padding:6px 10px;margin:14px 0 4px;font-weight:600}
.chap small{font-weight:400}.badge{display:inline-block;border:1px solid var(--line);border-radius:10px;padding:0 8px;font-size:12px;margin-left:6px;color:var(--mut)}
.badge.own{border-color:var(--own);color:var(--own);box-shadow:none}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid var(--line);padding:5px 6px;vertical-align:top;text-align:left;font-size:13px}
.tw{overflow-x:auto}.legend span{margin-right:12px}.hide-ok .row.ok{display:none}label{cursor:pointer}
@media (max-width:640px){.row{grid-template-columns:52px 1fr}.row .spk{grid-column:2}.row .txt{grid-column:1/-1}}
"""

JS = """
(function(){
var a=document.getElementById('master');
function seek(t){if(!a||!a.src)return;try{a.currentTime=Math.max(0,t);a.play();}catch(e){}}
document.addEventListener('click',function(ev){var el=ev.target.closest('[data-t]');if(el){seek(parseFloat(el.getAttribute('data-t')));}});
var cb=document.getElementById('onlylow');if(cb){cb.addEventListener('change',function(){document.body.classList.toggle('hide-ok',cb.checked);});}
})();
"""


CLIP_RE = re.compile(r"review/(?:clips|final_clips)/[A-Za-z0-9_.\-]+\.wav")


def questions_html(text: str) -> str:
    """The owner's questions (work/<take>/speech/OWNER_QUESTIONS.md) as a card at the top of the page, with an audio player for every clip path the text names."""
    out = ['<h2 id="questions">Questions for you</h2><p class="mut">What only your ear can settle (the opening first). Each question names its clip.</p><div class="card">',
           f'<div style="white-space:pre-wrap">{_e(text)}</div>']
    seen: list[str] = []
    for m in CLIP_RE.finditer(text):
        if m.group(0) not in seen:
            seen.append(m.group(0))
    for c in seen:
        out.append(f'<small>{_e(c)}</small><br><audio controls preload="none" src="{_e(c)}"></audio><br>')
    out.append("</div>")
    return "".join(out)


def word_html(w: dict, span_info: dict, cap: bool = False) -> str:
    if w["kind"] == "event":
        return f'<span class="w emo joke" title="{_e("laugh token: " + (w.get("evidence") or ""))}" data-t="{w["start"]}">{_e(w["text"])}</span>'
    t = (w["text"][:1].upper() + w["text"][1:] if cap else w["text"]) + w["punct"]
    cls = ["w"]
    if w["conf"] < 0.5:
        cls.append("q0")
    elif w["conf"] < LOW:
        cls.append("q1")
    elif w["conf"] < MID:
        cls.append("q2")
    fl = w.get("flags") or []
    if "needs_owner" in fl:
        cls.append("own")
    if "joke" in fl:
        cls.append("joke")
    tip = [f"conf {w['conf']}", w.get("conf_src", ""), mmss(w["start"])]
    sp = w.get("span")
    if sp and sp in span_info:
        s = span_info[sp]
        tip.append(f"span {sp} {s['decision']} {s['confidence']}")
        if s.get("alternatives"):
            tip.append("alternatives: " + " | ".join(map(str, s["alternatives"])))
        if s.get("consensus_text") is not None and w.get("edit") in ("changed", "inserted"):
            tip.append(f"vote said: {s['consensus_text']!r}")
    return f'<span class="{" ".join(cls)}" title="{_e(" · ".join(x for x in tip if x))}">{_e(t)}</span>'


def build_review(final: dict, chapters: dict | None, jokes: dict | None, who_clips: list[dict], evaluation: dict | None, clip_files: dict[str, str], master_audio: str = "audio/01-Recap-Tina.m4a",
                 owner_questions: str | None = None) -> str:
    span_info = {s["id"]: s for s in final["adjudication"]["spans"]}
    st = final["stats"]
    out = [f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Recap transcript review</title><style>{CSS}</style></head><body>']
    out.append('<header><h1>Zouk class recap: transcript review</h1>')
    out.append('<p class="mut">Local file for the owner. It contains the voices and words of real people: never publish it. Master clock = 01-Recap-Tina video time; click a time to play from there.</p>')
    out.append(f'<audio id="master" controls preload="none" src="{_e(master_audio)}"></audio>')
    out.append('<p class="legend"><span><span class="w q0">&nbsp;very unsure&nbsp;</span> conf &lt; 0.5</span><span><span class="w q1">&nbsp;unsure&nbsp;</span> &lt; 0.6</span><span><span class="w q2">&nbsp;somewhat&nbsp;</span> &lt; 0.8</span>'
               '<span><span class="w own">&nbsp;owner&nbsp;</span> the editors need you</span><span class="joke">joke token</span></p>')
    out.append(f'<p><b>{st["words"]}</b> words in <b>{st["utterances"]}</b> sentences / <b>{st["turns"]}</b> turns; <b>{st["spans_total"]}</b> disputed spans adjudicated ({st["spans_changed"]} changed the text), '
               f'<b>{st["spans_unclear"]}</b> unclear, <b>{st["spans_needs_owner"]}</b> need you. Mean word confidence {st["conf_mean"]}; {st["conf_ge_0.9"]} words at 0.9 or more, {st["conf_lt_0.6"]} below 0.6. '
               f'Speaker names rest on an acoustic prior (naming confidence {_e((final.get("naming") or {}).get("naming_confidence"))}), not on how the teachers address each other.</p>')
    out.append('<label><input type="checkbox" id="onlylow"> show only sentences with a doubtful word (conf &lt; 0.6)</label>')
    if chapters:
        out.append('<nav><b>Chapters:</b> ' + " ".join(f'<a href="#{c["id"]}">{_e(c["title"])}</a>' for c in chapters["chapters"]) + (' <a href="#questions">Questions for you</a>' if owner_questions else '')
                   + ' <a href="#owner">Proofread first</a> <a href="#jokes">Jokes</a> <a href="#who">Who is who</a> <a href="#eval">Evaluation</a></nav>')
    out.append('</header><main>')

    if owner_questions:
        out.append(questions_html(owner_questions))

    # proofread first
    out.append('<h2 id="owner">Proofread first</h2><p class="mut">Spans where the editors could only guess and the audio decides. Listen to the clip, then read the options.</p>')
    owner = [s for s in final["adjudication"]["spans"] if s["needs_owner"]]
    for s in owner:
        clip = clip_files.get(s["id"])
        out.append(f'<div class="card"><b data-t="{max(s["start"] - 1, 0)}" class="t">{mmss(s["start"])}-{mmss(s["end"])}</b> <span class="badge own">{_e(s["id"])}</span> <span class="badge">{_e(s["decision"])} {s["confidence"]}</span><br>'
                   f'<b>{_e(s["text"])}</b><br><small>vote said: {_e(s.get("consensus_text"))}</small>'
                   + (f'<br><small>alternatives: {_e(" | ".join(map(str, s["alternatives"])))}</small>' if s.get("alternatives") else "")
                   + f'<br><small>{_e(s.get("reason"))}</small>' + (f'<br><small>{_e(s.get("note"))}</small>' if s.get("note") else "")
                   + (f'<br><audio controls preload="none" src="{_e(clip)}"></audio>' if clip else "")
                   + "".join(f'<br><small>the whole opening, 10 s ({_e(c["id"])}): {_e(c.get("reason"))}</small><br><audio controls preload="none" src="review/clips/{_e(c.get("file"))}"></audio>'
                             for c in who_clips if "joke" in s["flags"] and c.get("id") == "c01") + '</div>')

    # transcript
    out.append('<h2 id="transcript">Transcript</h2>')
    ch = (chapters or {}).get("chapters", [])
    ci = 0
    for u in final["utterances"]:
        while ci < len(ch) and (ci == 0 or u["start"] >= ch[ci]["start"] - 1e-6):
            c = ch[ci]
            out.append(f'<div class="chap" id="{c["id"]}"><span class="t" data-t="{c["start"]}">{mmss(c["start"])}</span> {_e(c["title"])} <small>{_e(c["summary"])}</small></div>')
            ci += 1
        low = [w for w in u["words"] if w["kind"] == "word" and w["conf"] < LOW]
        cls = "row" + ("" if low else " ok")
        parts, cap = [], True
        for w in u["words"]:
            parts.append(word_html(w, span_info, cap))
            if w["kind"] == "word":
                cap = w["punct"] in SENTENCE_END
        ws = " ".join(parts)
        doubt = ' <span class="badge" title="the independent speaker evidence does not settle who says this">speaker?</span>' if any("speaker_unresolved" in (w.get("flags") or []) for w in u["words"]) else ""
        out.append(f'<div class="{cls}"><span class="t" data-t="{max(u["start"], 0)}">{mmss(u["start"])}</span><span class="spk {_e(u["speaker"])}">{_e(u["speaker"])}{doubt}</span><span class="txt">{ws}</span></div>')

    # other unclear spans
    out.append('<h2>Other unclear spans</h2><div class="tw"><table><tr><th>time<th>id<th>text<th>conf<th>why</tr>')
    for s in final["adjudication"]["spans"]:
        if not s["needs_owner"] and ("unclear" in s["flags"] or s["decision"] == "uncertain"):
            clip = clip_files.get(s["id"])
            out.append(f'<tr><td class="t" data-t="{max(s["start"] - 1, 0)}">{mmss(s["start"])}</td><td>{_e(s["id"])}</td><td>{_e(s["text"])}<br><small>vote: {_e(s.get("consensus_text"))}</small>'
                       + (f'<br><audio controls preload="none" src="{_e(clip)}"></audio>' if clip else "") + f'</td><td>{s["confidence"]}</td><td><small>{_e(s.get("reason"))}</small></td></tr>')
    out.append('</table></div>')

    # jokes
    if jokes:
        out.append('<h2 id="jokes">Jokes and laughter</h2>')
        f = jokes.get("found_by_pipeline") or {}
        out.append(f'<div class="card"><b>Did the pipeline find them on its own?</b><br>{_e(f.get("summary"))}</div>')
        for key in ("joke1_larissa_says_kadu", "joke2_kadu_red_hair"):
            j = f.get(key)
            if j:
                out.append(f'<div class="card"><b>{_e(key)}</b>: found on own evidence: <b>{_e(j.get("found_on_own_evidence"))}</b><ul>' + "".join(f"<li>{_e(x)}</li>" for x in j.get("found", []))
                           + f'</ul><small>missed: {_e(j.get("missed"))}<br>needed the hint for: {_e(j.get("needed_the_hint_for"))}</small></div>')
        out.append('<div class="tw"><table><tr><th>time<th>token / candidate<th>why</tr>')
        for t in jokes.get("tokens_in_transcript", []):
            out.append(f'<tr><td class="t" data-t="{max(t["t0"] - 3, 0)}">{mmss(t["t0"])}-{mmss(t["t1"])}</td><td class="joke emo">{_e(t["token"])} in the text<br><small>after: {_e(t.get("after_words"))}</small></td><td><small>{_e(t.get("evidence"))}</small></td></tr>')
        for c in jokes.get("laugh_candidates", []):
            out.append(f'<tr><td class="t" data-t="{max(c["t0"] - 2, 0)}">{mmss(c["t0"])}-{mmss(c["t1"])}</td><td>candidate (not in the text)<br><small>{_e(c.get("text"))}</small></td><td><small>{_e(c.get("why"))}</small></td></tr>')
        out.append('</table></div>')

    # who is who
    out.append('<h2 id="who">Who is who (30 clips from the speaker step)</h2><p class="mut">If these anchor clips are right the voice naming is right; the last column is only for you (it is not saved).</p><div class="tw"><table><tr><th>#<th>master s<th>phone<th>category<th>play<th>system says<th>why<th>who?</tr>')
    for c in who_clips:
        out.append(f'<tr><td>{_e(c.get("id"))}</td><td>{_e(c["master_t0"])}-{_e(c["master_t1"])}</td><td>{_e(c.get("cam"))}</td><td>{_e(c.get("category"))}</td>'
                   f'<td><audio controls preload="none" src="review/clips/{_e(c.get("file"))}"></audio></td><td>{_e(c.get("name"))} {_e(c.get("confidence"))}</td>'
                   f'<td><small>{_e(c.get("reason"))}</small><br><small class="mut">{_e(c.get("text"))}</small></td>'
                   '<td><select><option>?</option><option>Larissa</option><option>Kadu</option><option>both</option><option>someone else</option></select></td></tr>')
    out.append('</table></div>')

    if evaluation:
        out.append('<h2 id="eval">Evaluation</h2>')
        ex = evaluation.get("expected_accuracy", {})
        out.append(f'<div class="card">{_e(ex.get("statement"))}</div>')
        out.append('<div class="tw"><table><tr><th>source (phone / engine)<th>words<th>v0 vs source<th>final vs source</tr>')
        for r in evaluation.get("rows", []):
            out.append(f'<tr><td>{_e(r["cam"])} / {_e(r["engine"])}</td><td>{r["n_ref"]}</td><td>{r["pwer_v0"]:.3f}</td><td>{r["pwer_final"]:.3f}</td></tr>')
        out.append('</table></div>')
    out.append(f'</main><script>{JS}</script></body></html>')
    return "\n".join(out)
