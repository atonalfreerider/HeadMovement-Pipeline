"""Captions of the recap film: the teachers' words as word-highlight chunks (the film director's narration-timeline format), with speaker
colour keys and the laugh emoji as a caption word of its own.

Input: work/<take>/speech/captions.json (the final transcript's captions: words with times, speaker keys, kind word | emoji) - the
text, capitals and punctuation are the transcript's; this module only re-chunks them for each output shape and maps master time to film
time. Output: `narration/timeline.json` of the film folder (`"direction"` = the direction's name, which is how the director finds it):

  captions.vertical / captions.horizontal : [{line, show, hide, lines[], words[{text, start, end, kind?, emoji?}], speaker, speaker_key, color}]

`kind: "emoji"` words carry `emoji` = the code point in hex ("1f602"); the viewer draws them from an image (Assets/Film/FilmEmoji.cs:
`<film dir>/emoji/<hex>.png` made by `python -m dancecap.film_recap emoji`, else a built-in drawing), never from the text font.

Chunk rules (the demo film's, adapted to speech that has pauses and two voices): a chunk is one speaker's phrase; it breaks at a speaker
change, at a pause of PAUSE_S, after a sentence end once it has MIN_WORDS words, at a comma once it is 60 % full, and when the next word
would not fit (9:16: 2 lines of 24 characters, 16:9: 1 line of 62; an emoji counts 2). A chunk shows from 0.15 s before its first word to 0.4 s
after its last (or the next chunk). Chunks never overlap in time (the overlay shows one at a time): when two voices overlap, the second
chunk appears when the first word of the first ends.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

PAUSE_S = 0.9
EMOJI_GAP_S = 1.8
MIN_WORDS = 3
MAX_CHUNK_S = 6.5
SHOW_LEAD_S = 0.15
HIDE_LAG_S = 0.4
SHAPES = {"vertical": {"max_chars": 24, "max_lines": 2}, "horizontal": {"max_chars": 62, "max_lines": 1}}
DEFAULT_COLORS = {"speaker_larissa": "#F28C6B", "speaker_kadu": "#5BA8E8", "speaker_other": "#B8B8B8"}
SENTENCE_END = re.compile(r"[.?!]$|[.?!][\"')]$")
EMOJI_CODE = {"\U0001F602": "1f602"}


def wlen(text: str) -> int:
    """characters of a word for line-fitting: an emoji (anything outside the BMP) counts 2"""
    return sum(2 if ord(c) > 0xFFFF else 1 for c in text)


def is_emoji(text: str) -> bool:
    return bool(text) and all(ord(c) > 0xFFFF or 0x2600 <= ord(c) <= 0x27BF for c in text.strip())


def flatten_words(captions: dict) -> list[dict]:
    """all caption words of captions.json in time order: {text, start, end, kind, speaker, speaker_key, conf, check}"""
    out = []
    for c in captions.get("captions", []):
        for w in c.get("words", []):
            out.append({"text": w["text"], "start": float(w["start"]), "end": float(w["end"]), "kind": w.get("kind", "word"),
                        "speaker": c.get("speaker") or "", "speaker_key": c.get("speaker_key") or "speaker_other", "conf": w.get("conf"),
                        "check": bool(c.get("check")), "caption": c.get("id")})
    out.sort(key=lambda w: (w["start"], w["end"]))
    return out


def speaker_table(captions: dict) -> dict[str, dict]:
    t = {k: {"name": k.replace("speaker_", "").title(), "color": v} for k, v in DEFAULT_COLORS.items()}
    for s in captions.get("speakers", []):
        t[s["key"]] = {"name": s.get("name", s["key"]), "color": s.get("color_hex_suggestion") or DEFAULT_COLORS.get(s["key"], "#FFFFFF")}
    return t


def break_lines(words: list[dict], max_chars: int, max_lines: int) -> list[list[dict]] | None:
    """words split into <= max_lines lines of <= max_chars characters (balanced: the break nearest the middle that fits); None if impossible"""
    if not words:
        return []
    lens = [wlen(w["text"]) for w in words]
    total = sum(lens) + len(words) - 1
    if total <= max_chars:
        return [words]
    if max_lines == 1:
        return None
    best = None
    for k in range(1, len(words)):
        a = sum(lens[:k]) + k - 1
        b = sum(lens[k:]) + (len(words) - k) - 1
        if a <= max_chars and b <= max_chars:
            cost = abs(a - b)
            if best is None or cost < best[0]:
                best = (cost, k)
    if best is None:
        return None
    return [words[:best[1]], words[best[1]:]]


def fits(words: list[dict], shape: dict) -> bool:
    return break_lines(words, shape["max_chars"], shape["max_lines"]) is not None


def chunk_words(words: list[dict], shape: dict, breaks: list[float] | None = None) -> list[list[dict]]:
    """greedy chunks (see the module docstring); `breaks` = master times at which a chunk must end (film-time jumps)"""
    breaks = sorted(breaks or [])
    chunks: list[list[dict]] = []
    cur: list[dict] = []

    def crosses_break(a: float, b: float) -> bool:
        return any(a < t <= b for t in breaks)

    for w in words:
        if cur:
            gap = w["start"] - cur[-1]["end"]
            new = False
            if w["speaker_key"] != cur[-1]["speaker_key"]:
                new = True
            elif crosses_break(cur[-1]["end"], w["start"]):
                new = True
            elif w["kind"] == "emoji":
                new = gap > EMOJI_GAP_S          # the laugh belongs to the sentence it follows, even across a short pause
            elif gap > PAUSE_S:
                new = True
            elif w["start"] - cur[0]["start"] > MAX_CHUNK_S:
                new = True
            elif SENTENCE_END.search(cur[-1]["text"]) and len([x for x in cur if x["kind"] != "emoji"]) >= MIN_WORDS:
                new = True
            elif cur[-1]["text"].endswith(",") and sum(wlen(x["text"]) + 1 for x in cur) >= 0.6 * shape["max_chars"] * shape["max_lines"]:
                new = True
            elif not fits(cur + [w], shape):
                new = True
            if new:
                chunks.append(cur)
                cur = []
        cur.append(w)
    if cur:
        chunks.append(cur)
    return chunks


BRIDGE_S = 1.0       # same-speaker chunks closer than this stay on screen without a blank between them


def build_chunks(words: list[dict], shape: dict, film_time, speakers: dict[str, dict], breaks: list[float] | None = None, tag: str = "c") -> list[dict]:
    """caption chunks in FILM seconds. `film_time(master_s)` maps a word time to the film clock."""
    raw = chunk_words(words, shape, breaks)
    chunks: list[dict] = []
    prev_hide = -1.0
    for i, ws in enumerate(raw):
        lines = break_lines(ws, shape["max_chars"], shape["max_lines"]) or [ws]
        fw = []
        for w in ws:
            d = {"text": w["text"], "start": round(film_time(w["start"]), 3), "end": round(film_time(w["end"]), 3)}
            if w["kind"] == "emoji" or is_emoji(w["text"]):
                code = "".join(f"{ord(ch):x}" for ch in w["text"].strip()) if len(w["text"].strip()) == 1 or all(ord(ch) > 0xFFFF for ch in w["text"].strip()) else None
                d["kind"] = "emoji"
                d["emoji"] = EMOJI_CODE.get(w["text"].strip(), code)
            if w.get("conf") is not None and w["conf"] < 0.5:
                d["unsure"] = True
            fw.append(d)
        for k in range(1, len(fw)):             # a word never starts before the previous one starts
            fw[k]["start"] = max(fw[k]["start"], fw[k - 1]["start"])
            fw[k]["end"] = max(fw[k]["end"], fw[k]["start"] + 0.02)
        spk = ws[0]["speaker_key"]
        sp = speakers.get(spk, {"name": "", "color": "#FFFFFF"})
        chunks.append({"line": f"{tag}{i:03d}", "show": round(fw[0]["start"] - SHOW_LEAD_S, 3), "hide": round(fw[-1]["end"] + HIDE_LAG_S, 3),
                       "lines": [" ".join(x["text"] for x in ln) for ln in lines], "words": fw, "speaker": sp["name"], "speaker_key": spk,
                       "color": sp["color"]})
    # no overlap: a chunk ends at the next one's show; when that would cut its last word the next one waits
    for i in range(len(chunks) - 1):
        a, b = chunks[i], chunks[i + 1]
        last_end = a["words"][-1]["end"]
        if b["show"] < last_end:
            b["show"] = round(min(last_end + 0.02, b["words"][0]["end"]), 3)
        a["hide"] = round(max(min(a["hide"], b["show"] - 0.02), last_end + 0.04), 3)
        if a["hide"] > b["show"]:
            b["show"] = round(a["hide"] + 0.01, 3)
    # D2: a short blank between two chunks of ONE speaker read as a flicker (the audit saw a one-word chunk flash and a 0.7 s blank at 274-276 s):
    # the earlier chunk stays up until the next one comes when that is within BRIDGE_S
    for i in range(len(chunks) - 1):
        a, b = chunks[i], chunks[i + 1]
        gap = b["show"] - a["hide"]
        if a["speaker_key"] == b["speaker_key"] and 0.0 < gap <= BRIDGE_S:
            a["hide"] = round(b["show"] - 0.02, 3)
    for c in chunks:
        c["show"] = max(0.0, c["show"])
    return chunks


def timeline_doc(direction_name: str, film_duration: float, chunks: dict[str, list[dict]], speakers: dict[str, dict]) -> dict:
    return {"name": f"timeline_{direction_name}", "direction": direction_name, "film_duration_s": round(film_duration, 3),
            "time_base": "FILM time (seconds into the rendered video); the class audio is the soundtrack, there is no narration",
            "lines": [], "captions": chunks, "speakers": speakers,
            "caption_style": "the director's karaoke chunks: a chunk is one speaker's phrase, words coloured by `color`, the current word on the yellow box; "
                             "kind 'emoji' words are drawn from an image"}


def check_chunks(chunks: list[dict], shape: dict) -> list[str]:
    """problems: overlap in time, a chunk that does not fit its shape, words out of order"""
    bad = []
    for i, c in enumerate(chunks):
        if c["hide"] <= c["show"]:
            bad.append(f"{c['line']}: hide <= show")
        if i and c["show"] < chunks[i - 1]["hide"] - 1e-6:
            bad.append(f"{c['line']} overlaps {chunks[i - 1]['line']}")
        if len(c["lines"]) > shape["max_lines"]:
            bad.append(f"{c['line']}: {len(c['lines'])} lines")
        for ln in c["lines"]:
            if wlen(ln) + 0 > shape["max_chars"] + 1:
                bad.append(f"{c['line']}: a line of {wlen(ln)} characters: {ln!r}")
        for a, b in zip(c["words"], c["words"][1:]):
            if b["start"] < a["start"]:
                bad.append(f"{c['line']}: words out of order")
    return bad


# ------------------------------------------------------------------------------------------------ the emoji image


def system_emoji_font() -> Path | None:
    """a colour-emoji font on this machine (Windows: Segoe UI Emoji; macOS: Apple Color Emoji; Linux: Noto Color Emoji)"""
    for p in ("C:/Windows/Fonts/seguiemj.ttf", "/System/Library/Fonts/Apple Color Emoji.ttc", "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"):
        if Path(p).exists():
            return Path(p)
    return None


def render_emoji_png(char: str, out: Path, size: int = 256) -> Path | None:
    """Draw one emoji from the system's colour-emoji font into a transparent PNG (git-ignored film folder, never committed: the image is a
    font rendering, not part of this repository). Returns None when no colour font or no colour rendering is available (the viewer then
    draws its built-in version)."""
    font_path = system_emoji_font()
    if font_path is None:
        return None
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype(str(font_path), 109 if "seguiemj" in font_path.name.lower() else size)
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.text((size / 2, size / 2), char, font=font, embedded_color=True, anchor="mm")
    except Exception:                       # pragma: no cover - font/engine dependent
        return None
    bbox = img.getbbox()
    if bbox is None:
        return None
    # centre and fit into the square with a small margin
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    s = min((size * 0.92) / w, (size * 0.92) / h)
    crop = img.crop(bbox).resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    canvas.paste(crop, ((size - crop.width) // 2, (size - crop.height) // 2), crop)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    return out


def write_timeline(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
