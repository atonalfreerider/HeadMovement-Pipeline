"""The ASR outputs of all engines and phones as one list of words on the master clock (local, read-only).

Engines (raw/<cam>/<file>): whisper-large-v3.json and whisper-large-v3-turbo.json (passes auto / en / pt), parakeet-tdt-0.6b-v3.json,
canary-1b-v2.json. Each word keeps its source time (`t0`, `t1`), its master time (`m0`, `m1`), the engine key (`engine`, e.g.
"whisper-large-v3:en"), the phone (`cam`) and the engine's own probability when it has one (`prob`).
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

ENGINE_FILES = ["whisper-large-v3.json", "whisper-large-v3-turbo.json", "parakeet-tdt-0.6b-v3.json", "canary-1b-v2.json"]


@dataclass
class Word:
    cam: str
    engine: str
    text: str
    t0: float
    t1: float
    m0: float
    m1: float
    prob: float | None
    seg: int | None = None

    @property
    def norm(self) -> str:
        return norm_token(self.text)


def norm_token(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return re.sub(r"[^a-z0-9']+", "", s).replace("'", "")


def engine_keys(raw_dir: Path, cam: str) -> list[str]:
    out = []
    for f in ENGINE_FILES:
        p = Path(raw_dir) / cam / f
        if not p.exists():
            continue
        doc = json.loads(p.read_text(encoding="utf-8"))
        base = f[:-5]
        if "passes" in doc:
            out += [f"{base}:{ps['requested']}" for ps in doc["passes"]]
        else:
            out.append(base)
    return out


def load_engine(raw_dir: Path, cam: str, key: str, clock) -> list[Word]:
    """Words of one engine key ('whisper-large-v3:en' or 'parakeet-tdt-0.6b-v3') of one phone, moved to the master clock."""
    base, _, requested = key.partition(":")
    p = Path(raw_dir) / cam / f"{base}.json"
    if not p.exists():
        return []
    doc = json.loads(p.read_text(encoding="utf-8"))
    if "passes" in doc:
        ps = next((x for x in doc["passes"] if x["requested"] == requested), None)
        raw = ps["words"] if ps else []
    else:
        raw = doc.get("words", [])
    out = []
    for w in raw:
        if not w.get("word"):
            continue
        m0 = clock.to_master(w["start"]) if clock else w["start"]
        m1 = clock.to_master(w["end"]) if clock else w["end"]
        out.append(Word(cam, key, w["word"], float(w["start"]), float(w["end"]), m0, m1, w.get("prob"), w.get("seg")))
    return out


def load_words(raw_dir: Path, clocks: dict, engines: list[str] | None = None, cams: list[str] | None = None) -> list[Word]:
    """Every word of every phone that has a clock, sorted by master time."""
    words: list[Word] = []
    for d in sorted(Path(raw_dir).glob("[0-9][0-9]")):
        cam = d.name
        clk = clocks.get(cam)
        if clk is None or (cams and cam not in cams):
            continue
        for key in engine_keys(raw_dir, cam):
            if engines and key not in engines and key.split(":")[0] not in engines:
                continue
            words += load_engine(raw_dir, cam, key, clk)
    words.sort(key=lambda w: (w.m0, w.m1))
    return words


def text_between(words: list[Word], m0: float, m1: float, engine: str, cam: str | None = None) -> str:
    """The words of one engine (and phone) whose centre lies in [m0, m1], joined."""
    sel = [w for w in words if w.engine == engine and (cam is None or w.cam == cam) and m0 <= 0.5 * (w.m0 + w.m1) <= m1]
    return " ".join(w.text for w in sel)


def load_regions(regions_dir: Path, clocks: dict) -> list[Word]:
    """Words of the targeted multi-setting decodes (raw_regions/<region>/<cam>.json) on the master clock; the engine key is
    'region:<model>:<variant>' so they can be told apart from the full-length passes."""
    out: list[Word] = []
    for f in sorted(Path(regions_dir).glob("*/[0-9][0-9].json")):
        cam = f.stem
        clk = clocks.get(cam)
        if clk is None:
            continue
        doc = json.loads(f.read_text(encoding="utf-8"))
        for h in doc.get("hypotheses", []):
            key = f"region:{h['model']}:{h['variant']}"
            for w in h.get("words", []):
                if w.get("word"):
                    out.append(Word(cam, key, w["word"], float(w["start"]), float(w["end"]), clk.to_master(float(w["start"])), clk.to_master(float(w["end"])),
                                    w.get("prob")))
    out.sort(key=lambda w: (w.m0, w.m1))
    return out
