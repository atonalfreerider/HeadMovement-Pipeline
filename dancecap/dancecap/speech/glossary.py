"""Domain glossary of the recap class: English + Brazilian Portuguese zouk terms for ASR prompting and transcript fusion.

    python -m dancecap.speech glossary            # writes work/<take>/speech/glossary.json

Sources, in this order of authority:
  1. taxonomy/pt_br_terminology.json  (docs/ZOUK_TERMINOLOGY.md): the attested / partial Brazilian Portuguese move names with their
     English aliases (Corredor = Lateral, Viradinha, Cambre, Ondulacao, ...). Descriptive-only rows are NOT used as terms.
  2. the project's own analysis vocabulary (floorcraft / counterbalance / weight change / T axis ...), which the teachers
     may also say in English, and
  3. general zouk-class vocabulary (below) for words a class says all the time. These are marked source = "class" and are not
     claimed to be official terms.

Each entry: {canonical, lang, kind, en, variants [spellings an English or Portuguese recogniser may produce], source, weight}.
`weight` orders the prompt (higher first) and is a hint for the fusion's glossary rescue, nothing more.

Prompts: Whisper takes a 224-token prompt (initial_prompt) and `hotwords`; parakeet takes neither, so its output is rescued
afterwards by `rescue_candidates` (fuzzy match of a recognised word to a glossary entry, to be CONFIRMED by a second engine or by
the audio, never applied blindly).
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from ..take import PACKAGE_ROOT
from . import SPEECH

TERMINOLOGY = PACKAGE_ROOT / "taxonomy" / "pt_br_terminology.json"

# the other people of the class (the phones' owners; the teachers may call them by name). They are NOT teachers: a vocative of one
# of these names says nothing about Larissa or Kadu. "Redshirt" is the owner's nickname for camera 03's operator.
OTHERS = [("Ali", ["Ally", "Ollie"]), ("Tina", ["Tena", "Teena"]), ("Alyssa", ["Alissa", "Elisa", "Alisha"]), ("Tony", ["Toni"]),
          ("Liudmyla", ["Ludmila", "Lyudmila", "Ludmilla", "Mila"]), ("John", ["Jon"])]

PEOPLE = [
    {"canonical": "Larissa", "lang": "pt", "kind": "person", "variants": ["Larisa", "Lari", "Larrisa", "Clarissa", "Lorissa"],
     "weight": 100, "note": "teacher (the follower in the demo, long red hair)"},
    {"canonical": "Kadu", "lang": "pt", "kind": "person", "variants": ["Cadu", "Caduh", "Kadoo", "Cardu", "Kaddu"],
     "weight": 100, "note": "teacher (the leader in the demo, shaved head); nickname, spelled Kadu by the owner"},
]

# (canonical, lang, kind, english gloss, variants, weight). Variants are spellings a recogniser may produce for the SPOKEN word.
_CLASS = [
    # the project's own analysis words (the owner / the teachers may say them in English)
    ("counterbalance", "en", "technique", None, ["counter balance", "counter-balance", "contra balance"], 95),
    ("T axis", "en", "technique", None, ["t-axis", "tea axis", "T access", "T axes", "the T"], 95),
    ("weight change", "en", "technique", "troca de peso", ["weight shift", "wait change", "weight changes"], 95),
    ("weight", "en", "technique", "peso", [], 60),
    ("axis", "en", "technique", "eixo", ["access"], 70),
    ("pivot", "en", "technique", None, ["pivots", "pivoting"], 70),
    ("plant", "en", "technique", None, ["planted"], 50),
    ("frame", "en", "technique", None, ["frames"], 70),
    ("connection", "en", "technique", "conexão", ["connections"], 70),
    ("basic", "en", "move", "base", ["basics"], 70),
    ("base", "pt", "move", "basic", ["bass", "bace"], 80),
    ("lead", "en", "role", "condutor", ["leader", "leaders"], 75),
    ("follow", "en", "role", "conduzida", ["follower", "followers"], 75),
    ("head movement", "en", "move", "movimento de cabeça", ["head movements", "head motion"], 80),
    ("hair", "en", "move", "cabelo", [], 40),
    ("spiral", "en", "technique", None, ["spirals"], 50),
    ("transition", "en", "technique", "transição", ["transitions"], 50),
    ("compression", "en", "technique", "compressão", [], 40),
    ("extension", "en", "technique", "extensão", [], 40),
    ("spine", "en", "body", "coluna", [], 40),
    ("ribcage", "en", "body", "caixa torácica", ["rib cage", "ribs"], 40),
    ("hips", "en", "body", "quadril", ["hip"], 40),
    ("shoulders", "en", "body", "ombros", ["shoulder"], 40),
    ("elbow", "en", "body", "cotovelo", [], 30),
    ("knees", "en", "body", "joelhos", [], 30),
    ("count", "en", "timing", "tempo", ["counts"], 50),
    ("slow", "en", "timing", "lento", [], 40),
    ("quick", "en", "timing", "rápido", [], 40),
    ("on one", "en", "timing", None, ["on 1", "on the one"], 40),
    ("and", "en", "timing", "e", [], 10),
    # Brazilian zouk words used in every class (general usage; official status not claimed)
    ("zouk", "pt", "genre", None, ["zook", "zuke", "zoke"], 90),
    ("Brazilian zouk", "en", "genre", "zouk brasileiro", [], 80),
    ("sacada", "pt", "move", "leg displacement (gafieira / tango word)", ["cicada", "sakada", "secada", "sicada"], 85),   # said at master 220 s by the teachers, all engines hear it
    ("ginga", "pt", "technique", None, ["jinga", "ginja"], 60),
    ("bate-cabelo", "pt", "move", "hair flick", ["bate cabelo", "batty cabelo", "bati cabelo"], 70),
    ("chicote", "pt", "move", "whip", ["shicote", "chicot", "she-co-chi"], 60),
    ("movimento de cabeça", "pt", "move", "head movement", ["movimento de cabesa"], 70),
    ("troca de peso", "pt", "technique", "weight change", ["troca de pesso"], 70),
    ("eixo", "pt", "technique", "axis", ["eisho", "ei-xo"], 60),
    ("contratempo", "pt", "timing", "off-beat", ["contra tempo"], 40),
    ("tempo", "pt", "timing", None, [], 30),
    ("obrigada", "pt", "phrase", "thank you (f)", ["obrigado"], 30),
    ("muito bem", "pt", "phrase", "very good", [], 20),
    ("pode ser", "pt", "phrase", None, [], 10),
]


def normalise(s: str) -> str:
    """Lower case, accents stripped, punctuation to spaces: the key for glossary matching."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def _from_terminology(path: Path = TERMINOLOGY) -> list[dict]:
    if not path.exists():
        return []
    doc = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for r in doc.get("rows", []):
        status = r.get("status")
        canon = r.get("canonical")
        if canon and "From Behind" in canon:      # a derived English hybrid, never spoken as such
            continue
        if not canon or status == "unattested" and not r.get("canonical_is_pt"):
            # an English move name with no Portuguese term: the teachers still say it, in English
            if canon and len(canon) > 3 and not canon.isdigit():
                out.append({"canonical": canon, "lang": "en", "kind": "move", "en": canon, "variants": [], "source": "taxonomy:" + str(status),
                            "weight": 25})
            continue
        english = [a for a in r.get("english_aliases", []) if normalise(a) != normalise(canon)]
        out.append({"canonical": canon, "lang": "pt" if r.get("canonical_is_pt") else "en", "kind": "move",
                    "en": (english[0] if english else r.get("original") if normalise(r.get("original", "")) != normalise(canon) else None),
                    "variants": list(dict.fromkeys(english + [a for a in r.get("alternates_pt", [])]))[:6],
                    "source": f"taxonomy:{status}", "weight": {"attested": 85, "user_decision": 95, "partial": 55}.get(status, 25)})
    # Corredor is the owner's decision; Lateral is its alias and is still said in many schools
    out.append({"canonical": "lateral", "lang": "en", "kind": "move", "en": "corredor", "variants": ["laterals"], "source": "taxonomy:user_decision",
                "weight": 90})
    return out


def build() -> dict:
    terms: list[dict] = []
    for p in PEOPLE:
        terms.append({**p, "en": None, "source": "people"})
    for name, variants in OTHERS:
        terms.append({"canonical": name, "lang": "en", "kind": "person_other", "en": None, "variants": variants, "source": "people",
                      "weight": 35, "note": "student / phone operator, not a teacher"})
    for canon, lang, kind, en, variants, weight in _CLASS:
        terms.append({"canonical": canon, "lang": lang, "kind": kind, "en": en, "variants": variants, "source": "class", "weight": weight})
    terms += _from_terminology()
    seen: dict[str, dict] = {}
    for t in terms:
        k = normalise(t["canonical"])
        if not k:
            continue
        if k in seen:
            prev = seen[k]
            prev["variants"] = list(dict.fromkeys(prev["variants"] + t.get("variants", [])))
            prev["weight"] = max(prev["weight"], t.get("weight", 0))
            continue
        t["key"] = k
        seen[k] = t
    out = sorted(seen.values(), key=lambda t: (-t["weight"], t["key"]))
    return {"version": 1, "what": "zouk class glossary (English + Brazilian Portuguese) for ASR prompting and transcript fusion",
            "n_terms": len(out), "terms": out, "prompts": prompts(out)}


def _approx_tokens(s: str) -> int:
    return max(1, int(len(s) / 3.2) + 1)


def pick(terms: list[dict], max_tokens: int, lang: str | None = None) -> list[str]:
    """Canonical spellings, heaviest first, whole terms only, within an approximate token budget (Whisper cuts at 223 tokens)."""
    out, used = [], 0
    for t in terms:
        if lang and t["lang"] != lang and t["kind"] not in ("person", "person_other", "technique", "genre"):
            continue
        c = t["canonical"]
        n = _approx_tokens(c) + 1
        if used + n > max_tokens:
            break
        out.append(c)
        used += n
    return out


def prompts(terms: list[dict]) -> dict:
    """Prompt strings. Whisper's initial_prompt is a PREVIOUS-TEXT style prompt: plain sentences that contain the spellings."""
    top = pick(terms, 150)
    head = "A zouk dance class by two teachers, Larissa and Kadu, in English with Brazilian Portuguese terms. "
    return {
        "whisper_initial_prompt": head + "Terms: " + ", ".join(top) + ".",
        "whisper_initial_prompt_pt": "Aula de zouk brasileiro com os professores Larissa e Kadu. Termos: " + ", ".join(pick(terms, 150, "pt")) + ".",
        "hotwords": ", ".join(pick(terms, 200)),
        "people": [p["canonical"] for p in PEOPLE],
    }


def rescue_candidates(word: str, terms: list[dict], max_distance: int = 2) -> list[dict]:
    """Glossary entries a recognised word may stand for: an exact variant match first, then a small edit distance on the normalised
    spelling for words of 5+ letters. Returns [{term, how, distance}] (best first); the caller must still confirm."""
    w = normalise(word)
    if not w:
        return []
    out = []
    for t in terms:
        keys = [t["key"]] + [normalise(v) for v in t.get("variants", [])]
        if w in keys[1:]:
            out.append({"term": t["canonical"], "how": "variant", "distance": 0, "weight": t["weight"]})
        elif w == keys[0]:
            out.append({"term": t["canonical"], "how": "exact", "distance": 0, "weight": t["weight"]})
        elif len(w) >= 5 and " " not in keys[0] and abs(len(w) - len(keys[0])) <= max_distance:
            d = edit_distance(w, keys[0])
            if d <= max_distance:
                out.append({"term": t["canonical"], "how": "fuzzy", "distance": d, "weight": t["weight"]})
    out.sort(key=lambda r: ({"exact": 0, "variant": 1, "fuzzy": 2}[r["how"]], r["distance"], -r["weight"]))
    return out


def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def write(path: Path | None = None) -> Path:
    path = path or SPEECH / "glossary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = build()
    path.write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
    return path
