"""Narrated phrases -> dance moves: the move lexicon, offline fuzzy + phonetic matching, control words, and an optional
TypeSafe judgment for ambiguous phrases. Used by the annotation tool (dancecap.moves.annotate); see
docs/MOVES_RESEARCH.md section 7 (narration-to-timeline design).

    lex = Lexicon.from_graph(load_graph())
    lex.candidates("veraginha")                 -> [Candidate(move='viradinha', score=95.0, ...), ...]
    events = map_utterances(utterances, lex)    -> [{kind: move|unknown|control|ignored, move, phrase, ...}]

Lexicon: every move of taxonomy/zouk_graph.json (scaffold, proposed, accepted) with its display name, English Zouk1
name, aliases (incl. the user's learned ASR spellings from additions.toml), authored spelling variants and node names,
plus the vocabulary research's unresolved aliases (taxonomy/vocabulary_research.json unresolved_aliases: e.g. a word
heard but not yet identified - it never resolves to a move by itself, it surfaces as 'unknown' with candidates).

Matching one phrase (1-4 words, normalised: lower case, accents dropped, punctuation removed):
  exact     the phrase is a term (slug, name, alias, variant)                                score 100
  stem      equal after dropping -s / -es / -ing / -ed ("turns" -> turn)                     score 96
  fuzzy     string similarity (rapidfuzz ratio when installed, else difflib), spaces ignored
  phonetic  a consonant-skeleton sound key for English ASR renderings of Portuguese names (nh->n, lh->l, di/gi/ge->j,
            v->f, c->k/s, z->s, vowels dropped): veraginha / viradinha / "vira dinha" all -> 'frjn'.
            score = max(fuzzy, (fuzzy + sound) / 2 + 6 when the keys are equal and fuzzy >= 70).
            Portuguese-spelled terms (accents, nh / lh / rr / ç / -ão, -inha, -ada, gue / que) also get English-
            listener respellings (pt_respell: Toalha -> toalia, Viradinha -> viradinia / virajinia, Bumerangue ->
            bumerangi, Chicote -> shikoti / shikochi), compared with the phrase in a phonetic normal form (phon:
            ph->f, c->k/s, sh/ch->x, y->i, w->u, oo->u, h dropped, doubles collapsed, spaces ignored):
            score = 0.97 x ratio ("walia" -> Toalha, "boomerangy" -> Bumerangue); phrases of < 4 phonetic letters
            never match this way.
  link prior +4 for a move the graph links to from the previous narrated move (never above 99)
  prefix    the phrase is the first word(s) of a longer term ("howl" -> Howl Up, Howl Down)      score 72
  Short terms (<= 4 letters: Pop, Dip, Hug, Fall, Turn) only match exactly, by stem or as a prefix; a short
  Portuguese-spelled one also phonetically against a phrase of similar length >= 4 letters (Whisper's "Piau" ->
  Pião 97 via the Brazilian final o -> u respelling; "piano" -> Pião 86 stays an unknown with a candidate).
Decision: top >= ACCEPT (88) with the runner-up (another move) >= MARGIN (5) points behind -> 'matched';
  60 <= top < 88, or a close runner-up -> 'ambiguous': the TypeSafe judgment when enabled (judge_phrase: stated
  argument - "does the phrase name a move at all?" (Noul) + "which of these candidates?" (Choice); refused when the
  weakest of the two is < REFUSE (0.4)), otherwise left as 'unknown: <phrase>' with the candidates for the user.
  A close runner-up above ACCEPT without a judgment keeps the top move with confidence x 0.8.
Punctuated transcripts (Whisper writes the narrated list as "Open Break, Tilted Turn, Shadow Position,"): each item
  of <= 4 content words is one phrase - an accepted move with other content words left in the item is 'partial'
  (confidence x 0.5, flagged: "Tilted Turn" -> turn, "the graph may lack this variant"), an item with only weak
  matches is ONE unknown phrase with merged candidates ("shadow position", not "position").

Utterances (pause-delimited word groups): matched phrases are chosen greedily (score, then length), non-overlapping;
  a phrase never spans the recogniser's punctuation (Whisper writes a narrated list as "Open Break, Viradinha,").
  In an utterance of more than 4 content words only accepted matches count (weak look-alikes in talk are ignored).
  An utterance without a move: only control words -> control event; 1-4 content words -> 'unknown' event (the user
  named a move the graph does not know - it becomes a proposed move only when the user accepts it in the UI); longer
  talk -> 'ignored' (kept in the transcript). When the recogniser offered alternative hypotheses (SAPI grammar pass),
  they are tried for an utterance without an accepted move (method 'alternative', confidence <= 0.35: always
  flagged for review).
Control words: "undo" / "scratch that" / "delete that" / "cancel that" remove the previous move event (a leading "no
  wait" / "scratch that" / "correction" before a move replaces it); "again" / "same" / "repeat" / "one more" repeat the
  previous move (a new instance). A bare "no" is not a control word (recognisers hear it in noise). Fillers (English,
  Portuguese function words, counting digits) are ignored; a phrase that is exactly a term is never a filler.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

try:  # optional speed-up (MIT); difflib gives the same kind of ratio
    from rapidfuzz import fuzz as _rf_fuzz  # type: ignore
except Exception:  # noqa: BLE001
    _rf_fuzz = None

ACCEPT = 88.0
CONSIDER = 60.0
MARGIN = 5.0
LINK_BONUS = 4.0
REFUSE = 0.4
MAX_NGRAM = 4
SHORT = 4
PREFIX = 72.0
PARTIAL_FACTOR = 0.5   # confidence factor of a move matched inside a longer narrated item ("Tilted Turn" -> turn)

FILLERS = {"um", "uh", "uhm", "erm", "er", "ah", "hmm", "mm", "oh", "okay", "ok", "so", "like", "well", "yeah", "yes",
           "right", "alright", "now", "then", "and", "the", "a", "an", "this", "that", "here", "there", "is", "it",
           "its", "they", "she", "he", "we", "do", "does", "doing", "into", "to", "of", "in", "on", "at", "with",
           "go", "goes", "going", "gonna", "comes", "coming", "come", "another", "some", "kind", "sort", "thing",
           "next", "move", "moves", "again", "her", "him", "his", "their", "them", "there's", "thats", "what",
           "s", "t", "re", "ll", "ve", "d", "m", "just", "bit", "for", "by", "up", "out", "from",
           "one", "two", "three", "four", "five", "six", "seven", "eight", "count",  # counting along is not a move
           "1", "2", "3", "4", "5", "6", "7", "8"}
PT_FILLERS = {  # Portuguese function words (the narrator sometimes talks Portuguese); move names keep them inside
    "o", "os", "as", "uma", "umas", "uns", "de", "do", "da", "dos", "das", "e", "que", "agora", "aqui", "ali", "ela",
    "ele", "eles", "elas", "pra", "pro", "para", "com", "no", "na", "nos", "nas", "isso", "esse", "essa", "este",
    "esta", "entao", "ta", "ne", "mais", "outro", "outra", "depois", "ai", "la", "tem", "faz", "sim", "nao", "bom",
    "olha", "tipo", "assim", "aquele", "aquela", "ja", "vamos", "fazer", "movimento"}
FILLERS |= PT_FILLERS
# fillers that are also parts of move names ("Little Turn", "Walk Around", "Do The Wave", "Hold Behind" ...) stay
# matchable inside a phrase; a phrase that starts or ends with a filler and is not exactly a term scores
# EDGE_PENALTY lower per filler edge (so "the lateral" never beats "lateral"); an all-filler phrase is never a move.
EDGE_PENALTY = 3.0
CONTROL = {  # a bare "no" / "nope" is NOT a control word: recognisers produce it from noise and mumbles
    "undo": "undo", "scratch that": "undo", "delete that": "undo", "cancel that": "undo",
    "again": "repeat", "same": "repeat", "same again": "repeat", "repeat": "repeat", "one more": "repeat",
    "same move": "repeat", "and again": "repeat", "once more": "repeat",
}
LEADING_UNDO = {"scratch that", "undo", "no wait", "correction", "i mean"}  # "<this> <move>" replaces the last move
ALT_MIN_PROB = 0.1    # recogniser alternatives below this probability are ignored
ALT_MAX_CONF = 0.35   # ... and the ones used never get a higher confidence (always flagged for review); an
#                       alternative may carry its own conf_cap (asr.fuse_passes: the other Whisper language pass)

_ONES = "zero one two three four five six seven eight nine".split()
_TEENS = "ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


# ---------------------------------------------------------------------------------------------------------------
# text helpers
# ---------------------------------------------------------------------------------------------------------------

def normalise(text: str | None) -> str:
    """'Pião, ' -> 'piao'; 'Lateral-From Behind' -> 'lateral from behind' (lower case, ascii, single spaces)."""
    if not text:
        return ""
    t = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode().lower()
    t = t.replace("'", "")
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


def _two_digits(n: int) -> str:
    if n < 10:
        return _ONES[n]
    if n < 20:
        return _TEENS[n - 10]
    return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])


def spoken_numbers(term: str) -> list[str]:
    """'180' -> ['one eighty', 'one hundred eighty', 'one hundred and eighty']; [] for anything else."""
    if not term.isdigit() or not (0 < int(term) < 1000):
        return []
    n = int(term)
    if n < 100:
        return [_two_digits(n)]
    h, r = divmod(n, 100)
    if r == 0:
        return [f"{_ONES[h]} hundred"]
    rw = _two_digits(r)
    return [f"{_ONES[h]} {rw}", f"{_ONES[h]} hundred {rw}", f"{_ONES[h]} hundred and {rw}"]


def stem(tok: str) -> str:
    if len(tok) > 5 and tok.endswith("ing"):
        return tok[:-3]
    if len(tok) > 4 and tok.endswith("ed"):
        return tok[:-2]
    if len(tok) > 4 and tok.endswith("es"):
        return tok[:-2]
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def stem_phrase(p: str) -> str:
    return " ".join(stem(t) for t in p.split())


_SOUND_RULES = [(r"tch", "x"), (r"sch", "x"), (r"ch", "x"), (r"sh", "x"), (r"nh", "n"), (r"lh", "l"), (r"ph", "f"),
                (r"th", "t"), (r"ck", "k"), (r"qu", "k"), (r"gu(?=[ei])", "g"), (r"dj", "j"), (r"d(?=[iy])", "j"),
                (r"g(?=[eiy])", "j"), (r"c(?=[eiy])", "s"), (r"c", "k"), (r"q", "k"), (r"z", "s"), (r"v", "f"),
                (r"w", "u"), (r"y", "i")]


def sound_key(text: str) -> str:
    """Consonant skeleton for matching what an English recogniser writes for a (Portuguese) move name."""
    t = re.sub(r"[^a-z]", "", normalise(text))
    if not t:
        return ""
    for a, b in _SOUND_RULES:
        t = re.sub(a, b, t)
    head, rest = t[0], t[1:]
    rest = re.sub(r"[aeiouh]", "", rest)
    out = head + rest
    return re.sub(r"(.)\1+", r"\1", out)


_PT_CHARS = [("ções", "soins"), ("ção", "saun"), ("ões", "oins"), ("ão", "aun"), ("ãe", "ain"), ("õe", "oin"),
             ("ç", "s")]
_PT_LOOK = re.compile(r"[ãõçáéíóúâêôà]|nh|lh|rr|inh[oa]\b|ad[oa]\b|gu[ei]|qu[ei]|ao\b|eiro")


def looks_portuguese(raw: str | None) -> bool:
    """Accents, nh / lh / rr / ç / -ão, -inha / -inho, -ada / -ado, gue / que, -eiro: spelled the Portuguese way."""
    return bool(raw) and bool(_PT_LOOK.search(unicodedata.normalize("NFC", str(raw)).lower()))


def pt_respell(raw: str | None) -> list[str]:
    """How an English listener (or an English-decoding recogniser) writes a Portuguese term: the spelling, a plain
    respelling and a Brazilian one (di -> ji, ti -> chi, final e -> i, final o -> u). 'Viradinha' -> [..,
    'viradinia', 'virajinia'], 'Toalha' -> [.., 'toalia'], 'Bumerangue' -> [.., 'bumerangi']."""
    if not raw:
        return []
    t = unicodedata.normalize("NFC", str(raw)).lower()
    for a, b in _PT_CHARS:
        t = t.replace(a, b)
    t = normalise(t)
    for a, b in ((r"nh", "ni"), (r"lh", "li"), (r"rr", "r"), (r"x", "sh"), (r"ch", "sh"), (r"g(?=[ei])", "j"),
                 (r"qu(?=[ei])", "k"), (r"gu(?=[ei])", "g"), (r"ou", "o")):
        t = re.sub(a, b, t)
    base = re.sub(r"e\b", "i", t)
    pal = base
    for a, b in ((r"d(?=i)", "j"), (r"t(?=i)", "ch"), (r"o\b", "u")):
        pal = re.sub(a, b, pal)
    return list(dict.fromkeys(x for x in (normalise(raw), t, base, pal) if x))


_PHON_RULES = [(r"tch", "x"), (r"sch", "x"), (r"ch", "x"), (r"sh", "x"), (r"ph", "f"), (r"ck", "k"), (r"qu", "k"),
               (r"c(?=[eiy])", "s"), (r"c", "k"), (r"q", "k"), (r"z", "s"), (r"dj", "j"), (r"dg", "j"), (r"oo", "u"),
               (r"ee", "i"), (r"ea", "i"), (r"y", "i"), (r"w", "u"), (r"h", "")]


def phon(text: str | None) -> str:
    """Phonetic normal form of an English-orthography spelling, spaces dropped: 'toalia' -> 'toalia', 'Walia' ->
    'ualia', 'how' -> 'ou', 'shikochi' -> 'xikoxi'. Compared by fuzzy ratio, not equality."""
    t = re.sub(r"[^a-z]", "", normalise(text))
    for a, b in _PHON_RULES:
        t = re.sub(a, b, t)
    return re.sub(r"(.)\1+", r"\1", t)


def ratio(a: str, b: str) -> float:
    if _rf_fuzz is not None:
        return float(_rf_fuzz.ratio(a, b))
    return SequenceMatcher(None, a, b).ratio() * 100.0


def string_score(a: str, b: str) -> float:
    """0..100 similarity of two normalised phrases (spaces ignored; token order tolerated for multi-word)."""
    if a == b:
        return 100.0
    a2, b2 = a.replace(" ", ""), b.replace(" ", "")
    if a2 == b2:
        return 98.0
    if _rf_fuzz is None:
        sm = SequenceMatcher(None, a2, b2)
        if sm.real_quick_ratio() * 100 < CONSIDER - 15 or sm.quick_ratio() * 100 < CONSIDER - 15:
            return 0.0
        s = sm.ratio() * 100.0
    else:
        s = ratio(a2, b2)
    if " " in a or " " in b:
        s = max(s, 0.95 * ratio(" ".join(sorted(a.split())), " ".join(sorted(b.split()))))
    return min(s, 97.0)


# ---------------------------------------------------------------------------------------------------------------
# lexicon
# ---------------------------------------------------------------------------------------------------------------

@dataclass
class Term:
    text: str            # normalised
    move: str | None     # move slug; None for an unresolved alias
    kind: str            # slug | name | alias | name_en | variant | node | number | unresolved
    key: str = ""        # sound key
    stem: str = ""
    candidates: list = field(default_factory=list)  # unresolved alias: candidate terms (research)
    phon: tuple = ()     # phonetic forms of a Portuguese-spelled term (pt_respell -> phon); () for other terms


@dataclass
class Candidate:
    move: str | None
    score: float
    term: str
    kind: str
    method: str          # exact | stem | fuzzy | phonetic
    name: str | None = None
    candidates: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v not in (None, [], "")} | {"score": round(self.score, 1)}


class Lexicon:
    def __init__(self, moves: list[dict], links: list[tuple[str, str]] | None = None,
                 unresolved: list[dict] | None = None, node_names: dict[str, list[str]] | None = None):
        self.moves = {m["slug"]: m for m in moves}
        self.terms: list[Term] = []
        self.exact: dict[str, list[Term]] = {}
        self.adjacent: dict[str, set[str]] = {}
        for a, b in links or []:
            if a and b:
                self.adjacent.setdefault(a, set()).add(b)
        seen: set[tuple[str, str | None]] = set()
        pt_names: set[str] = set()  # the current move's Portuguese name (it differs from the English one)

        def add(text: str | None, move: str | None, kind: str, cands: list | None = None) -> None:
            n = normalise(text)
            if not n or (n, move) in seen:
                return
            seen.add((n, move))
            pt = looks_portuguese(text) or (n in pt_names and kind != "number")
            ph = tuple(dict.fromkeys(phon(r) for r in pt_respell(text))) if pt else ()
            t = Term(n, move, kind, sound_key(n), stem_phrase(n), list(cands or []), ph)
            self.terms.append(t)
            self.exact.setdefault(n, []).append(t)
            for sp in spoken_numbers(n):
                add(sp, move, "number")

        for m in moves:
            s = m["slug"]
            nm = normalise(m.get("name") or m.get("display_name"))
            pt_names.clear()
            if nm and nm != normalise(m.get("name_en")) and not nm.isdigit():
                pt_names.add(nm)
            add(s.replace("-", " "), s, "slug")
            add(m.get("name") or m.get("display_name"), s, "name")
            add(m.get("name_en"), s, "name_en")
            for a in m.get("aliases") or []:
                add(a, s, "alias")
            for v in m.get("variants") or []:
                add(v, s, "variant")
            for nn in (node_names or {}).get(s, []):
                add(nn, s, "node")
        for u in unresolved or []:
            if self.resolve_exact(u.get("alias")) is None:
                add(u.get("alias"), None, "unresolved", u.get("candidates"))
        self._cache: dict[tuple[str, str | None], list[Candidate]] = {}

    # -- construction -------------------------------------------------------------------------------------------
    @classmethod
    def from_graph(cls, graph, vocabulary: str | Path | None = "default") -> "Lexicon":
        """From a moves.graph.Graph (or a graph.json dict). Unresolved aliases: the graph's own list plus the
        vocabulary research's (vocabulary=None to skip the research file)."""
        data = graph.data if hasattr(graph, "data") else graph
        nodes = {n["id"]: n for n in data["nodes"]}
        links = []
        for l in data["links"]:
            a, b = nodes.get(l["from"], {}).get("move"), nodes.get(l["to"], {}).get("move")
            links.append((a, b))
        if hasattr(graph, "via"):
            for (a, b) in graph.via:
                links.append((nodes.get(a, {}).get("move"), nodes.get(b, {}).get("move")))
        node_names: dict[str, list[str]] = {}
        for n in data["nodes"]:
            if n.get("move") and n.get("authored_name"):
                node_names.setdefault(n["move"], []).append(n["authored_name"])
        unresolved = list(data.get("unresolved_aliases") or [])
        if vocabulary is not None:
            try:
                from . import VOCABULARY_PATH
                vp = Path(VOCABULARY_PATH if vocabulary == "default" else vocabulary)
                if vp.exists():
                    vd = json.loads(vp.read_text(encoding="utf-8"))
                    for u in vd.get("unresolved_aliases") or []:
                        if isinstance(u, dict) and u.get("alias"):
                            c = [x.get("term") if isinstance(x, dict) else str(x) for x in u.get("candidates") or []]
                            unresolved.append({"alias": u["alias"], "candidates": [x for x in c if x]})
            except (OSError, ValueError):
                pass
        return cls(data["moves"], links, unresolved, node_names)

    # -- queries ------------------------------------------------------------------------------------------------
    def name(self, move: str | None) -> str | None:
        m = self.moves.get(move or "")
        return (m.get("name") or m.get("display_name") or move) if m else None

    def resolve_exact(self, text: str | None) -> str | None:
        for t in self.exact.get(normalise(text), []):
            if t.move:
                return t.move
        return None

    def linked(self, prev: str | None, move: str | None) -> bool:
        return bool(prev and move and move in self.adjacent.get(prev, ()))

    def candidates(self, phrase: str, prev: str | None = None, limit: int = 6) -> list[Candidate]:
        """Best candidates (one per move / unresolved alias), score-sorted."""
        p = normalise(phrase)
        if not p:
            return []
        ck = (p, prev)
        if ck in self._cache:
            return self._cache[ck]
        best: dict[str, Candidate] = {}
        pkey, pstem, pph = sound_key(p), stem_phrase(p), phon(p)
        plen = len(p.replace(" ", ""))
        for t in self.terms:
            tlen = len(t.text.replace(" ", ""))
            if p == t.text:
                score, method = 100.0, "exact"
            elif pstem == t.stem or pstem == t.text or p == t.stem:
                score, method = 96.0, "stem"
            elif t.kind == "number":
                continue  # spoken numbers ("one eighty") only match exactly
            elif t.text.startswith(p + " ") and t.move:
                score, method = PREFIX, "prefix"  # "howl" -> Howl Up / Howl Down (ambiguous on purpose)
            elif min(plen, tlen) <= SHORT or not (0.5 <= plen / max(1, tlen) <= 2.0):
                # a short Portuguese-spelled name heard as an English spelling ("Piau" for Pião): phonetic forms only,
                # similar lengths, >= 4 letters each
                if not (t.phon and len(pph) >= 4 and min(plen, tlen) >= 4 and 0.75 <= plen / max(1, tlen) <= 1.34):
                    continue
                score, method = min(0.97 * max(ratio(pph, f) for f in t.phon), 99.0), "phonetic"
            else:
                s = string_score(p, t.text)
                method = "fuzzy"
                if s >= 70 and pkey and pkey == t.key and len(pkey) >= 3:
                    sp = 0.5 * s + 50.0 + 6.0
                    if sp > s:
                        s, method = sp, "phonetic"
                if t.phon and len(pph) >= 4:  # an English rendering of a Portuguese name
                    sp = 0.97 * max(ratio(pph, f) for f in t.phon)
                    if sp > s:
                        s, method = sp, "phonetic"
                score = min(s, 99.0)
            if score < CONSIDER - 15:
                continue
            if t.move and self.linked(prev, t.move) and score < 100:
                score = min(99.0, score + LINK_BONUS)
            k = t.move or f"?{t.text}"
            if k not in best or score > best[k].score:
                best[k] = Candidate(t.move, score, t.text, t.kind, method, self.name(t.move), list(t.candidates))
        out = sorted(best.values(), key=lambda c: (-c.score, c.move is None, c.move or ""))[:limit]
        self._cache[ck] = out
        return out

    def describe(self, move: str) -> str:
        m = self.moves.get(move, {})
        al = [a for a in (m.get("aliases") or []) if a][:6]
        d = m.get("name") or move
        if m.get("name_en") and m.get("name_en") != d:
            d += f" (English: {m['name_en']})"
        if al:
            d += "; also called " + ", ".join(al)
        if m.get("description"):
            d += ". " + str(m["description"])[:200]
        return d


# ---------------------------------------------------------------------------------------------------------------
# TypeSafe judgment (optional, text only)
# ---------------------------------------------------------------------------------------------------------------

@dataclass
class Judgment:
    move: str | None
    confidence: float
    names_a_move: float | None = None
    choice: str | None = None
    choice_confidence: float | None = None
    refused: bool = False
    error: str | None = None


def judge_phrase(phrase: str, candidates: list[Candidate], lex: Lexicon, utterance: str | None = None,
                 previous: str | None = None, client=None) -> Judgment:
    """Which candidate move does a misheard narrated phrase name? Stated-argument pattern: a Noul 'does the phrase
    name a dance move at all' and a Choice among the candidates (+ 'none_of_these'); the decision uses the weakest
    of the two and refuses below REFUSE (0.4). Sends only the words and the candidate names - never audio/video."""
    moves = [c.move for c in candidates if c.move][:6]
    for c in candidates:
        for x in c.candidates:  # unresolved alias -> its research candidates, when they are graph moves
            m = lex.resolve_exact(x)
            if m and m not in moves:
                moves.append(m)
    if not moves:
        return Judgment(None, 0.0, refused=True, error="no candidate moves")
    try:
        from typesafe_sdk import Choice, Noul
        if client is None:
            from ..judgments import client as ts_client
            client = ts_client()
        state = {"phrase": phrase, "utterance": utterance or phrase,
                 "context": "A dancer narrates a Brazilian zouk dance video aloud; speech recognition wrote this. "
                            "Portuguese move names are often misheard as English words.",
                 "previous_move": lex.name(previous) if previous else None}
        q = {
            "names_a_move": Noul(instructions="Does `phrase` name one specific partner-dance move (possibly misheard "
                                              "by speech recognition), rather than filler talk?"),
            "which_move": Choice(
                instructions="Assuming `phrase` names a zouk dance move, which of these moves is it? Consider how "
                             "speech recognition mishears Portuguese words.",
                criteria={**{m: lex.describe(m) for m in moves},
                          "none_of_these": "a different move that is not in this list"}),
        }
        r = client.system_one(state=state, questions=q)
        p = float(r.nouls["names_a_move"].noul)
        ch = r.choices["which_move"]
        choice, cc = str(ch.choice), float(ch.confidence)
    except Exception as e:  # noqa: BLE001 - offline / no key / API error: the phrase stays unknown
        return Judgment(None, 0.0, refused=True, error=f"{type(e).__name__}: {e}"[:300])
    weakest = min(p, cc)
    if choice == "none_of_these" or choice not in moves or weakest < REFUSE:
        return Judgment(None, weakest, p, choice, cc, refused=True)
    return Judgment(choice, weakest, p, choice, cc)


# ---------------------------------------------------------------------------------------------------------------
# utterances -> events
# ---------------------------------------------------------------------------------------------------------------

def tokens_of(words: list[dict]) -> list[tuple[str, int]]:
    """(normalised token, word index) - a recognised 'word' may hold several tokens ('vira-dinha', "that's")."""
    out = []
    for i, w in enumerate(words):
        for tok in normalise(w.get("word")).split():
            out.append((tok, i))
    return out


_BREAK = re.compile(r"[,.;:!?]\W*$")


def breaks_of(words: list[dict], toks: list[tuple[str, int]]) -> set[int]:
    """Token positions after which a phrase may not continue: the word ended with , . ; : ! ? (Whisper punctuates a
    narrated list: "Open Break, Viradinha, Corridor,")."""
    ends = {i for i, w in enumerate(words) if _BREAK.search(str(w.get("word") or ""))}
    return {j for j, (_, wi) in enumerate(toks) if wi in ends and (j + 1 == len(toks) or toks[j + 1][1] != wi)}


def _edge_penalty(toks: list[str], lex: Lexicon) -> float | None:
    """None when the phrase cannot name a move (all fillers); else the score penalty for filler edges."""
    if " ".join(toks) in lex.exact:
        return 0.0
    if all(t in FILLERS for t in toks):
        return None
    return EDGE_PENALTY * ((toks[0] in FILLERS) + (len(toks) > 1 and toks[-1] in FILLERS))


def _control_of(toks: list[str]) -> list[str]:
    """Control actions when the tokens are only control phrases (+ fillers), else []."""
    rest = " " + " ".join(toks) + " "
    acts = []
    for phrase in sorted(CONTROL, key=len, reverse=True):
        while f" {phrase} " in rest:
            acts.append(CONTROL[phrase])
            rest = rest.replace(f" {phrase} ", " ", 1)
    return acts if acts and all(t in FILLERS for t in rest.split()) else []


def match_tokens(toks: list[tuple[str, int]], lex: Lexicon, prev: str | None,
                 breaks: set[int] | None = None) -> list[dict]:
    """Non-overlapping matched phrases in one token list: [{i0, i1 (token span), phrase, cands}]; a phrase never
    continues past a token position in `breaks` (punctuation)."""
    found = []
    words = [t for t, _ in toks]
    breaks = breaks or set()
    for i in range(len(words)):
        for k in range(1, min(MAX_NGRAM, len(words) - i) + 1):
            if k > 1 and (i + k - 2) in breaks:
                break
            seg = words[i:i + k]
            pen = _edge_penalty(seg, lex)
            if pen is None:
                continue
            phrase = " ".join(seg)
            cands = lex.candidates(phrase, prev)
            if pen:
                cands = [Candidate(c.move, c.score - pen, c.term, c.kind, c.method, c.name, c.candidates)
                         for c in cands]
            if cands and cands[0].score >= CONSIDER:
                found.append({"i0": i, "i1": i + k, "phrase": phrase, "cands": cands})
    found.sort(key=lambda f: (-f["cands"][0].score, -(f["i1"] - f["i0"]), f["i0"]))
    used: set[int] = set()
    out = []
    for f in found:
        span = set(range(f["i0"], f["i1"]))
        if span & used:
            continue
        used |= span
        out.append(f)
    return sorted(out, key=lambda f: f["i0"])


def decide(cands: list[Candidate]) -> tuple[str, float]:
    """('matched' | 'close' | 'ambiguous', margin) for a candidate list (top first)."""
    if not cands:
        return "ambiguous", 0.0
    top = cands[0]
    second = next((c for c in cands[1:] if c.move != top.move), None)
    margin = top.score - second.score if second else 100.0
    if top.move and top.score >= ACCEPT and (margin >= MARGIN or top.score >= 100):
        return "matched", margin
    if top.move and top.score >= ACCEPT:
        return "close", margin
    return "ambiguous", margin


def _merge_cands(*lists) -> list[Candidate]:
    best: dict[str, Candidate] = {}
    for c in (c for lst in lists for c in lst):
        k = c.move or f"?{c.term}"
        if k not in best or c.score > best[k].score:
            best[k] = c
    return sorted(best.values(), key=lambda c: (-c.score, c.move is None, c.move or ""))


def consolidate_chunks(toks: list[tuple[str, int]], matches: list[dict], breaks: set[int], lex: Lexicon,
                       prev: str | None) -> list[dict]:
    """Punctuated transcripts (Whisper): each chunk between punctuation marks with <= MAX_NGRAM content words is one
    narrated item. Accepted matches inside an item with content words left over become 'partial' (the item names a
    variant the graph lacks - "Tilted Turn" -> turn, flagged for review); an item with only weak matches becomes ONE
    phrase of all its content words ("Shadow Position", not "position"), candidates merged."""
    if not breaks or not matches:
        return matches
    chunks, j0 = [], 0
    for j in range(len(toks)):
        if j in breaks or j == len(toks) - 1:
            chunks.append((j0, j + 1))
            j0 = j + 1
    out = []
    for c0, c1 in chunks:
        ms = [f for f in matches if c0 <= f["i0"] < c1]
        content = [j for j in range(c0, c1) if toks[j][0] not in FILLERS or toks[j][0] in lex.exact]
        if not ms:  # an item naming nothing the graph knows: unknown, like a stand-alone 1-4 word utterance
            if 1 <= len(content) <= MAX_NGRAM and not _control_of([toks[j][0] for j in range(c0, c1)]) and                     not any(toks[j][0] in CONTROL for j in content):
                phrase = " ".join(toks[j][0] for j in content)
                out.append({"i0": content[0], "i1": content[-1] + 1, "phrase": phrase,
                            "cands": lex.candidates(phrase, prev), "unmatched": True})
            continue
        if not content or len(content) > MAX_NGRAM:
            out += ms
            continue
        acc = [f for f in ms if decide(f["cands"])[0] != "ambiguous"]
        covered = {j for f in acc for j in range(f["i0"], f["i1"])}
        leftover = [toks[j][0] for j in content if j not in covered]
        phrase = " ".join(toks[j][0] for j in content)
        if acc:
            for f in acc:
                out.append({**f, "partial": {"phrase": phrase, "leftover": leftover, "i0": content[0],
                                             "i1": content[-1] + 1}} if leftover else f)
        elif len(ms) == 1 and ms[0]["i0"] == content[0] and ms[0]["i1"] == content[-1] + 1:
            out.append(ms[0])
        else:
            cands = _merge_cands(lex.candidates(phrase, prev), *[f["cands"] for f in ms])
            out.append({"i0": content[0], "i1": content[-1] + 1, "phrase": phrase, "cands": cands})
    return sorted(out, key=lambda f: f["i0"])


def _word_span(words: list[dict], idx: list[int]) -> tuple[float, float, float | None]:
    ws = [words[i] for i in sorted(set(idx))]
    probs = [float(w["prob"]) for w in ws if w.get("prob") is not None]
    return float(ws[0]["start"]), float(ws[-1]["end"]), (min(probs) if probs else None)


def map_utterances(utterances: list[dict], lex: Lexicon, judge: bool = False, client=None,
                   prev: str | None = None) -> list[dict]:
    """Utterances [{start, end, words:[{word, start, end, prob}], alternatives?:[{text, prob, words}]}] (audio
    seconds) -> events in audio order. Event: {kind: move|unknown|control|ignored, a0, a1 (audio seconds of the
    phrase), phrase, transcript (the utterance text), move, name, score, method, status, confidence (match x asr
    probability), word_prob, candidates, control, judgment}. Undo / repeat are already applied (an undone event is
    kept with kind 'undone' so the transcript display can show it)."""
    events: list[dict] = []

    def last_move_event() -> dict | None:
        return next((e for e in reversed(events) if e["kind"] in ("move", "unknown")), None)

    for ui, u in enumerate(utterances):
        words = u.get("words") or []
        if not words:  # the recogniser only offered low-confidence guesses here
            prev_move = (last_move_event() or {}).get("move") or prev
            alt_ev = _from_alternatives(u, lex, prev_move, u.get("text") or "", ui)
            if alt_ev:
                events.append(alt_ev)
            continue
        toks = tokens_of(words)
        text = " ".join(w.get("word", "") for w in words).strip()
        content = [t for t, _ in toks if t not in FILLERS or t in CONTROL]
        prev_move = (last_move_event() or {}).get("move") or prev
        brk = breaks_of(words, toks)
        matches = consolidate_chunks(toks, match_tokens(toks, lex, prev_move, brk), brk, lex, prev_move)
        core_n = len([t for t in content if t not in CONTROL])
        if core_n > MAX_NGRAM and not brk:  # long talk: only confident move names count, weak look-alikes are chatter
            matches = [f for f in matches if decide(f["cands"])[0] != "ambiguous"]
        elif core_n > MAX_NGRAM:  # a punctuated list: weak matches count only as whole short items
            matches = [f for f in matches if decide(f["cands"])[0] != "ambiguous" or
                       (f["i0"] == 0 or (f["i0"] - 1) in brk) and (f["i1"] == len(toks) or (f["i1"] - 1) in brk)]
        u_events: list[dict] = []
        if matches and all(decide(f["cands"])[0] == "ambiguous" for f in matches) and u.get("alternatives"):
            alt_ev = _from_alternatives(u, lex, prev_move, text, ui)  # the recogniser's own vocabulary guess wins
            if alt_ev:                                                  # over a weak look-alike of its dictation
                alt_ev["dictation_candidates"] = [c.to_dict() for c in matches[0]["cands"][:3]]
                events.append(alt_ev)
                continue
        if matches:
            lead_all = [t for t, _ in toks[:matches[0]["i0"]]]
            lead = " ".join(t for t in lead_all if t not in FILLERS)
            if lead_all and ("undo" in _control_of(lead_all) or lead in LEADING_UNDO
                             or any(lead.endswith(x) for x in LEADING_UNDO)):
                e = last_move_event()
                if e:
                    e["kind"] = "undone"
                a0 = float(words[0]["start"])
                u_events.append({"kind": "control", "control": "undo", "a0": a0, "a1": a0, "phrase": lead,
                                 "utterance": ui})
            for f in matches:
                if not f["cands"]:  # an unmatched narrated item with no candidates at all
                    idx = [toks[j][1] for j in range(f["i0"], f["i1"])]
                    a0, a1, wp = _word_span(words, idx)
                    u_events.append({"kind": "unknown", "a0": a0, "a1": a1, "phrase": f["phrase"], "transcript": text,
                                     "move": None, "name": None, "status": "unknown", "method": None, "score": 0.0,
                                     "confidence": None, "word_prob": wp, "utterance": ui, "candidates": []})
                    continue
                part = f.get("partial")
                idx = [toks[j][1] for j in range(*(part["i0"], part["i1"]) if part else (f["i0"], f["i1"]))]
                a0, a1, wp = _word_span(words, idx)
                cands = f["cands"]
                status, margin = decide(cands)
                top = cands[0]
                ev = {"kind": "move", "a0": a0, "a1": a1, "phrase": f["phrase"], "transcript": text,
                      "word_prob": wp, "utterance": ui, "candidates": [c.to_dict() for c in cands[:4]],
                      "score": round(top.score, 1), "margin": round(margin, 1)}
                asr = 1.0 if wp is None else max(0.2, min(1.0, wp))
                if part:  # "Tilted Turn": the item names more than the matched move - keep it, flagged
                    full = _merge_cands(lex.candidates(part["phrase"], prev_move), cands)
                    ev.update(move=top.move, method=top.method, status="partial", phrase=part["phrase"],
                              matched_phrase=f["phrase"], leftover=part["leftover"],
                              candidates=[c.to_dict() for c in full[:4]],
                              confidence=round(PARTIAL_FACTOR * top.score / 100 * asr, 3))
                elif status == "matched":
                    ev.update(move=top.move, method=top.method, status="matched",
                              confidence=round(top.score / 100 * asr, 3))
                elif status == "close" and not judge:
                    ev.update(move=top.move, method=top.method, status="close",
                              confidence=round(0.8 * top.score / 100 * asr, 3))
                else:
                    j = judge_phrase(f["phrase"], cands, lex, text, prev_move, client) if judge else None
                    if j is not None:
                        ev["judgment"] = {k: v for k, v in asdict(j).items() if v is not None}
                    if j is not None and j.move:
                        ev.update(move=j.move, method="judged", status="judged",
                                  confidence=round(j.confidence * asr, 3))
                    else:
                        ev.update(kind="unknown", move=None, method=top.method, status="unknown",
                                  confidence=None)
                if top.move is None and ev.get("move") is None:  # an unresolved alias ("heard, not identified")
                    ev["unresolved_alias"] = top.term
                ev["name"] = lex.name(ev.get("move"))
                u_events.append(ev)
                prev_move = ev.get("move") or prev_move
        else:
            acts = _control_of([t for t, _ in toks])
            a0, a1 = float(words[0]["start"]), float(words[-1]["end"])
            if acts:
                for act in acts:
                    e = last_move_event()
                    if act == "undo":
                        if e:
                            e["kind"] = "undone"
                        u_events.append({"kind": "control", "control": "undo", "a0": a0, "a1": a1,
                                         "phrase": text, "utterance": ui})
                    elif act == "repeat" and e and e.get("move"):
                        u_events.append({"kind": "move", "a0": a0, "a1": a1, "phrase": text, "transcript": text,
                                         "move": e["move"], "name": lex.name(e["move"]), "method": "repeat",
                                         "status": "matched", "score": 100.0, "confidence": e.get("confidence"),
                                         "word_prob": None, "utterance": ui, "candidates": []})
                    else:
                        u_events.append({"kind": "control", "control": act, "a0": a0, "a1": a1, "phrase": text,
                                         "utterance": ui, "note": "nothing to repeat"})
            else:
                alt_ev = _from_alternatives(u, lex, prev_move, text, ui)
                if alt_ev:
                    u_events.append(alt_ev)
                else:
                    core = [t for t in content if t not in CONTROL]
                    if 1 <= len(core) <= MAX_NGRAM:
                        phrase = " ".join(core)
                        cands = lex.candidates(phrase, prev_move)
                        probs = [float(w["prob"]) for w in words if w.get("prob") is not None]
                        u_events.append({"kind": "unknown", "a0": a0, "a1": a1, "phrase": phrase, "transcript": text,
                                         "move": None, "name": None, "status": "unknown", "method": None,
                                         "score": round(cands[0].score, 1) if cands else 0.0, "confidence": None,
                                         "word_prob": min(probs) if probs else None, "utterance": ui,
                                         "candidates": [c.to_dict() for c in cands[:4]]})
                    elif core:
                        u_events.append({"kind": "ignored", "a0": a0, "a1": a1, "phrase": text, "transcript": text,
                                         "utterance": ui})
        events.extend(u_events)
    return events


def _from_alternatives(u: dict, lex: Lexicon, prev: str | None, text: str, ui: int) -> dict | None:
    """A move from the recogniser's alternative hypotheses (only exact / stem / strong matches)."""
    for alt in u.get("alternatives") or []:
        aw = alt.get("words") or []
        if not aw:
            continue
        toks = tokens_of(aw)
        for f in match_tokens(toks, lex, prev, breaks_of(aw, toks)):
            top = f["cands"][0]
            if not top.move or top.score < ACCEPT:
                continue
            idx = [toks[j][1] for j in range(f["i0"], f["i1"])]
            a0, a1, _ = _word_span(aw, idx)
            p = float(alt.get("prob") or 0.0)
            if p < ALT_MIN_PROB:
                continue
            cap = float(alt.get("conf_cap") or ALT_MAX_CONF)
            return {"kind": "move", "a0": a0, "a1": a1, "phrase": f["phrase"], "transcript": text,
                    "move": top.move, "name": lex.name(top.move), "method": "alternative", "status": "alternative",
                    "score": round(top.score, 1), "confidence": round(min(cap, p) * top.score / 100, 3),
                    "word_prob": p, "utterance": ui, "alternative": alt.get("text"),
                    "candidates": [c.to_dict() for c in f["cands"][:4]]}
    return None


def summary(events: list[dict]) -> dict:
    kinds = [e["kind"] for e in events]
    return {"moves": kinds.count("move"), "unknown": kinds.count("unknown"), "control": kinds.count("control"),
            "undone": kinds.count("undone"), "ignored": kinds.count("ignored"),
            "sequence": [e.get("move") or f"unknown: {e.get('phrase')}" for e in events
                         if e["kind"] in ("move", "unknown")]}
