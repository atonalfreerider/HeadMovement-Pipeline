"""Transcript fusion: a ROVER-style consensus over every (phone, engine) hypothesis of the class (local, numpy, no model, no pod).

Seven phones heard one class and four ASR engines (faster-whisper large-v3 and large-v3-turbo, parakeet-tdt-0.6b-v3, canary-1b-v2) wrote
each phone's audio down, so the six phones that have a clock give 24 hypotheses of the same speech, every word timed on the master clock
(`sources.Clock`). This module turns them into ONE hypothesis with, for every word, the vote that produced it.

Core (pure python / numpy, unit-tested with synthetic hypotheses in tests/test_speech_fuse.py)

  1. canonical tokens      `pieces`: case / accents / punctuation / apostrophes dropped, spelling variants unified (ok = okay, uhm = um),
                           colloquial contractions expanded (gonna = going to), fillers marked; `merge_compounds` joins "counter balance" to
                           "counterbalance" in a hypothesis when other hypotheses at the same moment write the compound.
  2. confusion network     `build_network`: the hypotheses are aligned one by one into a word-transition network (ROVER). A hypothesis token
                           aligns to a slot with a dynamic programme whose substitution cost is the expected string / sound distance to the
                           slot's candidate words (0 = same word, 1 = unrelated), plus a TIME PENALTY (words more than 0.8 s from the slot's
                           median time pay up to `pen_lambda`, so a repeated "the" cannot slide across two seconds of speech) and a gap cost
                           for insertions / deletions. Order bias is reduced by re-aligning every hypothesis against the network built
                           from all the others (`refine_rounds`).
  3. voting                `vote`: per slot, each candidate word gets the summed weight of the hypotheses that wrote it; hypotheses that
                           have audio there but no word vote "nothing" (a word can be voted OUT); a phone that did not record yet / any more
                           does not vote. The winner, its share of all voting weight, its margin to the runner-up and the alternatives are kept.
  4. weights               `fit_reliability`: a logistic model of P(this token agrees with the vote of all the OTHER hypotheses) from the
                           engine, the engine's own word confidence (whisper probability, parakeet max-softmax; canary has none), the phone's
                           local SNR and its distance proxy (level relative to the loudest phone, fuse_levels.py), the word's length / duration.
                           The vote weight is that probability (power `cal_gamma`). `weight_mode` "uniform" / "prior" / "cal" select the ablations
                           measured in fuse_eval.py.

Everything is deterministic. Run it with `python -m dancecap.speech fuse` (consensus_v0.json, disagreement batches) and `fuse-eval`
(fusion_metrics.json, leave-one-source-out); the post-processing (speakers, turns, events, spans) lives in fuse_post.py / fuse_batches.py.
"""
from __future__ import annotations

import bisect
import itertools
import math
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .texts import norm_token

# ------------------------------------------------------------------------------------------------ engines

VOTING = ["whisper-large-v3:en", "whisper-large-v3-turbo:en", "parakeet-tdt-0.6b-v3", "canary-1b-v2"]   # :auto is :en with language-id noise, :pt translates
ENGINE_PRIOR = {"whisper-large-v3": 1.0, "whisper-large-v3-turbo": 0.9, "parakeet-tdt-0.6b-v3": 1.0, "canary-1b-v2": 0.9}
# how far a word time of each engine is trusted for the slot / word times: parakeet (TDT frames) and canary (80 ms frames, start only) are tight,
# whisper's cross-attention times smear after pauses (checked on the opening: "is" 2.32-3.78 s for a word said at 3.8 s)
ENGINE_TIME_W = {"whisper-large-v3": 1.0, "whisper-large-v3-turbo": 1.0, "parakeet-tdt-0.6b-v3": 3.0, "canary-1b-v2": 2.0}
FAMILY = {"whisper-large-v3": "whisper", "whisper-large-v3-turbo": "whisper", "parakeet-tdt-0.6b-v3": "parakeet", "canary-1b-v2": "canary"}
ENGINE_SHORT = {"whisper-large-v3": "wl3", "whisper-large-v3-turbo": "wl3t", "parakeet-tdt-0.6b-v3": "pk", "canary-1b-v2": "cn"}
ENGINE_IDX = {e: i for i, e in enumerate(ENGINE_PRIOR)}


@dataclass
class Params:
    gap: float = 0.6               # cost of an insertion / deletion (a substitution of unrelated words costs 1.0)
    exact_bonus: float = 0.15      # reward (negative cost) per unit of slot mass that is exactly the same word
    pen_lambda: float = 0.5        # time penalty at its maximum
    pen_tol: float = 0.8           # seconds between a token and the slot's median time that cost nothing
    pen_full: float = 3.0          # seconds at which the penalty is complete
    band_s: float = 5.0            # a token can only be aligned to slots within +-band_s of its own time
    refine_rounds: int = 2
    compounds: bool = True
    weight_mode: str = "cal"       # uniform | prior | cal
    cal_gamma: float = 1.0
    family_balance: float = 0.0    # exponent: weight / (engines of the family)**fb; whisper large-v3 and turbo are one family
    null_gamma: float = 1.0        # weight of a "nothing" vote relative to the hypothesis' local reliability
    cover_margin: float = 0.25
    min_share_keep: float = 0.5    # a non-null candidate is kept when it beats "nothing" ... see vote()
    overlap_keep_share: float = 1.01   # > 1 = off: keeping a minority word in an overlap window kept hallucinations on the laugh in the opening
    cal_l2: float = 2.0
    cal_min_loo_share: float = 0.5


# ------------------------------------------------------------------------------------------------ tokens

_EQUIV = {"ok": "okay", "okey": "okay", "umm": "um", "uhm": "um", "erm": "um", "hm": "um", "hmm": "um", "mm": "um", "mmm": "um", "mhm": "um",
          "uhhuh": "um", "uhh": "uh", "er": "uh", "eh": "uh", "ehm": "um", "allright": "alright", "yea": "yeah", "ya": "yeah"}
_EXPAND = {"gonna": ("going", "to"), "wanna": ("want", "to"), "gotta": ("got", "to"), "kinda": ("kind", "of"), "sorta": ("sort", "of"),
           "lemme": ("let", "me"), "gimme": ("give", "me"), "dunno": ("dont", "know"), "outta": ("out", "of"), "tryna": ("trying", "to"),
           "cuz": ("because",), "coz": ("because",), "cannot": ("can", "not")}
FILLERS = {"um", "uh"}
_DIGITS = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
           "10": "ten", "11": "eleven", "12": "twelve", "13": "thirteen", "14": "fourteen", "15": "fifteen", "16": "sixteen", "17": "seventeen",
           "18": "eighteen", "19": "nineteen", "20": "twenty"}
_PUNCT_END = re.compile(r"[\W_]+$", re.UNICODE)
_PUNCT_START = re.compile(r"^[\W_]+", re.UNICODE)


def trailing_punct(raw: str) -> str:
    """The sentence punctuation class a surface ends with: '', ',', '.', '?', '!' or '...'."""
    m = re.search(r"([.,!?;:…\-–—]+)[\"')\]]*$", raw.strip())
    if not m:
        return ""
    p = m.group(1)
    if "?" in p:
        return "?"
    if "!" in p:
        return "!"
    if p.endswith("...") or p == "…" or p.endswith("…"):
        return "..."
    if "." in p:
        return "."
    return ","


def bare(raw: str) -> str:
    """The surface without leading / trailing punctuation (inner apostrophes and hyphens stay)."""
    s = _PUNCT_START.sub("", raw.strip())
    s = _PUNCT_END.sub("", s)
    return s


def pieces(raw: str) -> list[tuple[str, str]]:
    """[(canonical norm, display text)] of one raw ASR word: usually one piece, none for pure punctuation, two for "gonna" / "no 'cause"."""
    out: list[tuple[str, str]] = []
    for part in raw.split():
        n = norm_token(part)
        if not n:
            continue
        if n == "cause" and part.lstrip().startswith(("'", "’")):
            n = "because"
        if n in _EXPAND:
            out += [(w, w) for w in _EXPAND[n]]
            continue
        n = _DIGITS.get(n, _EQUIV.get(n, n))
        out.append((n, bare(part) or part))
    return out


@dataclass
class Tok:
    text: str                  # display surface (no outer punctuation)
    norm: str                  # canonical comparison form
    t0: float                  # master seconds
    t1: float
    conf: float | None = None  # the engine's own word confidence
    punct: str = ""            # trailing punctuation class
    hid: int = -1
    idx: int = 0
    filler: bool = False
    snr: float = 10.0
    rel: float = 0.0
    w: float = 1.0             # vote weight (set by assign_weights)
    raw: str = ""

    @property
    def tk(self) -> float:
        """The time key: the centre of the word, with a smeared duration (whisper after a pause) capped at 1 s."""
        return self.t0 + min(self.t1 - self.t0, 1.0) / 2.0


@dataclass
class Hyp:
    hid: int
    cam: str
    engine: str                # file base name: whisper-large-v3 | whisper-large-v3-turbo | parakeet-tdt-0.6b-v3 | canary-1b-v2
    toks: list[Tok]
    cover: tuple[float, float]  # master seconds the phone recorded
    key: str = ""              # "<engine>:<pass>@<cam>"
    w0: float = 1.0

    @property
    def short(self) -> str:
        return f"{ENGINE_SHORT.get(self.engine, self.engine)}@{self.cam}"


def make_toks(words: list[dict], hid: int = -1) -> list[Tok]:
    """Tok list from [{word,start,end,prob}] (times already in the target clock). Multi-piece words split their time by letters."""
    out: list[Tok] = []
    for w in words:
        raw = w.get("word") or w.get("text") or ""
        ps = pieces(raw)
        if not ps:
            continue
        t0, t1 = float(w["start"]), float(w["end"])
        tot = sum(len(n) for n, _ in ps) or 1
        a = t0
        for k, (n, disp) in enumerate(ps):
            b = t1 if k == len(ps) - 1 else a + (t1 - t0) * len(n) / tot
            out.append(Tok(disp, n, a, b, w.get("prob"), trailing_punct(raw) if k == len(ps) - 1 else "", hid, 0, n in FILLERS, raw=raw))
            a = b
    for i, t in enumerate(out):
        t.idx = i
    return out


def merge_compounds(hyps: list[Hyp], max_gap: float = 0.45, bucket: float = 2.0, min_len: int = 5) -> int:
    """Join adjacent tokens of a hypothesis ("counter" "balance") into the compound ("counterbalance") when at least two OTHER hypotheses write
    that compound within 2 s: data-driven, so it needs no word list. Returns the number of merges."""
    seen: dict[tuple[str, int], set[int]] = defaultdict(set)
    for h in hyps:
        for t in h.toks:
            seen[(t.norm, int(t.tk // bucket))].add(h.hid)
    n = 0
    for h in hyps:
        out: list[Tok] = []
        i = 0
        toks = h.toks
        while i < len(toks):
            done = False
            for k in (3, 2):
                if i + k <= len(toks):
                    grp = toks[i: i + k]
                    if any(g.filler for g in grp) or any(grp[j + 1].t0 - grp[j].t1 > max_gap for j in range(k - 1)):
                        continue
                    cat = "".join(g.norm for g in grp)
                    if len(cat) < min_len:
                        continue
                    b = int(grp[0].tk // bucket)
                    others = set()
                    for bb in (b - 1, b, b + 1):
                        others |= seen.get((cat, bb), set())
                    others.discard(h.hid)
                    if len(others) >= 2 and not any(g.norm == cat for g in grp):
                        confs = [g.conf for g in grp if g.conf is not None]
                        out.append(Tok(" ".join(g.text for g in grp), cat, grp[0].t0, grp[-1].t1, min(confs) if confs else None, grp[-1].punct, h.hid,
                                       0, False, raw=" ".join(g.raw for g in grp)))
                        i += k
                        n += 1
                        done = True
                        break
            if not done:
                out.append(toks[i])
                i += 1
        for j, t in enumerate(out):
            t.idx = j
        h.toks = out
    return n


# ------------------------------------------------------------------------------------------------ string / sound similarity

_lev_cache: dict[tuple[str, str], int] = {}


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    key = (a, b)
    r = _lev_cache.get(key)
    if r is not None:
        return r
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    _lev_cache[key] = prev[-1]
    return prev[-1]


_PH = [("ph", "f"), ("ck", "k"), ("gh", ""), ("kn", "n"), ("wr", "r"), ("wh", "w"), ("qu", "kw"), ("x", "ks"), ("q", "k"), ("z", "s"), ("dg", "j"),
       ("tch", "ch"), ("ce", "se"), ("ci", "si"), ("cy", "sy"), ("c", "k"), ("gn", "n"), ("mb", "m")]


def phon(s: str) -> str:
    """A crude sound key: digraphs mapped, vowels (except a leading one) and h / w / y dropped, repeats collapsed. 'to' = 'too' = 'two', 'know' = 'no'."""
    for a, b in _PH:
        s = s.replace(a, b)
    if not s:
        return s
    out = [s[0]]
    for ch in s[1:]:
        if ch in "aeiouyhw" or (out and out[-1] == ch):
            continue
        out.append(ch)
    return "".join(out)


_sim_cache: dict[tuple[str, str], float] = {}


def sim(a: str, b: str) -> float:
    """Similarity of two canonical words, 1 = the same, 0 = unrelated."""
    if a == b:
        return 1.0
    key = (a, b) if a < b else (b, a)
    r = _sim_cache.get(key)
    if r is not None:
        return r
    la, lb = len(a), len(b)
    m = max(la, lb)
    s = 1.0 - levenshtein(a, b) / m
    pa, pb = phon(a), phon(b)
    if pa and pb:
        if pa == pb:
            s = max(s, 0.85)
        else:
            s = max(s, 0.7 * (1.0 - levenshtein(pa, pb) / max(len(pa), len(pb))))
    short, long_ = (a, b) if la <= lb else (b, a)
    if len(short) >= 3 and long_.startswith(short) and len(long_) - len(short) <= 3:      # plural / -ed / -ing
        s = max(s, 0.75)
    s = max(0.0, s)
    if s < 0.3:
        s = 0.0
    _sim_cache[key] = s
    return s


# ------------------------------------------------------------------------------------------------ the confusion network


def wmedian(vals: list[float], ws: list[float]) -> float:
    if len(vals) == 1:
        return vals[0]
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    tot = sum(ws)
    acc = 0.0
    for i in order:
        acc += ws[i]
        if acc >= 0.5 * tot:
            return vals[i]
    return vals[order[-1]]


class Slot:
    __slots__ = ("mem", "t", "cand", "wsum")

    def __init__(self):
        self.mem: dict[int, Tok] = {}
        self.t = 0.0
        self.cand: dict[str, float] = {}
        self.wsum = 0.0

    def add(self, h: Hyp, tok: Tok, w: float) -> None:
        self.mem[h.hid] = tok
        self.cand[tok.norm] = self.cand.get(tok.norm, 0.0) + w
        self.wsum += w
        self._time(hyp_engine[0])

    def _time(self, engines: dict[int, str]) -> None:
        vals = [t.tk for t in self.mem.values()]
        ws = [ENGINE_TIME_W.get(engines.get(hid, ""), 1.0) for hid in self.mem]
        self.t = wmedian(vals, ws)

    def remove(self, h: Hyp, w: float) -> bool:
        tok = self.mem.pop(h.hid, None)
        if tok is None:
            return False
        self.cand[tok.norm] -= w
        if self.cand[tok.norm] <= 1e-9:
            del self.cand[tok.norm]
        self.wsum -= w
        if self.mem:
            self._time(hyp_engine[0])
        return True


hyp_engine: list[dict[int, str]] = [{}]       # hid -> engine, set by build_network (a module-level cell: Slot stays small)

_vocab: dict[str, int] = {}
_vocab_str: dict[int, str] = {}
_simrows: dict[int, dict[int, float]] = {}


def _vid(s: str) -> int:
    i = _vocab.get(s)
    if i is None:
        i = _vocab[s] = len(_vocab)
        _vocab_str[i] = s
    return i


def align_hyp(slots: list[Slot], h: Hyp, hyps_in: list[Hyp], P: Params) -> list[Slot]:
    """Align the tokens of `h` into the slots (dynamic programme, banded in time) and return the new slot list; `h` is added to the slots."""
    toks = h.toks
    N, M = len(toks), len(slots)
    w = h.w0
    if M == 0:
        out = []
        for t in toks:
            s = Slot()
            s.add(h, t, w)
            out.append(s)
        return out
    if N == 0:
        return slots
    G = P.gap
    st = [s.t for s in slots]
    # eligible weight at each slot: the hypotheses already in the network that recorded at that moment
    lo = np.array([x.cover[0] - P.cover_margin for x in hyps_in])
    hi = np.array([x.cover[1] + P.cover_margin for x in hyps_in])
    ww = np.array([x.w0 for x in hyps_in])
    sta = np.array(st)
    Wel = ((sta[:, None] >= lo[None, :]) & (sta[:, None] <= hi[None, :])).astype(float) @ ww
    Wel = np.maximum(Wel, 1e-9)
    cand_lists: list[list[tuple[int, float]]] = []
    mgap: list[float] = []
    for j, s in enumerate(slots):
        cl = [(_vid(c), cw / Wel[j]) for c, cw in s.cand.items()]
        cand_lists.append(cl)
        mgap.append(max(0.0, 1.0 - sum(m for _, m in cl)))
    inside = [(h.cover[0] - P.cover_margin) <= t <= (h.cover[1] + P.cover_margin) for t in st]
    delc = [(1.0 - mgap[j]) * G if inside[j] else 0.0 for j in range(M)]
    # time bands
    env_max = list(itertools.accumulate(st, max))
    env_min = list(itertools.accumulate(reversed(st), min))[::-1]
    a_list: list[int] = []
    b_list: list[int] = []
    pa, pb = 0, 0
    for i, t in enumerate(toks):
        ti = t.tk
        a = bisect.bisect_left(env_max, ti - P.band_s)
        b = bisect.bisect_right(env_min, ti + P.band_s)
        a = max(pa, min(a, pb if i else a))
        b = max(pb, b)
        if i == 0:
            a = max(0, a)
        a_list.append(a)
        b_list.append(min(M, b))
        pa, pb = a, min(M, b)
    b_list[-1] = M
    for i in range(1, N):
        a_list[i] = min(a_list[i], b_list[i - 1])
    INF = 1e18
    # row 0
    prev = [INF] * (M + 1)
    prev[0] = 0.0
    for c in range(1, b_list[0] + 1):
        prev[c] = prev[c - 1] + delc[c - 1]
    bps: list[tuple[int, bytearray]] = []
    lam, tol, full, bonus = P.pen_lambda, P.pen_tol, P.pen_full, P.exact_bonus
    span = full - tol
    for i in range(N):
        t = toks[i]
        ti = t.tk
        ia = _vid(t.norm)
        simrow = _simrows.setdefault(ia, {})
        a, b = a_list[i], b_list[i]
        cur = [INF] * (M + 1)
        bp = bytearray(b - a + 1)
        for c in range(a, b + 1):
            best = INF
            code = 0
            if c >= 1:
                pv = prev[c - 1]
                if pv < INF:
                    j = c - 1
                    td = abs(ti - st[j])
                    s = mgap[j] * G
                    for ib, m in cand_lists[j]:
                        v = simrow.get(ib)
                        if v is None:
                            v = sim(t.norm, _vocab_str[ib])
                            simrow[ib] = v
                        s += m * (1.0 - v)
                        if v == 1.0:
                            s -= bonus * m
                    if td > tol:
                        s += lam * (td - tol) / span if td < full else lam
                    best = pv + s
                    code = 1
            pu = prev[c]
            if pu < INF:
                v = pu + G
                if v < best:
                    best = v
                    code = 2
            if c > a:
                pl = cur[c - 1]
                if pl < INF:
                    v = pl + delc[c - 1]
                    if v < best:
                        best = v
                        code = 3
            cur[c] = best
            bp[c - a] = code
        bps.append((a, bp))
        prev = cur
    if prev[M] >= INF:
        raise RuntimeError("alignment failed: band too narrow")
    # traceback
    i, c = N, M
    ops: list[tuple[str, int, int]] = []
    while i > 0 or c > 0:
        if i == 0:
            ops.append(("d", -1, c - 1))
            c -= 1
            continue
        a, bp = bps[i - 1]
        code = bp[c - a] if a <= c < a + len(bp) else 0
        if code == 1:
            ops.append(("m", i - 1, c - 1))
            i -= 1
            c -= 1
        elif code == 2:
            ops.append(("i", i - 1, -1))
            i -= 1
        elif code == 3:
            ops.append(("d", -1, c - 1))
            c -= 1
        else:       # unreachable cell (outside the band): finish with deletions then insertions
            if c > 0:
                ops.append(("d", -1, c - 1))
                c -= 1
            else:
                ops.append(("i", i - 1, -1))
                i -= 1
    ops.reverse()
    new: list[Slot] = []
    for kind, ti_, sj in ops:
        if kind == "m":
            s = slots[sj]
            s.add(h, toks[ti_], w)
            new.append(s)
        elif kind == "i":
            s = Slot()
            s.add(h, toks[ti_], w)
            new.append(s)
        else:
            new.append(slots[sj])
    return new


def _hyp_order(hyps: list[Hyp]) -> list[Hyp]:
    """Strongest first, engines interleaved: the backbone of the network has every family early (less order bias)."""
    cams = sorted({h.cam for h in hyps}, key=lambda c: -sum(t.snr for h in hyps if h.cam == c for t in h.toks) / max(1, sum(len(h.toks) for h in hyps if h.cam == c)))
    rank = {c: i for i, c in enumerate(cams)}
    return sorted(hyps, key=lambda h: (-(h.cover[1] - h.cover[0] > 200), rank[h.cam] + 0.3 * ENGINE_IDX.get(h.engine, 0), ENGINE_IDX.get(h.engine, 0)))


def build_network(hyps: list[Hyp], P: Params | None = None) -> list[Slot]:
    """The confusion network of all the hypotheses (progressive alignment + leave-one-out refinement)."""
    P = P or Params()
    hyp_engine[0] = {h.hid: h.engine for h in hyps}
    for h in hyps:
        for t in h.toks:
            t.hid = h.hid
    if P.compounds:
        merge_compounds(hyps)
    order = _hyp_order(hyps)
    slots: list[Slot] = []
    inside: list[Hyp] = []
    for h in order:
        slots = align_hyp(slots, h, inside, P)
        inside.append(h)
    for _ in range(P.refine_rounds):
        for h in order:
            others = [x for x in inside if x.hid != h.hid]
            kept: list[Slot] = []
            for s in slots:
                s.remove(h, h.w0)
                if s.mem:
                    kept.append(s)
            slots = align_hyp(kept, h, others, P)
    return slots


# ------------------------------------------------------------------------------------------------ weights


def attach_levels(hyps: list[Hyp], levels) -> None:
    """Local SNR and distance proxy of every token (fuse_levels.Levels)."""
    for h in hyps:
        for t in h.toks:
            t.snr, t.rel = levels.feats(h.cam, t.t0, t.t1)


FEATURE_NAMES = ["bias", "eng_wl3t", "eng_pk", "eng_cn", "snr", "rel", "conf_wl3", "conf_wl3t", "conf_pk", "no_conf", "log_dur", "len", "filler", "short_word"]


def token_features(h: Hyp, t: Tok) -> list[float]:
    e = h.engine
    conf = t.conf if t.conf is not None else None
    cl = 0.0 if conf is None else (conf - 0.8)
    return [1.0, float(e == "whisper-large-v3-turbo"), float(e == "parakeet-tdt-0.6b-v3"), float(e == "canary-1b-v2"),
            max(-5.0, min(25.0, t.snr)) / 10.0 - 0.5, max(-30.0, min(0.0, t.rel)) / 10.0 + 1.0,
            cl * 5.0 * (e == "whisper-large-v3"), cl * 5.0 * (e == "whisper-large-v3-turbo"), cl * 5.0 * (e == "parakeet-tdt-0.6b-v3"),
            float(conf is None), math.log(max(0.04, min(1.5, t.t1 - t.t0))) / 2.0 + 1.0, min(len(t.norm), 10) / 5.0 - 1.0, float(t.filler),
            float(len(t.norm) <= 2)]


def logistic_fit(X: np.ndarray, y: np.ndarray, l2: float = 2.0, iters: int = 25) -> np.ndarray:
    beta = np.zeros(X.shape[1])
    reg = l2 * np.eye(X.shape[1])
    reg[0, 0] = 0.0
    for _ in range(iters):
        z = np.clip(X @ beta, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        Wd = p * (1.0 - p) + 1e-6
        g = X.T @ (y - p) - reg @ beta
        H = (X * Wd[:, None]).T @ X + reg
        step = np.linalg.solve(H, g)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-6:
            break
    return beta


class LocalW:
    """The mean vote weight of a hypothesis around a moment: the strength of its "nothing here" vote."""

    def __init__(self, h: Hyp, half: float = 1.5):
        self.t = np.array([t.tk for t in h.toks])
        order = np.argsort(self.t)
        self.t = self.t[order]
        w = np.array([t.w for t in h.toks])[order]
        self.c = np.concatenate([[0.0], np.cumsum(w)])
        self.half = half
        self.mean = float(w.mean()) if len(w) else 0.5

    def at(self, t: float) -> float:
        a = int(np.searchsorted(self.t, t - self.half, "left"))
        b = int(np.searchsorted(self.t, t + self.half, "right"))
        return float((self.c[b] - self.c[a]) / (b - a)) if b > a else self.mean


def eligible(h: Hyp, t: float, P: Params) -> bool:
    return h.cover[0] - P.cover_margin <= t <= h.cover[1] + P.cover_margin


def set_prior_weights(hyps: list[Hyp], P: Params) -> None:
    fam_n: dict[str, set[str]] = defaultdict(set)
    for h in hyps:
        fam_n[FAMILY.get(h.engine, h.engine)].add(h.engine)
    for h in hyps:
        base = 1.0 if P.weight_mode == "uniform" else ENGINE_PRIOR.get(h.engine, 1.0)
        if P.family_balance:
            base /= len(fam_n[FAMILY.get(h.engine, h.engine)]) ** P.family_balance
        h.w0 = base
        for t in h.toks:
            t.w = base


def _loo_labels(slots: list[Slot], hyps: list[Hyp], P: Params) -> list[tuple[Hyp, Tok, int]]:
    """For every token: 1 if it is the word the OTHER hypotheses vote for, 0 if they vote another word or nothing; tokens in slots where the
    others are too split (top share < cal_min_loo_share) are left out."""
    byid = {h.hid: h for h in hyps}
    filler_engines = {h.engine for h in hyps if any(t.filler for t in h.toks)}
    out = []
    for s in slots:
        has_filler = any(t.filler for t in s.mem.values())
        elig = [h for h in hyps if eligible(h, s.t, P) and (not has_filler or h.engine in filler_engines)]
        if len(elig) < 3:
            continue
        tally: dict[str, float] = defaultdict(float)
        null = 0.0
        for h in elig:
            tok = s.mem.get(h.hid)
            if tok is None:
                null += h.w0 * P.null_gamma
            else:
                tally[tok.norm] += h.w0
        tot = null + sum(tally.values())
        elig_ids = {h.hid for h in elig}
        for hid, tok in s.mem.items():
            if hid not in elig_ids:
                continue
            h = byid[hid]
            tt = dict(tally)
            tt[tok.norm] -= h.w0
            rest = tot - h.w0
            if rest <= 0:
                continue
            cands = list(tt.items()) + [(None, null)]
            top, tw = max(cands, key=lambda kv: kv[1])
            if tw / rest < P.cal_min_loo_share:
                continue
            out.append((h, tok, int(top == tok.norm)))
    return out


def fit_reliability(slots: list[Slot], hyps: list[Hyp], P: Params) -> np.ndarray:
    """Logistic coefficients (FEATURE_NAMES) of P(token = the vote of all the others)."""
    rows = _loo_labels(slots, hyps, P)
    X = np.array([token_features(h, t) for h, t, _ in rows])
    y = np.array([lab for _, _, lab in rows], dtype=float)
    beta = logistic_fit(X, y, P.cal_l2)
    return beta


def assign_weights(hyps: list[Hyp], slots: list[Slot], P: Params) -> dict:
    """Set `Tok.w` for every token according to P.weight_mode; returns the calibration summary."""
    set_prior_weights(hyps, P)
    info: dict = {"mode": P.weight_mode}
    if P.weight_mode != "cal":
        return info
    beta = fit_reliability(slots, hyps, P)
    info["beta"] = {n: round(float(b), 4) for n, b in zip(FEATURE_NAMES, beta)}
    fam_n: dict[str, set[str]] = defaultdict(set)
    for h in hyps:
        fam_n[FAMILY.get(h.engine, h.engine)].add(h.engine)
    for h in hyps:
        fb = 1.0 / len(fam_n[FAMILY.get(h.engine, h.engine)]) ** P.family_balance if P.family_balance else 1.0
        X = np.array([token_features(h, t) for t in h.toks]) if h.toks else np.zeros((0, len(FEATURE_NAMES)))
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ beta, -30, 30))) if len(X) else np.zeros(0)
        p = np.clip(p, 0.05, 0.98)
        for t, pi in zip(h.toks, p):
            t.w = float(pi) ** P.cal_gamma * fb
    return info


# ------------------------------------------------------------------------------------------------ voting


@dataclass
class SlotResult:
    idx: int
    t: float
    winner: str | None              # canonical word, None when "nothing" won
    best_word: str | None           # the best non-null candidate
    share: float                    # winner's weight / all voting weight
    margin: float                   # winner's share - runner-up's share
    word_share: float               # best non-null candidate's share
    null_share: float
    tally: list[tuple[str | None, float, list[int]]]    # (word, weight, hids) best first, None = nothing
    total: float
    voters: int
    filler: bool = False
    eligible_cams: int = 0
    overlap: bool = False
    kept_by_overlap: bool = False


def vote(slots: list[Slot], hyps: list[Hyp], P: Params, overlaps: list[tuple[float, float]] | None = None) -> list[SlotResult]:
    """One SlotResult per slot (network order). Hypotheses with audio at the slot's moment that wrote nothing vote "nothing"; a slot whose
    candidate is a filler is judged only among the engines that write fillers at all."""
    local = {h.hid: LocalW(h) for h in hyps}
    filler_engines = {h.engine for h in hyps if any(t.filler for t in h.toks)}
    res: list[SlotResult] = []
    ov = overlaps or []
    for k, s in enumerate(slots):
        tally: dict[str, float] = defaultdict(float)
        who: dict[str, list[int]] = defaultdict(list)
        for hid, tok in s.mem.items():
            tally[tok.norm] += tok.w
            who[tok.norm].append(hid)
        top_word = max(tally.items(), key=lambda kv: kv[1])[0] if tally else None
        is_filler = top_word in FILLERS
        elig = [h for h in hyps if eligible(h, s.t, P) and (not is_filler or h.engine in filler_engines)]
        null = 0.0
        for h in elig:
            if h.hid not in s.mem:
                null += local[h.hid].at(s.t) * P.null_gamma
        tot = sum(tally.values()) + null
        if tot <= 0:
            tot = 1e-9
        entries = sorted([(n, v, who[n]) for n, v in tally.items()], key=lambda e: -e[1])
        entries.append((None, null, []))
        entries.sort(key=lambda e: -e[1])
        winner, ww, _ = entries[0]
        second = entries[1][1] if len(entries) > 1 else 0.0
        in_ov = any(a <= s.t <= b for a, b in ov)
        word_entries = [e for e in entries if e[0] is not None]
        best_word, bw = (word_entries[0][0], word_entries[0][1]) if word_entries else (None, 0.0)
        kept = False
        if winner is None and in_ov and best_word is not None and bw / tot >= P.overlap_keep_share:
            winner, ww, kept = best_word, bw, True
            second = null
        res.append(SlotResult(k, s.t, winner, best_word, ww / tot, (ww - second) / tot, bw / tot, null / tot, [(n, v / tot, ids) for n, v, ids in entries],
                              tot, len(elig), is_filler, len({h.cam for h in elig}), in_ov, kept))
    return res


# ------------------------------------------------------------------------------------------------ the consensus


@dataclass
class Consensus:
    slots: list[Slot]
    results: list[SlotResult]
    hyps: list[Hyp]
    params: Params
    calibration: dict = field(default_factory=dict)

    def words(self) -> list[tuple[str, float]]:
        """[(canonical word, time)] of the consensus (slots whose winner is a word, fillers included)."""
        return [(r.winner, r.t) for r in self.results if r.winner is not None]


def consensus(hyps: list[Hyp], P: Params | None = None, overlaps: list[tuple[float, float]] | None = None) -> Consensus:
    """Align, calibrate the weights, vote. `hyps` keep their tokens (merge_compounds may change them)."""
    P = P or Params()
    hyps = [h for h in hyps if h.toks]
    set_prior_weights(hyps, P)
    slots = build_network(hyps, P)
    info = assign_weights(hyps, slots, P)
    res = vote(slots, hyps, P, overlaps)
    return Consensus(slots, res, hyps, P, info)


def best_word_time(s: Slot, winner: str, engines: dict[int, str]) -> tuple[float, float]:
    """Start / end of the consensus word: the median over the hypotheses that wrote it, weighted by the engine's timing reliability. The end of a
    canary word is its start + 80 ms (no real end), so canary only counts for the start."""
    mem = [(hid, t) for hid, t in s.mem.items() if t.norm == winner]
    if not mem:
        mem = list(s.mem.items())
    ws = [ENGINE_TIME_W.get(engines.get(hid, ""), 1.0) for hid, _ in mem]
    t0 = wmedian([t.t0 for _, t in mem], ws)
    ends = [(t.t1, w) for (hid, t), w in zip(mem, ws) if engines.get(hid) != "canary-1b-v2"]
    if ends:
        t1 = wmedian([e for e, _ in ends], [w for _, w in ends])
    else:
        t1 = t0 + float(np.median([t.t1 - t.t0 for _, t in mem]))
    return t0, max(t1, t0 + 0.04)


# ------------------------------------------------------------------------------------------------ loading the real hypotheses


def load_hyps(raw_dir: Path, clocks: dict, cams: list[str] | None = None, engines: list[str] | None = None, hid0: int = 0) -> list[Hyp]:
    """The voting hypotheses of the phones that have a clock (or of one clockless phone, in its own time, when `clocks[cam]` is the identity)."""
    import json
    from .texts import load_engine
    out: list[Hyp] = []
    hid = hid0
    for d in sorted(Path(raw_dir).glob("[0-9][0-9]")):
        cam = d.name
        clk = clocks.get(cam)
        if clk is None or (cams and cam not in cams):
            continue
        for key in VOTING:
            if engines and key not in engines and key.split(":")[0] not in engines:
                continue
            words = load_engine(Path(raw_dir), cam, key, clk)
            if not words:
                continue
            base = key.split(":")[0]
            dur = None
            p = Path(raw_dir) / cam / f"{base}.json"
            try:
                dur = float(json.loads(p.read_text(encoding="utf-8")).get("duration") or 0.0)
            except (OSError, ValueError):
                dur = None
            cover = (clk.to_master(0.0), clk.to_master(dur)) if dur else (words[0].m0, words[-1].m1)
            toks = make_toks([{"word": w.text, "start": w.m0, "end": w.m1, "prob": w.prob} for w in words], hid)
            out.append(Hyp(hid, cam, base, toks, cover, f"{key}@{cam}", ENGINE_PRIOR.get(base, 1.0)))
            hid += 1
    return out
