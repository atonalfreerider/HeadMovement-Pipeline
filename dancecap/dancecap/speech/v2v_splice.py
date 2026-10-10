"""Word-level repair of converted pieces by splicing (agent V2).

Speech-to-speech of the noisy phone audio garbles different words in different conversions of the SAME sentence (measured: per piece the WER of
15 variants runs from 0.0 to 1.0; best-of-N picks help, but a sentence that is right in one try at its start and in another try at its end is still
wrong as a whole). Every converted try is on the same time base (aligned to its source piece), so a piece can be assembled from the words that
the independent ASR understood correctly, switching from one try to another only in a pause (a low-energy valley of the speech-band envelope).

    ref words (the transcript, normalised) ----ASR of try t (words + times)----> per try and ref word: correct? where (start, end)?
    Viterbi over the words: cost 1 per wrong word of the chosen try, `SWITCH_COST` per change of try (allowed only where both tries have a valley)
    render: equal-power crossfade of `FADE_S` at each cut, each try level-matched to the first one.

The result is a NEW candidate (a piece audio) that is checked with the ASR again like any other try: it is used only if its measured WER is lower.
"""
from __future__ import annotations

import numpy as np

from . import v2v as V

SWITCH_COST = 0.7        # a change of try costs less than one wrong word, so a switch must fix at least one word
FADE_S = 0.02
VALLEY_DB = 9.0          # the cut must lie in a valley at least this far below the piece's speech level (P90 of the 10 ms envelope)
MAX_OVERLAP_S = 0.05     # the end of the last word of try a may pass the start of the next word of try b by this much (cut at the middle)


def align_pairs(ref: list[str], hyp: list[str]) -> list[tuple[int | None, int | None]]:
    """Levenshtein alignment: [(i_ref or None, j_hyp or None)] in order (a match or a substitution has both; an insertion only j; a deletion only i)"""
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    out = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            out.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            out.append((i - 1, None))
            i -= 1
        else:
            out.append((None, j - 1))
            j -= 1
    return out[::-1]


def word_status(ref_text: str, asr_words: list) -> tuple[list[str], list[bool], list[tuple[float, float] | None]]:
    """(ref tokens, ok flags, spans) of one try: ref token i is ok when the ASR heard exactly that token; its span is the (start, end) of the ASR word that
    carries the matched token (None when it was not heard correctly). asr_words = [[text, start, end], ...]"""
    ref = V.norm_words(ref_text)
    htok, hspan = [], []
    for w, s, e in asr_words:
        toks = V.norm_words(w)
        for k, t in enumerate(toks):
            htok.append(t)
            hspan.append((float(s), float(e)))
    ok = [False] * len(ref)
    spans: list[tuple[float, float] | None] = [None] * len(ref)
    for i, j in align_pairs(ref, htok):
        if i is not None and j is not None and ref[i] == htok[j]:
            ok[i] = True
            spans[i] = hspan[j]
    return ref, ok, spans


def valley_time(env: np.ndarray, lo: float, hi: float, level_db: float, frame_s: float = 0.01) -> float | None:
    """the time (s) of the lowest frame of the envelope (dB) in [lo, hi] if it is at least VALLEY_DB below level_db, else None"""
    i0, i1 = max(0, int(lo / frame_s)), min(len(env), int(np.ceil(hi / frame_s)) + 1)
    if i1 - i0 < 1:
        return None
    k = int(np.argmin(env[i0:i1]))
    if env[i0 + k] > level_db - VALLEY_DB:
        return None
    return (i0 + k + 0.5) * frame_s


def cut_between(a_spans: list, b_spans: list, w: int, env_a: np.ndarray, env_b: np.ndarray, lvl_a: float, lvl_b: float) -> float | None:
    """cut time for switching from try a (up to word w) to try b (from word w+1): inside [end of word w in a, start of word w+1 in b], found in a valley of
    BOTH tries' envelopes; None when impossible (a word is missing, the words overlap by more than MAX_OVERLAP_S, or no valley)"""
    sa, sb = a_spans[w], b_spans[w + 1]
    if sa is None or sb is None:
        return None
    lo, hi = sa[1], sb[0]
    if hi < lo:
        if lo - hi > MAX_OVERLAP_S:
            return None
        lo = hi = 0.5 * (lo + hi)
    pad = 0.03
    ta = valley_time(env_a, lo - pad, hi + pad, lvl_a)
    tb = valley_time(env_b, lo - pad, hi + pad, lvl_b)
    if ta is None or tb is None:
        return None
    return 0.5 * (ta + tb)


def choose_path(ok: list[list[bool]], cuts: dict[tuple[int, int, int], float | None], n_words: int, switch_cost: float = SWITCH_COST) -> tuple[list[int], float]:
    """Viterbi over words: state = try. cost = number of wrong words of the chosen try + switch_cost per change; a change after word w from try a to b is allowed
    only when cuts[(a, b, w)] is not None. Returns (try index per word, total cost). The first try (index 0) breaks ties."""
    T = len(ok)
    INF = 1e9
    cost = np.full((n_words, T), INF)
    back = np.zeros((n_words, T), dtype=int)
    for t in range(T):
        cost[0, t] = 0.0 if ok[t][0] else 1.0
    for w in range(1, n_words):
        for b in range(T):
            stay = cost[w - 1, b]
            best, arg = stay, b
            for a in range(T):
                if a == b:
                    continue
                if cuts.get((a, b, w - 1)) is None:
                    continue
                c = cost[w - 1, a] + switch_cost
                if c < best - 1e-9:
                    best, arg = c, a
            cost[w, b] = best + (0.0 if ok[b][w] else 1.0)
            back[w, b] = arg
    t = int(np.argmin(cost[-1]))
    total = float(cost[-1, t])
    path = [t]
    for w in range(n_words - 1, 0, -1):
        t = int(back[w, t])
        path.append(t)
    return path[::-1], total


def render(tries_audio: list[np.ndarray], path: list[int], cuts: dict[tuple[int, int, int], float], sr: int, fade_s: float = FADE_S) -> tuple[np.ndarray, list[dict]]:
    """concatenate segments of the tries along `path`: a change between words w and w+1 happens at cuts[(a, b, w)] seconds with an equal-power crossfade of fade_s
    (centred on the cut); all tries have the same length (the source piece's). Returns (audio, list of the changes made)."""
    n = len(tries_audio[0])
    out = np.zeros(n, dtype=np.float64)
    seg_start = 0
    cur = path[0]
    changes = []
    bounds = []                                          # (sample, from_try, to_try)
    for w in range(1, len(path)):
        if path[w] != path[w - 1]:
            c = cuts[(path[w - 1], path[w], w - 1)]
            bounds.append((int(round(c * sr)), path[w - 1], path[w]))
            changes.append({"after_word": w - 1, "from": path[w - 1], "to": path[w], "t": round(c, 3)})
    half = max(1, int(fade_s * sr / 2))
    cur_w = np.zeros(n)
    prev = 0
    segs = []
    for (c, a, b) in bounds:
        segs.append((prev, c, a))
        prev = c
    segs.append((prev, n, path[-1] if not bounds else bounds[-1][2]))
    # weights: each try's weight is 1 inside its segment with equal-power ramps (sin/cos) of 2*half samples across each cut
    for idx, (s, e, t) in enumerate(segs):
        w = np.zeros(n)
        w[s:e] = 1.0
        if idx > 0:
            lo, hi = max(0, s - half), min(n, s + half)
            w[lo:hi] = np.sin(0.5 * np.pi * np.linspace(0, 1, hi - lo))
        if idx < len(segs) - 1:
            lo, hi = max(0, e - half), min(n, e + half)
            w[lo:hi] = np.cos(0.5 * np.pi * np.linspace(0, 1, hi - lo))
        out += w * tries_audio[t]
    return out.astype(np.float32), changes
