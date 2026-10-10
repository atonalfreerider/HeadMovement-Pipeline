"""Who is who: name the global voices with independent evidence, and look for the class's jokes.

JOKE_NOTE (owner facts, 2026-10-09): Larissa introduces herself as "Kadu" as a joke (the voice that says it is Larissa), and Kadu says
"Look at my red hair" (a joke: he is mostly bald, Larissa has long red hair). So: a single self-introduction, name mention or remark
about appearance is NEVER proof of who a voice is. Evidence used, in this order of weight:

  1. ADDRESSING over the whole class. A name said as a vocative ("Kadu, can you show...") or in the third person ("Larissa and I...")
     is said by someone who is NOT that person, and the next teacher voice to speak after a vocative is, more likely than not, the
     person addressed. Every such event is a small log-likelihood-ratio step towards one of the two assignments of the two teacher
     voices (A = Larissa, B = Kadu, or swapped); the steps are capped so that one chatty stretch cannot decide. Self-introductions
     ("I'm Kadu", "my name is ...") carry NO weight: they are collected as claims and reported as joke candidates when they contradict
     the other evidence.
  2. A pitch PRIOR (median F0 of the voice's voiced windows): Larissa is a woman and Kadu a man (owner's brief, take file), so the
     higher-pitched teacher voice is a priori Larissa. A prior only: it moves the log-odds by a fixed step when the two voices are well
     apart, never beyond what the addressing evidence can overturn.
  3. Which phone is closest / loudest for each voice (descriptive; shown in the review list, not used in the score).

Confidence is reported per voice and per utterance (turn confidence x naming confidence) and is NOT a calibrated probability.
"""
from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field

import numpy as np

from .glossary import edit_distance, normalise, OTHERS, PEOPLE
from .texts import Word, norm_token

JOKE_NOTE = ("The owner says Larissa introduces herself as 'Kadu' as a joke and Kadu says 'Look at my red hair' (he is mostly bald; "
             "Larissa has long red hair). No voice is named from a self-introduction, a name mention or a remark about appearance.")

LAUGH_EMOJI = "\U0001F602"      # face with tears of joy, the owner's token for the jokes

TEACHERS = {"larissa": {normalise(v) for v in ["Larissa"] + PEOPLE[0]["variants"]},
            "kadu": {normalise(v).replace(" ", "") for v in ["Kadu"] + PEOPLE[1]["variants"] if " " not in v}}
STUDENTS = {normalise(n): n for n, _ in OTHERS} | {normalise(v): n for n, vs in OTHERS for v in vs}

GREETINGS = {"hello", "hi", "hey", "hola", "oi", "ola", "okay", "ok"}
SENT_END = re.compile(r"[.?!]$")
VOC_AFTER = {"can", "could", "would", "will", "show", "let", "lets", "come", "go", "do", "try", "look", "please", "you", "your", "are", "did",
             "what", "how", "why", "where", "when", "maybe", "now", "thank", "thanks", "great", "good", "yes", "no", "right", "okay", "ok",
             "stand", "stay", "take", "give", "help", "tell", "explain", "demonstrate", "lead", "follow", "start", "stop", "wait", "listen"}
VOC_BEFORE = {"thanks", "thank", "hey", "ok", "okay", "yes", "right", "so", "well", "now", "please", "oh", "uh", "um", "hi", "hello", "good",
              "great", "no", "sorry", "yeah"}
THIRD_AFTER = {"is", "was", "will", "can", "could", "does", "did", "has", "have", "s", "would", "should", "and", "or", "here", "there", "over",
               "doing", "going", "starts", "started", "leads", "follows", "who", "also", "too", "then"}
THIRD_BEFORE = {"with", "for", "to", "from", "by", "than", "like", "about", "of", "ask", "asks", "tell", "tells", "thanks", "thank", "behind", "beside", "near",
                "give", "gives", "gave", "show", "shows", "help", "helps", "let", "bring", "brings", "take", "takes", "hold", "holds", "pull", "pulls", "push",
                "pushes", "lead", "leads", "follow", "follows", "hug", "hugs", "see", "watch", "look"}


def name_of(token: str) -> str | None:
    """'larissa' | 'kadu' | a student's name | None for a (normalised) token. Larissa also matches within one edit; Kadu only by list
    (four letters leave no room for fuzz)."""
    t = norm_token(token)
    if not t:
        return None
    if t in TEACHERS["larissa"] or (len(t) >= 6 and edit_distance(t, "larissa") <= 1):
        return "larissa"
    if t in TEACHERS["kadu"]:
        return "kadu"
    if t in STUDENTS and len(t) >= 3:
        return "student:" + STUDENTS[t]
    return None


def classify_context(before: list[str], after: list[str], raw_name: str) -> str:
    """Kind of a name mention from its raw neighbours (with punctuation): 'self_strong' | 'self_weak' | 'vocative' | 'third' | 'unknown'."""
    nb = [norm_token(x) for x in before]
    na = [norm_token(x) for x in after]
    nb = [x for x in nb if x]
    na = [x for x in na if x]
    if nb[-1:] == ["im"] or nb[-2:] == ["i", "am"] or nb[-3:] == ["my", "name", "is"] or nb[-2:] == ["name", "is"] or nb[-2:] == ["call", "me"] \
            or nb[-3:] == ["they", "call", "me"] or nb[-2:] == ["name", "s"]:
        return "self_strong"
    if nb[-2:] == ["this", "is"] or (nb[-1:] == ["its"] and nb[-2:-1] and nb[-2] in GREETINGS):
        return "self_weak"
    prev_raw = before[-1] if before else ""
    sentence_start = not before or bool(SENT_END.search(prev_raw))
    if raw_name.rstrip().endswith(","):
        return "vocative"
    if nb[-2:] == ["thank", "you"] or (nb[-1:] and nb[-1] in VOC_BEFORE) or (sentence_start and na[:1] and na[0] in VOC_AFTER):
        return "vocative"
    if before and before[-1].endswith(",") and SENT_END.search(raw_name):
        return "vocative"
    if (na[:1] and na[0] in THIRD_AFTER) or (nb[-1:] and nb[-1] in THIRD_BEFORE) or (na[:1] == ["and"] and na[1:2] in (["i"], ["me"])):
        return "third"
    if nb[-1:] == ["and"] or (na[:1] == ["and"]):
        return "third"
    return "unknown"


KIND_PRIORITY = ["self_strong", "vocative", "third", "self_weak", "unknown"]


@dataclass
class Mention:
    id: str
    name: str                      # larissa | kadu | student:<Name>
    kind: str
    t: float                       # master seconds (mean over the streams that heard it)
    t0: float
    t1: float
    support: int                   # distinct (phone, engine) streams
    engines: list[str]
    cams: list[str]
    context: str                   # text around it from the best stream
    speaker: int | None = None     # global voice id
    speaker_conf: float = 0.0
    kinds: dict = field(default_factory=dict)


def _streams(words: list[Word]) -> dict[tuple[str, str], list[Word]]:
    out: dict[tuple[str, str], list[Word]] = {}
    for w in words:
        out.setdefault((w.cam, w.engine), []).append(w)
    for v in out.values():
        v.sort(key=lambda w: w.m0)
    return out


def find_mentions(words: list[Word], merge_s: float = 1.5, ctx: int = 6) -> list[Mention]:
    """All name mentions of all streams, merged across streams by name and master time."""
    raw: list[dict] = []
    for (cam, eng), ws in _streams(words).items():
        for i, w in enumerate(ws):
            nm = name_of(w.text)
            if not nm:
                continue
            before = [x.text for x in ws[max(0, i - ctx):i]]
            after = [x.text for x in ws[i + 1:i + 1 + ctx]]
            raw.append({"name": nm, "cam": cam, "engine": eng, "m0": w.m0, "m1": w.m1, "kind": classify_context(before, after, w.text),
                        "text": " ".join(before + [f"[{w.text}]"] + after), "prob": w.prob})
    raw.sort(key=lambda r: (r["name"], r["m0"]))
    groups: list[list[dict]] = []
    for r in raw:
        if groups and groups[-1][0]["name"] == r["name"] and r["m0"] - groups[-1][-1]["m0"] <= merge_s:
            groups[-1].append(r)
        else:
            groups.append([r])
    out: list[Mention] = []
    for g in groups:
        kinds: dict[str, int] = {}
        for r in g:
            kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
        known = {k: v for k, v in kinds.items() if k != "unknown"}
        if known:
            top = max(known.values())
            kind = next(k for k in KIND_PRIORITY if known.get(k) == top)
        else:
            kind = "unknown"
        streams = {(r["cam"], r["engine"]) for r in g}
        best = next((r for r in g if r["engine"].startswith("parakeet")), None) or next((r for r in g if r["engine"].endswith(":en")), g[0])
        out.append(Mention("", g[0]["name"], kind, round(float(np.mean([r["m0"] for r in g])), 2), round(min(r["m0"] for r in g), 2),
                           round(max(r["m1"] for r in g), 2), len(streams), sorted({r["engine"] for r in g}), sorted({r["cam"] for r in g}),
                           best["text"], kinds=kinds))
    out.sort(key=lambda m: m.t)
    for i, m in enumerate(out):
        m.id = f"m{i:03d}"
    return out


# ------------------------------------------------------------------------------------------------ speaker at a time


def voice_at(t0: float, t1: float, mom_t0: np.ndarray, mom_t1: np.ndarray, path: np.ndarray, marg: np.ndarray) -> tuple[int | None, float]:
    """(voice, confidence) over [t0, t1]: the label holding most of the overlap with the moments; confidence = its share x the mean
    margin-based support. None when no moment overlaps (nobody was detected speaking)."""
    t1 = max(t1, t0 + 0.3)
    lo = np.searchsorted(mom_t1, t0, side="right")
    hi = np.searchsorted(mom_t0, t1, side="left")
    if hi <= lo:
        return None, 0.0
    seg = path[lo:hi]
    w = np.minimum(mom_t1[lo:hi], t1) - np.maximum(mom_t0[lo:hi], t0)
    w = np.clip(w, 0, None)
    if w.sum() <= 0:
        return None, 0.0
    shares: dict[int, float] = {}
    for lab, ww in zip(seg, w):
        shares[int(lab)] = shares.get(int(lab), 0.0) + float(ww)
    best = max(shares, key=shares.get)
    share = shares[best] / sum(shares.values())
    m = float(np.mean(marg[lo:hi][seg == best]))
    return best, round(share * (1.0 / (1.0 + math.exp(-(m - 0.03) / 0.03))), 3)


# ------------------------------------------------------------------------------------------------ naming


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def pitch_prior(f_a: float | None, f_b: float | None) -> tuple[float, str]:
    """Log-odds that voice A is Larissa (the woman) rather than Kadu (the man), from the two voices' median F0. A prior of at most +-2.5."""
    if not f_a or not f_b:
        return 0.0, "no pitch for one of the voices"
    d = f_a - f_b
    if abs(d) < 25:
        return 0.0, f"F0 {f_a:.0f} vs {f_b:.0f} Hz too close to say"
    mag = 2.5 if abs(d) >= 70 else 1.5 if abs(d) >= 45 else 0.7
    return (mag if d > 0 else -mag), f"F0 {f_a:.0f} vs {f_b:.0f} Hz: the higher voice is the likelier woman"


def gender_prior(g_a: float | None, g_b: float | None) -> tuple[float, str]:
    """Log-odds that voice A is Larissa (the woman) from the AudioSet speech tags averaged over each voice's moments: g = mean(P(female
    speech) - P(male speech)). Independent of the pitch tracker (a trained classifier on 10 s clips). At most +-2.5, a prior only."""
    if g_a is None or g_b is None:
        return 0.0, "no gender tags for one of the voices"
    d = g_a - g_b
    if abs(d) < 0.06:
        return 0.0, f"AudioSet tags F-M {g_a:+.2f} vs {g_b:+.2f}: too close to say"
    return max(-2.5, min(2.5, 8.0 * d)), f"AudioSet tags (female - male speech) {g_a:+.2f} vs {g_b:+.2f}: the more female-tagged voice is the likelier woman"


def addressing_evidence(mentions: list[Mention], turns: list[dict], a: int, b: int, p_event: float = 0.8, cap: float = 6.0,
                        response_s: float = 8.0) -> dict:
    """Log-odds of H (A = Larissa, B = Kadu) against H' (swapped) from vocative / third-person mentions of the two names and from who
    answers a vocative. `turns` are the global diarization turns (start, end, voice)."""
    step = math.log(p_event / (1 - p_event))
    events = []
    total = 0.0
    for m in mentions:
        if m.name not in ("larissa", "kadu") or m.kind not in ("vocative", "third") or m.speaker not in (a, b):
            continue
        rel = min(1.0, 0.4 + 0.2 * m.support) * max(0.2, m.speaker_conf)
        w = rel * (1.0 if m.kind == "vocative" else 0.7)
        # said by A about "Kadu" or by B about "Larissa" supports H
        supports_h = (m.speaker == a and m.name == "kadu") or (m.speaker == b and m.name == "larissa")
        total += (step if supports_h else -step) * w
        events.append({"id": m.id, "t": m.t, "kind": m.kind, "name": m.name, "speaker": m.speaker, "weight": round(w, 3),
                       "supports": "A=Larissa" if supports_h else "A=Kadu", "type": "mention"})
        if m.kind == "vocative":
            nxt = next((t for t in turns if t["start"] >= m.t0 and t["start"] <= m.t1 + response_s and t["voice"] in (a, b) and t["voice"] != m.speaker), None)
            if nxt is not None:
                resp_a = nxt["voice"] == a
                supports_h = (resp_a and m.name == "larissa") or ((not resp_a) and m.name == "kadu")
                w2 = 0.6 * max(0.2, nxt.get("confidence", 0.5))
                total += (step if supports_h else -step) * w2
                events.append({"id": m.id, "t": round(nxt["start"], 2), "kind": "response", "name": m.name, "speaker": nxt["voice"],
                               "weight": round(w2, 3), "supports": "A=Larissa" if supports_h else "A=Kadu", "type": "response"})
    capped = max(-cap, min(cap, total))
    return {"llr": round(capped, 3), "llr_uncapped": round(total, 3), "n_events": len(events), "events": events}


def name_voices(voices: list[dict], mentions: list[Mention], turns: list[dict], roles: dict | None = None) -> dict:
    """Assign Larissa / Kadu to the two teacher voices (the two with the most speech) and 'other_k' to the rest.
    Returns {assignment: {voice: name}, posterior_a_is_larissa, text, pitch, ...}."""
    ranked = sorted(voices, key=lambda v: -v["seconds"])
    out: dict = {"assignment": {}, "teachers": [], "note": JOKE_NOTE}
    if len(ranked) < 2:
        out["error"] = "fewer than two voices found"
        return out
    a, b = ranked[0]["voice"], ranked[1]["voice"]
    ev = addressing_evidence(mentions, turns, a, b)
    prior_f0, why_f0 = pitch_prior(ranked[0].get("f0_median"), ranked[1].get("f0_median"))
    prior_g, why_g = gender_prior(ranked[0].get("gender_f_minus_m"), ranked[1].get("gender_f_minus_m"))
    # the pitch tracker is weak on a quiet low voice (18-25 % of its frames voiced), so it counts at most 1/2 of the classifier prior
    prior_f0 = max(-1.0, min(1.0, prior_f0 * (0.5 if min(ranked[0].get("f0_voiced_frames", 1) or 1, ranked[1].get("f0_voiced_frames", 1) or 1) < 0.4 else 1.0)))
    prior = prior_f0 + prior_g
    why = f"{why_g}; {why_f0}"
    llr = ev["llr"] + prior
    p = sigmoid(llr)
    larissa, kadu = (a, b) if p >= 0.5 else (b, a)
    only_prior = abs(ev["llr"]) < 1.0
    out.update({"teachers": [a, b], "addressing": ev, "evidence": "acoustic prior only: no usable addressing event" if not ev["n_events"] else (
        "addressing is thin (|llr| < 1): the acoustic prior decides" if only_prior else "addressing and acoustic prior"), "pitch_prior": {"llr": round(prior, 3), "why": why, "f0_llr": round(prior_f0, 3), "gender_tag_llr": round(prior_g, 3)},
                "llr_total": round(llr, 3), "posterior_a_is_larissa": round(p, 3),
                "naming_confidence": round(min(max(p, 1 - p), 0.85) if only_prior else max(p, 1 - p), 3),
                "agreement": "addressing and pitch agree" if (ev["llr"] * prior > 0) else (
                    "only one kind of evidence" if (ev["llr"] == 0 or prior == 0) else "addressing and pitch DISAGREE")})
    out["assignment"] = {str(larissa): "Larissa", str(kadu): "Kadu"}
    k = 1
    for v in ranked[2:]:
        out["assignment"][str(v["voice"])] = f"other_{k}"
        k += 1
    return out


# ------------------------------------------------------------------------------------------------ jokes


APPEARANCE = re.compile(r"\b(my\s+(?:red\s+|long\s+)?hair|your\s+hair|red\s+hair|long\s+hair|redhead|ginger|bald|shaved\s+head|no\s+hair|my\s+beard)\b|\bhair\b", re.I)
RED_NEAR = re.compile(r"\bred\b", re.I)


def find_jokes(mentions: list[Mention], utterances: list[dict], names: dict, laughter: list[dict], texts_by_time=None,
               reaction_window: tuple[float, float] = (-0.5, 6.0)) -> list[dict]:
    """Candidates for the owner's two jokes and any other contradiction between what a voice says about itself and who the voice is.

    * 'self_introduction': a self_strong / self_weak mention whose name differs from the voice's inferred name -> the voice is joking,
      claiming a name that is not its own (Larissa introducing herself as Kadu);
    * 'appearance': an utterance with an appearance word (red hair, bald, ...) said by a voice whose person does not fit (Kadu about
      "my red hair": Larissa has the red hair);
    Each candidate gets the laughter / cheering / crowd events that START within `reaction_window` seconds of the utterance and the
    time where the reaction token (U+1F602) belongs: the start of the first such event, else the end of the utterance. Candidates are
    scored, never asserted.
    """
    assign = names.get("assignment", {})
    conf = names.get("naming_confidence", 0.0)
    out = []

    def person(v: int | None) -> str | None:
        n = assign.get(str(v)) if v is not None else None
        return n if n in ("Larissa", "Kadu") else None

    def reaction(t_end: float, t_start: float) -> tuple[list[dict], float]:
        evs = [e for e in laughter if t_start + reaction_window[0] <= e["t0"] <= t_end + reaction_window[1]]
        evs.sort(key=lambda e: e["t0"])
        return evs, (evs[0]["t0"] if evs else t_end)

    for m in mentions:
        if m.kind not in ("self_strong", "self_weak") or m.name not in ("larissa", "kadu") or m.speaker is None:
            continue
        who = person(m.speaker)
        claimed = m.name.capitalize()
        if who is None:
            continue
        evs, t_tok = reaction(m.t1, m.t0)
        contradicts = who != claimed
        out.append({"kind": "self_introduction", "t": m.t, "t_token": round(t_tok, 2), "voice": m.speaker, "voice_name": who, "claimed": claimed,
                    "contradiction": contradicts, "strength": m.kind, "text": m.context, "support": m.support,
                    "reaction": [{"id": e["id"], "kind": e["kind"], "t0": e["t0"], "peak": e["peak"]} for e in evs[:3]],
                    "emoji": LAUGH_EMOJI if contradicts else None,
                    "score": round(conf * (0.5 if m.kind == "self_weak" else 1.0) * (1.0 if contradicts else 0.0) * min(1.0, 0.5 + 0.25 * m.support) + 0.15 * bool(evs), 3),
                    "why": (f"the voice we name {who} says '{claimed}' about itself" if contradicts else
                            f"{who} says their own name '{claimed}' (consistent; not a joke by itself)")})
    for u in utterances:
        text = u.get("text", "")
        mt = APPEARANCE.search(text)
        if not mt:
            continue
        who = person(u.get("voice"))
        phrase = " ".join(mt.group(0).lower().split())
        mine = bool(re.search(r"\bmy\s+(?:red\s+|long\s+)?hair\b|\bmy\s+beard\b", text, re.I))
        red = bool(RED_NEAR.search(text))
        # red / long hair is Larissa's; bald / shaved head is Kadu's. A remark about OWN hair by the voice we name Kadu, or about own baldness by
        # the voice we name Larissa, contradicts the person: the owner's joke ("Look at my red hair", he is mostly bald).
        owner = "Larissa" if re.search(r"red|long|ginger|hair", phrase) and not re.search(r"bald|shaved|no\s+hair", phrase) else "Kadu"
        contradicts = bool(who and mine and who != owner)
        evs, t_tok = reaction(u["end"], u["start"])
        out.append({"kind": "appearance", "t": u["start"], "t_end": u["end"], "t_token": round(t_tok, 2), "voice": u.get("voice"), "voice_name": who, "phrase": phrase,
                    "own_hair_remark": mine, "red_in_the_same_stretch": red, "fits": owner, "contradiction": contradicts, "text": text[:300],
                    "reaction": [{"id": e["id"], "kind": e["kind"], "t0": e["t0"], "peak": e["peak"]} for e in evs[:3]],
                    "emoji": LAUGH_EMOJI if contradicts else None,
                    "score": round(conf * (1.0 if contradicts else 0.15) * (1.0 if red else 0.7) + 0.15 * bool(evs), 3),
                    "why": (f"'{phrase}' is a remark about the speaker's own hair; the voice we name {who} has none to speak of (hair is {owner}'s)" if contradicts else
                            f"an appearance word ('{phrase}'); voice is {who or 'not named'}")})
    out.sort(key=lambda j: -j["score"])
    return out
