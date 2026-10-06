"""Semantic judgments via TypeSafe System One (Jev): typed answers + probabilities, code owns the workflow.

Credentials: TYPESAFE_API_KEY, else the file named by TYPESAFE_API_KEY_FILE (default ~/Desktop/jev-key.txt).
The key is never logged or written anywhere.

Judgments here:
  route_lesson_command  natural-language lesson control -> hm_* Unity CLI commands (function-calling pattern)
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from typesafe_sdk import Choice, Noul, TypeSafeClient

_client: TypeSafeClient | None = None


def client() -> TypeSafeClient:
    global _client
    if _client is None:
        key = os.environ.get("TYPESAFE_API_KEY")
        if not key:
            key_file = Path(os.environ.get("TYPESAFE_API_KEY_FILE", Path.home() / "Desktop" / "jev-key.txt"))
            if not key_file.exists():
                raise RuntimeError("set TYPESAFE_API_KEY or TYPESAFE_API_KEY_FILE")
            key = key_file.read_text().strip()
        _client = TypeSafeClient(api_key=key)
    return _client


# ---------------------------------------------------------------------------
# lesson command routing
# ---------------------------------------------------------------------------

# A "measure" is one zouk 8-count (also called a bar or an eight-count).
ACTIONS = {
    "play": "explicitly start or resume playback (loading or opening a dance alone is not a request to play)",
    "pause": "stop or freeze playback, e.g. to look around or reposition the camera",
    "restart": "go back to the very beginning of the dance",
    "next_measure": "move ahead by exactly one measure (8-count) relative to where playback is now, without naming a measure number",
    "previous_measure": "move back by exactly one measure (8-count) relative to where playback is now, without naming a measure number",
    "next_beat": "step forward by a single beat",
    "previous_beat": "step back by a single beat",
    "goto_measure": "jump to a specific measure identified by its number (a measure is an 8-count / eight-count / bar)",
    "loop_on": "start repeating a measure (8-count) over and over",
    "loop_off": "stop repeating / looping the measure",
    "speed": "change how fast the dance plays (slower, faster, half speed, normal speed)",
    "layers": "show or hide a visual layer: footprints on the floor, tension lines, the room, camera frustums, the text overlay",
    "view": "move the viewing camera to a particular viewpoint (top-down, front, side, behind) or zoom closer/farther",
    "load": "open or switch to a different recorded dance / performance",
}

LAYERS = {
    "floor": "the footprint / floor pattern / step timing marks",
    "tension": "the partner connection lines showing tension or compression",
    "splats": "the reconstructed room / environment splat / background",
    "cameras": "the virtual camera frustums",
    "hud": "the text overlay / heads-up display / info panel",
}

SPEEDS = {"half": 0.5, "three_quarter": 0.75, "normal": 1.0}

VIEWS = {  # azimuth, elevation, radius (radius None = keep); relative zoom handled in code
    "top": (0.0, 80.0, 3.5),
    "front": (90.0, 15.0, 3.0),
    "side": (0.0, 15.0, 3.0),
    "behind": (270.0, 15.0, 3.0),
    "closer": (None, None, 1.8),
    "farther": (None, None, 6.0),
}

NUMBER_WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen".split())}
ORDINALS = {w: i + 1 for i, w in enumerate(
    "first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth".split())}


@dataclass
class Routed:
    commands: list[tuple[str, dict]] = field(default_factory=list)  # (hm_* command, args) in execution order
    confidence: float = 1.0  # weakest judgment the plan depends on
    clarify: str | None = None  # set when the request is too uncertain to act on
    debug: dict = field(default_factory=dict)


def _numbers(text: str) -> list[int]:
    # "8-count" / "eight-count" names the unit, not a measure number
    text = re.sub(r"\b(8|eight)[\s-]*counts?\b", " measure ", text, flags=re.IGNORECASE)
    found = [int(n) for n in re.findall(r"\b(\d{1,3})(?:st|nd|rd|th)?\b", text)]
    for word in re.findall(r"[a-z]+", text.lower()):
        if word in NUMBER_WORDS:
            found.append(NUMBER_WORDS[word])
        if word in ORDINALS:
            found.append(ORDINALS[word])
    return list(dict.fromkeys(found))


def _step_speed(current: float, direction: int) -> float:
    ladder = sorted(SPEEDS.values())
    i = min(range(len(ladder)), key=lambda k: abs(ladder[k] - current))
    return ladder[max(0, min(len(ladder) - 1, i + direction))]


def route_lesson_command(text: str, lesson_state: dict | None = None, act_threshold: float = 0.5) -> Routed:
    """Map one request (possibly several intents) to hm_* commands.

    Lesson controls are cheap and reversible, so every action with p >= act_threshold runs; the request is
    only bounced back for clarification when nothing reaches the threshold. `confidence` reports the weakest
    judgment the plan relies on.
    """
    state = {"request": text, "lesson": lesson_state or {}}
    q: dict = {}
    for name, meaning in ACTIONS.items():
        q[f"do_{name}"] = Noul(instructions=f"Does `request` ask to {meaning}?")
    q["speed_value"] = Choice(
        instructions="If `request` sets a playback speed, which speed does it want?",
        criteria={"half": "half speed, very slow", "three_quarter": "three quarter speed, 75 percent",
                  "normal": "normal / full / regular speed", "slower_step": "slower, without saying how much",
                  "faster_step": "faster, without saying how much", "none": "no speed requested"})
    # stated-argument pattern: a separate yes/no for "is this layer mentioned at all", direction only if so
    for layer, meaning in LAYERS.items():
        q[f"mentions_{layer}"] = Noul(instructions=f"Does `request` refer specifically to {meaning}?")
        q[f"layer_{layer}"] = Choice(
            instructions=f"Assuming `request` refers to {meaning}, should it be shown or hidden?",
            criteria={"show": "turn it on / show it / bring it back", "hide": "turn it off / hide it / remove it"})
    q["view_value"] = Choice(
        instructions="If `request` asks to move the viewing camera, which view?",
        criteria={**{k: None for k in VIEWS}, "none": "no camera change requested"})
    numbers = _numbers(text)
    if numbers:
        q["measure_number"] = Choice(
            instructions="Which of these numbers in `request` is the measure (8-count) the user wants to go to or loop?",
            criteria={**{str(n): None for n in numbers}, "none": "none of them is a measure number"})
    captures = (lesson_state or {}).get("captures") or []
    if captures:
        q["capture_name"] = Choice(
            instructions="Does `request` ask to open a different recorded dance? If so which one?",
            criteria={**{c: None for c in captures}, "none": "no different dance requested"})

    r = client().system_one(state=state, questions=q)
    nouls = {k: v.noul for k, v in r.nouls.items()}
    choices = {k: (v.choice, v.confidence) for k, v in r.choices.items()}
    out = Routed(debug={"nouls": nouls, "choices": choices})

    wanted = [a for a in ACTIONS if nouls[f"do_{a}"] >= act_threshold]
    if not wanted:
        near = sorted(ACTIONS, key=lambda a: -nouls[f"do_{a}"])[:2]
        out.clarify = "I couldn't match that to a lesson control. Did you mean: " + \
            " or ".join(ACTIONS[a] for a in near if nouls[f"do_{a}"] > 0.2) if nouls[f"do_{near[0]}"] > 0.2 \
            else "I couldn't match that to a lesson control."
    confidences = [max(nouls[f"do_{a}"], 1 - nouls[f"do_{a}"]) for a in wanted]

    cmds = out.commands
    cap, cap_conf = choices.get("capture_name", ("none", 1.0))
    if "load" in wanted and cap == "none":
        out.clarify = out.clarify or "Which recording? " + ", ".join((lesson_state or {}).get("captures") or [])
    if "load" in wanted and cap != "none":
        cmds.append(("hm_load", {"name": cap}))
        confidences.append(cap_conf)
    measure, m_conf = choices.get("measure_number", ("none", 1.0))
    if "restart" in wanted:
        cmds.append(("hm_transport", {"action": "restart"}))
    if "goto_measure" in wanted or (measure != "none" and "loop_on" in wanted):
        if measure == "none":
            out.clarify = out.clarify or "Which measure?"
        else:
            cmds.append(("hm_transport", {"action": "measure", "measure": int(measure)}))
            confidences.append(m_conf)
    for a, action in (("next_measure", "measure+"), ("previous_measure", "measure-"),
                      ("next_beat", "beat+"), ("previous_beat", "beat-")):
        if a in wanted:
            cmds.append(("hm_transport", {"action": action}))
    if "speed" in wanted:
        sv, s_conf = choices["speed_value"]
        confidences.append(s_conf)
        current = float((lesson_state or {}).get("speed") or 1.0)
        if sv in SPEEDS:
            cmds.append(("hm_transport", {"action": "speed", "speed": SPEEDS[sv]}))
        elif sv in ("slower_step", "faster_step"):  # relative request -> concrete speed, computed in code
            cmds.append(("hm_transport", {"action": "speed",
                                          "speed": _step_speed(current, -1 if sv == "slower_step" else 1)}))
    if "loop_on" in wanted:
        cmds.append(("hm_transport", {"action": "loop_on"}))
    elif "loop_off" in wanted:
        cmds.append(("hm_transport", {"action": "loop_off"}))
    layer_mentions = [layer for layer in LAYERS if nouls[f"mentions_{layer}"] >= act_threshold]
    if "layers" in wanted or layer_mentions and ("layers" in wanted or "view" in wanted):
        for layer in layer_mentions:
            v, l_conf = choices[f"layer_{layer}"]
            cmds.append(("hm_layer", {"layer": layer, "visible": v == "show"}))
            confidences += [l_conf, nouls[f"mentions_{layer}"]]
    if "view" in wanted:
        v, v_conf = choices["view_value"]
        if v in VIEWS:
            az, el, rad = VIEWS[v]
            args = {"radius": rad}
            if az is not None:
                args.update(azimuth=az, elevation=el)
            cmds.append(("hm_orbit", args))
            confidences.append(v_conf)
    if "pause" in wanted:
        cmds.append(("hm_transport", {"action": "pause"}))
    elif "play" in wanted:
        cmds.append(("hm_transport", {"action": "play"}))

    out.confidence = min(confidences) if confidences else 0.0
    return out


# ---------------------------------------------------------------------------
# dancer role cross-check (dancers stage)
# ---------------------------------------------------------------------------

LEAD_DESCRIPTION = "black t-shirt, jeans, shaved head"
FOLLOW_DESCRIPTION = "long red hair, white crop top, light ripped jeans"


def judge_roles(cams: dict[str, dict[str, dict]]) -> dict[str, tuple[str, float]]:
    """Select (never generate) which of two measured candidates matches the lead's clothing description.

    cams: {cam: {"A": descriptors, "B": descriptors}} where descriptors are code-measured colour facts
    (colour names + HSV medians + red-pixel fraction above the shoulders). Only clothing/hair colour is
    compared; nothing else about the people is asked or inferred. Returns {cam: ("A"|"B"|"unclear", p)}.
    """
    state = {"lead_clothing": LEAD_DESCRIPTION, "partner_clothing": FOLLOW_DESCRIPTION,
             "measurement_notes": "top/trousers = median brightness of the pixels (black < dark < medium < light "
                                  "< white); red_above_shoulders = share of saturated red pixels around the head",
             "cameras": cams}
    q = {f"cam_{c}": Choice(
        instructions=f"In camera {c}, which measured candidate's colours match the clothing description "
                     f"'{LEAD_DESCRIPTION}' (the other should match '{FOLLOW_DESCRIPTION}')?",
        criteria={"A": "candidate A matches the black t-shirt / jeans description",
                  "B": "candidate B matches the black t-shirt / jeans description",
                  "unclear": "the measurements do not distinguish them"}) for c in cams}
    r = client().system_one(state=state, questions=q)
    return {c: (r.choices[f"cam_{c}"].choice, float(r.choices[f"cam_{c}"].confidence)) for c in cams}
