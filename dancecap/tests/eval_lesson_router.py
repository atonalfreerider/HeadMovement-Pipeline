"""Live evaluation of the TypeSafe lesson-command router against hand-labelled requests.

    python tests/eval_lesson_router.py      (needs TYPESAFE_API_KEY or ~/Desktop/jev-key.txt)

Each case lists the hm_* commands that must appear (order-insensitive, exact args) and whether a
clarification is expected. Prints per-case results and accuracy; exits 1 below the target.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dancecap.judgments import route_lesson_command  # noqa: E402

LESSON = {"captures": ["00_SyntheticDemo", "LarissaKadu"], "capture": "LarissaKadu", "measure": 2,
          "measureCount": 10, "speed": 1.0, "playing": True, "loopMeasure": False}

CASES = [
    ("pause", [("hm_transport", {"action": "pause"})], False),
    ("hold on, freeze it so I can walk around them", [("hm_transport", {"action": "pause"})], False),
    ("play it again from the top", [("hm_transport", {"action": "restart"}), ("hm_transport", {"action": "play"})], False),
    ("loop measure 3 at half speed", [("hm_transport", {"action": "measure", "measure": 3}),
                                      ("hm_transport", {"action": "loop_on"}),
                                      ("hm_transport", {"action": "speed", "speed": 0.5})], False),
    ("go to the fifth eight-count", [("hm_transport", {"action": "measure", "measure": 5})], False),
    ("hide the room and the camera frustums", [("hm_layer", {"layer": "splats", "visible": False}),
                                               ("hm_layer", {"layer": "cameras", "visible": False})], False),
    ("show me the footprints from above", [("hm_layer", {"layer": "floor", "visible": True}),
                                           ("hm_orbit", {"radius": 3.5, "azimuth": 0.0, "elevation": 80.0})], False),
    ("back one beat", [("hm_transport", {"action": "beat-"})], False),
    ("a little slower please", [("hm_transport", {"action": "speed", "speed": 0.75})], False),
    ("normal speed", [("hm_transport", {"action": "speed", "speed": 1.0})], False),
    ("next measure", [("hm_transport", {"action": "measure+"})], False),
    ("zoom in on them", [("hm_orbit", {"radius": 1.8})], False),
    ("open the synthetic demo", [("hm_load", {"name": "00_SyntheticDemo"})], False),
    ("turn off the tension lines", [("hm_layer", {"layer": "tension", "visible": False})], False),
    ("what's the weather tomorrow", [], True),
]


def norm(cmds):
    return sorted(json.dumps([c, a], sort_keys=True) for c, a in cmds)


def main():
    ok = 0
    for text, expected, expect_clarify in CASES:
        r = route_lesson_command(text, LESSON)
        got_cmds = norm(r.commands)
        want = norm(expected)
        cmd_ok = got_cmds == want if not expect_clarify else not r.commands or r.clarify is not None
        clarify_ok = (r.clarify is not None) == expect_clarify
        passed = cmd_ok and clarify_ok
        ok += passed
        print(f"{'PASS' if passed else 'FAIL'} conf={r.confidence:.2f} {text!r}")
        if not passed:
            print(f"     got  {r.commands} clarify={r.clarify!r}")
            print(f"     want {expected} clarify={expect_clarify}")
    acc = ok / len(CASES)
    print(f"\n{ok}/{len(CASES)} = {acc:.0%}")
    sys.exit(0 if acc >= 0.85 else 1)


if __name__ == "__main__":
    main()
