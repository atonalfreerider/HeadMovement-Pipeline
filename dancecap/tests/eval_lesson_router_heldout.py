"""Held-out phrases for the lesson router - written after tuning on eval_lesson_router.py and NOT used to tune.
Report this score as the router's generalisation estimate."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_lesson_router as base  # noqa: E402

base.CASES = [
    ("stop", [("hm_transport", {"action": "pause"})], False),
    ("can we see measure seven again and again", [("hm_transport", {"action": "measure", "measure": 7}),
                                                  ("hm_transport", {"action": "loop_on"})], False),
    ("ok stop looping", [("hm_transport", {"action": "loop_off"})], False),
    ("bring the floor marks back", [("hm_layer", {"layer": "floor", "visible": True})], False),
    ("view from the side", [("hm_orbit", {"radius": 3.0, "azimuth": 0.0, "elevation": 15.0})], False),
    ("rewind one eight count", [("hm_transport", {"action": "measure-"})], False),
    ("full speed and play", [("hm_transport", {"action": "speed", "speed": 1.0}),
                             ("hm_transport", {"action": "play"})], False),
    ("get rid of the text on screen", [("hm_layer", {"layer": "hud", "visible": False})], False),
    ("pull the camera back", [("hm_orbit", {"radius": 6.0})], False),
    ("who is dancing", [], True),
]

if __name__ == "__main__":
    base.main()
