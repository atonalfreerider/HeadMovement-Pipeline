"""farm refit of a chunk whose spec already masks a camera (a two-view profile's cam_masks): the refit keeps those masks."""
from dancecap.farm import recap_refit as RR


def _base():
    steps = [{"name": n, "cmd": f"python -m dancecap.x {n}", "timeout_min": 5, "optional": False, "stage": "front"}
             for n in ("chunk_take", "front")]
    steps.append({"name": "mask", "cmd": "python -m dancecap.recap_front mask ll2g4320 --mask 01:163:176,01:206:212.633", "timeout_min": 10,
                  "optional": False, "stage": "front", "save": ["dancecap/work/lucas_luiza_ll2g4320/mamma/ma_2d/"]})
    steps += [{"name": n, "cmd": f"python -m dancecap.x {n}", "timeout_min": 5, "optional": False, "stage": "fit"}
              for n in ("qa_mask", "await_pin", "fit", "refine", "quality", "evidence")]
    return {"id": "ll2g4320", "steps": steps, "inputs": [], "estimate": {}}


def test_refit_merges_the_profile_masks_with_its_own():
    spec = RR.refit_spec(_base(), [("02", 180.0, 185.0), ("01", 163.0, 176.0)], "chunks/ll2g4320/ckpt/front.tar", "r1", capture="lucas-luisa-jnj")
    mask = next(s for s in spec["steps"] if s["name"] == "mask")
    assert mask["cmd"].endswith("--mask 01:163:176,01:206:212.633,02:180:185") and mask["save"]
    assert [m["cam"] for m in spec["refit"]["masks"]] == ["01", "01", "02"]
    assert [s["name"] for s in spec["steps"]][:3] == ["chunk_take", "mask", "qa_mask"]
