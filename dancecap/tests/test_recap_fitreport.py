"""Pure parts of dancecap.recap_fitreport (agent F1): summarise a pose_quality json, apply the gates, print the table."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dancecap import recap_fitreport as FR  # noqa: E402


def good_quality() -> dict:
    def cam(m, p):
        return {"lead": {"median": m, "p90": p, "n": 900}, "follow": {"median": m + 0.5, "p90": p + 1, "n": 900}}
    return {
        "frames": 1648, "span_s": [0.0, 54.9],
        "reproj_px": {"whole": {"00": cam(4.0, 7.0), "01": cam(4.5, 8.0), "05": cam(9.0, 14.0)},
                      "all_cams_per_frame_median": {"lead": {"median": 5.3, "p90": 7.6}, "follow": {"median": 5.8, "p90": 8.4}}},
        "reproj_feet_px": {"all_cams_per_frame_median": {"lead": {"median": 6.8}, "follow": {"median": 6.6}}},
        "jitter_4p5_12hz_mm": {"lead": {"whole_rms": 0.5}, "follow": {"whole_rms": 0.7}},
        "floor": {"lead": {"frac_below_15mm": 0.0, "frac_above_30mm": 0.27}, "follow": {"frac_below_15mm": 0.0, "frac_above_30mm": 0.15}},
        "rest_height_m": {"lead": {"fit": 1.768, "rel_vs_target": 0.009}, "follow": {"fit": 1.668, "rel_vs_target": -0.005}},
    }


def test_summarise_reads_the_numbers_the_gates_need():
    s = FR.summarise(good_quality())
    assert s["frames"] == 1648 and set(s["dancers"]) == {"lead", "follow"}
    L = s["dancers"]["lead"]
    assert L["allcam_median_px"] == 5.3 and L["allcam_p90_px"] == 7.6 and L["cam_median_px"]["05"] == 9.0
    assert L["rest_height_rel"] == 0.009 and L["jitter_rms_mm"] == 0.5 and L["frac_below_15mm"] == 0.0
    assert FR.gate_failures(s) == []


def test_gates_report_each_failure_and_loosen_the_attached_phones():
    q = good_quality()
    q["reproj_px"]["whole"]["05"]["lead"]["median"] = 12.0      # attached phone: allowed up to 14
    q["reproj_px"]["whole"]["00"]["lead"]["median"] = 12.0      # main phone: not allowed above 10
    q["reproj_px"]["all_cams_per_frame_median"]["follow"]["median"] = 9.5
    q["rest_height_m"]["follow"]["rel_vs_target"] = -0.05
    q["floor"]["lead"]["frac_below_15mm"] = 0.2
    bad = FR.gate_failures(FR.summarise(q))
    text = " | ".join(bad)
    assert "lead: camera 00 median 12.0" in text and "camera 05" not in text
    assert "follow: allcam_median_px 9.50" in text and "follow: rest height off by -5.0 %" in text and "lead: frac_below_15mm 0.20" in text


def test_missing_numbers_are_failures_not_silent_passes():
    q = copy.deepcopy(good_quality())
    del q["reproj_px"]["all_cams_per_frame_median"]
    bad = FR.gate_failures(FR.summarise(q))
    assert any("allcam_median_px missing" in b for b in bad)


def test_table_has_one_row_per_chunk_and_marks_chunks_without_quality():
    rep = {"chunks": {"g0000": {"state": "pass", "failures": [], **FR.summarise(good_quality())}, "g1588": {"state": "no quality json yet"}}}
    t = FR.table(rep).splitlines()
    assert len(t) == 4 and t[2].startswith("| g0000 | pass | 1648 |") and "5.3 / 7.6" in t[2] and t[3].startswith("| g1588 | no quality json yet")
