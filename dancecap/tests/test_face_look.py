"""face_look: lip / beard measures on a synthetic face (no dancer data)."""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from dancecap import face_look as fl  # noqa: E402


def _synthetic(lips_bgr=(60, 40, 200), chin_bgr=(40, 50, 60)):
    """200 px image, skin background, a lip ring, a dark chin patch; MediaPipe-indexed landmarks for the used points."""
    img = np.full((200, 200, 3), (150, 170, 210), np.uint8)          # BGR skin
    lmk = np.zeros((478, 3))
    lmk[33, :2], lmk[263, :2] = (70, 80), (130, 80)                  # IOD 60
    lmk[50, :2], lmk[280, :2] = (65, 110), (135, 110)                # cheeks
    ang = np.linspace(0, 2 * np.pi, len(fl.LIPS_OUTER), endpoint=False)
    lmk[fl.LIPS_OUTER, 0], lmk[fl.LIPS_OUTER, 1] = 100 + 20 * np.cos(ang), 140 + 9 * np.sin(ang)
    lmk[fl.LIPS_INNER, 0], lmk[fl.LIPS_INNER, 1] = 100 + 12 * np.cos(ang), 140 + 2 * np.sin(ang)
    lmk[17, :2], lmk[152, :2] = (100, 149), (100, 185)               # lower lip bottom, chin tip
    lmk[0, :2], lmk[2, :2] = (100, 131), (100, 118)                  # upper lip top, nose base
    rg = fl.regions(img.shape, lmk)
    img[rg["lips"]] = lips_bgr
    cv2.circle(img, (100, 167), 12, chin_bgr, -1)
    return img, lmk


def test_measure_sees_red_lips_and_dark_chin():
    img, lmk = _synthetic()
    r = fl.measure(img, lmk)
    assert r["iod"] == pytest.approx(60.0)
    assert r["lip_da"] > 15 and r["lip_chroma_ratio"] > 1.2
    assert r["chin_dL"] < -30
    assert abs(r["must_dL"]) < 3                                     # moustache region is plain skin here


def test_degrade_scales_landmarks_and_keeps_signal():
    img, lmk = _synthetic()
    small, lm2 = fl.degrade(img, lmk, target_iod=30.0)
    assert small.shape[0] == 100 and fl.iod_of(lm2) == pytest.approx(30.0)
    r = fl.measure(small, lm2)
    assert r is not None and r["lip_da"] > 5 and r["chin_dL"] < -15


def test_compare_flags_only_real_differences():
    views = [{"lip_da": v, "lip_dL": -10.0, "lip_chroma_ratio": 1.0, "chin_dL": -25.0, "must_dL": -30.0}
             for v in (-2.0, -1.0, 0.0, 1.0, 2.0)]
    p = {"lip_da": 12.0, "lip_dL": -11.0, "lip_chroma_ratio": 1.05, "chin_dL": -50.0, "must_dL": -31.0}
    c = fl.compare(p, views)
    assert c["lip_da"]["differs"] and c["chin_dL"]["differs"]
    assert not c["lip_dL"]["differs"] and not c["must_dL"]["differs"] and not c["lip_chroma_ratio"]["differs"]
