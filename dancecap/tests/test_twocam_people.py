"""dancecap.cameras.twocam_people: dancers out of all detections; fractional-frame interpolation."""
import numpy as np

from dancecap.cameras import twocam_people as TP


def _det():
    # frame 0: judge (small box), white dancer, red dancer; frame 1: only the white dancer; frame 2: nobody big
    box = np.array([[0, 0, 40, 200], [100, 50, 220, 450], [300, 60, 420, 440],     # frame 0
                    [110, 50, 230, 450], [0, 0, 40, 200],                          # frame 1
                    [0, 0, 40, 200]], np.float32)                                  # frame 2
    frame = np.array([0, 0, 0, 1, 1, 2])
    rgb = np.array([[90, 90, 90], [210, 205, 200], [190, 40, 50], [212, 206, 201], [90, 90, 90], [90, 90, 90]], np.float32)
    kp = np.zeros((6, 17, 3), np.float32)
    for k in range(6):
        kp[k, :, 0] = box[k, 0] + 10
        kp[k, :, 1] = box[k, 1] + np.arange(17) * 5
        kp[k, :, 2] = 0.9
    return {"frame": frame, "box": box, "conf": np.full(6, .9, np.float32), "kp": kp, "rgb": rgb}


def test_dancers_picked_by_size_and_told_apart_by_brightness():
    tab = TP.dancer_table(_det(), 3, h_min=300)
    assert tab["ok"][0].all()
    assert tab["kp"][0, TP.CARLOS, 0, 0] == 110 and tab["kp"][0, TP.ALINE, 0, 0] == 310     # white = Carlos
    assert tab["ok"][1].tolist() == [True, False]          # a lone bright box is Carlos
    assert not tab["ok"][2].any()                          # the judge is not a dancer


def test_interpolation_between_frames():
    tab = TP.dancer_table(_det(), 3, h_min=300)
    out = TP.interp_kp(tab, np.array([0.0, 0.5, 1.0]), TP.CARLOS)
    assert np.isclose(out[0, 0, 0], 110) and np.isclose(out[2, 0, 0], 120)
    assert np.isclose(out[1, 0, 0], 115)
    al = TP.interp_kp(tab, np.array([0.5]), TP.ALINE)
    assert np.isnan(al[0, 0, 0])                            # absent in frame 1 -> no interpolated Aline
