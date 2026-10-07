"""avatar_texture.camera_views poses the bodies at the frame's sync-corrected time (frames.json t + delta), the time
base of the MAMMA grid; without sync/offsets.json nothing changes."""
import json

import numpy as np

from dancecap import avatar_texture as at


def _work(tmp_path, delta_s: float | None):
    w = tmp_path / "w"
    (w / "frames" / "01").mkdir(parents=True)
    (w / "cameras").mkdir()
    F = 60
    t = 40.0 + np.arange(F) / 30.0
    (w / "frames" / "01" / "frames.json").write_text(json.dumps(
        {"camera": "01", "slope": 1.0, "offset": 3.0, "frames": [{"file": f"{i:06d}.jpg", "t": float(x)} for i, x in
                                                                 enumerate(t)]}))
    np.savez(w / "cameras" / "01.npz", t=t, K=np.tile(np.eye(3), (F, 1, 1)), R=np.tile(np.eye(3), (F, 1, 1)),
             tvec=np.zeros((F, 3)), dist=np.zeros(F), ok=np.ones(F, bool))
    if delta_s is not None:
        (w / "sync").mkdir()
        (w / "sync" / "offsets.json").write_text(json.dumps(
            {"offsets": {"01": {"delta": delta_s}}, "clocks": {"alignment": None, "sources": {"01": [1.0, 3.0]}}}))
    return w


def test_views_are_posed_at_the_sync_corrected_time(tmp_path):
    w0 = _work(tmp_path / "a", None)
    v0, _ = at.camera_views(w0, ["01"], 40.2, 41.5, 0.1)
    assert all(v["t"] == v["t_frame"] for v in v0)
    w = _work(tmp_path / "b", 0.09)
    v, _ = at.camera_views(w, ["01"], 40.2, 41.5, 0.1)
    assert v and all(abs(x["t"] - (x["t_frame"] + 0.09)) < 1e-9 for x in v)
    assert all(40.2 <= x["t"] <= 41.5 for x in v)  # the window is in corrected time
    assert {x["i"] for x in v} != {x["i"] for x in v0}  # same frame choice rule, window shifted by delta
