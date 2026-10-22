"""The one-window rule of the Pipeline Monitor (2026-10-22): launches and new jobs point the open window at their view
through work/monitor/focus.json instead of opening another window."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from the workspace root too
from dancecap.pipeline import monitor  # noqa: E402


class _P:
    def __init__(self, d):
        self.d = d

    def monitor_dir(self):
        return self.d


def test_focus_round_trip_and_only_app_routes(tmp_path, monkeypatch):
    monkeypatch.setattr(monitor, "_paths", lambda: _P(tmp_path))
    assert monitor.read_focus() == {}
    t0 = time.time()
    monitor.set_focus("#/analysis/20261022-010000-x", "analyze 20261022-010000-x")
    f = monitor.read_focus()
    assert f["route"] == "#/analysis/20261022-010000-x" and f["reason"].startswith("analyze") and f["ts"] >= t0
    monitor.set_focus("https://example.com/", "never")                 # only this app's own "#/..." views
    assert monitor.read_focus()["route"] == "#/analysis/20261022-010000-x"
    (tmp_path / "focus.json").write_text("{broken")
    assert monitor.read_focus() == {}


def test_show_reuses_the_open_window_and_closes_extras(tmp_path, monkeypatch):
    monkeypatch.setattr(monitor, "_paths", lambda: _P(tmp_path))
    opened, raised, closed = [], [], []
    monkeypatch.setattr(monitor, "open_window", lambda url, shell="edge-app": opened.append(url))
    monkeypatch.setattr(monitor, "_raise_window", lambda h: raised.append(h))
    monkeypatch.setattr(monitor, "monitor_windows", lambda: [])
    assert monitor.show(8790, "#/farm/x", "launch") == "opened" and opened == ["http://127.0.0.1:8790/#/farm/x"]
    monkeypatch.setattr(monitor, "monitor_windows", lambda: [11, 22, 33])
    import ctypes
    if sys.platform == "win32":
        monkeypatch.setattr(ctypes.windll.user32, "PostMessageW", lambda h, msg, w, l: closed.append((h, msg)))
    assert monitor.show(8790, "#/runpod", "launch") == "reused"
    assert raised == [11] and len(opened) == 1
    if sys.platform == "win32":
        assert closed == [(22, 0x0010), (33, 0x0010)]                  # WM_CLOSE to every window but the front one
    assert json.loads((tmp_path / "focus.json").read_text())["route"] == "#/runpod"


def test_monitor_windows_never_raises():
    assert isinstance(monitor.monitor_windows(), list)
