"""Shared test setup: no test reads the live Runpod balance (remote.cap() follows the account when DANCECAP_RUNPOD_CAP is
unset), so a test never calls runpodctl or writes work/runpod/balance_cap.json in the real tree."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from the workspace root too


@pytest.fixture(autouse=True)
def _no_live_runner_processes(monkeypatch):
    """No test sees the `remote run` processes of real sessions on this PC: the orchestrator waits while one exists
    (locks.external_session_runner), and each scan of ~500 processes took 3.5 s (2026-10-22: two sessions of another
    worktree were running, and test_pipeline_orchestrator ran for more than 20 min instead of seconds)."""
    try:
        from dancecap.pipeline import runpod_view
    except ImportError:
        return
    monkeypatch.setattr(runpod_view, "_runner_processes", lambda: 0)


@pytest.fixture(autouse=True)
def _no_live_runpod_balance(monkeypatch):
    monkeypatch.setenv("DANCECAP_RUNPOD_BALANCE", "off")
    try:
        from dancecap import remote
    except ImportError:
        return
    monkeypatch.setattr(remote, "_balance_memo", {})
