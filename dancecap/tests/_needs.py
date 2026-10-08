"""Skip helpers for tests that read the real take configuration or git-ignored local data.

A clean checkout has neither: the take configs (dancecap/takes/*.toml) are committed apart from the code on purpose
(they hold dancer-specific data and may be dropped from a push), and run outputs, models and footage (dancecap/work/,
mamma-turing/, prototype/, ...) are git-ignored. A test that needs them must SKIP with a reason there, never fail.

    from _needs import needs_take, needs_paths

    @needs_take("larissa_kadu_full")
    def test_something_about_the_real_take(): ...

    @needs_paths(WORKSPACE / "prototype" / "x", why="prototype run output")
    def test_something_about_a_local_run(): ...
"""
from __future__ import annotations

from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]      # .../dancecap
WORKSPACE = PACKAGE_ROOT.parent                         # the repository root (holds dancecap/, prototype/, mamma-turing/ ...)


def take_file(name: str) -> Path:
    return PACKAGE_ROOT / "takes" / f"{name}.toml"


def needs_take(*names: str):
    """skipif mark: the named take configurations (dancecap/takes/<name>.toml) are not in this checkout."""
    missing = [n for n in names if not take_file(n).is_file()]
    return pytest.mark.skipif(bool(missing), reason="take config not in this checkout: %s" % ", ".join(missing))


def footage_present(name: str) -> bool:
    """True when every source video of the take (git-ignored footage) is on disk."""
    try:
        from dancecap.take import Take
        take = Take.load(name)
        return bool(take.sources) and all(Path(s.path).is_file() for s in take.sources)
    except Exception:
        return False


def needs_footage(name: str):
    """skipif mark: the take's source videos are not on this machine (footage never goes through git)."""
    return pytest.mark.skipif(not footage_present(name), reason="source footage of %s not present" % name)


def needs_paths(*paths, why: str = "git-ignored local data"):
    """skipif mark: some of these (git-ignored) paths do not exist in this checkout."""
    missing = [str(p) for p in paths if not Path(p).exists()]
    return pytest.mark.skipif(bool(missing), reason="%s not present: %s" % (why, ", ".join(Path(m).name for m in missing)))
