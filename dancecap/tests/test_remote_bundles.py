"""remote_jobs.prune_bundles: keeps the newest bundles and survives a concurrent session deleting a bundle mid-prune."""
import os
import time
from pathlib import Path

from dancecap import remote_jobs as rj


def _bundles(tmp_path, n, age_s):
    now = time.time()
    out = []
    for i in range(n):
        p = tmp_path / f"bundle_{i:012x}.tar.gz"
        p.write_bytes(b"x")
        p.with_suffix("").with_suffix(".json").write_text("{}")
        os.utime(p, (now - age_s - i, now - age_s - i))     # bundle 0 is the newest
        out.append(p)
    return out


def test_prune_keeps_the_newest_and_skips_young_ones(tmp_path):
    ps = _bundles(tmp_path, 5, age_s=10)
    assert rj.prune_bundles(tmp_path, keep=3, min_age_s=3600) == []          # all younger than the minimum age
    gone = rj.prune_bundles(tmp_path, keep=3, min_age_s=0)
    assert sorted(gone) == sorted(p.name for p in ps[3:])
    assert all(p.exists() for p in ps[:3]) and not any(p.exists() for p in ps[3:])


def test_prune_survives_a_bundle_deleted_between_glob_and_stat(tmp_path, monkeypatch):
    ps = _bundles(tmp_path, 5, age_s=10)
    vanished = ps[4]
    real_stat = Path.stat

    def stat(self, *a, **k):
        if self == vanished:
            raise FileNotFoundError(2, "deleted by another session", str(self))
        return real_stat(self, *a, **k)

    monkeypatch.setattr(Path, "stat", stat)
    gone = rj.prune_bundles(tmp_path, keep=3, min_age_s=0)
    assert gone == [ps[3].name]
