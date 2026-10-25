"""`python -m dancecap physics <take> --smplx-dir <dir> --out-dir <dir>` (DESIGN 4.1 / 6.1 Phase D item 3): the physics stage reads
another fit folder of the take (e.g. smplx_sides) and writes its output elsewhere, so a comparison run never touches work/<take>/physics."""
import numpy as np
import pytest


class _Stop(Exception):
    pass


def test_physics_reads_the_given_fit_folder_and_out_dir(tmp_path, monkeypatch):
    from dancecap import cli, physics
    import dancecap.take as take_mod
    work = tmp_path / "work" / "sometake"
    for d, v in (("smplx", 0.0), ("smplx_alt", 1.0)):
        (work / d).mkdir(parents=True)
        for r in ("lead", "follow"):
            np.savez(work / d / f"{r}.npz", t=np.arange(3) / 30.0, transl=np.full((3, 3), v))
    monkeypatch.setattr(take_mod, "PACKAGE_ROOT", tmp_path)
    seen = {}

    def fake_run(work_, cfg, beats, masses=None, statures=None, soles=None, npzs=None, out_dir=None, **kw):
        seen.update(work=work_, npzs=npzs, out_dir=out_dir)
        raise _Stop()
    monkeypatch.setattr(physics, "run", fake_run)
    with pytest.raises(_Stop):
        cli.main(["physics", "sometake", "--smplx-dir", "smplx_alt", "--out-dir", "sides/analyses/physics"])
    assert seen["work"] == work
    assert set(seen["npzs"]) == {"lead", "follow"} and float(seen["npzs"]["lead"]["transl"][0, 0]) == 1.0
    assert seen["out_dir"] == work / "sides" / "analyses" / "physics"
    # without the flags: physics.run reads work/<take>/smplx itself (npzs None) and writes work/<take>/physics (out_dir None)
    seen.clear()
    with pytest.raises(_Stop):
        cli.main(["physics", "sometake"])
    assert seen["npzs"] is None and seen["out_dir"] is None
    with pytest.raises(SystemExit):
        cli.main(["physics", "sometake", "--smplx-dir", "smplx_missing"])
