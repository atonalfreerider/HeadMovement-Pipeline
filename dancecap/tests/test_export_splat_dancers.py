"""takes/<take>.toml [export] splat_dancers -> capture.json splat_dancers (owner 2026-10-23: splat dancers by default for Carlos &
Aline), and export_unity's install keeping an existing splats/ folder (a re-export must not delete the splat dancers).

    .venv-nopo4d/Scripts/python.exe -m pytest dancecap/tests/test_export_splat_dancers.py -q -p no:cacheprovider
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dancecap import export_unity as eu  # noqa: E402
from dancecap.take import PACKAGE_ROOT  # noqa: E402


def _toml(tmp_path, body: str) -> Path:
    p = tmp_path / "t.toml"
    p.write_text(body, encoding="utf-8")
    return p


def test_absent_key_means_no_flag(tmp_path):
    assert eu.resolve_splat_dancers("x", _toml(tmp_path, '[export]\nsmplx_dir = "smplx"\n')) is None
    assert eu.resolve_splat_dancers("x", _toml(tmp_path, 'name = "x"\n')) is None
    assert eu.resolve_splat_dancers("x", tmp_path / "missing.toml") is None


@pytest.mark.parametrize("body,want", [('"on"', "on"), ('"off"', "off"), ('"ON"', "on"), (' " Off "', "off"),
                                       ("true", "on"), ("false", "off")])
def test_values(tmp_path, body, want):
    assert eu.resolve_splat_dancers("x", _toml(tmp_path, f"[export]\nsplat_dancers = {body}\n")) == want


@pytest.mark.parametrize("bad", ['"yes"', "1", '["on"]', '""'])
def test_bad_values_are_refused(tmp_path, bad):
    with pytest.raises(ValueError):
        eu.resolve_splat_dancers("x", _toml(tmp_path, f"[export]\nsplat_dancers = {bad}\n"))


def test_carlos_aline_asks_for_splats_and_the_recap_does_not():
    if not (PACKAGE_ROOT / "takes" / "carlos_aline.toml").exists():
        pytest.skip("take TOML not in this checkout")
    assert eu.resolve_splat_dancers("carlos_aline") == "on"
    for take in ("larissa_kadu_recap", "john_lula"):
        if (PACKAGE_ROOT / "takes" / f"{take}.toml").exists():
            assert eu.resolve_splat_dancers(take) is None


def _export_dir(root: Path) -> Path:
    out = root / "export" / "07_X"
    out.mkdir(parents=True)
    (out / "capture.json").write_text(json.dumps({"version": 3, "splat_dancers": "on"}))
    (out / "smplx_lead.bin").write_bytes(b"new")
    (out / "sub").mkdir()
    (out / "sub" / "a.txt").write_text("new")
    return out


def test_install_keeps_splats_and_replaces_the_rest(tmp_path):
    out = _export_dir(tmp_path)
    dst = tmp_path / "StreamingAssets" / "07_X"
    (dst / "splats" / "quest_15k").mkdir(parents=True)
    (dst / "splats" / "lead.splatav").write_bytes(b"SPAV-old")
    (dst / "splats" / "quest_15k" / "lead.json").write_text("{}")
    (dst / "splats.meta").write_text("meta")
    (dst / "smplx_lead.bin").write_bytes(b"old")
    (dst / "stale.json").write_text("old")
    (dst / "sub").mkdir()
    (dst / "sub" / "b.txt").write_text("old")
    kept = eu.install_capture(out, dst)
    assert kept == ["splats", "splats.meta"]
    assert (dst / "splats" / "lead.splatav").read_bytes() == b"SPAV-old"
    assert (dst / "splats" / "quest_15k" / "lead.json").exists() and (dst / "splats.meta").exists()
    assert (dst / "smplx_lead.bin").read_bytes() == b"new"
    assert not (dst / "stale.json").exists() and not (dst / "sub" / "b.txt").exists()
    assert (dst / "sub" / "a.txt").read_text() == "new"
    assert json.loads((dst / "capture.json").read_text())["splat_dancers"] == "on"


def test_install_into_a_new_folder(tmp_path):
    out = _export_dir(tmp_path)
    dst = tmp_path / "SA" / "07_X"
    assert eu.install_capture(out, dst) == []
    assert sorted(p.name for p in dst.iterdir()) == ["capture.json", "smplx_lead.bin", "sub"]


def test_install_refuses_to_merge_an_export_that_carries_splats(tmp_path):
    out = _export_dir(tmp_path)
    (out / "splats").mkdir()
    dst = tmp_path / "SA" / "07_X"
    (dst / "splats").mkdir(parents=True)
    with pytest.raises(ValueError):
        eu.install_capture(out, dst)


# ---------------------------------------------------------------- fix round 1 (2026-10-23): verifier F4 / F5
def test_refused_install_leaves_the_destination_as_it_was(tmp_path):
    out = _export_dir(tmp_path)
    (out / "splats").mkdir()
    dst = tmp_path / "SA" / "07_X"
    (dst / "splats").mkdir(parents=True)
    (dst / "splats" / "follow.splatav").write_bytes(b"splat")
    (dst / "capture.json").write_text("old")
    (dst / "cameras").mkdir()
    (dst / "cameras" / "c.json").write_text("cam")
    (dst / "smplx_lead.bin").write_bytes(b"old")
    before = sorted((p.relative_to(dst).as_posix(), p.read_bytes() if p.is_file() else None) for p in dst.rglob("*"))
    with pytest.raises(ValueError):
        eu.install_capture(out, dst)
    after = sorted((p.relative_to(dst).as_posix(), p.read_bytes() if p.is_file() else None) for p in dst.rglob("*"))
    assert after == before                                   # nothing deleted before the refusal


def test_manifest_extras_reads_the_toml_in_order(tmp_path):
    t = _toml(tmp_path, '[export]\nsplat_dancers = "on"\nfloor_yaw_deg = -1.0\n'
                        'role_hidden = { lead = [ { from = 0.0, to = 4.0, fade_in = 1.0 } ] }\n')
    ex = eu.manifest_extras("x", t)
    assert list(ex) == ["floor_yaw_deg", "role_hidden", "splat_dancers"]
    assert ex["floor_yaw_deg"] == -1.0 and ex["splat_dancers"] == "on"
    assert ex["role_hidden"] == {"lead": [{"from": 0.0, "to": 4.0, "fade_in": 1.0}]}
    assert eu.manifest_extras("x", _toml(tmp_path, '[export]\nsmplx_dir = "s"\n')) == {}


def test_export_writes_the_manifest_extras():
    """export() builds capture.json's TOML keys through manifest_extras (the path TOML -> capture.json)."""
    import inspect
    src = inspect.getsource(eu.export)
    assert "manifest.update(manifest_extras(take))" in src
    for key in ("floor_yaw_deg", "role_hidden", "props", "splat_dancers"):
        assert f'manifest["{key}"]' not in src               # no second, divergent path


@pytest.mark.parametrize("capture,take", [("06_LarissaKaduRecap", "larissa_kadu_recap"), ("07_CarlosAline", "carlos_aline"),
                                          ("08_JohnLula", "john_lula")])
def test_installed_captures_carry_exactly_the_manifest_extras(capture, take):
    cj = PACKAGE_ROOT.parent / "head-movement" / "Assets" / "StreamingAssets" / capture / "capture.json"
    if not cj.exists() or not (PACKAGE_ROOT / "takes" / f"{take}.toml").exists():
        pytest.skip("capture data / take TOML not in this checkout (git-ignored likeness data)")
    d = json.loads(cj.read_text())
    ex = json.loads(json.dumps(eu.manifest_extras(take)))
    assert {k: d[k] for k in ("floor_yaw_deg", "role_hidden", "props", "splat_dancers") if k in d} == ex
