"""sides config + take handling (DESIGN 8): SidesCfg and the [sides] TOML table, the adapter choice from the chunks' select.json unions
(synthetic folders for all four kinds; the real takes when their work folders exist), and no take constant inside the package.

    .venv-nopo4d\\Scripts\\python.exe -m pytest dancecap/tests/test_sides_takes.py -q -p no:cacheprovider     (from the workspace root)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from dancecap.sides import adapters
from dancecap.sides.config import SidesCfg, apply_overrides, load_cfg

ROOT = Path(__file__).resolve().parents[1]
SIDES_DIR = ROOT / "dancecap" / "sides"
WORK = ROOT / "work"


def test_cfg_defaults_and_toml_table():
    c = SidesCfg()
    assert c.identity_gate_m == 0.60 and c.prior_nats == 0.5 and c.blind_prior_nats == 1.5    # D1 (b) is the default
    assert c.p_flip_band == 1e-6 and c.post_decide == 0.9 and c.post_exit == 0.7
    txt = """
name = "x"
[sides]
blind_prior_nats = 0.5
inlier_px = 12
obs_dirs = ["a/obs"]
[sides.priors.R4.face]
back = 1.0
"""
    c2 = load_cfg(toml_text=txt)
    assert c2.blind_prior_nats == 0.5 and c2.inlier_px == 12.0 and isinstance(c2.inlier_px, float)
    assert c2.obs_dirs == ("a/obs",)
    assert c2.priors["R4"]["face"]["back"] == 1.0 and "away" in c2.priors["R4"]
    with pytest.raises(KeyError):
        load_cfg(toml_text="[sides]\nno_such_key = 1\n")
    c3 = apply_overrides(c, {"workers": 4})
    assert c3.workers == 4 and c.workers == 1
    assert json.dumps(c.to_json())          # serialisable (written into evidence.json / relations.json)
    # a take TOML without [sides] gives the defaults
    assert load_cfg(toml_text='name = "y"\n') == SidesCfg()


def _chunked(root: Path, name: str, cams_per_chunk: list) -> Path:
    w = root / name
    w.mkdir()
    chunks = []
    for i, cams in enumerate(cams_per_chunk):
        nm = f"c{i}"
        chunks.append({"name": nm})
        cw = root / f"{name}_{nm}" / "mamma"
        cw.mkdir(parents=True)
        (cw / "select.json").write_text(json.dumps({"cams": cams}))
    (w / "stitch.json").write_text(json.dumps({"chunks": chunks}))
    return w


def test_adapter_kind_from_select_unions(tmp_path):
    assert adapters.take_kind(_chunked(tmp_path, "m", [["00", "01", "02", "05"], ["00", "01", "02", "03", "05", "06"]])) == "multi"
    assert adapters.take_kind(_chunked(tmp_path, "t", [["01", "02"], ["01", "02"]])) == "two"
    assert adapters.take_kind(_chunked(tmp_path, "o", [["01"], ["01"], ["01"]])) == "mono"
    s = tmp_path / "s"
    (s / "mamma").mkdir(parents=True)
    (s / "mamma" / "select.json").write_text(json.dumps({"cams": ["01", "02", "03"]}))
    assert adapters.take_kind(s) == "single_set"
    with pytest.raises(FileNotFoundError):
        adapters.take_kind(tmp_path / "nothing")
    with pytest.raises(NotImplementedError):
        adapters.adapter_for("t", SidesCfg(), tmp_path / "t")


@pytest.mark.parametrize("take,kind", [("larissa_kadu_recap", "multi"), ("carlos_aline", "two"), ("john_lula", "mono"),
                                       ("larissa_kadu_38", "single_set")])
def test_adapter_kind_of_the_real_takes(take, kind):
    w = WORK / take
    if not ((w / "stitch.json").exists() or (w / "mamma" / "select.json").exists()):
        pytest.skip(f"work/{take} is absent")
    assert adapters.take_kind(w) == kind


def test_no_take_constants_in_the_package():
    names = sorted(p.stem for p in (ROOT / "takes").glob("*.toml"))
    assert names, "takes/*.toml not found"
    pat = re.compile(r"\b(" + "|".join(re.escape(n) for n in names) + r")\b")
    for p in SIDES_DIR.glob("*.py"):
        src = p.read_text(encoding="utf-8")
        assert not pat.search(src), (p.name, pat.search(src).group(0))
        for bad in ("hc.TAKE", "zc.OUT", "jl_facing.TAKE", "hc.WORK", "zc.WORK", "hc.OUT", "hc.SRC_FIT", "hc.CAMS"):
            assert bad not in src, (p.name, bad)


def test_unwrap_and_splice_values_come_from_sidescfg():
    """one source of truth: SidesCfg's defaults equal the module dataclasses' and a [sides] override reaches them"""
    from dataclasses import asdict
    from dancecap.sides.splice import SpliceCfg
    from dancecap.sides.unwrap import UnwrapCfg
    c = SidesCfg()
    assert asdict(c.unwrap_cfg()) == asdict(UnwrapCfg())
    assert asdict(c.splice_cfg()) == asdict(SpliceCfg())
    c2 = load_cfg(toml_text="[sides]\nunwrap_guard_mode = \"gate\"\nsplice_exit_hold_s = 0.5\nspin_err_deg = 40\n")
    assert c2.unwrap_cfg().guard_mode == "gate" and c2.splice_cfg().exit_hold_s == 0.5 and c2.splice_cfg().defect_deg == 40.0


def test_work_cfg_json_layer(tmp_path, monkeypatch):
    """work/<take>/sides/cfg.json (take data kept out of the tracked TOML) applies after the TOML table and before explicit overrides"""
    from dancecap.sides import config as cf
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps({"_what": "x", "plan_owner_windows": [[1.0, 2.0]], "blind_prior_nats": 0.5}), encoding="utf-8")
    monkeypatch.setattr(cf, "work_cfg_path", lambda take: p)
    monkeypatch.setattr(cf, "toml_path", lambda take: tmp_path / "none.toml")
    c = cf.load_cfg("anytake")
    assert c.plan_owner_windows == ([1.0, 2.0],) and c.blind_prior_nats == 0.5
    c = cf.load_cfg("anytake", {"blind_prior_nats": 1.5})
    assert c.blind_prior_nats == 1.5
    assert cf.load_cfg("anytake", work_overrides=False) == SidesCfg()
