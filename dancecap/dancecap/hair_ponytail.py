"""Ponytail restyle of a hair groom (take-agnostic): `hair_style = "ponytail"` in a take's [dancers.<role>] block.

The demo's groom system (hair_groom.py + head-movement Assets/Hair) has one style, sleek loose hair. This module turns any groom of
the current version (roots on the scalp, colliders, render, dynamics) into a PONYTAIL groom: the same guide roots, the loose
style replaced by `style.type = "ponytail"` with a `style.ponytail` block that the viewer's Assets/Hair/HairPonytail.cs reads:

  * the hair is gathered at a tie on the back / crown of the head (tie_az_deg / tie_el_deg from the head-sphere centre; az 0 =
    facing, 180 = straight back; high = 50-60 deg elevation), the scalp hair swept back and smooth toward it (the cap's comb
    runs along great circles to the tie, the guides hug the head capsule), a clean hairline (the cap's hairline table is kept);
  * the tail leaves the tie along the tie normal drooping toward gravity, tail_length_m = tail_frac x rest height (hangs about to
    the shoulder blades for 0.20 x 1.70 m), fanned out into a sheet (flare), each strand offset in the bundle toward the side it
    entered from; cards start at the tie and turn around the tail axis (a tube, readable from every side);
  * the scalp part is pinned to the head, the tail is soft (swings with the head / body, the existing XPBD guide solver with the
    body colliders); a scrunchie ring and a small barrette (tie.* ) are drawn at the tie;
  * the card layer table is replaced (no side-part cards, fewer inner / mid / outer cards: they only cover the tail).

Every other take is unchanged: apply_ponytail only runs for `hair_style = "ponytail"` (take_hooks, called from hair_groom.seed_from
and hair_groom.restyle). Colour: avatar/hair_palette.json of the take's work dir (likeness colours stay in work/, never in a tracked
file): {"mid": [r, g, b], "highlight": [r, g, b], "root": .., "tip": .., "shadow": .., "albedo_gain": 1.85, "tip_glow": 1.0,
"copper_lift": [1.1, 1.25, 1.2], "copper_lift_m": 0.12} (sRGB 0-255); absent = the groom's own colour is kept.

    python -m dancecap.hair_ponytail <take> [--from-take <other>] [--out FILE]      # groom for the take (seed + ponytail + palette)
    python -m dancecap.hair_ponytail --check FILE                                    # validate a groom's ponytail block
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

PONYTAIL = {
    "tie_az_deg": 180.0,
    "tie_el_deg": 52.0,
    "tie_lift_m": 0.010,
    "tail_frac_of_height": 0.22,      # tail length from the tie as a fraction of the avatar's rest height
    "tail_length_jitter": 0.06,
    "tail_droop0": 0.35,
    "tail_down_slerp": 0.28,
    "bundle_radius_m": [0.030, 0.020],
    "flare": 2.4,
    "offset_ramp_m": 0.05,
    "skull_hold": 0.9,
    "tail_hold": {"root": 0.22, "falloff": 0.18, "settle_root": 0.10, "settle_falloff": 0.30},
    "deviation": 0.8,
    "card_start_back_m": 0.02,
    "neighbour_blend": 0.5,
    "skull_layer_m": 0.004,
    "tie": {"enabled": True, "band_radius_m": 0.009, "colour": [0.91, 0.84, 0.84],
            "clip": {"enabled": True, "az_deg": -158.0, "el_deg": 30.0, "angle_deg": -35.0, "length_m": 0.030, "width_m": 0.013,
                     "colour": [0.36, 0.72, 0.90]}},
}

# card layers of the ponytail: only the tail is carded (the scalp cap covers the scalp), no side-part cards
CARD_LAYERS = [
    {"name": "inner", "count": 18, "width_m": [0.040, 0.050], "offset_m": -0.003, "length_factor": [0.88, 0.92], "tiles": [0, 1],
     "bulge": 0.08, "ao": 0.72, "edge_fade": True},
    {"name": "mid", "count": 34, "width_m": [0.030, 0.036], "offset_m": 0.0, "length_factor": [0.97, 1.0], "tiles": [2, 4],
     "bulge": 0.15, "ao": 0.8, "edge_fade": True},
    {"name": "outer", "count": 40, "width_m": [0.024, 0.010], "offset_m": 0.004, "length_factor": [0.95, 1.03], "tiles": [5, 6],
     "bulge": 0.15, "ao": 1.0, "edge_fade": True, "regions": [{"name": "all", "count": 40, "abs_az": [0, 180]}]},
    {"name": "part", "count": 0, "width_m": [0.02, 0.014], "offset_m": 0.002, "length_m": [0.10, 0.16], "tiles": [2, 4], "bulge": 0.1,
     "ao": 0.9, "edge_fade": True},
    {"name": "flyaway", "count": 10, "width_m": [0.006, 0.003], "offset_m": 0.008, "length_factor": [0.90, 1.02], "tiles": [7, 7],
     "bulge": 0.0, "ao": 1.0, "edge_fade": True, "short": 0, "wobble_m": [0.006, 0.012]},
]


def deep_merge(base: dict, over: dict) -> dict:
    out = json.loads(json.dumps(base))
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def validate(p: dict) -> list[str]:
    """Problems of a style.ponytail block (empty = fine)."""
    bad = []
    if not 90.0 <= abs(float(p.get("tie_az_deg", 180))) <= 180.0:
        bad.append("tie_az_deg must be on the back half of the head (90..180)")
    if not 20.0 <= float(p.get("tie_el_deg", 52)) <= 85.0:
        bad.append("tie_el_deg must be 20..85 (nape .. crown)")
    if not 0.10 <= float(p.get("tail_length_m", 0.34)) <= 0.80:
        bad.append("tail_length_m must be 0.10..0.80")
    if not 0.0 <= float(p.get("tail_droop0", 0.35)) <= 1.0:
        bad.append("tail_droop0 must be 0..1")
    br = p.get("bundle_radius_m", [0.02, 0.015])
    if len(br) != 2 or min(br) <= 0 or max(br) > 0.06:
        bad.append("bundle_radius_m must be two radii in (0, 0.06]")
    if not 1.0 <= float(p.get("flare", 2.2)) <= 4.0:
        bad.append("flare must be 1..4")
    return bad


def srgb_to_linear(c) -> list[float]:
    out = []
    for x in c:
        x = float(x) / 255.0
        out.append(x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4)
    return [round(v, 4) for v in out]


def hex_of(c) -> str:
    return "#" + "".join(f"{int(round(float(x))):02x}" for x in c)


def apply_ponytail(groom: dict, rest_height_m: float | None = None, params: dict | None = None) -> dict:
    """A copy of a groom (version 2) restyled as a ponytail: style, cards and dynamics blocks; roots, colliders, render, colour,
    face box are kept. `params` overrides PONYTAIL entries (a nested dict merges). Guide `len` is informational only: the viewer
    builds the ponytail's rest shapes and lengths itself (HairPonytail.cs)."""
    g = json.loads(json.dumps(groom))
    h = float(rest_height_m or (g.get("model") or {}).get("rest_height_m") or 1.70)
    p = deep_merge(PONYTAIL, params or {})
    p["tail_length_m"] = round(float(p.pop("tail_frac_of_height")) * h, 4)
    problems = validate(p)
    if problems:
        raise ValueError("ponytail: " + "; ".join(problems))
    g["version"] = 2
    st = dict(g.get("style") or {})
    st["type"] = "ponytail"
    st["ponytail"] = p
    st["note"] = ("ponytail (dancecap.hair_ponytail): guide roots on the scalp, rest shapes built by the viewer from style.ponytail "
                  "(great circle over the head capsule to the tie, then the tail); the loose-hair comb keys are ignored")
    g["style"] = st
    cards = json.loads(json.dumps(g.get("cards") or {}))
    cards["layers"] = json.loads(json.dumps(CARD_LAYERS))
    cards["note"] = "ponytail card table: cards cover the tail only (start just above the tie); the scalp cap covers the scalp"
    g["cards"] = cards
    sh = g.setdefault("shape", {})
    sh["texture"] = ("high ponytail: scalp hair swept back and smooth to a tie on the crown / back of the head, the tail hangs about "
                     "to the shoulder blades, fanned out; hair tie + small barrette")
    dyn = dict(g.get("dynamics") or {})
    dyn["ponytail"] = {"note": "scalp part pinned (style.ponytail.skull_hold), tail soft (tail_hold), deviation cone style.ponytail.deviation"}
    g["dynamics"] = dyn
    g["ponytail"] = {"applied": time.strftime("%Y-%m-%dT%H:%M:%S"), "rest_height_m": round(h, 4)}
    return g


def recolour(g: dict, palette: dict) -> dict:
    """The groom's colour replaced by a palette {mid, highlight[, root, tip, shadow]} (sRGB 0-255) + render options."""
    g = json.loads(json.dumps(g))
    col = g.setdefault("colour", {})
    for k in ("root", "mid", "tip", "shadow", "highlight"):
        if k in palette:
            s = [int(round(float(x))) for x in palette[k]]
            col[k] = {**(col.get(k) if isinstance(col.get(k), dict) else {}), "linear_srgb": srgb_to_linear(s), "srgb8": s, "hex": hex_of(s)}
    col["source"] = "hair_ponytail.recolour: take palette (work dir)"
    r = g.setdefault("render", {})
    gain = float(palette.get("albedo_gain", r.get("albedo_gain", 1.85)))
    alb = [round(min(0.95, x * gain), 4) for x in col["mid"]["linear_srgb"]]
    r["albedo_gain"] = gain
    r["albedo_linear"] = alb
    r["albedo_portrait_linear"] = alb
    r["albedo_default"] = "video"
    r["albedo_source"] = r["albedo_portrait_source"] = "palette.mid x albedo_gain"
    kk = r.setdefault("kajiya_kay", {})
    if "highlight" in col:
        kk["highlight_linear"] = col["highlight"]["linear_srgb"]
        kk["highlight_portrait_linear"] = col["highlight"]["linear_srgb"]
    if "cap_tint" in palette:                       # the hair at the scalp reads darker than the tail (photo): a multiplier on the cap
        r["cap_tint"] = [float(x) for x in palette["cap_tint"]]
    if "copper_lift" in palette:
        r["copper_lift"] = [float(x) for x in palette["copper_lift"]]
        r["copper_lift_m"] = float(palette.get("copper_lift_m", r.get("copper_lift_m", 0.1)))
    glow = float(palette.get("tip_glow", 1.0))
    if glow != 1.0 and isinstance(r.get("tip_emission", {}).get("e"), list):
        r["tip_emission"]["e"] = [round(float(x) * glow, 5) for x in r["tip_emission"]["e"]]
        r["tip_emission"]["scaled_by"] = glow
    g["recoloured"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    return g


def hair_style_of(take) -> str:
    """[dancers.follow] hair_style of a take's TOML ('' = the default loose style)."""
    return str(((take.cfg.get("dancers") or {}).get("follow") or {}).get("hair_style") or "")


def hair_params_of(take) -> dict:
    """[dancers.follow.hair_ponytail] overrides of PONYTAIL (nested tables merge)."""
    return dict(((take.cfg.get("dancers") or {}).get("follow") or {}).get("hair_ponytail") or {})


def take_hooks(g: dict, take) -> dict:
    """Called by hair_groom.seed_from / restyle after the loose-style restyle: a take whose follow has hair_style = "ponytail" gets the
    ponytail restyle (+ its work/avatar/hair_palette.json colour); any other take is returned unchanged."""
    if hair_style_of(take) != "ponytail":
        return g
    g = apply_ponytail(g, params=hair_params_of(take))
    pal = Path(take.work) / "avatar" / "hair_palette.json"
    own = (take.cfg.get("dancers") or {}).get("follow") or {}
    if pal.exists():
        palette = json.loads(pal.read_text(encoding="utf-8"))
        for key, name in (("hair_tip_glow", "tip_glow"), ("hair_albedo_gain", "albedo_gain")):   # the TOML flips the palette's options
            if key in own:
                palette[name] = float(own[key])
        g = recolour(g, palette)
    return g


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m dancecap.hair_ponytail", description=__doc__.split("\n")[0])
    ap.add_argument("take", nargs="?")
    ap.add_argument("--from-take", help="seed the groom from this other take's groom (hair_groom.seed_from)")
    ap.add_argument("--check", help="validate the ponytail block of this groom file")
    ap.add_argument("--no-export", action="store_true")
    a = ap.parse_args(argv)
    if a.check:
        g = json.loads(Path(a.check).read_text(encoding="utf-8"))
        p = (g.get("style") or {}).get("ponytail")
        probs = ["no style.ponytail block"] if not p else validate(p)
        print(json.dumps({"ok": not probs, "problems": probs}))
        raise SystemExit(1 if probs else 0)
    if not a.take:
        ap.error("a take name (or --check FILE) is required")
    from . import hair_groom
    from .take import Take
    take = Take.load(a.take)
    if hair_style_of(take) != "ponytail":
        raise SystemExit(f"{a.take}: [dancers.follow] hair_style is not 'ponytail'")
    if a.from_take:
        hair_groom.seed_from(take, Take.load(a.from_take), do_export=not a.no_export)
    else:
        hair_groom.restyle(take, do_export=not a.no_export)


if __name__ == "__main__":
    main()
