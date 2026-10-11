"""Controls of the rhythm score (agent S7, 2026-10-11): does the metric read high where the dancing is trusted to be in time with the music, and ~0 where it cannot be?

    python -m dancecap.rhythm_controls [--out DIR] [--calibrate] [--only NAME ...]          # also: python -m dancecap.rhythm controls

POSITIVE controls: the demo (a professional couple on a studio song with a careful audio alignment; the whole take, its first half = the TRAINING half the reference levels come from,
its second half = the held-out half) and the 38 s reference span (a second pose fit of the same performance); the heat itself is out-of-sample.
NEGATIVE controls: the same motion against metronomes at the wrong tempo (0.78 .. 1.30 x), against ANOTHER song's grid (the other take's grid, shifted onto this take's span), against a grid
with a randomly wandering tempo; plus the tests that must NOT change the score (a constant offset of the grid: the score is phase-free by design; time reversal: a sinusoidal lock is
invariant to it).
NO-DANCE control: the recap (two teachers talking: they barely step) against a plausible song grid: low, and with low confidence / insufficient lock evidence.
Everything is printed and written to work/<take>/rhythm/controls.json + controls.png (the heat's folder).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import rhythm as rh
from . import rhythm_entrain as ent

TAKES_FILE = Path(__file__).resolve().parents[1] / "work" / "rhythm_controls_takes.json"      # git-ignored: which takes play which role (names stay out of the tracked code)
ROLES_DOC = '{"demo": "<take>", "demo_audio": "song/studio_reference.wav", "ref38": "<take>", "heat": "<take>", "recap": "<take or null>"}'


def _takes() -> dict:
    """the takes of the controls from work/rhythm_controls_takes.json: demo (a professional couple with a trusted audio alignment; `demo_audio` = its soundtrack inside the take's work folder),
    ref38 (a short reference span of the same performance), heat (the take the score is for), recap (a take of people who talk and barely step, or null)"""
    if not TAKES_FILE.exists():
        raise SystemExit(f"error: {TAKES_FILE} missing; it names the takes of the controls: {ROLES_DOC}")
    return json.loads(TAKES_FILE.read_text(encoding="utf-8"))


def _role(name: str):
    t = _takes()
    if name == "demo":
        return (t["demo"], t.get("demo_audio"))
    if t.get(name) is None:
        return None
    return (t[name], None)


def _work(take: str) -> Path:
    from .take import Take
    return Take.load(take).work


def load(take: str, audio: str | None, cfg: rh.Config) -> rh.Inputs:
    w = _work(take)
    return rh.load_inputs(take, audio=(w / audio) if audio else None, cfg=cfg, work=w)


def stretch_grid(g: rh.Grid, s: float) -> rh.Grid:
    """the same grid played at tempo x s about its middle (a metronome at the wrong tempo)"""
    mid = 0.5 * (g.t[0] + g.t[-1])
    f = lambda x: mid + (np.asarray(x, float) - mid) / s                               # noqa: E731
    return rh.Grid(f(g.t), g.slot, g.type, g.measure, g.conf, g.song, [(float(f(a)) if abs(a) < 1e8 else a, float(f(b)) if abs(b) < 1e8 else b, k) for a, b, k in g.no_beat], g.seam_note)


def wandering_grid(g: rh.Grid, seed: int, sd: float = 0.10, corr_s: float = 25.0) -> rh.Grid:
    """the grid's slots re-timed by a smoothly wandering tempo (relative sd `sd`, correlation time corr_s): a metronome that drifts"""
    rng = np.random.default_rng(seed)
    n = len(g.t)
    p8 = g.p8
    k = max(3, int(corr_s / p8))
    w = rng.normal(0.0, 1.0, n + 4 * k)
    w = np.convolve(w, np.ones(k) / math.sqrt(k), "same")[2 * k:2 * k + n]
    w = w / max(float(np.std(w)), 1e-9) * sd
    dt = np.r_[0.0, np.diff(g.t)] * (1.0 + w)
    t = g.t[0] + np.cumsum(dt)
    t = t - (t.mean() - g.t.mean())
    return rh.Grid(t, g.slot, g.type, g.measure, g.conf, g.song, [(a, b, k_) for a, b, k_ in g.no_beat], g.seam_note)


def other_grid(g_other: rh.Grid, inp: rh.Inputs) -> rh.Grid:
    """another take's grid (its tempo, its song) moved onto this take's span: the first beat-covered second of the other grid lands at this grid's first beat"""
    shift = float(inp.grid.t[0] - g_other.t[0])
    g = g_other.shifted(shift)
    keep = g.t < inp.grid.t[-1]
    g2 = rh.Grid(g.t[keep], g.slot[keep], g.type[keep], g.measure[keep], g.conf[keep], [s for s, kp in zip(g.song, keep) if kp] if g.song else [], g.no_beat, g.seam_note)
    return g2


def reverse_inputs(inp: rh.Inputs) -> rh.Inputs:
    """the motion played backwards against the same grid"""
    b = inp.body
    T = float(b.t[0] + b.t[-1])
    body = ent.Body(b.t, {r: {k: v[::-1].copy() for k, v in b.sig[r].items()} for r in rh.ROLES}, b.source)
    steps = {r: sorted([{**s, "t": T - s["t"]} for s in inp.steps[r]], key=lambda x: x["t"]) for r in rh.ROLES}
    return rh.Inputs(inp.grid, steps, inp.music, inp.motion, inp.t0, inp.t1, dict(inp.meta), body=body)


def with_grid(inp: rh.Inputs, grid: rh.Grid, music_shift_s: float = 0.0) -> rh.Inputs:
    """the same motion against another grid; with `music_shift_s` the soundtrack's accents are moved too (a wrong SONG: neither its beats nor its accents belong to the motion)"""
    m = inp.music
    if music_shift_s and m is not None:
        m = rh.Music(m.acc_t + music_shift_s, m.acc_s, [(a + music_shift_s, b + music_shift_s) for a, b in m.breaks], m.level_t + music_shift_s, m.level_db, m.source)
    return rh.Inputs(grid, inp.steps, m, inp.motion, inp.t0, inp.t1, dict(inp.meta), {}, inp.body)


def crop_grid(g: rh.Grid, a: float, b: float) -> rh.Grid:
    """the grid restricted to [a, b] (the averages restart at a): for the halves of a take"""
    m = (g.t >= a) & (g.t <= b)
    p8 = g.p8
    nb = [(x, y, k) for x, y, k in g.no_beat if y > a and x < b and abs(x) < 1e8 and abs(y) < 1e8]
    nb += [(-1e9, float(g.t[m][0] - p8 / 2), "none"), (float(g.t[m][-1] + p8 / 2), 1e9, "none")]
    return rh.Grid(g.t[m], g.slot[m], g.type[m], g.measure[m], g.conf[m], [s for s, k in zip(g.song, m) if k] if g.song else [], rh._merge_intervals(nb), g.seam_note)


def measure(name: str, kind: str, inp: rh.Inputs, cfg: rh.Config, note: str = "") -> dict:
    t0 = time.time()
    A = rh.run_series(inp, cfg)
    f = rh.aggregate(A, cfg)
    conf = rh.confidence_summary(A, inp.grid, inp, cfg, f)
    lo, hi, sd = rh.noise_interval(A, cfg, f)
    row = {"name": name, "kind": kind, "score": rh._r(f["score"], 1), "components": {k: rh._r(f[k]) for k in rh.COMPONENTS}, "interval_90": [rh._r(lo, 1), rh._r(hi, 1)],
           "confidence": conf["label"], "confidence_value": conf["value"], "lock_evidence_z": conf["lock_evidence_z"], "insufficient_evidence": conf["insufficient_evidence"],
           "n_samples": f["n_samples"], "seconds": round(time.time() - t0, 1), "note": note}
    print(f"  {name:50s} {kind:9s} score {row['score']!s:>5s}  " + " ".join(f"{k}={row['components'][k]}" for k in rh.COMPONENTS) +
          f"  conf {conf['label']} z={conf['lock_evidence_z']}{'  INSUFFICIENT' if conf['insufficient_evidence'] else ''}", flush=True)
    return row


def calibrate(inp: rh.Inputs, cfg: rh.Config) -> dict:
    """The reference levels: on the demo's FIRST half (cropped: the averages restart) the 90th percentile, past the first tau_long seconds and pooled over the two roles, of the statistics
    the skills are made of (long-window body lock at the chosen level, the same over the recent window, foot-touchdown concentration, lead-follow coherence, accents excess)."""
    g = inp.grid
    inp1 = with_grid(inp, crop_grid(g, g.t[0], float(np.median(g.t))))
    A = rh.run_series(inp1, cfg)
    S, K = A.S, A.K
    m = A.valid & (A.elapsed > cfg.ent.tau_long_s)
    p90 = lambda x: float(np.percentile(x[np.isfinite(x)], 90))                                 # noqa: E731
    pooled = lambda arrs: np.concatenate([a[m] for a in arrs])                                  # noqa: E731
    idx = np.arange(len(A.t))
    out = {"first_half_ends_at_s": round(float(A.t[A.ends[-1]]), 1),
           "ref_body": p90(pooled([K.raw[("body", r)] for r in rh.ROLES])),
           "ref_recent_check": p90(pooled([S.ema_body[r][S.best_body[r], idx] for r in rh.ROLES])),
           "ref_steps": p90(pooled([K.raw[("steps", r)] for r in rh.ROLES])),
           "ref_couple": p90(S.couple[m])}
    raw, _ = rh.accent_series(A.events, A.t, A.valid, cfg)
    out["ref_accent"] = p90(raw[m])
    out["baseline_body_mean"] = float(np.nanmean(np.concatenate([K.baseline[("body", r)][0][m] for r in rh.ROLES])))
    return {k: round(v, 4) for k, v in out.items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.rhythm_controls", description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", help="output folder (default work/<heat>/rhythm)")
    ap.add_argument("--calibrate", action="store_true", help="print the reference levels from the demo's training half and stop")
    ap.add_argument("--only", nargs="*", help="run only the controls whose name contains one of these words")
    a = ap.parse_args(argv)
    cfg = rh.Config()
    demo = load(*_role("demo"), cfg)
    if a.calibrate:
        print(json.dumps(calibrate(demo, cfg), indent=1))
        return 0
    heat = load(*_role("heat"), rh.Config())
    ref38 = load(*_role("ref38"), rh.Config())
    rows = []

    def want(name):
        return not a.only or any(w in name for w in a.only)

    def add(name, kind, inp, note="", cfg_=None):
        if want(name):
            rows.append(measure(name, kind, inp, cfg_ or rh.Config(), note))

    print("positive controls")
    add("demo (whole take)", "positive", demo)
    for nm, inp in (("demo", demo), ("heat", heat)):
        g = inp.grid
        mid = float(np.median(g.t))
        add(f"{nm}, first half (averages restarted)", "positive" if nm == "demo" else "heat", with_grid(inp, crop_grid(g, g.t[0], mid)))
        add(f"{nm}, second half (averages restarted)", "positive" if nm == "demo" else "heat", with_grid(inp, crop_grid(g, mid + 1.0, g.t[-1])))
    add("38 s reference span (second pose fit)", "positive", ref38, note="short take: the averages fill slowly")
    add("the heat (out-of-sample)", "heat", heat)
    for nm, inp in (("demo", demo), ("heat", heat)):
        print(f"negative controls on the {nm}")
        for s in (0.78, 0.82, 1.15, 1.22, 1.30):
            add(f"{nm}: grid at {s:.2f} x the tempo", "negative", with_grid(inp, stretch_grid(inp.grid, s), 3.3))
        other = heat if nm == "demo" else demo
        add(f"{nm}: the other take's song grid", "negative", with_grid(inp, other_grid(other.grid, inp), 3.3), note="a different song's tempo and beat positions")
        for sd in range(2):
            add(f"{nm}: wandering-tempo grid (seed {sd})", "negative", with_grid(inp, wandering_grid(inp.grid, sd), 3.3))
        add(f"{nm}: grid shifted by +0.31 s (must not change the score)", "invariant", with_grid(inp, inp.grid.shifted(0.31)), note="the score is phase-free by design")
        add(f"{nm}: motion played backwards (must stay locked)", "invariant", reverse_inputs(inp), note="a sinusoidal lock is invariant to reversing time")
    print("no-dance control")
    if want("recap") and _role("recap") is not None:
        try:
            recap = load(*_role("recap"), rh.Config())
        except (ValueError, KeyError, TypeError) as e:
            rc = _recap_with_borrowed_grid(demo, _role("recap")[0])
            print(f"  (the recap has no beat grid: {type(e).__name__}; the pipeline refuses it. Scoring its motion against the demo's tempo grid instead)")
            recap = rc
        add("recap: teachers talking, against a 76 bpm grid", "no-dance", recap, note="no dance: the lock should be absent")
    out_dir = Path(a.out) if a.out else _work(_role("heat")[0]) / "rhythm"
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "reference_levels": {**{k: v for k, v in vars(rh.Config().ent).items() if k.startswith("ref_")}, "ref_accent": rh.Config().ref_accent},
           "rows": rows}
    (out_dir / "controls.json").write_text(json.dumps(doc, indent=1, allow_nan=False), encoding="utf-8")
    try:
        _plot(rows, out_dir / "controls.png")
    except Exception as e:                                                                  # noqa: BLE001
        print("plot failed:", e)
    print("[controls] wrote", out_dir / "controls.json")
    return 0


def _recap_with_borrowed_grid(demo: rh.Inputs, take: str) -> rh.Inputs:
    w = _work(take)
    d = json.loads((w / "timing" / "timing.json").read_text(encoding="utf-8"))
    steps = {r: [{"t": float(f["t"]), "foot": f.get("foot", ""), "sigma": float(f.get("sigma") or 0.02)} for f in d.get("footsteps", {}).get(r, [])] for r in rh.ROLES}
    body = ent.load_body(w)
    p8 = demo.grid.p8
    t = 1.0 + np.arange(int((body.t[-1] - 2.0) / p8)) * p8
    rows = [[x, (1, 3, 3, 2, 1, 3, 2, 3)[i % 8], i // 8, i % 8] for i, x in enumerate(t)]
    grid = rh.build_grid(rows, no_beat_none=[], conf=np.full(len(t), 0.8), default_conf=0.8)
    return rh.Inputs(grid, steps, rh.music_from_grid(grid), rh.load_motion(w / "timing" / "kinematics.npz"), float(body.t[0]), float(body.t[-1]),
                     {"take": take, "av_pinned": False}, {}, body)


def _plot(rows: list, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"positive": "#2a9d8f", "heat": "#e76f51", "negative": "#8d99ae", "invariant": "#bcb8b1", "no-dance": "#6d597a"}
    fig, ax = plt.subplots(figsize=(11, 0.34 * len(rows) + 1.5))
    y = np.arange(len(rows))[::-1]
    ax.barh(y, [r["score"] or 0 for r in rows], color=[col.get(r["kind"], "#999") for r in rows])
    for yy, r in zip(y, rows):
        lo, hi = r["interval_90"]
        if lo is not None and hi is not None:
            ax.plot([lo, hi], [yy, yy], color="k", lw=1.0)
    ax.set_yticks(y, [r["name"] for r in rows], fontsize=8)
    ax.set_xlim(0, 100)
    ax.set_xlabel("final score (bar), 90% sampling interval (line)")
    ax.set_title("rhythm score controls: positive (green), heat (orange), wrong-tempo / other-song negatives (grey), invariance checks, no-dance")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
