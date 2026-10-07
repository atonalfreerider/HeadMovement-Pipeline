"""Narration clocks: recording time -> video time -> label time, reaction-lag compensation and beat snapping.

The browser logs the player against the recording clock while the user narrates (every ~100 ms and on every player
event): entries [{a, v, cam, rate, paused, ev}]
  a       seconds since the first recorded audio sample (the recording's own clock: AudioWorklet frames, or
          performance.now() since MediaRecorder start in the fallback recorder)
  v       video.currentTime of camera `cam` (presentation seconds of that file; never frame / fps - the phone videos
          are variable frame rate)
  rate    playbackRate (0.25 .. 1), paused (bool), ev: tick | start | play | pause | seeking | seeked | ratechange |
          camera | tap | stop
Label time (the labels.json time base) = to_label(cam, v): the take's reference clock (Source.to_reference, affine)
or, for a lone video, the video time itself.

Mapping a recording time `a` (TimeMap.label_time):
  between two entries whose label times agree with continuous playback (t_{i+1} ~ t_i + rate_i * slope * (a_{i+1} -
  a_i), within 0.25 s + 10 %), interpolate linearly; otherwise (a seek, a camera switch to another time) extrapolate
  from the earlier entry (paused -> held). Before the first / after the last entry: extrapolate from it.
  Words spoken while paused therefore map to the paused position; speed changes scale the video advance.
Reaction lag (TimeMap.event_time): the narrator names a move `lag` seconds (wall clock, default 0.4) after it starts,
  so the event is the label time shown at a_word - lag (= t_word - lag x rate during playback). The lag never reaches
  back across a seek (it stops at the first entry after the jump).
Snapping (snap_start): start = a user tap near the event (taps are logged with 'M'), else a proposal boundary
  (proposals.json), else the grid point of the chosen level at or just before the event (cost: distance before,
  2 x distance after; window [t - 1 unit, t + 0.2 s]), else the nearest eighth within one eighth, else unsnapped.
  Levels: measure (slot 0) | half (type-1 eighths, slots 0 and 4 - the default) | beat (even slots) | eighth | none.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

ZOUK_PATTERN = [1, 3, 3, 2, 1, 3, 2, 3]
LEVELS = ("measure", "half", "beat", "eighth", "none")
TIMING_SCORE = {"tap": 1.0, "proposal": 0.9, "grid": 0.85, "grid-shifted": 0.8, "eighth": 0.75, "none": 0.6}


def _f(x, default=None):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


@dataclass
class Entry:
    a: float
    v: float
    cam: str
    rate: float
    paused: bool
    ev: str
    t: float  # label time
    k: float  # d(label) / d(recording) while in this state


class TimeMap:
    def __init__(self, entries: Iterable[dict], to_label: Callable[[str, float], float] | None = None,
                 slope: Callable[[str], float] | None = None, tol: float = 0.25):
        to_label = to_label or (lambda cam, v: float(v))
        slope = slope or (lambda cam: 1.0)
        rows = []
        for i, e in enumerate(entries):
            a, v = _f(e.get("a")), _f(e.get("v"))
            if a is None or v is None:
                continue
            cam = str(e.get("cam") or "")
            rate = _f(e.get("rate"), 1.0) or 0.0
            paused = bool(e.get("paused"))
            t = float(to_label(cam, v))
            rows.append((a, i, Entry(a, v, cam, rate, paused, str(e.get("ev") or "tick"), t,
                                     0.0 if paused else rate * float(slope(cam)))))
        rows.sort(key=lambda r: (r[0], r[1]))
        self.e: list[Entry] = [r[2] for r in rows]
        self.tol = tol
        self.cont: list[bool] = []
        for x, y in zip(self.e, self.e[1:]):
            da = y.a - x.a
            pred = x.t + x.k * da
            self.cont.append(abs(y.t - pred) <= tol + 0.1 * da)
        self._a = np.array([x.a for x in self.e], float)

    def __len__(self) -> int:
        return len(self.e)

    @property
    def empty(self) -> bool:
        return not self.e

    def _index(self, a: float) -> int:
        """Index of the last entry with entry.a <= a (-1 before the first)."""
        return int(np.searchsorted(self._a, a, side="right")) - 1

    def label_time(self, a: float) -> float:
        if self.empty:
            return float(a)
        i = self._index(a)
        if i < 0:
            e = self.e[0]
            return e.t - e.k * (e.a - a)
        e = self.e[i]
        if i < len(self.e) - 1 and self.cont[i]:
            n = self.e[i + 1]
            if n.a > e.a:
                return e.t + (n.t - e.t) * (a - e.a) / (n.a - e.a)
            return e.t
        return e.t + e.k * (a - e.a)

    def camera_at(self, a: float) -> str | None:
        if self.empty:
            return None
        return self.e[max(0, self._index(a))].cam

    def breaks(self) -> list[float]:
        """Recording times of the first entry after each discontinuity (seek / jump)."""
        return [self.e[i + 1].a for i, c in enumerate(self.cont) if not c]

    def event_time(self, a_word: float, lag: float) -> tuple[float, float]:
        """(label time, recording time used) of an event named at a_word, lag seconds (wall clock) earlier."""
        a_e = a_word - max(0.0, lag)
        for b in self.breaks():
            if a_e < b <= a_word:
                a_e = b
        return self.label_time(a_e), a_e

    def spans(self, min_len: float = 0.0) -> list[tuple[float, float]]:
        """Label-time spans the narrator saw (continuous runs of the log), merged."""
        if self.empty:
            return []
        out = []
        lo = hi = self.e[0].t
        for i, x in enumerate(self.e[1:]):
            if self.cont[i]:
                lo, hi = min(lo, x.t), max(hi, x.t)
            else:
                out.append((lo, hi))
                lo = hi = x.t
        out.append((lo, hi))
        out = sorted(s for s in out if s[1] - s[0] >= min_len)
        merged: list[list[float]] = []
        for s in out:
            if merged and s[0] <= merged[-1][1] + 1e-6:
                merged[-1][1] = max(merged[-1][1], s[1])
            else:
                merged.append(list(s))
        return [(float(a), float(b)) for a, b in merged]

    def span_of(self, t: float, spans: list[tuple[float, float]] | None = None, slack: float = 0.05):
        for s in spans if spans is not None else self.spans():
            if s[0] - slack <= t <= s[1] + slack:
                return s
        return None

    def taps(self) -> list[float]:
        return [x.a for x in self.e if x.ev == "tap"]


# ---------------------------------------------------------------------------------------------------------------
# beat grid
# ---------------------------------------------------------------------------------------------------------------

def measure_phase(types: list[int]) -> int:
    """Index of the first downbeat: the phase whose shifted zouk pattern matches the grid's types best (0 when the
    grid carries no type information)."""
    if len(types) < 8 or len(set(types)) == 1:
        return 0
    scores = [float(np.mean([types[i] == ZOUK_PATTERN[(i - p) % 8] for i in range(len(types))])) for p in range(8)]
    return int(np.argmax(scores))


class Grid:
    """The take's zouk eighth grid (beats.json [[t, type], ...], 8 eighths per measure, label time)."""

    def __init__(self, rows: list):
        rows = [r for r in rows if _f(r[0]) is not None]
        self.t = np.array([float(r[0]) for r in rows], float)
        self.type = np.array([int(r[1]) if len(r) > 1 else 3 for r in rows], int)
        self.phase = measure_phase(list(self.type))
        self.slot = (np.arange(len(self.t)) - self.phase) % 8

    def __len__(self) -> int:
        return len(self.t)

    def mask(self, level: str) -> np.ndarray:
        if level == "measure":
            return self.slot == 0
        if level == "half":
            return self.slot % 4 == 0
        if level == "beat":
            return self.slot % 2 == 0
        return np.ones(len(self.t), bool)

    def points(self, level: str) -> np.ndarray:
        return self.t[self.mask(level)]

    def unit(self, level: str) -> float:
        p = self.points(level)
        return float(np.median(np.diff(p))) if len(p) > 1 else 1.0

    def eighth_index(self, t: float) -> int | None:
        """Index of the grid eighth containing t (None outside the grid)."""
        if not len(self.t) or t < self.t[0] - 1e-6:
            return None
        i = int(np.searchsorted(self.t, t + 1e-6, side="right")) - 1
        if i >= len(self.t) - 1 and t > self.t[-1] + self.unit("eighth"):
            return None
        return i

    def to_list(self) -> list:
        return [[round(float(t), 6), int(k), int(s)] for t, k, s in zip(self.t, self.type, self.slot)]


def snap_start(t: float, grid: Grid | None, level: str = "half", taps: Iterable[float] = (),
               boundaries: Iterable[float] = (), back: float | None = None, fwd: float = 0.2) -> tuple[float, str]:
    """(snapped start, how) for an event at label time t; how in TIMING_SCORE."""
    level = level if level in LEVELS else "half"
    unit = grid.unit(level if level != "none" else "half") if grid is not None and len(grid) > 1 else 1.0
    back = max(0.5, back if back is not None else unit)
    eighth = grid.unit("eighth") if grid is not None and len(grid) > 1 else 0.25
    lo, hi = t - back, t + fwd

    def nearest(cands) -> float | None:
        c = [x for x in cands if lo <= x <= hi]
        return min(c, key=lambda x: abs(x - t)) if c else None

    tp = nearest(taps)
    if tp is not None:
        if grid is not None and len(grid):
            j = int(np.argmin(np.abs(grid.t - tp)))
            if abs(grid.t[j] - tp) <= 0.5 * eighth:
                tp = float(grid.t[j])
        return float(tp), "tap"
    if level == "none":
        return float(t), "none"
    bd = nearest(boundaries)
    if bd is not None:
        return float(bd), "proposal"
    if grid is None or not len(grid):
        return float(t), "none"
    pts = grid.points(level)
    c = pts[(pts >= lo) & (pts <= hi)]
    if len(c):
        cost = np.where(c <= t, t - c, 2.0 * (c - t))
        return float(c[int(np.argmin(cost))]), "grid"
    j = int(np.argmin(np.abs(grid.t - t)))
    if abs(grid.t[j] - t) <= eighth:
        return float(grid.t[j]), "eighth"
    return float(t), "none"
