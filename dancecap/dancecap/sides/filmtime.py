"""Dance time -> film time of a take's current film direction (work/<take>/film/direction.json, its segments with dance / film / speed).

The 1.0x segments map dance seconds to film seconds; segments at another speed (the 0.5x replays) are listed as extra film spans of the same
dance time. Without a direction file, film time = dance time. (zouk_contacts.film_time_of reads an older direction file of the recap.)
"""
from __future__ import annotations

import json
from pathlib import Path


class FilmMap:
    def __init__(self, segments: list | None = None, source: str | None = None):
        self.segs = []          # (dance0, dance1, film0, speed, id)
        for s in segments or []:
            d, f = s.get("dance"), s.get("film")
            if not d or not f:
                continue
            self.segs.append((float(d[0]), float(d[1]), float(f[0]), float(s.get("speed", 1.0) or 1.0), str(s.get("id", ""))))
        self.source = source

    @classmethod
    def for_work(cls, work: Path) -> "FilmMap":
        p = Path(work) / "film" / "direction.json"
        if p.exists():
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                return cls(d.get("segments", []), str(p))
            except (OSError, ValueError):
                pass
        return cls([], None)

    def main(self, t: float) -> float | None:
        """film seconds of dance time t in the 1.0x segments (identity without a direction; None outside every segment)"""
        if not self.segs:
            return float(t)
        best = None
        for a, b, f0, sp, _ in self.segs:
            if abs(sp - 1.0) < 1e-6 and a - 1e-6 <= t <= b + 1e-6:
                best = f0 + (t - a)
                break
        if best is None:
            # between segments (a cut): the nearest 1.0x segment start/end, shifted
            ones = [(a, b, f0) for a, b, f0, sp, _ in self.segs if abs(sp - 1.0) < 1e-6]
            if not ones:
                return None
            a, b, f0 = min(ones, key=lambda x: min(abs(t - x[0]), abs(t - x[1])))
            best = f0 + (t - a)
        return round(best, 2)

    def extra(self, t: float) -> list[tuple[float, float, str]]:
        """[(film seconds, speed, segment id)] of the other-speed segments (replays) that show dance time t"""
        out = []
        for a, b, f0, sp, sid in self.segs:
            if abs(sp - 1.0) >= 1e-6 and a - 1e-6 <= t <= b + 1e-6:
                out.append((round(f0 + (t - a) / sp, 2), sp, sid))
        return out


def mmss(t: float | None) -> str:
    if t is None:
        return "-"
    t = float(t)
    return f"{int(t // 60)}:{t % 60:04.1f}"
