"""dancers_geoveto.decide: veto only stretches explained by NEITHER dancer (wrong person), not partner confusion."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dancecap import dancers_geoveto as gv  # noqa: E402


def test_decide_vetoes_wrong_person_only():
    G = 300
    t = np.arange(G) / 30
    rng = np.random.default_rng(0)
    own = rng.uniform(5, 15, G)
    other = rng.uniform(60, 120, G)
    own[50:90] = 350          # mask on a spectator: far from both dancers
    other[50:90] = 330
    own[150:170] = 90         # embrace: the role's detection is the partner's (explained by the other dancer)
    other[150:170] = 8
    own[200] = 400            # single-instant spike: not a run
    other[200] = 400
    costs = {"07": {"t": t, "native": np.arange(G), "lead": own, "follow": own.copy(), "lead_other": other,
                    "follow_other": other.copy()}}
    dec = gv.decide(costs)
    v = dec["07"]["lead"]["veto_grid"]
    assert v[50:90].all() and v[50 - gv.DILATE] and not v[50 - gv.DILATE - 1]
    assert not v[150:170].any() and not v[200]
    assert len(dec["07"]["lead"]["runs"]) == 1 and dec["07"]["lead"]["isolated_spikes"] == 1
