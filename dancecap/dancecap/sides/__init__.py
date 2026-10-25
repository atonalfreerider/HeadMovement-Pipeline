"""sides: decide from the cameras which side each limb is on relative to the partner (and where the joints really are), then repair only
the windows where the fit breaks a decided side (DESIGN: the owner's rule of 2026-10-24, "determine which side things are on and
minimize the error of the real world positions as a rule for all dances").

    python -m dancecap.sides evidence --take <t>     # triangulated joints, per-view bits      -> work/<t>/sides/evidence/
    python -m dancecap.sides decide --take <t>       # relation HMMs, yaw track, violations    -> work/<t>/sides/relations.npz, ...
    python -m dancecap.sides report --take <t>       # film-time violation list + lanes PNG    -> work/<t>/sides/report.md
    python -m dancecap.sides job --take <t>          # the pod job (evidence + decide)         -> work/<t>/sides/jobs/sides_<t>_A.json

No take constant lives in this package: a take is a parameter, its overrides an optional [sides] table of takes/<t>.toml.
"""
