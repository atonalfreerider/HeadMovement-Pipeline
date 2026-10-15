"""mono4d: one phone video -> a 4D reconstruction (provider API) -> virtual cameras -> MAMMA (built 2026-10-15).

The route the owner asked for: "feed a monocular video to the api, get a 4d gaussian, and create novel camera angles so
we can run MAMMA". MAMMA needs several calibrated, synchronised views; a single phone gives one. A 4D provider
(World Labs Atlas, Gracia, a self-hosted model) reconstructs the performance, we place virtual cameras around the
couple in OUR world frame and render (or have the provider render) every view, and the rendered views enter the
standard pipeline as a "virtual take" whose cameras are exact by construction.

What the views are NOT: independent measurements. Everything a generated view shows that the phone did not see is the
provider's prior (Atlas: "imagines a plausible way to fill in the gaps"). MAMMA triangulates whatever it is given, so the
route can at best match the provider's own 3D understanding of the dancers; whether that beats our one-camera fit
(dancecap.jl_mono + the MAMMA_MONO path) is an empirical question answered by the hold-out protocol in holdout.py on a
take we already captured with 7-8 phones. Findings and the decision record: docs/MONO4D_NOVEL_VIEWS.md.

Modules (all local, CPU, numpy unless stated):
    rig.py           virtual camera specs: a ring around the couple, the real poses of held-out phones; Sim3 alignment
                     of a provider's scene frame to the world; camera trajectories in OpenCV and OpenGL conventions
    providers.py     request/result contract and the providers: manual drop (any vendor), World Labs World API
                     (public, static worlds only; Atlas plugs in when World Labs opens it), Gracia (.mint/.ply drop)
    render_gracia.py standalone renderer of a Gracia .mint/.ply asset at our cameras (graciasdk, Python >= 3.12, Vulkan)
    virtual_take.py  rendered views -> work/<virtual take>/ in the CONTRACTS.md layout, so detect / dancers / mamma run
                     on it unchanged; the manifest marks every view synthetic
    holdout.py       evaluation against a multi-camera reference fit: MPJPE, PA-MPJPE, pelvis, foot height, 2D PCK
                     and mask IoU of a generated view against the real phone at the same pose
    judge.py         TypeSafe (Jev) visual judgments of generated frames: duplicated dancers, broken anatomy, render
                     defects, same pose as the real frame (images are a "dreamer-ready" TypeSafe feature)
    vendor_watch.py  TypeSafe (Jev) judgments of vendor pages: does a provider now state monocular input, 4D output,
                     self-serve API, camera control, export? -> a dated readiness table and its changes
    gate.py          the decision: may a run's generated views enter MAMMA? numeric checks + judgments -> quality_gate.json

    python -m dancecap.mono4d --help
"""
