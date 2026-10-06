"""Per-frame cameras for unsynchronised, handheld, zooming phones: static-room SfM + per-frame localisation.

    python -m dancecap cameras larissa_kadu [--cams ..] [--stages map,loc,world,qa]

  features.py  ALIKED + LightGlue with people/phone/caption masks pre-filled before extraction
  mapping.py   keyframes + window subset -> COLMAP database -> incremental mapping (one camera per image)
  localize.py  every window frame: 2D-2D to registered neighbours -> 2D-3D -> P4Pf / P3P LO-RANSAC + refine
  smooth.py    robust local-linear smoothing of pose and log-focal
  attach.py    cameras outside the main model: own sub-model placed by the building vertical + the dancers
  vertical.py  building verticals: vertical vanishing points (LSD, people masked) -> gravity; horizontal edges
  world.py     world_from_sfm: gravity (building vertical), floor, metric scale from the dancers, origin
  report.py    report.json gates (G2a), layout / focal plots, reprojection overlays
  pipeline.py  orchestration (outputs per CONTRACTS.md: cameras/sfm, cameras/<cam>.npz,
               cameras/world_from_sfm.json, cameras/report.json, cameras/qa/)
"""
