"""Placeholder for the Atlas HTTP API.

World Labs has not published Atlas API docs (as of 2026-09). When access lands,
implement submit/fetch here against the real endpoints; the rest of the pipeline
only depends on AtlasJob/AtlasResult. Keep credentials in the WORLDLABS_API_KEY
environment variable, never in session.toml.
"""
from __future__ import annotations

import os

from .base import AtlasJob, AtlasProvider, AtlasResult


class HttpAtlasProvider(AtlasProvider):
    def __init__(self, work_dir, cfg):
        super().__init__(work_dir, cfg)
        self.base_url = cfg.get("base_url", "")
        self.api_key = os.environ.get("WORLDLABS_API_KEY", "")

    def submit(self, job: AtlasJob) -> str:
        # TODO(atlas-api): upload job.source_videos, send job.virtual_cameras as
        # camera trajectories (cameras.to_trajectory_json gives both conventions),
        # request RGB video per view (+ environment splat), return the job id.
        raise NotImplementedError("Atlas API not published yet - use [atlas] provider = \"manual\"")

    def fetch(self, job: AtlasJob, job_id: str) -> AtlasResult | None:
        # TODO(atlas-api): poll job status; download each view to
        # work/atlas/renders/<camera>.mp4 and reuse ManualAtlasProvider.fetch for validation.
        raise NotImplementedError


def make_provider(name: str, work_dir, cfg) -> "AtlasProvider":
    from .manual import ManualAtlasProvider
    providers = {"manual": ManualAtlasProvider, "http": HttpAtlasProvider}
    if name not in providers:
        raise ValueError(f"unknown atlas provider {name!r}; choose from {sorted(providers)}")
    return providers[name](work_dir, cfg)
