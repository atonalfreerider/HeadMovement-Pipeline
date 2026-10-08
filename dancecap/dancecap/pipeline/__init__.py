"""dancecap.pipeline: the end-to-end capture pipeline (multi-, two- and single-view) and its monitor backend.

    python -m dancecap.pipeline monitor --launch           open the Pipeline Monitor (starts the server if needed)
    python -m dancecap.pipeline stages [--variant multi]    the stage registry / DAG in order
    python -m dancecap.pipeline status <take>               every stage's status from the files on disk
    python -m dancecap.pipeline plan <take> --target X      time / pod cost / budget of a run (no side effects)
    python -m dancecap.pipeline run <take> --target X       run the plan's local stages in DAG order (pods only
                                                            through a plan approved in the monitor)
    python -m dancecap.pipeline create --spec new.json      register a new capture (videos + soundtrack + metadata)
    python -m dancecap.pipeline probe <take>                ffprobe facts + camera-type suggestion per video
    python -m dancecap.pipeline triggers <take>             classifier intervals -> viewer triggers

Modules: registry (stages + DAG per variant), status (done/stale/... from files), estimate (workload, p50/p90,
budget), plan (plans + approval gates), orchestrator (runs), runs + runwrap (process registry), intake (new takes),
camtypes (camera types adapter), takes (tolerant TOML reader), runpod_view (pods, sessions, ledger), triggers,
monitor (Flask/waitress server, launcher). Docs: docs/PIPELINE.md, docs/PIPELINE_MONITOR.md, docs/PIPELINE_PROGRESS.md.
"""
