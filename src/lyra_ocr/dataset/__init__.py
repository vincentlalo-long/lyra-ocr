"""Dataset management: media ingestion, runs, ground truth, and job records."""

from lyra_ocr.dataset.ground_truth import (
    GroundTruthSegment,
    load_segments,
    promote_run,
    write_segments,
)
from lyra_ocr.dataset.models import JobRecord, RunRecord, VideoSource
from lyra_ocr.dataset.registry import (
    create_dataset,
    create_run,
    derive_video_id,
    freeze_dataset,
    get_dataset,
    get_run,
    get_video,
    list_datasets,
    list_runs,
    run_dir,
    runs_root,
    set_split,
    sync_entry,
    sync_run_entry,
    upsert_video,
)

__all__ = [
    "GroundTruthSegment",
    "JobRecord",
    "RunRecord",
    "VideoSource",
    "create_dataset",
    "create_run",
    "derive_video_id",
    "freeze_dataset",
    "get_dataset",
    "get_run",
    "get_video",
    "list_datasets",
    "list_runs",
    "load_segments",
    "promote_run",
    "run_dir",
    "runs_root",
    "set_split",
    "sync_entry",
    "sync_run_entry",
    "upsert_video",
    "write_segments",
]
