"""Dataset management: golden-compatible manifests, video sources, job records."""

from lyra_ocr.dataset.models import JobRecord, VideoSource
from lyra_ocr.dataset.registry import (
    create_dataset,
    derive_video_id,
    freeze_dataset,
    get_dataset,
    list_datasets,
    set_split,
    sync_entry,
)

__all__ = [
    "JobRecord",
    "VideoSource",
    "create_dataset",
    "derive_video_id",
    "freeze_dataset",
    "get_dataset",
    "list_datasets",
    "set_split",
    "sync_entry",
]
