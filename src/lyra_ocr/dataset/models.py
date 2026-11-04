"""Data models for dataset building sources and pipeline jobs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class VideoSource:
    """Normalized description of one video to ingest."""

    kind: str  # "youtube" | "local"
    url: str = ""
    path: str = ""
    yt_id: str = ""
    title: str = ""
    channel: str = ""
    duration: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "VideoSource":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


JOB_STATES = ("queued", "running", "done", "failed", "cancelled", "skipped")
TERMINAL_STATES = ("done", "failed", "cancelled", "skipped")


@dataclass
class JobRecord:
    """One queued dataset item with progress and failure bookkeeping."""

    job_id: str
    dataset: str
    video_id: str
    source: dict[str, Any]
    state: str = "queued"
    stage: str = ""
    percent: float = 0.0
    error: str = ""
    failed_stage: str = ""
    attempts: int = 0
    created_at: int = 0
    updated_at: int = 0
    artifact_dir: str = ""
    video_path: str = ""
    options: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "JobRecord":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES
