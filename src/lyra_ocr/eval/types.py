"""Data structures and type definitions for Phase 1 Evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class TrackPrediction:
    video_id: str
    track_id: str
    model_label: str
    model_confidence: float
    model_decision: str
    start_ms: int
    end_ms: int
    duration_ms: int
    dominant_text: str
    bbox: list[int] = field(default_factory=list)


@dataclass
class TrackLabel:
    video_id: str
    track_id: str
    human_label: str
    binary: str = "non_lyric"
    tracking_error: bool = False
    corrected_text: Optional[str] = None
    notes: str = ""
    annotator: str = "human"
    reviewed_at: str = ""
    artifact_hash: Optional[str] = None
    config_hash: Optional[str] = None


@dataclass
class EvaluationItem:
    pred: TrackPrediction
    label: Optional[TrackLabel] = None
    is_missing: bool = False

    @property
    def has_label(self) -> bool:
        return self.label is not None and bool(self.label.human_label)

    @property
    def is_tracking_error(self) -> bool:
        return self.label is not None and self.label.tracking_error


@dataclass
class PerClassMetrics:
    precision: float
    recall: float
    f1: float
    support: int


@dataclass
class BinaryLyricMetrics:
    precision: float
    recall: float
    f1: float
    support_lyric: int
    support_non_lyric: int


@dataclass
class LeakageMetrics:
    count_rate: float
    duration_rate: float
    channel_count: int
    leaked_tracks_count: int


@dataclass
class FalseDropMetrics:
    rate: float
    count: int
    total_golden_lyrics: int


@dataclass
class EvaluationReport:
    total_tracks: int
    evaluated_tracks: int
    unlabeled_tracks: int
    tracking_error_tracks: int
    coverage_rate: float
    unknown_rate: float
    macro_f1: float
    micro_f1: float
    binary_lyric: BinaryLyricMetrics
    leakage: LeakageMetrics
    false_drop: FalseDropMetrics
    per_class: dict[str, PerClassMetrics]
    confusion_matrix: dict[str, dict[str, int]]
    per_video: dict[str, Any]
    gate_passed: bool
    gate_reasons: list[str]
    gate_thresholds: dict[str, float] = field(default_factory=dict)
    binary_lyric_ci: dict[str, list[float]] = field(default_factory=dict)
    tracking_miss_tracks: int = 0
    tracking_miss_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
