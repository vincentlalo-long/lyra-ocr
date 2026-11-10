from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class CorrectionInfo:
    method: str = "none"  # none|consensus|dictionary|human
    changed: bool = False
    confidence: float = 0.0
    diff: str = ""


@dataclass
class LyricCue:
    start_ms: int
    end_ms: int
    text: str
    raw_text: str = ""
    source_track_ids: list[int] = field(default_factory=list)
    source_segment_ids: list[int] = field(default_factory=list)
    confidence: float = 0.0
    decision: str = "accept"
    review_required: bool = False
    bbox: list[int] = field(default_factory=list)
    x_center: float = 0.0
    y_center: float = 0.0
    correction: CorrectionInfo = field(default_factory=CorrectionInfo)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LrcMetadata:
    title: str = ""
    artist: str = ""
    album: str = ""
    by: str = "Lyra OCR"
    offset_ms: int = 0


@dataclass
class CorrectionConfig:
    method: str = "dictionary"  # none|consensus|dictionary
    max_edit_ratio: float = 0.20
    min_token_len: int = 3
    min_vocab_freq: int = 2
