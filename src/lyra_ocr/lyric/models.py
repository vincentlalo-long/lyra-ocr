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
    qa_flags: list[str] = field(default_factory=list)
    ocr_confidence: float | None = None
    text_quality: float | None = None
    layer_id: int | None = None
    lang: str = "und"

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
    # Conservative dictionary/NLP word repair (method == "dictionary").
    enable_nlp: bool = True
    nlp_min_token_len: int = 4
    nlp_max_edit: int = 1
    nlp_split_penalty: float = 4.0
    # Text-quality gates applied by select_cues().
    min_ocr_confidence: float = 0.55
    review_ocr_confidence: float = 0.75
    min_text_quality: float = 0.55
    min_text_quality_hard: float = 0.20
    min_text_quality_multiword: float = 0.50
    min_gate_words: int = 3
    max_oov_ratio: float = 0.40


@dataclass
class ReferenceConfig:
    replace_similarity: float = 0.85
    review_similarity: float = 0.70
    caption_replace_similarity: float = 0.90
    min_token_coverage: float = 0.70
    min_length_ratio: float = 0.60
    max_length_ratio: float = 1.80
    enable_gap_fill: bool = True
    enable_caption_gap_fill: bool = True
    rescue_non_lyric: bool = True
    allow_edge_gap_fill: bool = False
    split_overlapping_reference_cues: bool = True

