from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class CorrectionInfo:
    method: str = "none"  # none|consensus|dictionary|human
    changed: bool = False
    confidence: float = 0.0
    diff: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> CorrectionInfo:
        if not isinstance(d, dict):
            return cls()
        return cls(
            method=str(d.get("method", "none")),
            changed=bool(d.get("changed", False)),
            confidence=float(d.get("confidence", 0.0)),
            diff=str(d.get("diff", "")),
        )


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
    words: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LyricCue:
        corr_raw = d.get("correction")
        if isinstance(corr_raw, dict):
            correction = CorrectionInfo.from_dict(corr_raw)
        elif isinstance(corr_raw, CorrectionInfo):
            correction = corr_raw
        else:
            correction = CorrectionInfo()

        raw_text = d.get("raw_text")
        text = str(d.get("text", "")).strip()
        if raw_text is None:
            raw_text = str(d.get("text", ""))
        else:
            raw_text = str(raw_text)

        tids = d.get("source_track_ids")
        if tids is None:
            tids = d.get("track_ids", [])
        sids = d.get("source_segment_ids")
        if sids is None:
            sids = d.get("segment_ids", [])

        ocr_conf = d.get("ocr_confidence")
        if ocr_conf is not None:
            ocr_conf = float(ocr_conf)

        t_qual = d.get("text_quality")
        if t_qual is not None:
            t_qual = float(t_qual)

        return cls(
            start_ms=int(d.get("start_ms", 0)),
            end_ms=int(d.get("end_ms", 0)),
            text=text,
            raw_text=raw_text,
            source_track_ids=list(tids or []),
            source_segment_ids=list(sids or []),
            confidence=float(d.get("confidence", 0.0)),
            decision=str(d.get("decision", "accept")),
            review_required=bool(d.get("review_required", False)),
            bbox=list(d.get("bbox") or []),
            x_center=float(d.get("x_center", 0.0)),
            y_center=float(d.get("y_center", 0.0)),
            correction=correction,
            qa_flags=list(d.get("qa_flags") or []),
            ocr_confidence=ocr_conf,
            text_quality=t_qual,
            layer_id=d.get("layer_id"),
            lang=str(d.get("lang", "und")),
            words=list(d.get("words") or []),
        )


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
    repair_romaji: bool = False


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
    enable_audio_align: bool = True
    enable_asr: bool = True
    enhanced_lrc: bool = False
    repair_romaji: bool = False

