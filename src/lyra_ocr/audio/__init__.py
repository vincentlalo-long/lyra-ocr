"""Audio processing module: extraction, VAD, and forced alignment."""

from __future__ import annotations

from lyra_ocr.audio.asr import (
    ASRSegment,
    ASRTranscript,
    ASRWord,
    align_multiline_cues_with_asr,
    asr_to_reference_bundle,
    asr_vocal_gate_score,
    load_asr_transcript,
    save_asr_transcript,
    transcribe_audio_words,
)
from lyra_ocr.audio.extract import extract_audio_16k, separate_vocals_demucs
from lyra_ocr.audio.forced_align import align_cues_with_audio
from lyra_ocr.audio.vad import compute_vocal_activity_profile, vocal_active_ratio

__all__ = [
    "ASRWord",
    "ASRSegment",
    "ASRTranscript",
    "save_asr_transcript",
    "load_asr_transcript",
    "transcribe_audio_words",
    "asr_to_reference_bundle",
    "align_multiline_cues_with_asr",
    "asr_vocal_gate_score",
    "extract_audio_16k",
    "separate_vocals_demucs",
    "compute_vocal_activity_profile",
    "vocal_active_ratio",
    "align_cues_with_audio",
]
