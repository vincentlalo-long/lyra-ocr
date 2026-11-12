"""Audio processing module: extraction, VAD, and forced alignment."""

from __future__ import annotations

from lyra_ocr.audio.extract import extract_audio_16k, separate_vocals_demucs
from lyra_ocr.audio.forced_align import align_cues_with_audio
from lyra_ocr.audio.vad import compute_vocal_activity_profile, vocal_active_ratio

__all__ = [
    "extract_audio_16k",
    "separate_vocals_demucs",
    "compute_vocal_activity_profile",
    "vocal_active_ratio",
    "align_cues_with_audio",
]
