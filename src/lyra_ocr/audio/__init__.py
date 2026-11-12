"""Audio processing module: extraction, VAD, and forced alignment."""

from __future__ import annotations

from lyra_ocr.audio.extract import extract_audio_16k, separate_vocals_demucs

__all__ = [
    "extract_audio_16k",
    "separate_vocals_demucs",
]
