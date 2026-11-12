"""Windowed acoustic onset alignment and word-level timing generation."""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np
from scipy import signal

from lyra_ocr.lyric.models import LyricCue

logger = logging.getLogger(__name__)


def _compute_onset_envelope(
    audio: np.ndarray,
    sample_rate: int = 16000,
    hop_ms: int = 10,
) -> np.ndarray:
    """Compute 10ms frame vocal band onset envelope."""
    if len(audio) == 0:
        return np.zeros(0, dtype=np.float32)

    nyq = 0.5 * sample_rate
    low = 300.0 / nyq
    high = min(3400.0 / nyq, 0.99)
    sos = signal.butter(2, [low, high], btype="bandpass", output="sos")
    filtered = signal.sosfiltfilt(sos, audio)

    hop_samples = int(sample_rate * (hop_ms / 1000.0))
    n_frames = int(len(filtered) / hop_samples)
    if n_frames <= 0:
        return np.zeros(0, dtype=np.float32)

    energies = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        s = i * hop_samples
        e = s + hop_samples
        frame = filtered[s:e]
        energies[i] = np.sqrt(np.mean(frame ** 2)) if len(frame) > 0 else 0.0

    diff = np.diff(energies, prepend=energies[0])
    onset_env = np.maximum(0.0, diff) + 0.3 * energies
    return onset_env


def _tokenize_words(text: str) -> list[str]:
    return re.findall(r"[\w']+", text)


def align_cues_with_audio(
    cues: list[LyricCue],
    audio_16k: np.ndarray,
    *,
    sample_rate: int = 16000,
    window_pad_ms: int = 1000,
    max_drift_ms: int = 1200,
) -> list[LyricCue]:
    """Refine cue start/end timestamps via acoustic onset detection and generate word-level timings."""
    if not cues or audio_16k is None or len(audio_16k) == 0:
        return cues

    audio_len_ms = int(len(audio_16k) / sample_rate * 1000)

    for cue in cues:
        search_start_ms = max(0, cue.start_ms - 600)
        search_end_ms = min(audio_len_ms, cue.start_ms + 800)

        start_samp = int(search_start_ms * sample_rate / 1000)
        end_samp = int(search_end_ms * sample_rate / 1000)
        search_audio = audio_16k[start_samp:end_samp]

        onset_refined_ms = cue.start_ms

        if len(search_audio) > sample_rate * 0.1:
            hop_ms = 10
            env = _compute_onset_envelope(search_audio, sample_rate=sample_rate, hop_ms=hop_ms)
            if len(env) > 0:
                max_energy = np.max(env)
                if max_energy > 0.01:
                    threshold = 0.25 * max_energy
                    candidate_frames = np.where(env >= threshold)[0]
                    if len(candidate_frames) > 0:
                        first_frame = candidate_frames[0]
                        candidate_onset_ms = search_start_ms + first_frame * hop_ms
                        drift = abs(candidate_onset_ms - cue.start_ms)
                        if drift <= max_drift_ms:
                            onset_refined_ms = candidate_onset_ms

        # Drift safety gate
        if abs(onset_refined_ms - cue.start_ms) <= max_drift_ms:
            cue.start_ms = onset_refined_ms

        if cue.end_ms <= cue.start_ms:
            cue.end_ms = cue.start_ms + 2500

        # Word-level proportional allocation
        words = _tokenize_words(cue.text)
        if words:
            total_chars = sum(len(w) for w in words)
            duration = max(500, cue.end_ms - cue.start_ms)
            word_entries: list[dict[str, Any]] = []
            curr_ms = cue.start_ms
            for idx, w in enumerate(words):
                weight = len(w) / max(1, total_chars)
                w_dur = max(100, int(round(duration * weight)))
                w_end = min(cue.end_ms, curr_ms + w_dur) if idx < len(words) - 1 else cue.end_ms
                word_entries.append({
                    "word": w,
                    "start_ms": curr_ms,
                    "end_ms": w_end,
                })
                curr_ms = w_end
            cue.words = word_entries

    # Maintain monotonicity within each layer
    by_layer: dict[Any, list[LyricCue]] = {}
    for c in cues:
        by_layer.setdefault(c.layer_id, []).append(c)

    for lcues in by_layer.values():
        lcues.sort(key=lambda c: c.start_ms)
        for i in range(len(lcues) - 1):
            if lcues[i].end_ms > lcues[i + 1].start_ms:
                lcues[i].end_ms = max(lcues[i].start_ms + 100, lcues[i + 1].start_ms)

    return cues
