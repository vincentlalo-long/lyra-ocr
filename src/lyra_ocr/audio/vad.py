"""Vocal activity detection using band-pass filtering and spectral energy."""

from __future__ import annotations

import logging
import numpy as np
from scipy import signal

logger = logging.getLogger(__name__)


def compute_vocal_activity_profile(
    audio: np.ndarray,
    sample_rate: int = 16000,
    hop_ms: int = 50,
) -> np.ndarray:
    """Compute frame-level vocal activity probability profile in [0.0, 1.0].

    Uses a 4th order Butterworth band-pass filter isolating the 300Hz - 3400Hz
    singing vocal formant range, combined with short-time frame RMS energy analysis.
    """
    if audio is None or len(audio) == 0:
        return np.zeros(0, dtype=np.float32)

    audio = audio.astype(np.float32)

    # 4th order Butterworth band-pass filter (300Hz - 3400Hz)
    nyq = 0.5 * sample_rate
    low = 300.0 / nyq
    high = min(3400.0 / nyq, 0.99)
    sos = signal.butter(4, [low, high], btype="bandpass", output="sos")
    filtered = signal.sosfiltfilt(sos, audio)

    hop_samples = int(sample_rate * (hop_ms / 1000.0))
    if hop_samples <= 0:
        hop_samples = int(sample_rate * 0.05)

    n_frames = int(len(audio) / hop_samples)
    if n_frames <= 0:
        return np.zeros(0, dtype=np.float32)

    profile = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        start = i * hop_samples
        end = min(len(audio), start + hop_samples)
        frame_filt = filtered[start:end]
        frame_orig = audio[start:end]

        rms_filt = float(np.sqrt(np.mean(frame_filt ** 2))) if len(frame_filt) > 0 else 0.0
        rms_orig = float(np.sqrt(np.mean(frame_orig ** 2))) if len(frame_orig) > 0 else 0.0

        vocal_ratio = (rms_filt / (rms_orig + 1e-6)) if rms_orig > 1e-5 else 0.0
        energy_weight = min(1.0, rms_filt / 0.015)

        if rms_orig < 0.005:
            prob = 0.0
        else:
            prob = min(1.0, vocal_ratio * energy_weight * 1.5)

        profile[i] = prob

    return profile


def vocal_active_ratio(
    profile: np.ndarray,
    start_ms: int,
    end_ms: int,
    hop_ms: int = 50,
) -> float:
    """Calculate the fraction of active vocal frames within [start_ms, end_ms]."""
    if profile is None or len(profile) == 0:
        return 1.0  # neutral fallback if no audio profile available

    start_frame = max(0, int(start_ms / hop_ms))
    end_frame = min(len(profile), int(end_ms / hop_ms))

    if start_frame >= end_frame:
        return 0.0

    slice_frames = profile[start_frame:end_frame]
    active_count = np.sum(slice_frames >= 0.35)
    return float(active_count / len(slice_frames))
