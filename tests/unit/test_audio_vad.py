import math
import numpy as np
import pytest

from lyra_ocr.audio.vad import compute_vocal_activity_profile, vocal_active_ratio
from lyra_ocr.lyric.models import LyricCue
from lyra_ocr.lyric.reference import ReferenceBundle, ReferenceLine, fill_reference_gaps


def _generate_tone(freq: float, duration_sec: float, sr: int = 16000, amp: float = 0.5) -> np.ndarray:
    t = np.linspace(0, duration_sec, int(duration_sec * sr), endpoint=False)
    return (amp * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)


def test_vad_profile_silence_vs_vocal_tone():
    sr = 16000
    # 2 seconds of silence followed by 2 seconds of 1000Hz tone (vocal band)
    silence = np.zeros(sr * 2, dtype=np.float32)
    tone = _generate_tone(1000.0, 2.0, sr=sr, amp=0.4)
    audio = np.concatenate([silence, tone])

    profile = compute_vocal_activity_profile(audio, sample_rate=sr, hop_ms=50)
    assert len(profile) == int(len(audio) / (sr * 0.05))

    # Silence section [0, 2000ms] should have very low activity
    silence_ratio = vocal_active_ratio(profile, start_ms=0, end_ms=2000, hop_ms=50)
    assert silence_ratio < 0.10

    # Vocal band tone section [2000, 4000ms] should have high activity
    tone_ratio = vocal_active_ratio(profile, start_ms=2000, end_ms=4000, hop_ms=50)
    assert tone_ratio > 0.80


def test_vad_rejects_low_rumble_noise():
    sr = 16000
    # 50Hz sub-bass rumble (below 300Hz vocal band)
    rumble = _generate_tone(50.0, 2.0, sr=sr, amp=0.4)
    profile = compute_vocal_activity_profile(rumble, sample_rate=sr, hop_ms=50)
    ratio = vocal_active_ratio(profile, start_ms=0, end_ms=2000, hop_ms=50)
    assert ratio < 0.20


def test_fill_reference_gaps_skips_silence_when_vad_provided():
    cues = [
        LyricCue(start_ms=1000, end_ms=3000, text="First line here"),
        LyricCue(start_ms=10000, end_ms=13000, text="Third line here"),
    ]
    ref_bundle = ReferenceBundle(
        lines=[
            ReferenceLine(text="First line here", start_ms=1000, end_ms=3000),
            ReferenceLine(text="Second line here", start_ms=5000, end_ms=8000),
            ReferenceLine(text="Third line here", start_ms=10000, end_ms=13000),
        ],
        source="lrclib_synced",
    )

    # 15 seconds audio: [0..15000ms], but [4000..9000ms] is pure silence
    audio = np.zeros(16000 * 15, dtype=np.float32)
    # Add active vocal tone during cues 1 and 3
    audio[16 * 1000 : 16 * 3000] = _generate_tone(1000.0, 2.0)
    audio[16 * 10000 : 16 * 13000] = _generate_tone(1000.0, 3.0)

    profile = compute_vocal_activity_profile(audio, sample_rate=16000, hop_ms=50)

    merged, gap_res = fill_reference_gaps(cues, ref_bundle, vocal_profile=profile)
    # The gap fill in the silent window [5000..8000ms] must be blocked
    assert gap_res.n_filled == 0
    assert len(merged) == 2
    assert any("silent" in w for w in gap_res.warnings)
