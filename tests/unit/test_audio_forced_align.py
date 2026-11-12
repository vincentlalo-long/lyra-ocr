import math
import numpy as np
import pytest

from lyra_ocr.audio.forced_align import align_cues_with_audio
from lyra_ocr.lyric.models import LyricCue


def _generate_tone(freq: float, duration_sec: float, sr: int = 16000, amp: float = 0.5) -> np.ndarray:
    t = np.linspace(0, duration_sec, int(duration_sec * sr), endpoint=False)
    return (amp * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)


def test_align_cues_with_audio_refines_onset_and_populates_words():
    sr = 16000
    # 6 seconds total
    audio = np.zeros(sr * 6, dtype=np.float32)
    # Vocal onset begins at 1000ms and lasts until 3000ms
    audio[16 * 1000 : 16 * 3000] = _generate_tone(800.0, 2.0, sr=sr, amp=0.5)

    # Visual OCR cue had slight lag: start_ms = 1250, end_ms = 3200
    cues = [LyricCue(start_ms=1250, end_ms=3200, text="hello world today", layer_id=1)]

    aligned = align_cues_with_audio(cues, audio, sample_rate=sr, max_drift_ms=1200)
    assert len(aligned) == 1
    cue = aligned[0]
    # Onset refined closer to 1000ms
    assert abs(cue.start_ms - 1000) <= 150
    # Word level timings generated
    assert len(cue.words) == 3
    words = [w["word"] for w in cue.words]
    assert words == ["hello", "world", "today"]
    assert cue.words[0]["start_ms"] >= cue.start_ms
    assert cue.words[-1]["end_ms"] <= cue.end_ms + 100


def test_align_cues_with_audio_drift_gate_rejects_distant_onset():
    sr = 16000
    audio = np.zeros(sr * 10, dtype=np.float32)
    # Burst at 1000ms
    audio[16 * 1000 : 16 * 2000] = _generate_tone(800.0, 1.0, sr=sr, amp=0.5)

    # Cue is at 6000ms (drift = 5000ms > 1200ms)
    cues = [LyricCue(start_ms=6000, end_ms=8000, text="distant phrase", layer_id=1)]

    aligned = align_cues_with_audio(cues, audio, sample_rate=sr, max_drift_ms=1200)
    # Drift gate blocks shifting start_ms
    assert aligned[0].start_ms == 6000
