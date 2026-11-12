import math
import struct
import wave
from pathlib import Path
import numpy as np
import pytest

from lyra_ocr.audio.extract import extract_audio_16k, separate_vocals_demucs


def _create_test_wav(path: Path, duration_sec: float = 1.0, freq: float = 440.0, sr: int = 16000) -> Path:
    n_samples = int(duration_sec * sr)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        for i in range(n_samples):
            val = int(32767.0 * 0.5 * math.sin(2.0 * math.pi * freq * i / sr))
            wf.writeframesraw(struct.pack("<h", val))
    return path


def test_extract_audio_16k_valid_audio(tmp_path: Path):
    audio_file = tmp_path / "test_16k.wav"
    _create_test_wav(audio_file, duration_sec=1.0, freq=440.0, sr=16000)

    audio = extract_audio_16k(audio_file)
    assert audio is not None
    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
    assert abs(len(audio) - 16000) < 100
    assert -1.0 <= np.min(audio) <= 0.0
    assert 0.0 <= np.max(audio) <= 1.0


def test_extract_audio_16k_caching(tmp_path: Path):
    audio_file = tmp_path / "test_cache.wav"
    _create_test_wav(audio_file, duration_sec=0.5, freq=440.0, sr=16000)
    cache_dir = tmp_path / "audio_cache"

    # First call: creates cache
    audio1 = extract_audio_16k(audio_file, cache_dir=cache_dir)
    assert audio1 is not None
    cached_file = cache_dir / "audio_16k.npy"
    assert cached_file.exists()

    # Second call: reads from cache
    audio2 = extract_audio_16k(audio_file, cache_dir=cache_dir)
    assert audio2 is not None
    assert np.array_equal(audio1, audio2)


def test_extract_audio_16k_nonexistent_returns_none(tmp_path: Path):
    nonexistent = tmp_path / "does_not_exist.mp4"
    assert extract_audio_16k(nonexistent) is None


def test_separate_vocals_demucs_graceful_missing(tmp_path: Path, monkeypatch):
    import importlib.util
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    dummy_vid = tmp_path / "dummy.mp4"
    dummy_vid.write_bytes(b"dummy")
    res = separate_vocals_demucs(dummy_vid, cache_dir=tmp_path / "cache")
    assert res is None
