import math
import shutil
import struct
import wave
from pathlib import Path
import numpy as np
import pytest

from lyra_ocr.audio.extract import extract_audio_16k, separate_vocals_demucs

requires_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


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


@requires_ffmpeg
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


@requires_ffmpeg
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


def test_isolate_vocals_midside_hpss_suppresses_stereo_panned_instruments_and_percussion(
    tmp_path: Path, monkeypatch
):
    import importlib.util
    from lyra_ocr.audio.extract import extract_isolated_vocals_16k, isolate_vocals_midside_hpss

    sr = 16000
    t_half = np.linspace(0.0, 1.0, sr, endpoint=False, dtype=np.float32)
    rng = np.random.default_rng(42)

    # [0..1s]: hard-panned stereo accompaniment (L=+sin, R=-sin) + percussive noise bursts
    panned_tone = 0.5 * np.sin(2.0 * np.pi * 500.0 * t_half)
    perc_bursts = np.zeros_like(t_half)
    for idx in range(800, sr, 2000):
        perc_bursts[idx : idx + 120] = 0.4 * rng.standard_normal(120).astype(np.float32)
    left_0 = panned_tone + perc_bursts
    right_0 = -panned_tone - perc_bursts

    # [1..2s]: center-panned harmonic singing tone (L = R = 440Hz + 880Hz)
    vocal_tone = (
        0.4 * np.sin(2.0 * np.pi * 440.0 * t_half)
        + 0.25 * np.sin(2.0 * np.pi * 880.0 * t_half)
    ).astype(np.float32)

    stereo = np.vstack(
        [
            np.column_stack([left_0, right_0]),
            np.column_stack([vocal_tone, vocal_tone]),
        ]
    ).astype(np.float32)

    isolated = isolate_vocals_midside_hpss(stereo, sample_rate=sr)
    assert isolated.dtype == np.float32
    assert isolated.shape == (2 * sr,)

    in_rms_0_1 = float(np.sqrt(np.mean(stereo[:sr] ** 2)))
    iso_rms_0_1 = float(np.sqrt(np.mean(isolated[:sr] ** 2)))
    assert iso_rms_0_1 < 0.20 * in_rms_0_1

    in_rms_1_2 = float(np.sqrt(np.mean(vocal_tone ** 2)))
    iso_rms_1_2 = float(np.sqrt(np.mean(isolated[sr:] ** 2)))
    assert iso_rms_1_2 > 0.60 * in_rms_1_2

    if shutil.which("ffmpeg") is not None:
        monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
        stereo_wav = tmp_path / "stereo.wav"
        pcm = np.clip(stereo * 32767.0, -32768, 32767).astype("<i2")
        with wave.open(str(stereo_wav), "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm.tobytes())

        cache_dir = tmp_path / "vocal_cache"
        vocals1 = extract_isolated_vocals_16k(stereo_wav, cache_dir=cache_dir)
        assert vocals1 is not None
        assert (cache_dir / "vocals_16k.npy").is_file()
        vocals2 = extract_isolated_vocals_16k(stereo_wav, cache_dir=cache_dir)
        assert vocals2 is not None
        assert np.array_equal(vocals1, vocals2)
