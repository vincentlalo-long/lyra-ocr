"""Audio extraction and source separation utilities.

References:
- Syed et al. (2025), "Exploiting Music Source Separation for Automatic Lyrics
  Transcription with Whisper", IEEE ICME 2025W, https://arxiv.org/abs/2506.15514
- Rouard et al. (2023), "Hybrid Transformers for Music Source Separation (HTDemucs)",
  IEEE ICASSP 2023, https://arxiv.org/abs/2211.08553
- Fitzgerald (2010), "Harmonic/Percussive Separation Using Median Filtering", DAFx-10
- CixiSVesper/dsh-soundtry, https://github.com/CixiSVesper/dsh-soundtry
"""

from __future__ import annotations

import importlib.util
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy import ndimage, signal

from lyra_ocr.config import DEFAULT_VOCAL_ISOLATION_THRESHOLDS, VocalIsolationThresholds

logger = logging.getLogger(__name__)


def extract_audio_16k(
    video_path: str | Path,
    cache_dir: str | Path | None = None,
) -> np.ndarray | None:
    """Extract mono 16kHz float32 audio samples in [-1.0, 1.0] from a video or audio file."""
    src = Path(video_path)
    if not src.is_file():
        logger.warning("Media file not found: %s", src)
        return None

    if cache_dir is not None:
        cdir = Path(cache_dir)
        cdir.mkdir(parents=True, exist_ok=True)
        cached_npy = cdir / "audio_16k.npy"
        if cached_npy.is_file():
            try:
                return np.load(cached_npy)
            except Exception as exc:
                logger.warning("Failed to load cached audio %s: %s", cached_npy, exc)

    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(src),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "s16le",
        "-",
    ]

    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        raw_bytes = proc.stdout
    except Exception as exc:
        logger.warning("ffmpeg audio extraction failed for %s: %s", src, exc)
        return None

    if not raw_bytes:
        return None

    pcm16 = np.frombuffer(raw_bytes, dtype=np.int16)
    audio_float = (pcm16.astype(np.float32) / 32768.0).copy()

    if cache_dir is not None:
        try:
            cached_npy = Path(cache_dir) / "audio_16k.npy"
            np.save(cached_npy, audio_float)
        except Exception as exc:
            logger.warning("Failed to save audio cache %s: %s", cached_npy, exc)

    return audio_float


def separate_vocals_demucs(
    video_path: str | Path,
    cache_dir: str | Path,
    model: str = "htdemucs",
) -> Path | None:
    """Extract isolated vocal track using Demucs if installed."""
    if importlib.util.find_spec("demucs") is None:
        logger.info("demucs package not installed; skipping neural vocal separation.")
        return None

    src = Path(video_path)
    if not src.is_file():
        return None

    out_dir = Path(cache_dir) / "demucs"
    vocal_target = out_dir / model / src.stem / "vocals.wav"
    if vocal_target.is_file():
        return vocal_target

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "demucs",
        "--two-stems",
        "vocals",
        "-n",
        model,
        "-o",
        str(out_dir),
        str(src),
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if vocal_target.is_file():
            return vocal_target
    except Exception as exc:
        logger.warning("Demucs vocal separation failed: %s", exc)

    return None


def isolate_vocals_midside_hpss(
    stereo_16k: np.ndarray,
    sample_rate: int = 16000,
    *,
    thresholds: VocalIsolationThresholds | None = None,
) -> np.ndarray:
    """Isolate center-panned harmonic singing vocals via Mid-Side STFT masking and HPSS."""
    if stereo_16k is None:
        return np.zeros(0, dtype=np.float32)

    arr = np.asarray(stereo_16k, dtype=np.float32)
    if arr.size == 0:
        return np.zeros(0, dtype=np.float32)

    if arr.ndim == 1:
        mid = arr
        side = np.zeros_like(mid)
    elif arr.ndim == 2 and arr.shape[1] == 1:
        mid = arr[:, 0]
        side = np.zeros_like(mid)
    elif arr.ndim == 2 and arr.shape[1] >= 2:
        left = arr[:, 0]
        right = arr[:, 1]
        mid = 0.5 * (left + right)
        side = 0.5 * (left - right)
    else:
        mid = arr.reshape(-1)
        side = np.zeros_like(mid)

    n_samples = len(mid)
    if n_samples == 0:
        return np.zeros(0, dtype=np.float32)

    cfg = thresholds or DEFAULT_VOCAL_ISOLATION_THRESHOLDS
    nperseg = min(cfg.stft_nperseg, n_samples)
    noverlap = min(cfg.stft_noverlap, max(0, nperseg - 1))

    freqs, _, z_mid = signal.stft(mid, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)
    mag_mid = np.abs(z_mid)

    if np.any(side):
        _, _, z_side = signal.stft(side, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)
        mag_side = np.abs(z_side)
        center_mask = np.clip(
            (mag_mid - cfg.side_suppression_weight * mag_side) / (mag_mid + 1e-7),
            0.0,
            1.0,
        ) ** 2
    else:
        center_mask = np.ones_like(mag_mid, dtype=np.float32)

    center_mag = mag_mid * center_mask
    harm = ndimage.median_filter(center_mag, size=(1, cfg.hpss_harmonic_kernel))
    perc = ndimage.median_filter(mag_mid, size=(cfg.hpss_percussive_kernel, 1))
    hpss_mask = (harm ** 2) / (harm ** 2 + perc ** 2 + 1e-7)

    freq_mask = (
        (freqs >= cfg.vocal_band_low_hz) & (freqs <= cfg.vocal_band_high_hz)
    ).astype(np.float32)[:, None]

    z_vocals = z_mid * center_mask * hpss_mask * freq_mask
    _, vocals = signal.istft(z_vocals, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)

    if len(vocals) >= n_samples:
        return vocals[:n_samples].astype(np.float32)
    out = np.zeros(n_samples, dtype=np.float32)
    out[: len(vocals)] = vocals.astype(np.float32)
    return out


def extract_isolated_vocals_16k(
    video_path: str | Path,
    cache_dir: str | Path | None = None,
) -> np.ndarray | None:
    """Extract a 16kHz mono isolated vocal waveform via Demucs or Mid-Side HPSS fallback."""
    src = Path(video_path)
    if not src.is_file():
        logger.warning("Media file not found: %s", src)
        return None

    cdir: Path | None = None
    if cache_dir is not None:
        cdir = Path(cache_dir)
        cdir.mkdir(parents=True, exist_ok=True)
        cached_npy = cdir / "vocals_16k.npy"
        if cached_npy.is_file():
            try:
                return np.load(cached_npy)
            except Exception as exc:
                logger.warning("Failed to load cached isolated vocals %s: %s", cached_npy, exc)

    vocals_wav: Path | None = None
    if cdir is not None:
        vocals_wav = separate_vocals_demucs(src, cdir)
        if vocals_wav is not None:
            vocals_audio = extract_audio_16k(vocals_wav, cache_dir=None)
            if vocals_audio is not None and len(vocals_audio) > 0:
                try:
                    np.save(cdir / "vocals_16k.npy", vocals_audio)
                except Exception as exc:
                    logger.warning("Failed to save vocals cache %s: %s", cdir / "vocals_16k.npy", exc)
                return vocals_audio
    elif importlib.util.find_spec("demucs") is not None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            vocals_wav = separate_vocals_demucs(src, tmp_dir)
            if vocals_wav is not None:
                vocals_audio = extract_audio_16k(vocals_wav, cache_dir=None)
                if vocals_audio is not None and len(vocals_audio) > 0:
                    return vocals_audio

    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(src),
        "-vn",
        "-ac",
        "2",
        "-ar",
        "16000",
        "-f",
        "s16le",
        "-",
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        raw_bytes = proc.stdout
    except Exception as exc:
        logger.warning("ffmpeg stereo vocal extraction failed for %s: %s", src, exc)
        return None

    if not raw_bytes:
        return None

    pcm16 = np.frombuffer(raw_bytes, dtype=np.int16)
    n_pairs = len(pcm16) // 2
    if n_pairs == 0:
        return None

    stereo_16k = pcm16[: n_pairs * 2].reshape(-1, 2).astype(np.float32) / 32768.0
    vocals = isolate_vocals_midside_hpss(stereo_16k, sample_rate=16000)

    if cdir is not None and vocals.size > 0:
        try:
            np.save(cdir / "vocals_16k.npy", vocals)
        except Exception as exc:
            logger.warning("Failed to save vocals cache %s: %s", cdir / "vocals_16k.npy", exc)

    return vocals
