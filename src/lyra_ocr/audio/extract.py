"""Audio extraction and source separation utilities."""

from __future__ import annotations

import importlib.util
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np

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
