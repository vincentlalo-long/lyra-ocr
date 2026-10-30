"""Shared pytest fixtures and test helpers for lyra-ocr."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def sample_probe_json() -> str:
    """Mock ffprobe JSON output."""
    return json.dumps(
        {
            "streams": [
                {
                    "codec_type": "video",
                    "width": 1920,
                    "height": 1080,
                    "r_frame_rate": "30/1",
                    "avg_frame_rate": "30/1",
                    "duration": "120.500",
                    "tags": {"rotate": "0"},
                },
                {
                    "codec_type": "audio",
                    "codec_name": "aac",
                },
            ],
            "format": {
                "duration": "120.500",
                "size": "50000000",
            },
        }
    )
