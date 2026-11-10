"""Unit tests for probe.py (ffprobe parser)."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from lyra_ocr.video.probe import probe_video


def test_probe_missing_file():
    with pytest.raises(FileNotFoundError):
        probe_video("/non/existent/video.mp4")


def test_probe_valid_json(tmp_path, sample_probe_json):
    dummy_video = tmp_path / "dummy.mp4"
    dummy_video.write_bytes(b"dummy")

    mock_res = MagicMock()
    mock_res.returncode = 0
    mock_res.stdout = sample_probe_json

    with patch("subprocess.run", return_value=mock_res), patch("shutil.which", return_value="/usr/bin/ffprobe"):
        meta = probe_video(dummy_video)

    assert meta["width"] == 1920
    assert meta["height"] == 1080
    assert meta["fps"] == 30.0
    assert meta["duration"] == 120.5
    assert meta["rotation"] == 0
    assert meta["has_audio"] is True


def test_probe_fractional_fps(tmp_path):
    dummy_video = tmp_path / "dummy.mp4"
    dummy_video.write_bytes(b"dummy")

    data = {
        "streams": [
            {
                "codec_type": "video",
                "width": 1280,
                "height": 720,
                "r_frame_rate": "30000/1001",
                "duration": "60.0",
                "side_data_list": [{"rotation": -90}],
            }
        ],
        "format": {"duration": "60.0"},
    }

    mock_res = MagicMock(returncode=0, stdout=json.dumps(data))
    with patch("subprocess.run", return_value=mock_res), patch("shutil.which", return_value="/usr/bin/ffprobe"):
        meta = probe_video(dummy_video)

    assert meta["fps"] == 29.97
    assert meta["rotation"] == 270  # -90 % 360
    assert meta["width"] == 720  # swapped from 1280 due to 270 deg rotation
    assert meta["height"] == 1280  # swapped from 720 due to 270 deg rotation
    assert meta["has_audio"] is False


def test_probe_no_video_stream(tmp_path):
    dummy_video = tmp_path / "dummy.mp4"
    dummy_video.write_bytes(b"dummy")

    data = {"streams": [{"codec_type": "audio"}], "format": {}}
    mock_res = MagicMock(returncode=0, stdout=json.dumps(data))
    with patch("subprocess.run", return_value=mock_res), patch("shutil.which", return_value="/usr/bin/ffprobe"):
        with pytest.raises(RuntimeError, match="No video stream found"):
            probe_video(dummy_video)
