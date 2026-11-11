from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from lyra_ocr.video.downloader import download_video


def test_download_video_format_string_unrestricted_ext(tmp_path: Path):
    captured_options = {}

    class FakeYoutubeDL:
        def __init__(self, options):
            captured_options.update(options)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

        def extract_info(self, url, download=True):
            return {
                "id": "abc123",
                "title": "Test Video",
                "duration": 180,
                "height": 720,
                "format_id": "247+140",
            }

        def prepare_filename(self, info):
            p = tmp_path / "Test_Video [abc123].mp4"
            p.write_bytes(b"dummy")
            return str(p)

    with patch("yt_dlp.YoutubeDL", FakeYoutubeDL):
        res = download_video("https://youtube.com/watch?v=abc123", str(tmp_path), max_height=720)

    # 1. Format string must not restrict bestvideo to [ext=mp4], because merge_output_format converts
    fmt = captured_options.get("format", "")
    assert "[ext=mp4]" not in fmt
    assert "bestvideo[height<=720]+bestaudio/best[height<=720]/best" in fmt
    assert captured_options.get("merge_output_format") == "mp4"

    # 2. Result must record height and format_id
    assert res["height"] == 720
    assert res["format_id"] == "247+140"
