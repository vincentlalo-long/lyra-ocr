"""Unit tests for frames.py (PTS extraction and frame sampling)."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from lyra_ocr.video.frames import sample_frames


def test_sample_frames_missing_file():
    with pytest.raises(FileNotFoundError):
        sample_frames("/non/existent/video.mp4", "/tmp/out")


def test_sample_frames_pts_extraction(tmp_path):
    dummy_video = tmp_path / "video.mp4"
    dummy_video.write_bytes(b"dummy")

    out_dir = tmp_path / "out"

    # Simulate ffmpeg stderr with showinfo
    fake_stderr = (
        "[Parsed_showinfo_1 @ 0x1] n: 0 pts: 0 pts_time:0.0 duration_time:0.5\n"
        "[Parsed_showinfo_1 @ 0x1] n: 1 pts: 1 pts_time:0.512 duration_time:0.5\n"
        "[Parsed_showinfo_1 @ 0x1] n: 2 pts: 2 pts_time:1.025 duration_time:0.5\n"
    )

    def fake_subprocess_run(cmd, **kwargs):
        # Create dummy extracted frame files in frames_dir
        frames_dir = out_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        (frames_dir / "frame_000000.jpg").write_bytes(b"jpg0")
        (frames_dir / "frame_000001.jpg").write_bytes(b"jpg1")
        (frames_dir / "frame_000002.jpg").write_bytes(b"jpg2")
        return MagicMock(returncode=0, stderr=fake_stderr)

    with patch("subprocess.run", side_effect=fake_subprocess_run), patch("shutil.which", return_value="/usr/bin/ffmpeg"):
        entries = sample_frames(dummy_video, out_dir, fps=2.0, start_sec=10.0)

    assert len(entries) == 3
    # Frame 0: 10.0 + 0.0 -> 10000ms
    assert entries[0]["frame_id"] == 0
    assert entries[0]["timestamp_ms"] == 10000
    assert entries[0]["image_path"] == "frames/frame_000000.jpg"

    # Frame 1: 10.0 + 0.512 -> 10512ms
    assert entries[1]["frame_id"] == 1
    assert entries[1]["timestamp_ms"] == 10512

    # Frame 2: 10.0 + 1.025 -> 11025ms
    assert entries[2]["frame_id"] == 2
    assert entries[2]["timestamp_ms"] == 11025

    # Check frame_index.json on disk
    index_file = out_dir / "frame_index.json"
    assert index_file.is_file()
    saved = json.loads(index_file.read_text(encoding="utf-8"))
    assert len(saved) == 3
    assert saved[0]["timestamp_ms"] == 10000
