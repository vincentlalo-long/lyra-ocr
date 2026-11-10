import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from lyra_ocr.lyric.models import LyricCue
from lyra_ocr.lyric.timing import refine_timings
from lyra_ocr.video.frames import FramePacket


def test_refine_timings_disabled():
    cues = [LyricCue(start_ms=1000, end_ms=3000, text="TEST")]
    res = refine_timings(cues, "video.mp4", {}, "/tmp/nonexistent", enabled=False)
    assert res == cues
    assert res[0].start_ms == 1000


def test_refine_timings_missing_files(tmp_path: Path):
    cues = [LyricCue(start_ms=1000, end_ms=3000, text="TEST")]
    # Video exists but artifacts dir has no frame_index.json
    video_path = tmp_path / "vid.mp4"
    video_path.write_bytes(b"data")
    res = refine_timings(cues, str(video_path), {}, str(tmp_path), enabled=True)
    assert res == cues


def test_refine_timings_with_mock_scan(tmp_path: Path):
    video_path = tmp_path / "vid.mp4"
    video_path.write_bytes(b"data")

    # Create dummy frame and frame_index.json
    import cv2

    frame_img = np.zeros((100, 200, 3), dtype=np.uint8)
    # Put a white rectangle at [20, 20, 40, 40]
    frame_img[20:40, 20:40] = 255
    cv2.imwrite(str(tmp_path / "frame_0.jpg"), frame_img)

    frame_index = {
        "0": {"image_path": "frame_0.jpg", "timestamp_ms": 1000}
    }
    (tmp_path / "frame_index.json").write_text(json.dumps(frame_index), encoding="utf-8")

    tracks_data = {
        "tracks": [
            {
                "track_id": 1,
                "occurrences": [
                    {"frame_id": 0, "confidence": 0.99, "bbox": [20, 20, 40, 40]}
                ],
            }
        ]
    }
    (tmp_path / "tracks.json").write_text(json.dumps(tracks_data), encoding="utf-8")

    cue = LyricCue(
        start_ms=1000,
        end_ms=3000,
        text="TEST",
        source_track_ids=[1],
        bbox=[20, 20, 40, 40],
    )

    # Mock frames returned during dense scan
    # Let frame at 800ms have no match (zeros)
    # Let frame at 900ms have match (white box)
    # Let frame at 2800ms have match (white box)
    # Let frame at 3100ms have no match (zeros)
    match_frame = np.zeros((100, 200, 3), dtype=np.uint8)
    match_frame[20:40, 20:40] = 255

    no_match_frame = np.zeros((100, 200, 3), dtype=np.uint8)

    mock_packets = [
        FramePacket(frame_id=1, timestamp_ms=800, pts_time=0.8, image_path="", image_bgr=no_match_frame, jpeg_bytes=b"", pts_source=""),
        FramePacket(frame_id=2, timestamp_ms=900, pts_time=0.9, image_path="", image_bgr=match_frame, jpeg_bytes=b"", pts_source=""),
        FramePacket(frame_id=3, timestamp_ms=2800, pts_time=2.8, image_path="", image_bgr=match_frame, jpeg_bytes=b"", pts_source=""),
        FramePacket(frame_id=4, timestamp_ms=3100, pts_time=3.1, image_path="", image_bgr=no_match_frame, jpeg_bytes=b"", pts_source=""),
    ]

    with patch("lyra_ocr.lyric.timing.iter_sampled_frames", return_value=iter(mock_packets)):
        refined = refine_timings(
            [cue],
            str(video_path),
            {"probe": {"duration": 10.0, "width": 200, "height": 100}},
            str(tmp_path),
            fps=25.0,
            window_ms=500,
            enabled=True,
        )

    assert len(refined) == 1
    # 900ms is the first matching frame, so start_ms should be 900
    assert refined[0].start_ms == 900
    # 2800ms is the last matching frame + frame_interval (40ms) -> 2840ms
    assert refined[0].end_ms == 2840
