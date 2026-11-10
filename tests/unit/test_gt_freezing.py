import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from lyra_ocr.dataset import ground_truth
from lyra_ocr.dataset.ground_truth import GroundTruthSegment
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from auto_ground_truth import process_video


def test_process_video_does_not_overwrite_frozen_gt(tmp_path: Path):
    dataset_dir = tmp_path / "datasets" / "Test_DS"
    dataset_dir.mkdir(parents=True)
    video_dir = dataset_dir / "videos"
    video_dir.mkdir(parents=True)
    video_file = video_dir / "vid1.mp4"
    video_file.write_bytes(b"dummy")

    # 1. Establish an existing frozen Ground Truth
    frozen_segment = GroundTruthSegment(
        video_id="vid1",
        segment_id="gt_original",
        start_ms=1000,
        end_ms=3000,
        human_label="lyric",
        text="CANONICAL FROZEN TEXT",
        annotator="human_gold",
    )
    ground_truth.write_segments(dataset_dir, "vid1", [frozen_segment])

    # Verify frozen GT is written
    loaded_before = ground_truth.load_segments(dataset_dir, "vid1")
    assert len(loaded_before) == 1
    assert loaded_before[0].text == "CANONICAL FROZEN TEXT"

    # 2. Run process_video with a tracker producing different text
    fake_tracks = [
        MagicMock(
            track_id=99,
            to_dict=lambda: {
                "track_id": 99,
                "start_ms": 2000,
                "end_ms": 4000,
                "label": "lyric",
                "features": {"dominant_text": "DIFFERENT TRACKER TEXT"},
            },
        )
    ]

    with patch("auto_ground_truth.run_phase1_pipeline") as mock_pipe, \
         patch("auto_ground_truth.get_reference_lyrics", return_value=["CANONICAL FROZEN TEXT"]):
        mock_pipe.return_value = {"tracks": fake_tracks}

        run_dir = dataset_dir / "runs" / "tracker_run_1"
        run_dir.mkdir(parents=True)
        (run_dir / "videos" / "vid1").mkdir(parents=True)
        (run_dir / "videos" / "vid1" / "tracks.json").write_text(
            json.dumps({"tracks": [fake_tracks[0].to_dict()]}), encoding="utf-8"
        )

        video_entry = {"video_id": "vid1", "video_path": "videos/vid1.mp4", "title": "Song"}

        # process_video with write_gt=False (or default non-force behavior)
        process_video(
            video_entry,
            dataset_dir,
            run_dir,
            dataset_dir / "lyrics_cache",
            tracker="bytetrack",
            write_gt=False,
        )

    # 3. Assert frozen GT was NOT overwritten
    loaded_after = ground_truth.load_segments(dataset_dir, "vid1")
    assert len(loaded_after) == 1
    assert loaded_after[0].text == "CANONICAL FROZEN TEXT"
    assert loaded_after[0].segment_id == "gt_original"
