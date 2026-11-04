"""Unit tests for tracking algorithms: Greedy, ByteTrack, SORT, and ST-DBSCAN."""

import pytest
from lyra_ocr.trackers import (
    BaseTracker,
    ByteTextTracker,
    GreedyTracker,
    KalmanSORTTracker,
    ST_DBSCANTracker,
    get_tracker,
)


def make_box(xmin, ymin, xmax, ymax, text, conf=0.98, script="latin"):
    return {
        "polygon": [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]],
        "bbox": [xmin, ymin, xmax, ymax],
        "text_raw": text,
        "text_norm": text,
        "confidence": conf,
        "model_id": "PP-OCRv6_rec_small",
        "script": script,
    }


def test_tracker_factory_resolution():
    assert isinstance(get_tracker("greedy"), GreedyTracker)
    assert isinstance(get_tracker("bytetrack"), ByteTextTracker)
    assert isinstance(get_tracker("sort"), KalmanSORTTracker)
    assert isinstance(get_tracker("dbscan"), ST_DBSCANTracker)

    with pytest.raises(ValueError, match="Unknown tracker"):
        get_tracker("non_existent_tracker")


def test_bytetrack_recovers_low_confidence_blurred_frame():
    """Verify ByteTrack avoids ID switch when a middle frame suffers from motion blur / low confidence."""
    tracker = ByteTextTracker(frame_width=1280, frame_height=720, high_threshold=0.70, low_threshold=0.25)

    # Frame 0: Clear detection (conf = 0.95)
    tracker.update({
        "frame_id": 0,
        "timestamp_ms": 1000,
        "boxes": [make_box(300, 600, 800, 650, "We can be alone in the castle in the sky", conf=0.95)],
    })
    assert len(tracker.active_tracks) == 1
    t0_id = tracker.active_tracks[0].track_id

    # Frame 1: Motion blur causes low OCR confidence (conf = 0.40), but position matches
    tracker.update({
        "frame_id": 1,
        "timestamp_ms": 1500,
        "boxes": [make_box(302, 601, 798, 651, "We can be alone in the castle in the sky", conf=0.40)],
    })
    # Should maintain the same track (no ID switch)
    assert len(tracker.active_tracks) == 1
    assert tracker.active_tracks[0].track_id == t0_id
    assert len(tracker.active_tracks[0].occurrences) == 2

    # Frame 2: Clear again (conf = 0.98)
    tracker.update({
        "frame_id": 2,
        "timestamp_ms": 2000,
        "boxes": [make_box(300, 600, 800, 650, "We can be alone in the castle in the sky", conf=0.98)],
    })
    tracker.finalize(video_duration_ms=5000)

    assert len(tracker.all_tracks) == 1
    assert tracker.all_tracks[0].track_id == t0_id
    assert len(tracker.all_tracks[0].occurrences) == 3


def test_bytetrack_discards_isolated_low_confidence_noise():
    """Verify ByteTrack never initiates new tracks from low confidence noise."""
    tracker = ByteTextTracker(frame_width=1280, frame_height=720, high_threshold=0.70, low_threshold=0.25)

    # Frame 0: Only a low confidence noise box (conf = 0.35)
    tracker.update({
        "frame_id": 0,
        "timestamp_ms": 1000,
        "boxes": [make_box(50, 50, 100, 70, "NOISE", conf=0.35)],
    })
    # Should NOT spawn a track
    assert len(tracker.active_tracks) == 0


def test_sort_tracker_tracks_moving_kinetic_text():
    """Verify KalmanSORTTracker tracks moving text across frames where static IoU would drift."""
    tracker = KalmanSORTTracker(frame_width=1280, frame_height=720, iou_threshold=0.20)

    # Moving lyric line traveling horizontally by 30px per frame
    # Frame 0: x in [100, 400]
    tracker.update({
        "frame_id": 0,
        "timestamp_ms": 1000,
        "boxes": [make_box(100, 600, 400, 650, "Moving Lyric")],
    })
    t_id = tracker.active_tracks[0].track_id

    # Frame 1: shifted to [130, 430]
    tracker.update({
        "frame_id": 1,
        "timestamp_ms": 1500,
        "boxes": [make_box(130, 600, 430, 650, "Moving Lyric")],
    })
    assert len(tracker.active_tracks) == 1
    assert tracker.active_tracks[0].track_id == t_id

    # Frame 2: shifted to [160, 460]
    tracker.update({
        "frame_id": 2,
        "timestamp_ms": 2000,
        "boxes": [make_box(160, 600, 460, 650, "Moving Lyric")],
    })
    assert len(tracker.active_tracks) == 1
    assert tracker.active_tracks[0].track_id == t_id

    tracker.finalize(video_duration_ms=5000)
    assert len(tracker.all_tracks) == 1
    assert len(tracker.all_tracks[0].occurrences) == 3


def test_st_dbscan_tube_clustering():
    """Verify ST_DBSCAN discovers spatio-temporal text tubes and filters isolated noise."""
    tracker = ST_DBSCANTracker(frame_width=1280, frame_height=720, min_samples=2, eps=0.25)

    # 1. Consistent lyric tube at bottom (t = 1000, 1500, 2000)
    for fid, t_ms in enumerate([1000, 1500, 2000]):
        tracker.update({
            "frame_id": fid,
            "timestamp_ms": t_ms,
            "boxes": [
                make_box(300, 600, 800, 650, "Bottom Subtitle Line", conf=0.95),
                make_box(50, 50, 200, 80, "CHANNEL_LOGO", conf=0.98),
            ],
        })

    # Isolated 1-frame noise in middle of screen (should be treated as noise or outlier)
    tracker.update({
        "frame_id": 3,
        "timestamp_ms": 2500,
        "boxes": [make_box(500, 300, 600, 350, "RANDOM_FLICKER", conf=0.90)],
    })

    tracks = tracker.finalize(video_duration_ms=5000)
    # Should discover 2 primary tubes: Subtitle Line and CHANNEL_LOGO
    assert len(tracks) == 2
    dominant_texts = {t.features["dominant_text"] for t in tracks}
    assert "Bottom Subtitle Line" in dominant_texts
    assert "CHANNEL_LOGO" in dominant_texts
