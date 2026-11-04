"""Unit tests for temporal text tracking (track.py)."""

import pytest
from lyra_ocr.ocr import TextLine


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


def test_iou_and_centroid():
    from lyra_ocr.track import bbox_iou, centroid_distance

    box1 = [100, 100, 200, 150]
    box2 = [100, 100, 200, 150]
    assert bbox_iou(box1, box2) == 1.0

    # Non-overlapping
    box3 = [300, 300, 400, 350]
    assert bbox_iou(box1, box3) == 0.0

    # 50% overlap approx
    box4 = [150, 100, 250, 150]
    iou = bbox_iou(box1, box4)
    assert 0.3 < iou < 0.4

    # Centroid distance
    dist = centroid_distance(box1, box2, 1280, 720)
    assert dist == 0.0


def test_tracker_associates_same_text_across_frames():
    from lyra_ocr.track import Tracker

    tracker = Tracker(frame_width=1280, frame_height=720, iou_threshold=0.3, max_gap_ms=1000)

    # Frame 0 (t = 1000ms)
    f0 = {
        "frame_id": 0,
        "timestamp_ms": 1000,
        "boxes": [
            make_box(500, 600, 800, 650, "Hello World"),
            make_box(50, 50, 200, 80, "CHANNEL LOGO"),
        ],
    }
    tracker.update(f0)
    assert len(tracker.active_tracks) == 2

    # Frame 1 (t = 1500ms) with slight box jitter
    f1 = {
        "frame_id": 1,
        "timestamp_ms": 1500,
        "boxes": [
            make_box(502, 601, 801, 652, "Hello World"),
            make_box(50, 50, 200, 80, "CHANNEL LOGO"),
        ],
    }
    tracker.update(f1)
    assert len(tracker.active_tracks) == 2

    # Both tracks should have 2 occurrences and 0 text changes
    for tr in tracker.active_tracks:
        assert len(tr.occurrences) == 2
        assert tr.n_text_changes == 0


def test_tracker_detects_text_change_in_same_region():
    from lyra_ocr.track import Tracker

    tracker = Tracker(frame_width=1280, frame_height=720, iou_threshold=0.3, max_gap_ms=1000)

    # Frame 0: Line 1
    tracker.update({
        "frame_id": 0,
        "timestamp_ms": 1000,
        "boxes": [make_box(500, 600, 800, 650, "First lyric line")],
    })

    # Frame 1: Still Line 1
    tracker.update({
        "frame_id": 1,
        "timestamp_ms": 1500,
        "boxes": [make_box(500, 600, 800, 650, "First lyric line")],
    })

    # Frame 2: Singer changes to Line 2 at the same position
    tracker.update({
        "frame_id": 2,
        "timestamp_ms": 2000,
        "boxes": [make_box(505, 598, 805, 649, "Second lyric line")],
    })

    tracker.finalize(video_duration_ms=5000)
    all_tracks = tracker.all_tracks
    assert len(all_tracks) == 1
    tr = all_tracks[0]
    assert len(tr.occurrences) == 3
    assert tr.n_text_changes == 1
    assert len(tr.segments) == 2
    assert tr.segments[0].text == "First lyric line"
    assert tr.segments[1].text == "Second lyric line"


def test_track_features_computation():
    from lyra_ocr.track import Tracker, compute_track_features

    tracker = Tracker(frame_width=1280, frame_height=720)
    tracker.update({
        "frame_id": 0,
        "timestamp_ms": 0,
        "boxes": [make_box(50, 50, 200, 80, "MY_CHANNEL")],
    })
    tracker.update({
        "frame_id": 1,
        "timestamp_ms": 10000,
        "boxes": [make_box(50, 50, 200, 80, "MY_CHANNEL")],
    })
    tracker.finalize(video_duration_ms=10000)

    tr = tracker.all_tracks[0]
    feat = compute_track_features(tr, video_duration_ms=10000, frame_width=1280, frame_height=720)

    assert feat["lifetime_ratio"] == 1.0
    assert feat["start_time_ratio"] == 0.0
    assert feat["n_text_changes"] == 0
    assert feat["is_corner"] is True
    assert feat["spatial_variance"] == 0.0


def test_tracker_legacy_schema_compatibility():
    from lyra_ocr.track import Tracker

    tracker = Tracker(frame_width=1280, frame_height=720)
    # Legacy schema has "text", no "text_raw" or "text_norm" or "model_id"
    legacy_frame = {
        "frame_id": 0,
        "timestamp_ms": 20000,
        "boxes": [
            {
                "polygon": [[48, 534], [460, 539], [459, 632], [46, 627]],
                "bbox": [46, 534, 460, 632],
                "text": "SUGARTiMe",
                "confidence": 0.8507,
                "script": "latin",
            }
        ],
    }
    tracker.update(legacy_frame)
    tracker.finalize(video_duration_ms=25000)

    assert len(tracker.all_tracks) == 1
    tr = tracker.all_tracks[0]
    occ = tr.occurrences[0]
    assert occ.text_raw == "SUGARTiMe"
    assert occ.text_norm == "SUGARTiMe"
    assert occ.model_id == "unknown"
    assert tr.segments[0].text == "SUGARTiMe"


def test_tracker_invalid_schema_missing_text():
    from lyra_ocr.track import Tracker

    tracker = Tracker(frame_width=1280, frame_height=720)
    invalid_frame = {
        "frame_id": 0,
        "timestamp_ms": 1000,
        "boxes": [
            {
                "polygon": [[10, 10], [50, 10], [50, 50], [10, 50]],
                "bbox": [10, 10, 50, 50],
                "confidence": 0.9,
            }
        ],
    }
    with pytest.raises(ValueError, match="missing 'text' or 'text_raw'"):
        tracker.update(invalid_frame)


def test_text_track_to_dict_includes_occurrences():
    from lyra_ocr.track import TextTrack, DetectionOccurrence

    tr = TextTrack(track_id=1, start_ms=1000, end_ms=2000, bbox=[10, 20, 30, 40])
    tr.occurrences = [
        DetectionOccurrence(
            frame_id=1,
            timestamp_ms=1000,
            polygon=[[10, 20], [30, 20], [30, 40], [10, 40]],
            bbox=[10, 20, 30, 40],
            text_raw="Hello",
            text_norm="Hello",
            confidence=0.95,
            model_id="PP-OCRv6_rec_small",
            script="latin",
        )
    ]
    d = tr.to_dict()
    assert "occurrences" in d
    assert len(d["occurrences"]) == 1
    assert d["occurrences"][0]["frame_id"] == 1
    assert d["occurrences"][0]["text_raw"] == "Hello"
    assert d["num_occurrences"] == 1
