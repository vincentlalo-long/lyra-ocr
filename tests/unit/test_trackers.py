"""Unit tests for tracking algorithms: Greedy baseline and two-stage ByteTrack-style tracker."""

import pytest
from lyra_ocr.trackers import (
    ByteTextTracker,
    GreedyTracker,
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
    assert isinstance(get_tracker("byte"), ByteTextTracker)

    for removed in ("sort", "kalman", "dbscan", "st_dbscan", "tube"):
        with pytest.raises(ValueError, match="Unknown tracker"):
            get_tracker(removed)

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


def test_twostage_absorbs_low_confidence_ghost_that_greedy_turns_into_a_track():
    """Differential test against the greedy baseline on the failure mode two-stage exists for.

    A low-confidence box overlapping an already-matched lyric track (OCR ghost of
    the previous line, translucent watermark) must not become its own track and
    must not pollute the lyric track. Greedy has no confidence tiering: the box
    is unmatched, so it spawns a second track. This is the mechanism behind the
    golden-set observation of greedy emitting more tracks than bytetrack.
    """
    lyric_hi = make_box(300, 600, 800, 650, "We can be alone in the castle in the sky", conf=0.95)
    lyric_hi2 = make_box(300, 600, 800, 650, "We can be alone in the castle in the sky", conf=0.93)
    ghost_lo = make_box(302, 596, 798, 646, "CASTLE IN THE SKY", conf=0.40)

    def run(tracker):
        tracker.update({"frame_id": 0, "timestamp_ms": 1000, "boxes": [lyric_hi]})
        tracker.update({"frame_id": 1, "timestamp_ms": 1500, "boxes": [lyric_hi2, ghost_lo]})
        tracker.finalize(video_duration_ms=5000)
        return tracker.all_tracks

    bt_tracks = run(ByteTextTracker(frame_width=1280, frame_height=720))
    assert len(bt_tracks) == 1, "two-stage must drop the low-confidence ghost"
    assert len(bt_tracks[0].occurrences) == 2
    assert {o.text_norm for o in bt_tracks[0].occurrences} == {"We can be alone in the castle in the sky"}

    greedy_tracks = run(GreedyTracker(frame_width=1280, frame_height=720))
    assert len(greedy_tracks) == 2, "baseline behaviour being documented: ghost spawns a track"
