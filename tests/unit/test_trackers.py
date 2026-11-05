"""Unit tests for tracking algorithms: Greedy baseline and two-stage ByteTrack-style tracker."""

import pytest
from lyra_ocr.trackers import (
    ByteTextTracker,
    GreedyTracker,
    get_tracker,
    known_tracker_names,
)

LINE_A = "We can be alone in the castle in the sky"
LINE_B = "The moon is a lantern hanging low"
LINE_C = "Carry me home through the dark water"
LYRIC_BOX = [300, 600, 800, 650]
WATERMARK_BOX = [40, 20, 320, 55]


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

    assert known_tracker_names() == ["bytetrack", "greedy"]


@pytest.mark.parametrize("tracker_cls", [GreedyTracker, ByteTextTracker])
def test_text_turnover_starts_a_new_track_in_the_same_position(tracker_cls):
    """Two lyric lines share one position; they must not become one track.

    Spatial association alone cannot tell them apart, so every strategy inherits
    the turnover split from BaseTracker.association().
    """
    tracker = tracker_cls(frame_width=1280, frame_height=720)
    texts = [LINE_A, LINE_A, LINE_B, LINE_B]
    for i, text in enumerate(texts):
        tracker.update(
            {"frame_id": i, "timestamp_ms": i * 1000, "boxes": [make_box(*LYRIC_BOX, text)]}
        )
    tracks = tracker.finalize(video_duration_ms=5000)

    assert len(tracks) == 2
    first, second = tracks
    assert {o.text_norm for o in first.occurrences} == {LINE_A}
    assert {o.text_norm for o in second.occurrences} == {LINE_B}
    assert (first.start_ms, first.end_ms) == (0, 1000)
    assert (second.start_ms, second.end_ms) == (2000, 3000)
    assert [t.n_text_changes for t in tracks] == [0, 0]


@pytest.mark.parametrize("tracker_cls", [GreedyTracker, ByteTextTracker])
def test_ocr_jitter_of_one_line_does_not_split_it(tracker_cls):
    """Character-level OCR noise stays above the turnover threshold (0.60)."""
    tracker = tracker_cls(frame_width=1280, frame_height=720)
    for i, text in enumerate([LINE_A, "We can be alone in the castle in the 5ky", LINE_A]):
        tracker.update(
            {"frame_id": i, "timestamp_ms": i * 1000, "boxes": [make_box(*LYRIC_BOX, text)]}
        )
    tracks = tracker.finalize(video_duration_ms=5000)

    assert len(tracks) == 1
    assert len(tracks[0].occurrences) == 3


def test_static_watermark_survives_while_lyric_slot_rotates():
    """One constant overlay plus three consecutive lines => four distinct tracks."""
    tracker = GreedyTracker(frame_width=1280, frame_height=720)
    lines = [LINE_A, LINE_B, LINE_C]
    frame_id = 0
    for line in lines:
        for _ in range(3):
            tracker.update(
                {
                    "frame_id": frame_id,
                    "timestamp_ms": frame_id * 1000,
                    "boxes": [
                        make_box(*WATERMARK_BOX, "@lyraofficial"),
                        make_box(*LYRIC_BOX, line),
                    ],
                }
            )
            frame_id += 1
    tracks = tracker.finalize(video_duration_ms=frame_id * 1000)

    assert len(tracks) == 4
    watermark = [t for t in tracks if t.occurrences[0].text_norm == "@lyraofficial"]
    assert len(watermark) == 1
    assert len(watermark[0].occurrences) == frame_id
    lyric_lines = sorted(
        (t.start_ms, t.occurrences[0].text_norm) for t in tracks if t not in watermark
    )
    assert [text for _, text in lyric_lines] == lines


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
