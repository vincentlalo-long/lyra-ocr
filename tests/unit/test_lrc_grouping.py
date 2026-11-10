import pytest
from lyra_ocr.lyric.models import LyricCue
from lyra_ocr.lyric.grouping import dedupe_and_group, _bbox_iou


def test_bbox_iou():
    box1 = [0, 0, 100, 100]
    box2 = [0, 0, 100, 100]
    assert _bbox_iou(box1, box2) == 1.0

    box3 = [200, 200, 300, 300]
    assert _bbox_iou(box1, box3) == 0.0

    box4 = [50, 0, 150, 100]
    iou = _bbox_iou(box1, box4)
    assert 0.3 < iou < 0.4


def test_dedupe_duplicate_tracks_within_window():
    cue1 = LyricCue(
        start_ms=1000,
        end_ms=3000,
        text="I FEEL THE NIGHT EXPLODE",
        source_track_ids=[1],
        confidence=0.95,
        y_center=0.5,
        bbox=[100, 300, 500, 350],
    )
    cue2 = LyricCue(
        start_ms=1200,
        end_ms=3200,
        text="I FEEL THE NIGHT EXPLODE",
        source_track_ids=[2],
        confidence=0.98,
        y_center=0.52,
        bbox=[105, 305, 505, 355],
    )

    result = dedupe_and_group([cue1, cue2])
    assert len(result) == 1
    # Kept cue with higher confidence (cue2) and merged track ids
    assert result[0].source_track_ids == [1, 2]
    assert result[0].confidence == 0.98


def test_keep_distinct_lines_same_timestamp():
    # Two different lyrics at the same time at different vertical positions (multi-line)
    cue_top = LyricCue(
        start_ms=5000,
        end_ms=8000,
        text="LINE ONE TOP",
        y_center=0.2,
        source_track_ids=[10],
    )
    cue_bottom = LyricCue(
        start_ms=5000,
        end_ms=8000,
        text="LINE TWO BOTTOM",
        y_center=0.8,
        source_track_ids=[11],
    )

    result = dedupe_and_group([cue_bottom, cue_top])
    assert len(result) == 2
    # Sorted by y_center
    assert result[0].text == "LINE ONE TOP"
    assert result[1].text == "LINE TWO BOTTOM"


def test_clamp_negative_start_ms():
    cue = LyricCue(
        start_ms=-200,
        end_ms=1000,
        text="NEGATIVE START",
        source_track_ids=[1],
    )
    result = dedupe_and_group([cue])
    assert len(result) == 1
    assert result[0].start_ms == 0


def test_grouping_mixed_int_and_str_track_ids():
    cue1 = LyricCue(
        start_ms=1000,
        end_ms=2000,
        text="SAME TEXT",
        source_track_ids=[1],
        y_center=0.5,
    )
    cue2 = LyricCue(
        start_ms=1000,
        end_ms=2000,
        text="SAME TEXT",
        source_track_ids=["1"],
        y_center=0.5,
    )
    cue3 = LyricCue(
        start_ms=1000,
        end_ms=2000,
        text="OTHER TEXT",
        source_track_ids=[],
        y_center=0.5,
    )
    # Should not raise TypeError: '<' not supported between instances of 'int' and 'str'
    result = dedupe_and_group([cue1, cue2, cue3])
    assert len(result) == 2
