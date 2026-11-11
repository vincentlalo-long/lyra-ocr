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


def test_dedupe_same_line_one_second_apart():
    cue1 = LyricCue(
        start_ms=17000,
        end_ms=18500,
        text="My heart does something strange",
        source_track_ids=[1],
        confidence=0.90,
        y_center=0.9,
        bbox=[200, 620, 900, 660],
    )
    cue2 = LyricCue(
        start_ms=18000,
        end_ms=19500,
        text="My heart does something strange",
        source_track_ids=[2],
        confidence=0.95,
        y_center=0.9,
        bbox=[202, 621, 902, 661],
    )

    result = dedupe_and_group([cue1, cue2])
    assert len(result) == 1
    assert result[0].source_track_ids == [1, 2]


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


def test_dedupe_does_not_merge_different_layers():
    cue1 = LyricCue(
        start_ms=1000,
        end_ms=3000,
        text="SAME TEXT",
        layer_id=0,
        confidence=0.95,
        y_center=0.2,
    )
    cue2 = LyricCue(
        start_ms=1000,
        end_ms=3000,
        text="SAME TEXT",
        layer_id=1,
        confidence=0.95,
        y_center=0.8,
    )
    result = dedupe_and_group([cue1, cue2])
    assert len(result) == 2
    assert {r.layer_id for r in result} == {0, 1}


def test_dedupe_clamps_end_ms_to_next_cue_start_within_same_layer():
    cue1 = LyricCue(
        start_ms=1000,
        end_ms=4000,  # overshoots cue2
        text="LINE ONE",
        layer_id=0,
    )
    cue2 = LyricCue(
        start_ms=2500,
        end_ms=5000,
        text="LINE TWO",
        layer_id=0,
    )
    # Different layer cue starting at 2000ms should NOT clamp cue1
    cue_diff_layer = LyricCue(
        start_ms=2000,
        end_ms=6000,
        text="DIFF LAYER LINE",
        layer_id=1,
    )
    result = dedupe_and_group([cue1, cue2, cue_diff_layer])
    cues_layer0 = [c for c in result if c.layer_id == 0]
    assert cues_layer0[0].end_ms == 2500  # clamped to next start_ms in same layer
    assert cues_layer0[1].end_ms == 5000
    cues_layer1 = [c for c in result if c.layer_id == 1]
    assert cues_layer1[0].end_ms == 6000

