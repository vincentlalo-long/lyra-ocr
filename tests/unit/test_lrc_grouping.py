import pytest
from lyra_ocr.lyric.models import LyricCue
from lyra_ocr.lyric.grouping import dedupe_and_group, merge_duplicate_layers, consolidate_language_layers, _bbox_iou


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


def test_merge_duplicate_layers_collapses_shifted_band():
    # The same lyric stream captured twice: one band ~0.07 lower and shifted by
    # one line, as happens with a two-line lyric display split into tracks.
    top_texts = ["I was born this way", "Can't stop my fire", "I walk in like thunder", "But then I see you standing there", "Quiet smile, bright eyes"]
    bottom_texts = ["Can't stop my fire", "I walk in like thunder", "But then I see you standing there", "Quiet smile, bright eyes", "A line only the bottom shows"]
    top = [
        LyricCue(start_ms=1000 + i * 3000, end_ms=3800 + i * 3000, text=text, layer_id=9, lang="en", y_center=0.73)
        for i, text in enumerate(top_texts)
    ]
    bottom = [
        LyricCue(start_ms=1500 + i * 3000, end_ms=4300 + i * 3000, text=text, layer_id=10, lang="en", y_center=0.80)
        for i, text in enumerate(bottom_texts)
    ]

    result = merge_duplicate_layers(top + bottom)

    assert {c.layer_id for c in result} == {9}
    # Repeating lines from the duplicate band are dropped, not duplicated.
    texts = [c.text for c in result]
    assert texts.count("Can't stop my fire") == 1
    assert texts.count("I walk in like thunder") == 1
    # A line that only the duplicate band had is kept on the surviving layer.
    assert "A line only the bottom shows" in texts


def test_merge_duplicate_layers_keeps_distinct_bands():
    top = [LyricCue(start_ms=i * 2000, end_ms=i * 2000 + 1500, text=f"LINE {i}", layer_id=1, lang="en", y_center=0.3) for i in range(5)]
    bottom = [LyricCue(start_ms=i * 2000, end_ms=i * 2000 + 1500, text=f"OTHER {i}", layer_id=2, lang="en", y_center=0.75) for i in range(5)]

    result = merge_duplicate_layers(top + bottom)

    assert {c.layer_id for c in result} == {1, 2}
    assert len(result) == 10


def test_merge_duplicate_layers_allows_y_gap_between_rows():
    top = [LyricCue(start_ms=i * 2000, end_ms=i * 2000 + 1500, text=f"LINE {i}", layer_id=1, lang="en", y_center=0.72) for i in range(5)]
    stacked = [LyricCue(start_ms=i * 2000, end_ms=i * 2000 + 1500, text=f"LINE {i}", layer_id=2, lang="en", y_center=0.82) for i in range(5)]

    result = merge_duplicate_layers(top + stacked)

    assert len(result) == 5
    assert {c.layer_id for c in result} == {1}


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


def test_consolidate_language_layers_complementary():
    main = [
        LyricCue(start_ms=10000, end_ms=14000, text="Main verse line one", layer_id=1, lang="en"),
        LyricCue(start_ms=20000, end_ms=24000, text="Main verse line two", layer_id=1, lang="en"),
        LyricCue(start_ms=30000, end_ms=34000, text="Main verse line three", layer_id=1, lang="en"),
    ]
    adlib = [
        LyricCue(start_ms=15000, end_ms=17000, text="(let's go)", layer_id=2, lang="en"),
        LyricCue(start_ms=25000, end_ms=27000, text="(yeah yeah)", layer_id=2, lang="en"),
    ]
    result = consolidate_language_layers(main + adlib)
    assert len(result) == 5
    assert {c.layer_id for c in result} == {1}
    assert [c.text for c in result] == [
        "Main verse line one",
        "(let's go)",
        "Main verse line two",
        "(yeah yeah)",
        "Main verse line three",
    ]


def test_consolidate_language_layers_overlapping_noise_filtered():
    dom = [
        LyricCue(start_ms=i * 3000, end_ms=i * 3000 + 2500, text=f"Dominant line {i}", layer_id=1, lang="en")
        for i in range(12)
    ]
    noise = [
        LyricCue(start_ms=1000, end_ms=2500, text="Noise overlay", layer_id=2, lang="en"),
        LyricCue(start_ms=4000, end_ms=5500, text="Noise overlay", layer_id=2, lang="en"),
    ]
    result = consolidate_language_layers(dom + noise)
    assert len(result) == 12
    assert {c.layer_id for c in result} == {1}


def test_consolidate_keeps_companion_multiline_band():
    dom = [
        LyricCue(
            start_ms=i * 3000 + 1000,
            end_ms=i * 3000 + 3500,
            text=f"Dominant first line number {i} extra words here",
            layer_id=1,
            lang="en",
            x_center=0.50,
            y_center=0.81,
        )
        for i in range(15)
    ]
    companion = [
        LyricCue(
            start_ms=1000,
            end_ms=3500,
            text="and companion second line continues here with more words",
            layer_id=2,
            lang="en",
            x_center=0.50,
            y_center=0.88,
        ),
        LyricCue(
            start_ms=4000,
            end_ms=6500,
            text="another companion second line that must survive consolidation",
            layer_id=2,
            lang="en",
            x_center=0.50,
            y_center=0.88,
        ),
    ]
    result = consolidate_language_layers(dom + companion)
    texts = [c.text for c in result]
    assert {c.layer_id for c in result} == {1}
    assert "and companion second line continues here with more words" in texts
    assert "another companion second line that must survive consolidation" in texts
    assert len(result) == 17


def test_resolve_simultaneous_multiline_cues_merges_short_two_line_wrap():
    from lyra_ocr.lyric.grouping import resolve_simultaneous_multiline_cues

    cues = [
        LyricCue(start_ms=1000, end_ms=3200, text="In the castle", y_center=0.81, layer_id=1, lang="en", source_track_ids=[1]),
        LyricCue(start_ms=1000, end_ms=3200, text="of the sky", y_center=0.88, layer_id=1, lang="en", source_track_ids=[2]),
    ]
    resolved = resolve_simultaneous_multiline_cues(cues)
    assert len(resolved) == 1
    assert resolved[0].text == "In the castle of the sky"
    assert resolved[0].start_ms == 1000
    assert resolved[0].end_ms == 3200
    assert resolved[0].source_track_ids == [1, 2]


def test_resolve_simultaneous_multiline_cues_staggers_three_line_stanza():
    from lyra_ocr.lyric.grouping import resolve_simultaneous_multiline_cues

    cues = [
        LyricCue(start_ms=10000, end_ms=19000, text="First stanza line here", y_center=0.72, layer_id=1, lang="en", source_track_ids=[1]),
        LyricCue(start_ms=10000, end_ms=19000, text="Second stanza line here", y_center=0.80, layer_id=1, lang="en", source_track_ids=[2]),
        LyricCue(start_ms=10000, end_ms=19000, text="Third stanza line here", y_center=0.88, layer_id=1, lang="en", source_track_ids=[3]),
    ]
    resolved = resolve_simultaneous_multiline_cues(cues)
    assert len(resolved) == 3
    assert [c.text for c in resolved] == [
        "First stanza line here",
        "Second stanza line here",
        "Third stanza line here",
    ]
    starts = [c.start_ms for c in resolved]
    assert starts[0] == 10000
    assert starts[0] < starts[1] < starts[2]
    assert len(set(starts)) == 3
    assert all("multiline_staggered" in c.qa_flags for c in resolved)


def test_share_solitary_cross_layer_cues_inherits_english_bridge():
    from lyra_ocr.lyric.grouping import share_solitary_cross_layer_cues

    cues: list[LyricCue] = []
    # Layers 1 (ja) and 2 (ja-Latn) each have 8 cues, with a 5-second gap [30000..35000]
    timestamps = [
        (5000, 8000),
        (10000, 13000),
        (15000, 18000),
        (25000, 30000),
        (35000, 38000),
        (40000, 43000),
        (45000, 48000),
        (50000, 53000),
    ]
    for idx, (s, e) in enumerate(timestamps):
        cues.append(LyricCue(start_ms=s, end_ms=e, text=f"日本語ライン{idx}", layer_id=1, lang="ja"))
        cues.append(LyricCue(start_ms=s, end_ms=e, text=f"nihongo rain {idx}", layer_id=2, lang="ja-Latn"))
        cues.append(LyricCue(start_ms=s, end_ms=e, text=f"English line {idx}", layer_id=3, lang="en"))

    # Solitary shared cue only displayed on layer 3 (en) inside the 30000..35000ms gap
    cues.append(
        LyricCue(start_ms=31000, end_ms=34000, text="Fly to the sky", layer_id=3, lang="en")
    )

    result = share_solitary_cross_layer_cues(cues)

    ja_inherited = [c for c in result if c.layer_id == 1 and c.text == "Fly to the sky"]
    ro_inherited = [c for c in result if c.layer_id == 2 and c.text == "Fly to the sky"]
    assert len(ja_inherited) == 1
    assert len(ro_inherited) == 1
    assert ja_inherited[0].lang == "ja"
    assert ro_inherited[0].lang == "ja-Latn"
    assert "cross_layer_inherited" in ja_inherited[0].qa_flags
    assert "cross_layer_inherited" in ro_inherited[0].qa_flags
    assert ja_inherited[0].start_ms == 31000
    assert ja_inherited[0].end_ms == 34000




