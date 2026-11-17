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
        y_center=0.8,
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


def test_resolve_simultaneous_multiline_cues_merges_words():
    from lyra_ocr.lyric.grouping import resolve_simultaneous_multiline_cues

    cues = [
        LyricCue(
            start_ms=1000,
            end_ms=3200,
            text="In the castle",
            y_center=0.81,
            layer_id=1,
            lang="en",
            source_track_ids=[1],
            words=[
                {"word": "In", "start_ms": 1000, "end_ms": 1400},
                {"word": "the", "start_ms": 1400, "end_ms": 1800},
                {"word": "castle", "start_ms": 1800, "end_ms": 2000},
            ],
        ),
        LyricCue(
            start_ms=1000,
            end_ms=3200,
            text="of the sky",
            y_center=0.88,
            layer_id=1,
            lang="en",
            source_track_ids=[2],
            words=[
                {"word": "of", "start_ms": 2000, "end_ms": 2400},
                {"word": "the", "start_ms": 2400, "end_ms": 2800},
                {"word": "sky", "start_ms": 2800, "end_ms": 3200},
            ],
        ),
    ]
    resolved = resolve_simultaneous_multiline_cues(cues)
    assert len(resolved) == 1
    assert resolved[0].text == "In the castle of the sky"
    assert len(resolved[0].words) == 6
    assert [w["word"] for w in resolved[0].words] == ["In", "the", "castle", "of", "the", "sky"]


def test_resolve_simultaneous_multiline_cues_dedupes_and_sorts_words():
    from lyra_ocr.lyric.grouping import resolve_simultaneous_multiline_cues

    cues = [
        LyricCue(
            start_ms=1000,
            end_ms=3200,
            text="In the castle",
            y_center=0.81,
            layer_id=1,
            lang="en",
            source_track_ids=[1],
            words=[
                {"word": "the", "start_ms": 1400, "end_ms": 1800},
                {"word": "In", "start_ms": 1000, "end_ms": 1400},
                {"word": "castle", "start_ms": 1800, "end_ms": 2200},
                {"word": "overlap", "start_ms": 2200, "end_ms": 2500},
            ],
        ),
        LyricCue(
            start_ms=1000,
            end_ms=3200,
            text="of the sky",
            y_center=0.88,
            layer_id=1,
            lang="en",
            source_track_ids=[2],
            words=[
                {"word": "overlap", "start_ms": 2200, "end_ms": 2500},
                {"word": "sky", "start_ms": 2800, "end_ms": 3200},
                {"word": "of", "start_ms": 2500, "end_ms": 2800},
            ],
        ),
    ]
    resolved = resolve_simultaneous_multiline_cues(cues)
    assert len(resolved) == 1
    assert len(resolved[0].words) == 6
    assert [w["word"] for w in resolved[0].words] == ["In", "the", "castle", "overlap", "of", "sky"]
    assert [w["start_ms"] for w in resolved[0].words] == [1000, 1400, 1800, 2200, 2500, 2800]



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


def test_resolve_simultaneous_multiline_cues_does_not_absorb_sequential_under_long_banner():
    from lyra_ocr.lyric.grouping import resolve_simultaneous_multiline_cues

    cues = [
        LyricCue(start_ms=1000, end_ms=20000, text="Long background banner", layer_id=1, lang="en", y_center=0.90),
        LyricCue(start_ms=1000, end_ms=4000, text="Line one", layer_id=1, lang="en", y_center=0.75),
        LyricCue(start_ms=5000, end_ms=9000, text="Line two", layer_id=1, lang="en", y_center=0.75),
        LyricCue(start_ms=10000, end_ms=14000, text="Line three", layer_id=1, lang="en", y_center=0.75),
    ]
    resolved = resolve_simultaneous_multiline_cues(cues)
    by_text = {c.text: (c.start_ms, c.end_ms) for c in resolved}
    assert by_text["Line one"] == (1000, 4000)
    assert by_text["Line two"] == (5000, 9000)
    assert by_text["Line three"] == (10000, 14000)
    assert by_text["Long background banner"] == (1000, 20000)


def test_dedupe_merges_long_held_line_split_across_frames():
    from lyra_ocr.audio.asr import ASRSegment, ASRTranscript, ASRWord

    cue1 = LyricCue(
        start_ms=10000,
        end_ms=13500,
        text="I will always love you",
        source_track_ids=[1],
        confidence=0.92,
        y_center=0.8,
        bbox=[100, 600, 800, 650],
        layer_id=1,
    )
    cue2 = LyricCue(
        start_ms=14000,
        end_ms=17500,
        text="I will always love you",
        source_track_ids=[2],
        confidence=0.95,
        y_center=0.8,
        bbox=[100, 600, 800, 650],
        layer_id=1,
    )
    # ASR only hears "love you" around 14000ms (held continuation), no "I will" onset
    transcript = ASRTranscript(
        language="en",
        segments=[
            ASRSegment(
                text="love you",
                start_ms=13800,
                end_ms=15000,
                words=[
                    ASRWord("love", 13800, 14200),
                    ASRWord("you", 14500, 15000),
                ],
            )
        ],
    )
    result = dedupe_and_group([cue1, cue2], asr_transcript=transcript)
    assert len(result) == 1
    assert result[0].start_ms == 10000
    assert result[0].end_ms == 17500
    assert sorted(result[0].source_track_ids) == [1, 2]


def test_dedupe_keeps_genuine_repeated_line_confirmed_by_asr():
    from lyra_ocr.audio.asr import ASRSegment, ASRTranscript, ASRWord

    cue1 = LyricCue(
        start_ms=10000,
        end_ms=12500,
        text="Let it go",
        source_track_ids=[1],
        confidence=0.95,
        y_center=0.8,
        bbox=[100, 600, 600, 650],
        layer_id=1,
    )
    cue2 = LyricCue(
        start_ms=13000,
        end_ms=15500,
        text="Let it go",
        source_track_ids=[2],
        confidence=0.95,
        y_center=0.8,
        bbox=[100, 600, 600, 650],
        layer_id=1,
    )
    transcript = ASRTranscript(
        language="en",
        segments=[
            ASRSegment(
                text="Let it go Let it go",
                start_ms=10000,
                end_ms=16000,
                words=[
                    ASRWord("Let", 10100, 10400),
                    ASRWord("it", 10400, 10600),
                    ASRWord("go", 10600, 11200),
                    ASRWord("Let", 13050, 13300),
                    ASRWord("it", 13300, 13500),
                    ASRWord("go", 13500, 14000),
                ],
            )
        ],
    )
    result = dedupe_and_group([cue1, cue2], asr_transcript=transcript)
    assert len(result) == 2
    assert result[0].start_ms == 10000
    assert result[1].start_ms == 13000


def test_dedupe_and_group_merges_dedupes_and_sorts_cue_words():
    from lyra_ocr.audio.asr import ASRSegment, ASRTranscript, ASRWord

    cue1 = LyricCue(
        start_ms=10000,
        end_ms=13500,
        text="I will always love you",
        source_track_ids=[1],
        confidence=0.90,
        y_center=0.8,
        bbox=[100, 600, 800, 650],
        layer_id=1,
        words=[
            {"word": "I", "start_ms": 10000, "end_ms": 10500},
            {"word": "will", "start_ms": 10500, "end_ms": 11000},
            {"word": "always", "start_ms": 11000, "end_ms": 12000},
            {"word": "love", "start_ms": 12000, "end_ms": 13000},  # duplicate in cue2
        ],
    )
    cue2 = LyricCue(
        start_ms=14000,
        end_ms=17500,
        text="I will always love you",
        source_track_ids=[2],
        confidence=0.95,
        y_center=0.8,
        bbox=[100, 600, 800, 650],
        layer_id=1,
        words=[
            {"word": "love", "start_ms": 12000, "end_ms": 13000},  # duplicate
            {"word": "you", "start_ms": 13500, "end_ms": 17500},
        ],
    )
    transcript = ASRTranscript(
        language="en",
        segments=[
            ASRSegment(
                text="love you",
                start_ms=13800,
                end_ms=15000,
                words=[
                    ASRWord("love", 13800, 14200),
                    ASRWord("you", 14500, 15000),
                ],
            )
        ],
    )
    res = dedupe_and_group([cue1, cue2], asr_transcript=transcript)
    assert len(res) == 1
    assert len(res[0].words) == 5
    assert [w["word"] for w in res[0].words] == ["I", "will", "always", "love", "you"]
    assert [w["start_ms"] for w in res[0].words] == [10000, 10500, 11000, 12000, 13500]


def test_consolidate_language_layers_separates_parallel_bilingual_misvoted_layers():
    # Case 1: Stale lang='en' on both Romaji and Spanish layers -> fresh detect_line_lang separates them
    ro_lines = [
        "Yoru ni magirete kaze wo kanjite",
        "Kimi no koe ga kikoeru kara",
        "Hikari no saki he hashiridasu yo",
        "Sono te wo tsuyoku nigirishimete",
        "Kawaranai yume wo mitsumete iru",
        "Ashita no sora he habataku tame ni",
        "Namida wo fuite waratte misete",
        "Bokura no mirai ga kagayaku made",
    ]
    es_lines = [
        "Aunque llore ya nada volvera como antes",
        "El futuro contigo siempre sera mejor",
        "Solo quiero olvidar este gran dolor",
        "Nunca voy a olvidar tu dulce voz",
        "Aunque pase el tiempo seguire aqui",
        "Buscando la luz en este mundo gris",
        "Siempre estaras dentro de mi corazon",
        "Hasta que podamos volver a vernos hoy",
    ]
    cues_case1 = [
        LyricCue(
            start_ms=i * 3000 + 1000,
            end_ms=i * 3000 + 3500,
            text=txt,
            layer_id=1,
            lang="en",
            y_center=0.85,
        )
        for i, txt in enumerate(ro_lines)
    ] + [
        LyricCue(
            start_ms=i * 3000 + 1000,
            end_ms=i * 3000 + 3500,
            text=txt,
            layer_id=2,
            lang="en",
            y_center=0.15,
        )
        for i, txt in enumerate(es_lines)
    ]
    res1 = consolidate_language_layers(cues_case1)
    assert {c.layer_id for c in res1} == {1, 2}
    assert {c.lang for c in res1 if c.layer_id == 1} == {"ja-Latn"}
    assert {c.lang for c in res1 if c.layer_id == 2} == {"es"}

    # Case 2: Borderline Romaji layer (both layers vote 'en' under detect_line_lang) vs English layer
    borderline_ro = [
        f"kaze no naka de we rise {i}" for i in range(8)
    ]
    en_lines = [
        f"Walking through the dark shadows tonight {i}" for i in range(8)
    ]
    cues_case2 = [
        LyricCue(
            start_ms=i * 3000 + 1000,
            end_ms=i * 3000 + 3500,
            text=txt,
            layer_id=1,
            lang="en",
            y_center=0.85,
        )
        for i, txt in enumerate(borderline_ro)
    ] + [
        LyricCue(
            start_ms=i * 3000 + 1000,
            end_ms=i * 3000 + 3500,
            text=txt,
            layer_id=2,
            lang="en",
            y_center=0.15,
        )
        for i, txt in enumerate(en_lines)
    ]
    res2 = consolidate_language_layers(cues_case2)
    assert {c.layer_id for c in res2} == {1, 2}
    assert {c.lang for c in res2 if c.layer_id == 1} == {"ja-Latn"}
    assert {c.lang for c in res2 if c.layer_id == 2} == {"en"}

    # Case 3: Borderline unaccented Spanish layer (both vote 'en' under detect_line_lang) vs English layer
    borderline_es = [
        f"el mundo fades away forever {i}" for i in range(8)
    ]
    cues_case3 = [
        LyricCue(
            start_ms=i * 3000 + 1000,
            end_ms=i * 3000 + 3500,
            text=txt,
            layer_id=1,
            lang="en",
            y_center=0.15,
        )
        for i, txt in enumerate(borderline_es)
    ] + [
        LyricCue(
            start_ms=i * 3000 + 1000,
            end_ms=i * 3000 + 3500,
            text=txt,
            layer_id=2,
            lang="en",
            y_center=0.85,
        )
        for i, txt in enumerate(en_lines)
    ]
    res3 = consolidate_language_layers(cues_case3)
    assert {c.layer_id for c in res3} == {1, 2}
    assert {c.lang for c in res3 if c.layer_id == 1} == {"es"}
    assert {c.lang for c in res3 if c.layer_id == 2} == {"en"}


def test_resolve_simultaneous_multiline_cues_collapses_same_y_flicker():
    from lyra_ocr.lyric.grouping import resolve_simultaneous_multiline_cues

    cues = [
        LyricCue(
            start_ms=1000,
            end_ms=1600,
            text="Dai",
            y_center=0.85,
            confidence=0.99,
            layer_id=1,
            lang="ja-Latn",
            source_track_ids=[1],
            source_segment_ids=[0],
        ),
        LyricCue(
            start_ms=1000,
            end_ms=3200,
            text="Daitan futeki",
            y_center=0.86,
            confidence=0.88,
            layer_id=1,
            lang="ja-Latn",
            source_track_ids=[2],
            source_segment_ids=[1],
            words=[{"word": "Daitan", "start_ms": 1000, "end_ms": 2000}],
        ),
        LyricCue(
            start_ms=1100,
            end_ms=4000,
            text="Daitan futeki ni haikara",
            y_center=0.855,
            confidence=0.95,
            layer_id=1,
            lang="ja-Latn",
            source_track_ids=[3],
            source_segment_ids=[2],
            words=[{"word": "haikara", "start_ms": 2500, "end_ms": 4000}],
        ),
    ]
    resolved = resolve_simultaneous_multiline_cues(cues)
    assert len(resolved) == 1
    assert resolved[0].text == "Daitan futeki ni haikara"
    assert resolved[0].start_ms == 1000
    assert resolved[0].end_ms == 4000
    assert resolved[0].source_track_ids == [1, 2, 3]
    assert resolved[0].source_segment_ids == [0, 1, 2]
    assert "multiline_staggered" not in resolved[0].qa_flags
    assert [w["word"] for w in resolved[0].words] == ["Daitan", "haikara"]


def test_dedupe_and_group_merges_same_y_fade_transition():
    # Fade-in short partial read followed within 800ms by full stable line at same y_center
    fade_in = LyricCue(
        start_ms=1000,
        end_ms=1800,  # 800ms <= 1400ms
        text="Senbonzakura yoru",
        confidence=0.96,
        y_center=0.85,
        bbox=[100, 600, 500, 640],
        layer_id=1,
        source_track_ids=[1],
    )
    stable = LyricCue(
        start_ms=2200,  # gap_ms = 400 <= 800
        end_ms=6000,
        text="Senbonzakura yoru ni magire",
        confidence=0.90,
        y_center=0.86,  # |Δy| = 0.01 < 0.032
        bbox=[100, 600, 700, 640],
        layer_id=1,
        source_track_ids=[2],
    )
    res = dedupe_and_group([fade_in, stable])
    assert len(res) == 1
    assert res[0].text == "Senbonzakura yoru ni magire"
    assert res[0].start_ms == 1000
    assert res[0].end_ms == 6000
    assert res[0].source_track_ids == [1, 2]

    # Fade-out short read after stable line
    stable2 = LyricCue(
        start_ms=10000,
        end_ms=13800,
        text="Kimi no koe mo todokanai yo",
        confidence=0.92,
        y_center=0.85,
        layer_id=1,
        source_track_ids=[3],
    )
    fade_out = LyricCue(
        start_ms=14300,  # gap_ms = 500 <= 800
        end_ms=15200,  # 900ms <= 1400ms
        text="Kimi no koe mo todok",
        confidence=0.95,
        y_center=0.85,
        layer_id=1,
        source_track_ids=[4],
    )
    res2 = dedupe_and_group([stable2, fade_out])
    assert len(res2) == 1
    assert res2[0].text == "Kimi no koe mo todokanai yo"
    assert res2[0].start_ms == 10000
    assert res2[0].end_ms == 15200


def test_dedupe_and_group_fade_transition_preserves_distinct_short_lines_and_respects_asr():
    from lyra_ocr.audio.asr import ASRSegment, ASRTranscript, ASRWord

    # Distinct short consecutive lines ("I see you" -> "I love you", sim=0.737, min_dur<=1400, gap=400)
    # must stay separate WITHOUT ASR and WITH ASR.
    c1 = LyricCue(
        start_ms=1000,
        end_ms=2200,
        text="I see you",
        confidence=0.92,
        y_center=0.85,
        layer_id=1,
        lang="en",
        source_track_ids=[1],
    )
    c2 = LyricCue(
        start_ms=2600,
        end_ms=3800,
        text="I love you",
        confidence=0.94,
        y_center=0.85,
        layer_id=1,
        lang="en",
        source_track_ids=[2],
    )
    res_no_asr = dedupe_and_group([c1, c2])
    assert [c.text for c in res_no_asr] == ["I see you", "I love you"]

    asr = ASRTranscript(
        language="en",
        segments=[
            ASRSegment(
                text="I see you I love you",
                start_ms=1000,
                end_ms=3900,
                words=[
                    ASRWord("I", 1020, 1180),
                    ASRWord("see", 1200, 1500),
                    ASRWord("you", 1520, 1900),
                    ASRWord("I", 2600, 2750),
                    ASRWord("love", 2760, 3100),
                    ASRWord("you", 3120, 3600),
                ],
            )
        ],
    )
    c1_asr = LyricCue(
        start_ms=1000,
        end_ms=2200,
        text="I see you",
        confidence=0.92,
        y_center=0.85,
        layer_id=1,
        lang="en",
        source_track_ids=[1],
    )
    c2_asr = LyricCue(
        start_ms=2600,
        end_ms=3800,
        text="I love you",
        confidence=0.94,
        y_center=0.85,
        layer_id=1,
        lang="en",
        source_track_ids=[2],
    )
    res_with_asr = dedupe_and_group([c1_asr, c2_asr], asr_transcript=asr)
    assert [c.text for c in res_with_asr] == ["I see you", "I love you"]

    # Even when sim >= 0.75 ("I need you" -> "I feel you", sim=0.80), ASR new onset must block fade merge
    asr_need_feel = ASRTranscript(
        language="en",
        segments=[
            ASRSegment(
                text="I need you I feel you",
                start_ms=1000,
                end_ms=3900,
                words=[
                    ASRWord("I", 1020, 1180),
                    ASRWord("need", 1200, 1500),
                    ASRWord("you", 1520, 1900),
                    ASRWord("I", 2600, 2750),
                    ASRWord("feel", 2760, 3100),
                    ASRWord("you", 3120, 3600),
                ],
            )
        ],
    )
    res_need_feel = dedupe_and_group(
        [
            LyricCue(start_ms=1000, end_ms=2200, text="I need you", confidence=0.92, y_center=0.85, layer_id=1, lang="en", source_track_ids=[1]),
            LyricCue(start_ms=2600, end_ms=3800, text="I feel you", confidence=0.94, y_center=0.85, layer_id=1, lang="en", source_track_ids=[2]),
        ],
        asr_transcript=asr_need_feel,
    )
    assert [c.text for c in res_need_feel] == ["I need you", "I feel you"]

    # True CJK/Latin fade-in/fade-out OCR flickers still collapse to 1 cue
    cjk_fade = dedupe_and_group([
        LyricCue(start_ms=1000, end_ms=1800, text="此処は宴の", confidence=0.85, y_center=0.85, layer_id=1, lang="ja", source_track_ids=[10]),
        LyricCue(start_ms=2100, end_ms=5500, text="此処は宴 鋼の檻", confidence=0.93, y_center=0.85, layer_id=1, lang="ja", source_track_ids=[11]),
    ])
    assert len(cjk_fade) == 1
    assert cjk_fade[0].text == "此処は宴 鋼の檻"

    cjk_variant_fade = dedupe_and_group([
        LyricCue(start_ms=6000, end_ms=6900, text="少年少女戦国無双", confidence=0.88, y_center=0.85, layer_id=1, lang="ja", source_track_ids=[12]),
        LyricCue(start_ms=7200, end_ms=10500, text="少年少女戰国無双", confidence=0.94, y_center=0.85, layer_id=1, lang="ja", source_track_ids=[13]),
    ])
    assert len(cjk_variant_fade) == 1

    # Substring containment / >=4-char prefix flicker with sim in [0.62, 0.75) still collapses
    suffix_fade = dedupe_and_group([
        LyricCue(start_ms=11000, end_ms=11800, text="鋼の檻だよ", confidence=0.85, y_center=0.85, layer_id=1, lang="ja", source_track_ids=[14]),
        LyricCue(start_ms=12100, end_ms=15500, text="此処は宴 鋼の檻だよ", confidence=0.94, y_center=0.85, layer_id=1, lang="ja", source_track_ids=[15]),
    ])
    assert len(suffix_fade) == 1
    assert suffix_fade[0].text == "此処は宴 鋼の檻だよ"

    # When ASR only has the first cue's onset (no new onset at second cue), fade-in still collapses
    asr_single_line = ASRTranscript(
        language="ja",
        segments=[
            ASRSegment(
                text="此処は宴鋼の檻",
                start_ms=1000,
                end_ms=4500,
                words=[ASRWord("此処", 1020, 1400), ASRWord("は", 1400, 1600)],
            )
        ],
    )
    cjk_fade_asr = dedupe_and_group(
        [
            LyricCue(start_ms=1000, end_ms=1500, text="此処は宴の", confidence=0.85, y_center=0.85, layer_id=1, lang="ja", source_track_ids=[16]),
            LyricCue(start_ms=1700, end_ms=5000, text="此処は宴 鋼の檻", confidence=0.93, y_center=0.85, layer_id=1, lang="ja", source_track_ids=[17]),
        ],
        asr_transcript=asr_single_line,
    )
    assert len(cjk_fade_asr) == 1
    assert cjk_fade_asr[0].text == "此処は宴 鋼の檻"



def test_share_solitary_cross_layer_cues_strict_global_check():
    from lyra_ocr.lyric.grouping import share_solitary_cross_layer_cues

    cues: list[LyricCue] = []
    # 3 major layers: ja (1), es (2), en (3)
    timestamps = [
        (5000, 8000),
        (10000, 13000),
        (15000, 18000),
        (25000, 29000),
        (35000, 38000),
        (40000, 43000),
        (45000, 48000),
        (50000, 53000),
    ]
    for idx, (s, e) in enumerate(timestamps):
        cues.append(LyricCue(start_ms=s, end_ms=e, text=f"日本語ライン{idx}", layer_id=1, lang="ja"))
        cues.append(LyricCue(start_ms=s, end_ms=e, text=f"Linea en espanol {idx}", layer_id=2, lang="es"))
        cues.append(LyricCue(start_ms=s, end_ms=e, text=f"English line {idx}", layer_id=3, lang="en"))

    # 1. Offset English cue in layer 3 that overlaps layer 1 (ja) within +-800ms (even though es has a gap 19000..24000)
    cues.append(LyricCue(start_ms=20000, end_ms=22500, text="Offset translation line", layer_id=3, lang="en"))
    cues.append(LyricCue(start_ms=21500, end_ms=24500, text="日本語の重複ライン", layer_id=1, lang="ja"))

    # 2. Non-English (Spanish) solitary cue in layer 2 during global silence -> must NOT be shared
    cues.append(LyricCue(start_ms=55000, end_ms=57500, text="Aunque llore siempre", layer_id=2, lang="es"))

    # 3. Genuine 1.6s English hook during global silence [29000..35000] (no layer overlaps [31000-800, 32600+800] = [30200, 33400])
    cues.append(LyricCue(start_ms=31000, end_ms=32600, text="Hacking to the Gate", layer_id=3, lang="en"))

    # Extend spans past 58000 so 55000..57500 is inside span_min..span_max
    cues.append(LyricCue(start_ms=59000, end_ms=61000, text="日本語ラスト", layer_id=1, lang="ja"))
    cues.append(LyricCue(start_ms=59000, end_ms=61000, text="Ultima linea", layer_id=2, lang="es"))
    cues.append(LyricCue(start_ms=59000, end_ms=61000, text="Last English line", layer_id=3, lang="en"))

    res = share_solitary_cross_layer_cues(cues)

    # Offset English cue was NOT copied to es
    assert not any(c.layer_id == 2 and c.text == "Offset translation line" for c in res)
    # Spanish cue was NOT copied to ja or en
    assert not any(c.layer_id in (1, 3) and c.text == "Aunque llore siempre" for c in res)
    # 1.6s English hook WAS inherited into both ja (1) and es (2)
    assert any(c.layer_id == 1 and c.text == "Hacking to the Gate" and "cross_layer_inherited" in c.qa_flags for c in res)
    assert any(c.layer_id == 2 and c.text == "Hacking to the Gate" and "cross_layer_inherited" in c.qa_flags for c in res)


def test_layer_consolidation_exports_match_grouping_reexports():
    from lyra_ocr.lyric import grouping, layer_consolidation

    assert grouping.merge_duplicate_layers is layer_consolidation.merge_duplicate_layers
    assert grouping.consolidate_language_layers is layer_consolidation.consolidate_language_layers
    assert grouping.share_solitary_cross_layer_cues is layer_consolidation.share_solitary_cross_layer_cues
    assert callable(grouping.dedupe_and_group)
    assert callable(grouping.resolve_simultaneous_multiline_cues)


def test_grouping_thresholds_govern_dedupe_and_layer_consolidation():
    from lyra_ocr.config import GroupingThresholds
    from lyra_ocr.lyric.grouping import resolve_simultaneous_multiline_cues
    from lyra_ocr.lyric.layer_consolidation import (
        consolidate_language_layers,
        merge_duplicate_layers,
        share_solitary_cross_layer_cues,
    )

    # 1. dedupe_text_similarity governs dedupe_and_group merging
    cue_a = LyricCue(start_ms=1000, end_ms=3000, text="Walk through the night", y_center=0.82, layer_id=1, confidence=0.9)
    cue_b = LyricCue(start_ms=3050, end_ms=5050, text="Walk toward the night", y_center=0.86, layer_id=1, confidence=0.95)
    assert len(dedupe_and_group([cue_a, cue_b])) == 2
    merged = dedupe_and_group(
        [
            LyricCue(start_ms=1000, end_ms=3000, text="Walk through the night", y_center=0.82, layer_id=1, confidence=0.9),
            LyricCue(start_ms=3050, end_ms=5050, text="Walk toward the night", y_center=0.86, layer_id=1, confidence=0.95),
        ],
        thresholds=GroupingThresholds(dedupe_text_similarity=0.70),
    )
    assert len(merged) == 1

    # 2. duplicate_max_y_gap governs merge_duplicate_layers
    band1 = [LyricCue(start_ms=i * 2000, end_ms=i * 2000 + 1500, text=f"Line number {i}", layer_id=1, lang="en", y_center=0.75) for i in range(4)]
    band2 = [LyricCue(start_ms=i * 2000, end_ms=i * 2000 + 1500, text=f"Line number {i}", layer_id=2, lang="en", y_center=0.82) for i in range(4)]
    assert {c.layer_id for c in merge_duplicate_layers(band1 + band2)} == {1}
    kept_both = merge_duplicate_layers(
        band1 + band2,
        thresholds=GroupingThresholds(duplicate_max_y_gap=0.02),
    )
    assert {c.layer_id for c in kept_both} == {1, 2}

    # 3. short_wrap_max_dur_ms governs resolve_simultaneous_multiline_cues wrap merging
    wrap_pair = [
        LyricCue(start_ms=1000, end_ms=3200, text="In the castle", y_center=0.81, layer_id=1, lang="en"),
        LyricCue(start_ms=1000, end_ms=3200, text="of the sky", y_center=0.88, layer_id=1, lang="en"),
    ]
    assert len(resolve_simultaneous_multiline_cues(wrap_pair)) == 1
    staggered = resolve_simultaneous_multiline_cues(
        [
            LyricCue(start_ms=1000, end_ms=3200, text="In the castle", y_center=0.81, layer_id=1, lang="en"),
            LyricCue(start_ms=1000, end_ms=3200, text="of the sky", y_center=0.88, layer_id=1, lang="en"),
        ],
        thresholds=GroupingThresholds(short_wrap_max_dur_ms=1000),
    )
    assert len(staggered) == 2

    # 4. consolidate_overlap_threshold governs complementary vs overlapping merge in consolidate_language_layers
    dom_cues = [LyricCue(start_ms=i * 4000, end_ms=i * 4000 + 2000, text=f"Primary line {i}", layer_id=1, lang="en", y_center=0.80) for i in range(10)]
    sec_cues = [
        LyricCue(start_ms=500, end_ms=1800, text="Unique overlay alpha", layer_id=2, lang="en", y_center=0.20),
        LyricCue(start_ms=22500, end_ms=23500, text="Unique bridge beta", layer_id=2, lang="en", y_center=0.20),
    ]
    # 1 of 2 sec_cues overlaps (overlap_ratio=0.5): dropped under default 0.25, kept under 0.60
    assert len(consolidate_language_layers(dom_cues + sec_cues)) == 10
    kept_comp = consolidate_language_layers(
        [LyricCue(start_ms=i * 4000, end_ms=i * 4000 + 2000, text=f"Primary line {i}", layer_id=1, lang="en", y_center=0.80) for i in range(10)]
        + [
            LyricCue(start_ms=500, end_ms=1800, text="Unique overlay alpha", layer_id=2, lang="en", y_center=0.20),
            LyricCue(start_ms=22500, end_ms=23500, text="Unique bridge beta", layer_id=2, lang="en", y_center=0.20),
        ],
        thresholds=GroupingThresholds(consolidate_overlap_threshold=0.60),
    )
    assert len(kept_comp) == 12

    # 5. share_min_cue_dur_ms governs share_solitary_cross_layer_cues
    base = []
    for idx, (s, e) in enumerate([(5000, 8000), (10000, 13000), (15000, 18000), (25000, 29000), (35000, 38000), (40000, 43000), (45000, 48000), (50000, 53000)]):
        base.append(LyricCue(start_ms=s, end_ms=e, text=f"日本語ライン{idx}", layer_id=1, lang="ja"))
        base.append(LyricCue(start_ms=s, end_ms=e, text=f"English line {idx}", layer_id=2, lang="en"))
    base.append(LyricCue(start_ms=31000, end_ms=33500, text="Fly high tonight", layer_id=2, lang="en"))
    assert any(c.layer_id == 1 and c.text == "Fly high tonight" for c in share_solitary_cross_layer_cues(base))
    assert not any(
        c.layer_id == 1 and c.text == "Fly high tonight"
        for c in share_solitary_cross_layer_cues(base, thresholds=GroupingThresholds(share_min_cue_dur_ms=4000))
    )




