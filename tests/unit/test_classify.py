"""Unit tests for rule-based classification (classify.py)."""

import pytest
from lyra_ocr.track import TextTrack, TextSegment, DetectionOccurrence


def create_mock_track(
    track_id: int,
    start_ms: int,
    end_ms: int,
    texts: list[str],
    bbox: list[int],
    video_duration_ms: int = 100000,
    spatial_variance: float = 0.0,
) -> TextTrack:
    occurrences = []
    # Build occurrences evenly across start_ms and end_ms
    step = max(500, (end_ms - start_ms) // max(1, len(texts)))
    t = start_ms
    for txt in texts:
        occurrences.append(
            DetectionOccurrence(
                frame_id=len(occurrences),
                timestamp_ms=t,
                polygon=[[bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]]],
                bbox=bbox,
                text_raw=txt,
                text_norm=txt,
                confidence=0.98,
                model_id="PP-OCRv6_rec_small",
                script="latin",
            )
        )
        t += step

    track = TextTrack(track_id=track_id)
    track.occurrences = occurrences
    track.start_ms = start_ms
    track.end_ms = end_ms
    track.bbox = bbox

    # Segments
    segments = []
    for i, txt in enumerate(texts):
        segments.append(
            TextSegment(
                segment_id=i,
                start_ms=start_ms + i * step,
                end_ms=start_ms + (i + 1) * step,
                text=txt,
                confidence=0.98,
                num_occurrences=1,
            )
        )
    track.segments = segments
    track.n_text_changes = max(0, len(texts) - 1)
    track.features = {
        "lifetime_ratio": (end_ms - start_ms) / video_duration_ms,
        "start_time_ratio": start_ms / video_duration_ms,
        "n_text_changes": track.n_text_changes,
        "spatial_variance": spatial_variance,
        "is_corner": bbox[0] < 100 and bbox[1] < 100,
        "aspect_ratio": (bbox[2] - bbox[0]) / max(1, bbox[3] - bbox[1]),
        "mean_centroid": [
            ((bbox[0] + bbox[2]) / 2.0) / 1280.0,
            ((bbox[1] + bbox[3]) / 2.0) / 720.0,
        ],
        "dominant_text": texts[0] if texts else "",
    }
    return track


def test_classify_persistent_channel_watermark():
    from lyra_ocr.classify import classify_track

    # Watermark spans 80% of video, text never changes
    track = create_mock_track(
        track_id=1,
        start_ms=0,
        end_ms=80000,
        texts=["CLARX & HARDDOPE"],
        bbox=[30, 30, 200, 60],
        video_duration_ms=100000,
    )
    result = classify_track(track, metadata={"channel": "Clarx & Harddope", "title": "Castle"})

    assert result.label == "non_lyric"
    assert result.decision == "reject"
    assert "Persistent" in result.reason or "channel" in result.reason.lower()


def test_classify_intro_title_card():
    from lyra_ocr.classify import classify_track

    # Title card appears only from 0 to 5 seconds
    track = create_mock_track(
        track_id=2,
        start_ms=0,
        end_ms=5000,
        texts=["CASTLE"],
        bbox=[500, 200, 780, 260],
        video_duration_ms=100000,
    )
    result = classify_track(track, metadata={"channel": "Clarx & Harddope", "title": "Castle"})

    assert result.label == "non_lyric"
    assert result.decision == "reject"
    assert "title" in result.reason.lower() or "intro" in result.reason.lower()


def test_classify_opening_lyric_line_is_not_an_intro_card():
    """A turnover-split track can start at t=0; position alone must not reject it.

    The intro-card rule used to fire on any short static text at the very start,
    which was safe while one merged track held the whole lyric stream. Now that
    each line is its own track, the first sung cue of a video looks identical to
    a title card unless the text itself is title-like.
    """
    from lyra_ocr.classify import classify_track

    track = create_mock_track(
        track_id=30,
        start_ms=0,
        end_ms=4000,
        texts=["We can be alone in the castle in the sky"],
        bbox=[300, 600, 980, 650],
        video_duration_ms=100000,
    )
    result = classify_track(track, metadata={"channel": "Clarx & Harddope", "title": "Castle"})

    assert result.decision != "reject"
    assert "intro_bounded_card" not in result.qa_flags


def test_classify_all_caps_intro_card_is_still_rejected():
    from lyra_ocr.classify import classify_track

    track = create_mock_track(
        track_id=31,
        start_ms=0,
        end_ms=4000,
        texts=["CASTLE IN THE SKY"],
        bbox=[300, 600, 980, 650],
        video_duration_ms=100000,
    )
    result = classify_track(track, metadata={})

    assert result.label == "non_lyric"
    assert result.decision == "reject"
    assert "intro_bounded_card" in result.qa_flags


def test_classify_title_match_inside_intro_still_rejected():
    from lyra_ocr.classify import classify_track

    # Regression guard: a title card inside the first 25% stays an auto-reject.
    track = create_mock_track(
        track_id=20,
        start_ms=10000,
        end_ms=14000,
        texts=["CASTLE"],
        bbox=[500, 200, 780, 260],
        video_duration_ms=100000,
    )
    result = classify_track(track, metadata={"channel": "Clarx & Harddope", "title": "Castle"})

    assert result.label == "non_lyric"
    assert result.decision == "reject"
    assert "metadata_title_match" in result.qa_flags


def test_classify_title_match_outside_intro_defers_to_review():
    from lyra_ocr.classify import classify_track

    # Repeated hook equal to the song title, mid-video and in reading zone:
    # not an intro card, so it must be classified as lyric under review.
    track = create_mock_track(
        track_id=21,
        start_ms=55000,
        end_ms=60000,
        texts=["CASTLE"],
        bbox=[300, 600, 980, 650],
        video_duration_ms=100000,
    )
    result = classify_track(track, metadata={"channel": "Clarx & Harddope", "title": "Castle"})

    assert result.label == "lyric"
    assert result.decision == "review"
    assert "metadata_title_match" in result.qa_flags
    assert "title_match_outside_intro" in result.qa_flags
    assert "potential_chorus" in result.qa_flags


def test_classify_title_match_outside_intro_outside_reading_zone():
    from lyra_ocr.classify import classify_track

    track = create_mock_track(
        track_id=22,
        start_ms=55000,
        end_ms=60000,
        texts=["CASTLE"],
        bbox=[10, 10, 150, 40],
        video_duration_ms=100000,
    )
    result = classify_track(track, metadata={"channel": "Clarx & Harddope", "title": "Castle"})

    assert result.label == "non_lyric"
    assert result.decision == "review"
    assert "metadata_title_match" in result.qa_flags
    assert "title_match_outside_intro" in result.qa_flags
    assert "potential_chorus" not in result.qa_flags


def test_classify_recurring_lyric_stream():
    from lyra_ocr.classify import classify_track

    # Lyric track changes text multiple times in same stable lower region
    track = create_mock_track(
        track_id=3,
        start_ms=20000,
        end_ms=60000,
        texts=[
            "We can be alone in the castle in the sky",
            "No one knows where we are",
            "Shadows dancing through the dark",
        ],
        bbox=[300, 600, 980, 650],
        video_duration_ms=100000,
        spatial_variance=0.001,
    )
    result = classify_track(track, metadata={"channel": "Clarx & Harddope", "title": "Castle"})

    assert result.label == "lyric"
    assert result.decision == "accept"
    assert "recurring text changes" in result.reason.lower() or "lyric" in result.reason.lower()


def test_classify_recurring_changes_in_intro_are_not_auto_accepted():
    from lyra_ocr.classify import classify_track

    # Credit carousel: 3 distinct texts inside the first 5 seconds of a 100s video.
    track = create_mock_track(
        track_id=10,
        start_ms=0,
        end_ms=5000,
        texts=["CASTLE", "CLARX & HARDDOPE", "OFFICIAL VIDEO"],
        bbox=[300, 200, 980, 250],
        video_duration_ms=100000,
    )
    result = classify_track(track, metadata={})

    assert result.label == "non_lyric"
    assert result.decision == "review"
    assert "only_intro_outro" in result.qa_flags


def test_classify_all_caps_promo_is_not_auto_accepted_as_lyric():
    from lyra_ocr.classify import classify_track

    track = create_mock_track(
        track_id=11,
        start_ms=40000,
        end_ms=45000,
        texts=["NEW VIDEO EVERY WEEK"],
        bbox=[300, 300, 900, 340],
        video_duration_ms=100000,
    )
    result = classify_track(track, metadata={})

    assert result.label != "lyric"
    assert result.decision != "accept"


def test_classify_sentence_case_single_cue_is_lyric():
    from lyra_ocr.classify import classify_track

    track = create_mock_track(
        track_id=12,
        start_ms=20000,
        end_ms=24500,
        texts=["We can be alone in the castle in the sky"],
        bbox=[300, 600, 980, 650],
        video_duration_ms=100000,
    )
    result = classify_track(track, metadata={})

    assert result.label == "lyric"
    assert result.decision == "accept"


def test_classify_isolated_single_frame_text_is_scene_text():
    from lyra_ocr.classify import classify_track

    track = create_mock_track(
        track_id=13,
        start_ms=20000,
        end_ms=20000,
        texts=["4ILCAR"],
        bbox=[1025, 304, 1131, 391],
        video_duration_ms=200000,
    )
    result = classify_track(track, metadata={})

    assert result.label == "non_lyric"
    assert result.decision == "reject"
    assert "single_frame_isolated" in result.qa_flags


def test_classify_supplementary_subscribe_tag():
    from lyra_ocr.classify import classify_track

    track = create_mock_track(
        track_id=4,
        start_ms=10000,
        end_ms=15000,
        texts=["SUBSCRIBE FOR MORE"],
        bbox=[500, 650, 750, 690],
        video_duration_ms=100000,
    )
    result = classify_track(track)

    assert result.label == "non_lyric"
    assert result.decision == "reject"
    assert "subscribe" in result.reason.lower() or "keyword" in result.reason.lower()


def test_classify_erratic_scene_text():
    from lyra_ocr.classify import classify_track

    track = create_mock_track(
        track_id=5,
        start_ms=30000,
        end_ms=35000,
        texts=["POSTER TEXT"],
        bbox=[200, 300, 350, 350],
        video_duration_ms=100000,
        spatial_variance=0.15,  # High variance (moving around frame)
    )
    result = classify_track(track)

    assert result.label == "non_lyric"
    assert result.decision == "reject"
    assert "variance" in result.reason.lower() or "drift" in result.reason.lower()


def test_classify_all_caps_lyric_line_is_accepted():
    from lyra_ocr.classify import classify_track

    # Modern lyric video line styled in ALL CAPS
    track = create_mock_track(
        track_id=14,
        start_ms=16000,
        end_ms=23000,
        texts=["YOU READ MY EYES, MY VIBE, MY BODY SPEAKS"],
        bbox=[72, 153, 568, 179],
        video_duration_ms=176000,
    )
    result = classify_track(track, metadata={"title": "When I Say So", "channel": "Vibe Music"})

    assert result.label == "lyric"
    assert result.decision == "accept"
    assert "reading zone" in result.reason.lower() or "lyric" in result.reason.lower()


def test_classify_short_lyric_line_is_accepted():
    from lyra_ocr.classify import classify_track

    # Short lyric cue (2 words) in reading zone
    track = create_mock_track(
        track_id=15,
        start_ms=25000,
        end_ms=28000,
        texts=["I miss you"],
        bbox=[400, 500, 600, 540],
        video_duration_ms=176000,
    )
    result = classify_track(track, metadata={"title": "When I Say So", "channel": "Vibe Music"})

    assert result.label == "lyric"
    assert result.decision == "accept"


def test_classify_intro_lower_third_lyric_line_is_accepted():
    from lyra_ocr.classify import classify_track

    # Lower-third lyric line early in video (start_ms=4000 of 212000, mean_cy=0.85)
    track = create_mock_track(
        track_id=50,
        start_ms=4000,
        end_ms=7500,
        texts=["YOUVE BEEN RUNNIN ROUND"],
        bbox=[300, 580, 980, 640],
        video_duration_ms=212000,
    )
    result = classify_track(track, metadata={"title": "Attention"})

    assert result.label == "lyric"
    assert result.decision == "accept"


def test_classify_early_repeated_word_lyric_not_treated_as_card():
    from lyra_ocr.classify import classify_track

    # Real Attention-style first lyric: 8 words but only 4 unique tokens, high
    # on screen early in the video. It must not be mistaken for a title card.
    track = create_mock_track(
        track_id=53,
        start_ms=9000,
        end_ms=13000,
        texts=["YOUVE BEEN RUNNIN ROUND RUNNIN ROUND RUNNIN"],
        bbox=[320, 304, 960, 354],
        video_duration_ms=212000,
    )
    result = classify_track(track, metadata={"title": "Attention"})

    assert result.label == "lyric"
    assert result.decision == "accept"


def test_classify_isolated_presents_and_sound_only():
    from lyra_ocr.classify import classify_track

    track_presents = create_mock_track(
        track_id=51,
        start_ms=1000,
        end_ms=4000,
        texts=["Presents"],
        bbox=[500, 300, 780, 360],
        video_duration_ms=100000,
    )
    res_presents = classify_track(track_presents)
    assert res_presents.label == "non_lyric"

    track_sound = create_mock_track(
        track_id=52,
        start_ms=1000,
        end_ms=4000,
        texts=["SOUND"],
        bbox=[500, 300, 780, 360],
        video_duration_ms=100000,
    )
    res_sound = classify_track(track_sound)
    assert res_sound.label == "non_lyric"


def test_effective_duration_ms():
    from lyra_ocr.track import effective_duration_ms

    assert effective_duration_ms(1000, 1000, 1000) == 1000
    assert effective_duration_ms(1000, 3000, 1000) == 3000
    assert effective_duration_ms(1000, 1000, 0) == 0


def test_classify_one_frame_lyric_with_sample_interval():
    from lyra_ocr.classify import classify_track

    track = create_mock_track(
        track_id=60,
        start_ms=15000,
        end_ms=15000,
        texts=["Castle in the sky"],
        bbox=[400, 300, 800, 340],
        video_duration_ms=100000,
    )
    result = classify_track(track, sample_interval_ms=1000)
    assert result.label == "lyric"
    assert result.decision == "accept"


def test_classify_chorus_title_hook_in_active_lyric_layer_is_accepted():
    from lyra_ocr.classify import classify_track

    v_dur = 240000
    meta = {"title": "Echoes in the Rain", "channel": "Studio Band"}
    p1 = create_mock_track(
        track_id=70,
        start_ms=15000,
        end_ms=18500,
        texts=["Opening line of the verse"],
        bbox=[300, 600, 980, 650],
        video_duration_ms=v_dur,
    )
    p2 = create_mock_track(
        track_id=71,
        start_ms=20000,
        end_ms=23500,
        texts=["Second line of the verse"],
        bbox=[300, 600, 980, 650],
        video_duration_ms=v_dur,
    )
    # 59s / 240s = 0.2458 <= 0.25, previously deleted by the flat 25% intro cutoff
    hook = create_mock_track(
        track_id=72,
        start_ms=59000,
        end_ms=62500,
        texts=["Echoes in the Rain"],
        bbox=[300, 600, 980, 650],
        video_duration_ms=v_dur,
    )
    for t in (p1, p2, hook):
        t.features["layer_id"] = 1
        t.features["layer_kind"] = "lyric"

    result = classify_track(
        hook,
        metadata=meta,
        video_duration_ms=v_dur,
        all_tracks=[p1, p2, hook],
    )
    assert result.label == "lyric"
    assert result.decision == "accept"
    assert result.confidence == pytest.approx(0.88)
    assert "title_hook_in_lyric_layer" in result.qa_flags
    assert "potential_chorus" in result.qa_flags


def test_classify_title_card_without_prior_lyric_tracks_is_rejected():
    from lyra_ocr.classify import classify_track

    v_dur = 240000
    meta = {"title": "Echoes in the Rain", "channel": "Studio Band"}
    # Short (<800ms) prior flashes must not qualify as established lyric lines
    flash1 = create_mock_track(
        track_id=80,
        start_ms=5000,
        end_ms=5500,
        texts=["Flash text one"],
        bbox=[300, 600, 980, 650],
        video_duration_ms=v_dur,
    )
    flash2 = create_mock_track(
        track_id=81,
        start_ms=10000,
        end_ms=10500,
        texts=["Flash text two"],
        bbox=[300, 600, 980, 650],
        video_duration_ms=v_dur,
    )
    # Delayed title card at 15% with 0 valid prior >=800ms lyric tracks
    title_card = create_mock_track(
        track_id=82,
        start_ms=36000,
        end_ms=40000,
        texts=["Echoes in the Rain"],
        bbox=[300, 600, 980, 650],
        video_duration_ms=v_dur,
    )
    for t in (flash1, flash2, title_card):
        t.features["layer_id"] = 1
        t.features["layer_kind"] = "lyric"

    result = classify_track(
        title_card,
        metadata=meta,
        video_duration_ms=v_dur,
        all_tracks=[flash1, flash2, title_card],
    )
    assert result.label == "non_lyric"
    assert result.decision == "reject"
    assert "metadata_title_match" in result.qa_flags


def test_classify_all_tracks_rescues_chorus_title_hook_and_rejects_intro_card():
    from lyra_ocr.classify import classify_all_tracks

    v_dur = 240000
    meta = {"title": "Echoes in the Rain", "channel": "Studio Band"}
    intro_card = create_mock_track(
        track_id=90,
        start_ms=1000,
        end_ms=5000,
        texts=["Echoes in the Rain"],
        bbox=[440, 200, 840, 260],
        video_duration_ms=v_dur,
    )
    verse_lines = [
        (91, 15000, 18500, "Walking down the empty street"),
        (92, 20000, 23500, "Neon lights beneath my feet"),
        (93, 25000, 28500, "Waiting for the storm to clear"),
        (94, 30000, 33500, "Every whisper drawing near"),
    ]
    lyric_tracks = [
        create_mock_track(
            track_id=tid,
            start_ms=s,
            end_ms=e,
            texts=[txt],
            bbox=[320, 580, 960, 620],
            video_duration_ms=v_dur,
        )
        for tid, s, e, txt in verse_lines
    ]
    chorus_hook = create_mock_track(
        track_id=95,
        start_ms=58000,
        end_ms=61500,
        texts=["Echoes in the Rain"],
        bbox=[320, 580, 960, 620],
        video_duration_ms=v_dur,
    )
    all_tracks = [intro_card, *lyric_tracks, chorus_hook]
    results = classify_all_tracks(all_tracks, metadata=meta, video_duration_ms=v_dur)

    assert results[0].label == "non_lyric"
    assert results[0].decision == "reject"
    assert "metadata_title_match" in results[0].qa_flags

    assert results[-1].label == "lyric"
    assert results[-1].decision == "accept"
    assert "title_hook_in_lyric_layer" in results[-1].qa_flags
    assert "potential_chorus" in results[-1].qa_flags


@pytest.mark.parametrize(
    "credit_text",
    [
        "by Producer-X",
        "by Kurousa-P",
        "Compilation: Track Anthology",
        "Animation: Studio Alpha",
        "Music: Composer Name",
        "Lyrics: Writer Name",
        "Vocal: Singer Name",
        "Illust - Artist Name",
        "Arranged: Arranger",
        "Mix: Audio Engineer",
        "Mastered: Mastering Lab",
        "Subtitles: Fan Sub",
        "Edit: Video Editor",
    ],
)
def test_classify_intro_production_credit_prefixes_rejected(credit_text: str):
    from lyra_ocr.classify import classify_track

    track = create_mock_track(
        track_id=100,
        start_ms=2000,
        end_ms=6000,
        texts=[credit_text],
        bbox=[320, 580, 960, 630],
        video_duration_ms=200000,
    )
    # Even if clustered into a lyric layer, intro credit prefix must reject it
    track.features["layer_id"] = 1
    track.features["layer_kind"] = "lyric"

    result = classify_track(track, video_duration_ms=200000)
    assert result.label == "non_lyric"
    assert result.decision == "reject"
    assert result.confidence == pytest.approx(0.95)
    assert "intro_credit_pattern" in result.qa_flags


def test_classify_credit_like_prefix_outside_intro_window_not_rejected():
    from lyra_ocr.classify import classify_track

    track = create_mock_track(
        track_id=101,
        start_ms=50000,
        end_ms=53500,
        texts=["by the riverside we wait"],
        bbox=[320, 580, 960, 630],
        video_duration_ms=100000,
    )
    result = classify_track(track, video_duration_ms=100000)
    assert result.label == "lyric"
    assert result.decision == "accept"
    assert "intro_credit_pattern" not in result.qa_flags


@pytest.mark.parametrize(
    ("phrase", "layer_kind"),
    [
        ("by your side I stay", "sparse"),
        ("by your side I stay", "lyric"),
        ("by the riverside we wait", "sparse"),
        ("by the riverside we wait", "lyric"),
        ("by starlight we keep running on", "lyric"),
    ],
)
def test_classify_early_english_by_verse_phrases_not_rejected(phrase: str, layer_kind: str):
    from lyra_ocr.classify import classify_track

    track = create_mock_track(
        track_id=102,
        start_ms=2000,
        end_ms=5500,
        texts=[phrase],
        bbox=[320, 580, 960, 630],
        video_duration_ms=200000,
    )
    track.features["layer_id"] = 1
    track.features["layer_kind"] = layer_kind

    result = classify_track(track, video_duration_ms=200000)
    assert result.label == "lyric"
    assert result.decision == "accept"
    assert "intro_credit_pattern" not in result.qa_flags


def test_classify_intro_credit_zero_video_duration_without_start_ratio_not_rejected():
    from lyra_ocr.classify import classify_track

    track = create_mock_track(
        track_id=103,
        start_ms=50000,
        end_ms=53500,
        texts=["by Kurousa-P"],
        bbox=[320, 580, 960, 630],
        video_duration_ms=100000,
    )
    # Remove start_time_ratio to simulate caller invoking classify_track with video_duration_ms=0
    track.features.pop("start_time_ratio", None)
    track.features["lifetime_ratio"] = 0.02

    result = classify_track(track, video_duration_ms=0)
    assert "intro_credit_pattern" not in result.qa_flags
    assert result.label == "lyric"
    assert result.decision == "accept"


@pytest.mark.parametrize(
    ("text", "start_ms", "end_ms"),
    [
        ("Thanks for watching", 242000, 252000),
        ("WATCHING!", 244000, 252000),
        ("(Color Coded Lyrics Kan/Rom/Eng)", 2000, 10000),
        ("fripSide (フリップサイド)", 3000, 11000),
        ("Anime: Shigatsu wa Kimi no Uso", 5000, 12000),
        ("Edition- by Kurousa-P", 240000, 250000),
        ("my", 0, 12000),
        ("my", 243000, 251000),
    ],
)
def test_classify_rejects_intro_outro_signoffs_and_split_title_fragments(
    text: str, start_ms: int, end_ms: int
):
    from lyra_ocr.classify import classify_track

    v_dur = 255000
    track = create_mock_track(
        track_id=104,
        start_ms=start_ms,
        end_ms=end_ms,
        texts=[text],
        bbox=[320, 580, 960, 630],
        video_duration_ms=v_dur,
    )
    result = classify_track(
        track,
        metadata={"title": "only my railgun", "artist": "fripSide", "channel": "Geneon"},
        video_duration_ms=v_dur,
    )
    assert result.label == "non_lyric"
    assert result.decision == "reject"


def test_classify_preserves_sung_one_word_title_token_in_boundary_window():
    from lyra_ocr.classify import classify_track

    v_dur = 255000
    meta = {"title": "only my railgun", "artist": "fripSide", "channel": "Geneon"}

    # 1. Inside an active lyric layer, a 1-word title token is preserved
    lyric_layer_track = create_mock_track(
        track_id=105,
        start_ms=5000,
        end_ms=8000,
        texts=["Railgun"],
        bbox=[320, 580, 960, 630],
        video_duration_ms=v_dur,
    )
    lyric_layer_track.features["layer_id"] = 1
    lyric_layer_track.features["layer_kind"] = "lyric"
    res_layer = classify_track(lyric_layer_track, metadata=meta, video_duration_ms=v_dur)
    assert res_layer.label == "lyric"
    assert res_layer.decision == "accept"

    # 2. Outside lyric layer but with active vocal_profile, a 1-word title token is preserved
    sparse_voiced_track = create_mock_track(
        track_id=106,
        start_ms=5000,
        end_ms=8000,
        texts=["Railgun"],
        bbox=[320, 580, 960, 630],
        video_duration_ms=v_dur,
    )
    import numpy as np

    vocal_profile = np.zeros(5100, dtype=np.float32)
    vocal_profile[100:160] = 0.9
    res_voiced = classify_track(
        sparse_voiced_track,
        metadata=meta,
        video_duration_ms=v_dur,
        vocal_profile=vocal_profile,
    )
    assert res_voiced.label == "lyric"
    assert res_voiced.decision == "accept"

    # 3. Outside lyric layer and unvoiced in vocal_profile, split title fragment is still rejected
    sparse_silent_track = create_mock_track(
        track_id=107,
        start_ms=5000,
        end_ms=8000,
        texts=["my"],
        bbox=[320, 580, 960, 630],
        video_duration_ms=v_dur,
    )
    silent_profile = np.zeros(5100, dtype=np.float32)
    res_silent = classify_track(
        sparse_silent_track,
        metadata=meta,
        video_duration_ms=v_dur,
        vocal_profile=silent_profile,
    )
    assert res_silent.label == "non_lyric"
    assert res_silent.decision == "reject"




