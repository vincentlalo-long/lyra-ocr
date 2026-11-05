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

    # Repeated hook equal to the song title, mid-video: not an intro card, so it
    # must go to human review instead of being auto-rejected.
    track = create_mock_track(
        track_id=21,
        start_ms=55000,
        end_ms=60000,
        texts=["CASTLE"],
        bbox=[300, 600, 980, 650],
        video_duration_ms=100000,
    )
    result = classify_track(track, metadata={"channel": "Clarx & Harddope", "title": "Castle"})

    assert result.label == "non_lyric"
    assert result.decision == "review"
    assert "metadata_title_match" in result.qa_flags
    assert "title_match_outside_intro" in result.qa_flags


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
