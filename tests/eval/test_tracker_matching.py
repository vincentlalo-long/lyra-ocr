"""Unit tests for Hungarian spatio-temporal and text track matching."""

from lyra_ocr.eval.matching import compute_temporal_overlap, match_predictions_to_golden
from lyra_ocr.eval.types import EvaluationItem, TrackLabel, TrackPrediction


def make_pred(
    vid: str,
    tid: str,
    start_ms: int,
    end_ms: int,
    bbox: list[int],
    text: str,
) -> TrackPrediction:
    return TrackPrediction(
        video_id=vid,
        track_id=tid,
        model_label="lyric",
        model_confidence=0.9,
        model_decision="accept",
        start_ms=start_ms,
        end_ms=end_ms,
        duration_ms=end_ms - start_ms,
        dominant_text=text,
        bbox=bbox,
    )


def make_gold_item(
    vid: str,
    tid: str,
    start_ms: int,
    end_ms: int,
    bbox: list[int],
    text: str,
    human_label: str = "lyric",
    tracking_error: bool = False,
) -> EvaluationItem:
    p = make_pred(vid, tid, start_ms, end_ms, bbox, text)
    lbl = TrackLabel(
        video_id=vid,
        track_id=tid,
        human_label=human_label,
        binary="lyric" if human_label == "lyric" else "non_lyric",
        tracking_error=tracking_error,
    )
    return EvaluationItem(pred=p, label=lbl)


def test_compute_temporal_overlap():
    # Complete overlap
    assert compute_temporal_overlap(1000, 2000, 1000, 2000) == 1.0
    # No overlap
    assert compute_temporal_overlap(1000, 2000, 2000, 3000) == 0.0
    # 50% overlap: inter = 500, union = 1500 -> 1/3
    assert abs(compute_temporal_overlap(1000, 2000, 1500, 2500) - (500 / 1500)) < 1e-4
    # Single-frame tracks: same timestamp is a full match, different is none
    assert compute_temporal_overlap(20000, 20000, 20000, 20000) == 1.0
    assert compute_temporal_overlap(20000, 20000, 20500, 20500) == 0.0
    # Point vs interval: point fully inside counts as perfect overlap
    assert compute_temporal_overlap(2000, 2000, 1000, 5000) == 1.0
    assert compute_temporal_overlap(1000, 5000, 2000, 2000) == 1.0
    # Point vs interval: point outside the interval cannot overlap
    assert compute_temporal_overlap(6000, 6000, 1000, 5000) == 0.0
    assert compute_temporal_overlap(900, 900, 1000, 5000) == 0.0


def test_match_predictions_to_golden_exact_match():
    pred = make_pred("v1", "p1", 1000, 5000, [100, 100, 500, 150], "Never gonna give you up")
    gold = make_gold_item("v1", "g1", 1000, 5000, [100, 100, 500, 150], "Never gonna give you up", human_label="lyric")

    paired, matches, unmatched = match_predictions_to_golden([pred], [gold])

    assert len(matches) == 1
    assert len(unmatched) == 0
    assert matches[0].pred_track.track_id == "p1"
    assert matches[0].gold_track.track_id == "g1"
    assert matches[0].temporal_overlap == 1.0
    assert matches[0].spatial_iou == 1.0
    assert matches[0].text_similarity == 1.0

    # Paired item should have transferred label with prediction's track_id
    assert len(paired) == 1
    assert paired[0].pred.track_id == "p1"
    assert paired[0].label is not None
    assert paired[0].label.track_id == "p1"
    assert paired[0].label.human_label == "lyric"


def test_match_predictions_cross_video_isolation():
    """Ensure tracks from different videos never match even if coords are identical."""
    pred = make_pred("video_A", "p1", 1000, 5000, [100, 100, 500, 150], "Identical lyrics")
    gold = make_gold_item("video_B", "g1", 1000, 5000, [100, 100, 500, 150], "Identical lyrics")

    paired, matches, unmatched = match_predictions_to_golden([pred], [gold])

    assert len(matches) == 0
    assert len(unmatched) == 1
    assert paired[0].label is None


def test_match_predictions_threshold_rejections():
    """Test rejection when temporal, spatial, or text similarity falls below threshold."""
    # 1. Low temporal overlap
    pred_time = make_pred("v1", "p1", 1000, 2000, [100, 100, 500, 150], "Same text")
    gold_time = make_gold_item("v1", "g1", 1900, 5000, [100, 100, 500, 150], "Same text")
    _, matches_t, un_t = match_predictions_to_golden([pred_time], [gold_time])
    assert len(matches_t) == 0
    assert len(un_t) == 1

    # 2. Low spatial IoU (disjoint boxes)
    pred_space = make_pred("v1", "p2", 1000, 5000, [0, 0, 100, 50], "Same text")
    gold_space = make_gold_item("v1", "g2", 1000, 5000, [500, 500, 600, 550], "Same text")
    _, matches_s, un_s = match_predictions_to_golden([pred_space], [gold_space])
    assert len(matches_s) == 0
    assert len(un_s) == 1

    # 3. Low text similarity
    pred_text = make_pred("v1", "p3", 1000, 5000, [100, 100, 500, 150], "Completely different text")
    gold_text = make_gold_item("v1", "g3", 1000, 5000, [100, 100, 500, 150], "XYZ 123456789")
    _, matches_txt, un_txt = match_predictions_to_golden([pred_text], [gold_text])
    assert len(matches_txt) == 0
    assert len(un_txt) == 1


def test_single_frame_golden_track_matches_interval_prediction():
    """A point-in-time golden track must match a prediction covering that moment."""
    pred = make_pred("v1", "p1", 1000, 5000, [100, 100, 500, 150], "Never gonna give you up")
    gold = make_gold_item("v1", "g1", 3000, 3000, [100, 100, 500, 150], "Never gonna give you up")

    paired, matches, unmatched = match_predictions_to_golden([pred], [gold])

    assert len(matches) == 1
    assert matches[0].temporal_overlap == 1.0
    assert len(unmatched) == 0
    assert not [it for it in paired if it.is_missing]


def test_unmatched_golden_track_becomes_missing_false_drop():
    """A golden lyric with no prediction must surface as a missing false drop."""
    # Prediction matches only the channel watermark; the lyric line is absent.
    pred = make_pred("v1", "p1", 1000, 5000, [50, 50, 200, 80], "CLARX")
    gold_lyric = make_gold_item(
        "v1", "g1", 1000, 5000, [300, 600, 850, 650], "Never gonna give you up", human_label="lyric"
    )
    gold_channel = make_gold_item(
        "v1", "g2", 1000, 5000, [50, 50, 200, 80], "CLARX", human_label="channel"
    )

    paired, matches, unmatched = match_predictions_to_golden(
        [pred], [gold_lyric, gold_channel]
    )

    assert len(matches) == 1
    assert len(unmatched) == 0
    missing = [it for it in paired if it.is_missing]
    assert len(missing) == 1
    assert missing[0].label is not None
    assert missing[0].label.human_label == "lyric"
    assert missing[0].pred.model_label == "missing"


def test_match_predictions_hungarian_assignment_optimal():
    """Test 2-vs-2 assignment where Hungarian algorithm picks the global optimum."""
    p1 = make_pred("v1", "p1", 1000, 3000, [100, 100, 300, 150], "Line One")
    p2 = make_pred("v1", "p2", 3000, 5000, [100, 100, 300, 150], "Line Two")

    g1 = make_gold_item("v1", "g1", 1000, 3000, [100, 100, 300, 150], "Line One", human_label="lyric")
    g2 = make_gold_item("v1", "g2", 3000, 5000, [100, 100, 300, 150], "Line Two", human_label="lyric")

    paired, matches, unmatched = match_predictions_to_golden([p1, p2], [g1, g2])

    assert len(matches) == 2
    assert len(unmatched) == 0
    matched_map = {m.pred_track.track_id: m.gold_track.track_id for m in matches}
    assert matched_map["p1"] == "g1"
    assert matched_map["p2"] == "g2"
