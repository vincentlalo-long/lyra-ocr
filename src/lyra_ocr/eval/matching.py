"""Track-to-Golden matching via spatio-temporal and text similarity with Hungarian assignment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from lyra_ocr.eval.types import EvaluationItem, TrackLabel, TrackPrediction
from lyra_ocr.track import bbox_iou, string_similarity


@dataclass
class TrackMatch:
    pred_track: TrackPrediction
    gold_track: TrackPrediction
    gold_label: TrackLabel
    temporal_overlap: float
    spatial_iou: float
    text_similarity: float
    match_score: float


def compute_temporal_overlap(
    start_a: int, end_a: int, start_b: int, end_b: int
) -> float:
    """Compute 1D intersection over union for two time intervals."""
    # Point-in-time tracks (single sampled frame) have zero-length intervals.
    # Two detections at the same timestamp are a perfect temporal match;
    # otherwise they cannot overlap.
    point_a = start_a == end_a
    point_b = start_b == end_b
    if point_a and point_b:
        return 1.0 if start_a == start_b else 0.0
    # A point track fully contained inside an interval track counts as a
    # perfect overlap of the point; otherwise the two cannot match.
    if point_a or point_b:
        pt = start_a if point_a else start_b
        iv_s = start_b if point_a else start_a
        iv_e = end_b if point_a else end_a
        return 1.0 if iv_s <= pt <= iv_e else 0.0
    inter = max(0, min(end_a, end_b) - max(start_a, start_b))
    union = max(1, max(end_a, end_b) - min(start_a, start_b))
    return float(inter) / float(union)


def match_predictions_to_golden(
    predicted_tracks: list[TrackPrediction],
    golden_items: list[EvaluationItem],
    min_temporal_overlap: float = 0.40,
    min_spatial_iou: float = 0.40,
    min_text_similarity: float = 0.50,
) -> tuple[list[EvaluationItem], list[TrackMatch], list[TrackPrediction]]:
    """Match arbitrary predicted tracks to golden reference tracks using Hungarian assignment.

    Returns:
        paired_items: EvaluationItem list with transferred golden labels for matched tracks
        matches: list of TrackMatch records with alignment statistics
        unmatched_preds: predicted tracks that could not be mapped to any golden track
    """
    # Only golden tracks from videos present in this prediction batch are
    # relevant; otherwise a per-video call would mark every other video's
    # golden tracks as missing.
    pred_video_ids = {p.video_id for p in predicted_tracks}
    valid_golden = [
        it
        for it in golden_items
        if it.has_label and it.label is not None and it.pred.video_id in pred_video_ids
    ]
    if not predicted_tracks or not valid_golden:
        # No matches possible
        empty_items = [EvaluationItem(pred=p, label=None) for p in predicted_tracks]
        return empty_items, [], list(predicted_tracks)

    n_pred = len(predicted_tracks)
    n_gold = len(valid_golden)
    cost_matrix = np.full((n_pred, n_gold), 1000.0, dtype=np.float32)

    similarity_matrix = np.zeros((n_pred, n_gold, 3), dtype=np.float32)

    for i, p in enumerate(predicted_tracks):
        for j, g in enumerate(valid_golden):
            # Only match within same video_id
            if p.video_id != g.pred.video_id:
                continue

            t_ov = compute_temporal_overlap(p.start_ms, p.end_ms, g.pred.start_ms, g.pred.end_ms)
            s_iou = bbox_iou(p.bbox, g.pred.bbox) if p.bbox and g.pred.bbox else 0.0
            t_sim = string_similarity(p.dominant_text, g.pred.dominant_text)

            similarity_matrix[i, j] = [t_ov, s_iou, t_sim]

            # Gating condition
            if (
                t_ov >= min_temporal_overlap
                and s_iou >= min_spatial_iou
                and t_sim >= min_text_similarity
            ):
                score = 0.40 * t_ov + 0.30 * s_iou + 0.30 * t_sim
                cost_matrix[i, j] = 1.0 - score

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched_pairs: list[EvaluationItem] = []
    matches_detail: list[TrackMatch] = []
    matched_pred_indices: set[int] = set()
    matched_gold_indices: set[int] = set()

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] < 500.0:
            pred = predicted_tracks[r]
            gold_item = valid_golden[c]
            gold_lbl = gold_item.label
            assert gold_lbl is not None

            sims = similarity_matrix[r, c]
            score = 1.0 - cost_matrix[r, c]

            # Clone label with current track_id
            paired_lbl = TrackLabel(
                video_id=pred.video_id,
                track_id=pred.track_id,
                human_label=gold_lbl.human_label,
                tracking_error=gold_lbl.tracking_error,
                notes=f"Matched to golden track {gold_item.pred.track_id} (score={score:.2f})",
                annotator=gold_lbl.annotator,
                reviewed_at=gold_lbl.reviewed_at,
            )

            matched_pairs.append(EvaluationItem(pred=pred, label=paired_lbl))
            matches_detail.append(
                TrackMatch(
                    pred_track=pred,
                    gold_track=gold_item.pred,
                    gold_label=gold_lbl,
                    temporal_overlap=round(float(sims[0]), 3),
                    spatial_iou=round(float(sims[1]), 3),
                    text_similarity=round(float(sims[2]), 3),
                    match_score=round(float(score), 3),
                )
            )
            matched_pred_indices.add(r)
            matched_gold_indices.add(c)

    unmatched_preds = [p for idx, p in enumerate(predicted_tracks) if idx not in matched_pred_indices]
    for un_p in unmatched_preds:
        matched_pairs.append(EvaluationItem(pred=un_p, label=None))

    # Golden tracks without any corresponding prediction are real tracking misses.
    # Without this, a tracker that drops an entire lyric line would still report
    # false_drop = 0 because the missing track never entered the evaluation set.
    for idx, gold_item in enumerate(valid_golden):
        if idx in matched_gold_indices or gold_item.label is None:
            continue
        g = gold_item.pred
        missing_pred = TrackPrediction(
            video_id=g.video_id,
            track_id=g.track_id,
            model_label="missing",
            model_confidence=0.0,
            model_decision="missing",
            start_ms=g.start_ms,
            end_ms=g.end_ms,
            duration_ms=g.duration_ms,
            dominant_text=g.dominant_text,
            bbox=g.bbox,
        )
        matched_pairs.append(
            EvaluationItem(pred=missing_pred, label=gold_item.label, is_missing=True)
        )

    return matched_pairs, matches_detail, unmatched_preds
