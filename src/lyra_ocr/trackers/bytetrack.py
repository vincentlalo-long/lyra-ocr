"""ByteTrack-style two-stage association text tracker.

Academic Reference:
Y. Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box",
ECCV 2022. https://arxiv.org/abs/2110.06864
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from lyra_ocr.track import DetectionOccurrence, TextTrack, bbox_iou, centroid_distance
from lyra_ocr.trackers.base import BaseTracker


class ByteTextTracker(BaseTracker):
    """Two-stage association tracker matching high-confidence boxes first, then recovering low-confidence boxes.

    Stage 1 matches confident detections (conf >= high_threshold) using Hungarian IoU matching.
    Stage 2 associates remaining unmatched tracks with low-confidence boxes (low_threshold <= conf < high_threshold)
    to maintain track continuity through motion blur and compression artifacts without spawning false tracks.
    """

    def __init__(
        self,
        frame_width: int = 1280,
        frame_height: int = 720,
        high_threshold: float = 0.70,
        low_threshold: float = 0.25,
        match_threshold_high: float = 0.20,
        match_threshold_low: float = 0.15,
        max_gap_ms: int = 2000,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            frame_width=frame_width,
            frame_height=frame_height,
            max_gap_ms=max_gap_ms,
        )
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.match_threshold_high = match_threshold_high
        self.match_threshold_low = match_threshold_low

    def update(self, frame_data: dict[str, Any]) -> None:
        """Process frame OCR detections via two-stage Hungarian assignment."""
        timestamp_ms = frame_data.get("timestamp_ms", 0)
        detections = self.parse_detections(frame_data)

        # 1. Partition detections into high and low confidence pools
        high_dets: list[DetectionOccurrence] = []
        low_dets: list[DetectionOccurrence] = []
        for det in detections:
            if det.confidence >= self.high_threshold:
                high_dets.append(det)
            elif det.confidence >= self.low_threshold:
                low_dets.append(det)

        matched_track_ids: set[int] = set()
        matched_high_indices: set[int] = set()

        # 2. Stage 1: Associate Active Tracks with High-Confidence Detections
        if self.active_tracks and high_dets:
            n_tracks = len(self.active_tracks)
            n_dets = len(high_dets)
            cost_matrix = np.ones((n_tracks, n_dets), dtype=np.float32)

            for t_idx, tr in enumerate(self.active_tracks):
                for d_idx, det in enumerate(high_dets):
                    iou = bbox_iou(tr.bbox, det.bbox)
                    c_dist = centroid_distance(
                        tr.bbox, det.bbox, self.frame_width, self.frame_height
                    )
                    # Cost is 1 - IoU, discounted by close centroid proximity
                    cost_matrix[t_idx, d_idx] = 1.0 - (0.7 * iou + 0.3 * max(0.0, 1.0 - c_dist))

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for r, c in zip(row_ind, col_ind):
                tr = self.active_tracks[r]
                det = high_dets[c]
                iou = bbox_iou(tr.bbox, det.bbox)
                c_dist = centroid_distance(
                    tr.bbox, det.bbox, self.frame_width, self.frame_height
                )

                # Gating condition
                if iou >= self.match_threshold_high or (iou >= 0.10 and c_dist <= 0.08):
                    self.add_occurrence_to_track(tr, det)
                    matched_track_ids.add(tr.track_id)
                    matched_high_indices.add(c)

        unmatched_tracks = [
            tr for tr in self.active_tracks if tr.track_id not in matched_track_ids
        ]
        unmatched_high_dets = [
            det for idx, det in enumerate(high_dets) if idx not in matched_high_indices
        ]

        # 3. Stage 2: Associate Remaining Active Tracks with Low-Confidence Detections
        if unmatched_tracks and low_dets:
            n_unmatched = len(unmatched_tracks)
            n_low = len(low_dets)
            cost_matrix_low = np.ones((n_unmatched, n_low), dtype=np.float32)

            for t_idx, tr in enumerate(unmatched_tracks):
                for d_idx, det in enumerate(low_dets):
                    iou = bbox_iou(tr.bbox, det.bbox)
                    c_dist = centroid_distance(
                        tr.bbox, det.bbox, self.frame_width, self.frame_height
                    )
                    cost_matrix_low[t_idx, d_idx] = 1.0 - (0.5 * iou + 0.5 * max(0.0, 1.0 - c_dist))

            row_ind_low, col_ind_low = linear_sum_assignment(cost_matrix_low)
            for r, c in zip(row_ind_low, col_ind_low):
                tr = unmatched_tracks[r]
                det = low_dets[c]
                iou = bbox_iou(tr.bbox, det.bbox)
                c_dist = centroid_distance(
                    tr.bbox, det.bbox, self.frame_width, self.frame_height
                )

                # Relaxed gating for low-confidence recovery
                if iou >= self.match_threshold_low or (iou >= 0.08 and c_dist <= 0.08):
                    self.add_occurrence_to_track(tr, det)
                    matched_track_ids.add(tr.track_id)

        # 4. Handle tracks still unmatched after Stage 2
        still_active: list[TextTrack] = []
        for tr in self.active_tracks:
            if tr.track_id not in matched_track_ids:
                if (timestamp_ms - tr.end_ms) > self.max_gap_ms:
                    self.finished_tracks.append(tr)
                else:
                    still_active.append(tr)
            else:
                still_active.append(tr)
        self.active_tracks = still_active

        # 5. Spawn new tracks strictly from unmatched HIGH-confidence detections
        for det in unmatched_high_dets:
            new_track = self.create_new_track(det)
            self.active_tracks.append(new_track)
        # Note: Unmatched low-confidence detections are safely ignored as noise
