"""Greedy spatial IoU + centroid distance baseline tracker."""

from __future__ import annotations

from typing import Any, Optional

from lyra_ocr.track import DetectionOccurrence, TextTrack, bbox_iou, centroid_distance
from lyra_ocr.trackers.base import BaseTracker


class GreedyTracker(BaseTracker):
    """Frame-by-frame tracker associating boxes via greedy bipartite matching.

    Uses IoU and centroid distance heuristics. Baseline implementation.
    """

    def __init__(
        self,
        frame_width: int = 1280,
        frame_height: int = 720,
        iou_threshold: float = 0.30,
        centroid_threshold: float = 0.08,
        max_gap_ms: int = 2000,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            frame_width=frame_width,
            frame_height=frame_height,
            max_gap_ms=max_gap_ms,
        )
        self.iou_threshold = iou_threshold
        self.centroid_threshold = centroid_threshold

    def update(self, frame_data: dict[str, Any]) -> None:
        """Process a single frame OCR result and greedily associate boxes to tracks."""
        timestamp_ms = frame_data.get("timestamp_ms", 0)
        detections = self.parse_detections(frame_data)

        matched_tracks: set[int] = set()
        unmatched_detections: list[DetectionOccurrence] = []

        # Greedy matching based on spatial similarity
        for det in detections:
            best_track: Optional[TextTrack] = None
            best_score = -1.0

            for tr in self.active_tracks:
                if tr.track_id in matched_tracks:
                    continue

                iou = bbox_iou(det.bbox, tr.bbox)
                c_dist = centroid_distance(
                    det.bbox, tr.bbox, self.frame_width, self.frame_height
                )

                # Matching condition: high IoU OR close centroids
                if iou >= self.iou_threshold or (
                    iou >= 0.15 and c_dist <= self.centroid_threshold
                ):
                    score = iou + (1.0 - c_dist)
                    if score > best_score:
                        best_score = score
                        best_track = tr

            if best_track is not None:
                matched_tracks.add(best_track.track_id)
                self.associate(best_track, det)
            else:
                unmatched_detections.append(det)

        # Handle active tracks not matched in this frame
        still_active: list[TextTrack] = []
        for tr in self.active_tracks:
            if tr.track_id not in matched_tracks:
                if (timestamp_ms - tr.end_ms) > self.max_gap_ms:
                    self.finished_tracks.append(tr)
                else:
                    still_active.append(tr)
            else:
                still_active.append(tr)
        self.active_tracks = still_active

        # Spawn new tracks for unmatched detections
        for det in unmatched_detections:
            new_track = self.create_new_track(det)
            self.active_tracks.append(new_track)
