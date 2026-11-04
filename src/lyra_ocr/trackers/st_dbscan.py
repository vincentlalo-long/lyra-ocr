"""Spatio-Temporal Tube Clustering (ST-DBSCAN) text tracker.

Academic Reference:
S. Sakaguchi, J. Kato, M. Goto, S. Uchida, "Lyric Video Analysis Using Text Detection and Tracking",
2020. https://arxiv.org/abs/2006.11933
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.cluster import DBSCAN

from lyra_ocr.track import DetectionOccurrence, TextTrack
from lyra_ocr.trackers.base import BaseTracker


class ST_DBSCANTracker(BaseTracker):
    """Batch spatio-temporal tube clustering tracker using DBSCAN density discovery.

    Maps all detections across the video into a 4D metric space [cx/W, cy/H, t/T, log(area)]
    and clusters dense spatio-temporal text tubes without sequential greedy drift or error accumulation.
    """

    def __init__(
        self,
        frame_width: int = 1280,
        frame_height: int = 720,
        eps: float = 0.18,
        min_samples: int = 2,
        weight_x: float = 3.0,
        weight_y: float = 5.0,
        weight_t: float = 1.0,
        weight_scale: float = 0.5,
        max_gap_ms: int = 3000,
        min_confidence: float = 0.35,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            frame_width=frame_width,
            frame_height=frame_height,
            max_gap_ms=max_gap_ms,
        )
        self.eps = eps
        self.min_samples = min_samples
        self.weight_x = weight_x
        self.weight_y = weight_y
        self.weight_t = weight_t
        self.weight_scale = weight_scale
        self.min_confidence = min_confidence

        self._accumulated_detections: list[DetectionOccurrence] = []
        self._max_observed_timestamp_ms = 0

    def update(self, frame_data: dict[str, Any]) -> None:
        """Accumulate detections from each frame for global spatio-temporal clustering."""
        detections = self.parse_detections(frame_data)
        for det in detections:
            if det.confidence >= self.min_confidence:
                self._accumulated_detections.append(det)
                if det.timestamp_ms > self._max_observed_timestamp_ms:
                    self._max_observed_timestamp_ms = det.timestamp_ms

    def finalize(self, video_duration_ms: int = 0) -> list[TextTrack]:
        """Execute spatio-temporal DBSCAN clustering and construct segmented TextTracks."""
        if not self._accumulated_detections:
            return []

        v_dur = max(
            1000,
            video_duration_ms if video_duration_ms > 0 else self._max_observed_timestamp_ms + 1000,
        )
        W = max(1, self.frame_width)
        H = max(1, self.frame_height)
        frame_area = max(1.0, float(W * H))

        # 1. Build normalized 4D coordinate feature vectors
        n_points = len(self._accumulated_detections)
        X = np.zeros((n_points, 4), dtype=np.float32)

        for i, det in enumerate(self._accumulated_detections):
            b = det.bbox
            cx = (b[0] + b[2]) / 2.0
            cy = (b[1] + b[3]) / 2.0
            w = max(1.0, float(b[2] - b[0]))
            h = max(1.0, float(b[3] - b[1]))
            area = w * h

            norm_x = (cx / W) * self.weight_x
            norm_y = (cy / H) * self.weight_y
            norm_t = (det.timestamp_ms / v_dur) * self.weight_t
            # Log area ratio normalized
            log_scale = math.log(max(1e-6, area / frame_area))
            norm_scale = (log_scale / 10.0) * self.weight_scale

            X[i] = [norm_x, norm_y, norm_t, norm_scale]

        # 2. Run DBSCAN clustering
        db = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric="euclidean")
        labels = db.fit_predict(X)

        # 3. Group detections by cluster
        clusters: dict[int, list[DetectionOccurrence]] = {}
        for idx, lbl in enumerate(labels):
            if lbl == -1:
                # Noise points: discarded or treated as isolated detections
                continue
            clusters.setdefault(lbl, []).append(self._accumulated_detections[idx])

        # 4. Convert each spatio-temporal tube into TextTrack(s)
        self.finished_tracks = []
        for cluster_id, tube_dets in clusters.items():
            # Sort chronologically
            tube_dets.sort(key=lambda d: d.timestamp_ms)

            # Deduplicate multiple detections in same frame by choosing highest confidence
            frame_map: dict[int, DetectionOccurrence] = {}
            for d in tube_dets:
                if d.frame_id not in frame_map or d.confidence > frame_map[d.frame_id].confidence:
                    frame_map[d.frame_id] = d
            sorted_dets = sorted(frame_map.values(), key=lambda d: d.timestamp_ms)

            # Split tube if temporal gap exceeds max_gap_ms
            sub_tubes: list[list[DetectionOccurrence]] = []
            curr_tube: list[DetectionOccurrence] = []
            for d in sorted_dets:
                if not curr_tube:
                    curr_tube.append(d)
                else:
                    gap = d.timestamp_ms - curr_tube[-1].timestamp_ms
                    if gap > self.max_gap_ms:
                        sub_tubes.append(curr_tube)
                        curr_tube = [d]
                    else:
                        curr_tube.append(d)
            if curr_tube:
                sub_tubes.append(curr_tube)

            # Build TextTrack for each coherent temporal tube
            for sub in sub_tubes:
                first_det = sub[0]
                track = self.create_new_track(first_det)
                for det in sub[1:]:
                    self.add_occurrence_to_track(track, det)
                self.finished_tracks.append(track)

        # Finalize features across all finished tracks
        return super().finalize(video_duration_ms=video_duration_ms)
