"""SORT-Text tracker with Linear Kalman Motion Filter.

Academic References:
1. A. Bewley et al., "Simple Online and Realtime Tracking", ICIP 2016. https://arxiv.org/abs/1602.00763
2. C. Wang et al., "An End-to-end Video Text Detector with Online Tracking", ICDAR 2019. https://arxiv.org/abs/1908.07135
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from lyra_ocr.track import DetectionOccurrence, TextTrack, bbox_iou, centroid_distance
from lyra_ocr.trackers.base import BaseTracker


def bbox_to_z(bbox: list[int] | list[float]) -> np.ndarray:
    """Convert [xmin, ymin, xmax, ymax] box to [u, v, s, r] measurement vector.

    u, v: Center coordinates
    s: Area scale (width * height)
    r: Aspect ratio (width / height)
    """
    w = max(1.0, float(bbox[2] - bbox[0]))
    h = max(1.0, float(bbox[3] - bbox[1]))
    u = float(bbox[0]) + w / 2.0
    v = float(bbox[1]) + h / 2.0
    s = w * h
    r = w / h
    return np.array([u, v, s, r], dtype=np.float32)


def x_to_bbox(x: np.ndarray) -> list[int]:
    """Convert [u, v, s, r, ...] Kalman state vector back to [xmin, ymin, xmax, ymax] integer box."""
    u, v, s, r = float(x[0]), float(x[1]), float(x[2]), float(x[3])
    s = max(1.0, s)
    r = max(0.01, r)

    w = math.sqrt(s * r)
    h = s / max(1e-5, w)

    xmin = int(round(u - w / 2.0))
    ymin = int(round(v - h / 2.0))
    xmax = int(round(u + w / 2.0))
    ymax = int(round(v + h / 2.0))
    return [xmin, ymin, xmax, ymax]


class KalmanBoxTracker:
    """8D Linear Kalman Filter for bounding box tracking with constant velocity motion model."""

    count = 0

    def __init__(self, bbox: list[int] | list[float]) -> None:
        # State vector: [u, v, s, r, u_dot, v_dot, s_dot, 0]
        self.x = np.zeros(8, dtype=np.float32)
        z = bbox_to_z(bbox)
        self.x[:4] = z

        # State transition matrix F (8x8)
        self.F = np.eye(8, dtype=np.float32)
        for i in range(3):
            self.F[i, i + 4] = 1.0  # u += u_dot, v += v_dot, s += s_dot

        # Measurement matrix H (4x8)
        self.H = np.zeros((4, 8), dtype=np.float32)
        self.H[:4, :4] = np.eye(4, dtype=np.float32)

        # Covariance matrix P
        self.P = np.eye(8, dtype=np.float32) * 10.0
        self.P[4:, 4:] *= 100.0  # high uncertainty on initial velocities

        # Process noise Q
        self.Q = np.eye(8, dtype=np.float32)
        self.Q[:2, :2] *= 1.0
        self.Q[2, 2] *= 2.0
        self.Q[3, 3] *= 0.01  # aspect ratio rarely drifts
        self.Q[4:7, 4:7] *= 0.01
        self.Q[7, 7] *= 0.0001

        # Measurement noise R
        self.R = np.eye(4, dtype=np.float32)
        self.R[:2, :2] *= 2.0
        self.R[2, 2] *= 10.0
        self.R[3, 3] *= 10.0

        self.time_since_update = 0
        self.history: list[np.ndarray] = []

    def predict(self) -> list[int]:
        """Advance state prediction and return predicted bounding box."""
        if (self.x[6] + self.x[2]) <= 0.0:
            self.x[6] = 0.0  # Prevent scale shrinking to zero or negative

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        self.time_since_update += 1
        pred_box = x_to_bbox(self.x)
        self.history.append(self.x.copy())
        return pred_box

    def update(self, bbox: list[int] | list[float]) -> None:
        """Update Kalman filter state with new observed detection box."""
        self.time_since_update = 0
        self.history = []

        z = bbox_to_z(bbox)
        y = z - self.H @ self.x  # Innovation residual
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        I = np.eye(8, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P


class KalmanSORTTracker(BaseTracker):
    """SORT-style online text tracker using Kalman Filter motion prediction and Hungarian bipartite matching.

    Maintains trajectory continuity even during kinetic text transitions, camera pan/zoom, or scrolling lyrics.
    """

    def __init__(
        self,
        frame_width: int = 1280,
        frame_height: int = 720,
        iou_threshold: float = 0.25,
        min_confidence: float = 0.40,
        max_gap_ms: int = 2000,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            frame_width=frame_width,
            frame_height=frame_height,
            max_gap_ms=max_gap_ms,
        )
        self.iou_threshold = iou_threshold
        self.min_confidence = min_confidence
        self.kalman_filters: dict[int, KalmanBoxTracker] = {}

    def update(self, frame_data: dict[str, Any]) -> None:
        """Update tracks via Kalman motion prediction and Hungarian matching."""
        timestamp_ms = frame_data.get("timestamp_ms", 0)
        raw_detections = self.parse_detections(frame_data)
        detections = [d for d in raw_detections if d.confidence >= self.min_confidence]

        # 1. Predict new positions for all currently active tracks
        predicted_boxes: list[list[int]] = []
        for tr in self.active_tracks:
            kf = self.kalman_filters.get(tr.track_id)
            if kf is not None:
                pbox = kf.predict()
            else:
                pbox = tr.bbox
            predicted_boxes.append(pbox)

        matched_track_ids: set[int] = set()
        matched_det_indices: set[int] = set()

        # 2. Hungarian matching on (1 - IoU) cost matrix
        if self.active_tracks and detections:
            n_tracks = len(self.active_tracks)
            n_dets = len(detections)
            cost_matrix = np.ones((n_tracks, n_dets), dtype=np.float32)

            for t_idx, pbox in enumerate(predicted_boxes):
                for d_idx, det in enumerate(detections):
                    iou = bbox_iou(pbox, det.bbox)
                    c_dist = centroid_distance(
                        pbox, det.bbox, self.frame_width, self.frame_height
                    )
                    cost_matrix[t_idx, d_idx] = 1.0 - (0.8 * iou + 0.2 * max(0.0, 1.0 - c_dist))

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for r, c in zip(row_ind, col_ind):
                tr = self.active_tracks[r]
                pbox = predicted_boxes[r]
                det = detections[c]

                iou = bbox_iou(pbox, det.bbox)
                c_dist = centroid_distance(
                    pbox, det.bbox, self.frame_width, self.frame_height
                )

                # Gating condition
                if iou >= self.iou_threshold or (iou >= 0.10 and c_dist <= 0.08):
                    # Update Kalman Filter
                    kf = self.kalman_filters.get(tr.track_id)
                    if kf is not None:
                        kf.update(det.bbox)
                    self.add_occurrence_to_track(tr, det)
                    matched_track_ids.add(tr.track_id)
                    matched_det_indices.add(c)

        # 3. Handle active tracks not matched in this frame
        still_active: list[TextTrack] = []
        for tr in self.active_tracks:
            if tr.track_id not in matched_track_ids:
                if (timestamp_ms - tr.end_ms) > self.max_gap_ms:
                    self.finished_tracks.append(tr)
                    self.kalman_filters.pop(tr.track_id, None)
                else:
                    still_active.append(tr)
            else:
                still_active.append(tr)
        self.active_tracks = still_active

        # 4. Spawn new tracks for unmatched detections
        for idx, det in enumerate(detections):
            if idx not in matched_det_indices:
                new_track = self.create_new_track(det)
                self.kalman_filters[new_track.track_id] = KalmanBoxTracker(det.bbox)
                self.active_tracks.append(new_track)
