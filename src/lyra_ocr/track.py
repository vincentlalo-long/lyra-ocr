"""Temporal text tracking across sampled video frames.

Groups spatial detections across time into continuous TextTrack streams,
calculates track-level features, and segments text transitions.
"""

from __future__ import annotations

import difflib
import math
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Optional


def bbox_iou(boxA: list[int] | list[float], boxB: list[int] | list[float]) -> float:
    """Calculate Intersection over Union (IoU) of two [xmin, ymin, xmax, ymax] boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    boxAArea = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    boxBArea = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])

    unionArea = boxAArea + boxBArea - interArea
    if unionArea <= 0.0:
        return 0.0
    return interArea / unionArea


def centroid_distance(
    boxA: list[int] | list[float],
    boxB: list[int] | list[float],
    frame_width: int,
    frame_height: int,
) -> float:
    """Calculate normalized Euclidean distance between box centroids [0.0, 1.0]."""
    cA_x = (boxA[0] + boxA[2]) / 2.0
    cA_y = (boxA[1] + boxA[3]) / 2.0
    cB_x = (boxB[0] + boxB[2]) / 2.0
    cB_y = (boxB[1] + boxB[3]) / 2.0

    diag = math.sqrt(frame_width**2 + frame_height**2)
    if diag <= 0.0:
        return 0.0
    dist = math.sqrt((cA_x - cB_x) ** 2 + (cA_y - cB_y) ** 2)
    return dist / diag


def string_similarity(a: str, b: str) -> float:
    """Calculate normalized character similarity ratio using SequenceMatcher."""
    s1 = a.strip().lower()
    s2 = b.strip().lower()
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return difflib.SequenceMatcher(None, s1, s2).ratio()


@dataclass
class DetectionOccurrence:
    frame_id: int
    timestamp_ms: int
    polygon: list[list[int]]
    bbox: list[int]
    text_raw: str
    text_norm: str
    confidence: float
    model_id: str
    script: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "timestamp_ms": self.timestamp_ms,
            "polygon": self.polygon,
            "bbox": self.bbox,
            "text_raw": self.text_raw,
            "text_norm": self.text_norm,
            "confidence": round(self.confidence, 4),
            "model_id": self.model_id,
            "script": self.script,
        }


@dataclass
class TextSegment:
    segment_id: int
    start_ms: int
    end_ms: int
    text: str
    confidence: float
    num_occurrences: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "text": self.text,
            "confidence": round(self.confidence, 4),
            "num_occurrences": self.num_occurrences,
        }


@dataclass
class TextTrack:
    track_id: int
    start_ms: int = 0
    end_ms: int = 0
    bbox: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    occurrences: list[DetectionOccurrence] = field(default_factory=list)
    segments: list[TextSegment] = field(default_factory=list)
    n_text_changes: int = 0
    features: dict[str, Any] = field(default_factory=dict)
    classification: Optional[Any] = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "track_id": self.track_id,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "duration_ms": max(0, self.end_ms - self.start_ms),
            "bbox": self.bbox,
            "n_text_changes": self.n_text_changes,
            "features": self.features,
            "segments": [s.to_dict() for s in self.segments],
            "occurrences": [occ.to_dict() for occ in self.occurrences],
            "num_occurrences": len(self.occurrences),
        }
        if self.classification:
            d["label"] = self.classification.label
            d["confidence"] = round(self.classification.confidence, 4)
            d["reason"] = self.classification.reason
            d["decision"] = self.classification.decision
            d["qa_flags"] = self.classification.qa_flags
        return d


def compute_track_features(
    track: TextTrack,
    video_duration_ms: int,
    frame_width: int,
    frame_height: int,
) -> dict[str, Any]:
    """Compute temporal, spatial, and textual statistical metrics for a track."""
    v_dur = max(1, video_duration_ms)
    lifetime_ms = max(0, track.end_ms - track.start_ms)
    lifetime_ratio = lifetime_ms / v_dur
    start_time_ratio = track.start_ms / v_dur

    # Spatial statistics
    centroids = []
    aspect_ratios = []
    area_ratios = []
    frame_area = max(1, frame_width * frame_height)

    for occ in track.occurrences:
        b = occ.bbox
        cx = (b[0] + b[2]) / 2.0
        cy = (b[1] + b[3]) / 2.0
        centroids.append((cx / max(1, frame_width), cy / max(1, frame_height)))

        w = max(1, b[2] - b[0])
        h = max(1, b[3] - b[1])
        aspect_ratios.append(w / h)
        area_ratios.append((w * h) / frame_area)

    if centroids:
        mean_cx = sum(c[0] for c in centroids) / len(centroids)
        mean_cy = sum(c[1] for c in centroids) / len(centroids)
        spatial_variance = sum(
            (c[0] - mean_cx) ** 2 + (c[1] - mean_cy) ** 2 for c in centroids
        ) / len(centroids)
    else:
        mean_cx, mean_cy, spatial_variance = 0.5, 0.5, 0.0

    mean_aspect_ratio = (
        sum(aspect_ratios) / len(aspect_ratios) if aspect_ratios else 1.0
    )
    mean_area_ratio = sum(area_ratios) / len(area_ratios) if area_ratios else 0.0

    # Corner check (strictly in top/bottom corners, not horizontal center)
    is_corner = (
        (mean_cx < 0.20 or mean_cx > 0.80)
        and (mean_cy < 0.18 or mean_cy > 0.85)
    )

    # Dominant text
    if track.segments:
        # Segment with most occurrences
        best_seg = max(track.segments, key=lambda s: s.num_occurrences)
        dominant_text = best_seg.text
    elif track.occurrences:
        dominant_text = track.occurrences[0].text_norm
    else:
        dominant_text = ""

    avg_conf = (
        sum(o.confidence for o in track.occurrences) / len(track.occurrences)
        if track.occurrences
        else 0.0
    )

    features = {
        "lifetime_ms": lifetime_ms,
        "lifetime_ratio": round(lifetime_ratio, 4),
        "start_time_ratio": round(start_time_ratio, 4),
        "n_text_changes": track.n_text_changes,
        "spatial_variance": round(spatial_variance, 6),
        "mean_centroid": [round(mean_cx, 4), round(mean_cy, 4)],
        "aspect_ratio": round(mean_aspect_ratio, 3),
        "area_ratio": round(mean_area_ratio, 5),
        "is_corner": is_corner,
        "dominant_text": dominant_text,
        "avg_confidence": round(avg_conf, 4),
    }
    track.features = features
    return features


class Tracker:
    """Frame-by-frame tracker associating bounding geometries across temporal samples."""

    def __init__(
        self,
        frame_width: int = 1280,
        frame_height: int = 720,
        iou_threshold: float = 0.30,
        centroid_threshold: float = 0.08,
        max_gap_ms: int = 2000,
    ) -> None:
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.iou_threshold = iou_threshold
        self.centroid_threshold = centroid_threshold
        self.max_gap_ms = max_gap_ms

        self.next_track_id = 1
        self.active_tracks: list[TextTrack] = []
        self.finished_tracks: list[TextTrack] = []

    def update(self, frame_data: dict[str, Any]) -> None:
        """Process a single frame OCR result and associate boxes to tracks."""
        frame_id = frame_data.get("frame_id", 0)
        timestamp_ms = frame_data.get("timestamp_ms", 0)
        raw_boxes = frame_data.get("boxes", [])

        # Parse detections (backward-compatible with legacy schema providing 'text')
        detections: list[DetectionOccurrence] = []
        for b in raw_boxes:
            if "text_raw" not in b and "text" not in b:
                raise ValueError(
                    f"Invalid detection schema in frame {frame_id}: missing 'text' or 'text_raw' in {b}"
                )
            raw_text = b.get("text_raw") or b.get("text") or ""
            norm_text = b.get("text_norm") or unicodedata.normalize("NFC", raw_text)
            model_id = b.get("model_id", "unknown")

            detections.append(
                DetectionOccurrence(
                    frame_id=frame_id,
                    timestamp_ms=timestamp_ms,
                    polygon=b.get("polygon", []),
                    bbox=b.get("bbox", [0, 0, 0, 0]),
                    text_raw=raw_text,
                    text_norm=norm_text,
                    confidence=float(b.get("confidence", 0.0)),
                    model_id=model_id,
                    script=b.get("script", "latin"),
                )
            )

        matched_tracks: set[int] = set()
        unmatched_detections: list[DetectionOccurrence] = []

        # Greedy bipartite matching based on spatial similarity
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

                # Matching condition: high IoU OR very close centroids
                if iou >= self.iou_threshold or (
                    iou >= 0.15 and c_dist <= self.centroid_threshold
                ):
                    score = iou + (1.0 - c_dist)
                    if score > best_score:
                        best_score = score
                        best_track = tr

            if best_track is not None:
                matched_tracks.add(best_track.track_id)
                self._add_occurrence_to_track(best_track, det)
            else:
                unmatched_detections.append(det)

        # Handle active tracks that were NOT matched in this frame
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
            new_track = TextTrack(
                track_id=self.next_track_id,
                start_ms=det.timestamp_ms,
                end_ms=det.timestamp_ms,
                bbox=det.bbox,
                occurrences=[det],
                segments=[
                    TextSegment(
                        segment_id=0,
                        start_ms=det.timestamp_ms,
                        end_ms=det.timestamp_ms,
                        text=det.text_norm,
                        confidence=det.confidence,
                        num_occurrences=1,
                    )
                ],
                n_text_changes=0,
            )
            self.next_track_id += 1
            self.active_tracks.append(new_track)

    def _add_occurrence_to_track(
        self, track: TextTrack, det: DetectionOccurrence
    ) -> None:
        track.occurrences.append(det)
        track.end_ms = det.timestamp_ms
        track.bbox = det.bbox

        # Check text continuity with latest segment
        if track.segments:
            latest_seg = track.segments[-1]
            sim = string_similarity(det.text_norm, latest_seg.text)

            # Same text continuing (allow minor OCR character jitter >= 0.85)
            if sim >= 0.85 or det.text_norm == latest_seg.text:
                latest_seg.end_ms = det.timestamp_ms
                latest_seg.num_occurrences += 1
                latest_seg.confidence = (
                    latest_seg.confidence * (latest_seg.num_occurrences - 1)
                    + det.confidence
                ) / latest_seg.num_occurrences
            else:
                # Text content changed within this spatial region!
                track.n_text_changes += 1
                new_seg = TextSegment(
                    segment_id=len(track.segments),
                    start_ms=det.timestamp_ms,
                    end_ms=det.timestamp_ms,
                    text=det.text_norm,
                    confidence=det.confidence,
                    num_occurrences=1,
                )
                track.segments.append(new_seg)

    def finalize(self, video_duration_ms: int = 0) -> list[TextTrack]:
        """Close all remaining active tracks and compute statistical features."""
        for tr in self.active_tracks:
            self.finished_tracks.append(tr)
        self.active_tracks = []

        # Sort by start_ms
        self.finished_tracks.sort(key=lambda t: t.start_ms)

        for tr in self.finished_tracks:
            compute_track_features(
                tr,
                video_duration_ms=video_duration_ms,
                frame_width=self.frame_width,
                frame_height=self.frame_height,
            )

        return self.finished_tracks

    @property
    def all_tracks(self) -> list[TextTrack]:
        return self.finished_tracks + self.active_tracks


def track_ocr_results(
    frames_ocr_data: list[dict[str, Any]] | dict[str, Any],
    video_duration_ms: int = 0,
    frame_width: int = 1280,
    frame_height: int = 720,
    iou_threshold: float = 0.30,
    max_gap_ms: int = 2000,
) -> list[TextTrack]:
    """Convenience helper to run Tracker over a list of frame OCR dictionaries (or the full ocr_raw payload)."""
    if isinstance(frames_ocr_data, dict):
        frames_list = frames_ocr_data.get("frames", [])
        if not video_duration_ms and "probe" in frames_ocr_data:
            dur_sec = frames_ocr_data["probe"].get("duration", 0)
            video_duration_ms = int(dur_sec * 1000)
    else:
        frames_list = frames_ocr_data

    tracker = Tracker(
        frame_width=frame_width,
        frame_height=frame_height,
        iou_threshold=iou_threshold,
        max_gap_ms=max_gap_ms,
    )
    for frame in frames_list:
        tracker.update(frame)
    return tracker.finalize(video_duration_ms=video_duration_ms)
