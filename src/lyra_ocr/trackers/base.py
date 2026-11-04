"""Base class and common interfaces for video text tracking strategies."""

from __future__ import annotations

import unicodedata
from abc import ABC, abstractmethod
from typing import Any

from lyra_ocr.track import (
    DetectionOccurrence,
    TextSegment,
    TextTrack,
    compute_track_features,
    string_similarity,
)


class BaseTracker(ABC):
    """Abstract base class for all text tracking strategies."""

    def __init__(
        self,
        frame_width: int = 1280,
        frame_height: int = 720,
        max_gap_ms: int = 2000,
        **kwargs: Any,
    ) -> None:
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.max_gap_ms = max_gap_ms
        self.next_track_id = 1
        self.active_tracks: list[TextTrack] = []
        self.finished_tracks: list[TextTrack] = []

    def parse_detections(self, frame_data: dict[str, Any]) -> list[DetectionOccurrence]:
        """Extract and normalize DetectionOccurrence instances from raw frame data."""
        frame_id = frame_data.get("frame_id", 0)
        timestamp_ms = frame_data.get("timestamp_ms", 0)
        raw_boxes = frame_data.get("boxes", [])

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
        return detections

    def add_occurrence_to_track(
        self,
        track: TextTrack,
        det: DetectionOccurrence,
        smooth_box: bool = True,
    ) -> None:
        """Add a detection occurrence to a track and update text continuity segments."""
        track.end_ms = det.timestamp_ms
        if smooth_box:
            # Weighted average smoothing of bounding box [xmin, ymin, xmax, ymax]
            track.bbox = [
                int(0.7 * track.bbox[0] + 0.3 * det.bbox[0]),
                int(0.7 * track.bbox[1] + 0.3 * det.bbox[1]),
                int(0.7 * track.bbox[2] + 0.3 * det.bbox[2]),
                int(0.7 * track.bbox[3] + 0.3 * det.bbox[3]),
            ]
        else:
            track.bbox = det.bbox
        track.occurrences.append(det)

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

    def create_new_track(self, det: DetectionOccurrence) -> TextTrack:
        """Create and return a new TextTrack instance for an unmatched detection."""
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
        return new_track

    @abstractmethod
    def update(self, frame_data: dict[str, Any]) -> None:
        """Process a single frame OCR result and associate boxes to tracks."""
        pass

    def finalize(self, video_duration_ms: int = 0) -> list[TextTrack]:
        """Close active tracks, sort chronologically, and compute statistical features."""
        for tr in self.active_tracks:
            self.finished_tracks.append(tr)
        self.active_tracks = []

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
