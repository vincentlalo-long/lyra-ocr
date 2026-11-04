"""Temporal text tracking across sampled video frames.

Groups spatial detections across time into continuous TextTrack streams,
calculates track-level features, and segments text transitions.
"""

from __future__ import annotations

import difflib
import math
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

    distinct_changes = 0
    if len(track.segments) >= 2:
        for idx in range(1, len(track.segments)):
            prev_t = track.segments[idx - 1].text
            curr_t = track.segments[idx].text
            if string_similarity(prev_t, curr_t) < 0.65:
                distinct_changes += 1

    features = {
        "lifetime_ms": lifetime_ms,
        "lifetime_ratio": round(lifetime_ratio, 4),
        "start_time_ratio": round(start_time_ratio, 4),
        "n_text_changes": track.n_text_changes,
        "distinct_text_changes": distinct_changes,
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


def _track_dominant_text(track: TextTrack) -> str:
    feat_text = (track.features or {}).get("dominant_text", "")
    if feat_text:
        return feat_text
    if track.segments:
        return max(track.segments, key=lambda s: s.num_occurrences).text
    if track.occurrences:
        return track.occurrences[0].text_norm
    return ""


def _merge_occurrences(dst: TextTrack, src: TextTrack) -> None:
    combined = sorted(
        dst.occurrences + src.occurrences,
        key=lambda o: (o.timestamp_ms, o.frame_id),
    )
    seen: set[tuple[int, tuple[int, ...]]] = set()
    occurrences: list[DetectionOccurrence] = []
    for occ in combined:
        key = (occ.frame_id, tuple(occ.bbox))
        if key in seen:
            continue
        seen.add(key)
        occurrences.append(occ)

    dst.occurrences = occurrences
    dst.start_ms = min(o.timestamp_ms for o in occurrences)
    dst.end_ms = max(o.timestamp_ms for o in occurrences)
    dst.bbox = occurrences[-1].bbox
    dst.segments = []
    dst.n_text_changes = 0

    for occ in occurrences:
        if not dst.segments:
            dst.segments.append(
                TextSegment(
                    segment_id=0,
                    start_ms=occ.timestamp_ms,
                    end_ms=occ.timestamp_ms,
                    text=occ.text_norm,
                    confidence=occ.confidence,
                    num_occurrences=1,
                )
            )
            continue

        latest = dst.segments[-1]
        if string_similarity(occ.text_norm, latest.text) >= 0.85:
            latest.end_ms = occ.timestamp_ms
            latest.num_occurrences += 1
            latest.confidence = (
                latest.confidence * (latest.num_occurrences - 1) + occ.confidence
            ) / latest.num_occurrences
        else:
            dst.n_text_changes += 1
            dst.segments.append(
                TextSegment(
                    segment_id=len(dst.segments),
                    start_ms=occ.timestamp_ms,
                    end_ms=occ.timestamp_ms,
                    text=occ.text_norm,
                    confidence=occ.confidence,
                    num_occurrences=1,
                )
            )


def merge_fragmented_tracks(
    tracks: list[TextTrack],
    video_duration_ms: int = 0,
    frame_width: int = 1280,
    frame_height: int = 720,
    max_gap_ms: int = 800,
    iou_threshold: float = 0.35,
    centroid_threshold: float = 0.04,
    text_similarity_threshold: float = 0.90,
) -> list[TextTrack]:
    """Conservatively merge track fragments caused by short detection gaps.

    A merge requires the same spatial region *and* either near-identical text or
    a tiny OCR fragment, with a temporal gap no larger than ``max_gap_ms``. This
    removes 1-frame flicker fragments without collapsing repeated chorus lines
    that are separated by a longer gap.
    """
    if not tracks:
        return []

    ordered = sorted(tracks, key=lambda t: (t.start_ms, t.track_id))
    groups: list[TextTrack] = []

    for track in ordered:
        if not track.occurrences:
            continue

        track_text = _track_dominant_text(track)
        merged = False

        for group in reversed(groups):
            gap = track.start_ms - group.end_ms
            if gap > max_gap_ms or gap < -max_gap_ms:
                continue

            spatial_match = (
                bbox_iou(track.bbox, group.bbox) >= iou_threshold
                or centroid_distance(track.bbox, group.bbox, frame_width, frame_height)
                <= centroid_threshold
            )
            text_sim = string_similarity(track_text, _track_dominant_text(group))
            tiny_fragment = min(len(track_text.strip()), len(_track_dominant_text(group).strip())) <= 2

            if spatial_match and (text_sim >= text_similarity_threshold or tiny_fragment):
                _merge_occurrences(group, track)
                merged = True
                break

        if not merged:
            groups.append(track)

    groups.sort(key=lambda t: t.start_ms)
    for track in groups:
        compute_track_features(
            track,
            video_duration_ms=video_duration_ms,
            frame_width=frame_width,
            frame_height=frame_height,
        )
    return groups


class Tracker:
    """Frame-by-frame tracker associating bounding geometries across temporal samples.

    Maintained for backward compatibility. Instantiates GreedyTracker.
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        from lyra_ocr.trackers.greedy import GreedyTracker

        return GreedyTracker(*args, **kwargs)


def track_ocr_results(
    frames_ocr_data: list[dict[str, Any]] | dict[str, Any],
    video_duration_ms: int = 0,
    frame_width: int = 1280,
    frame_height: int = 720,
    iou_threshold: float = 0.30,
    max_gap_ms: int = 2000,
    tracker_name: str = "greedy",
    **kwargs: Any,
) -> list[TextTrack]:
    """Convenience helper to run a tracker over frame OCR data (list or ocr_raw dict)."""
    from lyra_ocr.trackers import get_tracker

    if isinstance(frames_ocr_data, dict):
        frames = frames_ocr_data.get("frames", [])
        if not video_duration_ms and "probe" in frames_ocr_data:
            dur_sec = frames_ocr_data["probe"].get("duration", 0)
            video_duration_ms = int(dur_sec * 1000)
    else:
        frames = frames_ocr_data

    tracker = get_tracker(
        name=tracker_name,
        frame_width=frame_width,
        frame_height=frame_height,
        iou_threshold=iou_threshold,
        max_gap_ms=max_gap_ms,
        **kwargs,
    )
    for frame in frames:
        tracker.update(frame)
    return tracker.finalize(video_duration_ms=video_duration_ms)
