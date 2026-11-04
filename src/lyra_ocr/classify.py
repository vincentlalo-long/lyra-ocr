"""Rule-based track classifier for Phase 1.

Classifies TextTrack instances into standardized taxonomy:
- lyric: core song lyrics (decision: accept)
- channel: watermark, channel logo, artist handle (decision: reject)
- song_title: introductory title cards (decision: reject)
- supplementary: UI text, subscribe buttons, credit blocks (decision: reject)
- scene_text: text on background scenery moving with camera (decision: reject)
- unknown: ambiguous items routed to human verification (decision: review)
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from lyra_ocr.track import TextTrack, compute_track_features, string_similarity


@dataclass
class ClassificationResult:
    label: str  # lyric, channel, song_title, supplementary, scene_text, unknown
    confidence: float
    reason: str
    decision: str  # accept, reject, review
    qa_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "reason": self.reason,
            "decision": self.decision,
            "qa_flags": self.qa_flags,
        }


SUPPLEMENTARY_KEYWORDS = [
    "subscribe",
    "subscribers",
    "subscriber",
    "follow us",
    "patreon",
    "instagram",
    "facebook",
    "twitter",
    "tiktok",
    "download",
    "stream now",
    "directed by",
    "produced by",
    "mixed by",
    "mastered by",
    "visualizer",
    "official video",
    "official audio",
]


def classify_track(
    track: TextTrack,
    metadata: Optional[dict[str, Any]] = None,
    video_duration_ms: int = 0,
    frame_width: int = 1280,
    frame_height: int = 720,
) -> ClassificationResult:
    """Classify a single TextTrack into the standardized taxonomy using rule heuristics."""
    meta = metadata or {}
    v_channel = meta.get("channel", "").strip().lower()
    v_title = meta.get("title", "").strip().lower()

    # Ensure features exist
    if not track.features:
        compute_track_features(
            track,
            video_duration_ms=video_duration_ms,
            frame_width=frame_width,
            frame_height=frame_height,
        )

    feat = track.features
    lifetime_ratio = feat.get("lifetime_ratio", 0.0)
    start_time_ratio = feat.get("start_time_ratio", 0.0)
    n_text_changes = feat.get("n_text_changes", 0)
    spatial_variance = feat.get("spatial_variance", 0.0)
    aspect_ratio = feat.get("aspect_ratio", 1.0)
    is_corner = feat.get("is_corner", False)
    dominant_text = feat.get("dominant_text", "").strip()
    dom_lower = dominant_text.lower()
    duration_ms = track.end_ms - track.start_ms

    qa_flags: list[str] = []

    # Rule 1: Supplementary / Social Keywords
    for kw in SUPPLEMENTARY_KEYWORDS:
        if kw in dom_lower:
            res = ClassificationResult(
                label="supplementary",
                confidence=0.96,
                reason=f"Contains supplementary/social/credit keyword '{kw}'",
                decision="reject",
                qa_flags=qa_flags,
            )
            track.classification = res
            return res

    # Rule 2: Erratic Scene Text (High Spatial Variance)
    if spatial_variance > 0.04:
        res = ClassificationResult(
            label="scene_text",
            confidence=0.92,
            reason=f"High spatial variance ({spatial_variance:.4f}) indicating trajectory drift",
            decision="reject",
            qa_flags=["high_spatial_variance"],
        )
        track.classification = res
        return res

    # Rule 3: Channel Watermark / Artist Attribution
    if v_channel:
        sim_chan = string_similarity(dom_lower, v_channel)
        chan_tokens = set(re.findall(r"\w+", v_channel))
        dom_tokens = set(re.findall(r"\w+", dom_lower))
        sig_chan_tokens = {t for t in chan_tokens if len(t) >= 3}
        token_chan_match = bool(
            sig_chan_tokens
            and (
                sig_chan_tokens.issubset(dom_tokens)
                or (
                    sig_chan_tokens.intersection(dom_tokens)
                    and n_text_changes == 0
                    and len(dom_tokens) <= 5
                )
            )
        )
        chan_len_ratio = len(v_channel) / max(1, len(dom_lower))
        if sim_chan >= 0.75 or token_chan_match or (v_channel in dom_lower and chan_len_ratio >= 0.4):
            res = ClassificationResult(
                label="channel",
                confidence=0.98,
                reason=f"Matched video channel/artist metadata '{v_channel}'",
                decision="reject",
                qa_flags=["metadata_channel_match"],
            )
            track.classification = res
            return res

    # Persistent invariant text (> 45% of total video)
    if lifetime_ratio >= 0.45 and n_text_changes <= 1:
        res = ClassificationResult(
            label="channel",
            confidence=0.95,
            reason=f"Persistent invariant text across {lifetime_ratio*100:.1f}% of video duration",
            decision="reject",
            qa_flags=["long_lifetime_invariant"],
        )
        track.classification = res
        return res

    # Channel handle symbols (@handle or #handle)
    if dominant_text.startswith("@") or (dominant_text.startswith("#") and len(dominant_text) > 3):
        res = ClassificationResult(
            label="channel",
            confidence=0.90,
            reason="Contains channel handle prefix (@ or #)",
            decision="reject",
            qa_flags=["social_handle_tag"],
        )
        track.classification = res
        return res

    # Rule 4: Intro Title Card / Song Title
    if v_title:
        sim_title = string_similarity(dom_lower, v_title)
        title_len_ratio = len(v_title) / max(1, len(dom_lower))
        title_tokens = set(re.findall(r"\w+", v_title))
        dom_tokens = set(re.findall(r"\w+", dom_lower))
        sig_title_tokens = {
            t for t in title_tokens if len(t) >= 3 and t not in {"the", "and", "for", "with", "this"}
        }
        token_title_match = bool(
            sig_title_tokens
            and dom_tokens.issubset(sig_title_tokens)
            and n_text_changes == 0
            and len(dom_tokens) <= 4
        )
        is_title_match = (
            sim_title >= 0.75
            or token_title_match
            or (v_title in dom_lower and title_len_ratio >= 0.6)
        )
        if is_title_match and (start_time_ratio <= 0.25 or n_text_changes == 0):
            res = ClassificationResult(
                label="song_title",
                confidence=0.98,
                reason=f"Title text matching video title metadata '{v_title}'",
                decision="reject",
                qa_flags=["metadata_title_match"],
            )
            track.classification = res
            return res

    # Static text strictly in intro (< 10% of duration or <= 12 seconds)
    if start_time_ratio <= 0.05 and track.end_ms <= 15000 and n_text_changes <= 1 and duration_ms <= 10000:
        if 2000 <= duration_ms <= 9000 and not is_corner:
            res = ClassificationResult(
                label="song_title",
                confidence=0.88,
                reason="Static title card bounded strictly within video intro",
                decision="reject",
                qa_flags=["intro_bounded_card"],
            )
            track.classification = res
            return res

    # Rule 5: Lyric Stream (Recurring Text Changes in Stable Region)
    if n_text_changes >= 2 and spatial_variance <= 0.02:
        res = ClassificationResult(
            label="lyric",
            confidence=0.98,
            reason=f"Recurring text changes ({n_text_changes} changes) in stable region (lyric stream)",
            decision="accept",
            qa_flags=[],
        )
        track.classification = res
        return res

    # Rule 6: Transient Single Lyric Line in Active Portion
    if n_text_changes <= 1:
        # Typical lyric line duration is 1.5s - 7.5s
        if 1500 <= duration_ms <= 8500 and not is_corner and aspect_ratio >= 1.5:
            res = ClassificationResult(
                label="lyric",
                confidence=0.90,
                reason="Transient text line within typical lyric duration and aspect ratio",
                decision="accept",
                qa_flags=["single_cue_track"],
            )
            track.classification = res
            return res

    # Rule 7: Ambiguous / Unknown Fallback
    res = ClassificationResult(
        label="unknown",
        confidence=0.50,
        reason="Ambiguous trajectory / insufficient observations for deterministic rule match",
        decision="review",
        qa_flags=["needs_human_audit"],
    )
    track.classification = res
    return res


def classify_all_tracks(
    tracks: list[TextTrack],
    metadata: Optional[dict[str, Any]] = None,
    video_duration_ms: int = 0,
    frame_width: int = 1280,
    frame_height: int = 720,
) -> list[ClassificationResult]:
    """Run rule-based classification across all tracks in a pipeline."""
    results = []
    for tr in tracks:
        res = classify_track(
            tr,
            metadata=metadata,
            video_duration_ms=video_duration_ms,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        results.append(res)
    return results
