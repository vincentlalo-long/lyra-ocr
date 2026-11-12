"""Rule-based track classifier for Phase 1.

Classifies TextTrack instances into the final binary schema:
- lyric: core song lyrics (decision: accept)
- non_lyric: every distractor subtype (decision: reject or review)

The ``reason`` and ``qa_flags`` fields preserve the diagnostic subtype
(watermark, title card, supplementary text, scene text, ambiguous) without
making it part of the ground-truth label space.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from lyra_ocr.track import (
    TextTrack,
    compute_track_features,
    effective_duration_ms,
    string_similarity,
)


@dataclass
class ClassificationResult:
    label: str  # lyric, non_lyric
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


CHAT_CONTENT_RE = re.compile(
    r"(?:"
    r"[:;=8][\-^o']?[)(\]\[DPpOo/\\|]"  # emoticons such as :) or :D
    r"|[oO0]\s*\(\s*['\u2019]?\^\s*\)\s*[oO0]"  # o('^)o style
    r"|!{2,}|\?!|!\?"  # excited chat punctuation
    r"|\bchat\b|\blivestream\b|\bstream\b"  # stream/chat vocabulary
    r")",
    re.IGNORECASE,
)


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
    "official lyric video",
    "lyric video",
    "lyrics video",
    "music video",
    "new video",
    "video every",
    "link in bio",
    "link in description",
    "watch now",
    "out now",
    "presents",
    "sound only",
]


def classify_track(
    track: TextTrack,
    metadata: Optional[dict[str, Any]] = None,
    video_duration_ms: int = 0,
    frame_width: int = 1280,
    frame_height: int = 720,
    sample_interval_ms: int = 0,
    vocal_profile: Optional[Any] = None,
) -> ClassificationResult:
    """Classify a single TextTrack into lyric vs non-lyric using rule heuristics."""
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
    duration_ms = effective_duration_ms(track.start_ms, track.end_ms, sample_interval_ms)
    lifetime_ratio = feat.get("lifetime_ratio", 0.0)
    if sample_interval_ms > 0 and video_duration_ms > 0:
        lifetime_ratio = duration_ms / max(1, video_duration_ms)
    start_time_ratio = (
        track.start_ms / max(1, video_duration_ms)
        if video_duration_ms > 0
        else feat.get("start_time_ratio", 0.0)
    )
    n_text_changes = feat.get("n_text_changes", 0)
    spatial_variance = feat.get("spatial_variance", 0.0)
    aspect_ratio = feat.get("aspect_ratio", 1.0)
    area_ratio = feat.get("area_ratio")
    if area_ratio is None:
        _bb = track.bbox or [0, 0, 0, 0]
        area_ratio = (max(0, _bb[2] - _bb[0]) * max(0, _bb[3] - _bb[1])) / max(
            1, frame_width * frame_height
        )
    is_corner = feat.get("is_corner", False)
    dominant_text = feat.get("dominant_text", "").strip()
    dom_lower = dominant_text.lower()
    dom_tokens = set(re.findall(r"\w+", dom_lower))
    mean_centroid = feat.get("mean_centroid")
    if mean_centroid and len(mean_centroid) >= 2:
        mean_cx, mean_cy = float(mean_centroid[0]), float(mean_centroid[1])
    elif track.bbox and len(track.bbox) >= 4:
        mean_cx = ((track.bbox[0] + track.bbox[2]) / 2.0) / max(1, frame_width)
        mean_cy = ((track.bbox[1] + track.bbox[3]) / 2.0) / max(1, frame_height)
    else:
        mean_cx, mean_cy = 0.5, 0.5
    distinct_changes = feat.get("distinct_text_changes", n_text_changes)
    word_count = len(dominant_text.split())
    layer_kind = feat.get("layer_kind", "sparse")
    in_lyric_layer = (layer_kind == "lyric")
    # Spatial prior: active reading zone (center or lower-third), not in extreme corners
    in_reading_zone = in_lyric_layer or (
        not is_corner
        and 0.10 <= mean_cx <= 0.90
        and 0.20 <= mean_cy <= 0.95
        and aspect_ratio >= 1.0
    )
    # Card-shaped text: short overlays strictly bounded to the intro/outro window
    # are typically title cards or credit slides. Casing (uppercase vs lowercase)
    # is never used to judge lyrics, as song lyrics are frequently styled in all-caps.
    looks_like_card = word_count <= 4 and (
        (start_time_ratio <= 0.01 and track.start_ms == 0)
        or mean_cy <= 0.65
        or (area_ratio >= 0.03 and aspect_ratio <= 3.5)
    )
    end_time_ratio = start_time_ratio + lifetime_ratio
    only_intro_outro = (not in_lyric_layer) and (
        (end_time_ratio <= 0.08 and looks_like_card)
        or (start_time_ratio >= 0.95 and looks_like_card)
    ) and (video_duration_ms > 0 or lifetime_ratio > 0)

    qa_flags: list[str] = []

    # Rule 0: Isolated single fragment or credit word
    if word_count == 1 and dom_lower in ("sound", "only", "presents"):
        res = ClassificationResult(
            label="non_lyric",
            confidence=0.85,
            reason=f"Isolated fragment or title credit word '{dominant_text}'",
            decision="reject" if dom_lower == "presents" else "review",
            qa_flags=["isolated_fragment_word"],
        )
        track.classification = res
        return res

    # Rule 1: Supplementary / Social Keywords
    for kw in SUPPLEMENTARY_KEYWORDS:
        if kw in dom_lower:
            res = ClassificationResult(
                label="non_lyric",
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
            label="non_lyric",
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
                label="non_lyric",
                confidence=0.98,
                reason=f"Matched video channel/artist metadata '{v_channel}'",
                decision="reject",
                qa_flags=["metadata_channel_match"],
            )
            track.classification = res
            return res

    # Rule 3b: Right-rail chat / stream overlay.
    # Live-stream comments occupy a narrow right-hand column, last only a few
    # seconds, and are much wider than they are tall. They are not lyrics even
    # though they can fall inside a naive reading-zone test.
    if (
        mean_cx >= 0.78
        and area_ratio <= 0.02
        and duration_ms <= 5000
        and aspect_ratio >= 3.0
    ):
        res = ClassificationResult(
            label="non_lyric",
            confidence=0.72,
            reason="Right-rail short-lived wide overlay (chat/stream comment)",
            decision="review",
            qa_flags=["chat_overlay_suspect"],
        )
        track.classification = res
        return res

    # Rule 3c: Chat-like content cues anywhere on screen. Review, never reject:
    # words like "stream" can legitimately appear in a lyric, but emoticons and
    # repeated exclamation marks are strong chat signatures.
    if CHAT_CONTENT_RE.search(dominant_text):
        res = ClassificationResult(
            label="non_lyric",
            confidence=0.62,
            reason="Chat/stream content cue in on-screen text",
            decision="review",
            qa_flags=["chat_like_content"],
        )
        track.classification = res
        return res

    # Persistent invariant text (> 45% of total video)
    if lifetime_ratio >= 0.45 and n_text_changes <= 1:
        res = ClassificationResult(
            label="non_lyric",
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
            label="non_lyric",
            confidence=0.90,
            reason="Contains channel handle prefix (@ or #)",
            decision="reject",
            qa_flags=["social_handle_tag"],
        )
        track.classification = res
        return res

    # Vocal Activity Prior: in silent intro/outro sections without lyric layer, reject static overlay
    if vocal_profile is not None and not in_lyric_layer:
        from lyra_ocr.audio.vad import vocal_active_ratio
        vocal_ratio = vocal_active_ratio(vocal_profile, track.start_ms, track.end_ms)
        if vocal_ratio < 0.05 and (start_time_ratio < 0.20 or start_time_ratio > 0.80):
            res = ClassificationResult(
                label="non_lyric",
                confidence=0.88,
                reason=f"Silent section overlay (vocal active ratio {vocal_ratio:.2f})",
                decision="reject",
                qa_flags=["vad_silent_overlay"],
            )
            track.classification = res
            return res

    # Rule 4: Intro Title Card / Song Title
    if v_title:
        sim_title = string_similarity(dom_lower, v_title)
        title_len_ratio = len(v_title) / max(1, len(dom_lower))
        title_tokens = set(re.findall(r"\w+", v_title))
        sig_title_tokens = {
            t for t in title_tokens if len(t) >= 3 and t not in {"the", "and", "for", "with", "this"}
        }
        token_title_match = bool(
            sig_title_tokens
            and dom_tokens.issubset(sig_title_tokens)
            and n_text_changes == 0
            and len(dom_tokens) <= 4
            and (len(dom_tokens) / len(sig_title_tokens) >= 0.50)
        )
        is_title_match = (
            sim_title >= 0.75
            or token_title_match
            or (v_title in dom_lower and title_len_ratio >= 0.6)
        )
        if is_title_match and start_time_ratio <= 0.25:
            res = ClassificationResult(
                label="non_lyric",
                confidence=0.98,
                reason=f"Title text matching video title metadata '{v_title}'",
                decision="reject",
                qa_flags=["metadata_title_match"],
            )
            track.classification = res
            return res
        if is_title_match:
            # Matching the title mid-video is often a repeated hook/chorus, not an
            # intro card: if in reading zone and transient, classify as lyric for review.
            if in_reading_zone and 500 <= duration_ms <= 16000 and not only_intro_outro:
                res = ClassificationResult(
                    label="lyric",
                    confidence=0.75,
                    reason=(
                        f"Mid-video text matching video title metadata '{v_title}' "
                        "in reading zone (likely chorus/hook)"
                    ),
                    decision="review",
                    qa_flags=["metadata_title_match", "title_match_outside_intro", "potential_chorus"],
                )
            else:
                res = ClassificationResult(
                    label="non_lyric",
                    confidence=0.55,
                    reason=(
                        f"Title text matching video title metadata '{v_title}' "
                        "outside the intro window"
                    ),
                    decision="review",
                    qa_flags=["metadata_title_match", "title_match_outside_intro"],
                )
            track.classification = res
            return res

    # Rule L2: Static Overlay / Watermark Layer
    if layer_kind == "overlay":
        res = ClassificationResult(
            label="non_lyric",
            confidence=0.92,
            reason=f"Static overlay/watermark layer (id={feat.get('layer_id')})",
            decision="reject",
            qa_flags=["static_overlay_layer"],
        )
        track.classification = res
        return res

    # Check natural language characters
    has_text = any(
        ch.isalpha() or "\u4e00" <= ch <= "\u9fff" or "\u3040" <= ch <= "\u30ff" or "\uac00" <= ch <= "\ud7af"
        for ch in dominant_text
    )

    # Rule L1: Confirmed Lyric Layer Member
    if (
        in_lyric_layer
        and has_text
        and len(dominant_text.strip()) >= 2
        and not only_intro_outro
        and duration_ms >= 400
    ):
        res = ClassificationResult(
            label="lyric",
            confidence=0.95,
            reason=f"Member of dynamic lyric layer (id={feat.get('layer_id')})",
            decision="accept",
            qa_flags=["lyric_layer_member"],
        )
        track.classification = res
        return res

    # Static text strictly in intro (< 10% of duration or <= 12 seconds)
    if not in_lyric_layer and start_time_ratio <= 0.05 and track.end_ms <= 15000 and n_text_changes <= 1 and duration_ms <= 10000:
        if 2000 <= duration_ms <= 9000 and not is_corner and looks_like_card:
            res = ClassificationResult(
                label="non_lyric",
                confidence=0.88,
                reason="Static title card bounded strictly within video intro",
                decision="reject",
                qa_flags=["intro_bounded_card"],
            )
            track.classification = res
            return res

    # Rule 5: Lyric Stream (Recurring Distinct Text Changes in Stable Region)
    if distinct_changes >= 2 and spatial_variance <= 0.02 and layer_kind != "overlay":
        if only_intro_outro:
            # Intro/credit carousels also change text; never auto-accept them.
            res = ClassificationResult(
                label="non_lyric",
                confidence=0.55,
                reason="Recurring text change confined to intro/outro (possible credit carousel)",
                decision="review",
                qa_flags=["only_intro_outro"],
            )
        else:
            res = ClassificationResult(
                label="lyric",
                confidence=0.98,
                reason=f"Recurring distinct text changes ({distinct_changes} changes) in stable region (lyric stream)",
                decision="accept",
                qa_flags=[],
            )
        track.classification = res
        return res

    # Rule 5a: Lower-third lyric line with OCR character fluctuation.
    # A real lyric line sampled at low fps can alternate between OCR variants, so
    # n_text_changes is often > 1 without being a new text object. The reading
    # zone (rather than the corner/side prior) is what separates it from overlay
    # noise; readability is judged later by the OCR/text-quality gates.
    if (
        500 <= duration_ms <= 16000
        and in_reading_zone
        and not is_corner
        and word_count >= 2
        and n_text_changes >= 1
        and not only_intro_outro
    ):
        res = ClassificationResult(
            label="lyric",
            confidence=0.90,
            reason="Transient lower-third line with OCR character fluctuation",
            decision="accept",
            qa_flags=["ocr_fluctuation_lyric_line"],
        )
        track.classification = res
        return res

    # Rule 5b: Static text with OCR character fluctuation in a corner/side region.
    # Without metadata title evidence this is not confidently non-lyric,
    # so route it to review instead of guessing. The reading zone is excluded so
    # lower-third lyric lines are not mistaken for corner overlays.
    if (
        distinct_changes == 0
        and n_text_changes >= 1
        and not in_reading_zone
        and (mean_cx < 0.25 or mean_cx > 0.75 or is_corner or mean_cy > 0.95)
    ):
        res = ClassificationResult(
            label="non_lyric",
            confidence=0.60,
            reason="Static corner/side text with OCR character fluctuation; needs review",
            decision="review",
            qa_flags=["ocr_fluctuation_static"],
        )
        track.classification = res
        return res

    # Rule 6: Transient Single Lyric Line in Active Reading Zone
    if n_text_changes <= 1:
        # Check natural language characters (Latin / CJK / Vietnamese)
        has_text = any(
            ch.isalpha() or "\u4e00" <= ch <= "\u9fff" or "\u3040" <= ch <= "\u30ff"
            for ch in dominant_text
        )
        # Spatial prior: horizontally centered, within active reading zone (center or lower-third),
        # not tucked into extreme corners
        in_reading_zone = (
            not is_corner
            and 0.10 <= mean_cx <= 0.90
            and 0.20 <= mean_cy <= 0.95
            and aspect_ratio >= 1.0
        )
        if (
            500 <= duration_ms <= 16000
            and in_reading_zone
            and has_text
            and len(dominant_text.strip()) >= 2
            and not only_intro_outro
        ):
            res = ClassificationResult(
                label="lyric",
                confidence=0.92,
                reason="Transient text line within typical lyric reading zone and duration",
                decision="accept",
                qa_flags=["single_cue_track"],
            )
            track.classification = res
            return res

    # Rule 6b: Isolated single-observation text.
    # A real subtitle/lyric line persists across several sampled frames; a
    # one-frame detection is background noise or transient scene text, unless
    # it is a full multi-word phrase within the lyric reading zone.
    if len(track.occurrences) <= 1 and n_text_changes == 0 and not is_corner:
        if (
            len(dominant_text.split()) >= 2
            and in_reading_zone
            and aspect_ratio >= 2.0
            and has_text
            and not only_intro_outro
        ):
            res = ClassificationResult(
                label="lyric",
                confidence=0.85,
                reason="Single-frame sampled phrase within lyric reading zone",
                decision="accept",
                qa_flags=["single_frame_lyric_cue"],
            )
            track.classification = res
            return res
        res = ClassificationResult(
            label="non_lyric",
            confidence=0.80,
            reason="Isolated single-frame text with no temporal persistence",
            decision="reject",
            qa_flags=["single_frame_isolated"],
        )
        track.classification = res
        return res

    # Rule 7: Ambiguous Fallback
    res = ClassificationResult(
        label="non_lyric",
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
    sample_interval_ms: int = 0,
    vocal_profile: Optional[Any] = None,
) -> list[ClassificationResult]:
    """Run rule-based classification across all tracks in a pipeline."""
    # Discover spatial text layers across the video timeline if not already assigned
    from lyra_ocr.layers import discover_layers

    has_layers = any("layer_id" in (tr.features or {}) for tr in tracks)
    if not has_layers and tracks:
        discover_layers(
            tracks,
            frame_w=frame_width,
            frame_h=frame_height,
            video_duration_ms=video_duration_ms,
        )

    results = []
    for tr in tracks:
        res = classify_track(
            tr,
            metadata=metadata,
            video_duration_ms=video_duration_ms,
            frame_width=frame_width,
            frame_height=frame_height,
            sample_interval_ms=sample_interval_ms,
            vocal_profile=vocal_profile,
        )
        results.append(res)
    return results
