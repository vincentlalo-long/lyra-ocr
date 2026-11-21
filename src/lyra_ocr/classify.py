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

from lyra_ocr.config import DEFAULT_CLASSIFICATION_THRESHOLDS, ClassificationThresholds
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

_INTRO_CREDIT_PREFIX_RE = re.compile(
    r"(?:^(?:anime|edition|music|lyrics?|vocal(?:s)?|illust(?:ration)?|movie|video|animation|compilation|arranged?|arrangement|mix(?:ed)?|master(?:ing|ed)?|encoded|subtitles?|translated|sub(?:bed)?|edit(?:ed)?)\s*[:\-]|(?:^|[\-\|/:]\s*)by\s+(?!(?:the|a|an|my|your|our|his|her|their|this|that|all|now|far)\b)[A-Za-z0-9_\-]+)",
    re.IGNORECASE,
)

_SIGNOFF_WORD_RE = re.compile(r"^watching[!！.]*$", re.IGNORECASE)

_METADATA_ENTITY_STOPWORDS = frozenset({
    "the", "and", "for", "with", "this", "that", "from", "you", "are", "was",
    "not", "but", "all", "can", "one", "out", "her", "his", "our", "your",
})


def _is_intro_credit_text(text: str, *, in_lyric_layer: bool, word_count: int) -> bool:
    """Return True when ``text`` matches an intro/outro production-credit prefix or infix."""
    if not _INTRO_CREDIT_PREFIX_RE.search(text):
        return False
    if in_lyric_layer and word_count >= 5 and re.match(r"by\s+", text, re.IGNORECASE):
        return False
    return True


def _matches_metadata_entity(
    dom_lower: str,
    dom_tokens: set[str],
    entity: str,
    *,
    n_text_changes: int,
    cfg: ClassificationThresholds,
    require_full_entity_or_high_sim: bool = False,
) -> bool:
    """Return True when ``dom_lower`` matches channel or artist metadata."""
    if not entity:
        return False
    sim = string_similarity(dom_lower, entity)
    ent_tokens = set(re.findall(r"\w+", entity))
    sig_ent_tokens = {
        t
        for t in ent_tokens
        if len(t) >= cfg.channel_token_min_len and t not in _METADATA_ENTITY_STOPWORDS
    }
    partial_match = (
        not require_full_entity_or_high_sim
        and sig_ent_tokens.intersection(dom_tokens)
        and n_text_changes == 0
        and len(dom_tokens) <= cfg.channel_partial_max_dom_tokens
    )
    token_match = bool(
        sig_ent_tokens
        and (sig_ent_tokens.issubset(dom_tokens) or partial_match)
    )
    len_ratio = len(entity) / max(1, len(dom_lower))
    substring_match = (
        entity not in _METADATA_ENTITY_STOPWORDS
        and entity in dom_lower
        and len_ratio >= cfg.channel_substring_min_len_ratio
    )
    return (
        sim >= cfg.channel_min_similarity
        or token_match
        or substring_match
    )


def is_split_title_fragment(text: str, title: str) -> bool:
    """Return True when a 1-word cue/track is a split fragment of a multi-word video title."""
    stripped = (text or "").strip()
    if not title or not stripped or len(stripped.split()) != 1:
        return False
    clean_tokens = re.findall(r"\w+", stripped.lower())
    if len(clean_tokens) != 1 or len(clean_tokens[0]) < 2:
        return False
    title_tokens = set(re.findall(r"\w+", title.lower()))
    if len(title_tokens) < 2:
        return False
    return clean_tokens[0] in title_tokens


def is_boundary_distractor_text(
    text: str,
    *,
    title: str = "",
    artist: str = "",
    channel: str = "",
    in_lyric_layer: bool = False,
    n_text_changes: int = 0,
    thresholds: ClassificationThresholds | None = None,
) -> bool:
    """Return True when boundary text matches credit, sign-off, or artist/channel patterns."""
    cfg = thresholds or DEFAULT_CLASSIFICATION_THRESHOLDS
    stripped = (text or "").strip()
    if not stripped:
        return False
    dom_lower = stripped.lower()
    if any(kw in dom_lower for kw in SUPPLEMENTARY_KEYWORDS):
        return True
    word_count = len(stripped.split())
    if word_count == 1 and _SIGNOFF_WORD_RE.match(stripped):
        return True
    if _is_intro_credit_text(stripped, in_lyric_layer=in_lyric_layer, word_count=word_count):
        return True
    dom_tokens = set(re.findall(r"\w+", dom_lower))
    if _matches_metadata_entity(
        dom_lower,
        dom_tokens,
        channel.strip().lower(),
        n_text_changes=n_text_changes,
        cfg=cfg,
        require_full_entity_or_high_sim=True,
    ):
        return True
    if _matches_metadata_entity(
        dom_lower,
        dom_tokens,
        artist.strip().lower(),
        n_text_changes=n_text_changes,
        cfg=cfg,
        require_full_entity_or_high_sim=True,
    ):
        return True
    return False


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
    "thanks for watching",
    "thank you for watching",
    "color coded lyrics",
    "kan/rom/eng",
    "rom/eng",
    "romaji/español",
    "sub. español",
]


def classify_track(
    track: TextTrack,
    metadata: Optional[dict[str, Any]] = None,
    video_duration_ms: int = 0,
    frame_width: int = 1280,
    frame_height: int = 720,
    sample_interval_ms: int = 0,
    vocal_profile: Optional[Any] = None,
    all_tracks: Optional[list[TextTrack]] = None,
    *,
    thresholds: Optional[ClassificationThresholds] = None,
) -> ClassificationResult:
    """Classify a single TextTrack into lyric vs non-lyric using rule heuristics."""
    cfg = thresholds or DEFAULT_CLASSIFICATION_THRESHOLDS
    meta = metadata or {}
    v_channel = meta.get("channel", "").strip().lower()
    v_artist = meta.get("artist", "").strip().lower()
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
        and cfg.reading_zone_cx_min <= mean_cx <= cfg.reading_zone_cx_max
        and cfg.reading_zone_cy_min <= mean_cy <= cfg.reading_zone_cy_max
        and aspect_ratio >= cfg.reading_zone_min_aspect_ratio
    )
    # Card-shaped text: short overlays strictly bounded to the intro/outro window
    # are typically title cards or credit slides. Casing (uppercase vs lowercase)
    # is never used to judge lyrics, as song lyrics are frequently styled in all-caps.
    looks_like_card = word_count <= cfg.card_max_words and (
        (start_time_ratio <= cfg.card_start_ratio_max and track.start_ms == 0)
        or mean_cy <= cfg.card_cy_max
        or (area_ratio >= cfg.card_min_area_ratio and aspect_ratio <= cfg.card_max_aspect_ratio)
    )
    end_time_ratio = start_time_ratio + lifetime_ratio
    only_intro_outro = (not in_lyric_layer) and (
        (end_time_ratio <= cfg.intro_end_ratio_max and looks_like_card)
        or (start_time_ratio >= cfg.outro_start_ratio_min and looks_like_card)
    ) and (video_duration_ms > 0 or lifetime_ratio > 0)

    qa_flags: list[str] = []
    has_timeline = video_duration_ms > 0 or feat.get("start_time_ratio") is not None
    in_intro_signoff = has_timeline and start_time_ratio <= cfg.intro_signoff_window_ratio
    in_outro_signoff = (
        has_timeline
        and (
            start_time_ratio >= (1.0 - cfg.outro_signoff_window_ratio)
            or end_time_ratio >= (1.0 - cfg.outro_signoff_window_ratio)
        )
    )
    in_boundary_window = in_intro_signoff or in_outro_signoff

    track_vocal_ratio: float | None = None
    if vocal_profile is not None:
        from lyra_ocr.audio.vad import vocal_active_ratio

        track_vocal_ratio = vocal_active_ratio(vocal_profile, track.start_ms, track.end_ms)

    # Rule 0a: Split 1-word title-card fragment in intro/outro (e.g. "only" / "my")
    # Preserve sung 1-word title tokens inside an active lyric layer or backed by active vocals.
    if (
        not in_lyric_layer
        and has_timeline
        and (
            start_time_ratio <= cfg.split_title_intro_ratio
            or start_time_ratio >= (1.0 - cfg.split_title_outro_ratio)
            or end_time_ratio >= (1.0 - cfg.split_title_outro_ratio)
        )
        and is_split_title_fragment(dominant_text, v_title)
    ):
        has_active_vocals = (
            track_vocal_ratio is not None
            and track_vocal_ratio >= cfg.boundary_gate_vocal_ratio_min
        )
        if not has_active_vocals:
            res = ClassificationResult(
                label="non_lyric",
                confidence=0.95,
                reason=f"Split single-word title card fragment '{dominant_text}' in intro/outro",
                decision="reject",
                qa_flags=["split_title_card_fragment"],
            )
            track.classification = res
            return res

    # Rule 0b: Single-word intro/outro sign-off (e.g. "WATCHING!")
    if in_boundary_window and word_count == 1 and _SIGNOFF_WORD_RE.match(dominant_text):
        res = ClassificationResult(
            label="non_lyric",
            confidence=0.95,
            reason=f"Intro/outro sign-off word '{dominant_text}'",
            decision="reject",
            qa_flags=["intro_outro_signoff"],
        )
        track.classification = res
        return res

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
    if spatial_variance > cfg.max_spatial_variance_non_lyric:
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
    if _matches_metadata_entity(
        dom_lower,
        dom_tokens,
        v_channel,
        n_text_changes=n_text_changes,
        cfg=cfg,
        require_full_entity_or_high_sim=in_lyric_layer,
    ):
        res = ClassificationResult(
            label="non_lyric",
            confidence=0.98,
            reason=f"Matched video channel/artist metadata '{v_channel}'",
            decision="reject",
            qa_flags=["metadata_channel_match"],
        )
        track.classification = res
        return res

    if (
        in_boundary_window
        and v_artist
        and v_artist != v_channel
        and _matches_metadata_entity(
            dom_lower,
            dom_tokens,
            v_artist,
            n_text_changes=n_text_changes,
            cfg=cfg,
            require_full_entity_or_high_sim=in_lyric_layer,
        )
    ):
        res = ClassificationResult(
            label="non_lyric",
            confidence=0.98,
            reason=f"Matched video artist metadata '{v_artist}' in intro/outro",
            decision="reject",
            qa_flags=["metadata_artist_match"],
        )
        track.classification = res
        return res

    # Rule 3b: Right-rail chat / stream overlay.
    # Live-stream comments occupy a narrow right-hand column, last only a few
    # seconds, and are much wider than they are tall. They are not lyrics even
    # though they can fall inside a naive reading-zone test.
    if (
        mean_cx >= cfg.chat_rail_min_cx
        and area_ratio <= cfg.chat_rail_max_area_ratio
        and duration_ms <= cfg.chat_rail_max_dur_ms
        and aspect_ratio >= cfg.chat_rail_min_aspect_ratio
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
    if (
        lifetime_ratio >= cfg.persistent_min_lifetime_ratio
        and n_text_changes <= cfg.persistent_max_text_changes
    ):
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
    if dominant_text.startswith("@") or (
        dominant_text.startswith("#") and len(dominant_text) >= cfg.hashtag_min_len
    ):
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
        if vocal_ratio < cfg.vad_silent_max_vocal_ratio and (
            start_time_ratio < cfg.vad_intro_start_ratio_max
            or start_time_ratio > cfg.vad_outro_start_ratio_min
        ):
            res = ClassificationResult(
                label="non_lyric",
                confidence=0.88,
                reason=f"Silent section overlay (vocal active ratio {vocal_ratio:.2f})",
                decision="reject",
                qa_flags=["vad_silent_overlay"],
            )
            track.classification = res
            return res

    # Intro / Outro Production Credit Prefix & Infix Filter
    if (
        has_timeline
        and (start_time_ratio <= cfg.intro_credit_start_ratio_max or in_outro_signoff)
        and _is_intro_credit_text(dominant_text, in_lyric_layer=in_lyric_layer, word_count=word_count)
    ):
        res = ClassificationResult(
            label="non_lyric",
            confidence=0.95,
            reason="Intro/outro production credit pattern",
            decision="reject",
            qa_flags=["intro_credit_pattern"],
        )
        track.classification = res
        return res

    # Rule 4: Intro Title Card / Song Title
    if v_title:
        sim_title = string_similarity(dom_lower, v_title)
        title_len_ratio = len(v_title) / max(1, len(dom_lower))
        title_tokens = set(re.findall(r"\w+", v_title))
        sig_title_tokens = {
            t for t in title_tokens if len(t) >= cfg.title_token_min_len and t not in {"the", "and", "for", "with", "this"}
        }
        split_frag_sung = is_split_title_fragment(dominant_text, v_title) and (
            in_lyric_layer
            or (
                track_vocal_ratio is not None
                and track_vocal_ratio >= cfg.boundary_gate_vocal_ratio_min
            )
        )
        token_title_match = bool(
            not split_frag_sung
            and sig_title_tokens
            and dom_tokens.issubset(sig_title_tokens)
            and n_text_changes == 0
            and len(dom_tokens) <= cfg.title_token_max_dom_tokens
            and (len(dom_tokens) / len(sig_title_tokens) >= cfg.title_token_min_coverage)
        )
        is_title_match = (
            sim_title >= cfg.title_min_similarity
            or token_title_match
            or (v_title in dom_lower and title_len_ratio >= cfg.title_substring_min_len_ratio)
        )
        if is_title_match:
            layer_id = feat.get("layer_id")
            if all_tracks is not None and layer_id is not None:
                n_prior_lyric_layer_tracks = sum(
                    1
                    for t in all_tracks
                    if t is not track
                    and (t.features or {}).get("layer_id") == layer_id
                    and t.end_ms <= track.start_ms + cfg.chorus_prior_track_end_slack_ms
                    and effective_duration_ms(t.start_ms, t.end_ms, sample_interval_ms)
                    >= cfg.chorus_prior_track_min_dur_ms
                )
                if (
                    layer_kind == "lyric"
                    and n_prior_lyric_layer_tracks >= cfg.chorus_min_prior_tracks
                    and cfg.lyric_min_dur_ms <= duration_ms <= cfg.lyric_max_dur_ms
                ):
                    res = ClassificationResult(
                        label="lyric",
                        confidence=0.88,
                        reason=(
                            f"Chorus/hook matching video title metadata '{v_title}' "
                            f"inside active lyric layer (id={layer_id}, prior={n_prior_lyric_layer_tracks})"
                        ),
                        decision="accept",
                        qa_flags=["title_hook_in_lyric_layer", "potential_chorus"],
                    )
                    track.classification = res
                    return res
                reject_intro_title = (
                    start_time_ratio <= cfg.title_intro_start_ratio_with_layers
                    or n_prior_lyric_layer_tracks == 0
                )
            else:
                reject_intro_title = start_time_ratio <= cfg.title_intro_start_ratio_fallback

            if reject_intro_title:
                res = ClassificationResult(
                    label="non_lyric",
                    confidence=0.98,
                    reason=f"Title text matching video title metadata '{v_title}'",
                    decision="reject",
                    qa_flags=["metadata_title_match"],
                )
                track.classification = res
                return res
            # Matching the title mid-video is often a repeated hook/chorus, not an
            # intro card: if in reading zone and transient, classify as lyric for review.
            if in_reading_zone and cfg.lyric_min_dur_ms <= duration_ms <= cfg.lyric_max_dur_ms and not only_intro_outro:
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
        and len(dominant_text.strip()) >= cfg.lyric_min_chars
        and not only_intro_outro
        and duration_ms >= cfg.lyric_layer_min_dur_ms
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
    if (
        not in_lyric_layer
        and start_time_ratio <= cfg.intro_card_start_ratio_max
        and track.end_ms <= cfg.intro_card_end_ms_max
        and n_text_changes <= cfg.intro_card_max_text_changes
        and duration_ms <= cfg.intro_card_max_dur_ms
    ):
        if (
            cfg.intro_card_strict_min_dur_ms <= duration_ms <= cfg.intro_card_strict_max_dur_ms
            and not is_corner
            and looks_like_card
        ):
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
    if (
        distinct_changes >= cfg.lyric_stream_min_distinct_changes
        and spatial_variance <= cfg.lyric_stream_max_spatial_variance
        and layer_kind != "overlay"
    ):
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
        cfg.lyric_min_dur_ms <= duration_ms <= cfg.lyric_max_dur_ms
        and in_reading_zone
        and not is_corner
        and word_count >= cfg.fluctuation_lyric_min_words
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
        and (
            mean_cx < cfg.corner_side_cx_min
            or mean_cx > cfg.corner_side_cx_max
            or is_corner
            or mean_cy > cfg.reading_zone_cy_max
        )
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
            and cfg.reading_zone_cx_min <= mean_cx <= cfg.reading_zone_cx_max
            and cfg.reading_zone_cy_min <= mean_cy <= cfg.reading_zone_cy_max
            and aspect_ratio >= cfg.reading_zone_min_aspect_ratio
        )
        if (
            cfg.lyric_min_dur_ms <= duration_ms <= cfg.lyric_max_dur_ms
            and in_reading_zone
            and has_text
            and len(dominant_text.strip()) >= cfg.lyric_min_chars
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
            len(dominant_text.split()) >= cfg.single_frame_min_words
            and in_reading_zone
            and aspect_ratio >= cfg.single_frame_min_aspect_ratio
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
    *,
    thresholds: Optional[ClassificationThresholds] = None,
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
            all_tracks=tracks,
            thresholds=thresholds,
        )
        results.append(res)
    return results
