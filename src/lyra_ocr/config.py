"""Centralized, immutable threshold configurations for the Lyra OCR pipeline.

Each stage exposes a frozen dataclass whose defaults preserve the empirically
tuned production constants while allowing callers and experiments to inspect or
override thresholds in one place without touching algorithm code.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LayerDiscoveryThresholds:
    """Spatial clustering and layer-kind classification thresholds for ``layers.py``."""

    text_similarity_threshold: float = 0.70
    cluster_cy_tolerance: float = 0.030
    cluster_cx_tolerance: float = 0.25
    cluster_cy_weight: float = 4.0
    center_cx_min: float = 0.30
    center_cx_max: float = 0.70
    max_cx_stddev: float = 0.14
    hud_code_max_len: int = 6
    hud_code_min_ratio: float = 0.60
    lyric_min_tracks: int = 4
    lyric_min_distinct_texts: int = 4
    lyric_alt_min_tracks: int = 8
    lyric_min_distinct_ratio: float = 0.30
    lyric_min_median_dur_ms: int = 500
    overlay_min_tracks: int = 3
    overlay_max_distinct_texts: int = 2
    companion_y_dist_min: float = 0.025
    companion_y_dist_max: float = 0.16
    companion_min_time_overlap_ratio: float = 0.50


@dataclass(frozen=True)
class GroupingThresholds:
    """Layer consolidation, cue deduplication, and multiline grouping thresholds."""

    # Text matching and duplicate band merging
    substring_match_min_len: int = 8
    layer_text_similarity: float = 0.90
    duplicate_min_containment: float = 0.70
    duplicate_max_y_gap: float = 0.12

    # Parallel bilingual layer detection
    bilingual_min_overlap_ratio: float = 0.45
    bilingual_min_cues: int = 8
    bilingual_min_len_ratio: float = 0.45
    bilingual_min_y_separation: float = 0.035
    bilingual_min_romaji_score: float = 0.30
    bilingual_min_spanish_rate: float = 0.10
    bilingual_dominant_ratio: float = 2.0

    # Same-language layer consolidation
    consolidate_overlap_threshold: float = 0.25
    consolidate_overlap_margin_ms: int = 200
    companion_band_y_min: float = 0.02
    companion_band_y_max: float = 0.18
    companion_band_x_min: float = 0.28
    companion_band_x_max: float = 0.72
    noisy_layer_min_dom_ratio: float = 0.35
    noisy_layer_min_cues: int = 10

    # Simultaneous multi-line resolution & same-Y flicker collapse
    short_wrap_max_dur_ms: int = 4000
    short_wrap_max_words: int = 12
    short_wrap_max_cjk_chars: int = 22
    min_stagger_gap_ms: int = 350
    same_y_band_tolerance: float = 0.032
    same_y_min_containment: float = 0.70
    simultaneous_start_delta_ms: int = 600
    simultaneous_min_iou: float = 0.60
    flicker_min_meaningful_chars: int = 4
    wrap_y_diff_min: float = 0.015
    wrap_y_diff_max: float = 0.18

    # Cue deduplication & fade-transition merging
    dedupe_max_y_diff: float = 0.08
    dedupe_min_bbox_iou: float = 0.30
    dedupe_text_similarity: float = 0.88
    dedupe_affix_similarity: float = 0.80
    dedupe_affix_min_chars: int = 10
    fade_max_gap_ms: int = 800
    fade_max_dur_ms: int = 1400
    fade_min_similarity: float = 0.62
    fade_strong_similarity: float = 0.75
    fade_affix_min_chars: int = 4
    fade_short_text_max_chars: int = 3
    merge_max_start_delta_ms: int = 1500
    merge_overlap_gap_ms: int = 200
    merge_split_max_gap_ms: int = 2500
    same_track_max_gap_ms: int = 8000
    upgrade_min_extra_chars: int = 3
    upgrade_max_conf_drop: float = 0.05

    # Cross-layer solitary cue sharing
    share_min_gap_ms: int = 1200
    share_min_cue_dur_ms: int = 900
    share_major_layer_min_cues: int = 8
    share_major_layer_min_small_cues: int = 2
    share_major_layer_min_ratio: float = 0.25
    share_solitary_pad_ms: int = 800
    share_span_margin_ms: int = 5000
    share_neighbor_slack_ms: int = 200


@dataclass(frozen=True)
class ClassificationThresholds:
    """Rule-based lyric vs non-lyric track classification thresholds for ``classify.py``."""

    # Active reading zone & intro/outro card priors
    reading_zone_cx_min: float = 0.10
    reading_zone_cx_max: float = 0.90
    reading_zone_cy_min: float = 0.20
    reading_zone_cy_max: float = 0.95
    reading_zone_min_aspect_ratio: float = 1.0
    card_max_words: int = 4
    card_start_ratio_max: float = 0.01
    card_cy_max: float = 0.65
    card_min_area_ratio: float = 0.03
    card_max_aspect_ratio: float = 3.5
    intro_end_ratio_max: float = 0.08
    outro_start_ratio_min: float = 0.95

    # Trajectory drift & channel/chat/persistence filters
    max_spatial_variance_non_lyric: float = 0.04
    channel_token_min_len: int = 3
    channel_partial_max_dom_tokens: int = 5
    channel_min_similarity: float = 0.75
    channel_substring_min_len_ratio: float = 0.4
    chat_rail_min_cx: float = 0.78
    chat_rail_max_area_ratio: float = 0.02
    chat_rail_max_dur_ms: int = 5000
    chat_rail_min_aspect_ratio: float = 3.0
    persistent_min_lifetime_ratio: float = 0.45
    persistent_max_text_changes: int = 1
    hashtag_min_len: int = 4

    # Audio VAD & intro credit / title matching
    vad_silent_max_vocal_ratio: float = 0.05
    vad_intro_start_ratio_max: float = 0.20
    vad_outro_start_ratio_min: float = 0.80
    intro_credit_start_ratio_max: float = 0.15
    title_token_min_len: int = 3
    title_token_max_dom_tokens: int = 4
    title_token_min_coverage: float = 0.50
    title_min_similarity: float = 0.75
    title_substring_min_len_ratio: float = 0.6
    chorus_prior_track_end_slack_ms: int = 500
    chorus_prior_track_min_dur_ms: int = 800
    chorus_min_prior_tracks: int = 2
    lyric_min_dur_ms: int = 500
    lyric_max_dur_ms: int = 16000
    title_intro_start_ratio_with_layers: float = 0.12
    title_intro_start_ratio_fallback: float = 0.25

    # Layer membership, intro cards, lyric streams & fallback rules
    lyric_min_chars: int = 2
    lyric_layer_min_dur_ms: int = 400
    intro_card_start_ratio_max: float = 0.05
    intro_card_end_ms_max: int = 15000
    intro_card_max_text_changes: int = 1
    intro_card_max_dur_ms: int = 10000
    intro_card_strict_min_dur_ms: int = 2000
    intro_card_strict_max_dur_ms: int = 9000
    lyric_stream_min_distinct_changes: int = 2
    lyric_stream_max_spatial_variance: float = 0.02
    fluctuation_lyric_min_words: int = 2
    corner_side_cx_min: float = 0.25
    corner_side_cx_max: float = 0.75
    single_frame_min_words: int = 2
    single_frame_min_aspect_ratio: float = 2.0


@dataclass(frozen=True)
class OcrPreprocessThresholds:
    """Crop preprocessing, soft B&W normalization, whitespace & fallback thresholds."""

    # Standard recognition preprocessing
    upscale_trigger_height_px: int = 36
    upscale_min_scale: float = 2.0
    upscale_target_height_px: float = 56.0
    clahe_min_std: float = 8.0
    clahe_clip_limit: float = 1.8
    clahe_tile_grid_size: tuple[int, int] = (4, 4)
    border_pad_px: int = 4

    # Border background distance & soft B&W stroke normalization
    border_y_fraction_div: int = 8
    border_x_fraction_div: int = 20
    soft_bw_min_height_px: int = 48
    soft_bw_low_percentile: float = 15.0
    soft_bw_high_percentile: float = 98.0

    # Visual word-space reconstruction
    spaced_max_token_len_guard: int = 14
    unspaced_min_len_guard: int = 9
    whitespace_min_crop_height_px: int = 4
    whitespace_min_crop_width_px: int = 8
    whitespace_min_fg_max: float = 18.0
    whitespace_fg_ref_percentile: float = 98.0
    whitespace_fg_ref_max_ratio: float = 0.5
    whitespace_ink_thresh_min: float = 18.0
    whitespace_ink_thresh_ratio: float = 0.25
    whitespace_smooth_kernel_px: int = 3
    whitespace_min_gap_px: int = 6
    whitespace_min_gap_height_ratio: float = 0.45
    whitespace_valley_density_max: float = 0.04
    whitespace_valley_rel_pos_min: float = 0.08
    whitespace_valley_rel_pos_max: float = 0.92
    whitespace_min_split_char_gap: int = 2

    # Fallback recognition routing
    latin_alt_min_unspaced_len: int = 8
    fallback_min_confidence: float = 0.88


@dataclass(frozen=True)
class ASRAlignmentThresholds:
    """Vocal language routing and word-level ASR alignment thresholds for ``asr.py``."""

    non_en_min_cues: int = 5
    word_window_pad_ms: int = 400
    fuzzy_token_similarity_min: float = 0.70
    word_match_ratio_min: float = 0.50
    total_similarity_min: float = 0.55


DEFAULT_LAYER_DISCOVERY_THRESHOLDS = LayerDiscoveryThresholds()
DEFAULT_GROUPING_THRESHOLDS = GroupingThresholds()
DEFAULT_CLASSIFICATION_THRESHOLDS = ClassificationThresholds()
DEFAULT_OCR_PREPROCESS_THRESHOLDS = OcrPreprocessThresholds()
DEFAULT_ASR_ALIGNMENT_THRESHOLDS = ASRAlignmentThresholds()

__all__ = [
    "ASRAlignmentThresholds",
    "ClassificationThresholds",
    "DEFAULT_ASR_ALIGNMENT_THRESHOLDS",
    "DEFAULT_CLASSIFICATION_THRESHOLDS",
    "DEFAULT_GROUPING_THRESHOLDS",
    "DEFAULT_LAYER_DISCOVERY_THRESHOLDS",
    "DEFAULT_OCR_PREPROCESS_THRESHOLDS",
    "GroupingThresholds",
    "LayerDiscoveryThresholds",
    "OcrPreprocessThresholds",
]
