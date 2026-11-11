from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from lyra_ocr.labels import load_labels_jsonl
from lyra_ocr.lyric.correct import build_corpus_vocab, correct_text
from lyra_ocr.lyric.models import CorrectionConfig, CorrectionInfo, LyricCue, ReferenceConfig
from lyra_ocr.lyric.reference import best_reference_similarity, line_similarity
from lyra_ocr.lyric.text_repair import oov_ratio, text_quality


def load_artifacts(artifacts_dir: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Return (manifest, tracks, human_labels) with human_labels keyed by str(track_id)."""
    p = Path(artifacts_dir)

    # 1. Manifest
    manifest_path = p / "manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            loaded_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(loaded_manifest, dict):
                manifest = loaded_manifest
        except Exception:
            manifest = {}

    # 2. Tracks
    tracks_path = p / "tracks.json"
    tracks: list[dict[str, Any]] = []
    if tracks_path.exists():
        try:
            raw_tracks = json.loads(tracks_path.read_text(encoding="utf-8"))
            if isinstance(raw_tracks, dict) and "tracks" in raw_tracks:
                raw_list = raw_tracks["tracks"]
            elif isinstance(raw_tracks, list):
                raw_list = raw_tracks
            else:
                raw_list = []
            tracks = [t for t in raw_list if isinstance(t, dict)] if isinstance(raw_list, list) else []
        except Exception:
            tracks = []

    # 3. Human labels
    labels_path = p / "labels.jsonl"
    human_labels: dict[str, dict[str, Any]] = {}
    if labels_path.exists():
        records = load_labels_jsonl(labels_path)
        for (_, tid), row in records.items():
            if not isinstance(row, dict):
                continue
            if row.get("annotator") == "silver_lyrics_alignment":
                continue
            human_labels[str(tid)] = row

    return manifest, tracks, human_labels


def _segment_confidence(seg: dict[str, Any], track: dict[str, Any]) -> float | None:
    """Return the segment OCR confidence, or None when it was never recorded.

    Older artifacts (and hand-written fixtures) omit confidence. Treating a
    missing value as 0.0 would gate every such cue, so unknown stays unknown and
    the confidence gate is skipped for that cue.
    """
    features = track.get("features") or {}
    if not isinstance(features, dict):
        features = {}
    for container, key in (
        (seg, "confidence"),
        (track, "confidence"),
        (features, "avg_confidence"),
    ):
        if key in container and container.get(key) is not None:
            try:
                return float(container[key])
            except (TypeError, ValueError):
                continue
    return None


def _quality_gate(
    conf: float | None,
    quality: float,
    n_words: int,
    oov: float,
    config: CorrectionConfig,
) -> str:
    """Return a review reason for unreadable text, or an empty string."""
    if conf is not None and conf < config.min_ocr_confidence:
        return "low_ocr_confidence"
    if (
        conf is not None
        and conf < config.review_ocr_confidence
        and quality < config.min_text_quality
    ):
        return "low_text_quality"
    if quality < config.min_text_quality_hard:
        return "low_text_quality"
    # High OCR confidence can still be attached to unreadable text (blurred or
    # stylized 360p lines). Multi-word lines below this floor are not lyrics.
    if n_words >= config.min_gate_words and quality < config.min_text_quality_multiword:
        return "low_text_quality"
    if n_words >= config.min_gate_words and oov >= config.max_oov_ratio:
        return "low_text_quality"
    return ""


def select_cues(
    tracks: list[dict[str, Any]],
    human_labels: dict[str, dict[str, Any]],
    *,
    include_review: bool = False,
    video_id: str = "",
    correct_method: str = "dictionary",
    frame_width: int = 1280,
    frame_height: int = 720,
    config: CorrectionConfig | None = None,
    reference_lines: list[str] | None = None,
    reference_bundle: Any = None,
    reference_source: str = "",
    reference_config: ReferenceConfig | None = None,
    sample_interval_ms: int = 1000,
) -> tuple[list[LyricCue], list[dict[str, Any]]]:
    accepted_cues: list[LyricCue] = []
    skipped_records: list[dict[str, Any]] = []
    cfg = config if config is not None else CorrectionConfig(method=correct_method)
    ref_cfg = reference_config if reference_config is not None else ReferenceConfig()
    # Quality/OOV scoring and medoid clustering both need corpus frequencies.
    corpus_vocab: dict[str, int] = build_corpus_vocab(tracks, human_labels=human_labels)

    eff_ref_source = reference_source
    eff_ref_lines: list[str] = []
    if reference_bundle is not None:
        eff_ref_source = eff_ref_source or getattr(reference_bundle, "source", "")
        eff_ref_lines = [getattr(l, "text", str(l)) for l in getattr(reference_bundle, "lines", [])]
    elif reference_lines:
        eff_ref_lines = list(reference_lines)

    for track in tracks:
        if not isinstance(track, dict):
            continue
        tid = str(track.get("track_id", ""))
        h = human_labels.get(tid) or {}
        if not isinstance(h, dict):
            h = {}
        features = track.get("features") or {}
        if not isinstance(features, dict):
            features = {}
        layer_id = features.get("layer_id")
        track_lang = features.get("lang") or features.get("line_lang") or "und"

        # Spatial info
        bbox = track.get("bbox", [])
        mean_centroid = features.get("mean_centroid")
        if mean_centroid and len(mean_centroid) >= 2:
            x_center = float(mean_centroid[0])
            y_center = float(mean_centroid[1])
        elif bbox and len(bbox) >= 4:
            x_center = float((bbox[0] + bbox[2]) / 2.0 / max(1, frame_width))
            y_center = float((bbox[1] + bbox[3]) / 2.0 / max(1, frame_height))
        else:
            x_center = 0.0
            y_center = 0.0

        # Segment-level evaluation
        segments = track.get("segments") or []
        if not isinstance(segments, list):
            segments = []
        segments = [seg for seg in segments if isinstance(seg, dict)]
        if not segments:
            fallback_seg: dict[str, Any] = {
                "segment_id": None,
                "start_ms": track.get("start_ms", 0),
                "end_ms": track.get("end_ms", 0),
                "text": features.get("dominant_text", ""),
            }
            fallback_conf = _segment_confidence({}, track)
            if fallback_conf is not None:
                fallback_seg["confidence"] = fallback_conf
            segments = [fallback_seg]

        for seg in segments:
            seg_id = seg.get("segment_id")
            h_seg = human_labels.get(f"{tid}.{seg_id}") if seg_id is not None else None
            seg_h = h_seg if (h_seg and (h_seg.get("human_label") or h_seg.get("human_decision") or "tracking_error" in h_seg)) else h

            # 1. Label and decision determination (segment preferred, fallback to track)
            if seg_h.get("human_label"):
                seg_label = seg_h["human_label"]
                seg_decision = seg_h.get("human_decision") or ("accept" if seg_label == "lyric" else "reject")
            else:
                seg_label = track.get("label", "non_lyric")
                seg_decision = track.get("decision", "review")

            # 2. Tracking error check (segment-specific takes precedence, else track-level)
            is_tracking_error = bool(h_seg.get("tracking_error")) if (h_seg and "tracking_error" in h_seg) else bool(h.get("tracking_error"))
            if is_tracking_error:
                skipped_records.append({
                    "track_id": track.get("track_id"),
                    "segment_id": seg_id,
                    "reason": "tracking_error",
                    "label": seg_label,
                    "decision": seg_decision,
                    "text": seg.get("text", "") or features.get("dominant_text", ""),
                })
                continue

            # 3. Acceptance condition per segment
            is_ref_rescued = False
            seg_text_raw = seg.get("text", "") or features.get("dominant_text", "")
            seg_dur = (seg.get("end_ms", 0) - seg.get("start_ms", 0)) or (track.get("end_ms", 0) - track.get("start_ms", 0))
            eff_seg_dur = seg_dur if seg_dur > 0 else sample_interval_ms
            qa_flags = list(track.get("qa_flags") or [])

            # Conservative rescue:
            # Trusted manual or LRCLIB reference can rescue a non_lyric track.
            # Human labels always win, and known card/credit/overlay subtypes
            # remain rejected even when their text happens to match a lyric.
            has_human_label = bool(seg_h.get("human_label") or seg_h.get("human_decision"))
            blocked_rescue_flags = {
                "intro_bounded_card",
                "metadata_title_match",
                "only_intro_outro",
                "social_handle_tag",
                "metadata_channel_match",
            }
            valid_rescue_source = eff_ref_source in (
                "manual",
                "lrclib_synced",
                "lrclib_plain",
                "lrclib_romanized",
            )
            if (
                seg_label != "lyric"
                and not has_human_label
                and ref_cfg.rescue_non_lyric
                and valid_rescue_source
                and eff_seg_dur >= 1000
                and not any(f in qa_flags for f in blocked_rescue_flags)
                and (len(seg_text_raw.split()) >= 2 or any("\u4e00" <= ch <= "\u9fff" for ch in seg_text_raw))
                and eff_ref_lines
            ):
                # For Romaji or Romanized references, compare space-stripped strings to tolerate word boundary variations
                if track_lang == "ja-Latn" or eff_ref_source == "lrclib_romanized":
                    raw_no_space = re.sub(r"\s+", "", seg_text_raw.lower())
                    best_sim = max(
                        (line_similarity(raw_no_space, re.sub(r"\s+", "", rl.lower())) for rl in eff_ref_lines),
                        default=0.0,
                    )
                    threshold = min(0.75, ref_cfg.replace_similarity)
                else:
                    best_sim = best_reference_similarity(seg_text_raw, eff_ref_lines)
                    threshold = ref_cfg.replace_similarity

                if best_sim >= threshold:
                    seg_label = "lyric"
                    seg_decision = "accept"
                    is_ref_rescued = True
                    qa_flags.append("reference_rescue")

            if seg_decision == "reject":
                is_accepted = False
            else:
                is_chorus_rescue = (
                    "potential_chorus" in qa_flags
                    or (seg_label == "lyric" and "title_match_outside_intro" in qa_flags)
                )
                is_accepted = (seg_label == "lyric") and (
                    seg_decision == "accept"
                    or (include_review and seg_decision == "review")
                    or is_chorus_rescue
                )

            if not is_accepted:
                reason = "non_lyric" if seg_label != "lyric" else "review"
                skipped_records.append({
                    "track_id": track.get("track_id"),
                    "segment_id": seg_id,
                    "reason": reason,
                    "label": seg_label,
                    "decision": seg_decision,
                    "text": seg_text_raw,
                })
                continue

            # 4. Text resolution & corrected_text
            # If multi-segment track, ONLY use segment-specific corrected text.
            # If single-segment or no segments, track-level corrected text is allowed.
            if h_seg and (h_seg.get("corrected_text") or "").strip():
                seg_human_text = h_seg["corrected_text"].strip()
            elif len(segments) <= 1 and (h.get("corrected_text") or "").strip():
                seg_human_text = h["corrected_text"].strip()
            else:
                seg_human_text = ""

            if seg_human_text:
                final_text = seg_human_text
                raw_text = seg.get("text") or features.get("dominant_text") or seg_human_text
                corr_info = CorrectionInfo(method="human", changed=True, confidence=1.0)
            else:
                final_text, raw_text, corr_info = correct_text(
                    track, tracks, cfg, vocab=corpus_vocab, segment=seg if seg_id is not None else None
                )

            if not final_text.strip():
                skipped_records.append({
                    "track_id": track.get("track_id"),
                    "segment_id": seg_id,
                    "reason": "empty_text",
                    "label": seg_label,
                    "decision": seg_decision,
                    "text": raw_text,
                })
                continue

            # 5. OCR-confidence / text-quality gate. Human corrections bypass it.
            seg_conf = _segment_confidence(seg, track)
            quality = text_quality(final_text, corpus_vocab)
            n_words = len(final_text.split())
            oov = oov_ratio(final_text, corpus_vocab)
            # A trusted reference line proves the OCR text is real lyrics even
            # when the garbled spelling trips the quality gate.
            reference_match = best_reference_similarity(final_text, eff_ref_lines)
            is_non_en = bool(track_lang and track_lang not in ("en", "und"))
            gate_reason = "" if (seg_human_text or reference_match >= 0.55 or is_non_en) else _quality_gate(
                seg_conf, quality, n_words, oov, cfg
            )
            if gate_reason and not include_review:
                skipped_records.append({
                    "track_id": track.get("track_id"),
                    "segment_id": seg_id,
                    "reason": gate_reason,
                    "label": seg_label,
                    "decision": "review",
                    "text": final_text,
                    "ocr_confidence": seg_conf,
                    "text_quality": round(float(quality), 4),
                    "qa_flags": [gate_reason],
                })
                continue

            cue_decision = "review" if gate_reason else seg_decision
            cue_flags = [gate_reason] if gate_reason else []
            if is_ref_rescued and "reference_rescue" not in cue_flags:
                cue_flags.append("reference_rescue")

            seg_start_ms = int(seg.get("start_ms", track.get("start_ms", 0)))
            seg_end_ms = int(seg.get("end_ms", seg_start_ms))
            if seg_end_ms <= seg_start_ms:
                seg_end_ms = seg_start_ms + 300

            cue = LyricCue(
                start_ms=seg_start_ms,
                end_ms=seg_end_ms,
                text=final_text,
                raw_text=raw_text,
                source_track_ids=[track.get("track_id")] if track.get("track_id") is not None else [],
                source_segment_ids=[seg_id] if seg_id is not None else [],
                confidence=float(seg_conf if seg_conf is not None else 0.0),
                decision=cue_decision,
                review_required=(cue_decision == "review"),
                bbox=bbox,
                x_center=x_center,
                y_center=y_center,
                correction=corr_info,
                qa_flags=cue_flags,
                ocr_confidence=seg_conf,
                text_quality=round(float(quality), 4),
                layer_id=layer_id,
                lang=track_lang,
            )
            accepted_cues.append(cue)

    return accepted_cues, skipped_records
