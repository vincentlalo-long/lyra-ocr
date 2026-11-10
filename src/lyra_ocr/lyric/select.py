from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lyra_ocr.labels import load_labels_jsonl
from lyra_ocr.lyric.correct import build_corpus_vocab, correct_text
from lyra_ocr.lyric.models import CorrectionConfig, CorrectionInfo, LyricCue


def load_artifacts(artifacts_dir: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Return (manifest, tracks, human_labels) with human_labels keyed by str(track_id)."""
    p = Path(artifacts_dir)

    # 1. Manifest
    manifest_path = p / "manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}

    # 2. Tracks
    tracks_path = p / "tracks.json"
    tracks: list[dict[str, Any]] = []
    if tracks_path.exists():
        try:
            raw_tracks = json.loads(tracks_path.read_text(encoding="utf-8"))
            if isinstance(raw_tracks, dict) and "tracks" in raw_tracks:
                tracks = raw_tracks["tracks"]
            elif isinstance(raw_tracks, list):
                tracks = raw_tracks
        except Exception:
            tracks = []

    # 3. Human labels
    labels_path = p / "labels.jsonl"
    human_labels: dict[str, dict[str, Any]] = {}
    if labels_path.exists():
        records = load_labels_jsonl(labels_path)
        for (_, tid), row in records.items():
            if row.get("annotator") == "silver_lyrics_alignment":
                continue
            human_labels[str(tid)] = row

    return manifest, tracks, human_labels


def select_cues(
    tracks: list[dict[str, Any]],
    human_labels: dict[str, dict[str, Any]],
    *,
    include_review: bool = False,
    video_id: str = "",
    correct_method: str = "dictionary",
    frame_width: int = 1280,
    frame_height: int = 720,
) -> tuple[list[LyricCue], list[dict[str, Any]]]:
    accepted_cues: list[LyricCue] = []
    skipped_records: list[dict[str, Any]] = []
    corpus_vocab = build_corpus_vocab(tracks, human_labels=human_labels) if correct_method == "dictionary" else None

    for track in tracks:
        tid = str(track.get("track_id", ""))
        h = human_labels.get(tid) or {}

        # Spatial info
        bbox = track.get("bbox", [])
        mean_centroid = track.get("features", {}).get("mean_centroid")
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
        if not segments:
            segments = [{
                "segment_id": None,
                "start_ms": track.get("start_ms", 0),
                "end_ms": track.get("end_ms", 0),
                "text": track.get("features", {}).get("dominant_text", ""),
                "confidence": track.get("confidence", 0.0),
            }]

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
                    "text": seg.get("text", "") or track.get("features", {}).get("dominant_text", ""),
                })
                continue

            # 3. Acceptance condition per segment
            if seg_decision == "reject":
                is_accepted = False
            else:
                qa_flags = track.get("qa_flags") or []
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
                    "text": seg.get("text", "") or track.get("features", {}).get("dominant_text", ""),
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
                raw_text = seg.get("text") or track.get("features", {}).get("dominant_text") or seg_human_text
                corr_info = CorrectionInfo(method="human", changed=True, confidence=1.0)
            else:
                cfg = CorrectionConfig(method=correct_method)
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
                confidence=float(seg.get("confidence", track.get("confidence", 0.0))),
                decision=seg_decision,
                review_required=(seg_decision == "review"),
                bbox=bbox,
                x_center=x_center,
                y_center=y_center,
                correction=corr_info,
            )
            accepted_cues.append(cue)

    return accepted_cues, skipped_records
