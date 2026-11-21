# Lyra OCR — Data Contracts & Artifact Specifications

This document defines the formal data contracts, coordinate systems, and serialization schemas used throughout the Lyra OCR pipeline.

---

## 1. Overview of Artifacts

Every execution pipeline persists standardized intermediate artifacts to allow inspection, regression testing, and modular processing without re-running expensive vision inference steps:

| Artifact | Generating Component | Description |
|---|---|---|
| `manifest.json` | Execution Coordinator (`pipeline.py`) | Run configuration, `config_hash`, tool versions, model identifiers, and input metadata. |
| `frame_index.json` | Stream Synchronization Service (`video/frames.py`) | Mapping between frame identifiers, relative image file paths, and millisecond timestamps. |
| `ocr_raw.json` | Vision & Recognition Pipeline (`pipeline.py`) | Raw frame-by-frame spatial detections (polygons, bboxes) and recognized text lines. |
| `roi_preview.jpg` | Review Renderer (`review.py`) | Representative annotated image frame for visual quality audit. |
| `tracks.json` | Temporal Tracking & Classification (`track.py`, `classify.py`) | Temporal tracks clustered across frames with features, segments, category labels, and `qa_flags`. |
| `review.csv` / `review_tracks.csv` | Classification Subsystem (`review.py`) | Tabular review sheets for rapid human-in-the-loop audit and confirmation. |
| `labels.jsonl` | Labeling Store (`labels.py`, `review.py`) | Append-only JSON Lines trail of human annotations and corrections. |
| `audio_16k.npy` | Audio Extractor (`audio/extract.py`) | Cached mono 16 kHz float32 PCM waveform extracted via FFmpeg. |
| `vocals_16k.npy` | Audio Extractor (`audio/extract.py`) | Cached mono 16 kHz float32 isolated vocal stem waveform (Demucs or Mid-Side HPSS fallback). |
| `asr_transcript.json` | Word-Level ASR (`audio/asr.py`) | Cached `faster-whisper` transcript with segment and word-level millisecond timestamps. |
| `reference_cache/<sha1>.json` | Reference Models (`lyric/ref_models.py`) | Cached `ReferenceBundle` responses from LRCLIB or YouTube captions keyed by normalized query hash. |
| `<name>.cues.json` | Lyric Export (`lyric/export.py`) | Complete cue-level assembly manifest with `words`, `layer_id`, `lang`, `qa_flags`, `skipped`, and provenance. |
| `<name>.lrc` / `<name>.<lang>.lrc` | Lyric Export (`lyric/export.py`) | Synchronized primary and per-language LRC lyric files (standard `[mm:ss.xx]` or Enhanced `<mm:ss.xx>`). |

---

## 2. Coordinate System & Geometry Conventions

1. **Origin**: Coordinate `(0, 0)` is located at the **top-left corner** of the frame.
2. **Polygon Representation (`polygon`)**:
   - Represented as a convex 4-point list of `[x, y]` coordinates:
     ```json
     [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
     ```
   - Ordering follows standard clockwise quadrilateral vertex sequencing: top-left, top-right, bottom-right, bottom-left.
3. **Bounding Box Representation (`bbox`)**:
   - Axis-aligned rectangle defined as `[xmin, ymin, xmax, ymax]`:
     ```json
     [xmin, ymin, xmax, ymax]
     ```
   - All coordinates are integer pixel values constrained within `[0, frame_width]` and `[0, frame_height]`.
4. **Timestamp Representation**:
   - All timestamps across internal artifacts are stored as non-negative integer **milliseconds (`ms`)**.
   - Timing is mapped to final centisecond format (`[mm:ss.xx]`) exclusively during `.lrc` file export.

---

## 3. Schema Definitions

### 3.1. `manifest.json`
Captures run-level parameters, environment provenance, and model signatures for auditability.

```json
{
  "video_id": "input_video_1a2b3c4d",
  "video_path": "/path/to/input_video.mp4",
  "metadata": {
    "title": "Song Title",
    "channel": "Artist Channel"
  },
  "tracker": {
    "name": "bytetrack",
    "iou_threshold": 0.30,
    "max_gap_ms": 2000
  },
  "config_hash": "e3b0c44298fc1c14",
  "probe": {
    "video_path": "/path/to/input_video.mp4",
    "duration": 141.224,
    "fps": 60.0,
    "width": 1280,
    "height": 720,
    "rotation": 0,
    "has_audio": true
  },
  "config": {
    "sample_fps": 2.0,
    "start_sec": 0.0,
    "sample_frames": 5
  },
  "runtime": {
    "requested": "auto",
    "device": "cpu",
    "providers": ["CPUExecutionProvider"],
    "rec_batch_size": 8,
    "intra_op_threads": -1,
    "inter_op_threads": 1,
    "dedup": false,
    "frame_source": "pipe",
    "pipeline_version": "3"
  },
  "tool_versions": {
    "python": "3.11.16",
    "rapidocr": "3.10.0",
    "onnxruntime": "1.31.1",
    "onnxruntime_providers": "CPUExecutionProvider",
    "opencv": "5.0.0",
    "ffmpeg": "ffmpeg version 8.1.3"
  },
  "models": {
    "det": "PP-OCRv6_det_small",
    "rec": "PP-OCRv6_rec_small"
  },
  "timestamp_created": 1730310000
}
```

### 3.2. `frame_index.json`
Records the exact Presentation Time Stamp (PTS) for each sampled frame.

```json
[
  {
    "frame_id": 0,
    "timestamp_ms": 20000,
    "pts_time": 20.0,
    "image_path": "frames/frame_000000.jpg"
  },
  {
    "frame_id": 1,
    "timestamp_ms": 20500,
    "pts_time": 20.5,
    "image_path": "frames/frame_000001.jpg"
  }
]
```

When frame dedup is enabled, a reused entry additionally carries `"reused": true`
and `"dedup_of_frame": <frame_id>`; its `image_path` points at the JPEG of the
frame it reused. `"pts_source": "estimated"` marks a timestamp recovered from
the sampling interval instead of ffmpeg's `showinfo` PTS (streaming mode only).

### 3.3. `ocr_raw.json`
Stores verbatim OCR outputs and spatial coordinates for every processed frame.

```json
{
  "video_path": "/path/to/input_video.mp4",
  "manifest_path": "manifest.json",
  "total_frames_processed": 5,
  "avg_frame_time_ms": 315.89,
  "total_ocr_calls": 5,
  "frames": [
    {
      "frame_id": 0,
      "timestamp_ms": 20000,
      "pts_time": 20.0,
      "image_path": "frames/frame_000000.jpg",
      "elapsed_ms": 472.0,
      "detect_ms": 180.2,
      "recognize_ms": 291.8,
      "boxes": [
        {
          "polygon": [[274, 2], [1001, 6], [1001, 55], [274, 51]],
          "bbox": [274, 2, 1001, 55],
          "text_raw": "We can be alone in the castle in the sky",
          "text_norm": "We can be alone in the castle in the sky",
          "confidence": 0.9957,
          "model_id": "PP-OCRv6_rec_small",
          "script": "latin"
        }
      ]
    }
  ]
}
```

`total_ocr_calls` counts every non-reused frame sent through detection and
recognition, including frames with zero detections; only dedup-reused frames
are excluded. `detect_ms` / `recognize_ms` split frame latency for profiling. A
reused frame carries `"reused": true`, `"dedup_of_frame": <frame_id>` and
`elapsed_ms: 0.0`.

---

## 4. Track, Classification & Review Specifications

### 4.1. `tracks.json` Structure & Label Taxonomy

Each text box across temporal frames is clustered into a continuous `TextTrack` stream containing spatial-temporal features (including `layer_id`, `layer_kind`, `lang`, and `line_lang` populated by `layers.py`), segment transitions, and full frame-level `occurrences`:

```json
{
  "total_tracks": 1,
  "tracks": [
    {
      "track_id": 1,
      "start_ms": 20000,
      "end_ms": 24500,
      "duration_ms": 4500,
      "bbox": [274, 2, 1001, 55],
      "n_text_changes": 0,
      "features": {
        "lifetime_ratio": 0.05,
        "spatial_variance": 0.0,
        "is_corner": false,
        "dominant_text": "We can be alone in the castle in the sky",
        "line_lang": "en",
        "layer_id": 1,
        "layer_kind": "lyric",
        "lang": "en"
      },
      "segments": [
        {
          "segment_id": 0,
          "start_ms": 20000,
          "end_ms": 24500,
          "text": "We can be alone in the castle in the sky",
          "confidence": 0.9957,
          "num_occurrences": 10
        }
      ],
      "occurrences": [
        {
          "frame_id": 0,
          "timestamp_ms": 20000,
          "polygon": [[274, 2], [1001, 2], [1001, 55], [274, 55]],
          "bbox": [274, 2, 1001, 55],
          "text_raw": "We can be alone in the castle in the sky",
          "text_norm": "We can be alone in the castle in the sky",
          "confidence": 0.9957,
          "model_id": "PP-OCRv6_rec_small",
          "script": "latin"
        }
      ],
      "num_occurrences": 10,
      "label": "lyric",
      "confidence": 0.95,
      "reason": "Member of dynamic lyric layer (id=1)",
      "decision": "accept",
      "qa_flags": ["lyric_layer_member"]
    }
  ]
}
```

> **Note on `occurrences`**: Storing full per-frame detection occurrences preserves exact frame-level spatial mapping, allowing interactive UI tools to link any bounding box on any frame directly to its corresponding `track_id`.

#### Label Taxonomy:

The final label space is binary. Diagnostic subtype information lives in `reason` and `qa_flags`, not in the label itself.

| Label | Description | Example Pattern |
|---|---|---|
| `lyric` | Core song lyrics intended for subtitle/LRC output. | Recurring text changes within a stable spatial region. |
| `non_lyric` | Every distractor: watermarks, title cards, social tags, scene text, or ambiguous tracks. | Metadata match, long invariant lifetime, high trajectory drift, or low confidence. |

### 4.2. `review.csv` Structure

Tabular format for rapid human-in-the-loop review. Machine predictions and human verification decisions are explicitly decoupled into separate columns to preserve baseline evaluation records:

```csv
track_id,start_ms,end_ms,machine_label,machine_decision,confidence,suggested_text,human_label,human_decision,review_note
1,20000,24500,lyric,accept,0.99,"We can be alone in the castle in the sky",,,
2,0,140000,non_lyric,reject,0.98,"CLARX & HARDDOPE",,,Matched persistent artist watermark
```

- `machine_label` & `machine_decision`: Model prediction baseline (`accept`, `reject`, `review`).
- `human_label` & `human_decision`: Populated by reviewers (`accept`, `reject`). Empty by default upon initial export.

### 4.3. `labels.jsonl` Audit Trail Specification

To support reproducible active learning and reviewer auditing without mutating the base machine output, human annotations can be recorded in an append-only JSON Lines file:

```json
{"track_id": "1", "human_label": "lyric", "human_decision": "accept", "annotator": "reviewer_alice", "note": "verified subtitle line", "timestamp": 1730700000}
```

When present, `export_review_csv()` ingests `labels.jsonl` to pre-populate `human_label` and `human_decision`.

---

## 5. Audio, Reference & Lyric Export Specifications

### 5.1. `asr_transcript.json` Schema (`audio/asr.py`)

Cached word-level transcription produced by `transcribe_audio_words()` (using `faster-whisper` with `word_timestamps=True`, `vad_filter=True`). Serialized via `ASRTranscript.to_dict()` / `save_asr_transcript()`:

```json
{
  "language": "en",
  "model_size": "small",
  "segments": [
    {
      "text": "We can be alone in the castle in the sky",
      "start_ms": 19980,
      "end_ms": 24420,
      "language": "en",
      "words": [
        {
          "word": "We",
          "start_ms": 19980,
          "end_ms": 20160,
          "probability": 0.94
        },
        {
          "word": "can",
          "start_ms": 20160,
          "end_ms": 20340,
          "probability": 0.97
        }
      ]
    }
  ]
}
```

| Field | Type | Description |
|---|---|---|
| `language` | `string` | Whisper-detected or normalized ISO-639-1 language code (`ja-Latn` maps to `ja`). |
| `model_size` | `string` | `faster-whisper` model size identifier (default `"small"`). |
| `segments` | `ASRSegment[]` | Ordered list of recognized speech/vocal segments. |
| `segments[].text` | `string` | Trimmed segment transcript string. |
| `segments[].start_ms`, `end_ms` | `int` | Segment boundaries in non-negative integer milliseconds (`end_ms >= start_ms`). |
| `segments[].language` | `string` | Segment-level language code. |
| `segments[].words` | `ASRWord[]` | Word-level timings (`word`, `start_ms`, `end_ms`, `probability` defaulting to `1.0`). |

### 5.2. `<name>.cues.json` Schema (`lyric/export.py`, `lyric/models.py`)

Written alongside `<name>.lrc` by `write_cues_json()` at the end of `build_lrc_from_artifacts()`. Records run provenance, build cache options, reference statistics, layer manifest, accepted `LyricCue` entries, and `skipped` records:

```json
{
  "video_id": "input_video_1a2b3c4d",
  "video_path": "/path/to/input_video.mp4",
  "duration_ms": 141224,
  "tracker": "bytetrack",
  "enhanced": true,
  "options": {
    "cache_version": 8,
    "include_review": false,
    "correct_method": "dictionary",
    "refine": false,
    "offset_ms": 0,
    "title": "Song Title",
    "artist": "Artist Name",
    "enhanced": true,
    "correction_config": {},
    "reference_config": {},
    "source_url": "",
    "reference": { "mode": "off", "sha1": "" },
    "no_caption_fallback": false
  },
  "has_labels": false,
  "reference_source": "lrclib_synced",
  "n_reference_matches": 12,
  "n_reference_gap_fills": 1,
  "reference_warnings": [],
  "primary_lang": "en",
  "layers": [
    {
      "layer_id": 1,
      "lang": "en",
      "path": "/path/to/output/song.lrc",
      "n_cues": 24
    }
  ],
  "metadata": {
    "title": "Song Title",
    "artist": "Artist Name",
    "album": "",
    "by": "Lyra OCR",
    "offset_ms": 0
  },
  "n_cues": 24,
  "n_skipped": 2,
  "review_required": 0,
  "cues": [
    {
      "start_ms": 20000,
      "end_ms": 24500,
      "text": "We can be alone",
      "raw_text": "We can be alone",
      "source_track_ids": [1],
      "source_segment_ids": [0],
      "confidence": 0.9957,
      "decision": "accept",
      "review_required": false,
      "bbox": [274, 2, 1001, 55],
      "x_center": 0.498,
      "y_center": 0.0396,
      "correction": {
        "method": "dictionary",
        "changed": false,
        "confidence": 0.9957,
        "diff": ""
      },
      "qa_flags": [],
      "ocr_confidence": 0.9957,
      "text_quality": 1.0,
      "layer_id": 1,
      "lang": "en",
      "words": [
        { "word": "We", "start_ms": 20000, "end_ms": 20900 },
        { "word": "can", "start_ms": 20900, "end_ms": 22250 },
        { "word": "be", "start_ms": 22250, "end_ms": 23150 },
        { "word": "alone", "start_ms": 23150, "end_ms": 24500 }
      ]
    }
  ],
  "skipped": [
    {
      "track_id": 2,
      "segment_id": 0,
      "reason": "non_lyric",
      "label": "non_lyric",
      "decision": "reject",
      "text": "OFFICIAL VIDEO"
    }
  ]
}
```

#### `LyricCue` Field Reference (`lyric/models.py`)

| Field | Type | Default | Description |
|---|---|---|---|
| `start_ms`, `end_ms` | `int` | — | Cue time bounds in integer milliseconds (`end_ms > start_ms`). |
| `text` | `string` | — | Final cue text emitted to `.lrc`. |
| `raw_text` | `string` | `""` | Pre-correction OCR/segment text. |
| `source_track_ids` | `int[]` | `[]` | Contributing `TextTrack.track_id` values (merged cues sort & deduplicate IDs). |
| `source_segment_ids` | `int[]` | `[]` | Contributing `TextSegment.segment_id` values. |
| `confidence` | `float` | `0.0` | Effective cue confidence in `[0.0, 1.0]`. |
| `decision` | `string` | `"accept"` | Selection decision (`"accept"`, `"review"`, or `"human_accept"`). |
| `review_required` | `bool` | `false` | `true` when flagged for human review (`decision == "review"` or uncertain alignment). |
| `bbox` | `int[]` | `[]` | Axis-aligned `[xmin, ymin, xmax, ymax]` pixel coordinates. |
| `x_center`, `y_center` | `float` | `0.0` | Normalized centroid coordinates in `[0.0, 1.0]`. |
| `correction` | `object` | `{}` | `CorrectionInfo`: `{method, changed, confidence, diff}` (`method` in `none`, `consensus`, `dictionary`, `human`, `reference`, `reference_caption`, `reference_gap_fill`). |
| `qa_flags` | `string[]` | `[]` | Diagnostic provenance and quality flags attached across stages (see below). |
| `ocr_confidence` | `float \| null` | `null` | Segment/track OCR recognition confidence (`null` if absent from legacy fixtures). |
| `text_quality` | `float \| null` | `null` | Readability score from `text_repair.text_quality()` in `[0.0, 1.0]`. |
| `layer_id` | `int \| null` | `null` | Spatial text band identifier assigned by `layers.py` / `layer_consolidation.py`. |
| `lang` | `string` | `"und"` | Language code (`en`, `ja`, `ja-Latn`, `es`, `vi`, `zh`, `ko`, `und`). |
| `words` | `object[]` | `[]` | Word-level timing spans `{"word": string, "start_ms": int, "end_ms": int}` populated by `align_multiline_cues_with_asr()` or `align_cues_with_audio()`. |

#### `qa_flags` Vocabulary

| Stage | Flag | Meaning |
|---|---|---|
| `select.py` | `low_ocr_confidence`, `low_text_quality` | Tripped `_quality_gate()` (emitted on cues when `include_review=True`, or in `skipped[].qa_flags`). |
| `select.py` | `reference_rescue` | `non_lyric` track rescued via high-similarity match against manual or LRCLIB reference. |
| `build.py` | `asr_unvoiced` | `asr_vocal_gate_score(cue, asr_transcript) < 0.20`; skipped if `eff_conf < 0.65` and `not include_review`. |
| `build.py` | `boundary_unvoiced_or_credit` | Boundary intro/outro cue failed vocal-activity/ASR gating or matched credit, sign-off, artist/channel, or split-title patterns in `filter_boundary_unvoiced_and_distractor_cues()`. |
| `build.py` | `cross_layer_gap_rescued` | Dropped non-rejected track recovered by `audit_and_recover_missing_lyrics()` to fill a parallel major-layer coverage gap. |
| `ref_align.py` | `reference_uncertain` | Matched cue fell below replacement/coverage/ratio guards (unless `ja-Latn` with `repair_romaji=False`). |
| `ref_align.py` | `reference_timing_split` | Overlapping OCR interval partitioned using synchronized reference timestamps. |
| `ref_align.py` | `caption_gap_fill` | Gap-filled cue originated from a YouTube caption reference source. |
| `grouping.py` | `multiline_merged` | Two adjacent short wrapped lines merged into one cue (`_try_merge_short_wrap`). |
| `grouping.py` | `multiline_staggered` | Simultaneous multi-line cues staggered proportionally when ASR alignment was unavailable. |
| `layer_consolidation.py` | `cross_layer_inherited` | Solitary English cue cloned into a silence gap of another major layer. |
| `audio/asr.py` | `asr_aligned` | Multi-line cue timestamps and `words` assigned via DP alignment to `ASRTranscript`. |
| `classify.py` | `lyric_layer_member`, `potential_chorus`, `title_hook_in_lyric_layer`, `title_match_outside_intro`, `metadata_title_match`, `metadata_channel_match`, `metadata_artist_match`, `static_overlay_layer`, `intro_bounded_card`, `only_intro_outro`, `intro_credit_pattern`, `intro_outro_signoff`, `split_title_card_fragment`, `vad_silent_overlay`, `long_lifetime_invariant`, `high_spatial_variance`, `social_handle_tag`, `chat_overlay_suspect`, `chat_like_content`, `isolated_fragment_word`, `ocr_fluctuation_lyric_line`, `ocr_fluctuation_static`, `single_cue_track`, `single_frame_lyric_cue`, `single_frame_isolated`, `needs_human_audit` | Track-level classification diagnostics stored in `tracks.json` (and `review_tracks.csv`). |

### 5.3. Multi-Layer `.lrc` & Enhanced `<mm:ss.xx>` Output Rules (`lyric/export.py`)

1. **Metadata Headers (`write_lrc`)**:
   - Emits non-empty ID tags `[ti:<title>]`, `[ar:<artist>]`, `[al:<album>]`, `[by:<by>]` (default `"Lyra OCR"`), followed unconditionally by `[offset:<offset_ms>]`.
   - Values are sanitized by `_sanitize_tag_value()`: `[` and `]` are replaced with spaces, and ASCII control characters (`0x00–0x1F`, `0x7F`) are stripped.
2. **Line Timestamp Formatting & Monotonicity**:
   - `format_lrc_timestamp(ms)` converts milliseconds to rounded non-negative centiseconds (`cs = max(0, int(round(ms / 10.0)))`) and formats `[mm:ss.xx]`.
   - Cues are sorted by `(start_ms, y_center)`, newlines/CRs in `cue.text` are collapsed to spaces, empty lines are skipped, and timestamps are clamped monotonically (`current_start_ms = max(cue.start_ms, prev_start_ms)`).
   - Duplicate `(centiseconds, cleaned_text)` pairs are omitted.
3. **Enhanced LRC Word Tags (`enhanced=True`)**:
   - When `enhanced=True` and `cue.words` is non-empty, each word entry with non-empty `word` is formatted using `format_enhanced_tag(w.get("start_ms", current_start_ms))` as `<mm:ss.xx><word>` and joined with single spaces after the line timestamp:
     ```text
     [00:20.00]<00:20.00>We <00:20.90>can <00:22.25>be <00:23.15>alone
     ```
   - For CJK/Hangul scripts (`ja`, `zh`, `ko`), `align_cues_with_audio()` tokenizes individual characters into `cue.words` so Enhanced LRC emits per-character `<mm:ss.xx>` tags.
   - When `enhanced=False` or `cue.words` is empty, `write_lrc()` emits standard plain lines `[mm:ss.xx]<text>`.
4. **Multi-Layer Splitting & Primary Language Selection (`write_lrc_layers`)**:
   - Runs `consolidate_language_layers(cues)` before grouping by `layer_id`.
   - **Ghost-layer filtering**: When `max_cues_in_layer >= 8`, layers with `< 4` cues and `< 0.15 * max_cues_in_layer` are dropped.
   - **Single-layer output**: If `<= 1` layer remains, only `<name>.lrc` is written (no `.<lang>.lrc` sidecar).
   - **Multi-layer output**: Layers sharing the same voted language (`vote_layer_lang`) are consolidated into one file `<name>.<lang>.lrc` per distinct language (sorted by `(start_ms, end_ms)`), avoiding `-2`/`-3` suffixes.
   - **Primary `.lrc` selection (`_layer_priority_key`)**:
     - If `preferred_lang` is specified (not `""` or `"auto"`): tier `0` = `lang == preferred_lang`, tier `1` = other `lang != "und"`, tier `2` = `"und"`.
     - If `preferred_lang == "auto"` (default): tier `0` = any `lang != "und"` with `n_cues >= 0.7 * max_cues`, tier `1` = other `lang != "und"`, tier `2` = `"und"`.
     - Tie-breakers across all tiers: `(-n_cues, 0 if lang not in ("en", "und") else 1, lang)` — preferring larger cue counts, then non-English identified languages over English/`"und"` when cue counts are comparable.
