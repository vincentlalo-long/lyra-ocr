# Lyra OCR — Data Contracts & Artifact Specifications

This document defines the formal data contracts, coordinate systems, and serialization schemas used throughout the Lyra OCR pipeline.

---

## 1. Overview of Artifacts

Every execution pipeline persists standardized intermediate artifacts to allow inspection, regression testing, and modular processing without re-running expensive vision inference steps:

| Artifact | Generating Component | Description |
|---|---|---|
| `manifest.json` | Execution Coordinator | Run configuration, tool versions, model identifiers, and input metadata. |
| `frame_index.json` | Stream Synchronization Service | Mapping between frame identifiers, relative image file paths, and millisecond timestamps. |
| `ocr_raw.json` | Vision & Recognition Pipeline | Raw frame-by-frame spatial detections (polygons, bboxes) and recognized text lines. |
| `roi_preview.jpg` | Execution Coordinator | Representative annotated image frame for visual quality audit. |
| `tracks.json` *(Phase 1)* | Temporal Tracking Engine | Temporal tracks clustered across frames with assigned category labels and classification rationale. |
| `review.csv` *(Phase 1)* | Classification Subsystem | Tabular review sheet for rapid human-in-the-loop audit and confirmation. |

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
  "video_path": "/path/to/input_video.mp4",
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
  "tool_versions": {
    "python": "3.11.16",
    "rapidocr": "3.10.0",
    "onnxruntime": "1.31.1",
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

### 3.3. `ocr_raw.json`
Stores verbatim OCR outputs and spatial coordinates for every processed frame.

```json
{
  "video_path": "/path/to/input_video.mp4",
  "manifest_path": "manifest.json",
  "total_frames_processed": 5,
  "avg_frame_time_ms": 315.89,
  "frames": [
    {
      "frame_id": 0,
      "timestamp_ms": 20000,
      "pts_time": 20.0,
      "image_path": "frames/frame_000000.jpg",
      "elapsed_ms": 472.0,
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

---

## 4. Forward Specifications (Phase 1)

### 4.1. `tracks.json` Structure & Label Taxonomy

Each text box across temporal frames is clustered into a continuous `TextTrack` stream containing spatial-temporal features, segment transitions, and full frame-level `occurrences`:

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
        "dominant_text": "We can be alone in the castle in the sky"
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
      "confidence": 0.99,
      "reason": "Recurring text changes in stable subtitle zone",
      "decision": "accept",
      "qa_flags": []
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
