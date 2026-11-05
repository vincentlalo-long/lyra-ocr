# Lyra OCR

A modular pipeline and command-line toolchain for video frame sampling, scene text detection, and lyric OCR (English / Romaji / Japanese).

---

## 🗺️ Roadmap & Phases

> **Current Progress:** **Phase 1 (Lyric Identification & Tracking) — Implemented (Evaluation Pending) 🟡**

| Phase | Milestone | Purpose | Status |
|:---:|---|---|:---:|
| **0** | **Foundations** | Video probing, PTS frame extraction, full-frame detection, and decoupled OCR engine. | Completed ✅ |
| **1** | **Lyric Identification** | Temporal text tracking to filter out watermarks, channel logos, and title cards from actual lyrics. | Implemented (Pending Expanded Gate) 🟡 |
| **2** | **Timing & LRC** | Refining cue timestamps and exporting standard line-level `.lrc` lyric files. | Upcoming ⏳ |
| **3** | **Trained Classifier** | Training lightweight models on user-confirmed review data to automate filtering. | Planned 📋 |
| **4** | **Audio & Karaoke** | Voice alignment and syllable tracking for word-by-word karaoke synchronization. | Planned 📋 |

---

## 📖 Documentation

- [System Architecture & Design Philosophy](doc/ARCHITECTURE.md): Layered subsystem decomposition, domain entities, decoupled vision pipeline, and fundamental architectural decisions (ADRs).
- [Data Contracts & Artifact Specifications](doc/SPECIFICATION.md): Formal schemas, coordinate systems, and serialization contracts for all pipeline artifacts (`manifest.json`, `frame_index.json`, `ocr_raw.json`, `tracks.json`, `review.csv`, `labels.jsonl`).
- [Mathematical Foundations of DBNet (PDF)](doc/research/dbnet-mathematics.pdf): Theoretical monograph covering Differentiable Binarization, gradient mechanics, adaptive threshold calibration, and polygon decoding geometry.
- [Video Text Tracking & Semantic Discrimination (PDF)](doc/research/video-text-tracking-and-identification.pdf): Academic monograph covering two-level hierarchical tracking, ByteTrack/SORT/DBSCAN formulations, the negative elimination cascade, and CLEAR MOT evaluation metrics.

---

## 🔮 Up Next

- **Phase 2: Dense Boundary Timing & LRC Export**: Sub-sampling boundary frames around confirmed lyric tracks to pin start/end cues at centisecond precision and export standard line-level `.lrc` files.
- **Line Grouping & Multiline Stanzas**: Grouping multi-line lyric blocks and separating translation/secondary text tracks.

---

## 📌 Current Features

- **Video Metadata Inspection** ([`src/lyra_ocr/video/probe.py`](src/lyra_ocr/video/probe.py)): Analyzes video files using `ffprobe` to extract resolution, FPS, duration, rotation, and audio stream details.
- **Precision PTS Frame Sampling** ([`src/lyra_ocr/video/frames.py`](src/lyra_ocr/video/frames.py)): Samples frames at a configurable rate (default: 2.0 FPS) while recording exact Presentation Time Stamps (PTS) from container metadata.
- **Full-Frame Scene Text Detection** ([`src/lyra_ocr/ocr/detect.py`](src/lyra_ocr/ocr/detect.py)): Detects text boxes across the entire frame without hardcoded region-of-interest assumptions using DBNet (`PP-OCRv6_det_small`).
- **Text Recognition & Normalization** ([`src/lyra_ocr/ocr/engine.py`](src/lyra_ocr/ocr/engine.py)): Crops and recognizes text lines with RapidOCR (`PP-OCRv6_rec_small`), normalizes Unicode text to NFC, and classifies script types (`latin`, `japan`, `mixed`).
- **Temporal Text Tracking** ([`src/lyra_ocr/track.py`](src/lyra_ocr/track.py)): Clusters frame-level OCR detections across time into continuous `TextTrack` streams via spatial IoU and string edit-distance continuity.
- **Rule-Based Distractor Discrimination** ([`src/lyra_ocr/classify.py`](src/lyra_ocr/classify.py)): Classifies each track as `lyric` or `non_lyric`, while retaining diagnostic reasons for watermarks, title cards, and scenery text.
- **Human-in-the-Loop Review & Audit** ([`src/lyra_ocr/review.py`](src/lyra_ocr/review.py)): Automatically exports structured `tracks.json`, human-auditable `review.csv`, and color-coded visual verification overlays (`roi_preview.jpg`).
- **Lyric Identification & Smoke CLI** (`lyra-ocr identify` / `lyra-ocr smoke`): Complete command-line pipeline from raw video to classified lyric tracks.
- **Automated Test Suite**: Unit and evaluation tests covering probe extraction, PTS indexing, Unicode normalization, tracking association, rule classification, label persistence, Hungarian track matching, and metric calculations.

---

## ⚙️ Requirements

- **Python**: 3.11 or 3.12
- **System Tools**: `ffmpeg` and `ffprobe` available on `PATH`
- **Core Dependencies**: `rapidocr>=3.0.0`, `onnxruntime>=1.17.0`, `opencv-python>=4.8.0`, `numpy>=1.24.0`, `pillow>=10.0.0`, `yt-dlp>=2024.0.0`, `pytest>=8.0.0`, `scipy>=1.12.0`, `scikit-learn>=1.4.0`
- **Optional Dependencies**: `matplotlib>=3.8.0` (for publication evaluation charts)

---

## 🚀 Getting Started

### 1. Installation

```bash
git clone https://github.com/vincentlalo-long/lyra-ocr.git
cd lyra-ocr
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[eval]"
```

### 2. Run Lyric Identification (Phase 1)

Execute full-frame detection, recognition, temporal tracking, and rule-based lyric discrimination:

```bash
lyra-ocr identify path/to/video.mp4 \
  --out ./output \
  --title "Song Title" \
  --channel "Channel Name" \
  --tracker bytetrack \
  -n 10 -s 20.0
```

#### Supported Tracking Strategies (`--tracker`):
- `greedy` (default baseline): Spatial IoU + centroid distance greedy association.
- `bytetrack`: 2-stage association (Zhang et al., ECCV 2022) recovering low-confidence blurred detections via Hungarian assignment.
- `sort`: 8D Linear Kalman Filter motion prediction (Bewley et al., ICIP 2016) for kinetic/moving typography and camera pan/zoom.
- `dbscan`: Spatio-temporal tube density clustering (Sakaguchi et al., 2020) across $[cx/W, cy/H, t/T, \log(\text{area})]$.

Outputs generated in `./output`:
- `tracks.json`: Clustered `TextTrack` entities with frame-level `occurrences`, temporal statistics, segment transitions, and binary labels (`lyric`, `non_lyric`).
- `review.csv`: Tabular spreadsheet for human-in-the-loop audit (`track_id,start_ms,end_ms,machine_label,machine_decision,confidence,suggested_text,human_label,human_decision,review_note`).
- `labels.jsonl`: Append-only human annotation audit trail with hash provenance.
- `manifest.json`: Tool versions, model metadata, config hash, and video configuration.
- `ocr_raw.json`: Frame-by-frame text lines with 4-point polygons, bounding boxes, raw/normalized text, and confidence scores.
- `frame_index.json`: Extracted frame indices mapped to millisecond PTS timestamps.
- `roi_preview.jpg`: Color-coded visual audit preview (Green = Lyric, Red = Non-Lyric).
- `frames/`: Directory containing extracted JPEG frames.

### 3. Human Review & Annotation CLI

Export blind batch CSV for offline labeling, inspect status, or ingest annotations:

```bash
# Check labeling coverage
lyra-ocr label status --artifacts ./output

# Export blind review CSV (machine labels omitted to prevent bias)
lyra-ocr label export --artifacts ./output --out label_batch.csv --blind

# Ingest completed annotations into labels.jsonl
lyra-ocr label ingest --artifacts ./output --csv label_batch.csv --annotator reviewer_name
```

### 4. Run Evaluation

Evaluate predictions against a labeled dataset manifest with Hungarian matching and optional charts:

```bash
lyra-ocr eval --golden path/to/dataset --out outputs/eval --charts --compare-trackers
```

### 5. Run Automated Tests

```bash
pytest
```
