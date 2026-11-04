# Lyra OCR

A modular pipeline and command-line toolchain for video frame sampling, scene text detection, and lyric OCR (English / Romaji / Japanese).

---

## 🗺️ Roadmap & Phases

> **Current Progress:** **Phase 1 (Lyric Identification & Tracking) — Completed ✅**

| Phase | Milestone | Purpose | Status |
|:---:|---|---|:---:|
| **0** | **Foundations** | Video probing, PTS frame extraction, full-frame detection, and decoupled OCR engine. | Completed ✅ |
| **1** | **Lyric Identification** | Temporal text tracking to filter out watermarks, channel logos, and title cards from actual lyrics. | Completed ✅ |
| **2** | **Timing & LRC** | Refining cue timestamps and exporting standard line-level `.lrc` lyric files. | Upcoming ⏳ |
| **3** | **Trained Classifier** | Training lightweight models on user-confirmed review data to automate filtering. | Planned 📋 |
| **4** | **Audio & Karaoke** | Voice alignment and syllable tracking for word-by-word karaoke synchronization. | Planned 📋 |

---

## 📖 Documentation

- [System Architecture & Design Philosophy](doc/ARCHITECTURE.md): Layered subsystem decomposition, domain entities, decoupled vision pipeline, and fundamental architectural decisions (ADRs).
- [Data Contracts & Artifact Specifications](doc/SPECIFICATION.md): Formal schemas, coordinate systems, and serialization contracts for all pipeline artifacts (`manifest.json`, `frame_index.json`, `ocr_raw.json`, `tracks.json`, `review.csv`).
- [Mathematical Foundations of DBNet (PDF)](doc/research/dbnet-mathematics.pdf): Theoretical monograph covering Differentiable Binarization, gradient mechanics, adaptive threshold calibration, and polygon decoding geometry.

---

## 🔮 Up Next

- **Phase 2: Dense Boundary Timing & LRC Export**: Sub-sampling boundary frames around confirmed lyric tracks to pin start/end cues at centisecond precision and export standard line-level `.lrc` files.
- **Line Grouping & Multiline Stanzas**: Grouping multi-line lyric blocks and separating translation/secondary text tracks.

---

## 📌 Current Features

- **Video Metadata Inspection** ([`src/lyra_ocr/probe.py`](src/lyra_ocr/probe.py)): Analyzes video files using `ffprobe` to extract resolution, FPS, duration, rotation, and audio stream details.
- **Precision PTS Frame Sampling** ([`src/lyra_ocr/frames.py`](src/lyra_ocr/frames.py)): Samples frames at a configurable rate (default: 2.0 FPS) while recording exact Presentation Time Stamps (PTS) from container metadata.
- **Full-Frame Scene Text Detection** ([`src/lyra_ocr/detect.py`](src/lyra_ocr/detect.py)): Detects text boxes across the entire frame without hardcoded region-of-interest assumptions using DBNet (`PP-OCRv6_det_small`).
- **Text Recognition & Normalization** ([`src/lyra_ocr/ocr.py`](src/lyra_ocr/ocr.py)): Crops and recognizes text lines with RapidOCR (`PP-OCRv6_rec_small`), normalizes Unicode text to NFC, and classifies script types (`latin`, `japan`, `mixed`).
- **Temporal Text Tracking** ([`src/lyra_ocr/track.py`](src/lyra_ocr/track.py)): Clusters frame-level OCR detections across time into continuous `TextTrack` streams via spatial IoU and string edit-distance continuity.
- **Rule-Based Distractor Discrimination** ([`src/lyra_ocr/classify.py`](src/lyra_ocr/classify.py)): Accurately discriminates lyrics (`lyric`) from persistent channel logos (`channel`), intro title cards (`song_title`), and scenery text (`scene_text`).
- **Human-in-the-Loop Review & Audit** ([`src/lyra_ocr/review.py`](src/lyra_ocr/review.py)): Automatically exports structured `tracks.json`, human-auditable `review.csv`, and color-coded visual verification overlays (`roi_preview.jpg`).
- **Lyric Identification & Smoke CLI** (`lyra-ocr identify` / `lyra-ocr smoke`): Complete command-line pipeline from raw video to classified lyric tracks.
- **Local Video Downloader** ([`src/lyra_ocr/web.py`](src/lyra_ocr/web.py)): A lightweight local server providing background job management for video downloads.
- **Automated Test Suite**: 30 unit tests covering probe extraction, PTS indexing, Unicode normalization, tracking association, rule classification, and artifact serialization.

---

## ⚙️ Requirements

- **Python**: 3.11 or 3.12
- **System Tools**: `ffmpeg` and `ffprobe` available on `PATH`
- **Core Dependencies**: `rapidocr>=3.0.0`, `onnxruntime>=1.17.0`, `opencv-python>=4.8.0`, `numpy>=1.24.0`, `pillow>=10.0.0`, `yt-dlp>=2024.0.0`, `pytest>=8.0.0`

---

## 🚀 Getting Started

### 1. Installation

```bash
cd /home/danglong/projects/lyra-ocr
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### 2. Run Lyric Identification (Phase 1)

Execute full-frame detection, recognition, temporal tracking, and rule-based lyric discrimination:

```bash
lyra-ocr identify path/to/video.mp4 --out ./output --title "Song Title" --channel "Channel Name" -n 10 -s 20.0
```

Outputs generated in `./output`:
- `tracks.json`: Clustered `TextTrack` entities with temporal statistics, segment breakdown, assigned taxonomy labels (`lyric`, `channel`, `song_title`, `supplementary`, `scene_text`, `unknown`), and decision rationales.
- `review.csv`: Tabular spreadsheet for human-in-the-loop audit (`track_id,start_ms,end_ms,label,confidence,suggested_text,decision,review_note`).
- `manifest.json`: Tool versions, model metadata, and video configuration.
- `ocr_raw.json`: Frame-by-frame text lines with 4-point polygons, bounding boxes, raw/normalized text, and confidence scores.
- `frame_index.json`: Extracted frame indices mapped to millisecond PTS timestamps.
- `roi_preview.jpg`: Color-coded visual audit preview (Green = Lyric, Red = Channel/Watermark, Yellow = Song Title).
- `frames/`: Directory containing extracted JPEG frames.

### 3. Run Unit Tests

```bash
pytest tests/unit
```

### 4. Run Local Web Interface

```bash
python -m lyra_ocr
```

Access the web interface at <http://127.0.0.1:8765>.
