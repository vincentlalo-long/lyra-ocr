# Lyra OCR

A modular pipeline and command-line toolchain for video frame sampling, scene text detection, and lyric OCR (English / Romaji / Japanese).

---

## 🗺️ Roadmap & Phases

> **Current Progress:** **Phase 0 (Foundations) — Completed ✅ (English / Romaji / Japanese verified)**

| Phase | Milestone | Purpose | Status |
|:---:|---|---|:---:|
| **0** | **Foundations** | Video probing, PTS frame extraction, full-frame detection, and decoupled OCR engine. | Completed ✅ |
| **1** | **Lyric Identification** | Temporal text tracking to filter out watermarks, channel logos, and title cards from actual lyrics. | Upcoming ⏳ |
| **2** | **Timing & LRC** | Refining cue timestamps and exporting standard line-level `.lrc` lyric files. | Planned 📋 |
| **3** | **Trained Classifier** | Training lightweight models on user-confirmed review data to automate filtering. | Planned 📋 |
| **4** | **Audio & Karaoke** | Voice alignment and syllable tracking for word-by-word karaoke synchronization. | Planned 📋 |

---

## 📖 Documentation

- [System Architecture & Design Philosophy](doc/ARCHITECTURE.md): Layered subsystem decomposition, domain entities, decoupled vision pipeline, and fundamental architectural decisions (ADRs).
- [Data Contracts & Artifact Specifications](doc/SPECIFICATION.md): Formal schemas, coordinate systems, and serialization contracts for all pipeline artifacts (`manifest.json`, `frame_index.json`, `ocr_raw.json`, `tracks.json`, `review.csv`).
- [Mathematical Foundations of DBNet (PDF)](doc/research/dbnet-mathematics.pdf): Theoretical monograph covering Differentiable Binarization, gradient mechanics, adaptive threshold calibration, and polygon decoding geometry.

---

## 🔮 Up Next

- **Phase 1: Temporal Text Tracking**: Grouping frame-by-frame text detections into continuous `TextTrack` streams using spatial IoU overlap and string edit-distance continuity.
- **Heuristic Distractor Discrimination**: Filtering out persistent channel watermarks, stationary logos, and introductory title cards from actual dynamic lyrics.
- **Review Artifact Export**: Generating structured `tracks.json` and tabular `review.csv` for human-in-the-loop verification.

---

## 📌 Current Features

- **Video Metadata Inspection** ([`src/lyra_ocr/probe.py`](src/lyra_ocr/probe.py)): Analyzes video files using `ffprobe` to extract resolution, FPS, duration, rotation, and audio stream details.
- **Precision PTS Frame Sampling** ([`src/lyra_ocr/frames.py`](src/lyra_ocr/frames.py)): Samples frames at a configurable rate (default: 2.0 FPS) while recording exact Presentation Time Stamps (PTS) from container metadata.
- **Full-Frame Scene Text Detection** ([`src/lyra_ocr/detect.py`](src/lyra_ocr/detect.py)): Detects text boxes across the entire frame without hardcoded region-of-interest assumptions using DBNet (`PP-OCRv6_det_small`).
- **Text Recognition & Normalization** ([`src/lyra_ocr/ocr.py`](src/lyra_ocr/ocr.py)): Crops and recognizes text lines with RapidOCR (`PP-OCRv6_rec_small`), normalizes Unicode text to NFC, and classifies script types (`latin`, `japan`, `mixed`).
- **Smoke Test & Audit CLI** (`lyra-ocr smoke`): Generates audit-ready artifacts including `manifest.json`, `ocr_raw.json`, `frame_index.json`, and visual annotated preview images (`roi_preview.jpg`).
- **Local Video Downloader** ([`src/lyra_ocr/web.py`](src/lyra_ocr/web.py)): A lightweight local server providing background job management for video downloads.
- **Automated Test Suite**: Unit test suite covering probe extraction, PTS indexing, Unicode normalization, script detection, and artifact schemas.

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

### 2. Run OCR Smoke Test via CLI

Run full-frame detection and recognition on any local video file:

```bash
lyra-ocr smoke path/to/video.mp4 --out ./smoke_output --sample-frames 5 --start-sec 10.0
```

Outputs generated in `./smoke_output`:
- `manifest.json`: Tool versions, model metadata, and video configuration.
- `ocr_raw.json`: Frame-by-frame text lines with both 4-point polygons, bounding boxes, raw/normalized text, and confidence scores.
- `frame_index.json`: Extracted frame indices mapped to millisecond timestamps.
- `roi_preview.jpg`: High-resolution annotated image highlighting detected text boxes and recognized labels.
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
