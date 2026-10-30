# Lyra OCR

A modular toolchain and local workspace for video frame sampling, scene text detection, and multilingual OCR.

---

## 🗺️ Roadmap & Phases

> **Current Progress:** **Phase 0 (Foundations) — EN ✅ / JP ✅ / VI In Progress ⏳**

| Phase | Milestone | Purpose | Status |
|:---:|---|---|:---:|
| **0** | **Foundations** | Video probing, PTS frame extraction, full-frame detection, and multilingual OCR engine. | EN ✅ / JP ✅ / VI ⏳ |
| **1** | **Lyric Identification** | Temporal text tracking to filter out watermarks, channel logos, and title cards from actual lyrics. | Upcoming ⏳ |
| **2** | **Timing & LRC** | Refining cue timestamps and exporting standard line-level `.lrc` lyric files. | Planned 📋 |
| **3** | **Trained Classifier** | Training lightweight models on user-confirmed review data to automate filtering. | Planned 📋 |
| **4** | **Audio & Karaoke** | Voice alignment and syllable tracking for word-by-word karaoke synchronization. | Planned 📋 |

---

## 🔮 Up Next

- **Vietnamese OCR Integration**: Integrating a dedicated Vietnamese recognition model (such as VietOCR) to accurately handle Vietnamese tone marks and diacritics.
- **Technical Analysis Documentation**: Preparing comprehensive architectural, technical analysis, and benchmark reports (to be published in `doc/`).
- **Web UI Feature Expansion**: Upgrading the local web interface from a simple video downloader into a visual workspace featuring video playback, frame scrubbing, ROI/box visualization, and interactive OCR inspection.

---

## 📌 Current Features

- **Video Metadata Inspection** ([`src/lyra_ocr/probe.py`](src/lyra_ocr/probe.py)): Analyzes video files using `ffprobe` to extract resolution, FPS, duration, rotation, and audio stream details.
- **Precision PTS Frame Sampling** ([`src/lyra_ocr/frames.py`](src/lyra_ocr/frames.py)): Samples frames at a configurable rate (default: 2.0 FPS) while recording exact Presentation Time Stamps (PTS) from container metadata.
- **Full-Frame Scene Text Detection** ([`src/lyra_ocr/detect.py`](src/lyra_ocr/detect.py)): Detects text boxes across the entire frame without hardcoded region-of-interest assumptions using DBNet (`PP-OCRv6_det_small`).
- **Multilingual Recognition & Normalization** ([`src/lyra_ocr/ocr.py`](src/lyra_ocr/ocr.py)): Crops and recognizes text lines with RapidOCR (`PP-OCRv6_rec_small`), normalizes Unicode text to NFC, and classifies script types (`latin`, `japan`, `mixed`).
- **Smoke Test & Audit CLI** (`lyra-ocr smoke`): Generates audit-ready artifacts including `manifest.json`, `ocr_raw.json`, `frame_index.json`, and visual annotated preview images (`roi_preview.jpg`).
- **Local Web Interface** ([`src/lyra_ocr/web.py`](src/lyra_ocr/web.py)): A lightweight local server providing background job management for video downloads, currently being expanded into an interactive OCR workstation.
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
