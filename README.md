# Lyra OCR

A modular pipeline and CLI toolchain for extracting hardcoded lyrics from video into synchronized `.lrc` subtitles (supporting English, Japanese Kanji/Kana, Romaji `ja-Latn`, and multilingual videos).

---

## Architecture & Current Capabilities

The pipeline runs in two main stages (via the `lyra-ocr` CLI or Web UI `lyra-ocr review-ui`):

1. **Detection & Temporal Tracking (`identify` / `smoke`)**:
   - Inspects stream metadata (`ffprobe`) and samples frames by PTS (`ffmpeg`, `pipe` or `disk` mode, optional `--dedup`).
   - Detects text boxes with RapidOCR DBNet (`src/lyra_ocr/ocr/detect.py`, including horizontal projection profile splitting for tall 2-line boxes) and recognizes text lines (`src/lyra_ocr/ocr/engine.py`, upscaling small crops `< 36px`, applying contrast-gated CLAHE, and border padding).
   - Associates frame detections into continuous `TextTrack` streams (`greedy` or `bytetrack`) and classifies `lyric` vs `non_lyric` using heuristic rules (`src/lyra_ocr/classify.py`: watermarks, intro title cards, chat overlays).
2. **Multilingual & Multimodal LRC Export (`lrc`)**:
   - Spatial layer discovery and language voting (`src/lyra_ocr/layers.py`, `src/lyra_ocr/lang.py`): separates concurrent language layers (`ja`, `ja-Latn`, `en`, etc.) while preserving stacked multi-line companion bands.
   - Offline text repair (`dictionary`: SymSpell English word-boundary/spell repair plus reference-free Hepburn Romaji confusable repair in `src/lyra_ocr/lyric/text_repair.py`).
   - Audio signals (`src/lyra_ocr/audio/`): 16kHz mono extraction, RMS VAD onset alignment (`forced_align.py`), and `faster-whisper` word-level ASR (`asr.py`, cached to `asr_transcript.json`) used to:
     - partition timestamps across simultaneous multi-line stanza cues (`resolve_simultaneous_multiline_cues`, falling back to proportional character-length staggering);
     - provide fallback reference alignment when external lyrics are unavailable;
     - gate low-confidence unvoiced overlays (`< 0.65`) or flag them with `asr_unvoiced` via `asr_vocal_gate_score`.
   - Optional reference alignment (`--use-reference`): prioritizes Manual > LRCLIB > YouTube captions, performs N:M monotonic DP alignment (`1:1`, `1:2`, `1:3`, `2:1`, `3:1`), and handles cross-script `ja -> ja-Latn` romanized repair with ateji protection and strict zero-CJK safety on Romaji layers.
   - Layer-aware export (`src/lyra_ocr/lyric/export.py`): writes `<name>.lrc` (primary layer: `ja-Latn` > non-English original > `en`), per-language `<name>.<lang>.lrc` files, and `<name>.cues.json` with bounding boxes, confidences, skipped tracks, and reference provenance.

### Current Limitations

- Text detection and multi-line splitting target **horizontal** text lines (vertical *tategaki* splitting is not implemented).
- `faster-whisper` runs directly on mixed 16kHz mono audio (no Demucs/UVR vocal separation step).
- On heavy background music or sung ateji not present in dictionaries/references, accuracy still depends on OCR crop readability and LRCLIB/ASR overlap.
- The benchmark metrics below cover **Phase 1** tracking/classification on the 25-video English `Fix_lyric_EN` subset; subsequent multilingual, ASR, and LRC features are verified through the unit/eval test suite and local runs.

---

## Documentation

- [System Architecture](doc/ARCHITECTURE.md): Subsystem decomposition, domain entities, design decisions, and academic/engineering references.
- [Data Contracts & Specifications](doc/SPECIFICATION.md): Artifact schemas (`manifest.json`, `frame_index.json`, `ocr_raw.json`, `tracks.json`, `review.csv`, `labels.jsonl`).
- [Language Support & Policy Matrix](doc/LANGUAGE_SUPPORT.md): Script detection, `LanguageProfile` registry, and language-specific OCR/repair policies.
- [Research Tracker & Literature Review](doc/RESEARCH_TRACKER.md): Deep-dive comparisons of academic papers and open-source GitHub repositories informing each pipeline stage.

---

## References & Technical Foundations

Lyra OCR combines techniques from recent peer-reviewed papers and battle-tested open-source repositories across OCR, multi-object tracking, and automatic lyrics transcription (ALT):

### Academic Papers
1. **Automatic Lyrics Transcription, Vocal Source Separation & RMS-VAD**
   - **Syed, J., Meresman Higgs, I., Cífka, O., & Sandler, M. (2025).** *"Exploiting Music Source Separation for Automatic Lyrics Transcription with Whisper."* IEEE ICME 2025 Workshop (`arXiv:2506.15514`). [https://arxiv.org/abs/2506.15514](https://arxiv.org/abs/2506.15514)  
     *Used in:* `src/lyra_ocr/audio/extract.py`, `src/lyra_ocr/audio/vad.py`, `src/lyra_ocr/audio/asr.py`, `src/lyra_ocr/lyric/build.py` — Isolating the vocal stem prior to computing normalized RMS-VAD ($\text{VAD}[n] = \text{RMS}_{\text{voc}}[n] / P_{95}(\text{RMS}_{\text{voc}})$) and gating `faster-whisper` transcription / intro-outro non-lyric overlays.
   - **Défossez, A. (2021) / Rouard, S., Massa, F., & Défossez, A. (2023).** *"Hybrid Spectrogram and Waveform Source Separation"* / *"Hybrid Transformers for Music Source Separation (HTDemucs)."* IEEE ICASSP 2023 (`arXiv:2211.08553`). [https://arxiv.org/abs/2211.08553](https://arxiv.org/abs/2211.08553)  
     *Used in:* `src/lyra_ocr/audio/extract.py` (`separate_vocals_demucs`).
   - **Fitzgerald, D. (2010).** *"Harmonic/Percussive Separation Using Median Filtering."* 13th International Conference on Digital Audio Effects (DAFx-10).  
     *Used in:* `src/lyra_ocr/audio/extract.py` (`isolate_vocals_midside_hpss`) — Zero-PyTorch Stereo Mid-Side + STFT median-filtering harmonic vocal isolation fallback.
   - **Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2023).** *"Robust Speech Recognition via Large-Scale Weak Supervision (Whisper)."* ICML 2023 (`arXiv:2212.04356`). [https://arxiv.org/abs/2212.04356](https://arxiv.org/abs/2212.04356)  
     *Used in:* `src/lyra_ocr/audio/asr.py` (`faster-whisper` word-level timestamp extraction, vocal onset verification, and multi-line stanza partitioning).

2. **Scene Text Stroke Decomposition, Binarization & OCR Detection/Recognition**
   - **Yang, M. et al. (2025).** *"DecompST: Decomposition Looking Useful for Scene Text Segmentation (Training-Free Text-Outline-Background CIE Lab Decomposition)."* (`arXiv:2512.00854`). [https://arxiv.org/abs/2512.00854](https://arxiv.org/abs/2512.00854)  
     *Used in:* `src/lyra_ocr/ocr/engine.py` (`_compute_border_fg_distance`, `to_soft_bw_stroke_crop`, `has_multicolor_strokes`) — CIE Lab border-referenced foreground distance with dark-outline peeling on bright backgrounds and bimodal chroma detection for dual-color karaoke highlight lines.
   - **Bonechi, S. et al. (2019).** *"Weak Supervision for Generating Pixel-Level Annotations in Scene Text Segmentation (CAM / Foreground-Background Contrast Normalization)."* Pattern Recognition Letters (`arXiv:1905.11229`). [https://arxiv.org/abs/1905.11229](https://arxiv.org/abs/1905.11229)  
     *Used in:* `src/lyra_ocr/ocr/engine.py` (`to_soft_bw_stroke_crop`) — Soft monochrome foreground-to-background contrast mask normalizing multi-colored karaoke strokes to uniform white-on-black before CRNN/SVTR recognition.
   - **Liao, M., Wan, Z., Yao, C., Chen, K., & Bai, X. (2020).** *"Real-Time Scene Text Detection with Differentiable Binarization (DBNet)."* AAAI 2020 (`arXiv:1911.08947`). [https://arxiv.org/abs/1911.08947](https://arxiv.org/abs/1911.08947)  
     *Used in:* `src/lyra_ocr/ocr/detect.py` (RapidOCR / PaddleOCR PP-OCR DBNet detector).

3. **Temporal Multi-Object & Video Text Tracking**
   - **Zhang, Y., Sun, P., Jiang, Y., Yu, D., Weng, F., Yuan, Z., Luo, P., Liu, W., & Wang, X. (2022).** *"ByteTrack: Multi-Object Tracking by Associating Every Detection Box."* ECCV 2022 (`arXiv:2110.06864`). [https://arxiv.org/abs/2110.06864](https://arxiv.org/abs/2110.06864)  
     *Used in:* `src/lyra_ocr/trackers/bytetrack.py` (`ByteTextTracker` two-stage high/low-confidence Hungarian box association).
   - **He, H. et al. (2024).** *"GoMatching: A Simple Baseline for Video Text Spotting via Long and Short Term Matching."* NeurIPS 2024 (`arXiv:2401.07080`). [https://arxiv.org/abs/2401.07080](https://arxiv.org/abs/2401.07080)  
     *Used in:* `src/lyra_ocr/trackers/base.py` & `src/lyra_ocr/lyric/select.py` (Turnover split detection & 1-frame cross-fade/jitter segment consolidation).

### Open-Source GitHub Repositories
- [**CixiSVesper/dsh-soundtry**](https://github.com/CixiSVesper/dsh-soundtry) — Multi-tier vocal stem isolation (`MDX-Net ONNX` / `Demucs` / `HPSS`), vocal level alignment, and Whisper `condition_on_previous_text=False` + hallucination filtering (`src/lyra_ocr/audio/extract.py`, `src/lyra_ocr/audio/asr.py`).
- [**YaoFANGUK/video-subtitle-extractor**](https://github.com/YaoFANGUK/video-subtitle-extractor) — Spatial Y-band slot clustering and frequency-based static watermark rejection (`src/lyra_ocr/layers.py`, `src/lyra_ocr/classify.py`).
- [**HurryPeng/MagiaTimeline**](https://github.com/HurryPeng/MagiaTimeline) — Progressive karaoke/typewriter prefix absorption and visual boundary timing refinement (`src/lyra_ocr/trackers/base.py`, `src/lyra_ocr/lyric/timing.py`).
- [**timminator/VideOCR**](https://github.com/timminator/VideOCR) — Zone-based temporal gap merging and edit-similarity deduplication (`src/lyra_ocr/lyric/grouping.py`, `src/lyra_ocr/lyric/layer_consolidation.py`).
- [**banjuman/vilm-lyrics-aligner**](https://github.com/banjuman/vilm-lyrics-aligner) & [**iamjrmh/usersync**](https://github.com/iamjrmh/usersync) — Vocal-stem Whisper word-timestamp alignment for LRC generation (`src/lyra_ocr/audio/asr.py`, `src/lyra_ocr/audio/forced_align.py`).
- [**RapidAI/RapidOCR**](https://github.com/RapidAI/RapidOCR) & [**SYSTRAN/faster-whisper**](https://github.com/SYSTRAN/faster-whisper) — Core ONNX Runtime OCR and CTranslate2 Whisper inference engines.

---

## Requirements

- **Python**: `>= 3.11`
- **System Tools**: `ffmpeg` and `ffprobe` on `PATH`
- **Core Dependencies** (`pyproject.toml`): `rapidocr>=3.0.0`, `onnxruntime>=1.17.0`, `opencv-python>=4.8.0`, `numpy>=1.24.0`, `pillow>=10.0.0`, `scipy>=1.10.0`, `yt-dlp>=2024.0.0`, `symspellpy>=6.7.7`, `pykakasi>=2.3.0`, `faster-whisper>=1.0.0`

---

## Getting Started

### 1. Installation

```bash
git clone https://github.com/vincentlalo-long/lyra-ocr.git
cd lyra-ocr
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,eval]"
```

*(Optional GPU support for ONNX / faster-whisper)*:

```bash
pip uninstall -y onnxruntime
pip install "onnxruntime-gpu[cuda,cudnn]==1.31.1"
```

### 2. Extract & Export `.lrc` Subtitles

```bash
# End-to-end from video to .lrc (artifacts saved under ./lyra_output/<video_stem>)
lyra-ocr lrc path/to/video.mp4 -o song.lrc --tracker bytetrack --device auto

# Enable online reference alignment (Manual / LRCLIB / YouTube captions fallback)
lyra-ocr lrc path/to/video.mp4 -o song.lrc --tracker bytetrack --use-reference

# Re-export .lrc from existing artifacts without re-running video OCR
lyra-ocr lrc --artifacts ./lyra_output/video -o song.lrc
```

Key options for `lyra-ocr lrc`:
- `-o, --out`: Output `.lrc` path (also writes `<name>.<lang>.lrc` per language layer and `<name>.cues.json`).
- `--artifacts`: Existing Phase 1 artifact directory to skip re-running OCR.
- `--tracker`: `greedy` (CLI default) or `bytetrack` (recommended for transitions/karaoke highlights).
- `--correct`: `none`, `consensus`, or `dictionary` (default: `dictionary`).
- `--refine`: Dense frame scan near track boundaries to refine timestamps.
- `--include-review`: Include cues marked `review` instead of filtering them out.
- `--use-reference` / `--no-caption-fallback`: Enable reference repair and gap filling via LRCLIB / YouTube captions.
- `--offset-ms`: Shift global LRC offset tag (`[offset:+/-ms]`).
- `--force`: Ignore cache and re-run the pipeline.

### 3. Identification, Labeling & Web UI (`identify`, `label`, `review-ui`)

```bash
# Run frame sampling, OCR, tracking, and lyric classification
lyra-ocr identify path/to/video.mp4 --out ./output --tracker bytetrack --device auto

# Blind offline CSV annotation workflow
lyra-ocr label export --artifacts ./output --out label_batch.csv --blind
lyra-ocr label ingest --artifacts ./output --csv label_batch.csv --annotator human

# Launch interactive Web UI (Runs / Review / Eval / Datasets / LRC Studio)
lyra-ocr review-ui --port 8000
```

---

## Testing & Evaluation

### Automated Tests

```bash
pytest
```

Per `pyproject.toml`, `pytest` runs fast unit and evaluation tests (`-m 'not slow and not gpu'`) by default.

### Phase 1 Benchmark (25-Video `Fix_lyric_EN` Subset)

Phase 1 track classification and association evaluated against automatic Silver Ground Truth across 25 English lyric videos (>1,350 text tracks):

| Metric | Baseline Heuristic | Greedy (Tuned) | ByteTrack (Tuned) |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 82.93% | 93.30% | 93.08% |
| **Lyric Precision** | 99.60% | 99.75% | 99.83% |
| **Lyric Recall** | 81.83% | 92.91% | 92.93% |
| **F1-Score** | 89.84% | 96.21% | 96.26% |
| **Generated Tracks** | 1,646 (30 vid) | 1,417 (25 vid) | 1,359 (25 vid) |

Evaluation commands:

```bash
lyra-ocr eval --golden path/to/dataset --out outputs/eval --charts --compare-trackers
python scripts/eval_lrc_text.py outputs/song/song.cues.json
```
