# Changelog

All notable changes and development milestones for **Lyra OCR** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and milestone entries below trace the project's progression from Phase 0 (`v0.1.0`) through architectural hardening (`v0.5.0`) on the pre-release `0.1.x` package series.

---

## [v0.5.0] — Architectural Hardening, Packaging & Community Governance (Unreleased)

### Added
- **Centralized configuration (`src/lyra_ocr/config.py`)**: Extracted tunable constants into immutable frozen dataclasses (`LayerDiscoveryThresholds`, `GroupingThresholds`, `ClassificationThresholds`, `OcrPreprocessThresholds`).
- **Modular reference & layer consolidation subsystems**:
  - Split `lyric/reference.py` into `lyric/ref_models.py`, `lyric/ref_providers.py`, and `lyric/ref_align.py` while keeping `reference.py` as a backwards-compatible facade.
  - Extracted cross-layer operations into `src/lyra_ocr/lyric/layer_consolidation.py` (`merge_duplicate_layers`, `consolidate_language_layers`, `share_solitary_cross_layer_cues`).
- **OCR stroke normalization (`src/lyra_ocr/ocr/engine.py`)**: Added color-invariant soft B&W stroke normalization (`to_soft_bw_stroke_crop`) and intra-word false-split protection.
- **Packaging, CI & legal documentation**: Added MIT `LICENSE`, PEP 621 metadata in `pyproject.toml`, `THIRD_PARTY_NOTICES.md`, and GitHub Actions CI (`.github/workflows/ci.yml` for Python 3.11 & 3.12).
- **Architecture & governance docs**: Updated `doc/ARCHITECTURE.md` and `doc/SPECIFICATION.md`, added `doc/LANGUAGE_SUPPORT.md`, and added `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `CHANGELOG.md`, and `.github/` issue/PR templates.

### Fixed
- **Language & quality gate accuracy (`lang.py`, `lyric/select.py`)**: Made Romaji and unaccented Spanish detection fault-tolerant (`_normalize_romaji_ocr_token`, expanded `_SPANISH_UNAMBIGUOUS_WORDS`), and prevented the English OOV quality gate from dropping non-English lyric layers.
- **Grouping & classification (`classify.py`, `lyric/grouping.py`, `lyric/layer_consolidation.py`)**: Collapsed same-Y flicker duplicates, preserved parallel bilingual subtitle streams (`ja-Latn` + `en`/`es`, `es` + `en`), guarded cross-layer cue sharing, accepted chorus title hooks inside active lyric layers, and rejected intro credit prefixes.
- **Web LRC Studio persistence (`web/lrc_api.py`, `lyric/build.py`)**: Persisted enhanced LRC toggle and cue metadata on save, prevented channel metadata leaks, and avoided multi-cue alignment guard review-count drift.

---

## [v0.4.0] — Multimodal Audio Signals, Enhanced LRC & Cross-Script Romaji Repair

### Added
- **Audio & VAD subsystem (`src/lyra_ocr/audio/`)**:
  - `extract.py`: 16 kHz mono float32 PCM extraction (`extract_audio_16k`, cached to `audio_16k.npy`).
  - `vad.py`: 4th-order Butterworth band-pass (300–3400 Hz) vocal activity profiling (`compute_vocal_activity_profile`, `vocal_active_ratio`).
  - `forced_align.py`: 10 ms vocal-onset snapping and proportional word timing alignment (`align_cues_with_audio`).
  - `asr.py`: Word-level `faster-whisper` ASR (`transcribe_audio_words`, cached to `asr_transcript.json`) for unvoiced overlay gating (`asr_vocal_gate_score`), split-line prefix onset verification (`asr_has_new_prefix_onset`), multi-line DP stanza partitioning (`align_multiline_cues_with_asr`), and fallback reference generation.
- **Enhanced LRC (`--enhanced`)**: Word-level `<mm:ss.xx>` karaoke timing tags and `.cues.json` `words` metadata across CLI and Web LRC Studio.
- **OCR crop & spacing upgrades (`ocr/detect.py`, `ocr/engine.py`)**: Small-crop upscaling with contrast-gated CLAHE and padding, tall 2-line box splitting (`split_multiline_crop_quads`), and column-wise visual ink-valley word space reconstruction (`reconstruct_visual_word_spaces`).
- **Cross-script Kanji-to-Romaji repair (`lyric/reference.py`, `lyric/text_repair.py`)**: Added `pykakasi`-backed Kanji-to-Romaji token repair with ateji protection and Hepburn confusable fixes, while setting **screen-faithful Romaji** (`repair_romaji = False`) as the default.

---

## [v0.3.0] — Multilingual Support, Spatial Layers & Web LRC Studio

### Added
- **Coarse script vs. fine-grained language detection (`src/lyra_ocr/lang.py`, `layers.py`)**:
  - Line classification (`detect_line_lang`, `romaji_score`) and spatial layer clustering/voting (`discover_layers`, `vote_layer_lang`) supporting `en`, `ja`, `ja-Latn`, `es`, `vi`, `zh`, and `ko`.
- **N:M Reference Alignment & Caption Fallback (`lyric/reference.py`, `lyric/captions.py`)**:
  - Monotonic dynamic programming alignment (`1:1`, `1:2`, `1:3`, `2:1`, `3:1`) against manual references, LRCLIB (`synced` / `plain`), and YouTube captions with SHA-1 disk caching and VAD-checked gap filling.
- **Multi-layer LRC export (`lyric/export.py`)**: Emits primary `<name>.lrc` and per-language `<name>.<lang>.lrc` sidecar files.
- **Web LRC Studio (`web/lrc_api.py`, `web/static/`)**: Interactive 60 fps video-synced LRC editor with layer/language switching, direct YouTube URL input, and artifact/run deletion APIs.

---

## [v0.2.0] — Phase 2 LRC Export & Evaluation (`lyra-ocr lrc`)

### Added
- **LRC pipeline (`src/lyra_ocr/lyric/`)**: Added `lyra-ocr lrc` CLI subcommand, `build_lrc_from_artifacts()`, `build_lrc_from_video()`, cue selection (`select.py`), consensus + SymSpell English dictionary repair (`correct.py`, `text_repair.py`), cue deduplication/grouping (`grouping.py`), and dense template-matching timing refinement (`timing.py`).
- **LRC evaluation (`src/lyra_ocr/eval/lrc_eval.py`, `visual_timing.py`, `scripts/eval_lrc_text.py`)**: Added visual timing delta metrics, text accuracy/order-inversion evaluation, canonical ground truth freezing, and segment-level track review round-trips (`review.py`, `labels.py`).

---

## [v0.1.0] — Phase 0 & Phase 1 Core Pipeline (Detection, Tracking, Classification & Review UI)

### Added
- **Video & OCR foundation (`video/`, `ocr/`)**: Container metadata probing (`ffprobe`), PTS-synchronized frame sampling (`ffmpeg` `pipe` and `disk` modes), full-frame RapidOCR DBNet detection (`PP-OCRv6_det_small`), perspective rectification, and `PP-OCRv6_rec_small` / `latin_PP-OCRv5_rec_mobile` recognition.
- **Tracking & classification (`track.py`, `trackers/`, `classify.py`)**: Implemented `greedy` and `bytetrack` trackers with text-turnover track splitting (`TURNOVER_SPLIT_SIMILARITY`), fragmented track merging, and heuristic `lyric` vs. `non_lyric` classification.
- **Artifact contracts, CLI & Web Review UI**: Established `manifest.json`, `frame_index.json`, `ocr_raw.json`, `tracks.json`, `review.csv`, and `labels.jsonl`; added `lyra-ocr smoke`, `identify`, `label`, `eval`, and `review-ui`.
