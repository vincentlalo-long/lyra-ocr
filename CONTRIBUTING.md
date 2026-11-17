# Contributing to Lyra OCR

Thank you for your interest in contributing to **Lyra OCR**! This guide covers local environment setup, repository layout, adding or refining language profiles, testing/linting standards, and pull request conventions.

---

## 1. Prerequisites

- **Python**: `>= 3.11` (CI tests on Python 3.11 and 3.12)
- **System binaries**: `ffmpeg` and `ffprobe` available on `PATH`
- **Optional GPU acceleration**: CUDA 12 / cuDNN for `onnxruntime-gpu` and `faster-whisper` (CPU execution works out of the box via `onnxruntime`)

---

## 2. Development Setup

Clone the repository, create a virtual environment, and install the package in editable mode with development and evaluation extras (`pyproject.toml`):

```bash
git clone https://github.com/vincentlalo-long/lyra-ocr.git
cd lyra-ocr

# Using standard venv
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,eval]"

# Or using uv
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[dev,eval]"
```

*(Optional CUDA support for ONNX Runtime / `faster-whisper`)*:

```bash
pip uninstall -y onnxruntime
pip install "onnxruntime-gpu[cuda,cudnn]==1.31.1"
```

Verify the installation:

```bash
lyra-ocr --help
pytest tests/unit -q
```

---

## 3. Codebase Tour

Lyra OCR is organized into a two-stage pipeline plus evaluation, dataset, and local web tooling under `src/lyra_ocr/`:

### Entry Points
- `src/lyra_ocr/cli.py` — CLI dispatcher (`lyra-ocr` subcommands: `smoke`, `identify`, `label`, `eval`, `review-ui`, `lrc`).
- `src/lyra_ocr/config.py` — Centralized, immutable frozen threshold dataclasses (`LayerDiscoveryThresholds`, `GroupingThresholds`, `ClassificationThresholds`, `OcrPreprocessThresholds`).

### Phase 1: Video Ingestion, OCR, Tracking & Classification (`identify` / `smoke`)
- `src/lyra_ocr/video/` — Container inspection (`probe.py`), PTS-synchronized frame sampling (`frames.py`), perceptual frame deduplication (`dedup.py`), and media download helper (`downloader.py`).
- `src/lyra_ocr/ocr/` — Full-frame DBNet text detection and 2-line quad splitting (`detect.py`), sequence recognition with Latin fallback, soft B&W stroke normalization, and visual ink-valley word-space reconstruction (`engine.py`), plus ONNX Runtime device selection (`runtime.py`).
- `src/lyra_ocr/track.py` & `src/lyra_ocr/trackers/` — Text-identity-aware temporal association (`base.py`, `greedy.py`, `bytetrack.py`) and fragmented track merging.
- `src/lyra_ocr/classify.py` — Heuristic classification of `TextTrack` objects into `lyric` vs. `non_lyric` (watermarks, intro/outro cards, chat rails, silent overlays).
- `src/lyra_ocr/pipeline.py` & `src/lyra_ocr/provenance.py` — End-to-end Phase 1 orchestration and artifact provenance metadata.

### Phase 2: Layers, Audio Signals, Reference Alignment & LRC Export (`lrc`)
- `src/lyra_ocr/layers.py` — Spatial 2D layer clustering (`lyric`, `overlay`, `sparse`) and layer-level language voting.
- `src/lyra_ocr/lang.py` — Fine-grained per-line language detection (`en`, `ja`, `ja-Latn`, `es`, `vi`, `zh`, `ko`, `und`), Romaji scoring, and layer language voting.
- `src/lyra_ocr/audio/` — 16 kHz mono PCM extraction (`extract.py`), band-pass vocal activity detection (`vad.py`), word-level `faster-whisper` ASR (`asr.py`), and 10 ms onset / proportional word timing alignment (`forced_align.py`).
- `src/lyra_ocr/lyric/` —
  - `models.py`: `LyricCue`, `LrcMetadata`, `CorrectionConfig`, `ReferenceConfig`.
  - `select.py`, `correct.py`, `text_repair.py`: Human label overrides, quality gate, consensus medoid selection, and English-scoped SymSpell repair.
  - `reference.py` (facade over `ref_models.py`, `ref_providers.py`, `ref_align.py`) & `captions.py`: LRCLIB / YouTube caption providers, SHA-1 caching, monotonic DP alignment, and VAD-checked gap filling.
  - `layer_consolidation.py` & `grouping.py`: Duplicate band merging, parallel bilingual layer separation, same-Y flicker collapse, simultaneous multi-line resolution, fade deduplication, and cross-layer cue sharing.
  - `timing.py`, `export.py`, `build.py`: Dense boundary refinement, multi-layer `.lrc` / Enhanced LRC / `.cues.json` emission, and Phase 2 orchestration.

### Labeling, Evaluation & Interactive Web UI (`label`, `eval`, `review-ui`)
- `src/lyra_ocr/labels.py` & `src/lyra_ocr/review.py` — Blind CSV review export/ingest and `labels.jsonl` persistence.
- `src/lyra_ocr/dataset/` & `src/lyra_ocr/eval/` — Golden dataset registry, tracking/LRC evaluation metrics, and report/chart generation.
- `src/lyra_ocr/web/` — Local HTTP review & LRC Studio server (`lyra-ocr review-ui`), split into `server.py`, `base.py`, `review_api.py`, `lrc_api.py`, `dataset_api.py`, `eval_api.py`, `jobs.py`, `runners.py`, and `static/`.

For architectural details and data schemas, see [doc/ARCHITECTURE.md](doc/ARCHITECTURE.md) and [doc/SPECIFICATION.md](doc/SPECIFICATION.md).

---

## 4. How to Add or Improve a Language Profile

Before modifying language rules, review [doc/LANGUAGE_SUPPORT.md](doc/LANGUAGE_SUPPORT.md) to see how coarse OCR script routing (`ocr/engine.py`) interacts with fine-grained lyric language classification (`lang.py`).

Follow these steps in order:

1. **Line & Layer Language Detection (`src/lyra_ocr/lang.py`)**
   - Add Unicode script ranges or lexical/diacritic heuristics to `detect_line_lang()`.
   - Place unambiguous scripts (e.g., Kana, Hangul, Thai) before Latin disambiguation. For Latin-script languages, order checks carefully against `es`, `vi`, and `ja-Latn` so shared diacritics or short words do not create false positives on English (`en`) or other supported languages.
   - Update `vote_layer_lang()` if lines in one script should promote or harmonize with another within the same spatial layer (analogous to `zh` promoting to `ja` when Kana is present).

2. **Layer Consolidation & Parallel Bilingual Separation (`src/lyra_ocr/lyric/layer_consolidation.py` & `src/lyra_ocr/config.py`)**
   - If the language commonly appears as a parallel subtitle stream alongside another language (e.g., dual-language music videos), add a layer rate estimator and detection branch in `_detect_parallel_bilingual_langs()`.
   - Store any new numeric thresholds on `GroupingThresholds` in `src/lyra_ocr/config.py` rather than hardcoding constants inside algorithm functions.

3. **Downstream Pipeline Verification**
   - **OCR script routing & spacing (`src/lyra_ocr/ocr/engine.py`)**: Confirm `detect_script()` routes crops to an appropriate RapidOCR model and that `reconstruct_visual_word_spaces()` only runs on spaced scripts.
   - **Repair & quality gate (`src/lyra_ocr/lyric/text_repair.py`, `select.py`)**: Confirm English SymSpell repair remains restricted to `track_lang == "en"` and non-English lines bypass `_quality_gate()` via `is_non_en`.
   - **Unspaced tokenization & wrap joining (`src/lyra_ocr/lyric/ref_models.py`, `audio/forced_align.py`, `lyric/grouping.py`)**: For unspaced scripts, update token regexes and character-count wrap limits.
   - **ASR language normalization (`src/lyra_ocr/audio/asr.py`)**: Verify `_normalize_whisper_lang()` maps any BCP-47 subtag to the Whisper base code.

4. **Unit Tests (`tests/unit/`)**
   - Add targeted unit tests in `tests/unit/test_lang.py` (line detection, edge cases, and negative/English guards), plus tests in `tests/unit/test_lrc_select.py`, `tests/unit/test_lrc_grouping.py`, and `tests/unit/test_lrc_export.py` verifying OOV protection, bilingual layer consolidation, and `<name>.<lang>.lrc` export.

---

## 5. Testing, Linting & Commit Conventions

### Testing & TDD Expectations
- Follow **test-driven development** for bug fixes and new features: reproduce the failure or specify the new behavior in a unit test first, implement the minimal change, and verify the full unit suite passes.
- Keep `tests/unit/` fast, deterministic, and offline (mock `ffmpeg`, ONNX Runtime, Whisper, LRCLIB, and `yt-dlp` network/subprocess calls). Mark hardware- or media-heavy tests with `@pytest.mark.slow` or `@pytest.mark.gpu`.

Run unit tests and linting locally before pushing (matches `.github/workflows/ci.yml`):

```bash
# Fast unit suite (excludes slow and gpu markers per pyproject.toml)
pytest tests/unit -q

# If running without an editable install
PYTHONPATH=src .venv/bin/python -m pytest tests/unit -q

# Linter (ruff settings in pyproject.toml)
ruff check src tests
```

### Commit Messages
Use [Conventional Commits](https://www.conventionalcommits.org/) with an optional subsystem scope:

- `feat(scope): ...` — new capability or pipeline stage
- `fix(scope): ...` — bug fix
- `refactor(scope): ...` — internal restructuring without behavior change
- `docs(scope): ...` — documentation updates
- `test(scope): ...` — adding or updating tests
- `chore(scope): ...` — dependency, CI, or repository maintenance

---

## 6. Pull Request Workflow

1. Fork the repository and create a topic branch from `main`.
2. Make focused commits with clear tests and updated documentation (`doc/` or `README.md`) when behavior or CLI flags change.
3. Run `ruff check src tests` and `pytest tests/unit -q` and confirm all checks pass.
4. Open a pull request filling out `.github/pull_request_template.md`.
