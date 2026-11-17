# Third-Party Notices and Attributions

**Lyra OCR** (`lyra-ocr`) itself is licensed under the **MIT License** (see [LICENSE](LICENSE)).

This document lists the third-party software libraries, pretrained machine learning models, linguistic dictionaries, external CLI tools, and online services used by `lyra-ocr`, along with their applicable licenses, attribution notices, runtime caching/invocation behavior, and user obligations. Unless explicitly stated otherwise, `lyra-ocr` does not vendor third-party model weights or system binaries directly in this repository; they are installed via Python packaging, loaded from system `PATH`, or downloaded at runtime by their respective upstream libraries.

---

## Summary Overview

| Component | Category | License | Usage in `lyra-ocr` |
| :--- | :--- | :--- | :--- |
| **RapidOCR** (`rapidocr`) | OCR Runtime | Apache-2.0 | Text detection & recognition pipeline (`ocr/detect.py`, `ocr/engine.py`) |
| **PaddleOCR Models** (`PP-OCRv6`, `PP-OCRv5`) | Pretrained Models | Apache-2.0 | `PP-OCRv6_det_small`, `PP-OCRv6_rec_small`, `latin_PP-OCRv5_rec_mobile` |
| **faster-whisper** | ASR Wrapper | MIT | Word-level ASR (`audio/asr.py`) |
| **CTranslate2** | Inference Engine | MIT | Backend used by `faster-whisper` |
| **OpenAI Whisper Models** | Pretrained Models | MIT | Speech recognition weights (default `small`) |
| **SymSpellPy** (`symspellpy`) | Spell Repair Code | MIT | Offline English split/spell repair (`lyric/text_repair.py`) |
| **SymSpell English Frequency Dictionaries** | Lexical Data | **CC-BY-4.0** | Word & bigram frequency lexicons bundled with `symspellpy` |
| **PyKakasi** (`pykakasi`) | Japanese NLP | **GPL-3.0-or-later** | Runtime Japanese Kanji/Kana -> Hepburn Romaji (`ref_providers.py`, `asr.py`) |
| **Jaconv** (`jaconv`) | Japanese Text Utils | MIT | Katakana -> Hiragana normalization prior to `pykakasi` |
| **OpenCV** (`opencv-python`) | Vision / Image Ops | Apache-2.0 | Frame/crop processing, CLAHE, perspective warps |
| **NumPy** (`numpy`) | Numerical Computing | BSD-3-Clause | Array operations across video, audio, OCR, and eval |
| **SciPy** (`scipy`) | Scientific Computing | BSD-3-Clause | Signal filtering (`audio/`), Hungarian matching (`trackers/`, `eval/`) |
| **ONNX Runtime** (`onnxruntime` / `-gpu`) | Inference Engine | MIT | CPU/CUDA ONNX execution for RapidOCR (`ocr/runtime.py`) |
| **Pillow** (`pillow`) | Imaging | MIT-CMU (HPND) | Audit overlay rendering (`review.py`, `cli.py`) |
| **PyYAML** (`pyyaml`) | Configuration | MIT | Runtime dependency (and RapidOCR YAML config parsing) |
| **yt-dlp** (`yt-dlp`) | Media / Metadata CLI & Lib | Unlicense | Optional YouTube video/caption/playlist fetch |
| **Matplotlib** (`matplotlib`, `[eval]`) | Plotting (Optional) | PSF / Matplotlib | Evaluation chart rendering (`eval/plots.py`) |
| **Demucs** (`demucs`, Optional) | Vocal Separation | MIT | Optional subprocess vocal isolation (`audio/extract.py`) |
| **FFmpeg / ffprobe** | External System CLI | LGPL-2.1+ / GPL-2+ | Frame extraction, probing, and 16 kHz PCM audio extraction |
| **LRCLIB API** | External Web Service | Service Terms / Copyright | Optional online lyrics lookup (`lyric/ref_providers.py`) |
| **YouTube Captions** | External Web Service | YouTube ToS / Copyright | Optional secondary fallback captions (`lyric/captions.py`) |

---

## 1. OCR Engine & Pretrained Models (RapidOCR & PaddleOCR)

- **RapidOCR (`rapidocr>=3.0.0`)**
  - **License:** Apache License, Version 2.0 (`Apache-2.0`)
  - **Copyright:** RapidOCR Authors
  - **Upstream:** <https://github.com/RapidAI/RapidOCR> / <https://rapidai.github.io/RapidOCRDocs>
- **PaddleOCR Models (`PP-OCRv6` / `PP-OCRv5`)**
  - **License:** Apache License, Version 2.0 (`Apache-2.0`)
  - **Copyright:** Baidu, Inc. and/or applicable PaddleOCR rights holders
  - **Upstream:** <https://github.com/PaddlePaddle/PaddleOCR>

**Usage & Attribution Notice:**
`lyra-ocr` invokes RapidOCR through `src/lyra_ocr/ocr/detect.py` (`TextDetector`), `src/lyra_ocr/ocr/engine.py` (`OCREngine`), and `src/lyra_ocr/ocr/runtime.py` (`rapidocr_params`), with text angle classification disabled (`Global.use_cls = False`). Specifically:
- **Detection:** RapidOCR DBNet detector (defaulting to `PP-OCRv6_det_small` / upstream `PP-OCRv6_small_det`).
- **Recognition:** Default unified model `PP-OCRv6_rec_small` (upstream `PP-OCRv6_small_rec`), plus secondary Latin fallback `latin_PP-OCRv5_rec_mobile` (`OCRVersion.PPOCRV5`, `LangRec.LATIN`, `ModelType.MOBILE`) for unspaced or low-confidence Latin crops.

RapidOCR distributes converted ONNX model representations under Apache-2.0 while upstream weight copyright remains with Baidu / PaddleOCR rights holders. Model weights are downloaded/cached by RapidOCR at runtime and are not stored in this repository. Use of Baidu, PaddlePaddle, PaddleOCR, or RapidOCR trademarks must not imply endorsement.

---

## 2. Speech Recognition & Acoustic Models (`faster-whisper`, `CTranslate2`, OpenAI Whisper)

- **faster-whisper (`faster-whisper>=1.0.0`)**
  - **License:** MIT License
  - **Copyright:** SYSTRAN
  - **Upstream:** <https://github.com/SYSTRAN/faster-whisper>
- **CTranslate2 (`ctranslate2`)**
  - **License:** MIT License
  - **Copyright:** OpenNMT
  - **Upstream:** <https://opennmt.net> / <https://github.com/OpenNMT/CTranslate2>
- **OpenAI Whisper Models**
  - **License:** MIT License
  - **Copyright:** OpenAI
  - **Upstream:** <https://github.com/openai/whisper>

**Usage & Runtime Behavior:**
`src/lyra_ocr/audio/asr.py` (`transcribe_audio_words`) uses `faster_whisper.WhisperModel` (default `model_size="small"`, `device="auto"`, falling back to `device="cpu", compute_type="int8"`) with `word_timestamps=True` and `vad_filter=True` on 16 kHz mono float32 audio. CTranslate2-converted Whisper model weights are downloaded at runtime (e.g., via Hugging Face Hub) on first use; resulting transcripts are cached locally in `<artifacts_dir>/asr_transcript.json`.

---

## 3. Spelling Repair & Frequency Dictionaries (`SymSpellPy` & Wolf Garbe Dictionaries)

- **SymSpellPy (`symspellpy>=6.7.7`)**
  - **License:** MIT License
  - **Copyright:**
    - Copyright (c) 2025 mmb L (Python port: <https://github.com/mammothb/symspellpy>)
    - Copyright (c) 2021 Wolf Garbe (Original SymSpell C# implementation: <https://github.com/wolfgarbe/SymSpell>)
- **English Frequency Dictionary Data** (`frequency_dictionary_en_82_765.txt`, `frequency_bigramdictionary_en_243_342.txt`)
  - **License:** **Creative Commons Attribution 4.0 International (CC BY 4.0)** (<https://creativecommons.org/licenses/by/4.0/>)
  - **Attribution:** Wolf Garbe / SymSpell English frequency dictionaries (derived from Google Books Ngram data and the Spell Checker Oriented Word Lists / SCOWL).

**Usage & Runtime Behavior:**
`src/lyra_ocr/lyric/text_repair.py` loads `frequency_dictionary_en_82_765.txt` and `frequency_bigramdictionary_en_243_342.txt` offline directly from the installed `symspellpy` package directory for conservative English word-boundary segmentation and spell correction. No network calls are made during dictionary repair. Redistribution of the dictionary data requires retaining Wolf Garbe's attribution and CC-BY-4.0 notice.

---

## 4. Japanese Transliteration & Kana Normalization (`PyKakasi` & `Jaconv`)

- **PyKakasi (`pykakasi>=2.3.0`)**
  - **License:** **GNU General Public License v3.0 or later (`GPL-3.0-or-later`)**
  - **Upstream:** <https://codeberg.org/miurahr/pykakasi>
- **Jaconv (`jaconv>=0.3.4`)**
  - **License:** MIT License
  - **Upstream:** <https://github.com/ikegami-yukino/jaconv>

**Usage & Licensing Notice:**
`lyra-ocr` imports `pykakasi` and `jaconv` conditionally at runtime inside `src/lyra_ocr/lyric/ref_providers.py` (`romanize_bundle`) and `src/lyra_ocr/audio/asr.py` (`_get_kakasi`, `_romanize_ja_text`) with `ImportError` fallbacks when unavailable. They are used exclusively to normalize Katakana okurigana to Hiragana (`jaconv.kata2hira`) and transliterate Japanese Kanji/Kana reference or ASR text into Hepburn Romaji (`ja-Latn`).

Because `pykakasi` is licensed under **GPL-3.0-or-later**, any downstream party that bundles or redistributes `pykakasi` together with `lyra-ocr` in a combined binary or packaged distribution must review and satisfy GPLv3+ compliance obligations for that combined distribution (or omit `pykakasi` if Japanese reference/ASR romanization is not required).

---

## 5. Core Runtime Python Libraries

The following core libraries are declared in `pyproject.toml` / `requirements.txt`:

- **OpenCV (`opencv-python>=4.8.0`)** — **Apache License 2.0** (<https://github.com/opencv/opencv-python>). Used for frame decoding/resizing, duplicate frame checks, perspective crop warping, CLAHE contrast enhancement, and border padding.
- **NumPy (`numpy>=1.24.0`)** — **BSD-3-Clause** (distribution metadata includes `BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0` for bundled components; <https://numpy.org>). Used across frame, audio, geometry, tracking, and evaluation arrays.
- **SciPy (`scipy>=1.10.0`)** — **BSD-3-Clause** (<https://scipy.org/>). Used for audio signal filtering (`audio/vad.py`, `audio/forced_align.py`), linear sum assignment (`trackers/bytetrack.py`, `eval/matching.py`), and Theil-Sen regression (`eval/lrc_eval.py`).
- **ONNX Runtime (`onnxruntime>=1.17.0`, optional `onnxruntime-gpu`)** — **MIT License** (Copyright Microsoft Corporation; <https://onnxruntime.ai>). Used in `ocr/runtime.py` and `provenance.py` to select CPU/CUDA execution providers for RapidOCR. Optional NVIDIA CUDA/cuDNN libraries installed via `onnxruntime-gpu[cuda,cudnn]` are governed by NVIDIA's proprietary EULA/redistributable terms.
- **Pillow (`pillow>=10.0.0`)** — **MIT-CMU / HPND License** (<https://python-pillow.org/>). Used in `review.py` and `cli.py` to render visual audit overlays (`Image`, `ImageDraw`, `ImageFont`).
- **PyYAML (`pyyaml>=6.0`)** — **MIT License** (<https://pyyaml.org/>). Declared as a core dependency and used by RapidOCR configuration loading.
- **yt-dlp (`yt-dlp>=2024.0.0`)** — **The Unlicense** (<https://github.com/yt-dlp/yt-dlp>). Used in `video/downloader.py`, `lyric/captions.py`, and `dataset/youtube.py`.

---

## 6. Optional Evaluation, Vocal Separation & Development Dependencies

- **Matplotlib (`matplotlib>=3.8.0`, `[project.optional-dependencies].eval`)** — **Python Software Foundation License / Matplotlib License** (<https://matplotlib.org>). Imported lazily in `src/lyra_ocr/eval/plots.py` (`render_evaluation_charts`) when `--charts` is requested; degrades cleanly if omitted.
- **Demucs (`demucs`, optional runtime integration)** — **MIT License** (Meta Platforms, Inc.; <https://github.com/facebookresearch/demucs>). Not listed in `pyproject.toml`; `src/lyra_ocr/audio/extract.py` (`separate_vocals_demucs`) checks `importlib.util.find_spec("demucs")` and invokes `python -m demucs --two-stems vocals` only if installed by the user.
- **pytest (`pytest>=8.0.0`, `[dev]`)** — **MIT License** (<https://docs.pytest.org/>). Test runner only.
- **Ruff (`ruff>=0.4.0`, `[dev]`)** — **MIT License** (<https://github.com/astral-sh/ruff>). Linter only.

---

## 7. External CLI Tools (`FFmpeg` / `ffprobe` & `yt-dlp`)

- **FFmpeg & ffprobe**
  - **License:** **LGPL v2.1+** or **GPL v2+** (depending on the user's installed build and enabled codecs such as `libx264`/`libx265`).
  - **Upstream:** <https://ffmpeg.org/>
  - **Usage:** `lyra-ocr` does not bundle FFmpeg binaries. `src/lyra_ocr/video/probe.py`, `src/lyra_ocr/video/frames.py`, `src/lyra_ocr/audio/extract.py`, and `src/lyra_ocr/provenance.py` invoke `ffmpeg` and `ffprobe` found on `PATH` as external subprocesses. Users are responsible for installing a build of FFmpeg compatible with their intended use and complying with its LGPL/GPL license terms.
- **yt-dlp**
  - **License:** The Unlicense (Public Domain).
  - **Usage:** Invoked programmatically via the `yt_dlp` Python API in `video/downloader.py` (video download), `lyric/captions.py` (metadata/caption extraction with `skip_download=True`), and `dataset/youtube.py` (flat channel/playlist enumeration with a 600-second in-memory cache).

---

## 8. External APIs & Online Services (`LRCLIB` & YouTube)

Online lookups are **opt-in** (disabled unless `--use-reference` or a remote YouTube URL is provided by the user):

- **LRCLIB (`https://lrclib.net/api`)**
  - **Usage (`src/lyra_ocr/lyric/ref_providers.py`):** Queries `GET /api/search` with track/artist metadata using `User-Agent: LyraOCR/1.0 (local lyric alignment)` and a 6.0-second per-request timeout (`DEFAULT_TIMEOUT = 6.0`).
  - **Caching & Rate-Limiting:** Successful results are cached atomically on disk under `<artifacts_dir>/reference_cache/<sha1>.json` (`load_cached_reference_bundle` / `save_cached_reference_bundle`) so subsequent exports reuse local data instead of re-querying LRCLIB (unless `--force` is passed or the cache is deleted).
  - **Obligations:** Users must respect LRCLIB's service availability and fair-use expectations and ensure any fetched lyrics are used in accordance with applicable copyright law.
- **YouTube & YouTube Captions**
  - **Usage (`src/lyra_ocr/video/downloader.py`, `src/lyra_ocr/lyric/captions.py`, `src/lyra_ocr/lyric/build.py`):** When `--use-reference` is enabled and LRCLIB yields no usable lines (and `--no-caption-fallback` is not set), `lyra-ocr` falls back to extracting official or auto-generated YouTube captions (never auto-translated non-English tracks) using `yt-dlp` (`skip_download=True`) and `urllib.request` (`DEFAULT_CAPTION_TIMEOUT = 6.0`s). `lyric/build.py` caches caption lookup bundles in `reference_cache` with a 3600-second TTL (`CAPTION_CACHE_TTL_SECONDS = 3600.0`).
  - **Obligations:** Users are solely responsible for complying with YouTube's Terms of Service, copyright restrictions, and local laws when downloading media, enumerating playlists/channels, or retrieving captions.

---

## 9. System Fonts (Optional Visual Overlays)

`src/lyra_ocr/review.py` (`_find_unicode_font`) opportunistically searches common Linux system font paths (Google Noto Sans / Noto Sans Mono CJK, DejaVu Sans, Liberation Sans) to render Unicode overlay previews, falling back to Pillow's built-in default font when none are present. No third-party font files are bundled in this repository.
