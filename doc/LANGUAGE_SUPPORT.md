# Lyra OCR — Language & Script Support Matrix

Lyra OCR separates **coarse OCR script routing** (`ocr/engine.py: detect_script`) from **fine-grained lyric language classification** (`lang.py: detect_line_lang`, `vote_layer_lang`). This separation enables script-appropriate OCR preprocessing, conservative text repair, bilingual layer splitting, and multi-layer `.lrc` export without distorting non-English lyrics.

---

## 1. Language Support Matrix

| Code | Language / Script | OCR Model Routing (`ocr/engine.py`) | Dictionary / NLP Repair (`correct.py`, `text_repair.py`) | English OOV / Quality Gate (`select.py`) | Reference Alignment & Gap Fill (`ref_providers.py`, `ref_align.py`) | Word Tokenization & Grouping (`forced_align.py`, `grouping.py`) | Default Extraction Mode |
|---|---|---|---|---|---|---|---|
| `en` | English (Latin) | `PP-OCRv6_rec_small` + `latin_PP-OCRv5_rec_mobile` fallback + soft B&W + visual space rebuild | Corpus consensus + edit-1 corpus vocab + SymSpell English lexicon/bigram repair (`repair_text`) | Enforced (`_quality_gate`); bypassed by human labels, `reference_match >= 0.55`, or trusted lyric layer (`conf >= 0.70`, `quality >= 0.20`) | LRCLIB (`synced`/`plain`), manual, YouTube captions, or `faster-whisper` ASR | Whitespace word tokens; eligible for `share_solitary_cross_layer_cues()` | Reference-assisted + conservative English repair |
| `ja` | Japanese (Kanji / Hiragana / Katakana) | `PP-OCRv6_rec_small` + soft B&W fallback (`script == "japan"` or `"mixed"`) | CJK tokens skipped (`_is_cjk`, `_CJK_RE`); `text_quality = 1.0` | Bypassed (`is_non_en = True`) | Aligns directly against `ja` reference or `ja` ASR | Character-level CJK/Kana tokens (`_tokenize_words`, `_ALIGN_TOKEN_RE`); multiline wraps join with `sep = ""` (`short_wrap_max_cjk_chars = 22`) | Screen-faithful + `ja` reference alignment |
| `ja-Latn` | Japanese Romaji (Latin) | `PP-OCRv6_rec_small` + `latin_PP-OCRv5_rec_mobile` fallback + soft B&W + visual space rebuild | **Disabled by default** (`CorrectionConfig.repair_romaji = False` -> consensus medoid only); optional `repair_romaji_confusables` when enabled | Bypassed (`is_non_en = True`) | **Disabled by default** (`ReferenceConfig.repair_romaji = False`); when enabled, `pykakasi` Hepburn romanization + `repair_romaji_cue_text` with particle equivalence (`wa`/`ha`, `e`/`he`) | Whitespace word tokens; Whisper language normalized `ja-Latn -> ja` | **Screen-faithful by default** (opt-in reference repair) |
| `es` | Spanish (Latin, accented or unaccented) | `PP-OCRv6_rec_small` + `latin_PP-OCRv5_rec_mobile` fallback + soft B&W + visual space rebuild | Non-ASCII accented chars protected (`_NON_ASCII_LATIN_RE`); SymSpell `repair_text` restricted to `track_lang == "en"` | Bypassed (`is_non_en = True`) — prevents English OOV gate from dropping unaccented Spanish | Aligns against `es` LRCLIB, manual, caption, or ASR lines | Whitespace word tokens; parallel `es` + `en` or `ja-Latn` + `es` streams separated cleanly | Screen-faithful + `es` reference alignment |
| `vi` | Vietnamese (Latin with diacritics) | `PP-OCRv6_rec_small` + `latin_PP-OCRv5_rec_mobile` fallback + soft B&W + visual space rebuild | Non-ASCII diacritics protected (`_NON_ASCII_LATIN_RE`); SymSpell `repair_text` restricted to `track_lang == "en"` | Bypassed (`is_non_en = True`) | Aligns against `vi` LRCLIB, manual, caption, or ASR lines | Whitespace word tokens | Screen-faithful + `vi` reference alignment |
| `zh` | Chinese (CJK Unified Ideographs without Kana/Hangul) | `PP-OCRv6_rec_small` + soft B&W fallback (`script == "japan"`) | CJK tokens skipped; `text_quality = 1.0` | Bypassed (`is_non_en = True`) | Aligns against `zh` LRCLIB, manual, caption, or ASR lines | Character-level CJK tokens; multiline wraps join with `sep = ""` (`short_wrap_max_cjk_chars = 22`) | Screen-faithful + `zh` reference alignment |
| `ko` | Korean (Hangul) | `PP-OCRv6_rec_small` + soft B&W fallback | Hangul tokens skipped (`_is_cjk` includes `\uac00-\ud7af`); `text_quality = 1.0` | Bypassed (`is_non_en = True`) | Aligns against `ko` LRCLIB, manual, caption, or ASR lines | Character-level Hangul tokens; multiline wraps join with `sep = ""` | Screen-faithful + `ko` reference alignment |
| `und` | Undetermined (empty, digits, punctuation only) | Default recognition | Consensus / fallback | Evaluated conservatively | Falls back to layer-voted language when present | Inherited from layer vote during `layers.py` / `export.py` | Inherited from spatial layer |

---

## 2. How Each Script & Language Is Detected

### 2.1. Coarse OCR Script Routing (`src/lyra_ocr/ocr/engine.py`)
`detect_script(text)` classifies each recognized OCR box for model routing and whitespace reconstruction:
- **`japan`**: Contains Hiragana (`U+3040–U+309F`) or Katakana (`U+30A0–U+30FF`), or contains CJK Unified Ideographs (`U+4E00–U+9FFF`) with no Latin characters.
- **`mixed`**: Contains both CJK Unified Ideographs and Latin alphabetic characters.
- **`latin`**: Any non-empty string with Latin letters (including `en`, `es`, `vi`, `ja-Latn`).
- **`empty`**: Empty or whitespace-only text.

When `script == "latin"` and either `len(text_norm) >= 8` with no spaces (`latin_alt_min_unspaced_len`) or recognition confidence `< 0.88` (`fallback_min_confidence`), `OCREngine.process_detections()` attempts a fallback pass with `latin_PP-OCRv5_rec_mobile`, optional soft B&W stroke normalization (`to_soft_bw_stroke_crop`), and column-wise ink-valley space reconstruction (`reconstruct_visual_word_spaces`).

### 2.2. Per-Line Language Classification (`src/lyra_ocr/lang.py`)
`detect_line_lang(text)` normalizes input with Unicode `NFC` and evaluates detectors in strict precedence order:

1. **Japanese (`ja`)**: Any Hiragana (`U+3040–U+309F`) or Katakana (`U+30A0–U+30FF`) character.
2. **Korean (`ko`)**: Any Hangul Syllable (`U+AC00–U+D7AF`), Hangul Jamo (`U+1100–U+11FF`), or Compatibility Jamo (`U+3130–U+318F`) character.
3. **Chinese (`zh`)**: CJK Unified Ideographs (`U+4E00–U+9FFF`) or Extension A (`U+3400–U+4DBF`) with no Kana, Hangul, or Latin letters.
4. **Spanish (`es`)**:
   - Triggered immediately by Spanish-distinct characters (`ñ`, `¿`, `¡` via `_SPANISH_CHARS_RE`), **or**
   - When Vietnamese-unique diacritics (`_VIETNAMESE_UNIQUE_RE`) are absent and unambiguous Spanish tokens (`_SPANISH_UNAMBIGUOUS_WORDS`, including accented and unaccented forms such as `corazón`/`corazon`, `más`/`mas`, `siempre`, `nunca`, `contigo`) appear `>= 2` times, or `>= 1` time alongside an ambiguous short Spanish token (`_SPANISH_AMBIGUOUS_SHORT`) or a Spanish accented vowel (`[áéíóúü]`).
   - Evaluated *before* Vietnamese so shared acute vowels (`á`, `é`, `í`, `ó`, `ú`) in Spanish lyrics do not misclassify as `vi`.
5. **Vietnamese (`vi`)**: Any Vietnamese diacritic or `đ` (`_VIETNAMESE_RE`), evaluated after Spanish disambiguation.
6. **Japanese Romaji (`ja-Latn`)**:
   - Computed via `romaji_score(text)` when Latin characters are present; `detect_line_lang()` returns `"ja-Latn"` if `romaji_score >= 0.6` and the line has `>= 2` alphabetic tokens.
   - `romaji_score()` checks each token against Hepburn/Kunrei-shiki mora structure (`_ROMAJI_TOKEN_RE`) with fault-tolerant OCR confusable normalization (`_normalize_romaji_ocr_token`, e.g. `l -> t`/`i`, `ll -> tt`, `ck -> ok` for scoring only).
   - **False-positive guards**: Requires `>= 50%` valid morae; excludes English stopwords (`_ENGLISH_STOPWORDS`) and English CVCe silent-`e` words (`_ENGLISH_SILENT_E_RE`, minus genuine 2-mora Romaji words like `kaze`, `yume`, `dare`, `sora` in `_GENUINE_2MORA_ROMAJI_E`); rejects short lines (`<= 4` tokens) with English words and zero positive Japanese particles/morphology (`_ROMAJI_PARTICLES_AND_FUNCTION`, `_ROMAJI_MORPHOLOGICAL_SUBSTRINGS`); and adds a `+0.15` boost when `>= 2` positive Japanese hits occur.
7. **English / Generic Latin (`en`)**: Any remaining text with alphabetic Latin characters.
8. **Undetermined (`und`)**: Empty, digits, or punctuation-only strings.

### 2.3. Layer-Level Language Voting & Bilingual Separation (`layers.py`, `lyric/layer_consolidation.py`)
- **Kanji Promotion (`vote_layer_lang`)**: Pure-Kanji lines initially tagged `zh` are promoted to `ja` whenever the same spatial layer contains `ja` (Kana) lines.
- **Preservation Across Re-Voting (`_revote_layer_languages`)**: Re-voting from consolidated cue text never downgrades an already-identified non-English layer (`ja`, `ja-Latn`, `es`, `vi`, `zh`, `ko`) back to the generic `"en"` Latin fallback.
- **Parallel Bilingual Layer Detection (`_detect_parallel_bilingual_langs`)**: When two vertically separated spatial bands (`abs(sec_y - dom_y) >= 0.035`) overlap temporally (`>= 45%`, at least `8` cues each, length ratio `>= 0.45`), `consolidate_language_layers()` checks `romaji_score` and `_layer_spanish_rate()` to preserve parallel bilingual pairs (`ja-Latn` + `en`, `ja-Latn` + `es`, `es` + `en`) as distinct `.<lang>.lrc` outputs instead of collapsing them.

---

## 3. Reference Repair vs. Screen-Faithful Extraction

Lyra OCR applies different correction policies depending on whether the on-screen text is English, non-English native script, or romanized Japanese:

### 3.1. When Reference Repair Applies
- **English (`en`)** and **native-script non-English layers (`ja`, `zh`, `ko`, `es`, `vi`)**:
  - When a manual reference, LRCLIB match (`lrclib_synced` / `lrclib_plain`), YouTube caption bundle, or `faster-whisper` ASR bundle (`asr_whisper`) matches the layer's voted language (`select_reference_lines()`), `align_reference()` performs monotonic many-to-many DP alignment.
  - Cue text is replaced when `similarity >= replace_similarity` (`0.85`, or `0.90` for captions), `token_coverage >= min_token_coverage` (`0.70`), and `length_ratio` is in `[0.60, 1.80]`.
  - Human-annotated cues (`correction.method == "human"`) are **never** overwritten by dictionary or reference repair.
  - Synchronized references (`manual`, `lrclib_synced`, `youtube_captions`) also populate missing inter-cue gaps via `fill_reference_gaps()`, guarded by `vocal_active_ratio >= 0.08`.
- **Non-Lyric Reference Rescue (`select.py`)**: If a track was classified `non_lyric` without a human rejection and without blocking card/metadata flags (`intro_bounded_card`, `metadata_title_match`, `only_intro_outro`, `social_handle_tag`, `metadata_channel_match`), a high-similarity match against `manual` or `lrclib_*` rescues it into the lyric stream with `qa_flags += ["reference_rescue"]`.

### 3.2. When Screen-Faithful Extraction Applies
- **Japanese Romaji (`ja-Latn`) — Default (`repair_romaji = False`)**:
  - Both `CorrectionConfig.repair_romaji` and `ReferenceConfig.repair_romaji` default to `False`.
  - `correct_text()` stops at spacing-aware consensus medoid selection (`method="consensus"`) and skips corpus-dictionary substitution and `repair_romaji_confusables()`.
  - `select_reference_lines()` returns `None` for `ja-Latn`, `build_lrc_from_artifacts()` skips `asr_to_reference_bundle()` for `ja-Latn`, and `align_reference()` / `fill_reference_gaps()` leave `ja-Latn` cues untouched without adding `reference_uncertain` flags.
  - **Rationale**: On-screen Romaji videos frequently use stylized spacing, ateji readings, or Hepburn/Waapuro variants (`wo`/`o`, `ha`/`wa`, `ou`/`oo`) that automated `pykakasi` romanization of Kanji references would overwrite inaccurately.
  - **Opt-In (`repair_romaji = True`)**: Enables `pykakasi` Hepburn conversion (`romanize_bundle`), `repair_romaji_confusables()`, and token-level `repair_romaji_cue_text()` with particle normalization (`wa`/`ha`, `e`/`he`, preserving `o`/`wo`) and zero-CJK safety stripping.
- **Accented Latin (`es`, `vi`) & CJK/Hangul (`ja`, `zh`, `ko`) without References**:
  - `text_repair.repair_text()` returns non-ASCII Latin and CJK/Hangul strings verbatim, `text_quality()` returns `1.0`, and `oov_ratio()` returns `0.0` so non-English lines are never mangled by the English SymSpell dictionary or dropped by `_quality_gate()`.

---

## 4. Adding a New Language Profile

To add first-class support for a new language (for example, French `fr`, Thai `th`, or Indonesian `id`), update the following touchpoints in order:

1. **Line & Layer Detection (`src/lyra_ocr/lang.py`)**:
   - Add script Unicode range checks or lexical/diacritic patterns in `detect_line_lang()`.
   - Place unambiguous script checks before Latin disambiguation, and ensure Latin-script languages are ordered carefully against `es`, `vi`, and `ja-Latn` (and do not trigger false positives on English).
   - Update `vote_layer_lang()` if the language shares characters with another script (analogous to `zh -> ja` promotion when Kana is present).
2. **Parallel Bilingual Consolidation (`src/lyra_ocr/lyric/layer_consolidation.py`, `src/lyra_ocr/config.py`)**:
   - If the language commonly co-occurs as parallel bilingual subtitles alongside `en` or another language, add a layer rate estimator and branch in `_detect_parallel_bilingual_langs()` (and add any thresholds to `GroupingThresholds` in `src/lyra_ocr/config.py`).
3. **OCR Script & Preprocessing (`src/lyra_ocr/ocr/engine.py`)**:
   - Verify `detect_script()` routes crops appropriately. If `PP-OCRv6_rec_small` cannot recognize the script accurately, register a script-specific RapidOCR model in `OCREngine._get_model()` and route to it in `OCREngine.process_detections()`.
   - Verify whether `reconstruct_visual_word_spaces()` should run (Latin/spaced scripts) or be skipped (unspaced scripts).
4. **Repair & Quality Gate Protection (`src/lyra_ocr/lyric/text_repair.py`, `src/lyra_ocr/lyric/correct.py`, `src/lyra_ocr/lyric/select.py`)**:
   - Confirm `_is_cjk` / `_NON_ASCII_LATIN_RE` or `track_lang == "en"` guards prevent `repair_text()` from mutating the new language using English SymSpell frequencies.
   - Confirm `is_non_en` in `select_cues()` (`track_lang not in ("en", "und")`) keeps `_quality_gate()` from rejecting valid lines.
5. **Tokenization & Multiline Grouping (`src/lyra_ocr/lyric/ref_models.py`, `src/lyra_ocr/audio/forced_align.py`, `src/lyra_ocr/lyric/grouping.py`)**:
   - If the script is unspaced (like Thai or CJK), add its Unicode range to `_ALIGN_TOKEN_RE` in `ref_models.py`, `_tokenize_words()` in `forced_align.py`, and `_is_cjk_heavy()` / `_try_merge_short_wrap()` in `grouping.py` so word-level Enhanced LRC tags and multiline wrap joins behave correctly.
6. **ASR & Reference Selection (`src/lyra_ocr/audio/asr.py`, `src/lyra_ocr/lyric/ref_providers.py`, `src/lyra_ocr/lyric/export.py`)**:
   - Add any subtag mapping in `_normalize_whisper_lang()` if Whisper expects a base ISO-639-1 code.
   - Verify `select_reference_lines()` and `write_lrc_layers()` emit `<name>.<lang>.lrc` cleanly.
7. **Unit Tests (`tests/unit/`)**:
   - Add unit tests covering `detect_line_lang()`, `vote_layer_lang()`, `select_cues()` (no English OOV drops), `consolidate_language_layers()`, and `write_lrc_layers()`.
