# Phase 2 Remaining Review Fixes (P0, P1, P2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decisively fix all remaining review issues across Ground Truth generation, silver label pollution, LRC cache invalidation, segment review round-trip, `--include-review` leak prevention, zero-prediction eval penalty, timing metrics clarification, registry/jobs race conditions, probe rotation dimensions, YouTube URL handling, web review validation, vocabulary filtering, alphanumeric protection, and test gaps.

**Architecture:**
- Separate Silver GT / Silver Labels from Canonical Human Ground Truth and LRC export/evaluation.
- Enforce option-aware and artifact-mtime-aware LRC caching with auto-invalidation upon pipeline rerun.
- Extend review export and label ingestion to support segment-level IDs (`track_id.segment_id`) with full round-trip fidelity.
- Prevent non-lyric leaking into LRC under `--include-review` and ensure human `reject` always overrides machine chorus rescue.
- Ensure Hungarian track matching penalizes trackers for videos with 0 predictions by evaluating against expected video IDs.
- Fix ffmpeg probe rotation dimension swap, registry locks, job queue cancellation races, and YouTube URL string IDs.

**Tech Stack:** Python 3.11+, pytest, NumPy, SciPy (linear_sum_assignment), FFprobe, SQLite/JSONL/CSV.

---

## Global Constraints
- Preserve Phase 1 JSON artifact schemas (`manifest.json`, `tracks.json`, `frame_index.json`).
- Zero regressions: all 209+ existing unit/eval tests must continue to pass.
- Iron Law of Verification: no completion claims without fresh verification command evidence.
- Every fix must be verified by automated unit or evaluation tests.

---

## Task Decomposition

### Task 1: Silver Label Isolation & Golden Ground Truth Incomplete-Line Handling
- Modify `scripts/auto_ground_truth.py`:
  - Write silver labels to `silver_labels.jsonl` instead of `labels.jsonl` (or tag with `annotator="silver_lyrics_alignment"`).
  - When aligning tracks with reference lyrics, identify reference lyric lines that were NOT matched by any track and record them in the ground truth as unobserved/missed segments (`source_track_id=None`, `notes="unmatched_reference_line"`).
- Modify `src/lyra_ocr/dataset/ground_truth.py:promote_run()`:
  - Skip records with `annotator == "silver_lyrics_alignment"`.
- Modify `src/lyra_ocr/lyric/select.py:load_artifacts()`:
  - Ignore records in `labels.jsonl` where `annotator == "silver_lyrics_alignment"`.
- Tests: `tests/unit/test_gt_silver_isolation.py`.

### Task 2: Option-Aware LRC Cache & Automatic Invalidation on Pipeline Rerun
- Modify `src/lyra_ocr/lyric/build.py`:
  - In `build_lrc_from_artifacts()`: check cache signature in `cues.json` (options: `include_review`, `correct_method`, `offset_ms`, `refine`, `title`, `artist`; and mtime/hash of `tracks.json` and `labels.jsonl`). If options or artifacts changed, invalidate cache and recompute.
  - In `build_lrc_from_video()`: when `needs_pipeline` is True, delete stale `.lrc` and `.cues.json` and pass `force=True` to `build_lrc_from_artifacts()`. Also verify `manifest.get("video_path")` matches input `video_path`.
- Tests: `tests/unit/test_lrc_build.py:test_cache_invalidation_after_pipeline_rerun`.

### Task 3: Segment-Level Review CSV Round-Trip
- Modify `src/lyra_ocr/labels.py:ingest_label_batch_csv()`:
  - Parse `track_id` with segment notation (e.g. `"42.1"`): resolve base track ID `"42"` against `tracks_map`. Accept row if base track exists.
- Modify `src/lyra_ocr/lyric/select.py:select_cues()`:
  - Look up human label for segment: `h = human_labels.get(f"{tid}.{seg_id}") or human_labels.get(tid)`. Apply `corrected_text` specifically to the matching segment.
- Tests: `tests/unit/test_review_labels.py:test_segment_review_csv_roundtrip`.

### Task 4: Fix `--include-review` Non-Lyric Leakage and Human Reject Precedence
- Modify `src/lyra_ocr/lyric/select.py:select_cues()`:
  - Enforce: if `decision == "reject"`, track is NEVER accepted (human rejection always wins).
  - Enforce: `label` must be `"lyric"`. Never rescue a track with `label == "non_lyric"`.
- Remove duplicate return statement at lines 179-181 of `select.py`.
- Tests: `tests/unit/test_lrc_select.py:test_include_review_does_not_leak_non_lyric`.

### Task 5: Zero-Prediction Evaluation Penalty & Visual Timing Display Polish
- Modify `src/lyra_ocr/eval/matching.py:match_predictions_to_golden()`:
  - Accept `expected_video_ids: set[str] | None = None`. If provided, include all golden items for those videos.
  - If `predicted_tracks` is empty or missing a video, generate `is_missing=True` items for all golden tracks of those videos, penalizing recall.
- Modify `src/lyra_ocr/eval/dataset_eval.py`:
  - Pass `expected_video_ids` to `match_predictions_to_golden()`.
- Modify `src/lyra_ocr/eval/visual_timing.py`:
  - Require `n_measured_cues >= 3` for `status="ok"`, else `"insufficient_cues"`.
- Modify `scripts/eval_lrc.py`:
  - Calculate visual timing statistics across ALL results (not just `scorable_results`).
  - Print "Visual Refine Delta (ms)" instead of "Visual Med Error".
- Tests: `tests/eval/test_tracker_matching.py:test_matching_penalizes_zero_pred_video`.

### Task 6: Probe Video Rotation Swap, YouTube URL ID Extraction, Web Review Validation, & Concurrency Locks
- Modify `src/lyra_ocr/video/probe.py:probe_video()`:
  - Swap `width, height = height, width` when `rotation in (90, 270)`.
- Modify `src/lyra_ocr/dataset/youtube.py:flat_entries_from_url()`:
  - Extract 11-char YouTube ID from URL or skip if invalid.
- Modify `src/lyra_ocr/dataset/registry.py`:
  - Wrap `set_split()` and `freeze_dataset()` in `with MANIFEST_LOCK:`.
- Modify `src/lyra_ocr/web/jobs.py`:
  - Do not pop running jobs from `self._jobs`; initialize cancel events in `_load_all()`.
- Modify `src/lyra_ocr/web/review_api.py`:
  - Validate `track_id` exists in `tracks.json` (or base track if segment notation).
- Tests: `tests/unit/test_probe.py`, `tests/unit/test_youtube.py`, `tests/unit/test_web_review_api.py`.

### Task 7: Vocabulary Filtering, Alphanumeric Protection, & Documentation Cleanup
- Modify `src/lyra_ocr/lyric/correct.py`:
  - `build_corpus_vocab()`: accept `human_labels` and filter out human-rejected tracks.
  - `_replace_token()`: protect ordinals (`1st`, `2nd`, `3rd`), years, and short alphanumerics (`2u`, `3d`) from erroneous correction, while preserving leetspeak replacement (`EXPL0DE` -> `EXPLODE`).
- Modify `README.md`:
  - Add caveat note on Silver GT benchmarks.
  - Correct title-card intro threshold description to $\le 25\%$.
  - Remove `scikit-learn` from requirements list.
- Run full test suite to verify 0 regressions.
