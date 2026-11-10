# Phase 2 Review Fixes & Reliability Hardening Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix critical evaluation circularity (GT freezing), multi-segment track cue splitting, config-aware artifact reuse, visual timing metric semantics, chorus title-matching rescue, tracker same-frame separation, and vocabulary safety.

**Architecture:** 
1. Decouple and freeze Ground Truth generation in `auto_ground_truth.py` and `compare_trackers.py` so both trackers evaluate on immutable canonical ground truth.
2. Refactor cue selection in `select.py` to generate `LyricCue` per `TextSegment` rather than per entire track, properly setting `source_segment_ids`.
3. Validate runtime configurations (`config_hash`) in `build_lrc_from_video` before reusing artifacts.
4. Update `visual_timing.py` to report `refinement_delta_ms` and eliminate hardcoded oracle confidence.
5. Fix classification & recovery of chorus lines matching song titles in reading zones.
6. Guard tracker retirement/replacement against same-frame re-matching in `greedy.py` and `bytetrack.py`.
7. Restrict vocabulary derivation in `correct.py` to accepted lyric tracks and prevent ad-lib clobbering in `track.py`.

**Tech Stack:** Python 3.11, pytest, numpy, scipy, rapidocr, opencv-python

**Spec:** `doc/PHASE2_LRC_PLAN.md` & User Review Findings

## Global Constraints
- Python >= 3.11
- Zero external network calls during unit tests
- Preserve all existing Phase 1 JSON artifact schemas (`manifest.json`, `tracks.json`, `frame_index.json`)
- Maintain 100% passing test suite across `tests/unit` and `tests/eval`

## Review Focus
1. Multi-segment tracks with 2 distinct lines must produce 2 distinct `LyricCue`s at their respective start/end times.
2. `compare_trackers.py` must never overwrite existing canonical `ground_truth/` files.
3. If artifacts exist with different FPS or tracker, `build_lrc_from_video` must re-run the pipeline unless explicitly forced otherwise.
4. Chorus lines containing title tokens in reading zones must not be permanently dropped from LRC output.
5. Mixed track IDs and segment IDs must remain strictly comparable without `TypeError`.

---

### Task 1: Fix Ground Truth Freezing in Evaluation (`scripts/auto_ground_truth.py` & `scripts/compare_trackers.py`)

**Files:**
- Modify: `scripts/auto_ground_truth.py:315-330`
- Modify: `scripts/compare_trackers.py:120-140`
- Test: `tests/unit/test_gt_freezing.py`

**Interfaces:**
- Consumes: `generate_silver_ground_truth(dataset, video_id, run_id, ...)`
- Produces: Canonical frozen ground truth segments under `datasets/<ds>/ground_truth/<video_id>.json` that `compare_trackers.py` reads without overwriting.

- [ ] **Step 1: Write failing test verifying GT is preserved and not overwritten**
- [ ] **Step 2: Run test to verify it fails**
- [ ] **Step 3: Update `compare_trackers.py` to generate canonical GT once if absent, and never overwrite on subsequent tracker runs**
- [ ] **Step 4: Run test to verify it passes**
- [ ] **Step 5: Commit**

---

### Task 2: Segment-Level Cue Extraction & Traceability (`src/lyra_ocr/lyric/select.py`)

**Files:**
- Modify: `src/lyra_ocr/lyric/select.py:50-145`
- Test: `tests/unit/test_lrc_select.py`

**Interfaces:**
- Consumes: `tracks.json` with `track["segments"]`
- Produces: `LyricCue` per `TextSegment` with exact `source_segment_ids`, segment `start_ms`/`end_ms`, and segment-specific text.

- [ ] **Step 1: Write failing test in `test_lrc_select.py` where a track has 2 distinct segments**
- [ ] **Step 2: Run test to verify it fails (expected: currently returns 1 cue instead of 2)**
- [ ] **Step 3: Refactor `select_cues` in `select.py` to iterate over segments if present, or fallback to track**
- [ ] **Step 4: Run test to verify it passes**
- [ ] **Step 5: Commit**

---

### Task 3: Config-Aware Artifact Reuse (`src/lyra_ocr/lyric/build.py`)

**Files:**
- Modify: `src/lyra_ocr/lyric/build.py:110-150`
- Test: `tests/unit/test_lrc_build.py`

**Interfaces:**
- Consumes: `manifest.json` (`tracker`, `config.sample_fps`, `config_hash`)
- Produces: Detection of stale artifacts when requested `fps`, `tracker`, or `start_sec` mismatch existing manifest.

- [ ] **Step 1: Write failing test in `test_lrc_build.py` verifying that changing FPS re-runs pipeline**
- [ ] **Step 2: Run test to verify it fails**
- [ ] **Step 3: Implement config compatibility check in `build_lrc_from_video`**
- [ ] **Step 4: Run test to verify it passes**
- [ ] **Step 5: Commit**

---

### Task 4: Rescue Mid-Video Chorus Lines Matching Title in Reading Zone (`src/lyra_ocr/classify.py` & `select.py`)

**Files:**
- Modify: `src/lyra_ocr/classify.py:220-235`
- Modify: `src/lyra_ocr/lyric/select.py:75-95`
- Test: `tests/unit/test_classify.py`, `tests/unit/test_lrc_select.py`

**Interfaces:**
- Consumes: `track["features"]`, `track["classification"]`
- Produces: Classification of chorus lines in reading zone as `lyric` (with `decision="review"` or `"accept"`), rescuable in `select.py`.

- [ ] **Step 1: Write failing test for chorus line containing song title mid-video in reading zone**
- [ ] **Step 2: Run test to verify it fails**
- [ ] **Step 3: Update `classify.py` and `select.py` to mark/rescue reading-zone title matches**
- [ ] **Step 4: Run test to verify it passes**
- [ ] **Step 5: Commit**

---

### Task 5: Prevent Tracker Same-Frame Re-matching & Protect Short Ad-libs (`src/lyra_ocr/track.py`, `greedy.py`, `bytetrack.py`)

**Files:**
- Modify: `src/lyra_ocr/trackers/greedy.py:60-75`
- Modify: `src/lyra_ocr/trackers/bytetrack.py:90-110`
- Modify: `src/lyra_ocr/track.py:335-350`
- Test: `tests/unit/test_trackers.py`, `tests/unit/test_track.py`

**Interfaces:**
- Consumes: Frame detections and active track pools
- Produces: Safe track retirement without immediate re-association in the same frame; safe fragment merging that preserves distinct short words ("Oh" vs "Yeah").

- [ ] **Step 1: Write failing tests for same-frame re-matching and short ad-lib preservation**
- [ ] **Step 2: Run tests to verify failure**
- [ ] **Step 3: Update `greedy.py`, `bytetrack.py`, and `track.py`**
- [ ] **Step 4: Run tests to verify they pass**
- [ ] **Step 5: Commit**

---

### Task 6: Safe Vocabulary Derivation & Visual Timing Semantics (`src/lyra_ocr/lyric/correct.py`, `visual_timing.py`, `scripts/eval_lrc.py`, `lrc_eval.py`)

**Files:**
- Modify: `src/lyra_ocr/lyric/correct.py:45-70`
- Modify: `src/lyra_ocr/eval/visual_timing.py:15-80`
- Modify: `src/lyra_ocr/eval/lrc_eval.py:55-75`
- Modify: `scripts/eval_lrc.py:100-140`
- Test: `tests/eval/test_visual_timing.py`, `tests/unit/test_lrc_correct.py`, `tests/eval/test_lrc_eval.py`

**Interfaces:**
- Consumes: Accepted lyric cues for vocabulary; visual timing comparison with clear naming (`refinement_delta_ms`).
- Produces: Correct vocabulary without watermark leakage; honest metric reporting without hardcoded confidence.

- [ ] **Step 1: Write failing tests for lyric-only vocabulary derivation, empty reference scorable=False, and visual timing refinement naming**
- [ ] **Step 2: Run tests to verify failure**
- [ ] **Step 3: Implement fixes in `correct.py`, `visual_timing.py`, `lrc_eval.py`, and `eval_lrc.py`**
- [ ] **Step 4: Run tests to verify they pass**
- [ ] **Step 5: Commit**

---

### Task 7: Minor Cleanups & Full Regression Verification

**Files:**
- Modify: `src/lyra_ocr/cli.py` (optional video argument if `--artifacts` given)
- Modify: `scripts/compare_trackers.py` & docs (accurate ByteTrack description without claiming Kalman)
- Modify: `src/lyra_ocr/lyric/select.py` (frame_width parameterization)
- Modify: `src/lyra_ocr/eval/reference.py` (strip section headers)
- Modify: `scripts/export_lrc.py` (pass `--force`)

- [ ] **Step 1: Apply minor cleanups**
- [ ] **Step 2: Run full project test suite (`pytest tests/unit tests/eval`)**
- [ ] **Step 3: Run manual QA on real dataset videos**
- [ ] **Step 4: Commit**
