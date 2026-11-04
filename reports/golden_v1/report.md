# Phase 1 Evaluation & Quality Gate Report

> [!TIP]
> **PHASE 1 QUALITY GATE: PASSED**
> All operational requirements satisfied.

## 1. Summary Metrics

| Metric | Value | Threshold Target | Status |
|---|---|---|---|
| **Label Coverage** | `100.0%` (6/6) | `>= 100.0%` | **PASS** |
| **Lyric Precision** | `1.0000` | `>= 0.9000` | **PASS** |
| **Lyric Recall** | `1.0000` | `>= 0.8500` | **PASS** |
| **Lyric F1** | `1.0000` | - | INFO |
| **Distractor Leakage Rate** | `0.0%` (0 tracks) | `<= 0.0%` | **PASS** |
| **False Drop Rate** | `0.0%` (0 drops) | `<= 15.0%` | **PASS** |
| **Unknown Rate** | `16.7%` | `<= 10.0%` | **INFO** (target) |
| **Tracking Miss Rate** | `0.0%` (0 golden tracks) | `<= 15.0%` | **PASS** |
| **Macro-F1 (6-class)** | `0.6111` | - | INFO |
| **Micro-F1 (6-class / accuracy)** | `0.8333` | - | INFO |
| **Tracking Errors (Excluded)** | `0` | - | INFO |


_95% Wilson CI — lyric precision [0.342, 1.000], recall [0.342, 1.000]_

## 2. Per-Class Taxonomy Breakdown

| Category | Precision | Recall | F1-Score | Support (Ground Truth) |
|---|---|---|---|---|
| `lyric` | 1.0000 | 1.0000 | 1.0000 | 2 |
| `channel` | 1.0000 | 1.0000 | 1.0000 | 1 |
| `song_title` | 1.0000 | 0.5000 | 0.6667 | 2 |
| `supplementary` | 0.0000 | 0.0000 | 0.0000 | 0 |
| `scene_text` | 1.0000 | 1.0000 | 1.0000 | 1 |
| `unknown` | 0.0000 | 0.0000 | 0.0000 | 0 |


## 3. Confusion Matrix (Rows = Ground Truth, Columns = Predicted)

| True \ Pred | `lyric` | `channel` | `song_title` | `supplementary` | `scene_text` | `unknown` |
|---|---|---|---|---|---|---|
| **lyric** | 2 | 0 | 0 | 0 | 0 | 0 |
| **channel** | 0 | 1 | 0 | 0 | 0 | 0 |
| **song_title** | 0 | 0 | 1 | 0 | 0 | 1 |
| **supplementary** | 0 | 0 | 0 | 0 | 0 | 0 |
| **scene_text** | 0 | 0 | 0 | 0 | 1 | 0 |
| **unknown** | 0 | 0 | 0 | 0 | 0 | 0 |


## 4. Per-Video Performance

| Video ID | Tracks | Evaluated | Missing | Lyric F1 | Lyric Precision | Lyric Recall | Leakage Rate | False Drop Rate |
|---|---|---|---|---|---|---|---|---|
| `castle_en_9cac48e3` | 3 | 3 | 0 | 1.0000 | 1.0000 | 1.0000 | 0.0% | 0.0% |
| `song3_en_78305c47` | 3 | 3 | 0 | 1.0000 | 1.0000 | 1.0000 | 0.0% | 0.0% |

