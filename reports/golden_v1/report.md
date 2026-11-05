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
| **Tracking Miss Rate** | `0.0%` (0 golden tracks) | `<= 15.0%` | **PASS** |
| **Tracking Errors (Excluded)** | `0` | - | INFO |


_95% Wilson CI — lyric precision [0.342, 1.000], recall [0.342, 1.000]_

## 2. Per-Video Performance

| Video ID | Tracks | Evaluated | Missing | Lyric F1 | Lyric Precision | Lyric Recall | Leakage Rate | False Drop Rate |
|---|---|---|---|---|---|---|---|---|
| `castle_en_9cac48e3` | 3 | 3 | 0 | 1.0000 | 1.0000 | 1.0000 | 0.0% | 0.0% |
| `song3_en_78305c47` | 3 | 3 | 0 | 1.0000 | 1.0000 | 1.0000 | 0.0% | 0.0% |

