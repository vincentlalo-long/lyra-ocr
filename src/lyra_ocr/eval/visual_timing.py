from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from lyra_ocr.lyric.models import LyricCue

logger = logging.getLogger(__name__)


def measure_visual_timing(
    video_path: str,
    cues: list[LyricCue],
    artifacts_dir: str,
    *,
    fps: float = 25.0,
    window_ms: int = 1000,
) -> dict[str, Any]:
    v_path = Path(video_path)
    art_path = Path(artifacts_dir)

    if not cues or not v_path.is_file() or not art_path.is_dir():
        return {
            "status": "not_measured",
            "refinement_delta_ms": None,
            "median_abs_error_ms": None,
            "p90_refinement_delta_ms": None,
            "p90_abs_error_ms": None,
            "percent_within_100ms": 0.0,
            "percent_within_300ms": 0.0,
            "oracle_confidence_mean": None,
            "n_measured_cues": 0,
        }

    try:
        from lyra_ocr.lyric.timing import refine_timings

        cues_copy = [copy.deepcopy(c) for c in cues]
        manifest_path = art_path / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}

        refined, refined_indices = refine_timings(
            cues_copy,
            video_path,
            manifest,
            artifacts_dir,
            fps=fps,
            window_ms=window_ms,
            enabled=True,
            return_meta=True,
        )

        errors_ms: list[float] = []
        for idx in refined_indices:
            orig = cues[idx]
            ref = refined[idx]
            err = abs(orig.start_ms - ref.start_ms)
            errors_ms.append(float(err))

        if not errors_ms:
            return {
                "status": "unmatched_visual",
                "metric_type": "refinement_delta",
                "refinement_delta_ms": None,
                "median_abs_error_ms": None,
                "p90_refinement_delta_ms": None,
                "p90_abs_error_ms": None,
                "percent_within_100ms": 0.0,
                "percent_within_300ms": 0.0,
                "oracle_confidence_mean": None,
                "n_measured_cues": 0,
            }

        arr = np.array(errors_ms)
        median_err = float(np.median(arr))
        p90_err = float(np.percentile(arr, 90))
        p100 = float(np.mean(arr <= 100.0))
        p300 = float(np.mean(arr <= 300.0))
        cue_conf_mean = float(np.mean([cues[i].confidence for i in refined_indices])) if refined_indices else None

        if len(errors_ms) < 3:
            return {
                "status": "insufficient_cues",
                "metric_type": "refinement_delta",
                "refinement_delta_ms": round(median_err, 1),
                "median_abs_error_ms": round(median_err, 1),
                "p90_refinement_delta_ms": None,
                "p90_abs_error_ms": None,
                "percent_within_100ms": round(p100, 3),
                "percent_within_300ms": round(p300, 3),
                "oracle_confidence_mean": round(cue_conf_mean, 4) if cue_conf_mean is not None else None,
                "n_measured_cues": len(errors_ms),
            }

        return {
            "status": "ok",
            "metric_type": "refinement_delta",
            "refinement_delta_ms": round(median_err, 1),
            "median_abs_error_ms": round(median_err, 1),
            "p90_refinement_delta_ms": round(p90_err, 1),
            "p90_abs_error_ms": round(p90_err, 1),
            "percent_within_100ms": round(p100, 3),
            "percent_within_300ms": round(p300, 3),
            "oracle_confidence_mean": round(cue_conf_mean, 4) if cue_conf_mean is not None else None,
            "n_measured_cues": len(errors_ms),
        }
    except Exception as e:
        logger.warning("measure_visual_timing error: %s", e)
        return {
            "status": "error",
            "error": str(e),
            "refinement_delta_ms": None,
            "median_abs_error_ms": None,
            "p90_refinement_delta_ms": None,
            "p90_abs_error_ms": None,
            "percent_within_100ms": 0.0,
            "percent_within_300ms": 0.0,
            "oracle_confidence_mean": None,
            "n_measured_cues": 0,
        }
