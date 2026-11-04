"""Tracking strategies registry and factory."""

from __future__ import annotations

from typing import Any

from lyra_ocr.trackers.base import BaseTracker
from lyra_ocr.trackers.greedy import GreedyTracker
from lyra_ocr.trackers.bytetrack import ByteTextTracker
from lyra_ocr.trackers.sort_tracker import KalmanSORTTracker
from lyra_ocr.trackers.st_dbscan import ST_DBSCANTracker


def get_tracker(name: str = "greedy", **kwargs: Any) -> BaseTracker:
    """Instantiate a tracker strategy by name."""
    canonical = name.strip().lower()
    if canonical in ("greedy", "baseline"):
        return GreedyTracker(**kwargs)
    elif canonical in ("bytetrack", "byte"):
        return ByteTextTracker(**kwargs)
    elif canonical in ("sort", "kalman"):
        return KalmanSORTTracker(**kwargs)
    elif canonical in ("dbscan", "st_dbscan", "tube"):
        return ST_DBSCANTracker(**kwargs)
    else:
        raise ValueError(
            f"Unknown tracker '{name}'. Supported: 'greedy', 'bytetrack', 'sort', 'dbscan'"
        )


__all__ = [
    "BaseTracker",
    "GreedyTracker",
    "ByteTextTracker",
    "KalmanSORTTracker",
    "ST_DBSCANTracker",
    "get_tracker",
]
