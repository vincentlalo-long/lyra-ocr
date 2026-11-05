"""Tracking strategies registry and factory."""

from __future__ import annotations

from typing import Any

from lyra_ocr.trackers.base import BaseTracker
from lyra_ocr.trackers.bytetrack import ByteTextTracker
from lyra_ocr.trackers.greedy import GreedyTracker


def get_tracker(name: str = "greedy", **kwargs: Any) -> BaseTracker:
    """Instantiate a tracker strategy by name."""
    canonical = name.strip().lower()
    if canonical in ("greedy", "baseline"):
        return GreedyTracker(**kwargs)
    elif canonical in ("bytetrack", "byte"):
        return ByteTextTracker(**kwargs)
    else:
        raise ValueError(f"Unknown tracker '{name}'. Supported: 'greedy', 'bytetrack'")


__all__ = [
    "BaseTracker",
    "GreedyTracker",
    "ByteTextTracker",
    "get_tracker",
]
