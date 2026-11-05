"""Tracking strategies registry and factory."""

from __future__ import annotations

from typing import Any

from lyra_ocr.trackers.base import BaseTracker
from lyra_ocr.trackers.bytetrack import ByteTextTracker
from lyra_ocr.trackers.greedy import GreedyTracker

#: Accepted ``--tracker`` spellings mapped to their implementation.
TRACKER_ALIASES: dict[str, str] = {
    "greedy": "greedy",
    "baseline": "greedy",
    "bytetrack": "bytetrack",
    "byte": "bytetrack",
}

_TRACKER_CLASSES: dict[str, type[BaseTracker]] = {
    "greedy": GreedyTracker,
    "bytetrack": ByteTextTracker,
}


def known_tracker_names() -> list[str]:
    """Canonical tracker names, for validating run options before any inference."""
    return sorted(_TRACKER_CLASSES)


def resolve_tracker_name(name: str) -> str:
    """Canonical implementation name for any accepted tracker spelling."""
    key = str(name).strip().lower()
    if key not in TRACKER_ALIASES:
        raise ValueError(
            f"Unknown tracker '{name}'. Supported: {', '.join(known_tracker_names())}"
        )
    return TRACKER_ALIASES[key]


def get_tracker(name: str = "greedy", **kwargs: Any) -> BaseTracker:
    """Instantiate a tracker strategy by name."""
    return _TRACKER_CLASSES[resolve_tracker_name(name)](**kwargs)


__all__ = [
    "BaseTracker",
    "GreedyTracker",
    "ByteTextTracker",
    "get_tracker",
    "known_tracker_names",
    "resolve_tracker_name",
]
