from __future__ import annotations

from typing import Any, Protocol


class AudioTimingProvider(Protocol):
    def align(self, video_path: str, reference_lyrics: list[str]) -> list[tuple[int, str]]:
        ...


def get_audio_provider() -> AudioTimingProvider | None:
    """Return an eval-only forced-alignment provider if installed, else None."""
    return None


def get_audio_timing_report() -> dict[str, Any]:
    return {
        "status": "not_measured",
        "reason": "no eval-only forced-alignment provider configured in v0",
    }
