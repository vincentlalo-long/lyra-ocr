"""Near-duplicate frame detection for skipping redundant OCR work.

Lyric videos often hold a frame almost unchanged for many samples. Comparing a
small grayscale thumbnail lets the pipeline reuse the previous OCR result and
avoid writing a second identical JPEG. Thresholds are deliberately conservative:
``mean_abs`` catches whole-frame stasis, ``max_abs`` catches a local change (one
glyph flipping) that a mean over 128x72 would dilute.
"""

from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np

from lyra_ocr.ocr.engine import TextLine

DEDUP_SIGNATURE_SIZE = (128, 72)  # (width, height)
DEDUP_MEAN_ABS_MAX = 0.004  # normalised mean absolute difference (0..1)
DEDUP_MAX_ABS_MAX = 8  # maximum per-pixel difference (0..255)


def frame_signature(image_bgr: np.ndarray) -> np.ndarray:
    """Small grayscale thumbnail used as the frame's dedup signature."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, DEDUP_SIGNATURE_SIZE, interpolation=cv2.INTER_AREA)


def is_duplicate(prev_signature: np.ndarray, signature: np.ndarray) -> bool:
    """True when two signatures are close enough to reuse OCR output."""
    if prev_signature is None or signature is None:
        return False
    if prev_signature.shape != signature.shape:
        return False

    diff = cv2.absdiff(prev_signature, signature)
    mean_abs = float(diff.mean()) / 255.0
    max_abs = int(diff.max())
    return mean_abs <= DEDUP_MEAN_ABS_MAX and max_abs <= DEDUP_MAX_ABS_MAX


def clone_text_line(line: TextLine) -> TextLine:
    """Copy a TextLine deeply enough that reused frames stay independent."""
    return replace(
        line,
        polygon=[list(pt) for pt in line.polygon],
        bbox=list(line.bbox),
    )
