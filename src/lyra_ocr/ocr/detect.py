"""Text detection wrapper using RapidOCR (DBNet detector only)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .runtime import RuntimeConfig, rapidocr_params, resolve_runtime


@dataclass
class Detection:
    """Bounding box detection result without text recognition."""

    polygon: list[list[int]]  # 4 points: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    bbox: list[int]  # [xmin, ymin, xmax, ymax]
    score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def polygon_to_bbox(polygon: list[list[int | float]]) -> list[int]:
    """Convert a 4-point polygon to axis-aligned [xmin, ymin, xmax, ymax]."""
    xs = [pt[0] for pt in polygon]
    ys = [pt[1] for pt in polygon]
    return [int(round(min(xs))), int(round(min(ys))), int(round(max(xs))), int(round(max(ys)))]


def crop_quad(image: np.ndarray, polygon: list[list[int | float]]) -> np.ndarray:
    """Crop a quadrilateral from image using perspective transform."""
    pts = np.array(polygon, dtype=np.float32)
    # Calculate width and height of the bounding quad
    w1 = np.linalg.norm(pts[0] - pts[1])
    w2 = np.linalg.norm(pts[2] - pts[3])
    width = int(max(w1, w2))

    h1 = np.linalg.norm(pts[0] - pts[3])
    h2 = np.linalg.norm(pts[1] - pts[2])
    height = int(max(h1, h2))

    if width <= 0 or height <= 0:
        xmin, ymin, xmax, ymax = polygon_to_bbox(polygon)
        return image[max(0, ymin) : max(0, ymax), max(0, xmin) : max(0, xmax)]

    dst_pts = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(pts, dst_pts)
    warped = cv2.warpPerspective(image, matrix, (width, height), flags=cv2.INTER_LINEAR)
    return warped


class TextDetector:
    """Det-only wrapper using DBNet via RapidOCR."""

    def __init__(self, runtime: RuntimeConfig | None = None, **kwargs: Any) -> None:
        try:
            from rapidocr import RapidOCR
        except ImportError as exc:
            raise ImportError("rapidocr>=3.0 is required. Install via pip install rapidocr") from exc

        self._runtime = runtime or resolve_runtime()
        params: dict[str, Any] = rapidocr_params(self._runtime, use_det=True, use_rec=False)
        if "params" in kwargs:
            params.update(kwargs.pop("params"))

        self._engine = RapidOCR(params=params, **kwargs)

    def runtime_info(self) -> dict[str, Any]:
        """Effective execution providers reported by the ONNX sessions.

        Sessions are created lazily on first inference, so call this after at
        least one ``detect``; before that ``providers`` is empty.
        """
        providers: list[str] = []
        try:
            providers = list(self._engine.text_det.session.session.get_providers())
        except Exception:  # noqa: BLE001 - introspection must never fail OCR
            providers = []
        return {
            "device": self._runtime.device,
            "providers": providers,
            "rec_batch_size": self._runtime.rec_batch_size,
        }

    def detect(self, image: str | Path | np.ndarray) -> list[Detection]:
        """Detect text regions in an image frame.

        Args:
            image: Path to image file or numpy BGR array.

        Returns:
            List of Detection instances containing only polygon, bbox, and detection score.
        """
        img_input = str(image) if isinstance(image, Path) else image
        out = self._engine(img_input)

        boxes = getattr(out, "boxes", None)
        scores = getattr(out, "scores", None)

        if boxes is None or len(boxes) == 0:
            return []

        detections: list[Detection] = []
        for i, pts in enumerate(boxes):
            poly = [[int(round(pt[0])), int(round(pt[1]))] for pt in pts]
            score = round(float(scores[i]), 4) if scores and i < len(scores) else 1.0
            bbox = polygon_to_bbox(poly)
            detections.append(Detection(polygon=poly, bbox=bbox, score=score))

        return detections
