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


def _quad_dimensions(pts: np.ndarray) -> tuple[float, float]:
    w = float(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3])))
    h = float(max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2])))
    return w, h


def split_multiline_crop_quads(
    frame: np.ndarray,
    quads: list[np.ndarray],
    scores: list[float],
) -> tuple[list[np.ndarray], list[float]]:
    """Split tall 2-line merged quads along their horizontal gradient valley."""
    if frame is None or getattr(frame, "size", 0) == 0 or not quads:
        return quads, scores

    out_quads: list[np.ndarray] = []
    out_scores: list[float] = []

    for i, quad in enumerate(quads):
        score = float(scores[i]) if i < len(scores) else 1.0
        pts = np.asarray(quad, dtype=np.float32)
        if pts.shape != (4, 2):
            out_quads.append(quad)
            out_scores.append(score)
            continue

        w, h = _quad_dimensions(pts)
        if h < 52.0 or h <= 0.0 or (w / h) >= 12.0:
            out_quads.append(quad)
            out_scores.append(score)
            continue

        crop = crop_quad(frame, pts.tolist())
        if crop is None or crop.size == 0 or crop.shape[0] < 52 or crop.shape[1] < 4:
            out_quads.append(quad)
            out_scores.append(score)
            continue

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        grad_x = np.abs(np.diff(gray.astype(np.float32), axis=1))
        row_energy = grad_x.mean(axis=1)

        ch = gray.shape[0]
        cand_start = max(18, int(round(0.32 * ch)))
        cand_end = min(ch - 18, int(round(0.68 * ch))) + 1
        if cand_end <= cand_start:
            out_quads.append(quad)
            out_scores.append(score)
            continue

        y_split = cand_start + int(np.argmin(row_energy[cand_start:cand_end]))
        e_top = float(row_energy[:y_split].max()) if y_split > 0 else 0.0
        e_bot = float(row_energy[y_split + 1 :].max()) if y_split + 1 < ch else 0.0
        e_valley = float(row_energy[y_split])

        if min(e_top, e_bot) < 5.0 or e_valley >= 0.25 * max(e_top, e_bot):
            out_quads.append(quad)
            out_scores.append(score)
            continue

        r = float(y_split) / float(ch)
        left_mid = pts[0] + r * (pts[3] - pts[0])
        right_mid = pts[1] + r * (pts[2] - pts[1])

        top_quad = np.stack([pts[0], pts[1], right_mid, left_mid], axis=0)
        bot_quad = np.stack([left_mid, right_mid, pts[2], pts[3]], axis=0)

        orig_dtype = np.asarray(quad).dtype
        if np.issubdtype(orig_dtype, np.integer):
            top_quad = np.round(top_quad).astype(orig_dtype)
            bot_quad = np.round(bot_quad).astype(orig_dtype)
        else:
            top_quad = top_quad.astype(orig_dtype)
            bot_quad = bot_quad.astype(orig_dtype)

        out_quads.extend([top_quad, bot_quad])
        out_scores.extend([score, score])

    return out_quads, out_scores


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

        quads = [np.asarray(pts, dtype=np.float32) for pts in boxes]
        score_list = [
            round(float(scores[i]), 4) if scores and i < len(scores) else 1.0
            for i in range(len(quads))
        ]

        has_tall = any(_quad_dimensions(q)[1] >= 52.0 for q in quads if q.shape == (4, 2))
        if has_tall:
            frame_bgr = image if isinstance(image, np.ndarray) else cv2.imread(str(image))
            if frame_bgr is not None:
                quads, score_list = split_multiline_crop_quads(frame_bgr, quads, score_list)

        detections: list[Detection] = []
        for i, pts in enumerate(quads):
            poly = [[int(round(pt[0])), int(round(pt[1]))] for pt in pts]
            score = score_list[i]
            bbox = polygon_to_bbox(poly)
            detections.append(Detection(polygon=poly, bbox=bbox, score=score))

        return detections
