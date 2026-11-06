"""Text recognition (REC) engine with model registry and Unicode normalization."""

from __future__ import annotations

import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .detect import Detection, crop_quad
from .runtime import RuntimeConfig, rapidocr_params, resolve_runtime


@dataclass
class TextLine:
    """Recognized text line with raw/normalized text, geometry, and model provenance."""

    polygon: list[list[int]]  # 4 points: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    bbox: list[int]  # [xmin, ymin, xmax, ymax]
    text_raw: str
    text_norm: str
    confidence: float
    model_id: str
    script: str = "latin"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_text(text: str, form: str = "NFC") -> str:
    """Normalize text using Unicode normalization and strip zero-width artifacts."""
    if not text:
        return ""
    # Strip zero-width and control invisible characters
    cleaned = (
        text.replace("\u200b", "")
        .replace("\u200c", "")
        .replace("\u200d", "")
        .replace("\ufeff", "")
        .strip()
    )
    return unicodedata.normalize(form, cleaned)


def detect_script(text: str) -> str:
    """Classify the dominant script of a text string."""
    if not text or not text.strip():
        return "empty"

    has_kana = False
    has_cjk = False
    has_latin = False

    for ch in text:
        code = ord(ch)
        # Hiragana: 0x3040-0x309F, Katakana: 0x30A0-0x30FF
        if (0x3040 <= code <= 0x309F) or (0x30A0 <= code <= 0x30FF):
            has_kana = True
        # CJK Unified Ideographs: 0x4E00-0x9FFF
        elif 0x4E00 <= code <= 0x9FFF:
            has_cjk = True
        elif ch.isalpha():
            has_latin = True

    if has_kana or (has_cjk and not has_latin):
        return "japan"
    if has_cjk and has_latin:
        return "mixed"
    return "latin"


class OCREngine:
    """Text recognition engine supporting multiple REC models and script routing."""

    DEFAULT_MODEL = "PP-OCRv6_rec_small"

    def __init__(
        self,
        default_model: str = DEFAULT_MODEL,
        runtime: RuntimeConfig | None = None,
    ) -> None:
        try:
            from rapidocr import RapidOCR
        except ImportError as exc:
            raise ImportError("rapidocr>=3.0 is required. Install via pip install rapidocr") from exc

        self._default_model = default_model
        self._runtime = runtime or resolve_runtime()
        # Registry mapping model_id -> RapidOCR instance
        self._models: dict[str, Any] = {}

        # Pre-initialize default unified model
        self._models[self.DEFAULT_MODEL] = RapidOCR(params=self._rec_params())

    def _rec_params(self) -> dict[str, Any]:
        """Recognition-only RapidOCR params with batching and runtime settings."""
        return rapidocr_params(self._runtime, use_det=False, use_rec=True)

    def runtime_info(self) -> dict[str, Any]:
        """Effective execution providers reported by the recognition session."""
        providers: list[str] = []
        try:
            for engine in self._models.values():
                providers = list(engine.text_rec.session.session.get_providers())
                break
        except Exception:  # noqa: BLE001 - introspection must never fail OCR
            providers = []
        return {
            "device": self._runtime.device,
            "providers": providers,
            "rec_batch_size": self._runtime.rec_batch_size,
        }

    def _get_model(self, model_id: str) -> Any:
        if model_id in self._models:
            return self._models[model_id]

        from rapidocr import LangRec, ModelType, OCRVersion, RapidOCR

        if model_id == "latin_PP-OCRv5_rec_mobile":
            params = self._rec_params()
            params.update(
                {
                    "Rec.ocr_version": OCRVersion.PPOCRV5,
                    "Rec.lang_type": LangRec.LATIN,
                    "Rec.model_type": ModelType.MOBILE,
                }
            )
            engine = RapidOCR(params=params)
            self._models[model_id] = engine
            return engine

        # Fallback to default
        return self._models[self.DEFAULT_MODEL]

    def recognize_crop(
        self,
        crop: np.ndarray,
        model_id: str | None = None,
    ) -> tuple[str, float, str]:
        """Recognize text in a cropped image.

        Returns:
            Tuple of (text_raw, confidence, actual_model_id).
        """
        if crop is None or crop.size == 0:
            return "", 0.0, self.DEFAULT_MODEL

        chosen_model = model_id or self._default_model
        engine = self._get_model(chosen_model)

        out = engine(crop)
        txts = getattr(out, "txts", None)
        scores = getattr(out, "scores", None)

        if not txts or len(txts) == 0:
            return "", 0.0, chosen_model

        text_raw = str(txts[0])
        score = round(float(scores[0]), 4) if scores and len(scores) > 0 else 0.0
        return text_raw, score, chosen_model

    def recognize_crops(
        self,
        crops: list[np.ndarray],
        model_id: str | None = None,
    ) -> list[tuple[str, float, str]]:
        """Recognize a list of line crops in as few ONNX calls as possible.

        RapidOCR's ``recognize_txt`` sorts by width for padding efficiency but
        returns results in the original order, so callers can map results back
        to detections positionally. Empty crops keep a placeholder result and
        are never sent to the model.

        Returns:
            One ``(text_raw, confidence, actual_model_id)`` per input crop.
        """
        if not crops:
            return []

        chosen_model = model_id or self._default_model
        engine = self._get_model(chosen_model)
        results: list[tuple[str, float, str]] = [("", 0.0, chosen_model) for _ in crops]

        valid: list[tuple[int, np.ndarray]] = [
            (i, crop)
            for i, crop in enumerate(crops)
            if crop is not None and getattr(crop, "size", 0) > 0
        ]
        if not valid:
            return results

        out = engine.recognize_txt([crop for _, crop in valid])
        txts = getattr(out, "txts", None) or ()
        scores = getattr(out, "scores", None) or ()

        for pos, (original_idx, _crop) in enumerate(valid):
            text = str(txts[pos]) if pos < len(txts) else ""
            score = round(float(scores[pos]), 4) if pos < len(scores) else 0.0
            results[original_idx] = (text, score, chosen_model)

        return results

    def process_detections(
        self,
        image: np.ndarray | str | Path,
        detections: list[Detection],
        model_id: str | None = None,
    ) -> list[TextLine]:
        """Recognize text for all detected bounding quads in a frame.

        Args:
            image: Frame image as numpy array or file path.
            detections: List of Detection instances.
            model_id: Optional model to use from registry.

        Returns:
            List of TextLine instances.
        """
        if isinstance(image, (str, Path)):
            img_bgr = cv2.imread(str(image))
            if img_bgr is None:
                raise FileNotFoundError(f"Failed to read image at: {image}")
        else:
            img_bgr = image

        crops = [crop_quad(img_bgr, det.polygon) for det in detections]
        recognized = self.recognize_crops(crops, model_id=model_id)

        lines: list[TextLine] = []
        for det, (text_raw, conf, used_model) in zip(detections, recognized):
            text_norm = normalize_text(text_raw, form="NFC")
            script = detect_script(text_norm)

            lines.append(
                TextLine(
                    polygon=det.polygon,
                    bbox=det.bbox,
                    text_raw=text_raw,
                    text_norm=text_norm,
                    confidence=conf,
                    model_id=used_model,
                    script=script,
                )
            )

        return lines
