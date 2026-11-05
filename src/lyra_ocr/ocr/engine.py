"""Text recognition (REC) engine with model registry and Unicode normalization."""

from __future__ import annotations

import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .detect import Detection, crop_quad


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

    def __init__(self, default_model: str = DEFAULT_MODEL) -> None:
        try:
            from rapidocr import RapidOCR
        except ImportError as exc:
            raise ImportError("rapidocr>=3.0 is required. Install via pip install rapidocr") from exc

        self._default_model = default_model
        # Registry mapping model_id -> RapidOCR instance
        self._models: dict[str, Any] = {}

        # Pre-initialize default unified model
        self._models[self.DEFAULT_MODEL] = RapidOCR(
            params={
                "Global.use_det": False,
                "Global.use_cls": False,
                "Global.use_rec": True,
            }
        )

    def _get_model(self, model_id: str) -> Any:
        if model_id in self._models:
            return self._models[model_id]

        from rapidocr import LangRec, ModelType, OCRVersion, RapidOCR

        if model_id == "latin_PP-OCRv5_rec_mobile":
            engine = RapidOCR(
                params={
                    "Global.use_det": False,
                    "Global.use_cls": False,
                    "Global.use_rec": True,
                    "Rec.ocr_version": OCRVersion.PPOCRV5,
                    "Rec.lang_type": LangRec.LATIN,
                    "Rec.model_type": ModelType.MOBILE,
                }
            )
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

        lines: list[TextLine] = []
        for det in detections:
            crop = crop_quad(img_bgr, det.polygon)
            text_raw, conf, used_model = self.recognize_crop(crop, model_id=model_id)

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
