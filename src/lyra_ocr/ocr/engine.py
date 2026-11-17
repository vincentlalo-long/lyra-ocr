"""Text recognition (REC) engine with model registry and Unicode normalization."""

from __future__ import annotations

import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from lyra_ocr.config import DEFAULT_OCR_PREPROCESS_THRESHOLDS, OcrPreprocessThresholds

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


def preprocess_crop_for_recognition(
    crop: np.ndarray | None,
    *,
    thresholds: OcrPreprocessThresholds | None = None,
) -> np.ndarray | None:
    """Upscale tiny text crops, enhance local contrast via CLAHE, and add border padding."""
    if crop is None or getattr(crop, "size", 0) == 0:
        return crop

    cfg = thresholds or DEFAULT_OCR_PREPROCESS_THRESHOLDS
    img = crop
    h, w = img.shape[:2]
    if h < cfg.upscale_trigger_height_px:
        scale = max(cfg.upscale_min_scale, cfg.upscale_target_height_px / max(1, h))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # Apply subtle CLAHE only on images with non-trivial texture/contrast to avoid
    # amplifying flat-background noise or distorting uniform synthetic crops.
    if img.ndim == 3 and img.shape[2] == 3 and float(img.std()) >= cfg.clahe_min_std:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip_limit, tileGridSize=cfg.clahe_tile_grid_size)
        l_enhanced = clahe.apply(l_chan)
        img = cv2.cvtColor(cv2.merge((l_enhanced, a_chan, b_chan)), cv2.COLOR_LAB2BGR)
    elif img.ndim == 2 and float(img.std()) >= cfg.clahe_min_std:
        clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip_limit, tileGridSize=cfg.clahe_tile_grid_size)
        img = clahe.apply(img)

    pad = cfg.border_pad_px
    return cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)


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


def _compute_border_fg_distance(
    img: np.ndarray,
    *,
    thresholds: OcrPreprocessThresholds | None = None,
) -> np.ndarray | None:
    """Compute per-pixel max-channel color distance from the median border background."""
    if img.ndim not in (2, 3):
        return None
    h, w = img.shape[:2]
    if h < 1 or w < 1:
        return None

    cfg = thresholds or DEFAULT_OCR_PREPROCESS_THRESHOLDS
    by = max(1, h // cfg.border_y_fraction_div)
    bx = max(1, w // cfg.border_x_fraction_div)
    border_mask = np.zeros((h, w), dtype=bool)
    border_mask[:by, :] = True
    border_mask[-by:, :] = True
    border_mask[:, :bx] = True
    border_mask[:, -bx:] = True

    border_pixels = img[border_mask]
    bg = np.median(border_pixels, axis=0).astype(np.float32)
    diff = np.abs(img.astype(np.float32) - bg)
    if img.ndim == 3:
        return np.max(diff, axis=2)
    return diff


def to_soft_bw_stroke_crop(
    crop: np.ndarray | None,
    *,
    thresholds: OcrPreprocessThresholds | None = None,
) -> np.ndarray | None:
    """Normalize a text crop into a color-invariant, anti-aliased B&W stroke image."""
    if crop is None or getattr(crop, "size", 0) == 0:
        return crop

    cfg = thresholds or DEFAULT_OCR_PREPROCESS_THRESHOLDS
    img = crop
    h, w = img.shape[:2]
    min_h = cfg.soft_bw_min_height_px
    if h < min_h:
        scale = float(min_h) / max(1, h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(min_h, int(round(h * scale)))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    fg_dist = _compute_border_fg_distance(img, thresholds=cfg)
    if fg_dist is None:
        return crop

    p_low = float(np.percentile(fg_dist, cfg.soft_bw_low_percentile))
    p_high = float(np.percentile(fg_dist, cfg.soft_bw_high_percentile))
    hi = p_high if p_high > p_low else float(np.max(fg_dist))
    if hi <= p_low:
        norm = np.zeros(fg_dist.shape, dtype=np.uint8)
    else:
        norm = np.clip((fg_dist - p_low) * (255.0 / (hi - p_low)), 0.0, 255.0).astype(np.uint8)

    bgr = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)
    pad = cfg.border_pad_px
    return cv2.copyMakeBorder(bgr, pad, pad, pad, pad, cv2.BORDER_REPLICATE)


def reconstruct_visual_word_spaces(
    crop: np.ndarray,
    text: str,
    *,
    thresholds: OcrPreprocessThresholds | None = None,
) -> str:
    """Reconstruct missing word spaces in Latin/Romaji text from color-invariant ink valleys."""
    if crop is None or getattr(crop, "size", 0) == 0:
        return text
    if not text or not text.strip():
        return text
    if detect_script(text) != "latin":
        return text

    cfg = thresholds or DEFAULT_OCR_PREPROCESS_THRESHOLDS
    stripped = text.strip()
    if " " in stripped:
        if max(len(tok) for tok in stripped.split()) <= cfg.spaced_max_token_len_guard:
            return text
    elif len(stripped) < cfg.unspaced_min_len_guard:
        return text

    if crop.ndim not in (2, 3):
        return text
    h, w = crop.shape[:2]
    if h < cfg.whitespace_min_crop_height_px or w < cfg.whitespace_min_crop_width_px:
        return text

    fg_dist = _compute_border_fg_distance(crop, thresholds=cfg)
    if fg_dist is None:
        return text

    fg_max = float(np.max(fg_dist))
    if fg_max < cfg.whitespace_min_fg_max:
        return text

    fg_ref = max(
        float(np.percentile(fg_dist, cfg.whitespace_fg_ref_percentile)),
        cfg.whitespace_fg_ref_max_ratio * fg_max,
    )
    ink_thresh = max(cfg.whitespace_ink_thresh_min, cfg.whitespace_ink_thresh_ratio * fg_ref)
    ink = (fg_dist >= ink_thresh).astype(np.float32)

    y_coords, x_coords = np.where(ink > 0)
    if len(x_coords) == 0:
        return text

    x_left = int(np.min(x_coords))
    x_right = int(np.max(x_coords))
    y_top = int(np.min(y_coords))
    y_bottom = int(np.max(y_coords))
    h_ink = int(y_bottom - y_top + 1)
    span = x_right - x_left
    if span <= 0 or h_ink <= 2:
        return text

    # Compute column-wise vertical ink density D(x) = (1/H) * sum_y I[ink(y, x)]
    d_x = np.sum(ink, axis=0) / float(h)

    # Smooth D(x) with a box filter
    k = max(1, cfg.whitespace_smooth_kernel_px)
    kernel = np.ones(k, dtype=np.float32) / float(k)
    d_smooth = np.convolve(d_x, kernel, mode="same")

    min_gap = max(cfg.whitespace_min_gap_px, int(cfg.whitespace_min_gap_height_ratio * h_ink))
    valleys: list[tuple[float, float]] = []
    in_valley = False
    start_x = 0

    for x in range(x_left, x_right + 1):
        if d_smooth[x] <= cfg.whitespace_valley_density_max:
            if not in_valley:
                in_valley = True
                start_x = x
        else:
            if in_valley:
                in_valley = False
                end_x = x - 1
                gap_width = end_x - start_x + 1
                if gap_width >= min_gap:
                    x_mid = (start_x + end_x) / 2.0
                    r_k = (x_mid - x_left) / float(span)
                    if cfg.whitespace_valley_rel_pos_min < r_k < cfg.whitespace_valley_rel_pos_max:
                        valleys.append((r_k, x_mid))

    if in_valley:
        end_x = x_right
        gap_width = end_x - start_x + 1
        if gap_width >= min_gap:
            x_mid = (start_x + end_x) / 2.0
            r_k = (x_mid - x_left) / float(span)
            if cfg.whitespace_valley_rel_pos_min < r_k < cfg.whitespace_valley_rel_pos_max:
                valleys.append((r_k, x_mid))

    if not valleys:
        return text

    no_space_text = text.replace(" ", "")
    total_len = len(no_space_text)
    if total_len < 2:
        return text

    existing_splits: list[int] = []
    char_count = 0
    for ch in text:
        if ch == " ":
            if char_count not in existing_splits and 0 < char_count < total_len:
                existing_splits.append(char_count)
        else:
            char_count += 1

    # Only reconstruct if text has fewer spaces than detected visual valleys
    if len(existing_splits) >= len(valleys):
        return text

    density_max = cfg.whitespace_valley_density_max
    total_ink_cols = max(1, int(np.count_nonzero(d_smooth[x_left : x_right + 1] > density_max)))

    new_splits = list(existing_splits)
    for r_k, x_mid in valleys:
        # Map valley to character index using cumulative ink columns before valley
        ink_cols_before = int(np.count_nonzero(d_smooth[x_left : int(round(x_mid)) + 1] > density_max))
        r_ink = ink_cols_before / float(total_ink_cols) if total_ink_cols > 0 else r_k
        cand_i = int(round(r_ink * total_len))
        if cand_i < 1 or cand_i > total_len - 1:
            continue
        if any(abs(cand_i - s) < cfg.whitespace_min_split_char_gap for s in new_splits):
            continue
        new_splits.append(cand_i)

    new_splits.sort()
    if len(new_splits) == len(existing_splits):
        return text

    splits = [0] + sorted(set(new_splits)) + [total_len]
    words = [no_space_text[splits[idx] : splits[idx + 1]] for idx in range(len(splits) - 1)]
    return " ".join(words)


class OCREngine:
    """Text recognition engine supporting multiple REC models and script routing."""

    DEFAULT_MODEL = "PP-OCRv6_rec_small"

    def __init__(
        self,
        default_model: str = DEFAULT_MODEL,
        runtime: RuntimeConfig | None = None,
        *,
        preprocess_thresholds: OcrPreprocessThresholds | None = None,
    ) -> None:
        try:
            from rapidocr import RapidOCR
        except ImportError as exc:
            raise ImportError("rapidocr>=3.0 is required. Install via pip install rapidocr") from exc

        self._default_model = default_model
        self._runtime = runtime or resolve_runtime()
        self._preprocess_thresholds = preprocess_thresholds or DEFAULT_OCR_PREPROCESS_THRESHOLDS
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
            try:
                engine = RapidOCR(params=params)
                self._models[model_id] = engine
                return engine
            except Exception:
                fallback = self._models[self.DEFAULT_MODEL]
                self._models[model_id] = fallback
                return fallback

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

        proc_crop = preprocess_crop_for_recognition(crop, thresholds=self._preprocess_thresholds)
        out = engine(proc_crop)
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
            (i, preprocess_crop_for_recognition(crop, thresholds=self._preprocess_thresholds))
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

        cfg = self._preprocess_thresholds
        lines: list[TextLine] = []
        for det, crop, (text_raw, conf, used_model) in zip(detections, crops, recognized):
            text_norm = normalize_text(text_raw, form="NFC")
            script = detect_script(text_norm)

            if script == "latin":
                needs_alt = (
                    len(text_norm) >= cfg.latin_alt_min_unspaced_len and " " not in text_norm
                ) or (conf < cfg.fallback_min_confidence)
                if needs_alt and used_model != "latin_PP-OCRv5_rec_mobile":
                    try:
                        alt_raw, alt_conf, alt_model = self.recognize_crop(
                            crop,
                            model_id="latin_PP-OCRv5_rec_mobile",
                        )
                        if alt_raw and alt_model == "latin_PP-OCRv5_rec_mobile":
                            alt_norm = normalize_text(alt_raw, form="NFC")
                            if (" " not in text_norm and " " in alt_norm) or (alt_conf > conf):
                                text_raw = alt_raw
                                text_norm = alt_norm
                                conf = alt_conf
                                used_model = alt_model
                                script = detect_script(text_norm)
                    except Exception:
                        pass

            if conf < cfg.fallback_min_confidence and crop is not None and getattr(crop, "size", 0) > 0:
                try:
                    bw_crop = to_soft_bw_stroke_crop(crop, thresholds=cfg)
                    bw_raw, bw_conf, bw_model = self.recognize_crop(
                        bw_crop,
                        model_id=used_model or model_id,
                    )
                    if bw_raw and bw_conf > conf:
                        text_raw = bw_raw
                        text_norm = normalize_text(bw_raw, form="NFC")
                        conf = bw_conf
                        used_model = bw_model
                        script = detect_script(text_norm)
                except Exception:
                    pass

            if script == "latin":
                spaced = reconstruct_visual_word_spaces(crop, text_norm, thresholds=cfg)
                if spaced != text_norm:
                    text_norm = spaced
                    text_raw = spaced

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
