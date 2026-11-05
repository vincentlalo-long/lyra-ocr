"""Scene text detection and recognition.

`detect` wraps the DBNet detector (polygons only); `engine` crops polygons and
runs recognition, producing normalized `TextLine` records.
"""

from lyra_ocr.ocr.detect import Detection, TextDetector, crop_quad, polygon_to_bbox
from lyra_ocr.ocr.engine import OCREngine, TextLine, detect_script, normalize_text

__all__ = [
    "Detection",
    "TextDetector",
    "OCREngine",
    "TextLine",
    "crop_quad",
    "detect_script",
    "normalize_text",
    "polygon_to_bbox",
]
