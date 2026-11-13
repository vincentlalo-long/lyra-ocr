"""Unit tests for batched recognition (fake RapidOCR, no models loaded)."""

from unittest.mock import patch

import numpy as np
import pytest

from lyra_ocr.ocr import runtime as runtime_mod
from lyra_ocr.ocr.detect import Detection, crop_quad
from lyra_ocr.ocr.engine import OCREngine, preprocess_crop_for_recognition


class FakeOut:
    def __init__(self, txts, scores):
        self.txts = txts
        self.scores = scores


class EmptyOut:
    txts = ()
    scores = ()


class FakeRapidOCR:
    """Records batch calls; encodes each crop's mean pixel value as its text."""

    instances: list["FakeRapidOCR"] = []

    def __init__(self, params=None, **kwargs):
        self.params = params or {}
        self.rec_calls: list[int] = []
        FakeRapidOCR.instances.append(self)

    def __call__(self, img):
        return FakeOut(("single",), (0.5,))

    def recognize_txt(self, imgs):
        self.rec_calls.append(len(imgs))
        txts = tuple(f"t{int(round(float(np.asarray(c).mean())))}" for c in imgs)
        return FakeOut(txts, tuple(0.9 for _ in imgs))


@pytest.fixture(autouse=True)
def _cpu_runtime(monkeypatch):
    monkeypatch.setattr(runtime_mod, "available_providers", lambda: ["CPUExecutionProvider"])
    monkeypatch.delenv("LYRA_OCR_DEVICE", raising=False)
    FakeRapidOCR.instances.clear()


def _make_engine(**kwargs) -> OCREngine:
    with patch("rapidocr.RapidOCR", FakeRapidOCR):
        return OCREngine(**kwargs)


def test_recognize_crops_preserves_order_in_one_call():
    engine = _make_engine()
    crops = [
        np.full((8, 8, 3), 10, np.uint8),
        np.full((8, 8, 3), 20, np.uint8),
        np.full((8, 8, 3), 30, np.uint8),
    ]

    results = engine.recognize_crops(crops)

    assert [text for text, _score, _model in results] == ["t10", "t20", "t30"]
    assert FakeRapidOCR.instances[0].rec_calls == [3]
    assert all(model == OCREngine.DEFAULT_MODEL for _t, _s, model in results)
    assert results[0][1] == 0.9


def test_recognize_crops_empty_list():
    engine = _make_engine()
    assert engine.recognize_crops([]) == []


def test_recognize_crops_skips_empty_crops():
    engine = _make_engine()
    crops = [
        np.full((8, 8, 3), 10, np.uint8),
        np.zeros((0, 0, 3), np.uint8),
    ]

    results = engine.recognize_crops(crops)

    assert results[0][0] == "t10"
    assert results[1] == ("", 0.0, OCREngine.DEFAULT_MODEL)
    assert FakeRapidOCR.instances[0].rec_calls == [1]


def test_recognize_crops_handles_missing_txts():
    engine = _make_engine()
    FakeRapidOCR.instances[0].recognize_txt = lambda imgs: EmptyOut()

    results = engine.recognize_crops([np.full((8, 8, 3), 10, np.uint8)])

    assert results == [("", 0.0, OCREngine.DEFAULT_MODEL)]


def test_process_detections_batches_all_crops_in_detection_order():
    engine = _make_engine()
    image = np.zeros((40, 200, 3), np.uint8)
    image[:, :100] = 30
    image[:, 100:] = 60
    detections = [
        Detection(polygon=[[0, 0], [100, 0], [100, 40], [0, 40]], bbox=[0, 0, 100, 40], score=0.9),
        Detection(
            polygon=[[100, 0], [200, 0], [200, 40], [100, 40]],
            bbox=[100, 0, 200, 40],
            score=0.9,
        ),
    ]
    expected = [
        f"t{int(round(float(preprocess_crop_for_recognition(crop_quad(image, det.polygon)).mean())))}"
        for det in detections
    ]

    lines = engine.process_detections(image, detections)

    assert [line.text_raw for line in lines] == expected
    assert FakeRapidOCR.instances[0].rec_calls == [2]
    assert [line.bbox for line in lines] == [det.bbox for det in detections]


def test_rec_batch_size_and_runtime_reach_rapidocr_params():
    runtime = runtime_mod.resolve_runtime("cpu", 16)
    engine = _make_engine(runtime=runtime)

    assert FakeRapidOCR.instances[0].params["Rec.rec_batch_num"] == 16
    assert FakeRapidOCR.instances[0].params["EngineConfig.onnxruntime.use_cuda"] is False
    assert engine.runtime_info()["rec_batch_size"] == 16
