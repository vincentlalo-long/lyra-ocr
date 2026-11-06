"""Unit tests for OCR device/provider resolution (no GPU required)."""

import pytest

from lyra_ocr.ocr import runtime as runtime_mod
from lyra_ocr.ocr.runtime import rapidocr_params, resolve_runtime

CPU = ["CPUExecutionProvider"]
GPU = ["CUDAExecutionProvider", "CPUExecutionProvider"]


def _providers(monkeypatch, providers):
    monkeypatch.setattr(runtime_mod, "available_providers", lambda: list(providers))


def test_auto_uses_cpu_when_no_cuda(monkeypatch):
    _providers(monkeypatch, CPU)
    monkeypatch.delenv("LYRA_OCR_DEVICE", raising=False)

    rt = resolve_runtime()

    assert rt.request == "auto"
    assert rt.device == "cpu"
    assert rt.use_cuda is False

    params = rapidocr_params(rt, use_det=True, use_rec=False)
    assert params["Global.use_det"] is True
    assert params["Global.use_rec"] is False
    assert params["Global.use_cls"] is False
    assert params["EngineConfig.onnxruntime.use_cuda"] is False
    assert params["Rec.rec_batch_num"] == 8


def test_auto_prefers_cuda(monkeypatch):
    _providers(monkeypatch, GPU)
    monkeypatch.delenv("LYRA_OCR_DEVICE", raising=False)

    rt = resolve_runtime("auto", 16)

    assert rt.device == "cuda"
    assert rt.use_cuda is True
    assert rt.rec_batch_size == 16
    assert "CUDAExecutionProvider" in rt.providers

    params = rapidocr_params(rt, use_det=False, use_rec=True)
    assert params["EngineConfig.onnxruntime.use_cuda"] is True
    assert params["Rec.rec_batch_num"] == 16


def test_cuda_request_falls_back_to_cpu(monkeypatch):
    _providers(monkeypatch, CPU)
    monkeypatch.delenv("LYRA_OCR_DEVICE", raising=False)

    rt = resolve_runtime("cuda")

    assert rt.request == "cuda"
    assert rt.device == "cpu"
    assert rt.use_cuda is False


def test_cuda_strict_raises(monkeypatch):
    _providers(monkeypatch, CPU)

    with pytest.raises(RuntimeError):
        resolve_runtime("cuda", strict=True)


def test_cpu_request_ignores_cuda_and_env(monkeypatch):
    _providers(monkeypatch, GPU)
    monkeypatch.setenv("LYRA_OCR_DEVICE", "cuda")

    rt = resolve_runtime("cpu")

    assert rt.device == "cpu"
    assert rt.use_cuda is False


def test_env_overrides_auto(monkeypatch):
    _providers(monkeypatch, GPU)
    monkeypatch.setenv("LYRA_OCR_DEVICE", "cpu")

    assert resolve_runtime("auto").device == "cpu"

    monkeypatch.setenv("LYRA_OCR_DEVICE", "cuda")
    assert resolve_runtime("auto").device == "cuda"


def test_invalid_env_falls_back_to_auto(monkeypatch):
    _providers(monkeypatch, CPU)
    monkeypatch.setenv("LYRA_OCR_DEVICE", "tpu")

    assert resolve_runtime("auto").device == "cpu"


@pytest.mark.parametrize("bad_batch", [0, -1])
def test_invalid_batch_size_rejected(bad_batch):
    with pytest.raises(ValueError):
        resolve_runtime("cpu", bad_batch)


def test_invalid_device_rejected():
    with pytest.raises(ValueError):
        resolve_runtime("tpu")
