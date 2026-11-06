"""ONNX Runtime device/provider selection and RapidOCR parameter mapping.

One place decides whether OCR runs on CUDA or CPU, how many recognition crops
are batched per ONNX call, and which session thread settings RapidOCR receives.
GPU is an accelerator only: every failure path falls back to CPU so machines
without CUDA behave exactly as before.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

ENV_DEVICE = "LYRA_OCR_DEVICE"
VALID_REQUESTS = ("auto", "cpu", "cuda")
DEFAULT_REC_BATCH_SIZE = 8
DEFAULT_CUDNN_CONV_ALGO_SEARCH = "HEURISTIC"

_preload_attempted = False


@dataclass(frozen=True)
class RuntimeConfig:
    """Resolved OCR runtime settings shared by detection and recognition."""

    request: str = "auto"  # auto|cpu|cuda (what the caller asked for)
    device: str = "cpu"  # cpu|cuda (what is actually used)
    rec_batch_size: int = DEFAULT_REC_BATCH_SIZE
    intra_op_threads: int = -1
    inter_op_threads: int = 1
    use_cuda: bool = False
    use_coreml: bool = False
    cudnn_conv_algo_search: str = DEFAULT_CUDNN_CONV_ALGO_SEARCH
    providers: tuple[str, ...] = ()


def _import_onnxruntime():
    try:
        import onnxruntime as ort
    except ImportError:
        return None
    return ort


def _preload_dlls(ort) -> None:
    """Load CUDA/cuDNN DLLs once before session creation (onnxruntime-gpu only).

    CPU-only onnxruntime ships ``preload_dlls`` too but has nothing to load;
    any failure here must not break the CPU path.
    """
    global _preload_attempted
    if _preload_attempted:
        return
    _preload_attempted = True

    preload = getattr(ort, "preload_dlls", None)
    if preload is None:
        return
    try:
        preload(cuda=True, cudnn=True)
    except TypeError:
        try:
            preload()
        except Exception as exc:  # noqa: BLE001 - accelerator is optional
            logger.debug("onnxruntime.preload_dlls() failed: %s", exc)
    except Exception as exc:  # noqa: BLE001 - accelerator is optional
        logger.debug("onnxruntime.preload_dlls(cuda=True, cudnn=True) failed: %s", exc)


def available_providers() -> list[str]:
    """Execution providers compiled into the installed onnxruntime package."""
    ort = _import_onnxruntime()
    if ort is None:
        return []
    _preload_dlls(ort)
    try:
        return list(ort.get_available_providers())
    except Exception as exc:  # noqa: BLE001 - never break OCR for provenance
        logger.debug("onnxruntime.get_available_providers() failed: %s", exc)
        return []


def resolve_runtime(
    request: str = "auto",
    rec_batch_size: int = DEFAULT_REC_BATCH_SIZE,
    *,
    strict: bool = False,
    intra_op_threads: int = -1,
    inter_op_threads: int = 1,
) -> RuntimeConfig:
    """Resolve ``request`` into a concrete device with a safe CPU fallback.

    - ``auto``: CUDA when the installed onnxruntime exposes
      ``CUDAExecutionProvider``, otherwise CPU.
    - ``cuda``: CUDA when available; with ``strict`` raise, otherwise warn and
      fall back to CPU.
    - ``cpu``: always CPU.

    ``LYRA_OCR_DEVICE`` overrides the device when ``request`` is ``auto``.
    """
    if rec_batch_size < 1:
        raise ValueError(f"rec_batch_size must be >= 1 (got {rec_batch_size})")

    req = (request or "auto").strip().lower()
    if req not in VALID_REQUESTS:
        raise ValueError(f"device must be one of {VALID_REQUESTS} (got {request!r})")

    env_req = os.environ.get(ENV_DEVICE, "").strip().lower()
    if req == "auto" and env_req:
        if env_req in VALID_REQUESTS:
            req = env_req
        else:
            logger.warning("ignoring invalid %s=%r", ENV_DEVICE, env_req)

    providers = available_providers()
    has_cuda = "CUDAExecutionProvider" in providers

    use_cuda = False
    if req == "cuda":
        if has_cuda:
            use_cuda = True
        elif strict:
            raise RuntimeError(
                "CUDAExecutionProvider is not available; install onnxruntime-gpu "
                "with the [cuda,cudnn] extras or use device=auto/cpu"
            )
        else:
            logger.warning(
                "CUDA requested but CUDAExecutionProvider is unavailable; falling back to CPU"
            )
    elif req == "auto":
        use_cuda = has_cuda

    return RuntimeConfig(
        request=req,
        device="cuda" if use_cuda else "cpu",
        rec_batch_size=int(rec_batch_size),
        intra_op_threads=int(intra_op_threads),
        inter_op_threads=int(inter_op_threads),
        use_cuda=use_cuda,
        providers=tuple(providers),
    )


def rapidocr_params(runtime: RuntimeConfig, *, use_det: bool, use_rec: bool) -> dict:
    """Build the RapidOCR parameter dict for a detector/recogniser instance."""
    return {
        "Global.use_det": use_det,
        "Global.use_cls": False,
        "Global.use_rec": use_rec,
        "EngineConfig.onnxruntime.use_cuda": runtime.use_cuda,
        "EngineConfig.onnxruntime.use_coreml": runtime.use_coreml,
        "EngineConfig.onnxruntime.intra_op_num_threads": runtime.intra_op_threads,
        "EngineConfig.onnxruntime.inter_op_num_threads": runtime.inter_op_threads,
        "EngineConfig.onnxruntime.cuda_ep_cfg.cudnn_conv_algo_search": (
            runtime.cudnn_conv_algo_search
        ),
        "Rec.rec_batch_num": runtime.rec_batch_size,
    }


def runtime_report(runtime: RuntimeConfig, providers: list[str] | None = None) -> dict:
    """Manifest/provenance block describing the effective OCR runtime."""
    return {
        "requested": runtime.request,
        "device": runtime.device,
        "providers": list(providers) if providers else list(runtime.providers),
        "rec_batch_size": runtime.rec_batch_size,
        "intra_op_threads": runtime.intra_op_threads,
        "inter_op_threads": runtime.inter_op_threads,
    }
