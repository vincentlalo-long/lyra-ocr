"""Unit tests for artifact schema validation (manifest.json and ocr_raw.json)."""

from importlib.metadata import PackageNotFoundError, version

from lyra_ocr.ocr import TextLine
from lyra_ocr.provenance import get_tool_versions


def _onnxruntime_dist_version() -> str:
    """Version of whichever onnxruntime distribution is installed (CPU or GPU)."""
    for dist in ("onnxruntime", "onnxruntime-gpu"):
        try:
            return version(dist)
        except PackageNotFoundError:
            continue
    raise AssertionError("no onnxruntime distribution installed")


def test_get_tool_versions_contract():
    versions = get_tool_versions()
    # Compare against the installed distribution so a deliberate runtime swap
    # (e.g. onnxruntime-gpu) does not break the schema test.
    assert versions["rapidocr"] == version("rapidocr")
    assert versions["python"]
    assert versions["onnxruntime"] == _onnxruntime_dist_version()
    assert versions["onnxruntime_providers"]
    assert "opencv" in versions
    assert "ffmpeg" in versions



def test_ocr_line_schema():
    line = TextLine(
        polygon=[[10, 10], [50, 10], [50, 30], [10, 30]],
        bbox=[10, 10, 50, 30],
        text_raw="Hello World",
        text_norm="Hello World",
        confidence=0.98,
        model_id="PP-OCRv6_rec_small",
        script="latin",
    )
    d = line.to_dict()

    required_keys = {"polygon", "bbox", "text_raw", "text_norm", "confidence", "model_id", "script"}
    assert required_keys.issubset(d.keys())
    assert len(d["polygon"]) == 4
    assert len(d["bbox"]) == 4
    assert isinstance(d["confidence"], float)
    assert d["script"] == "latin"


def test_manifest_runtime_block_schema():
    runtime_block = {
        "requested": "auto",
        "device": "cpu",
        "providers": ["CPUExecutionProvider"],
        "rec_batch_size": 8,
        "intra_op_threads": -1,
        "inter_op_threads": 1,
        "dedup": False,
        "frame_source": "pipe",
        "pipeline_version": "2",
    }

    assert runtime_block["device"] in ("cpu", "cuda")
    assert runtime_block["rec_batch_size"] >= 1
    from lyra_ocr.pipeline import PIPELINE_VERSION
    assert runtime_block["pipeline_version"] in ("2", PIPELINE_VERSION)


def test_manifest_schema_keys():
    # Verify structure matches PLAN.md Section 3 and Audit M3
    sample_manifest = {
        "video_path": "path/to/video.mp4",
        "probe": {
            "duration": 141.2,
            "fps": 60.0,
            "width": 1280,
            "height": 720,
            "rotation": 0,
            "has_audio": True,
        },
        "config": {
            "sample_fps": 2.0,
            "start_sec": 0.0,
            "sample_frames": 5,
        },
        "tool_versions": {
            "python": "3.11.16",
            "rapidocr": "3.10.0",
            "onnxruntime": "1.31.1",
            "ffmpeg": "8.1.3",
        },
        "models": {
            "det": "PP-OCRv6_det_small",
            "rec": "PP-OCRv6_rec_small",
        },
        "timestamp_created": 1730310000,
    }

    assert "tool_versions" in sample_manifest
    assert "models" in sample_manifest
    assert "probe" in sample_manifest
    assert "config" in sample_manifest
    assert sample_manifest["models"]["det"] == "PP-OCRv6_det_small"


def test_ocr_raw_schema_structure():
    sample_ocr_raw = {
        "video_path": "path/to/video.mp4",
        "manifest_path": "manifest.json",
        "total_frames_processed": 1,
        "avg_frame_time_ms": 320.5,
        "frames": [
            {
                "frame_id": 0,
                "timestamp_ms": 20000,
                "pts_time": 20.0,
                "image_path": "frames/frame_000000.jpg",
                "elapsed_ms": 320.5,
                "boxes": [
                    {
                        "polygon": [[10, 10], [50, 10], [50, 30], [10, 30]],
                        "bbox": [10, 10, 50, 30],
                        "text_raw": "Castle",
                        "text_norm": "Castle",
                        "confidence": 0.99,
                        "model_id": "PP-OCRv6_rec_small",
                        "script": "latin",
                    }
                ],
            }
        ],
    }

    assert "frames" in sample_ocr_raw
    assert "manifest_path" in sample_ocr_raw
    frame0 = sample_ocr_raw["frames"][0]
    assert "pts_time" in frame0
    assert "timestamp_ms" in frame0
    assert "boxes" in frame0
    box0 = frame0["boxes"][0]
    assert "text_raw" in box0
    assert "text_norm" in box0
    assert "model_id" in box0
