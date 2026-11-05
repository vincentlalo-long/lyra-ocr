"""Environment provenance: tool and library versions recorded in artifacts."""

from __future__ import annotations

import platform
import shutil
import subprocess


def get_tool_versions() -> dict[str, str]:
    """Inspect installed versions of dependencies and external tools."""
    versions: dict[str, str] = {
        "python": platform.python_version(),
    }
    try:
        from importlib.metadata import version
        versions["rapidocr"] = version("rapidocr")
    except Exception:
        versions["rapidocr"] = "unknown"

    try:
        import onnxruntime
        versions["onnxruntime"] = onnxruntime.__version__
    except Exception:
        versions["onnxruntime"] = "unknown"

    try:
        import cv2
        versions["opencv"] = cv2.__version__
    except Exception:
        versions["opencv"] = "unknown"

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        res = subprocess.run([ffmpeg_bin, "-version"], capture_output=True, text=True, check=False)
        first_line = res.stdout.splitlines()[0] if res.stdout else ""
        versions["ffmpeg"] = first_line.split("Copyright")[0].strip()
    else:
        versions["ffmpeg"] = "missing"

    return versions
