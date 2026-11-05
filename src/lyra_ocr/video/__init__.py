"""Video input: metadata probing and PTS frame sampling.

`download_video` lives in `.downloader` and is intentionally not re-exported so
importing this package does not pull in yt-dlp.
"""

from lyra_ocr.video.frames import sample_frames
from lyra_ocr.video.probe import probe_video

__all__ = ["probe_video", "sample_frames"]
