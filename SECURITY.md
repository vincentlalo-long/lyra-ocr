# Security Policy

## Supported Versions

Security fixes are applied to the active `main` development branch (`0.1.x` pre-release series).

| Version / Branch | Supported |
|---|---|
| `main` (`0.1.x`) | Yes |
| Older commits / unmaintained forks | No |

---

## Reporting a Vulnerability

Please **do not** report suspected security vulnerabilities in public GitHub issues or pull requests.

Instead, report them privately by emailing **`nickmoi3056@gmail.com`** (or via GitHub Private Vulnerability Reporting on [`vincentlalo-long/lyra-ocr`](https://github.com/vincentlalo-long/lyra-ocr) when enabled) with:

- A clear description of the issue and potential impact
- Exact steps to reproduce (CLI flags, API request, or sample input structure)
- Affected files, functions, or environment details (OS, Python version)

You should receive an acknowledgement and triage update once the report is reviewed.

---

## Operational Security Notes

### Local Web UI (`lyra-ocr review-ui`)

The interactive review and LRC Studio web server (`lyra-ocr review-ui`, implemented in `src/lyra_ocr/web/server.py` and `src/lyra_ocr/web/base.py`) is designed **strictly for local, single-user workflows**:

- **Default binding**: `lyra-ocr review-ui` binds to `127.0.0.1` on port `8000` by default (`--host 127.0.0.1`, `--port 8000`).
- **No authentication or multi-tenant isolation**: The HTTP server has no login, session token, or role-based access control, and responses include permissive CORS headers (`Access-Control-Allow-Origin: *`) for local development convenience.
- **Do NOT expose on `0.0.0.0` on untrusted networks**: Running `lyra-ocr review-ui --host 0.0.0.0` exposes endpoints that can read media/artifacts, write `.lrc` and label files, delete artifact/run directories, and enqueue background `yt-dlp`, `ffmpeg`, and OCR jobs within `--workspace-root`. If remote access is necessary, keep `--host 127.0.0.1` and forward the port over an authenticated SSH tunnel or an authenticated reverse proxy.
- **Built-in containment guards**:
  - File path parameters are checked through `_safe_resolve()` (`src/lyra_ocr/web/base.py`) to prevent directory traversal outside `--workspace-root` (default: current working directory).
  - Static UI assets are resolved strictly under `STATIC_DIR`.
  - `POST` request bodies are bounded by `MAX_BODY_BYTES` (8 MiB in `src/lyra_ocr/web/server.py`).

### Untrusted Media, URLs & External Services

- Video files and YouTube URLs passed to `lyra-ocr` or `review-ui` are processed by `ffmpeg`, `ffprobe`, `yt-dlp`, OpenCV, and ONNX Runtime. Treat media from untrusted sources accordingly and keep system `ffmpeg` and `yt-dlp` updated.
- Enabling `--use-reference` (or LRC Studio reference lookup) sends track title/artist metadata queries to external providers (LRCLIB and, unless `--no-caption-fallback` is set, YouTube captions via `yt-dlp`). Omit `--use-reference` if video metadata must stay strictly offline.
