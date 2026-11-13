# Lyra OCR

Pipeline và CLI trích xuất lyric hardcode từ video thành phụ đề `.lrc` đồng bộ thời gian (hỗ trợ tiếng Anh, tiếng Nhật Kanji/Kana, Romaji `ja-Latn`, và các video đa ngôn ngữ).

---

## Tổng quan kiến trúc & Trạng thái hiện tại

Pipeline gồm 2 giai đoạn chính (chạy qua CLI `lyra-ocr` hoặc giao diện Web `lyra-ocr review-ui`):

1. **Nhận diện & Theo dõi theo thời gian (`identify` / `smoke`)**:
   - Đọc metadata (`ffprobe`), trích xuất frame theo PTS (`ffmpeg`, chế độ `pipe` hoặc `disk`, tuỳ chọn `--dedup`).
   - Phát hiện vùng chữ bằng RapidOCR DBNet (`src/lyra_ocr/ocr/detect.py`, có tách box 2 dòng theo horizontal projection profile) và nhận diện chữ (`src/lyra_ocr/ocr/engine.py`, phóng to crop chữ nhỏ `< 36px`, áp dụng CLAHE cho vùng đủ tương phản, padding viền).
   - Gom nhóm theo không gian-thời gian thành `TextTrack` (`greedy` hoặc `bytetrack`) và phân loại `lyric` / `non_lyric` theo rule (`src/lyra_ocr/classify.py`: watermark, title card ở intro, chat/overlay).
2. **Dựng & Xuất file LRC đa ngôn ngữ (`lrc`)**:
   - Phân tầng không gian & phát hiện ngôn ngữ (`src/lyra_ocr/layers.py`, `src/lyra_ocr/lang.py`): giữ riêng các layer `ja`, `ja-Latn`, `en`, v.v., đồng thời giữ lại companion band khi lyric hiển thị nhiều dòng xếp chồng.
   - Sửa lỗi OCR không cần mạng (`dictionary`: từ điển SymSpell tiếng Anh cho tách/sửa từ + sửa lỗi ký tự dễ nhầm của Romaji Hepburn trong `src/lyra_ocr/lyric/text_repair.py`).
   - Tín hiệu audio (`src/lyra_ocr/audio/`): tách audio 16kHz mono, profile RMS VAD + căn chỉnh onset (`forced_align.py`), và ASR cấp từng từ bằng `faster-whisper` (`asr.py`) được cache vào `asr_transcript.json` để:
     - tách timestamp cho stanza nhiều dòng hiện đồng thời (`resolve_simultaneous_multiline_cues`, fallback sang chia tỷ lệ độ dài chữ nếu không khớp ASR);
     - đối chiếu/sửa text khi không có reference ngoài;
     - gắn cờ `asr_unvoiced` / lọc overlay độ tin cậy thấp (`< 0.65`) ở các đoạn không có giọng hát (`asr_vocal_gate_score`).
   - Đối chiếu reference tuỳ chọn (`--use-reference`): thứ tự ưu tiên Manual > LRCLIB > YouTube captions, căn chỉnh N:M (`1:1`, `1:2`, `1:3`, `2:1`, `3:1`), chuyển tự `ja -> ja-Latn` có bảo vệ ateji (giữ cách đọc OCR khi độ tin cậy cao) và không để lọt ký tự CJK vào layer Romaji.
   - Xuất file (`src/lyra_ocr/lyric/export.py`): tạo `<name>.lrc` (layer chính, ưu tiên `ja-Latn` > ngôn ngữ gốc > `en`), các file `<name>.<lang>.lrc` cho từng layer, và `<name>.cues.json` lưu bbox, độ tin cậy, log bỏ qua (`skipped`) và nguồn reference.

### Giới hạn hiện tại

- Pipeline phát hiện/tách dòng hiện chỉ nhắm vào chữ **nằm ngang** (chưa hỗ trợ chia cột chữ dọc *tategaki*).
- `faster-whisper` chạy trực tiếp trên audio hỗn hợp 16kHz mono (chưa tích hợp bước tách nhạc/giọng bằng Demucs/UVR).
- Khi video có nhạc nền át giọng hoặc lyric ateji không khớp từ điển/reference, độ chính xác vẫn phụ thuộc vào chất lượng crop OCR ban đầu và dữ liệu LRCLIB/ASR.
- Các chỉ số benchmark dưới đây chỉ mới đo trên tập tiếng Anh **Phase 1 (`Fix_lyric_EN`)**; các module xử lý đa ngôn ngữ, ASR và LRC sau đó được kiểm chứng qua unit test và test mẫu cục bộ.

---

## Tài liệu kỹ thuật

- [Kiến trúc hệ thống](doc/ARCHITECTURE.md): phân rã module, entity domain và các quyết định thiết kế.
- [Đặc tả dữ liệu & Artifact](doc/SPECIFICATION.md): schema cho `manifest.json`, `frame_index.json`, `ocr_raw.json`, `tracks.json`, `review.csv`, `labels.jsonl`.

---

## Yêu cầu môi trường

- **Python**: `>= 3.11`
- **Công cụ hệ thống**: `ffmpeg` và `ffprobe` có trên `PATH`
- **Dependencies chính** (`pyproject.toml`): `rapidocr>=3.0.0`, `onnxruntime>=1.17.0`, `opencv-python>=4.8.0`, `numpy>=1.24.0`, `pillow>=10.0.0`, `scipy>=1.10.0`, `yt-dlp>=2024.0.0`, `symspellpy>=6.7.7`, `pykakasi>=2.3.0`, `faster-whisper>=1.0.0`

---

## Cài đặt & Sử dụng

### 1. Cài đặt

```bash
git clone https://github.com/vincentlalo-long/lyra-ocr.git
cd lyra-ocr
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,eval]"
```

*(Tuỳ chọn GPU cho ONNX / faster-whisper)*:

```bash
pip uninstall -y onnxruntime
pip install "onnxruntime-gpu[cuda,cudnn]==1.31.1"
```

### 2. Chạy trích xuất & Xuất `.lrc`

```bash
# Chạy toàn bộ pipeline từ video sang .lrc (lưu artifact vào ./lyra_output/<video_stem>)
lyra-ocr lrc path/to/video.mp4 -o song.lrc --tracker bytetrack --device auto

# Bật đối chiếu reference (Manual / LRCLIB / YouTube captions)
lyra-ocr lrc path/to/video.mp4 -o song.lrc --tracker bytetrack --use-reference

# Xuất lại .lrc từ thư mục artifact có sẵn (không chạy lại OCR video)
lyra-ocr lrc --artifacts ./lyra_output/video -o song.lrc
```

Một số cờ hay dùng của `lyra-ocr lrc`:
- `-o, --out`: đường dẫn `.lrc` đầu ra (sinh thêm `<name>.<lang>.lrc` nếu có nhiều layer ngôn ngữ và `<name>.cues.json`).
- `--artifacts`: thư mục artifact từ bước `identify` / `lrc` trước đó.
- `--tracker`: `greedy` hoặc `bytetrack` (`greedy` là mặc định trong CLI; `bytetrack` được khuyến nghị cho video chuyển cảnh/karaoke).
- `--correct`: `none`, `consensus`, `dictionary` (mặc định `dictionary`).
- `--refine`: quét thêm frame quanh biên để tinh chỉnh mốc thời gian.
- `--include-review`: giữ cả cue đang ở trạng thái `review`.
- `--use-reference` / `--no-caption-fallback`: bật sửa chữ & bù khoảng trống bằng LRCLIB / YouTube captions.
- `--offset-ms`: chỉnh tag `[offset:+/-ms]`.
- `--force`: bỏ qua cache và chạy lại pipeline.

### 3. Chạy chỉ bước nhận diện & review (`identify`, `label`, `review-ui`)

```bash
# Bước 1: trích xuất tracks.json, review.csv, roi_preview.jpg
lyra-ocr identify path/to/video.mp4 --out ./output --tracker bytetrack --device auto

# Gán nhãn review CSV ngoại tuyến
lyra-ocr label export --artifacts ./output --out label_batch.csv --blind
lyra-ocr label ingest --artifacts ./output --csv label_batch.csv --annotator human

# Hoặc mở Web UI (Runs / Review / Eval / Datasets / LRC Studio)
lyra-ocr review-ui --port 8000
```

---

## Kiểm thử & Đánh giá

### Chạy unit test

```bash
pytest
```

Mặc định trong `pyproject.toml`, `pytest` chạy các test nhanh (`-m 'not slow and not gpu'`).

### Benchmark Phase 1 (25 video `Fix_lyric_EN`)

Đánh giá phân loại & theo dõi track Phase 1 trên tập 25 video tiếng Anh thuộc `Fix_lyric_EN` (>1.350 track so với Silver GT tự động):

| Metric | Baseline Heuristic | Greedy (Tuned) | ByteTrack (Tuned) |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 82.93% | 93.30% | 93.08% |
| **Lyric Precision** | 99.60% | 99.75% | 99.83% |
| **Lyric Recall** | 81.83% | 92.91% | 92.93% |
| **F1-Score** | 89.84% | 96.21% | 96.26% |
| **Tracks tạo ra** | 1,646 (30 vid) | 1,417 (25 vid) | 1,359 (25 vid) |

Lệnh đánh giá lại từ dataset hoặc file LRC:

```bash
lyra-ocr eval --golden path/to/dataset --out outputs/eval --charts --compare-trackers
python scripts/eval_lrc_text.py outputs/song/song.cues.json
```
