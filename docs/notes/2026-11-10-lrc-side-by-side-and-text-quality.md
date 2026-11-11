# Report: LRC Studio side-by-side + nâng chất lượng text bằng reference lyrics

- **Ngày:** 2026-11-10
- **Branch:** `feat/phase2-lrc`
- **Trạng thái:** đã implement và kiểm chứng, **chưa commit** (theo yêu cầu)

---

## 1. Mục tiêu

1. Có giao diện **side-by-side** khi test file LRC: bên trái video chạy, bên phải danh sách LRC tương ứng, dòng đang phát được highlight.
2. Giảm lỗi text OCR: thay vì chỉ dựa vào dictionary nội bộ, đối chiếu với **lyrics tham chiếu** để sửa chữ.
3. Viết report lại toàn bộ thay đổi.

---

## 2. Giao diện side-by-side (video ↔ LRC)

### Backend

- Thêm endpoint `GET /api/media?path=...` trong `src/lyra_ocr/web/lrc_api.py`:
  - Chỉ phục vụ file **bên trong workspace** (dùng `_safe_resolve`).
  - Hỗ trợ **HTTP Range** (`206 Partial Content`, `Content-Range`, `Accept-Ranges`) để tua video mượt.
  - Stream theo chunk 64 KB, bỏ qua `BrokenPipeError` khi người dùng tua/đóng tab.
- Đăng ký route trong `src/lyra_ocr/web/server.py`.
- `POST /api/lrc/generate` trả thêm `video_path` (lấy từ manifest/build), `reference_source`, `n_reference_matches`.
- `GET /api/lrc?path=...` trả thêm `video_path` đọc từ `.cues.json`, nên mở lại LRC cũ vẫn có video.

### Frontend

- `index.html`:
  - Ô **“Side-by-side Test: Load Existing .LRC”** + nút **Load**.
  - Thẻ `<video id="studioVideo">` nằm trên karaoke preview trong panel trái.
  - Ô **Reference Lyrics File** và checkbox **Match Lyrics from LRCLIB**.
- `lrc_studio.js`:
  - Nút Play/Pause, thanh tua và click timestamp điều khiển trực tiếp `<video>` khi có video; `timeupdate`/`seeked` đồng bộ `playbackTimeMs` với `video.currentTime`.
  - `loadedmetadata` cập nhật độ dài thật của video.
  - Nếu không có video (hoặc video lỗi), giữ nguyên chế độ player mô phỏng cũ.
  - Load file `.lrc` có sẵn: ưu tiên cues từ `.cues.json`; nếu không có thì **parse timestamp ngay trong JS**, nên test được cả file LRC thuần.
  - Dòng đang phát được highlight trong danh sách cue + hiển thị ở karaoke overlay dưới video.
- `app.css`: style `.studio-video-wrapper` / `.studio-video`.

**Cách dùng:** dán đường dẫn `.lrc` → **Load** (hoặc Generate LRC như cũ) → panel chia đôi: trái video, phải LRC. Bấm timestamp hoặc tua để nhảy đúng vị trí.

---

## 3. Nâng chất lượng text: reference lyrics alignment

### Vì sao không thêm “model lớn” ngay

- LLM/ASR lớn sẽ tăng latency, cần GPU/network và có nguy cơ **bịa lyric** (hallucination) — tệ hơn OCR sai.
- Với lyric có sẵn, **đối chiếu với nguồn đúng** cho kết quả chính xác và rẻ hơn nhiều. Đây là hướng đã chọn.

### Nguồn tham chiếu

1. **File local** (`.lrc` hoặc `.txt`) do người dùng cung cấp — ưu tiên cao nhất.
2. **LRCLIB** (`https://lrclib.net/api/search`) — miễn phí, không cần API key, có cả synced/plain lyrics. Lookup best-effort, timeout 6 s, lỗi mạng chỉ log warning và bỏ qua (không làm hỏng pipeline).

### Thuật toán (`src/lyra_ocr/lyric/reference.py`)

- Chuẩn hoá để so khớp: lowercase, bỏ dấu, bỏ punctuation/khoảng trắng.
- **Monotonic alignment** bằng DP: tối đa tổng similarity, cho phép bỏ dòng OCR hoặc bỏ dòng reference, nên chorus lặp và dòng bị mất vẫn khớp đúng thứ tự.
- Ngưỡng khớp mặc định `0.55`; dưới ngưỡng thì **giữ nguyên text OCR** (không phá tên riêng/câu lạ).
- **Chỉ thay text, không thay timestamp**: timing vẫn do OCR/visual quyết định, đúng nguyên tắc “runtime chỉ dùng video”.
- Cue được sửa có `correction.method = "reference"` và `diff` để truy vết.
- **Gate relief:** nếu text OCR khớp reference ≥ 0.55 thì bỏ qua quality gate — cứu các dòng thật nhưng bị OCR làm sai chính tả (trước đây bị loại vì `low_text_quality`).
- Cache signature trong `build.py` có thêm mode + SHA-1 của reference text, nên đổi nguồn reference là cache tự invalidate.
- `.cues.json` ghi thêm `reference_source` và `n_reference_matches`.

### API/UI

- `POST /api/lrc/generate` nhận thêm:
  - `reference_path`: đường dẫn file lyrics local.
  - `use_reference`: bật lookup LRCLIB (dùng `title` + `artist` đang nhập).
- UI hiển thị badge **“Reference Matches”** trong thanh thống kê.

---

## 4. Kiểm chứng

| Kiểm tra | Kết quả |
|---|---|
| Full suite `.venv/bin/python -m pytest tests/unit tests/eval -q` | **259 passed, 11 skipped, 1 deselected** |
| `node --check src/lyra_ocr/web/static/js/lrc_studio.js` | OK |
| `GET /api/media` với `Range: bytes=0-1023` trên video thật | **HTTP 206**, đúng 1024 byte, `Accept-Ranges: bytes` |
| Build artifact Punishing thật | 67 cues, không traceback |
| Align reference trên cue thật “...makes me **feet** this way” | sửa đúng thành “...makes me **feel** this way”, `replaced=1` |
| Unit test mới `tests/unit/test_lyric_reference.py` | parse LRC/txt, sửa text giữ timing, không phá dòng không khớp, tích hợp build |

---

## 5. Hạn chế

- LRCLIB cần mạng và **title/artist đúng**; nếu metadata sai thì không tìm được lyrics. Không có fallback tải từ nguồn khác.
- Reference sửa **text**, không sửa timing; dòng quá nát (< 0.55 similarity) vẫn giữ text OCR.
- Chorus lặp nhiều lần vẫn xử lý được nhờ DP, nhưng nếu hai dòng gần giống nhau hoàn toàn thì có thể khớp lệch một dòng.
- `/api/media` chỉ phục vụ file trong workspace (chủ ý, để tránh đọc file ngoài).
- Không thêm model lớn/LLM; nếu sau này cần xử lý video không có lyric nguồn thì mới cân nhắc ASR/LLM.

---

## 6. File thay đổi

- **Mới:** `src/lyra_ocr/lyric/reference.py`, `tests/unit/test_lyric_reference.py`, report này.
- `src/lyra_ocr/lyric/build.py`: options reference, gate relief, trả `video_path`/`reference_*`.
- `src/lyra_ocr/lyric/select.py`: nhận `reference_lines`, bỏ quality gate khi khớp reference.
- `src/lyra_ocr/lyric/export.py`: ghi `reference_source`, `n_reference_matches` vào cues JSON.
- `src/lyra_ocr/web/lrc_api.py`: endpoint media (Range), đọc reference file, trả video path.
- `src/lyra_ocr/web/server.py`: route `/api/media`.
- `src/lyra_ocr/web/static/index.html`, `js/lrc_studio.js`, `css/app.css`: UI side-by-side + load LRC + reference.

---

## 7. Bước tiếp theo (đề xuất)

1. Thêm nút “Use video from artifact” để chọn video thủ công khi manifest thiếu `video_path`.
2. Lưu reference lyrics đã fetch vào artifact để lần sau offline vẫn dùng lại được.
3. Đánh giá định lượng text quality trước/sau reference bằng `scripts/eval_lrc_text.py` trên bộ có ground truth.
4. Nếu cần xử lý cả video không có lyric nguồn: thêm ASR (Whisper) như một nguồn reference thứ ba, vẫn giữ timing OCR.
