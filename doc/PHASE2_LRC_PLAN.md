# LYRA OCR — PHASE 2 PLAN HỢP NHẤT (LRC EXPORT + EVAL)

> Spec duy nhất cho agent implement Phase 2. Không cần ngữ cảnh hội thoại trước.
> Repo root: `/home/danglong/projects/lyra-ocr`.
> Trạng thái tài liệu: implementation plan (chưa code).

---

## 1. Bối cảnh & mục tiêu

### Hiện trạng
- Phase 1 đã xong: probe → sample frames → OCR → tracking → classification → `tracks.json`.
- Có 2 tracker: `greedy`, `bytetrack`; chất lượng track-level F1 ~96%.
- **Chưa có LRC export.** Đây là mục tiêu chính.
- Metric track-level hiện tại có vấn đề: silver GT sinh từ chính tracks của run (`scripts/auto_ground_truth.py:319`) → circular, không phản ánh đúng chất lượng LRC.

### Mục tiêu
1. **M1 (ưu tiên tuyệt đối):** có lệnh `lyra-ocr lrc <video> -o song.lrc` xuất LRC hoàn chỉnh từ artifact Phase 1, càng sớm càng tốt.
2. **M2:** chạy end-to-end từ video nếu chưa có artifact; batch export cho cả run/dataset.
3. **M3:** refine timing bằng dense frame scan.
4. **Song song/sau:** xây eval LRC thật (visual timing + text selection), không chặn M1.

### Không làm trong M1
- Không dùng LRCLIB/YouTube description lúc runtime.
- Không thêm dependency mới.
- Không sửa classifier/tracker Phase 1.
- Không làm audio alignment/forced alignment.
- Không annotate tay gold LRC.

---

## 2. Quyết định đã chốt (không hỏi lại)

| Chủ đề | Quyết định |
|---|---|
| Nguồn LRC runtime | 100% từ video OCR |
| Lyrics ngoài | Chỉ dùng **offline eval** (LRCLIB plain lyrics), không dùng production |
| Correction | Dictionary + NLP thuần, **không LLM** |
| Timing target | Báo cáo **2 kênh riêng**: visual (on-screen) + audio |
| Audio timing v0 | `not_measured` (chưa có forced alignment), có interface chờ |
| Manual gold | Không; bù bằng synthetic + dual-detector consensus + proxy |
| Timing tolerance | Mục tiêu ±300ms |
| Human review | Auto + review nhanh; `decision=review` mặc định loại khỏi LRC |
| Tracker mặc định cho LRC | `bytetrack` |
| Thứ tự ship | LRC export trước, eval sau |

---

## 3. Audit hiện trạng (bắt buộc đọc trước khi code)

### File/module liên quan
| Thành phần | Vị trí | Ghi chú |
|---|---|---|
| Pipeline Phase 1 | `src/lyra_ocr/pipeline.py` | `run_phase1_pipeline()` tạo `manifest.json`, `tracks.json`, `ocr_raw.json`, `frame_index.json`, `frames/`, `review.csv`, `labels.jsonl` |
| Tracking | `src/lyra_ocr/track.py` | `TextTrack` = one text identity (ADR 4.7); `features.dominant_text` là segment nhiều occurrence nhất |
| Classification | `src/lyra_ocr/classify.py` | `label` ∈ `lyric/non_lyric`; `decision` ∈ `accept/reject/review` |
| Export tracks | `src/lyra_ocr/review.py` | `export_tracks_json()` tạo `{total_tracks, tracks:[...]}` |
| Human labels | `src/lyra_ocr/labels.py` | `load_labels_jsonl()` trả `{(video_id, track_id): row}`, bản mới nhất thắng |
| Human labels helper | `src/lyra_ocr/review.py` | `load_human_labels(labels_path, video_id)` |
| CLI | `src/lyra_ocr/cli.py` | `main()` dùng subparsers; thêm `lrc` tại đây |
| Frame sampling | `src/lyra_ocr/video/frames.py` | `iter_sampled_frames(..., start_sec, max_frames, save_frames=False)` |
| Frames index | `frame_index.json` | Map `frame_id → image_path`, có `reused`/`dedup_of_frame` |

### Schema `tracks.json`
```json
{
  "total_tracks": 2,
  "tracks": [
    {
      "track_id": 2,
      "start_ms": 18000,
      "end_ms": 23000,
      "duration_ms": 5000,
      "bbox": [165, 151, 473, 180],
      "n_text_changes": 0,
      "features": {
        "dominant_text": "I FEEL THE NIGHT EXPLODE",
        "mean_centroid": [0.24, 0.22],
        "avg_confidence": 0.99
      },
      "segments": [
        {"segment_id": 0, "start_ms": 18000, "end_ms": 23000,
         "text": "I FEEL THE NIGHT EXPLODE", "confidence": 0.99, "num_occurrences": 10}
      ],
      "occurrences": [
        {"frame_id": 36, "timestamp_ms": 18000, "bbox": [165,151,473,180],
         "text_norm": "I FEEL THE NIGHT EXPLODE", "confidence": 0.99, "script": "latin"}
      ],
      "num_occurrences": 10,
      "label": "lyric",
      "confidence": 0.98,
      "decision": "accept",
      "qa_flags": []
    }
  ]
}
```

### Pitfalls phải xử lý
- `tracks.json` **không** có `image_path` trong occurrence. Muốn lấy frame phải qua `frame_index.json`.
- Track có thể thiếu `label`/`decision` (artifact cũ) → mặc định `non_lyric`/`review`.
- `dominant_text` có thể rỗng → fallback `segments[].text` → `occurrences[0].text_norm`.
- `decision=review` **không** đồng nghĩa lyric; mặc định loại.
- `labels.jsonl` có `human_label`, `human_decision`, `corrected_text`. `corrected_text` ưu tiên cao nhất.
- `manifest.json` có `probe.duration` (giây), `metadata.title`, `metadata.channel`, `tracker.name`.
- `frame_index.json` có thể có `reused=true`, `dedup_of_frame`; resolve đúng `image_path`.
- `pytest` mặc định bỏ test `slow`/`gpu`; unit test mới không được cần ffmpeg/OCR.
- Không sửa schema `tracks.json`, `manifest.json`, `ocr_raw.json`.

---

## 4. Kiến trúc đích & file map

### Tạo mới
```text
src/lyra_ocr/lyric/__init__.py
src/lyra_ocr/lyric/models.py
src/lyra_ocr/lyric/correct.py
src/lyra_ocr/lyric/select.py
src/lyra_ocr/lyric/grouping.py
src/lyra_ocr/lyric/timing.py          # M3, có thể stub ở M1
src/lyra_ocr/lyric/export.py
src/lyra_ocr/lyric/build.py
src/lyra_ocr/eval/visual_timing.py    # eval track
src/lyra_ocr/eval/lrc_eval.py         # eval track
src/lyra_ocr/eval/reference.py        # eval track
src/lyra_ocr/eval/audio_timing.py     # stub interface
scripts/export_lrc.py                 # batch export M2
scripts/eval_lrc.py                   # eval CLI
tests/unit/test_lrc_export.py
tests/unit/test_lrc_select.py
tests/unit/test_lrc_correct.py
tests/unit/test_lrc_grouping.py
tests/unit/test_lrc_build.py
tests/eval/test_lrc_eval.py
tests/eval/test_synthetic_noise.py
```

### Sửa
```text
src/lyra_ocr/cli.py                   # thêm subcommand lrc
README.md                             # usage
```

Không sửa `pipeline.py`, `classify.py`, `track.py` trong M1.

---

## 5. Data models (`src/lyra_ocr/lyric/models.py`)

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

@dataclass
class CorrectionInfo:
    method: str = "none"          # none|consensus|dictionary|human
    changed: bool = False
    confidence: float = 0.0
    diff: str = ""

@dataclass
class LyricCue:
    start_ms: int
    end_ms: int
    text: str
    raw_text: str = ""
    source_track_ids: list[int] = field(default_factory=list)
    source_segment_ids: list[int] = field(default_factory=list)
    confidence: float = 0.0
    decision: str = "accept"
    review_required: bool = False
    bbox: list[int] = field(default_factory=list)
    x_center: float = 0.0
    y_center: float = 0.0
    correction: CorrectionInfo = field(default_factory=CorrectionInfo)

    def to_dict(self) -> dict[str, Any]: ...

@dataclass
class LrcMetadata:
    title: str = ""
    artist: str = ""
    album: str = ""
    by: str = "Lyra OCR"
    offset_ms: int = 0

@dataclass
class CorrectionConfig:
    method: str = "dictionary"    # none|consensus|dictionary
    max_edit_ratio: float = 0.20
    min_token_len: int = 3
    min_vocab_freq: int = 2
```

`to_dict()` phải serialize cả `correction`.

---

## 6. Module implementation spec

### 6.1 `select.py`

```python
def load_artifacts(artifacts_dir: str | Path) -> tuple[dict, list[dict], dict]:
    """Return (manifest, tracks, human_labels) với human_labels keyed by str(track_id)."""

def select_cues(
    tracks: list[dict],
    human_labels: dict[str, dict],
    *,
    include_review: bool = False,
    video_id: str = "",
    correct_method: str = "dictionary",
    frame_height: int = 720,
) -> tuple[list[LyricCue], list[dict]]:
    """Return (accepted_cues, skipped_records)."""
```

Với từng track:
1. `tid = str(track.get("track_id"))`; `h = human_labels.get(tid) or {}`.
2. Xác định label/decision:
   - Nếu `h.get("human_label")`: `label = h["human_label"]`; `decision = h.get("human_decision") or ("accept" if label=="lyric" else "reject")`.
   - Ngược lại: `label = track.get("label", "non_lyric")`; `decision = track.get("decision", "review")`.
3. Điều kiện nhận:
   - `label == "lyric"` và (`decision == "accept"` hoặc (`include_review` và `decision == "review"`)).
   - Nếu `h.get("tracking_error")` → loại.
4. Text:
   - `human_text = (h.get("corrected_text") or "").strip()`.
   - Nếu có → dùng, `correction.method="human"`.
   - Nếu không → `correct_text(track, all_tracks, config)`.
   - Rỗng sau fallback → skipped `empty_text`.
5. `start = int(track.get("start_ms", 0))`; `end = int(track.get("end_ms", start))`; nếu `end <= start`: `end = start + 300`.
6. `bbox` từ track; `mean_centroid` từ `features`; fallback tính từ bbox.
   - `y_center = mean_centroid[1]` hoặc `(bbox[1]+bbox[3])/2 / frame_height`.
7. Tạo `LyricCue`; `review_required = decision == "review"`.
8. Skipped record dạng:
```json
{"track_id": 1, "reason": "non_lyric", "label": "non_lyric", "decision": "reject", "text": "Vibelyusic"}
```

### 6.2 `correct.py`

```python
def correct_text(
    track: dict,
    all_tracks: list[dict],
    config: CorrectionConfig,
) -> tuple[str, str, CorrectionInfo]:
    """Return (final_text, raw_text, correction_info)."""
```

Bước:
1. Thu variants: `occurrences[].text_norm`, `segments[].text`, `features.dominant_text`.
   - Bỏ rỗng; `unicodedata.normalize("NFC", s)`; trim; collapse whitespace.
2. `method == "none"`: chọn variant confidence cao nhất.
3. **Consensus**:
   - Cluster với `string_similarity >= 0.85` (dùng `lyra_ocr.track.string_similarity`).
   - Cluster score = `sum(conf)` hoặc `count * avg_conf`.
   - Chọn cluster điểm cao nhất; medoid = variant có tổng similarity cao nhất, tie-break confidence.
4. **Dictionary (corpus-derived)** nếu `method == "dictionary"`:
   - Build vocab token frequency từ text của tất cả tracks.
   - Chỉ áp dụng cho text Latin; skip CJK/Japanese.
   - Tokenize `\w+`, giữ mapping vị trí.
   - Token giữ nguyên nếu:
     - `len(token) < min_token_len`, hoặc
     - `vocab[token.lower()] >= min_vocab_freq`, hoặc
     - token chứa số/ký tự đặc biệt.
   - Candidate: token trong vocab cùng case pattern, edit distance ≤ `max(1, int(len*max_edit_ratio))`.
   - Chỉ sửa nếu candidate duy nhất và `freq(candidate) >= 3 * max(1, freq(token))`.
   - Giữ casing: upper/capitalize theo token gốc.
5. Cleanup: collapse spaces, trim; không thêm/bớt dấu câu; không đổi ngôn ngữ.
6. Nếu final rỗng → fallback medoid.
7. `CorrectionInfo(changed=final != medoid, method=..., confidence=...)`.

**Nguyên tắc:** conservative. Correction sai tệ hơn OCR. Nếu nghi ngờ, giữ nguyên.

### 6.3 `grouping.py`

```python
def dedupe_and_group(cues: list[LyricCue]) -> list[LyricCue]:
```

1. Sort `(start_ms, y_center, track_id)`.
2. Dedupe nếu:
   - `string_similarity(text_a, text_b) >= 0.90`, và
   - `abs(start_a - start_b) <= 500`, và
   - `abs(y_a - y_b) <= 0.08` hoặc bbox IoU >= 0.3.
   - Giữ cue confidence cao hơn; nếu text dài hơn đáng kể thì ưu tiên dài hơn.
   - Gộp `source_track_ids`.
3. Không gộp text khác nhau cùng thời gian (stanza/translation) → giữ riêng, cùng `start_ms`.
4. Clamp `start_ms >= 0`.

### 6.4 `timing.py` (M3, optional)

```python
def refine_timings(
    cues: list[LyricCue],
    video_path: str,
    manifest: dict,
    artifacts_dir: str,
    *,
    fps: float = 25.0,
    window_ms: int = 1000,
    enabled: bool = True,
) -> list[LyricCue]:
```

M1: nếu `enabled=False`, thiếu `frame_index.json`/`frames/` → trả nguyên cues, log warning.

M3:
1. Load `frame_index.json`, map `frame_id → image_path`.
2. Với mỗi cue:
   - Lấy occurrence confidence cao nhất; lấy `frame_id`.
   - Load frame, crop bbox mở rộng 10% → template grayscale.
3. `iter_sampled_frames(video_path, artifacts_dir/"refine_tmp", fps=25, start_sec=(start-window)/1000, max_frames=ceil(2*window/1000*fps), save_frames=False)`.
4. Mỗi frame: crop vùng bbox, `cv2.matchTemplate(..., TM_CCOEFF_NORMED)`, lấy max score.
5. `threshold = max(0.70, 0.90 * max_score)`.
6. `visual_start` = timestamp frame đầu score ≥ threshold; `visual_end` = timestamp frame cuối + frame_interval.
7. Nếu tìm thấy: `start_ms = visual_start`; `end_ms = max(visual_start+300, visual_end)`.
8. Không tìm thấy hoặc template quá nhỏ → giữ coarse.
9. Không dùng OCR trong refine M3 (tránh nặng và tương quan eval).

### 6.5 `export.py`

```python
def format_lrc_timestamp(ms: int) -> str:
    cs = max(0, int(round(ms / 10.0)))
    minutes = cs // 6000
    rem = cs % 6000
    seconds = rem // 100
    centi = rem % 100
    return f"[{minutes:02d}:{seconds:02d}.{centi:02d}]"

def write_lrc(
    out_path: str | Path,
    cues: list[LyricCue],
    metadata: LrcMetadata,
) -> None: ...

def write_cues_json(
    out_path: str | Path,
    cues: list[LyricCue],
    metadata: LrcMetadata,
    *,
    video_id: str,
    video_path: str,
    duration_ms: int,
    tracker: str,
    skipped: list[dict],
) -> None: ...
```

Quy tắc LRC:
- Header chỉ tag non-empty:
```text
[ti:...]
[ar:...]
[al:...]
[by:Lyra OCR]
[offset:0]
```
- Mỗi cue: `f"{format_lrc_timestamp(start_ms)}{text}"`.
- Text: replace `\r`, `\n` bằng space; strip; bỏ cue rỗng.
- Sort `(start_ms, y_center)`.
- Dedupe exact `(start_cs, text)`.
- Timestamp không giảm: nếu cue sau start < cue trước → clamp bằng start trước.
- UTF-8, không BOM.
- `end_ms` chỉ lưu `lrc_cues.json`, không bắt buộc trong LRC chuẩn.
- Sanitize `[`/`]` trong tag values (thay bằng space) để không phá format.

Ví dụ:
```text
[ti:Tell It to My Heart]
[ar:Taylor Dayne]
[by:Lyra OCR]
[offset:0]
[00:18.00]I FEEL THE NIGHT EXPLODE
[00:18.00]WHEN WE'RE TOGETHER
[00:26.00]EMOTION OVERLOAD
```

### 6.6 `build.py`

```python
def build_lrc_from_artifacts(
    artifacts_dir: str | Path,
    out_lrc: str | Path,
    *,
    include_review: bool = False,
    correct_method: str = "dictionary",
    refine: bool = False,
    offset_ms: int = 0,
    title: str | None = None,
    artist: str | None = None,
    force: bool = False,
) -> dict:
```

Trình tự:
1. `manifest, tracks, human_labels = load_artifacts(artifacts_dir)`.
2. Không có tracks → `ValueError` hướng dẫn chạy pipeline.
3. `video_id = manifest.get("video_id") or ...`.
4. Metadata:
   - `title or manifest.metadata.title or ""`.
   - `artist or manifest.metadata.channel or ""`.
   - `offset_ms`.
5. `cues, skipped = select_cues(...)`.
6. `if refine: cues = refine_timings(...)` bọc try/except → fallback coarse.
7. `cues = dedupe_and_group(cues)`.
8. `write_lrc(out_lrc, cues, metadata)`.
9. `write_cues_json(out_lrc.with_suffix(".cues.json"), ...)`.
10. Return:
```python
{"video_id": ..., "out_lrc": str(...), "n_cues": len(cues),
 "n_skipped": len(skipped), "review_required": sum(c.review_required for c in cues),
 "skipped": skipped}
```

```python
def build_lrc_from_video(
    video_path: str | Path,
    out_lrc: str | Path,
    *,
    artifacts_dir: str | Path | None = None,
    tracker: str = "bytetrack",
    fps: float = 2.0,
    title: str = "",
    channel: str = "",
    device: str = "auto",
    rec_batch_size: int = 8,
    dedup: bool = False,
    frame_source: str = "pipe",
    force: bool = False,
    **build_kwargs,
) -> dict:
```

- `artifacts_dir` mặc định `./lyra_output/<video_stem>/`.
- Nếu `force` hoặc thiếu `manifest.json`/`tracks.json`: gọi `run_phase1_pipeline(...)`.
- Sau đó gọi `build_lrc_from_artifacts`.
- Không OCR lại nếu artifact đã đủ.

### 6.7 CLI (`src/lyra_ocr/cli.py`)

Thêm subparser:
```python
p_lrc = subparsers.add_parser("lrc", help="Export synchronized .lrc from a video")
p_lrc.add_argument("video")
p_lrc.add_argument("-o", "--out", default="./lyrics.lrc")
p_lrc.add_argument("--artifacts", default=None)
p_lrc.add_argument("--tracker", choices=["greedy", "bytetrack"], default="bytetrack")
p_lrc.add_argument("--fps", type=float, default=2.0)
p_lrc.add_argument("--start-sec", type=float, default=0.0)
p_lrc.add_argument("--max-frames", type=int, default=None)
p_lrc.add_argument("--title", default="")
p_lrc.add_argument("--channel", default="")
p_lrc.add_argument("--device", choices=["auto","cpu","cuda"], default="auto")
p_lrc.add_argument("--rec-batch-size", type=int, default=8)
p_lrc.add_argument("--dedup", action=argparse.BooleanOptionalAction, default=False)
p_lrc.add_argument("--frame-source", choices=["pipe","disk"], default="pipe")
p_lrc.add_argument("--refine", action=argparse.BooleanOptionalAction, default=False)
p_lrc.add_argument("--correct", choices=["none","consensus","dictionary"], default="dictionary")
p_lrc.add_argument("--include-review", action="store_true")
p_lrc.add_argument("--offset-ms", type=int, default=0)
p_lrc.add_argument("--force", action="store_true")
```

Dispatch:
```python
elif args.command == "lrc":
    return run_lrc(args)
```

`run_lrc`:
- Gọi `build_lrc_from_video(...)`.
- In:
```text
Wrote 57 cues to song.lrc
Skipped 12 tracks (non_lyric=10, review=2)
Review required: 3
```
- 0 cue → warning, return `2`; lỗi khác → `1`.
- Nên refactor `_add_phase1_args(p, default_tracker)` để tránh lặp; không bắt buộc nếu rủi ro.

### 6.8 Batch script (`scripts/export_lrc.py`)

```bash
python scripts/export_lrc.py \
  --dataset Fix_lyric_EN \
  --run-id bytetrack-1794243340-de75c2 \
  --out-dir ./lrc_out \
  --include-review --correct dictionary
```

- Đọc `<dataset>/runs/<run_id>/run.json`.
- Với mỗi video entry: resolve `artifact_dir`, gọi `build_lrc_from_artifacts`, ghi `<out-dir>/<video_id>.lrc`.
- Hỗ trợ `--limit`, `--force`.
- In summary số video, số cue, lỗi.

---

## 7. Evaluation spec (track riêng, không chặn M1)

### 7.1 Nguyên tắc
- Không dùng LRCLIB làm oracle tuyệt đối.
- Visual timing là primary, tự đo từ video.
- Text/selection dùng LRCLIB **plain lyrics offline** làm inventory.
- Audio timing: `not_measured` v0, có interface.
- Không manual gold; bù bằng synthetic + dual detector + proxy.

### 7.2 `src/lyra_ocr/eval/visual_timing.py`

```python
def measure_visual_timing(
    video_path: str,
    cues: list[LyricCue],
    artifacts_dir: str,
    *,
    fps: float = 25.0,
    window_ms: int = 1000,
) -> dict:
```

Với mỗi cue:
- Lấy template từ occurrence confidence cao nhất qua `frame_index.json`.
- Scan dày 25–30 FPS quanh `start/end`.
- 2 detector độc lập:
  - D1: template matching NCC.
  - D2: OCR trên crop + text similarity (dùng `OCREngine`, optional vì nặng).
- Consensus → `appear_ms`, `disappear_ms`, `oracle_confidence`.
- Metrics: signed bias, MAE, median, p90, `% ≤100ms`, `% ≤300ms`, end error.
- Cache theo `(video_id, track_id)`.

### 7.3 `src/lyra_ocr/eval/reference.py`

```python
def load_plain_lyrics(dataset_dir: str | Path, video_id: str) -> list[str]:
    """Đọc lyrics cache offline; nếu thiếu có thể fetch LRCLIB nhưng chỉ trong eval."""

def load_synced_lrc(path: str | Path) -> list[tuple[int, str]]:
    """Parse [mm:ss.xx]text, expand multi-tag lines."""
```

- Plain lyrics: list lines, dùng `datasets/<ds>/lyrics_cache/<video_id>.json` (đã có dạng list).
- Synced LRC: optional, chỉ dùng supplementary.

### 7.4 `src/lyra_ocr/eval/lrc_eval.py`

```python
def evaluate_text_selection(
    cues: list[LyricCue],
    reference_lines: list[str],
    *,
    match_threshold: float = 0.60,
) -> dict:
```

- Normalize: lowercase, bỏ punctuation, collapse spaces.
- Match generated ↔ reference bằng similarity matrix + Hungarian (scipy đã có).
- Metrics:
  - `line_recall = matched_ref / len(ref_lines)`
  - `line_precision = matched_gen / len(gen_lines)`
  - `line_f1`
  - CER/WER trên matched pairs (Levenshtein).
  - `extra_line_rate`.
- Version gate:
  - Nếu `line_recall < 0.5` và `line_precision < 0.5` → `scorable=false` (khác version).
  - Không đưa video unscorable vào aggregate.

```python
def evaluate_synced_reference(
    cues: list[LyricCue],
    reference_lrc: list[tuple[int, str]],
    *,
    min_match_rate: float = 0.6,
) -> dict:
```

- Text-first match, bỏ timestamp.
- Nếu match rate < `min_match_rate` → `scorable=false`.
- Nếu đủ: fit robust affine `t_gen ≈ a * t_ref + b` bằng `scipy.stats.theilslopes` hoặc RANSAC.
- Report riêng:
  - `offset_ms = b`
  - `scale = a`
  - `raw_error`, `residual_after_align`
  - `scorable`
- Không dùng làm gate.

### 7.5 `src/lyra_ocr/eval/audio_timing.py`

```python
from typing import Protocol

class AudioTimingProvider(Protocol):
    def align(self, video_path: str, reference_lyrics: list[str]) -> list[tuple[int, str]]: ...

def get_audio_provider() -> AudioTimingProvider | None:
    return None  # v0
```

Report:
```json
"timing_audio": {"status": "not_measured", "reason": "no eval-only forced-alignment provider"}
```

### 7.6 Synthetic benchmark

- Sinh text sạch → nhiễu OCR:
  - char confusion: `0/O`, `1/l/I`, `5/S`, `8/B`, `rn/m`, `vv/w`, `cl/d`.
  - deletion 1–10%, insertion 1–5%, substitution 1–5%.
  - merge/split spaces, random casing.
  - seed cố định.
- Đo CER/WER trước/sau correction.
- `hallucination_rate` = ký tự bị sửa trên text sạch (không nhiễu) / tổng ký tự.
- Test video synthetic bằng ffmpeg `drawtext` + LRC biết trước → assert export/timing chính xác.

### 7.7 Eval output & gates

Per-video JSON:
```json
{
  "video_id": "...",
  "metric_kind": "automatic_proxy",
  "scorable": true,
  "timing_visual": {
    "median_abs_error_ms": 120,
    "p90_abs_error_ms": 280,
    "percent_within_100ms": 0.42,
    "percent_within_300ms": 0.93,
    "oracle_confidence_mean": 0.88
  },
  "timing_reference": {
    "status": "ok",
    "kind": "lrclib_synced_weak",
    "scorable": true,
    "offset_ms": 1200,
    "scale": 1.01,
    "residual_median_ms": 180
  },
  "timing_audio": {"status": "not_measured"},
  "text_selection": {
    "line_recall": 0.91,
    "line_precision": 0.97,
    "line_f1": 0.94,
    "cer": 0.05,
    "wer": 0.08,
    "extra_line_rate": 0.02
  },
  "warnings": []
}
```

Gate v0 (chỉ dùng visual + text, không dùng LRCLIB timing):
- `% cue start lệch ≤300ms` (high-confidence) ≥ 90%.
- median |start error| ≤ 150ms.
- Selection recall (video scorable) ≥ 90%.
- Extra-line ≤ 2%.
- CER ≤ 8%.
- Correction synthetic: giảm ≥50% CER; hallucination ≤1% trên dòng sạch.
- Audio: chỉ report, không gate.

---

## 8. Tests

### Unit LRC
- `test_lrc_export.py`
  - `format_lrc_timestamp(0) == "[00:00.00]"`.
  - `format_lrc_timestamp(61234) == "[01:01.23]"`.
  - `format_lrc_timestamp(3600000) == "[60:00.00]"`.
  - Metadata chỉ tag non-empty.
  - Newline trong text → space.
  - Sort + dedupe + timestamp không giảm.
- `test_lrc_select.py`
  - lyric/accept → cue.
  - non_lyric/reject → skipped.
  - lyric/review → skipped mặc định; included với `include_review`.
  - human `corrected_text` thắng machine.
  - human `non_lyric` override machine lyric.
  - `tracking_error=True` → skip.
  - missing label/decision → skip an toàn.
- `test_lrc_correct.py`
  - Consensus majority.
  - NFC normalization.
  - Human corrected_text ưu tiên.
  - Dictionary không sửa proper noun hiếm.
  - Sửa token khi candidate vượt trội tần suất; giữ casing.
  - `method="none"` giữ confidence cao nhất.
- `test_lrc_grouping.py`
  - Duplicate track 200ms → dedupe.
  - Text khác, cùng start, y khác → giữ cả 2.
  - start âm → clamp 0.
- `test_lrc_build.py`
  - Temp dir với `manifest.json`, `tracks.json`, `labels.jsonl` mini.
  - Gọi `build_lrc_from_artifacts`; assert `.lrc` + `.cues.json`.
  - Không cần ffmpeg/OCR.

### Eval tests
- `test_lrc_eval.py`: text match, version gate, metrics.
- `test_synthetic_noise.py`: CER improvement + hallucination.
- Visual timing: mock frames + template (numpy), assert boundaries.

Chạy:
```bash
pytest tests/unit/test_lrc_*.py -q
pytest tests/eval/test_lrc_eval.py tests/eval/test_synthetic_noise.py -q
```

---

## 9. Manual QA

```bash
lyra-ocr lrc "datasets/Fix_lyric_EN/videos/<file>.mp4" \
  -o /tmp/test.lrc --tracker bytetrack --device cpu --force
```

Kiểm tra:
- File tồn tại, UTF-8, parse được.
- Timestamp tăng dần.
- Không có watermark/channel/title card.
- Số dòng xấp xỉ số câu lyric reference (lệch do version là bình thường).
- Playback:
```bash
mpv --sub-file=/tmp/test.lrc "datasets/Fix_lyric_EN/videos/<file>.mp4"
```
- Chạy 3 video khác thể loại; ghi lỗi: dòng rác, thiếu, timing lệch, text sai.
- `pytest` toàn bộ (không slow/gpu) pass.

---

## 10. Milestones & thứ tự implement

### M1 — Fast ship (ưu tiên tuyệt đối)
1. `models.py` + `export.py` + test export.
2. `select.py` + `correct.py` + tests.
3. `grouping.py` + tests.
4. `build.py::build_lrc_from_artifacts` + test build.
5. CLI `lrc` chỉ đọc artifact sẵn có; thiếu thì báo lỗi rõ.
6. README usage.
→ Đã xuất được LRC từ artifact.

### M2 — End-to-end từ video
7. `build_lrc_from_video`: chạy `run_phase1_pipeline` khi thiếu/`--force`.
8. `scripts/export_lrc.py` batch.
9. Manual QA 3–5 video.

### M3 — Timing refine
10. `timing.py` template matching dense scan.
11. CLI `--refine`.
12. Tests mock.

### M4 — Eval LRC
13. `visual_timing.py` + `lrc_eval.py` + `reference.py`.
14. `scripts/eval_lrc.py`.
15. Synthetic benchmark.
16. Audio stub.

### M5 — Correction nâng cao
17. Corpus vocab + edit-distance rules.
18. Optional TTA (không bắt buộc).

---

## 11. Acceptance criteria

### M1
- LRC hợp lệ, có ít nhất 1 cue cho video có lyric.
- Không crash khi artifact thiếu field.
- Unit tests pass.
- Thời gian chạy không tăng đáng kể (refine off).

### M2
- `lyra-ocr lrc video.mp4 -o song.lrc` chạy end-to-end.
- Artifact tái sử dụng; `--force` rerun đúng.
- Batch script xuất được nhiều video.

### M3+
- Visual timing gates: ≥90% cue ≤300ms; median ≤150ms.
- Text selection: recall ≥90%, extra ≤2%, CER ≤8% trên video scorable.
- Audio: report `not_measured`.

---

## 12. Pitfalls / Do-not

- **Không** dùng `features.dominant_text` một mình; có thể rỗng hoặc lỗi.
- **Không** sửa text quá tay; correction sai tệ hơn OCR.
- **Không** đưa `decision=review` vào LRC mặc định.
- **Không** dùng `end_ms` làm thời điểm kết thúc LRC chuẩn; chỉ dùng start.
- **Không** format timestamp bằng float trực tiếp; dùng centisecond integer.
- **Không** đảo thứ tự dòng cùng timestamp; sort theo `y_center`.
- **Không** crash nếu `frame_index.json`/`frames/` thiếu; fallback coarse.
- **Không** thêm dependency mới ở M1.
- **Không** sửa classifier để ép track thành lyric; xử lý ở select.
- **Không** ghi đè `tracks.json`/`labels.jsonl`.
- **Không** dùng LRCLIB/YouTube description lúc runtime.
- `labels.jsonl` có thể nhiều bản ghi cùng track; loader lấy bản mới nhất.
- Track id trong labels là string; luôn `str(track_id)`.
- Giữ Unicode NFC; không lower/upper toàn bộ text.
- `--include-review` → set `review_required=true` trong `lrc_cues.json`.

---

## 13. Fixtures mẫu cho `test_lrc_build`

`manifest.json`:
```json
{
  "video_id": "vid1",
  "video_path": "/tmp/vid1.mp4",
  "metadata": {"title": "Song", "channel": "Artist"},
  "probe": {"duration": 120.0, "width": 1280, "height": 720},
  "config": {"sample_fps": 2.0},
  "tracker": {"name": "bytetrack"}
}
```

`tracks.json`:
```json
{
  "total_tracks": 3,
  "tracks": [
    {"track_id": 1, "start_ms": 0, "end_ms": 120000, "bbox": [10,10,100,40],
     "features": {"dominant_text": "Vibelyusic", "mean_centroid": [0.03,0.03]},
     "segments": [], "occurrences": [], "label": "non_lyric", "decision": "reject"},
    {"track_id": 2, "start_ms": 18000, "end_ms": 23000, "bbox": [165,151,473,180],
     "features": {"dominant_text": "I FEEL THE NIGHT EXPLODE", "mean_centroid": [0.24,0.22]},
     "segments": [{"segment_id":0,"start_ms":18000,"end_ms":23000,
       "text":"I FEEL THE NIGHT EXPLODE","confidence":0.99,"num_occurrences":10}],
     "occurrences": [], "label": "lyric", "confidence": 0.98, "decision": "accept"},
    {"track_id": 3, "start_ms": 26000, "end_ms": 32000, "bbox": [180,179,458,206],
     "features": {"dominant_text": "WHEN WE'RE TOGETHER", "mean_centroid": [0.25,0.26]},
     "segments": [{"segment_id":0,"start_ms":26000,"end_ms":32000,
       "text":"WHEN WE'RE TOGETHER","confidence":0.97,"num_occurrences":8}],
     "occurrences": [], "label": "lyric", "confidence": 0.95, "decision": "accept"}
  ]
}
```

Expected LRC:
```text
[ti:Song]
[ar:Artist]
[by:Lyra OCR]
[offset:0]
[00:18.00]I FEEL THE NIGHT EXPLODE
[00:26.00]WHEN WE'RE TOGETHER
```

---

## 14. Out of scope / future

- Audio forced alignment (WhisperX/aeneas/MFA) — chỉ khi duyệt eval-only dependency.
- LLM correction — đã loại.
- Manual gold LRC — đã loại.
- Karaoke word-level timing — Phase 4.
- Web UI export button — sau M2.
- Tối ưu tracker/classifier — chỉ khi eval LRC chỉ ra lỗi cụ thể.

---

**Kết thúc plan.** Agent implement nên bắt đầu từ M1 mục 1→6, chạy test sau mỗi bước, rồi mới sang M2/M3.
