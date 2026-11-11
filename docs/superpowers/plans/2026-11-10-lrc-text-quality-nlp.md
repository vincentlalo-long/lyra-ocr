# Phase 4 — Reference-first LRC Text Repair (Single-Lyric)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers/subagent-driven-development (recommended) or superpowers/executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Status:** Ready for implementation.

**Goal:** Improve runtime `.lrc` text quality on single-lyric videos by using trusted reference lyrics to repair OCR cue text and recover missing lyric lines, while preserving visual OCR timing for cues that were already detected.

**Scope:** Single-lyric videos only — at most one active lyric line at any timestamp. Multi-line/simultaneous lyrics, karaoke word timing, and audio forced-alignment remain deferred.

**Architecture:** Add a reference-source layer with two providers: LRCLIB first, YouTube captions as fallback. Parse reference lines with timestamps when available, align them monotonically against OCR cues, replace whole cue text only when similarity and coverage pass a source-aware threshold, and fill unambiguous missing lines only from synchronized references. Cache every provider result and expose provenance in `.cues.json`.

**Replaces:** The Phase 3 rule “reference lyrics are evaluation-only” is superseded. Reference lyrics may now repair LRC text and fill missing lines, but timing for detected cues remains OCR-derived. Multi-line reference assignment is still out of scope.

**Tech Stack:** Python 3.11+, pytest, RapidOCR, `yt-dlp`, `symspellpy`, existing `tracks.json` / `cues.json` schemas.

---

## 1. Evidence from the current implementation

### 1.1 Existing artifacts

- Attention sample: 51 accepted cues, 37 skipped, 360p/1 fps artifacts, `reference_source: null`.
- Railgun sample: 162 skipped, multilingual JP/EN/ES, overlapping/duplicated cues, 360p/1 fps artifacts.
- Both samples still show missing line starts, word concatenation, and false-positive/negative classification:
  - a real lyric line was classified as `non_lyric` and skipped;
  - `Presents`, `SOUND`, and `ONLY` were accepted as lyric fragments;
  - multiple cues share the same start or end too early.

### 1.2 Reference availability (verified)

- `yttool` is not installed or present in this repo. The available tool is `yt-dlp` (`yt_dlp 2026.10.26`).
- Attention has YouTube automatic captions in `en` and `en-orig`. They contain timestamps but are ASR text with rolling duplicates and recognition errors.
- Railgun has no YouTube captions or subtitles.
- LRCLIB **does** return synchronized lyrics for both samples:
  - Attention: multiple candidates, including a 213 s version close to the 212 s video.
  - Railgun `sister's noise` by `fripSide`: synchronized Japanese lyrics available.
- Railgun currently fails reference lookup because `build_lrc_from_artifacts()` passes the full YouTube title as the track name and leaves the artist empty. The full title contains `『Lyrics AMV』`, bracketed show names, and both track/artist in one string.

### 1.3 Current code limitations

- `src/lyra_ocr/lyric/reference.py`
  - `parse_lyrics_text()` strips all LRC timestamps, so synced references cannot recover missing cues.
  - `align_reference()` only accepts a list of plain lines, uses `min_similarity=0.55`, and always replaces matched cue text.
  - Matching ignores token coverage, length ratio, source trust, and monotonic gap penalties.
  - No provider fallback to YouTube captions.
- `src/lyra_ocr/lyric/build.py`
  - Reference lookup is opt-in and uses unparsed manifest metadata.
  - Reference cache key has only `manual` / `lrclib` / `off`, not the resolved provider or query.
  - Reference alignment runs after dedupe/grouping, so gap-filled lines would not be deduped.
- `src/lyra_ocr/lyric/select.py`
  - A reference match at `0.55` can bypass the text-quality gate.
  - Reference evidence cannot rescue a track already labelled `non_lyric`.
- `scripts/auto_ground_truth.py`
  - Fetches YouTube description and LRCLIB plain lyrics only, never captions.
  - Uses LRCLIB plain lyrics, so it cannot generate synchronized gap-fill references.
- `scripts/eval_lrc_text.py`
  - Reports CER/WER and lexical metrics, but not missing-line recall or per-cue replacement quality.

---

## 2. Success criteria

For the single-lyric benchmark set:

- **Existing cues:** every cue whose OCR text is 80–90% similar to a trusted reference is replaced with the complete reference sentence, including missing leading/trailing words.
- **Missing lines:** when a synchronized reference is available, lyric lines inside visual gaps are recovered without overlapping existing cues.
- **No harmful replacement:** a high-confidence OCR line is not replaced by a lower-trust caption line unless similarity is high and token coverage/length guards pass.
- **Timing:** detected cues keep OCR start/end timing. Only gap-filled cues use reference timestamps.
- **Provenance:** `.cues.json` records `reference_source`, `n_reference_matches`, `n_reference_gap_fills`, and warnings.
- **Fallback:** a failed or unavailable reference provider never blocks LRC generation.
- **Regression:** classifier false positives/negatives on the known samples are fixed or explicitly routed to `review`.
- **Tests:** focused tests and the full non-slow suite pass; all provider tests are offline/mocked.

### Threshold policy

Initial defaults, configurable:

| Source | Replace existing cue | Mark review only | Gap fill |
| --- | ---: | ---: | --- |
| manual `.lrc/.txt` | 0.85 | 0.70–0.85 | yes, if timestamps exist |
| LRCLIB synced | 0.85 | 0.70–0.85 | yes, high confidence |
| LRCLIB plain | 0.85 | 0.70–0.85 | no timestamps → no |
| YouTube captions | 0.90 | 0.75–0.90 | yes, flagged as caption-sourced |

Additional guard for replacement:

- token coverage of the reference line `>= 0.70`;
- length ratio `0.60 <= len(ref_tokens) / len(ocr_tokens) <= 1.80`;
- reference line has at least one alphabetic token (or CJK token for CJK references);
- do not replace a cue whose text is already an exact reference match.

---

## 3. Design decisions

1. **Provider priority**
   - Manual reference file.
   - LRCLIB synchronized lyrics (`syncedLyrics`).
   - LRCLIB plain lyrics (`plainLyrics`) — text repair only, no gap fill.
   - YouTube captions — fallback only when LRCLIB returns no usable lyrics.
   - No reference.

2. **Captions are fallback, not authority**
   - Use original-language captions only. Do not use auto-translated `en` captions for a non-English song.
   - Prefer `info["language"]`, then `en-orig`, then `en` only when the original language is English.
   - Caption text has lower trust and a stricter replacement threshold.

3. **Whole-sentence replacement**
   - When a match passes the threshold, replace the entire cue text with the reference line.
   - Preserve `raw_text`, OCR timing, bbox, and source track IDs.
   - Set `CorrectionInfo.method` to `reference`, `reference_caption`, or `reference_gap_fill`.

4. **Gap fill is synchronized-only**
   - Plain text can fix existing cue text but cannot invent timing.
   - Only insert a missing line when its reference interval lies in an unambiguous gap between matched cues and does not overlap an existing cue.
   - LRCLIB synced gap fills may be accepted; caption gap fills are accepted but flagged with `caption_gap_fill`.

5. **Conservative rescue**
   - A trusted LRCLIB/manual reference may rescue a `non_lyric` track only when:
     - similarity passes the replacement threshold;
     - the track has at least two words or a CJK phrase;
     - duration is at least 1.0 s;
     - the track is not a known intro/outro credit card.
   - Caption references never rescue `non_lyric` tracks.

6. **Caching**
   - Cache provider results under the artifact directory, e.g. `reference_cache/<sha1>.json`.
   - Cache key includes resolved track, artist, duration bucket, provider list, and parser version.
   - Include reference mode/config in the LRC cues cache signature.

7. **No new ML dependency in this phase**
   - Whisper/faster-whisper is deferred until after the reference path is measured.
   - No LLM contextual rewriting.

---

## 4. Data model and public API

Add to `src/lyra_ocr/lyric/reference.py` (or a small sibling module if it becomes too large):

```python
@dataclass
class ReferenceLine:
    text: str
    start_ms: int | None = None
    end_ms: int | None = None
    source: str = ""
    confidence: float = 0.0

@dataclass
class ReferenceBundle:
    lines: list[ReferenceLine]
    source: str = ""
    provider: str = ""
    language: str = ""
    warnings: list[str] = field(default_factory=list)
```

Add to `src/lyra_ocr/lyric/models.py`:

```python
@dataclass
class ReferenceConfig:
    replace_similarity: float = 0.85
    review_similarity: float = 0.70
    caption_replace_similarity: float = 0.90
    min_token_coverage: float = 0.70
    min_length_ratio: float = 0.60
    max_length_ratio: float = 1.80
    enable_gap_fill: bool = True
    enable_caption_gap_fill: bool = True
    rescue_non_lyric: bool = True
```

Public behavior:

- `build_lrc_from_artifacts(..., use_online_reference=True)` means “auto provider”: LRCLIB first, captions fallback.
- `reference_source` values: `"manual"`, `"lrclib_synced"`, `"lrclib_plain"`, `"youtube_captions"`, `""`.
- Keep existing `reference_lines: list[str]` parameters working for backwards compatibility, but route new code through `ReferenceBundle`.
- Extend result/cues JSON with:
  - `n_reference_gap_fills`;
  - `reference_warnings`;
  - per-cue `reference` metadata when a cue was changed or inserted.

---

## 5. Tasks

### Task 0: Offline fixtures and baseline metrics

**Files:**
- Create: `tests/fixtures/reference/attention_caption.vtt` (short synthetic excerpt, not full lyrics)
- Create: `tests/fixtures/reference/lrclib_synced.json` (synthetic response)
- Create: `tests/fixtures/reference/caption.json3` (synthetic events with rolling duplicates)
- Modify: `scripts/eval_lrc_text.py`
- Test: `tests/unit/test_eval_lrc_text.py`

**Implementation:**
- Add metrics for reference evaluation:
  - `missing_reference_lines` count;
  - `reference_line_recall` when timestamps are available;
  - `replaced_cue_count`;
  - `suspicious_replacement_count` (length ratio outside guard or similarity in review band).
- Keep `reference_note` but change it to “evaluation comparison; runtime provenance is recorded in cues.json”.
- Add `--reference-source`/`--reference-file` only if needed; do not fetch network in the eval script.
- Record baseline metrics for the Attention and Railgun artifacts before changing behavior.

**Acceptance:**
- Fixtures are small, synthetic/permissive, and contain no full copyrighted lyrics.
- `scripts/eval_lrc_text.py` still works on existing `.cues.json`.
- New metrics are covered by unit tests.

- [x] Add synthetic fixtures.
- [x] Extend metrics and tests.
- [x] Capture baseline JSON for both samples.

### Task 1: Parse video metadata and build a reference query

**Files:**
- Modify: `src/lyra_ocr/lyric/reference.py`
- Modify: `src/lyra_ocr/lyric/build.py`
- Test: `tests/unit/test_lyric_reference.py`

**Implementation:**
- Add a `ReferenceQuery` dataclass or tuple: `track`, `artist`, `duration_ms`, `raw_title`.
- Add `parse_reference_query(title, channel, duration_ms)`:
  - strip decorative brackets: `『』「」【】[]()`;
  - remove prefixes/suffixes such as `Lyrics AMV`, `Official Video`, `Official Lyric Video`, `MV`, `Audio`, `Lyrics`, `(Lyrics)`;
  - split on ` - ` / ` – ` / ` — ` when it yields a plausible artist/track pair;
  - remove `feat.`, `ft.`, `prod.` suffixes from the track;
  - if artist is empty, derive from channel; strip ` - Topic`, `VEVO`, and trailing channel decorations;
  - preserve the raw title for metadata output.
- In `build_lrc_from_artifacts()`:
  - keep `metadata.title` / `metadata.artist` as the user-visible metadata;
  - use `parse_reference_query()` for provider lookup when explicit `title`/`artist` are not supplied.
- Add tests with:
  - `Charlie Puth - Attention (Lyrics)` → track `Attention`, artist `Charlie Puth`;
  - `『Lyrics AMV』 Toaru Kagaku no Railgun S OP 1 Full 「sister's noise - fripSide」` → track `sister's noise`, artist `fripSide`;
  - `Artist - Song (Official Video)` and `Song [Lyrics]`;
  - channel fallback `Artist - Topic`.

**Acceptance:**
- Railgun metadata resolves to `sister's noise` / `fripSide`.
- Explicit user title/artist still take precedence.
- No network calls in tests.

- [x] Add failing metadata tests.
- [x] Implement metadata parsing and wire it into build.
- [x] Run `pytest tests/unit/test_lyric_reference.py -q`.

### Task 2: Add timed reference parsing and LRCLIB provider

**Files:**
- Modify: `src/lyra_ocr/lyric/reference.py`
- Modify: `src/lyra_ocr/lyric/models.py`
- Test: `tests/unit/test_lyric_reference.py`

**Implementation:**
- Add `parse_reference_text(text, *, source="", language="") -> list[ReferenceLine]`:
  - preserve LRC start timestamps;
  - support multiple timestamps per line;
  - strip metadata tags such as `[ar:]`, `[ti:]`, `[offset:]`;
  - keep plain-text lines with no timestamps as `start_ms=None`;
  - ignore section tags like `[Chorus]` unless they contain lyric text.
- Keep `parse_lyrics_text()` as a compatibility wrapper returning `[line.text]`.
- Add `fetch_reference_bundle(query, *, timeout=...) -> ReferenceBundle`:
  - call LRCLIB search once with `track_name` + `artist_name`;
  - accept both `syncedLyrics` and `plainLyrics`;
  - score candidates by track similarity, artist similarity, and duration tolerance (use tolerance, not exact duration);
  - prefer synced over plain when scores are close;
  - return `[]`/empty bundle on any failure; never raise.
- Add `source`: `"lrclib_synced"` or `"lrclib_plain"`.
- Preserve the raw response only in cache, not in `.cues.json`.

**Tests (mock `urllib.request.urlopen`):**
- synced LRC timestamps survive parsing;
- plain response is accepted for text repair but has no timestamps;
- duration mismatch within tolerance still selects the correct candidate;
- HTTP error returns an empty bundle;
- multiple timestamps on one line produce multiple `ReferenceLine`s with the same text.

**Acceptance:**
- Attention returns a synced candidate close to 212 s.
- Railgun returns a synced Japanese candidate after Task 1 metadata parsing.

- [x] Add failing parser/provider tests.
- [x] Implement `ReferenceLine` / `ReferenceBundle` / LRCLIB bundle fetch.
- [x] Run `pytest tests/unit/test_lyric_reference.py -q`.

### Task 3: Add provider cache

**Files:**
- Modify: `src/lyra_ocr/lyric/reference.py`
- Modify: `src/lyra_ocr/lyric/build.py`
- Test: `tests/unit/test_lyric_reference.py`, `tests/unit/test_lrc_build.py`

**Implementation:**
- Add `reference_cache_dir(artifacts_dir) -> Path` using `artifacts_dir / "reference_cache"`.
- Add `cache_key(query, providers, parser_version)` as SHA-1 of normalized JSON.
- Save/load `ReferenceBundle` as JSON with:
  - `bundle_version`;
  - resolved query;
  - provider/source/language/warnings;
  - lines including timestamps and confidence.
- Cache LRCLIB and caption results separately so a caption fallback can be reused without repeating `yt-dlp`.
- Do not use a stale cache when the resolved track/artist/duration bucket changes.
- Include reference mode/config/cache key in `build_lrc_from_artifacts()` target options so `.cues.json` is invalidated when the reference source changes.

**Acceptance:**
- Second call with the same key performs no network request (mock assertion).
- Changing artist or duration bucket misses the cache.
- Cache files are valid JSON and survive a process restart.

- [x] Add failing cache tests.
- [x] Implement cache read/write and cache-signature integration.
- [x] Run `pytest tests/unit/test_lyric_reference.py tests/unit/test_lrc_build.py -q`.

### Task 4: Add YouTube caption fallback provider

**Files:**
- Create: `src/lyra_ocr/lyric/captions.py`
- Modify: `src/lyra_ocr/lyric/reference.py`
- Test: `tests/unit/test_lyric_captions.py`
- Fixture: `tests/fixtures/reference/caption.json3`, `tests/fixtures/reference/attention_caption.vtt`

**Implementation:**
- Add `fetch_youtube_caption_bundle(video_url_or_id, *, timeout=...) -> ReferenceBundle | None`:
  - lazy-import `yt_dlp`;
  - use `YoutubeDL({"skip_download": True, "quiet": True, "no_warnings": True})`;
  - read `automatic_captions` first, then `subtitles`;
  - choose language in this order: `info["language"]`, `en-orig`, `en` only if the original language is English;
  - never use auto-translated captions for a non-English original;
  - prefer `json3`, then `vtt`;
  - fetch the caption URL with a short timeout; return `None` on failure.
- Add `parse_caption_json3(data)` and `parse_caption_vtt(text)`:
  - ignore `[Music]`, `[Applause]`, empty events;
  - strip `<c>`, `<00:...>`, and VTT positioning tags;
  - merge rolling duplicate lines;
  - preserve start/end milliseconds;
  - split overly long caption events into lines only when a natural boundary is available;
  - mark source `"youtube_captions"` and confidence lower than LRCLIB.
- Do not call captions when LRCLIB returned usable lines.

**Tests:**
- synthetic JSON3 with overlapping/rolling events produces deduped ordered lines;
- VTT tags and `[Music]` are removed;
- non-English original with only translated `en` captions returns `None`;
- `yt-dlp` failure returns `None` and does not raise;
- no network is used in unit tests (mock `YoutubeDL` and `urlopen`).

- [x] Add caption fixtures and failing tests.
- [x] Implement caption fetch/parsing.
- [x] Run `pytest tests/unit/test_lyric_captions.py -q`.

### Task 5: Replace the alignment engine with source-aware matching

**Files:**
- Modify: `src/lyra_ocr/lyric/reference.py`
- Modify: `src/lyra_ocr/lyric/models.py`
- Test: `tests/unit/test_lyric_reference.py`

**Implementation:**
- Add `line_match_stats(ocr_text, reference_text) -> MatchStats` with:
  - character similarity;
  - token coverage of reference tokens;
  - length ratio;
  - optional CJK-aware tokenization.
- Replace/extend `align_reference()` with:
  - `align_reference(cues, reference, *, config=ReferenceConfig(), source="") -> AlignmentResult`;
  - monotonic dynamic programming with skip penalties and diagonal match preference;
  - match only when similarity and coverage pass the review threshold;
  - replace text only when similarity passes the source-aware replacement threshold and guards pass;
  - for 0.70–0.85 matches, keep OCR text, set `review_required=True`, and add `reference_uncertain`;
  - keep the old list-of-strings call path for compatibility;
  - preserve OCR timing for existing cues.
- Add `AlignmentResult` with matched count, replaced count, review count, and matched reference indices.

**Tests:**
- `I FEEL THE N1GHT` → `I feel the night`;
- a 0.80 similarity line with good coverage replaces when using LRCLIB, but not when using captions;
- a short fragment does not replace a longer full sentence;
- a high-confidence cue is not overwritten by a low-coverage caption;
- monotonic DP handles chorus repeats and one skipped reference line;
- `reference_uncertain` is set without changing text in the review band.

- [x] Add failing similarity/threshold tests.
- [x] Implement match stats and source-aware alignment.
- [x] Run `pytest tests/unit/test_lyric_reference.py -q`.

### Task 6: Recover missing reference lines

**Files:**
- Modify: `src/lyra_ocr/lyric/reference.py`
- Modify: `src/lyra_ocr/lyric/build.py`
- Test: `tests/unit/test_lyric_reference.py`, `tests/unit/test_lrc_build.py`

**Implementation:**
- Add `fill_reference_gaps(cues, reference, *, config, source) -> GapFillResult`.
- Only consider `ReferenceLine`s with `start_ms`/`end_ms`.
- Build recovered cues when:
  - the line lies between two matched cues (or after the last matched cue but before video duration);
  - no existing cue overlaps the candidate interval;
  - the gap is at least the reference line duration;
  - the line has at least two tokens or a CJK phrase;
  - source is `manual`, `lrclib_synced`, or `youtube_captions`.
- Clip to `[0, duration_ms]` and adjust start/end to remain monotonic.
- Create `LyricCue` with:
  - `text` = reference text;
  - `raw_text` = reference text;
  - `start_ms` / `end_ms` = reference timestamps;
  - `correction.method = "reference_gap_fill"`;
  - `qa_flags = ["caption_gap_fill"]` for captions;
  - confidence lower for captions than for LRCLIB.
- Deduplicate identical text/timestamp pairs.
- Run dedupe/grouping after gap fill so inserted lines cannot duplicate adjacent cues.

**Tests:**
- missing line between two matched cues is inserted;
- occupied gap is not filled;
- plain-text reference does not gap-fill;
- caption-sourced fill is flagged;
- end timestamp is clipped to video duration;
- inserted cues remain sorted and monotonic.

- [x] Add failing gap-fill tests.
- [x] Implement gap fill.
- [x] Run `pytest tests/unit/test_lyric_reference.py tests/unit/test_lrc_build.py -q`.

### Task 7: Integrate into build/select/classifier and cache

**Files:**
- Modify: `src/lyra_ocr/lyric/build.py`
- Modify: `src/lyra_ocr/lyric/select.py`
- Modify: `src/lyra_ocr/classify.py`
- Modify: `src/lyra_ocr/lyric/export.py`
- Test: `tests/unit/test_lrc_build.py`, `tests/unit/test_lrc_select.py`, `tests/unit/test_classify.py`

**Implementation:**
- `build_lrc_from_artifacts()`:
  - resolve manual/LRCLIB/caption provider in priority order;
  - use `ReferenceBundle` everywhere;
  - pass reference text to `select_cues()` for the existing quality-gate bypass, but only for trusted sources;
  - order: `select_cues` → `align_reference` → `fill_reference_gaps` → `dedupe_and_group` → write;
  - update result fields and cache signature.
- `select_cues()`:
  - keep `reference_lines` compatibility;
  - add optional `reference_bundle` / `reference_source`;
  - rescue a `non_lyric` track only under the conservative conditions in §3.5;
  - never rescue from captions;
  - add `reference_rescue` to `qa_flags`.
- `classify.py`:
  - add regression tests for `YOUVE BEEN RUNNIN...`-style real lyric lines;
  - reject/review short title/credit fragments such as `Presents`;
  - reject/review fragment-only `SOUND` / `ONLY` when they do not match a reference line;
  - do not break normal lower-third lyrics.
- `export.py`:
  - add `reference_source`, `n_reference_matches`, `n_reference_gap_fills`, `reference_warnings`;
  - keep schema changes additive.

**Tests:**
- provider priority: manual wins, LRCLIB wins over captions, captions only when LRCLIB empty;
- LRCLIB synced reference recovers the missing Attention line start;
- Railgun reference resolves after metadata parsing;
- a `non_lyric` real lyric line is rescued by trusted reference;
- captions cannot rescue `non_lyric`;
- cache invalidates when provider/config changes;
- `.cues.json` records provenance.

- [x] Add failing integration tests.
- [x] Implement build/select/classifier/export changes.
- [x] Run focused LRC tests.

### Task 8: Expose reference controls in CLI and Web UI

**Files:**
- Modify: `src/lyra_ocr/cli.py`
- Modify: `src/lyra_ocr/web/lrc_api.py`
- Modify: `src/lyra_ocr/web/static/index.html`
- Modify: `src/lyra_ocr/web/static/js/lrc_studio.js`
- Test: `tests/unit/test_web_server.py`

**Implementation:**
- CLI `lyra-ocr lrc`:
  - add `--use-reference` (auto: LRCLIB then captions);
  - add `--no-caption-fallback`;
  - keep default off for offline/backward compatibility;
  - print reference source, matched count, gap-fill count, and warnings.
- Web API:
  - keep `use_reference` payload compatibility;
  - add `no_caption_fallback`;
  - return `n_reference_gap_fills` and `reference_warnings`;
  - default remains opt-in.
- UI:
  - rename the checkbox label to “Auto reference lyrics (LRCLIB, YouTube captions fallback)”;
  - show reference source and gap-fill count in the result stats;
  - show a warning when captions are the source.
- README:
  - document provider priority, thresholds, opt-in behavior, and caption fallback limitations.

**Tests:**
- API passes the new flags through;
- response includes new fields;
- CLI forwards `use_reference` to the builder;
- no reference → behavior unchanged.

- [x] Add failing CLI/API tests.
- [x] Implement CLI/API/UI/documentation changes.
- [x] Run `pytest tests/unit/test_web_server.py -q`.

### Task 9: Validate on Attention and Railgun, then close the plan

**Files:**
- Modify: `README.md` if not already done
- Update: this plan’s checkboxes
- Output: regenerated `.lrc` and metrics JSON under `outputs/` or `/tmp/opencode/`

**Commands:**
```bash
.venv/bin/python -m pytest \
  tests/unit/test_lyric_reference.py \
  tests/unit/test_lyric_captions.py \
  tests/unit/test_lrc_build.py \
  tests/unit/test_lrc_select.py \
  tests/unit/test_classify.py \
  tests/unit/test_eval_lrc_text.py -q

.venv/bin/python -m pytest tests/unit tests/eval -q
```

**Manual validation:**
- Regenerate from existing Attention artifacts with `--use-reference --force`.
- Regenerate from existing Railgun artifacts with `--use-reference --force`.
- Inspect:
  - visible missing line starts are restored;
  - concatenated words are gone on lines that matched a reference;
  - no wrong hook/chorus replacement;
  - no duplicate gap-filled lines;
  - cue timing for detected lines is unchanged;
  - source/provenance is recorded.
- Compare before/after:
  - `n_reference_matches`;
  - `n_reference_gap_fills`;
  - `missing_reference_lines`;
  - `reference_line_recall`;
  - suspicious replacement count;
  - classifier false positives/negatives.

**Acceptance:**
- Attention uses LRCLIB synced (caption fallback only if LRCLIB is unavailable).
- Railgun resolves `sister's noise` / `fripSide` and uses LRCLIB synced Japanese lyrics.
- Multi-line behavior remains explicitly out of scope and is documented.
- Full non-slow test suite passes.
- Working tree changes are left uncommitted unless the user asks for a commit.

- [x] Run focused tests.
- [x] Run full non-slow suite.
- [x] Regenerate and inspect both sample LRCs.
- [x] Update README and checkboxes.

---

## 6. Validation matrix

| Case | Expected |
| --- | --- |
| Manual `.lrc` with timestamps | text repair + gap fill |
| Manual `.txt` without timestamps | text repair only |
| LRCLIB synced | text repair + gap fill |
| LRCLIB plain | text repair only |
| YouTube captions, no LRCLIB | text repair + flagged gap fill |
| No reference / network failure | unchanged LRC, no exception |
| Non-English original with translated-only captions | no caption fallback |
| High-confidence OCR vs low-coverage caption | OCR kept |
| `non_lyric` real lyric line + trusted LRCLIB | rescue with flag |
| `non_lyric` real lyric line + captions only | no rescue |

---

## 7. Risks and mitigations

- **Wrong replacement:** use source-aware thresholds, coverage/length guards, and review band; never replace exact matches; test high-confidence OCR protection.
- **ASR caption errors:** captions are fallback only, original-language only, stricter threshold, provenance flagged.
- **Chorus repeats / monotonic alignment:** DP with skip penalties and tests covering repeated lines.
- **Gap fill overlaps:** require an empty interval, clip to duration, re-dedupe after fill.
- **Copyrighted lyric fixtures:** use short synthetic fixtures only; do not commit full song lyrics.
- **Network flakiness:** cache, timeout, empty-bundle fallback, no exceptions.
- **Cache staleness:** include provider/query/parser/config version in cache key.
- **360p source limitation:** reference repair can fix text but cannot recover timing that was never detected; report remaining gaps as warnings instead of inventing timing.

---

## 8. Deferred

- Multi-line/simultaneous lyric assignment.
- Karaoke word-level timing.
- Audio/forced alignment.
- Whisper/faster-whisper gating (evaluate after reference path is measured).
- Full language-model contextual correction.
- Automatic translation or cross-language lyric matching.
- Automatic download of copyrighted reference lyrics into the repository.
