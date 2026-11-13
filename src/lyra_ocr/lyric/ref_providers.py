"""Online lookup, LRCLIB integration, and language selection/romanization for reference lyrics."""

from __future__ import annotations

import json
import logging
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from lyra_ocr.lyric.ref_models import (
    ReferenceBundle,
    ReferenceLine,
    ReferenceQuery,
    line_similarity,
    load_cached_reference_bundle,
    parse_reference_text,
    reference_cache_key,
    save_cached_reference_bundle,
)

logger = logging.getLogger(__name__)

LRCLIB_API = "https://lrclib.net/api"
DEFAULT_TIMEOUT = 6.0
_CJK_CHAR_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]+")


def fetch_reference_bundle(
    query: ReferenceQuery,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    cache_dir: Path | str | None = None,
    force: bool = False,
    ocr_lines: list[str] | None = None,
) -> ReferenceBundle:
    """Fetch reference lyrics bundle from LRCLIB. Never raises on failure."""
    if not (query.track or "").strip():
        return ReferenceBundle()

    key = ""
    if cache_dir is not None:
        key = reference_cache_key(query, provider="lrclib")
        if not force:
            cached = load_cached_reference_bundle(cache_dir, key)
            if cached is not None:
                return cached

    # Multi-query strategy: try specific track+artist first, then track-only, then full-text q=
    search_strategies: list[dict[str, str]] = []
    if query.artist.strip():
        search_strategies.append({"track_name": query.track.strip(), "artist_name": query.artist.strip()})
    search_strategies.append({"track_name": query.track.strip()})
    if query.artist.strip():
        search_strategies.append({"q": f"{query.track.strip()} {query.artist.strip()}"})
    search_strategies.append({"q": query.track.strip()})

    all_items: list[dict[str, Any]] = []
    seen_ids: set[Any] = set()

    for params in search_strategies:
        url = f"{LRCLIB_API}/search?{urllib.parse.urlencode(params)}"
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "LyraOCR/1.0 (local lyric alignment)"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw_bytes = response.read()
                data = json.loads(raw_bytes.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("LRCLIB lookup failed for %s: %s", params, exc)
            continue

        if isinstance(data, dict):
            data = [data]
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    iid = item.get("id")
                    if iid is not None:
                        if iid in seen_ids:
                            continue
                        seen_ids.add(iid)
                    all_items.append(item)
        if all_items:
            break

    if not all_items:
        return ReferenceBundle(warnings=["no_lrclib_results"])

    best_item = None
    best_score = float("-inf")
    best_overlap = 0.0

    for item in all_items:
        if not isinstance(item, dict):
            continue
        synced = str(item.get("syncedLyrics") or "").strip()
        plain = str(item.get("plainLyrics") or "").strip()
        if not synced and not plain:
            continue

        track_sim = line_similarity(query.track, str(item.get("trackName") or ""))
        artist_sim = line_similarity(query.artist, str(item.get("artistName") or ""))
        if track_sim < 0.25 and "q" not in search_strategies[0]:
            continue

        diff: float | None = None
        if query.duration_ms:
            try:
                diff = abs(float(item.get("duration") or 0.0) - query.duration_ms / 1000.0)
            except (TypeError, ValueError):
                diff = None
        if diff is not None and diff > 60.0 and not (track_sim >= 0.85 and artist_sim >= 0.70):
            continue

        score = 2.0 * track_sim + 1.5 * artist_sim
        if diff is not None:
            if diff <= 3.0:
                score += 2.0
            elif diff <= 10.0:
                score += 1.0
            elif diff > 30.0:
                score -= 2.0

        if synced:
            score += 0.5  # Prefer synced over plain

        # Lexical OCR overlap calculation
        cand_overlap = 0.0
        if ocr_lines:
            cand_raw = synced if synced else plain
            cand_parsed = parse_reference_text(cand_raw)
            cand_texts = [l.text for l in cand_parsed]
            if cand_texts:
                matches = sum(
                    1 for ol in ocr_lines
                    if any(line_similarity(ol, ct) >= 0.70 for ct in cand_texts)
                )
                cand_overlap = matches / max(1, len(ocr_lines))
                score += 4.0 * cand_overlap

        if score > best_score:
            best_score = score
            best_item = item
            best_overlap = cand_overlap

    if not best_item:
        return ReferenceBundle(warnings=["no_suitable_lrclib_match"])

    # Overlap gate: if we have >= 5 OCR lines, candidate lyrics must share at least 15% vocabulary
    if ocr_lines and len(ocr_lines) >= 5 and best_overlap < 0.15:
        return ReferenceBundle(warnings=["reference_low_ocr_overlap"])

    synced_lyrics = str(best_item.get("syncedLyrics") or "").strip()
    plain_lyrics = str(best_item.get("plainLyrics") or "").strip()

    bundle = ReferenceBundle()
    if synced_lyrics:
        lines = parse_reference_text(synced_lyrics, source="lrclib_synced")
        if lines:
            bundle = ReferenceBundle(
                lines=lines,
                source="lrclib_synced",
                provider="lrclib",
            )
    elif plain_lyrics:
        lines = parse_reference_text(plain_lyrics, source="lrclib_plain")
        if lines:
            bundle = ReferenceBundle(
                lines=lines,
                source="lrclib_plain",
                provider="lrclib",
            )

    if cache_dir is not None and key and bundle.source:
        save_cached_reference_bundle(cache_dir, key, bundle, query)

    return bundle


def romanize_bundle(bundle: ReferenceBundle) -> ReferenceBundle | None:
    """Convert Japanese reference lines (Kanji/Kana) to Romanized (Hepburn) lyrics using pykakasi."""
    if not bundle or not bundle.lines:
        return None

    try:
        import pykakasi
    except ImportError:
        logger.info("pykakasi not installed; skipping reference romanization")
        return None

    try:
        import jaconv
    except ImportError:
        jaconv = None

    kks = pykakasi.kakasi()
    romanized_lines: list[ReferenceLine] = []
    suffixes_to_attach = {"て", "で", "た", "だ", "ない", "ます", "ました", "たい", "ず", "ぬ"}

    for l in bundle.lines:
        if not l.text.strip():
            continue
        # Convert Katakana okurigana to Hiragana for proper compound verb romanization
        text_norm = jaconv.kata2hira(l.text) if jaconv else l.text
        res = kks.convert(text_norm)
        hepburn_tokens: list[str] = []
        cursor = 0
        for idx, item in enumerate(res):
            orig = str(item.get("orig", ""))
            hira = str(item.get("hira", ""))
            hep = str(item.get("hepburn", "")).strip()

            found_at = text_norm.find(orig, cursor) if orig else -1
            preceded_by_space = False
            if found_at > cursor:
                preceded_by_space = bool(text_norm[cursor:found_at].strip() == "" and found_at > cursor)
            if found_at >= 0:
                cursor = found_at + len(orig)

            if orig == "は" and idx > 0:
                hep = "wa"
            elif orig == "へ" and idx > 0:
                hep = "e"

            hep_clean = _CJK_CHAR_RE.sub("", hep).strip()
            if not hep_clean:
                continue

            if (
                orig in suffixes_to_attach
                and not preceded_by_space
                and hepburn_tokens
                and hira == orig
            ):
                hepburn_tokens[-1] = f"{hepburn_tokens[-1]}{hep_clean}"
            else:
                hepburn_tokens.append(hep_clean)

        romaji_text = _CJK_CHAR_RE.sub("", " ".join(hepburn_tokens)).strip()
        romaji_text = re.sub(r"\s+", " ", romaji_text).strip()
        if not romaji_text:
            continue
        romanized_lines.append(
            ReferenceLine(
                text=romaji_text,
                start_ms=l.start_ms,
                end_ms=l.end_ms,
                source="lrclib_romanized",
                language="ja-Latn",
            )
        )

    if not romanized_lines:
        return None

    return ReferenceBundle(
        lines=romanized_lines,
        source="lrclib_romanized",
        provider=bundle.provider,
        language="ja-Latn",
        warnings=list(bundle.warnings),
    )


def select_reference_lines(
    bundle: ReferenceBundle | None,
    target_lang: str,
    *,
    repair_romaji: bool = False,
) -> ReferenceBundle | None:
    """Select reference lines matching a target layer language, romanizing ja -> ja-Latn."""
    if not bundle or not bundle.lines:
        return None

    if target_lang == "ja-Latn" and not repair_romaji:
        return None

    from lyra_ocr.lang import detect_line_lang, vote_layer_lang

    bundle_lang = bundle.language or vote_layer_lang(
        [l.language or detect_line_lang(l.text) for l in bundle.lines]
    )

    if target_lang == "ja-Latn":
        if bundle_lang == "ja-Latn":
            return bundle
        ja_lines = [
            l for l in bundle.lines if (l.language or detect_line_lang(l.text)) == "ja"
        ]
        if bundle_lang == "ja" or ja_lines:
            src_bundle = ReferenceBundle(
                lines=ja_lines or bundle.lines,
                source=bundle.source,
                provider=bundle.provider,
                language="ja",
                warnings=list(bundle.warnings),
            )
            return romanize_bundle(src_bundle)
        return None

    # If the bundle as a whole matches the target layer
    if bundle_lang == target_lang or (bundle_lang in ("en", "und") and target_lang in ("en", "und")):
        return bundle

    # Otherwise, check if individual lines match the target language
    matching_lines: list[ReferenceLine] = []
    for line in bundle.lines:
        line_lang = line.language or detect_line_lang(line.text)
        if line_lang == target_lang:
            matching_lines.append(line)

    if not matching_lines:
        return None

    return ReferenceBundle(
        lines=matching_lines,
        source=bundle.source,
        provider=bundle.provider,
        language=target_lang,
        warnings=list(bundle.warnings),
    )


def fetch_reference_lines(
    title: str,
    artist: str = "",
    duration_ms: int = 0,
    *,
    timeout: float = DEFAULT_TIMEOUT,
) -> list[str]:
    """Look lyrics up on LRCLIB. Returns [] on any failure (never raises)."""
    q = ReferenceQuery(track=title, artist=artist, duration_ms=duration_ms)
    bundle = fetch_reference_bundle(q, timeout=timeout)
    return [line.text for line in bundle.lines]
