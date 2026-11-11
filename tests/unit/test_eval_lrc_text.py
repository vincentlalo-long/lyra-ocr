import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.eval_lrc_text import compute_text_metrics, load_cues


def test_compute_text_metrics_flags_quality_problems():
    cues = [
        {"text": "I walk in like thunder", "confidence": 0.95},
        {"text": "Long time no see!", "confidence": 0.90},
        {"text": "Myheartdoessomeihngstrange", "confidence": 0.92},
        {"text": "garbled text here", "confidence": 0.40},
    ]
    metrics = compute_text_metrics(cues, skipped=[{"reason": "review"}])

    assert metrics["n_cues"] == 4
    assert metrics["n_skipped"] == 1
    assert metrics["low_confidence_count"] == 1
    assert metrics["overlay_leak_count"] >= 1
    assert metrics["suspicious_text_count"] >= 1
    assert metrics["oov_rate"] > 0


def test_reference_metrics_are_zero_for_identical_text():
    cues = [{"text": "I walk in like thunder", "confidence": 0.95}]
    metrics = compute_text_metrics(cues, reference_text="I walk in like thunder")

    assert metrics["cer"] == 0.0
    assert metrics["wer"] == 0.0
    assert "evaluation comparison; runtime provenance is recorded in cues.json" in metrics["reference_note"]


def test_reference_evaluation_metrics_and_replacement_stats():
    cues = [
        {
            "text": "I walk in like thunder",
            "raw_text": "I WALK IN L1KE THUNDER",
            "confidence": 0.95,
            "start_ms": 1000,
            "correction": {"method": "reference", "changed": True},
        },
        {
            "text": "Completely different text that replaced something",
            "raw_text": "short",
            "confidence": 0.90,
            "start_ms": 5000,
            "correction": {"method": "reference_caption", "changed": True},
        },
    ]
    ref_text = (
        "[00:01.00]I walk in like thunder\n"
        "[00:05.00]Second line here\n"
        "[00:10.00]Third line not detected\n"
    )
    metrics = compute_text_metrics(cues, reference_text=ref_text)

    assert metrics["replaced_cue_count"] == 2
    assert metrics["suspicious_replacement_count"] >= 1  # length ratio outside guard
    assert metrics["missing_reference_lines"] >= 1
    assert 0.0 < metrics["reference_line_recall"] < 1.0


def test_load_cues_from_artifact_directory(tmp_path: Path):
    artifacts = tmp_path / "art"
    artifacts.mkdir()
    payload = {
        "cues": [{"text": "A readable line", "confidence": 0.9}],
        "skipped": [{"reason": "non_lyric"}],
    }
    (artifacts / "song.cues.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    cues, skipped = load_cues(artifacts)
    assert cues[0]["text"] == "A readable line"
    assert skipped[0]["reason"] == "non_lyric"


def test_eval_supports_cjk_tokens_and_repeated_timestamps():
    cues = [
        {"text": "君の声が聞こえる", "confidence": 0.95},
        {"text": "君の声が聞こえる", "confidence": 0.95},
    ]
    ref_text = "[00:01.00][00:05.00]君の声が聞こえる"

    metrics = compute_text_metrics(cues, reference_text=ref_text)

    assert metrics["reference_line_recall"] == 1.0
    assert metrics["tokens"] > 0
    assert metrics["cer"] == 0.0
