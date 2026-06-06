from src.evaluation.asr_wer import (
    compute_wer,
    evaluate_asr_wer,
    match_asr_annotation_pairs,
    normalize_text,
)


def test_normalize_text_lowercases_and_strips_punctuation():
    assert normalize_text("Hello, WORLD!!") == "hello world"


def test_compute_wer_returns_expected_edit_rate():
    result = compute_wer("hello world", "hello there")

    assert result["wer"] == 0.5
    assert result["substitutions"] == 1


def test_evaluate_asr_wer_uses_words_and_asr_text():
    result = evaluate_asr_wer(
        {"meeting_id": "M1", "text": "hello there", "model_id": "test", "word_timestamp_mode": "fallback"},
        {"meeting_id": "M1", "config": "ihm", "words": [{"word": "hello"}, {"word": "world"}]},
    )

    assert result["condition"] == "clean_headset"
    assert result["reference_text"] == "hello world"
    assert result["hypothesis_text"] == "hello there"
    assert result["wer"] == 0.5


def test_match_asr_annotation_pairs_only_returns_shared_meetings(tmp_path):
    asr_dir = tmp_path / "asr"
    annotation_dir = tmp_path / "annotations"
    asr_dir.mkdir()
    annotation_dir.mkdir()
    (asr_dir / "M1_whisper.json").write_text("{}")
    (asr_dir / "M2_whisper.json").write_text("{}")
    (annotation_dir / "M1_annotations.json").write_text("{}")

    assert match_asr_annotation_pairs(asr_dir, annotation_dir) == [
        (asr_dir / "M1_whisper.json", annotation_dir / "M1_annotations.json")
    ]
