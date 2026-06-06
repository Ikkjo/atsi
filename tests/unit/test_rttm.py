import pytest

from src.diarization.rttm import (
    filter_short_rttm_segments,
    merge_adjacent_rttm_segments,
    read_rttm,
    write_rttm,
)


def test_write_rttm_basic(tmp_path):
    segments = [
        {"start": 0.0, "end": 1.5, "speaker_id": "A"},
        {"start": 2.0, "end": 3.5, "speaker_id": "B"},
    ]
    path = tmp_path / "test.rttm"
    write_rttm(segments, path, file_id="meeting01")

    assert path.exists()
    text = path.read_text()
    lines = text.strip().split("\n")
    assert len(lines) == 2
    assert "SPEAKER meeting01 1 0.000 1.500 <NA> <NA> A <NA> <NA>" in lines[0]
    assert "SPEAKER meeting01 1 2.000 1.500 <NA> <NA> B <NA> <NA>" in lines[1]


def test_write_rttm_uses_output_stem_as_file_id(tmp_path):
    segments = [
        {"start": 0.0, "end": 1.0, "speaker_id": "A"},
    ]
    path = tmp_path / "my_meeting.rttm"
    write_rttm(segments, path)
    text = path.read_text()
    assert "SPEAKER my_meeting 1" in text


def test_write_rttm_empty(tmp_path):
    path = tmp_path / "empty.rttm"
    write_rttm([], path)
    assert path.exists()
    assert path.read_text() == ""


def test_write_rttm_preserves_duration(tmp_path):
    segments = [
        {"start": 0.0, "end": 2.0, "duration": 1.5, "speaker_id": "A"},
    ]
    path = tmp_path / "dur.rttm"
    write_rttm(segments, path)
    text = path.read_text()
    assert "1.500" in text


def test_read_rttm_roundtrip(tmp_path):
    segments = [
        {"start": 0.0, "end": 1.0, "speaker_id": "A"},
        {"start": 1.0, "end": 2.5, "speaker_id": "B"},
    ]
    path = tmp_path / "roundtrip.rttm"
    write_rttm(segments, path, file_id="meeting01")
    read = read_rttm(path)

    assert len(read) == 2
    assert read[0]["start"] == 0.0
    assert read[0]["end"] == 1.0
    assert read[0]["speaker_id"] == "A"
    assert read[0]["duration"] == 1.0
    assert read[1]["start"] == 1.0
    assert read[1]["end"] == 2.5
    assert read[1]["speaker_id"] == "B"
    assert read[1]["file_id"] == "meeting01"


def test_read_rttm_comments_and_blank_lines(tmp_path):
    path = tmp_path / "comments.rttm"
    path.write_text(
        "# This is a comment\n"
        "\n"
        "SPECKER meeting01 1 0.0 1.0 <NA> <NA> A <NA> <NA>\n"
        "SPEAKER meeting01 1 2.0 1.0 <NA> <NA> B <NA> <NA>\n"
        "SHORT\n"
    )
    # Note: "SPECKER" is a typo but still parses; "SHORT" is malformed and skipped.
    read = read_rttm(path)
    assert len(read) == 2
    assert read[0]["speaker_id"] == "A"
    assert read[1]["speaker_id"] == "B"


def test_read_rttm_file_not_found():
    with pytest.raises(FileNotFoundError):
        read_rttm("/nonexistent/path/file.rttm")


def test_merge_adjacent_same_speaker():
    segments = [
        {"start": 0.0, "end": 1.0, "speaker_id": "A"},
        {"start": 1.2, "end": 2.0, "speaker_id": "A"},
        {"start": 2.5, "end": 3.0, "speaker_id": "B"},
    ]
    merged = merge_adjacent_rttm_segments(segments, max_gap_s=0.5)
    assert len(merged) == 2
    assert merged[0]["start"] == 0.0
    assert merged[0]["end"] == 2.0
    assert merged[1]["speaker_id"] == "B"


def test_merge_adjacent_does_not_merge_large_gap():
    segments = [
        {"start": 0.0, "end": 1.0, "speaker_id": "A"},
        {"start": 2.0, "end": 3.0, "speaker_id": "A"},
    ]
    merged = merge_adjacent_rttm_segments(segments, max_gap_s=0.5)
    assert len(merged) == 2


def test_merge_adjacent_different_speakers():
    segments = [
        {"start": 0.0, "end": 1.0, "speaker_id": "A"},
        {"start": 1.1, "end": 2.0, "speaker_id": "B"},
    ]
    merged = merge_adjacent_rttm_segments(segments, max_gap_s=0.5)
    assert len(merged) == 2


def test_merge_adjacent_empty():
    assert merge_adjacent_rttm_segments([]) == []


def test_filter_short_segments():
    segments = [
        {"start": 0.0, "end": 0.15, "speaker_id": "A"},
        {"start": 1.0, "end": 1.5, "speaker_id": "B"},
    ]
    filtered = filter_short_rttm_segments(segments, min_duration_s=0.2)
    assert len(filtered) == 1
    assert filtered[0]["speaker_id"] == "B"


def test_filter_short_segments_empty():
    assert filter_short_rttm_segments([]) == []
