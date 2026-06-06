import pytest

from src.diarization.vad import (
    merge_speech_regions,
    normalize_speech_regions,
    prepare_asr_speech_regions,
    split_long_speech_regions,
)


def test_normalize_speech_regions_sorts_and_drops_invalid_regions():
    assert normalize_speech_regions(
        [
            {"start": 5, "end": 4},
            {"start": 3, "end": 4, "score": "0.5"},
            {"start": 1, "end": 2},
        ]
    ) == [
        {"start": 1.0, "end": 2.0, "score": None},
        {"start": 3.0, "end": 4.0, "score": 0.5},
    ]


def test_merge_speech_regions_merges_short_gaps():
    regions = merge_speech_regions(
        [
            {"start": 0.0, "end": 1.0, "score": 0.8},
            {"start": 1.3, "end": 2.0, "score": 0.6},
            {"start": 3.0, "end": 4.0, "score": 0.7},
        ],
        max_gap_s=0.5,
    )

    assert regions == [
        {"start": 0.0, "end": 2.0, "score": 0.7},
        {"start": 3.0, "end": 4.0, "score": 0.7},
    ]


def test_split_long_speech_regions_splits_contiguously():
    assert split_long_speech_regions([{"start": 0.0, "end": 5.0}], max_duration_s=2.0) == [
        {"start": 0.0, "end": 2.0, "score": None},
        {"start": 2.0, "end": 4.0, "score": None},
        {"start": 4.0, "end": 5.0, "score": None},
    ]


def test_split_long_speech_regions_rejects_non_positive_duration():
    with pytest.raises(ValueError):
        split_long_speech_regions([{"start": 0.0, "end": 5.0}], max_duration_s=0.0)


def test_prepare_asr_speech_regions_applies_merge_then_split():
    regions = prepare_asr_speech_regions(
        [
            {"start": 0.0, "end": 1.0},
            {"start": 1.2, "end": 5.0},
        ],
        merge_gap_s=0.5,
        max_duration_s=2.0,
    )

    assert regions == [
        {"start": 0.0, "end": 2.0, "score": None},
        {"start": 2.0, "end": 4.0, "score": None},
        {"start": 4.0, "end": 5.0, "score": None},
    ]
